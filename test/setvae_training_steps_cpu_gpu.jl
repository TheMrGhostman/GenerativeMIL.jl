"""
Unit tests for SetVAE training/eval helpers:
- elbo_with_logging
- optim_step
- valid_step

Covers both unmasked and masked batch paths, scalar and vector beta,
and explicit Optimisers.setup(AdaMax(...), model) usage.
"""

using Test
using Random
using Flux
using CUDA
using Zygote
using Optimisers
using MLUtils
using Distances, Statistics, AbstractTrees, LinearAlgebra, NearestNeighbors

const Mask = Union{AbstractArray{Bool}, Nothing}
MaskT{T} = Union{AbstractArray{Bool}, AbstractArray{T}, Nothing}

include(joinpath(@__DIR__, "..", "src", "losses", "chamfer_distance.jl"))
include(joinpath(@__DIR__, "..", "src", "losses", "masked_chamfer_distance.jl"))
include(joinpath(@__DIR__, "..", "src", "building_blocks", "mlps.jl"))
include(joinpath(@__DIR__, "..", "src", "building_blocks", "attention.jl"))
include(joinpath(@__DIR__, "..", "src", "building_blocks", "prior.jl"))
include(joinpath(@__DIR__, "..", "src", "building_blocks", "transformer_blocks.jl"))

abstract type AbstractGenModel end
include(joinpath(@__DIR__, "..", "src", "models", "SetVAE.jl"))

Random.seed!(123)

all_finite(x::Number) = isfinite(x)
all_finite(x) = all(isfinite.(Array(x)))

function params_l2(model)
    return sum(sum(abs2, Array(p)) for p in Flux.trainables(model))
end

function make_model(device::Function=identity)
    model = SetVAE(
        3,      # input dim
        8,      # hidden dim
        2,      # heads
        [4, 3], # induced set sizes
        [4, 3], # latent dims
        2,      # zed depth
        16,     # zed hidden dim
        relu,
        5,
        3,
    )
    return device(model)
end

function make_batch(T::Type{<:AbstractFloat}=Float32; d::Int=3, n::Int=8, bs::Int=6)
    x = randn(T, d, n, bs)
    x_mask = rand(Bool, 1, n, bs)
    for b in 1:bs
        x_mask[1, 1, b] = true
    end
    return x, x_mask
end

function run_training_step_tests(device::Function, devname::String)
    @testset "SetVAE training helpers ($devname)" verbose=true begin
        model = make_model(device)
        x_cpu, x_mask_cpu = make_batch(Float32)
        x = device(x_cpu)
        x_mask = device(x_mask_cpu)
        β_vec = fill(0.5f0, length(model.decoder.layers)) |> device

        @testset "basic forward pass" begin
            xhat_u_s, kld_u_s, klds_u_s, zs_u_s = model(x; β=1f0)
            @test size(xhat_u_s) == size(x)
            @test all_finite(xhat_u_s)
            @test all_finite(kld_u_s)
            @test length(klds_u_s) == length(model.decoder.layers)
            @test length(zs_u_s) == length(model.decoder.layers)
            @test all(all_finite, klds_u_s)
            @test all(all_finite, zs_u_s)

            xhat_u_v, kld_u_v, klds_u_v, zs_u_v = model(x; β=β_vec)
            @test size(xhat_u_v) == size(x)
            @test all_finite(xhat_u_v)
            @test all_finite(kld_u_v)
            @test length(klds_u_v) == length(model.decoder.layers)
            @test length(zs_u_v) == length(model.decoder.layers)
            @test all(all_finite, klds_u_v)
            @test all(all_finite, zs_u_v)

            xhat_m_s, kld_m_s, klds_m_s, zs_m_s = model(x, x_mask; β=1f0)
            @test size(xhat_m_s) == size(x)
            @test all_finite(xhat_m_s)
            @test all_finite(kld_m_s)
            @test length(klds_m_s) == length(model.decoder.layers)
            @test length(zs_m_s) == length(model.decoder.layers)
            @test all(all_finite, klds_m_s)
            @test all(all_finite, zs_m_s)

            xhat_m_v, kld_m_v, klds_m_v, zs_m_v = model(x, x_mask; β=β_vec)
            @test size(xhat_m_v) == size(x)
            @test all_finite(xhat_m_v)
            @test all_finite(kld_m_v)
            @test length(klds_m_v) == length(model.decoder.layers)
            @test length(zs_m_v) == length(model.decoder.layers)
            @test all(all_finite, klds_m_v)
            @test all(all_finite, zs_m_v)
        end

        @testset "elbo_with_logging" begin
            loss_u, logs_u = elbo_with_logging(model, x; β=0.7f0, logpdf=chamfer_distance)
            @test all_finite(loss_u)
            @test all_finite(logs_u.ℒ)
            @test all_finite(logs_u.ℒ_rec)
            @test all_finite(logs_u.ℒₖₗ)
            @test length(logs_u.ℒₖₗₛ) == length(model.decoder.layers)
            @test isapprox(Float32(loss_u), Float32(logs_u.ℒ); rtol=1f-5, atol=1f-5)

            loss_m, logs_m = elbo_with_logging(model, x, x_mask; β=β_vec, logpdf=masked_chamfer_distance)
            @test all_finite(loss_m)
            @test all_finite(logs_m.ℒ)
            @test all_finite(logs_m.ℒ_rec)
            @test all_finite(logs_m.ℒₖₗ)
            @test length(logs_m.ℒₖₗₛ) == length(model.decoder.layers)
            @test isapprox(Float32(loss_m), Float32(logs_m.ℒ); rtol=1f-5, atol=1f-5)
        end

        @testset "optim_step with explicit Optimisers.setup" begin
            opt = Optimisers.setup(Optimisers.AdaMax(1f-1), model);

            p0 = params_l2(model)
            model, opt, logs_u = optim_step(model, x, opt, chamfer_distance, identity; β=1f0);
            p1 = params_l2(model)
            @test all_finite(logs_u.ℒ)
            @test all_finite(logs_u.ℒ_rec)
            @test all_finite(logs_u.ℒₖₗ)
            @test abs(p1 - p0) > 0f0

            model, opt, logs_m = optim_step(model, (x, x_mask), opt, masked_chamfer_distance, identity; β=β_vec);
            p2 = params_l2(model)
            @test all_finite(logs_m.ℒ)
            @test all_finite(logs_m.ℒ_rec)
            @test all_finite(logs_m.ℒₖₗ)
            @test abs(p2 - p1) > 0f0
        end

        @testset "valid_step" begin
            dl_u = DataLoader(x_cpu, batchsize=2, shuffle=false)
            vlogs_u, vloss_u = valid_step(model, dl_u, chamfer_distance; β=1f0, device=device)
            @test all_finite(vloss_u)
            @test all_finite(vlogs_u.ℒᵥ)
            @test all_finite(vlogs_u.ℒᵥ_rec)
            @test all_finite(vlogs_u.ℒᵥₖₗ)
            @test length(vlogs_u.ℒᵥₖₗₛ) == length(model.decoder.layers)
            @test isapprox(Float32(vloss_u), Float32(vlogs_u.ℒᵥ); rtol=1f-5, atol=1f-5)

            dl_m = DataLoader((x_cpu, x_mask_cpu), batchsize=2, shuffle=false)
            vlogs_m, vloss_m = valid_step(model, dl_m, masked_chamfer_distance; β=β_vec, device=device)
            @test all_finite(vloss_m)
            @test all_finite(vlogs_m.ℒᵥ)
            @test all_finite(vlogs_m.ℒᵥ_rec)
            @test all_finite(vlogs_m.ℒᵥₖₗ)
            @test length(vlogs_m.ℒᵥₖₗₛ) == length(model.decoder.layers)
            @test isapprox(Float32(vloss_m), Float32(vlogs_m.ℒᵥ); rtol=1f-5, atol=1f-5)
        end
    end
end

@testset "SetVAE elbo/optim/valid CPU+GPU" begin
    run_training_step_tests(identity, "CPU");

    if CUDA.functional()
        run_training_step_tests(cu, "GPU");
    else
        @test_skip "CUDA not functional in this environment"
    end
end
