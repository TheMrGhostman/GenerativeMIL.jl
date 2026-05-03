"""
Unit tests for SetVAE HierarchicalEncoder/HierarchicalDecoder on CPU and GPU.

Checks:
- forward pass shapes and finiteness
- decoder beta-vector path (per-layer KL scaling)
- beta-vector length validation
- gradients are finite and non-zero
- one update step changes weights (encoder and decoder)
- decoder logging buffers (klds/zs assignments) do not break Zygote
"""

using Test
using Random
using Flux
using CUDA
#using cuDNN
using Zygote
using LinearAlgebra
using AbstractTrees
using Distances, MLUtils, Statistics


const Mask = Union{AbstractArray{Bool}, Nothing}
MaskT{T} = Union{AbstractArray{Bool}, AbstractArray{T}, Nothing}

include(joinpath(@__DIR__, "..", "src", "building_blocks", "mlps.jl"))
include(joinpath(@__DIR__, "..", "src", "building_blocks", "attention.jl"))
include(joinpath(@__DIR__, "..", "src", "building_blocks", "prior.jl"))
include(joinpath(@__DIR__, "..", "src", "building_blocks", "transformer_blocks.jl"))

abstract type AbstractGenModel end
include(joinpath(@__DIR__, "..", "src", "models", "SetVAE.jl"))

Random.seed!(42)

all_finite(x::Number) = isfinite(x)
all_finite(x) = all(isfinite.(Array(x)))

function backward_update_check!(model, lossfn; η::Float32=1f-3)
    ps_obj = Flux.params(model)
    ps = collect(ps_obj)
    @test !isempty(ps)

    l0 = lossfn()
    @test all_finite(l0)

    gs = Zygote.gradient(() -> lossfn(), ps_obj)

    have_grad = false
    grad_norm_sum = 0.0
    backups = map(copy, ps)

    for p in ps
        g = gs[p]
        if g !== nothing
            have_grad = true
            @test all_finite(g)
            grad_norm_sum += norm(Array(g))
            p .-= η .* g
        end
    end

    @test have_grad
    @test grad_norm_sum > 0.0

    delta_sum = 0.0
    for (p, b) in zip(ps, backups)
        delta_sum += norm(Array(p .- b))
    end
    @test delta_sum > 0.0

    l1 = lossfn()
    @test all_finite(l1)
end

function run_hierarchical_tests(device::Function, devname::String)
    @testset "SetVAE HierarchicalEncoder/Decoder ($devname)" verbose=true begin
        T = Float32
        idim = 3
        hdim = 8
        heads = 2
        is_sizes = [4, 3]
        zdims = [4, 3]
        vb_depth = 2
        vb_hdim = 16

        n = 6
        bs = 2

        model = SetVAE(idim, hdim, heads, is_sizes, zdims, vb_depth, vb_hdim, relu, 5, idim)
        model = device(model)

        x = device(randn(T, idim, n, bs))
        x_mask = device(rand(Bool, 1, n, bs))

        @testset "Encoder forward shapes" begin
            x_enc, h_encs = model.encoder(x)
            @test size(x_enc) == (hdim, n, bs)
            @test length(h_encs) == length(model.encoder.layers)
            @test all_finite(x_enc)
            for h in h_encs
                @test size(h, 1) == hdim
                @test size(h, 3) == bs
                @test all_finite(h)
            end

            x_enc_m, h_encs_m = model.encoder(x, x_mask)
            @test size(x_enc_m) == (hdim, n, bs)
            @test length(h_encs_m) == length(model.encoder.layers)
            @test all_finite(x_enc_m)
            for h in h_encs_m
                @test size(h, 1) == hdim
                @test size(h, 3) == bs
                @test all_finite(h)
            end
        end

        @testset "Decoder forward shapes + beta path" begin
            _, h_encs = model.encoder(x, x_mask)
            z = model.prior(n, bs)

            x_out, klds, zs, kld_loss = model.decoder(z, h_encs, x_mask)
            @test size(x_out) == (idim, n, bs)
            @test length(klds) == length(model.decoder.layers)
            @test length(zs) == length(model.decoder.layers)
            @test all_finite(x_out)
            @test all_finite(kld_loss)
            for kld in klds
                @test all_finite(kld)
            end
            for zlayer in zs
                @test size(zlayer, 3) == bs
                @test all_finite(zlayer)
            end

            βv = device(fill(T(0.5), length(model.decoder.layers)))
            x_out_b, klds_b, zs_b, kld_loss_b = model.decoder(z, h_encs, x_mask, βv)
            @test size(x_out_b) == (idim, n, bs)
            @test length(klds_b) == length(model.decoder.layers)
            @test length(zs_b) == length(model.decoder.layers)
            @test all_finite(kld_loss_b)

            βbad = device(fill(T(1), length(model.decoder.layers) - 1))
            @test_throws ArgumentError model.decoder(z, h_encs, x_mask, βbad)
        end

        @testset "Encoder gradients + update" begin
            loss_enc() = begin
                x_enc, h_encs = model.encoder(x, x_mask)
                l = mean(abs2, x_enc)
                for h in h_encs
                    l += mean(abs2, h)
                end
                l
            end
            backward_update_check!(model.encoder, loss_enc)
        end

        @testset "Decoder gradients + update (logging assignments path)" begin
            _, h_encs_fixed = model.encoder(x, x_mask)
            z_fixed = model.prior(n, bs)

            loss_dec() = begin
                x_out, klds, zs, kld_loss = model.decoder(z_fixed, h_encs_fixed, x_mask)
                l = mean(abs2, x_out) + kld_loss
                # Touch logging outputs to ensure this path is exercised during AD.
                for i in eachindex(klds)
                    l += 1f-4 * klds[i]
                    l += 1f-6 * mean(abs2, zs[i])
                end
                l
            end
            backward_update_check!(model.decoder, loss_dec)
        end

        @testset "Decoder gradients + update (vector beta)" begin
            _, h_encs_fixed = model.encoder(x, x_mask)
            z_fixed = model.prior(n, bs)
            βv = device(fill(T(1), length(model.decoder.layers)))

            loss_dec_beta() = begin
                x_out, _, _, kld_loss = model.decoder(z_fixed, h_encs_fixed, x_mask, βv)
                mean(abs2, x_out) + kld_loss
            end
            backward_update_check!(model.decoder, loss_dec_beta)
        end
    end
end

@testset "SetVAE HierarchicalEncoder/Decoder CPU+GPU" begin
    run_hierarchical_tests(identity, "CPU");

    if CUDA.functional()
        run_hierarchical_tests(cu, "GPU");
    else
        @test_skip "CUDA not functional in this environment"
    end
end
