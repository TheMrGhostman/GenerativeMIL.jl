"""
Smoke + unit tests for transformer blocks.

Covers CPU and GPU (if CUDA is functional):
- MultiheadAttentionBlock
- InducedSetAttentionBlock
- InducedSetAttentionHalfBlock
- VariationalBottleneck
- AttentiveBottleneckLayer
- AttentiveHalfBlock

For each major block we check:
- forward shape/finite sanity
- backward finite gradients
- non-zero gradient signal
- one-step parameter update changes weights
"""

using Test
using Random
using LinearAlgebra
using Flux
using CUDA
using Zygote
using MLUtils
using Distances
using Statistics

const Mask = Union{AbstractArray{Bool}, Nothing}
MaskT{T} = Union{AbstractArray{Bool}, AbstractArray{T}, Nothing}

include(joinpath(@__DIR__, "..", "src", "building_blocks", "mlps.jl"))
include(joinpath(@__DIR__, "..", "src", "building_blocks", "attention.jl"))
include(joinpath(@__DIR__, "..", "src", "building_blocks", "prior.jl"))
include(joinpath(@__DIR__, "..", "src", "building_blocks", "transformer_blocks.jl"))

Random.seed!(42)

all_finite(x) = all(isfinite.(Array(x)))

function to_scalar(x)
    return x isa Number ? Float64(x) : Float64(Array(x)[])
end

function backward_update_check!(model, lossfn; η::Float32=1f-3)
    ps_obj = Flux.params(model)
    ps = collect(ps_obj)
    @test !isempty(ps)

    loss_before = to_scalar(lossfn())
    @test isfinite(loss_before)

    gs = Zygote.gradient(() -> lossfn(), ps_obj)

    grad_norm_sum = 0.0
    finite_ok = true
    have_grad = false

    backups = map(copy, ps)

    for p in ps
        g = gs[p]
        if g !== nothing
            have_grad = true
            finite_ok &= all_finite(g)
            grad_norm_sum += norm(Array(g))
            p .-= η .* g
        end
    end

    @test have_grad
    @test finite_ok
    @test grad_norm_sum > 0.0

    delta_sum = 0.0
    for (p, b) in zip(ps, backups)
        delta_sum += norm(Array(p .- b))
    end
    @test delta_sum > 0.0

    loss_after = to_scalar(lossfn())
    @test isfinite(loss_after)
end

function run_block_tests(device::Function, devname::String)
    @testset "transformer blocks ($devname)" verbose=true begin
        T = Float32
        d = 8
        n = 7
        m = 5
        bs = 2
        heads = 2
        zdim = 4
        hidden = 16
        depth = 2

        x = device(randn(T, d, n, bs))
        q = device(randn(T, d, m, bs))
        v = device(randn(T, d, n, bs))
        x_mask = device(rand(Bool, 1, n, bs))
        q_mask = device(rand(Bool, 1, m, bs))
        h_enc_slots = device(randn(T, d, m, bs))

        @testset "MultiheadAttentionBlock" begin
            mab = device(MultiheadAttentionBlock(d, heads; attention_fn=attention))

            y_self = mab(x)
            @test size(y_self) == size(x)
            @test all_finite(y_self)

            y_cross = mab(q, v)
            @test size(y_cross) == size(q)
            @test all_finite(y_cross)

            y_mask = mab(q, v, q_mask, x_mask)
            @test size(y_mask) == size(q)
            @test all_finite(y_mask)

            lossfn() = mean(abs2, mab(q, v, q_mask, x_mask))
            backward_update_check!(mab, lossfn)
        end

        @testset "InducedSetAttentionBlock" begin
            isab = device(InducedSetAttentionBlock(m, d, heads))

            x_out, h = isab(x)
            @test size(x_out) == size(x)
            @test size(h) == (d, m, bs)
            @test all_finite(x_out)
            @test all_finite(h)

            x_out_m, h_m = isab(x, x_mask)
            @test size(x_out_m) == size(x)
            @test size(h_m) == (d, m, bs)
            @test all_finite(x_out_m)
            @test all_finite(h_m)

            lossfn() = begin
                xo, hh = isab(x, x_mask)
                mean(abs2, xo) + mean(abs2, hh)
            end
            backward_update_check!(isab, lossfn)
        end

        @testset "InducedSetAttentionHalfBlock" begin
            isab_h = device(InducedSetAttentionHalfBlock(m, d, heads))

            x_passthrough, h = isab_h(x)
            @test size(x_passthrough) == size(x)
            @test size(h) == (d, m, bs)
            @test all_finite(h)

            x_passthrough_m, h_m = isab_h(x, x_mask)
            @test size(x_passthrough_m) == size(x)
            @test size(h_m) == (d, m, bs)
            @test all_finite(h_m)

            lossfn() = begin
                _, hh = isab_h(x, x_mask)
                mean(abs2, hh)
            end
            backward_update_check!(isab_h, lossfn)
        end

        @testset "VariationalBottleneck" begin
            vb = device(VariationalBottleneck(d, zdim, d, hidden, depth, relu))

            z, h_hat, kld_none = vb(h_enc_slots)
            @test size(z) == (zdim, m, bs)
            @test size(h_hat) == (d, m, bs)
            @test kld_none === nothing
            @test all_finite(z)
            @test all_finite(h_hat)

            z2, h_hat2, kld = vb(h_enc_slots, h_enc_slots)
            @test size(z2) == (zdim, m, bs)
            @test size(h_hat2) == (d, m, bs)
            @test size(kld) == (zdim, m, bs)
            @test all_finite(z2)
            @test all_finite(h_hat2)
            @test all_finite(kld)

            lossfn() = begin
                _, hh, kl = vb(h_enc_slots, h_enc_slots)
                mean(abs2, hh) + mean(kl)
            end
            backward_update_check!(vb, lossfn)
        end

        @testset "AttentiveBottleneckLayer" begin
            abl = device(AttentiveBottleneckLayer(m, d, heads, zdim, hidden, depth, relu))

            xo_gen, kl_gen, h_hat_gen, z_gen = abl(x)
            @test size(xo_gen) == size(x)
            @test kl_gen === nothing
            @test size(h_hat_gen) == (d, m, bs)
            @test size(z_gen) == (zdim, m, bs)
            @test all_finite(xo_gen)
            @test all_finite(h_hat_gen)
            @test all_finite(z_gen)

            xo_inf, kl_inf, h_hat_inf, z_inf = abl(x, h_enc_slots)
            @test size(xo_inf) == size(x)
            @test kl_inf isa Number
            @test isfinite(Float64(kl_inf))
            @test size(h_hat_inf) == (d, m, bs)
            @test size(z_inf) == (zdim, m, bs)
            @test all_finite(xo_inf)
            @test all_finite(h_hat_inf)
            @test all_finite(z_inf)

            xo_m, kl_m, h_hat_m, z_m = abl(x, h_enc_slots, x_mask)
            @test size(xo_m) == size(x)
            @test kl_m isa Number
            @test isfinite(Float64(kl_m))
            @test size(h_hat_m) == (d, m, bs)
            @test size(z_m) == (zdim, m, bs)
            @test all_finite(xo_m)
            @test all_finite(h_hat_m)
            @test all_finite(z_m)

            lossfn() = begin
                xo, kl, _, _ = abl(x, h_enc_slots, x_mask)
                mean(abs2, xo) + Float32(kl)
            end
            backward_update_check!(abl, lossfn)
        end

        @testset "AttentiveHalfBlock" begin
            ahb = device(AttentiveHalfBlock(m, d, heads, zdim, hidden, depth, relu))

            xo, kl, h_hat, z = ahb(x, h_enc_slots, x_mask)
            @test size(xo) == size(x)
            @test kl isa Number
            @test isfinite(Float64(kl))
            @test size(h_hat) == (d, m, bs)
            @test size(z) == (zdim, m, bs)
            @test all_finite(xo)
            @test all_finite(h_hat)
            @test all_finite(z)

            lossfn() = begin
                xh, klh, hh, zh = ahb(x, h_enc_slots, x_mask)
                mean(abs2, xh) + Float32(klh) + mean(abs2, hh) + mean(abs2, zh)
            end
            backward_update_check!(ahb, lossfn)
        end
    end
end

@testset "transformer_blocks smoke CPU/GPU" begin
    run_block_tests(identity, "CPU")

    if CUDA.functional()
        run_block_tests(cu, "GPU")
    else
        @test_skip "CUDA not functional in this environment"
    end
end
