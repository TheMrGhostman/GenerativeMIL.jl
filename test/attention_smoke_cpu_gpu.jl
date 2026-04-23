"""
Smoke + unit tests for attention primitives and MultiheadAttention.

Covers both CPU and GPU (when CUDA is functional):
- attention (3D, 4D)
- slot_attention (3D, 4D)
- mask construction and masking behavior
- MultiheadAttention forward pass (masked + unmasked)
- basic backward pass sanity for MultiheadAttention
"""

using Test
using Random
using Flux
using CUDA
using Zygote

const Mask = Union{AbstractArray{Bool}, Nothing}
MaskT{T} = Union{AbstractArray{Bool}, AbstractArray{T}, Nothing}

include(joinpath(@__DIR__, "..", "src", "building_blocks", "attention.jl"))

Random.seed!(42)

all_finite(x) = all(isfinite.(Array(x)))

@testset "attention smoke tests (CPU)" begin
    T = Float32
    d = 8
    vd = 8
    m = 5
    n = 7
    bs = 3
    h = 2

    Q3 = randn(T, d, m, bs)
    K3 = randn(T, d, n, bs)
    V3 = randn(T, vd, n, bs)

    @testset "mask helpers" begin
        Xmask = rand(Bool, 1, m, bs)
        Ymask = rand(Bool, 1, n, bs)

        att_mask = _build_attention_mask(T, Xmask, Ymask, m, n, bs)
        @test size(att_mask) == (n, m, 1, bs)

        att_mask_3d = _build_attention_mask(T, Xmask, Ymask, m, n, bs; multihead=false)
        @test size(att_mask_3d) == (n, m, bs)

        att_mask_y = _build_attention_mask(T, nothing, Ymask, m, n, bs)
        @test size(att_mask_y) == (n, 1, 1, bs)

        att_mask_x = _build_attention_mask(T, Xmask, nothing, m, n, bs)
        @test size(att_mask_x) == (1, m, 1, bs)

        @test _build_attention_mask(T, nothing, nothing, m, n, bs) === nothing
    end

    @testset "attention 3D" begin
        O = attention(Q3, K3, V3)
        @test size(O) == (vd, m, bs)
        @test all_finite(O)

        Ymask = rand(Bool, 1, n, bs)
        am = _build_attention_mask(T, nothing, Ymask, m, n, bs; multihead=false)
        Om = attention(Q3, K3, V3, am)
        @test size(Om) == (vd, m, bs)
        @test all_finite(Om)
    end

    @testset "attention 4D" begin
        dhead = d ÷ h
        vdhead = vd ÷ h

        Q4 = randn(T, dhead, m, h, bs)
        K4 = randn(T, dhead, n, h, bs)
        V4 = randn(T, vdhead, n, h, bs)

        O4 = attention(Q4, K4, V4)
        @test size(O4) == (vdhead, m, h, bs)
        @test all_finite(O4)
    end

    @testset "slot_attention 3D/4D" begin
        O3 = slot_attention(Q3, K3, V3)
        @test size(O3) == (vd, m, bs)
        @test all_finite(O3)

        dhead = d ÷ h
        vdhead = vd ÷ h
        Q4 = randn(T, dhead, m, h, bs)
        K4 = randn(T, dhead, n, h, bs)
        V4 = randn(T, vdhead, n, h, bs)

        O4 = slot_attention(Q4, K4, V4)
        @test size(O4) == (vdhead, m, h, bs)
        @test all_finite(O4)
    end

    @testset "MultiheadAttention forward" begin
        mh = MultiheadAttention(d, d, h, attention)

        out = mh(Q3, K3, V3)
        @test size(out) == (d, m, bs)
        @test all_finite(out)

        Xmask = rand(Bool, 1, m, bs)
        Ymask = rand(Bool, 1, n, bs)
        outm = mh(Q3, K3, Xmask, Ymask)
        @test size(outm) == (d, m, bs)
        @test all_finite(outm)
    end

    @testset "MultiheadAttention backward" begin
        mh = MultiheadAttention(d, d, h, attention)
        loss() = mean(abs2, mh(Q3, K3, V3))

        gs = Zygote.gradient(() -> loss(), Flux.params(mh))
        gWQ = gs[mh.WQ.weight]
        gWK = gs[mh.WK.weight]
        gWV = gs[mh.WV.weight]
        gWO = gs[mh.WO.weight]

        @test gWQ !== nothing
        @test gWK !== nothing
        @test gWV !== nothing
        @test gWO !== nothing
        @test all_finite(gWQ)
        @test all_finite(gWK)
        @test all_finite(gWV)
        @test all_finite(gWO)

        # Non-zero gradient sanity: ensures learning signal exists.
        ngWQ = norm(gWQ)
        ngWK = norm(gWK)
        ngWV = norm(gWV)
        ngWO = norm(gWO)
        @test (ngWQ + ngWK + ngWV + ngWO) > 0f0

        # One manual SGD step should change at least one parameter tensor.
        η = 1f-3
        WQ0 = copy(mh.WQ.weight)
        WK0 = copy(mh.WK.weight)
        WV0 = copy(mh.WV.weight)
        WO0 = copy(mh.WO.weight)

        mh.WQ.weight .-= η .* gWQ
        mh.WK.weight .-= η .* gWK
        mh.WV.weight .-= η .* gWV
        mh.WO.weight .-= η .* gWO

        dWQ = norm(mh.WQ.weight .- WQ0)
        dWK = norm(mh.WK.weight .- WK0)
        dWV = norm(mh.WV.weight .- WV0)
        dWO = norm(mh.WO.weight .- WO0)
        @test (dWQ + dWK + dWV + dWO) > 0f0
    end
end

@testset "attention smoke tests (GPU)" begin
    if !CUDA.functional()
        @test_skip "CUDA not functional in this environment"
    else
        T = Float32
        d = 8
        vd = 8
        m = 5
        n = 7
        bs = 2
        h = 2

        Q3 = CUDA.randn(T, d, m, bs)
        K3 = CUDA.randn(T, d, n, bs)
        V3 = CUDA.randn(T, vd, n, bs)

        @testset "attention 3D GPU" begin
            O = CUDA.@sync attention(Q3, K3, V3)
            @test size(O) == (vd, m, bs)
            @test O isa CuArray
            @test all_finite(O)

            Ymask = cu(rand(Bool, 1, n, bs))
            am = _build_attention_mask(T, nothing, Ymask, m, n, bs; multihead=false)
            Om = CUDA.@sync attention(Q3, K3, V3, am)
            @test size(Om) == (vd, m, bs)
            @test Om isa CuArray
            @test all_finite(Om)
        end

        @testset "slot_attention 3D GPU" begin
            O = CUDA.@sync slot_attention(Q3, K3, V3)
            @test size(O) == (vd, m, bs)
            @test O isa CuArray
            @test all_finite(O)
        end

        @testset "MultiheadAttention forward GPU" begin
            mh = cu(MultiheadAttention(d, d, h, attention))

            out = CUDA.@sync mh(Q3, K3, V3)
            @test size(out) == (d, m, bs)
            @test out isa CuArray
            @test all_finite(out)

            Xmask = cu(rand(Bool, 1, m, bs))
            Ymask = cu(rand(Bool, 1, n, bs))
            outm = CUDA.@sync mh(Q3, K3, Xmask, Ymask)
            @test size(outm) == (d, m, bs)
            @test outm isa CuArray
            @test all_finite(outm)
        end

        @testset "MultiheadAttention backward GPU" begin
            mh = cu(MultiheadAttention(d, d, h, attention))
            loss() = mean(abs2, mh(Q3, K3, V3))

            gs = Zygote.gradient(() -> loss(), Flux.params(mh))
            gWQ = gs[mh.WQ.weight]
            gWK = gs[mh.WK.weight]
            gWV = gs[mh.WV.weight]
            gWO = gs[mh.WO.weight]

            @test gWQ !== nothing
            @test gWK !== nothing
            @test gWV !== nothing
            @test gWO !== nothing
            @test all_finite(gWQ)
            @test all_finite(gWK)
            @test all_finite(gWV)
            @test all_finite(gWO)

            # Non-zero gradient sanity on GPU.
            ngWQ = norm(Array(gWQ))
            ngWK = norm(Array(gWK))
            ngWV = norm(Array(gWV))
            ngWO = norm(Array(gWO))
            @test (ngWQ + ngWK + ngWV + ngWO) > 0f0

            # One manual SGD step should change at least one parameter tensor.
            η = 1f-3
            WQ0 = copy(mh.WQ.weight)
            WK0 = copy(mh.WK.weight)
            WV0 = copy(mh.WV.weight)
            WO0 = copy(mh.WO.weight)

            mh.WQ.weight .-= η .* gWQ
            mh.WK.weight .-= η .* gWK
            mh.WV.weight .-= η .* gWV
            mh.WO.weight .-= η .* gWO

            dWQ = norm(Array(mh.WQ.weight .- WQ0))
            dWK = norm(Array(mh.WK.weight .- WK0))
            dWV = norm(Array(mh.WV.weight .- WV0))
            dWO = norm(Array(mh.WO.weight .- WO0))
            @test (dWQ + dWK + dWV + dWO) > 0f0
        end
    end
end
