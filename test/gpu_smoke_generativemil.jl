"""
Fast GPU smoke tests for core operations used in GenerativeMIL.

Covers:
- CUDA availability and basic allocation
- softmax on GPU
- repeat on GPU
- gumbel_softmax soft/hard behavior on GPU
- MixtureOfGaussians forward on GPU
- MixtureOfGaussians backward/update sanity on GPU

Intended to finish quickly and provide PASS/FAIL output.
"""

using Random
using Statistics
using LinearAlgebra
using Flux
using CUDA
using Zygote
using MLUtils
using Distances
using BenchmarkTools

abstract type AbstractPriorDistribution end

struct MixtureOfGaussians{T <: AbstractFloat, A1 <: AbstractVector{T}, A3 <: AbstractArray{T, 3}} <: AbstractPriorDistribution
    α::A1
    μ::A3
    Σ::A3
    trainable::Bool

    function MixtureOfGaussians(
        α::A1,
        μ::A3,
        Σ::A3,
        trainable::Bool,
    ) where {T <: AbstractFloat, A1 <: AbstractVector{T}, A3 <: AbstractArray{T, 3}}
        Ds, K, D3 = size(μ)
        D3 == 1 || throw(ArgumentError("MixtureOfGaussians: expected size(μ, 3) == 1, got $(D3)."))
        length(α) == K || throw(ArgumentError("MixtureOfGaussians: expected length(α) == K ($K), got $(length(α))."))
        size(Σ) == (Ds, K, 1) || throw(ArgumentError("MixtureOfGaussians: expected size(Σ) == ($Ds, $K, 1), got $(size(Σ))."))
        return new{T, A1, A3}(α, μ, Σ, trainable)
    end
end

Flux.@layer MixtureOfGaussians
Flux.trainable(MoG::MixtureOfGaussians) = MoG.trainable ? (α = MoG.α, μ = MoG.μ, Σ = MoG.Σ) : ()

function sample_sphere(dim::Int, n_points::Int)
    norm_(x, d=1) = sqrt.(sum(abs2, x, dims=d))
    x = randn(Float32, dim, n_points)
    return x ./ norm_(x, 1)
end

function MixtureOfGaussians(dim::Int, n_mixtures::Int, trainable::Bool=true; downscale=10f0, ϵ=1f-3)
    μs_2d = sample_sphere(dim, n_mixtures)
    pp = Distances.pairwise(Distances.euclidean, μs_2d)
    var_ = pp .+ LinearAlgebra.Diagonal(LinearAlgebra.diag(pp) .+ Inf) |> minimum
    μs = MLUtils.unsqueeze(μs_2d, 3)
    Σs = ones(Float32, dim, n_mixtures, 1) .* Float32(var_ / downscale + ϵ)
    Σs = log.(exp.(Σs) .- 1f0)
    αs = ones(Float32, n_mixtures)
    return MixtureOfGaussians(αs, μs, Σs, trainable)
end

function gumbel_softmax(logits::AbstractArray{T}; τ::T=1f0, hard::Bool=false, ϵ=T(1.0e-10)) where {T <: AbstractFloat}
    g = -log.(-log.(MLUtils.rand_like(logits) .+ ϵ) .+ ϵ)
    y = Flux.softmax((logits .+ g) ./ τ)

    if !hard
        return y
    else
        y_hard = zero(y)
        Zygote.ignore() do
            _, ind = findmax(y, dims=1)
            y_hard[ind] .= one(T)
        end
        return y + Zygote.ignore(y_hard - y)
    end
end

function (MoG::MixtureOfGaussians)(sample_size::Int, batch_size::Int)
    Ds, K, _ = size(MoG.μ)
    α_logits = repeat(reshape(MoG.α, (K, 1, 1, 1)), 1, sample_size, 1, batch_size)
    αₒₕ = gumbel_softmax(α_logits, hard=true)
    αₒₕ = permutedims(αₒₕ, (3, 2, 1, 4))

    μ = reshape(MoG.μ, (Ds, 1, K, 1))
    Σ = reshape(MoG.Σ, (Ds, 1, K, 1))

    μ_mixed = reshape(sum(μ .* αₒₕ, dims=3), (:, sample_size, batch_size))
    Σ_mixed = reshape(sum(Σ .* αₒₕ, dims=3), (:, sample_size, batch_size))

    ϵ = MLUtils.randn_like(μ_mixed)
    return μ_mixed .+ Flux.softplus.(Σ_mixed) .* ϵ
end

const SEP = "="^80

mutable struct SmokeSummary
    passed::Int
    failed::Int
end

function ok!(summary::SmokeSummary, name::String)
    summary.passed += 1
    println("PASS: " * name)
end

function fail!(summary::SmokeSummary, name::String, err)
    summary.failed += 1
    println("FAIL: " * name)
    println("  error: " * sprint(showerror, err))
end

function assert_true(cond::Bool, msg::String)
    cond || error(msg)
end

function all_finite(x)
    return all(isfinite.(Array(x)))
end

function run_smoke()
    println("\n" * SEP)
    println("GenerativeMIL Fast GPU Smoke Tests")
    println(SEP)

    summary = SmokeSummary(0, 0)

    try
        assert_true(CUDA.functional(), "CUDA.functional() is false")
        x = CUDA.rand(Float32, 32, 32)
        y = CUDA.rand(Float32, 32, 32)
        s = sum(x * y)
        assert_true(x * y isa CuArray{Float32}, "basic matmul did not stay on GPU")
        ok!(summary, "cuda_basic")
    catch err
        fail!(summary, "cuda_basic", err)
    end

    try
        logits = CUDA.randn(Float32, 7, 5)
        probs = Flux.softmax(logits; dims=1)
        sums = sum(probs; dims=1)
        max_dev = maximum(abs.(Array(sums) .- 1f0))
        assert_true(max_dev < 1f-4, "softmax sums are not close to 1 (max dev = $max_dev)")
        assert_true(probs isa CuArray, "softmax output moved to CPU")
        ok!(summary, "softmax_gpu")
    catch err
        fail!(summary, "softmax_gpu", err)
    end

    try
        t = CUDA.randn(Float32, 1, 1, 6, 1)
        r = repeat(t, 1, 8, 1, 3)
        assert_true(size(r) == (1, 8, 6, 3), "repeat output size mismatch")
        assert_true(r isa CuArray, "repeat output moved to CPU")
        ok!(summary, "repeat_gpu")
    catch err
        fail!(summary, "repeat_gpu", err)
    end

    try
        Random.seed!(42)
        logits = CUDA.randn(Float32, 6, 4)
        y_soft = gumbel_softmax(logits; τ=1f0, hard=false)
        sums_soft = sum(y_soft; dims=1)
        max_dev_soft = maximum(abs.(Array(sums_soft) .- 1f0))
        assert_true(max_dev_soft < 1f-3, "gumbel_softmax soft sums mismatch (max dev = $max_dev_soft)")

        Random.seed!(42)
        y_hard = gumbel_softmax(logits; τ=1f0, hard=true)
        sums_hard = Array(sum(y_hard; dims=1))
        vals = Array(y_hard)
        is_binary = all((abs.(vals .- 0f0) .< 1f-5) .| (abs.(vals .- 1f0) .< 1f-5))
        assert_true(maximum(abs.(sums_hard .- 1f0)) < 1f-4, "gumbel_softmax hard sums mismatch")
        assert_true(is_binary, "gumbel_softmax hard output is not one-hot")
        assert_true(y_soft isa CuArray && y_hard isa CuArray, "gumbel_softmax output moved to CPU")
        ok!(summary, "gumbel_softmax_gpu")
    catch err
        fail!(summary, "gumbel_softmax_gpu", err)
    end

    try
        mog_cpu = MixtureOfGaussians(8, 4, true)
        mog = MixtureOfGaussians(cu(mog_cpu.α), cu(mog_cpu.μ), cu(mog_cpu.Σ), true)
        z = mog(10, 3)
        assert_true(size(z) == (8, 10, 3), "MoG forward output size mismatch")
        assert_true(z isa CuArray, "MoG forward output moved to CPU")
        assert_true(all_finite(z), "MoG forward contains NaN/Inf")
        ok!(summary, "mog_forward_gpu")
    catch err
        fail!(summary, "mog_forward_gpu", err)
    end

    try
        Random.seed!(7)
        mog_cpu = MixtureOfGaussians(6, 3, true)
        mog = MixtureOfGaussians(cu(mog_cpu.α), cu(mog_cpu.μ), cu(mog_cpu.Σ), true)
        target = CUDA.randn(Float32, 6, 6, 2)

        function lossfn(m)
            z = m(6, 2)
            return mean((z .- target) .^ 2)
        end

        loss_before = lossfn(mog)
        grads = Zygote.gradient(mog) do model
            lossfn(model)
        end
        gs = grads[1]

        ga = gs.α
        gm = gs.μ
        gsig = gs.Σ

        assert_true(ga !== nothing && gm !== nothing && gsig !== nothing, "some gradients are nothing")
        assert_true(all_finite(ga) && all_finite(gm) && all_finite(gsig), "gradients contain NaN/Inf")

        lr = 1f-3
        a0 = Array(copy(mog.α))
        m0 = Array(copy(mog.μ))
        s0 = Array(copy(mog.Σ))

        mog.α .-= lr .* ga
        mog.μ .-= lr .* gm
        mog.Σ .-= lr .* gsig

        da = norm(Array(mog.α) .- a0)
        dm = norm(Array(mog.μ) .- m0)
        ds = norm(Array(mog.Σ) .- s0)

        assert_true((da > 0f0) || (dm > 0f0) || (ds > 0f0), "no parameter changed after update")

        loss_after = lossfn(mog)
        assert_true(isfinite(loss_before) && isfinite(loss_after), "loss is not finite")
        ok!(summary, "mog_backward_gpu")
    catch err
        fail!(summary, "mog_backward_gpu", err)
    end

    println("\n" * SEP)
    println("Summary")
    println(SEP)
    println("passed: $(summary.passed)")
    println("failed: $(summary.failed)")

    if summary.failed == 0
        println("GPU_SMOKE: PASS")
        return 0
    else
        println("GPU_SMOKE: FAIL")
        return 1
    end
end

exit(run_smoke())
