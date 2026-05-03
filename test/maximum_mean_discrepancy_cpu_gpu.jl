"""
Unit + smoke tests for maximum_mean_discrepancy (MMD).

Covers:
- CPU correctness properties (symmetry, finiteness, batched consistency)
- sigma vector (multi-scale RBF) behavior
- optional distance_kernel path
- GPU correctness against CPU (when CUDA is functional)
- lightweight speed comparison vs chamfer_distance
"""

using Test
using Random
using Flux
using CUDA
using BenchmarkTools
using MLUtils
using Zygote
using Statistics
using NearestNeighbors
using Distances
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "losses", "chamfer_distance.jl"))
include(joinpath(@__DIR__, "..", "src", "losses", "maximum_mean_discrepancy.jl"))

Random.seed!(42)

all_finite(x::Number) = isfinite(x)
all_finite(x) = all(isfinite.(Array(x)))

function make_batch(T::Type, d::Int, n::Int, bs::Int)
    x = randn(T, d, n, bs)
    y = randn(T, d, n, bs)
    return x, y
end

@testset "maximum_mean_discrepancy" verbose=true begin
    T = Float32
    d = 3
    n = 32
    bs = 5

    x, y = make_batch(T, d, n, bs)

    @testset "CPU correctness" begin
        # 2D symmetry and finiteness.
        x2 = @view x[:, :, 1]
        y2 = @view y[:, :, 1]
        mxy = maximum_mean_discrepancy(x2, y2; sigma=1f0)
        myx = maximum_mean_discrepancy(y2, x2; sigma=1f0)
        @test all_finite(mxy)
        @test all_finite(myx)
        @test isapprox(mxy, myx; rtol=1f-5, atol=1f-6)

        # 3D should match average over per-batch 2D calls.
        m3 = maximum_mean_discrepancy(x, y; sigma=1f0)
        m_ref = zero(T)
        for b in 1:bs
            m_ref += maximum_mean_discrepancy(@view(x[:, :, b]), @view(y[:, :, b]); sigma=1f0)
        end
        m_ref /= T(bs)
        @test all_finite(m3)
        @test isapprox(m3, m_ref; rtol=2f-5, atol=2f-6)
    end

    @testset "multi-sigma RBF" begin
        sigmas = [0.5f0, 1.0f0, 2.0f0]

        m2_multi = maximum_mean_discrepancy(@view(x[:, :, 1]), @view(y[:, :, 1]); sigma=sigmas)
        m2_avg = mean([maximum_mean_discrepancy(@view(x[:, :, 1]), @view(y[:, :, 1]); sigma=s) for s in sigmas])
        @test isapprox(m2_multi, m2_avg; rtol=1f-5, atol=1f-6)

        m3_multi = maximum_mean_discrepancy(x, y; sigma=sigmas)
        m3_avg = mean([maximum_mean_discrepancy(x, y; sigma=s) for s in sigmas])
        @test isapprox(m3_multi, m3_avg; rtol=2f-5, atol=2f-6)
    end

    @testset "distance_kernel path" begin
        # IMQ-like kernel as a function of squared distances.
        dkernel(d) = 1f0 ./ (1f0 .+ d)

        m2 = maximum_mean_discrepancy(@view(x[:, :, 1]), @view(y[:, :, 1]); distance_kernel=dkernel)
        m3 = maximum_mean_discrepancy(x, y; distance_kernel=dkernel)

        @test all_finite(m2)
        @test all_finite(m3)

        # Symmetry should hold for symmetric distance kernels.
        m2_sym = maximum_mean_discrepancy(@view(y[:, :, 1]), @view(x[:, :, 1]); distance_kernel=dkernel)
        @test isapprox(m2, m2_sym; rtol=1f-5, atol=1f-6)
    end

    @testset "GPU correctness" begin
        if !CUDA.functional()
            @test_skip "CUDA not functional in this environment"
        else
            xg = cu(x)
            yg = cu(y)

            m_cpu = maximum_mean_discrepancy(x, y; sigma=1f0)
            m_gpu = CUDA.@sync maximum_mean_discrepancy(xg, yg; sigma=1f0)
            @test all_finite(m_gpu)
            @test isapprox(Float32(m_gpu), m_cpu; rtol=2f-4, atol=2f-5)

            sigmas = [0.5f0, 1.0f0, 2.0f0]
            m_cpu_ms = maximum_mean_discrepancy(x, y; sigma=sigmas)
            m_gpu_ms = CUDA.@sync maximum_mean_discrepancy(xg, yg; sigma=sigmas)
            @test all_finite(m_gpu_ms)
            @test isapprox(Float32(m_gpu_ms), m_cpu_ms; rtol=3f-4, atol=3f-5)

            dkernel(d) = 1f0 ./ (1f0 .+ d)
            m_cpu_dk = maximum_mean_discrepancy(x, y; distance_kernel=dkernel)
            m_gpu_dk = CUDA.@sync maximum_mean_discrepancy(xg, yg; distance_kernel=dkernel)
            @test all_finite(m_gpu_dk)
            @test isapprox(Float32(m_gpu_dk), m_cpu_dk; rtol=3f-4, atol=3f-5)
        end
    end

    @testset "gradient smoke (CPU/GPU)" begin
        function gaussian_fit_gradient_case(use_gpu::Bool)
            Tloc = Float32
            dloc = 2
            nloc = 96
            bsloc = 6

            # Unknown data-generating Gaussian.
            μ_true = reshape(Tloc[1.4f0, -0.9f0], dloc, 1, 1)
            logσ_true = reshape(log.(Tloc[0.7f0, 1.2f0]), dloc, 1, 1)

            # Fixed noises keep the objective deterministic across optimization steps.
            z_target = randn(Tloc, dloc, nloc, bsloc)
            z_model = randn(Tloc, dloc, nloc, bsloc)

            target = μ_true .+ exp.(logσ_true) .* z_target

            # Trainable Gaussian parameters (intentionally off-target).
            μ = reshape(Tloc[-1.0f0, 0.8f0], dloc, 1, 1)
            logσ = reshape(log.(Tloc[1.8f0, 0.4f0]), dloc, 1, 1)

            if use_gpu
                μ_true = cu(μ_true)
                logσ_true = cu(logσ_true)
                z_target = cu(z_target)
                z_model = cu(z_model)
                target = cu(target)
                μ = cu(μ)
                logσ = cu(logσ)
            end

            loss(μp, logσp) = begin
                xhat = μp .+ exp.(logσp) .* z_model
                maximum_mean_discrepancy(xhat, target; sigma=[0.5f0, 1.0f0, 2.0f0])
            end

            l0 = loss(μ, logσ)
            gμ, glogσ = Zygote.gradient(loss, μ, logσ)

            @test gμ !== nothing
            @test glogσ !== nothing
            @test all_finite(gμ)
            @test all_finite(glogσ)

            # A few optimization steps should improve the objective on this toy task.
            μ_opt = copy(μ)
            logσ_opt = copy(logσ)
            η = Tloc(1f-1)
            nsteps = 12
            for _ in 1:nsteps
                dμ, dlogσ = Zygote.gradient(loss, μ_opt, logσ_opt)
                μ_opt .-= η .* dμ
                logσ_opt .-= η .* dlogσ
            end
            l1 = loss(μ_opt, logσ_opt)

            @test all_finite(l0)
            @test all_finite(l1)
            @info "Gaussian fit gradient case" l0=Float32(l0) l1=Float32(l1)
            @test Float32(l1) < Float32(l0)
        end

        gaussian_fit_gradient_case(false)

        if CUDA.functional()
            gaussian_fit_gradient_case(true)
        else
            @test_skip "CUDA not functional in this environment"
        end
    end

    @testset "smoke benchmark mmd vs chamfer" begin
        xb, yb = make_batch(T, 3, 256, 24)

        t_cpu_mmd = @belapsed maximum_mean_discrepancy($xb, $yb; sigma=1f0)
        t_cpu_chamfer = @belapsed chamfer_distance($xb, $yb)
        ratio_cpu = t_cpu_mmd / t_cpu_chamfer

        @info "MMD vs chamfer benchmark (CPU)" mmd=t_cpu_mmd chamfer=t_cpu_chamfer ratio_mmd_over_chamfer=ratio_cpu

        @test t_cpu_mmd > 0
        @test t_cpu_chamfer > 0
        @test isfinite(ratio_cpu)

        if CUDA.functional()
            xbg = cu(xb)
            ybg = cu(yb)

            t_gpu_mmd = @belapsed CUDA.@sync maximum_mean_discrepancy($xbg, $ybg; sigma=1f0)
            t_gpu_chamfer = @belapsed CUDA.@sync chamfer_distance($xbg, $ybg)
            ratio_gpu = t_gpu_mmd / t_gpu_chamfer

            @info "MMD vs chamfer benchmark (GPU)" mmd=t_gpu_mmd chamfer=t_gpu_chamfer ratio_mmd_over_chamfer=ratio_gpu

            # Expected in many setups: MMD is somewhat slower than chamfer.
            # Keep this as a smoke check only, not a strict perf gate.
            @test t_gpu_mmd > 0
            @test t_gpu_chamfer > 0
            @test isfinite(ratio_gpu)
        else
            @test_skip "CUDA not functional in this environment"
        end
    end

    @testset "smoke benchmark mmd (multi-sigma) vs chamfer" begin
        xb, yb = make_batch(T, 3, 1024, 2)
        σ_vec = [0.25f0, 0.5f0, 1.0f0, 2.0f0, 4.0f0]

        t_cpu_mmd = @belapsed maximum_mean_discrepancy($xb, $yb; sigma=[0.25f0, 0.5f0, 1.0f0, 2.0f0, 4.0f0])
        t_cpu_chamfer = @belapsed chamfer_distance($xb, $yb)
        ratio_cpu = t_cpu_mmd / t_cpu_chamfer

        @info "MMD (5 σs) vs chamfer benchmark (CPU)" mmd=t_cpu_mmd chamfer=t_cpu_chamfer ratio_mmd_over_chamfer=ratio_cpu

        @test t_cpu_mmd > 0
        @test t_cpu_chamfer > 0
        @test isfinite(ratio_cpu)

        if CUDA.functional()
            xbg = cu(xb);
            ybg = cu(yb);

            t_gpu_mmd = @belapsed CUDA.@sync maximum_mean_discrepancy($xbg, $ybg; sigma=[0.25f0, 0.5f0, 1.0f0, 2.0f0, 4.0f0])
            t_gpu_chamfer = @belapsed CUDA.@sync chamfer_distance($xbg, $ybg)
            ratio_gpu = t_gpu_mmd / t_gpu_chamfer

            @info "MMD (5 σs) vs chamfer benchmark (GPU)" mmd=t_gpu_mmd chamfer=t_gpu_chamfer ratio_mmd_over_chamfer=ratio_gpu

            # Expected in many setups: MMD is somewhat slower than chamfer.
            # Keep this as a smoke check only, not a strict perf gate.
            @test t_gpu_mmd > 0
            @test t_gpu_chamfer > 0
            @test isfinite(ratio_gpu)
        else
            @test_skip "CUDA not functional in this environment"
        end
    end

end

