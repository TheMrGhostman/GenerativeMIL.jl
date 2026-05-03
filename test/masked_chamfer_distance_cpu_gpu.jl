"""
Unit tests and lightweight benchmarks for masked Chamfer distance.

Checks:
- new masked implementation matches the previous CPU loop version
- single-mask and two-mask variants both behave correctly
- GPU execution works and matches CPU results
- simple benchmark comparison for CPU old vs CPU new and GPU new vs CPU new
"""

using Test
using Random
using Flux
using CUDA
using BenchmarkTools
using Statistics
using NearestNeighbors
using Zygote

include(joinpath(@__DIR__, "..", "src", "losses", "chamfer_distance.jl"))
include(joinpath(@__DIR__, "..", "src", "losses", "masked_chamfer_distance.jl"))


Random.seed!(42)

all_finite(x::Number) = isfinite(x)
all_finite(x) = all(isfinite.(Array(x)))


function old_masked_chamfer_distance_cpu(x::AbstractArray{T,3}, y::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    x_cpu = cpu(x)
    y_cpu = cpu(y)
    x_mask_cpu = cpu(x_mask)

    total = zero(T)
    nb = size(x_cpu, 3)
    @inbounds for i in 1:nb
        total += chamfer_distance(
            Array(_unmask(view(x_cpu, :, :, i), view(x_mask_cpu, :, :, i))),
            Array(view(y_cpu, :, :, i))
        )
    end
    return total / T(nb)
end

function old_masked_chamfer_distance_cpu(x::AbstractArray{T,3}, y::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}, y_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    x_cpu = cpu(x)
    y_cpu = cpu(y)
    x_mask_cpu = cpu(x_mask)
    y_mask_cpu = cpu(y_mask)

    total = zero(T)
    nb = size(x_cpu, 3)
    @inbounds for i in 1:nb
        total += chamfer_distance(
            Array(_unmask(view(x_cpu, :, :, i), view(x_mask_cpu, :, :, i))),
            Array(_unmask(view(y_cpu, :, :, i), view(y_mask_cpu, :, :, i)))
        )
    end
    return total / T(nb)
end

function make_mask(d::Int, n::Int, bs::Int; keep_prob::Float64=0.75)
    mask = rand(Bool, 1, n, bs)
    for b in 1:bs
        mask[1, 1, b] = true
        if all(!, @view mask[1, :, b])
            mask[1, 1, b] = true
        end
    end
    return mask
end

function make_data(T::Type, d::Int, n::Int, bs::Int)
    x = randn(T, d, n, bs)
    y = randn(T, d, n, bs)
    x_mask = make_mask(d, n, bs)
    y_mask = make_mask(d, n, bs)
    return x, y, x_mask, y_mask
end

@testset "masked chamfer distance" begin
    T = Float32
    d = 3
    n = 24
    bs = 4

    x, y, x_mask, y_mask = make_data(T, d, n, bs)

    @testset "CPU correctness" begin
        old_single = old_masked_chamfer_distance_cpu(x, y, x_mask)
        new_single = masked_chamfer_distance(x, y, x_mask)
        @test isfinite(old_single)
        @test isfinite(new_single)
        @test isapprox(new_single, old_single; rtol=1f-4, atol=1f-5)

        old_double = old_masked_chamfer_distance_cpu(x, y, x_mask, y_mask)
        new_double = masked_chamfer_distance(x, y, x_mask, y_mask)
        @test isfinite(old_double)
        @test isfinite(new_double)
        @test isapprox(new_double, old_double; rtol=1f-4, atol=1f-5)
    end

    @testset "GPU correctness" begin
        if !CUDA.functional()
            @test_skip "CUDA not functional in this environment"
        else
            #CUDA.allowscalar(false)
            xg = cu(x)
            yg = cu(y)
            xmg = cu(x_mask)
            ymg = cu(y_mask)

            gpu_single = CUDA.@sync masked_chamfer_distance(xg, yg, xmg)
            cpu_single = masked_chamfer_distance(x, y, x_mask)
            @test isfinite(gpu_single)
            @test isapprox(Float32(gpu_single), cpu_single; rtol=1f-4, atol=1f-5)

            gpu_double = CUDA.@sync masked_chamfer_distance(xg, yg, xmg, ymg)
            cpu_double = masked_chamfer_distance(x, y, x_mask, y_mask)
            @test isfinite(gpu_double)
            @test isapprox(Float32(gpu_double), cpu_double; rtol=1f-4, atol=1f-5)
        end
    end

    @testset "speed benchmarks" begin
        x_b, y_b, x_mask_b, y_mask_b = make_data(T, d, n, bs)

        cpu_old_single = @belapsed old_masked_chamfer_distance_cpu($x_b, $y_b, $x_mask_b)
        cpu_new_single = @belapsed masked_chamfer_distance($x_b, $y_b, $x_mask_b)
        cpu_old_double = @belapsed old_masked_chamfer_distance_cpu($x_b, $y_b, $x_mask_b, $y_mask_b)
        cpu_new_double = @belapsed masked_chamfer_distance($x_b, $y_b, $x_mask_b, $y_mask_b)

        @info "masked chamfer benchmark (CPU single mask)" old_cpu=cpu_old_single new_cpu=cpu_new_single speedup=cpu_old_single / cpu_new_single
        @info "masked chamfer benchmark (CPU two masks)" old_cpu=cpu_old_double new_cpu=cpu_new_double speedup=cpu_old_double / cpu_new_double

        @test cpu_old_single > 0
        @test cpu_new_single > 0
        @test cpu_old_double > 0
        @test cpu_new_double > 0

        if CUDA.functional()
            CUDA.allowscalar(false)
            xg = cu(x_b)
            yg = cu(y_b)
            xmg = cu(x_mask_b)
            ymg = cu(y_mask_b)

            gpu_new_single = @belapsed CUDA.@sync masked_chamfer_distance($xg, $yg, $xmg)
            gpu_new_double = @belapsed CUDA.@sync masked_chamfer_distance($xg, $yg, $xmg, $ymg)
            cpu_new_single_for_gpu = cpu_new_single
            cpu_new_double_for_gpu = cpu_new_double

            @info "masked chamfer benchmark (GPU vs CPU new, single mask)" cpu_new=cpu_new_single_for_gpu gpu_new=gpu_new_single speedup=cpu_new_single_for_gpu / gpu_new_single
            @info "masked chamfer benchmark (GPU vs CPU new, two masks)" cpu_new=cpu_new_double_for_gpu gpu_new=gpu_new_double speedup=cpu_new_double_for_gpu / gpu_new_double

            @test gpu_new_single > 0
            @test gpu_new_double > 0
        else
            @test_skip "CUDA not functional in this environment"
        end
    end
end
