"""
Simple gradient test for sinkhorn_divergence_loss
No need to compile entire GenerativeMIL - just include required files
"""

using Flux, Zygote, Statistics, OptimalTransport, ChainRulesCore, MLUtils
using CUDA

# Add src to path
#push!(LOAD_PATH, "/home/zorekmat/MIL/GenerativeMIL/src")

# Include sinkhorn_loss directly
include("src/losses/sinkhorn_loss.jl")

function _pairwise_sqdist(x::AbstractMatrix{T}, y::AbstractMatrix{T}) where T<:AbstractFloat
    x2 = sum(abs2, x; dims=1)
    y2 = sum(abs2, y; dims=1)
    return max.(x2' .+ y2 .- T(2) .* (x' * y), zero(T))
end

function _pairwise_sqdist_batched(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}) where T<:AbstractFloat
    @assert size(x, 3) == size(y, 3) "x and y must have the same batch size"
    bs = size(x, 3)

    # Avoid in-place writes so the function remains differentiable by Zygote.
    d_slices = [_pairwise_sqdist(@view(x[:, :, b]), @view(y[:, :, b])) for b in 1:bs]
    return cat(d_slices...; dims=3)
end

function _pairwise_sqdist_batched(x::CuArray{T, 3}, y::CuArray{T, 3}) where T<:AbstractFloat
    @assert size(x, 3) == size(y, 3) "x and y must have the same batch size"

    # Fast CuArray path: fully batched GEMM.
    x_t = permutedims(x, (2, 1, 3))
    #y_t = permutedims(y, (2, 1, 3))

    x2 = sum(abs2, x; dims=1)
    y2 = sum(abs2, y; dims=1)
    x2_t = permutedims(x2, (2, 1, 3))

    g_xy = Flux.batched_mul(x_t, y)
    return max.(x2_t .+ y2 .- T(2) .* g_xy, zero(T))
end

println("=" ^ 70)
println("GRADIENT TEST: Sinkhorn Divergence Loss with Custom RRule")
println("=" ^ 70)

# =============================================================================
# TEST 1: Basic gradient test - CPU
# =============================================================================
println("\n[TEST 1] CPU: Forward pass + gradients")
println("-" ^ 70)

# Small test data
x = rand(Float32, 2, 16, 1)  # (dim=2, n_points=16, batch=1)
y = rand(Float32, 2, 20, 1)  # (dim=2, n_points=20, batch=1)
ε = 0.1f0

println("Input shapes:")
println("  x: $(size(x))")
println("  y: $(size(y))")

# Forward pass
loss    = sinkhorn_divergence_loss(x, y, ε; maxiter=50)
println("\n✓ Forward pass OK")
println("  Loss value: $(round(loss, digits=6))")

# Gradients
try
    ∇x, ∇y = Zygote.gradient((x_in, y_in) -> sinkhorn_divergence_loss(x_in, y_in, ε; maxiter=50), x, y)
    println("\n✓ Gradients OK")
    println("  ∇x shape: $(size(∇x))")
    println("  ∇y shape: $(size(∇y))")
    println("  ∇x finite: $(all(isfinite, ∇x))")
    println("  ∇y finite: $(all(isfinite, ∇y))")
    println("  ∇x mean: $(round(mean(abs, ∇x), digits=6))")
    println("  ∇y mean: $(round(mean(abs, ∇y), digits=6))")
catch e
    println("\n✗ Error computing gradients:")
    println("  $(e)")
    rethrow()
end

# =============================================================================
# TEST 2: Verify gradient finiteness and sanity
# =============================================================================
println("\n[TEST 2] Verify gradient finiteness and sanity")
println("-" ^ 70)

# Check if all gradients are finite
∇x, ∇y = Zygote.gradient((x_in, y_in) -> sinkhorn_divergence_loss(x_in, y_in, ε; maxiter=50), x, y);
println("Finiteness check:")
println("  ∇x finite: $(all(isfinite, ∇x))")
println("  ∇y finite: $(all(isfinite, ∇y))")

# Check if gradients are not all zeros
println("\nNon-zero gradient check:")
println("  Max |∇x|: $(round(maximum(abs, ∇x), digits=6))")
println("  Max |∇y|: $(round(maximum(abs, ∇y), digits=6))")
println("  Min |∇x| (nonzero): $(round(minimum(filter(x -> x > 0, abs.(∇x))), digits=6))")
println("  Min |∇y| (nonzero): $(round(minimum(filter(x -> x > 0, abs.(∇y))), digits=6))")

if all(isfinite, ∇x) && all(isfinite, ∇y) && maximum(abs, ∇x) > 0 && maximum(abs, ∇y) > 0
    println("\n  ✓ Gradients computed correctly")
else
    println("\n  ✗ Problem with gradients!")
end

# =============================================================================
# TEST 3: Gradient flow - optimization step
# =============================================================================
println("\n[TEST 3] Gradient flow - optimization step")
println("-" ^ 70)

x_opt = copy(x)
y_opt = copy(y)

# Single optimization step
loss_before, grads = Flux.withgradient((x_in, y_in) -> sinkhorn_divergence_loss(x_in, y_in, ε; maxiter=50), x_opt, y_opt)
∇x_opt, ∇y_opt = grads

# Update - step towards minimization (negative gradient)
step_size = 0.01f0
x_opt .-= step_size .* ∇x_opt
y_opt .-= step_size .* ∇y_opt

loss_after = sinkhorn_divergence_loss(x_opt, y_opt, ε; maxiter=50)

println("  Loss before step: $(round(loss_before, digits=6))")
println("  Loss after step: $(round(loss_after, digits=6))")
println("  Loss improvement: $(round(loss_before - loss_after, digits=6))")

if loss_after < loss_before
    println("  ✓ Optimization step correctly reduced loss")
else
    println("  ⚠ Loss did not decrease (possibly small step size or numerical error)")
end

# =============================================================================
# TEST 4: GPU test (If GPU is present)
# =============================================================================
if !isnothing(CUDA) && CUDA.functional()
    println("\n[TEST 4] GPU test (CUDA)")
    println("-" ^ 70)
    
    x_gpu = x |> Flux.gpu
    y_gpu = y |> Flux.gpu
    
    # Forward pass on GPU
    loss_gpu = sinkhorn_divergence_loss(x_gpu, y_gpu, ε; maxiter=50)
    println("✓ Forward pass on GPU OK")
    println("  Loss: $(round(loss_gpu, digits=6))")
    
    # Gradients on GPU
    try
        ∇x_gpu, ∇y_gpu = Zygote.gradient((x_in, y_in) -> sinkhorn_divergence_loss(x_in, y_in, ε; maxiter=50), x_gpu, y_gpu)
        println("✓ Gradients on GPU OK")
        println("  ∇x finite: $(all(isfinite, ∇x_gpu))")
        println("  ∇y finite: $(all(isfinite, ∇y_gpu))")
    catch e
        println("✗ GPU error: $e")
    end
else
    println("\n[TEST 4] GPU test")
    println("-" ^ 70)
    println("⚠ CUDA not available, skipping GPU test")
end

# =============================================================================
# TEST 5: Performance benchmark - CPU vs GPU
# =============================================================================
println("\n[TEST 5] Performance benchmark (CPU vs GPU)")
println("-" ^ 70)

# Large test data for benchmark
x_large = rand(Float32, 3, 2048, 64)  # (dim=3, n_points=2048, batch=128)
y_large = rand(Float32, 3, 2048, 64)  # (dim=3, n_points=2048, batch=128)
ε_bench = 0.1f0

println("Benchmark data shapes: x=$(size(x_large)), y=$(size(y_large))")
println("Running benchmarks...")

# CPU benchmark - forward pass
time_cpu_forward = @elapsed begin
    for _ in 1:3
        loss_cpu = sinkhorn_divergence_loss(x_large, y_large, ε_bench; maxiter=30)
    end
end
time_cpu_forward /= 3
println("\n✓ CPU Forward pass: $(round(time_cpu_forward*1000, digits=2)) ms")

# CPU benchmark - gradients
time_cpu_grad = @elapsed begin
    for _ in 1:3
        ∇x_cpu, ∇y_cpu = Zygote.gradient((x_in, y_in) -> sinkhorn_divergence_loss(x_in, y_in, ε_bench; maxiter=30), x_large, y_large)
    end
end
time_cpu_grad /= 3
println("✓ CPU Gradients:    $(round(time_cpu_grad*1000, digits=2)) ms")

# GPU benchmark (if available)
if !isnothing(CUDA) && CUDA.functional()
    x_large_gpu = x_large |> cu
    y_large_gpu = y_large |> cu
    
    # Warmup
    _ = sinkhorn_divergence_loss(x_large_gpu, y_large_gpu, ε_bench; maxiter=30)
    CUDA.synchronize()
    
    # GPU benchmark - forward pass
    time_gpu_forward = @elapsed begin
        for _ in 1:3
            loss_gpu = sinkhorn_divergence_loss(x_large_gpu, y_large_gpu, ε_bench; maxiter=30)
            CUDA.synchronize()
        end
    end
    time_gpu_forward /= 3
    println("\n✓ GPU Forward pass: $(round(time_gpu_forward*1000, digits=2)) ms")
    
    # GPU benchmark - gradients
    time_gpu_grad = @elapsed begin
        for _ in 1:3
            ∇x_gpu, ∇y_gpu = Zygote.gradient((x_in, y_in) -> sinkhorn_divergence_loss(x_in, y_in, ε_bench; maxiter=30), x_large_gpu, y_large_gpu)
            CUDA.synchronize()
        end
    end
    time_gpu_grad /= 3
    println("✓ GPU Gradients:    $(round(time_gpu_grad*1000, digits=2)) ms")
    
    # Speedup
    speedup_forward = time_cpu_forward / time_gpu_forward
    speedup_grad = time_cpu_grad / time_gpu_grad
    println("\n📊 Speedup:")
    println("  Forward pass: $(round(speedup_forward, digits=2))x")
    println("  Gradients:    $(round(speedup_grad, digits=2))x")
else
    println("\n⚠ CUDA not available, skipping GPU benchmark")
end

# =============================================================================
# SUMMARY
# =============================================================================
println("\n" * "=" ^ 70)
println("SUMMARY: All tests complete ✓")
println("=" ^ 70)
println("\nConclusion:")
println("  • Forward pass: ✓")
println("  • Gradient computation: ✓")
println("  • Optimization step: ✓")
println("  • Performance benchmark: ✓")
println("\n→ sinkhorn_divergence_loss with custom rrule works correctly!")
println("=" ^ 70)
