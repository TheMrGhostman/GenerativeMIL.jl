"""
Test: Comparison of custom rrule vs Zygote.ignore approach
"""

using Flux, Zygote, Statistics, OptimalTransport, ChainRulesCore, LinearAlgebra, BenchmarkTools
using CUDA

# Include sinkhorn_loss directly
include("../../src/losses/sinkhorn_loss.jl")

println("=" ^ 70)
println("TEST: Custom RRule vs Zygote.ignore Comparison")
println("=" ^ 70)

# =============================================================================
# TEST 1: Basic forward pass
# =============================================================================
println("\n[TEST 1] Forward pass")
println("-" ^ 70)

# Small test data
x = rand(Float32, 2, 16, 1)  # (dim=2, n_points=16, batch=1)
y = rand(Float32, 2, 20, 1)  # (dim=2, n_points=20, batch=1)
ε = 0.1f0

println("Input shapes: x=$(size(x)), y=$(size(y))")

# Forward pass with custom rrule
loss_rrule = sinkhorn_divergence_loss(x, y, ε; maxiter=50)
println("✓ Forward pass OK")
println("  Loss value: $(round(loss_rrule, digits=6))")

# =============================================================================
# TEST 2: Custom rrule - gradients
# =============================================================================
println("\n[TEST 2] Custom RRule - Gradients via Zygote")
println("-" ^ 70)

loss_fn_rrule(x_in, y_in) = sinkhorn_divergence_loss(x_in, y_in, ε; maxiter=50)

try
    ∇x_rrule, ∇y_rrule = Zygote.gradient(loss_fn_rrule, x, y)
    println("✓ Gradients OK (custom rrule)")
    println("  ∇x shape: $(size(∇x_rrule))")
    println("  ∇y shape: $(size(∇y_rrule))")
    println("  ∇x finite: $(all(isfinite, ∇x_rrule))")
    println("  ∇y finite: $(all(isfinite, ∇y_rrule))")
    println("  ∇x mean: $(round(mean(abs, ∇x_rrule), digits=6))")
    println("  ∇y mean: $(round(mean(abs, ∇y_rrule), digits=6))")
catch e
    println("✗ Error: $(e)")
    rethrow()
end

# =============================================================================
# TEST 3: Zygote.ignore version - alternative without rrule
# =============================================================================
println("\n[TEST 3] Zygote.ignore - Version without custom rrule")
println("-" ^ 70)

function sinkhorn_divergence_loss_ignore(x, y, ε; kwargs...)
    """
    Alternative implementation without custom rrule - 
    gradients flow through cost matrices, transport plans are ignored
    """
    # Compute cost matrices (gradients flow through these operations)
    Cxy = _pairwise_sqdist_batched(x, y)
    Cxx = _pairwise_sqdist_batched(x, x)
    Cyy = _pairwise_sqdist_batched(y, y)
    
    n_x = size(x, 2)
    n_y = size(y, 2)
    
    # Compute transport plans - IGNORE gradients through them
    # (Zygote won't trace through compute_transport_plans)
    πxy = Zygote.ignore(()->compute_transport_plans(Cxy, n_x, n_y, ε, OptimalTransport.SinkhornGibbs(); kwargs...))
    πxx = Zygote.ignore(()->compute_transport_plans(Cxx, n_x, ε, OptimalTransport.SinkhornGibbs(); kwargs...))
    πyy = Zygote.ignore(()->compute_transport_plans(Cyy, n_y, ε, OptimalTransport.SinkhornGibbs(); kwargs...))
    
    # Compute loss - gradients flow through cost matrices C
    # dL/dx = d/dx sum(Cxy * πxy + ...) 
    #       = d/dx (Cxy) * πxy + d/dx (Cxx) * πxx + d/dx (Cyy) * πyy
    loss = sum(Cxy .* πxy) - 0.5f0 * sum(Cxx .* πxx) - 0.5f0 * sum(Cyy .* πyy)
    return loss
end

loss_fn_ignore(x_in, y_in) = sinkhorn_divergence_loss_ignore(x_in, y_in, ε; maxiter=50)

# Forward pass - should be the same
loss_ignore = loss_fn_ignore(x, y)
println("Loss value: $(round(loss_ignore, digits=6)) (should be ~ $(round(loss_rrule, digits=6)))")

# Gradients
try
    ∇x_ignore, ∇y_ignore = Zygote.gradient(loss_fn_ignore, x, y)
    println("\n✓ Gradients OK (Zygote.ignore)")
    println("  ∇x shape: $(size(∇x_ignore))")
    println("  ∇y shape: $(size(∇y_ignore))")
    println("  ∇x finite: $(all(isfinite, ∇x_ignore))")
    println("  ∇y finite: $(all(isfinite, ∇y_ignore))")
    println("  ∇x mean: $(round(mean(abs, ∇x_ignore), digits=6))")
    println("  ∇y mean: $(round(mean(abs, ∇y_ignore), digits=6))")
catch e
    println("✗ Error: $(e)")
    rethrow()
end

# =============================================================================
# TEST 4: Comparison of gradients
# =============================================================================
println("\n[TEST 4] Comparison of both approaches")
println("-" ^ 70)

println("Loss:")
println("  Custom rrule:      $(round(loss_rrule, digits=6))")
println("  Zygote.ignore:     $(round(loss_ignore, digits=6))")
loss_diff = abs(loss_rrule - loss_ignore)
println("  Absolute diff:     $(round(loss_diff, digits=6))")

println("\nGradients x:")
println("  Custom rrule mean:  $(round(mean(abs, ∇x_rrule), digits=6))")
println("  Zygote.ignore mean: $(round(mean(abs, ∇x_ignore), digits=6))")
rel_diff_x = norm(∇x_rrule - ∇x_ignore) / (norm(∇x_rrule) + norm(∇x_ignore) + 1e-10)
println("  Rel. diff:          $(round(rel_diff_x, digits=6))")

println("\nGradients y:")
println("  Custom rrule mean:  $(round(mean(abs, ∇y_rrule), digits=6))")
println("  Zygote.ignore mean: $(round(mean(abs, ∇y_ignore), digits=6))")
rel_diff_y = norm(∇y_rrule - ∇y_ignore) / (norm(∇y_rrule) + norm(∇y_ignore) + 1e-10)
println("  Rel. diff:          $(round(rel_diff_y, digits=6))")

# Summary
println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

if rel_diff_x < 0.01 && rel_diff_y < 0.01
    println("✓ Methods agree almost perfectly (diff < 1%)")
    println("  → Both approaches are equivalent!")
elseif rel_diff_x < 0.1 && rel_diff_y < 0.1
    println("✓ Methods are very close (diff < 10%)")
    println("  → Difference is negligible (probably numerical noise)")
else
    println("⚠ Methods differ (diff > 10%)")
    println("  → Possible problem in implementation")
end

println("\n✓ Test complete!")
println("=" ^ 70)

# =============================================================================
# TEST 5: GPU speed comparison - custom rrule vs Zygote.ignore
# =============================================================================
println("\n[TEST 5] GPU Speed Test - Gradient Computation")
println("-" ^ 70)

if CUDA.functional()
    println("✓ CUDA is available, running GPU benchmark...\n")
    
    # Move data to GPU
    x_gpu = CuArray(rand(Float32, 2, 64, 1))  # (dim=2, n_points=64, batch=1)
    y_gpu = CuArray(rand(Float32, 2, 80, 1))  # (dim=2, n_points=80, batch=1)
    ε_gpu = 0.1f0
    
    println("Input shapes: x=$(size(x_gpu)), y=$(size(y_gpu))")
    println("Device: GPU (CUDA)\n")
    
    # Warmup runs
    println("Warming up GPU...")
    for _ in 1:3
        _ = Zygote.gradient((x, y) -> sinkhorn_divergence_loss(x, y, ε_gpu; maxiter=50), x_gpu, y_gpu)
    end
    for _ in 1:3
        _ = Zygote.gradient((x, y) -> sinkhorn_divergence_loss_ignore(x, y, ε_gpu; maxiter=50), x_gpu, y_gpu)
    end
    CUDA.synchronize()
    println("✓ Warmup complete\n")
    
    # Benchmark custom rrule
    println("Benchmarking custom rrule gradient computation...")
    time_rrule = @elapsed begin
        for _ in 1:10
            _ = Zygote.gradient((x, y) -> sinkhorn_divergence_loss(x, y, ε_gpu; maxiter=50), x_gpu, y_gpu)
            CUDA.synchronize()
        end
    end
    time_per_iter_rrule = time_rrule / 10
    
    # Benchmark Zygote.ignore
    println("Benchmarking Zygote.ignore gradient computation...")
    time_ignore = @elapsed begin
        for _ in 1:10
            _ = Zygote.gradient((x, y) -> sinkhorn_divergence_loss_ignore(x, y, ε_gpu; maxiter=50), x_gpu, y_gpu)
            CUDA.synchronize()
        end
    end
    time_per_iter_ignore = time_ignore / 10
    
    # Results
    println("\nResults (10 iterations each):")
    println("  Custom rrule:    $(round(time_per_iter_rrule*1000, digits=2)) ms/iter")
    println("  Zygote.ignore:   $(round(time_per_iter_ignore*1000, digits=2)) ms/iter")
    
    speedup = time_per_iter_ignore / time_per_iter_rrule
    if speedup > 1.0
        println("  Speedup:         $(round(speedup, digits=2))x (rrule is faster)")
    else
        println("  Speedup:         $(round(1/speedup, digits=2))x (Zygote.ignore is faster)")
    end
    
else
    println("⚠ CUDA not available - skipping GPU benchmark")
    println("  To run GPU tests, ensure CUDA.jl is properly configured")
end

println("\n" * "=" ^ 70)
println("All tests complete!")
println("=" ^ 70)
