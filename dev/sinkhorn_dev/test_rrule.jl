using Flux, Statistics, CUDA, Zygote, OptimalTransport, ChainRulesCore
using BenchmarkTools

# Add src to path
push!(LOAD_PATH, "/home/zorekmat/MIL/GenerativeMIL/src")
using GenerativeMIL

# Test data
T = Float32
x = rand(T, 3, 64, 4) |> cu;  # (d=3, n_x=64, bs=4)
y = rand(T, 3, 128, 4) |> cu; # (d=3, n_y=128, bs=4)

ε = 0.1f0

println("=" ^ 60)
println("Testing sinkhorn_divergence_loss with rrule")
println("=" ^ 60)

# Test 1: Forward pass
println("\n1. Forward pass:")
loss = GenerativeMIL.Losses.sinkhorn_divergence_loss(x, y, ε; maxiter=100)
println("Loss value: $loss")

# Test 2: Compute gradients
println("\n2. Computing gradients via Zygote...")
try
    g = Zygote.gradient((x_in, y_in) -> GenerativeMIL.Losses.sinkhorn_divergence_loss(x_in, y_in, ε; maxiter=100), x, y)
    println("✓ Gradient computation successful")
    println("  ∇x shape: $(size(g[1]))")
    println("  ∇y shape: $(size(g[2]))")
    println("  ∇x contains NaN: $(any(isnan, g[1]))")
    println("  ∇y contains NaN: $(any(isnan, g[2]))")
    
    if g[1] !== nothing && g[2] !== nothing
        println("  ∇x range: [$(minimum(g[1])), $(maximum(g[1]))]")
        println("  ∇y range: [$(minimum(g[2])), $(maximum(g[2]))]")
    end
catch e
    println("✗ Gradient computation failed:")
    println("  Error: $e")
    println("  Stacktrace:")
    showerror(stdout, e, catch_backtrace())
end

# Test 3: Numerical gradient check (with smaller batch for speed)
println("\n3. Numerical gradient check (finite differences):")
try
    x_small = rand(T, 2, 16, 2) |> cu
    y_small = rand(T, 2, 16, 2) |> cu
    eps_fd = 1e-4f0
    
    loss_fn(x, y) = GenerativeMIL.Losses.sinkhorn_divergence_loss(x, y, ε; maxiter=50)
    
    # Compute loss and gradient
    loss = loss_fn(x_small, y_small)
    g = Zygote.gradient((x_in, y_in) -> loss_fn(x_in, y_in), x_small, y_small)
    
    # Numerical gradient for first element of x
    x_pert = copy(x_small)
    x_pert[1, 1, 1] += eps_fd
    loss_pert = loss_fn(x_pert, y_small)
    numerical_grad_x = (loss_pert - loss) / eps_fd
    analytical_grad_x = g[1][1, 1, 1]
    
    rel_error = abs(numerical_grad_x - analytical_grad_x) / (abs(numerical_grad_x) + eps_fd)
    println("  Relative error for x[1,1,1]: $rel_error")
    println("  Numerical grad: $numerical_grad_x")
    println("  Analytical grad: $analytical_grad_x")
    
    if rel_error < 1e-2f0
        println("  ✓ Gradient check passed (error < 1%)")
    else
        println("  ⚠ Gradient check warning (error > 1%)")
    end
catch e
    println("✗ Numerical gradient check failed:")
    println("  Error: $e")
end

# Test 4: Benchmark
println("\n4. Performance benchmark:")
try
    @time loss = GenerativeMIL.Losses.sinkhorn_divergence_loss(x, y, ε; maxiter=100)
    println("✓ Forward pass benchmark complete")
    
    @time g = Zygote.gradient((x_in, y_in) -> GenerativeMIL.Losses.sinkhorn_divergence_loss(x_in, y_in, ε; maxiter=100), x, y)
    println("✓ Gradient computation benchmark complete")
catch e
    println("✗ Benchmark failed: $e")
end

println("\n" * "=" ^ 60)
println("Test complete")
println("=" ^ 60)
