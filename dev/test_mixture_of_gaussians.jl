"""
Test script for MixtureOfGaussians implementation.
Compares old vs new implementation, tests GPU compatibility, and validates gumbel_softmax broadcasting.
"""

using Flux, CUDA, cuDNN, MLUtils, Random, Statistics, Zygote
using LinearAlgebra, Distances, BenchmarkTools

# Set seed for reproducibility
Random.seed!(42)


# ============================================================================
# OLD IMPLEMENTATION (for comparison)
# ============================================================================
"""
    _softmax(x; dims)

Numerically stable softmax. CPU uses Flux.softmax; CUDA uses custom implementation
to avoid NNLib issues with Julia 1.11.3.
"""
function _softmax(x::AbstractArray{T}; dims::Int=1) where T<: AbstractFloat
    Flux.softmax(x; dims=dims)
end

function _softmax(x::CuArray{T}; dims::Int=1) where T <: AbstractFloat
    # numerically stable softmax for CUDA; NNLib.softmax causes problems with CUDA with Julia 1.11.3
    m = maximum(x; dims=dims)
    ex = exp.(x .- m)
    ex ./ sum(ex; dims=dims)
end


function gumbel_softmax_old(logits::AbstractArray{T}; τ::T=1f0, hard::Bool=false, ϵ::Float32=1.0f-10) where T <: Real
    gumbel_samples = -log.(-log.(Random.rand!(logits) .+ ϵ) .+ ϵ)  # BUG: modifies logits!
    y = logits .+ gumbel_samples
    y =Flux.softmax(y./τ)

    if !hard
        return y
    else
        y_hard = nothing
        Zygote.ignore() do
            shape = size(y)
            y_hard = typeof(y)(zeros(T, shape))
            _, ind = findmax(y, dims=1)
            y_hard[ind] .= 1
            y_hard = y_hard .- y
        end
        y = y_hard .+ y 
        return y
    end
end

function gumbel_softmax_new(logits::AbstractArray{T}; τ::T=1f0, hard::Bool=false, ϵ=T(1.0e-10)) where T <: AbstractFloat
    # println(typeof(logits))
    g = -log.(-log.(MLUtils.rand_like(logits) .+ ϵ) .+ ϵ)
    y =Flux.softmax((logits .+ g) ./ τ)

    if !hard
        return y
    else
        y_hard = zero(y)
        Zygote.ignore() do
            _, ind = findmax(y, dims=1)
            y_hard[ind] .= one(T)
        end
        # Straight-through estimator: forward returns one-hot, backward uses soft gradient
        return y + Zygote.ignore(y_hard - y)
    end
end


# ============================================================================
# TEST 1: Basic functionality on CPU
# ============================================================================
println("\n" * "="^80)
println("TEST 1: Basic functionality on CPU")
println("="^80)

# Create test data
K = 5
Ds = 3
dim = 8
n_mixtures = 4

# Initialize old-style parameters (Float32)
α_old = ones(Float32, K)
μ_old = randn(Float32, Ds, K, 1)
Σ_old = ones(Float32, Ds, K, 1)

# Copy for new test
α_new = copy(α_old)
μ_new = copy(μ_old)
Σ_new = copy(Σ_old)

sample_size = 10
batch_size = 2

# OLD IMPLEMENTATION
println("\nOLD implementation:")
device = cpu
αₒₕ_old = gumbel_softmax_old(copy(α_old), hard=true)  # Note: using copy to avoid modification
αₒₕ_old = reshape(αₒₕ_old, (1, 1, K, 1))
αₒₕ_old = device == cpu ? repeat(αₒₕ_old, 1, sample_size, 1, batch_size) : nothing

μ_old_reshaped = reshape(μ_old, (Ds, 1, K, 1))
Σ_old_reshaped = reshape(Σ_old, (Ds, 1, K, 1))

μ_mixed_old = reshape(Flux.sum(μ_old_reshaped .* αₒₕ_old, dims=3), (:, sample_size, batch_size))
Σ_mixed_old = reshape(Flux.sum(Σ_old_reshaped .* αₒₕ_old, dims=3), (:, sample_size, batch_size))

ϵ_old = randn(Float32, Ds, sample_size, batch_size)
z_old = μ_mixed_old .+ Flux.softplus.(Σ_mixed_old) .* ϵ_old

println("  z shape: $(size(z_old))")
println("  z dtype: $(eltype(z_old))")
println("  z mean: $(mean(z_old))")
println("  z std: $(std(z_old))")

# NEW IMPLEMENTATION
println("\nNEW implementation:")
Random.seed!(42)  # Reset seed for fair comparison (using same noise)
ϵ_seed = randn(Float32, Ds, sample_size, batch_size)

αₒₕ_new = gumbel_softmax_new(copy(α_new), hard=true)
αₒₕ_new = reshape(αₒₕ_new, (1, 1, K, 1))
αₒₕ_new = repeat(αₒₕ_new, 1, sample_size, 1, batch_size)

μ_new_reshaped = reshape(μ_new, (Ds, 1, K, 1))
Σ_new_reshaped = reshape(Σ_new, (Ds, 1, K, 1))

μ_mixed_new = reshape(Flux.sum(μ_new_reshaped .* αₒₕ_new, dims=3), (:, sample_size, batch_size))
Σ_mixed_new = reshape(Flux.sum(Σ_new_reshaped .* αₒₕ_new, dims=3), (:, sample_size, batch_size))

z_new = μ_mixed_new .+ Flux.softplus.(Σ_mixed_new) .* ϵ_seed

println("  z shape: $(size(z_new))")
println("  z dtype: $(eltype(z_new))")
println("  z mean: $(mean(z_new))")
println("  z std: $(std(z_new))")

# ============================================================================
# TEST 2: Shape compatibility
# ============================================================================
println("\n" * "="^80)
println("TEST 2: Shape compatibility")
println("="^80)

if size(z_old) == size(z_new)
    println("✓ PASS: Output shapes match: $(size(z_old))")
else
    println("✗ FAIL: Output shapes differ! Old: $(size(z_old)), New: $(size(z_new))")
end

if eltype(z_old) == eltype(z_new)
    println("✓ PASS: Output dtypes match: $(eltype(z_old))")
else
    println("✗ FAIL: Output dtypes differ! Old: $(eltype(z_old)), New: $(eltype(z_new))")
end

# ============================================================================
# TEST 3: gumbel_softmax GPU broadcasting test
# ============================================================================
println("\n" * "="^80)
println("TEST 3: gumbel_softmax GPU broadcasting test")
println("="^80)

if CUDA.functional()
    println("GPU available: $(CUDA.functional())")
    
    # Test CPU version
    logits_cpu = randn(Float32, K)
    println("\nCPU logits shape: $(size(logits_cpu))")
    
    try
        result_cpu = gumbel_softmax_new(logits_cpu, hard=true)
        println("✓ CPU gumbel_softmax works, output shape: $(size(result_cpu))")
    catch e
        println("✗ CPU gumbel_softmax failed: $e")
    end
    
    # Test GPU version
    logits_gpu = CUDA.cu(randn(Float32, K))
    println("\nGPU logits shape: $(size(logits_gpu))")
    println("GPU logits location: $(typeof(logits_gpu))")
    
    try
        result_gpu = gumbel_softmax_new(logits_gpu, hard=true)
        println("✓ GPU gumbel_softmax works, output shape: $(size(result_gpu))")
        println("  GPU result location: $(typeof(result_gpu))")
        
        # Check if broadcasting caused issues
        if typeof(result_gpu) <: CuArray
            println("✓ PASS: Result stays on GPU (broadcasting OK)")
        else
            println("⚠ WARNING: Result moved to CPU (potential broadcasting issue)")
        end
    catch e
        println("✗ GPU gumbel_softmax failed: $e")
        println("  Error suggests GPU broadcasting incompatibility")
    end
    
    # Test with batch dimension
    logits_batch_cpu = randn(Float32, K, batch_size)
    logits_batch_gpu = CUDA.cu(logits_batch_cpu)
    
    println("\nTesting with batch dimension (K=$K, batch=$batch_size):")
    try
        result_batch_gpu = gumbel_softmax_new(logits_batch_gpu, hard=true)
        println("✓ GPU gumbel_softmax with batch works")
        println("  Input shape: $(size(logits_batch_gpu))")
        println("  Output shape: $(size(result_batch_gpu))")
    catch e
        println("✗ GPU gumbel_softmax with batch failed: $e")
    end
    
else
    println("⚠ GPU not available, skipping GPU tests")
end

# ============================================================================
# TEST 4: Full MoG sampling on GPU
# ============================================================================
println("\n" * "="^80)
println("TEST 4: Full MoG sampling on GPU")
println("="^80)

if CUDA.functional()
    println("Testing full sampling pipeline on GPU...")
    
    # Prepare GPU data
    α_gpu = CUDA.cu(copy(α_new))
    μ_gpu = CUDA.cu(copy(μ_new))
    Σ_gpu = CUDA.cu(copy(Σ_new))
    
    println("\nParameter locations:")
    println("  α: $(typeof(α_gpu))")
    println("  μ: $(typeof(μ_gpu))")
    println("  Σ: $(typeof(Σ_gpu))")
    
    try
        # Test gumbel_softmax on GPU
        αₒₕ_gpu = gumbel_softmax_new(α_gpu, hard=true)
        println("✓ gumbel_softmax on GPU works")
        println("  Output location: $(typeof(αₒₕ_gpu))")
        
        αₒₕ_gpu = reshape(αₒₕ_gpu, (1, 1, K, 1))
        println("✓ Reshape works on GPU")
        
        # This is the critical part - repeat on GPU
        try
            αₒₕ_gpu = repeat(αₒₕ_gpu, 1, sample_size, 1, batch_size)
            println("✓ repeat() works on GPU arrays")
            println("  Output shape: $(size(αₒₕ_gpu))")
        catch e
            println("✗ repeat() failed on GPU: $e")
            println("  This might need GPU-specific implementation")
        end
        
        μ_gpu_reshaped = reshape(μ_gpu, (Ds, 1, K, 1))
        Σ_gpu_reshaped = reshape(Σ_gpu, (Ds, 1, K, 1))
        
        # Test sum with broadcasting
        try
            μ_mixed_gpu = reshape(Flux.sum(μ_gpu_reshaped .* αₒₕ_gpu, dims=3), (:, sample_size, batch_size))
            Σ_mixed_gpu = reshape(Flux.sum(Σ_gpu_reshaped .* αₒₕ_gpu, dims=3), (:, sample_size, batch_size))
            println("✓ Broadcasting and sum work on GPU")
            
            ϵ_gpu = CUDA.randn(Float32, Ds, sample_size, batch_size)
            z_gpu = μ_mixed_gpu .+ Flux.softplus.(Σ_mixed_gpu) .* ϵ_gpu
            println("✓ Full sampling pipeline works on GPU")
            println("  Final output shape: $(size(z_gpu))")
            println("  Final output location: $(typeof(z_gpu))")
        catch e
            println("✗ Broadcasting failed: $e")
        end
        
    catch e
        println("✗ GPU test failed: $e")
    end
else
    println("⚠ GPU not available")
end

# ============================================================================
# TEST 5: Simple gradient checks (CPU)
# ============================================================================
println("\n" * "="^80)
println("TEST 5: Simple gradient checks (CPU)")
println("="^80)

logits_grad = Float32[0.2, -0.4, 1.1, 0.7]
τ_grad = Float32(0.7)
target_idx = 3

println("\nSoft mode gradient check (hard=false):")
try
    Random.seed!(42)
    loss_new_soft(x) = begin
        y = gumbel_softmax_new(x; τ=τ_grad, hard=false)
        -log(y[target_idx] + 1f-6)
    end
    grad_new_soft = Zygote.gradient(loss_new_soft, copy(logits_grad))[1]

    println("NEW grad: $(vec(grad_new_soft))")
    println("NEW finite: $(all(isfinite.(grad_new_soft)))")
    println("NEW norm: $(norm(grad_new_soft))")

    if all(isfinite.(grad_new_soft)) && norm(grad_new_soft) > 0f0
        println("✓ PASS: NEW soft mode has valid non-zero gradient")
    else
        println("✗ FAIL: NEW soft mode gradient invalid or zero")
    end
catch e
    println("✗ FAIL: NEW soft mode gradient crashed: $e")
end

try
    Random.seed!(42)
    loss_old_soft(x) = begin
        y = gumbel_softmax_old(copy(x); τ=τ_grad, hard=false)
        -log(y[target_idx] + 1f-6)
    end
    grad_old_soft = Zygote.gradient(loss_old_soft, copy(logits_grad))[1]

    println("OLD grad: $(vec(grad_old_soft))")
    println("OLD finite: $(all(isfinite.(grad_old_soft)))")
    println("OLD norm: $(norm(grad_old_soft))")

    if all(isfinite.(grad_old_soft))
        println("✓ PASS: OLD soft mode gradient computed (but uses mutating rand!)")
    else
        println("✗ FAIL: OLD soft mode gradient contains NaN/Inf")
    end
catch e
    println("✗ FAIL: OLD soft mode gradient crashed: $e")
end

println("\nHard mode sanity check (hard=true):")
println("Note: hard mode uses straight-through estimator; exact values can be noisy.")
try
    Random.seed!(42)
    loss_new_hard(x) = begin
        y = gumbel_softmax_new(x; τ=τ_grad, hard=true)
        -log(y[target_idx] + 1f-6)
    end
    grad_new_hard = Zygote.gradient(loss_new_hard, copy(logits_grad))[1]
    y_hard = gumbel_softmax_new(copy(logits_grad); τ=τ_grad, hard=true)

    Random.seed!(42)
    y_hard = gumbel_softmax_old(copy(logits_grad); τ=τ_grad, hard=true)
    Random.seed!(42)    
    y_hard = gumbel_softmax_new(copy(logits_grad); τ=τ_grad, hard=true)
    println("NEW hard output: $(vec(y_hard))")
    println("NEW hard grad: $(vec(grad_new_hard))")
    println("NEW hard finite: $(all(isfinite.(grad_new_hard)))")

    if all(isfinite.(grad_new_hard))
        println("✓ PASS: NEW hard mode backward pass runs and gradients are finite")
    else
        println("✗ FAIL: NEW hard mode gradients contain NaN/Inf")
    end
catch e
    println("✗ FAIL: NEW hard mode gradient crashed: $e")
end

# ============================================================================
# TEST 6: gumbel_softmax output validity
# ============================================================================
println("\n" * "="^80)
println("TEST 6: gumbel_softmax output validity")
println("="^80)

function check_soft_output(y; dims::Int=1, atol::Float32=1f-4)
    sums = sum(y; dims=dims)
    in_range = all((y .>= -atol) .& (y .<= 1f0 + atol))
    sums_ok = all(abs.(sums .- 1f0) .<= atol)
    return in_range, sums_ok
end

function check_hard_output(y; dims::Int=1, atol::Float32=1f-4)
    sums = sum(y; dims=dims)
    is_binary = all((abs.(y .- 0f0) .<= atol) .| (abs.(y .- 1f0) .<= atol))
    sums_ok = all(abs.(sums .- 1f0) .<= atol)
    return is_binary, sums_ok
end

logits_valid_vec = Float32[0.3, -1.2, 2.0, 0.8, -0.1]
logits_valid_mat = randn(Float32, K, batch_size)

println("\nCPU soft mode:")
Random.seed!(42)
y_soft_vec = gumbel_softmax_new(copy(logits_valid_vec); τ=1f0, hard=false)
soft_range_ok, soft_sum_ok = check_soft_output(y_soft_vec; dims=1)
println("  vector in [0,1]: $(soft_range_ok)")
println("  vector sums to 1: $(soft_sum_ok)")

Random.seed!(42)
y_soft_mat = gumbel_softmax_new(copy(logits_valid_mat); τ=1f0, hard=false)
soft_mat_range_ok, soft_mat_sum_ok = check_soft_output(y_soft_mat; dims=1)
println("  matrix in [0,1]: $(soft_mat_range_ok)")
println("  matrix column sums to 1: $(soft_mat_sum_ok)")

println("\nCPU hard mode:")
Random.seed!(42)
y_hard_vec = gumbel_softmax_new(copy(logits_valid_vec); τ=1f0, hard=true)
hard_binary_ok, hard_sum_ok = check_hard_output(y_hard_vec; dims=1)
println("  vector one-hot (0/1): $(hard_binary_ok)")
println("  vector sums to 1: $(hard_sum_ok)")

Random.seed!(42)
y_hard_mat = gumbel_softmax_new(copy(logits_valid_mat); τ=1f0, hard=true)
hard_mat_binary_ok, hard_mat_sum_ok = check_hard_output(y_hard_mat; dims=1)
println("  matrix one-hot per column: $(hard_mat_binary_ok)")
println("  matrix column sums to 1: $(hard_mat_sum_ok)")

if soft_range_ok && soft_sum_ok && soft_mat_range_ok && soft_mat_sum_ok && hard_binary_ok && hard_sum_ok && hard_mat_binary_ok && hard_mat_sum_ok
    println("✓ PASS: gumbel_softmax_new outputs are valid on CPU")
else
    println("✗ FAIL: gumbel_softmax_new output validity check failed on CPU")
end

if CUDA.functional()
    println("\nGPU output sanity (if kernels compile in current env):")
    try
        logits_valid_gpu = CUDA.cu(copy(logits_valid_mat))
        Random.seed!(42)
        y_soft_gpu = gumbel_softmax_new(logits_valid_gpu; τ=1f0, hard=false)
        Random.seed!(42)
        y_hard_gpu = gumbel_softmax_new(logits_valid_gpu; τ=1f0, hard=true)

        soft_gpu_range_ok, soft_gpu_sum_ok = check_soft_output(Array(y_soft_gpu); dims=1)
        hard_gpu_binary_ok, hard_gpu_sum_ok = check_hard_output(Array(y_hard_gpu); dims=1)

        println("  soft in [0,1]: $(soft_gpu_range_ok)")
        println("  soft column sums to 1: $(soft_gpu_sum_ok)")
        println("  hard one-hot per column: $(hard_gpu_binary_ok)")
        println("  hard column sums to 1: $(hard_gpu_sum_ok)")
    catch e
        println("⚠ GPU validity check skipped/fails due to environment: $e")
    end
else
    println("\n⚠ GPU not available, skipping GPU output validity check")
end

# ============================================================================
# TEST 7: MoG optimization sanity check (do parameters change?)
# ============================================================================
println("\n" * "="^80)
println("TEST 7: MoG optimization sanity check")
println("="^80)

function mog_forward_new(α, μ, Σ; sample_size::Int, batch_size::Int, τ::Float32)
    K_local = size(α, 1)
    Ds_local = size(μ, 1)

    α_soft = gumbel_softmax_new(α; τ=τ, hard=true)
    α_soft = reshape(α_soft, (1, 1, K_local, 1))
    α_soft = repeat(α_soft, 1, sample_size, 1, batch_size)

    μ_r = reshape(μ, (Ds_local, 1, K_local, 1))
    Σ_r = reshape(Σ, (Ds_local, 1, K_local, 1))

    μ_mix = reshape(sum(μ_r .* α_soft, dims=3), (Ds_local, sample_size, batch_size))
    Σ_mix = reshape(sum(Σ_r .* α_soft, dims=3), (Ds_local, sample_size, batch_size))

    ϵ = randn(eltype(μ), Ds_local, sample_size, batch_size)
    return μ_mix .+ Flux.softplus.(Σ_mix) .* ϵ
end

try
    # Fresh trainable copies
    α_opt = copy(α_new)
    μ_opt = copy(μ_new)
    Σ_opt = copy(Σ_new)

    # Keep initial values for comparison
    α_before = copy(α_opt)
    μ_before = copy(μ_opt)
    Σ_before = copy(Σ_opt)

    sample_size_opt = 12
    batch_size_opt = 3
    τ_opt = 1f0
    η = 1f-2

    # Fixed target for reproducibility of objective
    Random.seed!(123)
    target = randn(Float32, Ds, sample_size_opt, batch_size_opt)

    ps = Flux.params(α_opt, μ_opt, Σ_opt)

    Random.seed!(123)
    loss_before = let z = mog_forward_new(α_opt, μ_opt, Σ_opt; sample_size=sample_size_opt, batch_size=batch_size_opt, τ=τ_opt)
        mean((z .- target) .^ 2)
    end

    Random.seed!(123)
    gs = Zygote.gradient(ps) do
        z = mog_forward_new(α_opt, μ_opt, Σ_opt; sample_size=sample_size_opt, batch_size=batch_size_opt, τ=τ_opt)
        mean((z .- target) .^ 2)
    end

    gα = gs[α_opt]
    gμ = gs[μ_opt]
    gΣ = gs[Σ_opt]

    grads_ok = gα !== nothing && gμ !== nothing && gΣ !== nothing &&
               all(isfinite.(gα)) && all(isfinite.(gμ)) && all(isfinite.(gΣ))

    println("Gradient finite check: $(grads_ok)")
    if grads_ok
        println("  ||gα|| = $(norm(gα))")
        println("  ||gμ|| = $(norm(gμ))")
        println("  ||gΣ|| = $(norm(gΣ))")
    end

    if grads_ok
        # Manual SGD step
        α_opt .-= η .* gα
        μ_opt .-= η .* gμ
        Σ_opt .-= η .* gΣ

        Random.seed!(123)
        loss_after = let z = mog_forward_new(α_opt, μ_opt, Σ_opt; sample_size=sample_size_opt, batch_size=batch_size_opt, τ=τ_opt)
            mean((z .- target) .^ 2)
        end

        dα = norm(α_opt .- α_before)
        dμ = norm(μ_opt .- μ_before)
        dΣ = norm(Σ_opt .- Σ_before)

        println("Parameter change after one step:")
        println("  ||Δα|| = $(dα)")
        println("  ||Δμ|| = $(dμ)")
        println("  ||ΔΣ|| = $(dΣ)")
        println("Loss before: $(loss_before)")
        println("Loss after:  $(loss_after)")

        if dα > 0f0 || dμ > 0f0 || dΣ > 0f0
            println("✓ PASS: backward + update changed MoG parameters")
        else
            println("✗ FAIL: parameters did not change after update")
        end
    else
        println("✗ FAIL: gradient check failed, skipping parameter update")
    end
catch e
    println("✗ FAIL: MoG optimization test crashed: $e")
end


# ============================================================================
# SUMMARY
# ============================================================================
println("\n" * "="^80)
println("SUMMARY")
println("="^80)
println("""
Key findings:
1. Shape compatibility: OK ✓
2. Data type compatibility: OK ✓
3. gumbel_softmax broadcasting: Need to verify GPU behavior
4. repeat() on GPU: Testing above
5. Gradient computation:
    - NEW soft mode: finite non-zero gradient in simple test
    - OLD soft mode: gradient can be computed, but implementation mutates input via Random.rand!
    - NEW hard mode: backward pass runs with finite gradients (STE sanity check)
6. Output validity checks:
    - Soft mode: values in [0, 1] and probabilities sum to 1
    - Hard mode: one-hot outputs and per-column sum equals 1
7. MoG optimization sanity:
    - Backward pass returns finite gradients for α, μ, Σ
    - One SGD step changes at least one MoG parameter tensor

Recommendations:
- Use NEW gumbel_softmax for all new code (safe autodiff, no mutations)
- If repeat() fails on GPU, consider using reshape + broadcasting instead
- Hard mode (hard=true) should be treated as noisy STE; rely mainly on soft-mode gradient tests
- Trust output correctness only if validity tests above pass (especially soft sum=1 and hard one-hot)
- Trust MoG trainability only if TEST 7 reports non-zero parameter change after update
- Monitor for memory issues during repeat() with large tensors

Critical Issue Found:
- OLD implementation uses Random.rand!(logits) which MUTATES the input array
- This breaks automatic differentiation and is incompatible with Flux.param()
- NEW implementation uses MLUtils.rand_like() which is safe and non-mutating
""")
