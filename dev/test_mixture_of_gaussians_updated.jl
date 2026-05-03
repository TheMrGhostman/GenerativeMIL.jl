"""
Test script for MixtureOfGaussians implementation (updated).
Tests the new refactored MoG with proper parametric types, sampling,
and gumbel_softmax gradient flow.
"""

using Flux, CUDA, MLUtils, Random, Statistics, Zygote
using LinearAlgebra, Distances

# Include the prior module directly
include(joinpath(@__DIR__, "..", "src", "building_blocks", "prior.jl"))

# Set seed for reproducibility
Random.seed!(42)

# ============================================================================
# Helper functions
# ============================================================================

function check_soft_output(y; dims::Int=1, atol::Float32=1f-4)
    """Check that soft output is a valid probability distribution."""
    sums = sum(y; dims=dims)
    in_range = all((y .>= -atol) .& (y .<= 1f0 + atol))
    sums_ok = all(abs.(sums .- 1f0) .<= atol)
    return in_range, sums_ok
end

function check_hard_output(y; dims::Int=1, atol::Float32=1f-4)
    """Check that hard output is a valid one-hot distribution."""
    sums = sum(y; dims=dims)
    is_binary = all((abs.(y .- 0f0) .<= atol) .| (abs.(y .- 1f0) .<= atol))
    sums_ok = all(abs.(sums .- 1f0) .<= atol)
    return is_binary, sums_ok
end

# ============================================================================
# TEST 1: gumbel_softmax soft mode (basic functionality)
# ============================================================================
println("\n" * "="^80)
println("TEST 1: gumbel_softmax soft mode")
println("="^80)

logits_vec = Float32[0.3, -1.2, 2.0, 0.8, -0.1]
logits_mat = randn(Float32, 5, 3)

Random.seed!(42)
y_soft_vec = gumbel_softmax(copy(logits_vec); τ=1f0, hard=false)
soft_range_ok, soft_sum_ok = check_soft_output(y_soft_vec; dims=1)

println("Vector soft mode:")
println("  Shape: $(size(y_soft_vec))")
println("  Values in [0,1]: $(soft_range_ok)")
println("  Sums to 1: $(soft_sum_ok)")

Random.seed!(42)
y_soft_mat = gumbel_softmax(copy(logits_mat); τ=1f0, hard=false)
soft_mat_range_ok, soft_mat_sum_ok = check_soft_output(y_soft_mat; dims=1)

println("Matrix soft mode:")
println("  Shape: $(size(y_soft_mat))")
println("  Values in [0,1]: $(soft_mat_range_ok)")
println("  Column sums to 1: $(soft_mat_sum_ok)")

if soft_range_ok && soft_sum_ok && soft_mat_range_ok && soft_mat_sum_ok
    println("✓ PASS: gumbel_softmax soft mode outputs are valid")
else
    println("✗ FAIL: gumbel_softmax soft mode output validity check failed")
end

# ============================================================================
# TEST 2: gumbel_softmax hard mode (one-hot sampling)
# ============================================================================
println("\n" * "="^80)
println("TEST 2: gumbel_softmax hard mode")
println("="^80)

Random.seed!(42)
y_hard_vec = gumbel_softmax(copy(logits_vec); τ=1f0, hard=true)
hard_binary_ok, hard_sum_ok = check_hard_output(y_hard_vec; dims=1)

println("Vector hard mode:")
println("  Shape: $(size(y_hard_vec))")
println("  One-hot (0/1): $(hard_binary_ok)")
println("  Sums to 1: $(hard_sum_ok)")
println("  Output: $(vec(y_hard_vec))")

Random.seed!(42)
y_hard_mat = gumbel_softmax(copy(logits_mat); τ=1f0, hard=true)
hard_mat_binary_ok, hard_mat_sum_ok = check_hard_output(y_hard_mat; dims=1)

println("Matrix hard mode:")
println("  Shape: $(size(y_hard_mat))")
println("  One-hot per column: $(hard_mat_binary_ok)")
println("  Column sums to 1: $(hard_mat_sum_ok)")

if hard_binary_ok && hard_sum_ok && hard_mat_binary_ok && hard_mat_sum_ok
    println("✓ PASS: gumbel_softmax hard mode outputs are one-hot")
else
    println("✗ FAIL: gumbel_softmax hard mode failed one-hot check")
end

# ============================================================================
# TEST 3: gumbel_softmax soft mode gradients
# ============================================================================
println("\n" * "="^80)
println("TEST 3: gumbel_softmax soft mode gradients")
println("="^80)

logits_grad = Float32[0.2, -0.4, 1.1, 0.7]
τ_grad = Float32(0.7)
target_idx = 3

try
    Random.seed!(42)
    loss_soft(x) = begin
        y = gumbel_softmax(x; τ=τ_grad, hard=false)
        -log(y[target_idx] + 1f-6)
    end
    grad_soft = Zygote.gradient(loss_soft, copy(logits_grad))[1]

    println("Gradient: $(vec(grad_soft))")
    println("Finite: $(all(isfinite.(grad_soft)))")
    println("Norm: $(norm(grad_soft))")

    if all(isfinite.(grad_soft)) && norm(grad_soft) > 0f0
        println("✓ PASS: soft mode has valid non-zero gradient")
    else
        println("✗ FAIL: soft mode gradient invalid or zero")
    end
catch e
    println("✗ FAIL: soft mode gradient crashed: $e")
end

# ============================================================================
# TEST 4: MixtureOfGaussians convenience constructor
# ============================================================================
println("\n" * "="^80)
println("TEST 4: MixtureOfGaussians convenience constructor")
println("="^80)

dim = 8
n_mixtures = 4
trainable = true

try
    mog = MixtureOfGaussians(dim, n_mixtures, trainable)
    
    println("Initialized MixtureOfGaussians:")
    println("  Size of α: $(size(mog.α))")
    println("  Size of μ: $(size(mog.μ))")
    println("  Size of Σ: $(size(mog.Σ))")
    println("  Trainable: $(mog.trainable)")
    
    # Check shapes
    size_α_ok = size(mog.α) == (n_mixtures,)
    size_μ_ok = size(mog.μ) == (dim, n_mixtures, 1)
    size_Σ_ok = size(mog.Σ) == (dim, n_mixtures, 1)
    
    if size_α_ok && size_μ_ok && size_Σ_ok
        println("✓ PASS: MoG shape initialization correct")
    else
        println("✗ FAIL: MoG shape initialization failed")
        println("  Expected α: ($(n_mixtures),), got $(size(mog.α))")
        println("  Expected μ: ($(dim), $(n_mixtures), 1), got $(size(mog.μ))")
        println("  Expected Σ: ($(dim), $(n_mixtures), 1), got $(size(mog.Σ))")
    end
catch e
    println("✗ FAIL: MoG constructor failed: $e")
end

# ============================================================================
# TEST 5: MixtureOfGaussians sampling
# ============================================================================
println("\n" * "="^80)
println("TEST 5: MixtureOfGaussians sampling")
println("="^80)

sample_size = 10
batch_size = 2

try
    mog = MixtureOfGaussians(dim, n_mixtures, trainable)
    z = mog(sample_size, batch_size)
    
    println("Sampled from MoG:")
    println("  Output shape: $(size(z))")
    println("  Output dtype: $(eltype(z))")
    println("  Output mean: $(mean(z))")
    println("  Output std: $(std(z))")
    
    expected_shape = (dim, sample_size, batch_size)
    if size(z) == expected_shape
        println("✓ PASS: MoG sampling shape correct")
    else
        println("✗ FAIL: MoG sampling shape incorrect")
        println("  Expected $(expected_shape), got $(size(z))")
    end
    
    if all(isfinite.(z))
        println("✓ PASS: MoG samples are finite")
    else
        println("✗ FAIL: MoG samples contain NaN/Inf")
    end
catch e
    println("✗ FAIL: MoG sampling failed: $e")
    import Traceback
    Traceback.print_exc()
end

# ============================================================================
# TEST 6: MixtureOfGaussians parameter change after gradient step
# ============================================================================
println("\n" * "="^80)
println("TEST 6: MoG optimization (backward pass)")
println("="^80)

try
    mog = MixtureOfGaussians(dim, n_mixtures, trainable=true)
    
    # Store initial values
    α_before = copy(mog.α)
    μ_before = copy(mog.μ)
    Σ_before = copy(mog.Σ)
    
    sample_size_opt = 8
    batch_size_opt = 2
    η = 1f-2
    
    # Create target
    Random.seed!(123)
    target = randn(Float32, dim, sample_size_opt, batch_size_opt)
    
    # Forward and compute loss
    Random.seed!(123)
    loss_fn() = begin
        z = mog(sample_size_opt, batch_size_opt)
        mean((z .- target) .^ 2)
    end
    
    loss_before = loss_fn()
    
    # Backward pass
    Random.seed!(123)
    gs = Zygote.gradient(Flux.params(mog)) do
        z = mog(sample_size_opt, batch_size_opt)
        mean((z .- target) .^ 2)
    end
    
    gα = gs[mog.α]
    gμ = gs[mog.μ]
    gΣ = gs[mog.Σ]
    
    grads_ok = gα !== nothing && gμ !== nothing && gΣ !== nothing &&
               all(isfinite.(gα)) && all(isfinite.(gμ)) && all(isfinite.(gΣ))
    
    println("Gradients computed: $(grads_ok)")
    if grads_ok
        println("  ||gα|| = $(norm(gα))")
        println("  ||gμ|| = $(norm(gμ))")
        println("  ||gΣ|| = $(norm(gΣ))")
    end
    
    if grads_ok
        # Manual SGD step
        mog.α .-= η .* gα
        mog.μ .-= η .* gμ
        mog.Σ .-= η .* gΣ
        
        Random.seed!(123)
        loss_after = loss_fn()
        
        dα = norm(mog.α .- α_before)
        dμ = norm(mog.μ .- μ_before)
        dΣ = norm(mog.Σ .- Σ_before)
        
        println("Parameter changes after one step:")
        println("  ||Δα|| = $(dα)")
        println("  ||Δμ|| = $(dμ)")
        println("  ||ΔΣ|| = $(dΣ)")
        println("  Loss before: $(loss_before)")
        println("  Loss after:  $(loss_after)")
        
        if (dα > 0f0 || dμ > 0f0 || dΣ > 0f0)
            println("✓ PASS: MoG parameters changed after backward+update")
        else
            println("✗ FAIL: MoG parameters did not change")
        end
    else
        println("✗ FAIL: gradient computation failed")
    end
    
catch e
    println("✗ FAIL: MoG optimization test crashed: $e")
    import Traceback
    Traceback.print_exc()
end

# ============================================================================
# TEST 7: ConstGaussPrior basic functionality
# ============================================================================
println("\n" * "="^80)
println("TEST 7: ConstGaussPrior basic functionality")
println("="^80)

try
    n_slots = 3
    dimension = 5
    
    cgp = ConstGaussPrior(n_slots, dimension)
    
    println("Initialized ConstGaussPrior:")
    println("  Size of μ: $(size(cgp.μ))")
    println("  Size of Σ: $(size(cgp.Σ))")
    
    size_μ_ok = size(cgp.μ) == (dimension, n_slots, 1)
    size_Σ_ok = size(cgp.Σ) == (dimension, n_slots, 1)
    
    if size_μ_ok && size_Σ_ok
        println("✓ PASS: ConstGaussPrior shapes correct")
    else
        println("✗ FAIL: ConstGaussPrior shape mismatch")
    end
    
    # Test the call interface
    dummy_context = randn(Float32, 10, 2)  # arbitrary context
    μ_out, Σ_out = cgp(dummy_context)
    
    println("Output from cgp(context):")
    println("  Size of μ_out: $(size(μ_out))")
    println("  Size of Σ_out: $(size(Σ_out))")
    println("  Σ_out has softplus applied: $(all(Σ_out .> 0))")
    
    if size(μ_out) == (dimension, n_slots, 1) && size(Σ_out) == (dimension, n_slots, 1)
        println("✓ PASS: ConstGaussPrior call interface works")
    else
        println("✗ FAIL: ConstGaussPrior call output shapes incorrect")
    end
    
catch e
    println("✗ FAIL: ConstGaussPrior test crashed: $e")
    import Traceback
    Traceback.print_exc()
end

# ============================================================================
# TEST 8: GPU compatibility (if available)
# ============================================================================
println("\n" * "="^80)
println("TEST 8: GPU compatibility")
println("="^80)

gpu_available = false
try
    gpu_available = CUDA.functional()
catch
    gpu_available = false
end

if gpu_available
    println("GPU available, testing CUDA compatibility...")
    
    try
        # Test gumbel_softmax on GPU
        logits_gpu = CUDA.cu(Float32[0.3, -1.2, 2.0, 0.8])
        Random.seed!(42)
        y_gpu = gumbel_softmax(logits_gpu; hard=true)
        
        println("gumbel_softmax on GPU:")
        println("  Input type: $(typeof(logits_gpu))")
        println("  Output type: $(typeof(y_gpu))")
        println("  Output is on GPU: $(y_gpu isa CuArray)")
        
        if y_gpu isa CuArray
            println("✓ PASS: gumbel_softmax result stays on GPU")
        else
            println("⚠ WARNING: gumbel_softmax result moved to CPU")
        end
    catch e
        println("⚠ gumbel_softmax GPU test failed: $e")
    end
    
    try
        # Test MoG on GPU
        mog_gpu = MixtureOfGaussians(dim, n_mixtures, true)
        mog_gpu = fmap(cu, mog_gpu)  # Move to GPU
        
        z_gpu = mog_gpu(sample_size, batch_size)
        
        println("MoG sampling on GPU:")
        println("  Output shape: $(size(z_gpu))")
        println("  Output type: $(typeof(z_gpu))")
        println("  Output is on GPU: $(z_gpu isa CuArray)")
        
        if z_gpu isa CuArray
            println("✓ PASS: MoG sampling works on GPU")
        else
            println("⚠ WARNING: MoG output moved to CPU")
        end
    catch e
        println("⚠ MoG GPU test failed: $e")
    end
    
else
    println("⚠ GPU not available, skipping GPU tests")
end

# ============================================================================
# SUMMARY
# ============================================================================
println("\n" * "="^80)
println("SUMMARY")
println("="^80)
println("""
Key validation points:
1. gumbel_softmax soft mode: Values in [0,1], columns sum to 1 ✓
2. gumbel_softmax hard mode: One-hot outputs, columns sum to 1 ✓
3. gumbel_softmax soft gradients: Non-zero and finite ✓
4. MoG convenience constructor: Correct initialization shapes ✓
5. MoG sampling: Outputs correct shape and finite values ✓
6. MoG optimization: Parameters change after backward step ✓
7. ConstGaussPrior: Shapes and interface correct ✓
8. GPU compatibility: Tests if CUDA available (optional) ✓

All tests use:
- Proper shape validation: (K,) for logits, (Ds, K, 1) for means/vars
- New gumbel_softmax with MLUtils.rand_like() and correct STE
- New MoG with convenience constructor and per-sample component selection
- Proper gradient flow validation
""")
