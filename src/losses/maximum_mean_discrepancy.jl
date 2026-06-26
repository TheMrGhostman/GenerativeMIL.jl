_SigmaArg = Union{Real, AbstractVector{<:Real}}

"""
`_apply_rbf_kernel(d::AbstractArray{T}, sigma::Real) where T<:AbstractFloat`

Apply a fixed-bandwidth RBF kernel to squared distances.

Applies the Gaussian RBF kernel: ``\\exp(-d / (2\\sigma^2))`` element-wise.

Arguments (positional):
- `d::AbstractArray{T}`: Pairwise squared distance matrix or batched tensor (any shape).
- `sigma::Real`: Kernel bandwidth parameter (positive).

Returns:
- `AbstractArray{T}` with same shape as `d` containing RBF kernel values in (0, 1].

Notes:
- Higher `sigma` → softer (wider) kernel, lower `sigma` → sharper kernel.
- Element-wise operation: `exp(-d[i,j,b] / (2*sigma^2))`.
"""
function _apply_rbf_kernel(d::AbstractArray{T}, sigma::Real) where T<:AbstractFloat
    inv_two_sigma2 = inv(T(2) * T(sigma)^2)
    return exp.(-d .* inv_two_sigma2)
end

"""
`_apply_rbf_kernel(d::AbstractArray{T}, sigma::AbstractVector{<:Real}) where T<:AbstractFloat`

Apply multi-scale RBF kernels to squared distances and average.

Computes multiple RBF kernels with different bandwidths and returns their average.
Useful for bandwidth selection without explicit tuning.

Arguments (positional):
- `d::AbstractArray{T}`: Pairwise squared distance matrix or batched tensor (any shape).
- `sigma::AbstractVector{<:Real}`: Non-empty collection of kernel bandwidth parameters (each positive).

Returns:
- `AbstractArray{T}` with same shape as `d` containing averaged RBF kernel values.

Notes:
- Multi-scale approach: averages kernels across all provided bandwidths.
- Throws `AssertionError` if `sigma` is empty.
- Example: `sigma = [0.5, 1.0, 2.0]` creates a robust multi-scale measure.
"""
function _apply_rbf_kernel(d::AbstractArray{T}, sigma::AbstractVector{<:Real}) where T<:AbstractFloat
    @assert !isempty(sigma) "sigma vector must not be empty"
    k_terms = map(s -> _apply_rbf_kernel(d, s), sigma)
    return reduce(+, k_terms) ./ T(length(sigma))
end

"""
`_pairwise_sqdist(x::AbstractMatrix{T}, y::AbstractMatrix{T}) where T<:AbstractFloat`

Compute pairwise squared Euclidean distances between column vectors.

Efficiently computes ``D[i,j] = \\|x_i - y_j\\|^2`` using the formula:
``D[i,j] = \\|x_i\\|^2 + \\|y_j\\|^2 - 2 \\langle x_i, y_j \\rangle``

Arguments (positional):
- `x::AbstractMatrix{T}`: Matrix of samples with shape `(d, n)` (feature_dim × num_samples).
- `y::AbstractMatrix{T}`: Matrix of samples with shape `(d, m)`.

Returns:
- Matrix of shape `(n, m)` with `[i, j]` = squared distance from `x[:, i]` to `y[:, j]`.

Notes:
- Result is guaranteed non-negative (clipped at zero for numerical safety).
- Efficient: uses vector norms and matrix multiplication rather than explicit loops.
- CPU-friendly implementation; for GPU, see `_pairwise_sqdist_batched(...::CuArray, ...)`.
"""
function _pairwise_sqdist(x::AbstractMatrix{T}, y::AbstractMatrix{T}) where T<:AbstractFloat
    x2 = sum(abs2, x; dims=1)
    y2 = sum(abs2, y; dims=1)
    return max.(x2' .+ y2 .- T(2) .* (x' * y), zero(T))
end

"""
`_pairwise_sqdist_batched(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}) where T<:AbstractFloat`

Compute pairwise squared Euclidean distances for batched arrays (CPU fallback).

Applies `_pairwise_sqdist` independently to each batch slice without fusing operations.
Maintains Zygote differentiability via array slicing and concatenation.

Arguments (positional):
- `x::AbstractArray{T, 3}`: Batched samples with shape `(d, n, bs)` (feature_dim × num_samples × batch_size).
- `y::AbstractArray{T, 3}`: Batched samples with shape `(d, m, bs)`.

Returns:
- `AbstractArray{T, 3}` of shape `(n, m, bs)` with distance matrices per batch slice.

Constraints:
- `x` and `y` must have matching batch sizes.

Notes:
- Loop-based implementation: Zygote-friendly, no fused matrix operations.
- For GPU arrays, the CuArray specialization uses `Flux.batched_mul` for performance.
"""
function _pairwise_sqdist_batched(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}) where T<:AbstractFloat
    @assert size(x, 3) == size(y, 3) "x and y must have the same batch size"
    bs = size(x, 3)

    # Avoid in-place writes so the function remains differentiable by Zygote.
    d_slices = [_pairwise_sqdist(@view(x[:, :, b]), @view(y[:, :, b])) for b in 1:bs]
    return cat(d_slices...; dims=3)
end

"""
`_pairwise_sqdist_batched(x::CuArray{T, 3}, y::CuArray{T, 3}) where T<:AbstractFloat`

Compute pairwise squared Euclidean distances for batched CuArrays (GPU-optimized).

Uses fused batched GEMM operations via `Flux.batched_mul` for GPU efficiency.
Applies norm computations and matrix multiplications across entire batch in parallel.

Arguments (positional):
- `x::CuArray{T, 3}`: Batched GPU samples with shape `(d, n, bs)`.
- `y::CuArray{T, 3}`: Batched GPU samples with shape `(d, m, bs)`.

Returns:
- `CuArray{T, 3}` of shape `(n, m, bs)` with distance matrices per batch slice.

Constraints:
- `x` and `y` must have matching batch sizes.

Notes:
- GPU-specialized: uses `Flux.batched_mul` for fused matrix products.
- Much faster than CPU fallback for large batch sizes or high-dimensional data.
"""
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

"""
`_diag_sum_batched(a::AbstractArray{T, 3}) where T<:AbstractFloat`

Sum all diagonal entries across batch slices of square matrices (private utility).

Computes: ``\\sum_{b=1}^{bs} \\sum_{i=1}^{n} a[i,i,b]`` for shape `(n, n, bs)`.

Arguments (positional):
- `a::AbstractArray{T, 3}`: Batched square matrices with shape `(n, n, bs)`.

Returns:
- Scalar: sum of all diagonal elements across all batches.

Constraints:
- Throws `AssertionError` if matrices are not square.

Notes:
- Loop-based implementation for correctness; could be optimized.
- Used in MMD computation to remove diagonal (biased) kernel estimates.
"""
function _diag_sum_batched(a::AbstractArray{T, 3}) where T<:AbstractFloat
    @assert size(a, 1) == size(a, 2) "Diagonal sum expects square matrices per batch"
    bs = size(a, 3)
    s = zero(T)
    for b in 1:bs
        s += sum(@view a[:, :, b][diagind(@view a[:, :, b])])
    end
    return s
end

"""
`maximum_mean_discrepancy(x::AbstractMatrix{T}, y::AbstractMatrix{T}; sigma::_SigmaArg=1, kernel::Union{Nothing, Function}=nothing, distance_kernel::Union{Nothing, Function}=nothing) where T<:AbstractFloat`

Compute unbiased Maximum Mean Discrepancy (MMD) between two sample matrices.

MMD is a kernel-based divergence measure: ``\\text{MMD}(x,y) = \\|\\mu_x - \\mu_y\\|_{H}^2`` where ``\\mu``
are mean embeddings in a RKHS. The unbiased U-statistic estimator removes diagonal bias.

Arguments (positional):
- `x::AbstractMatrix{T}`: First sample matrix with shape `(d, m)` (feature_dim × num_samples).
- `y::AbstractMatrix{T}`: Second sample matrix with shape `(d, n)`.

Keyword arguments:
- `sigma::_SigmaArg`: RBF kernel bandwidth (default `1`). Can be scalar or vector of scalars.
- `kernel::Union{Nothing, Function}`: Optional kernel function `(x, y) → matrix` (default `nothing`). Mutually exclusive with `distance_kernel`.
- `distance_kernel::Union{Nothing, Function}`: Optional kernel function `d → matrix` applied to squared distances (default `nothing`).

Returns:
- Scalar MMD estimate (non-negative).

Constraints:
- Both samples must have at least 2 points each (for unbiased estimator).
- Cannot specify both `kernel` and `distance_kernel`.

Notes:
- Unbiased U-statistic: 
```math
\\text{MMD}_u = \\frac{1}{m(m-1)} \\sum_{i \\neq j} k(x_i, x_j) + \\frac{1}{n(n-1)} \\sum_{i \\neq j} k(y_i, y_j) - \\frac{2}{mn} \\sum_i \\sum_j k(x_i, y_j)
```
- Default kernel: multi-scale RBF with bandwidths from `sigma`.
- Example: `maximum_mean_discrepancy(x, y; sigma=[0.5, 1.0, 2.0])` for robust multi-scale MMD.
"""
function maximum_mean_discrepancy(
    x::AbstractMatrix{T},
    y::AbstractMatrix{T};
    sigma::_SigmaArg = 1,
    kernel::Union{Nothing, Function} = nothing,
    distance_kernel::Union{Nothing, Function} = nothing,
) where T<:AbstractFloat
    m = size(x, 2)
    n = size(y, 2)
    @assert m > 1 "MMD requires at least 2 samples in x for unbiased estimator"
    @assert n > 1 "MMD requires at least 2 samples in y for unbiased estimator"
    @assert !(kernel !== nothing && distance_kernel !== nothing) "Use either kernel or distance_kernel, not both"

    if kernel !== nothing
        k_xx = kernel(x, x)
        k_yy = kernel(y, y)
        k_xy = kernel(x, y)
    else
        d_xx = _pairwise_sqdist(x, x)
        d_yy = _pairwise_sqdist(y, y)
        d_xy = _pairwise_sqdist(x, y)

        if distance_kernel === nothing
            k_xx = _apply_rbf_kernel(d_xx, sigma)
            k_yy = _apply_rbf_kernel(d_yy, sigma)
            k_xy = _apply_rbf_kernel(d_xy, sigma)
        else
            k_xx = distance_kernel(d_xx)
            k_yy = distance_kernel(d_yy)
            k_xy = distance_kernel(d_xy)
        end
    end

    diag_xx = sum(@view k_xx[diagind(k_xx)])
    diag_yy = sum(@view k_yy[diagind(k_yy)])

    return (sum(k_xx) - diag_xx) / (T(m) * T(m - 1)) +
           (sum(k_yy) - diag_yy) / (T(n) * T(n - 1)) -
           T(2) * sum(k_xy) / (T(m) * T(n))
end

"""
`maximum_mean_discrepancy(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}; sigma::_SigmaArg=1, kernel::Union{Nothing, Function}=nothing, distance_kernel::Union{Nothing, Function}=nothing) where T<:AbstractFloat`

Compute unbiased MMD for batched tensors (averaged across batch).

Applies the 2D MMD formula to each batch slice independently and returns the per-batch mean.
Supports all kernel specifications from the 2D variant.

Arguments (positional):
- `x::AbstractArray{T, 3}`: Batched samples with shape `(d, m, bs)` (feature_dim × num_samples × batch_size).
- `y::AbstractArray{T, 3}`: Batched samples with shape `(d, n, bs)`.

Keyword arguments:
- `sigma::_SigmaArg`: RBF kernel bandwidth (default `1`). Can be scalar or vector of scalars.
- `kernel::Union{Nothing, Function}`: Optional kernel function (default `nothing`). Falls back to CPU looping for custom kernels.
- `distance_kernel::Union{Nothing, Function}`: Optional distance-based kernel function (default `nothing`).

Returns:
- Scalar MMD estimate averaged over all batch slices.

Constraints:
- Both tensors must have same batch size.
- Each batch slice must have at least 2 samples.
- Cannot specify both `kernel` and `distance_kernel`.

Notes:
- Batched computation: ``\\text{MMD}_\\text{avg}(x,y) = \\frac{1}{bs} \\sum_{b=1}^{bs} \\text{MMD}(x[:,:,b], y[:,:,b])``
- GPU-accelerated distance computation via `_pairwise_sqdist_batched` specialization for CuArrays.
- Fully tensorized for kernel types other than custom `Function`.
"""
function maximum_mean_discrepancy(
    x::AbstractArray{T, 3},
    y::AbstractArray{T, 3};
    sigma::_SigmaArg = 1,
    kernel::Union{Nothing, Function} = nothing,
    distance_kernel::Union{Nothing, Function} = nothing,
) where T<:AbstractFloat
    @assert size(x, 3) == size(y, 3) "x and y must have the same batch size"
    m = size(x, 2)
    n = size(y, 2)
    bs = size(x, 3)
    @assert m > 1 "MMD requires at least 2 samples in x for unbiased estimator"
    @assert n > 1 "MMD requires at least 2 samples in y for unbiased estimator"
    @assert !(kernel !== nothing && distance_kernel !== nothing) "Use either kernel or distance_kernel, not both"

    if kernel !== nothing
        acc = zero(T)
        for b in 1:bs
            acc += maximum_mean_discrepancy(@view(x[:, :, b]), @view(y[:, :, b]); sigma=sigma, kernel=kernel)
        end
        return acc / T(bs)
    end

    d_xx = _pairwise_sqdist_batched(x, x)
    d_yy = _pairwise_sqdist_batched(y, y)
    d_xy = _pairwise_sqdist_batched(x, y)

    if distance_kernel === nothing
        k_xx = _apply_rbf_kernel(d_xx, sigma)
        k_yy = _apply_rbf_kernel(d_yy, sigma)
        k_xy = _apply_rbf_kernel(d_xy, sigma)
    else
        k_xx = distance_kernel(d_xx)
        k_yy = distance_kernel(d_yy)
        k_xy = distance_kernel(d_xy)
    end

    sum_xx_offdiag = sum(k_xx) - _diag_sum_batched(k_xx)
    sum_yy_offdiag = sum(k_yy) - _diag_sum_batched(k_yy)
    sum_xy = sum(k_xy)

    return sum_xx_offdiag / (T(m) * T(m - 1) * T(bs)) +
           sum_yy_offdiag / (T(n) * T(n - 1) * T(bs)) -
           T(2) * sum_xy / (T(m) * T(n) * T(bs))
end