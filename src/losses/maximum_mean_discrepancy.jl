_SigmaArg = Union{Real, AbstractVector{<:Real}}

"""
Apply an RBF kernel to a matrix of squared distances.

Arguments
- `d::AbstractArray{T}`: Pairwise squared distance matrix or batched distance tensor.
- `sigma::Real`: Kernel bandwidth.

Returns
- An array with the same shape as `d` containing the RBF kernel values.
"""
function _apply_rbf_kernel(d::AbstractArray{T}, sigma::Real) where T<:AbstractFloat
    inv_two_sigma2 = inv(T(2) * T(sigma)^2)
    return exp.(-d .* inv_two_sigma2)
end

"""
Apply a multi-scale RBF kernel to a matrix of squared distances.

Arguments
- `d::AbstractArray{T}`: Pairwise squared distance matrix or batched distance tensor.
- `sigma::AbstractVector{<:Real}`: Non-empty collection of bandwidths to average over.

Returns
- An array with the same shape as `d` containing the mean RBF kernel value across all bandwidths.
"""
function _apply_rbf_kernel(d::AbstractArray{T}, sigma::AbstractVector{<:Real}) where T<:AbstractFloat
    @assert !isempty(sigma) "sigma vector must not be empty"
    k_terms = map(s -> _apply_rbf_kernel(d, s), sigma)
    return reduce(+, k_terms) ./ T(length(sigma))
end

"""
Compute pairwise squared Euclidean distances between the columns of two matrices.

Arguments
- `x::AbstractMatrix{T}`: Matrix of samples with shape `(d, n)`.
- `y::AbstractMatrix{T}`: Matrix of samples with shape `(d, m)`.

Returns
- A matrix of shape `(n, m)` containing squared distances between all column pairs.
"""
function _pairwise_sqdist(x::AbstractMatrix{T}, y::AbstractMatrix{T}) where T<:AbstractFloat
    x2 = sum(abs2, x; dims=1)
    y2 = sum(abs2, y; dims=1)
    return max.(x2' .+ y2 .- T(2) .* (x' * y), zero(T))
end

"""
Compute pairwise squared Euclidean distances for each batch slice of two tensors.

Arguments
- `x::AbstractArray{T, 3}`: Batched samples with shape `(d, n, bs)`.
- `y::AbstractArray{T, 3}`: Batched samples with shape `(d, m, bs)`.

Returns
- A tensor of shape `(n, m, bs)` containing squared distances per batch.
"""
function _pairwise_sqdist_batched(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}) where T<:AbstractFloat
    @assert size(x, 3) == size(y, 3) "x and y must have the same batch size"
    bs = size(x, 3)

    # Avoid in-place writes so the function remains differentiable by Zygote.
    d_slices = [_pairwise_sqdist(@view(x[:, :, b]), @view(y[:, :, b])) for b in 1:bs]
    return cat(d_slices...; dims=3)
end

"""
Compute pairwise squared Euclidean distances for batched CuArrays using a fused batched GEMM path.

Arguments
- `x::CuArray{T, 3}`: Batched samples with shape `(d, n, bs)`.
- `y::CuArray{T, 3}`: Batched samples with shape `(d, m, bs)`.

Returns
- A `CuArray` of shape `(n, m, bs)` containing squared distances per batch.
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
Sum the diagonal entries of each matrix in a batched 3D tensor.

Arguments
- `a::AbstractArray{T, 3}`: Batched square matrices with shape `(n, n, bs)`.

Returns
- The scalar sum of all diagonal entries across all batch slices.
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
Compute unbiased MMD between two sample matrices `(d, n_samples)`.

Default uses RBF kernel from pairwise squared distances. You can pass:
- `kernel(x, y)` for a direct point-kernel, or
- `distance_kernel(d)` for kernels defined from squared distances.

`sigma` can be a scalar or a vector (multi-scale RBF average).

Arguments
- `x::AbstractMatrix{T}`: First sample matrix with shape `(d, m)`.
- `y::AbstractMatrix{T}`: Second sample matrix with shape `(d, n)`.
- `sigma::_SigmaArg`: RBF bandwidth, either a scalar or a non-empty vector of scalars.
- `kernel::Union{Nothing, Function}`: Optional kernel function operating directly on sample matrices.
- `distance_kernel::Union{Nothing, Function}`: Optional kernel function operating on squared distances.

Returns
- The unbiased MMD estimate as a scalar value.
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
Compute unbiased MMD for batched tensors `(d, n_samples, bs)`.

This method is generic for `AbstractArray{T,3}`. Fast CuArray distance
computation is handled by private `_pairwise_sqdist_batched` specialization.

Default uses RBF kernel from pairwise squared distances. You can pass:
- `kernel(x, y)` for a direct point-kernel (per-batch fallback), or
- `distance_kernel(d)` for kernels defined from squared distances.

`sigma` can be a scalar or a vector (multi-scale RBF average).

Arguments
- `x::AbstractArray{T, 3}`: First batched sample tensor with shape `(d, m, bs)`.
- `y::AbstractArray{T, 3}`: Second batched sample tensor with shape `(d, n, bs)`.
- `sigma::_SigmaArg`: RBF bandwidth, either a scalar or a non-empty vector of scalars.
- `kernel::Union{Nothing, Function}`: Optional kernel function operating directly on sample matrices.
- `distance_kernel::Union{Nothing, Function}`: Optional kernel function operating on squared distances.

Returns
- The mean unbiased MMD estimate across all batch slices as a scalar value.
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