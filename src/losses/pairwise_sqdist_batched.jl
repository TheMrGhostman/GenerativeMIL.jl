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
    x2 = sum(abs2, x; dims=1)
    y2 = sum(abs2, y; dims=1)
    
    x_t = permutedims(x, (2, 1, 3))
    x2_t = permutedims(x2, (2, 1, 3))

    g_xy = Flux.batched_mul(x_t, y)
    return max.(x2_t .+ y2 .- T(2) .* g_xy, zero(T))
end



"""
Masked variant: entries involving an invalid point (x_mask/y_mask == false) are `+Inf`.
x_mask: (1, n, bs), y_mask: (1, m, bs). Meant for nearest-neighbour search (argmin) inside
Zygote.@ignore -- do not differentiate through the Inf entries.
"""
function _pairwise_sqdist_batched(x::CuArray{T,3}, y::CuArray{T,3}, x_mask::CuArray{Bool,3}, y_mask::CuArray{Bool,3}) where T<:AbstractFloat
    @assert size(x, 3) == size(y, 3) "x and y must have the same batch size"

    x_t  = permutedims(x, (2, 1, 3))
    x2   = sum(abs2, x; dims=1)                                  # (1, n, bs)
    y2   = sum(abs2, y; dims=1)                                  # (1, m, bs)

    # mask goes into the small norm vectors, not the big matrix
    x2 = ifelse.(x_mask, x2, T(Inf))                             # (1, n, bs) -- tiny
    y2 = ifelse.(y_mask, y2, T(Inf))                             # (1, m, bs) -- tiny
    x2_t = permutedims(x2, (2, 1, 3))                            # (n, 1, bs)

    g_xy = Flux.batched_mul(x_t, y)
    return max.(x2_t .+ y2 .- T(2) .* g_xy, zero(T))             # same single fused kernel as before
end

# CPU (and generic) fallback: unmasked result + one broadcast with the penalties
function _pairwise_sqdist_batched(x::AbstractArray{T,3}, y::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}, y_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    px = permutedims(ifelse.(x_mask, zero(T), T(Inf)), (2, 1, 3))   # (n, 1, bs)
    py = ifelse.(y_mask, zero(T), T(Inf))                            # (1, m, bs)
    return _pairwise_sqdist_batched(x, y) .+ px .+ py
end