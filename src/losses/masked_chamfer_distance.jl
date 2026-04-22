"""
    _unmask(x, mask)

Extract valid points from one set slice.

The input is reshaped to `(d, n)` and only the columns selected by `mask`
are kept. This helper is primarily useful for debugging and CPU fallbacks.
"""
function _unmask(x::AbstractArray{T}, mask::AbstractArray{Bool}) where {T<:AbstractFloat}
    x2 = reshape(x, size(x, 1), :)
    return x2[:, vec(mask)]
end


"""
    _masked_chamfer_distance_slice(x, y, x_mask, y_mask)

Compute masked Chamfer distance for a single set slice.

The function works on 2D slices `(d, n)` and boolean masks of length `n`.
It computes the full pairwise squared distance matrix, masks invalid rows or
columns with `Inf`, and returns the sum of the two directional averages.
"""
function _masked_chamfer_distance_slice(
    x::AbstractArray{T,2},
    y::AbstractArray{T,2},
    x_mask::AbstractArray{Bool,1},
    y_mask::AbstractArray{Bool,1},
) where {T<:AbstractFloat}
    size(x, 1) == size(y, 1) || throw(ArgumentError("masked_chamfer_distance: x and y must have the same feature dimension."))
    size(x, 2) == length(x_mask) || throw(ArgumentError("masked_chamfer_distance: x and x_mask must have the same number of points."))
    size(y, 2) == length(y_mask) || throw(ArgumentError("masked_chamfer_distance: y and y_mask must have the same number of points."))

    x_valid = x[:, x_mask]
    y_valid = y[:, y_mask]

    if isempty(x_valid) || isempty(y_valid)
        return zero(T)
    end

    dist = (reshape(sum(x_valid .^ 2, dims=1), :, 1) .+ reshape(sum(y_valid .^ 2, dims=1), 1, :)) .- (2f0 .* (transpose(x_valid) * y_valid))
    return mean(minimum(dist, dims=2)) + mean(minimum(dist, dims=1))
end


"""
    _masked_chamfer_distance_batched(x, y, x_mask, y_mask)

Vectorized masked Chamfer distance for batched tensors.

The implementation loops only over the batch dimension and keeps the heavy
work inside each batch slice as matrix operations. This avoids the fragile
3D broadcasting path while remaining fast on GPU.
"""
function _masked_chamfer_distance_batched(
    x::AbstractArray{T,3},
    y::AbstractArray{T,3},
    x_mask::AbstractArray{Bool,3},
    y_mask::AbstractArray{Bool,3},
) where {T<:AbstractFloat}
    size(x, 3) == size(y, 3) || throw(ArgumentError("masked_chamfer_distance: x and y must have the same batch size."))
    size(x, 2) == size(x_mask, 2) || throw(ArgumentError("masked_chamfer_distance: x and x_mask must have the same number of points."))
    size(y, 2) == size(y_mask, 2) || throw(ArgumentError("masked_chamfer_distance: y and y_mask must have the same number of points."))
    size(x_mask, 3) == size(x, 3) || throw(ArgumentError("masked_chamfer_distance: x and x_mask must have the same batch size."))
    size(y_mask, 3) == size(y, 3) || throw(ArgumentError("masked_chamfer_distance: y and y_mask must have the same batch size."))

    total = zero(T)
    @inbounds for b in 1:size(x, 3)
        total += _masked_chamfer_distance_slice(
            view(x, :, :, b),
            view(y, :, :, b),
            vec(view(x_mask, 1, :, b)),
            vec(view(y_mask, 1, :, b)),
        )
    end
    return total / T(size(x, 3))
end


"""
    masked_chamfer_distance(x, y, x_mask)

Chamfer distance for batched tensors where only `x` contains padding.

The result is averaged over the batch dimension.
"""
function masked_chamfer_distance(x::AbstractArray{T,3}, y::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    y_mask = fill!(similar(x_mask), true)
    return _masked_chamfer_distance_batched(x, y, x_mask, y_mask)
end


"""
    masked_chamfer_distance(x, y, x_mask, y_mask)

Chamfer distance for batched tensors where both inputs may contain padding.

The result is averaged over the batch dimension.
"""
function masked_chamfer_distance(x::AbstractArray{T,3}, y::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}, y_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    return _masked_chamfer_distance_batched(x, y, x_mask, y_mask)
end


"""
    chamfer_distance(x, y; x_mask)

Masked Chamfer distance wrapper for the single-mask case.
"""
function chamfer_distance(x::AbstractArray{T,3}, y::AbstractArray{T,3}; x_mask::AbstractArray{Bool,3}) where T <: AbstractFloat
    return masked_chamfer_distance(x, y, x_mask)
end


"""
    chamfer_distance(x, y; x_mask, y_mask)

Masked Chamfer distance wrapper for the two-mask case.
"""
function chamfer_distance(x::AbstractArray{T,3}, y::AbstractArray{T,3}; x_mask::AbstractArray{Bool,3}, y_mask::AbstractArray{Bool,3}) where T <: AbstractFloat
    return masked_chamfer_distance(x, y, x_mask, y_mask)
end


"""
    masked_chamfer_distance_cpu(x, y, x_mask)

CPU fallback for the single-mask variant of masked Chamfer distance.
"""
function masked_chamfer_distance_cpu(x::AbstractArray{T,3}, y::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    return masked_chamfer_distance(cpu(x), cpu(y), cpu(x_mask))
end


"""
    masked_chamfer_distance_cpu(x, y, x_mask, y_mask)

CPU fallback for masked Chamfer distance.

This is the safe entry point when `x` and `y` may live on GPU. It moves the
inputs to CPU once, then evaluates `masked_chamfer_distance` there.
"""
function masked_chamfer_distance_cpu(x::AbstractArray{T,3}, y::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}, y_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    return masked_chamfer_distance(cpu(x), cpu(y), cpu(x_mask), cpu(y_mask))
end
