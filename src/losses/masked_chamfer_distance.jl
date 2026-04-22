"""
    _unmask(x, mask)

Remove masked elements from a single batched point-cloud slice.

The function reshapes the input to `(d, n)` and keeps only columns where
`mask` is `true`. It returns a dense matrix so downstream Chamfer distance
calls stay simple and type-stable.
"""
function _unmask(x::AbstractArray{T}, mask::AbstractArray{Bool}) where {T<:AbstractFloat}
    x2 = reshape(x, size(x, 1), :)
    return x2[:, vec(mask)]
end


"""
    masked_chamfer_distance(x, y, x_mask)

Compute Chamfer distance for a batched set tensor `x` against an unmasked
reference tensor `y`, using `x_mask` to remove padded elements from `x`.

This method averages the per-batch Chamfer distance over the batch dimension.
"""
function masked_chamfer_distance(x::AbstractArray{T,3}, y::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    nb = size(x, 3)
    nb == size(y, 3) || throw(ArgumentError("masked_chamfer_distance: x and y must have the same batch size."))

    loss = zero(T)
    @inbounds for i in 1:nb
        loss += chamfer_distance(_unmask(view(x, :, :, i), view(x_mask, :, :, i)), view(y, :, :, i))
    end
    return loss / T(nb)
end


"""
    masked_chamfer_distance(x, y, x_mask, y_mask)

Compute Chamfer distance for two batched set tensors, removing padded elements
from both inputs before evaluating the per-batch distance.

This method averages the per-batch Chamfer distance over the batch dimension.
"""
function masked_chamfer_distance(x::AbstractArray{T,3}, y::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}, y_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    nb = size(x, 3)
    nb == size(y, 3) || throw(ArgumentError("masked_chamfer_distance: x and y must have the same batch size."))

    loss = zero(T)
    @inbounds for i in 1:nb
        loss += chamfer_distance(
            _unmask(view(x, :, :, i), view(x_mask, :, :, i)),
            _unmask(view(y, :, :, i), view(y_mask, :, :, i))
        )
    end
    return loss / T(nb)
end


function chamfer_distance(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}; x_mask::AbstractArray{Bool, 3}) where T <: AbstractFloat
    return masked_chamfer_distance(x, y, x_mask)
end

function chamfer_distance(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}; x_mask::AbstractArray{Bool, 3}, y_mask::AbstractArray{Bool, 3}) where T <: AbstractFloat
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

