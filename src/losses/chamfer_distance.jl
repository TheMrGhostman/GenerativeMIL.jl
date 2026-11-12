# I borrowed this implementation from Flux3D.jl, which is licensed under MIT License.
# I could use this package but since I need only this function, I just copy-paste the code here to avoid adding more dependencies.

"""
    chamfer_distance(A::PointCloud, B::PointCloud; w1=1.0, w2=1.0)

Computes the chamfer distance between PointCloud `A` and `B`. 

`w1` and `w2` are optional arguments for specifying the weighted mean of between two 
pcloud for computing metrics.
"""
chamfer_distance(
    A::AbstractArray{<:Number},
    B::AbstractArray{<:Number};
    w1::Number = 1.0,
    w2::Number = 1.0,
) = _chamfer_distance(Float32.(A), Float32.(B), Float32(w1), Float32(w2))

chamfer_distance(
    A::AbstractArray{Float32},
    B::AbstractArray{Float32};
    w1::Number = 1.0,
    w2::Number = 1.0,
) = _chamfer_distance(A, B, Float32(w1), Float32(w2))

# ---- masked API (masks are positional: keyword arguments do not take part in dispatch, so a keyword-mask method
# ---- would replace the unmasked 2-argument method above) ------------------------------------------------------------

"""
    chamfer_distance(A, B, A_mask, B_mask; w1=1, w2=1)
    chamfer_distance(A, B, A_mask; w1=1, w2=1)        # all points of B valid

Masked Chamfer distance for batched sets `A (d, n, bs)`, `B (d, m, bs)` with validity masks `A_mask (1, n, bs)`,
`B_mask (1, m, bs)`: per bag, mean over valid points of the squared distance to the nearest valid point of the other
set (both directions, weighted by `w1`/`w2`), then mean over the batch. An empty bag (either side) contributes 0.
Vectorised (CPU and GPU), see `_chamfer_distance(A, B, A_mask, B_mask, w1, w2)`. Replaces the former loop-based
implementation (`masked_chamfer_distance.jl`, kept only as a test reference) with the same semantics.
"""
chamfer_distance(
    A::AbstractArray{T,3},
    B::AbstractArray{T,3},
    A_mask::AbstractArray{Bool,3},
    B_mask::AbstractArray{Bool,3};
    w1::Number = 1.0,
    w2::Number = 1.0,
) where T<:AbstractFloat = _chamfer_distance(A, B, A_mask, B_mask, T(w1), T(w2))

chamfer_distance(
    A::AbstractArray{T,3},
    B::AbstractArray{T,3},
    A_mask::AbstractArray{Bool,3};
    kwargs...,
) where T<:AbstractFloat = chamfer_distance(A, B, A_mask, Zygote.@ignore(_all_valid_mask(B)); kwargs...)

_all_valid_mask(x::AbstractArray{T,3}) where T = fill!(similar(x, Bool, 1, size(x, 2), size(x, 3)), true)

function _chamfer_distance(
    A::AbstractArray{Float32,2},
    B::AbstractArray{Float32,2},
    w1::Float32 = 1.0f0,
    w2::Float32 = 1.0f0,
)
    A = reshape(A, size(A)..., 1)
    B = reshape(B, size(B)..., 1)
    return _chamfer_distance(A, B, w1, w2)
end

function _chamfer_distance(
    A::AbstractArray{Float32,3},
    B::AbstractArray{Float32,3},
    w1::Float32 = 1.0f0,
    w2::Float32 = 1.0f0,
)
    nn_for_A, nn_for_B = Zygote.@ignore _nearest_neighbors(A, B)

    dist_A_to_B = mean((A .- B[:, nn_for_A]) .^ 2) * 3.0f0
    dist_B_to_A = mean((B .- A[:, nn_for_B]) .^ 2) * 3.0f0

    distance = (w1 * dist_A_to_B) + (w2 * dist_B_to_A)
    return distance
end

"""
    _chamfer_distance(A, B, A_mask, B_mask, w1=1, w2=1)

Masked, vectorised Chamfer distance for batched sets `A (d, n, bs)`, `B (d, m, bs)` with validity masks
`A_mask (1, n, bs)`, `B_mask (1, m, bs)`. Same semantics as the loop-based `masked_chamfer_distance`:
per bag, mean over valid points of the squared distance to the nearest *valid* point of the other set (each
direction), weighted sum of the two directions, then mean over the batch. Any dimension `d` (no `* 3` factor).
A bag with no valid point on either side contributes 0. Padded point values must be finite.
"""
function _chamfer_distance(
    A::AbstractArray{T,3},
    B::AbstractArray{T,3},
    A_mask::AbstractArray{Bool,3},
    B_mask::AbstractArray{Bool,3},
    w1::T = one(T),
    w2::T = one(T),
) where T<:AbstractFloat
    nn_for_A, nn_for_B = Zygote.@ignore _nearest_neighbors(A, B, A_mask, B_mask)   # (n, bs), (m, bs)
    wA = Zygote.@ignore T.(A_mask)                                                   # (1, n, bs)
    wB = Zygote.@ignore T.(B_mask)                                                   # (1, m, bs)

    dA = sum(abs2, A .- B[:, nn_for_A]; dims=1)    # (1, n, bs) squared distance of each A point to its NN in B
    dB = sum(abs2, B .- A[:, nn_for_B]; dims=1)    # (1, m, bs)

    # per-bag mean over valid points only (padded points get weight 0; max(.,1) avoids 0/0 for empty bags)
    dist_A_to_B = sum(dA .* wA; dims=2) ./ max.(sum(wA; dims=2), one(T))     # (1, 1, bs)
    dist_B_to_A = sum(dB .* wB; dims=2) ./ max.(sum(wB; dims=2), one(T))     # (1, 1, bs)

    # bag empty on either side: NN indices are arbitrary there -> contribution 0 (as in masked_chamfer_distance)
    ok = Zygote.@ignore T.(any(A_mask; dims=2) .& any(B_mask; dims=2))       # (1, 1, bs)
    return mean((w1 .* dist_A_to_B .+ w2 .* dist_B_to_A) .* ok)
end

# helper functions

function _nearest_neighbors(x::Array{Float32,3}, y::Array{Float32,3})
    nn_for_x = cat(
        [
            CartesianIndex.(reduce(vcat, NearestNeighbors.knn(KDTree(y[:, :, i]), x[:, :, i], 1)[1]), i)
            for i = 1:size(x, 3)
        ]...,
        dims = 2,
    )
    nn_for_y = cat(
        [
            CartesianIndex.(reduce(vcat, NearestNeighbors.knn(KDTree(x[:, :, i]), y[:, :, i], 1)[1]), i)
            for i = 1:size(x, 3)
        ]...,
        dims = 2,
    )
    return nn_for_x, nn_for_y
end

function _nearest_neighbors(x::CuArray{Float32,3}, y::CuArray{Float32,3})
    P = _pairwise_sqdist_batched(x, y)                                          # (n, m, bs)
    bgrid = reshape(1:size(x, 3), 1, :)
    jx = getindex.(dropdims(argmin(P; dims=2); dims=2), 2)                      # (n, bs): best y point per x point
    iy = getindex.(dropdims(argmin(P; dims=1); dims=1), 1)                      # (m, bs): best x point per y point
    return CartesianIndex.(jx, bgrid), CartesianIndex.(iy, bgrid)
end

"""
    _nearest_neighbors(x, y, x_mask, y_mask)

Masked nearest neighbours for `x (d, n, bs)`, `y (d, m, bs)`, masks `(1, n, bs)` / `(1, m, bs)`. Works on CPU and GPU
(the masked `_pairwise_sqdist_batched` puts `+Inf` on every pair with an invalid point). Returns
`nn_for_x (n, bs)` and `nn_for_y (m, bs)` of `CartesianIndex{2}` (point, bag), on the same device as `x` (no copy to
CPU), so `y[:, nn_for_x]` gathers directly. Rows of invalid points (or of a bag whose other side is empty) get an
arbitrary index -- the caller must give them zero weight. Index search only; call inside `Zygote.@ignore`.
"""
function _nearest_neighbors(x::AbstractArray{T,3}, y::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}, y_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    P = _pairwise_sqdist_batched(x, y, x_mask, y_mask)                          # (n, m, bs), Inf where invalid
    bgrid = reshape(1:size(x, 3), 1, :)
    jx = getindex.(dropdims(argmin(P; dims=2); dims=2), 2)                      # (n, bs): best y point per x point
    iy = getindex.(dropdims(argmin(P; dims=1); dims=1), 1)                      # (m, bs): best x point per y point
    return CartesianIndex.(jx, bgrid), CartesianIndex.(iy, bgrid)
end