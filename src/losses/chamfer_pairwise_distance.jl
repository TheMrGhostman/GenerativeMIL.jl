"""
`_nn_indices_clusters(x::AbstractArray{T,4}, y::AbstractArray{T,4}) where T<:AbstractFloat`

Find, for every point in every `(l,m)` cluster pair, the nearest-neighbor index needed for a
per-cluster-pair Chamfer distance. Computed via one fused batched squared-distance matrix
(`(N,L)` and `(N,M)` merged into the point axis so a single `batched_mul` covers every cluster
pair at once), then `argmin` restricted to the matching cluster's points on each side. Meant to
be called inside `Zygote.@ignore` — index search only, not meant to be differentiated through.

Arguments (positional):
- `x`: first cluster set tensor `(D, N, L, BS)` — `D` feature dim, `N` points per cluster, `L`
  clusters, `BS` batch size.
- `y`: second cluster set tensor `(D, N, M, BS)` — same `D`/`N`/`BS` as `x`, `M` clusters.

Returns:
- `nb`: nearest-neighbor index into `y`'s point axis for each `(n_a, l, m, bs)`, shape `(N,L,M,BS)`.
- `na`: nearest-neighbor index into `x`'s point axis for each `(n_b, l, m, bs)`, shape `(N,L,M,BS)`.
"""
function _nn_indices_clusters(x::AbstractArray{T,4}, y::AbstractArray{T,4}) where T<:AbstractFloat
    D, N, L, BS = size(x)
    _, N2, M, _ = size(y)
    @assert N == N2 "points-per-cluster must match between x and y"

    xr = reshape(x, D, N*L, BS)
    yr = reshape(y, D, N*M, BS)

    P  = _pairwise_sqdist_batched(xr, yr)          # (N*L, N*M, BS) — forward-only, thrown away
    P5 = reshape(P, N, L, N, M, BS)                # (n_a, l, n_b, m, bs)

    nb = getindex.(dropdims(argmin(P5, dims=3), dims=3), 3)   # (N,L,M,BS) — best n_b per (n_a,l,m,bs)
    na = getindex.(dropdims(argmin(P5, dims=1), dims=1), 1)   # (L,N,M,BS) — best n_a per (n_b,l,m,bs)
    na = permutedims(na, (2,1,3,4))                            # -> (N,L,M,BS), N in y's-point-axis position
    return nb, na
end


"""
`chamfer_pairwise_distance(x::AbstractArray{T,4}, y::AbstractArray{T,4}; w1::T=one(T), w2::T=one(T), agg::Function=mean) where T<:AbstractFloat`

Compute the Chamfer distance between every pair of clusters in `x` and `y`, returning a dense
`(L,M,BS)` cost matrix (e.g. for Hungarian matching at the outer/cluster level). Nearest-neighbor
search runs inside `Zygote.@ignore` (see `_nn_indices_clusters`); only the gather + squared-error
step is differentiated, keeping the backward graph proportional to `D·N·L·M·BS` rather than the
full `N²·L·M·BS` pairwise distance tensor.

Arguments (positional):
- `x`: first cluster set tensor `(D, N, L, BS)`.
- `y`: second cluster set tensor `(D, N, M, BS)` — same `D`/`N`/`BS` as `x`.

Arguments (keyword):
- `w1`: weight for the `x → y` direction (default `one(T)`).
- `w2`: weight for the `y → x` direction (default `one(T)`).
- `agg`: aggregation function over the per-cluster points, e.g. `mean` or `sum` (default `mean`).

Returns:
- `(L, M, BS)` tensor of pairwise Chamfer distances between every cluster in `x` and every
  cluster in `y`, per batch element.
"""
function chamfer_pairwise_distance(x::AbstractArray{T,4}, y::AbstractArray{T,4}; w1::T=one(T), w2::T=one(T), agg::Function=mean) where T<:AbstractFloat
    D, N, L, BS = size(x)
    _, _, M, _  = size(y)

    ci_y, ci_x = Zygote.@ignore begin
        nb, na = _nn_indices_clusters(x, y)
        mgrid, bsgrid, lgrid = reshape(1:M,1,1,M,1), reshape(1:BS,1,1,1,BS), reshape(1:L,1,L,1,1)
        CartesianIndex.(nb, mgrid, bsgrid), CartesianIndex.(na, lgrid, bsgrid)
    end

    B_matched   = y[:, ci_y]       # (D,N,L,M,BS)
    dist_A_to_B = dropdims(agg(sum(abs2, reshape(x,D,N,L,1,BS) .- B_matched; dims=1); dims=2), dims=(1,2))

    A_matched   = x[:, ci_x]        # (D,N,L,M,BS)
    dist_B_to_A = dropdims(agg(sum(abs2, A_matched .- reshape(y,D,N,1,M,BS); dims=1); dims=2), dims=(1,2))

    return w1 .* dist_A_to_B .+ w2 .* dist_B_to_A   # (L, M, BS)
end

sum_chamfer_pairwise_distance(x::AbstractArray{T,4}, y::AbstractArray{T,4}; w1::T=one(T), w2::T=one(T)) where T<:AbstractFloat =
    chamfer_pairwise_distance(x, y; w1=w1, w2=w2, agg=sum)

# ---------------------------------------------------------------------------------------------------------------------
# Masked variants: variable number of valid points per cluster.
# ---------------------------------------------------------------------------------------------------------------------
const _PointMask4 = Union{AbstractArray{Bool,4}, Nothing}

_full_point_mask(x::AbstractArray{T,4}) where T = fill!(similar(x, Bool, 1, size(x, 2), size(x, 3), size(x, 4)), true)

function _check_point_mask(x::AbstractArray{T,4}, m::AbstractArray{Bool,4}, name::String) where T
    size(m) == (1, size(x, 2), size(x, 3), size(x, 4)) ||
        throw(DimensionMismatch("$name must have size (1, N, ·, BS) = $((1, size(x, 2), size(x, 3), size(x, 4))), got $(size(m))"))
    return nothing
end

"""
`_nn_indices_clusters(x, y, x_mask, y_mask)`

Masked counterpart of `_nn_indices_clusters(x, y)`. `x_mask (1,N,L,BS)` / `y_mask (1,N,M,BS)` mark valid points.
Invalid point pairs are set to `Inf` in the fused distance tensor `P5 (n_a,l,n_b,m,bs)` before both `argmin`s, so every
valid point gets its nearest *valid* neighbour in the other cluster. Rows of invalid points (and every row when the other
cluster is empty) are all `Inf` and return an arbitrary index -- the caller must give them zero weight. Index search
only; call inside `Zygote.@ignore`.
"""
function _nn_indices_clusters(x::AbstractArray{T,4}, y::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}, y_mask::AbstractArray{Bool,4}) where T<:AbstractFloat
    D, N, L, BS = size(x)
    _, N2, M, _ = size(y)
    @assert N == N2 "points-per-cluster must match between x and y"

    P  = _pairwise_sqdist_batched(reshape(x, D, N*L, BS), reshape(y, D, N*M, BS))
    P5 = reshape(P, N, L, N, M, BS)                                             # (n_a, l, n_b, m, bs)
    valid = reshape(x_mask, N, L, 1, 1, BS) .& reshape(y_mask, 1, 1, N, M, BS)  # (n_a, l, n_b, m, bs)
    P5 = ifelse.(valid, P5, T(Inf))

    nb = getindex.(dropdims(argmin(P5, dims=3), dims=3), 3)   # (N,L,M,BS)
    na = getindex.(dropdims(argmin(P5, dims=1), dims=1), 1)   # (L,N,M,BS)
    na = permutedims(na, (2,1,3,4))                            # -> (N,L,M,BS)
    return nb, na
end

# masked reduction over the point axis (dims=2) of d (1,N,L,M,BS) with 0/1 weights w; only mean and sum are supported
function _masked_point_agg(d::AbstractArray{T}, w::AbstractArray{T}, agg::Function) where T<:AbstractFloat
    s = sum(d .* w; dims=2)
    agg === sum  && return s
    agg === mean && return s ./ max.(sum(w; dims=2), one(T))   # mean over valid points; empty cluster -> 0
    throw(ArgumentError("masked chamfer_pairwise_distance supports agg = mean or sum, got $agg"))
end

"""
`chamfer_pairwise_distance(x, y, x_mask, y_mask; w1=1, w2=1, agg=mean)`

Masked version of `chamfer_pairwise_distance(x, y)`: Chamfer distance between every pair of clusters, using only the
valid points of each cluster. Returns `(L, M, BS)`, same as the unmasked version, and equals it when all masks are true.

Arguments (positional):
- `x`: `(D, N, L, BS)`, `y`: `(D, N, M, BS)` -- same `D`/`N`/`BS`.
- `x_mask`: `(1, N, L, BS)` Bool, or `nothing` (all points valid).
- `y_mask`: `(1, N, M, BS)` Bool, or `nothing` (all points valid).

Arguments (keyword):
- `w1`, `w2`: weights of the `x → y` and `y → x` directions.
- `agg`: `mean` (mean over valid points) or `sum` (sum over valid points); other functions throw.

Notes:
- Fully vectorised: masks are applied to the fused distance tensor before `argmin` (see the masked
  `_nn_indices_clusters`); only the gather + weighted squared error is differentiated. Padded points get zero gradient.
- Empty cluster (all points masked): every pair `(l, m)` involving it gets cost 0 (same convention as
  `masked_chamfer_distance`). A cost of 0 would look like a perfect match, so empty clusters must be excluded from
  matching -- `hungarian_match` already drops padding ground-truth clusters via its cluster mask.
- Padded point values must be finite (e.g. zeros): they are gathered and multiplied by 0, and `0 * NaN = NaN`.
"""
function chamfer_pairwise_distance(x::AbstractArray{T,4}, y::AbstractArray{T,4}, x_mask::_PointMask4, y_mask::_PointMask4; w1::T=one(T), w2::T=one(T), agg::Function=mean) where T<:AbstractFloat
    
    isnothing(x_mask) && isnothing(y_mask) && return chamfer_pairwise_distance(x, y; w1=w1, w2=w2, agg=agg)
    D, N, L, BS = size(x)
    _, _, M, _  = size(y)
    xm = isnothing(x_mask) ? Zygote.@ignore(_full_point_mask(x)) : x_mask
    ym = isnothing(y_mask) ? Zygote.@ignore(_full_point_mask(y)) : y_mask
    Zygote.@ignore _check_point_mask(x, xm, "x_mask")
    Zygote.@ignore _check_point_mask(y, ym, "y_mask")

    ci_y, ci_x = Zygote.@ignore begin
        nb, na = _nn_indices_clusters(x, y, xm, ym)
        mgrid, bsgrid, lgrid = reshape(1:M,1,1,M,1), reshape(1:BS,1,1,1,BS), reshape(1:L,1,L,1,1)
        CartesianIndex.(nb, mgrid, bsgrid), CartesianIndex.(na, lgrid, bsgrid)
    end
    wx = Zygote.@ignore T.(reshape(xm, 1, N, L, 1, BS))   # weights of x points (x → y direction)
    wy = Zygote.@ignore T.(reshape(ym, 1, N, 1, M, BS))   # weights of y points (y → x direction)

    B_matched   = y[:, ci_y]                                                     # (D,N,L,M,BS)
    d_A_to_B    = sum(abs2, reshape(x, D, N, L, 1, BS) .- B_matched; dims=1)     # (1,N,L,M,BS)
    dist_A_to_B = dropdims(_masked_point_agg(d_A_to_B, wx, agg), dims=(1,2))     # (L,M,BS)

    A_matched   = x[:, ci_x]                                                     # (D,N,L,M,BS)
    d_B_to_A    = sum(abs2, A_matched .- reshape(y, D, N, 1, M, BS); dims=1)     # (1,N,L,M,BS)
    dist_B_to_A = dropdims(_masked_point_agg(d_B_to_A, wy, agg), dims=(1,2))     # (L,M,BS)

    # pairs where either cluster is empty: nearest-neighbour indices are arbitrary there -> cost 0
    pair_ok = Zygote.@ignore T.(reshape(any(xm; dims=2), L, 1, BS) .& reshape(any(ym; dims=2), 1, M, BS))   # (L,M,BS)
    return (w1 .* dist_A_to_B .+ w2 .* dist_B_to_A) .* pair_ok   # (L, M, BS)
end

sum_chamfer_pairwise_distance(x::AbstractArray{T,4}, y::AbstractArray{T,4}, x_mask::_PointMask4, y_mask::_PointMask4; w1::T=one(T), w2::T=one(T)) where T<:AbstractFloat =
    chamfer_pairwise_distance(x, y, x_mask, y_mask; w1=w1, w2=w2, agg=sum)
