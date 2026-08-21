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
