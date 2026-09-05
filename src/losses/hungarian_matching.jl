# Hungarian-matching based set reconstruction loss for fixed-cardinality slot decoders
# (DETR-style). Matches predicted slots to ground-truth (masked, variable-cardinality)
# set elements by minimum-cost bipartite assignment, then computes a reconstruction loss
# over matched pairs plus a binary existence loss over all slots.

"""
`_pairwise_sqeuclidean(a::AbstractMatrix{T}, b::AbstractMatrix{T}) where T<:AbstractFloat`

Pairwise squared Euclidean distance matrix between the columns of two matrices, via the
`‖a‖²+‖b‖²-2a·b` expansion (one matmul, no explicit loop over columns).

Arguments (positional):
- `a`: first set of column vectors `(d, n)`.
- `b`: second set of column vectors `(d, m)`.

Returns:
- `(n, m)` matrix of squared distances, clamped to `≥0` to guard against negative values from
  floating-point cancellation.
"""
_pairwise_sqeuclidean(a::AbstractMatrix{T}, b::AbstractMatrix{T}) where T<:AbstractFloat = begin
    a2 = sum(abs2, a, dims=1)  # (1, n)
    b2 = sum(abs2, b, dims=1)  # (1, m)
    max.(a2' .+ b2 .- 2 .* (a' * b), zero(T)) # (n, m)
end

"""
`hungarian_match(x̂::AbstractArray{T,3}, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}) where T<:AbstractFloat`

For each batch element, solve the linear assignment problem between predicted slots `x̂` and
masked, variable-cardinality ground-truth set elements `x`. Cost is squared Euclidean distance.
Masked-out ground-truth entries are dropped from the cost matrix before assignment (never
zeroed/`Inf`-filled), so they can never be selected as a match. Index bookkeeping only, not meant
to be differentiated through directly -- wrap calls in `Zygote.@ignore`.

Arguments (positional):
- `x̂`: predicted slots `(d, n_slots, bs)`.
- `x`: ground-truth set elements `(d, n_max, bs)`.
- `x_mask`: validity mask `(1, n_max, bs)` for the ground-truth entries.

Returns:
- `matched_pred::Vector{Vector{Int}}`: per-batch-element matched predicted-slot indices.
- `matched_gt::Vector{Vector{Int}}`: per-batch-element matched ground-truth indices, same
  order/length as `matched_pred`.
"""
function hungarian_match(x̂::AbstractArray{T,3}, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    _, n_slots, bs = size(x̂)
    x̂_cpu, x_cpu, mask_cpu = Array(x̂), Array(x), Array(x_mask)
    matched_pred = Vector{Vector{Int}}(undef, bs)
    matched_gt = Vector{Vector{Int}}(undef, bs)
    for b in 1:bs
        gt_idx = findall(vec(mask_cpu[1, :, b]))
        if isempty(gt_idx)
            matched_pred[b], matched_gt[b] = Int[], Int[]
            continue
        end
        C = _pairwise_sqeuclidean(x̂_cpu[:, :, b], x_cpu[:, gt_idx, b]) # (n_slots, n_gt)
        assignment, _ = Hungarian.hungarian(C)
        pred_idx = findall(!=(0), assignment)
        matched_pred[b] = pred_idx
        matched_gt[b] = gt_idx[assignment[pred_idx]]
    end
    return matched_pred, matched_gt
end

"""
`hungarian_match(C::AbstractArray{T,3}, l_mask::AbstractArray{Bool,4}) where T<:AbstractFloat`

Shape adapter: drops the singleton per-cluster dim from a `(1, 1, L, BS)` mask and forwards to
`hungarian_match(C, l_mask::AbstractArray{Bool,3})`.
"""
function hungarian_match(C::AbstractArray{T,3}, l_mask::AbstractArray{Bool,4}) where T<:AbstractFloat
    if size(l_mask, 1) != 1 || size(l_mask, 2) != 1
        throw(ArgumentError("l_mask must have shape (1, 1, L, BS)"))
    end
    new_mask = dropdims(l_mask, dims=2) # (1, L, BS)
    return hungarian_match(C, new_mask)
end

"""
`hungarian_match(C::AbstractArray{T,3}, l_mask::AbstractArray{Bool,4}, m_mask::AbstractArray{Bool,4}) where T<:AbstractFloat`

Shape adapter: drops the singleton per-cluster dim from both `(1, 1, L, BS)` and `(1, 1, M, BS)`
masks and forwards to `hungarian_match(C, x_mask::AbstractArray{Bool,3}, x̂_mask::AbstractArray{Bool,3})`
(the dual-masked, both-sides-variable-cardinality version).
"""
function hungarian_match(C::AbstractArray{T,3}, l_mask::AbstractArray{Bool,4}, m_mask::AbstractArray{Bool,4}) where T<:AbstractFloat
    if size(l_mask, 1) != 1 || size(l_mask, 2) != 1
        throw(ArgumentError("l_mask must have shape (1, 1, L, BS)"))
    end
    new_l_mask = dropdims(l_mask, dims=2) # (1, L, BS)
    new_m_mask = dropdims(m_mask, dims=2) # (1, M, BS)
    return hungarian_match(C, new_l_mask, new_m_mask)
end

"""
`hungarian_match(C::AbstractArray{T,3}, l_mask::AbstractArray{Bool,4}) where T<:AbstractFloat`

Solve the linear assignment problem per batch element from a precomputed cost matrix and return
both the matched pairs and the slot-existence target in one pass. Masked-out `l` columns are
dropped from the cost matrix before assignment (never zeroed/`Inf`-filled), so they can never be
selected as a match. The matched `(m, l, bs)` triples are returned as a single `CartesianIndex{3}`
array, so `C[c_ml]` gathers the matched costs directly and reconstruction loss reduces to
`mean(C[c_ml])` with no further per-batch indexing. Index bookkeeping only, not meant to be
differentiated through -- wrap the call in `Zygote.@ignore`.

Arguments (positional):
- `C`: cost matrix `(M, L, BS)` -- `M` predicted slots, `L` ground-truth clusters, `BS` batch size
  (e.g. from `chamfer_pairwise_distance`).
- `l_mask`: validity mask `(1, 1, L, BS)` for the `L` ground-truth clusters.

Returns:
- `c_ml::Vector{CartesianIndex{3}}`: matched `(m, l, bs)` triples, one per matched pair across the
  whole batch -- indexes directly into `C` or into any `(..., M, L, BS)`-shaped tensor.
- `exist_target::AbstractArray{T,3}`: binary existence target `(1, M, BS)`, `1` at every matched
  `m`, `0` elsewhere (unmatched/padding slots).
"""
function hungarian_match(C::AbstractArray{T,3}, l_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    M, _, BS = size(C)
    C_cpu, mask_cpu = Array(C), Array(l_mask)
    #ci_m = CartesianIndex{2}[]   # (m, bs) — index into a (..., M, BS) tensor
    #ci_l = CartesianIndex{2}[]   # (l, bs) — index into a (..., L, BS) tensor
    c_ml = CartesianIndex{3}[]  # (m, l, bs) — index into a (..., M, L, BS) tensor
    exist_target = zeros_like(C, (1, M, BS))  # (1, M, BS) - 
    for b in 1:BS
        l_idx = findall(vec(mask_cpu[1, :, b]))
        isempty(l_idx) && continue

        Cb = C_cpu[:, l_idx, b]                       # (M, n_valid_l) — masked columns just aren't there
        assignment, _ = Hungarian.hungarian(Cb)
        matched = filter(c -> c[2] != 0, CartesianIndex.(1:M, assignment))   # (m, position-in-l_idx)

        #append!(ci_m, CartesianIndex.(getindex.(matched, 1), b))
        #append!(ci_l, CartesianIndex.(l_idx[getindex.(matched, 2)], b))
        append!(c_ml, CartesianIndex.(getindex.(matched, 1), l_idx[getindex.(matched, 2)], b))
        exist_target[1, getindex.(matched, 1), b] .= one(T)
    end
    return c_ml, exist_target
end

"""
`hungarian_match(C::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}, x̂_mask::AbstractArray{Bool,3}) where T<:AbstractFloat`

Solve the linear assignment problem per batch element from a precomputed cost matrix when *both*
sides are variable-cardinality and masked (unlike `hungarian_match(C, l_mask)`, which only masks
the `L` ground-truth side and assumes a fixed `M` predicted slots). Masked-out rows (`x̂`) and
columns (`x`) are dropped from the cost matrix before assignment (never zeroed/`Inf`-filled), so
they can never be selected as a match. The matched `(m, l, bs)` triples are returned as a single
`CartesianIndex{3}` array, so `C[c_ml]` gathers the matched costs directly and reconstruction loss
reduces to `mean(C[c_ml])` with no further per-batch indexing. No existence target is produced here
-- with both sides masked there's no fixed-`M` slot layout to score existence against. Index
bookkeeping only, not meant to be differentiated through -- wrap the call in `Zygote.@ignore`.

Unlike `hungarian_match(C, l_mask)`, `x̂_mask` here is *known structural padding* (e.g. fewer
predicted elements exist than `M` at this level), not something the model needs to learn -- so
`exist_target` doesn't teach "does this slot exist" (already given), it teaches which of the
*structurally valid* slots actually got matched to a real ground-truth element vs. are excess/
orphaned predictions (the same DETR-style suppression signal as `hungarian_match(C, l_mask)`,
restricted to the valid subset). Padding positions (`x̂_mask == false`) are left at `0` in
`exist_target` but carry no training signal of their own -- mask `logits_exist`/`exist_target` by
`x̂_mask` before computing the BCE loss so padding positions aren't scored.

Arguments (positional):
- `C`: cost matrix `(M, L, BS)` -- `M` predicted elements, `L` ground-truth elements, `BS` batch
  size (e.g. from `chamfer_pairwise_distance`).
- `x_mask`: validity mask `(1, L, BS)` for the `L` ground-truth elements (columns of `C`).
- `x̂_mask`: validity mask `(1, M, BS)` for the `M` predicted elements (rows of `C`) -- structural
  padding, known in advance (unlike the existence the `exist_target` return trains for).

Returns:
- `c_ml::Vector{CartesianIndex{3}}`: matched `(m, l, bs)` triples, one per matched pair across the
  whole batch -- indexes directly into `C` or into any `(..., M, L, BS)`-shaped tensor.
- `exist_target::AbstractArray{T,3}`: binary existence target `(1, M, BS)`, `1` at every matched
  `m`, `0` elsewhere (unmatched valid slots and padding slots alike -- mask by `x̂_mask` before use).
"""
function hungarian_match(C::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}, x̂_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    M, _, BS = size(C)
    C_cpu, x_mask_cpu, x̂_mask_cpu = Array(C), Array(x_mask), Array(x̂_mask)
    c_ml = CartesianIndex{3}[]  # (m, l, bs) — index into a (..., M, L, BS) tensor
    exist_target = zeros_like(C, (1, M, BS))  # (1, M, BS) — mask by x̂_mask before use
    for b in 1:BS
        x_idx = findall(vec(x_mask_cpu[1, :, b]))
        x̂_idx = findall(vec(x̂_mask_cpu[1, :, b]))
        isempty(x_idx) && continue
        isempty(x̂_idx) && continue

        Cb = C_cpu[x̂_idx, x_idx, b]                       # (n_valid_m, n_valid_l) — masked rows/cols just aren't there
        assignment, _ = Hungarian.hungarian(Cb)
        # assignment has length n_valid_m (= length(x̂_idx)), not M; entries are positions within x_idx, or 0 if unmatched
        matched = filter(c -> c[2] != 0, CartesianIndex.(1:length(x̂_idx), assignment))   # (position-in-x̂_idx, position-in-x_idx)

        append!(c_ml, CartesianIndex.(x̂_idx[getindex.(matched, 1)], x_idx[getindex.(matched, 2)], b))
        exist_target[1, x̂_idx[getindex.(matched, 1)], b] .= one(T)
    end
    return c_ml, exist_target
end


"""
`hungarian_matching_loss(x̂::AbstractArray{T,N}, x::AbstractArray{T,N}, x_mask::AbstractArray{Bool,N}, logits_exist::AbstractArray{T,3}, distance::Function=chamfer_pairwise_distance) where {T<:AbstractFloat, N}`

Generalized (3D or 4D) counterpart of `hungarian_matching_loss` that plugs in an arbitrary
`distance` function instead of a fixed metric: matches predicted elements `x̂` to a
variable-cardinality, masked set of ground-truth elements `x` via Hungarian assignment on a
`distance`-based cost matrix, then computes matched-pair reconstruction + existence loss. Works at
the cluster level (`N=4`, each "element" itself a point set, e.g. with `distance=chamfer_pairwise_distance`)
or at the plain point level (`N=3`, e.g. with a batched pairwise-distance function such as
`_pairwise_sqdist_batched`); any other `N` raises an `ArgumentError`.

Arguments (positional):
- `x̂`: predicted elements -- clusters `(D, N, M, BS)` if `N=4`, plain points `(D, M, BS)` if `N=3`.
- `x`: ground-truth elements -- clusters `(D, N, L, BS)` if `N=4`, plain points `(D, L, BS)` if `N=3`.
- `x_mask`: validity mask for the `L` ground-truth elements -- `(1, 1, L, BS)` if `N=4`,
  `(1, L, BS)` if `N=3` (matches the shape `hungarian_match` dispatches on).
- `logits_exist`: existence logits `(1, M, BS)`.
- `distance`: pairwise-distance function used to build the `(M, L, BS)` cost matrix (default
  `chamfer_pairwise_distance`, which only supports `N=4`; pass a 3D-compatible function, e.g.
  `_pairwise_sqdist_batched`, when calling with `N=3` inputs).

Returns:
- `ℒ_rec`: mean cost over matched pairs (gathered directly from the cost matrix via `C[c_ml]`,
  see `hungarian_match`).
- `ℒ_exist`: binary cross-entropy existence loss over all `M` predicted slots.
"""
function hungarian_matching_loss(x̂::AbstractArray{T,N}, x::AbstractArray{T,N}, x_mask::AbstractArray{Bool,N}, logits_exist::AbstractArray{T,3}, distance::Function = chamfer_pairwise_distance) where {T<:AbstractFloat, N}
    N in (3, 4) || throw(ArgumentError("hungarian_matching_loss only supports 3D or 4D arrays, got $(N)D"))
    M = size(x̂, N - 1) # number of predicted slots

    C = distance(x̂, x) # (M, L, BS)
    matched_indices, exist_target = Zygote.@ignore hungarian_match(C, x_mask)
    n_matched = length(matched_indices)

    ℒ_rec = n_matched > 0 ? mean(C[matched_indices]) : zero(T)
    ℒ_exist = Flux.Losses.logitbinarycrossentropy(logits_exist, exist_target; agg=mean) * T(M) # sum over all slots, not mean 
    #matched_frac = T(n_matched) / T(M * BS)
    return ℒ_rec, ℒ_exist
end