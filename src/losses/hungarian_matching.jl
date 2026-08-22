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

function hungarian_match(C::AbstractArray{T,3}, l_mask::AbstractArray{Bool,4}) where T<:AbstractFloat
    if size(l_mask, 1) != 1 || size(l_mask, 2) != 1
        throw(ArgumentError("l_mask must have shape (1, 1, L, BS)"))
    end
    new_mask = dropdims(l_mask, dims=2) # (1, L, BS)
    return hungarian_match(C, new_mask)
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
`hungarian_matching_loss(x̂::AbstractArray{T,3}, logits_exist::AbstractArray{T,3}, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}) where T<:AbstractFloat`

Matched-pair reconstruction (mean squared error over matched slots) + existence (binary
cross-entropy over all slots) loss for a fixed-cardinality slot decoder, aligned to a
variable-cardinality masked ground-truth set via Hungarian matching. `hungarian_match` does the
CPU-only combinatorial matching inside `Zygote.@ignore`; the reconstruction term is then a plain
`for b in 1:bs` loop gathering matched slots per batch element (kept this way -- not the fused
single-gather version -- for compatibility with other experiments that depend on this exact code
path).

Arguments (positional):
- `x̂`: predicted slots `(d, n_slots, bs)`.
- `logits_exist`: existence logits `(1, n_slots, bs)`.
- `x`: ground-truth set elements `(d, n_max, bs)`.
- `x_mask`: validity mask `(1, n_max, bs)` for the ground-truth entries.

Returns:
- `ℒ_rec`: mean squared error over matched pairs.
- `ℒ_exist`: binary cross-entropy existence loss over all slots.
- `matched_frac`: fraction of `n_slots * bs` slots that were matched.
"""
function hungarian_matching_loss(x̂::AbstractArray{T,3}, logits_exist::AbstractArray{T,3}, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    _, n_slots, bs = size(x̂)
    matched_pred, matched_gt, exist_target = Zygote.@ignore begin
        mp, mg = hungarian_match(x̂, x, x_mask)
        t = zeros(T, 1, n_slots, bs)
        for b in 1:bs
            t[1, mp[b], b] .= one(T)
        end
        mp, mg, t
    end

    rec_sum = zero(T)
    n_matched = 0
    for b in 1:bs
        pi, gi = matched_pred[b], matched_gt[b]
        isempty(pi) && continue
        rec_sum = rec_sum + sum(abs2, x̂[:, pi, b] .- x[:, gi, b])
        n_matched += length(pi)
    end
    ℒ_rec = n_matched > 0 ? rec_sum / n_matched : zero(T)
    ℒ_exist = Flux.Losses.logitbinarycrossentropy(logits_exist, exist_target; agg=Flux.mean)
    matched_frac = T(n_matched) / T(n_slots * bs)
    return ℒ_rec, ℒ_exist, matched_frac
end


"""
`hungarian_matching_loss(x̂::AbstractArray{T,4}, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}, logits_exist::AbstractArray{T,3}, distance::Function=chamfer_pairwise_distance) where T<:AbstractFloat`

Outer/cluster-level counterpart of `hungarian_matching_loss` for 3D tensors: matches predicted
clusters `x̂` to a variable-cardinality, masked set of ground-truth clusters `x` (each cluster
itself a point set) via Hungarian assignment on a `distance`-based cost matrix, e.g.
`chamfer_pairwise_distance`, then computes matched-pair reconstruction + existence loss.

Arguments (positional):
- `x̂`: predicted clusters `(D, N, M, BS)`.
- `x`: ground-truth clusters `(D, N, L, BS)`.
- `x_mask`: validity mask `(1, 1, L, BS)` for the `L` ground-truth clusters.
- `logits_exist`: existence logits `(1, M, BS)`.
- `distance`: pairwise cluster-distance function used to build the `(M, L, BS)` cost matrix
  (default `chamfer_pairwise_distance`).

Returns:
- `ℒ_rec`: mean cost over matched cluster pairs (gathered directly from the cost matrix via
  `C[c_ml]`, see `hungarian_match`).
- `ℒ_exist`: binary cross-entropy existence loss over all `M` predicted slots.
"""
function hungarian_matching_loss(x̂::AbstractArray{T,4}, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}, logits_exist::AbstractArray{T,3}, distance::Function = chamfer_pairwise_distance) where T<:AbstractFloat
    _, _, M, BS = size(x̂)

    C = distance(x̂, x) # (M, L, BS)
    matched_indices, exist_target = Zygote.@ignore hungarian_match(C, x_mask)
    n_matched = length(matched_indices)

    ℒ_rec = n_matched > 0 ? mean(C[matched_indices]) : zero(T)
    ℒ_exist = Flux.Losses.logitbinarycrossentropy(logits_exist, exist_target; agg=mean) * T(M) # sum over all slots, not mean 
    #matched_frac = T(n_matched) / T(M * BS)
    return ℒ_rec, ℒ_exist
end
