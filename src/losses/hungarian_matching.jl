# Hungarian-matching based set reconstruction loss for fixed-cardinality slot decoders
# (DETR-style). Matches predicted slots to ground-truth (masked, variable-cardinality)
# set elements by minimum-cost bipartite assignment, then computes a reconstruction loss
# over matched pairs plus a binary existence loss over all slots.

_pairwise_sqeuclidean(a::AbstractMatrix{T}, b::AbstractMatrix{T}) where T<:AbstractFloat = begin
    a2 = sum(abs2, a, dims=1)  # (1, n)
    b2 = sum(abs2, b, dims=1)  # (1, m)
    max.(a2' .+ b2 .- 2 .* (a' * b), zero(T)) # (n, m)
end

"""
    hungarian_match(x̂, x, x_mask)

For each batch element, solve the linear assignment problem between predicted slots `x̂`
`(d, n_slots, bs)` and masked ground-truth set elements `x` `(d, n_max, bs)` (valid
entries marked by `x_mask` `(1, n_max, bs)`). Cost is squared Euclidean distance.

Returns per-batch vectors of matched predicted-slot indices and matched ground-truth
indices. Index bookkeeping only, not meant to be differentiated through directly -- wrap
calls in `ChainRulesCore.ignore_derivatives`.
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

function hungarian_match(C::AbstractArray{T,3}, l_mask::AbstractArray{Bool,4}) where T<:AbstractFloat
    M, _, BS = size(C)
    C_cpu, mask_cpu = Array(C), Array(l_mask)
    #ci_m = CartesianIndex{2}[]   # (m, bs) — index into a (..., M, BS) tensor
    #ci_l = CartesianIndex{2}[]   # (l, bs) — index into a (..., L, BS) tensor
    c_ml = CartesianIndex{3}[]  # (m, l, bs) — index into a (..., M, L, BS) tensor
    exist_target = zeros_like(C, (1, M, BS))  # (1, M, BS) - 
    for b in 1:BS
        l_idx = findall(vec(mask_cpu[1, 1, :, b]))
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
    hungarian_matching_loss(x̂, logits_exist, x, x_mask)

Matched-pair reconstruction (mean squared error over matched slots) + existence (binary
cross-entropy over all slots) loss for a fixed-cardinality slot decoder, aligned to a
variable-cardinality masked ground-truth set via Hungarian matching.

Returns `(ℒ_rec, ℒ_exist, matched_frac)`.
"""
"""
    hungarian_matching_loss(x̂, logits_exist, x, x_mask)

GPU-safe *and* GPU-fast. `hungarian_match` (above) does the CPU-only combinatorial
matching inside `Zygote.@ignore`. The naive way to turn that into a reconstruction loss
is a `for b in 1:bs` Julia loop indexing `x̂`/`x` per batch element -- but on GPU that's
`bs` separate tiny kernel launches, each with its own launch+sync overhead, which measured
~10x *slower* than plain CPU for this workload (92ms vs 9ms/call at batch=128). Instead,
the matched (slot, batch) and (gt_position, batch) pairs are flattened into a single pair
of linear index vectors covering the *whole* batch (cheap, CPU-only, inside the same
`Zygote.@ignore` block), and `x̂`/`x` are reshaped to merge their slot/batch dims so ONE
gather + subtract + sum handles every matched pair across the whole batch in one shot
(measured ~2.4ms/call on the same workload -- faster than CPU, not just "not slower").
`ℒ_rec` is numerically identical to the old per-batch-loop version, just computed as one
batched op instead of `bs` sequential ones.
"""
function hungarian_matching_loss(x̂::AbstractArray{T,3}, logits_exist::AbstractArray{T,3}, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    d, n_slots, bs = size(x̂)
    _, n_max, _ = size(x)
    on_gpu = x̂ isa CUDA.CuArray

    pred_flat, gt_flat, exist_target = Zygote.@ignore begin
        mp, mg = hungarian_match(x̂, x, x_mask)
        t = zeros(T, 1, n_slots, bs)
        pf, gf = Int[], Int[]
        for b in 1:bs
            t[1, mp[b], b] .= one(T)
            append!(pf, mp[b] .+ (b - 1) * n_slots) # (slot, b) -> linear index into merged (n_slots*bs)
            append!(gf, mg[b] .+ (b - 1) * n_max)   # (gt_pos, b) -> linear index into merged (n_max*bs)
        end
        t = on_gpu ? CUDA.cu(t) : t   # match exist_target's device to x̂/logits_exist
        pf = on_gpu ? CUDA.cu(pf) : pf
        gf = on_gpu ? CUDA.cu(gf) : gf
        pf, gf, t
    end

    n_matched = length(pred_flat)
    if n_matched > 0
        x̂_flat = reshape(x̂, d, n_slots * bs) # view, no copy -- x̂/x stay on their original device
        x_flat = reshape(x, d, n_max * bs)
        diff = x̂_flat[:, pred_flat] .- x_flat[:, gt_flat] # ONE gather for every matched pair in the batch
        ℒ_rec = sum(abs2, diff) / n_matched
    else
        ℒ_rec = zero(T)
    end
    ℒ_exist = Flux.Losses.logitbinarycrossentropy(logits_exist, exist_target; agg=Flux.mean)
    matched_frac = T(n_matched) / T(n_slots * bs)
    return ℒ_rec, ℒ_exist, matched_frac
end


function hungarian_matching_loss(x̂::AbstractArray{T,4}, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}, logits_exist::AbstractArray{T,3}, distance::Function = chamfer_pairwise_distance) where T<:AbstractFloat
    _, _, M, BS = size(x̂)

    C = distance(x̂, x) # (M, L, BS)
    matched_indices, exist_target = Zygote.@ignore hungarian_match(C, x_mask)
    n_matched = length(matched_indices)

    ℒ_rec = n_matched > 0 ? mean(C[matched_indices]) : zero(T)
    ℒ_exist = Flux.Losses.logitbinarycrossentropy(logits_exist, exist_target; agg=mean) * T(M) # sum over all slots, not mean 
    #matched_frac = T(n_matched) / T(M * BS)
    return ℒ_rec, ℒ_exist#, matched_frac
end
