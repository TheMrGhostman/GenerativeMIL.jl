using Revise
using DrWatson
@quickactivate

using Random
using Statistics
using JLD2
using MLUtils
using ProgressBars
using Flux
using Zygote
using Optimisers
using Hungarian
using CUDA

using GenerativeMIL
import GenerativeMIL: elbo_with_logging, optim_step, valid_step, _pairwise_sqdist_batched



_pairwise_sqeuclidean(a::AbstractMatrix{T}, b::AbstractMatrix{T}) where T<:AbstractFloat = begin
    a2 = sum(abs2, a, dims=1)
    b2 = sum(abs2, b, dims=1)
    max.(a2' .+ b2 .- 2 .* (a' * b), zero(T))
end

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
        C = _pairwise_sqeuclidean(x̂_cpu[:, :, b], x_cpu[:, gt_idx, b]) # (n_slots, n_gt), plain CPU Matrix
        assignment, _ = Hungarian.hungarian(C)
        pred_idx = findall(!=(0), assignment)
        matched_pred[b] = pred_idx
        matched_gt[b] = gt_idx[assignment[pred_idx]]
    end
    return matched_pred, matched_gt
end

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



# sim model DSQVAE 
model_cpu = cpu(model);
model_gpu = cu(model);

xc, xcm = first(dataloaders.train);
xg, xgm = xc |> cu, xcm |> cu

xc ≈ cpu(xg)

size(xc), size(xcm), size(xg), size(xgm)
# (D_z, N_slots, batch_size) 
# set decoder -> (D_z, NS, BS) -> (D_x, N, NS, BS) -> reshape -> (D_x, N, NS*BS)
# (D_x, N, NS*BS) -> since NS are independent when they get out of the dcoder, it is just reordering 
# N is theoretically N_i when cardinality is variable, but we can just pad with zeros to N_max and use a mask to ignore the padded values.

# x̂, logits_exist, μ_z, Σ_z
oc, lec, _, _ = model_cpu(xc, xcm)
og, leg, _, _ = model_gpu(xg, xgm)

ocg = cu(oc);



d, n_slots, bs = size(oc)
_, n_max, _ = size(xc)
## hungarian match
# x̂_cpu, x_cpu, mask_cpu = Array(x̂), Array(x), Array(x_mask)
matched_pred = Vector{Vector{Int}}(undef, bs)
matched_gt = Vector{Vector{Int}}(undef, bs)

b = 1
gt_idx = findall(vec(xcm[1, :, b])) # gt_idx = [1, 2, 3, 4, 5, 6, 7, 8]
gtm_idx = findall(xcm[1,:,:]) 

t1 = _pairwise_sqeuclidean(oc[:, :, b], xc[:, gt_idx, b])
t2 = _pairwise_sqdist_batched(oc, xc)
t1 ≈ t2[:,gt_idx,b] 

C_cpu = [_pairwise_sqeuclidean(oc[:, :, bb], xc[:, findall(vec(xcm[1, :, bb])), bb]) for bb in 1:bs]

reduce(hcat, C_cpu) ≈ t2[:, gtm_idx]

hungarian_match(oc, xc, xcm) == hungarian_match(ocg, xg, xgm)

t2[:, gtm_idx]

hungarian_match(oc, xc, xcm)
hungarian_match(og, xg, xgm)

hungarian_match(oc, xc, xcm)[1]
hungarian_match(og, xg, xgm)[1]

hungarian_match2(oc, xc, xcm)[1]


res_1 =[]
res_2 =[]
C2 = _pairwise_sqdist_batched(oc, xc)
C3 = _pairwise_sqdist_batched(og, xg)
C4 = cpu(C3)
for b in 1:bs
    gt_idx = findall(vec(mask_cpu[1, :, b]))
    C1 = _pairwise_sqeuclidean(oc[:, :, b], xc[:, gt_idx, b]) # (n_slots, n_gt), plain CPU Matrix
    dif = sum(abs2, C1 - C2[:, gt_idx, b])
    push!(res_1, dif)
    dif = sum(abs2, C1 - C4[:, gt_idx, b])
    push!(res_2, dif)
end
res_1 |> sum
res_2 |> sum


@benchmark hungarian_match($oc, $xc, $xcm)
@benchmark hungarian_match($ocg, $xg, $xgm)

function hungarian_match(x̂::CuArray{T,3}, x::CuArray{T,3}, x_mask::CuArray{Bool,3}) where T<:AbstractFloat
    _, n_slots, bs = size(x̂)
    matched_pred = Vector{Vector{Int}}(undef, bs)
    matched_gt = Vector{Vector{Int}}(undef, bs)

    C = _pairwise_sqdist_batched(x̂, x) # (n_slots, n_max, bs), plain GPU CuArray
    C = Array(C) # (n_slots, n_gt, bs), plain cpu matrix
    mask_cpu = Array(x_mask)

    for b in 1:bs
        gt_idx = findall(vec(mask_cpu[1, :, b]))
        if isempty(gt_idx)
            matched_pred[b], matched_gt[b] = Int[], Int[]
            continue
        end
        assignment, _ = Hungarian.hungarian(C[:, gt_idx, b])
        pred_idx = findall(!=(0), assignment)
        matched_pred[b] = pred_idx
        matched_gt[b] = gt_idx[assignment[pred_idx]]
    end
    return matched_pred, matched_gt
end


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
        C = _pairwise_sqeuclidean(x̂_cpu[:, :, b], x_cpu[:, gt_idx, b]) # (n_slots, n_gt), plain CPU Matrix
        assignment, _ = Hungarian.hungarian(C)
        pred_idx = findall(!=(0), assignment)
        matched_pred[b] = pred_idx
        matched_gt[b] = gt_idx[assignment[pred_idx]]
    end
    return matched_pred, matched_gt
end

pred_flat, gt_flat, exist_target = Zygote.@ignore begin
    mp, mg = hungarian_match(oc, xc, xcm)
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