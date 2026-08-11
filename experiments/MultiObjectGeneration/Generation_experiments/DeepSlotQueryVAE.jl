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

using GenerativeMIL
import GenerativeMIL: elbo_with_logging, optim_step, valid_step

# =====================================================================================
# Same outer-set-only prototype as first_test.jl, but ablating two architecture ideas
# discussed after that run:
#   1) a DEEPER decoder: n_layers rounds of (self-attention among slots, cross-attention
#      to the global code), instead of just one round -- gives slots more iterations to
#      negotiate/specialize before being read out (this is what real DETR's 6-layer
#      decoder is doing; a single layer is known to underperform).
#   2) a TENSOR latent instead of a single vector: Z = {z1,...,z_mz} via PMA(m_z, ...)
#      instead of PMA(1, ...). With m_z=1 cross-attention is degenerate (softmax over one
#      key is always weight 1, so every slot gets an identical copy of Z); with m_z>1 it
#      becomes real attention -- different slots can pull different combinations of the
#      m_z summary tokens.
# Struct is named DeepSlotQueryVAE (not SlotQueryVAE) so this file can be `include`d or
# run in the same session as first_test.jl without a struct-redefinition clash.
# =====================================================================================

const EMBED_RUN_ID = 394507 # z_dim = 8 VAE pretrain run

data = load(datadir("MultiObjectGeneration", "vae_mnist_$(EMBED_RUN_ID)", "predictions", "predictions.jld2"))
μs, ys = Array(Float32.(data["μs"])), Int.(data["ys"])
const EMBED_DIM = size(μs, 1)

Random.seed!(20260810)
perm = randperm(length(ys))
n_valid_pool = 6000
valid_idx, train_idx = perm[1:n_valid_pool], perm[n_valid_pool+1:end]
μs_train, ys_train = μs[:, train_idx], ys[train_idx]
μs_valid, ys_valid = μs[:, valid_idx], ys[valid_idx]

# -------------------------------------------------------------------------------------
# Synthetic bag ("outer set") sampling with controlled duplicate injection -- identical
# to first_test.jl.
# -------------------------------------------------------------------------------------

function sample_bag_indices(class_pool::Dict{Int,Vector{Int}};
        n_min::Int=4, n_max::Int=12, p_force_duplicate::Float64=0.6, p_exact_duplicate::Float64=0.3)
    n = rand(n_min:n_max)
    idx = Int[]
    if rand() < p_force_duplicate
        c = rand(collect(keys(class_pool)))
        i1 = rand(class_pool[c])
        i2 = rand() < p_exact_duplicate ? i1 : rand(class_pool[c]) # exact vs same-class-different-instance
        append!(idx, (i1, i2))
    end
    while length(idx) < n
        c = rand(collect(keys(class_pool)))
        push!(idx, rand(class_pool[c]))
    end
    return idx[1:n]
end

function make_bag_dataset(μs::AbstractMatrix{Float32}, ys::Vector{Int}, n_bags::Int, N_max::Int; kwargs...)
    class_pool = Dict(c => findall(==(c), ys) for c in unique(ys))
    d = size(μs, 1)
    x = zeros(Float32, d, N_max, n_bags)
    mask = falses(1, N_max, n_bags)
    labels = Vector{Vector{Int}}(undef, n_bags)
    for b in 1:n_bags
        idx = sample_bag_indices(class_pool; n_max=N_max, kwargs...)
        n_b = length(idx)
        x[:, 1:n_b, b] .= μs[:, idx]
        mask[1, 1:n_b, b] .= true
        labels[b] = ys[idx]
    end
    return x, mask, labels
end

const N_MAX = 12

x_train, mask_train, labels_train = make_bag_dataset(μs_train, ys_train, 8000, N_MAX)
x_valid, mask_valid, labels_valid = make_bag_dataset(μs_valid, ys_valid, 800, N_MAX)

dataloaders = (
    train = DataLoader((x_train, mask_train), batchsize=128, shuffle=true, partial=true),
    valid = DataLoader((x_valid, mask_valid), batchsize=128, shuffle=false, partial=true),
)

# =====================================================================================
# DeepSlotQueryVAE -- DETR-style fixed-learned-query decoder, deepened + multi-token Z.
# =====================================================================================

struct DeepSlotQueryVAE{E<:PoolEncoder, PT<:SplitLayer, ZT<:Flux.Dense, SAT<:MultiheadAttentionBlock,
        CAT<:MultiheadAttentionBlock, OT<:Flux.Dense, EXT<:Flux.Dense, QT<:AbstractMatrix{<:AbstractFloat}}
    encoder::E
    prior::PT
    z_to_hidden::ZT
    self_attns::Vector{SAT}
    cross_attns::Vector{CAT}
    output_head::OT
    exist_head::EXT
    queries::QT
end

Flux.@layer DeepSlotQueryVAE

function DeepSlotQueryVAE(embed_dim::Int, hidden_dim::Int, heads::Int, n_slots::Int, z_dim::Int, m_z::Int, n_layers::Int; activation::Function=relu)
    prepool = Flux.Dense(embed_dim, hidden_dim, activation)
    pooling = PMA(m_z, hidden_dim, heads) # m_z induced points -> Z is a set of m_z tokens, not one vector
    postpool = Flux.Dense(hidden_dim, hidden_dim, activation)
    encoder = PoolEncoder(prepool, pooling, postpool)

    prior = SplitLayer(hidden_dim, (z_dim, z_dim), (identity, Flux.softplus))
    z_to_hidden = Flux.Dense(z_dim, hidden_dim)

    self_attns = [MultiheadAttentionBlock(hidden_dim, heads; attention_fn=attention) for _ in 1:n_layers]
    cross_attns = [MultiheadAttentionBlock(hidden_dim, heads; attention_fn=attention) for _ in 1:n_layers]

    output_head = Flux.Dense(hidden_dim, embed_dim)
    exist_head = Flux.Dense(hidden_dim, 1)

    queries = randn(Float32, hidden_dim, n_slots)

    return DeepSlotQueryVAE(encoder, prior, z_to_hidden, self_attns, cross_attns, output_head, exist_head, queries)
end

function (m::DeepSlotQueryVAE)(x::AbstractArray{T,3}, x_mask::Union{AbstractArray{Bool},Nothing}=nothing) where T <: AbstractFloat
    bs = size(x, ndims(x))
    h = m.encoder(x, x_mask)                # (hidden, m_z, bs)
    μ_z, Σ_z = m.prior(h)                     # (z_dim, m_z, bs) each
    z = μ_z + Σ_z .* MLUtils.randn_like(μ_z)
    Z = m.z_to_hidden(z)                      # (hidden, m_z, bs)

    slots = repeat(m.queries, 1, 1, bs)       # (hidden, n_slots, bs)
    for (sa, ca) in zip(m.self_attns, m.cross_attns)
        slots = sa(slots)
        slots = ca(slots, Z)                  # real (non-degenerate) attention when m_z > 1
    end

    x̂ = m.output_head(slots)                   # (embed_dim, n_slots, bs)
    logits_exist = m.exist_head(slots)          # (1, n_slots, bs)
    return x̂, logits_exist, μ_z, Σ_z
end

function generate(m::DeepSlotQueryVAE, n_samples::Int; m_z::Int)
    z_dim = size(m.prior.μ.weight, 1)
    z = randn(Float32, z_dim, m_z, n_samples)
    Z = m.z_to_hidden(z)
    slots = repeat(m.queries, 1, 1, n_samples)
    for (sa, ca) in zip(m.self_attns, m.cross_attns)
        slots = sa(slots)
        slots = ca(slots, Z)
    end
    x̂ = m.output_head(slots)
    logits_exist = m.exist_head(slots)
    return x̂, logits_exist
end

# --- Hungarian-matched reconstruction + existence loss (identical to first_test.jl) ---

_pairwise_sqeuclidean(a::AbstractMatrix{T}, b::AbstractMatrix{T}) where T<:AbstractFloat = begin
    a2 = sum(abs2, a, dims=1)
    b2 = sum(abs2, b, dims=1)
    max.(a2' .+ b2 .- 2 .* (a' * b), zero(T))
end

function hungarian_match(x̂::AbstractArray{T,3}, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    _, n_slots, bs = size(x̂)
    matched_pred = Vector{Vector{Int}}(undef, bs)
    matched_gt = Vector{Vector{Int}}(undef, bs)
    for b in 1:bs
        gt_idx = findall(vec(x_mask[1, :, b]))
        if isempty(gt_idx)
            matched_pred[b], matched_gt[b] = Int[], Int[]
            continue
        end
        C = _pairwise_sqeuclidean(x̂[:, :, b], x[:, gt_idx, b]) # (n_slots, n_gt)
        assignment, _ = Hungarian.hungarian(C)
        pred_idx = findall(!=(0), assignment)
        matched_pred[b] = pred_idx
        matched_gt[b] = gt_idx[assignment[pred_idx]]
    end
    return matched_pred, matched_gt
end

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

# --- training interface: same elbo/optim_step/valid_step pattern as src/models --------

function elbo_with_logging(model::DeepSlotQueryVAE, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}; β::AbstractFloat=1f0, λ_exist::AbstractFloat=1f0, kwargs...) where T <: AbstractFloat
    x̂, logits_exist, μ_z, Σ_z = model(x, x_mask)
    ℒ_rec, ℒ_exist, matched_frac = hungarian_matching_loss(x̂, logits_exist, x, x_mask)
    ℒ_kld = kl_divergence(μ_z, Σ_z)
    ℒ = ℒ_rec + T(λ_exist) * ℒ_exist + T(β) * ℒ_kld
    return ℒ, (ℒ=ℒ, ℒ_rec=ℒ_rec, ℒ_exist=ℒ_exist, ℒ_kld=ℒ_kld, β=β, matched_frac=matched_frac)
end

function optim_step(model::DeepSlotQueryVAE, batch::Tuple, opt::NamedTuple; β::AbstractFloat=1f0, λ_exist::AbstractFloat=1f0, kwargs...)
    x, x_mask = batch
    (loss, logs), (∇model, ∇x) = Zygote.withgradient(model, x) do m, xx
        elbo_with_logging(m, xx, x_mask; β=β, λ_exist=λ_exist, kwargs...)
    end
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

function valid_step(model::DeepSlotQueryVAE, dataloader; β::AbstractFloat=1f0, λ_exist::AbstractFloat=1f0, kwargs...)
    ℒ, ℒ_rec, ℒ_exist, ℒ_kld, matched = 0f0, 0f0, 0f0, 0f0, 0f0
    for (x, x_mask) in dataloader
        loss, logs = elbo_with_logging(model, x, x_mask; β=β, λ_exist=λ_exist, kwargs...)
        ℒ += loss; ℒ_rec += logs.ℒ_rec; ℒ_exist += logs.ℒ_exist; ℒ_kld += logs.ℒ_kld; matched += logs.matched_frac
    end
    n = length(dataloader)
    return (ℒᵥ=ℒ/n, ℒᵥ_rec=ℒ_rec/n, ℒᵥ_exist=ℒ_exist/n, ℒᵥ_kld=ℒ_kld/n, matched_fracᵥ=matched/n), ℒ/n
end

# =====================================================================================
# Train
# =====================================================================================

args = (;
    embed_dim = EMBED_DIM,
    hidden_dim = 64,
    heads = 4,
    n_slots = N_MAX,
    z_dim = 16,
    m_z = 3,       # NEW: number of latent summary tokens (was implicitly 1 in first_test.jl)
    n_layers = 3,  # NEW: number of stacked self+cross-attention rounds (was implicitly 1)
    β = 0.01f0,
    λ_exist = 2f0,
    epochs = 100,
)

model = DeepSlotQueryVAE(args.embed_dim, args.hidden_dim, args.heads, args.n_slots, args.z_dim, args.m_z, args.n_layers)
opt = Optimisers.setup(AdamW(; eta=1e-3, lambda=1e-4), model);

for epoch in 1:args.epochs
    logs = nothing
    for batch in tqdm(dataloaders.train)
        global model, opt # top-level nested-loop reassignment is ambiguous soft scope otherwise (Julia gotcha)
        model, opt, logs = optim_step(model, batch, opt; β=args.β, λ_exist=args.λ_exist)
    end
    vlogs, _ = valid_step(model, dataloaders.valid; β=args.β, λ_exist=args.λ_exist)
    println("Epoch $epoch | train: $(logs) | valid: $(vlogs)")
end

# =====================================================================================
# Diagnostics -- identical to first_test.jl, for direct comparison
# =====================================================================================

nearest_class(v::AbstractVector{Float32}, μs::AbstractMatrix{Float32}, ys::Vector{Int}) = ys[argmin(vec(sum(abs2, μs .- v, dims=1)))]

# 1) Reconstruction on a hand-picked duplicate-heavy bag: two "1"s + one "2" + one "7"
dup_idx = vcat(
    first(findall(==(1), ys_valid), 2),
    first(findall(==(2), ys_valid), 1),
    first(findall(==(7), ys_valid), 1),
)
x_dup = zeros(Float32, EMBED_DIM, N_MAX, 1)
mask_dup = falses(1, N_MAX, 1)
x_dup[:, 1:length(dup_idx), 1] .= μs_valid[:, dup_idx]
mask_dup[1, 1:length(dup_idx), 1] .= true
println("\nGround truth bag classes: ", ys_valid[dup_idx])

x̂_dup, logits_dup, _, _ = model(x_dup, mask_dup)
mp, mg = hungarian_match(x̂_dup, x_dup, mask_dup)
matched_gt_for_slot = Dict(zip(mp[1], mg[1]))

println("\nAll $N_MAX slots (existence = σ(logit); >0.5 means the model thinks this slot is real):")
for slot in 1:N_MAX
    exist_prob = round(Flux.sigmoid(logits_dup[1, slot, 1]); digits=2)
    pred_class = nearest_class(x̂_dup[:, slot, 1], μs_valid, ys_valid)
    if haskey(matched_gt_for_slot, slot)
        gt = matched_gt_for_slot[slot]
        gt_class = ys_valid[dup_idx[gt]]
        dist = round(sqrt(sum(abs2, x̂_dup[:, slot, 1] .- x_dup[:, gt, 1])); digits=3)
        println("slot $slot [MATCHED,   existence=$exist_prob] -> gt class $gt_class | nearest-neighbor predicted class $pred_class | L2 dist $dist")
    else
        println("slot $slot [unmatched, existence=$exist_prob] -> nearest-neighbor predicted class $pred_class (no ground truth to compare against)")
    end
end

σ.(logits_dup) .>= 0.5
sum(σ.(logits_dup) .>= 0.5)

# 1b) Threshold-only view: ignore Hungarian matching entirely and just take whichever
# slots the model itself thinks are real (existence >= 0.5), then find each one's closest
# ground-truth element directly. This is a different check than the matched view above --
# Hungarian always forces a clean one-to-one assignment, so it can hide failure modes like
# several "active" slots collapsing onto the same object, or the model activating a
# different number of slots than the true cardinality (4 here).
active_slots = findall(vec(σ.(logits_dup[1, :, 1]) .>= 0.5f0))
println("\nThreshold-only view: $(length(active_slots))/$N_MAX slots predicted active (existence >= 0.5), vs $(length(dup_idx)) true objects")
for slot in active_slots
    gt_dists = [sqrt(sum(abs2, x̂_dup[:, slot, 1] .- x_dup[:, g, 1])) for g in 1:length(dup_idx)]
    closest_gt = argmin(gt_dists)
    closest_gt_class = ys_valid[dup_idx[closest_gt]]
    println("slot $slot -> closest ground truth: class $closest_gt_class (gt position $closest_gt) | L2 dist $(round(gt_dists[closest_gt]; digits=3))")
end

# 2) Slot collapse check: mean pairwise distance among the n_slots outputs (unconditional)
x̂_gen, logits_gen = generate(model, 200; m_z=args.m_z)
collapse_dists = Float32[]
for b in 1:size(x̂_gen, 3)
    S = x̂_gen[:, :, b]
    for i in 1:size(S, 2), j in i+1:size(S, 2)
        push!(collapse_dists, sqrt(sum(abs2, S[:, i] .- S[:, j])))
    end
end
println("\nMean pairwise slot distance (unconditional generation): ", mean(collapse_dists), " (near 0 => collapse)")

# 3) Unconditional generation: decode active slots (existence logit > 0) to class labels
println("\nExample generated sets:")
for b in 1:10
    active = findall(vec(logits_gen[1, :, b]) .> 0)
    classes = [nearest_class(x̂_gen[:, s, b], μs_valid, ys_valid) for s in active]
    println("sample $b -> $(sort(classes))")
end
