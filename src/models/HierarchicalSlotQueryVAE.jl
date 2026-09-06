struct HierarchicalSlotQueryVAE{E<:PoolEncoder, DSQ<:DeepSlotQueryVAE, D, O} <: AbstractGenModel
    encoder::E
    deep_slot_query::DSQ
    decoder::D
    output::O
end

Flux.@layer HierarchicalSlotQueryVAE

# this works only for outter encoder with one vecotr as latent. not tensor!!!!!!
function (m::HierarchicalSlotQueryVAE)(x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}) where T <: AbstractFloat
    dₓ, n, l, bs = size(x)
    x_reshaped = reshape(x, dₓ, n, l*bs) # (dₓ, n, l*bs)

    h = m.encoder(x_reshaped) #NOTE: I do not have to add mask here! since l became part of batch size, attention or pooling will ignore it as both are interested in first 2 dimensions. 
    h = multiplicative_masking(reshape(h, :, 1, l, bs), x_mask) # 1 is because m_z of PMA is 1 for this version of encoder
    h = dropdims(h, dims=2) # (hidden, l, bs)
    h_mask = isnothing(x_mask) ? nothing : dropdims(x_mask, dims=2) # (1, l, bs)
    x̂, logits_exist, μ_z, Σ_z = m.deep_slot_query(h, h_mask)
    dₕ, n_slots, bs = size(x̂)

    prior = MLUtils.randn_like(x, (dₕ, n, n_slots * bs));
    x̂ = reshape(x̂, dₕ, 1, n_slots * bs) # (dₕ, 1, n_slots * bs)
    x̂ = m.decoder(prior, x̂) # (dₕ, n, n_slots * bs)
    x̂ = reshape(x̂, dₕ, n, n_slots, bs)
    x̂ = m.output(x̂)
    return x̂, logits_exist, μ_z, Σ_z
end

function elbo_with_logging(model::HierarchicalSlotQueryVAE, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}, logpdf::Function=chamfer_pairwise_distance; β::AbstractFloat=1f0, λ_exist::AbstractFloat=1f0, kwargs...) where T <: AbstractFloat

    ŷ, logits_exist, μ_z, Σ_z = model(x, x_mask)
    ℒ_rec, ℒ_exist = hungarian_matching_loss(ŷ, x, x_mask, logits_exist, logpdf)
    ℒₖₗ = mean(kl_divergence(μ_z, Σ_z))
    ℒ = ℒ_rec + λ_exist * ℒ_exist + β * ℒₖₗ
    logs = (ℒ = ℒ, ℒ_rec = ℒ_rec, ℒ_exist = ℒ_exist, ℒₖₗ = ℒₖₗ, λ_exist = λ_exist, β = β)
    return ℒ, logs
end

function optim_step(model::HierarchicalSlotQueryVAE, batch::Tuple{X, M, L}, opt::NamedTuple, logpdf, device::Function=cpu; β=1f0, λ_exist=1f0, kwargs...) where {X <: AbstractArray{<:AbstractFloat,4}, M <: AbstractArray{Bool,4}, L <: AbstractVector{Int}}
    x, x_mask, _ = batch
    return optim_step(model, (x, x_mask), opt, logpdf, device; β=β, λ_exist=λ_exist, kwargs...)
end

function optim_step(model::HierarchicalSlotQueryVAE, batch::Tuple{X, M}, opt::NamedTuple, logpdf, device::Function=cpu; β=1f0, λ_exist=1f0, kwargs...) where {X <: AbstractArray{<:AbstractFloat,4}, M <: AbstractArray{Bool,4}}
    x, x_mask = device.(batch)
    (loss, logs), (∇model,) = Zygote.withgradient(model) do m
        elbo_with_logging(m, x, x_mask, logpdf; β=β, λ_exist=λ_exist)
    end
    #return loss, logs, ∇model
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

function valid_step(model::HierarchicalSlotQueryVAE, dataloader::DataLoader, logpdf; β=1f0, λ_exist=1f0, device::Function=cpu, kwargs...)
    ℒ, ℒ_rec, ℒₖₗ, ℒ_exist = 0f0, 0f0, 0f0, 0f0
    for batch in dataloader
        x, x_mask = length(batch) == 3 ? (batch[1], batch[2]) : batch # TODO: make it more robust to different batch formats
        x, x_mask = device(x), device(x_mask)
        loss, logs = elbo_with_logging(model, x, x_mask, logpdf; β=β, λ_exist=λ_exist)

        ℒ += loss
        ℒ_rec += logs.ℒ_rec
        ℒₖₗ += logs.ℒₖₗ
        ℒ_exist += logs.ℒ_exist
    end

    n = length(dataloader)
    logs = (; ℒᵥ = ℒ/n, ℒᵥ_rec = ℒ_rec/n, ℒᵥ_exist = ℒ_exist/n, ℒᵥₖₗ = ℒₖₗ/n,)
    return logs, ℒ/n
end

function HierarchicalSlotQueryVAE(; 
    idim::Int, e_hdim::Int, e_pre_depth::Int, e_pma_heads::Int, e_post_depth::Int, e_pma_attention_fn, dsq_emb_dim::Int, dsq_hdim::Int, dsq_pre_depth::Int, dsq_indices::Int, dsq_heads::Int, dsq_post_depth::Int, dsq_pma_attention_fn, dsq_zdim::Int, dsq_n_slots::Int, dsq_n_attn_layers::Int, d_mha_layers::Int, d_heads::Int, o_hdim::Int, o_depth::Int, activation=relu, output_activation=identity, kwargs...
)

    e_pma_attention_fn = (e_pma_attention_fn isa String) ? eval(Symbol(e_pma_attention_fn)) : e_pma_attention_fn
    dsq_pma_attention_fn = (dsq_pma_attention_fn isa String) ? eval(Symbol(dsq_pma_attention_fn)) : dsq_pma_attention_fn
    activation = (activation isa String) ? eval(Symbol(activation)) : activation
    output_activation = (output_activation isa String) ? eval(Symbol(output_activation)) : output_activation

    # first (outer) stage bag encoder.
    encoder = PoolEncoder(
        create_mlp(idim, e_hdim, e_pre_depth, e_hdim, activation),
        PMA(1, e_hdim, e_pma_heads; attention_fn=e_pma_attention_fn),
        create_mlp(e_hdim, e_hdim, e_post_depth, e_hdim, activation)
    )

    # second (inner) stage encoder, deep slot query encoder
    dsq_encoder = PoolEncoder(
        create_mlp(dsq_emb_dim, dsq_hdim, dsq_pre_depth, dsq_hdim, activation),
        PMA(dsq_indices, dsq_hdim, dsq_heads; attention_fn=dsq_pma_attention_fn),
        create_mlp(dsq_hdim, dsq_hdim, dsq_post_depth, dsq_hdim, activation)
    )

    dsq_prior = SplitLayer(dsq_hdim, (dsq_zdim, dsq_zdim), (identity, Flux.softplus))
    dsq_z_to_hidden = Flux.Dense(dsq_zdim, dsq_hdim)

    # second (inner) stage decoder, deep slot query decoder
    dsq_self_attns  = [MultiheadAttentionBlock(dsq_hdim, dsq_heads; attention_fn=attention) for _ in 1:dsq_n_attn_layers]
    dsq_cross_attns = [MultiheadAttentionBlock(dsq_hdim, dsq_heads; attention_fn=attention) for _ in 1:dsq_n_attn_layers]
    dsq_decoder = TransformerDecoder(dsq_self_attns, dsq_cross_attns)

    dsq_output_head = Flux.Dense(dsq_hdim, dsq_emb_dim)
    dsq_exist_head  = Flux.Dense(dsq_hdim, 1)

    dsq_queries = randn(Float32, dsq_hdim, dsq_n_slots)

    dsq = DeepSlotQueryVAE(dsq_encoder, dsq_prior, dsq_z_to_hidden, dsq_decoder, dsq_output_head, dsq_exist_head, dsq_queries)

    decoder = CrossAttentionDecoder(
        [MultiheadAttentionBlock(dsq_emb_dim, d_heads; activation=activation, attention_fn=attention) for _ in 1:d_mha_layers]
    )

    output = create_mlp(dsq_emb_dim, o_hdim, o_depth, idim, output_activation)

    return HierarchicalSlotQueryVAE(encoder, dsq, decoder, output)
end


"""
`migrate_dsq_state(dsq_state::NamedTuple)`

Old (pre-refactor) `deep_slot_query` state had `self_attns`/`cross_attns` as flat sibling
keys. The current `DeepSlotQueryVAE` nests them under a single `decoder` (`TransformerDecoder`)
field instead. Re-nests an old state into the new shape; a no-op if the state is already in the
new shape (already has a `decoder` key).
"""
function migrate_dsq_state(dsq_state::NamedTuple)
    haskey(dsq_state, :decoder) && return dsq_state
    return (
        encoder = dsq_state.encoder,
        prior = dsq_state.prior,
        z_to_hidden = dsq_state.z_to_hidden,
        decoder = (self_attns = dsq_state.self_attns, cross_attns = dsq_state.cross_attns),
        output_head = dsq_state.output_head,
        exist_head = dsq_state.exist_head,
        queries = dsq_state.queries,
    )
end

