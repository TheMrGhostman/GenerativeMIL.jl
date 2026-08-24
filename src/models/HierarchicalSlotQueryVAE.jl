struct CrossAttentionDecoder{CA<:Vector{<:MultiheadAttentionBlock}}
    cross_attns::CA
end

Flux.@layer CrossAttentionDecoder

function (m::CrossAttentionDecoder)(x::AbstractArray{T}, z::AbstractArray{T}) where T <: AbstractFloat
    for ca in m.cross_attns
        x = ca(x, z)
    end
    return x
end

struct TransformerDecoder{SA<:Vector{<:MultiheadAttentionBlock}, CA<:Vector{<:MultiheadAttentionBlock}}
    self_attns::SA
    cross_attns::CA
end

Flux.@layer TransformerDecoder

function (m::TransformerDecoder)(x::AbstractArray{T}, z::AbstractArray{T}) where T <: AbstractFloat
    for (sa, ca) in zip(m.self_attns, m.cross_attns)
        x = sa(x)
        x = ca(x, z)
    end
    return x
end


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

    dsq_output_head = Flux.Dense(dsq_hdim, dsq_emb_dim)
    dsq_exist_head  = Flux.Dense(dsq_hdim, 1)

    dsq_queries = randn(Float32, dsq_hdim, dsq_n_slots)

    dsq = DeepSlotQueryVAE(dsq_encoder, dsq_prior, dsq_z_to_hidden, dsq_self_attns, dsq_cross_attns, dsq_output_head, dsq_exist_head, dsq_queries)

    decoder = CrossAttentionDecoder(
        [MultiheadAttentionBlock(dsq_emb_dim, d_heads; activation=activation, attention_fn=attention) for _ in 1:d_mha_layers]
    )

    output = create_mlp(dsq_emb_dim, o_hdim, o_depth, idim, output_activation)

    return HierarchicalSlotQueryVAE(encoder, dsq, decoder, output)
end

function build_hierarchical_slot_query_vae(D, M, BS)
    # D: data dimension
    # N: number of points per bag
    # L: number of bags
    # M: number of slots
    # BS: batch size

    prepool = Flux.Dense(D, 16, relu)
    pooling = PMA(1, 16, 4; attention_fn = slot_attention) # m_z induced points -> Z is a set of m_z tokens, not one vector
    postpool = Flux.Dense(16, 16, relu)
    encoder = PoolEncoder(prepool, pooling, postpool)

    deep_slot_query = DeepSlotQueryVAE(16, 64, 4, M, 16, 2, 2)

    decoder = MultiheadAttentionBlock(16, 4; activation=relu, attention_fn=attention)
    output = Flux.Dense(16, D)

end
