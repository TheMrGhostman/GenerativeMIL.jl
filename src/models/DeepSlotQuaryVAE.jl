
struct DeepSlotQueryVAE{E<:PoolEncoder, PT<:SplitLayer, ZT<:Flux.Dense, DT<:TransformerDecoder,
        OT<:Flux.Dense, EXT<:Flux.Dense, QT<:AbstractMatrix{<:AbstractFloat}}
    encoder::E
    prior::PT
    z_to_hidden::ZT
    decoder::DT
    output_head::OT
    exist_head::EXT
    queries::QT
end

Flux.@layer DeepSlotQueryVAE

function DeepSlotQueryVAE(embed_dim::Int, hidden_dim::Int, heads::Int, n_slots::Int, z_dim::Int, m_z::Int, n_layers::Int; pma_attention_fn::Function=attention, activation::Function=relu)
    prepool = Flux.Dense(embed_dim, hidden_dim, activation)
    pooling = PMA(m_z, hidden_dim, heads; attention_fn=pma_attention_fn) # m_z induced points -> Z is a set of m_z tokens, not one vector
    postpool = Flux.Dense(hidden_dim, hidden_dim, activation)
    encoder = PoolEncoder(prepool, pooling, postpool)

    prior = SplitLayer(hidden_dim, (z_dim, z_dim), (identity, Flux.softplus))
    z_to_hidden = Flux.Dense(z_dim, hidden_dim)

    self_attns = [MultiheadAttentionBlock(hidden_dim, heads; attention_fn=attention) for _ in 1:n_layers]
    cross_attns = [MultiheadAttentionBlock(hidden_dim, heads; attention_fn=attention) for _ in 1:n_layers]
    decoder = TransformerDecoder(self_attns, cross_attns)

    output_head = Flux.Dense(hidden_dim, embed_dim)
    exist_head = Flux.Dense(hidden_dim, 1)

    queries = randn(Float32, hidden_dim, n_slots)

    return DeepSlotQueryVAE(encoder, prior, z_to_hidden, decoder, output_head, exist_head, queries)
end

function (m::DeepSlotQueryVAE)(x::AbstractArray{T,3}, x_mask::Union{AbstractArray{Bool},Nothing}=nothing) where T <: AbstractFloat
    bs = size(x, ndims(x))
    h = m.encoder(x, x_mask)                # (hidden, m_z, bs)
    μ_z, Σ_z = m.prior(h)                     # (z_dim, m_z, bs) each
    z = μ_z + Σ_z .* MLUtils.randn_like(μ_z)
    Z = m.z_to_hidden(z)                      # (hidden, m_z, bs)

    slots = repeat(m.queries, 1, 1, bs)       # (hidden, n_slots, bs)
    slots = m.decoder(slots, Z)               # real (non-degenerate) cross-attention when m_z > 1

    x̂ = m.output_head(slots)                   # (embed_dim, n_slots, bs)
    logits_exist = m.exist_head(slots)          # (1, n_slots, bs)
    return x̂, logits_exist, μ_z, Σ_z
end

function generate(m::DeepSlotQueryVAE, n_samples::Int; m_z::Int)
    z_dim = size(m.prior.μ.weight, 1)
    z = randn(Float32, z_dim, m_z, n_samples)
    Z = m.z_to_hidden(z)
    slots = repeat(m.queries, 1, 1, n_samples)
    slots = m.decoder(slots, Z)
    x̂ = m.output_head(slots)
    logits_exist = m.exist_head(slots)
    return x̂, logits_exist
end

function elbo_with_logging(model::DeepSlotQueryVAE, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}, logpdf::Function=pairwise_logitcrossentropy; β::AbstractFloat=1f0, λ_exist::AbstractFloat=1f0, kwargs...) where T <: AbstractFloat
    x̂, logits_exist, μ_z, Σ_z = model(x, x_mask)
    ℒ_rec, ℒ_exist = hungarian_matching_loss(x̂, x, x_mask, logits_exist, logpdf;) # FIXME / TODO: add hungarian matching loss into GenerativeMIL.jl and make it a separate function, so that it can be used in other models as well
    ℒ_kld = kl_divergence(μ_z, Σ_z)
    ℒ = ℒ_rec + T(λ_exist) * ℒ_exist + T(β) * ℒ_kld
    return ℒ, (ℒ=ℒ, ℒ_rec=ℒ_rec, ℒ_exist=ℒ_exist, ℒ_kld=ℒ_kld, β=β)
end

function optim_step(model::DeepSlotQueryVAE, batch::Tuple, opt::NamedTuple, logpdf, device::Function=cpu; β::AbstractFloat=1f0, λ_exist::AbstractFloat=1f0, kwargs...)
    x, x_mask = device.(batch)
    (loss, logs), (∇model, ∇x) = Zygote.withgradient(model, x) do m, xx
        elbo_with_logging(m, xx, x_mask, logpdf; β=β, λ_exist=λ_exist, kwargs...)
    end
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

function valid_step(model::DeepSlotQueryVAE, dataloader::DataLoader, logpdf; β::AbstractFloat=1f0, λ_exist::AbstractFloat=1f0, device::Function=cpu, kwargs...)
    ℒ, ℒ_rec, ℒ_exist, ℒ_kld, matched = 0f0, 0f0, 0f0, 0f0, 0f0
    for (x, x_mask) in dataloader
        x, x_mask = device(x), device(x_mask)
        loss, logs = elbo_with_logging(model, x, x_mask, logpdf; β=β, λ_exist=λ_exist, kwargs...)
        ℒ += loss; ℒ_rec += logs.ℒ_rec; ℒ_exist += logs.ℒ_exist; ℒ_kld += logs.ℒ_kld;
    end
    n = length(dataloader)
    return (ℒᵥ=ℒ/n, ℒᵥ_rec=ℒ_rec/n, ℒᵥ_exist=ℒ_exist/n, ℒᵥ_kld=ℒ_kld/n), ℒ/n
end