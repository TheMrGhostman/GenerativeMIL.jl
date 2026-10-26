
struct DeepSlotQueryVAE{E<:PoolEncoder, PT<:SplitLayer, ZT<:Flux.Dense, DT<:TransformerDecoder,
        OT<:Union{Flux.Dense, Flux.Chain}, EXT<:Union{Flux.Dense, Flux.Chain}, QT<:AbstractMatrix{<:AbstractFloat}}
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
    x̂, logits_exist = decode(m, z)
    return x̂, logits_exist, μ_z, Σ_z
end

"""
`decode(m::DeepSlotQueryVAE, z)`

Decoder half of the forward pass: from a latent `z (z_dim, m_z, bs)` to all `n_slots` slots.
Used for training (via the forward pass), reconstruction from μ_z and generation from z ~ N(0, I).
A slot exists if its `logits_exist > 0`.

Returns `(x̂ (embed_dim, n_slots, bs), logits_exist (1, n_slots, bs))`.
"""
function decode(m::DeepSlotQueryVAE, z::AbstractArray{T,3}) where T <: AbstractFloat
    bs = size(z, 3)
    Z = m.z_to_hidden(z)                      # (hidden, m_z, bs)
    slots = repeat(m.queries, 1, 1, bs)       # (hidden, n_slots, bs)
    slots = m.decoder(slots, Z)               # real (non-degenerate) cross-attention when m_z > 1
    x̂ = m.output_head(slots)                   # (embed_dim, n_slots, bs)
    logits_exist = m.exist_head(slots)          # (1, n_slots, bs)
    return x̂, logits_exist
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
    return ℒ, (ℒ=ℒ, ℒ_rec=ℒ_rec, ℒ_exist=ℒ_exist, ℒ_kld=ℒ_kld, β=β, λ=λ_exist)
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

# kwargs constructor (config-driven training). Mirrors `build_model` from
# experiments/.../DeepSlotQueryVAE_Categorical.jl: MLP pre/post-pool layers in the encoder and MLP
# output/existence heads, unlike the single-Dense positional constructor above. `n_slots` defaults
# to `n_max` (the data's padded slot count), so the model can emit as many elements as a bag holds.
function dsqvae_constructor_from_named_tuple(; idim::Int, hdim::Int, heads::Int, zdim::Int, m_z::Int=1, n_max::Int=0,
    n_slots::Int=n_max, n_layers::Int=2, att_layers::Int=2, out_layers::Int=2, ext_layers::Int=1, activation=relu,
    init_seed=nothing, kwargs...)
    n_slots > 0 || throw(ArgumentError("n_slots must be given (or n_max passed from the data)"))
    activation = activation isa Function ? activation : eval(:($(Symbol(activation))))
    (init_seed !== nothing) ? Random.seed!(init_seed) : nothing
    encoder = PoolEncoder(
        create_mlp(idim, hdim, n_layers, hdim, activation),
        PMA(m_z, hdim, heads),
        create_mlp(hdim, hdim, n_layers, hdim, activation)
    )
    prior = SplitLayer(hdim, (zdim, zdim), (identity, Flux.softplus))
    z_to_hidden = Flux.Dense(zdim, hdim)
    decoder = TransformerDecoder(
        [MultiheadAttentionBlock(hdim, heads; attention_fn=attention) for _ in 1:att_layers],
        [MultiheadAttentionBlock(hdim, heads; attention_fn=attention) for _ in 1:att_layers]
    )
    output_head = create_mlp(hdim, hdim, out_layers, idim, activation; out_identity=true)
    exist_head = create_mlp(hdim, hdim, ext_layers, 1, activation; out_identity=true)
    queries = randn(Float32, hdim, n_slots)
    model = DeepSlotQueryVAE(encoder, prior, z_to_hidden, decoder, output_head, exist_head, queries)
    (init_seed !== nothing) ? Random.seed!() : nothing
    return model
end

# Single forward pass returning (x̂, loss, logs) -- used by `reconstruction_check`. x̂ is returned as
# (x̂ = slot logits (D, n_slots, bs), logits_exist = (1, n_slots, bs)) so saved reconstructions keep
# which slots the model claims exist (logits_exist > 0).
function reconstruct_and_log(model::DeepSlotQueryVAE, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}, logpdf; β=1f0, λ_exist=1f0, kwargs...) where T <: AbstractFloat
    x̂, logits_exist, μ_z, Σ_z = model(x, x_mask)
    ℒ_rec, ℒ_exist = hungarian_matching_loss(x̂, x, x_mask, logits_exist, logpdf)
    ℒ_kld = kl_divergence(μ_z, Σ_z)
    ℒ = ℒ_rec + T(λ_exist) * ℒ_exist + T(β) * ℒ_kld
    return (x̂=x̂, logits_exist=logits_exist), ℒ, (ℒ=ℒ, ℒ_rec=ℒ_rec, ℒ_exist=ℒ_exist, ℒ_kld=ℒ_kld)
end
