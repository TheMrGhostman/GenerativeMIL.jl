
struct HierarchicalSlotQueryVAE{E<:PoolEncoder, DSQ<:DeepSlotQueryVAE, D, O<:Flux.Dense}
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

    h = m.encoder(x_reshaped) #TODO: correct masking and add mask here
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

function elbo_with_logging(model::HierarchicalSlotQueryVAE, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}; β::AbstractFloat=1f0, λ_exist::AbstractFloat=1f0, kwargs...) where T <: AbstractFloat

    ŷ, logits_exist, μ_z, Σ_z = model(x, x_mask)
    ℒ_rec, ℒ_exist = hungarian_matching_loss(ŷ, x, x_mask, logits_exist)
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
    (loss, logs), (∇model) = Zygote.withgradient(model) do m
        elbo_with_logging(m, x, x_mask; β=β, λ_exist=λ_exist)
    end
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
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

    return HierarchicalSlotQueryVAE(encoder, deep_slot_query, decoder, output)
end
