struct DiscreteBagVAE{E<:PoolEncoder, QP, HT<:Flux.Dense, D<:TransformerDecoder, OT<:Flux.Dense, CT} <: AbstractGenModel
    encoder::E
    query_prior::QP
    z_to_kv::HT
    decoder::D
    output_head::OT
    cardinality_head::CT
end

Flux.@layer DiscreteBagVAE

function DiscreteBagVAE(dₓ::Int, dₕ::Int, m_z::Int, d_z::Int, n_heads::Int, n_layers::Int, att_layers::Int, cp_layers::Int, activation::Function=relu, dropout::Real=0.0, gaussian_mlp::Bool=true, cp_pre_layers::Int=1; card_pooling::Union{Nothing, Symbol}=nothing)

    pool_fn = PMA(m_z, dₕ, n_heads)
    pool_fn = isnothing(card_pooling) ? pool_fn :  CardinalityPooling(pool_fn, card_pooling)
    cfd = isnothing(card_pooling) ? 0 : feature_dim(pool_fn)                                     # cardinality feature dimension

    encoder = PoolEncoder(
        create_mlp(dₓ, dₕ, n_layers, dₕ, activation),
        pool_fn,
        gaussian_mlp ? create_gaussian_mlp(dₕ + cfd, dₕ, n_layers, d_z, activation; softplus_=true) : create_mlp(dₕ + cfd, dₕ, n_layers, d_z, activation) 
    )

    query_prior = MLUtils.randn_like
    z_to_kv = Flux.Dense(d_z, dₕ)
    decoder = TransformerDecoder(
        [MultiheadAttentionBlock(dₕ, n_heads; attention_fn=attention) for _ in 1:att_layers],
        [MultiheadAttentionBlock(dₕ, n_heads; attention_fn=attention) for _ in 1:att_layers]
    )
    
    output_head = Flux.Dense(dₕ, dₓ)
    # cardinality is a property of the whole latent set z (d_z, m_z, bs), one value per bag -> n̂ (1, 1, bs).
    # m_z = 1: a plain MLP on the single token; m_z > 1: pool the m_z tokens into one (PMA with a single
    # seed) first, so no token has to carry the cardinality on its own.
    cardinality_head = if m_z > 1
        PoolEncoder(
            with_dropout(create_mlp(d_z, dₕ, cp_pre_layers, dₕ, activation), dropout),
            PMA(1, dₕ, n_heads),
            with_dropout(create_mlp(dₕ, dₕ, cp_layers, 1, activation, out_identity=true), dropout)
        )
    else
        with_dropout(create_mlp(d_z, dₕ, cp_layers, 1, activation, out_identity=true), dropout)
    end

    return DiscreteBagVAE(encoder, query_prior, z_to_kv, decoder, output_head, cardinality_head)
end

produce_z(x::Tuple{T, T}) where T <: AbstractArray = x[1] .+ x[2] .* MLUtils.randn_like(x[1])
produce_z(x::AbstractArray) = x 

# The cardinality head regresses the min-max scaled cardinality c / n_max ∈ (0, 1] (n_max = padded
# slot count), not the raw count -- keeps the target near the head's initial output scale and the
# MSE comparable to ℒ_rec. See `cardinality_loss`.

# Converts the cardinality head's raw regression output n̂ (1, 1, bs) into an integer
# cardinality per batch element: averages over dim 2 (a no-op now that the head pools the m_z
# latent tokens itself, kept so it accepts per-token outputs too), applies the same softplus
# used in the training loss to keep the estimate non-negative, rescales by n_max, then rounds and
# clamps into a valid slot count in 1:n_max.
cardinality_from_nhat(n̂::AbstractArray{<:AbstractFloat,3}, n_max::Int) =
    clamp.(round.(Int, n_max .* dropdims(mean(Flux.softplus.(Array(n̂)), dims=2), dims=(1,2))), 1, n_max)

# cardinality prediction loss: MSE between softplus(n̂) (1, 1, bs) and each bag's true element
# count (sum of its mask, not the constant padded width) scaled by n_max = size(x_mask, 2).
# Also returns the mean absolute error in number of elements (not differentiated), for logging
# that stays comparable across λ.
function cardinality_loss(n̂::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}) where T <: AbstractFloat
    n_max = size(x_mask, 2)
    true_card = T.(sum(x_mask, dims=2))  # (1, 1, bs), same shape as n̂
    ĉ = Flux.softplus.(n̂)
    ℒ_card = Flux.mse(ĉ, true_card ./ n_max)
    card_mae = Zygote.@ignore mean(abs.(n_max .* mean(ĉ, dims=2) .- true_card))
    return ℒ_card, card_mae
end

# Builds a (1, n, bs) existence mask from the cardinality head's prediction n̂, mirroring
# x_mask's layout (first `card[b]` query slots existing, rest padding). CPU-only index
# bookkeeping (like hungarian_match) -- only meant to run at inference/generation time, never
# under gradient tracking, so callers should wrap it in Zygote.@ignore.
function pred_card_mask(n̂::AbstractArray{<:AbstractFloat,3}, n::Int, bs::Int)
    card = cardinality_from_nhat(n̂, n)
    mask = falses(1, n, bs)
    for (b, c) in enumerate(card)
        mask[1, 1:c, b] .= true
    end
    return mask
end


function (m::DiscreteBagVAE)(x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}; gt_card::Bool=true, no_grad_card::Bool=false) where T <: AbstractFloat
    hₑ = m.encoder(x, x_mask)                                     # (dₕ, m_z, bs) — mask padding out of the pooling attention
    z = produce_z(hₑ)                                            # (d_z, m_z, bs)
    x̂, _, n̂ = decode(m, z, size(x, 2); card_mask = gt_card ? x_mask : nothing, no_grad_card=no_grad_card)
    return x̂, hₑ, n̂
end

"""
`decode(m::DiscreteBagVAE, z, n_slots; card_mask=nothing)`

Decoder half of the forward pass: from a latent `z (d_z, m_z, bs)` to `n_slots` output slots.
Used for training (via the forward pass), reconstruction from μ_z and generation from z ~ N(0, I).

- `card_mask`: `(1, n_slots, bs)` mask of the slots to decode; `nothing` uses the cardinality
  head's own prediction (first `ĉ` slots, as at generation time).

Returns `(x̂ (dₓ, n_slots, bs), query_mask (1, n_slots, bs), n̂ (1, 1, bs))`.
"""
function decode(m::DiscreteBagVAE, z::AbstractArray{T,3}, n_slots::Int; card_mask=nothing, no_grad_card::Bool=false) where T <: AbstractFloat
    bs = size(z, 3)
    n̂ = m.cardinality_head(no_grad_card ? Zygote.dropgrad(z) : z)       # (1, 1, bs) — PoolEncoder head pools over the m_z tokens
    h = m.z_to_kv(z)                                             # (dₕ, m_z, bs)
    query_mask = if isnothing(card_mask)
        Zygote.@ignore begin
            mask_cpu = pred_card_mask(n̂, n_slots, bs)
            z isa CuArray ? CuArray(mask_cpu) : mask_cpu
        end
    else
        card_mask
    end
    q = m.query_prior(h, (size(h,1), n_slots, bs))               # (dₕ, n_slots, bs) - or this could be different prior
    x̂ = m.decoder(q, h, query_mask)                              # (dₕ, n_slots, bs) — masked self/cross-attention over query slots
    x̂ = m.output_head(x̂)                                         # (dₓ, n_slots, bs)
    return x̂, query_mask, n̂
end

# no_grad_card=true: the cardinality loss trains only the cardinality head, no gradient flows from it into z / the encoder
function elbo_with_logging(model::DiscreteBagVAE, x::AbstractArray{T, 3}, x_mask::AbstractArray{Bool, 3}, logpdf::Function; β=1f0, λ=1f0, no_grad_card::Bool=false, kwargs...) where T <: AbstractFloat
    x̂, hₑ, n̂ = model(x, x_mask; gt_card=true, no_grad_card=no_grad_card)  # training always decodes against ground-truth cardinality
    # without a gaussian encoder hₑ is a plain array, and destructuring it would silently take its first two elements
    hₑ isa Tuple || throw(ArgumentError("elbo_with_logging needs gaussian_mlp=true (got deterministic encoder output); use use_elbo=false"))
    μ_z, Σ_z = hₑ
    # KL divergence
    ℒₖₗ = GenerativeMIL.kl_divergence(μ_z, Σ_z) |> mean
    # reconstruction loss
    C = logpdf(x̂, x) # (N, N, BS)
    matched_indices, _ = Zygote.@ignore hungarian_match(C, x_mask, x_mask)
    n_matched = length(matched_indices)
    ℒ_rec = n_matched > 0 ? mean(C[matched_indices]) : zero(T)
    # cardinality prediction loss on the scaled target
    ℒ_card, card_mae = cardinality_loss(n̂, x_mask)
    # total objective
    ℒ = ℒ_rec + β * ℒₖₗ + λ * ℒ_card
    # logging
    logs = (ℒ = ℒ, ℒ_rec=ℒ_rec, ℒₖₗ=ℒₖₗ, ℒ_card=ℒ_card, card_mae=card_mae, β=β, λ=λ)
    return ℒ, logs
end

function loss_with_logging(model::DiscreteBagVAE, x::AbstractArray{T, 3}, x_mask::AbstractArray{Bool, 3}, logpdf::Function; λ=1f0, no_grad_card::Bool=false, kwargs...) where T <: AbstractFloat
    x̂, _, n̂ = model(x, x_mask; gt_card=true, no_grad_card=no_grad_card)  # training always decodes against ground-truth cardinality
    # reconstruction loss
    C = logpdf(x̂, x) # (N, N, BS)
    matched_indices, _ = Zygote.@ignore hungarian_match(C, x_mask, x_mask)
    n_matched = length(matched_indices)
    ℒ_rec = n_matched > 0 ? mean(C[matched_indices]) : zero(T)
    # cardinality prediction loss on the scaled target
    ℒ_card, card_mae = cardinality_loss(n̂, x_mask)
    # total objective
    ℒ = ℒ_rec + λ * ℒ_card
    # logging
    logs = (ℒ = ℒ, ℒ_rec=ℒ_rec, ℒ_card=ℒ_card, card_mae=card_mae, λ=λ)
    return ℒ, logs
end

function optim_step(model::DiscreteBagVAE, batch::Tuple{X, M}, opt::NamedTuple, logpdf, device::Function=cpu; use_elbo::Bool=true, β=1f0, λ=1f0, kwargs...) where {X <: AbstractArray{<:AbstractFloat,3}, M <: AbstractArray{Bool,3}}
    x, x_mask = batch
    x, x_mask = device(x), device(x_mask)
    (loss, logs), (∇model,) = Zygote.withgradient(model) do m
        use_elbo ? elbo_with_logging(m, x, x_mask, logpdf; β=β, λ=λ, kwargs...) : loss_with_logging(m, x, x_mask, logpdf; λ=λ, kwargs...)
    end
    #return loss, logs, ∇model
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

function valid_step(model::DiscreteBagVAE, dataloader::DataLoader, logpdf; use_elbo::Bool=true, β=1f0, λ=1f0, device::Function=cpu, kwargs...)
    ℒ, ℒ_rec, ℒₖₗ, ℒ_card, card_mae = 0f0, 0f0, 0f0, 0f0, 0f0
    Flux.testmode!(model, true)
    for batch in dataloader
        x, x_mask = length(batch) == 3 ? (batch[1], batch[2]) : batch # TODO: make it more robust to different batch formats
        x, x_mask = device(x), device(x_mask)
        loss, logs = use_elbo ? elbo_with_logging(model, x, x_mask, logpdf; β=β, λ=λ, kwargs...) : loss_with_logging(model, x, x_mask, logpdf; λ=λ, kwargs...)

        ℒ += loss
        ℒ_rec += logs.ℒ_rec
        haskey(logs, :ℒₖₗ) && (ℒₖₗ += logs.ℒₖₗ)
        ℒ_card += logs.ℒ_card
        card_mae += logs.card_mae
    end
    Flux.testmode!(model, false)
    n = length(dataloader)
    logs = (; ℒᵥ = ℒ/n, ℒᵥ_rec = ℒ_rec/n, ℒᵥₖₗ = ℒₖₗ/n, ℒᵥ_card = ℒ_card/n, ℒᵥ_card_mae = card_mae/n)
    return logs, ℒ/n
end


function discretebagvae_constructor_from_named_tuple(; idim::Int, hdim::Int, heads::Int, zdim::Int, m_z::Int=1, n_layers::Int=3,
    att_layers::Int=2, cp_layers::Int=3, cp_pre_layers::Int=1, activation=relu, dropout::Real=0.0, gaussian_mlp::Bool=true, card_pooling=nothing, init_seed=nothing, kwargs...)  # kwargs: other config keys (model_type, n_max, ...) are ignored
    activation = activation isa Function ? activation : eval(:($(Symbol(activation))))
    (init_seed !== nothing) ? Random.seed!(init_seed) : nothing
    card_pooling = isnothing(card_pooling) ? nothing : Symbol(card_pooling)
    model = DiscreteBagVAE(idim, hdim, m_z, zdim, heads, n_layers, att_layers, cp_layers, activation, dropout, gaussian_mlp, cp_pre_layers, card_pooling=card_pooling)
    (init_seed !== nothing) ? Random.seed!() : nothing
    return model
end

######################################
### Score functions and evaluation ###
######################################

function reconstruct(model::DiscreteBagVAE, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}; gt_card::Bool=true, kwargs...) where T <: AbstractFloat
    Flux.testmode!(model, true)
    x̂, _, _ = model(x, x_mask; gt_card=gt_card)
    Flux.testmode!(model, false)
    return x̂
end

# Single forward pass returning (x̂, loss, logs) -- used by `reconstruction_check`. Done in one pass
# (not elbo_with_logging + a second forward) since the decoder query is fresh noise on every call,
# so x̂ and the loss would otherwise come from different samples. x̂ are logits (argmax over dim 1
# gives the predicted category of each slot). `reconstruction_check` forwards only β, so by default
# use_elbo follows the encoder: KL term iff it is gaussian (hₑ = (μ_z, Σ_z)).
function reconstruct_and_log(model::DiscreteBagVAE, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}, logpdf; β=1f0, λ=1f0, use_elbo::Union{Bool, Nothing}=nothing, kwargs...) where T <: AbstractFloat
    Flux.testmode!(model, true)
    x̂, hₑ, n̂ = model(x, x_mask; gt_card=true)
    Flux.testmode!(model, false)
    use_elbo = isnothing(use_elbo) ? hₑ isa Tuple : use_elbo
    (use_elbo && !(hₑ isa Tuple)) && throw(ArgumentError("use_elbo=true needs gaussian_mlp=true"))
    C = logpdf(x̂, x) # (N, N, BS)
    matched_indices, _ = hungarian_match(C, x_mask, x_mask)
    ℒ_rec = length(matched_indices) > 0 ? mean(C[matched_indices]) : zero(T)
    ℒ_card, card_mae = cardinality_loss(n̂, x_mask)
    if use_elbo
        μ_z, Σ_z = hₑ
        ℒₖₗ = GenerativeMIL.kl_divergence(μ_z, Σ_z) |> mean
        ℒ = ℒ_rec + β * ℒₖₗ + λ * ℒ_card
        logs = (ℒ = ℒ, ℒ_rec=ℒ_rec, ℒₖₗ=ℒₖₗ, ℒ_card=ℒ_card, card_mae=card_mae)
    else
        ℒ = ℒ_rec + λ * ℒ_card
        logs = (ℒ = ℒ, ℒ_rec=ℒ_rec, ℒ_card=ℒ_card, card_mae=card_mae)
    end
    return x̂, ℒ, logs
end
