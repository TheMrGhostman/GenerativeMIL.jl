struct StructuredMnistHSQVAE{E<:PoolEncoder, DSQ<:DeepSlotQueryVAE, D, C, O} <: AbstractGenModel
    encoder::E
    deep_slot_query::DSQ
    decoder::D
    digit_cardinality::C
    output::O
end

Flux.@layer StructuredMnistHSQVAE

# VAE (deep_slot_query.prior is a SplitLayer -> μ, σ; trained with elbo_with_logging) or deterministic AE
# (prior is a plain Dense -> z, no σ; trained with loss_with_logging). Chosen by the constructor kwarg `variational`.
is_variational(m::StructuredMnistHSQVAE) = !(m.deep_slot_query.prior isa Flux.Dense)

# inner embedding dimension (dsq_emb_dim) = input width of the output MLP (first Dense, robust to dropout wrappers)
_inner_emb_dim(m::StructuredMnistHSQVAE) = size(first(l for l in Flux.modules(m.output) if l isa Flux.Dense).weight, 2)

"""
`encode(m::StructuredMnistHSQVAE, x (dₓ, n, L, BS), x_mask (1, n, L, BS)) -> (μ_z, Σ_z)`

Inner (per-digit) encoder, then the deep-slot-query bag encoder and its latent head. Returns the posterior
parameters `(z_dim, m_z, BS)`; for the AE (`variational=false`) `μ_z` is the deterministic latent z and `Σ_z = nothing`.
"""
function encode(m::StructuredMnistHSQVAE, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}) where T <: AbstractFloat
    dₓ, n, l, bs = size(x)
    _, nm, lm, bsm = size(x_mask)
    @assert n == nm && l == lm && bs == bsm "x and x_mask must have the same dimensions, but got x: $(size(x)), x_mask: $(size(x_mask))"

    # inner encoder on every digit
    x_reshaped = reshape(x, dₓ, n, l*bs)              # (dₓ, n, l*bs)
    x_mask_reshaped = reshape(x_mask, 1, n, l*bs)     # (1, n, l*bs)
    h = m.encoder(x_reshaped, x_mask_reshaped)        # (dₕ, m_z, l*bs)
    dₕ, dmz, _ = size(h)
    h = reshape(h, dₕ*dmz, l, bs)                     # (dₕ*m_z, l, bs)
    h_mask = dropdims(any(x_mask, dims=2), dims=2)    # (1, l, bs) digit mask

    # outer (bag) encoder of the deep slot query model + latent head
    dsq = m.deep_slot_query
    hz = dsq.prior(dsq.encoder(h, h_mask))            # SplitLayer -> (μ_z, Σ_z); Dense (AE) -> z
    return hz isa Tuple ? hz : (hz, nothing)
end

"""
`decode(m::StructuredMnistHSQVAE, z (z_dim, m_z, BS), n::Int) -> (x̂ (dₓ, n, n_slots, BS), logits_exist (1, n_slots, BS), n̂ (·, 1, n_slots, BS))`

Deep-slot-query decoder (digit slots + existence logits), digit-cardinality head, then the inner decoder that turns
every slot into `n` points (from random point queries). Used by the forward pass, and for reconstruction / generation
from a given latent (e.g. sampled from a latent model fitted on the AE codes).
"""
function decode(m::StructuredMnistHSQVAE, z::AbstractArray{T,3}, n::Int) where T <: AbstractFloat
    x̂, logits_exist = decode(m.deep_slot_query, z)   # (dₕ*m_z, n_slots, bs)
    dhmz, n_slots, bs = size(x̂)
    dₕ = _inner_emb_dim(m)
    dmz = dhmz ÷ dₕ
    x̂ = reshape(x̂, dₕ, dmz, n_slots*bs)              # (dₕ, m_z, n_slots*bs)

    # cardinality prediction for inner decoder
    n̂ = m.digit_cardinality(x̂);
    n̂ = reshape(n̂, 1, 1, n_slots, bs);
    # inner decoder
    prior = MLUtils.randn_like(x̂, (dₕ, n, n_slots * bs));
    x̂ = m.decoder(prior, x̂)                          # (dₕ, n, n_slots*bs)
    x̂ = reshape(x̂, dₕ, n, n_slots, bs)
    x̂ = m.output(x̂)
    return x̂, logits_exist, n̂
end

function (m::StructuredMnistHSQVAE)(x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}) where T <: AbstractFloat
    μ_z, Σ_z = encode(m, x, x_mask)
    z = isnothing(Σ_z) ? μ_z : μ_z + Σ_z .* MLUtils.randn_like(μ_z)   # AE: deterministic latent
    x̂, logits_exist, n̂ = decode(m, z, size(x, 2))
    return x̂, logits_exist, μ_z, Σ_z, n̂
end


# Hungarian-matched reconstruction, cardinality and existence losses shared by elbo_with_logging (VAE) and
# loss_with_logging (AE).
function _matching_losses(ŷ, logits_exist, n̂, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}, distance::Function; α, card_distance::Function, card_after_hm::Bool) where T <: AbstractFloat
    M = size(ŷ, 3) # number of predicted slots
    C_rec = distance(ŷ, x, nothing, x_mask) # (M, L, BS)
    C_card = card_distance(n̂, x_mask)
    # There is option to ignore the cardinality loss for the hungarian matching, and only use propagate cardinality loss after the matching. This is useful when the cardinality loss is not a good metric for matching, but only for regularization.
    C = card_after_hm ? C_rec : C_rec .+ α .* C_card

    x_mask = any(x_mask, dims=2);

    matched_indices, exist_target = Zygote.@ignore hungarian_match(C, x_mask)
    n_matched = length(matched_indices)

    ℒ_rec = n_matched > 0 ? mean(C_rec[matched_indices]) : zero(T)
    ℒ_card = n_matched > 0 ? mean(C_card[matched_indices]) : zero(T)
    ℒ_exist = Flux.Losses.logitbinarycrossentropy(logits_exist, exist_target; agg=mean) * T(M) # sum over all slots, not mean
    return ℒ_rec, ℒ_card, ℒ_exist
end

_check_no_λ_exist(kwargs) = haskey(kwargs, :λ_exist) && throw(ArgumentError("StructuredMnistHSQVAE: λ_exist was renamed to λ (existence-loss weight); use λ in train.loss_kwargs"))

"""
`elbo_with_logging(model, x, x_mask, distance; β, λ, α, ...)` -- VAE loss `ℒ_rec + α ℒ_card + λ ℒ_exist + β ℒₖₗ`.
Only for `variational=true` models (the AE is trained with [`loss_with_logging`](@ref)).
"""
function elbo_with_logging(model::StructuredMnistHSQVAE, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}, distance::Function=chamfer_pairwise_distance; β::AbstractFloat=1f0, λ::AbstractFloat=1f0, α::AbstractFloat=1f0, card_distance::Function=pairwise_lin_card_distance, card_after_hm::Bool=false, kwargs...) where T <: AbstractFloat
    _check_no_λ_exist(kwargs)
    is_variational(model) || throw(ArgumentError("elbo_with_logging: deterministic (AE) StructuredMnistHSQVAE has no KL term; use loss_with_logging"))

    ŷ, logits_exist, μ_z, Σ_z, n̂ = model(x, x_mask)
    ℒ_rec, ℒ_card, ℒ_exist = _matching_losses(ŷ, logits_exist, n̂, x, x_mask, distance; α=α, card_distance=card_distance, card_after_hm=card_after_hm)

    ℒₖₗ = mean(kl_divergence(μ_z, Σ_z))
    ℒ = ℒ_rec + α * ℒ_card + λ * ℒ_exist + β * ℒₖₗ
    logs = (ℒ = ℒ, ℒ_rec = ℒ_rec, ℒ_card = ℒ_card, ℒ_exist = ℒ_exist, λℒ_exist = λ * ℒ_exist , ℒₖₗ = ℒₖₗ, βℒₖₗ = β * ℒₖₗ, λ = λ, β = β, α = α)
    return ℒ, logs
end

"""
`loss_with_logging(model, x, x_mask, distance; γ, λ, α, ...)` -- AE loss `ℒ_rec + α ℒ_card + λ ℒ_exist + γ ℒ_z`.

`ℒ_z = mean(½ Σ_d z_d²)` (sum over z_dim, mean over the m_z tokens and the batch, like `kl_divergence`) is a RAE-style
latent-norm penalty (Ghosh et al. 2020); it is exactly the μ² part of the Gaussian KL, so γ has roughly the scale of
β. `γ = 0` (default) is a pure AE. Only for `variational=false` models; `β` passed by the trainer is ignored.
"""
function loss_with_logging(model::StructuredMnistHSQVAE, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}, distance::Function=chamfer_pairwise_distance; γ::AbstractFloat=0f0, λ::AbstractFloat=1f0, α::AbstractFloat=1f0, card_distance::Function=pairwise_lin_card_distance, card_after_hm::Bool=false, kwargs...) where T <: AbstractFloat
    _check_no_λ_exist(kwargs)
    is_variational(model) && throw(ArgumentError("loss_with_logging: variational StructuredMnistHSQVAE; use elbo_with_logging"))

    ŷ, logits_exist, z, _, n̂ = model(x, x_mask)
    ℒ_rec, ℒ_card, ℒ_exist = _matching_losses(ŷ, logits_exist, n̂, x, x_mask, distance; α=α, card_distance=card_distance, card_after_hm=card_after_hm)

    ℒ_z = mean(sum(z .^ 2; dims=1)) / 2
    ℒ = ℒ_rec + α * ℒ_card + λ * ℒ_exist + γ * ℒ_z
    logs = (ℒ = ℒ, ℒ_rec = ℒ_rec, ℒ_card = ℒ_card, ℒ_exist = ℒ_exist, λℒ_exist = λ * ℒ_exist, ℒ_z = ℒ_z, γℒ_z = γ * ℒ_z, λ = λ, γ = γ, α = α)
    return ℒ, logs
end

# the loss matching the architecture: ELBO for the VAE, AE loss otherwise
_loss_fn(m::StructuredMnistHSQVAE) = is_variational(m) ? elbo_with_logging : loss_with_logging

function optim_step(model::StructuredMnistHSQVAE, batch::Tuple{X, M, L}, opt::NamedTuple, logpdf, device::Function=cpu; β=1f0, λ=1f0, α=1f0, γ=0f0, kwargs...) where {X <: AbstractArray{<:AbstractFloat,4}, M <: AbstractArray{Bool,4}, L <: AbstractArray{Int}}
    x, x_mask, _ = batch
    return optim_step(model, (x, x_mask), opt, logpdf, device; β=β, λ=λ, α=α, γ=γ, kwargs...)
end

function optim_step(model::StructuredMnistHSQVAE, batch::Tuple{X, M}, opt::NamedTuple, logpdf, device::Function=cpu; β=1f0, λ=1f0, α=1f0, γ=0f0, kwargs...) where {X <: AbstractArray{<:AbstractFloat,4}, M <: AbstractArray{Bool,4}}
    x, x_mask = device.(batch)
    lossf = _loss_fn(model)
    (loss, logs), (∇model,) = Zygote.withgradient(model) do m
        lossf(m, x, x_mask, logpdf; β=β, λ=λ, α=α, γ=γ, kwargs...)
    end
    #return loss, logs, ∇model
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

function valid_step(model::StructuredMnistHSQVAE, dataloader::DataLoader, logpdf; β=1f0, λ=1f0, α=1f0, γ=0f0, device::Function=cpu, kwargs...)
    variational = is_variational(model)
    lossf = _loss_fn(model)
    ℒ, ℒ_rec, ℒ_lat, ℒ_exist, ℒ_card = 0f0, 0f0, 0f0, 0f0, 0f0   # ℒ_lat: ℒₖₗ (VAE) or ℒ_z (AE)
    for batch in dataloader
        x, x_mask = length(batch) == 3 ? (batch[1], batch[2]) : batch # TODO: make it more robust to different batch formats
        x, x_mask = device(x), device(x_mask)
        loss, logs = lossf(model, x, x_mask, logpdf; β=β, λ=λ, α=α, γ=γ, kwargs...)

        ℒ += loss
        ℒ_rec += logs.ℒ_rec
        ℒ_lat += variational ? logs.ℒₖₗ : logs.ℒ_z
        ℒ_exist += logs.ℒ_exist
        ℒ_card += logs.ℒ_card
    end

    n = length(dataloader)
    logs = variational ?
        (; ℒᵥ = ℒ/n, ℒᵥ_rec = ℒ_rec/n, ℒᵥ_exist = ℒ_exist/n, ℒᵥₖₗ = ℒ_lat/n, ℒᵥ_card = ℒ_card/n) :
        (; ℒᵥ = ℒ/n, ℒᵥ_rec = ℒ_rec/n, ℒᵥ_exist = ℒ_exist/n, ℒᵥ_z = ℒ_lat/n, ℒᵥ_card = ℒ_card/n)
    return logs, ℒ/n
end



function StructuredMnistHSQVAE(; 
    idim::Int, e_hdim::Int, e_pre_depth::Int, e_pma_heads::Int, e_pma_mz::Int, e_post_depth::Int, e_pma_attention_fn, dsq_emb_dim::Int, dsq_hdim::Int, dsq_pre_depth::Int, dsq_indices::Int, dsq_heads::Int, dsq_post_depth::Int, dsq_pma_attention_fn, dsq_zdim::Int, dsq_n_slots::Int, dsq_n_attn_layers::Int, d_mha_layers::Int, d_heads::Int, o_hdim::Int, o_depth::Int, cp_pre_layers::Int, cp_layers::Int, activation=relu, output_activation=identity, e_pma_card_pooling::CD1=nothing, dsq_pma_card_pooling::CD2=nothing, e_dropout::Real=0.0, dsq_dropout::Real=0.0, cp_dropout::Real=0.0, cp_out_dim::Int=1, variational::Bool=true, kwargs...
) where {CD1 <:Union{Nothing, Symbol, String}, CD2 <:Union{Nothing, Symbol, String}}

    e_pma_card_pooling = isnothing(e_pma_card_pooling) ? nothing : Symbol(e_pma_card_pooling)
    dsq_pma_card_pooling = isnothing(dsq_pma_card_pooling) ? nothing : Symbol(dsq_pma_card_pooling)

    e_pma_attention_fn = (e_pma_attention_fn isa String) ? eval(Symbol(e_pma_attention_fn)) : e_pma_attention_fn
    dsq_pma_attention_fn = (dsq_pma_attention_fn isa String) ? eval(Symbol(dsq_pma_attention_fn)) : dsq_pma_attention_fn
    activation = (activation isa String) ? eval(Symbol(activation)) : activation
    output_activation = (output_activation isa String) ? eval(Symbol(output_activation)) : output_activation

    # first (outer) stage bag encoder.
    e_pma = PMA(e_pma_mz, e_hdim, e_pma_heads; attention_fn=e_pma_attention_fn)
    e_pma = isnothing(e_pma_card_pooling) ? e_pma : CardinalityPooling(e_pma, e_pma_card_pooling)
    e_cfd = isnothing(e_pma_card_pooling) ? 0 : feature_dim(e_pma)

    encoder = PoolEncoder(
        create_mlp(idim, e_hdim, e_pre_depth, e_hdim, activation; dropout=e_dropout),
        e_pma,
        create_mlp(e_hdim + e_cfd, e_hdim, e_post_depth, e_hdim, activation, dropout=e_dropout)
    )

    # second (inner) stage encoder, deep slot query encoder
    dsq_pma = PMA(dsq_indices, dsq_hdim, dsq_heads; attention_fn=dsq_pma_attention_fn)
    dsq_pma = isnothing(dsq_pma_card_pooling) ? dsq_pma : CardinalityPooling(dsq_pma, dsq_pma_card_pooling)
    dsq_cfd = isnothing(dsq_pma_card_pooling) ? 0 : feature_dim(dsq_pma)

    dsq_encoder = PoolEncoder(
        create_mlp(dsq_emb_dim*e_pma_mz, dsq_hdim, dsq_pre_depth, dsq_hdim, activation, dropout=dsq_dropout),
        dsq_pma,
        create_mlp(dsq_hdim + dsq_cfd, dsq_hdim, dsq_post_depth, dsq_hdim, activation, dropout=dsq_dropout)
    )

    # variational=true: Gaussian posterior (μ, softplus σ) -> VAE; false: deterministic latent z -> AE (no σ parameters)
    dsq_prior = variational ? SplitLayer(dsq_hdim, (dsq_zdim, dsq_zdim), (identity, Flux.softplus)) : Flux.Dense(dsq_hdim, dsq_zdim)
    dsq_z_to_hidden = Flux.Dense(dsq_zdim, dsq_hdim)

    # second (inner) stage decoder, deep slot query decoder
    dsq_self_attns  = [MultiheadAttentionBlock(dsq_hdim, dsq_heads; attention_fn=attention) for _ in 1:dsq_n_attn_layers]
    dsq_cross_attns = [MultiheadAttentionBlock(dsq_hdim, dsq_heads; attention_fn=attention) for _ in 1:dsq_n_attn_layers]
    dsq_decoder = TransformerDecoder(dsq_self_attns, dsq_cross_attns)

    dsq_output_head = Flux.Dense(dsq_hdim, dsq_emb_dim*e_pma_mz)
    dsq_exist_head  = Flux.Dense(dsq_hdim, 1)

    dsq_queries = randn(Float32, dsq_hdim, dsq_n_slots)

    dsq = DeepSlotQueryVAE(dsq_encoder, dsq_prior, dsq_z_to_hidden, dsq_decoder, dsq_output_head, dsq_exist_head, dsq_queries)

    decoder = CrossAttentionDecoder(
        [MultiheadAttentionBlock(dsq_emb_dim, d_heads; activation=activation, attention_fn=attention) for _ in 1:d_mha_layers]
    )

    output = create_mlp(dsq_emb_dim, o_hdim, o_depth, idim, output_activation)

    cardinality_head = if e_pma_mz > 1
        PoolEncoder(
            create_mlp(dsq_emb_dim, dsq_hdim, cp_pre_layers, dsq_hdim, activation, dropout=cp_dropout),
            PMA(1, dsq_hdim, d_heads),
            create_mlp(dsq_emb_dim, dsq_hdim, cp_layers, cp_out_dim, activation, out_identity=true, dropout=cp_dropout)
        )
    else
        create_mlp(dsq_emb_dim, dsq_hdim, cp_layers, cp_out_dim, activation, out_identity=true, dropout=cp_dropout)
    end

    return StructuredMnistHSQVAE(encoder, dsq, decoder, cardinality_head, output)
end
