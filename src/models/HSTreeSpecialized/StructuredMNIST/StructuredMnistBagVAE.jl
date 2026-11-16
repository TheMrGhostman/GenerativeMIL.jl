struct StructuredMnistBagVAE{E<:PoolEncoder, DBV<:DiscreteBagVAE, D,  C, O} <: AbstractGenModel
    encoder::E
    disc_bagvae::DBV
    decoder::D
    digit_cardinality::C
    output::O
end

Flux.@layer StructuredMnistBagVAE


# gt_card=true (training): teacher forcing -- the outer decoder decodes exactly the ground-truth digits (GT digit mask);
# gt_card=false: it uses its own predicted digit count (first n̂ slots), as at generation.
# no_grad_card=true: the digit-count loss trains only the digit-count head (no gradient into z / the encoder).
function (m::StructuredMnistBagVAE)(x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}; gt_card::Bool=true, no_grad_card::Bool=false) where T <: AbstractFloat
    μ_z, Σ_z = encode(m, x, x_mask)
    z = isnothing(Σ_z) ? μ_z : μ_z + Σ_z .* MLUtils.randn_like(μ_z)   # AE: deterministic latent
    gt_digit_mask = _digit_mask(x_mask)                               # (1, L, bs)
    n_digits = size(gt_digit_mask, 2)
    n_points = size(x, 2)
    x̂, digit_mask, n̂_digits, n̂_points = decode(m, z, n_digits, n_points; card_mask = gt_card ? gt_digit_mask : nothing, no_grad_card=no_grad_card)
    return x̂, μ_z, Σ_z, digit_mask, n̂_digits, n̂_points
end

_last(l) = l isa Flux.Chain ? l[end] : l
is_variational(m::StructuredMnistBagVAE) = _last(m.disc_bagvae.encoder.postpool) isa SplitLayer


_inner_emb_dim(m::StructuredMnistBagVAE) = size(first(l for l in Flux.modules(m.output) if l isa Flux.Dense).weight, 2)

# (1, n_points, L, bs) point mask -> (1, L, bs) digit mask (a digit exists if it has at least one valid point)
_digit_mask(x_mask::AbstractArray{Bool,4}) = dropdims(any(x_mask; dims=2); dims=2)


function encode(m::StructuredMnistBagVAE, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}) where T <: AbstractFloat
    dₓ, n, l, bs = size(x)
    _, nm, lm, bsm = size(x_mask)
    @assert n == nm && l == lm && bs == bsm "x and x_mask must have the same dimensions, but got x: $(size(x)), x_mask: $(size(x_mask))"

    # inner encoder on every digit
    x_reshaped = reshape(x, dₓ, n, l*bs)              # (dₓ, n, l*bs)
    x_mask_reshaped = reshape(x_mask, 1, n, l*bs)     # (1, n, l*bs)
    h = m.encoder(x_reshaped, x_mask_reshaped)        # (dₕ, m_z, l*bs)
    dₕ, dmz, _ = size(h)
    h = reshape(h, dₕ*dmz, l, bs)                     # (dₕ*m_z, l, bs)
    h_mask = _digit_mask(x_mask)                      # (1, l, bs) digit mask

    # outer (bag) encoder 
    hz = m.disc_bagvae.encoder(h, h_mask)             # SplitLayer -> (μ_z, Σ_z); Dense (AE) -> z
    return hz isa Tuple ? hz : (hz, nothing)
end

function decode(m::StructuredMnistBagVAE, z::AbstractArray{T,3}, n_digits::Int, n_points::Int; card_mask=nothing, no_grad_card::Bool=false) where T <: AbstractFloat
    # outer level: n_digits digit embeddings; digit_mask = GT digit mask (teacher forcing) or predicted from n̂_digits
    x̂, digit_mask, n̂_digits = decode(m.disc_bagvae, z, n_digits; card_mask=card_mask, no_grad_card=no_grad_card)  # (e_hdim*m_z, n_digits, bs)
    bs = size(x̂, 3)
    dₕ = _inner_emb_dim(m)
    dmz = size(x̂, 1) ÷ dₕ
    x̂ = reshape(x̂, dₕ, dmz, n_digits*bs)                       # (dₕ, m_z, n_digits*bs)

    # inner level: points per digit
    n̂_points = reshape(m.digit_cardinality(x̂), 1, 1, n_digits, bs)
    q = MLUtils.randn_like(x̂, (dₕ, n_points, n_digits*bs))
    x̂ = reshape(m.decoder(q, x̂), dₕ, n_points, n_digits, bs)
    x̂ = m.output(x̂)                                            # (dₓ, n, n_digits, bs)
    x̂ = x̂ .* reshape(digit_mask, 1, 1, n_digits, bs)            # padding digits -> exact zeros
    return x̂, digit_mask, n̂_digits, n̂_points
end


# ---- losses ---------------------------------------------------------------------------------------------------------
#
# ℒ = ℒ_rec + α₁ ℒ_card_digits + α₂ ℒ_card_points + β ℒₖₗ   (VAE, elbo_with_logging)
# ℒ = ℒ_rec + α₁ ℒ_card_digits + α₂ ℒ_card_points + γ ℒ_z   (AE,  loss_with_logging)
#
# - ℒ_card_digits: digit count per bag (outer DiscreteBagVAE head), MSE on count / L -- no matching needed.
# - ℒ_rec, ℒ_card_points: per digit, on the Hungarian-matched (predicted slot, ground-truth digit) pairs.
# No existence head: the digit count (teacher-forced in training) decides which predicted slots exist.

"""
`_bag_matching_losses(ŷ, digit_mask, n̂_points, x, x_mask, distance; α₂, card_distance, card_after_hm)`

Hungarian matching of the active predicted digit slots to the ground-truth digits, and the matched-pair losses.

The predicted side is restricted in the MATCHING, not in the distance: `hungarian_match(C, gt_mask, digit_mask)` drops
inactive predicted slots (rows) before solving, so they can never be matched. (Masking ŷ inside the masked Chamfer
would be wrong: it sets the cost of pairs with an empty cluster to 0, which the matching would then prefer.) Under
teacher forcing #active slots == #ground-truth digits, so every digit gets exactly one partner; with `gt_card=false`
surplus slots / digits stay unmatched.

Returns `(ℒ_rec, ℒ_card_points, n_unmatched_gt)`; `n_unmatched_gt` (GT digits without a partner) is for logging only.
"""
function _bag_matching_losses(ŷ, digit_mask, n̂_points, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}, distance::Function; α₂, card_distance::Function, card_after_hm::Bool) where T <: AbstractFloat
    C_rec = distance(ŷ, x, nothing, x_mask)                    # (M, L, BS); active predicted digits have all n points
    C_card = card_distance(n̂_points, x_mask)                   # (M, L, BS); softplus(n̂) vs count / npoints
    C = card_after_hm ? C_rec : C_rec .+ α₂ .* C_card          # point-count term in the matching cost (unless card_after_hm)

    gt_mask = _digit_mask(x_mask)                               # (1, L, BS)
    c_ml, _ = Zygote.@ignore hungarian_match(C, gt_mask, digit_mask)
    n_matched = length(c_ml)
    n_unmatched_gt = Zygote.@ignore sum(Array(gt_mask)) - n_matched

    ℒ_rec = n_matched > 0 ? mean(C_rec[c_ml]) : zero(T)
    ℒ_card_points = n_matched > 0 ? mean(C_card[c_ml]) : zero(T)
    return ℒ_rec, ℒ_card_points, n_unmatched_gt
end

# loss weights of the HSQVAE models don't exist here; a copied config would otherwise be silently ignored (α₁=α₂=1)
function _check_bag_loss_kwargs(kwargs)
    for k in (:λ, :λ_exist, :α)
        haskey(kwargs, k) && throw(ArgumentError("StructuredMnistBagVAE: unknown loss weight $k; use α₁ (digit count), α₂ (point count), β (VAE) / γ (AE)"))
    end
end

"""
`elbo_with_logging(model::StructuredMnistBagVAE, x, x_mask, distance; β, α₁, α₂, ...)` -- VAE loss
`ℒ_rec + α₁ ℒ_card_digits + α₂ ℒ_card_points + β ℒₖₗ`. Only for `variational=true` (the AE uses `loss_with_logging`).

`gt_card=true` (default) teacher-forces the digit count; `card_after_hm=true` keeps the point-count term out of the
matching cost; `no_grad_card=true` trains the digit-count head without gradient into the latent.
"""
function elbo_with_logging(model::StructuredMnistBagVAE, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}, distance::Function=chamfer_pairwise_distance; β::AbstractFloat=1f0, α₁::AbstractFloat=1f0, α₂::AbstractFloat=1f0, card_distance::Function=pairwise_lin_card_distance, card_after_hm::Bool=false, gt_card::Bool=true, no_grad_card::Bool=false, kwargs...) where T <: AbstractFloat
    _check_bag_loss_kwargs(kwargs)
    is_variational(model) || throw(ArgumentError("elbo_with_logging: deterministic (AE) StructuredMnistBagVAE has no KL term; use loss_with_logging"))

    ŷ, μ_z, Σ_z, digit_mask, n̂_digits, n̂_points = model(x, x_mask; gt_card=gt_card, no_grad_card=no_grad_card)
    ℒ_rec, ℒ_card_points, n_unmatched = _bag_matching_losses(ŷ, digit_mask, n̂_points, x, x_mask, distance; α₂=α₂, card_distance=card_distance, card_after_hm=card_after_hm)
    ℒ_card_digits, card_mae_digits = cardinality_loss(n̂_digits, _digit_mask(x_mask))

    ℒₖₗ = mean(kl_divergence(μ_z, Σ_z))
    ℒ = ℒ_rec + α₁ * ℒ_card_digits + α₂ * ℒ_card_points + β * ℒₖₗ
    logs = (ℒ = ℒ, ℒ_rec = ℒ_rec, ℒ_card_digits = ℒ_card_digits, ℒ_card_points = ℒ_card_points, ℒₖₗ = ℒₖₗ, β = β, α₁ = α₁, α₂ = α₂, βℒₖₗ = β * ℒₖₗ, α₁ℒ_card_digits = α₁ * ℒ_card_digits, α₂ℒ_card_points = α₂ * ℒ_card_points, n_unmatched = n_unmatched, card_mae_digits = card_mae_digits,)
    return ℒ, logs
end

"""
`loss_with_logging(model::StructuredMnistBagVAE, x, x_mask, distance; γ, α₁, α₂, ...)` -- AE loss
`ℒ_rec + α₁ ℒ_card_digits + α₂ ℒ_card_points + γ ℒ_z`, `ℒ_z = mean(½ Σ_d z_d²)` (RAE latent penalty, same definition
as in StructuredMnistHSQVAE; `γ = 0` default = pure AE). Only for `variational=false`; `β` from the trainer is ignored.
"""
function loss_with_logging(model::StructuredMnistBagVAE, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}, distance::Function=chamfer_pairwise_distance; γ::AbstractFloat=0f0, α₁::AbstractFloat=1f0, α₂::AbstractFloat=1f0, card_distance::Function=pairwise_lin_card_distance, card_after_hm::Bool=false, gt_card::Bool=true, no_grad_card::Bool=false, kwargs...) where T <: AbstractFloat
    _check_bag_loss_kwargs(kwargs)
    is_variational(model) && throw(ArgumentError("loss_with_logging: variational StructuredMnistBagVAE; use elbo_with_logging"))

    ŷ, z, _, digit_mask, n̂_digits, n̂_points = model(x, x_mask; gt_card=gt_card, no_grad_card=no_grad_card)
    ℒ_rec, ℒ_card_points, n_unmatched = _bag_matching_losses(ŷ, digit_mask, n̂_points, x, x_mask, distance; α₂=α₂, card_distance=card_distance, card_after_hm=card_after_hm)
    ℒ_card_digits, card_mae_digits = cardinality_loss(n̂_digits, _digit_mask(x_mask))

    ℒ_z = mean(sum(z .^ 2; dims=1)) / 2
    ℒ = ℒ_rec + α₁ * ℒ_card_digits + α₂ * ℒ_card_points + γ * ℒ_z

    logs = (ℒ = ℒ, ℒ_rec = ℒ_rec, ℒ_card_digits = ℒ_card_digits, , ℒ_card_points = ℒ_card_points, ℒ_z = ℒ_z, α₁ = α₁, α₂ = α₂, γ = γ, α₁ℒ_card_digits = α₁ * ℒ_card_digits, α₂ℒ_card_points = α₂ * ℒ_card_points, γℒ_z = γ * ℒ_z,
         , n_unmatched = n_unmatched, card_mae_digits = card_mae_digits)
    return ℒ, logs
end

# the loss matching the architecture: ELBO for the VAE, AE loss otherwise
_loss_fn(m::StructuredMnistBagVAE) = is_variational(m) ? elbo_with_logging : loss_with_logging

# ---- training / validation steps (called by train_model!) -----------------------------------------------------------

function optim_step(model::StructuredMnistBagVAE, batch::Tuple{X, M, L}, opt::NamedTuple, logpdf, device::Function=cpu; β=1f0, α₁=1f0, α₂=1f0, γ=0f0, kwargs...) where {X <: AbstractArray{<:AbstractFloat,4}, M <: AbstractArray{Bool,4}, L <: AbstractArray{Int}}
    x, x_mask, _ = batch
    return optim_step(model, (x, x_mask), opt, logpdf, device; β=β, α₁=α₁, α₂=α₂, γ=γ, kwargs...)
end

function optim_step(model::StructuredMnistBagVAE, batch::Tuple{X, M}, opt::NamedTuple, logpdf, device::Function=cpu; β=1f0, α₁=1f0, α₂=1f0, γ=0f0, kwargs...) where {X <: AbstractArray{<:AbstractFloat,4}, M <: AbstractArray{Bool,4}}
    x, x_mask = device.(batch)
    lossf = _loss_fn(model)
    (loss, logs), (∇model,) = Zygote.withgradient(model) do m
        lossf(m, x, x_mask, logpdf; β=β, α₁=α₁, α₂=α₂, γ=γ, kwargs...)
    end
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

function valid_step(model::StructuredMnistBagVAE, dataloader::DataLoader, logpdf; β=1f0, α₁=1f0, α₂=1f0, γ=0f0, device::Function=cpu, kwargs...)
    variational = is_variational(model)
    lossf = _loss_fn(model)
    ℒ, ℒ_rec, ℒ_cd, mae_d, ℒ_cp, ℒ_lat = 0f0, 0f0, 0f0, 0f0, 0f0, 0f0   # ℒ_lat: ℒₖₗ (VAE) or ℒ_z (AE)
    for batch in dataloader
        x, x_mask = length(batch) == 3 ? (batch[1], batch[2]) : batch
        x, x_mask = device(x), device(x_mask)
        loss, logs = lossf(model, x, x_mask, logpdf; β=β, α₁=α₁, α₂=α₂, γ=γ, kwargs...)
        ℒ += loss
        ℒ_rec += logs.ℒ_rec
        ℒ_cd += logs.ℒ_card_digits
        mae_d += logs.card_mae_digits
        ℒ_cp += logs.ℒ_card_points
        ℒ_lat += variational ? logs.ℒₖₗ : logs.ℒ_z
    end
    n = length(dataloader)
    common = (; ℒᵥ = ℒ/n, ℒᵥ_rec = ℒ_rec/n, ℒᵥ_card_digits = ℒ_cd/n, ℒᵥ_card_mae_digits = mae_d/n, ℒᵥ_card_points = ℒ_cp/n)
    logs = variational ? merge(common, (; ℒᵥₖₗ = ℒ_lat/n)) : merge(common, (; ℒᵥ_z = ℒ_lat/n))
    return logs, ℒ/n
end

## Constructor

function StructuredMnistBagVAE(; 
    idim::Int, e_hdim::Int, e_pre_depth::Int, e_pma_heads::Int, e_pma_mz::Int, e_post_depth::Int, e_pma_attention_fn, dbv_emb_dim::Int, dbv_hdim::Int, dbv_depth::Int, dbv_indices::Int, dbv_heads::Int, dbv_pma_attention_fn, dbv_zdim::Int, dbv_n_attn_layers::Int, d_mha_layers::Int, d_heads::Int, o_hdim::Int, o_depth::Int, cp_pre_layers::Int, cp_layers::Int, activation=relu, output_activation=identity, e_pma_card_pooling::CD1=nothing, dbv_pma_card_pooling::CD2=nothing, e_dropout::Real=0.0, dbv_dropout::Real=0.0, cp_dropout::Real=0.0, cp_out_dim::Int=1, variational::Bool=true, kwargs...
) where {CD1 <:Union{Nothing, Symbol, String}, CD2 <:Union{Nothing, Symbol, String}}

    e_pma_card_pooling = isnothing(e_pma_card_pooling) ? nothing : Symbol(e_pma_card_pooling)
    dbv_pma_card_pooling = isnothing(dbv_pma_card_pooling) ? nothing : Symbol(dbv_pma_card_pooling)

    e_pma_attention_fn = (e_pma_attention_fn isa String) ? eval(Symbol(e_pma_attention_fn)) : e_pma_attention_fn
    dbv_pma_attention_fn = (dbv_pma_attention_fn isa String) ? eval(Symbol(dbv_pma_attention_fn)) : dbv_pma_attention_fn
    activation = (activation isa String) ? eval(Symbol(activation)) : activation
    output_activation = (output_activation isa String) ? eval(Symbol(output_activation)) : output_activation

    # first (inner) stage bag encoder.
    e_pma = PMA(e_pma_mz, e_hdim, e_pma_heads; attention_fn=e_pma_attention_fn)
    e_pma = isnothing(e_pma_card_pooling) ? e_pma : CardinalityPooling(e_pma, e_pma_card_pooling)
    e_cfd = isnothing(e_pma_card_pooling) ? 0 : feature_dim(e_pma)

    encoder = PoolEncoder(
        create_mlp(idim, e_hdim, e_pre_depth, e_hdim, activation; dropout=e_dropout),
        e_pma,
        create_mlp(e_hdim + e_cfd, e_hdim, e_post_depth, dbv_emb_dim, activation, dropout=e_dropout)  # per-token output = dbv_emb_dim
    )

    # second (outer) stage encoder, deep slot query encoder
    dbv = DiscreteBagVAE(;
        dₓ=dbv_emb_dim*e_pma_mz, dₕ=dbv_hdim, m_z=dbv_indices, d_z=dbv_zdim, n_heads=dbv_heads, pma_attention_fn=dbv_pma_attention_fn, n_layers=dbv_depth, att_layers=dbv_n_attn_layers, cp_layers=cp_layers, activation=activation, dropout=dbv_dropout, gaussian_mlp=variational, cp_pre_layers=cp_pre_layers, cp_out_dim=cp_out_dim,
        card_pooling=dbv_pma_card_pooling
    )

    decoder = CrossAttentionDecoder(
        [MultiheadAttentionBlock(dbv_emb_dim, d_heads; activation=activation, attention_fn=attention) for _ in 1:d_mha_layers]
    )

    output = create_mlp(dbv_emb_dim, o_hdim, o_depth, idim, output_activation)

    cardinality_head = if e_pma_mz > 1
        PoolEncoder(
            create_mlp(dbv_emb_dim, dbv_hdim, cp_pre_layers, dbv_hdim, activation, dropout=cp_dropout),
            PMA(1, dbv_hdim, d_heads),
            create_mlp(dbv_hdim, dbv_hdim, cp_layers, cp_out_dim, activation, out_identity=true, dropout=cp_dropout)  # input = PMA output (dbv_hdim)
        )
    else
        create_mlp(dbv_emb_dim, dbv_hdim, cp_layers, cp_out_dim, activation, out_identity=true, dropout=cp_dropout)
    end

    return StructuredMnistBagVAE(encoder, dbv, decoder, cardinality_head, output)
end
