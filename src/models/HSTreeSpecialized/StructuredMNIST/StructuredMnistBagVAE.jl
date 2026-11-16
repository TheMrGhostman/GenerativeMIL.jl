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


