struct StructuredMnistHSQVAE{E<:PoolEncoder, DSQ<:DeepSlotQueryVAE, D, C, O} <: AbstractGenModel
    encoder::E
    deep_slot_query::DSQ
    decoder::D
    digit_cardinality::C
    output::O
end

Flux.@layer StructuredMnistHSQVAE

# this works only for outter encoder with one vecotr as latent. not tensor!!!!!!
function (m::StructuredMnistHSQVAE)(x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}) where T <: AbstractFloat
    dₓ, n, l, bs = size(x)
    _, nm, lm, bs = size(x_mask)
    @assert n == nm && l == lm && bs == bs "x and x_mask must have the same dimensions, but got x: $(size(x)), x_mask: $(size(x_mask))"
    
    # reshape inputs
    x_reshaped = reshape(x, dₓ, n, l*bs) # (dₓ, n, l*bs)
    x_mask_reshaped = reshape(x_mask, 1, n, l*bs) # (1, n, l*bs)

    #  inner encoder
    h = m.encoder(x_reshaped, x_mask_reshaped) # (hidden, m_z, l*bs)
    dₕ, dmz, _ = size(h) # (dₕ, mz, l*bs)
    h = reshape(h, dₕ*dmz, l, bs) # (dₕ*m_z, l, bs)
    h_mask = isnothing(x_mask) ? nothing : dropdims(any(x_mask, dims=2), dims=2) # (1, l, bs)

    # outter encoder-decoder, deep slot query vae
    x̂, logits_exist, μ_z, Σ_z = m.deep_slot_query(h, h_mask)        # (dₕ*dmz, n_slots, bs)
    _, n_slots, bs = size(x̂)

    # reshape x̂ for inner decoder
    x̂ = reshape(x̂, dₕ, dmz, n_slots*bs) # (dₕ, dmz, n_slot*bs)

    # Inner decoder
    # cardinality prediction for inner decoder
    n̂ = m.digit_cardinality(x̂);
    n̂ = reshape(n̂, 1, 1, n_slots, bs);  # TODO think about 3D vs 4D output 
    # inner decoder
    prior = MLUtils.randn_like(x, (dₕ, n, n_slots * bs));
    x̂ = m.decoder(prior, x̂) # (dₕ, n, n_slots * bs)
    x̂ = reshape(x̂, dₕ, n, n_slots, bs)
    x̂ = m.output(x̂)
    return x̂, logits_exist, μ_z, Σ_z, n̂
end


function elbo_with_logging(model::StructuredMnistHSQVAE, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}, distance::Function=chamfer_pairwise_distance; β::AbstractFloat=1f0, λ_exist::AbstractFloat=1f0, α::AbstractFloat=1f0, card_distance::Function=pairwise_lin_card_distance, card_after_hm::Bool=false, kwargs...) where T <: AbstractFloat

    ŷ, logits_exist, μ_z, Σ_z, n̂ = model(x, x_mask)

    x_mask = any(x_mask, dims=2); 

    # hungarian matching loss for reconstruction and existence
    M = size(ŷ, 3) # number of predicted slots
    C_rec = distance(ŷ, x) # (M, L, BS)
    C_card = card_distance(n̂, x_mask) 
    # There is option to ignore the cardinality loss for the hungarian matching, and only use propagate cardinality loss after the matching. This is useful when the cardinality loss is not a good metric for matching, but only for regularization.
    C = card_after_hm ? C_rec : C_rec .+ α .* C_card 

    matched_indices, exist_target = Zygote.@ignore hungarian_match(C, x_mask)
    n_matched = length(matched_indices)

    ℒ_rec = n_matched > 0 ? mean(C_rec[matched_indices]) : zero(T)
    ℒ_card = n_matched > 0 ? mean(C_card[matched_indices]) : zero(T)
    ℒ_exist = Flux.Losses.logitbinarycrossentropy(logits_exist, exist_target; agg=mean) * T(M) # sum over all slots, not mean 
    
    ℒₖₗ = mean(kl_divergence(μ_z, Σ_z))
    ℒ = ℒ_rec + α * ℒ_card + λ_exist * ℒ_exist + β * ℒₖₗ 
    logs = (ℒ = ℒ, ℒ_rec = ℒ_rec, ℒ_card = ℒ_card, ℒ_exist = ℒ_exist, ℒₖₗ = ℒₖₗ, λ_exist = λ_exist, β = β, α = α)
    return ℒ, logs
end

function optim_step(model::StructuredMnistHSQVAE, batch::Tuple{X, M, L}, opt::NamedTuple, logpdf, device::Function=cpu; β=1f0, λ_exist=1f0, kwargs...) where {X <: AbstractArray{<:AbstractFloat,4}, M <: AbstractArray{Bool,4}, L <: AbstractVector{Int}}
    x, x_mask, _ = batch
    return optim_step(model, (x, x_mask), opt, logpdf, device; β=β, λ_exist=λ_exist, kwargs...)
end

function optim_step(model::StructuredMnistHSQVAE, batch::Tuple{X, M}, opt::NamedTuple, logpdf, device::Function=cpu; β=1f0, λ_exist=1f0, kwargs...) where {X <: AbstractArray{<:AbstractFloat,4}, M <: AbstractArray{Bool,4}}
    x, x_mask = device.(batch)
    (loss, logs), (∇model,) = Zygote.withgradient(model) do m
        elbo_with_logging(m, x, x_mask, logpdf; β=β, λ_exist=λ_exist)
    end
    #return loss, logs, ∇model
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

function valid_step(model::StructuredMnistHSQVAE, dataloader::DataLoader, logpdf; β=1f0, λ_exist=1f0, device::Function=cpu, kwargs...)
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
function StructuredMnistHSQVAE(; 
    idim::Int, e_hdim::Int, e_pre_depth::Int, e_pma_heads::Int, e_pma_mz::Int, e_post_depth::Int, e_pma_attention_fn, dsq_emb_dim::Int, dsq_hdim::Int, dsq_pre_depth::Int, dsq_indices::Int, dsq_heads::Int, dsq_post_depth::Int, dsq_pma_attention_fn, dsq_zdim::Int, dsq_n_slots::Int, dsq_n_attn_layers::Int, d_mha_layers::Int, d_heads::Int, o_hdim::Int, o_depth::Int, cp_pre_layers::Int, cp_layers::Int, activation=relu, output_activation=identity, e_pma_card_pooling::CD1=nothing, dsq_pma_card_pooling::CD2=nothing, e_dropout::Real=0.0, dsq_dropout::Real=0.0, cp_dropout::Real=0.0, kwargs...
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

    dsq_prior = SplitLayer(dsq_hdim, (dsq_zdim, dsq_zdim), (identity, Flux.softplus))
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
            create_mlp(dsq_emb_dim, dsq_hdim, cp_layers, 1, activation, out_identity=true, dropout=cp_dropout)
        )
    else
        create_mlp(dsq_emb_dim, dsq_hdim, cp_layers, 1, activation, out_identity=true, dropout=cp_dropout)
    end

    return StructuredMnistHSQVAE(encoder, dsq, decoder, cardinality_head, output)
end
