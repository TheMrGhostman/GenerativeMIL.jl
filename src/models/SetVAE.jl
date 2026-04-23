"""
Hierarchical encoder used by SetVAE.

Fields:
- `expansion`: initial projection from input space to hidden space
- `layers`: stack of encoder blocks that produce skip connections

Notes:
- The encoder stores intermediate hidden states for decoder skip connections.
"""
struct HierarchicalEncoder{E,L}
    expansion::E
    layers::L
end

Flux.@layer HierarchicalEncoder

"""
Encode a batch of sets with an optional mask.

Arguments:
- `x`: input tensor `(d, n, bs)`
- `x_mask`: optional boolean mask `(1, n, bs)`

Returns:
- `h`: encoded hidden representation.
- `h_encs`: decoder-ordered `Zygote.Buffer` of skip states.

Notes:
- `h_encs` is written in reversed order to match decoder traversal order.
"""
function (m::HierarchicalEncoder)(x::AbstractArray{T}, x_mask::Mask=nothing) where T <: AbstractFloat
    x = isnothing(x_mask) ? m.expansion(x) : multiplicative_masking(m.expansion(x), x_mask)
    h_encs = Zygote.Buffer(Vector{typeof(x)}(undef, length(m.layers)))
    for (i, layer) in enumerate(m.layers)
        x, h_enc = layer(x, x_mask)
        h_encs[length(m.layers) - i + 1] = h_enc
    end
    return x, h_encs
end

AbstractTrees.children(m::HierarchicalEncoder) = (("Expansion", m.expansion), m.layers)
AbstractTrees.printnode(io::IO, m::HierarchicalEncoder) = print(io, "HierarchicalEncoder - ($(length(m.layers)) depth)")

"""
Hierarchical decoder used by SetVAE.

Fields:
- `expansion`: projection from prior samples to hidden space
- `layers`: stack of attentive bottleneck decoder layers
- `reduction`: projection from hidden space to output space

Notes:
- Decoder returns both reconstruction and KL diagnostics per layer.
"""
struct HierarchicalDecoder{E,L,R}
    expansion::E # expansion of prior samples
    layers::L
    reduction::R
end

Flux.@layer HierarchicalDecoder

"""
Decode latent samples with optional mask and uniform KL weighting.

Arguments:
- `z`: prior sample tensor.
- `h_encs`: encoder skip states from `HierarchicalEncoder`.
- `x_mask`: optional boolean mask.

Returns:
- `x̂`: reconstructed batch.
- `klds`: per-layer KL terms.
- `zs`: per-layer latent samples.
- `ℒₖₗ`: total KL loss (sum of `klds`).
"""
function (m::HierarchicalDecoder)(z::AbstractArray{T}, h_encs::Zygote.Buffer, x_mask::Mask=nothing) where T <: AbstractFloat
    x = multiplicative_masking(m.expansion(z), x_mask)
    zs = Vector{typeof(z)}(undef, length(m.layers))
    klds = Vector{T}(undef, length(m.layers))
    kld_loss = zero(T)
    for (i, (layer, h_enc)) in enumerate(zip(m.layers, h_encs))
        x, kld, _, z = layer(x, h_enc, x_mask) 
        Zygote.@ignore begin
             klds[i] = kld
             zs[i] = z
        end
        kld_loss += kld
    end
    x = multiplicative_masking(m.reduction(x), x_mask)
    return x, klds, zs, kld_loss
end

"""
Decode latent samples with optional mask and per-layer KL weighting.

Arguments:
- `z`: prior sample tensor.
- `h_encs`: encoder skip states from `HierarchicalEncoder`.
- `x_mask`: optional boolean mask.
- `β`: vector of per-layer KL weights, same length as `m.layers`.

Returns:
- `x̂`: reconstructed batch.
- `klds`: raw per-layer KL terms.
- `zs`: per-layer latent samples.
- `ℒₖₗ`: weighted KL loss, `sum(β[i] * klds[i])`.
"""
function (m::HierarchicalDecoder)(z::AbstractArray{T}, h_encs::Zygote.Buffer, x_mask::Mask, β::AbstractVector{<:AbstractFloat}) where T <: AbstractFloat
    n_layers = length(m.layers)
    length(β) == n_layers || throw(ArgumentError("Length of β ($(length(β))) must equal number of decoder layers ($n_layers)."))
    β_local = T.(collect(β)) # trick to ensure correct type and allow indexing

    x = multiplicative_masking(m.expansion(z), x_mask)
    zs = Vector{typeof(z)}(undef, n_layers)
    klds = Vector{T}(undef, n_layers)
    kld_loss = zero(T)
    for (i, (layer, h_enc)) in enumerate(zip(m.layers, h_encs))
        x, kld, _, z = layer(x, h_enc, x_mask)
        Zygote.@ignore begin
            klds[i] = kld
            zs[i] = z
        end
        kld_loss += β_local[i] * kld
    end
    x = multiplicative_masking(m.reduction(x), x_mask)
    return x, klds, zs, kld_loss
end


"""
Hierarchical variational autoencoder for set-valued inputs.

Fields:
- `encoder`: hierarchical encoder with skip outputs
- `decoder`: hierarchical decoder with bottleneck layers
- `prior`: latent prior distribution

Notes:
- The model forward pass supports optional masks and scalar/vector KL weights.
"""
struct SetVAE{E<:HierarchicalEncoder, D<:HierarchicalDecoder, P<:AbstractPriorDistribution} <: AbstractGenModel
    encoder::E
    decoder::D
    prior::P
end

AbstractTrees.children(m::HierarchicalDecoder) = (("Expansion", m.expansion), m.layers, ("Reduction", m.reduction))
AbstractTrees.printnode(io::IO, m::HierarchicalDecoder) = print(io, "HierarchicalDecoder - ($(length(m.layers)) depth)")

Flux.@layer SetVAE

"""
Run encoder and sample from the prior for a given batch.

Arguments:
- `svae`: SetVAE model instance.
- `x`: input batch `(d, n, bs)`.
- `x_mask`: optional boolean mask `(1, n, bs)`.

Returns:
- `z` has shape `(prior_dim, n_points, batch_size)`
- `h_encs` are encoder skip states in decoder order
"""
function _forward_encoder_and_prior(svae::SetVAE, x::AbstractArray{T}, x_mask::Mask) where T <: AbstractFloat
    _, h_encs = svae.encoder(x, x_mask) 
    _, sample_size, bs = size(x)
    z = svae.prior(sample_size, bs)
    return z, h_encs
end

"""
Normalize scalar KL weight to a per-layer vector with element type `T`.

Arguments:
- `β`: scalar KL weight.
- `n_layers`: number of decoder layers.
- `T`: target element type.

Returns:
- Vector of length `n_layers` with element type `T`.
"""
function _normalize_β(β::AbstractFloat, n_layers::Int, ::Type{T}) where {T<:AbstractFloat}
    return fill(T(β), n_layers)
end

"""
Validate and normalize per-layer KL weights to element type `T`.

Arguments:
- `β`: KL weights per decoder layer.
- `n_layers`: expected number of decoder layers.
- `T`: target element type.

Returns:
- `Vector{T}` of length `n_layers`.

Throws:
- `ArgumentError` when `length(β) != n_layers`.
"""
function _normalize_β(β::AbstractVector{<:AbstractFloat}, n_layers::Int, ::Type{T}) where {T<:AbstractFloat}
    length(β) == n_layers || throw(ArgumentError("Length of β ($(length(β))) must equal number of decoder layers ($n_layers)."))
    return T.(collect(β))
end

"""
Forward pass of SetVAE with optional mask and scalar/vector KL weighting.

Arguments:
- `x`: input set batch `(d, n, bs)`
- `x_mask`: optional boolean mask `(1, n, bs)`
- `β`: scalar or per-layer KL weights

Returns:
- `x̂`: reconstructed set batch.
- `ℒₖₗ`: total KL loss after applying `β` weights.
- `ℒₖₗₛ`: raw per-layer KL values.
- `zs`: per-layer latent samples.
"""
function (svae::SetVAE)(x::AbstractArray{T}, x_mask::Mask=nothing; β::BetaArg=1f0) where T <: AbstractFloat
    β_vec = _normalize_β(β, length(svae.decoder.layers), T)
    z, h_encs = _forward_encoder_and_prior(svae, x, x_mask)
    x̂, ℒₖₗₛ, zs, ℒₖₗ = svae.decoder(z, h_encs, x_mask, β_vec)
    return x̂, ℒₖₗ, ℒₖₗₛ, zs
end





"""
Compute ELBO and logging values for unmasked batches.

Arguments:
- `model`: SetVAE instance.
- `x`: input batch `(d, n, bs)`.
- `β`: scalar or per-layer KL weights.
- `logpdf`: reconstruction loss function (default `chamfer_distance`).

Returns:
- Total loss `ℒ = ℒ_rec + ℒₖₗ`.
- Named tuple with keys `ℒ`, `ℒ_rec`, `ℒₖₗ`, `ℒₖₗₛ`, `β`.
"""
function elbo_with_logging(model::SetVAE, x::AbstractArray{T,3}; β::BetaArg=1f0, logpdf::Function=chamfer_distance, kwargs... ) where T <: AbstractFloat
    x̂, ℒₖₗ, ℒₖₗₛ, _ = model(x; β=β)
    ℒ_rec = logpdf(x̂, x)
    ℒ = ℒ_rec + ℒₖₗ
    return ℒ, (ℒ = ℒ, ℒ_rec = ℒ_rec, ℒₖₗ = ℒₖₗ, ℒₖₗₛ = ℒₖₗₛ, β = β)
end


"""
Compute ELBO and logging values for masked batches.

Arguments:
- `model`: SetVAE instance.
- `x`: input batch `(d, n, bs)`.
- `x_mask`: boolean mask `(1, n, bs)`.
- `β`: scalar or per-layer KL weights.
- `logpdf`: masked reconstruction loss function (default `masked_chamfer_distance`).

Returns:
- Total loss `ℒ = ℒ_rec + ℒₖₗ`.
- Named tuple with keys `ℒ`, `ℒ_rec`, `ℒₖₗ`, `ℒₖₗₛ`, `β`.
"""
function elbo_with_logging(model::SetVAE, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool, 3}; β::BetaArg=1f0, logpdf::Function=masked_chamfer_distance, kwargs...) where T <: AbstractFloat
    x̂, ℒₖₗ, ℒₖₗₛ, _ = model(x, x_mask; β=β)
    ℒ_rec = logpdf(x̂, x, x_mask, x_mask)
    ℒ = ℒ_rec + ℒₖₗ
    return ℒ, (ℒ = ℒ, ℒ_rec = ℒ_rec, ℒₖₗ = ℒₖₗ, ℒₖₗₛ = ℒₖₗₛ, β = β)
end


"""
One optimization step for unmasked SetVAE batches.

Arguments:
- `model`: SetVAE instance.
- `batch`: input batch `(d, n, bs)`.
- `opt`: optimizer state returned by `Optimisers.setup`.
- `logpdf`: reconstruction loss function.
- `device`: device transfer function (`cpu`, `gpu`, `identity`, ...).
- `β`: scalar or per-layer KL weights.

`opt` is expected to be created with `Optimisers.setup(rule, model)`.

Returns:
- Updated `model`.
- Updated optimizer state `opt`.
- Logging tuple from `elbo_with_logging`.
"""
function optim_step(model::SetVAE, batch::AbstractArray{T,3}, opt::NamedTuple, logpdf::Function, device::Function=cpu; β=1f0, kwargs...) where T <: AbstractFloat
    batch = device(batch)
    (loss, logs), (∇model, ∇data) = Zygote.withgradient(model, batch) do m, x
        elbo_with_logging(m, x; logpdf=logpdf, β=β)
    end
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

"""
One optimization step for masked SetVAE batches.

Arguments:
- `model`: SetVAE instance.
- `batch`: tuple `(X, X_mask)`.
- `opt`: optimizer state returned by `Optimisers.setup`.
- `logpdf`: reconstruction loss function.
- `device`: device transfer function (`cpu`, `gpu`, `identity`, ...).
- `β`: scalar or per-layer KL weights.

Returns:
- Updated `model`.
- Updated optimizer state `opt`.
- Logging tuple from `elbo_with_logging`.
"""
function optim_step(model::SetVAE, batch::Tuple{AbstractArray{T,3}, AbstractArray{Bool,3}}, opt::NamedTuple, logpdf::Function, device::Function=cpu; β=1f0, kwargs...) where T <: AbstractFloat
    X, X_mask = batch
    X, X_mask = device(X), device(X_mask)
    (loss, logs), (∇model, ∇x, ∇x_mask) = Zygote.withgradient(model, X, X_mask) do m, x, x_mask
        elbo_with_logging(m, x, x_mask; logpdf=logpdf, β=β) #TODO check if x_mask will not cause issues with Zygote gradient tracking
    end
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

"""
Validation loop for SetVAE.

Supports both dataloaders yielding `x` and dataloaders yielding `(x, x_mask)`.

Arguments:
- `model`: SetVAE instance.
- `dataloader`: iterable of batches.
- `logpdf`: reconstruction loss function.
- `β`: scalar or per-layer KL weights.
- `device`: device transfer function (`cpu`, `gpu`, `identity`, ...).

Returns:
- `logs`: named tuple with `ℒᵥ`, `ℒᵥ_rec`, `ℒᵥₖₗ`, `ℒᵥₖₗₛ`.
- `early_stopping_loss`: scalar validation loss (`ℒᵥ`).
"""
function valid_step(model::SetVAE, dataloader::DataLoader, logpdf::Function; β=1f0, device::Function=cpu, kwargs...)
    ℒ, ℒ_rec, ℒₖₗ = 0f0, 0f0, 0f0
    ℒₖₗₛ = zeros(Float32, length(model.decoder.layers))
    for batch in dataloader
        loss, logs = if batch isa Tuple && length(batch) == 2
            x, x_mask = batch
            x = device(x)
            x_mask = device(x_mask)
            elbo_with_logging(model, x, x_mask; logpdf=logpdf, β=β)
        else
            x = device(batch)
            elbo_with_logging(model, x; logpdf=logpdf, β=β)
        end

        ℒ += loss
        ℒ_rec += logs.ℒ_rec
        ℒₖₗ += logs.ℒₖₗ
        ℒₖₗₛ .+= Float32.(logs.ℒₖₗₛ)
    end

    n = length(dataloader)
    logs = (; ℒᵥ = ℒ/n, ℒᵥ_rec = ℒ_rec/n, ℒᵥₖₗ = ℒₖₗ/n, ℒᵥₖₗₛ = ℒₖₗₛ ./ n)
    return logs, ℒ/n
end




######################################
###          Constructors          ###
######################################

"""
Build SetVAE from explicit architecture hyperparameters.

Arguments define encoder/decoder dimensions, number of heads, induced set
sizes, latent dimensions, and prior configuration.

Arguments:
- `input_dim`: feature dimension of input points.
- `hidden_dim`: hidden feature width in transformer blocks.
- `heads`: number of attention heads.
- `induced_set_sizes`: induced set sizes for hierarchical blocks.
- `latent_dims`: latent dimensions for bottleneck layers.
- `zed_depth`: depth of latent MLPs in bottleneck layers.
- `zed_hidden_dim`: hidden width of latent MLPs.
- `activation`: activation function used in latent MLPs.
- `n_mixtures`: number of mixture components in prior.
- `prior_dim`: latent prior dimension.
- `output_activation`: final output activation.

Returns:
- Constructed `SetVAE` instance.
"""
function SetVAE(input_dim::Int, hidden_dim::Int, heads::Int, induced_set_sizes::AbstractVector{<:Integer}, 
    latent_dims::AbstractVector{<:Integer}, zed_depth::Int, zed_hidden_dim::Int, activation::Function=Flux.relu, 
    n_mixtures::Int=5, prior_dim::Int=3, output_activation::Function=identity) 
    #prior_type::AbstractPriorDistribution=MixtureOfGaussians)

    (length(induced_set_sizes) !=length(latent_dims)) ? error("induced sets and latent dims have different lengths") : nothing

    # ENCODER
    enc_blocks = Union{InducedSetAttentionBlock, InducedSetAttentionHalfBlock}[]
    for iss in induced_set_sizes[1:end-1]
        isab = InducedSetAttentionBlock(iss, hidden_dim, heads)
        push!(enc_blocks, isab)
    end
    half_block = InducedSetAttentionHalfBlock(induced_set_sizes[end], hidden_dim, heads)
    push!(enc_blocks, half_block)

    encoder = HierarchicalEncoder(
        Flux.Dense(input_dim, hidden_dim),
        enc_blocks
    )

    # Prior # FIXME another option for prior distribution 
    prior = MixtureOfGaussians(prior_dim, n_mixtures, true)

    #DECODER
    dec_blocks = Union{AttentiveHalfBlock, AttentiveBottleneckLayer}[]
    half_block = AttentiveHalfBlock(induced_set_sizes[end], hidden_dim, heads, latent_dims[end], zed_hidden_dim, zed_depth, activation)
    push!(dec_blocks, half_block)

    for (iss, zdim) in zip(reverse(induced_set_sizes)[2:end], reverse(latent_dims)[2:end])
        abl = AttentiveBottleneckLayer(iss, hidden_dim, heads, zdim, zed_hidden_dim, zed_depth, activation)
        push!(dec_blocks, abl)
    end
    decoder = HierarchicalDecoder(
        Flux.Dense(prior_dim, hidden_dim),
        dec_blocks,
        Flux.Dense(hidden_dim, input_dim, x->output_activation(x))
    )
    return SetVAE(encoder, decoder, prior)
end

"""
Build SetVAE from a named-tuple style config (typically loaded from JSON/TOML).

Expected keys include `idim`, `hdim`, `heads`, `is_sizes`, `zdims`,
`vb_depth`, `vb_hdim`, `activation`, and prior/output settings.

Returns:
- Constructed `SetVAE` instance.
"""
function setvae_constructor_from_named_tuple(
    ;idim, hdim, heads, is_sizes, zdims, vb_depth, vb_hdim, activation, 
    n_mixtures=5, prior_dim, output_activation=identity, prior="mog", 
    init_seed=nothing, kwargs...)
    #n_mixtures = (n_mixtures === nothing) ? 5 : n_mixtures
    #output_activation = (output_activation === nothing) ? identity : output_activation
    activation = eval(:($(Symbol(activation))))
    (init_seed !== nothing) ? Random.seed!(init_seed) : nothing
    model = SetVAE(
        idim, hdim, heads, is_sizes, zdims, vb_depth, vb_hdim, 
        activation, n_mixtures, prior_dim, output_activation#, prior_type
        )
    (init_seed !== nothing) ? Random.seed!() : nothing
    return model
end


######################################
### Score functions and evaluation ###
######################################

"""
Reconstruct a set batch using posterior encoder states and prior samples.

Arguments:
- `x`: input tensor `(d, n, bs)`
- `x_mask`: optional boolean mask `(1, n, bs)`

Returns:
- Reconstructed set batch `x̂` with the same shape as `x`.
"""
function reconstruct(svae::SetVAE, x::AbstractArray{T}, x_mask::Mask=nothing; kwargs...) where T <: AbstractFloat
    Flux.testmode!(svae, true)
    x̂, _, _, _ = svae(x, x_mask; kwargs...)
    Flux.testmode!(svae, false)
    return x̂
end

"""
Apply `reconstruct` to an iterable of sets and return CPU outputs.

`data` is expected to contain individual sets (or compatible outputs of
preprocessing helpers). Reconstruction runs with `DataLoader(batchsize=1)`.
When `testmode=true`, the model is switched to Flux test mode.

Arguments:
- `vae`: SetVAE instance.
- `data`: collection of sets.
- `testmode`: whether to switch model into Flux test mode.

Returns:
- Vector of reconstructed sets moved to CPU, one output per input sample.
"""
function transform_and_reconstruct(vae::SetVAE, data::AbstractArray; testmode=true)
    # expect to get output from GroupAD.Models.unpack_mill(tr_data) or list of "sets"
    dataloader = Flux.Data.DataLoader(data, batchsize=1) 
    # we could iterate via data itself (batchsize=1) but we decided to use dataloader instaed
    X̂ = []
    vae = (testmode) ? Flux.testmode!(vae, true) : vae # to testmode
    for batch in dataloader
        x, x_mask = transform_batch(batch, true) 
        x̂ = reconstruct(vae, x, x_mask)
        push!(X̂, x̂ |> Flux.squeezebatch |>cpu)
    end
    return X̂
end