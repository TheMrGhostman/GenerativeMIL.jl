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
`(m::HierarchicalEncoder)(x::AbstractArray{T}, x_mask::Mask=nothing) where T <: AbstractFloat`

Encode a batch of sets with optional masking.

Arguments (positional):
- `x`: input tensor `(d, n, bs)`.
- `x_mask`: optional boolean mask `(1, n, bs)` (default `nothing`).

Returns:
- `h`: encoded hidden representation.
- `h_encs`: `Zygote.Buffer` of intermediate skip states in reversed (decoder) order.

Notes:
- States in `h_encs` are ordered from deepest to shallowest for decoder skip connections.
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
`(m::HierarchicalDecoder)(z::AbstractArray{T}, h_encs::Zygote.Buffer, x_mask::Mask=nothing) where T <: AbstractFloat`

Decode latent samples with optional masking and uniform KL weighting.

Arguments (positional):
- `z`: prior sample tensor `(prior_dim, n_points, batch_size)`.
- `h_encs`: encoder skip states from `HierarchicalEncoder` (in decoder order).
- `x_mask`: optional boolean mask `(1, n, bs)` (default `nothing`).

Returns:
- `x̂`: reconstructed batch.
- `klds`: vector of raw per-layer KL divergence values.
- `zs`: vector of per-layer latent samples from variational layers.
- `kld_loss`: total KL loss as scalar (sum of all `klds`).
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
`(m::HierarchicalDecoder)(z::AbstractArray{T}, h_encs::Zygote.Buffer, x_mask::Mask, β::AbstractVector{<:AbstractFloat}) where T <: AbstractFloat`

Decode latent samples with optional masking and per-layer KL weighting.

Arguments (positional):
- `z`: prior sample tensor `(prior_dim, n_points, batch_size)`.
- `h_encs`: encoder skip states from `HierarchicalEncoder` (in decoder order).
- `x_mask`: optional boolean mask `(1, n, bs)`.
- `β`: vector of per-layer KL weights (length must equal number of decoder layers).

Returns:
- `x̂`: reconstructed batch.
- `klds`: vector of raw per-layer KL divergence values.
- `zs`: vector of per-layer latent samples from variational layers.
- `kld_loss`: weighted total KL loss as scalar: `sum(β[i] * klds[i])`.

Throws:
- `ArgumentError` if `length(β)` does not match number of decoder layers.
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
`_forward_encoder_and_prior(svae::SetVAE, x::AbstractArray{T}, x_mask::Mask) where T <: AbstractFloat`

Run encoder and sample from prior for a given batch.

Arguments (positional):
- `svae`: SetVAE model instance.
- `x`: input batch `(d, n, bs)`.
- `x_mask`: optional boolean mask `(1, n, bs)`.

Returns:
- `z`: prior sample tensor with shape `(prior_dim, n_points, batch_size)`.
- `h_encs`: encoder skip states in decoder order (from `HierarchicalEncoder`).
"""
function _forward_encoder_and_prior(svae::SetVAE, x::AbstractArray{T}, x_mask::Mask) where T <: AbstractFloat
    _, h_encs = svae.encoder(x, x_mask) 
    _, sample_size, bs = size(x)
    z = svae.prior(sample_size, bs)
    return z, h_encs
end

"""
`_normalize_β(β::AbstractFloat, n_layers::Int, ::Type{T}) where T<:AbstractFloat`

Normalize scalar KL weight to a per-layer vector with element type `T`.

Arguments (positional):
- `β`: scalar KL weight.
- `n_layers`: number of decoder layers.
- `T`: target element type (e.g., `Float32`, `Float64`).

Returns:
- Vector of length `n_layers` filled with `T(β)`, one weight per layer.
"""
function _normalize_β(β::AbstractFloat, n_layers::Int, ::Type{T}) where T<:AbstractFloat
    return fill(T(β), n_layers)
end

"""
`_normalize_β(β::AbstractVector{<:AbstractFloat}, n_layers::Int, ::Type{T}) where T<:AbstractFloat`

Validate and normalize per-layer KL weights to element type `T`.

Arguments (positional):
- `β`: vector of KL weights per decoder layer.
- `n_layers`: expected number of decoder layers.
- `T`: target element type (e.g., `Float32`, `Float64`).

Returns:
- `Vector{T}` of length `n_layers` with type-converted weights.

Throws:
- `ArgumentError` if `length(β) != n_layers`.
"""
function _normalize_β(β::AbstractVector{<:AbstractFloat}, n_layers::Int, ::Type{T}) where T<:AbstractFloat
    length(β) == n_layers || throw(ArgumentError("Length of β ($(length(β))) must equal number of decoder layers ($n_layers)."))
    return T.(collect(β))
end

"""
`(svae::SetVAE)(x::AbstractArray{T}, x_mask::Mask=nothing; β::BetaArg=1f0) where T <: AbstractFloat`

Forward pass of SetVAE with optional masking and scalar/vector KL weighting.

Arguments (positional):
- `x`: input set batch `(d, n, bs)`.
- `x_mask`: optional boolean mask `(1, n, bs)` (default `nothing`).

Keyword arguments:
- `β`: scalar or per-layer KL weights (default `1f0`). Scalar broadcasts to all layers; vector must match number of layers.

Returns:
- `x̂`: reconstructed set batch with same shape as input.
- `ℒₖₗ`: total weighted KL loss after applying `β` scaling.
- `ℒₖₗₛ`: vector of raw per-layer KL values.
- `zs`: vector of per-layer latent samples from variational layers.
"""
function (svae::SetVAE)(x::AbstractArray{T}, x_mask::Mask=nothing; β::BetaArg=1f0) where T <: AbstractFloat
    β_vec = _normalize_β(β, length(svae.decoder.layers), T)
    z, h_encs = _forward_encoder_and_prior(svae, x, x_mask)
    x̂, ℒₖₗₛ, zs, ℒₖₗ = svae.decoder(z, h_encs, x_mask, β_vec)
    return x̂, ℒₖₗ, ℒₖₗₛ, zs
end




"""
`elbo_with_logging(model::SetVAE, x::AbstractArray{T,3}, logpdf::Function=chamfer_distance; β::BetaArg=1f0, kwargs...) where T <: AbstractFloat`

Compute ELBO and logging values for unmasked batches with generic loss function.

Arguments (positional):
- `model`: SetVAE instance.
- `x`: input batch `(d, n, bs)`.
- `logpdf`: reconstruction loss function (default `chamfer_distance`).

Keyword arguments:
- `β`: scalar or per-layer KL weights.
- `kwargs...`: additional keyword arguments.

Returns:
- Total loss `ℒ = ℒ_rec + ℒₖₗ`.
- Named tuple with keys `ℒ`, `ℒ_rec`, `ℒₖₗ`, `ℒₖₗₛ`, `β`.
"""
function elbo_with_logging(model::SetVAE, x::AbstractArray{T,3}, logpdf::Function=chamfer_distance; β::BetaArg=1f0, kwargs... ) where T <: AbstractFloat
    x̂, ℒₖₗ, ℒₖₗₛ, _ = model(x; β=β)
    ℒ_rec = logpdf(x̂, x)#TODO add weights to logpdf so we have sum over points instead of mean
    ℒ = ℒ_rec + ℒₖₗ
    return ℒ, (ℒ = ℒ, ℒ_rec = ℒ_rec, ℒₖₗ = ℒₖₗ, ℒₖₗₛ = ℒₖₗₛ, β = β)
end

"""
`elbo_with_logging(model::SetVAE, x::AbstractArray{T,3}, logpdf::MMD_EMA_Loss; β::BetaArg=1f0, kwargs...) where T <: AbstractFloat`

Compute ELBO and logging values for unmasked batches with `MMD_EMA_Loss`.
Updates EMA sigma estimate in Zygote.@ignore block to preserve gradient flow.

Arguments (positional):
- `model`: SetVAE instance.
- `x`: input batch `(d, n, bs)`.
- `logpdf`: MMD_EMA_Loss instance with encapsulated sigma EMA state.

Keyword arguments:
- `β`: scalar or per-layer KL weights.
- `kwargs...`: additional keyword arguments.

Returns:
- Total loss `ℒ = ℒ_rec + ℒₖₗ`.
- Named tuple with keys `ℒ`, `ℒ_rec`, `ℒₖₗ`, `ℒₖₗₛ`, `β`, `σᵣ` (current EMA sigma).
"""
function elbo_with_logging(model::SetVAE, x::AbstractArray{T,3}, logpdf::MMD_EMA_Loss; β::BetaArg=1f0, update_sigma::Bool=true, kwargs... ) where T <: AbstractFloat 
    x̂, ℒₖₗ, ℒₖₗₛ, _ = model(x; β=β)
    Zygote.@ignore begin
        σₙ = compute_rbf_sigma_estimate(x̂, x)
        (update_sigma) && update_ema_sigma!(logpdf, σₙ)
    end
    ℒ_rec = logpdf(x̂, x)
    ℒ = ℒ_rec + ℒₖₗ
    return ℒ, (ℒ = ℒ, ℒ_rec = ℒ_rec, ℒₖₗ = ℒₖₗ, ℒₖₗₛ = ℒₖₗₛ, β = β, σᵣ = logpdf.σᵣ)
end

"""
`elbo_with_logging(model::SetVAE, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}, logpdf::Function=masked_chamfer_distance; β::BetaArg=1f0, kwargs...) where T <: AbstractFloat`

Compute ELBO and logging values for masked batches with generic loss function.

Arguments (positional):
- `model`: SetVAE instance.
- `x`: input batch `(d, n, bs)`.
- `x_mask`: boolean mask `(1, n, bs)` indicating which points are valid.
- `logpdf`: masked reconstruction loss function (default `masked_chamfer_distance`).

Keyword arguments:
- `β`: scalar or per-layer KL weights.
- `kwargs...`: additional keyword arguments.

Returns:
- Total loss `ℒ = ℒ_rec + ℒₖₗ`.
- Named tuple with keys `ℒ`, `ℒ_rec`, `ℒₖₗ`, `ℒₖₗₛ`, `β`.
"""
function elbo_with_logging(model::SetVAE, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool, 3}, logpdf::Function=masked_chamfer_distance; β::BetaArg=1f0, kwargs...) where T <: AbstractFloat
    x̂, ℒₖₗ, ℒₖₗₛ, _ = model(x, x_mask; β=β)
    ℒ_rec = logpdf(x̂, x, x_mask, x_mask)
    ℒ = ℒ_rec + ℒₖₗ
    return ℒ, (ℒ = ℒ, ℒ_rec = ℒ_rec, ℒₖₗ = ℒₖₗ, ℒₖₗₛ = ℒₖₗₛ, β = β)
end


"""
`optim_step(model::SetVAE, batch::AbstractArray{T,3}, opt::NamedTuple, logpdf, device::Function=cpu; β=1f0, kwargs...) where T <: AbstractFloat`

One optimization step for unmasked SetVAE batches with generic loss function.

Arguments (positional):
- `model`: SetVAE instance.
- `batch`: input batch `(d, n, bs)`.
- `opt`: optimizer state returned by `Optimisers.setup(rule, model)`.
- `logpdf`: reconstruction loss function (e.g., `chamfer_distance` or `MMD_EMA_Loss`).
- `device`: device transfer function (default `cpu`; e.g., `gpu`, `identity`, ...).

Keyword arguments:
- `β`: scalar or per-layer KL weights.
- `kwargs...`: additional keyword arguments.

Returns:
- Updated `model`.
- Updated optimizer state `opt`.
- Logging tuple from `elbo_with_logging`.
"""
function optim_step(model::SetVAE, batch::AbstractArray{T,3}, opt::NamedTuple, logpdf, device::Function=cpu; β=1f0, kwargs...) where T <: AbstractFloat
    batch = device(batch)
    (loss, logs), (∇model, ∇data) = Zygote.withgradient(model, batch) do m, x
        elbo_with_logging(m, x, logpdf; β=β)
    end
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end



"""
`optim_step(model::SetVAE, batch::Tuple{AbstractArray{T,3}, AbstractArray{Bool,3}}, opt::NamedTuple, logpdf::Function, device::Function=cpu; β=1f0, kwargs...) where T <: AbstractFloat`

One optimization step for masked SetVAE batches with generic loss function.

Arguments (positional):
- `model`: SetVAE instance.
- `batch`: tuple `(X, X_mask)` where X is `(d, n, bs)` and X_mask is `(1, n, bs)` boolean.
- `opt`: optimizer state returned by `Optimisers.setup(rule, model)`.
- `logpdf`: masked reconstruction loss function (e.g., `masked_chamfer_distance`).
- `device`: device transfer function (default `cpu`; e.g., `gpu`, `identity`, ...).

Keyword arguments:
- `β`: scalar or per-layer KL weights.
- `kwargs...`: additional keyword arguments.

Returns:
- Updated `model`.
- Updated optimizer state `opt`.
- Logging tuple from `elbo_with_logging`.
"""
function optim_step(model::SetVAE, batch::Tuple{AbstractArray{T,3}, AbstractArray{Bool,3}}, opt::NamedTuple, logpdf::Function, device::Function=cpu; β=1f0, kwargs...) where T <: AbstractFloat
    X, X_mask = batch
    X, X_mask = device(X), device(X_mask)
    (loss, logs), (∇model, ∇x, ∇x_mask) = Zygote.withgradient(model, X, X_mask) do m, x, x_mask
        elbo_with_logging(m, x, x_mask, logpdf; β=β) #TODO check if x_mask will not cause issues with Zygote gradient tracking
    end
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

"""
`valid_step(model::SetVAE, dataloader::DataLoader, logpdf::Function; β=1f0, device::Function=cpu, kwargs...) where T <: AbstractFloat`

Validation loop for SetVAE.
Supports both dataloaders yielding `x` and dataloaders yielding `(x, x_mask)` tuples.

Arguments (positional):
- `model`: SetVAE instance.
- `dataloader`: iterable of batches (unmasked or masked tuples).
- `logpdf`: reconstruction loss function (generic or masked variant).

Keyword arguments:
- `β`: scalar or per-layer KL weights.
- `device`: device transfer function (default `cpu`; e.g., `gpu`, `identity`, ...).
- `kwargs...`: additional keyword arguments.

Returns:
- `logs`: named tuple with `ℒᵥ`, `ℒᵥ_rec`, `ℒᵥₖₗ`, `ℒᵥₖₗₛ` (averaged over dataloader).
- `early_stopping_loss`: scalar validation loss `ℒᵥ` (averaged).
"""
function valid_step(model::SetVAE, dataloader::DataLoader, logpdf; β=1f0, device::Function=cpu, kwargs...)
    ℒ, ℒ_rec, ℒₖₗ = 0f0, 0f0, 0f0
    ℒₖₗₛ = zeros(Float32, length(model.decoder.layers))
    for batch in dataloader
        loss, logs = if batch isa Tuple && length(batch) == 2
            x, x_mask = batch
            x = device(x)
            x_mask = device(x_mask)
            elbo_with_logging(model, x, x_mask, logpdf; β=β, update_sigma=false) # don't update sigma during validation
        else
            x = device(batch)
            elbo_with_logging(model, x, logpdf; β=β, update_sigma=false)
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
`SetVAE(input_dim::Int, hidden_dim::Int, heads::Int, induced_set_sizes::AbstractVector{<:Integer}, latent_dims::AbstractVector{<:Integer}, zed_depth::Int, zed_hidden_dim::Int, activation::Function=Flux.relu, expansion_depth::Int=1, expansion_hidden_dim::Int=0, n_mixtures::Int=5, prior_dim::Int=3, output_activation::Function=identity)`

Build SetVAE from explicit architecture hyperparameters.

Arguments (positional):
- `input_dim`: feature dimension of input points.
- `hidden_dim`: hidden feature width in transformer blocks.
- `heads`: number of attention heads.
- `induced_set_sizes`: vector of induced set sizes for hierarchical encoder blocks.
- `latent_dims`: vector of latent dimensions for bottleneck layers (must match length of `induced_set_sizes`).
- `zed_depth`: depth of latent MLPs in bottleneck layers.
- `zed_hidden_dim`: hidden width of latent MLPs.

Keyword arguments:
- `activation`: activation function for latent MLPs (default `Flux.relu`).
- `expansion_depth`: depth of input/output expansion MLPs (default `1`).
- `expansion_hidden_dim`: hidden width of expansion MLPs (default `0`).
- `n_mixtures`: number of mixture components in prior (default `5`).
- `prior_dim`: latent prior dimension (default `3`).
- `output_activation`: final output activation (default `identity`).

Returns:
- Constructed `SetVAE` instance.

Throws:
- `ErrorException` if `length(induced_set_sizes) != length(latent_dims)`.
"""
function SetVAE(input_dim::Int, hidden_dim::Int, heads::Int, induced_set_sizes::AbstractVector{<:Integer}, 
    latent_dims::AbstractVector{<:Integer}, zed_depth::Int, zed_hidden_dim::Int, activation::Function=Flux.relu, 
    expansion_depth::Int=1, expansion_hidden_dim::Int=0, n_mixtures::Int=5, prior_dim::Int=3, output_activation::Function=identity) 
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
        create_mlp(input_dim, expansion_hidden_dim, expansion_depth, hidden_dim, activation),#Flux.Dense(input_dim, hidden_dim),
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
        create_mlp(hidden_dim, expansion_hidden_dim, expansion_depth, input_dim, output_activation) #Flux.Dense(hidden_dim, input_dim, x->output_activation(x))
    )
    return SetVAE(encoder, decoder, prior)
end

"""
`setvae_constructor_from_named_tuple(; idim, hdim, heads, is_sizes, zdims, vb_depth, vb_hdim, activation, expansion_depth=1, expansion_hidden_dim=0, n_mixtures=5, prior_dim=32, output_activation=identity, prior="mog", init_seed=nothing, kwargs...)`

Build SetVAE from a named-tuple style configuration (typically loaded from JSON/TOML/YAML).

Keyword arguments (required):
- `idim`: input feature dimension.
- `hdim`: hidden feature width.
- `heads`: number of attention heads.
- `is_sizes`: induced set sizes (vector).
- `zdims`: latent dimensions (vector).
- `vb_depth`: variational bottleneck depth.
- `vb_hdim`: variational bottleneck hidden dimension.
- `activation`: activation function name as string (e.g., `"relu"`, `"sigmoid"`).

Keyword arguments (optional):
- `expansion_depth`: depth of expansion MLPs (default `1`).
- `expansion_hidden_dim`: hidden width of expansion MLPs (default `0`).
- `n_mixtures`: number of mixture components (default `5`).
- `prior_dim`: prior dimension (default `32`).
- `output_activation`: output activation (default `identity`).
- `prior`: prior type (default `"mog"`; currently unused).
- `init_seed`: random seed for initialization (default `nothing`; if provided, seed is reset after construction).
- `kwargs...`: additional arguments (ignored).

Returns:
- Constructed `SetVAE` instance.
"""
function setvae_constructor_from_named_tuple(
    ;idim, hdim, heads, is_sizes, zdims, vb_depth, vb_hdim, activation, 
    expansion_depth=1, expansion_hidden_dim=0, n_mixtures=5, prior_dim=32, 
    output_activation=identity, prior="mog", init_seed=nothing, kwargs...)
    #n_mixtures = (n_mixtures === nothing) ? 5 : n_mixtures
    #output_activation = (output_activation === nothing) ? identity : output_activation
    activation = eval(:($(Symbol(activation))))
    (init_seed !== nothing) ? Random.seed!(init_seed) : nothing
    model = SetVAE(
        idim, hdim, heads, is_sizes, zdims, vb_depth, vb_hdim, 
        activation, expansion_depth, expansion_hidden_dim, n_mixtures, 
        prior_dim, output_activation#, prior_type
        )
    (init_seed !== nothing) ? Random.seed!() : nothing
    return model
end


######################################
### Score functions and evaluation ###
######################################

"""
`reconstruct(svae::SetVAE, x::AbstractArray{T}, x_mask::Mask=nothing; kwargs...) where T <: AbstractFloat`

Reconstruct a set batch by running forward pass in test mode (evaluation mode).

Arguments (positional):
- `svae`: SetVAE model instance.
- `x`: input tensor `(d, n, bs)`.
- `x_mask`: optional boolean mask `(1, n, bs)` (default `nothing`).

Keyword arguments:
- `kwargs...`: additional arguments passed to model forward pass (e.g., `β`).

Returns:
- Reconstructed set batch `x̂` with the same shape as input `x`.

Notes:
- Model is switched to Flux test mode before forward pass and restored after.
- In test mode, dropout and other stochastic layers are disabled.
"""
function reconstruct(svae::SetVAE, x::AbstractArray{T}, x_mask::Mask=nothing; kwargs...) where T <: AbstractFloat
    Flux.testmode!(svae, true)
    x̂, _, _, _ = svae(x, x_mask; kwargs...)
    Flux.testmode!(svae, false)
    return x̂
end



function reconstruct_and_log(model::SetVAE, x::AbstractArray{T}, x_mask::Mask, logpdf; β=1f0) where T <: AbstractFloat
    Flux.testmode!(model, true)
    x̂, ℒₖₗ, ℒₖₗₛ, _ = x_mask === nothing ? model(x; β=β) : model(x, x_mask; β=β)
    ℒ_rec = x_mask === nothing ? logpdf(x̂, x) : logpdf(x̂, x, x_mask, x_mask)
    loss = ℒ_rec + ℒₖₗ
    logs = (ℒ=loss, ℒ_rec=ℒ_rec, ℒₖₗ=ℒₖₗ, ℒₖₗₛ=ℒₖₗₛ, β=β)
    Flux.testmode!(model, false)
    return x̂, loss, logs
end



"""
`transform_and_reconstruct(vae::SetVAE, data::AbstractArray; testmode=true)`

Reconstruct all samples in a dataset using batched evaluation.

Iterates over `data` with batchsize=1, applies `reconstruct` to each sample, and returns CPU results.

Arguments (positional):
- `vae`: SetVAE model instance.
- `data`: collection of sets (iterable or array; will be wrapped in DataLoader).

Keyword arguments:
- `testmode`: whether to switch model into Flux test mode (default `true`). In test mode, stochastic layers are disabled.

Returns:
- Vector of reconstructed sets moved to CPU, one output (squeezed) per input sample.

Notes:
- Uses `transform_batch(batch, true)` to extract `(x, x_mask)` from each sample.
- Outputs are processed with `Flux.squeezebatch` to remove singleton batch dimension.
- All results are moved to CPU before return.
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