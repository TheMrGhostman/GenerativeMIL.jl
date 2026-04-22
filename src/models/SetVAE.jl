const BetaArg = Union{AbstractFloat,AbstractVector{<:AbstractFloat}}

struct HierarchicalEncoder{E,L}
    expansion::E
    layers::L
end

Flux.@layer HierarchicalEncoder

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

struct HierarchicalDecoder{E,L,R}
    expansion::E # expansion of prior samples
    layers::L
    reduction::R
end

Flux.@layer HierarchicalDecoder

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


struct SetVAE{E<:HierarchicalEncoder, D<:HierarchicalDecoder, P<:AbstractPriorDistribution} <: AbstractGenModel
    encoder::E
    decoder::D
    prior::P
end

AbstractTrees.children(m::HierarchicalDecoder) = (("Expansion", m.expansion), m.layers, ("Reduction", m.reduction))
AbstractTrees.printnode(io::IO, m::HierarchicalDecoder) = print(io, "HierarchicalDecoder - ($(length(m.layers)) depth)")

Flux.@layer SetVAE

function _forward_encoder_and_prior(svae::SetVAE, x::AbstractArray{T}, x_mask::Mask) where T <: AbstractFloat
    _, h_encs = svae.encoder(x, x_mask) 
    _, sample_size, bs = size(x)
    z = svae.prior(sample_size, bs)
    return z, h_encs
end

function _normalize_β(β::AbstractFloat, n_layers::Int, ::Type{T}) where {T<:AbstractFloat}
    return fill(T(β), n_layers)
end

function _normalize_β(β::AbstractVector{<:AbstractFloat}, n_layers::Int, ::Type{T}) where {T<:AbstractFloat}
    length(β) == n_layers || throw(ArgumentError("Length of β ($(length(β))) must equal number of decoder layers ($n_layers)."))
    return T.(collect(β))
end

function (svae::SetVAE)(x::AbstractArray{T}, x_mask::Mask=nothing; β::BetaArg=1f0) where T <: AbstractFloat
    β_vec = _normalize_β(β, length(svae.decoder.layers), T)
    z, h_encs = _forward_encoder_and_prior(svae, x, x_mask)
    x̂, ℒₖₗₛ, zs, ℒₖₗ = svae.decoder(z, h_encs, x_mask, β_vec)
    return x̂, ℒₖₗ, ℒₖₗₛ, zs
end





function elbo_with_logging(model::SetVAE, x::AbstractArray{T,3}; β::BetaArg=1f0, logpdf::Function=chamfer_distance, kwargs... ) where T <: AbstractFloat
    x̂, ℒₖₗ, ℒₖₗₛ, _ = model(x; β=β)
    ℒ_rec = logpdf(x̂, x)
    ℒ = ℒ_rec + ℒₖₗ
    return ℒ, (ℒ = ℒ, ℒ_rec = ℒ_rec, ℒₖₗ = ℒₖₗ, ℒₖₗₛ = ℒₖₗₛ, β = β)
end


function elbo_with_logging(model::SetVAE, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool, 3}; β::BetaArg=1f0, logpdf::Function=masked_chamfer_distance, kwargs...) where T <: AbstractFloat
    x̂, ℒₖₗ, ℒₖₗₛ, _ = model(x, x_mask; β=β)
    ℒ_rec = logpdf(x̂, x, x_mask, x_mask)
    ℒ = ℒ_rec + ℒₖₗ
    return ℒ, (ℒ = ℒ, ℒ_rec = ℒ_rec, ℒₖₗ = ℒₖₗ, ℒₖₗₛ = ℒₖₗₛ, β = β)
end


function optim_step(model::SetVAE, batch::AbstractArray{T,3}, opt::NamedTuple, logpdf::Function, device::Function=cpu; β=1f0, kwargs...) where T <: AbstractFloat
    batch = device(batch)
    (loss, logs), (∇model, ∇data) = Zygote.withgradient(model, batch) do m, x
        elbo_with_logging(m, x; logpdf=logpdf, β=β)
    end
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

function optim_step(model::SetVAE, batch::Tuple{AbstractArray{T,3}, AbstractArray{Bool,3}}, opt::NamedTuple, logpdf::Function, device::Function=cpu; β=1f0, kwargs...) where T <: AbstractFloat
    X, X_mask = batch
    X, X_mask = device(X), device(X_mask)
    (loss, logs), (∇model, ∇x, ∇x_mask) = Zygote.withgradient(model, X, X_mask) do m, x, x_mask
        elbo_with_logging(m, x, x_mask; logpdf=logpdf, β=β) #TODO check if x_mask will not cause issues with Zygote gradient tracking
    end
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

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

function reconstruct(vae::SetVAE, x::AbstractArray{T}, x_mask::Mask=nothing) where T <: AbstractFloat
    _, h_encs = vae.encoder(x, x_mask)
    _, sample_size, bs = size(x)
    z = vae.prior(sample_size, bs)
    x̂, _, _, _ = vae.decoder(z, h_encs, x_mask)
    return x̂
end

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