struct HierarchicalEncoder
    expansion::Flux.Dense
    layers
end

Flux.@functor HierarchicalEncoder

function (m::HierarchicalEncoder)(x::AbstractArray{<:Real}, x_mask::AbstractArray{Bool})
    x = m.expansion(x) .* x_mask
    h_encs = Zygote.Buffer(Array{Any}(undef, length(m.layers)))
    for (i, layer) in enumerate(m.layers)
        x, h_enc = layer(x, x_mask)
        h_encs[length(m.layers) - i + 1] = h_enc
    end
    return x, h_encs
end

AbstractTrees.children(m::HierarchicalEncoder) = (("Expansion", m.expansion), m.layers)
AbstractTrees.printnode(io::IO, m::HierarchicalEncoder) = print(io, "HierarchicalEncoder - ($(length(m.layers)) depth)")

struct HierarchicalDecoder
    expansion::Flux.Dense # expansion of prior samples
    layers
    reduction::Flux.Dense
end

Flux.@functor HierarchicalDecoder

function (m::HierarchicalDecoder)(z::AbstractArray{<:Real}, h_encs, x_mask::AbstractArray{Bool})
    x = m.expansion(z) .* x_mask
    zs = []
    klds = []
    kld_loss = 0
    for (layer, h_enc) in zip(m.layers, h_encs)
        x, kld, _, z = layer(x, h_enc, x_mask) 
        Zygote.ignore() do
            push!(klds, kld)
            push!(zs, z)
        end
        kld_loss += kld
    end
    x = m.reduction(x) .* x_mask
    return x, klds, zs, kld_loss
end

AbstractTrees.children(m::HierarchicalDecoder) = (("Expansion", m.expansion), m.layers, ("Reduction", m.reduction))
AbstractTrees.printnode(io::IO, m::HierarchicalDecoder) = print(io, "HierarchicalDecoder - ($(length(m.layers)) depth)")


struct SetVAE 
    encoder::HierarchicalEncoder
    decoder::HierarchicalDecoder
    prior::AbstractPriorDistribution
end

Flux.@functor SetVAE

function loss(vae::SetVAE, x::AbstractArray{<:Real}, x_mask::AbstractArray{Bool}, β::Float32=1f0)
    _, h_encs = vae.encoder(x, x_mask) # no need for x
    #h_encs = reverse(h_encs)
    _, sample_size, bs = size(x_mask)
    z = vae.prior(sample_size, bs)
    x̂, _, _, klds = vae.decoder(z, h_encs, x_mask)
    loss = masked_chamfer_distance_cpu(x, x̂, x_mask, x_mask) +  β * klds
    return loss, klds
end


######################################
###          Constructors          ###
######################################

function SetVAE(input_dim::Int, hidden_dim::Int, heads::Int, induced_set_sizes::Array{Int,1}, 
    latent_dims::Array{Int,1}, zed_depth::Int, zed_hidden_dim::Int, activation::Function=Flux.relu, 
    n_mixtures::Int=5, prior_dim::Int=3, output_activation::Function=identity) 
    #prior_type::AbstractPriorDistribution=MixtureOfGaussians)

    (length(induced_set_sizes) !=length(latent_dims)) ? error("induced sets and latent dims have different lengths") : nothing

    # ENCODER
    enc_blocks = []
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
    dec_blocks = []
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

function reconstruct(vae::SetVAE, x::AbstractArray{<:Real}, x_mask::AbstractArray{Bool})
    _, h_encs = vae.encoder(x, x_mask)
    _, sample_size, bs = size(x_mask)
    z = vae.prior(sample_size, bs)
    x̂, _, _, _ = vae.decoder(z, h_encs, x_mask)
    return x̂
end

function transform_and_reconstruct(vae::SetVAE, data::Vector; testmode=true)
    # expect to get output from GroupAD.Models.unpack_mill(tr_data) or list of "sets"
    dataloader = Flux.Data.DataLoader(data, batchsize=1) 
    # we could iterate via data itself (batchsize=1) but we decided to use dataloader instaed
    X̂ = []
    vae = (testmode) ? Flux.testmode!(vae, true) : vae # to testmode
    for batch in dataloader
        x, x_mask = transform_batch(batch, true) # i copied clone of transform_batch into models.utils
        x̂ = reconstruct(vae, x, x_mask)
        push!(X̂, x̂ |> Flux.squeezebatch |>cpu)
    end
    return X̂
end