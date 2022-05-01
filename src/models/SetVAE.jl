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


struct SetVAE 
    encoder::HierarchicalEncoder
    decoder::HierarchicalDecoder
    prior::AbstractPriorDistribution
end

Flux.@functor SetVAE

function loss(vae::SetVAE, x::AbstractArray{<:Real}, x_mask::AbstractArray{Bool}, β::Float32=1f0)
    _, h_encs = vae.encoder(x, x_mask) # no need for x
    h_encs = reverse(h_encs)
    _, sample_size, bs = size(x_mask)
    z = vae.prior(sample_size, bs)
    x̂, _, _, klds = vae.decoder(z, h_encs, x_mask)
    loss = chemfer_distance(x, x̂) +  β * klds
    return loss
end

function SetVAE(input_dim::Int, hidden_dim::Int, heads::Int, induced_set_sizes::Array{Int,1}, 
    latent_dims::Array{Int,1}, zed_depth::Int, zed_hidden_dim::Int, activation::Function=Flux.relu, 
    n_mixtures::Int=5, prior_dim::Int=3)

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

    # Prior
    prior = MixtureOfGaussians(prior_dim, n_mixtures, true)

    #DECODER
    dec_blocks = []
    for (iss, zdim) in zip(reverse(induced_set_sizes), reverse(latent_dims))
        abl = AttentiveBottleneckLayer(iss, hidden_dim, heads, zdim, zed_hidden_dim, zed_depth, activation)
        push!(dec_blocks, abl)
    end
    decoder = HierarchicalDecoder(
        Flux.Dense(prior_dim, hidden_dim),
        dec_blocks,
        Flux.Dense(hidden_dim, input_dim)
    )
    return SetVAE(encoder, decoder, prior)
end