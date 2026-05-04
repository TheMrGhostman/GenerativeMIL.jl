struct SetGenerator
    expansion # can be Flux.Dense, identity function or whatever
    layers
    reduction # can be Flux.Dense, identity function or whatever
end

Flux.@functor SetGenerator

function (m::SetGenerator)(x::AbstractArray{<:Real}, 
    x_mask::Union{AbstractArray{Bool}, Nothing}=nothing; const_module::Module=Base)

    x = mask(m.expansion(x), x_mask)
    for layer in m.layers
        x, _ = layer(x, x_mask, const_module=const_module)
        # we don't need Induced Set
    end
    x = mask(m.reduction(x), x_mask)
    return x
end

struct SetDiscriminator
    expansion
    features # feature matching
    reduction
    class # fake/real
end

Flux.@functor SetDiscriminator

function (m::SetDiscriminator)(x::AbstractArray{<:Real}, 
    x_mask::Union{AbstractArray{Bool}, Nothing}=nothing; const_module::Module=Base)

    features = mask(m.expansion(x), x_mask)
    for layer in m.layers
        features, _ = layer(x, x_mask, const_module=const_module)
        # we don't need Induced Set
    end
    features = mask(m.reduction(features), x_mask)
    # FIXME global pooling
    x = m.class(x) # FIXME
    return featuers
end


struct SetGAN
    prior
    generator 
    discriminator
end

Flux.@functor SetGAN

function SetGAN(input_dim::Int, hidden_dim::Int, heads::Int, induced_set_sizes::Array{Int,1}, 
    activation::Function=Flux.relu, prior_type="gaussian", prior_dim::Int=3, output_activation::Function=identity)

    # GENERATOR
    ## blocks
    generator_blocks = []
    for iss in induced_set_sizes 
        isab = InducedSetAttentionBlock(iss, hidden_dim, heads)
        push!(generator_blocks, isab)
    end
    ## expansion
    ## reduction
    generator = SetGenerator(
        Flux.Dense(prior_dim, hidden_dim, activation),
        generator_blocks,
        Flux.Dense(hidden_dim, input_dim, output_activation)
    )
    

    # DISCRIMINATOR
    ## features
    discriminator_blocks = []
    for iss in reverese(induced_set_sizes)
        isab = InducedSetAttentionBlock(iss, hidden_dim, heads)
        push!(discriminator_blocks, isab)
    end
    ## reduction
    ## class
    discriminator = SetDiscriminator(
        Flux.Dense(input_dim, hidden_dim, activation),
        discriminator_blocks,
        Flux.Dense(hidden_dim, 1, Flux.sigmoid) # FIXME 
        # instead of Dense reduction i can use mab with induced set of size 1
    )   


    # prior
    @assert prior_type in ["gaussian", "gaussian_sphere", "uniform"] "Unknown prior distribution" 
    # maybe mixture of gaussians
    #TODO add prior


end