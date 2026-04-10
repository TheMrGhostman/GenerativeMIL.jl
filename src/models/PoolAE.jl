struct PoolModel
    encoder::PoolEncoder # building_blocks/pooling_layers
    generator #context generator
    decoder
end

Flux.@layer PoolModel

AbstractTrees.children(m::PoolModel) = (("Encoder", m.encoder), ("Generator", m.generator), ("Decoder", m.decoder))
AbstractTrees.printnode(io::IO, m::PoolModel) = print(io, "PoolModel")

function (m::PoolModel)(x::AbstractArray{T, 3}; kld::Bool=false) where T <: Real
    _, n, bs = size(x)
    h = m.encoder(x)
    μ, Σ = m.generator(h)
    z = μ .+ Σ .* randn_like(x, (size(μ, 1), n, bs)) # MLUtils.randn_like
    x̂ = m.decoder(z)
    if kld
        𝓛ₖₗ = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σ.^2) - μ.^2  - Σ.^2, dims=1)) 
        return x̂, 𝓛ₖₗ
    else
        return x̂
    end 
end

loss(model::PoolModel, x::AbstractArray{<:Real, 3}) = chamfer_distance(model(x), x)

function loss_with_kld(model::PoolModel, x::AbstractArray{<:Real, 3}; β::Float32=1f0, logging::Bool=false)
    x̂, 𝓛ₖₗ = model(x, kld=true)
    𝓛ᵣₑ = chamfer_distance(x̂, x)
    𝓛 = 𝓛ᵣₑ .+ β .* 𝓛ₖₗ
    return (logging) ? (𝓛, 𝓛ᵣₑ, 𝓛ₖₗ) : 𝓛
end

function PoolModel(idim, prpdim, prpdepth, popdim, popdepth, zdim, decdim, decdepth, 
    poolf="mean-max",  gen_sigma="scalar", activation::Function=swish)
    """
    -----------------------
    PoolModel constructor
    -----------------------
    idim        -> input dimensions
    prpdim      -> hidden dimension for Pre Pooling part (PreP)
    prpdepth    -> number of layers in PreP
    popdim      -> hidden dimension for Post Pooling part (PostP)
    popdepth    -> number of layers in PostP
    zdim        -> dimension of latent space
    decdim      -> hidden dimension for decoder part
    decdepth    -> number of layers in decoder
    poolf       -> pooling function (\"mean-max\", \"mean\", \"max\", \"attention\", \"PMA\")
    gen_sigma   -> type of variance in generator (\"scalar\" or \"diag\")
    activation  -> activation function 
    """
    
    if gen_sigma == "scalar"
        gen_out_dim = 1
    elseif gen_sigma == "diag"
        gen_out_dim = zdim
    else
        error("Unkown type of vairance")
    end

    prepool = Flux.Chain(
        Flux.Dense(idim, prpdim, activation), 
        [Flux.Dense(prpdim, prpdim, activation) for i=1:prpdepth-1]...
    )

    multiplier=1
    if poolf=="mean-max"
        fpool = x->cat(mean(x, dims=2), maximum(x, dims=2), dims=1)
        multiplier = 2
    elseif poolf=="mean"
        fpool = x->mean(x, dims=2)
    elseif poolf=="max"
        fpool = x->maximum(x, dims=2)
    elseif poolf=="attention"
        fpool = AttentionPooling(Flux.Chain(
                Dense(prpdim, prpdim, activation),
                Dense(prpdim,1)
                ))
    elseif poolf=="PMA"
        fpool = PMA(1, prpdim, 4)
    else
        error("Unknown pooling function")
    end

    postpool = Flux.Chain(
        Flux.Dense(multiplier*prpdim, popdim, activation), 
        [Flux.Dense(popdim, popdim, activation) for i=1:popdepth-1]...
    )

    decoder = Flux.Chain(
        Flux.Dense(zdim, decdim, activation), 
        [Flux.Dense(decdim, decdim, activation) for i=1:decdepth-2]...,
        Flux.Dense(decdim, idim), 
    )

    encoder = PoolEncoder(prepool, fpool, postpool)
    generator = SplitLayer(popdim, (zdim, gen_out_dim), (identity, Flux.softplus))
    return PoolModel(encoder, generator, decoder)
end


function poolmodel_constructor_from_named_tuple(;idim, prpdim, prpdepth, popdim, popdepth, zdim, decdim, decdepth, 
    poolf="mean-max",  gen_sigma="scalar", activation="swish", init_seed=nothing, kwargs...)

    activation = eval(:($(Symbol(activation))))
    (init_seed !== nothing) ? Random.seed!(init_seed) : nothing
    model = PoolModel(idim, prpdim, prpdepth, popdim, popdepth, zdim, decdim, decdepth, 
        poolf, gen_sigma, activation
        )    
    (init_seed !== nothing) ? Random.seed!() : nothing
    return model
end
