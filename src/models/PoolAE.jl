struct PoolModel<:AbstractGenModel
    encoder::PoolEncoder # building_blocks/pooling_layers
    generator #context generator
    decoder
end

Flux.@layer PoolModel

AbstractTrees.children(m::PoolModel) = (("Encoder", m.encoder), ("Generator", m.generator), ("Decoder", m.decoder))
AbstractTrees.printnode(io::IO, m::PoolModel) = print(io, "PoolModel")

function (m::PoolModel)(x::AbstractArray{Float32, 3}; kld::Bool=false)
    _, n, bs = size(x)
    h = m.encoder(x)
    μ, Σ = m.generator(h)
    z = μ .+ Σ .* MLUtils.randn_like(h, (size(μ, 1), n, bs)) # MLUtils.randn_like
    x̂ = m.decoder(z)
    if kld
        ℒₖₗ = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σ.^2) .- μ.^2  .- Σ.^2, dims=1)) 
        return x̂, ℒₖₗ
    else
        return x̂
    end 
end

loss(model::PoolModel, x::AbstractArray{Float32, 3}; loss_function::Function=chamfer_distance, kwargs...) = loss_function(model(x), x)

function loss_with_logging(model::PoolModel, x::AbstractArray{Float32, 3}; loss_function::Function=chamfer_distance, kwargs...)
    x̂ = model(x)
    ℒ_rec = loss_function(x̂, x)
    return ℒ_rec, (;ℒ_rec = ℒ_rec)
end


function elbo_with_logging(model::PoolModel, x::AbstractArray{Float32, 3}; β::Float32=1f0, logpdf::Function=chamfer_distance, kwargs...)
    # not sure if ELBO makes sense for this model but we can still compute it if we want to
    x̂, ℒ_kld = model(x; kld=true)
    ℒ_rec = logpdf(x̂, x)
    return ℒ_rec + β * ℒ_kld , (ℒ_rec = ℒ_rec, ℒ_kld = ℒ_kld, β = β)
end

function optim_step(model::PoolModel, batch::AbstractArray{Float32, 3}, opt::NamedTuple, logpdf::Function, device::Function=cpu; kwargs...)
    # 1) move data to device
    batch = batch |> device
    # 2) compute gradients
    (loss, logs), (∇model, ∇data) = Zygote.withgradient(model, batch) do m, x
        loss_with_logging(m, x; loss_function=logpdf)
    end
    # 3) update weights
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

function valid_step(model::PoolModel, dataloader::DataLoader, logpdf::Function; device::Function=cpu, kwargs...)
    ℒ, ℒ_rec = 0, 0
    for batch in dataloader
        x = batch |> device
        loss, logs = loss_with_logging(model, x; loss_function=logpdf)
        ℒ += loss
        ℒ_rec += logs.ℒ_rec
    end
    n = length(dataloader)
    logs = (;ℒᵥ = ℒ/n, ℒᵥ_rec = ℒ_rec/n)
    return logs, ℒ/n # total loss for early stopping
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
