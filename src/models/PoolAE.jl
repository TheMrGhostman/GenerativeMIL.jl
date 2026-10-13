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

function loss_with_logging(model::PoolModel, x::AbstractArray{Float32, 3}, loss_function::Function=chamfer_distance; kwargs...)
    x̂ = model(x)
    ℒ_rec = loss_function(x̂, x)
    return ℒ_rec, (;ℒ = ℒ_rec, ℒ_rec = ℒ_rec)
end

function loss_with_logging(model::PoolModel, x::AbstractArray{Float32, 3}, loss_function::MMD_EMA_Loss; update_sigma::Bool=true, kwargs...)
    x̂ = model(x)
    Zygote.@ignore begin
        σₙ = compute_rbf_sigma_estimate(x̂, x)
        (update_sigma) && update_ema_sigma!(loss_function, σₙ)
    end
    ℒ_rec = loss_function(x̂, x)
    return ℒ_rec, (;ℒ = ℒ_rec, ℒ_rec = ℒ_rec, σᵣ = loss_function.σᵣ)
end


function elbo_with_logging(model::PoolModel, x::AbstractArray{Float32, 3}, logpdf::Function=chamfer_distance; β::Float32=1f0, kwargs...)
    # not sure if ELBO makes sense for this model but we can still compute it if we want to
    x̂, ℒ_kld = model(x; kld=true)
    ℒ_rec = logpdf(x̂, x)
    return ℒ_rec + β * ℒ_kld , (ℒ_rec = ℒ_rec, ℒ_kld = ℒ_kld, β = β)
end

function optim_step(model::PoolModel, batch::AbstractArray{Float32, 3}, opt::NamedTuple, logpdf, device::Function=cpu; kwargs...)
    # 1) move data to device
    batch = batch |> device
    # 2) compute gradients
    (loss, logs), (∇model, ∇data) = Zygote.withgradient(model, batch) do m, x
        loss_with_logging(m, x, logpdf; kwargs...)
    end
    # 3) update weights
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

function valid_step(model::PoolModel, dataloader::DataLoader, logpdf; device::Function=cpu, kwargs...)
    ℒ, ℒ_rec = 0, 0
    for batch in dataloader
        x = batch |> device
        loss, logs = loss_with_logging(model, x, logpdf; update_sigma=false, kwargs...)
        ℒ += loss
        ℒ_rec += logs.ℒ_rec
    end
    n = length(dataloader)
    logs = (;ℒᵥ = ℒ/n, ℒᵥ_rec = ℒ_rec/n)
    return logs, ℒ/n # total loss for early stopping
end

function reconstruct(model::PoolModel, x::AbstractArray{T}; kwargs...) where T <: AbstractFloat
    Flux.testmode!(model, true)
    x̂ = model(x; kwargs...)
    Flux.testmode!(model, false)
    return x̂
end


"""
`encode_set(model::PoolModel, x)`

Set-level encoding `h = encoder(x)` (deterministic; PoolModel is trained without KL, so `h` has no prior).

Returns:
- `h`: `(popdim, 1, batch_size)`.
"""
function encode_set(model::PoolModel, x::AbstractArray{T, 3}) where T <: AbstractFloat
    Flux.testmode!(model, true)
    try
        return model.encoder(x)
    finally
        Flux.testmode!(model, false)
    end
end

"""
`decode_from_encoding(model::PoolModel, h, cardinality::Int)`

Decode set encodings `h` into sets of `cardinality` points, exactly as in the forward pass:
`μ, Σ = generator(h)`, one `z = μ + Σ ε` per point, `x̂ = decoder(z)`.

Returns:
- `x̂`: `(idim, cardinality, batch_size)`.
- `(μ, Σ, z)`: generator parameters `(zdim, 1, bs)` / `(1 or zdim, 1, bs)` and per-point latents `(zdim, cardinality, bs)`.
"""
function decode_from_encoding(model::PoolModel, h::AbstractArray{T, 3}, cardinality::Int) where T <: AbstractFloat
    Flux.testmode!(model, true)
    try
        μ, Σ = model.generator(h)
        z = μ .+ Σ .* MLUtils.randn_like(μ, (size(μ, 1), cardinality, size(μ, 3)))
        return model.decoder(z), (μ, Σ, z)
    finally
        Flux.testmode!(model, false)
    end
end


function reconstruct_and_log(model::PoolModel,x::AbstractArray{T}, x_mask::Mask, loss_function; β=1f0) where T <: AbstractFloat
    Flux.testmode!(model, true)
    x̂ = model(x)
    ℒ_rec = loss_function(x̂, x)
    logs = (;ℒ = ℒ_rec, ℒ_rec = ℒ_rec)
    Flux.testmode!(model, false)
    return x̂, ℒ_rec, logs
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

    fpool, multiplier = _make_pooling(poolf, prpdim, activation; pool_hidden_dim=prpdim)

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
