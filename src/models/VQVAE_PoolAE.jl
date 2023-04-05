stopgrad(x) = x
Zygote.@nograd stopgrad

function (q::Quantizer)(z::AbstractArray{<:Real}, y::T) where T<:Real # supervised forward pass
    # flatten input zₑ #d, n, bs = size(zₑ)
    size_ = size(z)
    d, n_elements = size_[1], prod(size_[2:end]) # now it works for nD array

    zₑ = reshape(z, d, n_elements)
    # Compute distance (e - x)² = e² - 2ex + x²
    dist = (sum(zₑ .^2, dims=1) 
            .+ sum(q.embedding .^2, dims=1)' 
            .- (2 .* q.embedding' * zₑ ))
    
    # Supervised encodings
    close_indices = Zygote.ignore(()->(Matrix([CartesianIndex(y, 1);;])))
    close_encodings = zeros_like(zₑ, (size(q.embedding, 2), n_elements))
    Zygote.ignore(()->(close_encodings[close_indices] .+= 1)) # need to exclude gradient or it will crash

    # Encoding # unsupervised version
    encoding_indices = argmin(dist, dims=1)
    encodings = zeros_like(zₑ, (size(q.embedding, 2), n_elements)) # there has to be transposition or dropdims
    Zygote.ignore(()->(encodings[encoding_indices] .+= 1)) # need to exclude gradient or it will crash

    indicator = 1 .- (encoding_indices .== close_indices)[:]

    # Quantize and unflatten
    quantized = q.embedding * encodings # (d, e) * (e, bs) -> (d, bs)
    quantized = reshape(quantized, size_)

    close_quantized = q.embedding * close_encodings
    close_quantized = reshape(close_quantized, size_)

    return quantized, close_quantized, indicator
end

struct VQ_PoolAE
    encoder::PoolEncoder
    quantizer::Quantizer
    generator
    decoder
end

Flux.@functor VQ_PoolAE

function (model::VQ_PoolAE)(x::AbstractArray{T,2}) where T<:Real
    d, n = size(x) 
    zₑ =  model.encoder(x) # (d, n) -> (dz, 1)
    quantized, _ = model.quantizer(zₑ) # (dz, 1) -> (dz, 1)
    quantized = zₑ + stopgrad(quantized - zₑ) # gradient bypass
    μ, Σ = model.generator(quantized)  # (dz, 1) -> ((dd, 1), (dd, 1))
    zₛ = μ .+ Σ .* randn_like(μ, (size(μ, 1), n)) # ((dd, 1), (dd, 1)) -> (dd, n)
    x̂ = model.decoder(zₛ)  # (dd, n) -> (d, n)
    return x̂
end

function loss_gradient(model::VQ_PoolAE, x::AbstractArray{T,2}; β=0.25) where T<:Real
    d, n = size(x) 
    zₑ =  model.encoder(x)
    quantized, _ = model.quantizer(zₑ)
    
    # Embedding loss (e -> zₑ) + Commitment loss (zₑ -> e)
    e_latent_loss = Flux.Losses.mse(stopgrad(quantized), zₑ) # gradient propagation to encoder
    q_latent_loss = Flux.Losses.mse(quantized, stopgrad(zₑ)) # gradient propagation to embedding

    𝓛ₗₐₜₑₙₜ = q_latent_loss + β * e_latent_loss

    # Bypass of gradients from decoder to encoder 
    quantized = zₑ + stopgrad(quantized - zₑ)

    μ, Σ = model.generator(quantized)  # (dz, 1) -> ((dd, 1), (dd, 1))
    zₛ = μ .+ Σ .* randn_like(μ, (size(μ, 1), n)) # ((dd, 1), (dd, 1)) -> (dd, n)

    x̂ = model.decoder(zₛ)  
    𝓛 = Flux3D.chamfer_distance(x, x̂) + 𝓛ₗₐₜₑₙₜ # in paper was mse(x, x̂) / data_variance
end


function loss_gradient(model::VQ_PoolAE, x::AbstractArray{<:Real,2}, y::Real; β=0.25, γ=0.1)
    d, n = size(x) 
    zₑ =  model.encoder(x)
    pred_quantized, true_quantized, indicator = model.quantizer(zₑ, y)
    
    # Embedding loss (e -> zₑ) + Commitment loss (zₑ -> e)
    e_latent_loss = Flux.Losses.mse(stopgrad(true_quantized), zₑ) # gradient propagation to encoder
    q_latent_loss = Flux.Losses.mse(true_quantized, stopgrad(zₑ)) # gradient propagation to embedding

    d_latent_loss = indicator[1] .* Flux.Losses.mse(stopgrad(pred_quantized), zₑ) # gradient propagation to encoder
    x_latent_loss = indicator[1] .* Flux.Losses.mse(pred_quantized, stopgrad(zₑ)) # gradient propagation to embedding


    𝓛ₗₐₜₑₙₜ = q_latent_loss + β * e_latent_loss - x_latent_loss - γ * d_latent_loss

    # Bypass of gradients from decoder to encoder 
    quantized = zₑ + stopgrad(true_quantized - zₑ)

    μ, Σ = model.generator(quantized)  # (dz, 1) -> ((dd, 1), (dd, 1))
    zₛ = μ .+ Σ .* randn_like(μ, (size(μ, 1), n)) # ((dd, 1), (dd, 1)) -> (dd, n)

    x̂ = model.decoder(zₛ)  
    𝓛 = Flux3D.chamfer_distance(x, x̂) + 𝓛ₗₐₜₑₙₜ # in paper was mse(x, x̂) / data_variance
end



function loss_ema(model::VQ_PoolAE, x::AbstractArray{T,2}; β=0.25f0, γ::T=T(0.99), ϵ::T=T(1e-5), trainmode::Bool=true) where T<:Real
    d, n = size(x) 
    zₑ =  model.encoder(x)
    quantized, encodings = model.quantizer(zₑ)
    # Commitment loss (zₑ -> embedding)
    e_latent_loss = Flux.Losses.mse(stopgrad(quantized), zₑ) # gradient propagation to encoder
    𝓛ₗₐₜₑₙₜ = β * e_latent_loss

    if trainmode 
        # Exponential Moving Average update
        Zygote.ignore(()->ema_update!(model.quantizer, zₑ, encodings; γ=γ, ϵ=ϵ))
        # Zygote.ignore is probably not needed but .... just to be sure
    end

    # Bypass of gradients from decoder to encoder 
    quantized = zₑ + stopgrad(quantized - zₑ)

    μ, Σ = model.generator(quantized)  # (dz, 1) -> ((dd, 1), (dd, 1))
    zₛ = μ .+ Σ .* randn_like(μ, (size(μ, 1), n)) # ((dd, 1), (dd, 1)) -> (dd, n)

    x̂ = model.decoder(zₛ)  
    
    𝓛 = Flux3D.chamfer_distance(x, x̂) + 𝓛ₗₐₜₑₙₜ 
end

function vq_poolae_constructor_from_named_tuple(;idim=3, prpdim=64, prpdepth=3, popdim=128, popdepth=3, zdim=32, 
    decdim=64, decdepth=3, poolf="mean-max", gen_sigma="scalar", n_embed=128, ema=false, activation::String="swish", 
    init_seed=nothing, kwargs...)
    """
    -----------------------
    VQ_PoolAE constructor
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

    activation = eval(:($(Symbol(activation))))
    (init_seed !== nothing) ? Random.seed!(init_seed) : nothing
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
    if ema
        emb = randn(Float32, popdim, n_embed) 
        quan = VectorQuantizerEMA(emb, zeros(Float32, n_embed, 1), rand_like(emb))
    else
        emb = randn(Float32, popdim, n_embed) 
        #emb = Float32.(rand(Uniform(-1/n_embed, 1/n_embed), zdim, n_embed))
        quan = VectorQuantizer(emb)
    end

    encoder = PoolEncoder(prepool, fpool, postpool)
    generator = SplitLayer(popdim, (zdim, gen_out_dim), (identity, Flux.softplus))
    (init_seed !== nothing) ? Random.seed!() : nothing
    return VQ_PoolAE(encoder, quan, generator, decoder)
end





struct VectorGaussianQuantizerEMA <: Quantizer 
    embedding_mean::AbstractArray{<:Real, 2}
    embedding_std::AbstractArray{<:Real, 2}
    n::AbstractArray{<:Real, 2}
    m::AbstractArray{<:Real, 2}
    s::AbstractArray{<:Real, 2}
end

Flux.@functor VectorGaussianQuantizerEMA
Flux.trainable(q::VectorGaussianQuantizerEMA) = ()#q.embedding_mean, q.embedding_std, 
# no tranable of ema

function (q::VectorGaussianQuantizerEMA)(z::AbstractArray{<:Real,2})
    # flatten input zₑ #d, n, bs = size(zₑ)
    size_ = size(z)
    d, n_elements = size_[1], prod(size_[2:end]) # now it works for nD array

    zₑ = reshape(z, d, n_elements)

    ss = (q.embedding_mean .- zₑ).^2
    norm_const = 0.5 .*log.(prod(q.embedding_std, dims=1))
    dist = (Flux.sum(-0.5 .*(log(2*pi) .+ ss./q.embedding_std), dims=1) .- norm_const)'
    # Flux.sum(-0.5 .*(log(2*pi) .+ ((μ_q .- zₑ).^2)./Σ_q), dims=1) .- 0.5 .*log.(prod(Σ_q, dims=1))
    
    # Encoding
    encoding_indices = argmin(dist, dims=1)
    encodings = zeros_like(zₑ, (size(q.embedding_mean, 2), n_elements)) # there has to be transposition or dropdims
    Zygote.ignore(()->(encodings[encoding_indices] .+= 1)) # need to exclude gradient or it will crash

    # Quantize and unflatten
    quantized_mean = q.embedding_mean * encodings # (d, e) * (e, 1) -> (d, 1)
    quantized_std = q.embedding_std * encodings # (d, e) * (e, 1) -> (d, 1)
    return reshape(quantized_mean, size_), reshape(quantized_std, size_), encodings

end

function ema_update!(q::VectorGaussianQuantizerEMA, z, encodings; γ=0.99f0, ϵ=1f-5) 
    q.n .= q.n .* γ .+ (1f0 - γ) .* sum(encodings, dims=2) # (e, 1)
    N = sum(q.n)
    q.n .= (q.n .+ ϵ) ./ (N .+ size(q.embedding_mean, 2) .* ϵ) .* N # (e, 1)

    d = size(z, 1)
    dw = reshape(z, d, :) * encodings' # (d, n_elements) x (n_elements, e) -> (d, e)

    q.m .= q.m .* γ .+ (1f0 - γ) .* dw # (d, e)
    q.s .= q.s .* γ .+ (1f0 - γ) .* (dw .- q.m).^2    # dw is sum but only for n>1
    # FIXME for gpu vectorized training this (q.s) needs to be fixed
    q.embedding_mean .= q.m ./ q.n' # (d, e) .* (1, e) -> (d, e)
    q.embedding_std .= sqrt.(q.s ./ q.n') # (d, e) .* (1, e) -> (d, e)
end


struct VGQ_PoolAE
    encoder::PoolEncoder
    quantizer::VectorGaussianQuantizerEMA
    decoder
end

Flux.@functor VGQ_PoolAE

function (model::VGQ_PoolAE)(x::AbstractArray{<:Real, 2})
    d, n = size(x) # (d, n)
    zₑ =  model.encoder(x) # (d, n) -> (zdim, 1) | PoolEncoder
    μ_q, Σ_q, encodings = model.quantizer(zₑ) # (zdim, 1) -> ((zdim, 1), (zdim, 1), (zdim, e))

    quantized = μ_q .+ Σ_q .* randn_like(zₑ, (size(μ_q,1), n)) # ((zdim, 1) .+ (zdim, 1) .* (zdim, n)) -> (zdim, n)
    quantized = zₑ .+ stopgrad(quantized .- zₑ) # ((zdim, 1) .+ (zdim, n) .- (zdim, 1)) -> (zdim, n)
    x̂ = model.decoder(quantized)  # (zdim, n) -> (d, n)
end

function loss_ema(model::VGQ_PoolAE, x::AbstractArray{T,2}; β=0.25f0, γ::T=T(0.99), ϵ::T=T(1e-5), trainmode::Bool=true) where T<:Real
    d, n = size(x) # (d, n)
    zₑ =  model.encoder(x) # (d, n) -> (zdim, 1) | PoolEncoder
    μ_q, Σ_q, encodings = model.quantizer(zₑ) # (zdim, 1) -> ((zdim, 1), (zdim, 1), (zdim, e))
    # Commitment loss (zₑ -> embedding)
    e_latent_loss = (Flux.sum(-0.5 .*(log(2*pi) .+ ((μ_q .- zₑ).^2)./Σ_q)) - 0.5*log(prod(Σ_q))) # scalar
    # equivalent to -logpdf(MultvariateNormal(μ_q, Σ_q), zₑ)
    𝓛ₗₐₜₑₙₜ = β * e_latent_loss / n # scale loss function

    if trainmode 
        # Exponential Moving Average update
        Zygote.ignore(()->ema_update!(model.quantizer, zₑ, encodings; γ=γ, ϵ=ϵ)) 
    end

    # Bypass of gradients from decoder to encoder 
    quantized = μ_q .+ Σ_q .* randn_like(zₑ, (size(μ_q,1), n)) # ((zdim, 1) .+ (zdim, 1) .* (zdim, n)) -> (zdim, n)
    quantized = zₑ .+ stopgrad(quantized .- zₑ) # ((zdim, 1) .+ (zdim, n) .- (zdim, 1)) -> (zdim, n)
    x̂ = model.decoder(quantized)  # (zdim, n) -> (d, n)
    𝓛 = Flux3D.chamfer_distance(x, x̂) + 𝓛ₗₐₜₑₙₜ 
end

function vgq_poolae_constructor_from_named_tuple(
    ;idim=3, prpdim=64, prpdepth=3, popdim=64, popdepth=3, zdim=3, n_embed=128, 
    decdim=64, decdepth=3, poolf="mean-max", activation="swish", init_seed=nothing, kwargs...)
    
    activation = eval(:($(Symbol(activation))))
    (init_seed !== nothing) ? Random.seed!(init_seed) : nothing

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
        [Flux.Dense(popdim, popdim, activation) for i=1:popdepth-2]...,
        Flux.Dense(popdim, zdim)
    )
    
    decoder = Flux.Chain(
        Flux.Dense(zdim, decdim, activation), 
        [Flux.Dense(decdim, decdim, activation) for i=1:decdepth-2]...,
        Flux.Dense(decdim, idim), 
    )

    emb_m = sample_sphere(zdim, n_embed)
    emb_s = ones(Float32, zdim, n_embed);
    quan = VectorGaussianQuantizerEMA(
        emb_m, 
        emb_s, 
        zeros(Float32, n_embed, 1), 
        rand_like(emb_m), 
        ones_like(emb_m)
    );

    encoder = PoolEncoder(prepool, fpool, postpool)
    model = VGQ_PoolAE(encoder, quan, decoder)

    (init_seed !== nothing) ? Random.seed!() : nothing
    return model
end





function Base.show(io::IO, m::VectorGaussianQuantizerEMA)
    print(io, "VectorGaussianQuantizerEMA")
    print(io, "\n- embedding_mean = $(m.embedding_mean |> size) | $(m.embedding_mean  |> typeof)\
    | mean ~ $(m.embedding_mean  |> Flux.mean)")
    print(io, "\n- embedding_std = $(m.embedding_std |> size) | $(m.embedding_std |> typeof) \
    | mean ~ $(m.embedding_std |> Flux.mean)")
    print(io, "\n- nᵗᵢ = $(m.n |> size) | $(m.n  |> typeof) | mean ~ $(m.n  |> Flux.mean)")
    print(io, "\n- mᵗᵢ = $(m.m |> size) | $(m.m  |> typeof) | mean ~ $(m.m  |> Flux.mean)")
    print(io, "\n- sᵗᵢ = $(m.s |> size) | $(m.s  |> typeof) | mean ~ $(m.s  |> Flux.mean)")
end