stopgrad(x) = x
Zygote.@nograd stopgrad

abstract type Quantizer end
struct VectorQuantizer <: Quantizer 
    embedding::AbstractArray{<:Real, 2}
end

struct VectorQuantizerEMA <: Quantizer 
    embedding::AbstractArray{<:Real, 2}
    n::AbstractArray{<:Real, 2}
    m::AbstractArray{<:Real, 2}
end

Flux.@functor VectorQuantizer
Flux.@functor VectorQuantizerEMA
Flux.trainable(q::Quantizer) = (q.embedding)

function (q::Quantizer)(z::AbstractArray{<:Real})
    # flatten input zₑ #d, n, bs = size(zₑ)
    size_ = size(z)
    d, n_elements = size_[1], prod(size_[2:end]) # now it works for nD array

    zₑ = reshape(z, d, n_elements)
    # Compute distance (e - x)² = e² - 2ex + x²
    dist = (sum(zₑ .^2, dims=1) 
            .+ sum(q.embedding .^2, dims=1)' 
            .- (2 .* q.embedding' * zₑ ))

    # Encoding
    encoding_indices = argmin(dist, dims=1)
    encodings = zeros_like(zₑ, (size(q.embedding, 2), n_elements)) # there has to be transposition or dropdims
    Zygote.ignore(()->(encodings[encoding_indices] .+= 1)) # need to exclude gradient or i will crash

    # Quantize and unflatten
    quantized = q.embedding * encodings # (d, e) * (e, bs) -> (d, bs)
    quantized = reshape(quantized, size_)
    return quantized, encodings
end

function ema_update!(q::VectorQuantizerEMA, z, encodings; γ=0.99f0, ϵ=1f-5)
    q.n .= q.n .* γ .+ (1f0 - γ) .* sum(encodings, dims=2) # (e, 1)
    N = sum(q.n)
    q.n .= (q.n .+ ϵ) ./ (N .+ size(q.embedding, 2) .* ϵ) .* N # (e, 1)

    d = size(z, 1)
    dw = reshape(z, d, :) * encodings' # (d, n_elements) x (n_elements, e) -> (d, e)

    q.m .= q.m .* γ .+ (1f0 - γ) .* dw # (d, e)
    q.embedding .= q.m ./ q.n' # (d, e) .* (1, e) -> (d, e)
end

struct VQVAE
    encoder
    quantizer::Quantizer
    decoder
end

Flux.@functor VQVAE
Flux.trainable(model::VQVAE) = (model.encoder, model.quantizer, model.decoder)

function (model::VQVAE)(x::AbstractArray{T}) where T<:Real
    zₑ =  model.encoder(x)
    quantized, _ = model.quantizer(zₑ)
    quantized = zₑ + stopgrad(quantized - zₑ) # gradient bypass
    x̂ = model.decoder(quantized)  
    return x̂
end


function loss_gradient(model::VQVAE, x::AbstractArray{T}; β::T=T(1)) where T<:Real
    zₑ =  model.encoder(x)
    quantized, _ = model.quantizer(zₑ)
    
    # Embedding loss (e -> zₑ) + Commitment loss (zₑ -> e)
    e_latent_loss = Flux.Losses.mse(stopgrad(quantized), zₑ) # gradient propagation to encoder
    q_latent_loss = Flux.Losses.mse(quantized, stopgrad(zₑ)) # gradient propagation to embedding

    𝓛ₗₐₜₑₙₜ = q_latent_loss + β * e_latent_loss

    # Bypass of gradients from decoder to encoder 
    quantized = zₑ + stopgrad(quantized - zₑ)
    x̂ = model.decoder(quantized)  
    
    𝓛 = Flux.Losses.mse(x, x̂) + 𝓛ₗₐₜₑₙₜ # in paper was mse(x, x̂) / data_variance
end

function loss_ema(model::VQVAE, x::AbstractArray{T}; β::T=T(1), γ::T=T(0.99), ϵ::T=T(1e-5), trainmode::Bool=true) where T<:Real
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
    x̂ = model.decoder(quantized)  
    
    𝓛 = Flux.Losses.mse(x, x̂) + 𝓛ₗₐₜₑₙₜ 
end