stopgrad(x) = x
Zygote.@nograd stopgrad

abstract type Quantizer end

struct VectorQuantizer <: Quantizer 
    embedding
end


struct VectorQuantizerEMA <: Quantizer 
    embedding
    nₑₘₐ
    mₑₘₐ
end

struct VQVAE
    encoder
    latent_embedding
    decoder
    nₑₘₐ::Union{AbstractArray{<:Real, 2}, Nothing} # not needed for gradient version
    mₑₘₐ::Union{AbstractArray{<:Real, 2}, Nothing} # not needed for gradient version
end

Flux.@functor VQVAE

Flux.trainable(model::VQVAE) = (model.encoder, model.latent_embedding, model.decoder)

function vector_quantization(model::VQVAE, z::AbstractArray{<:Real})
    # flatten input zₑ #d, n, bs = size(zₑ)
    size_ = size(z)
    d, n_elements = size_[1], prod(size_[2:end]) # now it works for nD array

    zₑ = reshape(z, d, n_elements)
    # Compute distance (e - x)² = e² - 2ex + x²
    dist = (sum(zₑ .^2, dims=1) 
            .+ sum(model.latent_embedding .^2, dims=1)' 
            .- (2 .* model.latent_embedding' * zₑ ))

    # Encoding
    encoding_indices = argmin(dist, dims=1)
    encodings = zeros_like(zₑ, (size(model.latent_embedding, 2), n_elements)) # there has to be transposition or dropdims
    Zygote.ignore(()->(encodings[encoding_indices] .+= 1)) # need to exclude gradient or i will crash

    # Quantize and unflatten
    quantized = model.latent_embedding * encodings # (d, e) * (e, bs) -> (d, bs)
    quantized = reshape(quantized, size_)
    return quantized, encodings
end

function (model::VQVAE)(x::AbstractArray{T}) where T<:Real
    zₑ =  model.encoder(x)
    quantized, _ = vector_quantization(model, zₑ)
    quantized = zₑ + stopgrad(quantized - zₑ) # gradient bypass
    x̂ = model.decoder(quantized)  
    return x̂
end


function loss_gradient(model::VQVAE, x::AbstractArray{T}; β::T=T(1)) where T<:Real
    zₑ =  model.encoder(x)
    quantized, _ = vector_quantization(model, zₑ)
    
    # Embedding loss (e -> zₑ) + Commitment loss (zₑ -> e)
    e_latent_loss = Flux.Losses.mse(stopgrad(quantized), zₑ) # gradient propagation to encoder
    q_latent_loss = Flux.Losses.mse(quantized, stopgrad(zₑ)) # gradient propagation to embedding

    𝓛ₗₐₜₑₙₜ = q_latent_loss + β * e_latent_loss

    # Bypass of gradients from decoder to encoder 
    quantized = zₑ + stopgrad(quantized - zₑ)

    x̂ = model.decoder(quantized)  
    
    𝓛 = Flux.Losses.mse(x, x̂) + 𝓛ₗₐₜₑₙₜ # in paper was mse(x, x̂) / data_variance
end

function loss_ema(model::VQVAE, x::AbstractArray{T}; γ::T=T(0.99), ϵ::T=T(1e-5), trainmode::Bool=true) where T<:Real
    zₑ =  model.encoder(x)
    quantized, encodings = vector_quantization(model, zₑ)
    # Commitment loss (zₑ -> embedding)
    e_latent_loss = Flux.Losses.mse(stopgrad(quantized), zₑ) # gradient propagation to encoder
    𝓛ₗₐₜₑₙₜ = β * e_latent_loss

    if trainmode 
        if isempty(model.ema.N)
            model.ema.N = 0
            model.ema.m = randn_like(model.latent_embedding)
        end
        # Exponential Moving Average update
        ema_cs = ema_cs .* γ .+ (1f0 - γ) .* sum(encodings, dims=2) # (e, 1)
        n = sum(ema_cs) # scalar
        ema_cs = (ema_cs .+ ϵ) ./ (n .+ size(model.latent_embedding, 2) .* ϵ) .* n #(e, 1)
        d = size(zₑ, 1) 
        dw = reshape(zₑ, d, :) * encodings' # (d, n*bs) * (n*bs, e) -> (d, e) ≈ latent_embedding
        ema_w = ema_w .* γ .+ (1f0 - γ) .* dw # (d, e)
        model.latent_embedding .= ema_w ./ ema_cs # (d, e)
    end

    # Bypass of gradients from decoder to encoder 
    quantized = zₑ + stopgrad(quantized - zₑ)

    x̂ = model.decoder(quantized)  
    
    𝓛 = Flux.Losses.mse(x, x̂) + 𝓛ₗₐₜₑₙₜ 
end
