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
Flux.trainable(q::Quantizer) = (q.embedding,)

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
    Zygote.ignore(()->(encodings[encoding_indices] .+= 1)) # need to exclude gradient or it will crash

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
#Flux.trainable(model::VQVAE) = (model.encoder, model.quantizer, model.decoder)

function (model::VQVAE)(x::AbstractArray{T}) where T<:Real
    zₑ =  model.encoder(x)
    quantized, _ = model.quantizer(zₑ)
    quantized = zₑ + stopgrad(quantized - zₑ) # gradient bypass
    x̂ = model.decoder(quantized)  
    return x̂
end


function loss_gradient(model::VQVAE, x::AbstractArray{T}; β=0.25) where T<:Real
    zₑ =  model.encoder(x)
    quantized, _ = model.quantizer(zₑ)
    
    # Embedding loss (e -> zₑ) + Commitment loss (zₑ -> e)
    e_latent_loss = Flux.Losses.mse(stopgrad(quantized), zₑ) # gradient propagation to encoder
    q_latent_loss = Flux.Losses.mse(quantized, stopgrad(zₑ)) # gradient propagation to embedding

    𝓛ₗₐₜₑₙₜ = q_latent_loss + β * e_latent_loss

    # Bypass of gradients from decoder to encoder 
    quantized = zₑ + stopgrad(quantized - zₑ)
    x̂ = model.decoder(quantized)  
    #Zygote.ignore() do
    #    println("rec_loss: $(round(Flux.Losses.mse(x, x̂), digits=4)), q_loss: $(round(q_latent_loss, digits=4)), e_loss: $(round(e_latent_loss, digits=4))")
    #end
    𝓛 = Flux.Losses.mse(x, x̂) + 𝓛ₗₐₜₑₙₜ # in paper was mse(x, x̂) / data_variance
end

function loss_ema(model::VQVAE, x::AbstractArray{T}; β=0.25f0, γ::T=T(0.99), ϵ::T=T(1e-5), trainmode::Bool=true) where T<:Real
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


function vqvae_constructor_from_named_tuple(
    ;idim=3, hdim=64, depth=3, zdim=3, n_embed=128, ema::Bool=false, 
    activation="swish", init_seed=nothing, kwargs...)
    
    activation = eval(:($(Symbol(activation))))
    (init_seed !== nothing) ? Random.seed!(init_seed) : nothing

    enc = []
    dec = []
    if depth==1
        enc = [Dense(idim, zdim)]
        dec = [Dense(zdim, idim)]
    elseif depth > 1
        push!(enc, Dense(idim, hdim, activation))
        push!(dec, Dense(zdim, hdim, activation))
        for i=1:depth-2
            push!(enc, Dense(hdim, hdim, activation))
            push!(dec, Dense(hdim, hdim, activation))
        end
        push!(enc, Dense(hdim, zdim))
        push!(dec, Dense(hdim, idim))
    end

    if ema
        emb = randn(Float32, zdim, n_embed) 
        quan = VectorQuantizerEMA(emb, zeros(Float32, n_embed, 1), rand_like(emb))
    else
        emb = randn(Float32, zdim, n_embed) 
        #emb = Float32.(rand(Uniform(-1/n_embed, 1/n_embed), zdim, n_embed))
        quan = VectorQuantizer(emb)
    end
    model = VQVAE(
        Chain(enc...),
        quan,
        Chain(dec...))

    (init_seed !== nothing) ? Random.seed!() : nothing
    return model
end

function Base.show(io::IO, m::VectorQuantizer)
    print(io, "VectorQuantizer")
    print(io, "\n- embedding = $(m.embedding |> size) | $(m.embedding  |> typeof) | mean ~ $(m.embedding  |> Flux.mean)")
end

function Base.show(io::IO, m::VectorQuantizerEMA)
    print(io, "VectorQuantizerEMA")
    print(io, "\n- embedding = $(m.embedding |> size) | $(m.embedding  |> typeof) | mean ~ $(m.embedding  |> Flux.mean)")
    print(io, "\n- nᵗᵢ = $(m.n |> size) | $(m.n  |> typeof) | mean ~ $(m.n  |> Flux.mean)")
    print(io, "\n- mᵗᵢ = $(m.m |> size) | $(m.m  |> typeof) | mean ~ $(m.m  |> Flux.mean)")
end