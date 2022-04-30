struct SplitLayer
    μ::Flux.Dense
    σ::Flux.Dense
end

Flux.@functor SplitLayer

function SplitLayer(in::Int, out::NTuple{2, Int}, acts::NTuple{2, Function})
	SplitLayer(
		Flux.Dense(in, out[1], acts[1]),
		Flux.Dense(in, out[2], acts[2])
	)
end

function (m::SplitLayer)(x)
	return (m.μ(x), m.σ(x))
end


struct MaskedDense
    dense::Flux.Dense
end

Flux.@functor MaskedDense

function MaskedDense(in, out, σ=identity; bias=true)
    m = Flux.Dense(in, out, σ, bias=bias)
    return MaskedDense(m)
end

function (m::MaskedDense)(x::AbstractArray{<:Real}, mask::Nothing=nothing)
    return m.dense(x)
end

function (m::MaskedDense)(x::AbstractArray{<:Real}, mask::AbstractArray{<:Real}) 
    # masking of input as well as of output
    return m.dense(x .* mask) .* mask
end





struct VariationalAutoencoder
    encoder::Flux.Chain
    decoder::Flux.Chain
end

Flux.@functor VariationalAutoencoder

kl_divergence(μ, Σ) = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σ.^2) - μ.^2  - Σ.^2, dims=1)) 

function (vae::VariationalAutoencoder)(x::AbstractArray{T}) where T <: Real
    μ, Σ = vae.encoder(x)
    z = μ + Σ * randn(Float32)
    x̂ = vae.decoder(z)
    return x̂
end

function loss(vae::VariationalAutoencoder, x::AbstractArray{T}, β::Float32=1f0) where T <: Real
    μ, Σ = vae.encoder(x)
    z = μ + Σ * randn(Float32)
    x̂ = vae.decoder(z)

    𝓛_rec = mse(x, x̂) #Flux.Losses.mse(x, x̂)
    𝓛_kld = kl_divergence(μ, Σ) 
    return 𝓛_rec + β * 𝓛_kld
end

function VariationalAutoencoder(in_dim::Int, z_dim::Int, out_dim::Int; hidden::Int=32, depth::Int=1, activation::Function=identity)
    encoder = []
    decoder = []
    if depth>=2
        push!(encoder, Flux.Dense(in_dim, hidden, activation))
        push!(decoder, Flux.Dense(z_dim, hidden, activation))
        for i=1:depth-2
            push!(encoder, Flux.Dense(hidden, hidden, activation))
            push!(decoder, Flux.Dense(hidden, hidden, activation))
        end
        push!(encoder, SplitLayer(hidden, (z_dim, z_dim), (identity, softplus)))
        push!(decoder, Flux.Dense(hidden, out_dim))
    elseif depth==1
        push!(encoder, SplitLayer(in_dim, (z_dim, z_dim), (identity, softplus)))
        push!(decoder, Flux.Dense(in_dim, out_dim))
    else
        @error("Incorrect depth of VariationalAutoencoder")
    end
    encoder = Flux.Chain(encoder...)
    decoder = Flux.Chain(decoder...)
    return VariationalAutoencoder(encoder, decoder)
end

"""

function check(x)
    print("size -> $(size(x)) | mean -> $(Flux.mean(x)) | var -> $(Flux.var(x)) | sum -> $(Flux.sum(x)) | not zero elements -> $(sum(x .!= 0))) ")
end

"""