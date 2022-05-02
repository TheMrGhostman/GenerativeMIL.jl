struct PoolAE 
    pre_encoder
    bottleneck
    decoder
end

Flux.@functor PoolAE

function (m::PoolAE)(x::AbstractArray{T, 2}) where T <: Real
    set_size = size(x, 2) 
    features = m.pre_encoder(x) # (D, N) -> (Hidden_dim, N)
    maxes = maximum(features, dims=2) # (HD, N) -> (HD,1)
    means = Flux.mean(features, dims=2) # (HD, N) -> (HD,1)
    features = repeat(vcat(means, maxes), 1, set_size) # (2*HD, 1) -> (2*HD, N)

    μ, Σ = m.bottleneck(features)
    z = μ + Σ .* randn(Float32, size(μ)...)

    kld = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σ.^2) - μ.^2  - Σ.^2, dims=1)) 

    x̂ = m.decoder(z)
    return x̂, kld
end

function loss(m::PoolAE, x::AbstractArray{T, 2}, β::Float32=1f0) where T <: Real
    x̂, 𝓛_kld = m(x)
    𝓛_rec = chamfer_distance(x, x̂)
    return 𝓛_rec + β .* 𝓛_kld
end