struct PoolAE 
    pre_encoder
    bottleneck
    generator
    decoder
end

Flux.@functor PoolAE

function (m::PoolAE)(x::AbstractArray{T, 2}) where T <: Real
    set_size = size(x, 2) 
    features = m.pre_encoder(x) # (D, N) -> (Hidden_dim, N)
    maxes = maximum(features, dims=2) # (HD, N) -> (HD,1)
    means = Flux.mean(features, dims=2) # (HD, N) -> (HD,1)
    features = vcat(means, maxes) # (2*HD, 1)

    μ, Σ = m.bottleneck(features)
    z = μ + Σ .* randn(Float32, size(μ)...)
    
    kld = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σ.^2) - μ.^2  - Σ.^2, dims=1)) 

    μ₁, Σ₁ = m.generator(z)
    gen_z = μ₁ .+ Σ₁ .* randn(Float32, size(μ₁, 1), set_size)

    x̂ = m.decoder(gen_z)
    return x̂, kld
end

function loss(m::PoolAE, x::AbstractArray{T, 2}, β::Float32=1f0) where T <: Real
    x̂, 𝓛_kld = m(x)
    𝓛_rec = chamfer_distance(x, x̂)
    return 𝓛_rec + β .* 𝓛_kld, 𝓛_kld
end

function batched_loss(m::PoolAE, x, beta=1f0)
    bs = length(x)
    𝓛 = 0
    kld = 0
    for i in x
        l, k = loss(m, i, beta)
        𝓛 += l
        kld += k
    end
    return 𝓛/bs, kld/bs
end

function PoolAE(in_dim, hidden, latent_dim, pre_pool_hidden, gen_sigma="scalar", activation::Function=swish)
    if gen_sigma == "scalar"
        gen_out_dim = 1
    elseif gen_sigma == "diag"
        gen_out_dim = pre_pool_hidden
    else
        error("Unkown type of vairance")
    end

    pre_enc = Flux.Chain(
        Flux.Dense(in_dim, hidden, activation), 
        Flux.Dense(hidden, hidden, activation),
        Flux.Dense(hidden, pre_pool_hidden, activation)
    )
    bottle = Flux.Chain(
        Flux.Dense(2*pre_pool_hidden, hidden, activation),
        Flux.Dense(hidden, hidden, activation),
        SplitLayer(hidden, (latent_dim, latent_dim), (identity, Flux.softplus))
    )
    gen = Flux.Chain(
        Flux.Dense(latent_dim, hidden, activation),
        Flux.Dense(hidden, hidden, activation),
        SplitLayer(hidden, (pre_pool_hidden, gen_out_dim), (identity, Flux.softplus))
    )
    dec = Flux.Chain(
        Flux.Dense(pre_pool_hidden, hidden, activation),
        Flux.Dense(hidden, hidden, activation),
        Flux.Dense(hidden, in_dim)
    )
    return PoolAE(pre_enc, bottle, gen, dec)
end