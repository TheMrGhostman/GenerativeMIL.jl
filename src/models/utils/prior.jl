abstract type AbstractPriorDistribution end

struct MixtureOfGaussians{T <: Real} <: AbstractPriorDistribution
    # p(x| α, μ, Σ) = ∑ αₖ ⋅ p(x| μₖ, Σₖ)
    K::Int # n_mixtures
    Ds::Int # dimension of space
    α::AbstractArray{T}
    μ::AbstractArray{T}
    Σ::AbstractArray{T} # diagonal 
    trainable::Bool
end

function Base.show(io::IO, m::MixtureOfGaussians)
    print(io, "MixtureOfGaussians(")
    print(io, "\n\t - K = $(m.K) \n\t - Ds = $(m.Ds) \n\t - α = $(m.α)")
    print(io, "\n\t - μ = $(m.μ) \n\t - Σ = $(m.Σ) \n\t - trainable = $(m.trainable) \n\t ) ")
end

Flux.@functor MixtureOfGaussians

Flux.trainable(MoG::MixtureOfGaussians) = MoG.trainable ? (MoG.α, MoG.μ, MoG.Σ) : ()

#Flux.@functor MixtureOfGaussians # all parameters α, μ and Σ are now trainable

function (MoG::MixtureOfGaussians)(sample_size::Int, batch_size)
    # sample_size = ...
    αₒₕ = gumbel_softmax(MoG.α, hard=true)
    αₒₕ = reshape(αₒₕ, (1, 1, MoG.K, 1))
    αₒₕ = repeat(αₒₕ, 1, sample_size, 1, 1) # (K) -> (1, 1, K, 1) -> (1, ss, K, 1),
    #cat([reshape(αₒₕ, (1, 1, MoG.K, 1)) for i=1:sample_size]..., dims=2)
    αₒₕ = repeat(αₒₕ, 1, 1, 1, batch_size)  # (1, ss, K, 1) -> (1, ss, K, bs)
    # cat([αₒₕ for i =1:batch_size]..., dims=4)

    μ = reshape(MoG.μ, (MoG.Ds, 1, MoG.K, 1)) # (Ds, K, 1) -> (Ds, 1, K, 1) 
    Σ = reshape(MoG.Σ, (MoG.Ds, 1, MoG.K, 1)) # (Ds, K, 1) -> (Ds, 1, K, 1) 

    # (1, ss, K, bs) * (Ds, 1, K, 1) -> (Ds, ss, K, bs) -> (Ds, ss, 1, bs) -> (Ds, ss, bs)
    #μ = Flux.sum( μ .* αₒₕ  , dims=3)[:,:,1,:] 
    #Σ = Flux.sum( Σ .* αₒₕ  , dims=3)[:,:,1,:] 
    μ = reshape(Flux.sum( μ .* αₒₕ  , dims=3), (:,sample_size, batch_size))
    Σ = reshape(Flux.sum( Σ .* αₒₕ  , dims=3), (:,sample_size, batch_size))

    # samples from N(0,1) -> (Ds, ss, bs)
    ϵ = randn(Float32, MoG.Ds, sample_size, batch_size)
    z = μ + Flux.softplus.(Σ) .* ϵ # (Ds, ss, bs) + (Ds, ss, bs) * (Ds, ss, bs) -> (Ds, ss, bs)
    return z
end

function MixtureOfGaussians(dim::Int, n_mixtures::Int, trainable::Bool=true) #Union{Int, Tuple}
    # initial α ∈ 1^{k} ~ (n_mixtures, bs)
    # random initial μ ∈ R^{d, k} ~ (dim, n_mixtures, bs)
    # random initial Σ ∈ R₊^{d, k} ~ (dim, n_mixtures, bs)  
    μs = randn(Float32, dim, n_mixtures, 1)
    Σs = abs.(randn(Float32, dim, n_mixtures, 1))       
    αs = ones(Float32, n_mixtures)
    return MixtureOfGaussians(n_mixtures, dim, αs, μs, Σs, trainable)
end

function gumbel_softmax(logits::AbstractArray{T}; τ::T=1f0, hard::Bool=false, eps::Float32=1.0f-10) where T <: Real
    # logits ∈ R^{n_classes} ~ (n_classes, bs) 
    # τ ... non-negative scalar temeperature (default=1.0) https://arxiv.org/pdf/1611.01144.pdf
    # gumbel_samples = -log.(-log.(rand(Float32, size(logits)) + 1e-10) + 1e-10) # alternative version
    #.+ rand(Gumbel(Float32(0), Float32(1)), size(logits))
    gumbel_samples = -log.(-log.(Random.rand!(logits) .+ eps) .+ eps)
    y = logits .+ gumbel_samples
    y = Flux.softmax(y./τ)

    if !hard
        return y
    else
        y_hard = nothing
        Zygote.ignore() do
            # we don't want for this block of code computing gradients
            shape = size(y)
            y_hard = zeros(T, shape)
            _, ind = findmax(y, dims=1)
            y_hard[ind] .= 1
            y_hard = y_hard .- y
        end
        #print(y_hard)
        # now we bypass gradients from y_hard to y
        y = y_hard .+ y 
        return y
    end
end
