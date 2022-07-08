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
    print(io, "\n\t - K = $(m.K) \n\t - Ds = $(m.Ds)")
    print(io, "\n\t - α = $(m.α |> size) | $(m.α |> typeof) | mean ~ $(m.α |> Flux.mean)")
    print(io, "\n\t - μ = $(m.μ |> size) | $(m.μ |> typeof) | mean ~ $(m.μ |> Flux.mean)")
    print(io, "\n\t - Σ = $(m.Σ |> size) | $(m.Σ |> typeof) | mean ~ $(m.Σ |> Flux.mean)")
    print(io, "\n\t - trainable = $(m.trainable) \n\t ) ")
end

Flux.@functor MixtureOfGaussians

Flux.trainable(MoG::MixtureOfGaussians) = MoG.trainable ? (MoG.α, MoG.μ, MoG.Σ) : ()

#Flux.@functor MixtureOfGaussians # all parameters α, μ and Σ are now trainable

function (MoG::MixtureOfGaussians)(sample_size::Int, batch_size; const_module::Module=Base)
    # sample_size = ...
    αₒₕ = gumbel_softmax(MoG.α, hard=true)
    αₒₕ = reshape(αₒₕ, (1, 1, MoG.K, 1))
    αₒₕ = const_module.ones(Float32, 1,sample_size,MoG.K,batch_size) .* αₒₕ
    # on gpu is much faster then repeat
    #αₒₕ = repeat(αₒₕ, 1, sample_size, 1, 1) # (K) -> (1, 1, K, 1) -> (1, ss, K, 1),
    #cat([reshape(αₒₕ, (1, 1, MoG.K, 1)) for i=1:sample_size]..., dims=2)
    #αₒₕ = repeat(αₒₕ, 1, 1, 1, batch_size)  # (1, ss, K, 1) -> (1, ss, K, bs)
    # cat([αₒₕ for i =1:batch_size]..., dims=4)

    μ = reshape(MoG.μ, (MoG.Ds, 1, MoG.K, 1)) # (Ds, K, 1) -> (Ds, 1, K, 1) 
    Σ = reshape(MoG.Σ, (MoG.Ds, 1, MoG.K, 1)) # (Ds, K, 1) -> (Ds, 1, K, 1) 

    # (1, ss, K, bs) * (Ds, 1, K, 1) -> (Ds, ss, K, bs) -> (Ds, ss, 1, bs) -> (Ds, ss, bs)
    #μ = Flux.sum( μ .* αₒₕ  , dims=3)[:,:,1,:] 
    #Σ = Flux.sum( Σ .* αₒₕ  , dims=3)[:,:,1,:] 
    μ = reshape(Flux.sum( μ .* αₒₕ  , dims=3), (:,sample_size, batch_size))
    Σ = reshape(Flux.sum( Σ .* αₒₕ  , dims=3), (:,sample_size, batch_size))

    # samples from N(0,1) -> (Ds, ss, bs)
    # tyoeof(μ)(x) works only if has the same size/shape as μ !!!!!
    ϵ = const_module.randn(Float32, MoG.Ds, sample_size, batch_size)
    z = μ + Flux.softplus.(Σ) .* ϵ # (Ds, ss, bs) + (Ds, ss, bs) * (Ds, ss, bs) -> (Ds, ss, bs)
    return z
end

function sample_sphere(dim, n_points)
    norm_(x,d=1) = sqrt.(sum(abs2, x, dims=d))
    x = randn(Float32, dim, n_points)
    return x ./ norm_(x, 1)
end

function MixtureOfGaussians(dim::Int, n_mixtures::Int, trainable::Bool=true; downscale=10f0, eps=1f-3) #Union{Int, Tuple}
    # initial α ∈ 1^{k} ~ (n_mixtures, bs)
    # random initial μ ∈ R^{d, k} ~ (dim, n_mixtures, bs)
    # random initial Σ ∈ R₊^{d, k} ~ (dim, n_mixtures, bs)  

    # Original dummy random initialization
    #μs = randn(Float32, dim, n_mixtures, 1)
    #Σs = abs.(randn(Float32, dim, n_mixtures, 1))   

    # Spherical random initialization   
    ## much easier is to sample from shpere then constructing Fibonacci sphere lattice
    μs = sample_sphere(dim, n_mixtures) |> Flux.unsqueeze(3) 
    ## pick general vairance, so gaussians don't overlap
    pp = Distances.pairwise(Distances.euclidean, μs)
    var_ = pp + LinearAlgebra.Diagonal(LinearAlgebra.diag(pp) .+ Inf) |> minimum
    Σs = ones(Float32, dim, n_mixtures, 1) .* Float32(var_ / downscale + eps)
    Σs = log.(exp.(Σs) - 1) # inverse to softplus in forward/sampling function 
    ## alpha is kept uniform at start
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
            y_hard = typeof(y)(zeros(T, shape)) # !!!!! this will break if y_hard would be diferent size then y
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

struct ConstGaussPrior
    μ::AbstractArray
    Σ::AbstractArray
end

Flux.@functor ConstGaussPrior

Flux.trainable(cgp::ConstGaussPrior) = (cgp.μ, cgp.Σ)

function (cgp::ConstGaussPrior)(h::AbstractArray{<:Real, 3})
    # computing prior μ, Σ from h
    const_module = (typeof(h) == CuArray{Float32, 3, CUDA.Mem.DeviceBuffer}) ? CUDA : Base
    _, sample_size, batch_size = size(h)
    μ = const_module.ones(Float32, 1, 1, batch_size) .* cgp.μ
    Σ = const_module.ones(Float32, 1, 1, batch_size) .* Flux.softplus.(cgp.Σ)
    return μ, Σ
end

function ConstGaussPrior(m::Int, dimension::Int)
    μ = randn(Float32, dimension, m, 1)
    Σ = ones(Float32, dimension, m, 1)
    return ConstGaussPrior(μ, Σ)
end

function Base.show(io::IO, m::ConstGaussPrior)
    print(io, "ConstGaussPrior(")
    print(io, "\n\t - μ = $(m.μ |> size) | $(m.μ |> typeof) | mean ~ $(m.μ |> Flux.mean)")
    print(io, "\n\t - Σ = $(m.Σ |> size) | $(m.Σ |> typeof) | mean ~ $(m.Σ |> Flux.mean) \n\t ) ")
end