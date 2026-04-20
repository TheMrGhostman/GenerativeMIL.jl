"""
    AbstractPriorDistribution

Abstract base type for prior probability distributions.
"""
abstract type AbstractPriorDistribution end


"""
    MixtureOfGaussians{T, A1, A3}

Mixture of Gaussians prior: p(x| α, μ, Σ) = ∑ αₖ ⋅ N(x| μₖ, Σₖ).

`A1` is the concrete vector type for logits, `A3` is the concrete 3D tensor type
for component means and variances.

# Fields
- `α::A1`: mixture weights (logits), shape `(K,)`.
- `μ::A3`: component means, shape `(Ds, K, 1)`.
- `Σ::A3`: component log-stds, shape `(Ds, K, 1)` (diagonal).
- `trainable::Bool`: whether parameters are trainable.
"""
struct MixtureOfGaussians{T <: AbstractFloat, A1 <: AbstractVector{T}, A3 <: AbstractArray{T, 3}} <: AbstractPriorDistribution
    α::A1
    μ::A3
    Σ::A3
    trainable::Bool

    function MixtureOfGaussians(
        α::A1,
        μ::A3,
        Σ::A3,
        trainable::Bool,
    ) where {T <: AbstractFloat, A1 <: AbstractVector{T}, A3 <: AbstractArray{T, 3}}
        Ds, K, D3 = size(μ)
        D3 == 1 || throw(ArgumentError("MixtureOfGaussians: expected size(μ, 3) == 1, got $(D3)."))
        length(α) == K || throw(ArgumentError("MixtureOfGaussians: expected length(α) == K ($K), got $(length(α))."))
        size(Σ) == (Ds, K, 1) || throw(ArgumentError("MixtureOfGaussians: expected size(Σ) == ($Ds, $K, 1), got $(size(Σ))."))
        return new{T, A1, A3}(α, μ, Σ, trainable)
    end
end

Flux.@layer MixtureOfGaussians

Flux.trainable(MoG::MixtureOfGaussians) = MoG.trainable ? (α = MoG.α, μ = MoG.μ, Σ = MoG.Σ) : ()

"""
    (MoG::MixtureOfGaussians)(sample_size, batch_size)

Sample from mixture of Gaussians.
Selects component via gumbel-softmax, then samples from selected Gaussians.

# Arguments
- `sample_size::Int`: number of samples per component.
- `batch_size::Int`: batch size.

# Returns
- `AbstractArray`: samples of shape (Ds, sample_size, batch_size).
"""
function (MoG::MixtureOfGaussians)(sample_size::Int, batch_size::Int)
    Ds, K, _ = size(MoG.μ)
    #αₒₕ = repeat(αₒₕ, 1, sample_size, 1, 1) # (K) -> (1, 1, K, 1) -> (1, ss, K, 1),
    #cat([reshape(αₒₕ, (1, 1, K, 1)) for i=1:sample_size]..., dims=2)
    #αₒₕ = repeat(αₒₕ, 1, 1, 1, batch_size)  # (1, ss, K, 1) -> (1, ss, K, bs)
    # cat([αₒₕ for i =1:batch_size]..., dims=4)
    αₒₕ = gumbel_softmax(MoG.α, hard=true)
    αₒₕ = reshape(αₒₕ, (1, 1, K, 1))
    αₒₕ = repeat(αₒₕ, 1, sample_size, 1, batch_size)

    μ = reshape(MoG.μ, (Ds, 1, K, 1)) # (Ds, K, 1) -> (Ds, 1, K, 1)
    Σ = reshape(MoG.Σ, (Ds, 1, K, 1)) # (Ds, K, 1) -> (Ds, 1, K, 1)

    # (1, ss, K, bs) * (Ds, 1, K, 1) -> (Ds, ss, K, bs) -> (Ds, ss, 1, bs) -> (Ds, ss, bs)
    #μ = Flux.sum( μ .* αₒₕ  , dims=3)[:,:,1,:] 
    #Σ = Flux.sum( Σ .* αₒₕ  , dims=3)[:,:,1,:] 
    μ_mixed = reshape(Flux.sum(μ .* αₒₕ, dims=3), (:, sample_size, batch_size))
    Σ_mixed = reshape(Flux.sum(Σ .* αₒₕ, dims=3), (:, sample_size, batch_size))

    # samples from N(0,1) -> (Ds, ss, bs)
    ϵ = MLUtils.randn_like(μ_mixed)
    # (Ds, ss, bs) + (Ds, ss, bs) * (Ds, ss, bs) -> (Ds, ss, bs)
    z = μ_mixed .+ Flux.softplus.(Σ_mixed) .* ϵ
    return z
end

"""
    sample_sphere(dim, n_points)

Sample n_points uniformly from the unit sphere in dim dimensions.
Used for initializing mixture component means.
"""
function sample_sphere(dim::Int, n_points::Int)
    norm_(x, d=1) = sqrt.(sum(abs2, x, dims=d))
    x = randn(Float32, dim, n_points)
    return x ./ norm_(x, 1)
end

function MixtureOfGaussians(dim::Int, n_mixtures::Int, trainable::Bool=true; downscale=10f0, ϵ=1f-3) #Union{Int, Tuple}
    # initial α ∈ 1^{k} ~ (n_mixtures, bs)
    # random initial μ ∈ R^{d, k} ~ (dim, n_mixtures, bs)
    # random initial Σ ∈ R₊^{d, k} ~ (dim, n_mixtures, bs)  

    # Original dummy random initialization
    #μs = randn(Float32, dim, n_mixtures, 1)
    #Σs = abs.(randn(Float32, dim, n_mixtures, 1))   

    # Spherical random initialization   
    ## much easier is to sample from shpere then constructing Fibonacci sphere lattice
    μs_2d = sample_sphere(dim, n_mixtures)
    ## pick general vairance, so gaussians don't overlap
    pp = Distances.pairwise(Distances.euclidean, μs_2d)
    var_ = pp .+ LinearAlgebra.Diagonal(LinearAlgebra.diag(pp) .+ Inf) |> minimum
    μs = Flux.unsqueeze(μs_2d, 3)
    Σs = ones(Float32, dim, n_mixtures, 1) .* Float32(var_ / downscale + ϵ)
    Σs = log.(exp.(Σs) .- 1f0) # inverse to softplus in forward/sampling function 
    ## alpha is kept uniform at start
    αs = ones(Float32, n_mixtures)
    return MixtureOfGaussians(αs, μs, Σs, trainable)
end

"""
    gumbel_softmax(logits; τ=1.0, hard=false, ϵ=1e-10)

Gumbel-softmax trick for differentiable categorical sampling.
Reference: https://arxiv.org/pdf/1611.01144.pdf

# Arguments
- `logits::AbstractArray{T}`: input logits of shape (n_classes) or (n_classes, batch).
- `τ::T`: temperature (greater than 0, lower is sharper).
- `hard::Bool`: if true, use straight-through estimator for hard one-hot samples.
- `ϵ::Real`: numerical stability constant.

# Returns
- `AbstractArray{T}`: soft (or hard) categorical samples same shape as logits.
"""
function gumbel_softmax(logits::AbstractArray{T}; τ::T=1f0, hard::Bool=false, ϵ=T(1.0e-10)) where T <: AbstractFloat
    g = -log.(-log.(MLUtils.rand_like(logits) .+ ϵ) .+ ϵ)
    y = _softmax((logits .+ g) ./ τ)

    if !hard
        return y
    else
        y_hard = zero(y)
        Zygote.ignore() do
            _, ind = findmax(y, dims=1)
            y_hard[ind] .= one(T)
        end
        # Straight-through estimator: forward returns one-hot, backward uses soft gradient
        return y + Zygote.ignore(y_hard - y)
    end
end

"""
    ConstGaussPrior{T, A3}

Constant Gaussian prior (independent of context).
Used in models where latent distribution is fixed across batch.

`A3` is the concrete 3D tensor type used for both parameters.

# Fields
- `μ::A3`: mean tensor, shape `(dimension, n_slots, 1)`.
- `Σ::A3`: log-std tensor, shape `(dimension, n_slots, 1)`.
"""
struct ConstGaussPrior{T <: AbstractFloat, A3 <: AbstractArray{T, 3}}
    μ::A3
    Σ::A3

    function ConstGaussPrior(μ::A3, Σ::A3) where {T <: AbstractFloat, A3 <: AbstractArray{T, 3}}
        size(μ, 3) == 1 || throw(ArgumentError("ConstGaussPrior: expected size(μ, 3) == 1, got $(size(μ, 3))."))
        size(Σ, 3) == 1 || throw(ArgumentError("ConstGaussPrior: expected size(Σ, 3) == 1, got $(size(Σ, 3))."))
        size(μ) == size(Σ) || throw(ArgumentError("ConstGaussPrior: expected size(μ) == size(Σ), got $(size(μ)) and $(size(Σ))."))
        return new{T, A3}(μ, Σ)
    end
end

#ConstGaussPrior(μ::A3, Σ::A3) where {T <: AbstractFloat, A3 <: AbstractArray{T, 3}} =
#    ConstGaussPrior{T, A3}(μ, Σ)

Flux.@layer ConstGaussPrior

Flux.trainable(cgp::ConstGaussPrior) = (μ = cgp.μ, Σ = cgp.Σ)

"""
    (cgp::ConstGaussPrior)(h)

Return constant prior parameters, ignoring context h.
Outputs are broadcast to match h's batch dimension.

# Arguments
- `h::AbstractArray{Real}`: context tensor (ignored), used for batch/device inference.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: (μ, Σ_softplus) with same batch size as h.
"""
function (cgp::ConstGaussPrior)(::AbstractArray{T}) where T
    return cgp.μ, Flux.softplus.(cgp.Σ)
end

"""
    ConstGaussPrior(n_slots, dimension)

Construct a constant Gaussian prior with learnable parameters.

# Arguments
- `n_slots::Int`: number of latent slots.
- `dimension::Int`: latent dimension.

# Returns
- `ConstGaussPrior`: initialized constant prior.
"""
function ConstGaussPrior(n_slots::Int, dimension::Int)
    μ = randn(Float32, dimension, n_slots, 1)
    Σ = ones(Float32, dimension, n_slots, 1)
    return ConstGaussPrior(μ, Σ)
end