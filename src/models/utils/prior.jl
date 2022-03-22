# code for induction sets
using Distributions
using Flux
using Random: rand!
using Zygote

struct MixtureOfGaussians{T <: Real}
    # p(x| α, μ, Σ) = ∑ αₖ ⋅ p(x| μₖ, Σₖ)
    α::AbstractArray{T}
    μ::AbstractArray{T}
    Σ::AbstractArray{T} # diagonal 
    trainable::Bool
end

Flux.trainable(MoG::MixtureOfGaussians) = MoG.trainable ? (MoG.α, MoG.μ, MoG.Σ) : ()

#Flux.@functor MixtureOfGaussians # all parameters α, μ and Σ are now trainable

function (MoG::MixtureOfGaussians)(sample_size::Int)
    nothing
end

function MixtureOfGaussians(dims::Union{Int, tuple}, n_mixtures::Int, trainable::Bool=true)
    nothing
end

function gumbel_softmax(logits::AbstractArray{T}, τ::T=1f0, hard::Bool=false, eps::Float32=1.0f-10) where T <: Real
    # logits ∈ R^{n_classes} ~ (n_classes, bs) 
    # τ ... non-negative scalar temeperature (default=1.0) https://arxiv.org/pdf/1611.01144.pdf
    # gumbel_samples = -log.(-log.(rand(Float32, size(logits)) + 1e-10) + 1e-10) # alternative version
    #.+ rand(Gumbel(Float32(0), Float32(1)), size(logits))
    gumbel_samples = -log.(-log.(rand!(logits) .+ eps) .+ eps)
    y = logits .+ gumbel_samples
    y = Flux.softmax(y./τ)

    if !hard
        return y
    else
        Zygote.ignore() do
            # we don't want for this block of code computing gradients
            shape = size(y)
            y_hard = zeros(T, shape)
            _, ind = findmax(y, dims=1)
            y_hard[ind] .= 1
            y_hard = y_hard .- y
        end
        # now we bypass gradients from y_hard to y
        y = y_hard .+ y 
        return y
    end
end
