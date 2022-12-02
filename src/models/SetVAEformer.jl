abstract type AbstractTransformerEncoder end #<:PoolEncoder
abstract type AbstractTransformerDecoder end 

struct PercieverIO <: AbstractTransformerEncoder
    # this struct follow general idea behind PercieverIO
    Prepool::Union{Flux.Dense, Flux.Chain}
    Pooling::AttentionPooling 
    MABs # Chain{MultiheadAttentionBlock}
end

Flux.@functor PercieverIO

function (m::PercieverIO)(x::AbstractArray{<:Real, 3}, x_mask::Union{AbstractArray{<:Real}, Nothing}=nothing)
    # x ∈ ℝ^{n,d} ~ (d, n, bs) 
    # x_mask ∈ ℝ^{n} ~ (1, n, bs) 
    x = mask(m.Prepool(x), x_mask) # (d, n, bs) -> (d₁, n, bs)
    # FIXME not working yet. think about mask(m.pooling, x, x_mask)
    x = m.Pooling(x, x_mask) # (d₁, n, bs) -> (d₁, n, bs)*(n, m, bs) ->  (d₁, m, bs) , m << n
    # mask is not needed from here
    x = m.MABs(x)
end


struct CrossAttentionDecoder <: AbstractTransformerDecoder
    PreMAB::Union{Flux.Dense, Flux.Chain}
    MABs# Vector{MultiheadAttentionBlock}
    PostMAB::Union{Flux.Dense, Flux.Chain}
end

Flux.@functor CrossAttentionDecoder

function (m::CrossAttentionDecoder)(x::AbstractArray{T, 3}, z::AbstractArray{T, 3}) where T <:Real
    x = m.PreMAB(x)
    for mab in m.MABs
        x = mab(x, z)
    end
    return m.PostMAB(x)
end

struct SetVAEformer
    Encoder::AbstractTransformerEncoder
    Vae
    Decoder::AbstractTransformerDecoder
    Prior::AbstractPriorDistribution
end


