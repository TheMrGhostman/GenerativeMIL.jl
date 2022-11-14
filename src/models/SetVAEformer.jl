abstract type AbstractTransformerEncoder end #<:PoolEncoder

struct PercieverIO <: AbstractTransformerEncoder
    # this struct follow general idea behind PercieverIO
    prepool::Union{Flux.Dense, Flux.Chain}
    pooling::AttentionPooling 
    mabs # Flux.Chain{MultiheadAttentionBlock}
end

Flux.@functor PercieverIO

function (m::PercieverIO)(x::AbstractArray{<:Real, 3}, x_mask::Union{AbstractArray{<:Real}, Nothing}=nothing)
    # x ∈ ℝ^{n,d} ~ (d, n, bs) 
    # x_mask ∈ ℝ^{n} ~ (1, n, bs) 
    x = mask(m.prepool(x), x_mask) # (d, n, bs) -> (d₁, n, bs)
    # FIXME not working yet. think about mask(m.pooling, x, x_mask)
    x = m.pooling(x, x_mask) # (d₁, n, bs) -> (d₁, n, bs)*(n, m, bs) ->  (d₁, m, bs) , m << n
    # mask is not needed from here
    x = m.mabs(x)
end

struct SetVAEformer
    encoder::AbstractTransformerEncoder
    vae
    decoder::AbstractTransformerDecoder
    prior::AbstractPriorDistribution
end


