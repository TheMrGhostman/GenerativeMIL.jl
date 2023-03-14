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

function (m::CrossAttentionDecoder)(x::AbstractArray{T, 3}, kv::AbstractArray{T, 3}) where T <:Real
    # x ∈ ℝ^{d,d} ~ (d, n, bs)  ... Random samples from prior / Query
    # kv ∈ ℝ^{m,d} ~ (d, m, bs) ... Key and Value for Cross Attention  
    # operations are O(n ⋅ m ⋅ d) where m << n ... not quadratic with n as SA O(n² ⋅ d)

    x = m.PreMAB(x)
    for mab in m.MABs
        x = mab(x, kv)
    end
    return m.PostMAB(x)
end

struct SetVAEformer
    encoder::AbstractTransformerEncoder
    vae
    decoder::AbstractTransformerDecoder
    prior::AbstractPriorDistribution
end


function (m::SetVAEformer)(x::AbstractArray{<:Real, 3}, x_mask::Union{AbstractArray{<:Real}, Nothing}=nothing)
    h = m.encoder(x, x_mask)
    zₖᵥ = m.vae(h)
    zₚᵣᵢₒᵣ = m.prior(x)
    x̂ = m.decoder(zₚᵣᵢₒᵣ, zₖᵥ)
end

