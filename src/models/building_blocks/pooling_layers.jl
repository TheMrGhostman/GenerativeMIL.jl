abstract type AbstractPooling end

struct AttentionPooling <: AbstractPooling 
    # paper: Attention-based Deep Multiple Instance Learning
    # https://arxiv.org/abs/1802.04712
    ff::Union{Flux.Chain, Flux.Dense}
end

Flux.@functor AttentionPooling

function (m::AbstractPooling)(x::AbstractArray{<:Real, 3}; squeeze::Bool=false) # TODO add masked version
    # equivalent to single head attention with trainable query
    # x ~ (d, n, BS)
    # for classification purposes -> ff output_dim (d_ff) == 1 for pooling
    scores = Flux.softmax(m.ff(x), dims=2) # scores ~ (d_ff, n, BS)
    scores = permutedims(scores, (2,1,3)) # (1, n, BS)^T -> (n, 1, BS)
    h = Flux.batched_mul(x, scores) # (d, n, BS) *(n, 1, BS) -> (d, 1, BS)
    # decide whether to drop redundant dimension or not (typicaly true for classification)
    # if true (d, 1, BS) -> (d, BS)
    h = (size(h, 2) == 1 && squeeze) ? dropdims(h, dims=2) : h 
end


struct PMA <: AbstractPooling 
    # known as Pooling by Multihead Attention 
    # paper: Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks
    # https://arxiv.org/pdf/1810.00825
    layer::InducedSetAttentionHalfBlock # ISAHB is generalized PMA
end

Flux.@functor PMA

function (m::PMA)(x::AbstractArray{<:Real, 3}, x_mask::Union{AbstractArray{Bool}, Nothing}=nothing; squeeze::Bool=false) 
    d, n, bs = size(x)
    _, h = m.layer(x, x_mask)
    h = (size(h, 2) == 1 && squeeze) ? dropdims(h, dims=2) : h 
end

function PMA(m::Int, hidden_dim::Int, heads::Int)
    PMA(InducedSetAttentionHalfBlock(m, hidden_dim, heads))
end

# placeholder structure for pooling encoder 
struct PoolEncoder
    prepool
    pooling::Union{AbstractPooling, Function}
    postpool
end

Flux.@functor PoolEncoder

AbstractTrees.children(m::PoolEncoder) = (("Pre-Pool", m.prepool), ("Pooling", m.pooling), ("Post-Pool", m.postpool))
AbstractTrees.children((name, m)::Tuple{String, PoolEncoder}) = (("Pre-Pool", m.prepool), ("Pooling", m.pooling), ("Post-Pool", m.postpool))

AbstractTrees.printnode(io::IO, m::PoolEncoder) = print(io, "PoolEncoder")
AbstractTrees.printnode(io::IO, (name, m)::Tuple{String, PoolEncoder}) = print(io, "$(name) -- PoolEncoder")

function (m::PoolEncoder)(x::AbstractArray{<:Real})
    h = m.prepool(x)
    h = m.pooling(h)
    h = m.postpool(h)
end

function (m::PoolEncoder)(x::AbstractArray{<:Real}, x_mask::Union{AbstractArray{Bool}, Nothing})
    h = mask(m.prepool(x), x_mask)
    h = m.pooling(h, mask) # TODO fix
    h = m.postpool(h)
end