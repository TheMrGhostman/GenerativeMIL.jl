abstract type AbstractPooling end

struct AttentionPooling <: AbstractPooling 
    # paper: Attention-based Deep Multiple Instance Learning
    # https://arxiv.org/abs/1802.04712
    ff::Union{Flux.Chain, Flux.Dense}
end

Flux.@functor AttentionPooling

function (m::AbstractPooling)(x::AbstractArray{<:Real, 3})
    # equivalent to single head attention with trainable query
    # x ~ (d, n, BS)
    # for classification purposes -> ff output_dim (d_ff) == 1 for pooling
    scores = permutedims(scores, (2,1,3)) 
    scores = Flux.softmax(m.ff(x), dims=2) # scores ~ (d_ff, n, BS)
    h = Flux.batched_mul(x, scores) # (d, n, BS) *(n, 1, BS) -> (d, 1, BS)
    # decide whether to drop redundant dimension or not (typicaly true for classification)
    # if true (d, 1, BS) -> (d, BS)
    h = (size(h, 2) == 1) ? dropdims(h, dims=2) : h 
end


struct PMA <: AbstractPooling 
    # known as Pooling by Multihead Attention 
    # paper: Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks
    # https://arxiv.org/pdf/1810.00825
    layer::InducedSetAttentionHalfBlock # ISAHB is generalized PMA
end

Flux.@functor PMA

function (m::PMA)(x::AbstractArray{<:Real, 3})
    d, n, bs = size(x)
    const_module = (typeof(x) <: CUDA.CuArray) ? CUDA : Base 
    _, h = m.layer(x, nothing, const_module=const_module)
    h = (size(h, 2) == 1) ? dropdims(h, dims=2) : h 
end

function PMA(m::Int, hidden_dim::Int, heads::Int)
    PMA(InducedSetAttentionHalfBlock(m, hidden_dim, heads))
end

# placeholder structure for pooling encoder 
struct PoolEncoder
    prepool
    pooling
    postpool
end

Flux.@functor PoolEncoder

function (m::PoolEncoder)(x::AbstractArray{<:Real})
    h = m.prepool(x)
    h = m.pooling(h)
    h = m.postpool(h)
end
