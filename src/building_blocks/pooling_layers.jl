abstract type AbstractPooling end

struct AttentionPooling <: AbstractPooling 
    # paper: Attention-based Deep Multiple Instance Learning
    # https://arxiv.org/abs/1802.04712
    ff::Union{Flux.Chain, Flux.Dense}
end

Flux.@layer AttentionPooling

function (m::AbstractPooling)(x::AbstractArray{T, 3}; squeeze::Bool=false) where T <: AbstractFloat# TODO add masked version
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

Flux.@layer PMA

function (m::PMA)(x::AbstractArray{T, 3}, x_mask::Mask=nothing; squeeze::Bool=false) where T <: AbstractFloat
    d, n, bs = size(x)
    _, h = m.layer(x, x_mask)
    h = (size(h, 2) == 1 && squeeze) ? dropdims(h, dims=2) : h 
end

function PMA(m::Int, hidden_dim::Int, heads::Int)
    PMA(InducedSetAttentionHalfBlock(m, hidden_dim, heads))
end

masked_mean(x, mask; dims=2) = sum(x, dims=dims) ./ sum(mask, dims=2) # TODO find if i did not figure this out anywhere
masked_maximum(x, mask; dims=2) = maximum(x .* mask, dims=dims) #FIXME set masked values to -Inf

mean_max_pooling_(x) = cat(Flux.mean(x, dims=2), maximum(x, dims=2), dims=1)
mean_pooling_(x) = Flux.mean(x, dims=2)
max_pooling_(x) = maximum(x, dims=2)


function _make_pooling(poolf::String, feature_dim::Int, activation::Function; pool_hidden_dim::Int=feature_dim, pma_heads::Int=4)
    if poolf == "mean-max"
        return mean_max_pooling_, 2
    elseif poolf == "mean"
        return mean_pooling_, 1
    elseif poolf == "max"
        return max_pooling_, 1
    elseif poolf == "attention"
        score_net = Flux.Chain(
            Dense(feature_dim, pool_hidden_dim, activation),
            Dense(pool_hidden_dim, 1)
        )
        return AttentionPooling(score_net), 1
    elseif poolf == "PMA"
        return PMA(1, feature_dim, pma_heads), 1
    else
        error("Unknown pooling function: $poolf. Use mean, max, mean-max, attention, or PMA.")
    end
end


# placeholder structure for pooling encoder 
struct PoolEncoder
    prepool::Union{Chain, Dense}
    pooling::Union{AbstractPooling, Function}
    postpool::Union{Chain, Dense}
end

Flux.@layer PoolEncoder

AbstractTrees.children(m::PoolEncoder) = (("Pre-Pool", m.prepool), ("Pooling", m.pooling), ("Post-Pool", m.postpool))
AbstractTrees.children((name, m)::Tuple{String, PoolEncoder}) = (("Pre-Pool", m.prepool), ("Pooling", m.pooling), ("Post-Pool", m.postpool))

AbstractTrees.printnode(io::IO, m::PoolEncoder) = print(io, "PoolEncoder")
AbstractTrees.printnode(io::IO, (name, m)::Tuple{String, PoolEncoder}) = print(io, "$(name) -- PoolEncoder")

function (m::PoolEncoder)(x::AbstractArray{T}) where T <: AbstractFloat
    h = m.prepool(x)
    h = m.pooling(h)
    h = m.postpool(h)
end

function (m::PoolEncoder)(x::AbstractArray{T}, x_mask::Mask) where T <: AbstractFloat
    h = multiplicative_masking(m.prepool(x), x_mask)
    h = m.pooling(h, x_mask)
    h = m.postpool(h)
end


