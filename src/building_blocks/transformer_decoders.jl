struct CrossAttentionDecoder{CA<:Vector{<:MultiheadAttentionBlock}}
    cross_attns::CA
end

Flux.@layer CrossAttentionDecoder

function (m::CrossAttentionDecoder)(x::AbstractArray{T}, z::AbstractArray{T}) where T <: AbstractFloat
    for ca in m.cross_attns
        x = ca(x, z)
    end
    return x
end

struct TransformerDecoder{SA<:Vector{<:MultiheadAttentionBlock}, CA<:Vector{<:MultiheadAttentionBlock}}
    self_attns::SA
    cross_attns::CA
end

Flux.@layer TransformerDecoder

function (m::TransformerDecoder)(x::AbstractArray{T}, z::AbstractArray{T}) where T <: AbstractFloat
    for (sa, ca) in zip(m.self_attns, m.cross_attns)
        x = sa(x)
        x = ca(x, z)
    end
    return x
end

# call (Q, V, Q_mask, V_mask), so no new attention-masking logic is needed.
function (m::TransformerDecoder)(x::AbstractArray{T}, z::AbstractArray{T}, x_mask::AbstractArray{Bool}) where T <: AbstractFloat
    for (sa, ca) in zip(m.self_attns, m.cross_attns)
        x = sa(x, x, x_mask, x_mask)   # masked self-attention among query/object slots
        x = ca(x, z, x_mask, nothing)  # masked cross-attention; z (latent) has no padding
    end
    return x
end

