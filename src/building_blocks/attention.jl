"""
    MultiheadAttention{F}

Multi-head attention module with learnable projections for queries, keys, values, and output.

# Fields
- `heads::Int`: number of attention heads.
- `WQ`: query projection layer (no bias).
- `WK`: key projection layer (no bias).
- `WV`: value projection layer (no bias).
- `WO`: output projection layer (no bias).
- `attention::F`: attention function (standard or slot attention).
"""
struct MultiheadAttention{F}
    # Dense layers without bias !! or it will break masking
    heads::Int
    WQ::Flux.Dense
    WK::Flux.Dense
    WV::Flux.Dense
    WO::Flux.Dense
    attention::F 
end

Flux.@layer MultiheadAttention

Flux.trainable(mh::MultiheadAttention) = (WQ = mh.WQ, WK = mh.WK, WV = mh.WV, WO = mh.WO)

"""
    MultiheadAttention(input_dim, hidden_dim, heads, attention_fn=attention)

Construct a multi-head attention module with learnable projections.

# Arguments
- `input_dim::Integer`: input feature dimension for Q, K, V.
- `hidden_dim::Integer`: projected dimension, must be divisible by `heads`.
- `heads::Integer`: number of parallel attention heads (must be > 0).
- `attention_fn::Function`: attention kernel function (standard or slot attention).

# Returns
- `MultiheadAttention`: initialized multi-head attention module.

# Throws
- `ArgumentError`: if `heads <= 0` or `hidden_dim` not divisible by `heads`.
"""
function MultiheadAttention(input_dim::Integer, hidden_dim::Integer, heads::Integer, attention_fn::F = attention) where {F}

    in_dim = Int(input_dim)
    hid_dim = Int(hidden_dim)
    nheads = Int(heads)

    nheads > 0 || throw(ArgumentError("heads must be > 0"))
    hid_dim % nheads == 0 || throw(ArgumentError("hidden_dim must be divisible by heads"))

    WQ = Flux.Dense(in_dim, hid_dim, bias=false)
    WK = Flux.Dense(in_dim, hid_dim, bias=false)
    WV = Flux.Dense(in_dim, hid_dim, bias=false)
    WO = Flux.Dense(hid_dim, hid_dim, bias=false)

    return MultiheadAttention{F}(nheads, WQ, WK, WV, WO, attention_fn)
end


"""
    (mh::MultiheadAttention)(Q, K, V, mask=nothing)

Apply multi-head attention with queries `Q`, keys `K`, and values `V`.

# Arguments
- `Q::AbstractArray{<:AbstractFloat}`: query tensor `(d, m, bs)`.
- `K::AbstractArray{<:AbstractFloat}`: key tensor `(d, n, bs)`.
- `V::AbstractArray{<:AbstractFloat}`: value tensor `(vd, n, bs)`.
- `mask::Mask`: optional attention mask applied additively.

# Returns
- `AbstractArray`: attended output tensor `(vd, m, bs)`.
"""
function (mh::MultiheadAttention)(Q::AbstractArray{T}, K::AbstractArray{T}, V::AbstractArray{T}, mask::Mask=nothing) where {T<:AbstractFloat}
    Qp = mh.WQ(Q)
    Kp = mh.WK(K)
    Vp = mh.WV(V)
    return _forward_mha(mh, Qp, Kp, Vp, mask, nothing)
end

"""
    (mh::MultiheadAttention)(X, Y, X_mask=nothing, Y_mask=nothing)

Masked multi-head attention variant with separate masks for queries and keys/values.

# Arguments
- `X::AbstractArray{<:AbstractFloat}`: query input tensor `(d, m, bs)`.
- `Y::AbstractArray{<:AbstractFloat}`: key/value input tensor `(d, n, bs)`.
- `X_mask::Mask`: optional mask for query positions.
- `Y_mask::Mask`: optional mask for key/value positions.

# Returns
- `AbstractArray`: masked attended output tensor `(d, m, bs)`.
"""
function (mh::MultiheadAttention)(X::AbstractArray{T}, Y::AbstractArray{T}, X_mask::Mask=nothing, Y_mask::Mask=nothing) where {T<:AbstractFloat}
    Qp = mh.WQ(X)
    Kp = mh.WK(Y)
    Vp = mh.WV(Y)
    att_mask = _build_attention_mask(T, X_mask, Y_mask, size(Qp, 2), size(Kp, 2), size(Qp, 3); multihead=true)
    return _forward_mha(mh, Qp, Kp, Vp, att_mask, X_mask)
end

# Shared internals
"""
    _forward_mha(mh, Q, K, V, att_mask, out_mask)

Internal forward pass for multi-head attention.
Splits heads, applies attention, merges heads, applies output projection.
"""
function _forward_mha(mh::MultiheadAttention, Q::AbstractArray{T}, K::AbstractArray{T}, V::AbstractArray{T}, att_mask::MaskT{T}, out_mask::Mask) where T<:AbstractFloat

    Qh, Kh, Vh, dᵥ, m, bs, _ = _split_heads(Q, K, V, mh.heads)
    values = isnothing(att_mask) ? mh.attention(Qh, Kh, Vh) : mh.attention(Qh, Kh, Vh, att_mask)
    values = _merge_heads(values, dᵥ, m, bs)
    values = multiplicative_masking(values, out_mask)
    return mh.WO(values)
end

"""
    _split_heads(Q, K, V, heads)

Reshape and permute tensors to separate multi-head dimensions.
Q: (d, m, bs) → (d_head, m, heads, bs)
K, V: (d, n, bs) → (d_head, n, heads, bs)
"""
function _split_heads(Q::AbstractArray{T}, K::AbstractArray{T}, V::AbstractArray{T}, heads::Int) where T<:AbstractFloat
    d, m, bs = size(Q)
    _, n, _ = size(K)
    dᵥ, _, _ = size(V)

    head_qk = d ÷ heads
    head_v = dᵥ ÷ heads

    Qh = permutedims(reshape(Q, (head_qk, heads, m, bs)), (1, 3, 2, 4))
    Kh = permutedims(reshape(K, (head_qk, heads, n, bs)), (1, 3, 2, 4))
    Vh = permutedims(reshape(V, (head_v, heads, n, bs)), (1, 3, 2, 4))

    return Qh, Kh, Vh, dᵥ, m, bs, n
end

"""
    _merge_heads(values, dᵥ, m, bs)

Reverse split_heads operation. Merges multi-head dimension back.
(d_head, m, heads, bs) → (d, m, bs)
"""
function _merge_heads(values::AbstractArray, dᵥ::Int, m::Int, bs::Int)
    reshape(permutedims(values, (1, 3, 2, 4)), (dᵥ, m, bs))
end

additive_masking(X::AbstractArray{T}, mask::AbstractArray{T}) where T<:AbstractFloat = X .+ mask
additive_masking(X::AbstractArray{T}, ::Nothing) where T<:AbstractFloat = X

multiplicative_masking(X::AbstractArray{T}, mask::AbstractArray{T}) where T <: AbstractFloat    = X .* mask
multiplicative_masking(X::AbstractArray{T}, mask::AbstractArray{Bool}) where T <: AbstractFloat = X .* mask
multiplicative_masking(X::AbstractArray{T}, ::Nothing) where T <: AbstractFloat = X

to_additive_mask(::Type{T}, mask::AbstractArray{Bool}) where T<:AbstractFloat = ifelse.(mask, zero(T), T(-1e30))
to_additive_mask(::Type{T}, mask::AbstractArray{T}) where T<:AbstractFloat = mask
to_additive_mask(::Type{T}, ::Nothing) where T<:AbstractFloat = nothing

"""
    _build_attention_mask(T, X_mask, Y_mask, m, n, bs; multihead=true)

Combine query and key masks into attention weight mask.
Reshapes and broadcasts to either `(n, m, 1, bs)` (multihead) or `(n, m, bs)` (single-head/3D).
Converts bool masks to additive form (0 for valid, -inf for invalid).
"""
function _build_attention_mask(::Type{T}, X_mask::Mask, Y_mask::Mask, m::Int, n::Int, bs::Int; multihead::Bool=true) where T<:AbstractFloat
    if X_mask === nothing && Y_mask === nothing
        return nothing
    end
    xshape = multihead ? (1, m, 1, bs) : (1, m, bs)
    yshape = multihead ? (n, 1, 1, bs) : (n, 1, bs)

    if X_mask === nothing
        return to_additive_mask(T, reshape(Y_mask, yshape))
    elseif Y_mask === nothing
        return to_additive_mask(T, reshape(X_mask, xshape))
    else
        x_bool = reshape(X_mask, xshape)
        y_bool = reshape(Y_mask, yshape)
        return to_additive_mask(T, x_bool .& y_bool)
    end
end

"""
    attention(Q, K, V, mask=nothing)

Standard dot-product attention kernel with scaled softmax.
Supports both 3D (single head) and 4D (multi-head) tensor formats.

# Arguments
- `Q::AbstractArray`: query tensor.
- `K::AbstractArray`: key tensor.
- `V::AbstractArray`: value tensor.
- `mask::Mask`: optional additive mask for attention weights.

# Returns
- `AbstractArray`: weighted sum of values scaled by attention weights.
"""
function attention(Q::AbstractArray{T, 3}, K::AbstractArray{T, 3}, V::AbstractArray{T, 3}, mask::MaskT{T}=nothing) where T <: AbstractFloat
    # Attention for 3D tensors
    # Q ∈ ℝ^{m,d} ~ (d, m, bs)
    # K ∈ ℝ^{n,d} ~ (d, n, bs)
    # V ∈ ℝ^{n,vd} ~ (vd, n, bs)
    mask = to_additive_mask(T, mask)    
    _attention(Q, K, V, (2,1,3), mask)
end

function attention(Q::AbstractArray{T, 4}, K::AbstractArray{T, 4}, V::AbstractArray{T, 4}, mask::MaskT{T}=nothing) where T <: AbstractFloat
    # Attention for 4D tensors
    # Q ∈ ℝ^{h,m,d} ~ (d, m, h, bs)
    # K ∈ ℝ^{h,n,d} ~ (d, n, h, bs)
    # V ∈ ℝ^{h,n,vd} ~ (vd, n, h, bs)  
    mask = to_additive_mask(T, mask)       
    _attention(Q, K, V, (2,1,3,4), mask)
end

"""
    _attention(Q, K, V, pdims, mask)

Dot-product attention kernel: softmax(Q K^T / sqrt(d_k)) V.
Softmax is applied over keys (dims=1), normalizing each query's attention.
Used by standard attention mechanism.
"""
function _attention(Q::AbstractArray{T}, K::AbstractArray{T}, V::AbstractArray{T}, pdims::Tuple, mask::Union{AbstractArray{T}, Nothing}) where T <: AbstractFloat
    #  batched_mul ... matrix multiplication in first (last two) dimensions
    dₖ = size(K, 1)
    dₖ = inv(sqrt(T(dₖ)))
    Kᵀ = permutedims(K, pdims) 
    A = batched_mul(Kᵀ, Q) .* dₖ    # (n, d, h, BS) ⊠ (d, m, h, BS) -> (n, m, h, BS) 
    A = additive_masking(A, mask)   # if mask not nothing A will be masked
    A = Flux.softmax(A, dims=1)         # softmax over n for each m, standard softmax;
    return batched_mul(V, A)        # (vd, n, h, bs) ⊠ (n, m, h, BS) -> (vd, m, h, BS)
end


"""
    slot_attention(Q, K, V, mask=nothing)

Slot attention kernel normalizing attention over set positions (not features).
Used in Set Transformer and Slot Attention mechanisms.
Supports both 3D (single head) and 4D (multi-head) tensor formats.

# Arguments
- `Q::AbstractArray`: query tensor (slots).
- `K::AbstractArray`: key tensor (set elements).
- `V::AbstractArray`: value tensor (set elements).
- `mask::Mask`: optional additive mask for attention weights.

# Returns
- `AbstractArray`: weighted mean of values per slot, normalized over set.
"""
function slot_attention(Q::AbstractArray{T, 3}, K::AbstractArray{T, 3}, V::AbstractArray{T, 3}, mask::MaskT{T}=nothing) where T <: AbstractFloat
    # tensor shape -> (feature_dim, n - samples in set, BS)
    # Q ∈ ℝ^{m,d} ~ (d, m, bs)
    # K ∈ ℝ^{n,d} ~ (d, n, bs)
    # V ∈ ℝ^{n,vd} ~ (vd, n, bs) 
    # Output: (vd, n, bs) ⊠ (n, m, BS) -> (vd, m, BS)
    mask = to_additive_mask(T, mask)
    return _slot_attention(Q, K, V, (2,1,3), mask)
end

function slot_attention(Q::AbstractArray{T, 4}, K::AbstractArray{T, 4}, V::AbstractArray{T, 4}, mask::MaskT{T}=nothing) where T <: AbstractFloat
    # Attention for 4D tensors
    # Q ∈ ℝ^{h,m,d} ~ (d, m, h, bs)
    # K ∈ ℝ^{h,n,d} ~ (d, n, h, bs)
    # V ∈ ℝ^{h,n,vd} ~ (vd, n, h, bs) 
    # Output: (vd, n, h, bs) ⊠ (n, m, h, BS) -> (vd, m, h, BS)
    mask = to_additive_mask(T, mask)
    return _slot_attention(Q, K, V, (2,1,3,4), mask)
end

"""
    _slot_attention(Q, K, V, pdims, mask)

Slot attention: softmax over slots (dims=2), then weighted mean per set element.
Normalizes output by sum of attention weights for numerical stability.
Used in Set Transformer and set-based architectures.
"""
function _slot_attention(Q::AbstractArray{T}, K::AbstractArray{T}, V::AbstractArray{T}, pdims::Tuple, mask::Union{AbstractArray{T}, Nothing}) where T <: AbstractFloat
    dₖ = size(K, 1)
    dₖ = inv(sqrt(T(dₖ)))
    Kᵀ = permutedims(K, pdims)
    # 3D Singlehead # (n, d, BS) ⊠ (d, m, BS) -> (n, m, BS)
    # 3D Multihead # (n, d, h) ⊠ (d, m, h) -> (n, m, h)
    # 4D Mulithead # (n, d, h, BS) ⊠ (d, m, h, BS) -> (n, m, h, BS)
    A = batched_mul(Kᵀ, Q) .* dₖ 
    A = additive_masking(A, mask) # if mask not nothing A will be masked
    # softmax around dims=2 cause problem with CUDA
    A = Flux.softmax(A, dims=2)  # softmax over m for each n; normalizes samples not features
    O = batched_mul(V, A)
    O = O ./ Flux.sum(A .+ 1f-5, dims=1) # weighted mean; normalizes features for each sample
end
