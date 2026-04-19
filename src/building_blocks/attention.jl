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

function Base.show(io::IO, ::MIME"text/plain", m::MultiheadAttention{F}) where F
    attention_name = m.attention === attention ? "standard" : 
                     m.attention === slot_attention ? "slot" : 
                     "attention (custom)"
    styled_io = IOContext(io, :color => true)
    
    print(io, "MultiheadAttention{$(F)}\n")
    print(io, "  Heads: $(m.heads), Attention: $attention_name\n\n")
    
    # Počítej parametry pro každou vrstvu
    layers = [("WQ", m.WQ), ("WK", m.WK), ("WV", m.WV), ("WO", m.WO)]
    total_params = 0
    total_bytes = 0
    num_arrays = 0
    
    for (name, layer) in layers
        params = length(layer.weight) # no bias
        bytes = params * sizeof(eltype(layer.weight))
        total_params += params
        total_bytes += bytes
        num_arrays += 1
        
        print(io, "  $name: $(layer)")
        Base.printstyled(styled_io, ", # $params parameters\n"; color=:light_black)
    end
    
    Base.printstyled(styled_io, ")  # Total: $num_arrays arrays, $total_params parameters, $total_bytes bytes."; color=:light_black)
end

AbstractTrees.children(m::MultiheadAttention) = (("W_Query ", m.WQ), ("W_Key ", m.WK), ("W_Value ", m.WV), ("W_Output", m.WO), m.attention)
AbstractTrees.printnode(io::IO, m::MultiheadAttention) = print(io, "MultiheadAttention - ($(m.heads) heads)")


function (mh::MultiheadAttention)(Q::AbstractArray{T}, K::AbstractArray{T}, V::AbstractArray{T}, mask::Mask=nothing) where {T<:AbstractFloat}
    Qp = mh.WQ(Q)
    Kp = mh.WK(K)
    Vp = mh.WV(V)
    return _forward_mha(mh, Qp, Kp, Vp, mask, nothing)
end

function (mh::MultiheadAttention)(X::AbstractArray{T}, Y::AbstractArray{T}, X_mask::Mask=nothing, Y_mask::Mask=nothing) where {T<:AbstractFloat}
    Qp = mh.WQ(X)
    Kp = mh.WK(Y)
    Vp = mh.WV(Y)
    att_mask = _build_attention_mask(T, X_mask, Y_mask, size(Qp, 2), size(Kp, 2), size(Qp, 3))
    return _forward_mha(mh, Qp, Kp, Vp, att_mask, X_mask)
end


# Shared internals 
function _forward_mha(mh::MultiheadAttention, Q::AbstractArray{T}, K::AbstractArray{T}, V::AbstractArray{T}, att_mask::MaskT{T}, out_mask::Mask) where T<:AbstractFloat

    Qh, Kh, Vh, dᵥ, m, bs, _ = _split_heads(Q, K, V, mh.heads)
    values = isnothing(att_mask) ? mh.attention(Qh, Kh, Vh) : mh.attention(Qh, Kh, Vh, att_mask)
    values = _merge_heads(values, dᵥ, m, bs)
    values = multiplicative_masking(values, out_mask)
    return mh.WO(values)
end

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

function _build_attention_mask(::Type{T}, X_mask::Mask, Y_mask::Mask, m::Int, n::Int, bs::Int) where {T<:AbstractFloat}
    if X_mask === nothing && Y_mask === nothing
        return nothing
    end

    if X_mask === nothing
        return to_additive_mask(T, reshape(Y_mask, (n, 1, 1, bs)))
    elseif Y_mask === nothing
        return to_additive_mask(T, reshape(X_mask, (1, m, 1, bs)))
    else
        x_bool = reshape(X_mask, (1, m, 1, bs))
        y_bool = reshape(Y_mask, (n, 1, 1, bs))
        return to_additive_mask(T, x_bool .& y_bool)
    end
end

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

function _attention(Q::AbstractArray{T}, K::AbstractArray{T}, V::AbstractArray{T}, pdims::Tuple, mask::Union{AbstractArray{T}, Nothing}) where T <: AbstractFloat
    #  batched_mul ... matrix multiplication in first (last two) dimensions
    dₖ = size(K, 1)
    dₖ = inv(sqrt(T(dₖ)))
    Kᵀ = permutedims(K, pdims) 
    A = batched_mul(Kᵀ, Q) .* dₖ    # (n, d, h, BS) ⊠ (d, m, h, BS) -> (n, m, h, BS) 
    A = additive_masking(A, mask)   # if mask not nothing A will be masked
    A = _softmax(A, dims=1)         # softmax over n for each m, standard softmax;
    return batched_mul(V, A)        # (vd, n, h, bs) ⊠ (n, m, h, BS) -> (vd, m, h, BS)
end


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
    A = _softmax(A, dims=2)  # softmax over m for each n; normalizes samples not features
    O = batched_mul(V, A)
    O = O ./ Flux.sum(A .+ 1f-5, dims=1) # weighted mean; normalizes features for each sample
end

function _softmax(x::AbstractArray{T}; dims::Int=1) where T<: AbstractFloat
    Flux.softmax(x; dims=dims)
end

function _softmax(x::CuArray{T}; dims::Int=1) where T <: AbstractFloat
    # numerically stable softmax for CUDA; NNLib.softmax causes problems with CUDA with Julia 1.11.3
    m = maximum(x; dims=dims)
    ex = exp.(x .- m)
    ex ./ sum(ex; dims=dims)
end