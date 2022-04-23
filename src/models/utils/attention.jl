using Flux
using Transformers: batchedmul # can do 4D tensors

struct MultiheadAttention
    heads::Int32
    WQ::Flux.Dense
    WK::Flux.Dense
    WV::Flux.Dense
    WO::Flux.Dense
    attention::Function # type of attention -> attention or slot_attention
end

Flux.@functor MultiheadAttention

Flux.trainable(mh::MultiheadAttention) = (mh.WQ, mh.WK, mh.WV, mh.WO)

# simple constructor
function MultiheadAttention(input_dim::Int, hidden_dim::Int, heads::Int, attention::Function=attention)
    (hidden_dim % heads != 0) ? error("hidden_dim modulo heads must be equall to zero!!!") : nothing
    WQ = Flux.Dense(input_dim, hidden_dim, bias=false)
    WK = Flux.Dense(input_dim, hidden_dim, bias=false)
    WV = Flux.Dense(input_dim, hidden_dim, bias=false)
    WO = Flux.Dense(hidden_dim, hidden_dim, bias=false)
    return MultiheadAttention(heads, WQ, WK, WV, WO, attention)
end

function (mh::MultiheadAttention)(Q::AbstractArray{T}, K::AbstractArray{T}, V::AbstractArray{T}) where T <: Real
    # Project Q, K, V to new ones
    Q = mh.WQ(Q)
    K = mh.WK(K)
    V = mh.WV(V)
    # get correct dims
    d, m, bs = size(Q);
    _, n, _ = size(K);
    d_v, _, _ = size(V);
    # compute head dimensions   
    head_qk = Int(d // mh.heads)
    head_v = Int(d_v // mh.heads)
    # split into separate Heads and permute dims
    Q = permutedims(reshape(Q, (head_qk, mh.heads, m, bs)), (1,3,2,4))
    K = permutedims(reshape(K, (head_qk, mh.heads, n, bs)), (1,3,2,4))
    V = permutedims(reshape(V, (head_v, mh.heads, n, bs)), (1,3,2,4))
    # run through attention 
    values = mh.attention(Q, K, V)
    # permute dims back and reshape
    values = reshape(permutedims(values, (1,3,2,4)), (d_v, m, bs))
    # project values
    return mh.WO(values)
end

function (mh::MultiheadAttention)(X::AbstractArray{T}, Y::AbstractArray{T}, 
    X_mask::Union{BitArray, Nothing}=nothing, Y_mask::Union{BitArray, Nothing}=nothing) where T <: Real
    # X_mask ~ (1, m, BS)
    # Y_mask ~ (1, n, BS)
    Q = (X_mask !== nothing) ? mh.WQ(X) .* X_mask : mh.WQ(X)
    K = (Y_mask !== nothing) ? mh.WK(Y) .* Y_mask : mh.WK(Y)
    V = (Y_mask !== nothing) ? mh.WV(Y) .* Y_mask : mh.WV(Y)
    # get correct dims
    d, m, bs = size(Q);
    _, n, _ = size(K);
    d_v, _, _ = size(V);
    # compute head dimensions   
    head_qk = Int(d // mh.heads)
    head_v = Int(d_v // mh.heads)
    # split into separate Heads and permute dims
    Q = permutedims(reshape(Q, (head_qk, mh.heads, m, bs)), (1,3,2,4))
    K = permutedims(reshape(K, (head_qk, mh.heads, n, bs)), (1,3,2,4))
    V = permutedims(reshape(V, (head_v, mh.heads, n, bs)), (1,3,2,4))

    if (X_mask === nothing) & (Y_mask === nothing)
        mask = nothing
    elseif (X_mask === nothing) & (Y_mask !== nothing)
        mask = reshape(Y_mask, (n, 1, 1, bs))
    elseif (X_mask !== nothing) & (Y_mask === nothing)
        mask = reshape(X_mask, (1, m, 1, bs))
    else 
        error("Both X_mask and Y_mask are not nothing!!")
    end
    values = mh.attention(Q, K, V, mask) 
    # permute dims back and reshape
    values = reshape(permutedims(values, (1,3,2,4)), (d_v, m, bs))
    values = (X_mask !== nothing) ? values.* X_mask : values
    # project values
    #output = (X_mask !== nothing) ? mh.WO(values) .* X_mask : mh.WO(values)
    return mh.WO(values)
end
# masked version


function attention(Q::AbstractArray{T, 3}, K::AbstractArray{T, 3}, V::AbstractArray{T, 3}) where T <: Real
    # Attention for 3D tensors
    # Q ∈ ℝ^{m,d} ~ (d, m, bs)
    # K ∈ ℝ^{n,d} ~ (d, n, bs)
    # V ∈ ℝ^{n,vd} ~ (vd, n, bs)         
    dₖ = size(K, 1)
    dₖ = convert(Float32, 1/sqrt(dₖ))
    Kᵀ = permutedims(K, (2,1,3))
    # batched_mul \boxtimes
    A = (Kᵀ ⊠ Q) .* dₖ  # (n, d, BS) ⊠ (d, m, BS) -> (n, m, BS)
    A = Flux.softmax(A, dims=1) # softmax over n for each m, standard softmax; 
    return V ⊠ A # (vd, n, bs) ⊠ (n, m, BS) -> (vd, m, BS)
end

function attention(Q::AbstractArray{T, 4}, K::AbstractArray{T, 4}, V::AbstractArray{T, 4}) where T <: Real
    # Attention for 4D tensors
    # Q ∈ ℝ^{h,m,d} ~ (d, m, h, bs)
    # K ∈ ℝ^{h,n,d} ~ (d, n, h, bs)
    # V ∈ ℝ^{h,n,vd} ~ (vd, n, h, bs)         
    dₖ = size(K, 1)
    dₖ = convert(Float32, 1/sqrt(dₖ))
    Kᵀ = permutedims(K, (2,1,3,4))
    # batched_mul can do only 3D tensors 
    A = batchedmul(Kᵀ, Q) .* dₖ  #  (n, d, h, BS) ⊠ (d, m, h, BS) -> (n, m, h, BS)
    A = Flux.softmax(A, dims=1) # softmax over n for each m, standard softmax; 
    return batchedmul(V, A)  # (vd, n, h, bs) ⊠ (n, m, h, BS) -> (vd, m, h, BS)
end

function slot_attention(Q::AbstractArray{T, 3}, K::AbstractArray{T, 3}, V::AbstractArray{T, 3}) where T <: Real
    # tensor shape -> (feature_dim, n - samples in set, BS)
    # Q ∈ ℝ^{m,d} ~ (d, m, bs)
    # K ∈ ℝ^{n,d} ~ (d, n, bs)
    # V ∈ ℝ^{n,vd} ~ (vd, n, bs)         
    dₖ = size(K, 1)
    dₖ = convert(Float32, 1/sqrt(dₖ))
    Kᵀ = permutedims(K, (2,1,3))
    # batched_mul \boxtimes
    A = (Kᵀ ⊠ Q) .* dₖ  # (n, d, BS) ⊠ (d, m, BS) -> (n, m, BS)
    A = Flux.softmax(A, dims=2) # softmax over m for each n; normalizes samples not features
    W = A ./ Flux.sum(A, dims=1) # weighted mean; normalizes features for each sample
    return V ⊠ W # (vd, n, bs) ⊠ (n, m, BS) -> (vd, m, BS)
end

function slot_attention(Q::AbstractArray{T, 4}, K::AbstractArray{T, 4}, V::AbstractArray{T, 4}, mask::Union{BitArray, Nothing}=nothing) where T <: Real
    # Attention for 4D tensors
    # Q ∈ ℝ^{h,m,d} ~ (d, m, h, bs)
    # K ∈ ℝ^{h,n,d} ~ (d, n, h, bs)
    # V ∈ ℝ^{h,n,vd} ~ (vd, n, h, bs)           
    dₖ = size(K, 1)
    dₖ = convert(Float32, 1/sqrt(dₖ))
    Kᵀ = permutedims(K, (2,1,3,4))
    # batched_mul can do only 3D tensors 
    A = batchedmul(Kᵀ, Q) .* dₖ # (n, d, h, BS) ⊠ (d, m, h, BS) -> (n, m, h, BS)
    A = (mask !== nothing) ? A .* mask : A
    A = Flux.softmax(A, dims=2) # softmax over m for each n; normalizes samples not features
    W = A ./ Flux.sum(A, dims=1) # weighted mean; normalizes features for each sample
    return batchedmul(V, W) # (vd, n, h, bs) ⊠ (n, m, h, BS) -> (vd, m, h, BS)
end


