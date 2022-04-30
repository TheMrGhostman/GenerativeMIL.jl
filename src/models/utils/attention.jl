using Flux
using Transformers: batchedmul # can do 4D tensors

struct MultiheadAttention
    heads::Int32
    WQ::Union{Flux.Dense, MaskedDense}
    WK::Union{Flux.Dense, MaskedDense}
    WV::Union{Flux.Dense, MaskedDense}
    WO::Union{Flux.Dense, MaskedDense}
    attention::Function # type of attention -> attention or slot_attention
end

Flux.@functor MultiheadAttention

Flux.trainable(mh::MultiheadAttention) = (mh.WQ, mh.WK, mh.WV, mh.WO)

# simple constructor
function MultiheadAttention(input_dim::Int, hidden_dim::Int, heads::Int, attention::Function=attention)
    (hidden_dim % heads != 0) ? error("hidden_dim modulo heads must be equall to zero!!!") : nothing
    WQ = MaskedDense(input_dim, hidden_dim, bias=false)
    WK = MaskedDense(input_dim, hidden_dim, bias=false)
    WV = MaskedDense(input_dim, hidden_dim, bias=false)
    WO = MaskedDense(hidden_dim, hidden_dim, bias=false)
    return MultiheadAttention(heads, WQ, WK, WV, WO, attention)
end

function Base.show(io::IO, m::MultiheadAttention)
    print(io, "MultiheadAttention(")
    print(io, "\n\t - heads = $(m.heads) \n\t - WQ = $(m.WQ) \n\t - WK = $(m.WK)")
    print(io, "\n\t - WV = $(m.WV) \n\t - WO = $(m.WO) \n\t - attention = $(m.attention) \n\t ) ")
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
    X_mask::Union{AbstractArray{Bool}, Nothing}=nothing, Y_mask::Union{AbstractArray{Bool}, Nothing}=nothing) where T <: Real
    # X_mask ~ (1, m, BS)
    # Y_mask ~ (1, n, BS)
    Q = mh.WQ(X, X_mask) #(X_mask !== nothing) ? mh.WQ(X) .* X_mask : mh.WQ(X)
    K = mh.WK(Y, Y_mask) #(Y_mask !== nothing) ? mh.WK(Y) .* Y_mask : mh.WK(Y)
    V = mh.WV(Y, Y_mask) #(Y_mask !== nothing) ? mh.WV(Y) .* Y_mask : mh.WV(Y)
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
        #println(Y_mask |> size, Y_mask |> typeof)
    elseif (X_mask !== nothing) & (Y_mask === nothing)
        mask = reshape(X_mask, (1, m, 1, bs))
        #println(X_mask |> size, X_mask |> typeof)
    else 
        error("Both X_mask and Y_mask are not nothing!!")
    end
    #Zygote.ignore() do
    #mask = Array{Float32}(mask)
    n_mask = -1.0f30 .* (1f0 .- mask)
    mask = mask + n_mask
    #println(mask |> size, mask |> typeof)
    #end

    values = mh.attention(Q, K, V, mask) 
    # permute dims back and reshape
    values = reshape(permutedims(values, (1,3,2,4)), (d_v, m, bs))
    #values = (X_mask !== nothing) ? values.* X_mask : values
    # project values
    #output = (X_mask !== nothing) ? mh.WO(values) .* X_mask : mh.WO(values)
    return mh.WO(values, X_mask)
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

function attention(Q::AbstractArray{T, 4}, K::AbstractArray{T, 4}, V::AbstractArray{T, 4},
    mask::Union{AbstractArray{Bool}, Nothing}=nothing) where T <: Real
    # Attention for 4D tensors
    # Q ∈ ℝ^{h,m,d} ~ (d, m, h, bs)
    # K ∈ ℝ^{h,n,d} ~ (d, n, h, bs)
    # V ∈ ℝ^{h,n,vd} ~ (vd, n, h, bs)         
    dₖ = size(K, 1)
    dₖ = convert(Float32, 1/sqrt(dₖ))
    Kᵀ = permutedims(K, (2,1,3,4))
    # batched_mul can do only 3D tensors 
    A = batchedmul(Kᵀ, Q) .* dₖ  #  (n, d, h, BS) ⊠ (d, m, h, BS) -> (n, m, h, BS)
    A = (mask !== nothing) ? A .* mask : A
    A = Flux.softmax(A, dims=1) # softmax over n for each m, standard softmax; 
    return batchedmul(V, A)  # (vd, n, h, bs) ⊠ (n, m, h, BS) -> (vd, m, h, BS)
end


function slot_attention(Q::AbstractArray{T, 3}, K::AbstractArray{T, 3}, V::AbstractArray{T, 3},
    mask::Union{AbstractArray{Bool}, AbstractArray{T} , Nothing}=nothing) where T <: Real
    # tensor shape -> (feature_dim, n - samples in set, BS)
    # Q ∈ ℝ^{m,d} ~ (d, m, bs)
    # K ∈ ℝ^{n,d} ~ (d, n, bs)
    # V ∈ ℝ^{n,vd} ~ (vd, n, bs) 
    # Output: (vd, n, bs) ⊠ (n, m, BS) -> (vd, m, BS)
    return _slot_attention(Q, K, V, batched_mul, (2,1,3), mask=mask)
end

function slot_attention(Q::AbstractArray{T, 4}, K::AbstractArray{T, 4}, V::AbstractArray{T, 4}, 
    mask::Union{AbstractArray{Bool}, AbstractArray{T} , Nothing}=nothing) where T <: Real
    # Attention for 4D tensors
    # Q ∈ ℝ^{h,m,d} ~ (d, m, h, bs)
    # K ∈ ℝ^{h,n,d} ~ (d, n, h, bs)
    # V ∈ ℝ^{h,n,vd} ~ (vd, n, h, bs) 
    # Output: (vd, n, h, bs) ⊠ (n, m, h, BS) -> (vd, m, h, BS)
    return _slot_attention(Q, K, V, batchedmul, (2,1,3,4), mask=mask)
end

function _slot_attention(Q::AbstractArray{T}, K::AbstractArray{T}, V::AbstractArray{T}, matrixmul::Function, pdims::Tuple;
    mask::Union{AbstractArray{Bool}, AbstractArray{T} , Nothing}=nothing) where T <: Real
    dₖ = size(K, 1)
    dₖ = convert(Float32, 1/sqrt(dₖ))
    Kᵀ = permutedims(K, pdims)
    # 3D Singlehead # (n, d, BS) ⊠ (d, m, BS) -> (n, m, BS)
    # 3D Multihead # (n, d, h) ⊠ (d, m, h) -> (n, m, h)
    # 4D Mulithead # (n, d, h, BS) ⊠ (d, m, h, BS) -> (n, m, h, BS)
    A = matrixmul(Kᵀ, Q) .* dₖ 
    A = (mask !== nothing) ? A .* mask : A
    A = Flux.softmax(A, dims=2) # softmax over m for each n; normalizes samples not features
    W = matrixmul(V, A)
    W = W ./ Flux.sum(A .+ 1f-5 , dims=1) # weighted mean; normalizes features for each sample
    return W
end


