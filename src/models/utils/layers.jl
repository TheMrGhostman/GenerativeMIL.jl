struct MultiheadAttentionBlock
    FF::Flux.Dense
    Multihead::MultiheadAttention
    LN1::Flux.LayerNorm
    LN2::Flux.LayerNorm
end

Flux.@functor MultiheadAttentionBlock

function MultiheadAttentionBlock(hidden_dim::Int, heads::Int)
    # input_dim is equall to hidden_dim, if not there would be problem in "Q.+Multihead()"
    mh = MultiheadAttention(hidden_dim, hidden_dim, heads, slot_attention)
    ff = Flux.Dense(hidden_dim, hidden_dim)
    ln1 = Flux.LayerNorm(hidden_dim)
    ln2 = Flux.LayerNorm(hidden_dim)
    return MultiheadAttentionBlock(ff, mh, ln1, ln2)
end

function (mab::MultiheadAttentionBlock)(Q::AbstractArray{T}, V::AbstractArray{T}) where T <: Real
    # Q ∈ ℝ^{m,d} ~ (d, m, bs)
    # V ∈ ℝ^{n,d} ~ (d, n, bs) 
    a = mab.LN1(Q .+ mab.Multihead(Q, V, V)) # (d, m, bs) .+ (d, m, bs)
    return mab.LN2(a .+ mab.FF(a))
end

function (mab::MultiheadAttentionBlock)(Q::AbstractArray{T}, V::AbstractArray{T}, 
    Q_mask::Union{AbstractArray{Bool}, Nothing}=nothing, V_mask::Union{AbstractArray{Bool}, Nothing}=nothing) where T <: Real
    # Q ∈ ℝ^{m,d} ~ (d, m, bs)
    # V ∈ ℝ^{n,d} ~ (d, n, bs) 
    # Q_mask ∈ ℝ^{m} ~ (1, m, bs) 
    # V_mask ∈ ℝ^{n} ~ (1, n, bs) 
    a = mab.LN1(Q .+ mab.Multihead(Q, V, Q_mask, V_mask)) # (d, m, bs) .+ (d, m, bs)
    a = mab.LN2(a .+ mab.FF(a)) # because of masking
    return (Q_mask !== nothing) ? a .* Q_mask : a
end


struct InducedSetAttentionBlock
    MAB1::MultiheadAttentionBlock
    MAB2::MultiheadAttentionBlock
    I::AbstractArray{<:Real} # Inducing points #TODO add as trainable parameter
end

Flux.@functor InducedSetAttentionBlock

Flux.trainable(isab::InducedSetAttentionBlock) = (isab.MAB1, isab.MAB2, isab.I)

# simple constructor
function InducedSetAttentionBlock(m::Int, hidden_dim::Int, heads::Int)
    mab1 = MultiheadAttentionBlock(hidden_dim, heads)
    mab2 = MultiheadAttentionBlock(hidden_dim, heads)
    I = randn(Float32, hidden_dim, m) # keep batch size as free parameter
    return InducedSetAttentionBlock(mab1, mab2, I)
end

function (isab::InducedSetAttentionBlock)(x::AbstractArray{T}) where T <: Real
    # I ∈ ℝ^{m,d} ~ (d, m, bs)
    # x ∈ ℝ^{n,d} ~ (d, n, bs) 
    # MAB1((d, m, bs), (d, n, bs)) -> (d, m, bs)
    I = repeat(isab.I, 1, 1, size(x)[end])# (d, m, 1) -> (d, m, bs) 
    h = isab.MAB1(I, x) # h ~ (d, m, bs)
    # MAB2((d, n, bs), (d, m, bs)) -> (d, n, bs)
    return isab.MAB2(x, h), h # (d, n, bs), (d, m, bs)
end

function (isab::InducedSetAttentionBlock)(x::AbstractArray{T}, 
    x_mask::Union{AbstractArray{Bool}, Nothing}=nothing) where T <: Real
    # I ∈ ℝ^{m,d} ~ (d, m, bs)
    # x ∈ ℝ^{n,d} ~ (d, n, bs) 
    # x_mask ∈ ℝ^{n} ~ (1, n, bs) 
    # MAB1((d, m, bs), (d, n, bs), _, (1, n, bs)) -> (d, m, bs)
    I = repeat(isab.I, 1, 1, size(x)[end])# (d, m, 1) -> (d, m, bs) 
    h = isab.MAB1(I, x, nothing, x_mask) # h ~ (d, m, bs)
    # MAB2((d, n, bs), (d, m, bs), (1, n, bs), _) -> (d, n, bs)
    return isab.MAB2(x, h, x_mask, nothing), h # (d, n, bs), (d, m, bs)
end

struct InducedSetAttentionHalfBlock
    MAB1::MultiheadAttentionBlock
    I::AbstractArray{<:Real} # Inducing points 
end

Flux.@functor InducedSetAttentionHalfBlock

Flux.trainable(isab::InducedSetAttentionHalfBlock) = (isab.MAB1, isab.I)

# simple constructor
function InducedSetAttentionHalfBlock(m::Int, hidden_dim::Int, heads::Int)
    mab1 = MultiheadAttentionBlock(hidden_dim, heads)
    I = randn(Float32, hidden_dim, m) # keep batch size as free parameter
    return InducedSetAttentionHalfBlock(mab1, I)
end

function (isab::InducedSetAttentionHalfBlock)(x::AbstractArray{T}) where T <: Real
    # I ∈ ℝ^{m,d} ~ (d, m, bs)
    # x ∈ ℝ^{n,d} ~ (d, n, bs) 
    # MAB1((d, m, bs), (d, n, bs)) -> (d, m, bs)
    I = repeat(isab.I, 1, 1, size(x)[end]) # (d, m, 1) -> (d, m, bs) 
    h = isab.MAB1(I, x) # h ~ (d, m, bs)
    return x, h
end

function (isab::InducedSetAttentionHalfBlock)(x::AbstractArray{T},
    x_mask::Union{AbstractArray{Bool}, Nothing}=nothing) where T <: Real
    # I ∈ ℝ^{m,d} ~ (d, m, bs)
    # x ∈ ℝ^{n,d} ~ (d, n, bs) 
    # x_mask ∈ ℝ^{n} ~ (1, n, bs) 
    # MAB1((d, m, bs), (d, n, bs)) -> (d, m, bs)
    I = repeat(isab.I, 1, 1, size(x)[end]) # (d, m, 1) -> (d, m, bs) 
    h = isab.MAB1(I, x, nothing, x_mask) # h ~ (d, m, bs)
    return x, h
end

struct VariationalBottleneck
    prior::Flux.Chain
    posterior::Flux.Chain
    decoder::Flux.Chain
end

Flux.@functor VariationalBottleneck

function (vb::VariationalBottleneck)(h::AbstractArray{T}) where T <: Real
    # computing prior μ, Σ from h
    μ, Σ = vb.prior(h)
    z = μ + Σ * randn(Float32)
    ĥ = vb.decoder(z)
    return z, ĥ, nothing
end

function (vb::VariationalBottleneck)(h::AbstractArray{T}, h_enc::AbstractArray{T}) where T <: Real
    # computing prior μ, Σ from h as well as posterior from h_enc
    μ, Σ = vb.prior(h)
    Δμ, ΔΣ = vb.posterior(h .+ h_enc)
    z = (μ + Δμ) + (Σ .* ΔΣ) * randn(Float32)
    ĥ = vb.decoder(z)
    kld = 0.5 * ( (Δμ.^2 ./ Σ.^2) + ΔΣ.^2 - log.(ΔΣ.^2) .- 1f0 ) # TODO sum/mean .... fix this
    # kld_loss = Flux.mean(Flux.sum(kld, dims=(1,2))) # mean over BatchSize , sum over Dz and Induced Set
    return z, ĥ, kld
end    

function VariationalBottleneck(
    in_dim::Int, z_dim::Int, out_dim::Int, hidden::Int=32, depth::Int=1, activation::Function=identity
)
    encoder_μ = []
    encoder_Δμ = []
    decoder = []
    if depth>=2
        push!(encoder_μ, Flux.Dense(in_dim, hidden, activation))
        push!(encoder_Δμ, Flux.Dense(in_dim, hidden, activation))
        push!(decoder, Flux.Dense(z_dim, hidden, activation))
        for i=1:depth-2
            push!(encoder_μ, Flux.Dense(hidden, hidden, activation))
            push!(encoder_Δμ, Flux.Dense(hidden, hidden, activation))
            push!(decoder, Flux.Dense(hidden, hidden, activation))
        end
        push!(encoder_μ, SplitLayer(hidden, (z_dim, z_dim), (identity, softplus)))
        push!(encoder_Δμ, SplitLayer(hidden, (z_dim, z_dim), (identity, softplus)))
        push!(decoder, Flux.Dense(hidden, out_dim))
    elseif depth==1
        push!(encoder_μ, SplitLayer(in_dim, (z_dim, z_dim), (identity, softplus)))
        push!(encoder_Δμ, SplitLayer(in_dim, (z_dim, z_dim), (identity, softplus)))
        push!(decoder, Flux.Dense(z_dim, out_dim))
    else
        @error("Incorrect depth of VariationalBottleneck")
    end
    encoder_μ = Flux.Chain(encoder_μ...)
    encoder_Δμ = Flux.Chain(encoder_Δμ...)
    decoder = Flux.Chain(decoder...)
    return VariationalBottleneck(encoder_μ, encoder_Δμ, decoder)
end


struct AttentiveBottleneckLayer
    MAB1::MultiheadAttentionBlock
    MAB2::MultiheadAttentionBlock
    VB::VariationalBottleneck
    I::AbstractArray{<:Real}
end

Flux.@functor AttentiveBottleneckLayer

Flux.trainable(abl::AttentiveBottleneckLayer) = (abl.MAB1, abl.MAB2, abl.VB, abl.I)

function (abl::AttentiveBottleneckLayer)(x::AbstractArray{T}) where T <: Real
    # generation
    # I ∈ ℝ^{m,d} ~ (d, m, bs)
    # x ∈ ℝ^{n,d} ~ (d, n, bs) 
    # MAB1((d, m, bs), (d, n, bs)) -> (d, m, bs)
    I = repeat(abl.I, 1, 1, size(x)[end])# (d, m, 1) -> (d, m, bs)
    h = abl.MAB1(I, x) # h ~ (d, m, bs)
    z, ĥ, kld = abl.VB(h) # z, h, kld ~ (zdim, m, bs), (d, m, bs), kld_loss 
    # MAB2((d, n, bs), (d, m, bs)) -> (d, n, bs)
    return abl.MAB2(x, ĥ), kld, ĥ, z  # (d, n, bs), (d, m, bs)
end

function (abl::AttentiveBottleneckLayer)(x::AbstractArray{T}, h_enc::AbstractArray{T}) where T <: Real
    # inference
    # I     ∈ ℝ^{m,d} ~ (d, m, bs)
    # x     ∈ ℝ^{n,d} ~ (d, n, bs) 
    # h_enc ∈ ℝ^{n,d} ~ (d, m, bs) 
    # MAB1((d, m, bs), (d, n, bs)) -> (d, m, bs)
    I = repeat(abl.I, 1, 1, size(x)[end]) # (d, m, 1) -> (d, m, bs)
    h = abl.MAB1(I, x) # h ~ (d, m, bs)
    z, ĥ, kld = abl.VB(h, h_enc) # z, h, kld ~ (zdim, m, bs), (d, m, bs), kld_loss (d, m, bs)
    kld = Flux.mean(Flux.sum(kld, dims=(1,2)))
    # MAB2((d, n, bs), (d, m, bs)) -> (d, n, bs)
    return abl.MAB2(x, ĥ), kld, ĥ, z # (d, n, bs), scalar, (zdim, m, bs), ...
end

function (abl::AttentiveBottleneckLayer)(x::AbstractArray{T}, h_enc::AbstractArray{T}, 
    x_mask::Union{AbstractArray{Bool}, Nothing}=nothing) where T <: Real
    # inference
    # I     ∈ ℝ^{m,d} ~ (d, m, bs)
    # x     ∈ ℝ^{n,d} ~ (d, n, bs) 
    # h_enc ∈ ℝ^{n,d} ~ (d, m, bs) 
    # MAB1((d, m, bs), (d, n, bs)) -> (d, m, bs)
    I = repeat(abl.I, 1, 1, size(x)[end]) # (d, m, 1) -> (d, m, bs)
    h = abl.MAB1(I, x, nothing, x_mask) # h ~ (d, m, bs)
    z, ĥ, kld = abl.VB(h, h_enc) # z, h, kld ~ (zdim, m, bs), (d, m, bs), kld_loss (d, m, bs)
    kld = Flux.mean(Flux.sum(kld, dims=(1,2)))
    # MAB2((d, n, bs), (d, m, bs)) -> (d, n, bs)
    return abl.MAB2(x, ĥ, x_mask, nothing), kld, ĥ, z # (d, n, bs), scalar, (zdim, m, bs), ...
end

# simple constructor
function AttentiveBottleneckLayer(
    m::Int, hidden_dim::Int, heads::Int, z_dim::Int, hidden::Int, depth::Int, activation::Function=identity
)
    mab1 = MultiheadAttentionBlock(hidden_dim, heads)
    mab2 = MultiheadAttentionBlock(hidden_dim, heads)
    I = randn(Float32, hidden_dim, m) # keep batch size as free parameter
    vb = VariationalBottleneck(hidden_dim, z_dim, hidden_dim, hidden, depth, activation)
    return AttentiveBottleneckLayer(mab1, mab2, vb, I)
end
