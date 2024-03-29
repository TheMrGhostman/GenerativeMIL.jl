"""
    Special Transformer blocks with attention

    *Block*                                 | Used in (papers)
    ----------------------------------------|------------------------
    1) Multihead Attention Blocks           | SetTransformer, SetVAE
    2) InducedSetAttentionBlock             | SetTransformer, SetVAE
    3) InducedSetAttentionHalfBlock ≈ PMA   | SetTransformer, SetVAE
    4) VariationalBottleneck                | SetVAE
    5) AttentiveBottleneckLayer             | SetVAE
    6) AttentiveHalfBlock                   | SetVAE

"""

struct MultiheadAttentionBlock
    FF::Union{Flux.Dense, Flux.Chain}
    Multihead::MultiheadAttention
    LN1::Flux.LayerNorm
    LN2::Flux.LayerNorm
end

Flux.@functor MultiheadAttentionBlock

function MultiheadAttentionBlock(hidden_dim::Int, heads::Int, activation::Function=relu)
    # input_dim is equall to hidden_dim, if not there would be problem in "Q.+Multihead()"
    mh = MultiheadAttention(hidden_dim, hidden_dim, heads, slot_attention)
    ff = Flux.Chain(
        Flux.Dense(hidden_dim, hidden_dim, activation),
        Flux.Dense(hidden_dim, hidden_dim)
    )
    ln1 = Flux.LayerNorm((hidden_dim,1))
    ln2 = Flux.LayerNorm((hidden_dim,1))
    return MultiheadAttentionBlock(ff, mh, ln1, ln2)
end

function Base.show(io::IO, m::MultiheadAttentionBlock)
    print(io, "MultiheadAttentionBlock(")
    print(io, "\n\t - Mulithead = $(m.Multihead) \n\t - FF = $(m.FF) \n\t - LN1 = $(m.LN1) \n\t - LN2 = $(m.LN2) \n ) ")
end

AbstractTrees.children(m::MultiheadAttentionBlock) = (m.Multihead, ("FeedForward", m.FF), ("LayerNorm 1", m.LN1), ("LayerNorm 2", m.LN2))
AbstractTrees.printnode(io::IO, m::MultiheadAttentionBlock) = print(io, "MultiheadAttentionBlock")

function (mab::MultiheadAttentionBlock)(V::AbstractArray{T}) where T <: Real
    # Self Attention
    # V ∈ ℝ^{n,d} ~ (d, n, bs) 
    a = mab.LN1(V + mab.Multihead(V, V, V)) # (d, n, bs) .+ (d, n, bs)
    return mab.LN2(a + mab.FF(a)) 
end

function (mab::MultiheadAttentionBlock)(Q::AbstractArray{T}, V::AbstractArray{T}) where T <: Real
    # Q ∈ ℝ^{m,d} ~ (d, m, bs)
    # V ∈ ℝ^{n,d} ~ (d, n, bs) 
    a = mab.LN1(Q + mab.Multihead(Q, V, V)) # (d, m, bs) .+ (d, m, bs)
    return mab.LN2(a + mab.FF(a)) 
end

function (mab::MultiheadAttentionBlock)(Q::AbstractArray{T}, V::AbstractArray{T}, 
    Q_mask::Mask=nothing, V_mask::Mask=nothing) where T <: Real
    # Q ∈ ℝ^{m,d} ~ (d, m, bs)
    # V ∈ ℝ^{n,d} ~ (d, n, bs) 
    # Q_mask ∈ ℝ^{m} ~ (1, m, bs) 
    # V_mask ∈ ℝ^{n} ~ (1, n, bs) 
    Q = mask(Q, Q_mask)
    V = mask(V, V_mask)
    a = mab.LN1(Q + mab.Multihead(Q, V, Q_mask, V_mask)) # (d, m, bs) + (d, m, bs) 
    a = mab.LN2(a + mab.FF(a)) 
    return mask(a, Q_mask)
end


struct InducedSetAttentionBlock
    MAB1::MultiheadAttentionBlock
    MAB2::MultiheadAttentionBlock
    I::AbstractArray{<:Real} 
end

Flux.@functor InducedSetAttentionBlock

Flux.trainable(isab::InducedSetAttentionBlock) = (isab.MAB1, isab.MAB2, isab.I)

function Base.show(io::IO, m::InducedSetAttentionBlock)
    print(io, "InducedSetAttentionBlock(")
    print(io, " - MAB1 = $(m.MAB1)\n - MAB2 = $(m.MAB2) \n - I = $(size(m.I)) - $(typeof(m.I)) \n ) ")
end

AbstractTrees.children(m::InducedSetAttentionBlock) = (m.MAB1, m.MAB2, ("Induced Set", m.I))
AbstractTrees.printnode(io::IO, m::InducedSetAttentionBlock) = print(io, "InducedSetAttentionBlock - ($(size(m.I,2)) Induced Sets)")

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

function (isab::InducedSetAttentionBlock)(x::AbstractArray{T}, x_mask::Mask=nothing) where T <: Real
    # I ∈ ℝ^{m,d} ~ (d, m, bs)
    # x ∈ ℝ^{n,d} ~ (d, n, bs) 
    # x_mask ∈ ℝ^{n} ~ (1, n, bs) 
    # MAB1((d, m, bs), (d, n, bs), _, (1, n, bs)) -> (d, m, bs)
    device = get_device(isab.I)
    I = device.ones(Float32, 1,1, size(x)[end]) .* isab.I # (d, m, 1) -> (d, m, bs) 
    # repeat(isab.I, 1, 1, size(x)[end]) ----> repeat on gpu is extremly slow!!
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

function Base.show(io::IO, m::InducedSetAttentionHalfBlock)
    print(io, "InducedSetAttentionHalfBlock(")
    print(io, " - MAB1 = $(m.MAB1)\n - I = $(size(m.I)) - $(typeof(m.I)) \n ) ")
end

AbstractTrees.children(m::InducedSetAttentionHalfBlock) = (m.MAB1, ("Induced Set", m.I))
AbstractTrees.printnode(io::IO, m::InducedSetAttentionHalfBlock) = print(io, "InducedSetAttentionHalfBlock - ($(size(m.I,2)) Induced Sets)")

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

function (isab::InducedSetAttentionHalfBlock)(x::AbstractArray{T}, x_mask::Mask=nothing) where T <: Real
    # I ∈ ℝ^{m,d} ~ (d, m, bs)
    # x ∈ ℝ^{n,d} ~ (d, n, bs) 
    # x_mask ∈ ℝ^{n} ~ (1, n, bs) 
    # MAB1((d, m, bs), (d, n, bs)) -> (d, m, bs)
    device = get_device(isab.I)
    I = device.ones(Float32, 1,1, size(x)[end]) .* isab.I # (d, m, 1) -> (d, m, bs) 
    # repeat(isab.I, 1, 1, size(x)[end]) ----> repeat on gpu is extremly slow!! 
    h = isab.MAB1(I, x, nothing, x_mask) # h ~ (d, m, bs)
    return x, h
end

struct VariationalBottleneck
    prior::Union{Flux.Chain, ConstGaussPrior}
    posterior::Flux.Chain
    decoder::Flux.Chain
end

Flux.@functor VariationalBottleneck

function Base.show(io::IO, m::VariationalBottleneck)
    print(io, "VariationalBottleneck(")
    print(io, "\n\t - prior = $(m.prior) \n\t - posterior = $(m.posterior)")
    print(io, "\n\t - decoder = $(m.decoder) \n ) ")
end

AbstractTrees.children(m::VariationalBottleneck) = (("Prior", m.prior), ("Posterior", m.posterior), ("Decoder", m.decoder))
AbstractTrees.printnode(io::IO, m::VariationalBottleneck) = print(io, "VariationalBottleneck - ($(size(m.decoder[1].weight, 2)) zdim)")

function (vb::VariationalBottleneck)(h::AbstractArray{T}) where T <: Real
    # computing prior μ, Σ from h
    μ, Σ = vb.prior(h)
    z = μ + Σ * randn(Float32, size(μ)...)
    ĥ = vb.decoder(z)
    return z, ĥ, nothing
end

function (vb::VariationalBottleneck)(h::AbstractArray{T}, h_enc::AbstractArray{T}) where T <: Real
    device = get_device(vb.posterior)
    # computing prior μ, Σ from h as well as posterior from h_enc
    μ, Σ = vb.prior(h)
    Δμ, ΔΣ = vb.posterior(h + h_enc)
    z = (μ + Δμ) + (Σ .* ΔΣ) .* device.randn(Float32, size(μ)...)
    ĥ = vb.decoder(z)
    kld = 0.5 * ( (Δμ.^2 ./ Σ.^2) + ΔΣ.^2 - log.(ΔΣ.^2) .- 1f0 )
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

function Base.show(io::IO, m::AttentiveBottleneckLayer)
    print(io, "AttentiveBottleneckLayer(")
    print(io, " - MAB1 = $(m.MAB1)\n - MAB2 = $(m.MAB2)")
    print(io, " - VB = $(m.VB) \n")
    print(io, " - I = $(size(m.I)) - $(typeof(m.I)) \n ) ")
end

AbstractTrees.children(m::AttentiveBottleneckLayer) = (m.MAB1, m.MAB2, m.VB, ("Induced Sets", m.I))
AbstractTrees.printnode(io::IO, m::AttentiveBottleneckLayer) = print(io, "AttentiveBottleneckLayer - ($(size(m.I,2)) Induced Sets)")

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
    device = get_device(abl.I)
    I = device.ones(Float32, 1,1, size(x)[end]) .* abl.I # (d, m, 1) -> (d, m, bs)
    #I = repeat(abl.I, 1, 1, size(x)[end]) # (d, m, 1) -> (d, m, bs)
    h = abl.MAB1(I, x) # h ~ (d, m, bs)
    z, ĥ, kld = abl.VB(h, h_enc) # z, h, kld ~ (zdim, m, bs), (d, m, bs), kld_loss (d, m, bs)
    kld = Flux.mean(Flux.sum(kld, dims=(1,2)))
    # MAB2((d, n, bs), (d, m, bs)) -> (d, n, bs)
    return abl.MAB2(x, ĥ), kld, ĥ, z # (d, n, bs), scalar, (zdim, m, bs), ...
end

function (abl::AttentiveBottleneckLayer)(x::AbstractArray{T}, h_enc::AbstractArray{T}, x_mask::Mask=nothing) where T <: Real
    # inference
    # I     ∈ ℝ^{m,d} ~ (d, m, bs)
    # x     ∈ ℝ^{n,d} ~ (d, n, bs) 
    # h_enc ∈ ℝ^{n,d} ~ (d, m, bs) 
    # MAB1((d, m, bs), (d, n, bs)) -> (d, m, bs)
    device = get_device(abl.I)
    I = device.ones(Float32, 1,1, size(x)[end]) .* abl.I # (d, m, 1) -> (d, m, bs)
    #I = repeat(abl.I, 1, 1, size(x)[end]) # (d, m, 1) -> (d, m, bs)
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

struct AttentiveHalfBlock
    MAB1::MultiheadAttentionBlock
    VB::VariationalBottleneck
end

Flux.@functor AttentiveHalfBlock

Flux.trainable(abl::AttentiveHalfBlock) = (abl.MAB1, abl.VB)

function Base.show(io::IO, m::AttentiveHalfBlock)
    print(io, "AttentiveHalfBlock(")
    print(io, " - MAB1 = $(m.MAB1)\n")
    print(io, " - VB = $(m.VB) \n ) ")
end

AbstractTrees.children(m::AttentiveHalfBlock) = (m.MAB1, m.VB)
AbstractTrees.printnode(io::IO, m::AttentiveHalfBlock) = print(io, "AttentiveHalfBlock")

function (abl::AttentiveHalfBlock)(x::AbstractArray{T}, h_enc::AbstractArray{T}, x_mask::Mask=nothing) where T <: Real
    # inference
    # I     ∈ ℝ^{m,d} ~ (d, m, bs)
    # x     ∈ ℝ^{n,d} ~ (d, n, bs) 
    # h_enc ∈ ℝ^{n,d} ~ (d, m, bs) 
    # z, h, kld ~ (zdim, m, bs), (d, m, bs), kld_loss (d, m, bs)
    device = get_device(abl.MAB1)
    h_const = device.zeros(Float32, size(h_enc)...)
    z, ĥ, kld = abl.VB(h_const, h_enc)
    #vb_const(abl.VB, h_const, h_enc, const_module=const_module) 
    kld = Flux.mean(Flux.sum(kld, dims=(1,2)))
    # MAB2((d, n, bs), (d, m, bs)) -> (d, n, bs)
    return abl.MAB1(x, ĥ, x_mask, nothing), kld, ĥ, z # (d, n, bs), scalar, (zdim, m, bs), ...
end

function AttentiveHalfBlock(
    m::Int, hidden_dim::Int, heads::Int, z_dim::Int, hidden::Int, depth::Int, activation::Function=identity
)
    mab1 = MultiheadAttentionBlock(hidden_dim, heads)

    encoder_Δμ = []
    decoder = []
    if depth>=2
        push!(encoder_Δμ, Flux.Dense(hidden_dim, hidden, activation))
        push!(decoder, Flux.Dense(z_dim, hidden, activation))
        for i=1:depth-2
            push!(encoder_Δμ, Flux.Dense(hidden, hidden, activation))
            push!(decoder, Flux.Dense(hidden, hidden, activation))
        end
        push!(encoder_Δμ, SplitLayer(hidden, (z_dim, z_dim), (identity, softplus)))
        push!(decoder, Flux.Dense(hidden, hidden_dim))
    elseif depth==1
        push!(encoder_Δμ, SplitLayer(hidden_dim, (z_dim, z_dim), (identity, softplus)))
        push!(decoder, Flux.Dense(z_dim, hidden_dim))
    else
        @error("Incorrect depth of VariationalBottleneck")
    end
    encoder_Δμ = Flux.Chain(encoder_Δμ...)
    decoder = Flux.Chain(decoder...)

    vb = VariationalBottleneck(ConstGaussPrior(m, z_dim), encoder_Δμ, decoder)

    return AttentiveHalfBlock(mab1, vb)
end