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

# MultiheadAttention Block ----------------------------------------------------------------------------------------------------------

"""
    MultiheadAttentionBlock{FFT, MHT, L1, L2}

Residual multi-head attention block with feed-forward network and two layer norms.

# Fields
- `FF`: feed-forward subnetwork.
- `Multihead`: multi-head attention module.
- `LN1`: first layer norm (post-attention residual).
- `LN2`: second layer norm (post-FF residual).
"""
struct MultiheadAttentionBlock{FFT, MHT<:MultiheadAttention, L1<:Flux.LayerNorm, L2<:Flux.LayerNorm}
    FF::FFT
    Multihead::MHT
    LN1::L1
    LN2::L2
end

Flux.@layer MultiheadAttentionBlock

"""
    MultiheadAttentionBlock(hidden_dim, heads, activation=relu; attention_fn=slot_attention)

Create a `MultiheadAttentionBlock` with hidden size `hidden_dim` and `heads` attention heads.
The feed-forward branch uses two dense layers with optional `activation` in the first layer.

# Arguments
- `hidden_dim::Int`: hidden feature dimension of input/output.
- `heads::Int`: number of attention heads.
- `activation`: activation function in the first FF layer.
- `attention_fn`: attention kernel used by `MultiheadAttention`.

# Returns
- `MultiheadAttentionBlock`: initialized attention block module.
"""
function MultiheadAttentionBlock(hidden_dim::Int, heads::Int, activation=relu; attention_fn=slot_attention)
    # input_dim is equall to hidden_dim, if not there would be problem in "Q.+Multihead()"
    mh = MultiheadAttention(hidden_dim, hidden_dim, heads, attention_fn)
    ff = Flux.Chain(
        Flux.Dense(hidden_dim, hidden_dim, activation),
        Flux.Dense(hidden_dim, hidden_dim)
    )
    ln1 = Flux.LayerNorm(hidden_dim)
    ln2 = Flux.LayerNorm(hidden_dim)
    return MultiheadAttentionBlock(ff, mh, ln1, ln2)
end


"""
    (mab::MultiheadAttentionBlock)(X)

Apply self-attention to input `X` and return transformed features with residual and layer norm.

# Arguments
- `X::AbstractArray{<:AbstractFloat}`: input tensor of shape `(d, n, bs)`.

# Returns
- `AbstractArray`: transformed tensor with the same shape as `X`.
"""
function (mab::MultiheadAttentionBlock)(X::AbstractArray{T}) where T <: AbstractFloat
    # Self Attention
    # V ∈ ℝ^{n,d} ~ (d, n, bs) 
    a = mab.LN1(X + mab.Multihead(X, X, X)) # (d, n, bs) .+ (d, n, bs)
    return mab.LN2(a + mab.FF(a)) 
end

"""
    (mab::MultiheadAttentionBlock)(Q, V)

Apply cross-attention where `Q` are queries and `V` are keys/values.

# Arguments
- `Q::AbstractArray{<:AbstractFloat}`: query tensor `(d, m, bs)`.
- `V::AbstractArray{<:AbstractFloat}`: key/value tensor `(d, n, bs)`.

# Returns
- `AbstractArray`: transformed query tensor with shape `(d, m, bs)`.
"""
function (mab::MultiheadAttentionBlock)(Q::AbstractArray{T}, V::AbstractArray{T}) where T <: AbstractFloat
    # Q ∈ ℝ^{m,d} ~ (d, m, bs)
    # V ∈ ℝ^{n,d} ~ (d, n, bs) 
    a = mab.LN1(Q + mab.Multihead(Q, V, V)) # (d, m, bs) .+ (d, m, bs)
    return mab.LN2(a + mab.FF(a)) 
end

"""
    (mab::MultiheadAttentionBlock)(Q, V, Q_mask=nothing, V_mask=nothing)

Masked cross-attention variant. The output is additionally masked by `Q_mask`.

# Arguments
- `Q::AbstractArray{<:AbstractFloat}`: query tensor `(d, m, bs)`.
- `V::AbstractArray{<:AbstractFloat}`: key/value tensor `(d, n, bs)`.
- `Q_mask::Mask`: optional query mask broadcastable to query positions.
- `V_mask::Mask`: optional key/value mask broadcastable to key positions.

# Returns
- `AbstractArray`: masked transformed query tensor with shape `(d, m, bs)`.
"""
function (mab::MultiheadAttentionBlock)(Q::AbstractArray{T}, V::AbstractArray{T}, Q_mask::Mask, V_mask::Mask) where T <: AbstractFloat
    # Q ∈ ℝ^{m,d} ~ (d, m, bs)
    # V ∈ ℝ^{n,d} ~ (d, n, bs) 
    # Q_mask ∈ ℝ^{m} ~ (1, m, bs) 
    # V_mask ∈ ℝ^{n} ~ (1, n, bs) 
    a = mab.LN1(Q + mab.Multihead(Q, V, Q_mask, V_mask)) # (d, m, bs) + (d, m, bs) 
    a = mab.LN2(a + mab.FF(a)) 
    return multiplicative_masking(a, Q_mask)
end


# InducedSetAttentionBlock ----------------------------------------------------------------------------------------------------------

"""
    InducedSetAttentionBlock{M1, M2, IT}

Two-stage induced set attention block (ISAB) with learnable inducing points `I`.

# Fields
- `MAB1`: attention from inducing points to input set.
- `MAB2`: attention from input set to induced representation.
- `I`: matrix of learnable inducing points `(hidden_dim, n_slots)`.
"""
struct InducedSetAttentionBlock{M1<:MultiheadAttentionBlock, M2<:MultiheadAttentionBlock, IT<:AbstractMatrix{<:AbstractFloat}}
    MAB1::M1
    MAB2::M2
    I::IT
end

Flux.@layer InducedSetAttentionBlock

"""
    InducedSetAttentionBlock(n_slots, hidden_dim, heads; kwargs...)

Construct an ISAB with `n_slots` inducing vectors of size `hidden_dim`.
Keyword arguments are forwarded to internal `MultiheadAttentionBlock` constructors.

# Arguments
- `n_slots::Int`: number of inducing points.
- `hidden_dim::Int`: feature dimension.
- `heads::Int`: number of attention heads.
- `kwargs...`: forwarded keyword args for `MultiheadAttentionBlock`.

# Returns
- `InducedSetAttentionBlock`: initialized ISAB module.
"""
function InducedSetAttentionBlock(n_slots::Int, hidden_dim::Int, heads::Int; kwargs...)
    mab1 = MultiheadAttentionBlock(hidden_dim, heads; kwargs...)
    mab2 = MultiheadAttentionBlock(hidden_dim, heads; kwargs...)
    I = randn(Float32, hidden_dim, n_slots) # keep batch size as free parameter
    return InducedSetAttentionBlock(mab1, mab2, I)
end

"""
    (isab::InducedSetAttentionBlock)(x)

Run unmasked ISAB pass and return `(x_out, h)` where `h` are induced latent features.

# Arguments
- `x::AbstractArray{<:AbstractFloat}`: input set tensor `(d, n, bs)`.

# Returns
- `Tuple{AbstractArray, AbstractArray}`:
    - `x_out`: transformed set tensor `(d, n, bs)`.
    - `h`: induced representation `(d, m, bs)`.
"""
function (isab::InducedSetAttentionBlock)(x::AbstractArray{T}) where T <: AbstractFloat
    h = isab.MAB1(repeat(isab.I, 1, 1, size(x, ndims(x))), x)
    return isab.MAB2(x, h), h
end

"""
    (isab::InducedSetAttentionBlock)(x, x_mask=nothing)

Run masked ISAB pass and return `(x_out, h)`.
`x_mask` is applied in the internal attention operations.

# Arguments
- `x::AbstractArray{<:AbstractFloat}`: input set tensor `(d, n, bs)`.
- `x_mask::Mask`: optional mask for valid set elements.

# Returns
- `Tuple{AbstractArray, AbstractArray}`:
    - `x_out`: masked transformed set tensor `(d, n, bs)`.
    - `h`: induced representation `(d, m, bs)`.
"""
function (isab::InducedSetAttentionBlock)(x::AbstractArray{T}, x_mask::Mask) where T <: AbstractFloat
    # I ∈ ℝ^{m,d} ~ (d, m, bs)
    # x ∈ ℝ^{n,d} ~ (d, n, bs) 
    # x_mask ∈ ℝ^{n} ~ (1, n, bs) 
    # MAB1((d, m, bs), (d, n, bs), _, (1, n, bs)) -> (d, m, bs)
    I = repeat(isab.I, 1, 1, size(x, ndims(x))) # (d, m, 1) -> (d, m, bs)
    h = isab.MAB1(I, x, nothing, x_mask) # h ~ (d, m, bs)
    # MAB2((d, n, bs), (d, m, bs), (1, n, bs), _) -> (d, n, bs)
    return isab.MAB2(x, h, x_mask, nothing), h # (d, n, bs), (d, m, bs)
end


# InducedSetAttentionHalfBlock ------------------------------------------------------------------------------------------------------

"""
    InducedSetAttentionHalfBlock{M1, IT}

Single-stage induced attention block with learnable inducing points `I`.
This is a lighter alternative to full ISAB.

# Fields
- `MAB1`: attention from inducing points to input set.
- `I`: matrix of learnable inducing points `(hidden_dim, n_slots)`.
"""
struct InducedSetAttentionHalfBlock{M1<:MultiheadAttentionBlock, IT<:AbstractMatrix{<:AbstractFloat}}
    MAB1::M1
    I::IT
end

Flux.@layer InducedSetAttentionHalfBlock

"""
    InducedSetAttentionHalfBlock(n_slots, hidden_dim, heads)

Construct a half ISAB block with `n_slots` inducing vectors.

# Arguments
- `n_slots::Int`: number of inducing points.
- `hidden_dim::Int`: feature dimension.
- `heads::Int`: number of attention heads.

# Returns
- `InducedSetAttentionHalfBlock`: initialized half-ISAB module.
"""
function InducedSetAttentionHalfBlock(n_slots::Int, hidden_dim::Int, heads::Int)
    mab1 = MultiheadAttentionBlock(hidden_dim, heads)
    I = randn(Float32, hidden_dim, n_slots) # keep batch size as free parameter
    return InducedSetAttentionHalfBlock(mab1, I)
end

"""
    (isab::InducedSetAttentionHalfBlock)(x)

Run unmasked half ISAB pass and return `(x, h)` where `h` is the induced representation.

# Arguments
- `x::AbstractArray{<:AbstractFloat}`: input set tensor `(d, n, bs)`.

# Returns
- `Tuple{AbstractArray, AbstractArray}`:
    - original `x`.
    - induced representation `h` with shape `(d, m, bs)`.
"""
(isab::InducedSetAttentionHalfBlock)(x::AbstractArray{<:AbstractFloat})  = (x, isab.MAB1(repeat(isab.I, 1, 1, size(x, ndims(x))), x))

"""
    (isab::InducedSetAttentionHalfBlock)(x, x_mask=nothing)

Masked half ISAB pass returning `(x, h)`.

# Arguments
- `x::AbstractArray{<:AbstractFloat}`: input set tensor `(d, n, bs)`.
- `x_mask::Mask`: optional mask for valid set elements.

# Returns
- `Tuple{AbstractArray, AbstractArray}`:
    - original `x`.
    - masked induced representation `h` with shape `(d, m, bs)`.
"""
function (isab::InducedSetAttentionHalfBlock)(x::AbstractArray{T}, x_mask::Mask) where T <: AbstractFloat
    # I ∈ ℝ^{m,d} ~ (d, m, bs)
    # x ∈ ℝ^{n,d} ~ (d, n, bs) 
    # x_mask ∈ ℝ^{n} ~ (1, n, bs) 
    # MAB1((d, m, bs), (d, n, bs)) -> (d, m, bs)
    I = repeat(isab.I, 1, 1, size(x, ndims(x))) # (d, m, 1) -> (d, m, bs)
    h = isab.MAB1(I, x, nothing, x_mask) # h ~ (d, m, bs)
    return x, h
end


# VariationalBottleneck -------------------------------------------------------------------------------------------------------------

"""
    VariationalBottleneck{FFT, PT, DT}

Variational latent bottleneck with prior, posterior correction, and decoder networks.

# Fields
- `prior`: network mapping context to prior parameters `(μ, Σ)`.
- `posterior`: network producing posterior correction `(Δμ, ΔΣ)`.
- `decoder`: network mapping latent sample `z` back to feature space.
"""
struct VariationalBottleneck{FFT, PT, DT}
    prior::FFT#Union{Flux.Chain, ConstGaussPrior}
    posterior::PT#Flux.Chain
    decoder::DT#Flux.Chain
end

Flux.@layer VariationalBottleneck

"""
    VariationalBottleneck(in_dim, z_dim, out_dim, hidden=32, depth=1, activation=identity)

Construct a variational bottleneck MLP stack.
Returns a module with prior, posterior, and decoder subnetworks.

# Arguments
- `in_dim::Int`: input feature dimension.
- `z_dim::Int`: latent feature dimension.
- `out_dim::Int`: output feature dimension.
- `hidden::Int`: hidden layer width for `depth >= 2`.
- `depth::Int`: number of layers in each branch.
- `activation::Function`: activation used in hidden layers.

# Returns
- `VariationalBottleneck`: initialized variational module.
"""
function VariationalBottleneck(in_dim::Int, z_dim::Int, out_dim::Int, hidden::Int=32, depth::Int=1, activation::Function=identity)
    if depth < 1
        @error("Incorrect depth of VariationalBottleneck")
    end

    encoder_μ = create_gaussian_mlp(in_dim, hidden, depth, (z_dim, z_dim), activation; softplus_=true)
    encoder_Δμ = create_gaussian_mlp(in_dim, hidden, depth, (z_dim, z_dim), activation; softplus_=true)
    decoder = create_mlp(z_dim, hidden, depth, out_dim, activation; out_identity=true)

    return VariationalBottleneck(encoder_μ, encoder_Δμ, decoder)
end

"""
    (vb::VariationalBottleneck)(h)

Generation mode. Samples latent `z` from prior computed from `h` and decodes it.
Returns `(z, h_hat, nothing)`.

# Arguments
- `h::AbstractArray{<:AbstractFloat}`: prior context tensor.

# Returns
- `Tuple{AbstractArray, AbstractArray, Nothing}`:
    - `z`: sampled latent tensor.
    - `h_hat`: decoded tensor.
    - `nothing`: KL term is not used in pure generation mode.
"""
function (vb::VariationalBottleneck)(h::AbstractArray{T}) where T <: AbstractFloat
    # computing prior μ, Σ from h
    μ, Σ = vb.prior(h)
    z = μ + Σ .* MLUtils.randn_like(μ)
    ĥ = vb.decoder(z)
    return z, ĥ, nothing
end

"""
    (vb::VariationalBottleneck)(h, h_enc)

Inference mode with posterior correction from encoder features `h_enc`.
Returns `(z, h_hat, L_kl)` where `L_kl` is elementwise KL term.

# Arguments
- `h::AbstractArray{<:AbstractFloat}`: prior context tensor.
- `h_enc::AbstractArray{<:AbstractFloat}`: encoder guidance tensor.

# Returns
- `Tuple{AbstractArray, AbstractArray, AbstractArray}`:
    - `z`: sampled latent tensor.
    - `h_hat`: decoded tensor.
    - `L_kl`: elementwise KL contribution tensor.
"""
function (vb::VariationalBottleneck)(h::AbstractArray{T}, h_enc::AbstractArray{T}) where T <: AbstractFloat
    # computing prior μ, Σ from h as well as posterior from h_enc
    μ, Σ = vb.prior(h)
    Δμ, ΔΣ = vb.posterior(h + h_enc)
    z = (μ + Δμ) + (Σ .* ΔΣ) .* MLUtils.randn_like(μ)
    ĥ = vb.decoder(z)
    ℒₖₗ = 0.5 * ( (Δμ.^2 ./ Σ.^2) + ΔΣ.^2 - log.(ΔΣ.^2) .- 1f0 )
    # kld_loss = Flux.mean(Flux.sum(kld, dims=(1,2))) # mean over BatchSize , sum over Dz and Induced Set
    return z, ĥ, ℒₖₗ
end    


# AttentiveBottleneckLayer ----------------------------------------------------------------------------------------------------------

"""
    AttentiveBottleneckLayer{M1, M2, VT, IT}

Attention-based bottleneck block combining induced attention with a variational bottleneck.

# Fields
- `MAB1`: attention from inducing points to input set.
- `MAB2`: attention from input set to reconstructed induced features.
- `VB`: variational bottleneck operating in induced space.
- `I`: learnable inducing points `(hidden_dim, n_slots)`.
"""
struct AttentiveBottleneckLayer{M1<:MultiheadAttentionBlock, M2<:MultiheadAttentionBlock, VT<:VariationalBottleneck, IT<:AbstractMatrix{<:AbstractFloat}}
    MAB1::M1
    MAB2::M2
    VB::VT
    I::IT
end

Flux.@layer AttentiveBottleneckLayer

"""
    AttentiveBottleneckLayer(n_slots, hidden_dim, heads, z_dim, hidden, depth, activation=identity)

Construct an attentive bottleneck layer with learnable inducing points and variational module.

# Arguments
- `n_slots::Int`: number of inducing points.
- `hidden_dim::Int`: feature dimension in attention space.
- `heads::Int`: number of attention heads.
- `z_dim::Int`: latent dimension in variational bottleneck.
- `hidden::Int`: hidden width in bottleneck MLPs.
- `depth::Int`: depth of bottleneck MLPs.
- `activation::Function`: activation in bottleneck hidden layers.

# Returns
- `AttentiveBottleneckLayer`: initialized attentive bottleneck module.
"""
function AttentiveBottleneckLayer(n_slots::Int, hidden_dim::Int, heads::Int, z_dim::Int, hidden::Int, depth::Int, activation::Function=identity)
    mab1 = MultiheadAttentionBlock(hidden_dim, heads)
    mab2 = MultiheadAttentionBlock(hidden_dim, heads)
    I = randn(Float32, hidden_dim, n_slots) # keep batch size as free parameter
    vb = VariationalBottleneck(hidden_dim, z_dim, hidden_dim, hidden, depth, activation)
    return AttentiveBottleneckLayer(mab1, mab2, vb, I)
end

"""
    (abl::AttentiveBottleneckLayer)(x)

Generation pass. Encodes `x` into induced space, samples latent variables, decodes them,
and projects back to input set space.

Returns `(x_out, L_kl, h_hat, z)` where `L_kl` is `nothing` in generation mode.

# Arguments
- `x::AbstractArray{<:AbstractFloat}`: input set tensor `(d, n, bs)`.

# Returns
- `Tuple{AbstractArray, Nothing, AbstractArray, AbstractArray}`:
    - `x_out`: reconstructed/transformed set tensor.
    - `L_kl`: `nothing` in generation mode.
    - `h_hat`: reconstructed induced representation.
    - `z`: sampled latent representation.
"""
function (abl::AttentiveBottleneckLayer)(x::AbstractArray{T}) where T <: AbstractFloat
    # generation
    # I ∈ ℝ^{m,d} ~ (d, m, bs)
    # x ∈ ℝ^{n,d} ~ (d, n, bs) 
    # MAB1((d, m, bs), (d, n, bs)) -> (d, m, bs)
    I = repeat(abl.I, 1, 1, size(x, ndims(x)))# (d, m, 1) -> (d, m, bs)
    h = abl.MAB1(I, x) # h ~ (d, m, bs)
    z, ĥ, ℒₖₗ = abl.VB(h) # z, h, kld ~ (zdim, m, bs), (d, m, bs), kld_loss 
    # MAB2((d, n, bs), (d, m, bs)) -> (d, n, bs)
    return abl.MAB2(x, ĥ), ℒₖₗ, ĥ, z  # (d, n, bs), (d, m, bs)
end

"""
    (abl::AttentiveBottleneckLayer)(x, h_enc)

Inference pass with posterior correction from `h_enc`.
Returns `(x_out, L_kl_scalar, h_hat, z)`.

# Arguments
- `x::AbstractArray{<:AbstractFloat}`: input set tensor `(d, n, bs)`.
- `h_enc::AbstractArray{<:AbstractFloat}`: encoder guidance tensor in induced space.

# Returns
- `Tuple{AbstractArray, AbstractFloat, AbstractArray, AbstractArray}`:
    - `x_out`: transformed set tensor.
    - `L_kl_scalar`: batch-averaged KL scalar.
    - `h_hat`: reconstructed induced representation.
    - `z`: sampled latent representation.
"""
function (abl::AttentiveBottleneckLayer)(x::AbstractArray{T}, h_enc::AbstractArray{T}) where T <: AbstractFloat
    # inference
    # I     ∈ ℝ^{m,d} ~ (d, m, bs)
    # x     ∈ ℝ^{n,d} ~ (d, n, bs) 
    # h_enc ∈ ℝ^{n,d} ~ (d, m, bs) 
    # MAB1((d, m, bs), (d, n, bs)) -> (d, m, bs)
    I = repeat(abl.I, 1, 1, size(x, ndims(x))) # (d, m, 1) -> (d, m, bs)
    h = abl.MAB1(I, x) # h ~ (d, m, bs)
    z, ĥ, ℒₖₗ = abl.VB(h, h_enc) # z, h, kld ~ (zdim, m, bs), (d, m, bs), kld_loss (d, m, bs)
    ℒₖₗ = Flux.mean(Flux.sum(ℒₖₗ, dims=(1,2)))
    # MAB2((d, n, bs), (d, m, bs)) -> (d, n, bs)
    return abl.MAB2(x, ĥ), ℒₖₗ, ĥ, z # (d, n, bs), scalar, (zdim, m, bs), ...
end

"""
    (abl::AttentiveBottleneckLayer)(x, h_enc, x_mask=nothing)

Masked inference variant of attentive bottleneck layer.
Returns `(x_out, L_kl_scalar, h_hat, z)`.

# Arguments
- `x::AbstractArray{<:AbstractFloat}`: input set tensor `(d, n, bs)`.
- `h_enc::AbstractArray{<:AbstractFloat}`: encoder guidance tensor in induced space.
- `x_mask::Mask`: optional mask for valid set elements.

# Returns
- `Tuple{AbstractArray, AbstractFloat, AbstractArray, AbstractArray}`:
    - `x_out`: masked transformed set tensor.
    - `L_kl_scalar`: batch-averaged KL scalar.
    - `h_hat`: reconstructed induced representation.
    - `z`: sampled latent representation.
"""
function (abl::AttentiveBottleneckLayer)(x::AbstractArray{T}, h_enc::AbstractArray{T}, x_mask::Mask) where T <: AbstractFloat
    # inference
    # I     ∈ ℝ^{m,d} ~ (d, m, bs)
    # x     ∈ ℝ^{n,d} ~ (d, n, bs) 
    # h_enc ∈ ℝ^{n,d} ~ (d, m, bs) 
    # MAB1((d, m, bs), (d, n, bs)) -> (d, m, bs)
    I = repeat(abl.I, 1, 1, size(x, ndims(x))) # (d, m, 1) -> (d, m, bs)
    h = abl.MAB1(I, x, nothing, x_mask) # h ~ (d, m, bs)
    z, ĥ, ℒₖₗ = abl.VB(h, h_enc) # z, h, kld ~ (zdim, m, bs), (d, m, bs), kld_loss (d, m, bs)
    ℒₖₗ = Flux.mean(Flux.sum(ℒₖₗ, dims=(1,2)))
    # MAB2((d, n, bs), (d, m, bs)) -> (d, n, bs)
    return abl.MAB2(x, ĥ, x_mask, nothing), ℒₖₗ, ĥ, z # (d, n, bs), scalar, (zdim, m, bs), ...
end



"""
    AttentiveHalfBlock{MAB, VBT}

Lightweight attentive block with one attention projection and variational bottleneck.

# Fields
- `MAB1`: attention block used to project back to set space.
- `VB`: variational bottleneck with constant Gaussian prior.
"""
struct AttentiveHalfBlock{MAB<:MultiheadAttentionBlock, VBT<:VariationalBottleneck}
    MAB1::MAB
    VB::VBT
end

Flux.@layer AttentiveHalfBlock


"""
    AttentiveHalfBlock(m, hidden_dim, heads, z_dim, hidden, depth, activation=identity)

Construct an attentive half block with constant Gaussian prior over `m` latent slots.

# Arguments
- `m::Int`: number of latent slots in constant prior.
- `hidden_dim::Int`: feature dimension in attention space.
- `heads::Int`: number of attention heads.
- `z_dim::Int`: latent dimension in variational bottleneck.
- `hidden::Int`: hidden width in bottleneck MLPs.
- `depth::Int`: depth of bottleneck MLPs.
- `activation::Function`: activation in bottleneck hidden layers.

# Returns
- `AttentiveHalfBlock`: initialized attentive half-block module.
"""
function AttentiveHalfBlock(m::Int, hidden_dim::Int, heads::Int, z_dim::Int, hidden::Int, depth::Int, activation::Function=identity)
    mab1 = MultiheadAttentionBlock(hidden_dim, heads)

    if depth < 1
        @error("Incorrect depth of VariationalBottleneck")
    end

    posterior = create_gaussian_mlp(hidden_dim, hidden, depth, (z_dim, z_dim), activation; softplus_=true)
    decoder = create_mlp(z_dim, hidden, depth, hidden_dim, activation; out_identity=true)

    vb = VariationalBottleneck(ConstGaussPrior(m, z_dim), posterior, decoder)

    return AttentiveHalfBlock(mab1, vb)
end

"""
    (abl::AttentiveHalfBlock)(x, h_enc, x_mask=nothing)

Inference pass of the half block.
Samples from variational bottleneck using a zero prior context and applies attention back to `x`.

Returns `(x_out, L_kl_scalar, h_hat, z)`.

# Arguments
- `x::AbstractArray{<:AbstractFloat}`: input set tensor `(d, n, bs)`.
- `h_enc::AbstractArray{<:AbstractFloat}`: encoder guidance tensor.
- `x_mask::Mask`: optional mask for valid set elements.

# Returns
- `Tuple{AbstractArray, AbstractFloat, AbstractArray, AbstractArray}`:
    - `x_out`: transformed set tensor.
    - `L_kl_scalar`: batch-averaged KL scalar.
    - `h_hat`: reconstructed latent-conditioned representation.
    - `z`: sampled latent representation.
"""
function (abl::AttentiveHalfBlock)(x::AbstractArray{T}, h_enc::AbstractArray{T}, x_mask::Mask=nothing) where T <: AbstractFloat
    # inference
    # I     ∈ ℝ^{m,d} ~ (d, m, bs)
    # x     ∈ ℝ^{n,d} ~ (d, n, bs) 
    # h_enc ∈ ℝ^{n,d} ~ (d, m, bs) 
    # z, h, kld ~ (zdim, m, bs), (d, m, bs), kld_loss (d, m, bs)
    h_const = MLUtils.zeros_like(h_enc) 
    z, ĥ, ℒₖₗ = abl.VB(h_const, h_enc)
    #vb_const(abl.VB, h_const, h_enc, const_module=const_module) 
    ℒₖₗ = Flux.mean(Flux.sum(ℒₖₗ, dims=(1,2)))
    # MAB2((d, n, bs), (d, m, bs)) -> (d, n, bs)
    return abl.MAB1(x, ĥ, x_mask, nothing), ℒₖₗ, ĥ, z # (d, n, bs), scalar, (zdim, m, bs), ...
end