# Latent density models fitted AFTER training on set encodings ("ex-post density estimation",
# Ghosh et al. 2020). Used to sample from models without a usable prior (PoolModel) and, later,
# to compare a fitted density with a model's own prior. Generic: works on any (d, N) matrix of encodings.

abstract type AbstractLatentDensity end

"""
`LatentGaussian`

Full-covariance Gaussian `N(μ, L Lᵀ)` over latent vectors (columns). Stored on CPU in Float64.
"""
struct LatentGaussian <: AbstractLatentDensity
    μ::Vector{Float64}
    L::Matrix{Float64}      # lower Cholesky factor of the covariance
    jitter::Float64         # ridge added to the covariance diagonal before Cholesky
end

"""
`fit_latent_gaussian(H::AbstractMatrix; rel_jitter=1e-6)`

Maximum-likelihood Gaussian fit to encodings `H` of shape `(d, N)` (one encoding per column).
A small ridge `rel_jitter * tr(C)/d` is added to the covariance so the Cholesky factorisation
is stable even if some latent dimensions are (almost) constant.
"""
function fit_latent_gaussian(H::AbstractMatrix; rel_jitter::Real=1e-6)
    H = Float64.(Array(H))
    d, N = size(H)
    N > d || @warn "fewer encodings than dimensions (N=$N ≤ d=$d): covariance is rank deficient, ridge dominates"
    μ = vec(mean(H, dims=2))
    C = cov(H, dims=2, corrected=false)     # ML estimate
    jitter = rel_jitter * tr(C) / d
    L = cholesky(Symmetric(C + jitter * I)).L
    return LatentGaussian(μ, Matrix(L), jitter)
end

"""
`Distributions.logpdf(g::LatentGaussian, H::AbstractMatrix)`

Log-density of every column of `H` `(d, N)`. Returns a `Vector{Float64}` of length `N`.
Used to compare density models on VALID encodings (higher mean log-likelihood = better fit).
"""
function Distributions.logpdf(g::LatentGaussian, H::AbstractMatrix)
    d = length(g.μ)
    U = LowerTriangular(g.L) \ (Float64.(Array(H)) .- g.μ)    # whitened residuals
    logdet = 2 * sum(log, diag(g.L))
    return vec(-0.5 .* (d * log(2π) + logdet .+ sum(abs2, U, dims=1)))
end

"""
`sample_latent(rng, g::LatentGaussian, n::Int)`

Draw `n` latent vectors, returned as a `Float32` matrix `(d, n)` on CPU.
"""
sample_latent(rng::AbstractRNG, g::LatentGaussian, n::Int) = Float32.(g.μ .+ g.L * randn(rng, length(g.μ), n))

"""
`collect_encodings(model, dataloader; device=cpu)`

Encode every set of `dataloader` with `encode_set(model, x)` and return all encodings as a
`Float32` matrix `(d, N)` on CPU (the singleton set dimension of `(d, 1, bs)` is dropped).
The dataloader must yield plain arrays `(idim, n, bs)` (i.e. created with `x_only=true`).
"""
function collect_encodings(model, dataloader; device::Function=cpu)
    model = model |> device
    hs = [Array(reshape(encode_set(model, x |> device), :, size(x, 3))) for x in dataloader]
    return reduce(hcat, hs)
end

"""
`generate_samples(model::PoolModel, density::AbstractLatentDensity, cardinality::Int, num_samples::Int=10; rng=Random.default_rng(), device=cpu)`

Generate new sets with PoolModel: sample set encodings `h` from a density fitted to the TRAIN
encodings, then decode them with `decode_from_encoding` (same generator + decoder as reconstruction).

Returns:
- `x̂`: `(idim, cardinality, num_samples)`.
- `(h, μ, Σ, z)`: sampled encodings `(d, 1, num_samples)`, generator parameters and per-point latents.
"""
function generate_samples(model::PoolModel, density::AbstractLatentDensity, cardinality::Int, num_samples::Int=10;
                          rng::AbstractRNG=Random.default_rng(), device::Function=cpu)
    h = reshape(sample_latent(rng, density, num_samples), :, 1, num_samples) |> device
    x̂, (μ, Σ, z) = decode_from_encoding(model, h, cardinality)
    return x̂, (h, μ, Σ, z)
end


# ─────────────────────────────── stage-2 latent VAE ───────────────────────────────

"""
`LatentVAE`

Small MLP VAE on (standardised) latent vectors, used as a learned density over encodings
("two-stage VAE", Dai & Wipf 2019). Generative model: u ~ N(0, I), h = dec(u) + γ ε with a LEARNED
scalar decoder std γ (Gaussian likelihood), so the reconstruction/KL balance is not an arbitrary
hyperparameter. Inputs are standardised with the train statistics `μₕ, σₕ` (stored in the struct).
"""
struct LatentVAE{E, D, V<:AbstractVector{Float32}, S<:AbstractVector{Float32}} <: AbstractLatentDensity
    encoder::E          # h -> (μᵤ, σᵤ)
    decoder::D          # u -> ĥ (standardised space)
    logγ::V             # length-1 vector: log of the decoder std (trainable)
    μₕ::S               # standardisation (train encodings), not trained; moves with the model to GPU
    σₕ::S
end

Flux.@layer LatentVAE trainable=(encoder, decoder, logγ)

"""
`LatentVAE(hdim_in, zdim; hdim=128, depth=2, activation=Flux.swish, μₕ, σₕ)`

`depth` = number of Dense layers in the encoder and in the decoder (the encoder's last layer is the
Gaussian split layer).
"""
function LatentVAE(hdim_in::Int, zdim::Int; hdim::Int=128, depth::Int=2, activation=Flux.swish,
                   μₕ::AbstractVector, σₕ::AbstractVector)
    depth >= 1 || error("depth must be ≥ 1")
    encoder = create_gaussian_mlp(hdim_in, hdim, depth, zdim, activation; softplus_=true)
    decoder = create_mlp(zdim, hdim, depth, hdim_in, activation; out_identity=true)
    return LatentVAE(encoder, decoder, [0f0], Float32.(collect(μₕ)), Float32.(collect(σₕ)))
end

standardise(m::LatentVAE, H) = (H .- m.μₕ) ./ m.σₕ
unstandardise(m::LatentVAE, Z) = Z .* m.σₕ .+ m.μₕ

"""
`latent_vae_loss(m::LatentVAE, H; γ_min=1f-3)`

Negative ELBO in nats per encoding (mean over the batch), for raw encodings `H (d, bs)`.
Returns `(loss, (rec, kl, γ))`. `rec` includes the Gaussian normalisation (d·log γ + d/2·log 2π), so the loss
is a proper bound on -log p(h) in the STANDARDISED space.
"""
function latent_vae_loss(m::LatentVAE, H::AbstractMatrix{T}; γ_min::T=T(1f-3)) where T <: AbstractFloat
    X = standardise(m, H)
    μᵤ, σᵤ = m.encoder(X)
    u = μᵤ .+ σᵤ .* MLUtils.randn_like(μᵤ)
    X̂ = m.decoder(u)
    γ = max(exp(m.logγ[1]), γ_min)
    d = size(X, 1)
    rec = mean(sum((X .- X̂) .^ 2, dims=1)) / (2γ^2) + d * log(γ) + T(0.5) * d * T(log(2π))
    kl = mean(T(0.5) .* sum(μᵤ .^ 2 .+ σᵤ .^ 2 .- 1 .- 2 .* log.(σᵤ), dims=1))
    return rec + kl, (rec = rec, kl = kl, γ = γ)
end

"""
`elbo_raw(m::LatentVAE, H)`

ELBO lower bound of log p(h) per encoding in the RAW encoding space (adds the standardisation Jacobian
-Σ log σₕ), so it is directly comparable with `logpdf(::LatentGaussian, H)` on the same encodings.
"""
function elbo_raw(m::LatentVAE, H::AbstractMatrix)
    loss, _ = latent_vae_loss(m, H)
    return -loss - sum(log.(m.σₕ))
end

"""
`sample_latent(rng, m::LatentVAE, n)`

u ~ N(0, I) -> decoder MEAN -> unstandardise (the decoder noise γ ε is not added, as usual for two-stage VAEs).
Runs on CPU; returns Float32 `(d, n)`.
"""
function sample_latent(rng::AbstractRNG, m::LatentVAE, n::Int)
    m = m |> cpu
    zdim = size(m.decoder.layers[1].weight, 2)
    return unstandardise(m, m.decoder(randn(rng, Float32, zdim, n)))
end
