struct NeuralStatistician{E, S, I, LD, OD} <: AbstractGenModel
    shared_encoder::E           # shared encoder for both q(c|X) and q(z|X,c) ≈> hᵢ = shared_encoder(xᵢ) 
    statistic_net::S            # q(c|X) - outputs parameters of context distribution
    inference_net::I            # q(zᵢ|xᵢ,c) - outputs parameters of latent distribution
    latent_decoder::LD          # p(zᵢ|c) - outputs parameters of latent distribution
    observation_decoder::OD     # p(xᵢ| zᵢ, c) - outputs parameters of observation distribution
end

Flux.@layer NeuralStatistician

function (m::NeuralStatistician)(x::AbstractArray{T, 3}) where T <: AbstractFloat
    # x: (input_dim, n_points, batch_size)
    hᵢ = m.shared_encoder(x) # (hidden_dim, n_points, batch_size)

    μ_c, Σ_c = m.statistic_net(hᵢ) # (context_dim, 1, batch_size) - parameters of q(c|X)
    c = μ_c .+ Σ_c .* MLUtils.randn_like(μ_c) # (context_dim, 1, batch_size) - sampled context vector
    cᵢ = repeat(c, 1, size(x, 2), 1); # (context_dim, n_points, batch_size) - repeat context for each point

    μ_zᵢ, Σ_zᵢ = m.inference_net(hᵢ, cᵢ) # (latent_dim, n_points, batch_size) - parameters of q(zᵢ|xᵢ,c)
    
    zᵢ = μ_zᵢ .+ Σ_zᵢ .* MLUtils.randn_like(μ_zᵢ) # (latent_dim, n_points, batch_size) - sampled latent variables
    x̂ = m.observation_decoder(zᵢ, cᵢ) # (output_dim * 2, n_points, batch_size) - parameters of p(xᵢ|zᵢ,c)

    return x̂
end


function elbo_with_logging(m::NeuralStatistician, x::AbstractArray{T, 3}, logpdf::Function=Flux.mse; β₁::F=1f0, β₂::F=1f0, reconstruct::Bool=false) where {T <: AbstractFloat, F <: AbstractFloat}
    # x: (input_dim, n_points, batch_size)
    hᵢ = m.shared_encoder(x) # (hidden_dim, n_points, batch_size)

    μ_c, Σ_c = m.statistic_net(hᵢ) # (context_dim, 1, batch_size) - parameters of q(c|X)
    c = μ_c .+ Σ_c .* MLUtils.randn_like(μ_c) # (context_dim, 1, batch_size) - sampled context vector
    cᵢ = repeat(c, 1, size(x, 2), 1); # (context_dim, n_points, batch_size) - repeat context for each point

    μ_zᵢ, Σ_zᵢ = m.inference_net(hᵢ, cᵢ) # (latent_dim, n_points, batch_size) - parameters of q(zᵢ|xᵢ,c)
    μ_ẑᵢ, Σ_ẑᵢ = m.latent_decoder(c) # (latent_dim, n_points, batch_size) - parameters of p(zᵢ|c)
    
    zᵢ = μ_zᵢ .+ Σ_zᵢ .* MLUtils.randn_like(μ_zᵢ) # (latent_dim, n_points, batch_size) - sampled latent variables
    x̂ = m.observation_decoder(zᵢ, cᵢ) # (output_dim * 2, n_points, batch_size) - parameters of p(xᵢ|zᵢ,c)

    ℒᵣ = logpdf(x̂, x) # reconstruction loss
    ℒₖₗ_z =  Flux.mean(0.5f0 * sum(log.(Σ_ẑᵢ.^2) .- log.(Σ_zᵢ.^2) .+ (Σ_zᵢ.^2 .+ (μ_zᵢ .- μ_ẑᵢ).^2) ./ Σ_ẑᵢ.^2 .- 1f0, dims=1)) #kld N(μ_zᵢ, Σ_zᵢ) || N(μ_ẑᵢ, Σ_ẑᵢ)
    ℒₖₗ_c = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σ_c.^2) .- μ_c.^2  .- Σ_c.^2, dims=1)) # KL divergence for c (assuming p(c) is standard normal)
    ℒ = ℒᵣ + β₁ * ℒₖₗ_z + β₂ * ℒₖₗ_c
    if reconstruct
        return x̂, ℒ, (ℒ = ℒ, ℒ_rec = ℒᵣ, ℒₖₗ_z = ℒₖₗ_z, ℒₖₗ_c = ℒₖₗ_c, β₁ = β₁, β₂ = β₂)
    end
    return ℒ, (ℒ = ℒ, ℒ_rec = ℒᵣ, ℒₖₗ_z = ℒₖₗ_z, ℒₖₗ_c = ℒₖₗ_c, β₁ = β₁, β₂ = β₂)
end




function optim_step(model::NeuralStatistician, batch::AbstractArray{T, 3}, opt::NamedTuple, logpdf::Function, device::Function=cpu; β=1f0, kwargs...) where T <: AbstractFloat
    # 1) move data to device
    batch = batch |> device
    # 2) compute gradients
    (loss, logs), ∇model = Zygote.withgradient(model) do m
        elbo_with_logging(m, batch, logpdf; β₁ = β, β₂ = β)
    end
    # 3) update weights
    opt, model = Optimisers.update(opt, model, ∇model[1])
    return model, opt, logs
end


function reconstruct(model::NeuralStatistician, x::AbstractArray{T, 3}; kwargs...) where T <: AbstractFloat
    Flux.testmode!(model, true)
    x̂ = model(x)
    Flux.testmode!(model, false)
    return x̂
end

function reconstruct_and_log(model::SetVAE, x::AbstractArray{T}, x_mask::Mask, logpdf; β=1f0, kwargs...) <: AbstractFloat
    Flux.testmode!(model, true)
    x̂, loss, logs = elbo_with_logging(model, x, logpdf; β₁ = β, β₂ = β, reconstruct=true)
    Flux.testmode!(model, false)
    return x̂, loss, logs
end


function neuralstatistician_constructor_from_named_tuple(; idim::Int, hdim::Int, vdim::Int, cdim::Int, zdim::Int, 
    poolf::String="mean", enc_nlayers::Int=3, inner_nlayers::Int=2, activation="relu",init_seed=nothing, kwargs...)

    activation_fn = _resolve_activation(activation)
    init_seed !== nothing && Random.seed!(init_seed)

    pool, pooled_multiplier = _make_pooling(poolf, vdim, activation_fn; pool_hidden_dim=hdim)

    # shared encoder produces per-point features hᵢ
    shared_encoder = create_mlp(idim, hdim, enc_nlayers, vdim, activation_fn)

    statistic_net = Flux.Chain(
        pool,
        create_gaussian_mlp(pooled_multiplier * vdim, hdim, inner_nlayers, cdim, activation_fn; softplus_=true)
    )
    inference_net = Flux.Chain(
        x -> cat(x[1], x[2], dims=1), # concatenate hᵢ and c
        create_gaussian_mlp(hdim + cdim, hdim, inner_nlayers, zdim, activation_fn; softplus_=true)
    )
    latent_decoder = create_gaussian_mlp(cdim, hdim, inner_nlayers, zdim, activation_fn; softplus_=true)
    observation_decoder = Flux.Chain(
        x -> cat(x[1], x[2], dims=1), # concatenate zᵢ and c
        create_mlp(zdim + cdim, hdim, inner_nlayers, idim, activation_fn; out_identity=true)
    )

    return NeuralStatistician(shared_encoder, statistic_net, inference_net, latent_decoder, observation_decoder)
end