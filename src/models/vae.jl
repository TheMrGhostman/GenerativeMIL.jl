

struct VariationalAutoencoder <: AbstractGenModel
    encoder::Flux.Chain # TODO update to AbstractEncoders
    decoder::Flux.Chain # TODO update to AbstractDecoders
end

Flux.@layer VariationalAutoencoder

function (vae::VariationalAutoencoder)(x::AbstractArray{T}) where T <: Real
    μ, Σ = vae.encoder(x)
    z = μ + Σ .* MLUtils.randn_like(μ)
    x̂ = vae.decoder(z)
    return x̂
end

function elbo_with_logging(vae::VariationalAutoencoder, x::AbstractArray{T}; β::Float32=1f0, logpdf::Function=Flux.Losses.mse) where T <: Real
    μ, Σ = vae.encoder(x)
    z = μ + Σ .* MLUtils.randn_like(μ)
    x̂ = vae.decoder(z)

    ℒ_rec = logpdf(x, x̂)
    ℒ_kld = kl_divergence(μ, Σ)
    return ℒ_rec + β * ℒ_kld, (ℒ_rec = ℒ_rec, ℒ_kld = ℒ_kld, β = β) 
end

function VariationalAutoencoder(in_dim::Int, z_dim::Int, out_dim::Int; hidden::Int=32, depth::Int=1, activation::Function=identity)
    encoder = []
    decoder = []
    if depth>=2
        push!(encoder, Flux.Dense(in_dim, hidden, activation))
        push!(decoder, Flux.Dense(z_dim, hidden, activation))
        for i=1:depth-2
            push!(encoder, Flux.Dense(hidden, hidden, activation))
            push!(decoder, Flux.Dense(hidden, hidden, activation))
        end
        push!(encoder, SplitLayer(hidden, (z_dim, z_dim), (identity, softplus)))
        push!(decoder, Flux.Dense(hidden, out_dim))
    elseif depth==1
        push!(encoder, SplitLayer(in_dim, (z_dim, z_dim), (identity, softplus)))
        push!(decoder, Flux.Dense(in_dim, out_dim))
    else
        @error("Incorrect depth of VariationalAutoencoder")
    end
    encoder = Flux.Chain(encoder...)
    decoder = Flux.Chain(decoder...)
    return VariationalAutoencoder(encoder, decoder)
end



#optim_step(model, batch, opt, loss_function, device; β=β, ∇skip=grad_skip, kwargs...);
function optim_step(model::VariationalAutoencoder, batch::AbstractArray{Float32}, opt::NamedTuple, logpdf::Function, device::Function=cpu; β::Float32=1f0, kwargs...)

    # 1) move data to device
    batch = batch |> device
    # 2) compute gradients
    (loss, logs), (∇model, ∇data) = Zygote.withgradient(model, batch) do m, x
        elbo_with_logging(m, x; β = β, logpdf=logpdf)
    end
    # 3) update weights
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

#valid_step(model, dataloader, loss_function, β; device=device, kwargs...)
function valid_step(model::VariationalAutoencoder, dataloader::DataLoader, logpdf::Function; β::Float32, device::Function=cpu, kwargs...)
    ℒ, ℒ_rec, ℒ_kld = 0, 0, 0
    for batch in dataloader
        x = batch |> device
        loss, logs = elbo_with_logging(model, x; β = β, logpdf=logpdf)
        ℒ += loss
        ℒ_rec += logs.ℒ_rec
        ℒ_kld += logs.ℒ_kld
    end
    #TODO if rec or totol_loss to use for EarlyStopping figure out later.
    n = length(dataloader)
    logs = (ℒᵥ = ℒ/n, ℒᵥ_rec = ℒ_rec/n, ℒᵥ_kld = ℒ_kld/n)
    return logs, ℒ/n # total loss for early stopping
end