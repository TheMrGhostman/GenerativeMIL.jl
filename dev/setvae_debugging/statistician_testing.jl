T = Float32

ns = neuralstatistician_constructor_from_namedtuple(idim=3, hdim=32, vdim=32, cdim=64, zdim=16, poolf = "mean", enc_nlayers=3, inner_nlayers = 2)
m = ns

x = rand(T, 3, 512, 128); # (idim, n_points, batch_size)

# x: (input_dim, n_points, batch_size)
hᵢ = m.shared_encoder(x); # (hidden_dim, n_points, batch_size)
hᵢ |> size


μ_c, Σ_c = m.statistic_net(hᵢ); # (context_dim, 1, batch_size) - parameters of q(c|X)
μ_c |> size
Σ_c |> size


c = μ_c .+ Σ_c .* MLUtils.randn_like(μ_c); # (context_dim, 1, batch_size) - sampled context vector
c |> size

cᵢ = repeat(c, 1, size(x, 2), 1); # (context_dim, n_points, batch_size) - repeat context for each point
cᵢ |> size

μ_zᵢ, Σ_zᵢ = m.inference_net(hᵢ, cᵢ); # (latent_dim, n_points, batch_size) - parameters of q(zᵢ|xᵢ,c)
μ_zᵢ |> size
Σ_zᵢ |> size

μ_ẑᵢ, Σ_ẑᵢ = m.latent_decoder(c); # (latent_dim, n_points, batch_size) - parameters of p(zᵢ|c)
μ_ẑᵢ |> size
Σ_ẑᵢ |> size

zᵢ = μ_zᵢ .+ Σ_zᵢ .* MLUtils.randn_like(μ_zᵢ); # (latent_dim, n_points, batch_size) - sampled latent variables
zᵢ |> size

x̂ = m.observation_decoder(zᵢ, cᵢ); # (output_dim * 2, n_points, batch_size) - parameters of p(xᵢ|zᵢ,c)
x̂ |> size

logpdf=Flux.mse
ℒᵣ = logpdf(x̂, x) # reconstruction loss

β₁, β₂ = 0.1f0, 0.1f0

ℒₖₗ_z =  Flux.mean(0.5f0 * sum(log.(Σ_ẑᵢ.^2) .- log.(Σ_zᵢ.^2) .+ (Σ_zᵢ.^2 .+ (μ_zᵢ .- μ_ẑᵢ).^2) ./ Σ_ẑᵢ.^2 .- 1f0, dims=1)) #kld N(μ_zᵢ, Σ_zᵢ) || N(μ_ẑᵢ, Σ_ẑᵢ)
ℒₖₗ_c =  - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σ_c.^2) .- μ_c.^2  .- Σ_c.^2, dims=1)) # KL divergence for c (assuming p(c) is standard normal)
ℒ = ℒᵣ + β₁ * ℒₖₗ_z + β₂ * ℒₖₗ_c


elbo_with_logging(ns, x)
elbo_with_logging(ns, x; β₁=0.1f0, β₂=0.1f0, logpdf=Flux.mse)


optimiser = Optimisers.Adam()
opt = Optimisers.setup(optimiser, ns)

(loss, logs), (∇model, ∇data) = Zygote.withgradient(ns, x) do mm, batch
    elbo_with_logging(mm, batch; logpdf=logpdf)
end

opt, ns = Optimisers.update(opt, ns, ∇model)

function step_w_data(model, data, opt)
    (loss, logs), (∇model, ∇data) = Zygote.withgradient(model, data) do mm, batch
        elbo_with_logging(mm, batch; logpdf=Flux.mse)
    end

    opt, model = Optimisers.update(opt, model, ∇model)
end

function step_wo_data(model, data, opt)
    (loss, logs), ∇model = Zygote.withgradient(model) do mm
        elbo_with_logging(mm, data; logpdf=Flux.mse)
    end

    opt, model = Optimisers.update(opt, model, ∇model[1])
end


using BenchmarkTools
ns1 = deepcopy(ns)
ns2 = deepcopy(ns)
opt1 = Optimisers.setup(optimiser, ns1);
opt2 = Optimisers.setup(optimiser, ns2);

@benchmark step_w_data($ns1, $x, $opt1)
@benchmark step_wo_data($ns2, $x, $opt2)