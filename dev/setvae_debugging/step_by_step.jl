using Revise
using Random
using Serialization
using YAML
using Flux
using MLUtils
using Zygote
using GenerativeMIL

using CairoMakie

pth = joinpath(@__DIR__)
include(joinpath(pth, "functions.jl"))

N = 512
data = load_2d_mnist(N, true);
#plot_mnist_samples(data, 6)


cfg = load_cfg(joinpath(pth, "default_confg.yaml"))
model_cfg = Dict{Symbol,Any}(cfg[:model])
model_cfg[:activation] = resolve_activation(model_cfg[:activation])
model_cfg[:output_activation] = resolve_activation(get(model_cfg, :output_activation, "identity"))
model = setvae_constructor_from_named_tuple(; idim=2, dict2nt(model_cfg)...);


# encoder
model.encoder.expansion
model.encoder.layers
isab1 = model.encoder.layers[1]
isab2 = model.encoder.layers[2]
isab3 = model.encoder.layers[3]
isab4 = model.encoder.layers[4]
isab5 = model.encoder.layers[5]
isab6 = model.encoder.layers[6]
isab7 = model.encoder.layers[7] # halfblock
# prior
model.prior
# decoder
model.decoder.expansion # from prior to decoder hidden dimension
abl1 = model.decoder.layers[1] # attentive bottleneck layer # halfblock; no induced points as they would not be updated. 
abl2 = model.decoder.layers[2] # attentive bottleneck layer 
abl3 = model.decoder.layers[3] # attentive bottleneck layer 
abl4 = model.decoder.layers[4] # attentive bottleneck layer
abl5 = model.decoder.layers[5] # attentive bottleneck layer
abl6 = model.decoder.layers[6] # attentive bottleneck layer
abl7 = model.decoder.layers[7] # attentive bottleneck layer # 
model.decoder.reduction

#mab = abl1.MAB1
#vbc = abl1.VB

copy_prior = deepcopy(model.prior);
copy_vb = deepcopy(abl1.VB);


lvlone = (
    isabhalf=isab7, 
    ablhalf=abl1, 
    expansion=model.encoder.expansion, 
    reduction=model.decoder.reduction, 
    prior=model.prior, 
    decoder_expansion=model.decoder.expansion
)

function forward(lvl, x)
    x = lvl.expansion(x)
    _, h_enc = lvl.isabhalf(x)
    _, sample_size, bs = size(x)
    z₀ = lvl.prior(sample_size, bs) # (Ds, ss, bs)
    z₀ = lvl.decoder_expansion(z₀) # (d_dec, ss, bs)
    x̂, kld, ĥ, z = lvl.ablhalf(z₀, h_enc) # (Ds, ss, bs)
    x̂ = lvl.reduction(x̂) # (idim, ss, bs)
    return x̂, kld
    #return z₀, h_enc
end

x = data[:,:,1:64];
forward(lvlone, x) .|>size

function elbol(model, x; logpdf)
    x̂, kld = forward(model, x)
    ℒᵣ = logpdf(x̂, x) # reconstruction loss
    ℒ = ℒᵣ + 0.01 * kld
    return ℒ, (ℒ = ℒ, ℒ_rec = ℒᵣ, KL = kld)
end

function opt_stp(model, batch, opt; logpdf=chamfer_distance, device=cpu, verbose::Bool=false)
    batch = batch |> device
    (loss, logs), ∇model = Zygote.withgradient(model) do m
        elbol(m, batch; logpdf=logpdf)
    end
    (verbose) && @show logs
    opt, model = Optimisers.update(opt, model, ∇model[1])
    return model, opt, logs
end

opt = Optimisers.setup(Adam(0.003), lvlone);
for i in 1:100
    lvlone, opt, logs = opt_stp(lvlone, x, opt; logpdf=chamfer_distance, verbose=true);
end


x̂ = forward(lvlone, x)[1];
plot_mnist_samples(x, x̂, 4)


