using Revise
using DrWatson
@quickactivate


using Random
using Serialization
using YAML
using GenerativeMIL
using Flux
using JLD2, JSON3
using MLUtils

using ProgressBars
using CairoMakie

include("/home/zorekmat/MIL/GenerativeMIL/experiments/VisualizationAndUnderstanding/src/pc_functions.jl")

cd("/home/zorekmat/MIL/GenerativeMIL/experiments/VisualizationAndUnderstanding/HSQVAE/")

GenerativeMIL._mnist_balanced_path() = "/home/zorekmat/MIL/GenerativeMIL/data/datasets/mnist_pc/mnist_4x_point_clouds_3x900_matrix.jls"

dict2nt(x) = (; (Symbol(k) => v for (k, v) in x)...)


function reconstruct(m::HierarchicalSlotQueryVAE, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}, N::Int=256) where T <: AbstractFloat
    dₓ, n, l, bs = size(x)
    x_reshaped = reshape(x, dₓ, n, l*bs) # (dₓ, n, l*bs)

    h = m.encoder(x_reshaped) 
    h = multiplicative_masking(reshape(h, :, 1, l, bs), x_mask) # 1 is because m_z of PMA is 1 for this version of encoder
    h = dropdims(h, dims=2) # (hidden, l, bs)
    h_mask = isnothing(x_mask) ? nothing : dropdims(x_mask, dims=2) # (1, l, bs)
    x̂, logits_exist, μ_z, Σ_z = m.deep_slot_query(h, h_mask)
    dₕ, n_slots, bs = size(x̂)

    prior = MLUtils.randn_like(x, (dₕ, N, n_slots * bs));
    x̂ = reshape(x̂, dₕ, 1, n_slots * bs) # (dₕ, 1, n_slots * bs)
    x̂ = m.decoder(prior, x̂) # (dₕ, n, n_slots * bs)
    x̂ = reshape(x̂, dₕ, N, n_slots, bs)
    x̂ = m.output(x̂)
    return x̂, logits_exist, μ_z, Σ_z
end


function load_everything(model_name, epoch_num; lpadn=4)
    cfg_name = split(model_name, "_ID")[1]

    trainlog_path = "/home/zorekmat/MIL/GenerativeMIL/data/HGenExperiments/mnist_clock/hsqvae/seed=1/$(model_name)/trainlog.jsonl"
    cfg_path = "/home/zorekmat/MIL/GenerativeMIL/experiments/MultiObjectGeneration/HSQVAE_experiments/configs/mnist_clock_configs/$(cfg_name).yml"
    model_path = "/home/zorekmat/MIL/GenerativeMIL/data/HGenExperiments/mnist_clock/hsqvae/seed=1/$(model_name)/models/model_ep=$(lpad(epoch_num, lpadn, '0')).jls"
    return model_path, cfg_path, trainlog_path, model_name, cfg_name, epoch_num
end 


model_path, cfg_path, trainlog_path, model_name, cfg_name, epoch_num = load_everything("cd_hsqvae_c006_ID-898626", 710)
model_path, cfg_path, trainlog_path, model_name, cfg_name, epoch_num = load_everything("cd_hsqvae_c010_ID-055096", 1000)
model_path, cfg_path, trainlog_path, model_name, cfg_name, epoch_num = load_everything("cd_hsqvae_c008_ID-002385", 470)
model_path, cfg_path, trainlog_path, model_name, cfg_name, epoch_num = load_everything("cd_hsqvae_c005_ID-446607", 720)


cfg = load_cfg(cfg_path)

data_cfg  = Dict{Symbol,Any}(cfg[:data]) 
model_cfg = Dict{Symbol,Any}(cfg[:model])
train_cfg = Dict{Symbol,Any}(cfg[:train])


dataloaders = create_dataloaders(batch_size=get(train_cfg, :batch_size, 16), x_only=false, data_cfg);

model = deserialize(model_path).model;

X = first(dataloaders.valid);
x, x_mask, x_label = X;
size(x)
size(x_mask)

x̂, logits_exist, _, _ = model(x, x_mask);

ℒ_rec, ℒ_exist = hungarian_matching_loss(x̂, x, x_mask, logits_exist, chamfer_pairwise_distance)

X = first(dataloaders.valid); #.train
x, x_mask, x_label = X;

x̂, logits_exist, _, _ = model(x, x_mask);
id = 3#36
sum(x_label[:, id] .>= 0)
logits_exist[:,:,id] .|> sigmoid .|> round .|> Int 
logits_exist[:,:,id] .|> sigmoid .|> round .|> Int |> sum

#x̂|>size
#plot_mnist_sample(x[:,:,1, id])

fig = plot_mnist_samples_with_exist_title(x[:,:,:,id], x̂[:,:,:,id], logits_exist[:,:,id] .|> sigmoid, x_label[:, id] ,12)

save("figures/mnist_clock/$(model_name)/$(cfg_name)_ep$(epoch_num)_id$(id).png", fig)
# c010 # id=3


ind = 3
sum(x_label[:, ind] .>= 0)
logits_exist[:,:,ind] .|> sigmoid .|> round .|> Int |> sum


fig = plot_mnist_samples_with_exist_title(x[:,:,:,ind], x̂[:,:,:,ind], logits_exist[:,:,ind] .|> sigmoid, x_label[:, ind] ,12)

save("/home/zorekmat/MIL/GenerativeMIL/experiments/VisualizationAndUnderstanding/HSQVAE/figures/mnist_clock/$(model_name)/$(cfg_name)_ep$(epoch_num)_id$(ind).png", fig)


if !ispath("/home/zorekmat/MIL/GenerativeMIL/experiments/VisualizationAndUnderstanding/HSQVAE/figures/mnist_clock/$(model_name)")
    mkpath("/home/zorekmat/MIL/GenerativeMIL/experiments/VisualizationAndUnderstanding/HSQVAE/figures/mnist_clock/$(model_name)")
end

for ind in tqdm(1:64)
    fig = plot_mnist_samples_with_exist_title(x[:,:,:,ind], x̂[:,:,:,ind], logits_exist[:,:,ind] .|> sigmoid, x_label[:, ind] ,12);
    #save("/home/zorekmat/MIL/GenerativeMIL/experiments/VisualizationAndUnderstanding/HSQVAE/figures/mnist_clock/$(model_name)/$(cfg_name)_ep$(epoch_num)_id$(ind).png", fig);
end

# 1, 2, 3, 18, 36. 37

#serialize("/home/zorekmat/MIL/GenerativeMIL/experiments/VisualizationAndUnderstanding/HSQVAE/sources/mnist_clock_$(model_name)_$(cfg_name)_ep$(epoch_num)_batch_1_valid.jls", (x = x, y = x̂, logits_exist = logits_exist, x_label = x_label))

#deserialize("/home/zorekmat/MIL/GenerativeMIL/experiments/VisualizationAndUnderstanding/HSQVAE/sources/mnist_clock_$(model_name)_$(cfg_name)_ep$(epoch_num)_batch_1_valid.jls")


x̂, logits_exist, _, _ = reconstruct(model, x, x_mask, 2048);
ind = 36

sum(x_label[:, ind] .>= 0)
logits_exist[:,:,ind] .|> sigmoid .|> round .|> Int |> sum

fig = plot_mnist_samples_with_exist_title(x[:,:,:,ind], x̂[:,:,:,ind], logits_exist[:,:,ind] .|> sigmoid, x_label[:, ind] ,12)

save("/home/zorekmat/MIL/GenerativeMIL/experiments/VisualizationAndUnderstanding/HSQVAE/figures/mnist_clock/$(model_name)/$(cfg_name)_ep$(epoch_num)_id$(ind)_N=2048.png", fig)

#serialize("/home/zorekmat/MIL/GenerativeMIL/experiments/VisualizationAndUnderstanding/HSQVAE/sources/mnist_clock_$(model_name)_$(cfg_name)_ep$(epoch_num)_batch_1_valid_N=2048.jls",(x = x, y = x̂, logits_exist = logits_exist, x_label = x_label))

history = deserialize("/home/zorekmat/MIL/GenerativeMIL/data/HGenExperiments/mnist_clock/hsqvae/seed=1/cd_hsqvae_c010_ID-055096/history.jls");

ax, fx = get(history[:ℒ]);
lines(ax[100:end], fx[100:end], color=:blue, label="ℒ")

ax, fx = get(history[:ℒ_exist]);
lines(ax, fx, color=:blue, label="ℒ_exist")
lines(ax[100:end], fx[100:end], color=:blue, label="ℒ_exist")

#tx = Float32.(reshape(fx, (length(fx), 1, 1)));
#w = Float32.(reshape(0.9.^(11:-1:1), (11, 1, 1)));
#conv(tx, w)

#fig = Figure()
#lines!(Axis(fig[1, 1]), ax[1:2:end], fx[1:2:end], color=:blue, label="ℒ")

#lines!(Axis(fig[1, 1]), ax[1+5:2:end-5], conv(tx, w; pad=5)[1+5:2:end-5, 1, 1], color=:red, label="ℒ_smooth")
#fig