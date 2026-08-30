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
using CUDA

using ProgressBars
using CairoMakie
using StatsBase


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

function process_dataset(model, dataset)
    xs = []
    xhats = []
    logits = []
    ys = []
    for (x, x_mask, y) in dataset
        x̂, logits_exist, _, _ = model(x, x_mask);
        push!(xs, x|>cpu)
        push!(xhats, x̂|>cpu)
        push!(logits, logits_exist|>cpu)
        push!(ys, y|>cpu)
    end
    return (xs=xs, xhats=xhats, logits=logits, ys=ys)
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
model_path, cfg_path, trainlog_path, model_name, cfg_name, epoch_num = load_everything("cd_hsqvae_c110_ID-369875", 1000)
model_path, cfg_path, trainlog_path, model_name, cfg_name, epoch_num = load_everything("cd_hsqvae_c102_ID-475284", 1000)



cfg = load_cfg(cfg_path)

data_cfg  = Dict{Symbol,Any}(cfg[:data]) 
model_cfg = Dict{Symbol,Any}(cfg[:model])
train_cfg = Dict{Symbol,Any}(cfg[:train])


dataloaders = create_dataloaders(batch_size=get(train_cfg, :batch_size, 16), x_only=false, data_cfg);


model = deserialize(model_path).model;
model_gpu = cu(model);
val = process_dataset(model_gpu, CuIterator(dataloaders.valid));

xs = cat(val.xs..., dims=4);
xhats = cat(val.xhats..., dims=4);
logits = cat(val.logits..., dims=3);
ys = cat(val.ys..., dims=2);

predicted_cardinalities = dropdims(sum(Int.(round.(sigmoid.(logits))), dims=2), dims=(1,2)) ;
actual_cardinalities = dropdims(sum(Int.(ys .>= 0), dims=1), dims=(1));

countmap(abs.(predicted_cardinalities .- actual_cardinalities))

correct_cardinalities_idx = findall(==(0),abs.(predicted_cardinalities .- actual_cardinalities))

to_save = (
    xs=xs, 
    xhats=xhats, 
    logits=logits, 
    ys=ys, 
    predicted_cardinalities=predicted_cardinalities,
    actual_cardinalities=actual_cardinalities,
    correct_cardinalities_idx=correct_cardinalities_idx
);

serialize("sources/predictions_$(model_name)_ep$(epoch_num).jls", to_save)