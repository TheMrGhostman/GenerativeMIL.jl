using Revise
using Random
using Serialization
using Flux
using MLUtils
using Statistics
using StatsBase

T = Float32

reconstruct_and_log(m, x, mask, loss; β =1) = (x .+ randn(Float32, size(x)) .* 0.1f0, 1, (ℒ= 0, ℒ_rec = 0, ℒₖₗ = [1, 2, 3], β = β))

lf(x,y, args...) = vec(sum(Flux.Losses.mse(x,y; agg=identity), dims=(1,2)))

device = identity
dl = MLUtils.DataLoader(randn(T, 3, 16, 128), batchsize=8);

loss_functions = Dict(:l2 => lf, :ch => lf)

batch = first(dl);
β = 1f0
idx = 1
dataloader = dl
model(x) = MLUtils.randn_like(x)

xhat_batch
base_logs

(name, fn) = first(loss_fns)


using GenerativeMIL, MLUtils, YAML, ProgressBars
using GenerativeMIL: CUDA, OptimalTransport, Zygote
using GenerativeMIL: _pairwise_sqdist_batched, _nearest_neighbors, compute_transport_plans, _contributions, device_like

df = deserialize("/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/mnist/poolmodel/seed=1/cd_poolmodel_c001_ID-462882/models/model_ep=2000.jls") #FIXME: rerun models

df.model

function load_cfg(path::String)
    yaml = YAML.load_file(path; dicttype=Dict{Symbol,Any})
    return Dict(Symbol(k) => v for (k, v) in yaml)
end

configs = load_cfg("/home/zorekmat/MIL/GenerativeMIL/experiments/GenerationExperiments/PoolModel_experiments/configs/mnist_configs/cd_poolmodel_c001.yml")

train_cfg = configs[:train]
data_cfg = configs[:data]

dataloaders = create_dataloaders(batch_size=get(train_cfg, :batch_size, 16), x_only=true, data_cfg)

loss_functions = Dict{Symbol, Function}(
    :chamfer => (x,y,args...) -> chamfer_distance_eval(x,y),
    :sh      => (x,y,args...) -> sinkhorn_divergence_loss_eval(x,y,1f0),
    :dcd     => (x,y,args...) -> density_aware_chamfer_distance_eval(x,y,1f0),
    :mmd     => (x,y,args...) -> maximum_mean_discrepancy_rbf_eval(x,y; sigma=1.32f0 .* [0.25, 0.5, 1.0]),
)



o = reconstruction_eval(df.model, dataloaders.test, loss_functions; idx=1, device=cpu, verbose=true)
o = reconstruction_eval_repeated(df.model, dataloaders.test, loss_functions, 3; verbose=true)


cu_model = df.model |> cu;
o = reconstruction_eval(cu_model, dataloaders.test, loss_functions; idx=1, device=cu, verbose=true)
o = reconstruction_eval_repeated(cu_model, dataloaders.test, loss_functions, 3; device=cu, verbose=true)


configs1 = load_cfg("/home/zorekmat/MIL/GenerativeMIL/experiments/GenerationExperiments/PoolModel_experiments/configs/mnist_configs/mmd_poolmodel_c003.yml")

train_cfg1 = configs1[:train]
data_cfg1 = configs1[:data]
dataloaders1 = create_dataloaders(batch_size=get(train_cfg1, :batch_size, 16), x_only=true, data_cfg1);
dataloaders == dataloaders1
dataloaders.test == dataloaders1.test
tst0 = cat([x for x in tqdm(dataloaders.test)]..., dims=3);
tst1 = cat([x for x in tqdm(dataloaders1.test)]..., dims=3);
tst0 ≈ tst1

