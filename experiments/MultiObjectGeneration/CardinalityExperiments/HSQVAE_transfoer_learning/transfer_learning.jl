using Revise
using DrWatson
@quickactivate

using Random
using Serialization
using YAML
using GenerativeMIL
using Flux, CUDA
using JLD2, JSON3
using MLUtils, Zygote

using ProgressBars
using Statistics

function load_cfg(path::String)
    yaml = YAML.load_file(path; dicttype=Dict{Symbol,Any})
    return Dict(Symbol(k) => v for (k, v) in yaml)
end

function relative_volume(X::AbstractArray{T,3}; mu, sigma, min_x, max_x, min_y, max_y, upsample::Int=4) where T<:AbstractFloat
    X  = (X .* (sigma .+ eps(T))) .+ mu
    nx = round(Int, (max_x - min_x) / upsample) + 1
    ny = round(Int, (max_y - min_y) / upsample) + 1
    max_volume = nx * ny

    ix = @. floor(Int, (X[1,:,:] - min_x) / upsample)
    iy = @. floor(Int, (X[2,:,:] - min_y) / upsample)
    lin = ix .* ny .+ iy

    volumes = Vector{Int}(undef, size(X, 3))
    Threads.@threads for id in 1:size(X, 3)
        volumes[id] = length(Set(@view lin[:, id]))
    end

    return Float32.(volumes ./ max_volume)
end

function hsqvae_frozen(m::HierarchicalSlotQueryVAE, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}) where T <: AbstractFloat
    dₓ, n, l, bs = size(x)
    x_reshaped = reshape(x, dₓ, n, l*bs) # (dₓ, n, l*bs)

    h = m.encoder(x_reshaped) #NOTE: I do not have to add mask here! since l became part of batch size, attention or pooling will ignore it as both are interested in first 2 dimensions. 
    h = multiplicative_masking(reshape(h, :, 1, l, bs), x_mask) # 1 is because m_z of PMA is 1 for this version of encoder
    h = dropdims(h, dims=2) # (hidden, l, bs)
    h_mask = isnothing(x_mask) ? nothing : dropdims(x_mask, dims=2) # (1, l, bs)
    ẑ, logits_exist, μ_z, Σ_z = m.deep_slot_query(h, h_mask)
    dₕ, n_slots, bs = size(ẑ)

    #frozen_dsq = copy(x̂)
    prior = MLUtils.randn_like(x, (dₕ, n, n_slots * bs));
    x̂ = reshape(ẑ, dₕ, 1, n_slots * bs) # (dₕ, 1, n_slots * bs)
    x̂ = m.decoder(prior, x̂) # (dₕ, n, n_slots * bs)
    x̂ = reshape(x̂, dₕ, n, n_slots, bs)
    x̂ = m.output(x̂)
    return x̂, logits_exist, ẑ
end

function transfer_learning(model, x, x_mask; norm_mu=Float32[39.768215; 39.768215; 0.31068918;;;], norm_sigma=Float32[23.837578; 27.632265; 0.3938166;;;])

    copy_of_x = cpu(x);
    relative_volumes = relative_volume(reshape(copy_of_x, (size(copy_of_x,1), size(copy_of_x,2), :)); mu=norm_mu, sigma=norm_sigma, min_x=1, max_x=112, min_y=1, max_y=112, upsample=4);
    
    rv = reshape(relative_volumes, (1, size(copy_of_x,3), size(copy_of_x,4))) .* cpu(x_mask[1,:,:,:])

    x̂, logits_exist, ẑ = hsqvae_frozen(model, x, x_mask)
    C = chamfer_pairwise_distance(x̂, x); # (M, L, BS)
    matched_indices, exist_target = hungarian_match(C, x_mask)
    return x̂, logits_exist, ẑ, matched_indices, exist_target, rv
end

function create_transfered_dataset(model, dataloader)
    ẑs = []
    ids = []
    ys = []
    ρ = []
    for (x, x_mask, y) in dataloader
        _, _, ẑ, matched_indices, _, rv = transfer_learning(model, x, x_mask)
        push!(ẑs, cpu(ẑ))
        push!(ids, matched_indices)
        push!(ys, cpu(y))
        push!(ρ, rv)
    end
    return (ẑ = ẑs, id = ids, y=ys, ρ = ρ)
end

minmax_scaler(x; min_x, max_x) = (x .- min_x) ./ (max_x - min_x)
mms(x) = minmax_scaler(x; min_x=0.09988f0, max_x=0.46016f0)

##########

cd("/home/zorekmat/MIL/GenerativeMIL/experiments/MultiObjectGeneration/CardinalityExperiments/HSQVAE_transfoer_learning")

GenerativeMIL._mnist_balanced_path() = "/home/zorekmat/MIL/GenerativeMIL/data/datasets/mnist_pc/mnist_4x_point_clouds_3x900_matrix.jls"

dict2nt(x) = (; (Symbol(k) => v for (k, v) in x)...)


function load_everything(model_name, epoch_num; lpadn=4)
    cfg_name = split(model_name, "_ID")[1]

    trainlog_path = "/home/zorekmat/MIL/GenerativeMIL/data/HGenExperiments/mnist_clock/hsqvae/seed=1/$(model_name)/trainlog.jsonl"
    cfg_path = "/home/zorekmat/MIL/GenerativeMIL/experiments/MultiObjectGeneration/HSQVAE_experiments/configs/mnist_clock_configs/$(cfg_name).yml"
    model_path = "/home/zorekmat/MIL/GenerativeMIL/data/HGenExperiments/mnist_clock/hsqvae/seed=1/$(model_name)/models/model_ep=$(lpad(epoch_num, lpadn, '0')).jls"
    return model_path, cfg_path, trainlog_path, model_name, cfg_name, epoch_num
end 

model_path, cfg_path, trainlog_path, model_name, cfg_name, epoch_num = load_everything("cd_hsqvae_c010_ID-055096", 1000)


cfg = load_cfg(cfg_path)

data_cfg  = Dict{Symbol,Any}(cfg[:data]) 
model_cfg = Dict{Symbol,Any}(cfg[:model])
train_cfg = Dict{Symbol,Any}(cfg[:train])

norm_mu = Float32[39.768215; 39.768215; 0.31068918;;;]
norm_sigma = Float32[23.837578; 27.632265; 0.3938166;;;]

dataloaders = create_dataloaders(batch_size=get(train_cfg, :batch_size, 16), x_only=false, data_cfg);
model = deserialize(model_path).model;


x, x_mask, y = first(dataloaders.train)



relative_volumes = relative_volume(reshape(x, (size(x,1), size(x,2), :)); mu=norm_mu, sigma=norm_sigma, min_x=1, max_x=112, min_y=1, max_y=112, upsample=4);
rv = reshape(relative_volumes, (1, size(x,3), size(x,4))) .* x_mask[1,:,:,:]

x̂, exist_logits, frozen_dsq = hsqvae_frozen(model, x, x_mask)
card_head = create_mlp(model_cfg[:dsq_emb_dim], 64, 3, 1, gelu; out_identity=true)
card_head(frozen_dsq)

C = chamfer_pairwise_distance(x̂, x); # (M, L, BS)
matched_indices, exist_target = Zygote.@ignore hungarian_match(C, x_mask);

pwd = GenerativeMIL._pairwise_sqdist_batched(card_head(frozen_dsq), rv .* 10);
pwd[matched_indices] |> mean


# functional 
_, _, frozen_dsq, matched_indices, _, rv = transfer_learning(model, x, x_mask);
N̂ = card_head(frozen_dsq)
pwd = GenerativeMIL._pairwise_sqdist_batched(N̂, mms(rv));
pwd[matched_indices] |> mean



# we can start from embedding of whole sample 
# save μ, Σ, output from DSQ, matching indices from hungarian match. 
# process whole dataset and save values from above. 
# then train model on those values. 
# 1) naive transfer learning; train just cardinality head on frozen DSQ output.
function loss_f(model, z, ids, ρ)
    N̂ = model(z)
    pwd = GenerativeMIL._pairwise_sqdist_batched(N̂, mms(ρ) .* 10);
    loss = mean(pwd[ids])
    return loss
end

model_gpu = cu(model);
frozen_dataset = create_transfered_dataset(model_gpu, tqdm(CuIterator(dataloaders.train)));

card_head = create_mlp(model_cfg[:dsq_emb_dim], 64, 3, 1, gelu; out_identity=true)
opt = Optimisers.setup(AdamW(), card_head);

epochs = 10
logs = []
eid = []
for e in 1:epochs
    frozen_dataset = create_transfered_dataset(model_gpu, tqdm(CuIterator(dataloaders.train)));
    for i in tqdm(axes(frozen_dataset.ẑ,1))
        z = frozen_dataset.ẑ[i];
        id = frozen_dataset.id[i];
        ρ = frozen_dataset.ρ[i];
        loss, ∇model =  Zygote.withgradient(card_head) do m
            loss_f(m, z, id, ρ)
        end
        opt, card_head = Optimisers.update(opt, card_head, ∇model[1])
        push!(logs, loss)
        push!(eid, e)
    end
end

frozen_dataset_valid = create_transfered_dataset(model_gpu, tqdm(CuIterator(dataloaders.valid)));
valid_logs = []
for i in tqdm(axes(frozen_dataset_valid.ẑ,1))
    z = frozen_dataset_valid.ẑ[i];
    id = frozen_dataset_valid.id[i];
    ρ = frozen_dataset_valid.ρ[i];
    loss = loss_f(card_head, z, id, ρ)
    push!(valid_logs, loss)
end


inverse_mms(x; min_x, max_x, scale=10f0) = x ./ scale .* (max_x - min_x) .+ min_x

# Relative (%) and absolute-point cardinality error over matched slots, gathered directly
# from card_head predictions and rho targets (not via the squared-distance matrix), since
# relative error needs the paired values themselves rather than (pred-target)^2.
function cardinality_errors(card_head, dataset; min_x=0.09988f0, max_x=0.46016f0, scale=10f0, max_points=256)
    rel_errs = Float32[]
    abs_pt_errs = Float32[]
    pts_per_density = max_points / max_x

    for i in axes(dataset.ẑ, 1)
        ids = dataset.id[i]
        isempty(ids) && continue
        N̂ = card_head(dataset.ẑ[i]) # (1, M, BS)
        rho = dataset.ρ[i]          # (1, L, BS)

        pred_idx = CartesianIndex.(1, getindex.(ids, 1), getindex.(ids, 3))
        targ_idx = CartesianIndex.(1, getindex.(ids, 2), getindex.(ids, 3))

        pred_rv = inverse_mms(N̂[pred_idx]; min_x=min_x, max_x=max_x, scale=scale)
        actual_rv = rho[targ_idx]

        append!(rel_errs, abs.(pred_rv .- actual_rv) ./ actual_rv .* 100)
        append!(abs_pt_errs, abs.(pred_rv .- actual_rv) .* pts_per_density)
    end

    return (
        mean_rel_error_pct   = mean(rel_errs),
        median_rel_error_pct = median(rel_errs),
        mean_abs_point_error = mean(abs_pt_errs),
        rel_errors           = rel_errs,
    )
end

c_errors = cardinality_errors(card_head, frozen_dataset_valid)


# 2) unfreeze DSQ and train both DSQ and cardinality head. (tiny tiny learning rate for DSQ, larger learning rate for cardinality head).
