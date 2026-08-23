using Revise
using DrWatson
@quickactivate

using Random
using Statistics
using JLD2
using MLUtils
using ProgressBars
using Flux
using Zygote
using Optimisers
using Hungarian
using CUDA

using GenerativeMIL
using GenerativeMIL: _nearest_neighbors, _pairwise_sqdist_batched
#import GenerativeMIL: elbo_with_logging, optim_step, valid_step

using BenchmarkTools

function build_test_batch(D, N, L, M, BS)
    # D: data dimension
    # N: number of points per bag
    # L: number of bags
    # M: number of slots
    # BS: batch size

    X = randn(Float32, D, N, L, BS)
    Y = randn(Float32, D, N, M, BS)
    mask = falses(1, 1, L, BS)
    for b in 1:BS
        n_clusters = rand(1:L)
        @show n_clusters
        mask[1, 1, 1:n_clusters, b] .= true
    end
    X = X .* mask
    return X, Y, mask
end

## make data
function make_cpu_gpu_batch(D, N, L, M, BS)
    X, Y, mask = build_test_batch(D, N, L, M, BS)
    X_gpu = cu(X)
    Y_gpu = cu(Y)
    mask_gpu = cu(mask)
    return (X, Y, mask), (X_gpu, Y_gpu, mask_gpu)
end

function make_cpu_gpu_batch_for_model(D, N, L, BS)
    X, _, mask = build_test_batch(D, N, L, 1, BS)
    X_gpu = cu(X)
    mask_gpu = cu(mask)
    return (X, mask), (X_gpu, mask_gpu)
end


function model_elbo(model, x, x_mask)
    ŷ, logits_exist, μ_z, Σ_z = model(x, x_mask)
    ℒ_rec, ℒ_exist = hungarian_matching_loss(ŷ, x, x_mask, logits_exist)
    ℒ_kl = mean(GenerativeMIL.kl_divergence(μ_z, Σ_z))
    return ℒ_rec + ℒ_exist + ℒ_kl
end

model_cpu = build_hierarchical_slot_query_vae(3, 12, 2);
model_gpu = cu(model_cpu);

compute_grad_cpu() = Zygote.gradient(m -> model_elbo(m, x, x_mask), model_cpu)
compute_grad_gpu() = CUDA.@sync Zygote.gradient(m -> model_elbo(m, xc, xc_mask), model_gpu)

## gpu testing of backward of model
(x, x_mask), (xc, xc_mask) = make_cpu_gpu_batch_for_model(3, 256, 12, 1);

@benchmark model_elbo($model_cpu, $x, $x_mask)
@benchmark model_elbo($model_gpu, $xc, $xc_mask)

@benchmark compute_grad_cpu()
@benchmark compute_grad_gpu() 

## gpu testing of backward of model
(x, x_mask), (xc, xc_mask) = make_cpu_gpu_batch_for_model(3, 256, 12, 2);

@benchmark model_elbo($model_cpu, $x, $x_mask)
@benchmark model_elbo($model_gpu, $xc, $xc_mask)

@benchmark compute_grad_cpu()
@benchmark compute_grad_gpu() 

## gpu testing of backward of model
(x, x_mask), (xc, xc_mask) = make_cpu_gpu_batch_for_model(3, 256, 12, 4);

@benchmark model_elbo($model_cpu, $x, $x_mask)
@benchmark model_elbo($model_gpu, $xc, $xc_mask)

@benchmark compute_grad_cpu()
@benchmark compute_grad_gpu() 


## gpu testing of backward of model
(x, x_mask), (xc, xc_mask) = make_cpu_gpu_batch_for_model(3, 256, 12, 8);

@benchmark model_elbo($model_cpu, $x, $x_mask)
@benchmark model_elbo($model_gpu, $xc, $xc_mask)

@benchmark compute_grad_cpu()
@benchmark compute_grad_gpu() 

## gpu testing of backward of model
(x, x_mask), (xc, xc_mask) = make_cpu_gpu_batch_for_model(3, 256, 12, 16);

@benchmark model_elbo($model_cpu, $x, $x_mask)
@benchmark model_elbo($model_gpu, $xc, $xc_mask)

@benchmark compute_grad_cpu()
@benchmark compute_grad_gpu() 

## gpu testing of backward of model
(x, x_mask), (xc, xc_mask) = make_cpu_gpu_batch_for_model(3, 256, 12, 32);

@benchmark model_elbo($model_cpu, $x, $x_mask)
@benchmark model_elbo($model_gpu, $xc, $xc_mask)

@benchmark compute_grad_cpu()
@benchmark compute_grad_gpu() 

## gpu testing of backward of model
(x, x_mask), (xc, xc_mask) = make_cpu_gpu_batch_for_model(3, 256, 12, 48);

@benchmark model_elbo($model_cpu, $x, $x_mask)
@benchmark model_elbo($model_gpu, $xc, $xc_mask)

@benchmark compute_grad_cpu()
@benchmark compute_grad_gpu()

## gpu testing of backward of model
(x, x_mask), (xc, xc_mask) = make_cpu_gpu_batch_for_model(3, 256, 12, 64);

@benchmark model_elbo($model_cpu, $x, $x_mask)
@benchmark model_elbo($model_gpu, $xc, $xc_mask)

@benchmark compute_grad_cpu()
@benchmark compute_grad_gpu()
#@benchmark compute_grad_gpu() seconds=30

## gpu testing of backward of model
CUDA.reclaim(); GC.gc(true)

## fit linear dependancy of time on batch size, for BS in [1, 2, 4, 8, 16, 32, 64]
using CairoMakie

X = [
    1   10.764   40.068 1 1;
    2   11.107   42.680 1 4;
    4   13.775   47.988 1 16;
    8   18.538   63.844 1 8*8;
    16  29.219   79.234 1 16*16;
    32  50.455  132.010 1 32*32;
    #48  70.668  251.680 1 48*48;
    48  70.668  933.203 1 48*48;
    #64  94.851  894.640 1 64*64; # once measured
    64  94.851  1693.000 1 64*64;
    #64 795.547 1693.000;
]

## linear
β = X[1:end-1, [4,1]] \ X[1:end-1, [2,3]] # linear regression coefficients for CPU and GPU backward pass time vs batch size
β[4]/β[2],  β[3]/β[1] 

fig = Figure();
axes = Axis(fig[1, 1], xlabel="Batch Size", ylabel="Time (ms)", title="Backward Pass Time vs Batch Size");
lines!(axes, X[:, 1], X[:, 2], label="GPU-forward", color=:blue)
lines!(axes, X[:, 1], X[:, 3], label="GPU-backward", color=:red)
lines!(axes, X[:, 1], β[2] * X[:, 1] .+ β[1], label="GPU-forward linear fit (y = $(round(β[2], digits=2)) * x + $(round(β[1], digits=2)))", color=:green, linestyle=:dash)
lines!(axes, X[:, 1], β[4] * X[:, 1] .+ β[3], label="GPU-backward linear fit (y = $(round(β[4], digits=2)) * x + $(round(β[3], digits=2)))", color=:orange, linestyle=:dash)
#f[1, 2] = Legend(fig, ax, "Trig Functions", framevisible = false)
axislegend(axes, unique=true, position=:lt)
fig

β = X[1:end-2, [4,1]] \ X[1:end-2, [2,3]] # linear regression coefficients for CPU and GPU backward pass time vs batch size
β[4]/β[2],  β[3]/β[1] 


## polynomial
β = X[:, [5,1]] \ X[:, 3]
A = 1:64 

fig = Figure();
axes = Axis(fig[1, 1], xlabel="Batch Size", ylabel="Time (ms)", title="Backward Pass Time vs Batch Size");
lines!(axes, X[:, 1], X[:, 2], label="GPU-forward", color=:blue)
lines!(axes, X[:, 1], X[:, 3], label="GPU-backward", color=:red)
lines!(axes, A, β[1] * A.^2 .+ β[2] * A, label="GPU-backward quadratic fit", color=:orange, linestyle=:dash)
fig

# -> most likely piecewise linear, with a change in slope at around batch=32

