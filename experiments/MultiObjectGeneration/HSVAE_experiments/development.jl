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

batch = build_test_batch(3, 8, 4, 5, 2);
X, Y, mask = batch;

Xr = reshape(X, 3, 8, 4*2)

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



## testing

prepool = Flux.Dense(3, 16, relu)
pooling = PMA(1, 16, 4; attention_fn = slot_attention) # m_z induced points -> Z is a set of m_z tokens, not one vector
postpool = Flux.Dense(16, 16, relu)
encoder = PoolEncoder(prepool, pooling, postpool)

xe = encoder(Xr);
size(xe) # (hidden, m_z, bs)

xe4d = reshape(xe, 16, 1, 4, 2) 
xe4d .* mask

xe3d = mapslices(x->encoder(unsqueeze(x, dims=3)), X, dims=(1,2))

xe3d ≈ xe # only on batchsize = 1
xe4d ≈ xe3d

encoder(Xr, reshape(mask, 1, 1, 4*2)) # <- error problem with definition of build_mask

encoder(Xr, reshape(mask, 1, 1, 4*2) .* ones(Bool, 1, 8, 1)) ≈ xe

repeat(reshape(mask, (1,1,4*2)), 1, 8, 1) ≈ reshape(mask, 1, 1, 4*2) .* ones(Bool, 1, 8, 1)

sum(mask, dims=(1,2,3)) #  cardinality check


dₓ, n, l, bs = size(X)
x = reshape(X, dₓ, n, l*bs)
x = encoder(x);
x = reshape(x, 16, 1, l, bs) .* mask;
x = dropdims(x, dims=2)
m = dropdims(mask, dims=2)
size(x)

inner_model = DeepSlotQueryVAE(16, 64, 4, 5, 16, 2, 2)

x̂, logits_exist, μ_z, Σ_z = inner_model(x, m) # <- error problem with definition of build_mask
size(x̂), size(logits_exist)
dₕ, n_slots, bs = size(x̂)

decoder = MultiheadAttentionBlock(16, 4; activation=relu, attention_fn=attention)
output = Flux.Dense(16, 3)

prior = MLUtils.randn_like(x, (dₕ, n, n_slots * bs));
size(prior), size(x̂)

x̂ = reshape(x̂, (dₕ, 1, n_slots * bs))
x̂ = decoder(prior, x̂)
size(x̂)
x̂ = reshape(x̂, dₕ, n, n_slots, bs);
size(x̂)
x̂ = output(x̂)

x̂ |> size


xx = reshape(X, 3, n*l, bs);
x̂x̂ = reshape(x̂, 3, n* n_slots, bs);

size(x̂x̂), size(xx)

loss = GenerativeMIL.chamfer_distance_eval(x̂x̂, xx)


function chamfer_distance_clusters(x::AbstractArray{T,4}, y::AbstractArray{T,4}; w1::T=one(T), w2::T=one(T), agg::Function=mean) where T<:AbstractFloat
    # Plain differentiable minimum(dims=k) instead of argmin+gather: benchmarked with
    # Zygote.gradient on both CPU and GPU (RTX 4070 SUPER) at small size (D=8,N=32,L=M=6,BS=4) --
    # this version was at least as fast as the argmin+ignore+gather version on GPU (21.8ms vs
    # 24.6ms median, forward+backward) because argmin(dims=k) on GPU is ~100x slower than plain
    # minimum(dims=k) (serial value+index-tracking kernel vs the fast parallel-reduction path),
    # which ate up whatever the smaller backward graph was supposed to save. The argmin version
    # only won on CPU (3.4ms vs 4.4ms) and also had a latent bug: CartesianIndex.(nb, mgrid, bsgrid)
    # sat outside Zygote.@ignore, which crashed Zygote.gradient on GPU (mixing CPU Range args with
    # CuArray indices inside Zygote's broadcast pullback). This version has no indices at all, so
    # that whole class of bug doesn't apply.
    D, N, L, BS = size(x)
    _, N2, M, _  = size(y)
    @assert N == N2 "points-per-cluster must match between x and y"

    xr = reshape(x, D, N*L, BS)
    yr = reshape(y, D, N*M, BS)

    P  = _pairwise_sqdist_batched(xr, yr)          # (N*L, N*M, BS)
    P5 = reshape(P, N, L, N, M, BS)                # (n_a, l, n_b, m, bs)

    dist_A_to_B = dropdims(agg(dropdims(minimum(P5, dims=3), dims=3), dims=1), dims=1)   # (L,M,BS)
    dist_B_to_A = dropdims(agg(dropdims(minimum(P5, dims=1), dims=1), dims=2), dims=2)   # (L,M,BS)

    return w1 .* dist_A_to_B .+ w2 .* dist_B_to_A   # (L, M, BS)
end

function hungarian_match_clusters(C::AbstractArray{T,3}, l_mask::AbstractArray{Bool,4}) where T<:AbstractFloat
    # C      :: (L, M, BS)   cost matrix from chamfer_distance_clusters
    # l_mask :: (1, 1, L, BS) validity mask for the L (ground-truth) clusters
    L, M, BS = size(C)
    C_cpu, mask_cpu = Array(C), Array(l_mask)
    matched_l = Vector{Vector{Int}}(undef, BS)
    matched_m = Vector{Vector{Int}}(undef, BS)
    for b in 1:BS
        l_idx = findall(vec(mask_cpu[1, 1, :, b]))
        if isempty(l_idx)
            matched_l[b], matched_m[b] = Int[], Int[]
            continue
        end
        Cb = C_cpu[l_idx, :, b]                       # (n_valid_l, M) — masked rows just aren't there
        assignment, _ = Hungarian.hungarian(Cb)
        valid = findall(!=(0), assignment)
        matched_l[b] = l_idx[valid]
        matched_m[b] = assignment[valid]
    end
    return matched_l, matched_m
end


loss_A, loss_B = chamfer_distance_pairwise(xx, x̂x̂)

loss_A = reshape(loss_A, 3, n, l, bs);
loss_B = reshape(loss_B, 3, n, n_slots, bs);

loss_A |>size, loss_B |> size

sum(loss_A, dims=(1)) |> size, sum(loss_B, dims=(1)) |> size


chamfer_distance_clusters(X, x̂)


using BenchmarkTools

@benchmark chamfer_distance_clusters($X, $x̂)

X_gpu = cu(X);
x̂_gpu = cu(x̂);
@benchmark chamfer_distance_clusters($X_gpu, $x̂_gpu)
@benchmark chamfer_distance_clusters_A($X_gpu, $x̂_gpu)


X_big = rand(Float32, 3, 256, 12, 2);
Y_big = rand(Float32, 3, 256, 12, 2);

X_big_gpu = cu(X_big);
Y_big_gpu = cu(Y_big);

@benchmark chamfer_distance_clusters($X_big, $Y_big)
@benchmark chamfer_distance_clusters($X_big_gpu, $Y_big_gpu)

@belapsed chamfer_distance_clusters($X_big, $Y_big)
@belapsed chamfer_distance_clusters($X_big_gpu, $Y_big_gpu)



loss_f(x, y) = sum(chamfer_distance_clusters(x, y))
grad_sync() = CUDA.@sync Zygote.gradient(loss_f, X_big_gpu, Y_big_gpu)

@benchmark Zygote.gradient($loss_f, $X_big, $Y_big)
@benchmark $grad_sync()


function _nn_indices_clusters(x::AbstractArray{T,4}, y::AbstractArray{T,4}) where T
    D, N, L, BS = size(x); _, N2, M, _ = size(y)
    xr = reshape(x, D, N*L, BS); yr = reshape(y, D, N*M, BS)
    P  = _pairwise_sqdist_batched(xr, yr)
    P5 = reshape(P, N, L, N, M, BS)
    nb = getindex.(dropdims(argmin(P5, dims=3), dims=3), 3)
    na = getindex.(dropdims(argmin(P5, dims=1), dims=1), 1)
    na = permutedims(na, (2,1,3,4))
    return nb, na
end

function chamfer_distance_clusters_A(x::AbstractArray{T,4}, y::AbstractArray{T,4}; agg::Function=mean) where T
    D, N, L, BS = size(x); _, _, M, _ = size(y)
    ci_y, ci_x = Zygote.@ignore begin
        nb, na = _nn_indices_clusters(x, y)
        mgrid, bsgrid, lgrid = reshape(1:M,1,1,M,1), reshape(1:BS,1,1,1,BS), reshape(1:L,1,L,1,1)
        CartesianIndex.(nb, mgrid, bsgrid), CartesianIndex.(na, lgrid, bsgrid)
    end
    B_matched   = y[:, ci_y]
    dist_A_to_B = dropdims(agg(sum(abs2, reshape(x,D,N,L,1,BS) .- B_matched; dims=1); dims=2), dims=(1,2))
    A_matched   = x[:, ci_x]
    dist_B_to_A = dropdims(agg(sum(abs2, A_matched .- reshape(y,D,N,1,M,BS); dims=1); dims=2), dims=(1,2))
    return dist_A_to_B .+ dist_B_to_A
end


chamfer_distance_clusters_A(X_big, Y_big) ≈ chamfer_distance_clusters(X_big, Y_big)
chamfer_distance_clusters_A(X_big_gpu, Y_big_gpu) ≈ chamfer_distance_clusters(X_big_gpu, Y_big_gpu)

chamfer_distance_clusters_A(X_big, Y_big; agg=sum) ≈ chamfer_distance_clusters(X_big, Y_big; agg=sum)

chamfer_distance_clusters_A(X_big_gpu, Y_big_gpu; agg=sum) ≈ chamfer_distance_clusters(X_big_gpu, Y_big_gpu; agg=sum)


@benchmark chamfer_distance_clusters_A($X_big, $Y_big)
@benchmark chamfer_distance_clusters_A($X_big_gpu, $Y_big_gpu)

@belapsed chamfer_distance_clusters_A($X_big, $Y_big)
@belapsed chamfer_distance_clusters_A($X_big_gpu, $Y_big_gpu)

@benchmark chamfer_distance_clusters($X_big, $Y_big)
@benchmark chamfer_distance_clusters($X_big_gpu, $Y_big_gpu)

lossA(x, y) = sum(chamfer_distance_clusters_A(x, y))
gradA_sync() = CUDA.@sync Zygote.gradient(lossA, X_big_gpu, Y_big_gpu)

@benchmark Zygote.gradient($lossA, $X_big, $Y_big)
@benchmark $gradA_sync()

@benchmark Zygote.gradient($loss_f, $X_big, $Y_big)
@benchmark $grad_sync()



@benchmark chamfer_distance_clusters_A($X_big, $Y_big)
@benchmark chamfer_distance_clusters($X_big, $Y_big)


@benchmark chamfer_distance_clusters_A($X_big_gpu, $Y_big_gpu)
@benchmark chamfer_distance_clusters($X_big_gpu, $Y_big_gpu)


@benchmark Zygote.gradient($lossA, $X_big, $Y_big)
@benchmark Zygote.gradient($loss_f, $X_big, $Y_big)


@benchmark $gradA_sync()
@benchmark $grad_sync()


@belapsed Zygote.gradient($lossA, $X_big, $Y_big)
@belapsed Zygote.gradient($loss_f, $X_big, $Y_big)
@belapsed $gradA_sync()
@belapsed $grad_sync()


# forward pass is faster on "chamfer_distance_clusters" 
"""
julia> @belapsed chamfer_distance_clusters($X_big, $Y_big)
0.0442831

julia> @belapsed chamfer_distance_clusters($X_big_gpu, $Y_big_gpu)
0.0148587

julia> @belapsed chamfer_distance_clusters_A($X_big, $Y_big)
0.2714206

julia> @belapsed chamfer_distance_clusters_A($X_big_gpu, $Y_big_gpu)
0.0284307
"""

# backward pass (gradients) is faster on "chamfer_distance_clusters_A"  on both gpu and cpu
"""
julia> @belapsed Zygote.gradient($lossA, $X_big, $Y_big)
0.2711171

julia> @belapsed Zygote.gradient($loss_f, $X_big, $Y_big)
1.0792912

julia> @belapsed $gradA_sync()
0.0348334

julia> @belapsed $grad_sync()
2.1228208
"""



function build_hierarchical_slot_query_vae(D, M, BS)
    # D: data dimension
    # N: number of points per bag
    # L: number of bags
    # M: number of slots
    # BS: batch size

    prepool = Flux.Dense(D, 16, relu)
    pooling = PMA(1, 16, 4; attention_fn = slot_attention) # m_z induced points -> Z is a set of m_z tokens, not one vector
    postpool = Flux.Dense(16, 16, relu)
    encoder = PoolEncoder(prepool, pooling, postpool)

    deep_slot_query = DeepSlotQueryVAE(16, 64, 4, M, 16, 2, 2)

    decoder = MultiheadAttentionBlock(16, 4; activation=relu, attention_fn=attention)
    output = Flux.Dense(16, D)

    return HierarchicalSlotQueryVAE(encoder, deep_slot_query, decoder, output)
end


## basic tests
model = build_hierarchical_slot_query_vae(3, 5, 2)
model_gpu = cu(model);

y, logits_exist, μ_z, Σ_z = model(X, mask);

forward_and_loss(model, X, mask) = begin
    y, logits_exist, μ_z, Σ_z = model(X, mask)
    loss = sum(chamfer_distance_clusters_A(X, y))
    return loss
end

forward_and_loss(model, X, mask)

@benchmark forward_and_loss($model, $X, $mask)

forward_and_loss(model_gpu, cu(X), cu(mask))

batch = build_test_batch(3, 256, 12, 1, 16);
xx, __, xx_mask = batch;
xxc = cu(xx);
xxc_mask = cu(xx_mask);


@benchmark forward_and_loss($model, $xx, $xx_mask)
@benchmark forward_and_loss($model_gpu, $xxc, $xxc_mask)


@elapsed grads = Zygote.gradient(m -> forward_and_loss(m, xx, xx_mask), model)
@elapsed grads = Zygote.gradient(m -> forward_and_loss(m, xxc, xxc_mask), model_gpu)


@benchmark Zygote.gradient(m -> forward_and_loss(m, $xx, $xx_mask), $model)
@benchmark Zygote.gradient(m -> forward_and_loss(m, $xxc, $xxc_mask), $model_gpu)

## bigger model -> (3, 256, 12, 16) data

model2_cpu = build_hierarchical_slot_query_vae(3, 12, 2);
model2_gpu = cu(model2_cpu);
batch = build_test_batch(3, 256, 12, 1, 16);
xx, __, xx_mask = batch;
xxc = cu(xx);
xxc_mask = cu(xx_mask);
@info size(xx), size(xx_mask), size(xxc), size(xxc_mask)

@benchmark forward_and_loss($model2_cpu, $xx, $xx_mask)
@benchmark forward_and_loss($model2_gpu, $xxc, $xxc_mask)


@benchmark Zygote.gradient(m -> forward_and_loss(m, $xx, $xx_mask), $model2_cpu)
@benchmark Zygote.gradient(m -> forward_and_loss(m, $xxc, $xxc_mask), $model2_gpu)

## bigger model -> (3, 256, 12, 32) data

model2_cpu = build_hierarchical_slot_query_vae(3, 12, 2);
model2_gpu = cu(model2_cpu);
batch = build_test_batch(3, 256, 12, 1, 32);
xx, __, xx_mask = batch;
xxc = cu(xx);
xxc_mask = cu(xx_mask);
@info size(xx), size(xx_mask), size(xxc), size(xxc_mask)

@benchmark forward_and_loss($model2_cpu, $xx, $xx_mask)
@benchmark forward_and_loss($model2_gpu, $xxc, $xxc_mask)


@benchmark Zygote.gradient(m -> forward_and_loss(m, $xx, $xx_mask), $model2_cpu)
@benchmark Zygote.gradient(m -> forward_and_loss(m, $xxc, $xxc_mask), $model2_gpu)


## bigger model -> (3, 256, 12, 64) data

model2_cpu = build_hierarchical_slot_query_vae(3, 12, 2);
model2_gpu = cu(model2_cpu);
batch = build_test_batch(3, 256, 12, 1, 64);
xx, __, xx_mask = batch;
xxc = cu(xx);
xxc_mask = cu(xx_mask);
@info size(xx), size(xx_mask), size(xxc), size(xxc_mask)

@benchmark forward_and_loss($model2_cpu, $xx, $xx_mask)
@benchmark forward_and_loss($model2_gpu, $xxc, $xxc_mask)


@benchmark Zygote.gradient(m -> forward_and_loss(m, $xx, $xx_mask), $model2_cpu)
@benchmark Zygote.gradient(m -> forward_and_loss(m, $xxc, $xxc_mask), $model2_gpu)

## bigger model -> (3, 512, 12, 1) data

model2_cpu = build_hierarchical_slot_query_vae(3, 12, 2);
model2_gpu = cu(model2_cpu);
batch = build_test_batch(3, 512, 12, 1, 1);
xx, __, xx_mask = batch;
xxc = cu(xx);
xxc_mask = cu(xx_mask);
@info size(xx), size(xx_mask), size(xxc), size(xxc_mask)

@benchmark forward_and_loss($model2_cpu, $xx, $xx_mask)
@benchmark forward_and_loss($model2_gpu, $xxc, $xxc_mask)


@benchmark Zygote.gradient(m -> forward_and_loss(m, $xx, $xx_mask), $model2_cpu)
@benchmark Zygote.gradient(m -> forward_and_loss(m, $xxc, $xxc_mask), $model2_gpu)


## bigger model -> (3, 512, 12, 8) data

model2_cpu = build_hierarchical_slot_query_vae(3, 12, 2);
model2_gpu = cu(model2_cpu);
batch = build_test_batch(3, 512, 12, 1, 8);
xx, __, xx_mask = batch;
xxc = cu(xx);
xxc_mask = cu(xx_mask);
@info size(xx), size(xx_mask), size(xxc), size(xxc_mask)

@benchmark forward_and_loss($model2_cpu, $xx, $xx_mask)
@benchmark forward_and_loss($model2_gpu, $xxc, $xxc_mask)


@benchmark Zygote.gradient(m -> forward_and_loss(m, $xx, $xx_mask), $model2_cpu)
@benchmark Zygote.gradient(m -> forward_and_loss(m, $xxc, $xxc_mask), $model2_gpu)

## test
model2_cpu = build_hierarchical_slot_query_vae(3, 12, 2);
model2_gpu = cu(model2_cpu);
batch = build_test_batch(3, 512, 12, 1, 8);
xx, __, xx_mask = batch;
xxc = cu(xx);
xxc_mask = cu(xx_mask);
@info size(xx), size(xx_mask), size(xxc), size(xxc_mask)

forward_and_loss(model, X, mask) = begin
    y, logits_exist, μ_z, Σ_z = model(X, mask)
    loss = sum(chamfer_pairwise_distance(X, y))
    return loss
end

forward_and_loss2(model, X, mask, idxs) = begin
    y, logits_exist, μ_z, Σ_z = model(X, mask)
    loss = sum(chamfer_pairwise_distance(X, y)[idxs])
    return loss
end

idxs = [CartesianIndex.(i,i,j) for i in 1:12 for j in 1:8]
tmp = forward_and_loss2(model2_cpu, xx, xx_mask, idxs)

@benchmark forward_and_loss2($model2_gpu, $xxc, $xxc_mask, $idxs)
@benchmark forward_and_loss($model2_gpu, $xxc, $xxc_mask)



## Sanity checks here
batch = build_test_batch(3, 8, 4, 5, 2);
x, y, x_mask = batch;

pdm = chamfer_pairwise_distance(x, y);
size(pdm) # (L, M, BS)

function sanity_check_chamfer_distance_clusters(x, y)
    pdm = chamfer_pairwise_distance(x, y)
    bool_decision = zeros_like(pdm, Bool);
    D, N, L, BS = size(x)
    _, _, M, _  = size(y)
    for l in 1:L
        for m in 1:M
            for b in 1:BS
                x1 = x[:, :, l, b]
                y1 = y[:, :, m, b]
                #@assert chamfer_distance(x1, y1) ≈ pdm[l, m, b]
                bool_decision[l, m, b] = chamfer_distance(x1, y1) ≈ pdm[l, m, b]
            end
        end
    end
    bool_decision
end

sc = sanity_check_chamfer_distance_clusters(x, y)
sc |> all
@info "sanity check for chamfer_distance_clusters: $(sc |> all) -->  $(mean(sc)*100)% of the pairs match"


## hungarian matching implementation
_pairwise_sqeuclidean(a::AbstractMatrix{T}, b::AbstractMatrix{T}) where T<:AbstractFloat = begin
    a2 = sum(abs2, a, dims=1)
    b2 = sum(abs2, b, dims=1)
    max.(a2' .+ b2 .- 2 .* (a' * b), zero(T))
end


function hungarian_match(x̂::AbstractArray{T,3}, x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}) where T<:AbstractFloat
    _, n_slots, bs = size(x̂)
    matched_pred = Vector{Vector{Int}}(undef, bs)
    matched_gt = Vector{Vector{Int}}(undef, bs)
    for b in 1:bs
        gt_idx = findall(vec(x_mask[1, :, b]))
        if isempty(gt_idx)
            matched_pred[b], matched_gt[b] = Int[], Int[]
            continue
        end
        C = _pairwise_sqeuclidean(x̂[:, :, b], x[:, gt_idx, b]) # (n_slots, n_gt)
        assignment, _ = Hungarian.hungarian(C)
        pred_idx = findall(!=(0), assignment)
        matched_pred[b] = pred_idx
        matched_gt[b] = gt_idx[assignment[pred_idx]]
    end
    return matched_pred, matched_gt
end



## testing hungarian matching implementation / development
batch = build_test_batch(1, 8, 6, 5, 1);
x, y, x_mask = batch;
xc = cu(x);
xc_mask = cu(x_mask);

C = chamfer_pairwise_distance(y, x) # (M, L, BS)
#C_gpu = cu(C)
C .* x_mask[1, :, :, 1] # mask out invalid rows | mask ~ (1, 1, L, BS)

x[1, :, :, 1] # valid points in cluster 1
y[1, :, :, 1] # valid points in cluster 1

## reverse engineering |> porting new 4D data into 3D data for hungarian matching and compare distancs
function pairwise_ch(y::AbstractArray{T,3}, x::AbstractArray{T,3}) where T<:AbstractFloat
    D, N, M = size(y)
    _, N2, L = size(x)
    out = zeros(T, M, L)
    for l in 1:L
        for m in 1:M
            x1 = x[:, :, l]
            y1 = y[:, :, m]
            #@assert chamfer_distance(x1, y1) ≈ pdm[l, m, b]
            out[m, l] = chamfer_distance(y1, x1)
        end
    end
    return out
end

function hungarian_match_test(x̂::AbstractArray{T,4}, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}) where T<:AbstractFloat
    _, n,  n_slots, bs = size(x̂)
    matched_pred = Vector{Vector{Int}}(undef, bs)
    matched_gt = Vector{Vector{Int}}(undef, bs)
    for b in 1:bs
        gt_idx = findall(vec(x_mask[1, 1, :, b]))
        if isempty(gt_idx)
            matched_pred[b], matched_gt[b] = Int[], Int[]
            continue
        end
        C = pairwise_ch(x̂[:, :, :, b], x[:, :, gt_idx, b]) # (n_slots, n_gt)
        #display(C)
        assignment, _ = Hungarian.hungarian(C)
        pred_idx = findall(!=(0), assignment)
        matched_pred[b] = pred_idx
        matched_gt[b] = gt_idx[assignment[pred_idx]]
    end
    return matched_pred, matched_gt
end


batch = build_test_batch(3, 1, 6, 5, 1);
x, y, x_mask = batch;

C = chamfer_pairwise_distance(y, x) # (M, L, BS)
C .* x_mask[1, :, :, 1] # mask out invalid rows | mask ~ (1, 1, L, BS)

xd = dropdims(x, dims=2)
yd = dropdims(y, dims=2)

pairwise_ch(y[:,:,:,1], x[:,:,:,1]) ≈  chamfer_pairwise_distance(y, x)[:,:,1]
#mean(abs2, pairwise_ch(y[:,:,:,1], x[:,:,:,1]) .- chamfer_pairwise_distance(y, x)[:,:,1])
chamfer_pairwise_distance(y, x)[:,:,1]

hungarian_match_test(y, x, x_mask)


## matrix based implementation of hungarian matching
"""
`hungarian_match(C::AbstractArray{T,3}, l_mask::AbstractArray{Bool,4}) where T<:AbstractFloat`

Solve the linear assignment problem per batch element and return the matched pairs as
`CartesianIndex` arrays, so the result indexes directly into `(..., M, BS)`- and
`(..., L, BS)`-shaped tensors (e.g. `x̂[:, :, ci_m]`, `x[:, :, ci_l]`) without any further
per-batch bookkeeping. Masked-out `l` columns are dropped from the cost matrix before assignment
(never zeroed/`Inf`-filled), so they can never be selected as a match. Index bookkeeping only —
not meant to be differentiated through, wrap the call in `Zygote.@ignore`.

Arguments (positional):
- `C`: cost matrix `(M, L, BS)` — `M` predicted slots, `L` ground-truth clusters, `BS` batch size
  (e.g. from `chamfer_pairwise_distance`).
- `l_mask`: validity mask `(1, 1, L, BS)` for the `L` ground-truth clusters.

Returns:
- `ci_m::Vector{CartesianIndex{2}}`: matched `(m, bs)` pairs, one per matched pair across the
  whole batch.
- `ci_l::Vector{CartesianIndex{2}}`: matched `(l, bs)` pairs, same order/length as `ci_m`.
"""
function hungarian_match(C::AbstractArray{T,3}, l_mask::AbstractArray{Bool,4}) where T<:AbstractFloat
    M, _, BS = size(C)
    C_cpu, mask_cpu = Array(C), Array(l_mask)
    #ci_m = CartesianIndex{2}[]   # (m, bs) — index into a (..., M, BS) tensor
    #ci_l = CartesianIndex{2}[]   # (l, bs) — index into a (..., L, BS) tensor
    c_ml = CartesianIndex{3}[]  # (m, l, bs) — index into a (..., M, L, BS) tensor
    exist_target = zeros_like(C, (1, M, BS))  # (1, M, BS) - 
    for b in 1:BS
        l_idx = findall(vec(mask_cpu[1, 1, :, b]))
        isempty(l_idx) && continue

        Cb = C_cpu[:, l_idx, b]                       # (M, n_valid_l) — masked columns just aren't there
        assignment, _ = Hungarian.hungarian(Cb)
        matched = filter(c -> c[2] != 0, CartesianIndex.(1:M, assignment))   # (m, position-in-l_idx)

        #append!(ci_m, CartesianIndex.(getindex.(matched, 1), b))
        #append!(ci_l, CartesianIndex.(l_idx[getindex.(matched, 2)], b))
        append!(c_ml, CartesianIndex.(getindex.(matched, 1), l_idx[getindex.(matched, 2)], b))
        exist_target[1, getindex.(matched, 1), b] .= one(T)
    end
    return c_ml, exist_target
end

hungarian_match(chamfer_pairwise_distance(y, x), x_mask) == hungarian_match_test(y, x, x_mask)


hungarian_match(chamfer_pairwise_distance(y, x), x_mask)
hungarian_match_test(y, x, x_mask)

## now moveon hungarian matching loss function for nested vae

batch = build_test_batch(3, 4, 6, 5, 2);
x, y, x_mask = batch;


C = chamfer_pairwise_distance(y, x) # (M, L, BS)
idxs, exist_target = hungarian_match(C, x_mask)

t = zeros(Float32, 1, 6, 1)
t[1, mp[1], 1] .= one(Float32)
t

mp, mg = hungarian_match(C, x_mask)


function hungarian_matching_loss(x̂::AbstractArray{T,4}, x::AbstractArray{T,4}, x_mask::AbstractArray{Bool,4}, logits_exist::AbstractArray{T,3}, distance::Function = chamfer_pairwise_distance) where T<:AbstractFloat
    _, _, M, BS = size(x̂)

    C = distance(x̂, x) # (M, L, BS)
    matched_indices, exist_target = Zygote.@ignore hungarian_match(C, x_mask)
    n_matched = length(matched_indices)

    ℒ_rec = n_matched > 0 ? mean(C[matched_indices]) : zero(T)
    ℒ_exist = Flux.Losses.logitbinarycrossentropy(logits_exist, exist_target; agg=mean) * T(M) # sum over all slots, not mean 
    #matched_frac = T(n_matched) / T(M * BS)
    return ℒ_rec, ℒ_exist#, matched_frac
end

rnd_pred = randn(Float32, 1, 5, 2)
hungarian_matching_loss(y, x, x_mask, rnd_pred)
hungarian_matching_loss(y, x, x_mask, rnd_pred, (x,y) -> chamfer_pairwise_distance(x,y; agg=sum))


## gpu testing 
N, M, L, BS = 256, 10, 12, 8
batch = build_test_batch(3, N, L, M, BS);
x, y, x_mask = batch;
xc = cu(x);
yc = cu(y);
xc_mask = cu(x_mask);
rnd_pred = randn_like(x, (1, M, BS));
rnd_pred_gpu = cu(rnd_pred);

hungarian_matching_loss(y, x, x_mask, rnd_pred)

hungarian_matching_loss(yc, xc, xc_mask, rnd_pred_gpu)

@benchmark hungarian_matching_loss($y, $x, $x_mask, $rnd_pred)
@benchmark hungarian_matching_loss($yc, $xc, $xc_mask, $rnd_pred_gpu)

## gpu testing of backward of model
(x, y, x_mask), (xc, yc, xc_mask) = make_cpu_gpu_batch(3, 256, 12, 1, 8);

model_cpu = build_hierarchical_slot_query_vae(3, 12, 2);
model_gpu = cu(model_cpu);


function model_elbo(model, x, x_mask)
    ŷ, logits_exist, μ_z, Σ_z = model(x, x_mask)
    ℒ_rec, ℒ_exist = hungarian_matching_loss(ŷ, x, x_mask, logits_exist)
    ℒ_kl = mean(GenerativeMIL.kl_divergence(μ_z, Σ_z))
    return ℒ_rec + ℒ_exist + ℒ_kl
end

@benchmark model_elbo($model_cpu, $x, $x_mask)
@benchmark model_elbo($model_gpu, $xc, $xc_mask)

compute_grad_cpu() = Zygote.gradient(m -> model_elbo(m, x, x_mask), model_cpu)
compute_grad_gpu() = CUDA.@sync Zygote.gradient(m -> model_elbo(m, xc, xc_mask), model_gpu)

compute_grad_cpu();
compute_grad_gpu();

@benchmark compute_grad_cpu()
@benchmark compute_grad_gpu()

## gpu testing of backward of model
(x, y, x_mask), (xc, yc, xc_mask) = make_cpu_gpu_batch(3, 512, 12, 1, 8);

@benchmark model_elbo($model_cpu, $x, $x_mask)
@benchmark model_elbo($model_gpu, $xc, $xc_mask)

@benchmark compute_grad_cpu()
@benchmark compute_grad_gpu()

## gpu testing of backward of model
(x, y, x_mask), (xc, yc, xc_mask) = make_cpu_gpu_batch(3, 512, 12, 1, 32);

@benchmark model_elbo($model_cpu, $x, $x_mask)
@benchmark model_elbo($model_gpu, $xc, $xc_mask)

@benchmark compute_grad_cpu()
@benchmark compute_grad_gpu()

## gpu testing of backward of model
(x, y, x_mask), (xc, yc, xc_mask) = make_cpu_gpu_batch(3, 256, 12, 1, 48);

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
(x, x_mask), (xc, xc_mask) = make_cpu_gpu_batch_for_model(3, 256, 12, 64);

@benchmark model_elbo($model_cpu, $x, $x_mask)
@benchmark model_elbo($model_gpu, $xc, $xc_mask)

@benchmark compute_grad_cpu()
@benchmark compute_grad_gpu()
@benchmark compute_grad_gpu() seconds=60

