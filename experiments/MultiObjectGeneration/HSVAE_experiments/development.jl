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


## Sanity checks here
batch = build_test_batch(3, 8, 4, 5, 2);
x, y, x_mask = batch;

pdm = chamfer_distance_clusters_A(x, y);
size(pdm) # (L, M, BS)

function sanity_check_chamfer_distance_clusters(x, y)
    pdm = chamfer_distance_clusters_A(x, y)
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
@info "sanity check for chamfer_distance_clusters: $(sc |> all) -->  $(mean(sc)*100)% of the pairs match"


