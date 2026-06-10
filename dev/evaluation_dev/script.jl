using Revise
using Random
using Serialization
using YAML
using Flux
using MLUtils
using Statistics
using StatsBase
using Zygote
using GenerativeMIL
using GenerativeMIL: pairwise, chamfer_distance
using GenerativeMIL.NearestNeighbors
using BenchmarkTools
using LinearAlgebra

using CairoMakie

pth = joinpath(@__DIR__)
include(joinpath(pth, "functions.jl"))

o = deserialize("/Users/ghosty/AI_Center/GenerativeMIL.jl/dev/setvae_debugging/ins/airplane/setvae/airplane-c401-reconstructions_final.jls")

T = Float32
rx1 = randn(T, 3, 4, 10);
rx2 = randn(T, 3, 4, 10);

rxv1 = collect(eachslice(rx1, dims=3));
rxv2 = collect(eachslice(rx2, dims=3));

rxt1 =  [rx1[:, :, i:i] for i in axes(rx1, 3)];
rxt2 =  [rx2[:, :, i:i] for i in axes(rx2, 3)];


chamfer_distance(rx1, rx2)
chamfer_distance.(rxt1, rxt2) |> mean
#chamfer_distance(rxv1[1], rxv2[1]) # does not work

pairwise(chamfer_distance, rxt1, rxt2)

nn_pwd = pairwise(chamfer_distance, vcat(rxt1, rxt2); symmetric=true)
labels = vcat(fill(1, length(rxt1)), fill(2, length(rxt2)))

@benchmark one_nn(nn_pwd, labels; exclude_self=true)

@benchmark nn_pwd[sortperm(nn_pwd .+ diagm(20,20,fill(Inf, 20)), dims=1)[1,:]]
#7 times slower then one_nn but still pretty fast

@benchmark begin
    tmp = rem.(sortperm(nn_pwd .+ diagm(20,20,fill(Inf, 20)), dims=1)[1,:], 20) #.+ 1
    tmp[tmp .== 0] .= 20
    getindex(labels, tmp)
end


mean(one_nn(nn_pwd, labels; exclude_self=true) .== labels)

one_nn_accuracy(nn_pwd, labels; exclude_self=true)


x_hat = tensor_to_vector_of_matrices(o.xhat);
x = tensor_to_vector_of_matrices(o.x);
X = vcat(x, x_hat);
labels = vcat(fill(1, length(x)), fill(2, length(x_hat)));
pdm = pairwise(chamfer_distance, X; symmetric=true) # this is the bottleneck, it is too slow. maybe i can speed it up by sending it to gpu. 

one_nn_accuracy(pdm, labels; exclude_self=false)