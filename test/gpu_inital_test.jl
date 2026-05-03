using Test
using Random
using Flux
using CUDA
#using cuDNN
using Zygote
using LinearAlgebra
using AbstractTrees
using Distances, MLUtils, Statistics


using CUDA, Flux

x = cu(randn(Float32, 10, 16)) # (feature_dim, batch_size

x .+ 1f0

Flux.softmax(x, dims=1)

Flux.softmax(x, dims=2)