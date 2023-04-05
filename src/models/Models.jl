module Models

using Flux
using Flux3D
using Zygote
using Distributions
using LinearAlgebra
using Statistics
using StatsBase
using Random
using Distances
using CUDA
using MLDataPattern
using MLUtils
using ValueHistories
using Mill
# schedulers and warmups
using ParameterSchedulers
# because of prepocessing
using PaddedViews
using AbstractTrees


export check

# functions and modules needed for SetVAE
include("building_blocks/attention.jl")
include("building_blocks/prior.jl")
include("building_blocks/layers.jl")
include("building_blocks/made.jl")
include("building_blocks/pooling_layers.jl")
include("utils.jl")
include("building_blocks/losses.jl") # masked_chamfer_distance_cpu
include("SetVAE.jl")
include("FoldingVAE.jl")
include("PoolAE.jl")
include("VQVAE.jl")
include("VQVAE_PoolAE.jl")
include("fits.jl")
include("training.jl")
include("train_steps.jl")
#include("SetTransformer.jl")


Zygote.@adjoint CUDA.ones(x...) = CUDA.ones(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.randn(x...) = CUDA.randn(x...), _ -> map(_ -> nothing, x)
end