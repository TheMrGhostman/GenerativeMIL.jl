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

export check

# functions and modules needed for SetVAE
include("building_blocks/attention.jl")
include("building_blocks/prior.jl")
include("building_blocks/layers.jl")
include("utils.jl")
#include("utils/losses.jl")
include("SetVAE.jl")
include("PoolAE.jl")

Zygote.@adjoint CUDA.ones(x...) = CUDA.ones(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.randn(x...) = CUDA.randn(x...), _ -> map(_ -> nothing, x)
end