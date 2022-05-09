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


# functions and modules needed for SetVAE
include("utils/attention.jl")
include("utils/prior.jl")
include("utils/layers.jl")
include("utils/utils.jl")
include("utils/dataset.jl")
#include("utils/losses.jl")
include("SetVAE.jl")
include("PoolAE.jl")

Zygote.@adjoint CUDA.ones(x...) = CUDA.ones(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.randn(x...) = CUDA.randn(x...), _ -> map(_ -> nothing, x)
end