module GenerativeMIL

using DrWatson
using Flux
using Flux3D
using Distributions
using MLDataPattern
using AbstractTrees
using Random

export check

include("dataset.jl")
include("models/Models.jl")
include("evaluation.jl")


end