module GenerativeMIL

using DrWatson
using Flux
using Flux3D
using Distributions
using MLDataPattern
using AbstractTrees

export check

include("dataset.jl")
include("models/Models.jl")


end