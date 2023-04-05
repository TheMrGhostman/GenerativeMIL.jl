module GenerativeMIL

using DrWatson
using Flux
using Flux3D
using Distributions
using MLDataPattern
using AbstractTrees
using Random

dict2nt(x) = (; (Symbol(k) => v for (k,v) in x)...)
export check, dict2nt, Models

include("dataset.jl")
include("models/Models.jl")
include("evaluation.jl")

end