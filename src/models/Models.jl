module Models

using Flux
using Distributions
using LinearAlgebra
using Statistics
using StatsBase
using Random


# functions and modules needed for SetVAE
include("utils/attention.jl")
include("utils/layers.jl")
include("utils/prior.jl")
include("utils/utils.jl")

end