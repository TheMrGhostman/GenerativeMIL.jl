module GenerativeMIL


# Basic Packages
using DrWatson
using Random
using StatsBase
using Distributions
using LinearAlgebra
using Statistics
# Deep Learning & Gradients Packages
using Flux
using Zygote
using CUDA
using cuDNN # necessary to work for |> gpu
using MLUtils
using Optimisers
# Training related Packages
#using MLDataPattern
using ParameterSchedulers # schedulers and warmups
using ValueHistories
using JSON3
# Preprocessing & Data
using PaddedViews
using HDF5
using Serialization
# Multi Instance Learning Library
using Mill
# Visualization
using AbstractTrees
# Auxilary
using Distances
# for chamfer distance
using NearestNeighbors 



dict2nt(x) = (; (Symbol(k) => v for (k,v) in x)...)

Zygote.@adjoint CUDA.ones(x...) = CUDA.ones(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.randn(x...) = CUDA.randn(x...), _ -> map(_ -> nothing, x)

export check, dict2nt, Models #TODO

Mask = Union{AbstractArray{Bool}, Nothing}
MaskT{T} = Union{AbstractArray{Bool}, AbstractArray{T} , Nothing}

# Loading & Helper functions for datasets
include("dataset.jl")
export load_modelnet10, load_mnist

# Model's Building Blocks
include("building_blocks/attention.jl")
include("building_blocks/prior.jl")
include("building_blocks/transformer_blocks.jl")
include("building_blocks/layers.jl")
include("building_blocks/made.jl")
include("building_blocks/pooling_layers.jl")
include("building_blocks/losses.jl") # masked_chamfer_distance_cpu
include("building_blocks/encoders_and_decoders.jl")

# Model Zoo 
include("models/Models.jl")

# Everything related to model training
include("model_training/fits.jl")
include("model_training/training.jl")
include("model_training/train_steps.jl")
include("model_training/early_stopping.jl")
export EarlyStopping
include("model_training/chamfer_distance.jl")
export chamfer_distance

# Utils and helper functions
include("utils.jl")
export unpack_mill, check, get_device, mask, unmask
export WarmupCosine, WarmupLinear
include("json_logger.jl")
export JSONLLogger, log!


# Temporary evaluation function
include("evaluation.jl")

# TODO export functions

end