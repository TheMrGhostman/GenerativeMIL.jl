module GenerativeMIL


# Basic Packages
using DrWatson
using Random
using StatsBase
using Distributions
using LinearAlgebra
using Statistics
using ProgressBars
# Deep Learning & Gradients Packages
using Flux
using Zygote
using CUDA
#using cuDNN # necessary to work for |> gpu
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
# Auxilary & priors
using Distances
# for chamfer distance
using NearestNeighbors 



dict2nt(x) = (; (Symbol(k) => v for (k,v) in x)...)

Zygote.@adjoint CUDA.ones(x...) = CUDA.ones(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.randn(x...) = CUDA.randn(x...), _ -> map(_ -> nothing, x)

export check, dict2nt, Models #TODO

const Mask = Union{AbstractArray{Bool}, Nothing}
const MaskT{T} = Union{AbstractArray{Bool}, AbstractArray{T} , Nothing}
const BetaArg = Union{AbstractFloat,AbstractVector{<:AbstractFloat}}


# Loading & Helper functions for datasets
include("dataset.jl")
export load_modelnet10, load_mnist

# logger for 
include("json_logger.jl")
export JSONLLogger, log!, close

# Utils and helper functions
include("utils.jl")
export unpack_mill, check, get_device, unmask, lpad_number

# Losses
include("losses/Losses.jl")

# Model's Building building_blocks
include("building_blocks/Building_Blocks.jl")

# Model Zoo 
include("models/Models.jl")

# Everything related to model training
include("model_training/schedulers.jl")
export WarmupCosine, WarmupLinear, CreateLrScheduler, CreateAnealer
include("model_training/early_stopping.jl")
export EarlyStopping
include("model_training/fits.jl")
include("model_training/train_steps.jl")
include("model_training/training.jl")
export train_model!, validation_check


# Temporary evaluation function
include("evaluation.jl")

include("printing.jl")

# TODO export functions

end