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
using Flux3D
using Zygote
using CUDA
using MLUtils
# Training related Packages
using MLDataPattern
using ParameterSchedulers # schedulers and warmups
using ValueHistories
# Preprocessing
using PaddedViews
# Multi Instance Learning Library
using Mill
# Visualization
using AbstractTrees

dict2nt(x) = (; (Symbol(k) => v for (k,v) in x)...)

Zygote.@adjoint CUDA.ones(x...) = CUDA.ones(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)
Zygote.@adjoint CUDA.randn(x...) = CUDA.randn(x...), _ -> map(_ -> nothing, x)

export check, dict2nt, Models

Mask = Union{AbstractArray{Bool}, Nothing}
MaskT{T} = Union{AbstractArray{Bool}, AbstractArray{T} , Nothing}

# Loading & Helper functions for datasets
include("dataset.jl")

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
include("models/SetVAE.jl")
include("models/FoldingVAE.jl")
include("models/PoolAE.jl")
include("models/SetTransformer.jl")
include("models/SetVAEformer.jl") # TODO finish this
include("models/vae.jl")
include("models/VQVAE.jl")
include("models/VQVAE_PoolAE.jl")

# Everything related to model training
include("model_training/fits.jl")
include("model_training/training.jl")
include("model_training/train_steps.jl")

# Utils and helper functions
include("utils.jl")

# Temporary evaluation function
include("evaluation.jl")

# TODO export functions

end