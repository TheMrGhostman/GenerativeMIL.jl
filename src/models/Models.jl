# TODO remove and move to GenerativeMIL.jl

abstract type AbstractGenModel end 
# if you want to use build-in functions you have to specify this for every new model! #TODO think about this more later
loss(model::AbstractGenModel, args...) = error("loss not implemented for $(typeof(model))")
reconstruct(model::AbstractGenModel, args...) = error("reconstruct not implemented for $(typeof(model))")
valid_step(model::AbstractGenModel, args...; kwargs...) = error("valid_step not implemented for $(typeof(model))")
optim_step(model::AbstractGenModel, args...; kwargs...) = error("optim_step not implemented for $(typeof(model))")

include("SetVAE.jl")
export SetVae, loss, loss_gpu
export setvae_constructor_from_named_tuple

include("FoldingVAE.jl")
export FoldingNet_VAE, simple_loss, logging_loss
export foldingnet_constructor_from_named_tuple

include("PoolAE.jl")
export PoolModel, loss, loss_with_logging, loss
export poolmodel_constructor_from_named_tuple

include("SetTransformer.jl")
export SetClassifier, loss

include("SetVAEformer.jl") # TODO finish this

include("vae.jl")
export VariationalAutoencoder, elbo_with_logging, optim_step, valid_step

include("VQVAE.jl")
export VectorQuantizer, VectorQuantizerEMA
export VQVAE, loss_ema, loss_gradient
export vqvae_constructor_from_named_tuple

include("VQVAE_PoolAE.jl")
export VectorGaussianQuantizerEMA, ema_update!
export VQ_PoolAE, VGQ_PoolAE, loss_gradient, loss_ema
export vgq_poolae_constructor_from_named_tuple, vgq_poolae_constructor_from_named_tuple