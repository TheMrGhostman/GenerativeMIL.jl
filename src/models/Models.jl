# TODO remove and move to GenerativeMIL.jl


include("SetVAE.jl")
export SetVae, loss, loss_gpu
export setvae_constructor_from_named_tuple

include("FoldingVAE.jl")
export FoldingNet_VAE, simple_loss, logging_loss
export foldingnet_constructor_from_named_tuple

include("PoolAE.jl")
export PoolModel, loss, loss_with_kld
export poolmodel_constructor_from_named_tuple

include("SetTransformer.jl")
export SetClassifier, loss

include("SetVAEformer.jl") # TODO finish this

include("vae.jl")
export VariationalAutoencoder, loss

include("VQVAE.jl")
export VectorQuantizer, VectorQuantizerEMA
export VQVAE, loss_ema, loss_gradient
export vqvae_constructor_from_named_tuple

include("VQVAE_PoolAE.jl")
export VectorGaussianQuantizerEMA, ema_update!
export VQ_PoolAE, VGQ_PoolAE, loss_gradient, loss_ema
export vgq_poolae_constructor_from_named_tuple, vgq_poolae_constructor_from_named_tuple