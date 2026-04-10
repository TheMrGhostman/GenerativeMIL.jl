# TODO remove and move to GenerativeMIL.jl


include("SetVAE.jl")
export SetVae, loss, loss_gpu, setvae_constructor_from_named_tuple

include("FoldingVAE.jl")

include("PoolAE.jl")

include("SetTransformer.jl")

include("SetVAEformer.jl") # TODO finish this

include("vae.jl")

include("VQVAE.jl")

include("VQVAE_PoolAE.jl")
