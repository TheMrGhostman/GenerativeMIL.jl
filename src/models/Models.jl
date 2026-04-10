module Models

include("models/SetVAE.jl")
export SetVae, loss, loss_gpu, setvae_constructor_from_named_tuple

include("models/FoldingVAE.jl")

include("models/PoolAE.jl")

include("models/SetTransformer.jl")

include("models/SetVAEformer.jl") # TODO finish this

include("models/vae.jl")

include("models/VQVAE.jl")

include("models/VQVAE_PoolAE.jl")

end