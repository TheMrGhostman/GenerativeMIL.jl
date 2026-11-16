include("StructuredMNIST/StructuredMnistHSQVAE.jl")
include("StructuredMNIST/StructuredMnistBagVAE.jl")
export StructuredMnistHSQVAE, StructuredMnistBagVAE
export encode, is_variational   # encode/decode: latent <-> bag; is_variational: VAE (true) or AE (false)
