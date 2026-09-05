# middle step in organizing this repository
# TODO do structure properly

include("utils.jl")

include("mlps.jl")
export SplitLayer, create_mlp, create_gaussian_mlp

include("attention.jl")
export MultiheadAttention, attention, slot_attention, additive_masking, multiplicative_masking

include("prior.jl")
export sample_sphere, gumbel_softmax
export MixtureOfGaussians, ConstGaussPrior

include("transformer_blocks.jl")
export MultiheadAttentionBlock, InducedSetAttentionBlock, InducedSetAttentionHalfBlock
export VariationalBottleneck, AttentiveBottleneckLayer, AttentiveHalfBlock

include("made.jl")
export MaskedDense, MaskedGaussian, MADE

include("flow_layers.jl")
export ActNorm, Invertible1x1Conv, ConcatSquashDense #AffineCoupling TODO finish

include("pooling_layers.jl")
export AbstractPooling
export AttentionPooling, PMA, PoolEncoder

include("transformer_decoders.jl")
export CrossAttentionDecoder, TransformerDecoder

include("show_methods.jl")


