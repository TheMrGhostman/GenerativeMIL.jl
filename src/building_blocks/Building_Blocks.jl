# middle step in organizing this repository
# TODO do structure properly

include("attention.jl")
export MultiheadAttention, attention, slot_attention, _softmax

include("prior.jl")

include("transformer_blocks.jl")

include("layers.jl")

include("made.jl")

include("pooling_layers.jl")

include("encoders_and_decoders.jl")
