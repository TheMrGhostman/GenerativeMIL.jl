_resolve_activation(activation::Function) = activation

function _resolve_activation(activation::String)
    symbol = Symbol(activation)
    isdefined(Flux, symbol) || error("Unknown activation function: $activation")
    return getfield(Flux, symbol)
end