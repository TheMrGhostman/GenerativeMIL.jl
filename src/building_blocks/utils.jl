_resolve_activation(activation::Function) = activation

function _resolve_activation(activation::String)
    symbol = Symbol(activation)
    isdefined(Flux, symbol) || error("Unknown activation function: $activation")
    return getfield(Flux, symbol)
end

_dimension_concat(x) = cat(x..., dims=1) # i had problem with deserialization of lambda funtions, so i just named it