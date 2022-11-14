
struct ActNorm{T <: Real}
    loc::AbstractArray{T}
    scale::AbstractArray{T}
    initialized::Union{Array{Bool, 1}, Bool} # [false]
end

Flux.trainable(m::ActNorm) = (m.loc, m.scale)

function (m::ActNorm)(x::AbstractArray{T, 3}, reverse::Bool=false) where T<:Real #TODO simplify
    dims = (size(m.loc, 2) == 1 ) ? size(x[1,:,:]) : size(x[:,1,:])
    device = get_device(m.loc)
    logdet = device.zeros(Float32, dims... )
    return m(x, logdet, reverse)
    # similar(x[:,1,:]) .* 0 .+ Flux.sum(log.(abs.(m.scale)))
    # TODO MLUtils.ones_like
    #logdet = ones_like(m.scale, x) .* Flux.sum(log.(abs.(m.scale))) # FIXME ones_like
end

function (m::ActNorm)(X::Tuple{T,T}, reverse::Bool=false) where T<:AbstractArray{<:Real}
    x, logdet = X
    return m(x, logdet, reverse)
end

function (m::ActNorm)(x::AbstractArray{T, 3}, logdet::AbstractArray{T, 2}, reverse::Bool=false) where T<:Real
    # for m.loc ~ (ch,1,1) -> X ~ (d, n, bs)
    if reverse
        x = x ./ m.scale .- m.loc
    else
        if all(m.initialized .== false) # works for both array and value
            initialize(m, x);
        end
        logdet = logdet .+ Flux.sum(log.(abs.(m.scale)))
        x = m.scale .* (x .+ m.loc)
    end
    return x, logdet
end

function initialize(m::ActNorm, x::AbstractArray{<:Real, 3})
    dims = (size(m.loc, 2) == 1 ) ? (2,3) : (1,3)
    mean_ = Flux.mean(x, dims=dims)
    std_ = Flux.std(x, dims=dims)
    m.loc .= - mean_
    m.scale .= (1 ./ (std_ .+ 1e-6))
    m.initialized .= true
end

function ActNorm(in_channels::Int, dim::Int=1)
    if dim == 1
        dims = (in_channels, 1, 1)
    elseif dim == 2
        dims = (1, in_channels, 1)
    else
        error("You set dim to unexpected value. Please check input or change it.")
    end
    loc = zeros(Float32, dims... )
    scale = ones(Float32, dims...)
    return ActNorm(loc, scale, false)
end


struct Invertible1x1Conv
    weight::AbstractArray{<:Real}
end

Flux.trainable(m::Invertible1x1Conv) = (m.weight,)

function (m::Invertible1x1Conv)(x::AbstractArray{T, 3}, reverse::Bool=false) where T<:Real
    return m(x, MLUtils.zeros_like(x[:,1,:]), reverse) # zeros_like ≈ MLUtils.zeros_like
end

function (m::Invertible1x1Conv)(X::Tuple{T,T}, reverse::Bool=false) where T<:AbstractArray{<:Real}
    x, logdet = X
    return m(x, logdet, reverse)
end

function (m::Invertible1x1Conv)(x::AbstractArray{T, 3}, logdet::AbstractArray{T, 2}, reverse::Bool=false) where T<:Real
    # x ~ (d, n, bs)
    x = permtedims(x, (2,1,3)) # (d, n, bs) -> (n, d, bs)
    if reverse
        # inv ≈ LinearAlgebra.inv
        weight_inv = LinearAlgebra.inv(dropdims(m.weight, dims=1))
        x = Flux.conv(x, reshape(weight_inv, 1, size(m.weight)...))
        x = permutedims(x, (2,1,3))
    else
        # logabsdet ≈ LinearAlgebra.logabsdet -> (value, sign)
        logdet = logdet .+ LinearAlgebra.logabsdet(dropdims(m.weight, dims=1))[1]
        x = Flux.conv(x, m.weight)
        x = permutedims(x, (2,1,3))
    end
    return x, logdet
end

function Invertible1x1Conv(channels)
    weight,_ = LinearAlgebre.qr(randn(Float32, channels, channels))
    weight = Array(reshape(weight, 1, channels, channels))
    weight[:,1] = (det(weight) < 0) ? -1 .* weight[:,1] : weight[:,1]
    return Invertible1x1Conv(weight)
end


struct ConcatSquashDense
    layer
    context_gate
    context_bias
end

Flux.@functor ConcatSquashDense

(m::ConcatSquashDense)(x::Tuple{T, T}) where T<:AbstractArray{<:Real} = m(x...)

function (m::ConcatSquashDense)(x::AbstractArray{T, 3}, context::AbstractArray{T,2}) where T<:Real
    # x ~ (d₁, n, bs)
    # context ~ (d₂, bs)
    bias_ = Flux.unsqueeze(m.context_bias(context), 2) # shape depends on x shape
    gate_ = Flux.unsqueeze(m.context_gate(context), 2) 
    out = m.layer(x) .* gate_ .+ bias_
    return out, context
end

function ConcatSquashDense(in_features, in_context, out_features, zeros_init::Bool=false)
    init_ = (zeros_init) ? Flux.zeros32 : Flux.glorot_uniform # how to initialize weights
    layer_ = Flux.Dense(in_features, out_features, init=init_)
    gate_ = Flux.Dense(in_context, out_features, Flux.σ, init=init_) # sigmoid activation
    bias_ = Flux.Dense(in_context, out_features, bias=false, init=init_)
    return ConcatSquashDense(layer_, gate_, bias_)
end


function tuple_activation(f::Function, x::Tuple{T, T}) where T<:AbstractArray{<:Real}
    x, context = x
    return (f.(x), context)
end


struct AffineCoupling
    layers₁::Flux.Chain
    layers₂::Flux.Chain
end

Flux.@functor AffineCoupling

function (m::AffineCoupling)(x::T, std_in::T, context::T, reverse::Bool=false) where T<:AbstractArray{<:Real}
    #TODO
end

function (m::AffineCoupling)(x::Union{T, Tuple{T, T}}, std_in::T, reverse::Bool=false) where T<:AbstractArray{<:Real}
 #TODO
    # 1) split x, context = x if x<:Tuple
    # 2) split x along spatial dimensions -> xₐ, xᵦ
    # 3) cat xₐ = cat(xₐ, std_in)
    # 4) cat xₐ = cat(xₐ, std_in)
end

function reverse(m::AffineCoupling, x, std_in)