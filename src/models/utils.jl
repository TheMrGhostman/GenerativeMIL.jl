struct SplitLayer
    μ::Flux.Dense
    σ::Flux.Dense
end

Flux.@functor SplitLayer

function SplitLayer(in::Int, out::NTuple{2, Int}, acts::NTuple{2, Function})
	SplitLayer(
		Flux.Dense(in, out[1], acts[1]),
		Flux.Dense(in, out[2], acts[2])
	)
end

function (m::SplitLayer)(x)
	return (m.μ(x), m.σ(x))
end


struct MaskedDense
    dense::Flux.Dense
end

Flux.@functor MaskedDense

function MaskedDense(in, out, σ=identity; bias=true)
    m = Flux.Dense(in, out, σ, bias=bias)
    return MaskedDense(m)
end

function (m::MaskedDense)(x::AbstractArray{<:Real}, mask::Nothing=nothing)
    return m.dense(x)
end

function (m::MaskedDense)(x::AbstractArray{<:Real}, mask::AbstractArray{<:Real}) 
    # masking of input as well as of output
    return m.dense(x .* mask) .* mask
end

# mask function
function mask(x::AbstractArray{<:Real}, mask::Nothing=nothing)
    return x
end

function mask(x::AbstractArray{<:Real}, mask::AbstractArray{<:Real})
    return x .* mask
end

function unmask(x, mask, output_dim=3)
    x = reshape(x, (output_dim,:))
    mask = reshape(mask, (1,:))
    x_masked = ones(size(x)...) .* mask
    x = reshape(x[x_masked .== 1], (output_dim,:))
    return x
end


function check(x::AbstractArray{<:Real})
    println("size -> $(size(x)) | type -> $(typeof(x)) | mean -> $(Flux.mean(x)) | var -> $(Flux.var(cpu(x))) | sum -> $(Flux.sum(x)) | not zero -> $(sum(x .!= 0)) | n_elements -> $(prod(size(x))) ")
end

function shifted_tanh(x, bias=1, scale=2)
    x = Flux.tanh.(x)
    x = (x .+ bias) ./ scale
end