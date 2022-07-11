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


function transform_batch(x, max=false)
    a_mask = [ones(size(a)) for a in x];
    if max
        max_set = maximum(size.(x))[end];
    else
        max_set = minimum(size.(x))[end]; #minimum
    end
    b = map(a->Array{Float32}(PaddedView(0, a, (3, max_set))), x);
    b_mask = map(a->Array(PaddedView(0, a, (3, max_set))), a_mask);
    c = cat(b..., dims=3);
    c_mask = cat(b_mask..., dims=3) .> 0; # mask as BitArray
    c_mask = Array(c_mask[1:1,:,:]);
    return c, c_mask
end


# Some functions stolen from GroupAD
"""
	unpack_mill(dt<:Tuple{BagNode,Any})

Takes Tuple of BagNodes and bag labels and returns
both in a format that is fit for Flux.train!
"""
function unpack_mill(dt::T) where T <: Tuple{BagNode,Any}
    bag_labels = dt[2]
	bag_data = [dt[1][i].data.data for i in 1:Mill.length(dt[1])]
    return bag_data, bag_labels
end
"""
	unpack_mill(dt<:Tuple{Array,Any})

To ensure reproducibility of experimental loop and the fit! function for models,
this function returns unchanged input, if input is a Tuple of Arrays.
Used in toy problems.
"""
function unpack_mill(dt::T) where T <: Tuple{Array,Any}
    bag_labels = dt[2]
	bag_data = dt[1]
    return bag_data, bag_labels
end