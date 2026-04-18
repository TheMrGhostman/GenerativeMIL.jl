# mask function
lpad_number(ep, epochs) = lpad(string(ep), length(string(epochs)), "0")

function unmask(x, mask)
    output_dim = size(x, 1)
    x = reshape(x, (output_dim,:))
    mask = reshape(mask, (1,:))
    x_masked = ones(size(x)...) .* mask
    x = reshape(x[x_masked .== 1], (output_dim,:))
    return x
end

"""
function unmask(x, mask, output_dim=3)# FIXME to accept other output_dims
    x = reshape(x, (output_dim,:))
    mask = reshape(mask, (1,:))
    x_masked = ones(size(x)...) .* mask
    x = reshape(x[x_masked .== 1], (output_dim,:))
    return x
end
"""

function check(x::AbstractArray{<:Real})
    println("size -> $(size(x)) | type -> $(typeof(x)) | mean -> $(Flux.mean(x)) | var -> $(Flux.var(cpu(x))) | sum -> $(Flux.sum(x)) | not zero -> $(sum(x .!= 0)) | n_elements -> $(prod(size(x))) ")
end

function get_device(m)
    """
    Fuction get_device returns CUDA/Base
     according to type of stored weights of model "m"
    """
    p = Flux.trainables(m)
    tp = typeof(p[1])
    (tp <: CUDA.CuArray) ? CUDA : Base
end

function shifted_tanh(x, bias=1, scale=2)
    x = Flux.tanh.(x)
    x = (x .+ bias) ./ scale
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
