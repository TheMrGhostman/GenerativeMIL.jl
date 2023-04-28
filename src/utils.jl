# mask function
function mask(x::AbstractArray{<:Real}, mask::Nothing=nothing)
    return x
end

function mask(x::AbstractArray{<:Real}, mask::AbstractArray{<:Real})
    return x .* mask
end

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
    p = Flux.params(m)
    tp = typeof(p[1])
    (tp <: CUDA.CuArray) ? CUDA : Base
end

function shifted_tanh(x, bias=1, scale=2)
    x = Flux.tanh.(x)
    x = (x .+ bias) ./ scale
end

function transform_batch(x::AbstractArray{T,3}, kwargs...) where T<:Real
    return MLUtils.getobs(x), ones(Bool,size(x[1:1,:,:]))
end

function transform_batch(x::AbstractArray{T,1}, max=false) where T<:AbstractArray
    a_mask = [ones(size(a)) for a in x];
    feature_dim = size(x[1],1)
    if max
        max_set = maximum(size.(x))[end];
    else
        max_set = minimum(size.(x))[end]; #minimum
    end
    b = map(a->Array{Float32}(PaddedView(0, a, (feature_dim, max_set))), x);
    b_mask = map(a->Array(PaddedView(0, a, (feature_dim, max_set))), a_mask);
    c = cat(b..., dims=3);
    c_mask = cat(b_mask..., dims=3) .> 0; # mask as BitArray
    c_mask = Array(c_mask[1:1,:,:]);
    return c, c_mask
end


"""
scheduler with warmup
using ParameterSchedulers
x = [1:1200...]
s = WarmupLinear(0, 0.1, 0.001, 200, 1000, CosAnneal(λ0=0.001, λ1=0.1, period=1000))

lineplot(x, s.(x); border= :none)
    ┌─────────────────────────────────────────────┐ 
0.1 │⠀⠀⠀⣸⠉⠉⠓⠢⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
    │⠀⠀⢀⡇⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
    │⠀⠀⡼⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
    │⠀⢠⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
    │⠀⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠳⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
    │⢰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠓⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
  0 │⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠲⢤⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ 
    └─────────────────────────────────────────────┘ 
    0                                          2000 
"""
WarmupLinear(startlr, initlr, warmup, total_iters, schedule) =
    ParameterSchedulers.Sequence(
        ParameterSchedulers.Triangle(λ0 = startlr, λ1 = initlr, period = 2 * warmup) => warmup,
        schedule => total_iters
    )

WarmupCosine(startlr, initlr, finallr, warmup, total_iters) =
    ParameterSchedulers.Sequence(
        ParameterSchedulers.Triangle(λ0 = startlr, λ1 = initlr, period = 2 * warmup) => warmup,
        ParameterSchedulers.CosAnneal(λ0 = finallr, λ1 = initlr, period=total_iters) => total_iters,
        finallr => Inf # to prevent periodicity of cosine
    )

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

AbstractTrees.children((name, m)::Tuple{String, Union{Flux.Dense, Flux.LayerNorm}}) = () # expand for all flux layers
AbstractTrees.printnode(io::IO, (name, m)::Tuple{String, Union{Flux.Dense, Flux.LayerNorm}}) = print(io, "$(name) -- $(m)")

AbstractTrees.children((name, m)::Tuple{String, Flux.Chain}) = (m) # expand for all flux layers
AbstractTrees.printnode(io::IO, (name, m)::Tuple{String, Flux.Chain}) = print(io, "$(name) -- Chain")

AbstractTrees.children((name, m)::Tuple{String, SplitLayer}) = (("μ", m.μ), ("σ", m.σ)) 
AbstractTrees.printnode(io::IO, (name, m)::Tuple{String, SplitLayer}) = print(io, "$(name) -- SplitLayer")

AbstractTrees.children((name, m)::Tuple{String, AbstractArray}) = () 
AbstractTrees.printnode(io::IO, (name, x)::Tuple{String, AbstractArray}) = print(io, "$(name) -- \
    $(size(x)) | $(typeof(x)) | mean~$(round(Flux.mean(x), digits=3)) | xᵢ≠0: $(sum(x .!= 0)) | n(x): $(prod(size(x))) ")