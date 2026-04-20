"""
    create_mlp(idim, hdim, depth, odim, activation; out_identity=false)

Create a fully-connected MLP with uniform hidden dimensions.
Can use `Function` or `String` to specify the activation.

# Arguments
- `idim::Int`: input feature dimension.
- `hdim::Int`: hidden layer width (used for all intermediate layers).
- `depth::Int`: number of layers in the network.
- `odim::Int`: output feature dimension.
- `activation::Union{Function, String}`: hidden layer activation (relu, tanh, etc.).
- `out_identity::Bool`: if `true`, output layer uses identity activation; otherwise uses `activation`.

# Returns
- `Flux.Chain`: constructed MLP chain.
"""
function create_mlp(idim::T, hdim::T, depth::T, odim::T, activation::Function; out_identity::Bool=false) where T<:Int
    hdims = repeat([hdim], depth-1)
    ins_ = vcat(idim, hdims)
    outs_ = vcat(hdims, odim)
    layers = repeat([Flux.Dense], depth)
    activations = repeat([activation], depth-1)
    activations = (out_identity) ? vcat(activations, identity) : vcat(activations, activation)
    _create_chain(ins_, outs_, layers, activations)
end

function create_mlp(idim::T, hdim::T, depth::T, odim::T, activation::String; out_identity::Bool=false) where T<:Int
    activation = eval(:($(Symbol(activation))))
    create_mlp(idim, hdim, depth, odim, activation, out_identity=out_identity)
end

#create_mlp(;idim::T=3, hdim::T=64, depth::T=3, odim::T=3, activation="relu") where T<:Int = create_mlp(idim, hdim, depth, odim, activation)

"""
    create_gaussian_mlp(idim, hdim, depth, odim, activation; softplus_=true)

Create an MLP with a Gaussian (mean/std) split output layer.
The final layer outputs both mean and standard deviation via a `SplitLayer`.

# Arguments
- `idim::Int`: input feature dimension.
- `hdim::Int`: hidden layer width (used for all intermediate layers).
- `depth::Int`: number of layers (including the split output layer).
- `odim::Union{Int, Tuple{Int, Int}}`: output dimension (automatically duplicated if scalar).
- `activation::Union{Function, String}`: hidden layer activation (relu, tanh, etc.).
- `softplus_::Bool`: if `true`, apply softplus to the std output; otherwise identity.

# Returns
- `Flux.Chain`: MLP ending with `SplitLayer` for Gaussian outputs.
"""
function create_gaussian_mlp(idim::T, hdim::T, depth::T, odim::U, activation::Function; softplus_::Bool=true) where {T<:Int, U<:Union{Int, NTuple{2, Int}}}
    hdims = repeat([hdim], depth-1)
    ins_ = vcat(idim, hdims)
    outs_ = vcat(hdims, odim)
    layers = vcat(repeat([Flux.Dense], depth-1), [SplitLayer])
    _softplus = (softplus_) ? Flux.softplus : identity
    activations = vcat(repeat([activation], depth-1), [(identity, _softplus)])
    _create_chain(ins_, outs_, layers, activations)
end 

function create_gaussian_mlp(idim::T, hdim::T, depth::T, odim::U, activation::String; softplus_::Bool=true) where {T<:Int, U<:Union{Int, NTuple{2, Int}}}
    activation = eval(:($(Symbol(activation))))
    create_gaussian_mlp(idim, hdim, depth, odim, activation; softplus_=softplus_)
end

#create_gaussian_mlp(;idim=3, hdim=64, depth=3, odim=3, activation="relu") where T<:Int = create_gaussian_mlp(idim, hdim, depth, odim, activation)

"""
    _create_chain(ins_, outs_, layers, activations)

Internal helper to build a Flux.Chain from layer specifications.
Zips input dimensions, output dimensions, layer types, and activations,
constructs each layer, and chains them together.
"""
function _create_chain(ins_::Vector, outs_::Vector, layers::Vector, activations::Vector)
    @assert length(ins_)==length(outs_)==length(layers)==length(activations)
    chain = map(zip(layers, ins_, outs_, activations)) do iter
        layer, in, out, activation = iter
        layer(in, out, activation)
    end
    Flux.Chain(chain...)
end




"""
    SplitLayer{M, S}

Layer that splits input into two parallel outputs via separate sub-networks.
Commonly used for Gaussian distributions (mean and std/log-std).

# Fields
- `μ::M`: network producing mean output.
- `σ::S`: network producing std/variance output.
"""
struct SplitLayer{M, S}
    μ::M
    σ::S
end

Flux.@layer SplitLayer

"""
    (m::SplitLayer)(x)

Apply both sub-networks to input and return tuple of outputs.

# Arguments
- `x::AbstractArray{<:AbstractFloat}`: input tensor.

# Returns
- `Tuple{AbstractArray, AbstractArray}`: tuple `(μ(x), σ(x))`.
"""
(m::SplitLayer)(x::AbstractArray{<: AbstractFloat}) = (m.μ(x), m.σ(x))

"""
    SplitLayer(in, out, acts)

Construct a `SplitLayer` with dual dense projections.

# Arguments
- `in::Int`: input feature dimension.
- `out::Union{Int, Tuple{Int, Int}}`: output dimension (scalar broadcasts to `(out, out)`).
- `acts::Tuple{Function, Function}`: activation functions for `(μ, σ)` branches.

# Returns
- `SplitLayer`: initialized split layer with dual `Dense` branches.
"""
### Constructors | Dense version
SplitLayer(in::Int, out::Int, acts::NTuple{2, Function}) = SplitLayer(in, (out, out), acts)
function SplitLayer(in::Int, out::NTuple{2, Int}, acts::NTuple{2, Function})
	SplitLayer(
		Flux.Dense(in, out[1], acts[1]),
		Flux.Dense(in, out[2], acts[2])
	)
end
