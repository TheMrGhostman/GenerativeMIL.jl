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

function _create_chain(ins_::Vector, outs_::Vector, layers::Vector, activations::Vector)
    @assert length(ins_)==length(outs_)==length(layers)==length(activations)
    chain = map(zip(layers, ins_, outs_, activations)) do iter
        layer, in, out, activation = iter
        layer(in, out, activation)
    end
    Flux.Chain(chain...)
end




struct SplitLayer{M, S}
    μ::M
    σ::S
end

Flux.@functor SplitLayer

(m::SplitLayer)(x::AbstractArray{<: AbstractFloat}) = (m.μ(x), m.σ(x))

### Constructors | Dense version
SplitLayer(in::Int, out::Int, acts::NTuple{2, Function}) = SplitLayer(in, (out, out), acts)
function SplitLayer(in::Int, out::NTuple{2, Int}, acts::NTuple{2, Function})
	SplitLayer(
		Flux.Dense(in, out[1], acts[1]),
		Flux.Dense(in, out[2], acts[2])
	)
end
