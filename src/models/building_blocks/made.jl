struct MaskedDense
    dense
    mask
end

Flux.@functor MaskedDense
Flux.trainable(m::MaskedDense) = (m.dense,)

function (m::MaskedDense)(x::AbstractArray{<:Real, 2})
    W, b, σ = m.dense.weight, m.dense.bias, m.dense.σ
    return σ.((W .* m.mask) * x .+ b)
end

function (m::MaskedDense)(x::AbstractArray{<:Real, 3})
    # this reshapeing is acutualy used in Flux implementation as well :-)
    # https://github.com/FluxML/Flux.jl/blob/master/src/layers/basic.jl#L175-L177
    W, b, σ = m.dense.weight, m.dense.bias, m.dense.σ
    sizex = size(x)
    x = reshape(x, sizex[1], :)
    x = σ.((W .* m.mask) * x .+ b)
    return reshape(x, :, sizex[2:end]...)
end
 

struct MaskedGaussian
    μ::MaskedDense
    Σ::MaskedDense
end

Flux.@functor MaskedGaussian
Flux.trainable(m::MaskedGaussian) = (m.μ,m.Σ)

(m::MaskedGaussian)(x::AbstractArray{<:Real}) = (m.μ(x), m.Σ(x))


function get_ordering(D, K; seed=nothing)
    @assert K >= (D-1)
    core_ordering = collect(1:D-1)
    (seed!==nothing) ? Random.seed!(seed) : nothing
    add_ordering = (K>(D-1)) ? rand(core_ordering, K-D+1) : []
    ordering = Random.shuffle(vcat(core_ordering, add_ordering))
    (seed!==nothing) ? Random.seed!() : nothing
    return ordering
end

function create_mask(weight, ordering, input_ordering; final_layer=false)
    m = ones(size(weight)...)
    mask = (final_layer) ? (m .* ordering) .> input_ordering' : (m .* ordering) .>= input_ordering' 
    return Array(mask)
end


function MADE(
    in_dim::Int, 
    hidden_sizes::AbstractArray, 
    activation::Function=identity, 
    gaussian=true,
    ordering::Union{AbstractArray, String}="natural"
    )   
    if ordering=="natural"
        ordering = collect(1:in_dim)
    end
    # 1:K-1 layers
    layers = []
    ord_in = ordering
    h_in = in_dim
    for h_out in hidden_sizes
        l = Flux.Dense(h_in, h_out, activation)
        ord_out = get_ordering(in_dim, h_out)
        mask = create_mask(l.weight, ord_out, ord_in)
        ord_in, h_in = ord_out, h_out
        push!(layers, MaskedDense(l, mask))
    end
    # Final layer
    lμ = Flux.Dense(h_in, in_dim, activation)
    mask = create_mask(lμ.weight, ordering, ord_in, final_layer=true)
    if gaussian
        lΣ = Flux.Dense(h_in, in_dim, softplus)
        push!(layers, MaskedGaussian(MaskedDense(lμ, mask), MaskedDense(lΣ, mask)))
    else
        push!(layers, MaskedDense(lμ, mask))
    end
    return Flux.Chain(layers...)
end