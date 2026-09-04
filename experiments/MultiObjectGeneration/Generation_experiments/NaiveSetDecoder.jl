using Revise
using DrWatson
@quickactivate

using Random
using Statistics
using JLD2
using MLUtils
using ProgressBars
using Flux
using Zygote
using Optimisers
using Hungarian
using Distances
using NNlib: logsoftmax, batched_mul, batched_transpose

using GenerativeMIL
import GenerativeMIL: elbo_with_logging, optim_step, valid_step, TransformerDecoder


# datset is just random digits (one-hot later) from 1 to 10 in bags of cardinality from 4 up to 12
function make_bag_digit_dataset(n_bags::Int, N_max::Int, digits=1:10; kwargs...)
    x = zeros(Float32, length(digits), N_max, n_bags)
    mask = falses(1, N_max, n_bags)
    labels = Vector{Vector{Int}}(undef, n_bags)
    for b in 1:n_bags
        # I need to sample digits from 
        len = rand(1:N_max)
        elements = rand(digits, len)
        x[:, 1:len, b] .= Flux.onehotbatch(elements, digits)
        mask[1, 1:len, b] .= true
        labels[b] = elements
    end
    return x, mask, labels
end

const N_MAX = 12
const DIGITS = 1:10

x_train, mask_train, labels_train = make_bag_digit_dataset(8000, N_MAX, DIGITS);
x_valid, mask_valid, labels_valid = make_bag_digit_dataset(800, N_MAX, DIGITS);

dataloaders = (
    train = DataLoader((x_train, mask_train), batchsize=128, shuffle=true, partial=true),
    valid = DataLoader((x_valid, mask_valid), batchsize=128, shuffle=false, partial=true),
)



struct NaiveSetModel{E<:PoolEncoder, PT<:SplitLayer, ZT<:Flux.Dense, D<:TransformerDecoder, OT<:Flux.Dense}
    encoder::E
    z_prior::PT
    z_to_hidden::ZT
    decoder::D
    output_head ::OT
end

Flux.@layer NaiveSetModel

function NaiveSetModel(dₓ::Int, dₕ::Int, m_z::Int, d_z::Int, n_heads::Int, n_layers::Int, att_layers::Int, activation::Function=relu)

    encoder = PoolEncoder(
        create_mlp(dₓ, dₕ, n_layers, dₕ, activation),
        PMA(m_z, dₕ, n_heads),
        create_mlp(dₕ, dₕ, n_layers, dₕ, activation)
    )
    z_prior = SplitLayer(dₕ, (d_z, d_z),(identity, Flux.softplus))
    decoder = TransformerDecoder(
        [MultiheadAttentionBlock(dₕ, n_heads; attention_fn=attention) for _ in 1:att_layers],
        [MultiheadAttentionBlock(dₕ, n_heads; attention_fn=attention) for _ in 1:att_layers]
    )
    z_to_hidden = Flux.Dense(d_z, dₕ)
    output_head = Flux.Dense(dₕ, dₓ)

    return NaiveSetModel(encoder, z_prior, z_to_hidden, decoder, output_head)
end


function (m::NaiveSetModel)(x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}) where T <: AbstractFloat
    dₓ, n, bs = size(x)                         # (dₓ, n, bs)
    h = m.encoder(x)                            # (dₕ, m_z, bs)
    μ_z, Σ_z = m.z_prior(h)                     # (d_z, m_z, bs)
    z = μ_z + Σ_z .* MLUtils.randn_like(μ_z)    # (d_z, m_z, bs)
    h = m.z_to_hidden(z)                        # (dₕ, m_z, bs)
    q = randn(T, size(h, 1), n, bs)             # (dₕ, n, bs)
    x̂ = m.decoder(q, h)                         # (dₕ, n, bs)
    x̂ = m.output_head(x̂)                        # (dₓ, n, bs)
    return x̂, μ_z, Σ_z
end


# x̂, x: (D, N) → (N, N) with CE[i,j] = crossentropy(x̂[:,i], x[:,j])
# x̂, x: (D, N, BS) → (N, N, BS), batched per bag (no cross-bag pairs)
pairwise_logitcrossentropy(x̂::AbstractMatrix, x::AbstractMatrix) = -logsoftmax(x̂; dims=1)' * x

function pairwise_logitcrossentropy(x̂::AbstractArray{<:Real,3}, x::AbstractArray{<:Real,3})
    logŷ = logsoftmax(x̂; dims=1)
    return -batched_mul(batched_transpose(logŷ), x)
end

# for debugging purposes
function for_pairwise_logitcrossentropy(x̂::AbstractMatrix, x::AbstractMatrix)
    # (D, N)
    out = zeros(Float32, size(x̂, 2), size(x, 2))
    for i in axes(x̂, 2)
        for j in axes(x, 2)
             out[i, j] = -sum(x[:,j] .* logsoftmax(x̂[:,i]))
        end
    end
    out
end




args = (;
    dₓ = length(DIGITS),
    hidden_dim = 64,
    heads = 4,
    z_dim = 16,
    m_z = 3,       # NEW: number of latent summary tokens (was implicitly 1 in first_test.jl)
    n_layers = 3,  # NEW: number of stacked self+cross-attention rounds (was implicitly 1)
    att_layers = 2, # NEW: number of stacked self-attention rounds in the decoder (was implicitly 1)
    β = 0.01f0,
    epochs = 100,
)

vae = NaiveSetModel(args.dₓ, args.hidden_dim, args.m_z, args.z_dim, args.heads, args.n_layers, args.att_layers)  

x, m = first(dataloaders.train)
x̂, μ, Σ = vae(x, m);
(x̂, μ, Σ) .|> size


pairwise_logitcrossentropy(x̂[:,:,1], x[:,:,1])   # (N, N)
pairwise_logitcrossentropy(x̂, x)                 # (N, N, bs)
