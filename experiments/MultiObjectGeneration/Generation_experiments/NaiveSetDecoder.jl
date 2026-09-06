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

using GenerativeMIL
using GenerativeMIL.NNlib: logsoftmax, batched_mul, batched_transpose
import GenerativeMIL: elbo_with_logging, optim_step, valid_step, TransformerDecoder

using CUDA

const N_MAX = 12
const DIGITS = 1:10

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



# x̂, x: (D, N) → (N, N) with CE[i,j] = crossentropy(x̂[:,i], x[:,j])
# x̂, x: (D, N, BS) → (N, N, BS), batched per bag (no cross-bag pairs)
pairwise_logitcrossentropy(x̂::AbstractMatrix, x::AbstractMatrix) = -logsoftmax(x̂; dims=1)' * x

function pairwise_logitcrossentropy(x̂::AbstractArray{T,3}, x::AbstractArray{T,3}) where T <: AbstractFloat
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
    h = m.encoder(x, x_mask)                    # (dₕ, m_z, bs) — mask padding out of the pooling attention
    μ_z, Σ_z = m.z_prior(h)                     # (d_z, m_z, bs)
    z = μ_z + Σ_z .* MLUtils.randn_like(μ_z)    # (d_z, m_z, bs)
    h = m.z_to_hidden(z)                        # (dₕ, m_z, bs)
    q = MLUtils.randn_like(h, (size(h, 1), n, bs))             # (dₕ, n, bs)
    x̂ = m.decoder(q, h, x_mask)                 # (dₕ, n, bs) — masked self/cross-attention over query slots
    x̂ = m.output_head(x̂)                        # (dₓ, n, bs)
    return x̂, μ_z, Σ_z
end



function elbo_with_logging(model::NaiveSetModel, x::AbstractArray{T, 3}, x_mask::AbstractArray{Bool, 3}, logpdf::Function=pairwise_logitcrossentropy; β=1f0, kwargs...) where T <: AbstractFloat
    dₓ, n, bs = size(x)
    x̂, μ_z, Σ_z = model(x, x_mask)
    # KL divergence
    ℒₖₗ = GenerativeMIL.kl_divergence(μ_z, Σ_z) |> mean
    # reconstruction loss
    C = logpdf(x̂, x) # (N, N, BS)
    matched_indices, _ = Zygote.@ignore hungarian_match(C, x_mask, x_mask)
    n_matched = length(matched_indices)
    ℒ_rec = n_matched > 0 ? mean(C[matched_indices]) : zero(T)
    # total objective
    ℒ = ℒ_rec + β * ℒₖₗ
    # logging
    logs = (ℒ = ℒ, ℒ_rec=ℒ_rec, ℒₖₗ=ℒₖₗ)
    return ℒ, logs
end

function optim_step(model::NaiveSetModel, batch::Tuple{X, M}, opt::NamedTuple, logpdf; β=1f0, kwargs...) where {X <: AbstractArray{<:AbstractFloat,3}, M <: AbstractArray{Bool,3}}
    x, x_mask = batch
    (loss, logs), (∇model,) = Zygote.withgradient(model) do m
        elbo_with_logging(m, x, x_mask, logpdf; β=β, kwargs...)
    end
    #return loss, logs, ∇model
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

function valid_step(model::NaiveSetModel, dataloader::DataLoader, logpdf; β=1f0, device::Function=cpu, kwargs...)
    ℒ, ℒ_rec, ℒₖₗ = 0f0, 0f0, 0f0
    for batch in dataloader
        x, x_mask = length(batch) == 3 ? (batch[1], batch[2]) : batch # TODO: make it more robust to different batch formats
        x, x_mask = device(x), device(x_mask)
        loss, logs = elbo_with_logging(model, x, x_mask, logpdf; β=β, kwargs...)

        ℒ += loss
        ℒ_rec += logs.ℒ_rec
        ℒₖₗ += logs.ℒₖₗ
    end

    n = length(dataloader)
    logs = (; ℒᵥ = ℒ/n, ℒᵥ_rec = ℒ_rec/n, ℒᵥₖₗ = ℒₖₗ/n,)
    return logs, ℒ/n
end


# --- reconstruction sanity checks on hand-picked bags -----------------------
# The decoder's query `q` (line ~100) is fresh Gaussian noise on every forward
# pass, so a single reconstruction is not representative — we draw several
# stochastic samples and report both exact-multiset match rate and a partial
# credit score based on multiset overlap (max bipartite matching under label
# equality, which for exact-equality edges is just min(count) per label).
function multiset_overlap(a, b)
    counts_b = Dict{eltype(b),Int}()
    for v in b
        counts_b[v] = get(counts_b, v, 0) + 1
    end
    overlap = 0
    for v in unique(a)
        overlap += min(count(==(v), a), get(counts_b, v, 0))
    end
    return overlap
end

function reconstruct_bag(model::NaiveSetModel, digits::AbstractVector{<:Integer}; digits_alphabet=DIGITS, N_max::Int=N_MAX, device::Function=cpu)
    n = length(digits)
    n <= N_max || throw(ArgumentError("bag of length $n exceeds N_max=$N_max"))

    x = zeros(Float32, length(digits_alphabet), N_max, 1)
    mask = falses(1, N_max, 1)
    x[:, 1:n, 1] .= Flux.onehotbatch(digits, digits_alphabet)
    mask[1, 1:n, 1] .= true

    x̂, _, _ = model(device(x), device(mask))
    x̂ = Array(x̂)
    return [digits_alphabet[argmax(view(x̂, :, i, 1))] for i in 1:n]  # only the n slots we actually gave existence to
end

function evaluate_reconstruction(model::NaiveSetModel, digits::AbstractVector{<:Integer}; digits_alphabet=DIGITS, N_max::Int=N_MAX, device::Function=cpu, n_samples::Int=20)
    true_sorted = sort(collect(digits))
    predictions = [reconstruct_bag(model, digits; digits_alphabet, N_max, device) for _ in 1:n_samples]
    exact_match_rate = mean(sort(p) == true_sorted for p in predictions)
    mean_element_accuracy = mean(multiset_overlap(p, digits) / length(digits) for p in predictions)
    return (; input=collect(digits), predictions, exact_match_rate, mean_element_accuracy)
end

function print_reconstruction_report(model::NaiveSetModel, digits::AbstractVector{<:Integer}; kwargs...)
    r = evaluate_reconstruction(model, digits; kwargs...)
    println("  input: ", r.input, "  (sorted: ", sort(r.input), ")")
    for (i, p) in enumerate(r.predictions[1:min(end, 5)])
        tag = sort(p) == sort(r.input) ? "✓" : "✗"
        println("    sample $i: ", p, "  $tag")
    end
    println("  exact_match_rate=$(r.exact_match_rate)  mean_element_accuracy=$(r.mean_element_accuracy)")
    return r
end

const TEST_CASE_1 = [1, 7, 1, 2]
const TEST_CASE_2 = collect(1:8)
const TEST_CASE_3 = [9, 9, 5, 2, 9, 3, 6 , 5]


args = (;
    dₓ = length(DIGITS),
    hidden_dim = 64,
    heads = 4,
    z_dim = 16,
    m_z = 1,       # NEW: number of latent summary tokens (was implicitly 1 in first_test.jl)
    n_layers = 3,  # NEW: number of stacked self+cross-attention rounds (was implicitly 1)
    att_layers = 2, # NEW: number of stacked self-attention rounds in the decoder (was implicitly 1)
    β = 0.01f0,
    epochs = 100,
    n_train_batches = 8000,
    n_valid_batches = 800,
)



x_train, mask_train, labels_train = make_bag_digit_dataset(args.n_train_batches, N_MAX, DIGITS);
x_valid, mask_valid, labels_valid = make_bag_digit_dataset(args.n_valid_batches, N_MAX, DIGITS);

dataloaders = (
    train = DataLoader((x_train, mask_train), batchsize=128, shuffle=true, partial=true),
    valid = DataLoader((x_valid, mask_valid), batchsize=128, shuffle=false, partial=true),
)


#vae = NaiveSetModel(args.dₓ, args.hidden_dim, args.m_z, args.z_dim, args.heads, args.n_layers, args.att_layers)  
#x, m = first(dataloaders.train)
#x̂, μ, Σ = vae(x, m);
#(x̂, μ, Σ) .|> size
#pairwise_logitcrossentropy(x̂[:,:,1], x[:,:,1])       # (N, N)
#C = pairwise_logitcrossentropy(x̂, x)                 # (N, N, bs)

#elbo_with_logging(vae, x, m, pairwise_logitcrossentropy; β=args.β)


model = NaiveSetModel(args.dₓ, args.hidden_dim, args.m_z, args.z_dim, args.heads, args.n_layers, args.att_layers)  
model = cu(model)
opt = Optimisers.setup(AdamW(; eta=1e-3, lambda=1e-4), model);

for epoch in 1:args.epochs
    logs = nothing
    for batch in tqdm(CuIterator(dataloaders.train))
        global model, opt # top-level nested-loop reassignment is ambiguous soft scope otherwise (Julia gotcha)
        model, opt, logs = optim_step(model, batch, opt, pairwise_logitcrossentropy; β=args.β)
    end
    vlogs, _ = valid_step(model, dataloaders.valid, pairwise_logitcrossentropy; β=args.β, device=cu)
    println("Epoch $epoch | train: $(logs) | valid: $(vlogs)")

    if epoch % 10 == 0 || epoch == args.epochs
        println("-- reconstruction check: $TEST_CASE_1 --")
        print_reconstruction_report(model, TEST_CASE_1; device=cu)
        println("-- reconstruction check: $TEST_CASE_2 --")
        print_reconstruction_report(model, TEST_CASE_2; device=cu)
        println("-- reconstruction check: $TEST_CASE_3 --")
        print_reconstruction_report(model, TEST_CASE_3; device=cu)
    end
end


