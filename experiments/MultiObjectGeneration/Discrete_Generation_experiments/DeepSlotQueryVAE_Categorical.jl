using Revise
using DrWatson
@quickactivate

using Random
using Statistics
using MLUtils
using ProgressBars
using Flux
using Optimisers
using CUDA

using GenerativeMIL


const N_MAX = 12
const DIGITS = 1:10

# datset is just random digits (one-hot later) from 1 to 10 in bags of cardinality from 4 up to 12
function make_bag_digit_dataset(n_bags::Int, N_max::Int, digits=1:10; kwargs...)
    x = zeros(Float32, length(digits), N_max, n_bags)
    mask = falses(1, N_max, n_bags)
    labels = Vector{Vector{Int}}(undef, n_bags)
    for b in 1:n_bags
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


# --- reconstruction sanity checks on hand-picked bags -----------------------
# Unlike NaiveSetModel, DeepSlotQueryVAE has its own exist_head, so we don't get to
# assume x̂_mask == x_mask -- existence of each of the n_slots outputs is read off the
# predicted logits_exist (threshold at 0, i.e. sigmoid > 0.5). The query slots
# themselves (m.queries) are fixed learned parameters, not resampled noise like in
# NaiveSetModel, so the only stochasticity across repeated calls comes from the latent
# z sample -- we still draw several samples for a robust estimate.
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

function reconstruct_bag(model::DeepSlotQueryVAE, digits::AbstractVector{<:Integer}; digits_alphabet=DIGITS, N_max::Int=N_MAX, device::Function=cpu, exist_threshold::Real=0)
    n = length(digits)
    n <= N_max || throw(ArgumentError("bag of length $n exceeds N_max=$N_max"))

    x = zeros(Float32, length(digits_alphabet), N_max, 1)
    mask = falses(1, N_max, 1)
    x[:, 1:n, 1] .= Flux.onehotbatch(digits, digits_alphabet)
    mask[1, 1:n, 1] .= true

    x̂, logits_exist, _, _ = model(device(x), device(mask))
    x̂, logits_exist = Array(x̂), Array(logits_exist)
    existing = findall(>(exist_threshold), vec(logits_exist[1, :, 1]))  # slots the model claims exist
    return [digits_alphabet[argmax(view(x̂, :, i, 1))] for i in existing]
end

function evaluate_reconstruction(model::DeepSlotQueryVAE, digits::AbstractVector{<:Integer}; digits_alphabet=DIGITS, N_max::Int=N_MAX, device::Function=cpu, n_samples::Int=20, exist_threshold::Real=0)
    true_sorted = sort(collect(digits))
    predictions = [reconstruct_bag(model, digits; digits_alphabet, N_max, device, exist_threshold) for _ in 1:n_samples]
    exact_match_rate = mean(sort(p) == true_sorted for p in predictions)
    mean_element_accuracy = mean(multiset_overlap(p, digits) / length(digits) for p in predictions)
    mean_predicted_cardinality = mean(length.(predictions))
    return (; input=collect(digits), predictions, exact_match_rate, mean_element_accuracy, mean_predicted_cardinality)
end

function print_reconstruction_report(model::DeepSlotQueryVAE, digits::AbstractVector{<:Integer}; kwargs...)
    r = evaluate_reconstruction(model, digits; kwargs...)
    println("  input: ", r.input, "  (sorted: ", sort(r.input), ", n=$(length(r.input)))")
    for (i, p) in enumerate(r.predictions[1:min(end, 5)])
        tag = sort(p) == sort(r.input) ? "✓" : "✗"
        println("    sample $i: ", p, "  $tag")
    end
    println("  exact_match_rate=$(r.exact_match_rate)  mean_element_accuracy=$(r.mean_element_accuracy)  mean_predicted_cardinality=$(r.mean_predicted_cardinality)")
    return r
end

const TEST_CASE_1 = [1, 7, 1, 2]
const TEST_CASE_2 = collect(1:8)
const TEST_CASE_3 = [9, 9, 5, 2, 9, 3, 6 , 5]


args = (;
    embed_dim = length(DIGITS),
    hidden_dim = 64,
    heads = 4,
    n_slots = N_MAX,   # max number of objects the model can predict per bag
    z_dim = 16,
    m_z = 1,           # number of latent summary tokens produced by the pooling encoder
    n_layers = 2,      # stacked self+cross-attention rounds in the slot decoder
    β = 0.01f0,
    λ_exist = 2f0,
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


model = DeepSlotQueryVAE(args.embed_dim, args.hidden_dim, args.heads, args.n_slots, args.z_dim, args.m_z, args.n_layers);
#x,m = first(dataloaders.train)
#elbo_with_logging(model, x, m, pairwise_logitcrossentropy; β=args.β, λ_exist=args.λ_exist)  # sanity check: forward pass works
#optim_step(model, (x,m), Optimisers.setup(AdamW(; eta=1e-3, lambda=1e-4), model), pairwise_logitcrossentropy; β=args.β, λ_exist=args.λ_exist)

#cu_model = cu(model);
#elbo_with_logging(cu_model, cu(x), cu(m), pairwise_logitcrossentropy; β=args.β, λ_exist=args.λ_exist)  # sanity check: forward pass works
#optim_step(cu_model, (x,m), Optimisers.setup(AdamW(; eta=1e-3, lambda=1e-4), cu_model), pairwise_logitcrossentropy, cu; β=args.β, λ_exist=args.λ_exist)

model = cu(model);
opt = Optimisers.setup(AdamW(; eta=1e-3, lambda=1e-4), model);

for epoch in 1:args.epochs
    logs = nothing
    for batch in tqdm(CuIterator(dataloaders.train))
        global model, opt # top-level nested-loop reassignment is ambiguous soft scope otherwise (Julia gotcha)
        model, opt, logs = optim_step(model, batch, opt, pairwise_logitcrossentropy, identity; β=args.β, λ_exist=args.λ_exist)
    end
    vlogs, _ = valid_step(model, dataloaders.valid, pairwise_logitcrossentropy; β=args.β, λ_exist=args.λ_exist, device=cu)
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
