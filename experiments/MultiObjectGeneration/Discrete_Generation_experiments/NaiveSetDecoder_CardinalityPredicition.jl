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


# Converts the cardinality head's raw regression output n̂ (1, m_z, bs) into an integer
# cardinality per batch element: pools over the m_z latent tokens, applies the same softplus
# used in the training loss to keep the estimate non-negative, then rounds and clamps into a
# valid slot count in 1:n_max.
cardinality_from_nhat(n̂::AbstractArray{<:AbstractFloat,3}, n_max::Int) =
    clamp.(round.(Int, dropdims(mean(Flux.softplus.(Array(n̂)), dims=2), dims=(1,2))), 1, n_max)

# Builds a (1, n, bs) existence mask from the cardinality head's prediction n̂, mirroring
# x_mask's layout (first `card[b]` query slots existing, rest padding). CPU-only index
# bookkeeping (like hungarian_match) -- only meant to run at inference/generation time, never
# under gradient tracking, so callers should wrap it in Zygote.@ignore.
function pred_card_mask(n̂::AbstractArray{<:AbstractFloat,3}, n::Int, bs::Int)
    card = cardinality_from_nhat(n̂, n)
    mask = falses(1, n, bs)
    for (b, c) in enumerate(card)
        mask[1, 1:c, b] .= true
    end
    return mask
end

# Splices a Dropout(p) after every layer but the last in a Chain built by create_mlp, without
# needing to touch create_mlp itself. p<=0 is a no-op (returns chain unchanged) so existing
# callers that don't pass dropout keep building exactly the same Chain as before. Dropout layers
# are found by Flux.@layer's functor traversal like any other child, so they automatically go
# inactive outside of gradient computation (plain `model(x, mask)` calls) with no explicit
# trainmode!/testmode! calls needed.
function with_dropout(chain::Flux.Chain, p::Real)
    p <= 0 && return chain
    layers = chain.layers
    spliced = Any[]
    for (i, l) in enumerate(layers)
        push!(spliced, l)
        i < length(layers) && push!(spliced, Flux.Dropout(p))
    end
    return Flux.Chain(spliced...)
end

struct NaiveSetModelCP{E<:PoolEncoder, PT<:SplitLayer, ZT<:Flux.Dense, D<:TransformerDecoder, OT<:Flux.Dense, CT<:Flux.Chain}
    encoder::E
    z_prior::PT
    z_to_hidden::ZT
    decoder::D
    output_head ::OT
    cardinality_head::CT
end

Flux.@layer NaiveSetModelCP

function NaiveSetModelCP(dₓ::Int, dₕ::Int, m_z::Int, d_z::Int, n_heads::Int, n_layers::Int, att_layers::Int, cp_layers::Int, activation::Function=relu, dropout::Real=0.0)

    encoder = PoolEncoder(
        create_mlp(dₓ, dₕ, n_layers, dₕ, activation),
        PMA(m_z, dₕ, n_heads),
        create_mlp(dₕ, dₕ, n_layers, dₕ, activation),
    )
    z_prior = SplitLayer(dₕ, (d_z, d_z),(identity, Flux.softplus))
    decoder = TransformerDecoder(
        [MultiheadAttentionBlock(dₕ, n_heads; attention_fn=attention) for _ in 1:att_layers],
        [MultiheadAttentionBlock(dₕ, n_heads; attention_fn=attention) for _ in 1:att_layers]
    )
    z_to_hidden = Flux.Dense(d_z, dₕ)
    output_head = Flux.Dense(dₕ, dₓ)
    cardinality_head = with_dropout(create_mlp(d_z, dₕ, cp_layers, 1, activation, out_identity=true), dropout)

    return NaiveSetModelCP(encoder, z_prior, z_to_hidden, decoder, output_head, cardinality_head)
end


# gt_card controls only which mask gates the decoder's query slots: the ground-truth x_mask
# (training — the cardinality head is trained alongside via an auxiliary loss in
# elbo_with_logging, but never gates the forward pass it's trained under, see there) or the
# head's own prediction from z (generation/inference, when no ground-truth mask exists). n̂ is
# always computed and returned so callers can inspect/train it either way.
function (m::NaiveSetModelCP)(x::AbstractArray{T,3}, x_mask::AbstractArray{Bool,3}; gt_card::Bool=true) where T <: AbstractFloat
    dₓ, n, bs = size(x)                                          # (dₓ, n, bs)
    h = m.encoder(x, x_mask)                                     # (dₕ, m_z, bs) — mask padding out of the pooling attention
    μ_z, Σ_z = m.z_prior(h)                                      # (d_z, m_z, bs)
    z = μ_z + Σ_z .* MLUtils.randn_like(μ_z)                     # (d_z, m_z, bs)
    n̂ = m.cardinality_head(z)                                    # (1, m_z, bs)
    h = m.z_to_hidden(z)                                         # (dₕ, m_z, bs)
    query_mask = if gt_card
        x_mask
    else
        Zygote.@ignore begin
            mask_cpu = pred_card_mask(n̂, n, bs)
            x_mask isa CuArray ? CuArray(mask_cpu) : mask_cpu
        end
    end
    q = MLUtils.randn_like(h, (size(h,1), n, bs))                # (dₕ, n, bs)
    x̂ = m.decoder(q, h, query_mask)                              # (dₕ, n, bs) — masked self/cross-attention over query slots
    x̂ = m.output_head(x̂)                                         # (dₓ, n, bs)
    return x̂, μ_z, Σ_z, n̂
end


function elbo_with_logging(model::NaiveSetModelCP, x::AbstractArray{T, 3}, x_mask::AbstractArray{Bool, 3}, logpdf::Function=pairwise_logitcrossentropy; β=1f0, λ=1f0, kwargs...) where T <: AbstractFloat
    dₓ, n, bs = size(x)
    x̂, μ_z, Σ_z, n̂ = model(x, x_mask; gt_card=true)  # training always decodes against ground-truth cardinality
    # KL divergence
    ℒₖₗ = GenerativeMIL.kl_divergence(μ_z, Σ_z) |> mean
    # reconstruction loss
    C = logpdf(x̂, x) # (N, N, BS)
    matched_indices, _ = Zygote.@ignore hungarian_match(C, x_mask, x_mask)
    n_matched = length(matched_indices)
    ℒ_rec = n_matched > 0 ? mean(C[matched_indices]) : zero(T)
    # cardinality prediction loss: supervise n̂ (via softplus, kept non-negative) against each
    # bag's true element count (sum of its mask), not the constant padded width n
    true_card = sum(x_mask, dims=2)  # (1, 1, bs); broadcasts against n̂'s (1, m_z, bs)
    ℒ_card = Flux.mse(Flux.softplus.(n̂), Float32.(true_card))
    # total objective
    ℒ = ℒ_rec + β * ℒₖₗ + λ * ℒ_card
    # logging
    logs = (ℒ = ℒ, ℒ_rec=ℒ_rec, ℒₖₗ=ℒₖₗ, ℒ_card=ℒ_card, β=β, λ=λ)
    return ℒ, logs
end

function optim_step(model::NaiveSetModelCP, batch::Tuple{X, M}, opt::NamedTuple, logpdf; β=1f0, λ=1f0, kwargs...) where {X <: AbstractArray{<:AbstractFloat,3}, M <: AbstractArray{Bool,3}}
    x, x_mask = batch
    (loss, logs), (∇model,) = Zygote.withgradient(model) do m
        elbo_with_logging(m, x, x_mask, logpdf; β=β, λ=λ, kwargs...)
    end
    #return loss, logs, ∇model
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, logs
end

function valid_step(model::NaiveSetModelCP, dataloader::DataLoader, logpdf; β=1f0, λ=1f0, device::Function=cpu, kwargs...)
    ℒ, ℒ_rec, ℒₖₗ, ℒ_card = 0f0, 0f0, 0f0, 0f0
    Flux.testmode!(model, true)
    for batch in dataloader
        x, x_mask = length(batch) == 3 ? (batch[1], batch[2]) : batch # TODO: make it more robust to different batch formats
        x, x_mask = device(x), device(x_mask)
        loss, logs = elbo_with_logging(model, x, x_mask, logpdf; β=β, λ=λ, kwargs...)

        ℒ += loss
        ℒ_rec += logs.ℒ_rec
        ℒₖₗ += logs.ℒₖₗ
        ℒ_card += logs.ℒ_card
    end
    Flux.testmode!(model, false)
    n = length(dataloader)
    logs = (; ℒᵥ = ℒ/n, ℒᵥ_rec = ℒ_rec/n, ℒᵥₖₗ = ℒₖₗ/n, ℒᵥ_card = ℒ_card/n)
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

# gt_card=true: decode against the true cardinality (n slots, like before) but still report
# what the head would have predicted, for a "how good would generation-time cardinality be"
# diagnostic. gt_card=false: decode against the head's own predicted cardinality (via the same
# pred_card_mask used at generation time), so the reported digits are exactly what unconditional
# generation would produce.
function reconstruct_bag(model::NaiveSetModelCP, digits::AbstractVector{<:Integer}; digits_alphabet=DIGITS, N_max::Int=N_MAX, device::Function=cpu, gt_card::Bool=true)
    n = length(digits)
    n <= N_max || throw(ArgumentError("bag of length $n exceeds N_max=$N_max"))

    x = zeros(Float32, length(digits_alphabet), N_max, 1)
    mask = falses(1, N_max, 1)
    x[:, 1:n, 1] .= Flux.onehotbatch(digits, digits_alphabet)
    mask[1, 1:n, 1] .= true
    Flux.testmode!(model, true)
    x̂, _, _, n̂ = model(device(x), device(mask); gt_card)
    Flux.testmode!(model, false)
    x̂ = Array(x̂)
    pred_cardinality = cardinality_from_nhat(n̂, N_max)[1]  # single bag -> scalar

    n_out = gt_card ? n : pred_cardinality
    pred_digits = [digits_alphabet[argmax(view(x̂, :, i, 1))] for i in 1:n_out]
    return pred_digits, pred_cardinality
end

function evaluate_reconstruction(model::NaiveSetModelCP, digits::AbstractVector{<:Integer}; digits_alphabet=DIGITS, N_max::Int=N_MAX, device::Function=cpu, n_samples::Int=20, gt_card::Bool=true)
    true_sorted = sort(collect(digits))
    results = [reconstruct_bag(model, digits; digits_alphabet, N_max, device, gt_card) for _ in 1:n_samples]
    predictions = first.(results)
    pred_cardinalities = last.(results)
    exact_match_rate = mean(sort(p) == true_sorted for p in predictions)
    mean_element_accuracy = mean(multiset_overlap(p, digits) / length(digits) for p in predictions)
    mean_predicted_cardinality = mean(pred_cardinalities)
    return (; input=collect(digits), predictions, pred_cardinalities, exact_match_rate, mean_element_accuracy, mean_predicted_cardinality)
end

function print_reconstruction_report(model::NaiveSetModelCP, digits::AbstractVector{<:Integer}; gt_card::Bool=true, kwargs...)
    r = evaluate_reconstruction(model, digits; gt_card, kwargs...)
    mode = gt_card ? "ground-truth cardinality" : "predicted cardinality"
    println("  [$mode] input: ", r.input, "  (sorted: ", sort(r.input), ", n=$(length(r.input)))")
    for (i, p) in enumerate(r.predictions[1:min(end, 5)])
        tag = sort(p) == sort(r.input) ? "✓" : "✗"
        println("    sample $i: ", p, "  (pred_cardinality=$(r.pred_cardinalities[i]))  $tag")
    end
    println("  exact_match_rate=$(r.exact_match_rate)  mean_element_accuracy=$(r.mean_element_accuracy)  mean_predicted_cardinality=$(r.mean_predicted_cardinality)")
    return r
end

const TEST_CASE_1 = [1, 7, 1, 2]
const TEST_CASE_2 = collect(1:8)
const TEST_CASE_3 = [9, 9, 5, 2, 9, 3, 6 , 5]

silu(x) = x .* σ.(x)

scheduler = (
    type = "sigmoidal_cyclical",
    max_value = 0.05f0,
    beta_warmup = 0.0005f0,
    warmup_epochs = 300,
    rise_epochs = 125,
    hold_epochs = 50,
    cycles = 4,
    slope_factor = 12f0/100
)

args = (;
    dₓ = length(DIGITS),
    hidden_dim = 64,
    heads = 4,
    z_dim = 16,
    m_z = 1,       # NEW: number of latent summary tokens (was implicitly 1 in first_test.jl)
    n_layers = 3,  # NEW: number of stacked self+cross-attention rounds (was implicitly 1)
    att_layers = 2, # NEW: number of stacked self-attention rounds in the decoder (was implicitly 1)
    cp_layers = 3,  # NEW: number of layers in the cardinality prediction head (was implicitly 1)
    dropout=0.2,
    β = 0.01f0,
    λ = 0.1f0,
    epochs = 1000,
    n_train_batches = 8000,
    n_valid_batches = 800,
    #beta = beta_scheduler,
    scheduler = scheduler,
    activation = relu, #gelu, #x-> x · σ.(x)
    ui=Int(rand(1:10^6))  # optional unique identifier for this run, used for naming output directory if model_dir is not set
)




# Training bags are resampled fresh every epoch (see the loop below) instead of drawn once here,
# since make_bag_digit_dataset is just cheap random sampling — the model then never sees the same
# fixed n_train_batches bags twice, which keeps it from overfitting to one static training set.
# The validation set stays fixed across epochs so ℒᵥ is comparable epoch-to-epoch.
x_valid, mask_valid, labels_valid = make_bag_digit_dataset(args.n_valid_batches, N_MAX, DIGITS);
valid_loader = DataLoader((x_valid, mask_valid), batchsize=128, shuffle=false, partial=true)
#CyclicalSigmoidSchedule(max_value, beta_warmup, warmup_epochs, rise_epochs, hold_epochs, cycles; slope_factor=12f0/rise_epochs)
beta_scheduler = GenerativeMIL.CyclicalSigmoidSchedule(
    args.scheduler.max_value,
    args.scheduler.beta_warmup,
    args.scheduler.warmup_epochs,
    args.scheduler.rise_epochs,
    args.scheduler.hold_epochs,
    args.scheduler.cycles;
    slope_factor=args.scheduler.slope_factor
)

#beta_scheduler = x-> args.beta

#vae = NaiveSetModelCP(args.dₓ, args.hidden_dim, args.m_z, args.z_dim, args.heads, args.n_layers, args.att_layers, args.cp_layers)  
#x, m = first(dataloaders.train)
#x̂, μ, Σ = vae(x, m);   
#(x̂, μ, Σ) .|> size
#pairwise_logitcrossentropy(x̂[:,:,1], x[:,:,1])       # (N, N)
#C = pairwise_logitcrossentropy(x̂, x)                 # (N, N, bs)

#elbo_with_logging(vae, x, m, pairwise_logitcrossentropy; β=args.β)

#x̂, μ, Σ, n̂ = vae(x, m; gt_card=true)  # training always decodes against ground-truth cardinality; 
#x̂, μ, Σ, n̂ = vae(x, m; gt_card=false) 

model = NaiveSetModelCP(args.dₓ, args.hidden_dim, args.m_z, args.z_dim, args.heads, args.n_layers, args.att_layers, args.cp_layers, args.activation, args.dropout);
model = cu(model);
opt = Optimisers.setup(AdamW(; eta=1e-3, lambda=1e-4), model);

for epoch in 1:args.epochs
    logs = nothing
    β = beta_scheduler(epoch)
    x_train, mask_train, _ = make_bag_digit_dataset(args.n_train_batches, N_MAX, DIGITS)
    train_loader = DataLoader((x_train, mask_train), batchsize=128, shuffle=true, partial=true)
    Flux.trainmode!(model)
    for batch in tqdm(CuIterator(train_loader))
        global model, opt # top-level nested-loop reassignment is ambiguous soft scope otherwise (Julia gotcha)
        model, opt, logs = optim_step(model, batch, opt, pairwise_logitcrossentropy; β=β, λ=args.λ)
    end
    Flux.testmode!(model)
    vlogs, _ = valid_step(model, valid_loader, pairwise_logitcrossentropy; β=β, λ=args.λ, device=cu)
    println("Epoch $epoch | train: $(logs) | valid: $(vlogs)")

    if epoch % 10 == 0 || epoch == args.epochs
        for tc in (TEST_CASE_1, TEST_CASE_2, TEST_CASE_3)
            println("-- reconstruction check (ground-truth cardinality): $tc --")
            print_reconstruction_report(model, tc; device=cu, gt_card=true)
            println("-- reconstruction check (predicted cardinality): $tc --")
            print_reconstruction_report(model, tc; device=cu, gt_card=false)
        end
    end
end


# save model and opt state
model_state = Flux.state(model|>cpu);
opt_state = Flux.state(opt|>cpu);
model_state_dir = joinpath("B:\\Github-Repos\\GenerativeMIL.jl\\data\\MultiObjectGeneration\\NaiveSetModel_CP\\naivesetmodel_cp_ui=$(args.ui)", "model_state")
mkpath(model_state_dir)
jldsave(joinpath(model_state_dir, "model_state_final.jld2"), model_state = model_state, opt_state = opt_state, args = args)


##  relu
# Epoch 99 | train: (ℒ = 0.20067011f0, ℒ_rec = 0.060231928f0, ℒₖₗ = 51.189003f0, ℒ_card = 1.1484368f0, β = 0.0005f0, λ = 0.1f0) | valid: (ℒᵥ = 0.2541466f0, ℒᵥ_rec = 0.081563175f0, ℒᵥₖₗ = 50.53027f0, ℒᵥ_card = 1.4731833f0)
# Epoch 100 | train: (ℒ = 0.2028878f0, ℒ_rec = 0.104561254f0, ℒₖₗ = 52.397068f0, ℒ_card = 0.7212801f0, β = 0.0005f0, λ = 0.1f0) | valid: (ℒᵥ = 0.26070455f0, ℒᵥ_rec = 0.113363944f0, ℒᵥₖₗ = 52.344482f0, ℒᵥ_card = 1.2116833f0)

##  gelu 
# Epoch 99 | train: (ℒ = 0.31423312f0, ℒ_rec = 0.18404405f0, ℒₖₗ = 51.03629f0, ℒ_card = 1.0467093f0, β = 0.0005f0, λ = 0.1f0) | valid: (ℒᵥ = 0.4294079f0, ℒᵥ_rec = 0.19727376f0, ℒᵥₖₗ = 50.206543f0, ℒᵥ_card = 2.0703084f0)
# Epoch 100 | train: (ℒ = 0.43544662f0, ℒ_rec = 0.29028273f0, ℒₖₗ = 52.769905f0, ℒ_card = 1.1877896f0, β = 0.0005f0, λ = 0.1f0) | valid: (ℒᵥ = 0.41868263f0, ℒᵥ_rec = 0.19286802f0, ℒᵥₖₗ = 52.039703f0, ℒᵥ_card = 1.9979473f0)

## silu
# Epoch 99 | train: (ℒ = 0.41047603f0, ℒ_rec = 0.26918f0, ℒₖₗ = 50.639606f0, ℒ_card = 1.1597621f0, β = 0.0005f0, λ = 0.1f0) | valid: (ℒᵥ = 0.45102736f0, ℒᵥ_rec = 0.2744412f0, ℒᵥₖₗ = 50.317314f0, ℒᵥ_card = 1.514275f0)
# Epoch 100 | train: (ℒ = 0.41911846f0, ℒ_rec = 0.23857251f0, ℒₖₗ = 49.689873f0, ℒ_card = 1.5570099f0, β = 0.0005f0, λ = 0.1f0) | valid: (ℒᵥ = 0.43323693f0, ℒᵥ_rec = 0.24316442f0, ℒᵥₖₗ = 50.80605f0, ℒᵥ_card = 1.646695f0)
