using Revise
using DrWatson
@quickactivate

using ArgParse
using Random
using Statistics
using Serialization
using YAML
using JLD2, JSON3
using OrderedCollections

using CUDA
using MLUtils
using ProgressBars
using Flux
using Zygote
using Optimisers
using Hungarian
using Distances

using GenerativeMIL
using GenerativeMIL.NNlib: logsoftmax, batched_mul, batched_transpose


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
function reconstruct_bag(model::DiscreteBagVAE, digits::AbstractVector{<:Integer}; digits_alphabet=DIGITS, N_max::Int=N_MAX, device::Function=cpu, gt_card::Bool=true)
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

function evaluate_reconstruction(model::DiscreteBagVAE, digits::AbstractVector{<:Integer}; digits_alphabet=DIGITS, N_max::Int=N_MAX, device::Function=cpu, n_samples::Int=20, gt_card::Bool=true)
    true_sorted = sort(collect(digits))
    results = [reconstruct_bag(model, digits; digits_alphabet, N_max, device, gt_card) for _ in 1:n_samples]
    predictions = first.(results)
    pred_cardinalities = last.(results)
    exact_match_rate = mean(sort(p) == true_sorted for p in predictions)
    mean_element_accuracy = mean(multiset_overlap(p, digits) / length(digits) for p in predictions)
    mean_predicted_cardinality = mean(pred_cardinalities)
    return (; input=collect(digits), predictions, pred_cardinalities, exact_match_rate, mean_element_accuracy, mean_predicted_cardinality)
end

function print_reconstruction_report(model::DiscreteBagVAE, digits::AbstractVector{<:Integer}; gt_card::Bool=true, kwargs...)
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





dict2nt(x) = (; (Symbol(k) => v for (k, v) in x)...)

function load_cfg(path::String)
    yaml = YAML.load_file(path; dicttype=Dict{Symbol,Any})
    return Dict(Symbol(k) => v for (k, v) in yaml)
end

function resolve_activation(x)
    x isa Function && return x
    return eval(Symbol(x))
end

function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "config_file"
            arg_type = String
            default = joinpath(@__DIR__, "configs", "test_configs", "cd_setvae_test1.yml")
            help = "YAML configuration file"
        "seed"
            arg_type = Int
            default = 1
            help = "random seed"
        "time_limit"
            arg_type = Int
            default = 24
            help = "training time budget in hours"
        "model_dir"
            arg_type = String
            default = ""
            help = "optional output directory override"
        "epochs"
            arg_type = Int
            default = -1
            help = "optional epoch override"
        "ui"
            arg_type = Int
            default = Int(rand(1:10^6))
            help = "optional unique identifier for this run, used for naming output directory if model_dir is not set"
    end

    args = parse_args(ARGS, s; as_symbols=true)
    cfg = load_cfg(args[:config_file])

    data_cfg = Dict{Symbol,Any}(cfg[:data])
    model_cfg = Dict{Symbol,Any}(cfg[:model])
    train_cfg = Dict{Symbol,Any}(cfg[:train])

    model_name = model_cfg[:model_name]

    data_cfg[:seed] = args[:seed]
    train_cfg[:max_train_time] = Int((args[:time_limit] - 0.5) * 3600)
    args[:epochs] > 0 && (train_cfg[:epochs] = args[:epochs])
    !isempty(args[:model_dir]) && (train_cfg[:model_dir] = args[:model_dir])
    if !isdir(train_cfg[:model_dir])
        train_cfg[:model_dir] = datadir("GenExperiments", "$(data_cfg[:dataset])", model_cfg[:model_name] , "seed=$(args[:seed])", "$(train_cfg[:model_dir])_ID-$(lpad_number(args[:ui], Int(1e5)))" )
    end

    model_cfg[:activation] = resolve_activation(model_cfg[:activation])
    model_cfg[:output_activation] = resolve_activation(get(model_cfg, :output_activation, "identity"))




end














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
    gaussian_mlp = true, # if false, z_prior is just a linear layer (no Σ_z) and z is deterministic
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


model = DiscreteBagVAE(args.dₓ, args.hidden_dim, args.m_z, args.z_dim, args.heads, args.n_layers, args.att_layers, args.cp_layers, args.activation, args.dropout, args.gaussian_mlp);
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

