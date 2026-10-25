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
using Flux
using Optimisers

using GenerativeMIL
using GenerativeMIL: cardinality_from_nhat


# --- model switch -----------------------------------------------------------
# model_cfg[:model_type] -> constructor taking (; idim, model_cfg...). Add new discrete-set models here.
const MODEL_CONSTRUCTORS = Dict(
    "discretebagvae" => discretebagvae_constructor_from_named_tuple,
    "discretebagae" => discretebagvae_constructor_from_named_tuple,  # same model, deterministic (gaussian_mlp=false, use_elbo=false); separate name -> separate output dir
    # "dsqvae" => dsqvae_constructor_from_named_tuple, # TODO: needs a kwargs constructor for DeepSlotQueryVAE
)

function build_model(model_type::AbstractString, model_cfg; idim::Int)
    key = lowercase(model_type)
    haskey(MODEL_CONSTRUCTORS, key) || error("Unknown model_type: $model_type (known: $(join(keys(MODEL_CONSTRUCTORS), ", ")))")
    return MODEL_CONSTRUCTORS[key](; idim=idim, dict2nt(model_cfg)...)
end


# --- reconstruction sanity checks on hand-picked bags -----------------------
# The decoder's query `q` is fresh Gaussian noise on every forward pass, so a single
# reconstruction is not representative — we draw several stochastic samples and report both
# exact-multiset match rate and a partial credit score based on multiset overlap (max bipartite
# matching under label equality, which for exact-equality edges is just min(count) per label).
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

# gt_card=true: decode against the true cardinality (n slots) but still report what the head would
# have predicted, for a "how good would generation-time cardinality be" diagnostic. gt_card=false:
# decode against the head's own predicted cardinality (via the same pred_card_mask used at
# generation time), so the reported digits are exactly what unconditional generation would produce.
function reconstruct_bag(model::DiscreteBagVAE, digits::AbstractVector{<:Integer}; n_classes::Int, N_max::Int, device::Function=cpu, gt_card::Bool=true)
    n = length(digits)
    n <= N_max || throw(ArgumentError("bag of length $n exceeds N_max=$N_max"))

    x = zeros(Float32, n_classes, N_max, 1)
    mask = falses(1, N_max, 1)
    x[:, 1:n, 1] .= Flux.onehotbatch(digits, 1:n_classes)
    mask[1, 1:n, 1] .= true
    Flux.testmode!(model, true)
    x̂, _, n̂ = model(device(x), device(mask); gt_card)
    Flux.testmode!(model, false)
    x̂ = Array(x̂)
    pred_cardinality = cardinality_from_nhat(n̂, N_max)[1]  # single bag -> scalar

    n_out = gt_card ? n : pred_cardinality
    pred_digits = [argmax(view(x̂, :, i, 1)) for i in 1:n_out]
    return pred_digits, pred_cardinality
end

function evaluate_reconstruction(model::DiscreteBagVAE, digits::AbstractVector{<:Integer}; n_samples::Int=20, gt_card::Bool=true, kwargs...)
    true_sorted = sort(collect(digits))
    results = [reconstruct_bag(model, digits; gt_card, kwargs...) for _ in 1:n_samples]
    predictions = first.(results)
    pred_cardinalities = last.(results)
    exact_match_rate = mean(sort(p) == true_sorted for p in predictions)
    mean_element_accuracy = mean(multiset_overlap(p, digits) / length(digits) for p in predictions)
    mean_predicted_cardinality = mean(pred_cardinalities)
    return (; input=collect(digits), gt_card, predictions, pred_cardinalities, exact_match_rate, mean_element_accuracy, mean_predicted_cardinality)
end

function print_reconstruction_report(r)
    mode = r.gt_card ? "ground-truth cardinality" : "predicted cardinality"
    println("  [$mode] input: ", r.input, "  (sorted: ", sort(r.input), ", n=$(length(r.input)))")
    for (i, p) in enumerate(r.predictions[1:min(end, 5)])
        tag = sort(p) == sort(r.input) ? "✓" : "✗"
        println("    sample $i: ", p, "  (pred_cardinality=$(r.pred_cardinalities[i]))  $tag")
    end
    println("  exact_match_rate=$(r.exact_match_rate)  mean_element_accuracy=$(r.mean_element_accuracy)  mean_predicted_cardinality=$(r.mean_predicted_cardinality)")
end

const TEST_CASES = ([1, 7, 1, 2], collect(1:8), [9, 9, 5, 2, 9, 3, 6, 5])

# runs every test case that fits the dataset (alphabet size, max cardinality) in both decoding modes
function test_case_reports(model::DiscreteBagVAE; n_classes::Int, N_max::Int, device::Function=cpu, verbose::Bool=true)
    reports = []
    for tc in TEST_CASES
        (length(tc) <= N_max && maximum(tc) <= n_classes) || continue
        for gt_card in (true, false)
            r = evaluate_reconstruction(model, tc; gt_card, n_classes, N_max, device)
            verbose && print_reconstruction_report(r)
            push!(reports, r)
        end
    end
    return reports
end


# --- config helpers ---------------------------------------------------------
function load_cfg(path::String)
    yaml = YAML.load_file(path; dicttype=Dict{Symbol,Any})
    return Dict(Symbol(k) => v for (k, v) in yaml)
end

function resolve_activation(x)
    x isa Function && return x
    return eval(Symbol(x))
end

# YAML floats are Float64 -> keep loss weights Float32 so they don't promote the loss
_f32(v) = v isa AbstractFloat ? Float32(v) : v

function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "config_file"
            arg_type = String
            default = joinpath(@__DIR__, "configs", "test_configs", "discretebagvae_test1.yml")
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

    model_type = get(model_cfg, :model_type, get(model_cfg, :model_name, "discretebagvae"))

    data_cfg[:seed] = args[:seed]
    train_cfg[:max_train_time] = Int((args[:time_limit] - 0.5) * 3600)
    args[:epochs] > 0 && (train_cfg[:epochs] = args[:epochs])
    !isempty(args[:model_dir]) && (train_cfg[:model_dir] = args[:model_dir])
    if !isdir(train_cfg[:model_dir])
        train_cfg[:model_dir] = datadir("GenExperiments", "$(data_cfg[:dataset])", model_type, "seed=$(args[:seed])", "$(train_cfg[:model_dir])_ID-$(lpad_number(args[:ui], Int(1e5)))" )
    end

    model_cfg[:activation] = resolve_activation(get(model_cfg, :activation, "relu"))

    dataloaders = create_dataloaders(data_cfg; batch_size=get(train_cfg, :batch_size, 128), x_only=true)
    idim = size(first(dataloaders[:train])[1], 1)   # batch = (x, x_mask); x is (n_classes, N_max, bs)
    N_max = size(first(dataloaders[:train])[1], 2)

    model = build_model(model_type, model_cfg; idim=idim)
    lr = get(train_cfg, :lr, 1f-3)
    optimiser = Optimisers.AdamW(; eta=lr, lambda = get(train_cfg, :weight_decay, 0));

    loss_cfg = get(train_cfg, :loss_function, "pairwise_logitcrossentropy")
    loss_function = create_loss_function(loss_cfg)

    beta_scheduler_cfg = get(train_cfg, :beta_annealer, get(train_cfg, :beta_anealer, get(train_cfg, :beta, 1f0)))
    beta_scheduler = create_beta_scheduler(beta_scheduler_cfg)
    lr_scheduler_cfg = get(train_cfg, :lr_scheduler, nothing)
    lr_scheduler = create_lr_scheduler(lr_scheduler_cfg, lr, get(train_cfg, :epochs, 1000))

    # model-specific loss weights / switches (e.g. λ, use_elbo for DiscreteBagVAE; λ_exist for DSQVAE),
    # forwarded by train_model! to optim_step and valid_step
    loss_kwargs = (; (Symbol(k) => _f32(v) for (k, v) in get(train_cfg, :loss_kwargs, Dict()))...)

    train_kwargs = (;
        use_gpu = get(train_cfg, :use_gpu, true),
        model_dir = train_cfg[:model_dir],
        verbose = get(train_cfg, :verbose, false),
        valid_check_interval = get(train_cfg, :valid_check_interval, 1000),
        validation_check_after_epoch = get(train_cfg, :validation_check_after_epoch, true),
        checkpoint_interval_epochs = get(train_cfg, :checkpoint_interval_epochs, 10),
        epochs = get(train_cfg, :epochs, 1000),
        early_stopping = get(train_cfg, :early_stopping, true),
        patience = get(train_cfg, :patience, 10^4),
        max_train_time = get(train_cfg, :max_train_time, Int(23.5 * 3600)),
        grad_skip = get(train_cfg, :grad_skip, false),
        validation_verbose = get(train_cfg, :validation_verbose, false),
        save_val_predictions = get(train_cfg, :save_val_predictions, false),
        val_prediction_count = get(train_cfg, :val_prediction_count, 16),
        val_prediction_interval_epochs = get(train_cfg, :val_prediction_interval_epochs, nothing),
        val_prediction_dirname = get(train_cfg, :val_prediction_dirname, "val_predictions"),
    )

    # save config as json
    mkpath(train_kwargs.model_dir)
    open(joinpath(train_kwargs.model_dir, "config.json"), "w") do io
        JSON3.pretty(io, (;train_cfg=train_cfg, model_cfg=cfg[:model], data_cfg=data_cfg, train_kwargs=train_kwargs, loss_kwargs=loss_kwargs))
    end

    train_time = @elapsed result = train_model!(
        model,
        (train=dataloaders[:train], valid=dataloaders[:valid]),
        optimiser;
        loss_function = loss_function,
        β_scheduler = beta_scheduler,
        lr_scheduler = lr_scheduler,
        train_kwargs...,
        loss_kwargs...
    );

    # save model and opt state
    model_state = Flux.state(result.model|>cpu);
    opt_state = Flux.state(result.opt|>cpu);
    model_state_dir = joinpath(train_kwargs.model_dir, "model_state")
    mkpath(model_state_dir)
    jldsave(joinpath(model_state_dir, "model_state_final.jld2"), model_state = model_state, opt_state = opt_state)

    serialize(joinpath(train_kwargs.model_dir, "history.jls"), result.history)
    @info "Saved training history" file=joinpath(train_kwargs.model_dir, "history.jls")

    run_config_file = joinpath(train_kwargs.model_dir, "run_config.jls")
    serialize(run_config_file, (
        args = args,
        data_cfg = data_cfg,
        model_cfg = model_cfg,
        train_cfg = train_cfg,
        loss_cfg = loss_cfg,
        loss_kwargs = loss_kwargs,
        train_kwargs = train_kwargs,
        beta_scheduler_cfg = beta_scheduler_cfg,
        lr_scheduler_cfg = lr_scheduler_cfg,
        train_time = train_time,
    ))
    @info "Saved run configuration" file=run_config_file

    # test-set reconstructions (note: reconstruction_check only forwards β, so ℒ there uses the
    # model's default loss weights -- compare the individual components)
    _device = (train_kwargs.use_gpu && CUDA.functional()) ? cu : cpu
    results_dir = joinpath(train_kwargs.model_dir, "results")
    mkpath(results_dir)
    β_final = Float32(beta_scheduler(train_kwargs.epochs))
    out_final = reconstruction_check(result.model, dataloaders[:test], loss_function; β=β_final, device=_device)
    out_es = reconstruction_check(_device(result.best_model), dataloaders[:test], loss_function; β=β_final, device=_device)
    serialize(joinpath(results_dir, "reconstructions_final.jls"), out_final)
    serialize(joinpath(results_dir, "reconstructions_ES_best.jls"), out_es)
    @info "Saved reconstructions into " folder=results_dir

    # hand-picked bag diagnostics (DiscreteBagVAE-specific)
    if result.model isa DiscreteBagVAE
        n_classes = idim
        println("-- test-case reconstructions, final model --")
        tc_final = test_case_reports(result.model; n_classes, N_max, device=_device)
        println("-- test-case reconstructions, ES-best model --")
        tc_es = test_case_reports(_device(result.best_model); n_classes, N_max, device=_device)
        serialize(joinpath(results_dir, "test_cases.jls"), (final=tc_final, ES_best=tc_es))
    end

    @info "Training finished"
    return result
end

main()
