# Collect already computed reconstruction evaluations into summary tables.
#
# READ-ONLY with respect to experiment data: it only reads files in data/GenExperiments
# and writes new files into experiments/EvaluationOfGenExperiments/collected_results/.
#
# Inputs (per run folder data/GenExperiments/<dataset>/<model>/seed=<s>/<run>/):
#   evaluation/evaluation_reconstruction.jls  -> per-sample metrics, (valid, test) × metric × (repeats × samples) + labels
#   run_config.jls                            -> configs used for training (β schedule, epochs, train time, ...)
#   trainlog.jsonl                            -> per-epoch logs; we take the LAST train and LAST valid record
#
# Outputs (collected_results/):
#   runs.{jls,csv}       one row per run (metadata + final training state), including runs without evaluation
#   results.{jls,csv}    one row per run × split × metric (summary statistics of the per-sample values)
#   per_class.{jls,csv}  one row per run × split × metric × class label (mean per class)
#
# Usage (from repository root or anywhere, DrWatson finds the project):
#   julia --project experiments/EvaluationOfGenExperiments/collect_reconstruction_results.jl

using DrWatson
@quickactivate

using GenerativeMIL   # needed to deserialize run_config.jls (contains activation functions etc.)
using Serialization
using Statistics
using JSON3
using ProgressBars

# ─────────────────────────────── settings ───────────────────────────────
const DATASETS = ["mnist", "shapenet_class", "shapenet_multiple_classes"]   # modelnet10 intentionally excluded
const MODELS   = ["setvae", "poolmodel", "neuralstatistician"]
const SEED     = 1
const OUT_DIR  = joinpath(@__DIR__, "collected_results")

# folder name -> name used in thesis tables
const DATASET_DISPLAY = Dict("mnist" => "mnist", "shapenet_class" => "airplane", "shapenet_multiple_classes" => "core5")

# MNIST evaluations computed before this time used the wrong data split (seed 666 instead of 1, ~64% leakage).
# Fix was made on 2026-10-09; any MNIST evaluation file older than that day must be re-evaluated.
const MNIST_SPLIT_FIX_TIME = 1791496800.0  # unix time of 2026-10-09 00:00 CEST (`date -d '2026-10-09 00:00' +%s`)

# ─────────────────────────────── helpers ───────────────────────────────

"""
Parse run folder name `<trainloss>_<model>[_<variant>]_c<idx>_ID-<uid>` into parts.
Example: `cd_setvae_e_c011_ID-123456` -> (train_loss="cd", config_name="cd_setvae_e_c011", config_idx=11)
"""
function parse_run_name(name::AbstractString)
    config_name = split(name, "_ID-")[1]
    train_loss = split(config_name, "_")[1]
    m = match(r"_c(\d+)$", config_name)
    config_idx = isnothing(m) ? missing : parse(Int, m.captures[1])
    return (; train_loss=String(train_loss), config_name=String(config_name), config_idx)
end

"""
Return the last record with mode=="train" and the last with mode=="valid" from trainlog.jsonl
(scans from the end, so it does not parse the whole file).
"""
function last_train_valid_records(path::String)
    isfile(path) || return (nothing, nothing)
    last_train, last_valid = nothing, nothing
    for line in Iterators.reverse(readlines(path))
        isempty(strip(line)) && continue
        rec = JSON3.read(line)
        mode = get(rec, :mode, "")
        if mode == "train" && isnothing(last_train)
            last_train = rec
        elseif mode == "valid" && isnothing(last_valid)
            last_valid = rec
        end
        !isnothing(last_train) && !isnothing(last_valid) && break
    end
    return (last_train, last_valid)
end

# Scalar from a JSON record, or `missing` if the key is absent (models log different KL terms).
getnum(rec, key) = (isnothing(rec) || !haskey(rec, key)) ? missing : Float64(rec[key])

"""
β schedule type and configured maximum β of a run. Forms found in run_config.jls:
- SetVAE:             Dict with :type ("step_linear", "sigmoidal", "cyclical_sigmoidal") and :max_value
- NeuralStatistician: a plain number (constant β)
- PoolModel:          no `beta_scheduler_cfg` field; train_poolmodel.jl trains without a KL term, so β is unused
"""
function beta_info(model, rc)
    model == "poolmodel" && return (beta_schedule = "none (no KL)", beta_max = missing)
    b = (isnothing(rc) || !hasproperty(rc, :beta_scheduler_cfg)) ? nothing : rc.beta_scheduler_cfg
    b isa Number && return (beta_schedule = "constant", beta_max = Float64(b))
    b isa AbstractDict && return (beta_schedule = String(string(get(b, :type, "constant"))),
                                  beta_max = Float64(get(b, :max_value, get(b, :value, NaN))))
    return (beta_schedule = missing, beta_max = missing)
end

"""
One row of the runs table. Every row has the same columns; model-specific values are `missing`.
"""
function run_row(dataset, model, run_dir)
    name = basename(run_dir)
    p = parse_run_name(name)
    eval_file = joinpath(run_dir, "evaluation", "evaluation_reconstruction.jls")
    rc_file = joinpath(run_dir, "run_config.jls")
    rc = isfile(rc_file) ? deserialize(rc_file) : nothing
    tr, va = last_train_valid_records(joinpath(run_dir, "trainlog.jsonl"))

    binfo = beta_info(model, rc)
    mc = (isnothing(rc) || !hasproperty(rc, :model_cfg)) ? Dict{Symbol,Any}() : rc.model_cfg
    n_points = (isnothing(rc) || !hasproperty(rc, :data_cfg)) ? missing : get(rc.data_cfg, :npoints, missing)
    is_ns = model == "neuralstatistician"
    kl_z = getnum(va, :ℒᵥₖₗ_z)
    return (
        dataset        = DATASET_DISPLAY[dataset],
        model          = model,
        train_loss     = p.train_loss,
        config_name    = p.config_name,
        config_idx     = p.config_idx,
        run_name       = name,
        evaluated      = isfile(eval_file),
        eval_mtime     = isfile(eval_file) ? mtime(eval_file) : missing,
        # training budget / progress
        epochs_planned = (isnothing(rc) || !hasproperty(rc, :train_kwargs)) ? missing : rc.train_kwargs.epochs,
        epoch_reached  = getnum(va, :epoch),
        train_time_h   = (isnothing(rc) || !hasproperty(rc, :train_time)) ? missing : rc.train_time / 3600,
        # β: configured schedule + its maximum (identifies the run together with the ID),
        # and the value actually in effect at the end (sigmoidal schedules may not reach beta_max)
        beta_schedule  = binfo.beta_schedule,
        beta_max       = binfo.beta_max,
        beta_final     = getnum(tr, :β),
        # final validation losses as logged during training (model-specific; not comparable across models!)
        valid_loss     = getnum(va, :ℒᵥ),
        valid_rec      = getnum(va, :ℒᵥ_rec),
        valid_kl       = getnum(va, :ℒᵥₖₗ),      # SetVAE
        valid_kl_z     = kl_z,                   # NeuralStatistician: nats PER POINT (summed over zdim, averaged over points)
        valid_kl_c     = getnum(va, :ℒᵥₖₗ_c),    # NeuralStatistician: nats PER SET (summed over cdim, averaged over sets)
        n_points       = n_points,
        valid_kl_z_set = kl_z * n_points,        # NeuralStatistician: nats per set sent through the local latents zᵢ
        # NeuralStatistician latent sizes (missing for other models)
        ns_zdim        = is_ns ? get(mc, :zdim, missing) : missing,
        ns_cdim        = is_ns ? get(mc, :cdim, missing) : missing,
        ns_hdim        = is_ns ? get(mc, :hdim, missing) : missing,
        run_dir        = run_dir,
    )
end

"""
Summary rows for one evaluated run.
`losses[metric]` is a (n_repeats × n_samples) matrix: row = repeat, column = sample.
"""
function result_rows(meta, ev)
    rows, class_rows = NamedTuple[], NamedTuple[]
    for split in (:valid, :test)
        e = getproperty(ev, split)
        labels = e.labels
        for (metric, L) in e.losses
            L = Float64.(L)
            # Sinkhorn eval (plain SinkhornGibbs in Float32) can underflow and return NaN for a few samples.
            # Drop the samples (columns) with a NaN in any repeat and record how many were dropped.
            ok = vec(.!any(isnan, L, dims=1))
            n_nan_samples = count(!, ok)
            L = L[:, ok]
            labels_ok = labels[ok]
            per_sample = vec(mean(L, dims=1))          # average over repeats -> one value per sample
            per_repeat = vec(mean(L, dims=2))          # average over samples -> one value per repeat
            sample_std = size(L, 1) > 1 ? vec(std(L, dims=1)) : fill(NaN, size(L, 2))
            q25, med, q75 = quantile(per_sample, [0.25, 0.5, 0.75])
            push!(rows, (
                dataset = meta.dataset, model = meta.model, train_loss = meta.train_loss,
                config_name = meta.config_name, run_name = meta.run_name,
                split = String(split),
                metric = String(metric),
                n_samples = size(L, 2),
                n_nan_samples = n_nan_samples,                                # samples dropped because of NaN (0 almost always)
                n_repeats = size(L, 1),
                mean = mean(per_sample),
                std_over_repeats = size(L, 1) > 1 ? std(per_repeat) : NaN,   # how much the overall score moves between repeats
                median = med, q25 = q25, q75 = q75,
                mean_sample_std = mean(sample_std),                           # stochasticity of reconstructing one sample
            ))
            for c in sort(unique(labels_ok))
                idx = labels_ok .== c
                push!(class_rows, (
                    dataset=meta.dataset, model=meta.model, train_loss=meta.train_loss,
                    config_name=meta.config_name, run_name=meta.run_name,
                    split=String(split), metric=String(metric), class=c,
                    n_samples=count(idx), mean=mean(per_sample[idx]),
                ))
            end
        end
    end
    return rows, class_rows
end

# Minimal CSV writer (avoids adding CSV.jl/DataFrames as dependencies).
csvcell(x) = ismissing(x) ? "" : x isa AbstractString ? (occursin(r"[\",\n]", x) ? "\"" * replace(x, "\"" => "\"\"") * "\"" : x) : string(x)
function write_csv(path, rows)
    isempty(rows) && return
    cols = keys(first(rows))
    open(path, "w") do io
        println(io, join(string.(cols), ","))
        for r in rows
            println(io, join((csvcell(r[c]) for c in cols), ","))
        end
    end
end

# ─────────────────────────────── main ───────────────────────────────
function main()
    mkpath(OUT_DIR)
    runs, results, per_class = NamedTuple[], NamedTuple[], NamedTuple[]
    failed = Tuple{String,String}[]

    for dataset in DATASETS, model in MODELS
        dir = datadir("GenExperiments", dataset, model, "seed=$(SEED)")
        isdir(dir) || (@warn "missing directory, skipped" dir; continue)
        run_dirs = filter(isdir, readdir(dir, join=true))
        @info "collecting" dataset model n_runs=length(run_dirs)
        for run_dir in tqdm(run_dirs)
            try
                meta = run_row(dataset, model, run_dir)
                push!(runs, meta)
                meta.evaluated || continue
                ev = deserialize(joinpath(run_dir, "evaluation", "evaluation_reconstruction.jls"))
                r, c = result_rows(meta, ev)
                append!(results, r); append!(per_class, c)
            catch e
                push!(failed, (run_dir, sprint(showerror, e)))
            end
        end
    end

    # ── sanity report (printed, so you can check before using the tables) ──
    println("\n=== runs per dataset / model (evaluated / total) ===")
    for dataset in values(DATASET_DISPLAY), model in MODELS
        rs = filter(r -> r.dataset == dataset && r.model == model, runs)
        isempty(rs) && continue
        println(rpad("$(dataset)/$(model)", 32), count(r -> r.evaluated, rs), " / ", length(rs))
    end
    stale = filter(r -> r.dataset == "mnist" && r.evaluated && r.eval_mtime < MNIST_SPLIT_FIX_TIME, runs)
    if !isempty(stale)
        @warn "$(length(stale)) MNIST evaluations predate the split fix (seed 666 vs 1) -> their MNIST results are INVALID until re-evaluated"
    end
    not_eval = filter(r -> !r.evaluated, runs)
    isempty(not_eval) || println("\nnot evaluated: \n  ", join(getfield.(not_eval, :run_dir), "\n  "))
    isempty(failed) || println("\nFAILED to collect:\n  ", join(["$(p): $(m)" for (p, m) in failed], "\n  "))

    # ── save ──
    for (name, rows) in (("runs", runs), ("results", results), ("per_class", per_class))
        serialize(joinpath(OUT_DIR, "$(name).jls"), rows)
        write_csv(joinpath(OUT_DIR, "$(name).csv"), rows)
        println("saved $(name): $(length(rows)) rows")
    end
    return runs, results, per_class
end

main()
