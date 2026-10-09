# Build comparison tables from the collected reconstruction results.
#
# Input:  collected_results/{runs,results,per_class}.jls   (made by collect_reconstruction_results.jl)
# Output: tables/reconstruction_tables.md                   human-readable tables (one section per table)
#         tables/<table>.{jls,csv}                          the same tables in long format for plotting / LaTeX
#
# Principles:
#   - A run is identified by its ID (run_name); it is characterised by (train_loss, beta_schedule, beta_max).
#   - Every evaluated run is a candidate; nothing is excluded by hand.
#   - Selection uses ONLY the valid split; everything reported uses ONLY the test split.
#   - All metrics (cd, dcd, sh, mmd) are distances -> lower is better.
#   - NeuralStatistician runs are split into "corr." / "no corr." by how much information goes through the
#     per-point latents zᵢ (see NS_KLZ_MAX). Only "no corr." competes in the model comparison (T1, T4, T6).
#
# Usage:
#   julia --project experiments/EvaluationOfGenExperiments/make_reconstruction_tables.jl

using DrWatson
@quickactivate

using Serialization
using Statistics
using Printf

# ─────────────────────────────── settings ───────────────────────────────
const IN_DIR   = joinpath(@__DIR__, "collected_results")
const OUT_DIR  = joinpath(@__DIR__, "tables")
const DATASETS = ["mnist", "airplane", "core5"]
const MODELS   = ["setvae", "poolmodel", "neuralstatistician"]
const METRICS  = ["cd", "dcd", "sh", "mmd"]

# How the best run of a group is chosen on the VALID split:
#   "mean_rank" -> rank candidates on each metric separately, choose the lowest average rank
#                  (does not favour any single metric, e.g. does not favour CD-trained models)
#   "cd" / "dcd" / "sh" / "mmd" -> choose the lowest valid mean of that one metric
const SELECTION = "mean_rank"

# NeuralStatistician point-correspondence criterion.
# valid_kl_z = KL(q(zᵢ|xᵢ,c) || p(zᵢ|c)) in nats PER POINT (final validation value from trainlog.jsonl).
# On average it upper-bounds the information the encoder sends about each point through zᵢ, i.e. the
# "copy x̂ᵢ ≈ xᵢ" shortcut. A run with valid_kl_z ≥ NS_KLZ_MAX is labelled "corr." (uses correspondence).
# 0.1 nat ≈ 0.14 bit per point: far too little to tell where on the shape a point lies.
const NS_KLZ_MAX = 0.1
const NS_NOCORR = "neuralstatistician (no corr.)"
const NS_CORR   = "neuralstatistician (corr.)"
const VARIANTS  = ["setvae", "poolmodel", NS_NOCORR, NS_CORR]       # display order
const COMPARED  = ["setvae", "poolmodel", NS_NOCORR]                 # variants in the model comparison (T1, T4, T6)

# Same constant as in the collector: MNIST evaluations older than this used the leaked split.
const MNIST_SPLIT_FIX_TIME = 1791496800.0

# ─────────────────────────────── load ───────────────────────────────
const RUNS    = deserialize(joinpath(IN_DIR, "runs.jls"))
const RESULTS = deserialize(joinpath(IN_DIR, "results.jls"))
const CLASSES = deserialize(joinpath(IN_DIR, "per_class.jls"))

runkey(r) = (r.dataset, r.model, r.run_name)
const STAT = Dict((runkey(r)..., r.split, r.metric) => r for r in RESULTS)   # (dataset, model, run, split, metric) -> summary row
has_all_metrics(r) = all(haskey(STAT, (runkey(r)..., s, m)) for s in ("valid", "test"), m in METRICS)

valid_mean(r, m) = STAT[(runkey(r)..., "valid", m)].mean
test_stat(r, m)  = STAT[(runkey(r)..., "test", m)]
test_mean_or_missing(r, m) = haskey(STAT, (runkey(r)..., "test", m)) ? test_stat(r, m).mean : missing

"Model label used for grouping: NeuralStatistician is split by the correspondence criterion."
function variant(r)
    r.model != "neuralstatistician" && return r.model
    (ismissing(r.valid_kl_z) || r.valid_kl_z >= NS_KLZ_MAX) && return NS_CORR   # missing KL -> conservatively "corr."
    return NS_NOCORR
end
vorder(v) = something(findfirst(==(v), VARIANTS), length(VARIANTS) + 1)
dorder(d) = something(findfirst(==(d), DATASETS), length(DATASETS) + 1)

# ─────────────────────────────── helpers ───────────────────────────────

"Group a vector by a key function -> Dict(key => vector of members)."
function groupby(f, xs)
    d = Dict{Any, Vector{Any}}()
    for x in xs
        push!(get!(d, f(x), Any[]), x)
    end
    return d
end

"Rank of each value, 1 = smallest. NaN is ranked last. Ties get consecutive ranks (order of appearance)."
ranks(v) = invperm(sortperm(v, lt = (a, b) -> isnan(b) ? !isnan(a) : (!isnan(a) && a < b)))

"Choose the best candidate on the valid split according to SELECTION; ties broken by valid CD."
function select_best(cands)
    if SELECTION == "mean_rank"
        score = zeros(length(cands))
        for m in METRICS
            score .+= ranks([valid_mean(r, m) for r in cands])
        end
        score ./= length(METRICS)
    elseif SELECTION in METRICS
        score = [valid_mean(r, SELECTION) for r in cands]
    else
        error("Unknown SELECTION = $(SELECTION)")
    end
    order = sortperm(collect(zip(score, [valid_mean(r, "cd") for r in cands])))
    return cands[first(order)]
end

"Row describing the selected run of a group: group key + which run won + its TEST metrics."
function selected_row(groupkey::NamedTuple, cands)
    b = select_best(cands)
    row = (; groupkey..., n_candidates = length(cands), run_name = b.run_name, base_model = b.model,
           train_loss_sel = b.train_loss, beta_schedule = b.beta_schedule, beta_max = b.beta_max,
           epoch_reached = b.epoch_reached, kl_z_per_point = b.valid_kl_z)
    for m in METRICS
        t = test_stat(b, m)
        row = merge(row, NamedTuple{(Symbol("$(m)"), Symbol("$(m)_std"), Symbol("$(m)_sample_std"))}(
            (t.mean, t.std_over_repeats, t.mean_sample_std)))
    end
    return row
end

skey(x) = ismissing(x) ? "" : string(x)   # sortable key for possibly-missing values
medianskip(v) = (w = collect(skipmissing(v)); isempty(w) ? missing : median(w))
minskip(v)    = (w = collect(skipmissing(v)); isempty(w) ? missing : minimum(w))

# ── formatting ──
fmt(x) = (ismissing(x) || (x isa AbstractFloat && isnan(x))) ? "–" : x isa AbstractFloat ? @sprintf("%.4g", x) : string(x)
pm(m, s) = "$(fmt(m)) ± $(fmt(s))"

"Write a markdown table. `bold_min_cols` = column indices where the smallest numeric value (from `values`) is bolded."
function md_table(io, header, rows; bold_min_cols = Int[], values = nothing)
    println(io, "| ", join(header, " | "), " |")
    println(io, "|", join(fill("---", length(header)), "|"), "|")
    for (i, r) in enumerate(rows)
        cells = copy(r)
        for c in bold_min_cols
            col = [values[j][c] for j in eachindex(rows)]
            finite = filter(x -> !ismissing(x) && !isnan(x), col)
            !isempty(finite) && !ismissing(values[i][c]) && values[i][c] == minimum(finite) && (cells[c] = "**$(cells[c])**")
        end
        println(io, "| ", join(cells, " | "), " |")
    end
    println(io)
end

# Minimal CSV writer (same as in the collector).
csvcell(x) = ismissing(x) ? "" : x isa AbstractString ? (occursin(r"[\",\n]", x) ? "\"" * replace(x, "\"" => "\"\"") * "\"" : x) : string(x)
function write_csv(path, rows)
    isempty(rows) && return
    cols = keys(first(rows))
    open(path, "w") do io
        println(io, join(string.(cols), ","))
        foreach(r -> println(io, join((csvcell(r[c]) for c in cols), ",")), rows)
    end
end
save_table(name, rows) = (serialize(joinpath(OUT_DIR, "$(name).jls"), rows); write_csv(joinpath(OUT_DIR, "$(name).csv"), rows))

"Markdown block for a table of selected runs (T1 / T2), best value per metric column in bold."
function md_selected(io, rows; loss_col = :train_loss_sel)
    vals = [Any[r.model, getproperty(r, loss_col), r.beta_schedule, r.beta_max, r.n_candidates,
                (getproperty(r, Symbol(m)) for m in METRICS)...] for r in rows]
    md_table(io, ["model", "train loss", "β schedule", "β max", "#runs", METRICS...],
             [[r.model, getproperty(r, loss_col), skey(r.beta_schedule), fmt(r.beta_max), string(r.n_candidates),
               (pm(getproperty(r, Symbol(m)), getproperty(r, Symbol("$(m)_std"))) for m in METRICS)...] for r in rows];
             bold_min_cols = collect(6:5+length(METRICS)), values = vals)
end

# ─────────────────────────────── tables ───────────────────────────────
function main()
    mkpath(OUT_DIR)
    all_runs = filter(r -> r.dataset in DATASETS && r.model in MODELS, RUNS)
    evaluated = filter(r -> r.evaluated, all_runs)
    incomplete = filter(r -> !has_all_metrics(r), evaluated)
    cands = [merge(r, (model_v = variant(r),)) for r in evaluated if has_all_metrics(r)]
    compared = filter(r -> r.model_v in COMPARED, cands)

    # T1: best run per (dataset, model) over ALL train losses and β -> "which model reconstructs best"
    #     (only COMPARED variants: NeuralStatistician without point correspondence)
    t1 = [selected_row((dataset = k[1], model = k[2]), g) for (k, g) in groupby(r -> (r.dataset, r.model_v), compared)]
    sort!(t1, by = r -> (dorder(r.dataset), vorder(r.model)))

    # T2: best run per (dataset, model variant, train_loss) -> train loss × eval metric (both NS variants shown)
    t2 = [selected_row((dataset = k[1], model = k[2], train_loss = k[3]), g)
          for (k, g) in groupby(r -> (r.dataset, r.model_v, r.train_loss), cands)]
    sort!(t2, by = r -> (dorder(r.dataset), vorder(r.model), r.train_loss))

    # T3: best run per (dataset, model variant, train_loss, beta_schedule, beta_max) -> effect of β
    t3 = [selected_row((dataset = k[1], model = k[2], train_loss = k[3], beta_schedule = k[4], beta_max = k[5]), g)
          for (k, g) in groupby(r -> (r.dataset, r.model_v, r.train_loss, r.beta_schedule, r.beta_max), cands)]
    sort!(t3, by = r -> (dorder(r.dataset), vorder(r.model), r.train_loss, skey(r.beta_schedule), skey(r.beta_max)))

    # T4: ranks of the T1 models per dataset and metric on TEST
    t4 = NamedTuple[]
    for ds in DATASETS
        rows = filter(r -> r.dataset == ds, t1)
        isempty(rows) && continue
        for m in METRICS
            for (r, k) in zip(rows, ranks([getproperty(r, Symbol(m)) for r in rows]))
                push!(t4, (dataset = ds, model = r.model, metric = m, rank = k, n_models = length(rows)))
            end
        end
    end

    # T5: hyperparameter sensitivity -> distribution of TEST means over ALL runs of a (dataset, model variant, train_loss)
    t5 = NamedTuple[]
    for (k, g) in groupby(r -> (r.dataset, r.model_v, r.train_loss), cands), m in METRICS
        v = [test_stat(r, m).mean for r in g]
        q = quantile(v, [0.0, 0.25, 0.5, 0.75, 1.0])
        push!(t5, (dataset = k[1], model = k[2], train_loss = k[3], metric = m, n_runs = length(v),
                   best = q[1], q25 = q[2], median = q[3], q75 = q[4], worst = q[5]))
    end
    sort!(t5, by = r -> (dorder(r.dataset), vorder(r.model), r.train_loss, r.metric))

    # T6: per-class TEST means of the runs selected in T1
    sel = Dict((r.dataset, r.base_model, r.run_name) => r.model for r in t1)
    t6 = [merge(c, (model = sel[(c.dataset, c.model, c.run_name)],))
          for c in CLASSES if c.split == "test" && haskey(sel, (c.dataset, c.model, c.run_name))]
    sort!(t6, by = c -> (dorder(c.dataset), vorder(c.model), c.metric, c.class))

    # T7: NeuralStatistician – every run: latent sizes, information through zᵢ vs c, train loss, test metrics.
    #     Uses ALL NS runs (also not evaluated / missing a metric), so KL can be inspected everywhere.
    ns = filter(r -> r.model == "neuralstatistician", all_runs)
    t7 = [(; dataset = r.dataset, train_loss = r.train_loss, run_name = r.run_name,
           zdim = r.ns_zdim, cdim = r.ns_cdim, hdim = r.ns_hdim, n_points = r.n_points,
           kl_z_per_point = r.valid_kl_z, kl_z_per_set = r.valid_kl_z_set, kl_c_per_set = r.valid_kl_c,
           correspondence = variant(r) == NS_CORR,
           (Symbol(m) => test_mean_or_missing(r, m) for m in METRICS)...) for r in ns]
    sort!(t7, by = r -> (dorder(r.dataset), r.train_loss, skey(r.zdim), r.run_name))

    # T7 summary: per (dataset, train_loss, zdim) – medians over runs
    t7s = NamedTuple[]
    for (k, g) in groupby(r -> (r.dataset, r.train_loss, r.zdim), t7)
        push!(t7s, (; dataset = k[1], train_loss = k[2], zdim = k[3], n_runs = length(g),
                    n_no_corr = count(r -> !r.correspondence, g),
                    kl_z_per_point = medianskip(getfield.(g, :kl_z_per_point)),
                    kl_z_per_set = medianskip(getfield.(g, :kl_z_per_set)),
                    kl_c_per_set = medianskip(getfield.(g, :kl_c_per_set)),
                    (Symbol("$(m)_median") => medianskip([getproperty(r, Symbol(m)) for r in g]) for m in METRICS)...,
                    cd_best = minskip(getfield.(g, :cd))))
    end
    sort!(t7s, by = r -> (dorder(r.dataset), r.train_loss, skey(r.zdim)))

    for (name, rows) in (("t1_best_per_model", t1), ("t2_best_per_model_trainloss", t2), ("t3_best_per_beta", t3),
                         ("t4_model_ranks", t4), ("t5_sensitivity", t5), ("t6_per_class", t6),
                         ("t7_neuralstatistician_runs", t7), ("t7_neuralstatistician_summary", t7s))
        save_table(name, rows)
    end

    # ─────────────────────────────── markdown report ───────────────────────────────
    open(joinpath(OUT_DIR, "reconstruction_tables.md"), "w") do io
        println(io, "# Reconstruction results\n")
        println(io, "- Selection on **valid** by `$(SELECTION)`; all numbers are **test** (mean over samples ± std over $(test_stat(first(cands), "cd").n_repeats) evaluation repeats of the same trained model). Lower is better; best per column in bold.")
        println(io, "- Candidates: $(length(cands)) evaluated runs with all metrics.")
        println(io, "- NeuralStatistician split by KL of the per-point latents: `corr.` = KL_z ≥ $(NS_KLZ_MAX) nats/point (information flows point-to-point through zᵢ), `no corr.` = below. Only `no corr.` enters T1, T4, T6.")
        isempty(incomplete) || println(io, "- Skipped (missing some metric): ", join(getfield.(incomplete, :run_name), ", "))
        nanrows = filter(r -> get(r, :n_nan_samples, 0) > 0, RESULTS)
        isempty(nanrows) || println(io, "- Samples dropped because the metric returned NaN (Sinkhorn underflow): ",
            join(["$(r.run_name) $(r.split).$(r.metric) $(r.n_nan_samples)/$(r.n_samples + r.n_nan_samples)" for r in nanrows], "; "))
        stale = count(r -> r.dataset == "mnist" && r.eval_mtime < MNIST_SPLIT_FIX_TIME, cands)
        stale > 0 && println(io, "- ⚠️ **$(stale) MNIST evaluations predate the split fix -> MNIST numbers are INVALID (train/test leakage) until re-evaluated.**")
        println(io)

        println(io, "## T1 – Best run per model (any train loss, any β; NeuralStatistician only without correspondence)\n")
        for ds in DATASETS
            rows = filter(r -> r.dataset == ds, t1); isempty(rows) && continue
            println(io, "### $(ds)\n")
            any(r -> r.model == NS_NOCORR, rows) || println(io, "_No NeuralStatistician run without correspondence on this dataset._\n")
            md_selected(io, rows)
        end

        println(io, "## T2 – Best run per model × train loss (rows: train loss, columns: eval metric)\n")
        for ds in DATASETS
            rows = filter(r -> r.dataset == ds, t2); isempty(rows) && continue
            println(io, "### $(ds)\n")
            md_selected(io, rows; loss_col = :train_loss)
        end

        println(io, "## T3 – Effect of β (best run per model × train loss × β schedule × β max)\n")
        for ds in DATASETS
            rows = filter(r -> r.dataset == ds, t3); isempty(rows) && continue
            println(io, "### $(ds)\n")
            md_table(io, ["model", "train loss", "β schedule", "β max", "#runs", "run", METRICS...],
                     [[r.model, r.train_loss, skey(r.beta_schedule), fmt(r.beta_max), string(r.n_candidates), r.run_name,
                       (fmt(getproperty(r, Symbol(m))) for m in METRICS)...] for r in rows])
        end

        println(io, "## T4 – Model ranks on test (from T1; 1 = best)\n")
        dss = [ds for ds in DATASETS if any(x -> x.dataset == ds, t4)]
        rows = Vector{String}[]
        for mdl in COMPARED
            rk = filter(x -> x.model == mdl, t4); isempty(rk) && continue
            cells = [(k = filter(x -> x.dataset == ds && x.metric == m, rk); isempty(k) ? "–" : "$(only(k).rank)/$(only(k).n_models)") for ds in dss for m in METRICS]
            push!(rows, [mdl, cells..., fmt(mean(getfield.(rk, :rank)))])
        end
        md_table(io, ["model", ["$(ds) $(m)" for ds in dss for m in METRICS]..., "mean rank"], rows)
        println(io, "_Mean rank averages over the cells the model appears in; a model missing on a dataset (e.g. NeuralStatistician on core5) is ranked among fewer models there._\n")

        println(io, "## T5 – Sensitivity to hyperparameters (test CD over ALL runs; other metrics in t5_sensitivity.csv)\n")
        for ds in DATASETS
            rows = filter(r -> r.dataset == ds && r.metric == "cd", t5); isempty(rows) && continue
            println(io, "### $(ds)\n")
            md_table(io, ["model", "train loss", "#runs", "best", "q25", "median", "q75", "worst"],
                     [[r.model, r.train_loss, string(r.n_runs), fmt(r.best), fmt(r.q25), fmt(r.median), fmt(r.q75), fmt(r.worst)] for r in rows])
        end

        println(io, "## T6 – Per-class test CD of the T1 runs (other metrics in t6_per_class.csv)\n")
        for ds in DATASETS
            rows = filter(c -> c.dataset == ds && c.metric == "cd", t6); isempty(rows) && continue
            classes = sort(unique(getfield.(rows, :class)))
            length(classes) < 2 && continue   # airplane = single class
            println(io, "### $(ds)\n")
            tbl = [[mdl, (fmt(only(filter(c -> c.model == mdl && c.class == k, rows)).mean) for k in classes)...]
                   for mdl in COMPARED if any(c -> c.model == mdl, rows)]
            md_table(io, ["model", string.(classes)...], tbl)
        end

        println(io, "## T7 – NeuralStatistician: latent size, information flow and reconstruction\n")
        println(io, "KL_z is nats **per point**, KL_c nats **per set**; KL_z/set = KL_z × #points. Medians over runs with the same train loss and zdim (cdim/hdim vary inside a group). Every run in t7_neuralstatistician_runs.csv.\n")
        for ds in DATASETS
            rows = filter(r -> r.dataset == ds, t7s); isempty(rows) && continue
            println(io, "### $(ds)\n")
            md_table(io, ["train loss", "zdim", "#runs", "#no corr.", "KL_z / point", "KL_z / set", "KL_c / set",
                          ("$(m) (median)" for m in METRICS)..., "cd (best)"],
                     [[r.train_loss, fmt(r.zdim), string(r.n_runs), string(r.n_no_corr), fmt(r.kl_z_per_point), fmt(r.kl_z_per_set),
                       fmt(r.kl_c_per_set), (fmt(getproperty(r, Symbol("$(m)_median"))) for m in METRICS)..., fmt(r.cd_best)] for r in rows])
        end
    end

    println("tables written to $(OUT_DIR)")
    isempty(incomplete) || @warn "$(length(incomplete)) evaluated runs miss some metric and were skipped" runs = getfield.(incomplete, :run_name)
    return t1, t2, t3, t4, t5, t6, t7, t7s
end

main()
