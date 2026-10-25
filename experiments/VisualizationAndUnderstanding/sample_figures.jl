# Qualitative figure per dataset: ground truth × reconstruction × generated samples, one row per model.
#
#                 Reconstruction                      Generated samples
#   Test set      x₁  x₂  x₃  x₄  x₅
#   NS            x̂₁  x̂₂  x̂₃  x̂₄  x̂₅        g₁  g₂  g₃  g₄  g₅
#   PoolModel     ...                         ...
#   SetVAE        ...                         ...
#
# Shows in one place that a model can reconstruct well and still generate badly (e.g. Neural Statistician).
# Models: the runs of the compact generation table (best training loss per model and dataset) of a selection version,
# except where LOSS_OVERRIDES picks another training loss of the same table (e.g. MNIST NS: CD)
# (make_generation_comparison.jl -> EvaluationOfGenExperiments/tables/generation_comparison_selected.jls).
#
# Data, nothing is recomputed:
#   - ground truth + reconstructions: <run>/results/reconstructions_final.jls (whole TEST split, final model; saved at the
#     end of training). All models see the same test sets in the same order; on ShapeNet each run drew its own random
#     2048-point subset of every test shape, so the ground-truth row (taken from the first model) shows the same
#     shapes with a different point subset than the other runs saw.
#   - class labels of the test sets: <run>/evaluation/evaluation_reconstruction.jls (test.labels)
#   - generated sets: EvaluationOfGenExperiments/generation_smoke/<run>/generation_test_n<N>[_latent=..].jls (field
#     `samples` = the FIRST 16 generated sets in generation order, i.e. not selected by quality)
#
# Which sets are shown (edit GT_INDICES / GEN_INDICES below to pick others, e.g. after regenerating samples):
#   ground truth: by default the first test set of each class (MNIST / Core5), the first test sets (airplane)
#   generated:    by default the first `ngen` stored samples
#
# Usage (from the repository root):
#   julia --project=experiments/VisualizationAndUnderstanding experiments/VisualizationAndUnderstanding/sample_figures.jl [dataset] [selection] [nrec] [ngen] [layout] [colors]
#     dataset:   core5 (default) | mnist | airplane | all
#     selection: recb1 (default, recsel with SetVAE β_max = 1) | rec | gen
#     nrec/ngen: number of reconstruction / generated columns (default 5 / 5)
#     layout:    grid (whole figure with row labels) | rows (one figure per row, for LaTeX subfigures) | both (default)
#     colors:    model (default, MODEL_COLORS per row) | single (all rows dark grey; files get the suffix _mono); rows only
# Output: VisualizationAndUnderstanding/figures/samples_<dataset>_<selection>.{pdf,png}            (grid)
#         ..._row0_testset, ..._row1_neuralstatistician, ..._row2_poolmodel, ..._row3_setvae      (rows; same width,
#         same column positions, no row labels; row0 has the block titles and an empty generated block)
#         The runs and indices used are printed.

include(joinpath(@__DIR__, "thesis_style.jl"))   # Latin Modern, sizes in pt, MODEL_COLORS, save_figure
using GenerativeMIL   # needed to deserialize the result files
using Serialization
using Random
using Statistics

const REPO     = normpath(joinpath(@__DIR__, "..", ".."))
const SELECTED = deserialize(joinpath(REPO, "experiments", "EvaluationOfGenExperiments", "tables", "generation_comparison_selected.jls"))
const GEN_DIR  = joinpath(REPO, "experiments", "EvaluationOfGenExperiments", "generation_smoke")
const RUNS_DIR = joinpath(REPO, "data", "GenExperiments")
const FIG_DIR  = joinpath(@__DIR__, "figures")
const DATA_FOLDER = Dict("mnist" => "mnist", "airplane" => "shapenet_class", "core5" => "shapenet_multiple_classes")
const MODELS      = ["neuralstatistician", "poolmodel", "setvae"]
const MODEL_LABEL = Dict("neuralstatistician" => "NS", "poolmodel" => "PoolModel", "setvae" => "SetVAE")
const CLASS_NAMES = Dict("core5" => Dict(1 => "car", 2 => "chair", 3 => "airplane", 4 => "sofa", 5 => "table"))

# ── model overrides: (dataset, model) => training loss. Instead of the model's best row of the compact table, use
#    its row for this loss in the same selection (i.e. the run selected for that loss, e.g. by reconstruction CD).
const LOSS_OVERRIDES = Dict(("mnist", "neuralstatistician") => "cd")   # author: NS with the best CD reconstruction

# ── manual picks (indices into the TEST split / into the stored generated samples); `nothing` = default rule ──
const GT_INDICES  = Dict{String, Union{Nothing, Vector{Int}}}("mnist" => nothing, "airplane" => nothing, "core5" => nothing)
const GEN_INDICES = Dict{String, Union{Nothing, Vector{Int}}}("mnist" => nothing, "airplane" => nothing, "core5" => nothing)

# ── look ──
# 3D camera per ShapeNet dataset (y axis up); airplanes from higher up, otherwise the wings are seen edge-on
const VIEW = Dict("core5" => (azimuth = 1.15π, elevation = 0.12π), "airplane" => (azimuth = 1.2π, elevation = 0.3π))
const POINTSIZE_3D = 0.9                             # pt
const POINTSIZE_2D = (0.2, 2.2)                      # pt, MNIST: marker size scaled by the third coordinate
const C_GT = colorant"#333333"
const ROW_TITLES = ("Reconstruction", "Generation")  # block titles of the first row figure (row-by-row version)
const LABEL_W = 46                                   # pt, width of the row-label column
const ZOOM_3D = 0.72                                 # < 1 zooms in: the fixed camera fits the whole bounding cube

# ─────────────────────────────── data ───────────────────────────────
run_dir(ds, model, run) = joinpath(RUNS_DIR, DATA_FOLDER[ds], model, "seed=1", run)

"The selected run of every model on `ds` (compact table of selection `sel`), in MODELS order."
function selected_runs(ds, sel)
    out = []
    for m in MODELS
        i = get(SELECTED.best_row_per_model[sel], (ds, m), nothing)
        if haskey(LOSS_OVERRIDES, (ds, m))
            i = findfirst(s -> s.dataset == ds && s.model == m && s.loss == LOSS_OVERRIDES[(ds, m)], SELECTED.S)
            @info "override: $(m) on $(ds) uses its $(LOSS_OVERRIDES[(ds, m)]) row"
        end
        isnothing(i) && (@warn "no selected $(m) run on $(ds)"; continue)
        x = getproperty(SELECTED.S[i], sel)
        (ismissing(x) || isnothing(x)) && (@warn "no selected $(m) run on $(ds)"; continue)
        push!(out, (model = m, loss = SELECTED.S[i].loss, run = x.run_name, latent = x.latent, N = x.N))
    end
    return out
end

"Ground-truth test sets (default: first of each class, or the first ones), with their labels."
function gt_indices(ds, labels, nrec)
    !isnothing(GT_INDICES[ds]) && return GT_INDICES[ds][1:min(nrec, end)]
    classes = sort(unique(labels))
    length(classes) == 1 && return collect(1:nrec)
    picks = [findfirst(==(c), labels) for c in classes]
    return picks[1:min(nrec, end)]
end

function load_dataset_rows(ds, sel, nrec, ngen)
    runs = selected_runs(ds, sel)
    isempty(runs) && error("no selected runs for $(ds)")
    ev = deserialize(joinpath(run_dir(ds, runs[1].model, runs[1].run), "evaluation", "evaluation_reconstruction.jls"))
    labels = ev.test.labels
    rows = []
    gt, idx = nothing, Int[]
    for r in runs
        rec = deserialize(joinpath(run_dir(ds, r.model, r.run), "results", "reconstructions_final.jls"))
        @assert size(rec.x, 3) == length(labels) "test split size differs between the reconstructions and the evaluation labels"
        if isnothing(gt)
            idx = gt_indices(ds, labels, nrec)
            gt = rec.x[:, :, idx]
        end
        g = deserialize(joinpath(GEN_DIR, r.run, "generation_test_n$(r.N)$(isempty(r.latent) ? "" : "_latent=$(r.latent)").jls"))
        gidx = isnothing(GEN_INDICES[ds]) ? collect(1:min(ngen, size(g.samples, 3))) : GEN_INDICES[ds][1:min(ngen, end)]
        push!(rows, (r..., rec = rec.xhat[:, :, idx], gen = g.samples[:, :, gidx], gen_idx = gidx))
    end
    return (gt = gt, gt_idx = idx, gt_labels = labels[idx], rows = rows)
end

# ─────────────────────────────── drawing ───────────────────────────────
"MNIST: 2D scatter, y flipped (image orientation), marker size from the third coordinate (as in src/pc_functions.jl)."
function draw_2d!(cell, x, color, lims; view = nothing)
    ax = Axis(cell; aspect = DataAspect(), limits = lims)
    hidedecorations!(ax); hidespines!(ax)
    s = x[3, :]
    lo, hi = extrema(s)
    ms = hi > lo ? POINTSIZE_2D[1] .+ (s .- lo) ./ (hi - lo) .* (POINTSIZE_2D[2] - POINTSIZE_2D[1]) : fill(mean(POINTSIZE_2D), length(s))
    scatter!(ax, x[1, :], -x[2, :]; markersize = ms, color = color, strokewidth = 0)
    return ax
end

"ShapeNet: 3D scatter, y up (coordinates plotted as x, z, y), fixed camera and limits."
function draw_3d!(cell, x, color, lims; view)
    ax = Axis3(cell; aspect = :data, azimuth = view.azimuth, elevation = view.elevation, perspectiveness = 0,
               limits = lims, protrusions = 0, viewmode = :fit)
    hidedecorations!(ax); hidespines!(ax)
    scatter!(ax, x[1, :], x[3, :], x[2, :]; markersize = POINTSIZE_3D, color = color, strokewidth = 0)
    return ax
end

"Common limits for every cell of a dataset (from the ground truth, so all models are shown at the same scale)."
function cell_limits(ds, gt)
    q(v, p) = sort(vec(v))[clamp(round(Int, p * length(v)), 1, length(v))]
    pad(lo, hi) = (lo - 0.05(hi - lo), hi + 0.05(hi - lo))
    if ds == "mnist"
        return (pad(q(gt[1, :, :], 0.0), q(gt[1, :, :], 1.0)), pad(-q(gt[2, :, :], 1.0), -q(gt[2, :, :], 0.0)))
    end
    r = maximum(abs, [q(gt[k, :, :], p) for k in 1:3 for p in (0.005, 0.995)])   # symmetric cube, robust to outliers
    r *= ZOOM_3D
    return (-r, r, -r, r, -r, r)
end

"Cell drawing function for a dataset (2D for MNIST, 3D with the dataset's camera for ShapeNet)."
cell_drawer(ds) = (cell, x, color, lims) -> ds == "mnist" ? draw_2d!(cell, x, color, lims) : draw_3d!(cell, x, color, lims; view = VIEW[ds])

function sample_figure(ds, D)
    nrec, ngen = size(D.gt, 3), size(first(D.rows).gen, 3)
    lims = cell_limits(ds, D.gt)
    draw! = cell_drawer(ds)
    cellw = (TEXTWIDTH_PT - LABEL_W - 6) / (nrec + ngen + 0.5)      # 0.5 cell = gap between the two blocks
    fig = Figure(size = (TEXTWIDTH_PT, cellw * (length(D.rows) + 1) + 24))
    g = fig[1, 1] = GridLayout()
    Label(g[0, 1:nrec], "Reconstruction"; tellwidth = false)
    Label(g[0, nrec + 2:nrec + 1 + ngen], "Generated samples"; tellwidth = false)
    # row 1: ground truth (reconstruction block only)
    Label(g[1, 0], "Test set"; halign = :right, tellheight = false)
    for j in 1:nrec
        draw!(g[1, j], D.gt[:, :, j], C_GT, lims)
    end
    for (i, r) in enumerate(D.rows)
        Label(g[i + 1, 0], MODEL_LABEL[r.model]; halign = :right, tellheight = false)
        for j in 1:nrec
            draw!(g[i + 1, j], r.rec[:, :, j], MODEL_COLORS[r.model], lims)
        end
        for j in 1:ngen
            draw!(g[i + 1, nrec + 1 + j], r.gen[:, :, j], MODEL_COLORS[r.model], lims)
        end
    end
    colsize!(g, nrec + 1, Fixed(cellw / 2))                       # the gap column
    for j in vcat(1:nrec, nrec + 2:nrec + 1 + ngen)
        colsize!(g, j, Fixed(cellw))
    end
    for i in 1:length(D.rows) + 1
        rowsize!(g, i, Fixed(cellw))
    end
    colsize!(g, 0, Fixed(LABEL_W))
    colgap!(g, 0); colgap!(g, 1, 6); rowgap!(g, 0)
    resize_to_layout!(fig)
    return fig
end

"""
One ROW of the grid as its own figure, for LaTeX subfigures (rows named by subcaptions there). No row label.
`gen = nothing` leaves the generated block empty (the test-set row). `titles = true` adds the two block titles
(only the first row). All row figures of a dataset have the same width and column positions, so they stack exactly.
"""
function row_figure(ds, rec, gen, color, lims; titles = false)
    nrec, ngen = size(rec, 3), isnothing(gen) ? NGEN_FOR_LAYOUT[] : size(gen, 3)
    draw! = cell_drawer(ds)
    cellw = (TEXTWIDTH_PT - 6) / (nrec + ngen + 0.5)
    fig = Figure(size = (TEXTWIDTH_PT, cellw + (titles ? 3FONTSIZE : 0)))
    g = fig[1, 1] = GridLayout(1, nrec + 1 + ngen)
    if titles
        Label(g[0, 1:nrec], ROW_TITLES[1]; tellwidth = false)
        Label(g[0, nrec + 2:nrec + 1 + ngen], ROW_TITLES[2]; tellwidth = false)
    end
    r = titles ? 2 : 1                                             # Label at row 0 shifts the cell row
    for j in 1:nrec
        draw!(g[r, j], rec[:, :, j], color, lims)
    end
    isnothing(gen) || for j in 1:ngen
        draw!(g[r, nrec + 1 + j], gen[:, :, j], color, lims)
    end
    colsize!(g, nrec + 1, Fixed(cellw / 2))
    for j in vcat(1:nrec, nrec + 2:nrec + 1 + ngen)
        colsize!(g, j, Fixed(cellw))
    end
    rowsize!(g, r, Fixed(cellw))
    colgap!(g, 0); rowgap!(g, 0)
    resize_to_layout!(fig)
    return fig
end
const NGEN_FOR_LAYOUT = Ref(5)   # generated-block width of the test-set row (same as the model rows)

"Save the rows: <base>_row0_testset, <base>_row1_<model>, ... `colors = :model` (MODEL_COLORS) or `:single` (all C_GT)."
function save_rows(ds, D, base; colors = :model)
    lims = cell_limits(ds, D.gt)
    NGEN_FOR_LAYOUT[] = size(first(D.rows).gen, 3)
    suffix = colors == :single ? "_mono" : ""
    paths = [save_figure("$(base)_row0_testset$(suffix)", row_figure(ds, D.gt, nothing, C_GT, lims; titles = true))]
    for (i, r) in enumerate(D.rows)
        col = colors == :single ? C_GT : MODEL_COLORS[r.model]
        push!(paths, save_figure("$(base)_row$(i)_$(r.model)$(suffix)", row_figure(ds, r.rec, r.gen, col, lims)))
    end
    return paths
end

# ─────────────────────────────── main ───────────────────────────────
function main(args)
    dsarg = length(args) >= 1 ? args[1] : "core5"
    sel   = Symbol(length(args) >= 2 ? args[2] : "recb1")
    nrec  = length(args) >= 3 ? parse(Int, args[3]) : 5
    ngen  = length(args) >= 4 ? parse(Int, args[4]) : 5
    layout = length(args) >= 5 ? args[5] : "both"                  # grid | rows | both
    colors = Symbol(length(args) >= 6 ? args[6] : "model")         # model | single (row figures only)
    layout in ("grid", "rows", "both") || error("layout must be grid, rows or both")
    colors in (:model, :single) || error("colors must be model or single")
    sel in (:rec, :recb1, :gen) || error("selection must be rec, recb1 or gen")
    for ds in (dsarg == "all" ? ["mnist", "airplane", "core5"] : [dsarg])
        D = load_dataset_rows(ds, sel, nrec, ngen)
        names = haskey(CLASS_NAMES, ds) ? [CLASS_NAMES[ds][l] for l in D.gt_labels] : string.(D.gt_labels)
        println("\n$(ds): test sets $(D.gt_idx) (classes $(join(names, ", ")))")
        for r in D.rows
            println("  $(rpad(MODEL_LABEL[r.model], 10)) $(r.run)$(isempty(r.latent) ? "" : " ($(r.latent))")  loss $(r.loss)  generated samples $(r.gen_idx)")
        end
        path = joinpath(FIG_DIR, "samples_$(ds)_$(sel)")
        if layout in ("grid", "both")
            save_figure(path, sample_figure(ds, D))
            println("saved $(path).{pdf,png}")
        end
        if layout in ("rows", "both")
            for p in save_rows(ds, D, path; colors)
                println("saved $(p).{pdf,png}")
            end
        end
    end
end

main(ARGS)
