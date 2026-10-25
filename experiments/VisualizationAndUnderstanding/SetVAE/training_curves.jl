# Training curves of one SetVAE run: loss, reconstruction, KL (total and per latent layer), β and learning rate.
#
# Reads ONLY <run>/trainlog.jsonl (no model loading, no GPU). Two figure variants:
#   1. grid    – one panel per quantity, each on its own scale (losses and KL on log scale)
#   2. overlay – all curves in one panel, each min-max scaled to [0, 1]
#                (loss/KL curves are log10-transformed BEFORE min-max scaling, otherwise the huge values of
#                 the first epochs squash the rest of the curve into a flat line at 0)
#
# Train curves: per-iteration records averaged per epoch. Valid curves: one record per epoch.
# NOTE: the logged ℒₖₗ / ℒᵥₖₗ is β · KL (the weighted term in the loss), NOT the KL itself
#   (checked: last epoch of c401, ℒᵥₖₗ = 3.0257 = 0.1 × Σ ℒᵥₖₗₛ = 0.1 × 30.25). The KL plotted here is the
#   sum of the per-layer terms ℒₖₗₛ, i.e. loss = reconstruction + β · KL.
#
# Usage (from the repository root):
#   julia --project=experiments/VisualizationAndUnderstanding experiments/VisualizationAndUnderstanding/SetVAE/training_curves.jl [model] [variant]
#     ARGS[1] model:   a preset from RUN_PRESETS (best | cyclical | sigmoidal | cyclical_core5 | sigmoidal_core5; default best) or a path to a run folder
#     ARGS[2] variant: grid (default) | overlay | both
#   e.g. ... training_curves.jl cyclical
# Output: SetVAE/figures/training_curves_<run name>_<variant>.{pdf,png}

using JSON3
using Statistics
using CairoMakie

const REPO = normpath(joinpath(@__DIR__, "..", "..", ".."))
const RUNS_DIR = joinpath(REPO, "data", "GenExperiments")
# Presets (selection from tables/t3_best_per_beta.csv and tables/scheduler_pairs.csv, 2026-10-14):
const RUN_PRESETS = Dict(
    # best SetVAE on ShapeNet airplane in the main table (T1): step_linear, β_max = 0.1, CD, 4000 epochs
    "best"      => ("shapenet_class", "cd_setvae_c401_ID-982605", "airplane, step-linear β_max = 0.1"),
    # best SetVAE with a CYCLICAL schedule = best SetVAE on MNIST in T1: cyclical_sigmoidal, β_max = 1, CD
    "cyclical"  => ("mnist", "cd_setvae_e_c007_ID-654685", "MNIST, cyclical β_max = 1"),
    # its paired run: SAME architecture, sigmoidal schedule (for a side-by-side comparison)
    "sigmoidal" => ("mnist", "cd_setvae_e_c017_ID-292250", "MNIST, sigmoidal β_max = 1 (same architecture as \"cyclical\")"),
    # best SetVAE with a cyclical schedule on ShapeNet core5 (T3; CD, β_max = 1, trained the full 1150 epochs)
    "cyclical_core5"  => ("shapenet_multiple_classes", "cd_setvae_e_c606_ID-883914", "core5, cyclical β_max = 1"),
    # its paired run: same architecture, sigmoidal (1100 epochs; the same config was also trained as c510_ID-018048)
    "sigmoidal_core5" => ("shapenet_multiple_classes", "cd_setvae_e_c506_ID-880578", "core5, sigmoidal β_max = 1 (same architecture as \"cyclical_core5\")"),
)
function resolve_run(arg)
    haskey(RUN_PRESETS, arg) || return (arg, basename(arg), "")
    ds, run, desc = RUN_PRESETS[arg]
    return (joinpath(RUNS_DIR, ds, "setvae", "seed=1", run), run, desc)
end
const FIG_DIR = joinpath(@__DIR__, "figures")

# ─────────────────────────────── read log ───────────────────────────────
"""
Parse trainlog.jsonl into per-epoch curves.
Returns a NamedTuple of vectors indexed by epoch (train: mean over iterations of that epoch).
"""
function read_trainlog(path::String)
    tr = Dict{Int, Vector{Any}}()           # epoch => train records
    va = Dict{Int, Any}()                   # epoch => last valid record
    for line in eachline(path)
        isempty(strip(line)) && continue
        r = JSON3.read(line)
        e = Int(r[:epoch])
        if r[:mode] == "train"
            push!(get!(tr, e, Any[]), r)
        elseif r[:mode] == "valid"
            va[e] = r
        end
    end
    epochs = sort(collect(keys(tr)))
    m(e, k) = mean(Float64(x[k]) for x in tr[e])
    vepochs = sort(collect(keys(va)))
    nlayers = length(first(values(va))[:ℒᵥₖₗₛ])
    return (
        epoch     = epochs,
        loss      = [m(e, :ℒ) for e in epochs],
        rec       = [m(e, :ℒ_rec) for e in epochs],
        kl        = [mean(sum(Float64, x[:ℒₖₗₛ]) for x in tr[e]) for e in epochs],   # KL summed over layers
        bkl       = [m(e, :ℒₖₗ) for e in epochs],                                   # logged ℒₖₗ = β · KL
        beta      = [m(e, :β) for e in epochs],
        lr        = [m(e, :η) for e in epochs],
        vepoch    = vepochs,
        vloss     = [Float64(va[e][:ℒᵥ]) for e in vepochs],
        vrec      = [Float64(va[e][:ℒᵥ_rec]) for e in vepochs],
        vkl       = [sum(Float64, va[e][:ℒᵥₖₗₛ]) for e in vepochs],                 # KL summed over layers
        vbkl      = [Float64(va[e][:ℒᵥₖₗ]) for e in vepochs],                        # logged ℒᵥₖₗ = β · KL
        vkl_layer = [[Float64(va[e][:ℒᵥₖₗₛ][l]) for e in vepochs] for l in 1:nlayers],
    )
end

"True if a curve is constant up to floating-point noise."
is_constant(v) = (maximum(v) - minimum(v)) <= 1e-6 * maximum(abs.(v))

"Min-max scale to [0, 1]; constant curves become 0.5 so they stay visible."
function minmax(v)
    lo, hi = extrema(v)
    hi == lo && return fill(0.5, length(v))
    return (v .- lo) ./ (hi - lo)
end

# ─────────────────────────────── figures ───────────────────────────────
const C_TRAIN = Makie.wong_colors()[1]
const C_VALID = Makie.wong_colors()[2]

function plot_grid(c; title = "")
    fig = Figure(size = (1180, 900), fontsize = 14)
    kw = (xlabel = "epoch",)
    ax_loss = Axis(fig[1, 1]; title = "Loss (ELBO objective)", yscale = log10, kw...)
    lines!(ax_loss, c.epoch, c.loss, color = C_TRAIN, label = "train")
    lines!(ax_loss, c.vepoch, c.vloss, color = C_VALID, label = "valid")
    axislegend(ax_loss, position = :rt)

    ax_rec = Axis(fig[1, 2]; title = "Reconstruction loss", yscale = log10, kw...)
    lines!(ax_rec, c.epoch, c.rec, color = C_TRAIN, label = "train")
    lines!(ax_rec, c.vepoch, c.vrec, color = C_VALID, label = "valid")
    axislegend(ax_rec, position = :rt)

    ax_kl = Axis(fig[2, 1]; title = "KL divergence (sum over latent layers)", yscale = log10, kw...)
    lines!(ax_kl, c.epoch, c.kl, color = C_TRAIN, label = "train")
    lines!(ax_kl, c.vepoch, c.vkl, color = C_VALID, label = "valid")
    axislegend(ax_kl, position = :rt)

    ax_kll = Axis(fig[2, 2]; title = "KL per latent layer (valid)", yscale = log10, kw...)
    cols = Makie.wong_colors()
    for (l, v) in enumerate(c.vkl_layer)
        lines!(ax_kll, c.vepoch, v, color = cols[mod1(l, length(cols))], label = "$(l)")
    end
    Legend(fig[2, 3], ax_kll, "latent layer", labelsize = 11, framevisible = false)   # outside: curves fill the whole panel

    ax_beta = Axis(fig[3, 1]; title = "β (KL weight)", kw...)
    lines!(ax_beta, c.epoch, c.beta, color = :black)

    ax_lr = Axis(fig[3, 2]; title = "learning rate η", kw...)
    lines!(ax_lr, c.epoch, c.lr, color = :black)
    is_constant(c.lr) && ylims!(ax_lr, 0, 2 * maximum(c.lr))   # constant LR (up to float noise): flat line, not zoomed noise

    isempty(title) || Label(fig[0, 1:2], title, fontsize = 16, font = :bold)
    return fig
end

function plot_overlay(c; title = "")
    fig = Figure(size = (1000, 500), fontsize = 14)
    ax = Axis(fig[1, 1]; xlabel = "epoch", ylabel = "min-max scaled",
              title = isempty(title) ? "Training curves (losses/KL: log10, then min-max)" : "$(title) – losses/KL: log10, then min-max")
    cols = Makie.wong_colors()
    lines!(ax, c.vepoch, minmax(log10.(c.vloss)), color = cols[1], label = "valid loss (log)")
    lines!(ax, c.vepoch, minmax(log10.(c.vrec)),  color = cols[2], label = "valid reconstruction (log)")
    lines!(ax, c.vepoch, minmax(log10.(c.vkl)),   color = cols[3], label = "valid KL (log)")
    lines!(ax, c.epoch,  minmax(c.beta),          color = cols[4], label = "β", linewidth = 2.5)
    lines!(ax, c.epoch,  minmax(c.lr),            color = cols[5], label = "learning rate", linestyle = :dash)
    Legend(fig[1, 2], ax)
    return fig
end

# ─────────────────────────────── main ───────────────────────────────
function main(args)
    run_dir, name, desc = resolve_run(length(args) >= 1 ? args[1] : "best")
    variant = length(args) >= 2 ? args[2] : "grid"
    variant in ("grid", "overlay", "both") || error("ARGS[2] must be grid, overlay or both (got $(variant))")
    isdir(run_dir) || error("unknown model \"$(run_dir)\": use one of $(join(sort(collect(keys(RUN_PRESETS))), ", ")) or a run folder path")
    log     = joinpath(run_dir, "trainlog.jsonl")
    isfile(log) || error("trainlog not found: $(log)")

    c = read_trainlog(log)
    @info "read $(log)" epochs = length(c.epoch) valid_points = length(c.vepoch) kl_layers = length(c.vkl_layer)
    is_constant(c.lr) && @info "learning rate is constant ($(first(c.lr))) – no LR scheduler in this run"

    mkpath(FIG_DIR)
    title = isempty(desc) ? name : "$(name)  ($(desc))"
    figs = Pair{String, Any}[]
    variant in ("grid", "both")    && push!(figs, "grid" => plot_grid(c; title = title))
    variant in ("overlay", "both") && push!(figs, "overlay" => plot_overlay(c; title = title))
    for (v, fig) in figs
        for ext in ("pdf", "png")
            f = joinpath(FIG_DIR, "training_curves_$(name)_$(v).$(ext)")
            save(f, fig)
            println("saved ", f)
        end
    end
end

main(ARGS)
