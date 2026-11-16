# Config generator for HSTree generation experiments on mnist_clock / mnist_clock_unbalanced:
#   model smhsqvae = StructuredMnistHSQVAE (model_type structured_mnist_hsqvae)
#   model smbagvae = StructuredMnistBagVAE (model_type structured_mnist_bagvae)
# Self-contained (config_generator_mine.jl runs its old grids at top level when included).
#
# Run from the repo root:   julia --project scripts/config_generator_hstrees.jl
# (only the batch calls at the bottom of this file are executed; existing files are skipped unless overwrite=true)
#
# Config name  <agg><loss>_<model>_c<T><H><XY>,  e.g. mcd_smhsqvae_c1000
#   prefix  mcd = Chamfer with agg "mean" (per-digit mean over valid points), scd = agg "sum".
#           scd keeps the SAME raw α, β, λ as mcd (not multiplied by npoints -- scaling all of them would only rescale
#           the whole loss, which AdamW cancels), so in scd reconstruction weighs ~npoints times more.
#   T  thousands : 1 = npoints 256, batch 64, 1000 epochs | 2 = npoints 512, batch 16, 500 epochs
#   H  hundreds  : β  0 = 1e-5 | 1 = 1e-3 | 2 = sigmoid (max 1e-3, milestone 80 % of epochs, slope 15/epochs)
#   XY last two  : XY = 2k + d,  d = dropout (even = 0, odd = 0.1; e_dropout = dsq_dropout = cp_dropout)
#                  k = 2·(α,λ index) + (e_pma_mz - 1):
#                      k 0 = mz1 α1 λ0.01    1 = mz2 α1 λ0.01    2 = mz1 α1 λ0.1    3 = mz2 α1 λ0.1
#                      k 4 = mz1 α10 λ0.01   5 = mz2 α10 λ0.01   6 = mz1 α10 λ0.1   7 = mz2 α10 λ0.1
#                  The dropout-0.1 twin of a config is its number + 1.
#                  smbagvae has no λ and no α; the same code picks (α₂ point count, α₁ digit count) instead, with α₂ in
#                  the place of α (per-digit point count, as in smhsqvae) and α₁ in the place of λ (digit level):
#                      k 0 = mz1 α₂1 α₁1     1 = mz2 α₂1 α₁1     2 = mz1 α₂1 α₁10    3 = mz2 α₂1 α₁10
#                      k 4 = mz1 α₂10 α₁1    5 = mz2 α₂10 α₁1    6 = mz1 α₂10 α₁10   7 = mz2 α₂10 α₁10
#                  (tied α₁ = α₂: k 0,1 and 6,7 -- both terms are squared errors on a count fraction, same scale)
# The same name means the same setting on every dataset, except: on the balanced dataset (mnist_clock, every digit
# has all npoints points) α (smbagvae: α₂) is forced to 0 (no point-count head needed), so k = 4..7 are not generated there.

using YAML, OrderedCollections

const HST_CONFIG_DIR = joinpath(@__DIR__, "..", "experiments", "HSTreesGeneration", "configs")

# ---- the coding ---------------------------------------------------------------------------------------------------
const HST_NPOINTS = Dict(1 => (npoints=256, batch_size=64, epochs=1000), 2 => (npoints=512, batch_size=16, epochs=500))
const HST_BETA = Dict(0 => 1e-5, 1 => 1e-3, 2 => :sigmoid)
const HST_SIGMOID_MAX = 1e-3
const HST_ALPHA_LAMBDA = Dict(0 => (α=1.0, λ=0.01), 1 => (α=1.0, λ=0.1), 2 => (α=10.0, λ=0.01), 3 => (α=10.0, λ=0.1))
const HST_ALPHA_BAG = Dict(0 => (α₂=1.0, α₁=1.0), 1 => (α₂=1.0, α₁=10.0), 2 => (α₂=10.0, α₁=1.0), 3 => (α₂=10.0, α₁=10.0))
const HST_MODEL_TYPE = Dict("smhsqvae" => "structured_mnist_hsqvae", "smbagvae" => "structured_mnist_bagvae")
const HST_DROPOUT = Dict(0 => 0.0, 1 => 0.1)
const HST_AGG_PREFIX = Dict("mean" => "mcd", "sum" => "scd")

"""
    hst_config_id(; n, β, αλ, mz, d)

Config number from the digit codes: `n` ∈ 1:2 (npoints), `β` ∈ 0:2, `αλ` ∈ 0:3, `mz` ∈ 1:2 (e_pma_mz), `d` ∈ 0:1 (dropout).
"""
function hst_config_id(; n::Int, β::Int, αλ::Int, mz::Int, d::Int)
    (haskey(HST_NPOINTS, n) && haskey(HST_BETA, β) && haskey(HST_ALPHA_LAMBDA, αλ) && mz in 1:2 && haskey(HST_DROPOUT, d)) ||
        throw(ArgumentError("invalid codes n=$n β=$β αλ=$αλ mz=$mz d=$d"))
    k = 2αλ + (mz - 1)
    return 1000n + 100β + 2k + d
end

function hst_decode_id(id::Int)
    n, β, xy = id ÷ 1000, (id ÷ 100) % 10, id % 100
    k, d = xy ÷ 2, xy % 2
    αλ, mz = k ÷ 2, k % 2 + 1
    hst_config_id(; n=n, β=β, αλ=αλ, mz=mz, d=d) == id || throw(ArgumentError("invalid config number $id"))
    return (n=n, β=β, αλ=αλ, mz=mz, d=d)
end

hst_config_name(id::Int; agg::String="mean", model::String="smhsqvae") = "$(HST_AGG_PREFIX[agg])_$(model)_c$(lpad(id, 4, '0'))"

# ---- config pieces (shared settings = the tested cd_smhsqvae_mnist_clock_*_test.yml) ------------------------------
function hst_data_config(dataset::String, npoints::Int)
    d = OrderedDict{String,Any}("dataset" => dataset)
    if dataset == "mnist_clock_unbalanced"
        d["min_points"] = 20
    elseif dataset == "mnist_clock"
        d["point_mask"] = true            # StructuredMnistHSQVAE needs the (1, npoints, L) point mask
    else
        throw(ArgumentError("unsupported dataset $dataset"))
    end
    merge!(d, OrderedDict{String,Any}("npoints" => npoints, "n_samples" => 10000, "min_digits" => 2, "max_digits" => 12,
        "sample_on_fly" => true, "normalize" => "global", "ratio" => 0.2))
    return d
end

function hst_model_config(e_pma_mz::Int, dropout::Float64; model::String="smhsqvae")
    model == "smbagvae" && return hst_bag_model_config(e_pma_mz, dropout)
    model == "smhsqvae" || throw(ArgumentError("unknown model $model"))
    OrderedDict{String,Any}(
        "model_type" => "structured_mnist_hsqvae",
        "e_hdim" => 64, "e_pre_depth" => 3, "e_pma_heads" => 4, "e_pma_mz" => e_pma_mz, "e_post_depth" => 2,
        "e_pma_attention_fn" => "slot_attention",
        "dsq_emb_dim" => 64, "dsq_hdim" => 64, "dsq_pre_depth" => 2, "dsq_indices" => 4, "dsq_heads" => 4,
        "dsq_post_depth" => 2, "dsq_pma_attention_fn" => "slot_attention", "dsq_zdim" => 32, "dsq_n_slots" => 12,
        "dsq_n_attn_layers" => 2, "d_mha_layers" => 2, "d_heads" => 4, "o_hdim" => 64, "o_depth" => 2,
        "cp_pre_layers" => 2, "cp_layers" => 2,
        "e_pma_card_pooling" => "linear", "dsq_pma_card_pooling" => "linear",
        "e_dropout" => dropout, "dsq_dropout" => dropout, "cp_dropout" => dropout,
        "activation" => "gelu", "output_activation" => "identity",
    )
end

# StructuredMnistBagVAE: the smhsqvae settings with the deep-slot-query middle (dsq_*) replaced by the DiscreteBagVAE
# (dbv_*; dbv_depth = its MLP depth, dbv_indices = its m_z). No dsq_n_slots: the number of digit slots comes from the data.
function hst_bag_model_config(e_pma_mz::Int, dropout::Float64)
    OrderedDict{String,Any}(
        "model_type" => "structured_mnist_bagvae",
        "e_hdim" => 64, "e_pre_depth" => 3, "e_pma_heads" => 4, "e_pma_mz" => e_pma_mz, "e_post_depth" => 2,
        "e_pma_attention_fn" => "slot_attention",
        "dbv_emb_dim" => 64, "dbv_hdim" => 64, "dbv_depth" => 2, "dbv_indices" => 4, "dbv_heads" => 4,
        "dbv_pma_attention_fn" => "slot_attention", "dbv_zdim" => 32, "dbv_n_attn_layers" => 2,
        "d_mha_layers" => 2, "d_heads" => 4, "o_hdim" => 64, "o_depth" => 2,
        "cp_pre_layers" => 2, "cp_layers" => 2,
        "e_pma_card_pooling" => "linear", "dbv_pma_card_pooling" => "linear",
        "e_dropout" => dropout, "dbv_dropout" => dropout, "cp_dropout" => dropout,
        "activation" => "gelu", "output_activation" => "identity",
    )
end

function hst_train_config(; agg::String, loss_kwargs::AbstractDict, β, batch_size::Int, epochs::Int)
    t = OrderedDict{String,Any}(
        "loss_function" => OrderedDict{String,Any}("type" => "chamfer_pairwise_distance", "w1" => 1, "w2" => 1, "agg" => agg),
        "loss_kwargs" => OrderedDict{String,Any}(loss_kwargs),
        "lr" => 0.0003, "weight_decay" => 0.0001,
        "epochs" => epochs, "batch_size" => batch_size,
    )
    if β === :sigmoid
        t["beta_annealer"] = OrderedDict{String,Any}("type" => "sigmoidal", "max_value" => HST_SIGMOID_MAX,
            "milestone" => round(Int, 0.8 * epochs), "slope_factor" => 15 / epochs)   # 0.015 @ 1000, 0.03 @ 500 (same shape)
    else
        t["beta"] = β
    end
    merge!(t, OrderedDict{String,Any}(
        "use_gpu" => true, "valid_check_interval" => 1000, "validation_check_after_epoch" => true,
        "checkpoint_interval_epochs" => 10, "early_stopping" => true, "patience" => 100000, "grad_skip" => false,
        "save_val_predictions" => false, "val_prediction_count" => 16, "val_prediction_interval_epochs" => 10,
        "val_prediction_dirname" => "val_predictionsdata", "verbose" => true,
    ))
    return t
end

"""
    make_hst_config(id::Int; dataset="mnist_clock_unbalanced", agg="mean", model="smhsqvae") -> (name, cfg)

Builds the config with number `id` (coding at the top of the file). On the balanced dataset (`mnist_clock`) the
point-count weight (α, smbagvae: α₂) is 0 and only k = 0..3 are valid.
"""
function make_hst_config(id::Int; dataset::String="mnist_clock_unbalanced", agg::String="mean", model::String="smhsqvae")
    haskey(HST_AGG_PREFIX, agg) || throw(ArgumentError("agg must be \"mean\" or \"sum\", got $agg"))
    haskey(HST_MODEL_TYPE, model) || throw(ArgumentError("model must be one of $(keys(HST_MODEL_TYPE)), got $model"))
    c = hst_decode_id(id)
    np, β, dropout = HST_NPOINTS[c.n], HST_BETA[c.β], HST_DROPOUT[c.d]
    balanced = dataset == "mnist_clock"
    balanced && c.αλ >= 2 && throw(ArgumentError("config $id has a point-count weight of 10; on the balanced dataset it is 0, use k = 0..3"))
    loss_kwargs = if model == "smbagvae"
        w = HST_ALPHA_BAG[c.αλ]
        OrderedDict{String,Any}("α₁" => w.α₁, "α₂" => balanced ? 0.0 : w.α₂)
    else
        w = HST_ALPHA_LAMBDA[c.αλ]
        OrderedDict{String,Any}("λ" => w.λ, "α" => balanced ? 0.0 : w.α)
    end
    cfg = OrderedDict{String,Any}(
        "data" => hst_data_config(dataset, np.npoints),
        "model" => hst_model_config(c.mz, dropout; model=model),
        "train" => hst_train_config(; agg=agg, loss_kwargs=loss_kwargs, β=β, batch_size=np.batch_size, epochs=np.epochs),
    )
    return hst_config_name(id; agg=agg, model=model), cfg
end

function hst_header(id::Int, dataset::String, agg::String; model::String="smhsqvae")
    c = hst_decode_id(id); np = HST_NPOINTS[c.n]
    β = HST_BETA[c.β] === :sigmoid ? "sigmoid (max $(HST_SIGMOID_MAX))" : string(HST_BETA[c.β])
    bal = dataset == "mnist_clock"
    weights = if model == "smbagvae"
        w = HST_ALPHA_BAG[c.αλ]; "α₁=$(w.α₁) α₂=$(bal ? "0 (balanced)" : w.α₂)"
    else
        w = HST_ALPHA_LAMBDA[c.αλ]; "α=$(bal ? "0 (balanced)" : w.α) λ=$(w.λ)"
    end
    return """
    # $(hst_config_name(id; agg=agg, model=model)) on $dataset -- generated by scripts/config_generator_hstrees.jl (coding explained there)
    # agg=$agg | npoints=$(np.npoints) batch=$(np.batch_size) epochs=$(np.epochs) | β=$β | $weights | e_pma_mz=$(c.mz) | dropout=$(HST_DROPOUT[c.d])
    """
end

"""
    make_hst_configs(ids; datasets=("mnist_clock_unbalanced", "mnist_clock"), aggs=("mean",), model="smhsqvae", overwrite=false)

Writes every config number in `ids` for every dataset and aggregation to `configs/<dataset>_configs/<name>.yml`.
Numbers with a point-count weight of 10 (k = 4..7) are skipped on the balanced dataset. Existing files are skipped
unless `overwrite=true`.
"""
function make_hst_configs(ids; datasets=("mnist_clock_unbalanced", "mnist_clock"), aggs=("mean",), model::String="smhsqvae", overwrite::Bool=false)
    written = String[]
    for dataset in datasets, agg in aggs, id in ids
        dataset == "mnist_clock" && hst_decode_id(id).αλ >= 2 && continue
        name, cfg = make_hst_config(id; dataset=dataset, agg=agg, model=model)
        dir = joinpath(HST_CONFIG_DIR, "$(dataset)_configs"); mkpath(dir)
        file = joinpath(dir, "$name.yml")
        if isfile(file) && !overwrite
            println("exists, skipped: ", file); continue
        end
        open(file, "w") do io
            write(io, hst_header(id, dataset, agg; model=model))
            YAML.write(io, cfg)
        end
        push!(written, file); println("wrote: ", file)
    end
    return written
end

# all numbers of a sub-grid, e.g. hst_ids(n=1:2, β=0:1, αλ=0, mz=1:2, d=0)
hst_ids(; n=1:2, β=0:2, αλ=0:3, mz=1:2, d=0:1) =
    sort([hst_config_id(; n=a, β=b, αλ=c, mz=m, d=dd) for a in n for b in β for c in αλ for m in mz for dd in d])

# ---- batches ------------------------------------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    # batch 1 (2026-11-13): agg mean, dropout 0, α1 λ0.01, β ∈ {1e-5, 1e-3}, both npoints, both e_pma_mz
    #   -> mcd_smhsqvae_c1000, c1002, c1100, c1102, c2000, c2002, c2100, c2102 on both datasets
    make_hst_configs(hst_ids(n=1:2, β=0:1, αλ=0, mz=1:2, d=0))
    # StructuredMnistBagVAE (2026-11-16): tied weights α₁ = α₂ ∈ {1, 10} (codes 0 and 3 -> XY 00-03 and 12-15), all
    # npoints / β / e_pma_mz / dropout: 48 configs on mnist_clock_unbalanced; 24 on mnist_clock (α₂ = 0 there, so
    # code 3 is skipped and only α₁ = α₂ = 1 is written)
    make_hst_configs(hst_ids(αλ=[0, 3]); model="smbagvae")
end
