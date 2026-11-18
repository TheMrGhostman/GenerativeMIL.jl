# MNIST point clouds (single digits): load_mnist
# (included from DataProcessing.jl; dispatcher and create_dataloaders are in load_datasets.jl)

"""
`_instance_split(n::Int, seed::Int; validation::Bool, ratio::Real)`

Deterministic train/validation/test split of `n` instances, shared by `load_mnist`, `load_mnist_clock` and
`load_mnist_clock_unbalanced` (so the three always use the same split for the same `seed`):
`randperm(MersenneTwister(seed), n)`, first 80 % train+validation, last 20 % test; with `validation=true` the
train+validation part is split again with `ratio` going to validation. Returns `(train_idx, val_idx, test_idx)`
(`val_idx` is empty when `validation=false`).
"""
function _instance_split(n::Int, seed::Int; validation::Bool, ratio::Real)
    perm = randperm(MersenneTwister(seed), n)
    n_train_test = round(Int, 0.8 * n)
    train_val_idx = perm[1:n_train_test]
    test_idx = perm[n_train_test+1:end]
    validation || return train_val_idx, Int[], test_idx
    n_train = round(Int, (1 - ratio) * length(train_val_idx))
    return train_val_idx[1:n_train], train_val_idx[n_train+1:end], test_idx
end

# ---- normalisation of MNIST point clouds (x, y, intensity) -------------------------------------------------------------

# Global statistics for `normalize = "global"`: computed once in Float64 over all points of the TRAINING digits of the
# natural MNIST point clouds (`_instance_split(60000, 666; validation=true, ratio=0.2)`), with one pooled sigma shared
# by x and y (isotropic: digits are zoomed, not stretched) and a separate sigma for intensity. Training splits of other
# seeds give the same values to < 0.05 %, the balanced (900-point) clouds to ~1.4 % -- so one set is used for every seed
# and for mnist, mnist_clock and mnist_clock_unbalanced alike (a digit gets the same coordinates in all of them).
const MNIST_GLOBAL_MU    = Float32[58.502, 58.3337, 0.5104]
const MNIST_GLOBAL_SIGMA = Float32[20.6597, 20.6597, 0.3624]

"""
    mnist_global_normalize(x)
    mnist_global_denormalize(z)

`normalize = "global"` for MNIST point clouds: `z = (x .- MNIST_GLOBAL_MU) ./ MNIST_GLOBAL_SIGMA` per dimension
(x, y share one sigma). Works on any array with the 3 point dimensions first, e.g. `(3, N, BS)` or
`(3, N, L, BS)`, or a vector of `(3, N_i)` clouds. `mnist_global_denormalize` is the exact inverse (for plots in pixel
coordinates).
"""
mnist_global_normalize(x::AbstractArray{T}) where T<:AbstractFloat =
    (x .- T.(MNIST_GLOBAL_MU)) ./ T.(MNIST_GLOBAL_SIGMA)
mnist_global_normalize(xs::AbstractVector{<:AbstractMatrix}) = map(mnist_global_normalize, xs)

mnist_global_denormalize(z::AbstractArray{T}) where T<:AbstractFloat =
    z .* T.(MNIST_GLOBAL_SIGMA) .+ T.(MNIST_GLOBAL_MU)
mnist_global_denormalize(zs::AbstractVector{<:AbstractMatrix}) = map(mnist_global_denormalize, zs)

"""
`_normalize_mnist(xs, normalize)`

Shared `normalize` option of the MNIST loaders:
- `false`: no normalisation;
- `true`: the legacy `normalize_point_cloud` (kept unchanged so old runs load the data they were trained on; note
  that its balanced/3-D method sums in Float32 and gets mu and sigma noticeably wrong for the full dataset);
- `"global"` / `:global`: [`mnist_global_normalize`](@ref) (recommended for new runs).
"""
function _normalize_mnist(xs, normalize::Union{Bool,AbstractString,Symbol})
    normalize === false && return xs
    normalize === true && return normalize_point_cloud(xs)
    Symbol(normalize) === :global && return mnist_global_normalize(xs)
    throw(ArgumentError("Unknown normalize option: $(repr(normalize)). Expected false, true or \"global\"."))
end

"""
    load_mnist(npoints=512; validation=true, cardinality_count=:balanced,
               sample_on_fly=false, normalize=false, ratio=0.2, seed=666, kwargs...)

Load MNIST point-cloud data and create train/validation/test splits.

# Arguments
- `npoints`: number of points requested after sampling.
- `validation`: if `true`, returns train/val/test; otherwise train/test.
- `cardinality_count`: `:balanced` or `:natural` source format, or `:unbalanced`: single digits exactly as in
  `mnist_clock_unbalanced` (per-digit point count + point mask, see [`_load_mnist_unbalanced`](@ref)); takes the
  extra kwargs `min_points` and `augment_points`.
- `sample_on_fly`: enable lazy sampling on the training split.
- `normalize`: `false`, `true` (legacy `normalize_point_cloud`) or `"global"` (fixed global MNIST statistics,
  recommended for new runs), see `_normalize_mnist`.
- `ratio`: validation ratio from the train/validation pool.
- `seed`: random seed for deterministic split/sampling behavior.
- `kwargs...`: reserved for API compatibility.

# Returns
- If `validation=true`:
    - `((x_train, y_train), (x_val, y_val), (x_test, y_test))`
- If `validation=false`:
    - `((x_train, y_train), (x_test, y_test))`

# Notes
- Validation and test splits are always pre-sampled.
- On-the-fly sampling is applied only to training data.
"""
function load_mnist(npoints=512; validation::Bool=true, cardinality_count::Symbol=:balanced, sample_on_fly::Bool=false, normalize::Union{Bool,AbstractString,Symbol}=false, ratio::AbstractFloat=0.2, seed::Int=666, kwargs...)
    cardinality_count == :unbalanced && return _load_mnist_unbalanced(npoints; validation=validation, sample_on_fly=sample_on_fly, normalize=normalize, ratio=ratio, seed=seed, kwargs...)
    if cardinality_count == :balanced
        dict_loaded = Serialization.deserialize(_mnist_balanced_path())
        xs = dict_loaded["features"]
        y = dict_loaded["targets"]
        @assert npoints <= size(xs, 2) "Number of requested points ($npoints) is greater than dataset cardinality ($(size(xs, 2)))."
    elseif cardinality_count == :natural
        dict_loaded = Serialization.deserialize(_mnist_natural_path())
        xs = dict_loaded["features"]
        y = dict_loaded["targets"]
        min_points = minimum(size.(xs, 2))
        @assert npoints <= min_points "Number of requested points ($npoints) is greater than minimum available cardinality ($min_points)."
    else
        error("Unknown cardinality_count: $cardinality_count. Expected :balanced or :natural")
    end

    # Optional preprocessing hook.
    xs = _normalize_mnist(xs, normalize)   # false | true (legacy) | "global"

    # Train/valid/test split from deterministic shuffled indices.
    train_idx, val_idx, test_idx = _instance_split(length(y), seed; validation=validation, ratio=ratio)

    y_train = y[train_idx]
    y_test = y[test_idx]

    Random.seed!(seed) # Ensure reproducibility for any sampling in the dataset loading process.
    if cardinality_count == :balanced
        x_train_full = xs[:, :, train_idx]
        x_val_full = validation ? xs[:, :, val_idx] : nothing
        x_test_full = xs[:, :, test_idx]

        # Training branch can be static or sampled on every observation access.
        x_train = sample_on_fly ? mapobs(pc -> sample_fixed_n_from_matrix(pc, npoints), x_train_full) : sample_fixed_n_from_matrix(x_train_full, npoints)

        # Validation and test are always fixed/pre-sampled.
        x_val = validation ? sample_fixed_n_from_matrix(x_val_full, npoints) : nothing
        x_test = sample_fixed_n_from_matrix(x_test_full, npoints)
    else
        x_train_full = xs[train_idx]
        x_val_full = validation ? xs[val_idx] : nothing
        x_test_full = xs[test_idx]

        # Natural cardinality: on-the-fly mapobs or static pre-sampling.
        x_train = sample_on_fly ? mapobs(pc -> sample_fixed_n_unsqueeze(pc, npoints), x_train_full) : _stack_point_clouds(sample_fixed_n.(x_train_full, npoints))

        # Validation and test are always fixed/pre-sampled.
        x_val = validation ? _stack_point_clouds(sample_fixed_n.(x_val_full, npoints)) : nothing
        x_test = _stack_point_clouds(sample_fixed_n.(x_test_full, npoints))
    end

    if validation
        y_val = y[val_idx]
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    end

    return (x_train, y_train), (x_test, y_test)

end

# ---- single digits as in mnist_clock_unbalanced (digit classifier for the HSTree evaluation) ---------------------------

"""
`_masked_digit(rng, pc::AbstractMatrix, k::Int, npoints::Int)`

`k` points of the point cloud `pc` (without replacement) in positions `1:k` of a zero-padded `(D, npoints, 1)` array,
plus the point mask `(1, npoints, 1)`. Same layout as one digit slot of `sample_one_unbalanced_bag`.
"""
function _masked_digit(rng, pc::AbstractMatrix, k::Int, npoints::Int)
    x = zeros(Float32, size(pc, 1), npoints, 1)
    m = falses(1, npoints, 1)
    idx = sample(rng, axes(pc, 2), k, replace=false)
    x[:, 1:k, 1] .= pc[:, idx]
    m[1, 1:k, 1] .= true
    return x, m
end

# eager version for validation / test: (D, npoints, N), BitArray (1, npoints, N)
function _masked_digits(rng, pcs::AbstractVector{<:AbstractMatrix}, ks::AbstractVector{Int}, npoints::Int)
    xm = [_masked_digit(rng, pc, k, npoints) for (pc, k) in zip(pcs, ks)]
    return cat(first.(xm)...; dims=3), cat(last.(xm)...; dims=3)
end

"""
`_load_mnist_unbalanced(npoints; min_points=1, augment_points=false, validation, sample_on_fly, normalize, ratio, seed)`

Single MNIST digits exactly as they appear in `mnist_clock_unbalanced` for the same `npoints`, `min_points` and
`seed`: same natural point clouds, same instance split (`_instance_split`), same per-instance point count
`_scaled_point_counts` (`k = clamp(round(n_natural · npoints / 6320), min_points, npoints)`), same normalization.
Used to train the digit classifier of the HSTree evaluation.

`augment_points=true` (training split only) draws the point count uniformly from `k:npoints` instead of `k`, so the
classifier also sees dense versions of every digit (generated digits always have all `npoints` points).

Returns `(x, mask, y)` triples: `x (3, npoints, N)`, `mask::BitArray (1, npoints, N)`, `y::Vector{Int}`. With
`sample_on_fly` the training split is a lazy `mapobs` of per-digit `(x (3, npoints, 1), mask (1, npoints, 1), y)`
triples (collated by `masked_digit_collate_fn`). Validation and test are always fixed (`MersenneTwister(seed + 1/2)`).
"""
function _load_mnist_unbalanced(npoints=512; min_points::Int=1, augment_points::Bool=false, validation::Bool=true, sample_on_fly::Bool=false, normalize::Union{Bool,AbstractString,Symbol}=false, ratio::AbstractFloat=0.2, seed::Int=666, kwargs...)
    dict_loaded = Serialization.deserialize(_mnist_natural_path())
    xs = dict_loaded["features"]
    y = dict_loaded["targets"]
    n_natural = size.(xs, 2)
    ref_points = maximum(n_natural)
    @assert npoints <= minimum(n_natural) "npoints ($npoints) must not exceed the smallest natural digit ($(minimum(n_natural))) -- augment_points samples up to npoints points without replacement."
    @assert 1 <= min_points <= npoints "min_points ($min_points) must be in 1:npoints ($npoints)."
    ns = _scaled_point_counts(n_natural, npoints, ref_points, min_points)
    xs = _normalize_mnist(xs, normalize)

    train_idx, val_idx, test_idx = _instance_split(length(y), seed; validation=validation, ratio=ratio)

    xs_tr, ns_tr, y_tr = xs[train_idx], ns[train_idx], y[train_idx]
    train_k(rng, i) = augment_points ? rand(rng, ns_tr[i]:npoints) : ns_tr[i]
    train = if sample_on_fly
        mapobs(i -> (_masked_digit(Random.default_rng(), xs_tr[i], train_k(Random.default_rng(), i), npoints)..., y_tr[i]), 1:length(train_idx))
    else
        rng = MersenneTwister(seed)
        (_masked_digits(rng, xs_tr, [train_k(rng, i) for i in eachindex(xs_tr)], npoints)..., y_tr)
    end
    test = (_masked_digits(MersenneTwister(seed + 2), xs[test_idx], ns[test_idx], npoints)..., y[test_idx])
    validation || return train, test
    valid = (_masked_digits(MersenneTwister(seed + 1), xs[val_idx], ns[val_idx], npoints)..., y[val_idx])
    return train, valid, test
end
