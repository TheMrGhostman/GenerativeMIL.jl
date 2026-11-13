# Bags of MNIST digits: mnist_clock (fixed points per digit) and mnist_clock_unbalanced
# (ink-proportional points per digit, 4-D point mask)
# (included from DataProcessing.jl; dispatcher and create_dataloaders are in load_datasets.jl)

"""
`sample_one_bag(rng, class_pool::Dict, classes::AbstractVector, xs::AbstractArray{T,3}, npoints::Int, min_digits::Int, max_digits::Int) where T`

Draw a single random "bag": a variable-cardinality set of digit point clouds packed into a
fixed-size `(D, npoints, max_digits)` slot tensor. Cardinality and class composition (including
duplicates -- classes are drawn independently per slot, with replacement) come fresh from `rng`;
points for each drawn digit instance are freshly subsampled from `xs`'s native resolution, one
instance at a time. Padding slots (beyond the drawn cardinality) are left as zero, marked invalid
in `mask`, and get the sentinel label `-1`.

Arguments (positional):
- `rng`: random number generator.
- `class_pool`: `Dict` mapping class label -> instance indices into `xs`.
- `classes`: available class labels, e.g. `collect(keys(class_pool))`.
- `xs`: full-resolution point-cloud pool `(D, N_full, n_instances)` -- NOT pre-subsampled, so
  every draw gets its own fresh `npoints`-point subsample.
- `npoints`: number of points to keep per digit after subsampling.
- `min_digits`, `max_digits`: inclusive range for the bag cardinality.

Returns:
- `data::Array{Float32,3}`: `(D, npoints, max_digits)`, zero in padding slots.
- `mask::BitArray{3}`: `(1, 1, max_digits)`, `true` in valid slots.
- `labels::Vector{Int}`: `(max_digits,)`, class label in valid slots, `-1` in padding slots.
"""
function sample_one_bag(rng, class_pool::Dict, classes::AbstractVector, xs::AbstractArray{T,3}, npoints::Int, min_digits::Int, max_digits::Int) where T
    D = size(xs, 1)
    data = zeros(Float32, D, npoints, max_digits)
    mask = falses(1, 1, max_digits)
    labels = fill(-1, max_digits)

    n = rand(rng, min_digits:max_digits)
    for j in 1:n
        c = rand(rng, classes)
        inst = rand(rng, class_pool[c])
        pt_idx = sample(rng, axes(xs, 2), npoints, replace=false)
        data[:, :, j] .= xs[:, pt_idx, inst]
        mask[1, 1, j] = true
        labels[j] = c
    end
    return data, mask, labels
end

"""
`_build_bags(rng, class_pool::Dict, classes::AbstractVector, xs::AbstractArray{T,3}, npoints::Int, min_digits::Int, max_digits::Int, n_samples::Int) where T`

Eagerly build `n_samples` bags via repeated calls to [`sample_one_bag`](@ref), stacked into batch
tensors. Used for the fixed/pre-sampled (non-`sample_on_fly`) path, and always for
validation/test regardless of `sample_on_fly`.

Returns:
- `data::Array{Float32,4}`: `(D, npoints, max_digits, n_samples)`.
- `mask::BitArray{4}`: `(1, 1, max_digits, n_samples)`.
- `labels::Array{Int,2}`: `(max_digits, n_samples)`.
"""
function _build_bags(rng, class_pool::Dict, classes::AbstractVector, xs::AbstractArray{T,3}, npoints::Int, min_digits::Int, max_digits::Int, n_samples::Int) where T
    D = size(xs, 1)
    data = zeros(Float32, D, npoints, max_digits, n_samples)
    mask = falses(1, 1, max_digits, n_samples)
    labels = fill(-1, max_digits, n_samples)
    for i in 1:n_samples
        d, m, l = sample_one_bag(rng, class_pool, classes, xs, npoints, min_digits, max_digits)
        data[:, :, :, i] .= d
        mask[:, :, :, i] .= m
        labels[:, i] .= l
    end
    return data, mask, labels
end

"""
`load_mnist_clock(npoints=512; n_samples=60000, min_digits=2, max_digits=12, validation=true, normalize=false, sample_on_fly=false, ratio=0.2, seed=666, point_mask=false, kwargs...)`

Build a "bag of MNIST digits" dataset: each sample is a variable-cardinality outer set of
`min_digits`..`max_digits` individual MNIST point-cloud digits (with class duplicates occurring
naturally, each a distinct instance), packed into a fixed `(D, npoints, max_digits)` slot tensor
with a validity mask -- directly compatible with `chamfer_pairwise_distance`/`hungarian_match`/
`hungarian_matching_loss`'s `(D,N,L,BS)`/`(1,1,L,BS)` convention. Train/validation/test are split
by underlying digit *instance* before any bag is built (mirrors [`load_mnist`](@ref)), so no
individual digit image can appear in more than one split.

Arguments (positional):
- `npoints`: number of points requested per digit after sampling.

Arguments (keyword):
- `n_samples`: number of bags for the training split; also used for validation/test.
- `min_digits`, `max_digits`: inclusive range for bag cardinality (`max_digits` is the fixed slot
  count `L`).
- `validation`: if `true`, returns train/val/test; otherwise train/test.
- `normalize`: `false`, `true` (legacy `normalize_point_cloud`) or `"global"` (fixed global MNIST statistics,
  recommended for new runs), see `_normalize_mnist`.
- `sample_on_fly`: if `true`, training bags are drawn fresh (via `MLUtils.mapobs`) every time
  they're accessed instead of being fixed at load time -- both bag composition and per-digit point
  subsampling vary from access to access. Validation/test are always fixed/pre-sampled.
- `ratio`: validation ratio from the train/validation instance pool.
- `seed`: random seed for the instance split and the fixed (non-on-the-fly) sampling paths.
- `point_mask`: if `true`, return the point mask `(1, npoints, max_digits)` (slot mask repeated over the points)
  instead of the slot mask `(1, 1, max_digits)`; needed by `StructuredMnistHSQVAE` (same mask layout as
  `mnist_clock_unbalanced`). Default `false` keeps the old output.
- `kwargs...`: reserved for API compatibility.

Returns:
- If `validation=true`: `(train_data, valid_data, test_data)`; if `validation=false`:
  `(train_data, test_data)`.
- `valid_data`/`test_data` (and `train_data` when `sample_on_fly=false`) are
  `(data, mask, labels)` triples as documented in [`_build_bags`](@ref).
- `train_data` when `sample_on_fly=true` is instead a `MLUtils.mapobs`-wrapped lazy collection of
  `n_samples` observations, each a fresh `(data, mask, labels)` triple as documented in
  [`sample_one_bag`](@ref) -- since composition varies per access, `n_samples` here means "steps
  per epoch", not "size of a fixed corpus".

Notes:
- `labels` are not needed by the reconstruction/matching loss -- they're carried through purely so
  bags can be colored/labeled for plots and diagnostics later.
"""
function load_mnist_clock(npoints=512; n_samples::Int=60000, min_digits::Int=2, max_digits::Int=12, validation::Bool=true, normalize::Union{Bool,AbstractString,Symbol}=false, sample_on_fly::Bool=false, ratio::AbstractFloat=0.2, seed::Int=666, point_mask::Bool=false, kwargs...)

    dict_loaded = Serialization.deserialize(_mnist_balanced_path())
    xs = dict_loaded["features"]
    ys = dict_loaded["targets"]
    @assert npoints <= size(xs, 2) "Number of requested points ($npoints) is greater than dataset cardinality ($(size(xs, 2)))."

    # Optional preprocessing hook.
    xs = _normalize_mnist(xs, normalize)   # false | true (legacy) | "global"

    # Train/valid/test split from deterministic shuffled indices, BEFORE any bag is built --
    # so no individual digit instance can appear in more than one split (mirrors load_mnist).
    train_idx, val_idx, test_idx = _instance_split(length(ys), seed; validation=validation, ratio=ratio)

    xs_train, ys_train = xs[:, :, train_idx], ys[train_idx]
    class_pool_train = Dict(c => findall(==(c), ys_train) for c in unique(ys_train))
    classes_train = collect(keys(class_pool_train))

    # Training branch can be static/reproducible or sampled fresh on every access.
    train_data = if sample_on_fly
        mapobs(_ -> sample_one_bag(Random.default_rng(), class_pool_train, classes_train, xs_train, npoints, min_digits, max_digits), 1:n_samples)
    else
        _build_bags(MersenneTwister(seed), class_pool_train, classes_train, xs_train, npoints, min_digits, max_digits, n_samples)
    end

    # Validation and test are always fixed/pre-sampled, for reproducible evaluation.
    xs_test, ys_test = xs[:, :, test_idx], ys[test_idx]
    class_pool_test = Dict(c => findall(==(c), ys_test) for c in unique(ys_test))
    classes_test = collect(keys(class_pool_test))
    test_data = _build_bags(MersenneTwister(seed + 2), class_pool_test, classes_test, xs_test, npoints, min_digits, max_digits, n_samples)

    if validation
        xs_val, ys_val = xs[:, :, val_idx], ys[val_idx]
        class_pool_val = Dict(c => findall(==(c), ys_val) for c in unique(ys_val))
        classes_val = collect(keys(class_pool_val))
        val_data = _build_bags(MersenneTwister(seed + 1), class_pool_val, classes_val, xs_val, npoints, min_digits, max_digits, n_samples)
    end

    if point_mask   # (1, 1, L) slot mask -> (1, npoints, L) point mask (all points of a valid slot are valid)
        train_data = sample_on_fly ? mapobs(_with_point_mask, train_data) : _with_point_mask(train_data)
        test_data = _with_point_mask(test_data)
        validation && (val_data = _with_point_mask(val_data))
    end
    return validation ? (train_data, val_data, test_data) : (train_data, test_data)
end

"""
`_with_point_mask((data, mask, labels))`

Repeat the slot mask `(1, 1, L[, BS])` of a `mnist_clock` bag (or stacked bags) over the point dimension of `data`,
giving the point mask `(1, N, L[, BS])` used by `mnist_clock_unbalanced` / `StructuredMnistHSQVAE`.
"""
function _with_point_mask(t::Tuple)
    data, mask, labels = t
    reps = ntuple(i -> i == 2 ? size(data, 2) ÷ size(mask, 2) : 1, ndims(mask))
    return data, repeat(mask, reps...), labels
end

# ---- mnist_clock_unbalanced: variable number of points per digit ------------------------------------------------------

"""
`_scaled_point_counts(n_natural::AbstractVector{Int}, npoints::Int, ref_points::Int, min_points::Int)`

Point count of every digit instance for `mnist_clock_unbalanced`: the natural (ink-proportional) count scaled so that
the largest digit (`ref_points` natural points) gets `npoints`, i.e.
`n_i = clamp(round(Int, n_natural_i * npoints / ref_points), min_points, npoints)`. A property of the instance, not of
the draw, so the same digit always has the same cardinality.
"""
_scaled_point_counts(n_natural::AbstractVector{Int}, npoints::Int, ref_points::Int, min_points::Int) =
    clamp.(round.(Int, n_natural .* (npoints / ref_points)), min_points, npoints)

"""
`sample_one_unbalanced_bag(rng, class_pool::Dict, classes::AbstractVector, xs::AbstractVector{<:AbstractMatrix}, ns::AbstractVector{Int}, npoints::Int, min_digits::Int, max_digits::Int)`

Like [`sample_one_bag`](@ref), but every digit has its own number of points `ns[inst] <= npoints` (see
[`_scaled_point_counts`](@ref)), freshly subsampled (without replacement) from the instance's natural point cloud
`xs[inst]`. Valid points occupy positions `1:ns[inst]` of the slot; the rest of the slot, and padding slots, are zero.

Returns:
- `data::Array{Float32,3}`: `(D, npoints, max_digits)`, zero in padded points and padding slots.
- `mask::BitArray{3}`: `(1, npoints, max_digits)`, `true` for valid points (a padding slot is all `false`, so
  `any(mask; dims=2)` gives the slot mask of `mnist_clock`).
- `labels::Vector{Int}`: `(max_digits,)`, class label in valid slots, `-1` in padding slots.
"""
function sample_one_unbalanced_bag(rng, class_pool::Dict, classes::AbstractVector, xs::AbstractVector{<:AbstractMatrix}, ns::AbstractVector{Int}, npoints::Int, min_digits::Int, max_digits::Int)
    D = size(first(xs), 1)
    data = zeros(Float32, D, npoints, max_digits)
    mask = falses(1, npoints, max_digits)
    labels = fill(-1, max_digits)

    n = rand(rng, min_digits:max_digits)
    for j in 1:n
        c = rand(rng, classes)
        inst = rand(rng, class_pool[c])
        k = ns[inst]
        pt_idx = sample(rng, axes(xs[inst], 2), k, replace=false)
        data[:, 1:k, j] .= xs[inst][:, pt_idx]
        mask[1, 1:k, j] .= true
        labels[j] = c
    end
    return data, mask, labels
end

"""
`_build_unbalanced_bags(rng, class_pool, classes, xs, ns, npoints, min_digits, max_digits, n_samples)`

Eager (fixed) version of [`sample_one_unbalanced_bag`](@ref), stacked over `n_samples` bags.

Returns `data (D, npoints, max_digits, n_samples)`, `mask::BitArray{4} (1, npoints, max_digits, n_samples)` and
`labels (max_digits, n_samples)`.
"""
function _build_unbalanced_bags(rng, class_pool::Dict, classes::AbstractVector, xs::AbstractVector{<:AbstractMatrix}, ns::AbstractVector{Int}, npoints::Int, min_digits::Int, max_digits::Int, n_samples::Int)
    D = size(first(xs), 1)
    data = zeros(Float32, D, npoints, max_digits, n_samples)
    mask = falses(1, npoints, max_digits, n_samples)
    labels = fill(-1, max_digits, n_samples)
    for i in 1:n_samples
        d, m, l = sample_one_unbalanced_bag(rng, class_pool, classes, xs, ns, npoints, min_digits, max_digits)
        data[:, :, :, i] .= d
        mask[:, :, :, i] .= m
        labels[:, i] .= l
    end
    return data, mask, labels
end

"""
`load_mnist_clock_unbalanced(npoints=512; n_samples=60000, min_digits=2, max_digits=12, min_points=1, validation=true, normalize=false, sample_on_fly=false, ratio=0.2, seed=666, kwargs...)`

Unbalanced version of [`load_mnist_clock`](@ref): the same bags of `min_digits`..`max_digits` MNIST digits, but every
digit has its own number of points, given by its natural ink-proportional count (natural MNIST point clouds, 928..6320
points) scaled so that the largest digit in the dataset gets exactly `npoints` (thin "1"s get the fewest points,
bold "0"/"8"s the most). Points are zero-padded to `npoints` inside each slot and the mask marks valid **points**:

- `data (D, npoints, max_digits, BS)`, `mask (1, npoints, max_digits, BS)`, `labels (max_digits, BS)`.

The 4-D point mask is what `StructuredMnistHSQVAE` and the masked `chamfer_pairwise_distance` take; the slot mask
of `mnist_clock` is `any(mask; dims=2)`. Split by digit instance before any bag is built (as in `load_mnist_clock`).

Keyword arguments as in [`load_mnist_clock`](@ref), plus:
- `min_points`: lower bound on the per-digit point count (default 1; at `npoints=512` the smallest digit already
  gets ~75 points, so it only matters for small `npoints`).

Notes:
- The per-digit count is a property of the instance (fixed), only *which* points are taken is random. With
  `sample_on_fly=true` the training points (and bag composition) are redrawn on every access.
- `npoints` must not exceed the largest natural count (6320), otherwise a digit could need more points than it has.
"""
function load_mnist_clock_unbalanced(npoints=512; n_samples::Int=60000, min_digits::Int=2, max_digits::Int=12, min_points::Int=1, validation::Bool=true, normalize::Union{Bool,AbstractString,Symbol}=false, sample_on_fly::Bool=false, ratio::AbstractFloat=0.2, seed::Int=666, kwargs...)

    dict_loaded = Serialization.deserialize(_mnist_natural_path())
    xs = dict_loaded["features"]       # Vector of (D, n_natural_i) matrices
    ys = dict_loaded["targets"]
    n_natural = size.(xs, 2)
    ref_points = maximum(n_natural)
    @assert npoints <= ref_points "Number of requested points ($npoints) is greater than the largest natural digit ($ref_points)."
    @assert 1 <= min_points <= npoints "min_points ($min_points) must be in 1:npoints ($npoints)."
    ns = _scaled_point_counts(n_natural, npoints, ref_points, min_points)

    # Optional preprocessing hook.
    xs = _normalize_mnist(xs, normalize)   # false | true (legacy) | "global"

    # Train/valid/test split by digit instance, BEFORE any bag is built (same split as load_mnist_clock).
    train_idx, val_idx, test_idx = _instance_split(length(ys), seed; validation=validation, ratio=ratio)

    pool(idx) = (xs[idx], ns[idx], Dict(c => findall(==(c), ys[idx]) for c in unique(ys[idx])))

    xs_train, ns_train, class_pool_train = pool(train_idx)
    classes_train = collect(keys(class_pool_train))
    train_data = if sample_on_fly
        mapobs(_ -> sample_one_unbalanced_bag(Random.default_rng(), class_pool_train, classes_train, xs_train, ns_train, npoints, min_digits, max_digits), 1:n_samples)
    else
        _build_unbalanced_bags(MersenneTwister(seed), class_pool_train, classes_train, xs_train, ns_train, npoints, min_digits, max_digits, n_samples)
    end

    # Validation and test are always fixed/pre-sampled, for reproducible evaluation.
    xs_test, ns_test, class_pool_test = pool(test_idx)
    test_data = _build_unbalanced_bags(MersenneTwister(seed + 2), class_pool_test, collect(keys(class_pool_test)), xs_test, ns_test, npoints, min_digits, max_digits, n_samples)

    if validation
        xs_val, ns_val, class_pool_val = pool(val_idx)
        val_data = _build_unbalanced_bags(MersenneTwister(seed + 1), class_pool_val, collect(keys(class_pool_val)), xs_val, ns_val, npoints, min_digits, max_digits, n_samples)
        return train_data, val_data, test_data
    end

    return train_data, test_data
end
