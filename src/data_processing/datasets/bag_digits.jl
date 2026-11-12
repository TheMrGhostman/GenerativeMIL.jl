# Bags of categorical (one-hot) digits: bag_digits
# (included from DataProcessing.jl; dispatcher and create_dataloaders are in load_datasets.jl)

"""
`sample_one_digit_bag(rng, n_classes, min_card, max_card)`

Draw one bag of `rand(rng, min_card:max_card)` categorical elements ("digits") uniformly from
`1:n_classes` (duplicates allowed), one-hot encoded and zero-padded to `max_card` slots.

Returns:
- `x::Matrix{Float32}` `(n_classes, max_card)`, `mask::BitMatrix` `(1, max_card)`,
  `labels::Vector{Int}` (the drawn elements, length = cardinality).
"""
function sample_one_digit_bag(rng, n_classes::Int, min_card::Int, max_card::Int)
    len = rand(rng, min_card:max_card)
    elements = rand(rng, 1:n_classes, len)
    x = zeros(Float32, n_classes, max_card)
    mask = falses(1, max_card)
    x[:, 1:len] .= Flux.onehotbatch(elements, 1:n_classes)
    mask[1, 1:len] .= true
    return x, mask, elements
end

function _build_digit_bags(rng, n_classes::Int, min_card::Int, max_card::Int, n_samples::Int)
    x = zeros(Float32, n_classes, max_card, n_samples)
    mask = falses(1, max_card, n_samples)
    labels = Vector{Vector{Int}}(undef, n_samples)
    for b in 1:n_samples
        x[:, :, b], mask[:, :, b], labels[b] = sample_one_digit_bag(rng, n_classes, min_card, max_card)
    end
    return x, mask, labels
end

"""
`load_bag_digits(npoints=nothing; n_samples=10000, n_eval_samples=n_samples, n_classes=10, min_digits=1, max_digits=12, validation=true, sample_on_fly=false, seed=666, kwargs...)`

Synthetic discrete set dataset: each sample is a bag of `min_digits`..`max_digits` categorical
elements drawn uniformly from `1:n_classes` (duplicates allowed), one-hot encoded into a fixed
`(n_classes, max_digits)` slot tensor with a validity mask. Same `(data, mask, labels)` format as
[`load_mnist_clock`](@ref), one level lower (elements are one-hot vectors, not point clouds).

Arguments:
- `npoints`: ignored (kept so `create_dataloaders` can call every loader the same way).
- `n_samples`: number of training bags (steps per epoch when `sample_on_fly=true`).
- `n_eval_samples`: number of validation and test bags each.
- `n_classes`: alphabet size (one-hot dimension).
- `min_digits`, `max_digits`: inclusive bag cardinality range (`max_digits` is the slot count).
- `sample_on_fly`: if `true`, training bags are drawn fresh on every access (via `MLUtils.mapobs`).
  Validation/test are always fixed.
- `seed`: train/valid/test are built from `MersenneTwister(seed)`, `seed+1`, `seed+2`.

Returns:
- `(train_data, valid_data, test_data)` if `validation=true`, else `(train_data, test_data)`.
"""
function load_bag_digits(npoints=nothing; n_samples::Int=10000, n_eval_samples::Int=n_samples, n_classes::Int=10, min_digits::Int=1, max_digits::Int=12, validation::Bool=true, sample_on_fly::Bool=false, seed::Int=666, kwargs...)
    @assert 1 <= min_digits <= max_digits "need 1 <= min_digits <= max_digits (got $min_digits, $max_digits)"
    train_data = if sample_on_fly
        mapobs(_ -> sample_one_digit_bag(Random.default_rng(), n_classes, min_digits, max_digits), 1:n_samples)
    else
        _build_digit_bags(MersenneTwister(seed), n_classes, min_digits, max_digits, n_samples)
    end
    test_data = _build_digit_bags(MersenneTwister(seed + 2), n_classes, min_digits, max_digits, n_eval_samples)
    if validation
        val_data = _build_digit_bags(MersenneTwister(seed + 1), n_classes, min_digits, max_digits, n_eval_samples)
        return train_data, val_data, test_data
    end
    return train_data, test_data
end
