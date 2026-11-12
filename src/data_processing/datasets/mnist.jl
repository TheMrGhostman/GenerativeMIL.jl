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

"""
    load_mnist(npoints=512; validation=true, cardinality_count=:balanced,
               sample_on_fly=false, normalize=false, ratio=0.2, seed=666, kwargs...)

Load MNIST point-cloud data and create train/validation/test splits.

# Arguments
- `npoints`: number of points requested after sampling.
- `validation`: if `true`, returns train/val/test; otherwise train/test.
- `cardinality_count`: `:balanced` or `:natural` source format.
- `sample_on_fly`: enable lazy sampling on the training split.
- `normalize`: apply point-cloud normalization before splitting.
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
function load_mnist(npoints=512; validation::Bool=true, cardinality_count::Symbol=:balanced, sample_on_fly::Bool=false, normalize::Bool=false, ratio::AbstractFloat=0.2, seed::Int=666, kwargs...)
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
    if normalize
        xs = normalize_point_cloud(xs)
    end

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
