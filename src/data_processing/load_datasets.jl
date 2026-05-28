
"""
    load_dataset(name::String, args...; kwargs...)

Dispatch dataset loading based on dataset name.

# Arguments
- `name`: dataset identifier.
    - `"modelnet10"` -> [`load_modelnet10`](@ref)
    - `"mnist"` -> [`load_mnist`](@ref)
- `args...`, `kwargs...`: forwarded to the selected loader.

# Returns
- The value returned by the selected dataset loader.

# Throws
- `ErrorException` when `name` is unknown.
"""
function load_dataset(name::String, args...; kwargs...)
    if name == "modelnet10"
        return load_modelnet10(args...; kwargs...)
    elseif name == "mnist"
        return load_mnist(args...; kwargs...)
    elseif name == "modelnet10_flux3d"
        return load_modelnet10_flux3d(args...; kwargs...)
    else
        error("Unknown dataset: $name")
    end
end

"""
    _cfgget(cfg, key::Symbol, default)

Fetch a configuration value from a dictionary-like object or struct.

# Arguments
- `cfg`: configuration object (`AbstractDict` or a struct-like object).
- `key`: requested key as `Symbol`.
- `default`: fallback value when `key` is missing.

# Returns
- `cfg[key]` if present,
- otherwise `cfg[String(key)]` for dictionary inputs,
- otherwise `getproperty(cfg, key)` for struct-like inputs,
- otherwise `default`.
"""
function _cfgget(cfg, key::Symbol, default)
    if cfg isa AbstractDict
        if haskey(cfg, key)
            return cfg[key]
        end
        skey = String(key)
        return haskey(cfg, skey) ? cfg[skey] : default
    end
    return hasproperty(cfg, key) ? getproperty(cfg, key) : default
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
    rng_split = MersenneTwister(seed)
    perm = randperm(rng_split, length(y))
    n_train_test = round(Int, 0.8 * length(y))
    train_val_idx = perm[1:n_train_test]
    test_idx = perm[n_train_test+1:end]

    if validation
        n_train = round(Int, (1 - ratio) * length(train_val_idx))
        train_idx = train_val_idx[1:n_train]
        val_idx = train_val_idx[n_train+1:end]
    else
        train_idx = train_val_idx
        val_idx = Int[]
    end

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


"""
    load_modelnet10(npoints=2048; type="all", validation=true, ratio=0.2,
                    seed=666, balanced_classes=false, sample_on_fly=false,
                    normalize=false, kwargs...)

Load the selected ModelNet point-cloud dataset and create train/validation/test splits.

# Arguments
- `npoints`: number of points requested after sampling.
- `validation`: if `true`, returns train/val/test; otherwise train/test.
- `ratio`: validation ratio from the training split.
- `seed`: random seed for deterministic split/sampling behavior.
- `balanced_classes`: choose the balanced or unbalanced dataset file.
- `sample_on_fly`: enable lazy sampling on the training split.
- `normalize`: apply point-cloud normalization before splitting.
- `kwargs...`: reserved for API compatibility.

# Returns
- If `validation=true`:
    - `((x_train, y_train), (x_val, y_val), (x_test, y_test))`
- If `validation=false`:
    - `((x_train, y_train), (x_test, y_test))`

# Notes
- The test split is fixed and is loaded directly from the dataset file.
- `npoints` must be at most `8196`.
- On-the-fly sampling is applied only to training data.
"""
function load_modelnet10(npoints=2048; validation::Bool=true, ratio::AbstractFloat=0.2, seed::Int=666, balanced_classes::Bool=false, sample_on_fly::Bool=false, normalize::Bool=false, kwargs...)
    npoints <= 8196 || error("Number of requested points ($npoints) is greater than the dataset maximum (8196).")

    dict_loaded = Serialization.deserialize(_modelnet10_path(balanced_classes))
    train_split = dict_loaded.train
    test_split = dict_loaded.test

    x_train_full = train_split["features"]
    y_train_full = train_split["targets"]
    #train_classes = train_split["classes"]
    x_test_full = test_split["features"]
    y_test_full = test_split["targets"]
    #test_classes = test_split["classes"]

    if normalize
        # normalization to unit shpere. normal standardization does not make sense
        # because you normalize airplanes and house plants together. model can not learn that
        x_train_full = normalize_point_clouds_into_unit_shpere(x_train_full)
        x_test_full = normalize_point_clouds_into_unit_shpere(x_test_full)
    end

    rng_split = MersenneTwister(seed)
    perm = randperm(rng_split, length(y_train_full))

    if validation
        n_train = round(Int, (1 - ratio) * length(perm))
        train_idx = perm[1:n_train]
        val_idx = perm[n_train+1:end]
    else
        train_idx = perm
        val_idx = Int[]
    end

    y_train = y_train_full[train_idx]
    y_test = y_test_full

    if validation
        y_val = y_train_full[val_idx]
    end

    x_train_subset = x_train_full[:, :, train_idx]
    x_test_subset = x_test_full

    Random.seed!(seed)
    if sample_on_fly
        x_train = mapobs(pc -> sample_fixed_n_from_matrix(pc, npoints), x_train_subset)
    else
        x_train = sample_fixed_n_from_matrix(x_train_subset, npoints)
    end

    x_test = sample_fixed_n_from_matrix(x_test_subset, npoints)

    if validation
        x_val_subset = x_train_full[:, :, val_idx]
        x_val = sample_fixed_n_from_matrix(x_val_subset, npoints)
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    end

    return (x_train, y_train), (x_test, y_test)
end


"""
    load_modelnet10_flux3d(npoints=2048; type="all", validation=true, ratio=0.2, seed=666, kwargs...)

Load ModelNet10 (from Flux3D.jl) point-cloud data and create train/validation/test splits.

# Arguments
- `npoints`: points per object (also determines source file variant).
- `type`: `"all"` or a specific class name (e.g. `"chair"`).
- `validation`: if `true`, returns train/val/test; otherwise train/test.
- `ratio`: validation ratio for the training split.
- `seed`: random seed for deterministic split.
- `kwargs...`: reserved for API compatibility.

# Returns
- If `validation=true`:
    - `((X_train, Y_train), (X_val, Y_val), (X_test, Y_test))`
- If `validation=false`:
    - `((X_train, Y_train), (X_test, Y_test))`

# Notes
- Data are loaded from `_modelnet10_flux3d_path(npoints)`.
- If `type` is a known class name, training data are filtered to that class,
    and `Y_val`/`Y_test` are converted to one-vs-rest binary targets.
"""
function load_modelnet10_flux3d(npoints=2048; type="all", validation::Bool=true, ratio=0.2, seed::Int=666, kwargs...)
    #load data
    data = HDF5.h5open(_modelnet10_flux3d_path(npoints))
    X_train, X_test, Y_train, Y_test = data["X_train"]|>read, data["X_test"]|>read, data["Y_train"]|>read, data["Y_test"]|>read

    titles = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]

    if validation
        #(X_train,Y_train), (X_val,Y_val) = train_test_split(X_train, Y_train, ratio, seed=seed)# splitobs((X_train, Y_train), at=ratio, shuffle=true)
        Random.seed!(seed) # Ensure reproducibility for the train/validation split.
        train_idx, val_idx = splitobs(axes(X_train, 3), at=ratio, shuffle=true)
        X_val = X_train[:, :, val_idx]
        Y_val = Y_train[val_idx]
        X_train = X_train[:, :, train_idx]
        Y_train = Y_train[train_idx]
        if type in titles
            class_idx = only(findall(titles .== type))
            X_train = X_train[:, :, Y_train .== class_idx]
            Y_train = Y_train[Y_train .== class_idx]
            Y_val = Y_val .!= class_idx
            Y_test = Y_test .!= class_idx
        end
        data = ((X_train, Y_train), (X_val, Y_val), (X_test, Y_test)) 
    else
        if type in titles
            class_idx = only(findall(titles .== type))
            X_train = X_train[:, :, Y_train .== class_idx]
            Y_train = Y_train[Y_train .== class_idx]
            Y_test = Y_test .!= class_idx
        end
        data = ((X_train, Y_train), (X_test, Y_test)) 
    end
    return data
end


"""
    create_dataloaders(data_cfg; batch_size=32,
                       train_collate_fn=nothing,
                       valid_collate_fn=nothing,
                       test_collate_fn=nothing)

Build train/validation/test `MLUtils.DataLoader`s from a dataset config.

# Arguments
- `data_cfg`: dictionary or struct-like config. Typical keys include:
    - `dataset`, `npoints`, `validation`, `ratio`, `seed`,
    - and dataset-specific options such as
        `cardinality_count`, `sample_on_fly`, `normalize`, `balanced_classes`, `type`.
- `batch_size`: dataloader batch size.
- `train_collate_fn`: optional train collate function.
- `valid_collate_fn`: optional validation collate function.
- `test_collate_fn`: optional test collate function.

# Returns
- Named tuple `(train, valid, test)` where:
    - `train` is a `DataLoader`,
    - `valid` is a `DataLoader` or `nothing` when `validation=false`,
    - `test` is a `DataLoader`.

# Notes
- If `sample_on_fly && cardinality_count == :natural`, default
    `on_fly_collate_fn` is used for training unless a custom collate function is
    provided.
"""
function create_dataloaders(data_cfg; batch_size::Int=32, x_only::Bool=false, train_collate_fn=nothing, valid_collate_fn=nothing, test_collate_fn=nothing)
    dataset_name = String(_cfgget(data_cfg, :dataset, "mnist"))
    npoints = _cfgget(data_cfg, :npoints, 512)
    validation = _cfgget(data_cfg, :validation, true)
    cardinality_count = Symbol(_cfgget(data_cfg, :cardinality_count, :balanced))
    balanced_classes = _cfgget(data_cfg, :balanced_classes, false)
    sample_on_fly = _cfgget(data_cfg, :sample_on_fly, false)
    normalize = _cfgget(data_cfg, :normalize, false)
    ratio = _cfgget(data_cfg, :ratio, 0.2)
    seed = _cfgget(data_cfg, :seed, 666)

    # Support special positional args for some datasets (e.g. ModelNet10 expects a `type` positional arg)
    type_name = _cfgget(data_cfg, :type, "all")

    data = load_dataset(
        dataset_name,
        npoints;
        validation=validation,
        cardinality_count=cardinality_count,
        balanced_classes=balanced_classes,
        sample_on_fly=sample_on_fly,
        normalize=normalize,
        ratio=ratio,
        seed=seed,
        type=type_name
    )

    if x_only
        train_data = data[1][1]
        valid_data = validation ? data[2][1] : nothing
        test_data = validation ? data[3][1] : data[2][1]
    else
        train_data = data[1]
        valid_data = validation ? data[2] : nothing
        test_data = validation ? data[3] : data[2]

    end

    if sample_on_fly && cardinality_count == :natural
        train_collate_fn = isnothing(train_collate_fn) ? on_fly_collate_fn : train_collate_fn
    end

    train_loader = isnothing(train_collate_fn) ? DataLoader(train_data, batchsize=batch_size, shuffle=true) : DataLoader(train_data, batchsize=batch_size, shuffle=true, collate=train_collate_fn)
    valid_loader = if validation
        isnothing(valid_collate_fn) ? DataLoader(valid_data, batchsize=batch_size, shuffle=false) : DataLoader(valid_data, batchsize=batch_size, shuffle=false, collate=valid_collate_fn)
    else
        nothing
    end

    test_loader = isnothing(test_collate_fn) ? DataLoader(test_data, batchsize=batch_size, shuffle=false) : DataLoader(test_data, batchsize=batch_size, shuffle=false, collate=test_collate_fn)

    return (train=train_loader, valid=valid_loader, test=test_loader)
end
