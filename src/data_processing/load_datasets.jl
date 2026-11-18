# Dataset dispatcher (`load_dataset`), config helper (`_cfgget`) and `create_dataloaders`.
# The individual loaders live in datasets/*.jl, one file per dataset family.

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
    elseif name == "shapenet_class"
        return load_shapenet_class(args...; kwargs...)
    elseif name == "shapenet_multiple_classes"
        return load_shapenet_multiple_classes(args...; kwargs...)
    elseif name == "mnist_clock"
        return load_mnist_clock(args...; kwargs...)
    elseif name == "mnist_clock_unbalanced"
        return load_mnist_clock_unbalanced(args...; kwargs...)
    elseif name == "bag_digits"
        return load_bag_digits(args...; kwargs...)
    elseif name == "topic_bags"
        return load_topic_bags(args...; kwargs...)
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
    create_dataloaders(data_cfg; batch_size=32,
                       train_collate_fn=nothing,
                       valid_collate_fn=nothing,
                       test_collate_fn=nothing)

Build train/validation/test `MLUtils.DataLoader`s from a dataset config.

# Arguments
- `data_cfg`: dictionary or struct-like config. Typical keys include:
    - `dataset`, `npoints`, `validation`, `ratio`, `seed`,
    - and dataset-specific options such as
        `cardinality_count`, `sample_on_fly`, `normalize`, `balanced_classes`, `type`,
        and (for `dataset="mnist_clock"`) `n_samples`, `min_digits`, `max_digits`
        (`dataset="mnist_clock_unbalanced"` additionally `min_points`; `dataset="mnist_clock"` optionally
        `point_mask=true` for models that need a `(1, npoints, L)` point mask).
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
- If `sample_on_fly && dataset == "mnist_clock"`, default `bag_collate_fn` is used for training
    unless a custom collate function is provided (see `load_mnist_clock`).
- Else if `sample_on_fly && cardinality_count == :natural`, default
    `on_fly_collate_fn` is used for training unless a custom collate function is
    provided.
- `x_only=true` is not meaningful for `dataset="mnist_clock"` (the mask isn't optional metadata
    the way a label is) -- leave it `false` (the default) for that dataset.
"""
function create_dataloaders(data_cfg; batch_size::Int=32, x_only::Bool=false, train_collate_fn=nothing, valid_collate_fn=nothing, test_collate_fn=nothing, verbose::Bool=false)
    dataset_name = String(_cfgget(data_cfg, :dataset, "mnist"))
    npoints = _cfgget(data_cfg, :npoints, 512)
    validation = _cfgget(data_cfg, :validation, true)
    cardinality_count = Symbol(_cfgget(data_cfg, :cardinality_count, :balanced))
    balanced_classes = _cfgget(data_cfg, :balanced_classes, false)
    upper_bound_n = _cfgget(data_cfg, :upper_bound_n, 10000)
    sample_on_fly = _cfgget(data_cfg, :sample_on_fly, false)
    normalize = _cfgget(data_cfg, :normalize, false)
    ratio = _cfgget(data_cfg, :ratio, 0.2)
    seed = _cfgget(data_cfg, :seed, 666)

    # Support special positional args for some datasets (e.g. ModelNet10 expects a `type` positional arg)
    type_name = _cfgget(data_cfg, :type, "all")

    # mnist_clock-specific (bag-of-digits) options -- ignored via kwargs... by every other loader.
    n_samples = _cfgget(data_cfg, :n_samples, 60000)
    min_digits = _cfgget(data_cfg, :min_digits, 2)
    max_digits = _cfgget(data_cfg, :max_digits, dataset_name == "topic_bags" ? 16 : 12)  # topic_bags needs 16 slots
    # bag_digits-specific options
    n_eval_samples = _cfgget(data_cfg, :n_eval_samples, n_samples)
    n_classes = _cfgget(data_cfg, :n_classes, 10)
    # mnist_clock_unbalanced-specific option
    min_points = _cfgget(data_cfg, :min_points, 1)
    # mnist with cardinality_count "unbalanced": training point count drawn from k:npoints (digit classifier)
    augment_points = _cfgget(data_cfg, :augment_points, false)
    # mnist_clock: point mask (1, npoints, L) instead of slot mask (1, 1, L), needed by StructuredMnistHSQVAE
    point_mask = _cfgget(data_cfg, :point_mask, false)
    # masked bag datasets returning (data, mask, labels) triples
    is_bag_dataset = dataset_name in ("mnist_clock", "mnist_clock_unbalanced", "bag_digits", "topic_bags")

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
        type=type_name,
        upper_bound_n=upper_bound_n,
        n_samples=n_samples,
        min_digits=min_digits,
        max_digits=max_digits,
        n_eval_samples=n_eval_samples,
        n_classes=n_classes,
        min_points=min_points,
        augment_points=augment_points,
        point_mask=point_mask,
        verbose=verbose,
    )

    if x_only && is_bag_dataset
        # mask isn't optional metadata here (unlike a label) -- x_only means "drop labels",
        # not "drop the mask too", so this keeps (data, mask) rather than data alone.
        # data[1] (train) is a mapobs-wrapped lazy object when sample_on_fly, so data[1][1]/[2]
        # would mean "observation 1"/"observation 2", not "first/second field" -- needs its own
        # lazy (data,mask) wrap in that case. valid/test are always eager, so plain indexing is fine.
        train_data = sample_on_fly ? mapobs(t -> (t[1], t[2]), data[1]) : (data[1][1], data[1][2])
        valid_data = validation ? (data[2][1], data[2][2]) : nothing
        test_data = validation ? (data[3][1], data[3][2]) : (data[2][1], data[2][2])
    elseif x_only
        train_data = data[1][1]
        valid_data = validation ? data[2][1] : nothing
        test_data = validation ? data[3][1] : data[2][1]
    else
        train_data = data[1]
        valid_data = validation ? data[2] : nothing
        test_data = validation ? data[3] : data[2]
    end

    if sample_on_fly && is_bag_dataset
        train_collate_fn = isnothing(train_collate_fn) ? bag_collate_fn : train_collate_fn
    elseif sample_on_fly && dataset_name == "mnist" && cardinality_count == :unbalanced
        train_collate_fn = isnothing(train_collate_fn) ? masked_digit_collate_fn : train_collate_fn
    elseif sample_on_fly && cardinality_count == :natural
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
