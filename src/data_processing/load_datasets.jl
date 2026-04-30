
function load_dataset(name::String, args...; kwargs...)
    if name == "modelnet10"
        return load_modelnet10(args...; kwargs...)
    elseif name == "mnist"
        return load_mnist(args...; kwargs...)
    #elseif name == "mnist_standardized"
    #    return load_and_standardize_mnist()
    #elseif name == "mnist_scaled"
    #    return load_and_scale_mnist()
    else
        error("Unknown dataset: $name")
    end
end


_mnist_balanced_path() = datadir("datasets/mnist_pc/mnist_4x_point_clouds_3x900_matrix.jls")
_mnist_natural_path() = datadir("datasets/mnist_pc/mnist_4x_point_clouds_all_vec.jls")

sample_fixed_n(pc, npoints) = pc[:, sample(axes(pc, 2), min(npoints, size(pc, 2)), replace=false)]
sample_fixed_n_unsqueeze(pc, npoints) = unsqueeze(pc[:, sample(axes(pc, 2), min(npoints, size(pc, 2)), replace=false)], dims=3)

function sample_fixed_n_from_matrix(xs::AbstractArray, npoints::Int)
    idx = sample(axes(xs, 2), min(npoints, size(xs, 2)), replace=false)
    return xs[:, idx, :]
end

function _stack_point_clouds(pcs)
    n = length(pcs)
    n == 0 && error("Cannot stack an empty point-cloud collection")
    d = size(pcs[1], 1)
    p = size(pcs[1], 2)
    x = Array{eltype(pcs[1])}(undef, d, p, n)
    @inbounds for (i, pc) in pairs(pcs)
        x[:, :, i] = pc
    end
    return x
end

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


function load_mnist(npoints=512, trans_fn=identity; validation::Bool=true, cardinality_count::Symbol=:balanced, sample_on_fly::Bool=false, normalize::Bool=false, ratio::AbstractFloat=0.2, seed::Int=666)
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
        if xs isa AbstractArray{<:Real,3}
            xs = trans_fn(xs)
        else
            xs = map(trans_fn, xs)
        end
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

function on_fly_collate_fn(batch::Vector{Tuple{X, Y}}) where {X <: AbstractArray{<:AbstractFloat, 3}, Y <: Int}
    pc1, y1 = batch[1]
    BS = length(batch)
    n_points = size(pc1, 2)
    x = Array{eltype(pc1)}(undef, size(pc1, 1), n_points, BS)
    y = Vector{typeof(y1)}(undef, BS)
    @inbounds for (i, (pc, label)) in pairs(batch)
        copyto!(@view(x[:, :, i]), pc)
        y[i] = label
    end
    return (x, y)
end

function create_dataloaders(data_cfg; batch_size::Int=32, train_collate_fn=nothing, valid_collate_fn=nothing, test_collate_fn=nothing)
    dataset_name = String(_cfgget(data_cfg, :dataset, "mnist"))
    npoints = _cfgget(data_cfg, :npoints, 512)
    trans_fn = _cfgget(data_cfg, :trans_fn, identity)
    validation = _cfgget(data_cfg, :validation, true)
    cardinality_count = Symbol(_cfgget(data_cfg, :cardinality_count, :balanced))
    sample_on_fly = _cfgget(data_cfg, :sample_on_fly, false)
    normalize = _cfgget(data_cfg, :normalize, false)
    ratio = _cfgget(data_cfg, :ratio, 0.2)
    seed = _cfgget(data_cfg, :seed, 666)

    # Support special positional args for some datasets (e.g. ModelNet10 expects a `type` positional arg)
    type_name = _cfgget(data_cfg, :type, "all")
    model_path = _cfgget(data_cfg, :model_path, nothing)

    if dataset_name == "modelnet10"
        data = load_dataset(
            dataset_name,
            npoints,
            type_name;
            validation=validation,
            ratio=ratio,
            seed=seed,
            path=model_path,
        )
    else
        data = load_dataset(
            dataset_name,
            npoints,
            trans_fn;
            validation=validation,
            cardinality_count=cardinality_count,
            sample_on_fly=sample_on_fly,
            normalize=normalize,
            ratio=ratio,
            seed=seed,
        )
    end

    train_data = data[1]
    valid_data = validation ? data[2] : nothing
    test_data = validation ? data[3] : data[2]

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



function load_modelnet10(npoints=2048, type="all"; validation::Bool=true, ratio=0.2, seed::Int=666, path::Union{Nothing,String}=nothing)
    """
    npoints     ... Number of points per object ( 512 / 1024 / 2048 )
    type        ... Type data -> \"all\" or one-class name e.g. \"chair\", \"monitor\"
    validatoin  ... Return validation set (\"true\") or not (\"false\")
    seed        ... Random seed for validation split.
    """
    #load data
    path_to_open = isnothing(path) ? datadir("datasets/modelnet10/modelnet10_$(npoints).h5") : path
    data = HDF5.h5open(path_to_open)
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
