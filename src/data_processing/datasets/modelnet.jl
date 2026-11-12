# ModelNet10 point clouds: load_modelnet10, load_modelnet10_flux3d
# (included from DataProcessing.jl; dispatcher and create_dataloaders are in load_datasets.jl)

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
