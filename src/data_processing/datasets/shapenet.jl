# ShapeNet point clouds: load_shapenet_class, load_shapenet_multiple_classes
# (included from DataProcessing.jl; dispatcher and create_dataloaders are in load_datasets.jl)

"""
    load_shapenet_class(classname::String, npoints::Int=2048; normalize=false, 
                        validation=true, sample_on_fly=false, set_y_for_class=0)

Load train/validation/test splits for a single ShapeNet class from a serialized
subset file.

# Arguments
- `classname`: class name stored in the file name, for example `"car"`.
- `npoints`: number of points to sample from each split (max 15000).
- `normalize`: if `true`, normalize the full class dataset using shared
    statistics across all points in all splits before sampling.
- `validation`: if `true`, returns train/val/test; otherwise train/test.
- `sample_on_fly`: if `true`, apply lazy sampling on the training split.
- `set_y_for_class`: label value for all samples.

# Returns
- If `validation=true`:
    - `((x_train, y_train), (x_val, y_val), (x_test, y_test))`
- If `validation=false`:
    - `((x_train, y_train), (x_test, y_test))`

# Notes
- Validation and test splits are always pre-sampled.
- On-the-fly sampling is applied only to training data when enabled.
"""
function load_shapenet_class(classname::String, npoints::Int=2048; normalize::Bool=false, validation::Bool=true, sample_on_fly::Bool=false, set_y_for_class::Int=0, kwargs...)
    path = _shapenet_class_15K_path(classname)
    isfile(path) || error("Unknown ShapeNet class subset: $(classname) (missing file: $(path))")

    data = Serialization.deserialize(path)
    hasproperty(data, :train) || error("Invalid ShapeNet subset file: missing `train` split at $(path)")
    hasproperty(data, :valid) || error("Invalid ShapeNet subset file: missing `valid` split at $(path)")
    hasproperty(data, :test) || error("Invalid ShapeNet subset file: missing `test` split at $(path)")

    x_train = data.train
    x_valid = data.valid
    x_test = data.test

    @assert npoints <= size(x_train, 2) "Number of requested points ($npoints) is greater than the dataset cardinality ($(size(x_train, 2)))."

    if normalize
        x_train, x_valid, x_test = _normalize_point_cloud_dataset(x_train, x_valid, x_test)
    end

    y_train = fill(set_y_for_class, axes(x_train, 3))
    y_valid = fill(set_y_for_class, axes(x_valid, 3))
    y_test  = fill(set_y_for_class, axes(x_test , 3))

    if sample_on_fly
        x_train = mapobs(pc -> sample_fixed_n_from_matrix(pc, npoints), x_train)
    else
        x_train = sample_fixed_n_from_matrix(x_train, npoints)
    end
    x_valid = sample_fixed_n_from_matrix(x_valid, npoints)
    x_test  = sample_fixed_n_from_matrix(x_test , npoints)

    if validation
        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
    else
        return (x_train, y_train), (x_test, y_test)
    end
end

function load_shapenet_class(npoints::Int=2048; type::AbstractString="car", kwargs...)
    return load_shapenet_class(String(type), npoints; kwargs...)
end


"""
    load_shapenet_multiple_classes(classes, npoints::Int=2048; normalize=false,
                                    validation=true, sample_on_fly=false,
                                    balanced_classes=true, verbose=false, kwargs...)

Load train/validation/test splits for multiple ShapeNet classes and merge them into single tensors.

# Arguments
- `classes`: vector of class names to load, e.g. `["car", "chair", "airplane"]`.
- `npoints`: number of points to sample from each split (max 15000).
- `normalize`: if `true`, normalize the full merged dataset using shared statistics
    across all points in all splits before sampling.
- `validation`: if `true`, returns train/val/test; otherwise train/test.
- `sample_on_fly`: if `true`, apply lazy sampling on the training split.
- `balanced_classes`: if `true`, oversample smaller classes in training set to match
    the largest class cardinality.
- `verbose`: if `true`, enable verbose output during normalization.
- `kwargs...`: reserved for API compatibility.

# Returns
- If `validation=true`:
    - `((x_train, y_train), (x_val, y_val), (x_test, y_test))`
- If `validation=false`:
    - `((x_train, y_train), (x_test, y_test))`
- Each `x_*` tensor has shape `(D, N, BS)` where `D` is dimensionality (typically 3),
    `N` is the sampled point count, and `BS` is batch size.
- Each `y_*` label vector contains class indices (1-indexed by enumeration order).

# Notes
- Validation and test splits are always pre-sampled.
- On-the-fly sampling is applied only to training data when enabled.
- Class labels are assigned based on enumeration order: class at index `i` gets label `i`.
- When `balanced_classes=true`, smaller classes are oversampled with replacement.
"""
function load_shapenet_multiple_classes(classes, npoints::Int=2048; normalize::Bool=false, validation::Bool=true, sample_on_fly::Bool=false, balanced_classes::Bool=true, verbose::Bool=false, upper_bound_n::Int=10000, kwargs...)
    #classes = ["car", "chair", "airplane", "lamp", "table"]
    data_train = Dict{String, Array{Float32, 3}}()
    data_valid = Dict{String, Array{Float32, 3}}()
    data_test = Dict{String, Array{Float32, 3}}()
    dclasses = Dict{String, Int}()
    for (i, classname) in enumerate(classes)
        path = _shapenet_class_15K_path(classname)
        isfile(path) || error("Unknown ShapeNet class subset: $(classname) (missing file: $(path))")
        
        data = Serialization.deserialize(path)
        hasproperty(data, :train) || error("Invalid ShapeNet subset file: missing `train` split at $(path)")
        hasproperty(data, :valid) || error("Invalid ShapeNet subset file: missing `valid` split at $(path)")
        hasproperty(data, :test) || error("Invalid ShapeNet subset file: missing `test` split at $(path)")

        data_train[classname] = data.train
        data_valid[classname] = data.valid
        data_test[classname] = data.test
        dclasses[classname] = i
        verbose && @info "class: $(classname) -> $(size(data.train))"
        #data[classname] = load_shapenet_class(classname, npoints; normalize, validation, sample_on_fly, kwargs...)
    end

    if balanced_classes # balance the number of samples in each class (in training set) by oversampling the smaller classes with replacement
        max_n = min(maximum(size.(values(data_train), 3)), upper_bound_n)
        #y_train = Int[]
        for classname in classes
            n = size(data_train[classname], 3)
            if n < max_n
                verbose && @info "upsampling class $(classname) | from n=$(n) to n̂ = $(max_n)"
                idxs = ((max_n - n) > n) ? sample(1:n, max_n - n, replace=true) : sample(1:n, max_n - n, replace=false)
                data_train[classname] = cat(data_train[classname], data_train[classname][:, :, idxs]; dims=3)
            else
                verbose && @info "downsampling class $(classname) | from n=$(n) to n̂ = $(max_n)"
                idxs = (n==max_n) ? (1:n) : sample(1:n, max_n, replace=false)
                data_train[classname] = data_train[classname][:, :, idxs]
            end
            #append!(y_train, fill(dclasses[classname], size(data_train[classname], 3)))
        end
    end

    # merge training train/valid/test dicts into single tensors
    x_train, y_train = _merge_dict_into_tensor(data_train, dclasses)
    x_valid, y_valid = _merge_dict_into_tensor(data_valid, dclasses)
    x_test, y_test = _merge_dict_into_tensor(data_test, dclasses)

    @assert npoints <= size(x_train, 2) "Number of requested points ($npoints) is greater than the dataset cardinality ($(size(x_train, 2)))."

    if normalize
        x_train, x_valid, x_test = _normalize_point_cloud_dataset(x_train, x_valid, x_test; verbose=verbose)
    end

    if sample_on_fly
        x_train = mapobs(pc -> sample_fixed_n_from_matrix(pc, npoints), x_train)
    else
        x_train = sample_fixed_n_from_matrix(x_train, npoints)
    end
    x_valid = sample_fixed_n_from_matrix(x_valid, npoints)
    x_test  = sample_fixed_n_from_matrix(x_test , npoints)

    if validation
        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
    else
        return (x_train, y_train), (x_test, y_test)
    end
end


function load_shapenet_multiple_classes(npoints::Int=2048; type::AbstractString="core5", verbose::Bool=false, kwargs...)
    classes = if type == "core5"
        ["car", "chair", "airplane", "sofa", "table"]
    elseif type == "core10"
        ["car", "chair", "airplane", "sofa", "table", "display", "lamp", "loudspeaker", "rifle", "watercraft"]
    else
        error("Unknown set classes name. For now only `core5` and `core10` are recognised")
    end
    verbose && @info "classes: $(classes)"
    return load_shapenet_multiple_classes(classes, npoints; verbose=verbose, kwargs...)
end
