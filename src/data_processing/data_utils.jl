"""
    normalize_point_cloud(pc::AbstractArray{T,3}) where T<:AbstractFloat

Normalize a batched point-cloud tensor feature-wise.

# Arguments
- `pc`: point-cloud tensor of shape `(D, N, BS)` where:
    - `D` is point dimensionality (typically 3),
    - `N` is the number of points,
    - `BS` is the number of samples.

# Returns
- A normalized tensor with the same shape `(D, N, BS)`.

# Notes
- Mean and standard deviation are computed independently per dimension,
    across points and samples.
"""
function normalize_point_cloud(pc::AbstractArray{T, 3}) where T<:AbstractFloat
    mu = mean(pc, dims=(2,3))
    sigma = std(pc, dims=(2,3))
    return (pc .- mu) ./ (sigma .+ eps(T))
end

"""
    normalize_point_cloud(pcs::Vector{<:AbstractArray{T,2}}) where T<:AbstractFloat

Normalize a vector of variable-cardinality point clouds.

# Arguments
- `pcs`: vector of point clouds, each of shape `(D, N_i)`.

# Returns
- A vector of normalized point clouds with unchanged per-sample shapes.

# Notes
- Global statistics are estimated per dimension over the whole collection.
"""
function normalize_point_cloud(pcs::Vector{<:AbstractArray{T, 2}}) where T<:AbstractFloat
     #TODO fixme
    d1 = getindex.(pcs, 1, :)
    d2 = getindex.(pcs, 2, :)
    d3 = getindex.(pcs, 3, :)
    mu = [mean(mean.(d1)), mean(mean.(d2)), mean(mean.(d3))]
    sigma = [mean(std.(d1)), mean(std.(d2)), mean(std.(d3))]
    return map(p -> (p .- mu) ./ (sigma .+ eps(T)), pcs)
end

"""
    _normalize_point_cloud_dataset(pcs::AbstractArray{T, 3}...) where T<:AbstractFloat

Normalize one or more batched point-cloud tensors with first tensor statistics with scalar sigma.
    μ = (μ_x, μ_y, x_z)
    σ = scalar
    X = (X .- μ) ./ σ

# Arguments
- `pcs`: one or more tensors of shape `(D, N, BS)`.

# Returns
- A tuple with each tensor normalized using the same per-dimension mean and
  standard deviation computed on first element from provided tensors.
"""
function _normalize_point_cloud_dataset(pcs::AbstractArray{T, 3}...; verbose::Bool=false) where T<:AbstractFloat
    isempty(pcs) && error("No point-cloud tensors provided for normalization")
    μ = mean(pcs[1], dims=(2,3))
    σ = std(vec(pcs[1]))
    verbose && println("μ = $(μ) \n σ = $(σ)")
    return tuple(((pc .- μ) ./ (σ .+ eps(T)) for pc in pcs)...)
end


"""
        normalize_point_clouds_into_unit_shpere(pc::AbstractArray{T, 3}) where T<:AbstractFloat

Normalize a batched collection of point clouds into the unit sphere per sample.

# Arguments
- `pc`: point-cloud tensor of shape `(D, N, BS)` where `D` is dimensionality,
    `N` number of points, and `BS` number of samples in the batch.

# Returns
- A tensor of the same shape `(D, N, BS)` where each sample has been:
    - translated so its centroid is at the origin, and
    - scaled so the farthest point from the centroid lies at distance `1`.

# Notes
- Scaling is performed per-sample by dividing by the maximum distance from the
    centroid to any point in that sample. To avoid division by zero, `eps(T)` is
    added to the denominator.
"""
function normalize_point_clouds_into_unit_shpere(pc::AbstractArray{T, 3}) where T<:AbstractFloat
    # pc is datasets of point clouds (D, N, BS), where D is dimension of pc, N is number of points, and BS is batch size or just number of all point clouds
    # Compute per-sample centroids (D x 1 x BS)
    μs = mean(pc, dims=2)
    # Center points per-sample
    centered = pc .- μs
    # Squared distances of each point to its sample centroid: (1 x N x BS)
    sqd = sum(abs2.(centered), dims=1)
    # Maximum squared distance per sample (1 x 1 x N), then take sqrt -> scale per-sample
    max_sq = maximum(sqd, dims=2)
    scale = sqrt.(max_sq)
    # Avoid division by zero, compute inverse scale broadcastable to (D,N,BS)
    inv_scale = 1 ./(scale .+ eps(T))
    # Return centered point clouds scaled so the farthest point is at distance 1
    return centered .* inv_scale
end


"""
    sample_fixed_n(pc, npoints)

Sample up to `npoints` points from a single point cloud without replacement.

# Arguments
- `pc`: point cloud of shape `(D, N)`.
- `npoints`: requested number of points.

# Returns
- Array of shape `(D, min(npoints, N))`.
"""
sample_fixed_n(pc, npoints) = pc[:, sample(axes(pc, 2), min(npoints, size(pc, 2)), replace=false)]

"""
    sample_fixed_n_unsqueeze(pc, npoints)

Equivalent to [`sample_fixed_n`](@ref), with a singleton third dimension.

# Arguments
- `pc`: point cloud of shape `(D, N)`.
- `npoints`: requested number of points.

# Returns
- Array of shape `(D, min(npoints, N), 1)`.
"""
sample_fixed_n_unsqueeze(pc, npoints) = unsqueeze(pc[:, sample(axes(pc, 2), min(npoints, size(pc, 2)), replace=false)], dims=3)

"""
    sample_fixed_n_from_matrix(xs::AbstractArray, npoints::Int)

Sample the point dimension of a batched tensor without replacement.

# Arguments
- `xs`: tensor of shape `(D, N, BS)`.
- `npoints`: requested number of points.

# Returns
- Tensor of shape `(D, min(npoints, N), BS)`.
"""
function sample_fixed_n_from_matrix(xs::AbstractArray, npoints::Int)
    idx = sample(axes(xs, 2), min(npoints, size(xs, 2)), replace=false)
    return xs[:, idx, :]
end

"""
    _stack_point_clouds(pcs)

Stack a vector of identically shaped 2D point clouds into one 3D tensor.

# Arguments
- `pcs`: vector of point clouds, each of shape `(D, N)`.

# Returns
- Tensor of shape `(D, N, BS)` where `BS = length(pcs)`.

# Throws
- `ErrorException` if `pcs` is empty.
"""
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


"""
    _merge_dict_into_tensor(dict_data::Dict{String, AbstractArray{T,3}}, dict_classes::Dict{String, Int})

Merge a dictionary of class-indexed point-cloud tensors into a single concatenated tensor.

# Arguments
- `dict_data`: dictionary mapping class names (strings) to 3D tensors of shape `(D, N, BS_i)`,
    where `D` is dimensionality, `N` is number of points, and `BS_i` is batch size for class `i`.
- `dict_classes`: dictionary mapping the same class names to their integer class labels.

# Returns
- Tuple `(x_merged, y_merged)` where:
    - `x_merged`: concatenated tensor of shape `(D, N, total_BS)` where `total_BS = sum(BS_i)`.
    - `y_merged`: label vector of shape `(total_BS,)` with class indices repeated according
        to the number of samples in each class.

# Throws
- `AssertionError` if class name sets differ between dictionaries.
- `AssertionError` if dimensions `D` or `N` are inconsistent across classes.

# Notes
- Classes are processed in sorted order to ensure deterministic output.
- This function acts as an efficient alternative to repeated `cat` operations.
"""
function _merge_dict_into_tensor(dict_data::Dict{String, Array{T,3}}, dict_classes::Dict{String, Int}) where T<: AbstractFloat
    data_keys = sort(collect(keys(dict_data)))
    classes_keys = sort(collect(keys(dict_classes)))
    @assert data_keys == classes_keys "data_keys and classes_keys are not the same!!!"
    # ensure all tensors have the same D and N, collect per-class sample counts
    first_key = data_keys[1]
    d = size(dict_data[first_key], 1)
    p = size(dict_data[first_key], 2)
    nsamples = [size(dict_data[k], 3) for k in data_keys]
    @assert all(size(dict_data[k],1) == d for k in data_keys) && all(size(dict_data[k],2) == p for k in data_keys) "Dimensions of samples or number of points is not the same for all classes"
    total = sum(nsamples)
    x_train_data = zeros(T, d, p, total)
    y_train_data = Vector{Int}(undef, total)
    pos = 1
    @inbounds for classname in data_keys
        n = size(dict_data[classname], 3)
        if n > 0
            x_train_data[:, :, pos:pos+n-1] = dict_data[classname]
            y_train_data[pos:pos+n-1] .= dict_classes[classname]
            pos += n
        end
    end
    return x_train_data, y_train_data
end



"""
    on_fly_collate_fn(batch)

Collate function for on-the-fly sampled point-cloud batches.

# Arguments
- `batch`: vector of tuples `(pc, label)` where `pc` has shape `(D, N, 1)`.

# Returns
- Tuple `(x, y)` where:
    - `x` has shape `(D, N, BS)`,
    - `y` is a label vector of length `BS`.
"""
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

"""
    bag_collate_fn(batch)

Collate function for on-the-fly sampled "bag of point clouds" batches (see
`load_mnist_clock`/`sample_one_bag`), where each observation is itself already a full
`(data, mask, labels)` triple for one bag rather than a single `(x, y)` pair.

# Arguments
- `batch`: vector of tuples `(data, mask, labels)` where `data` has shape `(D, N, L)`, `mask`
  has shape `(1, 1, L)`, and `labels` has shape `(L,)`.

# Returns
- Tuple `(x, mask, labels)` where:
    - `x` has shape `(D, N, L, BS)`,
    - `mask` has shape `(1, 1, L, BS)`,
    - `labels` has shape `(L, BS)`.
"""
function bag_collate_fn(batch::Vector{Tuple{X, M, L}}) where {X <: AbstractArray{<:AbstractFloat, 3}, M <: AbstractArray{Bool, 3}, L <: AbstractVector{Int}}
    data1, mask1, labels1 = batch[1]
    BS = length(batch)
    x = Array{eltype(data1)}(undef, size(data1)..., BS)
    mask = Array{Bool}(undef, size(mask1)..., BS)
    labels = Array{eltype(labels1)}(undef, length(labels1), BS)
    @inbounds for (i, (data, m, l)) in pairs(batch)
        copyto!(@view(x[:, :, :, i]), data)
        copyto!(@view(mask[:, :, :, i]), m)
        copyto!(@view(labels[:, i]), l)
    end
    return (x, mask, labels)
end

"""
    bag_collate_fn(batch)

Label-free variant of [`bag_collate_fn`](@ref) for `x_only=true`: each observation is a
`(data, mask)` pair (no labels) -- used so downstream consumers (older 2-arg models,
`CuIterator`) never see, and never move to GPU, labels that are only needed for plotting.

# Arguments
- `batch`: vector of tuples `(data, mask)` where `data` has shape `(D, N, L)` and `mask` has
  shape `(1, 1, L)`.

# Returns
- Tuple `(x, mask)` where `x` has shape `(D, N, L, BS)` and `mask` has shape `(1, 1, L, BS)`.
"""
function bag_collate_fn(batch::Vector{Tuple{X, M}}) where {X <: AbstractArray{<:AbstractFloat, 3}, M <: AbstractArray{Bool, 3}}
    data1, mask1 = batch[1]
    BS = length(batch)
    x = Array{eltype(data1)}(undef, size(data1)..., BS)
    mask = Array{Bool}(undef, size(mask1)..., BS)
    @inbounds for (i, (data, m)) in pairs(batch)
        copyto!(@view(x[:, :, :, i]), data)
        copyto!(@view(mask[:, :, :, i]), m)
    end
    return (x, mask)
end



