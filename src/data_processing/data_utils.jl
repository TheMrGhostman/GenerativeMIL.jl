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
function _normalize_point_cloud_dataset(pcs::AbstractArray{T, 3}...) where T<:AbstractFloat
    isempty(pcs) && error("No point-cloud tensors provided for normalization")
    μ = mean(pcs[1], dims=(2,3))
    σ = std(vec(pcs[1]))
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
