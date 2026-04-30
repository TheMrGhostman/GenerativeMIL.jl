"""
    normalize_point_cloud(pc::AbstractArray{T,3}) where T<:AbstractFloat

Normalize a batched point-cloud tensor feature-wise.

# Arguments
- `pc`: point-cloud tensor of shape `(D, P, N)` where:
    - `D` is point dimensionality (typically 3),
    - `P` is the number of points,
    - `N` is the number of samples.

# Returns
- A normalized tensor with the same shape `(D, P, N)`.

# Notes
- Mean and standard deviation are computed independently per dimension,
    across points and samples.
"""
function normalize_point_cloud(pc::AbstractArray{T, 3}) where T<:AbstractFloat
    mu = mean(mean(pc, dims=(3)), dims=2)
    sigma = mean(std(pc, dims=(3)), dims=2)
    return (pc .- mu) ./ (sigma .+ eps(T))
end

"""
    normalize_point_cloud(pcs::Vector{<:AbstractArray{T,2}}) where T<:AbstractFloat

Normalize a vector of variable-cardinality point clouds.

# Arguments
- `pcs`: vector of point clouds, each of shape `(D, P_i)`.

# Returns
- A vector of normalized point clouds with unchanged per-sample shapes.

# Notes
- Global statistics are estimated per dimension over the whole collection.
"""
function normalize_point_cloud(pcs::Vector{<:AbstractArray{T, 2}}) where T<:AbstractFloat
    d1 = getindex.(pcs, 1, :)
    d2 = getindex.(pcs, 2, :)
    d3 = getindex.(pcs, 3, :)
    mu = [mean(mean.(d1)), mean(mean.(d2)), mean(mean.(d3))]
    sigma = [mean(std.(d1)), mean(std.(d2)), mean(std.(d3))]
    return map(p -> (p .- mu) ./ (sigma .+ eps(T)), pcs)
end

"""
    sample_fixed_n(pc, npoints)

Sample up to `npoints` points from a single point cloud without replacement.

# Arguments
- `pc`: point cloud of shape `(D, P)`.
- `npoints`: requested number of points.

# Returns
- Array of shape `(D, min(npoints, P))`.
"""
sample_fixed_n(pc, npoints) = pc[:, sample(axes(pc, 2), min(npoints, size(pc, 2)), replace=false)]

"""
    sample_fixed_n_unsqueeze(pc, npoints)

Equivalent to [`sample_fixed_n`](@ref), with a singleton third dimension.

# Arguments
- `pc`: point cloud of shape `(D, P)`.
- `npoints`: requested number of points.

# Returns
- Array of shape `(D, min(npoints, P), 1)`.
"""
sample_fixed_n_unsqueeze(pc, npoints) = unsqueeze(pc[:, sample(axes(pc, 2), min(npoints, size(pc, 2)), replace=false)], dims=3)

"""
    sample_fixed_n_from_matrix(xs::AbstractArray, npoints::Int)

Sample the point dimension of a batched tensor without replacement.

# Arguments
- `xs`: tensor of shape `(D, P, N)`.
- `npoints`: requested number of points.

# Returns
- Tensor of shape `(D, min(npoints, P), N)`.
"""
function sample_fixed_n_from_matrix(xs::AbstractArray, npoints::Int)
    idx = sample(axes(xs, 2), min(npoints, size(xs, 2)), replace=false)
    return xs[:, idx, :]
end

"""
    _stack_point_clouds(pcs)

Stack a vector of identically shaped 2D point clouds into one 3D tensor.

# Arguments
- `pcs`: vector of point clouds, each of shape `(D, P)`.

# Returns
- Tensor of shape `(D, P, N)` where `N = length(pcs)`.

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
- `batch`: vector of tuples `(pc, label)` where `pc` has shape `(D, P, 1)`.

# Returns
- Tuple `(x, y)` where:
    - `x` has shape `(D, P, BS)`,
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
