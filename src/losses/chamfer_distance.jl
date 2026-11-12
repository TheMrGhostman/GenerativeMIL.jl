# I borrowed this implementation from Flux3D.jl, which is licensed under MIT License.
# I could use this package but since I need only this function, I just copy-paste the code here to avoid adding more dependencies.

"""
    chamfer_distance(A::PointCloud, B::PointCloud; w1=1.0, w2=1.0)

Computes the chamfer distance between PointCloud `A` and `B`. 

`w1` and `w2` are optional arguments for specifying the weighted mean of between two 
pcloud for computing metrics.
"""
chamfer_distance(
    A::AbstractArray{<:Number},
    B::AbstractArray{<:Number};
    w1::Number = 1.0,
    w2::Number = 1.0,
) = _chamfer_distance(Float32.(A), Float32.(B), Float32(w1), Float32(w2))

chamfer_distance(
    A::AbstractArray{Float32},
    B::AbstractArray{Float32};
    w1::Number = 1.0,
    w2::Number = 1.0,
) = _chamfer_distance(A, B, Float32(w1), Float32(w2))

function _chamfer_distance(
    A::AbstractArray{Float32,2},
    B::AbstractArray{Float32,2},
    w1::Float32 = 1.0f0,
    w2::Float32 = 1.0f0,
)
    A = reshape(A, size(A)..., 1)
    B = reshape(B, size(B)..., 1)
    return _chamfer_distance(A, B, w1, w2)
end

function _chamfer_distance(
    A::AbstractArray{Float32,3},
    B::AbstractArray{Float32,3},
    w1::Float32 = 1.0f0,
    w2::Float32 = 1.0f0,
)
    nn_for_A, nn_for_B = Zygote.@ignore _nearest_neighbors(A, B)

    dist_A_to_B = mean((A .- B[:, nn_for_A]) .^ 2) * 3.0f0
    dist_B_to_A = mean((B .- A[:, nn_for_B]) .^ 2) * 3.0f0

    distance = (w1 * dist_A_to_B) + (w2 * dist_B_to_A)
    return distance
end

function _chamfer_distance(
    A::AbstractArray{Float32,3},
    B::AbstractArray{Float32,3},
    A_mask::AbstractArray{Bool,3},
    B_mask::AbstractArray{Bool,3},
    w1::Float32 = 1.0f0,
    w2::Float32 = 1.0f0,
)
    nn_for_A, nn_for_B = Zygote.@ignore _nearest_neighbors(A, B, A_mask, B_mask)

    dist_A_to_B = mean((A .- B[:, nn_for_A]) .^ 2) * 3.0f0
    dist_B_to_A = mean((B .- A[:, nn_for_B]) .^ 2) * 3.0f0

    distance = (w1 * dist_A_to_B) + (w2 * dist_B_to_A)
    return distance
end

# helper functions

function _nearest_neighbors(x::Array{Float32,3}, y::Array{Float32,3})
    nn_for_x = cat(
        [
            CartesianIndex.(reduce(vcat, NearestNeighbors.knn(KDTree(y[:, :, i]), x[:, :, i], 1)[1]), i)
            for i = 1:size(x, 3)
        ]...,
        dims = 2,
    )
    nn_for_y = cat(
        [
            CartesianIndex.(reduce(vcat, NearestNeighbors.knn(KDTree(x[:, :, i]), y[:, :, i], 1)[1]), i)
            for i = 1:size(x, 3)
        ]...,
        dims = 2,
    )
    return nn_for_x, nn_for_y
end

function _nearest_neighbors(x::CuArray{Float32,3}, y::CuArray{Float32,3})
    P = _pairwise_sqdist_batched(x, y) # (n, m, bs)

    nn_for_x = argmin(P; dims = 2) |> Array
    nn_for_y = argmin(P; dims = 1) |> Array

    nn_for_x = reshape(map(x -> CartesianIndex(x[2], x[3]), nn_for_x), :, size(x, 3))
    nn_for_y = reshape(map(y -> CartesianIndex(y[1], y[3]), nn_for_y), :, size(y, 3))
    return nn_for_x, nn_for_y
end

function _nearest_neighbors(x::CuArray{Float32,3}, y::CuArray{Float32,3}, x_mask::CuArray{Bool,3}, y_mask::CuArray{Bool,3})
    P = _pairwise_sqdist_batched(x, y, x_mask, y_mask) # (n, m, bs)

    nn_for_x = argmin(P; dims = 2) |> Array
    nn_for_y = argmin(P; dims = 1) |> Array

    nn_for_x = reshape(map(x -> CartesianIndex(x[2], x[3]), nn_for_x), :, size(x, 3))
    nn_for_y = reshape(map(y -> CartesianIndex(y[1], y[3]), nn_for_y), :, size(y, 3))
    return nn_for_x, nn_for_y
end