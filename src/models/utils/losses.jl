function SumMinDist(A, B)
    # minimum_{j}(|| a_i - b_j||^2)
    # Chamfer Distance between two 2D!! sets / bags / point clouds
    # A ∈ R^{N,D} , B ∈ R^{M,D}
    dist = Distances.pairwise(SqEuclidean(), A, B, dims=1) # -> distance matrix R^{N,M}
    dist = minimum(dist, dims=2) # over j
    return Flux.sum(dist)
end

function ChamferDistanceLoss(x_true, x_pred)
    # x ∈ R ^ {BS, Ni, D}
    if ndims == 3
        nothing
    end
    
    dist_to_x = SumMinDist(x_true, x_pred) # TODO fix ! now just for single instance
    dist_to_y = SumMinDist(x_pred, x_true) # TODO fix ! now just for single instance
    return dist_to_x + dist_to_y
end