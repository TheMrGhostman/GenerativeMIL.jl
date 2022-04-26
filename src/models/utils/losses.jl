function SumMinDist(A, B)
    # minimum_{j}(|| a_i - b_j||^2)
    # Chamfer Distance between two 2D!! sets / bags / point clouds
    # A ∈ R^{N,D} , B ∈ R^{M,D}
    dist = Distances.pairwise(SqEuclidean(), A, B, dims=2) # -> distance matrix R^{N,M}
    dist = minimum(dist, dims=1) # over j
    return Flux.sum(dist)
end

function ChamferDistanceLoss(x_true, x_pred)
    # x ∈ R ^ {D, Ni, bs}
    if ndims(x_true) == ndims(x_pred) == 3
        total_loss = 0
        bs = size(x_true, 3)
        for i in 1:bs
            dist_to_x = SumMinDist(x_true[:,:,i], x_pred[:,:,i]) 
            dist_to_y = SumMinDist(x_pred[:,:,i], x_true[:,:,i])
            chamfer_instance = dist_to_x + dist_to_y
            total_loss += chamfer_instance
        end
    elseif ndims(x_true) == ndims(x_pred) == 2
        bs = 1
        dist_to_x = SumMinDist(x_true, x_pred) 
        dist_to_y = SumMinDist(x_pred, x_true)
        chamfer_instance = dist_to_x + dist_to_y
        total_loss = chamfer_instance 
    else
        error("Unsupported input sizes!! | x_pred $(size(x_pred)) | x_true $(size(x_true))")
    end
    return total_loss / bs
end


"""
# setvae loss
# kl ~ (d, m, bs)
# kls ~ (d, m, bs, layer)
# kld_loss = mean{ sum[ sum( sum{ kls, per d }, per m ), per layer ], per bs } ~ scalar
# β = β₀ * min(1, epoch / kl_warmup_epochs) # no KLD until a few epochs done default kl_warmup_epochs=0
# loss = beta * kl_loss + l2_loss # -> l2_loss is ChamferDistanceLoss

# topdown kld -> β ⋅ KLD
# topdown_kl = [kl.detach().mean(dim=0) / float(scale * args.z_dim) for scale, kl in zip(args.z_scales, kls)]
# m ~ z_scales
"""

