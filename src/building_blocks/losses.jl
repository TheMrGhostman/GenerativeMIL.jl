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

function masked_chamfer_distance(x, y, x_mask, y_mask)
    return Flux.mean([
            chamfer_distance(
                unmask(x[:,:,i:i],x_mask[:,:,i:i]), 
                unmask(y[:,:,i:i],y_mask[:,:,i:i])
            ) for i=1:size(x,3)])
end

function masked_chamfer_distance_cpu(x, y, x_mask, y_mask)
    x, x_mask = x|>cpu, x_mask|>cpu
    y, y_mask = y|>cpu, y_mask|>cpu
    return Flux.mean([
            Flux3D.chamfer_distance(
                unmask(x[:,:,i:i],x_mask[:,:,i:i]), 
                unmask(y[:,:,i:i],y_mask[:,:,i:i])
            ) for i=1:size(x,3)])
end

kl_divergence(μ, Σ) = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σ.^2) - μ.^2  - Σ.^2, dims=1)) 