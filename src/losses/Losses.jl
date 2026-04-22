# middle step in organizing this repository
# TODO do structure properly

""" old note for myself, to be updated
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



kl_divergence(μ, Σ) = - Flux.mean(0.5f0 * sum(1f0 .+ log.(Σ.^2) - μ.^2  - Σ.^2, dims=1)) 
export kl_divergence

include("chamfer_distance.jl")
export chamfer_distance
include("masked_chamfer_distance.jl")
export chamfer_distance, masked_chamfer_distance, masked_chamfer_distance_cpu