using DrWatson
@quickactivate "GenerativeMIL"
using Flux
using Distributions

using GenerativeMIL
using GenerativeMIL.Models

function info(model::Flux.Chain; text=("",""))
	for i=1:length(model)
		@info("$(text[1])- $(model[i])")
	end
end

function info(mab::MultiheadAttentionBlock; text=("",""))
	@info("$(text[1])MultiheadAttentionBlock $(text[2])")
    @info("$(text[1])- Attention -> $(mab.Multihead)")
    @info("$(text[1])- Feed Forward -> $(mab.FF)")
    @info("$(text[1])- Layer Norm (inner) -> $(mab.LN1)")
    @info("$(text[1])- Layer Norm (outer) -> $(mab.LN2)")
end

function info(isab::InducedSetAttentionBlock; text=("",""))
	@info("$(text[1])InducedSetAttentionBlock $(text[2])")
    info(isab.MAB1, text=("$(text[1])-"," (inner)"))
    info(isab.MAB2, text=("$(text[1])-"," (outer)"))
    @info("$(text[1])- Induced Set -> $(typeof(isab.I)), $(size(isab.I))")
end

function info(vb::VariationalBottleneck; text=("",""))
    @info("$(text[1])VariationalBottleneck $(text[2])")
    @info("$(text[1])- Prior")
    info(vb.prior, text=("$(text[1])-",""))
    @info("$(text[1])- Posterior")
    info(vb.posterior, text=("$(text[1])-",""))
    @info("$(text[1])- Decoder")
    info(vb.decoder, text=("$(text[1])-",""))
end

function info(abl::AttentiveBottleneckLayer; text=("",""))
	@info("$(text[1])AttentiveBottleneckLayer $(text[2])")
    info(abl.MAB1, text=("$(text[1])-"," (inner)"))
    info(abl.MAB2, text=("$(text[1])-"," (outer)"))
    info(abl.VB, text=("$(text[1])-",""))
    @info("$(text[1])- Induced Set -> $(typeof(abl.I)), $(size(abl.I))")
end

# test MAB
function mab_unit()
    v = randn(Float32,16,10,6) # dim=16, seq_len/set_size=10, bs=6
    q = randn(Float32,16,2,6) # induced set with -> latent "set" with 2 elements
    mab = MultiheadAttentionBlock(16, 4)
    o = mab(q,v)
    @info "MAB check" 
    @info "------------------------------------------"
    info(mab)
    @info "------------------------------------------"
    @info "dim(V)=$(size(v)) | dim(Q)=$(size(q)) | dim(MAB(Q,V))=$(size(o))"
    @info "=========================================="
end

function isab_unit()
    x = randn(Float32,16,10,6) # dim=16, seq_len/set_size=10, bs=6
    isab = InducedSetAttentionBlock(2, 16, 4)
    b_i = batched_I(isab, size(x,3))
    o, h = isab(x)
    @info "ISAB check" 
    @info "------------------------------------------"
    info(isab)
    @info "------------------------------------------"
    @info "dim(X)=$(size(x)) | dim(I)=$(size(b_i)) | dim(ISAB(X))=($(size(o)),$(size(h)))"
    @info "=========================================="
end

function vb_unit(depth::Int=3)
    x = randn(Float32,3,10,6) # dim=16, seq_len/set_size=10, bs=6
    y = randn(Float32,3,10,6)
    vb = VariationalBottleneck(3, 2, 3, 8, depth, swish)
    z, xx, kld = vb(x)
    z, xy, kld = vb(x, y)
    @info "VariationalBottleneck check" 
    @info "------------------------------------------"
    info(vb)
    @info "------------------------------------------"
    @info "dim(h)=$(size(x)) | dim(h_enc)=$(size(y)) | dim(VB(h))=$(size(xx)) |"
    @info "dim(VB(h, h_enc)) = (z, ̂h, kld) = ($(size(z)), $(size(xy)), $(size(kld)))"
    @info "=========================================="
end

function abl_unit(depth::Int=3)
    m, h_dim, heads, zdim, hidden = (2, 16, 4, 3, 32)
    x = randn(Float32, h_dim, 10, 6) # dim=16, seq_len/set_size=10, bs=6
    h_enc = randn(Float32, h_dim, m, 6)
    abl = AttentiveBottleneckLayer(m, h_dim, heads, zdim, hidden, depth, swish)
    b_i = batched_I(abl, size(x,3))
    o1, kld1, h1, z1 = abl(x)
    o2, kld2, h2, z2 = abl(x, h_enc)
    @info "AttentiveBottleneckLayer check" 
    @info "------------------------------------------"
    info(abl)
    @info "------------------------------------------"
    @info "dim(x)=$(size(x)) | dim(I)=$(size(b_i)) | dim(h_enc)=($(size(h_enc)))"
    @info "------------------------------------------"
    @info "Generation"
    @info "dim(ABL(x)) = (o, kld, ̂h, z) = ($(size(o1)), $(kld1), $(size(h1)), $(size(z1)))"
    @info "------------------------------------------"
    @info "Inference"
    @info "dim(ABL(x, h_enc)) = (o, kld, ̂h, z) = ($(size(o2)), $(size(kld2)), $(size(h2)), $(size(z2)))"
    @info "=========================================="
end

#mab_unit()
#isab_unit()
#vb_unit(3)
#vb_unit(1)
#abl_unit(2)