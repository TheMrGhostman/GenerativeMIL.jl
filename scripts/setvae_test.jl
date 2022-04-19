using DrWatson
@quickactivate "GenerativeMIL"
using Flux
using Distributions

using GenerativeMIL
using GenerativeMIL.Models

function size(t::Tuple{Array{Float32, 3}, Array{Float32, 3}})
    (size(t[1]), size(t[2]))
end


# simple batched version
# input x ...  bs = 8, set size = 32 , input dim = 3
bs = 8
ss = 32
in_d = 3
x = randn(Float32, in_d, ss, bs) # (3,32,8)
h0 = randn(Float32, in_d, ss, bs) # later will be MixtureOfGaussians

# ENCODER

# expand input to bigger dimension
# input dim = 3, output dim = 16 
o_d = 16
expand = Dense(in_d, o_d)
expand_h0 = Dense(in_d, o_d)

# first ISAB
# size of induced set = 10
# hidden dim = 16 
sis1 = 10
h_d = o_d
heads = 4
isab1 = Models.InducedSetAttentionBlock(sis1, h_d, heads)
# returns x_out, h_out

# second ISAB
# almost the same params
# size of induced set = 5
sis2 = 5
isab2 = Models.InducedSetAttentionBlock(sis2, h_d, heads)

# third ISAB
# size of induced set = 1
sis3 = 1
isab3 = Models.InducedSetAttentionBlock(sis3, h_d, heads)


# DECODER 


# first ABL  (reverse ordering)
# size of induce set = 1 = sis3
# heads = 4 
# zdim3 = 4 # dimansion of latent space
# depth = 2 # depth of VariationalBottleneck
# hid3 = 9 # number of neurons in VariationalBottleneck
zdim3 = 4
depth3 = 2
hid3 = 9
n_mixtures = 5

mog = Models.MixtureOfGaussians(in_d, n_mixtures)

abl3 = Models.AttentiveBottleneckLayer(sis3, h_d, heads, zdim3, hid3, depth3, Flux.sigmoid)

# second ABL
abl2 = Models.AttentiveBottleneckLayer(sis2, h_d, heads, zdim3, hid3, depth3, Flux.sigmoid)

# third ABL
abl1 = Models.AttentiveBottleneckLayer(sis1, h_d, heads, zdim3, hid3, depth3, Flux.sigmoid)

# reverse expand
r_expand = Dense(o_d, in_d)


# Inference
# Forward Pass
## Encoder
x0 = expand(x);
x1, h1 = isab1(x0);
x2, h2 = isab2(x1);
x3, h3 = isab3(x2);

## Prior
p_ss = size(x, 2)
p_bs = size(x, 3)

h0 = mog(p_ss, p_bs);

## Decoder
xx2, kld3, hh2, z3 = abl3(expand_h0(h0), h3);
xx1, kld2, hh1, z2 = abl2(xx2, h2);
xx0, kld1, hh0, z1 = abl1(xx1, h1);
xx = r_expand(xx0);


# Generation
g2, gkld3, gh2, gz3 = abl3(expand_h0(h0));
g1, gkld2, gh1, gz2 = abl3(g2);
g0, gkld1, gh0, gz1 = abl3(g1);
g = r_expand(g0);

