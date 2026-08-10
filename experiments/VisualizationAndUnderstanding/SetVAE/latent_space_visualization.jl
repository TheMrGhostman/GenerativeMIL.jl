using Revise
using DrWatson
@quickactivate

using ArgParse
using Random
using Serialization
using YAML
using GenerativeMIL
using Flux
using CUDA
using MLUtils
using ProgressBars

using UMAP, CairoMakie
using Statistics

include("../src/vizualization_functions.jl")

##
function encode_svae(model, x::AbstractArray{T, 3}) where T<:AbstractFloat
    _, c = model.encoder(x)
    return copy(c)
end


function encode_shallow_svae(model, dataloader::DataLoader)
    embeddings = []
    labels = []
    for (x, y) in tqdm(dataloader)
        c = only(encode_svae(model, x)) # shallow svae has only single latent variable -> only
        push!(embeddings, c)
        push!(labels, y)
    end
    embeddings = dropdims(cat(embeddings..., dims=3), dims=2)
    labels = reduce(vcat, labels)
    return embeddings, labels
end

# encode -> sample and decode -> hat{encoder}
function encode_and_sample_shallow_svae(model, dataloader::DataLoader)
    μs = []
    Σs = []
    labels = []
    for (x, y) in tqdm(dataloader)
        c = only(encode_svae(model, x)) # shallow svae has only single latent variable -> only
        μ, Σ = model.decoder.layers[1].VB.posterior(c)
        
        push!(μs, μ)
        push!(Σs, Σ)
        push!(labels, y)
    end
    μs = dropdims(cat(μs..., dims=3), dims=2)
    Σs = dropdims(cat(Σs..., dims=3), dims=2)
    labels = reduce(vcat, labels)
    return μs, Σs, labels
end

## load dataset
data_mnist_cfg = Dict{Symbol, Any}(
    :sample_on_fly => false, 
    :cardinality_count => "balanced", 
    :dataset => "mnist", 
    :normalize => true, 
    :npoints => 512, 
    :ratio => 0.2
)

GenerativeMIL._mnist_balanced_path() = "/home/zorekmat/MIL/GenerativeMIL/data/datasets/mnist_pc/mnist_4x_point_clouds_3x900_matrix.jls"

dataloaders = create_dataloaders(batch_size=32, x_only=false, data_mnist_cfg)

## load model
# e means new config, o means old config
f05e = deserialize("/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/mnist/setvae/seed=1/cd_setvae_e_c005_ID-304491/models/model_ep=1100.jls");
f10o = deserialize("/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/mnist/setvae/seed=1/cd_setvae_c010_ID-686410/models/model_ep=1000.jls");
#cd_setvae_c009_ID-653802

## prepare predictions
_, t = f05e.model.encoder(first(dataloaders.valid)[1]) # first test
t = encode_svae(f05e.model, first(dataloaders.valid)[1])

c05e, Y = encode_shallow_svae(f05e.model, dataloaders.valid)   
c10o, _ = encode_shallow_svae(f10o.model, dataloaders.valid)   

μ05e, Σ05e, _ = encode_and_sample_shallow_svae(f05e.model, dataloaders.valid)  
μ10o, Σ10o, _ = encode_and_sample_shallow_svae(f10o.model, dataloaders.valid)

## C005_e
## visualization of C features
u05e = UMAP.fit(c05e, 2; n_neighbors=51)
e05e = u05e.embedding

f0, _ = scatter_by_class(e05e, Y; title = "UMAP of values C_feat - SetVAE_c005_e", legend_markersize=30)
save("c005_e-cd-C_feat_umap.png", f0)


## visualization of q(z|c) mean μc
u05e1 = UMAP.fit(μ05e, 2; n_neighbors=51)
e05e1 = u05e1.embedding

f1, _ = scatter_by_class(e05e1, Y; title = "UMAP of values μc - SetVAE_c005_e", legend_markersize=30)
save("c005_e-cd_mu-c_umap.png", f1)

u05e2 = UMAP.fit(μ05e, 3; n_neighbors=51)
e05e2 = u05e2.embedding

f2,_ = scatter3d_by_class(e05e2, Y, title = "UMAP of values μc - SetVAE_c005_e", legend_markersize=30)
save("c005_e-cd_mu-c_umap_3d.png", f2)

## visualization of q(z|c) samples; just transform the samples to 2D
c_samples = μ05e .+ Σ05e .* MLUtils.randn_like(μ05e);
e05e1_samples = UMAP.transform(u05e1,c_samples).embedding

f3, _ = scatter_by_class(e05e1_samples, Y; title = "UMAP transformation of samples from c - SetVAE_c005_e", legend_markersize=30)
save("c005_e-cd_c_samples_umap.png", f3)

## heatmap
cs1 = [mean(c05e[:,Y.==i], dims=2) for i in 0:9];
pwd = GenerativeMIL.Distances.pairwise(GenerativeMIL.Distances.SqEuclidean(), cs1, symmetric=true)

f4, _ = heatmap_by_class(pwd, 0:9; title = "Pairwise distance of centroids c005_e", colorbar_label = "squared euclidean distance")
save("c005_e-cd_C-feat-centroids_pwd_heatmap.png", f4)


cs2 = [mean(μ05e[:,Y.==i], dims=2) for i in 0:9];
pwd2 = GenerativeMIL.Distances.pairwise(GenerativeMIL.Distances.SqEuclidean(), cs2, symmetric=true)

f5, _ = heatmap_by_class(pwd2, 0:9; title = "Pairwise distance of centroids c005_e", colorbar_label = "squared euclidean distance")
save("c005_e-cd_C-mu-centroids_pwd_heatmap.png", f5)



## C010o
## visualization of C features
u10o = UMAP.fit(c10o, 2; n_neighbors=51)
e10o = u10o.embedding

f6, _ = scatter_by_class(e10o, Y; title = "UMAP of values C_feat - SetVAE_c010", legend_markersize=30)
save("c010-cd-C_feat_umap.png", f6)


## visualization of q(z|c) mean μc
u10o1 = UMAP.fit(μ10o, 2; n_neighbors=51)
e10o1 = u10o1.embedding

f7, _ = scatter_by_class(e10o1, Y; title = "UMAP of values μc - SetVAE_c010", legend_markersize=30)
save("c010-cd_mu-c_umap.png", f7)

u10o2 = UMAP.fit(μ10o, 3; n_neighbors=51)
e10o2 = u10o2.embedding

f8,_ = scatter3d_by_class(e10o2, Y, title = "UMAP of values μc - SetVAE_c010", legend_markersize=30)
save("c010-cd_mu-c_umap_3d.png", f8)

## visualization of q(z|c) samples; just transform the samples to 2D
c_samples = μ10o .+ Σ10o .* MLUtils.randn_like(μ10o);
e10o1_samples = UMAP.transform(u10o1,c_samples).embedding

f9, _ = scatter_by_class(e10o1_samples, Y; title = "UMAP transformation of samples from c - SetVAE_c010", legend_markersize=30)
save("c010-cd_c_samples_umap.png", f9)

## heatmap
cs1 = [mean(c10o[:,Y.==i], dims=2) for i in 0:9];
pwd1 = GenerativeMIL.Distances.pairwise(GenerativeMIL.Distances.SqEuclidean(), cs1, symmetric=true)

f10, _ = heatmap_by_class(pwd1, 0:9; title = "Pairwise distance of centroids c010", colorbar_label = "squared euclidean distance")
save("c01O-cd_C-feat-centroids_pwd_heatmap.png", f10)


cs2 = [mean(μ10o[:,Y.==i], dims=2) for i in 0:9];
pwd2 = GenerativeMIL.Distances.pairwise(GenerativeMIL.Distances.SqEuclidean(), cs2, symmetric=true)

f11, _ = heatmap_by_class(pwd2, 0:9; title = "Pairwise distance of centroids c010", colorbar_label = "squared euclidean distance")
save("c010-cd_C-mu-centroids_pwd_heatmap.png", f11)
