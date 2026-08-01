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

#results = deserialize("/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/mnist/poolmodel/seed=1/cd_poolmodel_c003_ID-400778/run_config.jls")
df = deserialize("/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/mnist/poolmodel/seed=1/cd_poolmodel_c003_ID-400778/models/model_ep=2000.jls")

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
model = df.model
e = model.encoder
g = model.generator

c_embeddings = []
z_μ = []
z_Σ = []
Y = []
for (x, y) in tqdm(dataloaders.valid)
    #x = x |> cu
    c = e(x)
    push!(c_embeddings, c)
    zμ, zΣ = g(c)
    push!(z_μ, zμ)
    push!(z_Σ, zΣ)
    push!(Y, y)
end

C = dropdims(cat(c_embeddings..., dims=3), dims=2)
Z_μ = dropdims(cat(z_μ..., dims=3), dims=2);
Z_Σ = dropdims(cat(z_Σ..., dims=3), dims=2);
Y = reduce(vcat, Y);

umap = UMAP.fit(C, 2; n_neighbors=51)
emb = umap.embedding

f, ax = scatter_by_class(emb, Y; title = "UMAP of values after PoolEncoder c003", legend_markersize=30)

save("c003_C_umap.png", f)


umap1 = UMAP.fit(Z_μ, 2; n_neighbors=51)
emb1 = umap1.embedding

f1, ax1 = scatter_by_class(emb1, Y, title = "UMAP of values after Generator c003", legend_markersize=30)

save("c003_Z_mu_umap.png", f1)

cs = [mean(C[:,Y.==i], dims=2) for i in 0:9];
cs_e = UMAP.transform(umap,hcat(cs...)).embedding

f2, ax2 = scatter_by_class(cs_e, 0:9; title = "UMAP of centroids after PoolEncoder c003", legend_markersize=30)
save("c003_C-centroids_umap.png", f2)

pwd = GenerativeMIL.Distances.pairwise(GenerativeMIL.Distances.SqEuclidean(), cs, symmetric=true)
f3, ax3 = heatmap_by_class(pwd, 0:9; title = "Pairwise distance of centroids c003", colorbar_label = "squared euclidean distance")
save("c003_C-centroids_pwd_heatmap.png", f3)



# sampling

z_i = Z_μ .+ Z_Σ .* MLUtils.randn_like(Z_μ);
z_ie = UMAP.transform(umap1,z_i).embedding
f4, ax4 = scatter_by_class(z_ie, Y; title = "UMAP of random samples after Generator c003", legend_markersize=30)

save("c003_Z_samples_umap_transform.png", f4)

umap2 = UMAP.fit(z_i, 2; n_neighbors=101)
emb2 = umap2.embedding
f5, ax5 = scatter_by_class(emb2, Y; title = "UMAP of random samples after Generator c003", legend_markersize=30)

save("c003_Z_samples_umap_fit.png", f5)


z_i = [Z_μ .+ Z_Σ .* MLUtils.randn_like(Z_μ) for i in 1:10];
z_i = reduce(hcat, z_i)
z_ie = UMAP.transform(umap1,z_i).embedding
f6, ax6 = scatter_by_class(z_ie, vcat([Y for i in 1:10]...); title = "UMAP of random samples after Generator c003", legend_markersize=30)

save("c003_Z_samples_umap_transform2.png", f6)
