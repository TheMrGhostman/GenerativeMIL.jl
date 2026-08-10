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
function encode_ns(model, x::AbstractArray{T, 3}) where T <: AbstractFloat
    hᵢ = model.shared_encoder(x)

    feature_emb = model.statistic_net[2][1:end-1](model.statistic_net[1](hᵢ))
    μ_c, Σ_c = model.statistic_net(hᵢ)
    c = μ_c .+ Σ_c .* MLUtils.randn_like(μ_c) # (context_dim, 1, batch_size) - sampled context vector
    cᵢ = repeat(c, 1, size(x, 2), 1); # (context_dim, n_points, batch_size) - repeat context for each point
    
    μ_zᵢ, Σ_zᵢ = model.inference_net(hᵢ, cᵢ) 
    return (;hi = hᵢ, μc = μ_c, Σc = Σ_c, c=c, μz = μ_zᵢ, Σz = Σ_zᵢ, c_feat = feature_emb)
end

function encode_ns(model, dataloader::DataLoader)
    outs = []
    Y = []
    for (x,y) in tqdm(dataloader)
        o = encode_ns(model, x)
        push!(Y, y)
        push!(outs, o)
    end

    dic = Dict{Symbol, AbstractArray{Float32, 3}}()
    for key in collect(keys(outs[1]))
        v = cat(getfield.(outs, key)..., dims=3)
        dic[key] = v
    end
    nt = NamedTuple{Tuple(Symbol.(keys(dic)))}(values(dic))
    return nt, reduce(vcat, Y)
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
f02 = deserialize("/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/mnist/neuralstatistician/seed=1/cd_neuralstatistician_c002_ID-073619/models/model_ep=500.jls");
f06 = deserialize("/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/mnist/neuralstatistician/seed=1/cd_neuralstatistician_c006_ID-847396/models/model_ep=500.jls");
f08 = deserialize("/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/mnist/neuralstatistician/seed=1/cd_neuralstatistician_c008_ID-722138/models/model_ep=500.jls");

m02 = f02.model
m06 = f06.model
m08 = f08.model

## prepare predictions
encode_ns(m02, first(dataloaders.valid)[1])

o02, Y = encode_ns(m02, dataloaders.valid)
o06, _ = encode_ns(m06, dataloaders.valid)
o08, _ = encode_ns(m08, dataloaders.valid)

## C002
## visualization of C features
u02 = UMAP.fit(dropdims(o02.c_feat, dims=2), 2; n_neighbors=51)
e02 = u02.embedding

f0, _ = scatter_by_class(e02, Y; title = "UMAP of values C_feat - NS_c002", legend_markersize=30)
save("c002-cd_C_feat_umap.png", f0)

## visualization of mean-max of hᵢ
u02 = UMAP.fit(dropdims(mean(o02.hi, dims=2), dims=2), 2; n_neighbors=51)
e02 = u02.embedding
f0, _ = scatter_by_class(e02, Y; title = "UMAP of values mean(hi) - NS_c002", legend_markersize=30)
save("c002-cd_mean_hi_umap.png", f0)

## visualization of 
u02 = UMAP.fit(dropdims(o02.μc, dims=2), 2; n_neighbors=51)
e02 = u02.embedding
f0, _ = scatter_by_class(e02, Y; title = "UMAP of values μc - NS_c002", legend_markersize=30)
save("c002-cd_mu-c_umap.png", f0)

## heatmap
cs = [mean(o02.c_feat[:,1,Y.==i], dims=2) for i in 0:9];
pwd = GenerativeMIL.Distances.pairwise(GenerativeMIL.Distances.SqEuclidean(), cs, symmetric=true)
f1, _ = heatmap_by_class(pwd, 0:9; title = "Pairwise distance of centroids c003", colorbar_label = "squared euclidean distance")
save("c002-cd_C-feat-centroids_pwd_heatmap.png", f1)



## C006
## visualization of C features
u06 = UMAP.fit(dropdims(o06.c_feat, dims=2), 2; n_neighbors=51)
e06 = u06.embedding

f6, _ = scatter_by_class(e06, Y; title = "UMAP of values C_feat - NS_c006", legend_markersize=30)
save("c006-cd_C-feat_umap.png", f6)

## visualization of μc
u06 = UMAP.fit(dropdims(o06.μc, dims=2), 2; n_neighbors=51)
e06 = u06.embedding

f6, _ = scatter_by_class(e06, Y; title = "UMAP of values μc - NS_c006", legend_markersize=30)
save("c006-cd_mu-c_umap.png", f6)


## C008
## visualization of C features
u08 = UMAP.fit(dropdims(o08.c_feat, dims=2), 2; n_neighbors=51)
e08 = u08.embedding

f6, _ = scatter_by_class(e08, Y; title = "UMAP of values C_feat - NS_c008", legend_markersize=30)
save("c008-cd_C-feat_umap.png", f6)

## visualization of μc
u08 = UMAP.fit(dropdims(o08.μc, dims=2), 2; n_neighbors=51)
e08 = u08.embedding

f6, _ = scatter_by_class(e08, Y; title = "UMAP of values μc - NS_c008", legend_markersize=30)
save("c008-cd_mu-c_umap.png", f6)




## L2 norm as loss funciton 
## load model
f02 = deserialize("/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/mnist/neuralstatistician/seed=1/l2_neuralstatistician_c002_ID-412522/models/model_ep=1000.jls");
f06 = deserialize("/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/mnist/neuralstatistician/seed=1/l2_neuralstatistician_c006_ID-939514/models/model_ep=1000.jls");
f08 = deserialize("/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/mnist/neuralstatistician/seed=1/l2_neuralstatistician_c008_ID-349190/models/model_ep=1000.jls");

m02 = f02.model
m06 = f06.model
m08 = f08.model

## prepare predictions
encode_ns(m02, first(dataloaders.valid)[1])

o02, Y = encode_ns(m02, dataloaders.valid)
o06, _ = encode_ns(m06, dataloaders.valid)
o08, _ = encode_ns(m08, dataloaders.valid)

## C002
## visualization of C features
u02 = UMAP.fit(dropdims(o02.c_feat, dims=2), 2; n_neighbors=51)
e02 = u02.embedding

f0, _ = scatter_by_class(e02, Y; title = "UMAP of values C_feat - NS_c002", legend_markersize=30)
save("c002-l2_C_feat_umap.png", f0)

## visualization of mean-max of hᵢ
u02 = UMAP.fit(dropdims(mean(o02.hi, dims=2), dims=2), 2; n_neighbors=51)
e02 = u02.embedding
f0, _ = scatter_by_class(e02, Y; title = "UMAP of values mean(hi) - NS_c002", legend_markersize=30)
save("c002-l2_mean_hi_umap.png", f0)

## visualization of 
u02 = UMAP.fit(dropdims(o02.μc, dims=2), 2; n_neighbors=51)
e02 = u02.embedding
f0, _ = scatter_by_class(e02, Y; title = "UMAP of values μc - NS_c002", legend_markersize=30)
save("c002-l2_mu-c_umap.png", f0)

## heatmap
cs = [mean(o02.c_feat[:,1,Y.==i], dims=2) for i in 0:9];
pwd = GenerativeMIL.Distances.pairwise(GenerativeMIL.Distances.SqEuclidean(), cs, symmetric=true)
f1, _ = heatmap_by_class(pwd, 0:9; title = "Pairwise distance of centroids c003", colorbar_label = "squared euclidean distance")
save("c002-l2_C-feat-centroids_pwd_heatmap.png", f1)


## C006
## visualization of C features
u06 = UMAP.fit(dropdims(o06.c_feat, dims=2), 2; n_neighbors=51)
e06 = u06.embedding

f6, _ = scatter_by_class(e06, Y; title = "UMAP of values C_feat - NS_c006", legend_markersize=30)
save("c006-l2_C-feat_umap.png", f6)

## visualization of μc
u06 = UMAP.fit(dropdims(o06.μc, dims=2), 2; n_neighbors=51)
e06 = u06.embedding

f6, _ = scatter_by_class(e06, Y; title = "UMAP of values μc - NS_c006", legend_markersize=30)
save("c006-l2_mu-c_umap.png", f6)


## C008
## visualization of C features
u08 = UMAP.fit(dropdims(o08.c_feat, dims=2), 2; n_neighbors=51)
e08 = u08.embedding

f6, _ = scatter_by_class(e08, Y; title = "UMAP of values C_feat - NS_c008", legend_markersize=30)
save("c008-l2_C-feat_umap.png", f6)

## visualization of μc
u08 = UMAP.fit(dropdims(o08.μc, dims=2), 2; n_neighbors=51)
e08 = u08.embedding

f6, _ = scatter_by_class(e08, Y; title = "UMAP of values μc - NS_c008", legend_markersize=30)
save("c008-l2_mu-c_umap.png", f6)

