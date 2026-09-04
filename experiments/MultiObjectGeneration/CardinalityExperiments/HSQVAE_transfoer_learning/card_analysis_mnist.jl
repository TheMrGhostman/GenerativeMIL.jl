using Revise
using DrWatson
using Pkg; Pkg.activate("/home/zorekmat/MIL/GenerativeMIL/experiments/VisualizationAndUnderstanding")


using Random
using Serialization
using YAML
using GenerativeMIL
using Flux
using JLD2, JSON3
using MLUtils

using ProgressBars
using CairoMakie
using Statistics

exp_path = "/home/zorekmat/MIL/GenerativeMIL/experiments/MultiObjectGeneration/CardinalityExperiments/"

GenerativeMIL._mnist_balanced_path() = "/home/zorekmat/MIL/GenerativeMIL/data/datasets/mnist_pc/mnist_4x_point_clouds_3x900_matrix.jls"

data = deserialize(GenerativeMIL._mnist_balanced_path())

data["features"] |> size
X = data["features"];
Y = data["targets"];

x = X[:,:,1];
pwx = pairwise((x,y)->abs(x.-y), x[1,:], x[1,:]); 
pwy = pairwise((x,y)->abs(x.-y), x[2,:], x[2,:]);
δₓ = pwx[pwx.>0] |> minimum
δᵧ = pwy[pwy.>0] |> minimum

min_x = minimum(X[1,:,:])
min_y = minimum(X[2,:,:])
max_x = maximum(X[1,:,:])
max_y = maximum(X[2,:,:])


# coordinates live on a native_res x upsample grid; rasterizing back down to native_res
# before counting occupied cells is what makes the count shape-sensitive (see card_analysis_mnist.jl history).
# Pass min_x/max_x/min_y/max_y from the full dataset (not the batch) so relative volumes
# computed on different batches stay comparable to each other.
function relative_volume(X::AbstractArray{<:Real,3}; min_x, max_x, min_y, max_y, upsample::Int=4)
    nx = round(Int, (max_x - min_x) / upsample) + 1
    ny = round(Int, (max_y - min_y) / upsample) + 1
    max_volume = nx * ny

    ix = @. floor(Int, (X[1,:,:] - min_x) / upsample)
    iy = @. floor(Int, (X[2,:,:] - min_y) / upsample)
    lin = ix .* ny .+ iy

    volumes = Vector{Int}(undef, size(X, 3))
    Threads.@threads for id in 1:size(X, 3)
        volumes[id] = length(Set(@view lin[:, id]))
    end

    return volumes ./ max_volume
end

relative_volumes = relative_volume(X; min_x, max_x, min_y, max_y, upsample=4);

unique(relative_volumes)

cls0 = relative_volumes[Y .== 0];
mean(cls0), std(cls0)

cls1 = relative_volumes[Y .== 1];
mean(cls1), std(cls1)

cls_mean = [mean(relative_volumes[Y .== i]) for i in 0:9]
cls_std  = [ std(relative_volumes[Y .== i]) for i in 0:9]

# --- relative volume distribution per class, 10 rows sharing one x axis ---
# One color throughout: the row (digit label) already carries class identity,
# so color doesn't need to repeat it - this keeps all ten panels comparable by shape alone.
bin_edges = range(extrema(relative_volumes)...; length = 41)
bar_color = "#2a78d6"

fig = Figure(size = (600, 1400))
axs = Axis[]
for d in 0:9
    ax = Axis(fig[d + 1, 1]; ylabel = string(d), ylabelrotation = 0)
    hist!(ax, relative_volumes[Y .== d]; bins = bin_edges, color = bar_color, strokewidth = 0)
    hideydecorations!(ax, label = false)
    hidespines!(ax, :t, :r, :l)
    d < 9 && hidexdecorations!(ax, grid = false)
    push!(axs, ax)
end
linkxaxes!(axs...)
axs[end].xlabel = "relative volume"
Label(fig[0, 1], "Relative digit volume by class"; fontsize = 18, font = :bold)
rowgap!(fig.layout, 4)

save(joinpath(exp_path, "relative_volume_by_class.png"), fig)
fig


id1 = rand(1:length(relative_volumes), 1000);
id2 = rand(1:length(relative_volumes), 1000);
tmp = pairwise((x,y)->abs2.(100 .* (x .- y)), relative_volumes[id1], relative_volumes[id2])

mean(tmp), median(tmp), maximum(tmp), minimum(tmp)

function normalize_point_cloud(pc::AbstractArray{T, 3}) where T<:AbstractFloat
    mu = mean(pc, dims=(2,3))
    sigma = std(pc, dims=(2,3))
    return (pc .- mu) ./ (sigma .+ eps(T))
end

mu = mean(X, dims=(2,3))
sigma = std(X, dims=(2,3))

nX = (X .- mu) ./ (sigma .+ eps(Float32));

X ≈ (nX .* (sigma .+ eps(Float32))) .+ mu