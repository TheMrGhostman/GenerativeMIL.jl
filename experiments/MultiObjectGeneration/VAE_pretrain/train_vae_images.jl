using Revise
using DrWatson
@quickactivate

using ArgParse
using Random
using Serialization
using YAML
using OrderedCollections
using GenerativeMIL
using Flux
using CUDA
using Statistics
using JLD2, JSON3
using MLUtils
using ProgressBars

using CairoMakie
using Dates

dict2nt(x) = (; (Symbol(k) => v for (k, v) in x)...)

function plot_batch_heatmaps(pred_batch, gt_batch; n::Int=8, image_size::Tuple{Int, Int}=(28, 28), colorrange=(0, 1))
    #pred_batch = Array(pred_batch)
    #gt_batch = Array(gt_batch)
    n = min(n, size(pred_batch, ndims(pred_batch)), size(gt_batch, ndims(gt_batch)))

    fig = Figure(size = (800, 220 * n))
    for i in 1:n
        gt_sample = ndims(gt_batch) == 2 ? reshape(@view(gt_batch[:, i]), image_size...) : dropdims(@view(gt_batch[:, :, :, i]); dims=3)
        pred_sample = ndims(pred_batch) == 2 ? reshape(@view(pred_batch[:, i]), image_size...) : dropdims(@view(pred_batch[:, :, :, i]); dims=3)

        gt_img = reverse(gt_sample, dims=2)
        pred_img = reverse(pred_sample, dims=2)

        ax_gt = Axis(fig[i, 1], title = i == 1 ? "Ground truth" : "")
        ax_pred = Axis(fig[i, 2], title = i == 1 ? "Prediction" : "")
        heatmap!(ax_gt, gt_img)#; colormap = :grays, colorrange = colorrange)
        heatmap!(ax_pred, pred_img)#; colormap = :grays, colorrange = colorrange)
        hidedecorations!(ax_gt)
        hidedecorations!(ax_pred)
    end

    return fig
end

function plot_class_scatter(points, labels; title::String="Class scatter", markersize::Real=12, alpha::Real=0.9, class_names=nothing, palette=Makie.wong_colors())
    points = Array(points)
    labels = collect(labels)

    if size(points, 1) == 2
        xs = points[1, :]
        ys = points[2, :]
    elseif size(points, 2) == 2
        xs = points[:, 1]
        ys = points[:, 2]
    else
        error("points must have shape 2×N or N×2")
    end

    @assert length(labels) == length(xs) "labels must have the same length as the number of points"

    classes = unique(labels)
    fig = Figure(size = (800, 800))
    ax = Axis(fig[1, 1], title = title, aspect = DataAspect())

    for (i, class_label) in enumerate(classes)
        idx = findall(==(class_label), labels)
        display_label = class_names === nothing ? string(class_label) : class_names[class_label]
        scatter!(ax, xs[idx], ys[idx]; color = palette[mod1(i, length(palette))], markersize = markersize, alpha = alpha, label = display_label)
    end

    axislegend(ax; position = :rb)
    return fig
end

"""
using MLDatasets
train = MNIST(split=:train)
test = MNIST(split=:test)
serialize(
    joinpath(datadir("datasets"), "mnist_images.jld2"),
    (x_train = train.features, y_train = train.targets, x_test = test.features, y_test = test.targets)
)
"""

data = deserialize(joinpath(datadir("datasets"), "mnist_images.jld2"))
dataloaders = (
    train = DataLoader(
        (flatten(data.x_train), data.y_train),#reshape(data.x_train, 28, 28, 1,:)
        batchsize=512,
        shuffle=true,
        partial=true
    ),
    test = DataLoader(
        (flatten(data.x_test), data.y_test),
        batchsize=256,
        shuffle=false,
        partial=true
    )
)


args = (;
    name = "VariationalAutoencoder",
    z_dim = 2,
    in_dim = 28*28,
    out_dim = 28*28,
    hidden = 128,
    depth = 3,
    activation = relu
)


#model = VariationalAutoencoder(encoder, decoder)
model = VariationalAutoencoder(args.in_dim, args.z_dim, args.out_dim; hidden=args.hidden, depth=args.depth, activation=args.activation)
model = cu(model)
opt = Optimisers.setup(AdamW(; eta=1e-4, lambda=1e-4), model);

#loss_f = (x,y)->Flux.Losses.mse(σ.(x), y) * 728
#loss_f = (x,y)->Flux.Losses.logitbinarycrossentropy(x,y; agg=u->mean(sum(u, dims=1)))
#loss_f = Flux.Losses.logitbinarycrossentropy
#loss_f = my_logitbinarycrossentropy

loss_f = (x,y) -> mean(sum(Flux.Losses.logitbinarycrossentropy.(x, y), dims=1))


# test
x = first(dataloaders.train)[1];
x̂ = model(cu(x))|>cpu;

elbo_with_logging(model, x|>cu, my_logitbinarycrossentropy; β=1f0)

_,_,logs = optim_step(model, x, opt, loss_f, cu; β=1f0);
logs



for i in 1:300
    logs= nothing
    for batch in tqdm(dataloaders.train)
        x = batch[1]
        model, opt, logs = optim_step(model, x, opt, loss_f, cu; β=0.5f0); #loss_f 1f0
    end
    #model, opt, logs = optim_step(model, x, opt, loss_f, cu; β=1f0);
    println("Epoch $i: $(logs)")
end

x̂ = model(cu(x))|>cpu;

heatmap(reshape(x, 28, 28, 1, :)[:, end:-1:1, 1, 1])
heatmap(reshape(σ.(x̂), 28, 28, 1, :)[:, end:-1:1, 1, 1])

x = first(dataloaders.test)[1];
y = first(dataloaders.test)[2];
x̂ = model(cu(x))|>cpu;
plot_batch_heatmaps(σ.(x̂), x; n=8, image_size=(28, 28), colorrange=(0, 1))
μ, Σ = model.encoder(cu(x))|>cpu
z = μ + Σ .* MLUtils.randn_like(μ)
plot_class_scatter(z, y)

model.encoder(cu(x))[1]|>cpu

model.encoder(cu(x))[1]|>cpu|>mean
model.encoder(cu(x))[2]|>cpu|>mean

minimum(10f0, i ./ 10f0)

zs, ys, μs, Σs = [], [], [], []
for (x,y) in tqdm(dataloaders.train)
    μ, Σ = model.encoder(cu(x))
    z = μ + Σ .* MLUtils.randn_like(μ)
    push!(zs, z)
    push!(ys, y)
    push!(μs, μ)
    push!(Σs, Σ)
end

zs = reduce(hcat, zs);
ys = reduce(vcat, ys);
μs = reduce(hcat, μs);
Σs = reduce(hcat, Σs);

plot_class_scatter(zs, ys)

# save model and opt state
model_state = Flux.state(model|>cpu);
opt_state = Flux.state(opt|>cpu);
ui = Int(rand(1:10^6))

model_state_dir = datadir("MultiObjectGeneration", "vae_mnist_$(ui)", "model_state")
models_dir = datadir("MultiObjectGeneration", "vae_mnist_$(ui)", "models")
predictions_dir = datadir("MultiObjectGeneration", "vae_mnist_$(ui)", "predictions")
mkpath(datadir("MultiObjectGeneration", "vae_mnist_$(ui)"))
mkpath(model_state_dir)
mkpath(models_dir)
mkpath(predictions_dir)

open(joinpath(datadir("MultiObjectGeneration", "vae_mnist_$(ui)"), "config.json"), "w") do io
    JSON3.pretty(io, merge(args, (; activation = string(args.activation),)))
end

jldsave(joinpath(model_state_dir, "model_state_final.jld2"), model_state = model_state, opt_state = opt_state)
serialize(joinpath(models_dir, "model_final.jls"), (model = model|>cpu, opt = opt|>cpu))

jldsave(joinpath(predictions_dir, "predictions.jld2"), zs = zs, ys = ys, μs = μs, Σs = Σs)
serialize(joinpath(predictions_dir, "predictions.jls"), (zs = zs, ys = ys, μs = μs, Σs = Σs))

save(joinpath(predictions_dir, "predictions.png"), plot_batch_heatmaps(σ.(model(cu(x))|>cpu), x; n=8, image_size=(28, 28), colorrange=(0, 1)))