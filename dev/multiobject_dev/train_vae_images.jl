using Revise
#using DrWatson
#@quickactivate

using ArgParse
using Random
using Serialization
using YAML
using OrderedCollections
using GenerativeMIL
using Flux
using CUDA
using JLD2, JSON3
using MLUtils
using ProgressBars

dict2nt(x) = (; (Symbol(k) => v for (k, v) in x)...)

"""
using MLDatasets
train = MNIST(split=:train)
test = MNIST(split=:test)
serialize(
    joinpath(datadir("datasets"), "mnist_images.jld2"),
    (x_train = train.features, y_train = train.targets, x_test = test.features, y_test = test.targets)
)
"""

data = deserialize(joinpath("B:\\Github-Repos\\GenerativeMIL.jl\\data\\datasets", "mnist_images.jld2"))
dataloaders = (
    train = DataLoader(
        (reshape(data.x_train, 28, 28, 1,:), data.y_train),
        batchsize=512,
        shuffle=true,
        partial=true
    ),
    test = DataLoader(
        (reshape(data.x_test, 28, 28, 1,:), data.y_test),
        batchsize=256,
        shuffle=false,
        partial=true
    )
)

args = (;
    model = (;
        name = "ConvolutionalVariationalAutoencoder",
        z_dim = 16,
        channels = 4,
        activation = leakyrelu
    )
)

# number of neurons is given by the number of channels in the last convolutional layer times the spatial dimensions of the feature map. For MNIST, this is 64 channels * 4 * 4 = 1024 neurons.
encoder = Flux.Chain(
    Flux.Conv((4, 4), 1=>args.model.channels, args.model.activation; stride=2, pad=0),
    Flux.Conv((3, 3), args.model.channels=>args.model.channels * 2, args.model.activation; stride=2, pad=1),
    Flux.Conv((3, 3), args.model.channels * 2=>args.model.channels * 4, args.model.activation; stride=2, pad=1),
    Flux.flatten,
    SplitLayer(
        16 * args.model.channels * 4, 
        (args.model.z_dim, args.model.z_dim), 
        (identity, softplus))
)

decoder = Flux.Chain(
    Flux.Dense(args.model.z_dim, 16 * args.model.channels * 4, args.model.activation),
    x->reshape(x, (4, 4, args.model.channels * 4, :)),
    Flux.ConvTranspose((3, 3), args.model.channels * 4=>args.model.channels * 2, args.model.activation; stride=2, pad=1),
    Flux.ConvTranspose((3, 3), args.model.channels * 2=>args.model.channels, args.model.activation; stride=2, pad=1),
    Flux.ConvTranspose((4, 4), args.model.channels=>1, σ; stride=2, pad=0)
)

model = VariationalAutoencoder(encoder, decoder)
model = cu(model)
loss_f = (x,y)->Flux.Losses.mse(x,y) * 28*28
opt = Optimisers.setup(AdamW(; eta=1e-3, lambda=1e-4), model);



# test
elbo_with_logging(model, x|>gpu, Flux.Losses.mse; β=1f0)

_,_,logs = optim_step(model, x, opt, loss_f, cpu; β=1f0);
logs


for i in 1:1000
    model, opt, logs = optim_step(model, x, opt, loss_f, gpu; β=1f0);
    println("Epoch $i: $(logs)")
end

x̂ = model(gpu(x))|>cpu;

heatmap(x[end:-1:1, :, 1, 1])
heatmap(x̂[end:-1:1, :, 1, 1])



function testing_sizees()
    e = Flux.Chain(
            Flux.Conv((4, 4), 1=>2, relu; stride=2, pad=0),
            Flux.Conv((3, 3), 2=>3, relu; stride=2, pad=1),
            Flux.Conv((3, 3), 3=>4, relu; stride=2, pad=1)
        )

    d = Flux.Chain(
            Flux.ConvTranspose((3, 3), 4=>3, relu; stride=2, pad=1),
            Flux.ConvTranspose((3, 3), 3=>2, relu; stride=2, pad=1),
            Flux.ConvTranspose((4, 4), 2=>1, relu; stride=2, pad=0),
            #Flux.ConvTranspose((3, 3), 1=>1, relu; stride=2, pad=1),
            #Flux.Conv((3, 3), 1=>1, σ; stride=2, pad=1)
        )

    o = copy(randn(Float32, 28, 28, 1, 16));
    for layer in e.layers
        si = size(o)
        o = layer(o)
        println("size(o): $si -> $(size(o))")
    end

    #o = copy(x)
    for layer in d.layers
        si = size(o)
        o = layer(o)
        println("size(o): $si -> $(size(o))")
    end
end

using CUDA, Flux

T = Float32
x = randn(T, 28,28,1,16);
xg = cu(x);
m =  Flux.Conv((4, 4), 1=>2, relu; stride=2, pad=0);
mg = cu(m);
m1 = Flux.Dense(28*28, 128, relu)
m1g = cu(m1)

typeof(x)
typeof(xg)
typeof(m)
typeof(mg)
typeof(m1)
typeof(m1g)

m(x) |> size
mg(xg) |> size
m1(reshape(x, :, 16)) |> size
m1g(reshape(xg, :, 16)) |> size
m1(reshape(x, :, 16)) |> typeof
m1g(reshape(xg, :, 16)) |> typeof


# 1. Definice sítě na GPU
model = Chain(
    Conv((3, 3), 3 => 16, relu), 
    Flux.flatten, 
    Dense(16 * 26 * 26 => 10)
) |> gpu

# 2. Správně formátovaná testovací data: Float32 a 4 rozměry (W, H, C, N)
test_x = rand(Float32, 28, 28, 3, 2) |> gpu;

# 3. Dopředný průchod
vystup = model(test_x)
println("Úspěch! Rozměr výstupu: ", size(vystup))