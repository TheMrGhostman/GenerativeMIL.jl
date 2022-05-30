using DrWatson
using GenerativeMIL
using Flux
using Flux3D: chamfer_distance
using MLDataPattern
using ProgressMeter: Progress, next!
using PyPlot
using BSON
using Dates
using GenerativeMIL: transform_batch
using GenerativeMIL.Models: loss, check
using CUDA

BS = 64

train, test = GenerativeMIL.load_and_standardize_mnist();

dataloader = RandomBatches(train[1], size=BS)

sv = GenerativeMIL.Models.SetVAE(3,64,4,[16,8,4,2], [2,4,8,16], 1, 32, Flux.relu, 5, 10) |> gpu;
print(sv)
ps = Flux.params(sv);
losses = []
learning_rate = 1e-3
beta = 0.01f0
iters = 100 * fld(60000,BS) # 100 epochs
opt = ADAM(learning_rate)
progress = Progress(iters)


for (i, batch) in enumerate(dataloader)
    x, x_mask = transform_batch(batch,true)
    x, x_mask = x|>gpu, x_mask|>gpu
    
    loss_, back = Flux.pullback(ps) do 
        loss(sv, x, x_mask, beta, const_module=CUDA) 
    end;
    grad = back((1f0,0f0));
    Flux.Optimise.update!(opt, ps, grad);
    #@info "loss = $(loss)"
    push!(losses, loss_)
    next!(progress; showvalues=[(:iters, "$(i)/$(iters)"),(:loss, loss_[1]),(:klds, loss_[2])])
    if i == iters
        break
    end
end

tagsave(datadir("model_test_$(now)_gpu.bson"), Dict(:model => sv, :loss => map(x->x[1], losses), :klds => map(x->x[2], losses), :lr => learning_rate, :beta => beta, :iters =>iters), safe=true)
