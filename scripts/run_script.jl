using DrWatson
using GenerativeMIL
using Flux
using Flux3D: chamfer_distance
using MLDataPattern
using ProgressMeter: Progress, next!
using PaddedViews
using PyPlot
using BSON

BS = 64

function loss_f(m::GenerativeMIL.Models.SetVAE, x::AbstractArray{<:Real}, x_mask::AbstractArray{Bool}, β::Float32=0.01f0)
    #encoder
    x1 = m.encoder.expansion(x) .* x_mask
    x1, h_enc1 = m.encoder.layers[1](x1, x_mask)
    x1, h_enc2 = m.encoder.layers[2](x1, x_mask)
    x1, h_enc3 = m.encoder.layers[3](x1, x_mask)
    x1, h_enc4 = m.encoder.layers[4](x1, x_mask)
    
    #println("encoder", x1)
    _, sample_size, bs = size(x_mask)
    z = m.prior(sample_size, bs)
    #println("sampling", z)
    #decoder
    klds = 0
    x1 = m.decoder.expansion(z) .* x_mask
    x1, kld, _,_ = m.decoder.layers[1](x1, h_enc4, x_mask)
    klds += kld
    x1, kld, _,_ = m.decoder.layers[2](x1, h_enc3, x_mask)
    klds += kld
    x1, kld, _,_ = m.decoder.layers[3](x1, h_enc2, x_mask)
    klds += kld
    x1, kld, _,_ = m.decoder.layers[4](x1, h_enc1, x_mask)
    klds += kld
    x1 = m.decoder.reduction(x1) .* x_mask
    #println("decoder", x1)
    #loss = ChamferDistanceLoss(x, x1) + β * klds
    loss = chamfer_distance(x, x1) + β * klds
    return loss, klds
end

train, test = GenerativeMIL.Models.load_and_standardize_mnist();

dataloader = RandomBatches(train[1], size=BS)

sv = GenerativeMIL.Models.SetVAE(3,64,4,[16,8,4,2], [2,4,8,16], 5, 32);

ps = Flux.params(sv);
opt = ADAM()
losses = []
iters = 100 * fld(60000,BS) # 100 epochs
progress = Progress(iters)
beta = 0.01f0

for (i, batch) in enumerate(dataloader)
    x, x_mask = GenerativeMIL.Models.transform_batch(batch,true)#GenerativeMIL.Models.transform_batch(batch)
    #println(x|>size, x_mask|>size)
    loss, back = Flux.pullback(ps) do 
        loss_f(sv, x, x_mask, beta) 
    end;
    grad = back((1f0,0f0));
    Flux.Optimise.update!(opt, ps, grad);
    #@info "loss = $(loss)"
    push!(losses, loss)
    next!(progress; showvalues=[(:iters, "$(i)/$(iters)"),(:loss, loss[1]),(:klds, loss[2])])
    if i == -1 #placehodler for annealing
        beta = 0.01f0
    elseif i == iters
        break
    end
end

tagsave(datadir("model_test_1.bson"), Dict(:model => sv, :loss => map(x->x[1], losses), :klds => map(x->x[2], losses)), safe=true)
