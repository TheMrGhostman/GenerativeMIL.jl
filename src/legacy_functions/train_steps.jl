using Flux: Params
using Flux.Optimise: AbstractOptimiser

# SetVAE

function train_step(
    ps::Params, model::SetVAE, batch, loss::Function, optimizer::AbstractOptimiser, scheduler;
    iter=1, anealer=x->1, to_gpu::Bool=false, kwargs...)
    
    # batch and parameter preprocessing
    beta = anealer(iter)
    x, x_mask = transform_batch(batch, true)
    x = (to_gpu) ? x|>gpu : x
    x_mask = (to_gpu) ? x_mask|>gpu : x_mask

    # Forward Pass
    loss_, back = Flux.pullback(ps) do 
        loss(model, x, x_mask, beta) 
    end;
    # Backward Pass
    grad = back((1f0,0f0));
    # get learning rate from scheduler
    optimizer.eta = scheduler(iter)
    # Optimalisation step
    Flux.Optimise.update!(optimizer, ps, grad);
    return (training_loss = loss_[1], training_kld_loss = loss_[2],)
end

function valid_loop(model::SetVAE, loss::Function, dataloader; 
    anealer=anealer, iter=i, to_gpu::Bool=false, kwargs...)

    beta = anealer(iter)
    total_loss, total_kld = 0, 0
    for batch in dataloader
        xv, xv_mask = transform_batch(batch, true)
        xv = (to_gpu) ? xv|>gpu : xv
        xv_mask = (to_gpu) ? xv_mask|>gpu : xv_mask
        # compute validation loss
        v_loss, v_kld = loss(model, xv, xv_mask, beta);
        total_loss += v_loss;
        total_kld += v_kld;
    end   
    # compute losses
    total_loss /= length(dataloader)
    total_kld /= length(dataloader)
    return total_loss, (val_loss = total_loss, val_kld_loss = total_kld,)
end

# FoldingNet_VAE -> default train_step

# PoolModel (the same for batched data without masks)
function train_step(
    ps::Params, model::PoolModel, batch, loss::Function, optimizer::AbstractOptimiser, scheduler;
    iter=1, to_gpu::Bool=false, kwargs...)
    
    # batch and parameter preprocessing
    x, x_mask = transform_batch(batch, true)
    x = (to_gpu) ? x|>gpu : x #x_mask = (to_gpu) ? x_mask|>gpu : x_mask

    # Forward Pass
    loss_, back = Flux.pullback(ps) do 
        loss(model, x) 
    end;
    # Backward Pass
    grad = back(1f0);
    # get learning rate from scheduler
    optimizer.eta = scheduler(iter)
    # Optimalisation step
    Flux.Optimise.update!(optimizer, ps, grad);
    return (training_loss = loss_,)
end

function valid_loop(model::PoolModel, loss::Function, dataloader; 
    to_gpu::Bool=false, kwargs...)

    total_loss = 0
    for batch in dataloader
        xv, xv_mask = transform_batch(batch, true)
        xv = (to_gpu) ? xv|>gpu : xv
        #xv_mask = (to_gpu) ? xv_mask|>gpu : xv_mask
        # compute validation loss
        v_loss = loss(model, xv);
        total_loss += v_loss;
    end   
    # compute losses
    total_loss /= length(dataloader)
    return total_loss, (val_loss = total_loss,)
end


# VQ_PoolAE supervised version
function train_step(
    ps::Params, model::VQ_PoolAE, batch, loss::Function, optimizer::AbstractOptimiser, scheduler;
    iter=1, kwargs...)
    # broadcasting version
    loss_f(x, y) = loss(model, x, y)
    # get obs / to make sure
    x, y = MLUtils.getobs(batch)
    # forward pass
    loss_, back = Flux.pullback(ps) do 
        Flux.mean(loss_f.(x, y))
    end;
    # backward pass
    grad = back(1f0);
    # get lr from scheduler
    optimizer.eta = scheduler(iter)
    # optimize step
    Flux.Optimise.update!(optimizer, ps, grad);
    return (training_loss = loss_, )
end

function valid_loop(model::VQ_PoolAE, loss::Function, dataloader; kwargs...)
    loss_f(x, y) = loss(model, x, y)
    total_loss=0
    for batch in dataloader
        x,y = MLUtils.getobs(batch)
        v_loss = Flux.mean(loss_f.(x,y))
        total_loss += v_loss;
    end
    total_loss /= length(dataloader)
    return total_loss, (val_loss = total_loss,)
end