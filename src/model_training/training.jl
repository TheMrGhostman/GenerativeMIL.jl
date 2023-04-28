function StatsBase.fit!(model, data::Tuple, loss::Function; 
    batchsize=64, epochs=1000, early_stopping::Bool=true, patience::Int=50, 
    lr_sch=false, lr=0.001, milestones=[0.02, 0.8], lrscale=5,
    beta=1.0, beta_anealing=50, check_every=20, max_train_time=82800, verbose=true, kwargs...)

    # 1) save start time for checking of time budget
    start_time = time()
    
    # 2) logging | setup history log
    history = ValueHistories.MVHistory()

    # 3) check for GPU / if found model is set to GPU 
    global to_gpu = false
    try
        if CUDA.devices() |> length > 0
            to_gpu = true
            model = model |> gpu
        end
        @info "GPU goes brrrrrr"
    catch
        @info "No GPU found"
    end

    @info "Module of model -> $(get_device(model))"

    # 4) initialize early stipping procedure
    early_stop = (early_stopping) ? EarlyStopping(model, patience) : x->0 

    # 5) preprocess data
    tr_data, val_data = data[1], data[2]

    # 6) get max iteration 
    max_iters = get_max_iters(tr_data, batchsize, epochs; verbose=verbose)

    # 7) create dataloaders for train and validation dataset # batchsize
    dataloaders = CreateDataloaders(tr_data, val_data, batchsize, max_iters, verbose=verbose, as_dict=true)

    # 8) initialize optimizer
    optimizer = ADAM(lr)
    ps = Flux.params(model)

    # 9) learining rate scheduler
    scheduler = CreateLrScheduler(lr_sch, lr, max_iters; milestones=milestones, scale=lrscale)

    # 10) β scheduler / anealer
    anealer = CreateAnealer(beta, fld(beta_anealing*max_iters, epochs))

    # 11) iterate over dataloader
    for (i, batch) in enumerate(dataloaders["train"])
        # 11)a) logging = train_step(model, batch, loss_function, optimizer, lr_scheduler, anealer)
        logging = train_step(ps, model, batch, loss, optimizer, scheduler; iter=i, anealer=anealer, to_gpu=to_gpu)

        # 11)b) logging to history
        foreach(key->push!(history, key, i, logging[key]), keys(logging))
        push!(history, :iter, i, i)

        if mod(i, check_every) == 0  
            # 11)c) es_loss, logging = valid_step(model, val_dataloader, anealer)
            es_loss, logging_v = valid_loop(model, loss, dataloaders["valid"]; anealer=anealer, iter=i, to_gpu=to_gpu)

            # 11)d) val logging to history
            foreach(key->push!(history, key, i, logging_v[key]), keys(logging_v))

            # 11)d) verbose losses 
            if verbose
                tr_l = join(map(key->" $(key): $(round(logging[key], digits=9, RoundUp)) |", keys(logging)))
                va_l = join(map(key->" $(key): $(round(logging_v[key], digits=9, RoundUp)) |", keys(logging_v)))
                @info join(("$(num2dig(i,max_iters)) - training -> ", tr_l, "| validation -> ", va_l))
            end

            # 11)e) early stopping step & terminate training criterion
            if early_stop(es_loss, model) 
                @info "Stopped training after $(i) iterations."
                break
            end
        end
        # 11)f) terminate training criterion
        if (time() - start_time > max_train_time) | (i > max_iters)
            @info "Stopped training after $(i) iterations, $((time() - start_time) / 3600) hours."
            break
        end
    end
    # 12) return everything
    best_model = early_stop.best_model |> cpu
    iter_performed = length(get(history, :iter)[1])
    npars = sum(map(p -> length(p), Flux.params(best_model)))
    return (history = history, iterations = iter_performed, model = best_model, npars = npars)
end

# Default functions train_step and valid_step | can be specialized for each model

function train_step(params, model, batch, loss, optimizer, scheduler; iter=1, kwargs...)
    # broadcasting version
    loss_f(x) = loss(model, x)
    # get obs / to make sure
    x = MLUtils.getobs(batch)
    # forward pass
    loss_, back = Flux.pullback(params) do 
        Flux.mean(loss_f.(x))
    end;
    # backward pass
    grad = back(1f0);
    # get lr from scheduler
    optimizer.eta = scheduler(iter)
    # optimize step
    Flux.Optimise.update!(optimizer, params, grad);
    return (training_loss = loss_, )
end

function valid_loop(model, loss, dataloader; kwargs...)
    total_loss = 0
    loss_f(x) = loss(model, x)
    for batch in dataloader
        x = MLUtils.getobs(batch)
        v_loss = Flux.mean(loss_f.(x))
        total_loss += v_loss;
    end
    total_loss /= length(dataloader)
    return total_loss, (val_loss = total_loss, )
end


mutable struct EarlyStopping
    best_model
    best_loss
    patience
    curr_patience
end

function EarlyStopping(model, patience)
    return EarlyStopping(deepcopy(model), Inf, copy(patience), copy(patience))
end

function (es::EarlyStopping)(loss, model)
    if loss < es.best_loss
        es.best_loss = loss
        es.curr_patience = deepcopy(es.patience)
        es.best_model = deepcopy(model)
    else
        es.curr_patience -= 1 
    end
    (es.curr_patience == 0) ? true : false # to stop: true/false
end

size_last_dim(data::AbstractArray{T, N}) where {T,N} = size(data, N)
size_last_dim(data::Tuple{AbstractArray{T, N}, Any}) where {T,N} = size(data[1], N)

function get_max_iters(data, batchsize, epochs; verbose::Bool=true)
    N =  size_last_dim(data)
    multiplier_ = fld(N, batchsize)
    max_iters = (multiplier_ == 0) ? epochs : epochs * multiplier_
    if verbose @info "dataset//batchsize = $(multiplier_) -> max_iters = $(max_iters)" end;
    return max_iters
end

function CreateDataloaders(tr_data, val_data, batchsize=64, max_iters=1; as_dict::Bool=true, kwargs...)
    # TODO modify for supervised data, i.e. tr_data::Tuple
    dl_train = MLDataPattern.RandomBatches(tr_data, size=batchsize, count=max_iters)
    dl_valid = Flux.Data.DataLoader(val_data, batchsize=batchsize)
    return (as_dict) ? Dict("train"=>dl_train, "valid"=>dl_valid) : (dl_train, dl_valid)
end


function CreateLrScheduler(sch_name, lr, max_iters; milestones=[0.02, 0.8], scale=5, kwargs...)
    @assert sch_name in [false, "false", "Linear2ndHalf", "WarmupCosine"] # this can be expanded later
    if sch_name == "WarmupCosine"
        scheduler = WarmupCosine(1e-7, lr*scale, lr, Int(milestones[1] * max_iters), Int(milestones[2] * max_iters))
        # from 0 to milestones[1]% iters there is linear increase of learing rate with "scale"
        # from milestones[1]% to milestones[2]% there is cosine decay of learing rate 
        # from milestones[2]% to 100% iters there is constant learing rate 
    elseif sch_name == "Linear2ndHalf"
        scheduler = it -> lr .* min.(1.0, 2.0 - it/(0.5*max_iters))
        #lr .* min.(1.0, map(x -> 2.0 - x/(0.5*max_iters), 1:max_iters)) 
        # learning rate decay (0%,50%) -> 1 , (50%, 100%) -> linear(1->0)
    else
        scheduler = x -> lr
        # constant learning rate 
    end
    return scheduler
end


function CreateAnealer(max_value, milestone)
    new_value = it->max_value * min(1f0, it/milestone)
end

function num2dig(x, max)
    (map(x->"0", 1:(floor(log10(max))-floor(log10(x))))...,string(x)) |> join
end