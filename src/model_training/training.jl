function train_model!(
    model::AbstractGenModel, 
    dataloaders::NamedTuple{(:train, :valid), <:Tuple{DataLoader, DataLoader}},
    optimiser::Optimisers.AbstractRule, 
    loss_function::Function=chamfer_distance,
    β_scheduler::Union{Function, Scheduler, Sequence, Nothing} = x->0f0, 
    lr_scheduler::Union{Function, Scheduler, Sequence, Nothing} = nothing; # here starts kwargs
    use_gpu::Bool=true,
    model_dir::String="", 
    verbose::Bool=false, 
    valid_check_interval::Int = 1000,
    checkpoint_interval_epochs::Int=10,
    epochs::Int=1000, 
    early_stopping::Bool=true,
    patience::Int = 10^4,
    max_train_time::Int=82800, 
    grad_skip::Union{Real, Bool}=false,
    kwargs...
)

    train_model!(model, dataloaders, opt; loss_function=loss_function, β_scheduler=β_scheduler, lr_scheduler=lr_scheduler, valid_step=valid_step, use_gpu=use_gpu, model_dir=model_dir, verbose=verbose, valid_check_interval=valid_check_interval, checkpoint_interval_epochs=checkpoint_interval_epochs, epochs=epochs, early_stopping=early_stopping, patience=patience, max_train_time=max_train_time, grad_skip=grad_skip, kwargs...)

end


function train_model!(
    model::AbstractGenModel, 
    dataloaders::NamedTuple{(:train, :valid), <:Tuple{DataLoader, DataLoader}}, 
    optimiser::Optimisers.AbstractRule; ## KWARGS FROM HERE
    loss_function::Function=chamfer_distance, 
    β_scheduler::Union{Function, Scheduler, Sequence, Nothing} = x->0f0,
    lr_scheduler::Union{Function, Scheduler, Sequence, Nothing} = nothing,
    use_gpu::Bool=true,
    model_dir::String="", 
    verbose::Bool=false, 
    valid_check_interval::Int = 1000,
    checkpoint_interval_epochs::Int=10,
    epochs::Int=1000, 
    early_stopping::Bool=true,
    patience::Int = 10^4,
    max_train_time::Int=82800, 
    grad_skip::Union{Real, Bool}=false,
    kwargs...
)

    # 1) save start time for checking of time budget
    start_time = time()
    
    # 2a) logging | setup history log
    history = ValueHistories.MVHistory()
    # 2b) initialize jsonl logger
    if !isdir(model_dir); begin mkdir(model_dir); mkdir(joinpath(model_dir, "models")); end; end;
    json_logger = JSONLLogger(joinpath(model_dir, "trainlog.jsonl"))

    # 3) check for GPU / if found model is set to GPU 
    if use_gpu & CUDA.functional() 
        device = gpu
        @info "GPU goes brrrrr"
    else
        device = cpu
        @info "No GPU found"
    end
    # 3b) move model to device, either GPU or CPU
    model = model |> device
    # 3c) init optim from model
    opt = Optimisers.setup(optimiser, model)
    η = optimiser.eta;

    # 4) initialize early stipping procedure
    early_stop = (early_stopping) ? EarlyStopping(model, patience) : EarlyStopping(model, Inf)
    stop_training = false

    # 5) get max iteration and idx
    max_iters = length(dataloaders.train)
    global idx = 0

    try
        for epoch in 1:epochs
            # 6a) β anealing, if default x->0, If β not needed for model, it is just ommited. **args
            β = β_scheduler(epoch)
            # 6b) learning rate scheduler
            if !isnothing(lr_scheduler)
                ηₙ = lr_scheduler(epoch) 
                Optimisers.adjust!(opt, ηₙ)
                push!(history, :η, ηₙ)
                η = ηₙ
            end

            for (it, batch) in tqdm(enumerate(dataloaders.train))
                # 6c) global iter 
                idx = (epoch-1)*max_iters + it
                # 6d) perform one optimization step
                model, opt, logs = optim_step(model, batch, opt, loss_function, device; β=β, ∇skip=grad_skip, kwargs...); # β is part of log if used
                # 6e)  Logging and checking
                ## Logging to value histores and jsonl file
                foreach(p->push!(history, p.first, idx, p.second), pairs(logs))
                log!(json_logger, merge((;idx=idx, epoch=epoch, iter=it, mode="train"), logs, (;η = η))) 
                # 7) do validation and early stopping if necessery
                if mod(it, valid_check_interval) == 0 
                    stop_training = validation_check(
                        model, dataloaders.valid, loss_function, β, device, history, json_logger, early_stop, idx;
                        tr_log=logs, verbose=verbose, epoch_info=(epoch, epochs), iter_info=(it, max_iters)
                    )
                    if stop_training; break; end
                end
                # 8) time termination criterion
                if (time() - start_time > max_train_time)
                    stop_training = true
                    break
                end
            end # end of training within epoch
            if stop_training; break; end # propagation of early stopping criterion
            # 9) save checkpoint

            if mod(epoch, checkpoint_interval_epochs) == 0
                @info "Saving checkpoint after epoch $(epoch)"
                serialize(
                    joinpath(model_dir, "models", "model_ep=$(pad_epoch(epoch, epochs)).jls"), 
                    (model = model |> cpu, epoch = epoch, iter = max_iters, idx = idx)
                )
            end
            # 10) after epoch validation stop
            validation_check(
                model, dataloaders.valid, loss_function, β, device, history, json_logger, early_stop, epoch*max_iters; 
                verbose=verbose, epoch_info=(epoch, epochs), iter_info=(0, max_iters)
            );
        end
        close(json_logger)
    catch e   
        # if error happens stop training and log error, return what we have
        @info "Stopped training after $((time() - start_time) / 3600) hours due to error \n $(e)"
    finally
        # close logger
        close(json_logger)
    end
    # save best model
    epoch_=floor(Int, idx / max_iters)
    it_ = mod(idx, max_iters)
    serialize(
        joinpath(model_dir, "models", "best_model_ep=$(pad_epoch(epoch_, epochs))_iter=$(pad_epoch(it_, max_iters)).jls"), 
        (model = early_stop.best_model |> cpu, epoch = epoch_, iter = it_, idx = idx)
    )

    return model, history
end

function validation_check(
    model::AbstractGenModel, 
    dataloader::DataLoader, 
    loss_function::Function, 
    β::Union{Vector{T}, T}, 
    device::Function, 
    history::MVHistory, 
    logger::JSONLLogger, 
    early_stopper::EarlyStopping,
    idx::Union{Int, Nothing}=nothing; 
    tr_log::Union{NamedTuple, Nothing}=nothing,
    verbose::Bool=false,
    epoch_info::Tuple{Int, Int}=(0, 0),
    iter_info::Tuple{Int, Int}=(0, 0),
    kwargs...
) where T <: Real
    # 7a) validation loop or step
    vlogs, es_loss = valid_step(model, dataloader, loss_function; β=β, device=device, kwargs...)
    # 7b) logging of validation step
    push_fn = !isnothing(idx) ? p->push!(history, p.first, idx, p.second) : p->push!(history, p.first, p.second)
    foreach(push_fn, pairs(vlogs))

    log!(logger, merge((;idx=idx, epoch=epoch_info[1], iter=iter_info[1], mode="valid"), vlogs))
    # 7c) verbose losses 
    if verbose
        tr_logs = !isnothing(tr_log) ? tr_log : (;)
        tr_l = join(map(key->" $(key): $(round(tr_logs[key], digits=9, RoundUp)) |", keys(tr_logs)))
        va_l = join(map(key->" $(key): $(round(vlogs[key], digits=9, RoundUp)) |", keys(vlogs)))
        @info join(("ep: $(pad_epoch(epoch_info...)) | it: $(pad_epoch(iter_info...)) - training -> ", tr_l, "| validation -> ", va_l)) #FIXME epochs & iters
    end

    # 7d) early stopping step & terminate training criterion
    stop_training =  early_stopper(es_loss, model) ? begin @info "Stopped training after $(i) iterations."; true; end : false

    stop_training
end






# TODO delete after this

function fit!(
    model, 
    data::Tuple, 
    optim_step::Function,
    loss_function::Function=chamfer_distance; 
    batchsize=64, 
    epochs=1000, 
    early_stopping::Bool=true, 
    patience::Int=50, 
    lr_sch=false, 
    lr=0.001, 
    milestones=[0.02, 0.8], 
    lrscale=5,
    beta=1.0, 
    beta_anealing=50, 
    check_every=20, 
    max_train_time=82800, 
    verbose=true, 
    kwargs...
)

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

function num2dig(x, max)
    (map(x->"0", 1:(floor(log10(max))-floor(log10(x))))...,string(x)) |> join
end