function train_model!(
    model::AbstractGenModel, 
    dataloaders::NamedTuple{(:train, :valid), <:Tuple{DataLoader, DataLoader}},
    optimiser::Optimisers.AbstractRule, 
    loss_function::Function=chamfer_distance,
    β_scheduler::Union{Function, Scheduler, Sequence} = x->0f0, 
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

    train_model!(model, dataloaders, optimiser; loss_function=loss_function, β_scheduler=β_scheduler, lr_scheduler=lr_scheduler, use_gpu=use_gpu, model_dir=model_dir, verbose=verbose, valid_check_interval=valid_check_interval, checkpoint_interval_epochs=checkpoint_interval_epochs, epochs=epochs, early_stopping=early_stopping, patience=patience, max_train_time=max_train_time, grad_skip=grad_skip, kwargs...)

end


function train_model!(
    model::AbstractGenModel, 
    dataloaders::NamedTuple{(:train, :valid), <:Tuple{DataLoader, DataLoader}}, 
    optimiser::Optimisers.AbstractRule; ## KWARGS FROM HERE
    loss_function::Function=chamfer_distance, 
    β_scheduler::Union{Function, Scheduler, Sequence, Nothing} = x->0f0,
    lr_scheduler::Union{Function, Scheduler, Sequence, Nothing} = nothing,
    use_gpu::Bool=true,
    model_dir::String=pwd(), 
    verbose::Bool=false, 
    valid_check_interval::Int = 1000,
    checkpoint_interval_epochs::Int=10,
    epochs::Int=1000, 
    early_stopping::Bool=true,
    patience::Int = 10^4,
    max_train_time::Int=82800, 
    grad_skip::Union{Real, Bool}=false,
    validation_verbose::Bool=false,
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
                        tr_log=logs, verbose=validation_verbose, epoch_info=(epoch, epochs), iter_info=(it, max_iters)
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

            if mod(epoch, checkpoint_interval_epochs) == 0 || epoch == epochs
                @info "Saving checkpoint after epoch $(epoch)"
                serialize(
                    joinpath(model_dir, "models", "model_ep=$(lpad_number(epoch, epochs)).jls"), 
                    (model = model |> cpu, epoch = epoch, iter = max_iters, idx = idx)
                )
            end
            # 10) after epoch validation stop
            validation_check(
                model, dataloaders.valid, loss_function, β, device, history, json_logger, early_stop, epoch*max_iters; 
                verbose=verbose, epoch_info=(epoch, epochs), iter_info=(0, max_iters)
            );

        end
    catch e   
        # if error happens stop training and log error, return what we have
        @info "Stopped training after $((time() - start_time) / 3600) hours due to error \n $(e)"
    finally
        # close logger
        close(json_logger)
        # save best model
        epoch_=floor(Int, idx / max_iters)
        it_ = mod(idx, max_iters)
        serialize(
            joinpath(model_dir, "models", "best_model.jls"), 
            (model = early_stop.best_model |> cpu, best_loss=early_stop.best_loss, reached_epoch = epoch_, reached_iter = it_, idx = idx)
        )
        # saving best model and if error is occured it saved the last model, so we have something to work with
        # reached_epoch and reached_iter are for logging purposes, to know how long the training went on before error happened, if it happens. If not, they will just be the same as last checkpoint.
        # it saves BEST model not LAST model
    end
    @info "training took $(time() - start_time) s " #TODO make me nicer
    return (model=model, opt=opt, history=history)
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
) where T <: AbstractFloat
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
        @info join(("ep: $(lpad_number(epoch_info...)) | it: $(lpad_number(iter_info...)) - training -> ", tr_l, "| validation -> ", va_l)) #FIXME epochs & iters
    end

    # 7d) early stopping step & terminate training criterion
    stop_training =  early_stopper(es_loss, model) ? begin @info "Stopped training after $(idx) iterations."; true; end : false

    stop_training
end
