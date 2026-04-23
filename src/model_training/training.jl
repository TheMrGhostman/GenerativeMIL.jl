"""
Create a small validation preview batch from a dataloader batch.

Arguments:
- `batch`: either `x` or `(x, x_mask)` as produced by validation dataloader.
- `n_samples`: number of samples to keep from the batch.

Returns:
- Preview batch in the same structural format as input (`x` or `(x, x_mask)`).
"""
function _validation_preview_batch(batch, n_samples::Int)
    ns = max(1, n_samples)
    if batch isa Tuple && length(batch) == 2
        x, x_mask = batch
        bs = size(x, 3)
        k = min(ns, bs)
        return view(x, :, :, 1:k), view(x_mask, :, :, 1:k)
    else
        x = batch
        bs = size(x, 3)
        k = min(ns, bs)
        return view(x, :, :, 1:k)
    end
end


"""
Save a small reconstruction snapshot from the validation set.

Arguments:
- `model`: trained model instance.
- `valid_loader`: validation dataloader.
- `device`: active device transfer function.
- `out_dir`: target directory for serialized snapshots.
- `epoch`: current epoch index.
- `max_epochs`: total number of epochs.
- `n_samples`: number of validation samples to store.

Returns:
- Path to saved file as `String`, or `nothing` when saving is skipped.

Notes:
- The function is best-effort: if reconstruction is not available for a model,
  snapshot creation is skipped with a warning.
"""
function _save_validation_prediction_snapshot(
    model::AbstractGenModel,
    valid_loader::DataLoader,
    device::Function,
    out_dir::String,
    epoch::Int,
    max_epochs::Int,
    n_samples::Int,
)
    iterator = iterate(valid_loader)
    if isnothing(iterator)
        @warn "Validation loader is empty; skipping prediction snapshot."
        return nothing
    end

    raw_batch = first(iterator)
    preview = _validation_preview_batch(raw_batch, n_samples)

    x_cpu = nothing
    xhat_cpu = nothing
    mask_cpu = nothing

    try
        if preview isa Tuple && length(preview) == 2
            x, x_mask = preview
            x_dev = device(x)
            x_mask_dev = device(x_mask)
            xhat = reconstruct(model, x_dev, x_mask_dev)
            x_cpu = cpu(x_dev)
            mask_cpu = cpu(x_mask_dev)
            xhat_cpu = cpu(xhat)
        else
            x = preview
            x_dev = device(x)
            xhat = reconstruct(model, x_dev)
            x_cpu = cpu(x_dev)
            xhat_cpu = cpu(xhat)
        end
    catch err
        @warn "Could not save validation prediction snapshot; skipping." exception=(err, catch_backtrace())
        return nothing
    end

    mkpath(out_dir)
    file = joinpath(out_dir, "predictions_ep=$(lpad_number(epoch, max_epochs)).jls")
    payload = (
        epoch = epoch,
        n_samples = size(x_cpu, 3),
        x = x_cpu,
        xhat = xhat_cpu,
        x_mask = mask_cpu,
    )
    serialize(file, payload)
    return file
end


"""
Train a model using positional argument style for loss and schedulers.

Arguments:
- `model`: model instance implementing `optim_step` and `valid_step`.
- `dataloaders`: named tuple `(train, valid)` with two `DataLoader`s.
- `optimiser`: Optimisers.jl rule (e.g. `Adam(1e-3)`).
- `loss_function`: reconstruction/loss function passed to training steps.
- `β_scheduler`: function/scheduler returning beta value per epoch.
- `lr_scheduler`: optional learning-rate scheduler per epoch.

Keyword Arguments:
- `use_gpu`: whether to use GPU when available.
- `model_dir`: directory where logs/checkpoints are saved.
- `verbose`: verbose validation logging after each epoch.
- `valid_check_interval`: validate every N train iterations.
- `checkpoint_interval_epochs`: save model checkpoint every N epochs.
- `epochs`: maximum training epochs.
- `early_stopping`: enable early stopping.
- `patience`: early stopping patience in iterations.
- `max_train_time`: hard wall-clock limit in seconds.
- `grad_skip`: optional gradient-skipping threshold flag.

Returns:
- `(model=model, opt=opt, history=history)`.
"""
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


"""
Train a model with keyword-oriented API.

Arguments:
- `model`: model instance implementing `optim_step` and `valid_step`.
- `dataloaders`: named tuple `(train, valid)` with two `DataLoader`s.
- `optimiser`: Optimisers.jl rule used to build optimizer state.

Keyword Arguments:
- `loss_function`: training loss function.
- `β_scheduler`: function/scheduler returning beta per epoch.
- `lr_scheduler`: optional learning-rate scheduler per epoch.
- `use_gpu`: whether to train on GPU if available.
- `model_dir`: output directory for logs and checkpoints.
- `verbose`: epoch-level validation verbosity.
- `valid_check_interval`: validation cadence within epoch.
- `checkpoint_interval_epochs`: model checkpoint cadence.
- `epochs`: maximum number of epochs.
- `early_stopping`: enable/disable early stopping.
- `patience`: patience for early stopping.
- `max_train_time`: wall-clock training limit in seconds.
- `grad_skip`: optional gradient-skip control passed to `optim_step`.
- `validation_verbose`: verbosity for in-epoch validation checks.
- `save_val_predictions`: whether to save validation recon snapshots.
- `val_prediction_count`: number of validation samples per snapshot.
- `val_prediction_interval_epochs`: snapshot cadence; defaults to
    `checkpoint_interval_epochs` when `nothing`.
- `val_prediction_dirname`: output subdirectory under `model_dir`.

Returns:
- `(model=model, opt=opt, history=history)`.

Notes:
- On checkpoint epochs, both model state and (optionally) validation preview
    reconstructions are serialized to disk.
"""
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
    save_val_predictions::Bool=true,
    val_prediction_count::Int=16,
    val_prediction_interval_epochs::Union{Int, Nothing}=nothing,
    val_prediction_dirname::String="val_predictions",
    kwargs...
)

    # 1) save start time for checking of time budget
    start_time = time()
    
    # 2a) logging | setup history log
    history = ValueHistories.MVHistory()
    # 2b) initialize jsonl logger
    mkpath(joinpath(model_dir, "models"))
    json_logger = JSONLLogger(joinpath(model_dir, "trainlog.jsonl"))

    # 3) check for GPU / if found model is set to GPU 
    if use_gpu && CUDA.functional() 
        device = cu
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
    idx = 0
    pred_interval = isnothing(val_prediction_interval_epochs) ? checkpoint_interval_epochs : val_prediction_interval_epochs
    val_pred_dir = joinpath(model_dir, val_prediction_dirname)

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

            if save_val_predictions && pred_interval > 0 && (mod(epoch, pred_interval) == 0 || epoch == epochs)
                saved_file = _save_validation_prediction_snapshot(
                    model,
                    dataloaders.valid,
                    device,
                    val_pred_dir,
                    epoch,
                    epochs,
                    val_prediction_count,
                )
                if !isnothing(saved_file)
                    @info "Saved validation prediction snapshot" file=saved_file
                end
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

"""
Run one validation pass, log metrics and apply early-stopping update.

Arguments:
- `model`: model instance.
- `dataloader`: validation dataloader.
- `loss_function`: loss passed to `valid_step`.
- `β`: beta value used for validation step.
- `device`: active device transfer function.
- `history`: metric history container.
- `logger`: JSONL logger.
- `early_stopper`: early-stopping controller.
- `idx`: global iteration index used for logging.

Keyword Arguments:
- `tr_log`: optional train-step logs for verbose side-by-side printing.
- `verbose`: whether to print validation summary.
- `epoch_info`: tuple `(epoch, total_epochs)`.
- `iter_info`: tuple `(iter, max_iters)`.

Returns:
- `Bool`: `true` when training should stop, otherwise `false`.
"""
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
