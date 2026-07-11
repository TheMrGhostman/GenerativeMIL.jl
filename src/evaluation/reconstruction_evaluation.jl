

function _namedtuple_without_key(nt::NamedTuple, key::Symbol)
    if key ∉ keys(nt)
        return nt
    end
    return NamedTuple{Tuple(filter(k -> k != key, keys(nt)))}(Tuple(values(nt)[i] for (i, k) in enumerate(keys(nt)) if k != key))
end


function reconstruction_eval(
    model,
    dataloader::DataLoader,
    loss_functions::Dict{Symbol, Function};
    β=0f0,
    device::Function=cpu,
    return_reconstructions::Bool=false,
    idx::Union{Int, Nothing}=nothing,
    verbose::Bool=false,
    kwargs...
)
    isempty(loss_functions) && error("loss_functions cannot be empty.")
    loss_fns = collect(pairs(loss_functions))
    loss_names = collect(keys(loss_functions))
    losses_per_sample = Dict{Symbol, Array}()
    logs_per_batch = NamedTuple[]
    xs = Array[]
    xhats = Array[]
    x_masks = Any[]
    # pre allocate losses_per_samples?

    iterator_ = verbose ? tqdm(dataloader) : dataloader

    batch_idx = 0
    for batch in iterator_
        batch_idx += 1
        if batch isa Tuple && length(batch) == 2
            x, x_mask = batch
            x_dev = device(Array(x))
            x_mask_dev = device(Array(x_mask))
        else
            x = batch
            x_dev = device(Array(x))
            x_mask_dev = nothing
        end
        # loss_function is set to dummy loss so it is not slowing down the reconstruction evaluation
        xhat_batch, _, base_logs = reconstruct_and_log(model, x_dev, x_mask_dev, (x,y)->0; β=β)

        base_logs_ = _namedtuple_without_key(base_logs, :ℒ) # we do not need total loss as we are pass dummy reconstruction loss function
        base_logs_ = _namedtuple_without_key(base_logs_, :ℒ_rec) # the same reason as for ℒ
        base_logs_ = _namedtuple_without_key(base_logs_, :β) # β will be the same for all batches, so no need to log it
        
        push!(logs_per_batch, merge(base_logs_, (; batch_idx=batch_idx, eval_idx=idx)))

        push!(xs, cpu(x_dev))
        push!(xhats, cpu(xhat_batch))
        (x_mask_dev !== nothing) || push!(x_masks, x_mask_dev)

        batch_loss_vectors = Pair{Symbol, Vector}[]
        for (name, fn) in loss_fns
            lvals = fn(xhat_batch, x_dev, x_mask_dev)
            push!(batch_loss_vectors, name => lvals)
            if haskey(losses_per_sample, name)
                append!(losses_per_sample[name], lvals)
            else
                losses_per_sample[name] = lvals
            end
        end
    end

     # concatenate list of batches in xhats into one tensor along 3rd dim
    if !isempty(xhats) && length(first(dataloader)) != 2 # only concatenate if unmasked, otherwise we keep the batch structure for logging
        xhats_ = cat(xhats, dims=3);
        xs_ = cat(xs, dims=3)
    else
        xhats_ = xhats
        xs_ = xs
    end

    result = (
        n_batches=length(logs_per_batch),
        n_samples=length(losses_per_sample[loss_names[1]]),
        loss_names=loss_names,
        losses=losses_per_sample,
        logs=logs_per_batch
    )

    if return_reconstructions
        return merge(result, (; xhat=xhats, x=xs, x_mask=x_masks))
    end

    return result
end

function _merge_reconstruction_eval_runs(runs::Vector{NamedTuple})
    isempty(runs) && error("No runs provided for merge.")

    n_samples = runs[1].n_samples
    loss_names = runs[1].loss_names
    for (run_idx, run) in enumerate(runs)
        run.n_samples == n_samples || error("Run $(run_idx) has different sample count: $(run.n_samples) vs $(n_samples).")
        run.loss_names == loss_names || error("Run $(run_idx) has different loss names/order.")
    end

    merged_losses = Dict{Symbol, Matrix{Float32}}()
    for name in loss_names
        per_run_losses = [run.losses[name] for run in runs]
        merged_losses[name] = permutedims(reduce(hcat, per_run_losses))
    end

    return (
        n_runs=length(runs),
        n_samples=n_samples,
        loss_names=loss_names,
        losses=merged_losses,
    )
end

function reconstruction_eval_repeated(
    model,
    dataloader::DataLoader,
    loss_functions::Dict{Symbol, Function},
    n_runs::Int;
    kwargs...
)
    n_runs > 0 || error("n_runs must be > 0.")
    runs = NamedTuple[]
    for i in 1:n_runs
        push!(runs, reconstruction_eval(model, dataloader, loss_functions; idx = i, kwargs...))
    end

    merged = _merge_reconstruction_eval_runs(runs)
    return merged
end
