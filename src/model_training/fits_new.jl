function log_to_file(logs::NamedTuple, model_dir::String)
    file_path = joinpath(model_dir, "trainlog.txt")

    if !isfile(file_path)
        open(file_path, "w") do io
            println(io, join(keys(logs), "\t"))
            println(io, join(values(logs), "\t"))
        end
        println("Creating Training Log")
    else
        open(file_path, "a") do IO
            println(io, join(values(logs), "\t"))
        end
    end
end

function validation_check(
    model, 
    dataloader::DataLoader,  
    βs::Union{Vector, Real, Nothing}, 
    loss_function::Function, 
    best_vloss::T=0; 
    device::Function=cpu, 
    save_best::Bool=false, 
    model_dir::String=""
) where T <: Real

    # I assume that valid step just one for every method  .... valid_step::Function,
    # TODO if we consider valid_step to exist, this will be done

    vloss, vlogs = valid_step(model, dataloader, βs, loss_function; device = device)
    if (vloss < best_vloss) & (save_best)
        best_vloss = vloss
        serialize(
            joinpath(model_dir, "models", "best_model.jls"),
            (:model = model |> cpu, :epoch = epoch, :iter = iter, :βs = βs)
        )
    end
    vlogs, best_vloss
end

pad_epoch(ep, epochs) = lpad(string(ep), length(string(epochs)), "0")


function train_model!(
    model, 
    dataloaders::NamedTuple{(:train, :valid), <:Tuple{DataLoader, DataLoader}},
    optim_step::Function, 
    state_tree::NamedTuple, 
    βs::Function, 
    loss_function::Function=chamfer_distance; # here starts kwargs
    valid_step::Function=valid_step,
    use_gpu::Bool=true,
    model_dir::String="", 
    valid_check_interval::Int = 1000,
    checkpoint_interval_epochs::Int=10,
    save_best::Bool=true,
    epochs::Int=1000, 
    max_train_time::Int=82800, # TODO incorporate me
    grad_skip::Union{Real, Bool}=false
)

    history = ValueHistories.MVHistory()
    best_vloss = Inf

    device = use_gpu ? gpu : cpu
    max_iters = length(dataloaders.train)

    for epoch in 1:epochs
        for (it, batch) in tqdm(enumerate(dataloaders.train))
            # perform one optimization step
            β = βs(epoch)
            model, state_tree, logs = optim_step(model, batch |> device, state_tree, β, loss_function; ∇skip=grad_skip);
            # logs will contaion betas, and is named tuple
            # Logging and checking
            ## Logging to value histores
            foreach(p->push!(history, p.first, p.second), pairs(logs))
            ## do validation if time
            if mod(it, valid_check_interval) == 0
                vlogs, best_vloss = validation_check(
                    model, dataloaders.valid, β, loss_function, best_vloss; 
                    device=device, save_best=save_best, model_dir=model_dir
                )
                foreach(p->push!(history, p.first, p.second), pairs(vlogs))
                #TODO fix log_to_file for val loss. 
            else 
                ## logging to txt file
                log_to_file(logs, model_dir)
            end
        end 
        if mod(epoch, checkpoint_interval_epochs) == 0
            serialize(
                joinpath(model_dir, "models", "model_$(pad_epoch(epoch, epochs))ep_$(max_iters)iter.jls"), 
                (:model = model |> cpu, :epoch = epoch, :iter = max_iters, :βs = β)
            )
        end
        # TODO add validation check after epoch
    end

    # TODO put exception to NaN and Infs
    return model, history
end






function StatsBase.fit!(model::SetVAE, data::Tuple, loss::Function; epochs=1000, max_train_time=82800, 
    batchsize=64, lr=0.001, lr_decay=false, beta=0.01, beta_anealing=0, patience=50,
	check_interval::Int=20, hmil_data::Bool=true, kwargs...)

    # setup history log 
    history = ValueHistories.MVHistory()
    # save original model into best model and save orignal patience
    best_model = deepcopy(model)
    patience_ = deepcopy(patience)

    # Change device to gpu if gpu found
    global to_gpu = false

    try  
        if CUDA.devices() |> length > 0
            to_gpu = true
            model = model |> gpu
        end
        @info "GPU go brrrrr"
    catch
        @info "No GPU found"
    end

    print("module of model -> ", get_device(model))
    #print(model)
    
    x_train, l_training = unpack_mill(data[1])
    x_val_, l_val = unpack_mill(data[2])
    x_val = nothing
    try # FIXME
        x_val = (hmil_data) ? x_val_[l_val .== 0] : x_val_ #FIXME if X_val is 3D or 2D tensor it is not working for hmil_data=true
    catch e
        x_val = (hmil_data) ? x_val_[:,:,l_val .== 0] : x_val_
        @info "inside try catch \"hmil data\" "
    end
    @info "zeros in val set => l_val=0 : $(sum(l_val.==0)) | l_val=1 : $(sum(l_val.==1)) | x_val -> $(size(x_val))"
    # Convert epochs to iterations
    if fld(length(x_train), batchsize) == 0
        max_iters = epochs
        @info "dataset//batchsize == 0 -> max_iters = $(epochs)"
    else
        N = (typeof(x_train)<:AbstractArray{<:Real, 3}) ? size(x_train,3) : length(x_train)
        max_iters = epochs * fld(N, batchsize) # epochs to iters // length(x_train)
        @info "dataset//batchsize > 0 -> max_iters = $(max_iters)"
    end
    # Prepare schedulers
    @assert lr_decay in [true, false, "true", "false", "WarmupCosine"]
    if lr_decay == "WarmupCosine"
        scheduler = WarmupCosine(1e-7, lr*5, lr, Int(0.02 * max_iters), Int(0.8 * max_iters)) #FIXME 0.05 , lr*10
        # from 0 to 5% iters there is linear increase of learing rate
        # from 5% to 80% there is cosine decay of learing rate 
        # from 80% to 100% iters there is constant learing rate 
    elseif (lr_decay == "true") || (lr_decay == true)
        scheduler = it -> lr .* min.(1.0, 2.0 - it/(0.5*max_iters))
        #lr .* min.(1.0, map(x -> 2.0 - x/(0.5*max_iters), 1:max_iters)) 
        # learning rate decay (0%,50%) -> 1 , (50%, 100%) -> linear(1->0)
    else
        # just constant learing rate
        scheduler = x -> lr
    end

    # create dataloaders
    dataloader = MLDataPattern.RandomBatches(x_train, size=batchsize, count=max_iters)
    val_dl = Flux.Data.DataLoader(x_val, batchsize=batchsize)

    # prepere early stopping criterion and start time
    best_val_loss = Inf
    start_time = time()
    nan_ = false

    global final_beta = Float32(beta)#
    global anealing = Float32(beta_anealing) * fld(length(x_train), batchsize)
    opt = ADAM(lr)
    ps = Flux.params(model)

    # infinite for loop via RandomBatches / stopping criterion later
    for (i, batch) in enumerate(dataloader)
        # Training stage
        beta = final_beta * min(1f0, i/anealing)
        x, x_mask = transform_batch(batch, true)
        x = (to_gpu) ? x|>gpu : x
        x_mask = (to_gpu) ? x_mask|>gpu : x_mask
 
        # forward
        loss_, back = Flux.pullback(ps) do 
            loss(model, x, x_mask, beta) 
        end;
        # backward only total loss
        grad = back((1f0,0f0));
        # get lr from scheduler
        opt.eta = scheduler(i)
        # optimise
        Flux.Optimise.update!(opt, ps, grad);
        # Logging
        push!(history, :lr, i, Float32(scheduler(i)))
        push!(history, :training_loss, i, loss_[1])
        push!(history, :training_kld_loss, i, loss_[2])
        push!(history, :beta, i, beta)

        # Validation stage
        if mod(i, check_interval) == 0
            # compute validation loss
            total_val_loss, total_val_kld = 0, 0
            for batch in val_dl
                xv, xv_mask = transform_batch(batch, true)
                xv = (to_gpu) ? xv|>gpu : xv
                xv_mask = (to_gpu) ? xv_mask|>gpu : xv_mask
                # compute validation loss
                v_loss, v_kld = loss(model, xv, xv_mask, beta);
                total_val_loss += v_loss;
                total_val_kld += v_kld;
            end   
            # compute losses
            total_val_loss /= length(val_dl)
            total_val_kld /= length(val_dl)
            push!(history, :val_loss, i, total_val_loss)
            push!(history, :val_kld_loss, i, total_val_kld)
            @info "$i - training -> loss: $(loss_[1]) | kld: $(loss_[2]) || validation -> loss: $(total_val_loss) | kld: $(total_val_kld)"
            # check for nans
            if isnan(total_val_loss) || isnan(total_val_kld) || isnan(loss_[1]) || isnan(loss_[2])
                @warn "Encountered invalid values in loss function."
                nan_ = true
				break
                #error("Encountered invalid values in loss function.")
			end
            # Early stopping 
            if total_val_loss < best_val_loss
				best_val_loss = total_val_loss
				patience_ = deepcopy(patience)
                best_model = deepcopy(model)
            else # else stop if the model has not improved for `patience` iterations
				patience_ -= 1
				# @info "Patience is: $_patience."
				if patience_ == 0
					@info "Stopped training after $(i) iterations."
					break
				end
			end
        end
        # Time constrain and iteratins constrain
        if (time() - start_time > max_train_time) | (i > max_iters) # stop early if time is running out
            best_model = deepcopy(model)
            @info "Stopped training after $(i) iterations, $((time() - start_time) / 3600) hours."
            break
        end
    end
    best_model = best_model |> cpu # copy model back to cpu
    (history = history, iterations = length(get(history, :training_loss)), model = best_model, npars = sum(map(p -> length(p), Flux.params(model))), nan = nan_)
end
