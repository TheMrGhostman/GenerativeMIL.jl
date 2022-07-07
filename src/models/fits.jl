# StatsBase.fit! was tested (line by line) on cpu
function StatsBase.fit!(model::SetVAE, data::Tuple, loss::Function; epochs=1000, max_train_time=82800, 
    batchsize=64, lr=0.001, lr_decay=false, beta=0.01, beta_anealing=0, patience=50,
	check_interval::Int=20, kwargs...)

    # setup history log 
    history = MVHistory()
    # save original model into best model and save orignal patience
    best_model = deepcopy(model)
    patience_ = deepcopy(patience)

    # Change device to gpu if gpu found
    global const_module = Base
    global to_gpu = false

    try  
        if CUDA.devices() |> length > 0
            const_module = CUDA
            to_gpu = true
            model = model |> gpu
        end
    catch
        @info "No GPU found"
    end

    # prepare data for bag model 
    x_train, l_training = GroupAD.Models.unpack_mill(data[1])
    x_val_, l_val = GroupAD.Models.unpack_mill(data[2])
    x_val = x_val_[l_val .== 0]

    # Convert epochs to iterations
    max_iters = epochs * fld(length(x_train), batchsize) # epochs to iters

    # create dataloaders
    dataloader = MLDataPattern.RandomBatches(x_train, size=batchsize, count=max_iters)
    val_dl = Flux.Data.DataLoader(x_val, batchsize=batchsize)

    # prepere early stopping criterion and start time
    best_val_loss = Inf
    start_time = time()

    global final_beta = beta
    global anealing = beta_anealing * fld(length(x_train), batchsize)
    opt = ADAM(lr)
    ps = Flux.params(model)

    # infinite for loop via RandomBatches / stopping criterion later
    for (i, batch) in enumerate(dataloader)
        # Training stage
        beta = final_beta * min(1f0, i/anealing)
        x, x_mask = GenerativeMIL.transform_batch(batch,true)
        x = (to_gpu) ? x|>gpu : x
        x_mask = (to_gpu) ? x_mask|>gpu : x_mask
 
        # forward
        loss_, back = Flux.pullback(ps) do 
            loss(model, x, x_mask, beta, const_module=const_module) 
        end;
        # backward only total loss
        grad = back((1f0,0f0));
        Flux.Optimise.update!(opt, ps, grad);
        # Logging
        push!(history, :training_loss, i, loss_[1])
        push!(history, :training_kld_loss, i, loss_[2])
        push!(history, :beta, i, beta)

        # Validation stage
        if mod(i, check_interval) == 0
            # compute validation loss
            total_val_loss, total_val_kld = 0, 0
            for batch in val_dl
                xv, xv_mask = GenerativeMIL.transform_batch(batch, true)
                xv = (to_gpu) ? xv|>gpu : xv
                xv_mask = (to_gpu) ? xv_mask|>gpu : xv_mask
                # compute validation loss
                v_loss, v_kld = loss(model, xv, xv_mask, beta, const_module=const_module);
                total_val_loss += v_loss;
                total_val_kld += v_kld;
            end   
            total_val_loss /= length(val_dl)
            total_val_kld /= length(val_dl)
            push!(history, :val_loss, i, total_val_loss)
            push!(history, :val_kld_loss, i, total_val_kld)
            @info "$i - training -> loss: $(loss_[1]) | kld: $(loss_[2]) || validation -> loss: $(total_val_loss) | kld: $(total_val_kld)"
            # check for nans
            if isnan(total_val_loss) || isnan(total_val_kld) || isnan(loss_[1]) || isnan(loss_[2])
				error("Encountered invalid values in loss function.")
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
    (history = history, iterations = i, model = best_model, npars = sum(map(p -> length(p), Flux.params(model))))
end