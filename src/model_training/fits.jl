# StatsBase.fit! was tested (line by line) on cpu
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


function StatsBase.fit!(model::FoldingNet_VAE, data::Tuple, loss::Function; epochs=1000, max_train_time=82800, 
    batchsize=64, lr=0.001, beta=1f0, patience=50, check_interval::Int=20, kwargs...)
    #logging_loss::Union{Function, Nothing}=nothing,
    # purely cpu training

    # setup history log 
    history = ValueHistories.MVHistory()
    # save original model into best model and save orignal patience
    best_model = deepcopy(model)
    patience_ = deepcopy(patience)

    print(model)
    # prepare data for bag model 
    x_train, l_training = unpack_mill(data[1])
    x_val_, l_val = unpack_mill(data[2])
    x_val = x_val_[l_val .== 0]

    # Convert epochs to iterations
    if fld(length(x_train), batchsize) == 0
        max_iters = epochs
        @info "dataset//batchsize == 0 -> max_iters = $(epochs)"
    else
        max_iters = epochs * fld(length(x_train), batchsize) # epochs to iters
        @info "dataset//batchsize > 0 -> max_iters = $(max_iters)"
    end

    # create dataloaders
    dataloader = MLDataPattern.RandomBatches(x_train, size=batchsize, count=max_iters)
    val_dl = Flux.Data.DataLoader(x_val, batchsize=batchsize)

    # prepere early stopping criterion and start time
    best_val_loss = Inf
    start_time = time()

    global final_beta = beta
    opt = ADAM(lr)
    ps = Flux.params(model)

    loss_f(x) = loss(model, x; β=final_beta)

    for (i, batch) in enumerate(dataloader)
        # forward
        loss_, back = Flux.pullback(ps) do 
            Flux.mean(loss_f.(batch))#loss(model, x)
        end;
        # backward
        grad = back(1f0);
        # optimise
        Flux.Optimise.update!(opt, ps, grad);
        # Logging
        push!(history, :training_loss, i, loss_)
        #push!(history, :training_kld_ori_loss, i, loss_[2])
        #push!(history, :training_kld_rec_loss, i, loss_[3])
        push!(history, :beta, i, final_beta)

        # Validation stage
        if mod(i, check_interval) == 0
            # compute validation loss
            total_val_loss = 0
            for batch in val_dl
                v_loss = Flux.mean(loss_f.(batch))
                total_val_loss += v_loss;
            end   
            # compute losses
            total_val_loss /= length(val_dl)
            push!(history, :val_loss, i, total_val_loss)
            @info "$i - training -> loss: $(loss_) || validation -> loss: $(total_val_loss)"
            # check for nans
            if isnan(total_val_loss) || isnan(loss_)
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
    (history = history, iterations = length(get(history, :training_loss)), model = best_model, npars = sum(map(p -> length(p), Flux.params(model))))
end

#Union{PoolModel, VQVAE}
function StatsBase.fit!(model::PoolModel, data::Tuple, loss::Function; epochs=1000, max_train_time=82800, 
    batchsize=64, lr=0.001, patience=50, check_interval::Int=20, hmil_data::Bool=true, kwargs...)
    #TODO fix for masked version
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

    @info "module of model -> $(get_device(model))"
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

    # create dataloaders
    dataloader = MLDataPattern.RandomBatches(x_train, size=batchsize, count=max_iters)
    val_dl = Flux.Data.DataLoader(x_val, batchsize=batchsize)

    # prepere early stopping criterion and start time
    best_val_loss = Inf
    start_time = time()
    nan_ = false

    opt = ADAM(lr)
    ps = Flux.params(model);

    # infinite for loop via RandomBatches / stopping criterion later
    for (i, batch) in enumerate(dataloader)
        # Training stage
        x, x_mask = transform_batch(batch, true)
        x = (to_gpu) ? x|>gpu : x   #TODO check this
        #x_mask = (to_gpu) ? x_mask|>gpu : x_mask

        # forward
        loss_, back = Flux.pullback(ps) do 
            loss(model, x) 
        end;
        # backward only total loss
        grad = back(1f0);
        # optimise
        Flux.Optimise.update!(opt, ps, grad);
        # Logging
        push!(history, :training_loss, i, loss_)
        #push!(history, :training_kld_loss, i, loss_[2])
        #push!(history, :beta, i, beta)

        # Validation stage
        if mod(i, check_interval) == 0
            # compute validation loss
            total_val_loss, total_val_kld = 0, 0
            for batch in val_dl
                xv, xv_mask = transform_batch(batch, true)
                xv = (to_gpu) ? xv|>gpu : xv
                #xv_mask = (to_gpu) ? xv_mask|>gpu : xv_mask
                # compute validation loss
                v_loss = loss(model, xv);
                total_val_loss += v_loss;
            end   
            # compute losses
            total_val_loss /= length(val_dl)
            push!(history, :val_loss, i, total_val_loss)
            @info "$i - training -> loss: $(loss_) || validation -> loss: $(total_val_loss)"
            # check for nans
            if isnan(total_val_loss) || isnan(total_val_kld) || isnan(loss_)
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



function StatsBase.fit!(model::Union{VQVAE,VGQ_PoolAE}, data::Tuple, loss::Function; epochs=1000, max_train_time=82800, 
    batchsize=64, lr=0.001, beta=1f0, patience=50, check_interval::Int=20, ad_data::Bool=false, kwargs...)
    #logging_loss::Union{Function, Nothing}=nothing,
    # purely cpu training

    # setup history log 
    history = ValueHistories.MVHistory()
    # save original model into best model and save orignal patience
    best_model = deepcopy(model)
    patience_ = deepcopy(patience)

    println(model)
    # prepare data for bag model 
    x_train, l_training = unpack_mill(data[1])
    x_val_, l_val = unpack_mill(data[2])
    x_val = nothing
    try 
        x_val = (ad_data) ? x_val_[l_val .== 0] : x_val_ #FIXME if X_val is 3D or 2D tensor it is not working for hmil_data=true
    catch e
        x_val = (ad_data) ? x_val_[:,:,l_val .== 0] : x_val_
        @info "inside try catch \"hmil data\" "
    end

    # Convert epochs to iterations
    if fld(length(x_train), batchsize) == 0
        max_iters = epochs
        @info "dataset//batchsize == 0 -> max_iters = $(epochs)"
    else
        max_iters = epochs * fld(length(x_train), batchsize) # epochs to iters
        @info "dataset//batchsize > 0 -> max_iters = $(max_iters)"
    end

    # create dataloaders
    dataloader = MLDataPattern.RandomBatches(x_train, size=batchsize, count=max_iters)
    val_dl = Flux.Data.DataLoader(x_val, batchsize=batchsize)

    # prepere early stopping criterion and start time
    best_val_loss = Inf
    start_time = time()

    global final_beta = beta
    opt = ADAM(lr)
    ps = Flux.params(model)

    loss_f(x) = loss(model, x; β=final_beta)

    for (i, batch) in enumerate(dataloader)
        # forward
        loss_, back = Flux.pullback(ps) do 
            Flux.mean(loss_f.(batch))#loss(model, x)
        end;
        # backward
        grad = back(1f0);
        # optimise
        Flux.Optimise.update!(opt, ps, grad);
        # Logging
        push!(history, :training_loss, i, loss_)
        #push!(history, :training_kld_ori_loss, i, loss_[2])
        #push!(history, :training_kld_rec_loss, i, loss_[3])
        push!(history, :beta, i, final_beta)

        # Validation stage
        if mod(i, check_interval) == 0
            # compute validation loss
            total_val_loss = 0
            for batch in val_dl
                v_loss = Flux.mean(loss_f.(batch))
                total_val_loss += v_loss;
            end   
            # compute losses
            total_val_loss /= length(val_dl)
            push!(history, :val_loss, i, total_val_loss)
            @info "$i - training -> loss: $(loss_) || validation -> loss: $(total_val_loss)"
            # check for nans
            if isnan(total_val_loss) || isnan(loss_)
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
    (history = history, iterations = length(get(history, :training_loss)), model = best_model, npars = sum(map(p -> length(p), Flux.params(model))))
end


function fit_gpu_ready!(model::FoldingNet_VAE, data::Tuple, loss::Function; epochs=1000, max_train_time=82800, 
    batchsize=64, lr=0.001, beta=1f0, patience=50, check_interval::Int=20, hmil_data::Bool=true, kwargs...)
    
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

    @info "module of model -> $(get_device(model))"
    #print(model)
    
    x_train, l_training = unpack_mill(data[1])
    x_val_, l_val = unpack_mill(data[2])
    x_val = nothing
    try # FIXME
        x_val = (hmil_data) ? x_val_[l_val .== 0] : x_val_ #FIXME if X_val 2D tensor it is not working for hmil_data=true
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

    # since training is unsupervised we will use knn neighbors as targed in dataloader
    # it will speed up training signicifantly because knn is the most expencive part of training
    @info "fitting knn"
    @time x_train_knn = knn(x_train, model.encoder.n_neighbors)
    @time x_val_knn = knn(x_val, model.encoder.n_neighbors)
    @info "knn fitted"

    # create dataloaders
    dataloader = MLDataPattern.RandomBatches((x_train, x_train_knn), size=batchsize, count=max_iters);
    val_dl = Flux.Data.DataLoader((x_val,x_val_knn), batchsize=batchsize);

    # prepere early stopping criterion and start time
    best_val_loss = Inf
    start_time = time()
    nan_ = false

    loss_f(x::AbstractArray{<:AbstractArray}) = Flux.mean(map(y->loss(model, y; β=final_beta, γ=final_beta),x)) # cpu loss ... list of 2D matrices
    loss_f(x::AbstractArray{<:Real}, x_mask::Nothing=nothing) = loss(model, x; β=final_beta, γ=final_beta) # cpu loss 2D and 3D tensors 
    loss_f(x::AbstractArray{<:Real}, kidx::AbstractArray{<:Real}) = loss(model, x, kidx; β=final_beta, γ=final_beta) # cpu loss 2D and 3D tensors 

    transf(x::AbstractArray{<:AbstractArray}) = (x, nothing)
    transf(x::AbstractArray{<:Real}) = transform_batch(x, true)
    transf(x::Tuple) = (transform_batch(x[1], true)[1], transform_batch(x[2], true)[1])

    @info "input beta=>$(beta)"
    global final_beta = beta
    opt = ADAM(lr)
    ps = Flux.params(model)

    # infinite for loop via RandomBatches / stopping criterion later
    for (i, batch) in enumerate(dataloader)
        # Training stage
        x, kidx = transf(batch)
        x = (to_gpu) ? x|>gpu : x   #TODO check this
        kidx = (to_gpu) ? kidx|>gpu : kidx  
        #x = (to_gpu) ? batch|>gpu : batch  
        #x_mask = (to_gpu) ? x_mask|>gpu : x_mask

        # forward
        loss_, back = Flux.pullback(ps) do 
            loss_f(x, kidx)
         end;
        # backward only total loss
        grad = back(1f0);
        # optimise
        Flux.Optimise.update!(opt, ps, grad);
        # Logging
        push!(history, :training_loss, i, loss_)
        push!(history, :beta, i, final_beta)

        # Validation stage
        if mod(i, check_interval) == 0
            # compute validation loss
            total_val_loss, total_val_kld = 0, 0
            for batch in val_dl
                xv, kidx = transf(batch)
                xv = (to_gpu) ? xv|>gpu : xv
                kidx = (to_gpu) ? kidx|>gpu : kidx
                #xv_mask = (to_gpu) ? xv_mask|>gpu : xv_mask
                # compute validation loss
                v_loss = loss_f(xv, kidx);
                total_val_loss += v_loss;
            end   
            # compute losses
            total_val_loss /= length(val_dl)
            push!(history, :val_loss, i, total_val_loss)
            @info "$i - training -> loss: $(loss_) || validation -> loss: $(total_val_loss)"
            # check for nans
            if isnan(total_val_loss) || isnan(total_val_kld) || isnan(loss_)
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