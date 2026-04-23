using Revise
using DrWatson 
#@quickactivate
using ArgParse
using StatsBase
using Random
using Serialization
using GenerativeMIL
using Flux
using Zygote
using CUDA
#using GenerativeMIL: transform_batch, train_test_split, load_modelnet10
#using GenerativeMIL.Models: check, loss, loss_gpu, unpack_mill
#using MLDataPattern
using MLUtils
#using FileIO #for loading of data and logging
using HDF5, JSON3


s = ArgParseSettings()
@add_arg_table! s begin
    "npoints"
		arg_type = Int
		default = 512
		help = "cardinality of observation"
    "seed"
        arg_type = Int
        help = "seed"
        default = 1
	"random_seed"
		default = 0
		arg_type = Int
		help = "random seed for sample_params function (to be able to train multile seeds in parallel)"
	"time_limit"
		default = 24
		arg_type = Int
		help = "Time (in hours) reserved for training. After exceeding this time model training will be stopped regardless of epoch"
end
parsed_args = parse_args(ARGS, s; as_symbols=true)
ap = NamedTuple{Tuple(keys(parsed_args))}(values(parsed_args))
@info ap
# npoints, seed, random_seed, time_limit = 512, 1, 1, 1


#load data
data = load_modelnet10(ap.npoints, "all", validation=true, seed=ap.seed);

@info "Data loaded -- train -> $(data[1][1]|>size) | test -> $(data[2][1]|>size)"
#define model 
modelname = "setvae"

parameters = (levels = 2, hdim = 32, heads = 4, activation = "relu", prior = "mog", prior_dim = 32, vb_depth = 2, vb_hdim = 32, 
	is_sizes = [32, 16], zdims = [16, 16], batchsize = 16, lr = 0.0003f0, lr_decay = false, 
	beta = 1f0, beta_anealing = 100f0, epochs = 8000, init_seed = rand(1:Int(1e8)))

#parameters = (levels = 7, hdim = 64, heads = 4, activation = "swish", prior = "mog", prior_dim = 32, vb_depth = 2, vb_hdim = 64, 
#	is_sizes = [32, 16, 8, 4, 2, 1, 1], zdims = [16, 16, 16, 16, 16, 16, 16], batchsize = 100, lr = 0.0003f0, lr_decay = false, 
#	beta = 1f0, beta_anealing = 100f0, epochs = 8000, init_seed = rand(1:Int(1e8)))#vb_depth = 1, vb_hdim = 0, lr_decay=WarmupCosine

@info "Parameters sampled: \n $(parameters)"
#debugging
model = GenerativeMIL.setvae_constructor_from_named_tuple( ;idim=size(data[1][1],1), parameters...)
#info = fit!(model, data, loss; max_train_time=Int((time_limit-0.5)*3600), patience=200, check_interval=20, hmil_data=false, parameters...) 




















function fit(data, parameters; time_limit=365*24) #time_limit = one year
	# construct model - constructor should only accept kwargs
	model = GenerativeMIL.setvae_constructor_from_named_tuple( ;idim=size(data[1][1],1), parameters...)

	# fit train data
	# max. train time: 24 hours, over 10 CPU cores -> 2.4 hours of training for each model
	# the full traning time should be 48 hours to ensure all scores are calculated
	# training time is decreased automatically for less cores!
	try 
		global info, fit_t, _, _, _ = @timed fit!(model, data, loss_gpu; max_train_time=Int((time_limit-0.5)*3600), 
				patience=300, check_interval=20, hmil_data=false, parameters...)
	catch e
		# return an empty array if fit fails so nothing is computed
		@info "Failed training due to \n$e"
		return (fit_t = NaN, history=nothing, npars=nothing, model=nothing), [] 
	end

	# construct return information - put e.g. the model structure here for generative models
	training_info = (
		fit_t = fit_t,
		history = info.history,
		npars = info.npars,
		model = info.model,
		nan = info.nan
		)
	
	# now return the info to be saved and an array of tuples (anomaly score function, hyperparatemers)
	return training_info, [
		(x -> GenerativeMIL.Models.transform_and_reconstruct(info.model, x), 
		merge(parameters, (score = "reconstructed_input",)))
	]
	#((x, x_mask) -> GenerativeMIL.Models.reconstruct(info.model, x, x_mask), merge(parameters, (score = "reconstructed_input",)))
end




#training of model
training_info, results = fit(data, parameters, time_limit=time_limit)


savepath = datadir("experiments")
dataset = "ModelNet10-$(npoints)"
_savepath = joinpath(savepath, "$(modelname)/$(dataset)/seed=$(seed)")
mkpath(_savepath)

# save the model separately			
if training_info.model !== nothing
    modelf = joinpath(_savepath, savename("model", parameters, "bson", digits=5))
    tagsave(
        modelf, 
        Dict("model"=>training_info.model,
            "fit_t"=>training_info.fit_t,
            "history"=>training_info.history,
            "parameters"=>parameters,
			"NaN"=>training_info.nan # if loss turn to nan or not
            ), 
        safe = true)
    (@info "Model saved to $modelf")

    training_info = merge(training_info, (model = nothing, history=nothing))
end

# here define what additional info should be saved together with parameters, scores, labels and predict times
save_entries = merge(training_info, (modelname = modelname, seed = seed))


function experiment_evaluation(score_fun, parameters, data, savepath; verb=true, save_result=true, save_entries...)
	#temporary function
	tr_data, val_data, tst_data = data
	# unpack data for easier manipulation
	tr_data, tr_lab = unpack_mill(tr_data)
	val_data, val_lab = unpack_mill(val_data)
	tst_data, tst_lab = unpack_mill(tst_data)

	# create reconstructed bags
	tr_rec = cat(score_fun(tr_data)..., dims=3)
	val_rec= cat(score_fun(val_data)..., dims=3)
	tst_rec= cat(score_fun(tst_data)..., dims=3)

	tr_score = chamfer_distance(tr_data, tr_rec)
	val_score = chamfer_distance(val_data, val_rec)
	tst_score = chamfer_distance(tst_data, tst_rec)

	savef = joinpath(savepath, savename(merge(parameters, (type = "reconstructed_input",)), "bson", digits=5))
	result = (
		parameters = merge(parameters, (type = "reconstructed_input",)),
		loss_train = tr_score,
		loss_valid = val_score, 
		loss_test  = tst_score,
		)
	result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(merge(result, save_entries))]) # this has to be a Dict 
	if save_result
		tagsave(savef, result, safe = true)
		verb ? (@info "Results saved to $savef") : nothing
	end
	return result
end


# now loop over all anomaly score funs
@time for result in results
    experiment_evaluation(result..., data, _savepath; save_entries...)
end
