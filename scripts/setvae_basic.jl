using DrWatson
@quickactivate
using ArgParse
using GroupAD
import StatsBase: fit!, predict
using StatsBase
using BSON
#generative MIL
using GenerativeMIL
using Flux
using Zygote
using CUDA
using GenerativeMIL: transform_batch
using GenerativeMIL.Models: check, loss
using MLDataPattern

s = ArgParseSettings()
@add_arg_table! s begin
   "max_seed"
        arg_type = Int
        help = "seed"
        default = 1
    "dataset"
        default = "MNIST"
        arg_type = String
        help = "dataset"
	"anomaly_classes"
		arg_type = Int
		default = 10
		help = "number of anomaly classes"
	"method"
		default = "leave-one-out"
		arg_type = String
		help = "method for data creation -> \"leave-one-out\" or \"leave-one-in\" "
   "contamination"
        default = 0.0
        arg_type = Float64
        help = "training data contamination rate"
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, anomaly_classes, method, contamination = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "setvae_basic"
# sample parameters, should return a Dict of model kwargs 
"""
	sample_params()

Should return a named tuple that contains a sample of model parameters.
"""
function sample_params()
	# MNIST has idim = 3 -> fewer possibilities for sampling
	# +/- 4608 possible combinations, some of sample supports are just placehodlers for future options
	default(x) = 16*ones(Int, size(x)...)
	model_par_vec = (
		2:5, 				# :levels -> number of "sampling skip-connections"
		2 .^(4:6), 			# :hdim -> number of neurons in All dense layers except VariationalBottleneck layers
		[4],				# :heads -> number of heads in multihead attentions
		["relu", "swish"], 	# :activation -> type activation functions in model (mainly for outout from MAB)
		["gaussian", "mog"],# :prior -> prior distribution for decoder, mog has defaultly 4 mixtures
		2 .^(4:5), 			# :prior_dim -> dimension of prior distributino ("noise") 
		[1], 				# :vb_depth -> nlayers in VariationalBottleneck
		[0], 				# :vb_hdim -> hidden dimension in VariationalBottleneck, for :vb_depth=1 is not used
	)
	induced_set_pars = ( 
		[2 .^(5:-1:1)], 	# :is_sizes -> induced sets sides in top-down encoder 				
		[reverse, default] 	# :zdims	-> latent dimension at each level ("skip-connection")
	)
	training_par_vec = (
		2 .^ (6:7), 		# :batchsize -> size of one training batch
		10f0 .^(-4:-3),		# :lr -> learning rate
		[false],			# :lr_decay -> boolean value if to use learning rate decay after half of epochs. 
		10f0 .^ (-3:-1),	# :beta -> final β scaling factor for KL divergence
		[0, 50], 			# :beta_anealing -> number of anealing epochs!!, if 0 then NO anealing
		[200], 				# :epochs -> n of iid iterations (depends on bs and datasize) proportional to n of :epochs 
		1:Int(1e8), 		# :init_seed -> init seed for random samling for experiment instace 
	);
	model_argnames = ( :levels, :hdim, :heads, :activation, :prior, :prior_dim, :vb_depth, :vb_hdim)
	training_argnames = ( :batchsize, :lr, :lr_decay, :beta, :beta_anealing, :epochs, :init_seed )

	model_params = (;zip(model_argnames, map(x->sample(x, 1)[1], model_par_vec))...)
	training_params = (;zip(training_argnames, map(x->sample(x, 1)[1], training_par_vec))...)
	# IS params
	levels = model_params[:levels];
	is_sizes = sample(induced_set_pars[1])[1:levels]
	zdims = sample(induced_set_pars[2])(is_sizes) 

	return merge(model_params,(is_sizes=is_sizes, zdims=zdims), training_params)
end


"""
	fit(data, parameters)
This is the most important function - returns `training_info` and a tuple or a vector of tuples `(score_fun, final_parameters)`.
`training_info` contains additional information on the training process that should be saved, the same for all anomaly score functions.
Each element of the return vector contains a specific anomaly score function - there can be multiple for each trained model.
Final parameters is a named tuple of names and parameter values that are used for creation of the savefile name.
"""
function fit(data, parameters)
	# construct model - constructor should only accept kwargs
	model = GenerativeMIL.Models.setvae_constructor_from_named_tuple(
		;idim=size(data[1][1],1), parameters...
	)

	# fit train data
	# max. train time: 24 hours, over 10 CPU cores -> 2.4 hours of training for each model
	# the full traning time should be 48 hours to ensure all scores are calculated
	# training time is decreased automatically for less cores!
	try
		# number of available cores
		cores = Threads.nthreads()
		global info, fit_t, _, _, _ = @timed fit!(model, data, loss; max_train_time=24*3600*cores/max_seed/anomaly_classes, 
			patience=200, check_interval=5, parameters...)
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
		model = info.model
		)

	# now return the info to be saved and an array of tuples (anomaly score function, hyperparatemers)
	return training_info, [
		(x -> GroupAD.Models.reconstruct(info.model, x),
			merge(parameters, (score = "reconstructed_input",)))
	]
end

"""
	edit_params(data, parameters)
	
This modifies parameters according to data. Default version only returns the input arg. 
Overload for models where this is needed.
"""
function edit_params(data, parameters, class, method)
	merge(parameters, (method = method, class = class, ))
end
