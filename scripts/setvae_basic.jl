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
	# MNIST has idim = 2 -> fewer possibilities for sampling
	# zdim: 1, 2, 4, 8, 12, 16
	# hdim: 4, 8, 16, 32, 64
	par_vec = ([1, 2, 4, 8, 12, 16], 2 .^(2:6),  ["scalar", "diagonal"], 10f0 .^(-4:-3), 2 .^ (5:7), ["relu", "swish", "tanh"], 3:4, 1:Int(1e8),
		["mean", "maximum", "median"])
	argnames = (:zdim, :hdim, :var, :lr, :batchsize, :activation, :nlayers, :init_seed, :aggregation)
	parameters = (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
	# ensure that zdim < hdim
	while parameters.zdim >= parameters.hdim
		parameters = merge(parameters, (zdim = sample(par_vec[1])[1],))
	end
	return parameters
end