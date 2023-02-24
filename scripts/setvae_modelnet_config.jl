using DrWatson
@quickactivate
using ArgParse
using StatsBase
using BSON
using Random
using ValueHistories
#generative MIL
using GenerativeMIL
using Flux
using Flux3D
using Zygote
using CUDA
using GenerativeMIL: transform_batch, train_test_split, load_modelnet10, simple_experiment_evaluation
using GenerativeMIL.Models: check, loss, loss_gpu, unpack_mill
using MLDataPattern

#using FileIO #for loading of data
using HDF5
using YAML
using OrderedCollections
dict2nt(x) = (; (Symbol(k) => v for (k,v) in x)...)

s = ArgParseSettings()
@add_arg_table! s begin
    "config_file"
		default = "configs/default.yml"
		arg_type = String
		help = "path and name to config file for model's hyperparameters"
    "npoints"
		arg_type = Int
		default = 512
		help = "cardinality of observation"
    "seed"
        arg_type = Int
        help = "seed"
        default = 1
    "time_limit"
		default = 24
		arg_type = Int
		help = "Time (in hours) reserved for training. After exceeding this time model training will be stopped regardless of epoch"
end
parsed_args = parse_args(ARGS, s)
@unpack config_file, npoints, seed, time_limit = parsed_args

# parse hyperparameters
hyper = YAML.load_file(config_file, dicttype=OrderedDict);
parameters = dict2nt(hyper);
@info parameters

# load data
dataset = "ModelNet10-$(npoints)"
data = load_modelnet10(npoints, "all", validation=true, seed=seed);

# construct model
modelname = "setvae"
model = GenerativeMIL.Models.setvae_constructor_from_named_tuple( ;idim=size(data[1][1],1), parameters...);

# train model
#try 
global info, fit_t, _, _, _ = @timed fit!(model, data, loss_gpu; max_train_time=Int((time_limit-0.5)*3600), 
            patience=parameters[:patience], check_interval=parameters[:check_every], hmil_data=false, parameters...);
    try
        println("try")
    catch e
    # return an empty array if fit fails so nothing is computed
    @info "Failed training due to \n$e"
    return (fit_t = NaN, history=nothing, npars=nothing, model=nothing), [] 
end

#store results
training_info = (fit_t = fit_t, history = info.history, npars = info.npars, model = info.model, nan = info.nan);


# savepath
_savepath = datadir("experiments", "$(modelname)/$(dataset)/seed=$(seed)"); mkpath(_savepath)

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
end;


# EVALUTATION
eval_results = [(x -> GenerativeMIL.Models.transform_and_reconstruct(info.model, x), merge(parameters, (score = "r_input",))),];

# here define what additional info should be saved together with parameters, scores, labels and predict times
save_entries = merge(training_info, (modelname = modelname, seed = seed))

# now loop over all anomaly score funs
@time for result in eval_results
    simple_experiment_evaluation(result..., data, _savepath; save_entries...);
end
