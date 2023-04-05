using DrWatson
@quickactivate
using ArgParse
using StatsBase
using Flux
using Flux3D
#generative MIL
using GenerativeMIL
using Zygote
using GenerativeMIL.Models: check, loss, unpack_mill, loss_gradient, loss_ema, vq_poolae_constructor_from_named_tuple
using GenerativeMIL: load_and_standardize_mnist, train_test_split

using ValueHistories, MLDataPattern, BSON
using YAML
using OrderedCollections
dict2nt(x) = (; (Symbol(k) => v for (k,v) in x)...)

s = ArgParseSettings()
@add_arg_table! s begin
    "config_file"
		default = "configs/vqvae/default.yml"
		arg_type = String
		help = "path and name to config file for model's hyperparameters"
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
@unpack config_file, seed, time_limit = parsed_args

# parse hyperparameters
hyper = YAML.load_file(config_file, dicttype=OrderedDict);
parameters = dict2nt(hyper);
@info parameters

# load data
dataset = "MNIST-all"

train, test = load_and_standardize_mnist();
train, val = train_test_split(train[1], train[2], 0.2; seed=seed);

# construct model
modelname = "S-VQ_PoolAE"
model = vq_poolae_constructor_from_named_tuple(;idim=3, parameters...);

# loss function
loss_f = (parameters[:ema]) ? loss_ema : loss_gradient


# training
try
	global info, fit_t, _, _, _ = @timed fit!(model, (train, val, test), loss_f; max_train_time=Int((time_limit-0.5)*3600), 
            patience=parameters[:patience], check_every=parameters[:check_every], 
			ad_data=true, parameters...);
catch e
    # return an empty array if fit fails so nothing is computed
    @info "Failed training due to \n$e"
    return (fit_t = NaN, history=nothing, npars=nothing, model=nothing), [] 
end

#store results
training_info = (fit_t = fit_t, history = info.history, npars = info.npars, model = info.model);


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
            "parameters"=>parameters
            ), 
        safe = true)
    (@info "Model saved to $modelf")

    training_info = merge(training_info, (model = nothing,))
end;


# savepath
_savepath = datadir("experiments", "$(modelname)/not_optimal/$(dataset)/seed=$(seed)"); mkpath(_savepath)

# save the model separately			
if model !== nothing
    modelf = joinpath(_savepath, savename("model", parameters, "bson", digits=5))
    tagsave(
        modelf, 
        Dict("model"=>model,
            "fit_t"=>training_info.fit_t,
            "history"=>training_info.history,
            "parameters"=>parameters
            ), 
        safe = true)
    (@info "Model saved to $modelf")

    training_info = merge(training_info, (model = nothing,))
end;
