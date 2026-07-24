# Evaluation metafunction ... i just call it function similar to train, you proved metrics list of paths to models you need to evaluate and it should do all dirty work



# interate over all models in a directory and evaluate them ... done
# load all models and evaluate them ... done
# evaluate a single model ... done
# add labels to evaluations so i can evaluate class specific metrics
    # redo dataloader to return labels as well, and then pass them to the evaluation function



function evaluate_reconstructions(paths::Vector{String}, dataloader::DataLoader, loss_functions::Dict{Symbol, Function}; find_best::Bool=false, verbose::Bool=true, kwargs...)
    init_data_cofig = deserialize(joinpath(first(paths), "run_config.jls"))[:data_cfg]
    config_file_init = (ratio=get(init_data_cofig, :ratio, 0.2), seed=get(init_data_cofig, :seed, 1)) # initial config file, just confirm if all models were trained on the same seed and ratio of train/valid/test. we need it to be sure that test data was not seen by any model and that we are feeding correct dataloader so results are obtained from the same valid and test set
    
    errors = [[],[]]
    verbose_list = [[],[]]
    runs = NamedTuple[]
    for path in paths
        # TODO add try except statement
        try
            df = deserialize(joinpath(path, "run_config.jls"))
            data_config = df[:data_cfg]
            @assert get(data_config, :ratio, "notspecified") == config_file_init.ratio && get(data_config, :seed, "notspecified") == config_file_init[:seed] "data_cfg of $(path) is different from init_data_config ($(first(paths)))!!"

            # find  
            model_path = joinpath(path, "models")    
            @assert isdir(model_path) "Models forlder for $(path) does not exist!!"
            model_checkpoints = readdir(model_path);
            max_epoch_model = maximum(model_checkpoints) # this is ugly ugly thing but works! # REVIEW: review and redo

            model_path = find_best ? joinpath(model_path, "best_model.jls") : joinpath(model_path,max_epoch_model) 
            
            verbose && push!(verbose_list[1], path)
            verbose && push!(verbose_list[2], max_epoch_model)

            @assert isfile(model_path)
            #df = deserialize(model_path)
            #df = df.model
            # REDO: 
            #model = df.model
            # TODO: add labels from dataloader 
            #run = reconstruction_eval(model, dataloader, loss_functions; kwargs...)
            #push!(runs, run)
        catch e 
            push!(errors[1], e)
            push!(errors[2], path)
        end
    end
    #return _merge_reconstruction_eval_runs(runs)
    return errors, verbose_list
end


pth_to_model = "/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/mnist/poolmodel/seed=1/"
model_paths = readdir(pth_to_model, join=true) 
model_path = first(model_paths)

run_config = deserialize(joinpath(model_path, "run_config.jls"))
keys(run_config)
run_config[:data_cfg]

o = evaluate_reconstructions(readdir("/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/mnist/poolmodel/seed=1/", join=true), DataLoader(randn(3,3)), Dict{Symbol, Function}(:a=>identity), find_best=false, verbose=true);

o[1] # errors
o[2] # verbose list

o = evaluate_reconstructions(readdir("/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/mnist/poolmodel/seed=1/", join=true), DataLoader(randn(3,3)), Dict{Symbol, Function}(:a=>identity), find_best=true);

o[1] # errors
o[2] # verbose list

o = evaluate_reconstructions(readdir("/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/mnist/setvae/seed=1/", join=true), DataLoader(randn(3,3)), Dict{Symbol, Function}(:a=>identity), find_best=false, verbose=true);

o[1][1]

o = evaluate_reconstructions(readdir("/home/zorekmat/MIL/GenerativeMIL/data/GenExperiments/shapenet_class/setvae/seed=1/", join=true), DataLoader(randn(3,3)), Dict{Symbol, Function}(:a=>identity), find_best=false, verbose=true);

o[1] # errors