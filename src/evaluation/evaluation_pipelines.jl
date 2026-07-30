# Evaluation metafunction ... i just call it function similar to train, you proved metrics list of paths to models you need to evaluate and it should do all dirty work



# interate over all models in a directory and evaluate them ... done
# load all models and evaluate them ... done
# evaluate a single model ... done
# add labels to evaluations so i can evaluate class specific metrics
    # redo dataloader to return labels as well, and then pass them to the evaluation function



function evaluate_reconstructions(paths::Vector{String}, data_cfg::Dict, loss_functions::Dict{Symbol, Function}; find_best::Bool=false, device::Function=cpu, verbose::Bool=true, batch_size::I=64, valid_repeats::I=2, test_repeats::I=5, jsonl_log_path::Union{String, Nothing}=nothing, kwargs...) where I <: Int

    errors = [[],[]]
    runs_valid = NamedTuple[]
    runs_test = NamedTuple[]

    dataloaders = create_dataloaders(batch_size=batch_size, x_only=false, data_cfg)
    y_v = reduce(vcat,getindex.(collect(dataloaders.valid),2))
    y_t = reduce(vcat,getindex.(collect(dataloaders.test),2))
    dataloaders = create_dataloaders(batch_size=batch_size, x_only=true, data_cfg)

    verbose && @info "dataloaders created"
    
    # Initialize JSONL logger if path is provided
    logger = !isnothing(jsonl_log_path) ? JSONLLogger(jsonl_log_path) : nothing

    for path in paths
        # TODO add try except statement
        try
            @assert isfile(joinpath(path, "run_config.jls")) "Run config for $(path) does not exist!!"
            # find  model
            model_paths = joinpath(path, "models")    
            @assert isdir(model_paths) "Models forlder for $(path) does not exist!!"
            model_checkpoints = readdir(model_paths);
            max_epoch_model = maximum(model_checkpoints) # this is ugly ugly thing but works! # REVIEW: review and redo

            model_path = find_best ? joinpath(model_paths, "best_model.jls") : joinpath(model_paths, max_epoch_model) 
            
            @assert isfile(model_path)
            df = deserialize(model_path)
            model = df.model |> device

            o_v = reconstruction_eval_repeated(model, dataloaders.valid, loss_functions, valid_repeats; device=device, verbose=true)
            o_t = reconstruction_eval_repeated(model, dataloaders.test, loss_functions, test_repeats; device=device, verbose=true)
            
            o_v = merge(o_v, (;labels = y_v))
            o_t = merge(o_t, (;labels = y_t))

            out_nt = (valid=o_v, test=o_t)

            save_path = joinpath(path, "evaluation")
            !isdir(save_path) && mkdir(save_path) 
            # save as jls
            serialize(joinpath(save_path, "evaluation_reconstruction.jls"), out_nt)
            # save as json for compatibility
            open(joinpath(save_path, "evaluation_reconstruction.json"), "w") do io
                JSON3.pretty(io, out_nt)
            end
            
            push!(runs_valid, o_v)
            push!(runs_test, o_t)
            
            # Log successful processing
            !isnothing(logger) && log!(logger, (;path=path, status="done"))
        catch e 
            push!(errors[1], e)
            push!(errors[2], path)
            # Log error
            !isnothing(logger) && log!(logger, (;path=path, status="error", error_message=string(e)))
        end
    end
    
    # Close logger if it was initialized
    !isnothing(logger) && close(logger)
    
    return errors, runs_valid, runs_test
end
