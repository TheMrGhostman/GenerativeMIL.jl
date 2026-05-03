using Random, Dates, DataStructures

function yaml_scalar(x)
    if x === nothing
        return "null"
    elseif x isa Bool
        return x ? "true" : "false"
    elseif x isa Number
        return string(x)
    elseif x isa String
        s = replace(x, "\""=>"\\\"")
        return "\"$s\""
    else
        return "\"$(string(x))\""
    end
end

function yaml_value(v, indent)
    if v isa Dict
        return dict_to_yaml(v, indent)
    elseif v isa AbstractVector
        # inline array like [a, b]
        parts = [ (el isa Dict) ? "\n" * indentstr(indent+1) * dict_to_yaml(el, indent+1) : yaml_scalar(el) for el in v ]
        if any(x->startswith(x, "\n"), parts)
            # Mixed complex arrays: use block-style
            lines = ["[]"]
            for el in v
                if el isa Dict
                    push!(lines, indentstr(indent) * "- " * "")
                    push!(lines, dict_to_yaml(el, indent+1))
                else
                    push!(lines, indentstr(indent) * "- " * yaml_scalar(el))
                end
            end
            return join(lines, "\n")
        else
            return "[" * join(parts, ", ") * "]"
        end
    else
        return yaml_scalar(v)
    end
end

function indentstr(n)
    return repeat("  ", n)
end

function dict_to_yaml(d::OrderedDict, indent::Int=0)
    lines = String[]
    pref = indentstr(indent)
    for (k,v) in d
        if v isa Dict
            push!(lines, "$(pref)$(k):")
            push!(lines, dict_to_yaml(v, indent+1))
        else
            val = yaml_value(v, indent)
            if startswith(val, "\n")
                push!(lines, "$(pref)$(k):")
                push!(lines, val)
            else
                push!(lines, "$(pref)$(k): $(val)")
            end
        end
    end
    return join(lines, "\n")
end



function save_to_file(cfg, outdir, i)
    ts = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    fname = joinpath(outdir, "cfg_$(ts)_$(lpad(string(i),3,'0')).yml")
    open(fname, "w") do io
        write(io, dict_to_yaml(cfg))
    end
    println("Wrote: ", fname)
    
end


########################################
# base configs

function base_data_confg(dataset="mnist", npoints=512; cardinality_count="balanced", sample_on_fly=false, normalize=true)
    @assert dataset in ["mnist", "modelnet10"] "Unsupported dataset: $dataset"
    
    dict = OrderedDict("npoints" => npoints, "dataset" => dataset)

    if dataset == "mnist"
        dict = merge(dict, OrderedDict("cardinality_count" => cardinality_count,"sample_on_fly" => sample_on_fly, "normalize" => normalize))
    elseif dataset == "modelnet10"
        dict = merge(dict, OrderedDict("type" => "all",))
    else
        error("Unknown dataset: $dataset")
    end
    return dict
end



function base_setvae_config(;
    hdim=64, heads=4, activation="gelu", prior_dim=32, vb_depth=2, vb_hdim=64, n_mixtures=5,
    is_sizes=[32, 16, 8, 4, 2, 1, 1], zdims=[16, 16, 16, 16, 16, 16, 16], expansion_depth=1, expansion_hidden_dim=0, output_activation="identity"
)

    return OrderedDict(
        "hdim" => hdim,
        "heads" => heads,
        "activation" => activation,
        "prior_dim" => prior_dim,
        "vb_depth" => vb_depth,
        "vb_hdim" => vb_hdim,
        "is_sizes" => is_sizes,
        "zdims" => zdims,
        "expansion_depth" => expansion_depth,
        "expansion_hidden_dim" => expansion_hidden_dim,
        "n_mixtures" => n_mixtures,
        "output_activation" => output_activation,
    )
end


function base_poolmodel_config(;
    prpdim=64, prpdepth=3, popdim=64, popdepth=3, zdim=64, decdim=64, decdepth=3, poolf="mean-max", gen_sigma="scalar", activation="gelu", init_seed=1, output_activation="identity"
)

    return OrderedDict(
        "prpdim" => prpdim,
        "prpdepth" => prpdepth,
        "popdim" => popdim,
        "popdepth" => popdepth,
        "zdim" => zdim,
        "decdim" => decdim,
        "decdepth" => decdepth,
        "poolf" => poolf,
        "gen_sigma" => gen_sigma,
        "activation" => activation,
        "init_seed" => init_seed,
        "output_activation" => output_activation
    )
end


function base_train_config(model_dir="cd_setvae_c17";
    loss_function=OrderedDict("type" => "chamfer_distance"),
    lr=0.001,
    lr_scheduler="WarmupCosine",
    epochs=500,
    batch_size=256,
    beta=1.0,
    beta_anealer="linear",
    use_gpu=true,
    valid_check_interval=150,
    validation_check_after_epoch=false,
    checkpoint_interval_epochs=10,
    early_stopping=true,
    patience=100000,
    grad_skip=false,
    save_val_predictions=true,
    val_prediction_count=16,
    val_prediction_interval_epochs=10,
    val_prediction_dirname="val_predictionsdata",
    seed=1,
    verbose=true
)

    lr_scheduler_ = lr_scheduler == "WarmupCosine" ? OrderedDict("type"=>"WarmupCosine","milestones"=>[0.02,0.8],"scale"=>10) : nothing
    beta_anealer_ = beta_anealer == "linear" ? OrderedDict("type"=>"linear","max_value"=>1.0,"milestone"=>450) : beta_anealer


    return OrderedDict(
        "loss_function" => loss_function,
        "lr" => lr,
        "lr_scheduler" => lr_scheduler_,
        "epochs" => epochs,
        "batch_size" => batch_size,
        "beta" => beta,
        "beta_anealer" => beta_anealer_,
        "use_gpu" => use_gpu,
        "valid_check_interval" => valid_check_interval,
        "validation_check_after_epoch" => validation_check_after_epoch,
        "checkpoint_interval_epochs" => checkpoint_interval_epochs,
        "early_stopping" => early_stopping,
        "patience" => patience,
        "grad_skip" => grad_skip,
        "save_val_predictions" => save_val_predictions,
        "val_prediction_count" => val_prediction_count,
        "val_prediction_interval_epochs" => val_prediction_interval_epochs,
        "val_prediction_dirname" => val_prediction_dirname,
        "seed" => seed,
        "model_dir" => model_dir,
        "verbose" => verbose
    )
end


function make_base_config(model="setvae")
    basic_model_cfg = if model == "setvae" 
        base_setvae_config 
    elseif model == "poolmodel"
        base_poolmodel_config
    else
        error("Unknown model: $model")
    end

    return OrderedDict(
        "data" => base_data_confg(),
        "model" => basic_model_cfg(),
        "train" => base_train_config(),
    )
end

