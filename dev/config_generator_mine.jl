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
    items = collect(d)
    for (idx, (k,v)) in enumerate(items)
        if v isa Dict || v isa OrderedDict
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
        # add a blank line between top-level sections
        if indent == 0 && idx < length(items)
            push!(lines, "")
        end
    end
    return join(lines, "\n")
end


function save_to_file(cfg, outdir, filename)
    fname = joinpath(outdir, "$(filename).yml")
    open(fname, "w") do io
        write(io, dict_to_yaml(cfg))
    end
    println("Wrote: ", fname)
end

#function save_to_file(cfg, outdir, i)
#    ts = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
#    fname = joinpath(outdir, "cfg_$(ts)_$(lpad(string(i),3,'0')).yml")
#    open(fname, "w") do io
#        write(io, dict_to_yaml(cfg))
#    end
#    println("Wrote: ", fname)
#    
#end


########################################
# base configs

function base_data_config(dataset="mnist", npoints=512; cardinality_count="balanced", sample_on_fly=false, normalize=true)
    @assert dataset in ["mnist", "modelnet10"] "Unsupported dataset: $dataset"
    
    dict = OrderedDict("dataset" => dataset, "npoints" => npoints)

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
    is_sizes=[32, 16, 8, 4, 2, 1, 1], zdims=[16, 16, 16, 16, 16, 16, 16], expansion_depth=1, expansion_hidden_dim=0, output_activation="identity", kwargs...
)

    return OrderedDict(
        "model_type" => "setvae",
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
    prpdim=64, prpdepth=3, popdim=64, popdepth=3, zdim=64, decdim=64, decdepth=3, poolf="mean-max", gen_sigma="scalar", activation="gelu", init_seed=1, output_activation="identity", kwargs...
)

    return OrderedDict(
        "model_type" => "poolmodel",
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


function base_train_config(;
    loss_function=OrderedDict("type" => "chamfer_distance"),
    lr=0.001,
    weight_decay=0,
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
    verbose=true,
    kwargs...
)

    lr_scheduler_ = lr_scheduler == "WarmupCosine" ? OrderedDict("type"=>"WarmupCosine","milestones"=>[0.02,0.8],"scale"=>10) : nothing
    beta_anealer_ = beta_anealer == "linear" ? OrderedDict("type"=>"linear","max_value"=>1.0,"milestone"=>floor(0.9*epochs)) : beta_anealer


    return OrderedDict(
        "loss_function" => loss_function,
        "lr" => lr,
        "weight_decay" => weight_decay,
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
        "verbose" => verbose
    )
end


function make_base_config(id; model="setvae", dataset="mnist", npoints=512, kwargs...)
    basic_model_cfg = if model == "setvae" 
        base_setvae_config 
    elseif model == "poolmodel"
        base_poolmodel_config
    else
        error("Unknown model: $model")
    end

    output=  OrderedDict(
        "data" => base_data_config(dataset, npoints; kwargs...),
        "model" => basic_model_cfg(kwargs...),
        "train" => base_train_config(kwargs...),
    )
    
    dist_ = output["train"]["loss_function"]["type"]
    dist = if dist_ == "chamfer_distance"
        "cd"
    elseif dist_ in ("maximum_mean_discrepancy", "maximum_mean_discrepency")
        "mmd"
    else
        error("Unknown loss function type: $dist_")
    end
    model_dir = join([dist, model, "c$(id)"], "_")
    output["train"]["model_dir"] = model_dir
    return output
end


function make_standard_grid_setvae_configs(pth::String; dataset="mnist", β = 1f0, save_cds=false, save_mmds=false)
    #TBS = 38400
    more_then_iters = 1000 # I just want to avoid triggering the validation checks, because I want to perform valitation after epoch only. 
    
    if dataset == "mnist"
        npoints = 512
        data_cfg = base_data_config("mnist", npoints; cardinality_count="balanced", sample_on_fly=false, normalize=true)
        cd_epochs = 500
        cd_batch_size= 256
        mmd_epochs = 300
        mmd_batch_size = 64
    elseif dataset == "modelnet10"
        npoints = 2048
        data_cfg = base_data_config("modelnet10", npoints)
        cd_epochs = 1000
        cd_batch_size= 128
        mmd_epochs = 200
        mmd_batch_size = 16
    else
        error("Unknown dataset: $dataset")
    end

    cd_train_cfgs = [
        (
            lr = 0.003, weight_decay=1e-4, lr_scheduler=nothing, epochs=cd_epochs, batch_size=cd_batch_size, beta=β, beta_anealer="linear", 
            loss_function=OrderedDict("type" => "chamfer_distance", "w1" => npoints, "w2" => npoints), 
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
        (
            lr = 0.0001, weight_decay=1e-4, lr_scheduler=nothing, epochs=cd_epochs, batch_size=cd_batch_size, beta=β, beta_anealer="linear", 
            loss_function=OrderedDict("type" => "chamfer_distance", "w1" => npoints, "w2" => npoints), 
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
        (
            lr = 0.0001, weight_decay=1e-4, lr_scheduler="WarmupCosine", epochs=cd_epochs, batch_size=cd_batch_size, beta=β, beta_anealer="linear", 
            loss_function=OrderedDict("type" => "chamfer_distance", "w1" => npoints, "w2" => npoints), 
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
    ]

    # If MMD is defined via EMA, then sigmas are scales [σ/4, σ/2, σ] and σ is updated via EMA. If not EMA, then σ is fixed and defined as [1/4, 1/2, 1/1]. (or different numbers)
    mmd_train_cfgs = [
        (
            lr = 0.003, weight_decay=1e-4, lr_scheduler=nothing, epochs=mmd_epochs, batch_size=mmd_batch_size, beta=β, beta_anealer="linear", 
            loss_function=OrderedDict("type" => "maximum_mean_discrepancy", "sigma" => [0.25, 0.5, 1.0], "sigma_init" => 1.7305675f0, "ema" => true, "decay" => 0.99, "loss_scale" => npoints, "kernel" => "rbf"), 
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
        (
            lr = 0.0001, weight_decay=1e-4, lr_scheduler=nothing, epochs=mmd_epochs, batch_size=mmd_batch_size, beta=β, beta_anealer="linear", 
            loss_function=OrderedDict("type" => "maximum_mean_discrepancy", "sigma" => [0.25, 0.5, 1.0], "sigma_init" => 1.7305675f0, "ema" => true, "decay" => 0.99, "loss_scale" => npoints,  "kernel" => "rbf"), 
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
        (
            lr = 0.0001, weight_decay=1e-4, lr_scheduler="WarmupCosine", epochs=mmd_epochs, batch_size=mmd_batch_size, beta=β, beta_anealer="linear", 
            loss_function=OrderedDict("type" => "maximum_mean_discrepancy", "sigma" => [0.25, 0.5, 1.0], "sigma_init" => 1.7305675f0, "ema" => true, "decay" => 0.99, "loss_scale" => npoints, "kernel" => "rbf"), 
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
    ]

    model_cfgs = [
        (
            hdim=64, heads=4, activation="relu", prior_dim=32, n_mixtures=4, vb_depth=1, vb_hdim=32, is_sizes=[32, 16, 8, 4, 2, 1, 1], zdims=[16, 16, 16, 16, 16, 16, 16], expansion_depth=1, expansion_hidden_dim=0, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="gelu", prior_dim=32, n_mixtures=4, vb_depth=1, vb_hdim=32, is_sizes=[32, 16, 8, 4, 2, 1, 1], zdims=[16, 16, 16, 16, 16, 16, 16], expansion_depth=1, expansion_hidden_dim=0, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="relu", prior_dim=32, n_mixtures=4, vb_depth=1, vb_hdim=32, is_sizes=[32, 16, 8], zdims=[16, 16, 32], expansion_depth=1, expansion_hidden_dim=0, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="gelu", prior_dim=32, n_mixtures=4, vb_depth=1, vb_hdim=32, is_sizes=[32, 16, 8], zdims=[16, 16, 32], expansion_depth=1, expansion_hidden_dim=0, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="relu", prior_dim=32, n_mixtures=4, vb_depth=2, vb_hdim=32, is_sizes=[4, 2, 1], zdims=[64, 64, 64], expansion_depth=2, expansion_hidden_dim=64, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="gelu", prior_dim=32, n_mixtures=4, vb_depth=2, vb_hdim=32, is_sizes=[4, 2, 1], zdims=[64, 64, 64], expansion_depth=2, expansion_hidden_dim=64, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="relu", prior_dim=32, n_mixtures=4, vb_depth=2, vb_hdim=32, is_sizes=[32], zdims=[32], expansion_depth=3, expansion_hidden_dim=64, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="gelu", prior_dim=32, n_mixtures=4, vb_depth=2, vb_hdim=32, is_sizes=[32], zdims=[32], expansion_depth=3, expansion_hidden_dim=64, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="relu", prior_dim=32, n_mixtures=4, vb_depth=2, vb_hdim=32, is_sizes=[1], zdims=[512], expansion_depth=5, expansion_hidden_dim=64, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="gelu", prior_dim=32, n_mixtures=4, vb_depth=2, vb_hdim=32, is_sizes=[1], zdims=[512], expansion_depth=5, expansion_hidden_dim=64, output_activation="identity"
        ),   
    ]

    cd_configs, mmd_configs = [], []
    cd_id = 1
    mmd_id = 1
    for (i, model_cfg) in enumerate(model_cfgs)
        for (configs, train_cfgs, save_flag, dist_sym) in ((cd_configs, cd_train_cfgs, save_cds, :cd), (mmd_configs, mmd_train_cfgs, save_mmds, :mmd))
            for train_cfg in train_cfgs
                cfg = OrderedDict(
                    "data" => data_cfg,
                    "model" => base_setvae_config(;model_cfg...),
                    "train" => base_train_config(;train_cfg...)
                )
                #@show cfg
                dist_ = cfg["train"]["loss_function"]["type"]
                dist = if dist_ == "chamfer_distance"
                    "cd"
                elseif dist_ in ("maximum_mean_discrepancy", "maximum_mean_discrepency")
                    "mmd"
                else
                    error("Unknown loss function type: $dist_")
                end
                id = dist_sym == :cd ? cd_id : mmd_id
                model_dir = join([dist, "setvae", "c$(lpad(string(id), 3, '0'))"], "_")  #lpad_number(ep, epochs) = lpad(string(ep), length(string(epochs)), "0")
                cfg["train"]["model_dir"] = model_dir
                push!(configs, cfg)
                if save_flag
                    save_to_file(cfg, pth, model_dir)
                end
                if dist_sym == :cd
                    cd_id += 1
                else
                    mmd_id += 1
                end
            end
        end
    end

    return cd_configs, mmd_configs
end

t = make_standard_grid_setvae_configs("B:\\Github-Repos\\GenerativeMIL.jl\\scripts\\test_folder\\mnist"; dataset="mnist", β = 1f0, save_cds=true, save_mmds=true);
t = make_standard_grid_setvae_configs("B:\\Github-Repos\\GenerativeMIL.jl\\scripts\\test_folder\\modelnet10"; dataset="modelnet10", β = 1f0, save_cds=true, save_mmds=true);
slength.(t)