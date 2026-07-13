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

function dict_to_yaml(d::Union{OrderedDict, Dict}, indent::Int=0)
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


########################################
# base configs

function base_data_config(dataset="mnist", npoints=512; cardinality_count="balanced", sample_on_fly=false, normalize=true, balanced_classes=true)
    @assert dataset in ["mnist", "modelnet10"] "Unsupported dataset: $dataset"
    
    dict = OrderedDict("dataset" => dataset, "npoints" => npoints)

    if dataset == "mnist"
        dict = merge(dict, OrderedDict("cardinality_count" => cardinality_count, "sample_on_fly" => sample_on_fly, "normalize" => normalize, "ratio" => 0.2))
    elseif dataset == "modelnet10"
        dict = merge(dict, OrderedDict("balanced_classes"=>balanced_classes, "normalize"=>normalize, "sample_on_fly" => sample_on_fly, "ratio" => 0.1))#OrderedDict("type" => "all",)
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

function base_train_config(;
    loss_function=OrderedDict("type" => "chamfer_distance"),
    lr=0.001,
    weight_decay=0,
    lr_scheduler="WarmupCosine",
    epochs=500,
    batch_size=256,
    beta=1.0,
    beta_anealer="linear",
    beta_milestone = 0.9,
    beta_initial = 0.0001,
    beta_slope = 0.015f0,
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
    if beta_anealer == "linear"
        beta_anealer_ = OrderedDict("type"=>"linear", "max_value"=>beta, "milestone"=>floor(last(beta_milestone)*epochs))
    elseif beta_anealer == "step_linear"
        @assert length(beta_milestone) == 2 && beta_milestone[1] < beta_milestone[2] "wrong milestones for beta scheduler, either not 2 values or not ascending"
        beta_anealer_ = OrderedDict(
            "type" => "step_linear",
            "initial" => beta_initial,
            "max_value" => beta,
            "milestones" => [floor(beta_milestone[1] * epochs), floor(beta_milestone[2] * epochs)],
        )
    elseif beta_anealer == "sigmoidal"
        @assert length(beta_milestone) == 1 "SigmoidSchedule takes only single milestone"
        beta_anealer_ = OrderedDict(
            "type" => "sigmoidal",
            "max_value" => beta, 
            "slope_factor" => beta_slope,
            "milestone" => beta_milestone,
        )
    else
        beta_anealer_ = beta_anealer
    end
    #beta_anealer_ = beta_anealer == "linear" ? OrderedDict("type"=>"linear","max_value"=>beta,"milestone"=>floor(0.9*epochs)) : beta_anealer

    return OrderedDict(
        "loss_function" => loss_function,
        "lr" => lr,
        "weight_decay" => weight_decay,
        "lr_scheduler" => lr_scheduler_,
        "epochs" => epochs,
        "batch_size" => batch_size,
        "beta" => beta,                                                         #TODO figure out if i can just delete it?
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



function make_standard_grid_setvae_configs(pth::String, init_id::Int = 1; dataset="mnist", cd_epochs=nothing, mmd_epochs=nothing, sh_epochs=nothing, dcd_epochs=nothing, β = 1f0, save_cds=false, save_mmds=false, save_shs=false, save_dcd=false, kwargs...)
    #TBS = 38400
    more_then_iters = 1000 # I just want to avoid triggering the validation checks, because I want to perform valitation after epoch only. 
    
    if dataset == "mnist"
        npoints = 512
        data_cfg = base_data_config("mnist", npoints; cardinality_count="balanced", sample_on_fly=false, normalize=true)
        cd_epochs = cd_epochs === nothing ? 1000 : cd_epochs
        cd_batch_size= 128
        mmd_epochs = mmd_epochs === nothing ? 300 : mmd_epochs
        mmd_batch_size = 32
        sh_epochs = sh_epochs === nothing ? 300 : sh_epochs
        sh_batch_size = 128
        dcd_epochs = dcd_epochs === nothing ? 1000 : dcd_epochs
        dcd_batch_size = 128
    elseif dataset == "modelnet10"
        npoints = 2048
        data_cfg = base_data_config("modelnet10", npoints; balanced_classes=true, sample_on_fly=false, normalize=true)
        cd_epochs = cd_epochs === nothing ? 1000 : cd_epochs
        cd_batch_size= 128
        mmd_epochs = mmd_epochs === nothing ? 200 : mmd_epochs
        mmd_batch_size = 16
        sh_epochs = sh_epochs === nothing ? 200 : sh_epochs
        sh_batch_size = 16
        dcd_epochs = dcd_epochs === nothing ? 1000 : dcd_epochs
        dcd_batch_size = 128
    elseif dataset == "airplane"
        npoints = 2048
        data_cfg = OrderedDict("dataset" => "shapenet_class", "npoints" => npoints, "normalize"=>true, "sample_on_fly" => true, "type" => "airplane")
        cd_epochs = cd_epochs === nothing ? 1000 : cd_epochs
        cd_batch_size= 128
        mmd_epochs = mmd_epochs === nothing ? 200 : mmd_epochs
        mmd_batch_size = 16
        sh_epochs = sh_epochs === nothing ? 200 : sh_epochs
        sh_batch_size = 16
        dcd_epochs = dcd_epochs === nothing ? 1000 : dcd_epochs
        dcd_batch_size = 128
    elseif dataset == "core5"
        npoints = 2048
        data_cfg = OrderedDict("dataset" => "shapenet_multiple_classes", "npoints" => npoints, "normalize"=>true, "sample_on_fly" => true, "type" => "core5", "balanced_classes"=>true,  "upper_bound_n" => 3000)
        cd_epochs = cd_epochs === nothing ? 1100 : cd_epochs
        cd_batch_size= 128
        mmd_epochs = mmd_epochs === nothing ? 200 : mmd_epochs
        mmd_batch_size = 16
        sh_epochs = sh_epochs === nothing ? 200 : sh_epochs
        sh_batch_size = 16
        dcd_epochs = dcd_epochs === nothing ? 1100 : dcd_epochs
        dcd_batch_size = 128
    else
        error("Unknown dataset: $dataset")
    end

    cyclical_anealer = OrderedDict(
        "type" => "cyclical_sigmoidal", 
        "beta_warmup" => 1e-5, 
        "max_value" => β, 
        "warmup_epochs" => 150,
        "rise_epochs" => 200,
        "hold_epochs" => 50, 
        "cycles" => 4,
        "slope_factor" => 12f0 / 200 ,
    )

    one_per_cent_sh = Int(sh_epochs * 0.01)
    
    cyclical_anealer_sh = OrderedDict(
        "type" => "cyclical_sigmoidal", 
        "beta_warmup" => 1e-5, 
        "max_value" => β, 
        "warmup_epochs" => 10 * one_per_cent_sh,
        "rise_epochs" => Int(4 * 4.5 * one_per_cent_sh),
        "hold_epochs" => Int(4.5 * one_per_cent_sh), 
        "cycles" => 4,
        "slope_factor" => 12f0 /  (4 * 4.5 * one_per_cent_sh),
    )

    one_per_cent_dcd = Int(sh_epochs * 0.01)
    
    cyclical_anealer_dcd = OrderedDict(
        "type" => "cyclical_sigmoidal", 
        "beta_warmup" => 1e-5, 
        "max_value" => β, 
        "warmup_epochs" => 10 * one_per_cent_dcd,
        "rise_epochs" => Int(4 * 4.5 * one_per_cent_dcd),
        "hold_epochs" => Int(4.5 * one_per_cent_dcd), 
        "cycles" => 4,
        "slope_factor" => 12f0 /  (4 * 4.5 * one_per_cent_dcd),
    )

    cd_train_cfgs = [
        (   # works with shallow models
            lr = 0.0003, weight_decay=1e-4, lr_scheduler=nothing, epochs=cd_epochs, batch_size=cd_batch_size, beta=β, beta_anealer=cyclical_anealer,
            loss_function=OrderedDict("type" => "chamfer_distance", "w1" => npoints, "w2" => npoints), 
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
        (   # works with shallow models
            lr = 0.0003, weight_decay=1e-4, lr_scheduler=nothing, epochs=cd_epochs, batch_size=cd_batch_size, beta=β, beta_anealer="sigmoidal", beta_milestone=800, beta_slope = 0.015f0,
            loss_function=OrderedDict("type" => "chamfer_distance", "w1" => npoints, "w2" => npoints), 
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
        (   # works with shallow models
            lr = 0.0003, weight_decay=1e-4, lr_scheduler=nothing, epochs=cd_epochs, batch_size=cd_batch_size, beta=β, beta_anealer="step_linear", beta_milestone=[0.4, 0.9], beta_initial=0.00001,
            loss_function=OrderedDict("type" => "chamfer_distance", "w1" => npoints, "w2" => npoints), 
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
    ]

    # If MMD is defined via EMA, then sigmas are scales [σ/4, σ/2, σ] and σ is updated via EMA. If not EMA, then σ is fixed and defined as [1/4, 1/2, 1/1]. (or different numbers)
    mmd_train_cfgs = [
        (
            lr = 0.0003, weight_decay=1e-4, lr_scheduler=nothing, epochs=mmd_epochs, batch_size=mmd_batch_size, beta=β, beta_anealer="step_linear", beta_milestone=[50/mmd_epochs, 0.9], beta_initial=0.00001,
            loss_function=OrderedDict("type" => "maximum_mean_discrepancy", "sigma" => [0.25, 0.5, 1.0], "sigma_init" => 1.7305675f0, "ema" => true, "decay" => 0.99, "loss_scale" => npoints, "kernel" => "rbf"), 
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
        (
            lr = 0.0001, weight_decay=1e-4, lr_scheduler="WarmupCosine", epochs=mmd_epochs, batch_size=mmd_batch_size, beta=β, beta_anealer="step_linear", beta_milestone=[50/mmd_epochs, 0.9], beta_initial=0.00001,
            loss_function=OrderedDict("type" => "maximum_mean_discrepancy", "sigma" => [0.25, 0.5, 1.0], "sigma_init" => 1.7305675f0, "ema" => true, "decay" => 0.99, "loss_scale" => npoints, "kernel" => "rbf"), 
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
    ]

    sh_train_cfgs = [
        (   # works with shallow models
            lr = 0.0003, weight_decay=1e-4, lr_scheduler=nothing, epochs=sh_epochs, batch_size=sh_batch_size, beta=β, beta_anealer=cyclical_anealer_sh,
            loss_function=OrderedDict("type" => "sinkhorn_divergence_loss", "loss_scale" => npoints,"eps" => 1.0, "maxiter" => 50, "regularization"=>false),
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
        (   # works with shallow models
            lr = 0.0003, weight_decay=1e-4, lr_scheduler=nothing, epochs=sh_epochs, batch_size=sh_batch_size, beta=β, beta_anealer="sigmoidal", beta_milestone = 80 * one_per_cent_sh, beta_slope = 0.015f0,
            loss_function=OrderedDict("type" => "sinkhorn_divergence_loss", "loss_scale" => npoints,"eps" => 1.0, "maxiter" => 50, "regularization"=>false),
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
        (   # works with shallow models
            lr = 0.0003, weight_decay=1e-4, lr_scheduler=nothing, epochs=sh_epochs, batch_size=sh_batch_size, beta=β, beta_anealer="step_linear", beta_milestone=[0.4, 0.9], beta_initial=0.00001,
            loss_function=OrderedDict("type" => "sinkhorn_divergence_loss", "loss_scale" => npoints,"eps" => 1.0, "maxiter" => 50, "regularization"=>false), 
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
    ]
    dcd_train_cfgs = [
        (   # works with shallow models
            lr = 0.0003, weight_decay=1e-4, lr_scheduler=nothing, epochs=dcd_epochs, batch_size=dcd_batch_size, beta=β, beta_anealer=cyclical_anealer_dcd,
            loss_function=OrderedDict("type" => "density_aware_chamfer_distance", "loss_scale" => npoints,"alpha"=>1000f0),
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
        (   # works with shallow models
            lr = 0.0003, weight_decay=1e-4, lr_scheduler=nothing, epochs=dcd_epochs, batch_size=dcd_batch_size, beta=β, beta_anealer="sigmoidal", beta_milestone = 80 * one_per_cent_dcd, beta_slope = 0.015f0,
            loss_function=OrderedDict("type" => "density_aware_chamfer_distance", "loss_scale" => npoints,"alpha"=>1000f0),
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
        (   # works with shallow models
            lr = 0.0003, weight_decay=1e-4, lr_scheduler=nothing, epochs=dcd_epochs, batch_size=dcd_batch_size, beta=β, beta_anealer="step_linear", beta_milestone=[0.4, 0.9], beta_initial=0.00001,
            loss_function=OrderedDict("type" => "density_aware_chamfer_distance", "loss_scale" => npoints,"alpha"=>1000f0), 
            valid_check_interval=more_then_iters, validation_check_after_epoch=true, checkpoint_interval_epochs=10, early_stopping=true, patience=100000, verbose=true
        ),
    ]

    model_cfgs = [
        (
            hdim=64, heads=4, activation="gelu", prior_dim=32, n_mixtures=4, vb_depth=1, vb_hdim=32, is_sizes=[32, 16, 8, 4, 2, 1, 1], zdims=[16, 16, 16, 16, 16, 16, 16], 
            expansion_depth=1, expansion_hidden_dim=0, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="gelu", prior_dim=32, n_mixtures=4, vb_depth=2, vb_hdim=32, is_sizes=[32, 16, 8], zdims=[16, 16, 32], 
            expansion_depth=2, expansion_hidden_dim=64, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="gelu", prior_dim=32, n_mixtures=4, vb_depth=2, vb_hdim=32, is_sizes=[4, 2, 1], zdims=[64, 64, 64], 
            expansion_depth=2, expansion_hidden_dim=64, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="gelu", prior_dim=32, n_mixtures=4, vb_depth=2, vb_hdim=32, is_sizes=[32], zdims=[32], 
            expansion_depth=3, expansion_hidden_dim=64, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="gelu", prior_dim=32, n_mixtures=4, vb_depth=2, vb_hdim=32, is_sizes=[1], zdims=[512], 
            expansion_depth=4, expansion_hidden_dim=64, output_activation="identity"
        ), 
        (
            hdim=64, heads=4, activation="gelu", prior_dim=32, n_mixtures=4, vb_depth=2, vb_hdim=32, is_sizes=[32, 16, 8, 4, 2, 1, 1], zdims=[16, 16, 16, 16, 16, 16, 16], 
            expansion_depth=2, expansion_hidden_dim=64, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="gelu", prior_dim=32, n_mixtures=4, vb_depth=2, vb_hdim=32, is_sizes=[32, 16, 8, 4, 2, 1, 1], zdims=[32, 32, 32, 32, 32, 32, 32], 
            expansion_depth=2, expansion_hidden_dim=64, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="gelu", prior_dim=32, n_mixtures=4, vb_depth=1, vb_hdim=64, is_sizes=[32, 16, 8, 4, 2, 1], zdims=[8, 16, 32, 64, 64, 128], 
            expansion_depth=2, expansion_hidden_dim=64, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="gelu", prior_dim=32, n_mixtures=4, vb_depth=2, vb_hdim=32, is_sizes=[32, 16, 8, 4, 2, 1], zdims=[8, 16, 32, 32, 32, 64], 
            expansion_depth=2, expansion_hidden_dim=64, output_activation="identity"
        ),
        (
            hdim=64, heads=4, activation="gelu", prior_dim=32, n_mixtures=4, vb_depth=2, vb_hdim=32, is_sizes=[32, 16, 8, 4, 2, 1], zdims=[8, 16, 32, 64, 64, 128], 
            expansion_depth=2, expansion_hidden_dim=64, output_activation="identity"
        ),
    ]


    cd_configs, mmd_configs, sh_configs, dcd_configs = [], [], [], []
    cd_id = init_id
    mmd_id = init_id
    sh_id = init_id
    dcd_id = init_id
    for (configs, train_cfgs, save_flag, dist_sym) in ((cd_configs, cd_train_cfgs, save_cds, :cd), (mmd_configs, mmd_train_cfgs, save_mmds, :mmd), (sh_configs, sh_train_cfgs, save_shs, :sh), (dcd_configs, dcd_train_cfgs, save_dcd, :dcd))
        for train_cfg in train_cfgs
            for model_cfg in model_cfgs
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
                elseif dist_ == "sinkhorn_divergence_loss"
                    "sh"
                elseif dist_ == "density_aware_chamfer_distance"
                    "dcd"
                else
                    error("Unknown loss function type: $dist_")
                end
                id = if dist_sym == :cd
                    cd_id
                elseif dist_sym == :mmd
                    mmd_id
                elseif dist_sym == :sh
                    sh_id
                elseif dist_sym == :dcd
                    dcd_id
                else 
                    error("unknown dist_sym")
                end
                model_dir = join([dist, "setvae", "e_c$(lpad(string(id), 3, '0'))"], "_")  #lpad_number(ep, epochs) = lpad(string(ep), length(string(epochs)), "0")
                cfg["train"]["model_dir"] = model_dir
                push!(configs, cfg)
                if save_flag
                    save_to_file(cfg, pth, model_dir)
                end
                if dist_sym == :cd
                    cd_id += 1
                elseif dist_sym == :mmd
                    mmd_id += 1
                elseif dist_sym == :sh
                    sh_id += 1
                elseif dist_sym == :dcd
                    dcd_id += 1
                else
                    error("unknown dist_sym")
                end
            end
        end
    end

    return cd_configs, mmd_configs, sh_configs, dcd_configs
end



t = make_standard_grid_setvae_configs("experiments/GenerationExperiments/SetVAE_experiments/configs/core5_configs_e", 501; dataset="core5", β = 1f0, save_cds=true, save_mmds=false, save_shs=false, cd_epochs=1100, sh_epochs=200);

t = make_standard_grid_setvae_configs("experiments/GenerationExperiments/SetVAE_experiments/configs/core5_configs_e", 551; dataset="core5", β = 0.5f0, save_cds=true, save_mmds=false, save_shs=false, cd_epochs=1100, sh_epochs=200);


t = make_standard_grid_setvae_configs("experiments/GenerationExperiments/SetVAE_experiments/configs/core5_configs_e", 601; dataset="core5", β = 1f0, save_cds=true, save_mmds=false, save_shs=false, cd_epochs=1150, sh_epochs=200);


t = make_standard_grid_setvae_configs("experiments/GenerationExperiments/SetVAE_experiments/configs/mnist_configs_e", 1; dataset="mnist", β = 1f0, save_cds=true, save_mmds=false, save_shs=true, cd_epochs=1100, sh_epochs=600);

t = make_standard_grid_setvae_configs("experiments/GenerationExperiments/SetVAE_experiments/configs/mnist_configs_e", 1; dataset="mnist", β = 1f0, save_cds=false, save_mmds=false, save_shs=false, save_dcd=true, cd_epochs=1100, sh_epochs=600, dcd_epochs=1100);