using Revise
using DrWatson
@quickactivate

using ArgParse
using Random
using Serialization
using YAML
using OrderedCollections
using GenerativeMIL
using Flux
using CUDA
using MLUtils

dict2nt(x) = (; (Symbol(k) => v for (k, v) in x)...)

function load_cfg(path::String)
    yaml = YAML.load_file(path; dicttype=Dict{Symbol,Any})
    return Dict(Symbol(k) => v for (k, v) in yaml)
end

function resolve_activation(x)
    x isa Function && return x
    return eval(Symbol(x))
end

function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "config_file"
            arg_type = String
            default = joinpath(@__DIR__, "configs", "setvae_c1.yml")
            help = "YAML configuration file"
        "seed"
            arg_type = Int
            default = 1
            help = "random seed"
        "time_limit"
            arg_type = Int
            default = 24
            help = "training time budget in hours"
        "model_dir"
            arg_type = String
            default = ""
            help = "optional output directory override"
        "epochs"
            arg_type = Int
            default = -1
            help = "optional epoch override"
        "ui"
            arg_type = Int
            default = Int(rand(1:10^6))
            help = "optional unique identifier for this run, used for naming output directory if model_dir is not set"
    end

    args = parse_args(ARGS, s; as_symbols=true)
    cfg = load_cfg(args[:config_file])

    data_cfg = Dict{Symbol,Any}(cfg[:data])
    model_cfg = Dict{Symbol,Any}(cfg[:model])
    train_cfg = Dict{Symbol,Any}(cfg[:train])

    data_cfg[:seed] = args[:seed]
    train_cfg[:max_train_time] = Int((args[:time_limit] - 0.5) * 3600)
    args[:epochs] > 0 && (train_cfg[:epochs] = args[:epochs])
    !isempty(args[:model_dir]) && (train_cfg[:model_dir] = args[:model_dir])
    if !isdir(train_cfg[:model_dir])
        train_cfg[:model_dir] = datadir("GenExperiments", "$(data_cfg[:dataset])", "seed=$(args[:seed])", "$(train_cfg[:model_dir])_ID-$(lpad_number(args[:ui], Int(1e5)))" )
    end

    model_cfg[:activation] = resolve_activation(model_cfg[:activation])
    model_cfg[:output_activation] = resolve_activation(get(model_cfg, :output_activation, "identity"))

    data = load_modelnet10(get(data_cfg, :npoints, 512), get(data_cfg, :type, "all"); validation=true, seed=args[:seed]);
    @info "Loaded ModelNet10" train=size(data[1][1]) valid=size(data[2][1]) test=size(data[3][1])
    
    dataloaders = (
        train = DataLoader(data[1][1], batchsize=get(train_cfg, :batch_size, 16), shuffle=true), 
        valid = DataLoader(data[2][1], batchsize=get(train_cfg, :batch_size, 16))
    )

    model = setvae_constructor_from_named_tuple(; idim=size(data[1][1], 1), dict2nt(model_cfg)...);
    lr = get(train_cfg, :lr, 1f-3)
    optimiser = Optimisers.AdaMax(lr);

    # Vytvoř β_scheduler z konfigurace
    beta_scheduler_cfg = get(train_cfg, :beta_anealer, get(train_cfg, :beta, 1f0))
    beta_scheduler = create_beta_scheduler(beta_scheduler_cfg)
    lr_scheduler_cfg = get(train_cfg, :lr_scheduler, nothing)
    lr_scheduler = create_lr_scheduler(lr_scheduler_cfg, lr, get(train_cfg, :epochs, 1000))

    train_kwargs = (; 
        use_gpu = get(train_cfg, :use_gpu, true),
        model_dir = get(train_cfg, :model_dir, datadir("experiments", "setvae_modelnet", "seed=$(args[:seed])")),
        verbose = get(train_cfg, :verbose, false),
        valid_check_interval = get(train_cfg, :valid_check_interval, 1000),
        validation_check_after_epoch = get(train_cfg, :validation_check_after_epoch, false),
        checkpoint_interval_epochs = get(train_cfg, :checkpoint_interval_epochs, 10),
        epochs = get(train_cfg, :epochs, 1000),
        early_stopping = get(train_cfg, :early_stopping, true),
        patience = get(train_cfg, :patience, 10^4),
        max_train_time = get(train_cfg, :max_train_time, Int(23.5 * 3600)),
        grad_skip = get(train_cfg, :grad_skip, false),
        validation_verbose = get(train_cfg, :validation_verbose, false),
        save_val_predictions = get(train_cfg, :save_val_predictions, true),
        val_prediction_count = get(train_cfg, :val_prediction_count, 16),
        val_prediction_interval_epochs = get(train_cfg, :val_prediction_interval_epochs, nothing),
        val_prediction_dirname = joinpath(get(train_cfg, :model_dir, ""), get(train_cfg, :val_prediction_dirname, "val_predictions")),
    )

    # Launcher handles config + dataloaders and passes resolved schedulers to train_model!.
    result = train_model!(
        model,
        dataloaders,
        optimiser;
        loss_function = chamfer_distance,
        β_scheduler = beta_scheduler,
        lr_scheduler = lr_scheduler,
        train_kwargs...
    );

    run_config_file = joinpath(train_kwargs.model_dir, "run_config.jls")
    serialize(run_config_file, (
        args = args,
        data_cfg = data_cfg,
        model_cfg = model_cfg,
        train_cfg = train_cfg,
        train_kwargs = train_kwargs,
        beta_scheduler_cfg = beta_scheduler_cfg,
        lr_scheduler_cfg = lr_scheduler_cfg,
    ))
    @info "Saved run configuration" file=run_config_file

    serialize(joinpath(train_kwargs.model_dir, "history.jls"), result.history)
    @info "Saved training history" file=joinpath(train_kwargs.model_dir, "history.jls")

    @info "Training finished"
    return result
end

main()