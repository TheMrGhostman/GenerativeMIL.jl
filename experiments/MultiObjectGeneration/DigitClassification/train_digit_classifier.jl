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
using JLD2, JSON3
using MLUtils
using Zygote
using Statistics

dict2nt(x) = (; (Symbol(k) => v for (k, v) in x)...)

function load_cfg(path::String)
    yaml = YAML.load_file(path; dicttype=Dict{Symbol,Any})
    return Dict(Symbol(k) => v for (k, v) in yaml)
end

function resolve_function(x)
    x isa Function && return x
    return eval(Symbol(x))
end

function accuracy_onehot(ŷ, y)
    ŷ_idx = argmax(ŷ; dims=1)
    y_idx = argmax(y; dims=1)
    return mean(vec(ŷ_idx .== y_idx))
end 

function build_classifier(;idim::Int, pre_hdim::Int, pre_depth::Int, pma_heads::Int, pma_attention_fn, post_hdim::Int, post_depth::Int, odim::Int, activation, kwargs...)
    clf = PoolEncoder(
        create_mlp(idim, pre_hdim, pre_depth, pre_hdim, resolve_function(activation)),
        PMA(1, pre_hdim, pma_heads; attention_fn=resolve_function(pma_attention_fn)),
        Flux.Chain(
            create_mlp(pre_hdim, post_hdim, post_depth, odim, resolve_function(activation); out_identity=true)...,
            flatten
        )
    )
    return clf
end


function GenerativeMIL.optim_step(model::PoolEncoder, batch::Tuple{X, Y}, opt::NamedTuple, logpdf, device::Function=cpu; unique_classes=1:2, kwargs...) where {X <: AbstractArray{<:AbstractFloat,3}, Y <: AbstractArray{Int}}
    x, y = batch
    x, y = device(x), device(Flux.onehotbatch(y, unique_classes))
    loss, (∇model,) = Zygote.withgradient(model) do m
        logpdf(m(x), y)
    end
    #return loss, logs, ∇model
    opt, model = Optimisers.update(opt, model, ∇model)
    return model, opt, (ℒ = loss,)
end

function GenerativeMIL.valid_step(model::PoolEncoder, dataloader::DataLoader, logpdf; device::Function=cpu, unique_classes=1:2, kwargs...)
    ℒ = 0f0
    acc = 0f0
    for batch in dataloader
        x, y = batch
        x, y = device(x), device(Flux.onehotbatch(y, unique_classes))
        ŷ = model(x)
        loss = logpdf(ŷ, y)
        acc += accuracy_onehot(ŷ, y)
        ℒ += loss
    end
    n = length(dataloader)
    logs = (; ℒᵥ = ℒ/n, accuracy = acc/n)
    return logs, ℒ/n
end




function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "config_file"
            arg_type = String
            default = joinpath(@__DIR__, "configs", "test_configs", "clf_test_cfg.yml")
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

    data_cfg  = Dict{Symbol,Any}(cfg[:data]) 
    model_cfg = Dict{Symbol,Any}(cfg[:model])
    train_cfg = Dict{Symbol,Any}(cfg[:train])

    data_cfg[:seed] = args[:seed]
    train_cfg[:max_train_time] = Int((args[:time_limit] - 0.5) * 3600)
    args[:epochs] > 0 && (train_cfg[:epochs] = args[:epochs])
    !isempty(args[:model_dir]) && (train_cfg[:model_dir] = args[:model_dir])
    if !isdir(train_cfg[:model_dir])
        train_cfg[:model_dir] = datadir("DigitClassification", "$(data_cfg[:dataset])", model_cfg[:model_type], "seed=$(args[:seed])", "$(train_cfg[:model_dir])_ID-$(lpad_number(args[:ui], Int(1e5)))" ) 
    end


    dataloaders = create_dataloaders(batch_size=get(train_cfg, :batch_size, 16), x_only=false, data_cfg)
    idim = size(first(dataloaders[:train])[1], 1)

    model = build_classifier(; idim=idim, dict2nt(model_cfg)...);
    lr = get(train_cfg, :lr, 1f-3)
    optimiser = Optimisers.AdamW(; eta=lr, lambda = get(train_cfg, :weight_decay, 0));

    loss_function = (x,y)->Flux.Losses.logitbinarycrossentropy(x,y)

    lr_scheduler_cfg = get(train_cfg, :lr_scheduler, nothing)
    lr_scheduler = create_lr_scheduler(lr_scheduler_cfg, lr, get(train_cfg, :epochs, 1000))

    train_kwargs = (; 
        use_gpu = get(train_cfg, :use_gpu, true),
        model_dir = get(train_cfg, :model_dir, datadir("experiments", model_cfg[:model_type], "seed=$(args[:seed])")),
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
        save_val_predictions = get(train_cfg, :save_val_predictions, false),
        unique_classes = sort(unique(dataloaders.train.data[2])),
    )

    # save config as json
    mkpath(train_kwargs.model_dir)
    open(joinpath(train_kwargs.model_dir, "config.json"), "w") do io
        JSON3.pretty(io, (;train_cfg=train_cfg, model_cfg=cfg[:model], data_cfg=data_cfg, train_kwargs=train_kwargs))
    end

    # Launcher handles config + dataloaders and passes resolved schedulers to train_model!.
    train_time = @elapsed result = train_model!(
        model,
        (train=dataloaders[:train], valid=dataloaders[:valid]),
        optimiser;
        loss_function = loss_function,
        lr_scheduler = lr_scheduler,
        train_kwargs...
    );

    # save model and opt state
    model_state = Flux.state(result.model|>cpu);
    opt_state = Flux.state(result.opt|>cpu);
    model_state_dir = joinpath(train_kwargs.model_dir, "model_state")
    mkpath(model_state_dir)
    jldsave(joinpath(model_state_dir, "model_state_final.jld2"), model_state = model_state, opt_state = opt_state)


    run_config_file = joinpath(train_kwargs.model_dir, "run_config.jls")
    serialize(run_config_file, (
        args = args,
        data_cfg = data_cfg,
        model_cfg = model_cfg,
        train_cfg = train_cfg,
        train_kwargs = train_kwargs,
        lr_scheduler_cfg = lr_scheduler_cfg,
        train_time = train_time,
    ))
    @info "Saved run configuration" file=run_config_file

    serialize(joinpath(train_kwargs.model_dir, "history.jls"), result.history)
    @info "Saved training history" file=joinpath(train_kwargs.model_dir, "history.jls")

    @info "Training finished"
    return result
end

main()