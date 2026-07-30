using Revise
using ArgParse
using DrWatson
@quickactivate

using Random
using Serialization
using Flux
#using Zygote
using Statistics
using StatsBase
using GenerativeMIL 
using MLUtils
#using YAML
using ProgressBars
using CUDA
#using GenerativeMIL: CUDA, OptimalTransport


    s = ArgParseSettings()
    @add_arg_table! s begin
        "model_name"
            arg_type = String
            default = "poolmodel"
            help = "Name of the model to evaluate (\"poolmodel\", \"setvae\" or \"neuralstatistician\" )"
        "dataset"
            arg_type = String
            default = "mnist"
            help = "Name of the dataset to evaluate on (\"mnist\", \"airplane\" or \"core5\")"
        "seed"
            arg_type = Int
            default = 1
            help = "Seed to evaluate."
        "valid_repeats"
            arg_type = Int
            default = 2
            help = "Number of evaluation repetitions on valid set. For selection of best hyperparameters."
        "test_repeats"
            arg_type = Int
            default = 5
            help = "Number of evaluation repetitions on test set. Main output to summarize performance of models."
        "loss_functions"
            arg_type = String
            default = "cd,sh,dcd,mmd"
            help = "comma-separated list of loss functions to evaluate (\"cd\", \"sh\", \"dcd\", \"mmd\"). \n No other losses are implemented."
        "sinkhorn_epsilon"
            arg_type = Real
            default = 1f0
            help = "Sinkhorn regularization parameter for the Sinkhorn divergence loss function. If sinkhorn loss is not used, this parameter is ignored."
        "dcd_alpha"
            arg_type = Real
            default = 1f0
            help = "Alpha parameter for the density-aware Chamfer distance loss function. If DCD loss is not used, this parameter is ignored."
        "mmd_sigma"
            arg_type = Real
            default = 1.32f0
            help = "RBF kernel bandwidth for the MMD loss function. If MMD loss is not used, this parameter is ignored."
        "mmd_multipliers"
            arg_type=String
            default="0.25,0.5,1.0"
            help = "comma-separated list of sigma multipliers, default: (\"0.25\", \"0.5\", \"1.0\")"
        "time_limit"
            arg_type = Int
            default = 24
            help = "training time budget in hours"
        "ui"
            arg_type = Int
            default = Int(rand(1:10^6))
            help = "optional unique identifier for this run, used for naming output directory if model_dir is not set"
    end

    args = parse_args(ARGS, s; as_symbols=true)

    # This I named datasets by loading functions instead of names like "airplane" or "core5", therefore this hashmap is used to map the dataset names to the corresponding folder names under which data are saved
    dataset_hashmap = Dict("mnist" => "mnist", "airplane" => "shapenet_class", "core5" => "shapenet_multiple_classes")
    dataset_name = get(dataset_hashmap, args[:dataset], args[:dataset])

    data_configs = Dict(
        "mnist" => Dict{Symbol, Any}(
            :sample_on_fly => false, 
            :cardinality_count => "balanced", 
            :dataset => "mnist", 
            :normalize => true, 
            :npoints => 512, 
            :ratio => 0.2
            ),
        "shapenet_class" => Dict{Symbol, Any}(
            :sample_on_fly => true, 
            :normalize => true, 
            :dataset => "shapenet_class", 
            :npoints => 2048, 
            :type => "airplane"
            ),
        "shapenet_multiple_classes" => Dict{Symbol, Any}(
            :sample_on_fly => true, 
            :normalize => true, 
            :dataset => "shapenet_multiple_classes", 
            :npoints => 2048, 
            :type => "core5", 
            :upper_bound_n => 3000, 
            :balanced_classes => true
        ),
    )
    data_cfg = data_configs[dataset_name]


    losses_to_use = Symbol.(split(args[:loss_functions], ","))
    α = Float32(args[:dcd_alpha])
    ϵ = Float32(args[:sinkhorn_epsilon])
    σᵢ = Float32(args[:mmd_sigma])
    multipliers = parse.(Float32, split(args[:mmd_multipliers], ","))

    loss_functions = Dict{Symbol, Function}(
        :cd      => (x,y,args...) -> GenerativeMIL.chamfer_distance_eval(x,y),
        :sh      => (x,y,args...) -> GenerativeMIL.sinkhorn_divergence_loss_eval(x,y,ϵ),
        :dcd     => (x,y,args...) -> GenerativeMIL.density_aware_chamfer_distance_eval(x,y,α),
        :mmd     => (x,y,args...) -> GenerativeMIL.maximum_mean_discrepancy_rbf_eval(x,y; sigma=σᵢ .* multipliers),
    )
    losses_to_drop = setdiff(keys(loss_functions), losses_to_use)
    loss_functions = delete!(loss_functions, losses_to_drop...)

    path_to_folder = readdir(datadir("GenExperiments/$(dataset_name)/$(args[:model_name])/seed=$(args[:seed])/"), join=true)

    o = evaluate_reconstructions(
        path_to_folder, 
        data_cfg, 
        loss_functions; 
        valid_repeats=args[:valid_repeats], 
        test_repeats=args[:test_repeats], 
        device=CUDA.functional() ? cu : cpu,
        find_best=false,  
        verbose=true
    );
