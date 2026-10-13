# Precompute everything in the generation evaluation that does NOT depend on the model. Run ONCE per
# (dataset, split, n_ref, seed), before generation_smoke_test.jl:
#
#   - D_RR: reference × reference distance matrix, for CD and DCD
#   - reference row: N random TRAIN shapes in place of generated ones vs. the same reference shapes
#     (MMD, COV, 1-NNA a perfect generator would get)
#
# Output: generation_smoke/reference/reference_<dataset>_<split>_n<N>_seed<seed>.jls
#
# Usage (GPU node), from experiments/EvaluationOfGenExperiments:
#   julia --project generation_reference.jl airplane [--split test] [--n_ref 0] [--batch_pairs 64]

include(joinpath(@__DIR__, "generation_common.jl"))

using ArgParse
using Dates
using CUDA
using Flux: cpu

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "dataset";       arg_type = String; required = true;  help = "\"mnist\", \"airplane\" or \"core5\""
        "--split";       arg_type = String; default = "test"; help = "reference split: \"valid\" or \"test\""
        "--n_ref";       arg_type = Int;    default = 0;      help = "subsample this many reference shapes (0 = all)"
        "--batch_pairs"; arg_type = Int;    default = 64;     help = "set pairs per distance batch"
        "--seed";        arg_type = Int;    default = 1;      help = "data split seed (must be 1)"
    end
    return parse_args(ARGS, s; as_symbols=true)
end

function main()
    args = parse_cli()
    println(args)
    device = CUDA.functional() ? cu : cpu
    @info "device" device CUDA.functional()

    data = load_reference_data(args[:dataset], args[:split], args[:n_ref], args[:seed])
    R, Tr = data.R, data.Tr
    N = size(R, 3)
    path = reference_path(args[:dataset], args[:split], N, args[:seed])
    isfile(path) && (@info "already exists, nothing to do" path; return)
    @info "data" split=args[:split] n_reference=N n_train_total=data.n_train_total npoints=size(R, 2)

    kw = (batch_pairs = args[:batch_pairs], device = device)
    D_RR, reference_row, seconds = Dict{String, Matrix{Float64}}(), Dict{String, Any}(), Dict{String, Any}()
    for (name, dist) in sort(collect(DISTANCES), by=first)
        t_rr = @elapsed D_RR[name] = pairwise_set_distances(dist, R, R; symmetric=true, kw...)
        t_row = @elapsed reference_row[name] = generation_metrics(pairwise_set_distances(dist, Tr, R; kw...),
                                                                  pairwise_set_distances(dist, Tr, Tr; symmetric=true, kw...), D_RR[name])
        seconds[name] = (D_RR = t_rr, reference_row = t_row)
        print_metrics(name, "train vs ref ", reference_row[name])
        println("    seconds: D_RR $(round(t_rr, digits=1)), reference row $(round(t_row, digits=1))")
    end

    # write to a temp file, then rename (atomic)
    mkpath(REF_DIR); tmp = "$(path).tmp_$(getpid())"
    serialize(tmp, (args = args, dataset = args[:dataset], split = args[:split], seed = args[:seed], N = N,
                    ref_idx = data.ref_idx, train_idx = data.train_idx, fingerprint = data_fingerprint(R),
                    D_RR = D_RR, reference_row = reference_row, seconds = seconds, created = now()))
    mv(tmp, path; force=true)
    @info "saved" path
end

main()
