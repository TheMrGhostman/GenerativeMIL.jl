# End-to-end generation test on ONE trained run (SetVAE or NeuralStatistician).
#
#   1. load the reference shapes (same function as generation_reference.jl) and the precomputed
#      reference file (D_RR + reference row); stop if it is missing or its fingerprint does not match
#   2. load the last checkpoint of the run (same rule as the reconstruction evaluation)
#   3. generate as many sets as there are reference shapes
#   4. CD and DCD: D_GR, D_GG (+ cached D_RR) -> MMD, COV, 1-NNA
#   5. save metrics (with the reference row copied in), timings and a few generated sets for plotting
#
# Run generation_reference.jl ONCE per (dataset, split, n_ref) before this script.
# Writes ONLY into experiments/EvaluationOfGenExperiments/generation_smoke/ (nothing in data/GenExperiments).
#
# Usage (GPU node), from experiments/EvaluationOfGenExperiments:
#   julia --project generation_smoke_test.jl <run_dir> <dataset> [--split test] [--n_ref 0] [--batch_pairs 64]

include(joinpath(@__DIR__, "generation_common.jl"))

using ArgParse
using Dates
using CUDA
using Flux: cpu

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "run_dir";       arg_type = String; required = true;  help = "run folder (contains models/ and run_config.jls)"
        "dataset";       arg_type = String; required = true;  help = "\"mnist\", \"airplane\" or \"core5\""
        "--split";       arg_type = String; default = "test"; help = "reference split: \"valid\" (selection) or \"test\" (reporting)"
        "--n_ref";       arg_type = Int;    default = 0;      help = "subsample this many reference shapes (0 = all); must match generation_reference.jl"
        "--batch_pairs"; arg_type = Int;    default = 64;     help = "set pairs per distance batch (memory ≈ 16 MB per pair at 2048 points)"
        "--gen_batch";   arg_type = Int;    default = 64;     help = "sets generated per batch"
        "--seed";        arg_type = Int;    default = 1;      help = "data split seed (must be 1, as in training) and sampling seed"
        "--n_save";      arg_type = Int;    default = 16;     help = "number of generated sets saved for plotting"
    end
    return parse_args(ARGS, s; as_symbols=true)
end

"Last checkpoint of a run (zero-padded epoch numbers, so lexicographic max = last epoch)."
function last_checkpoint(run_dir)
    files = filter(f -> startswith(f, "model_ep="), readdir(joinpath(run_dir, "models")))
    @assert !isempty(files) "no model_ep=*.jls checkpoints in $(run_dir)/models"
    return joinpath(run_dir, "models", maximum(files))
end

"Generate `n` sets of `npoints` points in batches; returns a CPU array (d, npoints, n)."
function generate(model, npoints, n, gen_batch)
    out = Array{Float32, 3}[]
    left = n
    while left > 0
        bs = min(gen_batch, left)
        x̂, _ = GenerativeMIL.generate_samples(model, npoints, bs)
        push!(out, Array(x̂))
        left -= bs
    end
    return cat(out...; dims=3)
end

function main()
    args = parse_cli()
    println(args)
    device = CUDA.functional() ? cu : cpu
    @info "device" device CUDA.functional()

    # ── reference shapes + precomputed reference quantities ──
    data = load_reference_data(args[:dataset], args[:split], args[:n_ref], args[:seed])
    R = data.R
    N, npoints = size(R, 3), size(R, 2)
    ref_path = reference_path(args[:dataset], args[:split], N, args[:seed])
    isfile(ref_path) || error("missing $(ref_path)\nrun first:  julia --project generation_reference.jl $(args[:dataset]) --split $(args[:split]) --n_ref $(args[:n_ref])")
    ref = deserialize(ref_path)
    @assert ref.fingerprint == data_fingerprint(R) && ref.ref_idx == data.ref_idx "reference file was computed on DIFFERENT reference shapes: $(ref_path)"
    @info "data" split=args[:split] n_reference=N npoints reference_file=ref_path

    # ── model + generation ──
    ckpt = last_checkpoint(args[:run_dir])
    model = deserialize(ckpt).model |> device
    @info "model" typeof(model).name.name ckpt
    Random.seed!(args[:seed]); CUDA.functional() && CUDA.seed!(args[:seed])
    t_gen = @elapsed G = generate(model, npoints, N, args[:gen_batch])
    @info "generated" size(G) n_nonfinite=count(!isfinite, G) seconds=round(t_gen, digits=1)

    # quick sanity statistics: generated vs. real shapes (coordinate range and spread per shape)
    spread(X) = mean(vec(std(X, dims=2)))
    println("coord range  real: ", extrema(R), "   generated: ", extrema(G))
    println("mean per-shape std  real: ", round(spread(R), digits=4), "   generated: ", round(spread(G), digits=4))

    # ── metrics ──
    results = Dict{String, Any}()
    matrices = Dict{String, Any}()       # D_GR, D_GG per metric (Float32), saved so new metrics never need regeneration
    kw = (batch_pairs = args[:batch_pairs], device = device)
    for (name, dist) in sort(collect(DISTANCES), by=first)
        t_model = @elapsed begin
            D_GR = pairwise_set_distances(dist, G, R; kw...)
            D_GG = pairwise_set_distances(dist, G, G; symmetric=true, kw...)
            m = generation_metrics(D_GR, D_GG, ref.D_RR[name])
        end
        matrices[name] = (D_GR = Float32.(D_GR), D_GG = Float32.(D_GG))
        results[name] = (model = m, reference_row = ref.reference_row[name], seconds = (model = t_model,))
        print_metrics(name, "model        ", m)
        print_metrics(name, "train vs ref ", ref.reference_row[name])
        println("    seconds: model $(round(t_model, digits=1))")
    end

    # ── save ──
    out = joinpath(OUT_DIR, basename(normpath(args[:run_dir])))
    mkpath(out)
    serialize(joinpath(out, "generation_$(args[:split])_n$(N).jls"), (
        args = args, checkpoint = ckpt, dataset = args[:dataset], ref_idx = data.ref_idx, reference_file = ref_path,
        results = results, seconds_generation = t_gen, created = now(),
        samples = G[:, :, 1:min(args[:n_save], N)],            # a few generated sets for plotting
        reference_examples = R[:, :, 1:min(args[:n_save], N)], # and real ones for comparison
    ))
    # distance matrices (rows = generated shapes in generation order, columns = reference shapes in ref_idx order);
    # D_RR is in the reference file. Allows any later metric without regenerating.
    serialize(joinpath(out, "distances_$(args[:split])_n$(N).jls"),
              (matrices = matrices, ref_idx = data.ref_idx, reference_file = ref_path, checkpoint = ckpt, created = now()))
    @info "saved" out
end

main()
