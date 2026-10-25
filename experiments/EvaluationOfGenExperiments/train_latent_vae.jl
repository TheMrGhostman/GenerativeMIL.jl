# Train a stage-2 latent VAE (`LatentVAE`, src/evaluation/latent_density.jl) on the set encodings h of ONE
# trained PoolModel run, and save it into <run_dir>/latent_model/.
#
#   1. ENCODINGS (computed once per run and saved, then reused by every VAE config and by the GMM):
#      the frozen PoolModel (last checkpoint) encodes every TRAIN shape K = --n_resample times, each time with a
#      new random subset of points (ShapeNet: sample_on_fly), so the VAE sees the subsampling variation
#      (augmentation); VALID shapes are fixed and encoded once.
#      -> <run_dir>/latent_model/encodings_K<K>.jls
#   2. VAE TRAINING on CPU (tiny MLP; the expensive part was the encoding): standardised encodings, negative ELBO
#      with learned decoder std γ, early stopping on the VALID negative ELBO, best epoch kept.
#   3. VALID log-likelihood of the VAE (ELBO, a lower bound) vs. a full-covariance Gaussian fitted on the train
#      encodings, both in the raw encoding space -> does the VAE model h better than a Gaussian?
#      -> <run_dir>/latent_model/vae_depth=<d>_hdim=<h>_zdim=<z>_K<K>.jls
#
# Usage, from experiments/EvaluationOfGenExperiments:
#   julia --project train_latent_vae.jl <run_dir> <dataset> [--depth 2] [--hdim 128] [--zdim 16] [--n_resample 10]
#   julia --project train_latent_vae.jl <run_dir> <dataset> --encode_only        # only step 1

include(joinpath(@__DIR__, "generation_common.jl"))   # DATA_CONFIGS, last_checkpoint

using ArgParse
using Dates
using CUDA
using Flux
using Flux: cpu
using MLUtils
using Optimisers
using Zygote

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "run_dir";       arg_type = String;  required = true;   help = "PoolModel run folder (contains models/)"
        "dataset";       arg_type = String;  required = true;   help = "\"mnist\", \"airplane\" or \"core5\""
        "--depth";       arg_type = Int;     default = 2;       help = "Dense layers in the encoder and in the decoder"
        "--hdim";        arg_type = Int;     default = 128;     help = "hidden width"
        "--zdim";        arg_type = Int;     default = 16;      help = "latent dimension of the latent VAE"
        "--activation";  arg_type = String;  default = "swish"; help = "activation function name"
        "--n_resample";  arg_type = Int;     default = 10;      help = "encodings per TRAIN shape (different point subsets); forced to 1 if points are not resampled"
        "--epochs";      arg_type = Int;     default = 300
        "--patience";    arg_type = Int;     default = 30;      help = "early stopping patience in epochs (valid -ELBO)"
        "--batch_size";  arg_type = Int;     default = 64;      help = "encodings per VAE batch"
        "--lr";          arg_type = Float64; default = 1e-3
        "--seed";        arg_type = Int;     default = 1;       help = "data split seed (must be 1), resampling and init seed"
        "--encode_only"; action = :store_true;                   help = "only compute and save the encodings"
    end
    return parse_args(ARGS, s; as_symbols=true)
end

# last_checkpoint: generation_common.jl

"""
Encode TRAIN shapes K times (new point subsets each pass, fixed shape order) and VALID shapes once.
Returns `(H_train (d, K·N), shape_idx, pass_idx, H_valid (d, Nv), K)`.
"""
function compute_encodings(pm, dataset, K, seed; batch_size=64)
    device = CUDA.functional() ? cu : cpu      # `cu`, not Flux `gpu`: gpu needs cuDNN, which this project must not load
    Random.seed!(seed)                         # ShapeNet point subsampling uses the global RNG
    data_cfg = merge(DATA_CONFIGS[dataset], Dict{Symbol, Any}(:seed => seed))
    dls = create_dataloaders(data_cfg; batch_size=batch_size, x_only=true)
    if K > 1 && !get(data_cfg, :sample_on_fly, false)
        @warn "$(dataset): training points are not resampled (sample_on_fly=false), so all K passes would be identical -> K = 1"
        K = 1
    end
    train = DataLoader(dls.train.data; batchsize=batch_size, shuffle=false)   # fixed shape order, so columns are traceable
    N = numobs(dls.train.data)
    Hs = [collect_encodings(pm, train; device=device) for _ in 1:K]           # each pass draws new point subsets
    H_valid = collect_encodings(pm, dls.valid; device=device)
    return (H_train = reduce(hcat, Hs), shape_idx = repeat(1:N, K), pass_idx = repeat(1:K, inner=N),
            H_valid = H_valid, K = K, device = string(device))
end

"Mean negative ELBO over a matrix of encodings, in batches."
function mean_loss(vae, H; bs=1024)
    tot = 0.0
    for idx in Iterators.partition(1:size(H, 2), bs)
        l, _ = latent_vae_loss(vae, H[:, idx]); tot += l * length(idx)
    end
    return tot / size(H, 2)
end

function main()
    args = parse_cli()
    println(args)
    @assert args[:seed] == 1 "training used the seed=1 data split"
    out_dir = joinpath(args[:run_dir], "latent_model")
    mkpath(out_dir)
    ckpt = last_checkpoint(args[:run_dir])
    # no point resampling (MNIST) -> K passes would be identical; set K = 1 HERE so that all file names say K1
    if args[:n_resample] > 1 && !get(DATA_CONFIGS[args[:dataset]], :sample_on_fly, false)
        @warn "$(args[:dataset]): training points are not resampled (sample_on_fly=false) -> --n_resample 1"
        args[:n_resample] = 1
    end

    # ── 1. encodings (load if already computed for this K) ──
    enc_path = joinpath(out_dir, "encodings_K$(args[:n_resample]).jls")
    if isfile(enc_path)
        enc = deserialize(enc_path)
        @assert same_file(enc.poolmodel_checkpoint, ckpt) "encodings were computed with another checkpoint: $(enc.poolmodel_checkpoint)"
        @info "loaded encodings" enc_path
    else
        pm = deserialize(ckpt).model
        @assert pm isa PoolModel "expected a PoolModel run, got $(typeof(pm))"
        pm = pm |> (CUDA.functional() ? cu : cpu)
        t_enc = @elapsed e = compute_encodings(pm, args[:dataset], args[:n_resample], args[:seed])
        enc = merge(e, (poolmodel_checkpoint = ckpt, dataset = args[:dataset], seed = args[:seed], seconds = t_enc, created = now()))
        tmp = "$(enc_path).tmp_$(getpid())"; serialize(tmp, enc); mv(tmp, enc_path; force=true)   # atomic
        @info "saved encodings" enc_path seconds=round(t_enc, digits=1)
    end
    H_train, H_valid = enc.H_train, enc.H_valid
    @info "encodings" dim=size(H_train, 1) n_train=size(H_train, 2) K=enc.K n_valid=size(H_valid, 2) checkpoint=ckpt
    args[:encode_only] && return

    # ── 2. latent VAE (CPU) ──
    μₕ, σₕ = vec(mean(H_train, dims=2)), max.(vec(std(H_train, dims=2)), 1f-6)
    Random.seed!(args[:seed])
    act = getfield(Flux, Symbol(args[:activation]))
    vae = LatentVAE(size(H_train, 1), args[:zdim]; hdim=args[:hdim], depth=args[:depth], activation=act, μₕ=μₕ, σₕ=σₕ)
    opt = Optimisers.setup(Optimisers.Adam(args[:lr]), vae)
    rng = MersenneTwister(args[:seed])

    history = NamedTuple[]
    best = (loss = Inf, epoch = 0, model = deepcopy(vae))
    t_train = @elapsed for epoch in 1:args[:epochs]
        tr = [0.0, 0.0, 0.0, 0]          # loss, rec, kl, batches
        for idx in Iterators.partition(randperm(rng, size(H_train, 2)), args[:batch_size])
            (l, logs), ∇ = Zygote.withgradient(m -> latent_vae_loss(m, H_train[:, idx]), vae)
            opt, vae = Optimisers.update(opt, vae, ∇[1])
            tr .+= [l, logs.rec, logs.kl, 1]
        end
        v = mean_loss(vae, H_valid)
        γ = exp(vae.logγ[1])
        push!(history, (epoch = epoch, train_loss = tr[1] / tr[4], train_rec = tr[2] / tr[4], train_kl = tr[3] / tr[4], valid_loss = v, γ = γ))
        v < best.loss && (best = (loss = v, epoch = epoch, model = deepcopy(vae)))
        (epoch % 10 == 0 || epoch == 1) && println("epoch $(epoch)  train $(round(tr[1]/tr[4], digits=3)) (rec $(round(tr[2]/tr[4], digits=3)), kl $(round(tr[3]/tr[4], digits=3)))  valid $(round(v, digits=3))  γ $(round(γ, sigdigits=3))  best@$(best.epoch)")
        epoch - best.epoch >= args[:patience] && (println("early stopping at epoch $(epoch) (best $(best.epoch))"); break)
    end

    # ── 3. VAE vs. Gaussian on VALID encodings (raw space, nats per encoding; higher = better) ──
    g = fit_latent_gaussian(H_train)
    gauss_ll = mean(GenerativeMIL.logpdf(g, H_valid))
    vae_elbo = mean(elbo_raw(best.model, H_valid))
    println("valid log-likelihood per encoding:  Gaussian $(round(gauss_ll, digits=2))   latent VAE (ELBO, lower bound) $(round(vae_elbo, digits=2))")

    # ── save ──
    path = joinpath(out_dir, "vae_depth=$(args[:depth])_hdim=$(args[:hdim])_zdim=$(args[:zdim])_K$(enc.K).jls")
    tmp = "$(path).tmp_$(getpid())"
    serialize(tmp, (model = best.model, args = args, best_epoch = best.epoch, best_valid_loss = best.loss,
                    history = history, valid_ll_gaussian = gauss_ll, valid_elbo_vae = vae_elbo,
                    encodings_file = enc_path, poolmodel_checkpoint = ckpt, train_seconds = t_train, created = now()))
    mv(tmp, path; force=true)
    @info "saved" path train_seconds=round(t_train, digits=1)
end

main()
