# Shared by generation_reference.jl and generation_smoke_test.jl.
# Everything that decides WHICH reference shapes are used lives here, so both scripts are guaranteed
# to use the same data (and the model script additionally verifies a fingerprint of it).

using DrWatson
@quickactivate

using Random
using Serialization
using Statistics
using GenerativeMIL

# Same data configs as run_evaluation_reconstruction.jl (+ seed = 1, as in training)
const DATA_CONFIGS = Dict(
    "mnist"    => Dict{Symbol, Any}(:sample_on_fly => false, :cardinality_count => "balanced", :dataset => "mnist",
                                    :normalize => true, :npoints => 512, :ratio => 0.2),
    "airplane" => Dict{Symbol, Any}(:sample_on_fly => true, :normalize => true, :dataset => "shapenet_class",
                                    :npoints => 2048, :type => "airplane"),
    "core5"    => Dict{Symbol, Any}(:sample_on_fly => true, :normalize => true, :dataset => "shapenet_multiple_classes",
                                    :npoints => 2048, :type => "core5", :upper_bound_n => 3000, :balanced_classes => true),
)

const DISTANCES = Dict(
    "cd"  => (x, y) -> GenerativeMIL.chamfer_distance_eval(x, y),
    "dcd" => (x, y) -> GenerativeMIL.density_aware_chamfer_distance_eval(x, y, 1f0),   # α = 1, as in the reconstruction evaluation
)

const OUT_DIR = joinpath(@__DIR__, "generation_smoke")
const REF_DIR = joinpath(OUT_DIR, "reference")

stack_loader(dl) = cat((Array(x) for x in dl)...; dims=3)

"""
`load_reference_data(dataset, split, n_ref, seed)`

Reference shapes R (from `split`, optionally a random subsample of `n_ref`) and N random TRAIN shapes Tr
(for the reference row). Fully determined by the arguments: the global RNG is seeded before loading
(ShapeNet valid/test point subsampling uses it) and the subsamples come from `MersenneTwister(seed)`.
"""
function load_reference_data(dataset::String, split::String, n_ref::Int, seed::Int)
    @assert seed == 1 "training used the seed=1 data split; another seed changes the MNIST split"
    Random.seed!(seed)
    data_cfg = merge(DATA_CONFIGS[dataset], Dict{Symbol, Any}(:seed => seed))
    dls = create_dataloaders(data_cfg; batch_size=256, x_only=true)
    R_full = stack_loader(getproperty(dls, Symbol(split)))
    Tr_full = stack_loader(dls.train)
    rng = MersenneTwister(seed)
    ref_idx = n_ref > 0 ? sort(randperm(rng, size(R_full, 3))[1:min(n_ref, end)]) : collect(1:size(R_full, 3))
    R = R_full[:, :, ref_idx]
    train_idx = sort(randperm(rng, size(Tr_full, 3))[1:size(R, 3)])
    return (R = R, Tr = Tr_full[:, :, train_idx], ref_idx = ref_idx, train_idx = train_idx, n_train_total = size(Tr_full, 3))
end

"File with the precomputed reference quantities for one (dataset, split, N, seed)."
reference_path(dataset, split, N, seed) = joinpath(REF_DIR, "reference_$(dataset)_$(split)_n$(N)_seed$(seed).jls")

"Fingerprint of the reference shapes: the model script checks it, so it can never silently use other shapes."
data_fingerprint(R) = hash(R)

function print_metrics(name, label, x)
    println("$(uppercase(name))  $(label) MMD=$(round(x.mmd, sigdigits=4))  COV=$(round(x.cov, digits=3))  ",
            "1-NNA=$(round(x.nna, digits=3))  (acc. gen half $(round(x.nna_gen, digits=3)), ref half $(round(x.nna_ref, digits=3)))",
            hasproperty(x, :div_ratio) ? "  diversity gen/ref=$(round(x.div_ratio, digits=3))" : "")
end
