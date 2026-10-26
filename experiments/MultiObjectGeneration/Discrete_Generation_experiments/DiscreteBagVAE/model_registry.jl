# Shared by DiscreteBagVAE_train.jl and Evaluation/*.jl, so training and evaluation build models the
# same way. Expects `using GenerativeMIL` in the including script.

# --- model switch -----------------------------------------------------------
# model_cfg[:model_type] -> constructor taking (; idim, n_max, model_cfg...). Add new discrete-set models here.
# Loss weights go in train.loss_kwargs and their names are model-specific: λ, use_elbo for
# DiscreteBagVAE; λ_exist for DeepSlotQueryVAE (a wrong name is silently ignored via kwargs...).
const MODEL_CONSTRUCTORS = Dict(
    "discretebagvae" => discretebagvae_constructor_from_named_tuple,
    "discretebagae" => discretebagvae_constructor_from_named_tuple,  # same model, deterministic (gaussian_mlp=false, use_elbo=false); separate name -> separate output dir
    "dsqvae" => dsqvae_constructor_from_named_tuple,
)

function build_model(model_type::AbstractString, model_cfg; idim::Int, n_max::Int)
    key = lowercase(model_type)
    haskey(MODEL_CONSTRUCTORS, key) || error("Unknown model_type: $model_type (known: $(join(keys(MODEL_CONSTRUCTORS), ", ")))")
    return MODEL_CONSTRUCTORS[key](; idim=idim, n_max=n_max, dict2nt(model_cfg)...)
end
