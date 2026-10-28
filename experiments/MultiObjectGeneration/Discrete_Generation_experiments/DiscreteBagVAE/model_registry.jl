# Shared by DiscreteBagVAE_train.jl and Evaluation/*.jl, so training and evaluation build models the
# same way. Expects `using GenerativeMIL` in the including script.

# --- model switch -----------------------------------------------------------
# model_cfg[:model_type] -> constructor taking (; idim, n_max, model_cfg...). Add new discrete-set models here.
# Loss weights go in train.loss_kwargs and their names are model-specific: λ, use_elbo, no_grad_card for
# DiscreteBagVAE; λ_exist for DeepSlotQueryVAE (a wrong name is silently ignored via kwargs...).
const MODEL_CONSTRUCTORS = Dict(
    "discretebagvae" => discretebagvae_constructor_from_named_tuple,
    "discretebagae" => discretebagvae_constructor_from_named_tuple,  # same model, deterministic (gaussian_mlp=false, use_elbo=false); separate name -> separate output dir
    "discretebagvae_ngc" => discretebagvae_constructor_from_named_tuple,  # same model, trained with loss_kwargs no_grad_card=true (cardinality loss doesn't reach the encoder); separate output dir
    "discretebagvae_caw" => discretebagvae_constructor_from_named_tuple,  # cardinality-aware encoder: model.card_pooling ("linear"/"log"/"both") appends the bag size after PMA
    "dsqvae" => dsqvae_constructor_from_named_tuple,
)

function build_model(model_type::AbstractString, model_cfg; idim::Int, n_max::Int)
    key = lowercase(model_type)
    haskey(MODEL_CONSTRUCTORS, key) || error("Unknown model_type: $model_type (known: $(join(keys(MODEL_CONSTRUCTORS), ", ")))")
    return MODEL_CONSTRUCTORS[key](; idim=idim, n_max=n_max, dict2nt(model_cfg)...)
end
