# Model switch for HSTree generation: model_cfg[:model_type] -> constructor. Shared by model_training.jl (and later
# the evaluation scripts), so training and evaluation build models the same way. Expects `using GenerativeMIL`.
# Each constructor takes keyword arguments only: the data-derived sizes (idim, ...) plus every key of the model:
# block, and must swallow keys it doesn't use via kwargs... (model_type is passed too).
# The model_type also names the output folder: data/HSTreeGenExperiments/<dataset>/<model_type>/..., so a model
# variant that should be evaluated separately gets its own entry here, even if it shares a constructor.
# Loss weights go in train.loss_kwargs; their names are model-specific (λ_exist for HSQVAE; a wrong name is
# silently ignored by the model's kwargs...).
const MODEL_CONSTRUCTORS = Dict(
    "hsqvae" => HierarchicalSlotQueryVAE,   # kwargs constructor in src/models/HierarchicalSlotQueryVAE.jl
)

function build_model(model_type::AbstractString, model_cfg; data_dims...)
    key = lowercase(model_type)
    haskey(MODEL_CONSTRUCTORS, key) || error("Unknown model_type: $model_type (known: $(join(keys(MODEL_CONSTRUCTORS), ", ")))")
    return MODEL_CONSTRUCTORS[key](; data_dims..., dict2nt(model_cfg)...)
end
