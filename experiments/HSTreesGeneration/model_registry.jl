# Model switch for HSTree generation: model_cfg[:model_type] -> constructor. Shared by model_training.jl (and later
# the evaluation scripts), so training and evaluation build models the same way. Expects `using GenerativeMIL`.
# Each constructor takes keyword arguments only: the data-derived sizes (idim, ...) plus every key of the model:
# block, and must swallow keys it doesn't use via kwargs... (model_type is passed too).
# The model_type also names the output folder: data/HSTreeGenExperiments/<dataset>/<model_type>/..., so a model
# variant that should be evaluated separately gets its own entry here, even if it shares a constructor.
# Loss weights go in train.loss_kwargs; their names are model-specific (λ_exist for HSQVAE, λ/α (+γ for the AE) for StructuredMnistHSQVAE, α₁/α₂ (+γ) for StructuredMnistBagVAE; a wrong name is
# silently ignored by the model's kwargs...).
const MODEL_CONSTRUCTORS = Dict(
    "hsqvae" => HierarchicalSlotQueryVAE,   # kwargs constructor in src/models/HierarchicalSlotQueryVAE.jl
    # src/models/HSTreeSpecialized/StructuredMNIST/; needs a point mask (mnist_clock with point_mask: true, or mnist_clock_unbalanced).
    # VAE and its deterministic AE variant share the constructor; model_type is the only switch (so they land in different folders).
    "structured_mnist_hsqvae" => (; variational=true, kw...) -> (variational === true ||
        throw(ArgumentError("model_type structured_mnist_hsqvae is the VAE; use model_type structured_mnist_hsqae for the AE")); StructuredMnistHSQVAE(; kw...)),
    "structured_mnist_hsqae" => (; variational=false, kw...) -> (variational === false ||
        throw(ArgumentError("model_type structured_mnist_hsqae is the AE; don't set variational")); StructuredMnistHSQVAE(; kw..., variational=false)),
    # StructuredMnistBagVAE (hierarchical DiscreteBagVAE, no slot-query middle / existence head): same VAE/AE switch.
    "structured_mnist_bagvae" => (; variational=true, kw...) -> (variational === true ||
        throw(ArgumentError("model_type structured_mnist_bagvae is the VAE; use model_type structured_mnist_bagae for the AE")); StructuredMnistBagVAE(; kw...)),
    "structured_mnist_bagae" => (; variational=false, kw...) -> (variational === false ||
        throw(ArgumentError("model_type structured_mnist_bagae is the AE; don't set variational")); StructuredMnistBagVAE(; kw..., variational=false)),
)

function build_model(model_type::AbstractString, model_cfg; data_dims...)
    key = lowercase(model_type)
    haskey(MODEL_CONSTRUCTORS, key) || error("Unknown model_type: $model_type (known: $(join(keys(MODEL_CONSTRUCTORS), ", ")))")
    return MODEL_CONSTRUCTORS[key](; data_dims..., dict2nt(model_cfg)...)
end
