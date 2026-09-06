# Verifies that a HierarchicalSlotQueryVAE checkpoint saved *before* the
# DeepSlotQueryVAE refactor (self_attns::Vector + cross_attns::Vector fields
# collapsed into a single decoder::TransformerDecoder field) can still be
# reconstructed and loaded, via a small state migration.
#
# Point RUN_DIR at any run directory that has config.json and
# model_state/model_state_final.jld2 (as written by train_hsqvae.jl).

using DrWatson
@quickactivate

using GenerativeMIL
using Flux
using JLD2, JSON3

const RUN_DIR = datadir("HGenExperiments", "mnist_clock", "cd_hsqvae_c010_ID-055096")
const STATE_PATH = joinpath(RUN_DIR, "model_state", "model_state_final.jld2")
const CONFIG_PATH = joinpath(RUN_DIR, "config.json")

to_native(x) = x isa AbstractString ? String(x) : x  # JSON3 strings aren't Base.String; model constructors check `isa String`

function load_model_cfg(config_path::String)
    cfg = JSON3.read(read(config_path, String))
    return Dict{Symbol,Any}(Symbol(k) => to_native(v) for (k, v) in cfg.model_cfg)
end

"""
`migrate_dsq_state(dsq_state::NamedTuple)`

Old (pre-refactor) `deep_slot_query` state had `self_attns`/`cross_attns` as flat sibling
keys. The current `DeepSlotQueryVAE` nests them under a single `decoder` (`TransformerDecoder`)
field instead. Re-nests an old state into the new shape; a no-op if the state is already in the
new shape (already has a `decoder` key).
"""
function migrate_dsq_state(dsq_state::NamedTuple)
    haskey(dsq_state, :decoder) && return dsq_state
    return (
        encoder = dsq_state.encoder,
        prior = dsq_state.prior,
        z_to_hidden = dsq_state.z_to_hidden,
        decoder = (self_attns = dsq_state.self_attns, cross_attns = dsq_state.cross_attns),
        output_head = dsq_state.output_head,
        exist_head = dsq_state.exist_head,
        queries = dsq_state.queries,
    )
end

migrate_hsqvae_state(state::NamedTuple) =
    merge(state, (deep_slot_query = migrate_dsq_state(state.deep_slot_query),))

function main()
    model_cfg = load_model_cfg(CONFIG_PATH)
    state = JLD2.load(STATE_PATH, "model_state")

    idim = size(state.encoder.prepool.layers[1].weight, 2) # inferred from the checkpoint itself
    @info "Reconstructing model from saved config" idim model_cfg

    model = HierarchicalSlotQueryVAE(; idim=idim, dict2nt(model_cfg)...)

    migrated = migrate_hsqvae_state(state)
    Flux.loadmodel!(model, migrated)
    @info "Checkpoint loaded into freshly constructed model successfully"

    @assert model.deep_slot_query.decoder.self_attns[1].FF.layers[1].weight == state.deep_slot_query.self_attns[1].FF.layers[1].weight
    @assert model.deep_slot_query.queries == state.deep_slot_query.queries
    @info "Spot-checked weights match the saved checkpoint"

    x = randn(Float32, idim, 8, 3, 2)     # (D, N, L, BS)
    x_mask = trues(1, 1, 3, 2)
    x̂, logits_exist, μ_z, Σ_z = model(x, x_mask)
    @info "Forward pass through loaded model ran fine" size(x̂) size(logits_exist)

    return model
end

model = main()
