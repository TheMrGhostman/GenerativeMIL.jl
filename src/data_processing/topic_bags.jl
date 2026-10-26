#################
### Topic bags ###
#################
# Bags of digits 1:10 with structure between elements: every bag first draws a hidden topic t ~ π,
# then a cardinality c ~ Uniform(card_t), then c digits from that topic. Topics (see
# experiments/.../DiscreteBagVAE/Evaluation/MDs/EVALUATION_DESIGN.md):
#   low      digits uniform on {1,2,3},        c ∈ 8:16  -- big bags, many repeats
#   high     digits uniform on {8,9,10},       c ∈ 1:6   -- small bags (cardinality depends on content)
#   even     digits uniform on {2,4,...,10},   c ∈ 4:12  -- disjoint support from `odd`,
#   odd      digits uniform on {1,3,...,9},    c ∈ 4:12     same cardinality range
#   dominant one digit d ~ U(1:10) per bag is 80 % of the bag, the rest uniform over the other 9, c ∈ 6:16
#   distinct all 10 digits without replacement (no repeats), c ∈ 3:10, rare (π = 0.05)
# In all topics except `distinct`, each element is replaced by a uniformly random digit with
# probability ε = TOPIC_BAGS_EPS, so every bag has a finite likelihood under the generator.
const TOPIC_BAGS_N_CLASSES = 10
const TOPIC_BAGS_EPS = 0.05
const TOPIC_BAGS_TOPICS = (
    (name="low",      kind=:uniform,  support=[1, 2, 3],         card=8:16, π=0.19),
    (name="high",     kind=:uniform,  support=[8, 9, 10],        card=1:6,  π=0.19),
    (name="even",     kind=:uniform,  support=[2, 4, 6, 8, 10],  card=4:12, π=0.19),
    (name="odd",      kind=:uniform,  support=[1, 3, 5, 7, 9],   card=4:12, π=0.19),
    (name="dominant", kind=:dominant, support=collect(1:10),     card=6:16, π=0.19),
    (name="distinct", kind=:distinct, support=collect(1:10),     card=3:10, π=0.05),
)
const TOPIC_BAGS_NAMES = [t.name for t in TOPIC_BAGS_TOPICS]

# per-element digit probabilities of a topic (with the ε noise already mixed in); for `dominant`
# this is the distribution given the dominant digit d
function _topic_theta(topic, d::Int=0)
    θ = zeros(Float64, TOPIC_BAGS_N_CLASSES)
    if topic.kind == :uniform
        θ[topic.support] .= 1 / length(topic.support)
    elseif topic.kind == :dominant
        θ .= 0.2 / (TOPIC_BAGS_N_CLASSES - 1)
        θ[d] = 0.8
    else
        error("no per-element θ for topic kind $(topic.kind)")
    end
    return (1 - TOPIC_BAGS_EPS) .* θ .+ TOPIC_BAGS_EPS / TOPIC_BAGS_N_CLASSES
end

"""
`sample_one_topic_bag(rng, max_card)`

Draw one topic bag (see the `TOPIC_BAGS_TOPICS` comment), one-hot encoded and zero-padded to
`max_card` slots (`max_card` must be ≥ 16, the largest topic cardinality).

Returns:
- `x::Matrix{Float32}` `(10, max_card)`, `mask::BitMatrix` `(1, max_card)`,
  `labels::Vector{Int}` (the drawn digits, length = cardinality), `topic::Int` (index into `TOPIC_BAGS_TOPICS`).
"""
function sample_one_topic_bag(rng, max_card::Int)
    t = sample(rng, 1:length(TOPIC_BAGS_TOPICS), Weights([tp.π for tp in TOPIC_BAGS_TOPICS]))
    topic = TOPIC_BAGS_TOPICS[t]
    c = rand(rng, topic.card)
    elements = if topic.kind == :distinct
        Random.randperm(rng, TOPIC_BAGS_N_CLASSES)[1:c]
    else
        d = rand(rng, 1:TOPIC_BAGS_N_CLASSES)   # dominant digit (unused by :uniform topics)
        draw() = if topic.kind == :uniform
            rand(rng, topic.support)
        else
            rand(rng) < 0.8 ? d : rand(rng, setdiff(1:TOPIC_BAGS_N_CLASSES, d))
        end
        [rand(rng) < TOPIC_BAGS_EPS ? rand(rng, 1:TOPIC_BAGS_N_CLASSES) : draw() for _ in 1:c]
    end
    x = zeros(Float32, TOPIC_BAGS_N_CLASSES, max_card)
    mask = falses(1, max_card)
    x[:, 1:c] .= Flux.onehotbatch(elements, 1:TOPIC_BAGS_N_CLASSES)
    mask[1, 1:c] .= true
    return x, mask, elements, t
end

function _build_topic_bags(rng, max_card::Int, n_samples::Int)
    x = zeros(Float32, TOPIC_BAGS_N_CLASSES, max_card, n_samples)
    mask = falses(1, max_card, n_samples)
    labels = Vector{Vector{Int}}(undef, n_samples)
    topics = Vector{Int}(undef, n_samples)
    for b in 1:n_samples
        x[:, :, b], mask[:, :, b], labels[b], topics[b] = sample_one_topic_bag(rng, max_card)
    end
    return x, mask, labels, topics
end

"""
`load_topic_bags(npoints=nothing; n_samples=10000, n_eval_samples=n_samples, max_digits=16, validation=true, sample_on_fly=false, seed=666, kwargs...)`

Synthetic discrete set dataset with dependencies between the elements of a bag: a mixture of six
topics over digits `1:10` (see the `TOPIC_BAGS_TOPICS` comment). Same layout as
[`load_bag_digits`](@ref) plus the topic of every bag: each split is `(data, mask, labels, topics)`
with `data (10, max_digits, N)`, `mask (1, max_digits, N)`, `labels` the per-bag digit vectors and
`topics::Vector{Int}`. With `x_only=true`, `create_dataloaders` keeps `(data, mask)` as for the
other bag datasets. The exact likelihood is [`topic_bags_logpdf`](@ref).

Arguments:
- `npoints`: ignored (kept so `create_dataloaders` can call every loader the same way).
- `n_samples`: number of training bags (steps per epoch when `sample_on_fly=true`).
- `n_eval_samples`: number of validation and test bags each.
- `max_digits`: slot count, must be ≥ 16 (largest topic cardinality). `min_digits` is ignored:
  cardinality ranges are part of the topics.
- `n_classes`: must be 10 (the topics are defined over digits 1:10).
- `sample_on_fly`: if `true`, training bags are drawn fresh on every access. Validation/test are always fixed.
- `seed`: train/valid/test are built from `MersenneTwister(seed)`, `seed+1`, `seed+2`.

Returns:
- `(train_data, valid_data, test_data)` if `validation=true`, else `(train_data, test_data)`.
"""
function load_topic_bags(npoints=nothing; n_samples::Int=10000, n_eval_samples::Int=n_samples, max_digits::Int=16, n_classes::Int=TOPIC_BAGS_N_CLASSES, validation::Bool=true, sample_on_fly::Bool=false, seed::Int=666, kwargs...)
    max_card = maximum(last(t.card) for t in TOPIC_BAGS_TOPICS)
    max_digits >= max_card || throw(ArgumentError("topic_bags needs max_digits >= $max_card (got $max_digits)"))
    n_classes == TOPIC_BAGS_N_CLASSES || throw(ArgumentError("topic_bags is defined for n_classes = $TOPIC_BAGS_N_CLASSES (got $n_classes)"))
    train_data = if sample_on_fly
        mapobs(_ -> sample_one_topic_bag(Random.default_rng(), max_digits), 1:n_samples)
    else
        _build_topic_bags(MersenneTwister(seed), max_digits, n_samples)
    end
    test_data = _build_topic_bags(MersenneTwister(seed + 2), max_digits, n_eval_samples)
    if validation
        val_data = _build_topic_bags(MersenneTwister(seed + 1), max_digits, n_eval_samples)
        return train_data, val_data, test_data
    end
    return train_data, test_data
end

# log p(h | t) for the count vector h of one bag (cardinality c = sum(h)), including p(c | t)
function _topic_loglik(topic, h::AbstractVector{<:Integer})
    c = sum(h)
    c in topic.card || return -Inf
    logp_c = -log(length(topic.card))
    if topic.kind == :uniform
        return logp_c + Distributions.logpdf(Multinomial(c, _topic_theta(topic)), h)
    elseif topic.kind == :dominant
        lps = [Distributions.logpdf(Multinomial(c, _topic_theta(topic, d)), h) for d in 1:TOPIC_BAGS_N_CLASSES]
        m = maximum(lps)
        return logp_c + m + log(sum(exp.(lps .- m))) - log(TOPIC_BAGS_N_CLASSES)
    else  # :distinct -- uniform over the C(10, c) subsets
        all(<=(1), h) || return -Inf
        return logp_c - log(binomial(TOPIC_BAGS_N_CLASSES, c))
    end
end

"""
`topic_bags_counts(digits)`

Count vector `h` (length 10) of a bag given as a vector of digits in `1:10`.
"""
topic_bags_counts(digits::AbstractVector{<:Integer}) = [count(==(k), digits) for k in 1:TOPIC_BAGS_N_CLASSES]

"""
`topic_bags_logpdf(h)`

Exact log-probability of one bag under the topic-bags generator,
`log Σ_t π_t · p(c | t) · p(h | c, t)`, from its count vector `h` (length 10, `c = sum(h)`; get it
from a digit vector with [`topic_bags_counts`](@ref)). Returns `-Inf` for a bag no topic can
produce (only possible for an empty bag or one longer than 16 elements).
"""
function topic_bags_logpdf(h::AbstractVector{<:Integer})
    lp = _topic_log_joint(h)
    m = maximum(lp)
    return isinf(m) ? -Inf : m + log(sum(exp.(lp .- m)))
end

"""
`topic_bags_posterior(h)`

Posterior topic probabilities `p(t | bag)` (vector over `TOPIC_BAGS_TOPICS`, names in
`TOPIC_BAGS_NAMES`) of one bag, from its count vector `h` (as in [`topic_bags_logpdf`](@ref)).
Returns all zeros for a bag no topic can produce.
"""
function topic_bags_posterior(h::AbstractVector{<:Integer})
    lp = _topic_log_joint(h)
    m = maximum(lp)
    isinf(m) && return zeros(length(lp))
    w = exp.(lp .- m)
    return w ./ sum(w)
end

function _topic_log_joint(h)
    length(h) == TOPIC_BAGS_N_CLASSES || throw(ArgumentError("expected a count vector of length $TOPIC_BAGS_N_CLASSES (use topic_bags_counts on a digit vector), got length $(length(h))"))
    return [log(t.π) + _topic_loglik(t, h) for t in TOPIC_BAGS_TOPICS]
end
