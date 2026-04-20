function _print_indented_text_plain(io::IO, x::Any, indent::AbstractString="\t")
    rendered = sprint(show, MIME"text/plain"(), x)
    for line in split(chomp(rendered), '\n')
        print(io, indent, line, '\n')
    end
end

# MultiheadAttention Block ------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", m::MultiheadAttentionBlock)
    styled_io = IOContext(io, :color => true)
    layers = [("Multihead", m.Multihead), ("FF", m.FF), ("LN1", m.LN1), ("LN2", m.LN2)]

    total_params = 0
    total_bytes = 0
    total_arrays = 0

    print(io, "MultiheadAttentionBlock(\n")

    for (name, layer) in layers
        trainables = collect(Flux.trainables(layer))
        params = sum(length, trainables)
        bytes = sum(length(p) * sizeof(eltype(p)) for p in trainables)

        total_params += params
        total_bytes += bytes
        total_arrays += length(trainables)

        print(io, "  $name:\n")
        _print_indented_text_plain(io, layer, "\t")
        #Base.printstyled(styled_io, "\n    # $params parameters\n"; color=:light_black)
    end

    Base.printstyled(styled_io, ")  # Total: $total_arrays arrays, $total_params parameters, $total_bytes bytes."; color=:light_black)
end

AbstractTrees.children(m::MultiheadAttentionBlock) = (m.Multihead, ("FeedForward", m.FF), ("LayerNorm 1", m.LN1), ("LayerNorm 2", m.LN2))
AbstractTrees.printnode(io::IO, m::MultiheadAttentionBlock) = print(io, "MultiheadAttentionBlock")



# InducedSetAttentionBlock ------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", m::InducedSetAttentionBlock)
    styled_io = IOContext(io, :color => true)
    trainables = collect(Flux.trainables(m))
    total_params = sum(length, trainables)
    total_bytes = sum(length(p) * sizeof(eltype(p)) for p in trainables)
    total_arrays = length(trainables)

    print(io, "InducedSetAttentionBlock(\n")
    print(io, "  MAB1:\n")
    _print_indented_text_plain(io, m.MAB1, "\t")
    print(io, "  MAB2: \n")
    _print_indented_text_plain(io, m.MAB2, "\t")
    println(io)
    println(io, "  I: $(size(m.I)) :: $(typeof(m.I))")

    Base.printstyled(styled_io, ")  # Total: $total_arrays arrays, $total_params parameters, $total_bytes bytes."; color=:light_black)
end

AbstractTrees.children(m::InducedSetAttentionBlock) = (m.MAB1, m.MAB2, ("Induced Set", m.I))
AbstractTrees.printnode(io::IO, m::InducedSetAttentionBlock) = print(io, "InducedSetAttentionBlock - ($(size(m.I,2)) Induced Sets)")


# InducedSetAttentionHalfBlock --------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", m::InducedSetAttentionHalfBlock)
    styled_io = IOContext(io, :color => true)
    trainables = collect(Flux.trainables(m))
    total_params = sum(length, trainables)
    total_bytes = sum(length(p) * sizeof(eltype(p)) for p in trainables)
    total_arrays = length(trainables)

    print(io, "InducedSetAttentionHalfBlock(\n")
    print(io, "  MAB1:\n")
    _print_indented_text_plain(io, m.MAB1, "\t")
    println(io, "  I: $(size(m.I)) :: $(typeof(m.I))")

    Base.printstyled(styled_io, ")  # Total: $total_arrays arrays, $total_params parameters, $total_bytes bytes."; color=:light_black)
end

AbstractTrees.children(m::InducedSetAttentionHalfBlock) = (m.MAB1, ("Induced Set", m.I))
AbstractTrees.printnode(io::IO, m::InducedSetAttentionHalfBlock) = print(io, "InducedSetAttentionHalfBlock - ($(size(m.I,2)) Induced Sets)")



# VariationalBottleneck ---------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", m::VariationalBottleneck)
    styled_io = IOContext(io, :color => true)
    trainables = collect(Flux.trainables(m))
    total_params = sum(length, trainables)
    total_bytes = sum(length(p) * sizeof(eltype(p)) for p in trainables)
    total_arrays = length(trainables)

    print(io, "VariationalBottleneck(\n")
    print(io, "  p(zˡ | z(<l) ):\n")
    _print_indented_text_plain(io, m.prior, "\t")
    print(io, "  q(zˡ | z(<l), x ): \n")
    _print_indented_text_plain(io, m.posterior, "\t")
    print(io, "  p(h | zˡ, z(<l), x): \n")
    _print_indented_text_plain(io, m.decoder, "\t")

    Base.printstyled(styled_io, ")  # Total: $total_arrays arrays, $total_params parameters, $total_bytes bytes."; color=:light_black)
end

AbstractTrees.children(m::VariationalBottleneck) = (("Prior", m.prior), ("Posterior", m.posterior), ("Decoder", m.decoder))
AbstractTrees.printnode(io::IO, m::VariationalBottleneck) = print(io, "VariationalBottleneck - ($(size(m.decoder[1].weight, 2)) zdim)")


# AttentiveBottleneckLayer ------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", m::AttentiveBottleneckLayer)
    styled_io = IOContext(io, :color => true)
    trainables = collect(Flux.trainables(m))
    total_params = sum(length, trainables)
    total_bytes = sum(length(p) * sizeof(eltype(p)) for p in trainables)
    total_arrays = length(trainables)

    print(io, "AttentiveBottleneckLayer(\n")
    print(io, "  MAB1:\n")
    _print_indented_text_plain(io, m.MAB1, "\t")
    print(io, "  MAB2:\n")
    _print_indented_text_plain(io, m.MAB2, "\t")
    print(io, "  VB:\n")
    _print_indented_text_plain(io, m.VB, "\t")
    println(io, "  I: $(size(m.I)) :: $(typeof(m.I))")

    Base.printstyled(styled_io, ")  # Total: $total_arrays arrays, $total_params parameters, $total_bytes bytes."; color=:light_black)
end


AbstractTrees.children(m::AttentiveBottleneckLayer) = (m.MAB1, m.MAB2, m.VB, ("Induced Sets", m.I))
AbstractTrees.printnode(io::IO, m::AttentiveBottleneckLayer) = print(io, "AttentiveBottleneckLayer - ($(size(m.I,2)) Induced Sets)")


# AttentiveHalfBlock ------------------------------------------------------------
function Base.show(io::IO, ::MIME"text/plain", m::AttentiveHalfBlock)
    styled_io = IOContext(io, :color => true)
    trainables = collect(Flux.trainables(m))
    total_params = sum(length, trainables)
    total_bytes = sum(length(p) * sizeof(eltype(p)) for p in trainables)
    total_arrays = length(trainables)

    print(io, "AttentiveHalfBlock(\n")
    print(io, "  MAB1:\n")
    _print_indented_text_plain(io, m.MAB1, "\t")
    print(io, "  VB:\n")
    _print_indented_text_plain(io, m.VB, "\t")

    Base.printstyled(styled_io, ")  # Total: $total_arrays arrays, $total_params parameters, $total_bytes bytes."; color=:light_black)
end


AbstractTrees.children(m::AttentiveHalfBlock) = (m.MAB1, m.VB)
AbstractTrees.printnode(io::IO, m::AttentiveHalfBlock) = print(io, "AttentiveHalfBlock")


# MultiheadAttention ------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", m::MultiheadAttention{F}) where F
    attention_name = m.attention === attention ? "standard" : 
                     m.attention === slot_attention ? "slot" : 
                     "attention (custom)"
    styled_io = IOContext(io, :color => true)
    
    print(io, "MultiheadAttention{$(F)}\n")
    print(io, "  Heads: $(m.heads), Attention: $attention_name\n\n")
    
    # Počítej parametry pro každou vrstvu
    layers = [("WQ", m.WQ), ("WK", m.WK), ("WV", m.WV), ("WO", m.WO)]
    total_params = 0
    total_bytes = 0
    num_arrays = 0
    
    for (name, layer) in layers
        params = length(layer.weight) # no bias
        bytes = params * sizeof(eltype(layer.weight))
        total_params += params
        total_bytes += bytes
        num_arrays += 1
        
        print(io, "  $name: $(layer)")
        Base.printstyled(styled_io, ", # $params parameters\n"; color=:light_black)
    end
    
    Base.printstyled(styled_io, ")  # Total: $num_arrays arrays, $total_params parameters, $total_bytes bytes."; color=:light_black)
end

AbstractTrees.children(m::MultiheadAttention) = (("W_Query ", m.WQ), ("W_Key ", m.WK), ("W_Value ", m.WV), ("W_Output", m.WO), m.attention)
AbstractTrees.printnode(io::IO, m::MultiheadAttention) = print(io, "MultiheadAttention - ($(m.heads) heads)")


# MixtureOfGaussians ------------------------------------------------------------

function _show_param_counts(io::IO, names::NTuple{N, String}, arrays::NTuple{N, Any}) where N
    counts = map(length, arrays)
    total = sum(counts)
    for i in eachindex(names)
        print(io, "\n\t   - ", names[i], " = ", counts[i])
    end
    print(io, "\n\t   - total = ", total)
end

_mean_or_nan(x) = isempty(x) ? NaN : Flux.mean(x)


function Base.show(io::IO, ::MIME"text/plain", m::MixtureOfGaussians)
    styled_io = IOContext(io, :color => true)
    Ds, K, _ = size(m.μ)
    ptotal = length(m.α) + length(m.μ) + length(m.Σ)
    ptrainable = m.trainable ? ptotal : 0
    trainables = collect(Flux.trainables(m))
    total_arrays = length(trainables)
    total_params = sum(length, trainables)
    total_bytes = sum(length(p) * sizeof(eltype(p)) for p in trainables)

    print(io, "MixtureOfGaussians(")
    print(io, "\n\t - shape: Ds=$(Ds), K=$(K)")
    print(io, "\n\t - trainable: $(m.trainable)")
    print(io, "\n\t - α: size=$(size(m.α)) | type=$(typeof(m.α)) | mean=$(_mean_or_nan(m.α))")
    print(io, "\n\t - μ: size=$(size(m.μ)) | type=$(typeof(m.μ)) | mean=$(_mean_or_nan(m.μ))")
    print(io, "\n\t - Σ: size=$(size(m.Σ)) | type=$(typeof(m.Σ)) | mean=$(_mean_or_nan(m.Σ))")
    print(io, "\n\t - parameters:")
        _show_param_counts(io, ("α", "μ", "Σ"), (m.α, m.μ, m.Σ))
    print(io, "\n\t   - trainable = $(ptrainable)")

    Base.printstyled(styled_io, "\n)  # Total: $total_arrays arrays, $total_params parameters, $total_bytes bytes."; color=:light_black)
end

#Base.show(io::IO, m::MixtureOfGaussians) = show(io, MIME"text/plain"(), m)

AbstractTrees.children(m::MixtureOfGaussians) = (("α", m.α), ("μ", m.μ), ("Σ", m.Σ))
AbstractTrees.printnode(io::IO, m::MixtureOfGaussians) = print(io, "MixtureOfGaussians - (mixtures ~ $(size(m.μ, 2)) | dim ~ $(size(m.μ, 1)) | trainable: $(m.trainable))")

# ConstGaussPrior ---------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", m::ConstGaussPrior)
    styled_io = IOContext(io, :color => true)
    dim, n_slots, _ = size(m.μ)
    trainables = collect(Flux.trainables(m))
    total_arrays = length(trainables)
    total_params = sum(length, trainables)
    total_bytes = sum(length(p) * sizeof(eltype(p)) for p in trainables)

    print(io, "ConstGaussPrior(")
    print(io, "\n\t - shape: dim=$(dim), n_slots=$(n_slots)")
    print(io, "\n\t - μ: size=$(size(m.μ)) | type=$(typeof(m.μ)) | mean=$(_mean_or_nan(m.μ))")
    print(io, "\n\t - Σ: size=$(size(m.Σ)) | type=$(typeof(m.Σ)) | mean=$(_mean_or_nan(m.Σ))")
    print(io, "\n\t - parameters:")
        _show_param_counts(io, ("μ", "Σ"), (m.μ, m.Σ))

    Base.printstyled(styled_io, "\n)  # Total: $total_arrays arrays, $total_params parameters, $total_bytes bytes."; color=:light_black)
end

#Base.show(io::IO, m::ConstGaussPrior) = show(io, MIME"text/plain"(), m)


AbstractTrees.children(m::ConstGaussPrior) = (("μ", m.μ), ("Σ", m.Σ))
AbstractTrees.printnode(io::IO, m::ConstGaussPrior) = print(io, "ConstGaussPrior - (n_slots=$(size(m.μ,2)) | dim=$(size(m.μ,1)))")