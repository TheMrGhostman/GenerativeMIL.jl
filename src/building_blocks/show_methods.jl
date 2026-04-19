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

