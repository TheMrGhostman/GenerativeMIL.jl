struct SetClassifier
    reduction # Dense or Chain / can be ommited with proper isab parameters
    isabs # list of InducedSetAttentionBlock(s)
    pooling # Pooling or InducedSetAttentionHalfBlock
    class # reduction to number of classes
    dropout :: Union{Flux.Dropout, Nothing} # Dropout around pooling
end

Flux.@functor SetClassifier

function (m::SetClassifier)(x::AbstractArray{<:Real}, x_mask::Mask=nothing)

    x = mask(m.reduction(x), x_mask)
    for layer in m.isabs
        x, _ = layer(x, x_mask)
        # we don't need Induced Set
    end
    x = (m.dropout !== nothing) ? m.dropout(x) : x
    _, x = m.pooling(x, x_mask)
    x = (m.dropout !== nothing) ? m.dropout(x) : x
    x = m.class(x)
    x = dropdims(x, dims=2) # drom empty dimension 
    return Flux.softmax(x)
end

function loss(
    m::SetClassifier, x::AbstractArray{T}, y::AbstractArray{<:Real}, x_mask::MaskT{T}=nothing) T<:Real
    
    ŷ = m(x, x_mask);
    loss_ = Flux.crossentropy(ŷ, y)
    return loss_
end

function SetClassifier(input_dim::Int, hidden_dim::Int, heads::Int, induced_set_sizes::Array{Int,1}, 
    reduction_depth::Int=1, classes::Int=10, dropout::Int=0, activation::Function=Flux.relu)
    # Reduction 
    reduction = nothing
    if reduction_depth > 1
        reduction = [
            Flux.Dense(input_dim, hidden_dim, activation)
        ]
        for i=1:reduction_depth-1
            push!(reduction, Flux.Dense(hidden_dim, hidden_dim, activation))
        end
        reduction = Flux.Chain(reduction...)
    else
        reduction = Flux.Dense(input_dim, hidden_dim, activation)
    end
    # isabs
    isabs = []
    for iss in induced_set_sizes
        isab = InducedSetAttentionBlock(iss, hidden_dim, heads)
        push!(isabs, isab)
    end
    # Classification part
    ## Pooling -> PMA from SetTransformer paper
    mh = MultiheadAttention(hidden_dim, hidden_dim, heads, attention)
    ff = Flux.Chain(
        Flux.Dense(hidden_dim, hidden_dim, activation),
        Flux.Dense(hidden_dim, hidden_dim, activation)
    )
    ln1 = Flux.LayerNorm((hidden_dim,1))
    ln2 = Flux.LayerNorm((hidden_dim,1))

    mab =  MultiheadAttentionBlock(ff, mh, ln1, ln2)
    I = randn(Float32, hidden_dim, 1) # keep batch size as free parameter
    pooling =  InducedSetAttentionHalfBlock(mab, I) 
    # TODO modifie parameters to get maybe more suitable output from halfblock ??

    dropout_ = (dropout <= 1 && dropout > 0) ? Flux.Dropout(dropout) : nothing
    class = Flux.Dense(hidden_dim, classes)
    return SetClassifier(reduction, isabs, pooling, class, dropout_)
end
  
