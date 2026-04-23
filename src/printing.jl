AbstractTrees.children((name, m)::Tuple{String, Union{Flux.Dense, Flux.LayerNorm}}) = () # expand for all flux layers
AbstractTrees.printnode(io::IO, (name, m)::Tuple{String, Union{Flux.Dense, Flux.LayerNorm}}) = print(io, "$(name) -- $(m)")

AbstractTrees.children((name, m)::Tuple{String, Flux.Chain}) = (m) # expand for all flux layers
AbstractTrees.printnode(io::IO, (name, m)::Tuple{String, Flux.Chain}) = print(io, "$(name) -- Chain")

AbstractTrees.children((name, m)::Tuple{String, SplitLayer}) = (("μ", m.μ), ("σ", m.σ)) 
AbstractTrees.printnode(io::IO, (name, m)::Tuple{String, SplitLayer}) = print(io, "$(name) -- SplitLayer")

AbstractTrees.children((name, m)::Tuple{String, AbstractArray}) = () 
AbstractTrees.printnode(io::IO, (name, x)::Tuple{String, AbstractArray}) = print(io, "$(name) -- \
    $(size(x)) | $(typeof(x)) | mean~$(round(Flux.mean(x), digits=3)) | xᵢ≠0: $(sum(x .!= 0)) | n(x): $(prod(size(x))) ")