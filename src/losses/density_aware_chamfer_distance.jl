#REVIEW: #TODO: make it gpu ready. now i assume that nx and ny will cause problems on gpu.

function density_aware_chamfer_distance(x::AbstractArray{T,3}, y::AbstractArray{T,3}, α::AbstractFloat=1f0) where T<:AbstractFloat
    # Compute pairwise squared distances
    ỹᵢ, x̃ᵢ = Zygote.@ignore _nearest_neighbors(x, y)

    ny = Zygote.@ignore device_like(x,_contributions(ỹᵢ)) # (N, BS) -> (1, N, BS)
    nx = Zygote.@ignore device_like(y,_contributions(x̃ᵢ)) # (M, BS) -> (1, M, BS)
    
    d_x = sum((x .- y[:, ỹᵢ]) .^ 2, dims=1) # (D, N, BS) -> (1,1,BS) 
    d_y = sum((y .- x[:, x̃ᵢ]) .^ 2, dims=1) # (D, M, BS) -> (1,1,BS)  # we assume that N=M to reflect paper

    d_x = T(1) .- exp.(-α .* d_x) ./ (ny .+ eps(T)) # (1, N, BS)
    d_y = T(1) .- exp.(-α .* d_y) ./ (nx .+ eps(T)) # (1, M, BS)

    dcd = T(0.5) .* (mean(d_x) + mean(d_y))
end

function _contributions(idx::AbstractArray)
    x = zeros(Float32, 1, size(idx)...)
    @inbounds for i in eachindex(idx)
        x[1, idx[i]] += 1f0
    end
    return x[:,idx]
end