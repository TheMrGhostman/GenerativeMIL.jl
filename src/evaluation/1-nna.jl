function one_nn(distances::AbstractMatrix, labels::AbstractVector; exclude_self::Bool = false)
	@assert size(distances, 1) == length(labels)

	nquery = size(distances, 2)
	predictions = Vector{eltype(labels)}(undef, nquery)

	if exclude_self && size(distances, 1) == size(distances, 2)
		@inbounds for j in 1:nquery
			best_index = 0
			best_distance = Inf

			for i in axes(distances, 1)
				i == j && continue
				distance = distances[i, j]
				if distance < best_distance
					best_distance = distance
					best_index = i
				end
			end
			predictions[j] = labels[best_index]
		end
	else
		@inbounds for j in 1:nquery
			predictions[j] = labels[argmin(view(distances, :, j))]
		end
	end
	return predictions
end



function one_nn_accuracy(pdm::AbstractMatrix{T}, labels::AbstractVector; exclude_self::Bool = false) where T<: AbstractFloat
    nn_labels = one_nn(pdm, labels; exclude_self=exclude_self)
    return mean(nn_labels .== labels)
end