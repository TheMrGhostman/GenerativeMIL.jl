mutable struct MMD_EMA_Loss{T<:AbstractFloat, S} 
    mmd_fn::Function  # function to compute MMD, e.g., maximum_mean_discrepancy; this contain kernel already
    σ_scales::S       # this can be a scalar or a vector of scales for multi-scale MMD
    σᵣ::T             # running estimate of the "right" σ scale, updated via EMA
    decay::T          # decay factor for EMA update of σᵣ
end

(loss::MMD_EMA_Loss)(x, y; kwargs...) = loss.mmd_fn(x, y; sigma=loss.σ_scales .* loss.σᵣ, kwargs...)

function update_ema_sigma!(loss::MMD_EMA_Loss, σₙ::T) where T<:AbstractFloat
    loss.σᵣ = loss.decay * loss.σᵣ + (T(1) - loss.decay) * σₙ
end

compute_rbf_sigma_estimate(x, y; kwargs...) = sqrt(median(sum(abs2, x .- y, dims=1) / 2))  

