# Implementation of evaluation metrics for point clouds, including Chamfer Distance, Sinkhorn Divergence, and Maximum Mean Discrepancy (MMD) with RBF kernel.
# Implementations are essentially the some as the ones in the loss functions, slightly less optimized for gradient computation, but more optimized for evaluation (e.g. no gradient tracking, no extra allocations, etc.)
# They also produce vector of losses per sample not averaged over the batch. This is essential for evaluation but unnecessary for training. 
# instead of reimplementing original functions to accomodate this loss per sample we decided to make the evaluation functions separate from the loss functions.
# The slowdown would be negligible for single batch computation, but it would be slowing experiments by hours in total. Which is again not that bad, but when we need to evaluate many hyperparameter settings it would be a significan slowdown from project-wise perspective.

function chamfer_distance_eval(
    A::AbstractArray{T, 3}, 
    B::AbstractArray{T, 3},
    args...; #NOTE : I ignote x_mask for this function and instead i put args
    agg::Function=mean) where T<:AbstractFloat

    nn_for_A, nn_for_B = Zygote.@ignore _nearest_neighbors(A, B)

    dist_A_to_B = dropdims(agg(sum((A .- B[:, nn_for_A]) .^ 2, dims=1), dims=2), dims=(1,2))
    dist_B_to_A = dropdims(agg(sum((B .- A[:, nn_for_B]) .^ 2, dims=1), dims=2), dims=(1,2))

    return dist_A_to_B .+ dist_B_to_A
end

function sinkhorn_divergence_loss_eval(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}, ε::T;kwargs...) where T<:AbstractFloat
    # Compute cost matrices
    Cxy = _pairwise_sqdist_batched(x, y)
    Cxx = _pairwise_sqdist_batched(x, x)
    Cyy = _pairwise_sqdist_batched(y, y)
    
    n_x = size(x, 2)
    n_y = size(y, 2)
    bs = size(x, 3)
    
    # Compute transport plans (offline, not differentiated)
    πxy = compute_transport_plans(Cxy, n_x, n_y, ε, OptimalTransport.SinkhornGibbs(); kwargs...)
    πxx = compute_transport_plans(Cxx, n_x, ε, OptimalTransport.SymmetricSinkhornGibbs(); kwargs...)
    πyy = compute_transport_plans(Cyy, n_y, ε, OptimalTransport.SymmetricSinkhornGibbs(); kwargs...)
    
    #@show size(πxy), size(πxy .* Cxy)
    # Compute divergence from plans
    div_xy = sum(πxy .* Cxy, dims = (1,2))# /bs
    div_xx = sum(πxx .* Cxx, dims = (1,2))# / bs
    div_yy = sum(πyy .* Cyy, dims = (1,2))# / bs
    
    loss = dropdims(div_xy - T(0.5) * (div_xx + div_yy), dims=(1,2))
    
    return loss
end


function density_aware_chamfer_distance_eval(x::AbstractArray{T,3}, y::AbstractArray{T,3}, α::AbstractFloat=1f0) where T<:AbstractFloat
    # Compute pairwise squared distances
    ỹᵢ, x̃ᵢ = Zygote.@ignore _nearest_neighbors(x, y)

    ny = Zygote.@ignore device_like(x,_contributions(ỹᵢ)) # (N, BS) -> (1, N, BS)
    nx = Zygote.@ignore device_like(y,_contributions(x̃ᵢ)) # (M, BS) -> (1, M, BS)
    
    d_x = sum((x .- y[:, ỹᵢ]) .^ 2, dims=1) # (D, N, BS) -> (1,N,BS) 
    d_y = sum((y .- x[:, x̃ᵢ]) .^ 2, dims=1) # (D, M, BS) -> (1,M,BS)  # we assume that N=M to reflect paper

    d_x = T(1) .- exp.(-α .* d_x) ./ (ny .+ eps(T)) # (1, N, BS)
    d_y = T(1) .- exp.(-α .* d_y) ./ (nx .+ eps(T)) # (1, M, BS)

    dcd = T(0.5) .* (mean(d_x, dims=(1,2)) + mean(d_y, dims=(1,2))) # mean is over N -> (1, 1, BS)
    return dropdims(dcd, dims=(1,2))
end



function _diag_batched(a::AbstractArray{T, 3}) where T<:AbstractFloat
    @assert size(a, 1) == size(a, 2) "Diagonal sum expects square matrices per batch"
    bs = size(a, 3)
    s = zeros(T, 1, 1, bs)
    for b in 1:bs
        s[1,1,b] = sum(@view a[:, :, b][diagind(@view a[:, :, b])])
    end
    return device_like(a,s)
end


function maximum_mean_discrepancy_rbf_eval(
    x::AbstractArray{T, 3},
    y::AbstractArray{T, 3};
    sigma::_SigmaArg = 1,
) where T<:AbstractFloat
    @assert size(x, 3) == size(y, 3) "x and y must have the same batch size"
    m = size(x, 2)
    n = size(y, 2)
    bs = size(x, 3)
    @assert m > 1 "MMD requires at least 2 samples in x for unbiased estimator"
    @assert n > 1 "MMD requires at least 2 samples in y for unbiased estimator"
    
    d_xx = _pairwise_sqdist_batched(x, x)
    d_yy = _pairwise_sqdist_batched(y, y)
    d_xy = _pairwise_sqdist_batched(x, y)

    
    k_xx = _apply_rbf_kernel(d_xx, sigma)
    k_yy = _apply_rbf_kernel(d_yy, sigma)
    k_xy = _apply_rbf_kernel(d_xy, sigma)

    #@info size.([k_xx, k_yy, k_xy])
    sum_xx_offdiag = sum(k_xx, dims=(1,2)) .- _diag_batched(k_xx)
    sum_yy_offdiag = sum(k_yy, dims=(1,2)) .- _diag_batched(k_yy)
    sum_xy = sum(k_xy, dims=(1,2))

    #@info size.([sum_xx_offdiag, sum_yy_offdiag, sum_xy])
    mmd = sum_xx_offdiag ./ (T(m) * T(m - 1)) + sum_yy_offdiag ./ (T(n) * T(n - 1)) - T(2) * sum_xy ./ (T(m) * T(n))
    return dropdims(mmd, dims=(1,2))

end