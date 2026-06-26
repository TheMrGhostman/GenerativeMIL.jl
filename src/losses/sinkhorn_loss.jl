
using ChainRulesCore # TODO remove

uniform_priors(::CuArray{T}, n::Int) where T <: AbstractFloat = cu(fill(T(1) ./ n, n))
uniform_priors(::Array{T}, n::Int) where T <: AbstractFloat = fill(T(1) ./ n, n)

"""
Compute transport plans for a batch using Sinkhorn algorithm.

Arguments
- `x::AbstractArray{T, 3}`: Input array of shape (d, n_x, bs)
- `y::AbstractArray{T, 3}`: Input array of shape (d, n_y, bs)  
- `C::AbstractArray{T, 3}`: Cost matrices of shape (n_x, n_y, bs)
- `ε::T`: Entropic regularization parameter
- `kwargs`: Additional arguments for sinkhorn

Returns
- Transport plans π of shape (n_x, n_y, bs)
"""
function compute_transport_plans(
    C::AbstractArray{T, 3}, 
    n_x::Int, 
    n_y::Int,
    ε::T, 
    alg=SinkhornGibbs();
    kwargs...
) where T<:AbstractFloat
    # uniform priors
    μ = uniform_priors(C, n_x)#fill(1f0 / n_x, n_x)
    ν = uniform_priors(C, n_y)#fill(1f0 / n_y, n_y)
    # pre-allocate plans
    plans = MLUtils.zeros_like(C)
    # Compute transport plans for each batch slice
    for b ∈ axes(C, 3)
        plans[:,:,b] .= OptimalTransport.sinkhorn(μ, ν, C[:,:, b], ε, alg; kwargs...)
    end
    
    return plans
end

function compute_transport_plans(
    C::AbstractArray{T, 3}, 
    n_x::Int, 
    ε::T, 
    alg=SymmetricSinkhornGibbs();
    kwargs...
) where T<:AbstractFloat
    # prior is uniform 
    μ = uniform_priors(C, n_x)
    # pre-allocate plans
    plans = MLUtils.zeros_like(C)

    # Compute transport plans for each batch slice
    for b ∈ axes(C, 3)
        plans[:,:,b] .= OptimalTransport.sinkhorn(μ, C[:,:, b], ε, alg; kwargs...)
    end
    
    return plans
end

"""
Compute Sinkhorn divergence loss with differentiable gradients via custom rrule.

This implementation computes transport plans offline and then uses them to 
calculate gradients, allowing Zygote to differentiate through the loss.

Arguments
- `x::AbstractArray{T, 3}`: Samples with shape (d, n_x, bs)
- `y::AbstractArray{T, 3}`: Samples with shape (d, n_y, bs)
- `ε::T`: Entropic regularization parameter
- `kwargs`: Additional arguments for sinkhorn (maxiter, tol, etc.)

Returns
- Scalar loss value
"""
function sinkhorn_divergence_loss(x::AbstractArray{T, 3}, y::AbstractArray{T, 3}, ε::T; regularization::Bool=false, kwargs...) where T<:AbstractFloat
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
    
    # Compute divergence from plans
    div_xy = sum(πxy .* Cxy) / bs
    div_xx = sum(πxx .* Cxx) / bs
    div_yy = sum(πyy .* Cyy) / bs
    
    loss = div_xy - T(0.5) * (div_xx + div_yy)
    
    return loss
end


"""
Custom rrule for sinkhorn_divergence_loss that uses precomputed transport plans.

Key insight: Transport plans π don't need Zygote.ignore() because the pullback 
function defines the gradient flow explicitly. Only cost matrices (C) flow gradients 
back to x and y. The rrule returns (NoTangent(), ∇x, ∇y, NoTangent()), which tells 
Zygote to stop tracing through the transport plan computation.
"""
function ChainRulesCore.rrule(
    ::typeof(sinkhorn_divergence_loss), 
    x::AbstractArray{T, 3}, 
    y::AbstractArray{T, 3}, 
    ε::T; 
    regularization::Bool=false,
    kwargs...
) where T<:AbstractFloat
    
    # Forward pass - cost matrices are differentiated
    Cxy = _pairwise_sqdist_batched(x, y)
    Cxx = _pairwise_sqdist_batched(x, x)
    Cyy = _pairwise_sqdist_batched(y, y)
    
    n_x = size(x, 2)
    n_y = size(y, 2)
    bs = size(x, 3)
    
    # Compute transport plans - simply call the function
    # Zygote.ignore() is NOT needed here because we have custom rrule
    # that defines the pullback behavior explicitly
    πxy = compute_transport_plans(Cxy, n_x, n_y, ε, OptimalTransport.SinkhornGibbs(); kwargs...)
    πxx = compute_transport_plans(Cxx, n_x, ε, OptimalTransport.SymmetricSinkhornGibbs(); kwargs...)
    πyy = compute_transport_plans(Cyy, n_y, ε, OptimalTransport.SymmetricSinkhornGibbs(); kwargs...)
    
    # Compute loss
    div_xy = sum(πxy .* Cxy) / bs
    div_xx = sum(πxx .* Cxx) / bs
    div_yy = sum(πyy .* Cyy) / bs
    
    loss = div_xy - T(0.5) * (div_xx + div_yy)
    
    # Pullback function
    function sinkhorn_divergence_loss_pullback(∇loss)
        # Scale by batch size normalization
        scale = ∇loss / bs
        
        # Gradients w.r.t. cost matrices (through transport plans)
        ∇Cxy = scale * πxy
        ∇Cxx = -T(0.5) * scale * πxx
        ∇Cyy = -T(0.5) * scale * πyy
        
        # Vectorized gradients w.r.t. inputs (GPU-friendly, no for-loops)
        ∇x = _gradient_x_vectorized(x, y, ∇Cxy, ∇Cxx)
        ∇y = _gradient_y_vectorized(x, y, ∇Cxy, ∇Cyy)
        
        return (NoTangent(), ∇x, ∇y, NoTangent())
    end
    
    return loss, sinkhorn_divergence_loss_pullback
end


"""
Vectorized gradient w.r.t. x - fully tensorized, GPU-optimized, no for-loops.

For OT(x,y): ∇x_i = 2 * sum_j ∇C_xy[i,j] * (x_i - y_j)
For OT(x,x): ∇x_i = 2 * sum_j (x_i - x_j) * (∇C_xx[i,j] + ∇C_xx[j,i])
"""
function _gradient_x_vectorized(
    x::AbstractArray{T, 3}, 
    y::AbstractArray{T, 3},
    ∇Cxy::AbstractArray{T, 3},
    ∇Cxx::AbstractArray{T, 3}
) where T<:AbstractFloat
    
    d, n_x, bs = size(x)
    _, n_y, _ = size(y)
    
    # ========== OT(x,y) term ==========
    # ∇x += 2 * (x * sum_j(∇C_xy[i,j]) - y @ ∇C_xy^T)
    
    # sum_j ∇C_xy[i,j] for all i
    sum_grad_xy = sum(∇Cxy, dims=2)  # (n_x, 1, bs)
    sum_grad_xy = dropdims(sum_grad_xy, dims=2)     # (n_x, bs)
    sum_grad_xy_reshaped = reshape(sum_grad_xy, 1, n_x, bs)  # (1, n_x, bs) for broadcasting
    
    # x[:, :, b] .* sum_grad_xy[:, b] for all b
    grad_xy_x_contrib = x .* sum_grad_xy_reshaped  # (d, n_x, bs)
    
    # y @ ∇C_xy^T for each batch using batched matrix multiplication
    # y: (d, n_y, bs), ∇Cxy^T: (n_y, n_x, bs)
    ∇Cxy_T = permutedims(∇Cxy, (2, 1, 3))  # (n_y, n_x, bs)
    grad_xy_y_contrib = Flux.batched_mul(y, ∇Cxy_T)  # (d, n_y, bs) @ (n_y, n_x, bs) = (d, n_x, bs)
    
    ∇x_xy = T(2) .* (grad_xy_x_contrib .- grad_xy_y_contrib)
    
    # ========== OT(x,x) term ==========
    # ∇x += 2 * (x * sum_j((∇C_xx[i,j] + ∇C_xx[j,i])) - ∇C_xx_sym @ x^T)
    
    # Symmetrize: ∇C_xx_sym[i,j] = ∇C_xx[i,j] + ∇C_xx[j,i]
    ∇Cxx_sym = ∇Cxx .+ permutedims(∇Cxx, (2, 1, 3))  # (n_x, n_x, bs)
    
    # sum_j ∇C_xx_sym[i,j] for all i
    sum_grad_xx = sum(∇Cxx_sym, dims=2)  # (n_x, 1, bs)
    sum_grad_xx = dropdims(sum_grad_xx, dims=2)         # (n_x, bs)
    sum_grad_xx_reshaped = reshape(sum_grad_xx, 1, n_x, bs)  # (1, n_x, bs)
    
    # x[:, :, b] .* sum_grad_xx[:, b] for all b
    grad_xx_x_contrib = x .* sum_grad_xx_reshaped  # (d, n_x, bs)
    
    # ∇C_xx_sym @ x^T for each batch
    # ∇Cxx_sym: (n_x, n_x, bs), x^T: (n_x, d, bs)
    x_T = permutedims(x, (2, 1, 3))  # (n_x, d, bs)
    grad_xx_x_contrib_from_j = Flux.batched_mul(∇Cxx_sym, x_T)  # (n_x, n_x, bs) @ (n_x, d, bs) = (n_x, d, bs)
    grad_xx_x_contrib_from_j = permutedims(grad_xx_x_contrib_from_j, (2, 1, 3))  # (d, n_x, bs)
    
    ∇x_xx = T(2) .* (grad_xx_x_contrib .- grad_xx_x_contrib_from_j)
    
    return ∇x_xy .+ ∇x_xx
end


"""
Vectorized gradient w.r.t. y - fully tensorized, GPU-optimized, no for-loops.

For OT(x,y): ∇y_j = 2 * sum_i ∇C_xy[i,j] * (y_j - x_i)
For OT(y,y): ∇y_j = 2 * sum_i (y_j - y_i) * (∇C_yy[i,j] + ∇C_yy[j,i])
"""
function _gradient_y_vectorized(
    x::AbstractArray{T, 3}, 
    y::AbstractArray{T, 3},
    ∇Cxy::AbstractArray{T, 3},
    ∇Cyy::AbstractArray{T, 3}
) where T<:AbstractFloat
    
    d, n_x, bs = size(x)
    _, n_y, _ = size(y)
    
    # ========== OT(x,y) term ==========
    # ∇y += 2 * (y * sum_i(∇C_xy[i,j]) - x @ ∇C_xy)
    
    # sum_i ∇C_xy[i,j] for all j
    sum_grad_xy = sum(∇Cxy, dims=1)  # (1, n_y, bs)
    sum_grad_xy = dropdims(sum_grad_xy, dims=1)     # (n_y, bs)
    sum_grad_xy_reshaped = reshape(sum_grad_xy, 1, n_y, bs)  # (1, n_y, bs)
    
    # y[:, :, b] .* sum_grad_xy[:, b] for all b
    grad_xy_y_contrib = y .* sum_grad_xy_reshaped  # (d, n_y, bs)
    
    # x @ ∇C_xy for each batch
    # x^T: (n_x, d, bs), ∇Cxy: (n_x, n_y, bs)
    x_T = permutedims(x, (2, 1, 3))  # (n_x, d, bs)
    ∇Cxy_T = permutedims(∇Cxy, (2, 1, 3))  # (n_y, n_x, bs)
    grad_xy_x_contrib = Flux.batched_mul(∇Cxy_T, x_T)  # (n_y, n_x, bs) @ (n_x, d, bs) = (n_y, d, bs)
    grad_xy_x_contrib = permutedims(grad_xy_x_contrib, (2, 1, 3))  # (d, n_y, bs)
    
    ∇y_xy = T(2) .* (grad_xy_y_contrib .- grad_xy_x_contrib)
    
    # ========== OT(y,y) term ==========
    # ∇y += 2 * (y * sum_i((∇C_yy[i,j] + ∇C_yy[j,i])) - ∇C_yy_sym^T @ y^T)
    
    # Symmetrize: ∇C_yy_sym[i,j] = ∇C_yy[i,j] + ∇C_yy[j,i]
    ∇Cyy_sym = ∇Cyy .+ permutedims(∇Cyy, (2, 1, 3))  # (n_y, n_y, bs)
    
    # sum_i ∇C_yy_sym[i,j] for all j
    sum_grad_yy = sum(∇Cyy_sym, dims=1)  # (1, n_y, bs)
    sum_grad_yy = dropdims(sum_grad_yy, dims=1)         # (n_y, bs)
    sum_grad_yy_reshaped = reshape(sum_grad_yy, 1, n_y, bs)  # (1, n_y, bs)
    
    # y[:, :, b] .* sum_grad_yy[:, b] for all b
    grad_yy_y_contrib = y .* sum_grad_yy_reshaped  # (d, n_y, bs)
    
    # ∇C_yy_sym @ y^T for each batch
    # ∇Cyy_sym: (n_y, n_y, bs), y^T: (n_y, d, bs)
    y_T = permutedims(y, (2, 1, 3))  # (n_y, d, bs)
    grad_yy_y_contrib_from_i = Flux.batched_mul(∇Cyy_sym, y_T)  # (n_y, n_y, bs) @ (n_y, d, bs) = (n_y, d, bs)
    grad_yy_y_contrib_from_i = permutedims(grad_yy_y_contrib_from_i, (2, 1, 3))  # (d, n_y, bs)
    
    ∇y_yy = T(2) .* (grad_yy_y_contrib .- grad_yy_y_contrib_from_i)
    
    return ∇y_xy .+ ∇y_yy
end
