
using ChainRulesCore # TODO remove

uniform_priors(::CuArray{T}, n::Int) = cu(fill(T(1) ./ n, n))
uniform_priors(::Array{T}, n::Int) = fill(T(1) ./ n, n)

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

Transport plans are computed with Zygote.ignore() so they don't propagate gradients.
Only cost matrices flow gradients back to x and y, using vectorized operations.
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
    
    # Compute transport plans - wrapped in Zygote.ignore() to prevent gradient flow
    # These are auxiliary data, not parameters we want to differentiate
    πxy = Zygote.ignore(() -> compute_transport_plans(Cxy, n_x, n_y, ε, OptimalTransport.SinkhornGibbs(); kwargs...))()
    πxx = Zygote.ignore(() -> compute_transport_plans(Cxx, n_x, ε, OptimalTransport.SinkhornGibbs(); kwargs...))()
    πyy = Zygote.ignore(() -> compute_transport_plans(Cyy, n_y, ε, OptimalTransport.SinkhornGibbs(); kwargs...))()
    
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
Vectorized gradient w.r.t. x - GPU-friendly, no for-loops.

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
    ∇x = zeros(T, d, n_x, bs)
    
    # Process each batch
    for b in 1:bs
        x_b = x[:, :, b]
        y_b = y[:, :, b]
        ∇Cxy_b = ∇Cxy[:, :, b]
        ∇Cxx_b = ∇Cxx[:, :, b]
        
        # OT(x,y) term: ∇x_i = 2 * sum_j ∇C_xy[i,j] * (x_i - y_j)
        #                     = 2 * (sum_j ∇C_xy[i,j] * x_i - ∇C_xy * y')
        sum_grad_xy = vec(sum(∇Cxy_b, dims=2))  # (n_x,) - sum over j
        # Broadcasting: (d, n_x) .* (n_x,)' = (d, n_x)
        grad_xy_x_contrib = x_b .* sum_grad_xy'
        grad_xy_y_contrib = ∇Cxy_b * y_b'  # (n_x, n_y) * (n_y, d) = (n_x, d)
        ∇x[:, :, b] .+= T(2) .* (grad_xy_x_contrib .- grad_xy_y_contrib)
        
        # OT(x,x) term: symmetrize gradient matrix
        # ∇x_i = 2 * sum_j (x_i - x_j) * (∇C_xx[i,j] + ∇C_xx[j,i])
        ∇Cxx_sym = ∇Cxx_b .+ ∇Cxx_b'  # (n_x, n_x) - symmetric sum
        sum_grad_xx = vec(sum(∇Cxx_sym, dims=2))  # (n_x,) - sum over j
        grad_xx_x_contrib = x_b .* sum_grad_xx'  # (d, n_x)
        grad_xx_x_contrib_from_j = ∇Cxx_sym * x_b'  # (n_x, n_x) * (n_x, d) = (n_x, d)
        ∇x[:, :, b] .+= T(2) .* (grad_xx_x_contrib .- grad_xx_x_contrib_from_j)
    end
    
    return ∇x
end


"""
Vectorized gradient w.r.t. y - GPU-friendly, no for-loops.

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
    ∇y = zeros(T, d, n_y, bs)
    
    # Process each batch
    for b in 1:bs
        x_b = x[:, :, b]
        y_b = y[:, :, b]
        ∇Cxy_b = ∇Cxy[:, :, b]
        ∇Cyy_b = ∇Cyy[:, :, b]
        
        # OT(x,y) term: ∇y_j = 2 * sum_i ∇C_xy[i,j] * (y_j - x_i)
        #                     = 2 * (sum_i ∇C_xy[i,j] * y_j - ∇C_xy' * x)
        sum_grad_xy = vec(sum(∇Cxy_b, dims=1))  # (n_y,) - sum over i
        grad_xy_y_contrib = y_b .* sum_grad_xy'  # (d, n_y)
        grad_xy_x_contrib = ∇Cxy_b' * x_b'  # (n_y, n_x) * (n_x, d) = (n_y, d)
        ∇y[:, :, b] .+= T(2) .* (grad_xy_y_contrib .- grad_xy_x_contrib)
        
        # OT(y,y) term: symmetrize gradient matrix
        # ∇y_j = 2 * sum_i (y_j - y_i) * (∇C_yy[i,j] + ∇C_yy[j,i])
        ∇Cyy_sym = ∇Cyy_b .+ ∇Cyy_b'  # (n_y, n_y) - symmetric sum
        sum_grad_yy = vec(sum(∇Cyy_sym, dims=2))  # (n_y,) - sum over i
        grad_yy_y_contrib = y_b .* sum_grad_yy'  # (d, n_y)
        grad_yy_y_contrib_from_i = ∇Cyy_sym * y_b'  # (n_y, n_y) * (n_y, d) = (n_y, d)
        ∇y[:, :, b] .+= T(2) .* (grad_yy_y_contrib .- grad_yy_y_contrib_from_i)
    end
    
    return ∇y
end
