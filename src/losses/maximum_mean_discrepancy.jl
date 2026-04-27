"""
Compute unbiased MMD between two sample matrices `(d, n_samples)`.

By default uses a fast RBF implementation based on GEMM (`x' * y`), which is
efficient on GPU. Pass `kernel` to use a custom kernel fallback.
"""
function maximum_mean_discrepancy(
    x::AbstractMatrix{T},
    y::AbstractMatrix{T};
    sigma::Real = 1,
    kernel::Union{Nothing, Function} = nothing,
) where T<:AbstractFloat
    m = size(x, 2)
    n = size(y, 2)
    @assert m > 1 "MMD requires at least 2 samples in x for unbiased estimator"
    @assert n > 1 "MMD requires at least 2 samples in y for unbiased estimator"

    if isnothing(kernel)
        inv_two_sigma2 = inv(T(2) * T(sigma)^2)

        # Pairwise squared Euclidean distances via matrix multiplication.
        x2 = sum(abs2, x; dims=1)
        y2 = sum(abs2, y; dims=1)

        d_xx = max.(x2' .+ x2 .- T(2) .* (x' * x), zero(T))
        d_yy = max.(y2' .+ y2 .- T(2) .* (y' * y), zero(T))
        d_xy = max.(x2' .+ y2 .- T(2) .* (x' * y), zero(T))

        k_xx = exp.(-d_xx .* inv_two_sigma2)
        k_yy = exp.(-d_yy .* inv_two_sigma2)
        k_xy = exp.(-d_xy .* inv_two_sigma2)

        # For RBF, diagonal terms are exactly exp(0)=1, so no explicit diag extraction.
        sum_xx_offdiag = sum(k_xx) - T(m)
        sum_yy_offdiag = sum(k_yy) - T(n)

        return sum_xx_offdiag / (T(m) * T(m - 1)) +
               sum_yy_offdiag / (T(n) * T(n - 1)) -
               T(2) * sum(k_xy) / (T(m) * T(n))
    end

    # Generic fallback for user-provided kernels.
    k_xx = kernel(x, x)
    k_yy = kernel(y, y)
    k_xy = kernel(x, y)

    diag_xx = sum(@view k_xx[diagind(k_xx)])
    diag_yy = sum(@view k_yy[diagind(k_yy)])

    return (sum(k_xx) - diag_xx) / (T(m) * T(m - 1)) +
           (sum(k_yy) - diag_yy) / (T(n) * T(n - 1)) -
           T(2) * sum(k_xy) / (T(m) * T(n))
end


"""
Compute MMD for batched GPU tensors `(d, n_samples, bs)`.

For default RBF mode (`kernel=nothing`), computation is fully vectorized across
batch dimension (no Julia loop) using batched GEMM.

For custom kernels, a per-batch fallback loop is used.
"""
function maximum_mean_discrepancy(
    x::CuArray{T, 3},
    y::CuArray{T, 3};
    sigma::Real = 1,
    kernel::Union{Nothing, Function} = nothing,
) where T<:AbstractFloat
    @assert size(x, 3) == size(y, 3) "x and y must have the same batch size"
    m = size(x, 2)
    n = size(y, 2)
    @assert m > 1 "MMD requires at least 2 samples in x for unbiased estimator"
    @assert n > 1 "MMD requires at least 2 samples in y for unbiased estimator"

    if isnothing(kernel)
        # Fully batched GPU path (no per-batch Julia loop).
        inv_two_sigma2 = inv(T(2) * T(sigma)^2)

        # Shapes:
        # x: (d, m, bs), y: (d, n, bs)
        # x_t: (m, d, bs), y_t: (n, d, bs)
        x_t = permutedims(x, (2, 1, 3))
        y_t = permutedims(y, (2, 1, 3))

        x2 = sum(abs2, x; dims=1)                  # (1, m, bs)
        y2 = sum(abs2, y; dims=1)                  # (1, n, bs)
        x2_t = permutedims(x2, (2, 1, 3))          # (m, 1, bs)
        y2_t = permutedims(y2, (2, 1, 3))          # (n, 1, bs)

        # Batched Gram matrices via GEMM on GPU.
        g_xx = Flux.batched_mul(x_t, x)            # (m, m, bs)
        g_yy = Flux.batched_mul(y_t, y)            # (n, n, bs)
        g_xy = Flux.batched_mul(x_t, y)            # (m, n, bs)

        d_xx = max.(x2_t .+ x2 .- T(2) .* g_xx, zero(T))
        d_yy = max.(y2_t .+ y2 .- T(2) .* g_yy, zero(T))
        d_xy = max.(x2_t .+ y2 .- T(2) .* g_xy, zero(T))

        k_xx = exp.(-d_xx .* inv_two_sigma2)
        k_yy = exp.(-d_yy .* inv_two_sigma2)
        k_xy = exp.(-d_xy .* inv_two_sigma2)

        # Sum over sample-pair dimensions, keep batch dimension.
        sum_xx_offdiag = sum(k_xx; dims=(1, 2)) .- T(m)
        sum_yy_offdiag = sum(k_yy; dims=(1, 2)) .- T(n)
        sum_xy = sum(k_xy; dims=(1, 2))

        mmd_per_batch = sum_xx_offdiag ./ (T(m) * T(m - 1)) .+
                        sum_yy_offdiag ./ (T(n) * T(n - 1)) .-
                        T(2) .* sum_xy ./ (T(m) * T(n))

        return sum(mmd_per_batch) / T(size(x, 3))
    end

    # Custom-kernel fallback.
    bs = size(x, 3)
    acc = zero(T)
    for b in 1:bs
        acc += maximum_mean_discrepancy(@view(x[:, :, b]), @view(y[:, :, b]); sigma=sigma, kernel=kernel)
    end
    return acc / T(bs)
end