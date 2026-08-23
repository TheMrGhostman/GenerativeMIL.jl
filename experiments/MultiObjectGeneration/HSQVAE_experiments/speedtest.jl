using Random, Statistics, Zygote, CUDA, Flux, BenchmarkTools
Random.seed!(1)

function _pairwise_sqdist_batched(x::AbstractArray{T,3}, y::AbstractArray{T,3}) where T
    x2 = sum(abs2, x; dims=1); y2 = sum(abs2, y; dims=1)
    x_t = permutedims(x, (2,1,3))
    g = Flux.batched_mul(x_t, y)
    x2_t = permutedims(x2, (2,1,3))
    return max.(x2_t .+ y2 .- T(2) .* g, zero(T))
end

# ---------- Version A: argmin + Zygote.@ignore + gather (the "backward-optimized" version) ----------
function _nn_indices_clusters(x::AbstractArray{T,4}, y::AbstractArray{T,4}) where T
    D, N, L, BS = size(x); _, N2, M, _ = size(y)
    xr = reshape(x, D, N*L, BS); yr = reshape(y, D, N*M, BS)
    P  = _pairwise_sqdist_batched(xr, yr)
    P5 = reshape(P, N, L, N, M, BS)
    nb = getindex.(dropdims(argmin(P5, dims=3), dims=3), 3)
    na = getindex.(dropdims(argmin(P5, dims=1), dims=1), 1)
    na = permutedims(na, (2,1,3,4))
    return nb, na
end

function chamfer_distance_clusters_A(x::AbstractArray{T,4}, y::AbstractArray{T,4}) where T
    D, N, L, BS = size(x); _, _, M, _ = size(y)
    ci_y, ci_x = Zygote.@ignore begin
        nb, na = _nn_indices_clusters(x, y)
        mgrid, bsgrid, lgrid = reshape(1:M,1,1,M,1), reshape(1:BS,1,1,1,BS), reshape(1:L,1,L,1,1)
        CartesianIndex.(nb, mgrid, bsgrid), CartesianIndex.(na, lgrid, bsgrid)
    end
    B_matched   = y[:, ci_y]
    dist_A_to_B = dropdims(mean(sum(abs2, reshape(x,D,N,L,1,BS) .- B_matched; dims=1); dims=2), dims=(1,2))
    A_matched   = x[:, ci_x]
    dist_B_to_A = dropdims(mean(sum(abs2, A_matched .- reshape(y,D,N,1,M,BS); dims=1); dims=2), dims=(1,2))
    return dist_A_to_B .+ dist_B_to_A
end

# ---------- Version B: plain differentiable minimum(dims=k), no indices at all ----------
function chamfer_distance_clusters_B(x::AbstractArray{T,4}, y::AbstractArray{T,4}) where T
    D, N, L, BS = size(x); _, _, M, _ = size(y)
    xr = reshape(x, D, N*L, BS); yr = reshape(y, D, N*M, BS)
    P  = _pairwise_sqdist_batched(xr, yr)
    P5 = reshape(P, N, L, N, M, BS)
    dist_A_to_B = dropdims(mean(dropdims(minimum(P5, dims=3), dims=3), dims=1), dims=1)
    dist_B_to_A = dropdims(mean(dropdims(minimum(P5, dims=1), dims=1), dims=2), dims=2)
    return dist_A_to_B .+ dist_B_to_A
end

lossA(x, y) = sum(chamfer_distance_clusters_A(x, y))
lossB(x, y) = sum(chamfer_distance_clusters_B(x, y))

# small, safe sizes
D, N, L, M, BS = 8, 32, 6, 6, 4
X = rand(Float32, D,N,L,BS); Y = rand(Float32, D,N,M,BS)

println("sanity: valA=", lossA(X,Y), "  valB=", lossB(X,Y))

println("\n== CPU: Version A (argmin+gather) — forward+backward ==")
Zygote.gradient(lossA, X, Y)  # warmup/compile
display(@benchmark Zygote.gradient(lossA, $X, $Y) samples=30 seconds=20)

println("\n\n== CPU: Version B (plain minimum) — forward+backward ==")
Zygote.gradient(lossB, X, Y)  # warmup/compile
display(@benchmark Zygote.gradient(lossB, $X, $Y) samples=30 seconds=20)

if CUDA.functional()
    Xg, Yg = cu(X), cu(Y)
    gradA_sync() = CUDA.@sync Zygote.gradient(lossA, Xg, Yg)
    gradB_sync() = CUDA.@sync Zygote.gradient(lossB, Xg, Yg)

    println("\n\n== GPU: Version A (argmin+gather) — forward+backward, synced ==")
    gradA_sync()  # warmup/compile
    display(@benchmark $gradA_sync() samples=30 seconds=20)

    println("\n\n== GPU: Version B (plain minimum) — forward+backward, synced ==")
    gradB_sync()  # warmup/compile
    display(@benchmark $gradB_sync() samples=30 seconds=20)
end
