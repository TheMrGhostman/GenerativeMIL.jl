using Flux
using BenchmarkTools

function chamfer_sqdist_version(x::CuArray{Float32,3}, y::CuArray{Float32,3})
    xx = sum(x .^ 2, dims = 1)
    yy = sum(y .^ 2, dims = 1)
    zz = Flux.batched_mul(permutedims(x, (2, 1, 3)), y)
    rx = reshape(xx, size(xx, 2), 1, :)
    ry = reshape(yy, 1, size(yy, 2), :)
    P = (rx .+ ry) .- (2 .* zz)
    return P
end


function mmd_sqdist_version(x::CuArray{Float32,3}, y::CuArray{Float32,3})
    x_t = permutedims(x, (2, 1, 3))
    y_t = permutedims(y, (2, 1, 3))

    x2 = sum(abs2, x; dims=1)
    y2 = sum(abs2, y; dims=1)
    x2_t = permutedims(x2, (2, 1, 3))

    g_xy = Flux.batched_mul(x_t, y)
    return max.(x2_t .+ y2 .- T(2) .* g_xy, zero(T))
end

function mmd_sqdist_version_v2(x::CuArray{Float32,3}, y::CuArray{Float32,3})
    x_t = permutedims(x, (2, 1, 3))

    x2 = sum(abs2, x; dims=1)
    y2 = sum(abs2, y; dims=1)
    x2_t = permutedims(x2, (2, 1, 3))

    g_xy = Flux.batched_mul(x_t, y)
    return max.(x2_t .+ y2 .- T(2) .* g_xy, zero(T))
end



T = Float32
x = rand(T, 3, 2048, 128) |> cu
y = rand(T, 3, 2048, 128) |> cu 