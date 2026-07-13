using Flux, Statistics, CUDA, Zygote, OptimalTransport
using BenchmarkTools, MLUtils 
using GenerativeMIL

T = Float32

function chamfer_sqdist_version(x::CuArray{T,3}, y::CuArray{T,3}) where T<:AbstractFloat
    xx = sum(x .^ 2, dims = 1)
    yy = sum(y .^ 2, dims = 1)
    zz = Flux.batched_mul(permutedims(x, (2, 1, 3)), y)
    rx = reshape(xx, size(xx, 2), 1, :)
    ry = reshape(yy, 1, size(yy, 2), :)
    P = (rx .+ ry) .- (2 .* zz)
    return P
end


function density_aware_chamfer_distance(x::AbstractArray{T,3}, y::AbstractArray{T,3}, α::AbstractFloat=1f0) where T<:AbstractFloat
    # Compute pairwise squared distances
    ỹᵢ, x̃ᵢ = Zygote.@ignore _nearest_neighbors(x, y)

    ny = Zygote.@ignore _contributions(ỹᵢ) # (N, BS) -> (1, N, BS)
    nx = Zygote.@ignore _contributions(x̃ᵢ) # (M, BS) -> (1, M, BS)
    
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

function fast_scatter_add!(x::AbstractArray, idx::AbstractArray, values::AbstractArray)
    @inbounds for i in eachindex(idx, values)
        x[idx[i]] += values[i]
    end
end

function fast_scatter_add!(x::AbstractArray, idx::AbstractArray, values::AbstractFloat)
    @inbounds for i in eachindex(idx)
        x[idx[i]] += values
    end
end

x = randn(T, 3, 10, 2);
y = randn(T, 3, 10, 2);

ỹᵢ, x̃ᵢ = Zygote.@ignore GenerativeMIL._nearest_neighbors(x, y)

ny = zeros(10,2)
fast_scatter_add!(ny, ỹᵢ, 1f0)
ny[ỹᵢ] #-> this is n_y

ny = _contributions(ỹᵢ) # (N, BS) -> (1, N, BS)

function tmp(yi)
    ny = zeros(Float32, size(yi)...)
    fast_scatter_add!(ny, yi, 1f0)
    ny[yi] 
end

function tmp2(d_x::AbstractArray{T,3}, ny) where T<:AbstractFloat
    return 1 .- exp.(- d_x) ./ (ny .+ eps(T))
end

function tmp3(d_x::AbstractArray{T,3}, ny) where T<:AbstractFloat
    ny = T(1) ./ ny .+ eps(T)
    return T(1) .- exp.(- d_x) .* ny
end


xx = randn(T, 3, 256, 32);
yy = randn(T, 3, 256, 32);

yyᵢ, xxᵢ = Zygote.@ignore GenerativeMIL._nearest_neighbors(xx, yy)

nyy = _contributions(yyᵢ) # (N, BS) -> (1, N, BS)

sxx = sum((xx .- yy[:, yyᵢ]) .^ 2, dims=1); # (D, N, BS) -> (1,1,BS)
@benchmark tmp2($sxx, $nyy)
@benchmark tmp3($sxx, $nyy)

@benchmark _contributions($yyᵢ)

#gpu test
x = randn(T, 3, 256, 32);
y = randn(T, 3, 256, 32);

xg = x |> cu;
yg = y |> cu;

l = density_aware_chamfer_distance(x, y, 1000f0)
@benchmark density_aware_chamfer_distance($x, $y, $1000f0)

lg = density_aware_chamfer_distance(xg, yg, 1000f0)
@benchmark density_aware_chamfer_distance($xg, $yg, $1000f0)

d = Flux.Dense(3,3);
dg = d |>cu;

g = Zygote.gradient((dd)->density_aware_chamfer_distance(xg, dd(yg), 1000f0), dg)

g[1].weight

@benchmark Zygote.gradient((dd)->density_aware_chamfer_distance($x, dd($y), $1000f0), $d)

@benchmark Zygote.gradient((dd)->density_aware_chamfer_distance($xg, dd($yg), $1000f0), $dg)

@benchmark Zygote.gradient((dd)->chamfer_difstance($xg, dd($yg)), $dg)




loss_cfg = Dict{Symbol, Any}(
    :type=>"density_aware_chamfer_distance",
    :loss_scale=>1, #512
    :alpha => 1000f0
    )

loss_function = create_loss_function(loss_cfg)

loss_function(xg, yg) ≈ loss_cfg[:loss_scale] * density_aware_chamfer_distance(xg, yg, loss_cfg[:alpha])