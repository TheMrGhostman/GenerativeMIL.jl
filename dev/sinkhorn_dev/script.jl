using Flux, Statistics, CUDA, Zygote, OptimalTransport
using BenchmarkTools

function chamfer_sqdist_version(x::CuArray{T,3}, y::CuArray{T,3}) where T<:AbstractFloat
    xx = sum(x .^ 2, dims = 1)
    yy = sum(y .^ 2, dims = 1)
    zz = Flux.batched_mul(permutedims(x, (2, 1, 3)), y)
    rx = reshape(xx, size(xx, 2), 1, :)
    ry = reshape(yy, 1, size(yy, 2), :)
    P = (rx .+ ry) .- (2 .* zz)
    return P
end


function mmd_sqdist_version(x::CuArray{T,3}, y::CuArray{T,3}) where T<:AbstractFloat
    x_t = permutedims(x, (2, 1, 3))
    y_t = permutedims(y, (2, 1, 3))

    x2 = sum(abs2, x; dims=1)
    y2 = sum(abs2, y; dims=1)
    x2_t = permutedims(x2, (2, 1, 3))

    g_xy = Flux.batched_mul(x_t, y)
    return max.(x2_t .+ y2 .- T(2) .* g_xy, zero(T))
end

function mmd_sqdist_version_v2(x::Array{T,2}, y::Array{T,2}) where T<:AbstractFloat
    x_t = permutedims(x, (2, 1, 3))

    x2 = sum(abs2, x; dims=1)
    y2 = sum(abs2, y; dims=1)
    x2_t = permutedims(x2, (2, 1, 3))

    g_xy = Flux.batched_mul(x_t, y)
    return max.(x2_t .+ y2 .- T(2) .* g_xy, zero(T))
end

function mmd_sqdist_version_v2(x::CuArray{T,3}, y::CuArray{T,3}) where T<:AbstractFloat
    x_t = permutedims(x, (2, 1, 3))

    x2 = sum(abs2, x; dims=1)
    y2 = sum(abs2, y; dims=1)
    x2_t = permutedims(x2, (2, 1, 3))

    g_xy = Flux.batched_mul(x_t, y)
    return max.(x2_t .+ y2 .- T(2) .* g_xy, zero(T))
end



T = Float32
x = rand(T, 3, 2048, 32) |> cu;
y = rand(T, 3, 2048, 32) |> cu;

@benchmark chamfer_sqdist_version($x, $y)
@benchmark mmd_sqdist_version($x, $y)
@benchmark mmd_sqdist_version_v2($x, $y)

o1 = chamfer_sqdist_version(x,y);
o2 = mmd_sqdist_version(x,y);
o3 = mmd_sqdist_version_v2(x,y);


o1 ≈ o2
o1 ≈ o3
o2 ≈ o3

####

using OptimalTransport 

C_xy = mmd_sqdist_version_v2(x,y);
C_x = mmd_sqdist_version_v2(x,x);
C_y = mmd_sqdist_version_v2(y,y);

μ = fill(1f0 / size(x, 2), size(x, 2), size(x, 3))|>cu
ν = fill(1f0 / size(y, 2), size(y, 2), size(y, 3))|>cu

sinkhorn_div = OptimalTransport.sinkhorn_divergence(μ, ν, C_xy, C_x, C_y, 0.1f0)

@benchmark OptimalTransport.sinkhorn_divergence($μ[:,1], $ν[:,1], $C_xy[:,:,1], $C_x[:,:,1], $C_y[:,:,1], 0.1f0)
OptimalTransport.sinkhorn_divergence(μ[:,1], ν[:,1], C_xy[:,:,1], C_x[:,:,1], C_y[:,:,1], 0.1f0)

C_xy|>size
C_x|>size
C_y|>size
μ |>size
ν |>size


function create_priors(::Array, n_points)
    μ = fill(1f0 / n_points, n_points)
    ν = fill(1f0 / n_points, n_points)
    return μ, ν
end

function create_priors(::CuArray, n_points)
    μ = fill(1f0 / n_points, n_points) |> cu
    ν = fill(1f0 / n_points, n_points) |> cu
    return μ, ν
end

function batched_sinkhorn_div(C_xy, C_xx, C_yy, ε)
    # X_batch a Y_batch jsou CuArray tvaru: (Dimenze, Počet_bodů, Velikost_batchi)
    n_points = size(C_xy[1], 1)
    
    # Rovnoměrné váhy bodů pro jednu ukázku
    μ, ν = create_priors(C_xy[1], n_points)
    
    # Vytvoření speciálního bufferu pro Zygote gradienty
    #losses = Zygote.Buffer(C_xy, batch_size)
    #losses = zeros(Float32, batch_size) |> cu
    
    losses = map((a,b,c)->sinkhorn_divergence(μ, ν, a,b,c, ε), C_xy, C_xx, C_yy)
    
    # Vrátíme průměrnou loss přes celou batch
    return losses
end

@benchmark batched_sinkhorn_div(eachslice(cpu(C_xy), dims=3), eachslice(cpu(C_x), dims=3), eachslice(cpu(C_y), dims=3), 1f0)

@benchmark batched_sinkhorn_div(eachslice(C_xy, dims=3), eachslice(C_x, dims=3), eachslice(C_y, dims=3), 1f0)

function batched_sinkhorn_div_v2(x, y, ε)
    # X_batch a Y_batch jsou CuArray tvaru: (Dimenze, Počet_bodů, Velikost_batchi)
    dim, n_points, batch_size = size(x)
    
    C_xy = eachslice(mmd_sqdist_version_v2(x,y), dims=3);
    C_x = eachslice(mmd_sqdist_version_v2(x,x), dims=3);
    C_y = eachslice(mmd_sqdist_version_v2(y,y), dims=3);

    # Rovnoměrné váhy bodů pro jednu ukázku
    μ, ν = create_priors(x, n_points)

    losses = map((a,b,c)->sinkhorn_divergence(μ, ν, a,b,c, ε; maxiter=50), C_xy, C_x, C_y)

    # Vrátíme průměrnou loss přes celou batch
    return losses
end

@benchmark batched_sinkhorn_div_v2($x, $y, 1f0)


@elapsed g = Zygote.gradient(Y->batched_sinkhorn_div_v2(x, Y, 1f0), y)


function batched_sinkhorn_div_v3(x, y, ε)
    # x a y jsou CuArray tvaru: (Dimenze, Počet_bodů, Velikost_batchi)
    dim, n_points, batch_size = size(x)
    
    # 1. Spočítáme celé 3D matice cen naráz na GPU (předpokládám, že vaše verze v2 vrací 3D CuArray)
    # Tvar matic: (n_points, n_points, batch_size)
    C_xy_all = mmd_sqdist_version_v2(x, y)
    C_x_all  = mmd_sqdist_version_v2(x, x)
    C_y_all  = mmd_sqdist_version_v2(y, y)

    # 2. Vytvoříme fixní solver (Zygote potřebuje mít solver explicitně jako objekt)

    # Rovnoměrné váhy bodů pro jednu ukázku
    μ, ν = create_priors(x, n_points)

    # 3. Zygote Buffer namísto eachslice + map
    # Buffer umožňuje bezpečný zápis na GPU, který Zygote umí derivovat
    losses = Zygote.Buffer(x, batch_size)

    for i in 1:batch_size
        # Vytahujeme 2D matice cen pomocí přímého indexování (Zygote umí derivovat)
        C_xy = C_xy_all[:, :, i]
        C_x  = C_x_all[:, :, i]
        C_y  = C_y_all[:, :, i]
        
        # Voláme sinkhorn_divergence, solver předáváme jako poslední argument
        losses[i] = sinkhorn_divergence(μ, ν, C_xy, C_x, C_y, ε, maxiter=50)
    end

    # copy(losses) převede buffer zpět na standardní CuArray a spočítáme průměr
    return mean(copy(losses))
end


using OptimalTransport
using CUDA
using Zygote
using Statistics

function batched_sinkhorn_div_v4(x, y, ε)
    # x a y jsou CuArray tvaru: (Dimenze, Počet_bodů, Velikost_batchi)
    dim, n_points, batch_size = size(x)
    
    # 1. Spočítáme celé 3D matice cen naráz na GPU
    C_xy_all = mmd_sqdist_version_v2(x, y)
    C_x_all  = mmd_sqdist_version_v2(x, x)
    C_y_all  = mmd_sqdist_version_v2(y, y)

    # Solver definovaný jako objekt s fixními iteracemi
    #stabilni_solver = SinkhornStabilized(; maxiter=50, tol=0.0)

    # Rovnoměrné váhy bodů pro jednu ukázku
    μ, ν = create_priors(x, n_points)

    # 2. Místo bufferu a for-smyčky použijeme čistou komprehenzi (List Comprehension)
    # Tento zápis vygeneruje standardní Julia pole na CPU, 
    # ale jednotlivé matice uvnitř a samotný sinkhorn_divergence běží na GPU.
    #losses = [
    #    sinkhorn_divergence(μ, ν, C_xy_all[:, :, i], C_x_all[:, :, i], C_y_all[:, :, i], ε)
    #    for i in 1:batch_size
    #]
    losses = [
        sinkhorn2(μ, ν, C_xy_all[:, :, i], ε; maxiter=50) # Bez explicitního solveru
        for i in 1:batch_size
        # A nezapomeňte pak ručně odečíst samo-vzdálenosti, pokud chcete divergenci
    ]

    # 3. Sečteme/zprůměrujeme výsledky (Zygote umí sum() nad obyčejným polem derivovat)
    return sum(losses) / batch_size
end


@benchmark batched_sinkhorn_div_v4($x, $y, 1f0)


@elapsed g = Zygote.gradient(Y->batched_sinkhorn_div_v4(x, Y, 1f0), y)



function batched_sinkhorn_div_v5(x, y, ε)
    # x a y jsou CuArray tvaru: (Dimenze, Počet_bodů, Velikost_batchi)
    dim, n_points, batch_size = size(x)
    
    # 1. Spočítáme celé 3D matice cen naráz na GPU
    C_xy_all = mmd_sqdist_version_v2(x, y) |> cpu
    C_x_all  = mmd_sqdist_version_v2(x, x) |> cpu
    C_y_all  = mmd_sqdist_version_v2(y, y) |> cpu

    # Solver definovaný jako objekt s fixními iteracemi
    #stabilni_solver = SinkhornStabilized(; maxiter=50, tol=0.0)

    # Rovnoměrné váhy bodů pro jednu ukázku
    μ, ν = create_priors(C_x_all, n_points)

    # 2. Místo bufferu a for-smyčky použijeme čistou komprehenzi (List Comprehension)
    # Tento zápis vygeneruje standardní Julia pole na CPU, 
    # ale jednotlivé matice uvnitř a samotný sinkhorn_divergence běží na GPU.
    losses = [
        sinkhorn_divergence(μ, ν, C_xy_all[:, :, i], C_x_all[:, :, i], C_y_all[:, :, i], ε)
        for i in 1:batch_size
    ]

    # 3. Sečteme/zprůměrujeme výsledky (Zygote umí sum() nad obyčejným polem derivovat)
    return sum(losses) / batch_size
end

@benchmark batched_sinkhorn_div_v5($x, $y, 1f0)

@elapsed g = Zygote.gradient(Y->batched_sinkhorn_div_v5(x, Y, 1f0), y)



#####
c_x  =  C_x[:,:,1];
c_y  =  C_y[:,:,1];
c_xy = C_xy[:,:,1];

μ = fill(1f0 / size(x, 2),  size(y, 2))|>cu
ν = fill(1f0 / size(y, 2),  size(y, 2))|>cu

OptimalTransport.sinkhorn_divergence(μ, ν, c_xy, c_x, c_y, 1f0)

sinkhorn2(μ, ν, c_xy, 1f0) - 0.5(sinkhorn2(μ, μ, c_x, 1f0) + sinkhorn2(ν, ν, c_y, 1f0) )

OptimalTransport.sinkhorn_divergence(μ, ν, c_xy, c_x, c_y, 1f0)
sinkhorn2(μ, ν, c_xy, 1f0, SinkhornGibbs()) - 0.5f0(sinkhorn2(μ, c_x, 1f0, OptimalTransport.SymmetricSinkhornGibbs()) + sinkhorn2(ν, c_y, 1f0, OptimalTransport.SymmetricSinkhornGibbs()) )

sinkhorn2(μ, ν, c_xy, 1f0, SinkhornGibbs()) - 0.5f0(sinkhorn2(μ, μ, c_x, 1f0, SinkhornGibbs()) + sinkhorn2(ν, ν, c_y, 1f0, SinkhornGibbs()) )

sinkhorn2(μ, ν, c_xy, 1f0, SinkhornGibbs()) 
sum(sinkhorn(μ, ν, c_xy, 1f0, SinkhornGibbs()) .* c_xy)

sinkhorn2(μ, ν, c_x, 1f0, OptimalTransport.SymmetricSinkhornGibbs())
sum(sinkhorn(μ, c_x, 1f0, OptimalTransport.SymmetricSinkhornGibbs()) .* c_x)

p_xy = sinkhorn(μ, ν, c_xy, 1f0, SinkhornGibbs());
p_x  = sinkhorn(μ, c_x, 1f0, OptimalTransport.SymmetricSinkhornGibbs());
p_y  = sinkhorn(ν, c_y, 1f0, OptimalTransport.SymmetricSinkhornGibbs());

p_xy |> size
c_xy |> size