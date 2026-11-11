# Smoke test for the masked chamfer_pairwise_distance(x, y, x_mask, y_mask).
# Run: julia --project test/chamfer_pairwise_masked_cpu_gpu.jl   (on a compute node; GPU part runs if CUDA works)
using Test, Random, Statistics
using Flux, Zygote, CUDA, BenchmarkTools
using GenerativeMIL
const G = GenerativeMIL

# brute-force reference: loop over cluster pairs, keep valid points, min squared distance in each direction
function ref_pairwise_chamfer(x, y, xm, ym; w1=1f0, w2=1f0, agg=mean)
    _, _, L, BS = size(x); M = size(y, 3)
    C = zeros(Float32, L, M, BS)
    for b in 1:BS, l in 1:L, m in 1:M
        a = x[:, vec(xm[1, :, l, b]), l, b]; c = y[:, vec(ym[1, :, m, b]), m, b]
        (isempty(a) || isempty(c)) && continue
        D = [sum(abs2, a[:, i] .- c[:, j]) for i in 1:size(a, 2), j in 1:size(c, 2)]
        C[l, m, b] = w1 * agg(vec(minimum(D, dims=2))) + w2 * agg(vec(minimum(D, dims=1)))
    end
    return C
end

function random_mask(N, K, BS; p=0.6)
    m = rand(Float32, 1, N, K, BS) .< p
    m[1, 1, :, :] .= true          # at least one valid point per cluster
    return m
end

Random.seed!(42)
N, L, M, BS = 20, 4, 5, 2
x = randn(Float32, 3, N, L, BS); y = randn(Float32, 3, N, M, BS);
xm = random_mask(N, L, BS); ym = random_mask(N, M, BS);

@testset "masked chamfer_pairwise_distance (CPU)" begin
    @testset "vs brute-force reference" begin
        for agg in (mean, sum), (w1, w2) in ((1f0, 1f0), (0.3f0, 2f0))
            C = chamfer_pairwise_distance(x, y, xm, ym; w1=w1, w2=w2, agg=agg)
            @test size(C) == (L, M, BS)
            @test C ≈ ref_pairwise_chamfer(x, y, xm, ym; w1=w1, w2=w2, agg=agg) rtol=1e-4
        end
        @test G.sum_chamfer_pairwise_distance(x, y, xm, ym) ≈ ref_pairwise_chamfer(x, y, xm, ym; agg=sum) rtol=1e-4
        @test_throws ArgumentError chamfer_pairwise_distance(x, y, xm, ym; agg=maximum)
        @test_throws DimensionMismatch chamfer_pairwise_distance(x, y, ym, ym)
    end

    @testset "all-true / nothing masks equal unmasked" begin
        tx, ty = trues(1, N, L, BS), trues(1, N, M, BS)
        C0 = chamfer_pairwise_distance(x, y)
        @test chamfer_pairwise_distance(x, y, tx, ty) ≈ C0
        @test chamfer_pairwise_distance(x, y, nothing, nothing) ≈ C0
        @test chamfer_pairwise_distance(x, y, tx, nothing) ≈ C0
        @test chamfer_pairwise_distance(x, y, nothing, ym) ≈ ref_pairwise_chamfer(x, y, tx, ym) rtol=1e-4
    end

    @testset "padded values do not matter" begin
        x2, y2 = copy(x), copy(y)
        x2[repeat(.!xm, 3, 1, 1, 1)] .= 1f3
        y2[repeat(.!ym, 3, 1, 1, 1)] .= -7f2
        @test chamfer_pairwise_distance(x2, y2, xm, ym) ≈ chamfer_pairwise_distance(x, y, xm, ym)
    end

    @testset "empty cluster" begin
        xe = copy(xm); xe[1, :, 2, 1] .= false     # cluster l=2 of bag 1 is empty
        C = chamfer_pairwise_distance(x, y, xe, ym)
        @test all(isfinite, C)
        @test all(C[2, :, 1] .== 0)                               # every pair with the empty cluster costs 0
        @test C ≈ ref_pairwise_chamfer(x, y, xe, ym) rtol=1e-4
    end

    @testset "fully empty y_mask / empty bag" begin
        y0 = falses(1, N, M, BS)
        C = chamfer_pairwise_distance(x, y, xm, y0)
        @test all(isfinite, C) && all(C .== 0)
        gx, gy = Zygote.gradient((a, b) -> sum(chamfer_pairwise_distance(a, b, xm, y0)), x, y)
        @test (isnothing(gx) || all(gx .== 0)) && (isnothing(gy) || all(gy .== 0))

        yb = copy(ym); yb[1, :, :, 2] .= false   # bag 2: no valid y point at all
        Cb = chamfer_pairwise_distance(x, y, xm, yb)
        @test all(Cb[:, :, 2] .== 0)
        @test Cb[:, :, 1] ≈ chamfer_pairwise_distance(x, y, xm, ym)[:, :, 1]
    end

    @testset "gradient" begin
        gx, gy = Zygote.gradient((a, b) -> sum(chamfer_pairwise_distance(a, b, xm, ym)), x, y)
        @test all(isfinite, gx) && all(isfinite, gy)
        @test all(gx[repeat(.!xm, 3, 1, 1, 1)] .== 0)   # padded points get no gradient
        @test all(gy[repeat(.!ym, 3, 1, 1, 1)] .== 0)
        @test any(!=(0), gx) && any(!=(0), gy)
    end

    @testset "timing (informative)" begin
        Nb, Lb, Mb, Bb = 64, 12, 12, 32
        xb, yb = randn(Float32, 3, Nb, Lb, Bb), randn(Float32, 3, Nb, Mb, Bb)
        xmb, ymb = random_mask(Nb, Lb, Bb), random_mask(Nb, Mb, Bb)
        chamfer_pairwise_distance(xb, yb); chamfer_pairwise_distance(xb, yb, xmb, ymb)   # compile
        t0 = @elapsed chamfer_pairwise_distance(xb, yb)
        t1 = @elapsed chamfer_pairwise_distance(xb, yb, xmb, ymb)
        tg = @elapsed Zygote.gradient(a -> sum(chamfer_pairwise_distance(a, yb, xmb, ymb)), xb)
        println("CPU N=$Nb L=M=$Lb BS=$Bb: unmasked $(round(t0, digits=3)) s, masked $(round(t1, digits=3)) s, masked fwd+bwd $(round(tg, digits=3)) s")
        @test true
    end
end

if CUDA.functional()
    @testset "masked chamfer_pairwise_distance (GPU)" begin
        C_cpu = chamfer_pairwise_distance(x, y, xm, ym)
        # move everything to GPU *outside* the differentiated closure (cu inside gradient makes Zygote trace CUDA allocs)
        xg, yg, xmg, ymg = cu(x), cu(y), cu(xm), cu(ym)
        C_gpu = chamfer_pairwise_distance(xg, yg, xmg, ymg)
        @test C_gpu isa CuArray
        @test Array(C_gpu) ≈ C_cpu rtol=1e-3
        gx = Zygote.gradient(a -> sum(chamfer_pairwise_distance(a, yg, xmg, ymg)), xg)[1]
        gx_cpu = Zygote.gradient(a -> sum(chamfer_pairwise_distance(a, y, xm, ym)), x)[1]
        @test gx isa CuArray
        @test all(isfinite, Array(gx))
        @test Array(gx) ≈ gx_cpu rtol=1e-3
        @test Array(chamfer_pairwise_distance(xg, yg, nothing, ymg)) ≈ chamfer_pairwise_distance(x, y, nothing, ym) rtol=1e-3
    end

    # ---- CPU vs GPU benchmark (informative, no tests) ---------------------------------------------------------------
    # minimum time over samples (BenchmarkTools); GPU timings wrapped in CUDA.@sync so kernels are actually finished
    let Nb = 64, Lb = 12, Mb = 12, Bb = 32
        xc, yc = randn(Float32, 3, Nb, Lb, Bb), randn(Float32, 3, Nb, Mb, Bb)
        xmc, ymc = random_mask(Nb, Lb, Bb), random_mask(Nb, Mb, Bb)
        xgg, ygg, xmgg, ymgg = cu(xc), cu(yc), cu(xmc), cu(ymc)

        fwd(a, b)             = sum(chamfer_pairwise_distance(a, b))
        fwd(a, b, am, bm)     = sum(chamfer_pairwise_distance(a, b, am, bm))
        grad(a, b)            = Zygote.gradient(q -> fwd(q, b), a)
        grad(a, b, am, bm)    = Zygote.gradient(q -> fwd(q, b, am, bm), a)
        bt(f) = @belapsed $f() samples=20 seconds=20 evals=1

        res = [
            ("unmasked fwd",     bt(() -> fwd(xc, yc)),                bt(() -> CUDA.@sync fwd(xgg, ygg))),
            ("masked fwd",       bt(() -> fwd(xc, yc, xmc, ymc)),      bt(() -> CUDA.@sync fwd(xgg, ygg, xmgg, ymgg))),
            ("unmasked fwd+bwd", bt(() -> grad(xc, yc)),               bt(() -> CUDA.@sync grad(xgg, ygg))),
            ("masked fwd+bwd",   bt(() -> grad(xc, yc, xmc, ymc)),     bt(() -> CUDA.@sync grad(xgg, ygg, xmgg, ymgg))),
        ]
        println("\nBenchmark N=$Nb, L=M=$Lb, BS=$Bb  (min time, ms) on ", CUDA.name(CUDA.device()))
        println(rpad("", 18), lpad("CPU", 10), lpad("GPU", 10), lpad("speedup", 10))
        for (name, tc, tg) in res
            println(rpad(name, 18), lpad(round(1e3tc, digits=1), 10), lpad(round(1e3tg, digits=1), 10), lpad("$(round(tc/tg, digits=1))x", 10))
        end
    end
else
    @info "CUDA not functional: GPU part skipped"
end

