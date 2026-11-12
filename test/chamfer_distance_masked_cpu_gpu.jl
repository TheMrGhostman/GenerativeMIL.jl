# Validity + speed test for the vectorised masked Chamfer distance in src/losses/chamfer_distance.jl
#   chamfer_distance(A, B, A_mask[, B_mask])  (exported by GenerativeMIL; also the masked loss of SetVAE)
# Reference: the former loop-based implementation src/losses/masked_chamfer_distance.jl, which is NOT part of the
# package any more -- it is loaded here into its own module so its names don't clash with GenerativeMIL's.
# CPU always; GPU part (validity + benchmarks) if CUDA is functional.
# Run on a compute node: julia --project test/chamfer_distance_masked_cpu_gpu.jl
using Test, Random, Statistics
using Flux, Zygote, CUDA, BenchmarkTools
using GenerativeMIL
const G = GenerativeMIL

module OldMaskedChamfer
    using Statistics
    include(joinpath(@__DIR__, "..", "src", "losses", "masked_chamfer_distance.jl"))
end
old_cd(x, y, xm, ym) = OldMaskedChamfer._masked_chamfer_distance_batched(x, y, xm, ym)   # per-bag loop reference

random_mask(n, bs; p=0.6) = (m = rand(Float32, 1, n, bs) .< p; m[1, 1, :] .= true; m)
pad_index(x, m) = repeat(.!m, size(x, 1), 1, 1)        # (d, n, bs) Bool: padded coordinates

gpu_ok = CUDA.functional()
Random.seed!(11)
d, n, m, bs = 3, 60, 45, 6
x, y = randn(Float32, d, n, bs), randn(Float32, d, m, bs)
xm, ym = random_mask(n, bs), random_mask(m, bs)

@testset "masked chamfer_distance (CPU)" begin
    @testset "vs old loop implementation" begin
        @test chamfer_distance(x, y, xm, ym) ≈ old_cd(x, y, xm, ym) rtol=1e-4
        @test chamfer_distance(x, y, xm) ≈ old_cd(x, y, xm, trues(1, m, bs)) rtol=1e-4         # single-mask variant
        # 2-D sets (any dimension, no hard-coded * 3)
        x2, y2 = randn(Float32, 2, n, bs), randn(Float32, 2, m, bs)
        @test chamfer_distance(x2, y2, xm, ym) ≈ old_cd(x2, y2, xm, ym) rtol=1e-4
        # Float64 works too
        @test chamfer_distance(Float64.(x), Float64.(y), xm, ym) ≈ old_cd(Float64.(x), Float64.(y), xm, ym) rtol=1e-8
    end

    @testset "weights w1 / w2" begin
        c1 = chamfer_distance(x, y, xm, ym; w1=1, w2=0)
        c2 = chamfer_distance(x, y, xm, ym; w1=0, w2=1)
        @test c1 + c2 ≈ chamfer_distance(x, y, xm, ym)
        @test chamfer_distance(x, y, xm, ym; w1=0.3, w2=2.0) ≈ 0.3f0 * c1 + 2f0 * c2
    end

    @testset "unmasked API untouched" begin
        @test chamfer_distance(x, y) isa Float32                                               # 2-arg call still works
        # all-true masks, equal bag sizes, d = 3: masked == old unmasked chamfer (its * 3 is exact for d = 3)
        @test chamfer_distance(x, y, trues(1, n, bs), trues(1, m, bs)) ≈ chamfer_distance(x, y) rtol=1e-4
    end

    @testset "padding, empty bags" begin
        xp, yp = copy(x), copy(y)
        xp[pad_index(x, xm)] .= 5f2; yp[pad_index(y, ym)] .= -3f2
        @test chamfer_distance(xp, yp, xm, ym) ≈ chamfer_distance(x, y, xm, ym)               # padded values ignored
        xe = copy(xm); xe[1, :, 2] .= false                                                    # bag 2 empty on x side
        @test isfinite(chamfer_distance(x, y, xe, ym))
        @test chamfer_distance(x, y, xe, ym) ≈ old_cd(x, y, xe, ym) rtol=1e-4                  # empty bag -> 0 in both
        @test chamfer_distance(x, y, falses(1, n, bs), ym) == 0
    end

    @testset "gradient vs old loop" begin
        gx, gy = Zygote.gradient((a, b) -> chamfer_distance(a, b, xm, ym), x, y)
        rx, ry = Zygote.gradient((a, b) -> old_cd(a, b, xm, ym), x, y)
        @test all(isfinite, gx) && all(isfinite, gy)
        @test gx ≈ rx rtol=1e-3 atol=1e-5
        @test gy ≈ ry rtol=1e-3 atol=1e-5
        @test all(gx[pad_index(x, xm)] .== 0) && all(gy[pad_index(y, ym)] .== 0)               # padding: no gradient
    end
end

if gpu_ok
    @testset "masked chamfer_distance (GPU)" begin
        # move everything to the GPU OUTSIDE differentiated closures
        xg, yg, xmg, ymg = cu(x), cu(y), cu(xm), cu(ym)
        c_cpu = chamfer_distance(x, y, xm, ym)
        c_gpu = chamfer_distance(xg, yg, xmg, ymg)
        @test c_gpu isa Float32
        @test c_gpu ≈ c_cpu rtol=1e-4
        @test chamfer_distance(xg, yg, xmg) ≈ chamfer_distance(x, y, xm) rtol=1e-4
        nnx, _ = G._nearest_neighbors(xg, yg, xmg, ymg)
        @test nnx isa CuArray                                                                  # indices stay on GPU
        gx, gy = Zygote.gradient((a, b) -> chamfer_distance(a, b, xmg, ymg), xg, yg)
        rx, ry = Zygote.gradient((a, b) -> chamfer_distance(a, b, xm, ym), x, y)
        @test gx isa CuArray
        @test Array(gx) ≈ rx rtol=1e-3 atol=1e-5
        @test Array(gy) ≈ ry rtol=1e-3 atol=1e-5
        # unmasked GPU path (indices now kept on GPU) still equals the CPU KDTree path
        @test chamfer_distance(xg, yg) ≈ chamfer_distance(x, y) rtol=1e-4
    end
else
    @info "CUDA not functional: GPU tests and GPU benchmarks skipped"
end

# ---- speed (informative): min time; GPU calls wrapped in CUDA.@sync -------------------------------------------------
function bench(d, n, bs)
    x, y = randn(Float32, d, n, bs), randn(Float32, d, n, bs)
    xm, ym = random_mask(n, bs), random_mask(n, bs)
    bt(f) = 1e3 * @belapsed($f(), samples=10, seconds=20, evals=1)
    t = Dict{String,Float64}()
    t["old cpu fwd"] = bt(() -> old_cd(x, y, xm, ym))
    t["old cpu bwd"] = bt(() -> Zygote.gradient(a -> old_cd(a, y, xm, ym), x))
    t["new cpu fwd"] = bt(() -> chamfer_distance(x, y, xm, ym))
    t["new cpu bwd"] = bt(() -> Zygote.gradient(a -> chamfer_distance(a, y, xm, ym), x))
    t["unm cpu fwd"] = bt(() -> chamfer_distance(x, y))
    t["unm cpu bwd"] = bt(() -> Zygote.gradient(a -> chamfer_distance(a, y), x))
    if gpu_ok
        xg, yg, xmg, ymg = cu(x), cu(y), cu(xm), cu(ym)
        t["old gpu fwd"] = bt(() -> CUDA.@sync old_cd(xg, yg, xmg, ymg))
        t["old gpu bwd"] = try bt(() -> CUDA.@sync Zygote.gradient(a -> old_cd(a, yg, xmg, ymg), xg)) catch; NaN end
        t["new gpu fwd"] = bt(() -> CUDA.@sync chamfer_distance(xg, yg, xmg, ymg))
        t["new gpu bwd"] = bt(() -> CUDA.@sync Zygote.gradient(a -> chamfer_distance(a, yg, xmg, ymg), xg))
        t["unm gpu fwd"] = bt(() -> CUDA.@sync chamfer_distance(xg, yg))
        t["unm gpu bwd"] = bt(() -> CUDA.@sync Zygote.gradient(a -> chamfer_distance(a, yg), xg))
    end
    r(v) = lpad(isnan(v) ? "n/a" : string(round(v, digits=2)), 11)
    q(a, b) = lpad(isnan(a) || isnan(b) ? "n/a" : "$(round(a / b, digits=1))x", 11)
    println("\n(d, n, bs) = ($d, $n, $bs)" * (gpu_ok ? "   GPU: $(CUDA.name(CUDA.device()))" : ""))
    println(rpad("time [ms]", 22), lpad("CPU", 11), (gpu_ok ? lpad("GPU", 11) * lpad("GPU spdup", 11) : ""))
    for (k, name) in (("old", "old loop (masked)"), ("new", "new (masked)"), ("unm", "unmasked"))
        for (s, sname) in (("fwd", "fwd"), ("bwd", "fwd+bwd"))
            c = t["$k cpu $s"]
            println(rpad("$name $sname", 22), r(c), (gpu_ok ? r(t["$k gpu $s"]) * q(c, t["$k gpu $s"]) : ""))
        end
    end
    println("new vs old loop speedup:  CPU fwd ", q(t["old cpu fwd"], t["new cpu fwd"]), "  CPU fwd+bwd ", q(t["old cpu bwd"], t["new cpu bwd"]),
            (gpu_ok ? "  GPU fwd " * q(t["old gpu fwd"], t["new gpu fwd"]) * "  GPU fwd+bwd " * q(t["old gpu bwd"], t["new gpu bwd"]) : ""))
end
println("\nBenchmark (GPU spdup = CPU time / GPU time)")
bench(3, 512, 32)      # SetVAE-like batches
bench(3, 2048, 16)     # large point clouds

