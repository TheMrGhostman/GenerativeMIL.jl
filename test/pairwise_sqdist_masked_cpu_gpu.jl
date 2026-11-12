# Validity + speed test for the masked _pairwise_sqdist_batched(x, y, x_mask, y_mask)
# (src/losses/pairwise_sqdist_batched.jl). CPU always, GPU part if CUDA is functional.
# Run on a compute node: julia --project test/pairwise_sqdist_masked_cpu_gpu.jl
#
# Masked semantics: entries (i, j, b) with a valid x point i and valid y point j equal the unmasked distance,
# every other entry is +Inf. Gradients are checked through two losses that are well defined on the masked matrix:
#   f_kernel(P) = sum(exp.(-P))                  (MMD-like kernel; exp(-Inf) = 0, so invalid entries drop out)
#   f_valid(P)  = sum(ifelse.(isfinite.(P), P, 0))  (sum over valid entries only)
# and compared to the same losses computed from the UNMASKED matrix with an explicit validity mask.
using Test, Random, Statistics
using Flux, Zygote, CUDA, BenchmarkTools
using GenerativeMIL
const G = GenerativeMIL
const sqd = G._pairwise_sqdist_batched

random_mask(n, bs; p=0.6) = (m = rand(Float32, 1, n, bs) .< p; m[1, 1, :] .= true; m)
pair_valid(xm, ym) = permutedims(xm, (2, 1, 3)) .& ym                  # (n, m, bs)

f_kernel(P) = sum(exp.(-P))
f_valid(P)  = sum(ifelse.(isfinite.(P), P, zero(eltype(P))))
# references built from the unmasked matrix
f_kernel_ref(P, V) = sum(exp.(-P) .* V)
f_valid_ref(P, V)  = sum(P .* V)

function check_validity(x, y, xm, ym; dev=identity, tag="")
    xd, yd, xmd, ymd = dev(x), dev(y), dev(xm), dev(ym)
    V  = pair_valid(xm, ym)
    Vd = dev(Float32.(V))
    P  = Array(sqd(xd, yd, xmd, ymd))
    P0 = Array(sqd(xd, yd))

    @testset "forward $tag" begin
        @test size(P) == size(P0)
        @test P[V] ≈ P0[V]                                  # valid entries = unmasked distances
        @test all(isinf, P[.!V]) && all(>(0), P[.!V])       # invalid entries = +Inf
        @test !any(isnan, P)
        # brute force on the first bag, valid points only
        xi, yj = findall(vec(xm[1, :, 1])), findall(vec(ym[1, :, 1]))
        ref = [sum(abs2, x[:, i, 1] .- y[:, j, 1]) for i in xi, j in yj]
        @test P[xi, yj, 1] ≈ ref rtol=1e-4 atol=1e-4
        # all-true masks = unmasked
        tx, ty = dev(trues(size(xm))), dev(trues(size(ym)))
        @test Array(sqd(xd, yd, tx, ty)) ≈ P0
        # all-false: everything Inf, no NaN
        Pe = Array(sqd(xd, yd, dev(falses(size(xm))), ymd))
        @test all(isinf, Pe) && !any(isnan, Pe)
    end

    @testset "gradient $tag" begin
        for (f, fref, name) in ((f_kernel, f_kernel_ref, "kernel"), (f_valid, f_valid_ref, "valid-sum"))
            gx, gy   = Zygote.gradient((a, b) -> f(sqd(a, b, xmd, ymd)), xd, yd)
            rx, ry   = Zygote.gradient((a, b) -> fref(sqd(a, b), Vd), xd, yd)
            gx, gy, rx, ry = Array(gx), Array(gy), Array(rx), Array(ry)
            @test all(isfinite, gx) && all(isfinite, gy)                         # no NaN/Inf leaking from Inf entries
            @test gx ≈ rx rtol=1e-3 atol=1e-4                                     # same as explicit masking
            @test gy ≈ ry rtol=1e-3 atol=1e-4
            @test all(gx[repeat(.!xm, size(x, 1), 1, 1)] .== 0)                  # padded points: zero gradient
            @test all(gy[repeat(.!ym, size(y, 1), 1, 1)] .== 0)
            @test any(!=(0), gx)
        end
        # all-false mask: kernel loss is 0 and its gradient is 0
        xm_empty = dev(falses(size(xm)))          # move to device OUTSIDE the differentiated closure
        gxe = Zygote.gradient(a -> f_kernel(sqd(a, yd, xm_empty, ymd)), xd)[1]
        @test isnothing(gxe) || all(Array(gxe) .== 0)
    end
    return P
end

Random.seed!(3)
d, n, m, bs = 3, 40, 30, 4
x, y = randn(Float32, d, n, bs), randn(Float32, d, m, bs)
xm, ym = random_mask(n, bs), random_mask(m, bs)

@testset "masked _pairwise_sqdist_batched (CPU)" begin
    check_validity(x, y, xm, ym; tag="CPU")
end

gpu_ok = CUDA.functional()
if gpu_ok
    @testset "masked _pairwise_sqdist_batched (GPU)" begin
        P_gpu = check_validity(x, y, xm, ym; dev=cu, tag="GPU")
        P_cpu = sqd(x, y, xm, ym)
        V = pair_valid(xm, ym)
        @test sqd(cu(x), cu(y), cu(xm), cu(ym)) isa CuArray                        # GPU method, stays on GPU
        @test P_gpu[V] ≈ P_cpu[V] rtol=1e-3 atol=1e-4                               # GPU = CPU
        @test isinf.(P_gpu) == isinf.(P_cpu)
    end
else
    @info "CUDA not functional: GPU tests and GPU benchmarks skipped"
end

# ---- speed (informative): min time over samples; GPU calls wrapped in CUDA.@sync ---------------------------------
function bench(d, n, m, bs)
    x, y = randn(Float32, d, n, bs), randn(Float32, d, m, bs)
    xm, ym = random_mask(n, bs), random_mask(m, bs)
    bt(f) = 1e3 * @belapsed($f(), samples=20, seconds=15, evals=1)
    fw(a, b)        = sum(sqd(a, b))                    # unmasked forward
    fwm(a, b, p, q) = f_kernel(sqd(a, b, p, q))         # masked forward (+ kernel sum, finite scalar)
    fw0(a, b)       = f_kernel(sqd(a, b))
    gr(a, b)        = Zygote.gradient((u, v) -> fw0(u, v), a, b)
    grm(a, b, p, q) = Zygote.gradient((u, v) -> fwm(u, v, p, q), a, b)

    t = Dict{String,Float64}()
    t["cpu fwd"]    = bt(() -> fw0(x, y));          t["cpu fwd m"]  = bt(() -> fwm(x, y, xm, ym))
    t["cpu bwd"]    = bt(() -> gr(x, y));           t["cpu bwd m"]  = bt(() -> grm(x, y, xm, ym))
    if gpu_ok
        xg, yg, xmg, ymg = cu(x), cu(y), cu(xm), cu(ym)
        t["gpu fwd"] = bt(() -> CUDA.@sync fw0(xg, yg));  t["gpu fwd m"] = bt(() -> CUDA.@sync fwm(xg, yg, xmg, ymg))
        t["gpu bwd"] = bt(() -> CUDA.@sync gr(xg, yg));   t["gpu bwd m"] = bt(() -> CUDA.@sync grm(xg, yg, xmg, ymg))
    end

    r(v) = lpad(round(v, digits=3), 10)
    ratio(a, b) = lpad("$(round(a / b, digits=2))x", 10)
    println("\n(d, n, m, bs) = ($d, $n, $m, $bs)" * (gpu_ok ? "   GPU: $(CUDA.name(CUDA.device()))" : ""))
    println("time [ms]          ", lpad("CPU", 10), lpad("CPU mask", 10), (gpu_ok ? lpad("GPU", 10) * lpad("GPU mask", 10) : ""))
    for (k, name) in (("fwd", "forward"), ("bwd", "fwd+bwd"))
        println(rpad(name, 19), r(t["cpu $k"]), r(t["cpu $k m"]), (gpu_ok ? r(t["gpu $k"]) * r(t["gpu $k m"]) : ""))
    end
    println("ratios             ", lpad("GPU spdup", 10), lpad("GPU spd m", 10), lpad("mask CPU", 10), lpad("mask GPU", 10))
    for (k, name) in (("fwd", "forward"), ("bwd", "fwd+bwd"))
        gpu_cols = gpu_ok ? ratio(t["cpu $k"], t["gpu $k"]) * ratio(t["cpu $k m"], t["gpu $k m"]) : lpad("-", 10)^2
        println(rpad(name, 19), gpu_cols, ratio(t["cpu $k m"], t["cpu $k"]), (gpu_ok ? ratio(t["gpu $k m"], t["gpu $k"]) : lpad("-", 10)))
    end
end
# GPU spdup = CPU time / GPU time (bigger = GPU faster); mask CPU/GPU = masked time / unmasked time (1.00x = free)
println("\nBenchmark (unmasked uses the same kernel-sum loss as masked, so fwd+bwd is comparable)")
bench(3, 768, 768, 32)      # chamfer_pairwise-like: N=64 points x 12 clusters, BS=32
bench(3, 2048, 2048, 16)    # bigger point clouds
