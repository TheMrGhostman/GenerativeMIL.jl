# Generation metrics between a set of generated shapes G and a set of reference shapes R
# (Yang et al. 2019, PointFlow; Achlioptas et al. 2018). Everything is computed from
# pairwise set-distance matrices, so the same code works for any set distance (CD, DCD, ...).
#
#   MMD   – mean over reference shapes of the distance to the closest generated shape (quality)
#   COV   – fraction of reference shapes that are the nearest neighbour of ≥ 1 generated shape (diversity)
#   1-NNA – leave-one-out 1-NN accuracy of telling G from R (50 % ideal, 100 % = trivially separable,
#           < 50 % = generated shapes are closer to R than R is to itself, e.g. memorisation)

"""
`pairwise_set_distances(dist, A, B; batch_pairs=64, device=cpu, symmetric=false)`

Matrix `D[i, j] = dist(A[:, :, i], B[:, :, j])` for two collections of sets `A (d, n, NA)` and `B (d, n, NB)`.

Arguments:
- `dist`: batched, per-pair set distance `dist(x, y) -> vector`, where `x[:, :, k]` is compared with `y[:, :, k]`
  (e.g. `chamfer_distance_eval`, `(x, y) -> density_aware_chamfer_distance_eval(x, y, 1f0)`).
- `A`, `B`: CPU arrays of sets.

Keyword arguments:
- `batch_pairs`: number of (i, j) pairs evaluated at once. Memory per pair is about `n × n` floats
  (2048 points -> ~16 MB), so 64 pairs ≈ 1 GB on GPU.
- `device`: `cpu` or `gpu` / `cu`.
- `symmetric`: `A` and `B` are the same collection; only i < j is computed, the diagonal is 0.

Returns:
- `Matrix{Float64}` of size `(NA, NB)`.
"""
function pairwise_set_distances(dist::Function, A::AbstractArray{T, 3}, B::AbstractArray{T, 3};
                                batch_pairs::Int=64, device::Function=cpu, symmetric::Bool=false) where T <: AbstractFloat
    NA, NB = size(A, 3), size(B, 3)
    symmetric && @assert NA == NB "symmetric=true requires A and B to be the same collection"
    pairs = symmetric ? [(i, j) for j in 1:NB for i in 1:j-1] : [(i, j) for j in 1:NB for i in 1:NA]
    D = zeros(Float64, NA, NB)
    for chunk in Iterators.partition(pairs, batch_pairs)
        is, js = first.(chunk), last.(chunk)
        d = Array(dist(A[:, :, is] |> device, B[:, :, js] |> device))
        for (k, (i, j)) in enumerate(chunk)
            D[i, j] = d[k]
            symmetric && (D[j, i] = d[k])
        end
    end
    return D
end

"""
`generation_metrics(D_GR, D_GG, D_RR)`

MMD, COV and 1-NNA from precomputed distance matrices:
- `D_GR (NG, NR)`: generated × reference
- `D_GG (NG, NG)`: generated × generated
- `D_RR (NR, NR)`: reference × reference (independent of the model; compute once per dataset and metric and reuse)

Returns a named tuple:
- `mmd`, `cov`, `nna` (1-NNA accuracy in [0, 1]), `nna_gen`, `nna_ref` (accuracy on the generated / reference half),
- `nn_ref_of_gen`: index of the nearest reference shape for every generated shape (used for COV and class coverage).
- `div_gen`, `div_ref`: diversity = mean distance between two DIFFERENT shapes within G / within R (off-diagonal mean),
  `div_ratio = div_gen / div_ref` (≈ 1 like real data, ≪ 1 = samples collapse to similar shapes).
"""
function generation_metrics(D_GR::AbstractMatrix, D_GG::AbstractMatrix, D_RR::AbstractMatrix)
    NG, NR = size(D_GR)
    @assert size(D_GG) == (NG, NG) && size(D_RR) == (NR, NR)
    NG == NR || @warn "1-NNA assumes |G| = |R| (got $NG vs $NR); with unequal sizes 50 % is not the ideal value"

    mmd = mean(minimum(D_GR, dims=1))                                  # for every reference: closest generated
    nn_ref_of_gen = vec(getindex.(argmin(D_GR, dims=2), 2))            # for every generated: closest reference
    cov = length(unique(nn_ref_of_gen)) / NR

    # 1-NNA on G ∪ R with leave-one-out (labels: 1 = generated, 0 = reference)
    D_all = [D_GG D_GR; permutedims(D_GR) D_RR]
    labels = [ones(Int, NG); zeros(Int, NR)]
    pred = one_nn(D_all, labels; exclude_self=true)
    correct = pred .== labels

    div_gen, div_ref = offdiag_mean(D_GG), offdiag_mean(D_RR)
    return (mmd = mmd, cov = cov, nna = mean(correct), nna_gen = mean(correct[1:NG]), nna_ref = mean(correct[NG+1:end]),
            nn_ref_of_gen = nn_ref_of_gen, div_gen = div_gen, div_ref = div_ref, div_ratio = div_gen / div_ref)
end

"Mean of the off-diagonal entries of a square matrix (mean distance between two different shapes)."
function offdiag_mean(D::AbstractMatrix)
    n = size(D, 1)
    return (sum(D) - sum(D[i, i] for i in 1:n)) / (n * (n - 1))
end
