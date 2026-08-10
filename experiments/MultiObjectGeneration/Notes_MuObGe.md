# Notes on Multi Object Generation

## MultiObject Generaiton

### second_test.jl --> deeper model with letent set instead of latent vector

~~~julia
julia> x̂_dup, logits_dup, _, _ = model(x_dup, mask_dup);

julia> mp, mg = hungarian_match(x̂_dup, x_dup, mask_dup)
([[4, 6, 9, 12]], [[1, 4, 3, 2]])

julia> matched_gt_for_slot = Dict(zip(mp[1], mg[1]))
Dict{Int64, Int64} with 4 entries:
  4  => 1
  6  => 4
  9  => 3
  12 => 2

julia> 

julia> println("\nAll $N_MAX slots (existence = σ(logit); >0.5 means the model thinks this slot is real):")

All 12 slots (existence = σ(logit); >0.5 means the model thinks this slot is real):

julia> for slot in 1:N_MAX
           exist_prob = round(Flux.sigmoid(logits_dup[1, slot, 1]); digits=2)
           pred_class = nearest_class(x̂_dup[:, slot, 1], μs_valid, ys_valid)
           if haskey(matched_gt_for_slot, slot)
               gt = matched_gt_for_slot[slot]
                       gt_class = ys_valid[dup_idx[gt]]
               dist = round(sqrt(sum(abs2, x̂_dup[:, slot, 1] .- x_dup[:, gt, 1])); digits=3)
               println("slot $slot [MATCHED,   existence=$exist_prob] -> gt class $gt_class | nearest-neighbor predicted class $pred_class | L2 dist $dist")
           else
               println("slot $slot [unmatched, existence=$exist_prob] -> nearest-neighbor predicted class $pred_class (no ground truth to compare against)")
           end
       end
slot 1 [unmatched, existence=0.19] -> nearest-neighbor predicted class 7 (no ground truth to compare against)
slot 2 [unmatched, existence=0.27] -> nearest-neighbor predicted class 9 (no ground truth to compare against)
slot 3 [unmatched, existence=0.15] -> nearest-neighbor predicted class 9 (no ground truth to compare against)
slot 4 [MATCHED,   existence=0.81] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 1.86
slot 5 [unmatched, existence=0.16] -> nearest-neighbor predicted class 7 (no ground truth to compare against)
slot 6 [MATCHED,   existence=0.67] -> gt class 7 | nearest-neighbor predicted class 7 | L2 dist 0.826
slot 7 [unmatched, existence=0.17] -> nearest-neighbor predicted class 4 (no ground truth to compare against)
slot 8 [unmatched, existence=0.09] -> nearest-neighbor predicted class 7 (no ground truth to compare against)
slot 9 [MATCHED,   existence=0.76] -> gt class 2 | nearest-neighbor predicted class 2 | L2 dist 0.652
slot 10 [unmatched, existence=0.15] -> nearest-neighbor predicted class 9 (no ground truth to compare against)
slot 11 [unmatched, existence=0.16] -> nearest-neighbor predicted class 7 (no ground truth to compare against)
slot 12 [MATCHED,   existence=0.51] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 1.512

julia> σ.(logits_dup) .>= 0.5
1×12×1 BitArray{3}:
[:, :, 1] =
 0  0  0  1  0  1  0  0  1  0  0  1

julia> sum(σ.(logits_dup) .>= 0.5)
4

julia> active_slots = findall(vec(σ.(logits_dup[1, :, 1]) .>= 0.5f0))
4-element Vector{Int64}:
  4
  6
  9
 12

julia> println("\nThreshold-only view: $(length(active_slots))/$N_MAX slots predicted active (existence >= 0.5), vs $(length(dup_idx)) true objects")

Threshold-only view: 4/12 slots predicted active (existence >= 0.5), vs 4 true objects

julia> for slot in active_slots
           gt_dists = [sqrt(sum(abs2, x̂_dup[:, slot, 1] .- x_dup[:, g, 1])) for g in 1:length(dup_idx)]
           closest_gt = argmin(gt_dists)
           closest_gt_class = ys_valid[dup_idx[closest_gt]]
           println("slot $slot -> closest ground truth: class $closest_gt_class (gt position $closest_gt) | L2 dist $(round(gt_dists[closest_gt]; digits=3))")
       end
slot 4 -> closest ground truth: class 1 (gt position 1) | L2 dist 1.86
slot 6 -> closest ground truth: class 7 (gt position 4) | L2 dist 0.826
slot 9 -> closest ground truth: class 2 (gt position 3) | L2 dist 0.652
slot 12 -> closest ground truth: class 1 (gt position 2) | L2 dist 1.512
~~~


### first_test.jl

~~~julia
julia> x_dup = zeros(Float32, EMBED_DIM, N_MAX, 1)
8×12×1 Array{Float32, 3}:
[:, :, 1] =
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0

julia> mask_dup = falses(1, N_MAX, 1)
1×12×1 BitArray{3}:
[:, :, 1] =
 0  0  0  0  0  0  0  0  0  0  0  0

julia> x_dup[:, 1:length(dup_idx), 1] .= μs_valid[:, dup_idx]
8×4 view(::Array{Float32, 3}, :, 1:4, 1) with eltype Float32:
  0.626752  -1.34975     0.578244    1.35428
 -0.962945  -1.76819    -0.141029    0.669627
  0.848421  -1.19736     0.0430917   0.304179
  1.51769   -0.642061    0.531055   -2.00413
 -0.900578  -0.692064   -2.52954    -1.57167
  1.47371    1.43464    -1.07251     2.26428
  0.47186    0.809609    1.69584     1.08115
 -0.690653   0.0707071   0.756924   -0.568717

julia> mask_dup[1, 1:length(dup_idx), 1] .= true
4-element view(::BitArray{3}, 1, 1:4, 1) with eltype Bool:
 1
 1
 1
 1

julia> println("\nGround truth bag classes: ", ys_valid[dup_idx])

Ground truth bag classes: [1, 1, 2, 7]

julia> 

julia> x̂_dup, logits_dup, _, _ = model(x_dup, mask_dup);

julia> mp, mg = hungarian_match(x̂_dup, x_dup, mask_dup)
([[1, 7, 8, 11]], [[2, 4, 1, 3]])

julia> matched_gt_for_slot = Dict(zip(mp[1], mg[1]))
Dict{Int64, Int64} with 4 entries:
  7  => 4
  11 => 3
  8  => 1
  1  => 2


julia> for slot in 1:N_MAX
           exist_prob = round(Flux.sigmoid(logits_dup[1, slot, 1]); digits=2)
           pred_class = nearest_class(x̂_dup[:, slot, 1], μs_valid, ys_valid)
           if haskey(matched_gt_for_slot, slot)
               gt = matched_gt_for_slot[slot]
               gt_class = ys_valid[dup_idx[gt]]
               dist = round(sqrt(sum(abs2, x̂_dup[:, slot, 1] .- x_dup[:, gt, 1])); digits=3)
               println("slot $slot [MATCHED,   existence=$exist_prob] -> gt class $gt_class | nearest-neighbor predicted class $pred_class | L2 dist $dist")
           else
               println("slot $slot [unmatched, existence=$exist_prob] -> nearest-neighbor predicted class $pred_class (no ground truth to compare against)")
           end
       end
slot 1 [MATCHED,   existence=0.6] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 1.85
slot 2 [unmatched, existence=0.18] -> nearest-neighbor predicted class 9 (no ground truth to compare against)
slot 3 [unmatched, existence=0.17] -> nearest-neighbor predicted class 8 (no ground truth to compare against)
slot 4 [unmatched, existence=0.19] -> nearest-neighbor predicted class 9 (no ground truth to compare against)
slot 5 [unmatched, existence=0.23] -> nearest-neighbor predicted class 8 (no ground truth to compare against)
slot 6 [unmatched, existence=0.19] -> nearest-neighbor predicted class 8 (no ground truth to compare against)
slot 7 [MATCHED,   existence=0.47] -> gt class 7 | nearest-neighbor predicted class 7 | L2 dist 1.996
slot 8 [MATCHED,   existence=0.57] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 1.343
slot 9 [unmatched, existence=0.37] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 10 [unmatched, existence=0.29] -> nearest-neighbor predicted class 1 (no ground truth to compare against)
slot 11 [MATCHED,   existence=0.62] -> gt class 2 | nearest-neighbor predicted class 2 | L2 dist 1.641
slot 12 [unmatched, existence=0.6] -> nearest-neighbor predicted class 2 (no ground truth to compare against)

julia> 

julia> σ.(logits_dup) .>= 0.5
1×12×1 BitArray{3}:
[:, :, 1] =
 1  0  0  0  0  0  0  1  0  0  1  1
~~~
- this seems to be working now. I am not completly sold on idea yet.


## Embeddings for training
- I need emebeddings for different objects so I try simulate its generation. I decided to used mnist again and for pretraining I will use image version of mnist, because we need only embedding from latent space of VAE and it doesn't matter if it is images or sets. (plus: I did not train model on normal data in a while.)

- [x] Pre-train VAE 
    - [x] build VAE for mnist
    - [x] train vae
    - [x] save embeddings of numbers

 
