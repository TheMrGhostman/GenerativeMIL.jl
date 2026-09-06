# Notes on Multi Object Generation


## Embeddings for training
- I need emebeddings for different objects so I try simulate its generation. I decided to used mnist again and for pretraining I will use image version of mnist, because we need only embedding from latent space of VAE and it doesn't matter if it is images or sets. (plus: I did not train model on normal data in a while.)

- [x] Pre-train VAE 
    - [x] build VAE for mnist
    - [x] train vae
    - [x] save embeddings of numbers

 

## MultiObject Generaiton

### SlotQueryVAE.jl

#### 100 epochs
~~~julia
Epoch 100 | train: (ℒ = 5.0168705f0, ℒ_rec = 3.6291149f0, ℒ_exist = 0.562639f0, ℒ_kld = 26.247772f0, β = 0.01f0, matched_frac = 0.6484375f0)

julia> x̂_dup, logits_dup, _, _ = model(x_dup, mask_dup);

julia> mp, mg = hungarian_match(x̂_dup, x_dup, mask_dup)
([[3, 6, 7, 11]], [[3, 4, 1, 2]])

julia> matched_gt_for_slot = Dict(zip(mp[1], mg[1]))
Dict{Int64, Int64} with 4 entries:
  6  => 4
  7  => 1
  11 => 2
  3  => 3


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
slot 1 [unmatched, existence=0.43] -> nearest-neighbor predicted class 9 (no ground truth to compare against)
slot 2 [unmatched, existence=0.37] -> nearest-neighbor predicted class 4 (no ground truth to compare against)
slot 3 [MATCHED,   existence=0.27] -> gt class 2 | nearest-neighbor predicted class 2 | L2 dist 1.693
slot 4 [unmatched, existence=0.34] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 5 [unmatched, existence=0.15] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 6 [MATCHED,   existence=0.66] -> gt class 7 | nearest-neighbor predicted class 7 | L2 dist 1.297
slot 7 [MATCHED,   existence=0.63] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 1.522
slot 8 [unmatched, existence=0.46] -> nearest-neighbor predicted class 9 (no ground truth to compare against)
slot 9 [unmatched, existence=0.42] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 10 [unmatched, existence=0.5] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 11 [MATCHED,   existence=0.71] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 2.137
slot 12 [unmatched, existence=0.26] -> nearest-neighbor predicted class 2 (no ground truth to compare against)

julia> 

julia> σ.(logits_dup) .>= 0.5
1×12×1 BitArray{3}:
[:, :, 1] =
 0  0  0  0  0  1  1  0  0  1  1  0

julia> active_slots = findall(vec(σ.(logits_dup[1, :, 1]) .>= 0.5f0))
4-element Vector{Int64}:
  6
  7
 10
 11

julia> println("\nThreshold-only view: $(length(active_slots))/$N_MAX slots predicted active (existence >= 0.5), vs $(length(dup_idx)) true objects")

Threshold-only view: 4/12 slots predicted active (existence >= 0.5), vs 4 true objects

julia> for slot in active_slots
           gt_dists = [sqrt(sum(abs2, x̂_dup[:, slot, 1] .- x_dup[:, g, 1])) for g in 1:length(dup_idx)]
           closest_gt = argmin(gt_dists)
           closest_gt_class = ys_valid[dup_idx[closest_gt]]
               println("slot $slot -> closest ground truth: class $closest_gt_class (gt position $closest_gt) | L2 dist $(round(gt_dists[closest_gt]; digits=3))")
       end
slot 6 -> closest ground truth: class 7 (gt position 4) | L2 dist 1.297
slot 7 -> closest ground truth: class 1 (gt position 1) | L2 dist 1.522
slot 10 -> closest ground truth: class 2 (gt position 3) | L2 dist 1.784
slot 11 -> closest ground truth: class 1 (gt position 1) | L2 dist 1.939 
~~~
- this seems to be working now. I am not completly sold on idea yet.

#### 200 epochs
~~~julia
Epoch 200 | train: (ℒ = 5.068699f0, ℒ_rec = 3.771557f0, ℒ_exist = 0.5125377f0, ℒ_kld = 27.20659f0, β = 0.01f0, matched_frac = 0.6744792f0)

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
slot 1 [unmatched, existence=0.63] -> nearest-neighbor predicted class 9 (no ground truth to compare against)
slot 2 [unmatched, existence=0.41] -> nearest-neighbor predicted class 8 (no ground truth to compare against)
slot 3 [MATCHED,   existence=0.53] -> gt class 2 | nearest-neighbor predicted class 3 | L2 dist 1.49
slot 4 [unmatched, existence=0.28] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 5 [unmatched, existence=0.19] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 6 [MATCHED,   existence=0.72] -> gt class 7 | nearest-neighbor predicted class 7 | L2 dist 1.154
slot 7 [MATCHED,   existence=0.64] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 1.549
slot 8 [unmatched, existence=0.46] -> nearest-neighbor predicted class 9 (no ground truth to compare against)
slot 9 [unmatched, existence=0.47] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 10 [unmatched, existence=0.55] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 11 [MATCHED,   existence=0.81] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 1.67
slot 12 [unmatched, existence=0.43] -> nearest-neighbor predicted class 1 (no ground truth to compare against)

julia> 

julia> σ.(logits_dup) .>= 0.5
1×12×1 BitArray{3}:
[:, :, 1] =
 1  0  1  0  0  1  1  0  0  1  1  0

julia> sum(σ.(logits_dup) .>= 0.5)
6

julia> active_slots = findall(vec(σ.(logits_dup[1, :, 1]) .>= 0.5f0))
6-element Vector{Int64}:
  1
  3
  6
  7
 10
 11

julia> println("\nThreshold-only view: $(length(active_slots))/$N_MAX slots predicted active (existence >= 0.5), vs $(length(dup_idx)) true objects")

Threshold-only view: 6/12 slots predicted active (existence >= 0.5), vs 4 true objects

julia> for slot in active_slots
           gt_dists = [sqrt(sum(abs2, x̂_dup[:, slot, 1] .- x_dup[:, g, 1])) for g in 1:length(dup_idx)]
           closest_gt = argmin(gt_dists)
           closest_gt_class = ys_valid[dup_idx[closest_gt]]
           println("slot $slot -> closest ground truth: class $closest_gt_class (gt position $closest_gt) | L2 dist $(round(gt_dists[closest_gt]; digits=3))")
       end
slot 1 -> closest ground truth: class 7 (gt position 4) | L2 dist 2.586
slot 3 -> closest ground truth: class 2 (gt position 3) | L2 dist 1.49
slot 6 -> closest ground truth: class 7 (gt position 4) | L2 dist 1.154
slot 7 -> closest ground truth: class 1 (gt position 1) | L2 dist 1.549
slot 10 -> closest ground truth: class 2 (gt position 3) | L2 dist 1.921
slot 11 -> closest ground truth: class 1 (gt position 2) | L2 dist 1.67
~~~

- not stable after 200 epochs, really random N like 9, 8, 7, 10 ... to much objects



### DeepSlotQueryVAE.jl --> deeper model with letent set
- deeper model with letent set (or maybe matrix but unordered so -> set) instead of latent vector

#### 100 epochs
~~~julia
Epoch 100 | train: (ℒ = 3.217992f0, ℒ_rec = 2.0927784f0, ℒ_exist = 0.39802346f0, ℒ_kld = 32.91666f0, β = 0.01f0, matched_frac = 0.6484375f0) | 
valid: (ℒᵥ = 3.6883626f0, ℒᵥ_rec = 2.4081383f0, ℒᵥ_exist = 0.47499183f0, ℒᵥ_kld = 33.024097f0, matched_fracᵥ = 0.65662205f0)

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
slot 1 [MATCHED,   existence=0.96] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 1.076
slot 2 [MATCHED,   existence=1.0] -> gt class 7 | nearest-neighbor predicted class 7 | L2 dist 0.271
slot 3 [unmatched, existence=0.25] -> nearest-neighbor predicted class 1 (no ground truth to compare against)
slot 4 [MATCHED,   existence=0.85] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 1.344
slot 5 [unmatched, existence=0.32] -> nearest-neighbor predicted class 1 (no ground truth to compare against)
slot 6 [unmatched, existence=0.15] -> nearest-neighbor predicted class 3 (no ground truth to compare against)
slot 7 [unmatched, existence=0.09] -> nearest-neighbor predicted class 1 (no ground truth to compare against)
slot 8 [MATCHED,   existence=0.87] -> gt class 2 | nearest-neighbor predicted class 2 | L2 dist 0.88
slot 9 [unmatched, existence=0.44] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 10 [unmatched, existence=0.47] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 11 [unmatched, existence=0.12] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 12 [unmatched, existence=0.83] -> nearest-neighbor predicted class 1 (no ground truth to compare against)

julia> 

julia> σ.(logits_dup) .>= 0.5
1×12×1 BitArray{3}:
[:, :, 1] =
 1  1  0  1  0  0  0  1  0  0  0  1

julia> sum(σ.(logits_dup) .>= 0.5)
5

julia> active_slots = findall(vec(σ.(logits_dup[1, :, 1]) .>= 0.5f0))
5-element Vector{Int64}:
  1
  2
  4
  8
 12

julia> println("\nThreshold-only view: $(length(active_slots))/$N_MAX slots predicted active (existence >= 0.5), vs $(length(dup_idx)) true objects")

Threshold-only view: 5/12 slots predicted active (existence >= 0.5), vs 4 true objects

julia> for slot in active_slots
           gt_dists = [sqrt(sum(abs2, x̂_dup[:, slot, 1] .- x_dup[:, g, 1])) for g in 1:length(dup_idx)]
           closest_gt = argmin(gt_dists)
           closest_gt_class = ys_valid[dup_idx[closest_gt]]
           println("slot $slot -> closest ground truth: class $closest_gt_class (gt position $closest_gt) | L2 dist $(round(gt_dists[closest_gt]; digits=3))")
       end
slot 1 -> closest ground truth: class 1 (gt position 2) | L2 dist 1.076
slot 2 -> closest ground truth: class 7 (gt position 4) | L2 dist 0.271
slot 4 -> closest ground truth: class 1 (gt position 1) | L2 dist 1.344
slot 8 -> closest ground truth: class 2 (gt position 3) | L2 dist 0.88
slot 12 -> closest ground truth: class 1 (gt position 2) | L2 dist 1.868
~~~

#### 200 epochs
~~~julia
Epoch 199 | train: (ℒ = 2.669004f0, ℒ_rec = 1.5651501f0, ℒ_exist = 0.36370692f0, ℒ_kld = 37.644005f0, β = 0.01f0, matched_frac = 0.65625f0) 
valid: (ℒᵥ = 3.8443434f0, ℒᵥ_rec = 2.401407f0, ℒᵥ_exist = 0.5348104f0, ℒᵥ_kld = 37.331604f0, matched_fracᵥ = 0.65662205f0)

Epoch 200 | train: (ℒ = 2.9041374f0, ℒ_rec = 1.8467029f0, ℒ_exist = 0.34192264f0, ℒ_kld = 37.358936f0, β = 0.01f0, matched_frac = 0.70442706f0)
valid: (ℒᵥ = 3.8778374f0, ℒᵥ_rec = 2.4217074f0, ℒᵥ_exist = 0.5415083f0, ℒᵥ_kld = 37.31131f0, matched_fracᵥ = 0.65662205f0)


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
slot 1 [unmatched, existence=0.99] -> nearest-neighbor predicted class 6 (no ground truth to compare against)
slot 2 [MATCHED,   existence=1.0] -> gt class 7 | nearest-neighbor predicted class 7 | L2 dist 0.392
slot 3 [unmatched, existence=0.17] -> nearest-neighbor predicted class 1 (no ground truth to compare against)
slot 4 [MATCHED,   existence=0.92] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 2.035
slot 5 [MATCHED,   existence=0.2] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 0.882
slot 6 [unmatched, existence=0.33] -> nearest-neighbor predicted class 1 (no ground truth to compare against)
slot 7 [unmatched, existence=0.02] -> nearest-neighbor predicted class 7 (no ground truth to compare against)
slot 8 [MATCHED,   existence=0.97] -> gt class 2 | nearest-neighbor predicted class 2 | L2 dist 0.511
slot 9 [unmatched, existence=0.34] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 10 [unmatched, existence=0.37] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 11 [unmatched, existence=0.13] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 12 [unmatched, existence=0.91] -> nearest-neighbor predicted class 1 (no ground truth to compare against)

julia> 

julia> σ.(logits_dup) .>= 0.5
1×12×1 BitArray{3}:
[:, :, 1] =
 1  1  0  1  0  0  0  1  0  0  0  1

julia> sum(σ.(logits_dup) .>= 0.5)
5

julia> active_slots = findall(vec(σ.(logits_dup[1, :, 1]) .>= 0.5f0))
5-element Vector{Int64}:
  1
  2
  4
  8
 12

julia> println("\nThreshold-only view: $(length(active_slots))/$N_MAX slots predicted active (existence >= 0.5), vs $(length(dup_idx)) true objects")

Threshold-only view: 5/12 slots predicted active (existence >= 0.5), vs 4 true objects

julia> for slot in active_slots
           gt_dists = [sqrt(sum(abs2, x̂_dup[:, slot, 1] .- x_dup[:, g, 1])) for g in 1:length(dup_idx)]
           closest_gt = argmin(gt_dists)
           closest_gt_class = ys_valid[dup_idx[closest_gt]]
           println("slot $slot -> closest ground truth: class $closest_gt_class (gt position $closest_gt) | L2 dist $(round(gt_dists[closest_gt]; digits=3))")
       end
slot 1 -> closest ground truth: class 1 (gt position 2) | L2 dist 1.104
slot 2 -> closest ground truth: class 7 (gt position 4) | L2 dist 0.392
slot 4 -> closest ground truth: class 1 (gt position 1) | L2 dist 2.035
slot 8 -> closest ground truth: class 2 (gt position 3) | L2 dist 0.511
slot 12 -> closest ground truth: class 1 (gt position 2) | L2 dist 1.959



x̂_gen, logits_gen = generate(model, 200; m_z=args.m_z)
collapse_dists = Float32[]
for b in 1:size(x̂_gen, 3)
    S = x̂_gen[:, :, b]
    for i in 1:size(S, 2), j in i+1:size(S, 2)
        push!(collapse_dists, sqrt(sum(abs2, S[:, i] .- S[:, j])))
    end
end
println("\nMean pairwise slot distance (unconditional generation): ", mean(collapse_dists), " (near 0 => collapse)")

julia> for b in 1:10
           active = findall(vec(logits_gen[1, :, b]) .> 0)
           classes = [nearest_class(x̂_gen[:, s, b], μs_valid, ys_valid) for s in active]
           println("sample $b -> $(sort(classes))")
       end
sample 1 -> [0, 9, 9]
sample 2 -> [2, 4, 4, 7, 7, 8, 9]
sample 3 -> [8, 9]
sample 4 -> [2, 4, 4, 4]
sample 5 -> [0, 0, 0, 0, 2, 4, 6, 8, 9]
sample 6 -> [5, 5, 6]
sample 7 -> [4, 4, 6, 6, 8, 8, 9]
sample 8 -> [1, 2, 4, 8, 9]
sample 9 -> [0, 0, 0, 0, 1, 2, 3, 3, 4, 5, 9, 9]
sample 10 -> [0, 0, 0, 2, 3, 3, 3, 4, 5, 6, 8, 8]
~~~


### QueryDistSlotVAE.jl
- difference is that here we sample queries (but sampling fixed set of 12 queries - no iid!!!) instead of keeping them as learnable parameters. 

~~~julia
Epoch 100 | train: (ℒ = 3.6112556f0, ℒ_rec = 2.617477f0, ℒ_exist = 0.47784224f0, ℒ_kld = 38.09437f0, β = 0.001f0, matched_frac = 0.6588542f0)

julia> x̂_dup, logits_dup, _, _ = model(x_dup, mask_dup);

julia> mp, mg = hungarian_match(x̂_dup, x_dup, mask_dup)
([[4, 8, 9, 10]], [[3, 1, 2, 4]])

julia> matched_gt_for_slot = Dict(zip(mp[1], mg[1]))
Dict{Int64, Int64} with 4 entries:
  4  => 3
  10 => 4
  9  => 2
  8  => 1

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
slot 1 [unmatched, existence=0.25] -> nearest-neighbor predicted class 3 (no ground truth to compare against)
slot 2 [unmatched, existence=0.35] -> nearest-neighbor predicted class 3 (no ground truth to compare against)
slot 3 [unmatched, existence=0.26] -> nearest-neighbor predicted class 8 (no ground truth to compare against)
slot 4 [MATCHED,   existence=0.69] -> gt class 2 | nearest-neighbor predicted class 2 | L2 dist 1.326
slot 5 [unmatched, existence=0.26] -> nearest-neighbor predicted class 3 (no ground truth to compare against)
slot 6 [unmatched, existence=0.31] -> nearest-neighbor predicted class 7 (no ground truth to compare against)
slot 7 [unmatched, existence=0.91] -> nearest-neighbor predicted class 1 (no ground truth to compare against)
slot 8 [MATCHED,   existence=0.38] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 1.01
slot 9 [MATCHED,   existence=0.88] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 0.947
slot 10 [MATCHED,   existence=0.94] -> gt class 7 | nearest-neighbor predicted class 7 | L2 dist 1.134
slot 11 [unmatched, existence=0.9] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 12 [unmatched, existence=0.4] -> nearest-neighbor predicted class 7 (no ground truth to compare against)

julia> 

julia> σ.(logits_dup) .>= 0.5
1×12×1 BitArray{3}:
[:, :, 1] =
 0  0  0  1  0  0  1  0  1  1  1  0

julia> 

julia> sum(σ.(logits_dup) .>= 0.5)
5

julia> active_slots = findall(vec(σ.(logits_dup[1, :, 1]) .>= 0.5f0))
5-element Vector{Int64}:
  4
  7
  9
 10
 11

julia> println("\nThreshold-only view: $(length(active_slots))/$N_MAX slots predicted active (existence >= 0.5), vs $(length(dup_idx)) true objects")

Threshold-only view: 5/12 slots predicted active (existence >= 0.5), vs 4 true objects

julia> for slot in active_slots
           gt_dists = [sqrt(sum(abs2, x̂_dup[:, slot, 1] .- x_dup[:, g, 1])) for g in 1:length(dup_idx)]
           closest_gt = argmin(gt_dists)
           closest_gt_class = ys_valid[dup_idx[closest_gt]]
           println("slot $slot -> closest ground truth: class $closest_gt_class (gt position $closest_gt) | L2 dist $(round(gt_dists[closest_gt]; digits=3))")
       end
slot 4 -> closest ground truth: class 2 (gt position 3) | L2 dist 1.326
slot 7 -> closest ground truth: class 1 (gt position 2) | L2 dist 1.973
slot 9 -> closest ground truth: class 1 (gt position 2) | L2 dist 0.947
slot 10 -> closest ground truth: class 7 (gt position 4) | L2 dist 1.134
slot 11 -> closest ground truth: class 2 (gt position 3) | L2 dist 1.492
~~~


### DeepQueryDistSlotVAE.jl
- this model is deeper version of previous one but still with single vector latent

#### 100 epochs
~~~julia
Epoch 100 | train: (ℒ = 3.5294573f0, ℒ_rec = 2.6130986f0, ℒ_exist = 0.45707044f0, ℒ_kld = 0.22177254f0, β = 0.01f0, matched_frac = 0.64453125f0)

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
slot 1 [unmatched, existence=0.22] -> nearest-neighbor predicted class 7 (no ground truth to compare against)
slot 2 [unmatched, existence=0.09] -> nearest-neighbor predicted class 4 (no ground truth to compare against)
slot 3 [MATCHED,   existence=0.41] -> gt class 7 | nearest-neighbor predicted class 7 | L2 dist 1.311
slot 4 [unmatched, existence=0.25] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 5 [MATCHED,   existence=0.66] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 1.137
slot 6 [unmatched, existence=0.5] -> nearest-neighbor predicted class 1 (no ground truth to compare against)
slot 7 [unmatched, existence=0.13] -> nearest-neighbor predicted class 1 (no ground truth to compare against)
slot 8 [MATCHED,   existence=0.29] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 1.486
slot 9 [unmatched, existence=0.6] -> nearest-neighbor predicted class 7 (no ground truth to compare against)
slot 10 [unmatched, existence=0.73] -> nearest-neighbor predicted class 7 (no ground truth to compare against)
slot 11 [unmatched, existence=0.09] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 12 [MATCHED,   existence=0.85] -> gt class 2 | nearest-neighbor predicted class 2 | L2 dist 1.544

julia> 

julia> σ.(logits_dup) .>= 0.5
1×12×1 BitArray{3}:
[:, :, 1] =
 0  0  0  0  1  0  0  0  1  1  0  1

julia> sum(σ.(logits_dup) .>= 0.5)
4

julia> active_slots = findall(vec(σ.(logits_dup[1, :, 1]) .>= 0.5f0))
4-element Vector{Int64}:
  5
  9
 10
 12

julia> println("\nThreshold-only view: $(length(active_slots))/$N_MAX slots predicted active (existence >= 0.5), vs $(length(dup_idx)) true objects")

Threshold-only view: 4/12 slots predicted active (existence >= 0.5), vs 4 true objects

julia> for slot in active_slots
           gt_dists = [sqrt(sum(abs2, x̂_dup[:, slot, 1] .- x_dup[:, g, 1])) for g in 1:length(dup_idx)]
           closest_gt = argmin(gt_dists)
           closest_gt_class = ys_valid[dup_idx[closest_gt]]
           println("slot $slot -> closest ground truth: class $closest_gt_class (gt position $closest_gt) | L2 dist $(round(gt_dists[closest_gt]; digits=3))")
       end
slot 5 -> closest ground truth: class 1 (gt position 2) | L2 dist 1.137
slot 9 -> closest ground truth: class 1 (gt position 1) | L2 dist 2.762
slot 10 -> closest ground truth: class 7 (gt position 4) | L2 dist 2.151
slot 12 -> closest ground truth: class 2 (gt position 3) | L2 dist 1.544
~~~

#### 200 epochs (not working as intended now)

~~~julia
Epoch 200 | train: (ℒ = 2.9667149f0, ℒ_rec = 2.1799822f0, ℒ_exist = 0.39321414f0, ℒ_kld = 0.030456733f0, β = 0.01f0, matched_frac = 0.65494794f0)

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
slot 1 [unmatched, existence=0.22] -> nearest-neighbor predicted class 1 (no ground truth to compare against)
slot 2 [unmatched, existence=0.49] -> nearest-neighbor predicted class 5 (no ground truth to compare against)
slot 3 [unmatched, existence=0.43] -> nearest-neighbor predicted class 7 (no ground truth to compare against)
slot 4 [unmatched, existence=0.08] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 5 [MATCHED,   existence=0.51] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 1.199
slot 6 [unmatched, existence=0.7] -> nearest-neighbor predicted class 1 (no ground truth to compare against)
slot 7 [unmatched, existence=0.15] -> nearest-neighbor predicted class 8 (no ground truth to compare against)
slot 8 [MATCHED,   existence=0.57] -> gt class 1 | nearest-neighbor predicted class 1 | L2 dist 1.634
slot 9 [unmatched, existence=0.65] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 10 [MATCHED,   existence=0.65] -> gt class 7 | nearest-neighbor predicted class 7 | L2 dist 1.162
slot 11 [unmatched, existence=0.14] -> nearest-neighbor predicted class 2 (no ground truth to compare against)
slot 12 [MATCHED,   existence=0.97] -> gt class 2 | nearest-neighbor predicted class 2 | L2 dist 1.38

julia> 

julia> σ.(logits_dup) .>= 0.5
1×12×1 BitArray{3}:
[:, :, 1] =
 0  0  0  0  1  1  0  1  1  1  0  1

julia> sum(σ.(logits_dup) .>= 0.5)
6

julia> active_slots = findall(vec(σ.(logits_dup[1, :, 1]) .>= 0.5f0))
6-element Vector{Int64}:
  5
  6
  8
  9
 10
 12

julia> println("\nThreshold-only view: $(length(active_slots))/$N_MAX slots predicted active (existence >= 0.5), vs $(length(dup_idx)) true objects")

Threshold-only view: 6/12 slots predicted active (existence >= 0.5), vs 4 true objects

julia> for slot in active_slots
           gt_dists = [sqrt(sum(abs2, x̂_dup[:, slot, 1] .- x_dup[:, g, 1])) for g in 1:length(dup_idx)]
           closest_gt = argmin(gt_dists)
           closest_gt_class = ys_valid[dup_idx[closest_gt]]
           println("slot $slot -> closest ground truth: class $closest_gt_class (gt position $closest_gt) | L2 dist $(round(gt_dists[closest_gt]; digits=3))")
       end
slot 5 -> closest ground truth: class 1 (gt position 2) | L2 dist 1.199
slot 6 -> closest ground truth: class 1 (gt position 1) | L2 dist 1.706
slot 8 -> closest ground truth: class 1 (gt position 1) | L2 dist 1.634
slot 9 -> closest ground truth: class 2 (gt position 3) | L2 dist 2.788
slot 10 -> closest ground truth: class 7 (gt position 4) | L2 dist 1.162
slot 12 -> closest ground truth: class 2 (gt position 3) | L2 dist 1.38
~~~

### TODO: difference between SlotQueryVAE and shallow SetVAE? 
- (?) look at difference between SlotQueryVAE and shallow SetVAE? 
  - (?) SetVAE has slot attention and slot-wise latent - what is difference to SlotQueryVAE's attention
  - (?) difference between slot-attention and PMA pooling
- (?) look at difference between priors? 
  - (?) learned prior querries (but fixed) vs learned mixture prior. What is theoretical difference?
  - I know that PQ are fixed and MP are sampled. 
  - Idea: samples from MP are iid and we do not influence sampling. learned PQ are probably strategicaly spread within space. 
- (?) are those two models generally the same, with differences of existence prediction?
  - when I thought about it now, it seems to be the same idea/set of ideas. But SlotQueryVAE has existence prediction on slots.  


# HSVAE 


### Speed testing of Chamfer_pairwise_distance on CPU vs GPU on different sizes
Loss is sum(chamfer_distance_clusters) / i.e. without hungarian matching

#### Data size ~ 
##### Forward: 
~~~julia

~~~
#### Backward: 
~~~julia

~~~

#### Data size ~ (3, 256, 12, 16)
##### Forward: 
~~~julia
julia> @benchmark forward_and_loss($model2_cpu, $xx, $xx_mask)
BenchmarkTools.Trial: 2 samples with 1 evaluation per sample.
 Range (min … max):  3.009 s …   3.137 s  ┊ GC (min … max): 24.95% … 27.03%
 Time  (median):     3.073 s              ┊ GC (median):    26.01%
 Time  (mean ± σ):   3.073 s ± 90.838 ms  ┊ GC (mean ± σ):  26.01% ±  1.46%

  █                                                       █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  3.01 s         Histogram: frequency by time        3.14 s <

 Memory estimate: 1.88 GiB, allocs estimate: 1275.

julia> @benchmark forward_and_loss($model2_gpu, $xxc, $xxc_mask)
BenchmarkTools.Trial: 187 samples with 1 evaluation per sample.
 Range (min … max):  26.111 ms …  34.753 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     26.524 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   26.746 ms ± 833.662 μs  ┊ GC (mean ± σ):  6.35% ± 9.52%

    ▄▆█▄█▃▃▅▂    ▁                                              
  ▄▄█████████▅▆▄▇█▆▅▃▅▁▁▃▁▄▁▄▁▃▁▃▁▁▃▁▁▁▁▁▃▃▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▃▄ ▃
  26.1 ms         Histogram: frequency by time         29.4 ms <

 Memory estimate: 558.56 KiB, allocs estimate: 14656.
~~~
#### Backward: 
~~~julia
julia> @benchmark Zygote.gradient(m -> forward_and_loss(m, $xx, $xx_mask), $model2_cpu)
BenchmarkTools.Trial: 2 samples with 1 evaluation per sample.
 Range (min … max):  3.026 s …    3.624 s  ┊ GC (min … max): 24.85% … 36.64%
 Time  (median):     3.325 s               ┊ GC (median):    31.28%
 Time  (mean ± σ):   3.325 s ± 422.719 ms  ┊ GC (mean ± σ):  31.28% ±  8.34%

  █                                                        █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  3.03 s         Histogram: frequency by time         3.62 s <

 Memory estimate: 2.04 GiB, allocs estimate: 17787.

julia> @benchmark Zygote.gradient(m -> forward_and_loss(m, $xxc, $xxc_mask), $model2_gpu)
BenchmarkTools.Trial: 68 samples with 1 evaluation per sample.
 Range (min … max):  71.764 ms … 78.859 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     73.619 ms              ┊ GC (median):    7.58%
 Time  (mean ± σ):   74.136 ms ±  1.511 ms  ┊ GC (mean ± σ):  8.42% ± 8.54%

          ▁▃▁▃█ ▃▃  ▁▃                     ▁                   
  ▄▁▁▁▁▁▄▄█████▇██▁▄██▄▁▄▄▄▁▇▄▁▄▄▇▁▁▄▁▄▁▄▇▁█▄▄▄▁▄▁▄▁▁▁▁▄▁▁▁▁▄ ▁
  71.8 ms         Histogram: frequency by time        77.9 ms <

 Memory estimate: 44.47 MiB, allocs estimate: 58478.
~~~


#### Data size ~ (3, 256, 12, 32)
##### Forward: 
~~~julia
julia> @benchmark forward_and_loss($model2_cpu, $xx, $xx_mask)
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 6.136 s (25.77% GC) to evaluate,
 with a memory estimate of 3.75 GiB, over 1563 allocations.

julia> @benchmark forward_and_loss($model2_gpu, $xxc, $xxc_mask)
BenchmarkTools.Trial: 105 samples with 1 evaluation per sample.
 Range (min … max):  46.571 ms …  51.872 ms  ┊ GC (min … max): 7.14% … 10.89%
 Time  (median):     47.954 ms               ┊ GC (median):    7.31%
 Time  (mean ± σ):   48.051 ms ± 863.343 μs  ┊ GC (mean ± σ):  6.45% ±  4.78%

       ▃▁▁    █▁▄▆▄█▆▄   ▁                                      
  ▄▁▁▁▄███▇▆▇▆████████▇▇▆█▆▆▄▁▁▆▄▁▄▄▄▄▁▁▁▁▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▁▄ ▄
  46.6 ms         Histogram: frequency by time         51.6 ms <

 Memory estimate: 564.83 KiB, allocs estimate: 14931.
~~~
#### Backward: 
~~~julia
julia> @benchmark Zygote.gradient(m -> forward_and_loss(m, $xx, $xx_mask), $model2_cpu)
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 6.167 s (24.76% GC) to evaluate,
 with a memory estimate of 4.09 GiB, over 18085 allocations.

julia> @benchmark Zygote.gradient(m -> forward_and_loss(m, $xxc, $xxc_mask), $model2_gpu)
BenchmarkTools.Trial: 42 samples with 1 evaluation per sample.
 Range (min … max):  117.432 ms … 123.394 ms  ┊ GC (min … max): 8.74% … 11.25%
 Time  (median):     118.966 ms               ┊ GC (median):    9.37%
 Time  (mean ± σ):   119.585 ms ±   1.703 ms  ┊ GC (mean ± σ):  9.40% ±  0.59%

   ▁▁    ▄▁▄ ▁ ▁      ▁         █                          ▁  ▁  
  ▆██▁▁▁▆███▆█▆█▁▆▆▁▁▁█▆▁▆▁▆▁▁▁▆█▁▁▁▆▆▁▁▁▆▁▁▆▁▆▆▁▁▁▁▁▁▁▁▁▁▁█▁▁█ ▁
  117 ms           Histogram: frequency by time          123 ms <

 Memory estimate: 86.12 MiB, allocs estimate: 60262.
~~~


#### Data size ~ (3, 256, 12, 64)
##### Forward: 
~~~julia
julia> @benchmark forward_and_loss($model2_cpu, $xx, $xx_mask)
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 12.869 s (20.05% GC) to evaluate,
 with a memory estimate of 7.51 GiB, over 2165 allocations.

julia> @benchmark forward_and_loss($model2_gpu, $xxc, $xxc_mask)
BenchmarkTools.Trial: 51 samples with 1 evaluation per sample.
 Range (min … max):  93.271 ms … 107.667 ms  ┊ GC (min … max): 2.06% … 5.04%
 Time  (median):     99.674 ms               ┊ GC (median):    3.31%
 Time  (mean ± σ):   99.248 ms ±   2.614 ms  ┊ GC (mean ± σ):  3.83% ± 1.52%

  ▁      ▁▁                     ▄▄█▁ █▄█▄ █▄                    
  █▁▆▁▁▁▁██▁▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▆▁████▆████▆██▆▆▆▁▁▆▁▁▁▆▁▁▁▁▁▁▁▆ ▁
  93.3 ms         Histogram: frequency by time          104 ms <

 Memory estimate: 575.56 KiB, allocs estimate: 15609.
~~~
#### Backward: 
~~~julia
julia> @benchmark Zygote.gradient(m -> forward_and_loss(m, $xx, $xx_mask), $model2_cpu)
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 12.343 s (24.44% GC) to evaluate,
 with a memory estimate of 8.17 GiB, over 18818 allocations.

julia> @benchmark Zygote.gradient(m -> forward_and_loss(m, $xxc, $xxc_mask), $model2_gpu)
BenchmarkTools.Trial: 7 samples with 1 evaluation per sample.
 Range (min … max):  588.072 ms … 868.158 ms  ┊ GC (min … max): 5.39% … 4.08%
 Time  (median):     774.590 ms               ┊ GC (median):    4.13%
 Time  (mean ± σ):   753.030 ms ± 114.592 ms  ┊ GC (mean ± σ):  4.26% ± 1.16%

  ▁ ▁                                     █           ▁ ▁     ▁  
  █▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁█▁█▁▁▁▁▁█ ▁
  588 ms           Histogram: frequency by time          868 ms <

 Memory estimate: 169.46 MiB, allocs estimate: 64457.
~~~


#### Data size ~ (3, 512, 12, 1)
##### Forward: 
~~~julia
julia> @benchmark forward_and_loss($model2_cpu, $xx, $xx_mask)
BenchmarkTools.Trial: 7 samples with 1 evaluation per sample.
 Range (min … max):  540.880 ms …    1.226 s  ┊ GC (min … max):  0.00% … 56.07%
 Time  (median):     603.110 ms               ┊ GC (median):     9.20%
 Time  (mean ± σ):   763.389 ms ± 280.232 ms  ┊ GC (mean ± σ):  28.32% ± 22.63%

  █  ███       █                                    █         █  
  █▁▁███▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁█ ▁
  541 ms           Histogram: frequency by time          1.23 s <

 Memory estimate: 456.04 MiB, allocs estimate: 934.

julia> @benchmark forward_and_loss($model2_gpu, $xxc, $xxc_mask)
BenchmarkTools.Trial: 395 samples with 1 evaluation per sample.
 Range (min … max):  10.999 ms … 23.696 ms  ┊ GC (min … max): 0.00% … 38.74%
 Time  (median):     11.835 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   12.673 ms ±  2.558 ms  ┊ GC (mean ± σ):  6.48% ± 12.25%

  ▅█▆▆▅▆▅▃▃▁▁ ▁                                                
  ███████████▆██▆▄▁▁▁▁▁▁▁▁▁▄▁▄▁▁▆▄▁▁▄▆▄▄▆▄▆▆▆▄▆▆▆▄▁▄▄▆▄▁▁▄▄▄▄ ▆
  11 ms        Histogram: log(frequency) by time      22.5 ms <

 Memory estimate: 548.42 KiB, allocs estimate: 14016.
~~~
#### Backward: 
~~~julia
julia> @benchmark Zygote.gradient(m -> forward_and_loss(m, $xx, $xx_mask), $model2_cpu)
BenchmarkTools.Trial: 6 samples with 1 evaluation per sample.
 Range (min … max):  589.117 ms …    1.420 s  ┊ GC (min … max):  5.64% … 59.21%
 Time  (median):     700.554 ms               ┊ GC (median):    16.54%
 Time  (mean ± σ):   875.752 ms ± 350.938 ms  ┊ GC (mean ± σ):  33.23% ± 24.31%

  ▁  █        ▁                                ▁              ▁  
  █▁▁█▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  589 ms           Histogram: frequency by time          1.42 s <

 Memory estimate: 478.48 MiB, allocs estimate: 17349.

julia> @benchmark Zygote.gradient(m -> forward_and_loss(m, $xxc, $xxc_mask), $model2_gpu)
BenchmarkTools.Trial: 115 samples with 1 evaluation per sample.
 Range (min … max):  34.815 ms … 95.944 ms  ┊ GC (min … max): 0.00% … 26.23%
 Time  (median):     39.148 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   43.556 ms ± 10.484 ms  ┊ GC (mean ± σ):  4.38% ±  8.00%

  ▂▁▇▃█▂                                                       
  ██████▇█▅▄▄▅▇▁▃▃▃▄▅▃▃▃▁▁▃▃▃▃▁▁▁▁▄▃▃▃▃▄▁▃▃▃▁▁▃▁▁▁▁▁▁▁▁▁▁▃▁▁▃ ▃
  34.8 ms         Histogram: frequency by time        78.1 ms <

 Memory estimate: 8.03 MiB, allocs estimate: 57269.
~~~



#### Data size ~ (3, 512, 12, 8)
##### Forward: 
~~~julia
julia> @benchmark forward_and_loss($model2_cpu, $xx, $xx_mask)
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 6.030 s (25.97% GC) to evaluate,
 with a memory estimate of 3.56 GiB, over 1124 allocations.

julia> @benchmark forward_and_loss($model2_gpu, $xxc, $xxc_mask)
BenchmarkTools.Trial: 109 samples with 1 evaluation per sample.
 Range (min … max):  44.517 ms … 51.843 ms  ┊ GC (min … max): 7.11% … 0.00%
 Time  (median):     45.385 ms              ┊ GC (median):    6.98%
 Time  (mean ± σ):   45.967 ms ±  1.550 ms  ┊ GC (mean ± σ):  5.98% ± 3.97%

      █▃▆▁▂ ▁▁                                                 
  ▇▄▇▆████████▃▆▇▆▄▃▄▃▇▃▁▃▁▁▁▁▁▁▁▁▁▁▃▁▁▃▁▁▁▁▁▁▃▁▁▃▃▁▃▁▃▃▃▁▃▃▃ ▃
  44.5 ms         Histogram: frequency by time        50.6 ms <

 Memory estimate: 561.09 KiB, allocs estimate: 14818.
~~~
#### Backward: 
~~~julia
julia> @benchmark Zygote.gradient(m -> forward_and_loss(m, $xx, $xx_mask), $model2_cpu)
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 5.999 s (23.15% GC) to evaluate,
 with a memory estimate of 3.73 GiB, over 17650 allocations.

julia> @benchmark Zygote.gradient(m -> forward_and_loss(m, $xxc, $xxc_mask), $model2_gpu)
BenchmarkTools.Trial: 54 samples with 1 evaluation per sample.
 Range (min … max):  90.572 ms … 100.359 ms  ┊ GC (min … max): 7.39% … 13.75%
 Time  (median):     93.492 ms               ┊ GC (median):    7.62%
 Time  (mean ± σ):   93.846 ms ±   2.253 ms  ┊ GC (mean ± σ):  6.85% ±  5.33%

    ▃ ▃     █▃█  ▃ ▃ ▃ ▃▃▃         ▃  ▃ ▃
  ▇▇█▇█▇▁▇▇▇███▇▁█▇█▇█▇███▇▁▇▇▁▇▇▇▇█▁▁█▁█▁▁▁▁▇▇▁▁▇▁▁▁▁▇▁▁▁▁▁▁▇ ▁
  90.6 ms         Histogram: frequency by time         99.5 ms <

 Memory estimate: 44.47 MiB, allocs estimate: 58785.
~~~

### Speed teseting of Hungarian_matching_loss on CPU vs GPU

#### Data size ~ (3, 256, 12, 8)
~~~julia
julia> @benchmark hungarian_matching_loss($y, $x, $x_mask, $rnd_pred)
BenchmarkTools.Trial: 4 samples with 1 evaluation per sample.
 Range (min … max):  984.508 ms …    1.771 s  ┊ GC (min … max):  8.38% … 48.77%
 Time  (median):        1.539 s               ┊ GC (median):    42.21%
 Time  (mean ± σ):      1.458 s ± 363.808 ms  ┊ GC (mean ± σ):  38.49% ± 18.98%

  █                            █                          █   █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁█ ▁
  985 ms           Histogram: frequency by time          1.77 s <

 Memory estimate: 770.86 MiB, allocs estimate: 801.

julia> @benchmark hungarian_matching_loss($yc, $xc, $xc_mask, $rnd_pred_gpu)
BenchmarkTools.Trial: 508 samples with 1 evaluation per sample.
 Range (min … max):  9.634 ms …  10.941 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     9.809 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   9.843 ms ± 169.381 μs  ┊ GC (mean ± σ):  1.69% ± 3.93%

       ▃▅▇▅▇█▄▇▅▁                                              
  ▂▃▃▇▅██████████▆▄▅▅▃▃▁▂▃▁▁▃▃▂▁▂▁▂▁▁▂▁▂▁▁▁▁▁▂▂▁▁▂▂▁▁▃▁▂▃▂▂▂▂ ▃
  9.63 ms         Histogram: frequency by time        10.6 ms <

 Memory estimate: 149.02 KiB, allocs estimate: 3361.
julia> 
~~~




### Speed teseting of Hungarian_matching_loss with model on CPU vs GPU

#### Data size ~ (3, 256, 12, 8)
##### Forward: 
~~~julia
julia> @benchmark model_elbo($model_cpu, $x, $x_mask)
BenchmarkTools.Trial: 4 samples with 1 evaluation per sample.
 Range (min … max):  1.305 s …    2.313 s  ┊ GC (min … max):  9.40% … 50.81%
 Time  (median):     1.748 s               ┊ GC (median):    35.95%
 Time  (mean ± σ):   1.779 s ± 413.461 ms  ┊ GC (mean ± σ):  35.91% ± 17.24%

  █                       █ █                              █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  1.3 s          Histogram: frequency by time         2.31 s <

 Memory estimate: 961.12 MiB, allocs estimate: 1681.

julia> @benchmark model_elbo($model_gpu, $xc, $xc_mask)
BenchmarkTools.Trial: 282 samples with 1 evaluation per sample.
 Range (min … max):  17.066 ms …  22.543 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     17.630 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   17.760 ms ± 516.107 μs  ┊ GC (mean ± σ):  4.38% ± 8.88%

        ▁▂▂▂▁█▂ ▂▄  ▂                                           
  ▃▁▄▁▃▆█████████████▃▇▇▄▅▄▆▇▄▃▄▄▄▃▃▃▃▁▄▃▄▃▁▃▁▃▃▃▁▁▃▃▁▃▁▃▁▁▃▁▃ ▃
  17.1 ms         Histogram: frequency by time         19.3 ms <

 Memory estimate: 639.89 KiB, allocs estimate: 15928.
~~~
#### Backward: 
~~~julia
julia> @benchmark compute_grad_cpu()
BenchmarkTools.Trial: 3 samples with 1 evaluation per sample.
 Range (min … max):  1.727 s …    2.277 s  ┊ GC (min … max): 33.99% … 49.90%
 Time  (median):     1.766 s               ┊ GC (median):    35.61%
 Time  (mean ± σ):   1.923 s ± 307.017 ms  ┊ GC (mean ± σ):  40.76% ±  8.75%

  █   █                                                    █  
  █▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  1.73 s         Histogram: frequency by time         2.28 s <

 Memory estimate: 1.02 GiB, allocs estimate: 18925.

julia> @benchmark compute_grad_gpu()
BenchmarkTools.Trial: 88 samples with 1 evaluation per sample.
 Range (min … max):  51.547 ms … 66.582 ms  ┊ GC (min … max): 0.00% … 19.43%
 Time  (median):     56.164 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   57.188 ms ±  3.861 ms  ┊ GC (mean ± σ):  4.27% ±  7.87%

     ▃ ▁        ▃▁█▄▆▁                                         
  ▄▆▆█▆█▄▁▄▁▄▁▁▁██████▆▇▆▆▁▆▁▁▄▁▆▄▁▁▄▁▄▁▁▄▄▄▄▄▆▆▆▇▁▁▁▁▁▁▁▁▄▁▆ ▁
  51.5 ms         Histogram: frequency by time        66.5 ms <

 Memory estimate: 23.82 MiB, allocs estimate: 62023.
~~~

#### Data size ~  (3, 512, 12, 8)
##### Forward: 
~~~julia
julia> @benchmark model_elbo($model_cpu, $x, $x_mask)
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 6.255 s (31.26% GC) to evaluate,
 with a memory estimate of 3.56 GiB, over 1688 allocations.

julia> @benchmark model_elbo($model_gpu, $xc, $xc_mask)
BenchmarkTools.Trial: 105 samples with 1 evaluation per sample.
 Range (min … max):  46.099 ms … 54.616 ms  ┊ GC (min … max): 3.46% … 0.00%
 Time  (median):     47.397 ms              ┊ GC (median):    5.27%
 Time  (mean ± σ):   47.994 ms ±  1.755 ms  ┊ GC (mean ± σ):  4.26% ± 2.97%

    ▂ ▆▅█▂ ▃ ▃                                                 
  █▇█▇████▇████▇▄▅▅▅▄█▁▅▁▇▅▄▅▁▄▁▁▄▁▄▁▁▁▁▁▅▄▁▁▄▄▄▁▄▁▄▁▁▁▁▁▁▁▁▄ ▄
  46.1 ms         Histogram: frequency by time        53.9 ms <

 Memory estimate: 649.12 KiB, allocs estimate: 16311.
~~~
#### Backward: 
~~~julia
julia> @benchmark compute_grad_cpu()
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 7.026 s (35.74% GC) to evaluate,
 with a memory estimate of 3.73 GiB, over 18961 allocations.

julia> @benchmark compute_grad_gpu()
BenchmarkTools.Trial: 53 samples with 1 evaluation per sample.
 Range (min … max):  92.462 ms … 101.273 ms  ┊ GC (min … max): 10.76% … 0.00%
 Time  (median):     94.347 ms               ┊ GC (median):     6.28%
 Time  (mean ± σ):   94.747 ms ±   1.799 ms  ┊ GC (mean ± σ):   5.81% ± 4.41%

       █       ▂  ▂    ▂  ▂▂         ▂                          
  ▅▅█▅▅██▅▅▁▅▅██████▁▅▁█▁███▁▁▁▁▁▅▁▁▁█▁▅▁▁▁▅▁▁▁▁▁▅▁▁▅▁▁▁▁▁▁▁▁▅ ▁
  92.5 ms         Histogram: frequency by time         99.5 ms <

 Memory estimate: 44.64 MiB, allocs estimate: 62559.
~~~

#### Data size ~ (3, 512, 12, 32)
##### Forward: 
~~~julia
julia> @benchmark model_elbo($model_cpu, $x, $x_mask)
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 21.947 s (17.56% GC) to evaluate,
 with a memory estimate of 14.25 GiB, over 3663 allocations.

julia> @benchmark model_elbo($model_gpu, $xc, $xc_mask)
BenchmarkTools.Trial: 3 samples with 1 evaluation per sample.
 Range (min … max):  1.014 s … 6.775 s  ┊ GC (min … max): 1.19% … 0.17%
 Time  (median):     1.061 s            ┊ GC (median):    1.14%
 Time  (mean ± σ):   2.950 s ± 3.313 s  ┊ GC (mean ± σ):  0.41% ± 0.59%

  █                                                     ▁  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  1.01 s        Histogram: frequency by time       6.78 s <

 Memory estimate: 848.55 KiB, allocs estimate: 20253.
~~~
#### Backward: 
~~~julia
julia> @benchmark compute_grad_cpu()
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 22.962 s (14.62% GC) to evaluate,
 with a memory estimate of 14.90 GiB, over 20957 allocations.

julia> @benchmark compute_grad_gpu()
BenchmarkTools.Trial: 3 samples with 1 evaluation per sample.
 Range (min … max):  2.431 s …   2.589 s  ┊ GC (min … max): 0.90% … 0.71%
 Time  (median):     2.478 s              ┊ GC (median):    0.88%
 Time  (mean ± σ):   2.499 s ± 80.989 ms  ┊ GC (mean ± σ):  0.83% ± 0.11%

  █               █                                       █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  2.43 s         Histogram: frequency by time        2.59 s <

 Memory estimate: 169.76 MiB, allocs estimate: 68330.
~~~


#### Data size ~ (3, 256, 12, 48)
##### Forward: 
~~~julia
julia> @benchmark model_elbo($model_cpu, $x, $x_mask)
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 8.738 s (24.55% GC) to evaluate,
 with a memory estimate of 5.63 GiB, over 5066 allocations.

julia> @benchmark model_elbo($model_gpu, $xc, $xc_mask)
BenchmarkTools.Trial: 3 samples with 1 evaluation per sample.
 Range (min … max):  1.754 s …    2.142 s  ┊ GC (min … max): 0.00% … 0.29%
 Time  (median):     2.006 s               ┊ GC (median):    0.29%
 Time  (mean ± σ):   1.967 s ± 196.907 ms  ┊ GC (mean ± σ):  0.20% ± 0.17%

  █                                    █                   █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  1.75 s         Histogram: frequency by time         2.14 s <

 Memory estimate: 1000.68 KiB, allocs estimate: 22334.
~~~
#### Backward: 
~~~julia
julia> @benchmark compute_grad_cpu()
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 9.955 s (29.25% GC) to evaluate,
 with a memory estimate of 6.13 GiB, over 22476 allocations.

julia> @benchmark compute_grad_gpu()
BenchmarkTools.Trial: 3 samples with 1 evaluation per sample.
 Range (min … max):  805.053 ms …    2.573 s  ┊ GC (min … max): 1.14% … 0.53%
 Time  (median):        2.219 s               ┊ GC (median):    0.41%
 Time  (mean ± σ):      1.866 s ± 935.369 ms  ┊ GC (mean ± σ):  0.41% ± 0.57%

  █                                               █           █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁█ ▁
  805 ms           Histogram: frequency by time          2.57 s <

 Memory estimate: 128.29 MiB, allocs estimate: 70678.
~~~


#### Data size ~ (3, 256, 12, 64)
##### Forward: 
~~~julia
julia> @benchmark model_elbo($model_cpu, $x, $x_mask)
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 11.519 s (20.55% GC) to evaluate,
 with a memory estimate of 7.51 GiB, over 6409 allocations.

julia> @benchmark model_elbo($model_gpu, $xc, $xc_mask)
BenchmarkTools.Trial: 4 samples with 1 evaluation per sample.
 Range (min … max):  781.209 ms … 2.755 s  ┊ GC (min … max): 0.00% … 0.12%
 Time  (median):        1.828 s            ┊ GC (median):    0.21%
 Time  (mean ± σ):      1.798 s ± 1.078 s  ┊ GC (mean ± σ):  0.18% ± 0.25%

  █    █                                                  ██  
  █▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁██ ▁
  781 ms         Histogram: frequency by time         2.75 s <

 Memory estimate: 1.08 MiB, allocs estimate: 23705.
~~~
#### Backward: 
~~~julia
julia> @benchmark compute_grad_cpu()
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 12.502 s (25.70% GC) to evaluate,
 with a memory estimate of 8.17 GiB, over 23824 allocations.

julia> @benchmark compute_grad_gpu()
BenchmarkTools.Trial: 6 samples with 1 evaluation per sample.
 Range (min … max):  746.331 ms …    1.014 s  ┊ GC (min … max): 1.27% … 0.00%
 Time  (median):     800.587 ms               ┊ GC (median):    1.12%
 Time  (mean ± σ):   848.053 ms ± 117.266 ms  ┊ GC (mean ± σ):  1.03% ± 0.96%

  █           █                                       ▁       ▁  
  █▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁█ ▁
  746 ms           Histogram: frequency by time          1.01 s <

 Memory estimate: 170.08 MiB, allocs estimate: 73532.
~~~


#### Data size ~ (3, 256, 12, 1) -> starting from empty gpu
##### Forward: 
~~~julia
julia> @benchmark model_elbo($model_cpu, $x, $x_mask)
BenchmarkTools.Trial: 23 samples with 1 evaluation per sample.
 Range (min … max):  135.245 ms … 540.885 ms  ┊ GC (min … max):  0.00% … 74.95%
 Time  (median):     149.658 ms               ┊ GC (median):     6.81%
 Time  (mean ± σ):   219.562 ms ± 142.841 ms  ┊ GC (mean ± σ):  37.08% ± 26.43%

  █▆▃                                                            
  ███▁▁▁▁▁▄▇▁▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▄▁▁▄▄ ▁
  135 ms           Histogram: frequency by time          541 ms <

 Memory estimate: 120.17 MiB, allocs estimate: 1024.

julia> @benchmark model_elbo($model_gpu, $xc, $xc_mask)
BenchmarkTools.Trial: 471 samples with 1 evaluation per sample.
 Range (min … max):   9.168 ms … 101.544 ms  ┊ GC (min … max): 0.00% … 71.80%
 Time  (median):      9.675 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   10.764 ms ±   6.034 ms  ┊ GC (mean ± σ):  3.97% ±  5.56%

  ██▅▂▁                                                         
  █████▅▅▁▅▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▄▁▅▅▆ ▆
  9.17 ms       Histogram: log(frequency) by time        36 ms <

 Memory estimate: 579.53 KiB, allocs estimate: 14713.
~~~
#### Backward: 
~~~julia
julia> @benchmark compute_grad_cpu()
BenchmarkTools.Trial: 21 samples with 1 evaluation per sample.
 Range (min … max):  147.418 ms … 705.834 ms  ┊ GC (min … max):  0.00% … 79.11%
 Time  (median):     164.424 ms               ┊ GC (median):     6.83%
 Time  (mean ± σ):   238.749 ms ± 178.563 ms  ┊ GC (mean ± σ):  36.92% ± 26.71%

  ▆█                                                             
  ███▁▁▁▁▁▁▄▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▁▁▁▄▁▁▁▄ ▁
  147 ms           Histogram: frequency by time          706 ms <

 Memory estimate: 132.58 MiB, allocs estimate: 18160.

julia> @benchmark compute_grad_gpu() 
BenchmarkTools.Trial: 125 samples with 1 evaluation per sample.
 Range (min … max):  32.745 ms … 129.640 ms  ┊ GC (min … max): 0.00% … 26.01%
 Time  (median):     37.435 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   40.068 ms ±  14.068 ms  ┊ GC (mean ± σ):  1.97% ±  4.01%

  ▇ ██                                                          
  █▃██▅▆▆▂▄▂▃▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▂ ▂
  32.7 ms         Histogram: frequency by time          126 ms <

 Memory estimate: 5.54 MiB, allocs estimate: 60235.
~~~


#### Data size ~ (3, 256, 12, 2) -> starting from empty gpu
##### Forward: 
~~~julia
julia> @benchmark model_elbo($model_cpu, $x, $x_mask)
BenchmarkTools.Trial: 13 samples with 1 evaluation per sample.
 Range (min … max):  266.194 ms … 821.895 ms  ┊ GC (min … max):  0.00% … 65.39%
 Time  (median):     296.854 ms               ┊ GC (median):     8.22%
 Time  (mean ± σ):   387.134 ms ± 189.438 ms  ┊ GC (mean ± σ):  29.57% ± 22.56%

    ██      ▃                                                    
  ▇▇██▁▁▁▁▁▁█▁▇▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▇▁▁▇ ▁
  266 ms           Histogram: frequency by time          822 ms <

 Memory estimate: 240.31 MiB, allocs estimate: 1103.

julia> @benchmark model_elbo($model_gpu, $xc, $xc_mask)
BenchmarkTools.Trial: 451 samples with 1 evaluation per sample.
 Range (min … max):   9.716 ms … 28.175 ms  ┊ GC (min … max): 0.00% … 34.93%
 Time  (median):     10.283 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   11.107 ms ±  3.241 ms  ┊ GC (mean ± σ):  3.71% ±  7.01%

  ▇██▅▄▁                                                       
  ██████▇█▅▅▁▁▄▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▄▅▅▄▁▅▅▇▄▁▄▁▁▅ ▇
  9.72 ms      Histogram: log(frequency) by time      25.8 ms <

 Memory estimate: 586.28 KiB, allocs estimate: 14823.
~~~
#### Backward: 
~~~julia
julia> @benchmark compute_grad_cpu()
BenchmarkTools.Trial: 13 samples with 1 evaluation per sample.
 Range (min … max):  282.601 ms … 901.775 ms  ┊ GC (min … max):  0.00% … 68.67%
 Time  (median):     300.741 ms               ┊ GC (median):     6.18%
 Time  (mean ± σ):   388.124 ms ± 209.610 ms  ┊ GC (mean ± σ):  26.55% ± 24.41%

  ▅█                                                             
  ██▅▁▁▁▁▁▁▁▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅▁▁▁▁▁▁▁▁▅ ▁
  283 ms           Histogram: frequency by time          902 ms <

 Memory estimate: 263.31 MiB, allocs estimate: 18234.

julia> @benchmark compute_grad_gpu() 
BenchmarkTools.Trial: 118 samples with 1 evaluation per sample.
 Range (min … max):  36.077 ms … 100.359 ms  ┊ GC (min … max): 0.00% … 20.16%
 Time  (median):     39.467 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   42.680 ms ±  11.698 ms  ┊ GC (mean ± σ):  2.76% ±  5.43%

  ▃█ ▇▇                                                         
  ██▇███▆▃▆▅▃▃▄▁▃▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▁▃▁▃▁▃ ▃
  36.1 ms         Histogram: frequency by time         87.1 ms <

 Memory estimate: 8.15 MiB, allocs estimate: 60437.
~~~


#### Data size ~ (3, 256, 12, 4) -> starting from empty gpu
##### Forward: 
~~~julia
julia> @benchmark model_elbo($model_cpu, $x, $x_mask)
BenchmarkTools.Trial: 6 samples with 1 evaluation per sample.
 Range (min … max):  561.901 ms …    1.220 s  ┊ GC (min … max):  3.61% … 55.97%
 Time  (median):     886.622 ms               ┊ GC (median):    37.13%
 Time  (mean ± σ):   886.017 ms ± 316.415 ms  ┊ GC (mean ± σ):  37.71% ± 25.19%

  ██       █                                        █      █  █  
  ██▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁█▁▁█ ▁
  562 ms           Histogram: frequency by time          1.22 s <

 Memory estimate: 480.58 MiB, allocs estimate: 1327.

julia> @benchmark model_elbo($model_gpu, $xc, $xc_mask)
BenchmarkTools.Trial: 363 samples with 1 evaluation per sample.
 Range (min … max):  12.100 ms … 23.145 ms  ┊ GC (min … max): 0.00% … 28.84%
 Time  (median):     13.156 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   13.775 ms ±  1.867 ms  ┊ GC (mean ± σ):  4.52% ±  9.56%

    ▂▄▇▆█▇█▂ ▂                                                 
  ▄▇████████▇█▆▆▂▄▃▁▂▃▂▃▃▂▁▂▂▂▁▁▁▂▂▁▁▂▄▁▄▄▃▃▃▂▃▃▂▂▁▃▁▃▂▂▂▁▁▂▂ ▃
  12.1 ms         Histogram: frequency by time        20.2 ms <

 Memory estimate: 604.46 KiB, allocs estimate: 15290.
~~~
#### Backward: 
~~~julia
julia> @benchmark compute_grad_cpu()
BenchmarkTools.Trial: 6 samples with 1 evaluation per sample.
 Range (min … max):  657.365 ms …    1.211 s  ┊ GC (min … max):  4.87% … 53.01%
 Time  (median):     931.049 ms               ┊ GC (median):    36.21%
 Time  (mean ± σ):   930.447 ms ± 247.197 ms  ┊ GC (mean ± σ):  36.11% ± 20.10%

  █     █   █                                      █  █       █  
  █▁▁▁▁▁█▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁█▁▁▁▁▁▁▁█ ▁
  657 ms           Histogram: frequency by time          1.21 s <

 Memory estimate: 524.75 MiB, allocs estimate: 18567.

julia> @benchmark compute_grad_gpu() 
BenchmarkTools.Trial: 105 samples with 1 evaluation per sample.
 Range (min … max):  41.520 ms … 85.811 ms  ┊ GC (min … max): 0.00% … 28.95%
 Time  (median):     44.968 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   47.988 ms ±  8.782 ms  ┊ GC (mean ± σ):  3.44% ±  6.60%

  ▅█ ▂▁▇▄ ▁                                                    
  ███████▆█▃▄▅▁▃▃▁▁▁▄▄▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▅▃▁▃▁▁▁▁▄▅▁▁▁▁▁▁▁▁▁▁▁▁▃ ▃
  41.5 ms         Histogram: frequency by time          79 ms <

 Memory estimate: 13.38 MiB, allocs estimate: 61319.
~~~


#### Data size ~ (3, 256, 12, 8) -> starting from empty gpu
##### Forward: 
~~~julia
julia> @benchmark model_elbo($model_cpu, $x, $x_mask)
BenchmarkTools.Trial: 4 samples with 1 evaluation per sample.
 Range (min … max):  1.248 s …    1.717 s  ┊ GC (min … max): 10.32% … 36.47%
 Time  (median):     1.687 s               ┊ GC (median):    35.57%
 Time  (mean ± σ):   1.584 s ± 225.187 ms  ┊ GC (mean ± σ):  30.84% ± 12.79%

  █                                                   █  █ █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁█▁█ ▁
  1.25 s         Histogram: frequency by time         1.72 s <

 Memory estimate: 961.11 MiB, allocs estimate: 1671.

julia> @benchmark model_elbo($model_gpu, $xc, $xc_mask)
BenchmarkTools.Trial: 282 samples with 1 evaluation per sample.
 Range (min … max):  17.174 ms …  20.438 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     17.552 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   17.730 ms ± 501.870 μs  ┊ GC (mean ± σ):  3.50% ± 7.16%

        ▄▃█▂▂▁▁                                                 
  ▅▆▃██████████▅▅▆▅▃▄▄▄▃▃▃▁▃▄▂▂▂▄▃▂▂▃▃▃▃▃▂▂▁▂▃▂▁▂▁▃▁▃▂▁▁▁▁▂▁▁▂ ▃
  17.2 ms         Histogram: frequency by time         19.4 ms <

 Memory estimate: 640.20 KiB, allocs estimate: 15868.
~~~
#### Backward: 
~~~julia
julia> @benchmark compute_grad_cpu()
BenchmarkTools.Trial: 3 samples with 1 evaluation per sample.
 Range (min … max):  1.776 s …   1.834 s  ┊ GC (min … max): 33.23% … 35.43%
 Time  (median):     1.782 s              ┊ GC (median):    35.16%
 Time  (mean ± σ):   1.797 s ± 31.557 ms  ┊ GC (mean ± σ):  34.61% ±  1.20%

  █     █                                                 █  
  █▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  1.78 s         Histogram: frequency by time        1.83 s <

 Memory estimate: 1.02 GiB, allocs estimate: 18932.

julia> @benchmark compute_grad_gpu() 
BenchmarkTools.Trial: 79 samples with 1 evaluation per sample.
 Range (min … max):  57.497 ms … 99.350 ms  ┊ GC (min … max): 0.00% … 15.88%
 Time  (median):     61.312 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   63.844 ms ±  7.294 ms  ┊ GC (mean ± σ):  3.51% ±  6.35%

  ▃▂█▂ ▅    ▂                                                  
  ████▇██▅▁██▄▄▄▄▅▇█▁▄▄▄▇▁▄▄▅▁▁▁▁▁▁▁▄▁▁▁▄▁▁▁▁▄▁▁▁▁▁▁▁▁▁▁▄▁▁▁▄ ▁
  57.5 ms         Histogram: frequency by time        87.7 ms <

 Memory estimate: 23.82 MiB, allocs estimate: 61937.
~~~


#### Data size ~ (3, 256, 12, 16) -> starting from empty gpu
##### Forward: 
~~~julia
julia> @benchmark model_elbo($model_cpu, $x, $x_mask)
BenchmarkTools.Trial: 2 samples with 1 evaluation per sample.
 Range (min … max):  3.501 s …   3.545 s  ┊ GC (min … max): 37.70% … 37.98%
 Time  (median):     3.523 s              ┊ GC (median):    37.84%
 Time  (mean ± σ):   3.523 s ± 30.881 ms  ┊ GC (mean ± σ):  37.84% ±  0.19%

  █                                                       █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  3.5 s          Histogram: frequency by time        3.54 s <

 Memory estimate: 1.88 GiB, allocs estimate: 2362.

julia> @benchmark model_elbo($model_gpu, $xc, $xc_mask)
BenchmarkTools.Trial: 172 samples with 1 evaluation per sample.
 Range (min … max):  27.408 ms … 38.840 ms  ┊ GC (min … max): 0.00% … 11.79%
 Time  (median):     28.586 ms              ┊ GC (median):    0.00%
 Time  (mean ± σ):   29.219 ms ±  1.779 ms  ┊ GC (mean ± σ):  4.26% ±  6.30%

    █▂▃▂▄ ▂                                                    
  ▄▆█████▆█▄▃▆▅▆▃▆▆▅▄▃▃▆▁▃▁▄▄▄▃▄▃▁▁▁▃▃▃▁▁▁▃▃▃▁▁▁▁▃▁▁▁▁▃▁▁▁▁▁▃ ▃
  27.4 ms         Histogram: frequency by time        35.6 ms <

 Memory estimate: 706.15 KiB, allocs estimate: 17024.
~~~
#### Backward: 
~~~julia
julia> @benchmark compute_grad_cpu()
BenchmarkTools.Trial: 2 samples with 1 evaluation per sample.
 Range (min … max):  3.145 s …    3.567 s  ┊ GC (min … max): 26.82% … 36.02%
 Time  (median):     3.356 s               ┊ GC (median):    31.71%
 Time  (mean ± σ):   3.356 s ± 298.894 ms  ┊ GC (mean ± σ):  31.71% ±  6.51%

  █                                                        █  
  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  3.14 s         Histogram: frequency by time         3.57 s <

 Memory estimate: 2.04 GiB, allocs estimate: 19637.

julia> @benchmark compute_grad_gpu() 
BenchmarkTools.Trial: 64 samples with 1 evaluation per sample.
 Range (min … max):  74.939 ms … 102.054 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     77.144 ms               ┊ GC (median):    4.54%
 Time  (mean ± σ):   79.234 ms ±   5.316 ms  ┊ GC (mean ± σ):  5.01% ± 5.14%

     █ ▅    ▂                                                   
  ▇▇████▇▇▄▄█▄▄▄▄▄▁▇▅▁▄▄▄▁▁▇▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▄▁▄▁▁▄ ▁
  74.9 ms         Histogram: frequency by time           95 ms <

 Memory estimate: 44.70 MiB, allocs estimate: 62838.
~~~

#### Data size ~ (3, 256, 12, 32) -> starting from empty gpu
##### Forward: 
~~~julia
julia> @benchmark model_elbo($model_cpu, $x, $x_mask)
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 6.356 s (31.35% GC) to evaluate,
 with a memory estimate of 3.75 GiB, over 3700 allocations.

julia> @benchmark model_elbo($model_gpu, $xc, $xc_mask)
BenchmarkTools.Trial: 102 samples with 1 evaluation per sample.
 Range (min … max):  47.635 ms … 54.018 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     48.837 ms              ┊ GC (median):    4.17%
 Time  (mean ± σ):   49.066 ms ±  1.217 ms  ┊ GC (mean ± σ):  3.92% ± 2.65%

    ▂▆ ▂ █ ▄ ▆▂█▆▄   ▄                                         
  ██████████▆██████▆▆██▄▁▄▁▄▄▆▄▆▁▁▁▁▁▁▁▁▁▄▁▁▄▁▁▄▁▁▁▁▄▁▁▁▁▄▁▁▄ ▄
  47.6 ms         Histogram: frequency by time        53.4 ms <

 Memory estimate: 843.61 KiB, allocs estimate: 19201.
~~~
#### Backward: 
~~~julia
julia> @benchmark compute_grad_cpu()
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 6.426 s (29.30% GC) to evaluate,
 with a memory estimate of 4.09 GiB, over 20977 allocations.

julia> @benchmark compute_grad_gpu() 
BenchmarkTools.Trial: 40 samples with 1 evaluation per sample.
 Range (min … max):  121.265 ms … 142.957 ms  ┊ GC (min … max): 5.63% … 5.23%
 Time  (median):     125.479 ms               ┊ GC (median):    5.78%
 Time  (mean ± σ):   127.409 ms ±   5.687 ms  ┊ GC (mean ± σ):  6.19% ± 1.63%

  ▁▁  ▄█ ▁▄  ▁▁ ▁         ▁     ▁                                
  ██▁▁██▆██▆▁██▆█▁▆▆▁▁▁▁▆▆█▁▆▁▁▁█▆▁▁▁▁▁▁▁▁▆▁▁▁▁▁▁▆▁▁▁▆▁▁▁▁▁▁▆▁▆ ▁
  121 ms           Histogram: frequency by time          143 ms <

 Memory estimate: 86.49 MiB, allocs estimate: 66391.
~~~

#### Data size ~ (3, 256, 12, 48) -> starting from empty gpu
##### Forward: 
~~~julia
julia> @benchmark model_elbo($model_cpu, $x, $x_mask)
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 8.806 s (25.22% GC) to evaluate,
 with a memory estimate of 5.63 GiB, over 5124 allocations.

julia> @benchmark model_elbo($model_gpu, $xc, $xc_mask)
BenchmarkTools.Trial: 71 samples with 1 evaluation per sample.
 Range (min … max):  69.787 ms …  73.717 ms  ┊ GC (min … max): 1.75% … 3.11%
 Time  (median):     71.001 ms               ┊ GC (median):    2.07%
 Time  (mean ± σ):   71.171 ms ± 777.181 μs  ┊ GC (mean ± σ):  2.27% ± 0.60%

       ▄▁      ▄▁▄ █▁█▁▁▄ ▄ ▁ ▁▁  ▄▁ ▄     ▁  ▄   ▁             
  ▆▁▁▆▆██▁▁▆▆▁▆███▆██████▁█▆█▆██▁▆██▆█▆▁▁▁▁█▁▁█▁▆▆█▆▁▁▁▁▁▁▁▁▁▆ ▁
  69.8 ms         Histogram: frequency by time         73.1 ms <

 Memory estimate: 1005.49 KiB, allocs estimate: 21818.
~~~
#### Backward: 
~~~julia
julia> @benchmark compute_grad_cpu()
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 9.241 s (25.59% GC) to evaluate,
 with a memory estimate of 6.13 GiB, over 22546 allocations.

julia> @benchmark compute_grad_gpu()
BenchmarkTools.Trial: 20 samples with 1 evaluation per sample.
 Range (min … max):  238.351 ms … 270.128 ms  ┊ GC (min … max): 5.15% … 5.30%
 Time  (median):     248.655 ms               ┊ GC (median):    5.09%
 Time  (mean ± σ):   251.680 ms ±   9.919 ms  ┊ GC (mean ± σ):  5.43% ± 0.75%

  ▁   ▁▁  ▁▁  ▁█ █       ▁ ▁ ▁   ▁       ▁   ▁      ▁▁       ▁▁  
  █▁▁▁██▁▁██▁▁██▁█▁▁▁▁▁▁▁█▁█▁█▁▁▁█▁▁▁▁▁▁▁█▁▁▁█▁▁▁▁▁▁██▁▁▁▁▁▁▁██ ▁
  238 ms           Histogram: frequency by time          270 ms <

 Memory estimate: 128.30 MiB, allocs estimate: 70286.

julia> @benchmark compute_grad_gpu()
BenchmarkTools.Trial: 18 samples with 1 evaluation per sample.
 Range (min … max):  171.233 ms …    1.183 s  ┊ GC (min … max): 7.12% … 3.54%
 Time  (median):     179.694 ms               ┊ GC (median):    7.52%
 Time  (mean ± σ):   293.134 ms ± 259.763 ms  ┊ GC (mean ± σ):  6.28% ± 1.58%

  █                                                              
  █▃▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▃▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃ ▁
  171 ms           Histogram: frequency by time          1.18 s <
~~~


#### Data size ~ (3, 256, 12, 64) -> starting from empty gpu
##### Forward: 
~~~julia
julia> @benchmark model_elbo($model_cpu, $x, $x_mask)
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 11.563 s (24.80% GC) to evaluate,
 with a memory estimate of 7.51 GiB, over 6379 allocations.

julia> @benchmark model_elbo($model_gpu, $xc, $xc_mask)
BenchmarkTools.Trial: 53 samples with 1 evaluation per sample.
 Range (min … max):  90.803 ms … 100.796 ms  ┊ GC (min … max): 5.21% … 8.60%
 Time  (median):     93.858 ms               ┊ GC (median):    3.98%
 Time  (mean ± σ):   94.851 ms ±   2.717 ms  ┊ GC (mean ± σ):  3.94% ± 1.83%

      ▃▃   ▃▃▃▃ ██▃   ▃             █   ▃    ▃ ▃      ▃         
  ▇▁▇▁██▁▇▇████▇███▇▇▇█▇▁▁▁▁▇▁▁▇▁▇▁▁█▁▁▇█▇▇▁▁█▇█▁▁▇▁▁▇█▁▁▁▁▁▁▇ ▁
  90.8 ms         Histogram: frequency by time          101 ms <

 Memory estimate: 1.07 MiB, allocs estimate: 23649.
~~~
#### Backward: 
~~~julia
julia> @benchmark compute_grad_cpu()
BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
 Single result which took 12.140 s (25.06% GC) to evaluate,
 with a memory estimate of 8.17 GiB, over 23787 allocations.

julia> @benchmark compute_grad_gpu()
BenchmarkTools.Trial: 7 samples with 1 evaluation per sample.
 Range (min … max):  596.866 ms …    1.623 s  ┊ GC (min … max): 3.11% … 1.14%
 Time  (median):     741.045 ms               ┊ GC (median):    2.66%
 Time  (mean ± σ):   894.640 ms ± 366.015 ms  ┊ GC (mean ± σ):  2.34% ± 0.96%

  █   █  ███                      █                           █  
  █▁▁▁█▁▁███▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
  597 ms           Histogram: frequency by time          1.62 s <

 Memory estimate: 170.08 MiB, allocs estimate: 74460.
~~~


#### Data size ~ 
##### Forward: 
~~~julia

~~~
#### Backward: 
~~~julia

~~~



# Discrete-bags Generation on Categorical data 

## NaiveSetModel (v1 baseline)
- it works quite well, exact matching rate is good and mean_element_accuracy is close to 100%
- mean_element_accuracy is very high, but question is if the cardinality prediction would not tank this. 
- here we use ground truth cardinality, so model only focus on reconstruction. When cardinality prediciton is added results will most likely change. 
- Options for upgrade: 
  - [ ] add cardinality predictor at the end 
  - [ ] add and train cardinality predictor but during training use ground truth cardinalities

~~~julia
Epoch 100 | train: (ℒ = 0.4553554f0, ℒ_rec = 0.32021052f0, ℒₖₗ = 13.514487f0) | valid: (ℒᵥ = 0.4697719f0, ℒᵥ_rec = 0.3347653f0, ℒᵥₖₗ = 13.500663f0)
-- reconstruction check: [1, 7, 1, 2] --
  input: [1, 7, 1, 2]  (sorted: [1, 1, 2, 7])
    sample 1: [1, 2, 7, 2]  ✗
    sample 2: [1, 2, 7, 1]  ✓
    sample 3: [1, 1, 2, 7]  ✓
    sample 4: [1, 1, 7, 2]  ✓
    sample 5: [7, 2, 1, 1]  ✓
  exact_match_rate=0.85  mean_element_accuracy=0.9625
-- reconstruction check: [1, 2, 3, 4, 5, 6, 7, 8] --
  input: [1, 2, 3, 4, 5, 6, 7, 8]  (sorted: [1, 2, 3, 4, 5, 6, 7, 8])
    sample 1: [4, 8, 6, 2, 5, 7, 8, 3]  ✗
    sample 2: [7, 1, 5, 3, 2, 8, 6, 4]  ✓
    sample 3: [8, 4, 3, 6, 5, 4, 2, 1]  ✗
    sample 4: [3, 6, 2, 4, 5, 8, 7, 1]  ✓
    sample 5: [4, 7, 3, 2, 8, 6, 5, 1]  ✓
  exact_match_rate=0.55  mean_element_accuracy=0.94375
-- reconstruction check: [9, 9, 5, 2, 9, 3, 6, 5] --
  input: [9, 9, 5, 2, 9, 3, 6, 5]  (sorted: [2, 3, 5, 5, 6, 9, 9, 9])
    sample 1: [9, 3, 9, 6, 5, 9, 2, 5]  ✓
    sample 2: [5, 3, 6, 5, 2, 9, 9, 9]  ✓
    sample 3: [2, 5, 3, 9, 6, 9, 9, 5]  ✓
    sample 4: [5, 9, 3, 9, 5, 9, 5, 5]  ✗
    sample 5: [5, 2, 2, 9, 6, 9, 5, 9]  ✗
  exact_match_rate=0.5  mean_element_accuracy=0.91875s
~~~



## DSQVAE Categorical 
- λ_exist = 2f0 is important. but still the biggest problem is cardinality.
- mean_element_accuracy is pretty much 100%, meaning i get all elements reconstructed correctly including duplicities, BUT mean_predicted_cardinality is not that great which is reason that exact_match_rate is very low (close to 0 most of the times)

~~~julia
Epoch 100 | train: (ℒ = 6.4969907f0, ℒ_rec = 0.08386141f0, ℒ_exist = 3.018949f0, ℒ_kld = 37.523148f0, β = 0.01f0) | valid: (ℒᵥ = 6.2949567f0, ℒᵥ_rec = 0.07423262f0, ℒᵥ_exist = 2.9235165f0, ℒᵥ_kld = 37.369175f0)
-- reconstruction check: [1, 7, 1, 2] --
  input: [1, 7, 1, 2]  (sorted: [1, 1, 2, 7], n=4)
    sample 1: [2, 1, 1, 7]  ✓
    sample 2: [2, 1, 1, 7, 1]  ✗
    sample 3: [2, 1, 1, 7]  ✓
    sample 4: [2, 1, 1, 7, 1]  ✗
    sample 5: [2, 1, 1, 7, 1]  ✗
  exact_match_rate=0.35  mean_element_accuracy=1.0  mean_predicted_cardinality=4.65
-- reconstruction check: [1, 2, 3, 4, 5, 6, 7, 8] --
  input: [1, 2, 3, 4, 5, 6, 7, 8]  (sorted: [1, 2, 3, 4, 5, 6, 7, 8], n=8)
    sample 1: [6, 8, 2, 8, 2, 1, 3, 7, 1, 5, 4, 3]  ✗
    sample 2: [6, 2, 8, 2, 1, 4, 7, 1, 5, 4, 3]  ✗
    sample 3: [6, 2, 8, 1, 4, 7, 1, 5, 4, 3]  ✗
    sample 4: [6, 2, 8, 1, 3, 7, 1, 5, 4, 3]  ✗
    sample 5: [6, 8, 2, 8, 2, 1, 4, 7, 5, 4, 3]  ✗
  exact_match_rate=0.0  mean_element_accuracy=1.0  mean_predicted_cardinality=11.15
-- reconstruction check: [9, 9, 5, 2, 9, 3, 6, 5] --
  input: [9, 9, 5, 2, 9, 3, 6, 5]  (sorted: [2, 3, 5, 5, 6, 9, 9, 9], n=8)
    sample 1: [6, 9, 2, 9, 9, 5, 5, 3]  ✓
    sample 2: [6, 9, 2, 9, 9, 5, 5, 3]  ✓
    sample 3: [6, 2, 9, 9, 5, 5, 3]  ✗
    sample 4: [6, 2, 9, 9, 5, 5, 3]  ✗
    sample 5: [6, 2, 9, 9, 5, 5, 3]  ✗
  exact_match_rate=0.45  mean_element_accuracy=0.93125  mean_predicted_cardinality=7.45
~~~

