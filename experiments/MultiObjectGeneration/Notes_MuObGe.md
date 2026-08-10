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