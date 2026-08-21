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


### Speed testing CPU vs GPU on diferent sizes
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

