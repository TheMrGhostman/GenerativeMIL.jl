# Notes on Multi Object Generation

## MultiObject Generaiton
# first_test.jl

~~~julia
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

 
