# Notes and TODOs
## To Remember : 
- path to setting.json on rci : /home/zorekmat/.vscode-server/data/Machine/settings.json


## Todos 21.8.2026
- [ ] Improve memory and speed on GPUs of Hungarian matching and pairwise chamfer distance. 
    - we can work only with cartesian indexes and intermediate PDM to compute hungarian matching, 
    - then when matching is found we can only compute with indexes we need for matching. we can save a lot of memory for gradients approximately HALF of it. 
    - Might be useful even for HTD distance


## Todos 20.8.2026
- [ ] sliced wasserstain impelementation
  - [ ] implement
  - [ ] sanity check

- [x] port DeepSlotQuerryVAE into GenerativeMIL
- [x] implement gpu-ready cluster pairwise Chamfer Distance
  - [x] do sanity check if reshapeing is the same as if i run it in nested for loops
- [ ] download MNIST PC data into MainGPU PC
- [ ] train HSVAE (specialized for gaussian clock)
  - [x] implement hungarian matching with gpu pairwise evaluation for gaussian clock -> vit CPCD
    - [x] implement hungarian part
    - [x] sanity checks
  - [x] implement forward and backward pass for HSVAE
    - [x] it can be done via masking
  - [ ] test shallow setvae if it is better then poolmodel? 
    - attention mask would allowed me to compute everything at once but PoolModel is also option. 
    - but when i use attention inside i can put it also on the outside. 



## Todos 1.8.2026
#### saving update 
- [x] update saving functions. 
  - [x] save config as JSON or YAML into results folder, so it is easier to load it later and run evaluation script on it. 
    - It will also make results more reproducible, because we will know exactly what config was used for training.
  - [x] save model as Flux.state and .jld2, so it is easier to load it later. This is also reason for saving config as JSON or YAML as it will be easier to reconstruct model from config and load state from .jld2 file.
- [x] think about .BSON -> this would be really good but i will not use it now. 
#### Visualization and small experiment
  - [x] make umap of arbitrary PoolModel on MNIST on E(x_i) and D(c) ... i.e. after pool encoder and after decoder (generator) before instance decoder
    - [x] try to color it by classes and see if it is separated in latent space.
    - [x] It my help understand how PoolModel works and if it is able to separate classes in latent space.
      - results are really nice, i think Neural Statistician might be nice as well. to see if "c" is actually used for anything
  - [x] Neural Statistician
    - [x] cd -> does not work. some classes are separete but mostly they overlay massively. that is not how latent space should look like.
    - [x] l2 -> obviously much worse then NS-cd and even more then PoolModel.
  - [x] SetVAE
    - [x] shallow 
    - [?] deep (think about this more)
#### retrain PoolModel on core5
- [x] poolmodel core5 

## Todos 30.7.2026
- [x] Evaluation (reconstruction) 
  - [x] mnist
    - [x] neural statistician
    - [x] poolmodel
    - [x] setvae
  - [x] airplane
    - [x] neural statistician
    - [x] poolmodel
    - [x] setvae
  - [x] core5
    - [x] nerual statistician
    - [x] poolmodel
    - [x] setvae

## Todos 24.7.2026
- [x] MNIST - PM - Sinkhorn -- can not be loaded -> needs retraining :/
- [x] most likely all neural statisticians too
  - [x] mnist
    - [x] l2
    - [x] cd
    - [x] dcd
    - [x] sh
    - [x] mmd
  - [] airplane
    - [x] l2
    - [x] cd
    - [x] dcd
    - [x] sh
    - [x] mmd
  - [x] core5
    - [x] cd
    - [x] l2
    - [ ] dcd (?)

## Todos 21.7.2026
- [x] try to plot UMAP from top (the deepest) layer for encoder, to see if classes are separated somehow in latent space
- [ ] try "guided" generation of samples 
  - [ ] if umap and separation works then i can use centroids of classes in latent space to generate samples from specific class, which may be interesting for MNIST and ShapeNetCore (core5)
    - [x] mnist
    - [ ] core5
#### reconstruction evaluation copyed from 7.7.2026
- [x] evaluation scripts
  - [x] evaluate FINAL models
  - [y] !!OPTIONAL!! evaluate on checkpoint with best reconstruciton on validation set in given run. (possible to do)
    - [x] prepare script to analyze histories and pick best checkpoint
    - [x] script can evaluate best model and also last saved checkpoint (most likely final model)


## Todos 13.7.2026
- alpha = 1000 seems to be too high, it is better to use alpha=1, at least i think so according to data ->> was like day and night !!
#### train density aware chamfer distance
  - [x] make config files
  - [x] MNIST
    - [x] poolmodel
      - [x] dcd
    - [x] neuralstatistician
      - [x] dcd
    - [x] setvae
      - [x] dcd
      - [x] cd (?)
        - [x] cyclical
        - [x] sigmoidal
        - [ ] linear (?)
  - [x] airplane
    - [x] poolmodel
      - [x] dcd
    - [x] neuralstatistician
      - [x] dcd (need to retrain 1.8.2026)
    - [ ] setvae
      - [ ] dcd
        - [x] cyclical (1-5 done, 6-10 in progress)
        - [ ] sigmoidal (?)
        - [ ] linear (?)
      - [ ] sh (?)
        - [ ] cyclical

## Todos 8.7.2026
- [x] add special version of mmd and dcd for evaluation 
  - [x] dcd
  - [x] mmd
- [x] density-aware chamfer distance
  - [x] implement
  - [x] start with only forward 
  - [x] write rrule or fast backward
  - [x] cpu ready
  - [x] make to gpu ready

## Todos 7.7.2026
- not able to load old PoolModel and NeuralStatistician (can with sinkhorn) -> retraining mnist
  - [x] MNIST
    - [x] poolmodel
      - [x] cd 
      - [x] mmd
      - [x] sh
    - [x] neuralstatistician
      - [x] cd
      - [x] mmd
      - [x] l2
  - [x] airplane
    - [x] poolmodel
      - [x] cd
      - [x] mmd (?)
      - [x] sh (?)
    - [x] neuralstatistician
      - [x] l2
      - [x] cd
      - [x] mmd (?)
      - [x] sh (?)
- [x] eval source code for renconstruction
  - [x] implement
  - [x] test on dummy case
  - [x] test on real case (Poolmodel for example)

## Todos 26.6.2026
- [ ] implement regularized sinkhorn 
  - [ ] froward
  - [ ] rrule
- !!!! not possible for now, I can not backpropagate throughout solver, I could write it myself but I think it is not worth it now. 

## Todos 22.6.2026
- [x] sinkhorn divergence
  - [x] implement it
  - [x] custom rrule
  - [x] testing on gpu
  - [x] run with poolmodel first (then if it works add to setvae and neural statistician)

- [x] cyclic sinusoidal scheduler
  - [x] implement
  - [x] add to framework
  - [x] test
  - [x] apply to core5
- [x] use elbo and reconstruction terms as evaluation measures
  - [x] mean and std per reconstruction
  - [x] mean and std within class 
  - [x] mean and std within dataset ? ?  

## Todos 8.6.2026
- [ ] Neural Statistician experiment
  - [ ] mix c and z from two different number (different classes)
    - the reconstruction should show us if information (about class or shape) is inside $z$ or $c$
    - I think $c$ will not be utilized that much or at all
    - we can take a look at loss with zdim=2 and zdim=32, we will see that one reaches much lower reconstruction loss then other, also kld_c will be close to zero. and kld_z will be high. all this points to situation in which most info is in $z$, and when I make bottleneck ($z$) tighter we have problems with reconstruction. 

## Todos 1.6.2026
- [ ] Evaluation script
  - [ ] Coverage
  - [ ] Minimum Matching Distance
  - [ ] FID
  - [ ] 1-NNA
  - [ ] final script

- [ ] ShapeNetCore Training / Experiments
  - [ ] SetVae training 
    - [x]   I. Stage : Airplane (only)
    - [x]  II. Stage : airplane, car, chair, table, sofa
    - [ ] III. Stage : + rifle, lamp, watercraft, loudspeaker, display
  - [ ] PoolModel
    - [x]   I. Stage : Airplane (only)
    - [x]  II. Stage : airplane, car, chair, table, sofa
    - [ ] III. Stage : + rifle, lamp, watercraft, loudspeaker, display
  - [ ] NeuralStatistician
    - [x]   I. Stage : Airplane (only)
    - [x]  II. Stage : airplane, car, chair, table, sofa
    - [ ] III. Stage : + rifle, lamp, watercraft, loudspeaker, display
  

- [x] MNIST point cloud Experiments
  - [x] NerualStatistician
    - [x] l2
    - [x] chamfer
    - [x] mmd


- [x] ShapeNetCore dataset
  - [x] download dataset
  - [x] try to download 15K version used in PointFlow and SetVAE
  - [x] write processing script
  - [x] write loaders
    - [x] write loader for 15K version
      - [x] make loader so it can select specific classes 
      - [ ] make loader that can pick multiple classes
        - [x] add global normalization
    - [ ] write loader for our processed version from raw data
      - [ ] make loader so it can select specific classes 
  - [x] test functions 
  - [x] add normalization to loaders
  - [x] pick intersting classes for all three stages

## Todos 15.5.2026
- [x] update training
    - [x] config generator
        - [x] setvae
        - [x] poolmodel
        - [x] neural statistician

    - [x] AdaMax to AdamW
    - [x] MMD with EMA estimator for sigma
        - I striped type for valid_step and optim_step, dispatch is on elbo_with_logging
        - Also train_model and validation_check have now loss function as union Function and MMD_EMA_loss
        - [x] code 
        - [x] incorporate to loss factory
        - [x] final testing
        - [x] add into config generator
        - [x] fix valid_step and optim_step for poolmodel!!!!
    - [x] new beta scheduler "step_linear"
        - it starts on inition value, stay on it till first milestone, then linearly ascend to max_value (final value)
        - final value is reached on second milestone and then beta is equal to final value till the end
        - [x] implement step_linear scheduler using ParameterSchedulers.Sequence
        - [x] prepare constructor so it can be built from yml config
        - [x] add it into config generator
        - [x] test it 

    - [x] add reconstruction loop for test set
        - [x] write function
        - [x] evaluate final model
        - [x] evaluate best model
        - [x] save to folder
        - [x] save basic results like final loss (maybe final log), test loss


    - [x] make new ModelNet dataset 
        - [x] download data
            - [x] testing
            - [x] rci
        - [x] pick 10 interesting classes -> ["airplane", "car", "chair", "bed", "table", "sofa", "monitor", "lamp", "plant", "tent"]
        - [x] sample points and save to file
            - [x] version with balanced classes (at least a little -> min sample size) 
        - [x] prepare loading function
        - [x] incorporate to create_dataloaders
