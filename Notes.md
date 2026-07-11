# Notes and TODOs
## To Remember : 
- path to setting.json on rci : /home/zorekmat/.vscode-server/data/Machine/settings.json

## Todos 8.7.2026
- [ ] add special version for mmd evaluation 
- [ ] density-aware chamfer distance
  - [ ] implement
  - [ ] start with only forward 
  - [ ] write rrule or fast backward

## Todos 7.7.2026
- not able to load old PoolModel and NeuralStatistician (can with sinkhorn) -> retraining mnist
  - [x] MNIST
    - [x] poolmodel
      - [x] cd 
      - [x] mmd
    - [x] neuralstatistician
      - [x] cd
      - [x] mmd
      - [x] l2
  - [ ] airplane
    - [ ] poolmodel
      - [x] cd
      - [x] mmd (?)
      - [ ] sh (?)
    - [ ] neuralstatistician
      - [x] l2
      - [x] cd
      - [x] mmd (?)
      - [ ] sh (?)
- [ ] evaluation scripts
  - [ ] evaluate FINAL models
  - [ ] !!OPTIONAL!! evaluate on checkpoint with best reconstruciton on validation set in given run.
    - [ ] prepare script to analyze histories and pick best checkpoint
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
