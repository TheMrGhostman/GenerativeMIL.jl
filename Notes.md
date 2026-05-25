# Notes and TODOs

## Todos 15.5.2026
- [ ] update training
    - [ ] config generator
        - [x] setvae
        - [ ] poolmodel
        - [ ] neural statistician

    - [x] AdaMax to AdamW
    - [x] MMD with EMA estimator for sigma
        - I striped type for valid_step and optim_step, dispatch is on elbo_with_logging
        - Also train_model and validation_check have now loss function as union Function and MMD_EMA_loss
        - [x] code 
        - [x] incorporate to loss factory
        - [x] final testing
        - [x] add into config generator
        - [ ] fix valid_step and optim_step for poolmodel!!!!
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
