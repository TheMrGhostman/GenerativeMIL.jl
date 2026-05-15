# Notes and TODOs

## Todos 15.5.2026
- [ ] update training
    - [ ] config generator
        - [x] setvae
        - [ ] poolmodel
        - [ ] neural statistician

    - [x] AdaMax to AdamW
    - [x] MMD with EMA estimator for sigma
        - [x] code 
        - [x] incorporate to loss factory
        - [x] final testing
        - [x] add into config generator
    - [x] new beta scheduler "step_linear"
        - it starts on inition value, stay on it till first milestone, then linearly ascend to max_value (final value)
        - final value is reached on second milestone and then beta is equal to final value till the end
        - [x] implement step_linear scheduler using ParameterSchedulers.Sequence
        - [x] prapare constructor so it can be built from yml config
        - [x] add it into config generator
        - [x] test it 

    - [ ] add reconstruction loop for test set
        - [ ] write function
        - [ ] evaluate final model
        - [ ] evaluate best model
        - [ ] save to folder
        - [ ] save basic results like final loss (maybe final log), test loss


    - [ ] make new ModelNet dataset 
        - [ ] download data
            - [x] testing
            - [ ] rci
        - [ ] pick 10 interesting classes
        - [ ] sample points and save to file
        - [ ] prepare loading function
        - [ ] incorporate to create_dataloaders
