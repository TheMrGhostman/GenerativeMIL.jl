# Notes and TODOs

## Todos 15.5.2026
- [ ] update training
    - [ ] config generator
        - [x] setvae
        - [ ] poolmodel
        - [ ] neural statistician

    - [x] AdaMax to AdamW
    - [ ] MMD with EMA estimator for sigma
        - [x] code 
        - [x] incorporate to loss factory
        - [ ] final testing
        - [x] add into config generator
    - [ ] add reconstruction loop for test set
        - [ ] write function
        - [ ] evaluate final model
        - [ ] evaluate best model
        - [ ] save to folder
    - [ ] make new ModelNet dataset 
        - [ ] download data
            - [x] testing
            - [ ] rci
        - [ ] pick 10 interesting classes
        - [ ] sample points and save to file
        - [ ] prepare loading function
        - [ ] incorporate to create_dataloaders
