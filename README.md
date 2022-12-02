# GenerativeMIL
Repository in still development!!!

This repository is being developed as complement to GroupAD.jl, where the most procedures are located. 
GenerativeMIL mostly provide advanced models for generative modeling of Muliti Instance Learning (MIL) and Set structured data. 
Models are implemented in the most optimal which we could think of. So there might be other better way.


| Implemented models | CPU training | GPU training | variable cardinality (in/out) | note |
|---|---|---|---|---|
| SetVAE | yes | yes | yes/yes | Implementation is 1:1 Python to Julia code from original repository. | 
| FoldingNet VAE | yes | no | yes/no | batched training on CPU via broadcasting |
| PoolModel (ours) | yes | no (Todo) | yes/yes | |
| SetTransformer | yes | yes | yes/no | classifier version only | 
| Autoregressive Flow | ? | ? |  | not finished |
| Inverse Autoregresive Flow | ? | ? |  | not finished |
| SoftPointFlow | ? | ? | yes/yes | not finished |
| SetVAEformer (ours) | yes | yes | yes/yes | not finished/ Similar to Vanilla SetVAE but better ;) | 


This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> GenerativeMIL

To (locally) reproduce this project, do the following:

1. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
2. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.


