# GenerativeMIL
Repository in still development!!!

This repository is being developed as complement to GroupAD.jl, where the most procedures are located. 
GenerativeMIL mostly provide advanced models for generative modeling of Muliti Instance Learning (MIL) and Set structured data. 
Models are implemented in the most optimal which we could think of. So there might be other better way.

## Model Zoo
| Implemented models | CPU training | GPU training | variable cardinality (in/out) | note |
|---|---|---|---|---|
| [SetVAE][setvae] | yes | yes | yes/yes | Implementation is 1:1 Python to Julia code from original repository. | 
| [FoldingNet VAE][foldingnet] | yes | no | yes/no | batched training on CPU via broadcasting |
| PoolModel (ours) | yes | no (Todo) | yes/yes | |
| [SetTransformer][settransformer] | yes | yes | yes/no | classifier version only | 
| [Masked Autoencoder for Distribution Estimation][made] (MADE) | yes | yes | yes |  / TODO add support for multiple masks[^1].|
| [Masked Autoregressive Flow][maf] (MAF)| ? | ? |  | not finished |
| [Inverse Autoregresive Flow][iaf] (IAF)| ? | ? |  | not finished |
| [SoftPointFlow][softflow] | ? | ? | yes/yes | not finished |
| SetVAEformer (ours) | yes | yes | yes/yes | not finished/ Similar to Vanilla SetVAE but better ;) | 

[^1]: This model is essentially building block for MAF, IAF and SoftPointFlow
[setvae]: https://openaccess.thecvf.com/content/CVPR2021/papers/Kim_SetVAE_Learning_Hierarchical_Composition_for_Generative_Modeling_of_Set-Structured_Data_CVPR_2021_paper.pdf
[foldingnet]: https://ieeexplore.ieee.org/document/9506795
[settransformer]: http://proceedings.mlr.press/v97/lee19d/lee19d.pdf
[made]: https://arxiv.org/pdf/1502.03509.pdf 
[af]: https://homepages.inf.ed.ac.uk/imurray2/pub/17maf/maf.pdf
[iaf]: https://arxiv.org/pdf/1606.04934.pdf
[softflow]: https://arxiv.org/pdf/2006.04604.pdf

## DrWatson
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


