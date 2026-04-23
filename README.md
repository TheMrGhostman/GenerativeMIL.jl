# GenerativeMIL

GenerativeMIL is a Julia project for generative modeling of multi-instance and set-structured data.
It complements GroupAD.jl with reusable building blocks, model implementations, and training utilities
for set-based generative and discriminative experiments.

The repository is still under active development, so APIs and model coverage may change.

## Highlights

- Set-based generative models and attention blocks implemented in Julia.
- CPU and GPU training paths for the main research models.
- DrWatson-based project layout for reproducible experiments.
- Support for variable-cardinality set data through masking where available.

## Current Status

The codebase is usable, but not all models are finished yet.
Some components are research prototypes rather than polished production APIs.

Important note for the current setup:

- Do not import `cuDNN` in this project for now. In the current environment it breaks `softmax`.

## Installation

1. Clone or download the repository.
2. Start Julia in the project directory.
3. Activate and instantiate the environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

If you use DrWatson workflows, you can also rely on `quickactivate` from within the project.

## Project Layout

- `src/` contains the package code: building blocks, models, losses, utilities, and evaluation helpers.
- `scripts/` contains runnable experiment and training entry points.
- `experiments/` contains experiment-specific runs and outputs.
- `test/` contains smoke tests for CPU and GPU paths.

## Model Zoo

| Implemented models | CPU training | GPU training | variable cardinality[^1] (in/out)[^2] | note |
|---|---|---|---|---|
| [SetVAE][setvae] | yes | yes | yes/yes | Implementation is close to the original Python code. |
| [FoldingNet VAE][foldingnet] | yes | yes [^5] | yes/no | Batched training on CPU via broadcasting. |
| PoolModel (ours) | yes | yes [^*] | yes/yes | Masked forward pass for variable cardinality on GPU is still TODO. |
| [SetTransformer][settransformer] | yes | yes | yes/no | Classifier version only. |
| [Masked Autoencoder for Distribution Estimation][made] (MADE) | yes | yes | possible[^3]/no | Multiple-mask support is still TODO. |
| [Masked Autoregressive Flow][maf] (MAF) | ? | ? |  | Not finished. |
| [Inverse Autoregresive Flow][iaf] (IAF) | ? | ? |  | Not finished. |
| [SoftPointFlow][softflow] | ? | ? | yes/yes | Not finished. |
| SetVAEformer (ours) | yes | yes | yes/yes | Work in progress. |

[^1]: Cardinality means the number of elements in a single bag/set. In real data this can differ per sample, which complicates batching.

[^2]: "in" variable cardinality means varying set sizes in the input batch; "out" variable cardinality means the model can generate outputs with a different number of elements than the input.

[^3]: This model has no cardinality reduction or expansion.

[^4]: This model is effectively a building block for MAF, IAF, and SoftPointFlow.

[^*]: PoolModel currently works only for constant cardinality.

[^5]: FoldingNet VAE is trainable on GPU via `fit_gpu_ready!`. It is a special case with fixed cardinality and without KLD of reconstructed encoding.

## Reproducibility

This project uses [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) to keep experiments organized and reproducible.
Most scripts assume the repository root as the active project directory.

## References

[setvae]: https://openaccess.thecvf.com/content/CVPR2021/papers/Kim_SetVAE_Learning_Hierarchical_Composition_for_Generative_Modeling_of_Set-Structured_Data_CVPR_2021_paper.pdf
[foldingnet]: https://ieeexplore.ieee.org/document/9506795
[settransformer]: http://proceedings.mlr.press/v97/lee19d/lee19d.pdf
[made]: https://arxiv.org/pdf/1502.03509.pdf
[maf]: https://homepages.inf.ed.ac.uk/imurray2/pub/17maf/maf.pdf
[iaf]: https://arxiv.org/pdf/1606.04934.pdf
[softflow]: https://arxiv.org/pdf/2006.04604.pdf



