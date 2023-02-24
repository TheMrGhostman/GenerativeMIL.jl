#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=150G

NPOINTS=$1

SEED=$2  
# seed, if seed =< 0 it is considered concrete single seed to train with

RANDOM_SEED=$3
# random seed for sample_params function (to be able to train multile seeds in parallel)

TIME_LIMIT=48

module load Julia/1.7.2-linux-x86_64
julia --project -e 'using Pkg; Pkg.instantiate(); @info("Instantiated") '

julia --project ./setvae_modelnet.jl ${NPOINTS} ${SEED} ${RANDOM_SEED} $TIME_LIMIT

