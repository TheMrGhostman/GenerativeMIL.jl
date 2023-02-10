#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpulong
#SBATCH --mem=150G

CONFIG=$1

NPOINTS=$2

SEED=$3  
# seed, if seed =< 0 it is considered concrete single seed to train with

TIME_LIMIT=48

module load Julia/1.7.2-linux-x86_64
julia --project -e 'using Pkg; Pkg.instantiate(); @info("Instantiated") '

julia --project ./setvae_modelnet_config.jl ${CONFIG} ${NPOINTS} ${SEED} $TIME_LIMIT

