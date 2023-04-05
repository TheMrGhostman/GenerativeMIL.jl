#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --mem=30G

CONFIG=$1

SEED=$2
# seed, if seed =< 0 it is considered concrete single seed to train with

TIME_LIMIT=$3

module load Julia/1.7.2-linux-x86_64
julia --project -e 'using Pkg; Pkg.instantiate(); @info("Instantiated") '

julia --project ./vq_poolae_config.jl ${CONFIG} ${SEED} $TIME_LIMIT

