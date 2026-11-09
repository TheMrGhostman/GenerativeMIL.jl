#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpufast
#SBATCH --mem=93G
#SBATCH --output=slurms/%x_%j.out


CONFIG=$1
SEED=$2
TIME_LIMIT=$3  
MODEL_DIR=$4
EPOCHS=$5

mkdir -p slurms

julia --project -e 'using Pkg; Pkg.instantiate(); @info("Instantiated") '
julia --project ./DiscreteBagVAE_train.jl ${CONFIG} ${SEED} ${TIME_LIMIT} ${MODEL_DIR} ${EPOCHS}

