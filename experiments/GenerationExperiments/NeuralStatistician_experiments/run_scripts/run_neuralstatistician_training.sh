#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=90G
#SBATCH --output=slurms/%x_%j.out


CONFIG=$1
SEED=$2
TIME_LIMIT=$3
MODEL_DIR=$4
EPOCHS=$5

mkdir -p slurms

julia --project -e 'using Pkg; Pkg.instantiate(); @info("Instantiated") '
julia --project ./train_neuralstatistician.jl ${CONFIG} ${SEED} ${TIME_LIMIT} ${MODEL_DIR} ${EPOCHS}
