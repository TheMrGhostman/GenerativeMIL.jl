#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=90G
#SBATCH --output=slurms/%x_%j.out


MODEL=${1:-"setvae"}
DATASET=${2:-"mnist"}
SEED=${3:-1}
VALID_REPEATS=${4:-2}
TEST_REPEATS=${5:-5}
LOSS_FUNCTIONS=${6:-"cd,sh,dcd,mmd"}
SINKHORN_EPSILON=${7:-1.0}
DCD_ALPHA=${8:-1.0}
MMD_SIGMA=${9:-1.32}
MMD_MULTIPLIERS=${10:-"0.25,0.5,1.0"}
TIME_LIMIT=${11:-24}

mkdir -p slurms

julia --project -e 'using Pkg; Pkg.instantiate(); @info("Instantiated") '
julia --project ./run_evaluation_reconstruction.jl ${MODEL} ${DATASET} ${SEED} ${VALID_REPEATS} ${TEST_REPEATS} ${LOSS_FUNCTIONS} ${SINKHORN_EPSILON} ${DCD_ALPHA} ${MMD_SIGMA} ${MMD_MULTIPLIERS} ${TIME_LIMIT}