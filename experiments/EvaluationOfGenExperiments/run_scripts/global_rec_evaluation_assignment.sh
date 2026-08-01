#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Models and datasets to evaluate
MODELS=("poolmodel" "setvae" "neuralstatistician")
DATASETS=("mnist" ) # "airplane" "core5"
SEEDS=(1)

# Evaluation parameters
VALID_REPEATS=2
TEST_REPEATS=5
LOSS_FUNCTIONS="cd,sh,dcd,mmd"
SINKHORN_EPSILON=1.0
DCD_ALPHA=1.0
MMD_SIGMA=1.32
MMD_MULTIPLIERS="0.25,0.5,1.0"
TIME_LIMIT=24

mkdir -p "${EXP_DIR}/slurms"

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            job_name="eval_${model}_${dataset}_s${seed}"
            echo "Submitting evaluation: model=${model}, dataset=${dataset}, seed=${seed}"
            sbatch --job-name="${job_name}" --chdir "${EXP_DIR}" \
                "${SCRIPT_DIR}/run_eval_reconstruction.sh" \
                ${model} ${dataset} ${seed} \
                ${VALID_REPEATS} ${TEST_REPEATS} \
                ${LOSS_FUNCTIONS} ${SINKHORN_EPSILON} \
                ${DCD_ALPHA} ${MMD_SIGMA} ${MMD_MULTIPLIERS} \
                ${TIME_LIMIT}
        done
    done
done

echo "All evaluation jobs submitted!"
