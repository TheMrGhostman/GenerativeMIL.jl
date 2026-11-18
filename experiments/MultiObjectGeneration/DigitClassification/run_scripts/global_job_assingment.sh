#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Uprav jen tyto 4 veci:
CONFIGS=(
    # HSTree evaluation classifiers (mnist_clock / mnist_clock_unbalanced digits); keep SEEDS=(1) -- the data split
    # seed must match the HST runs (seed=1)
    "configs/mnist_configs/clf_clock_bal.yml"
    "configs/mnist_configs/clf_clock_unb.yml"
)
SEEDS=(1)
TIME_LIMIT=24
EPOCHS=-1

mkdir -p "${EXP_DIR}/slurms"

for cfg in "${CONFIGS[@]}"; do
    cfg_name="$(basename "${cfg}" .yml)"
    for seed in "${SEEDS[@]}"; do
        model_dir="${cfg_name}"
        job_name="${cfg_name}_s${seed}"
        echo "Submitting ${cfg_name}, seed=${seed}"
        sbatch --job-name="${job_name}" --chdir "${EXP_DIR}" \
            "${SCRIPT_DIR}/run_clf_training.sh" \
            "${cfg}" "${seed}" "${TIME_LIMIT}" "${model_dir}" "${EPOCHS}"
    done
done


: '

    '