#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Uprav jen tyto veci:
CONFIGS=(
    "configs/core5_configs/l2_neuralstatistician_c001.yml"
    "configs/core5_configs/l2_neuralstatistician_c002.yml"
    "configs/core5_configs/l2_neuralstatistician_c003.yml"
    "configs/core5_configs/l2_neuralstatistician_c004.yml"
    "configs/core5_configs/l2_neuralstatistician_c005.yml"
    "configs/core5_configs/l2_neuralstatistician_c006.yml"
    "configs/core5_configs/l2_neuralstatistician_c007.yml"
    "configs/core5_configs/l2_neuralstatistician_c008.yml"
    "configs/core5_configs/l2_neuralstatistician_c009.yml"
    "configs/core5_configs/l2_neuralstatistician_c010.yml"
    "configs/core5_configs/l2_neuralstatistician_c011.yml"
    "configs/core5_configs/l2_neuralstatistician_c012.yml"

 
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
            "${SCRIPT_DIR}/run_neuralstatistician_training.sh" \
            "${cfg}" "${seed}" "${TIME_LIMIT}" "${model_dir}" "${EPOCHS}"
    done
done