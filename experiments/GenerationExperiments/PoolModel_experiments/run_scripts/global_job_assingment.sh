#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Uprav jen tyto veci:
CONFIGS=(
    "configs/cd_poolmodel_c1.yml"
    "configs/cd_poolmodel_c2.yml"
    "configs/cd_poolmodel_c3.yml"
    "configs/cd_poolmodel_c4.yml"
    "configs/cd_poolmodel_c5.yml"
    "configs/cd_poolmodel_c6.yml"
    "configs/cd_poolmodel_c7.yml"
    "configs/cd_poolmodel_c8.yml"
    "configs/mmd_poolmodel_c1.yml"
    "configs/mmd_poolmodel_c2.yml"
    "configs/mmd_poolmodel_c3.yml"
    "configs/mmd_poolmodel_c4.yml"
    "configs/mmd_poolmodel_c5.yml"
    "configs/mmd_poolmodel_c6.yml"
    "configs/mmd_poolmodel_c7.yml"
    "configs/mmd_poolmodel_c8.yml"
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
            "${SCRIPT_DIR}/run_poolmodel_modelnet.sh" \
            "${cfg}" "${seed}" "${TIME_LIMIT}" "${model_dir}" "${EPOCHS}"
    done
done