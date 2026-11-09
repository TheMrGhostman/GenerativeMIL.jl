#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Uprav jen tyto 4 veci:
CONFIGS=( configs/bag_digits_configs/dsqvae_caw_c0{01..12}.yml )


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
            "${SCRIPT_DIR}/run_discretebagvae_training.sh" \
            "${cfg}" "${seed}" "${TIME_LIMIT}" "${model_dir}" "${EPOCHS}"
    done
done


: '
CONFIGS=(
    "configs/bag_digits_configs/discretebagvae_caw_c001.yml"
    "configs/bag_digits_configs/discretebagvae_caw_c002.yml"
    "configs/bag_digits_configs/discretebagvae_caw_c003.yml"
    "configs/bag_digits_configs/discretebagvae_caw_c004.yml"
    "configs/bag_digits_configs/discretebagvae_caw_c005.yml"
    "configs/bag_digits_configs/discretebagvae_caw_c006.yml"
    "configs/bag_digits_configs/discretebagvae_caw_c007.yml"
    "configs/bag_digits_configs/discretebagvae_caw_c008.yml"
)
    '