#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Uprav jen tyto 4 veci:
CONFIGS=(
    "configs/mnist_configs_e/cd_setvae_e_c011.yml"
    "configs/mnist_configs_e/cd_setvae_e_c012.yml"
    "configs/mnist_configs_e/cd_setvae_e_c013.yml"
    "configs/mnist_configs_e/cd_setvae_e_c014.yml"
    "configs/mnist_configs_e/cd_setvae_e_c015.yml"
    "configs/mnist_configs_e/cd_setvae_e_c016.yml"
    "configs/mnist_configs_e/cd_setvae_e_c017.yml"
    "configs/mnist_configs_e/cd_setvae_e_c018.yml"
    "configs/mnist_configs_e/cd_setvae_e_c019.yml"
    "configs/mnist_configs_e/cd_setvae_e_c011.yml"

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
            "${SCRIPT_DIR}/run_setvae_training.sh" \
            "${cfg}" "${seed}" "${TIME_LIMIT}" "${model_dir}" "${EPOCHS}"
    done
done


: '

    '