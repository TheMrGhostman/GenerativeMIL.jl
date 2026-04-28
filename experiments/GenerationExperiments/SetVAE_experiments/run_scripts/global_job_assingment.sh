#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Uprav jen tyto 4 veci:
CONFIGS=(
    "configs/mmd_setvae_c7.yml"
    "configs/mmd_setvae_c8.yml"
    "configs/mmd_setvae_c9.yml"
    "configs/mmd_setvae_c10.yml"    
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
            "${SCRIPT_DIR}/run_setvae_modelnet.sh" \
            "${cfg}" "${seed}" "${TIME_LIMIT}" "${model_dir}" "${EPOCHS}"
    done
done


: '

"configs/mmd_setvae_c1.yml"
"configs/mmd_setvae_c2.yml"
"configs/mmd_setvae_c3.yml"
"configs/mmd_setvae_c4.yml"
"configs/mmd_setvae_c5.yml"
"configs/mmd_setvae_c6.yml"
"configs/mmd_setvae_c7.yml"
"configs/mmd_setvae_c8.yml"
"configs/mmd_setvae_c9.yml"
"configs/mmd_setvae_c10.yml"
"configs/mmd_setvae_c11.yml"
"configs/mmd_setvae_c12.yml"
"configs/mmd_setvae_c13.yml"
"configs/mmd_setvae_c14.yml"
"configs/setvae_c1.yml"
"configs/setvae_c2.yml"
"configs/setvae_c3.yml"
"configs/setvae_c4.yml"
"configs/setvae_c5.yml"
"configs/setvae_c6.yml"
"configs/setvae_c7.yml"
"configs/setvae_c8.yml"
"configs/setvae_c9.yml"
"configs/setvae_c10.yml"
"configs/setvae_c11.yml"
"configs/setvae_c12.yml"
"configs/setvae_c13.yml"
"configs/setvae_c14.yml"
    '