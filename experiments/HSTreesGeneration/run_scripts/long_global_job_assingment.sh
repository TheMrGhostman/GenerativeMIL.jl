#!/usr/bin/env bash
# Like global_job_assingment.sh, but 72 h jobs on gpulong (run_smhsqvae_training_long.sh).
# DRY_RUN=1 ./long_global_job_assingment.sh  only prints what would be submitted.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Uprav jen tyto 4 veci:
CONFIGS=(
    "configs/test_configs/cd_smhsqvae_mnist_clock_unbalanced_test.yml"
)
SEEDS=(1)
TIME_LIMIT=72
EPOCHS=-1

DRY_RUN=${DRY_RUN:-0}
mkdir -p "${EXP_DIR}/slurms"

for cfg in "${CONFIGS[@]}"; do
    [[ -f "${EXP_DIR}/${cfg}" ]] || { echo "Missing config: ${cfg}" >&2; exit 1; }
    cfg_name="$(basename "${cfg}" .yml)"
    case "${cfg}" in
        *mnist_clock_unbalanced_configs*) tag="unb" ;;
        *mnist_clock_configs*)            tag="bal" ;;
        *)                                tag="x" ;;
    esac
    for seed in "${SEEDS[@]}"; do
        model_dir="${cfg_name}"
        job_name="${tag}_${cfg_name}_s${seed}"
        echo "Submitting ${tag} ${cfg_name}, seed=${seed}"
        if [[ "${DRY_RUN}" == "1" ]]; then
            echo "  sbatch --job-name=${job_name} --chdir ${EXP_DIR} run_smhsqvae_training_long.sh ${cfg} ${seed} ${TIME_LIMIT} ${model_dir} ${EPOCHS}"
            continue
        fi
        sbatch --job-name="${job_name}" --chdir "${EXP_DIR}" \
            "${SCRIPT_DIR}/run_smhsqvae_training_long.sh" \
            "${cfg}" "${seed}" "${TIME_LIMIT}" "${model_dir}" "${EPOCHS}"
    done
done
