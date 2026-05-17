#!/bin/bash
#SBATCH --job-name=voxdet_eval
#SBATCH --time=14:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=voxdet_eval_%j.out
#SBATCH --error=voxdet_eval_%j.err

# Usage:
#   sbatch eval_job.sh <mode|config_file> <ckpt_path> <run_name> [save_path]
#
# Examples:
#   sbatch eval_job.sh baseline ./ckpts/voxdet-semantickitti-cam.ckpt voxdet-semantickitti-cam-eval
#   sbatch eval_job.sh distance ./ckpts/voxdet-semantickitti-cam.ckpt voxdet-semantickitti-cam-eval pred
#   sbatch eval_job.sh configs/voxdet-semantickitti-cam.py ./ckpts/voxdet-semantickitti-cam.ckpt voxdet-semantickitti-cam-eval

cd "${SLURM_SUBMIT_DIR:-.}"

RUN_SPEC=${1:?Usage: sbatch eval_job.sh <mode|config_file> <ckpt_path> <run_name> [save_path]}
CKPT_PATH=${2:?Usage: sbatch eval_job.sh <mode|config_file> <ckpt_path> <run_name> [save_path]}
RUN_NAME=${3:?Usage: sbatch eval_job.sh <mode|config_file> <ckpt_path> <run_name> [save_path]}
SAVE_PATH=${4:-}

case "${RUN_SPEC}" in
  baseline)
    CONFIG_FILE="configs/baseline-dev-semantickitti-cam.py"
    ;;
  distance)
    CONFIG_FILE="configs/baseline-dev-semantickitti-cam-distance.py"
    ;;
  foveated)
    CONFIG_FILE="configs/foveated-semantickitti-cam.py"
    ;;
  foveated-backbone)
    CONFIG_FILE="configs/foveated-backbone-dev-semantickitti-cam.py"
    ;;
  *.py)
    CONFIG_FILE="${RUN_SPEC}"
    ;;
  *)
    echo "Unknown run spec: ${RUN_SPEC}" >&2
    echo "Use baseline, distance, foveated, foveated-backbone, or a direct config path ending in .py" >&2
    exit 1
    ;;
esac

set -e

# Non-interactive batch shells often lack conda hook; common miniconda layout on Izar.
# shellcheck disable=SC1091
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook 2>/dev/null)" || true
fi
conda activate voxdet

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "RUN_NAME=$RUN_NAME"
echo "CKPT_PATH=$CKPT_PATH"
echo "SAVE_PATH=$SAVE_PATH"

CMD=(
  python main.py
  --eval
  --ckpt_path "${CKPT_PATH}"
  --config_path "${CONFIG_FILE}"
  --log_folder "${RUN_NAME}"
  --seed 42
  --log_every_n_steps 100
)

if [[ -n "${SAVE_PATH}" ]]; then
  CMD+=(--save_path "${SAVE_PATH}")
fi

CUDA_VISIBLE_DEVICES=0,1 "${CMD[@]}"