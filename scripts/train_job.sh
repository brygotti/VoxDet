#!/bin/bash
#SBATCH --job-name=voxdet_train
#SBATCH --time=20:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=voxdet_train_%j.out
#SBATCH --error=voxdet_train_%j.err

# Usage:
#   sbatch train_job.sh <mode|config_file> <wandb_api_key> <run_name> <num_gpus>
#
# Examples:
#   sbatch train_job.sh baseline <WANDB_KEY> voxdet-baseline 2
#   sbatch train_job.sh distance <WANDB_KEY> voxdet-distance 2
#   sbatch train_job.sh configs/baseline-dev-semantickitti-cam-distance.py <WANDB_KEY> voxdet-distance 2

cd "${SLURM_SUBMIT_DIR:-.}"

RUN_SPEC=${1:?Usage: sbatch train_job.sh <mode|config_file> <wandb_api_key> <run_name> <num_gpus>}
WANDB_KEY=${2:?Usage: sbatch train_job.sh <config_file> <wandb_api_key> <run_name> <num_gpus>}
RUN_NAME=${3:?Usage: sbatch train_job.sh <config_file> <wandb_api_key> <run_name> <num_gpus>}
NUM_GPUS=${4:?Usage: sbatch train_job.sh <config_file> <wandb_api_key> <run_name> <num_gpus>}

case "${RUN_SPEC}" in
  baseline)
    CONFIG_FILE="configs/baseline-dev-semantickitti-cam.py"
    ;;
  distance)
    CONFIG_FILE="configs/baseline-dev-semantickitti-cam-distance.py"
    ;;
  foveated)
    CONFIG_FILE="configs/foveated-backbone-dev-semantickitti-cam.py"
    ;;
  ablation-voxel-only)
    CONFIG_FILE="configs/ablation-voxel-only-dev-semantickitti-cam.py"
    ;;
  ablation-query-only)
    CONFIG_FILE="configs/ablation-query-only-dev-semantickitti-cam.py"
    ;;
  *.py)
    CONFIG_FILE="${RUN_SPEC}"
    ;;
  *)
    echo "Unknown run spec: ${RUN_SPEC}" >&2
    echo "Use baseline, distance, foveated, ablation-voxel-only, ablation-query-only, or a direct config path ending in .py" >&2
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

export WANDB_API_KEY="${WANDB_KEY}"
export WANDB_MODE=online
export WANDB_PROJECT=voxdet
export WANDB_NAME="${RUN_NAME}"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "RUN_NAME=$RUN_NAME"
echo "NUM_GPUS=$NUM_GPUS"

CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --config_path "${CONFIG_FILE}" \
  --log_folder "${RUN_NAME}" \
  --seed 42 \
  --log_every_n_steps 100