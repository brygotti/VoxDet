#!/bin/bash
#SBATCH --job-name=voxdet_dist
#SBATCH --time=24:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=voxdet_dist_%j.out
#SBATCH --error=voxdet_dist_%j.err

# Batch-submit VoxDet training on Izar.
#
# Usage:
#   sbatch run_voxdet_distance.sh

cd "${SLURM_SUBMIT_DIR:-.}"

set -e

# Non-interactive batch shells often lack conda hook; common miniconda layout on Izar:
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

export WANDB_MODE=online
export WANDB_PROJECT=voxdet
export WANDB_NAME=baseline-dev-semantickitti-cam-distance

CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --config_path configs/baseline-dev-semantickitti-cam-distance.py \
  --log_folder baseline-dev-semantickitti-cam-distance \
  --seed 42 \
  --log_every_n_steps 100