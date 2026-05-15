#!/bin/bash
#SBATCH --job-name=voxdet_vis
#SBATCH --time=14:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=voxdet_vis_%j.out
#SBATCH --error=voxdet_vis_%j.err

# Usage:
#   sbatch visualize_voxdet.sh <data_root> <prediction_root> <save_path>
#
# Examples:
#   sbatch visualize_voxdet.sh /scratch/izar/gotti/semantic_kitti pred visualize

cd "${SLURM_SUBMIT_DIR:-.}"

DATA_ROOT=${1:?Usage: sbatch visualize_voxdet.sh <data_root> <prediction_root> <save_path>}
PREDICTION_ROOT=${2:?Usage: sbatch visualize_voxdet.sh <data_root> <prediction_root> <save_path>}
SAVE_PATH=${3:?Usage: sbatch visualize_voxdet.sh <data_root> <prediction_root> <save_path>}

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
echo "DATA_ROOT=$DATA_ROOT"
echo "PREDICTION_ROOT=$PREDICTION_ROOT"
echo "SAVE_PATH=$SAVE_PATH"

CMD=(
  python tools/visualize.py
  --data_root "${DATA_ROOT}"
  --prediction_root "${PREDICTION_ROOT}"
  --save_path "${SAVE_PATH}"
)

CUDA_VISIBLE_DEVICES=0 "${CMD[@]}"