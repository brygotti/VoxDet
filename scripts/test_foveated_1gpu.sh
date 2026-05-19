#!/bin/bash
#SBATCH --job-name=voxdet_fov_test1gpu
#SBATCH --output=/home/lagutova/VI-Project/VoxDet/runs/fov_test_1gpu/slurm_%j.out
#SBATCH --error=/home/lagutova/VI-Project/VoxDet/runs/fov_test_1gpu/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --partition=gpu

# 1-GPU debug run for the foveated backbone config.
# CUDA_LAUNCH_BLOCKING=1 makes CUDA errors synchronous so the Python traceback
# points to the exact failing line instead of the next async sync point.
#
# Usage:  sbatch scripts/test_foveated_1gpu.sh <WANDB_API_KEY>

WANDB_KEY=$1

set -e

export WANDB_API_KEY=$WANDB_KEY
export WANDB_DIR="${TMPDIR:-/tmp}/wandb_${SLURM_JOB_ID:-$$}"
mkdir -p "$WANDB_DIR"
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

ROOT_DIR="/home/lagutova/VI-Project/VoxDet"
CONFIG="${ROOT_DIR}/configs/foveated-backbone-dev-semantickitti-cam.py"
LOG_FOLDER="${ROOT_DIR}/runs/fov_test_1gpu"

source /home/lagutova/miniconda/etc/profile.d/conda.sh
conda activate voxdet

mkdir -p "${LOG_FOLDER}"

cd "${ROOT_DIR}"

python main.py \
    --config_path    "${CONFIG}" \
    --log_folder     "${LOG_FOLDER}" \
    --seed           42 \
    --wandb \
    --wandb_project  voxdet \
    --wandb_run_name fov_test_1gpu
