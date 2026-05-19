#!/bin/bash
#SBATCH --job-name=voxdet_debug
#SBATCH --output=/home/lagutova/VI-Project/VoxDet/runs/debug_run/slurm_%j.out
#SBATCH --error=/home/lagutova/VI-Project/VoxDet/runs/debug_run/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

# Usage:
#   sbatch scripts/debug_job.sh <wandb_api_key>
#
# Runs on 1 GPU with CUDA_LAUNCH_BLOCKING=1 so the traceback points to the
# exact faulty CUDA kernel rather than the next sync point.

WANDB_KEY=$1

set -e

export WANDB_API_KEY="${WANDB_KEY}"
export CUDA_LAUNCH_BLOCKING=1

ROOT_DIR="/home/lagutova/VI-Project/VoxDet"
CONFIG="${ROOT_DIR}/configs/foveated-backbone-dev-semantickitti-cam.py"
LOG_FOLDER="${ROOT_DIR}/runs/debug_run"

source /home/lagutova/miniconda/etc/profile.d/conda.sh
conda activate voxdet

mkdir -p "${LOG_FOLDER}"

cd "${ROOT_DIR}"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config_path    "${CONFIG}" \
    --log_folder     "${LOG_FOLDER}" \
    --seed           42
