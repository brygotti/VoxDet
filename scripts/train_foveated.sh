#!/bin/bash
#SBATCH --job-name=voxdet_foveated
#SBATCH --output=/home/lagutova/VI-Project/VoxDet/runs/foveated_run/slurm_%j.out
#SBATCH --error=/home/lagutova/VI-Project/VoxDet/runs/foveated_run/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu          # <-- change to your cluster partition

WANDB_KEY=$1        # First argument: your W&B API key

set -e

export WANDB_API_KEY=$WANDB_KEY

# ── paths ────────────────────────────────────────────────────────────────────
ROOT_DIR="/home/lagutova/VI-Project/VoxDet"
CONFIG="${ROOT_DIR}/configs/voxdet-semantickitti-cam.py"
LOG_FOLDER="${ROOT_DIR}/runs/foveated_run"

# ── optional: resume from a checkpoint ───────────────────────────────────────
# CKPT_PATH="${ROOT_DIR}/runs/foveated_run/tensorboard/checkpoints/last.ckpt"

mkdir -p "${LOG_FOLDER}"

cd "${ROOT_DIR}"

python main.py \
    --config_path    "${CONFIG}" \
    --log_folder     "${LOG_FOLDER}" \
    --seed           42 \
    --wandb \
    --wandb_project  voxdet \
    --wandb_run_name foveated_voxdet
    # --ckpt_path "${CKPT_PATH}"   # uncomment to resume
