#!/bin/bash
#SBATCH --job-name=voxdet_fov_backbone
#SBATCH --output=/home/lagutova/VI-Project/VoxDet/runs/foveated_backbone_run/slurm_%j.out
#SBATCH --error=/home/lagutova/VI-Project/VoxDet/runs/foveated_backbone_run/slurm_%j.err
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
# wandb-core writes its port file to WANDB_DIR; on SLURM the home dir is
# NFS-mounted and too slow, causing the 30 s service-start timeout.
# Point wandb at local /tmp storage so the Go service starts immediately.
export WANDB_DIR="${TMPDIR:-/tmp}/wandb_${SLURM_JOB_ID:-$$}"
mkdir -p "$WANDB_DIR"

# ── paths ────────────────────────────────────────────────────────────────────
ROOT_DIR="/home/lagutova/VI-Project/VoxDet"
CONFIG="${ROOT_DIR}/configs/foveated-backbone-dev-semantickitti-cam.py"
LOG_FOLDER="${ROOT_DIR}/runs/foveated_backbone_run"

# ── optional: resume from a checkpoint ───────────────────────────────────────
# CKPT_PATH="${ROOT_DIR}/runs/foveated_backbone_run/tensorboard/checkpoints/last.ckpt"

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
    --wandb_run_name foveated_backbone
    # --ckpt_path "${CKPT_PATH}"   # uncomment to resume
