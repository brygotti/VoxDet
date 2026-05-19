#!/bin/bash
#SBATCH --job-name=flops_tokens
#SBATCH --output=/home/lagutova/VI-Project/VoxDet/runs/flops_tokens/slurm_%j.out
#SBATCH --error=/home/lagutova/VI-Project/VoxDet/runs/flops_tokens/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --partition=gpu

set -e

ROOT_DIR="/home/lagutova/VI-Project/VoxDet"
LOG_DIR="${ROOT_DIR}/runs/flops_tokens"

source /home/lagutova/miniconda/etc/profile.d/conda.sh
conda activate voxdet

mkdir -p "${LOG_DIR}"

cd "${ROOT_DIR}"

# Install fvcore if not already present (fast, no deps beyond torch)
pip install fvcore --quiet 2>/dev/null || true

python scripts/flops_tokens.py | tee "${LOG_DIR}/results.txt"
