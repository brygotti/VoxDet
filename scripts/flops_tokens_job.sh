#!/bin/bash
#SBATCH --job-name=flops_tokens
#SBATCH --output=flops_tokens_%j.out
#SBATCH --error=flops_tokens_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --partition=gpu

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

# Install fvcore if not already present (fast, no deps beyond torch)
pip install fvcore --quiet 2>/dev/null || true

python tools/flops_tokens.py | tee "./results.txt"
