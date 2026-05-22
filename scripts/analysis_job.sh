#!/bin/bash
#SBATCH --job-name=voxdet_analysis
#SBATCH --time=00:10:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=voxdet_analysis_%j.out
#SBATCH --error=voxdet_analysis_%j.err

cd "${SLURM_SUBMIT_DIR:-.}"

set -e

# shellcheck disable=SC1091
if [[ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook 2>/dev/null)" || true
fi
conda activate voxdet

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"

python tools/full_pipeline_analysis.py
