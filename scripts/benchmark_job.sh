#!/bin/bash
#SBATCH --job-name=voxdet_bench
#SBATCH --time=02:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=voxdet_bench_%j.out
#SBATCH --error=voxdet_bench_%j.err

# Usage:
#   sbatch benchmark_job.sh <mode|config_file> <ckpt_path> <batch_sizes> [measure] [num_warmup] [num_iters] [device] [extra_args...]
#
# Examples:
#   sbatch benchmark_job.sh baseline ./ckpts/voxdet-semantickitti-cam.ckpt "1 2 4" both 10 50 cuda
#   sbatch benchmark_job.sh distance ./ckpts/voxdet-semantickitti-cam.ckpt "1" e2e 5 30 cuda
#   sbatch benchmark_job.sh configs/foveated-semantickitti-cam.py ./ckpts/voxdet-semantickitti-cam.ckpt "1 2" model 10 50 cuda
#   sbatch benchmark_job.sh baseline ./ckpts/voxdet-semantickitti-cam.ckpt "1" model 10 50 cuda --profile_parts

cd "${SLURM_SUBMIT_DIR:-.}"

RUN_SPEC=${1:?Usage: sbatch benchmark_job.sh <mode|config_file> <ckpt_path> <batch_sizes> [measure] [num_warmup] [num_iters] [device] [extra_args...]}
CKPT_PATH=${2:?Usage: sbatch benchmark_job.sh <mode|config_file> <ckpt_path> <batch_sizes> [measure] [num_warmup] [num_iters] [device] [extra_args...]}
BATCH_SIZES=${3:?Usage: sbatch benchmark_job.sh <mode|config_file> <ckpt_path> <batch_sizes> [measure] [num_warmup] [num_iters] [device] [extra_args...]}
MEASURE=${4:-model}
NUM_WARMUP=${5:-10}
NUM_ITERS=${6:-50}
DEVICE=${7:-cuda}
EXTRA_ARGS=("${@:8}")

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

export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "CKPT_PATH=$CKPT_PATH"
echo "BATCH_SIZES=$BATCH_SIZES"
echo "MEASURE=$MEASURE"
echo "NUM_WARMUP=$NUM_WARMUP"
echo "NUM_ITERS=$NUM_ITERS"
echo "DEVICE=$DEVICE"

CMD=(
  python -u tools/benchmark_infer.py
  --config_path "${CONFIG_FILE}"
  --ckpt_path "${CKPT_PATH}"
  --batch_sizes ${BATCH_SIZES}
  --measure "${MEASURE}"
  --num_warmup "${NUM_WARMUP}"
  --num_iters "${NUM_ITERS}"
  --device "${DEVICE}"
)

CUDA_VISIBLE_DEVICES=0 "${CMD[@]}" "${EXTRA_ARGS[@]}"
