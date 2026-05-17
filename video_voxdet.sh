#!/bin/bash
#SBATCH --job-name=voxdet_video
#SBATCH --time=02:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=voxdet_video_%j.out
#SBATCH --error=voxdet_video_%j.err

set -euo pipefail
module load gcc ffmpeg

INPUT_DIR=${1:?Usage: sbatch video_voxdet.sh <input_dir> <output_path>}
OUTPUT_PATH=${2:?Usage: sbatch video_voxdet.sh <input_dir> <output_path>}

if [[ ! -d "${INPUT_DIR}" ]]; then
	echo "Input directory not found: ${INPUT_DIR}" >&2
	exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
	echo "ffmpeg not found in PATH" >&2
	exit 1
fi

ffmpeg -y \
	-framerate 5 \
	-pattern_type glob \
	-i "${INPUT_DIR}/*.png" \
	-c:v libx264 \
	-pix_fmt yuv420p \
	"${OUTPUT_PATH}"
