#!/bin/bash
#SBATCH --job-name=semantic_preprocess
#SBATCH --time=10:00:00
#SBATCH --account=cs-503
#SBATCH --qos=cs-503
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=semantic_preprocess_%j.out
#SBATCH --error=semantic_preprocess_%j.err

source /home/gotti/anaconda3/etc/profile.d/conda.sh
conda activate voxdet

cd /home/gotti/VoxDet/tools
python preprocess.py --kitti_root /scratch/izar/gotti/semantic_kitti/ --kitti_preprocess_root /scratch/izar/gotti/semantic_kitti/