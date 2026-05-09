#!/bin/bash
#SBATCH --job-name=voxdet_preprocess_depth
#SBATCH --output=/home/lagutova/VI-Project/VoxDet/runs/preprocess/slurm_%j.out
#SBATCH --error=/home/lagutova/VI-Project/VoxDet/runs/preprocess/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu          # <-- change to your cluster partition

set -e

source /home/lagutova/miniconda/etc/profile.d/conda.sh
conda activate voxdet

mkdir -p /home/lagutova/VI-Project/VoxDet/runs/preprocess

DATA_PATH='/scratch/izar/lagutova/semantickittii/dataset'
CKPT='/home/lagutova/VI-Project/VoxDet/preprocess/mobilestereonet/MSNet3D_SF_DS_KITTI2015.ckpt'
PREPROCESS_DIR='/home/lagutova/VI-Project/VoxDet/preprocess/mobilestereonet'

cd "${PREPROCESS_DIR}"

run_seq() {
    local baseline=$1
    local seq=$2
    python prediction.py \
        --datapath "${DATA_PATH}/sequences/${seq}" \
        --testlist "./filenames/${seq}.txt" \
        --num_seq "${seq}" \
        --loadckpt "${CKPT}" \
        --dataset kitti \
        --model MSNet3D \
        --savepath "${DATA_PATH}/depth" \
        --baseline "${baseline}"
}

for i in 00 01 02;          do run_seq 388.1823 $i; done
for i in 03;                do run_seq 389.6304 $i; done
for i in 04 05 06 07 08 09 10 11 12; do run_seq 381.8293 $i; done
for i in 13 14 15 16 17 18 19 20 21; do run_seq 388.1823 $i; done
