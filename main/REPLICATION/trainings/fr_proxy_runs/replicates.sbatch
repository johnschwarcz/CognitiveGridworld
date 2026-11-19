#!/bin/bash
# Replicates of the training-dynamics pair (2026-11-18): array index i -> replicate i // 2 + 1, network trained / echo
# (i % 2), 2,005 episodes each (run_replicate.py). At most 3 at a time (with the C = 3 run: our 4-GPU allowance).
#   sbatch --array=0-7%3 replicates.sbatch
#SBATCH --job-name=cgw-rep
#SBATCH --partition=gpu.q
#SBATCH --account=kadmon-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --exclude=elscn-108,elscn-189,elscn-191
#SBATCH --output=logs/%x-%A_%a.out
cd "$HOME/cgw"
NETS=(trained echo)
net=${NETS[$((SLURM_ARRAY_TASK_ID % 2))]}
rep=$((SLURM_ARRAY_TASK_ID / 2 + 1))
echo "replicate $rep $net on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
source "$HOME/miniforge3/bin/activate" cgw
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg PYTHONUNBUFFERED=1 TQDM_MININTERVAL=60 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run_replicate.py "$net" "$rep" 0
