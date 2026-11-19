#!/bin/bash
# Extensions of the replicates to 3,005 episodes (2026-11-18): array index i -> replicate i // 2 + 1, network trained / echo
# (i % 2), 1,005 episodes each from the saved state at 2,005 (run_extend.py). At most 3 at a time (with the C = 3 run: 4).
#   sbatch --array=4-15%3 extend.sbatch   (replicates 3-8; 1-2 run locally)
#SBATCH --job-name=cgw-ext
#SBATCH --partition=gpu.q
#SBATCH --account=kadmon-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
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
python run_extend.py "$net" "$rep" 0
