#!/bin/bash
# Several train_cluster_runs.py tasks on ONE GPU (2026-11-14): for small-batch runs (batch 1,000: ~2.2 GB of GPU memory and
# one CPU core each), so many fit beside each other. The groups (array index -> colon-separated task ids) are the script's
# argument:   sbatch --array=0-1 runs_pack.sbatch "320:321:322:323:324:325 326:327:328:329:330:331"
# Each task logs to logs/cgw-runs-task<id>-<jobid>_<i>.out; --budget-h 46 makes every run stop and save its net before the
# 48 h limit (continue later with RESUME). Never the 11 GB RTX 2080 Ti node or the Blackwell nodes (no cgw128 env).
#SBATCH --job-name=cgw-pack
#SBATCH --partition=gpu.q
#SBATCH --account=kadmon-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --exclude=elscn-108,elscn-189,elscn-191
#SBATCH --output=logs/%x-%A_%a.out
cd "$HOME/cgw"
read -ra P <<< "$1"
IFS=: read -ra T <<< "${P[$SLURM_ARRAY_TASK_ID]}"
echo "group $SLURM_ARRAY_TASK_ID on $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), tasks ${T[*]}"
source "$HOME/miniforge3/bin/activate" cgw
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg PYTHONUNBUFFERED=1 TQDM_MININTERVAL=60 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for t in "${T[@]}"; do
  python main/REPLICATION/trainings/train_cluster_runs.py --task "$t" --budget-h 46 \
    > "logs/cgw-runs-task$t-${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.out" 2>&1 &
done
wait
