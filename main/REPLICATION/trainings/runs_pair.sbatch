#!/bin/bash
# Two train_cluster_runs.py tasks on one GPU (2026-11-10): a batch-10k run keeps its GPU ~40% busy (one CPU core each) and
# needs ~14 GB, so two share a GPU. The pairs (array index -> two task ids) are the script's argument:
#   sbatch --array=0-8%3 runs_pair.sbatch "58:76 77:78 ..."
# Each task logs to logs/cgw-runs-task<id>-<jobid>.out. Never the 11 GB RTX 2080 Ti node.
#SBATCH --job-name=cgw-pair
#SBATCH --partition=gpu.q
#SBATCH --account=kadmon-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --exclude=elscn-108
#SBATCH --output=logs/%x-%A_%a.out
cd "$HOME/cgw"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU" in
  *Blackwell*|*"RTX PRO 6000"*) ENV=cgw128 ;;
  *) ENV=cgw ;;
esac
read -ra P <<< "$1"
IFS=: read -ra T <<< "${P[$SLURM_ARRAY_TASK_ID]}"
echo "pair $SLURM_ARRAY_TASK_ID on $(hostname): $GPU -> env $ENV, tasks ${T[*]}"
source "$HOME/miniforge3/bin/activate" "$ENV"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg PYTHONUNBUFFERED=1 TQDM_MININTERVAL=60 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # two processes share 32-96 GB
for t in "${T[@]}"; do
  python main/REPLICATION/trainings/train_cluster_runs.py --task "$t" --budget-h 13.5 \
    > "logs/cgw-runs-task$t-${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.out" 2>&1 &
done
wait
