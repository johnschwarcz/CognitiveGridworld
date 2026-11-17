#!/bin/bash
# runs_single.sbatch for runs longer than a day (2026-11-17): one train_cluster_runs.py task per GPU, up to 2 days; the
# wall-time budget (--budget-h, default 44) is the second argument. A STOP file (runs/STOP_<name>) saves the net earlier.
#   sbatch --array=0-1 --gres=gpu:a100:1 runs_single_long.sbatch "572 573" 44
# Each task logs to logs/cgw-runs-task<id>-<jobid>.out. Never the 11 GB RTX 2080 Ti node.
#SBATCH --job-name=cgw-long
#SBATCH --partition=gpu.q
#SBATCH --account=kadmon-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=elscn-108,elscn-189,elscn-191   # 189/191 = RTX PRO 6000 (Blackwell): no cgw128 env, every job there dies
#SBATCH --output=logs/%x-%A_%a.out
cd "$HOME/cgw"
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
case "$GPU" in
  *Blackwell*|*"RTX PRO 6000"*) ENV=cgw128 ;;
  *) ENV=cgw ;;
esac
read -ra T <<< "$1"
t=${T[$SLURM_ARRAY_TASK_ID]}
echo "task $t on $(hostname): $GPU -> env $ENV"
source "$HOME/miniforge3/bin/activate" "$ENV"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg PYTHONUNBUFFERED=1 TQDM_MININTERVAL=60 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main/REPLICATION/trainings/train_cluster_runs.py --task "$t" --budget-h "${2:-44}" \
  > "logs/cgw-runs-task$t-${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.out" 2>&1
