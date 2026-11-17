#!/bin/bash
# One train_cluster_runs.py task per GPU, 24 h (2026-11-13): for runs that do not fit runs_pair.sbatch's pairing, e.g. the
# batch-10k dream runs (a second Classifier forward/backward per iteration: ~1.9x the time of their twins, which took 8.1 h
# as pairs, so ~15.4 h; more GPU memory than half a GPU). The task list is the script's argument, one task per array index:
#   sbatch --array=0-1 runs_single.sbatch "252 253"
# Each task logs to logs/cgw-runs-task<id>-<jobid>.out. Never the 11 GB RTX 2080 Ti node.
#SBATCH --job-name=cgw-single
#SBATCH --partition=gpu.q
#SBATCH --account=kadmon-lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
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
python main/REPLICATION/trainings/train_cluster_runs.py --task "$t" --budget-h 22 \
  > "logs/cgw-runs-task$t-${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.out" 2>&1
