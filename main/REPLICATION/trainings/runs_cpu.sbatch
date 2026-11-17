#!/bin/bash
# One train_cluster_runs.py task per array index on a cluster CPU node (2026-11-13): for runs without a GPU, e.g. the
# imagination capability test (imagine_* / teach_* / rl_*: a fresh Classifier on a frozen trained net, batch 1,000,
# 15,000 training iterations; gpu_inference off in their configs). The argument is a list of task NAMES; each array task
# looks its index up in TASKS when it starts, so tasks appended or inserted elsewhere cannot shift it. 16 CPUs, 16 torch
# threads. Measured locally relative to a full RCC iteration: imagine_* ~1.0x once its real-batch forward runs without a
# graph, teach_* ~1.0x; peak RSS ~5 GB. Keep (running jobs x 16) under the qos's 350-CPU cap with the % limit:
#   sbatch --array=0-23%16 runs_cpu.sbatch "imagine_gen_s0 imagine_gen_s1 ..."
# Each task logs to logs/cgw-cpu-<name>-<jobid>_<index>.out. Nodes elscn-192/193 (another CPU, not timed) excluded.
#SBATCH --job-name=cgw-cpu
#SBATCH --partition=elscn.q
#SBATCH --account=kadmon-lab
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=14:00:00
#SBATCH --exclude=elscn-[23-27,29-35,37,39-40,54,56-57],elscn-192,elscn-193
#SBATCH --output=logs/%x-%A_%a.out
cd "$HOME/cgw"
read -ra N <<< "$1"
name=${N[$SLURM_ARRAY_TASK_ID]}
source "$HOME/miniforge3/bin/activate" cgw
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
       TQDM_MININTERVAL=60
T=main/REPLICATION/trainings
t=$(python -c "import sys; sys.argv = ['x']; sys.path.insert(0, '$T'); import train_cluster_runs as m; \
print([x[0] for x in m.TASKS].index('$name'))") || { echo "task $name not in TASKS"; exit 1; }
echo "task $t = $name on $(hostname): $(lscpu | grep 'Model name' | head -1), $SLURM_CPUS_PER_TASK CPUs"
python $T/train_cluster_runs.py --task "$t" --budget-h 13.5 \
  > "logs/cgw-cpu-$name-${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.out" 2>&1
