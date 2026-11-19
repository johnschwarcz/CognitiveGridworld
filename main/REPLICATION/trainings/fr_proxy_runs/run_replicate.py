"""Replicates of the training-dynamics pair for the early-learning panel (John 11-18: "please start the replicates on the
cluster"): the SAME world (numpy seed 20261029 builds the 500 variables' embeddings, so the probe environments stay
in-distribution), NEW initial weights (torch seed 20261029 + 1000 r) and a NEW episode stream (numpy reseeded with
20261029 + 2000 r when the episode loop starts), 2,005 episodes, checkpoints on the original schedule (every 25 to 500,
every 100 to 2,000; schedule()). Faster than the original: the Bayes targets on the GPU (TD_GPU_BAYES) and the recorder's
analyses off (checkpoints only); the training itself is the original's. Output: training_dynamics/replicate<r>/
<net>_s20261029_ep2005/. Usage: run_replicate.py <trained|echo> <replicate 1..4> [gpu]"""
import os, sys
ROOT = os.path.expanduser("~/cgw")
os.chdir(ROOT)
net, rep = sys.argv[1], int(sys.argv[2])
gpu = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.environ["TD_GPU_BAYES"] = "1"
os.environ["CGW_ROOT"] = ROOT                                  # the script defaults to the local path
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "trainings")]
import numpy as np
import torch
import train_training_dynamics as T
T.OUT = os.path.join(T.OUT, f"replicate{rep}")
_ms = torch.manual_seed
T.torch.manual_seed = lambda s: _ms(s + 1000 * rep)          # new initial weights, same world (numpy)
_loop = T.CognitiveGridworld.episode_loop


def loop(self, *a, **k):
    np.random.seed(20261029 + 2000 * rep)                     # a new episode stream once the world exists
    return _loop(self, *a, **k)


T.CognitiveGridworld.episode_loop = loop
T.Recorder._test = lambda self, env: None                     # analyses off: checkpoints only
T.Recorder._online = lambda self, env: None
print(f"replicate {rep} {net}: torch seed {20261029 + 1000 * rep}, episode stream seed {20261029 + 2000 * rep}",
      flush=True)
T.main(net, gpu, 2005, 20261029)
