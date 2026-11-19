"""Extension of a replicate of the training-dynamics pair from 2,005 to 3,005 episodes (John 11-18: "how long would it
take to train the 9 replicates until episode 3K?" -> "yes please go ahead"): the replicate's weights after its last
episode (final_net.pth) and its Adam + AMP-scaler state (adam_ep02005.pth) are loaded before the first episode, the
episode stream is reseeded (20261029 + 2000 r + 1: new episodes), and N = 1,005 more episodes run (a test every 5th;
loop episode e = true episode - 2,005, so the test pattern is the original's). The checkpoints at the true episodes
2,100-3,000 (every 100; loop e = 95, 195, ..., 995) are renamed to their true episodes. One update fewer than the
original run at the same episode (the replicate's forced last test at 2,004). Output: training_dynamics/replicate<r>/
<net>_s20261029_ep<N>/. Usage: run_extend.py <trained|echo> <replicate> [gpu] [N=1005]"""
import os, sys
_home = os.path.expanduser("~/cgw")
ROOT = os.environ.get("CGW_ROOT") or (_home if os.path.isdir(_home) else "/home/johns/Desktop/CognitiveGridworld")
os.chdir(ROOT)
net, rep = sys.argv[1], int(sys.argv[2])
gpu = int(sys.argv[3]) if len(sys.argv) > 3 else 0
N = int(sys.argv[4]) if len(sys.argv) > 4 else 1005
OFF = 2005
os.environ["TD_GPU_BAYES"] = "1"
os.environ["CGW_ROOT"] = ROOT
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "trainings")]
import numpy as np
import torch
import train_training_dynamics as T
T.OUT = os.path.join(T.OUT, f"replicate{rep}")
SRC = os.path.join(T.OUT, f"{net}_s20261029_ep2005")
_ms = torch.manual_seed
T.torch.manual_seed = lambda s: _ms(s + 1000 * rep)          # the replicate's construction (weights overwritten)
SCHED = [e - OFF for e in range(2100, OFF + N, 100)]          # loop episodes of the true 2,100, 2,200, ...
T.schedule = lambda episodes, dense_until=0: SCHED
_loop = T.CognitiveGridworld.episode_loop


def loop(self, *a, **k):
    m = self.model
    W = torch.load(os.path.join(SRC, "final_net.pth"), map_location="cpu", weights_only=False)["weights"]
    m.load_state_dict(W)                                       # in place: the optimizer keeps its parameters
    A = torch.load(os.path.join(SRC, f"adam_ep{OFF:05d}.pth"), map_location="cpu", weights_only=False)
    assert A["episode"] == OFF
    m.classifier_optim.load_state_dict(A["optim"])
    m.scaler.load_state_dict(A["scaler"])
    sd = m.state_dict()
    d = max((sd[k].float().cpu() - W[k].float()).abs().max().item() for k in W)
    print(f"replicate {rep} {net}: weights and Adam state of episode {OFF} loaded (max abs diff {d:.1e})", flush=True)
    assert d == 0
    np.random.seed(20261029 + 2000 * rep + 1)                 # a new episode stream for the extension
    return _loop(self, *a, **k)


T.CognitiveGridworld.episode_loop = loop
T.Recorder._test = lambda self, env: None                     # analyses off: checkpoints only
T.Recorder._online = lambda self, env: None
T.main(net, gpu, N, 20261029)
out = os.path.join(T.OUT, f"{net}_s20261029_ep{N}")
for e in SCHED:                                                # loop episode -> true episode
    os.replace(os.path.join(out, f"ckpt_ep{e:05d}_net.pth"), os.path.join(out, f"ckpt_ep{e + OFF:05d}_net.pth"))
print(f"renamed {len(SCHED)} checkpoints to true episodes", flush=True)
