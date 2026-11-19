"""John 11-18 14:00 ET: the training-dynamics pair (seed 20261029) at batch 80,000 for the first 105 episodes, a checkpoint
at every episode 0-10 and every test episode (5) to 100: train_training_dynamics.main with TD_BATCH=80000, TD_MICRO
(16,000-trajectory slices, gradients accumulated), TD_GPU_BAYES; output training_dynamics/batch80k/<net>_s20261029_ep105.
Usage: run_b80k.py <trained|echo> <gpu> [episodes] [out_subfolder]"""
import os, sys
ROOT = "/home/johns/Desktop/CognitiveGridworld"
os.chdir(ROOT)
os.environ.update(TD_BATCH="80000", TD_MICRO="16000", TD_GPU_BAYES="1")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "trainings")]
import torch
import train_training_dynamics as T
T.OUT = os.path.join(T.OUT, sys.argv[4] if len(sys.argv) > 4 else "batch80k")
_sched = T.schedule
T.schedule = lambda episodes, dense_until=0: _sched(episodes, dense_until=100)
EXTRA = set(range(1, 11))                                # training episodes 1-10 too (after their update)
_call = T.Recorder.__call__


def call(self, env):
    out = _call(self, env)
    if not env.test_set and env.e in EXTRA:
        m = env.model
        save = {"K": env.all_K, "Q": env.all_Q, "model_K": T.tnp(m.all_K, "np"), "model_Q": T.tnp(m.all_Q, "np"),
                "weights": {k: v.detach().cpu() for k, v in m.state_dict().items()}, "episode": env.e}
        path = os.path.join(self.out, f"ckpt_ep{env.e:05d}_net.pth")
        torch.save(save, path + ".tmp")
        os.replace(path + ".tmp", path)
    return out


T.Recorder.__call__ = call
T.main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 105, 20261029)
