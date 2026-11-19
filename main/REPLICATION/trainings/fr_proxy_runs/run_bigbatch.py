"""The training-dynamics pair (seed 20261029) at a large batch and a chosen learning rate for the first <episodes> episodes,
checkpoints at every episode 0-10 and every test episode (5) to 100 (John 11-18: batch 80,000 at lr 1e-3, then "can we
try lr .1 at batch 100K for 100 episodes?"). train_training_dynamics.main with TD_BATCH, TD_MICRO (slices, gradients
accumulated), TD_GPU_BAYES, and classifier_LR overridden through NETS[net] (cfg ends with **NETS[net]).
Usage: run_bigbatch.py <trained|echo> <gpu> <episodes> <subfolder> <batch> <micro> <lr>"""
import os, sys
ROOT = "/home/johns/Desktop/CognitiveGridworld"
os.chdir(ROOT)
net, gpu, episodes, sub, batch, micro, lr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], \
    sys.argv[6], float(sys.argv[7])
os.environ.update(TD_BATCH=batch, TD_MICRO=micro, TD_GPU_BAYES="1")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "trainings")]
import torch
import train_training_dynamics as T
T.OUT = os.path.join(T.OUT, sub)
T.NETS[net] = dict(T.NETS[net], classifier_LR=lr)
_sched = T.schedule
T.schedule = lambda episodes, dense_until=0: _sched(episodes, dense_until=100)
EXTRA = set(range(1, 11))
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
T.main(net, gpu, episodes, 20261029)
