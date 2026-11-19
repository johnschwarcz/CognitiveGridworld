"""John 11-18: "checkpointing every episode up to 10, every 2 up to 100, then every 5 up to 500", then "bring the every
2 episodes to 500" -> an
exact re-run of the training-dynamics pair (seed 20261029, the original settings, so bit-identical to the stored run)
for 505 episodes: the Recorder saves every test episode (every 5) to 500 (schedule dense_until=500), and this wrapper
also saves after the update of training episodes 1-10 and every even episode to 100. Output: training_dynamics/
dense500/<net>_s20261029_ep505/ckpt_ep<E>_net.pth. Usage: rerun_dense500.py <trained|echo> <gpu>"""
import os, sys
ROOT = "/home/johns/Desktop/CognitiveGridworld"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "trainings")]
import torch
import train_training_dynamics as T
T.OUT = os.path.join(T.OUT, "dense500")
_sched = T.schedule
T.schedule = lambda episodes, dense_until=0: _sched(episodes, dense_until=500)
EXTRA = set(range(1, 11)) | set(range(2, 501, 2))           # John 11-18: every 2 episodes to 500
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
T.main(sys.argv[1], int(sys.argv[2]), 505, 20261029)
