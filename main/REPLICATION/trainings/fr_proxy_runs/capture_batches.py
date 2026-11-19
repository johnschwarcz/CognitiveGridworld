"""Recreate the single-batch run's two batches (John 11-18: "only update on 1 batch, but evaluate both"): the same setup as
run_multistep.py ... single20k_k2000 3 20000 10000 (seed 20261029, the same Recorder and schedule, so numpy's global
draws are identical; the network's updates draw nothing from it), stopped at the start of the training episode. Saves the
episode-0 test batch (never trained on: the "second batch") and the episode-1 training batch: obs (B, T, 5), the
network's context input active_Z (B, 5, 2), goal_ind, goal_value, and Bayes's goal accuracy at T on each.
Usage: capture_batches.py <gpu> <out.npz>"""
import os, sys
ROOT = "/home/johns/Desktop/CognitiveGridworld"
os.chdir(ROOT)
gpu, OUTF = int(sys.argv[1]), sys.argv[2]
os.environ.update(TD_BATCH="20000", TD_MICRO="10000", TD_GPU_BAYES="1")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "trainings")]
import numpy as np
import torch
import train_training_dynamics as T
T.OUT = os.path.join("/tmp/claude-1002/-home-johns-Desktop-CognitiveGridworld/f79b4ed6-59c3-4812-b99b-c2001802302a/scratchpad", "capture_run")
Env = T.CognitiveGridworld
_fb = Env.forward_backward
got = {}


def grab(env, tag):
    m = env.model
    with torch.no_grad():
        m.MC_to_interactions()
    got[f"{tag}_obs"] = np.asarray(env.obs_flat).astype(bool)
    got[f"{tag}_Z"] = m.active_Z.detach().float().cpu().numpy()
    got[f"{tag}_goal_ind"] = np.asarray(env.goal_ind)
    got[f"{tag}_goal_value"] = np.asarray(env.goal_value)
    got[f"{tag}_ctx_vals"] = np.asarray(env.ctx_vals)
    got[f"{tag}_joint_acc_T"] = float(np.asarray(env.joint_acc).mean(0)[-1])
    got[f"{tag}_naive_acc_T"] = float(np.asarray(env.naive_acc).mean(0)[-1])


def fb(self):
    if self.test_set:
        out = _fb(self)
        grab(self, "second")                             # episode 0's test batch
        return out
    grab(self, "train")                                  # episode 1's batch, before any update
    np.savez(OUTF, **got)
    print("saved", OUTF, {k: v.shape if hasattr(v, "shape") else v for k, v in got.items()}, flush=True)
    os._exit(0)


Env.forward_backward = fb
T.main("trained", gpu, 3, 20261029)
