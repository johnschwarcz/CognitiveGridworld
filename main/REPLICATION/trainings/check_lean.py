"""The lean path of train_training_dynamics.py against the ordinary one (John 2026-11-04: only optimizations that keep a fair
comparison). Each case trains the same network from the same seed twice, in separate processes, ordinary and lean, and asserts
that every test's accuracy at every step, Bayes's accuracies, the final weights and the final Adam and scaler state are
bit-identical; that every lean training episode has cleared what it no longer computes; and that a lean run leaves what the
figures read (ckpt_ep00000_net.pth, DONE, run.json's lean flag). Cases: marginal output; goal and joint loss; whole batches (trained LSTM, frozen
LSTM) and micro-batches (even and uneven slices, frozen LSTM). In every micro-batch joint-loss run the target of each slice is
checked against that slice's Bayes goal marginal: a slicing bug in shared code hits both arms, so the comparison alone cannot
see it. Other bugs in code both paths share (gpu_belief, forward_compute, forward_pass_micro) are likewise invisible here.
Small batches, a few minutes, no overflow steps. Rerun after any change to Lean or LeanRecorder.
    MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=<gpu> /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/trainings/check_lean.py"""
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
EPISODES, SEED, BATCH = 47, 20261029, 512
CASES = (("echo_N200", {}), ("joint_trained_N100", {}), ("joint_trained_N100_jl", {}),
         ("joint_echo_N500", dict(hid_dim=200)), ("joint_echo_N500_jl", dict(hid_dim=200)),
         ("joint_echo_N500", dict(hid_dim=200, micro_batch=128)), ("joint_echo_N500_jl", dict(hid_dim=200, micro_batch=128)),
         ("joint_echo_N500", dict(hid_dim=200, micro_batch=100)), ("joint_echo_N500_jl", dict(hid_dim=200, micro_batch=100)),
         ("joint_echo_N500", dict(hid_dim=200, micro_batch=128, reservoir_dedup=True, reservoir_fp16=True)),
         ("joint_echo_N500_jl", dict(hid_dim=200, micro_batch=128, reservoir_dedup=True, reservoir_fp16=True)))


def load():
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    spec = importlib.util.spec_from_file_location("ttd", os.path.join(ROOT, "main/REPLICATION/trainings/train_training_dynamics.py"))
    ttd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ttd)
    return ttd


def worker(out, net, overrides):
    ttd = load()
    ttd.OUT = out
    ttd.NETS[net] = dict(ttd.NETS[net], **eval(overrides))
    seen = {}
    if os.environ["TD_LEAN"] == "1":
        original = ttd.Lean.forward_backward

        def check(self):
            if not self.test_set:
                seen["cleared"] = True
                assert all(getattr(self, k) is None for k in self.UNSET), "a training episode kept a stale value"
                assert self.joint_loss or self.model.joint_goal_belief.is_cuda, "the goal target left the GPU"
            return original(self)
        ttd.Lean.forward_backward = check
    elif net.endswith("_jl") and eval(overrides).get("micro_batch"):
        loss = ttd._joint_loss

        def aligned(self, eps=1e-8, power=0.5):
            s, n = self._jl_pos, self._jl_logq.shape[0]
            P = self._jl_logp[s:s + n].exp().double().view(n, self.step_num, 10, 10)
            jb = self.joint_goal_belief.double()
            err = torch.minimum((jb - P.sum(3)).abs().amax((1, 2)), (jb - P.sum(2)).abs().amax((1, 2))).max().item()
            assert err < 1e-5, f"a slice's joint target does not match its episodes: {err}"
            seen["aligned"] = True
            loss(self)
        ttd._joint_loss = aligned
    ttd.main(net, 0, episodes=EPISODES, seed=SEED)
    assert seen or os.environ["TD_LEAN"] == "0", "no lean training episode ran"
    assert not (net.endswith("_jl") and eval(overrides).get("micro_batch") and os.environ["TD_LEAN"] == "0") or seen, "no slice was checked"


def results(folder):
    z = [np.load(f) for f in sorted(glob.glob(os.path.join(folder, "test_*.npz")))]
    tests = {k: np.concatenate([q[k] for q in z]) for k in ("episode", "acc", "joint_acc", "naive_acc")}
    final = torch.load(os.path.join(folder, "final_net.pth"), map_location="cpu", weights_only=False)["weights"]
    return tests, final, torch.load(os.path.join(folder, f"adam_ep{EPISODES:05d}.pth"), map_location="cpu", weights_only=False)


def same_state(a, b):
    """nested optimizer and scaler state, exactly"""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return torch.equal(a, b) if torch.is_tensor(a) else a == b


def main():
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for net, overrides in CASES:
            got = {}
            for lean in ("0", "1"):
                out = os.path.join(tmp, net, lean)
                env = dict(os.environ, TD_BATCH=str(BATCH), TD_LEAN=lean)
                r = subprocess.run([sys.executable, __file__, "worker", out, net, repr(overrides)], env=env, capture_output=True, text=True)
                if r.returncode:
                    print(r.stderr[-2000:])
                    raise SystemExit(f"{net} lean={lean} crashed")
                got[lean] = results(os.path.join(out, f"{net}_s{SEED}_ep{EPISODES}"))
            t0, w0, o0 = got["0"]
            t1, w1, o1 = got["1"]
            same_tests = all(np.array_equal(t0[k], t1[k]) for k in t0)
            same_weights = w0.keys() == w1.keys() and all(torch.equal(w0[k], w1[k]) for k in w0)
            same_adam = same_state(o0, o1)
            run = os.path.join(tmp, net, "1", f"{net}_s{SEED}_ep{EPISODES}")
            files = (os.path.exists(os.path.join(run, "ckpt_ep00000_net.pth")) and json.load(open(os.path.join(run, "run.json")))["lean"]
                     and open(os.path.join(run, "progress.log")).read().rstrip().endswith("DONE"))
            print(f"{net:24s} {overrides or ''} tests {len(t0['episode'])} identical {same_tests}, weights {same_weights}, Adam {same_adam}, files {files}")
            bad += not (same_tests and same_weights and same_adam and files)
            shutil.rmtree(os.path.join(tmp, net))
    print("ALL IDENTICAL" if not bad else f"{bad} CASES DIFFER")
    raise SystemExit(bool(bad))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(*sys.argv[2:5])
    else:
        main()
