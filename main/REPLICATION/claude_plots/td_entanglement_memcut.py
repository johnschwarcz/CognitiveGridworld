"""Per-particle-step memory use, every checkpoint (John 11-01: the entanglement figure at 20 percentiles): the update
error weighted by Bayes's belief (the network's step-t increment applied to Bayes's belief at t-1, Jeffreys/2 against
Bayes's belief at t, both variables averaged) of the intact network and of the network with its memory cut to the last
step (td_entanglement_driver.py's rule: the cell rerun from its start state over input t only), stored per (probe,
step) so any binning is free later.
-> data main/DATA/experimental/training_dynamics/analysis/ent_memcut_<run>.npz: episode (K,), err_int, err_cut1
   (K, n, T) float16 (step 1: the cut equals the intact run); with "var" first (audit 11-01: pair each variable's error
   with its own entanglement, no average over the variables before the split): ent_memcut_var_<run>.npz, (K, n, T, 2)
Run from the repository root (GPU 0):
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_entanglement_memcut.py \\
        trained echo"""
import os
import sys
import time
import numpy as np
import torch

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_entanglement_driver as dr                                               # noqa: E402

td = dr.td
CHUNK = 2000


@torch.no_grad()
def errors(p, P, dev, per_var=False):
    n, T = len(P["goal"]), td.T
    shape = (n, T, td.C) if per_var else (n, T)
    e_int, e_cut = np.zeros(shape, np.float16), np.zeros(shape, np.float16)
    for c0 in range(0, n, CHUNK):
        sl = slice(c0, c0 + CHUNK)
        obs = torch.as_tensor(P["obs"][sl], dtype=torch.float32, device=dev)
        zin = torch.as_tensor(P["Z"][P["env"][sl]], device=dev).reshape(len(obs), -1)
        lj = torch.as_tensor(P["log_jb"][sl], device=dev).double()
        prior = torch.cat([torch.full_like(lj[:, :1], -np.log(td.R)), lj[:, :-1]], 1)
        m = len(obs)
        xp = dr._pre(p, obs, zin)
        u = dr._run(p, xp, *dr._start(p, m))
        u1 = dr._run(p, xp.reshape(m * T, 1, -1), *dr._start(p, m * T))[:, 0].reshape(m, T, -1)
        for out, uu in ((e_int, u), (e_cut, u1)):
            d = dr._readout(p, uu).double().reshape(m, T, td.C, td.R)
            e = dr._jeff(prior + d, lj)                                              # (m, T, 2) per variable
            out[sl] = (e if per_var else e.mean(-1)).cpu().numpy()
    return e_int, e_cut


def run(name, dev=torch.device("cuda:0"), per_var=False):
    path = os.path.join(dr.AN, f"ent_memcut_{'var_' if per_var else ''}{name}.npz")
    P = td.probes()
    eps = td.checkpoints(name)
    n, T = len(P["goal"]), td.T
    E_int = np.zeros((len(eps), n, T) + ((td.C,) if per_var else ()), np.float16)
    E_cut = np.zeros_like(E_int)
    t0 = time.time()
    for i, ep in enumerate(eps):
        w = torch.load(td.ckpt_path(name, ep), map_location="cpu", weights_only=False)["weights"]
        p = {x: v.to(dev).float() for x, v in w.items()}
        E_int[i], E_cut[i] = errors(p, P, dev, per_var)
        if i % 10 == 0:
            print(f"{time.strftime('%H:%M:%S')} {name} ep {ep} ({time.time() - t0:.0f} s): intact {E_int[i, :, 1:].mean():.4f}"
                  f", cut {E_cut[i, :, 1:].astype(np.float64).mean():.4f}", flush=True)
    np.savez(path, episode=np.array(eps), err_int=E_int, err_cut1=E_cut)
    print(f"saved {path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    args = sys.argv[1:]
    pv = bool(args) and args[0] == "var"
    for nm in args[1:] if pv else args:
        run(nm, per_var=pv)
