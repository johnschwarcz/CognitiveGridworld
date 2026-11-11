"""SUPERSEDED by td_tt_dose.py (John 11-09: read the effect at the manipulated step; one step per episode).
Inference test of the late activity-norm lock (John 11-09: causal or consequence?). At a checkpoint, replay the
10,000 probes; at steps 2-30, in the most entangled fifth of each step (rev = mean of the two variables' E, within-step
ranks) rescale u_t to the step's median norm (removing its extra norm), and continue the recurrence from the rescaled
state. Control: the SAME scale factors applied to a random fifth of each step (matched perturbation sizes). Outcome:
the belief error KL(O_t || q_t), per step at the scaled items and at T; also the scale factors' size.
Run from the repository root (GPU 0, ~1 min):
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_clamp_norm.py trained 49995"""
import os, sys, time
import numpy as np
import torch
ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, ROOT + "/main/REPLICATION", ROOT + "/main/REPLICATION/claude_plots"]
import td_analysis as td
import td_tt_order as to


def replay(m, P, dev, sel=None, fac=None):
    """sel (n, T) bool: items to rescale; fac (n, T): their scale factors (None: the median-norm clamp is computed)"""
    n = len(P["obs"])
    obs = torch.as_tensor(P["obs"], dtype=torch.float32, device=dev)
    zin = torch.as_tensor(P["Z"][P["env"]], device=dev).reshape(n, -1)
    W_ih, W_hh, b = td.ga.lstm_weights(m)
    h, c = td._start(m, n)
    lg, used = [], np.ones((n, td.T))
    with torch.no_grad():
        for t in range(td.T):
            x = td._readin(m, obs[:, t], zin)
            h, c, _ = td._cell(x @ W_ih.T + b + h @ W_hh.T, c)
            if sel is not None and t >= 1:
                s = torch.as_tensor(sel[:, t], device=dev)
                nrm = h.norm(dim=-1)
                if fac is None:
                    f = (nrm.median() / nrm)
                else:
                    f = torch.as_tensor(fac[:, t], dtype=h.dtype, device=dev)
                f = torch.where(s, f, torch.ones_like(f))
                used[:, t] = f.double().cpu().numpy()
                h = h * f[:, None]
            lg.append(m.classifier_readout(h))
        cum = torch.stack(lg, 1).double().cumsum(1)
        lq = td.marginals(cum, m.joint_out).double().cpu().numpy()
    return to.lsm(lq), used


def main(run="trained", ep=49995):
    dev = torch.device("cuda:0")
    P = np.load(os.path.join(td.OUT, "probes.npz"))
    lB = to.lsm(P["log_jb"].astype(np.float64))
    rev = P["ent"].astype(np.float64).mean(-1)                                   # (n, T)
    n = len(rev)
    rng = np.random.default_rng(0)
    top = np.zeros((n, td.T), bool)
    rnd = np.zeros((n, td.T), bool)
    for t in range(1, td.T):
        top[:, t] = rev[:, t] >= np.quantile(rev[:, t], .8)
        rnd[rng.permutation(n)[:top[:, t].sum()], t] = True
    m = td.build(run, ep, dev)
    td.load_weights(m, run, ep)

    def V(lq):
        return (np.exp(lB) * (lB - lq)).sum(-1).mean(-1)                          # (n, T), mean over the 2 variables
    l0, _ = replay(m, P, dev)
    l1, f1 = replay(m, P, dev, top)
    fac = np.ones((n, td.T))
    for t in range(1, td.T):                                                      # the entangled items' factors, shuffled
        fac[rnd[:, t], t] = rng.permutation(f1[top[:, t], t])
    l2, f2 = replay(m, P, dev, rnd, fac)
    v0, v1, v2 = V(l0), V(l1), V(l2)
    d0, d1, d2 = (np.diff(np.concatenate([np.zeros((n, 1)), v], 1), axis=1) for v in (v0, v1, v2))
    print(f"{run} raw {ep}: scale factors at the entangled fifth: median {np.median(f1[top]):.3f}, "
          f"mean |log f| {np.abs(np.log(f1[top])).mean():.3f}; random fifth (same factors) {np.abs(np.log(f2[rnd])).mean():.3f}")
    print(f"  per-step loss at the scaled steps: intact {d0[top].mean():.4f} -> clamped {d1[top].mean():.4f} "
          f"(entangled fifth) | intact {d0[rnd].mean():.4f} -> scaled {d2[rnd].mean():.4f} (random fifth)")
    print(f"  error at T, KL(O_T||q_T): intact {v0[:, -1].mean():.4f}; entangled-fifth clamp {v1[:, -1].mean():.4f}; "
          f"random-fifth, same factors {v2[:, -1].mean():.4f}")
    goal, truth = P["goal"], P["r"]
    ar = np.arange(n)
    for nm, l in (("intact", l0), ("entangled clamp", l1), ("random, same factors", l2)):
        print(f"  goal accuracy at T {nm}: {(l[ar, -1, goal].argmax(-1) == truth[ar, goal]).mean():.4f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "trained", int(a[1]) if len(a) > 1 else 49995)
