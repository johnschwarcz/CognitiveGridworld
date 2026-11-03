"""Variance explained by inference time WITHIN episodes, through training (John 11-01, audit of averaging order: "take
the variance across timesteps, and then average over episodes"). td_analysis's pool_* "step share" is the variance of
the probe-averaged trajectory across steps (average over episodes first): each episode's own movement over steps
cancels, so x (i.i.d. observations) scores 0 at every checkpoint. This replaces it.
-> data main/DATA/experimental/training_dynamics/analysis/time_within_<run>.npz (every checkpoint; GPU 0, ~3 s each)

For each signal s (rows as td_analysis pool_x / pool_u / pool_c: x, W_ih x and its gate blocks i, f, g, o; u, W_hh u_{t-1}
and its gate blocks (the drives of steps 2-30, from u_1..u_29); c), over the 10,000 probes i and their steps t (1-30;
the drives 2-30):
  C_pool   = < (s_it - m)(s_it - m)^T >                         all particle-steps pooled (m: their mean)
  C_within = < (s_it - sbar_i)(s_it - sbar_i)^T >               sbar_i = episode i's average over its steps
           = C_pool - Cov_i(sbar_i)
seen through each Gram G = W^T W of the drive (None: the signal itself):
  "share"  tr(C_within G) / tr(C_pool G)       the share of the variance that is movement over steps within episodes
  "pr"     tr(C_within G)^2 / tr((C_within G)^2)   the dimension that movement spans
  "step_pr" the mean over steps of the per-step PR (td_analysis fac, total factor): the cloud of particles at one step,
           then averaged (pooling the steps into one cloud adds the drift of the mean across steps as dimensions)
Stored per checkpoint: share, pr (rows) for keys x, u, c (rows = pool_x / pool_u / pool_c of td_analysis). "pr" is
the PR of the within-episode covariance POOLED over all episodes: it also counts how the directions of movement differ
between episodes (and the shared time course); verification 11-01: Trained u at the end 38 vs 5.7 per episode.
The per-episode version (episode mode -> time_episode_pr_<run>.npz): for each episode, the PR of its own centred
trajectory over its steps (eigenvalues of the steps x steps Gram; at most 29), then averaged over the episodes (the
first 2,000 permuted probes: all 50 environments), for x, W_ih x, u, W_hh u_{t-1} (whole drives) and c.
Run from the repository root (GPU 0):
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_time_within.py \\
        trained echo        (add "episode" first for the per-episode PR)"""
import os
import sys
import time
import glob
import numpy as np
import torch

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_analysis as td                                                          # noqa: E402
import gradflow_analysis as ga                                                    # noqa: E402

FAC_ROW = {"x": [0, 1, 2, 3, 4, 5], "u": [6, 7, 8, 9, 10, 11], "c": [12]}          # td_analysis FAC_ROWS per pool row


def _stats(S1, S2, n, Ts, grams):
    """[share, pr] per Gram from per-episode sums S1 (n, D) and the pooled sum of outer products S2 (D, D) (both of the
    signal shifted by a fixed reference: covariances do not change)"""
    m = S1.sum(0) / (n * Ts)
    Cpool = S2 / (n * Ts) - torch.outer(m, m)
    sbc = (S1 / Ts - m).float()
    Cbar = (sbc.T @ sbc).double() / n
    W = Cpool - Cbar
    out = []
    for G in grams:
        A, B = (W.float(), Cpool.float()) if G is None else (W.float() @ G, Cpool.float() @ G)
        tA, tB = A.diagonal().sum().item(), B.diagonal().sum().item()
        out.append([tA / tB, tA ** 2 / (A * A.T).sum().item()])
    return np.array(out)


@torch.no_grad()
def checkpoint(m, P, dev):
    n = len(P["obs"])
    obs = torch.as_tensor(P["obs"], dtype=torch.float32, device=dev)
    zin = torch.as_tensor(P["Z"][P["env"]], device=dev).reshape(n, -1)
    W_ih, W_hh, b = ga.lstm_weights(m)
    H = W_hh.shape[1]
    G_ih, G_hh = td._grams(W_ih, H), td._grams(W_hh, H)
    h, c = td._start(m, n)
    acc = {}
    for t in range(td.T):
        x = td._readin(m, obs[:, t], zin)
        h_new, c_new, _ = td._cell(x @ W_ih.T + b + h @ W_hh.T, c)
        for s, X in (("x", x), ("u", h_new), ("c", c_new)) + ((("u_prev", h),) if t >= 1 else ()):
            if s not in acc:                                       # [sum per episode, sum of outer products, steps, ref]
                acc[s] = [torch.zeros(X.shape, dtype=torch.float64, device=dev),
                          torch.zeros(X.shape[1], X.shape[1], dtype=torch.float64, device=dev), 0, X.mean(0)]
            Xs = X - acc[s][3]                                     # fp32 products, fp64 sums
            acc[s][0] += Xs.double()
            acc[s][1] += (Xs.T @ Xs).double()
            acc[s][2] += 1
        h, c = h_new, c_new
    st = {k: _stats(v[0], v[1], n, v[2], g) for k, g, v in (("x", G_ih, acc["x"]), ("u", [None], acc["u"]),
                                                            ("u_prev", G_hh[1:], acc["u_prev"]), ("c", [None], acc["c"]))}
    return {"x": st["x"], "u": np.concatenate([st["u"], st["u_prev"]]), "c": st["c"]}


EP_CHUNK, N_EP = 1000, 2000                                    # per-episode PR on the first 2,000 permuted probes


@torch.no_grad()
def episode_pr(m, P, dev):
    """the mean over episodes of the PR of each episode's own movement over steps (its centred trajectory)"""
    rows = P["perm"][:N_EP]
    n = len(rows)
    W_ih, W_hh, b = ga.lstm_weights(m)
    out = {k: 0. for k in ("x", "Wih_x", "u", "Whh_u", "c")}
    for c0 in range(0, n, EP_CHUNK):
        rb = rows[c0:c0 + EP_CHUNK]
        obs = torch.as_tensor(P["obs"][rb], dtype=torch.float32, device=dev)
        zin = torch.as_tensor(P["Z"][P["env"][rb]], device=dev).reshape(len(obs), -1)
        h, c = td._start(m, len(obs))
        X, U, Cs = [], [], []
        for t in range(td.T):
            x = td._readin(m, obs[:, t], zin)
            h, c, _ = td._cell(x @ W_ih.T + b + h @ W_hh.T, c)
            X.append(x)
            U.append(h)
            Cs.append(c)
        X, U, Cs = torch.stack(X, 1), torch.stack(U, 1), torch.stack(Cs, 1)            # (B, T, D)
        for k, S in (("x", X), ("Wih_x", X @ W_ih.T), ("u", U), ("Whh_u", U[:, :-1] @ W_hh.T), ("c", Cs)):
            S = S - S.mean(1, keepdim=True)                                          # the episode's own movement
            ev = torch.linalg.eigvalsh((S @ S.transpose(1, 2)).double()).clamp_min(0)
            out[k] += (ev.sum(-1) ** 2 / (ev ** 2).sum(-1).clamp_min(1e-300)).sum().item()
        del X, U, Cs
    return {k: v / n for k, v in out.items()}


def run_episode_pr(name, dev=torch.device("cuda:0")):
    path = os.path.join(td.OUT, f"time_episode_pr_{name}.npz")
    P = td.probes()
    files = sorted(glob.glob(os.path.join(td.OUT, name, "ck_?????.npz")))
    eps = [int(os.path.basename(f)[3:8]) for f in files]
    m = td.build(name, eps[0], dev)
    res = {k: [] for k in ("x", "Wih_x", "u", "Whh_u", "c")}
    t0 = time.time()
    for i, ep in enumerate(eps):
        td.load_weights(m, name, ep)
        o = episode_pr(m, P, dev)
        for k in res:
            res[k].append(o[k])
        if i % 20 == 0:
            print(f"{time.strftime('%H:%M:%S')} {name} ep {ep} ({time.time() - t0:.0f} s): per-episode PR u {o['u']:.2f}, "
                  f"c {o['c']:.2f}, x {o['x']:.2f}", flush=True)
    np.savez(path, episode=np.array(eps), **{k: np.array(v) for k, v in res.items()})
    print(f"saved {path} ({time.time() - t0:.0f} s)")


def run(name, dev=torch.device("cuda:0")):
    path = os.path.join(td.OUT, f"time_within_{name}.npz")
    P = td.probes()
    files = sorted(glob.glob(os.path.join(td.OUT, name, "ck_?????.npz")))
    eps = [int(os.path.basename(f)[3:8]) for f in files]
    m = td.build(name, eps[0], dev)
    res = {k: [] for k in ("x", "u", "c", "step_pr_x", "step_pr_u", "step_pr_c")}
    t0 = time.time()
    for i, (ep, f) in enumerate(zip(eps, files)):
        td.load_weights(m, name, ep)
        o = checkpoint(m, P, dev)
        fac = np.load(f)["fac"]                                                      # (T, rows, factors, [var, PR])
        for k in ("x", "u", "c"):
            res[k].append(o[k])
            res["step_pr_" + k].append(np.nanmean(fac[:, FAC_ROW[k], 0, 1], 0))
        if i % 20 == 0:
            print(f"{time.strftime('%H:%M:%S')} {name} ep {ep} ({time.time() - t0:.0f} s): within-episode time share "
                  f"u {o['u'][0, 0]:.3f}, c {o['c'][0, 0]:.3f}, x {o['x'][0, 0]:.3f}", flush=True)
    np.savez(path, episode=np.array(eps), **{k: np.array(v) for k, v in res.items()})
    print(f"saved {path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "episode":
        for nm in args[1:]:
            run_episode_pr(nm)
    else:
        for nm in args:
            run(nm)
