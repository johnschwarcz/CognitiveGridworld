"""(Moved 11-01 from this session's scratch explore_grids/fwd_grids.py into the repo; data now in
main/DATA/experimental/training_dynamics/summary_data/grids/.) Clean per-environment belief grids: the probe set's 50 environments x 100 cells x S fresh streams, replayed through
the trained RNN and the ESN at selected checkpoints (td_analysis's fp32 LSTM step; verified against the stored probe
logq), with Bayes and naive Bayes on the same observations.

Output (summary_data/grids): fwd_episodes.npz  (env, cell, r, obs; m_bayes / m_naive (n, T, 2) posterior means, a_bayes / a_naive
                                     (n, 2) argmax at T)
                  fwd_<run>.npz     (ep (K,) raw episodes, tr_ep; m (K, n, T, 2) float16 posterior means; a (K, n, 2)
                                     argmax at T; lqT (K, n, 2, 10) float16 log beliefs at T)
Run (GPU 0 only):
  CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_grids_fwd.py episodes
  CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_grids_fwd.py check trained 800
  CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_grids_fwd.py run trained"""
import os
import sys
import time
import numpy as np
import torch

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION")]
import td_analysis as td                                                          # noqa: E402

HERE = os.path.join(td.TD, "summary_data", "grids")                                 # the data, not this file
S = 24                                     # streams per cell
SEED = 20261111
T, R, NO = 30, 10, 5
EPS = (0, 75, 150, 300, 500, 600, 700, 800, 900, 1000, 1100, 1300, 1500, 2000, 3000, 5000, 10000, 20000, 49995)
BATCH = 30000
V = np.arange(R, dtype=np.float64)


def _lse(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def episodes():
    P = np.load(os.path.join(td.OUT, "probes.npz"))
    L, nE = P["L"], len(P["L"])
    env = np.repeat(np.arange(nE), R * R * S)
    cell = np.tile(np.repeat(np.arange(R * R), S), nE)
    r = np.stack([cell // R, cell % R], 1)
    rng = np.random.default_rng(SEED)
    obs = rng.random((len(env), T, NO)) < L[env, :, r[:, 0], r[:, 1]][:, None, :]
    m = {"bayes": np.zeros((len(env), T, 2), np.float32), "naive": np.zeros((len(env), T, 2), np.float32)}
    a = {"bayes": np.zeros((len(env), 2), np.int8), "naive": np.zeros((len(env), 2), np.int8)}
    for e in range(nE):
        ix = np.flatnonzero(env == e)
        o = obs[ix].astype(np.float64)
        lL, l1L = np.log(L[e]), np.log1p(-L[e])                                     # (5, R, R)
        ll = np.einsum("nti,ijk->ntjk", o, lL) + np.einsum("nti,ijk->ntjk", 1 - o, l1L)
        lam = np.cumsum(ll, 1)
        logpost = lam - _lse(lam.reshape(len(ix), T, -1), -1)[..., None, None]
        ljb = np.stack([_lse(logpost, 3), _lse(logpost, 2)], 2)                      # (n, T, 2, R)
        Lb = np.stack([L[e].mean(2), L[e].mean(1)], 1)                               # (5, 2, R)
        lpx = np.einsum("nti,ikr->ntkr", o, np.log(Lb)) + np.einsum("nti,ikr->ntkr", 1 - o, np.log1p(-Lb))
        cl = np.cumsum(lpx, 1)
        lnb = cl - _lse(cl, -1)[..., None]
        for k, lb in (("bayes", ljb), ("naive", lnb)):
            m[k][ix] = np.exp(lb) @ V
            a[k][ix] = lb[:, -1].argmax(-1)
    out = dict(env=env, cell=cell, r=r, obs=obs, Z=P["Z"], **{f"m_{k}": v for k, v in m.items()},
               **{f"a_{k}": v for k, v in a.items()})
    np.savez(os.path.join(HERE, "fwd_episodes.npz"), **out)
    acc = {k: (a[k] == r).mean() for k in a}
    print(f"{len(env)} episodes; accuracy at T (both variables): {acc}")


@torch.no_grad()
def replay(m, obs, Z, dev):
    """log marginal beliefs (n, T, 2, R) float32 for observations obs (n, T, 5) and context inputs Z (n, 10)"""
    W_ih, W_hh, b = td.ga.lstm_weights(m)
    out = []
    for s in range(0, len(obs), BATCH):
        o = torch.as_tensor(obs[s:s + BATCH], dtype=torch.float32, device=dev)
        zin = torch.as_tensor(Z[s:s + BATCH], dtype=torch.float32, device=dev)
        h, c = td._start(m, len(o))
        lg = []
        for t in range(T):
            x = td._readin(m, o[:, t], zin)
            h, c, _ = td._cell(x @ W_ih.T + b + h @ W_hh.T, c)
            lg.append(m.classifier_readout(h))
        cum = torch.stack(lg, 1).double().cumsum(1)
        out.append(td.marginals(cum, m.joint_out).float().cpu().numpy())
    return np.concatenate(out)


def check(run, ep):
    dev = torch.device("cuda:0")
    P = np.load(os.path.join(td.OUT, "probes.npz"))
    m = td.build(run, ep, dev)
    td.load_weights(m, run, ep)
    n = 2000
    lq = replay(m, P["obs"][:n], P["Z"][P["env"][:n]].reshape(n, -1), dev)
    st = np.load(os.path.join(td.OUT, run, f"ck_{ep:05d}.npz"))["logq"][:n].astype(np.float32)
    print(f"{run} {ep}: max |logq - stored| = {np.abs(lq - st).max():.4f} (float16 storage), "
          f"argmax agreement {np.mean(lq[:, -1].argmax(-1) == st[:, -1].argmax(-1)):.4f}")


def run(name):
    dev = torch.device("cuda:0")
    E = np.load(os.path.join(HERE, "fwd_episodes.npz"))
    n = len(E["env"])
    zin = E["Z"][E["env"]].reshape(n, -1)
    M = np.zeros((len(EPS), n, T, 2), np.float16)
    A = np.zeros((len(EPS), n, 2), np.int8)
    LQ = np.zeros((len(EPS), n, 2, R), np.float16)
    m = td.build(name, EPS[0], dev)
    for k, ep in enumerate(EPS):
        t0 = time.time()
        td.load_weights(m, name, ep)
        lq = replay(m, E["obs"], zin, dev)
        q = np.exp(lq - lq.max(-1, keepdims=True))
        M[k] = ((q @ V) / q.sum(-1)).astype(np.float16)
        A[k] = lq[:, -1].argmax(-1)
        LQ[k] = lq[:, -1].astype(np.float16)
        print(f"{name} {ep:5d}: accuracy at T {np.mean(A[k] == E['r']):.3f}  ({time.time() - t0:.1f} s)", flush=True)
    eps = np.array(EPS)
    np.savez(os.path.join(HERE, f"fwd_{name}.npz"), ep=eps, tr_ep=eps - eps // 5, m=M, a=A, lqT=LQ)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "episodes":
        episodes()
    elif cmd == "check":
        check(sys.argv[2], int(sys.argv[3]))
    elif cmd == "run":
        for nm in sys.argv[2:]:
            run(nm)
