"""Per-step belief updates of all four agents on all 50,000 episodes of the trained network's gradflow harvest
(25 Oct 2026) -> main/DATA/experimental/fig1c_compression/figdata/round4b/updates_50k.npz

Beliefs (goal-ordered [goal | contextual], 20 numbers): the Optimal and the Naive from the harvest (ep_joint_belief,
ep_naive_belief); the Trained RNN and the ESN (replayed on the trained network's episodes, CPU) from
fig1_checks.beliefs_alltime for the first 20,000 episodes and by the same replay (fig1._beliefs) for
episodes 20,000-49,999 (check: the replay of episodes 0-999 reproduces beliefs_alltime).
Kept: episodes of distinct-variable environments (as everywhere in fig1c).
Stored per agent and step t = 1..30:
    u_<a>   the particle's update, (E_t[r_g] - E_{t-1}[r_g], E_t[r_c] - E_{t-1}[r_c]), posterior means, E_0 = 4.5
    tv_<a>  the full belief's update per variable: total variation 1/2 sum_r |b_t(r) - b_{t-1}(r)| (the share of
            probability mass that moved), b_0 = uniform
and per episode: rev (house entanglement per step), fr_t (factorization regret per step: FR_g + FR_c, the house
Jeffreys/2 between the Optimal's and the Naive's marginals, clipped at 1e-15), env, rg, rc.
Run from the repository root (about 10 min on CPU, the replays):
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/fig1c_update_cache.py"""
import os
import sys
import time
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
OUT = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression", "figdata", "round4b", "updates_50k.npz")
AG = ("Joint", "Trained", "Naive", "Echo")
N_ALL, N_DONE, T, R = 50000, 20000, 30, 10
EPS = 1e-15


def jeffreys_half(p, q):
    p, q = np.clip(p, EPS, 1 - EPS), np.clip(q, EPS, 1 - EPS)
    return .5 * ((p - q) * (np.log(p) - np.log(q))).sum(-1)


def load():
    return np.load(OUT)


def build():
    import torch
    torch.set_num_threads(6)
    import fig1 as cf
    from fig1_checks import beliefs_alltime
    t0 = time.time()
    keys = list(cf.REPLAY) + ["ep_ctx_inds", "ep_ctx_vals", "ep_goal_ind", "ep_joint_belief", "ep_naive_belief", "rev"]
    f = cf._harvest("trained", keys)
    goal = f["ep_goal_ind"][:N_ALL].astype(int)
    ar = np.arange(N_ALL)
    go = lambda b: np.concatenate([b[ar, :, goal], b[ar, :, 1 - goal]], -1)      # (n, T, 2, R) raw -> goal-ordered
    B = {"Joint": go(f["ep_joint_belief"][:N_ALL]).astype(np.float32), "Naive": go(f["ep_naive_belief"][:N_ALL]).astype(np.float32)}
    BA = beliefs_alltime()
    print(f"harvest and beliefs_alltime loaded in {time.time() - t0:.0f} s", flush=True)
    checks = {}
    for name, net in (("Trained", "trained"), ("Echo", "echo")):
        D0 = {k: f[k][:1000] for k in cf.REPLAY}
        b0 = cf._beliefs(net, D0, 1, check=False, steps=range(T))
        a0 = np.arange(1000)
        b0 = np.concatenate([b0[a0, :, goal[:1000]], b0[a0, :, 1 - goal[:1000]]], -1)
        checks[name] = float(np.abs(b0 - BA[name][:1000]).max())
        print(f"{name}: replay check vs beliefs_alltime, max |diff| {checks[name]:.2e}", flush=True)
        D = {k: f[k][N_DONE:N_ALL] for k in cf.REPLAY}
        b = cf._beliefs(net, D, (N_ALL - N_DONE) // 1000, check=False, steps=range(T))     # (30000, T, 2, R) raw
        ae, ge = np.arange(N_ALL - N_DONE), goal[N_DONE:]
        extra = np.concatenate([b[ae, :, ge], b[ae, :, 1 - ge]], -1).astype(np.float32)
        B[name] = np.concatenate([BA[name], extra], 0)
        print(f"{name}: replayed episodes {N_DONE}-{N_ALL} in {time.time() - t0:.0f} s", flush=True)
    ci = np.sort(f["ep_ctx_inds"][:N_ALL].astype(int), 1)
    keep = ci[:, 0] != ci[:, 1]
    ctx = f["ep_ctx_vals"][:N_ALL].astype(int)
    out = dict(env=np.unique(ci[keep], axis=0, return_inverse=True)[1].ravel(), rg=ctx[ar, goal][keep],
               rc=ctx[ar, 1 - goal][keep], rev=f["rev"][:N_ALL][keep].astype(np.float32), agents=np.array(AG),
               check_trained=checks["Trained"], check_echo=checks["Echo"])
    J, Nv = B["Joint"].astype(np.float64), B["Naive"].astype(np.float64)
    out["fr_t"] = (jeffreys_half(J[..., :R], Nv[..., :R]) + jeffreys_half(J[..., R:], Nv[..., R:]))[keep].astype(np.float32)
    Rv = np.arange(R)
    for a in AG:
        b = B[a][keep].astype(np.float64).reshape(keep.sum(), T, 2, R)
        b /= b.sum(-1, keepdims=True)
        E = np.concatenate([np.full((len(b), 1, 2), (R - 1) / 2), (b * Rv).sum(-1)], 1)
        out["u_" + a] = np.diff(E, axis=1).astype(np.float32)                     # (n, T, 2)
        b_ = np.concatenate([np.full((len(b), 1, 2, R), 1 / R), b], 1)
        out["tv_" + a] = (.5 * np.abs(np.diff(b_, axis=1)).sum(-1)).astype(np.float32)   # (n, T, 2)
    np.savez_compressed(OUT, **out)
    print(f"saved {OUT}: {keep.sum():,} trajectories from {out['env'].max() + 1:,} environments, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    build()
