"""The gradflow figure's update measures for the alias scan's streams, at every step (27 Oct 2026; John: relate the
dynamics_interference xcorr measures to the jumps / splits, and colour the network's particles or grid by them)
-> main/DATA/experimental/fig1c_compression/figdata/round7_jumps/scan_credit_<net>.npz

Definitions (gradflow_analysis, used here through its own functions: lstm_input, forward_states, backward, StepLoss,
step_error, cut_segments): the step-t error e_t = dL_t/du_t through the read-out, the losses of BOTH variables summed
(the goal's and the other's, each the SANITY loss term against that variable's exact joint marginal), sent back
k = 0 .. min(15, t) steps as the gate-gradient pieces delta_k (4000), with inputs z_k = [x_{t-k}; u_{t-k-1}] (2000):
dW = sum_k delta_k z_k^T. With DD_kl = delta_k . delta_l and ZZ_kl = z_k . z_l:
    coh     credit coherence  sum DD / tr DD = |sum_k delta_k|^2 / sum_k |delta_k|^2
    dW      the update's energy  |dW|^2 = sum DD o ZZ
    dim     its effective dimension  (tr DD ZZ)^2 / tr((DD ZZ)^2)
    band_x  the cross-band share  sum_B (<dW_B, dW> - |dW_B|^2) / |dW|^2, B the 10% bands of u's variance on the
            network's fixed basis (cut_segments(evr, 10)); dW_B from the error's part in band B (one backward probe each)
At steps t < 16 the lags are truncated (k <= t), unlike the figure's steps 17-30: compare values within a step.
Also novelty -sum_i log p(o_ti | o_<t) (p = (ones before t + 1) / (t + 2)) and change |o_t - o_{t-1}|_1.
KMAX (env, default 15): the credit window; KMAX=4 (28 Oct, the fixed-window control for fig1c_two_processes: every step
from 5 on sends the error back the same 5 steps) -> scan_credit_<net>_k4.npz.
Streams: the first NS = 10 of each cell's 50 in every scan environment (1000 per environment; env NS=50: all 5000 ->
scan_credit_<net>_ns50.npz, John 10-28 "more trajs per env", about 50 min on a shared GPU 0); targets: the
stream's own exact joint marginals (float64, from its observations and the table; not the episode's).
Validation (mode check): the harvest's first 100 episodes against the per-(episode, step) values read from the caches
(steps 17-30; the mapping agent's series_trained.npz).
Run from the repository root:
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig1c_scan_credit.py check
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig1c_scan_credit.py scan trained
    (pairs trained: the split pairs' streams -> scan_credit_<net>_pairs.npz)"""
import os
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.chdir(ROOT)
import gradflow_analysis as ga                                                    # noqa: E402

SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
OUT = os.path.join(SCAN, "figdata", "round7_jumps")
REF = "/tmp/claude-1002/-home-johns-Desktop-CognitiveGridworld/7af6b091-54ff-45a0-b4dd-ef0a3bc9c778/scratchpad/map/series_trained.npz"
R, T, H, M = 10, 30, 1000, 50
KMAX = int(os.environ.get("KMAX", "15"))                                       # credit lags k <= KMAX
NS = int(os.environ.get("NS", "10"))                                               # streams per cell: the first NS
ENVS = [e for e in range(40) if e != 2]
GPU_FRACTION = .1                                                                  # GPU 0 is shared with trainings


def build_model(net):
    from main.CognitiveGridworld import CognitiveGridworld
    load_env, reservoir = ga.AGENTS[net]
    agent = CognitiveGridworld(**dict(ga.CFG, episodes=1), reservoir=reservoir, load_env=load_env,
                               external_logger=lambda env: {})
    return agent, agent.model


def bands(net):
    b = np.load(ga.BASIS.format(net))
    sid = ga.cut_segments(b["evr"], 10)
    ids = np.unique(sid)
    return b["V"].astype(np.float32), (sid[:, None] == ids[None]).astype(np.float32)


def measures(m, obs, Z, goal, Pg, Po, V, OH, chunk=50):
    """the four update measures (B, T) for observations obs (B, T, 5), inputs Z, goal indices, and the targets: the goal
    variable's and the other variable's exact joint marginals Pg, Po (B, T, R)"""
    dev = m.STM.device
    B = len(obs)
    m.batch_num = B                                                                # (lstm_input reshapes by it)
    m.obs_flat = torch.as_tensor(obs, dtype=torch.float32, device=dev)
    m.active_Z = torch.as_tensor(Z, dtype=torch.float32, device=dev)
    m.goal_ind = torch.as_tensor(goal, device=dev)
    m.joint_goal_belief = torch.as_tensor(Pg, device=dev)
    Vt, OHt = torch.as_tensor(V, device=dev), torch.as_tensor(OH, device=dev)
    out = {k: np.zeros((B, T)) for k in ("coh", "dW", "dim", "band_x")}
    with torch.no_grad(), torch.autocast(dev.type, enabled=False):
        x = ga.lstm_input(m)
        st, W_hh = ga.forward_states(m, x)
        r = m.classifier_readout(st["u"])
        loss = ga.StepLoss(m, r)
        mo = SimpleNamespace(ctx_num=m.ctx_num, realization_num=m.realization_num, goal_ind=1 - m.goal_ind.long(),
                             joint_goal_belief=torch.as_tensor(Po, device=dev), DKL_sym=m.DKL_sym)
        loss_o = ga.StepLoss(mo, r)
        Zk = torch.cat([x, st["u_prev"]], -1)
        for t in range(T):
            e = ga.step_error(m, loss, r, t) + ga.step_error(m, loss_o, r, t)
            Dk = torch.stack([d for _, d in ga.backward(st, W_hh, e, t, KMAX)], 1)        # (B, K, 4H)
            K = Dk.shape[1]
            zk = Zk[:, t - torch.arange(K, device=dev)]
            DD = torch.einsum("bkg,blg->bkl", Dk, Dk).double()
            ZZ = torch.einsum("bkd,bld->bkl", zk, zk).double()
            del Dk
            A = DD @ ZZ
            E = (DD * ZZ).sum((-1, -2))
            out["coh"][:, t] = (DD.sum((-1, -2)) / DD.diagonal(dim1=-2, dim2=-1).sum(-1)).cpu().numpy()
            out["dW"][:, t] = E.cpu().numpy()
            out["dim"][:, t] = (A.diagonal(dim1=-2, dim2=-1).sum(-1) ** 2 / (A * A.transpose(-1, -2)).sum((-1, -2))).cpu().numpy()
            xs = torch.zeros(B, device=dev, dtype=torch.float64)
            for c0 in range(0, B, chunk):
                sl = slice(c0, min(B, c0 + chunk))
                probes = ((e[sl] @ Vt)[:, None, :] * OHt.T[None]) @ Vt.T                   # (c, bands, H)
                Db = torch.stack([d for _, d in ga.backward({k: v[sl] for k, v in st.items()}, W_hh, probes, t, KMAX)], 1)
                Mo = torch.einsum("cksg,clsg->cskl", Db, Db).double()                       # delta_B,k . delta_B,l
                Qs = torch.einsum("cksg,clg->cskl", Db, Db.sum(2)).double()                 # delta_B,k . delta_l
                zz = ZZ[sl][:, None]
                xs[sl] = ((Qs * zz).sum((-1, -2)) - (Mo * zz).sum((-1, -2))).sum(-1)
                del Db, Mo, Qs
            out["band_x"][:, t] = (xs / E).cpu().numpy()
    return out


def sensory(obs):
    o = obs.astype(np.float64)
    ones = np.concatenate([np.zeros_like(o[:, :1]), np.cumsum(o, 1)[:, :-1]], 1)
    p1 = (ones + 1) / (np.arange(o.shape[1])[None, :, None] + 2)
    nov = -(o * np.log(p1) + (1 - o) * np.log(1 - p1)).sum(-1)
    chg = np.abs(np.diff(np.concatenate([np.zeros_like(o[:, :1]), o], 1), axis=1)).sum(-1)
    chg[:, 0] = 0
    return nov, chg


def joint_marginals(L, g, obs):
    """the exact joint posterior's two marginals in RAW context order (n, T, 2, R) for observations obs (n, T, 5)"""
    o = obs.astype(np.float64)
    ll = (np.einsum("nto,oab->ntab", o, np.log(L)) + np.einsum("nto,oab->ntab", 1 - o, np.log1p(-L))).cumsum(1)
    p = np.exp(ll - ll.max((2, 3), keepdims=True))
    p /= p.sum((2, 3), keepdims=True)
    return np.stack([p.sum(3), p.sum(2)], 2)


def check():
    agent, m = build_model("trained")
    V, OH = bands("trained")
    H5 = np.load(ga.CACHE.format("trained"))
    n = 100
    obs, Z, goal = (H5[k][:n] for k in ("ep_obs_flat", "ep_active_Z", "ep_goal_ind"))
    jb = H5["ep_joint_belief"][:n].astype(np.float64)                                    # (n, T, 2, R) raw
    Pg = H5["ep_joint_goal_belief"][:n].astype(np.float64)
    Po = jb[np.arange(n), :, 1 - goal.astype(int)]
    t0 = time.time()
    got = measures(m, obs, Z, goal, Pg, Po, V, OH)
    ref = np.load(REF)
    print(f"check, {n} harvest episodes: {time.time() - t0:.0f} s on {m.STM.device}")
    for k in ("coh", "dW", "dim", "band_x"):
        a, b = got[k][:, 16:30], ref[k][:n]
        rel = np.abs(a - b) / (np.abs(b) + 1e-12)
        print(f"   {k:7s} median rel diff {np.median(rel):.1e}, 99th pct {np.percentile(rel, 99):.1e}, r {np.corrcoef(a.ravel(), b.ravel())[0, 1]:.6f}")


def scan(net, which="cells"):
    """which = cells: the first NS streams of every cell; pairs: both streams of every split pair (split_pairs.npz)"""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_FRACTION, 0)
    agent, m = build_model(net)
    V, OH = bands(net)
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    eps, goal_e, obs_all, L_all = meta["eps"].astype(int), meta["goal"].astype(int), meta["obs"], meta["L"]
    with np.load(ga.CACHE.format("trained")) as f:
        Z_all = f["ep_active_Z"][eps]
    sp = np.load(os.path.join(SCAN, "figdata", "round4b", "split_pairs.npz"))
    cell_rows = (np.arange(R * R)[:, None] * M + np.arange(NS)[None]).ravel()
    out = {"rows": cell_rows} if which == "cells" else {}
    for p, e in enumerate(ENVS):
        te = time.time()
        rows = cell_rows if which == "cells" else np.unique(np.r_[sp["i"][sp["env"] == e], sp["j"][sp["env"] == e]])
        if not len(rows):
            continue
        if which == "pairs":
            out[f"rows_e{e:02d}"] = rows
        obs = obs_all[e, rows]
        jm = joint_marginals(L_all[e], int(goal_e[e]), obs)                                # (n, T, 2, R) raw
        g = int(goal_e[e])
        res = {}
        for h in np.array_split(np.arange(len(rows)), max(1, int(np.ceil(len(rows) / 500)))):   # <= 500 (GPU memory)
            part = measures(m, obs[h].astype(np.float32), np.repeat(Z_all[e:e + 1], len(h), 0),
                            np.full(len(h), g), jm[h, :, g], jm[h, :, 1 - g], V, OH)
            for k, v in part.items():
                res.setdefault(k, []).append(v)
        res = {k: np.concatenate(v) for k, v in res.items()}
        nov, chg = sensory(obs)
        for k, v in {**res, "nov": nov, "chg": chg}.items():
            out[f"{k}_e{e:02d}"] = v.astype(np.float32)
        print(f"{net} env {e} ({p + 1}/39): {time.time() - te:.0f} s; elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    tag = (("" if NS == 10 else f"_ns{NS}") if which == "cells" else "_pairs") + ("" if KMAX == 15 else f"_k{KMAX}")
    np.savez(os.path.join(OUT, f"scan_credit_{net}{tag}.npz"), **out)
    print(f"saved; {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        scan(sys.argv[2] if len(sys.argv) > 2 else "trained", "pairs" if sys.argv[1] == "pairs" else "cells")
