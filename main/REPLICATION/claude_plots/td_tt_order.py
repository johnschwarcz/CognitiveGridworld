"""Which is handled first through training: entanglement or factorization regret? (John 11-09: "can we understand if
managing entanglement precedes managing factorization regret or vice versa"). Per probe, step and variable, at every
td_analysis checkpoint of the Trained RNN and the ESN (114 each, the same 10,000 probes):
  V_t   belief error   KL(B_t || q_t): the network's excess log loss over Bayes at step t; naive's = FR_t = KL(B_t || N_t)
  U_t   update error   J/2 between the network's normalised step-t increment, softmax(log q_t - log q_t-1) (q_0
                       uniform), and Bayes's, softmax(log B_t - log B_t-1); naive's = the step's entanglement E_t (the
                       locked definition, probes 'ent': naive's normalised step likelihood against Bayes's increment)
  Uw_t  weighted update error (td_entanglement_memcut err_int, read, not recomputed): the network's step-t increment
                       applied to Bayes's belief at t-1, J/2 against B_t; naive's = the same with naive's step
                       likelihood (computed here, "Ew")
"Managing" a quantity = beating naive on the error that quantity is naive's error for: V below FR_t where FR_t is high,
U below E_t where E_t is high.
-> main/DATA/experimental/training_dynamics/through_training/order/
   refs.npz            FR (forward, per probe, step, variable; FR_0 = 0), E (= probes 'ent'), Ew, FRsym (= probes 'fr_sym': the superseded symmetric clipped KL; probes 'fr' is the forward one since 11-10)
   err_<run>.npz       episode (114,), x (114,), V, U (114, n, T, 2) float16
   summary.npz, cells.npz, carried.npz   per-environment sums on the decile grid (see summary(), cells(), carried())
Run from the repository root (CPU, ~4 min per network with 8 workers):
    OMP_NUM_THREADS=1 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_order.py harvest"""
import os
import sys
import glob
import time
import numpy as np
from multiprocessing import Pool

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import td_tt_gap as gap                                                                       # noqa: E402

AN = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics", "analysis")
OUT = os.path.join(gap.TT, "order")
T, R, NV = 30, 10, 2
RUNS = ("trained", "echo", "trained_p1", "trained_N200", "trained_N100", "trained_N50", "mlp55")


def lsm(x):
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def jhalf(la, lb):
    """J/2 between two normalised log distributions over the last axis"""
    return .5 * ((np.exp(la) - np.exp(lb)) * (la - lb)).sum(-1)


def incs(l):
    """normalised step increments of normalised log beliefs l (n, T, 2, R); the step-0 belief is uniform"""
    prev = np.concatenate([np.full_like(l[:, :1], -np.log(R)), l[:, :-1]], 1)
    return lsm(l - prev), prev


def refs():
    path = os.path.join(OUT, "refs.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    P = np.load(os.path.join(AN, "probes.npz"))
    lB, lN = lsm(P["log_jb"].astype(np.float64)), lsm(P["log_nb"].astype(np.float64))
    FR = (np.exp(lB) * (lB - lN)).sum(-1)
    dB, prevB = incs(lB)
    lnpx = lsm(np.log(np.maximum(P["npx"].astype(np.float64), 1e-300)))
    E = jhalf(lnpx, dB)
    Ew = jhalf(lsm(prevB + lnpx), lB)
    out = dict(FR=FR, E=E, Ew=Ew, FRsym=P["fr_sym"], E_stored=P["ent"], env=P["env"], goal=P["goal"])
    print(f"refs: E vs stored ent max |diff| {np.abs(E - P['ent']).max():.2e} (mean E {E.mean():.4f}); mean FR "
          f"{FR.mean():.4f}, FRsym {P['fr_sym'].mean():.4f}; mean Ew {Ew.mean():.4f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    return out


_G = {}


def _init():
    P = np.load(os.path.join(AN, "probes.npz"))
    lB = lsm(P["log_jb"].astype(np.float64))
    _G["lB"], _G["dB"] = lB, incs(lB)[0]


def _one(f):
    z = np.load(f)
    lq = lsm(z["logq"].astype(np.float64))
    lB, dB = _G["lB"], _G["dB"]
    V = (np.exp(lB) * (lB - lq)).sum(-1)
    U = jhalf(incs(lq)[0], dB)
    return int(z["episode"]), V.astype(np.float16), U.astype(np.float16)


def harvest(runs=RUNS, workers=8):
    refs()
    for run in runs:
        path = os.path.join(OUT, f"err_{run}.npz")
        if os.path.exists(path):
            continue
        files = sorted(glob.glob(os.path.join(AN, run, "ck_*.npz")))
        t0 = time.time()
        with Pool(workers, initializer=_init) as pool:
            res = pool.map(_one, files)
        ep = np.array([r[0] for r in res])
        np.savez(path, episode=ep, x=gap.train_x(ep), V=np.stack([r[1] for r in res]), U=np.stack([r[2] for r in res]))
        print(f"{run}: {len(files)} checkpoints in {time.time() - t0:.0f} s", flush=True)


K10 = 10
MATCH = (.03, .05)                  # the matched band of naive's loss added per step (nats), for the order test


def prev(a):
    """a at the previous step (step 0: 0)"""
    return np.concatenate([np.zeros_like(a[:, :1]), a[:, :-1]], 1)


def deciles(x, t0):
    """within-step deciles 0..9 of x (n, T, 2), probes and variables pooled at each step from index t0; -1 before"""
    b = np.full(x.shape, -1, np.int64)
    for t in range(t0, T):
        e = np.quantile(x[:, t], np.linspace(0, 1, K10 + 1)[1:-1])
        b[:, t] = np.searchsorted(e, x[:, t])
    return b


def _env_bins(env3, b, v, nb=K10):
    """per-environment sums of v over bins b (items with b < 0 dropped) -> (50, nb)"""
    ok = b >= 0
    return np.bincount(env3[ok] * nb + b[ok], v[ok], gap.N_ENV * nb).reshape(gap.N_ENV, nb)


def summary(runs=RUNS):
    """-> order/summary.npz. Loss added at step t: dV_t = V_t - V_t-1 (V_0 = 0), naive's dFR_t = FR_t - FR_t-1; their
    sum over steps is V_T and FR_T. Two binnings, per environment (sums; the figure's ratios are of sums):
      dec   within-step deciles of the step's entanglement E_t, all 30 steps: dec_<run> (K, 50, 10) sums of dV,
            n_dec (50, 10) counts, naive_dec (50, 10) sums of dFR, E_dec (10,) mean E per decile
      col   the order test, steps 2-30: the 10 x 10 grid of E_t deciles x FR_t-1 deciles (the state's FR before the
            step); naive_grid (10, 10) naive's mean dFR per cell (rows E, columns FR_t-1); matched (10, 10) = naive's
            mean in MATCH; the matched cells of each FR_t-1 decile pooled: col_<run> (K, 50, 10) sums of dV, n_col,
            naive_col (50, 10); E_col (10,) the mean E of the matched items of each column"""
    Rf = refs()
    FR, E = Rf["FR"], Rf["E"]
    env3 = np.broadcast_to(Rf["env"][:, None, None], FR.shape)
    dFR = FR - prev(FR)
    bE = deciles(E, 0)
    bE2, bF2 = deciles(E, 1), deciles(prev(FR), 1)
    cell = np.where((bE2 >= 0) & (bF2 >= 0), bE2 * K10 + bF2, -1)
    ok = cell >= 0
    grid = (np.bincount(cell[ok], dFR[ok], K10 * K10) / np.bincount(cell[ok], minlength=K10 * K10)).reshape(K10, K10)
    matched = (grid >= MATCH[0]) & (grid <= MATCH[1])
    col = np.where(ok & matched.ravel()[np.maximum(cell, 0)], bF2, -1)
    one = np.ones_like(FR)
    out = dict(naive_grid=grid, matched=matched, match=np.array(MATCH),
               n_dec=_env_bins(env3, bE, one), naive_dec=_env_bins(env3, bE, dFR),
               E_dec=np.array([E[bE == d].mean() for d in range(K10)]),
               n_col=_env_bins(env3, col, one), naive_col=_env_bins(env3, col, dFR),
               E_col=np.array([E[col == d].mean() if (col == d).any() else np.nan for d in range(K10)]),
               share_top5=float(dFR[deciles(E, 0) >= 8].sum() / dFR.sum()))
    for run in runs:
        z = np.load(os.path.join(OUT, f"err_{run}.npz"))
        V = z["V"].astype(np.float32)
        dV = V - np.concatenate([np.zeros_like(V[:, :, :1]), V[:, :, :-1]], 2)
        out[f"x_{run}"] = z["x"]
        out[f"dec_{run}"] = np.stack([_env_bins(env3, bE, d.astype(np.float64)) for d in dV])
        out[f"col_{run}"] = np.stack([_env_bins(env3, col, d.astype(np.float64)) for d in dV])
        print(f"summary {run}", flush=True)
    np.savez(os.path.join(OUT, "summary.npz"), runs=np.array(runs), **out)
    print("saved", os.path.join(OUT, "summary.npz"), "| top-entanglement fifth's share of naive's FR at T:",
          f"{out['share_top5']:.3f}")


def cells(runs=RUNS):
    """-> order/cells.npz: the order test on every cell of the 10 x 10 grid of within-step deciles of E_t (rows) and
    FR_t-1 (columns), steps 2-30, so that any band of equal naive loss (a diagonal of the grid) can be pooled by FR
    column or by E row: cell_<run> (K, 50, 100) per-environment sums of the network's loss added dV, naive_cell
    (50, 100) of naive's dFR, n_cell (50, 100) counts, E_cell / FRp_cell (100,) the cells' mean E_t and FR_t-1"""
    Rf = refs()
    FR, E = Rf["FR"], Rf["E"]
    env3 = np.broadcast_to(Rf["env"][:, None, None], FR.shape)
    dFR = FR - prev(FR)
    bE, bF = deciles(E, 1), deciles(prev(FR), 1)
    cell = np.where((bE >= 0) & (bF >= 0), bE * K10 + bF, -1)
    nb = K10 * K10
    out = dict(naive_cell=_env_bins(env3, cell, dFR, nb), n_cell=_env_bins(env3, cell, np.ones_like(FR), nb),
               E_cell=np.array([E[cell == c].mean() for c in range(nb)]),
               FRp_cell=np.array([prev(FR)[cell == c].mean() for c in range(nb)]))
    for run in runs:
        z = np.load(os.path.join(OUT, f"err_{run}.npz"))
        V = z["V"].astype(np.float32)
        dV = V - np.concatenate([np.zeros_like(V[:, :, :1]), V[:, :, :-1]], 2)
        out[f"x_{run}"] = z["x"]
        out[f"cell_{run}"] = np.stack([_env_bins(env3, cell, d.astype(np.float64), nb) for d in dV])
        print(f"cells {run}", flush=True)
    np.savez(os.path.join(OUT, "cells.npz"), runs=np.array(runs), **out)


def carried(runs=("trained", "echo")):
    """-> order/carried.npz (John 11-09: "how can the curve go below 0 and then rise?", "is the entanglement learning
    part of the rising?"): on the cells of cells(), per environment, the error each step INHERITS, i.e. the belief's
    error before the step, V_t-1 = KL(B_t-1 || q_t-1) (naive's: FR_t-1), and the part of it made at the earlier steps in
    the most entangled fifth (within-step fifths of E, all steps; V_t-1 = sum_s<t dV_s split by the fifth of s):
    Vp_cell_<run> / Vtop_cell_<run> (K, 50, 100) sums, naive FRp_cell / FRtop_cell (50, 100) sums"""
    Rf = refs()
    FR, E = Rf["FR"], Rf["E"]
    env3 = np.broadcast_to(Rf["env"][:, None, None], FR.shape)
    bE, bF = deciles(E, 1), deciles(prev(FR), 1)
    cell = np.where((bE >= 0) & (bF >= 0), bE * K10 + bF, -1)
    top = deciles(E, 0) >= 8                                                     # the most entangled fifth of each step
    nb = K10 * K10
    dFR = FR - prev(FR)
    out = dict(FRp_cell=_env_bins(env3, cell, prev(FR), nb),
               FRtop_cell=_env_bins(env3, cell, prev(np.cumsum(np.where(top, dFR, 0.), 1)), nb))
    for run in runs:
        z = np.load(os.path.join(OUT, f"err_{run}.npz"))
        Vp, Vt = [], []
        for v in z["V"]:
            v = v.astype(np.float64)
            dv = v - prev(v)
            Vp.append(_env_bins(env3, cell, prev(v), nb))
            Vt.append(_env_bins(env3, cell, prev(np.cumsum(np.where(top, dv, 0.), 1)), nb))
        out[f"Vp_cell_{run}"], out[f"Vtop_cell_{run}"], out[f"x_{run}"] = np.array(Vp), np.array(Vt), z["x"]
        print(f"carried {run}", flush=True)
    np.savez(os.path.join(OUT, "carried.npz"), **out)


if __name__ == "__main__":
    {"harvest": harvest, "summary": summary, "cells": cells, "carried": carried}[sys.argv[1]]()
