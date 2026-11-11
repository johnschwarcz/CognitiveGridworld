"""The checks behind fig_learn_order_diag's RESULTS (John 11-09: "how can the curve go below 0 and then rise? do you need
a symmetric KL? Is the rising good? Is the entanglement learning part of the rising?"; "an accumulating sum of
entanglement through time, or the max entanglement so far?").
Per probe, step and variable, at every checkpoint of a run (td_tt_order's 10,000 probes, 114 checkpoints):
  own_t    KL(O_t || norm(O_t-1 q_t / q_t-1)): the error the network's step-t update adds when applied to the OPTIMAL
           belief (>= 0); naive's: its step likelihood applied to O_t-1
  merge_t  KL(O_t || norm(q_t-1 O_t / O_t-1)) - KL(O_t-1 || q_t-1): what the CORRECT update does to the network's
           inherited error; dev_t = dV_t - merge_t: the network's step against that ideal step
The two splits of the per-step loss dV_t disagree (the inherited error and the update interact); the update's share is
quoted as the mean of own and dev (Shapley over the two orders).
-> through_training/order/own_<run>.npz, merge_<run>.npz (float16 (K, n, T, 2)); report() prints the RESULTS numbers
Run from the repository root (CPU, ~10 s per run with 8 workers):
    OMP_NUM_THREADS=1 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_order_split.py
    OMP_NUM_THREADS=8 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_order_split.py report"""
import os
import sys
import glob
import time
import numpy as np
from multiprocessing import Pool

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import td_tt_order as to                                                                      # noqa: E402

R = 10
_G = {}


def _first(l):
    """the belief before each step (step 0: uniform)"""
    return np.concatenate([np.full_like(l[:, :1], -np.log(R)), l[:, :-1]], 1)


def _init():
    P = np.load(os.path.join(to.AN, "probes.npz"))
    lB = to.lsm(P["log_jb"].astype(np.float64))
    _G["lB"], _G["lBp"] = lB, _first(lB)


def _one(f):
    z = np.load(f)
    lq = to.lsm(z["logq"].astype(np.float64))
    lqp = _first(lq)
    lB, lBp = _G["lB"], _G["lBp"]
    pB = np.exp(lB)
    own = (pB * (lB - to.lsm(lBp + lq - lqp))).sum(-1)
    merge = (pB * (lB - to.lsm(lqp + lB - lBp))).sum(-1) - (np.exp(lBp) * (lBp - lqp)).sum(-1)
    return int(z["episode"]), own.astype(np.float16), merge.astype(np.float16)


def harvest(runs=("trained", "echo")):
    for run in runs:
        if all(os.path.exists(os.path.join(to.OUT, f"{k}_{run}.npz")) for k in ("own", "merge")):
            continue
        files = sorted(glob.glob(os.path.join(to.AN, run, "ck_*.npz")))
        t0 = time.time()
        with Pool(8, initializer=_init) as pool:
            res = pool.map(_one, files)
        ep = np.array([r[0] for r in res])
        np.savez(os.path.join(to.OUT, f"own_{run}.npz"), episode=ep, own=np.stack([r[1] for r in res]))
        np.savez(os.path.join(to.OUT, f"merge_{run}.npz"), episode=ep, merge=np.stack([r[2] for r in res]))
        print(run, f"{time.time() - t0:.0f} s", flush=True)


def report(run="trained"):
    import fig_learn_order_diag as fo
    Rf = to.refs()
    FR, E = Rf["FR"], Rf["E"]
    dFR = FR - to.prev(FR)
    bE, bF = to.deciles(E, 1), to.deciles(to.prev(FR), 1)
    cell = np.where((bE >= 0) & (bF >= 0), bE * 10 + bF, -1)
    Z = fo.load()
    P = np.load(os.path.join(to.AN, "probes.npz"))
    lB = to.lsm(P["log_jb"].astype(np.float64))
    lnpx = to.lsm(np.log(np.maximum(P["npx"].astype(np.float64), 1e-300)))
    own_nv = (np.exp(lB) * (lB - to.lsm(_first(lB) + lnpx))).sum(-1)
    A, M = to.prev(np.cumsum(E, 1)), to.prev(np.maximum.accumulate(E, 1))

    def rk(a):
        return np.argsort(np.argsort(a.ravel())).astype(float)

    def sp(a, b):
        return np.mean([np.corrcoef(rk(a[:, t]), rk(b[:, t]))[0, 1] for t in range(1, 30)])
    print(f"within-step rank r with FR_t-1: sum of E before {sp(A, to.prev(FR)):.3f}, max so far "
          f"{sp(M, to.prev(FR)):.3f}, E_t {sp(E, to.prev(FR)):.3f}")
    for nm, v in (("E_t", E), ("FR_t-1", to.prev(FR))):
        ac = [np.mean([np.corrcoef(rk(v[:, t]), rk(v[:, t + k]))[0, 1] for t in range(1, 30 - k)]) for k in (1, 5, 10, 20)]
        e = v[:, 1:]
        print(f"{nm}: across-episode rank autocorrelation at 1/5/10/20 steps {np.round(ac, 3)}; between-episode share "
              f"of the variance {e.mean(1).var() / e.var():.3f}")
    z = np.load(os.path.join(to.OUT, f"err_{run}.npz"))
    x, VV = z["x"], z["V"]
    own = np.load(os.path.join(to.OUT, f"own_{run}.npz"))["own"]
    mg = np.load(os.path.join(to.OUT, f"merge_{run}.npz"))["merge"]
    q5 = to.deciles(E, 0) // 2
    for b in range(len(fo.BANDS)):
        en = fo.ends(Z, fo.BANDS[b])
        ms = {k: np.isin(cell, en[k]) for k in ("E", "FR")}
        print(f"\n{run}, band {fo.BANDS[b]}; naive: " + ", ".join(
            f"{k} end dFR {dFR[ms[k]].mean():.4f} own {own_nv[ms[k]].mean():.4f} sum E before {A[ms[k]].mean():.2f} "
            f"max {M[ms[k]].mean():.2f}" for k in ("E", "FR")))
        print("     x | V_t-1 E / FR | own E / FR | dV - own E / FR | Shapley update E / FR | removal share of the gap "
              "| top-fifth-made part of V_t-1, FR end")
        for i in (30, 33, 36, 42, 45, 50, 60, 72, 90, 113):
            v = VV[i].astype(np.float64)
            vp = to.prev(v)
            dv, o, m_ = v - vp, own[i].astype(np.float64), mg[i].astype(np.float64)
            r = {k: (vp[ms[k]].mean(), dv[ms[k]].mean(), o[ms[k]].mean(), (dv - m_)[ms[k]].mean()) for k in ms}
            sh = {k: (r[k][2] + r[k][3]) / 2 for k in ms}
            gap_ = r["E"][1] - r["FR"][1]
            top = to.prev(np.cumsum(np.where(q5 == 4, dv, 0.), 1))[ms["FR"]].mean()
            print(f"{x[i]:6.0f} | {r['E'][0]:.3f} {r['FR'][0]:.3f} | {r['E'][2]:.4f} {r['FR'][2]:.4f} | "
                  f"{r['E'][1] - r['E'][2]:+.4f} {r['FR'][1] - r['FR'][2]:+.4f} | {sh['E']:.4f} {sh['FR']:.4f} | "
                  f"{1 - (sh['E'] - sh['FR']) / gap_:.2f} | {top:.3f}")
    en = fo.ends(Z, fo.BANDS[fo.MAIN])
    print("\nmatched inherited error, 0.03-0.05 band: per-step loss (and naive's) where V_t-1 < .02")
    for i in (55, 72):
        v = VV[i].astype(np.float64)
        vp = to.prev(v)
        s = []
        for k in ("E", "FR"):
            mk = np.isin(cell, en[k]) & (vp < .02)
            s.append(f"{k} end {(v - vp)[mk].mean():.4f} (naive {dFR[mk].mean():.4f})")
        print(f"{x[i]:6.0f}: " + ", ".join(s))


if __name__ == "__main__":
    report() if sys.argv[1:] == ["report"] else harvest()
