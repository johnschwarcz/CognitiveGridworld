"""Cross-correlations of the heat-map measures with FR and with entanglement ACROSS episodes (John 11-09: "how the
learning signal cross-correlates with FR through training"). The locked statistic of td_xcorr_lags z-scores within each
episode, which suits entanglement (it fluctuates from step to step) but not FR, whose variance is 67% between episodes
and which drifts smoothly within one (td_tt_order: autocorrelation .94 / .82 / .72 / .51 at 1 / 5 / 10 / 20 steps); the
FR blocks of train_learn_masks also rank EPISODES at each step. Statistic here: at each step t of the window (steps
17-30), Pearson r over the probes between the measure at t + tau and the selector at t (both steps in the window), then
the mean over t (average after the statistic). Selectors: FR_t = KL(Bayes_t || naive_t) of probes.npz ('fr': the
paper's forward form, exact, since 11-10; both variables averaged; the caches built before then with the symmetric
clipped KL are kept as xcorr_across_<run>_sym_superseded.npz), and the entanglement rev_t (both variables averaged). Measures (td_analysis ck_*.npz):
activity norm_u, effdim_u, turn_u (10,000 probes); update cr_coh, cr_E, cr_dim, cr_bandx (5,000 probes, error steps
17-30); band_u's 10 band shares (lag 0); credit share by lag k (td_tt_credit_reach full_*.npz, 2,000 probes) against
the selector at the landing step t - k.
-> main/DATA/experimental/training_dynamics/analysis/xcorr_across_<run>.npz
Run from the repository root (CPU, ~2 min per network):
    /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_xcorr_across.py"""
import os
import sys
import glob
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

AN = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics", "analysis")
TAU = np.arange(-8, 9)
W0, W1 = 16, 30                                                                # 0-based window: steps 17-30
ACT = ("norm_u", "effdim_u", "turn_u")
UPD = ("cr_coh", "cr_E", "cr_dim", "cr_bandx")
_S = {}


def selectors():
    if not _S:
        P = np.load(os.path.join(AN, "probes.npz"))
        _S["fr"] = P["fr"].astype(np.float64).mean(-1)
        _S["ent"] = P["ent"].astype(np.float64).mean(-1)
    return _S


def _r_cols(a, b):
    """Pearson r between matching columns of a and b (n, m) over rows; NaN where either is constant"""
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(0) / den


def lagged(y, s, ty0):
    """y (n, Ty) whose column j is step ty0 + j; s (n, T) the selector at every step: r(tau) = mean over t in the window
    of r over probes between y at t + tau and s at t"""
    out = np.full(len(TAU), np.nan)
    Ty = y.shape[1]
    for i, tau in enumerate(TAU):
        t = np.arange(W0, W1)
        t = t[(t + tau >= W0) & (t + tau < W1) & (t + tau - ty0 >= 0) & (t + tau - ty0 < Ty)]
        if len(t):
            out[i] = np.nanmean(_r_cols(y[:, t + tau - ty0].astype(np.float64), s[:, t]))
    return out


def one(f):
    S = selectors()
    z = np.load(f)
    res = {"episode": int(os.path.basename(f)[3:8])}
    for sel, s in S.items():
        res[f"act_{sel}"] = np.stack([lagged(z[k], s, 0) for k in ACT])
        if "cr_coh" in z.files:
            rows = z["cr_rows"]
            ty0 = int(z["cr_steps"][0]) - 1
            res[f"upd_{sel}"] = np.stack([lagged(z[k], s[rows], ty0) for k in UPD])
        b = z["band_u"].astype(np.float64)                                         # (n, T, 10)
        res[f"band_{sel}"] = np.array([np.nanmean(_r_cols(b[:, W0:W1, j], s[:, W0:W1])) for j in range(b.shape[-1])])
    return res


def credit(run):
    """lag k x checkpoint: r over probes between the share an error at step t sends k steps back and the selector at the
    landing step t - k, per error step t, then the mean over t"""
    import td_tt_credit_reach as cr
    S = selectors()
    out = {}
    for f in sorted(glob.glob(os.path.join(cr.OUT, f"full_{run}_*.npz"))):
        z = np.load(f)
        E, rows = z["E"].astype(np.float64), z["rows"]
        sh = E / E.sum(-1, keepdims=True)
        t_idx = np.asarray(cr.SRC)
        for sel, s in S.items():
            r = [np.nanmean(_r_cols(sh[:, :, k], s[rows][:, t_idx - k])) for k in range(sh.shape[-1])]
            out.setdefault(sel, []).append((float(z["x"]), r))
    return {sel: (np.array([a for a, _ in sorted(v)]), np.array([b for _, b in sorted(v)])) for sel, v in out.items()}


def data(run, workers=8):
    cache = os.path.join(AN, f"xcorr_across_{run}.npz")
    files = sorted(glob.glob(os.path.join(AN, run, "ck_*.npz")))
    eps = np.array([int(os.path.basename(f)[3:8]) for f in files])
    if os.path.exists(cache):
        z = dict(np.load(cache))
        if np.array_equal(z["episode"], eps):
            return z
    from multiprocessing import Pool
    selectors()
    with Pool(workers) as pool:
        res = pool.map(one, files)
    out = {"episode": eps, "x": np.maximum(eps - eps // 5, 10.).astype(float)}
    for k in res[0]:
        if k != "episode":
            out[k] = np.stack([r[k] if k in r else np.full_like(res[-1][k], np.nan) for r in res])
    for sel, (x, R) in credit(run).items():
        out[f"credit_x_{sel}"], out[f"credit_{sel}"] = x, R
    np.savez(cache, **out)
    return out


if __name__ == "__main__":
    for run in ("trained", "echo"):
        d = data(run)
        i0 = int(np.flatnonzero(TAU == 0)[0])
        print(run, {k: v.shape for k, v in d.items()})
