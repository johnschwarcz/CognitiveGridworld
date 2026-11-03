"""Two processes in the alias scan: state-driven diagonal unfolding (early, front-loaded) and observation-driven
entanglement responses (every step) (28 Oct 2026, evening; John: "it seems like late step jumps may be more relevant
to entanglement. For the lags coupling, i wonder if this depends on time. There may be 2 seperate processes happening,
the early effect of the unfolding of the grid (corners, entanglement) and then late entanglement effects which are
invisible after averaging when compared to the early effects, but are what we saw when we looked at
cross-correlations. ... see if you can put everything we've learned together (do some deep analyses and
visualizations)")
-> main/REPLICATION/figure_bin/fig1c_jump_lags_time.png    figure_lags_time()
   main/REPLICATION/figure_bin/fig1c_xcorr_early_late.png  figure_xcorr()
   main/REPLICATION/figure_bin/fig1c_two_processes.png     figure_two_processes()
   the written synthesis: main/DATA/experimental/fig1c_compression/report_6_two_processes.md
Caches: figdata/round8_two_processes/{lags_time,xcorr_time,timeline}.npz (small; rebuilt from the round 7 caches).

Data: the alias scan (39 environments x 100 cells x 50 streams x 30 steps; all agents on the same streams). Jumps:
each step's move of an agent's posterior mean, Delta mu_t = (Delta E[r_g], Delta E[r_c]) (fig1c_crossings.jumps),
amplitude |Delta mu_t| (cells) and coupling angle phi_t from the nearest axis (0 deg: one variable moves, 45: both).
Measures as in fig1c_crossings: entanglement rev_t (the update into step t), factorization regret FR_t (both
variables), the joint split I(r_g; r_c) of the Optimal's belief (joint_mi), the observation's novelty
-sum_i log p(o_ti | o_<t) (p = (ones before t + 1) / (t + 2)) and the number of features changed since t - 1; the
Trained RNN's activity norm |u_t|, effective dimension PR(u_t) (fixed basis), turn 1 - cos(u_t, u_t-1), and its
update's credit coherence, |Delta W|^2, Delta W's effective dimension and cross-band share (fig1c_scan_credit; lags
truncated at k <= t before step 16, so values are compared within a step). 'Minus typical' = minus the mean of the
same cell's 50 streams at the same step (log10 first for the heavy-tailed FR and entanglement).

figure_lags_time(): John's question "does the lags x coupling picture depend on time?". The Optimal's jumps that are
large FOR THEIR STEP (top 10% of amplitude within environment x step: 565,500 jumps; an absolute threshold would
pick almost only early jumps, the typical jump shrinks from 1.2 to 0.07 cells), aligned on the jump (lags -2 .. +6),
by coupling level (0 - 15 / 15 - 30 / 30 - 40 / 40 - 45 deg), in three windows of arrival steps (3 - 6 / 7 - 16 /
17 - 30; every lag exists in every window), amplitude-matched across the levels (reweighted to the window's
distribution of within-step amplitude ranks; 10 bins), bands 5 - 95% over environments resampled (500). Row 4: the
diagonal (40 - 45) minus one-axis (0 - 15) contrast for every arrival step x lag (matched within the step).
figure_xcorr(): the locked dynamics_interference statistic (each series z-scored within a stream over a 14-step
window; the mean lagged product a[t + tau] b[t], b = entanglement) on the scan's streams, early window (steps 3 - 16)
vs late (17 - 30; the locked figure's window), raw (as the locked figure) and minus typical; the observation part
(novelty + features changed, one pooled within-stream fit, removed from the reference as gradflow_analysis's
'novelty' control does; the residual stays in entanglement's own SD, so 'via the observation' + 'beyond' = the whole
curve); the ESN on the same 1,000 streams per environment (the first 10 of each cell; its credit cache).
figure_two_processes(): the synthesis, time on the x axis: row 1 process A (large jumps' direction, the state before
diagonal jumps, the network's cross-band share at them, the grid's environment-specific distortion), row 2 process B
(entanglement's drivers, the Trained RNN's response, its part beyond the observation vs the ESN's, in sliding 8-step
windows; the corrections the large jumps make).

Measured 28 Oct (printed by the functions; steps are arrival steps, 1-based):
- Large jumps that are diagonal (phi >= 40 deg among the top 10% of the step; .11 = no preferred direction), steps 2 /
  5 / 10 / 20 / 30: Optimal .35 / .28 / .17 / .11 / .08, Trained RNN .27 / .21 / .12 / .08 / .06, Naive .10 / .04 / .02
  / .02 / .03, ESN .15 / .05 / .03 / .03 / .03; the Optimal's median jump 1.21 / .46 / .23 / .11 / .07 cells.
- PROCESS A, the step before a diagonal jump (diagonal minus one-axis, top 10%; top 2% in brackets), steps 3 / 5 / 10 /
  15 / 20 / 30: joint split I(r_g; r_c) +.22 / +.22 / +.17 / +.11 / +.09 / +.07 nats (+.22 / +.28 / +.29 / +.23 / +.19 /
  +.11); FR +.18 / +.21 / +.15 / +.09 / +.04 / -.01 log10 (+.21 / +.24 / +.22 / +.22 / +.14 / +.03); entanglement +.12
  / +.04 / -.00 / -.00 / -.01 / -.02 (early only; top 2% +.20 at step 3); activity eff. dim. -2.7 / -5.4 / -1.1 / -.4 /
  -.5 / +.1 dims (-4.5 / -10.4 / -7.4 / -4.3 / -1.9 / -1.3). At the jump: novelty -.62 / -.55 / -.57 / -.43 / -.35 /
  -.34 nats (diagonal jumps follow LESS surprising observations), entanglement -.09 / -.13 / -.09 / -.06 / -.03 / -.01
  log10, cross-band share +.01 / -.08 / -.30 / -.35 / -.27 / -.02 (top 2% -.03 / -.16 / -.53 / -1.07 / -.95 / -.26).
- Rows 1 - 3 (one-axis 0 - 15 vs diagonal 40 - 45 deg): early (steps 3 - 6) joint split at t - 1 +.02 vs +.24, FR -.05
  vs +.15, entanglement at t - 1 +.01 vs +.07 and at t +.20 vs +.09, novelty at t +.94 vs +.37, activity eff. dim. at t
  +7.7 vs +.4; mid (7 - 16) joint split +.03 vs +.18, FR -.04 vs +.10, entanglement at t +.23 vs +.14, cross-band share
  at t +.41 vs +.11; late (17 - 30) joint split +.02 vs +.10, FR -.03 vs -.02, entanglement at t +.24 vs +.21, novelty
  +1.25 vs +.89, activity eff. dim. +4.4 vs +3.3, cross-band share at t +.44 vs +.25 (t - 1: -.03 vs -.12). Medians:
  1.6 - 2.4 cells early, .67 - .94 mid, .32 - .34 late; n 78,000 / 195,000 / 273,000.
- Corners: the diagonal large jumps lie in the four corner 2 x 2 blocks .38 / .45 / .47 / .39 / .31 / .19 of the time at
  steps 3 / 5 / 10 / 15 / 20 / 30 (one-axis .21 - .14; all cells .16).
- The converse: one step after a strong joint split (I in the top 10% of its environment and step) the large jumps are
  diagonal .44 / .28 / .17 (steps 3 - 6 / 7 - 16 / 17 - 30), after the weaker half .10 / .06 / .04.
- What the large jumps do (change of the distance to the true cell, cells), steps 2 / 5 / 10 / 15 / 20 / 30: diagonal
  -2.19 / -1.07 / -.49 / -.35 / -.18 / -.09, one-axis -1.54 / -.64 / -.26 / -.16 / -.11 / -.06, every stream -.53 / -.17
  / -.06 / -.04 / -.02 / -.01.
- The grid's environment-specific distortion (cell x environment / (cell + cell x environment)) at t 1 / 4 / 8 / 30:
  Optimal .21 / .11 / .10 / .08, Trained RNN .24 / .12 / .10 / .09, Naive .28 / .31 / .35 / .45, ESN .30 / .32 / .36 /
  .49: the joint agents' grid is environment-independent by step 4 - 8.
- Fixed credit window (KMAX = 4; 1,000 streams per environment; steps 1 - 5 identical to the usual cache): the
  cross-band share at diagonal jumps (diagonal minus one-axis, at t) .01 / -.06 / -.18 / -.14 / -.09 / -.05 / +.05 at
  steps 3 / 5 / 6 / 10 / 15 / 20 / 30, at t - 1 about 0; the usual window on the same streams .01 / -.06 / -.15 / -.26 /
  -.38 / -.30 / +.03, at t - 1 -.19 to -.24 at steps 12 - 20. So the dip at the jump is there as soon as the window is
  full (step 5 - 6) and fades by step 20; its size, its lead (t - 1) and its late tail grow with the credit window.
  Coherence and dW energy contrasts are the same under both windows.
- PROCESS B, the cross-correlations. The scan's late window reproduces the locked figure (harvest, steps 17 - 30) at
  tau 0: novelty +.47 vs +.44, features changed +.22 vs +.20, norm +.37 vs +.39, eff. dim. +.27 vs +.26, turn +.26 vs
  +.26, coherence +.34 vs +.34, dW energy +.18 vs +.19, cross-band share +.30 vs +.29, dW eff. dim. -.17 vs -.18. Early
  window (steps 3 - 16), raw / minus typical: eff. dim. +.14 / +.20, coherence +.30 / +.31, dW energy -.02 / +.08,
  cross-band +.15 / +.26, dW eff. dim. -.11 / -.14 (late, minus typical: +.27, +.34, +.19, +.30, -.17). Over steps 3 -
  16 the cells' typical course holds 38 / 61 / 63 / 47% of a stream's variance of eff. dim., dW energy, dW eff. dim.,
  cross-band share (steps 17 - 30: 5 / 7 / 3 / 3%): that is what hides the response in the raw early series.
  Sliding 8-step windows (minus typical), centres 5.5 -> 26.5: novelty +.39 -> +.46, features changed +.22 -> +.20,
  jump size +.32 -> +.36 -> +.33, FR +.15 -> +.07; norm +.29 -> +.37, eff. dim. +.19 -> +.28, turn +.29 -> +.24,
  coherence +.28 -> +.32, cross-band share +.21 -> +.29 (raw +.03 -> +.29), dW eff. dim. -.14 -> -.18.
- Beyond the observation (novelty and change fitted out of entanglement), the Trained RNN at every window: norm +.29 ->
  +.25, coherence +.15, cross-band share +.10 -> +.12, eff. dim. +.07 - .08, dW eff. dim. -.10 -> -.12; the ESN within
  +-.09 (norm -.09 -> -.06). The locked figure's FULL sensory set (patterns at lags 0 - 1, observations at lags 2 - 4,
  all pairwise changes, novelty at lags 0 - 4; xcorr_full) gives the same on the scan within .04 (minus typical, early /
  late: norm +.27 / +.24, coherence +.15 / +.15, cross-band share +.11 / +.11; ESN within +-.06) and on the harvest
  (Trained norm +.28, coherence +.17, cross-band +.12; ESN within +-.03). Jump features (the Optimal's log size, angle
  and their product at t - 1, t, t + 1) explain ~10% of entanglement's within-stream variance; novelty and change ~20%.
- John's "late step jumps may be more relevant to entanglement": for the LARGE jumps, yes. The top 10% are equally
  exceptional in every window (log10 size +.54 / +.55 / +.58 above the cell's typical) but carry more entanglement late
  (+.14 / +.20 / +.23 log10 at the jump), because early a third of them are state-driven diagonal jumps, whose
  entanglement sits partly one step before and is lower at the jump. Per unit of log size over all jumps the slope is
  .28 / .30 / .23 (steps 3 - 6 / 7 - 16 / 17 - 30).
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig1c_two_processes.py \\
        [lags|xcorr|synthesis|all]"""
import os
import sys
import warnings
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
import plot_fig1c_measures_movie as mm                                            # noqa: E402
import fig1c_crossings as fc                                                      # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)
R, T, E, NS, S = 10, 30, 39, 50, 5000
OUT = os.path.join(mm.SCAN, "figdata", "round8_two_processes")
AGENTS = fc.AGENTS                                                                # (key, name) x 4
COL = {"Joint": "k", "Trained": "#1f77b4", "Naive": "0.55", "Echo": "#ff7f0e"}
LEVELS = fc.LEVELS                                                                # coupling levels (deg, colour)
WIN_LAGS = {"early": (3, 6), "mid": (7, 16), "late": (17, 30)}                    # arrival steps (1-based, incl.)
WIN_X = {"early": (3, 16), "late": (17, 30)}                                      # xcorr windows (1-based, incl.)
LAGS = np.arange(-2, 7)                                                           # jump-aligned lags
TAUS = np.arange(-6, 7)                                                           # xcorr lags
SLIDE = 8                                                                         # sliding xcorr window (steps)
NBIN = 10                                                                         # amplitude-rank bins (matching)
XKEYS = ("nov", "chg", "amp", "l2", "pr", "turn", "coh", "dW", "dim", "band_x", "fr", "mi")
LKEYS = ("mi", "fr", "rev", "nov", "pr", "band_x", "coh", "turn", "l2", "dist", "err", "amp")


# ═══════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════

def series(echo=False):
    """per-stream series (E, S, T) float64, step index t = 0 .. 29 (step t + 1): the Optimal's measures, the Trained
    RNN's network measures (all 5,000 streams), every agent's jump (E, S, T, 2; NaN at step 1), the Optimal's
    distance to the true cell; echo=True adds the ESN's network measures for its 1,000 streams per environment
    ("rows")"""
    meta = np.load(os.path.join(mm.SCAN, "scan_meta.npz"))
    V = {"rev": meta["rev"][mm.ENVS].astype(np.float64), "fr": mm.fr_both().astype(np.float64),
         "err": mm.err_both().astype(np.float64),
         "mi": np.load(os.path.join(mm.R7, "joint_mi.npz"))["mi"].astype(np.float64)}
    hid = np.load(os.path.join(mm.R7, "scan_hidden_trained.npz"))
    for k in ("l2", "pr", "turn"):
        V[k] = np.stack([hid[f"{k}_e{e:02d}"] for e in mm.ENVS]).astype(np.float64)
    cre = np.load(os.path.join(mm.R7, "scan_credit_trained_ns50.npz"))
    assert (cre["rows"] == np.arange(S)).all()
    for k in ("coh", "dW", "dim", "band_x", "nov", "chg"):
        V[k] = np.stack([cre[f"{k}_e{e:02d}"] for e in mm.ENVS]).astype(np.float64)
    J = fc.jumps()
    for a, _ in AGENTS:
        d = np.full((E, S, T, 2), np.nan)
        d[:, :, 1:] = J[a]
        V["d_" + a] = d
    del J
    B = np.load(os.path.join(mm.SCAN, "scan_Joint.npz"))["beliefs"][mm.ENVS].astype(np.float32)
    r = np.arange(R, dtype=np.float32)
    mu = np.stack([B[..., :R] @ r / B[..., :R].sum(-1), B[..., R:] @ r / B[..., R:].sum(-1)], -1)
    del B
    cell = meta["cells"][meta["cell_of"]].astype(np.float64)                         # (S, 2)
    V["dist"] = np.hypot(mu[..., 0] - cell[None, :, None, 0], mu[..., 1] - cell[None, :, None, 1]).astype(np.float64)
    V["cell"] = cell
    V["amp"] = np.hypot(V["d_Joint"][..., 0], V["d_Joint"][..., 1])
    if echo:
        ce = np.load(os.path.join(mm.R7, "scan_credit_echo.npz"))
        rows = ce["rows"]
        he = np.load(os.path.join(mm.R7, "scan_hidden_echo.npz"))
        Ve = {k: np.stack([he[f"{k}_e{e:02d}"][rows] for e in mm.ENVS]).astype(np.float64)
              for k in ("l2", "pr", "turn")}
        Ve.update({k: np.stack([ce[f"{k}_e{e:02d}"] for e in mm.ENVS]).astype(np.float64)
                   for k in ("coh", "dW", "dim", "band_x", "nov", "chg")})
        assert np.allclose(Ve["nov"], V["nov"][:, rows], atol=1e-4), "the ESN's streams are not the Trained's rows"
        V["echo"], V["rows"] = Ve, rows
    return V


def dep(V, ns=NS):
    """minus the mean of the same cell's ns streams at the same step (the streams are stored by cell)"""
    m = np.nanmean(V.reshape(V.shape[0], R * R, ns, -1), 2)
    return V - np.repeat(m, ns, axis=1)


def polar(d):
    return fc.polar(d)


def within_rank(amp):
    """each jump's amplitude rank among the environment's 5,000 streams at the same step, 0 .. 1 (NaN at step 1)"""
    rk = np.full(amp.shape, np.nan)
    rk[:, :, 1:] = np.argsort(np.argsort(amp[:, :, 1:], axis=1), axis=1) / (amp.shape[1] - 1)
    return rk


def level_of(phi):
    lev = np.full(phi.shape, -1, np.int8)
    for g, (lo, hi, _) in enumerate(LEVELS):
        lev[(phi >= lo) & ((phi < hi) if hi < 45 else (phi <= hi))] = g
    return lev


# ═══════════════════════════════════════════════════════════════════
# Jump-aligned curves by coupling x time (process A vs B at the jumps)
# ═══════════════════════════════════════════════════════════════════

def _transforms(V):
    """the departures the jump-aligned curves use: log10 for FR, entanglement, error, activity turn; raw otherwise"""
    lg = {"fr": 1e-3, "rev": None, "err": None, "turn": 1e-6}
    X = {}
    for k in LKEYS:
        v = V[k]
        if k in lg:
            v = np.log10(np.maximum(v, lg[k]) if lg[k] else v)
        X[k] = dep(v) if k != "amp" else v
    return X


def lags_time(q=.10, nboot=500, seed=0):
    """cached: per window x level x lag the matched mean of each measure and its 5 - 95% band; per arrival step x lag
    the diagonal minus one-axis contrast; counts"""
    path = os.path.join(OUT, f"lags_time_q{int(round(100 * q)):02d}.npz")
    if os.path.exists(path):
        with np.load(path) as f:
            return {k: f[k] for k in f}
    V = series()
    X = _transforms(V)
    amp, phi = polar(V["d_Joint"])
    rk = within_rank(amp)
    big = rk >= 1 - q
    e_, s_, t_ = np.nonzero(big)
    lev = level_of(phi[e_, s_, t_])
    rb = np.minimum(((rk[e_, s_, t_] - (1 - q)) / q * NBIN).astype(int), NBIN - 1)
    corner = ((np.minimum(V["cell"][:, 0], 9 - V["cell"][:, 0]) <= 1)
              & (np.minimum(V["cell"][:, 1], 9 - V["cell"][:, 1]) <= 1))
    out = {"lags": LAGS, "q": q, "n_events": len(e_)}
    rng = np.random.default_rng(seed)
    bidx = rng.integers(0, E, (nboot, E))
    ng, nl = len(LEVELS), len(LAGS)
    for w, (wn, (lo, hi)) in enumerate(WIN_LAGS.items()):
        inw = (t_ + 1 >= lo) & (t_ + 1 <= hi)
        pooled = np.bincount(rb[inw], minlength=NBIN) / inw.sum()
        wt = np.zeros(len(e_))
        for g in range(ng):
            m = inw & (lev == g)
            h = np.bincount(rb[m], minlength=NBIN) / max(m.sum(), 1)
            wt[m] = (pooled / np.maximum(h, 1e-12))[rb[m]]
            out[f"n_{wn}_{g}"] = m.sum()
            out[f"amp_{wn}_{g}"] = np.median(amp[e_[m], s_[m], t_[m]])
            out[f"corner_{wn}_{g}"] = corner[s_[m]].mean()
        for k in LKEYS:
            Ssum, Wsum = np.zeros((E, ng, nl)), np.zeros((E, ng, nl))
            for li, L in enumerate(LAGS):
                tt = t_ + L
                ok = inw & (lev >= 0) & (tt >= 0) & (tt < T)
                v = X[k][e_[ok], s_[ok], tt[ok]]
                fin = np.isfinite(v)
                cell_ = e_[ok][fin] * ng + lev[ok][fin]
                ww = wt[ok][fin]
                Ssum[:, :, li] = np.bincount(cell_, v[fin] * ww, minlength=E * ng).reshape(E, ng)
                Wsum[:, :, li] = np.bincount(cell_, ww, minlength=E * ng).reshape(E, ng)
            out[f"m_{wn}_{k}"] = Ssum.sum(0) / Wsum.sum(0)                           # (levels, lags)
            bs = Ssum[bidx].sum(1) / Wsum[bidx].sum(1)                                # (nboot, levels, lags)
            out[f"b_{wn}_{k}"] = np.percentile(bs, [5, 95], axis=0)
    # per arrival step x lag: diagonal (40 - 45) minus one-axis (0 - 15), matched within the step
    for k in LKEYS:
        out[f"map_{k}"] = np.full((T, nl), np.nan)
    for t in range(2, T):                                                           # arrival steps 3 .. 30
        m = t_ == t
        pooled = np.bincount(rb[m], minlength=NBIN) / m.sum()
        sel = {}
        for g in (0, ng - 1):
            mg = m & (lev == g)
            h = np.bincount(rb[mg], minlength=NBIN) / max(mg.sum(), 1)
            sel[g] = (mg, (pooled / np.maximum(h, 1e-12))[rb[mg]])
        for k in LKEYS:
            for li, L in enumerate(LAGS):
                if not 0 <= t + L < T:
                    continue
                vals = []
                for g in (ng - 1, 0):
                    mg, ww = sel[g]
                    v = X[k][e_[mg], s_[mg], t + L]
                    fin = np.isfinite(v)
                    vals.append((v[fin] * ww[fin]).sum() / ww[fin].sum())
                out[f"map_{k}"][t, li] = vals[0] - vals[1]
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    return out


def lags_k4(q=.10):
    """cached: the diagonal minus one-axis contrast per arrival step x lag for the update measures computed with a FIXED
    credit window (k <= 4 at every step from 5 on; fig1c_scan_credit.py with KMAX=4, NS=10) beside the same measures
    with the usual window (k <= min(15, t)) on the same 1,000 streams per environment; the jumps ranked within
    environment x step among all 5,000 streams, as in lags_time"""
    path = os.path.join(OUT, f"lags_k4_q{int(round(100 * q)):02d}.npz")
    if os.path.exists(path):
        with np.load(path) as f:
            return {k: f[k] for k in f}
    V = series()
    c4 = np.load(os.path.join(mm.R7, "scan_credit_trained_k4.npz"))
    rows = c4["rows"]
    amp, phi = polar(V["d_Joint"])
    rk = within_rank(amp)[:, rows]
    phi = phi[:, rows]
    out = {"lags": LAGS}
    keys = ("coh", "dW", "dim", "band_x")
    X = {}
    for k in keys:
        v4 = np.stack([c4[f"{k}_e{e:02d}"] for e in mm.ENVS]).astype(np.float64)
        v15 = V[k][:, rows]
        if k == "dW":
            v4, v15 = np.log10(v4), np.log10(v15)
        X[k + "_k4"], X[k + "_k15"] = dep(v4, 10), dep(v15, 10)
    big = rk >= 1 - q
    e_, s_, t_ = np.nonzero(big)
    lev = level_of(phi[e_, s_, t_])
    rb = np.minimum(((rk[e_, s_, t_] - (1 - q)) / q * NBIN).astype(int), NBIN - 1)
    for k in X:
        out[f"map_{k}"] = np.full((T, len(LAGS)), np.nan)
    ng = len(LEVELS)
    for t in range(2, T):
        m = t_ == t
        pooled = np.bincount(rb[m], minlength=NBIN) / m.sum()
        sel = {}
        for g in (0, ng - 1):
            mg = m & (lev == g)
            h = np.bincount(rb[mg], minlength=NBIN) / max(mg.sum(), 1)
            sel[g] = (mg, (pooled / np.maximum(h, 1e-12))[rb[mg]])
        for k in X:
            for li, L in enumerate(LAGS):
                if not 0 <= t + L < T:
                    continue
                vals = []
                for g in (ng - 1, 0):
                    mg, ww = sel[g]
                    v = X[k][e_[mg], s_[mg], t + L]
                    fin = np.isfinite(v)
                    vals.append((v[fin] * ww[fin]).sum() / ww[fin].sum())
                out[f"map_{k}"][t, li] = vals[0] - vals[1]
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    return out


def print_lags_time(Z):
    names = [f"{lo}-{hi}" for lo, hi, _ in LEVELS]
    print(f"top {100 * float(Z['q']):.0f}% for their step: {int(Z['n_events']):,} jumps; matched means, minus typical")
    for wn in WIN_LAGS:
        print(f"== {wn} (arrival steps {WIN_LAGS[wn][0]} - {WIN_LAGS[wn][1]}): n " + " / ".join(
            f"{int(Z[f'n_{wn}_{g}']):,}" for g in range(len(LEVELS))) + "; median amplitude " + " / ".join(
            f"{float(Z[f'amp_{wn}_{g}']):.2f}" for g in range(len(LEVELS))) + " cells; corner blocks " + " / ".join(
            f"{float(Z[f'corner_{wn}_{g}']):.2f}" for g in range(len(LEVELS))))
        for k in LKEYS:
            M = Z[f"m_{wn}_{k}"]
            print(f"   {k:6s} " + " | ".join(f"{names[g]}: " + " ".join(f"{v:+.2f}" for v in M[g, :5])
                                             for g in range(len(LEVELS))) + "  (lags -2 .. +2)")


# ═══════════════════════════════════════════════════════════════════
# Cross-correlations by time window (process B)
# ═══════════════════════════════════════════════════════════════════

def _zs(a):
    return (a - a.mean(-1, keepdims=True)) / (a.std(-1, keepdims=True) + 1e-12)


def _lagged(a, b, taus=TAUS):
    """gradflow_analysis._lagged: the mean over streams and steps of a[t + tau] * b[t], a and b (N, W) scaled"""
    out = []
    for L in taus:
        if L < 0:
            out.append((a[:, :L] * b[:, -L:]).mean())
        elif L > 0:
            out.append((a[:, L:] * b[:, :-L]).mean())
        else:
            out.append((a * b).mean())
    return np.array(out)


def _resid(y, Z):
    """y (N, W) z-scored within stream minus one pooled least-squares fit on Z (N, W, p), all centred within stream;
    the residual stays in y's units (gradflow_analysis's control)"""
    Zc = (Z - Z.mean(1, keepdims=True)).reshape(-1, Z.shape[-1])
    yc = (y - y.mean(1, keepdims=True)).reshape(-1)
    beta = np.linalg.lstsq(Zc, yc, rcond=None)[0]
    return y - (Zc @ beta).reshape(y.shape)


def _win(X, idx):
    return X[:, :, idx].reshape(-1, len(idx))


def xcorr_time():
    """cached: the locked statistic per (network, window, raw / minus typical, reference) for every measure, and the
    tau 0 / +1 values in sliding 8-step windows"""
    path = os.path.join(OUT, "xcorr_time.npz")
    if os.path.exists(path):
        with np.load(path) as f:
            return {k: f[k] for k in f}
    V = series(echo=True)
    rows = V["rows"]
    la = np.log10(np.maximum(V["amp"], 1e-4))
    phi = polar(V["d_Joint"])[1] / 45
    base = {k: V[k] for k in ("rev", "l2", "pr", "turn", "coh", "dW", "dim", "band_x", "nov", "chg", "fr", "mi")}
    base["amp"] = la
    base["phi"] = phi
    nets = {"trained": (base, NS, slice(None)),
            "trained10": ({k: v[:, rows] for k, v in base.items()}, 10, rows),
            "echo": ({**{k: base[k][:, rows] for k in ("rev", "nov", "chg", "amp", "phi", "fr", "mi")},
                      **V["echo"]}, 10, rows)}
    out = {"taus": TAUS}

    def refs_for(X, idx, which):
        ref = _zs(_win(X["rev"], idx))
        R_ = {"ent": ref}
        obs = np.stack([_win(X["nov"], idx), _win(X["chg"], idx)], -1)
        R_["beyond_obs"] = _resid(ref, obs)
        if which == "all":
            def sh(v, s_):
                return v[:, :, np.clip(idx + s_, 1, T - 1)].reshape(-1, len(idx))
            jf = np.stack([sh(X["amp"], s_) for s_ in (-1, 0, 1)] + [sh(X["phi"], s_) for s_ in (-1, 0, 1)]
                          + [sh(X["amp"], s_) * sh(X["phi"], s_) for s_ in (-1, 0, 1)], -1)
            R_["beyond_jumps"] = _resid(ref, jf)
            R_["beyond_both"] = _resid(ref, np.concatenate([jf, obs], -1))
        return R_

    for net, (B_, ns, _) in nets.items():
        for variant in ("raw", "dep"):
            X = {k: (dep(v, ns) if variant == "dep" else v) for k, v in B_.items()}
            for wn, (lo, hi) in WIN_X.items():
                idx = np.arange(lo - 1, hi)
                Rf = refs_for(X, idx, "all" if net == "trained" else "obs")
                for k in XKEYS:
                    if k not in X:
                        continue
                    a = _zs(_win(X[k], idx))
                    for rn, b in Rf.items():
                        out[f"{net}|{variant}|{wn}|{rn}|{k}"] = _lagged(a, b)
                # how much of a stream's variance over the window is its cell's typical course (raw series only)
                if variant == "raw":
                    for k in XKEYS:
                        if k in X and k not in ("amp", "phi"):
                            raw_ = _win(X[k], idx)
                            d_ = _win(dep(B_[k], ns), idx)
                            out[f"{net}|varshare|{wn}|{k}"] = np.array(
                                [np.mean(d_.var(1)) / np.mean(raw_.var(1))])
            # sliding windows: tau 0 and +1
            starts = np.arange(1, T - SLIDE + 1)                                    # step index of the first step
            for k in XKEYS:
                if k in X:
                    out[f"{net}|{variant}|slide|ent|{k}"] = np.zeros((len(starts), 2))
                    out[f"{net}|{variant}|slide|beyond_obs|{k}"] = np.zeros((len(starts), 2))
            for i, s0 in enumerate(starts):
                idx = np.arange(s0, s0 + SLIDE)
                Rf = refs_for(X, idx, "obs")
                for k in XKEYS:
                    if k not in X:
                        continue
                    a = _zs(_win(X[k], idx))
                    for rn, b in Rf.items():
                        out[f"{net}|{variant}|slide|{rn}|{k}"][i] = _lagged(a, b, taus=(0, 1))
            out["slide_mid"] = starts + 1 + (SLIDE - 1) / 2                            # window centre (1-based step)
        print("xcorr:", net, flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    return out


def _sensory_full(obs, nov, idx):
    """gradflow_analysis.sensory_set(kind="full") on the scan, for the window idx: the current and previous observation
    patterns (one-hot, 31 each), the per-feature change between them, the observations at lags 2 - 4, every other pair
    of lags 0 - 4's per-feature differences, the novelty at lags 0 - 4 (linear and squared), step dummies;
    lags before step 1 are clipped to step 1. obs (N, T, 5), nov (N, T) -> (N, W, 150)"""
    o = obs.astype(np.float64)
    pat = (obs * (1 << np.arange(5))).sum(-1)

    def at(x, L):
        return x[:, np.clip(idx - L, 0, None)]

    def oh(L):
        return (at(pat, L)[..., None] == np.arange(1, 32)).astype(np.float64)

    cols = [oh(0), oh(1), np.abs(at(o, 0) - at(o, 1))]
    cols += [at(o, L) for L in (2, 3, 4)]
    cols += [np.abs(at(o, a) - at(o, b)) for a in range(5) for b in range(a + 1, 5) if (a, b) != (0, 1)]
    cols += [np.stack([at(nov, L) for L in range(5)], -1), np.stack([at(nov, L) ** 2 for L in range(5)], -1)]
    cols.append(np.broadcast_to(np.eye(len(idx))[None, :, 1:], (len(o), len(idx), len(idx) - 1)))
    return np.concatenate(cols, -1)


def _resid_chunked(y, design, n, chunk=20000):
    """_resid with the design built per chunk of streams (normal equations; pinv for its collinear columns)"""
    G, g = 0, 0
    for c0 in range(0, n, chunk):
        Z = design(slice(c0, min(n, c0 + chunk)))
        Zc = (Z - Z.mean(1, keepdims=True)).reshape(-1, Z.shape[-1])
        yc = (y[c0:c0 + chunk] - y[c0:c0 + chunk].mean(1, keepdims=True)).reshape(-1)
        G, g = G + Zc.T @ Zc, g + Zc.T @ yc
    beta = np.linalg.pinv(G) @ g
    out = y.copy()
    for c0 in range(0, n, chunk):
        Z = design(slice(c0, min(n, c0 + chunk)))
        Zc = (Z - Z.mean(1, keepdims=True)).reshape(-1, Z.shape[-1])
        out[c0:c0 + chunk] -= (Zc @ beta).reshape(out[c0:c0 + chunk].shape)
    return out


def xcorr_full():
    """cached: the cross-correlations with entanglement beyond the locked figure's FULL sensory set (_sensory_full) per
    network x window, raw and minus typical"""
    path = os.path.join(OUT, "xcorr_full.npz")
    if os.path.exists(path):
        with np.load(path) as f:
            return {k: f[k] for k in f}
    V = series(echo=True)
    rows = V["rows"]
    obs = np.load(os.path.join(mm.SCAN, "scan_meta.npz"))["obs"][mm.ENVS]                  # (E, 5000, T, 5)
    keys = ("l2", "pr", "turn", "coh", "dW", "dim", "band_x")
    nets = {"trained": ({k: V[k] for k in keys + ("rev", "nov")}, NS, np.arange(S)),
            "echo": ({**{k: V["echo"][k] for k in keys}, "rev": V["rev"][:, rows], "nov": V["nov"][:, rows]}, 10, rows)}
    out = {"taus": TAUS}
    for net, (B_, ns, r_) in nets.items():
        ob = obs[:, r_].reshape(-1, T, 5)
        nv = B_["nov"].reshape(-1, T)
        for variant in ("raw", "dep"):
            X = {k: (dep(v, ns) if variant == "dep" else v) for k, v in B_.items()}
            for wn, (lo, hi) in WIN_X.items():
                idx = np.arange(lo - 1, hi)
                ref = _zs(_win(X["rev"], idx))
                res = _resid_chunked(ref, lambda sl: _sensory_full(ob[sl], nv[sl], idx), len(ref))
                for k in keys:
                    out[f"{net}|{variant}|{wn}|beyond_full|{k}"] = _lagged(_zs(_win(X[k], idx)), res)
            print("xcorr_full:", net, variant, flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    return out


def print_xcorr(Z):
    i0 = int(np.flatnonzero(Z["taus"] == 0)[0])
    for net in ("trained", "trained10", "echo"):
        for variant in ("raw", "dep"):
            for wn in WIN_X:
                refs = (("ent", "beyond_obs", "beyond_jumps", "beyond_both") if net == "trained"
                        else ("ent", "beyond_obs"))
                line = f"{net:9s} {variant} {wn:5s}:"
                for k in XKEYS:
                    key = f"{net}|{variant}|{wn}|ent|{k}"
                    if key in Z:
                        line += f" {k} " + "/".join(f"{Z[f'{net}|{variant}|{wn}|{r}|{k}'][i0]:+.2f}" for r in refs)
                print(line)
    print("variance of a stream's series over the window that is NOT its cell's typical course (raw):")
    for wn in WIN_X:
        print(f"  {wn}: " + " ".join(f"{k} {float(Z[f'trained|varshare|{wn}|{k}'][0]):.2f}" for k in XKEYS
                                     if f"trained|varshare|{wn}|{k}" in Z))


# ═══════════════════════════════════════════════════════════════════
# The timeline (synthesis)
# ═══════════════════════════════════════════════════════════════════

def timeline(q_list=(.10, .02)):
    """cached: per step the jumps' size and direction for the four agents, the corrections the large jumps make, the
    grid's shared part and environment-specific distortion for the four agents"""
    path = os.path.join(OUT, "timeline.npz")
    if os.path.exists(path):
        with np.load(path) as f:
            return {k: f[k] for k in f}
    V = series()
    out = {}
    for a, _ in AGENTS:
        amp, phi = polar(V["d_" + a])
        rk = within_rank(amp)
        out[f"med_{a}"] = np.nanmedian(amp.reshape(-1, T), 0)
        for q in q_list:
            big = rk >= 1 - q
            n = big.sum((0, 1)).astype(float)
            out[f"diag_{a}_q{int(round(100 * q)):02d}"] = ((big & (phi >= 40)).sum((0, 1)) / np.maximum(n, 1))
            out[f"axis_{a}_q{int(round(100 * q)):02d}"] = ((big & (phi < 15)).sum((0, 1)) / np.maximum(n, 1))
            out[f"angle_{a}_q{int(round(100 * q)):02d}"] = np.array(
                [np.nanmean(phi[:, :, t][big[:, :, t]]) if big[:, :, t].any() else np.nan for t in range(T)])
    # the Optimal's large jumps: change of the distance to the true cell over the jump, by direction
    amp, phi = polar(V["d_Joint"])
    rk = within_rank(amp)
    corr = np.full_like(V["dist"], np.nan)
    corr[:, :, 1:] = V["dist"][:, :, 1:] - V["dist"][:, :, :-1]
    corner = ((np.minimum(V["cell"][:, 0], 9 - V["cell"][:, 0]) <= 1)
              & (np.minimum(V["cell"][:, 1], 9 - V["cell"][:, 1]) <= 1))
    big = rk >= .9
    for nm, m in (("diag", big & (phi >= 40)), ("axis", big & (phi < 15)), ("all", big)):
        out[f"corr_{nm}"] = np.array([np.nanmean(corr[:, :, t][m[:, :, t]]) if m[:, :, t].any() else np.nan
                                      for t in range(T)])
        out[f"corner_{nm}"] = np.array([corner[np.nonzero(m[:, :, t])[1]].mean() if m[:, :, t].any() else np.nan
                                        for t in range(T)])
    out["dist_med"] = np.median(V["dist"].reshape(-1, T), 0)
    out["corr_every"] = np.nanmean(corr.reshape(-1, T), 0)
    # the grid: cell means split into the shared grid and the environment-specific distortion (plot_fig1c_mesh)
    import plot_fig1c_mesh as pm
    for a, _ in AGENTS:
        B = np.load(os.path.join(mm.SCAN, f"scan_{a}.npz"))["beliefs"][mm.ENVS].astype(np.float32)
        Mn = B.reshape(E, R * R, NS, T, 2 * R).mean(2).astype(np.float64)
        del B
        vc, ve, vi = pm.decompose(Mn)
        out[f"grid_{a}"], out[f"interf_{a}"] = vc, vi
        print("grid:", a, flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    return out


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

LCOLS = (("mi", "Joint split", r"$I(r_g; r_c)$ in the Optimal's belief", "nats"),
         ("fr", "Factorization regret", r"$\mathrm{FR}_t$, both variables", r"log$_{10}$ nats"),
         ("rev", "Entanglement", r"$\mathrm{rev}_t$, the update into $t$", r"log$_{10}$ nats"),
         ("nov", "Novelty of the observation", r"$-\Sigma_i \log p(o_{ti} \mid o_{<t})$", "nats"),
         ("pr", "Effective dim., activity", r"$\mathrm{PR}(u_t)$, Trained RNN", "dimensions"),
         ("band_x", "Cross-band share", r"of $\Delta W_t$, Trained RNN", "share"))
LEVEL_LABEL = (r"0 - 15$^\circ$ (one variable)", r"15 - 30$^\circ$", r"30 - 40$^\circ$",
               r"40 - 45$^\circ$ (both: diagonal)")
WIN_COL = {"early": "#41ab5d", "late": "k"}


def _style(ax, FS):
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)


def _letter(fig, ax, s, FS, W_, H_, dx=.55, dy=.42):
    x0, y0, w, h = ax.get_position().bounds
    fig.text(x0 - dx / W_, y0 + h + dy / H_, r"$\mathbf{%s}$" % s, fontsize=FS["label"] + 3, va="baseline")


def figure_lags_time(q=.10):
    """the Optimal's large-for-their-step jumps by coupling level, early / mid / late, and the diagonal minus one-axis
    contrast over arrival step x lag -> fig1c_jump_lags_time.png"""
    from main.utils import apply_fig_style, fig_path
    from matplotlib.lines import Line2D
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    Z = lags_time(q)
    PW, PH, GX, GY, X0, TOP, GY4, BOT = 2.3, 1.55, .82, .55, 1.75, 1.75, 1.45, .62
    nc = len(LCOLS)
    W_ = X0 + nc * PW + (nc - 1) * GX + .5
    H_ = TOP + 3 * PH + 2 * GY + GY4 + PH + BOT
    fig = plt.figure(figsize=(W_, H_))
    fig.text(.5, 1 - .28 / H_, f"Large jumps (the top {100 * q:.0f}% of their step), aligned on the jump, by its "
             "coupling "
             r"angle: the coupling picture changes with time", ha="center", va="center", fontsize=FS["label"] + 3)
    fig.text(.5, 1 - .68 / H_, f"alias scan, 39 environments x 5,000 streams: the Optimal's {int(Z['n_events']):,} "
             "jumps that are in the top 10% of their environment and step; each measure minus typical (the same "
             "cell's 50 streams at the same step);\nmatched for amplitude across the angles (within-step rank); bands "
             "5 - 95% over "
             "environments resampled; row 4: diagonal minus one-axis jumps at every arrival step",
             ha="center", va="center", fontsize=FS["tick"])
    keys_ = [Line2D([], [], color=col, lw=1.8, marker="o", ms=3, label=LEVEL_LABEL[g])
             for g, (_, _, col) in enumerate(LEVELS)]
    fig.legend(handles=keys_, loc="center", bbox_to_anchor=(.5, 1 - 1.13 / H_), ncol=4, frameon=False,
               fontsize=FS["tick"] + .5, handlelength=2.4, columnspacing=2.4)
    for c, (k, name, form, unit) in enumerate(LCOLS):
        lo_ = min(np.nanmin(Z[f"b_{wn}_{k}"][0]) for wn in WIN_LAGS)
        hi_ = max(np.nanmax(Z[f"b_{wn}_{k}"][1]) for wn in WIN_LAGS)
        lo_, hi_ = min(lo_, 0), max(hi_, 0)
        pad = .07 * (hi_ - lo_)
        x = X0 + c * (PW + GX)
        for r, wn in enumerate(WIN_LAGS):
            y = H_ - TOP - PH - r * (PH + GY)
            ax = fig.add_axes([x / W_, y / H_, PW / W_, PH / H_])
            M, B = Z[f"m_{wn}_{k}"], Z[f"b_{wn}_{k}"]
            for g, (_, _, col) in enumerate(LEVELS):
                ax.fill_between(LAGS, B[0][g], B[1][g], color=col, alpha=.17, lw=0)
                ax.plot(LAGS, M[g], color=col, lw=1.6, marker="o", ms=2.4)
            ax.axhline(0, color="0.6", lw=.8, ls=":")
            ax.axvline(0, color="k", lw=.6, ls=":")
            ax.set_xlim(LAGS[0] - .3, LAGS[-1] + .3), ax.set_xticks([-2, 0, 2, 4, 6])
            ax.set_ylim(lo_ - pad, hi_ + pad)
            ax.set_ylabel(unit, fontsize=FS["tick"] + .5, labelpad=2)
            _style(ax, FS)
            if r < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("steps from the jump", fontsize=FS["tick"] + .5, labelpad=1)
            if r == 0:
                ax.set_title(f"{name}\n{form}", fontsize=FS["tick"] + 1.5, pad=4)
            if c == 0:
                a_, b_ = WIN_LAGS[wn]
                n_ = sum(int(Z[f"n_{wn}_{g}"]) for g in range(len(LEVELS)))
                fig.text((x - 1.18) / W_, (y + PH / 2) / H_, f"{wn}\nsteps {a_} - {b_}\n{n_:,} jumps", ha="center",
                         va="center", fontsize=FS["label"], linespacing=1.4)
    # row 4: diagonal minus one-axis over arrival step x lag
    y4 = BOT
    fig.text(.5, (y4 + PH + .72) / H_, r"Diagonal (40 - 45$^\circ$) minus one-axis (0 - 15$^\circ$) jumps, at every "
             "arrival step (x) and step from the jump (y); dotted: the windows of rows 1 - 3",
             ha="center", va="center", fontsize=FS["label"] + .5)
    xe, ye = np.arange(2.5, T + 1), np.arange(LAGS[0] - .5, LAGS[-1] + 1)
    for c, (k, name, form, unit) in enumerate(LCOLS):
        ax = fig.add_axes([(X0 + c * (PW + GX)) / W_, y4 / H_, PW / W_, PH / H_])
        Mx = Z[f"map_{k}"][2:]                                                     # arrival steps 3 .. 30
        lim = np.nanpercentile(np.abs(Mx), 99)
        ax.set_facecolor("0.88")
        im = ax.pcolormesh(xe, ye, Mx.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="flat", rasterized=True)
        for b in (6.5, 16.5):
            ax.axvline(b, color="k", lw=.7, ls=":")
        ax.axhline(-.5, color="k", lw=.5, alpha=.4)
        ax.set_xlim(2.5, T + .5), ax.set_xticks([3, 10, 20, 30]), ax.set_yticks([-2, 0, 2, 4, 6])
        ax.set_xlabel("arrival step of the jump", fontsize=FS["tick"] + .5, labelpad=1)
        if c == 0:
            ax.set_ylabel("steps from the jump", fontsize=FS["tick"] + .5, labelpad=2)
        _style(ax, FS)
        ax.set_title(name, fontsize=FS["tick"] + 1.5, pad=3)
        cb = fig.colorbar(im, ax=ax, fraction=.07, pad=.03)
        cb.set_label(unit, fontsize=FS["tick"] - .5)
        cb.ax.tick_params(labelsize=FS["tick"] - 1.5)
    path = fig_path("fig1c_jump_lags_time.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("saved", path)
    return path


XCOLS = (("nov", "Novelty (input)", r"$-\Sigma_i \log p(o_{ti} \mid o_{<t})$"),
         ("l2", "Norm", r"$\|u_t\|$"),
         ("pr", "Effective dim.", r"$\mathrm{PR}(u_t)$"),
         ("turn", "Activity turn", r"$1 - \cos(u_t, u_{t-1})$"),
         ("coh", "Credit coherence", r"$\|\Sigma_k \delta_k\|^2 / \Sigma_k \|\delta_k\|^2$"),
         ("dW", r"$\Delta W$ energy", r"$\|\Delta W_t\|^2$"),
         ("band_x", "Cross-band share", r"of $\Delta W_t$"),
         ("dim", r"$\Delta W$ effective dim.", r"$\mathrm{PR}(\Delta W_t)$"))


def _locked():
    """the locked dynamics_interference curves (gradflow_analysis.figure_series; harvest, steps 17 - 30) on taus
    -6 .. 6:
    {(net, control): {key: curve}}"""
    import pickle
    out = {}
    for net in ("trained", "echo"):
        p = os.path.join(ROOT, "main", "DATA", "experimental", "gradflow_ctx", f"figure_series_{net}.pkl")
        st = pickle.load(open(p, "rb"))
        for ctrl, pct in ((None, 10), ("novelty", 1)):
            r = st[(ctrl, pct)]["r"]
            out[net, ctrl] = {k: np.asarray(r[k])[2:15] for k in r if np.asarray(r[k]).ndim == 1}
    return out


# the ESN's update measures barely move (fig1c_scan_credit, late window): effect sizes next to the r
ESN_NOTE = {"dim": "ESN: rank-one update;\nvalues 1.000 - 1.021 (p1 - p99),\nwithin-stream SD 0.4% of the level",
            "coh": "ESN: 3.20 - 3.71 (p1 - p99),\nwithin-stream SD 1.2%",
            "band_x": "ESN: about 0 (-0.016 - 0.014):\nits error sits in one band"}


def figure_xcorr():
    """the locked cross-correlation statistic on the scan, early vs late, raw vs minus typical, beyond the
    observation; the ESN on the same streams -> fig1c_xcorr_early_late.png"""
    from main.utils import apply_fig_style, fig_path
    from matplotlib.lines import Line2D
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    Z = xcorr_time()
    L = _locked()
    PW, PH, GX, GY, X0, TOP, BOT = 1.95, 1.6, .5, 1.3, 2.05, 1.95, .65
    nc = len(XCOLS)
    W_ = X0 + nc * PW + (nc - 1) * GX + .35
    H_ = TOP + 3 * PH + 2 * GY + BOT
    fig = plt.figure(figsize=(W_, H_))
    fig.text(.5, 1 - .28 / H_, "The entanglement cross-correlation is the same early and late once each cell's typical "
             "course is removed; about half of it goes through the observation, and only the Trained RNN's has a part "
             "beyond it", ha="center", va="center", fontsize=FS["label"] + 2)
    fig.text(.5, 1 - .7 / H_, "alias scan, 39 environments x 5,000 streams (row 3: the 1,000 per environment whose ESN "
             "updates exist); the locked statistic: each series z-scored within a stream over a 14-step window, the "
             "mean "
             r"of $x_{t+\tau}\,\mathrm{rev}_t$ ($\tau$ 0 = mean Pearson r);" + "\nearly window: steps 3 - 16, late: "
             "steps 17 - 30 (the locked figure's); 'minus typical': the same cell's streams' mean at that step removed "
             "first; beyond the observation: novelty and features changed fitted out of entanglement", ha="center",
             va="center", fontsize=FS["tick"])
    rows = (("Trained RNN, the whole curve", "trained"), ("Trained RNN, beyond the observation", "trained"),
            ("ESN, same streams", "echo"))
    lock = {0: L["trained", None], 1: L["trained", "novelty"], 2: L["echo", None]}
    lo_, hi_ = -.3, .5
    for r, (rname, net) in enumerate(rows):
        y = H_ - TOP - PH - r * (PH + GY)
        fig.text((X0 - 1.32) / W_, (y + PH / 2) / H_, rname.replace(", ", ",\n"), ha="center", va="center",
                 fontsize=FS["label"], linespacing=1.4)
        for c, (k, name, form) in enumerate(XCOLS):
            ax = fig.add_axes([(X0 + c * (PW + GX)) / W_, y / H_, PW / W_, PH / H_])
            if k in lock[r]:
                ax.plot(TAUS, lock[r][k], color="0.55", lw=3.2, alpha=.45, solid_capstyle="round")
            for wn in ("early", "late"):
                col = WIN_COL[wn] if net == "trained" else {"early": "#fdae6b", "late": "#d94801"}[wn]
                if r == 0:
                    ax.plot(TAUS, Z[f"trained|raw|{wn}|ent|{k}"], color=col, lw=1.2, ls=(0, (3, 2)))
                    ax.plot(TAUS, Z[f"trained|dep|{wn}|ent|{k}"], color=col, lw=1.7)
                elif r == 1:
                    ax.plot(TAUS, Z[f"trained|dep|{wn}|beyond_jumps|{k}"], color=col, lw=1.2, ls=(0, (1, 1.5)))
                    ax.plot(TAUS, Z[f"trained|dep|{wn}|beyond_obs|{k}"], color=col, lw=1.7)
                else:
                    ax.plot(TAUS, Z[f"echo|dep|{wn}|ent|{k}"], color=col, lw=1.7)
                    ax.plot(TAUS, Z[f"echo|dep|{wn}|beyond_obs|{k}"], color=col, lw=1.2, ls=(0, (3, 2)))
            ax.axhline(0, color="0.6", lw=.7, ls=":")
            ax.axvline(0, color="k", lw=.5, ls=":")
            ax.set_xlim(-6.3, 6.3), ax.set_xticks([-6, -3, 0, 3, 6]), ax.set_ylim(lo_, hi_)
            if c:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(r"r (entanglement at $t$)", fontsize=FS["tick"] + .5)
            ax.set_xlabel(r"$\tau$ (steps after entanglement)", fontsize=FS["tick"], labelpad=1)
            _style(ax, FS)
            if r == 0:
                ax.set_title(f"{name}\n{form}", fontsize=FS["tick"] + 1, pad=4)
            if net == "echo" and k in ESN_NOTE:
                ax.text(.97, .04, ESN_NOTE[k], transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=FS["tick"] - .5, color="#d94801", linespacing=1.2)
        # row legend
        if r == 0:
            h = [Line2D([], [], color=WIN_COL["late"], lw=1.7, label="late, minus typical"),
                 Line2D([], [], color=WIN_COL["late"], lw=1.2, ls=(0, (3, 2)), label="late, raw"),
                 Line2D([], [], color=WIN_COL["early"], lw=1.7, label="early, minus typical"),
                 Line2D([], [], color=WIN_COL["early"], lw=1.2, ls=(0, (3, 2)), label="early, raw"),
                 Line2D([], [], color="0.55", lw=3.2, alpha=.45, label="locked figure (harvest, steps 17 - 30)")]
        elif r == 1:
            h = [Line2D([], [], color=WIN_COL["late"], lw=1.7, label="late, beyond the observation"),
                 Line2D([], [], color=WIN_COL["late"], lw=1.2, ls=(0, (1, 1.5)), label="late, beyond the jumps"),
                 Line2D([], [], color=WIN_COL["early"], lw=1.7, label="early, beyond the observation"),
                 Line2D([], [], color=WIN_COL["early"], lw=1.2, ls=(0, (1, 1.5)), label="early, beyond the jumps"),
                 Line2D([], [], color="0.55", lw=3.2, alpha=.45, label="locked figure, novelty control")]
        else:
            h = [Line2D([], [], color="#d94801", lw=1.7, label="late"),
                 Line2D([], [], color="#d94801", lw=1.2, ls=(0, (3, 2)), label="late, beyond the observation"),
                 Line2D([], [], color="#fdae6b", lw=1.7, label="early"),
                 Line2D([], [], color="#fdae6b", lw=1.2, ls=(0, (3, 2)), label="early, beyond the observation"),
                 Line2D([], [], color="0.55", lw=3.2, alpha=.45, label="locked figure (ESN)")]
        fig.legend(handles=h, loc="center", bbox_to_anchor=(.5, (y + PH + (.62 if r == 0 else .48)) / H_), ncol=5,
                   frameon=False,
                   fontsize=FS["tick"], handlelength=2.6, columnspacing=1.8)
    path = fig_path("fig1c_xcorr_early_late.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("saved", path)
    return path


def figure_two_processes():
    """the synthesis: process A (state-driven diagonal jumps, early) and process B (observation-driven entanglement
    responses, every step) on one time axis -> fig1c_two_processes.png"""
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    TL = timeline()
    Z10, Z02 = lags_time(.10), lags_time(.02)
    XC = xcorr_time()
    t = np.arange(1, T + 1)
    ts = np.arange(3, T + 1)                                                        # arrival steps with lag -2 .. +6
    i_m1, i_0 = int(np.flatnonzero(LAGS == -1)[0]), int(np.flatnonzero(LAGS == 0)[0])
    PW, PH, GX, GY, X0, TOP, BOT = 2.9, 2.0, 1.05, 1.7, .9, 1.85, .62
    W_ = X0 + 4 * PW + 3 * GX + .35
    H_ = TOP + 2 * PH + GY + BOT
    fig = plt.figure(figsize=(W_, H_))
    fig.text(.5, 1 - .28 / H_, "Two processes: (A) state-driven diagonal jumps unfold the grid early; (B) every "
             "observation's entanglement drives the same network response at every step", ha="center", va="center",
             fontsize=FS["label"] + 3)
    fig.text(.5, 1 - .62 / H_, "alias scan, 39 environments x 5,000 streams x 30 steps (g: the 1,000 streams per "
             "environment with ESN updates); large jumps = the top 10% (thin dashed: the top 2%) of each "
             "environment's step", ha="center", va="center", fontsize=FS["tick"])
    rows_y = [H_ - TOP - PH - r * (PH + GY) for r in (0, 1)]
    for r, lab in enumerate(("A    state-driven diagonal jumps: the early unfolding",
                             "B    observation-driven entanglement responses: every step")):
        fig.text(.25 / W_, (rows_y[r] + PH + .85) / H_, lab, ha="left", va="center", fontsize=FS["label"] + 2.5,
                 color="#5e3c99" if r == 0 else "#1a7837")

    def ax_at(r, c, frac=(0, 1)):
        y0 = rows_y[r] + frac[0] * PH
        return fig.add_axes([(X0 + c * (PW + GX)) / W_, y0 / H_, PW / W_, (frac[1] - frac[0]) * PH / H_])

    def head(ax, name, form):
        ax.annotate(name, (.5, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points",
                    ha="center", va="baseline", fontsize=FS["label"] + .5)
        ax.annotate(form, (.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points", ha="center",
                    va="baseline", fontsize=FS["tick"] - .3, color="0.25")

    def steps_axis(ax, lo=1, label="step $t$"):
        ax.set_xlim(lo - .5, T + .5), ax.set_xticks([1, 10, 20, 30] if lo == 1 else [lo, 10, 20, 30])
        if label:
            ax.set_xlabel(label, labelpad=1)
        _style(ax, FS)

    # a: which way the large jumps go
    ax = ax_at(0, 0)
    for a, nm in AGENTS:
        ax.plot(t[1:], TL[f"diag_{a}_q10"][1:], color=COL[a], lw=2.4 if a == "Trained" else 1.5,
                alpha=.5 if a == "Trained" else 1, label=nm)
    ax.axhline(5 / 45, color="0.5", lw=.8, ls=":")
    ax.text(29.5, 5 / 45 + .008, "no preferred direction", ha="right", va="bottom", fontsize=FS["tick"] - 1,
            color="0.4")
    ax.set_ylim(0, None), ax.set_ylabel("share of the large jumps")
    steps_axis(ax, 2, "arrival step $t$")
    head(ax, "Large jumps that are diagonal", r"$\varphi_t \geq 40^\circ$: both variables move alike")
    ax.legend(fontsize=FS["tick"] - .5, frameon=False, loc="upper right")
    _letter(fig, ax, "a", FS, W_, H_)
    # b: the state before a diagonal jump
    ax = ax_at(0, 1)
    for k, col, lab in (("fr", "#b2182b", "factorization regret"), ("mi", "#5e3c99", "joint split $I(r_g; r_c)$"),
                        ("rev", "#2166ac", "entanglement")):
        ax.plot(ts, Z10[f"map_{k}"][2:, i_m1], color=col, lw=1.8, label=lab)
        ax.plot(ts, Z02[f"map_{k}"][2:, i_m1], color=col, lw=1.0, ls=(0, (3, 2)))
    ax.axhline(0, color="0.6", lw=.8, ls=":")
    ax.set_ylabel("diagonal minus one-axis\n(log$_{10}$ nats; MI: nats)", fontsize=FS["tick"] + .5)
    steps_axis(ax, 3, "arrival step $t$")
    head(ax, "The step before a diagonal jump", r"at $t - 1$: diagonal minus one-axis large jumps into $t$")
    ax.legend(fontsize=FS["tick"] - .5, frameon=False, loc="upper right")
    _letter(fig, ax, "b", FS, W_, H_)
    # c: the Trained RNN at diagonal jumps (two measures, two units)
    axc1 = ax_at(0, 2, (.56, 1))
    axc2 = ax_at(0, 2, (0, .44))
    for ax, k, L_, col, unit in ((axc1, "band_x", i_0, "#1f77b4", "share"), (axc2, "pr", i_m1, "#8c564b", "dims")):
        ax.plot(ts, Z10[f"map_{k}"][2:, L_], color=col, lw=1.8)
        ax.plot(ts, Z02[f"map_{k}"][2:, L_], color=col, lw=1.0, ls=(0, (3, 2)))
        ax.axhline(0, color="0.6", lw=.8, ls=":")
        ax.set_ylabel(unit, fontsize=FS["tick"] + .5, labelpad=2)
        steps_axis(ax, 3, None if ax is axc1 else "arrival step $t$")
    K4 = lags_k4()
    axc1.plot(ts, K4["map_band_x_k4"][2:, i_0], color="#08306b", lw=1.5, ls=(0, (1, 1.2)))
    axc1.set_ylim(-2.05, .15), axc1.set_yticks([-1.5, -1, -.5, 0])
    axc1.set_xticklabels([])
    axc1.text(.97, .06, r"cross-band share of $\Delta W_t$, at $t$" + "\n"
              + r"dotted: credit window fixed at $k \leq 4$",
              transform=axc1.transAxes, fontsize=FS["tick"] - .5, color="#1f77b4", va="bottom", ha="right",
              linespacing=1.15)
    axc2.text(.97, .06, r"activity eff. dim. $\mathrm{PR}(u)$, at $t - 1$", transform=axc2.transAxes,
              fontsize=FS["tick"] - .5, color="#8c564b", va="bottom", ha="right")
    head(axc1, "The Trained RNN at diagonal jumps", "diagonal minus one-axis: bands cancel, activity narrows")
    _letter(fig, axc1, "c", FS, W_, H_)
    # d: the grid's environment-specific distortion
    ax = ax_at(0, 3)
    for a, nm in AGENTS:
        vc, vi = TL[f"grid_{a}"], TL[f"interf_{a}"]
        ax.plot(t, vi / (vc + vi), color=COL[a], lw=2.4 if a == "Trained" else 1.5, alpha=.5 if a == "Trained" else 1,
                label=nm)
    ax.set_ylim(0, None), ax.set_ylabel("share of the grid's variance")
    steps_axis(ax)
    head(ax, "Environment-specific distortion of the grid", r"cell $\times$ environment / (cell + cell $\times$ "
         "environment)")
    _letter(fig, ax, "d", FS, W_, H_)
    # e: what entanglement goes with (sliding windows)
    mid = XC["slide_mid"]
    xl = "step $t$ (centre of an 8-step window)"
    ax = ax_at(1, 0)
    for k, col, lab in (("nov", "#1a7837", "novelty of the observation"), ("chg", "#a6dba0", "features changed"),
                        ("amp", "k", r"jump size $\log|\Delta\mu_t|$ (Optimal)"),
                        ("fr", "#b2182b", "factorization regret")):
        ax.plot(mid, XC[f"trained|dep|slide|ent|{k}"][:, 0], color=col, lw=1.8, label=lab)
    ax.axhline(0, color="0.6", lw=.8, ls=":")
    ax.set_ylim(-.38, .55), ax.set_ylabel(r"r with entanglement ($\tau = 0$)")
    steps_axis(ax, 1, xl)
    head(ax, "What entanglement goes with", "within-stream r, minus typical, sliding 8-step windows")
    ax.legend(fontsize=FS["tick"] - .5, frameon=False, loc="lower right")
    _letter(fig, ax, "e", FS, W_, H_)
    # f: the Trained RNN's response
    ax = ax_at(1, 1)
    fcols = (("l2", "#9467bd", r"norm $\|u_t\|$"), ("pr", "#8c564b", r"eff. dim. $\mathrm{PR}(u_t)$"),
             ("turn", "#e377c2", "activity turn"), ("coh", "#2ca02c", "credit coherence"),
             ("band_x", "#1f77b4", "cross-band share"), ("dim", "#7f7f7f", r"$\Delta W$ eff. dim."))
    for k, col, lab in fcols:
        ax.plot(mid, XC[f"trained|dep|slide|ent|{k}"][:, 0], color=col, lw=1.8, label=lab)
    ax.plot(mid, XC["trained|raw|slide|ent|band_x"][:, 0], color="#1f77b4", lw=1.1, ls=(0, (3, 2)))
    ax.axhline(0, color="0.6", lw=.8, ls=":")
    ax.set_ylim(-.62, .5), ax.set_ylabel(r"r with entanglement ($\tau = 0$)")
    steps_axis(ax, 1, xl)
    head(ax, "The Trained RNN's response", "solid: minus typical; dashed: cross-band share, raw")
    ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="lower center", ncol=2, columnspacing=1.0,
              handlelength=1.5)
    _letter(fig, ax, "f", FS, W_, H_)
    # g: beyond the observation, Trained vs ESN
    ax = ax_at(1, 2)
    for k, col, lab in (("l2", "#9467bd", "norm"), ("coh", "#2ca02c", "credit coherence"),
                        ("band_x", "#1f77b4", "cross-band share")):
        ax.plot(mid, XC[f"trained10|dep|slide|beyond_obs|{k}"][:, 0], color=col, lw=1.8, label=lab)
        ax.plot(mid, XC[f"echo|dep|slide|beyond_obs|{k}"][:, 0], color=col, lw=1.2, ls=(0, (3, 2)))
    ax.axhline(0, color="0.6", lw=.8, ls=":")
    ax.set_ylim(-.3, .5), ax.set_ylabel(r"r beyond the observation ($\tau = 0$)")
    steps_axis(ax, 1, xl)
    head(ax, "Beyond the observation's surprise", "solid: Trained RNN (learned); dashed: ESN (sensory only)")
    ax.legend(fontsize=FS["tick"] - .5, frameon=False, loc="upper right")
    _letter(fig, ax, "g", FS, W_, H_)
    # h: what the large jumps do
    ax = ax_at(1, 3)
    ax.plot(t[1:], TL["corr_diag"][1:], color="#542788", lw=1.8,
            label=r"diagonal large jumps ($\varphi \geq 40^\circ$)")
    ax.plot(t[1:], TL["corr_axis"][1:], color="#b35806", lw=1.8, label=r"one-axis large jumps ($\varphi < 15^\circ$)")
    ax.plot(t[1:], TL["corr_every"][1:], color="0.55", lw=1.2, ls=(0, (3, 2)), label="every stream")
    ax.axhline(0, color="0.6", lw=.8, ls=":")
    ax.set_ylabel("change of the distance to the\n" + r"true cell (cells; $< 0$: toward it)", fontsize=FS["tick"] + .5)
    steps_axis(ax, 2, "arrival step $t$")
    head(ax, "What the large jumps do (Optimal)", r"$d_t - d_{t-1}$, $d_t = |\mu_t - $ the true cell$|$")
    ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="lower right")
    _letter(fig, ax, "h", FS, W_, H_)
    path = fig_path("fig1c_two_processes.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    args = sys.argv[1:] or ["numbers"]
    for arg in args:
        if arg == "numbers":
            print_lags_time(lags_time())
            print_xcorr(xcorr_time())
            timeline()
        elif arg == "lags":
            figure_lags_time()
        elif arg == "xcorr":
            figure_xcorr()
        elif arg == "synthesis":
            figure_two_processes()
        elif arg == "all":
            figure_lags_time(), figure_xcorr(), figure_two_processes()
