#!/usr/bin/env python3
"""fr_theory_additions.py -- the threshold q* and the panels proposed for the theory figures (fig_fr_threshold, fig_fr_candidates).

John 11-11: "isnt q^* an analytical quantity? Why cant we get it? Id like to be able to show the condition where more episodes
leads to a decaying probability that the expected FR will ever dip.  Can you show a few figures of everything that you think should
be added? ... keep plots as simple and minimal as possible, clarifying what is analytical and what is numerical.  And please choose
values that emphasize differences so that important effects are not subtle."

WHAT q* IS, AND WHY IT IS EXACT ONLY IN THE TOY
  Corollary 1: with U_t, W_t the expected FR_t of the episodes at untrapped / trapped joint realizations,
      q*_t = (U_t - U_{t+1})^+ / ((U_t - U_{t+1})^+ + (W_{t+1} - W_t)),   q* = max_t q*_t,
  and the expected curve (1 - q) U_t + q W_t decreases somewhere exactly when q < q*.  q* is a deterministic function of the
  likelihood tables, but unlike Delta or the law of the step X it is not a ONE-STEP quantity: U_t and W_t are expectations over all
  observation histories of t steps (the transient).  With one binary observable a history is summarised by one count, t + 1 values:
  exact sums (the toy).  With the Gridworld's five observables it takes five counts, (t + 1)^5 values per step: beyond a few dozen
  steps U_t and W_t can only be estimated by simulation.  So: toy q* exact; Gridworld q* numerical (with a standard error).

COROLLARY 2 (noise or real), which the panels illustrate: with mu_t, sigma_t^2 the mean and variance of one episode's increment
  FR_{t+1} - FR_t, an average over N episodes moves against the sign of mu_t with probability <= sigma_t^2 / (N mu_t^2) (Chebyshev).
  Hence, over any horizon T:  q > q*  (mu_t > 0 at every step)  =>  P(the average decreases somewhere) <= sum_t p_t(N) -> 0;
                              q < q*  (mu_t < 0 at some step)   =>  P(the average decreases somewhere) >= max_t p_t(N) -> 1,
  with p_t(N) the probability that the average decreases at step t.

11-11 evening: every panel below now lives in fr_theory_figure.py (fig_fr_theory_all, all theory panels in one figure, to be
trimmed by John); `figure` here still writes the two standalone figures the v13 tex refers to.  New cache toy_fine: the union bound
of panel t of that figure on a fine grid of N (5 min on 12 cores).
11-12 (John: "add R=2 to panel k", then "can we try 2,3,4,5?"): new cache opt_more, fr_theory.build_optimum's points (growth rate vs.
observables) at R = 2, 3, 4 realizations per variable, for fig_fr_theory_all j (`data --only=opt_more`; R = 2 took 9 min on 13 workers,
R = 3 and 4 13 min on 19; a point depends only on its own seeds: the R = 2 points were kept when 3 and 4 were added, and one of them,
rebuilt, came out identical).  E[Delta] at d_o = 1 / 5 / 30: R = 2 .0061 / .0017 / 0; R = 3 .023 / .018 / .0004; R = 4 .034 / .036 /
.0067 (q .28 / .035 / 0; .42 / .14 / .003; .53 / .21 / .029); simulated slopes within 0.7 s.e. of it everywhere.

11-13 night (John: "avoid having to write 'toy' example"): new cache small (`data --only=small`, 35 s), row 1 and the critical-step panel
of fig_fr_theory_all from the Cognitive Gridworld itself with one observable and R = 3: build_small (selection rule, exact curves,
threshold, simulation).  With seed 3131: trap-free set 59 (no trap at lambda = 1, 2, 3), one-trap set 1; q* = .003244 at step 435.

PANELS  (lines = analytical, markers = simulation, as in fr_theory.py)
  fig_fr_threshold  (proposed third figure; 5.5 in wide = the manuscript's text width, drawn at print size for an upright figure)
    (a) toy, exact + simulation: P(an N-episode average decreases somewhere in the T_TOY steps of fig_fr_theory d) against N, for
        trapped fractions q below and above q*.  Lines: the two bounds above, from the EXACT law of the increment at every step
        (FFT convolution; Lugannani-Rice saddlepoint of the same law above N_FFT_MAX episodes; they agree to 3e-5 where both were
        computed), evaluated at the grid TOY_NS and at exactly the simulated N.  Markers: simulated averages.
    (b) Gridworld, simulation against tables: the threshold q*_t between steps of a log grid (markers, bootstrap SEM over sets),
        at three likelihood temperatures, against the trapped fraction q of the same interaction sets (lines, tables alone).
    (c) Gridworld, simulation: as (a) over the first T_POOL steps, for q = 0 (traps removed), q = 0.1 and the Gridworld's own q
        (averages of N paths drawn from a pool of full paths, untrapped and trapped in proportion q).
  fig_fr_candidates  (other panels that could be added; 9 in wide like the rotated figures)
    (A) toy, exact: P(the average decreases at the critical step t*, where q*_t is largest) against N: -> 1 below q*, -> 0 above.
    (B) Gridworld version of fig_fr_theory d: expected curve of populations in which a fraction q of the episodes is trapped (the
        Gridworld's untrapped and trapped episodes re-mixed; simulation), with the simulated q* and the Gridworld's own q.
    (C) Gridworld: P(an N-episode average decreases at step t) against t, simulated, against the asymptotic law of fig_fr_averages
        (tables alone): how soon the law applies (the task has 30 steps).
    (D) Gridworld: growth rate against the trapped fraction q of the interaction set: E[Delta] (tables alone) and simulated slopes
        (from the cache of fig_fr_theory f).

RESULTS (11-11; `numbers` prints them)
  toy: q* = 0.004114 at step 152.  P(dip in 3,000 steps): q*/2 -> 1.00 at every N up to 1e8; 1.1 q* -> .98 at 3e6, .17 at 1e7, 0 at
  3e7; 2 q* -> .03 at 1e6; 27 q* -> 0 at 1e5; every simulated value lies within its two bounds (+- 2 SEM).
  Gridworld: q* (windows of 3 grid steps) = .0033 / .0032 / .0028 at lambda = 1 / 2 / 3 against q = .50 / .47 / .51 (q / q* ~ 150);
  the untrapped component peaks at .50 / .44 / .46 nats (t = 708 / 56 / 18) and is still .32 / .24 / .19 nats at t = 10,000.
  P(dip in the first 250 steps): q = 0 -> 1 at every N up to 30,000; q = .1 -> .96 at 1,000, .17 at 3,000, 0 at 10,000; the
  Gridworld's own q -> .99 at 100, .32 at 300, 0 at 1,000.  Law of dips against the step (N = 10 / 30 / 100: .293 / .168 / .038):
  reached by t ~ 15; before that dips are rarer than the law.

usage (eureka env; caches in main/DATA/experimental/fr_theory/additions):
    python fr_theory_additions.py data [--quick]     ~45 min on 12 cores (FR_WORKERS=8 to use fewer; same results)
    python fr_theory_additions.py figure [--quick]   seconds
    python fr_theory_additions.py numbers            the values behind the statements about q* (text of the theory block)
"""
import os, pickle, sys, time
import numpy as np
from scipy.stats import norm

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import fr_theory as F                                                             # noqa: E402

OUT = os.path.join(ROOT, "main", "DATA", "experimental", "fr_theory", "additions")
CFG = dict(
    WORKERS=int(os.environ.get("FR_WORKERS", 12)),            # worker processes (results do not depend on it: see CHUNKS)
    CHUNKS=12, LAM=2.0,                                       # the Gridworld builds are cut into this many seeded pieces, whatever WORKERS is
    # ---- toy (the two tables of fig_fr_theory a, b; mixture of d)
    T_TOY=3000,                                               # the steps shown in fig_fr_theory d
    TOY_RATIOS=(0.5, 1.1, 1.5, 2.0, 5.0, 27.0),               # trapped fraction q as a multiple of q* (27 q* = 1/9: every set is table b)
    TOY_NS=tuple(float(v) for v in np.round(10 ** np.arange(3.0, 8.01, 0.25), 6)),     # N of the theory lines
    TOY_SIM_NS=(1e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8), TOY_PATHS=240,           # N and averages per point of the simulation
    N_FFT_MAX=1e6,                                            # exact FFT up to this N, saddlepoint of the same law above
    TOY_STEP_RATIOS=(0.0, 0.5, 0.9, 1.0, 1.1, 1.5, 5.0),      # panel A: q / q* at the critical step
    TOY_FINE={1.1: (3e6, 4e7), 2.0: (3e5, 2e6), 27.0: (2e4, 1.5e5)}, TOY_FINE_PER_DECADE=24,   # fig_fr_theory_all t: the union bound on a fine grid
                                                              # of N across the window where it falls from 1 to 0 (exact values, not interpolations)
    # ---- Gridworld
    GRID_NZ=2400, GRID_NEP=1000, GRID_TMAX=10_000, GRID_NT=81,        # components U_t, W_t on a log grid of steps (count-based)
    GRID_LAMS=(1.0, 2.0, 3.0),
    MIX_QS=(0.0, 0.001, 0.002, 0.02),                         # panel b: trapped fractions drawn besides q* and the Gridworld's q
    T_POOL=250, POOL_NZ=2004, POOL_NEP=500,                   # panel c: pool of full paths (the horizon of fig_fr_theory e)
    POOL_QS=(0.0, 0.1), POOL_NS=(10, 30, 100, 300, 1000, 3000, 10_000, 30_000), POOL_GROUPS=400,
    STEP_NZ=2000, STEP_NEP=300, STEP_NS=(10, 30, 100), STEP_NT=25,    # panel B
    # ---- fig_fr_theory_all v (11-11 late, John: more curves, a broader range, evenly spaced, structured values): bands of the trapped fraction
    QB_CENTERS=(0.2, 0.35, 0.5, 0.65, 0.8), QB_HALF=0.03, QB_NZ=200, QB_CHUNK=250, QB_MAX_DRAWS=150_000,   # sets per band, by drawing tables
    QB_EP=2000, QB_T0=10_000, QB_T=30, QB_NS=(10, 20, 50, 100, 200, 500, 1000, 2000, 5000),               # simulation (as build_thirds)
    QB_TUPLES=1000, QB_NTH=30, QB_NMAX=10_000,                                                            # theory: draws of R per N, log grid of N
    # ---- fig_fr_theory_all j (11-12, John: "add R=2 to panel k", then "can we try 2,3,4,5?"): growth rate vs. observables at more numbers
    OPT2_RS=(2, 3, 4), OPT2_NZ=F.CFG["OPT_NZ"], OPT2_NEP=F.CFG["OPT_NEP"],                                 # of realizations R, with the settings of fr_theory.build_optimum
    OPT2_FROM=tuple((k, F.CFG[k]) for k in ("OPT_DOS", "OPT_T1", "OPT_T2", "OPT_TACC", "LAM_MAIN")),      # (keyed: a change there rebuilds this)
    # ---- fig_fr_theory_all row 1 and its critical-step panel since 11-13 (John: "it looks best to avoid having to write 'toy' example"):
    #      the Cognitive Gridworld itself with SW_R realizations per variable and ONE observable, where every expectation is an exact sum
    SW_R=3, SW_SEED=3131, SW_LAMS=(1.0, 2.0, 3.0), SW_LAM=F.CFG["LAM_MAIN"], SW_NSTAT=2000,                # generator draws; temperatures; sets for the trap counts
    SW_T=500, SW_TMIX=3000, SW_TMAX=100_000, SW_PER_DECADE=40,                                            # steps: anatomy, threshold (every step), log grid
    SW_NEP=100_000, SW_SIM_STEPS=10, SW_SIM_PER_DECADE=3,                                                 # simulation: episodes per table; recorded steps (linear / log)
    SW_RATIOS=(0.5, 1.0, 1.1, 2.0), SW_NS=tuple(float(v) for v in np.round(10 ** np.arange(2.0, 9.01, 0.25), 6)),   # critical step: q / q*; N of the lines
    SW_SIM_NS=tuple(float(v) for v in np.round(10 ** np.arange(2.0, 9.01, 0.5), 6)), SW_PATHS=2000,       # ... N and averages per point of its simulation
)
QUICK = dict(T_TOY=600, TOY_PATHS=40, TOY_SIM_NS=(1e3, 1e4, 1e5, 1e6), GRID_NZ=120, GRID_NEP=200, POOL_NZ=120, POOL_NEP=200,
             POOL_GROUPS=60, POOL_NS=(10, 100, 1000), STEP_NZ=120, STEP_NEP=100, GRID_LAMS=(2.0,), OPT2_NZ=20, OPT2_NEP=30,
             SW_NEP=5000, SW_PATHS=200, SW_NSTAT=200, SW_TMAX=10_000)


# ============================================================================================================
# 1. TOY: exact law of one episode's increment at every step; dips of N-episode averages
# ============================================================================================================
TOY = {}


def toy_setup(T):
    """The tables of fig_fr_theory a (k = 0) and b (k = 1 of 9), FR_t by count of ones for t = 0..T (exact), the exact expected
    curves, and the mixture threshold (as mixture_threshold in fr_theory.py: a fraction w = 9 q of the sets is table b)."""
    base = dict(R=3, alpha=1.2, beta=0.72)
    ells = [F.additive_table(**base), F.additive_table(**base, perturb=[(0, 2, +1)], gamma=0.6)]
    FRs, Es = [], []
    for ell in ells:
        m = ell.mean(1); rows = []; e = np.zeros(T + 1)
        for t in range(T + 1):
            lB, lN, _, P, _ = F._beliefs(ell, m, t); f = F._fr_by_count(lB, lN)
            if t <= 1: assert np.abs(f).max() < 1e-12; f = np.zeros(t + 1)      # FR_0 = FR_1 = 0 exactly (one binary observable)
            rows.append(f); e[t] = P.mean(axis=(1, 2)) @ f
        FRs.append(rows); Es.append(e)
    w_star, t_star = F.mixture_threshold(Es[0], Es[1])
    TOY.update(ell=ells, FR=FRs, E=Es, T=T, q_star=w_star / 9, t_star=t_star)
    return TOY


def toy_atoms(t):
    """One episode's increment FR_{t+1} - FR_t in each table: values (2 (t + 1),) and probabilities (uniform joint realization)."""
    out = []
    for ell, FRt in zip(TOY["ell"], TOY["FR"]):
        _, _, _, P, _ = F._beliefs(ell, ell.mean(1), t); f0, f1 = FRt[t], FRt[t + 1]
        out.append((np.r_[f1[1:] - f0, f1[:-1] - f0], np.r_[(P * ell[None]).mean(axis=(1, 2)), (P * (1 - ell[None])).mean(axis=(1, 2))]))
    return out


def mix_atoms(atoms, q):
    (xa, wa), (xb, wb) = atoms; w = 9.0 * q
    return np.r_[xa, xb], np.r_[(1 - w) * wa, w * wb]


def dip_fft(x, w, N):
    """P(sum of N independent draws of the law (x, w) < 0), by FFT convolution on a ring (a sum within half a cell of 0 counts
    half).  Linear splitting of each atom between two cells keeps the mean exact and inflates the variance by < 0.2%."""
    mu = float(w @ x); sd = float(np.sqrt(max(w @ x ** 2 - mu ** 2, 0.0)))
    if sd == 0.0: return 0.0 if mu >= 0 else 1.0
    z = np.sqrt(N) * mu / sd
    if abs(z) > 12.0: return 0.0 if z > 0 else 1.0                       # beyond 12 s.d.: below 1e-28 for these bounded increments
    n = int(2 ** max(14, np.ceil(np.log2(270.0 * np.sqrt(N)))))
    ring = 2.0 * (N * abs(mu) + 12.0 * np.sqrt(N) * sd + np.abs(x).max()); h = ring / n
    j = x / h; j0 = np.floor(j).astype(np.int64); f = j - j0
    pmf = np.bincount(j0 % n, weights=w * (1 - f), minlength=n) + np.bincount((j0 + 1) % n, weights=w * f, minlength=n)
    pN = np.fft.irfft(np.fft.rfft(pmf) ** int(N), n)
    return float(np.clip(pN[n // 2:].sum() + 0.5 * pN[0], 0.0, 1.0))


def saddle(x, w):
    """Saddlepoint of the law (x, w) at 0: s with k'(s) = 0 for k(s) = ln sum_j w_j exp(s x_j); returns s, k(s), k''(s) (None if
    the law lies on one side of 0).  P(sum of N < 0) then follows for every N (dip_saddle)."""
    if x.min() >= 0 or x.max() <= 0: return None
    def k1(s):
        a = s * x; e = w * np.exp(a - a.max()); return float(e @ x / e.sum())
    mu = float(w @ x); sgn = -1.0 if mu > 0 else 1.0; b = sgn
    while sgn * k1(b) < 0: b *= 2.0
    lo, hi = (b, 0.0) if sgn < 0 else (0.0, b)
    for _ in range(70):
        mid = 0.5 * (lo + hi); (lo, hi) = (mid, hi) if k1(mid) < 0 else (lo, mid)
    s = 0.5 * (lo + hi); w = w / w.sum(); e1 = np.expm1(s * x); zm1 = float(w @ e1); p = w * (1.0 + e1) / (1.0 + zm1)   # k = ln(1 + zm1), kept precise near s = 0
    m1 = float(p @ x); k2 = float(p @ (x - m1) ** 2)
    return s, float(np.log1p(zm1)), k2, float(p @ (x - m1) ** 3)


def dip_saddle(sp, N):
    """Lugannani-Rice formula for P(sum of N < 0) from saddle()."""
    s, k, k2, k3 = sp
    wv = np.sign(s) * np.sqrt(max(-2.0 * N * k, 0.0)); u = s * np.sqrt(N * k2)
    corr = k3 / (6.0 * k2 ** 1.5 * np.sqrt(N)) if abs(wv) < 0.1 else 1.0 / wv - 1.0 / u      # the limit of 1/w - 1/u as w -> 0 (no cancellation)
    return float(np.clip(norm.cdf(wv) + norm.pdf(wv) * corr, 0.0, 1.0))


def _toy_theory_chunk(task):
    """p[q, N, t - t_lo] = P(the average of N episodes decreases at step t) for the steps of one chunk; also both methods where both
    apply (the largest FFT N), for the validation printed by `numbers`."""
    cfg, ts, qs = task; Ns = cfg["TOY_NS"]; out = np.zeros((len(qs), len(Ns), len(ts))); both = []
    for it, t in enumerate(ts):
        atoms = toy_atoms(int(t))
        for iq, q in enumerate(qs):
            x, w = mix_atoms(atoms, q); keep = w > 0; x, w = x[keep], w[keep]; sp = None
            for iN, N in enumerate(Ns):
                if N <= cfg["N_FFT_MAX"]: out[iq, iN, it] = dip_fft(x, w, N)
                else:
                    if sp is None: sp = saddle(x, w) or "one-sided"
                    out[iq, iN, it] = (0.0 if x.min() >= 0 else 1.0) if sp == "one-sided" else dip_saddle(sp, N)
            if t % 50 == 7:
                sp2 = saddle(x, w)
                if sp2 is not None: both.append((int(t), float(q), dip_fft(x, w, cfg["N_FFT_MAX"]), dip_saddle(sp2, cfg["N_FFT_MAX"])))
    return out, both


def build_toy_theory(cfg):
    toy_setup(cfg["T_TOY"]); qs = sorted(set(cfg["TOY_RATIOS"]) | set(cfg["TOY_STEP_RATIOS"])); q_abs = [r * TOY["q_star"] for r in qs]
    steps = np.arange(1, cfg["T_TOY"]); chunks = np.array_split(steps, 8 * cfg["WORKERS"])
    res = F.run_parallel(_toy_theory_chunk, [(cfg, c, q_abs) for c in chunks if len(c)], cfg["WORKERS"], "toy: exact law of the dips")
    p = np.concatenate([r[0] for r in res], axis=2)                      # (q, N, step 1..T-1)
    return dict(ratios=qs, Ns=np.array(cfg["TOY_NS"]), p_max=p.max(axis=2), p_sum=p.sum(axis=2), p_star=p[:, :, TOY["t_star"] - 1],
                both=[b for r in res for b in r[1]], q_star=TOY["q_star"], t_star=TOY["t_star"], E=TOY["E"], T=cfg["T_TOY"])


def build_toy_theory_extra(cfg):
    """The bounds of fig_fr_threshold a at the simulated N that are not on the grid of the lines, so that markers and lines are
    compared at exactly the same N (the bounds are steep in N: at a neighbouring grid value they differ visibly)."""
    extra = tuple(float(N) for N in cfg["TOY_SIM_NS"] if not np.any(np.isclose(N, cfg["TOY_NS"])))
    return build_toy_theory(dict(cfg, TOY_NS=extra, TOY_STEP_RATIOS=())) if extra else None


def build_toy_fine(cfg):
    """The union bound sum_t p_t(N) (and max_t p_t(N)) of fig_fr_theory_all t on a fine grid of N (TOY_FINE_PER_DECADE per decade)
    within TOY_FINE[r] for each trapped fraction r q* drawn there, by the same exact computation as build_toy_theory."""
    out = {}
    for r, (lo, hi) in cfg["TOY_FINE"].items():
        Ns = tuple(float(v) for v in np.round(10 ** np.arange(np.log10(lo), np.log10(hi) + 1e-9, 1.0 / cfg["TOY_FINE_PER_DECADE"]), 3))
        th = build_toy_theory(dict(cfg, TOY_RATIOS=(r,), TOY_STEP_RATIOS=(), TOY_NS=Ns))
        out[r] = dict(Ns=th["Ns"], p_sum=th["p_sum"][0], p_max=th["p_max"][0])
    return out


def toy_bounds(D, r):
    """N (sorted), max_t p_t(N) and sum_t p_t(N) for the trapped fraction r q*: the grid of the lines merged with the simulated N."""
    th, ex = D["toy_theory"], D.get("toy_theory_extra"); i = th["ratios"].index(r); Ns, lo, hi = th["Ns"], th["p_max"][i], th["p_sum"][i]
    if ex is not None:
        j = ex["ratios"].index(r); Ns, lo, hi = np.r_[Ns, ex["Ns"]], np.r_[lo, ex["p_max"][j]], np.r_[hi, ex["p_sum"][j]]
    o = np.argsort(Ns); return Ns[o], lo[o], hi[o]


def _toy_sim(task):
    """Fraction of `paths` averages of N toy episodes (trapped fraction q) that decrease at some step t < T.  The N episodes are
    not followed one by one: for each table and joint realization, the NUMBER of episodes at each count of ones is updated with
    binomial draws (the same law, at a cost that does not grow with N)."""
    q, N, paths, seed, T = task; rng = np.random.default_rng(seed); N = int(N)
    nb = rng.binomial(N, 9.0 * q, size=paths); per = [rng.multinomial(N - nb, np.full(9, 1 / 9)), rng.multinomial(nb, np.full(9, 1 / 9))]
    state = []
    for tab in (0, 1):
        for j, p in enumerate(TOY["ell"][tab].ravel()):
            wd = int(min(T + 2, 2 * np.ceil(8.5 * np.sqrt(T * p * (1 - p))) + 12)); c = np.zeros((paths, wd), np.int64); c[:, 0] = per[tab][:, j]
            state.append([tab, float(p), c, 0, wd])
    prev = np.zeros(paths); dipped = np.zeros(paths, bool)
    for t in range(T + 1):
        val = np.zeros(paths)
        for tab, p, c, lo, wd in state:
            hi = min(lo + wd, t + 1); val += c[:, :hi - lo] @ TOY["FR"][tab][t][lo:hi]
        if t: dipped |= val < prev - 1e-6
        prev = val
        if t == T: break
        for st in state:
            tab, p, c, lo, wd = st; cols = np.flatnonzero(c.any(axis=0))
            if len(cols):
                a, b = cols[0], cols[-1] + 1; assert b < wd, "window too narrow"
                mv = rng.binomial(c[:, a:b], p); c[:, a:b] -= mv; c[:, a + 1:b + 1] += mv
            new = max(0, int((t + 1) * p - wd / 2)); d = new - lo
            if d > 0:
                assert not c[:, :d].any(), "window too narrow"; c[:, :-d] = c[:, d:]; c[:, -d:] = 0; st[3] = new
    return float(dipped.mean())


def build_toy_sim(cfg):
    toy_setup(cfg["T_TOY"]); tasks = [(r * TOY["q_star"], N, cfg["TOY_PATHS"], 17 + 1000 * i + j, cfg["T_TOY"])
                                      for i, r in enumerate(cfg["TOY_RATIOS"]) for j, N in enumerate(cfg["TOY_SIM_NS"])]
    res = F.run_parallel(_toy_sim, tasks, cfg["WORKERS"], "toy: simulated averages")
    return dict(ratios=list(cfg["TOY_RATIOS"]), Ns=np.array(cfg["TOY_SIM_NS"]), paths=cfg["TOY_PATHS"],
                dip=np.array(res).reshape(len(cfg["TOY_RATIOS"]), len(cfg["TOY_SIM_NS"])))


# ============================================================================================================
# 2. GRIDWORLD: components U_t, W_t (count-based), pool of full paths, dips against the step
# ============================================================================================================
def grid_times(cfg): return np.unique(np.round(np.logspace(0, np.log10(cfg["GRID_TMAX"]), cfg["GRID_NT"])).astype(int))


def _grid_chunk(task):
    """Per interaction set: sum of FR_t over its untrapped / trapped episodes at every step of the log grid (each episode is
    followed along the grid: counts of ones are extended binomially, the exact observers need nothing else)."""
    cfg, lam, seed, nz = task; TS = grid_times(cfg); rng = np.random.default_rng(seed); erng = np.random.default_rng(seed + 10 ** 6)
    S = np.zeros((nz, 2, len(TS))); cnt = np.zeros((nz, 2)); qz = np.zeros(nz); dl = np.zeros(nz)
    for z in range(nz):
        ell = F.checked_table(F.make_env(rng, lam), lam); lg = F.table_logs(ell); tab = F.one_step_table(ell); d_o, R, Rc = ell.shape
        n_ep = cfg["GRID_NEP"]; rg, rc = erng.integers(0, R, n_ep), erng.integers(0, Rc, n_ep); pr = ell[:, rg, rc].T; tr = tab["trapped"][rg, rc]
        n = np.zeros_like(pr); prev = 0
        for i, t in enumerate(TS):
            n = n + erng.binomial(int(t) - prev, pr); prev = int(t); f = F.fr_counts(n, int(t), *lg)
            S[z, 1, i] = f[tr].sum(); S[z, 0, i] = f[~tr].sum()
        cnt[z] = ((~tr).sum(), tr.sum()); qz[z] = tab["trapped"].mean(); dl[z] = tab["Delta"][tab["trapped"]].mean() if tab["trapped"].any() else 0.0
    return S, cnt, qz, dl


def build_grid(cfg):
    out = {}
    for lam in cfg["GRID_LAMS"]:
        per = int(np.ceil(cfg["GRID_NZ"] / (2 * cfg["CHUNKS"])))
        res = F.run_parallel(_grid_chunk, [(cfg, lam, 31_000 + 97 * k + int(1000 * lam), per) for k in range(2 * cfg["CHUNKS"])], cfg["WORKERS"],
                             f"Gridworld components, lambda = {lam:g}")
        out[lam] = dict(TS=grid_times(cfg), S=np.concatenate([r[0] for r in res]), cnt=np.concatenate([r[1] for r in res]),
                        q=np.concatenate([r[2] for r in res]), dtrap=np.concatenate([r[3] for r in res]))
    return out


def components(G, rng=None, B=300):
    """U_t, W_t (pooled over sets), the window threshold q*_i = (U_i - U_{i+1})^+ / ((U_i - U_{i+1})^+ + (W_{i+1} - W_i)) between
    consecutive grid steps (the mixture decreases between them exactly when q < q*_i; it never exceeds the largest one-step
    threshold inside the window) and its maximum, with bootstrap standard errors over interaction sets."""
    def stat(idx):
        S, c = G["S"][idx].sum(0), G["cnt"][idx].sum(0); U, W = S[0] / c[0], S[1] / c[1]; dU, dW = U[:-1] - U[1:], W[1:] - W[:-1]
        qs = np.maximum(dU, 0) / (np.maximum(dU, 0) + dW); return U, W, qs
    nz = len(G["q"]); U, W, qs = stat(np.arange(nz)); rng = rng or np.random.default_rng(5)
    bs = [stat(rng.integers(0, nz, nz)) for _ in range(B)]
    return dict(TS=G["TS"], U=U, W=W, qs=qs, U_se=np.std([b[0] for b in bs], axis=0), W_se=np.std([b[1] for b in bs], axis=0),
                qs_se=np.std([b[2] for b in bs], axis=0), q_star=float(qs.max()), i_star=int(qs.argmax()),
                q_star_boot=np.array([b[2].max() for b in bs]), q=float(G["q"].mean()), q_se=float(G["q"].std() / np.sqrt(nz)),
                U_peak=float(U.max()), t_peak=int(G["TS"][U.argmax()]))


def _pool_chunk(task):
    cfg, seed, nz = task
    S = F.simulate(cfg["LAM"], nz, cfg["POOL_NEP"], cfg["T_POOL"], np.random.default_rng(seed), store_fr=True)
    return S["FR"], S["trapped"]


def build_pool(cfg):
    """Panel c: POOL_NZ x POOL_NEP full paths of T_POOL steps (from step 0).  For each trapped fraction q and each N, POOL_GROUPS
    averages of N paths drawn from the pool (untrapped and trapped paths in proportion q; None = as they come): the fraction of
    averages that decrease at some step."""
    per = int(np.ceil(cfg["POOL_NZ"] / cfg["CHUNKS"]))
    res = F.run_parallel(_pool_chunk, [(cfg, 77_000 + k, per) for k in range(cfg["CHUNKS"])], cfg["WORKERS"], "Gridworld: pool of paths")
    FR = np.concatenate([r[0] for r in res]); tr = np.concatenate([r[1] for r in res]); del res
    pools = (FR[~tr], FR[tr]); q_nat = float(tr.mean()); rng = np.random.default_rng(99); out = dict(q_nat=q_nat, n_paths=len(FR), curves={})
    for q in list(cfg["POOL_QS"]) + [None]:
        dips = []
        for N in cfg["POOL_NS"]:
            hit = 0
            for _ in range(cfg["POOL_GROUPS"]):
                nw = int(rng.binomial(N, q_nat if q is None else q)); tot = np.zeros(cfg["T_POOL"])
                for pool, m in ((pools[0], N - nw), (pools[1], nw)):
                    if m: tot += pool[rng.integers(0, len(pool), m)].sum(axis=0, dtype=np.float64)
                hit += bool(np.any(np.diff(tot) < 0))
            dips.append(hit / cfg["POOL_GROUPS"])
        out["curves"][q] = np.array(dips); print(f"   pool: q = {q}: {np.round(dips, 3)}", flush=True)
    out["Ns"] = np.array(cfg["POOL_NS"]); out["groups"] = cfg["POOL_GROUPS"]
    out["mean_u"], out["mean_w"] = pools[0].mean(0, dtype=np.float64), pools[1].mean(0, dtype=np.float64)
    return out


def _step_chunk(task):
    """Panel B: one-step changes FR_{t+1} - FR_t of STEP_NEP episodes per set at every step of a log grid (independent episodes per
    step), and the sets' tables (for the asymptotic law)."""
    cfg, seed, nz = task; TS = step_times(cfg); rng = np.random.default_rng(seed); erng = np.random.default_rng(seed + 10 ** 6)
    inc = np.zeros((len(TS), nz * cfg["STEP_NEP"]), np.float32); tabs = []
    for z in range(nz):
        ell = F.checked_table(F.make_env(rng, cfg["LAM"]), cfg["LAM"]); lg = F.table_logs(ell); d_o, R, Rc = ell.shape; tabs.append(ell)
        for i, t in enumerate(TS):
            rg, rc = erng.integers(0, R, cfg["STEP_NEP"]), erng.integers(0, Rc, cfg["STEP_NEP"]); pr = ell[:, rg, rc].T
            n = erng.binomial(int(t), pr).astype(float); f0 = F.fr_counts(n, int(t), *lg)
            inc[i, z * cfg["STEP_NEP"]:(z + 1) * cfg["STEP_NEP"]] = F.fr_counts(n + (erng.random(pr.shape) < pr), int(t) + 1, *lg) - f0
    return inc, tabs


def step_times(cfg): return np.unique(np.round(np.logspace(0, 4, cfg["STEP_NT"])).astype(int))


def build_steps(cfg):
    per = int(np.ceil(cfg["STEP_NZ"] / cfg["CHUNKS"]))
    res = F.run_parallel(_step_chunk, [(cfg, 55_000 + k, per) for k in range(cfg["CHUNKS"])], cfg["WORKERS"], "Gridworld: dips against the step")
    inc = np.concatenate([r[0] for r in res], axis=1); tabs = [t for r in res for t in r[1]]; TS = step_times(cfg); rng = np.random.default_rng(3)
    it = iter(tabs); pool = F.one_step_pool(cfg["LAM"], len(tabs), None, both_goals=False, env=lambda r_, l_: next(it))
    x, w, q = F.step_atoms(pool); lo, hi = F.dip_law(x, w, q, np.array(cfg["STEP_NS"])); sim, se = [], []
    for N in cfg["STEP_NS"]:
        ng = inc.shape[1] // N; p = np.array([np.mean(row[rng.permutation(inc.shape[1])[:ng * N]].reshape(ng, N).mean(1) < 0) for row in inc])
        sim.append(p); se.append(np.sqrt(np.maximum(p * (1 - p), 1.0 / ng) / ng))
    return dict(TS=TS, Ns=np.array(cfg["STEP_NS"]), sim=np.array(sim), se=np.array(se), law_lo=lo, law_hi=hi, n_sets=len(tabs), n_ep=cfg["STEP_NEP"])


def _qb_draw(task):
    """build_qbands: one chunk of QB_CHUNK table draws at LAM_MAIN (own seed): the tables whose trapped fraction falls in a band."""
    cfg, seed = task; lam = F.CFG["LAM_MAIN"]; rng = np.random.default_rng(seed); out = []
    for _ in range(cfg["QB_CHUNK"]):
        ell = F.checked_table(F.make_env(rng, lam), lam); f = float(F.one_step_table(ell)["trapped"].mean())
        b = [i for i, c in enumerate(cfg["QB_CENTERS"]) if abs(f - c) < cfg["QB_HALF"]]
        if b: out.append((b[0], f, ell))
    return out


def _qb_sim(task):
    """build_qbands: QB_EP episodes in each interaction set (own seed), each at its own joint realization, from step QB_T0: the changes
    of FR over the next QB_T steps (counts drawn binomially, exact observers; as fr_theory.build_thirds)."""
    cfg, tabs, seed = task; erng = np.random.default_rng(seed); t0, Tm, n_ep = cfg["QB_T0"], cfg["QB_T"], cfg["QB_EP"]; h = []
    for ell in tabs:
        lg = F.table_logs(ell); d_o, R, Rc = ell.shape; rg, rc = erng.integers(0, R, n_ep), erng.integers(0, Rc, n_ep); pr = ell[:, rg, rc].T
        n = erng.binomial(t0, pr).astype(float); fr = [F.fr_counts(n, t0, *lg)]
        for st in range(Tm):
            n = n + (erng.random(pr.shape) < pr); fr.append(F.fr_counts(n, t0 + st + 1, *lg))
        h.append(np.diff(np.array(fr), axis=0).T.astype(np.float32))
    return np.concatenate(h)


def build_qbands(cfg):
    """fig_fr_theory_all v (John 11-11 late: more curves, a broader range, evenly spaced, structured values): QB_NZ interaction sets in
    each band |q - c| < QB_HALF of the trapped fraction, c in QB_CENTERS, found by drawing tables at LAM_MAIN in chunks of QB_CHUNK with
    seeds 91,000 + chunk (taken in chunk order: the sets do not depend on WORKERS).  THEORY (tables alone): P(an average of N episodes
    dips within QB_T steps) = 1 - E_R[(1 - p_R)^T] (fr_theory.monotone_theory, QB_TUPLES draws of R per N).  SIMULATION: QB_EP episodes
    per set from step QB_T0 (as fr_theory.build_thirds), dealt at random into as many independent groups of N as they allow; the fraction
    of groups whose average decreases at some step of the QB_T."""
    lam, nb, W = F.CFG["LAM_MAIN"], len(cfg["QB_CENTERS"]), cfg["WORKERS"]; keep, qset = [[] for _ in range(nb)], [[] for _ in range(nb)]; nxt = 0
    while any(len(k) < cfg["QB_NZ"] for k in keep):
        if nxt * cfg["QB_CHUNK"] >= cfg["QB_MAX_DRAWS"]: raise SystemExit(f"build_qbands: after {nxt * cfg['QB_CHUNK']} tables the bands hold {[len(k) for k in keep]}")
        res = F.run_parallel(_qb_draw, [(cfg, 91_000 + nxt + i) for i in range(4 * W)], W, f"q bands: tables from {nxt * cfg['QB_CHUNK']:,}")
        nxt += 4 * W
        for r in res:
            for b, f, ell in r:
                if len(keep[b]) < cfg["QB_NZ"]: keep[b].append(ell); qset[b].append(f)
        print(f"   q bands: {[len(k) for k in keep]} sets after {nxt * cfg['QB_CHUNK']:,} tables", flush=True)
    Nth = np.unique(np.r_[np.round(np.logspace(np.log10(5), np.log10(cfg["QB_NMAX"]), cfg["QB_NTH"])).astype(int), np.array(cfg["QB_NS"], int)])
    grng = np.random.default_rng(97); bands = []
    for b, tabs in enumerate(keep):
        it = iter(tabs); pool = F.one_step_pool(lam, len(tabs), None, both_goals=False, env=lambda r_, l_: next(it)); q = float(pool["trapped"].mean())
        m, se = F.monotone_theory(pool, Nth, (cfg["QB_T"],), cfg["QB_TUPLES"], 9300 + 10 * b, workers=W)
        parts = [c for c in np.array_split(np.arange(len(tabs)), cfg["CHUNKS"]) if len(c)]
        h = np.concatenate(F.run_parallel(_qb_sim, [(cfg, [tabs[j] for j in c], 95_000 + 100 * b + k) for k, c in enumerate(parts)], W, f"q band {b + 1}: simulation"))
        sim, sse, ngs = [], [], []
        for N in cfg["QB_NS"]:
            ng = len(h) // N; avg = h[grng.permutation(len(h))[:ng * N]].reshape(ng, N, -1).mean(1); d = float(np.any(avg < 0, axis=1).mean())
            sim.append(d); sse.append(np.sqrt(max(d * (1 - d), 1.0 / ng) / ng)); ngs.append(ng)
        bands.append(dict(center=cfg["QB_CENTERS"][b], q=q, q_sets=np.array(qset[b]), Nth=Nth, theory=1 - m[:, 0], theory_se=se[:, 0], Ns=np.array(cfg["QB_NS"]),
                          sim=np.array(sim), se=np.array(sse), groups=np.array(ngs), n_sets=len(tabs), n_ep=cfg["QB_EP"]))
        print(f"   q band {b + 1}: q = {q:.3f}, done", flush=True)
    return dict(bands=bands, tables_drawn=nxt * cfg["QB_CHUNK"])


def _opt_summary(v, nt, n_ep):
    """One point of fr_theory._opt_point, summarized as fr_theory.derive does (mean and s.e. over its interaction sets)."""
    ms = lambda x: (float(np.mean(x)), float(np.std(x) / np.sqrt(len(x))))
    return dict(ed=ms(v["ed"]), slope=ms(v["slope"]), q=ms(v["q"]), gap=[ms(v["accJ"][:, i] - v["accN"][:, i]) for i in range(nt)],
                accJ=[ms(v["accJ"][:, i]) for i in range(nt)], no_trap=float(np.mean(v["q"] == 0)), n_sets=len(v["ed"]), n_ep=n_ep)


def build_opt_more(cfg):
    """fig_fr_theory_all j (John 11-12: "add R=2 to panel k", then "can we try 2,3,4,5?"): the points of fr_theory.build_optimum (per number
    of observables d_o: E[Delta] and the trapped fraction q of the tables; the simulated slope of the mean FR between OPT_T1 and OPT_T2 and
    both observers' accuracy at OPT_TACC, on the same interaction sets) at the numbers of realizations OPT2_RS, with its settings, its point
    function and its seeds (70,000 + 100 R + d_o: a point does not depend on which other points are built, and none is shared with its R =
    5, 10, 20), summarized as fr_theory.derive does.  A cache of its own: fr_theory's is keyed by its OPT_RS (10,000 sets x 39 points),
    which this leaves untouched.  One point per worker process."""
    fc = dict(F.CFG, OPT_NZ=cfg["OPT2_NZ"], OPT_NEP=cfg["OPT2_NEP"]); nt = len(fc["OPT_TACC"])
    tasks = [(fc, R, d) for d in sorted(fc["OPT_DOS"], reverse=True) for R in sorted(cfg["OPT2_RS"], reverse=True)]     # the slow points first
    res = F.run_parallel(F._opt_point, tasks, cfg["WORKERS"], "growth rate vs. observables, more R")
    return {(R, d): _opt_summary(v, nt, fc["OPT_NEP"]) for (_, R, d), v in zip(tasks, res)}


def _sw_efr_at(ell, ts):
    """Exact E[FR_t] of a one-observable table at the steps ts only (fr_theory.exact_efr computes every step)."""
    m = ell.mean(1); out = np.zeros(len(ts))
    for i, t in enumerate(ts):
        lB, lN, _, P, _ = F._beliefs(ell, m, int(t)); out[i] = (P.mean(axis=(1, 2)) * F._fr_by_count(lB, lN)).sum()
    return out


def _sw_sim(ell, ts, n_ep, rng):
    """Monte Carlo of one one-observable table: n_ep episodes, each at a uniformly drawn joint realization; at every step of ts
    (increasing) the mean and SEM of FR_t and of the Joint log loss -ln B_t(r_g).  The count of ones is drawn binomially between the
    recorded steps (exact: both observers depend on the observations through the count alone)."""
    R, Rc = ell.shape; m = ell.mean(1); rg, rc = rng.integers(0, R, n_ep), rng.integers(0, Rc, n_ep); p = ell[rg, rc]; k = np.zeros(n_ep, np.int64); prev = 0; out = []
    for t in ts:
        k += rng.binomial(int(t) - prev, p); prev = int(t); lB, lN, _, _, _ = F._beliefs(ell, m, int(t)); fr = F._fr_by_count(lB, lN)[k]; jl = -lB[k, rg]
        out.append((fr.mean(), fr.std() / np.sqrt(n_ep), jl.mean(), jl.std() / np.sqrt(n_ep)))
    return np.array(out)


def _sw_atoms(ells, t):
    """One episode's increment FR_{t+1} - FR_t in each table: its values and their probabilities over the outcomes (joint realization
    summed out: count of ones at t, next observation), as toy_atoms."""
    out = []
    for ell in ells:
        m = ell.mean(1); lB0, lN0, _, P, _ = F._beliefs(ell, m, t); f0 = F._fr_by_count(lB0, lN0); lB1, lN1, _, _, _ = F._beliefs(ell, m, t + 1); f1 = F._fr_by_count(lB1, lN1)
        out.append((np.r_[f1[1:] - f0, f1[:-1] - f0], np.r_[(P * ell[None]).mean(axis=(1, 2)), (P * (1 - ell[None])).mean(axis=(1, 2))]))
    return out


def build_small(cfg):
    """Row 1 and the critical-step panel of fig_fr_theory_all: the Cognitive Gridworld with SW_R realizations per variable and ONE
    observable (the paper's generator, fr_theory.make_env), where a history is one count and every expectation is an exact sum.
    SELECTION, by a stated rule: interaction sets are drawn with seed SW_SEED, each at every temperature of SW_LAMS (the draws do not
    depend on lambda); the first with no trapped joint realization at any of them is the TRAP-FREE set, the first with exactly one at
    SW_LAM the ONE-TRAP set.  (Also: the number of traps in SW_NSTAT other sets at SW_LAM, to say how common such sets are.)
    EXACT: the anatomy of both sets at SW_LAM to SW_T steps; E[FR_t] of the trap-free set at each temperature on a log grid to SW_TMAX;
    both sets at SW_LAM at every step to SW_TMIX, which gives the threshold q* of their mixture (Corollary 1; a fraction q / share of the
    episodes in the one-trap set, share = its trapped share) and the critical step t*; the law of the average increment at t* (FFT up to
    N_FFT_MAX episodes, saddlepoint of the same law above).
    SIMULATION: SW_NEP episodes per table and temperature (_sw_sim); at t*, SW_PATHS averages per N, the N episodes of an average drawn
    as the numbers of them that fall on each outcome (count of ones at t*, next observation, table): one multinomial draw."""
    lam0, R, lams = cfg["SW_LAM"], cfg["SW_R"], tuple(cfg["SW_LAMS"]); assert lam0 in lams
    rng = np.random.default_rng(cfg["SW_SEED"]); free = one = None; i = 0
    while free is None or one is None:
        if i >= 20_000: raise SystemExit(f"build_small: no trap-free / one-trap interaction set in {i} draws")
        st = rng.bit_generator.state; row = {}
        for lam in lams:
            rng.bit_generator.state = st; row[lam] = F.checked_table(F.make_env(rng, lam, R=R, d_o=1), lam)[0]
        k = {lam: F.table_stats(e)["k"] for lam, e in row.items()}
        if free is None and max(k.values()) == 0: free = (i, row)
        if one is None and k[lam0] == 1: one = (i, row[lam0])
        i += 1
    srng = np.random.default_rng(cfg["SW_SEED"] + 3); k_stat = np.array([F.table_stats(F.checked_table(F.make_env(srng, lam0, R=R, d_o=1), lam0)[0])["k"] for _ in range(cfg["SW_NSTAT"])])
    ea, eb = free[1][lam0], one[1]; an = {}
    for key, ell in (("a", ea), ("b", eb)):
        a_, st_ = F.exact_anatomy(ell, cfg["SW_T"]); an[key] = dict(ell=ell, an=a_, st=st_)
    share = an["b"]["st"]["k"] / eb.size; assert an["a"]["st"]["k"] == 0 and an["b"]["st"]["k"] == 1 and np.isclose(share, 1 / 9)       # mix_atoms: a fraction 9 q of the episodes
    tl = np.unique(np.round(np.logspace(0, np.log10(cfg["SW_TMAX"]), int(round(np.log10(cfg["SW_TMAX"]) * cfg["SW_PER_DECADE"])) + 1)).astype(int))
    temp = {lam: dict(ell=free[1][lam], st=F.table_stats(free[1][lam]), efr=_sw_efr_at(free[1][lam], tl)) for lam in lams}
    E0, E1 = F.exact_efr(ea, cfg["SW_TMIX"]), F.exact_efr(eb, cfg["SW_TMIX"]); w_star, t_star = F.mixture_threshold(E0, E1); q_star = w_star * share
    assert t_star < 0.5 * cfg["SW_TMIX"], "the critical step is too close to the end of the steps computed"
    # ---- the average increment at the critical step: exact law, and simulated averages
    atoms = _sw_atoms([ea, eb], t_star); Ns = np.array(cfg["SW_NS"]); simN = np.array(cfg["SW_SIM_NS"]); prng = np.random.default_rng(cfg["SW_SEED"] + 2)
    p_star, sim, both = np.zeros((len(cfg["SW_RATIOS"]), len(Ns))), np.zeros((len(cfg["SW_RATIOS"]), len(simN))), []
    for ir, r in enumerate(cfg["SW_RATIOS"]):
        x, w = mix_atoms(atoms, r * q_star); keep = w > 0; x, w = x[keep], w[keep]; w = w / w.sum(); sp = saddle(x, w)
        for iN, N in enumerate(Ns): p_star[ir, iN] = dip_fft(x, w, N) if N <= cfg["N_FFT_MAX"] else dip_saddle(sp, N)
        both.append((dip_fft(x, w, cfg["N_FFT_MAX"]), dip_saddle(sp, cfg["N_FFT_MAX"])))
        for iN, N in enumerate(simN): sim[ir, iN] = np.mean(prng.multinomial(int(round(N)), w, size=cfg["SW_PATHS"]) @ x < 0)
    # ---- simulated curves
    erng = np.random.default_rng(cfg["SW_SEED"] + 1); t_lin = np.arange(1, cfg["SW_SIM_STEPS"] + 1) * cfg["SW_T"] // cfg["SW_SIM_STEPS"]
    t_log = np.unique(np.round(10 ** (np.arange(0, np.log10(cfg["SW_TMAX"]) * cfg["SW_SIM_PER_DECADE"] + 0.5) / cfg["SW_SIM_PER_DECADE"])).astype(int))
    ts = np.unique(np.r_[t_lin, t_log]); sims = {("free", lam): _sw_sim(free[1][lam], ts, cfg["SW_NEP"], erng) for lam in lams}; sims[("one", lam0)] = _sw_sim(eb, ts, cfg["SW_NEP"], erng)
    return dict(i_free=free[0], i_one=one[0], n_drawn=i, k_stat=k_stat, an=an, temp=temp, tl=tl, E0=E0, E1=E1, E1l=_sw_efr_at(eb, tl), w_star=w_star, q_star=q_star, t_star=t_star,
                share=share, ratios=list(cfg["SW_RATIOS"]), Ns=Ns, p_star=p_star, both=both, sim_Ns=simN, sim=sim, sim_se=np.sqrt(np.maximum(sim * (1 - sim), 1.0 / cfg["SW_PATHS"]) / cfg["SW_PATHS"]),
                ts=ts, t_lin=t_lin, t_log=t_log, sims=sims, n_ep=cfg["SW_NEP"], paths=cfg["SW_PATHS"], lam=lam0, lams=lams, R=R)


def check_assumptions(lams=(1.0, 2.0, 3.0), nz=400, seed=404):
    """(A2) and (A3) in sampled interaction sets (the tex says they held in every one): per likelihood temperature, the smallest
    margins over nz sets.  (A2): observation laws of joint realizations with different goal realizations differ (max_i of the
    difference of their probabilities > 0).  (A3): no wrong action fits exactly as well as the optimal one, and the best Naive
    model is unique.  The margins are positive but come close to zero, which is why the limits are not uniform in Z."""
    out = {}
    for lam in lams:
        rng = np.random.default_rng(seed); a2, tie, gap = [], [], []
        for _ in range(nz):
            ell = F.checked_table(F.make_env(rng, lam), lam); d, R, Rc = ell.shape; kap = F.one_step_table(ell)["kap"]
            E = ell.reshape(d, R * Rc).T; rg = np.repeat(np.arange(R), Rc)
            a2.append(np.abs(E[:, None, :] - E[None, :, :]).max(-1)[rg[:, None] != rg[None, :]].min())
            kg = np.take_along_axis(kap, np.broadcast_to(np.arange(R)[:, None, None], (R, Rc, 1)), 2)[:, :, 0]
            g = np.abs(kap - kg[:, :, None]); g[np.arange(R)[:, None], np.arange(Rc)[None, :], np.arange(R)[:, None]] = np.inf; tie.append(g.min())
            srt = np.sort(kap, axis=2); gap.append((srt[:, :, 1] - srt[:, :, 0]).min())
        out[lam] = dict(a2=min(a2), tie=min(tie), gap=min(gap), ok=min(a2) > 0 and min(tie) > 0 and min(gap) > 0, n=nz)
    return out


def trap_origin(lams=(1.0, 2.0, 3.0), nz=300, seed=505):
    """Why about half of the joint realizations are trapped: the contextual variable moves an observable's probability more than
    the goal separates the Naive marginals (which average over the contextual variable).  Per likelihood temperature, over nz sets:
    s.d. of ell across the contextual variable's realizations (at a fixed goal realization), s.d. of the Naive marginal m across
    actions, and the mean mismatch kappa to the optimal action's Naive model and to the best other action's."""
    out = {}
    for lam in lams:
        rng = np.random.default_rng(seed); a, b, kt, ko = [], [], [], []
        for _ in range(nz):
            ell = F.checked_table(F.make_env(rng, lam), lam); d, R, Rc = ell.shape; tab = F.one_step_table(ell); kap = tab["kap"]
            a.append(ell.std(axis=2).mean()); b.append(tab["m"].std(axis=1).mean())
            kg = np.take_along_axis(kap, np.broadcast_to(np.arange(R)[:, None, None], (R, Rc, 1)), 2)[:, :, 0]; other = kap.copy()
            other[np.arange(R)[:, None], np.arange(Rc)[None, :], np.arange(R)[:, None]] = np.inf; kt.append(kg.mean()); ko.append(other.min(axis=2).mean())
        out[lam] = dict(sd_context=float(np.mean(a)), sd_naive=float(np.mean(b)), kap_true=float(np.mean(kt)), kap_best_other=float(np.mean(ko)), n=nz)
    return out


# ============================================================================================================
# 3. CACHES
# ============================================================================================================
KEYS = dict(toy_theory=("T_TOY", "TOY_RATIOS", "TOY_STEP_RATIOS", "TOY_NS", "N_FFT_MAX"),
            toy_theory_extra=("T_TOY", "TOY_RATIOS", "TOY_NS", "TOY_SIM_NS", "N_FFT_MAX"), toy_fine=("T_TOY", "TOY_FINE", "TOY_FINE_PER_DECADE", "N_FFT_MAX"), toy_sim=("T_TOY", "TOY_RATIOS", "TOY_SIM_NS", "TOY_PATHS"),
            grid=("GRID_NZ", "GRID_NEP", "GRID_TMAX", "GRID_NT", "GRID_LAMS"),
            pool=("LAM", "T_POOL", "POOL_NZ", "POOL_NEP", "POOL_QS", "POOL_NS", "POOL_GROUPS"), steps=("LAM", "STEP_NZ", "STEP_NEP", "STEP_NS", "STEP_NT"),
            qbands=("CHUNKS", "QB_CENTERS", "QB_HALF", "QB_NZ", "QB_CHUNK", "QB_EP", "QB_T0", "QB_T", "QB_NS", "QB_TUPLES", "QB_NTH", "QB_NMAX"),
            opt_more=("OPT2_RS", "OPT2_NZ", "OPT2_NEP", "OPT2_FROM"),
            small=("SW_R", "SW_SEED", "SW_LAMS", "SW_LAM", "SW_NSTAT", "SW_T", "SW_TMIX", "SW_TMAX", "SW_PER_DECADE", "SW_NEP", "SW_SIM_STEPS", "SW_SIM_PER_DECADE", "SW_RATIOS",
                   "SW_NS", "SW_SIM_NS", "SW_PATHS", "N_FFT_MAX"))


def load(cfg, tag, only=None):
    """Each cache is keyed by the configuration entries it depends on (KEYS) and rebuilt when one of them changes."""
    os.makedirs(OUT, exist_ok=True); D = {}
    for name, builder in (("toy_theory", build_toy_theory), ("toy_theory_extra", build_toy_theory_extra), ("toy_fine", build_toy_fine), ("toy_sim", build_toy_sim), ("grid", build_grid),
                          ("pool", build_pool), ("steps", build_steps), ("qbands", build_qbands), ("opt_more", build_opt_more), ("small", build_small)):
        if only and name not in only: continue
        path = os.path.join(OUT, f"{name}{tag}.pkl"); want = repr([(k, cfg[k]) for k in KEYS[name]])
        if os.path.exists(path):
            with open(path, "rb") as fh: sig, data = pickle.load(fh)
            if sig == want: D[name] = data; continue
            print(f"[cache] {os.path.basename(path)} is stale: rebuilding", flush=True)
        t0 = time.time(); data = builder(cfg)
        with open(path, "wb") as fh: pickle.dump((want, data), fh, protocol=4)
        print(f"[cache] built {os.path.basename(path)} in {time.time() - t0:.0f} s", flush=True); D[name] = data
    return D


if __name__ == "__main__":
    quick = "--quick" in sys.argv; cfg = dict(CFG, **(QUICK if quick else {})); tag = "_quick" if quick else ""
    only = [a.split("=")[1] for a in sys.argv if a.startswith("--only=")]
    cmd = sys.argv[1] if len(sys.argv) > 1 else "figure"
    D = load(cfg, tag, only or None)
    if cmd in ("figure", "numbers"):
        import fr_theory_additions_plot as P_                                       # the figures and the printed values
        if cmd == "numbers": P_.numbers(cfg, D)
        else: P_.figures(cfg, D, tag)
