"""The learning signal on entangled steps through training, cumulated over lags (John 11-02: "do you think you can show
the cumulative (through lag) cross-corr?"). Same statistic as fig_td_summary j / fig_minimal_set e (gradflow_analysis's
locked convention, plot_training_dynamics.summarize "xc_upd"), now summed over lags so that the early lagged signal and
the late lag-0 signal are on one scale.
-> main/REPLICATION/figure_bin/td_tt_credit_lags.png   figure()
   numbers()   every number below
   cache main/DATA/experimental/training_dynamics/through_training/credit_lags/credit_lags.npz

Definitions:
  coherence at step s (td_analysis.credit, "cr_coh"): both variables' errors at step s sent back 0-15 steps through the
    recurrence; ||Sigma_k delta_k||^2 / Sigma_k ||delta_k||^2 over the back-propagated errors delta_k at steps s - k (high:
    they point the same way and add up to one consistent weight change). Entanglement e_t = rev (mean of the two
    variables' entanglement at step t, probes.npz "ent").
  r(tau): within each episode, coherence and entanglement z-scored over steps 17-30; the product coh_(t+tau) e_t averaged
    over the episode's steps, then over episodes (tau > 0: the coherence of errors made tau steps after the entangled
    step). Figure a-c (John: "start from -8 lag"): K(L) = Sigma_{tau=-8..L} r(tau), the cumulative from lag -8 through L;
    in c, the first curve averages r(tau) over the checkpoints up to 100 training episodes (the initial network
    included) before cumulating; the white line in a, b is the lag L >= 0 where K peaks (3-checkpoint running mean), drawn where the largest sum over
    lags 0..L is above 0 in >= 95% of the resamples. Timing (d): C(L) = Sigma_{tau=0..L} r(tau) (lags 0..L summed);
    C_neg = Sigma_{tau=-8..-1} r(tau).
  Intervals: 1,000 resamples of the 50 test environments (td_tt_gap.boot_counts, the shared matrix; each environment's
    episode sums, ratio of sums). Crossings: half of the end value (mean of the last 3 checkpoints; recomputed in every
    resample), stays-past rule, log-interpolated (td_tt_gap.crossing).

Results (Trained RNN; [5-95%]; every number printed by numbers()):
  Errors made 1-4 steps after an entangled step get coherent credit first: Sigma_{tau=1..4} r reaches half its peak
  (.142) at 467 [432-593] training episodes, before accuracy passes naive (835 [798-876]; intervals separate), peaks at
  .284 (~2,800) and falls to .089 [.074-.103] at the end. The entangled step's own error comes later: r(0) is +.008 at
  2,000, half its end value (.169) at 6,885 [6,347-7,168], .338 [.327-.348] at the end: inside the late behaviour
  (e.g. the jump excess angle half-way at 6,248 [5,366-7,529], td_tt_timeline). Their sum C(4) is half-built at 844
  [712-1,220] and ends at .426 [.407-.444]: the positive signal near the entangled step appears with passing naive and
  then moves from lags 1-4 onto lag 0 (lags 1-4 lose ~.2 while lag 0 gains ~.33).
  Summed over all lags the within-episode cross-correlation is exactly 0 (checked: |sum| <= 1e-14 per episode), so the
  negative lags (-.356 at the end) and lags 5-8 (-.189) mirror the peak; they are not a separate effect.
  From lag -8 (figure a-c): averaged over the 6 checkpoints up to 100, K is flat (|K| <= .028). At 320 the first
  structure: a dip just after the entangled step (-.065 at L = 1) and a positive run peaking at L = 6. Then K falls over
  the negative lags to a trough just before the entangled step (L = -1) that deepens (-.149 at 880, -.208 at 2,400,
  -.309 at 8,000, -.356 at the end), rises over lags 0-4, and peaks at a lag that moves onto the entangled step (L = 5
  at 880, 4 at 2,400, 3 at 8,000, 2 at the end; the peak line is drawn from 320, where the lags 0-4 sum stays above 0
  in >= 95% of the resamples). The part below 0 mirrors the peak.
  ESN: no positive signal at any lag or checkpoint; at the end r(0) -.046, lags 1-4 -.118, C(4) -.164; its K has the
  reverse shape (highest just before the entangled step, a dip at lags 3-4), no established positive run.

Run from the repository root (CPU, about a minute; the cache once ~2 min):
    MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/td_tt_credit_lags.py [numbers figure]"""
import os
import sys
import glob
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
OUT = os.path.join(TD, "through_training", "credit_lags")
RUNS = ("trained", "echo")
NAME = {"trained": "Trained RNN", "echo": "ESN"}
COL = {"trained": "#1f77b4", "echo": "#ff7f0e"}
WIN = slice(16, 30)                                       # steps 17-30 (the locked window)
TAU = np.arange(-8, 9)
POS = TAU >= 0
SHORT, LONG = 4, 8                                        # C through lag 4 and through lag 8


def _zs(a):
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-12)


def lag_products(y, e):
    """per episode, the mean of z(y)_(t+tau) z(e)_t over its steps, for every tau: (n, len(TAU))"""
    a, b = _zs(np.asarray(y, np.float64)), _zs(np.asarray(e, np.float64))
    out = np.empty((len(a), len(TAU)))
    for j, lag in enumerate(TAU):
        if lag < 0:
            out[:, j] = (a[:, :lag] * b[:, -lag:]).mean(1)
        elif lag > 0:
            out[:, j] = (a[:, lag:] * b[:, :-lag]).mean(1)
        else:
            out[:, j] = (a * b).mean(1)
    return out


def build():
    """per run and checkpoint: the environment sums of the per-episode lag products and the episode counts"""
    path = os.path.join(OUT, "credit_lags.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    os.makedirs(OUT, exist_ok=True)
    P = np.load(os.path.join(AN, "probes.npz"))
    rev = P["ent"].astype(np.float64).mean(-1)                                        # (10000, 30)
    out = {}
    for run in RUNS:
        files = sorted(glob.glob(os.path.join(AN, run, "ck_*.npz")))
        S, eps = [], []
        for f in files:
            z = np.load(f)
            rows = z["cr_rows"]
            env = rows // gap.PER_ENV
            lp = lag_products(z["cr_coh"], rev[rows][:, WIN])
            S.append(np.stack([lp[env == k].sum(0) for k in range(gap.N_ENV)]))       # (50, TAU)
            eps.append(int(z["episode"]))
            cnt = np.bincount(env, minlength=gap.N_ENV).astype(np.float64)
        out[f"{run}_sum"] = np.stack(S)                                                  # (K, 50, TAU)
        out[f"{run}_cnt"] = cnt
        out[f"{run}_episode"] = np.array(eps)
        print(f"  {run}: {len(files)} checkpoints", flush=True)
    np.savez(path, **out)
    print("saved", path)
    return out


def curves(D, run):
    """full-sample r (K, TAU) and resampled (N_BOOT, K, TAU)"""
    s, c = D[f"{run}_sum"], D[f"{run}_cnt"]                                             # (K, 50, TAU), (50,)
    C = gap.boot_counts()
    full = s.sum(1) / c.sum()
    boot = np.einsum("be,ket->bkt", C, s) / (C @ c)[:, None, None]
    return full, boot


def cumul(r):
    """C(L) for L = 0..8 over the last axis (lags -8..8): (..., 9)"""
    return np.cumsum(r[..., POS], -1)


def _peak(y):
    """the maximum of the 3-checkpoint running mean"""
    return float(np.convolve(y, np.ones(3) / 3, "valid").max())


def first_crossing(x, y, lev):
    """the first training episode at which y (..., K) rises past lev (...,), log-interpolated; NaN if never"""
    y = np.atleast_2d(y)
    past = y > np.asarray(lev)[:, None]
    out = np.full(len(y), np.nan)
    lx = np.log(x)
    for i in np.where(past.any(1))[0]:
        k = int(np.argmax(past[i]))
        if k == 0:
            out[i] = x[0]
            continue
        f = (lev[i] - y[i, k - 1]) / (y[i, k] - y[i, k - 1])
        out[i] = np.exp(lx[k - 1] + np.clip(f, 0, 1) * (lx[k] - lx[k - 1]))
    return out


def zero_sum_check(run="trained", episode=49995):
    """per episode, Sigma over ALL lags (-13..13) of (pairs at that lag) x r_i(tau) = (Sigma_s z(coh)_s)(Sigma_t z(e)_t)
    = 0: within-episode z-scoring makes the lag-summed cross-correlation vanish, so a positive peak is mirrored by
    negative lags elsewhere. Returns the largest |weighted sum| over the episodes and the share of the weighted mass at
    lags -8..8"""
    P = np.load(os.path.join(AN, "probes.npz"))
    rev = P["ent"].astype(np.float64).mean(-1)
    z = np.load(os.path.join(AN, run, f"ck_{episode:05d}.npz"))
    a, b = _zs(z["cr_coh"].astype(np.float64)), _zs(rev[z["cr_rows"]][:, WIN])
    n = a.shape[1]
    tot = np.zeros(len(a))
    for lag in range(-(n - 1), n):
        if lag < 0:
            tot += (a[:, :lag] * b[:, -lag:]).sum(1)
        elif lag > 0:
            tot += (a[:, lag:] * b[:, :-lag]).sum(1)
        else:
            tot += (a * b).sum(1)
    return float(np.abs(tot).max())


def summary():
    D = build()
    S = {}
    for run in RUNS:
        x = gap.train_x(D[f"{run}_episode"])
        r, rb = curves(D, run)
        c, cb = cumul(r), cumul(rb)
        neg, negb = r[..., ~POS].sum(-1), rb[..., ~POS].sum(-1)
        S[run] = dict(x=x, r=r, rb=rb, C=c, Cb=cb, neg=neg, negb=negb)
        for name, y, yb in (("r0", c[:, 0], cb[..., 0]), ("C4", c[:, SHORT], cb[..., SHORT]),
                            ("C8", c[:, LONG], cb[..., LONG]), ("lag1_4", c[:, SHORT] - c[:, 0],
                                                                cb[..., SHORT] - cb[..., 0])):
            if name == "lag1_4":                     # rises, then falls: half of its peak, first crossing
                lev, levb = _peak(y) / 2, np.array([_peak(v) for v in yb]) / 2
                cbx = first_crossing(x, yb, levb)
                p = first_crossing(x, y[None], np.array([lev]))[0]
            else:                                    # half of the end value, for good
                lev, levb = y[-3:].mean() / 2, yb[:, -3:].mean(1) / 2
                cbx = gap.crossing(x, yb, levb)
                p = gap.crossing(x, y, lev)
            lo, hi, share = gap.interval(cbx)
            S[run][f"half_{name}"] = dict(level=lev, p=p, lo=lo, hi=hi, share=share, end=y[-3:].mean(),
                                         end_band=np.percentile(yb[:, -3:].mean(1), [5, 95]), peak=_peak(y),
                                         x_peak=float(x[int(np.argmax(np.convolve(y, np.ones(3) / 3, "same")))]))
    return S


def numbers():
    S = summary()
    D = build()
    ser = dict(np.load(os.path.join(AN, "series_trained.npz"), allow_pickle=True))["s"].item()
    eps = D["trained_episode"]
    ref = np.stack([ser[int(e)]["xc_upd"][0] for e in eps])
    print(f"check: r(tau) here vs plot_training_dynamics' cached xc_upd (coherence), Trained RNN: max |diff| "
          f"{np.abs(S['trained']['r'] - ref).max():.2e}")
    print(f"check: per episode, the pair-weighted sum of the lagged products over all lags -13..13 = 0 (max |sum| "
          f"{zero_sum_check():.1e} at the final Trained RNN checkpoint): the lag-summed cross-correlation is zero by "
          f"construction")
    for run in RUNS:
        s = S[run]
        print(f"== {NAME[run]}")
        for name, lab in (("r0", "r(0), lag 0 only"), ("lag1_4", "lags 1-4 summed"), ("C4", "C(4), lags 0-4"),
                          ("C8", "C(8), lags 0-8")):
            h = s[f"half_{name}"]
            hi = f"{h['hi']:,.0f}" if np.isfinite(h["hi"]) else "never"
            rule = "half its peak, first reached" if name == "lag1_4" else "half of the end value, for good"
            print(f"  {lab:22s} end {h['end']:+.3f} [{h['end_band'][0]:+.3f}, {h['end_band'][1]:+.3f}], peak "
                  f"{h['peak']:+.3f} at {h['x_peak']:,.0f}; {rule} ({h['level']:+.3f}): {h['p']:,.0f} "
                  f"[{h['lo']:,.0f}-{hi}] ({h['share']:.0%} cross)")
        for xv in (10, 400, 880, 1200, 2000, 4000, 10000, s["x"][-1]):
            k = int(np.argmin(np.abs(s["x"] - xv)))
            print(f"    at {s['x'][k]:7,.0f}: r(0) {s['C'][k, 0]:+.3f}  lags 1-4 {s['C'][k, SHORT] - s['C'][k, 0]:+.3f}"
                  f"  lags 5-8 {s['C'][k, LONG] - s['C'][k, SHORT]:+.3f}  C(8) {s['C'][k, LONG]:+.3f}  "
                  f"negative side {s['neg'][k]:+.3f}")
    print("== K(L) from lag -8 (figure a-c): trough (lowest K) and peak lag among L >= 0 (drawn where established)")
    for run in RUNS:
        k, _, peak = from_m8(S, run)
        x = S[run]["x"]
        first = x[np.isfinite(peak)][0] if np.isfinite(peak).any() else np.nan
        print(f"  {NAME[run]}: peak line drawn from {first:,.0f}" if np.isfinite(first) else f"  {NAME[run]}: no "
              f"established positive signal, no peak line")
        sel = x <= EARLY
        ke = np.cumsum(S[run]["r"][sel].mean(0))
        print(f"    mean of r over the {sel.sum()} checkpoints up to {EARLY}, then cumulated: K = "
              + " ".join(f"{v:+.3f}" for v in ke) + f" (largest |K| {np.abs(ke).max():.3f})")
        for xv in STAGES:
            j = stage_index(x, xv)
            print(f"    at {x[j]:7,.0f}: trough K {k[j].min():+.3f} at L {TAU[np.argmin(k[j])]:+d}; peak K among L >= 0 "
                  f"{k[j][POS].max():+.3f} at L {TAU[POS][np.argmax(k[j][POS])]:+d}; K(8) {k[j][-1]:+.3f}; drawn peak "
                  f"line {peak[j]:.0f}")
    return S


def _edges(x):
    lx = np.log(x)
    mid = (lx[1:] + lx[:-1]) / 2
    return np.exp(np.concatenate([[lx[0] - (mid[0] - lx[0])], mid, [lx[-1] + (lx[-1] - mid[-1])]]))


EARLY = 100                                              # c: r(tau) averaged over the checkpoints up to 100, then cumulated
STAGES = (320, 880, 2400, 8000, 39996)                   # profiles in c (training episodes; nearest checkpoint
#   on the log axis, stage_index). Chosen to be ~3x apart and visibly distinct: neighbouring profiles differ by RMS
#   >= .025 (0-100 mean vs 320) to .05; 12,800 and the end differ by only ~.03 and overlapped, so one late stage is kept


def stage_index(x, xv):
    """the checkpoint nearest to xv on the log axis"""
    return int(np.argmin(np.abs(np.log(x) - np.log(xv))))


def from_m8(S, run):
    """the cumulative from lag -8: K(L) = Sigma_{tau=-8}^{L} r(tau) for L = -8..8, full (K, 17) and resampled; the lag
    where it peaks among L >= 0 (3-checkpoint running mean), drawn only from the checkpoint after which the sum over lags
    0-4 stays above 0 in >= 95% of the resamples (an established positive signal near the entangled step)"""
    s = S[run]
    k, kb = np.cumsum(s["r"], -1), np.cumsum(s["rb"], -1)
    sm = np.apply_along_axis(lambda v: np.convolve(np.pad(v, 1, mode="edge"), np.ones(3) / 3, "valid"), 0, k)
    peak = TAU[POS][np.argmax(sm[:, POS], 1)].astype(float)
    est = (s["Cb"][..., SHORT] > 0).mean(0) >= .95
    good = np.zeros_like(est)
    if est[-1]:
        first = len(est) - int(np.argmax(~est[::-1])) if (~est).any() else 0
        good[first:] = True
    return k, kb, np.where(good, peak, np.nan)


def peak_lag_r(S, run):
    """the lag tau >= 0 where r(tau) peaks (3-checkpoint running mean), drawn with the same rule as from_m8's peak line"""
    s = S[run]
    sm = np.apply_along_axis(lambda v: np.convolve(np.pad(v, 1, mode="edge"), np.ones(3) / 3, "valid"), 0, s["r"])
    peak = TAU[POS][np.argmax(sm[:, POS], 1)].astype(float)
    _, _, kpeak = from_m8(S, run)
    return np.where(np.isfinite(kpeak), peak, np.nan)


LINE_LAGS = (-1, 0, 1, 2, 3, 4)                          # h: r(tau) through training for these lags


def figure():
    """-> figure_bin/td_tt_credit_lags.png (top row: cumulated from lag -8; bottom row: the same without cumulation)"""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    FL, FT, TS = fs["label"], fs["label"], 12.
    S = summary()
    syn = gap.load("jumps")
    ia = list(syn["names"]).index("goal accuracy at T")
    acc0, acc50 = float(syn["c0"][ia]), float(syn["c50"][ia])
    PW, PH, GX, GY, L, TOP, BOT = 3.55, 2.9, 1.0, 1.25, .95, 1.55, 2.15
    W = L + 4 * PW + 3 * GX + .6 + .3
    H = TOP + 2 * PH + GY + BOT
    fig = plt.figure(figsize=(W, H))
    xs = [L, L + PW + GX, L + 2 * PW + 2 * GX + .6, L + 3 * PW + 3 * GX + .6]
    ys = [BOT + PH + GY, BOT]
    axes = [[fig.add_axes([x0 / W, y0 / H, PW / W, PH / H]) for x0 in xs] for y0 in ys]
    K = {run: from_m8(S, run) for run in RUNS}
    xmax = float(_edges(S["trained"]["x"])[-1])
    halo = [pe.withStroke(linewidth=3.2, foreground="k")]
    st, se = S["trained"], S["echo"]
    blues = LinearSegmentedColormap.from_list("blues_trunc", plt.cm.Blues(np.linspace(.3, 1., 256)))
    snorm = LogNorm(vmin=EARLY, vmax=STAGES[-1])

    def xaxis(ax):
        ax.set_xscale("log")
        ax.set_xlim(8, xmax)
        ax.set_xticks([10, 1e2, 1e3, 1e4])
        ax.set_xticklabels(["0", "100", "1k", "10k"])
        ax.minorticks_off()
        ax.set_xlabel("training episodes", fontsize=FL)

    def heat(ax, s, mat, vmin, vmax, peak, letter, run, top):
        im = ax.pcolormesh(_edges(s["x"]), np.arange(TAU[0] - .5, TAU[-1] + 1), mat.T, cmap="viridis", vmin=vmin,
                           vmax=vmax, shading="flat", rasterized=True)
        ok = np.isfinite(peak)
        if ok.any():
            ax.plot(s["x"][ok], peak[ok], color="w", lw=1.8, path_effects=halo, zorder=5)
        ax.axhline(-.5, color="w", lw=.9, ls=(0, (2, 2)), zorder=4)
        xaxis(ax)
        ax.set_yticks(range(-8, 9, 4))
        ax.axvline(acc0, color="w", lw=1.1, ls=":")
        ax.axvline(acc50, color="w", lw=1.1, ls=(0, (5, 2, 1, 2)))
        if top:
            ax.set_ylabel("through lag $L$ (steps after the entangled step)", fontsize=FL)
            ax.set_title(f"{letter}   {NAME[run]}: cumulative from lag $-8$\n"
                         r"$K(L) = \Sigma_{\tau=-8}^{L}\, r(\mathrm{coh}_{t+\tau},\, e_t)$", loc="left", fontsize=TS,
                         linespacing=1.5)
        else:
            ax.set_ylabel(r"lag $\tau$ (steps after the entangled step)", fontsize=FL)
            ax.set_title(f"{letter}   {NAME[run]}: cross-correlation by lag\n"
                         r"$r(\tau) = r(\mathrm{coh}_{t+\tau},\, e_t)$", loc="left", fontsize=TS, linespacing=1.5)
        return im

    def colorbar(im, row, label):
        cax = fig.add_axes([(xs[1] + PW + .12) / W, (ys[row] + .15 * PH) / H, .13 / W, .7 * PH / H])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=FT - .5)
        cb.ax.tick_params(labelsize=FT - 1)

    def profiles(ax, f_of, ylab, letter, title, inset=(.56, .1)):
        sel = st["x"] <= EARLY
        ax.plot(TAU, f_of(st["r"][sel].mean(0)), color=blues(snorm(EARLY)), lw=1.9, marker="o", ms=2.8, zorder=3)
        for xv in STAGES:
            j = stage_index(st["x"], xv)
            ax.plot(TAU, f_of(st["r"][j]), color=blues(snorm(st["x"][j])), lw=1.9, marker="o", ms=2.8, zorder=3)
        ye = f_of(se["r"][-1])
        ax.plot(TAU, ye, color=COL["echo"], lw=1.6, marker="s", ms=3, zorder=2)
        ax.text(-1.2, ye[TAU == -1][0] + .012, f"ESN at {se['x'][-1]:,.0f}", color=COL["echo"], fontsize=FT - 1.5,
                ha="right", va="bottom")
        sm = plt.cm.ScalarMappable(norm=snorm, cmap=blues)
        cin = ax.inset_axes([inset[0], inset[1], .4, .045])
        cbs = fig.colorbar(sm, cax=cin, orientation="horizontal")
        cbs.set_ticks([100, 1000, 10000, 40000])
        cbs.set_ticklabels([f"0-{EARLY}\n(mean)", "1k", "10k", "40k"])
        cbs.ax.tick_params(labelsize=FT - 2.5, length=2, pad=1.5)
        cin.set_title("Trained RNN, training episodes", fontsize=FT - 2, pad=3)
        ax.axhline(0, color="k", lw=.7)
        ax.axvline(0, color="k", lw=.8, ls=(0, (2, 2)))
        ax.set_xlabel(ylab[0], fontsize=FL)
        ax.set_ylabel(ylab[1], fontsize=FL)
        ax.set_xticks(range(-8, 9, 2))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{letter}   {title[0]}\n{title[1]}", loc="left", fontsize=TS, linespacing=1.5)

    # top row: cumulated from lag -8
    vmin = float(min(K[r][0].min() for r in RUNS))
    vmax = float(max(K[r][0].max() for r in RUNS))
    for c, run, letter in ((0, "trained", "a"), (1, "echo", "b")):
        im = heat(axes[0][c], S[run], K[run][0], vmin, vmax, K[run][2], letter, run, True)
    colorbar(im, 0, "$K(L)$, summed $r$ from lag $-8$")
    profiles(axes[0][2], np.cumsum, (r"through lag $L$", "$K(L)$"), "c",
             ("The cumulative profile at stages", r"$K(L)$ against $L$; dashed: the entangled step ($L = 0$)"))
    ax = axes[0][3]
    parts = (("C4", "-", 2.3), ("lag1_4", (0, (5, 2)), 1.8), ("r0", ":", 2.0))
    for key, ls, lw in parts:
        for run in RUNS:
            s = S[run]
            y = s["C"][:, SHORT] if key == "C4" else s["C"][:, 0] if key == "r0" else s["C"][:, SHORT] - s["C"][:, 0]
            yb = s["Cb"][..., SHORT] if key == "C4" else s["Cb"][..., 0] if key == "r0" else \
                s["Cb"][..., SHORT] - s["Cb"][..., 0]
            ax.plot(s["x"], y, color=COL[run], ls=ls, lw=lw if run == "trained" else lw * .6, zorder=3)
            if run == "trained":
                lo, hi = np.percentile(yb, [5, 95], 0)
                ax.fill_between(s["x"], lo, hi, color=COL[run], alpha=.16, lw=0, zorder=2)
        h = st[f"half_{key}"]
        if np.isfinite(h["p"]):
            hi_ = h["hi"] if np.isfinite(h["hi"]) else xmax
            ax.errorbar([h["p"]], [h["level"]], xerr=[[h["p"] - h["lo"]], [hi_ - h["p"]]], fmt="o", ms=6.5,
                        mfc=COL["trained"], mec="k", ecolor="k", elinewidth=1.5, capsize=3, zorder=6)
    ax.axhline(0, color="k", lw=.7, zorder=1)
    ax.axvline(acc0, color="k", lw=1.1, ls=":")
    ax.axvline(acc50, color="k", lw=1.1, ls=(0, (5, 2, 1, 2)))
    xaxis(ax)
    ax.set_ylabel("summed $r$", fontsize=FL)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("d   Where the positive signal sits\n"
                 r"$r(0)$,  $\Sigma_{\tau=1}^{4} r(\tau)$,  their sum", loc="left", fontsize=TS, linespacing=1.5)
    # bottom row: the same without cumulation
    vmin = float(min(S[r]["r"].min() for r in RUNS))
    vmax = float(max(S[r]["r"].max() for r in RUNS))
    for c, run, letter in ((0, "trained", "e"), (1, "echo", "f")):
        im = heat(axes[1][c], S[run], S[run]["r"], vmin, vmax, peak_lag_r(S, run), letter, run, False)
    colorbar(im, 1, r"$r(\tau)$")
    profiles(axes[1][2], lambda v: v, (r"lag $\tau$", r"$r(\tau)$"), "g",
             ("The cross-correlation at stages", r"$r(\tau)$ against $\tau$; dashed: the entangled step ($\tau = 0$)"),
             inset=(.56, .72))
    ax = axes[1][3]
    lcol = plt.cm.viridis(np.linspace(0, .9, len(LINE_LAGS)))
    hl = []
    for lag, c in zip(LINE_LAGS, lcol):
        j = int(np.where(TAU == lag)[0][0])
        ax.plot(st["x"], st["r"][:, j], color=c, lw=2.0 if lag == 0 else 1.4, zorder=3)
        if lag == 0:
            lo, hi = np.percentile(st["rb"][..., j], [5, 95], 0)
            ax.fill_between(st["x"], lo, hi, color=c, alpha=.18, lw=0, zorder=2)
        hl.append(Line2D([], [], color=c, lw=2.0 if lag == 0 else 1.4,
                         label=(rf"$\tau = {lag:+d}$" if lag else r"$\tau = 0$")))
    hl.append(Line2D([], [], color=COL["echo"], lw=1.2, label=r"ESN, $\tau = 0$"))
    ax.legend(handles=hl, loc="upper left", ncol=2, frameon=False, fontsize=FT - 1.5, handlelength=1.6,
              columnspacing=1.0)
    ax.plot(se["x"], se["r"][:, int(np.where(TAU == 0)[0][0])], color=COL["echo"], lw=1.2, zorder=2)
    ax.axhline(0, color="k", lw=.7, zorder=1)
    ax.axvline(acc0, color="k", lw=1.1, ls=":")
    ax.axvline(acc50, color="k", lw=1.1, ls=(0, (5, 2, 1, 2)))
    xaxis(ax)
    ax.set_ylabel(r"$r(\tau)$", fontsize=FL)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("h   Single lags through training\n"
                 r"Trained RNN $r(\tau)$, $\tau = -1..4$ (band: $\tau = 0$)", loc="left", fontsize=TS, linespacing=1.5)
    h = [Line2D([], [], color="k", lw=2.3, label="d: lags 0-4 summed"),
         Line2D([], [], color="k", lw=1.8, ls=(0, (5, 2)), label="d: lags 1-4 (errors after the entangled step)"),
         Line2D([], [], color="k", lw=2.0, ls=":", label="d: lag 0 (the entangled step's own error)"),
         Line2D([], [], color=COL["trained"], lw=2.3, label="Trained RNN (d, h bands: 5-95%)"),
         Line2D([], [], color=COL["echo"], lw=1.4, label="ESN"),
         Line2D([], [], color="k", marker="o", ms=6.5, mfc=COL["trained"], mec="k", lw=1.5,
                label="d: half-way (lags 1-4: half its peak, first; others: half the end value, for good)"),
         Line2D([], [], color="w", lw=1.8, path_effects=halo,
                label=r"a, b: the lag where $K$ peaks; e, f: where $r$ peaks ($\geq 0$)"),
         Line2D([], [], color="k", lw=1.1, ls=":", label=f"Trained RNN accuracy passes naive ({acc0:,.0f})"),
         Line2D([], [], color="k", lw=1.1, ls=(0, (5, 2, 1, 2)), label=f"... and is half-way to Bayes ({acc50:,.0f})")]
    fig.legend(handles=h, loc="center", bbox_to_anchor=(.5, 1.18 / H), ncol=3, frameon=False, fontsize=FT - .5,
               columnspacing=1.6, handlelength=2.3)
    fig.text(.5, .36 / H, r"Coherence $= \Vert\Sigma_k\delta_k\Vert^2/\Sigma_k\Vert\delta_k\Vert^2$ over the "
             "back-propagated errors $\\delta_k$ (both variables' errors at a step, sent back $k$ = 0-15 steps); $e$ = "
             "mean entanglement of the two variables; $r$ within episodes (steps 17-30, z-scored), averaged over "
             "episodes.\nSummed over ALL lags (weighted by the step pairs) the within-episode cross-correlation is "
             "exactly 0, so what is below 0 before the entangled step and after lag 4 mirrors the positive peak.\nData: "
             "5,000 probe episodes (50 test environments) x 14 steps per checkpoint, 114 checkpoints per network (seed "
             "20261029); bands and the peak lines' 95% rule: 1,000 resamples of the 50 environments.", ha="center",
             va="center", fontsize=FT - 1.5, color="k", linespacing=1.5)
    l14, r0 = st["half_lag1_4"], st["half_r0"]
    fig.suptitle(f"The learning signal on entangled steps through training: errors made 1-4 steps after an entangled "
                 f"step get coherent credit first (half its peak at {l14['p']:,.0f}),\nthe entangled step's own error "
                 f"later (half its end value at {r0['p']:,.0f}); the ESN never", y=1 - .38 / H, fontsize=TS + .5,
                 linespacing=1.4)
    path = fig_path("td_tt_credit_lags.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    what = sys.argv[1:] or ["numbers", "figure"]
    if "numbers" in what:
        numbers()
    if "figure" in what:
        figure()
