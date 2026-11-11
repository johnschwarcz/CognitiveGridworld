"""Entanglement-locked cross-correlations through training, at every lag (John 11-08: panel j of td_summary "is a really
[good] way to see the cross-correlations emerging through training, can we make a figure of these for all of the
promising cross-correlations that could help us show the mechanism of learning a compositional representation?").
-> main/REPLICATION/figure_bin/td_xcorr_lags.png       beliefs | activity | update (the locked measures)
   main/REPLICATION/figure_bin/td_xcorr_lags_more.png  the other candidates: cell state and the top / bottom 10% bands
Statistic (the locked one, gradflow_analysis _zs / _lagged): each measure and the entanglement are z-scored within
each episode over steps 17-30, and r(tau) is the mean product of the measure at t + tau and the entanglement at t
(tau > 0: the measure comes after the entangled step). Pairing:
  activity, update, cell state, bands: the whole network's measure against rev, the mean of the two variables'
    entanglement (the locked convention);
  beliefs: each variable's measure against that variable's own entanglement, then the mean over the two variables
    (average after the statistic).
Rows: the Trained RNN and the ESN through training (seed 20261029: one world, identical initial weights; training
episodes = episodes with an update, the initial network drawn at 10 = "0"; dotted: the Trained RNN passes naive at 880),
then each map's columns at the training stages of td_summary k (40, 200, 1,040, end; the ESN at the end), one y-scale
per group. Grey references (beliefs; diamond: tau = 0): naive's own update error, and memory-free Bayes's update error = what
Bayes loses when its memory is cut to the last step (Bayes's memory use).
Measures (per particle-step):
  update error      the network's step-t update applied to Bayes's belief at t-1, Jeffreys / 2 against Bayes's belief
                    at t (td_entanglement_memcut, per variable)
  departure from naive  Jeffreys / 2 between the network's step-t update (the change of its log belief) and naive's
                    step-t likelihood; for Bayes this IS the entanglement (r = 1 at tau = 0); logq is stored in fp16
  memory use        the update error with the memory cut to the last step (the cell rerun from its start state over
                    input t) minus intact
  norm, eff. dim., turn  ||u_t||, the projection PR on the checkpoint's basis, 1 - cos(u_t, u_{t-1}) (td_analysis)
  coherence, dW energy, cross-band share, dW eff. dim.  the update measures of the error at each step (td_analysis
                    credit pass, 5,000 probes; the ESN's: what its frozen recurrent weights would receive)
  cell state        ||c_t|| and 1 - cos(c_t, c_{t-1})
  top / bottom band share  u's energy share in the PCs holding the first / last 10% of the variance (band 0 / 9 of
                    td_analysis' 10% bands; both are filled at every checkpoint)
Colour: one viridis scale per measure group (bright = more), the same for both networks.
Findings 11-08 (r at tau = 0 unless said; Trained RNN at 40 / 200 / 1,040 / end of training):
  departure from naive  .01 / .13 / .14 / .66 (ESN -.02 -> .18; Bayes 1 by construction)
  update error      .24 / .30 / .25 / .14 (ESN .30 -> .36; naive's own .50)
  memory use        -.02 / -.01 / .12 / .34 (ESN ~0; Bayes with its memory cut to the last step .37)
  norm              -.06 / -.12 / a peak at tau +2 (.14) / .36 (ESN -.17 -> -.22); eff. dim. .07 / .11 / .07 / .25
                    (ESN <= .03); turn .22-.30 at every stage, the ESN's .31-.32 (not specific to learning)
  coherence, dW energy, cross-band share  ~0 until ~1k, a peak at tau +2 at 1,040 (.09 each), on the step at the end
                    (.34 / .18 / .28); the ESN's ~-.04
  dW eff. dim.      .35 / .19 / -.01 / -.16 (+.16 at tau +2); ESN .38-.40
  more              top-band share -.09 / -.01 / -.08 / -.29 (the drop only after ~10k; ESN .18 -> .01); bottom-band
                    share .04 / .07 / .06 / .19 (+.23 at tau +1; ESN .28 -> .12); cell-state norm ~0 (ESN -.17 ->
                    -.23); cell-state turn as u's
Data: td_analysis ck_*.npz, ent_memcut_var_<run>.npz, probes.npz; the new statistics are cached in
main/DATA/experimental/training_dynamics/analysis/xcorr_lags_<run>.npz (~2 min per network to build).
Run from the repository root (CPU):
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_xcorr_lags.py"""
import os
import sys
import glob
import textwrap
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.colors import Normalize, LinearSegmentedColormap, LogNorm          # noqa: E402
from matplotlib.cm import ScalarMappable                                          # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import plot_training_dynamics as ptd                                              # noqa: E402
import fig_entanglement_driver as fd                                              # noqa: E402
import gradflow_analysis as ga                                                    # noqa: E402
import stage_bands as sb                                                          # noqa: E402

AN = ptd.AN
RUNS = ("trained", "echo")
COL = {"trained": "#1f77b4", "echo": "#ff7f0e"}
NAME = {"trained": "Trained RNN", "echo": "ESN"}
WIN = ptd.WIN
TAU = np.asarray(ga.TAU)
X0 = 10
STAGES = (40, 200, 1000, 39996)                    # td_summary k: the nearest checkpoint on the log axis
TS, FL, FT = 14, 12.5, 11.5
EXTRA = ("err", "dev", "mem", "norm_c", "turn_c", "band_top", "band_bot")
REF = {"err": ("naive", "naive", (0, (4, 2))), "mem": ("stepwise", "Bayes, memory cut to the last step", (0, (1, 1.5)))}

# (key, group, name, formula); keys ("act", i) / ("upd", i) are plot_training_dynamics' series, the rest the new cache
MAIN = (("err", "Beliefs", "Update error", r"$\frac{1}{2}J(B_{t-1}+\Delta^{net}_t,\ B_t)$"),
        ("dev", "Beliefs", "Departure from naive", r"$\frac{1}{2}J(\Delta^{net}_t,\ \Delta^{naive}_t)$"),
        ("mem", "Beliefs", "Memory use", r"error$_{cut}-$error$_{intact}$"),
        (("act", 0), "Activity", "Norm", r"$\Vert u_t\Vert_2$"),
        (("act", 1), "Activity", "Effective dim.", r"$(\Sigma_{pc}\,proj^2)^2/\Sigma_{pc}\,proj^4$"),
        (("act", 2), "Activity", "Activity turn", r"$1-\cos(u_t,u_{t-1})$"),
        (("upd", 0), "Update", "Credit coherence", r"$\Vert\Sigma_k\delta_k\Vert^2/\Sigma_k\Vert\delta_k\Vert^2$"),
        (("upd", 1), "Update", r"$\Delta W$ energy", r"$\Vert\Delta W\Vert^2$"),
        (("upd", 3), "Update", "Cross-band share", r"$1-\Sigma_B\Vert\Delta W_B\Vert^2/\Vert\Delta W\Vert^2$"),
        (("upd", 2), "Update", r"$\Delta W$ effective dim.", r"$(\Sigma_i\sigma_i^2)^2/\Sigma_i\sigma_i^4$"))
MORE = (("norm_c", "Cell state", "Cell-state norm", r"$\Vert c_t\Vert_2$"),
        ("turn_c", "Cell state", "Cell-state turn", r"$1-\cos(c_t,c_{t-1})$"),
        ("band_top", "Bands", "Top-band share", r"$\Vert P_{top}(u_t-\mu)\Vert^2/\Vert u_t-\mu\Vert^2$"),
        ("band_bot", "Bands", "Bottom-band share", r"$\Vert P_{bot}(u_t-\mu)\Vert^2/\Vert u_t-\mu\Vert^2$"))

# layout, inches
PW, PH, PH3, GAP, LM, RM = 2.6, 2.35, 2.1, .62, 1.3, .3


# ── data ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _xc_var(y, ent):
    """per variable against its own entanglement, then the mean over the two variables: (n, T, 2) -> (TAU,)"""
    return np.mean([ptd._xcorr(y[:, WIN, k], ent[:, WIN, k]) for k in range(2)], 0)


def _xc_band(b, rev):
    """a band share (n, T) that is NaN where the band is empty: the statistic only where it is filled at every step"""
    b = b[:, WIN].astype(np.float64)
    if np.isnan(b).any():
        return np.full(len(TAU), np.nan)
    return ptd._xcorr(b, rev[:, WIN])


def extra(run, P):
    """the new statistics of every checkpoint, (n_ck, TAU) each, cached in xcorr_lags_<run>.npz"""
    files = sorted(glob.glob(os.path.join(AN, run, "ck_*.npz")))
    eps = np.array([int(os.path.basename(f)[3:8]) for f in files])
    cache = os.path.join(AN, f"xcorr_lags_{run}.npz")
    if os.path.exists(cache):
        z = dict(np.load(cache))
        if np.array_equal(z["episode"], eps) and all(k in z for k in EXTRA):
            return z
    m = np.load(os.path.join(AN, f"ent_memcut_var_{run}.npz"))
    assert np.array_equal(m["episode"], eps), f"{run}: memory-cut checkpoints differ"
    ent = P["ent"].astype(np.float64)
    rev = ent.mean(-1)
    inc_n = P["inc_N"].astype(np.float64)
    out = {k: [] for k in EXTRA}
    for i, f in enumerate(files):
        ei, ec = m["err_int"][i].astype(np.float64), m["err_cut1"][i].astype(np.float64)
        out["err"].append(_xc_var(ei, ent))
        out["mem"].append(_xc_var(ec - ei, ent))
        z = np.load(f)
        lq = z["logq"].astype(np.float64)                                              # (n, T, 2, R) log beliefs
        start = np.full_like(lq[:, :1], -np.log(lq.shape[-1]))
        out["dev"].append(_xc_var(fd._jeff2(np.diff(np.concatenate([start, lq], 1), axis=1), inc_n), ent))
        out["norm_c"].append(ptd._xcorr(z["norm_c"][:, WIN], rev[:, WIN]))
        out["turn_c"].append(ptd._xcorr(z["turn_c"][:, WIN], rev[:, WIN]))
        band = z["band_u"]
        out["band_top"].append(_xc_band(band[..., 0], rev))
        out["band_bot"].append(_xc_band(band[..., -1], rev))
        print(f"{run} {eps[i]:>6}", flush=True)
    out = {k: np.array(v) for k, v in out.items()}
    out["episode"] = eps
    np.savez(cache, **out)
    return out


def data():
    """D[run][measure] (n_ck, TAU) and D[run]["x"]; R: the observers' reference profiles; where the Trained RNN passes
    naive"""
    P = ptd.probes()
    D = {}
    for run in RUNS:
        S = ptd.series(run, P)
        E = extra(run, P)
        assert np.array_equal(S["episode"], E["episode"]), f"{run}: checkpoints differ"
        d = {k: E[k] for k in EXTRA}
        for i in range(3):
            d[("act", i)] = S["xc_act"][:, i]
        for i in range(4):
            d[("upd", i)] = S["xc_upd"][:, i]
        d["x"] = np.asarray(S["x"], float)
        D[run] = d
    I = fd.inputs()
    R = {k: _xc_var(I[src], I["ent"]) for k, (src, _, _) in REF.items()}
    return D, R, fd.passes_naive(P)


# ── figure ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def log_edges(x):
    lx = np.log(x)
    mid = (lx[1:] + lx[:-1]) / 2
    return np.exp(np.concatenate([[lx[0] - (mid[0] - lx[0])], mid, [lx[-1] + (lx[-1] - mid[-1])]]))


def xaxis(ax, xe, label):
    ax.set_xscale("log")
    ax.set_xticks([X0, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.set_xlim(xe[0], xe[-1])
    ax.minorticks_off()
    if label:
        ax.set_xlabel("training episodes", fontsize=FL)


def groups_of(spec):
    out = []
    for j, (_, grp, _, _) in enumerate(spec):
        if not out or out[-1][0] != grp:
            out.append([grp, j, j])
        out[-1][2] = j
    return out


def group_norms(D, spec):
    """one colour scale per group over both networks: from the lowest r (at most 0) to the highest"""
    lim = {}
    for key, grp, _, _ in spec:
        v = np.concatenate([D[r][key].ravel() for r in RUNS])
        lo, hi = lim.get(grp, (0., 0.))
        lim[grp] = (min(lo, np.nanmin(v)), max(hi, np.nanmax(v)))
    return {g: Normalize(np.floor(lo * 20) / 20, np.ceil(hi * 20) / 20) for g, (lo, hi) in lim.items()}


def stage_rows(d, key):
    return [int(np.argmin(np.abs(np.log(d["x"]) - np.log(xv)))) for xv in STAGES]


def draw(D, R, cross, spec, name, note):
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11})
    norms = group_norms(D, spec)
    groups = groups_of(spec)
    nc = len(spec)
    W = LM + nc * PW + (nc - 1) * GAP + RM
    note_lines = textwrap.wrap(note, int(W * 12.5))
    y_note = .15 + .23 * len(note_lines)                                             # top of the note, the key above it
    y_leg = y_note + .45
    y2 = y_leg + .8                                                             # bottoms of the three rows
    y1 = y2 + PH3 + 1.1
    y0 = y1 + PH + .5
    ytop = y0 + PH
    H = ytop + 1.55
    fig = plt.figure(figsize=(W, H))
    xl = lambda j: LM + j * (PW + GAP)                                                 # noqa: E731
    ax = np.array([[fig.add_axes([xl(j) / W, y / H, PW / W, h / H]) for j in range(nc)]
                   for y, h in ((y0, PH), (y1, PH), (y2, PH3))])
    blues = LinearSegmentedColormap.from_list("blues_trunc", plt.cm.Blues(np.linspace(.35, 1., 256)))
    snorm = LogNorm(STAGES[0], STAGES[-1])
    te = np.concatenate([TAU - .5, [TAU[-1] + .5]])
    for j, (key, grp, title, formula) in enumerate(spec):
        for i, run in enumerate(RUNS):                                                # the maps
            a, d = ax[i, j], D[run]
            xe = log_edges(d["x"])
            a.pcolormesh(xe, te, np.ma.masked_invalid(d[key]).T, cmap="viridis", norm=norms[grp], shading="flat",
                         rasterized=True)
            a.axvline(cross, color="w", lw=1.1, ls=":", zorder=5)
            sb.strip(a)
            xaxis(a, xe, label=i == 1)
            a.set_yticks([-8, -4, 0, 4, 8])
            if j == 0:
                a.set_ylabel(r"lag $\tau$ (steps after)", fontsize=FL)
        ax[0, j].set_title(f"{title}\n{formula}", loc="left", fontsize=TS, pad=9)
        a, d = ax[2, j], D["trained"]                                                 # the maps' columns at stages
        for k in stage_rows(d, key):
            lab = f"{d['x'][k]:,.0f}" + (" (end)" if k == len(d["x"]) - 1 else "")
            a.plot(TAU, d[key][k], color=blues(snorm(d["x"][k])), lw=1.9, marker="o", ms=3, label=lab, zorder=3)
        a.plot(TAU, D["echo"][key][-1], color=COL["echo"], lw=1.7, marker="s", ms=3, label="ESN, end", zorder=2)
        if key in R:
            a.plot(TAU, R[key], color="0.3", lw=1.8, ls=REF[key][2], marker="D", ms=6, mfc="w", mew=1.5,
                   markevery=[int(np.flatnonzero(TAU == 0)[0])], zorder=5)
        a.axhline(0, color="k", lw=.7)
        a.axvline(0, color="k", lw=.8, ls=(0, (2, 2)))
        a.set_xticks([-8, -4, 0, 4, 8])
        a.set_xlim(-8.5, 8.5)
        a.set_xlabel(r"lag $\tau$ (steps after)", fontsize=FL)
        if j == 0:
            a.set_ylabel(r"$r$ with entanglement", fontsize=FL)
        a.spines[["top", "right"]].set_visible(False)
    for grp, j0, j1 in groups:                                                        # one y-scale per group
        lines = [l.get_ydata() for j in range(j0, j1 + 1) for l in ax[2, j].get_lines() if len(l.get_ydata()) > 2]
        lo, hi = np.nanmin(np.concatenate(lines)), np.nanmax(np.concatenate(lines))
        pad = .06 * (hi - lo)
        for j in range(j0, j1 + 1):
            ax[2, j].set_ylim(lo - pad, hi + pad)
        x0, x1 = xl(j0), xl(j1) + PW                                                  # colour bar + group name
        cax = fig.add_axes([(x0 + .15 * (x1 - x0)) / W, (ytop + .9) / H, .7 * (x1 - x0) / W, .14 / H])
        cb = plt.colorbar(ScalarMappable(norm=norms[grp], cmap="viridis"), cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=FT)
        cb.ax.axvline(0, color="w", lw=1.4)
        cb.ax.xaxis.set_label_position("top")
        cb.set_label(f"{grp}:  " + r"$r$ with entanglement", fontsize=16, labelpad=6)
    for i, (run, lab) in enumerate((("trained", NAME["trained"]), ("echo", NAME["echo"]), (None, "Columns at stages"))):
        y, h = ((y0, PH), (y1, PH), (y2, PH3))[i]
        fig.text(.3 / W, (y + h / 2) / H, lab, rotation=90, va="center", ha="center", fontsize=16 if run else 14,
                 color=COL[run] if run else "0.3")
    handles, labels = ax[2, 0].get_legend_handles_labels()                           # two keys side by side
    others = (handles[len(STAGES):], labels[len(STAGES):])
    for k, (src, lab, ls) in REF.items():
        if any(s[0] == k for s in spec):
            others[0].append(Line2D([], [], color="0.3", lw=1.8, ls=ls, marker="D", ms=6, mfc="w", mew=1.5))
            others[1].append(lab)
    for h_ in sb.legend_handles():
        others[0].append(h_)
        others[1].append("stage: " + h_.get_label())
    kw = dict(frameon=False, fontsize=FT + .5, title_fontsize=FT + .5, handlelength=2.2, columnspacing=1.4)
    fig.legend(handles[:len(STAGES)], labels[:len(STAGES)], loc="center right", bbox_to_anchor=(.5, y_leg / H),
               ncol=len(STAGES), title="Trained RNN after (training episodes):", alignment="left", **kw)
    fig.legend(*others, loc="center left", bbox_to_anchor=(.51, y_leg / H), ncol=len(others[0]), title=" ", **kw)
    fig.text(.5, y_note / H, "\n".join(note_lines), ha="center", va="top", fontsize=FL - 1,
             color="0.3", linespacing=1.5)
    path = fig_path(name)
    fig.savefig(path, dpi=125)
    plt.close(fig)
    print("saved", path)
    return path


NOTE = ("10,000 probe episodes (50 environments x 100 latent pairs x 2 trajectories) at each of 114 checkpoints per "
        "network (update measures: 5,000 of them); steps 17-30, z-scored within each episode; r(tau) = mean of the "
        "measure at t + tau times the entanglement at t. Beliefs: each variable against its own entanglement, then "
        "averaged over the two; all others: the whole network against the two variables' mean entanglement. "
        "Dotted: the Trained RNN passes naive (880 training episodes).")


def figure():
    D, R, cross = data()
    return [draw(D, R, cross, MAIN, "td_xcorr_lags.png", NOTE + " Departure from naive: for Bayes it is the "
                 "entanglement itself (r = 1 at tau = 0). ESN update measures: what its frozen recurrent weights would "
                 "receive."),
            draw(D, R, cross, MORE, "td_xcorr_lags_more.png", NOTE + " Bands: the PCs (of the checkpoint's own basis) "
                 "holding the first / last 10% of u's variance.")]


if __name__ == "__main__":
    figure()


# ── replication across the width / loss runs (no memory-cut caches there: the belief measures are departure only) ───

REP_RUNS = ("trained", "trained_p1", "trained_N200", "trained_N100", "trained_N50")
REP_COL = {"trained": "#08306b", "trained_p1": "#2171b5", "trained_N200": "#4292c6", "trained_N100": "#6baed6",
           "trained_N50": "#9ecae1"}
REP_LAB = {"trained": "N = 1000", "trained_p1": "N = 1000, loss power 1", "trained_N200": "N = 200",
           "trained_N100": "N = 100", "trained_N50": "N = 50"}
REP_SPEC = (("dev", "Departure from naive"), (("act", 0), "Norm"), (("act", 1), "Effective dim."),
            (("act", 2), "Activity turn"), (("upd", 0), "Credit coherence"), (("upd", 1), r"$\Delta W$ energy"),
            (("upd", 3), "Cross-band share"), (("upd", 2), r"$\Delta W$ effective dim."))


def dev_series(run, P):
    """departure from naive's r with each variable's own entanglement at every lag, every checkpoint (n_ck, TAU),
    cached in xcorr_dev_<run>.npz"""
    files = sorted(glob.glob(os.path.join(AN, run, "ck_*.npz")))
    eps = np.array([int(os.path.basename(f)[3:8]) for f in files])
    cache = os.path.join(AN, f"xcorr_dev_{run}.npz")
    if os.path.exists(cache):
        z = dict(np.load(cache))
        if np.array_equal(z["episode"], eps):
            return z["dev"]
    ent = P["ent"].astype(np.float64)
    inc_n = P["inc_N"].astype(np.float64)
    out = []
    for f in files:
        lq = np.load(f)["logq"].astype(np.float64)
        start = np.full_like(lq[:, :1], -np.log(lq.shape[-1]))
        out.append(_xc_var(fd._jeff2(np.diff(np.concatenate([start, lq], 1), axis=1), inc_n), ent))
    out = np.array(out)
    np.savez(cache, episode=eps, dev=out)
    return out


def replication():
    """td_xcorr_lags_rep.png: r with entanglement at lag 0 (top) and lag +2 (bottom) through training on each run's own
    clock (training episodes / its passing-naive time), every trained run"""
    import td_tt_count_code as cc
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    P = ptd.probes()
    i0, i2 = int(np.flatnonzero(TAU == 0)[0]), int(np.flatnonzero(TAU == 2)[0])
    fig, ax = plt.subplots(2, len(REP_SPEC), figsize=(3.0 * len(REP_SPEC) + 1, 7.4), sharex=True)
    fig.subplots_adjust(left=.05, right=.995, top=.86, bottom=.17, wspace=.35, hspace=.32)
    for run in REP_RUNS:
        S = ptd.series(run, P)
        d = {"dev": dev_series(run, P)}
        for i in range(3):
            d[("act", i)] = S["xc_act"][:, i]
        if "xc_upd" in S:
            for i in range(4):
                d[("upd", i)] = S["xc_upd"][:, i]
        c0 = cc.cached_series(run)["c0"]
        xx = np.asarray(S["x"], float) / c0
        for j, (key, name) in enumerate(REP_SPEC):
            if key not in d:
                continue
            for row, il in ((0, i0), (1, i2)):
                ax[row, j].plot(xx, d[key][:, il], color=REP_COL[run], lw=1.5, label=REP_LAB[run])
    for j, (key, name) in enumerate(REP_SPEC):
        ax[0, j].set_title(name, loc="left", fontsize=12.5)
        for row in range(2):
            a = ax[row, j]
            a.axhline(0, color="k", lw=.6)
            a.axvline(1, color=".55", ls=":", lw=1)
            a.set_xscale("log")
            a.spines[["top", "right"]].set_visible(False)
        ax[1, j].set_xlabel("training / passing-naive time")
    ax[0, 0].set_ylabel(r"$r$ with entanglement, lag 0")
    ax[1, 0].set_ylabel(r"$r$ at lag +2 (2 steps after)")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(.5, .04))
    fig.suptitle("The entanglement-locked signatures through training, replicated across widths and losses "
                 "(each run on its own clock; dotted: it passes naive)", fontsize=13, y=.97)
    path = fig_path("td_xcorr_lags_rep.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)
