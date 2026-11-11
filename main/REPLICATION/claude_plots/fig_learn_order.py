"""Draft figure for John's 11-09 asks: (1) more of "the ESN succeeds up to naive, and our theory tracks where it fails";
(2) opposite changes of the Trained RNN and the ESN during training; (3) "can we understand if managing entanglement
precedes managing factorization regret or vice versa". Data: td_tt_order.py (summary.npz; per probe, step and variable
errors at every td_analysis checkpoint of every run).
-> main/REPLICATION/figure_bin/fig_learn_order.png (5.5 in wide, 300 dpi)

DEFINITIONS
  B_t, N_t, q_t: Bayes's, naive's and the network's marginal belief of one variable after t observations (probes.npz:
  10,000 probe episodes = 50 test environments x 100 cells x 2, both variables pooled). FR_t = KL(B_t || N_t).
  Loss added at step t = KL(B_t || q_t) - KL(B_t-1 || q_t-1) (both 0 at t = 0): the step's share of the network's excess
  log loss at T (they sum to it); naive's is FR_t - FR_t-1, Bayes's 0. Entanglement E_t = 1/2 J between naive's
  normalised step likelihood and Bayes's normalised increment (probes 'ent', the locked definition).
  a  Mean loss added per step in within-step deciles of E_t (all 30 steps; probes and variables pooled at each step),
     at six checkpoints per network; naive dashed (the theory: what a factorized learner pays).
  b  Slope of the network's ten decile means on naive's (least squares with intercept) at every checkpoint: 1 = pays
     naive's per-step losses, 0 = pays none (Bayes). Bands: 5-95% over td_tt_gap's 1,000 resamples of the 50
     environments (decile means and slope recomputed in each). ESN + MLPs = mlp55: the ESN's frozen LSTM with 5 trained
     layers before and 5 after it (parameter-matched to the Trained RNN).
  c  Naive's mean loss added per step on the 10 x 10 grid of within-step deciles of E_t (rows) and of FR_t-1 (columns,
     the state's FR before the step), steps 2-30. White: the matched cells, naive's mean in 0.03-0.05 nats; the
     columns' matched cells sit on a diagonal from (most entangled, low FR) to (middle entanglement, high FR).
  d  Trained RNN: loss added in the matched cells of each FR_t-1 decile (pooled), as a share of naive's (ratio of sums).
  e  Training episodes until that share stays below 1/2 (td_tt_gap.crossing, log-interpolated), per FR_t-1 decile and
     run; bars 5-95% over the resamples. Runs (seed 20261029: the same world, probes and episode stream): the Trained
     RNN (N = 1000, loss power .5), N = 1000 at loss power 1 (trained_p1), N = 200, N = 100, the ESN + MLPs.
RESULTS (11-09; printed by `numbers`)
  Naive creates 72.9% of its FR at T in the most entangled fifth of steps (loss added per step by decile -.005 ...
    .120 / .226 / .496 nats).
  b  ESN slope .15 (80) / .34 (320) / .43 (880) / .52 (2.4k) / .75 (8k) / .93 [.86, .99] (end): it ends paying naive's
     per-step losses where naive pays them (top decile .499 vs naive's .496; bottom deciles .02 vs ~0). Trained RNN
     ~.4-.6 from 60 to 600 (.9 at 20-40: its loss is far above naive's everywhere then, .65-1.15 nats per step), .31
     (880), .12 (2.4k), .03 (8k), .00 (end). ESN + MLPs .87 at 400 -> .04 (2.4k) -> 0.
  e  half of naive's loss removed in the matched cells (FR_t-1 decile 2 = the most entangled ... 9): Trained RNN
     3,466 [2,759-4,014] ... 1,069 [857-1,404]; loss power 1 3,678 ... 1,054; N = 200 21,519 ... 4,824; N = 100 never
     (deciles 2-5) ... 14,076-39,885; N = 50 and ESN never; ESN + MLPs flat, 1,905-3,411. End share of naive's: Trained
     RNN .028 (decile 2) ... -.017 (decile 9); ESN 1.16-2.99.
  Not hedging: in high-FR states with an unentangled step the Trained RNN's accuracy rises from chance to .54 (Bayes
  .55, naive .07) with Bayes's entropy (1.08 vs 1.08). Cross-correlation locks with surprise held out (scratch check):
  the update-error lock is the only opposite change (Trained .14 -> .04, ESN .14 -> .20); credit and activity locks
  rise in the Trained RNN only (ESN flat).
Run: MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig_learn_order.py"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.colors import LogNorm, Normalize                                  # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from matplotlib.patches import Rectangle                                          # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import td_tt_gap as gap                                                           # noqa: E402
import td_tt_order as to                                                          # noqa: E402

TS, FL, FT = 11, 10, 10
FIG_W, DPI = 5.5, 300
X0, XMIN, XMAX = 10, 8, 7e4
COL = {"trained": "#1f77b4", "echo": "#ff7f0e", "mlp55": "#8c564b"}
NAME = {"trained": "Trained RNN", "echo": "ESN", "mlp55": "ESN + MLPs", "trained_p1": "N = 1000, loss power 1",
        "trained_N200": "N = 200", "trained_N100": "N = 100", "trained_N50": "N = 50"}
RUN_E = (("trained", "#08306b", "o", "-"), ("trained_p1", "#08306b", "s", (0, (3, 1.5))),
         ("trained_N200", "#2171b5", "o", "-"), ("trained_N100", "#6baed6", "o", "-"), ("mlp55", "#8c564b", "D", "-"))
STAGE_X = (80, 320, 880, 2400, 8000, 39996)
SHADE = np.linspace(.3, 1., len(STAGE_X))
FRCOL = plt.cm.viridis(np.linspace(0, .95, to.K10))
LEVEL = .5


def load():
    return dict(np.load(os.path.join(to.OUT, "summary.npz")))


def dec_curves(S, run, C):
    """loss added per decile (K, 10) and its resamples (K, B, 10)"""
    num, n = S[f"dec_{run}"], S["n_dec"]
    return num.sum(1) / n.sum(0), np.einsum("be,ked->kbd", C, num) / (C @ n)[None]


def slope(y, xn):
    xc = xn - xn.mean(-1, keepdims=True)
    return ((y - y.mean(-1, keepdims=True)) * xc).sum(-1) / (xc ** 2).sum(-1)


def col_ratio(S, run, C):
    """loss added in the matched cells of each FR_t-1 decile / naive's: (K, 10) and resamples (K, B, 10)"""
    num, den = S[f"col_{run}"], S["naive_col"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return num.sum(1) / den.sum(0), np.einsum("be,ked->kbd", C, num) / (C @ den)[None]


def crossings(S, run, C):
    """per FR_t-1 decile: the training episode after which the matched-cell share stays below LEVEL; 5-95%"""
    x = S[f"x_{run}"]
    r, rb = col_ratio(S, run, C)
    full = np.array([gap.crossing(x, r[:, d], LEVEL, sign=-1.) if np.isfinite(r[:, d]).all() else np.nan
                     for d in range(to.K10)])
    boot = np.stack([gap.crossing(x, rb[:, :, d].T, LEVEL, sign=-1.) if np.isfinite(r[:, d]).all()
                     else np.full(rb.shape[1], np.nan) for d in range(to.K10)])          # (10, B)
    lo, hi = np.full(to.K10, np.nan), np.full(to.K10, np.nan)
    for d in range(to.K10):
        b = boot[d]
        if np.isfinite(r[:, d]).all() and np.isfinite(b).mean() > .95:
            lo[d], hi[d] = np.nanpercentile(b, [5, 95])
    return full, lo, hi, boot


def train_axis(ax, label=True, cross=None):
    ax.set_xscale("log")
    ax.set_xticks([X0, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.set_xlim(XMIN, XMAX)
    ax.minorticks_off()
    if label:
        ax.set_xlabel("training episodes", fontsize=FL, labelpad=1)
    if cross is not None:
        ax.axvline(cross, color="0.4", lw=.9, ls=":", zorder=5)


def head(ax, letter, name, formula):
    for text, dy, fs in ((formula, 6, FL), (f"{letter}   {name}" if letter else name, 21, TS)):
        if text:
            ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, dy), textcoords="offset points",
                        ha="left", va="baseline", fontsize=fs, annotation_clip=False)


def stage_index(x, v):
    i = int(np.argmin(np.abs(np.log(x) - np.log(v))))
    assert abs(x[i] - v) / v < .05, (v, x[i])
    return i


def passes_naive():
    """the first checkpoint at which the Trained RNN's goal accuracy at T is above naive's (fig_learn_mechanism)"""
    import fig_learn_mechanism as F
    P = F.probes()
    return F.passes_naive({"trained": F.series("trained")}, P)[0]


def figure():
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({"xtick.labelsize": FT, "ytick.labelsize": FT, "axes.labelsize": FL, "legend.fontsize": FT,
                         "axes.linewidth": .7, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "xtick.major.pad": 2, "ytick.major.pad": 2})
    S = load()
    C = gap.boot_counts()
    cross = passes_naive()
    naive_dec = S["naive_dec"].sum(0) / S["n_dec"].sum(0)
    naive_boot = (C @ S["naive_dec"]) / (C @ S["n_dec"])
    xl, wp, gp = .55, 2.32, .16                       # row 1: two panels sharing y
    wb, xr = 1.95, 3.42                                # rows 2-3: left / right panel width, right column's left edge
    t1, h = .62, 1.3
    t2 = t1 + h + 1.0
    t3 = t2 + h + 1.0
    t4 = t3 + h + 1.0
    yk = t4 + h + .62
    Wf, Hf = FIG_W, yk + .2
    fig = plt.figure(figsize=(Wf, Hf))

    def add(x, top, w, hh):
        return fig.add_axes([x / Wf, (Hf - top - hh) / Hf, w / Wf, hh / Hf])

    # ── a: loss added per step by the step's entanglement, at six checkpoints ───────────────────────────────────────
    dx = np.arange(1, to.K10 + 1)
    axa = [add(xl, t1, wp, h), add(xl + wp + gp, t1, wp, h)]
    for ax, run, cmap in ((axa[0], "trained", plt.cm.Blues), (axa[1], "echo", plt.cm.Oranges)):
        y, _ = dec_curves(S, run, C)
        x = S[f"x_{run}"]
        ax.axhline(0, color="0.75", lw=.8, zorder=1)
        for s, sh in zip(STAGE_X, SHADE):
            ax.plot(dx, y[stage_index(x, s)], color=cmap(sh), lw=1.5, marker="o", ms=2.2, zorder=3)
        ax.plot(dx, naive_dec, color="k", lw=1.4, ls=(0, (4, 2)), marker="D", ms=2.6, zorder=4)
        ax.set_xlim(.5, 10.5)
        ax.set_ylim(-.04, .62)
        ax.set_xticks([1, 5, 10])
        ax.set_xlabel("entanglement of the step (decile)", fontsize=FL, labelpad=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(.04, .96, NAME[run], transform=ax.transAxes, color=COL[run], fontsize=TS, va="top", ha="left")
        ax.text(9.55, naive_dec[-1] + .015, "naive", fontsize=FT, ha="right", va="bottom")
    axa[1].tick_params(labelleft=False)
    axa[0].set_ylabel("nats per step", fontsize=FL, labelpad=2)
    hs = [Line2D([], [], color=plt.cm.Greys(sh), lw=1.5) for sh in SHADE]
    axa[0].legend(hs, ["80", "320", "880", "2.4k", "8k", "40k"], loc="upper left", bbox_to_anchor=(0, .86),
                  frameon=False, fontsize=FT - 1, handlelength=1.2, labelspacing=.15, borderaxespad=.2,
                  title="training episodes", title_fontsize=FT - 1, alignment="left")
    head(axa[0], "a", "Loss added at a step, by the step's entanglement",
         r"$\mathrm{KL}(B_t\Vert q_t)-\mathrm{KL}(B_{t-1}\Vert q_{t-1})$;  naive's: $\mathrm{FR}_t-\mathrm{FR}_{t-1}$")

    # ── b: slope on naive's per-step losses through training ────────────────────────────────────────────────────────
    axb = add(xl, t2, wb, h)
    for run in ("mlp55", "echo", "trained"):
        x = S[f"x_{run}"]
        y, yb = dec_curves(S, run, C)
        sl = slope(y, naive_dec)
        lo, hi = np.percentile(slope(yb, naive_boot[None]), [5, 95], 1)
        axb.fill_between(x, lo, hi, color=COL[run], alpha=.25, lw=0, zorder=2)
        axb.plot(x, sl, color=COL[run], lw=1.6, zorder=3)
    axb.axhline(1, color="k", lw=.9, ls=(0, (4, 2)), zorder=1)
    axb.axhline(0, color="0.75", lw=.8, zorder=1)
    axb.text(3e2, 1.04, "naive", fontsize=FT, va="bottom", ha="center")
    axb.text(40, -.1, "Bayes", fontsize=FT, va="bottom", ha="center", color="0.4")
    axb.set_ylim(-.12, 1.18)
    train_axis(axb, cross=cross)
    axb.set_ylabel("slope on naive's", fontsize=FL, labelpad=2)
    axb.spines[["top", "right"]].set_visible(False)
    for run, xx, yy in (("echo", 6e4, .99), ("trained", 3.5e3, .3), ("mlp55", 3.5e3, .14)):
        axb.text(xx, yy, NAME[run], color=COL[run], fontsize=FT - 1, ha="right" if run == "echo" else "left",
                 va="bottom" if run == "echo" else "center")
    head(axb, "b", "Share of naive's loss paid", r"slope of a's curve on naive's")

    # ── c: naive's loss added on the entanglement x state-FR grid; the matched cells ─────────────────────────────────
    hc = h
    axc = add(xr + .12, t2, hc, hc)
    g = np.where(S["naive_grid"] > 0, S["naive_grid"], np.nan)
    cm = plt.cm.viridis.copy()
    cm.set_bad("#2b1a3d")
    nrm = LogNorm(3e-3, 1.4)
    axc.imshow(g, origin="lower", cmap=cm, norm=nrm, extent=(.5, 10.5, .5, 10.5), aspect="auto",
               interpolation="nearest")
    for i, j in zip(*np.where(S["matched"])):
        axc.add_patch(Rectangle((j + .5, i + .5), 1, 1, fill=False, ec="w", lw=1.1, zorder=4))
    axc.set_xticks([1, 5, 10])
    axc.set_yticks([1, 5, 10])
    axc.set_xlabel("FR before the step (decile)", fontsize=FL, labelpad=1)
    axc.set_ylabel("entanglement (decile)", fontsize=FL, labelpad=1)
    cax = add(xr + .12 + hc + .08, t2, .07, hc)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cm), cax=cax)
    cb.set_ticks([.01, .1, 1])
    cb.set_ticklabels([".01", ".1", "1"])
    cb.ax.tick_params(labelsize=FT - 1, length=2, pad=1.5)
    cb.outline.set_linewidth(.5)
    cax.text(4.6, .5, "nats per step", rotation=90, transform=cax.transAxes, ha="center", va="center", fontsize=FT - 1)
    cols = [d for d in range(to.K10) if S["matched"][:, d].any()]
    axc.text(2.7, 10.1, "entangled step,\nlow-FR state", fontsize=FT - 2, color="w", ha="left", va="top", zorder=6)
    axc.text(9.6, 3.3, "high-FR\nstate", fontsize=FT - 2, color="w", ha="right", va="top", zorder=6)
    head(axc, "c", "Steps compared in d, e", r"outlined: equal naive loss")

    # ── d: Trained RNN at the matched cells through training, by the state's FR; before / after passing naive ─────
    x = S["x_trained"]
    r, _ = col_ratio(S, "trained", C)
    pre, post = x <= cross, x >= cross
    axd = add(xl, t3, wb, h)
    axd2 = add(xr, t3, wb, h)
    for d in cols:
        axd.plot(x[pre], r[pre, d], color=FRCOL[d], lw=1.4, zorder=3)
        axd2.plot(x[post], r[post, d], color=FRCOL[d], lw=1.4, zorder=3)
    for ax in (axd, axd2):
        ax.axhline(1, color="k", lw=.9, ls=(0, (4, 2)), zorder=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xlabel("training episodes", fontsize=FL, labelpad=1)
    axd2.axhline(LEVEL, color="0.55", lw=.8, ls=":", zorder=1)
    axd2.axhline(0, color="0.75", lw=.8, zorder=1)
    axd.set_xlim(X0 * .9, cross * 1.05)
    axd.set_xticks([X0, 1e2, cross])
    axd.set_xticklabels(["0", "100", f"{cross:,.0f}"])
    axd.set_yscale("log")
    axd.set_ylim(.5, 40)
    axd.set_yticks([1, 3, 10, 30])
    axd.set_yticklabels(["1", "3", "10", "30"])
    axd.set_ylabel("share of naive's", fontsize=FL, labelpad=2)
    axd.text(X0 * 1.05, .8, "naive", fontsize=FT, va="top")
    axd2.set_xlim(cross * .95, XMAX)
    axd2.set_xticks([cross, 1e4])
    axd2.set_xticklabels([f"{cross:,.0f}", "10k"])
    axd2.set_ylim(-.65, 1.6)
    axd2.set_yticks([0, .5, 1])
    axd2.set_yticklabels(["0", ".5", "1"])
    axd2.text(cross * 1.05, 1.04, "naive", fontsize=FT, va="bottom")
    cdx = add(xr + wb - .62, t3 + .08, .5, .07)
    fcm = plt.matplotlib.colors.ListedColormap([FRCOL[d] for d in cols])
    cbd = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(cols[0] + .5, cols[-1] + 1.5), cmap=fcm), cax=cdx,
                       orientation="horizontal")
    cbd.set_ticks([cols[0] + 1, cols[-1] + 1])
    cbd.ax.tick_params(labelsize=FT - 1, length=2, pad=1)
    cbd.outline.set_linewidth(.5)
    cdx.text(.5, 1.6, "FR decile", transform=cdx.transAxes, ha="center", va="bottom", fontsize=FT - 1)
    head(axd, "d", "Before passing naive", r"Trained RNN, matched steps: $\Sigma\Delta\mathrm{KL}\,/\,\Sigma\Delta\mathrm{FR}$")
    head(axd2, "", "After passing naive", "the same, linear scale")

    # ── e: training episodes to half of naive's, by the state's FR, every trained run ──────────────────────────────
    axe = add(xl, t4, wb, h)
    ytop = 6e4
    for run, c, mk, ls in RUN_E:
        full, lo, hi, _ = crossings(S, run, C)
        d_ok = [d for d in cols if np.isfinite(full[d])]
        d_no = [d for d in cols if not np.isfinite(full[d])]
        xs = np.array(d_ok) + 1
        axe.plot(xs, full[d_ok], color=c, ls=ls, lw=1.3, marker=mk, ms=3.2, zorder=3)
        ok = np.isfinite(lo[d_ok])
        axe.vlines(xs[ok], lo[d_ok][ok], hi[d_ok][ok], color=c, lw=.8, zorder=2)
        if d_no:
            axe.plot(np.array(d_no) + 1, np.full(len(d_no), ytop), color=c, ls="none", marker=mk, ms=3.2, mfc="w",
                     clip_on=False, zorder=4)
    axe.set_yscale("log")
    axe.set_ylim(4e2, ytop)
    axe.set_yticks([1e3, 1e4])
    axe.set_yticklabels(["1k", "10k"])
    axe.minorticks_off()
    axe.set_xlim(cols[0] + .5, cols[-1] + 1.5)
    axe.set_xticks(np.array(cols) + 1)
    axe.set_xlabel("FR before the step (decile)", fontsize=FL, labelpad=1)
    axe.set_ylabel("training episodes", fontsize=FL, labelpad=1)
    axe.spines[["top", "right"]].set_visible(False)
    axe.text(cols[0] + 4.3, ytop, "never", fontsize=FT - 1, ha="left", va="center", color="0.3")
    head(axe, "e", "Half of naive's loss removed", "matched cells; ESN: never")
    hs = [Line2D([], [], color=c, ls=ls, marker=mk, ms=3.2, lw=1.3) for run, c, mk, ls in RUN_E]
    fig.legend(hs, ["Trained RNN (N = 1000)", "N = 1000, loss power 1", "N = 200", "N = 100", "ESN + MLPs"],
               loc="center left", bbox_to_anchor=((xr + .1) / Wf, (Hf - t4 - h / 2) / Hf), ncol=1, frameon=False,
               fontsize=FT - 1, handlelength=1.8, labelspacing=.35)
    fig.text(.5, .1 / Hf, "50 environments x 200 probes x 114 checkpoints per run; one run each; dotted: the Trained "
             "RNN passes naive", ha="center", va="baseline", fontsize=FT - 2, color="0.3")
    out = fig_path("fig_learn_order.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("saved", out)
    return out


def numbers():
    S = load()
    C = gap.boot_counts()
    naive_dec = S["naive_dec"].sum(0) / S["n_dec"].sum(0)
    naive_boot = (C @ S["naive_dec"]) / (C @ S["n_dec"])
    print(f"naive's FR at T created in the top entanglement fifth of steps: {float(S['share_top5']):.3f}")
    print("E decile means:", " ".join(f"{v:.3f}" for v in S["E_dec"]))
    print("naive loss added per decile:", " ".join(f"{v:+.3f}" for v in naive_dec))
    for run in ("trained", "echo", "mlp55", "trained_p1", "trained_N200", "trained_N100", "trained_N50"):
        x = S[f"x_{run}"]
        y, yb = dec_curves(S, run, C)
        sl, sb = slope(y, naive_dec), slope(yb, naive_boot[None])
        lo, hi = np.percentile(sb, [5, 95], 1)
        k = int(np.argmax(sl))
        pts = "  ".join(f"{s}: {sl[stage_index(x, s)]:+.2f} [{lo[stage_index(x, s)]:+.2f}, {hi[stage_index(x, s)]:+.2f}]"
                        for s in STAGE_X)
        print(f"{run:13s} slope max {sl[k]:.2f} at {x[k]:.0f}; {pts}")
        if run in ("trained", "echo"):
            for s in STAGE_X:
                print(f"     {s:6d}: " + " ".join(f"{v:+.3f}" for v in y[stage_index(x, s)]))
    print("matched cells (E decile, FR_t-1 decile):", [(int(i) + 1, int(j) + 1) for i, j in zip(*np.where(S["matched"]))])
    print("matched columns' mean E:", " ".join(f"{v:.3f}" for v in S["E_col"]))
    print("naive's mean loss added in the matched cells of each column:",
          " ".join(f"{v:.3f}" for v in S["naive_col"].sum(0) / np.maximum(S["n_col"].sum(0), 1)))
    for run in ("trained", "trained_p1", "trained_N200", "trained_N100", "trained_N50", "mlp55", "echo"):
        full, lo, hi, boot = crossings(S, run, C)
        print(f"{run:13s} half-crossing by FR_t-1 decile: " + "  ".join(
            f"{d + 1}: {full[d]:6.0f} [{lo[d]:6.0f}-{hi[d]:6.0f}]" for d in range(to.K10) if S["matched"][:, d].any()))
        r, _ = col_ratio(S, run, C)
        print(f"{'':13s} end share by decile: " + " ".join(f"{v:+.3f}" for v in r[-1] if np.isfinite(v)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "numbers":
        numbers()
    else:
        figure()
