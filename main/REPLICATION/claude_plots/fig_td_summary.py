"""One figure of the training-dynamics panels John selected (11-01: "create 1 figure with all of them please (using the
same scaling for the x-axis across all)"; "fewer rows and more columns ... the x-axes line up along the entire
columns ... each panel the same width"; then: no share panels, no learning-signal r panel, ridge R^2 from u at step
30 only, update-measure heatmaps for the Trained RNN only; then: the SII score replaced by the correlation between
factorization regret and accuracy, no update-error panel, the network's PCA spectrum and PR next to the read-out's).
-> main/REPLICATION/figure_bin/td_summary.png
Layout: 4 rows x 7 equal columns, every x-axis the same (except k: lag): training episodes (episodes with an update -
a test episode trains nothing), log scale 8 to 70,000, the initial network at 10 (labelled 0); dotted line: where the
Trained RNN passes naive. Row 1: both networks (Trained RNN blue, ESN orange; seed 20261029: one world, identical
initial weights). Rows 2-3: Trained RNN above ESN. Row 4: Trained RNN.
  a accuracy at T against Bayes, memory-free (step-wise) Bayes and naive
  b on the probes where naive is wrong and Bayes right: how often the network gives naive's answer
  c Pearson r between each variable's factorization regret at T (FR = [KL(Bayes||naive) + KL(naive||Bayes)] / 2 of its
    marginal beliefs) and its own correctness at T, over the 10,000 probes x both variables (= the original fig1 (e),
    which used the goal variable's, up to noise); dotted: Bayes (-.02) and naive (-.43). Averaging FR and correctness
    over the two variables first (the 11-01 draft) adds cross terms: one variable's FR correlates +.13 to +.17 with the
    other variable's correctness for every observer, which flipped the sign (Bayes +.12).
  d ridge R^2 of the counts n_t (the observations summed so far) decoded from u at step 30 (fitted on 25 environments,
    tested on the other 25)
  e memory use of each variable, per fifth of that variable's own entanglement (both variables' values ranked
    together within each step: 4,000 of every step per fifth): the update error with the network's memory cut to the
    last step (the cell rerun from its start state over input t) minus intact; update error = Jeffreys / 2 between
    Bayes's belief at t-1 updated by the network's step-t logit increment and Bayes's belief at t (per variable);
    right edge: what memory-free Bayes lacks (diamonds). Audit 11-01: the first version averaged the errors over the
    two variables and split them by rev (their mean entanglement); paired, the Trained RNN's flip moves from
    880-1,120 to 800-1,040 training episodes (halves agree 100% both ways)
  f the network's beliefs: the PCA spectrum of its marginal beliefs as probabilities (20 numbers per particle) over
    the 10,000 probes at each step (18 PCs: each variable's probabilities sum to 1, which fixes 2 directions at zero
    variance), fractions of the variance and their participation ratio (white) per step, then averaged over the 30
    steps; right edge: Bayes's (12.9) and naive's (9.2) beliefs' PR on the same probes. End: Trained RNN 12.9, ESN
    10.7 (start 4.3)
  g the network state's PCA spectrum per step: each PC's fraction of u_t's variance over the 10,000 probes and its
    participation ratio PR = (sum lambda)^2 / sum lambda^2 (white; in PCs, on the same axis), averaged over the 30
    steps. End of training: PR Trained RNN 57, ESN 18 (both 10.7 at initialization; the Trained RNN's collapses to
    1.3 after its first 20 training episodes, the early saturation, then grows)
    f and g, averaging order (audit 11-01): the 11-01 draft pooled all steps into one cloud, which adds the drift of
    the average belief / state across steps as dimensions (state at initialization 1.8, end Trained 74, ESN 11;
    beliefs Bayes 14.2, naive 10.1, Trained 14.2, ESN 12.0)
  h, i belief grids (each cell's mean posterior means over 24 fresh trajectories, 50 environments x 100 cells, 19
    checkpoints; a grid triangle is folded when the grid is turned over there): % folded at T where naive's grid is
    folded and elsewhere; % folded after every observation t (white: 10%; dashed: Bayes reaches 10% at step 7)
  j credit coherence (of both variables' step errors, the locked convention) and its within-episode r with
    entanglement (rev: the mean of the two variables) at every lag tau
  k (John 11-02: "add panel g into the summary figure next to the cross-corr heatmap"): j's columns at training stages,
    r against tau at 40, 200, 1,040 and 39,996 training episodes (John: "40, 200, 1K, end"; the nearest checkpoints on
    the log axis), each curve labelled with its training episodes; the ESN at the end. The positive r moves from lags
    3-4 (~300-500) onto the entangled step (lag 0, from ~2k); td_tt_credit_lags has the bootstrap intervals and the
    cumulated version
  l-p the update measures' distributions over (probe, error step) at every checkpoint (white: median; dashed: 10% /
    90%); the ESN's are left out (the gradient its frozen LSTM would receive barely changes: coherence ~3.4, eff.
    dimension 1.0, cross-band share ~0; its update energy grows ~1,000x with its read-out gain)
Data: plot_training_dynamics (a, b, d, j, k), fig_entanglement_driver (e), td_summary_data (c, f-i, l-p; cached in
main/DATA/experimental/training_dynamics/summary_data).
Run from the repository root (CPU, ~1 min once td_summary_data's caches exist):
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig_td_summary.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm, Normalize     # noqa: E402
from matplotlib.cm import ScalarMappable                                          # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import plot_training_dynamics as ptd                                              # noqa: E402
import fig_entanglement_driver as fd                                              # noqa: E402
import td_summary_data as sd                                                      # noqa: E402
import gradflow_analysis as ga                                                    # noqa: E402

RUNS = ("trained", "echo")
COL = {"trained": "#1f77b4", "echo": "#ff7f0e"}
NAME = {"trained": "Trained RNN", "echo": "ESN"}
X0, XMIN, XMAX = 10, 8, 7e4
XR1, XR2, XSEP = 4.8e4, 6e4, 4.4e4                    # right-edge reference columns and their separator
KCOL = plt.cm.Purples(np.linspace(.3, 1., 5))         # entanglement fifths, least (light) to most (dark)
TS, FL, FT = 13, 11, 10                               # title, axis-label, legend sizes
NORM = {"lag": Normalize(-.06, .34), "bel": LogNorm(1e-4, 1), "spec": LogNorm(1e-5, 1), "fold": Normalize(0, 52),
        "dist": LogNorm(1e-4, 1)}


def xaxis(ax, label):
    ax.set_xscale("log")
    ax.set_xticks([X0, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.set_xlim(XMIN, XMAX)
    ax.minorticks_off()
    if label:
        ax.set_xlabel("training episodes", fontsize=FL)
    ax.spines[["top", "right"]].set_visible(False)


def edges(x):
    lx = np.log(x)
    mid = (lx[1:] + lx[:-1]) / 2
    return np.exp(np.concatenate([[lx[0] - (mid[0] - lx[0])], mid, [lx[-1] + (lx[-1] - mid[-1])]]))


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=TS)


# ── row 1: both networks ────────────────────────────────────────────────────────────────────────────────────────────

def a_accuracy(ax, S, P):
    goal = np.arange(len(P["goal"])), P["goal"]
    for name, key in (("Bayes", "bayes"), ("memory-free Bayes", "window1"), ("naive", "naive")):
        v = P[f"right_{key}"][goal].mean()
        ax.axhline(v, color="0.6", lw=.8, ls=":")
        ax.text(XMIN * 1.15, v + .008, name, fontsize=FT, color="0.35")
    for r in RUNS:
        ax.plot(S[r]["x"], S[r]["acc"][:, -1], color=COL[r], lw=2)
    ax.set_ylim(.07, .84)
    ax.set_ylabel("goal accuracy at T = 30", fontsize=FL)
    title(ax, "a   Accuracy at T")


def b_copy(ax, S):
    for r in RUNS:
        ax.plot(S[r]["x"], S[r]["agree"][:, 2], color=COL[r], lw=2)
    ax.set_ylabel("fraction answering as naive", fontsize=FL)
    title(ax, "b   Copying naive's errors\n(naive wrong, Bayes right)")


def c_fr_acc(ax):
    for r in RUNS:
        z = sd.fr_acc(r)
        ax.plot(z["x"], z["r"], color=COL[r], lw=2)
    for name, key in (("Bayes", "r_bayes"), ("naive", "r_naive")):
        v = float(z[key])
        ax.axhline(v, color="0.6", lw=.8, ls=":")
        ax.text(XMIN * 1.15, v + .012, name, fontsize=FT, color="0.35")
    ax.axhline(0, color="0.8", lw=.7)
    ax.set_ylabel("Pearson r over probes", fontsize=FL)
    title(ax, "c   Factorization regret vs accuracy\n" r"r(FR$_T$, correct$_T$), each variable its own")


def d_ridge(ax, S):
    for r in RUNS:
        ax.plot(S[r]["x"], S[r]["dec"][:, 9, 0, 0], color=COL[r], lw=2)
    ax.set_ylabel(r"held-out R$^2$", fontsize=FL)
    title(ax, r"d   Ridge R$^2$: counts $n_t$" "\n" r"decoded from $u$ at step 30")


# ── rows 2-3: Trained RNN above ESN ─────────────────────────────────────────────────────────────────────────────────

def e_memory(axs, Dd):
    lo = min(min(Dd[r]["b"].min(), Dd[r]["stepwise"].min()) for r in RUNS)
    hi = max(max(Dd[r]["b"].max(), Dd[r]["stepwise"].max()) for r in RUNS)
    for ax, r in zip(axs, RUNS):
        ax.set_ylim(lo - .005, hi + .005)
        d = Dd[r]
        for k in range(5):
            ax.plot(d["x_ab"], d["b"][:, k], color=KCOL[k], lw=1.8)
            ax.plot(XR2, d["stepwise"][k], "D", ms=4, color=KCOL[k], mec="k", mew=.8)
        ax.axvline(XSEP, color="0.85", lw=.8)
        ax.axhline(0, color="0.6", lw=.8)
        ax.set_ylabel(r"cut $-$ intact update error (nats)", fontsize=FL)
    title(axs[0], "e   Memory use, by own entanglement\nupdate error cut to last step $-$ intact")


def f_beliefs(axs):
    for ax, r in zip(axs, RUNS):
        z = sd.belief_spectrum_steps(r)
        evr, pr = z["evr"].mean(1), z["pr"].mean(1)                                 # per step, then over the steps
        K = evr.shape[1]
        ax.pcolormesh(edges(z["x"]), np.arange(.5, K + 1), np.clip(evr, 1e-12, None).T, cmap="viridis",
                      norm=NORM["bel"], shading="flat", rasterized=True)
        ax.plot(z["x"], pr, color="w", lw=2)
        for name, key, ls in (("Bayes", "pr_bayes", "-"), ("naive", "pr_naive", (0, (3, 2)))):
            v = float(z[key].mean())
            ax.plot([XSEP, XMAX], [v, v], color="k", lw=1.6, ls=ls)
            ax.text(XMAX, v, name, fontsize=FT - .5, ha="right", va="bottom", color="k")
        ax.set_ylim(.5, K - 1.5)                    # the last 2 PCs are fixed at 0 (each variable's probabilities sum to 1)
        ax.set_yticks([1, 5, 10, 15, 18])
        ax.set_ylabel("PC index", fontsize=FL)
        ax.text(.04, .95, f"PR: {pr[0]:.1f} at the start, {pr[-1]:.1f} at the end", transform=ax.transAxes,
                ha="left", va="top", fontsize=FT, color="w")
    title(axs[0], "f   Belief spectrum and PR, per step\n" r"marginal beliefs (20 probabilities); white: PR")


def g_spectrum(axs):
    for ax, r in zip(axs, RUNS):
        z = sd.state_spectrum_steps(r)
        evr, pr = z["evr"].astype(np.float64).mean(1), z["pr"].mean(1)               # per step, then over the steps
        K = evr.shape[1]
        ax.pcolormesh(edges(z["x"]), np.arange(.5, K + 1), np.clip(evr, 1e-12, None).T, cmap="viridis",
                      norm=NORM["spec"], shading="flat", rasterized=True)
        ax.plot(z["x"], pr, color="w", lw=2)
        ax.set_yscale("log")
        ax.set_ylim(.8, K)
        ax.set_yticks([1, 10, 100, 1000])
        ax.set_yticklabels(["1", "10", "100", "1000"])
        ax.set_ylabel("PC index", fontsize=FL)
        ax.text(.04, .95, f"PR: {pr[0]:.1f} at the start, {pr[-1]:.0f} at the end", transform=ax.transAxes,
                ha="left", va="top", fontsize=FT, color="w")
    title(axs[0], "g   Network state spectrum and PR, per step\n" r"$\lambda_k/\Sigma\lambda$ of $u_t$;  white: "
          r"PR $=(\Sigma\lambda)^2/\Sigma\lambda^2$")


def h_folds(axs, G):
    for ax, r in zip(axs, RUNS):
        ax.plot(G[f"x_{r}"], G[f"where_naive_{r}"], color=COL[r], lw=2, marker="o", ms=3)
        ax.plot(G[f"x_{r}"], G[f"elsewhere_{r}"], color=COL[r], lw=2, ls=(0, (4, 2)), marker="o", ms=3)
        ax.axhline(G["bayes_where_naive"], color="k", lw=1, ls=":")
        ax.set_ylim(0, 85)
        ax.set_ylabel("% of the grid folded at T", fontsize=FL)
    axs[0].legend(handles=[Line2D([], [], color="0.35", lw=2, marker="o", ms=3, label="where naive's grid folds"),
                           Line2D([], [], color="0.35", lw=2, ls=(0, (4, 2)), marker="o", ms=3, label="elsewhere"),
                           Line2D([], [], color="k", lw=1, ls=":", label="Bayes, where naive folds")],
                  fontsize=FT, frameon=False, loc="upper right")
    title(axs[0], "h   Belief grid folds at T\nwhere naive's grid folds")


def i_fold_steps(axs, G, level=10):
    T = sd.T
    below = np.flatnonzero(G["steps_bayes"] < level)
    t_bayes = int(below[0]) + 1 if len(below) else None                           # Bayes's first step under 10%
    for ax, r in zip(axs, RUNS):
        x, f = G[f"x_{r}"], G[f"steps_{r}"]
        ax.pcolormesh(edges(x), np.arange(.5, T + 1), f.T, cmap="viridis", norm=NORM["fold"], shading="flat",
                      rasterized=True)
        if (f < level).any():
            ax.contour(x, np.arange(1, T + 1), f.T, levels=[level], colors="w", linewidths=1.6)
        if t_bayes:
            ax.axhline(t_bayes, color="w", lw=1.2, ls=(0, (3, 2)))
            ax.text(XMIN * 1.3, t_bayes + .4, f"Bayes: 10% at step {t_bayes}", color="w", fontsize=FT, va="bottom")
        ax.set_ylabel("observations t", fontsize=FL)
    title(axs[0], f"i   % of the grid folded\nwhite: 10% (naive: never, {G['steps_naive'][-1]:.0f}% at T)")


# ── row 4: the Trained RNN's update ─────────────────────────────────────────────────────────────────────────────────

def j_lags(ax, S):
    s = S["trained"]
    ax.pcolormesh(s["x"], ga.TAU, s["xc_upd"][:, 0].T, cmap="viridis", norm=NORM["lag"], shading="nearest",
                  rasterized=True)
    ax.set_yticks([-8, -4, 0, 4, 8])
    ax.set_ylabel(r"lag $\tau$ (steps after)", fontsize=FL)
    title(ax, "j   Coherence (both variables' errors):\nr with mean entanglement, every lag")


K_STAGES = (40, 200, 1000, 39996)        # k (John 11-02): training episodes, the nearest checkpoint on the log axis


def k_profiles(ax, S):
    """j's columns at training stages K_STAGES: r against tau, each curve labelled with its training episodes"""
    import td_tt_credit_lags as tcl
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap
    blues = LinearSegmentedColormap.from_list("blues_trunc", plt.cm.Blues(np.linspace(.35, 1., 256)))
    snorm = LogNorm(vmin=K_STAGES[0], vmax=K_STAGES[-1])
    st, se = S["trained"], S["echo"]
    r_t, r_e = st["xc_upd"][:, 0], se["xc_upd"][:, 0]
    x = np.asarray(st["x"], float)
    tau = np.asarray(ga.TAU)
    stroke = [pe.withStroke(linewidth=2.6, foreground="w")]
    for xv in K_STAGES:
        j = tcl.stage_index(x, xv)
        prof, c = r_t[j], blues(snorm(x[j]))
        lab = f"{x[j]:,.0f}" + (" (end)" if j == len(x) - 1 else "")
        ax.plot(tau, prof, color=c, lw=2, marker="o", ms=3, zorder=3)
        i = int(np.argmax(prof))
        if prof[i] < .03 and x[j] < 100:                           # flat: a short leader from above-left
            ax.annotate(lab, xy=(-6, prof[tau == -6][0]), xytext=(-7.6, .075), color=blues(snorm(4 * x[j])),
                        fontsize=FT, ha="left", va="bottom", path_effects=stroke, zorder=6,
                        arrowprops=dict(arrowstyle="-", color=c, lw=1, shrinkA=1, shrinkB=2))
        elif prof[i] < .03:                                        # a small bump after the step: label above it
            ax.text(tau[i], prof[i] + .012, lab, color=c, fontsize=FT, ha="center", va="bottom", path_effects=stroke,
                    zorder=6)
        elif tau[i] == 0:                                          # peak on the entangled step: label to its left
            ax.text(-.35, prof[i], lab, color=c, fontsize=FT, ha="right", va="center", path_effects=stroke, zorder=6)
        else:                                                      # peak after it: label just right of and above it
            ax.text(tau[i] + .3, prof[i] + .008, lab, color=c, fontsize=FT, ha="left", va="bottom",
                    path_effects=stroke, zorder=6)
    ax.plot(tau, r_e[-1], color=COL["echo"], lw=1.6, marker="s", ms=3, zorder=2)
    ax.text(-1.3, r_e[-1][tau == -1][0] + .015, "ESN, end", color=COL["echo"], fontsize=FT, ha="right", va="bottom")
    ax.axhline(0, color="k", lw=.7)
    ax.axvline(0, color="k", lw=.8, ls=(0, (2, 2)))
    ax.set_xticks([-8, -4, 0, 4, 8])
    ax.set_xlim(-8.5, 8.5)
    ax.set_xlabel(r"lag $\tau$ (steps after)", fontsize=FL)
    ax.set_ylabel(r"$r$ with entanglement", fontsize=FL)
    ax.spines[["top", "right"]].set_visible(False)
    title(ax, "k   j's columns at training stages\n(Trained RNN, training episodes; ESN at the end)")


CR = (("cr_coh", "l", "Credit coherence", r"$\Vert\Sigma_k\delta_k\Vert^2/\Sigma_k\Vert\delta_k\Vert^2$", "coherence"),
      ("cr_E", "m", "Update energy", r"$\Vert dW\Vert^2$", r"$\Vert dW\Vert^2$"),
      ("cr_dim", "n", "Update eff. dimension", r"$(\Sigma_i\sigma_i^2)^2/\Sigma_i\sigma_i^4$ of $dW$", "eff. dimension"),
      ("cr_bandx", "o", "Cross-band share", r"$1-\Sigma_B\Vert dW_B\Vert^2/\Vert dW\Vert^2$", "cross-band share"),
      ("cr_bandx_fixed", "p", "Cross-band share", "on the final checkpoint's bands", "cross-band share"))


def credit(ax, k, head, ylab, r="trained"):
    z = sd.credit(r)
    fr = z[f"{k}_hist"] / z[f"{k}_hist"].sum(1, keepdims=True)
    ax.pcolormesh(edges(z["x"]), sd.BINS[k], fr.T, cmap="viridis", norm=NORM["dist"], shading="flat", rasterized=True)
    q = z[f"{k}_q"]
    ax.plot(z["x"], q[:, 4], color="w", lw=1.6)
    for c in (2, 6):
        ax.plot(z["x"], q[:, c], color="w", lw=.9, ls="--")
    if k in ("cr_coh", "cr_E"):
        ax.set_yscale("log")
    ax.set_ylim(sd.BINS[k][0], sd.BINS[k][-1])
    ax.set_ylabel(ylab, fontsize=FL)
    title(ax, head)


# ── keys ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def keys(cells, cross):
    for c in cells.values():
        c.axis("off")
    h = [Line2D([], [], color=COL[r], lw=2.6, label=NAME[r]) for r in RUNS]
    h += [Line2D([], [], color="0.55", lw=1, ls=":", label=f"Trained RNN passes naive\n({cross:,.0f} training episodes)"),
          Line2D([], [], marker="D", color="0.6", mec="k", ls="none", ms=5.5, label="e: memory-free Bayes (right edge)"),
          Line2D([], [], color="0.55", lw=1.6, label="l-p: median (white),\n10% / 90% (dashed)")]
    cells["legend"].legend(handles=h, loc="upper left", fontsize=FL, frameon=False, borderaxespad=0)
    bars = (("fifths", [.0, .75, .95, .07], ScalarMappable(norm=BoundaryNorm(np.linspace(0, 100, 6), 5),
                                                             cmap=ListedColormap(KCOL)),
             "e: fifth of the variable's own entanglement"),
            ("row2", [.0, .8, .95, .07], ScalarMappable(norm=NORM["bel"], cmap="viridis"),
             "f: fraction of the belief variance per PC"),
            ("row2", [.0, .4, .95, .07], ScalarMappable(norm=NORM["spec"], cmap="viridis"),
             "g: fraction of the variance per PC"),
            ("row3", [.0, .85, .95, .07], ScalarMappable(norm=NORM["fold"], cmap="viridis"), "i: % of the grid folded"),
            ("row3", [.0, .5, .95, .07], ScalarMappable(norm=NORM["lag"], cmap="viridis"),
             "j: r with entanglement"),
            ("row3", [.0, .15, .95, .07], ScalarMappable(norm=NORM["dist"], cmap="viridis"),
             "l-p: fraction of a checkpoint's values"))
    for cell, rect, sm, lab in bars:
        cax = cells[cell].inset_axes(rect)
        cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=FT + .5, labelpad=3)
        cb.ax.tick_params(labelsize=FT - .5)


def figure():
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({"xtick.labelsize": 10.5, "ytick.labelsize": 10.5})
    P = ptd.probes()
    S = {r: ptd.series(r, P) for r in RUNS}
    I = fd.inputs()
    Dd = {r: fd.series(r, I) for r in RUNS}
    G = sd.grids()
    cross = fd.passes_naive(P)
    fig = plt.figure(figsize=(30, 18))
    gs = fig.add_gridspec(4, 7, left=.048, right=.987, top=.94, bottom=.085, wspace=.36, hspace=.58)
    ax = {(i, j): fig.add_subplot(gs[i, j]) for i in range(4) for j in range(7) if not (i < 3 and j >= 5)}
    for i in range(3):                                                                # key cells span columns 5-6
        ax[i, 5] = fig.add_subplot(gs[i, 5:7])
    a_accuracy(ax[0, 0], S, P)
    b_copy(ax[0, 1], S)
    c_fr_acc(ax[0, 2])
    d_ridge(ax[0, 3], S)
    e_memory([ax[1, 0], ax[2, 0]], Dd)
    f_beliefs([ax[1, 1], ax[2, 1]])
    g_spectrum([ax[1, 2], ax[2, 2]])
    h_folds([ax[1, 3], ax[2, 3]], G)
    i_fold_steps([ax[1, 4], ax[2, 4]], G)
    j_lags(ax[3, 0], S)
    k_profiles(ax[3, 1], S)
    for j, (k, letter, name, formula, ylab) in enumerate(CR):
        credit(ax[3, j + 2], k, f"{letter}   {name}\n{formula}", ylab)
    key_cells = {"legend": ax[0, 4], "fifths": ax[0, 5], "row2": ax[1, 5], "row3": ax[2, 5]}
    keys(key_cells, cross)
    for (i, j), a in ax.items():
        if a in key_cells.values() or (i, j) == (3, 1):                               # k: a lag axis
            continue
        xaxis(a, label=i == 3)
        a.axvline(cross, color="0.55", lw=1, ls=":", zorder=5)
    for i, r in ((1, "trained"), (2, "echo"), (3, "trained")):                     # row labels
        pos = gs[i, 0].get_position(fig)
        fig.text(.012, (pos.y0 + pos.y1) / 2, NAME[r], rotation=90, va="center", ha="center", fontsize=16, color=COL[r])
    fig.text(.5, .03, "a-g, j-p: 10,000 probe episodes (50 environments x 100 latent pairs x 2 trajectories) at each of "
             "114 checkpoints per network (j-p: the update measures on 5,000 of them, error steps 17-30; d: fitted on 25 "
             "environments, tested on the other 25).\nf, g: spectra and PR of the 10,000 probes at each step, then averaged "
             "over the 30 steps.   h, i: 50 environments x 100 cells x 24 fresh trajectories at 19 checkpoints.",
             ha="center", va="top", fontsize=FL, color="0.3", linespacing=1.7)
    path = fig_path("td_summary.png")
    fig.savefig(path, dpi=125)
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    figure()
