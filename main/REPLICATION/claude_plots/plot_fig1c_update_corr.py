"""Joint movement of the two variables' beliefs (posterior means), averaged over steps, by factorization regret and
entanglement at the step, by the belief's path length and by the size of the single step (25 Oct 2026)
-> main/REPLICATION/figure_bin/fig1c_update_corr.png

Data: figdata/round4b/updates_50k.npz (fig1c_update_cache.py): the trained network's gradflow harvest, all 50,000
episodes, 48,968 kept trajectories from 1,225 environments, all four agents on the same episodes, steps t = 1..30.
Update sizes of the two variables at step t: |u_v(t)| = |E_t[r_v] - E_{t-1}[r_v]| (posterior means, E_0 = 4.5; grid
units). Joint movement of a group of (trajectory, step) points: the covariance of the two sizes, large only when both
beliefs move a lot in the same steps (the correlation times both sizes' spreads; grid units^2).
Columns (20 quantile bins each):
    FR_t: factorization regret at the step (the cache's fr_t: FR_g + FR_c, each KL(the Optimal's marginal || the
          Naive's), the paper's forward KL, exact, main.utils.factorization_regret; the symmetric Jeffreys/2 until
          the cache was patched on 2026-11-10), and E_t: the step's entanglement - bins cut anew at every step, the
          covariance per step and bin, then its mean over the 30 steps;
    S:    the belief's path length over all steps (sum_t ||u(t)||; each agent binned by its own S) - per step and bin,
          then the mean over steps;
    ||u(t)||: the single step's size, bins of all (trajectory, step) points pooled - the covariance over each bin's
          points. Within a bin of step size the two sizes trade off (|u_g|^2 + |u_c|^2 is about fixed), so the
          covariance there is at most about 0: near 0 when steps of that size move both variables alike, negative
          when they move one variable or the other.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/plot_fig1c_update_corr.py"""
import os
import sys
import numpy as np


def _root():
    for start in (globals().get("__file__"), globals().get("__vsc_ipynb_file__"), os.getcwd()):
        d = os.path.abspath(start) if start and os.path.exists(start) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return d
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError("project root not found")


ROOT = _root()
sys.path[:0] = [p for p in (ROOT,) if p not in sys.path]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from main.utils import apply_fig_style, fig_path                                  # noqa: E402

FS = apply_fig_style()
plt.rcParams["mathtext.fontset"] = "cm"
DATA = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression", "figdata", "round4b", "updates_50k.npz")
AG = ("Joint", "Trained", "Naive", "Echo")
NAME = {"Joint": "Optimal", "Trained": "Trained RNN", "Naive": "Naive", "Echo": "ESN"}
COLOR = {"Joint": "k", "Trained": "#1f77b4", "Naive": "0.5", "Echo": "#ff7f0e"}
STY = {"Trained": dict(lw=2.6, alpha=.45), "Joint": dict(lw=1.3), "Naive": dict(lw=1.3), "Echo": dict(lw=1.3)}
NB = 20
T = 30


def _bins(x):
    e = np.quantile(x, np.linspace(0, 1, NB + 1))
    return np.clip(np.searchsorted(e, x, side="right") - 1, 0, NB - 1)


def _cov(a, b):
    return ((a - a.mean(0)) * (b - b.mean(0))).sum(0) / (len(a) - 1)


def joint_movement(Z):
    """per grouping and agent: the joint movement per bin (NB,), and the bins' medians for S and ||u||"""
    size = {a: np.abs(Z["u_" + a].astype(np.float64)) for a in AG}
    C, med = {}, {}
    for key, X in (("fr", Z["fr_t"].astype(np.float64)), ("ent", Z["rev"].astype(np.float64))):
        q = np.stack([_bins(X[:, t]) for t in range(T)], 1)                       # (n, T), cut anew each step
        for a in AG:
            C[key, a] = np.array([np.mean([_cov(size[a][q[:, t] == k, t, 0], size[a][q[:, t] == k, t, 1])
                                           for t in range(T)]) for k in range(NB)])
    for a in AG:
        L = np.hypot(size[a][..., 0], size[a][..., 1])                            # (n, T) step sizes
        S = L.sum(1)
        q = _bins(S)
        C["S", a] = np.array([_cov(size[a][q == k, :, 0], size[a][q == k, :, 1]).mean() for k in range(NB)])
        med["S", a] = np.array([np.median(S[q == k]) for k in range(NB)])
        Lf, uf = L.ravel(), size[a].reshape(-1, 2)
        q = _bins(Lf)
        C["L", a] = np.array([_cov(uf[q == k, 0], uf[q == k, 1]) for k in range(NB)])
        med["L", a] = np.array([np.median(Lf[q == k]) for k in range(NB)])
    return C, med


def draw(fig, x0, y0, pw, ph, gx, C, med, letters="abc", legend_xy=None, vertical=False, formulas=True,
         legend_ncol=4):
    """the three joint-movement panels in fig, lower-left corner (x0, y0) inches, panels pw x ph inches gx apart (in a
    row; vertical: stacked in a column, gx apart); formulas: each panel's formula line under its name (False: the name
    only); the agents' legend (upper centre, legend_ncol columns) at legend_xy inches if given"""
    W, H = fig.get_size_inches()
    cols = (("fr", "Factorization regret at the step", "$FR_t$, quantile bin at each step",
             r"by step: $\mathrm{Cov}(|u_g(t)|, |u_c(t)|)$, mean over $t$"),
            ("ent", "Entanglement of the step", r"$\mathcal{E}_t$, quantile bin at each step",
             r"by step: $\mathrm{Cov}(|u_g(t)|, |u_c(t)|)$, mean over $t$"),
            ("S", "Size of the belief's movement", "path length $S$ (bin median, grid units)",
             r"by step: $\mathrm{Cov}(|u_g(t)|, |u_c(t)|)$, mean over $t$"))
    xs = np.arange(1, NB + 1)
    for c_, (key, cname, xlab, formula) in enumerate(cols):
        if vertical:
            ax = fig.add_axes([x0 / W, (y0 + (2 - c_) * (ph + gx)) / H, pw / W, ph / H])
        else:
            ax = fig.add_axes([(x0 + c_ * (pw + gx)) / W, y0 / H, pw / W, ph / H])
        for a in AG:
            xv = med[key, a] if key in ("S", "L") else xs
            ax.plot(xv, C[key, a], c=COLOR[a], marker="o", ms=2.2, **STY[a])
        ax.axhline(0, c="0.8", lw=.6, zorder=0)
        if key in ("S", "L"):
            ax.set_xscale("log")
            lo_, hi_ = min(med[key, a][0] for a in AG), max(med[key, a][-1] for a in AG)
            cand = [m * 10. ** k for k in range(-4, 3) for m in (1, 2, 5)]
            ax.set_xticks([v for v in cand if lo_ * .95 <= v <= hi_ * 1.05])
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%g" % v))
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
        else:
            ax.set_xlim(.5, NB + .5), ax.set_xticks([1, 10, 20])
        ax.set_xlabel(xlab, labelpad=1)
        if c_ == (1 if vertical else 0):
            ax.set_ylabel("joint movement\n" r"(grid units$^2$)" if vertical else r"joint movement (grid units$^2$)")
        ax.spines[["top", "right"]].set_visible(False)
        name_y = 17 if formulas else 6
        ax.annotate(cname, (.5, 1), xycoords="axes fraction", xytext=(0, name_y), textcoords="offset points",
                    ha="center", va="baseline", fontsize=FS["label"] + .5)
        if formulas:
            ax.annotate(formula, (.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                        ha="center", va="baseline", fontsize=FS["tick"] - .5, color="0.25")
        x_, yy, w, h = ax.get_position().bounds
        fig.text(x_ - .62 / W, yy + h + name_y / 72 / H, r"$\mathbf{%s}$" % letters[c_], fontsize=FS["label"] + 2,
                 va="baseline")
    if legend_xy is not None:
        hand = [Line2D([], [], c=COLOR[a], label=NAME[a], **STY[a]) for a in AG]
        fig.legend(handles=hand, loc="upper center", bbox_to_anchor=(legend_xy[0] / W, legend_xy[1] / H),
                   ncol=legend_ncol, frameon=False, fontsize=FS["tick"], handlelength=1.8, columnspacing=1.4)


def figure():
    Z = np.load(DATA)
    C, med = joint_movement(Z)
    pw, ph, gx = 2.9, 2.1, .85
    W, H = .85 + 3 * pw + 2 * gx + .3, 1.0 + ph + .95
    fig = plt.figure(figsize=(W, H))
    draw(fig, .85, 1.0, pw, ph, gx, C, med, legend_xy=(.5 * W, H - .04))
    fig.text(.85 / W, .5 / H, r"Joint movement: the covariance of the two variables' update sizes $|u_v(t)| = |E_t[r_v] - "
             r"E_{t-1}[r_v]|$ (posterior means), large only when both beliefs move a lot in the same steps." "\n" r"a, b: "
             r"bins cut anew at each step (1 = the lowest 5%%). %s trajectories from %s "
             r"environments (the whole harvest), pooled." % (f"{len(Z['env']):,}", f"{Z['env'].max() + 1:,}"),
             fontsize=FS["tick"] - 1, color="0.35", va="top", linespacing=1.5)
    path = fig_path("fig1c_update_corr.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path, C, med


if __name__ == "__main__":
    path, C, med = figure()
    print("saved", path)
    for key in ("fr", "ent", "S", "L"):
        print("%-3s " % key + " | ".join("%s %.4f -> %.4f -> %.4f" % (NAME[a], C[key, a][0], C[key, a][9], C[key, a][-1])
                                         for a in AG))
