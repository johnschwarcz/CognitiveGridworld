"""Where the joint and the factorized observers' update sizes differ: density of (|dx|, |dy|), Optimal - Naive and
Trained RNN - ESN (25 Oct 2026) -> main/REPLICATION/figure_bin/fig1c_update_sizes.png

Data: the trained network's gradflow harvest, all 50,000 episodes (48,968 kept trajectories from 1,225 environments;
all four agents on the same episodes), every step t = 1..30:
    (|dx|, |dy|) = (|u_g(t)|, |u_c(t)|), u_v(t) = E_t[r_v] - E_{t-1}[r_v] (posterior means, E_0 = 4.5).
Each agent's density: the share of its (trajectory, step) points in each 0.25 x 0.25 bin; shown: the difference of two
agents' densities (red: the first agent's steps land there more often, blue: less; black: equally, or no step of
either). Colour: symmetric log scale, in absolute shares per bin. Data: figdata/round4b/updates_50k.npz (all 50,000
harvest episodes; fig1c_update_cache.py).
On these axes a point's distance from the origin is the step's length and its angle how the step divides between the
two variables (dashed diagonal: equally).
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/plot_fig1c_update_sizes.py"""
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
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm                 # noqa: E402
from main.utils import apply_fig_style, fig_path                                  # noqa: E402

FS = apply_fig_style()
plt.rcParams["mathtext.fontset"] = "cm"
DATA = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression", "figdata", "round4b", "updates_50k.npz")
NAME = {"Joint": "Optimal", "Trained": "Trained RNN", "Naive": "Naive", "Echo": "ESN"}
PAIRS = (("Joint", "Naive"), ("Trained", "Echo"))
DIFF_CMAP = LinearSegmentedColormap.from_list(
    "bkr", [(0, "#a8d4ff"), (.22, "#2a6dff"), (.5, "#000000"), (.78, "#ff2d1a"), (1, "#ffbfa8")])
DIFF_CMAP.set_bad("black")
LIM, BIN, LINTHRESH, TOP_ENT = 5., .25, 5e-5, .75
MIN_STEP = .25                                                                    # steps shorter than this are left out


def densities(Z, sel=None):
    """each agent's share of points per bin (n_bins, n_bins) and the counts, over the steps where sel (n, T) is True"""
    edges = np.arange(0, LIM + BIN / 2, BIN)
    out = {}
    for a in NAME:
        u = np.abs(Z["u_" + a].astype(np.float64))
        u = (u[sel] if sel is not None else u).reshape(-1, 2)
        keep = np.hypot(u[:, 0], u[:, 1]) >= MIN_STEP
        n, _, _ = np.histogram2d(u[keep, 0], u[keep, 1], [edges, edges])
        out[a] = (n / keep.sum(), n, u, 1 - keep.mean())
    return out, edges


def draw(fig, rects, cax_rect, Z=None, ylabels=(True, False), compact=False, notes="full"):
    """the two difference maps (Optimal - Naive, Trained RNN - ESN) in fig: rects, the two panels' (x, y, w, h) in
    inches; cax_rect the shared colour bar's; ylabels which panels carry the y label and tick labels; compact: the
    annotations broken into shorter lines (for small panels); notes "full" (the standalone figure: the share of long
    steps near the diagonal, the red and blue masses, the left-out share) or "short" (only the long steps' share, at a
    readable size; one-line colour bar label)"""
    W, H = fig.get_size_inches()
    Z = np.load(DATA) if Z is None else Z
    rows = ((None, None),)
    dens = [densities(Z, sel) for sel, _ in rows]
    vmax = max(np.abs(d[a1][0] - d[a2][0]).max() for d, _ in dens for a1, a2 in PAIRS)
    norm = SymLogNorm(LINTHRESH, vmin=-vmax, vmax=vmax, base=10)
    d, edges = dens[0]
    for c, (a1, a2) in enumerate(PAIRS):
        x, y, w, h = rects[c]
        ax = fig.add_axes([x / W, y / H, w / W, h / H])
        D = d[a1][0] - d[a2][0]
        D = np.where((d[a1][1] + d[a2][1]) > 0, D, np.nan)                         # white: no step of either
        im = ax.imshow(D.T, origin="lower", extent=[0, edges[-1], 0, edges[-1]], cmap=DIFF_CMAP, norm=norm,
                       interpolation="nearest")
        ax.set_facecolor("black")
        ax.plot([0, LIM], [0, LIM], c="0.65", lw=.7, ls=(0, (3, 2)))
        ax.set_xlim(0, LIM), ax.set_ylim(0, LIM), ax.set_aspect("equal")
        ax.set_xticks(range(0, int(LIM) + 1)), ax.set_yticks(range(0, int(LIM) + 1))
        ax.set_xlabel(r"$|\Delta x| = |E_t[r_g] - E_{t-1}[r_g]|$", labelpad=1)
        if ylabels[c]:
            ax.set_ylabel(r"$|\Delta y| = |E_t[r_c] - E_{t-1}[r_c]|$", labelpad=1)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{NAME[a1]} $-$ {NAME[a2]}", fontsize=FS["label"] + .5,
                     pad=(30 if notes == "short" else 23 if compact else 15))
        txt = []
        for a in (a1, a2):                                                         # long steps near the diagonal
            u = d[a][2]
            big = np.hypot(u[:, 0], u[:, 1]) > 2
            diag = big & (np.minimum(u[:, 0], u[:, 1]) > .5 * np.maximum(u[:, 0], u[:, 1]))
            txt.append("%s %.0f%%" % (NAME[a], 100 * diag.sum() / max(big.sum(), 1)))
        if notes == "short":
            ax.text(.5, 1.015, "steps longer than 2 near the diagonal:\n" + ", ".join(txt), transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=FS["tick"], color="0.25", linespacing=1.25)
            continue
        ax.text(.5, 1.012, "steps longer than 2 that lie near the diagonal:" + ("\n" if compact else " ") + ", ".join(txt),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=FS["tick"] - 1.5, color="0.3",
                linespacing=1.2)
        Dz = np.nan_to_num(D)
        ax.text(.98, .98, ("red mass %.3f = blue mass %.3f\nleft out (steps under %.2f):%s%s %.0f%%, %s %.0f%%") % (
            Dz[Dz > 0].sum(), -Dz[Dz < 0].sum(), MIN_STEP, "\n" if compact else " ", NAME[a1], 100 * d[a1][3],
            NAME[a2], 100 * d[a2][3]), transform=ax.transAxes, ha="right", va="top", fontsize=FS["tick"] - 1.5,
            color="0.85", linespacing=1.2)
    x, y, w, h = cax_rect
    cx = fig.add_axes([x / W, y / H, w / W, h / H])
    ticks = [t for t in (-1e-1, -1e-2, -1e-3, -1e-4, 0, 1e-4, 1e-3, 1e-2, 1e-1) if abs(t) <= vmax]
    cb = fig.colorbar(im, cax=cx, ticks=ticks)
    cb.ax.set_yticklabels([("%g" % t) for t in ticks])
    if notes == "short":
        cb.set_label("share of steps:\njoint $-$ factorized agent", labelpad=4)
        cb.ax.text(0, 1.03, "joint agent\nmore often", transform=cb.ax.transAxes, ha="left", va="bottom",
                   fontsize=FS["tick"], color="#c0392b", linespacing=1.1)
        cb.ax.text(0, -.03, "factorized agent\nmore often", transform=cb.ax.transAxes, ha="left", va="top",
                   fontsize=FS["tick"], color="#2a6dff", linespacing=1.1)
    else:
        cb.set_label("difference in the share of steps per %g x %g bin\n(red: the first agent more often; symmetric "
                     "log scale)" % (BIN, BIN), labelpad=4)
    cb.outline.set_visible(False)
    return cb


def figure():
    Z = np.load(DATA)
    pw, gap = 3.0, .55
    W = .85 + 2 * pw + gap + .3 + .13 + 1.0
    H = .75 + pw + 1.15
    fig = plt.figure(figsize=(W, H))
    y0 = H - .75 - pw
    draw(fig, [(.85 + c * (pw + gap), y0, pw, pw) for c in range(2)], (.85 + 2 * pw + gap + .3, y0, .13, pw), Z)
    fig.text(.85 / W, .42 / H, "Update sizes of the posterior means at every step t = 1-30, %s trajectories from "
             "%s environments (the whole harvest), pooled; steps shorter than %.2f left out;\neach agent's share of its "
             "remaining steps per bin, then the difference; the dashed diagonal: both variables move by the same amount." % (
                 f"{len(Z['env']):,}", f"{Z['env'].max() + 1:,}", MIN_STEP),
             fontsize=FS["tick"] - 1.5, color="0.35", va="top", linespacing=1.4)
    path = fig_path("fig1c_update_sizes.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure())
