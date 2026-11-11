"""fig_learn_grids.png: the belief grids through training, for the paper (John 11-10: "let's move the grids into their
own figure, and start polishing it for the paper (no notes, consistency with existing paper figures, etc)"). Until 11-10
this was panel o of supp_learning_behaviour (fig_supp_learning.py).

Drawn at print size (the paper's text width 5.5 in; ticks and keys 7 pt, labels and titles 8 pt, row titles 9 pt bold, as
plot_matched_params_joint's submission figure; Computer Modern). Each grid: the 100 cells (r_0, r_1) of the two
variables at the networks' posterior means at T = 30, averaged over the 50 test environments (as paper Figure 16e),
neighbouring cells joined by thin grey lines, the true lattice dashed behind; each cell coloured by its Factorization
Regret (FR_T = KL(Joint || Naive) of the marginal beliefs at T, mean of the two variables, mean over the cell's episodes
and the environments; the paper's grey-to-red map, plot_fig1c_mesh.FR_CMAP, on a log scale fitted to the cells, .4-11
nats). Columns: training episodes (MESH_X; no checkpoint at 600 or 1,000: 560 and 1,040); last column: the observer
each network approaches (Joint Bayes for the Trained RNN, Naive Bayes for the ESN). Number under each grid: the distinct
answers per environment were drawn under each grid until John 11-10: "for the learn_grids let's remove the numbers
under the grids, match other styling choices to attached figure (keep the grid plotting how it is, but like for each
the color of "Trained" and "Echo"" (the attached: paper Figure 16's manifolds, rows Joint / Trained / Naive / Echo in
their colours, colour bar 'factorization regret' with arrows): now only in RESULTS.

11-11: merged into supp_learning_behaviour as extra rows, then moved back here ("lets move the grid rows back into their
own figure again"); points enlarged to match.
Caption draft (11-11, for the layout with Joint / Naive Bayes at the right of the rows):
  "The belief grid unfolds through training, last where Factorization Regret is high. Each point is one joint
  realization r = (r_1, r_2) of the two latent variables, placed at the network's posterior means of r_1 and r_2 at
  T = 30 (mean over 24 trajectories per realization, then over 50 pairs of testing variables) and colored by its
  Factorization Regret (log scale); grey lines join neighbouring realizations and the dashed lattice marks their true
  values. Columns: training episodes. The Fully Trained RNN (top) unfolds the grid toward Joint Bayes (top right), and
  the high-FR realizations along its border are the last to separate. The ESN's grid (bottom) instead stays folded
  where Factorization Regret is high, like Naive Bayes's (bottom right)."
RESULTS (distinct answers): Trained RNN 11.4 / 21.6 / 22.6 / 27.9 / 36.1 / 48.0 / 57.5 / 76.8 / 85.5 (Joint Bayes 94.4);
ESN 3.0 / 7.8 / 9.9 / 13.5 / 13.6 / 16.4 / 20.4 / 27.1 / 36.5 (Naive Bayes 44.3).
Data: fig_supp_learning.mesh_grids (fig_learn_fr's grid cache + td_grids_fwd's extra checkpoints): 50 test environments
x 100 cells x 24 trajectories per checkpoint. usage:  python fig_learn_grids.py"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402
from matplotlib.colors import LogNorm                                             # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402

FS = dict(tick=7, key=7, label=8, title=8, row=9)                                 # points at print size
W, DPI = 5.5, 400
R = 10
ROWS = (("trained", "Trained", "#1f77b4", "bayes", "Joint", "k"), ("echo", "Echo", "#ff7f0e", "naive", "Naive", "0.5"))


def data():
    """grids {run / observer: positions (50, 100, 2), modal answers (50, 100)} and the cells' FR_T (100,)"""
    import fig_supp_learning as fsl
    import fig_learn_fr as flf
    return fsl.mesh_grids(), fsl.MESH_X, flf.load("grids")["pair_env_fr"].mean(0)


def answers(modal):
    return np.mean([len(np.unique(m)) for m in modal])


def figure(path=None):
    import plot_fig1c_mesh as pm
    MG, stages, fr = data()
    apply_fig_style()
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.linewidth": .5})
    # John 11-10: "let's maybe put the cbar horizontally on the bottom, the figure is a bit too wide ... Maybe network
    # name can go above, training episodes can go below. I like where the Joint and Naive are though"; then (11-10)
    # "remove episodes 200 and 400, and put the Naive and Bayes grids below with the cbar"; then (11-11) "put the Joint
    # and Naive grid's back on the right as they were before": names above the rows, the reference observer at the
    # right of each row (Joint Bayes beside the Trained RNN, Naive Bayes beside the ESN), training episodes below; then
    # "with the cbar on the right as well": the colour bar vertical, right of the observers
    stages = tuple(x for x in stages if x not in (200, 400))
    lm, rm, gp, gref = .04, .62, .05, .16                                         # margins and gaps, inches (rm: the bar)
    pw = (W - lm - rm - (len(stages) - 1) * gp - gref) / (len(stages) + 1)
    name_h, rgap, nums, xl_h = .20, .06, .15, .17
    tops = [name_h, name_h + pw + rgap + name_h]
    y_num = tops[1] + pw + .03
    H = y_num + nums + xl_h + .06
    fig = plt.figure(figsize=(W, H))
    norm = LogNorm(np.floor(fr.min() * 10) / 10, np.ceil(fr.max()))
    cmap = pm.FR_CMAP
    idx = {(g, c): g * R + c for g in range(R) for c in range(R)}
    edges = np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                      if (g + dg, c + dc) in idx])
    cells = np.array([(g, c) for g in range(R) for c in range(R)], float)

    def grid(x0, top, pos, title=None, tcol="k"):
        ax = fig.add_axes([x0 / W, 1 - (top + pw) / H, pw / W, pw / H])
        ax.add_collection(LineCollection(cells[edges], colors="0.86", linewidths=.25, linestyles=(0, (3, 2))))
        P = pos.mean(0)                                                           # mean over the 50 environments
        ax.add_collection(LineCollection(P[edges], colors="0.7", linewidths=.3, zorder=2))
        ax.scatter(P[:, 0], P[:, 1], c=fr, cmap=cmap, norm=norm, s=4.5, lw=.2, edgecolors="0.3", zorder=3)
        ax.set_xlim(-.7, 9.7), ax.set_ylim(-.7, 9.7), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("0.7")
            sp.set_linewidth(.4)
        if title:
            ax.text(.5, 1.04, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=FS["title"],
                    color=tcol)
        return ax

    xs = [lm + j * (pw + gp) for j in range(len(stages))]
    x_ref = xs[-1] + pw + gref
    xc = lm + (len(stages) * (pw + gp) - gp) / 2                                  # the centre of the training columns
    for i, (run, row, rcol, ref, ref_name, refcol) in enumerate(ROWS):
        for j, xv in enumerate(stages):
            grid(xs[j], tops[i], MG[run][xv][0])
        grid(x_ref, tops[i], MG[ref][0], ref_name, refcol)                       # the observer, at the row's right
        fig.text(xc / W, 1 - (tops[i] - .045) / H, row, ha="center", va="baseline", fontsize=FS["row"], color=rcol)
    for j, xv in enumerate(stages):                                               # training episodes, below
        fig.text((xs[j] + pw / 2) / W, 1 - y_num / H, f"{xv:,}", ha="center", va="top", fontsize=FS["tick"])
    fig.text(xc / W, 1 - (y_num + nums) / H, "Training episodes", ha="center", va="top", fontsize=FS["label"])
    cx = fig.add_axes([(x_ref + pw + .12) / W, 1 - (tops[1] + pw) / H, .06 / W, (tops[1] + pw - tops[0]) / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cx, extend="both", extendfrac=.03)
    cb.ax.tick_params(labelsize=FS["tick"], length=2, width=.5, pad=1.5)
    cb.ax.tick_params(which="minor", length=1.2, width=.4)
    cb.outline.set_visible(False)
    cb.set_label("factorization regret", fontsize=FS["label"], labelpad=3)
    fig.canvas.draw()                                                             # every text inside the figure
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    lo, hi = cb.ax.get_ylim()
    texts = fig.texts + [t for ax in fig.axes for t in ax.texts] + [cb.ax.yaxis.label] + \
        [t for t, v in zip(cb.ax.get_yticklabels(), cb.ax.get_yticks()) if lo <= v <= hi]       # drawn ticks only
    for t in texts:
        e = t.get_window_extent(r)
        if t.get_text() and (e.x0 < fb.x0 - .5 or e.x1 > fb.x1 + .5 or e.y0 < fb.y0 - .5 or e.y1 > fb.y1 + .5):
            print(f"  WARNING: text {t.get_text()!r} leaves the figure")
    path = path or fig_path("fig_learn_grids.png")
    fig.savefig(path, dpi=DPI)
    fig.savefig(path[:-4] + ".svg")                                               # John 11-11: SVGs of this figure
    with plt.rc_context({"pdf.fonttype": 42}):                                    # and a PDF (TrueType fonts embedded)
        fig.savefig(path[:-4] + ".pdf")
    plt.close(fig)
    print("saved", path)
    for run, row, _, ref, ref_name, _ in ROWS:
        print(f"{row}: " + " / ".join(f"{answers(MG[run][xv][1]):.1f}" for xv in stages)
              + f"  ({ref_name} {answers(MG[ref][1]):.1f})")


if __name__ == "__main__":
    figure()
