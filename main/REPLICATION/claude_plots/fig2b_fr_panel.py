"""The paper's Fig. 2b Factorization Regret panel with the exact FR (John 11-13: "can you please send an update of this
figure with the corrected FR (no clip). tell me if it would be easier to send the single panel on its own").

The printed panel (collection_plotters.plot_perf, panel 4: Collector.joint_naive_DKL) is the forward KL(Joint || Naive)
of the goal belief with both beliefs clipped to [1e-15, 1 - 1e-15], which caps a term at 34.5 nats and bends the curves
at long times (t = 100: 6.1 / 13.1 / 14.7 nats for C = 2 / 3 / 4). Here the same forward KL, exact (log space, float64):
fr_definition_check.build_fig2b's cache fig2b_clip.npz 'C<C>_exact' (the environment's own episode loop, T = 100;
C = 2 / 3 / 4: 40,000 / 20,000 / 10,000 episodes; its clipped curves reproduce the printed ones to .1 nats), C = 1 = 0
(one variable: Joint = Naive). Exact at t = 100: 9.31 / 21.81 / 26.57 nats (se .12 / .24 / .41); the share of episodes
whose exact FR passes 34 nats at t = 100 (where the clip binds): 8.5 / 20.1 / 24.7%.

Drawn as plot_perf draws panel 4 (its rcParams, colours, dashed lines, markers at t = 0, 33, 66, 99) inside the same
8 x 8 in 2 x 2 layout, so the panel has the printed panel's size and type; only the panel is saved (the other three
axes hold the printed titles and labels for the layout and are not drawn). y ticks 0 / 8 / 16 / 24 (printed: 0 / 4 / 8
/ 12). -> figure_bin/fig2b_fr_panel.pdf / .svg (pdf.fonttype 42).  usage:  python fig2b_fr_panel.py [--png PATH]"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION")]
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.transforms import Bbox                                            # noqa: E402
from main.utils import fig_path                                                   # noqa: E402

DATA = os.path.join(ROOT, "main", "DATA", "experimental", "fr_definition", "fig2b_clip.npz")
T = 100
RC = {"font.family": "serif", "font.serif": "cmr10", "font.sans-serif": "cmss10", "font.monospace": "cmtt10",
      "axes.formatter.use_mathtext": True, "mathtext.fontset": "cm", "font.size": 13, "axes.labelsize": 13,
      "axes.titlesize": 18, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "pdf.fonttype": 42,
      "svg.fonttype": "path"}                                                     # collection_plotters' rcParams


def figure(png=None):
    F = np.load(DATA)
    dkl = np.stack([np.zeros(T)] + [F[f"C{C}_exact"] for C in (2, 3, 4)])            # (4, T): C = 1 ... 4
    with plt.rc_context(RC):
        fig, axs = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
        t, idx = np.arange(T), np.linspace(0, T - 1, 4, dtype=int)
        col = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
        for ax_ in axs:
            for ax in ax_:
                ax.grid(alpha=.3)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
        for ax, title, xl, yl in ((axs[0, 0], "Joint", "Inference Time", "Accuracy"),
                                  (axs[0, 1], "Naive", "Inference Time", None),
                                  (axs[1, 0], "Relative Accuracy", "Naive", "Joint")):     # the layout only
            ax.set(title=title, xlabel=xl, ylabel=yl, ylim=(0, 1.1))
            ax.set_yticks(np.linspace(0, 1, 6))
        ax = axs[1, 1]
        for c in range(4):
            ax.plot(t, dkl[c], c=col[c], ls="--", lw=1.5)
            ax.scatter(t[idx], dkl[c, idx], s=50, marker="o", facecolors=col[c], edgecolors="k", zorder=3)
        ax.set(ylabel=r"$\mathcal{D}_{\mathrm{KL}}$", xlabel="Inference Time", title="Factorization Regret")
        ax.set_ylim(None, dkl.max() * 1.05)
        ax.set_yticks([0, 8, 16, 24])
        ax.set_xticks([0, 25, 50, 75, 100])
        fig.canvas.draw()
        fig.set_layout_engine("none")                                  # freeze the 2 x 2 positions before hiding
        for a in (axs[0, 0], axs[0, 1], axs[1, 0]):
            a.set_visible(False)
        bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        bb = Bbox.from_extents(bb.x0 - .05, bb.y0 - .05, bb.x1 + .05, bb.y1 + .05)
        for ext in ("pdf", "svg"):
            fig.savefig(fig_path(f"fig2b_fr_panel.{ext}"), bbox_inches=bb)
        if png:
            fig.savefig(png, bbox_inches=bb, dpi=200)
        plt.close(fig)
    print("saved", fig_path("fig2b_fr_panel.pdf"), "and .svg;", " / ".join(f"C = {c + 1}: {dkl[c, -1]:.2f}" for c in range(4)))


if __name__ == "__main__":
    figure(sys.argv[sys.argv.index("--png") + 1] if "--png" in sys.argv else None)
