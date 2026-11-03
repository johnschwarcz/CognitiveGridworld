"""Where on the grid each measure is high, step by step (27 Oct 2026; John on the measures movie: "the effective dim
moves from external to internal, and the delta W energy is so strong at the corners before relaxing")
-> main/REPLICATION/figure_bin/fig1c_measures_rings.png

Panels (3 x 3, as the movie): one measure each, the Trained RNN's, on the alias scan (39 environments x 5,000 streams,
each cell's mean over its 50 streams; the data line at the top says so, John 10-28). Heat map: rows, the cells grouped
by their distance from the grid's border, min(r_g, r_c, 9 - r_g, 9 - r_c): the four corners, the rest of the edge (32
cells), rings 1, 2, 3 (28, 20, 12 cells) and the centre (4 cells); columns, step t; colour, the cells' mean percentile
(each cell's mean measure ranked among its environment's 100 cells at that step; the movies showed these ranks until
10-28, now raw values), mean over the group's cells and the 39 environments: 0.5 = no preference for the group. Viridis:
bright = more. Data: plot_fig1c_measures_movie.data(). Two more rows (John 10-28): the settled and the unsettled
particles (posterior means within 0.15 cells of a lattice node / farther than 0.35, as plot_fig1c_measures_settled),
their mean percentile among their environment's 5,000 streams at that step, mean over the environments.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_measures_rings.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402

GROUPS = ("corners", "edge", "ring 1", "ring 2", "ring 3", "centre")


def groups(R=10):
    """each goal-ordered cell's group index (0: corners .. 5: centre)"""
    g, c = np.meshgrid(np.arange(R), np.arange(R), indexing="ij")
    ring = np.minimum(np.minimum(g, c), np.minimum(R - 1 - g, R - 1 - c)).ravel()
    corner = ((g == 0) | (g == R - 1)).ravel() & ((c == 0) | (c == R - 1)).ravel()
    return np.where(corner, 0, ring + 1)


def figure():
    import plot_fig1c_measures_movie as mm
    from main.utils import fig_path
    import plot_fig1c_measures_settled as ms
    _, cell, Xs, strm, _ = mm.data()
    gi = groups()
    d = np.linalg.norm(Xs - np.rint(Xs), axis=-1)
    kinds = (d < ms.SETTLED, d > ms.UNSETTLED)                                     # (E, S, T) each
    rows = GROUPS + ("settled", "unsettled")
    T = mm.T
    lo, hi = .1, .9
    PW, PH, GX, GY, X0, YT = 3.0, 1.85, .95, .95, 1.1, 1.0
    W, H = X0 + 3 * PW + 2 * GX + 1.3, YT + 3 * PH + 2 * GY + .55
    out = {}
    with plt.rc_context(mm.RC):
        fig = plt.figure(figsize=(W, H), dpi=200, facecolor="white")
        fig.text(.5, 1 - .2 / H, f"Trained RNN, {Xs.shape[0]} environments x {Xs.shape[1]:,} streams each (each "
                 f"cell's mean over its {mm.NS} streams)", ha="center", va="center", fontsize=11)
        for n, (key, name, formula) in enumerate(mm.PANELS):
            r_, c_ = divmod(n, 3)
            x, y = X0 + c_ * (PW + GX), H - YT - PH - r_ * (PH + GY)
            ax = fig.add_axes([x / W, y / H, PW / W, PH / H])
            A = cell[key].mean(0)                                                        # (K, T) over environments
            Mg = np.stack([A[gi == k].mean(0) for k in range(len(GROUPS))]
                          + [((strm[key] * m).sum(1) / np.maximum(m.sum(1), 1)).mean(0) for m in kinds])  # (8, T)
            out[key] = Mg
            ax.imshow(Mg, aspect="auto", cmap="viridis", vmin=lo, vmax=hi, origin="upper", interpolation="nearest",
                      extent=(.5, T + .5, len(rows) - .5, -.5))
            ax.axhline(len(GROUPS) - .5, color="white", lw=2.5)                   # rings above, particles below
            ax.set_yticks(range(len(rows)), rows if c_ == 0 else [])
            ax.set_xticks([1, 10, 20, 30])
            ax.tick_params(labelsize=9, length=2)
            if r_ == 2:
                ax.set_xlabel("step $t$", fontsize=10)
            ax.set_title(name + "\n" + formula, fontsize=10, pad=4)
        cax = fig.add_axes([(X0 + 3 * PW + 2 * GX + .3) / W, (H - YT - 2 * PH - GY) / H, .13 / W, (2 * PH + GY) / H])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(lo, hi), cmap="viridis"), cax=cax)
        cb.set_ticks([lo, .5, hi]), cb.ax.tick_params(labelsize=9)
        cb.set_label("mean percentile within the environment and step\n(rings: cells among 100; settled / unsettled: "
                     "streams among 5,000)", fontsize=9.5)
        cb.outline.set_visible(False)
        path = fig_path("fig1c_measures_rings.png")
        fig.savefig(path)
        plt.close(fig)
    for key, Mg in out.items():
        print(f"{key:7s} " + "  ".join(f"{GROUPS[k]}: t1 {Mg[k, 0]:.2f} t5 {Mg[k, 4]:.2f} t15 {Mg[k, 14]:.2f} "
                                       f"t30 {Mg[k, 29]:.2f}" for k in (0, 1, 5)))
    return path


if __name__ == "__main__":
    print("saved", figure())
