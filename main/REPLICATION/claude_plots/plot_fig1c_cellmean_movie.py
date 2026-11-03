"""The representation as a sheet: each cell's mean belief over its 50 repeated streams in one environment, joined to its
grid neighbours, coloured by how much its streams move both variables together (25 Oct 2026)
-> main/REPLICATION/figure_bin/fig1c_movie_cellmeans.gif

Data: the alias scan (main/DATA/experimental/fig1c_compression), environment ENV (34, the env movies' environment):
100 cells x 50 streams (repetitions of the same environment and cell), all four agents on the same streams.
Particle of a cell: the mean of its 50 streams' beliefs (20 numbers), drawn in panel (c)'s T = 30 plane
(fig1.belief_geometry; as fig1c_movie_fixed), moving smoothly between steps (fig1_movie._blend).
Colour: C_t = Cov over the cell's 50 streams of the two update sizes |u_g(t)|, |u_c(t)| (u_v(t) = E_t[r_v] -
E_{t-1}[r_v], posterior means, E_0 = 4.5; grid units^2): red, the cell's streams move both variables a lot in the same
steps; blue, one or the other; grey, neither (or unrelated). Lines: the 10 x 10 grid of cells (each cell joined to
its neighbours in r_g and in r_c): an unfolded sheet keeps the cells' order, a folded one lays cells on top of each
other. Dashed, faint: the same grid of the cells' targets (fig1.targets: the Optimal's mean correct T = 30
belief), where a correct representation ends.
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_cellmean_movie.py"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize                  # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402
from PIL import Image                                                             # noqa: E402

SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
ENV, T, M = 34, 30, 50
AGENTS = (("Trained", "Echo"), ("Joint", "Naive"))                                # the fixed movie's 2 x 2 order
NAME = {"Joint": "Optimal", "Trained": "Trained RNN", "Naive": "Naive", "Echo": "ESN"}
COV_CMAP = LinearSegmentedColormap.from_list("cov", ["#2166ac", "#9ecae1", "#d0d0d0", "#fc9272", "#b2182b"])
VMAX = .6
DPI = 110


def data():
    import fig1 as cf
    import fig1_movie as cm
    G = cf.belief_geometry()
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    assert (meta["cell_of"] == np.repeat(np.arange(100), M)).all()               # streams stored cell by cell
    R = np.arange(10)
    cells = meta["cells"]
    home, _ = cf.targets([tuple(int(v) for v in c) for c in cells])                # (100, 20)
    out = {}
    for row in AGENTS:
        for a in row:
            B = np.load(os.path.join(SCAN, f"scan_{a}.npz"))["beliefs"][ENV].astype(np.float64)   # (5000, T, 20)
            b = B.reshape(5000, T, 2, 10) / B.reshape(5000, T, 2, 10).sum(-1, keepdims=True)
            E = np.concatenate([np.full((5000, 1, 2), 4.5), (b * R).sum(-1)], 1)
            u = np.abs(np.diff(E, axis=1)).reshape(100, M, T, 2)
            uc = u - u.mean(1, keepdims=True)
            C = (uc[..., 0] * uc[..., 1]).sum(1) / (M - 1)                        # (100, T)
            Mn = B.reshape(100, M, T, 20).mean(1)                                 # cell means (100, T, 20)
            out[a] = dict(C=C, M=Mn, home=home, proj=lambda V, D=G[(a, T)]: cm._proj(D, V), blend=cm._blend)
    return out, cells, cm.schedule()


def grid_edges(cells):
    """index pairs of cells that are grid neighbours (r_g + 1 or r_c + 1)"""
    idx = {(int(g), int(c)): k for k, (g, c) in enumerate(cells)}
    return np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                     if (g + dg, c + dc) in idx])


def movie():
    D, cells, sched = data()
    E = grid_edges(cells)
    pw, gap = 3.0, .45
    W, H = .35 + 2 * pw + gap + .3 + .14 + .75, .65 + 2 * (pw + .35) + .35
    fig = plt.figure(figsize=(W, H), facecolor="white", dpi=DPI)
    norm = Normalize(-VMAX, VMAX)
    art = []
    for r, row in enumerate(AGENTS):
        for c, a in enumerate(row):
            ax = fig.add_axes([(.35 + c * (pw + gap)) / W, (H - .65 - (r + 1) * pw - r * .35) / H, pw / W, pw / H])
            P = np.stack([D[a]["proj"](D[a]["M"][:, t]) for t in range(T)], 1)    # (100, T, 2)
            Hm = D[a]["proj"](D[a]["home"])                                         # targets (100, 2)
            allp = np.vstack([P.reshape(-1, 2), Hm])
            lo, hi = allp.min(0), allp.max(0)
            ctr, side = (lo + hi) / 2, (hi - lo).max() * 1.08
            ax.set_xlim(ctr[0] - side / 2, ctr[0] + side / 2), ax.set_ylim(ctr[1] - side / 2, ctr[1] + side / 2)
            ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(NAME[a], fontsize=11, pad=4)
            ax.add_collection(LineCollection(Hm[E], colors="0.72", linewidths=.6, linestyles=(0, (3, 2)), zorder=0))
            mesh = ax.add_collection(LineCollection(P[:, 0][E], colors="0.3", linewidths=.65, alpha=.8, zorder=2))
            dots = ax.scatter(*P[:, 0].T, s=20, c=D[a]["C"][:, 0], cmap=COV_CMAP, norm=norm, edgecolors="0.25",
                              linewidths=.3, zorder=3)
            art.append((a, P, mesh, dots))
    cx = fig.add_axes([(.35 + 2 * pw + gap + .3) / W, (H - .65 - 2 * pw - .35) / H, .14 / W, (2 * pw + .35) / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=COV_CMAP, norm=norm), cax=cx, extend="both")
    cb.set_label(r"$C_t = \mathrm{Cov}_{\mathrm{streams}}(|u_g(t)|, |u_c(t)|)$ of the cell (grid units$^2$)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    clock = fig.text(.35 / W, (H - .3) / H, "", fontsize=13, va="center")
    fig.text(.35 / W, .3 / H, "Environment %d of the alias scan: each dot is one cell's mean belief over its %d streams, "
             "in panel (c)'s T = 30 plane, joined to its grid neighbours;\ndashed: the same grid of the cells' targets; "
             "colour: whether the cell's streams move both variables together at the step (grey late: no movement)."
             % (ENV, M), fontsize=7, color="0.35", va="top", linespacing=1.3)

    def update(tau):
        s = int(np.floor(tau))
        f = min(tau - s, 1.)
        s1 = min(s + 1, T - 1)
        for a, P, mesh, dots in art:
            pos = D[a]["proj"](D[a]["blend"](D[a]["M"], tau, axis=1))            # (100, 2)
            dots.set_offsets(pos)
            dots.set_array((1 - f) * D[a]["C"][:, s] + f * D[a]["C"][:, s1])
            mesh.set_segments(pos[E])
        clock.set_text("step $t$ = %d" % (int(np.floor(tau + 1e-9)) + 1))

    def grab():
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    protos = []
    for tau in (0., 3., 9., T - 1.):
        update(tau)
        protos.append(np.asarray(grab()))
    pal = Image.fromarray(np.concatenate(protos, 0)).quantize(colors=255, method=Image.Quantize.MEDIANCUT,
                                                              dither=Image.Dither.NONE)
    frames = []
    for tau, _ in sched:
        update(tau)
        frames.append(grab().quantize(palette=pal, dither=Image.Dither.NONE))
    plt.close(fig)
    from main.utils import fig_path
    path = fig_path("fig1c_movie_cellmeans.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[ms for _, ms in sched], loop=0)
    print(f"saved {path}: {len(frames)} frames, {sum(ms for _, ms in sched) / 1000:.1f} s, {frames[0].size[0]} x "
          f"{frames[0].size[1]} px, {os.path.getsize(path) / 1e6:.1f} MB")
    for row in AGENTS:
        for a in row:
            C = D[a]["C"]
            print("%-8s cell C_t: mean at t=1/3/5/10/20: %s; time-mean per cell %.3f .. %.3f" % (
                a, np.round(C.mean(0)[[0, 2, 4, 9, 19]], 3), C.mean(1).min(), C.mean(1).max()))
    return path


if __name__ == "__main__":
    movie()
