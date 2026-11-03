"""Every step of the read-out drawn as an arch between its two time points, lifted by its signed delta (28 Oct 2026;
John on the (E[r_g], E[r_c], delta) cloud: "big diagonal steps are fine, our manifold should exist between points at 2
timepoints. This should be able to close the loops?")
-> main/REPLICATION/figure_bin/fig1c_step_arches.png

A step from x_t to x_{t+1} (posterior means (E[r_g], E[r_c]), t = 1 .. 29) with change (dg, dc) has the signed delta
delta = sign(dg dc) sqrt|dg dc| (cells): + while the estimate moves along the (0, 0) - (9, 9) diagonal (a step of
length L along it: L / sqrt 2; corner to corner: 9), - along (0, 9) - (9, 0), 0 along an axis. It is drawn as an arch
that leaves the plane at x_t, peaks at height delta halfway and lands at x_{t+1}: a jump between opposite corners spans
the diagonal, so the diagonal (on the plane) and the arch over it close a loop; (0, 0) - (9, 9) loops close above the
plane, (0, 9) - (9, 0) loops below. Colour: delta (red +, blue -); opacity grows with |delta| (small steps fade).
Columns: Optimal, Trained RNN, Naive, ESN (each its own read-out); rows: seen from above across (0, 0) - (9, 9) and
from below across (0, 9) - (9, 0). Data: the alias scan's 39 environments x N_PER_CELL streams per cell, all 29 steps.
numbers() (all 5,000 streams): the share of steps with |delta| > 2 and > 4.5 cells (half the corner-to-corner
span), the share of them that are +, and their mean length along their own diagonal.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_step_arches.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from mpl_toolkits.mplot3d import proj3d                                           # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection                           # noqa: E402
import plot_fig1c_lifted_plane as lp                                              # noqa: E402

N_PER_CELL, NSEG = 2, 8
ZMAX, A_MAX, GAMMA, LW = 6., .55, 1.3, .6


def deltas(X):
    """positions (..., T, 2) -> step starts, ends (..., T - 1, 2) and signed deltas (..., T - 1)"""
    a, b = X[..., :-1, :], X[..., 1:, :]
    p = (b[..., 0] - a[..., 0]) * (b[..., 1] - a[..., 1])
    return a, b, np.sign(p) * np.sqrt(np.abs(p))


def arches(agent):
    """the visible steps as arch segments (m * NSEG, 2, 3) and each segment's step delta"""
    X = lp.means(agent, np.flatnonzero(np.arange(5000) % 50 < N_PER_CELL)).astype(np.float64)
    a, b, z = deltas(X)
    a, b, z = a.reshape(-1, 2), b.reshape(-1, 2), z.ravel()
    keep = A_MAX * np.minimum(np.abs(z) / ZMAX, 1) ** GAMMA >= .01
    a, b, z = a[keep], b[keep], z[keep]
    s = np.linspace(0, 1, NSEG + 1)
    P = a[:, None] + s[None, :, None] * (b - a)[:, None]                         # (m, NSEG + 1, 2)
    h = np.clip(4 * z[:, None] * s * (1 - s), -ZMAX - 1, ZMAX + 1)
    Q = np.concatenate([P, h[..., None]], -1)
    seg = np.stack([Q[:, :-1], Q[:, 1:]], 2).reshape(-1, 2, 3)
    return seg, np.repeat(z, NSEG)


def numbers():
    for agent, name in lp.AGENTS:
        a, b, z = deltas(lp.means(agent).astype(np.float64))
        d = b - a
        own = np.where(z > 0, np.abs(d[..., 0] + d[..., 1]), np.abs(d[..., 0] - d[..., 1])) / np.sqrt(2)
        line = f"{name:12s} steps {z.size:,}:"
        for th in (2, 4.5):
            m = np.abs(z) > th
            line += (f" |delta| > {th}: {100 * m.mean():.3f}% (+ {100 * (z[m] > 0).mean():.0f}% of them; their mean "
                     f"length along their diagonal {own[m].mean():.1f} cells);")
        print(line, flush=True)


def figure():
    import plotting_helpers as ph
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    cmap = plt.get_cmap("RdBu_r")
    S, SH, GX, GY, X0, YT = 3.5, 3.0, .25, .1, .85, 1.0
    W, H = X0 + len(lp.AGENTS) * S + (len(lp.AGENTS) - 1) * GX + .15, YT + 2 * SH + GY + .1
    fig = plt.figure(figsize=(W, H))
    fig.text(.5, 1 - .22 / H, "Every step of the read-out as an arch from $x_t$ to $x_{t+1}$, peaking at its signed "
             r"delta $\delta$ = sign$(\Delta_g \Delta_c)\sqrt{|\Delta_g \Delta_c|}$ (cells): + along (0, 0) - (9, 9), "
             "- along "
             "(0, 9) - (9, 0); a corner-to-corner jump spans its diagonal", ha="center", va="center",
             fontsize=FS["label"])
    fig.text(.5, 1 - .48 / H, f"39 environments x {N_PER_CELL} streams per cell x 29 steps per agent; opacity grows "
             r"with $|\delta|$ (small steps fade)", ha="center", va="center", fontsize=FS["label"])
    sq = np.array([[0, 0], [9, 0], [9, 9], [0, 9], [0, 0]])
    for c_, (agent, name) in enumerate(lp.AGENTS):
        seg, z = arches(agent)
        col = cmap((np.clip(z, -ZMAX, ZMAX) + ZMAX) / (2 * ZMAX))
        col[:, 3] = A_MAX * np.minimum(np.abs(z) / ZMAX, 1) ** GAMMA
        mid = seg.mean(1)
        for r_, (elev, azim, view) in enumerate(((20, -45, "from above\nacross (0, 0) - (9, 9)"),
                                                 (-20, 45, "from below\nacross (0, 9) - (9, 0)"))):
            y = H - YT - SH - r_ * (SH + GY)
            ax = fig.add_axes([(X0 + c_ * (S + GX)) / W, y / H, S / W, SH / H], projection="3d")
            ax.set_proj_type("ortho")
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(-.25, 9.25), ax.set_ylim(-.25, 9.25), ax.set_zlim(-ZMAX - 1, ZMAX + 1)
            ax.set_box_aspect((1, 1, 1.0), zoom=1.05)
            ax.set_axis_off()
            depth = proj3d.proj_transform(mid[:, 0], mid[:, 1], mid[:, 2], ax.get_proj())[2]
            o = np.argsort(-depth, kind="stable")
            for v in range(10):
                ax.plot([v, v], [0, 9], 0, color="0.8", lw=.4)
                ax.plot([0, 9], [v, v], 0, color="0.8", lw=.4)
            ax.plot(sq[:, 0], sq[:, 1], 0, color="0.4", lw=.9)
            ax.add_collection3d(Line3DCollection(seg[o], colors=col[o], linewidths=LW))
            for gx_, gy_ in ((0, 0), (9, 9), (0, 9), (9, 0)):
                ax.text(gx_ + (-.9 if gx_ == 0 else .9), gy_ + (-.9 if gy_ == 0 else .9), 0, f"({gx_}, {gy_})",
                        ha="center", va="center", fontsize=FS["tick"], color="0.2")
            if r_ == 0:
                ax.set_title(name, fontsize=FS["label"] + 2, color=ph.AGENT_COLOR[agent], pad=-6)
            if c_ == 0:
                fig.text(.3 / W, (y + SH / 2) / H, view, rotation=90, ha="center", va="center",
                         fontsize=FS["label"])
    path = fig_path("fig1c_step_arches.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


if __name__ == "__main__":
    numbers()
    print("saved", figure())
