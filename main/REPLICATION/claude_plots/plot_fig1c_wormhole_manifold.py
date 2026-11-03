"""The wormholes as surfaces above and below the read-out plane, all steps pooled (28 Oct 2026; John after the
time-sliced tile maps: "again, splitting by time doesnt come out great. I'd like to see a 3D manifold that shows the
wormholes above and below the typical plane, we dont need to slice time for this.")
-> main/REPLICATION/figure_bin/fig1c_wormhole_manifold.png

The plane is cut into TILE x TILE-cell tiles; every particle at every step (30 steps x 39 environments x 5,000 streams,
pooled) sits on the tile of its posterior means (E[r_g], E[r_c]). On each tile, the upper surface is the Q-th percentile
of the particles' covariance and the lower surface the (100 - Q)-th: the envelope of the states, set by the few split
ones (most particles on any tile are settled, at covariance ~0, and would wash out a mean). Both surfaces lightly
smoothed (Gaussian, SMOOTH tiles); tiles with fewer than MIN_N particles at 0. Grey: the plane, z = 0. Views: from
above looking across the (0, 0) - (9, 9) diagonal (its wormhole side-on), from below across (0, 9) - (9, 0).
    belief covariance  the Optimal's own exact Cov(r_g, r_c): + while its belief is split along (0, 0) - (9, 9),
                       - along (0, 9) - (9, 0)
    path covariance    each agent's own: the covariance of its posterior means over a Gaussian window of steps around t
                       (sigma 1.5; plot_fig1c_lifted_plane.window): + while moving along (0, 0) - (9, 9), - along
                       (0, 9) - (9, 0) (the networks and the Naive have no belief covariance: their read-out is two
                       separate marginals)
Columns: the Optimal's belief covariance, then the path covariance of the Optimal, Trained RNN, Naive and ESN (one
height scale for the four); rows: seen from above and from below (orthographic).
Measured (the unsmoothed envelopes; mean over the tiles on each diagonal): Q = 99 upper surface on the (0, 0) - (9, 9)
diagonal vs on (0, 9) - (9, 0), lower surface on (0, 9) - (9, 0) vs on (0, 0) - (9, 9): Optimal belief 11.5 / 3.3,
-11.9 / -4.0 cells^2; path: Optimal 2.39 / 0.60, -2.53 / -0.69; Trained RNN 2.01 / 0.62, -2.05 / -0.65; Naive 0.63 /
0.39, -0.62 / -0.33; ESN 0.68 / 0.36, -0.70 / -0.32 (numbers() prints them).
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_wormhole_manifold.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from scipy.ndimage import gaussian_filter                                         # noqa: E402
import plot_fig1c_lifted_plane as lp                                              # noqa: E402

R = 10
TILE, MIN_N, Q, SMOOTH = .5, 200, 99, 1.25
EDGES = np.arange(-.25, R - .75 + 1e-9, TILE)
NB = len(EDGES) - 1
CTR = (EDGES[:-1] + EDGES[1:]) / 2
ZMAX = {"belief": 12., "path": 2.5}


def envelope(P, V):
    """positions (n, 2), values (n,) -> the upper (Q-th) and lower ((100 - Q)-th percentile) surfaces (NB, NB)"""
    i = np.clip(np.digitize(P[:, 0], EDGES) - 1, 0, NB - 1)
    j = np.clip(np.digitize(P[:, 1], EDGES) - 1, 0, NB - 1)
    k = i * NB + j
    o = np.argsort(k, kind="stable")
    ks, vs = k[o], V[o]
    cut = np.searchsorted(ks, np.arange(NB * NB + 1))
    out = np.zeros((2, NB * NB))
    for t in range(NB * NB):
        v = vs[cut[t]:cut[t + 1]]
        if len(v) >= MIN_N:
            out[:, t] = np.percentile(v, [Q, 100 - Q])
    return out.reshape(2, NB, NB)


def panels():
    """[(agent, title, kind, envelope (2, NB, NB))]: the Optimal's belief covariance, then each agent's path
    covariance"""
    out = []
    for agent, name in lp.AGENTS:
        X = lp.means(agent).astype(np.float64)
        P = X.reshape(-1, 2)
        if agent == "Joint":
            out.append((agent, name, "belief", envelope(P, lp.belief_cov().astype(np.float64).ravel())))
        out.append((agent, name, "path", envelope(P, lp.window(X)[1].ravel())))
    return out


def numbers(data):
    d = np.arange(NB)
    for agent, name, kind, (up, lo) in data:
        print(f"{name:12s} {kind:6s} upper surface on (0,0)-(9,9) {up[d, d].mean():.2f} vs on (0,9)-(9,0) "
              f"{up[d, NB - 1 - d].mean():.2f}; lower on (0,9)-(9,0) {lo[d, NB - 1 - d].mean():.2f} vs on (0,0)-(9,9) "
              f"{lo[d, d].mean():.2f}", flush=True)


def figure(data):
    import plotting_helpers as ph
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    gx, gy = np.meshgrid(CTR, CTR, indexing="ij")
    S, SH, GX, GY, X0, YT = 3.1, 2.7, .4, .1, .85, 1.0
    W, H = X0 + len(data) * S + (len(data) - 1) * GX + .15, YT + 2 * SH + GY + .1
    fig = plt.figure(figsize=(W, H))
    fig.text(.5, 1 - .22 / H, f"Upper surface: the {Q}th percentile, lower: the {100 - Q:g}st, of the covariance of "
             "the "
             "particles on each 0.5-cell tile of the read-out plane, all 30 steps pooled (39 environments x 5,000 "
             "streams each).  Grey: the plane", ha="center", va="center", fontsize=FS["label"])
    fig.text(.5, 1 - .48 / H, r"red: split along / moving along (0, 0) - (9, 9);  blue: along (0, 9) - (9, 0).  "
             r"Heights: $\pm$12 cells$^2$ (belief covariance), $\pm$2.5 cells$^2$ (path covariance)",
             ha="center", va="center", fontsize=FS["label"])
    sq = np.array([[0, 0], [9, 0], [9, 9], [0, 9], [0, 0]])
    for c_, (agent, name, kind, (up, lo)) in enumerate(data):
        zm = ZMAX[kind]
        up, lo = gaussian_filter(up, SMOOTH), gaussian_filter(lo, SMOOTH)
        for r_, (elev, azim, view) in enumerate(((20, -45, "from above\nacross (0, 0) - (9, 9)"),
                                                 (-20, 45, "from below\nacross (0, 9) - (9, 0)"))):
            y = H - YT - SH - r_ * (SH + GY)
            ax = fig.add_axes([(X0 + c_ * (S + GX)) / W, y / H, S / W, SH / H], projection="3d")
            ax.set_proj_type("ortho")
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(-.25, 9.25), ax.set_ylim(-.25, 9.25), ax.set_zlim(-zm, zm)
            ax.set_box_aspect((1, 1, .8), zoom=1.0)
            ax.set_axis_off()
            ax.plot_surface(gx, gy, np.zeros_like(gx), color="0.8", alpha=.35, linewidth=0, shade=False)
            for v in range(R):                                                     # the plane
                ax.plot([v, v], [0, 9], 0, color="0.5", lw=.4)
                ax.plot([0, 9], [v, v], 0, color="0.5", lw=.4)
            for Z, cm in ((up, "Reds"), (lo, "Blues")):
                col = plt.get_cmap(cm)(.12 + .88 * np.clip(np.abs(Z) / zm, 0, 1))
                col[..., 3] = .8
                ax.plot_surface(gx, gy, np.clip(Z, -zm, zm), facecolors=col, rstride=1, cstride=1, linewidth=0,
                                antialiased=True, shade=False)
            ax.plot(sq[:, 0], sq[:, 1], 0, color="0.3", lw=.9)
            for gx_, gy_ in ((0, 0), (9, 9), (0, 9), (9, 0)):
                ax.text(gx_ + (-.9 if gx_ == 0 else .9), gy_ + (-.9 if gy_ == 0 else .9), 0, f"({gx_}, {gy_})",
                        ha="center", va="center", fontsize=FS["tick"], color="0.2")
            if r_ == 0:
                ax.set_title(f"{name}\n{kind} covariance", fontsize=FS["label"] + 1, color=ph.AGENT_COLOR[agent],
                             pad=-8)
            if c_ == 0:
                fig.text(.3 / W, (y + SH / 2) / H, view, rotation=90, ha="center", va="center",
                         fontsize=FS["label"])
    path = fig_path("fig1c_wormhole_manifold.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


if __name__ == "__main__":
    data = panels()
    numbers(data)
    print("saved", figure(data))
