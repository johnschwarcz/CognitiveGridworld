"""Where in the read-out plane the particles move along a diagonal, step by step: the plane cut into tiles, each coloured
by the mean covariance of the particles on it, with each agent's manifold drawn on top (28 Oct 2026; John on the lifted
plane: "this visualization isnt coming out great, is there a simple way for us to visualize the manifold, with its tiles
colored by the average covariance or slope, etc.")
-> main/REPLICATION/figure_bin/fig1c_cov_tiles.png

Rows:
    Optimal, belief covariance   its own exact joint covariance Cov(r_g, r_c) (cells^2): + while its belief is split along
                                 the (0, 0) - (9, 9) diagonal, - along (0, 9) - (9, 0)
    each agent, path covariance  Optimal, Trained RNN, Naive, ESN: the covariance of its own posterior means over a
                                 Gaussian window of steps around t (sigma 1.5; plot_fig1c_lifted_plane.window): + while the
                                 particle moves along the (0, 0) - (9, 9) diagonal, - along (0, 9) - (9, 0); the networks
                                 have no belief covariance (their read-out is two separate marginals)
Columns: steps STEPS. Tiles: TILE x TILE cells of the plane, at the particles' raw posterior means (E[r_g], E[r_c]) at
that step; colour: the mean over the particles on the tile (all 39 environments x 5,000 streams pooled); blank: fewer
than MIN_N particles. Fixed ranges (red +, blue -): +-2 for the belief covariance, +-0.3 for the path covariance (early
steps exceed it: the colour bar's arrows). Lines: the agent's manifold, its 100 cell means (over the 39 environments x
50 streams) joined to their lattice neighbours. A tile per CELL (the cell's particles wherever they are) was tried
first: it shows the early four-quadrant pattern but washes out the diagonals.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/plot_fig1c_cov_tiles.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402
import plot_fig1c_lifted_plane as lp                                              # noqa: E402

E, K, M, T, R = 39, 100, 50, 30, 10
STEPS = (2, 4, 7, 12, 20)
TILE, MIN_N = .5, 20
LIM = {"belief": 2., "path": .3}


def tiles(P, V):
    """positions (n, 2), values (n,) -> the tile edges and the tiles' mean values (nan: fewer than MIN_N)"""
    ed = np.arange(-.25, R - .75 + 1e-9, TILE)
    nb = len(ed) - 1
    i = np.clip(np.digitize(P[:, 0], ed) - 1, 0, nb - 1)
    j = np.clip(np.digitize(P[:, 1], ed) - 1, 0, nb - 1)
    n = np.bincount(i * nb + j, minlength=nb * nb)
    s = np.bincount(i * nb + j, V, minlength=nb * nb)
    return ed, np.where(n >= MIN_N, s / np.maximum(n, 1), np.nan).reshape(nb, nb)


def rows():
    """[(label, quantity, positions (E, n, T, 2), values (E, n, T), manifold (K, T, 2))]"""
    out = []
    for agent, name in lp.AGENTS:
        X = lp.means(agent).astype(np.float64)
        grid = X.reshape(E, K, M, T, 2).mean((0, 2))
        if agent == "Joint":
            out.append((name, "belief", X, lp.belief_cov().astype(np.float64), grid))
        out.append((name, "path", X, lp.window(X)[1], grid))
    return out


def figure():
    import plotting_helpers as ph
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    edges, _ = lp_lattice()
    data = rows()
    S, GX, GY, X0, YT, XR = 2.2, .12, .12, 1.45, 1.0, 1.25
    W, H = X0 + len(STEPS) * S + (len(STEPS) - 1) * GX + XR, YT + len(data) * S + (len(data) - 1) * GY + .2
    fig = plt.figure(figsize=(W, H))
    fig.text(.5, 1 - .25 / H, "Tiles: the mean covariance of the particles on each 0.5-cell square of the read-out plane "
             "at step $t$ (39 environments x 5,000 streams each, pooled; blank: $<$ 20 particles).  Lines: the agent's "
             "manifold (its cell means)", ha="center", va="center", fontsize=FS["label"])
    fig.text(.5, 1 - .5 / H, r"belief covariance: Cov$(r_g, r_c)$ of the Optimal's posterior.  Path covariance: "
             r"$\Sigma_s w_{ts} (E[r_g]_s - m_t)(E[r_c]_s - m_t)$ over a Gaussian window of steps ($\sigma$ = 1.5), "
             "each agent's own read-out", ha="center", va="center", fontsize=FS["label"])
    cmap = plt.get_cmap("RdBu_r")
    for r_, (name, kind, X, V, grid) in enumerate(data):
        y = H - YT - S - r_ * (S + GY)
        lim = LIM[kind]
        for c_, t in enumerate(STEPS):
            ax = fig.add_axes([(X0 + c_ * (S + GX)) / W, y / H, S / W, S / H])
            ed, Z = tiles(X[:, :, t - 1].reshape(-1, 2), V[:, :, t - 1].ravel())
            im = ax.pcolormesh(ed, ed, Z.T, cmap=cmap, vmin=-lim, vmax=lim)
            ax.add_collection(LineCollection(grid[:, t - 1][edges], colors="0.25", linewidths=.45))
            ax.set_xlim(-.6, 9.6), ax.set_ylim(-.6, 9.6), ax.set_aspect("equal")
            ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
            ax.tick_params(labelsize=FS["tick"] - 1, length=2, pad=1)
            if c_:
                ax.set_yticklabels([])
            if r_ < len(data) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r"$E[r_g]$", labelpad=-6, fontsize=FS["tick"])
            if r_ == 0:
                ax.set_title(f"$t = {t}$", fontsize=FS["label"] + 1, pad=3)
        label = f"{name}\n" + ("belief covariance" if kind == "belief" else "path covariance")
        fig.text((X0 - .55) / W, (y + S / 2) / H, label, rotation=90, ha="center", va="center",
                 fontsize=FS["label"], color=ph.AGENT_COLOR[[a for a, n in lp.AGENTS if n == name][0]])
        fig.text((X0 - .2) / W, (y + S / 2) / H, r"$E[r_c]$", rotation=90, ha="center", va="center",
                 fontsize=FS["tick"])
        if r_ in (0, 1):                                                          # one colour bar per quantity
            hgt = S if r_ == 0 else (len(data) - 1) * S + (len(data) - 2) * GY
            cax = fig.add_axes([(W - XR + .25) / W, (y + S - hgt) / H, .13 / W, hgt / H])
            cb = fig.colorbar(im, cax=cax, extend="both")
            cb.set_ticks([-lim, 0, lim]), cb.ax.tick_params(labelsize=FS["tick"])
            cb.set_label(("belief" if r_ == 0 else "path") + r" covariance, cells$^2$", fontsize=FS["tick"])
    path = fig_path("fig1c_cov_tiles.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def lp_lattice():
    """the lattice's edges (index pairs over the goal-ordered cells k = 10 r_g + r_c) and the cells"""
    g = np.arange(R)
    e = [(R * a + b, R * (a + da) + b + db) for a in g for b in g for da, db in ((1, 0), (0, 1))
         if a + da < R and b + db < R]
    return np.array(e), np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)


if __name__ == "__main__":
    print("saved", figure())
