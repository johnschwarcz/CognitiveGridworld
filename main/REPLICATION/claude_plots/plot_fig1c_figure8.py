"""The figure 8 with the particles on it: the Optimal's belief states placed in the evidence geometry (28 Oct 2026;
John: "i expected the wormholes to have gaps, like our figure 8 discussion ... the data is a shape in 3 dimensional
space that we can visualize as a manifold. We expect to see our figure 8.")
-> main/REPLICATION/figure_bin/fig1c_figure8.png

Why not (E[r_g], E[r_c], a third axis): on those two axes opposite corners stay 9 sqrt 2 = 12.7 cells apart, so no third
axis can bring (0, 0) near (9, 9); a near-closed loop cannot form there. The figure 8 lives in the evidence geometry
(plot_fig1c_evidence_geometry.py): the 100 cells embedded by classical MDS of the sqrt-Jeffreys distances between their
observation distributions, where each diagonal is a near-closed loop (its ends, the opposite corners, close together)
and the two loops cross at the centre.
Here: environment E_ID (a typical one; its 3-D embedding holds 97% of the positive eigenvalues). Grey: the cells'
sheet (lattice neighbours joined). Thick: the two diagonals, the loops of the 8 (red (0, 0) - (9, 9), blue (0, 9) -
(9, 0)); dashed: each loop's gap between its corners. Points: every stream-step of the Optimal (5,000 streams x 30
steps) at its exact joint posterior's weighted average of the cells' coordinates, coloured by its belief covariance
Cov(r_g, r_c) (red +: split along (0, 0) - (9, 9); blue -: along (0, 9) - (9, 0)); opacity grows with |Cov|, so
settled states (Cov ~ 0, on the sheet) fade. A belief split between two cells sits between them: a (0, 0) / (9, 9)
split sits in that loop's gap.
numbers() (all 39 environments, each in its own embedding): the split states (|Cov| > SPLIT) of each sign, their mean
distance from their own gap's midpoint and from the other gap's, in units of the environment's median distance between
cells.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/plot_fig1c_figure8.py"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection                           # noqa: E402
import plot_fig1c_evidence_geometry as eg                                         # noqa: E402

R, E_ID, SPLIT, CMAX = 10, 34, 8., 12.
ENVS = [e for e in range(40) if e != 2]
MAIN = [R * k + k for k in range(R)]                                              # (k, k)
ANTI = [R * k + (R - 1 - k) for k in range(R)]                                    # (k, 9 - k)


def posterior(e):
    """the Optimal's exact joint posterior (n, T, 100), goal-ordered k = 10 r_g + r_c, and its Cov (n, T)"""
    meta = np.load(eg.SCAN_META)
    L, g, obs = meta["L"][e], int(meta["goal"][e]), meta["obs"][e].astype(np.float64)
    ll = (np.einsum("nto,oab->ntab", obs, np.log(L)) + np.einsum("nto,oab->ntab", 1 - obs, np.log1p(-L))).cumsum(1)
    p = np.exp(ll - ll.max((2, 3), keepdims=True))
    p /= p.sum((2, 3), keepdims=True)
    if g == 1:
        p = p.transpose(0, 1, 3, 2)
    r = np.arange(R, dtype=np.float64)
    cov = np.einsum("ntab,a,b->nt", p, r, r) - (p.sum(3) @ r) * (p.sum(2) @ r)
    return p.reshape(*p.shape[:2], R * R), cov


def numbers(D):
    """per environment: the split states' mean distance from their own gap's midpoint and from the other's"""
    rows = []
    for i, e in enumerate(ENVS):
        Y, _ = eg.embed(D["evidence"][i])
        med = np.median(eg._dist(Y)[np.triu_indices(R * R, 1)])
        p, cov = posterior(e)
        X = p @ Y
        mid = {1: (Y[0] + Y[99]) / 2, -1: (Y[9] + Y[90]) / 2}
        row = [np.linalg.norm(Y[0] - Y[99]) / med, np.linalg.norm(Y[9] - Y[90]) / med,
               np.linalg.norm(mid[1] - mid[-1]) / med]
        for s in (1, -1):
            m = (np.sign(cov) == s) & (np.abs(cov) > SPLIT)
            P = X[m]
            row += [m.sum(), np.linalg.norm(P - mid[s], axis=1).mean() / med if len(P) else np.nan,
                    np.linalg.norm(P - mid[-s], axis=1).mean() / med if len(P) else np.nan]
        rows.append(row)
    a = np.array(rows)
    md = np.nanmedian(a, 0)
    print(f"gaps / median pair: (0,0)-(9,9) {md[0]:.2f}, (0,9)-(9,0) {md[1]:.2f}; the two gaps' midpoints "
          f"{md[2]:.2f} apart (medians over 39 environments)")
    for j, nm in ((3, "+ splits (along (0,0)-(9,9))"), (6, "- splits (along (0,9)-(9,0))")):
        print(f"{nm}: {int(a[:, j].sum()):,} stream-steps; mean distance from their own gap's midpoint "
              f"{md[j + 1]:.2f}, from the other gap's {md[j + 2]:.2f}; nearer their own in "
              f"{int(np.sum(a[:, j + 1] < a[:, j + 2]))} of {int(np.sum(np.isfinite(a[:, j + 1])))} environments")


def figure(D):
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    Y, share = eg.embed(D["evidence"][ENVS.index(E_ID)])
    p, cov = posterior(E_ID)
    P, v = (p @ Y).reshape(-1, 3), cov.ravel()
    o = np.argsort(np.abs(v))                                                     # strongest splits drawn last
    col = plt.get_cmap("RdBu_r")((np.clip(v, -CMAX, CMAX) + CMAX) / (2 * CMAX))
    col[:, 3] = np.clip(.015 + .6 * (np.abs(v) / CMAX) ** 1.5, 0, .7)
    lat = np.array([(R * a + b, R * a + b + 1) for a in range(R) for b in range(R - 1)] +
                   [(R * a + b, R * (a + 1) + b) for a in range(R - 1) for b in range(R)])
    views = ((28, -62, "the (0, 0) - (9, 9) gap in front"), (-28, -62, "from below: the (0, 9) - (9, 0) gap"),
             (4, 118, "side view: the two gaps on opposite sides"))
    S, W, H = 4.6, 3 * 4.6 + .4, 4.9
    fig = plt.figure(figsize=(W, H))
    fig.text(.5, 1 - .22 / H, f"The evidence geometry of environment {E_ID} (3-D MDS of the cells' evidence distances; "
             f"{100 * share:.0f}% of it): the two diagonals are near-closed loops (thick), their gaps dashed.  Points: "
             "the Optimal's states (5,000 streams x 30 steps)", ha="center", va="center", fontsize=FS["label"])
    fig.text(.5, 1 - .48 / H, r"at its posterior's weighted average of the cells, coloured by its belief covariance "
             r"Cov$(r_g, r_c)$ (red: split along (0, 0) - (9, 9), blue: along (0, 9) - (9, 0); faint: settled)",
             ha="center", va="center", fontsize=FS["label"])
    lo, hi = Y.min(0) - .15, Y.max(0) + .15
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_axes([(.2 + i * S) / W, .02, S / W, (H - .85) / H], projection="3d")
        ax.set_proj_type("ortho")
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=1.35)
        ax.set_axis_off()
        ax.add_collection3d(Line3DCollection(Y[lat], colors="0.72", linewidths=.5))
        ax.scatter(P[o, 0], P[o, 1], P[o, 2], c=col[o], s=2, lw=0, depthshade=False)
        for idx, c in ((MAIN, "#8b0000"), (ANTI, "#08306b")):
            ax.plot(*Y[idx].T, color=c, lw=2.6)
            ax.plot(*Y[[idx[0], idx[-1]]].T, color=c, lw=1.2, ls=(0, (3, 2)))
        for k, mk, lab in ((0, "o", "(0, 0)"), (99, "o", "(9, 9)"), (9, "s", "(0, 9)"), (90, "s", "(9, 0)")):
            ax.scatter(*Y[k], s=45, marker=mk, facecolor="white", edgecolor="k", lw=1, depthshade=False)
            if i < 2 or k in (0, 99):                                              # side view: the squares overlap
                ax.text(*(Y[k] + .22 * (Y[k] - Y.mean(0)) / np.linalg.norm(Y[k] - Y.mean(0))), lab,
                        fontsize=FS["tick"], ha="center", va="center")
        ax.set_title(title, fontsize=FS["label"], pad=-4)
    path = fig_path("fig1c_figure8.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


if __name__ == "__main__":
    D, _, _ = eg.geometries()
    numbers(D)
    print("saved", figure(D))
