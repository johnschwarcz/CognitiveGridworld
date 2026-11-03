"""The geometry of the evidence, of the Optimal's updates and of the Naive's updates (27 Oct 2026; replaces the torus
figure: John asked not to force a torus reading, and the data show a curled sheet, not a torus; the Optimal's row and
panels g, h added after John asked how the Naive can curl less than the evidence)
-> main/REPLICATION/figure_bin/fig1c_evidence_geometry.png

Data: the alias scan's likelihood tables (main/DATA/experimental/fig1c_compression/scan_meta.npz 'L', 'goal'; 39
environments, e = 2 left out), every cell goal-ordered (r_g, r_c), cell k = 10 r_g + r_c; p_ik = P(o_i = 1 | k) for the
5 observations, l_ik = logit(p_ik). Rows: the evidence; the Optimal's updates; the Naive's updates.
Distances between cells k, k':
    evidence  sqrt of the Jeffreys divergence J(k, k') = sum_i (p_ik - p_ik')(l_ik - l_ik') (one step; ~ the distance
              in information for nearby cells)
    Optimal   |mu_k - mu_k'|, mu_k(j) = E[log p(o | j) | k] over the 100 cells j, centred over j: the expected per-step
              change of the Optimal's log-belief when the true cell is k
    Naive     |mu_k - mu_k'|, mu_k(r) = E[log q(o | r) | k], q the tables averaged over the other variable (the Naive's
              models): the expected per-step change of its two log-beliefs, each belief's 10 values centred (a constant
              shift leaves a belief unchanged)
    a, c, e  the 100 cells embedded by classical MDS of those distances, environment 34 (a typical one), rotated
             (orthogonal Procrustes) onto the cell lattice so the three share an orientation: grid neighbours joined,
             corners (0, 0), (9, 9): circles, (0, 9), (9, 0): squares
    b, d, f  distance from corner (0, 0) and from corner (0, 9) to every cell, median over the environments (square: the
             opposite corner)
    g        per environment: the opposite corners' distance (mean of the two pairs) against the adjacent corners'
             (mean of the four pairs), each divided by the median distance between all pairs of cells; large: the
             medians over environments. Evidence 0.51 / 1.48, Optimal 0.58 / 1.53, Naive 0.73 / 0.88; the Naive with
             every cell's update rescaled to the same size (direction kept): 0.95 / 1.19. The Naive's opposite corners
             are farther apart than the evidence's in 39 of 39 environments, its adjacent corners nearer in 37 of 39
    h        the size of each cell's expected update |mu_k| (nats per step), median over environments
Why the evidence curls: each observation's logit is A_i [cos(w v + a_i) - cos(w u + b_i)] / 2 with u = r_g + r_c,
v = r_g - r_c, w = 2 pi / 21 (period-21 sine features of the two diagonals), and the grid's diagonals span 19 of the 21
positions, so their ends come back near each other: only the opposite corners meet; adjacent corners stay far apart
(idealized, weak logits and isotropic bits: the chord metric on the u and v circles, 0.47 / 1.48). The Optimal reads
the evidence through the full joint tables and inherits the curl.
Why the Naive curls less (its sheet is pinched, not curled): its tables average each observation over the other
variable, which leaves sinusoids of r_g alone and of r_c alone, each spanning only 10 of the 21 positions; nothing in
its view wraps. Averaging also makes its evidence about one variable strong only when the other is near the middle
(idealized amplitude cos(w (r_c - 4.5)): 1 in the middle, 0.22 at a border; the tables: 0.6 vs 1.4 of the median), so
at the corners, where both variables are at a border, its updates are weakest (h). All four corners crowd together:
adjacent corners as well as opposite ones (g). Rescaled to equal update sizes, the Naive's opposite corners are no
closer than a typical pair (0.95): its corner closeness comes entirely from the weak corner updates. (Idealized
read-out: 0.64 / 0.45.) The Naive's updates are a linear read-out of the same evidence (a 20 x 5 matrix applied to the
cells' observation probabilities), so it cannot separate what the evidence does not, but it re-weights distances, and
ratios such as these can move either way.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_evidence_geometry.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.colors import LogNorm                                             # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection                           # noqa: E402

SCAN_META = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression", "scan_meta.npz")
R = 10
ENVS = [e for e in range(40) if e != 2]
CORNERS = ((0, 0), (0, 9))
OPPOSITE, ADJACENT = ((0, 99), (9, 90)), ((0, 9), (0, 90), (9, 99), (90, 99))
SHOW_ENV = 34
GEOMS = ("evidence", "Optimal", "Naive")
COLOR = {"evidence": "#6a3d9a", "Optimal": "k", "Naive": "0.5", "Naive, equal sizes": "0.5"}


def _dist(Y):
    return np.linalg.norm(Y[:, None] - Y[None], axis=-1)


def _ratios(D):
    """(opposite, adjacent): the corner pairs' mean distance / the median distance between all pairs of cells"""
    med = np.median(D[np.triu_indices(len(D), 1)])
    return (np.mean([D[a, b] for a, b in OPPOSITE]) / med, np.mean([D[a, b] for a, b in ADJACENT]) / med)


def geometries():
    """per environment (goal-ordered cells): distances {geometry: (E, K, K)}, update sizes {Optimal, Naive: (E, K)},
    corner ratios {geometry or 'Naive, equal sizes': (E, 2)}"""
    meta = np.load(SCAN_META)
    D = {g: [] for g in GEOMS}
    size = {"Optimal": [], "Naive": []}
    rat = {g: [] for g in GEOMS + ("Naive, equal sizes",)}
    for e in ENVS:
        L = meta["L"][e] if meta["goal"][e] == 0 else meta["L"][e].transpose(0, 2, 1)   # (5, r_g, r_c)
        p = L.reshape(len(L), R * R).T                                                  # (K, 5)
        l = np.log(p) - np.log1p(-p)
        ell = lambda q: p @ np.log(q).T + (1 - p) @ np.log1p(-q).T                     # E[log q(o | .) | k]
        muJ = ell(p)                                                                    # (K, 100)
        muJ -= muJ.mean(1, keepdims=True)
        mg, mc = ell(L.mean(2).T), ell(L.mean(1).T)                                    # (K, 10) each
        muN = np.concatenate([mg - mg.mean(1, keepdims=True), mc - mc.mean(1, keepdims=True)], 1)
        d = {"evidence": np.sqrt(np.maximum(((p[:, None] - p[None]) * (l[:, None] - l[None])).sum(-1), 0)),
             "Optimal": _dist(muJ), "Naive": _dist(muN)}
        for g in GEOMS:
            D[g].append(d[g])
            rat[g].append(_ratios(d[g]))
        rat["Naive, equal sizes"].append(_ratios(_dist(muN / np.linalg.norm(muN, axis=1, keepdims=True))))
        size["Optimal"].append(np.linalg.norm(muJ, axis=1))
        size["Naive"].append(np.linalg.norm(muN, axis=1))
    return ({g: np.array(v) for g, v in D.items()}, {g: np.array(v) for g, v in size.items()},
            {g: np.array(v) for g, v in rat.items()})


def embed(D):
    """classical MDS (3 dimensions), rotated by orthogonal Procrustes onto the centred lattice -> (K, 3), the share of
    the positive eigenvalues these 3 dimensions hold"""
    n = len(D)
    C = np.eye(n) - 1 / n
    w, V = np.linalg.eigh(-.5 * C @ (D ** 2) @ C)
    w, V = w[::-1], V[:, ::-1]
    X = V[:, :3] * np.sqrt(np.maximum(w[:3], 0))
    gg, cc = np.meshgrid(np.arange(R), np.arange(R), indexing="ij")
    Y = np.stack([gg.ravel() - 4.5, cc.ravel() - 4.5, np.zeros(n)], 1)
    U, _, Vt = np.linalg.svd(X.T @ Y)
    return X @ (U @ Vt), w[:3].sum() / w[w > 0].sum()


def _letter(fig, x, y, s, FS):
    W, H = fig.get_size_inches()
    fig.text(x / W, y / H, r"$\mathbf{%s}$" % s, fontsize=FS["label"] + 2, va="bottom")


def _sheet(ax, X, col, FS, title):
    idx = {(g, c): 10 * g + c for g in range(R) for c in range(R)}
    edges = np.array([(idx[(g, c)], idx[(g + dg, c + dc)]) for (g, c) in idx for dg, dc in ((1, 0), (0, 1))
                      if (g + dg, c + dc) in idx])
    ax.add_collection3d(Line3DCollection(X[edges], colors="0.6", linewidths=.6))
    ax.scatter(*X.T, c=col, s=18, depthshade=False, lw=0)
    for k, mk in ((0, "o"), (99, "o"), (9, "s"), (90, "s")):
        ax.scatter(*X[k], s=90, marker=mk, facecolors="none", edgecolors="k", linewidths=1.3, depthshade=False)
    lo_, hi_ = X.min(0), X.max(0)                                                  # equal scale: the box takes the
    ax.set_xlim(lo_[0], hi_[0]), ax.set_ylim(lo_[1], hi_[1]), ax.set_zlim(lo_[2], hi_[2])   # sheet's own extents
    ax.set_box_aspect((hi_ - lo_) / (hi_ - lo_).max(), zoom=1.12)
    ax.view_init(elev=12, azim=-40)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.pane.set_facecolor((1, 1, 1, 0)), a_.pane.set_edgecolor("0.85")
    ax.set_title(title, fontsize=FS["label"], pad=-2)


def _corner_maps(fig, ax_, x0, y0, M, FS, cbar_label, S=1.9, gap=2.2):
    norm = LogNorm(np.percentile(M[M > 0], 1), M.max())
    for n_, corner in enumerate(CORNERS):
        b = ax_(x0 + n_ * gap, y0, S, S)
        k = corner[0] * R + corner[1]
        im = b.imshow(np.maximum(M[k].reshape(R, R).T, norm.vmin), origin="lower", cmap="viridis", norm=norm,
                      extent=(-.5, R - .5, -.5, R - .5), interpolation="nearest")
        b.scatter(*corner, s=60, marker="o", facecolors="none", edgecolors="w", linewidths=1.4)
        b.scatter(9 - corner[0], 9 - corner[1], s=60, marker="s", facecolors="none", edgecolors="#ff6f61",
                  linewidths=1.6)
        b.set_xticks([0, 9]), b.set_yticks([0, 9])
        b.set_xlabel("$r_g$", labelpad=-6)
        if n_ == 0:
            b.set_ylabel("$r_c$", labelpad=-6)
        b.set_title("from (%d, %d)" % corner, fontsize=FS["tick"] + .5, pad=3)
    cx = ax_(x0 + gap + S + .15, y0, .12, S)
    cb = fig.colorbar(im, cax=cx)
    ticks = [v for v in (.1, .2, .5, 1, 2, 5, 10, 20, 50) if norm.vmin <= v <= norm.vmax]
    cb.set_ticks(ticks), cb.set_ticklabels(["%g" % v for v in ticks]), cb.ax.minorticks_off()
    cb.set_label(cbar_label, fontsize=FS["tick"])
    cb.ax.tick_params(labelsize=FS["tick"] - 1)
    cb.outline.set_visible(False)


def _header(fig, ax_, x0, y, name, formula, FS, w=4.5):
    hb = ax_(x0, y, w, .01)
    hb.axis("off")
    hb.text(.5, 0, name + "\n" + formula, ha="center", va="bottom", fontsize=FS["label"], transform=hb.transAxes,
            linespacing=1.35)


def figure():
    import fig1 as cf
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    D, size, rat = geometries()
    p = ENVS.index(SHOW_ENV)
    gg, cc = np.meshgrid(np.arange(R), np.arange(R), indexing="ij")
    col = np.asarray(cf.bivariate(gg.ravel(), cc.ravel()))
    W, H = 15.0, 12.3
    fig = plt.figure(figsize=(W, H))
    ax_ = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    S, xb, MS = 3.15, 4.5, 1.9
    ys = [H - .62 - S - r * (S + .78) for r in range(3)]                       # rows' bottoms (bottom margin ~.6)
    rows = (("evidence", "a", "The evidence: a curled sheet", r"MDS of $\sqrt{J}$",
             "How different two cells' observations are",
             r"$\sqrt{J(k, k')}$, $J = \Sigma_i\,(p_{ik} - p_{ik'})(\ell_{ik} - \ell_{ik'})$; "
             "square: the opposite corner",
             r"$\sqrt{J}$, median over environments"),
            ("Optimal", "c", "The Optimal's updates: curled like the evidence", r"MDS of $\|\mu_k - \mu_{k'}\|$",
             "How differently two cells move the Optimal's belief",
             r"$\|\mu_k - \mu_{k'}\|$, $\mu_k(j) = E[\log p(o \mid j) \mid k]$ over the 100 cells $j$, centred",
             "nats per step, median"),
            ("Naive", "e", "The Naive's updates: pinched, not curled", r"MDS of $\|\mu_k - \mu_{k'}\|$",
             "How differently two cells move the Naive's beliefs",
             r"$\|\mu_k - \mu_{k'}\|$, $\mu_k(r) = E[\log q(o \mid r) \mid k]$, each belief centred",
             "nats per step, median"))
    for r_, (g, letter, title, how, name, formula, cbl) in enumerate(rows):
        y = ys[r_]
        X, share = embed(D[g][p])
        ax = fig.add_axes([.5 / W, y / H, S / W, S / H], projection="3d")
        _sheet(ax, X, col, FS, "%s\n%s, environment %d (%d%% in 3 dims)" % (title, how, SHOW_ENV, round(100 * share)))
        _letter(fig, .1, y + S - .1, letter, FS)
        _corner_maps(fig, ax_, xb, y + .45, np.median(D[g], 0), FS, cbl, S=MS)
        _header(fig, ax_, xb - .1, y + .45 + MS + .42, name, formula, FS, w=4.35)
        _letter(fig, xb - .55, y + .45 + MS + .63, chr(ord(letter) + 1), FS)
    # g: opposite against adjacent corners, per environment
    ex, ew = 10.55, 3.7
    gy0, gy1 = ys[1] + .45, ys[0] + .45 + MS
    ax = ax_(ex, gy0, ew, gy1 - gy0)
    ax.axvline(1, color="0.6", lw=.9, ls=":"), ax.axhline(1, color="0.6", lw=.9, ls=":")
    for g in GEOMS + ("Naive, equal sizes",):
        v = rat[g]
        open_ = g == "Naive, equal sizes"
        ax.scatter(*v.T, s=16, facecolors="none" if open_ else COLOR[g], edgecolors=COLOR[g], linewidths=.8,
                   alpha=.85, zorder=2)
        ax.scatter(*np.median(v, 0), s=150, facecolors="white" if open_ else COLOR[g], edgecolors="k", linewidths=1.4,
                   zorder=4, label=g)
    ax.set_xlim(.2, 1.35), ax.set_ylim(.2, 2.45)
    ax.set_xlabel("opposite corners' distance / median pair", fontsize=FS["tick"] + .5)
    ax.set_ylabel("adjacent corners' distance / median pair", fontsize=FS["tick"] + .5)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(.23, 2.38, "curled: only the\nopposite corners meet", fontsize=FS["tick"] + .5, va="top", color="0.25")
    ax.text(.23, .23, "pinched: all four\ncorners crowd", fontsize=FS["tick"] + .5, va="bottom", color="0.25")
    ax.legend(loc="lower right", fontsize=FS["tick"], frameon=False, handletextpad=.3, borderaxespad=.2,
              markerscale=.6)
    _header(fig, ax_, ex - .2, ys[0] + .45 + MS + .42, "Curled or pinched",
            "per environment (large: median); dotted: a typical pair", FS, w=ew + .4)
    _letter(fig, ex - .75, ys[0] + .45 + MS + .63, "g", FS)
    # h: the size of each cell's expected update
    hs, hgap = 1.55, 2.12
    for n_, g in enumerate(("Optimal", "Naive")):
        b = ax_(ex + n_ * hgap, ys[2] + .45, hs, hs)
        Mz = np.median(size[g], 0).reshape(R, R).T
        im = b.imshow(Mz, origin="lower", cmap="viridis", extent=(-.5, R - .5, -.5, R - .5), interpolation="nearest")
        b.set_xticks([0, 9]), b.set_yticks([0, 9])
        b.set_xlabel("$r_g$", labelpad=-6)
        if n_ == 0:
            b.set_ylabel("$r_c$", labelpad=-6)
        b.set_title(g, fontsize=FS["tick"] + .5, pad=3)
        cx = ax_(ex + n_ * hgap + hs + .07, ys[2] + .45, .09, hs)
        cb = fig.colorbar(im, cax=cx)
        cb.ax.tick_params(labelsize=FS["tick"] - 1)
        cb.locator = plt.MaxNLocator(3)
        cb.update_ticks()
        cb.outline.set_visible(False)
    _header(fig, ax_, ex - .2, ys[2] + .45 + hs + .42, "How much each cell moves the beliefs",
            r"$\|\mu_k\|$, nats per step, median over environments", FS, w=ew + .4)
    _letter(fig, ex - .75, ys[2] + .45 + hs + .63, "h", FS)
    path = fig_path("fig1c_evidence_geometry.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    for g, v in rat.items():
        print(f"{g:20s} opposite {np.median(v[:, 0]):.2f} adjacent {np.median(v[:, 1]):.2f}")
    return path


if __name__ == "__main__":
    print("saved", figure())
