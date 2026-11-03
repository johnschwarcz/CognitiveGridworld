"""The Optimal crossing a wormhole: its joint and its two marginals on single streams, and what the marginals show of a
crossing (28 Oct 2026; John: "let's stick to trajectories of just the optimal joint and its marginals. Maybe
trajectories where we see splitting occur ... right now the plot is much too busy ... So far it looks like marginal
beliefs cannot cross through the wormhole. I wonder if we can identify a property of the marginals that help us identify
that the joint is crossing the wormhole?"; a first, five-agent version answered "what is this 3 dimensional space ...
are we able to visualize individual marginal belief trajectories on this wrapped manifold?")
-> main/REPLICATION/figure_bin/fig1c_figure8_traj.png        figure(): one crossing of each bridge
   main/REPLICATION/figure_bin/fig1c_wormhole_marginals.png  signature(): the marginals' signature, all beliefs
   main/REPLICATION/figure_bin/fig1c_marginal_cycles.png     cycles(): coupled vs independent marginal cycles, 4 agents

The space: the cells' common evidence geometry (plot_fig1c_figure8_wrap.pooled): for each environment the
sqrt-Jeffreys distance between every two cells' observation distributions, averaged over the 39 environments, embedded
by classical MDS in 3-D (88%): one fixed position Y_k per cell k. A belief p sits at sum_k p_k Y_k, a fixed linear
read-out of the joint belief (not the beliefs' own principal components, not (r_g, r_c, Cov)). The wrap: every belief of
the Optimal from step 8 on, the places visited >= 30 times thickened by 0.25 lattice spacings, coloured by the mean
Cov(r_g, r_c) nearby (plot_fig1c_figure8_wrap). Two marginals are placed through their product,
sum_gc b_g(g) b_c(c) Y_gc. A belief split between two opposite corners has the same marginals whichever pair it is
(half at 0 and half at 9 for both variables), so recombined it puts 1/4 on each of the four corners and sits at their
middle (x, near cells (4, 4) and (4, 5)), never on a bridge. Off the manifold? Into a thinly visited region, not
empty space: at the Optimal's wormhole steps (|Cov| > 15) a median 104 of its beliefs (steps 8 on) lie within the wrap's
thickness (0.25 spacings) of the recombined position, against 12,350 around a typical belief and 287 around the joint on
its bridge; the beliefs found there are broad and unsplit (7.2 effective cells vs 3.2 overall; |Cov| 1.2), their means
0.8 cells from the grid's centre: recombined, a crossing reads as a broad belief at the centre.
figure(): rows: a stream whose Optimal belief crosses the red (0, 0) - (9, 9) bridge, one crossing the blue (0, 9) -
(9, 0) bridge (pick(): a clean crossing, settled in one corner, one short split, settled in the opposite corner; the
largest peak |Cov| over the 39 environments); each row seen face-on to its loop (the plane through its diagonal's
cells, flat to 1%; VIEWS); the two loops' planes are nearly perpendicular, so no one view shows both (the wrap figure's
view sees the blue loop 73 deg from face-on). Columns: the joint on the wrap; its two
marginals recombined; the two marginals over the steps (bright = more), and the joint's Cov with the bound the
marginals set, +-sqrt(Var_g Var_c) (|Cov| can never exceed it). Stars: the steps with |Cov| > 15. Triangles: both
marginals cycle (each moves >= CYCLE = 0.25 of its mass between its two ends, values 0 - 1 and 8 - 9, from one step to
the next; flow()): they also mark the blue example's weaker crossing at step 6 (flows -0.70 / +0.32, Cov -10.8), which
|Cov| > 15 misses (John: "an undetected wormhole crossing ... very clear in the marginals").
signature(): all 5,850,000 beliefs of the Optimal in the scan (39 environments x 5,000 streams x 30 steps). A crossing
shows in the marginals: (1) both spread to their two ends at once: the bound sqrt(Var_g Var_c) reaches its maximum
20.25 only for two 50 / 50 splits between 0 and 9; it separates the 9,379 wormhole states (|Cov| > 15) from all other
beliefs perfectly (AUC 1.000); of the beliefs whose bound exceeds 15, 84% are wormhole states (89% from step 4, 94% from
step 8), their |Cov| a median 0.985 of the bound; (2) which bridge: the marginal means lie on the main diagonal for the
red bridge and on the other diagonal for the blue (median 0.15 cells off), so the sign of (E[r_g] - 4.5)(E[r_c] - 4.5)
names the bridge in 88% of wormhole states: 100% when the means are >= 1 cell from the centre, 82% at 0.5 - 1, 67% at
0.25 - 0.5, 49% (chance) within 0.25 cells, where the two bridges' marginals coincide.
cycles() (John: "can we identify crossings by locating where the marginal belief cycles? Do the ESN / naive marginal
not cycle like this? Or does the cycling only happen independentally?"): per step, the mass each marginal moves between
its ends, r_g against r_c, every stream-step of the scan. Cycles (>= 0.25) per 1,000 stream-steps, r_g / r_c, and the
share with the other variable cycling at the same step: Optimal 2.1 / 2.1, 46%; Trained RNN 2.3 / 2.3, 40%; Naive
3.6 / 3.1, 1%; ESN 1.6 / 1.0, 1% (at >= 0.5: 35%, 28%, 0%, 0%). The Naive's and the ESN's marginals cycle, the Naive's
more often than the Optimal's, but one variable at a time. The Optimal's 5,528 coupled cycles are crossings: its joint
|Cov| at that step exceeds 8 in 92% (15: 54%), and their directions name the bridge (same: Cov > 0, opposite: Cov < 0)
in 100.0%.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_figure8_traj.py [figure|signature|cycles]"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from scipy import ndimage                                                         # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection                           # noqa: E402
import shrinkwrap as sw                                                           # noqa: E402
import plot_fig1c_figure8 as f8                                                   # noqa: E402
import plot_fig1c_figure8_wrap as fw                                              # noqa: E402
import plot_fig1c_measures_movie as mm                                            # noqa: E402

R, T, VISITS = 10, 30, 30
RED, BLUE = "#b2182b", "#2166ac"
CYCLE = .25                                                                       # a cycle: >= this much mass moved
VIEWS = {1: (-11, 136), -1: (-7, 44)}                                             # each loop seen face-on
BLOCKS = {0: [(0, 0), (0, 1), (1, 0), (1, 1)], 99: [(9, 9), (9, 8), (8, 9), (8, 8)],
          9: [(0, 9), (0, 8), (1, 9), (1, 8)], 90: [(9, 0), (9, 1), (8, 0), (8, 1)]}


def surface(Z, m=VISITS, k=fw.FIG_THICK, alpha=.16):
    """the wrap's triangles and colours (as plot_fig1c_figure8_wrap.figure), the lattice spacing"""
    from scipy.spatial import cKDTree
    Y = Z["Y"]
    P = Z["X"][:, :, fw.FIG_T0 - 1:].reshape(-1, 3).astype(np.float64)
    CV = Z["CV"][:, :, fw.FIG_T0 - 1:].ravel()
    _, nn = fw.lattice_sheet(Y)
    step = fw.FIG_GRID
    lo, hi = Y.min(0) - 1., Y.max(0) + 1.
    n = np.ceil((hi - lo) / step).astype(int)
    idx = np.clip(np.floor((P - lo) / step).astype(int), 0, n - 1)
    C = np.bincount(np.ravel_multi_index(idx.T, n), minlength=int(np.prod(n))).reshape(n)
    D = ndimage.distance_transform_edt(~(C >= m)) * step
    rho = k * nn
    Tr = lo + step * (sw.march(-D, -rho) + .5)
    d, j = cKDTree(P).query(Tr.mean(1), k=64, distance_upper_bound=1.5 * rho)
    ok = np.isfinite(d)
    cov = np.where(ok, CV[np.minimum(j, len(CV) - 1)], 0.).sum(1) / np.maximum(ok.sum(1), 1)
    col = plt.get_cmap("RdBu_r")((np.clip(cov, -15, 15) + 15) / 30)
    light = np.array([.3, -.5, .8]) / np.linalg.norm([.3, -.5, .8])
    nrm = np.cross(Tr[:, 1] - Tr[:, 0], Tr[:, 2] - Tr[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    col[:, :3] *= (.6 + .4 * np.abs(nrm @ light))[:, None]
    col[:, 3] = alpha
    return Tr, col, nn


def pick():
    """the clearest clean crossing of each bridge -> [(environment, stream, sign, joint (T, 100), Cov (T,))], red
    first. Clean: the belief holds >= 0.85 in one corner's 2 x 2 block at >= 3 steps, then |Cov| > 15 (the bridge's
    sign) in one run of <= 3 steps, then >= 0.85 in the opposite corner's block at >= 3 steps and >= 0.9 at step 30;
    clearest: the largest peak |Cov| over the 39 environments"""
    best = {}
    for e in f8.ENVS:
        p, cov = f8.posterior(e)
        P = p.reshape(*p.shape[:2], R, R)
        mass = {k: sum(P[..., a, b] for a, b in v) for k, v in BLOCKS.items()}
        for sign, ends in ((1, (0, 99)), (-1, (9, 90))):
            on = sign * cov > 15
            for i in np.flatnonzero(on.any(1) & ((mass[ends[0]][:, -1] >= .9) | (mass[ends[1]][:, -1] >= .9))):
                steps = np.flatnonzero(on[i])
                if steps[-1] - steps[0] + 1 != len(steps) or len(steps) > 3:
                    continue
                peak = float(np.max(sign * cov[i]))
                for a, b in (ends, ends[::-1]):
                    if (mass[a][i, :steps[0]] >= .85).sum() >= 3 and (mass[b][i, steps[-1] + 1:] >= .85).sum() >= 3 \
                            and mass[b][i, -1] >= .9 and peak > best.get(sign, (0,))[0]:
                        best[sign] = (peak, (e, int(i), sign, p[i].copy(), cov[i].copy()))
    return [best[1][1], best[-1][1]]


def _marginals(p):
    P = p.reshape(T, R, R)
    return P.sum(2), P.sum(1)


def flow(b):
    """mass moved between a marginal's two ends from each step to the next (..., T - 1): + low {0, 1} -> high {8, 9},
    - high -> low; the mass that left one end and arrived at the other, min(drop at one end, rise at the other)"""
    lo, hi = b[..., :2].sum(-1), b[..., -2:].sum(-1)
    dlo, dhi = np.diff(lo, axis=-1), np.diff(hi, axis=-1)
    up, dn = np.minimum(-dlo, dhi), np.minimum(dlo, -dhi)
    return np.where(up > 0, up, 0) - np.where(dn > 0, dn, 0)


def figure():
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    Z = fw.pooled()
    Y = Z["Y"]
    Ym = Y.reshape(R, R, 3)
    Tr, col, nn = surface(Z)
    streams = pick()
    mid4 = Y[[0, 9, 90, 99]].mean(0)
    r = np.arange(R)
    S, TW = 4.3, 4.4
    W, H = 2 * S + TW + 1.25, 2 * S + 1.0
    fig = plt.figure(figsize=(W, H))
    fig.text(.5, 1 - .2 / H, "The Optimal crossing a wormhole: its joint posterior and its two marginals (alias scan, "
             "one stream per row); positions: the cells' evidence geometry, a belief at its probabilities' average "
             r"of the cells' positions", ha="center", va="center", fontsize=FS["label"])
    fig.text(.5, 1 - .45 / H, r"open circle: step 1, square: step 30, stars: steps with $|$Cov$|$ above 15; x: the "
             r"four corners' middle; right: triangles, both marginals cycle (each moves $\geq$ 0.25 of its mass "
             r"between its ends); grey band, $\pm\sqrt{Var_g\, Var_c}$; line, the joint's Cov", ha="center",
             va="center", fontsize=FS["label"])
    lo_, hi_ = Y.min(0) - .4, Y.max(0) + .4
    for row, (e, s, sign, p, cov) in enumerate(streams):
        bg, bc = _marginals(p)
        Xj = p @ Y
        Xm = np.einsum("tg,gcd,tc->td", bg, Ym, bc)
        cross = np.flatnonzero(np.abs(cov) > 15)
        fg, fc = flow(bg), flow(bc)
        cyc = np.flatnonzero((np.abs(fg) >= CYCLE) & (np.abs(fc) >= CYCLE)) + 1      # step t (flow t-1 -> t)
        c_br = RED if sign > 0 else BLUE
        y0 = H - .75 - (row + 1) * S
        for k, (X, title) in enumerate(((Xj, "joint posterior"), (Xm, "its two marginals, recombined"))):
            ax = fig.add_axes([(.45 + k * S) / W, y0 / H, S / W, S / H], projection="3d")
            ax.computed_zorder = False
            ax.set_proj_type("ortho")
            ax.view_init(*VIEWS[sign])
            ax.set_xlim(lo_[0], hi_[0]), ax.set_ylim(lo_[1], hi_[1]), ax.set_zlim(lo_[2], hi_[2])
            ax.set_box_aspect(hi_ - lo_, zoom=1.45)
            ax.set_axis_off()
            ax.add_collection3d(Poly3DCollection(Tr, facecolors=col, edgecolors="none", zorder=1))
            for kk, lab in ((0, "(0, 0)"), (99, "(9, 9)"), (9, "(0, 9)"), (90, "(9, 0)")):
                v3 = Y[kk] + .45 * (Y[kk] - Y.mean(0)) / np.linalg.norm(Y[kk] - Y.mean(0))
                ax.text(*v3, lab, fontsize=FS["tick"], ha="center", va="center", zorder=6)
            ax.scatter(*mid4, marker="x", s=45, color="k", lw=1.3, depthshade=False, zorder=7)
            ax.plot(*X.T, color="k", lw=1.2, zorder=5)
            ax.scatter(*X.T, color="k", s=7, depthshade=False, zorder=5)
            ax.scatter(*X[0], facecolor="white", edgecolor="k", s=36, lw=1.2, depthshade=False, zorder=6)
            ax.scatter(*X[-1], marker="s", color="k", s=36, depthshade=False, zorder=6)
            ax.scatter(*X[cross].T, marker="*", color=c_br, edgecolor="k", lw=.5, s=130, depthshade=False, zorder=7)
            ax.set_title(title, fontsize=FS["label"] + 1, pad=-12)
        fig.text(.32 / W, (y0 + S / 2) / H, f"crossing the {'red' if sign > 0 else 'blue'} bridge\n(environment {e}, "
                 f"stream {s})", rotation=90, ha="center", va="center", fontsize=FS["label"], color=c_br)
        x0, hp = 2 * S + .9, (S - 1.05) / 3
        for j, (b, lab) in enumerate(((bg, r"$b_g$: marginal of $r_g$"), (bc, r"$b_c$: marginal of $r_c$"))):
            ax = fig.add_axes([x0 / W, (y0 + .45 + (2 - j) * (hp + .12)) / H, TW / W, hp / H])
            ax.imshow(b.T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest",
                      extent=(.5, T + .5, -.5, 9.5))
            ax.set_yticks([0, 9]), ax.set_xticks([])
            ax.tick_params(labelsize=FS["tick"], length=2, pad=1)
            ax.set_ylabel(lab, fontsize=FS["tick"] + 1)
            for t in cross:
                ax.axvline(t + 1, color="w", lw=.6, ls=":")
            ax.scatter(cyc + 1, np.full(len(cyc), 9.9), marker="v", s=46, color="w", edgecolor="k", lw=.6,
                       clip_on=False, zorder=5)
        ax = fig.add_axes([x0 / W, (y0 + .45) / H, TW / W, hp / H])
        B = np.sqrt(np.maximum((bg @ r ** 2 - (bg @ r) ** 2) * (bc @ r ** 2 - (bc @ r) ** 2), 0))
        t = np.arange(1, T + 1)
        ax.fill_between(t, -B, B, color="0.85", lw=0)
        ax.plot(t, cov, color=c_br, lw=1.8)
        ax.axhline(0, color="0.5", lw=.6)
        ax.set_xlim(.5, T + .5), ax.set_ylim(-21, 21), ax.set_yticks([-20, 0, 20])
        ax.set_xlabel("step", fontsize=FS["tick"] + 1, labelpad=1)
        ax.tick_params(labelsize=FS["tick"], length=2, pad=1)
        print(f"coupled cycles at steps {', '.join(str(int(x) + 1) for x in cyc)} (flows "
              + ", ".join(f"{fg[x - 1]:+.2f}/{fc[x - 1]:+.2f}" for x in cyc) + ")")
        print(f"{'red' if sign > 0 else 'blue'}: environment {e}, stream {s}: |Cov| > 15 at steps "
              f"{', '.join(str(int(x) + 1) for x in cross)}; peak {cov[np.argmax(np.abs(cov))]:+.1f}; recombined "
              f"marginals {np.linalg.norm(Xm[cross] - Xj[cross], axis=1).mean() / nn:.2f} spacings from the joint "
              f"there")
    path = fig_path("fig1c_figure8_traj.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def signature():
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    r = np.arange(R, dtype=np.float64)
    CV = fw.pooled()["CV"].astype(np.float64)
    Bm = np.load(os.path.join(mm.SCAN, "scan_Joint.npz"))["beliefs"][mm.ENVS].astype(np.float64)
    bg = Bm[..., :R] / Bm[..., :R].sum(-1, keepdims=True)
    bc = Bm[..., R:] / Bm[..., R:].sum(-1, keepdims=True)
    del Bm
    mg, mc = bg @ r, bc @ r
    bound = np.sqrt(np.maximum((bg @ r ** 2 - mg ** 2) * (bc @ r ** 2 - mc ** 2), 0)).ravel()
    cv, mg, mc = CV.ravel(), mg.ravel(), mc.ravel()
    worm = np.abs(cv) > 15
    W, H = 15.6, 5.3
    fig = plt.figure(figsize=(W, H))
    fig.text(.5, 1 - .25 / H, f"What the two marginals show of a wormhole crossing: every belief of the Optimal in the "
             f"scan ({cv.size:,} = 39 environments x 5,000 streams x 30 steps); wormhole states: $|$Cov$|$ above 15 "
             f"({worm.sum():,})", ha="center", va="center", fontsize=FS["label"])
    ax = fig.add_axes([.7 / W, .75 / H, 4.2 / W, 3.7 / H])
    h, xe, ye = np.histogram2d(bound, cv, bins=[np.linspace(0, 20.5, 83), np.linspace(-20.5, 20.5, 83)])
    im = ax.imshow(np.log10(np.where(h > 0, h, np.nan)).T, origin="lower", aspect="auto", cmap="viridis",
                   extent=(xe[0], xe[-1], ye[0], ye[-1]), interpolation="nearest")
    ax.plot([0, 20.25], [0, 20.25], color="w", lw=.8, ls="--")
    ax.plot([0, 20.25], [0, -20.25], color="w", lw=.8, ls="--")
    ax.set_xlabel(r"the marginals' bound $\sqrt{Var_g\, Var_c}$", fontsize=FS["label"])
    ax.set_ylabel("the joint's Cov$(r_g, r_c)$", fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    big = bound > 15
    ax.set_title(f"(1) a crossing: both marginals spread to both ends\nbound above 15: {100 * worm[big].mean():.0f}% "
                 f"are wormhole states; dashed: $|$Cov$|$ = bound", fontsize=FS["label"], pad=5)
    cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.02)
    cb.set_label("log$_{10}$ beliefs", fontsize=FS["tick"])
    cb.ax.tick_params(labelsize=FS["tick"] - 1)
    ax = fig.add_axes([6.0 / W, .75 / H, 3.7 / W, 3.7 / H])
    o = np.random.default_rng(0).permutation(np.flatnonzero(worm))
    ax.scatter(mg[o], mc[o], s=2, lw=0, c=np.where(cv[o] > 0, RED, BLUE), alpha=.5)
    ax.plot([0, 9], [0, 9], color="0.4", lw=.6, ls=":"), ax.plot([0, 9], [9, 0], color="0.4", lw=.6, ls=":")
    ax.set_xlim(-.3, 9.3), ax.set_ylim(-.3, 9.3), ax.set_aspect("equal")
    ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
    ax.set_xlabel(r"$E[r_g]$", fontsize=FS["label"], labelpad=-6), ax.set_ylabel(r"$E[r_c]$", fontsize=FS["label"],
                                                                                  labelpad=-6)
    ax.tick_params(labelsize=FS["tick"])
    ax.set_title("(2) which bridge: the marginal means of the\nwormhole states (red: Cov $>$ 0, blue: Cov $<$ 0)",
                 fontsize=FS["label"], pad=5)
    ax = fig.add_axes([10.9 / W, .75 / H, 4.3 / W, 3.7 / H])
    d = np.hypot(mg - 4.5, mc - 4.5)[worm]
    agree = (np.sign((mg - 4.5) * (mc - 4.5)) == np.sign(cv))[worm]
    edges = np.array([0, .25, .5, .75, 1, 1.5, 2, 3, 7])
    k = np.digitize(d, edges) - 1
    share = [100 * agree[k == i].mean() for i in range(len(edges) - 1)]
    cnt = [(k == i).sum() for i in range(len(edges) - 1)]
    mid = np.minimum((edges[:-1] + edges[1:]) / 2, 3.5)
    ax.plot(mid, share, color="k", lw=1.6, marker="o", ms=4)
    for x_, y_, n_ in zip(mid, share, cnt):
        ax.text(x_, y_ - 6, f"{n_:,}", ha="center", va="top", fontsize=FS["tick"] - 1, color="0.4")
    ax.axhline(50, color="0.5", lw=.8, ls=":")
    ax.set_ylim(30, 104), ax.set_xlim(0, 3.8)
    ax.set_xlabel("distance of the marginal means from the centre (cells)", fontsize=FS["label"])
    ax.set_ylabel("% where the means' diagonal names the bridge", fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"sign of $(E[r_g] - 4.5)(E[r_c] - 4.5)$: {100 * agree.mean():.0f}% overall;\nchance at the centre, "
                 "where both bridges' marginals coincide (grey: counts)", fontsize=FS["label"], pad=5)
    path = fig_path("fig1c_wormhole_marginals.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"bound > 15: {big.sum():,}, wormhole share {100 * worm[big].mean():.1f}%; diagonal rule "
          f"{100 * agree.mean():.1f}%; by distance: " + ", ".join(f"{a:g}-{b:g}: {s_:.0f}% ({n_:,})" for a, b, s_, n_ in
                                       zip(edges[:-1], edges[1:], share, cnt)))
    return path


def cycles(level=CYCLE):
    """the marginals' cycles for the four agents -> fig1c_marginal_cycles.png: per step, the mass each marginal moved
    between its two ends (flow), r_g against r_c, over every stream-step of the scan; coupled cycles lie on the
    diagonals (same direction: along (0, 0) - (9, 9); opposite: along (0, 9) - (9, 0)), independent ones on the axes"""
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    CV = fw.pooled()["CV"]
    names = (("Joint", "Optimal"), ("Trained", "Trained RNN"), ("Naive", "Naive"), ("Echo", "ESN"))
    PW, GX = 3.3, .75
    W, H = .7 + 4 * PW + 3 * GX + 1.0, PW + 2.0
    fig = plt.figure(figsize=(W, H))
    edges = np.linspace(-1, 1, 81)
    for i, (agent, name) in enumerate(names):
        B = np.load(os.path.join(mm.SCAN, f"scan_{agent}.npz"))["beliefs"][mm.ENVS].astype(np.float32)
        Wg = flow(B[..., :R] / B[..., :R].sum(-1, keepdims=True))
        Wc = flow(B[..., R:] / B[..., R:].sum(-1, keepdims=True))
        del B
        g, c = np.abs(Wg) >= level, np.abs(Wc) >= level
        both = g & c
        same = both & (np.sign(Wg) == np.sign(Wc))
        moving = np.maximum(np.abs(Wg), np.abs(Wc)) >= .05
        h = np.histogram2d(Wg[moving], Wc[moving], bins=[edges, edges])[0]
        ax = fig.add_axes([(.7 + i * (PW + GX)) / W, .75 / H, PW / W, PW / H])
        im = ax.imshow(np.log10(np.where(h > 0, h, np.nan)).T, origin="lower", cmap="viridis", vmin=0, vmax=4.5,
                       extent=(-1, 1, -1, 1), interpolation="nearest")
        ax.plot([-1, 1], [-1, 1], color="0.5", lw=.5, ls=":"), ax.plot([-1, 1], [1, -1], color="0.5", lw=.5, ls=":")
        ax.set_xticks([-1, 0, 1]), ax.set_yticks([-1, 0, 1])
        ax.tick_params(labelsize=FS["tick"])
        ax.set_xlabel(r"$r_g$: mass moved low $\rightarrow$ high ($-$: high $\rightarrow$ low)",
                      fontsize=FS["tick"] + 1)
        if i == 0:
            ax.set_ylabel(r"$r_c$: mass moved between its ends", fontsize=FS["tick"] + 1)
        ax.set_title(f"{name}\ncycles ($\\geq$ {level:g}) per 1,000 stream-steps: $r_g$ {1000 * g.mean():.1f}, "
                     f"$r_c$ {1000 * c.mean():.1f}\nboth at once: {100 * both.sum() / max(g.sum(), 1):.0f}% of "
                     "$r_g$'s cycles", fontsize=FS["label"], pad=5)
        if agent == "Joint":
            cv = np.where(np.abs(CV[..., 1:]) > np.abs(CV[..., :-1]), CV[..., 1:], CV[..., :-1])[both]
            ok = np.mean(np.sign(cv) == np.sign(Wg * Wc)[both])
            note = (f"Optimal: at its {both.sum():,} coupled cycles the joint's $|$Cov$|$ exceeds 8 in "
                    f"{100 * np.mean(np.abs(cv) > 8):.0f}% (15: {100 * np.mean(np.abs(cv) > 15):.0f}%); the "
                    f"directions name the bridge (same: Cov $>$ 0, opposite: Cov $<$ 0) in {100 * ok:.1f}%")
        print(f"{name}: r_g cycles {g.sum():,}, r_c {c.sum():,}; both {both.sum():,} "
              f"({100 * both.sum() / max(g.sum(), 1):.0f}% / {100 * both.sum() / max(c.sum(), 1):.0f}%); same "
              f"direction {100 * same.sum() / max(both.sum(), 1):.0f}%")
    cax = fig.add_axes([(.7 + 4 * PW + 3 * GX + .2) / W, .75 / H, .13 / W, PW / H])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("log$_{10}$ stream-steps", fontsize=FS["tick"])
    cb.ax.tick_params(labelsize=FS["tick"] - 1)
    fig.text(.5, 1 - .2 / H, "Marginal cycles: per step, the mass each marginal moves between its two ends (values "
             "0 - 1 and 8 - 9), $r_g$ against $r_c$; every stream-step of the scan (39 environments x 5,000 streams x "
             "29 steps; moving steps shown)", ha="center", va="center", fontsize=FS["label"])
    fig.text(.5, .22 / H, note + ". Diagonals: both marginals cycle at once (coupled); axes: one alone.",
             ha="center", va="center", fontsize=FS["label"])
    path = fig_path("fig1c_marginal_cycles.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "figure"
    print("saved", {"figure": figure, "signature": signature, "cycles": cycles}[arg]())
