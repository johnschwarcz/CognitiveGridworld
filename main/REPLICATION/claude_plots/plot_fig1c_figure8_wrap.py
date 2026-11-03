"""The figure 8 shrink-wrapped from the Optimal's joint posteriors, pooled over all environments, with no rule
(28 Oct 2026; John: "shrinkwrap ... a topology with 2 holes for the optimal bayes as our figure 8 suggests?"; "what is a
split state? I want to make sure we are being principled"; "Would that give us the 8 manifold for free? ... average
over environments"; "fix the broad beliefs by looking only at data after some timepoint ... I'm not a fan of
handcrafted thresholds. I would want to see how this changes with different thresholds"; "color the sheet by the
average velocity")
-> main/REPLICATION/figure_bin/fig1c_figure8_wrap.png        figure(): the rule-free wrap
   main/REPLICATION/figure_bin/fig1c_figure8_thresholds.png  figure_sweep(): its topology over every threshold

The space: the evidence geometry common to the 39 environments (their evidence distance matrices averaged, classical
MDS, 3-D holds 88%; opposite corners 0.56 / 0.54 of the median pair apart, adjacent ones 1.54). Each belief, the
Optimal's exact JOINT posterior over the 100 cells at one stream-step, sits at its posterior-weighted average of the
cells' positions (pooled(): 39 x 5,000 streams x 30 steps, cached in POOLED).
figure(): every belief from step FIG_T0 on (4.49 M; no filter, no sheet, no corner rule), the places (grid FIG_GRID)
visited by >= m of them, thickened by FIG_THICK = 0.25 lattice spacings (John 10-28: "stick with the 0.25 spacing"; one
view, "only need the 2nd column angle"), for m = 10, 20, 30, 40, 50, 60. A hole count alone does not say which loops
the holes are, so each panel also gets the bridge test (bridges()): the voxels whose beliefs' mean Cov is beyond +-5
(each wrapped voxel takes the mean of the nearest visited voxel) form the red and the blue bridge; the 8 = 1 piece, 2
holes, 0 cavities, each bridge one piece touching both its corners, and removing either bridge leaves exactly 1 hole
(the other's). Measured: the 8 for m = 20 - 40 (from step 12: 15 - 30); m = 50: still 2 holes, but the red bridge is
broken and both holes run through the blue region; m = 60: 0 holes, both broken; m = 10: 4 holes, 2 cavities (thinly
visited places thickened). Cuts 5 and 8 agree at every m; at 3 the removed region carves into the sheet and changes the
count. Colour: each surface point's mean
Cov(r_g, r_c) = sum_gc p(g, c) g c - E[r_g] E[r_c], the covariance of the two variables under ONE belief (one stream's
joint posterior at one step; 0 for a belief on one cell, +20.25 for half on (0, 0) and half on (9, 9), -20.25 for
(0, 9) / (9, 0)), averaged over the beliefs within 1.5 thicknesses of the point (the 64 nearest at most): the
wormholes show up red and blue by themselves. (The 3 x 3 grid of 10-28, thickness 0.25 / 0.45 / 0.8 x >= 10 / 30 / 100
visits: 2 holes at >= 30 for all three; hole counts only, as figure_sweep below: the bridge test came after the >= 50
case at 0.25.)
figure_sweep() (sweep_t0 per start step, cached as round7_jumps/figure8_sweep_t<t0>.npz): holes for start steps
SWEEP_T0 x minimum visits SWEEP_M x thicknesses SWEEP_RHO, for all beliefs vs beliefs on about 2 cells or fewer
(1 / sum p^2 <= NEFF), data only vs + the interpolated sheet. Measured 28 Oct: from step 1 the broad early beliefs fill
the curl (0 holes, or foam at thin wraps); from step 8 or 12, all beliefs, >= 30 visits: 2 holes at every thickness
0.25 - 1.0 (from step 4: at 5 of 6; from step 16: 4 of 6, one wormhole thinning); >= 1 - 3 visits: rare paths fill
or fragment it; >= 100: too strict (the wormholes carry fewer); the sheet scaffold changes almost nothing (the data
cover it). The off-sheet connections at thickness 0.45 (each cluster's commonest pair of cells, from the beliefs' top
two cells): from step 8, >= 30 visits: (0, 9) / (9, 0) (99%), (0, 0) / (9, 9) (99%, 100%) and one 2-voxel cluster of
the neighbouring border cells (7, 9) / (8, 9); at lower counts more clusters, nearly all neighbouring border cells (the
curled sheet's edge), which cannot make a hole.
Earlier constructions (numbers_pooled, build / numbers / all_envs for single environments; kept as checks): the sheet
plus beliefs on <= 2 cells visited >= MIN_VISITS times: 2 holes at every thickness, all 41 off-sheet voxels in the two
opposite-corner gaps; the corner rule (>= TAU of the posterior in each of two opposite corners' 2 x 2 blocks, >= KAPPA
in both): 2 holes pooled; the Optimal's splits: opposite corners 12,042 vs adjacent corners 1,374, four corners 4;
adjacent-corner splits never add a hole (environments 10, 11); single environments: 2 holes in 14 of the 28 whose
sheet wraps cleanly, 1 where a wormhole is barely used. The step arches in (E[r_g], E[r_c], delta) wrap to a solid
piece (0 holes) at every level. Speed: steps through opposite-corner splits median 0.26 evidence units, other steps
0.067, between settled beliefs 0.008. shrinkwrap.py: exact hole counts on voxels + marching tetrahedra.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/plot_fig1c_figure8_wrap.py
        (the figure, with numbers_pooled); ... thresholds (the sweep figure); ... sweep <t0> (one start step)"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from scipy import ndimage                                                         # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection                           # noqa: E402
import shrinkwrap as sw                                                           # noqa: E402
import plot_fig1c_figure8 as f8                                                   # noqa: E402
import plot_fig1c_evidence_geometry as eg                                         # noqa: E402

_ARGS = sys.argv[1:] if len(sys.argv) > 1 and sys.argv[1].isdigit() else []
E_ID = int(_ARGS[0]) if _ARGS else 15                                             # single environments only
RHO_K = float(_ARGS[1]) if len(_ARGS) > 1 else .55
SWEEP_T0 = (1, 4, 8, 12, 16)                                                      # the threshold sweep: start step,
SWEEP_M = (1, 3, 10, 30, 100)                                                     # minimum visits per voxel,
SWEEP_RHO = (.25, .35, .45, .6, .8, 1.)                                           # thickness (neighbour spacings)
GRID = .05
TAU, KAPPA = .2, .8                                                               # a split: >= TAU each, >= KAPPA both
MIN_VISITS, NEFF = 20, 2.5                         # pooled bridges: voxels with >= MIN_VISITS beliefs on <= ~2 cells
FIG_T0, FIG_THICK, FIG_GRID = 8, .25, .05           # the figure: all beliefs from step FIG_T0 on, FIG_THICK spacings
POOLED = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression", "figdata", "round7_jumps",
                      "figure8_pooled.npz")
BLOCKS = {"00": (slice(0, 2), slice(0, 2)), "99": (slice(8, 10), slice(8, 10)), "09": (slice(0, 2), slice(8, 10)),
          "90": (slice(8, 10), slice(0, 2))}                                      # each corner's 2 x 2 block of cells


def corner_mass(p):
    """the posterior mass in each corner's 2 x 2 block (n, T) for joint posteriors p (n, T, 100), k = 10 r_g + r_c"""
    q = p.reshape(*p.shape[:2], 10, 10)
    return {k: q[:, :, a, b].sum((-1, -2)) for k, (a, b) in BLOCKS.items()}


def splits(m, A, B, tau=None, kappa=None):
    """beliefs split between corners A and B: at least tau in each block, at least kappa in the two together"""
    tau, kappa = TAU if tau is None else tau, KAPPA if kappa is None else kappa
    return (m[A] >= tau) & (m[B] >= tau) & (m[A] + m[B] >= kappa)


def build(e=E_ID, tau=None, kappa=None):
    """the embedding, the sheet's points, the wormhole states along (0, 0) - (9, 9) and (0, 9) - (9, 0), the
    adjacent-corner split states (the control), every stream's path in the embedding (n, T, 3), the neighbour
    spacing"""
    D3, _, _ = eg.geometries()
    Y, _ = eg.embed(D3["evidence"][f8.ENVS.index(e)])
    G = Y.reshape(10, 10, 3)
    u = np.arange(0, 9.0001, .1)
    i0 = np.minimum(np.floor(u).astype(int), 8)
    f = u - i0
    fa, fb = f[:, None, None], f[None, :, None]
    sheet = ((1 - fa) * (1 - fb) * G[i0][:, i0] + fa * (1 - fb) * G[i0 + 1][:, i0] + (1 - fa) * fb * G[i0][:, i0 + 1]
             + fa * fb * G[i0 + 1][:, i0 + 1]).reshape(-1, 3)
    p, _ = f8.posterior(e)
    m = corner_mass(p)
    X = p @ Y
    adj = np.zeros(p.shape[:2], bool)
    for A, B in (("00", "09"), ("00", "90"), ("99", "09"), ("99", "90")):
        adj |= splits(m, A, B, tau, kappa)
    nn = np.median([np.linalg.norm(G[a, b] - G[a + da, b + db]) for a in range(10) for b in range(10)
                    for da, db in ((1, 0), (0, 1)) if a + da < 10 and b + db < 10])
    return (Y, sheet, X[splits(m, "00", "99", tau, kappa)], X[splits(m, "09", "90", tau, kappa)], X[adj], X, nn)


def fields(Y, pts, step=GRID):
    """distance fields to each point set on one grid -> list of fields, grid origin"""
    lo, hi = Y.min(0) - 1., Y.max(0) + 1.
    n = np.ceil((hi - lo) / step).astype(int)
    out = []
    for P in pts:
        if not len(P):
            out.append(np.full(n, np.inf))
            continue
        M = np.ones(n, bool)
        M[tuple(np.clip(np.floor((P - lo) / step).astype(int), 0, n - 1).T)] = False
        out.append(ndimage.distance_transform_edt(M) * step)
    return out, lo


def surface_speed(X, points, radius, k=64):
    """the mean speed of the beliefs near each point: every step X_t -> X_{t+1} at its midpoint with its length
    (evidence units per step); per point, the mean over the (up to k nearest) midpoints within radius (nan: none)"""
    from scipy.spatial import cKDTree
    M = (X[:, 1:] + X[:, :-1]).reshape(-1, 3) / 2
    v = np.linalg.norm(X[:, 1:] - X[:, :-1], axis=-1).ravel()
    d, i = cKDTree(M).query(points, k=k, distance_upper_bound=radius)
    ok = np.isfinite(d)
    vals = np.where(ok, v[np.minimum(i, len(v) - 1)], 0.)
    n = ok.sum(1)
    return np.where(n > 0, vals.sum(1) / np.maximum(n, 1), np.nan)


def numbers():
    """environment E_ID: the topology over thicknesses for the sheet alone, with the wormhole states, and with the
    adjacent-corner splits instead (the control); then the wormhole result over the split thresholds"""
    Y, sheet, Pp, Pm, Pa, _, nn = build()
    (ds, dp, dm, da), _ = fields(Y, (sheet, Pp, Pm, Pa), .04)
    dw = np.minimum(dp, dm)
    print(f"environment {E_ID}: wormhole states + {len(Pp)}, - {len(Pm)}; adjacent-corner splits {len(Pa)}; "
          f"neighbour spacing {nn:.2f}")
    for k in (.3, .4, .45, .55, .65, .75):
        t0, t1, t2 = (sw.topology(f < k * nn) for f in (ds, np.minimum(ds, dw), np.minimum(ds, da)))
        print(f"  thickness {k:.2f}: sheet alone holes {t0['b1']} cavities {t0['b2']}; + wormhole states holes "
              f"{t1['b1']} cavities {t1['b2']}; + adjacent-corner splits holes {t2['b1']} cavities {t2['b2']}",
              flush=True)
    for tau, kappa in ((.1, .6), (.15, .6), (.2, .8), (.3, .9)):
        _, _, Pp_, Pm_, _, _, _ = build(E_ID, tau, kappa)
        (dp_, dm_), _ = fields(Y, (Pp_, Pm_), .04)
        t = sw.topology(np.minimum(ds, np.minimum(dp_, dm_)) < RHO_K * nn)
        print(f"  tau {tau}, kappa {kappa}: wormhole states + {len(Pp_)}, - {len(Pm_)}; at {RHO_K:g} spacings holes "
              f"{t['b1']} cavities {t['b2']}", flush=True)


def all_envs(ks=(.45, .65)):
    """the test in every environment -> rows (environment, states +, states -, then per thickness: sheet holes,
    sheet cavities, holes with the wormhole states, cavities with them)"""
    rows = []
    for e in f8.ENVS:
        Y, sheet, Pp, Pm, _, _, nn = build(e)
        (ds, dp, dm), _ = fields(Y, (sheet, Pp, Pm), .04)
        row = [e, len(Pp), len(Pm)]
        for k in ks:
            t0, t1 = sw.topology(ds < k * nn), sw.topology(np.minimum(ds, np.minimum(dp, dm)) < k * nn)
            row += [t0["b1"], t0["b2"], t1["b1"], t1["b2"]]
        rows.append(row)
        print(row, flush=True)
    return np.array(rows)


def lattice_sheet(Y):
    """the sheet (bilinear between the cells, 0.1-cell steps) and the median lattice-neighbour spacing"""
    G = Y.reshape(10, 10, 3)
    u = np.arange(0, 9.0001, .1)
    i0 = np.minimum(np.floor(u).astype(int), 8)
    f = u - i0
    fa, fb = f[:, None, None], f[None, :, None]
    sheet = ((1 - fa) * (1 - fb) * G[i0][:, i0] + fa * (1 - fb) * G[i0 + 1][:, i0] + (1 - fa) * fb * G[i0][:, i0 + 1]
             + fa * fb * G[i0 + 1][:, i0 + 1]).reshape(-1, 3)
    nn = np.median([np.linalg.norm(G[a, b] - G[a + da, b + db]) for a in range(10) for b in range(10)
                    for da, db in ((1, 0), (0, 1)) if a + da < 10 and b + db < 10])
    return sheet, nn


def pooled():
    """the common geometry (the 39 environments' evidence distances averaged, classical MDS) and every environment's
    Optimal beliefs in it: positions X (39, 5000, T, 3), effective number of cells NE = 1 / sum p^2, Cov(r_g, r_c) CV,
    the corner rule's wormhole mask WM (for comparison only); cached in POOLED (about 1 min)"""
    if not os.path.exists(POOLED):
        D3, _, _ = eg.geometries()
        Y, share = eg.embed(D3["evidence"].mean(0))
        out = {k: [] for k in ("X", "NE", "CV", "WM")}
        for e in f8.ENVS:
            p, cov = f8.posterior(e)
            m = corner_mass(p)
            out["X"].append((p @ Y).astype(np.float32))
            out["NE"].append((1 / (p ** 2).sum(-1)).astype(np.float32))
            out["CV"].append(cov.astype(np.float32))
            out["WM"].append(splits(m, "00", "99") | splits(m, "09", "90"))
        np.savez(POOLED, **{k: np.array(v) for k, v in out.items()}, Y=Y, share=share)
    with np.load(POOLED) as f:
        return {k: f[k] for k in f.files}


def pooled_fields(Z, step=.04, min_visits=MIN_VISITS):
    """distance fields on one grid: to the sheet, to the voxels visited by >= min_visits beliefs on <= ~2 cells (any
    cells), to the corner rule's wormhole states; the grid origin, the neighbour spacing"""
    Y = Z["Y"]
    sheet, nn = lattice_sheet(Y)
    lo, hi = Y.min(0) - 1., Y.max(0) + 1.
    n = np.ceil((hi - lo) / step).astype(int)

    def counts(Q):
        idx = np.clip(np.floor((Q - lo) / step).astype(int), 0, n - 1)
        return np.bincount(np.ravel_multi_index(idx.T, n), minlength=int(np.prod(n))).reshape(n)

    def dist(mask):
        return ndimage.distance_transform_edt(~mask) * step

    two = Z["NE"] <= NEFF
    ds = dist(counts(sheet) > 0)
    visited = counts(Z["X"][two]) >= min_visits
    return ds, dist(visited), dist(counts(Z["X"][Z["WM"]]) > 0), lo, nn, visited


def numbers_pooled():
    Z = pooled()
    Y = Z["Y"]
    med = np.median(eg._dist(Y)[np.triu_indices(100, 1)])
    print(f"common geometry ({100 * float(Z['share']):.0f}% in 3-D): opposite-corner gaps "
          f"{np.linalg.norm(Y[0] - Y[99]) / med:.2f} / {np.linalg.norm(Y[9] - Y[90]) / med:.2f} of the median pair, "
          f"adjacent corners "
          f"{np.mean([np.linalg.norm(Y[a] - Y[b]) for a, b in ((0, 9), (0, 90), (99, 9), (99, 90))]) / med:.2f}; "
          f"beliefs {Z['NE'].size:,}, on <= ~2 cells {(Z['NE'] <= NEFF).sum():,}")
    for mv in (5, 20, 50):
        ds, db, dw, _, nn, _ = pooled_fields(Z, min_visits=mv)
        for k in (.35, .45, .55):
            t0, t1, t2 = (sw.topology(f < k * nn) for f in (ds, np.minimum(ds, db), np.minimum(ds, dw)))
            print(f"  >= {mv} visits, thickness {k}: sheet alone holes {t0['b1']} cavities {t0['b2']}; + two-cell "
                  f"beliefs holes {t1['b1']} cavities {t1['b2']} pieces {t1['b0']}; + corner-rule wormhole states "
                  f"holes {t2['b1']} cavities {t2['b2']}", flush=True)


def bridges(B, cov, Y, lo, step, cut=5.):
    """the bridge test: for the red (mean Cov > cut) and the blue (< -cut) voxels of the wrapped set B: is it one
    piece touching both its corners (the piece nearest each corner's voxel is the same), and how many holes are left
    once it is removed -> ((joined, holes without it) red, (joined, holes without it) blue)"""
    n = np.array(B.shape)
    out = []
    for sel, ends in ((cov > cut, (0, 99)), (cov < -cut, (9, 90))):
        lab, _ = ndimage.label(B & sel, np.ones((3, 3, 3), bool))
        idx = ndimage.distance_transform_edt(lab == 0, return_indices=True)[1]
        vox = [tuple(np.clip(np.floor((Y[e] - lo) / step).astype(int), 0, n - 1)) for e in ends]
        near = [int(lab[tuple(idx[(slice(None),) + v])]) for v in vox]
        out.append((near[0] == near[1] and near[0] > 0, sw.topology(B & ~sel)["b1"]))
    return out


def figure(visits=(10, 20, 30, 40, 50, 60), k=FIG_THICK, ncol=3):
    """the rule-free wrap for several minimum visits, seen from one side: every belief of the Optimal from step FIG_T0
    on (joint posteriors, 39 environments pooled, the common geometry), the places (grid FIG_GRID) visited by >= m
    beliefs (one panel each) thickened by k lattice spacings; each panel's surface coloured by the mean joint covariance
    of the beliefs within 1.5 thicknesses of each point (up to the 64 nearest): red where beliefs split along (0, 0) -
    (9, 9) sit, blue along (0, 9) - (9, 0), white where beliefs sit on one cell; holes, cavities and pieces in each
    title"""
    from main.utils import apply_fig_style, fig_path
    from scipy.spatial import cKDTree
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    Z = pooled()
    Y = Z["Y"]
    P = Z["X"][:, :, FIG_T0 - 1:].reshape(-1, 3).astype(np.float64)
    CV = Z["CV"][:, :, FIG_T0 - 1:].ravel()
    _, nn = lattice_sheet(Y)
    step = FIG_GRID
    lo, hi = Y.min(0) - 1., Y.max(0) + 1.
    n = np.ceil((hi - lo) / step).astype(int)
    idx = np.clip(np.floor((P - lo) / step).astype(int), 0, n - 1)
    flat = np.ravel_multi_index(idx.T, n)
    C = np.bincount(flat, minlength=int(np.prod(n))).reshape(n)
    Mv = (np.bincount(flat, CV, minlength=int(np.prod(n))) / np.maximum(C.ravel(), 1)).reshape(n)
    tree = cKDTree(P)
    light = np.array([.3, -.5, .8]) / np.linalg.norm([.3, -.5, .8])
    clim = 15.
    cmap = plt.get_cmap("RdBu_r")
    nrow = -(-len(visits) // ncol)
    S, W = 3.6, ncol * 3.6 + 1.9
    H = 1.05 + nrow * S
    fig = plt.figure(figsize=(W, H))
    fig.text(.5, 1 - .2 / H, f"Every belief of the Optimal from step {FIG_T0} on ({len(P):,} joint posteriors, 39 "
             "environments pooled, their common evidence geometry), wrapped with no rule", ha="center", va="center",
             fontsize=FS["label"])
    fig.text(.5, 1 - .45 / H, f"the places visited by at least m beliefs, thickened by {k:g} lattice spacings; one "
             "panel per m", ha="center", va="center", fontsize=FS["label"])
    for i_, m in enumerate(visits):
        D, ind = ndimage.distance_transform_edt(~(C >= m), return_indices=True)
        D = D * step
        r_, c_ = divmod(i_, ncol)
        rho = k * nn
        top = sw.topology(D < rho)
        (jr, hr), (jb, hb) = bridges(D < rho, Mv[tuple(ind)], Y, lo, step)
        eight = (top["b0"], top["b1"], top["b2"]) == (1, 2, 0) and jr and jb and hr == 1 and hb == 1
        T = lo + step * (sw.march(-D, -rho) + .5)
        d, j = tree.query(T.mean(1), k=64, distance_upper_bound=1.5 * rho)
        ok = np.isfinite(d)
        cov = np.where(ok, CV[np.minimum(j, len(CV) - 1)], 0.).sum(1) / np.maximum(ok.sum(1), 1)
        col = cmap((np.clip(cov, -clim, clim) + clim) / (2 * clim))
        nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        col[:, :3] *= (.6 + .4 * np.abs(nrm @ light))[:, None]
        col[:, 3] = .9
        ax = fig.add_axes([(.5 + c_ * S) / W, (H - .85 - (r_ + 1) * S) / H, S / W, S / H], projection="3d")
        ax.set_proj_type("ortho")
        ax.view_init(elev=-28, azim=118)
        lo_, hi_ = Y.min(0) - .4, Y.max(0) + .4
        ax.set_xlim(lo_[0], hi_[0]), ax.set_ylim(lo_[1], hi_[1]), ax.set_zlim(lo_[2], hi_[2])
        ax.set_box_aspect(hi_ - lo_, zoom=1.25)
        ax.set_axis_off()
        ax.add_collection3d(Poly3DCollection(T, facecolors=col, edgecolors="none"))
        for kk, lab in ((0, "(0, 0)"), (99, "(9, 9)"), (9, "(0, 9)"), (90, "(9, 0)")):
            v3 = Y[kk] + .45 * (Y[kk] - Y.mean(0)) / np.linalg.norm(Y[kk] - Y.mean(0))
            ax.text(*v3, lab, fontsize=FS["tick"] - 1, ha="center", va="center")
        pcs = f"{top['b0']} piece{'s' if top['b0'] != 1 else ''}"
        holes = f"{top['b1']} hole{'s' if top['b1'] != 1 else ''}"
        cavs = f"{top['b2']} cavit{'ies' if top['b2'] != 1 else 'y'}"
        line2 = ("the 8: the two bridges are the two holes" if eight else
                 f"red bridge {'joined' if jr else 'broken'}, blue {'joined' if jb else 'broken'}")
        ax.set_title(f"$\\geq$ {m} visits: {holes}, {cavs}, {pcs}\n{line2}", fontsize=FS["tick"] + 1, pad=-2,
                     color="#1a7a3a" if eight else "k")
        print(f"visits {m}, thickness {k}: {top}; red joined {jr}, holes without it {hr}; blue joined {jb}, holes "
              f"without it {hb}; the 8: {eight}", flush=True)
    cax = fig.add_axes([(W - 1.05) / W, .25, .13 / W, .45])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-clim, clim), cmap=cmap), cax=cax, extend="both")
    cb.set_label("Cov$(r_g, r_c)$ within each belief (its joint posterior),\naveraged over the beliefs nearby\n"
                 "(red: split along (0, 0) - (9, 9); blue: along (0, 9) - (9, 0);\nwhite: on one cell)",
                 fontsize=FS["tick"])
    cb.ax.tick_params(labelsize=FS["tick"] - 1)
    path = fig_path("fig1c_figure8_wrap.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def top2():
    """each pooled belief's two most probable cells (39, 5000, T, 2) int8; cached next to POOLED"""
    path = POOLED.replace(".npz", "_top2.npz")
    if not os.path.exists(path):
        out = []
        for e in f8.ENVS:
            p, _ = f8.posterior(e)
            out.append(np.argsort(p, -1)[..., -2:][..., ::-1].astype(np.int8))
        np.savez(path, top2=np.array(out))
    with np.load(path) as f:
        return f["top2"]


def sweep_t0(t0, step=.05):
    """the wrap's topology for the beliefs from step t0 on, over beliefs (all / on <= ~2 cells) x scaffold (data only /
    + the sheet) x minimum visits x thickness -> round7_jumps/figure8_sweep_t<t0>.npz: res (2, 2, M, RHO, 3) = (pieces,
    holes, cavities); and the off-sheet connections at thickness 0.45 for all beliefs: per minimum visits, each
    cluster's size, its beliefs' commonest pair of cells and that pair's share"""
    Z = pooled()
    Y = Z["Y"]
    sheet, nn = lattice_sheet(Y)
    lo, hi = Y.min(0) - 1., Y.max(0) + 1.
    n = np.ceil((hi - lo) / step).astype(int)

    def vox(Q):
        return np.ravel_multi_index(np.clip(np.floor((Q - lo) / step).astype(int), 0, n - 1).T, n)

    def dist(mask):
        return ndimage.distance_transform_edt(~mask) * step

    ds = dist(np.bincount(vox(sheet), minlength=int(np.prod(n))).reshape(n) > 0)
    X = Z["X"][:, :, t0 - 1:].reshape(-1, 3)
    NE = Z["NE"][:, :, t0 - 1:].ravel()
    T2 = top2()[:, :, t0 - 1:].reshape(-1, 2)
    v_all = vox(X)
    res = np.zeros((2, 2, len(SWEEP_M), len(SWEEP_RHO), 3), int)
    conn = []
    for bi, sel in enumerate((np.ones(len(X), bool), NE <= NEFF)):
        C = np.bincount(v_all[sel], minlength=int(np.prod(n))).reshape(n)
        for mi, m in enumerate(SWEEP_M):
            dd = dist(C >= m)
            for si, base in enumerate((dd, np.minimum(dd, ds))):
                for ri, k in enumerate(SWEEP_RHO):
                    t = sw.topology(base < k * nn)
                    res[bi, si, mi, ri] = (t["b0"], t["b1"], t["b2"])
            if bi == 0:                                                            # the connections off the sheet
                off = (C >= m) & (ds > .45 * nn)
                lab, nl = ndimage.label(off, np.ones((3, 3, 3), bool))
                where = lab.ravel()[v_all]
                for c in range(1, nl + 1):
                    size = int((lab == c).sum())
                    if size < 2:
                        continue
                    pr = np.sort(T2[where == c], 1)
                    keys, cnt = np.unique(pr[:, 0].astype(int) * 100 + pr[:, 1], return_counts=True)
                    j = cnt.argmax()
                    conn.append((m, size, keys[j] // 100, keys[j] % 100, cnt[j] / cnt.sum()))
        print(f"t0 {t0}: {'all' if bi == 0 else 'two-cell'} beliefs done", flush=True)
    path = os.path.join(os.path.dirname(POOLED), f"figure8_sweep_t{t0}.npz")
    np.savez(path, res=res, conn=np.array(conn, float).reshape(-1, 5))
    return path


def _cell(k):
    return f"({int(k) // 10}, {int(k) % 10})"


def figure_sweep():
    """the threshold sweep as a grid of tables (computing the missing start steps first, one after another)"""
    from main.utils import apply_fig_style, fig_path
    from matplotlib.colors import ListedColormap, BoundaryNorm
    FS = apply_fig_style()
    R7d = os.path.dirname(POOLED)
    res, conn = {}, {}
    for t0 in SWEEP_T0:
        path = os.path.join(R7d, f"figure8_sweep_t{t0}.npz")
        if not os.path.exists(path):
            sweep_t0(t0)
        with np.load(path) as f:
            res[t0], conn[t0] = f["res"], f["conn"]
    rows = (("all beliefs, data only", 0, 0), ("all beliefs, + the sheet", 0, 1),
            ("beliefs on about 2 cells or fewer, data only", 1, 0),
            ("beliefs on about 2 cells or fewer, + the sheet", 1, 1))
    cmap = ListedColormap(["#d9d9d9", "#9ecae1", "#41ab5d", "#fdae6b", "#e6550d"])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5, 99], cmap.N)
    PW, PH, GX, GY, X0, YT = 2.6, 1.9, .3, .55, 2.0, 1.1
    W, H = X0 + len(SWEEP_T0) * PW + (len(SWEEP_T0) - 1) * GX + .2, YT + len(rows) * PH + (len(rows) - 1) * GY + .6
    fig = plt.figure(figsize=(W, H))
    fig.text(.5, 1 - .25 / H, "Holes in the shrink-wrapped Optimal beliefs (39 environments pooled, common evidence "
             "geometry) for every threshold: each cell the number of holes (green: 2, the figure 8)",
             ha="center", va="center", fontsize=FS["label"])
    fig.text(.5, 1 - .55 / H, "a dot: more than one piece;  a ring: cavities.  Columns: beliefs from step t0 on; rows "
             "within a table: minimum visits per voxel (grid 0.05); columns within a table: wrap thickness (lattice "
             "spacings)", ha="center", va="center", fontsize=FS["tick"])
    for r_, (name, bi, si) in enumerate(rows):
        for c_, t0 in enumerate(SWEEP_T0):
            x, y = X0 + c_ * (PW + GX), H - YT - PH - r_ * (PH + GY)
            ax = fig.add_axes([x / W, y / H, PW / W, PH / H])
            R_ = res[t0][bi, si]                                                   # (M, RHO, 3)
            ax.imshow(R_[..., 1], cmap=cmap, norm=norm, aspect="auto", origin="upper")
            for i in range(R_.shape[0]):
                for j in range(R_.shape[1]):
                    b0, b1, b2 = R_[i, j]
                    ax.text(j, i, str(b1), ha="center", va="center", fontsize=FS["tick"], color="k")
                    if b0 > 1:
                        ax.plot(j + .32, i - .28, "o", ms=2.5, color="k")
                    if b2 > 0:
                        ax.plot(j - .32, i - .28, "o", ms=4, mfc="none", mec="k", mew=.7)
            ax.set_xticks(range(len(SWEEP_RHO)), [f"{k:g}" for k in SWEEP_RHO] if r_ == len(rows) - 1 else [])
            ax.set_yticks(range(len(SWEEP_M)), [str(m) for m in SWEEP_M] if c_ == 0 else [])
            ax.tick_params(labelsize=FS["tick"] - 1, length=0)
            if r_ == 0:
                ax.set_title(f"from step {t0}", fontsize=FS["label"])
            if r_ == len(rows) - 1:
                ax.set_xlabel("thickness", fontsize=FS["tick"])
            if c_ == 0:
                ax.set_ylabel("min. visits", fontsize=FS["tick"])
                fig.text((X0 - 1.0) / W, (y + PH / 2) / H, name.replace(", ", ",\n"), ha="center", va="center",
                         fontsize=FS["label"], rotation=90)
    path = fig_path("fig1c_figure8_thresholds.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    for t0 in SWEEP_T0:
        for m in SWEEP_M:
            cs = conn[t0][conn[t0][:, 0] == m]
            desc = "; ".join(f"{int(sz)} voxels: {_cell(a)} / {_cell(b)} ({100 * sh:.0f}%)"
                             for _, sz, a, b, sh in cs[np.argsort(-cs[:, 1])][:6])
            print(f"from step {t0}, >= {m} visits: {len(cs)} off-sheet connections" + (f": {desc}" if len(cs) else ""))
    return path


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "sweep":
        print("saved", sweep_t0(int(sys.argv[2])))
    elif len(sys.argv) > 1 and sys.argv[1] == "thresholds":
        print("saved", figure_sweep())
    else:
        numbers_pooled()
        print("saved", figure())
