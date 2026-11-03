"""The figure 8 painted by the trained network's learning signal and activity (29 Oct 2026; John: "We have a bunch of
new gifs that visualize the grid folding and analysis of wormholes and such. Can you connect these to our attempt to
understand the learning dynamics, dimensionality and entanglement? Or are these unrelated. ... I want to see
meaningful striking and clear effects, with interesting visualizations. Avoid being generic.")
-> main/REPLICATION/figure_bin/fig1c_figure8_learning.png

The surface is plot_fig1c_figure8_wrap's rule-free wrap: every belief of the Optimal from step 8 on (39 environments x
5,000 streams, pooled in their common evidence geometry), the places visited by >= 30 beliefs, thickened by 0.25
lattice spacings. It has two holes, and its two bridges are the wormholes (beliefs split between opposite corners).
Each surface point is painted with the mean, over the beliefs within 1.5 thicknesses (the 64 nearest at most), of one
measure of the stream-step each belief belongs to, minus typical (the same cell's 50 streams at the same step):
    the Optimal's Cov(r_g, r_c) (the wrap's own colour: where the wormholes are, raw, not minus typical);
    the Trained RNN's cross-band share of its weight update Delta W_t (gradflow_analysis's measure; < 0: the update's
    band pieces cancel);
    its activity effective dimension PR(u_t) (the gradflow figure's 'Effective dim.');
    the cross-band share again with the credit window fixed at 5 steps (KMAX=4; 1,000 streams per environment);
    entanglement rev_t and factorization regret FR_t (both variables; log10).
The ESN is not shown: its cross-band share is ~0 at every stream-step (rank-one updates; raw SD 0.006 vs 1.04 for the
Trained RNN) and its activity dimension (SD 1.7) tracks neither the belief's spread nor novelty, so it is no contrast.
Panels g, h: per stream-step from step 8, each network measure against how unresolved the Optimal's belief is
(sqrt(Var r_g + Var r_c) from its two marginals, cells), for beliefs split across the wormhole (|Cov| > 8) and every
other belief (|Cov| <= 2); means and 5 - 95% bands over environments resampled (300).

Measured 29 Oct (voxels of 0.05 visited by >= 30 beliefs from step 8; a bridge voxel: mean Cov beyond +-5):
- Bridges vs sheet (|mean Cov| < 1), red / blue bridge / sheet:
  - Trained RNN cross-band share: -1.14 / -1.87 / -0.002. Every bridge voxel lies below the sheet's 5th percentile.
    With a fixed 5-step credit window (KMAX=4, 1,000 streams per environment): -0.16 / -0.53, 64% below.
  - Activity dimension: -2.6 / -1.9 / -0.25 (from step 4: -3.5 / -3.2).
  - ESN: cross-band share 0.00 / 0.00, activity dimension +0.06 / +0.06.
  - Entanglement: +0.02 / +0.03 / -0.001 (not distinctive). FR: +0.26 / +0.23 / -0.003.
- Same joint uncertainty, opposite network (MI 0.4 - 0.7, from step 8; MI only loosely matched, 0.47 vs 0.59):
  - Local splits (n 17,043): cross-band share +0.32, activity dimension +5.7.
  - Wormhole splits (n 14,502): cross-band share -0.88, activity dimension -0.9 (this -0.9 is NOT robust, below).
  - Cross-band: local > 0 and wormhole < 0 in 37 / 37 environments; dimension: wormhole < local in 34 / 37.
- Not the regret: off the wormholes (|Cov| < 2) high FR brings no cancellation. The top FR decile has cross-band
  +0.035 and dimension +0.6.
- Panels g and h, per stream-step from step 8, by how unresolved the Optimal's belief is (sqrt(Var r_g + Var r_c),
  cells; bins 0-.5 / .5-1 / 1-2 / 2-3 / 3-4 / 4-4.5):
  - Every other belief (|Cov| <= 2): activity dimension -0.6 / -0.6 / +1.1 / +6.2 / +9.8 / +13.1 (n 574 in the last
    bin); cross-band share within -0.08 .. +0.11.
  - Split across the wormhole (|Cov| > 8; bins 4-4.5 / 4.5-5 / 5-5.5 / 5.5-6 / 6-6.5): dimension +1.1 / +1.6 / -0.1 /
    -1.5 / -3.1; cross-band share -0.71 / -0.57 / -0.74 / -0.99 / -1.58 (fixed 5-step window -0.14 / -0.11 / -0.19 /
    -0.21 / -0.40).
  - At the same spread (4 - 4.5 cells) other beliefs carry +13.1 dimensions and the wormhole splits +1.1.
  - The ESN: dimension within +-0.07, cross-band share 0.00 everywhere.
- Across environments, uncertainty over cells the evidence aliases cancels the update in 39 / 39 environments: grid
  spread 1 - 3 cells, evidence spread < 0.5 lattice steps, median -0.44 vs settled beliefs; with the fixed window,
  87%.
Independent checks (29 Oct, 2 agents with their own code and exact posteriors recomputed from the observations;
scratchpad verify_fig8/):
- The cancellation is confirmed and specific, but the voxel numbers overstate it.
  - Per belief (|Cov| > 8, steps 8 - 30): -0.83 (median -0.49; red -0.63, blue -0.98), -0.56 per stream. < 0 in 38 / 39
    environments, and below the other beliefs in 38 / 39. The -1.1 / -1.9 come from the densest bridge voxels (19%
    of the wormhole beliefs).
  - Graded: |Cov| 2-4 / 4-8 / 8-12 / 12-16 / 16-21 give -0.08 / -0.32 / -0.57 / -0.86 / -1.33. It grows with step (-0.24
    at step 8 to ~-1.6 late) and with how long the split has lasted.
  - Fixed 5-step window: -0.19 per belief (< 0 in 29 / 39 environments), about 22% of the long-window effect on the same
    rows. So about 78% of it is credit sent back 5 - 15 steps, across the persisting split.
  - Specific to the correlated split. One-axis long splits (|Cov| < 2, spread > 3, n 7,312) give -0.01 while their
    dimension is +10. At the same spread (4 - 5 cells): wormhole -0.63 vs one-axis -0.08. Opposite- vs adjacent-corner
    splits (both 93 - 95% in corner cells, similar number of cells in play, NE 3.7 vs 3.2): -1.25 vs -0.19 (dimension
    -2.4 vs +14.3; only 5 environments have >= 10 adjacent splits).
  - Not explained by step, cell, novelty or change (< 3%), the baseline, or a few environments; FR-matched wormholes
    still give -0.83 while off-wormhole beliefs of equal or higher FR give ~0 in every environment.
- "Activity narrows" is NOT robust after step 10.
  - Wormhole beliefs sit at about the typical dimension: -0.06 pooled over steps 8 - 30, below typical in only 15 / 39
    environments, positive from step 13 on. They are clearly lower only early: steps 2 - 7 give -3.45, < 0 in 33 / 39.
  - What holds: they do not widen the way other beliefs of the same spread or MI do. Those predict ~+8; at spread
    4 - 5 the wormhole beliefs give +1.4 vs +13.2 for one-axis splits.
  - By NE (the number of cells in play, 1 / sum p^2) the wormhole beliefs are not unusually unresolved (64th
    percentile), and their dimension sits near what NE predicts (+0.29).
Links that did NOT hold (29 Oct, scratchpad bridge/):
- Both networks keep the two wormholes apart in their output-null activity. The class-mean distance is 4.6x (Trained)
  vs 3.0x (ESN) the settled spread along it, so holding the information does not separate them.
- The Trained RNN's widening is not larger where the factorized agents fold. With one variable unresolved, it is
  smaller when the other variable sits at an end than in the middle (in ~70% of environments).
- The joint-ESN trap's dead output cells are not set by the evidence geometry. Dead sets of different runs overlap at
  chance, and one run's 9 / 10 dead diagonal did not replicate: another run kept exactly the two diagonals alive.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/fig1c_figure8_learning.py"""
import os
import sys
import warnings
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from scipy import ndimage                                                         # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection                           # noqa: E402
import shrinkwrap as sw                                                           # noqa: E402
import plot_fig1c_figure8_wrap as fw                                              # noqa: E402
import fig1c_two_processes as tp                                                  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)
T0, VISITS, THICK, STEP = 8, 30, .25, .05
E, S, T = 39, 5000, 30
LOCAL, ANTI = "#e08214", "#542788"                                               # local split, wormhole split


def values():
    """per stream-step measures (E, S, T), minus typical; NaN where a network's measure was not computed"""
    V = tp.series()
    he = np.load(os.path.join(tp.mm.R7, "scan_hidden_echo.npz"))
    c4 = np.load(os.path.join(tp.mm.R7, "scan_credit_trained_k4.npz"))
    rows = c4["rows"]
    bx4 = np.full((E, S, T), np.nan)
    bx4[:, rows] = tp.dep(np.stack([c4[f"band_x_e{e:02d}"] for e in tp.mm.ENVS]).astype(np.float64), 10)
    ce = np.load(os.path.join(tp.mm.R7, "scan_credit_echo.npz"))
    bxe = np.full((E, S, T), np.nan)
    bxe[:, ce["rows"]] = tp.dep(np.stack([ce[f"band_x_e{e:02d}"] for e in tp.mm.ENVS]).astype(np.float64), 10)
    B = np.load(os.path.join(tp.mm.SCAN, "scan_Joint.npz"))["beliefs"][tp.mm.ENVS].astype(np.float32)
    r = np.arange(10, dtype=np.float32)
    var = 0
    for sl in (slice(0, 10), slice(10, 20)):                                      # Var(r_g) + Var(r_c): the trace
        q = B[..., sl] / B[..., sl].sum(-1, keepdims=True)                         # of the belief's covariance
        var = var + np.maximum(q @ r ** 2 - (q @ r) ** 2, 0)
    del B
    out = {"band_x": tp.dep(V["band_x"]), "band_x_k4": bx4, "band_x_esn": bxe, "pr": tp.dep(V["pr"]),
           "spread": np.sqrt(var).astype(np.float64),
           "pr_esn": tp.dep(np.stack([he[f"pr_e{e:02d}"] for e in tp.mm.ENVS]).astype(np.float64)),
           "rev": tp.dep(np.log10(V["rev"])), "fr": tp.dep(np.log10(np.maximum(V["fr"], 1e-3))), "mi": V["mi"]}
    return out


def surface():
    """the wrap (fw.figure's construction for one m): triangles (n, 3, 3), the beliefs' positions (N, 3) from step T0,
    their Cov, the geometry Y, the thickness rho"""
    Z = fw.pooled()
    Y = Z["Y"]
    P = Z["X"][:, :, T0 - 1:].reshape(-1, 3).astype(np.float64)
    CV = Z["CV"][:, :, T0 - 1:].ravel().astype(np.float64)
    _, nn = fw.lattice_sheet(Y)
    lo, hi = Y.min(0) - 1., Y.max(0) + 1.
    n = np.ceil((hi - lo) / STEP).astype(int)
    idx = np.clip(np.floor((P - lo) / STEP).astype(int), 0, n - 1)
    C = np.bincount(np.ravel_multi_index(idx.T, n), minlength=int(np.prod(n))).reshape(n)
    D = ndimage.distance_transform_edt(~(C >= VISITS)) * STEP
    rho = THICK * nn
    top = sw.topology(D < rho)
    Tr = lo + STEP * (sw.march(-D, -rho) + .5)
    return Tr, P, CV, Y, rho, top


def paint(tree, Tr, v, rho, k=64):
    """per triangle: the mean of v over the finite values among the k beliefs nearest its centre within 1.5 rho"""
    d, j = tree.query(Tr.mean(1), k=k, distance_upper_bound=1.5 * rho)
    ok = np.isfinite(d)
    vals = np.where(ok, v[np.minimum(j, len(v) - 1)], np.nan)
    return np.nanmean(vals, 1)


def bridge_numbers(V, CV_full):
    """voxel means on the bridges (mean Cov beyond +-5) vs the sheet (|mean Cov| < 1), voxels visited >= VISITS"""
    Z = fw.pooled()
    Y = Z["Y"]
    P = Z["X"][:, :, T0 - 1:].reshape(-1, 3).astype(np.float64)
    lo, hi = Y.min(0) - 1., Y.max(0) + 1.
    n = np.ceil((hi - lo) / STEP).astype(int)
    flat = np.ravel_multi_index(np.clip(np.floor((P - lo) / STEP).astype(int), 0, n - 1).T, n)
    uniq, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    mcov = np.bincount(inv, CV_full) / cnt
    vis = cnt >= VISITS
    sel = {"red bridge": vis & (mcov > 5), "blue bridge": vis & (mcov < -5), "sheet": vis & (np.abs(mcov) < 1)}
    for key, v in V.items():
        if key == "mi":
            continue
        vv = v[:, :, T0 - 1:].ravel()
        fin = np.isfinite(vv)
        s = np.bincount(inv[fin], vv[fin], minlength=len(uniq))
        c = np.bincount(inv[fin], minlength=len(uniq))
        mv = s / np.maximum(c, 1)
        w = {nm: (mv[m] * c[m]).sum() / max(c[m].sum(), 1) for nm, m in sel.items()}
        br = sel["red bridge"] | sel["blue bridge"]
        below = np.mean(mv[br] < np.percentile(mv[sel["sheet"]], 5))
        print(f"{key:10s} red {w['red bridge']:+.3f}  blue {w['blue bridge']:+.3f}  sheet {w['sheet']:+.4f}  "
              "bridge voxels "
              f"below the sheet's 5th percentile {below:.2f}")


PANELS = (("cov", "Where the wormholes are", r"$\mathrm{Cov}(r_g, r_c)$ in the Optimal's belief", 15., "RdBu_r",
           "cells$^2$ (red: (0,0)-(9,9) split; blue: (0,9)-(9,0))"),
          ("band_x", "Trained RNN: its update's pieces cancel",
           r"cross-band share of $\Delta W_t$ (pieces cancel: $< 0$)", 1.5, "PuOr", "share, minus typical"),
          ("pr", "Trained RNN: its activity does not widen there", r"effective dimension $\mathrm{PR}(u_t)$", 4.,
           "BrBG", "dimensions, minus typical (teal: wider)"),
          ("band_x_k4", "Same as b, credit window fixed at 5 steps",
           r"cross-band share, credit $k \leq 4$ (1,000 streams / environment)", 1.5, "PuOr", "share, minus typical"),
          ("rev", "Entanglement: not the bridges", r"$\mathrm{rev}_t$, the update into $t$", .15, "PiYG",
           r"log$_{10}$ nats, minus typical"),
          ("fr", "Factorization regret: high there", r"$\mathrm{FR}_t$, both variables", .4, "PiYG",
           r"log$_{10}$ nats, minus typical"))


def figure():
    from main.utils import apply_fig_style, fig_path
    from scipy.spatial import cKDTree
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    Tr, P, CV, Y, rho, top = surface()
    print("surface:", len(Tr), "triangles;", top, flush=True)
    V = values()
    tree = cKDTree(P)
    light = np.array([.3, -.5, .8]) / np.linalg.norm([.3, -.5, .8])
    nrm = np.cross(Tr[:, 1] - Tr[:, 0], Tr[:, 2] - Tr[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    shade = (.62 + .38 * np.abs(nrm @ light))[:, None]
    flat = {k: V[k][:, :, T0 - 1:].ravel() for k in V}
    flat["cov"] = CV
    S3, GX, GY, X0, TOP = 3.55, .25, 1.0, .25, 1.15
    W_ = X0 + 4 * S3 + 3 * GX + .2
    H_ = TOP + 2 * S3 + GY + 1.0
    fig = plt.figure(figsize=(W_, H_))
    fig.text(.5, 1 - .3 / H_, "On the figure 8's wormholes the trained network's update is large but its pieces "
             "cancel, and only there do unresolved beliefs fail to widen its activity; entanglement does not mark them",
             ha="center", va="center", fontsize=FS["label"] + 3)
    fig.text(.5, 1 - .66 / H_, f"every Optimal belief from step {T0} on ({len(P):,}: 39 environments x 5,000 streams, "
             "pooled in their common evidence geometry), wrapped at $\\geq$ 30 visits, 0.25 lattice spacings (the 8: "
             "2 holes); each point coloured by the beliefs within 1.5 thicknesses (64 at most)", ha="center",
             va="center", fontsize=FS["tick"])
    slots = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)]
    for (key, name, form, lim, cmap_name, unit), (r_, c_) in zip(PANELS, slots):
        col_v = paint(tree, Tr, flat[key], rho)
        cmap = plt.get_cmap(cmap_name)
        col = cmap((np.clip(np.nan_to_num(col_v), -lim, lim) + lim) / (2 * lim))
        col[:, :3] *= shade
        col[:, 3] = .92
        x, y = X0 + c_ * (S3 + GX), H_ - TOP - S3 - r_ * (S3 + GY)
        ax = fig.add_axes([x / W_, (y + .3) / H_, S3 / W_, (S3 - .3) / H_], projection="3d")
        ax.set_proj_type("ortho")
        ax.view_init(elev=-28, azim=118)
        lo_, hi_ = Y.min(0) - .4, Y.max(0) + .4
        ax.set_xlim(lo_[0], hi_[0]), ax.set_ylim(lo_[1], hi_[1]), ax.set_zlim(lo_[2], hi_[2])
        ax.set_box_aspect(hi_ - lo_, zoom=1.3)
        ax.set_axis_off()
        ax.add_collection3d(Poly3DCollection(Tr, facecolors=col, edgecolors="none"))
        for kk, lab in ((0, "(0, 0)"), (99, "(9, 9)"), (9, "(0, 9)"), (90, "(9, 0)")):
            v3 = Y[kk] + .45 * (Y[kk] - Y.mean(0)) / np.linalg.norm(Y[kk] - Y.mean(0))
            ax.text(*v3, lab, fontsize=FS["tick"] - 1, ha="center", va="center")
        fig.text((x + S3 / 2) / W_, (y + S3 + .05) / H_, f"{'abcdef'[slots.index((r_, c_))]}   {name}",
                 ha="center", va="bottom", fontsize=FS["label"] + .5, fontweight="bold")
        fig.text((x + S3 / 2) / W_, (y + S3 - .14) / H_, form, ha="center", va="bottom", fontsize=FS["tick"],
                 color="0.25")
        cax = fig.add_axes([(x + .45) / W_, (y + .12) / H_, (S3 - .9) / W_, .1 / H_])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-lim, lim), cmap=cmap), cax=cax,
                          orientation="horizontal", extend="both")
        cb.set_label(unit, fontsize=FS["tick"] - 1, labelpad=1)
        cb.ax.tick_params(labelsize=FS["tick"] - 1.5)
    # g, h: per stream-step from step T0: the network measure against how unresolved the belief is, beliefs split
    # across the wormhole (|Cov| > 8) vs every other belief (|Cov| <= 2), means and 5 - 95% bands over environments
    cvf = flat["cov"].reshape(E, S, T - T0 + 1)
    spf = flat["spread"].reshape(E, S, T - T0 + 1)
    groups = (("every other belief ($|\\mathrm{Cov}| \\leq 2$)", np.abs(cvf) <= 2, "0.15"),
              ("split across the wormhole ($|\\mathrm{Cov}| > 8$)", np.abs(cvf) > 8, ANTI))
    xe = np.arange(0, 6.51, .25)
    xm = (xe[:-1] + xe[1:]) / 2
    nb = len(xm)
    ib = np.clip(np.digitize(spf, xe) - 1, 0, nb - 1)
    rng = np.random.default_rng(0)
    boots = rng.integers(0, E, (300, E))

    def curve(v, m, min_n):
        v = v.reshape(E, S, T - T0 + 1)
        ok = m & np.isfinite(v)
        s_, c_ = np.zeros((E, nb)), np.zeros((E, nb))
        for e in range(E):
            s_[e] = np.bincount(ib[e][ok[e]], v[e][ok[e]], minlength=nb)
            c_[e] = np.bincount(ib[e][ok[e]], minlength=nb)
        mean = s_.sum(0) / np.maximum(c_.sum(0), 1)
        bs = s_[boots].sum(1) / np.maximum(c_[boots].sum(1), 1)
        keep = c_.sum(0) >= min_n
        return mean, np.percentile(bs, [5, 95], 0), keep, c_.sum(0)

    x0 = X0 + 2 * (S3 + GX) + .75
    gw, gh = S3 - .95, S3 - 1.3
    y0 = H_ - TOP - 2 * S3 - GY + .8
    gh_panels = (("pr", None, "g", "Unresolved beliefs widen the activity,", "except across the wormhole",
                  "activity effective dimension\nminus typical", (-8, 14), 300),
                 ("band_x", None, "h", "The update's pieces cancel", "only across the wormhole",
                  "cross-band share of $\\Delta W_t$\nminus typical", (-1.6, .6), 60))
    for i_, (kT, kE, letter, t1, t2, ylab, ylim, min_n) in enumerate(gh_panels):
        ax = fig.add_axes([(x0 + i_ * (gw + 1.0)) / W_, y0 / H_, gw / W_, gh / H_])
        for lab, m, col in groups:
            mean, band, keep, n_ = curve(flat[kT], m, min_n)
            ax.fill_between(xm[keep], band[0][keep], band[1][keep], color=col, alpha=.2, lw=0)
            ax.plot(xm[keep], mean[keep], color=col, lw=2, label=f"Trained RNN, {lab}")
        ax.axhline(0, color="0.5", lw=.7, ls=":")
        ax.set_xlim(0, 6.5), ax.set_ylim(*ylim)
        ax.set_xlabel(r"how unresolved: $\sqrt{\mathrm{Var}\,r_g + \mathrm{Var}\,r_c}$ (cells)", fontsize=FS["tick"],
                      labelpad=1)
        ax.set_ylabel(ylab, fontsize=FS["tick"])
        ax.tick_params(labelsize=FS["tick"] - .5)
        ax.spines[["top", "right"]].set_visible(False)
        fig.text((x0 + i_ * (gw + 1.0) + gw / 2) / W_, (y0 + gh + .42) / H_, f"{letter}   {t1}", ha="center",
                 va="bottom", fontsize=FS["label"] + .5, fontweight="bold")
        fig.text((x0 + i_ * (gw + 1.0) + gw / 2) / W_, (y0 + gh + .2) / H_, t2, ha="center", va="bottom",
                 fontsize=FS["label"] + .5, fontweight="bold")
        if i_ == 0:
            ax.legend(fontsize=FS["tick"] - 1.5, frameon=False, loc="lower right", handletextpad=.4)
    fig.text((x0 + gw + .5) / W_, (y0 - .62) / H_, "each belief's stream-step from step 8 (bands: 5 - 95% over "
             "environments resampled); at the same spread (4 - 5 cells): wormhole splits +1.4 dimensions\nvs one-axis "
             "splits +13.2, cross-band -0.63 vs -0.08; opposite- vs adjacent-corner splits (as few cells in play): "
             "cross-band -1.25 vs -0.19", ha="center", va="top", fontsize=FS["tick"] - 1, color="0.3")
    fig.text(.5, .22 / H_, "per belief the wormholes' cancellation averages -0.83 (median -0.49; $<$ 0 in 38 / 39 "
             "environments; the bridges' densest voxels -1.1 / -1.9); with a 5-step credit window -0.19: most of it is "
             "credit sent back 5 - 15 steps, across the persisting split; high factorization regret alone brings no "
             "cancellation; the update itself is 5x its typical size there (the error vs the Optimal 4x)", ha="center",
             va="center", fontsize=FS["tick"] - .5, color="0.3")
    path = fig_path("fig1c_figure8_learning.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("saved", path)
    bridge_numbers(V, np.load(os.path.join(tp.mm.R7, "figure8_pooled.npz"))["CV"][:, :, T0 - 1:].ravel())
    mi, cv = V["mi"][:, :, T0 - 1:], np.load(os.path.join(tp.mm.R7, "figure8_pooled.npz"))["CV"][:, :, T0 - 1:]
    split = (mi > .4) & (mi < .7)
    for nm, m in (("local", split & (np.abs(cv) < 1)), ("wormhole", split & (np.abs(cv) > 8))):
        print(f"{nm}: n {m.sum()}, MI {mi[m].mean():.2f}, cross-band {np.nanmean(V['band_x'][:, :, T0 - 1:][m]):+.3f}, "
              f"dimension {np.nanmean(V['pr'][:, :, T0 - 1:][m]):+.2f}, ESN dimension "
              f"{np.nanmean(V['pr_esn'][:, :, T0 - 1:][m]):+.2f}")
    return path


if __name__ == "__main__":
    figure()
