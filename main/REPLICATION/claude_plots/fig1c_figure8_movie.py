"""The figure 8 over time: fig1c_figure8_learning.png as a movie (29 Oct 2026; John: "Can you see what you wanted to
show in animated gifs?")
-> main/REPLICATION/figure_bin/fig1c_movie_figure8_learning.gif   movie_surface() (default)

movie_surface(): the wrap of fig1c_figure8_learning (every Optimal belief from step 8 on, >= 30 visits, 0.25 spacing),
drawn as NSURF of its surface points (painter's order, 3-level shading), each coloured at step t by the mean over the
beliefs near it (1.5 thicknesses, 32 nearest at most) from steps t - 1 .. t + 1 (painted(); light grey: no belief
there yet), linearly between steps; the camera swings +-SWING degrees in azimuth and +-8 in elevation around the wrap's
view; panels a Cov(r_g, r_c) (where the wormholes are), b the Trained RNN's cross-band share of Delta W_t, c its
activity effective dimension, d entanglement (b - d minus typical: the same cell's 50 streams at the same step).
Seen 29 Oct (test frames at steps 5 / 12 / 28): early the red bridge is occupied, the activity on it is narrow (c
brown) and entanglement there is high (d green), the cancellation weak (b); from step ~8 the cancellation on both
bridges deepens (b orange) while the narrowing fades (c pale): the time course of the verified numbers in
fig1c_figure8_learning's docstring (activity lower only at steps 2 - 7; cancellation growing with step and with how
long the split lasts).

movie_processes() -> fig1c_movie_figure8_processes.gif (John 29 Oct: "show the other relevant measures, like error and
effective dim etc. ... in particle form ... can we conclude that the wormhole corner splitting and entanglement
representation unfolding are different processes? Or the same process at different steps (early vs late)"): the
beliefs as particles in the evidence geometry (grey: Q_BACK streams per cell), with two rows of coloured particles from
Q_POOL streams per cell: top, beliefs split across a wormhole (|Cov| > 8); bottom, the top-3% most entangled steps of
their environment and step (not wormholes); columns: which beliefs, then each one's own value (raw, minus typical; no
averaging) of the RNN's error vs the Optimal, its update's size, the update's pieces (cross-band share), credit
coherence, activity effective dimension and turn; PuOr, purple above typical, orange below (matplotlib's PuOr runs
orange at 0 to purple at 1; the first test used PuOr_r and inverted the legend, caught on the test frame).
Measured 29 Oct (stream-steps, minus typical; wormhole splits vs top-10% entangled steps that are not wormholes):
- Steps 8 - 30: error +0.62 vs +0.02 (log10); update size +0.72 vs +0.06; cross-band share -0.83 vs +0.32; coherence
  -0.03 vs +0.12; activity dimension -0.06 vs +1.95; turn +0.03 vs +0.11; FR +0.26 vs +0.03; belief effective
  dimension -0.53 vs -0.03.
- Steps 2 - 7: error +0.32 vs +0.10; update size +0.35 vs +0.04; cross-band +0.04 vs +0.08; activity dimension -3.45 vs
  +2.48; coherence -0.01 vs +0.05.
- Overlap: 29% of wormhole stream-steps are top-10% entangled (2.9x chance); only 2.6% of the top-10% entangled steps
  are wormhole states (base 0.9%).
- Formation: wormhole splits form early (|Cov| crossing 8 upward; n 18,982; median step 4, IQR 2 - 7). The forming
  update is entangled: +0.16 log10, 79th percentile, top-10% in 33% (3.3x chance); the step before +0.07, 24%.
- Occupancy (% of stream-steps in a wormhole state) at steps 2 / 5 / 8 / 12 / 20 / 30: 4.5 / 2.4 / 1.3 / 0.66 / 0.21 /
  0.06; formed 2.9 / 0.82 / 0.36 / 0.16 / 0.04 / 0.01%.
- Reading: two processes, coupled at the start. Entangled updates seed the corner splits early. The split then persists
  as a state the network treats in the opposite way to entangled steps: large error and update, cancelling pieces, no
  widening, no coherence rise. Most entanglement (97%) never involves a wormhole and runs all episode. So this is not
  one process seen early and late.
movie_particles() (local means; kept, not used): every particle coloured by the mean of its 48 nearest beliefs. That
dilutes the sparse bridges; hard to read.

movie() (particles; kept, not the default): every belief of the Optimal as a particle moving through the evidence
geometry over the 30 steps, coloured by what the trained network does at that stream-step. Single stream-steps vary
more than the effect: only the cancellation panel picks out the bridges; the activity and entanglement panels look like
confetti. Hence the painted surface.

Particles: 39 environments x 100 cells x NS_CELL streams. Each sits at its stream's Optimal joint posterior averaged
over the cells' positions in the environments' common evidence geometry (plot_fig1c_figure8_wrap.pooled, 3-D MDS, 88%):
settled beliefs sit on their cell; a belief split between opposite corners sits on the short bridge between them (the
wormhole); broad beliefs float inside the curl. Grey scaffold: the 100 cells and their grid edges in that geometry.
Panels (colour = the stream-step's value; network measures and entanglement minus typical = the same cell's 50 streams
at the same step, then smoothed over steps with a Gaussian of SIGMA steps, renormalized at the ends):
    a  Cov(r_g, r_c) inside the Optimal's belief (where the wormholes are: red / blue)
    b  the Trained RNN's cross-band share of its weight update Delta W_t (orange: the update's pieces cancel)
    c  its activity effective dimension PR(u_t) (teal: wider)
    d  entanglement rev_t (log10)
Particles whose belief is split across a wormhole (|Cov| > 8) are drawn larger in every panel, so the bridges can be
found; draw order is random (strong colours are not put on top). Timing: Catmull-Rom between steps, STEP_S seconds per
step at FPS frames per second, a 1.2 s hold at the end; the camera swings +-SWING degrees around the wrap figure's view.
Numbers behind the effect: fig1c_figure8_learning.py's docstring (verified 29 Oct).
Run from the repository root (about 10 - 20 min; TEST=4,11 renders those steps as PNGs into TEST_DIR instead):
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig1c_figure8_movie.py \\
        [surface|processes|particles|particles_raw]"""
import os
import sys
import warnings
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402
from PIL import Image                                                             # noqa: E402
import plot_fig1c_measures_movie as mm                                            # noqa: E402
import plot_fig1c_figure8_wrap as fw                                              # noqa: E402
import fig1c_figure8_learning as fl                                               # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)
E, S, T, R, NS = 39, 5000, 30, 10, 50
NS_CELL = 6                                                                       # particles per cell
SIGMA = 1.                                                                        # smoothing over steps
FPS, STEP_S = 25, .5
ELEV, AZIM, SWING = -28., 118., 35.
LEVELS = 31                                                                       # colour levels per panel
DPI = 110
PANELS = (("cov", "Where the wormholes are", r"$\mathrm{Cov}(r_g, r_c)$ in the Optimal's belief", 15., "RdBu_r",
           "cells$^2$: red (0,0)-(9,9), blue (0,9)-(9,0)"),
          ("band_x", "Trained RNN: its learning signal", r"cross-band share of $\Delta W_t$ (orange: pieces cancel)",
           1.5, "PuOr", "minus typical"),
          ("pr", "Trained RNN: its activity", r"effective dimension $\mathrm{PR}(u_t)$ (teal: wider)", 8., "BrBG",
           "dimensions, minus typical"),
          ("rev", "Entanglement", r"$\mathrm{rev}_t$, the update into $t$", .3, "PiYG",
           r"log$_{10}$ nats, minus typical"))


def _smooth(V, sigma=SIGMA):
    """Gaussian smoothing over the last axis (steps), renormalized at the ends; NaN at step 1 left out"""
    if sigma <= 0:
        return V
    t = np.arange(T)
    K = np.exp(-.5 * ((t[:, None] - t[None]) / sigma) ** 2)                       # (T out, T in)
    ok = np.isfinite(V)
    num = np.einsum("...s,ts->...t", np.where(ok, V, 0.), K)
    den = np.einsum("...s,ts->...t", ok.astype(float), K)
    return num / np.maximum(den, 1e-12)


def data():
    """positions (n, T, 3), Cov (n, T), the panels' values (n, T) for the chosen particles; the geometry Y"""
    Z = fw.pooled()
    rows = (np.arange(R * R)[:, None] * NS + np.arange(NS_CELL)[None]).ravel()
    X = Z["X"][:, rows].reshape(-1, T, 3).astype(np.float64)
    cov = Z["CV"][:, rows].reshape(-1, T).astype(np.float64)
    V = fl.values()
    vals = {"cov": cov}
    for k in ("band_x", "pr", "rev"):
        vals[k] = _smooth(V[k][:, rows].reshape(-1, T))
    perm = np.random.default_rng(0).permutation(len(X))                            # random draw order
    return X[perm], cov[perm], {k: v[perm] for k, v in vals.items()}, Z["Y"]


def _view(elev, azim):
    """orthographic screen axes for a camera at (elev, azim) in degrees, as matplotlib's 3-D view"""
    e, a = np.radians(elev), np.radians(azim)
    right = np.array([-np.sin(a), np.cos(a), 0.])
    up = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    return np.stack([right, up], 1)                                               # (3, 2)


def _edges():
    """the grid's neighbour pairs among the 100 cells"""
    return [(a, b) for a in range(R * R) for b in range(R * R)
            if a < b and abs(a // R - b // R) + abs(a % R - b % R) == 1]


def _taus():
    return list(np.linspace(0, T - 1, int((T - 1) * STEP_S * FPS) + 1)) + [T - 1.] * int(1.2 * FPS)


def movie():
    from main.utils import fig_path
    plt.rcParams.update(mm.RC)
    X, cov, vals, Y = data()
    n = len(X)
    print(f"{n:,} particles", flush=True)
    centre = Y.mean(0)
    rad = np.percentile(np.linalg.norm(X.reshape(-1, 3) - centre, axis=1), 99.5) * 1.05
    edges = [(a, b) for a in range(R * R) for b in range(R * R) if a < b
             and abs(a // R - b // R) + abs(a % R - b % R) == 1]
    corners = ((0, "(0, 0)"), (99, "(9, 9)"), (9, "(0, 9)"), (90, "(9, 0)"))
    cmaps = {k: plt.get_cmap(cm) for k, _, _, _, cm, _ in PANELS}
    lev = np.linspace(0, 1, LEVELS)
    PW, PH, X0, GX, TOP, BOT = 3.3, 3.3, .2, .2, 1.05, 1.0
    W_ = X0 + 4 * PW + 3 * GX + .2
    H_ = TOP + PH + BOT
    fig = plt.figure(figsize=(W_, H_), dpi=DPI)
    fig.patch.set_facecolor("white")
    fig.text(.5, 1 - .27 / H_, "The figure 8 in motion: on the wormholes the trained network's learning signal "
             "cancels, "
             "and their uncertainty does not widen its activity", ha="center", va="center", fontsize=13)
    clock = fig.text(.5, 1 - .6 / H_, "", ha="center", va="center", fontsize=12)
    axes, scat, lines, labels = [], [], [], []
    for i, (key, name, form, lim, cm, unit) in enumerate(PANELS):
        x = X0 + i * (PW + GX)
        ax = fig.add_axes([x / W_, BOT / H_, PW / W_, PH / H_])
        ax.set_xlim(-rad, rad), ax.set_ylim(-rad, rad), ax.set_aspect("equal"), ax.axis("off")
        lc = LineCollection([], colors="0.78", linewidths=.6, zorder=1)
        ax.add_collection(lc)
        sc = ax.scatter(np.zeros(n), np.zeros(n), s=np.ones(n), c=np.zeros((n, 3)), marker="s", linewidths=0,
                        antialiased=False, zorder=2)
        labs = [ax.text(0, 0, lab, fontsize=8.5, ha="center", va="center", color="0.25", zorder=3)
                for _, lab in corners]
        fig.text((x + PW / 2) / W_, (BOT + PH + .3) / H_, f"{'abcd'[i]}   {name}", ha="center", va="bottom",
                 fontsize=11.5, fontweight="bold")
        fig.text((x + PW / 2) / W_, (BOT + PH + .1) / H_, form, ha="center", va="bottom", fontsize=9.5, color="0.25")
        cax = fig.add_axes([(x + .45) / W_, .5 / H_, (PW - .9) / W_, .1 / H_])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-lim, lim), cmap=cmaps[key]), cax=cax,
                          orientation="horizontal")
        cb.set_label(unit, fontsize=9, labelpad=1)
        cb.ax.tick_params(labelsize=8.5)
        axes.append(ax), scat.append(sc), lines.append(lc), labels.append(labs)
    fig.text(.5, .17 / H_, f"39 environments x {NS_CELL} streams per cell ({n:,} particles), each at its Optimal "
             "belief's "
             "place in the common evidence geometry; colour: that stream-step's value minus typical, smoothed over "
             f"+-{SIGMA:g} step; enlarged: beliefs split across a wormhole (|Cov| > 8)", ha="center", va="center",
             fontsize=8.5, color="0.3")
    taus = [float(v) for v in os.environ['TEST'].split(',')] if os.environ.get('TEST') else _taus()
    frames = []
    for f_, tau in enumerate(taus):
        ph = 2 * np.pi * f_ / len(taus)
        P = _view(ELEV + 8 * np.sin(ph), AZIM + SWING * np.sin(ph))
        pos = mm._blend(X, tau, axis=1) - centre                                    # (n, 3)
        xy = pos @ P
        s = int(np.clip(np.floor(tau), 0, T - 2))
        fr = tau - s
        cv_now = (1 - fr) * cov[:, s] + fr * cov[:, min(s + 1, T - 1)]
        big = np.abs(cv_now) > 8
        sizes = np.where(big, 9., 1.6) * (72 / DPI) ** 2
        Yp = (Y - centre) @ P
        seg = np.stack([Yp[[a for a, _ in edges]], Yp[[b for _, b in edges]]], 1)
        for i, (key, _, _, lim, _, _) in enumerate(PANELS):
            v = vals[key]
            vt = (1 - fr) * v[:, s] + fr * v[:, min(s + 1, T - 1)]
            q = np.round((np.clip(np.nan_to_num(vt), -lim, lim) + lim) / (2 * lim) * (LEVELS - 1)).astype(int)
            col = cmaps[key](lev[q])[:, :3]
            scat[i].set_offsets(xy)
            scat[i].set_sizes(sizes)
            scat[i].set_facecolors(col)
            lines[i].set_segments(seg)
            for (k_, _), tx in zip(corners, labels[i]):
                d = Yp[k_] / np.linalg.norm(Yp[k_]) * .12 * rad
                tx.set_position(Yp[k_] + d)
        clock.set_text(f"step {min(int(np.floor(tau)) + 1, T)} of {T}")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        if f_ % 50 == 0:
            print(f"frame {f_} / {len(taus)}", flush=True)
    plt.close(fig)
    if os.environ.get("TEST"):
        out = os.environ.get("TEST_DIR", ".")
        for tau, fr_ in zip(taus, frames):
            fr_.save(os.path.join(out, f"test_frame_{tau:04.1f}.png"))
        return None
    keys = []
    for key, *_ in PANELS:
        keys += [tuple(int(round(255 * c)) for c in cmaps[key](u)[:3]) for u in lev]
    keys += [(int(round(v)),) * 3 for v in np.linspace(0, 255, 64)]
    pal = np.unique(np.array(keys, np.uint8), axis=0)
    assert len(pal) <= 255, len(pal)
    quant = mm._Exact(pal)
    q = [quant(fr) for fr in frames]
    path = fig_path("fig1c_movie_figure8_learning.gif")
    q[0].save(path, save_all=True, append_images=q[1:], duration=1000 // FPS, loop=0)
    print(f"saved {path}: {len(q)} frames, {os.path.getsize(path) / 1e6:.1f} MB", flush=True)
    return path


# ═══════════════════════════════════════════════════════════════════
# The painted surface in motion (default): the step-by-step version of fig1c_figure8_learning.png
# ═══════════════════════════════════════════════════════════════════

SURF_PANELS = (("cov", "Where the wormholes are", r"$\mathrm{Cov}(r_g, r_c)$ in the Optimal's belief", 15., "RdBu_r",
                "cells$^2$: red (0,0)-(9,9), blue (0,9)-(9,0)"),
               ("band_x", "Trained RNN: its update's pieces",
                r"cross-band share of $\Delta W_t$ (orange: the pieces cancel)", 1.2, "PuOr", "minus typical"),
               ("pr", "Trained RNN: its activity", r"effective dimension $\mathrm{PR}(u_t)$ (teal: wider)", 5., "BrBG",
                "dimensions, minus typical"),
               ("rev", "Entanglement", r"$\mathrm{rev}_t$, the update into $t$", .12, "PiYG",
                r"log$_{10}$ nats, minus typical"))
SHADES = (.72, .86, 1.)                                                           # 3-level shading
NSURF = 60000                                                                     # surface points drawn


def painted(k=32, half=1):
    """per step t (index 0 .. T-1): each surface point's mean of each panel's value over the beliefs of steps
    t-half .. t+half within 1.5 thicknesses (k nearest at most); NaN where none. Cached in the scratch-free figdata
    folder of fig1c_two_processes (round8_two_processes/figure8_painted.npz)"""
    from scipy.spatial import cKDTree
    import fig1c_two_processes as tp
    path = os.path.join(tp.OUT, "figure8_painted.npz")
    if os.path.exists(path):
        with np.load(path) as f:
            return {kk: f[kk] for kk in f}
    Tr, _, _, Y, rho, _ = fl.surface()
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(Tr), min(NSURF, len(Tr)), replace=False))
    C = Tr[pick].mean(1)
    nrm = np.cross(Tr[pick, 1] - Tr[pick, 0], Tr[pick, 2] - Tr[pick, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    Z = fw.pooled()
    X = Z["X"].astype(np.float64)                                                 # (E, S, T, 3)
    V = fl.values()
    vals = {"cov": Z["CV"].astype(np.float64), "band_x": V["band_x"], "pr": V["pr"], "rev": V["rev"]}
    out = {kk: np.full((T, len(C)), np.nan, np.float32) for kk in vals}
    out["count"] = np.zeros((T, len(C)), np.int16)
    for t in range(1, T):
        sl = np.arange(max(1, t - half), min(T, t + half + 1))
        P = X[:, :, sl].reshape(-1, 3)
        tree = cKDTree(P)
        d, j = tree.query(C, k=k, distance_upper_bound=1.5 * rho)
        ok = np.isfinite(d)
        jj = np.minimum(j, len(P) - 1)
        out["count"][t] = ok.sum(1)
        for kk, v in vals.items():
            vv = v[:, :, sl].reshape(-1)[jj]
            vv = np.where(ok & np.isfinite(vv), vv, np.nan)
            out[kk][t] = np.nanmean(vv, 1)
        print(f"painted step {t + 1}", flush=True)
    out.update(C=C, nrm=nrm, Y=Y, rho=rho)
    np.savez(path, **out)
    return out


def movie_surface():
    from main.utils import fig_path
    plt.rcParams.update(mm.RC)
    Pd = painted()
    C, nrm, Y = Pd["C"], Pd["nrm"], Pd["Y"]
    centre = Y.mean(0)
    rad = np.max(np.linalg.norm(C - centre, axis=1)) * 1.08
    light = np.array([.3, -.5, .8]) / np.linalg.norm([.3, -.5, .8])
    shade_idx = np.digitize(np.abs(nrm @ light), [.4, .75])                      # 0 / 1 / 2
    cmaps = {k: plt.get_cmap(cm) for k, _, _, _, cm, _ in SURF_PANELS}
    LV = 17
    lev = np.linspace(0, 1, LV)
    corners = ((0, "(0, 0)"), (99, "(9, 9)"), (9, "(0, 9)"), (90, "(9, 0)"))
    PW, PH, X0, GX, TOP, BOT = 3.3, 3.3, .2, .2, 1.25, 1.3
    W_ = X0 + 4 * PW + 3 * GX + .2
    H_ = TOP + PH + BOT
    fig = plt.figure(figsize=(W_, H_), dpi=DPI)
    fig.patch.set_facecolor("white")
    fig.text(.5, 1 - .27 / H_, "The figure 8 over time: on the wormholes the trained network's update is large but "
             "its pieces cancel, and their uncertainty does not widen its activity", ha="center", va="center",
             fontsize=13)
    clock = fig.text(.5, 1 - .58 / H_, "", ha="center", va="center", fontsize=12.5, fontweight="bold")
    scat, labels = [], []
    for i, (key, name, form, lim, cm, unit) in enumerate(SURF_PANELS):
        x = X0 + i * (PW + GX)
        ax = fig.add_axes([x / W_, BOT / H_, PW / W_, PH / H_])
        ax.set_xlim(-rad, rad), ax.set_ylim(-rad, rad), ax.set_aspect("equal"), ax.axis("off")
        sc = ax.scatter(np.zeros(len(C)), np.zeros(len(C)), s=(3.2 * 72 / DPI) ** 2, c=np.zeros((len(C), 3)),
                        marker="s", linewidths=0, antialiased=False)
        labs = [ax.text(0, 0, lab, fontsize=9, ha="center", va="center", color="0.2") for _, lab in corners]
        fig.text((x + PW / 2) / W_, (BOT + PH + .33) / H_, f"{'abcd'[i]}   {name}", ha="center", va="bottom",
                 fontsize=11.5, fontweight="bold")
        fig.text((x + PW / 2) / W_, (BOT + PH + .12) / H_, form, ha="center", va="bottom", fontsize=9.5, color="0.25")
        cax = fig.add_axes([(x + .45) / W_, .82 / H_, (PW - .9) / W_, .1 / H_])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-lim, lim), cmap=cmaps[key]), cax=cax,
                          orientation="horizontal", extend="both")
        cb.set_label(unit, fontsize=9, labelpad=1)
        cb.ax.tick_params(labelsize=8.5)
        scat.append(sc), labels.append(labs)
    fig.text(.5, .16 / H_, "the 8: every belief of the Optimal from step 8 on, wrapped (39 environments x 5,000 "
             "streams); "
             "each point coloured by the mean over the beliefs near it at steps t - 1 .. t + 1 (light grey: none "
             "there yet)", ha="center", va="center", fontsize=9, color="0.3")
    taus = [float(v) for v in os.environ["TEST"].split(",")] if os.environ.get("TEST") else _taus()
    frames = []
    empty = np.array([.9, .9, .9])
    for f_, tau in enumerate(taus):
        tau = max(tau, 1.)                                                        # step 1 has no update
        ph = 2 * np.pi * f_ / max(len(taus), 1)
        Pv = _view(ELEV + 8 * np.sin(ph), AZIM + SWING * np.sin(ph))
        e, a = np.radians(ELEV + 8 * np.sin(ph)), np.radians(AZIM + SWING * np.sin(ph))
        depth = (C - centre) @ np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        order = np.argsort(-depth)                                                # far first (painter)
        xy = ((C - centre) @ Pv)[order]
        s = int(np.clip(np.floor(tau), 1, T - 2))
        fr = tau - s
        sh = np.array(SHADES)[shade_idx[order]][:, None]
        for i, (key, _, _, lim, _, _) in enumerate(SURF_PANELS):
            v0, v1 = Pd[key][s][order], Pd[key][min(s + 1, T - 1)][order]
            vt = np.where(np.isfinite(v0) & np.isfinite(v1), (1 - fr) * v0 + fr * v1, np.where(fr < .5, v0, v1))
            q = np.round((np.clip(np.nan_to_num(vt), -lim, lim) + lim) / (2 * lim) * (LV - 1)).astype(int)
            col = cmaps[key](lev[q])[:, :3]
            col = np.where(np.isfinite(vt)[:, None], col, empty)
            col = np.round(col * sh * 255) / 255
            scat[i].set_offsets(xy)
            scat[i].set_facecolors(col)
            for (k_, _), tx in zip(corners, labels[i]):
                p2 = (Y[k_] - centre) @ Pv
                tx.set_position(p2 / np.linalg.norm(p2) * (np.linalg.norm(p2) + .14 * rad))
        clock.set_text(f"step {min(int(round(tau)) + 1, T)} of {T}")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        if f_ % 50 == 0:
            print(f"frame {f_} / {len(taus)}", flush=True)
    plt.close(fig)
    if os.environ.get("TEST"):
        out = os.environ.get("TEST_DIR", ".")
        for tau, fr_ in zip(taus, frames):
            fr_.save(os.path.join(out, f"surf_frame_{tau:04.1f}.png"))
        return None
    keys = []
    for key, *_ in SURF_PANELS:
        for shv in SHADES:
            keys += [tuple(int(round(255 * np.round(c * shv * 255) / 255)) for c in cmaps[key](u)[:3]) for u in lev]
    for shv in SHADES:
        keys.append(tuple(int(round(255 * np.round(c * shv * 255) / 255)) for c in empty))
    keys += [(int(round(v)),) * 3 for v in np.linspace(0, 255, 32)]
    pal = np.unique(np.array(keys, np.uint8), axis=0)
    assert len(pal) <= 255, len(pal)
    quant = mm._Exact(pal)
    q = [quant(fr) for fr in frames]
    path = fig_path("fig1c_movie_figure8_learning.gif")
    q[0].save(path, save_all=True, append_images=q[1:], duration=1000 // FPS, loop=0)
    print(f"saved {path}: {len(q)} frames, {os.path.getsize(path) / 1e6:.1f} MB", flush=True)
    return path


# ═══════════════════════════════════════════════════════════════════
# Particles with every measure (John 29 Oct: "show the other relevant measures, like error and effective dim etc.
# ... in particle form")
# ═══════════════════════════════════════════════════════════════════

P_STREAMS = 4                                                                     # particles drawn per cell
P_K = 48                                                                          # neighbours for the local mean
P_STEP_S, P_DPI = .4, 85
P_PANELS = (("process", "Which process", "red / blue: split across a wormhole\ngreen: a top-3% entangled step", None),
            ("rev", "Entanglement", r"$\mathrm{rev}_t$ (log$_{10}$)", .3),
            ("fr", "Factorization regret", r"$\mathrm{FR}_t$, both variables (log$_{10}$)", .3),
            ("err", "Error of the Trained RNN", r"$\delta_t$ vs the Optimal (log$_{10}$)", .6),
            ("pr", "Activity: effective dimension", r"$\mathrm{PR}(u_t)$ (dimensions)", 4.),
            ("turn", "Activity: turn", r"$1 - \cos(u_t, u_{t-1})$ (log$_{10}$)", .12),
            ("bpr", "Beliefs: effective dimension", r"of the RNN's 20 read-out probabilities", .6),
            ("dW", "Update: size", r"$\|\Delta W_t\|^2$ (log$_{10}$)", .7),
            ("band_x", "Update: its pieces", r"cross-band share ($< 0$: they cancel)", .8),
            ("coh", "Update: credit coherence", r"$\|\Sigma_k \delta_k\|^2 / \Sigma_k \|\delta_k\|^2$", .12))
P_ENT = .97                                                                       # 'entangled': top 3% of the step
P_COLORS = {"other": (.82, .82, .82), "plus": (.75, .1, .12), "minus": (.13, .33, .7), "ent": (.1, .62, .25)}


def particle_data():
    """for the drawn particles (4 per cell): positions (n, T, 3), state codes (n, T) (0 other, 1 / 2 split across the
    (0,0)-(9,9) / (0,9)-(9,0) wormhole, 3 a top-10% entangled step), and each measure's mean over the P_K beliefs
    nearest the particle at the same step (all 5,000 streams of all 39 environments; minus typical); cached"""
    from scipy.spatial import cKDTree
    import fig1c_two_processes as tp
    path = os.path.join(tp.OUT, "figure8_particles.npz")
    if os.path.exists(path):
        with np.load(path) as f:
            return {k: f[k] for k in f}
    Z = fw.pooled()
    X = Z["X"].astype(np.float64)
    cov = Z["CV"].astype(np.float64)
    V = tp.series()
    D = {"rev": np.log10(V["rev"]), "fr": np.log10(np.maximum(V["fr"], 1e-3)), "err": np.log10(V["err"]),
         "pr": V["pr"], "turn": np.log10(V["turn"]),
         "bpr": np.load(os.path.join(tp.mm.R7, "belief_pr_Trained.npz"))["belief_pr"].astype(np.float64),
         "dW": np.log10(V["dW"]), "coh": V["coh"], "band_x": V["band_x"]}
    D = {k: tp.dep(v) for k, v in D.items()}
    rk = np.argsort(np.argsort(V["rev"], axis=1), axis=1) / (S - 1)
    rows = (np.arange(R * R)[:, None] * NS + np.arange(P_STREAMS)[None]).ravel()
    pos = X[:, rows]                                                              # (E, n_e, T, 3)
    state = np.zeros(pos.shape[:3], np.int8)
    state[cov[:, rows] > 8] = 1
    state[cov[:, rows] < -8] = 2
    erank = rk[:, rows].astype(np.float32)
    loc = {k: np.full(pos.shape[:3], np.nan, np.float32) for k in D}
    for t in range(1, T):
        tree = cKDTree(X[:, :, t].reshape(-1, 3))
        _, j = tree.query(pos[:, :, t].reshape(-1, 3), k=P_K)
        for k, v in D.items():
            loc[k][:, :, t] = np.nanmean(v[:, :, t].reshape(-1)[j], 1).reshape(pos.shape[:2])
        print(f"local means, step {t + 1}", flush=True)
    perm = np.random.default_rng(0).permutation(pos.shape[0] * pos.shape[1])
    out = {"pos": pos.reshape(-1, T, 3)[perm].astype(np.float32), "state": state.reshape(-1, T)[perm],
           "erank": erank.reshape(-1, T)[perm],
           **{k: v.reshape(-1, T)[perm] for k, v in loc.items()}, "Y": Z["Y"]}
    np.savez(path, **out)
    return out


def movie_particles():
    from main.utils import fig_path
    plt.rcParams.update(mm.RC)
    Pd = particle_data()
    pos, Y = Pd["pos"].astype(np.float64), Pd["Y"]
    state = np.where(Pd["state"] > 0, Pd["state"], np.where(Pd["erank"] >= P_ENT, 3, 0)).astype(np.int8)
    n = len(pos)
    centre = Y.mean(0)
    rad = np.percentile(np.linalg.norm(pos[:, 1:].reshape(-1, 3) - centre, axis=1), 99.7) * 1.04
    cmap = plt.get_cmap("PuOr")                                                   # orange low, purple high
    LV = 25
    lev = np.linspace(0, 1, LV)
    lims = {k: lim for k, _, _, lim in P_PANELS if k != "process"}
    edges = _edges()
    PW, GX, GY, X0, TOP, BOT = 3.0, .12, .75, .12, 1.55, .25
    W_ = X0 + 5 * PW + 4 * GX + .12
    H_ = TOP + 2 * PW + GY + BOT
    fig = plt.figure(figsize=(W_, H_), dpi=P_DPI)
    fig.patch.set_facecolor("white")
    fig.text(.5, 1 - .25 / H_, "Two processes on the figure 8: wormhole splits (red / blue) and entangled steps "
             "(green), "
             "and what the trained network does at each", ha="center", va="center", fontsize=13)
    clock = fig.text(.5, 1 - .55 / H_, "", ha="center", va="center", fontsize=12.5, fontweight="bold")
    fig.text(.5, 1 - .86 / H_, "particles: the Optimal's beliefs (39 environments x 4 streams per cell) in their "
             "common "
             "evidence geometry; colour in b - j: the mean over the 48 beliefs nearest the particle at that step\n"
             "(all 5,000 streams per environment), minus typical (the same cell's 50 streams at that step); purple: "
             "above typical, orange: below", ha="center", va="center", fontsize=9.5, color="0.3", linespacing=1.4)
    scat, lines = [], []
    for i, (key, name, form, _) in enumerate(P_PANELS):
        r_, c_ = divmod(i, 5)
        x, y = X0 + c_ * (PW + GX), H_ - TOP - PW - r_ * (PW + GY)
        ax = fig.add_axes([x / W_, y / H_, PW / W_, PW / H_])
        ax.set_xlim(-rad, rad), ax.set_ylim(-rad, rad), ax.set_aspect("equal"), ax.axis("off")
        lc = LineCollection([], colors="0.8", linewidths=.5, zorder=1)
        ax.add_collection(lc)
        sc = ax.scatter(np.zeros(n), np.zeros(n), s=np.ones(n), c=np.zeros((n, 3)), marker="s", linewidths=0,
                        antialiased=False, zorder=2)
        fig.text((x + PW / 2) / W_, (y + PW + .3) / H_, f"{'abcdefghij'[i]}   {name}", ha="center", va="bottom",
                 fontsize=11.5, fontweight="bold")
        fig.text((x + PW / 2) / W_, (y + PW + .05) / H_, form if key == "process" else
                 f"{form}, $\\pm${lims[key]:g}", ha="center", va="bottom", fontsize=9.5, color="0.3",
                 linespacing=1.15)
        scat.append(sc), lines.append(lc)
    taus = ([float(v) for v in os.environ["TEST"].split(",")] if os.environ.get("TEST") else
            list(np.linspace(1, T - 1, int((T - 2) * P_STEP_S * FPS) + 1)) + [T - 1.] * int(1.2 * FPS))
    frames = []
    col_state = np.array([P_COLORS["other"], P_COLORS["plus"], P_COLORS["minus"], P_COLORS["ent"]])
    for f_, tau in enumerate(taus):
        ph = 2 * np.pi * f_ / max(len(taus), 1)
        Pv = _view(ELEV + 8 * np.sin(ph), AZIM + SWING * np.sin(ph))
        p3 = mm._blend(pos, tau, axis=1) - centre
        xy = p3 @ Pv
        s = int(np.clip(np.floor(tau), 1, T - 2))
        fr = tau - s
        st = state[:, min(int(round(tau)), T - 1)]
        Yp = (Y - centre) @ Pv
        seg = np.stack([Yp[[a for a, _ in edges]], Yp[[b for _, b in edges]]], 1)
        for i, (key, *_) in enumerate(P_PANELS):
            if key == "process":
                col = col_state[st]
                sz = np.where(st > 0, 9., 2.2) * (72 / P_DPI) ** 2
                order = np.argsort(st > 0, kind="stable")                        # marked particles on top
                scat[i].set_offsets(xy[order]), scat[i].set_facecolors(col[order]), scat[i].set_sizes(sz[order])
            else:
                v = Pd[key]
                vt = (1 - fr) * v[:, s] + fr * v[:, min(s + 1, T - 1)]
                lim = lims[key]
                q = np.round((np.clip(np.nan_to_num(vt), -lim, lim) + lim) / (2 * lim) * (LV - 1)).astype(int)
                scat[i].set_offsets(xy)
                scat[i].set_facecolors(cmap(lev[q])[:, :3])
                scat[i].set_sizes(np.full(n, 2.8 * (72 / P_DPI) ** 2))
            lines[i].set_segments(seg)
        clock.set_text(f"step {min(int(round(tau)) + 1, T)} of {T}")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        if f_ % 50 == 0:
            print(f"frame {f_} / {len(taus)}", flush=True)
    plt.close(fig)
    if os.environ.get("TEST"):
        out = os.environ.get("TEST_DIR", ".")
        for tau, fr_ in zip(taus, frames):
            fr_.save(os.path.join(out, f"part_frame_{tau:04.1f}.png"))
        return None
    keys = [tuple(int(round(255 * c)) for c in cmap(u)[:3]) for u in lev]
    keys += [tuple(int(round(255 * c)) for c in v) for v in P_COLORS.values()]
    keys += [(int(round(v)),) * 3 for v in np.linspace(0, 255, 48)]
    pal = np.unique(np.array(keys, np.uint8), axis=0)
    assert len(pal) <= 255, len(pal)
    quant = mm._Exact(pal)
    q = [quant(fr) for fr in frames]
    path = fig_path("fig1c_movie_figure8_particles.gif")
    q[0].save(path, save_all=True, append_images=q[1:], duration=1000 // FPS, loop=0)
    print(f"saved {path}: {len(q)} frames, {os.path.getsize(path) / 1e6:.1f} MB", flush=True)
    return path


# ═══════════════════════════════════════════════════════════════════
# The two processes as particles (John 29 Oct: "show the other relevant measures, like error and effective dim etc.
# ... in particle form ... can we conclude that the wormhole corner splitting and entanglement representation
# unfolding are different processes? Or the same process at different steps")
# ═══════════════════════════════════════════════════════════════════

Q_POOL, Q_BACK = 10, 3                                                            # streams per cell: coloured / grey
Q_ENT = .97                                                                       # entangled: top 3% of env x step
Q_STEP_S, Q_DPI = .4, 80
Q_MEASURES = (("err", "Error of the RNN", r"vs the Optimal (log$_{10}$)", .8),
              ("dW", "Update: size", r"$\|\Delta W_t\|^2$ (log$_{10}$)", 1.),
              ("band_x", "Update: its pieces", r"cross-band share ($< 0$: cancel)", 1.5),
              ("coh", "Update: coherence", r"$\|\Sigma_k \delta_k\|^2 / \Sigma_k \|\delta_k\|^2$", .3),
              ("pr", "Activity: eff. dim.", r"$\mathrm{PR}(u_t)$ (dimensions)", 8.),
              ("turn", "Activity: turn", r"$1 - \cos(u_t, u_{t-1})$ (log$_{10}$)", .3))


def process_data():
    """the pool (Q_POOL streams per cell, 39 environments): positions (n, T, 3), Cov (n, T), entanglement rank within
    environment x step (n, T), each measure minus typical (n, T), raw per stream-step; cached"""
    import fig1c_two_processes as tp
    path = os.path.join(tp.OUT, "figure8_processes.npz")
    if os.path.exists(path):
        with np.load(path) as f:
            return {k: f[k] for k in f}
    Z = fw.pooled()
    V = tp.series()
    rows = (np.arange(R * R)[:, None] * NS + np.arange(Q_POOL)[None]).ravel()
    rk = np.argsort(np.argsort(V["rev"], axis=1), axis=1) / (S - 1)
    D = {"err": np.log10(V["err"]), "dW": np.log10(V["dW"]), "band_x": V["band_x"], "coh": V["coh"], "pr": V["pr"],
         "turn": np.log10(V["turn"])}
    out = {k: tp.dep(v)[:, rows].reshape(-1, T).astype(np.float32) for k, v in D.items()}
    out["pos"] = Z["X"][:, rows].reshape(-1, T, 3).astype(np.float32)
    out["cov"] = Z["CV"][:, rows].reshape(-1, T).astype(np.float32)
    out["erank"] = rk[:, rows].reshape(-1, T).astype(np.float32)
    out["back"] = np.tile(np.isin(np.arange(R * R * Q_POOL) % Q_POOL, np.arange(Q_BACK)), E)
    out["Y"] = Z["Y"]
    np.savez(path, **out)
    return out


def movie_processes():
    from main.utils import fig_path
    plt.rcParams.update(mm.RC)
    Pd = process_data()
    pos, cov, er, Y = Pd["pos"].astype(np.float64), Pd["cov"], Pd["erank"], Pd["Y"]
    back = Pd["back"]
    worm = np.abs(cov) > 8
    ent = (er >= Q_ENT) & ~worm
    centre = Y.mean(0)
    rad = np.percentile(np.linalg.norm(pos[:, 1:].reshape(-1, 3) - centre, axis=1), 99.7) * 1.04
    cmap = plt.get_cmap("PuOr")                                                   # orange low, purple high
    LV = 25
    lev = np.linspace(0, 1, LV)
    edges = _edges()
    PW, GX, GY, X0, TOP, BOT = 2.35, .1, .7, 1.15, 1.6, .3
    nc = 1 + len(Q_MEASURES)
    W_ = X0 + nc * PW + (nc - 1) * GX + .1
    H_ = TOP + 2 * PW + GY + BOT
    fig = plt.figure(figsize=(W_, H_), dpi=Q_DPI)
    fig.patch.set_facecolor("white")
    fig.text(.5, 1 - .25 / H_, "Two different processes: beliefs split across a wormhole (top) and the most entangled "
             "steps (bottom), and what the trained network does at each", ha="center", va="center", fontsize=13.5)
    clock = fig.text(.5, 1 - .56 / H_, "", ha="center", va="center", fontsize=13, fontweight="bold")
    fig.text(.5, 1 - .85 / H_, "particles: the Optimal's beliefs in their common evidence geometry (grey: 3 streams "
             "per "
             "cell of 39 environments); coloured: 10 streams per cell, each by its own value at that step, minus "
             "typical (the same cell's 50 streams);\npurple: above typical, orange: below; colour ranges in the "
             "titles", ha="center", va="center", fontsize=9.5, color="0.3", linespacing=1.4)
    rows_ = (("split across a wormhole\n(|Cov| > 8)", worm, "#7f2704"),
             ("a top-3% entangled step\n(not a wormhole)", ent, "#1a7837"))
    for r_, (lab, _, col) in enumerate(rows_):
        y = H_ - TOP - PW - r_ * (PW + GY)
        fig.text(.55 / W_, (y + PW / 2) / H_, lab.replace("|Cov| > 8", r"$|\mathrm{Cov}| > 8$"), ha="center",
                 va="center", fontsize=11, fontweight="bold", color=col, rotation=90, linespacing=1.3)
    axes = []
    for r_ in range(2):
        for c_ in range(nc):
            x, y = X0 + c_ * (PW + GX), H_ - TOP - PW - r_ * (PW + GY)
            ax = fig.add_axes([x / W_, y / H_, PW / W_, PW / H_])
            ax.set_xlim(-rad, rad), ax.set_ylim(-rad, rad), ax.set_aspect("equal"), ax.axis("off")
            lc = LineCollection([], colors="0.82", linewidths=.5, zorder=1)
            ax.add_collection(lc)
            bg = ax.scatter(np.zeros(back.sum()), np.zeros(back.sum()), s=(1.3 * 72 / Q_DPI) ** 2, c=[(.86,) * 3],
                            marker="s", linewidths=0, antialiased=False, zorder=2)
            fg = ax.scatter([], [], s=(3.2 * 72 / Q_DPI) ** 2, marker="s", linewidths=0, antialiased=False, zorder=3)
            if r_ == 0:
                if c_ == 0:
                    name, form = "Which beliefs", "coloured: the beliefs of the row"
                else:
                    key, name, form, lim = Q_MEASURES[c_ - 1]
                    form = f"{form}, $\\pm${lim:g}"
                fig.text((x + PW / 2) / W_, (y + PW + .3) / H_, name, ha="center", va="bottom", fontsize=11,
                         fontweight="bold")
                fig.text((x + PW / 2) / W_, (y + PW + .08) / H_, form, ha="center", va="bottom", fontsize=9,
                         color="0.3")
            axes.append((ax, lc, bg, fg, r_, c_))
    taus = ([float(v) for v in os.environ["TEST"].split(",")] if os.environ.get("TEST") else
            list(np.linspace(1, T - 1, int((T - 2) * Q_STEP_S * FPS) + 1)) + [T - 1.] * int(1.2 * FPS))
    frames = []
    for f_, tau in enumerate(taus):
        ph = 2 * np.pi * f_ / max(len(taus), 1)
        Pv = _view(ELEV + 8 * np.sin(ph), AZIM + SWING * np.sin(ph))
        xy = (mm._blend(pos, tau, axis=1) - centre) @ Pv
        s = int(np.clip(np.floor(tau), 1, T - 2))
        fr = tau - s
        tn = min(int(round(tau)), T - 1)
        Yp = (Y - centre) @ Pv
        seg = np.stack([Yp[[a for a, _ in edges]], Yp[[b for _, b in edges]]], 1)
        sel = (worm[:, tn], ent[:, tn])
        for ax, lc, bg, fg, r_, c_ in axes:
            lc.set_segments(seg)
            bg.set_offsets(xy[back])
            m = sel[r_]
            if c_ == 0:
                col = np.where((cov[m, tn] > 0)[:, None], [[.75, .1, .12]], [[.13, .33, .7]]) if r_ == 0 else \
                    np.tile([[.1, .62, .25]], (m.sum(), 1))
            else:
                key, _, _, lim = Q_MEASURES[c_ - 1]
                v = Pd[key][m]
                vt = (1 - fr) * v[:, s] + fr * v[:, min(s + 1, T - 1)]
                q = np.round((np.clip(np.nan_to_num(vt), -lim, lim) + lim) / (2 * lim) * (LV - 1)).astype(int)
                col = cmap(lev[q])[:, :3]
            fg.set_offsets(xy[m] if m.any() else np.zeros((0, 2)))
            fg.set_facecolors(col if m.any() else np.zeros((0, 3)))
        clock.set_text(f"step {tn + 1} of {T}:  {int(worm[:, tn].sum()):,} beliefs split across a wormhole, "
                       f"{int(ent[:, tn].sum()):,} at a top-3% entangled step")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        if f_ % 50 == 0:
            print(f"frame {f_} / {len(taus)}", flush=True)
    plt.close(fig)
    if os.environ.get("TEST"):
        out = os.environ.get("TEST_DIR", ".")
        for tau, fr_ in zip(taus, frames):
            fr_.save(os.path.join(out, f"proc_frame_{tau:04.1f}.png"))
        return None
    keys = [tuple(int(round(255 * c)) for c in cmap(u)[:3]) for u in lev]
    keys += [tuple(int(round(255 * c)) for c in v) for v in ((.75, .1, .12), (.13, .33, .7), (.1, .62, .25))]
    keys += [tuple(int(round(255 * c)) for c in mpl_hex) for mpl_hex in
             (plt.matplotlib.colors.to_rgb("#7f2704"), plt.matplotlib.colors.to_rgb("#1a7837"))]
    keys += [(int(round(v)),) * 3 for v in np.linspace(0, 255, 48)]
    pal = np.unique(np.array(keys, np.uint8), axis=0)
    assert len(pal) <= 255, len(pal)
    quant = mm._Exact(pal)
    q = [quant(fr) for fr in frames]
    path = fig_path("fig1c_movie_figure8_processes.gif")
    q[0].save(path, save_all=True, append_images=q[1:], duration=1000 // FPS, loop=0)
    print(f"saved {path}: {len(q)} frames, {os.path.getsize(path) / 1e6:.1f} MB", flush=True)
    return path


if __name__ == "__main__":
    {"particles_raw": movie, "surface": movie_surface,
     "particles": movie_particles, "processes": movie_processes}[sys.argv[1] if len(sys.argv) > 1 else "surface"]()
