"""Figure 1 (c) as a movie: each agent's beliefs at every step t = 1 .. T = 30, with the mean belief trajectories of
(c)'s four cells growing through time. From the repository root, as fig1.py (eureka environment, CPU):

    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \
        main/REPLICATION/claude_plots/fig1_movie.py [fixed] [bridge] [pcs] [env] [env_ent] [basis]   (default: all)

    fixed   -> figure_bin/fig1c_movie_fixed.gif: every step in the agent's own T = 30 plane ((c)'s right panels): one
               projection for the whole movie; the last frame is (c)'s T = 30 panels
    bridge  -> figure_bin/fig1c_movie_bridge.gif: (c)'s own course: the first frame is (c)'s t = 1 panel (its box
               magnified onto the T = 30 panel's, as (c) draws both at one size), the projection slides into the T = 30
               plane by t = BRIDGE, and the curves pass through (c)'s curves at every step
    pcs     -> figure_bin/fig1c_movie_pcs.gif: every step in its own plane by (c)'s rule (the 2 of the step's PCs 1-3
               holding the most latent variance, turned as (c) turns them), sliding from step to step; the first and
               last frames are (c)'s t = 1 and T = 30 clouds; under each panel, the latent variance of each step's PCs
               1-3 and the pair shown
    env     -> figure_bin/fig1c_movie_env.gif: the particles of ONE environment (alias scan environment ENV, a typical
               one): all 100 cells x 50 streams, each stream a dot coloured by its true cell, each agent in its fixed
               T = 30 plane; within a cell the streams differ only by the sampled observations
    env_ent -> figure_bin/fig1c_movie_env_entanglement.gif: the same particles coloured by the entanglement of the step
               being taken (its percentile among the environment's 5000 streams at that step, which removes the
               entanglement's rise over the first steps); the same streams in all four panels
    basis   -> figure_bin/fig1c_pc_basis.png: the per-step planes over t: the latent variance and the variance of each
               step's PCs 1-5, and the loadings of the plane's two axes

Timing: no holds between steps. The beliefs (and the pcs planes) are interpolated between steps by Catmull-Rom splines
(they pass through every step's values, with continuous speed); the clock runs as sqrt(t) (time per step ~ 1 /
sqrt(t + 1): beliefs move most early on, ~12% of the panel from t = 1 to 2, <1% late), eased in from rest.
Inputs: fig1's caches (beliefs.npz through belief_geometry's planes, the alias cache, the targets) and
fig1_checks' beliefs_alltime.npz (the networks at every step); the Bayes observers at every step from the
trained harvest; (env) the alias scan main/DATA/experimental/fig1c_compression (its README.txt and build_scan.py)."""
import os
import sys
import numpy as np
from PIL import Image
from matplotlib.colors import to_rgb


def _here():
    """this file's folder, also where VS Code's interactive window gives no usable __file__ (then from the repository
    holding the notebook's file or the working directory)"""
    starts = (globals().get("__file__"), globals().get("__vsc_ipynb_file__"), os.getcwd())
    for f in starts[:2]:
        if f and os.path.exists(f) and os.path.basename(f) == "fig1_movie.py":
            return os.path.dirname(os.path.abspath(f))
    for s in starts:
        d = os.path.abspath(s) if s and os.path.exists(s) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return os.path.join(d, "main", "REPLICATION", "claude_plots")
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError("project root not found")


sys.path.insert(0, _here())
import fig1 as cf                                                          # noqa: E402
from fig1 import (ga, plt, pe, Line2D, LineCollection, PCA, Ridge, N_EP, NAME, COL, MARKS,  # noqa: E402
                         MARK_SCALE)
from fig1_checks import beliefs_alltime                                    # noqa: E402

T = ga.CFG["step_num"]
MODES = ("fixed", "bridge", "pcs", "env", "env_ent")
AGENTS = (("Trained", "Echo"), ("Joint", "Naive"))                                # (c)'s 2 x 2 order
W_IN, DPI = 7.05, 150                                                             # the movie's width (in), resolution
BS, X0, XGAP, YNAME, YTOP = 2.4, .15, .25, .44, .55                               # panel side, left edge, column gap,
                                                                                  # names' band, clock band
STRIP, STRIP_GAP = .30, .30                                                       # (pcs) the PC strip under a panel
LEG_X = X0 + 2 * BS + XGAP + .32                                                  # the key column's left edge
FS = dict(t=13, name=11, lab=9, note=7.5, tick=7.5)
SCALE = BS / 1.11                                                                 # this panel to (c)'s (1.11 in)
S_PT = .12 * SCALE ** 2                                                           # (c)'s point area, scaled
PX = 1                                                                            # cloud points as PX x PX pixel squares
                                                                                  # without anti-aliasing (0: (c)'s dots;
                                                                                  # anti-aliased sub-pixel motion changes
                                                                                  # every pixel: GIFs 2.5x larger)
MS_START, MS_END, MS_HOME = 9, 10.5, 18                                           # marker sizes (pt), as (c)'s
SCAN = os.path.join(cf.ROOT, "main", "DATA", "experimental", "fig1c_compression")  # (env) the alias scan
ENT_CMAP = plt.matplotlib.colors.LinearSegmentedColormap.from_list(               # (env_ent) low -> high entanglement
    "ent", ["#c8c8c8", "#fdbb84", "#e34a33", "#7f0000"])
ENV = 34                                                                          # (env) its environment nearest the
                                                                                  # median of the 40 on the noise / grid
                                                                                  # statistics (panel (c)'s, e = 29, is
                                                                                  # extreme)
SUB = 8                                                                           # curve points per step
FPS, MOTION_S, EASE_S, HOLD_S = 25, 15., .7, (.5, 1.2)                            # frames per s; the motion's length,
                                                                                  # its ease-in; holds at t = 1, T
ACCENT = "#d62728"                                                                # (pcs) the pair shown, the cursor
RC = dict(cf.RC, **{"font.size": FS["lab"], "axes.labelsize": FS["lab"], "xtick.labelsize": FS["tick"],
                    "ytick.labelsize": FS["tick"]})


def _height(mode):
    return YTOP + 2 * (YNAME + BS + (STRIP + STRIP_GAP if mode == "pcs" else 0)) + .12


# ═══════════════════════════════════════════════════════════════════
# Data: beliefs at every step, and the planes they are seen in
# ═══════════════════════════════════════════════════════════════════

def beliefs_every_step():
    """each agent's goal-ordered beliefs (N_EP, T, 20) at every step on the trained harvest's first N_EP episodes, as
    belief_geometry orders them (the networks from beliefs_alltime, the ESN replayed on the trained network's
    episodes; the Bayes observers from the harvest), and each episode's r_g, r_c"""
    BA = beliefs_alltime()
    f = np.load(ga.CACHE.format("trained"))
    goal, ctx = f["ep_goal_ind"][:N_EP].astype(int), f["ep_ctx_vals"][:N_EP].astype(int)
    ar = np.arange(N_EP)
    out = {"Trained": BA["Trained"], "Echo": BA["Echo"]}
    for a, k in (("Joint", "ep_joint_belief"), ("Naive", "ep_naive_belief")):
        b = f[k][:N_EP]                                                           # (N, T, contexts, realizations)
        out[a] = np.concatenate([b[ar, :, goal], b[ar, :, 1 - goal]], -1).astype(np.float32)
    return out, ctx[ar, goal], ctx[ar, 1 - goal]


def step_planes(V, rg, rc):
    """each step's plane by (c)'s rule (belief_geometry): the PCA of that step's beliefs V[:, t], the 2 of PCs 1-3
    holding the most latent variance (single-PC R^2 of r_g plus r_c), the ridge decoders (alpha 1) of r_g and r_c from
    them, the plane turned by _align -> mean m (T, 20), orthonormal axes U (T, 20, 2) (a belief v is drawn at
    (v - m) U), pcs (T, 2) 1-based, lat (T, 5) single-PC latent R^2 of PCs 1-5, var (T, 5) their variance, r2 (T, 2)
    the plane's decoder R^2 of r_g, r_c"""
    S = {k: [] for k in ("m", "U", "pcs", "lat", "var", "r2")}
    for t in range(V.shape[1]):
        X = np.asarray(V[:, t], np.float64)
        pca = PCA(n_components=5).fit(X)
        P = pca.transform(X)
        lat = np.array([sum(np.corrcoef(P[:, q], y)[0, 1] ** 2 for y in (rg, rc)) for q in range(5)])
        sel = sorted(int(q) for q in np.argsort(lat[:3])[-2:])
        fits = [Ridge(alpha=1.0).fit(P[:, sel], y) for y in (rg, rc)]
        Q = cf._align(np.eye(2), fits[0].coef_, fits[1].coef_)[0]
        for k, v in (("m", pca.mean_), ("U", pca.components_[sel].T @ Q), ("pcs", [q + 1 for q in sel]), ("lat", lat),
                     ("var", pca.explained_variance_), ("r2", [m_.score(P[:, sel], y) for m_, y in zip(fits, (rg, rc))])):
            S[k].append(v)
    return {k: np.array(v) for k, v in S.items()}


def _proj(D, V):
    """beliefs V (..., 20) into a plane of belief_geometry, turned as (c) draws it: (v - m) C^T Q"""
    m, C, Q = D["proj"]
    return (np.asarray(V, np.float64) - m) @ C.T @ Q


def _box(D):
    """the data box (c)'s 1.11 in square panel shows for the cloud D (equal aspect, the data limits adjusted; within
    matplotlib's 0.5% tolerance it leaves them, so a side can differ by ~0.2%), as (centre, sides)"""
    fig = plt.figure(figsize=(1.11, 1.11))
    ax = fig.add_axes([0, 0, 1, 1])
    cf._geometry_panel(ax, D)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    plt.close(fig)
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2]), np.array([x1 - x0, y1 - y0])


def _cr(tau):
    """Catmull-Rom spline through the steps at (continuous) step index tau: the 4 neighbouring steps (clamped at the
    ends) and their weights (they sum to 1; the curve passes through every step, its slope is continuous)"""
    i = min(int(np.floor(tau)), T - 2)
    f = tau - i
    idx = np.clip(np.arange(i - 1, i + 3), 0, T - 1)
    w = .5 * np.array([-f ** 3 + 2 * f ** 2 - f, 3 * f ** 3 - 5 * f ** 2 + 2, -3 * f ** 3 + 4 * f ** 2 + f, f ** 3 - f ** 2])
    return idx, w


def _blend(X, tau, axis=0):
    """X (..., T, ...) at tau along axis (the steps), by _cr"""
    idx, w = _cr(tau)
    return np.tensordot(np.take(X, idx, axis=axis).astype(np.float64), w, axes=([axis], [0]))


def _polar(M):
    """the orthonormal factor of M (20, 2): the nearest matrix with orthonormal columns"""
    u, _, vt = np.linalg.svd(M, full_matrices=False)
    return u @ vt


class Course:
    """where beliefs are drawn at (continuous) step index tau, 0 .. T - 1 (t = tau + 1), for one agent, by mode. fixed:
    its T = 30 plane at every step. bridge: as (c)'s curves, (1 - w) A(P_1 v) + w P_30 v with w = smoothstep(tau /
    (BRIDGE - 1)), where A maps (c)'s t = 1 panel box onto its T = 30 panel box (the panels' common size, as (c)'s
    curves cross from one panel to the other). pcs: the step's own plane (step_planes S); between steps the
    Catmull-Rom blend of the planes' means and axes, the axes made orthonormal again (_polar), so the drawn plane never
    shrinks a distance"""

    def __init__(self, G, agent, mode, S=None):
        self.D1, self.D30, self.mode, self.S = G[(agent, 1)], G[(agent, T)], mode, S
        (c1, s1), (c30, s30) = _box(self.D1), _box(self.D30)
        self.A = lambda Y: c30 + (Y - c1) * (s30 / s1)

    def w(self, tau):
        if self.mode == "fixed":
            return np.ones_like(np.asarray(tau, np.float64))
        s = np.clip(np.asarray(tau, np.float64) / (cf.BRIDGE - 1), 0, 1)
        return s * s * (3 - 2 * s)

    def planes(self, V):
        """V (..., 20) -> its positions in the two planes (A applied to the first), (..., 2) each"""
        return self.A(_proj(self.D1, V)), _proj(self.D30, V)

    def proj(self, V, tau):
        """beliefs V (..., 20) drawn with the projection at tau -> (..., 2)"""
        if self.mode == "pcs":
            return (np.asarray(V, np.float64) - _blend(self.S["m"], tau)) @ _polar(_blend(self.S["U"], tau))
        P1, P30 = self.planes(V)
        w = float(self.w(tau))
        return (1 - w) * P1 + w * P30

    def at(self, Y1, Y30, tau):
        """(fixed, bridge; kept for analyses) positions at tau from per-step plane positions Y (n, T, 2): linear
        between steps, then blended"""
        i0 = int(np.floor(tau))
        i1, f = min(i0 + 1, T - 1), tau - i0
        lerp = lambda Y: (1 - f) * Y[:, i0] + f * Y[:, i1]
        w = float(self.w(tau))
        return (1 - w) * lerp(Y1) + w * lerp(Y30)

    def path(self, M):
        """a mean belief trajectory M (T, 20) at SUB points per step, each point drawn with its own step's projection
        (as (c)'s curves) -> tt (step indices), (len(tt), 2)"""
        tt = np.linspace(0, T - 1, (T - 1) * SUB + 1)
        return tt, np.stack([self.proj(_blend(M, tau), tau) for tau in tt])


def movie_data(mode):
    """everything a frame needs, per agent: the beliefs at every step (N, T, 20), their colours, the Course, the four
    cells' mean belief paths, the targets (20-dim, drawn with the frame's projection), the step planes (pcs, basis)"""
    G = cf.belief_geometry()
    BV, rg, rc = beliefs_every_step()
    AV, a_of, cells, _ = cf.alias_trajectories()
    home_v, _ = cf.targets(cells)
    out = {}
    for a in BV:
        S = step_planes(BV[a], rg, rc) if mode in ("pcs", "basis") else None
        C = Course(G, a, mode if mode != "basis" else "pcs", S)
        ends = {"fixed": ((T - 1, G[(a, T)]["X"]),), "bridge": ((0, C.A(G[(a, 1)]["X"])), (T - 1, G[(a, T)]["X"])),
                "pcs": ((0, G[(a, 1)]["X"]), (T - 1, G[(a, T)]["X"]))}[C.mode]
        for tau, X in ends:                                                       # the first / last frames are (c)'s
            assert np.abs(C.proj(BV[a][:, tau], tau) - X).max() < 2e-3, f"{a}: step {tau + 1} is not (c)'s cloud"
        tt = np.linspace(0, T - 1, (T - 1) * SUB + 1)
        M = [np.stack([_blend(AV[a][a_of == j].mean(0), tau) for tau in tt]) for j in range(len(cells))]
        out[a] = dict(C=C, V=BV[a], col=cf.bivariate(rg, rc), home=home_v, S=S, tt=tt, M=M,
                      paths=[(tt, np.stack([C.proj(v, tau) for v, tau in zip(M_, tt)])) for M_ in M])
    return out, cells


def env_data(e=ENV, ent=False):
    """(env) the scan's environment e at every step: per agent its 100 cells x 50 streams' beliefs (5000, T, 20),
    float16 in the scan, each in its fixed T = 30 plane and coloured by the stream's true cell; no curves or targets.
    The scan stores streams cell by cell, so they are drawn in a fixed random order (stacked dots show a random cell's
    colour, not always the last cell's). With ent, each stream's entanglement percentile at every step (the house
    entanglement of the scan, ranked among the environment's streams within the step), (5000, T)"""
    G = cf.belief_geometry()
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    perm = np.random.default_rng(0).permutation(len(meta["cell_of"]))
    cell = meta["cells"][meta["cell_of"][perm]]
    rev = meta["rev"][e][perm]
    pct = np.argsort(np.argsort(rev, 0), 0) / (len(rev) - 1)
    return {a: dict(C=Course(G, a, "fixed"), col=cf.bivariate(cell[:, 0], cell[:, 1]), home=np.zeros((0, 20)), paths=[],
                    S=None, V=np.load(os.path.join(SCAN, f"scan_{a}.npz"))["beliefs"][e][perm].astype(np.float32),
                    **({"pct": pct} if ent else {}))
            for a in ("Trained", "Echo", "Joint", "Naive")}


def trails(D, tau):
    """the four cells' mean belief paths as drawn at frame tau -> [(tt, (len(tt), 2))]: fixed, bridge: each point with
    its own step's projection (as (c)'s curves; the same in every frame); pcs: the whole path seen in the frame's own
    plane (a trace through the changing planes would draw the planes' turn as motion)"""
    if D["C"].mode == "pcs":
        return [(D["tt"], D["C"].proj(M_, tau)) for M_ in D["M"]]
    return D["paths"]


def limits(D, taus):
    """one square data box per agent holding every frame's cloud, the curves and the targets (with a margin)"""
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for tau in taus:
        pts = np.concatenate([D["C"].proj(_blend(D["V"], tau, axis=1), tau), D["C"].proj(D["home"], tau)]
                             + [P for _, P in trails(D, tau)])
        lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
    c, s = (lo + hi) / 2, (hi - lo).max() * 1.08
    return (c[0] - s / 2, c[0] + s / 2), (c[1] - s / 2, c[1] + s / 2)


def schedule():
    """(tau, ms) per frame at FPS: a short hold at t = 1; the steps with dtau/ds ~ sqrt(tau + 2) (time per step ~ 1 /
    sqrt(t + 1)), the clock eased in from rest over EASE_S (smoothstep speed); a short hold at T"""
    s = np.arange(int(round(MOTION_S * FPS)) + 1) / FPS
    v = np.clip(s / EASE_S, 0, 1)
    u = np.concatenate([[0], np.cumsum((v * v * (3 - 2 * v))[1:])])
    u *= 2 * (np.sqrt(T + 1) - np.sqrt(2)) / u[-1]                                # sqrt(tau + 2) = sqrt(2) + u / 2
    tau = np.clip((np.sqrt(2) + u / 2) ** 2 - 2, 0, T - 1)
    ms = np.full(len(tau), 1000 / FPS)
    ms[0], ms[-1] = 1000 * HOLD_S[0], 1000 * HOLD_S[1]
    return list(zip(tau.tolist(), ms.tolist()))


# ═══════════════════════════════════════════════════════════════════
# The movie
# ═══════════════════════════════════════════════════════════════════

def _tail(tt, P, tau):
    """the curve up to tau, in (c)'s style: segments, widths and greys in log t (light and thin at t = 1, black and
    thick at T, fixed in absolute time, so the last frame is (c)'s curve)"""
    k = np.searchsorted(tt, tau, side="right")                                    # the curve's points up to tau
    if k >= len(tt):
        pts, ts = P, tt
    else:                                                                         # and the head between two of them
        f = (tau - tt[k - 1]) / (tt[k] - tt[k - 1])
        pts, ts = np.concatenate([P[:k], (P[k - 1] + f * (P[k] - P[k - 1]))[None]]), np.append(tt[:k], tau)
    e = np.log1p(ts) / np.log(T)
    return np.stack([pts[:-1], pts[1:]], 1), (e[1:] + e[:-1]) / 2, pts[-1]


def _strip(fig, x, y, S):
    """(pcs) the step planes under a panel whose lower left is (x, y): the single-PC latent R^2 (r_g + r_c) of each
    step's PCs 1-3 (white 0 -> black 1), the pair shown dotted, a cursor at the frame's step -> the cursor"""
    H = fig.get_figheight()
    ax = fig.add_axes([(x + .32) / W_IN, (y - .07 - STRIP) / H, (BS - .32) / W_IN, STRIP / H])
    ax.imshow(S["lat"][:, :3].T, cmap="Greys", vmin=0, vmax=1, aspect="auto", interpolation="nearest",
              extent=(.5, T + .5, 3.5, .5))
    ax.scatter(np.repeat(np.arange(1, T + 1), 2), S["pcs"].ravel(), s=2.5, c=ACCENT, lw=0, zorder=4)
    ax.set_xlim(.5, T + .5), ax.set_ylim(3.5, .5)
    ax.set_yticks([1, 2, 3], ["PC1", "PC2", "PC3"])
    ax.set_xticks([1, 10, 20, 30])
    ax.tick_params(length=1.5, pad=1, labelsize=FS["tick"] - 1)
    for s_ in ax.spines.values():
        s_.set_linewidth(.5)
    return ax.axvline(1, c="k", lw=.9, zorder=5)


def build(mode, data, cells):
    """the figure and its moving artists -> fig, update(tau)"""
    taus = [tau for tau, _ in schedule()][::6] + [T - 1.]
    H = _height(mode)
    fig = plt.figure(figsize=(W_IN, H), dpi=DPI, facecolor="white")
    ax_ = lambda x, y, w, h: fig.add_axes([x / W_IN, y / H, w / W_IN, h / H])
    moving, cursors, notes = [], [], []
    row_h = YNAME + BS + (STRIP + STRIP_GAP if mode == "pcs" else 0)
    for r, agents in enumerate(AGENTS):
        y = H - YTOP - YNAME - BS - r * row_h
        for p, a in enumerate(agents):
            D = data[a]
            x = X0 + p * (BS + XGAP)
            ax = ax_(x, y, BS, BS)
            xl, yl = limits(D, taus)
            ax.set_xlim(*xl), ax.set_ylim(*yl)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            fig.text((x + BS / 2) / W_IN, (y + BS + .22) / H, NAME[a], ha="center", va="baseline",
                     fontsize=FS["name"], color=COL[a])
            p1, p30 = D["C"].D1["pcs"], D["C"].D30["pcs"]
            plane = fig.text((x + BS / 2) / W_IN, (y + BS + .07) / H, "$T = 30$ PCs %d-%d" % p30 if mode in ("fixed", "env", "env_ent")
                             else "$t = 1$ PCs %d-%d $\\to$ $T = 30$ PCs %d-%d" % (p1 + p30), ha="center",
                             va="baseline", fontsize=FS["note"], color="0.45")
            if mode == "pcs":
                cursors.append(_strip(fig, x, y, D["S"]))
                notes.append((plane, D["S"]["pcs"]))
            px = 2 if mode.startswith("env") else PX                              # (env: 5000 dots, not 20,000)
            dots = dict(marker="s", s=(px * 72 / DPI) ** 2, antialiased=False) if px else dict(s=S_PT)
            sc = ax.scatter(*D["C"].proj(D["V"][:, 0], 0.).T, lw=0, zorder=1, **dots,
                            c=ENT_CMAP(D["pct"][:, 0]) if "pct" in D else D["col"])
            homes, halos, lines, starts, heads = [], [], [], [], []
            for j, (rg_, rc_) in enumerate(cells):
                mk = MARKS[j]
                sf = MARK_SCALE.get(mk, 1)
                homes.append(ax.plot([], [], marker=mk, ms=MS_HOME * sf, mfc="white", mec="k", mew=1.1, ls="none",
                                     zorder=6.8, clip_on=False,
                                     path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])[0])
                halos.append(ax.add_collection(LineCollection([], colors="white", capstyle="round", zorder=6.9)))
                lines.append(ax.add_collection(LineCollection([], capstyle="round", zorder=7)))
                starts.append(ax.plot([], [], marker=mk, ms=MS_START * sf, mfc="white", mec="k", mew=.8, ls="none",
                                      zorder=8 if sf > 1 else 8.1, clip_on=False)[0])
                heads.append(ax.plot([], [], marker=mk, ms=MS_END * sf, mfc=cf.bivariate(rg_, rc_), mec="k", mew=.8,
                                     ls="none", zorder=8.2 if sf > 1 else 8.3, clip_on=False)[0])
            moving.append((D, sc, homes, halos, lines, starts, heads))
    # ── the clock: t over a bar of the T steps ──
    ty = H - .33
    t_txt = fig.text(X0 / W_IN, ty / H, "", fontsize=FS["t"], ha="left", va="center")
    bx0, bx1 = X0 + .95, X0 + 2 * BS + XGAP
    bar = ax_(bx0, ty - .05, bx1 - bx0, .10)
    bar.set_xlim(.5, T + .5), bar.set_ylim(0, 1)
    bar.axis("off")
    bar.barh(.5, T, left=.5, height=1, color="0.92", lw=0)
    fill = bar.barh(.5, 0, left=.5, height=1, color="0.35", lw=0)[0]
    for t_ in (1, T):
        fig.text((bx0 + (t_ - .5) / T * (bx1 - bx0)) / W_IN, (ty - .2) / H, f"${t_}$", ha="center", va="baseline",
                 fontsize=FS["tick"], color="0.4")
    _key(fig, cells, mode, H)

    def update(tau):
        for D, sc, homes, halos, lines, starts, heads in moving:
            C = D["C"]
            sc.set_offsets(C.proj(_blend(D["V"], tau, axis=1), tau))
            if "pct" in D:                                                        # the step being taken (t = 1: the
                sc.set_facecolor(ENT_CMAP(D["pct"][:, int(np.clip(np.ceil(tau - 1e-9), 0, T - 1))]))   # first step)
            Hm = C.proj(D["home"], tau)
            for j, (tt, P) in enumerate(trails(D, tau)):
                homes[j].set_data([Hm[j, 0]], [Hm[j, 1]])
                starts[j].set_data([P[0, 0]], [P[0, 1]])
                seg, e, q = _tail(tt, P, tau)
                wd = (.45 + 1.25 * e) * SCALE ** .6
                halos[j].set_segments(seg), halos[j].set_linewidths(wd + 1.4)
                lines[j].set_segments(seg), lines[j].set_linewidths(wd)
                lines[j].set_colors([str(.6 * (1 - v)) for v in e])
                heads[j].set_data([q[0]], [q[1]])
        for cur in cursors:
            cur.set_xdata([tau + 1, tau + 1])
        for txt, pcs in notes:                                                    # the plane of the nearest step
            txt.set_text("PCs %d-%d of step $t$" % tuple(pcs[int(np.floor(tau + .5))]))
        t_txt.set_text(f"$t = {int(np.floor(tau + .5)) + 1}$")                    # (round() rounds halves to even)
        fill.set_width(tau + 1)

    return fig, update


def _key(fig, cells, mode, H):
    """the right column: the latent space key with the cells' markers, the markers' and the curves' key, the plane
    (env_ent: the entanglement colour bar and its note)"""
    x, top, ks = LEG_X, H - .62, 1.15
    if mode == "env_ent":
        fig.text((x + .55) / W_IN, top / H, "Entanglement\nof this step", ha="center", va="baseline", fontsize=FS["lab"],
                 linespacing=1.1)
        cax = fig.add_axes([(x + .35) / W_IN, (top - .15 - 1.6) / H, .16 / W_IN, 1.6 / H])
        cax.imshow(np.linspace(0, 1, 256)[:, None], cmap=ENT_CMAP, aspect="auto", origin="lower", extent=(0, 1, 0, 100))
        cax.set_xticks([]), cax.yaxis.tick_right()
        cax.set_yticks([0, 50, 100], ["0", "50", "100"])
        cax.tick_params(length=1.5, pad=1)
        cax.set_ylabel("percentile", labelpad=2)
        cax.yaxis.set_label_position("right")
        fig.text(x / W_IN, (top - 2.05) / H, f"one environment (scan\n$e = {ENV}$, a typical one):\nall 100 cells x 50\n"
                 "streams; each dot is one\nstream, coloured by the\nentanglement of the step\nit is taking, ranked among"
                 "\nthe 5000 streams at that\nstep (mean entanglement\nrises over the first 5\nsteps; ranks remove that)"
                 "\n\nthe same streams in all\nfour panels; each agent's\nown $T = 30$ PC plane", ha="left", va="top",
                 fontsize=FS["note"], color="0.3", linespacing=1.25)
        return
    fig.text((x + ks / 2) / W_IN, top / H, "Latent space", ha="center", va="baseline", fontsize=FS["lab"])
    ax = fig.add_axes([(x + .12) / W_IN, (top - .1 - ks) / H, ks / W_IN, ks / H])
    R = cf.N_R
    gg, cc = np.meshgrid(np.arange(R), np.arange(R))
    ax.imshow(cf.bivariate(gg, cc), origin="lower", extent=(-.5, R - .5, -.5, R - .5), interpolation="nearest")
    for (rg_, rc_), m in zip(cells, MARKS):
        ax.scatter(rg_, rc_, marker=m, s=(.62 * MS_HOME * cf.MARK_SCALE_A.get(m, 1)) ** 2, facecolor="white",
                   edgecolor="k", lw=.8, clip_on=False, zorder=3)
    ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
    ax.tick_params(length=1.5, pad=1)
    ax.set_xlabel("$r_g$", labelpad=-4)
    ax.set_ylabel("$r_c$", labelpad=-4)
    y = top - ks - .62
    if mode == "env":
        fig.text(x / W_IN, (y + .1) / H, f"one environment (scan\n$e = {ENV}$, a typical one):\nall 100 cells x %d\n"
                 "streams; each dot is one\nstream, coloured by its\ntrue cell $(r_g, r_c)$\n\nwithin a cell, streams\n"
                 "differ only by their\nsampled observations\n($t = 1$: one 5-bit\nobservation, so at most\n32 distinct beliefs)"
                 "\n\neach agent's own\n$T = 30$ PC plane" % 50, ha="left",
                 va="top", fontsize=FS["note"], color="0.3", linespacing=1.25)
        return
    for mk, ms, mew, fc, lab in (("o", MS_START, .8, "white", "start ($t = 1$)"), ("o", MS_END, .8, "0.55", "now"),
                                 ("o", MS_HOME, 1.1, "white", "target")):
        fig.add_artist(Line2D([(x + .2) / W_IN], [y / H], transform=fig.transFigure, marker=mk, ms=ms, mfc=fc,
                              mec="k", mew=mew, ls="none"))
        fig.text((x + .45) / W_IN, y / H, lab, ha="left", va="center", fontsize=FS["lab"], color="0.25")
        y -= .3
    ex = np.linspace(x + .05, x + 1.05, 40)                                       # the curve: t = 1 -> T
    ev = np.log1p((np.arange(39) + .5) / 39 * (T - 1)) / np.log(T)
    fig.add_artist(LineCollection(np.stack([np.stack([ex[:-1], ex[1:]], 1) / W_IN, np.full((39, 2), y / H)], -1),
                                  transform=fig.transFigure, colors=[str(.6 * (1 - v)) for v in ev],
                                  linewidths=(.45 + 1.25 * ev) * SCALE ** .6, capstyle="round"))
    fig.text(ex[0] / W_IN, (y - .2) / H, "$t = 1$", ha="left", va="baseline", fontsize=FS["tick"], color="0.4")
    fig.text(ex[-1] / W_IN, (y - .2) / H, "$T$", ha="right", va="baseline", fontsize=FS["tick"], color="0.4")
    plane = {"fixed": "each agent's beliefs in\nits own $T = 30$ PC plane",
             "bridge": "each agent's own PC planes:\n$t = 1$'s, sliding into\n$T = 30$'s by $t = %d$" % cf.BRIDGE,
             "pcs": "each step's own PC plane\n(the 2 of PCs 1-3 holding\nthe most latent variance)\n\nstrips: latent $R^2$ of\n"
                    "each step's PCs 1-3\n(white 0, black 1); dots:\nthe pair shown"}[mode]
    note = (f"{N_EP:,} episodes,\ncoloured by $(r_g, r_c)$\n\n" + plane +
            "\n\ncurves: a cell's mean\nbelief over %d streams\nof one environment" % cf.ALIAS_M)
    fig.text(x / W_IN, (y - .42) / H, note, ha="left", va="top", fontsize=FS["note"], color="0.3", linespacing=1.25)


def _frame(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def movie(mode="fixed"):
    """render the movie of mode -> figure_bin/fig1c_movie_<mode>.gif (one 255-colour palette for all frames, no
    dithering, so still regions do not flicker; each frame's duration from schedule)"""
    data, cells = (env_data(ent=mode == "env_ent"), []) if mode.startswith("env") else movie_data(mode)
    sched = schedule()
    with plt.rc_context(RC):
        fig, update = build(mode, data, cells)
        protos = []
        for tau in (0., 4., 9., T - 1.):                                          # the palette, from four frames and
            update(tau)                                                           # swatches of the key colours (the
            protos.append(np.asarray(_frame(fig)))                                # names, the cells' markers), which a
        keys = [to_rgb(c) for c in (*COL.values(), "0.25", "0.3", "0.35", "0.4", "0.45", "0.92", "white")]  # median
        keys += [tuple(cf.bivariate(g, c)) for g, c in cells]                                                # cut can
        keys += [tuple(ENT_CMAP(v)[:3]) for v in np.linspace(0, 1, 12)] if mode == "env_ent" else []
        sw = np.repeat(np.repeat((255 * np.array(keys)[None]).round().astype(np.uint8), 60, 0),               # drop
                       min(60, protos[0].shape[1] // len(keys)), 1)
        sw = np.pad(sw, ((0, 0), (0, protos[0].shape[1] - sw.shape[1]), (0, 0)), constant_values=255)
        pal = Image.fromarray(np.concatenate(protos + [sw], 0)).quantize(colors=255, method=Image.Quantize.MEDIANCUT,
                                                                          dither=Image.Dither.NONE)
        frames = []
        for i, (tau, _) in enumerate(sched):
            update(tau)
            frames.append(_frame(fig).quantize(palette=pal, dither=Image.Dither.NONE))
            if i % 50 == 0:
                print(f"{mode}: frame {i + 1} of {len(sched)}", flush=True)
        plt.close(fig)
    path = ga.fig_path(f"fig1c_movie_{dict(env_ent='env_entanglement').get(mode, mode)}.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[ms for _, ms in sched], loop=0)
    print(f"saved {path}: {len(frames)} frames, {sum(ms for _, ms in sched) / 1000:.1f} s, "
          f"{frames[0].size[0]} x {frames[0].size[1]} px, {os.path.getsize(path) / 1e6:.1f} MB")
    return path


# ═══════════════════════════════════════════════════════════════════
# The step planes as a figure
# ═══════════════════════════════════════════════════════════════════

def fig_basis():
    """figure_bin/fig1c_pc_basis.png: each agent's step planes (step_planes, the pcs movie's) over t: the latent R^2 of
    each step's PCs 1-5 (dots: the pair (c)'s rule shows), the PCs' variance (absolute), and the loadings of the
    plane's two axes (after the turn: the r_g decoder's side and the r_c decoder's side) -> the pairs shown per step"""
    BV, rg, rc = beliefs_every_step()
    order = ("Trained", "Joint", "Echo", "Naive")                                 # each network next to its observer
    S = {a: step_planes(BV[a], rg, rc) for a in order}
    t = np.arange(1, T + 1)
    W, H, cw, cg, x0 = 7.5, 6.3, 1.45, .22, .55
    rows = ((4.92, .82), (3.62, .88), (1.95, 1.10), (.42, 1.10))                  # (bottom, height) in inches
    shades = ("k", "0.35", "0.6", "0.75", "0.85")
    vmax = max(np.abs(S[a]["U"]).max() for a in order)
    with plt.rc_context(cf.RC):
        fig = plt.figure(figsize=(W, H))
        for j, a in enumerate(order):
            x = x0 + j * (cw + cg)
            ax = [fig.add_axes([x / W, y / H, cw / W, h / H]) for y, h in rows]
            fig.text((x + cw / 2) / W, (H - .14) / H, NAME[a], ha="center", va="baseline", fontsize=cf.FS_TITLE,
                     color=COL[a])
            ax[0].imshow(S[a]["lat"].T, cmap="Greys", vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                         extent=(.5, T + .5, 5.5, .5))
            ax[0].scatter(np.repeat(t, 2), S[a]["pcs"].ravel(), s=2.5, c=ACCENT, lw=0, zorder=4)
            ax[0].set_yticks(range(1, 6))
            for k in range(5):
                ax[1].plot(t, S[a]["var"][:, k], c=shades[k], lw=1., label=f"PC{k + 1}")
            ax[1].set_ylim(0, max(S[b]["var"].max() for b in order) * 1.05)
            for i in (0, 1):
                im = ax[2 + i].imshow(S[a]["U"][:, :, i].T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                                      interpolation="nearest", extent=(.5, T + .5, 19.5, -.5))
                ax[2 + i].axhline(9.5, c="k", lw=.5)
                ax[2 + i].set_yticks([0, 9, 10, 19], ["0", "9", "0", "9"])
            for k, a_ in enumerate(ax):
                a_.set_xlim(.5, T + .5)
                a_.set_xticks([1, 10, 20, 30])
                a_.tick_params(labelleft=j == 0, labelbottom=k in (1, 3))
                if k == 1:
                    cf._clean(a_)
            if j == 0:
                ax[0].set_ylabel("PC", labelpad=1)
                ax[1].set_ylabel("Variance", labelpad=1)
                for i in (0, 1):
                    ax[2 + i].set_ylabel("$r_c$          $r_g$", labelpad=1)
            ax[1].set_xlabel("Step $t$", labelpad=1)
            ax[3].set_xlabel("Step $t$", labelpad=1)
        for (y, h), (name, formula) in zip(rows, (("Latent variance of each step's PCs", r"$R^2_{r_g} + R^2_{r_c}$ of PC $k$"
                                                   "  (dots: the pair shown)"),
                                                  ("Variance of each step's PCs", r"$\lambda_k$  (PC1 black $\to$ PC5 light)"),
                                                  ("Plane axis 1 loadings", r"$U_{\cdot 1}$  ($r_g$ decoder's side)"),
                                                  ("Plane axis 2 loadings", r"$U_{\cdot 2}$  ($r_c$ decoder's side)"))):
            fig.text(.5, (y + h + .2) / H, name, ha="center", va="baseline", fontsize=cf.FS_LAB)
            fig.text(.5, (y + h + .07) / H, formula, ha="center", va="baseline", fontsize=cf.FS_TICK, color="0.3")
        cax = fig.add_axes([(x0 + 4 * cw + 3 * cg + .08) / W, rows[3][0] / H, .05 / W, (rows[2][0] + rows[2][1] -
                                                                                         rows[3][0]) / H])
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_linewidth(.4)
        cb.ax.tick_params(labelsize=cf.FS_TICK, length=1.5, pad=1)
        cb.set_label("Loading", fontsize=cf.FS_LAB, labelpad=2)
        fig.savefig(ga.fig_path("fig1c_pc_basis.png"), dpi=300)
        plt.close(fig)
    return {a: [tuple(int(v) for v in pq) for pq in S[a]["pcs"]] for a in order}


if __name__ == "__main__":
    modes = [a for a in sys.argv[1:] if not a.startswith("-") and not a.endswith(".json")]   # (an interactive
    unknown = [a for a in modes if a not in MODES + ("basis",)]                             # kernel's own args)
    if unknown:
        sys.exit(f"unknown mode(s) {unknown}; the modes are {', '.join(MODES + ('basis',))}")
    for m in modes or MODES + ("basis",):
        print(fig_basis()) if m == "basis" else movie(m)
