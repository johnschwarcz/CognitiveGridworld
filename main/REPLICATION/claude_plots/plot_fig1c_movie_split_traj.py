"""Pairs that split, and the trajectories of their environments, in one movie (27 Oct 2026)
-> main/REPLICATION/figure_bin/fig1c_movie_split_traj.gif

Columns: Optimal, Trained RNN, Naive, ESN; a fifth column holds the keys.
    row 1  plot_fig1c_split_movie's pairs: three pairs of streams, one per alias-scan environment (15, 39, 20; selected
           in the Optimal: true cells at least 4 apart, together through t_s, at least 3 further apart two steps later);
           each stream's posterior means (E[r_g], E[r_c]), faint before its pair's split step, coloured by its true
           cell (its target; the bivariate key of row 3); each pair its own marker shape; stream i filled, stream j
           outlined; small: the streams now, large: their true cells; a step that moves a stream more than JUMP = 2
           cells is drawn dashed (a jump: the posterior mean crosses between two modes, the belief never travels the
           path; a stream can jump at several steps)
    row 2  each stream's signed position in its pair: (x - xbar) . e, xbar the pair's centroid (the mean of its two
           streams now) and e the unit vector from j's true cell to i's: 0 while the two are together, + (i) and - (j)
           once they split, ending at +-d/2 (half the cells' distance) when both reach their cells; the two lines mirror
           each other by construction. (Measured from the cells' fixed midpoint instead, these pairs sit together at
           j's cell before the split and i jumps across.) One solid line per stream in its colour (faint before the
           split, as row 1); the streams now: the pair's marker, i filled, j outlined (as row 1); black on the axis:
           the split step
    row 3  the trajectories of the same three environments (the research movie fig1c_movie_traj_4envs.gif, in one row):
           one stream per cell from each (the first of the cell's 50 streams; 300 particles, one fixed random drawing
           order), each coloured by its true cell, in the agent's fixed T = 30 PC plane (fig1.belief_geometry,
           as the fixed movie and panel (c)); circles: the cells' targets (fig1.targets)
Timing: the split movie's schedule (0.6 s per step around the three split steps, 0.3 s elsewhere) at 25 frames per
second. Data: figdata/round4b/split_pairs.npz (plot_fig1c_split_movie.build) and the scan's beliefs.
Trial version (John 10-28: "add a panel showing the entanglement and Factorization regret of these 6 trials ... i'll
probably undo it"): movie(extra=True) -> fig1c_movie_split_traj_entfr.gif, a row between rows 2 and 3 with the six
streams' entanglement and factorization regret (both variables: plot_fig1c_measures_movie.fr_both, since 2026-11-10
the mean over the two variables of the forward KL(Optimal || Naive), exact; the symmetric clipped KL before, which
gave these six curves other shapes and another order at T) over the steps, each minus its environment's mean at
that step (all 5,000 streams: both rise for every stream as beliefs sharpen, which hid the split on raw axes; a
per-cell baseline would offset a pair's two streams by their cells' different typical values), styled as row 2; the
default movie is unchanged. Only environment 15's pair saw the same observations until its split (its two lines
coincide until then); environment 39's and 20's pairs share only some observations, so their factorization regret
parts from step 3 - 5, before the split (10% of all 743 pairs share every observation through t_s - 1).
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_movie_split_traj.py [extra]"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402
from matplotlib.colors import to_rgb                                              # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from PIL import Image                                                             # noqa: E402

SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
AG = ("Joint", "Trained", "Naive", "Echo")
T, SUB, M = 30, 8, 50
DPI, FPS = 110, 25
PW, XG, X0 = 2.7, .3, .8                                                         # panel side, column gap, left edge
Y_CLOCK, Y_NAME, GAP12, DH, GAP23 = .8, .66, .55, 1.62, .62                      # bands and gaps (inches)
JUMP = 2.                                                                          # a jump: one step > 2 cells
LW_PATH = 1.4
JUMP_DASH = (0, (3.2, 2.2))
LW_S, YS = 1.8, 7.2                                                               # row 2: lines, y range +-YS
GAP24, DH4 = .62, 1.62                                                            # movie(extra): the added row
PRE_ALPHA = .3                                                                     # paths before the split (as the
                                                                                   # split movie's)
KX, KW = X0 + 4 * PW + 3 * XG + .45, 2.75                                        # the keys' column
W = KX + KW + .15
H = Y_CLOCK + Y_NAME + PW + GAP12 + DH + GAP23 + PW + .3
FS = dict(t=15, name=13.5, stat=10, row=11.5, lab=11.5, key=10.5, tick=9.5)
TRAJ = dict(lw=.3, alpha=.5, ms=2.4, home_ms=4.4, home_mew=.6, aa=True)        # row 3: thin anti-aliased lines
PAIR_MARKERS = ("o", "^", "D")                                                    # rows 1-2: the pairs' shapes


def _schedule(split_steps):
    """step parameter tau (0 = step 1) per frame: plot_fig1c_split_movie._frames at FPS"""
    slow = set()
    for s in split_steps:
        slow.update(range(max(s - 3, 0), min(s + 3, T - 1)))
    taus = [0.] * int(.8 * FPS)
    for s in range(T - 1):
        n = int(round((.6 if s in slow else .3) * FPS))
        taus += list(s + np.arange(n) / n)
    return taus + [T - 1.] * int(2.5 * FPS)


def traj_data(envs, n=1):
    """row 3, per agent, in its fixed T = 30 plane: the particles' positions at every step (len, T, 2), their
    Catmull-Rom paths at SUB points per step (len, len(tt), 2), colours, the 100 cells' targets (100, 2)"""
    import fig1 as cf
    import fig1_movie as cm
    G = cf.belief_geometry()
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    cells_all = meta["cells"]
    assert (meta["cell_of"] == np.repeat(np.arange(len(cells_all)), M)).all()
    js = np.concatenate([k * M + np.arange(n) for k in range(len(cells_all))])
    ev, js = np.repeat(np.asarray(envs, int), len(js)), np.tile(js, len(envs))
    p = np.random.default_rng(0).permutation(len(js))
    ev, js = ev[p], js[p]
    cell = cells_all[meta["cell_of"][js]]
    home, _ = cf.targets([tuple(int(v) for v in c) for c in cells_all])
    tt = np.linspace(0, T - 1, (T - 1) * SUB + 1)
    col = np.asarray(cf.bivariate(cell[:, 0], cell[:, 1]))
    out = {}
    for a in AG:
        D30 = G[(a, T)]
        V = np.load(os.path.join(SCAN, f"scan_{a}.npz"))["beliefs"][ev, js].astype(np.float64)
        out[a] = dict(V2=cm._proj(D30, V), tt=tt, P2=cm._proj(D30, np.stack([cm._blend(V, tau, axis=1) for tau in tt], 1)),
                      col=col, rgba=np.concatenate([col, np.full((len(col), 1), TRAJ["alpha"])], 1),
                      home2=cm._proj(D30, np.asarray(home)),
                      home_col=np.asarray(cf.bivariate(cells_all[:, 0], cells_all[:, 1])))
    return out


def traj_frame(D, tau):
    """every particle's trajectory so far (its path's sub-points up to tau, then its position at tau), its position,
    the targets; all in the fixed plane"""
    import fig1_movie as cm
    k = int(np.searchsorted(D["tt"], tau, side="right"))
    head = cm._blend(D["V2"], tau, axis=1)                                         # (Catmull-Rom is linear: blending
    trail = np.concatenate([D["P2"][:, :k], head[:, None]], 1)                     # the projections = projecting)
    return trail, head, D["home2"]


def traj_limits(D, taus):
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for tau in taus:
        tr, _, hm = traj_frame(D, tau)
        pts = np.concatenate([tr.reshape(-1, 2), hm])
        lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
    c, s = (lo + hi) / 2, (hi - lo).max() * 1.04
    return (c[0] - s / 2, c[0] + s / 2), (c[1] - s / 2, c[1] + s / 2)


class _Exact:
    """each pixel to its nearest palette colour, exactly (a lazily filled 24-bit table; the research movie's)"""

    def __init__(self, cols):
        self.cols = np.asarray(cols, np.float32)
        self.flat = np.asarray(cols, np.uint8).ravel().tolist()
        self.lut = np.full(1 << 24, -1, np.int16)

    def __call__(self, X):
        X = X.astype(np.int32)
        code = (X[..., 0] << 16) | (X[..., 1] << 8) | X[..., 2]
        miss = np.unique(code[self.lut[code] < 0])
        for k in range(0, len(miss), 20000):
            u = miss[k:k + 20000]
            rgb = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], 1).astype(np.float32)
            self.lut[u] = ((rgb[:, None] - self.cols[None]) ** 2).sum(-1).argmin(1)
        P = Image.fromarray(self.lut[code].astype(np.uint8), "P")
        P.putpalette(self.flat)
        return P


def movie(extra=False):
    """extra (John 10-28: "add a panel showing the entanglement and Factorization regret of these 6 trials"; a trial
    version, kept apart): a row between rows 2 and 3 with the six streams' entanglement and factorization regret
    (both variables, plot_fig1c_measures_movie.fr_both) over the steps, as row 2 -> fig1c_movie_split_traj_entfr.gif"""
    import fig1 as cf
    import fig1_movie as cm
    import plot_fig1c_split_movie as sm
    from main.utils import fig_path
    Z = np.load(sm.DATA)
    P, sts, envs = Z["P"], Z["shown_ts"], [int(e) for e in Z["shown_env"]]
    n = len(sts)
    pmk = PAIR_MARKERS[:n]
    tcol = [[np.asarray(cf.bivariate(int(round(c[0])), int(round(c[1]))), np.float64)[:3] for c in Z[k]]
            for k in ("c_i", "c_j")]                                              # each stream's colour: its true cell
    half = np.linalg.norm(Z["c_i"] - Z["c_j"], axis=-1) / 2                         # row 2: each pair's half distance
                                                                                   # and axis (j's cell -> i's)
    e_ax = (Z["c_i"] - Z["c_j"]) / (2 * half[:, None])
    both = (Z["err"] <= 1).all(-1).mean(1)
    TD = traj_data(envs)
    taus = _schedule(sts)
    cells = np.array([(g, c) for g in range(10) for c in range(10)])
    plt.rcParams["mathtext.fontset"] = "cm"
    Hh = H + (GAP24 + DH4 if extra else 0.)
    fig = plt.figure(figsize=(W, Hh), dpi=DPI, facecolor="white")
    ax_ = lambda x, y, w, h: fig.add_axes([x / W, y / Hh, w / W, h / Hh])
    y1 = Hh - Y_CLOCK - Y_NAME - PW
    y2 = y1 - GAP12 - DH
    y4 = y2 - GAP24 - DH4
    y3 = (y4 if extra else y2) - GAP23 - PW
    sty = ({"lw": LW_PATH}, {"lw": LW_PATH})                                       # both streams solid
    g = np.arange(10)
    lim_taus = taus[::10] + [T - 1.]
    cols = []
    for c_, a in enumerate(AG):
        x = X0 + c_ * (PW + XG)
        fig.text((x + PW / 2) / W, (y1 + PW + .36) / Hh, cm.NAME[a], ha="center", va="baseline", fontsize=FS["name"],
                 color=cm.COL[a])
        fig.text((x + PW / 2) / W, (y1 + PW + .12) / Hh,
                 "both streams end at their cells: %d%%" % round(100 * both[c_]),
                 ha="center", va="baseline", fontsize=FS["stat"], color="0.35")
        # row 1: the pairs
        a1 = ax_(x, y1, PW, PW)
        a1.scatter(*np.meshgrid(g, g), s=2, c="0.82", lw=0, zorder=0)
        a1.set_xlim(-.6, 9.6), a1.set_ylim(-.6, 9.6), a1.set_aspect("equal")
        a1.set_xticks([0, 3, 6, 9]), a1.set_yticks([0, 3, 6, 9])
        a1.tick_params(labelsize=FS["tick"])
        a1.set_xlabel(r"$E[r_g]$", fontsize=FS["lab"], labelpad=1)
        if c_ == 0:
            a1.set_ylabel(r"$E[r_c]$", fontsize=FS["lab"], labelpad=1)
        else:
            a1.set_yticklabels([])
        lines = []
        for p in range(n):
            ci, cj = tcol[0][p], tcol[1][p]
            a1.plot(*Z["c_i"][p], pmk[p], ms=11.5, color=ci, mec="0.2", mew=.6, zorder=2)     # true cells: large,
            a1.plot(*Z["c_j"][p], pmk[p], ms=11.5, mfc="white", mec=cj, mew=2.2, zorder=2)    # i filled, j outlined
            seg = [a1.add_collection(LineCollection([], linewidths=LW_PATH, capstyle="round", zorder=3 + .5 * k))
                   for k in (0, 1)]                                                   # each stream's path: per-step
            hi, = a1.plot([], [], pmk[p], ms=6.5, color=ci, mec="0.15", mew=.6, zorder=5)     # style; the streams
            hj, = a1.plot([], [], pmk[p], ms=6.5, mfc="white", mec=cj, mew=1.8, zorder=5)     # now: small, i
            lines.append((seg, hi, hj, (ci, cj)))                                             # filled, j outlined
        # row 2: each stream's signed position along its pair
        a2 = ax_(x, y2, PW, DH)
        a2.axhline(0, c="0.6", lw=.8, ls=":", zorder=0)
        dl, dh = [], []
        for p in range(n):
            ci, cj = tcol[0][p], tcol[1][p]
            a2.plot([sts[p] + 1], [-YS], pmk[p], ms=6.5, color="k", clip_on=False, zorder=6)  # the split step
            ls_ = []
            for al in (PRE_ALPHA, 1.):                                                        # before / from t_s
                for k in (0, 1):                                                              # (opaque tints)
                    ls_.append(a2.plot([], [], c=al * tcol[k][p] + 1 - al, lw=LW_S, solid_capstyle="round",
                                       zorder=3 + .1 * p + .05 * k)[0])
            dl.append(ls_)                                                                    # pre i, j; post i, j
            dh.append((a2.plot([], [], pmk[p], ms=6.5, color=ci, mec="0.15", mew=.6, zorder=7)[0],
                       a2.plot([], [], pmk[p], ms=6.5, mfc="white", mec=cj, mew=1.8, zorder=7)[0]))
        cur = a2.axvline(1, c="0.7", lw=.7, zorder=0)
        a2.set_xlim(.5, T + .5), a2.set_ylim(-YS, YS)
        a2.set_yticks([-5, 0, 5]), a2.set_xticks([1, 10, 20, 30])
        a2.tick_params(labelsize=FS["tick"])
        a2.spines[["top", "right"]].set_visible(False)
        a2.set_xlabel("step $t$", fontsize=FS["lab"], labelpad=1)
        if c_ > 0:
            a2.set_yticklabels([])
        # row 3: the environments' trajectories
        Dt = TD[a]
        a3 = ax_(x, y3, PW, PW)
        xl, yl = traj_limits(Dt, lim_taus)
        a3.set_xlim(*xl), a3.set_ylim(*yl), a3.set_aspect("equal", adjustable="box"), a3.axis("off")
        tr, hd, hm = traj_frame(Dt, 0.)
        lc = a3.add_collection(LineCollection(list(tr), colors=Dt["rgba"], linewidths=TRAJ["lw"], zorder=1,
                                              capstyle="round", joinstyle="round", antialiased=TRAJ["aa"]))
        ht = a3.scatter(*hm.T, s=TRAJ["home_ms"] ** 2, facecolors="none", edgecolors=Dt["home_col"],
                        linewidths=TRAJ["home_mew"], zorder=2)
        pt = a3.scatter(*hd.T, s=TRAJ["ms"] ** 2, c=Dt["col"], lw=0, zorder=3, antialiased=False)
        cols.append(dict(c_=c_, lines=lines, dl=dl, dh=dh, cur=cur, Dt=Dt, lc=lc, ht=ht, pt=pt))
    ext = []
    if extra:                                                                      # the six streams' entanglement, FR
        import plot_fig1c_measures_movie as mmv
        meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
        frb = mmv.fr_both()
        envs_all = [e for e in range(40) if e != 2]
        ser = {}
        for p in range(n):                                                         # each stream minus its
            e = int(Z["shown_env"][p])                                             # environment's mean at that
            ie = envs_all.index(e)                                                 # step (all 5,000 streams): a
            rv, fr_ = meta["rev"][e].astype(np.float64), frb[ie].astype(np.float64)   # per-cell baseline would
            for k, key in enumerate(("shown_i", "shown_j")):                       # offset the pair's streams
                st_ = int(Z[key][p])
                ser[p, k] = (rv[st_] - rv.mean(0), fr_[st_] - fr_.mean(0))
        G4 = .85                                                               # room for the 2nd y axis
        xw = (4 * PW + 3 * XG - G4) / 2
        for q, (title, ylab) in enumerate((("entanglement, minus its environment's mean at the step",
                                            r"rev$_t$ (nats)"),
                                           ("factorization regret (both variables), minus its environment's mean "
                                            "at the step", r"FR$_t$ (nats)"))):
            a4 = ax_(X0 + q * (xw + G4), y4, xw, DH4)
            vals = np.concatenate([ser[p, k][q] for p in range(n) for k in (0, 1)])
            lo_, hi_ = min(vals.min(), 0.), max(vals.max(), 0.)                  # the data's range, with 0
            pad_ = .08 * (hi_ - lo_)
            a4.set_ylim(lo_ - pad_, hi_ + pad_)
            a4.axhline(0, c="0.6", lw=.8, ls=":", zorder=0)
            a4.set_xlim(.5, T + .5), a4.set_xticks([1, 10, 20, 30])
            a4.tick_params(labelsize=FS["tick"])
            a4.spines[["top", "right"]].set_visible(False)
            a4.set_xlabel("step $t$", fontsize=FS["lab"], labelpad=1)
            a4.set_ylabel(ylab, fontsize=FS["lab"], labelpad=1)
            a4.set_title(title, fontsize=FS["lab"], pad=3)
            dl4, dh4 = [], []
            for p in range(n):
                ci, cj = tcol[0][p], tcol[1][p]
                a4.plot([sts[p] + 1], [a4.get_ylim()[0]], pmk[p], ms=6.5, color="k", clip_on=False, zorder=6)
                ls_ = []
                for al in (PRE_ALPHA, 1.):
                    for k in (0, 1):
                        ls_.append(a4.plot([], [], c=al * tcol[k][p] + 1 - al, lw=LW_S, solid_capstyle="round",
                                           zorder=3 + .1 * p + .05 * k)[0])
                dl4.append(ls_)
                dh4.append((a4.plot([], [], pmk[p], ms=6.5, color=ci, mec="0.15", mew=.6, zorder=7)[0],
                            a4.plot([], [], pmk[p], ms=6.5, mfc="white", mec=cj, mew=1.8, zorder=7)[0]))
            ext.append(dict(q=q, dl=dl4, dh=dh4, cur=a4.axvline(1, c="0.7", lw=.7, zorder=0)))
    for y, h, txt in ((y1, PW, "pairs that split"), (y2, DH, "position in the pair"),
                      (y3, PW, "their environments")):
        fig.text(.2 / W, (y + h / 2) / Hh, txt, rotation=90, ha="center", va="center", fontsize=FS["row"], color="0.3")
    # the clock
    ty = Hh - .42
    t_txt = fig.text(X0 / W, ty / Hh, "", fontsize=FS["t"], ha="left", va="center")
    bx0, bx1 = X0 + 1.15, X0 + 4 * PW + 3 * XG
    bar = ax_(bx0, ty - .06, bx1 - bx0, .12)
    bar.set_xlim(.5, T + .5), bar.set_ylim(0, 1), bar.axis("off")
    bar.barh(.5, T, left=.5, height=1, color="0.92", lw=0)
    fill = bar.barh(.5, 0, left=.5, height=1, color="0.35", lw=0)[0]
    for t_ in (1, T):
        fig.text((bx0 + (t_ - .5) / T * (bx1 - bx0)) / W, (ty - .3) / Hh, f"${t_}$", ha="center", va="baseline",
                 fontsize=FS["tick"], color="0.4")
    _keys(fig, cf, sm, sty, pmk, y1, y2, y3, envs, Hh)

    def update(tau):
        s = int(np.floor(tau))
        for C in cols:
            a_ = C["c_"]
            for p, (seg, hi, hj, cc) in enumerate(C["lines"]):
                path, head = sm._at(P[a_, p], tau)                                    # (2, s + 2, 2): steps 0..s, tau
                k0 = sts[p] - 1                                                        # index of step t_s
                for k, lc_ in enumerate(seg):
                    _style(lc_, path[k], P[a_, p, k], k0, cc[k])
                hi.set_data([head[0, 0]], [head[0, 1]]), hj.set_data([head[1, 0]], [head[1, 1]])
                sv = (path - path.mean(0)) @ e_ax[p]                                   # (2, s + 2): steps 0..s, tau
                xs_ = np.r_[np.arange(1, s + 2), tau + 1]
                m_ = len(xs_)
                for h_, sl in enumerate((slice(0, min(m_, k0 + 1)), slice(min(k0, m_), m_))):   # before / from t_s
                    for k in (0, 1):
                        C["dl"][p][2 * h_ + k].set_data(xs_[sl], sv[k, sl])
                for k in (0, 1):
                    C["dh"][p][k].set_data([xs_[-1]], [sv[k, -1]])
            C["cur"].set_xdata([tau + 1])
            tr, hd, hm = traj_frame(C["Dt"], tau)
            C["lc"].set_segments(list(tr))
            C["ht"].set_offsets(hm)
            C["pt"].set_offsets(hd)
        for X_ in ext:
            q = X_["q"]
            xs_ = np.r_[np.arange(1, s + 2), tau + 1]
            for p in range(n):
                k0 = sts[p] - 1
                for k in (0, 1):
                    v = ser[p, k][q]
                    vt = v[s] + (tau - s) * (v[min(s + 1, T - 1)] - v[s])
                    yv = np.r_[v[:s + 1], vt]
                    m_ = len(xs_)
                    for h_, sl in enumerate((slice(0, min(m_, k0 + 1)), slice(min(k0, m_), m_))):
                        X_["dl"][p][2 * h_ + k].set_data(xs_[sl], yv[sl])
                    X_["dh"][p][k].set_data([xs_[-1]], [yv[-1]])
            X_["cur"].set_xdata([tau + 1])
        t_txt.set_text(f"$t = {s + 1}$")
        fill.set_width(tau + 1)

    def grab():
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    protos = []
    for tau in (0., 6., 12., T - 1.):
        update(tau)
        protos.append(grab())
    keys = [(1., 1., 1.), (0., 0., 0.)] + [(v, v, v) for v in np.linspace(0, 1, 24)]
    keys += [to_rgb(c) for c in cm.COL.values()]
    keys += [tuple(al * np.array(to_rgb(cm.COL[k])) + 1 - al) for k in ("Trained", "Echo") for al in (.2, .4, .6, .8)]
    keys += [tuple(sm.PRE_ALPHA * c + 1 - sm.PRE_ALPHA) for cc in tcol for c in cc]   # the faint paths
    keys += [tuple(np.asarray(c, np.float64)[:3]) for c in cf.bivariate(cells[:, 0], cells[:, 1])]
    fixed = np.unique((255 * np.clip(np.array(keys, np.float64), 0, 1)).round().astype(np.uint8), axis=0)
    mc = Image.fromarray(np.concatenate(protos, 0)).quantize(colors=255 - len(fixed), method=Image.Quantize.MEDIANCUT,
                                                              dither=Image.Dither.NONE)
    rest = np.array(mc.getpalette()[:3 * (255 - len(fixed))], np.uint8).reshape(-1, 3)
    quant = _Exact(np.concatenate([fixed, rest]))
    print(f"palette: {len(fixed)} exact colours + {len(rest)} from the frames", flush=True)
    frames = []
    for i, tau in enumerate(taus):
        update(tau)
        frames.append(quant(grab()))
        if i % 50 == 0:
            print(f"frame {i + 1} of {len(taus)}", flush=True)
    plt.close(fig)
    path = fig_path("fig1c_movie_split_traj_entfr.gif" if extra else "fig1c_movie_split_traj.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0)
    print(f"saved {path}: {len(frames)} frames, {len(frames) / FPS:.1f} s, {frames[0].size[0]} x {frames[0].size[1]} px, "
          f"{os.path.getsize(path) / 1e6:.1f} MB")
    return path


def _style(lc, path, steps, k0, col):
    """one stream's path up to the frame (path: its points at steps 0..s and the head) as segments, each styled by its
    step: faint before the split (ending at or before step index k0), dashed if the full step moves more than JUMP"""
    if len(path) < 2:
        lc.set_segments([])
        return
    segs = np.stack([path[:-1], path[1:]], 1)
    k = np.arange(len(segs))
    disp = np.r_[np.linalg.norm(np.diff(steps, axis=0), axis=-1), 0.]                # (the head at the last step
    jump = disp[np.minimum(k, len(disp) - 1)] > JUMP                                 # repeats it: no move)
    rgba = np.tile(np.r_[col, 1.], (len(segs), 1))
    rgba[k < k0, 3] = PRE_ALPHA
    lc.set_segments(segs)
    lc.set_color(rgba)
    lc.set_linestyle([JUMP_DASH if j_ else "-" for j_ in jump])


def _keys(fig, cf, sm, sty, pmk, y1, y2, y3, envs, Hh=None):
    """the fifth column, beside each row: rows 1-2, the pairs' shapes, the two streams (fill: i, outline: j), now and
    true cell (small, large), before the split, a jump; row 2, the split step and the true distance; row 3, the colours
    (true cells, rows 1 and 3) and the target"""
    Hh = H if Hh is None else Hh
    x = KX

    def entry(y, lab, line=None, marker=None, lines=()):
        for ln in ([line] if line is not None else []) + list(lines):
            fig.add_artist(Line2D([(x + .02) / W, (x + .4) / W], [y / Hh] * 2, transform=fig.transFigure, **ln))
        if marker is not None:
            fig.add_artist(Line2D([(x + (.3 if (line is not None or lines) else .21)) / W], [y / Hh],
                                  transform=fig.transFigure, ls="none", **marker))
        fig.text((x + .5) / W, y / Hh, lab, ha="left", va="center", fontsize=FS["key"], color="0.2")

    fill, edge = "0.72", "0.25"                                                   # the key's stand-ins for i's, j's
    y = y1 + PW - .05                                                             # beside row 1
    fig.text(x / W, y / Hh, "Rows 1-2: three pairs", ha="left", va="top", fontsize=FS["lab"], color="0.15")
    y -= .45
    for p, mk in enumerate(pmk):
        entry(y, "environment %d" % envs[p], marker=dict(marker=mk, ms=7, color="0.55", mec="0.15", mew=.6))
        y -= .3
    y -= .08
    entry(y, "stream $i$ (fill)", line=dict(c=fill, **sty[0]), marker=dict(marker="o", ms=6.5, color=fill, mec="0.15",
                                                                           mew=.6))
    y -= .3
    entry(y, "stream $j$ (outline)", line=dict(c=edge, **sty[1]),
          marker=dict(marker="o", ms=6.5, mfc="white", mec=edge, mew=1.8))
    y -= .3
    for xm, ms in ((.1, 6.5), (.32, 11.5)):                                       # small: now; large: true cell
        fig.add_artist(Line2D([(x + xm) / W], [y / Hh], transform=fig.transFigure, ls="none", marker="o", ms=ms,
                              mfc="white", mec="0.45", mew=1.3))
    fig.text((x + .5) / W, y / Hh, "small: now, large: true cell", ha="left", va="center", fontsize=FS["key"],
             color="0.2")
    y -= .3
    entry(y, "before the split", line=dict(c="0.3", lw=LW_PATH, alpha=sm.PRE_ALPHA))
    y -= .3
    entry(y, "jump (one step $>$ %g cells)" % JUMP, line=dict(c="0.3", lw=LW_PATH, ls=JUMP_DASH))
    y = y2 + DH / 2 + .15                                                         # beside row 2 (its middle)
    entry(y, "0: together; $i$ $+$, $j$ $-$", line=dict(c="0.6", lw=.9, ls=":"))
    y -= .3
    entry(y, "split step", marker=dict(marker="o", ms=6.5, color="k"))
    top = y3 + PW + .3                                                            # beside row 3
    fig.text(x / W, top / Hh, "Row 3: one stream per cell", ha="left", va="top", fontsize=FS["lab"], color="0.15")
    ks = 1.55
    ax = fig.add_axes([(x + .3) / W, (top - .68 - ks) / Hh, ks / W, ks / Hh])
    R = cf.N_R
    gg, cc = np.meshgrid(np.arange(R), np.arange(R))
    ax.imshow(cf.bivariate(gg, cc), origin="lower", extent=(-.5, R - .5, -.5, R - .5), interpolation="nearest")
    ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
    ax.tick_params(length=2, pad=1, labelsize=FS["tick"])
    ax.set_xlabel("$r_g$", labelpad=-5, fontsize=FS["lab"])
    ax.set_ylabel("$r_c$", labelpad=-5, fontsize=FS["lab"])
    ax.set_title("true cell colour (rows 1, 3)", fontsize=FS["key"], color="0.2", pad=3)
    entry(top - .68 - ks - .45, "target", marker=dict(marker="o", ms=6, mfc="none", mec="0.35", mew=1))


if __name__ == "__main__":
    movie(extra="extra" in sys.argv[1:])
