"""Belief trajectories of 50 particles, each from a different environment, in the lowest and the highest 10% of
factorization regret at T (the research figure of 25 Oct 2026, the t = 30 frames of the session's
research7/traj_movie_random movies; ported 27 Oct) -> main/REPLICATION/figure_bin/fig1c_traj_FR_groups_T30.png

Two frames side by side, left the lowest 10% of FR, right the highest; each in fig1_movie's layout (7.05 in at
150 dpi: 1057 x 1132 px).
Draw (draw_episodes): the trained harvest's first 20,000 episodes (each one stream of one cell of one environment; the
episodes the step planes are fitted on). Eligible: distinct-variable environments, both values in {0, 3, 6, 9} (16
cells), and FR at T = 30 (both variables: Jeffreys/2 between the Optimal's and the Naive's marginals, each clipped to
[1e-15, 1 - 1e-15]) in the lowest / highest 10% of the eligible episodes; then 50 environments drawn uniformly without
replacement (seed 0) and one of each environment's eligible episodes uniformly. The draws equal the research figure's
(main/DATA/experimental/fig1c_compression/figdata/traj_fr_groups/random_draw_50_0_sub0369_fr{low,high}.npz; asserted).
Each frame: the four agents in (c)'s 2 x 2 order (Trained RNN, ESN / Optimal, Naive), each in step 30's own PC plane
(fig1_movie.step_planes, the pcs movie's: PCs of the 20,000 episodes' beliefs, one plane for all environments;
the box holds every frame of the movie); each particle's whole trajectory re-drawn in that plane (Catmull-Rom between
steps, fig1_movie._blend) and a filled circle at t = 30, in its true cell's colour (the bivariate key); hollow
circles: the 16 cells' targets (fig1.targets: the Optimal's mean correct T = 30 belief about the cell); under
each panel the latent R^2 of every step's PCs 1-3 (fig1_movie._strip; dots: the pair shown); the clock; the key
column (the latent space with the 16 drawn cells lit, the marks, the draw).
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_traj_fr_groups.py"""
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

REF = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression", "figdata", "traj_fr_groups")
SUB, FR_Q, N_ENV, SEED = (0, 3, 6, 9), .10, 50, 0
STY = dict(lw=.5, alpha=.8, ms=3.4, home_ms=5.2, home_mew=1., faint="0.82", faint_alpha=.5)   # the research movie's
_CACHE = {}


def _data():
    if not _CACHE:
        import fig1 as cf
        import fig1_movie as cm
        BV, rg, rc = cm.beliefs_every_step()
        _CACHE.update(cf=cf, cm=cm, G=cf.belief_geometry(), BV=BV, rg=np.asarray(rg), rc=np.asarray(rc))
    return _CACHE


def _jeffreys_half(p, q, eps=1e-15):
    p, q = np.clip(p, eps, 1 - eps), np.clip(q, eps, 1 - eps)
    return .5 * ((p - q) * (np.log(p) - np.log(q))).sum(-1)


def draw_episodes(group):
    """the 50 harvest episodes of the FR group ('low' or 'high') -> episode indices (50,), the number of environments
    holding eligible episodes, the group's FR cut (nats)"""
    D = _data()
    cf, BV, rg, rc = D["cf"], D["BV"], D["rg"], D["rc"]
    n = cf.N_EP
    ci = np.sort(cf._harvest("trained", ["ep_ctx_inds"])["ep_ctx_inds"][:n].astype(int), 1)
    J, N = BV["Joint"][:, -1].astype(np.float64), BV["Naive"][:, -1].astype(np.float64)
    frT = _jeffreys_half(J[:, :10], N[:, :10]) + _jeffreys_half(J[:, 10:], N[:, 10:])
    elig = (ci[:, 0] != ci[:, 1]) & np.isin(rg[:n], SUB) & np.isin(rc[:n], SUB)
    q = np.quantile(frT[elig], FR_Q if group == "low" else 1 - FR_Q)
    elig &= (frT <= q) if group == "low" else (frT >= q)
    ok = np.flatnonzero(elig)
    pairs, inv = np.unique(ci[ok], axis=0, return_inverse=True)
    inv = inv.ravel()
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(pairs), N_ENV, replace=False)
    eps = np.array([rng.choice(ok[inv == p]) for p in pick])
    ref = os.path.join(REF, f"random_draw_{N_ENV}_{SEED}_sub{''.join(map(str, SUB))}_fr{group}.npz")
    if os.path.exists(ref):
        assert np.array_equal(eps, np.load(ref)["episodes"]), f"the {group}-FR draw differs from {ref}"
    return eps, len(pairs), float(q)


def group_data(group):
    """per agent: the step planes and their Course (fig1_movie's pcs mode), the particles' beliefs (50, T, 20),
    their Catmull-Rom paths (50, len(tt), 20) and the 16 cells' targets (20-dim); and the colours and the draw"""
    D = _data()
    cf, cm, G, BV, rg, rc = D["cf"], D["cm"], D["G"], D["BV"], D["rg"], D["rc"]
    eps, n_avail, fr_cut = draw_episodes(group)
    cell = np.stack([rg[eps], rc[eps]], 1).astype(int)
    hc = np.array([(g, c) for g in SUB for c in SUB])
    home_all, _ = cf.targets([(g, c) for g in range(cf.N_R) for c in range(cf.N_R)])
    home = np.asarray(home_all)[hc[:, 0] * cf.N_R + hc[:, 1]]
    tt = np.linspace(0, cm.T - 1, (cm.T - 1) * cm.SUB + 1)
    out = {}
    for a in ("Trained", "Echo", "Joint", "Naive"):
        S = cm.step_planes(BV[a], rg, rc)
        V = np.asarray(BV[a][eps], np.float64)
        out[a] = dict(C=cm.Course(G, a, "pcs", S), S=S, V=V, tt=tt, home=home,
                      P20=np.stack([cm._blend(V, tau, axis=1) for tau in tt], 1))
    col = np.asarray(cf.bivariate(cell[:, 0], cell[:, 1]))
    return out, dict(col=col, rgba=np.concatenate([col, np.full((len(col), 1), STY["alpha"])], 1), group=group,
                     home_col=np.asarray(cf.bivariate(hc[:, 0], hc[:, 1])), n=len(eps), n_avail=n_avail, fr_cut=fr_cut)


def _at(D, tau):
    """frame tau in its own plane (Course.proj: the Catmull-Rom blend of the step planes): every particle's trajectory
    so far ((n, k + 1, 2): its path's sub-points up to tau, then its position at tau), its position (n, 2), the targets"""
    cm = _data()["cm"]
    k = int(np.searchsorted(D["tt"], tau, side="right"))
    head = D["C"].proj(cm._blend(D["V"], tau, axis=1), tau)
    trail = np.concatenate([D["C"].proj(D["P20"][:, :k], tau), head[:, None]], 1)
    return trail, head, D["C"].proj(D["home"], tau)


def _box(D, taus):
    """one square data box holding every frame's particles, trajectories and targets (the movie's; a margin)"""
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for tau in taus:
        tr, hd, hm = _at(D, tau)
        pts = np.concatenate([tr.reshape(-1, 2), hm])
        lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
    c, s = (lo + hi) / 2, (hi - lo).max() * 1.06
    return (c[0] - s / 2, c[0] + s / 2), (c[1] - s / 2, c[1] + s / 2)


def _key(fig, H, m):
    """the right column: the latent space (the 16 drawn cells lit), the marks, the draw"""
    D = _data()
    cf, cm = D["cf"], D["cm"]
    W_IN, FS = cm.W_IN, cm.FS
    x, top, ks = cm.LEG_X, H - .62, 1.15
    fig.text((x + ks / 2) / W_IN, top / H, "Latent space", ha="center", va="baseline", fontsize=FS["lab"])
    ax = fig.add_axes([(x + .12) / W_IN, (top - .1 - ks) / H, ks / W_IN, ks / H])
    R = cf.N_R
    gg, cc = np.meshgrid(np.arange(R), np.arange(R))
    lit = np.zeros((R, R), bool)
    lit[np.ix_(list(SUB), list(SUB))] = True
    fa = np.array(to_rgb(STY["faint"]))                                          # the other cells washed out
    ax.imshow(np.where(lit[..., None], np.asarray(cf.bivariate(gg, cc)), STY["faint_alpha"] * fa + 1 - STY["faint_alpha"]),
              origin="lower", extent=(-.5, R - .5, -.5, R - .5), interpolation="nearest")
    ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
    ax.tick_params(length=1.5, pad=1)
    ax.set_xlabel("$r_g$", labelpad=-4)
    ax.set_ylabel("$r_c$", labelpad=-4)
    y = top - ks - .62
    y += .12
    c_mid = tuple(cf.bivariate(5, 4))
    lab = dict(ha="left", va="center", fontsize=FS["note"] + .5, color="0.25")
    fig.add_artist(Line2D([(x + .2) / W_IN], [y / H], transform=fig.transFigure, marker="o", ms=STY["ms"] * 1.6,
                          mfc=c_mid, mec="none", ls="none"))
    fig.text((x + .42) / W_IN, y / H, "a particle now", **lab)
    y -= .24
    fig.add_artist(Line2D([(x + .05) / W_IN, (x + .35) / W_IN], [y / H, (y + .05) / H], transform=fig.transFigure,
                          color=c_mid, lw=STY["lw"] * 2, alpha=.8, solid_capstyle="round"))
    fig.text((x + .42) / W_IN, y / H, "its trajectory so far", **lab)
    y -= .24
    fig.add_artist(Line2D([(x + .2) / W_IN], [y / H], transform=fig.transFigure, marker="o", ms=STY["home_ms"] * 1.5,
                          mfc="none", mec=c_mid, mew=STY["home_mew"] * 1.5, ls="none"))
    fig.text((x + .42) / W_IN, y / H, "the cell's target", **lab)
    g = m["group"]
    who = (f"{N_ENV} environments drawn at\nrandom (of {m['n_avail']:,}), one\nrandom episode each: {m['n']}\nparticles, "
           "each coloured by\nits true cell $(r_g, r_c)$" +
           ",\nboth values in $\\{%s\\}$\n(16 cells; their targets)" % ", ".join(str(v) for v in SUB) +
           "\n\n%s 10%% of FR at $T = 30$\n(FR %s %.2g nats)" % ({"low": "the lowest", "high": "the highest"}[g],
                                                                {"low": "$\\leq$", "high": "$\\geq$"}[g], m["fr_cut"]))
    plane = ("\n\neach step's own PC plane,\nas in the pcs movie (PCs\nof 20,000 episodes' beliefs:\none plane for all "
             "environ-\nments); all trajectories\nre-drawn in the current plane")
    fig.text(x / W_IN, (y - .26) / H, who + plane + "\n\ntarget: the Optimal's mean\ncorrect $T = 30$ belief\nabout the "
             "cell\n\nstrips: latent $R^2$ of\neach step's PCs 1-3\n(white 0, black 1); dots:\nthe pair shown", ha="left",
             va="top", fontsize=FS["note"], color="0.3", linespacing=1.25)


def still(group):
    """the t = 30 frame of the FR group's movie -> RGB image"""
    cm = _data()["cm"]
    T, W_IN, BS, X0, XGAP, FS = cm.T, cm.W_IN, cm.BS, cm.X0, cm.XGAP, cm.FS
    data, m = group_data(group)
    taus = [tau for tau, _ in cm.schedule()][::6] + [T - 1.]                     # the movie's frames (its boxes)
    with plt.rc_context(cm.RC):
        row_h = cm.YNAME + BS + (cm.STRIP + cm.STRIP_GAP)                         # (the movie's sum: H = 7.55 in)
        H = cm.YTOP + 2 * row_h + .12
        fig = plt.figure(figsize=(W_IN, H), dpi=cm.DPI, facecolor="white")
        for r, agents in enumerate(cm.AGENTS):
            y = H - cm.YTOP - cm.YNAME - BS - r * row_h
            for p, a in enumerate(agents):
                D, x = data[a], X0 + p * (BS + XGAP)
                ax = fig.add_axes([x / W_IN, y / H, BS / W_IN, BS / H])
                xl, yl = _box(D, taus)
                ax.set_xlim(*xl), ax.set_ylim(*yl)
                ax.set_aspect("equal", adjustable="box")
                ax.axis("off")
                fig.text((x + BS / 2) / W_IN, (y + BS + .22) / H, cm.NAME[a], ha="center", va="baseline",
                         fontsize=FS["name"], color=cm.COL[a])
                fig.text((x + BS / 2) / W_IN, (y + BS + .07) / H, "PCs %d-%d of step $t$" % tuple(D["S"]["pcs"][T - 1]),
                         ha="center", va="baseline", fontsize=FS["note"], color="0.45")
                cm._strip(fig, x, y, D["S"]).set_xdata([T, T])
                tr, hd, hm = _at(D, T - 1.)
                ax.add_collection(LineCollection(list(tr), colors=m["rgba"], linewidths=STY["lw"], zorder=1,
                                                 capstyle="round", joinstyle="round", antialiased=True))
                ax.scatter(*hm.T, s=STY["home_ms"] ** 2, facecolors="none", edgecolors=m["home_col"],
                           linewidths=STY["home_mew"], zorder=2)
                ax.scatter(*hd.T, s=STY["ms"] ** 2, c=m["col"], lw=0, zorder=3, antialiased=True)
        ty = H - .33                                                              # the clock, at t = T
        fig.text(X0 / W_IN, ty / H, f"$t = {T}$", fontsize=FS["t"], ha="left", va="center")
        bx0, bx1 = X0 + .95, X0 + 2 * BS + XGAP
        bar = fig.add_axes([bx0 / W_IN, (ty - .05) / H, (bx1 - bx0) / W_IN, .10 / H])
        bar.set_xlim(.5, T + .5), bar.set_ylim(0, 1)
        bar.axis("off")
        bar.barh(.5, T, left=.5, height=1, color="0.92", lw=0)
        bar.barh(.5, T, left=.5, height=1, color="0.35", lw=0)
        for t_ in (1, T):
            fig.text((bx0 + (t_ - .5) / T * (bx1 - bx0)) / W_IN, (ty - .2) / H, f"${t_}$", ha="center", va="baseline",
                     fontsize=FS["tick"], color="0.4")
        _key(fig, H, m)
        im = cm._frame(fig)
        plt.close(fig)
    return im


def figure(path=None):
    """the two frames side by side -> path (default figure_bin/fig1c_traj_FR_groups_T30.png)"""
    from main.utils import fig_path
    a, b = still("low"), still("high")
    out = Image.new("RGB", (a.size[0] + b.size[0], a.size[1]), "white")
    out.paste(a, (0, 0))
    out.paste(b, (a.size[0], 0))
    path = path or fig_path("fig1c_traj_FR_groups_T30.png")
    out.save(path)
    return path


if __name__ == "__main__":
    print("saved", figure())
