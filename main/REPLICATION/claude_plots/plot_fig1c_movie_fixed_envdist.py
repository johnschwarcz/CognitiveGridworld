"""The fixed movie and the envdist movie as one 2 x 4 movie (27 Oct 2026)
-> main/REPLICATION/figure_bin/fig1c_movie_fixed_envdist.gif

Columns: Optimal, Trained RNN, Naive, ESN. Every panel is the agent's own T = 30 PC plane (fig1.belief_geometry),
the same in both rows and in every frame, at one scale per agent (one box holds both rows' contents at every step).
    row 1  fig1_movie's fixed movie: the beliefs of the trained harvest's first 20,000 episodes, each a dot
           coloured by its true cell (r_g, r_c), and the mean belief paths of panel (c)'s four cells (markers: start,
           now, target)
    row 2  plot_fig1c_mesh's envdist movie: each of the 39 alias-scan environments' grids (each cell's mean belief over
           its 50 streams, joined to its grid neighbours), each edge coloured by its distance from the environments'
           average grid (black), in cells of the targets' grid; dashed: the targets' grid
    column 5  the keys: the latent space (large), the markers and paths of row 1, the colour bar of row 2
Timing: fig1_movie.schedule() at its full 25 frames per second (beliefs between steps by its Catmull-Rom blend).
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_movie_fixed_envdist.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.colors import to_rgb                                              # noqa: E402
from PIL import Image                                                             # noqa: E402

ORDER = ("Joint", "Trained", "Naive", "Echo")
DPI = 100
BS, XG, X0 = 2.95, .3, .55                                                       # panel side, column gap, left edge
Y_CLOCK, Y_NAME, Y_GAP = .8, .62, .2                                             # top band, names band, row gap
KX, KW = X0 + 4 * BS + 3 * XG + .5, 2.55                                         # the keys' column: left edge, width
W = KX + KW + .25
H = Y_CLOCK + Y_NAME + BS + Y_GAP + BS + .3
FS = dict(t=15, name=13.5, plane=10, row=11.5, lab=12, key=11, tick=10)
ENV_LW, ENV_ALPHA = .5, .7


def movie():
    import fig1 as cf
    import fig1_movie as cmv
    import plot_fig1c_mesh as pm
    from main.utils import fig_path
    T = cmv.T
    data, cells = cmv.movie_data("fixed")
    envs = tuple(e for e in range(40) if e != 2)
    Dm, _, edges, _, cm = pm.load(envs)
    sched = cmv.schedule()
    scale = BS / 1.11
    with plt.rc_context(cmv.RC):
        fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor="white")
        ax_ = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
        dist_norm = plt.Normalize(0, pm.DIST_MAX)
        y1 = H - Y_CLOCK - Y_NAME - BS
        y2 = y1 - Y_GAP - BS
        taus = [tau for tau, _ in sched][::6] + [T - 1.]
        cols = []
        for c_, a in enumerate(ORDER):
            D, Dg = data[a], Dm[a]
            x = X0 + c_ * (BS + XG)
            # one square box per agent for both rows: the cloud, the paths, the targets and every environment's grid
            P = np.stack([Dg["proj"](Dg["Mn"][:, :, t]) for t in range(T)], 2)       # (E, K, T, 2)
            (x0r, x1r), (y0r, y1r) = cmv.limits(D, taus)
            lo = np.minimum([x0r, y0r], np.minimum(P.reshape(-1, 2).min(0), Dg["home"].min(0)))
            hi = np.maximum([x1r, y1r], np.maximum(P.reshape(-1, 2).max(0), Dg["home"].max(0)))
            ctr, side = (lo + hi) / 2, (hi - lo).max() * 1.02
            xl, yl = (ctr[0] - side / 2, ctr[0] + side / 2), (ctr[1] - side / 2, ctr[1] + side / 2)
            fig.text((x + BS / 2) / W, (y1 + BS + .3) / H, cmv.NAME[a], ha="center", va="baseline",
                     fontsize=FS["name"], color=cmv.COL[a])
            fig.text((x + BS / 2) / W, (y1 + BS + .1) / H, "$T = 30$ PCs %d-%d" % D["C"].D30["pcs"], ha="center",
                     va="baseline", fontsize=FS["plane"], color="0.45")
            # row 1: the particles (fig1_movie.build, fixed)
            a1 = ax_(x, y1, BS, BS)
            a1.set_xlim(*xl), a1.set_ylim(*yl), a1.set_aspect("equal", adjustable="box"), a1.axis("off")
            sc = a1.scatter(*D["C"].proj(D["V"][:, 0], 0.).T, lw=0, zorder=1, marker="s", s=(cmv.PX * 72 / DPI) ** 2,
                            antialiased=False, c=D["col"])
            homes, halos, lines, starts, heads = [], [], [], [], []
            for j, (rg_, rc_) in enumerate(cells):
                mk = cmv.MARKS[j]
                sf = cmv.MARK_SCALE.get(mk, 1)
                homes.append(a1.plot([], [], marker=mk, ms=cmv.MS_HOME * sf, mfc="white", mec="k", mew=1.1, ls="none",
                                     zorder=6.8, clip_on=False,
                                     path_effects=[cmv.pe.Stroke(linewidth=2.6, foreground="white"),
                                                   cmv.pe.Normal()])[0])
                halos.append(a1.add_collection(cmv.LineCollection([], colors="white", capstyle="round", zorder=6.9)))
                lines.append(a1.add_collection(cmv.LineCollection([], capstyle="round", zorder=7)))
                starts.append(a1.plot([], [], marker=mk, ms=cmv.MS_START * sf, mfc="white", mec="k", mew=.8, ls="none",
                                      zorder=8 if sf > 1 else 8.1, clip_on=False)[0])
                heads.append(a1.plot([], [], marker=mk, ms=cmv.MS_END * sf, mfc=cf.bivariate(rg_, rc_), mec="k",
                                     mew=.8, ls="none", zorder=8.2 if sf > 1 else 8.3, clip_on=False)[0])
            # row 2: every environment's grid (plot_fig1c_mesh, envdist)
            a2 = ax_(x, y2, BS, BS)
            a2.set_xlim(*xl), a2.set_ylim(*yl), a2.set_aspect("equal", adjustable="box")
            a2.set_xticks([]), a2.set_yticks([])
            for sp in a2.spines.values():
                sp.set_color("0.75"), sp.set_linewidth(.6)
            hm = Dg["home"]
            a2.add_collection(cmv.LineCollection(hm[edges], colors="0.78", linewidths=.55, linestyles=(0, (3, 2)),
                                                 zorder=0))
            unit = np.linalg.norm(hm[edges[:, 0]] - hm[edges[:, 1]], axis=1).mean()   # one cell of the targets' grid
            env_lc = []
            for k in range(len(envs)):
                lc = cmv.LineCollection(P[k, :, 0][edges], cmap=pm.FR_CMAP, norm=dist_norm, linewidths=ENV_LW,
                                        alpha=ENV_ALPHA, zorder=2)
                lc.set_array(np.zeros(len(edges)))
                env_lc.append(a2.add_collection(lc))
            avg = a2.add_collection(cmv.LineCollection(P[:, :, 0].mean(0)[edges], colors="k", linewidths=1.2, zorder=4))
            cols.append(dict(a=a, D=D, Dg=Dg, sc=sc, homes=homes, halos=halos, lines=lines, starts=starts, heads=heads,
                             env_lc=env_lc, avg=avg, unit=unit))
        # row labels
        for y, txt in ((y1, "beliefs of 20,000 episodes"), (y2, "grids of 39 environments")):
            fig.text((X0 - .22) / W, (y + BS / 2) / H, txt, rotation=90, ha="center", va="center", fontsize=FS["row"],
                     color="0.3")
        # the clock: t over a bar of the T steps
        ty = H - .42
        t_txt = fig.text(X0 / W, ty / H, "", fontsize=FS["t"], ha="left", va="center")
        bx0, bx1 = X0 + 1.15, X0 + 4 * BS + 3 * XG
        bar = ax_(bx0, ty - .06, bx1 - bx0, .12)
        bar.set_xlim(.5, T + .5), bar.set_ylim(0, 1), bar.axis("off")
        bar.barh(.5, T, left=.5, height=1, color="0.92", lw=0)
        fill = bar.barh(.5, 0, left=.5, height=1, color="0.35", lw=0)[0]
        for t_ in (1, T):
            fig.text((bx0 + (t_ - .5) / T * (bx1 - bx0)) / W, (ty - .3) / H, f"${t_}$", ha="center", va="baseline",
                     fontsize=FS["tick"], color="0.4")
        _keys(fig, cf, cmv, pm, cells, dist_norm, y1, y2)

    def update(tau):
        for col in cols:
            D, Dg = col["D"], col["Dg"]
            C = D["C"]
            col["sc"].set_offsets(C.proj(cmv._blend(D["V"], tau, axis=1), tau))
            Hm = C.proj(D["home"], tau)
            for j, (tt, Pp) in enumerate(cmv.trails(D, tau)):
                col["homes"][j].set_data([Hm[j, 0]], [Hm[j, 1]])
                col["starts"][j].set_data([Pp[0, 0]], [Pp[0, 1]])
                seg, e, q = cmv._tail(tt, Pp, tau)
                wd = (.45 + 1.25 * e) * scale ** .6
                col["halos"][j].set_segments(seg), col["halos"][j].set_linewidths(wd + 1.4)
                col["lines"][j].set_segments(seg), col["lines"][j].set_linewidths(wd)
                col["lines"][j].set_colors([str(.6 * (1 - v)) for v in e])
                col["heads"][j].set_data([q[0]], [q[1]])
            Mt = cm._blend(Dg["Mn"], tau, axis=2)                                    # (E, K, 20)
            Pt = np.stack([Dg["proj"](Mt[k]) for k in range(len(Mt))])
            pa = Pt.mean(0)
            col["avg"].set_segments(pa[edges])
            for k, lc in enumerate(col["env_lc"]):
                dv = np.linalg.norm(Pt[k] - pa, axis=1)
                lc.set_segments(Pt[k][edges])
                lc.set_array((dv[edges[:, 0]] + dv[edges[:, 1]]) / 2 / col["unit"])
        t_txt.set_text(f"$t = {int(np.floor(tau + .5)) + 1}$")
        fill.set_width(tau + 1)

    def grab():
        with plt.rc_context(cmv.RC):
            fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    protos = []
    for tau in (0., 4., 9., T - 1.):
        update(tau)
        protos.append(grab())
    pal = _palette(cf, cmv, pm, protos)
    frames = []
    for i, (tau, _) in enumerate(sched):
        update(tau)
        frames.append(Image.fromarray(grab()).quantize(palette=pal, dither=Image.Dither.NONE))
        if i % 50 == 0:
            print(f"frame {i + 1} of {len(sched)}", flush=True)
    plt.close(fig)
    path = fig_path("fig1c_movie_fixed_envdist.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[ms for _, ms in sched], loop=0)
    print(f"saved {path}: {len(frames)} frames, {sum(ms for _, ms in sched) / 1000:.1f} s, {frames[0].size[0]} x "
          f"{frames[0].size[1]} px, {os.path.getsize(path) / 1e6:.1f} MB")
    return path


def _palette(cf, cmv, pm, protos, n=255):
    """an explicit GIF palette: every colour the movie draws, exactly (the 100 cells' colours of the dots and the key,
    the agents' name colours, greys, the distance colour ramp at full strength and at the grids' alpha over white); the
    remaining slots from a median cut of sample frames (anti-aliased edges, blends)"""
    R = cf.N_R
    exact = [to_rgb(c) for c in cmv.COL.values()]
    exact += [tuple(np.asarray(cf.bivariate(g, c), np.float64)[:3]) for g in range(R) for c in range(R)]
    exact += [(v, v, v) for v in np.linspace(0, 1, 26)]
    ramp = [np.asarray(pm.FR_CMAP(v)[:3]) for v in np.linspace(0, 1, 24)]
    exact += [tuple(c) for c in ramp] + [tuple(ENV_ALPHA * c + (1 - ENV_ALPHA)) for c in ramp]
    exact = np.unique((255 * np.clip(np.array(exact, np.float64), 0, 1)).round().astype(np.uint8), axis=0)
    rest = n - len(exact)
    mc = Image.fromarray(np.concatenate(protos, 0)).quantize(colors=rest, method=Image.Quantize.MEDIANCUT,
                                                              dither=Image.Dither.NONE)
    extra = np.array(mc.getpalette(), np.uint8).reshape(-1, 3)[:rest]
    allc = np.concatenate([exact, extra], 0)[:n]
    pimg = Image.new("P", (1, 1))
    pimg.putpalette(allc.ravel().tolist() + [255, 255, 255] * (256 - len(allc)))
    print(f"palette: {len(exact)} exact colours + {len(extra)} from the frames", flush=True)
    return pimg


def _keys(fig, cf, cmv, pm, cells, dist_norm, y1, y2):
    """column 5: the latent space (beside row 1); the markers and paths of row 1, the colour bar and lines of row 2
    (beside row 2); y1, y2: the rows' bottoms (inches)"""
    from matplotlib.lines import Line2D
    T = cmv.T
    # beside row 1: the latent space, large
    ks = 2.35
    kx = KX + (KW - ks) / 2 + .12
    fig.text((kx + ks / 2) / W, (y1 + .3 + ks + .1) / H, "Latent space", ha="center", va="bottom", fontsize=FS["lab"])
    ax = fig.add_axes([kx / W, (y1 + .3) / H, ks / W, ks / H])
    R = cf.N_R
    gg, cc = np.meshgrid(np.arange(R), np.arange(R))
    ax.imshow(cf.bivariate(gg, cc), origin="lower", extent=(-.5, R - .5, -.5, R - .5), interpolation="nearest")
    for (rg_, rc_), m in zip(cells, cmv.MARKS):
        ax.scatter(rg_, rc_, marker=m, s=(.8 * cmv.MS_HOME * cf.MARK_SCALE_A.get(m, 1)) ** 2, facecolor="white",
                   edgecolor="k", lw=1., clip_on=False, zorder=3)
    ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
    ax.tick_params(length=2, pad=1.5, labelsize=FS["tick"])
    ax.set_xlabel("$r_g$", labelpad=-6, fontsize=FS["lab"])
    ax.set_ylabel("$r_c$", labelpad=-6, fontsize=FS["lab"])
    # beside row 2: the markers and paths of row 1
    x = KX + .05
    y = y2 + BS - .12
    for ms, mew, fc, lab in ((cmv.MS_START, .8, "white", "start ($t = 1$)"), (cmv.MS_END, .8, "0.55", "now"),
                             (cmv.MS_HOME, 1.1, "white", "target")):
        fig.add_artist(Line2D([(x + .2) / W], [y / H], transform=fig.transFigure, marker="o", ms=ms * .85, mfc=fc,
                              mec="k", mew=mew, ls="none"))
        fig.text((x + .5) / W, y / H, lab, ha="left", va="center", fontsize=FS["key"], color="0.2")
        y -= .34
    ex = np.linspace(x + .02, x + .4, 30)
    ev = np.log1p((np.arange(29) + .5) / 29 * (T - 1)) / np.log(T)
    fig.add_artist(cmv.LineCollection(np.stack([np.stack([ex[:-1], ex[1:]], 1) / W, np.full((29, 2), y / H)], -1),
                                      transform=fig.transFigure, colors=[str(.6 * (1 - v)) for v in ev],
                                      linewidths=(.45 + 1.25 * ev) * (BS / 1.11) ** .6, capstyle="round"))
    fig.text((x + .5) / W, y / H, "a cell's mean path", ha="left", va="center", fontsize=FS["key"], color="0.2")
    # the colour bar and lines of row 2
    yb = y - .72
    cax = fig.add_axes([(x + .05) / W, yb / H, (KW - .25) / W, .14 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=pm.FR_CMAP, norm=dist_norm), cax=cax, orientation="horizontal",
                      extend="max")
    cb.ax.tick_params(labelsize=FS["tick"], length=2, pad=1)
    cb.outline.set_visible(False)
    fig.text((x + .05) / W, (yb + .24) / H, "distance from the average (cells)", ha="left", va="bottom",
             fontsize=FS["key"], color="0.2")
    y = yb - .55
    for ls, c, lw, lab in (("-", "k", 1.4, "average grid"), ((0, (3, 2)), "0.6", 1., "targets' grid")):
        fig.add_artist(Line2D([(x + .02) / W, (x + .4) / W], [y / H] * 2, transform=fig.transFigure, c=c, lw=lw, ls=ls))
        fig.text((x + .5) / W, y / H, lab, ha="left", va="center", fontsize=FS["key"], color="0.2")
        y -= .32


if __name__ == "__main__":
    movie()
