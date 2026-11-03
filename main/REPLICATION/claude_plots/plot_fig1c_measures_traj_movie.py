"""The nine measures on the trajectories of the split pairs' environments (28 Oct 2026; John: "can we make a gif like
the attached (the splitting traj's environments) for our measures")
-> main/REPLICATION/figure_bin/fig1c_movie_measures_traj.gif          raw values
   main/REPLICATION/figure_bin/fig1c_movie_measures_traj_vs_step.gif  colour normalized per step (John: "can we also
       have a version that normalizes color per step"): each value minus the mean of all the scan's streams (39
       environments x 5,000) at that step, in the same units (additive, no rescaling), coolwarm, red above the step's
       mean, blue below, +- the 98th percentile of the drawn departures

As row 3 ("their environments") of plot_fig1c_movie_split_traj: the alias-scan environments of the three split pairs
(15, 39, 20), one stream per cell (the first of each cell's 50; 300 streams, the same fixed random drawing order), each
the Trained RNN's read-out in its fixed T = 30 PC plane (fig1.belief_geometry), drawn as its path so far (Catmull-Rom,
SUB points per step) with its head; open circles: the cells' targets (fig1.targets). Here 3 x 3 panels, one per measure
of plot_fig1c_measures_movie (the Trained RNN's), each path coloured along its length by that stream's measure at each
step (linear between steps) and its head by the measure now: raw values on each measure's own scale (the 2nd - 98th
percentile of the drawn values; log10 for positive heavy-tailed measures, as the measures movies), viridis, bright =
more; the measures movies' clock and timing.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_measures_traj_movie.py [raw|vs_step]"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402
import plot_fig1c_measures_movie as mm                                            # noqa: E402
import plot_fig1c_movie_split_traj as st                                          # noqa: E402

AGENT = "Trained"
SUB = 4                                                                           # path points per step (split: 8)
LW, HEAD, HOME = .6, 2.6, 4.4
DPI = 110


def streams(envs):
    """the split movie's row-3 particles (environment, stream), in its drawing order (st.traj_data)"""
    meta = np.load(os.path.join(st.SCAN, "scan_meta.npz"))
    js = np.arange(len(meta["cells"])) * st.M
    ev, js = np.repeat(np.asarray(envs, int), len(js)), np.tile(js, len(envs))
    p = np.random.default_rng(0).permutation(len(js))
    return ev[p], js[p]


def movie(mode="raw"):
    import plot_fig1c_split_movie as sm
    Z = np.load(sm.DATA)
    envs = [int(e) for e in Z["shown_env"]]
    D = st.traj_data(envs)[AGENT]
    ev, js = streams(envs)
    assert len(ev) == len(D["V2"])
    k_sub = st.SUB // SUB
    tt, P2 = D["tt"][::k_sub], D["P2"][:, ::k_sub]
    taus = mm._taus()
    xl, yl = st.traj_limits(D, taus[::10])
    _, _, _, strm, _ = mm.data(raw=True)
    rows = np.array([mm.ENVS.index(e) for e in ev])
    vals, lims, logs = {}, {}, {}
    for key, *_ in mm.PANELS:
        allv = strm[key]
        logs[key] = bool((allv > 0).all()) and np.percentile(allv, 98) > 20 * np.percentile(allv, 2)
        V = allv[rows, js].astype(np.float64)                                        # (300, T)
        if logs[key]:
            V = np.log10(V)
        if mode == "vs_step":                                                      # minus the step's scan mean
            V = V - (np.log10(allv) if logs[key] else allv).mean((0, 1))
            L_ = np.percentile(np.abs(V), 98)
            lims[key] = (-L_, L_)
        else:
            lims[key] = tuple(np.percentile(V, [2, 98]))
        vals[key] = V
    del strm
    sub_v = {k: np.stack([np.interp(tt, np.arange(mm.T), v) for v in V]) for k, V in vals.items()}
    cmap = plt.get_cmap("coolwarm" if mode == "vs_step" else "viridis")
    PW, GX, GY, X0, YTOP = 2.35, .35, 1.4, .45, 1.0
    W, H = X0 + 3 * PW + 2 * GX + 1.45, YTOP + 3 * PW + 2 * GY + .85
    with plt.rc_context(mm.RC):
        fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor="white")
        arts = []
        for n, (key, name, formula) in enumerate(mm.PANELS):
            r_, c_ = divmod(n, 3)
            x, y = X0 + c_ * (PW + GX), H - YTOP - PW - r_ * (PW + GY)
            ax = fig.add_axes([x / W, y / H, PW / W, PW / H])
            ax.set_xlim(*xl), ax.set_ylim(*yl), ax.set_aspect("equal", adjustable="box"), ax.axis("off")
            lc = ax.add_collection(LineCollection([], linewidths=LW, zorder=1, capstyle="round", antialiased=True))
            ax.scatter(*D["home2"].T, s=HOME ** 2, facecolors="none", edgecolors="0.55", linewidths=.6, zorder=2)
            hd = ax.scatter(*D["V2"][:, 0].T, s=HEAD ** 2, c=cmap(np.full(len(ev), .5)), lw=0, zorder=3)
            ax.set_title(name + "\n" + formula, fontsize=9.5, pad=4)
            lo_k, hi_k = lims[key]
            cax = fig.add_axes([x / W, (y - .42) / H, PW / W, .07 / H])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(lo_k, hi_k), cmap=cmap), cax=cax,
                              orientation="horizontal")
            cb.set_ticks([lo_k, (lo_k + hi_k) / 2, hi_k])
            cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2g}"))
            cb.ax.tick_params(labelsize=7, length=2, pad=1)
            cb.outline.set_visible(False)
            cb.set_label(("log10 " if logs[key] else "") + mm.UNITS[key]
                         + (" minus the step's mean" if mode == "vs_step" else ""), fontsize=7.5, labelpad=1)
            arts.append((key, lc, hd))
        clock = mm._clock(fig, W, H, X0, X0 + 3 * PW + 2 * GX)
        fig.text((X0 + 3 * PW + 2 * GX + .2) / W, (H - .5) / H, f"environments\n{', '.join(map(str, envs))} (the\n"
                 f"split pairs'), one\nstream per cell:\n{len(ev)} streams", fontsize=10, ha="left", va="center")
        fig.text((X0 + 3 * PW + 2 * GX + .2) / W, (H - 1.35) / H, "Trained RNN,\nits T = 30\nPC plane;\no: targets",
                 fontsize=9, ha="left", va="center", color="0.35")
        frames = []
        for i, tau in enumerate(taus):
            k = int(np.searchsorted(tt, tau, side="right"))
            head = mm._blend(D["V2"], tau, 1)                                           # (300, 2)
            trail = np.concatenate([P2[:, :k], head[:, None]], 1)                        # (300, k + 1, 2)
            segs = np.stack([trail[:, :-1], trail[:, 1:]], 2).reshape(-1, 2, 2)
            for key, lc, hd in arts:
                lo_k, hi_k = lims[key]
                c_seg = sub_v[key][:, :k].reshape(-1)
                lc.set_segments(segs)
                lc.set_color(cmap(np.clip((c_seg - lo_k) / (hi_k - lo_k), 0, 1)))
                hd.set_offsets(head)
                hd.set_facecolor(cmap(np.clip((mm._lin(vals[key], tau) - lo_k) / (hi_k - lo_k), 0, 1)))
            clock(tau)
            frames.append(mm._grab(fig))
            if i % 100 == 0:
                print(f"frame {i + 1} of {len(taus)}", flush=True)
        plt.close(fig)
    return mm._save(frames, cmap, "fig1c_movie_measures_traj%s.gif" % ("_vs_step" if mode == "vs_step" else ""),
                    levels=64)


if __name__ == "__main__":
    print(movie(sys.argv[1] if len(sys.argv) > 1 else "raw"))
