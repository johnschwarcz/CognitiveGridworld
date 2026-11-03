"""The moment of unfolding, seen on the grid: every stream of the alias scan aligned on an event, then averaged into
the environments' average grid at each step relative to the event (25 Oct 2026)
-> main/REPLICATION/figure_bin/fig1c_event_unfold_<event>.png and fig1c_movie_event_unfold_<event>.gif

Events (per stream; the same steps for all four agents, which all see the same observations):
    unfold   the Optimal's moment of unfolding: its largest one-step drop in the distance from the stream's own target
             (fig1.targets) within steps 6..25
    ent      the stream's peak entanglement (house rev) within steps 1..12 (streams with the peak at steps 6..12)
Event-aligned grid at relative step tau: each cell's mean over its aligned streams of the belief at t* + tau (20
numbers), averaged over the 39 environments; control: the same with every stream replaced by another stream of the
same cell and environment read at the same absolute steps (so the control has the event's timing but not its event).
Drawn in panel (c)'s fixed T = 30 plane of each agent (fig1.belief_geometry); dashed: the targets' grid; thin
grey: the control's grid; edges: factorization regret of the edge's cells at T.
Cache: main/DATA/experimental/fig1c_compression/figdata/round6_hidden/event_unfold.npz (build: about 3 min CPU).
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_event_unfold.py [unfold|ent] [movie]"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402
from matplotlib.colors import LogNorm                                             # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402

SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
CACHE = os.path.join(SCAN, "figdata", "round6_hidden", "event_unfold.npz")
ENVS = [e for e in range(40) if e != 2]
E, K, M, T = 39, 100, 50, 30
TAU = np.arange(-5, 6)
ORDER = ("Joint", "Trained", "Naive", "Echo")
NAME = {"Joint": "Optimal", "Trained": "Trained RNN", "Naive": "Naive", "Echo": "ESN"}
COLOR = {"Joint": "k", "Trained": "#1f77b4", "Naive": "0.5", "Echo": "#ff7f0e"}
EVENT_NAME = {"unfold": "the Optimal's moment of unfolding", "ent": "the stream's peak entanglement (steps 1-12)"}


def _partners(seed=0):
    """for every stream, a different stream of the same cell and environment (a derangement within each cell)"""
    rng = np.random.default_rng(seed)
    part = np.empty((E, K * M), int)
    for e in range(E):
        for k in range(K):
            idx = np.arange(k * M, (k + 1) * M)
            p = rng.permutation(M)
            while (p == np.arange(M)).any():
                p = rng.permutation(M)
            part[e, idx] = idx[p]
    return part


def build():
    import fig1 as cf
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    assert (meta["cell_of"] == np.repeat(np.arange(K), M)).all()
    cells = meta["cells"]
    rev = meta["rev"][ENVS].astype(np.float64)                                       # (E, 5000, T)
    home, _ = cf.targets([tuple(int(v) for v in c) for c in cells])
    B = {a: np.load(os.path.join(SCAN, f"scan_{a}.npz"))["beliefs"][ENVS].astype(np.float32) for a in ORDER}
    tgt = home[np.repeat(np.arange(K), M)].astype(np.float32)
    dJ = np.linalg.norm(B["Joint"] - tgt[None, :, None], axis=-1)                    # (E, 5000, T)
    d0 = np.linalg.norm(np.full(20, .1, np.float32) - tgt, axis=-1)                  # the uniform belief's distance
    drop = np.concatenate([d0[None, :, None] - dJ[:, :, :1], dJ[:, :, :-1] - dJ[:, :, 1:]], -1)
    events = {"unfold": (drop[:, :, 5:25].argmax(-1) + 5, None),
              "ent": (rev[:, :, :12].argmax(-1), lambda ev: ev >= 5)}
    part = _partners()
    ar_e, ar_s = np.arange(E)[:, None], np.arange(K * M)[None]
    out = dict(tau=TAU, home=home, fr=meta["FR"][ENVS].astype(np.float64).reshape(E, K, M, T).mean(2)[:, :, -1].mean(0))
    for name, (ev, vfun) in events.items():
        valid = np.ones_like(ev, bool) if vfun is None else vfun(ev)
        w = valid[..., None].astype(np.float32)
        out[f"{name}_tstar"], out[f"{name}_valid"] = ev, valid
        out[f"{name}_ent_ev"] = np.array([(rev[ar_e, ar_s, np.clip(ev + t, 0, T - 1)] * valid).sum() / valid.sum()
                                          for t in TAU])
        out[f"{name}_ent_ct"] = np.array([(rev[ar_e, part, np.clip(ev + t, 0, T - 1)] * valid).sum() / valid.sum()
                                          for t in TAU])
        for a in ORDER:
            ge, gc = [], []
            for t in TAU:
                tt = np.clip(ev + t, 0, T - 1)
                for src, acc in ((ar_s, ge), (part, gc)):
                    X = B[a][ar_e, src, tt] * w                                       # (E, 5000, 20)
                    acc.append(X.reshape(E, K, M, 20).sum(2) / w.reshape(E, K, M, 1).sum(2))
            out[f"{name}_{a}_ev"] = np.stack(ge, 2).astype(np.float32)               # (E, K, tau, 20)
            out[f"{name}_{a}_ct"] = np.stack(gc, 2).astype(np.float32)
            print(name, a, "done", flush=True)
    np.savez(CACHE, **out)
    return out


def load():
    return dict(np.load(CACHE)) if os.path.exists(CACHE) else build()


def _setup():
    import fig1 as cf
    import fig1_movie as cm
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    cells = meta["cells"]
    idx = {(int(g), int(c)): k for k, (g, c) in enumerate(cells)}
    edges = np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                      if (g + dg, c + dc) in idx])
    G = cf.belief_geometry()
    proj = {a: (lambda V, Dg=G[(a, T)]: cm._proj(Dg, np.asarray(V, np.float64))) for a in ORDER}
    return edges, proj


def _dist(Z, name, a):
    """distance of the event-aligned and the control average grid from the targets per cell: (tau, K) each"""
    h = Z["home"][:, None]
    return (np.linalg.norm(np.nanmean(Z[f"{name}_{a}_ev"], 0) - h, axis=-1).T,
            np.linalg.norm(np.nanmean(Z[f"{name}_{a}_ct"], 0) - h, axis=-1).T)


def figure(name="unfold"):
    from main.utils import apply_fig_style, fig_path
    import plot_fig1c_mesh as pm
    Z = load()
    edges, proj = _setup()
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    show = (-3, -2, -1, 0, 1, 2, 3)
    pw, gap = 1.55, .1
    W = .9 + len(show) * pw + (len(show) - 1) * gap + .25 + .12 + .9
    H = .55 + 4 * (pw + .1) + .75 + 1.9 + 1.0
    fig = plt.figure(figsize=(W, H))
    norm = LogNorm(.05, 30)
    frT = np.maximum(Z["fr"], 1e-6)
    ecol = np.sqrt(frT[edges[:, 0]] * frT[edges[:, 1]])
    for r, a in enumerate(ORDER):
        ev, ct = np.nanmean(Z[f"{name}_{a}_ev"], 0), np.nanmean(Z[f"{name}_{a}_ct"], 0)   # (K, tau, 20)
        hm = proj[a](Z["home"])
        allp = np.vstack([proj[a](ev[:, list(TAU).index(t)]) for t in show] + [hm])
        xl, yl = pm._box(allp)
        y0 = H - .55 - (r + 1) * (pw + .1)
        for c, t in enumerate(show):
            j = list(TAU).index(t)
            ax = fig.add_axes([(.9 + c * (pw + gap)) / W, y0 / H, pw / W, pw / H])
            ax.add_collection(LineCollection(hm[edges], colors="0.82", linewidths=.45, linestyles=(0, (3, 2))))
            ax.add_collection(LineCollection(proj[a](ct[:, j])[edges], colors="0.6", linewidths=.5, alpha=.8))
            lc = LineCollection(proj[a](ev[:, j])[edges], cmap=pm.FR_CMAP, norm=norm, linewidths=1.1)
            lc.set_array(ecol)
            ax.add_collection(lc)
            ax.set_xlim(*xl), ax.set_ylim(*yl), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(1.6 if t == 0 else .6), sp.set_color("k" if t == 0 else "0.5")
            if r == 0:
                ax.set_title(r"$\tau = %+d$" % t if t else r"$\tau = 0$ (event)", fontsize=FS["label"], pad=3)
            if c == 0:
                ax.text(-.08, .5, NAME[a], transform=ax.transAxes, rotation=90, ha="right", va="center",
                        fontsize=FS["label"] + .5, color=COLOR[a] if a != "Naive" else "0.35")
    cx = fig.add_axes([(.9 + len(show) * pw + (len(show) - 1) * gap + .25) / W, (H - .55 - 4 * (pw + .1)) / H,
                       .12 / W, (4 * (pw + .1) - .1) / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=pm.FR_CMAP, norm=norm), cax=cx, extend="both")
    cb.set_label("factorization regret of the edge's cells at $T$ (nats)", fontsize=FS["tick"])
    cb.outline.set_visible(False)
    fig.text(.9 / W, (H - .25) / H, "Every stream aligned on %s ($\\tau$ = 0), then averaged into the environments' "
             "average grid; thin grey: time-matched control; dashed: targets" % EVENT_NAME[name],
             fontsize=FS["label"] + .5, va="center")
    fig.text(.15 / W, (H - .25) / H, r"$\mathbf{a}$", fontsize=FS["label"] + 2, va="center")
    # curves
    by, bh = 1.0, 1.9
    bw = (W - .9 - .9 - 2 * .95) / 3
    axs = [fig.add_axes([(.9 + i * (bw + .95)) / W, by / H, bw / W, bh / H]) for i in range(3)]
    q = np.quantile(frT, [.2, .8])
    for a in ORDER:
        de, dc = _dist(Z, name, a)
        for ax, sel in ((axs[0], frT >= q[1]), (axs[1], frT <= q[0])):
            ax.plot(TAU, (de - dc)[:, sel].mean(1), c=COLOR[a], marker="o", ms=2.5,
                    lw=2.6 if a == "Trained" else 1.3, alpha=.45 if a == "Trained" else 1, label=NAME[a])
    for ax, lab in ((axs[0], "highest-FR fifth of cells"), (axs[1], "lowest-FR fifth of cells")):
        ax.axhline(0, c="0.8", lw=.6, zorder=0), ax.axvline(0, c="0.8", lw=.6, zorder=0)
        ax.set_title("Distance from target, %s" % lab, fontsize=FS["label"] + .5, pad=4)
        ax.set_ylabel(r"$\|\bar X - \mathrm{target}\|$, event $-$ control")
    lo = min(ax.get_ylim()[0] for ax in axs[:2]); hi = max(ax.get_ylim()[1] for ax in axs[:2])
    for ax in axs[:2]:
        ax.set_ylim(lo, hi)
    axs[0].legend(frameon=False, fontsize=FS["tick"] - 1.5, loc="upper right", labelspacing=.2)
    axs[2].plot(TAU, Z[f"{name}_ent_ev"], c="#d7301f", marker="o", ms=2.5, lw=1.4, label="event streams")
    axs[2].plot(TAU, Z[f"{name}_ent_ct"], c="0.5", marker="o", ms=2.5, lw=1.2, label="control streams")
    axs[2].axvline(0, c="0.8", lw=.6, zorder=0)
    axs[2].set_title(r"Entanglement around the event, $\mathcal{E}_{t^* + \tau}$", fontsize=FS["label"] + .5, pad=4)
    axs[2].set_ylabel("entanglement (nats)")
    axs[2].legend(frameon=False, fontsize=FS["tick"] - 1.5, loc="upper left")
    for i, ax in enumerate(axs):
        ax.set_xlabel(r"steps from the event, $\tau$", labelpad=1), ax.set_xticks(TAU[::1])
        ax.spines[["top", "right"]].set_visible(False)
        x0, y0, w, h = ax.get_position().bounds
        fig.text(x0 - .7 / W, y0 + h + 6 / 72 / H, r"$\mathbf{%s}$" % "bcd"[i], fontsize=FS["label"] + 2, va="baseline")
    import textwrap
    nv = int(Z[f"{name}_valid"].sum())
    fig.text(.9 / W, .42 / H, textwrap.fill(
        "Alias scan, 39 environments x 100 cells x 50 streams; %s aligned streams. Event: %s. Control: each aligned "
        "stream replaced by another stream of the same cell and environment, read at the same absolute steps. a: each "
        "agent in its own T = 30 plane (panel c). b, c: per cell, the distance of the event-aligned average grid from "
        "the cell's target minus the control's, mean over the cells of the FR fifth (20 cells)." % (
            f"{nv:,}", EVENT_NAME[name]), 260), fontsize=FS["tick"] - 1.5, color="0.35", va="top", linespacing=1.35)
    path = fig_path(f"fig1c_event_unfold_{name}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("saved", path)
    for a in ORDER:
        de, dc = _dist(Z, name, a)
        for g, sel in (("high-FR", frT >= q[1]), ("low-FR", frT <= q[0])):
            d = (de - dc)[:, sel].mean(1)
            print("%-8s %-7s event - control at tau -1 / 0: %+.3f / %+.3f (swing %+.3f)" % (
                a, g, d[list(TAU).index(-1)], d[list(TAU).index(0)], d[list(TAU).index(0)] - d[list(TAU).index(-1)]))


def movie(name="unfold"):
    """2 x 2 movie of the event-aligned average grids, tau sweeping -5 .. +5 (smooth), control in grey"""
    from PIL import Image
    import fig1_movie as cm
    import plot_fig1c_mesh as pm
    from main.utils import fig_path
    Z = load()
    edges, proj = _setup()
    norm = LogNorm(.05, 30)
    frT = np.maximum(Z["fr"], 1e-6)
    ecol = np.sqrt(frT[edges[:, 0]] * frT[edges[:, 1]])
    pw, gap = 3.0, .4
    W, H = .3 + 2 * pw + gap + 1.25, .6 + 2 * (pw + .32) + .5
    fig = plt.figure(figsize=(W, H), facecolor="white", dpi=80)
    art = []
    for r, row in enumerate((("Joint", "Trained"), ("Naive", "Echo"))):
        for c, a in enumerate(row):
            ax = fig.add_axes([(.3 + c * (pw + gap)) / W, (H - .6 - (r + 1) * pw - r * .32) / H, pw / W, pw / H])
            ev, ct = np.nanmean(Z[f"{name}_{a}_ev"], 0), np.nanmean(Z[f"{name}_{a}_ct"], 0)
            hm = proj[a](Z["home"])
            xl, yl = pm._box(np.vstack([proj[a](ev[:, j]) for j in range(len(TAU))] + [hm]))
            ax.set_xlim(*xl), ax.set_ylim(*yl), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(NAME[a], fontsize=11, pad=3)
            ax.add_collection(LineCollection(hm[edges], colors="0.82", linewidths=.5, linestyles=(0, (3, 2))))
            lcc = ax.add_collection(LineCollection(proj[a](ct[:, 0])[edges], colors="0.6", linewidths=.6, alpha=.8))
            lce = LineCollection(proj[a](ev[:, 0])[edges], cmap=pm.FR_CMAP, norm=norm, linewidths=1.4)
            lce.set_array(ecol)
            ax.add_collection(lce)
            art.append((a, ev, ct, lce, lcc, ax))
    cx = fig.add_axes([(.3 + 2 * pw + gap + .25) / W, (H - .6 - 2 * pw - .32) / H, .13 / W, (2 * pw + .32) / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=pm.FR_CMAP, norm=norm), cax=cx, extend="both")
    cb.set_label("factorization regret of the edge's cells at T (nats)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    clock = fig.text(.3 / W, (H - .27) / H, "", fontsize=13, va="center")
    import textwrap
    fig.text(.3 / W, .38 / H, textwrap.fill("Every stream aligned on %s (tau = 0) and averaged into the environments' "
             "average grid (colour: FR at T); thin grey: time-matched control (other streams of the same cell at the same "
             "steps); dashed: targets." % EVENT_NAME[name], 125), fontsize=7, color="0.35", va="top", linespacing=1.3)
    xs = np.arange(len(TAU), dtype=float)

    def update(x):
        j = int(np.floor(x)); f = x - j; j1 = min(j + 1, len(TAU) - 1)
        for a, ev, ct, lce, lcc, ax in art:
            lce.set_segments(proj[a]((1 - f) * ev[:, j] + f * ev[:, j1])[edges])
            lcc.set_segments(proj[a]((1 - f) * ct[:, j] + f * ct[:, j1])[edges])
            hit = abs(x - list(TAU).index(0)) < .5
            for sp in ax.spines.values():
                sp.set_linewidth(2.2 if hit else .8), sp.set_color("#d7301f" if hit else "k")
        tau = TAU[0] + x
        clock.set_text("steps from the event: tau = %+.0f" % np.round(tau) if abs(tau - np.round(tau)) < .05 else
                       "steps from the event: tau = %+.1f" % tau)

    sched = []
    for j in range(len(TAU)):
        sched.append((float(j), 900 if TAU[j] in (-1, 0) else 450))
        if j < len(TAU) - 1:
            sched += [(j + f, 60) for f in (.25, .5, .75)]
    sched[-1] = (sched[-1][0], 1500)

    def grab():
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    protos = []
    for x in (0., 4., 5., 10.):
        update(x)
        protos.append(np.asarray(grab()))
    pal = Image.fromarray(np.concatenate(protos, 0)).quantize(colors=255, method=Image.Quantize.MEDIANCUT,
                                                              dither=Image.Dither.NONE)
    frames = []
    for x, _ in sched:
        update(x)
        frames.append(grab().quantize(palette=pal, dither=Image.Dither.NONE))
    plt.close(fig)
    path = fig_path(f"fig1c_movie_event_unfold_{name}.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[ms for _, ms in sched], loop=0)
    print(f"saved {path}: {len(frames)} frames, {os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "build" in args:
        build()
    for nm in [a for a in args if a in EVENT_NAME] or ["unfold"]:
        figure(nm)
        if "movie" in args:
            movie(nm)
