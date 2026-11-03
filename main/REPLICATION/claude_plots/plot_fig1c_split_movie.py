"""Pairs of trajectories that stay together, then split (25 Oct 2026)
-> main/REPLICATION/figure_bin/fig1c_movie_split_pairs.gif

Data: the alias scan (main/DATA/experimental/fig1c_compression; 39 environments, e = 2 left out; 100 cells x 50 streams x
30 steps; all four agents on the same streams). Position of a particle = its posterior means x = (E[r_g], E[r_c]), as in
plot_fig1c_splitting.py.
Pairs are selected in the Optimal: two streams of different cells in one environment, true cells at least 4 grid units
apart, within 0.6 of each other from step 2 through step t_s (within 0.4 at t_s; t_s = 5..22) and at least 3 further
apart two steps later; each pair kept once, at its last t_s.
Shown: three pairs (distinct environments and true cells; the largest jump in each window of t_s, latest window
first); all four agents, the same streams in every panel.
Rows: the posterior-mean plane (each stream's path faint before its pair's split step t_s, then solid); the distance
between the pair's two streams over time.
Pairs and statistics are cached in figdata/round4b/split_pairs.npz (--rebuild: from the scan, about 2 min).
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_split_movie.py [--rebuild]"""
import os
import sys
import numpy as np


def _root():
    for start in (globals().get("__file__"), globals().get("__vsc_ipynb_file__"), os.getcwd()):
        d = os.path.abspath(start) if start and os.path.exists(start) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return d
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError("project root not found")


ROOT = _root()
sys.path[:0] = [p for p in (ROOT, os.path.join(ROOT, "main", "REPLICATION", "claude_plots")) if p not in sys.path]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from PIL import Image                                                             # noqa: E402
from main.utils import apply_fig_style, fig_path                                  # noqa: E402

FS = apply_fig_style()
plt.rcParams["mathtext.fontset"] = "cm"
SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
DATA = os.path.join(SCAN, "figdata", "round4b", "split_pairs.npz")
AG = ("Joint", "Trained", "Naive", "Echo")
NAME = {"Joint": "Optimal", "Trained": "Trained RNN", "Naive": "Naive", "Echo": "ESN"}
PAIR_COL = ("#7570b3", "#e7298a", "#c49a00")                                      # pairs, by split step
WINDOWS = ((11, 22), (8, 10), (6, 7))                                             # t_s windows, latest first
T = 30
FPS = 20
DPI = 110
PRE_ALPHA = .3                                                                    # paths before the split


def build():
    from scipy.spatial import cKDTree
    m = np.load(os.path.join(SCAN, "scan_meta.npz"))
    cells, cell_of, rev = m["cells"], m["cell_of"], m["rev"]
    envs = [e for e in range(40) if e != 2]
    R = np.arange(10)
    X, raw = {}, {}
    for a in AG:
        raw[a] = np.load(os.path.join(SCAN, f"scan_{a}.npz"))["beliefs"]           # (40, 5000, 30, 20) float16
        b = raw[a].astype(np.float32).reshape(*raw[a].shape[:3], 2, 10)
        X[a] = (b * R).sum(-1) / b.sum(-1)                                        # (40, 5000, 30, 2)
        print(a, "positions", flush=True)
    c_pos = cells[cell_of].astype(np.float32)
    cand = {}
    for e in envs:
        XJ = X["Joint"][e]
        for ts in range(5, 23):
            dd, jj = cKDTree(XJ[:, ts - 1]).query(XJ[:, ts - 1], k=25)
            i, j = np.repeat(np.arange(5000), 25), jj.ravel()
            ok = (cell_of[i] != cell_of[j]) & (dd.ravel() < .4) & (i < j)
            i, j = i[ok], j[ok]
            d = np.linalg.norm(XJ[i] - XJ[j], axis=-1)                            # (n, T)
            sel = ((d[:, 1:ts].max(1) < .6) & (d[:, ts + 1] - d[:, ts - 1] >= 3)
                   & (np.linalg.norm(c_pos[i] - c_pos[j], axis=1) >= 4))
            for i_, j_, d_ in zip(i[sel], j[sel], d[sel]):
                cand[(e, i_, j_)] = (ts, d_[ts + 1] - d_[ts - 1])                 # the last t_s wins
    keys = sorted(cand)
    e_, i_, j_ = (np.array([k[n] for k in keys]) for n in range(3))
    ts_ = np.array([cand[k][0] for k in keys])
    jump = np.array([cand[k][1] for k in keys])
    # where the two streams end, per agent: distance of each stream's final position to its own true cell
    err = np.stack([np.stack([np.linalg.norm(X[a][e_, s, -1] - c_pos[s], axis=-1) for s in (i_, j_)], 1) for a in AG])
    # the split step's entanglement: percentile among all of the environment's streams at the same step
    ent = (rev[e_, i_, ts_] + rev[e_, j_, ts_]) / 2
    pct = np.array([np.mean(rev[e][:, t] < v) for e, t, v in zip(e_, ts_, ent)])
    shown, used_env, used_cell = [], set(), set()
    for lo, hi in WINDOWS:
        for k in np.argsort(-jump):
            if lo <= ts_[k] <= hi and e_[k] not in used_env and not ({cell_of[i_[k]], cell_of[j_[k]]} & used_cell):
                shown.append((e_[k], i_[k], j_[k], ts_[k]))
                used_env.add(e_[k]), used_cell.update({cell_of[i_[k]], cell_of[j_[k]]})
                break
    shown.sort(key=lambda s: s[3])
    se, si, sj, sts = (np.array([s[n] for s in shown]) for n in range(4))
    P = np.stack([np.stack([X[a][se, s] for s in (si, sj)], 1) for a in AG])      # (4, n, 2, T, 2)
    B = np.stack([np.stack([raw[a][se, s].astype(np.float32) for s in (si, sj)], 1) for a in AG])   # (4, n, 2, T, 20)
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez_compressed(DATA, env=e_, i=i_, j=j_, ts=ts_, jump=jump, err=err, pct=pct, shown_env=se, shown_i=si,
                        shown_j=sj, shown_ts=sts, P=P, B=B, c_i=c_pos[si], c_j=c_pos[sj], n_env=len(envs),
                        agents=np.array(AG))
    print("saved", DATA)


def _frames(split_steps):
    """step parameter tau (0 = step 1) per frame: a hold, then 0.6 s per step around each shown split (two steps before
    to three after) and 0.3 s per step elsewhere, then a hold"""
    slow = set()
    for s in split_steps:                                                          # s = t_s (the last step together)
        slow.update(range(max(s - 3, 0), min(s + 3, T - 1)))
    taus = [0.] * int(.8 * FPS)
    for s in range(T - 1):
        n = int((.6 if s in slow else .3) * FPS)
        taus += list(s + np.arange(n) / n)
    return taus + [T - 1.] * int(2.5 * FPS)


def _at(Y, tau):
    """positions Y (..., T, 2) up to tau: the steps 0..floor(tau) and the interpolated point at tau"""
    s = int(np.floor(tau))
    f = tau - s
    head = Y[..., s, :] if f == 0 else (1 - f) * Y[..., s, :] + f * Y[..., s + 1, :]
    return np.concatenate([Y[..., :s + 1, :], head[..., None, :]], -2), head


def movie():
    Z = np.load(DATA)
    P, sts = Z["P"], Z["shown_ts"]
    n = len(sts)
    col = PAIR_COL[:n]
    D = np.linalg.norm(P[:, :, 0] - P[:, :, 1], axis=-1)                          # (4, n, T)
    true_d = np.linalg.norm(Z["c_i"] - Z["c_j"], axis=-1)
    both = (Z["err"] <= 1).all(-1).mean(1)                                        # per agent, share of all pairs
    pw, gap, dh = 2.3, .42, 1.1
    W = .62 + 4 * pw + 3 * gap + .2
    H = .95 + pw + .62 + dh + 1.05
    fig = plt.figure(figsize=(W, H), facecolor="white", dpi=DPI)                    # explicit: imports reset rcParams
    tt = np.arange(1, T + 1)
    art = []
    y_plane, y_dist = H - .95 - pw, H - .95 - pw - .62 - dh
    g = np.arange(10)
    sty = ({"lw": 1.2}, {"lw": 1.9, "ls": (0, (2.4, 1.4))})                        # stream i solid, j dashed (thicker)
    for a_, a in enumerate(AG):
        x0 = .62 + a_ * (pw + gap)
        ax = fig.add_axes([x0 / W, y_plane / H, pw / W, pw / H])
        ax.scatter(*np.meshgrid(g, g), s=1.5, c="0.82", lw=0, zorder=0)
        ax.set_xlim(-.6, 9.6), ax.set_ylim(-.6, 9.6), ax.set_aspect("equal")
        ax.set_xticks([0, 3, 6, 9]), ax.set_yticks([0, 3, 6, 9])
        if a_ == 0:
            ax.set_ylabel(r"$E[r_c]$", labelpad=1)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(r"$E[r_g]$", labelpad=1)
        ax.set_title(NAME[a], fontsize=FS["label"] + .5, pad=14)
        ax.text(.5, 1.015, "both streams end at their cells in %d%% of pairs" % round(100 * both[a_]),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=FS["tick"] - 1.5, color="0.3")
        lines = []
        for p in range(n):
            c = col[p]
            ax.plot(*Z["c_i"][p], "s", ms=7, color=c, zorder=2)
            ax.plot(*Z["c_j"][p], "s", ms=7, mfc="white", mec=c, mew=1.5, zorder=2)
            seg = [(ax.plot([], [], c=c, alpha=PRE_ALPHA, zorder=3, **sty[k])[0],      # before the split
                    ax.plot([], [], c=c, zorder=3.5, **sty[k])[0]) for k in (0, 1)]    # from the split on
            hi, = ax.plot([], [], "o", ms=5, color=c, mec="white", mew=.5, zorder=5)
            hj, = ax.plot([], [], "o", ms=5, mfc="white", mec=c, mew=1.4, zorder=5)
            lines.append((seg, hi, hj))
        bx = fig.add_axes([x0 / W, y_dist / H, pw / W, dh / H])
        dl = []
        for p in range(n):
            c = col[p]
            bx.axhline(true_d[p], c=c, lw=.6, ls=":", alpha=.8)
            bx.plot([sts[p] + 1], [-.35], "^", ms=4.5, color=c, clip_on=False, zorder=4)
            dl.append(bx.plot([], [], c=c, lw=1.3)[0])
        bx.set_xlim(.5, T + .5), bx.set_ylim(0, 13.5)
        bx.set_xticks([1, 10, 20, 30]), bx.set_yticks([0, 4, 8, 12])
        bx.spines[["top", "right"]].set_visible(False)
        bx.set_xlabel("Step $t$", labelpad=1)
        if a_ == 0:
            bx.set_ylabel(r"$\|x_i - x_j\|$", labelpad=1)
            bx.text(0, 1.06, "Distance between the pair's two streams", transform=bx.transAxes, fontsize=FS["tick"],
                    color="0.25", va="bottom")
        else:
            bx.set_yticklabels([])
        cur = bx.axvline(1, c="0.6", lw=.6)
        art.append((a_, lines, dl, cur))
    clock = fig.text(4.3 / W, (H - .3) / H, "", ha="left", va="center", fontsize=FS["label"] + 2)
    fig.text(.62 / W, (H - .3) / H, "Pairs that stay together, then split", fontsize=FS["label"] + 1, va="center")
    hand = [Line2D([], [], c="0.3", marker="o", ms=5, label="stream $i$ (path, posterior mean)", **sty[0]),
            Line2D([], [], c="0.3", marker="o", ms=5, mfc="white", mew=1.3, label="stream $j$", **sty[1]),
            Line2D([], [], c="0.3", marker="s", ms=6.5, lw=0, label="true cell of $i$"),
            Line2D([], [], c="0.3", marker="s", ms=6.5, mfc="white", mew=1.3, lw=0, label="true cell of $j$"),
            Line2D([], [], c="0.3", marker="^", ms=5, lw=0, label="split step $t_s + 1$"),
            Line2D([], [], c="0.3", lw=1.2, alpha=PRE_ALPHA, label="before the split (faint)")]
    fig.legend(handles=hand, loc="upper right", bbox_to_anchor=(1 - .2 / W, 1 - .04 / H), ncol=3, frameon=False,
               fontsize=FS["tick"] - .5, handlelength=1.8, columnspacing=1.2)
    fig.text(.62 / W, .6 / H, "Same streams in every panel; colours: pairs (environments %s), selected in the "
             "Optimal: different cells (at least 4 apart) within 0.6 of each other from step 2 through $t_s$,\nat least "
             "3 further apart two steps later (%d such pairs in %d environments; the split step's entanglement at the "
             "%dth percentile of same-step streams, median)." % (
                 ", ".join(str(e) for e in Z["shown_env"]), len(Z["ts"]), int(Z["n_env"]), round(100 * np.median(Z["pct"]))),
             fontsize=FS["tick"] - 1.5, color="0.35", va="top", linespacing=1.3)

    def update(tau):
        s = int(np.floor(tau))
        f = tau - s
        for a_, lines, dl, cur in art:
            for p, (seg, hi, hj) in enumerate(lines):
                path, head = _at(P[a_, p], tau)                                     # (2, s + 2, 2): steps 0..s, tau
                idx = np.r_[np.arange(s + 1), tau]
                k0 = sts[p] - 1                                                      # index of step t_s
                for k, (pre, post) in enumerate(seg):
                    m1, m2 = idx <= k0, idx >= k0
                    pre.set_data(path[k, m1, 0], path[k, m1, 1])
                    post.set_data(path[k, m2, 0], path[k, m2, 1])
                hi.set_data([head[0, 0]], [head[0, 1]]), hj.set_data([head[1, 0]], [head[1, 1]])
                dv = D[a_, p, :s + 1].tolist() + ([(1 - f) * D[a_, p, s] + f * D[a_, p, s + 1]] if f else [])
                dl[p].set_data(np.r_[tt[:s + 1], [s + 1 + f] if f else []], dv)
            cur.set_xdata([tau + 1])
        clock.set_text("step $t$ = %d" % (s + 1))

    def grab():
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    taus = _frames(sts)
    protos = []
    for tau in (0., 8., 22., T - 1.):
        update(tau)
        protos.append(np.asarray(grab()))
    pal = Image.fromarray(np.concatenate(protos, 0)).quantize(colors=255, method=Image.Quantize.MEDIANCUT,
                                                              dither=Image.Dither.NONE)
    frames = []
    for tau in taus:
        update(tau)
        frames.append(grab().quantize(palette=pal, dither=Image.Dither.NONE))
    plt.close(fig)
    path = fig_path("fig1c_movie_split_pairs.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0)
    print(f"saved {path}: {len(frames)} frames, {len(frames) / FPS:.1f} s, {frames[0].size[0]} x {frames[0].size[1]} px, "
          f"{os.path.getsize(path) / 1e6:.1f} MB")
    return path


if __name__ == "__main__":
    if "--rebuild" in sys.argv[1:] or not os.path.exists(DATA) or "B" not in np.load(DATA).files:
        build()
    with plt.rc_context({"figure.dpi": DPI, "savefig.dpi": DPI}):
        movie()
    Z = np.load(DATA)
    print("shown: environments", Z["shown_env"], "t_s", Z["shown_ts"])
    for a_, a in enumerate(AG):
        print("%-8s both streams within 1 of their cells: %.2f (median final error %.2f)" % (
            a, (Z["err"][a_] <= 1).all(-1).mean(), np.median(Z["err"][a_])))
