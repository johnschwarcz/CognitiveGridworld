"""fig_learn_holdout_movie.gif: the first two columns of fig_learn_holdout (figure_bands) as an animation through
training (John 11-11: "do you think you could turn the first 2 columns of holdout into an animation through training
time?").

Rows: the network that learned only from updates with FR below the low edge (red) / above the high edge (green);
columns: the equal FR chunks (.03 | 3) and (.1 | 1), each network learning the same count at every step of every batch
(train_learn_masks band modes; fig_learn_holdout's docstring). Each frame: accuracy at T by the probe state's FR_T
(half-decade bins from 1e-4 to 100 nats, plus the two open ends; each at its geometric-mean FR_T) at one checkpoint,
smoothed through training (a Gaussian of SD SMOOTH in log training episodes), solid on the learned FR range and dashed outside it
(generalization); faint: the earlier frames; grey in the (.1 | 1) column: the random control (the same count, random
updates) at the same checkpoint. Frames: 37 checkpoints, log-spaced from initialization to 9,600 training episodes
(ANIM); the last frame is held. Data: fig_learn_holdout's replays (through_training/fr_holdout/<run>_<raw>.npz; 10,000
probe episodes x 2 variables = 20,000 states).
usage:  python fig_learn_holdout_movie.py"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter                      # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import fig_learn_holdout as h                                                     # noqa: E402

ANIM = (0, 25, 50, 75, 100, 125, 150, 175, 225, 250, 275, 325, 375, 450, 500, 600, 700, 800, 900, 1000, 1200, 1400,
        1600, 1900, 2100, 2500, 2900, 3300, 3800, 4400, 5100, 5900, 6800, 7800, 9000, 10400, 12000)   # raw episodes
COLS = ((.03, 3), (.1, 1))
ROWS = (("lo", "learned low FR", "#d62728"), ("hi", "learned high FR", "#2ca02c"))
FS = dict(tick=9, label=10, title=11)
HOLD, MS, NT = 30, 90, 3                                                          # last-frame repeats, ms per frame, frames
                                                                                  # per checkpoint interval (tweened)
EDGES = np.r_[0, 10 ** np.arange(-4, 2.01, .5), np.inf]                           # half decades, 1e-4 to 100 nats + the
                                                                                  # open ends (>= 74 states per bin)
SMOOTH = .15                                                                      # Gaussian SD in log10 training episodes
RND = dict(color="0.4", lw=1.3, ls=(0, (1, 1.5)))                                 # the random control: dotted (John 11-11)
GEN = (0, (3, 1.6))                                                               # generalization: dashed


def profiles(edges=None, runs=None):
    """per run: (frames, bins) accuracy at T by FR_T bin (EDGES) at each ANIM checkpoint, then smoothed through training
    with a Gaussian of SD SMOOTH decades of training episodes (John 11-11: "smoothen a bit through episodes so the
    curves dont wiggle so much"); x = each bin's geometric-mean FR_T (John 11-11: "a larger range of FR")"""
    Z = h.load()
    r, fr = Z["r"], Z["fr_T"]
    edges = EDGES if edges is None else edges
    b = np.searchsorted(edges, fr, side="right") - 1                              # -1 / nb: outside the edges (dropped)
    nb = len(edges) - 1
    b = np.where((b >= 0) & (b < nb), b, nb)
    cnt = np.bincount(b.ravel(), minlength=nb + 1)[:nb]
    keep = cnt > 0                                                                # (no state above 100 nats: max 95)
    x = np.exp(np.bincount(b.ravel(), np.log(fr.ravel()), minlength=nb + 1)[:nb][keep] / cnt[keep])
    lt = np.log10(np.maximum([h.train_x(raw) for raw in ANIM], 10.))
    K = np.exp(-.5 * ((lt[:, None] - lt[None]) / SMOOTH) ** 2)
    K /= K.sum(1, keepdims=True)
    out = {}
    runs = runs or [h.band_name(side, lo, hi) for lo, hi in COLS for side, _, _ in ROWS] + [h.BRAND]
    for run in runs:
        P = np.stack([np.bincount(b.ravel(), (np.load(h.cache(run, raw))["ans"][:, -1] == r).ravel(),
                                  minlength=nb + 1)[:nb][keep] / cnt[keep] for raw in ANIM])
        out[run] = K @ P
    print("states per bin", cnt[keep].tolist(), "; x", np.round(x, 6).tolist())
    return x, out


def split(x, y, th, side):
    """the curve's learned part (solid) and generalization part (dashed), joined at the threshold"""
    yb = float(np.interp(np.log10(th), np.log10(x), y))
    inside = x < th if side == "lo" else x > th
    xi, yi, xo, yo = x[inside], y[inside], x[~inside], y[~inside]
    if side == "lo":
        return (np.r_[xi, th], np.r_[yi, yb]), (np.r_[th, xo], np.r_[yb, yo])
    return (np.r_[th, xi], np.r_[yb, yi]), (np.r_[xo, th], np.r_[yo, yb])


def pale(c, keep=.28):
    """c mixed with white, opaque (overlapping trails do not darken; John 11-11 saw the stacked early curves near chance
    as a stray "past curve")"""
    from matplotlib.colors import to_rgb
    return tuple(1 - keep * (1 - v) for v in to_rgb(c))


def panel(ax, side, th, c, run, with_rnd):
    """the artists of one panel: pale trails (solid / dashed, one per checkpoint), the current curve, the random
    control (dotted, split at the panel's threshold too)"""
    pc = pale(c)
    return dict(side=side, th=th, run=run,
                trail=[(ax.plot([], [], color=pc, lw=.6, zorder=2)[0], ax.plot([], [], color=pc, lw=.6, ls=GEN, zorder=2)[0])
                       for _ in ANIM],
                inn=ax.plot([], [], color=c, lw=1.8, zorder=5, solid_capstyle="round")[0],
                out=ax.plot([], [], color=c, lw=1.8, ls=GEN, zorder=5)[0],
                rnd=ax.plot([], [], zorder=4, **RND)[0] if with_rnd else None)


def animate(fig, W, H, panels, x, P, path):
    """tweened frames (NT per checkpoint interval, linear between the smoothed checkpoints), a clock and a log-time
    progress bar; trails appear at the checkpoints"""
    from matplotlib.patches import Rectangle
    K = len(ANIM)
    tx = np.array([h.train_x(raw) for raw in ANIM], float)
    frames = [(k, s / NT) for k in range(K - 1) for s in range(NT)] + [(K - 1, 0.)] * (1 + HOLD)
    clock = fig.text(.5, 1 - .16 / H, "", fontsize=FS["title"], ha="center", va="center")
    bar_ax = fig.add_axes([(W / 2 - 1.6) / W, 1 - .40 / H, 3.2 / W, .05 / H])
    xs = np.maximum(tx, 10.)
    bar_ax.set_xscale("log")
    bar_ax.set_xlim(xs[0], xs[-1])
    bar_ax.set_ylim(0, 1)
    bar_ax.set_axis_off()
    bar_ax.add_patch(Rectangle((xs[0], 0), xs[-1] - xs[0], 1, color="0.88", lw=0))
    bar = bar_ax.add_patch(Rectangle((xs[0], 0), 0, 1, color="0.35", lw=0))

    def at(run, k, a):
        return P[run][k] if a == 0 else (1 - a) * P[run][k] + a * P[run][k + 1]

    def draw(i):
        k, a = frames[i]
        for D in panels:
            (xi, yi), (xo, yo) = split(x, at(D["run"], k, a), D["th"], D["side"])
            D["inn"].set_data(xi, yi)
            D["out"].set_data(xo, yo)
            for j, (ti, to) in enumerate(D["trail"]):
                if j <= k:
                    (xi2, yi2), (xo2, yo2) = split(x, P[D["run"]][j], D["th"], D["side"])
                    ti.set_data(xi2, yi2)
                    to.set_data(xo2, yo2)
                else:
                    ti.set_data([], [])
                    to.set_data([], [])
            if D["rnd"] is not None:
                D["rnd"].set_data(x, at(h.BRAND, k, a))
        t = tx[k] if a == 0 else np.exp((1 - a) * np.log(max(tx[k], 1)) + a * np.log(tx[k + 1]))
        clock.set_text(f"training episodes: {int(round(t, -1 if t > 100 else 0)):,}")
        bar.set_width((xs[k] if a == 0 else np.exp((1 - a) * np.log(xs[k]) + a * np.log(xs[k + 1]))) - xs[0])
        return []
    anim = FuncAnimation(fig, draw, frames=len(frames), interval=MS, blit=False)
    anim.save(path, writer=PillowWriter(fps=1000 / MS), dpi=120)
    plt.close(fig)
    print("saved", path, len(frames), "frames")


def legend(fig):
    fig.legend([Line2D([], [], color="0.3", lw=1.8), Line2D([], [], color="0.3", lw=1.8, ls=GEN), Line2D([], [], **RND)],
               ["on the learned range", "generalization", "random updates, same count"], loc="lower center", ncol=3,
               bbox_to_anchor=(.5, .005), fontsize=FS["tick"], frameon=False, handlelength=2.2, columnspacing=1.6)


def axes_style(ax, i_bottom, j_left, xlim, ticks, labels, minor):
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels if i_bottom else [])
    ax.set_xticks(minor, minor=True)
    ax.set_xticklabels([], minor=True)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels(["0", "0.5", "1"] if j_left else [])
    ax.tick_params(labelsize=FS["tick"], length=3, pad=2)
    ax.spines[["top", "right"]].set_visible(False)


def movie(path=None):
    """both chunk pairs, 2 x 2 (fig_learn_holdout_movie.gif)"""
    apply_fig_style()
    plt.rcParams.update({"mathtext.fontset": "cm"})
    x, P = profiles()
    W, H = 7.4, 5.3
    fig = plt.figure(figsize=(W, H))
    left, pw, gap, top, ph, rgap = .95, 2.95, .25, .95, 1.55, .22
    panels = []
    for j, (lo, hi) in enumerate(COLS):
        st = h.sel_stats(h.band_name("lo", lo, hi))
        xc = left + j * (pw + gap) + pw / 2
        fig.text(xc / W, 1 - (top - .32) / H, rf"FR $<$ {h.fmt(lo)}  or  $>$ {h.fmt(hi)} nats", fontsize=FS["title"],
                 ha="center", va="bottom")
        fig.text(xc / W, 1 - (top - .10) / H, f"{100 * (1 - st['blocked']):.1f}% of updates learned", fontsize=FS["tick"],
                 ha="center", va="bottom", color="0.35")
        for i, (side, name, c) in enumerate(ROWS):
            t_ = top + i * (ph + rgap)
            ax = fig.add_axes([(left + j * (pw + gap)) / W, 1 - (t_ + ph) / H, pw / W, ph / H])
            th = lo if side == "lo" else hi
            ax.axvspan(*((1e-6, lo) if side == "lo" else (hi, 1e3)), color=c, alpha=.10, lw=0, zorder=0)
            ax.axvline(th, color=c, lw=.8, zorder=1)
            axes_style(ax, i == 1, j == 0, (10 ** -5.3, 10 ** 2.4), [1e-4, 1e-2, 1, 100],
                       [r"$10^{-4}$", "0.01", "1", "100"], [1e-3, 1e-1, 10])
            if j == 0:
                ax.set_ylabel("accuracy", fontsize=FS["label"], labelpad=3)
                fig.text(.16 / W, 1 - (t_ + ph / 2) / H, name, rotation=90, ha="center", va="center",
                         fontsize=FS["label"], color=c)
            panels.append(panel(ax, side, th, c, h.band_name(side, lo, hi), (lo, hi) == (.1, 1)))
    fig.text((left + pw + gap / 2) / W, 1 - (top + 2 * ph + rgap + .30) / H, r"FR$_T$ of the probe state (nats)",
             fontsize=FS["label"], ha="center", va="top")
    legend(fig)
    animate(fig, W, H, panels, x, P, path or fig_path("fig_learn_holdout_movie.gif"))


MID_EDGES = 10 ** np.arange(-3.5, 2.01, .5)                                       # half decades 10^-3.5 to 100: edges at
                                                                                  # .1, .32 and 1


def movie_single(chunk=(.1, 1), path=None):
    """John 11-11: "a version with just the 2nd panel (as 2 columns) and have the x-range and binning such that halfway
    through the x axes is in between the training chunks": the (.1 | 1) networks side by side (learned low FR | learned
    high FR), x from 10^-3.5 to 10^2.5, whose middle (log) is sqrt(.1 x 1) = .32 nats; half-decade bins with edges at
    .1, .32 and 1; the states below 10^-3.5 nats (174 of 20,000) are left out; dotted: the random control in both
    panels (John 11-11: "chance curve be a dotted line"; it learns at every FR, so it is not split); the past curves
    (trails) are split into solid / dashed like the current one"""
    apply_fig_style()
    plt.rcParams.update({"mathtext.fontset": "cm"})
    lo, hi = chunk
    runs = [h.band_name("lo", lo, hi), h.band_name("hi", lo, hi), h.BRAND]
    x, P = profiles(MID_EDGES, runs)
    left, pw, gap, top, ph = .75, 3.05, .35, 1.42, 1.95
    W, H = 7.4, top + ph + .78
    fig = plt.figure(figsize=(W, H))
    st = h.sel_stats(runs[0])
    fig.text(.5, 1 - (top - .62) / H, rf"FR $<$ {h.fmt(lo)}  or  $>$ {h.fmt(hi)} nats", fontsize=FS["title"],
             ha="center", va="bottom")
    fig.text(.5, 1 - (top - .43) / H, f"each network learns {100 * (1 - st['blocked']):.1f}% of the updates, the same "
             "number at every step", fontsize=FS["tick"], ha="center", va="bottom", color="0.35")
    panels = []
    for j, (side, name, c) in enumerate(ROWS):
        ax = fig.add_axes([(left + j * (pw + gap)) / W, 1 - (top + ph) / H, pw / W, ph / H])
        th = lo if side == "lo" else hi
        ax.axvspan(*((1e-6, lo) if side == "lo" else (hi, 1e4)), color=c, alpha=.10, lw=0, zorder=0)
        ax.axvline(th, color=c, lw=.8, zorder=1)
        ax.axvline(np.sqrt(lo * hi), color="0.75", lw=.6, ls=(0, (1, 2)), zorder=0)
        axes_style(ax, True, j == 0, (10 ** -3.5, 10 ** 2.5), [1e-3, 1e-1, 10], [r"$10^{-3}$", "0.1", "10"],
                   [1e-2, 1, 100])
        ax.set_title(name, fontsize=FS["label"], color=c, pad=4)
        if j == 0:
            ax.set_ylabel("accuracy", fontsize=FS["label"], labelpad=3)
        panels.append(panel(ax, side, th, c, runs[j], True))
    fig.text(.5, 1 - (top + ph + .30) / H, r"FR$_T$ of the probe state (nats)", fontsize=FS["label"], ha="center",
             va="top")
    legend(fig)
    animate(fig, W, H, panels, x, P, path or fig_path("fig_learn_holdout_movie_mid.gif"))


ALL = (0,) + tuple(range(25, 501, 25)) + tuple(range(600, 12001, 100)) \
    + tuple(range(12400, 37501, 400)) + (37500,)                                  # every stored checkpoint to 12,000
                                                                                  # (136), then every 4th of the
                                                                                  # continuation to raw 37,500 (30,000
                                                                                  # training episodes; John 11-11)
D_SMOOTH, D_FRAMES, D_MS, D_START, D_HOLD, D_TRAIL, D_DPI = .08, 220, 70, 0, 36, 11, 180
D_SWITCH, D_FRAMES_LOG, D_FRAMES_LIN, D_SMOOTH_LIN = 2000, 90, 170, 600   # frames: log-spaced to 2,000 episodes, then
                                                                          # evenly spaced in episodes (John 11-11: the
                                                                          # log spacing gave the last 8,000 episodes 14
                                                                          # frames); late smoothing <= ~600 episodes
FR_SD = .20                                                                       # Gaussian SD in log10 FR (kernel curve)
LX = np.round(np.arange(-3.5, 1.901, .05), 3)                                     # its points: every .05 decade, .1 and 1
                                                                                  # included (11-11: "start the gifs at
                                                                                  # training episode 20 ... the curve
                                                                                  # through FR to be a bit smoother")


def profiles_kernel(runs, raws, lx=LX, sd=FR_SD):
    """per run: (checkpoints, points) accuracy at T as a Gaussian-weighted mean over the probe states by log10 FR_T
    (SD sd decades) at each point of lx, in place of bins; every state counts (also those below 10^-3.5)"""
    Z = h.load()
    r, lf = Z["r"].ravel(), np.log10(np.maximum(Z["fr_T"].ravel(), 1e-12))
    K = np.exp(-.5 * ((lx[:, None] - lf[None]) / sd) ** 2)
    K /= K.sum(1, keepdims=True)
    out = {run: np.stack([K @ (np.load(h.cache(run, raw))["ans"][:, -1].ravel() == r).astype(float) for raw in raws])
           for run in runs}
    return 10 ** lx, out


def profiles_raw(edges, runs, raws):
    """per run: (checkpoints, bins) accuracy at T by FR_T bin, unsmoothed; x = each bin's geometric-mean FR_T"""
    Z = h.load()
    r, fr = Z["r"], Z["fr_T"]
    b = np.searchsorted(edges, fr, side="right") - 1
    nb = len(edges) - 1
    b = np.where((b >= 0) & (b < nb), b, nb)
    cnt = np.bincount(b.ravel(), minlength=nb + 1)[:nb]
    keep = cnt > 0
    x = np.exp(np.bincount(b.ravel(), np.log(fr.ravel()), minlength=nb + 1)[:nb][keep] / cnt[keep])
    out = {run: np.stack([np.bincount(b.ravel(), (np.load(h.cache(run, raw))["ans"][:, -1] == r).ravel(),
                                      minlength=nb + 1)[:nb][keep] / cnt[keep] for raw in raws]) for run in runs}
    return x, out


def dense_frames(runs):
    """movie_dense's frames for these runs: x (FR points), per run the smoothed accuracy curve at every frame and whether
    the run is past its last checkpoint (held), the frame times (training episodes), each run's log10 checkpoint times
    and the longest run's last time"""
    have = {run: [raw for raw in ALL if os.path.exists(h.cache(run, raw))] for run in runs}   # each run's own data
    P, LT = {}, {}
    for run in runs:
        x, Pr = profiles_kernel([run], have[run])
        P[run] = Pr[run]
        LT[run] = np.log10(np.maximum([h.train_x(raw) for raw in have[run]], 1.))
    tmax = max(10 ** LT[run][-1] for run in runs)
    G = np.r_[np.linspace(np.log10(20), np.log10(D_SWITCH), D_FRAMES_LOG),        # frame times (log10 episodes):
              np.log10(np.linspace(D_SWITCH, tmax, D_FRAMES_LIN + 1)[1:])]        # log, then even in episodes
    SD = np.minimum(D_SMOOTH, np.log10(1 + D_SMOOTH_LIN / 10 ** G))[:, None]      # kernel SD per frame (decades)
    F, HELD = {}, {}
    for run in runs:
        Wk = np.exp(-.5 * ((G[:, None] - LT[run][None, 1:]) / SD) ** 2)           # trained checkpoints only
        Wk /= Wk.sum(1, keepdims=True)
        Fr = Wk @ P[run][1:]
        held = G > LT[run][-1] + 1e-9                                            # past the run's last checkpoint:
        if held.any():                                                            # hold its last smoothed value
            Fr[held] = Fr[np.flatnonzero(~held)[-1]]
        F[run] = np.concatenate([np.repeat(P[run][:1], D_START, 0), Fr])
        HELD[run] = np.r_[np.zeros(D_START, bool), held]
    return x, F, HELD, np.r_[np.zeros(D_START), 10 ** G], LT, tmax


FILL_A, BG_A = .22, .07                              # the overlay's fill between the curves and its range shading
O_MS = 35                                            # the overlay's ms per frame (John 11-11: "make it run twice as fast")


class DurationWriter(PillowWriter):
    """PillowWriter with one duration per frame (ms): GIF delays are whole centiseconds, so 35 ms frames are written as
    alternating 30 / 40 ms (exactly twice the speed of 70 ms frames)"""
    def __init__(self, durations, **kw):
        super().__init__(**kw)
        self.durations = list(durations)

    def finish(self):
        self._frames[0].save(self.outfile, save_all=True, append_images=self._frames[1:], duration=self.durations,
                             loop=0)
T0, O_SMOOTH, O_CAP, O_START = 20., .25, 2500., 10   # the overlay's time axis u = log10(1 + t / T0) (from t = 0), its
                                                     # smoothing SD in u (3x movie_dense's .08 decades), capped at 2,500
                                                     # episodes late in training (4x its 600), and the untrained
                                                     # network held for O_START frames


Q_BINS = 20                                                                       # equal-count bins: 5% of states each


def profiles_quantile(runs, raws, n=Q_BINS):
    """per run: (checkpoints, n) accuracy at T in n bins of equal numbers of probe states by FR_T (John 11-12: "go back
    to showing it as continuous values, but with bins of equal % of the data"); x = each bin's geometric-mean FR_T"""
    Z = h.load()
    r, fr = Z["r"].ravel(), Z["fr_T"].ravel()
    b = np.minimum(np.argsort(np.argsort(fr, kind="stable"), kind="stable") * n // fr.size, n - 1)   # rank -> bin
    cnt = np.bincount(b, minlength=n)
    x = np.exp(np.bincount(b, np.log(np.maximum(fr, 1e-12)), minlength=n) / cnt)
    out = {run: np.stack([np.bincount(b, (np.load(h.cache(run, raw))["ans"][:, -1].ravel() == r), minlength=n) / cnt
                          for raw in raws]) for run in runs}
    return x, out


def smooth_frames(runs, sd=O_SMOOTH, cap=O_CAP, prof=None):
    """frames from the untrained network (t = 0) on (John 11-11: "start from 0 and use a much greater smoothing through
    episodes"): frame times evenly spaced in u = log10(1 + t / T0) up to D_SWITCH episodes (D_FRAMES_LOG; ~linear in t
    below T0, ~log above), then evenly in episodes (D_FRAMES_LIN); each frame the Gaussian-weighted mean, in u, of every
    checkpoint including the untrained one, SD sd in u (at t >> T0 the same as decades of training episodes), capped at
    cap episodes late in training, and ramped up from 0 over the first T0 episodes so that the first frame is exactly
    the untrained network. The kernel is one-sided at the last frames, which therefore average roughly the last 2,000
    episodes (the curves have levelled by then: the learned-low network's FR > 1 accuracy rises .003 per 1,000)."""
    have = {run: [raw for raw in ALL if os.path.exists(h.cache(run, raw))] for run in runs}
    P, U, last = {}, {}, {}
    for run in runs:
        x, Pr = (prof or profiles_kernel)([run], have[run])            # prof: e.g. binned
        P[run] = Pr[run]
        t = np.array([h.train_x(raw) for raw in have[run]], float)
        U[run], last[run] = np.log10(1 + t / T0), t[-1]
    tmax = max(last.values())
    T = np.r_[T0 * (10 ** np.linspace(0, np.log10(1 + D_SWITCH / T0), D_FRAMES_LOG) - 1),
              np.linspace(D_SWITCH, tmax, D_FRAMES_LIN + 1)[1:]]
    G = np.log10(1 + T / T0)
    SD = np.minimum(sd, np.log10(1 + cap / (T + T0)))                             # the cap (episodes) as an SD in u
    SD = np.maximum(SD * np.minimum(1, G / np.log10(2)), 1e-3)                    # ramp: ~0 at t = 0, full from T0
    F, HELD = {}, {}
    for run in runs:
        Wk = np.exp(-.5 * ((G[:, None] - U[run][None]) / SD[:, None]) ** 2)
        Wk /= Wk.sum(1, keepdims=True)
        Fr = Wk @ P[run]
        held = T > last[run] + 1e-6                                               # past the run's last checkpoint:
        if held.any():                                                            # hold its last smoothed value
            Fr[held] = Fr[np.flatnonzero(~held)[-1]]
        F[run], HELD[run] = Fr, held
    return x, F, HELD, T, last


def movie_overlay(chunk=(.1, 1), path=None, xscale="log", quantile=False):
    """John 11-11 (night): "can you please send a version with a single panel, red and green together, and no chance
    curve": movie_dense's two networks in one panel, each solid on its own training range (shaded in its colour) and
    dashed outside it, without the random control; trails and clock as movie_dense. Then "start from 0 and use a much
    greater smoothing through episodes": frames and smoothing from smooth_frames (the untrained network first, held for
    O_START frames). Then "remove the past lines but add a slight fill between the curves, colored by whichever curve is
    higher at that point. Also please xlim so that there is no empty space on the x-axis": no trails, the gap between
    the curves filled (alpha FILL_A) in the colour of the higher one, x from the first to the last curve point (10^-3.5
    to 10^1.9 nats) -> figure_bin/supp gifs/fig_learn_holdout_movie_overlay.gif. John 11-12: "constrain the xlim so that
    there is an equal amount of red as there is green" (log axis 10^-2.9 to 10^1.9), then "would you be able to show a
    version with a linear x-axis scaling. Or would the red and green look unbalanced?": xscale "linear" (FR in nats, 0 to
    79: the learned-low range FR < .1 is .13% of the width while 65% of the probe states lie below 1 nat;
    _overlay_linear.gif) and "pct" (x = the percentile of the probe states by FR_T, linear in the number of states: FR <
    .1 holds 24.1% of the 20,000 states, .1-1 41.4%, > 1 34.5%; _overlay_pct.gif)."""
    apply_fig_style()
    plt.rcParams.update({"mathtext.fontset": "cm"})
    from matplotlib.patches import Rectangle
    lo, hi = chunk
    runs = [h.band_name("lo", lo, hi), h.band_name("hi", lo, hi)]
    x, F, HELD, T_, last = smooth_frames(runs, prof=profiles_quantile if quantile else None)
    nF = len(T_)
    if quantile:
        print("bin x (geometric-mean FR_T):", np.round(x, 4).tolist())
    if xscale == "pct":                                                           # percent of the probe states below
        fr_all = np.sort(h.load()["fr_T"].ravel())
        xs, lo_x, hi_x = [np.searchsorted(fr_all, v) / fr_all.size * 100 for v in (x, lo, hi)]
    else:
        xs, lo_x, hi_x = x, lo, hi
    left, pw, top, ph, bot, right = .62, 4.4, .22, 2.75, 1.02, .15
    W, H = left + pw + right, top + ph + bot
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, 1 - (top + ph) / H, pw / W, ph / H])
    # x: to the last curve point on the right, and as many decades below the low edge as above the high one (John 11-12:
    # "constrain the xlim so that there is an equal amount of red as there is green"): 10^-2.9 to 10^1.9 nats
    if xscale == "log":
        half = min(np.log10(np.sqrt(lo * hi) / x[0]), np.log10(x[-1] / np.sqrt(lo * hi)))   # equal red and green,
        xl = np.sqrt(lo * hi) * 10. ** np.array([-half, half])                     # within the data on both sides
        tk = [v for v in (1e-3, 1e-2, 1e-1, 1, 10, 100) if xl[0] <= v <= xl[1]]
        axes_style(ax, True, True, tuple(xl), tk, [f"{v:g}" for v in tk], [])
    else:
        axes_style(ax, True, True, (1, 2), [1], [""], [])
        ax.set_xscale("linear")
        ax.minorticks_off()
        if xscale == "linear":
            ax.set_xlim(0, x[-1])
            ax.set_xticks([0, 20, 40, 60])
            ax.set_xticklabels(["0", "20", "40", "60"])
        else:
            ax.set_xlim(xs[0], xs[-1])
            ax.set_xticks([25, 50, 75])
            ax.set_xticklabels(["25%", "50%", "75%"])
    ax.set_xlabel("Factorization Regret" + (" (percentile of the probe states)" if xscale == "pct" else
                                            " (nats)" if xscale == "linear" else ""), fontsize=FS["label"], labelpad=2)
    ax.set_ylabel("accuracy", fontsize=FS["label"], labelpad=3)
    names = {"lo": rf"learned FR $<$ {h.fmt(lo)}", "hi": rf"learned FR $>$ {h.fmt(hi)}"}
    panels = []
    for j, (side, _, c) in enumerate(ROWS):
        th = lo_x if side == "lo" else hi_x
        far = (1e-6, 1e4) if xscale == "log" else (-1e6, 1e6)
        ax.axvspan(*((far[0], lo_x) if side == "lo" else (hi_x, far[1])), color=c, alpha=BG_A, lw=0, zorder=0)
        ax.axvline(th, color=c, lw=.8, zorder=1)
        panels.append(dict(side=side, th=th, run=runs[j], last=int(round(last[runs[j]])),
                           inn=ax.plot([], [], color=c, lw=2.0, zorder=5, solid_capstyle="round")[0],
                           out=ax.plot([], [], color=c, lw=2.0, ls=GEN, zorder=5)[0]))
    fills = []
    fig.legend([Line2D([], [], color=ROWS[0][2], lw=2.0), Line2D([], [], color="0.3", lw=2.0),
                Line2D([], [], color=ROWS[1][2], lw=2.0), Line2D([], [], color="0.3", lw=2.0, ls=GEN)],
               [names["lo"], "Training set", names["hi"], "Evaluation set"], loc="lower center", ncol=2,
               bbox_to_anchor=(.5, .01), fontsize=FS["label"], frameon=False, handlelength=2.2, columnspacing=2.4)
    clock = ax.text(.975, .975, "", transform=ax.transAxes, ha="right", va="top", fontsize=FS["label"])
    bar_ax = ax.inset_axes([.775, .885, .20, .026])                               # under the clock, in the green range
    bar_ax.set_xlim(0, 1)
    bar_ax.set_ylim(0, 1)
    bar_ax.set_axis_off()
    bar_ax.add_patch(Rectangle((0, 0), 1, 1, color="0.88", lw=0))
    bar = bar_ax.add_patch(Rectangle((0, 0), 0, 1, color="0.35", lw=0))
    note = ax.text(.975, .85, "", transform=ax.transAxes, ha="right", va="top", fontsize=FS["tick"])
    frames = [0] * O_START + list(range(nF)) + [nF - 1] * D_HOLD

    def draw(i):
        f = frames[i]
        held = [D for D in panels if HELD[D["run"]][f]]
        note.set_text("; ".join(f"held at {D['last']:,}" for D in held))
        note.set_color(held[0]["inn"].get_color() if held else "k")
        for D in panels:
            (xi, yi), (xo, yo) = split(xs, F[D["run"]][f], D["th"], D["side"])
            D["inn"].set_data(xi, yi)
            D["out"].set_data(xo, yo)
        for p in fills:                                                           # the gap, in the higher curve's colour
            p.remove()
        fills.clear()
        y_lo, y_hi = F[runs[0]][f], F[runs[1]][f]
        for a_, b_, c in ((y_lo, y_hi, ROWS[0][2]), (y_hi, y_lo, ROWS[1][2])):
            fills.append(ax.fill_between(xs, a_, b_, where=a_ >= b_, interpolate=True, color=c, alpha=FILL_A, lw=0,
                                         zorder=3))
        t = T_[f]
        n = int(round(t, -2 if t >= 1000 else (-1 if t >= 100 else 0)))
        clock.set_text(f"{n:,} episode" + ("" if n == 1 else "s"))
        bar.set_width(f / max(nF - 1, 1))
        return []
    anim = FuncAnimation(fig, draw, frames=len(frames), interval=O_MS, blit=False)
    path = path or os.path.join(os.path.dirname(fig_path("x.png")), "supp gifs", "fig_learn_holdout_movie_overlay"
                                + ("" if xscale == "log" else "_" + xscale) + (f"_q{Q_BINS}" if quantile else "") + ".gif")
    durations = [O_MS - 5 if k % 2 == 0 else O_MS + 5 for k in range(len(frames))]   # 30 / 40 ms: mean 35
    anim.save(path, writer=DurationWriter(durations, fps=1000 / O_MS), dpi=D_DPI)
    plt.close(fig)
    print("saved", path, len(frames), "frames")


EDGES3 = np.array([0, .1, 1, np.inf])                                             # the three bins: below, between, above


def movie_bins3(chunk=(.1, 1), path=None):
    """John 11-12: "Can we see a version with 3 bins: low, mid, high?": movie_overlay with accuracy at T averaged over the
    probe states in three FR_T bins (below .1: 24.1% of the 20,000 states; .1-1: 41.4%; above 1: 34.5%; profiles_raw)
    instead of the kernel curve, drawn as steps over three equal-width columns (so red and green are balanced): solid
    over the network's training bin, dashed over the others; each bin's gap filled in the higher network's colour.
    Frames, smoothing through episodes and speed as movie_overlay -> figure_bin/supp gifs/fig_learn_holdout_movie_bins3.gif"""
    apply_fig_style()
    plt.rcParams.update({"mathtext.fontset": "cm"})
    from matplotlib.patches import Rectangle
    lo, hi = chunk
    edges = np.array([0, lo, hi, np.inf])
    runs = [h.band_name("lo", lo, hi), h.band_name("hi", lo, hi)]
    _, F, HELD, T_, last = smooth_frames(runs, prof=lambda r_, raws: profiles_raw(edges, r_, raws))
    nF = len(T_)
    left, pw, top, ph, bot, right = .62, 4.4, .22, 2.75, 1.02, .15
    W, H = left + pw + right, top + ph + bot
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, 1 - (top + ph) / H, pw / W, ph / H])
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([rf"FR $<$ {h.fmt(lo)}", rf"{h.fmt(lo)} $<$ FR $<$ {h.fmt(hi)}", rf"FR $>$ {h.fmt(hi)}"])
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.tick_params(labelsize=FS["label"], length=3, pad=3)
    ax.tick_params(axis="y", labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Factorization Regret", fontsize=FS["label"], labelpad=3)
    ax.set_ylabel("accuracy", fontsize=FS["label"], labelpad=3)
    names = {"lo": rf"learned FR $<$ {h.fmt(lo)}", "hi": rf"learned FR $>$ {h.fmt(hi)}"}
    for side, _, c in ROWS:
        ax.axvspan(*((-1, .5) if side == "lo" else (1.5, 3)), color=c, alpha=BG_A, lw=0, zorder=0)
        ax.axvline(.5 if side == "lo" else 1.5, color=c, lw=.8, zorder=1)
    L = {side: (ax.plot([], [], color=c, lw=2.0, zorder=5, solid_capstyle="butt")[0],
                ax.plot([], [], color=c, lw=2.0, ls=GEN, zorder=5)[0]) for side, _, c in ROWS}
    fills = []
    fig.legend([Line2D([], [], color=ROWS[0][2], lw=2.0), Line2D([], [], color="0.3", lw=2.0),
                Line2D([], [], color=ROWS[1][2], lw=2.0), Line2D([], [], color="0.3", lw=2.0, ls=GEN)],
               [names["lo"], "Training set", names["hi"], "Evaluation set"], loc="lower center", ncol=2,
               bbox_to_anchor=(.5, .01), fontsize=FS["label"], frameon=False, handlelength=2.2, columnspacing=2.4)
    clock = ax.text(.975, .975, "", transform=ax.transAxes, ha="right", va="top", fontsize=FS["label"])
    bar_ax = ax.inset_axes([.775, .885, .20, .026])
    bar_ax.set_xlim(0, 1)
    bar_ax.set_ylim(0, 1)
    bar_ax.set_axis_off()
    bar_ax.add_patch(Rectangle((0, 0), 1, 1, color="0.88", lw=0))
    bar = bar_ax.add_patch(Rectangle((0, 0), 0, 1, color="0.35", lw=0))
    frames = [0] * O_START + list(range(nF)) + [nF - 1] * D_HOLD

    def draw(i):
        f = frames[i]
        a = {side: F[run][f] for (side, _, _), run in zip(ROWS, runs)}
        y = a["lo"]                                                               # learned low: solid over bin 0
        L["lo"][0].set_data([-.5, .5], [y[0], y[0]])
        L["lo"][1].set_data([.5, .5, 1.5, 1.5, 2.5], [y[0], y[1], y[1], y[2], y[2]])
        y = a["hi"]                                                               # learned high: solid over bin 2
        L["hi"][0].set_data([1.5, 2.5], [y[2], y[2]])
        L["hi"][1].set_data([-.5, .5, .5, 1.5, 1.5], [y[0], y[0], y[1], y[1], y[2]])
        for p in fills:
            p.remove()
        fills.clear()
        for b in range(3):                                                        # each bin's gap, higher colour
            up = "lo" if a["lo"][b] >= a["hi"][b] else "hi"
            fills.append(ax.fill_between([b - .5, b + .5], a["lo"][b], a["hi"][b], color=dict(
                (s_, c_) for s_, _, c_ in ROWS)[up], alpha=FILL_A, lw=0, zorder=3))
        t = T_[f]
        n = int(round(t, -2 if t >= 1000 else (-1 if t >= 100 else 0)))
        clock.set_text(f"{n:,} episode" + ("" if n == 1 else "s"))
        bar.set_width(f / max(nF - 1, 1))
        return []
    anim = FuncAnimation(fig, draw, frames=len(frames), interval=O_MS, blit=False)
    path = path or os.path.join(os.path.dirname(fig_path("x.png")), "supp gifs", "fig_learn_holdout_movie_bins3.gif")
    durations = [O_MS - 5 if k % 2 == 0 else O_MS + 5 for k in range(len(frames))]
    anim.save(path, writer=DurationWriter(durations, fps=1000 / O_MS), dpi=D_DPI)
    plt.close(fig)
    print("saved", path, len(frames), "frames")
    print("last frame (accuracy by bin):", {s_: np.round(F[r_][-1], 3).tolist() for (s_, _, _), r_ in zip(ROWS, runs)})


def movie_dense(chunk=(.1, 1), path=None):
    """John 11-11 (noon): "update the holdout gif (now in supgifs folder) to be less smoothened data but smoother
    animation (more episodes and frames and higher resolution images) ... remove the 'each network learns' comment and
    put the x-label on both panels (just as 'Factorization Regret') and remove 'same count' from the legend and try to
    minimize the empty space, i think more can maybe go inside the panels". Every stored checkpoint (ALL: 136 per run,
    every 25 raw episodes to 500, then every 100); D_FRAMES frames evenly spaced in log training episodes from the first
    trained checkpoint (20 episodes) to 9,600, each the Gaussian-weighted mean of the checkpoints around it (SD D_SMOOTH
    decades: a 12% window in training time; at 9,600 episodes it averages ~25 checkpoints, at 100 episodes ~2), after
    D_START frames of the untrained network; pale trails every D_TRAIL frames; D_DPI dpi. Bins as movie_single (axis
    10^-3.5 to 10^2.5, centre .32 nats between the chunks). 11-11 (afternoon): from 20 training episodes (no frames of
    the untrained network: D_START = 0), the curve through FR a kernel mean (profiles_kernel: SD FR_SD decades, points
    every .05 decade) instead of half-decade bins, D_SMOOTH .08. Since 11-11 15:40 it runs to raw 18,800 (15,040
    training episodes; the continued runs, train_learn_masks.py cont). The kernel is one-sided at the last frames (no
    checkpoint after 15,040), so the last frame averages roughly the last 3,000 episodes and slightly understates a
    curve that is still rising (the learned-low network on high FR: .518 at 15,040, still +.017 per 1,000 episodes).
    Since 11-11 18:00 (John: "can you please have the gif keep running with the red training episodes? I want to see how
    far it has gotten"): each network runs to its own last replayed checkpoint; the clock and frames follow the longest
    (the red, learned-low network, continued to 30,000 episodes); a curve whose network has no later checkpoint holds
    its last smoothed value, with a "held at ..." note in its panel (the random control: noted in the key). Then (John:
    "it looks like both stop at the same time? Are you sure the red is levelling off? or is the animation maybe a bit
    bugged?"): the frames were evenly spaced in log training episodes, so the 8,000 episodes after 15,040 had 14 of 220
    frames and the green network's last 1,900 only 3. Now log-spaced to 2,000 episodes (D_FRAMES_LOG), then evenly
    spaced in episodes (D_FRAMES_LIN), with the late smoothing capped near 600 episodes (D_SMOOTH_LIN); the bar shows
    progress through the animation."""
    apply_fig_style()
    plt.rcParams.update({"mathtext.fontset": "cm"})
    from matplotlib.patches import Rectangle
    lo, hi = chunk
    runs = [h.band_name("lo", lo, hi), h.band_name("hi", lo, hi), h.BRAND]
    x, F, HELD, T_, LT, tmax = dense_frames(runs)
    nF = len(T_)
    left, pw, gap, top, ph, bot = .62, 3.15, .30, .30, 2.15, .78
    W, H = left + 2 * pw + gap + .10, top + ph + bot
    fig = plt.figure(figsize=(W, H))
    names = {"lo": rf"learned FR $<$ {h.fmt(lo)}", "hi": rf"learned FR $>$ {h.fmt(hi)}"}
    panels = []
    for j, (side, _, c) in enumerate(ROWS):
        ax = fig.add_axes([(left + j * (pw + gap)) / W, 1 - (top + ph) / H, pw / W, ph / H])
        th = lo if side == "lo" else hi
        ax.axvspan(*((1e-6, lo) if side == "lo" else (hi, 1e4)), color=c, alpha=.10, lw=0, zorder=0)
        ax.axvline(th, color=c, lw=.8, zorder=1)
        ax.axvline(np.sqrt(lo * hi), color="0.75", lw=.6, ls=(0, (1, 2)), zorder=0)
        axes_style(ax, True, j == 0, (10 ** -3.5, 10 ** 2.5), [1e-3, 1e-1, 10], [r"$10^{-3}$", "0.1", "10"],
                   [1e-2, 1, 100])
        ax.set_title(names[side], fontsize=FS["title"], color=c, pad=4)
        ax.set_xlabel("Factorization Regret", fontsize=FS["label"], labelpad=2)
        if j == 0:
            ax.set_ylabel("accuracy", fontsize=FS["label"], labelpad=3)
        D = panel(ax, side, th, c, runs[j], True)
        D["trail"] = [(ax.plot([], [], color=pale(c), lw=.6, zorder=2)[0],
                       ax.plot([], [], color=pale(c), lw=.6, ls=GEN, zorder=2)[0]) for _ in range(0, nF, D_TRAIL)]
        D["inn"].set_linewidth(2.0)
        D["out"].set_linewidth(2.0)
        panels.append(D)
    rnd_last = int(round(10 ** LT[h.BRAND][-1]))
    fig.legend([Line2D([], [], color="0.3", lw=2.0), Line2D([], [], color="0.3", lw=2.0, ls=GEN), Line2D([], [], **RND)],
               ["Training set", "Evaluation set", "Training at random steps"
                + (f" (to {rnd_last:,} episodes)" if rnd_last < tmax - 1 else "")], loc="lower center", ncol=3,
               bbox_to_anchor=(.5, .01), fontsize=FS["tick"], frameon=False, handlelength=2.2, columnspacing=2.0)
               # one row under the panels: the longer labels do not fit the left panel's free corner
    rax = panels[1]["inn"].axes                                                   # the clock, inside the right panel
    clock = rax.text(.975, .975, "", transform=rax.transAxes, ha="right", va="top", fontsize=FS["label"])
    bar_ax = rax.inset_axes([.36, .918, .30, .028])                               # left of the clock, one line, above
                                                                                  # the curves
    bar_ax.set_xlim(0, 1)                                                         # progress through the animation
    bar_ax.set_ylim(0, 1)
    bar_ax.set_axis_off()
    bar_ax.add_patch(Rectangle((0, 0), 1, 1, color="0.88", lw=0))
    bar = bar_ax.add_patch(Rectangle((0, 0), 0, 1, color="0.35", lw=0))
    frames = list(range(nF)) + [nF - 1] * D_HOLD
    for D in panels:                                                              # "held at" note, under the clock line
        a_ = D["inn"].axes
        D["note"] = a_.text(.975, .84, "", transform=a_.transAxes, ha="right", va="top", fontsize=FS["tick"],
                            color=D["inn"].get_color())
        D["last"] = int(round(10 ** LT[D["run"]][-1]))

    def draw(i):
        f = frames[i]
        for D in panels:
            (xi, yi), (xo, yo) = split(x, F[D["run"]][f], D["th"], D["side"])
            D["inn"].set_data(xi, yi)
            D["out"].set_data(xo, yo)
            D["note"].set_text(f"held at {D['last']:,} episodes (still training)" if HELD[D["run"]][f] else "")
            for m, (ti, to) in enumerate(D["trail"]):
                if m * D_TRAIL <= f and not HELD[D["run"]][m * D_TRAIL]:
                    (a1, b1), (a2, b2) = split(x, F[D["run"]][m * D_TRAIL], D["th"], D["side"])
                    ti.set_data(a1, b1)
                    to.set_data(a2, b2)
                else:
                    ti.set_data([], [])
                    to.set_data([], [])
            D["rnd"].set_data(x, F[h.BRAND][f])
        t = T_[f]
        clock.set_text(f"{int(round(t, -2 if t >= 1000 else (-1 if t >= 100 else 0))):,} episodes")
        bar.set_width(f / max(nF - 1, 1))
        return []
    anim = FuncAnimation(fig, draw, frames=len(frames), interval=D_MS, blit=False)
    path = path or os.path.join(os.path.dirname(fig_path("x.png")), "supp gifs", "fig_learn_holdout_movie_mid.gif")
    anim.save(path, writer=PillowWriter(fps=1000 / D_MS), dpi=D_DPI)
    plt.close(fig)
    print("saved", path, len(frames), "frames")


if __name__ == "__main__":
    if "--bins3" in sys.argv:
        movie_bins3()
    elif "--overlay" in sys.argv:
        movie_overlay(xscale="linear" if "--linear" in sys.argv else "pct" if "--pct" in sys.argv else "log",
                      quantile="--quantile" in sys.argv)
    elif "--dense" in sys.argv:
        movie_dense()
    elif "--mid" in sys.argv:
        movie_single()
    else:
        movie()
