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
    + tuple(range(12400, 18801, 400))                                             # every stored checkpoint to 12,000
                                                                                  # (136), then every 4th of the
                                                                                  # continuation to raw 18,800 (15,040
                                                                                  # training episodes; John 11-11)
D_SMOOTH, D_FRAMES, D_MS, D_START, D_HOLD, D_TRAIL, D_DPI = .08, 220, 70, 0, 36, 11, 180
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
    curve that is still rising (the learned-low network on high FR: .518 at 15,040, still +.017 per 1,000 episodes)"""
    apply_fig_style()
    plt.rcParams.update({"mathtext.fontset": "cm"})
    from matplotlib.patches import Rectangle
    lo, hi = chunk
    runs = [h.band_name("lo", lo, hi), h.band_name("hi", lo, hi), h.BRAND]
    x, P = profiles_kernel(runs, ALL)
    tx = np.array([h.train_x(raw) for raw in ALL], float)
    lt = np.log10(np.maximum(tx, 1.))
    G = np.linspace(np.log10(20), np.log10(tx[-1]), D_FRAMES)                     # frame times (log10 episodes)
    Wk = np.exp(-.5 * ((G[:, None] - lt[None, 1:]) / D_SMOOTH) ** 2)              # trained checkpoints only
    Wk /= Wk.sum(1, keepdims=True)
    F = {run: np.concatenate([np.repeat(P[run][:1], D_START, 0), Wk @ P[run][1:]]) for run in runs}
    T_ = np.r_[np.zeros(D_START), 10 ** G]
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
    fig.legend([Line2D([], [], color="0.3", lw=2.0), Line2D([], [], color="0.3", lw=2.0, ls=GEN), Line2D([], [], **RND)],
               ["Training set", "Evaluation set", "Training at random steps"], loc="lower center", ncol=3,   # John 11-11
               bbox_to_anchor=(.5, .01), fontsize=FS["tick"], frameon=False, handlelength=2.2, columnspacing=2.0)
               # one row under the panels: the longer labels do not fit the left panel's free corner
    rax = panels[1]["inn"].axes                                                   # the clock, inside the right panel
    clock = rax.text(.975, .975, "", transform=rax.transAxes, ha="right", va="top", fontsize=FS["label"])
    bar_ax = rax.inset_axes([.36, .918, .30, .028])                               # left of the clock, one line, above
                                                                                  # the curves
    bar_ax.set_xscale("log")
    bar_ax.set_xlim(20, tx[-1])
    bar_ax.set_ylim(0, 1)
    bar_ax.set_axis_off()
    bar_ax.add_patch(Rectangle((20, 0), tx[-1] - 20, 1, color="0.88", lw=0))
    bar = bar_ax.add_patch(Rectangle((20, 0), 0, 1, color="0.35", lw=0))
    frames = list(range(nF)) + [nF - 1] * D_HOLD

    def draw(i):
        f = frames[i]
        for D in panels:
            (xi, yi), (xo, yo) = split(x, F[D["run"]][f], D["th"], D["side"])
            D["inn"].set_data(xi, yi)
            D["out"].set_data(xo, yo)
            for m, (ti, to) in enumerate(D["trail"]):
                if m * D_TRAIL <= f:
                    (a1, b1), (a2, b2) = split(x, F[D["run"]][m * D_TRAIL], D["th"], D["side"])
                    ti.set_data(a1, b1)
                    to.set_data(a2, b2)
                else:
                    ti.set_data([], [])
                    to.set_data([], [])
            D["rnd"].set_data(x, F[h.BRAND][f])
        t = T_[f]
        clock.set_text(f"{int(round(t, -2 if t >= 1000 else (-1 if t >= 100 else 0))):,} episodes")
        bar.set_width(max(t, 20) - 20)
        return []
    anim = FuncAnimation(fig, draw, frames=len(frames), interval=D_MS, blit=False)
    path = path or os.path.join(os.path.dirname(fig_path("x.png")), "supp gifs", "fig_learn_holdout_movie_mid.gif")
    anim.save(path, writer=PillowWriter(fps=1000 / D_MS), dpi=D_DPI)
    plt.close(fig)
    print("saved", path, len(frames), "frames")


if __name__ == "__main__":
    if "--dense" in sys.argv:
        movie_dense()
    elif "--mid" in sys.argv:
        movie_single()
    else:
        movie()
