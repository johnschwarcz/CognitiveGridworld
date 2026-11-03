"""Plotting helpers shared by the fig1c figures (main/REPLICATION/plot_fig1c_all.py), for any plotting script in
main/REPLICATION: paths, matplotlib styles, colours, panel furniture, interpolation in time, GIF output and three small
numerics. Each helper computes exactly what the research plotters it was taken from computed (bit for bit; checked).

    paths      ROOT, repo_root, data_dir
    style      style(name), name "default", "fig" or "movie": a complete rcParams state for one figure; MOVIE_RC,
               FIG_FONTS
    colours    bivariate (the (r_g, r_c) key) and its corners KEY, FR_CMAP, DIFF_CMAP, ENT_CMAP, AGENT_COLOR,
               AGENT_NAME, AGENT_NAME_BAYES
    furniture  ax_in, title2, letter, latent_key, clock_bar, square_limits
    time       catmull_rom, blend, polar, schedule, split_schedule, tail, tail_style
    GIF        grab, median_cut_palette, exact_palette, quantize, ExactQuantizer, save_gif
    numerics   grid_edges, posterior_means, jeffreys_half

Terms: a cell is (r_g, r_c), the values (0-9) of the goal and the contextual variable; a belief is goal-ordered, 20
numbers (the goal variable's 10 probabilities, then the contextual variable's 10); the step index tau = t - 1 (0: step
t = 1) runs continuously between steps in the movies. Positions and sizes of panel furniture are in inches.
Use from a script in main/REPLICATION (the script's folder is on sys.path):
    import plotting_helpers as ph
    with ph.style("fig"):
        ...
Run from the repository root with the eureka environment's python (lists the helpers):
    MPLBACKEND=Agg python main/REPLICATION/plotting_helpers.py"""
import contextlib
import os
import sys
import numpy as np


def repo_root():
    """the repository root: the first folder holding main/CognitiveGridworld.py above this file, VS Code's notebook
    file or the working directory (VS Code's interactive window may give a __file__ that does not exist)"""
    for start in (globals().get("__file__"), globals().get("__vsc_ipynb_file__"), os.getcwd()):
        d = os.path.abspath(start) if start and os.path.exists(start) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return d
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError("project root not found")


ROOT = repo_root()
sys.path[:0] = [p for p in (ROOT,) if p not in sys.path]                        # (main.utils)
import matplotlib                                                                # noqa: E402
import matplotlib.pyplot as plt                                                  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                            # noqa: E402
from PIL import Image                                                            # noqa: E402
from main.utils import FIG_FONTS, apply_fig_style                                # noqa: E402

__all__ = ["ROOT", "repo_root", "data_dir", "style", "MOVIE_RC", "FIG_FONTS", "bivariate", "KEY", "FR_CMAP",
           "DIFF_CMAP", "ENT_CMAP", "AGENT_COLOR", "AGENT_NAME", "AGENT_NAME_BAYES", "ax_in", "title2", "letter",
           "latent_key", "clock_bar", "square_limits", "catmull_rom", "blend", "polar", "schedule", "split_schedule",
           "tail", "tail_style", "grab", "median_cut_palette", "exact_palette", "quantize", "ExactQuantizer",
           "save_gif", "grid_edges", "posterior_means", "jeffreys_half"]


def data_dir(*parts):
    """the path main/DATA/<parts...> in the repository (not created)"""
    return os.path.join(ROOT, "main", "DATA", *parts)


# ═══════════════════════════════════════════════════════════════════
# Style
# ═══════════════════════════════════════════════════════════════════

MOVIE_RC = {"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
            "legend.fontsize": 6, "axes.linewidth": .6, "lines.linewidth": 1., "xtick.major.width": .6,
            "ytick.major.width": .6, "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5, "axes.labelpad": 2, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
            "mathtext.fontset": "cm"}                                            # the movies' type and lines (pt)
STYLES = ("default", "fig", "movie")


@contextlib.contextmanager
def style(name="fig", rc=None):
    """a complete matplotlib state for one figure, the same whatever ran before in the process: matplotlib's defaults
    (as a fresh process loads them), then the named style, then rc (a dict of further rcParams, optional); the previous
    rcParams come back on exit. Draw and grab every frame inside it: tick labels are made at drawing time. name:
        "default"  matplotlib's defaults (DejaVu Sans)
        "fig"      main.utils.apply_fig_style (cmr10 text, the house type sizes, 300 dpi figures) with math in
                   Computer Modern (mathtext.fontset "cm")
        "movie"    "fig" with MOVIE_RC on top (smaller type, thinner lines)
    Also usable as a decorator."""
    if name not in STYLES:
        raise ValueError(f"unknown style {name!r}; the styles are {', '.join(STYLES)}")
    with matplotlib.rc_context():
        matplotlib.rc_file_defaults()
        if name != "default":
            apply_fig_style()
            plt.rcParams["mathtext.fontset"] = "cm"
        if name == "movie":
            plt.rcParams.update(MOVIE_RC)
        if rc:
            plt.rcParams.update(rc)
        yield


# ═══════════════════════════════════════════════════════════════════
# Colours
# ═══════════════════════════════════════════════════════════════════

# the bivariate key: a teal ink for r_g and a magenta ink for r_c, mixed in OKLab (Ottosson 2020)
_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                 [0.0883024619, 0.2817188376, 0.6299787005]])
_LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                 [0.0259040371, 0.7827717662, -0.8086757660]])


def _lch(L, C, h):
    return np.array([L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h))])


def _srgb(lab):
    lin = np.clip((lab @ np.linalg.inv(_LAB).T) ** 3 @ np.linalg.inv(_LMS).T, 0, 1)
    return np.where(lin <= .0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - .055)


KEY = (_lch(.80, .06, 55), _lch(.56, .12, 192), _lch(.56, .15, 352), _lch(.28, .09, 285))
                                                         # OKLCh corners at (r_g, r_c) = (0, 0), (9, 0), (0, 9), (9, 9)


def bivariate(rg, rc, R=10):
    """the colour of each cell (r_g, r_c) (scalars or arrays) -> sRGB in [0, 1], (..., 3) float64: bilinear in OKLab
    between KEY's four corners, r_g adds teal and r_c magenta (both high: indigo, both low: a light peach). Each step of
    either value moves the colour by 3.4 / 3.3 OKLab x 100 (median; at least 1.6, also under simulated deuteranopia and
    protanopia); neither ink is the networks' blue or orange"""
    c00, c90, c09, c99 = KEY
    g, c = (np.asarray(v, np.float64)[..., None] / (R - 1) for v in (rg, rc))
    return _srgb((1 - g) * (1 - c) * c00 + g * (1 - c) * c90 + (1 - g) * c * c09 + g * c * c99)


FR_CMAP = LinearSegmentedColormap.from_list("fr", ["#bdbdbd", "#fdbb84", "#e34a33", "#7f0000"])
                                                         # grey -> dark red: factorization regret, grid distances
DIFF_CMAP = LinearSegmentedColormap.from_list(
    "bkr", [(0, "#a8d4ff"), (.22, "#2a6dff"), (.5, "#000000"), (.78, "#ff2d1a"), (1, "#ffbfa8")])
DIFF_CMAP.set_bad("black")                               # signed differences: blue < 0 (black) < red
ENT_CMAP = LinearSegmentedColormap.from_list("ent", ["#c8c8c8", "#fdbb84", "#e34a33", "#7f0000"])
                                                         # low -> high entanglement
AGENT_COLOR = {"Trained": "#1f77b4", "Echo": "#ff7f0e", "Joint": "k", "Naive": "0.5"}
AGENT_NAME = {"Trained": "Trained RNN", "Echo": "ESN", "Joint": "Optimal", "Naive": "Naive"}   # the stills' wording
AGENT_NAME_BAYES = {"Trained": "Trained RNN", "Echo": "ESN", "Joint": "Optimal Bayes", "Naive": "Naive Bayes"}
                                                         # the movies' wording (their panel titles)


# ═══════════════════════════════════════════════════════════════════
# Panel furniture
# ═══════════════════════════════════════════════════════════════════

def ax_in(fig, x, y, w, h, **kw):
    """axes with the lower left corner at (x, y) and size w x h, in inches of fig (kw to fig.add_axes, e.g.
    projection="3d")"""
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def title2(ax, name, formula):
    """a panel's two-line title (house rule): the quantity's name over its formula, centred over ax on fixed
    baselines 17 and 5 pt above it (name FIG_FONTS label + .5 pt, formula FIG_FONTS tick pt in grey .25); on an
    invisible strip of axes, a header over several panels"""
    ax.annotate(name, (.5, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points",
                ha="center", va="baseline", fontsize=FIG_FONTS["label"] + .5)
    ax.annotate(formula, (.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                ha="center", va="baseline", fontsize=FIG_FONTS["tick"], color="0.25")


def letter(fig, ax, s, dx=.62):
    """the panel letter s in bold Computer Modern (FIG_FONTS label + 2 pt) on the baseline of title2's name line, dx
    inches left of ax"""
    W, H = fig.get_size_inches()
    x0, y0, w, h = ax.get_position().bounds
    fig.text(x0 - dx / W, y0 + h + 17 / 72 / H, r"$\mathbf{%s}$" % s, fontsize=FIG_FONTS["label"] + 2, ha="left",
             va="baseline")


def latent_key(ax, rgb=None, R=10):
    """the latent space in ax: each cell (r_g, r_c) a unit square in its bivariate colour, r_g along x and r_c along y,
    ticks at 0 and R - 1; rgb (R, R, 3), indexed [r_c, r_g], replaces the colours -> the image"""
    if rgb is None:
        gg, cc = np.meshgrid(np.arange(R), np.arange(R))
        rgb = bivariate(gg, cc, R)
    im = ax.imshow(rgb, origin="lower", extent=(-.5, R - .5, -.5, R - .5), interpolation="nearest")
    ax.set_xticks([0, R - 1]), ax.set_yticks([0, R - 1])
    return im


def clock_bar(fig, x, y, bx0, bx1, T=30, h=.10, dy=.2, fs_t=13, fs_tick=7.5):
    """a movie's clock (inches): the text "$t = ...$" at x, left-aligned, centred on height y; a bar of the T steps
    from bx0 to bx1, h high and centred on y (light grey .92, filled in grey .35 up to the frame's step); "$1$" and
    "$T$" (fs_tick pt, grey .4) centred under its first and last slot, their baseline dy below y -> set(tau, t=None):
    shows step index tau (0: t = 1): the bar filled to tau + 1 slots, the text "$t = t$" (fs_t pt; default t: the
    nearest step, int(floor(tau + .5)) + 1)"""
    W, H = fig.get_size_inches()
    txt = fig.text(x / W, y / H, "", fontsize=fs_t, ha="left", va="center")
    bar = ax_in(fig, bx0, y - h / 2, bx1 - bx0, h)
    bar.set_xlim(.5, T + .5), bar.set_ylim(0, 1)
    bar.axis("off")
    bar.barh(.5, T, left=.5, height=1, color="0.92", lw=0)
    fill = bar.barh(.5, 0, left=.5, height=1, color="0.35", lw=0)[0]
    for t_ in (1, T):
        fig.text((bx0 + (t_ - .5) / T * (bx1 - bx0)) / W, (y - dy) / H, f"${t_}$", ha="center", va="baseline",
                 fontsize=fs_tick, color="0.4")

    def set_t(tau, t=None):
        txt.set_text(f"$t = {int(np.floor(tau + .5)) + 1 if t is None else t}$")
        fill.set_width(tau + 1)

    return set_t


def square_limits(points, margin=1.):
    """a square data box holding points (..., 2): centred on their bounding box, its side the larger extent times
    margin -> (x0, x1), (y0, y1). For a box over several frames, pass their stacked minima and maxima"""
    P = np.asarray(points).reshape(-1, 2)
    lo, hi = P.min(0), P.max(0)
    c, s = (lo + hi) / 2, (hi - lo).max() * margin
    return (c[0] - s / 2, c[0] + s / 2), (c[1] - s / 2, c[1] + s / 2)


# ═══════════════════════════════════════════════════════════════════
# Time
# ═══════════════════════════════════════════════════════════════════

def catmull_rom(tau, T=30):
    """the Catmull-Rom spline through T steps at step index tau (0 .. T - 1): the 4 neighbouring steps (clamped at the
    ends) and their weights (summing to 1; the curve passes through every step, its slope is continuous) -> idx (4,),
    w (4,)"""
    i = min(int(np.floor(tau)), T - 2)
    f = tau - i
    idx = np.clip(np.arange(i - 1, i + 3), 0, T - 1)
    w = .5 * np.array([-f ** 3 + 2 * f ** 2 - f, 3 * f ** 3 - 5 * f ** 2 + 2, -3 * f ** 3 + 4 * f ** 2 + f,
                       f ** 3 - f ** 2])
    return idx, w


def blend(X, tau, axis=0):
    """X (..., T, ...) at step index tau along axis (the steps, T = X.shape[axis]) by catmull_rom, float64"""
    idx, w = catmull_rom(tau, X.shape[axis])
    return np.tensordot(np.take(X, idx, axis=axis).astype(np.float64), w, axes=([axis], [0]))


def polar(M):
    """the orthonormal factor of M (n, k): the nearest matrix with orthonormal columns (u v^T of its SVD)"""
    u, _, vt = np.linalg.svd(M, full_matrices=False)
    return u @ vt


def schedule(T=30, fps=25, motion_s=15., ease_s=.7, hold_s=(.5, 1.2)):
    """(tau, ms) per frame at fps: a hold of hold_s[0] s at t = 1; the steps over motion_s s with dtau/ds ~
    sqrt(tau + 2) (time per step ~ 1 / sqrt(t + 1)), the clock eased in from rest over ease_s s (smoothstep speed); a
    hold of hold_s[1] s at T"""
    s = np.arange(int(round(motion_s * fps)) + 1) / fps
    v = np.clip(s / ease_s, 0, 1)
    u = np.concatenate([[0], np.cumsum((v * v * (3 - 2 * v))[1:])])
    u *= 2 * (np.sqrt(T + 1) - np.sqrt(2)) / u[-1]                               # sqrt(tau + 2) = sqrt(2) + u / 2
    tau = np.clip((np.sqrt(2) + u / 2) ** 2 - 2, 0, T - 1)
    ms = np.full(len(tau), 1000 / fps)
    ms[0], ms[-1] = 1000 * hold_s[0], 1000 * hold_s[1]
    return list(zip(tau.tolist(), ms.tolist()))


def split_schedule(split_steps, T=30, fps=25):
    """tau per frame at fps (every frame 1000 // fps ms): a hold of 0.8 s at t = 1; 0.6 s per step for the steps
    around each split step s of split_steps (indices s - 3 .. s + 2), 0.3 s per step elsewhere (frames per step
    rounded); a hold of 2.5 s at T"""
    slow = set()
    for s in split_steps:
        slow.update(range(max(s - 3, 0), min(s + 3, T - 1)))
    taus = [0.] * int(.8 * fps)
    for s in range(T - 1):
        n = int(round((.6 if s in slow else .3) * fps))
        taus += list(s + np.arange(n) / n)
    return taus + [T - 1.] * int(2.5 * fps)


def tail(tt, P, tau, T=30):
    """a curve drawn up to tau: of its points P (n, 2) at increasing step indices tt (n,), those up to tau and a head
    interpolated linearly at tau -> segments (k, 2, 2), each segment's log time e = log(1 + tau) / log(T) at its
    middle (0 at t = 1, 1 at T), the head (2,)"""
    k = np.searchsorted(tt, tau, side="right")                                   # the curve's points up to tau
    if k >= len(tt):
        pts, ts = P, tt
    else:                                                                        # and the head between two of them
        f = (tau - tt[k - 1]) / (tt[k] - tt[k - 1])
        pts, ts = np.concatenate([P[:k], (P[k - 1] + f * (P[k] - P[k - 1]))[None]]), np.append(tt[:k], tau)
    e = np.log1p(ts) / np.log(T)
    return np.stack([pts[:-1], pts[1:]], 1), (e[1:] + e[:-1]) / 2, pts[-1]


def tail_style(e, scale):
    """the style of the movies' mean belief paths at log times e (tail): line widths (.45 + 1.25 e) scale^0.6 pt (scale:
    the panel's side over the stills' 1.11 in) and greys from .6 at t = 1 to black at T (colour strings) -> widths,
    greys"""
    return (.45 + 1.25 * e) * scale ** .6, [str(.6 * (1 - v)) for v in e]


# ═══════════════════════════════════════════════════════════════════
# GIF output
# ═══════════════════════════════════════════════════════════════════

def grab(fig):
    """draw fig -> its pixels, RGB (height, width, 3) uint8"""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def median_cut_palette(protos, keys=(), n=255, block=60):
    """a GIF palette of n colours, PIL's median cut (no dithering) over sample frames protos (RGB (h, w, 3) uint8, one
    width w); with keys (RGB in [0, 1]), a strip of swatches under them, each block x min(block, w // len(keys)) px, so
    that the cut keeps those colours -> 'P' image for quantize"""
    protos = list(protos)
    if len(keys):
        sw = np.repeat(np.repeat((255 * np.array(keys)[None]).round().astype(np.uint8), block, 0),
                       min(block, protos[0].shape[1] // len(keys)), 1)
        sw = np.pad(sw, ((0, 0), (0, protos[0].shape[1] - sw.shape[1]), (0, 0)), constant_values=255)
        protos = protos + [sw]
    return Image.fromarray(np.concatenate(protos, 0)).quantize(colors=n, method=Image.Quantize.MEDIANCUT,
                                                               dither=Image.Dither.NONE)


def exact_palette(protos, exact, n=255):
    """a GIF palette of up to n colours: the colours exact (RGB in [0, 1]; to 8 bits, de-duplicated and sorted) as they
    are, the remaining slots from PIL's median cut of the sample frames protos (RGB (h, w, 3) uint8) -> 'P' image for
    quantize (its palette padded to 256 entries with white), the colours (k, 3) uint8, exact ones first (for
    ExactQuantizer), the number of exact colours"""
    exact = np.unique((255 * np.clip(np.array(exact, np.float64), 0, 1)).round().astype(np.uint8), axis=0)
    rest = n - len(exact)
    mc = Image.fromarray(np.concatenate(protos, 0)).quantize(colors=rest, method=Image.Quantize.MEDIANCUT,
                                                              dither=Image.Dither.NONE)
    cols = np.concatenate([exact, np.array(mc.getpalette(), np.uint8).reshape(-1, 3)[:rest]], 0)[:n]
    pal = Image.new("P", (1, 1))
    pal.putpalette(cols.ravel().tolist() + [255, 255, 255] * (256 - len(cols)))
    return pal, cols, len(exact)


def quantize(frame, palette):
    """frame (RGB array or image) -> 'P' image in the colours of palette (a 'P' image), by PIL's matching, no
    dithering: each pixel takes the colour nearest the lower corner of its 4 x 4 x 4 block of RGB values (each channel
    rounded down to a multiple of 4), so it can miss the nearest colour by a few levels (white can become an
    off-white); ExactQuantizer is exact"""
    img = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    return img.quantize(palette=palette, dither=Image.Dither.NONE)


class ExactQuantizer:
    """frame (RGB array (h, w, 3) uint8) -> 'P' image in the colours cols ((k, 3) uint8, k <= 256), each pixel to the
    nearest colour exactly (squared RGB distance, the first of ties; a lazily filled table of all 2^24 colours,
    32 MB)"""

    def __init__(self, cols):
        self.cols = np.asarray(cols, np.float32)
        self.flat = np.asarray(cols, np.uint8).ravel().tolist()
        self.lut = np.full(1 << 24, -1, np.int16)

    def __call__(self, X):
        X = np.asarray(X).astype(np.int32)
        code = (X[..., 0] << 16) | (X[..., 1] << 8) | X[..., 2]
        miss = np.unique(code[self.lut[code] < 0])
        for k in range(0, len(miss), 20000):
            u = miss[k:k + 20000]
            rgb = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], 1).astype(np.float32)
            self.lut[u] = ((rgb[:, None] - self.cols[None]) ** 2).sum(-1).argmin(1)
        P = Image.fromarray(self.lut[code].astype(np.uint8), "P")
        P.putpalette(self.flat)
        return P


def save_gif(frames, durations, path):
    """frames ('P' images) -> a looping GIF at path; durations: ms of each frame (a list) or of every frame (a
    number), passed to Pillow as they are (Pillow merges runs of identical frames, adding their durations) -> path"""
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    return path


# ═══════════════════════════════════════════════════════════════════
# Small numerics
# ═══════════════════════════════════════════════════════════════════

def grid_edges(cells):
    """the lattice's edges among cells (n, 2) of integer (r_g, r_c): the index pairs (k, k') of cells one step apart in
    r_g or in r_c, each once (from the lower cell), in the order of cells -> (n_edges, 2) int"""
    idx = {(int(g), int(c)): k for k, (g, c) in enumerate(cells)}
    return np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                     if (g + dg, c + dc) in idx])


def posterior_means(b, normalize="after"):
    """the posterior means (E[r_g], E[r_c]) of goal-ordered beliefs b (..., 20) -> (..., 2): sum_r r b(r) for each
    variable; normalize "after" divides it by sum_r b(r), "before" divides b by sum_r b(r) first, None takes b as
    normalized. The three differ in the last bits, and the arithmetic follows b's dtype (float32 b: float64 products
    and sums, float32 normalizer), so a port matches its original with the same choice and dtype"""
    b = np.asarray(b)
    b = b.reshape(*b.shape[:-1], 2, -1)
    R = np.arange(b.shape[-1])
    if normalize == "before":
        b = b / b.sum(-1, keepdims=True)
    m = (b * R).sum(-1)
    return m / b.sum(-1) if normalize == "after" else m


def jeffreys_half(p, q, eps=1e-15, clip_top=True):
    """half the Jeffreys divergence between distributions p and q over the last axis, .5 sum (p - q)(log p - log q)
    (nats), p and q clipped to [eps, 1 - eps] first (clip_top False: only from below)"""
    if clip_top:
        p, q = np.clip(p, eps, 1 - eps), np.clip(q, eps, 1 - eps)
    else:
        p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return .5 * ((p - q) * (np.log(p) - np.log(q))).sum(-1)


if __name__ == "__main__":
    import inspect
    for name in __all__:
        obj = globals()[name]
        if inspect.isfunction(obj) or inspect.isclass(obj):
            print(f"{name}{inspect.signature(obj)}\n    {(obj.__doc__ or '').strip().splitlines()[0]}")
        else:
            print(f"{name}: {type(obj).__name__}")
