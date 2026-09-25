import numpy as np
import torch
import gc
import os
import sys
import inspect
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from matplotlib.ticker import FuncFormatter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

_ROOT = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
while _ROOT != os.path.dirname(_ROOT) and not os.path.exists(os.path.join(_ROOT, 'main', 'CognitiveGridworld.py')):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
from main.CognitiveGridworld import CognitiveGridworld
from main.utils import fig_path, apply_fig_style

# Set False to re-collect; True re-plots from the cached harvest in seconds.
LOAD_HARVEST = True
HARVEST_CACHE = os.path.join(_ROOT, 'main', 'DATA', 'supplementary_harvest.npz')

F64, I64 = np.float64, np.int64

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

PLOT_CFG = {
    "figsize_A": (11.6, 3.4),    # 1 x 4 Diagnostics, all testing episodes (2.9 in/panel)
    "figsize_A_early": (10, 8),    # 2 x 3 Diagnostics: accuracy/PR over the early window
    "figsize_B": (11.6, 3.6),    # 1 x 4 Event-Triggered Dynamics
    "figsize_C": (17.5, 5.2),    # 1 x 3 Calibration
    "figsize_C2": (6.4, 3.4),    # 1 x 2 Calibration + gap
    "figsize_D": (12.2, 3.3),    # 1 x 4 Calibration by context count + FR mediation
    "figsize_E": (10.4, 3.3),    # 1 x 3 Calibration by tier, C inside the panels
    "early_epochs": 800,
    "smooth_w": 1,
    "line_width": 2.5,
    "title_fs": 18,
    "math_fs": 17,          # Subtitle font size
    "dyn_title_fs": 14,     # Dynamics panel: name and math share one line
    "label_fs": 14,
    "tick_fs": 12,
}

DYN_PARAMS = {
    "K_PCA": None,
    "E_START": -8, 
    "E_END": 8,
    "T_START": 12,
    "T_END": 30,
    "N_BANDS": 4
}

DYN_KEYS = ("l2", "pr", "head", "tail")
DIAG_KEYS = ("test_acc_through_training", "test_SII_coef_through_training",
             "test_model_update_dim_through_training", "test_model_input_dim_through_training")

# Joint green / Naive purple, matching the belief-surface and perf figures
TIER_COLORS = {"Expert": plt.cm.viridis(0.85), "Baseline": plt.cm.viridis(0.15)}
KIND_STYLE = {"Exact": dict(ls='-', marker='o', mec='k'),
              "Trained": dict(ls='--', marker='^', mec='r')}
DIAG_LS = '-'               # perfect calibration
DIAG_C = 'k'
DIAG_LW = .6                # the unity line is a reference, not a series: it should sit
                            # under the data rather than compete with it

# Confidence piles up against 1 -- half the episodes sit above 0.9 -- so a linear axis
# squeezes the whole turn-over into its last few percent. -log10(1 - p) spreads the tail
# while the ticks stay plain probabilities. Perfect calibration is then a curve, not a line.
CAL_XSCALE = "tail"          # "tail" | "linear"
# The same transform on y makes perfect calibration a straight line again, but the
# baselines then occupy 8.5% of the axis instead of 48%, so the turn-over -- the result --
# nearly vanishes. Worth switching on to look at the experts in the extreme tail.
CAL_YSCALE = "tail"          # "linear" | "tail"
TAIL_YTICKS = [0.1, 0.3, 0.5, 0.8, 0.9, 0.99, 0.999]
# The FR axis is a percentile, so it is a probability too -- and its action is all at the
# top. "tail" gives the upper half of FR 76% of the width instead of 48%. (logit is the
# wrong choice here: being symmetric, it spends its stretch on the low-FR end as well.)
SCISSOR_XSCALE = "tail"      # "linear" | "tail"
SCISSOR_XTICKS = [0.2, 0.5, 0.8, 0.9, 0.95, 0.975, 0.99]
TAIL_EPS = 5e-5              # 1/(2n) for n = 10,000 episodes per bin
TAIL_TICKS = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]


def _tail_fwd(v):
    return -np.log10(np.clip(1.0 - v, TAIL_EPS, 1.0))


def _tail_inv(v):
    return 1.0 - 10.0 ** (-v)


def _prob_clip(v):
    """An empirical rate of 0 or 1 from n draws sits at 1/(2n); neither transform admits
    the endpoints themselves."""
    return np.clip(v, TAIL_EPS, 1.0 - TAIL_EPS)


def _prob_axis(ax, which, mode, lo, hi, ticks=None, margin=0.25):
    """Put one probability axis on `mode`, with limits set in transformed units so the
    margin is even however hard the transform compresses an end."""
    setter = dict(x=(ax.set_xscale, ax.set_xticks, ax.xaxis, ax.set_xlim),
                  y=(ax.set_yscale, ax.set_yticks, ax.yaxis, ax.set_ylim))[which]
    set_scale, set_ticks, axis, set_lim = setter

    if mode == "logit":
        set_scale('logit')
        set_lim(_prob_clip(lo) * 0.7, 1.0 - (1.0 - _prob_clip(hi)) * 0.7)
        return

    if mode == "tail":
        set_scale('function', functions=(_tail_fwd, _tail_inv))
        if ticks is not None:
            set_ticks([t for t in ticks if lo <= t <= hi])
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        low = _tail_inv(_tail_fwd(lo) - margin)
        set_lim(low if low > 0 else -0.02, _tail_inv(_tail_fwd(hi) + margin))
        return

    set_lim(0.0, 1.0)
    set_ticks(np.linspace(0, 1, 6))


# name: (tier, kind, harvest prefix, which agent's harvest)
# 2 x 3 calibration by context count. ctx_2 rides on the main harvest; ctx_1 and ctx_3
# come from harvest_calib.py, which collects conf/acc/FR only.
CAL_CTX_SRC = {1: "calibration_ctx1.npz", 2: None, 3: "calibration_ctx3.npz"}
CAL_CTX_KEYS = ("joint_conf", "joint_acc", "naive_conf", "naive_acc",
                "net_conf", "net_acc", "fr")
# baselines drawn first so the experts land on top of them
CAL_DRAW_ORDER = ["Naive", "Echo State", "Joint", "Fully Trained"]
CAL_ROW_TITLES = ["Confidence-Accuracy calibration", "Dependence on Factorization Regret"]
# Absolute-FR bin edges for the mediation panel. Absolute, not percentile: the whole point
# is that the same regret means the same thing at every context count, which a per-C rank
# would hide.
CAL_FR_EDGES = np.array([0, .5, 1, 2, 4, 8, 16, 40])
CAL_SHORT = {"Joint": "Joint", "Fully Trained": "Trained",
             "Naive": "Naive", "Echo State": "Echo"}
CAL_MARKER = {"Exact": "o", "Trained": "^"}
CAL_XTICKS = [0.2, 0.4, 0.6, 0.8, 1.0]
# "C = k" goes inside each scatter panel: the region above the diagonal is empty by
# construction (accuracy never exceeds confidence there), and an axes title would spend a
# whole row of figure height on one symbol.
CAL_C_LABEL_XY = (.045, .965)
CAL_WPAD = .15
# tight_layout's wspace is ONE number applied to every boundary, so the margin the heatmap
# needs for its row labels was also inserted between the scatter panels -- which is why
# they sat 0.53 in apart while the gap before the heatmap was only 0.36 in. The scatter row
# is re-laid by hand afterwards: a small gap between panels, and the width that frees up
# split between the panels themselves and one deliberate gap before the heatmap.
CAL_SCATTER_GAP = .012   # figure fraction, between neighbouring scatter panels
CAL_GROUP_GAP = .025     # figure fraction, added before the heatmap column
# Heatmap row index: agent names on the major ticks, the C value on the minor ticks just
# inside them, so tight_layout reserves room for both (a bare ax.text would not be seen).
CAL_NAME_PAD = 19.0
CAL_TITLE_PAD = .062     # clears the "C" header sitting above the heatmap axes
CAL_RECT_TOP = .88
# s=32 with a 1.0 edge was too small at print scale: the red edge bled into the fill and
# the trained agents' tier colour read as orange and maroon instead of green and purple.
# Per shape, because a triangle holds about half the ink of a circle at the same `s`, so
# one size leaves the triangles' fill too thin to survive their own outline.
CAL_POINT_S = {"o": 37, "^": 53}
CAL_POINT_LW = .75
# Second version: the tier becomes the panel, which frees colour to carry C -- reinforced
# by size so the three still separate where they land on top of each other.
CAL_TIER_PANELS = {"Experts": ("Joint", "Fully Trained"),
                   "Baselines": ("Naive", "Echo State")}
# Translucent fills, no edge: C is the fill colour, and where points land on each other
# -- most of Experts -- they show through one another instead of hiding. Every point is
# the same size, so size says nothing and shape is left to say what computed the belief.
CAL_TIER_S = 30
CAL_TIER_ALPHA = .8
# "curve" replaces each agent's points with one line, so a panel carries 6 marks instead
# of 60. It cannot be a closed contour: per-episode accuracy is 0/1, so the cloud in this
# plane is two horizontal lines and a 90% region of it is degenerate. What IS continuous
# is confidence, so the line spans the central CAL_TIER_COVER of that distribution -- the
# 90% is a real share of the episodes, read along the axis that has a spread to share.
CAL_TIER_STYLE = "ols"     # "ols" | "misfit" | "logistic" | "kernel" | "curve" | "points" | "contour"
CAL_TIER_COVER = .90
CAL_TIER_BINS = 24
CAL_TIER_LS = {"Exact": ":", "Trained": "--"}
CAL_TIER_LW = 1.7
# Axis scaling for the tier panels. Confidence piles up against 1, so a linear axis
# spends most of its width where nothing happens -- the same argument the 1 x 2 figure's
# CAL_XSCALE makes. "tail" is -log10(1 - p): plain probability ticks, stretched at the
# top. "tail-x" stretches confidence only, which keeps accuracy readable as a rate.
CAL_TIER_SCALE = "linear"   # "linear" | "tail-x" | "tail" | "logit" | "loglog"
# Cox calibration: logit P(correct) = a + b logit(conf), maximum likelihood on every
# episode. Perfect calibration is a = 0, b = 1, so the two numbers ARE the result -- but
# a monotone two-parameter model cannot bend, and the baselines' curves do. Drawn thin
# and under the kernel when overlaid, so thick reads as data and thin as model.
CAL_TIER_FIT = False        # overlay the fit on whatever style is drawn
CAL_FIT_MODEL = "ols"       # "ols" (accuracy = a*conf + b) | "logistic" (Cox, on logits)
CAL_FIT_BAND = False        # shade the fitted line's confidence band. Off: at this n the
                            # 95% band is ~1% of the y range, so it only thickened the
                            # lines -- the numbers belong in the caption, not the page.
CAL_FIT_Z = 1.96            # 95%
CAL_FIT_LW = .9
CAL_ENV_ALPHA = .25         # the "misfit" style: shading between the fit and the DATA.
# The empirical side is equal-mass bins -- each point a raw fraction correct over ~n/BINS
# episodes, with a binomial SE -- and NOT the kernel curve. Measured against the kernel
# the envelope is wrong in both directions: boundary bias inflates it where the agent
# really is a straight line (Joint C=3 read 0.034 against 0.010 from the data) and the
# bandwidth smooths away the real departure (Naive C=3 read 0.052 against 0.092).
CAL_MISFIT_BINS = 20
CAL_FIT_ALPHA = .55
CAL_TIER_LINE_ALPHA = .85   # the six curves cross and overlap; a little transparency
                            # lets the one underneath stay readable through the one on top
# "contour": the episodes themselves, with no binning at all -- a smoothed 2D histogram
# of (confidence, correct), contoured at the level enclosing CAL_TIER_COVER of them.
# y is 0/1 per episode, so the smoothing in that direction is what makes a contour exist;
# CAL_CONTOUR_SMOOTH is in histogram cells, and the y figure is the one that decides how
# much of the shape is estimator rather than data.
CAL_CONTOUR_BINS = (140, 72)
CAL_CONTOUR_SMOOTH = (2.5, 5.0)
# "kernel": P(correct | confidence) by Nadaraya-Watson over every episode -- no bin edges
# anywhere, so the only choice is a continuous bandwidth instead of an arbitrary cut.
CAL_KR_BW = .02          # Gaussian bandwidth, in units of confidence. Doubling it from
                         # .01 settles the low-confidence ripple that log-log magnifies;
                         # .03 starts bowing the Experts curves off the diagonal, which
                         # is boundary bias, not data -- see the loglog_bw sweep.
CAL_KR_CELLS = 2000      # resolution of the sufficient statistics, not a binning of data
CAL_KR_BAND = True       # +/- 1 SE on the effective sample size behind each point
# Where to stop each curve.
#   "cover"  every curve rests on the same share of its own episodes, so the panels
#            compare like with like. The cut is taken entirely off the LOW end: confidence
#            piles up against 1, so a symmetric trim would spend half its budget on the
#            dense, well-estimated top -- which is where the baselines' turn-over is.
#   "se"     trim inward until +/- 1 SE is under CAL_KR_SE_MAX. Adapts per agent, but the
#            share of episodes behind each curve then varies (94-99.9% here).
#   "none"   the full observed range.
CAL_KR_TRIM = "se"
CAL_KR_COVER = .90
CAL_KR_SE_MAX = .02

CAL_AGENTS = {"Joint": ("Expert", "Exact", "joint", "trained"),
              "Fully Trained": ("Expert", "Trained", "net", "trained"),
              "Naive": ("Baseline", "Exact", "naive", "trained"),
              "Echo State": ("Baseline", "Trained", "net", "echo")}

AGENT_COLORS = {"Trained": "#1f77b4", "Echo": "#ff7f0e", "Joint": "#2ca02c", "Naive": "#d62728"}
# Marker outline in the calibration scatters: the two networks are outlined in red, the
# two exact Bayesian agents in black, so a glance separates measured from computed.
NET_EDGE, EXACT_EDGE = "#d62728", "k"
OBS_COLOR = "#4d4d4d"

# One shared look for every figure (main.utils.apply_fig_style). Type is fixed, so every
# figure matches as long as its panels stay near FIG_PANEL (5.75 x 3.3 in) -- which is
# what the figsize entries above are built to give.
FS = apply_fig_style()

# ═══════════════════════════════════════════════════════════════════
# Logic: Core Math & Signal Processing
# ═══════════════════════════════════════════════════════════════════

def _smooth(y, w):
    y = np.asarray(y, float)
    if w <= 1: return y
    k = np.ones(int(w)) / float(w)
    return np.convolve(np.pad(y, (int(w)//2, int(w)//2), mode="edge"), k, mode="valid")

def _ep_xy(y, lim=None):
    y = np.asarray(y, float)
    idx = np.arange(y.size) if lim is None else np.arange(min(y.size, int(lim)))
    return idx + 1, y[idx]

def pr_stat(evals):
    """Spectral Participation Ratio from covariance eigenvalues."""
    e = np.asarray(evals, float)
    return (e.sum(-1)**2) / ((e*e).sum(-1) + 1e-12)

def pr_gain(agent):
    """PR(RNN) - PR(read-in): how much dimensionality the recurrent update ADDS.

    PR(RNN) alone confounds the network's own expansion with whatever structure the
    read-in already handed it, so the difference is the part the recurrence is
    responsible for. Zero means the update rides on the input's existing dimensionality.
    """
    return (pr_stat(agent.test_model_update_dim_through_training)
            - (pr_stat(agent.test_model_input_dim_through_training) + 1e-12))


def approx_lik(b, eps=1e-99):
    b = np.maximum(b.astype(F64), eps)
    px = np.zeros_like(b)
    px[:, 0], r = b[:, 0], b[:, 1:] / b[:, :-1]
    px[:, 1:] = r / r.sum(-1, keepdims=True)
    return px

def step_dkl(p, q, eps=1e-99):
    p, q = np.maximum(p.astype(F64), eps), np.maximum(q.astype(F64), eps)
    p, q = p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)
    return 0.5 * np.sum(p * (np.log(p) - np.log(q)) + q * (np.log(q) - np.log(p)), -1)

def pca_upd(m, k_pca=None, chunk=100_000):
    """
    Computes global covariance across time and batch: Neuron x (Time * Batch),
    and projects the centered representations onto the principal axes.

    The axes are eigenvectors of the N x N Gram matrix rather than right singular
    vectors of the (B*T) x N matrix: the same basis up to sign, at fixed memory.
    """
    upd = m.model_update_flat
    B, T, N = upd.shape
    x = torch.as_tensor(np.nan_to_num(upd.reshape(B * T, N), 0., 0., 0.)).to(m.device, torch.float32)
    x -= x.mean(0, keepdim=True)

    G = torch.zeros(N, N, device=x.device, dtype=torch.float64)
    for i in range(0, x.shape[0], chunk):
        c = x[i:i + chunk].double()
        G += c.T @ c
    ev, V = torch.linalg.eigh(G)
    ev, V = ev.flip(-1).cpu().numpy(), V.flip(-1)
    V = (V if k_pca is None else V[:, :k_pca]).float()
    evr = np.maximum(ev, 0)[:V.shape[1]]
    evr = evr / max(evr.sum(), 1e-99)

    z = torch.empty(x.shape[0], V.shape[1], device=x.device, dtype=torch.float32)
    for i in range(0, x.shape[0], chunk):
        z[i:i + chunk] = x[i:i + chunk] @ V
    return z.reshape(B, T, -1).cpu().numpy().astype(F64), evr

def equal_variance_bands(evr, P):
    """Split the PC axis into P bands of equal explained variance.

    Each boundary is placed at the PC whose cumulative variance is NEAREST it. Advancing
    on the first PC to EXCEED the boundary leaves leading bands empty whenever one PC is
    large: the echo state's top PC alone carries 25%, so a 25% head would become 2 PCs.
    """
    cum, n = np.cumsum(evr), len(evr)
    edges, prev = [], -1
    for k in range(1, P):
        c = int(np.argmin(np.abs(cum - k / P)))
        c = min(max(c, prev + 1), n - (P - k))
        edges.append(c); prev = c
    bid = np.full(n, P - 1, dtype=I64)
    start = 0
    for b, e in enumerate(edges):
        bid[start:e + 1] = b; start = e + 1
    return bid


def step_participation_ratio(z, eps=1e-12):
    """
    Computes the instantaneous effective dimensionality of the projected state
    onto the global covariance axes: PR(proj(RNN_t)).
    """
    z_sq = z ** 2
    return (np.sum(z_sq, axis=-1) ** 2) / np.maximum(np.sum(z_sq ** 2, axis=-1), eps)

def step_sparsity(x, eps=1e-99):
    N = x.shape[-1]
    l1 = np.sum(np.abs(x), axis=-1)
    l2 = np.linalg.norm(x, axis=-1) + eps
    return (np.sqrt(N) - (l1 / l2)) / (np.sqrt(N) - 1.0)

def get_xcorr(a, b, lag_min, lag_max):
    B, T = a.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        a_n = (a - np.nanmean(a, 1, keepdims=True)) / (np.nanstd(a, 1, keepdims=True) + 1e-9)
        b_n = (b - np.nanmean(b, 1, keepdims=True)) / (np.nanstd(b, 1, keepdims=True) + 1e-9)
    lags = np.arange(lag_min, lag_max + 1)
    xc = np.zeros((B, len(lags)))
    for i, lag in enumerate(lags):
        if lag < 0: xc[:, i] = np.nanmean(a_n[:, :lag] * b_n[:, -lag:], axis=1)
        elif lag > 0: xc[:, i] = np.nanmean(a_n[:, lag:] * b_n[:, :-lag], axis=1)
        else: xc[:, i] = np.nanmean(a_n * b_n, axis=1)
    return np.nanmean(xc, 0)

def prep_model_dynamics(m, prm):
    ts, te = prm.get("T_START", 0), prm.get("T_END", None)
    u = m.model_update_flat.astype(F64)
    B, T, _ = u.shape
    if te is None: te = T

    # Metrics
    l2 = np.linalg.norm(u, ord=2, axis=-1)
    dist = np.zeros((B, T))
    dist[:, 1:] = np.linalg.norm(u[:, 1:] - u[:, :-1], axis=-1)
    
    obs = m.obs_flat.astype(F64)
    v_prev = np.zeros((B, T))
    v_prev[:, 1:] = np.mean((obs[:, 1:] - obs[:, :-1])**2, -1)

    # counts are a sufficient statistic here, so the running mean over strictly earlier
    # steps is the Bayesian rate estimate: this is a prediction error, not a difference
    sur = np.zeros((B, T))
    run = np.cumsum(obs, 1)[:, :-1] / np.arange(1, T)[None, :, None]
    sur[:, 1:] = np.mean((obs[:, 1:] - run)**2, -1)
    
    # the only metric here about what the network outputs rather than its internal state
    gb = np.asarray(m.model_goal_belief, F64)
    belief = np.zeros((B, T))
    belief[:, 1:] = step_dkl(gb[:, 1:], gb[:, :-1])

    z, evr = pca_upd(m, k_pca=prm.get("K_PCA", None))

    # share of each step's energy held by the top and bottom equal-variance bands. Share,
    # not energy: the head loses SHARE while its absolute energy stays flat, because the
    # rest of the spectrum grows around it.
    P = prm.get("N_BANDS", 4)
    bid = equal_variance_bands(evr, P)
    e2 = z * z
    share = np.tensordot(e2 / np.maximum(e2.sum(-1, keepdims=True), 1e-99),
                         np.eye(P, dtype=F64)[bid], (2, 0))
    band_pc = np.bincount(bid, minlength=P)
    band_var = np.array([evr[bid == k].sum() for k in range(P)])

    rev = step_dkl(m.naive_px / m.naive_px.sum(-1, keepdims=True), 
                   approx_lik(m.joint_belief.astype(F64))).mean(-1)
    
    return {
        "l2": l2[:, ts:te],
        "head": share[:, ts:te, 0],
        "tail": share[:, ts:te, -1],
        "spar": step_sparsity(u[:, ts:te]),
        "pr": step_participation_ratio(z[:, ts:te]),
        "dist": dist[:, ts:te],
        "v_prev": v_prev[:, ts:te],
        "sur": sur[:, ts:te],
        "belief": belief[:, ts:te],
        "rev": rev[:, ts:te],
        "band_pc": band_pc,
        "band_var": band_var,
    }

class Bunch(dict):
    __getattr__ = dict.__getitem__


class Harvester:
    """Per-episode reductions for the calibration and dynamics panels.

    Passed to CognitiveGridworld as `external_logger`, so it runs inside the episode
    and the (batch, step, hidden) activations it reads are wiped with that episode
    instead of accumulating. Episodes and batch elements are the same unit of
    randomness here, so B x E is the same sample as (B*E) x 1 at a fraction of the
    GPU memory.
    """

    def __init__(self, prm):
        self.prm = prm

    def __call__(self, env):
        if not env.test_set:
            return {}
        out = {}
        for tag, gb, ac in (("joint", env.joint_goal_belief, env.joint_acc),
                            ("naive", env.naive_goal_belief, env.naive_acc),
                            ("net", env.model_goal_belief, env.model_acc)):
            out[f"{tag}_conf"], out[f"{tag}_acc"] = _final_conf_acc(gb, ac)
        out["fr"] = np.asarray(env.SII, F64)[:, -1]

        # get_xcorr already reduces within each episode, so averaging the per-episode
        # results is exact; only PR's PCA basis is re-estimated per episode.
        d = prep_model_dynamics(env, self.prm)
        for key in DYN_KEYS:
            out[f"x_{key}"] = get_xcorr(d[key], d["rev"], self.prm["E_START"], self.prm["E_END"])
        for tag, i in (("head", 0), ("tail", -1)):
            out[f"{tag}_pc"] = d["band_pc"][i]
            out[f"{tag}_var"] = d["band_var"][i]
        return out


def run_agent(load_env, reservoir, cfg, prm):
    """One agent at a time: harvest what the figures need, then free it."""
    agent = CognitiveGridworld(**cfg, reservoir=reservoir, load_env=load_env,
                               external_logger=Harvester(prm))
    out = Bunch(agent.custom_log)
    out["chance"] = 1.0 / agent.realization_num
    for k in DIAG_KEYS:
        out[k] = np.asarray(getattr(agent, k))
    del agent
    gc.collect()
    torch.cuda.empty_cache()
    return out


def episode_xcorr(H):
    out = {key: H[f"x_{key}"].mean(0) for key in DYN_KEYS}
    out.update({k: float(np.mean(H[k])) for k in ("head_pc", "head_var", "tail_pc", "tail_var")
                if k in H})
    return Bunch(out)


def _compact(v):
    v = np.asarray(v)
    return v.astype(np.float32) if v.dtype == np.float64 else v


def _harvest_stamp(prm):
    return np.array([prm[k] for k in ("T_START", "T_END", "E_START", "E_END", "N_BANDS")], I64)


def collect_or_load(cfg, prm, cache=None, load=None):
    """Harvest both agents, or re-plot from the cached harvest.

    The GPU work is the whole cost here, so it is cached: with LOAD_HARVEST set, styling
    changes re-plot in seconds instead of re-running the episodes. A cache written under
    a different DYN_KEYS or a different window is re-collected rather than half-read.
    """
    cache = HARVEST_CACHE if cache is None else cache
    load = LOAD_HARVEST if load is None else load

    if load and os.path.exists(cache):
        D = np.load(cache)
        want = {f"{t}x_{k}" for k in DYN_KEYS for t in ("tr_", "ec_")}
        missing = sorted({k[3:] for k in want - set(D.files)})
        moved = "stamp" not in D.files or not np.array_equal(D["stamp"], _harvest_stamp(prm))
        if missing or moved:
            print("harvest stale, re-collecting: "
                  + (f"missing {missing}" if missing else "window or lag range changed"))
        else:
            print(f"harvest loaded: {cache}")
            return tuple(Bunch({k[3:]: D[k] for k in D.files if k.startswith(tag)})
                         for tag in ("tr_", "ec_"))

    trained = run_agent("/sanity/fully_trained_ctx_2_e5", False, cfg, prm)
    echo = run_agent("/sanity/reservoir_ctx_2_e5", True, cfg, prm)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, stamp=_harvest_stamp(prm),
                        **{f"tr_{k}": _compact(v) for k, v in trained.items()},
                        **{f"ec_{k}": _compact(v) for k, v in echo.items()})
    print(f"harvest saved: {cache}")
    return trained, echo


# ═══════════════════════════════════════════════════════════════════
# Plotting Functions
# ═══════════════════════════════════════════════════════════════════

def plot_diagnostics_full(axes, agent, name):
    """All testing episodes: accuracy and FR correlation against episode, then the FR
    correlation against accuracy and against the recurrent state's dimensionality."""
    def ex(y, lim=None): return _ep_xy(_smooth(y, PLOT_CFG["smooth_w"]), lim)

    # 1. Final Step Accuracy
    x, y = ex(agent.test_acc_through_training[:, -1])
    axes[0].plot(x, y, c=AGENT_COLORS[name], label=name, zorder=3)
    baseline_color = AGENT_COLORS["Joint"] if name == "Trained" else AGENT_COLORS["Naive"]
    baseline_label = "Joint" if name == "Trained" else "Naive"
    acc = float(np.mean(agent.joint_acc)) if name == "Trained" else float(np.mean(agent.naive_acc))
    axes[0].axhline(acc, c=baseline_color, ls="--", label=baseline_label, zorder=5)

    # 2. Correlation(FR, Acc.)
    x, y = ex(agent.test_SII_coef_through_training)
    axes[1].plot(x, y, c=AGENT_COLORS[name], zorder=3)
    axes[1].axhline(y=0, c='k', alpha=0.5)

    # 3. FR correlation against accuracy
    x_acc = _smooth(agent.test_acc_through_training[:, -1], PLOT_CFG["smooth_w"])
    y_sii = _smooth(agent.test_SII_coef_through_training, PLOT_CFG["smooth_w"])
    n = min(len(x_acc), len(y_sii))
    axes[2].plot(x_acc[:n], y_sii[:n], c=AGENT_COLORS[name], zorder=3)
    axes[2].axhline(y=0, c='k', alpha=0.5)

    # 4. FR correlation against PR(RNN) -- both internal, so this asks whether they move
    # together without accuracy as the intermediary
    y_pr = _smooth(pr_gain(agent), PLOT_CFG["smooth_w"])
    m = min(len(y_pr), len(y_sii))
    axes[3].plot(y_pr[:m], y_sii[:m], c=AGENT_COLORS[name], zorder=3)
    axes[3].axhline(y=0, c='k', alpha=0.5)


def plot_diagnostics_early(axes, agent, name):
    """2 x 3 over the first PLOT_CFG['early_epochs'] episodes, where the FR correlation
    emerges. Top row: accuracy in full (so the early window has context), then accuracy
    and the RNN participation ratio across that window. Bottom row: the FR correlation
    over the same window, then that correlation against each quantity plotted above it,
    so a column reads as one pairing."""
    lim = PLOT_CFG["early_epochs"]
    def ex(y, l=lim): return _ep_xy(_smooth(y, PLOT_CFG["smooth_w"]), l)
    def cut(y): return _smooth(y, PLOT_CFG["smooth_w"])[:lim]
    c = AGENT_COLORS[name]

    # (0,0) Final Step Accuracy -- full range, with the early window marked
    x, y = ex(agent.test_acc_through_training[:, -1], None)
    axes[0, 0].plot(x, y, c=c, label=name, zorder=3)
    baseline_color = AGENT_COLORS["Joint"] if name == "Trained" else AGENT_COLORS["Naive"]
    baseline_label = "Joint" if name == "Trained" else "Naive"
    acc = float(np.mean(agent.joint_acc)) if name == "Trained" else float(np.mean(agent.naive_acc))
    axes[0, 0].axhline(acc, c=baseline_color, ls="--", label=baseline_label, zorder=5)
    axes[0, 0].axvline(lim, c='0.4', ls=':', lw=1.5, zorder=2)

    # (0,1) Accuracy, early window
    x, y = ex(agent.test_acc_through_training[:, -1])
    axes[0, 1].plot(x, y, c=c, zorder=3)

    # (0,2) PR(RNN) - PR(read-in), early window: the dimensionality the update ADDS
    x, y = ex(pr_gain(agent))
    axes[0, 2].plot(x, y, c=c, zorder=3)
    axes[0, 2].axhline(y=0, c='k', alpha=0.5)

    # (1,0) Correlation(FR, Acc.), early window
    x, y = ex(agent.test_SII_coef_through_training)
    axes[1, 0].plot(x, y, c=c, zorder=3)

    y_sii = cut(agent.test_SII_coef_through_training)
    # (1,1) and (1,2): the same correlation against each panel above it
    for j, q in enumerate((agent.test_acc_through_training[:, -1], pr_gain(agent))):
        v = cut(q)
        n = min(len(v), len(y_sii))
        axes[1, j + 1].plot(v[:n], y_sii[:n], c=c, zorder=3)
    for j in range(3):
        axes[1, j].axhline(y=0, c='k', alpha=0.5)


def _equalize_row(axes, ylabels, tight=0.013, pad=0.004):
    """Equal panel widths, with gaps that show the grouping.

    Panels sharing a y-label share a scale and carry no repeated ticks, so they sit
    tight together; the gap only opens where a new y-label and its ticks need room.
    """
    fig = axes[0].get_figure()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    pos = [a.get_position() for a in axes]
    need = [p_.x0 - a.get_tightbbox(r).transformed(inv).x0 for a, p_ in zip(axes, pos)]
    # gap before panel i: wide if it starts a new group, tight otherwise
    starts = [i for i in range(1, len(axes)) if ylabels[i] != ylabels[i - 1]]
    wide = max([need[i] for i in starts], default=0) + pad
    gaps = [wide if i in starts else tight for i in range(1, len(axes))]
    left, right, n = pos[0].x0, pos[-1].x1, len(axes)
    w = (right - left - sum(gaps)) / n
    x = left
    for i, (a, p_) in enumerate(zip(axes, pos)):
        if i:
            x += w + gaps[i - 1]
        a.set_position([x, p_.y0, w, p_.height])


def _share_groups(axes, ylabels):
    """Panels showing the same quantity share one scale, so only the leftmost keeps its
    ticks; the reclaimed width goes to the plots via tight_layout."""
    start = 0
    for i in range(1, len(axes) + 1):
        if i == len(axes) or ylabels[i] != ylabels[start]:
            grp = list(axes[start:i])
            if len(grp) > 1:
                lo = min(a.get_ylim()[0] for a in grp)
                hi = max(a.get_ylim()[1] for a in grp)
                for a in grp:
                    a.set_ylim(lo, hi)
                for a in grp[1:]:
                    a.tick_params(labelleft=False)
            start = i


def _finalize_row(axes, titles, ylabels, xlabels):
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], pad=15)
        ax.set_ylabel("" if i and ylabels[i] == ylabels[i - 1] else ylabels[i])
        ax.set_xlabel(xlabels[i])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, ls='--', alpha=0.3)

    _share_groups(axes, ylabels)
    axes[0].legend(loc='lower right', frameon=False, ncols=2)
    plt.tight_layout(w_pad=0.5)
    _equalize_row(axes, ylabels)


def finalize_layout_full(axes):
    _finalize_row(axes,
        ["Final Step Accuracy", "Correlation(FR, Acc.)", "Correlation vs. Acc.",
         r"Correlation vs. PR(RNN) $-$ PR(read-in)"],
        ["Accuracy", "r (Correlation)", "r (Correlation)", "r (Correlation)"],
        ["Testing Episode", "Testing Episode", "Accuracy", r"PR(RNN) $-$ PR(read-in)"])


def finalize_layout_early(axes):
    """axes is (2, 3). Each row is finished independently: the rows carry different
    quantities, so they group and equalize separately."""
    PR = r"PR(RNN) $-$ PR(read-in)"
    rows = [(["Final Step Acc.", "Early Final Step Acc.", PR],
             ["Accuracy", "Accuracy", PR],
             ["Testing Episode"] * 3),
            (["Correlation(FR, Acc.)", "Correlation vs. Acc.", "Correlation vs. " + PR],
             ["r (Correlation)"] * 3,
             ["Testing Episode", "Accuracy", PR])]
    for r, (titles, ylabels, xlabels) in enumerate(rows):
        for i, ax in enumerate(axes[r]):
            ax.set_title(titles[i], pad=15)
            ax.set_ylabel("" if i and ylabels[i] == ylabels[i - 1] else ylabels[i])
            ax.set_xlabel(xlabels[i])
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(True, ls='--', alpha=0.3)
        _share_groups(axes[r], ylabels)
    axes[0, 0].legend(loc='lower right', frameon=False, ncols=2)
    plt.tight_layout(w_pad=0.5, h_pad=2.2)
    for r, (_, ylabels, _) in enumerate(rows):
        _equalize_row(axes[r], ylabels)


# ═══════════════════════════════════════════════════════════════════
# Calibration: is confidence a usable signal, or decoupled from accuracy?
# Adapted from coggrid.plotting.plots (_draw_calibration / _draw_confidence_scissor)
# ═══════════════════════════════════════════════════════════════════

def _final_conf_acc(goal_belief, accuracy):
    """Confidence in the answer the observer would give, and whether it was right."""
    b = np.asarray(goal_belief, float)
    return b[:, -1].max(-1), np.asarray(accuracy, float)[:, -1]


def _calibration_curve(confidence, correct, n_bins, equal_mass=True, min_frac=2e-4):
    """Empirical accuracy per confidence bin.

    Equal-mass (quantile) bins by default: confidence piles up against 1, so equal-width
    bins put 150x more episodes in the top bin than the bottom kept one and average away
    the turn-over in the baselines' curves. Equal mass gives every point the same
    standard error and spends resolution where the episodes are.
    """
    if equal_mass:
        edges = np.quantile(confidence, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges) - 1, 0, n_bins - 1)
    centres, accuracy = [], []
    for b in range(n_bins):
        in_bin = idx == b
        if in_bin.sum() == 0 or (not equal_mass and in_bin.mean() < min_frac):
            continue
        centres.append(confidence[in_bin].mean())
        accuracy.append(correct[in_bin].mean())
    return np.array(centres), np.array(accuracy)


def _fr_bins(fr, values, n_bins):
    """Mean of `values` within each FR-rank bin. Rank, not raw FR: the distribution has
    a long tail, and what matters is the ordering of episodes, not the units."""
    return np.array([values[b].mean() for b in np.array_split(np.argsort(fr), n_bins)])


def _series(H, name):
    tier, kind, pre, _ = CAL_AGENTS[name]
    return (TIER_COLORS[tier], KIND_STYLE[kind],
            H[f"{pre}_conf"].ravel(), H[f"{pre}_acc"].ravel(), H["fr"].ravel())


def _tidy(ax):
    ax.grid(True, ls='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)


def draw_calibration(ax, harvests, names, chance, n_bins=20, equal_mass=True,
                     xscale=None, yscale=None):
    """Accuracy against self-reported confidence, all four observers in one panel."""
    xscale = CAL_XSCALE if xscale is None else xscale
    yscale = CAL_YSCALE if yscale is None else yscale

    grid = 1.0 - np.logspace(0, np.log10(TAIL_EPS), 600)
    ax.plot(grid, grid, color=DIAG_C, lw=1.1, ls=DIAG_LS, zorder=1)

    xlo, xhi, ylo, yhi = 1.0, 0.0, 1.0, 0.0
    for name in names:
        colour, style, conf, acc, _ = _series(harvests[CAL_AGENTS[name][3]], name)
        x, y = _calibration_curve(conf, acc, n_bins, equal_mass)
        if yscale != "linear":
            y = _prob_clip(y)
        xlo, xhi = min(xlo, x.min()), max(xhi, x.max())
        ylo, yhi = min(ylo, y.min()), max(yhi, y.max())
        ax.plot(x, y, color=colour, lw=1.5, ls=style['ls'], zorder=3, solid_capstyle='round')

    _prob_axis(ax, 'x', xscale, xlo, xhi, TAIL_TICKS, margin=0.18)
    _prob_axis(ax, 'y', yscale, ylo, yhi, TAIL_YTICKS, margin=0.18)
    ax.set(xlabel="Confidence", ylabel="Accuracy")
    _tidy(ax)
    return ylo, yhi


def draw_overconfidence(ax, harvests, names, n_bins=20, xscale=None):
    """Accuracy as a fraction of the confidence claimed, against episode FR rank.

    A calibrated observer sits at 1 whatever its accuracy, so the ratio is scale-free: an
    observer already 7% short at low FR and one that collapses at high FR are comparable,
    which a difference of probabilities is not.
    """
    xscale = SCISSOR_XSCALE if xscale is None else xscale
    x = np.linspace(1.0 / n_bins, 1.0, n_bins) - 0.5 / n_bins   # bin centres as a CDF value
    ax.axhline(1.0, color='k', lw=1.0, zorder=1)

    for name in names:
        colour, style, conf, acc, fr = _series(harvests[CAL_AGENTS[name][3]], name)
        ratio = _fr_bins(fr, acc, n_bins) / _fr_bins(fr, conf, n_bins)
        ax.plot(x, ratio, color=colour, lw=1.5, ls=style['ls'], zorder=3,
                solid_capstyle='round')

    _prob_axis(ax, 'x', xscale, x[0], x[-1], SCISSOR_XTICKS, margin=0.08)
    if xscale != "logit":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:g}"))
    ax.set(ylim=(0.0, 1.06), ylabel="Accuracy / Confidence",
           xlabel="Episodes ranked by Factorization Regret (percentile)")
    ax.set_yticks(np.linspace(0, 1, 6))
    _tidy(ax)


def draw_overconfidence_degenerate(ax, harvests, names):
    """The C = 1 case: FR is identically zero, so there are no ranks to bin.

    Each agent has one number, not a curve. Drawing it as a level across the panel keeps
    the row readable while the empty x-axis says there is no ordering behind it.
    """
    ax.axhline(1.0, color='k', lw=1.0, zorder=1)
    for name in names:
        colour, style, conf, acc, _ = _series(harvests[CAL_AGENTS[name][3]], name)
        ax.plot([0, 1], [acc.mean() / conf.mean()] * 2, color=colour, lw=1.5,
                ls=style['ls'], zorder=3, solid_capstyle='round')
    ax.set(xlim=(0, 1), ylim=(0.0, 1.06), ylabel="Accuracy / Confidence",
           xlabel=r"FR $\equiv$ 0, so no ranking exists")
    ax.set_xticks([])
    ax.set_yticks(np.linspace(0, 1, 6))
    _tidy(ax)


def _calibration_legends(ax_cal, scale="linear"):
    """Experts / Baselines is the colour grouping, so it titles the legend rather than
    the panels: each scissor panel holds one of each."""
    def handle(name):
        tier, kind, _, _ = CAL_AGENTS[name]
        return Line2D([], [], color=TIER_COLORS[tier], ls=KIND_STYLE[kind]['ls'], lw=2.4,
                      label=name)

    # A transformed accuracy axis turns the curves diagonal, which vacates the top-left
    # and fills the right; a linear one keeps them flat-topped and vacates the right.
    flat = scale == "linear"

    ax_cal.legend(handles=[Line2D([], [], color=DIAG_C, lw=1.1, ls=DIAG_LS,
                                  label='perfect calibration'),
                           handle("Joint"), handle("Fully Trained"),
                           handle("Naive"), handle("Echo State")],
                  loc='upper right' if flat else 'upper left', frameon=False,
                  handlelength=2.6, fontsize=FS["legend"], labelspacing=0.4, borderpad=0.2)


def plot_calibration_1x2(axes, trained, echo, n_bins=20, prob_scale=None):
    """[calibration (all four), accuracy - confidence (all four)].

    The gap panel states the miscalibration directly instead of asking the reader to
    subtract two curves by eye, so all four agents fit in one panel: the experts lie on
    zero, the baselines fall away from it as Factorization Regret rises.
    """
    H = {"trained": trained, "echo": echo}
    chance = float(trained["chance"])
    xs = CAL_XSCALE if prob_scale is None else prob_scale
    ys = CAL_YSCALE if prob_scale is None else prob_scale
    sx = SCISSOR_XSCALE if prob_scale is None else prob_scale

    # one binning for both panels: 20 equal-mass bins, so every point in the figure
    # rests on the same number of episodes (25,000 at a 500k-episode harvest)
    draw_calibration(axes[0], H, list(CAL_AGENTS), chance, n_bins, xscale=xs, yscale=ys)
    draw_overconfidence(axes[1], H, list(CAL_AGENTS), n_bins, xscale=sx)
    _calibration_legends(axes[0], ys)

    for ax, title in zip(axes, ["Confidence-Accuracy calibration", "Dependence on Factorization Regret"]):
        ax.set_title(title, pad=15)
    plt.tight_layout()


def _load_ctx_calibration(ctx, harvests=None):
    """-> ({"trained": {...}, "echo": {...}}, chance) for one context count.

    Episodes and batch are the same unit of randomness here, so (E, B) and (E*B,) are the
    same sample; everything is flattened.
    """
    src = CAL_CTX_SRC[ctx]
    if src is None:                       # ctx_2 is already in the main harvest
        tr, ec = harvests
        return ({"trained": {k: np.ravel(tr[k]).astype(F64) for k in CAL_CTX_KEYS},
                 "echo": {k: np.ravel(ec[k]).astype(F64) for k in CAL_CTX_KEYS}},
                float(tr["chance"]))
    d = np.load(os.path.join(_ROOT, "main", "DATA", src))
    H = {side: {k: np.ravel(d[f"{tag}_{k}"]).astype(F64) for k in CAL_CTX_KEYS}
         for tag, side in (("tr", "trained"), ("ec", "echo"))}
    return H, float(d["chance"])


def _fr_absolute_bins(H, name):
    """Accuracy / confidence inside fixed absolute-FR bins, for one agent."""
    _, _, conf, acc, fr = _series(H[CAL_AGENTS[name][3]], name)
    i = np.clip(np.digitize(fr, CAL_FR_EDGES) - 1, 0, len(CAL_FR_EDGES) - 2)
    num = np.bincount(i, weights=acc, minlength=len(CAL_FR_EDGES) - 1)
    den = np.bincount(i, weights=conf, minlength=len(CAL_FR_EDGES) - 1)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def _cal_key(name):
    """colour, marker, edge -- colour is what the belief factorizes over, marker and edge
    are what computes it, so the two crossed factors get one visual channel each."""
    tier, kind = CAL_AGENTS[name][0], CAL_AGENTS[name][1]
    return (TIER_COLORS[tier], CAL_MARKER[kind],
            NET_EDGE if CAL_AGENTS[name][2] == "net" else EXACT_EDGE)


def draw_calibration_points(ax, harvests, n_bins=10):
    """One marker per equal-mass confidence bin, joint-belief agents drawn last."""
    for i, name in enumerate(CAL_DRAW_ORDER):
        colour, marker, edge = _cal_key(name)
        _, _, conf, acc, _ = _series(harvests[CAL_AGENTS[name][3]], name)
        x, y = _calibration_curve(conf, acc, n_bins, True)
        ax.scatter(x, y, s=CAL_POINT_S[marker], facecolor=colour, edgecolor=edge,
                   linewidth=CAL_POINT_LW, marker=marker, zorder=3 + i)
    ax.plot([0, 1], [0, 1], color=DIAG_C, lw=DIAG_LW, zorder=-7)
    ax.set(xlim=(.1, 1.04), ylim=(.1, 1.04), xlabel="Confidence",  xticks=CAL_XTICKS, yticks=CAL_XTICKS)
    _tidy(ax)


def draw_fr_mediation(fig, ax, data, ctxs=(2, 3)):
    """Miscalibration by ABSOLUTE Factorization Regret, one row per agent and C.

    Rows pair up within an agent, which is the mediation: raising C shifts the FR
    distribution rather than changing what a given FR does. Exact joint Bayes is the
    control that stays pale at every regret, so the panel cannot be read as "high regret
    is simply hard".
    """
    rows, cs = [], []
    for name in CAL_AGENTS:
        for c in ctxs:
            rows.append(_fr_absolute_bins(data[c], name))
            cs.append(c)
    im = ax.imshow(np.array(rows), aspect="auto", cmap="magma", vmin=0, vmax=1,
                   extent=[0, len(CAL_FR_EDGES) - 1, len(rows) - .5, -.5])
    for y in np.arange(len(ctxs) - .5, len(rows) - 1, len(ctxs)):
        ax.axhline(y, color="w", lw=2.0)

    # the agent names name each PAIR, so they sit on a major tick at the pair's centre
    # and the C values on minor ticks inside them -- "C" then heads that column once,
    # instead of every row repeating it
    ax.set_yticks([i * len(ctxs) + (len(ctxs) - 1) / 2 for i in range(len(CAL_AGENTS))])
    ax.set_yticklabels([CAL_SHORT[n] for n in CAL_AGENTS], fontsize=FS["annot"] - 1,
                       rotation=90, ha="right", va="center")
    ax.set_yticks(range(len(rows)), minor=True)
    ax.set_yticklabels([f"{c}" for c in cs], fontsize=FS["annot"] - 1, minor=True)
    ax.tick_params(axis="y", which="major", length=0, pad=CAL_NAME_PAD)
    ax.tick_params(axis="y", which="minor", length=0, pad=6)
    c_txt = ax.annotate("$C$", xy=(0, 1), xycoords="axes fraction", xytext=(-8.5, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=FS["annot"] - 1, annotation_clip=False)

    # a cell is an average over its bin, so the tick belongs at the cell's centre and
    # carries the bin's midpoint rather than sitting on a boundary it does not describe
    mids = (CAL_FR_EDGES[:-1] + CAL_FR_EDGES[1:]) / 2
    ax.set_xticks(np.arange(len(mids)) + .5)
    ax.set_xticklabels([f"{v:g}" for v in mids], fontsize=FS["tick"])
    ax.set_xlabel("Factorization Regret (nats)")
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="x", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=.036, pad=.02)
    cb.set_label("Accuracy / Confidence", fontsize=FS["label"])
    cb.ax.tick_params(labelsize=FS["tick"] - 1)
    return cb, c_txt


def _cal_ctxs():
    """The context counts whose calibration harvest is actually on disk."""
    return [c for c in CAL_CTX_SRC
            if CAL_CTX_SRC[c] is None
            or os.path.exists(os.path.join(_ROOT, "main", "DATA", CAL_CTX_SRC[c]))]


def _box_c_column(fig, ax, c_txt, pad=4.0, gap=2.0):
    """Rule a box round the C column and its title, so the two read as one index.

    Only the LEFT edge is measured from the glyphs -- the column's width is set by them,
    not by the axes. The other three are pinned to the heatmap: the right edge stops
    `gap` short of it instead of cutting in, and the bottom sits on the heatmap's own
    bottom instead of ending halfway through the last row, where the last label happens
    to fall.
    """
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    labs = [t.get_window_extent(r) for t in ax.get_yticklabels(minor=True)]
    lab = Bbox.union(labs + [c_txt.get_window_extent(r)]).transformed(inv)
    w_in, h_in = fig.get_size_inches()
    px, py, gx = pad / (w_in * 72), pad / (h_in * 72), gap / (w_in * 72)
    p = ax.get_position()
    # right edge is pinned off the heatmap; mirror that same gap on the left so the
    # column sits centred in its box instead of shoved against one side
    x1 = p.x0 - gx
    x0 = lab.x0 - (x1 - lab.x1)
    y0, y1 = p.y0, lab.y1 + py
    fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
                             fill=False, edgecolor="0.55", lw=.7, zorder=6))


def _pack_row(axs, gap, right_trim=0.):
    """Re-lay a row of axes across its own span with one small gap between them.

    Keeps the row's left edge, pulls its right edge in by `right_trim` to open a gap
    before whatever follows, and hands the reclaimed width back to the panels.
    """
    boxes = [a.get_position() for a in axs]
    x0, x1 = boxes[0].x0, boxes[-1].x1 - right_trim
    w = (x1 - x0 - gap * (len(axs) - 1)) / len(axs)
    for i, a in enumerate(axs):
        a.set_position([x0 + i * (w + gap), boxes[i].y0, w, boxes[i].height])


def _stretch_to(fig, axs, floor):
    """Pull axes down until their decorations reach `floor`, keeping their tops.

    The heatmap column carries no legend beneath it, so it takes that height back -- but
    measured to the bottom of its tick labels and xlabel, or the figure ends up ragged.
    """
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for a in axs:
        p = a.get_position()
        deco = p.y0 - a.get_tightbbox(r).transformed(inv).y0
        a.set_position([p.x0, floor + deco, p.width, p.y1 - floor - deco])


def _cal_row_titles(fig, axes, n_scatter, mid):
    """The two group titles, placed above whatever the panels actually occupy."""
    top = max(a.get_position().y1 for a in axes) + CAL_TITLE_PAD
    fig.text(mid, top, CAL_ROW_TITLES[0], ha="center", va="bottom",
             fontsize=FS["title"])
    h = axes[n_scatter].get_position()
    fig.text((h.x0 + h.x1) / 2, top, CAL_ROW_TITLES[1], ha="center", va="bottom",
             fontsize=FS["title"])


def plot_calibration_row4(fig, axes, harvests, n_bins=10):
    """Calibration at each context count, then the regret that explains the trend.

    C = 1 is the control: with one context there is nothing to factorize, so all four
    agents coincide on the diagonal. It cannot appear in the mediation panel at all,
    because FR is identically zero there -- a point, not a distribution.
    """
    ctxs = _cal_ctxs()
    data = {c: _load_ctx_calibration(c, harvests)[0] for c in ctxs}
    for col, ctx in enumerate(ctxs):
        draw_calibration_points(axes[col], data[ctx], n_bins)
        axes[col].text(*CAL_C_LABEL_XY, f"$C = {ctx}$", transform=axes[col].transAxes,
                       ha="left", va="top", fontsize=FS["title"] * .82)
        if col:
            axes[col].set_ylabel("")
            axes[col].tick_params(labelleft=False)
    axes[0].set_ylabel("Accuracy")
    cb, c_txt = draw_fr_mediation(fig, axes[len(ctxs)], data)

    handles = []
    for name in CAL_AGENTS:
        colour, marker, edge = _cal_key(name)
        handles.append(Line2D([], [], ls="none", marker=marker, markerfacecolor=colour,
                              markeredgecolor=edge, markeredgewidth=CAL_POINT_LW,
                              markersize=np.sqrt(CAL_POINT_S[marker]), label=name))
    fig.tight_layout(w_pad=CAL_WPAD, rect=(0, .06, 1, CAL_RECT_TOP))
    _pack_row(axes[:len(ctxs)], CAL_SCATTER_GAP, CAL_GROUP_GAP)
    fig.canvas.draw()
    box = [axes[i].get_position() for i in range(len(ctxs))]
    mid = (min(b.x0 for b in box) + max(b.x1 for b in box)) / 2
    leg = fig.legend(handles=handles, loc="upper center", ncols=4, frameon=False,
                     fontsize=FS["legend"],
                     bbox_to_anchor=(mid, min(b.y0 for b in box) - .11))
    fig.canvas.draw()
    lb = leg.get_window_extent().transformed(fig.transFigure.inverted())
    _stretch_to(fig, (axes[len(ctxs)], cb.ax), lb.y0)
    fig.canvas.draw()
    _box_c_column(fig, axes[len(ctxs)], c_txt)
    _cal_row_titles(fig, axes, len(ctxs), mid)
    return ctxs


def _cal_tier_axes(ax, scale, lim):
    """Put the tier panel on `scale`, with a diagonal drawn to match.

    Under any probability transform the diagonal stops being a straight segment between
    two endpoints -- 0 and 1 are off at infinity -- so it is drawn as a dense curve.
    """
    xlo, xhi, ylo, yhi = lim
    if scale == "linear":
        ax.plot([0, 1], [0, 1], color=DIAG_C, lw=DIAG_LW, zorder=-7)
        ax.set(xlim=(.1, 1.04), ylim=(.1, 1.04), xticks=CAL_XTICKS, yticks=CAL_XTICKS)
        return
    if scale == "loglog":
        ax.plot([1e-3, 1], [1e-3, 1], color=DIAG_C, lw=DIAG_LW, zorder=-7)
        ax.set(xscale="log", yscale="log", xlim=(.1, 1.05), ylim=(.1, 1.05))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        return
    grid = 1.0 - np.logspace(0, np.log10(TAIL_EPS), 600)
    ax.plot(grid, grid, color=DIAG_C, lw=DIAG_LW, zorder=-7)
    xm = "logit" if scale == "logit" else "tail"
    ym = {"tail": "tail", "tail-x": "linear", "logit": "logit"}[scale]
    _prob_axis(ax, "x", xm, xlo, xhi, TAIL_TICKS, margin=.18)
    _prob_axis(ax, "y", ym, ylo, yhi, TAIL_YTICKS, margin=.18)


def draw_calibration_by_tier(ax, tier, data, ctxs, n_bins=10, style=None, scale=None):
    """One panel per tier, all context counts inside it.

    Tier is the panel, so colour is free to carry C. The markers are open, which is what
    lets all three context counts share a panel: where they coincide -- most of Experts --
    an unfilled marker shows the ones underneath it rather than hiding them. Shape still
    says what computed the belief, and every point is the same size.
    """
    style = style or CAL_TIER_STYLE
    scale = scale or CAL_TIER_SCALE
    cols = _cal_c_colours(ctxs)
    lim = [1., 0., 1., 0.]          # xlo, xhi, ylo, yhi over everything drawn
    for i, c in enumerate(ctxs):
        for name in CAL_TIER_PANELS[tier]:
            _, marker, _ = _cal_key(name)
            kind = CAL_AGENTS[name][1]
            _, _, conf, acc, _ = _series(data[c][CAL_AGENTS[name][3]], name)
            if style == "misfit":
                # the gap between the straight line and the data itself: empirical
                # accuracy in equal-mass confidence bins, no kernel in the loop
                bx, by = _calibration_curve(conf, acc, CAL_MISFIT_BINS, True)
                fx, fp, a0, b0, _ = _ols_calibration(conf, acc)
                lim = [min(lim[0], fx.min()), max(lim[1], fx.max()),
                       min(lim[2], min(by.min(), fp.min())),
                       max(lim[3], max(by.max(), fp.max()))]
                ax.fill_between(bx, by, a0 * bx + b0, color=cols[c],
                                alpha=CAL_ENV_ALPHA, lw=0, zorder=2)
                ax.plot(fx, fp, color=cols[c], ls=CAL_TIER_LS[kind], lw=CAL_FIT_LW,
                        alpha=CAL_FIT_ALPHA, zorder=3)
                ax.plot(bx, by, color=cols[c], ls="none", marker=marker, ms=3.2,
                        alpha=CAL_TIER_LINE_ALPHA, zorder=4 + i)
            if style in ("ols", "logistic") or CAL_TIER_FIT:
                model = "ols" if style == "ols" else (
                    "logistic" if style == "logistic" else CAL_FIT_MODEL)
                if model == "ols":
                    fx, fp, _, _, fse = _ols_calibration(conf, acc)
                else:
                    fx, fp, _, _ = _cox_calibration(conf, acc)
                    fse = None
                solo = style in ("ols", "logistic")
                if solo and CAL_FIT_BAND and fse is not None:
                    ax.fill_between(fx, fp - CAL_FIT_Z * fse, fp + CAL_FIT_Z * fse,
                                    color=cols[c], alpha=.30, lw=0, zorder=2)
                lim = [min(lim[0], fx.min()), max(lim[1], fx.max()),
                       min(lim[2], fp.min()), max(lim[3], fp.max())]
                ax.plot(fx, fp, color=cols[c], ls=CAL_TIER_LS[kind],
                        lw=CAL_TIER_LW if solo else CAL_FIT_LW,
                        alpha=CAL_TIER_LINE_ALPHA if solo else CAL_FIT_ALPHA,
                        zorder=(3 + i) if solo else 2)
            if style == "kernel":
                gx, gp, gse = _kernel_calibration(conf, acc)
                lim = [min(lim[0], gx.min()), max(lim[1], gx.max()),
                       min(lim[2], np.nanmin(gp)), max(lim[3], np.nanmax(gp))]
                if CAL_KR_BAND:
                    ax.fill_between(gx, gp - gse, gp + gse, color=cols[c], alpha=.30,
                                    lw=0, zorder=2)
                ax.plot(gx, gp, color=cols[c], ls=CAL_TIER_LS[kind], lw=CAL_TIER_LW,
                        alpha=CAL_TIER_LINE_ALPHA, solid_capstyle="round",
                        dash_capstyle="round", zorder=3 + i)
            elif style == "contour":
                _density_contour(ax, conf, acc, cols[c], CAL_TIER_LS[kind],
                                 CAL_TIER_COVER, CAL_TIER_LW)   # opaque: single lines
            elif style == "curve":
                x, y = _cover_curve(conf, acc, CAL_TIER_BINS, CAL_TIER_COVER)
                ax.plot(x, y, color=cols[c], ls=CAL_TIER_LS[kind], lw=CAL_TIER_LW,
                        alpha=CAL_TIER_LINE_ALPHA, solid_capstyle="round",
                        dash_capstyle="round", zorder=3 + i)
            elif style == "points":
                x, y = _calibration_curve(conf, acc, n_bins, True)
                ax.scatter(x, y, s=CAL_TIER_S, facecolor=cols[c], edgecolor="none",
                           alpha=CAL_TIER_ALPHA, marker=marker, zorder=3 + i)
            # no trailing else: a fit-only style must not also draw the point fallback
    _cal_tier_axes(ax, scale, lim)
    ax.set_xlabel("Confidence")
    _tidy(ax)


def _density_contour(ax, conf, acc, colour, ls, cover, lw):
    """Contour enclosing `cover` of the episodes, straight from the raw pairs.

    No binning of confidence: a 2D histogram fine enough to be a density, Gaussian
    smoothed, then the single level whose enclosed mass is `cover`.
    """
    from scipy.ndimage import gaussian_filter
    H, xe, ye = np.histogram2d(conf, acc, bins=CAL_CONTOUR_BINS,
                               range=[[0., 1.], [0., 1.]])
    H = gaussian_filter(H, CAL_CONTOUR_SMOOTH, mode="nearest")
    flat = np.sort(H.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    lvl = flat[min(np.searchsorted(cum, cover), flat.size - 1)]
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    ax.contour(xc, yc, H.T, levels=[lvl], colors=[colour], linestyles=[ls],
               linewidths=lw, zorder=3)


def _kernel_calibration(conf, acc, bw=None, cells=None, trim=None):
    """P(correct | confidence) by kernel regression -- no binning of the data.

    Every episode enters through the two sums a Nadaraya-Watson estimate needs. Those
    sums are accumulated on a fine grid and convolved with the kernel, which is
    algebraically the same estimator as weighting each episode individually but avoids an
    n x grid distance matrix for a quarter-million episodes.

    Returns (x, p, se) over the observed confidence range, with the ends trimmed per
    CAL_KR_TRIM. `se` uses the effective sample size (sum w)^2 / sum w^2, so it widens
    where the kernel has little to average over, and is drawn whichever trim is in use.
    """
    bw = CAL_KR_BW if bw is None else bw
    cells = CAL_KR_CELLS if cells is None else cells
    trim = CAL_KR_TRIM if trim is None else trim

    edges = np.linspace(0., 1., cells + 1)
    i = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, cells - 1)
    n = np.bincount(i, minlength=cells).astype(F64)
    hit = np.bincount(i, weights=acc.astype(F64), minlength=cells)

    sigma = bw * cells
    half = int(np.ceil(4 * sigma))
    k = np.exp(-.5 * (np.arange(-half, half + 1) / sigma) ** 2)   # unnormalised
    num = np.convolve(hit, k, mode="same")
    den = np.convolve(n, k, mode="same")
    den2 = np.convolve(n, k * k, mode="same")

    x = (edges[:-1] + edges[1:]) / 2
    ok = den > 0
    p = np.divide(num, den, out=np.full_like(den, np.nan), where=ok)
    n_eff = np.divide(den * den, den2, out=np.full_like(den, np.nan), where=den2 > 0)
    se = np.sqrt(np.clip(p * (1 - p), 0, None) / n_eff)

    # den > 0 reaches a kernel radius past the data, so clip to what was observed
    m = ok & (x >= conf.min()) & (x <= conf.max())
    if trim == "cover":
        m &= x >= np.quantile(conf, 1. - CAL_KR_COVER)
    elif trim == "se":
        # trim inward from each end, rather than masking pointwise, so the curve stays
        # one unbroken run even if se crosses back over the threshold somewhere inside
        good = m & np.isfinite(se) & (se <= CAL_KR_SE_MAX)
        if good.any():
            i0 = int(np.argmax(good))
            i1 = len(good) - 1 - int(np.argmax(good[::-1]))
            m = np.zeros_like(m)
            m[i0:i1 + 1] = True
    return x[m], p[m], se[m]


def _ols_calibration(conf, acc, grid=400):
    """(x, y, a, b, se) for accuracy = a * confidence + b, least squares on every episode.

    A linear probability model: the outcome is 0/1, so this is the least-squares straight
    line through P(correct | confidence). Two numbers, no bandwidth, nothing to trim --
    and on LINEAR axes it draws as the straight line it is. Under any probability
    transform, log-log included, the same line renders as a curve.

    `se` is the pointwise standard error of the FITTED LINE, with heteroskedasticity-
    robust (HC0) covariance -- which is the only honest choice here, since a 0/1 outcome
    has variance p(1-p) that changes along x rather than the constant OLS assumes.
    """
    X = np.c_[conf, np.ones_like(conf)]
    beta, *_ = np.linalg.lstsq(X, acc, rcond=None)
    resid = acc - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    V = xtx_inv @ (X.T @ (X * resid[:, None] ** 2)) @ xtx_inv    # HC0
    gx = np.linspace(conf.min(), conf.max(), grid)
    G = np.c_[gx, np.ones_like(gx)]
    se = np.sqrt(np.einsum("ij,jk,ik->i", G, V, G))
    return gx, G @ beta, float(beta[0]), float(beta[1]), se


def _cox_calibration(conf, acc, grid=400):
    """(x, p, a, b) for logit P(correct) = a + b logit(conf), fit on every episode.

    The textbook calibration model for a binary outcome: no bandwidth, no bins, no
    boundary bias, and a and b are directly interpretable against (0, 1). It buys that
    by imposing a shape, which is exactly what it cannot be trusted to report.
    """
    x = _prob_clip(conf)
    X = np.log(x / (1 - x)).reshape(-1, 1)
    m = LogisticRegression(penalty=None, max_iter=300).fit(X, acc)
    b, a0 = float(m.coef_[0, 0]), float(m.intercept_[0])
    gx = np.linspace(conf.min(), conf.max(), grid)
    g = _prob_clip(gx)
    return gx, 1. / (1. + np.exp(-(a0 + b * np.log(g / (1 - g))))), a0, b


def _cover_curve(conf, acc, n_bins, cover):
    """The calibration curve over the central `cover` of the confidence distribution.

    Trimming by quantile rather than by a fixed confidence range keeps the share of
    episodes behind each curve the same across agents, which a fixed range would not:
    the four agents' confidence distributions sit in quite different places.
    """
    lo, hi = np.quantile(conf, [(1 - cover) / 2, 1 - (1 - cover) / 2])
    m = (conf >= lo) & (conf <= hi)
    return _calibration_curve(conf[m], acc[m], n_bins, True)


def _cal_c_colours(ctxs):
    """The paper's C ramp, so a context count is the same colour in every figure."""
    return dict(zip(ctxs, plt.cm.viridis(np.linspace(0.15, 0.85, len(ctxs)))))


def plot_calibration_tier(fig, axes, harvests, n_bins=10, style=None, scale=None):
    """[Experts, Baselines, FR mediation] -- the tier-split alternative to row4.

    Both agents of a tier share a panel, so the comparison that matters (does the
    measured one track the computed one?) happens within a panel rather than across two.
    """
    style = style or CAL_TIER_STYLE
    ctxs = _cal_ctxs()
    data = {c: _load_ctx_calibration(c, harvests)[0] for c in ctxs}
    cols = _cal_c_colours(ctxs)

    for col, tier in enumerate(CAL_TIER_PANELS):
        ax = axes[col]
        draw_calibration_by_tier(ax, tier, data, ctxs, n_bins, style, scale)
        ax.text(*CAL_C_LABEL_XY, tier, transform=ax.transAxes, ha="left", va="top",
                fontsize=FS["title"] * .82)
        # the agents of this panel, named here rather than in one shared key
        if style in ("curve", "contour", "kernel", "logistic", "ols", "misfit"):
            handles = [Line2D([], [], color="0.35", lw=CAL_TIER_LW,
                              alpha=CAL_TIER_LINE_ALPHA,
                              ls=CAL_TIER_LS[CAL_AGENTS[n][1]], label=n)
                       for n in CAL_TIER_PANELS[tier]]
        else:
            handles = [Line2D([], [], ls="none", marker=_cal_key(n)[1],
                              markerfacecolor=(.35, .35, .35, CAL_TIER_ALPHA),
                              markeredgecolor="none",
                              markersize=np.sqrt(CAL_TIER_S) + 1.4, label=n)
                       for n in CAL_TIER_PANELS[tier]]
        # add_artist, or the C key below would replace this one on the same axes
        ax.add_artist(ax.legend(handles=handles, loc="upper left", frameon=False,
                                fontsize=FS["legend"] - 1, handletextpad=.4,
                                labelspacing=.3, borderpad=.2,
                                bbox_to_anchor=(.02, .88)))
        if col:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("Accuracy")
    # C key only once, in the corner the points leave empty in both panels
    c_handles = ([Line2D([], [], color=cols[c], lw=CAL_TIER_LW,
                         alpha=CAL_TIER_LINE_ALPHA, label=f"$C = {c}$")
                  for c in ctxs] if style in ("curve", "contour", "kernel", "logistic", "ols", "misfit") else
                 [Line2D([], [], ls="none", marker="o",
                         markerfacecolor=(*cols[c][:3], CAL_TIER_ALPHA),
                         markeredgecolor="none",
                         markersize=np.sqrt(CAL_TIER_S) + 1.4, label=f"$C = {c}$")
                  for c in ctxs])
    axes[0].legend(handles=c_handles,
                   loc="lower right", frameon=False, fontsize=FS["legend"] - 1,
                   handletextpad=.4, labelspacing=.45, borderpad=.2)
    cb, c_txt = draw_fr_mediation(fig, axes[2], data)

    fig.tight_layout(w_pad=CAL_WPAD, rect=(0, 0, 1, CAL_RECT_TOP))
    _pack_row(axes[:2], CAL_SCATTER_GAP, CAL_GROUP_GAP)
    fig.canvas.draw()
    box = [axes[i].get_position() for i in range(2)]
    mid = (min(b.x0 for b in box) + max(b.x1 for b in box)) / 2
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    _stretch_to(fig, (axes[2], cb.ax),
                min(axes[i].get_tightbbox(r).transformed(inv).y0 for i in range(2)))
    fig.canvas.draw()
    _box_c_column(fig, axes[2], c_txt)
    _cal_row_titles(fig, axes, 2, mid)
    return ctxs


def plot_dynamics_1x4(axes, d_tr, d_ec):
    """Norm, effective dimensionality, then the share of that norm held by the top and
    bottom equal-variance bands.

    Read left to right: the norm rises, the effective dimensionality rises with it, and
    the share held by the top PCs falls while the bottom band's rises. So the extra norm
    is not going into the already-strong directions -- it is spread across many weak ones.

    Share, not energy. The head band's absolute energy barely moves; it loses share because
    the weaker directions grow around it, and the trained network's norm rises while the
    echo state's falls.
    """
    tau = np.arange(DYN_PARAMS["E_START"], DYN_PARAMS["E_END"] + 1)
    past, future = tau <= 0, tau >= 0
    axes = np.asarray(axes).ravel()
    pct = 100 // DYN_PARAMS["N_BANDS"]

    #  key, title, formula, band tag, which end of the spectrum
    band_math = r"$\sum_{pc \,\in\, PC} \mathrm{proj}_{pc}^2 \,/\, \|u_t\|^2$"
    pr_math = r"$(\sum_{pc} \mathrm{proj}_{pc}^2)^2 / \sum_{pc} \mathrm{proj}_{pc}^4$"
    metrics = [("l2",   "Norm",                  r"$\|u_t\|_2$", None,   ""),
               ("pr",   "Effective Dim.",        pr_math,          None,   ""),
               ("head", "Proj. onto top PCs",    band_math,        "head", "top"),
               ("tail", "Proj. onto bottom PCs", band_math,        "tail", "bottom")]

    # one scale across the four, so the panels can be compared for effect size
    vals = [d[k] for k, *_ in metrics for d in (d_tr, d_ec)]
    lo, hi = min(v.min() for v in vals), max(v.max() for v in vals)
    pad = 0.09 * (hi - lo)

    for c, (key, name, math, band, end) in enumerate(metrics):
        ax = axes[c]
        for data, label in ((d_tr, "Trained"), (d_ec, "Echo")):
            mx, color = data[key], AGENT_COLORS[label]

            # Past lags (Dotted, smaller markers, white edges)
            ax.plot(tau[past], mx[past], c=color, alpha=0.5, lw=2.0, ls=':',
                    marker='o', mec='white', mew=1, ms=4, zorder=3)

            # Future lags (Solid, larger markers, black edges)
            ax.plot(tau[future], mx[future], c=color, alpha=1.0, lw=2, ls='-',
                    marker='o', mec='k', mew=1, ms=5, label=label if c == 0 else "",
                    zorder=4)

        # name on top, formula beneath it: the one-line form made the titles very wide
        ax.set_title(name, fontsize=FS["title"] * 0.85, pad=FS["title"] * 2.9)
        ax.text(.5, 1.015, math, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=FS["title"] * 0.70)
        if band is not None and f"{band}_pc" in d_tr:
            # band sizes as a small legend, coloured by network. Red title if the two
            # networks' bands are not actually variance-matched.
            gap = abs(d_tr[f"{band}_var"] - d_ec[f"{band}_var"]) > 0.03
            h = [Line2D([], [], color=AGENT_COLORS[l], lw=2.5, marker='o', mec='k',
                        mew=1.5, ms=7,
                        label=f"{d[f'{band}_pc']:.0f} PC{'s' if d[f'{band}_pc'] > 1 else ''}")
                 for d, l in ((d_tr, "Trained"), (d_ec, "Echo"))]
            leg = ax.legend(handles=h, loc='upper center', frameon=False, ncol=2,
                            fontsize=FS["annot"], handlelength=1.4, columnspacing=1.4,
                            title=f"{end} {pct}% of variance")
            leg.get_title().set_fontsize(FS["annot"])
            leg.get_title().set_color('#b03030' if gap else '0.15')

        ax.set_xlabel(r"Lag ($\tau$)")
        ax.set_ylabel("r (Cross-correlation)" if c == 0 else "")
        ax.set_xticks(range(DYN_PARAMS["E_START"], DYN_PARAMS["E_END"] + 1, 4))
        ax.set_ylim(lo - pad, hi + pad)
        if c:
            ax.tick_params(labelleft=False)     # shared scale, one tick column
        ax.axhline(0, color='black', alpha=0.3, lw=1.2, zorder=1)
        ax.axvline(0, color='black', ls='--', alpha=0.4, lw=1.5, zorder=1)
        ax.grid(True, linestyle='--', alpha=0.2, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].legend(frameon=False, loc='upper left')


# ═══════════════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cuda, realization_num, step_num, hid_dim = 0, 10, 30, 1000
    state_num, obs_num, batch_num, episodes = 500, 5, 10000, 50

    # gpu_inference stays off: prep_model_dynamics reads naive_px, which the GPU path skips.
    CFG = dict(mode="SANITY", cuda=cuda, episodes=episodes, checkpoint_every=1,
               realization_num=realization_num, hid_dim=hid_dim, obs_num=obs_num,
               show_plots=False, batch_num=batch_num, step_num=step_num,
               state_num=state_num, learn_embeddings=False, classifier_LR=.001,
               ctx_num=2, training=False)

    # 1. Prepare Data -- one agent resident at a time, then cached to HARVEST_CACHE
    trained, echo = collect_or_load(CFG, DYN_PARAMS)
    data_tr, data_ec = episode_xcorr(trained), episode_xcorr(echo)

    # 2. Figure A1: Diagnostics over all testing episodes
    figA1, axesA1 = plt.subplots(1, 4, figsize=PLOT_CFG["figsize_A"])
    plot_diagnostics_full(axesA1, trained, "Trained")
    plot_diagnostics_full(axesA1, echo, "Echo")
    finalize_layout_full(axesA1)

    # 2b. Figure A2: Diagnostics over early learning only
    figA2, axesA2 = plt.subplots(2, 3, figsize=PLOT_CFG["figsize_A_early"], squeeze=False)
    plot_diagnostics_early(axesA2, trained, "Trained")
    plot_diagnostics_early(axesA2, echo, "Echo")
    finalize_layout_early(axesA2)

    # 2c. Figure E: the one calibration figure -- tiers as panels, C as colour, and the
    # regret that explains the trend. CAL_TIER_STYLE picks what is drawn; the kernel,
    # curve, points, contour, logistic and misfit styles stay available there but are not
    # emitted. plot_calibration_row4 draws the by-context-count alternative (a panel per
    # C, with the boxed C column on its heatmap) and is likewise kept but not called --
    # see PLOT_CFG["figsize_D"] and width_ratios [1, 1, 1, 1.5] for how it was built.
    figE, axesE = plt.subplots(1, 3, figsize=PLOT_CFG["figsize_E"],
                               gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
    ctxs = plot_calibration_tier(figE, axesE, (trained, echo))

    # 3. Figure B: Event Dynamics

    figB, axesB = plt.subplots(1, 4, figsize=PLOT_CFG["figsize_B"])
    plot_dynamics_1x4(axesB, data_tr, data_ec)
    figB.tight_layout()

    figA1.savefig(fig_path("diagnostics_panel_full.svg"), format="svg", bbox_inches="tight")
    figA2.savefig(fig_path("diagnostics_panel_early.svg"), format="svg", bbox_inches="tight")
    figE.savefig(fig_path("calibration_by_tier.svg"), format="svg",
                 bbox_inches="tight")
    print(f"calibration: columns {ctxs}")
    figB.savefig(fig_path("dynamics_panel.svg"), format="svg", bbox_inches="tight")
    plt.show()
