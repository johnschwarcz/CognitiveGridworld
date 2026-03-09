"""
TRAINING CURVES — Network diagnostics across training episodes.

Consolidates training_analysis.py into a single script with no redundant code.
Loads four models (echo/trained × ctx_num=1/2) and produces:
  • make_core_figure   – 15-panel grid (ACC, DIFF, CORR, gradients, VNE, PR, …)
  • plot_variance_across_batches – belief variance decomposition by circular distance
"""

import numpy as np; import torch; import os; import sys; import inspect
import matplotlib.pyplot as plt; from scipy.optimize import curve_fit
from matplotlib.colors import PowerNorm
from tqdm import tqdm
from matplotlib.colors import Normalize; from matplotlib.cm import ScalarMappable

path = inspect.getfile(inspect.currentframe())
path = os.path.dirname(os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

AGENT_COLORS = {"Trained": "C0", "Echo": "C1", "Joint": "C2", "Naive": "C3"}
V_ALPHA = 0.5
NET_Z0, NET_Z1 = 2, 3
BAYES_Z0, BAYES_Z1 = 5, 6

# ═══════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════

def pr(evals, eps=1e-12):
    e = np.asarray(evals, float)
    return (e.sum(-1) ** 2) / ((e * e).sum(-1) + eps)

def sym_dkl_pair(P, Q, eps=1e-4):
    P = np.clip(P, eps, 1.0 - eps)
    Q = np.clip(Q, eps, 1.0 - eps)
    P = P / P.sum(-1, keepdims=True)
    Q = Q / Q.sum(-1, keepdims=True)
    dPQ = np.sum(P * (np.log(P) - np.log(Q)), axis=-1)
    dQP = np.sum(Q * (np.log(Q) - np.log(P)), axis=-1)
    return 0.5 * (dPQ + dQP)

def vne(evals, eps=1e-12, norm=True):
    e = np.maximum(np.asarray(evals, float), 0.0)
    p = e / (e.sum(-1, keepdims=True) + eps)
    H = -(p * np.log(p + eps)).sum(-1)
    if norm:
        H = H / (np.log(float(e.shape[-1])) + eps)
    return H

def _smooth(y, w):
    y = np.asarray(y, float)
    if w <= 1:
        return y
    k = np.ones(int(w), float) / float(w)
    p = int(w) // 2
    return np.convolve(np.pad(y, (p, p), mode="edge"), k, mode="valid")

def _ep_xy(y, episode_lim=None):
    y = np.asarray(y, float)
    n = y.size
    if n == 0:
        return np.zeros(0, int), y

    if episode_lim is None:
        idx = np.arange(n)
    elif np.isscalar(episode_lim):
        m = int(episode_lim)
        if m < 1:
            m = 1
        if m > n:
            m = n
        idx = np.arange(m)
    elif isinstance(episode_lim, slice):
        idx = np.arange(n)[episode_lim]
    else:
        t = tuple(episode_lim)
        if len(t) == 2:
            start, stop = t
            step = 1
        elif len(t) == 3:
            start, stop, step = t
        else:
            start, stop, step = 1, n, 1
        if start is None:
            start = 1
        if stop is None:
            stop = n
        if step is None:
            step = 1
        s = int(start) - 1
        e = int(stop)
        st = int(step)
        if s < 0:
            s = 0
        if e > n:
            e = n
        if st < 1:
            st = 1
        if e <= s:
            e = min(n, s + 1)
        idx = np.arange(s, e, st)

    if idx.size == 0:
        idx = np.arange(min(1, n))
    return idx + 1, y[idx]

def _plot_tr_echo(a, x0, y0, x1, y1, title, zline=False):
    a.plot(x0, y0, c=AGENT_COLORS["Trained"], lw=2.5, label="Trained")
    a.plot(x1, y1, c=AGENT_COLORS["Echo"], lw=2.5, label="Echo")
    if zline:
        a.axhline(0, c="k", ls="--")
    a.set_title(title)
    a.legend(frameon=False, fontsize=9)

# ═══════════════════════════════════════════════════════════════════
# Variance bundle
# ═══════════════════════════════════════════════════════════════════

def _compute_variance_bundle(trained, echo, D=4):
    models = (trained, echo)
    T = min(int(trained.step_num), int(echo.step_num))
    R = int(trained.realization_num)

    idx = np.arange(R)
    delta = np.abs(idx[:, None] - idx[None, :])
    circ_dist = np.minimum(delta, R - delta).astype(np.int64)

    W = np.zeros((R, D, R), dtype=np.float64)
    for r1 in range(R):
        for d in range(D):
            m = circ_dist[r1] == d
            c = m.sum()
            if c > 0:
                W[r1, d, m] = 1.0 / c

    io_bar = np.zeros((2, 2), dtype=np.float64)
    avg_bel = np.full((2, T), np.nan)
    avg_joint = np.full((2, T), np.nan)
    avg_naive = np.full((2, T), np.nan)
    dist_bel = np.full((2, T, D), np.nan)
    dist_joint = np.full((2, T, D), np.nan)
    dist_naive = np.full((2, T, D), np.nan)

    for i, m in enumerate(models):
        inp = np.full((R, R, T), np.nan)
        out = np.full((R, R, T), np.nan)
        bmu = np.full((R, R, T), np.nan)
        jmu = np.full((R, R, T), np.nan)
        nmu = np.full((R, R, T), np.nan)
        bd = np.full((R, R, T, D), np.nan)
        jd = np.full((R, R, T, D), np.nan)
        nd = np.full((R, R, T, D), np.nan)

        for r1 in m.realization_range:
            for r2 in m.realization_range:
                mask = (m.goal_ind == 0) & (m.ctx_vals[:, 0] == r1) & (m.ctx_vals[:, 1] == r2)
                if mask.sum() > 1:
                    xin = m.model_input_flat[mask, :T]
                    xout = m.model_update_flat[mask, :T]
                    b = m.model_goal_belief[mask, :T]
                    j = m.joint_goal_belief[mask, :T]
                    n = m.naive_goal_belief[mask, :T]

                    inp[r1, r2] = xin.var(0).mean(-1)
                    out[r1, r2] = xout.var(0).mean(-1)

                    vb = b.var(0)
                    vj = j.var(0)
                    vn = n.var(0)

                    bmu[r1, r2] = vb.mean(-1)
                    jmu[r1, r2] = vj.mean(-1)
                    nmu[r1, r2] = vn.mean(-1)

                    Wr = W[r1]
                    bd[r1, r2] = vb @ Wr.T
                    jd[r1, r2] = vj @ Wr.T
                    nd[r1, r2] = vn @ Wr.T

        io_bar[i, 0] = np.nanmean(inp)
        io_bar[i, 1] = np.nanmean(out)
        avg_bel[i] = np.nanmean(bmu, axis=(0, 1))
        avg_joint[i] = np.nanmean(jmu, axis=(0, 1))
        avg_naive[i] = np.nanmean(nmu, axis=(0, 1))
        dist_bel[i] = np.nanmean(bd, axis=(0, 1))
        dist_joint[i] = np.nanmean(jd, axis=(0, 1))
        dist_naive[i] = np.nanmean(nd, axis=(0, 1))

    xax = np.arange(T) + 1
    return xax, io_bar, avg_bel, avg_joint, avg_naive, dist_bel, dist_joint, dist_naive

# ═══════════════════════════════════════════════════════════════════
# Logit bundle
# ═══════════════════════════════════════════════════════════════════

def _compute_logit_bundle(trained, echo):
    T = min(int(trained.step_num), int(echo.step_num))
    xax = np.arange(T) + 1
    R = int(trained.realization_num)
    eps = 1e-9
    cfg = {
        "Trained": (trained, "model"),
        "Echo": (echo, "model"),
        "Joint": (trained, "joint"),
        "Naive": (trained, "naive")
    }
    curves = {}

    for key in cfg:
        m, attr = cfg[key]
        vals = np.full((R, R, T), np.nan)
        bel = getattr(m, f"{attr}_goal_belief")[:, :T]
        for r1 in m.realization_range:
            for r2 in m.realization_range:
                mask = (m.goal_ind == 0) & (m.ctx_vals[:, 0] == r1) & (m.ctx_vals[:, 1] == r2)
                if mask.sum() > 1:
                    p = np.clip(bel[mask, :, r1], eps, 1.0 - eps)
                    vals[r1, r2] = np.log(p / (1.0 - p)).mean(0)
        curves[key] = np.nanmean(vals, axis=(0, 1))

    fn = lambda t, a, b, c: a * np.log(t) - b * t + c
    try:
        popt, _ = curve_fit(fn, xax, curves["Naive"], p0=(1.0, 0.05, -2.0), maxfev=10000)
    except Exception:
        popt = (0.5, 0.05, -2.0)
    a, b = popt[0], popt[1]
    return xax, curves, a * np.log(xax), b * (xax - 1.0)

# ═══════════════════════════════════════════════════════════════════
# Core figure (15-panel grid)
# ═══════════════════════════════════════════════════════════════════

def make_core_figure(
    trained,
    echo,
    smooth_w=100,
    lim=None,
    D=4,
    figsize=(24, 12),
    episode_lim=None,
    layout=(
        ("ACC", "DIFF", "CORR", "GRADD", "ENTD"),
        ("ENTIO", "PR", "PRIO", "GRADO", "GRADI"),
        ("VARBAR", "DIST0", "LOGITPERF", "LOGITSCALE", "NONE")
    ),
    single_panel=None,
    single_panel_figsize=(5.4, 4.1),
    single_panel_svg_path=None
):
    def ex(y):
        return _ep_xy(_smooth(y, smooth_w), episode_lim)

    fig, ax = plt.subplots(len(layout), len(layout[0]), figsize=figsize, constrained_layout=False)

    T = min(int(trained.step_num), int(echo.step_num))
    B0 = trained.joint_goal_belief[:, :T]
    B1 = trained.model_goal_belief[:, :T]
    B2 = trained.naive_goal_belief[:, :T]
    B3 = echo.model_goal_belief[:, :T]

    vb = _compute_variance_bundle(trained, echo, D=D)
    io_bar = vb[1]

    x_logit, logit_curves, log_v, lin_v = _compute_logit_bundle(trained, echo)

    def p_ACC(a):
        x0, y0 = ex(trained.test_acc_through_training[:, -1])
        x1, y1 = ex(echo.test_acc_through_training[:, -1])
        _plot_tr_echo(a, x0, y0, x1, y1, "ACC (final step)")
        if hasattr(trained, "naive_acc"):
            a.axhline(np.asarray(trained.naive_acc)[:, -1].mean(), c=AGENT_COLORS["Naive"], ls="--", label="naive mean", zorder=BAYES_Z0)
        if hasattr(trained, "joint_acc"):
            a.axhline(np.asarray(trained.joint_acc)[:, -1].mean(), c=AGENT_COLORS["Joint"], ls="--", label="joint mean", zorder=BAYES_Z1)
        a.set_ylim(0, 0.8)
        a.legend(frameon=False, fontsize=9)

    def p_DIFF(a):
        x0, y0 = ex((trained.test_net_naive_DKL_through_training - trained.test_net_joint_DKL_through_training)[:, -1])
        x1, y1 = ex((echo.test_net_naive_DKL_through_training - echo.test_net_joint_DKL_through_training)[:, -1])
        _plot_tr_echo(a, x0, y0, x1, y1, "DKL(naive) - DKL(joint) (final step)", zline=True)

    def p_CORR(a):
        x0, y0 = ex(trained.test_SII_coef_through_training)
        x1, y1 = ex(echo.test_SII_coef_through_training)
        _plot_tr_echo(a, x0, y0, x1, y1, "CORR (SII coef)")

    def p_GRADD(a):
        x0, y0 = ex(trained.readout_grad_log_through_training - trained.readin_grad_log_through_training)
        x1, y1 = ex(echo.readout_grad_log_through_training - echo.readin_grad_log_through_training)
        _plot_tr_echo(a, x0, y0, x1, y1, "GRAD(|readout|) - GRAD(|readin|)", zline=True)

    def p_GRADO(a):
        x0, y0 = ex(trained.readout_grad_log_through_training)
        x1, y1 = ex(echo.readout_grad_log_through_training)
        _plot_tr_echo(a, x0, y0, x1, y1, "GRAD(|readout|)")

    def p_GRADI(a):
        x0, y0 = ex(trained.readin_grad_log_through_training)
        x1, y1 = ex(echo.readin_grad_log_through_training)
        _plot_tr_echo(a, x0, y0, x1, y1, "GRAD(|readin|)")

    def p_ENTD(a):
        x0, y0 = ex(vne(trained.test_model_update_dim_through_training) - vne(trained.test_model_input_dim_through_training))
        x1, y1 = ex(vne(echo.test_model_update_dim_through_training) - vne(echo.test_model_input_dim_through_training))
        a.plot(x0, y0, c=AGENT_COLORS["Trained"], lw=2.5, ls="-", alpha=1, label="Trained", zorder=NET_Z0)
        a.plot(x1, y1, c=AGENT_COLORS["Echo"], lw=2.5, ls="-", alpha=1, label="Echo", zorder=NET_Z1)
        a.axhline(0, c="k", ls="--")
        m = np.max(np.abs(np.concatenate((y0, y1))))
        if m > 0:
            a.set_ylim(-1.05 * m, 1.05 * m)
        a.set_title("VNE(out) - VNE(in)")
        a.legend(frameon=False, fontsize=9)

    def p_ENTIO(a):
        x0o, y0o = ex(vne(trained.test_model_update_dim_through_training))
        x0i, y0i = ex(vne(trained.test_model_input_dim_through_training))
        x1o, y1o = ex(vne(echo.test_model_update_dim_through_training))
        x1i, y1i = ex(vne(echo.test_model_input_dim_through_training))
        a.plot(x0o, y0o, c=AGENT_COLORS["Trained"], lw=2.5, ls="-", alpha=1, label="Tr out", zorder=NET_Z0)
        a.plot(x0i, y0i, c=AGENT_COLORS["Trained"], lw=2.5, ls="-", alpha=V_ALPHA, label="Tr in", zorder=NET_Z0)
        a.plot(x1o, y1o, c=AGENT_COLORS["Echo"], lw=2.5, ls="-", alpha=1, label="Echo out", zorder=NET_Z1)
        a.plot(x1i, y1i, c=AGENT_COLORS["Echo"], lw=2.5, ls="-", alpha=V_ALPHA, label="Echo in", zorder=NET_Z1)
        a.set_title("VNE(input/output)")
        a.legend(frameon=False, fontsize=9)

    def p_PR(a):
        x0, y0 = ex(pr(trained.test_model_update_dim_through_training) - (pr(trained.test_model_input_dim_through_training) + 1e-12))
        x1, y1 = ex(pr(echo.test_model_update_dim_through_training) - (pr(echo.test_model_input_dim_through_training) + 1e-12))
        a.plot(x0, y0, c=AGENT_COLORS["Trained"], lw=2.5, ls="-", alpha=1, label="Trained", zorder=NET_Z0)
        a.plot(x1, y1, c=AGENT_COLORS["Echo"], lw=2.5, ls="-", alpha=1, label="Echo", zorder=NET_Z1)
        a.axhline(0, c="k", ls="--")
        m = np.max(np.abs(np.concatenate((y0, y1))))
        if m > 0:
            a.set_ylim(-1.05 * m, 1.05 * m)
        a.set_title("PR(out) - PR(in)")
        a.legend(frameon=False, fontsize=9)

    def p_PRIO(a):
        x0o, y0o = ex(pr(trained.test_model_update_dim_through_training))
        x0i, y0i = ex(pr(trained.test_model_input_dim_through_training))
        x1o, y1o = ex(pr(echo.test_model_update_dim_through_training))
        x1i, y1i = ex(pr(echo.test_model_input_dim_through_training))
        a.plot(x0o, y0o, c=AGENT_COLORS["Trained"], lw=2.5, ls="-", alpha=1, label="Tr out", zorder=NET_Z0)
        a.plot(x0i, y0i, c=AGENT_COLORS["Trained"], lw=2.5, ls="-", alpha=V_ALPHA, label="Tr in", zorder=NET_Z0)
        a.plot(x1o, y1o, c=AGENT_COLORS["Echo"], lw=2.5, ls="-", alpha=1, label="Echo out", zorder=NET_Z1)
        a.plot(x1i, y1i, c=AGENT_COLORS["Echo"], lw=2.5, ls="-", alpha=V_ALPHA, label="Echo in", zorder=NET_Z1)
        a.set_title("PR(input/output)")
        a.legend(frameon=False, fontsize=9)

    def p_DIST0(a):
        TT = T - 1
        means = np.zeros((4, TT), float)
        for k in range(TT):
            t1 = k + 1
            means[0, k] = sym_dkl_pair(B0[:, 0], B0[:, t1]).mean()  # joint
            means[1, k] = sym_dkl_pair(B1[:, 0], B1[:, t1]).mean()  # trained
            means[2, k] = sym_dkl_pair(B2[:, 0], B2[:, t1]).mean()  # naive
            means[3, k] = sym_dkl_pair(B3[:, 0], B3[:, t1]).mean()  # echo
        x = np.arange(TT) + 1
        a.plot(x, means[1], "-o", ms=2.2, lw=1.8, c=AGENT_COLORS["Trained"], label="trained", zorder=NET_Z0)
        a.plot(x, means[3], "-o", ms=2.2, lw=1.8, c=AGENT_COLORS["Echo"], label="echo", zorder=NET_Z1)
        a.plot(x, means[0], "--o", ms=2.2, lw=1.8, c=AGENT_COLORS["Joint"], label="joint", zorder=BAYES_Z0)
        a.plot(x, means[2], "--o", ms=2.2, lw=1.8, c=AGENT_COLORS["Naive"], label="naive", zorder=BAYES_Z1)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("t")
        a.set_ylabel("symDKL")
        a.set_title("DKL(B_0, B_t)")
        a.legend(frameon=False, fontsize=9)

    def p_VARBAR(a):
        vals = np.array((io_bar[0, 0], io_bar[0, 1], io_bar[1, 0], io_bar[1, 1]), float)
        x = np.arange(4)
        a.bar(
            x,
            vals,
            color=(AGENT_COLORS["Trained"], AGENT_COLORS["Trained"], AGENT_COLORS["Echo"], AGENT_COLORS["Echo"]),
            alpha=0.82,
            edgecolor="k"
        )
        a.set_xticks(x)
        a.set_xticklabels(("Tr\nIn", "Tr\nOut", "Echo\nIn", "Echo\nOut"))
        a.set_ylabel("variance")
        a.set_title("Input/Output variance\n(across batches; avg over t, features)")

    def p_LOGITPERF(a):
        a.plot(x_logit, logit_curves["Trained"], c=AGENT_COLORS["Trained"], lw=2.5, label="Trained", zorder=NET_Z0)
        a.plot(x_logit, logit_curves["Echo"], c=AGENT_COLORS["Echo"], lw=2.5, label="Echo", zorder=NET_Z1)
        a.plot(x_logit, logit_curves["Joint"], c=AGENT_COLORS["Joint"], lw=2.5, ls="--", label="Joint", zorder=BAYES_Z0)
        a.plot(x_logit, logit_curves["Naive"], c=AGENT_COLORS["Naive"], lw=2.5, ls="--", label="Naive", zorder=BAYES_Z1)
        a.set_title("True Positive Logit")
        a.set_xlabel("t")
        a.set_ylabel(r"$P(B_{tga^{\star}})$")
        a.grid(alpha=0.25)
        a.legend(frameon=False, loc=2, fontsize=9)

    def p_LOGITSCALE(a):
        a.fill_between(x_logit, -log_v, log_v, color=AGENT_COLORS["Joint"], alpha=0.15, label=r"Mutual Info envelope: $\pm\ln t$")
        a.plot(x_logit, log_v, ":", c=AGENT_COLORS["Joint"], lw=1.1)
        a.plot(x_logit, -log_v, ":", c=AGENT_COLORS["Joint"], lw=1.1)
        a.fill_between(x_logit, -lin_v, lin_v, color=AGENT_COLORS["Naive"], alpha=0.15, label=r"Total Correlation envelope: $\pm t$")
        a.plot(x_logit, lin_v, "--", c=AGENT_COLORS["Naive"], lw=1.1)
        a.plot(x_logit, -lin_v, "--", c=AGENT_COLORS["Naive"], lw=1.1)
        a.set_title("Contributions scale differently")
        a.set_xlabel("t")
        a.set_ylabel("Relative Magnitude")
        a.grid(alpha=0.25)
        a.legend(frameon=False, loc=2, fontsize=9)

    panels = {
        "ACC": p_ACC, "DIFF": p_DIFF, "CORR": p_CORR, "GRADD": p_GRADD, "GRADO": p_GRADO, "GRADI": p_GRADI,
        "ENTD": p_ENTD, "ENTIO": p_ENTIO, "PR": p_PR, "PRIO": p_PRIO, "DIST0": p_DIST0, "VARBAR": p_VARBAR,
        "LOGITPERF": p_LOGITPERF, "LOGITSCALE": p_LOGITSCALE
    }

    no_lim = ("DIST0", "VARBAR", "LOGITPERF", "LOGITSCALE")
    nr = len(layout)
    nc = len(layout[0])

    for r in range(nr):
        for c in range(nc):
            key = layout[r][c]
            a = ax[r, c]
            if key == "NONE" or key not in panels:
                a.axis("off")
            else:
                panels[key](a)
                if lim is not None and key not in no_lim:
                    a.set_xlim(1, lim)

    fig.suptitle("Core diagnostics (duplicates removed; heatmaps removed)", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=1.0, w_pad=0.8)

    sfig = None
    sax = None
    if (single_panel is not None) and (single_panel in panels):
        sfig, sax = plt.subplots(1, 1, figsize=single_panel_figsize, constrained_layout=False)
        panels[single_panel](sax)
        if lim is not None and single_panel not in no_lim:
            sax.set_xlim(1, lim)
        sfig.suptitle(single_panel, y=0.995, fontsize=11)
        sfig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        if single_panel_svg_path is not None:
            sfig.savefig(single_panel_svg_path, format="svg", bbox_inches="tight")

    return fig, ax, vb, sfig, sax

# ═══════════════════════════════════════════════════════════════════
# Variance across batches figure
# ═══════════════════════════════════════════════════════════════════

def plot_variance_across_batches(trained, echo, D=4, bundle=None):
    if bundle is None:
        bundle = _compute_variance_bundle(trained, echo, D=D)

    xax = bundle[0]
    avg_bel = bundle[2]
    avg_joint = bundle[3]
    avg_naive = bundle[4]
    dist_bel = bundle[5]
    dist_joint = bundle[6]
    dist_naive = bundle[7]
    eps = 1e-12

    fig, axes = plt.subplots(1, 1 + D, figsize=(3.8 * (1 + D), 3.8), constrained_layout=False)

    a0 = axes[0]
    a0.plot(xax, np.clip(avg_bel[0], eps, np.inf), c=AGENT_COLORS["Trained"], lw=2.5, label="Trained RNN", zorder=NET_Z0)
    a0.plot(xax, np.clip(avg_bel[1], eps, np.inf), c=AGENT_COLORS["Echo"], lw=2.5, label="Echo State", zorder=NET_Z1)
    a0.plot(xax, np.clip(avg_joint[0], eps, np.inf), c=AGENT_COLORS["Joint"], ls="--", lw=1.7, label="joint", zorder=BAYES_Z0)
    a0.plot(xax, np.clip(avg_naive[0], eps, np.inf), c=AGENT_COLORS["Naive"], ls="--", lw=1.7, label="naive", zorder=BAYES_Z1)
    a0.set_xscale("log")
    a0.set_yscale("log")
    a0.grid(True, which="both", alpha=0.25)
    a0.set_title(r"$B_r$ variance (avg over r1,r2,state)")
    a0.set_xlabel("t")
    a0.set_ylabel("variance across batches")
    a0.legend(frameon=False, fontsize=9)

    for d in range(D):
        ad = axes[d + 1]
        ad.plot(xax, np.clip(dist_bel[0, :, d], eps, np.inf), c=AGENT_COLORS["Trained"], lw=2.5, label="Trained RNN", zorder=NET_Z0)
        ad.plot(xax, np.clip(dist_bel[1, :, d], eps, np.inf), c=AGENT_COLORS["Echo"], lw=2.5, label="Echo State", zorder=NET_Z1)
        ad.plot(xax, np.clip(dist_joint[0, :, d], eps, np.inf), c=AGENT_COLORS["Joint"], ls="--", lw=1.6, label="joint", zorder=BAYES_Z0)
        ad.plot(xax, np.clip(dist_naive[0, :, d], eps, np.inf), c=AGENT_COLORS["Naive"], ls="--", lw=1.6, label="naive", zorder=BAYES_Z1)
        ad.set_xscale("log")
        ad.set_yscale("log")
        ad.grid(True, which="both", alpha=0.25)
        ad.set_title(f"Distance = {d}")
        ad.set_xlabel("t")
        if d > 0:
            ad.sharey(axes[1])
            plt.setp(ad.get_yticklabels(), visible=False)

    fig.suptitle(
        "Belief variance across batches/trials within fixed (r1,r2), then averaged\nLeft: overall average. Right: grouped by circular distance from r1.",
        y=0.995,
        fontsize=12
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), h_pad=0.8, w_pad=0.8)
    return fig, axes

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cuda = 0
    realization_num = 10
    step_num = 30
    hid_dim = 1000
    state_num = 500
    obs_num = 5
    training = False
    batch_num = 8000
    episodes = 50000 if training else 1

    echo = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 2,
        'training': training,
        'save_env': "/sanity/reservoir_ctx_2_e5" if training else None,
        'load_env': None if training else "/sanity/reservoir_ctx_2_e5"})
    if training:
        del(echo)

    trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .001, 'ctx_num': 2,
        'training': training,
        'save_env': "/sanity/fully_trained_ctx_2_e5" if training else None,
        'load_env': None if training else "/sanity/fully_trained_ctx_2_e5"})
    if training:
        del(trained)

    echo_1 = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 1,
        'training': training,
        'save_env': "/sanity/reservoir_ctx_1_e5" if training else None,
        'load_env': None if training else "/sanity/reservoir_ctx_1_e5"})
    if training:
        del(echo_1)

    trained_1 = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num, 'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .001, 'ctx_num': 1,
        'training': training,
        'save_env': "/sanity/fully_trained_ctx_1_e5" if training else None,
        'load_env': None if training else "/sanity/fully_trained_ctx_1_e5"})
    if training:
        del(trained_1)

    if not training:
        core_fig, core_ax, vbundle, single_fig, single_ax = make_core_figure(
            trained,
            echo,
            smooth_w=100,
            lim=None,
            D=4,
            episode_lim=None,
            single_panel=None,
            single_panel_svg_path=None
        )
        plt.show()

        var_fig, var_ax = plot_variance_across_batches(trained, echo, D=4, bundle=vbundle)
        plt.show()
