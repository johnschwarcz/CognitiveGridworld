import numpy as np
import torch
import os
import sys
import inspect
import warnings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

path = inspect.getfile(inspect.currentframe())
path = os.path.dirname(os.path.abspath(path))
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld
from main.utils import fig_path

F64, I64 = np.float64, np.int64

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

PLOT_CFG = {
    "figsize_A": (17, 6),        # 1 x 3 Diagnostics, all testing episodes
    "figsize_A_early": (28, 6),  # 1 x 5 Diagnostics, early learning
    "figsize_B": (28, 4),   # 1 x 5 Event-Triggered Dynamics
    "figsize_C": (23, 4.8),      # 1 x 4 Calibration
    "early_epochs": 400,
    "smooth_w": 1,
    "line_width": 2.5,
    "title_fs": 18,
    "math_fs": 17,          # Subtitle font size
    "label_fs": 14,
    "tick_fs": 12,
}

DYN_PARAMS = {
    "K_PCA": None,
    "E_START": -8, 
    "E_END": 8,
    "T_START": 20,
    "T_END": 30
}

AGENT_COLORS = {"Trained": "#1f77b4", "Echo": "#ff7f0e", "Joint": "#2ca02c", "Naive": "#d62728"}

plt.rcParams.update({
    'font.size': PLOT_CFG["label_fs"],
    'axes.labelsize': PLOT_CFG["label_fs"],
    'axes.titlesize': PLOT_CFG["title_fs"],
    'axes.titleweight': 'bold',
    'legend.fontsize': 12,
    'xtick.labelsize': PLOT_CFG["tick_fs"],
    'ytick.labelsize': PLOT_CFG["tick_fs"],
    'axes.linewidth': 1.5,
    'lines.linewidth': PLOT_CFG["line_width"],
    'figure.dpi': 300, 
    'font.family': 'sans-serif',
})

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

def pca_upd(m, k_pca=None):
    """
    Computes global covariance across time and batch: Neuron x (Time * Batch),
    and projects the centered representations onto the principal axes.
    """
    upd = m.model_update_flat.astype(F64)
    B, T, N = upd.shape
    x = np.nan_to_num(upd.reshape(B * T, N).copy(), 0., 0., 0.)
    x -= x.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    k = k_pca if k_pca is not None else vt.shape[0]
    return (x @ vt[:k].T).reshape(B, T, k).astype(F64)

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
    
    z = pca_upd(m, k_pca=prm.get("K_PCA", None))
    rev = step_dkl(m.naive_px / m.naive_px.sum(-1, keepdims=True), 
                   approx_lik(m.joint_belief.astype(F64))).mean(-1)
    
    return {
        "l2": l2[:, ts:te],
        "spar": step_sparsity(u[:, ts:te]),
        "pr": step_participation_ratio(z[:, ts:te]), 
        "dist": dist[:, ts:te],
        "v_prev": v_prev[:, ts:te],
        "rev": rev[:, ts:te]
    }

# ═══════════════════════════════════════════════════════════════════
# Plotting Functions
# ═══════════════════════════════════════════════════════════════════

def plot_diagnostics_full(axes, agent, name):
    """All testing episodes: accuracy, FR correlation, and the two against each other."""
    def ex(y, lim=None): return _ep_xy(_smooth(y, PLOT_CFG["smooth_w"]), lim)

    # 1. Final Step Accuracy
    x, y = ex(agent.test_acc_through_training[:, -1])
    axes[0].plot(x, y, c=AGENT_COLORS[name], label=name, zorder=3)
    baseline_color = AGENT_COLORS["Joint"] if name == "Trained" else AGENT_COLORS["Naive"]
    baseline_label = "Joint" if name == "Trained" else "Naive"
    acc = float(np.mean(agent.joint_acc[:, -1])) if name == "Trained" else float(np.mean(agent.naive_acc[:, -1]))
    axes[0].axhline(acc, c=baseline_color, ls="--", label=baseline_label, zorder=5)

    # 2. Correlation(FR, Acc.)
    x, y = ex(agent.test_SII_coef_through_training)
    axes[1].plot(x, y, c=AGENT_COLORS[name], zorder=3)
    axes[1].axhline(y=0, c='k', alpha=0.5)

    # 3. FR Correlation vs. Accuracy
    x_acc = _smooth(agent.test_acc_through_training[:, -1], PLOT_CFG["smooth_w"])
    y_sii = _smooth(agent.test_SII_coef_through_training, PLOT_CFG["smooth_w"])
    n = min(len(x_acc), len(y_sii))
    axes[2].plot(x_acc[:n], y_sii[:n], c=AGENT_COLORS[name], zorder=3)
    axes[2].axhline(y=0, c='k', alpha=0.5)


def plot_diagnostics_early(axes, agent, name):
    """First PLOT_CFG['early_epochs'] testing episodes, where the FR correlation emerges."""
    lim = PLOT_CFG["early_epochs"]
    def ex(y): return _ep_xy(_smooth(y, PLOT_CFG["smooth_w"]), lim)

    # 1. Early Learning Acc.
    x, y = ex(agent.test_acc_through_training[:, -1])
    axes[0].plot(x, y, c=AGENT_COLORS[name], label=name, zorder=3)

    # 2. Correlation(FR, Acc.)
    x, y = ex(agent.test_SII_coef_through_training)
    axes[1].plot(x, y, c=AGENT_COLORS[name], zorder=3)
    axes[1].axhline(y=0, c='k', alpha=0.5)

    # 3. PR(RNN)
    pr_rnn = pr_stat(agent.test_model_update_dim_through_training)
    x, y = ex(pr_rnn)
    axes[2].plot(x, y, c=AGENT_COLORS[name], zorder=3)

    # 4. PR(Read-in)
    pr_in = pr_stat(agent.test_model_input_dim_through_training)
    x, y = ex(pr_in)
    axes[3].plot(x, y, c=AGENT_COLORS[name], zorder=3)

    # 5. PR(RNN) - PR(Read-in)
    x, y = ex(pr_rnn - (pr_in + 1e-12))
    axes[4].plot(x, y, c=AGENT_COLORS[name], zorder=3)
    axes[4].axhline(y=0, c='k', alpha=0.5)


def _finalize_1x3(axes, titles, ylabels, xlabels):
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], pad=15)
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel(xlabels[i])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, ls='--', alpha=0.3)
    axes[0].legend(loc='lower right', frameon=False, ncols=2)
    plt.tight_layout()


def finalize_layout_full(axes):
    _finalize_1x3(axes,
        ["Final Step Accuracy", "Correlation(FR, Acc.)", "FR Correlation vs. Acc."],
        ["Accuracy", "r", "r (Correlation)"],
        ["Testing Episode", "Testing Episode", "Accuracy"])


def finalize_layout_early(axes):
    _finalize_1x3(axes,
        ["Early Learning Acc.", "Correlation(FR, Acc.)", "PR(RNN)", "PR(Read-in)",
         "PR(RNN) - PR(Read-in)"],
        ["Accuracy", "r", "PR", "PR", r"$\Delta$ PR"],
        ["Testing Episode"] * 5)

    # PR(RNN) and PR(Read-in) are the same quantity in the same units: share a scale
    lo = min(axes[2].get_ylim()[0], axes[3].get_ylim()[0])
    hi = max(axes[2].get_ylim()[1], axes[3].get_ylim()[1])
    axes[2].set_ylim(lo, hi)
    axes[3].set_ylim(lo, hi)


# ═══════════════════════════════════════════════════════════════════
# Calibration: is confidence a usable signal, or decoupled from accuracy?
# Adapted from coggrid.plotting.plots (_draw_calibration / _draw_confidence_scissor)
# ═══════════════════════════════════════════════════════════════════

def _final_conf_acc(goal_belief, accuracy):
    """Confidence in the answer the observer would give, and whether it was right."""
    b = np.asarray(goal_belief, float)
    return b[:, -1].max(-1), np.asarray(accuracy, float)[:, -1]


def _calibration_curve(confidence, correct, n_bins, min_count=20):
    """Empirical accuracy per confidence bin, plus each bin's share of episodes.

    Bins under `min_count` are dropped: their accuracy is mostly sampling noise
    and they land in the tails, where the eye discounts it least.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges) - 1, 0, n_bins - 1)
    centres, accuracy, weight = [], [], []
    for b in range(n_bins):
        in_bin = idx == b
        if in_bin.sum() < min_count:
            continue
        centres.append(confidence[in_bin].mean())
        accuracy.append(correct[in_bin].mean())
        weight.append(in_bin.mean())
    return np.array(centres), np.array(accuracy), np.array(weight)


def draw_calibration(ax, series, chance, n_bins=20):
    """Accuracy against self-reported confidence. `series`: (name, conf, correct)."""
    ax.plot([0, 1], [0, 1], color="0.7", lw=1.2, ls="--", zorder=1,
            label="perfect calibration")
    ax.axhline(chance, color="0.7", lw=2, ls=":", zorder=1)
    ax.text(0.02, chance + 0.02, "chance", color="#999999", fontsize=10)

    for name, conf_all, corr_all in series:
        conf, acc, weight = _calibration_curve(conf_all, corr_all, n_bins)
        colour = AGENT_COLORS[name]
        ax.plot(conf, acc, "-", color=colour, lw=1.6, zorder=3, label=name)
        # Marker area carries the episode count behind each point.
        ax.scatter(conf, acc, s=np.minimum(12 + 18 * n_bins * weight, 110),
                   color=colour, edgecolor="white", lw=0.6, zorder=4)

    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, fontsize=9, loc="upper left")


def draw_confidence_scissor(ax, series, chance, n_bins=20, shade=1, annotate=True):
    """Accuracy and confidence against episode FR rank. `series`: (name, conf, correct, fr).

    Rank rather than raw FR on x: the distribution has a long tail, and what
    matters is the ordering of episodes, not the units.
    """
    ax.axhline(chance, color="0.7", lw=2, ls=":", zorder=1)
    ax.text(98, chance + 0.025, "chance", color="#999999", fontsize=10, ha="right")
    x = np.linspace(100 / n_bins, 100, n_bins) - 50 / n_bins
    marker = "-o" if n_bins <= 25 else "-"

    curves = {}
    for name, conf, corr, fr in series:
        bins = np.array_split(np.argsort(np.asarray(fr, float)), n_bins)
        colour = AGENT_COLORS[name]
        acc_b = np.array([np.asarray(corr)[b].mean() for b in bins])
        conf_b = np.array([np.asarray(conf)[b].mean() for b in bins])
        curves[name] = (acc_b, conf_b)
        ax.plot(x, acc_b, marker, color=colour, lw=1.9, ms=4.0, zorder=3,
                label=f"{name} accuracy")
        ax.plot(x, conf_b, "--", color=colour, lw=1.4, alpha=0.8, zorder=2,
                label=f"{name} confidence")

    # Shade the gap for the factorized observer: that gap is the failure mode.
    name = series[shade][0]
    acc_b, conf_b = curves[name]
    colour = AGENT_COLORS[name]
    ax.fill_between(x, acc_b, conf_b, color=colour, alpha=0.12, zorder=0)
    if annotate:
        mid = max(0, n_bins - 2)
        ax.annotate("believes it is right\nthis often",
                    xy=(x[mid], conf_b[mid]), xytext=(x[mid] - 30, 0.93),
                    fontsize=8, color=colour, ha="center",
                    arrowprops=dict(arrowstyle="->", color=colour, lw=0.9))
        ax.annotate("actually is, this often",
                    xy=(x[mid], acc_b[mid]), xytext=(x[mid] - 50, 0.30),
                    fontsize=8, color=colour, ha="center",
                    arrowprops=dict(arrowstyle="->", color=colour, lw=0.9))

    ax.set(ylim=(0, 1), xlim=(0, 100))
    ax.set_xlabel("Episodes ranked by FR (percentile)")
    ax.set_ylabel("Probability")
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2,
              columnspacing=1.0, handlelength=1.6)


def plot_calibration_1x4(axes, trained, echo, n_bins=20):
    """[calibration Bayes, calibration networks, scissor Bayes, scissor networks].

    Bayesian observers are read off `trained`'s episodes; each network is read off
    its own agent's episodes. The scissor's x axis is a within-run percentile, so
    the two network curves remain comparable despite different episode draws.
    """
    chance = 1.0 / trained.realization_num
    j = _final_conf_acc(trained.joint_goal_belief, trained.joint_acc)
    n = _final_conf_acc(trained.naive_goal_belief, trained.naive_acc)
    t = _final_conf_acc(trained.model_goal_belief, trained.model_acc)
    e = _final_conf_acc(echo.model_goal_belief, echo.model_acc)
    fr_b, fr_t, fr_e = trained.SII[:, -1], trained.SII[:, -1], echo.SII[:, -1]

    draw_calibration(axes[0], [("Joint", *j), ("Naive", *n)], chance, n_bins)
    draw_calibration(axes[1], [("Trained", *t), ("Echo", *e)], chance, n_bins)
    draw_confidence_scissor(axes[2], [("Joint", *j, fr_b), ("Naive", *n, fr_b)],
                            chance, n_bins, shade=1)
    draw_confidence_scissor(axes[3], [("Trained", *t, fr_t), ("Echo", *e, fr_e)],
                            chance, n_bins, shade=1)

    for ax, title in zip(axes, ["Calibration: Bayes", "Calibration: Networks",
                                "Confidence vs FR: Bayes", "Confidence vs FR: Networks"]):
        ax.set_title(title, pad=15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()


def plot_dynamics_1x5(axes, d_tr, d_ec):
    tau = np.arange(DYN_PARAMS["E_START"], DYN_PARAMS["E_END"] + 1)
    past, future = tau <= 0, tau >= 0
    
    metrics = [
        ("l2", "Norm (L2)", r"$\|u_t\|_2$", 2),
        ("spar", "Sparsity", r"$\frac{\sqrt{N} - (\|u_t\|_1 / \|u_t\|_2)}{\sqrt{N}-1}$", 3),
        ("pr", "PR(proj(RNN))", r"$\frac{(\sum z_i^2)^2}{\sum z_i^4}$", 4),
        ("dist", "Distance", r"$\|u_t - u_{t-1}\|$", 1),
        ("v_prev", "Observation Change", r"$(o^{i}_{t} - o^{i}_{t-1})^2$", 0)
    ]
    
    for key, name, math, c in metrics:
        ax = axes[c]
        for data, label in [(d_tr, "Trained"), (d_ec, "Echo")]:
            color = AGENT_COLORS[label]
            mx = get_xcorr(data[key], data["rev"], DYN_PARAMS["E_START"], DYN_PARAMS["E_END"])
            
            # Past lags (Dotted, smaller markers, white edges)
            ax.plot(tau[past], mx[past], c=color, alpha=0.5, lw=2.0, ls=':', 
                    marker='o', mec='white', mew=1, ms=4, zorder=3)
            
            # Future lags (Solid, larger markers, black edges)
            ax.plot(tau[future], mx[future], c=color, alpha=1.0, lw=2.5, ls='-', 
                    marker='o', mec='k', mew=1.5, ms=7, label=label if c == 0 else "", zorder=4)
        
        ax.set_title(f"{name}\n", pad=25)
        ax.text(0.5, 1.05, math, transform=ax.transAxes, ha="center", 
                fontsize=PLOT_CFG["math_fs"], style='italic', alpha=0.85)
        
        ax.set_xlabel(r"Lag ($\tau$)")
        ax.set_ylabel("r" if c == 0 else "")
        ax.axhline(0, color='black', alpha=0.3, lw=1.2, zorder=1)
        ax.axvline(0, color='black', ls='--', alpha=0.4, lw=1.5, zorder=1)
        ax.grid(True, linestyle='--', alpha=0.2, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        
    axes[0].legend(frameon=False, loc='upper right')

# ═══════════════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cuda, realization_num, step_num, hid_dim = 0, 10, 30, 1000
    state_num, obs_num, batch_num = 500, 5, 10000

    trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1, 'checkpoint_every': 5, 
                                    'realization_num': realization_num, 'hid_dim': hid_dim, 'obs_num': obs_num, 
                                    'show_plots': False, 'batch_num': batch_num, 'step_num': step_num, 
                                    'state_num': state_num, 'learn_embeddings': False, 'reservoir': False, 
                                    'classifier_LR': .001, 'ctx_num': 2, 'training': False, 
                                    'load_env': "/sanity/fully_trained_ctx_2_e5"})       
    echo = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1, 'checkpoint_every': 5, 
                                 'realization_num': realization_num, 'hid_dim': hid_dim, 'obs_num': obs_num, 
                                 'show_plots': False, 'batch_num': batch_num, 'step_num': step_num, 
                                 'state_num': state_num, 'learn_embeddings': False, 'reservoir': True, 
                                 'classifier_LR': .001, 'ctx_num': 2, 'training': False, 
                                 'load_env': "/sanity/reservoir_ctx_2_e5"})    

    # 1. Prepare Data
    data_tr = prep_model_dynamics(trained, DYN_PARAMS)
    data_ec = prep_model_dynamics(echo, DYN_PARAMS)

    # 2. Figure A1: Diagnostics over all testing episodes
    figA1, axesA1 = plt.subplots(1, 3, figsize=PLOT_CFG["figsize_A"])
    plot_diagnostics_full(axesA1, trained, "Trained")
    plot_diagnostics_full(axesA1, echo, "Echo")
    finalize_layout_full(axesA1)
    figA1.savefig(fig_path("diagnostics_panel_full.svg"), format="svg", bbox_inches="tight")

    # 2b. Figure A2: Diagnostics over early learning only
    figA2, axesA2 = plt.subplots(1, 5, figsize=PLOT_CFG["figsize_A_early"])
    plot_diagnostics_early(axesA2, trained, "Trained")
    plot_diagnostics_early(axesA2, echo, "Echo")
    finalize_layout_early(axesA2)
    figA2.savefig(fig_path("diagnostics_panel_early.svg"), format="svg", bbox_inches="tight")

    # 2c. Figure C: Calibration
    figC, axesC = plt.subplots(1, 4, figsize=PLOT_CFG["figsize_C"])
    plot_calibration_1x4(axesC, trained, echo)
    figC.savefig(fig_path("calibration_panel.svg"), format="svg", bbox_inches="tight")

    # 3. Figure B: Event Dynamics
    figB, axesB = plt.subplots(1, 5, figsize=PLOT_CFG["figsize_B"])
    plt.subplots_adjust(bottom=0.2, wspace=0.4)
    plot_dynamics_1x5(axesB, data_tr, data_ec)
    figB.savefig(fig_path("dynamics_panel.svg"), format="svg", bbox_inches="tight")
    plt.show()