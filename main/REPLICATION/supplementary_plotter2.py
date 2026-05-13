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

F64, I64 = np.float64, np.int64

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

PLOT_CFG = {
    "figsize_A": (22, 6),   # 1 x 4 Diagnostics
    "figsize_B": (28, 4),   # 1 x 5 Event-Triggered Dynamics
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
    upd = m.model_update_flat.astype(F64)
    B, T, N = upd.shape
    x = np.nan_to_num(upd.reshape(B * T, N).copy(), 0., 0., 0.)
    x -= x.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(x, False)
    k = k_pca if k_pca is not None else vt.shape[0]
    return (x @ vt[:k].T).reshape(B, T, k).astype(F64)

def step_participation_ratio(x, eps=1e-99):
    x_sq = x ** 2
    return (np.sum(x_sq, axis=-1) ** 2) / np.maximum(np.sum(x_sq ** 2, axis=-1), eps)

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
    
    # Metrics
    l2 = np.linalg.norm(u, ord=2, axis=-1)
    dist = np.zeros((B, T))
    dist[:, 1:] = np.linalg.norm(u[:, 1:] - u[:, :-1], axis=-1)
    
    obs = m.obs_flat.astype(F64)
    v_prev = np.zeros((B, T))
    v_prev[:, 1:] = np.mean((obs[:, 1:] - obs[:, :-1])**2, -1)
    
    z = pca_upd(m)
    rev = step_dkl(m.naive_px / m.naive_px.sum(-1, keepdims=True), 
                   approx_lik(m.joint_belief.astype(F64))).mean(-1)
    
    if te is None: te = T
    
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

def plot_diagnostics_1x4(axes, agent, name):
    def ex(y, lim=None): return _ep_xy(_smooth(y, PLOT_CFG["smooth_w"]), lim)
    
    # Final Accuracy
    x, y = ex(agent.test_acc_through_training[:, -1])
    axes[0].plot(x, y, c=AGENT_COLORS[name], label=name, zorder=3)
    baseline_color = AGENT_COLORS["Joint"] if name == "Trained" else AGENT_COLORS["Naive"]
    baseline_label = "Joint" if name == "Trained" else "Naive"
    acc = float(np.mean(agent.joint_acc[:, -1])) if name == "Trained" else float(np.mean(agent.naive_acc[:, -1]))
    axes[0].axhline(acc, c=baseline_color, ls="--", label=baseline_label, zorder=5)

    # Early Acc, SII, and PR
    xi, yi = ex(agent.test_acc_through_training[:, -1], PLOT_CFG["early_epochs"])
    axes[1].plot(xi, yi, c=AGENT_COLORS[name], zorder=3)

    xi, yi = ex(agent.test_SII_coef_through_training, PLOT_CFG["early_epochs"])
    axes[2].plot(xi, yi, c=AGENT_COLORS[name], zorder=3)
    axes[2].axhline(y = 0, c = 'k', alpha = .5)

    y_raw = pr_stat(agent.test_model_update_dim_through_training) - (pr_stat(agent.test_model_input_dim_through_training) + 1e-12)
    xi, yi = ex(y_raw, PLOT_CFG["early_epochs"])
    axes[3].plot(xi, yi, c=AGENT_COLORS[name], zorder=3)
    axes[3].axhline(y = 0, c = 'k', alpha = .5)

def finalize_layout_A(axes):
    titles = ["Final Step Accuracy", "Early Learning Acc.", "Correlation(FR, Acc.)", "PR(RNN) - PR(Read-in)"]
    ylabels = ["Accuracy", "Accuracy", "r", r"$\Delta$ PR"]
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], pad=15)
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel("Testing Episode")
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, ls='--', alpha=0.3)
    axes[0].legend(loc='lower right', frameon=False, ncols=2)
    plt.tight_layout()

def plot_dynamics_1x5(axes, d_tr, d_ec):
    tau = np.arange(DYN_PARAMS["E_START"], DYN_PARAMS["E_END"] + 1)
    past, future = tau <= 0, tau >= 0
    
    metrics = [
        ("l2", "Norm (L2)", r"$\|u_t\|_2$", 2),
        ("spar", "Sparsity", r"$\frac{\sqrt{N} - (\|u_t\|_1 / \|u_t\|_2)}{\sqrt{N}-1}$", 3),
        ("pr", "PR(RNN)", r"$(\sum \lambda_i)^2 / \sum \lambda_i^2$", 4),
        ("dist", "Distance", r"$\|u_t - u_{t-1}\|$", 1),
        ("v_prev", "Observation Change", r"$(o^{i}_{t} - o^{i}_{t-1})^2$", 0)
    ]
    
    for key, name, math, c in metrics:
        ax = axes[c]
        for data, label in [(d_tr, "Trained"), (d_ec, "Echo")]:
            color = AGENT_COLORS[label]
            mx = get_xcorr(data[key], data["rev"], DYN_PARAMS["E_START"], DYN_PARAMS["E_END"])
            
            # Polished Past (Dotted, smaller markers, white edges)
            ax.plot(tau[past], mx[past], c=color, alpha=0.5, lw=2.0, ls=':', 
                    marker='o', mec='white', mew=1, ms=4, zorder=3)
            
            # Polished Future (Solid, larger markers, black edges)
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
    state_num, obs_num, batch_num = 500, 5, 20000

    # trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1, 'checkpoint_every': 5, 
    #                                 'realization_num': realization_num, 'hid_dim': hid_dim, 'obs_num': obs_num, 
    #                                 'show_plots': False, 'batch_num': batch_num, 'step_num': step_num, 
    #                                 'state_num': state_num, 'learn_embeddings': False, 'reservoir': False, 
    #                                 'classifier_LR': .001, 'ctx_num': 2, 'training': False, 
    #                                 'load_env': "/sanity/fully_trained_ctx_2_e5"})       
    # echo = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1, 'checkpoint_every': 5, 
    #                              'realization_num': realization_num, 'hid_dim': hid_dim, 'obs_num': obs_num, 
    #                              'show_plots': False, 'batch_num': batch_num, 'step_num': step_num, 
    #                              'state_num': state_num, 'learn_embeddings': False, 'reservoir': True, 
    #                              'classifier_LR': .001, 'ctx_num': 2, 'training': False, 
    #                              'load_env': "/sanity/reservoir_ctx_2_e5"})    

    # 1. Prepare Data
    data_tr = prep_model_dynamics(trained, DYN_PARAMS)
    data_ec = prep_model_dynamics(echo, DYN_PARAMS)

    # 2. Figure A: Diagnostics
    figA, axesA = plt.subplots(1, 4, figsize=PLOT_CFG["figsize_A"])
    plot_diagnostics_1x4(axesA, trained, "Trained")
    plot_diagnostics_1x4(axesA, echo, "Echo")
    finalize_layout_A(axesA)
    figA.savefig("diagnostics_panel.svg", format="svg", bbox_inches="tight")

    # 3. Figure B: Event Dynamics
    figB, axesB = plt.subplots(1, 5, figsize=PLOT_CFG["figsize_B"])
    plt.subplots_adjust(bottom=0.2, wspace=0.4)
    plot_dynamics_1x5(axesB, data_tr, data_ec)
    figB.savefig("dynamics_panel.svg", format="svg", bbox_inches="tight")
    plt.show()

