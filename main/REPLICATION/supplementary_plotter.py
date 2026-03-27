import numpy as np
import torch
import os
import sys
import inspect
import gc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

path = inspect.getfile(inspect.currentframe())
path = os.path.dirname(os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld

# ═══════════════════════════════════════════════════════════════════
# Plotting Hyperparameters & Configuration
# ═══════════════════════════════════════════════════════════════════

PLOT_CFG = {
    # Layout & Sizing
    "figsize_diag": (11, 5),         # 1x3 grid for diagnostics
    "figsize_pca": (22, 10),         # 3x9 grid for PCA (includes center spacer column)
    "col_ratios_pca": [1, 1, 1, 1, 0.4, 1, 1, 1, 1], # 0.4 is the width of the spacer
    "lim_epochs": 400,               # X-axis limit for training diagnostics
    "smooth_w": 1,                   # Smoothing window for line plots
    
    # PCA Scatter Styles
    "pca_scatter_s": 80,             # Size of points in PCA plots
    "pca_scatter_alpha": .1,# 0.25,       # Transparency of PCA points
    "cmap_sum": "viridis",           # Colormap for the SUM metric
    "cmap_diff": "inferno",          # Colormap for the DIFF metric
    "cmap_entropy": "plasma",        # Colormap for the ENTROPY metric
    
    # Line Plot Styles
    "line_width": 2.5,               # Standard line width
    
    # Layering (Z-Orders)
    "z_net_0": 2,                  
    "z_net_1": 3,
    "z_bayes_0": 5,
    "z_bayes_1": 6,
}

AGENT_COLORS = {"Trained": "C0", "Echo": "C1", "Joint": "C2", "Naive": "C3"}

# Set high-quality defaults for publication
plt.rcParams.update({
    'font.size': 15,
    'axes.labelsize': 15,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'axes.linewidth': 1.2,
    'lines.linewidth': PLOT_CFG["line_width"],
    'figure.dpi': 300, 
    'font.family': 'sans-serif',
})

# ═══════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════

def pr(evals, eps=1e-12):
    e = np.asarray(evals, float)
    return (e.sum(-1) ** 2) / ((e * e).sum(-1) + eps)

def calc_entropy(p):
    p = np.clip(p, 1e-20, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return -(p * np.log(p)).sum(axis=1)

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
        if m < 1: m = 1
        if m > n: m = n
        idx = np.arange(m)
    elif isinstance(episode_lim, slice):
        idx = np.arange(n)[episode_lim]
    else:
        start, stop, step = 1, n, 1
        s = int(start) - 1
        e = min(n, int(stop))
        idx = np.arange(max(0, s), e, max(1, int(step)))

    if idx.size == 0:
        idx = np.arange(min(1, n))
    return idx + 1, y[idx]

def compute_r2(X, y):
    return Ridge(alpha=1.0).fit(X, y).score(X, y)

def _format_academic_ax(a):
    """Removes top/right spines and adds a subtle grid for a cleaner look."""
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.grid(True, linestyle='--', alpha=0.4, zorder=0)

# ═══════════════════════════════════════════════════════════════════
# Plotting Architecture (Sequential Memory-Safe Loading)
# ═══════════════════════════════════════════════════════════════════

def plot_diagnostics(ax_row, agent, name, episode_lim=None):
    """Extracts and plots the line chart diagnostics for a single agent."""
    def ex(y):
        return _ep_xy(_smooth(y, PLOT_CFG["smooth_w"]), episode_lim)

    # ACC [Col 0]
    x, y = ex(agent.test_acc_through_training[:, -1])
    ax_row[0].plot(x, y, c=AGENT_COLORS[name], label=name, zorder=3)
    
    # CORR [Col 1]
    x, y = ex(agent.test_SII_coef_through_training)
    ax_row[1].plot(x, y, c=AGENT_COLORS[name], label=name, zorder=3)

    # PR DIFF [Col 2]
    pr_out = pr(agent.test_model_update_dim_through_training)
    pr_in = pr(agent.test_model_input_dim_through_training)
    x, y = ex(pr_out - (pr_in + 1e-12))
    zord = PLOT_CFG["z_net_0"] if name == "Trained" else PLOT_CFG["z_net_1"]
    ax_row[2].plot(x, y, c=AGENT_COLORS[name], label=name, zorder=zord)


def _plot_pca_scatter(ax_target, bel_raw, ctx_raw, t_idx, metric_type):
    """Helper to compute PCA and scatter a single manifold."""
    bel = np.asarray(bel_raw)
    ctx = np.asarray(ctx_raw)
    
    t_target = t_idx if t_idx >= 0 else bel.shape[1] + t_idx
    t_target = np.clip(t_target, 0, bel.shape[1] - 1)
    
    r1, r2 = ctx[:, 0].astype(int), ctx[:, 1].astype(int)
    Xs = bel[:, t_target]
    Xf = Xs.reshape(len(Xs), -1)
    
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xf)
    
    if metric_type == 'SUM':
        c_data, cmap = r1 + r2, PLOT_CFG["cmap_sum"]
    elif metric_type == 'DIFF':
        c_data, cmap = r1 - r2, PLOT_CFG["cmap_diff"]
    elif metric_type == 'ENTROPY':
        X3 = Xs.reshape(len(Xs), 2, -1) if Xs.ndim == 2 else Xs
        c_data = calc_entropy(X3[:, 0]) + calc_entropy(X3[:, 1])
        cmap = PLOT_CFG["cmap_entropy"]
        
    ax_target.scatter(Xp[:, 0], Xp[:, 1], c=c_data, cmap=cmap, 
                      s=PLOT_CFG["pca_scatter_s"], 
                      alpha=PLOT_CFG["pca_scatter_alpha"], 
                      edgecolor='none', rasterized=True)
    
    sc = compute_r2(Xp, c_data)
    ax_target.text(0.97, 0.03, f"$R^2$: {sc:.2f}", transform=ax_target.transAxes, ha='right', va='bottom', 
            fontweight='bold', fontsize=15, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=np.clip(1.5*sc, 0.2, 1), edgecolor='k'))
    
    ax_target.set_xticks([]); ax_target.set_yticks([])
    for spine in ax_target.spines.values():
        spine.set_visible(False)


def plot_pca_combined(ax_grid, agent, name):
    """Maps manifolds to their designated columns in the combined 3x9 PCA grid."""
    if name == "Trained":
        models_cfg = [
            {"bel": agent.model_belief_flat, "ctx": agent.ctx_vals, "name": "Trained", "c_offset": 0},
            {"bel": agent.joint_belief, "ctx": agent.ctx_vals, "name": "Joint", "c_offset": 1}
        ]
    else: # Echo
        models_cfg = [
            {"bel": agent.model_belief_flat, "ctx": agent.ctx_vals, "name": "Echo", "c_offset": 2},
            {"bel": agent.naive_belief, "ctx": agent.ctx_vals, "name": "Naive", "c_offset": 3}
        ]
    
    metrics = ["SUM", "DIFF", "ENTROPY"]

    for r, m_type in enumerate(metrics):
        for m_cfg in models_cfg:
            c_0 = m_cfg["c_offset"]
            
            # t=0 Plotting
            _plot_pca_scatter(ax_grid[r, c_0], m_cfg["bel"], m_cfg["ctx"], 0, m_type)
            if r == 0:
                ax_grid[r, c_0].set_title(f"{m_cfg['name']}", fontsize=15, fontweight='bold', pad=10)
            if c_0 == 0:
                ax_grid[r, c_0].set_ylabel(m_type, fontsize=15, fontweight='bold', labelpad=10)

            # t=end Plotting (Offset by 5 columns to skip the center spacer)
            c_end = c_0 + 5
            _plot_pca_scatter(ax_grid[r, c_end], m_cfg["bel"], m_cfg["ctx"], -1, m_type)
            if r == 0:
                ax_grid[r, c_end].set_title(f"{m_cfg['name']}", fontsize=15, fontweight='bold', pad=10)


def apply_dynamic_ylim(ax_target):
    """Helper to dynamically set ylim strictly based on data within lim_epochs."""
    lim_val = PLOT_CFG["lim_epochs"]
    valid_y = []
    for line in ax_target.get_lines():
        if line.get_color() == "black": continue # Skip baselines
        # Cast to array to safely handle <= logic
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        valid_y.extend(ydata[xdata <= lim_val])
            
    if len(valid_y) > 0:
        y_min, y_max = np.nanmin(valid_y), np.nanmax(valid_y)
        pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
        ax_target.set_ylim(y_min - pad, y_max + pad)

def finalize_diagnostics(fig, ax_row):
    """Applies final formatting to the 1x3 diagnostics figure."""
    
    # ACC [Col 0]
    apply_dynamic_ylim(ax_row[0])
    ax_row[0].set_ylabel("Accuracy")
    ax_row[0].set_title("Final Step Accuracy", loc="center", pad=15)
    ax_row[0].legend(frameon=False, loc="upper left", ncol=1)
    _format_academic_ax(ax_row[0])

    # CORR [Col 1]
    ax_row[1].axhline(0, c="black", ls="-", lw=1.2, alpha=0.6)
    apply_dynamic_ylim(ax_row[1])
    ax_row[1].set_ylabel("Coefficient")
    ax_row[1].set_title("Accuracy-SII Correlation", loc="center", pad=15)
    _format_academic_ax(ax_row[1])

    # PR DIFF [Col 2]
    ax_row[2].axhline(0, c="black", ls="-", lw=1.2, alpha=0.6)
    apply_dynamic_ylim(ax_row[2])
    ax_row[2].set_ylabel("PR(RNN) - PR(Read-in)")
    ax_row[2].set_title("Relative Participation Ratio", loc="center", pad=15)
    _format_academic_ax(ax_row[2])
    
    # Sync limits & add labels
    label_font = {'fontsize': 15, 'fontweight': 'normal', 'ha': 'center', 'va': 'top'}
    labels = ["(a)", "(b)", "(c)"]
    
    for c in range(3):
        ax_row[c].set_xlim(1, PLOT_CFG["lim_epochs"])
        ax_row[c].set_xlabel("Testing  Episode")
        ax_row[c].text(0.5, -0.22, labels[c], transform=ax_row[c].transAxes, **label_font)
        ax_row[c].set_xticks([100, 300])
    fig.tight_layout(pad=1.5, w_pad=0)


def finalize_pca_grid(fig, ax_grid):
    """Applies final layout formatting to the 3x9 PCA figure."""
    
    # Turn off the central spacer column
    for r in range(3):
        ax_grid[r, 4].axis("off")
        
    # Leave room at the bottom for our custom text labels
    fig.tight_layout(pad=1.5, w_pad=0.5, h_pad=1.5, rect=[0, 0.08, 1, 1])
    
    title_font = {'fontsize': 16, 'fontweight': 'bold', 'ha': 'center', 'va': 'top'}
    label_font = {'fontsize': 16, 'fontweight': 'normal', 'ha': 'center', 'va': 'top'}
    
    # Anchor the t=0 labels to the space between Column 1 and 2
    ax_grid[2, 1].text(1.0, -0.15, "First Observation", transform=ax_grid[2, 1].transAxes, **title_font)
    ax_grid[2, 1].text(1.0, -0.35, "(a)", transform=ax_grid[2, 1].transAxes, **label_font)
    
    # Anchor the t=end labels to the space between Column 6 and 7
    ax_grid[2, 6].text(1.0, -0.15, "Final Observation", transform=ax_grid[2, 6].transAxes, **title_font)
    ax_grid[2, 6].text(1.0, -0.35, "(b)", transform=ax_grid[2, 6].transAxes, **label_font)


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
    batch_num = 25000
    episodes = 50000 if training else 1

    # Setup the two distinct figures
    fig_diag, ax_diag = plt.subplots(1, 3, figsize=PLOT_CFG["figsize_diag"])
    fig_pca, ax_pca = plt.subplots(3, 9, figsize=PLOT_CFG["figsize_pca"], gridspec_kw={'width_ratios': PLOT_CFG["col_ratios_pca"]})

    # -----------------------------------------------------------------
    # Pass 1: Trained Agent (Load, Plot to both figs, Delete)
    # -----------------------------------------------------------------
    trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5,
        'realization_num': realization_num, 'hid_dim': hid_dim, 'obs_num': obs_num, 'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .001, 'ctx_num': 2,
        'training': training,
        'save_env': "/sanity/fully_trained_ctx_2_e5" if training else None,
        'load_env': None if training else "/sanity/fully_trained_ctx_2_e5"})
    
    plot_diagnostics(ax_diag, trained, "Trained", episode_lim=None)
    plot_pca_combined(ax_pca, trained, "Trained")
    
    del trained
    gc.collect()

    # -----------------------------------------------------------------
    # Pass 2: Echo Agent (Load, Plot to both figs, Delete)
    # -----------------------------------------------------------------
    echo = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5,
        'realization_num': realization_num, 'hid_dim': hid_dim, 'obs_num': obs_num, 'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 2,
        'training': training,
        'save_env': "/sanity/reservoir_ctx_2_e5" if training else None,
        'load_env': None if training else "/sanity/reservoir_ctx_2_e5"})
    
    plot_diagnostics(ax_diag, echo, "Echo", episode_lim=None)
    plot_pca_combined(ax_pca, echo, "Echo")
    
    del echo
    gc.collect()

    # -----------------------------------------------------------------
    # Finalize & Save All Figures
    # -----------------------------------------------------------------
    finalize_diagnostics(fig_diag, ax_diag)
    finalize_pca_grid(fig_pca, ax_pca)
    
    fig_diag.savefig("diagnostics.svg", format="svg", bbox_inches="tight", dpi=300)
    fig_pca.savefig("pca_manifolds.svg", format="svg", bbox_inches="tight", dpi=300)
    
    plt.show()

