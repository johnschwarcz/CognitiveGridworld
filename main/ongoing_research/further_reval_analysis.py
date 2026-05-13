import numpy as np, torch, os, sys, inspect
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
for p in ['/main', '/main/bayes', '/main/model']: sys.path.insert(0, path + p)
from main.CognitiveGridworld import CognitiveGridworld

F64, I64 = np.float64, np.int64

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14, 'axes.titleweight': 'bold', 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.dpi': 300, 'font.family': 'sans-serif'})

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

def step_participation_ratio(x, eps=1e-99):
    x_sq = x ** 2
    return (np.sum(x_sq, axis=-1) ** 2) / np.maximum(np.sum(x_sq ** 2, axis=-1), eps)

def compute_multi_anchor_geometry(m, num_pca_dims=150, max_offset=3, anchor_steps=[15], num_percentiles=20):
    rev_full = step_dkl(m.naive_px / m.naive_px.sum(-1, keepdims=True), approx_lik(m.joint_belief.astype(F64))).mean(-1)
    
    upd_raw = m.model_update_flat.astype(F64)
    B, T_total, N = upd_raw.shape
    
    for step in anchor_steps:
        if step - max_offset < 0 or step + max_offset >= T_total:
            raise ValueError(f"anchor_step {step} must allow +/- {max_offset} steps within sequence length {T_total}")
        
    offsets = list(range(-max_offset, max_offset + 1))
    
    # 1. Pool Re-evaluation across all provided anchor steps
    r_t_pooled = np.concatenate([rev_full[:, s] for s in anchor_steps]).flatten()
    p_edges = np.percentile(r_t_pooled, np.linspace(0, 100, num_percentiles + 1))
    p_edges[0], p_edges[-1] = -np.inf, np.inf 
    
    # 2. Fit PCA on the aggregated updates at the anchor steps
    upd_pooled = np.concatenate([upd_raw[:, s, :] for s in anchor_steps], axis=0)
    pca = PCA(n_components=num_pca_dims)
    pca.fit(upd_pooled)
    
    geom_data = {
        "l2": {o: np.zeros(num_percentiles) for o in offsets},
        "pr": {o: np.zeros(num_percentiles) for o in offsets}
    }
    
    # 3. Pre-calculate pooled offset geometries
    l2_all, pr_all = {}, {}
    for o in offsets:
        u_o_pooled = np.concatenate([upd_raw[:, s + o, :] for s in anchor_steps], axis=0)
        l2_all[o] = np.linalg.norm(u_o_pooled, axis=-1).flatten()
        z_o = pca.transform(u_o_pooled)
        pr_all[o] = step_participation_ratio(z_o).flatten()

    print(f"Mapping Geometry across {len(anchor_steps)} anchors and {num_percentiles} Re-evaluation bins...")
    for n in range(num_percentiles):
        mask = (r_t_pooled >= p_edges[n]) & (r_t_pooled <= p_edges[n+1] if n == num_percentiles - 1 else r_t_pooled < p_edges[n+1])
        
        if mask.sum() < 2: continue
        
        for o in offsets:
            geom_data["l2"][o][n] = np.mean(l2_all[o][mask])
            geom_data["pr"][o][n] = np.mean(pr_all[o][mask])
            
    return geom_data

def plot_geometric_state_space(geom_data, num_percentiles, max_offset, anchor_steps, plot_percentiles=[0.0, 0.25, 0.5, 0.75, 1.0]):
    fig = plt.figure(figsize=(32, 12))
    gs = fig.add_gridspec(2, 4, wspace=0.3, hspace=0.3, width_ratios = [.5,.5,1, 2])
    
    x_bins = np.arange(num_percentiles)
    tick_step = max(1, num_percentiles // 5)
    offsets = sorted(list(geom_data["l2"].keys()))
    positive_offsets = [o for o in offsets if o > 0]
    
    anchor_label = f"t={anchor_steps[0]}" if len(anchor_steps)==1 else f"t \in {{{anchor_steps[0]}...{anchor_steps[-1]}}}"

    # axes map
    ax0 = fig.add_subplot(gs[0, 0]) # Env L2
    ax1 = fig.add_subplot(gs[0, 1]) # Asym L2
    ax2 = fig.add_subplot(gs[1, 0]) # Env PR
    ax3 = fig.add_subplot(gs[1, 1]) # Asym PR
    ax4 = fig.add_subplot(gs[0, 2]) # Traj L2
    ax5 = fig.add_subplot(gs[1, 2]) # Traj PR
    ax_phase = fig.add_subplot(gs[:, 3]) # Phase Portrait (Full Height)

    # ═══════════════════════════════════════════════════════════════════
    # Column 1 & 2: Envelopes and Asymmetries
    # ═══════════════════════════════════════════════════════════════════
    for o in offsets:
        c, alpha, lw, zorder = ('black', 1.0, 3.5, 10) if o == 0 else (plt.cm.coolwarm((o + max_offset) / (2.0 * max_offset)), 0.5, 2.0, 5)
        label = f"Anchor ({anchor_label})" if o == 0 else f"t{o:+}"
        ax0.plot(x_bins, geom_data["l2"][o], c=c, alpha=alpha, lw=lw, label=label, zorder=zorder)
        ax2.plot(x_bins, geom_data["pr"][o], c=c, alpha=alpha, lw=lw, label=label, zorder=zorder)

    for o in positive_offsets:
        c = plt.cm.plasma(o / float(max_offset))
        label = rf"$\Delta={o}$"
        ax1.plot(x_bins, geom_data["l2"][o] - geom_data["l2"][-o], c=c, alpha=0.8, lw=2.5, label=label)
        ax3.plot(x_bins, geom_data["pr"][o] - geom_data["pr"][-o], c=c, alpha=0.8, lw=2.5, label=label)

    # ═══════════════════════════════════════════════════════════════════
    # Column 3 & 4: Trajectories and Phase Portrait
    # ═══════════════════════════════════════════════════════════════════
    selected_bins = [min(int(p * num_percentiles), num_percentiles - 1) for p in plot_percentiles]
    bin_labels = [f"Pct {int(p*100)}" for p in plot_percentiles]

    for idx, b in enumerate(selected_bins):
        c = plt.cm.viridis(idx / max(1, (len(selected_bins) - 1)))
        l2_v = [geom_data["l2"][o][b] for o in offsets]
        pr_v = [geom_data["pr"][o][b] for o in offsets]
        
        # Time-series Trajectories
        ax4.plot(offsets, l2_v, c=c, lw=3, marker='o', alpha=0.8, label=bin_labels[idx])
        ax5.plot(offsets, pr_v, c=c, lw=3, marker='o', alpha=0.8, label=bin_labels[idx])

        # Phase Portrait
        ax_phase.plot(pr_v, l2_v, c=c, lw=3, alpha=0.6)
        # Add directional arrows to phase plot
        dx, dy = np.diff(pr_v), np.diff(l2_v)
        ax_phase.quiver(pr_v[:-1], l2_v[:-1], dx, dy, color=c, angles='xy', scale_units='xy', scale=1, width=0.005)
        # Highlight anchor point
        anchor_idx = offsets.index(0)
        ax_phase.scatter(pr_v[anchor_idx], l2_v[anchor_idx], color=c, edgecolor='black', marker='*', s=300, zorder=15)

    # ═══════════════════════════════════════════════════════════════════
    # Formatting
    # ═══════════════════════════════════════════════════════════════════
    titles = [
        f"Magnitude ({anchor_label})", "Magnitude Asymmetry (F-P)",
        f"Dimensionality ({anchor_label})", "Dimensionality Asymmetry (F-P)",
        "Magnitude vs Offset", "Dimensionality vs Offset",
        "Phase Portrait: Dimensionality vs Magnitude"
    ]
    ylabels = [r"$||u||_2$", r"$\Delta ||u||_2$", "PR", r"$\Delta PR$", r"$||u||_2$", "PR", r"Update Magnitude $||u||_2$"]
    all_axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax_phase]
    
    for i, ax in enumerate(all_axes):
        ax.set_title(titles[i], pad=10)
        ax.set_ylabel(ylabels[i])
        ax.grid(True, ls='--', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        
        if i in [0, 1, 2, 3]:
            ax.set_xlabel("Re-evaluation Bin")
            ax.set_xticks(np.arange(0, num_percentiles, tick_step))
            if i in [1, 3]: ax.axhline(0, color='black', ls='--', alpha=0.5)
        elif i in [4, 5]:
            ax.set_xlabel("Temporal Offset")
            ax.set_xticks(offsets)
            ax.axvline(0, color='black', ls='--', alpha=0.5)
        else:
            ax.set_xlabel("Participation Ratio (PR)")

        ax.legend(frameon=False, fontsize=9, loc='best')

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    cuda, realization_num, step_num, hid_dim, state_num, obs_num, batch_num = 0, 10, 30, 1000, 500, 5, 20000
    trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1, 'checkpoint_every': 5, 'realization_num': realization_num, 'hid_dim': hid_dim, 'obs_num': obs_num, 'show_plots': False, 'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False, 'reservoir': False, 'classifier_LR': .001, 'ctx_num': 2, 'training': False, 'load_env': "/sanity/fully_trained_ctx_2_e5"})       
    # ═══════════════════════════════════════════════════════════════════
    # Analysis Configuration
    # ═══════════════════════════════════════════════════════════════════
    num_pca_dimensions = 1000     
    num_percentile_bins = 10
    TEMPORAL_WINDOW = 5          
    ANCHOR_STEPS = list(range(10,25))
    PERCENTILES_TO_PLOT = [0.05, 0.5, 0.95]
    print(f"Computing Geometric State-Space Dynamics for anchors {ANCHOR_STEPS}...")
    geom_data = compute_multi_anchor_geometry(        trained,         num_pca_dims=num_pca_dimensions,
        max_offset=TEMPORAL_WINDOW,        anchor_steps=ANCHOR_STEPS,         num_percentiles=num_percentile_bins    )
    fig = plot_geometric_state_space(        geom_data,         num_percentiles=num_percentile_bins, 
        max_offset=TEMPORAL_WINDOW,        anchor_steps=ANCHOR_STEPS,        plot_percentiles=PERCENTILES_TO_PLOT    )
    plt.show()