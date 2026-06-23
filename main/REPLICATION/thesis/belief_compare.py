import numpy as np
import os
import sys
import inspect
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib import gridspec 
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import PowerNorm

# Setup Paths
path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("root:", path)

sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')

from main.CognitiveGridworld import CognitiveGridworld

plt.rcParams.update({'font.family':'serif','font.size':13,'axes.labelsize':13,'axes.titlesize':16,'legend.fontsize':11})

# Define Global Styles
PLOT_STYLES = {
    'Exact': {'color': 'lightgray', 'linestyle': '-', 'marker': 'D', 'lw':4, 'mec':'k'},
    'Factorized': {'color': 'k', 'linestyle': '-', 'marker': 'D', 'lw':4, 'mec':'k'},
    'Fully-Trained': {'color': 'C2', 'linestyle': '-', 'lw':2},
    'Echo-State': {'color': 'C3', 'linestyle': '-', 'lw':2}
}

def center_beliefs(beliefs, ctx_vals, target=5):
    out = np.zeros_like(beliefs)
    for b in range(beliefs.shape[0]):
        for c in range(beliefs.shape[2]):
            out[b, :, c] = np.roll(beliefs[b, :, c], target - int(ctx_vals[b, c]), axis=-1)
    return out

def get_high_kl_indices(j, n, n_examples=5):
    """Identifies indices with high KL divergence between joint and naive beliefs."""
    eps = 1e-9
    kl = np.sum(j * (np.log(j + eps) - np.log(n + eps)), axis=-1)
    total_kl = kl[:,10:].sum(1)
    flat_indices = np.argsort(total_kl.flatten())[::-1][:n_examples]
    batch_indices = flat_indices // j.shape[2]
    ctx_indices = flat_indices % j.shape[2]
    return batch_indices, ctx_indices

def plot_summary_and_heatmaps(j, n, ft, echo, j_t, n_t, ft_t, echo_t, realization_num, step_num,
                              j_full, n_full, ft_full, j_echo_full, n_echo_full, echo_full):
    
    fig = plt.figure(figsize=(14, 8))
    
    master_gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 2.0], figure=fig, wspace=0.28)
    gs_left = master_gs[0].subgridspec(2, 1, hspace=0.4)
    gs_right = master_gs[1].subgridspec(2, 2, wspace=0.05, hspace=0.15)
    
    axs = {
        'S1': fig.add_subplot(gs_left[0, 0]),
        'S2': fig.add_subplot(gs_left[1, 0]),
        'H1': fig.add_subplot(gs_right[0, 0]),
        'H2': fig.add_subplot(gs_right[0, 1]),
        'H3': fig.add_subplot(gs_right[1, 0]),
        'H4': fig.add_subplot(gs_right[1, 1])
    }

    fig.canvas.draw() 
    h1_pos = axs['H1'].get_position()
    h2_pos = axs['H2'].get_position()
    heatmap_center_x = (h1_pos.x0 + h2_pos.x1) / 2
    
    fig.suptitle(
        'Network & Bayes MAP Agreement', 
        x=heatmap_center_x, 
        y=h1_pos.y1 + 0.02,
        horizontalalignment='center', 
        verticalalignment='bottom',
        fontsize=plt.rcParams['axes.titlesize'],
        fontweight=plt.rcParams['axes.titleweight'],
        fontfamily=plt.rcParams['font.family'] if 'font.family' in plt.rcParams else 'serif'
    )

    x_centered = np.arange(-4, 5)
    t = np.arange(step_num)
    
    data_map = {'Exact': j, 'Factorized': n, 'Fully-Trained': ft, 'Echo-State': echo}
    temporal_data_map = {'Exact': j_t, 'Factorized': n_t, 'Fully-Trained': ft_t, 'Echo-State': echo_t}

    for name, style in PLOT_STYLES.items():
        mean = data_map[name].mean(axis=(0, 1))[1:10]
        axs['S1'].plot(x_centered, mean, **style, label=name)
        
        temp_mean = temporal_data_map[name][:, :, :, 5].mean(axis=(0, 2))
        axs['S2'].plot(t, temp_mean, **style, label=name, ms=6)

    tick_vals = np.arange(1, 10)  
    tick_labels = [f"+{i - 5}" if i > 5 else (r"$r^{\bigstar}_g$" if i == 5 else f"{i - 5}") for i in tick_vals]

    axs['S1'].axvline(0, color='gray', ls=':', alpha=0.4) 
    axs['S1'].axhline(1 / realization_num, color='k', ls='--', alpha=0.4)
    axs['S1'].set(title='Final Belief', xlabel='Centered $r_g$', ylabel='Belief Mass', ylim=(-0.05, 1.05))
    axs['S1'].set_xticks(tick_vals - 5)
    axs['S1'].set_xticklabels(tick_labels)
    axs['S1'].set_xlim(-4.0, 4.0)
    axs['S1'].legend(frameon=True)

    axs['S2'].set(title='Evidence Accumulation', xlabel='Inference Step', ylabel=r'$P(r^{\bigstar}_g)$', xlim=(0, step_num - 1), ylim=(-.05, 1.05))

    def flatten_all(x):
        return np.argmax(x, axis=-1).reshape(-1)

    pairs = [
        ('H1', flatten_all(j_full), flatten_all(ft_full), 'Exact MAP', 'Fully-Trained MAP'),
        ('H2', flatten_all(n_full), flatten_all(ft_full), 'Factorized MAP', 'Fully-Trained MAP'),
        ('H3', flatten_all(j_echo_full), flatten_all(echo_full), 'Exact MAP', 'Echo-State MAP'),
        ('H4', flatten_all(n_echo_full), flatten_all(echo_full), 'Factorized MAP', 'Echo-State MAP')
    ]

    custom_cmap = LinearSegmentedColormap.from_list(
        'ContinuousBlackToBlue', 
        ['#000000', '#0d2b56', '#1a5292', '#3b82f6', '#93c5fd', '#ffffff']
    )

    for key, x_data, y_data, xl, yl in pairs:
        ax = axs[key]
        H, _, _ = np.histogram2d(x_data, y_data, bins=np.arange(-0.5, 10.5, 1))
        vmax = np.max(H)
        
        norm = PowerNorm(gamma=.7, vmin=0, vmax=vmax *.45)
        
        ax.imshow(H.T, origin='lower', cmap=custom_cmap, norm=norm, aspect='equal')
        ax.plot([-0.5, 9.5], [-0.5, 9.5], color='gray', ls='--', alpha=0.5)
        
        ax.add_patch(patches.Rectangle((4.5, 4.5), 1, 1, fill=False, edgecolor='red', linewidth=1.5))
        ax.text(
            5.0, 5.0, r'$r^{\bigstar}_g$', color='red', fontsize=13,
            fontweight='bold', ha='center', va='center'
        )
        
        ax.set_xlim(0.5, 9.5)
        ax.set_ylim(0.5, 9.5)
        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)

        if key in ['H3', 'H4']:
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel(xl)
        else:
            ax.set_xticklabels([])
            ax.set_xlabel('')

        if key in ['H1', 'H3']:
            ax.set_yticklabels(tick_labels)
            ax.set_ylabel(yl)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')

    plt.show()

def plot_map_trajectories(agent_ft, agent_echo, n_examples=3):
    total_plots = n_examples * n_examples
    
    fig, axs = plt.subplots(n_examples, 2 * n_examples, figsize=(6 * n_examples, 3 * n_examples), 
                            constrained_layout=True, sharex=True, sharey=True)
    fig.suptitle(f'Example MAP Trajectories', fontsize=12)

    b_idx_ft, c_idx_ft = get_high_kl_indices(agent_ft.joint_belief, agent_ft.naive_belief, n_examples=total_plots)
    b_idx_echo, c_idx_echo = get_high_kl_indices(agent_echo.joint_belief, agent_echo.naive_belief, n_examples=total_plots)

    col_configs = [
        ('Fully-Trained', axs[:, :n_examples], b_idx_ft, c_idx_ft, agent_ft),
        ('Echo-State', axs[:, n_examples:], b_idx_echo, c_idx_echo, agent_echo)
    ]

    for col_idx, (title, grid_axs, b_indices, c_indices, agent) in enumerate(col_configs):       
        agent_style = PLOT_STYLES['Fully-Trained'] if 'Fully-Trained' in title else PLOT_STYLES['Echo-State']
        it = np.nditer(grid_axs, flags=['multi_index', 'refs_ok'])
        for ax in grid_axs.flat:
            row, col = np.where(grid_axs == ax)
            flat_i = row[0] * n_examples + col[0]
            
            b, c = b_indices[flat_i], c_indices[flat_i]
            
            map_j = np.argmax(agent.joint_belief[b, :, c], axis=-1)
            map_n = np.argmax(agent.naive_belief[b, :, c], axis=-1)
            map_m = np.argmax(agent.model_belief[b, :, c], axis=-1)
            
            ax.plot(map_j, label='Joint', **PLOT_STYLES['Exact'], ms=0)
            ax.plot(map_n, label='Factorized', **PLOT_STYLES['Factorized'], ms=0)
            ax.plot(map_m, label='Network', **agent_style)
            if col_idx == 0:
                ax.fill_between(range(len(map_m)), map_m, map_j, color=agent_style['color'], alpha=0.5)
            if col_idx == 1:
                ax.fill_between(range(len(map_m)), map_m, map_n, color=agent_style['color'], alpha=0.1)

            xax = np.arange(len(map_j))
            From = -2 
            xax = xax[From:]
            ax.plot(xax, map_j[From:], **PLOT_STYLES['Exact'], ms=10)
            ax.plot(xax, map_n[From:], label='Factorized', **PLOT_STYLES['Factorized'], ms=10)
            ax.plot(xax, map_m[From:], label='Network', **agent_style, marker = 'o', ms=5)
                        
            if col[0] == 0: 
                if col_idx == 0:
                    ax.set_ylabel('MAP')
            if row[0] == n_examples - 1:
                ax.set_xlabel('Step')
            
            ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    plt.show()

def plot_combined_analysis(agent_ft, agent_echo, target_zs=[0.17, 0.37], tolerance=0.03, bins=6, over_ctx = "avg", both=True, eps=1e-9, w = 7, h = 4):
    """
    Combines the MI binned accuracy plot and the Z-effects grid distribution plot
    into a single unified figure with two columns.
    """
    fig = plt.figure(figsize=(w, h), constrained_layout=False)
    master_spec = fig.add_gridspec(1, 2, width_ratios=[.8, 1], wspace=0.25, left=0.08, right=0.92, bottom=0.12, top=0.88)
    
    ax_mi = fig.add_subplot(master_spec[0, 0])
    
    mi_runs = {}
    agents_data = [('Fully-Trained', agent_ft), ('Echo-State', agent_echo)]
    
    for title, agent in agents_data:
        jl_all = agent.joint_likelihood
        p_xy_all = jl_all / (np.sum(jl_all, axis=(-2, -1), keepdims=True) + eps)
        p_x_all = np.sum(p_xy_all, axis=-1, keepdims=True)
        p_y_all = np.sum(p_xy_all, axis=-2, keepdims=True)
        
        mi_all = np.sum(p_xy_all * np.log((p_xy_all + eps) / (p_x_all * p_y_all + eps)), axis=(-2, -1))
        mi_runs[title] = np.mean(mi_all, axis=1)

    all_mi_avg = np.concatenate([mi_runs['Fully-Trained'], mi_runs['Echo-State']])

    if type(bins) == int:
        quantiles = np.linspace(0, 100, bins + 1)
        bins = np.percentile(all_mi_avg, quantiles)
        bins = np.unique(bins)
    else:
        bins = np.array(bins)
    actual_num_bins = len(bins) - 1
    bin_centers = (bins[:-1] + bins[1:]) / 2

    Jmax_ft = np.argmax(agent_ft.joint_belief, -1)
    Nmax_ft = np.argmax(agent_ft.naive_belief, -1)
    Jmax_echo = np.argmax(agent_echo.joint_belief, -1)
    Nmax_echo = np.argmax(agent_echo.naive_belief, -1)    

    jmax_ft_acc = (Jmax_ft == agent_ft.ctx_vals[:,None,:])
    Nmax_ft_acc = (Nmax_ft == agent_ft.ctx_vals[:,None,:])
    jmax_echo_acc = (Jmax_echo == agent_echo.ctx_vals[:,None,:])
    Nmax_echo_acc = (Nmax_echo == agent_echo.ctx_vals[:,None,:])
    if over_ctx == "avg":
        jmax_ft_acc = jmax_ft_acc.mean((-1,-2))
        Nmax_ft_acc = Nmax_ft_acc.mean((-1,-2))
        jmax_echo_acc = jmax_echo_acc.mean((-1,-2))
        Nmax_echo_acc = Nmax_echo_acc.mean((-1,-2))
    if over_ctx == "max":
        jmax_ft_acc = jmax_ft_acc.max(-1).mean(-1)
        Nmax_ft_acc = Nmax_ft_acc.max(-1).mean(-1)
        jmax_echo_acc = jmax_echo_acc.max(-1).mean(-1)
        Nmax_echo_acc = Nmax_echo_acc.max(-1).mean(-1)

    all_joint_acc = np.concatenate([jmax_ft_acc, jmax_echo_acc])
    all_naive_acc = np.concatenate([Nmax_ft_acc, Nmax_echo_acc])
    
    base_bin_indices = np.digitize(all_mi_avg, bins) - 1
    v_centers_b, j_m, j_se, n_m, n_se = [], [], [], [], []
    
    for i in range(actual_num_bins):
        mask = (base_bin_indices == i)
        n_samples = np.sum(mask)
        if n_samples > 5:
            v_centers_b.append(bin_centers[i])
            j_m.append(np.mean(all_joint_acc[mask]))
            j_se.append(np.std(all_joint_acc[mask]) / np.sqrt(n_samples))
            n_m.append(np.mean(all_naive_acc[mask]))
            n_se.append(np.std(all_naive_acc[mask]) / np.sqrt(n_samples))
            
    v_centers_b = np.array(v_centers_b)
    v_centers_b = np.clip(v_centers_b, a_min = eps, a_max = None)
    ax_mi.plot(v_centers_b, j_m, color='gray', ls=':', label='Joint', marker = 'D', ms = 4, mew = 0.5, mec = 'k')
    ax_mi.fill_between(v_centers_b, np.array(j_m) - 1.96 * np.array(j_se), np.array(j_m) + 1.96 * np.array(j_se), color='gray', alpha=0.1)

    ax_mi.plot(v_centers_b, n_m, color='k', ls='--', label='Factorized', marker = 'D', ms = 4, mew = 0.5, mec = 'k')
    ax_mi.fill_between(v_centers_b, np.array(n_m) - 1.96 * np.array(n_se), np.array(n_m) + 1.96 * np.array(n_se), color='k', alpha=0.1)

    agents_config = [
        ('Fully-Trained', agent_ft, mi_runs['Fully-Trained'], PLOT_STYLES['Fully-Trained']['color']),
        ('Echo-State', agent_echo, mi_runs['Echo-State'], PLOT_STYLES['Echo-State']['color'])
    ]

    for title, agent, mi_avg, model_color in agents_config:
        model_acc_avg = np.argmax(agent.model_belief, -1) == agent.ctx_vals[:,None,:]
        if over_ctx == "avg":
            model_acc_avg = model_acc_avg.mean((-1, -2))
        if over_ctx == "max":
            model_acc_avg = model_acc_avg.max(-1).mean(-1)

        bin_indices = np.digitize(mi_avg, bins) - 1
        v_centers, m_m, m_se = [], [], []
        for i in range(actual_num_bins):
            mask = (bin_indices == i)
            n_samples = np.sum(mask)
            if n_samples > 5:
                v_centers.append(bin_centers[i])
                m_m.append(np.mean(model_acc_avg[mask]))
                m_se.append(np.std(model_acc_avg[mask]) / np.sqrt(n_samples))

        v_centers = np.array(v_centers)
        v_centers = np.clip(v_centers, a_min = eps, a_max = None)
        ax_mi.plot(v_centers, m_m, color=model_color, lw= 1, label=f'{title}')
        ax_mi.fill_between(v_centers, np.array(m_m) - 1.96 * np.array(m_se), np.array(m_m) + 1.96 * np.array(m_se), color=model_color, alpha=0.1)

    ax_mi.set_xlabel("MI($R_1, R_2$) (avg. over observations)", fontsize=11)
    ax_mi.set_ylabel(f"Accuracy ({over_ctx}. over context)", fontsize=11)
    ax_mi.grid(alpha=0.3, ls=':')
    ax_mi.legend(frameon=False, loc='lower left', ncol = 2, fontsize = 12)
    ax_mi.set_title("Accuracy vs. Mutual Information", fontsize=12, pad=17)
    
    if both:
        grid_size = len(target_zs)
        joint_likelihood_results = []

        if agent_ft.ctx_num != 2 or not hasattr(agent_ft, 'joint_Z'):
            print("[Fully-Trained] Z-effects skipped: requires ctx_num=2 and joint_Z attribute.")
            ax_dummy = fig.add_subplot(master_spec[0, 1])
            ax_dummy.text(0.5, 0.5, "Z-Effects Grid Data Unavailable", ha='center', va='center')
            ax_dummy.axis('off')
        else:
            ax_right_title = fig.add_subplot(master_spec[0, 1])
            ax_right_title.axis('off')
            ax_right_title.set_title("Example Generating Functions\n", fontsize=12, pad = -15)

            outer_grid = master_spec[0, 1].subgridspec(grid_size, grid_size, wspace=0.15, hspace=0.1)
            
            Z0 = agent_ft.joint_Z[:, :, 0].flatten()
            Z1 = agent_ft.joint_Z[:, :, 1].flatten()
            
            nl_flat = agent_ft.naive_likelihood.reshape(-1, agent_ft.ctx_num, agent_ft.realization_num)
            L0 = nl_flat[:, 0, :]
            L1 = nl_flat[:, 1, :]
            jl_flat = agent_ft.joint_likelihood.reshape(-1, agent_ft.realization_num, agent_ft.realization_num)
            
            x_vals = np.arange(agent_ft.realization_num)
            scale = agent_ft.realization_num - 1 
            global_max = max(L0.max(), L1.max())
            
            for i in range(grid_size):
                for j in range(grid_size):
                    row = grid_size - 1 - j  
                    col = i                  
                    
                    inner_grid = outer_grid[row, col].subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.0, hspace=0.0)
                    
                    ax_joint = fig.add_subplot(inner_grid[1, 0])
                    ax_marg_x = fig.add_subplot(inner_grid[0, 0], sharex=ax_joint)
                    ax_marg_y = fig.add_subplot(inner_grid[1, 1], sharey=ax_joint)
                    
                    ax_marg_x.axis('off')
                    ax_marg_y.axis('off')
                    
                    mask_x = np.abs(Z0 - target_zs[i]) <= tolerance
                    mask_y = np.abs(Z1 - target_zs[j]) <= tolerance
                    mask = mask_x & mask_y
                    
                    if mask.any():
                        mean_L0 = L0[mask].mean(axis=0)
                        mean_L1 = L1[mask].mean(axis=0)
                        mean_mat = jl_flat[mask].mean(axis=0)
                        
                        joint_likelihood_results.append({'likelihood': mean_mat})
                        
                        ax_marg_x.plot(x_vals, mean_L0, color='C0', lw=1.5)
                        ax_marg_x.fill_between(x_vals, mean_L0, color='C0', alpha=0.3)
                        ax_marg_x.set_ylim(0, global_max * 1.05)
                        
                        ax_marg_y.plot(mean_L1, x_vals, color='C1', lw=1.5)
                        ax_marg_y.fill_betweenx(x_vals, mean_L1, color='C1', alpha=0.3)
                        ax_marg_y.set_xlim(0, global_max * 1.05)
                        
                        ax_joint.imshow(mean_mat.T, cmap='viridis', origin='lower', vmin=0, aspect='auto')
                        ax_joint.set_xlim(-0.5, scale + 0.5)
                        ax_joint.set_ylim(-0.5, scale + 0.5)
                    
                    ax_joint.set_xticks([])
                    ax_joint.set_yticks([])
                    for spine in ax_joint.spines.values():
                        spine.set_color('gray')
                        spine.set_alpha(0.3)

            custom_lines = [Line2D([0], [0], color='C0', lw=2), Line2D([0], [0], color='C1', lw=2)]
    plt.show()

def plot_3d_map_surface_all_baselines(agent_ft, agent_echo):
    PLOT_STYLES = {
        'Exact': {'color': 'lightgray', 'linestyle': '--', 'marker': 'D', 'lw': 2, 'mec': 'k'},
        'Factorized': {'color': 'k', 'linestyle': '--', 'marker': 'D', 'lw': 2, 'mec': 'k'},
        'Fully-Trained': {'color': 'C2', 'linestyle': '-', 'lw': .25},
        'Echo-State': {'color': 'C3', 'linestyle': '-', 'lw': .25}
    }
    
    fig = plt.figure(figsize=(16, 8))    
    configs = [
        (1, agent_ft, 'Fully-Trained', 'Blues'),
        (2, agent_echo, 'Echo-State', 'Reds')
    ]
    x_range = np.arange(10)
    y_range = np.arange(10)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    for idx, agent, title, cmap_name in configs:
        ax = fig.add_subplot(1, 2, idx, projection='3d')
        
        onset = 0
        m_j = np.argmax(agent.joint_belief[:, onset:], axis=-1).flatten()
        m_f = np.argmax(agent.naive_belief[:, onset:], axis=-1).flatten()
        m_n = np.argmax(agent.model_belief[:, onset:], axis=-1).flatten()
                
        Z_grid = np.full((10, 10), np.nan)
        Alpha_grid = np.zeros((10, 10))
        
        for r in range(10):
            for c in range(10):
                target_x = X_grid[r, c]
                target_y = Y_grid[r, c]
                
                mask = (m_j == target_x) & (m_f == target_y)
                if np.any(mask):
                    Z_grid[r, c] = np.mean(m_n[mask])
                    Alpha_grid[r, c] = np.mean(np.abs(m_n[mask] == target_x))
        
        surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap=cmap_name, 
                               linewidth=0.5, edgecolors='#555555', alpha=1,
                               vmin=0, vmax=9)
        
        valid_cells = ~np.isnan(Z_grid)
        X_pts = X_grid[valid_cells]
        Y_pts = Y_grid[valid_cells]
        Z_pts = Z_grid[valid_cells]
        alphas = Alpha_grid[valid_cells]
        
        condlist = [alphas >= .5]
        step_alphas = [1.0]   
        step_sizes  = [40]     
        
        chosen_alphas = np.select(condlist, step_alphas, default=0.0)
        chosen_sizes  = np.select(condlist, step_sizes, default=0.0)
        
        render_mask = chosen_alphas > 0.0
        X_pts = X_pts[render_mask]
        Y_pts = Y_pts[render_mask]
        Z_pts = Z_pts[render_mask]
        chosen_alphas = chosen_alphas[render_mask]
        chosen_sizes = chosen_sizes[render_mask]
        
        rgba_colors = np.zeros((len(X_pts), 4))
        rgba_colors[:, 0] = 0.0            
        rgba_colors[:, 1] = 0.55            
        rgba_colors[:, 2] = 0.0            
        rgba_colors[:, 3] = chosen_alphas  
        
        ax.scatter(X_pts, Y_pts, Z_pts, c=rgba_colors, s=chosen_sizes, marker='o', 
                   edgecolors='k', depthshade=False, zorder=20)
        
        ax.plot([], [], 'o', color=(0.0, 0.55, 0.0), ms=5, label="Optimal MAP at least 50% of the time")
        
        ax.set_title(title, fontsize=12, fontweight='semibold', pad=10)
        ax.set_xlabel('Joint MAP (X)', fontsize=10, labelpad=5)
        ax.set_ylabel('Factorized MAP (Y)', fontsize=10, labelpad=5)
        ax.set_zlabel('Mean Network MAP (Z)', fontsize=10, labelpad=5)
        
        ax.set_xlim(-0.01, 9.01)
        ax.set_ylim(-0.01, 9.01)
        ax.set_zlim(-0.01, 9.01)
        
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_zticks(range(10))
        
        ax.view_init(elev=25, azim=-135)
    ax.legend(loc='upper left', frameon=True, fontsize=9)
        
    plt.tight_layout()
    plt.show()

def plot_belief_and_map_heatmaps(agent_ft, agent_echo):
    fig = plt.figure(figsize=(15, 14))
    
    gs_master = gridspec.GridSpec(2, 2, figure=fig, wspace=0.1, hspace=0.25, top=0.88, bottom=0.06, left=0.08, right=0.96)
    
    gs_hist_left = gs_master[0, 0].subgridspec(2, 2, wspace=0.05, hspace=0.15)
    gs_hist_right = gs_master[0, 1].subgridspec(2, 2, wspace=0.05, hspace=0.15)
    gs_map_left = gs_master[1, 0].subgridspec(2, 2, wspace=0.05, hspace=0.15)
    gs_map_right = gs_master[1, 1].subgridspec(2, 2, wspace=0.05, hspace=0.15)

    axs = {}
    for i in range(2):
        for j in range(2):
            axs[f'Hist_UL_{i}{j}'] = fig.add_subplot(gs_hist_left[i, j])
            axs[f'Hist_UR_{i}{j}'] = fig.add_subplot(gs_hist_right[i, j])
            axs[f'Map_LL_{i}{j}'] = fig.add_subplot(gs_map_left[i, j])
            axs[f'Map_LR_{i}{j}'] = fig.add_subplot(gs_map_right[i, j])

    fig.text(0.5, 0.9, 'Belief Covariance', ha='center', fontsize=20)
    fig.text(0.5, 0.44, '2D MAP Histograms', ha='center', fontsize=20)

    axs['Hist_UL_00'].set_title("Uncentered", x=1.025, pad=20, fontsize=15)
    axs['Hist_UR_00'].set_title("Centered", x=1.025, pad=20, fontsize=15)

    def get_soft_histogram(bayes_belief, net_belief):
        # Flatten batch, step, ctx dimensions treating them as samples (N, R)
        R = bayes_belief.shape[-1]
        b_flat = bayes_belief.reshape(-1, R)
        n_flat = net_belief.reshape(-1, R)
        
        # Calculate the sum of outer products (soft joint histogram)
        # N on rows (y-axis), B on cols (x-axis) to match original MAP plot formatting
        return (n_flat.T @ b_flat) / b_flat.shape[0]

    def to_map(belief):
        return np.argmax(belief, axis=-1).flatten()

    j_ft_u = agent_ft.joint_belief[:, -1]
    n_ft_u = agent_ft.naive_belief[:, -1]
    m_ft_u = agent_ft.model_belief[:, -1]
    
    j_echo_u = agent_echo.joint_belief[:, -1]
    n_echo_u = agent_echo.naive_belief[:, -1]
    m_echo_u = agent_echo.model_belief[:, -1]

    j_ft_c = center_beliefs(agent_ft.joint_belief, agent_ft.ctx_vals)[:, -1]
    n_ft_c = center_beliefs(agent_ft.naive_belief, agent_ft.ctx_vals)[:, -1]
    m_ft_c = center_beliefs(agent_ft.model_belief, agent_ft.ctx_vals)[:, -1]
    
    j_echo_c = center_beliefs(agent_echo.joint_belief, agent_echo.ctx_vals)[:, -1]
    n_echo_c = center_beliefs(agent_echo.naive_belief, agent_echo.ctx_vals)[:, -1]
    m_echo_c = center_beliefs(agent_echo.model_belief, agent_echo.ctx_vals)[:, -1]

    hist_pairs = [
        (axs['Hist_UL_00'], j_ft_u, m_ft_u, 'Exact', 'Fully-Trained', False, '00'),
        (axs['Hist_UL_01'], n_ft_u, m_ft_u, 'Factorized', 'Fully-Trained', False, '01'),
        (axs['Hist_UL_10'], j_echo_u, m_echo_u, 'Exact', 'Echo-State', False, '10'),
        (axs['Hist_UL_11'], n_echo_u, m_echo_u, 'Factorized', 'Echo-State', False, '11'),
        
        (axs['Hist_UR_00'], j_ft_c, m_ft_c, 'Exact', 'Fully-Trained', True, '00'),
        (axs['Hist_UR_01'], n_ft_c, m_ft_c, 'Factorized', 'Fully-Trained', True, '01'),
        (axs['Hist_UR_10'], j_echo_c, m_echo_c, 'Exact', 'Echo-State', True, '10'),
        (axs['Hist_UR_11'], n_echo_c, m_echo_c, 'Factorized', 'Echo-State', True, '11')
    ]
    
    map_pairs = [
        (axs['Map_LL_00'], to_map(j_ft_u), to_map(m_ft_u), 'Exact', 'Fully-Trained', False, '00'),
        (axs['Map_LL_01'], to_map(n_ft_u), to_map(m_ft_u), 'Factorized', 'Fully-Trained', False, '01'),
        (axs['Map_LL_10'], to_map(j_echo_u), to_map(m_echo_u), 'Exact', 'Echo-State', False, '10'),
        (axs['Map_LL_11'], to_map(n_echo_u), to_map(m_echo_u), 'Factorized', 'Echo-State', False, '11'),
        
        (axs['Map_LR_00'], to_map(j_ft_c), to_map(m_ft_c), 'Exact', 'Fully-Trained', True, '00'),
        (axs['Map_LR_01'], to_map(n_ft_c), to_map(m_ft_c), 'Factorized', 'Fully-Trained', True, '01'),
        (axs['Map_LR_10'], to_map(j_echo_c), to_map(m_echo_c), 'Exact', 'Echo-State', True, '10'),
        (axs['Map_LR_11'], to_map(n_echo_c), to_map(m_echo_c), 'Factorized', 'Echo-State', True, '11')
    ]

    custom_cmap = LinearSegmentedColormap.from_list(
        'ContinuousBlackToBlue', 
        ['#000000', '#0d2b56', '#1a5292', '#3b82f6', '#93c5fd', '#ffffff']
    )

    # --------------------------------------------------------------------------
    # RENDER CONTINUOUS BELIEF HISTOGRAMS (Soft Joint Probabilities)
    # --------------------------------------------------------------------------
    for ax, bayes_b, net_b, xl, yl, centered, key in hist_pairs:
        soft_hist = get_soft_histogram(bayes_b, net_b)
        
        norm = PowerNorm(gamma=.7, vmin=0, vmax=np.max(soft_hist) * 0.45)
        ax.imshow(soft_hist, origin='lower', cmap=custom_cmap, norm=norm, aspect='equal')
        
        ax.plot([-0.5, 9.5], [-0.5, 9.5], color='gray', ls='--', alpha=0.5)

        if centered:
            ax.add_patch(patches.Rectangle((4.5, 4.5), 1, 1, fill=False, edgecolor='red', linewidth=1.5))
            ax.text(
                5.0, 5.0, r'$r^{\bigstar}_g$', color='red', fontsize=13,
                fontweight='bold', ha='center', va='center'
            )
            ax_lims = (0.5, 9.5)
            tick_vals = np.arange(1, 10)
            tick_labels = [f"+{i - 5}" if i > 5 else (r"$r^{\bigstar}_g$" if i == 5 else f"{i - 5}") for i in tick_vals]
        else:
            ax_lims = (-0.5, 9.5)
            tick_vals = np.arange(10)
            tick_labels = tick_vals

        ax.set_xlim(ax_lims)
        ax.set_ylim(ax_lims)
        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)

        # X-Labels for Bottom Rows
        if key in ['10', '11']:
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel(f"{xl}")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel('')

        # Y-Labels for Far-Left columns
        if key in ['00', '10']:
            ax.set_yticklabels(tick_labels)
            ax.set_ylabel(f"{yl}")
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')

    # --------------------------------------------------------------------------
    # RENDER MAP AGREEMENT HISTOGRAMS
    # --------------------------------------------------------------------------
    for ax, bayes_m, net_m, xl, yl, centered, key in map_pairs:
        H, _, _ = np.histogram2d(bayes_m, net_m, bins=np.arange(-0.5, 10.5, 1))
        
        norm = PowerNorm(gamma=.7, vmin=0, vmax=np.max(H) * 0.45)
            
        ax.imshow(H.T, origin='lower', cmap=custom_cmap, norm=norm, aspect='equal')
        ax.plot([-0.5, 9.5], [-0.5, 9.5], color='gray', ls='--', alpha=0.5)

        if centered:
            ax.add_patch(patches.Rectangle((4.5, 4.5), 1, 1, fill=False, edgecolor='red', linewidth=1.5))
            ax.text(
                5.0, 5.0, r'$r^{\bigstar}_g$', color='red', fontsize=13,
                fontweight='bold', ha='center', va='center'
            )
            ax_lims = (0.5, 9.5)
            tick_vals = np.arange(1, 10)
            tick_labels = [f"+{i - 5}" if i > 5 else (r"$r^{\bigstar}_g$" if i == 5 else f"{i - 5}") for i in tick_vals]
        else:
            ax_lims = (-0.5, 9.5)
            tick_vals = np.arange(10)
            tick_labels = tick_vals

        ax.set_xlim(ax_lims)
        ax.set_ylim(ax_lims)
        ax.set_xticks(tick_vals)
        ax.set_yticks(tick_vals)

        # X-Labels for bottom of the MAP Grid 
        if key in ['10', '11']:
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel(f"{xl}")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel('')

        # Y-Labels for far-left column of MAP Grid 
        if key in ['00', '10']:
            ax.set_yticklabels(tick_labels)
            ax.set_ylabel(f"{yl}")
        else:
            ax.set_yticklabels([])
            ax.set_ylabel('')

    plt.show()

if __name__ == "__main__":
    cuda = 1
    obs_num = 5
    state_num = 500
    realization_num = 10
    batch_num = 20000
    step_num = 30
    ctx_num = 2
    h = 1000

    base_kwargs = {
        'mode':"sanity",'cuda':cuda,'episodes':1,'show_plots':False,
        'obs_num':obs_num,'training':False,'batch_num':batch_num,
        'step_num':step_num,'learn_embeddings':False,
        'realization_num':realization_num,'state_num':state_num
    }

    agent_ft = CognitiveGridworld(**base_kwargs, hid_dim=h, ctx_num=ctx_num, load_env=f"/sanity/fully_trained_ctx_{ctx_num}")
    agent_echo = CognitiveGridworld(**base_kwargs, hid_dim=h, ctx_num=ctx_num, load_env=f"/sanity/reservoir_ctx_{ctx_num}")

    agent_ft.prep_data_manager()
    agent_ft.episode_loop(disable_tqdm=True)

    agent_echo.prep_data_manager()
    agent_echo.episode_loop(disable_tqdm=True)

    # Plot Summaries and Heatmaps
    j_ft = center_beliefs(agent_ft.joint_belief, agent_ft.ctx_vals)
    n_ft = center_beliefs(agent_ft.naive_belief, agent_ft.ctx_vals)
    ft = center_beliefs(agent_ft.model_belief, agent_ft.ctx_vals)

    j_echo = center_beliefs(agent_echo.joint_belief, agent_echo.ctx_vals)
    n_echo = center_beliefs(agent_echo.naive_belief, agent_echo.ctx_vals)
    echo = center_beliefs(agent_echo.model_belief, agent_echo.ctx_vals)

    plot_summary_and_heatmaps(
        j_ft[:, -1], n_ft[:, -1], ft[:, -1], echo[:, -1], 
        j_ft, n_ft, ft, echo, 
        realization_num, step_num,
        j_ft, n_ft, ft, j_echo, n_echo, echo
    )
    plot_combined_analysis(agent_ft, agent_echo, target_zs= [0.17, 0.37], tolerance=.03, both = True, bins= 100, over_ctx = "max", w = 11, h = 6)
    plot_combined_analysis(agent_ft, agent_echo, target_zs= [0.17, 0.37], tolerance=.03, both = True, bins = [.005, .0075, .02, .04, .06, .08, .1, .12], w = 11, h = 6)
    
    plot_3d_map_surface_all_baselines(agent_ft, agent_echo)
    plot_map_trajectories(agent_ft, agent_echo, n_examples=6)
    
    # Unified 4x4 Array: Belief Histograms & MAP Agreement
    plot_belief_and_map_heatmaps(agent_ft, agent_echo)