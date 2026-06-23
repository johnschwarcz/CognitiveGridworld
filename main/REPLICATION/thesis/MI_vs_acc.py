import numpy as np
import os
import sys
import inspect
import matplotlib.pyplot as plt

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
    'Factorized': {'color': 'k', 'linestyle': '-', 'marker': 'D', 'lw':4, 'mec':'w'},
    'Fully-Trained': {'color': 'C2', 'linestyle': '-', 'lw':2},
    'Echo-State': {'color': 'C3', 'linestyle': '-', 'lw':2}
}

def analyze_likelihood_sets(data, tolerance=1e-5):
    """
    Analyzes sets of likelihoods to find unique sets and duplicates across batches.
    Order of observations MATTERS (no sorting along the observation dimension).
    """
    data_np = np.array(data)
    decimals = int(np.ceil(-np.log10(tolerance)))
    rounded_data = np.round(data_np, decimals=decimals)
    
    # Flatten everything past the batch dimension directly, preserving original sequence order
    if rounded_data.ndim > 1:
        batch_size = rounded_data.shape[0]
        final_data = rounded_data.reshape(batch_size, -1)
    else:
        final_data = rounded_data.reshape(1, -1)

    unique_sets, inverse_indices, counts = np.unique(final_data, axis=0, return_counts=True, return_inverse=True)
    num_unique = len(unique_sets)
    num_duplicates = len(final_data) - num_unique
    
    return num_unique, num_duplicates, unique_sets, counts, inverse_indices

def plot_agents_internal_duplicates(agent_ft, agent_echo, tolerance=1e-5, min_repetitions=2):
    """
    Plots internal duplicate groups for both agents side-by-side in a 1x2 panel layout.
    Panel 1 X-axis: Average Mutual Information I(R1; R2) (over observations).
    Panel 2 X-axis: Average KL Divergence (Joint || Naive Belief).
    Y-axis: Accuracy (avg. over steps and context)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_mi, ax_kl = axes
    
    eps = tolerance
    kl_eps = 1e-10 # Distinct epsilon for KL calculation to prevent log(0)

    # Dictionaries to store (x, y) points per model for regression calculation
    model_points_mi = {m: {'x': [], 'y': []} for m in PLOT_STYLES.keys()}
    model_points_kl = {m: {'x': [], 'y': []} for m in PLOT_STYLES.keys()}

    configs = [
        (agent_ft, 'Fully-Trained'),
        (agent_echo, 'Echo-State')
    ]

    for agent, name in configs:
        # Note: If the agent is None (e.g. commented out), this loop will fail. Ensure agents are instantiated.
        if agent is None:
            continue
            
        _, _, _, counts, inv_indices = analyze_likelihood_sets(agent.joint_likelihood, tolerance)
        dup_groups = np.where(counts >= min_repetitions)[0]
        
        if len(dup_groups) == 0:
            for ax in axes:
                ax.text(0.5, 0.5, f"No duplicate groups found\n(min_reps={min_repetitions})",
                             ha='center', va='center', transform=ax.transAxes)
                ax.grid(alpha=0.3, ls=':')
            ax_mi.set_xlabel("Mutual Information ($R_1, R_2$)")
            ax_kl.set_xlabel("KL Divergence (Joint || Naive)")
            continue

        # --- Compute Mutual Information profile across observations ---
        jl = agent.joint_likelihood
        p_xy = jl / (np.sum(jl, axis=(-2, -1), keepdims=True) + eps)
        p_x_kd = np.sum(p_xy, axis=-1, keepdims=True)
        p_y_kd = np.sum(p_xy, axis=-2, keepdims=True)
        mi = np.sum(p_xy * np.log((p_xy + eps) / (p_x_kd * p_y_kd + eps)), axis=(-2, -1))
        mi_flat = np.mean(mi, axis=1) # shape: batch

        # --- Compute KL Divergence (Joint vs Naive Belief) ---
        P = agent.joint_belief
        Q = agent.naive_belief
        kl = np.sum(P * np.log((P + kl_eps) / (Q + kl_eps)), axis=-1)
        # kl_flat = np.mean(kl, axis=(1,2)) # shape: batch
        kl_flat = np.mean(kl[:, -1], axis=-1) # shape: batch

        # Compute Context Accuracies (collapsed to batch dimension)
        acc_exact = (np.argmax(agent.joint_belief[:, -1], -1) == agent.ctx_vals).mean(axis=(1))
        acc_fact = (np.argmax(agent.naive_belief[:, -1], -1) == agent.ctx_vals).mean(axis=(1))
        acc_model = (np.argmax(agent.model_belief[:, -1], -1) == agent.ctx_vals).mean(axis=(1))
        models_to_plot = ['Exact' if name == 'Fully-Trained' else "Factorized", name]

        for i, g_idx in enumerate(dup_groups):
            mask = (inv_indices == g_idx)
            group_mi_mean = mi_flat[mask].mean()
            group_kl_mean = kl_flat[mask].mean()
            n_samples = np.sum(mask)

            data_map = {
                'Exact': acc_exact[mask],
                'Factorized': acc_fact[mask],
                name: acc_model[mask]
            }
            
            for model in models_to_plot:
                data_y = data_map[model]
                mean_val = data_y.mean()
                y_err = data_y.std() / np.sqrt(n_samples)

                # Store points for regression
                model_points_mi[model]['x'].append(group_mi_mean)
                model_points_mi[model]['y'].append(mean_val)
                
                model_points_kl[model]['x'].append(group_kl_mean)
                model_points_kl[model]['y'].append(mean_val)

                style = PLOT_STYLES[model]
                marker = style.get('marker', 'o')
                mec = style.get('mec', 'white') if model in ['Fully-Trained', 'Echo-State'] else style.get('mec', 'k')

                # Render points and vertical error lines for MI Plot
                ax_mi.errorbar(group_mi_mean, mean_val, yerr=y_err,
                             fmt=marker, color=style['color'], markeredgecolor=mec,
                             alpha=0.6, capsize=0, markersize=6, elinewidth=.5,
                             label=model if i == 0 else None)
                             
                # Render points and vertical error lines for KL Plot
                ax_kl.errorbar(group_kl_mean, mean_val, yerr=y_err,
                             fmt=marker, color=style['color'], markeredgecolor=mec,
                             alpha=0.6, capsize=0, markersize=6, elinewidth=.5,
                             label=model if i == 0 else None)

    # Plot Linear Regression Fits helper function
    def plot_regressions(ax, points_dict):
        for model, data in points_dict.items():
            if len(data['x']) > 1: # Require at least 2 points
                x_arr = np.array(data['x'])
                y_arr = np.array(data['y'])
                
                m, b = np.polyfit(x_arr, y_arr, 1)
                x_range = np.linspace(min(x_arr), max(x_arr), 100)
                y_fit = m * x_range + b
                
                ax.plot(x_range, y_fit, color=PLOT_STYLES[model]['color'], linestyle='--', lw=1, alpha=1, zorder=100)

    # Apply regressions to both plots
    # plot_regressions(ax_mi, model_points_mi)
    # plot_regressions(ax_kl, model_points_kl)

    # Format MI Axis
    ax_mi.set_xlabel("Mutual Information ($R_1, R_2$) (avg. over observations)")
    ax_mi.set_ylabel("Accuracy (avg. over context)")
    ax_mi.grid(alpha=0.3, ls=':')
    
    # Format KL Axis
    ax_kl.set_xlabel("Factorization Regret (avg. over context)")
    ax_kl.grid(alpha=0.3, ls=':')
    ax_kl.set_yticks([])
    # Deduplicate legends cleanly for both subplots
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), frameon=True, loc='best', ncol=2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cuda = 0
    obs_num = 5
    state_num = 500
    realization_num = 10
    batch_num = 25000
    step_num = 30
    ctx_num = 2
    h = 1000

    base_kwargs = {
        'mode':"sanity",'cuda':cuda,'episodes':1,'show_plots':False,
        'obs_num':obs_num,'training':False,'batch_num':batch_num,
        'step_num':step_num,'learn_embeddings':False, 'test_states': 20,
        'realization_num':realization_num,'state_num':state_num
    }

    agent_ft = CognitiveGridworld(**base_kwargs, hid_dim=h, ctx_num=ctx_num, load_env=f"/sanity/fully_trained_ctx_{ctx_num}")
    agent_echo = CognitiveGridworld(**base_kwargs, hid_dim=h, ctx_num=ctx_num, load_env=f"/sanity/reservoir_ctx_{ctx_num}")
    agent_ft.prep_data_manager()
    agent_ft.episode_loop(disable_tqdm=True)
    agent_echo.prep_data_manager()
    agent_echo.episode_loop(disable_tqdm=True)

    # ---------------------------------------------------------
    # --- UNIFIED 1X2 INTERNAL DUPLICATE PLOTS                ---
    # ---------------------------------------------------------
    plot_agents_internal_duplicates(agent_ft, agent_echo, tolerance=1e-20, min_repetitions=100)
    # ---------------------------------------------------------