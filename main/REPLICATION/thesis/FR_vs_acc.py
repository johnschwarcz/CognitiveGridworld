import numpy as np
import os
import sys
import inspect
import matplotlib.pyplot as plt
from scipy import stats
import itertools
from tqdm import tqdm
import pickle

# Setup Paths
path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("root:", path)

sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')

from main.CognitiveGridworld import CognitiveGridworld

# Global Font and Sizing Styles
plt.rcParams.update({'font.family':'serif','font.size':13,'axes.labelsize':13,'axes.titlesize':16,'legend.fontsize':15})

# Define Global Styles
PLOT_STYLES = {
    'Exact': {'color': 'lightgray', 'marker': 'D', 'lw':4, 'mec':'gray', 'ls': '--'},
    'Factorized': {'color': 'k', 'marker': 'D', 'lw':4, 'mec':'k', 'ls': '--'},
    'Fully-Trained': {'color': 'C2', 'marker': 'o', 'lw':2, 'mec': 'C2','ls': ':'},
    'Echo-State': {'color': 'C3', 'marker': 'o', 'lw':2, 'mec': 'C3', 'ls':':'}
}

def analyze_likelihood_sets(data, tolerance=1e-5):
    """
    Analyzes sets of likelihoods to find unique sets and duplicates across batches.
    Order of observations MATTERS (no sorting along the observation dimension).
    """
    data_np = np.array(data)
    decimals = int(np.ceil(-np.log10(tolerance)))
    
    if decimals <= 15:
        final_data = np.round(data_np, decimals=decimals)
    else:
        final_data = data_np
    
    if final_data.ndim > 1:
        batch_size = final_data.shape[0]
        final_data = final_data.reshape(batch_size, -1)
    else:
        final_data = final_data.reshape(1, -1)

    unique_sets, inverse_indices, counts = np.unique(final_data, axis=0, return_counts=True, return_inverse=True)
    num_unique = len(unique_sets)
    num_duplicates = len(final_data) - num_unique
    
    return num_unique, num_duplicates, unique_sets, counts, inverse_indices


def build_and_cache_raw_dataset(agent_ft, agent_echo, tolerance=1e-20, min_repetitions=100, n_samples=25, cache_file='raw_data_cache.pkl'):
    """
    Runs simulations and aggregates the raw X/Y data. 
    Saves PURE data. No statistical analysis or bootstrapping occurs here.
    """
    kl_eps = 1e-10 
    model_data = {m: {'x': [], 'y': []} for m in PLOT_STYLES.keys()}

    configs = [
        (agent_ft, 'Fully-Trained'),
        (agent_echo, 'Echo-State')
    ]

    print("\n" + "="*60)
    print(f"AGGREGATING RAW DATA ({n_samples} FULL-BATCH SIMULATIONS)")
    print("="*60)

    for i in tqdm(range(n_samples), desc="Simulating Environments", unit="run"):
        for agent, _ in configs:
            if agent is not None:
                agent.prep_data_manager()
                agent.episode_loop(disable_tqdm=True)

        for agent, name in configs:
            if agent is None:
                continue
            
            j_like = agent.joint_likelihood
            j_bel = agent.joint_belief
            n_bel = agent.naive_belief
            m_bel = agent.model_belief
            c_vals = agent.ctx_vals

            # Compute KL Divergence (per context)
            P = j_bel[:, -1]
            Q = n_bel[:, -1]
            kl_context = np.sum(P * np.log((P + kl_eps) / (Q + kl_eps)), axis=-1) 

            # Lazy Context Accuracies (per context)
            acc_dict = {}
            if name == 'Fully-Trained':
                models_to_extract = ['Exact', 'Fully-Trained']
                acc_dict['Exact'] = (np.argmax(j_bel[:, -1], -1) == c_vals).astype(float)
                acc_dict['Fully-Trained'] = (np.argmax(m_bel[:, -1], -1) == c_vals).astype(float)
            else:
                models_to_extract = ['Factorized', 'Echo-State']
                acc_dict['Factorized'] = (np.argmax(n_bel[:, -1], -1) == c_vals).astype(float)
                acc_dict['Echo-State'] = (np.argmax(m_bel[:, -1], -1) == c_vals).astype(float)

            # Find duplicates
            _, _, _, counts, inv_indices = analyze_likelihood_sets(j_like, tolerance)
            dup_groups = np.where(counts >= min_repetitions)[0]

            for g_idx in dup_groups:
                mask = (inv_indices == g_idx)
                group_kl_means = kl_context[mask].mean(axis=0)

                for model in models_to_extract:
                    mean_vals = acc_dict[model][mask].mean(axis=0)
                    model_data[model]['x'].extend(group_kl_means.tolist())
                    model_data[model]['y'].extend(mean_vals.tolist())

    # Save purely raw data to disk
    with open(cache_file, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"\n[+] Raw data successfully cached to {cache_file}")


def analyze_and_render_plot(cache_file='raw_data_cache.pkl', n_bootstraps=1000):
    """
    Loads raw data from disk, performs Bivariate statistical analysis (Bootstrapped Pearson r),
    and renders the matplotlib UI with a simplified, unified statistical bracket.
    """
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file {cache_file} not found. Please run with REBUILD_DATA = True first.")

    with open(cache_file, 'rb') as f:
        model_data = pickle.load(f)

    correlation_distributions = {m: [] for m in PLOT_STYLES.keys()}
    global_fits = {}

    print("\n" + "="*60)
    print("ANALYZING RAW DATA & BOOTSTRAPPING CORRELATIONS")
    print("="*60)

    # --- STATISTICAL ANALYSIS PHASE ---
    for model in ['Exact', 'Fully-Trained', 'Factorized', 'Echo-State']:
        data = model_data[model]
        if not data['x']:
            continue

        x_arr = np.array(data['x'])
        y_arr = np.array(data['y'])

        if len(x_arr) > 1 and np.std(x_arr) > 0 and np.std(y_arr) > 0:
            # Global Pearson Correlation and Linear Regression (for plotting the trendline)
            r, p_val = stats.pearsonr(x_arr, y_arr)
            res = stats.linregress(x_arr, y_arr)
            
            global_fits[model] = {
                'r': r, 
                'p_val': p_val,
                'm': res.slope,
                'b': res.intercept,
                'n_points': len(x_arr)
            }
            print(f"--- Model: {model} (N={len(x_arr)} points) ---")
            print(f"Global Correlation (r) : {r:.5f} (p={p_val:.3e})")

        # Execute on-the-fly bootstrapping using np.corrcoef for massive speedups
        if len(x_arr) >= 3 and np.std(x_arr) > 0 and np.std(y_arr) > 0:
            for _ in tqdm(range(n_bootstraps), desc=f"Bootstrapping {model}", leave=False):
                indices = np.random.choice(len(x_arr), size=len(x_arr), replace=True)
                x_boot, y_boot = x_arr[indices], y_arr[indices]

                if np.std(x_boot) > 0 and np.std(y_boot) > 0:
                    r_boot = np.corrcoef(x_boot, y_boot)[0, 1]
                    correlation_distributions[model].append(r_boot)

    print("\nRendering plots...")

    # --- PLOTTING PHASE ---
    fig, (ax_scatter, ax_dist) = plt.subplots(1, 2, figsize=(17, 7))

    # --- PANEL 1: SCATTER & TRENDLINES ---
    for model in ['Exact', 'Fully-Trained', 'Factorized', 'Echo-State']:
        if not model_data[model]['x']:
            continue

        x_arr = np.array(model_data[model]['x'])
        y_arr = np.array(model_data[model]['y'])
        style = PLOT_STYLES[model]
        marker = style.get('marker', 'o')
        mec = style.get('mec', 'white') if model in ['Fully-Trained', 'Echo-State'] else style.get('mec', 'k')

        # Draw Linear Fit Line
        if model in global_fits:
            fit = global_fits[model]
            label_str = f"{model} ($r$={fit['r']:.2f})"
            
            x_range = np.linspace(min(x_arr), max(x_arr), 100)
            y_fit = fit['m'] * x_range + fit['b']
            ax_scatter.plot(x_range, y_fit, color='w', linestyle='-', lw=8, alpha=1, zorder=90)
            ax_scatter.plot(x_range, y_fit, color=style['color'], linestyle=style['ls'], lw=3, alpha=1, zorder=100)
        else:
            label_str = model

        # Scatter
        ax_scatter.scatter(x_arr, y_arr, marker=marker, color=style['color'],
                           edgecolors='k', alpha=0.5, s=25, label=label_str, zorder=10)

    # Panel 1 Formatting
    ax_scatter.set_title("Factorization Regret vs Accuracy")
    ax_scatter.set_xlabel("Factorization Regret")
    ax_scatter.set_ylabel("Accuracy")
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['right'].set_visible(False)
    ax_scatter.grid(alpha=0.2, ls=':')
    ax_scatter.set_ylim([-.1,.95])
    handles, labels = ax_scatter.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        leg = ax_scatter.legend(by_label.values(), by_label.keys(), frameon=True, loc='best', ncol = 2)
        for lh in leg.legend_handles: 
            lh.set_alpha(1.0) 

    # --- PANEL 2: KDE DISTRIBUTIONS & UNIFIED STATS BRACKET ---
    valid_models = [m for m in correlation_distributions.keys() if len(correlation_distributions[m]) > 1]
    
    max_kde_y = 0
    dist_means = {}

    for model in valid_models:
        corrs = np.array(correlation_distributions[model])
        style = PLOT_STYLES[model]
        dist_means[model] = np.mean(corrs)

        if np.std(corrs) > 0:
            kde = stats.gaussian_kde(corrs)
            x_vals = np.linspace(max(-1, min(corrs) - np.std(corrs)), min(1, max(corrs) + np.std(corrs)), 200)
            y_vals = kde(x_vals)

            y_vals = y_vals / y_vals.sum()

            max_kde_y = max(max_kde_y, max(y_vals))
            
            ax_dist.plot(x_vals, y_vals, color=style['mec'], lw=2, ls = style['ls'])
            ax_dist.fill_between(x_vals, y_vals, alpha=0.2, color=style['color'])
        else:
            ax_dist.axvline(np.mean(corrs), color=style['color'], lw=2)

    # --- VISUAL POLISH: Unified Bracket Drawing Logic ---
    group_1 = [m for m in ['Factorized', 'Echo-State'] if m in valid_models]
    group_2 = [m for m in ['Exact', 'Fully-Trained'] if m in valid_models]

    if group_1 and group_2:
        # Calculate visual centers of the two clusters
        x1 = np.mean([dist_means[m] for m in group_1])
        x2 = np.mean([dist_means[m] for m in group_2])

        # Pool bootstrapped correlations for a unified statistical test
        boot_1 = np.concatenate([correlation_distributions[m] for m in group_1])
        boot_2 = np.concatenate([correlation_distributions[m] for m in group_2])

        min_len = min(len(boot_1), len(boot_2))
        diffs = boot_1[:min_len] - boot_2[:min_len]
        
        if len(diffs) > 0:
            p_val = 2 * min(np.mean(diffs > 0), np.mean(diffs < 0))
            
            if p_val < 0.001: sig_flag = "***"
            elif p_val < 0.01: sig_flag = "**"
            elif p_val < 0.05: sig_flag = "*"
            else: sig_flag = "n.s."
            
            h = max_kde_y * 1.15
            tick_len = max_kde_y * 0.05
            
            # Draw unified black bracket bridging the two clusters
            ax_dist.plot([x1, x2], [h, h], lw=1.2, color='black', zorder=2)
            ax_dist.plot([x1, x1], [h, h - tick_len], lw=1.2, color='black', zorder=2)
            ax_dist.plot([x2, x2], [h, h - tick_len], lw=1.2, color='black', zorder=2)
            
            # Place significance text
            ax_dist.text((x1 + x2) / 2, h + (max_kde_y * 0.02), sig_flag, ha='center', va='bottom', fontsize=14, color='black')

        ax_dist.set_ylim(bottom=0, top=h + (max_kde_y * 0.15))

    # Panel 2 Formatting
    ax_dist.set_title("Distribution of coefficients ($r$)")
    ax_dist.set_xlabel("Pearson Correlation Coefficient ($r$)")
    ax_dist.set_ylabel("Probability")
    ax_dist.spines['top'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)
    ax_dist.grid(alpha=0.2, ls=':')

    plt.tight_layout(w_pad=4.0)
    plt.savefig("FR_vs_acc.svg")
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
        'obs_num':obs_num,'training':False,'batch_num':batch_num, 'skip_training_analyses': True,
        'step_num':step_num,'learn_embeddings':False, 'subsample_states': 25, 
        'realization_num':realization_num,'state_num':state_num
    }

    # =========================================================================
    # WORKFLOW TOGGLE
    # Set to True  -> Runs environments and saves RAW data to cache
    # Set to False -> Loads raw data, runs bootstraps, and renders plot
    # =========================================================================
    REBUILD_DATA = False
    CACHE_FILENAME = 'raw_data_cache.pkl'

    if REBUILD_DATA:
        agent_ft = CognitiveGridworld(**base_kwargs, hid_dim=h, ctx_num=ctx_num, load_env=f"/sanity/fully_trained_ctx_{ctx_num}")
        agent_echo = CognitiveGridworld(**base_kwargs, hid_dim=h, ctx_num=ctx_num, load_env=f"/sanity/reservoir_ctx_{ctx_num}")
        
        build_and_cache_raw_dataset(
            agent_ft, agent_echo, 
            tolerance=1e-20, min_repetitions=100, 
            n_samples=25, 
            cache_file=CACHE_FILENAME
        )
    
    # ---------------------------------------------------------
    # --- ON-THE-FLY STATS & RENDERING FROM RAW CACHE       ---
    # ---------------------------------------------------------
    analyze_and_render_plot(cache_file=CACHE_FILENAME, n_bootstraps=15000)