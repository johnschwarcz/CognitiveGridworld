import numpy as np
import os
import sys
import inspect
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

# Setup Paths
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname(os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')

from main.CognitiveGridworld import CognitiveGridworld 

# Setup Plotting Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'axes.labelsize': 13,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
})

def plot_network_overlay(bayes_joint, bayes_naive, net_trained, net_echo, T, ctxs=1, reps=1):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, ctxs))
    t = np.arange(T)
    idx = np.linspace(0, T-1, 4, dtype=int)

    # Style mappings for Bayes (cond=0) vs Network (cond=1)
    markers = ['o', '^']
    mecs = ['k', 'r']
    linestyles = ['-', '--'] 
    ls_rel = [':', '--']

    for cond in range(2):
        # Select the correct data based on the condition
        mj = bayes_joint if cond == 0 else net_trained
        mn = bayes_naive if cond == 0 else net_echo

        for c in range(ctxs):
            # Average over repetitions
            m_j_c = mj[c].mean(0)
            m_n_c = mn[c].mean(0)

            # 1: Panel 1 - Experts (Exact & Trained)
            axs[0].plot(t, m_j_c, c=colors[c], lw=2.5, ls=linestyles[cond])
            axs[0].scatter(t[idx], m_j_c[idx], s=50, marker=markers[cond],
                           facecolors=colors[c], edgecolors=mecs[cond], zorder=3)

            # 2: Panel 2 - Baselines (Factorized & Echo-state)
            axs[1].plot(t, m_n_c, c=colors[c], lw=2.5, ls=linestyles[cond])
            axs[1].scatter(t[idx], m_n_c[idx], s=50, marker=markers[cond],
                           facecolors=colors[c], edgecolors=mecs[cond], zorder=3)

            # 3: Panel 3 - Relative Accuracy
            axs[2].plot(m_n_c, m_j_c, c=colors[c], lw=2.5, ls=ls_rel[cond], zorder=2)
            axs[2].scatter(m_n_c[idx], m_j_c[idx], s=50, marker=markers[cond],
                           facecolors=colors[c], edgecolors=mecs[cond], zorder=3)

            # Text Annotations for multipliers
            text_col = 'k' if cond == 0 else 'r'
            axs[2].annotate(r"$\times$" + f'{m_j_c[-1]/m_n_c[-1]:.0f}', 
                            (m_n_c[-1], m_j_c[-1]),
                            xytext=(-20, 10), textcoords='offset points', color=text_col)

            # Start annotation only on Bayes (cond=0)
            if cond == 0 and c == 0:
                axs[2].annotate('start', (m_n_c[0], m_j_c[0]), xytext=(-25, -30), textcoords='offset points')

    # Formatting and reference line
    axs[2].plot([0.1, 1.0], [0.1, 1.0], ls='-', c='gray', lw=1, zorder=1)

    axs[0].set(title="Experts", xlabel='Inference Time', ylabel='Accuracy', ylim=(0.0, 1.1))
    axs[1].set(title="Baselines", xlabel='Inference Time', ylim=(0.0, 1.1))
    axs[2].set(title="Relative Accuracy", xlabel='Factorized / Echo-state', ylabel='Exact / Trained', xlim=(0.1, 1.0), ylim=(0.1, 1.0))

    for ax in axs:
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ---------------- Legends ----------------
    
    # Panel 1: Context Colors (Bottom Right)
    c_handles = [Line2D([0], [0], color=colors[c], lw=2.5, label=f'$C={c+1}$') for c in range(ctxs)]
    leg_c = axs[0].legend(handles=c_handles, loc='lower right')
    axs[0].add_artist(leg_c)

    # Panel 1: Experts Legend (Upper Left)
    expert_handles = [
        Line2D([0], [0], color='gray', marker='o', markeredgecolor='k', markerfacecolor='none', markersize=8, ls='-', label='Exact'),
        Line2D([0], [0], color='gray', marker='^', markeredgecolor='r', markerfacecolor='none', markersize=8, ls='--', label='Trained')
    ]
    axs[0].legend(handles=expert_handles, loc='upper left')

    # Panel 2: Baselines Legend (Upper Left)
    base_handles = [
        Line2D([0], [0], color='gray', marker='o', markeredgecolor='k', markerfacecolor='none', markersize=8, ls='-', label='Factorized'),
        Line2D([0], [0], color='gray', marker='^', markeredgecolor='r', markerfacecolor='none', markersize=8, ls='--', label='Echo-state')
    ]
    axs[1].legend(handles=base_handles, loc='upper left')

    # Panel 3: Relative Math Legend (Lower Right)
    rel_handles = [
        Line2D([0], [0], color='none', marker='o', markeredgecolor='k', markerfacecolor='none', markersize=10, label=r"$\frac{\text{Exact}}{\text{Factorized}}$"),
        Line2D([0], [0], color='none', marker='^', markeredgecolor='r', markerfacecolor='none', markersize=10, label=r"$\frac{\text{Trained}}{\text{Echo-state}}$")
    ]
    axs[2].legend(handles=rel_handles, loc='lower right', fontsize=16)

    plt.savefig("thesis_fig3.svg", dpi=300)
    plt.show()


if __name__ == "__main__":
    cuda = 0
    obs_num = 5 
    state_num = 500  
    realization_num = 10 

    batch_num = 20
    step_num = 30 
    ctxs = 3
    reps = 5000 
    T = step_num
    
    # Pre-allocate arrays holding data before averaging the batch dimension
    j_acc, n_acc, ft_acc, echo_acc = [np.zeros((ctxs, reps, batch_num, T)) for _ in range(4)]
    # Base kwargs for network mode
    base_kwargs = {
        'mode': "sanity", 'cuda': cuda, 'episodes': 1, 
        'show_plots': False, 'obs_num': obs_num, 'training': False,
        'batch_num': batch_num, 'step_num': step_num, 'learn_embeddings': False,
        'realization_num': realization_num, 'state_num': state_num
    }
    
    for c in range(ctxs):
        h = 1000 if c < 2 else 5000 
        # 1. Initialize Trained Agent
        agent_ft = CognitiveGridworld(**base_kwargs,
            hid_dim=h, ctx_num=c+1, load_env=f"/sanity/fully_trained_ctx_{c+1}")
        # 2. Initialize Echo-state (Reservoir) Agent
        agent_echo = CognitiveGridworld(**base_kwargs,
            hid_dim=h, ctx_num=c+1, load_env=f"/sanity/reservoir_ctx_{c+1}") 
        for r in tqdm(range(reps), desc=f'Context {c+1}'):
            agent_ft.prep_data_manager()
            agent_ft.episode_loop(disable_tqdm=True)        
            agent_echo.prep_data_manager()
            agent_echo.episode_loop(disable_tqdm=True)                        
            # Store Bayes Accuracies (identical across both networks, so we grab from the first one)
            j_acc[c, r] = agent_ft.joint_acc
            n_acc[c, r] = agent_ft.naive_acc
            # Store Network Accuracies 
            ft_acc[c, r] = agent_ft.model_acc
            echo_acc[c, r] = agent_echo.model_acc
    # Mean over the batch dimension (axis=2) before passing to the plotting function
    j_acc_mean = j_acc.mean(axis=2)
    n_acc_mean = n_acc.mean(axis=2)
    ft_acc_mean = ft_acc.mean(axis=2)
    echo_acc_mean = echo_acc.mean(axis=2)
    plot_network_overlay(j_acc_mean, n_acc_mean, ft_acc_mean, echo_acc_mean, T, ctxs=ctxs, reps=reps)