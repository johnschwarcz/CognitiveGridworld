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

    plt.savefig("network_overlay.svg", dpi=300)
    plt.show()


def plot_training_convergence(ft_test_accs, echo_test_accs, echo_k_test_accs, joint_acc, naive_acc):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': (1, 1.5)}, tight_layout=True)

    # Extract the final-step theoretical bounds across the batch dimension
    opt_bound = joint_acc.mean(0)[-1]
    fact_bound = naive_acc.mean(0)[-1]

    # Clean color palette
    c_ft = '#2c7bb6'       # Distinct Blue for Trained
    c_echo = '#d7191c'     # Distinct Red for baseline Echo-state
    c_echo_k = plt.cm.magma(np.linspace(0.4, 0.8, 3)) # Gradient for scaling dimensions

    # --- Panel 1: Standard comparison ---
    ax[0].plot(ft_test_accs[:, -1], color=c_ft, lw=2.5, label=r'Trained ($N=1k$)')
    ax[0].plot(echo_test_accs[:, -1], color=c_echo, lw=2.5, label=r'Echo-state ($N=1k$)')

    # --- Panel 2: Scaling Echo-state ---
    labels_k = [r'Echo-state ($N=2k$)',
                r'Echo-state ($N=5k$)',
                r'Echo-state ($N=10k$)']
    for i in range(3):
        ax[1].plot(echo_k_test_accs[i][:, -1], color=c_echo_k[i], lw=1.5, label=labels_k[i])

    # --- Global Formatting for both panels ---
    for i in range(2):
        # Add the theoretical bounds as reference lines
        ax[i].axhline(opt_bound, c='k', lw=2, ls=':', zorder=10)
        ax[i].axhline(fact_bound, c='k', lw=2, ls='--', zorder=10)

        # Annotate bounds clearly
        ax[i].text(100, opt_bound + 0.05, 'Exact Bound', color='k', fontsize=12)
        ax[i].text(100, fact_bound + 0.05, 'Factorized Bound', color='k', fontsize=12)

        ax[i].set_ylim(0, 0.95)
        ax[i].grid(alpha=0.3)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_xlabel('Testing Episodes')
        ax[i].legend(loc='lower right')

    ax[0].set_ylabel('Final Step Accuracy')
    ax[1].set_yticklabels([])
    ax[0].set_title("Network Convergence")
    ax[1].set_title("Echo-state Scaling")

    plt.savefig("thesis_fig4.svg", dpi=300)
    plt.show()


if __name__ == "__main__":
    cuda = 0
    obs_num = 5 
    state_num = 500  
    realization_num = 10 
    step_num = 30 
    
    base_kwargs = {
        'mode': "sanity", 'cuda': cuda, 'episodes': 1, 
        'show_plots': False, 'obs_num': obs_num, 'training': False,
        'step_num': step_num, 'learn_embeddings': False,
        'realization_num': realization_num, 'state_num': state_num, 'ctx_num': 2
    }
    
    echo_k_test_accs, echo_k_train_accs = [np.empty(3, dtype=object) for _ in range(2)]
    
    agent_ft = CognitiveGridworld(**base_kwargs,
        batch_num=5000, hid_dim=1000, load_env=f"/sanity/fully_trained_ctx_2")  
    
    agent_echo = CognitiveGridworld(**base_kwargs,
        batch_num=20, hid_dim=1000, load_env=f"/sanity/reservoir_ctx_2")  
    
    ft_test_accs = agent_ft.test_acc_through_training
    ft_train_accs = agent_ft.train_acc_through_training
    echo_test_accs = agent_echo.test_acc_through_training
    echo_train_accs = agent_echo.train_acc_through_training
    joint_acc = agent_ft.joint_acc
    naive_acc = agent_ft.naive_acc
    
    for i, k in enumerate([2, 5, 10]):
        h = 1000 * k
        agent_echo_k = CognitiveGridworld(**base_kwargs,
            batch_num=20, hid_dim=h, load_env=f"/sanity/reservoir_ctx_2_{k}k")  
        echo_k_test_accs[i] = agent_echo_k.test_acc_through_training
        echo_k_train_accs[i] = agent_echo_k.train_acc_through_training

    plot_training_convergence(ft_test_accs, echo_test_accs, echo_k_test_accs, joint_acc, naive_acc)