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
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')

from main.CognitiveGridworld import CognitiveGridworld 
from main.utils import fig_path

# Setup Plotting Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def plot_polished_1x4_v1(joint_accs, naive_accs, std_bound, T, ctxs=1):
    """Version 1: The Generous vs Strict Bracketing (0-Smoothness)"""
    
    # Create a 2-row layout using subplot_mosaic.
    # Top row: A (Optimal), B (Naive), C (Relative)
    # Bottom row: D (Reward Lost, spanning across all 3 top columns)
    fig = plt.figure(figsize=(16, 12))
    mosaic = """
    ABC
    DDD
    """
    axs = fig.subplot_mosaic(mosaic, height_ratios = (1,2))
    
    # Assign the axes variables to match the existing plotting logic
    ax_j = axs['A']
    ax_n = axs['B']
    ax_rel = axs['C']
    ax_sub = axs['D']

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, ctxs))
    t = np.arange(T)
    
    for c in range(ctxs):
        # Average over repetitions
        mj = joint_accs[c].mean(0)  
        mn = naive_accs[c].mean(0)
        
        m_std_bound = std_bound[c].mean(0) 
        
        # Empirical Reward Lost (Suboptimality)
        emp_reward_lost = mj - mn  

        # --- Panel 1: Optimal Performance ---
        ax_j.plot(t, mj, c=colors[c], lw=3, label=f'C = {c+1}')
        
        # --- Panel 2: Factored Performance ---
        ax_n.plot(t, mn, c=colors[c], lw=3) 
        
        # --- Panel 3: Reward Lost & Information Cost Bounds (Now on bottom) ---
        ax_sub.plot(t, emp_reward_lost, c=colors[c], lw=2.25, ls='-', alpha=1) 
        ax_sub.plot(t, m_std_bound, c=colors[c], lw=2.6, ls=':', alpha=0.6, zorder=4) 
        ax_sub.fill_between(t, emp_reward_lost, m_std_bound, color=colors[c], alpha=0.05)

        # --- Panel 4: Relative Accuracy ---
        ax_rel.plot(mn, mj, c=colors[c], lw=2.5, ls='-', alpha=0.4, zorder=2)
        ax_rel.scatter(mn[0], mj[0], s=70, facecolors=colors[c], edgecolors='w', zorder=3)
        ax_rel.scatter(mn[-1], mj[-1], s=70, facecolors=colors[c], edgecolors='w', zorder=3)
        
        # Annotate the multipliers on the final point
        ax_rel.annotate(r"$\times$" + f'{mj[-1]/mn[-1]:.0f}', (mn[-1], mj[-1]), 
                        xytext=(-20, 15), textcoords='offset points', color='k', fontsize=11)
        if c == 0:
            ax_rel.annotate('start', (mn[0], mj[0]), xytext=(10, -15), textcoords='offset points', fontsize=11)

    ax_rel.plot([0.0, 1.05], [0.0, 1.05], ls='-', c='gray', lw=1, alpha=0.5, zorder=1)
    
    # --- Formatting & Titles ---
    shared_ylim = (0, 1.05)
    
    ax_j.set(title="Optimal (Exact)\nPerformance", xlabel="Time (Steps)", ylabel='Accuracy', ylim=shared_ylim, xlim=(0, T))
    ax_n.set(title="Naive (Factorized)\nPerformance", xlabel="Time (Steps)", ylabel = "Accuiracy", ylim=shared_ylim, xlim=(0, T))
    
    ax_sub.set(title="Reward Lost (Exact - Factorized)", xlabel="Time (Steps)", ylabel="Probability Gap", ylim=(0, 1.05), xlim=(0, T))
    ax_rel.set(title="Relative Accuracy\n", xlabel='Factorized Accuracy', ylabel='Exact Accuracy', xlim=(0.1, 1.05), ylim=(0.1, 1.05))

    for ax in [ax_j, ax_n, ax_sub, ax_rel]:
        ax.grid(alpha=0.2, linestyle='--')
        
    # --- Clean Legends ---
    ax_j.legend(loc='lower right')
    
    # Updated legend strictly matched to the expected trajectory Pinsker penalty
    std_legend = Line2D([0], [0], color='gray', lw=2.6, ls=':', alpha=0.6, label=r'$\mathbb{E}[\min(1, \sqrt{2D_{\mathrm{KL}}})]$')
    emp_gap_legend = Line2D([0], [0], color='gray', lw=2.5, ls='-', alpha=1, label='Empirical Gap')
    
    ax_sub.legend(handles=[std_legend, emp_gap_legend], ncol=1, loc='lower right', fontsize=12)

    plt.savefig(fig_path("thesis_fig2_bracket_bounds_zero_smooth.svg"), dpi=300, bbox_inches='tight')
    plt.show()


def safe_sqrt(x, eps = 1e-12):
    return np.sqrt(np.clip(x, eps, None))

if __name__ == "__main__":
    cuda = 1
    obs_num = 5 
    state_num = 500  
    realization_num = 10 

    batch_num = 500
    step_num = 1000# 5000 
    ctxs = 3
    reps = 250 # 500
    T = step_num
    
    # Using np.float32 to save RAM given the massive array sizes
    j_acc, n_acc, std_bound = [np.zeros((ctxs, reps, batch_num, T), dtype=np.float32) for _ in range(3)]
    
    for c in range(ctxs):
        for r in tqdm(range(reps), desc=f'Context {c+1}'):
            agent = CognitiveGridworld(**{'mode': None, 'cuda': cuda, 'episodes': 1,
                'hid_dim': None, 'show_plots': False, 'obs_num': obs_num, 'training': False,
                'batch_num': batch_num,'ctx_num': c+1, 'step_num': step_num,
                'realization_num': realization_num, 'state_num': state_num}) 
            
            j_acc[c, r] = agent.joint_acc
            n_acc[c, r] = agent.naive_acc
            
            eps = 1e-20
            P = agent.joint_belief + eps
            Q = agent.naive_belief + eps
            P /= P.sum(axis=-1, keepdims=True)
            Q /= Q.sum(axis=-1, keepdims=True)
            
            # Forward KL. Shape: (batch_num, T, ctx_num)
            kl_fwd = np.sum(P * np.log(P / Q), axis=-1) 
                        
            # Expected Trajectory Bound (Strictly tighter than Jensen's loosened bound)
            # Clip trajectory penalty to 1.0, then average across contexts. Shape: (batch_num, T)
            pinsker = safe_sqrt(2 * kl_fwd)
            BH = safe_sqrt(4 * (1 - np.exp(-kl_fwd)))
            std_bound[c, r] = np.clip(pinsker, 0, 1).mean(-1)
            # std_bound[c, r] = np.minimum(pinsker, BH).mean(-1)
            
    j_acc_mean = j_acc.mean(axis=2)
    n_acc_mean = n_acc.mean(axis=2)
    
    # Average across the joint law / batches
    std_bound_mean = std_bound.mean(axis=2)     
    
    plot_polished_1x4_v1(j_acc_mean, n_acc_mean, std_bound_mean, T, ctxs=ctxs)
