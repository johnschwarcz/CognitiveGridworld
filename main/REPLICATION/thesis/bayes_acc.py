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

def plot_joint_vs_naive_accuracy(joint_accs, naive_accs, DKLs, TVDs, Subopts, T, ctxs=1, reps=1):
    # Set up the 2x3 GridSpec layout
    fig = plt.figure(figsize=(18, 8), tight_layout=True)
    gs = fig.add_gridspec(2, 3)

    # Assign subplots to grid positions
    ax_j = fig.add_subplot(gs[0, 0])      # Top Left
    ax_n = fig.add_subplot(gs[0, 1])      # Top Middle
    ax_rel = fig.add_subplot(gs[0, 2])    # Top Right
    ax_kl = fig.add_subplot(gs[1, 0])     # Bottom Left
    ax_tvd = fig.add_subplot(gs[1, 1])    # Bottom Middle
    ax_sub = fig.add_subplot(gs[1, 2])    # Bottom Right
    
    axs = [ax_j, ax_n, ax_rel, ax_kl, ax_tvd, ax_sub]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, ctxs))
    t = np.arange(T)
    idx = np.linspace(0, T-1, 4, dtype=int)
    
    for c in range(ctxs):
        # Average over repetitions
        mj = joint_accs[c].mean(0)  
        mn = naive_accs[c].mean(0)
        m_kl = DKLs[c].mean(0)
        m_tvd = TVDs[c].mean(0)
        m_sub = Subopts[c].mean(0)

        # 1 & 2: Plot Accuracies
        ax_j.plot(t, mj, c=colors[c], lw=2.5, label=f'C = {c+1}')
        ax_n.plot(t, mn, c=colors[c], lw=2.5) 
        # 3: Plot Relative Accuracy
        ax_rel.plot(mn, mj, c=colors[c], lw=2.5, ls=':', zorder=2)
        ax_rel.scatter(mn[idx], mj[idx], s=50, facecolors=colors[c], edgecolors='k', zorder=3)
        # Annotations for Relative Accuracy
        ax_rel.annotate(r"$\times$" + f'{mj[-1]/mn[-1]:.0f}', (mn[-1], mj[-1]), 
                        xytext=(-15, 10), textcoords='offset points', color='k')
        if c == 0:
            ax_rel.annotate('start', (mn[0], mj[0]), xytext=(-10, -15), textcoords='offset points')
        # 4: Plot KL Divergence
        ax_kl.plot(t, m_kl, c=colors[c], lw=2.5, ls='--')        
        # 5: Plot TVD
        ax_tvd.plot(t, m_tvd, c=colors[c], lw=2.5)
        # Overlay Pinsker's bound for Context > 0 
        if c > 0:
            pinsker_bound = np.sqrt(m_kl / 2)
            ax_tvd.plot(t, pinsker_bound, c=colors[c], lw=2, ls='--', alpha=0.8)
        # 6: Plot Suboptimality
        ax_sub.plot(t, m_sub, c=colors[c], lw=2.5, ls = 'dotted')
        # Overlay 2xTVD Bound for Context > 0
        if c > 0:
            ax_sub.plot(t, 2 * m_tvd, c=colors[c], lw=2, ls='-', alpha=0.8)
    ax_rel.plot([0.1, 1.0], [0.1, 1.0], ls='-', c='gray', lw=1, zorder=1)
    # Set Titles and Labels
    ax_j.set(title="Optimal (Exact) Performance", xticklabels=[], ylabel='Accuracy', ylim=(0, 1.1))
    ax_n.set(title="Factored (Factorized) Performance", xticklabels=[], ylabel='Accuracy', ylim=(0, 1.1))
    ax_rel.set(title="Relative Accuracy", xlabel='Factorized', ylabel='Exact', xlim=(0.1, 1.0), ylim=(0.1, 1.0))
    ax_kl.set(title="Factorized Regret (FR)", xlabel='Inference Time', ylabel='KL( Exact || Factorized )')
    ax_tvd.set(title="Total Variation Distance (TVD)", xlabel='Inference Time', ylabel='A.U.')
    ax_sub.set(title=r'$\mathbb{E}[P_t(\hat{r}_g^{\mathrm{joint}}) - P_t(\hat{r}_g^{\mathrm{fact}})]$', xlabel='Inference Time', ylabel="A.U.", ylim=(0, None))
    # Global formatting
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    # ---------------- Legends ----------------
    # Panel 1: Context Colors
    ax_j.legend(loc='lower right', ncol=2)
     # Panel 5: TVD Style
    tvd_legend = Line2D([0], [0], color='gray', lw=2.5, ls='-', label='TVD')
    pinsker_legend = Line2D([0], [0], color='gray', lw=2, ls='--', alpha=0.8, label=r'$\sqrt{FR/2}$')
    ax_tvd.legend(handles=[pinsker_legend, tvd_legend], loc='upper left')   
    # Panel 6: Suboptimality Style
    sub_legend = Line2D([0], [0], color='gray', lw=2.5, ls=':', label='Subopt')
    sub_bound_legend = Line2D([0], [0], color='gray', lw=2, ls='-', alpha=0.8, label=r'$2 \times TVD$')
    ax_sub.legend(handles=[sub_bound_legend, sub_legend], loc='upper left')
    plt.savefig("thesis_fig2alt.svg", dpi=300)
    plt.show()

if __name__ == "__main__":
    cuda = 0
    obs_num = 5 
    state_num = 500  
    realization_num = 10 

    batch_num = 20
    step_num = 30 
    ctxs = 4
    reps = 5000 
    T = step_num
    
    # Pre-allocate arrays holding data before averaging the batch dimension
    j_acc, n_acc, kl_divs, tvd_divs, subopts = [np.zeros((ctxs, reps, batch_num, T)) for _ in range(5)]
    
    for c in range(ctxs):
        for r in tqdm(range(reps), desc=f'Context {c+1}'):
            agent = CognitiveGridworld(**{'mode': None, 'cuda': cuda, 'episodes': 1,
                'hid_dim': None, 'show_plots': False, 'obs_num': obs_num, 'training': False,
                'batch_num': batch_num,'ctx_num': c+1, 'step_num': step_num,
                'realization_num': realization_num, 'state_num': state_num}) 
            
            # Store Accuracies
            j_acc[c, r] = agent.joint_acc
            n_acc[c, r] = agent.naive_acc
            
            # Extract and normalize beliefs
            eps = 1e-12
            P = agent.joint_belief + eps
            Q = agent.naive_belief + eps
            P /= P.sum(axis=-1, keepdims=True)
            Q /= Q.sum(axis=-1, keepdims=True)
            
            # Calculate KL(Exact || Factorized)
            kl = np.sum(P * np.log(P / Q), axis=-1) 
            kl_divs[c, r] = kl.mean(axis=-1)
            
            # Calculate Total Variation Distance
            tvd = 0.5 * np.sum(np.abs(P - Q), axis=-1)
            tvd_divs[c, r] = tvd.mean(axis=-1)
            
            # Calculate Suboptimality
            idx_joint = np.argmax(P, axis=-1, keepdims=True)
            idx_fact = np.argmax(Q, axis=-1, keepdims=True)
            
            # Take the true probability mass of the chosen decisions
            p_joint_max = np.take_along_axis(P, idx_joint, axis=-1).squeeze(-1)
            p_fact_chosen = np.take_along_axis(P, idx_fact, axis=-1).squeeze(-1)
            
            # Suboptimality is the gap between the optimal decision's mass and factored decision's mass
            subopts[c, r] = (p_joint_max - p_fact_chosen).mean(axis=-1)
            
    # Mean over the batch dimension (axis=2) before passing to the plotting function
    j_acc_mean = j_acc.mean(axis=2)
    n_acc_mean = n_acc.mean(axis=2)
    kl_divs_mean = kl_divs.mean(axis=2)
    tvd_divs_mean = tvd_divs.mean(axis=2)
    subopts_mean = subopts.mean(axis=2)

    plot_joint_vs_naive_accuracy(j_acc_mean, n_acc_mean, kl_divs_mean, tvd_divs_mean, subopts_mean, T, ctxs=ctxs, reps=reps)
