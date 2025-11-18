import numpy as np
import torch
import os
import sys
import inspect
import pickle 

# --- Standardized Plotting Imports ---
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.stats import gaussian_kde
from numpy.linalg import LinAlgError

# --- Path Setup ---
# Kept original logic for path management.
try:
    path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    print("root:", path)
    sys.path.insert(0, path + '/main')
    sys.path.insert(0, path + '/main/bayes')
    sys.path.insert(0, path + '/main/model')
    from main.CognitiveGridworld import CognitiveGridworld
except (TypeError, FileNotFoundError):
    print("Running outside of a file context. CognitiveGridworld class may not be available.")

# --- Global Plotting Style Configuration (Single Source of Truth) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'cmr10',
    'font.sans-serif': 'cmss10',
    'font.monospace': 'cmtt10',
    'axes.formatter.use_mathtext': True,
    'font.size': 13,
    'axes.labelsize': 13,
    'axes.titlesize': 18,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
})


def plot_controllers_curve(naive_agent, joint_agent, online_net, offline_net, reps, path='.'):
    """Plots learning curves for different controllers."""
    controller_training_episodes = online_net.controller_training_episodes
    if controller_training_episodes == 0: return

    def get_mean_and_se(data, num_reps):
        mean = np.mean(data, axis=0)
        se = np.std(data, axis=0) / np.sqrt(num_reps)
        return mean, se

    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    agents_data = [\
        {'data': joint_agent.controller_training_logs['reward']/joint_agent.controller_training_logs['optimality'][:,None], 'label': 'Offline w/ Joint Likelihood', 'col': colors[3], 'alph': 1},
        {'data': offline_net.controller_training_logs['reward']/offline_net.controller_training_logs['optimality'][:,None], 'label': 'Offline w/ Generator','col': colors[2], 'alph': .5},
        {'data': online_net.controller_training_logs['reward']/online_net.controller_training_logs['optimality'][:,None], 'label': 'Online w/ Emperical Observations','col': colors[1], 'alph': .5},
        {'data': naive_agent.controller_training_logs['reward']/naive_agent.controller_training_logs['optimality'][:,None], 'label': 'Offline w/ Marginalized Likelihood','col': colors[0], 'alph': 1}]

    for agent in agents_data:
        agent['mean'], agent['se'] = get_mean_and_se(agent['data'], reps)
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 4))
    x_values = np.arange(1, controller_training_episodes + 1)

    ax.axhline(y=1, color='k', ls='--', label='Maximum Reward', lw = 2)

    for i, agent in enumerate(agents_data):
        ax.plot(x_values, agent['mean'], label=agent['label'], color=agent['col'], zorder=10, lw = 2, ls = '-', alpha = agent['alph'])
        ax.fill_between(x_values, agent['mean'] - agent['se'], agent['mean'] + agent['se'], color=agent['col'], alpha=0.2, zorder=-5)

    ax.set(xlabel='Training Epoch', ylabel='Reward')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))

    os.makedirs(path, exist_ok=True)
    plt.xscale('log')
    ax.legend(loc='lower center', ncol = 2)
    plt.savefig(os.path.join(path, "controller_comparison_curve.pdf"), bbox_inches="tight")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm, rcParams

def plot_controllers_and_trajectories(naive_agent, joint_agent, online_net, offline_net,reps, title=0, batches=[], alpha = 3e-4, power = .8):
    R_naive = np.asarray(naive_agent.controller_training_logs['reward']) / \
              np.asarray(naive_agent.controller_training_logs['optimality'])[:, None]
    R_joint = np.asarray(joint_agent.controller_training_logs['reward']) / \
              np.asarray(joint_agent.controller_training_logs['optimality'])[:, None]
    R_on = np.asarray(online_net.controller_training_logs['reward']) / \
           np.asarray(online_net.controller_training_logs['optimality'])[:, None]
    R_off = np.asarray(offline_net.controller_training_logs['reward']) / \
            np.asarray(offline_net.controller_training_logs['optimality'])[:, None]
    T_min = min(R_naive.shape[1], R_joint.shape[1], R_on.shape[1], R_off.shape[1])
    if T_min < 1: return
    J = int(np.floor(np.log10(T_min))) + 1
    idx = (10**np.arange(J) - 1).astype(np.int64)
    x = np.arange(idx.size)

    def mean_se(R):
        S = R[:, idx]
        return S.mean(0), S.std(0) / np.sqrt(reps)
    m_naive, se_naive = mean_se(R_naive)
    m_joint, se_joint = mean_se(R_joint)
    m_on, se_on = mean_se(R_on)
    m_off, se_off = mean_se(R_off)

    n = min(len(batches), reps)
    batch_idx = np.random.choice(reps, n, replace=False)
    for b_i, b in enumerate(batches):
        batch_idx[b_i] = b

    fig = plt.figure(figsize=(7, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, n, height_ratios=[1, 1])
    ax_bar = fig.add_subplot(gs[0, :])
    width = 0.18
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    ekw = dict(capsize=3, elinewidth=1, capthick=1)

    ax_bar.bar(x - 1.5 * width, m_naive, width, yerr=se_naive,label='Offline w/ Marginalized Likelihood', color=colors[0], error_kw=ekw)
    ax_bar.bar(x - 0.5 * width, m_on, width, yerr=se_on, label='Online w/ Empirical Observations', color=colors[1], error_kw=ekw)
    ax_bar.bar(x + 0.5 * width, m_off, width, yerr=se_off,label='Offline w/ Generator', color=colors[2], error_kw=ekw)
    ax_bar.bar(x + 1.5 * width, m_joint, width, yerr=se_joint,label='Offline w/ Joint Likelihood', color=colors[3], error_kw=ekw)
    ax_bar.axhline(1, color='k', ls='--', lw=1.5, alpha=.7)
    ax_bar.set(xticks=x, xticklabels=[rf"$10^{t}$" for t in range(J)],xlabel='Training episode', ylabel='% Maximum Reward')
    ax_bar.legend(loc='upper right', bbox_to_anchor=(0.56, .99),  ncol=1, frameon=False)
    ax_bar.grid(axis='y', alpha=0.15)
    ax_bar.set_ylim(0.5, 1.01)
    cmap_traj = cm.get_cmap('viridis')

    for i, bi in enumerate(batch_idx):
        ax = fig.add_subplot(gs[1, i])
        pref = offline_net.controller_training_logs['prefence_landscape'][bi]
        pol = np.asarray(offline_net.controller_training_logs['example_policy'][bi])

        pol = pol[:, 0]

        T, H, W = pol.shape
        extent = (-0.5, W - 0.5, H - 0.5, -0.5)
        ax.imshow(pref, cmap='magma', interpolation='spline16', origin='upper', aspect='equal', extent=extent)
        ax.set_xticks([]); ax.set_yticks([])
        flat = pol.reshape(T, -1)
        arg = np.argmax(flat, axis=1).astype(np.int64)
        rr, cc = np.divmod(arg, W)
        rr, cc = rr.astype(np.float128), cc.astype(np.float128)

        for t in range(1, T):
            rr[t] = rr[t - 1] + alpha * (rr[t] - rr[t - 1])
            cc[t] = cc[t - 1] + alpha * (cc[t] - cc[t - 1])

        F = 15
        plt.plot(cc[F:], rr[F:], c='k', lw=6, alpha = .2)
        plt.plot(cc[F:], rr[F:], c='k', lw=5.5, alpha = .2)
        plt.plot(cc[F:], rr[F:], c='k', lw=4.5, alpha = 1)
        nseg = rr.shape[0] - 1
        segs = np.zeros((nseg, 2, 2), np.float32)
        segs[:, 0, 0] = cc[:-1]; segs[:, 0, 1] = rr[:-1]
        segs[:, 1, 0] = cc[1:];  segs[:, 1, 1] = rr[1:]
        prog = np.linspace(0, .9, nseg, endpoint=True) ** power

        LC = LineCollection(segs, colors=cmap_traj(prog), linewidths=2.5, capstyle='round', joinstyle='round')
        LC.set_rasterized(True)                             
        LC.set_zorder(100)
        ax.add_collection(LC)
        ax.set_rasterization_zorder(0)
        ax.plot(cc[0], rr[0], 'o', ms=4, mec = 'red', mfc='red', mew=4, zorder = 1000, alpha = .3)
        ax.plot(cc[-1], rr[-1], marker='*', ms=12, mec='k', mfc='lime', mew=1, zorder = 1000)

        if i == 0:
            ax.set_ylabel(r"$r_c$")
        ax.set_xlabel(r"$r_{c'}$")
    rcParams['svg.fonttype'] = 'none'
    plt.savefig(f"controllers_and_trajectories_{title}.svg", bbox_inches="tight", dpi=600)
    plt.show()


if __name__ == "__main__":
    cuda = 0
    load = False
    save = False

    reps = 10
    obs_num = 5
    step_num = 30
    hid_dim = 1000
    state_num = 500
    batch_num = 8000
    realization_num = 10
    controller_LR = .0005
    eps = 2 if load else 10000

    joint, naive, online_net, offline_net = [ CognitiveGridworld(**{'mode': "RL", 'cuda': cuda, 'load_env': 'RL', 'show_plots': False,
        'episodes': 2, 'ctx_num': 2,  'realization_num': realization_num, 'batch_num': batch_num, 'training': False,
        'hid_dim': hid_dim,  'obs_num': obs_num, 'state_num': state_num, 'step_num': step_num,
        'controller_LR': controller_LR, 'learn_embeddings': True}) for _ in range(4)]

    online_net.train_controller(eps = eps, reps = reps)
    offline_net.train_controller(eps = eps, reps = reps, offline_teacher = 'generator')
    joint.train_controller(eps = eps, reps = reps, offline_teacher = 'joint')
    naive.train_controller(eps = eps, reps = reps, offline_teacher = 'naive')

    if save:
        with open('online_net.pkl', 'wb') as f:
            pickle.dump(online_net.controller_training_logs, f)
        with open('offline_net.pkl', 'wb') as f:
            pickle.dump(offline_net.controller_training_logs, f)
        with open('joint.pkl', 'wb') as f:
            pickle.dump(joint.controller_training_logs, f)
        with open('naive.pkl', 'wb') as f:
            pickle.dump(naive.controller_training_logs, f)

    if load:
        with open('online_net.pkl', 'wb') as f:
            online_net.controller_training_logs = pickle.load(f)
        with open('offline_net.pkl', 'wb') as f:
            offline_net.controller_training_logs = pickle.load(f)
        with open('joint.pkl', 'wb') as f:
            joint.controller_training_logs = pickle.load(f)
        with open('naive.pkl', 'wb') as f:
            naive.controller_training_logs = pickle.load(f)
            
    plot_controllers_curve(naive, joint,  online_net, offline_net, reps)
    plot_controllers_and_trajectories(naive, joint, online_net, offline_net, reps, title = 0, batches = [9, 6, 2, 8], alpha = .005, power = .3)

