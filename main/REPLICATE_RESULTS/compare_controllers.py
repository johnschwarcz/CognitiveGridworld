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
from matplotlib import rcParams
import matplotlib.cm as cm

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


def plot_controllers_and_trajectories(
    naive_agent, joint_agent, online_net, offline_net, reps, title=0, batches=(),
    alpha=3e-4, power=.8, width_ratios=(1.0, 2.2), figsize=(12, 4), traj_wspace=0.05,
    legend_row_height=0.32, traj_aspect="auto",
    bar_title="", traj_title="",
    bar_title_pad=6, traj_title_pad=6,
    cbar_label_pad=-10, cbar_tick_pad=2, cbar_tick_size=None,
    onset_text_pad=0.0, offset_text_pad=0.0,
    show_full_curves=True, full_curve_alpha=0.22, full_curve_lw=2.0,
    full_curve_fill=True, full_curve_fill_alpha=0.10,
    match_traj_height_to_bar=True, traj_height_scale=1.0
):
    R_naive = np.asarray(naive_agent.controller_training_logs['reward']) / np.asarray(naive_agent.controller_training_logs['optimality'])[:, None]
    R_joint = np.asarray(joint_agent.controller_training_logs['reward']) / np.asarray(joint_agent.controller_training_logs['optimality'])[:, None]
    R_on = np.asarray(online_net.controller_training_logs['reward']) / np.asarray(online_net.controller_training_logs['optimality'])[:, None]
    R_off = np.asarray(offline_net.controller_training_logs['reward']) / np.asarray(offline_net.controller_training_logs['optimality'])[:, None]
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

    if len(batches) > 0:
        n = min(reps, len(batches))
        batch_idx = np.asarray(batches, np.int64)[:n]
    else:
        n = min(4, reps)
        batch_idx = np.random.choice(reps, n, replace=False).astype(np.int64)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    outer = fig.add_gridspec(2, 2, width_ratios=width_ratios, height_ratios=(1.0, legend_row_height))

    ax_bar = fig.add_subplot(outer[0, 0])
    ax_bar_leg = fig.add_subplot(outer[1, 0])
    ax_traj_leg = fig.add_subplot(outer[1, 1])

    # width = 0.18
    width = 0.125
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    ekw = dict(capsize=3, elinewidth=1, capthick=1)

    if show_full_curves:
        x_full = np.log10(np.arange(T_min, dtype=np.float64) + 1.0)

        def plot_mean_fill(R, col):
            mu = R[:, :T_min].mean(0)
            se = R[:, :T_min].std(0) / np.sqrt(reps)
            if full_curve_fill:
                ax_bar.fill_between(x_full, mu - se, mu + se, color=col, alpha=full_curve_fill_alpha, zorder=0)
            ax_bar.plot(x_full, mu, color=col, alpha=full_curve_alpha, lw=full_curve_lw, zorder=1)

        plot_mean_fill(R_naive, colors[0])
        plot_mean_fill(R_on, colors[1])
        plot_mean_fill(R_off, colors[2])
        plot_mean_fill(R_joint, colors[3])

    ax_bar.bar(x - 1.5 * width, m_naive, width, yerr=se_naive, label='w/ Naive Bayes', color=colors[0], error_kw=ekw, zorder=3, alpha = 0.85)
    ax_bar.bar(x - 0.5 * width, m_on, width, yerr=se_on, label='Online', color=colors[1], error_kw=ekw, zorder=3, alpha = 0.85)
    ax_bar.bar(x + 0.5 * width, m_off, width, yerr=se_off, label='w/ Generator', color=colors[2], error_kw=ekw, zorder=3, alpha = 0.85)
    ax_bar.bar(x + 1.5 * width, m_joint, width, yerr=se_joint, label='w/ Joint Bayes', color=colors[3], error_kw=ekw, zorder=3, alpha = 0.85)

    ax_bar.axhline(1, color='k', ls='--', lw=1.5, alpha=.7, zorder=4)
    ax_bar.set(xticks=x, xticklabels=tuple(rf"$10^{t}$" for t in range(J)), xlabel='Episode', ylabel='% Maximum Reward')
    ax_bar.grid(axis='y', alpha=0.15)
    ax_bar.set_ylim(0.5, 1.01)
    xmax = np.log10(float(T_min)) if T_min > 1 else 0.0
    ax_bar.set_xlim(-.5, max(float(J - 1), xmax) + .5)
    if bar_title: ax_bar.set_title(bar_title, pad=bar_title_pad, fontsize=18)

    h, lab = ax_bar.get_legend_handles_labels()
    ax_bar_leg.axis('off')
    ax_bar_leg.legend(tuple(h), tuple(lab), loc='center', ncol=4, frameon=False, fontsize=13)

    gs_traj = outer[0, 1].subgridspec(1, n, wspace=traj_wspace)
    cmap_traj = cm.get_cmap('viridis')
    axs_traj = np.empty(n, dtype=object)

    for i in range(n):
        bi = int(batch_idx[i])
        ax = fig.add_subplot(gs_traj[0, i])
        axs_traj[i] = ax

        pref = offline_net.controller_training_logs['prefence_landscape'][bi]
        pol = np.asarray(offline_net.controller_training_logs['example_policy'][bi])[:, 0]
        T, H, W = pol.shape

        extent = (-0.5, W - 0.5, H - 0.5, -0.5)
        ax.imshow(pref, cmap='magma', interpolation='spline16', origin='upper', extent=extent, aspect=traj_aspect)
        ax.set_xticks(()); ax.set_yticks(())
        ax.set_aspect(traj_aspect)

        flat = pol.reshape(T, -1)
        arg = np.argmax(flat, axis=1).astype(np.int64)
        rr, cc = np.divmod(arg, W)
        rr = rr.astype(np.float128); cc = cc.astype(np.float128)

        for t in range(1, T):
            rr[t] = rr[t - 1] + alpha * (rr[t] - rr[t - 1])
            cc[t] = cc[t - 1] + alpha * (cc[t] - cc[t - 1])

        F = 15
        ax.plot(cc[F:], rr[F:], c='k', lw=6, alpha=.2)
        ax.plot(cc[F:], rr[F:], c='k', lw=5.5, alpha=.2)
        ax.plot(cc[F:], rr[F:], c='k', lw=4.5, alpha=1)

        nseg = rr.shape[0] - 1
        segs = np.zeros((nseg, 2, 2), np.float32)
        segs[:, 0, 0] = cc[:-1]; segs[:, 0, 1] = rr[:-1]
        segs[:, 1, 0] = cc[1:];  segs[:, 1, 1] = rr[1:]
        prog = np.linspace(0.0, 0.9, nseg, endpoint=True) ** power

        LC = LineCollection(segs, colors=cmap_traj(prog), linewidths=2.5, capstyle='round', joinstyle='round')
        LC.set_rasterized(True); LC.set_zorder(100)
        ax.add_collection(LC)
        ax.set_rasterization_zorder(0)

        ax.plot(cc[0], rr[0], 'o', ms=4, mec='red', mfc='red', mew=4, zorder=1000, alpha=.5)
        ax.plot(cc[-1], rr[-1], marker='*', ms=12, mec='k', mfc='lime', mew=1, zorder=1000)

        if i == 0: ax.set_ylabel(r"$r_c$", fontsize=13)
        ax.set_xlabel(r"$r_{c'}$", fontsize=13)

    ax_traj_leg.set_xlim(0, 1)
    ax_traj_leg.set_ylim(0, 1)
    ax_traj_leg.axis('off')

    sm = ScalarMappable(norm=Normalize(0.0, 0.9), cmap=cmap_traj)
    sm.set_array(np.zeros((1,), np.float32))

    y_mark = 0.66
    ax_traj_leg.plot(0.12, y_mark, marker='o', ms=7, mfc='red', mec='red', transform=ax_traj_leg.transAxes, clip_on=False)
    ax_traj_leg.text(0.12, 0.3 + onset_text_pad, 'start', ha='center', va='center', transform=ax_traj_leg.transAxes, fontsize=13)

    ax_traj_leg.plot(0.88, y_mark, marker='*', ms=12, mfc='lime', mec='k', transform=ax_traj_leg.transAxes, clip_on=False)
    ax_traj_leg.text(0.88, 0.3 + offset_text_pad, 'end', ha='center', va='center', transform=ax_traj_leg.transAxes, fontsize=13)

    cax = ax_traj_leg.inset_axes((0.24, 0.52, 0.52, 0.26))
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Episode', labelpad=cbar_label_pad, fontsize=13)
    cbar.set_ticks((0.0, 0.9))
    cbar.set_ticklabels(('early', 'late'), fontsize=13)
    cbar.ax.tick_params(pad=cbar_tick_pad)
    if cbar_tick_size is not None: cbar.ax.tick_params(labelsize=cbar_tick_size)

    fig.canvas.draw()

    if match_traj_height_to_bar:
        bp = ax_bar.get_position()
        bh = bp.height * float(traj_height_scale)
        by = bp.y0 + 0.5 * (bp.height - bh)
        for i in range(n):
            p = axs_traj[i].get_position()
            axs_traj[i].set_position((p.x0, by, p.width, bh))
        fig.canvas.draw()

    if traj_title:
        x0 = 1.0
        x1 = 0.0
        y1 = 0.0
        for i in range(n):
            p = axs_traj[i].get_position()
            if p.x0 < x0: x0 = p.x0
            if p.x1 > x1: x1 = p.x1
            if p.y1 > y1: y1 = p.y1
        dy = traj_title_pad / (72.0 * fig.get_size_inches()[1])
        fig.text(0.5 * (x0 + x1), y1 + dy, traj_title, ha='center', va='bottom', fontsize=18)

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
    batch_num = 50 if load else 8000
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
        with open(os.path.join("main//DATA", "online_net.pkl"), 'wb') as f:
            pickle.dump(online_net.controller_training_logs, f)
        with open(os.path.join("main//DATA", "offline_net.pkl"), 'wb') as f:
            pickle.dump(offline_net.controller_training_logs, f)
        with open(os.path.join("main//DATA", "joint.pkl"), 'wb') as f:
            pickle.dump(joint.controller_training_logs, f)
        with open(os.path.join("main//DATA", "naive.pkl"), 'wb') as f:
            pickle.dump(naive.controller_training_logs, f)

    if load:
        with open(os.path.join("main//DATA", "online_net.pkl"), 'rb') as f:
            online_net.controller_training_logs = pickle.load(f)
        with open(os.path.join("main//DATA", "offline_net.pkl"), 'rb') as f:
            offline_net.controller_training_logs = pickle.load(f)
        with open(os.path.join("main//DATA", "joint.pkl"), 'rb') as f:
            joint.controller_training_logs = pickle.load(f)
        with open(os.path.join("main//DATA", "naive.pkl"), 'rb') as f:
            naive.controller_training_logs = pickle.load(f)
        print("Done loading.")
              


    plot_controllers_and_trajectories(
        naive, joint, online_net, offline_net, reps, title=0, batches=(8, 5, 7),
        alpha=.0005, power=.3, width_ratios=(.8,1), traj_wspace=0.0, figsize=(16, 4.5),
        bar_title="Controller Learning Curves", traj_title="Example Offline Learning Trajectories w/Generator",
        cbar_label_pad=-10, cbar_tick_pad=2.4, bar_title_pad=8, traj_title_pad=8,
        show_full_curves=True, full_curve_alpha=1, full_curve_lw=1,
        full_curve_fill=True, full_curve_fill_alpha=0.12,
        match_traj_height_to_bar=True, traj_height_scale=0.96)
