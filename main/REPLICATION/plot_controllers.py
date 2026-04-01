import numpy as np, torch, os, sys, inspect, pickle
import matplotlib.pyplot as plt, matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter

# =============================================================================
# Path setup
# =============================================================================
path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld
logscale = 1

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'cmr10',
    'font.sans-serif': 'cmss10',
    'font.monospace': 'cmtt10',
    'axes.formatter.use_mathtext': True,
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})

DARK_DOT = np.array((30.0, 70.0, 78.0), np.float64) / 255.0
LIGHT_DOT = np.array((75.0, 196.0, 214.0), np.float64) / 255.0

def draw_expected_obs_column(ax, probs_one, size_lo=25.0, size_hi=125.0, edge_lw=2.0):
    p = np.clip(np.asarray(probs_one, np.float64), 0.0, 1.0)
    n = p.size
    y = np.arange(n - 1, -1, -1, dtype=np.float64)
    x = np.zeros(n, np.float64)
    cols = DARK_DOT[None, :] * (1.0 - p[:, None]) + LIGHT_DOT[None, :] * p[:, None]
    sizes = size_lo + (size_hi - size_lo) * p
    ax.scatter(x, y, s=sizes, c=cols, edgecolors='k', linewidth=edge_lw, zorder=3, clip_on=False)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.8, float(n) - 0.2)
    ax.set_aspect('auto') 
    ax.patch.set_alpha(0.0)
    ax.axis('off')

def draw_expected_obs_strip(ax, probs_by_step, size_lo, size_hi, edge_lw=1.0):
    p = np.clip(np.asarray(probs_by_step, np.float64), 0.0, 1.0)
    T, n = p.shape
    xx = np.repeat(np.arange(T, dtype=np.float64)[:, None], n, axis=1)
    yy = np.repeat(np.arange(n - 1, -1, -1, dtype=np.float64)[None, :], T, axis=0)
    cols = DARK_DOT[None, None, :] * (1.0 - p[..., None]) + LIGHT_DOT[None, None, :] * p[..., None]
    sizes = size_lo + (size_hi - size_lo) * p
    ax.scatter(xx.reshape(T * n), yy.reshape(T * n), s=sizes.reshape(T * n), c=cols.reshape(T * n, 3), edgecolors='k', linewidth=edge_lw, zorder=3, clip_on=False)
    ax.set_xlim(-0.8, float(T) - 0.2)
    ax.set_ylim(-0.8, float(n) - 0.2)
    ax.patch.set_alpha(0.0)
    ax.axis('off')

def draw_generated_obs_block(
    fig,
    ax_anchor,
    obs_probs_full,
    cmap_traj,
    opt_obs,
    strip_size_lo=25.0,
    strip_size_hi=100.0,
    strip_y=0.35,  
    strip_h=0.55,  
    strip_xpad=None,
    cbar_y=0.14,
    cbar_h=0.09,
    cbar_xpad=0.15,
    title_fs=18,
    label_fs=18,
):
    ax_anchor.axis('off')
    ax_anchor.set_xlim(0.0, 1.0)
    ax_anchor.set_ylim(0.0, 1.0)
    
    if strip_xpad is None:
        strip_xpad = cbar_xpad
        
    # Variables for the target column to the right
    target_w = 0.04
    gap = 0.04
    
    # Calculate strip width so it is perfectly centered, ignoring the target column
    strip_w = 1.0 - 2.0 * strip_xpad

    ax_obs_strip = ax_anchor.inset_axes((strip_xpad, strip_y, strip_w, strip_h))
    draw_expected_obs_strip(ax_obs_strip, obs_probs_full, size_lo=strip_size_lo, size_hi=strip_size_hi, edge_lw=.25)
    ax_obs_strip.set_title("Generated Observations", pad=5, fontsize=title_fs)

    # New target column (sticking out to the right)
    ax_target = ax_anchor.inset_axes((strip_xpad + strip_w + gap, strip_y, target_w, strip_h))
    draw_expected_obs_column(ax_target, opt_obs, size_lo=strip_size_lo, size_hi=strip_size_hi, edge_lw=.25)
    
    # Text for "Pref." on the right vertically
    ax_target.text(2, 0.5, "Preference", rotation=270, va='center', ha='center', fontsize=title_fs, transform=ax_target.transAxes, clip_on=False)

    # --- ALIGN COLORBAR EXACTLY WITH OBSERVATIONS ---
    # Retrieve number of steps (T) to calculate the exact data limits
    T = obs_probs_full.shape[0]
    
    # draw_expected_obs_strip sets xlim to (-0.8, T - 0.2)
    data_range = (T - 0.2) - (-0.8)
    
    # Data coordinates of the first and last observation column centers
    first_dot_x = 0.0
    last_dot_x = float(T - 1)
    
    # Convert to fractional width of the strip axes
    frac_start = (first_dot_x - (-0.8)) / data_range
    frac_end = (last_dot_x - (-0.8)) / data_range
    
    # Calculate exact position and width for the colorbar relative to ax_anchor
    cbar_true_x = strip_xpad + frac_start * strip_w
    cbar_true_w = (frac_end - frac_start) * strip_w
    cbar_true_end = cbar_true_x + cbar_true_w

    # Colorbar
    sm = ScalarMappable(norm=Normalize(0.9, 0.0), cmap=cmap_traj)
    sm.set_array(np.zeros((1,), np.float32))
    cax = ax_anchor.inset_axes((cbar_true_x, cbar_y, cbar_true_w, cbar_h))
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Training', labelpad=6, fontsize=label_fs)
    cbar.set_ticks(())
    cbar.ax.tick_params(pad=2, labelsize=12)

    cy = cbar_y + cbar_h / 2.0
    
    # Position markers dynamically based on the exact colorbar edges
    marker_offset = 0.05
    
    start_x = cbar_true_x - marker_offset 
    ax_anchor.plot(start_x, cy, marker='o', ms=10, mfc='red', mec='red', clip_on=False)
    ax_anchor.text(start_x, cbar_y - 0.1, 'Start', ha='center', va='top', fontsize=label_fs)
    
    end_x = cbar_true_end + marker_offset
    ax_anchor.plot(end_x, cy, marker='*', ms=15, mfc='lime', mec='k', clip_on=False)
    ax_anchor.text(end_x, cbar_y - 0.1, 'End', ha='center', va='top', fontsize=label_fs)

def get_obs_steps(num_steps, obs_every):
    step = max(1, int(obs_every))
    steps = np.arange(0, num_steps, step, dtype=np.int64)
    if steps.size == 0:
        return np.array((0,), np.int64)
    if steps[-1] != num_steps - 1:
        steps = np.concatenate((steps, np.array((num_steps - 1,), np.int64)))
    return steps

def make_trajectory(start_r, start_c, end_r, end_c, num_steps):
    rr = np.linspace(start_r, end_r, num_steps, dtype=np.float64)
    cc = np.linspace(start_c, end_c, num_steps, dtype=np.float64)
    return rr, cc

def get_controller_reward(agent):
    return np.asarray(agent.controller_training_logs['reward']) / np.asarray(agent.controller_training_logs['optimality'])[:, None]

def plot_controller_training_panel(ax_curve, ax_legend, joint_agent, online_net, offline_net):
    R_online = get_controller_reward(online_net)
    R_offline = get_controller_reward(offline_net)
    R_joint = get_controller_reward(joint_agent)

    # Use modern colormap fetching
    cmap_base = plt.colormaps['viridis'] if hasattr(plt, 'colormaps') else cm.get_cmap('viridis')

    agents_data = (
        (R_online, 'Online', cmap_base(0.25)),
        (R_joint, 'Offline w/ Joint', cmap_base(0.75)),
        (R_offline, 'Offline w/ Generator', cmap_base(0.50)),
    )

    max_T = 0
    for R, label, col in agents_data:
        agent_reps, agent_T = R.shape        
        max_T = max(max_T, agent_T)
        x_agent = np.arange(1, agent_T + 1, dtype=np.float64)
        mu_agent = R.mean(axis=0)
        se_agent = R.std(axis=0) / np.sqrt(float(agent_reps))
        ax_curve.fill_between(x_agent, mu_agent - se_agent, mu_agent + se_agent, color=col, label = label, alpha=0.5, zorder=-20)
    max_T = 0
    for R, label, col in agents_data:
        agent_reps, agent_T = R.shape        
        max_T = max(max_T, agent_T)
        x_agent = np.arange(1, agent_T + 1, dtype=np.float64)
        mu_agent = R.mean(axis=0)
        se_agent = R.std(axis=0) / np.sqrt(float(agent_reps))
        ax_curve.plot(x_agent, mu_agent - se_agent, color=col, lw=2, zorder=20)
        ax_curve.plot(x_agent, mu_agent + se_agent, color=col, lw=2, zorder=20)
                
    if max_T < 1:
        ax_curve.axis('off')
        return
        
    # -------------------------------------------------------------
    # Formatting for Curve Axes (Log-Log)
    # -------------------------------------------------------------
    if logscale:
        ax_curve.set_xscale('log')
        ax_curve.set_yscale('log')
    ax_curve.axhline(1.0, color='k', ls='--', lw=2.5, alpha=1, zorder=100)
    ax_curve.set_xlim(1.0, float(max_T))
    
    # Ensure tick labels look like standard numbers instead of scientific notation
    ax_curve.xaxis.set_major_formatter(ScalarFormatter())
    ax_curve.yaxis.set_major_formatter(ScalarFormatter())
    
    ax_curve.set_ylabel("% Max Reward", fontsize=18)
    ax_curve.grid(axis='y', alpha=0.15)
    
    # MODIFIED: Extract handles/labels from ax_curve and explicitly give them to ax_legend
    handles, labels = ax_curve.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc='center', ncol=3, frameon=False, fontsize=18)
    ax_legend.axis('off') 
    
    ax_curve.set_title("Learning Curves", fontsize=18, pad=15)
    ax_curve.set_xlabel("Controller Training Episodes", fontsize=18, labelpad=10)

    ax_curve.set_ylim([.6, 1.05])
    ax_curve.set_yticks([.6, .7, .8, .9, 1.0])
    ax_curve.set_xticks([1, 10, 100, 1000])

# =============================================================================
# ALIGNED: Combined Observation Likelihoods & Landscape Plot
# =============================================================================
def plot_combined_figure(
    offline_net,    rep,    target_prefs=None,
    obs_every=3,    obs_where="below",
    joint_agent=None,    online_net=None,
    
    # Layout Controls
    figsize=(18, 6), 
    master_width_ratios=(0.4, 1.3), 
    master_wspace=0.1,
    main_width_ratios=(0.4, 1), 
    main_wspace=0.1
):
    JL_full = np.squeeze(offline_net.joint_likelihood)
    JL = JL_full[rep] if JL_full.ndim > 3 else JL_full
    obs_num, H, W = JL.shape
    logged_pref_landscape = offline_net.controller_training_logs['prefence_landscape'][rep]

    if target_prefs is not None:
        opt_obs = np.asarray(target_prefs, np.int64)
    elif 'preferences' in offline_net.controller_training_logs:
        opt_obs = np.asarray(offline_net.controller_training_logs['preferences'][rep], np.int64)
    elif hasattr(offline_net, 'preferences') and np.asarray(offline_net.preferences).ndim > 1:
        opt_obs = np.asarray(offline_net.preferences[rep], np.int64)
    else:
        opt_obs = np.zeros(obs_num, np.int64)
        for i in range(obs_num):
            score_1 = np.sum(logged_pref_landscape * JL[i])
            score_0 = np.sum(logged_pref_landscape * (1.0 - JL[i]))
            opt_obs[i] = 1 if score_1 > score_0 else 0

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    
    gs_master = fig.add_gridspec(
        1, 2, 
        left=0.02, right=0.98, top=0.92, bottom=0.10,
        width_ratios=master_width_ratios, 
        wspace=master_wspace
    )

    # =========================================================================
    # LEFT SECTION
    # =========================================================================
    gs_left = gs_master[0].subgridspec(
        obs_num, 6, 
        width_ratios=(.25, .25, 1.0, 1.0, .1, 1.0),
        hspace=0.2, 
        wspace=0.2
    )

    cum_landscape = np.ones((H, W), np.float64)

    for i in range(obs_num):
        ax_dot = fig.add_subplot(gs_left[i, 0])
        ax_p0 = fig.add_subplot(gs_left[i, 2])
        ax_p1 = fig.add_subplot(gs_left[i, 3])
        ax_cum = fig.add_subplot(gs_left[i, 5]) 

        # Left the Omegas untouched, but removed the large scatter points
        ax_dot.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
        ax_dot.set_ylabel(rf"$\Omega^{{{i+1}}} = {opt_obs[i]}$", fontsize=18, rotation=0, labelpad=0, va='center')
        ax_dot.set_xticks([])
        ax_dot.set_yticks([])
        for spine in ax_dot.spines.values(): spine.set_visible(False)

        if opt_obs[i] == 1:
            alpha_p0, alpha_p1 = 0.18, 1.0
            ax_p0.axis('off')
            ax_p1.set_xticks(())
            ax_p1.set_yticks(())
            for spine in ax_p1.spines.values(): spine.set_linewidth(2)
            preferred_likelihood = JL[i]
        else:
            alpha_p0, alpha_p1 = 1.0, 0.18
            ax_p1.axis('off')
            ax_p0.set_xticks(())
            ax_p0.set_yticks(())
            for spine in ax_p0.spines.values(): spine.set_linewidth(2)
            preferred_likelihood = 1.0 - JL[i]

        ax_p0.imshow(1.0 - JL[i], cmap='viridis', origin='upper', extent=(0, W, H, 0), alpha=alpha_p0, aspect='auto')
        ax_p1.imshow(JL[i], cmap='viridis', origin='upper', extent=(0, W, H, 0), alpha=alpha_p1, aspect='auto')
        
        cum_landscape *= preferred_likelihood
        display_landscape = np.power(cum_landscape, 1.0 / (i + 1.0))
        
        ax_cum.imshow(display_landscape, cmap='magma', origin='upper', extent=(0, W, H, 0), aspect='auto')
        ax_cum.set_xticks(())
        ax_cum.set_yticks(())
        for spine in ax_cum.spines.values(): spine.set_linewidth(2)
        
        if i == 0:
            pad_val = 10
            ax_p1.set_title(r"$\ell_{\mathbf{z}^i}(\mathbf{r})$", x=.5, pad=pad_val, va='bottom', ha='center', fontsize=15)
            ax_p0.set_title(r"$1 - \ell_{\mathbf{z}^i}(\mathbf{r})$", x=.5, pad=pad_val, va='bottom', ha='center', fontsize=15)
            ax_cum.set_title("Cum.", pad=pad_val, va='bottom', ha='center')

    # =========================================================================
    # RIGHT SECTION
    # =========================================================================
    gs_right = gs_master[1].subgridspec(
        1, 2, 
        width_ratios=main_width_ratios,
        wspace=main_wspace
    )

    pref_landscape = display_landscape 

    # Standardized Height Ratios for Top (Main Plots) and Bottom (Legends/Obs)
    shared_hr = (.9, 0.3)
    shared_hs = 0.18

    # --- LANDSCAPE & OBS AREA (Col 0) ---
    gs_mid = gs_right[0].subgridspec(
        2, 1,
        height_ratios=shared_hr,
        hspace=shared_hs
    )

    ax_land = fig.add_subplot(gs_mid[0])
    ax_gen_block = fig.add_subplot(gs_mid[1])

    ax_land.imshow(pref_landscape, cmap='magma', origin='upper', extent=(0, W, H, 0), aspect='auto')
    ax_land.set_xticks(())
    ax_land.set_yticks(())
    ax_land.set_title("Preference Landscape", fontsize=18, pad=15)
    
    for spine in ax_land.spines.values():
        spine.set_linewidth(3.5)

    start_r, start_c = np.unravel_index(np.argmin(pref_landscape), pref_landscape.shape)
    end_r, end_c = np.unravel_index(np.argmax(pref_landscape), pref_landscape.shape)
    num_steps = 25
    rr_traj, cc_traj = make_trajectory(start_r, start_c, end_r, end_c, num_steps)
    rr_traj += 0.5
    cc_traj += 0.5

    cmap_traj = plt.colormaps['viridis'] if hasattr(plt, 'colormaps') else cm.get_cmap('viridis')
    power = 0.8

    ax_land.plot(cc_traj, rr_traj, c='k', lw=10, alpha=.15)
    ax_land.plot(cc_traj, rr_traj, c='k', lw=8, alpha=.95)

    segs = np.stack((np.c_[cc_traj[:-1], rr_traj[:-1]], np.c_[cc_traj[1:], rr_traj[1:]]), axis=1)
    lc = LineCollection(segs, colors=cmap_traj(np.linspace(0.0, 0.9, segs.shape[0]) ** power), linewidths=5, capstyle='round', joinstyle='round')
    lc.set_rasterized(True)
    lc.set_zorder(100)
    ax_land.add_collection(lc)
    ax_land.set_rasterization_zorder(0)

    ax_land.plot(cc_traj[0], rr_traj[0], 'o', ms=8.0, mec='red', mfc='red', mew=2.4, zorder=1000, alpha=0.92)
    ax_land.plot(cc_traj[-1], rr_traj[-1], marker='*', ms=17, mec='k', mfc='lime', mew=1, zorder=1000)

    steps = get_obs_steps(num_steps, obs_every)
    box_w = 0.055
    box_h = 0.25 
    gap = 0.022

    traj_pts = np.empty((num_steps, 2), np.float64)
    traj_pts[:, 0] = cc_traj
    traj_pts[:, 1] = rr_traj
    traj_axes = ax_land.transAxes.inverted().transform(ax_land.transData.transform(traj_pts))

    for j in range(steps.size):
        step = int(steps[j])
        r = int(np.clip(np.round(rr_traj[step]), 0, H - 1))
        c = int(np.clip(np.round(cc_traj[step]), 0, W - 1))
        k0 = max(step - 1, 0)
        k1 = min(step + 1, num_steps - 1)
        tan = traj_axes[k1] - traj_axes[k0]
        nrm = np.array((-tan[1], tan[0]), np.float64)
        nrm_len = np.hypot(nrm[0], nrm[1])

        if nrm_len < 1e-12: nrm[0], nrm[1] = 0.0, 1.0
        else: nrm /= nrm_len

        if str(obs_where).lower() == "above":
            if nrm[1] < 0.0: nrm *= -1.0
        else:
            if nrm[1] > 0.0: nrm *= -1.0

        center = traj_axes[step] + nrm * (0.5 * box_h + gap)
        x0 = np.clip(center[0] - 0.5 * box_w, 0.01, 0.99 - box_w)
        y0 = np.clip(center[1] - 0.5 * box_h, 0.02, 0.98 - box_h)
        ax_obs = ax_land.inset_axes((x0, y0, box_w, box_h), transform=ax_land.transAxes)
        draw_expected_obs_column(ax_obs, JL[:, r, c])

    r_idx = np.clip(np.round(rr_traj).astype(np.int64), 0, H - 1)
    c_idx = np.clip(np.round(cc_traj).astype(np.int64), 0, W - 1)
    obs_probs_full = JL[:, r_idx, c_idx].T
    
    # Passed opt_obs here to draw the new Target column
    draw_generated_obs_block(
        fig, ax_gen_block, obs_probs_full, cmap_traj, opt_obs=opt_obs,
        strip_size_lo=25.0, strip_size_hi=100.0,
        strip_y=0.15, strip_h=0.75, strip_xpad=0.0,
        cbar_y=-0.1, cbar_h=0.15, cbar_xpad=0.15,
        title_fs=18, label_fs=18)

    # --- TRAINING CURVE AREA (Col 1) ---
    gs_train = gs_right[1].subgridspec(
        2, 2, 
        width_ratios=(.1, 0.9), 
        height_ratios=(0.9, 0.1), 
        hspace=.1, 
        wspace=0.1 
    )
    
    ax_curve = fig.add_subplot(gs_train[0, 1]) 
    ax_legend = fig.add_subplot(gs_train[1, 1]) 
    
    plot_controller_training_panel(ax_curve, ax_legend, joint_agent, online_net, offline_net)

    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig("combined_landscape_aligned.svg", bbox_inches="tight", dpi=600)


# =============================================================================
# NEW: Standalone 1x6 Trajectory Figure
# =============================================================================
def plot_example_trajectories_1x6(offline_net, batches=(8, 5, 7, 9, 4, 1), alpha_traj=0.0005, power_traj=0.3):
    n_traj = len(batches)
    fig, axs = plt.subplots(1, n_traj, figsize=(10, 2), constrained_layout=True)
    
    if n_traj == 1:
        axs = [axs]

    cmap_traj = plt.colormaps['viridis'] if hasattr(plt, 'colormaps') else cm.get_cmap('viridis')

    for i in range(n_traj):
        bi = int(batches[i])
        ax = axs[i]

        pref = offline_net.controller_training_logs['prefence_landscape'][bi]
        pol = np.asarray(offline_net.controller_training_logs['example_policy'][bi])
        T_pol, H_pol, W_pol = pol.shape

        extent = (-0.5, W_pol - 0.5, H_pol - 0.5, -0.5)
        ax.imshow(pref, cmap='magma',origin='upper', extent=extent, aspect='auto')
        ax.set_xticks(())
        ax.set_yticks(())

        flat = pol.reshape(T_pol, -1)
        arg = np.argmax(flat, axis=1).astype(np.int64)
        rr_pol, cc_pol = np.divmod(arg, W_pol)
        rr_pol = rr_pol.astype(np.float64)
        cc_pol = cc_pol.astype(np.float64)

        for t in range(1, T_pol):
            rr_pol[t] = rr_pol[t - 1] + alpha_traj * (rr_pol[t] - rr_pol[t - 1])
            cc_pol[t] = cc_pol[t - 1] + alpha_traj * (cc_pol[t] - cc_pol[t - 1])

        F = 15
        ax.plot(cc_pol[F:], rr_pol[F:], c='k', lw=8, alpha=.2)
        ax.plot(cc_pol[F:], rr_pol[F:], c='k', lw=6, alpha=.2)
        ax.plot(cc_pol[F:], rr_pol[F:], c='k', lw=4, alpha=1)

        nseg = rr_pol.shape[0] - 1
        segs = np.zeros((nseg, 2, 2), np.float32)
        segs[:, 0, 0] = cc_pol[:-1]
        segs[:, 0, 1] = rr_pol[:-1]
        segs[:, 1, 0] = cc_pol[1:]
        segs[:, 1, 1] = rr_pol[1:]
        
        prog = np.log(np.linspace(1, 5.5, segs.shape[0]))


        LC = LineCollection(segs, colors=cmap_traj(prog), linewidths=2, capstyle='round', joinstyle='round')
        LC.set_rasterized(True)
        LC.set_zorder(100)
        ax.add_collection(LC)
        ax.set_rasterization_zorder(0)

        ax.plot(cc_pol[0], rr_pol[0], 'o', ms=8, mec='red', mfc='red', mew=4, zorder=1000, alpha=.5)
        ax.plot(cc_pol[-1], rr_pol[-1], marker='*', ms=15, mec='k', mfc='lime', mew=1, zorder=1000)

        if i == 0: 
            ax.set_ylabel(r"$r_c$", fontsize=15)
        ax.set_xlabel(r"$r_{c'}$", fontsize=15)

    plt.savefig("trajectories_1x6.svg", bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    cuda = 0
    reps = 20
    eps = 2
    env_args = {
        'mode': "RL",
        'cuda': cuda,
        'load_env': 'RL',
        'show_plots': False,
        'episodes': 2,
        'ctx_num': 2,
        'realization_num': 10,
        'batch_num': 10,
        'training': False,
        'hid_dim': 1000,
        'obs_num': 5,
        'state_num': 500,
        'step_num': 30,
        'learn_embeddings': True}

    # Initialized only 3 agents now, skipping naive
    joint, online_net, offline_net = tuple(CognitiveGridworld(**env_args) for _ in range(3))

    agents = {
        'online_net': (online_net, None),
        'offline_net': (offline_net, 'generator'),
        'joint': (joint, 'joint')}
    
    for name, pair in agents.items():
        agent, teacher = pair
        filepath = os.path.join("main/DATA/controller", f"{name}.pkl")
        with open(filepath, 'rb') as f:
            agent.controller_training_logs = pickle.load(f)
    
    rep = 0
    pref = np.array((1, 0, 0, 0, 1), np.int64)
    batches_to_plot = (10, 18, 2, 6, 8, 0)
    
    # Run the perfectly aligned combination (example landscapes removed)
    plot_combined_figure(
        offline_net,
        rep=rep,
        target_prefs=pref,
        obs_every=30,
        obs_where="below",
        joint_agent=joint,
        online_net=online_net
    )
    
    # Run the standalone 1x6 trajectory figure
    plot_example_trajectories_1x6(
        offline_net,
        batches=batches_to_plot,
        alpha_traj=.005,
    )
    plt.show()