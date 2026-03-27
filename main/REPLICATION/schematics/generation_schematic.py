import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# Parameters
cmap = plt.cm.viridis
features, time, N = 4, 25, 8
x, y = np.random.randint(0, N), np.random.randint(0, N)
feature_mats = np.zeros((features, N, N))
observations = np.zeros((features, time), dtype=int)
for i in range(features):
    mat = gaussian_filter(np.random.rand(N, N), sigma=3)
    mat -= mat.min()
    mat /= mat.max()
    feature_mats[i] = mat
    for t in range(time):
        p = mat[y, x]
        observations[i, t] = np.random.rand() < p

fig = plt.figure(figsize=(20, 10))
outer = gridspec.GridSpec(3, 1, height_ratios=[3, 0.1, 2], hspace=0.4)
top_gs = gridspec.GridSpecFromSubplotSpec(1, features, subplot_spec=outer[0], wspace=0.6)
mid_gs = gridspec.GridSpecFromSubplotSpec(1, features, subplot_spec=outer[1])
bot_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[2])
norm = Normalize(vmin=0, vmax=1)

obs_col = (85/255, 220/255, 247/255)
obs_col_2 = tuple(c * 0.9 for c in obs_col)
dark_dot = tuple(c * 0.3 for c in obs_col)

def add_circle_shadow(ax, x, y, radius, shadow_color='k', alpha=0.1, offset=(0.05, -0.05)):
    shadow = Circle((x+offset[0], y+offset[1]), radius=radius,
                    linewidth=0, facecolor=shadow_color, alpha=alpha, zorder=1)
    ax.add_patch(shadow)

# TOP ROW
for i in range(features):
    mat = feature_mats[i]
    ax = fig.add_subplot(top_gs[0, i])
    ax.imshow(mat, cmap=cmap, norm=norm, origin='lower', zorder=2)
    for lw, a in zip([8, 6, 4], [0.3, 0.6, 0.9]):
        ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, linewidth=lw, edgecolor=obs_col, facecolor='none', alpha=a, zorder=4))
    ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, linewidth=2, edgecolor='white', facecolor='none', zorder=50))
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_title(r'R$_0$', color='g', fontsize=18, weight='bold')
    if i == 0:
        ax.set_ylabel(r'R$_1$', color='g', fontsize=18, weight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(rf"$P(obs_{i}=0 | R_0, R_1)$", fontsize=18)

# MIDDLE ROW
for i in range(features):
    ax = fig.add_subplot(mid_gs[0, i])
    for spine in ax.spines.values(): spine.set_visible(False)
    for t in range(time):
        color = dark_dot if observations[i, t] else obs_col_2
        s = 20 if observations[i, t] else 60
        edge = None if observations[i, t] else 'k'
        ax.scatter(t, 0, s=s, color=color, zorder=2, edgecolors=edge)
    ax.set_xlim(-0.5, time-0.5)
    ax.set_ylim(-1, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(time/2, -3, rf"obs$_{i}$ trajectory", fontsize=16, ha='center', va='top')

# BOTTOM GRID
ax_all = fig.add_subplot(bot_gs[0])
for spine in ax_all.spines.values(): spine.set_visible(False)

for i in range(features):
    y_pos = features - i - 1
    ax_all.add_line(Line2D([-1, time], [y_pos, y_pos], lw=1, color='gray', zorder=0))

    for t in range(time):
        val = observations[i, t]
        shadow_color = obs_col if val else 'k'
        color = dark_dot if val else obs_col_2
        radius = .3 if val else .4
        alpha = .5 if val else .1
        add_circle_shadow(ax_all, t, y_pos, radius=radius+.05, shadow_color=shadow_color, alpha=alpha)
        if val:
            ax_all.add_patch(Circle((t, y_pos), radius=radius, color=color, zorder=3))
        else:
            ax_all.add_patch(Circle((t, y_pos), radius=radius + .025, color=shadow_color, zorder=3, alpha=alpha))
            ax_all.add_patch(Circle((t, y_pos), radius=radius, color=color, zorder=4))

ax_all.set_xlim(-0.5, time-0.5)
ax_all.set_ylim(-0.5, features-0.5)
ax_all.set_xticks([]); ax_all.set_yticks([])
ax_all.set_ylabel(r"$T$", labelpad=20, fontsize=20, rotation=0)
ax_all.tick_params(colors='white')

plt.tight_layout()
plt.savefig("generation_schematic.svg")
plt.show()
