import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
from matplotlib.transforms import Bbox
from matplotlib.colors import PowerNorm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set() 

def angle_between(v1, v2):
    return np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])) % 360

def draw_interaction_panel(ax, vK, vQ, active, highlight=None):
    ax.set(xlim=[-1.5, 1.5], ylim=[-1.5, 1.5], aspect='equal')
    ax.axis('off')

    for vk, vq in zip(vK, vQ):
        ax.arrow(0, 0, *vk, fc='C0', ec='C0', lw=2, alpha=0.4, head_width=0.1, head_length=0.1)
        ax.arrow(0, 0, *vq, fc='C1', ec='C1', lw=2, alpha=0.4, head_width=0.1, head_length=0.1)

    for c in active:
        is_highlight = (c == highlight)
        if is_highlight:
            for v in [vK[c], vQ[c]]:
                ax.arrow(0, 0, *v, fc='white', ec='green', lw=8, alpha=0.5,
                         head_width=0.1, head_length=0.1, zorder=4)

        ax.arrow(0, 0, *vK[c], fc='C0', ec='C0', lw=4, head_width=0.1, head_length=0.1, zorder=5)
        ax.arrow(0, 0, *vQ[c], fc='C1', ec='C1', lw=4, head_width=0.1, head_length=0.1, zorder=5)

    radius, alpha, color = 0.3, 0.2, 'purple'
    for (from_idx, to_idx) in [(active[0], active[1]), (active[1], active[0])]:
        v_from, v_to = vK[from_idx], vQ[to_idx]
        theta1 = np.degrees(np.arctan2(v_to[1], v_to[0])) % 360
        theta2 = np.degrees(np.arctan2(v_from[1], v_from[0])) % 360
        for width in [None, 0]:
            ax.add_patch(Wedge((0, 0), radius, theta1, theta2, color=color, alpha=alpha, width=width))

    for i, (vk, vq) in enumerate(zip(vK, vQ)):
        pos = 0.5 * (vk + vq)
        label = str(i + 1)
        if i in active:
            if i == highlight:
                ax.text(*pos, label, fontsize=11, fontweight='bold', ha='center', va='center', color='green',
                        bbox=dict(boxstyle='circle,pad=0.25', facecolor='white', edgecolor='green', lw=2, alpha=0.5))
            else:
                ax.text(*pos, label, fontsize=13, fontweight='bold', ha='center', va='center', color='black')
        else:
            ax.text(*pos, label, fontsize=12, ha='center', va='center', color='gray', alpha=0.5)

# --- Interaction Panel Setup ---
num_states = 3
angles_K = np.linspace(0, 2*np.pi, num_states, endpoint=False)
angles_Q = np.linspace(np.pi/3, 2*np.pi + np.pi/3, num_states, endpoint=False)
vK = np.c_[np.cos(angles_K), np.sin(angles_K)]
vQ = np.c_[np.cos(angles_Q), np.sin(angles_Q)]

fig, axs = plt.subplots(3, 3, figsize=(14, 6), height_ratios=[1, 0.3, 0.3])
for row in [1, 2]:
    for ax in axs[row]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
pairs = [(0, 1), (1, 2), (0, 2)]
highlights = [0, 1, 2]

for ax, pair, hi in zip(axs[0], pairs, highlights):
    draw_interaction_panel(ax, vK, vQ, pair, highlight=hi)

plt.tight_layout()
fig.canvas.draw()

def bbox_union(axes):
    return Bbox.union([ax.get_position() for ax in axes])

group1 = [axs[r][c] for r in range(3) for c in [0, 1]]
group2 = [axs[r][2] for r in range(3)]

for bbox in [bbox_union(group1), bbox_union(group2)]:
    fig.add_artist(Rectangle(
        (bbox.x0, bbox.y0 - 0.02),
        bbox.width, bbox.height,
        transform=fig.transFigure,
        fill=False, edgecolor='k', linewidth=2))

bbox1 = bbox_union([axs[0, 0], axs[0, 1]])
center_x = (bbox1.x0 + bbox1.x1) / 2
top_y = bbox1.y1 + 0.01
fig.text(center_x, top_y, "Training", ha='center', va='bottom', fontsize=16, fontweight = 'bold')
axs[0, 2].set_title("Testing", fontsize = 16, fontweight = 'bold')
plt.savefig("goal_schematic.svg")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set()

# --- Parameters ---
R, T = 10, 200 # 500
pairs = 2
eps = 0
t2 = np.linspace(0, T-1, (T-1)*10 + 1)
r2 = np.linspace(0, R-1, (R-1)*10 + 1)
X2, Y2 = np.meshgrid(t2, r2)
times = np.arange(T)

def upsample(belief):
    belief_up = np.array([np.interp(t2, np.arange(T), belief[r]) for r in range(R)])
    belief_up = np.array([np.interp(r2, np.arange(R), belief_up[:, t]) for t in range(len(t2))]).T
    return belief_up

# Generate fake belief data
rng = np.random.default_rng()
belief = np.zeros((2, R, T))
goal_value = np.zeros(2, dtype=int)
goal_ind = np.zeros(2, dtype=int)
norm = np.linspace(.01,2,T)**2
x = np.arange(R)

# --- Plotting ---
for g_i, goal_index in enumerate([0, 1, 0]):
    # fig = plt.figure(figsize=(60, 8))
    fig = plt.figure(figsize=(30, 30))
    # ax1 = fig.add_axes([0, .25, 1, .25], projection='3d')  # [left, bottom, width, height]
    # ax2 = fig.add_axes([0, .1, 1, .25], projection='3d')
    ax1 = fig.add_axes([0.0, 0.1, 0.45, 0.8], projection='3d') 
    ax2 = fig.add_axes([0.25, 0.1, 0.45, 0.8], projection='3d')

    fig.patch.set_facecolor( (1,1,1,0))
    
    for ind in range(2):
        s, e = rng.integers(0, R, size=2)
        means = np.linspace(s, e, T)
        means[-50:] = e
        for t in range(T):
            # std = 20 * np.exp(-t/20) + .5
            std = 30 * np.exp(-t/30) + .5
            n = norm[:t+1]/norm[:t+1].sum()
            b = np.exp(-0.5 * ((x - (means[:t+1]*n).sum()) / std) ** 2)
            b /= b.sum()
            belief[ind, :, t] = b
        goal_value[ind] = int(round(means[-1]))
        goal_ind[ind] = int(round(means[-1]))

    for ind, ax in enumerate([ax1, ax2]):
        goal = ind == goal_index
        alpha = 1 if goal else .2
        gv = goal_value[ind]
        b = belief[ind]
        b[:, 0] = 0
        B_up = upsample(b)
        B_up[B_up <= eps] = np.nan
        B_up[:, 0][B_up[:, 0] > eps] = 0
        B_up[0][B_up[0] > eps] = 0
        B_up[:, 0] = 0

        xt = times[-1]
        bar_x = np.full(2, xt)
        bar_y = np.full(2, gv)

        pale = (0.8, 0.9, 1, 1)
        ax.set_facecolor((1,1,1,0))
        # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([.8, 1, .4, 1]))
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, .6, 1, 1]))

        for name in ('xaxis', 'yaxis', 'zaxis'):
            axis = getattr(ax, name)
            axis.set_pane_color((1, 1, 1, 0))  
            axis._axinfo['grid']['color'] = (1,1,1,0)
            axis._axinfo['grid']['linewidth'] = 1

        ax.plot_surface(X2, Y2, B_up, cmap='rocket', edgecolor='none',
                        rstride=1, antialiased=False, alpha = alpha ** 2,
                        norm=PowerNorm(1, vmin=eps, vmax=1 - eps), zorder=0)

        # for v in range(R):
        #     ax.plot(times, np.full(T, v), b[v], '-', color=pale, alpha=0.3 * alpha, lw=2, zorder=10)
        #     ax.plot(times, np.full(T, v), b[v], '-', color=pale, alpha=1 * alpha, lw=.5, zorder=10)

        if goal:
            ax.plot(times, np.full(T, gv), b[gv], '-g', lw=5, alpha=0.3, zorder=100)
            ax.plot(times, np.full(T, gv), b[gv], '-g', lw=5, zorder=100)            
        if goal:
            ax.set_ylabel("Evaluate", fontsize=50, c = 'g', labelpad=10, fontweight='bold')
        ax.view_init(30, 200)
        # ax.view_init(30, 180)
        ax.set_xlim(1, T)
        ax.set_ylim(0, R-1)
        ax.set_zlim(0, .9)
        ax.set_zticks([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(pad=0)
        ax.xaxis.line.set_color((1, 1, 1, 0))
        ax.yaxis.line.set_color((1, 1, 1, 0))
        ax.zaxis.line.set_color((1, 1, 1, 0))

    fig.canvas.draw()
    plt.savefig(f"evaluate_{g_i}.png")
    plt.show()

