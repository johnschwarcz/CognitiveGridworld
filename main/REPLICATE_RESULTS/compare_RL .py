import numpy as np; import torch; import os; import sys; import inspect
import pylab as plt; from matplotlib.colors import PowerNorm; import matplotlib as mpl
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld
from utils import tnp

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



def stretch3d(ax, sx=1, sy=1, sz=1):
    """Stretches the 3D plot axes for better visualization."""
    try:
        ax.set_box_aspect((sx, sy, sz))
    except AttributeError:
        # Fallback for older Matplotlib versions
        M = np.diag([sx, sy, sz, 1.0])
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), M)

def _vir_lum(N=1000, eps=0.02, gamma=1.0):
    """Generates a perceptually uniform lightness array from viridis."""
    t = np.linspace(0, 1, N)
    y = (mpl.cm.viridis(t)[:, :3] @ [0.2126, 0.7152, 0.0722])
    y = (y - y.min()) / (y.max() - y.min())
    y = eps + (1 - 2 * eps) * y
    return y**gamma

# def cmap_black2blue(N=1000, hue=0.55, sat=0.8, gamma=1, vmin=.15, vmax=1):
def cmap_black2blue(N=1000, hue=0.55, sat=1, gamma=1, vmin=.15, vmax=1):
    """Creates a perceptually uniform colormap from black to a specified blue."""
    L = _vir_lum(N, gamma=gamma)
    L = vmin + (vmax - vmin) * L
    H = np.full(N, hue)
    S = np.full(N, sat)
    rgb = mpl.colors.hsv_to_rgb(np.stack([H, S, L], 1))
    return mpl.colors.ListedColormap(np.c_[rgb, np.ones(N)], 'black2blue_perc')

def plot_belief(self, agent = "net"):
    #######################################
    """ PLOTTING BELIEF """
    #######################################
    i = np.random.randint(self.batch_num)
    R = self.realization_num
    T = self.step_num

    t2 = np.linspace(0, T-1, (T-1)* 10 +1)
    r2 = np.linspace(0, R-1, (R-1)* 10 +1)
    times = np.arange(T)
    X2, Y2 = np.meshgrid(t2, r2)

    net_belief = tnp(self.model.classifier_goal_belief[i].T, 'np')
    joint_belief = self.joint_goal_belief[i].T
    naive_belief = self.naive_goal_belief[i].T
    
    beliefs = [net_belief, joint_belief, naive_belief]
    titles = [f"net belief", "joint belief", "independent belief"]
    for title, belief in zip(titles, beliefs):
        fig, ax = plt.subplots(1, 1, figsize=(25,25), subplot_kw={'projection':'3d'})

        gv = self.goal_value[i]
        g_i = self.goal_ind[i]       
        belief[:,0] = 0
        B_up = upsample(belief, T, R, t2, r2)
        b = (0.8, 0.9, 1, 1)
        axes_names = ('xaxis','yaxis','zaxis')
        axes = (ax.xaxis, ax.yaxis, ax.zaxis)
        for name in ('xaxis','yaxis','zaxis'):
            axis = getattr(ax, name)
            axis.set_pane_color((0,0,0,0))
            axis._axinfo['grid']['linewidth'] = 0 

        norm = PowerNorm(1, vmin=0, vmax=1.1)       
        epses = [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, .2]
        alphs = np.linspace(0,1, len(epses))**2
        for eps, alph in zip(epses, alphs):
            B_up[B_up <= eps] = np.nan
            B_up[:,0][B_up[:,0] > eps] = 0
            B_up[0][B_up[0] > eps] = 0
            B_up[:,0] = 0
            xt = times[-1]
            bar_x = np.full(2, xt)
            bar_y = np.full(2, gv)            
            ax.plot_surface(X2, Y2, B_up, cmap='rocket', edgecolor='none', zorder = 0,
                            rstride=1, antialiased=False, norm=norm, alpha = alph)
        for v in range(R):
            ax.plot(times, np.full(T, v), belief[v], '-', color = b, alpha = .3, lw=20, zorder=100)
            ax.plot(times, np.full(T, v), belief[v], '-', color = b, alpha = 1, lw=5, zorder=200)
        ax.plot(times, np.full(T, gv), belief[gv], '-g', lw=20, alpha=.3, zorder=200)
        ax.plot(times, np.full(T, gv), belief[gv], '-', color = 'limegreen', lw=5, zorder=300)
        ax.plot(bar_x, bar_y, [belief[gv,-1], 1], 'r-', lw=5, zorder=300)
        ax.plot(bar_x, bar_y, [belief[gv,-1], 1], 'r-', lw=20, alpha = .3, zorder=200)

        ax.xaxis.set_rotate_label(False)
        ax.set_zlabel(''); 
        ax.zaxis.set_rotate_label(False)
        ax.view_init(30,200)
        ax.set_xlim(.5,T)
        ax.set_ylim(0,R-1); ax.set_zlim(0,1)

        ax.set_zticks([])
        ax.set_xticks([])
        ax.set_yticks([])

        ax.tick_params(axis='y', pad=20)
        ax.zaxis._axinfo['juggled'] = (0, 1, 2)
        
        fig.canvas.draw()
        plt.savefig(os.path.join(path, f"{title}_belief_traj.png"),\
            bbox_inches="tight"); plt.show()

def upsample(Z, T, R, t2, r2):
    A = np.empty((R, t2.size))
    B = np.empty((r2.size, t2.size))
    for j in range(R): A[j] = np.interp(t2, np.arange(T), Z[j])
    for k in range(t2.size): B[:,k] = np.interp(r2, np.arange(R), A[:,k])
    return B

def plot_RL_training(joint, naive, D, step_num):
    D = D[5:]

    x = np.arange(D.shape[1])  # steps
    y = np.arange(D.shape[0])  # episodes
    X, Y = np.meshgrid(x, y)

    nep, jep = 0, D.shape[0] - 1
    xax = x
    jax = np.full_like(xax, jep)
    nax = np.full_like(xax, nep)

    is_below_surface = naive < D[nep, :]
    naive_masked = np.ma.masked_where(is_below_surface, naive)
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    custom_cmap = cmap_black2blue()

    # --- Plotting Code ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 10), subplot_kw={'projection': '3d', 'computed_zorder': False})
    elav, azim = 10, -150
    lb = to_rgba('#e6f3ff', 0.8)
    lb2 = to_rgba("#aaebff", 0.8)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color(lb)

    stretch3d(ax, sx=1, sy=1.5, sz=1)
    surf = ax.plot_surface(X, Y, D, cmap=custom_cmap, edgecolor='none', antialiased=False, lw=0)
    surf.set_rasterized(True)

    ax.plot_wireframe(X, Y, D, rcount=10, ccount=9, color=lb2, linewidth=4, alpha = .2)
    ax.plot_wireframe(X, Y, D, rcount=10, ccount=9, color=lb2, linewidth=.4, alpha = 1)
    ax.plot_wireframe(X, Y, D, rcount=10, ccount=0, color=lb2, linewidth=6, alpha = .2)
    ax.plot_wireframe(X, Y, D, rcount=10, ccount=0, color=lb2, linewidth=.8, alpha = 1)

    XJ, YJ = np.meshgrid(x, y)
    ZJ_naive = np.broadcast_to(naive, (y.size, x.size))
    ax.plot_surface(XJ, YJ, ZJ_naive, color=colors[0], alpha=0.2, linewidth=0, zorder=-1000)
    ax.plot_surface(XJ, YJ, ZJ_naive, color=colors[0], alpha=0.05, linewidth=0)
    ZJ_joint = np.broadcast_to(joint, (y.size, x.size))
    ax.plot_surface(XJ, YJ, ZJ_joint, color=colors[3], alpha=0.2, linewidth=0, zorder=-1000)


    ax.plot(0, y, D[:,0], color='k', lw=6, ls='-', alpha=1, zorder=-2000)
    ax.plot(x, nax[0], D[0], color='k', lw=8, ls='-', alpha=1, zorder=-2000)
    ax.plot(xax, jax, D[-1], color='k', lw=9, ls='-', alpha=1, zorder=-2000)
    ax.plot(x, nax[0], D[0], color=lb2, lw=1.5, ls='-', alpha=1, zorder=2000)
    ax.plot(xax, jax, D[-1], color=lb2, lw=6, ls='-', alpha=.3, zorder=2000)
    ax.plot(xax, jax, D[-1], color=lb2, lw=2, ls='-', alpha=1, zorder=2000)

    lw_thin, lw_thick = 1.5, 1.5 * 4
    for lw, ls, alpha in [(lw_thin, '--', 1.0), (lw_thick, '-', 0.3)]:

        # ax.plot(xax, jax, D[-1], color=lb2, lw=lw/2, ls='-', alpha=alpha, zorder=1000)
        ax.plot(step_num-1, y, D[:,-1], color='k', lw=lw/1.2, ls='-', alpha=1, zorder=-2000)
        ax.plot(step_num-1, y, D[:,-1], color=lb2, lw=lw/4, ls='-', alpha=alpha, zorder=2000)
        ax.plot(0, y, D[:,0], color=lb2, lw=lw/2, ls='-', alpha=alpha, zorder=2000)
        # Plot 'Joint' lines (thin and thick)
        ax.plot(xax, jax, joint, color=colors[3], lw=lw, ls=ls, alpha=alpha, zorder=3000)
        ax.plot(xax, nax, joint, color=colors[3], lw=lw, ls=ls, alpha=alpha, zorder=3000)
        ax.plot(x[-1], y, joint[-1], color=colors[3], lw=lw, ls=ls, alpha=alpha, zorder=3000)
        ax.plot(x[0], y, joint[0], color=colors[3], lw=lw, ls=ls, alpha=alpha, zorder=3000)
        # Plot 'Independent' (naive_masked) lines (thin and thick)
        ax.plot(xax, nax, naive_masked, color=colors[0], lw=lw, ls=ls, alpha=alpha, zorder=3000)
        ax.plot(x[-1], y, naive[-1], color=colors[0], lw=lw, ls=ls, alpha=alpha, zorder=-1000)
        ax.plot(x[0], y, naive[0], color=colors[0], lw=lw, ls=ls, alpha=alpha, zorder=1000)

        ax.plot(xax, jax, naive_masked, color=colors[0], lw=lw, ls=ls, alpha=alpha/4)
        ax.plot(xax, nax, naive_masked, color=colors[0], lw=lw, ls=ls, alpha=alpha/4)
        ax.plot(x[-1], y, naive[-1], color=colors[0], lw=lw, ls=ls, alpha=alpha/4)
        ax.plot(x[0], y, naive[0], color=colors[0], lw=lw, ls=ls, alpha=alpha/4)

    ax.view_init(elev=elav, azim=azim)
    ax.set_xlabel('Inference Time')
    ax.set_ylabel('Testing Episode')
    ax.set_zlabel('Performance')
    # ax.set_yticks(y[::300])
    ax.set_zticks([.1, .4, .7])
    plt.savefig(f"acc_surface_{custom_cmap.name}.pdf", bbox_inches="tight", dpi=300)
    plt.show()
    

if __name__ == "__main__":

    cuda = 0
    ctx_num = 2
    obs_num = 5 
    step_num = 30
    batch_num = 10
    state_num = 500
    realization_num = 10

    net = CognitiveGridworld(**{'mode': 'RL', 'cuda': cuda, 'load_env': "RL", 'state_num': state_num,
    'ctx_num': ctx_num, 'batch_num': batch_num, 'obs_num': obs_num,
    'step_num': step_num, 'realization_num': realization_num,
    'training': False, 'episodes': 1, 'show_plots': False})    
    
    bayes = CognitiveGridworld(**{'mode': None, 'cuda': cuda, 'ctx_num': ctx_num, 'show_plots': False,
        'obs_num': obs_num, 'step_num': step_num, 'realization_num': realization_num,
        'batch_num': 10000, 'episodes': 1})    
    joint = bayes.joint_acc.mean(0)
    naive = bayes.naive_acc.mean(0)
    D = net.test_acc_through_training

    # plot_belief(net)
    plot_RL_training(joint, naive, D, step_num)

