import pylab as plt; import numpy as np
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from .plotting_helpers import Plotting_helpers
        
class Plotting_anim(Plotting_helpers):
    
    def show(self, fps=15, batch=None):
        self.angles = np.linspace(0, 2 * np.pi, self.realization_num, endpoint=False)
        self.sample_batch(batch)
        gv = self.goal_value[self.b]
        of = 2 * (self.obs_flat[self.b] - 0.5)
        numer = np.cumsum(of, axis=0)
        obs_wave = numer / (1 + self.step_range[:, None])
        b_mu_x, b_mu_y, b_dist = self.belief_to_circular(self.joint_goal_belief[self.b])
        m_mu_x, m_mu_y, m_dist = self.belief_to_circular(self.model_goal_belief[self.b])
        x, y = np.cos(self.angles), np.sin(self.angles)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # === Ax0: Observations ===
        obs_height =  .1
        ax[0].set_ylim(-1.1, 1.1)
        ax[0].axhline(0, color='C0')
        ax[0].set_yticks([])
        wave_bars = ax[0].bar(self.obs_range, obs_wave[0], alpha = .5, color = 'C0')
        obs_bars = ax[0].bar(self.obs_range, obs_wave[0] * obs_height, color = 'C0', width = .5)
        obs_line, = ax[0].plot(self.obs_range, obs_wave[0], lw = 2)
        # === Ax1: Belief curves ===
        ax[1].set_ylim(-.02, 1.1)
        ax[1].set_yticks([])
        bayes_line, = ax[1].plot(self.realization_range, b_dist[0], color='C0', lw = 2)
        model_line, = ax[1].plot(self.realization_range, m_dist[0], color='C1', lw = 2)
        rr = self.realization_range 
        def get_poly_patch(y, color):
            verts = np.column_stack(([rr[0], *rr, rr[-1]], [0, *y, 0]))
            return patches.Polygon(verts, closed=True, color=color, alpha=0.5)
        bf_patch = get_poly_patch(b_dist[0], 'C0')
        mf_patch = get_poly_patch(m_dist[0], 'C1')
        ax[1].plot([gv, gv], [.95, 2], ls = '-', color='g', alpha = .5, lw = 15, zorder = -10)
        ax[1].add_patch(bf_patch)
        ax[1].add_patch(mf_patch)
        # === Ax2: Circular view ===
        ax[2].add_artist(plt.Circle((0, 0), 1, color='k', fill=False))
        ax[2].plot(x, y, 'o', color='k', ms=8)
        ax[2].plot(x[gv], y[gv], 'g*', ms=35)
        ax[2].plot(0, 0, 'k+', ms=10)
        ax[2].set_xlim(-1.2, 1.2)
        ax[2].set_ylim(-1.2, 1.2)
        ax[2].set_yticks([])
        
        bl, = ax[2].plot([b_mu_x[:1]], [b_mu_y[:1]], '-o', color='C0', alpha = .2, lw = 5, ms = 2, mec = 'k')
        ml, = ax[2].plot([m_mu_x[:1]], [m_mu_y[:1]], '-o', color='C1', alpha = .2, lw = 5, ms = 2, mec = 'k')
        bd, = ax[2].plot([b_mu_x[0]], [b_mu_y[0]], 'o', color='C0', ms=15, mec='k', mew=2)
        md, = ax[2].plot([m_mu_x[0]], [m_mu_y[0]], 'o', color='C1', ms=15, mec='k', mew=2)        

        progress_bars = [self.progress_bar(a) for a in ax]

        # === Update function ===
        def update(t):
            for b in self.obs_range:
                wave_bars[b].set_height(obs_wave[t][b])
                obs_bars[b].set_height(of[t][b]*obs_height)
            obs_line.set_ydata(obs_wave[t])
            bayes_line.set_ydata(b_dist[t])
            model_line.set_ydata(m_dist[t])

            def update_patch(patch, y):
                x = self.realization_range
                verts = np.column_stack(([x[0], *x, x[-1]], [0, *y, 0]))
                patch.set_xy(verts)

            update_patch(bf_patch, b_dist[t])
            update_patch(mf_patch, m_dist[t])
            bl.set_data(b_mu_x[:t+1], b_mu_y[:t+1])
            ml.set_data(m_mu_x[:t+1], m_mu_y[:t+1])
            bd.set_data(b_mu_x[t], b_mu_y[t])
            md.set_data(m_mu_x[t], m_mu_y[t])    
            progress = (t + 1) / self.step_num
            for bar in progress_bars:
                bar.set_width(progress)
            return list(wave_bars) + [bayes_line, model_line, bf_patch, mf_patch, bd, md]
        
        interval = 1000 // fps  # milliseconds per frame
        self.anim = FuncAnimation(fig, update, frames=self.step_num, interval=interval, blit=False)
        plt.show()
        
    def progress_bar(self, ax, bar_height = 0.02):
        bar = patches.Rectangle((0, 0), 0, bar_height,
            transform=ax.transAxes, color='r', alpha=0.5, clip_on=False)
        ax.add_patch(bar)
        return bar

    def belief_to_circular(self, belief):
        mu = np.sum(belief * self.realization_range[None,:], axis=-1)
        max_val = np.max(belief, axis=-1)
        mu_angle = 2 * np.pi * mu/self.realization_num
        agent_x= max_val * np.cos(mu_angle)
        agent_y = max_val * np.sin(mu_angle)
        return agent_x, agent_y
    
    def show_circular_static(self, gv=0):
        self.angles = np.linspace(0, 2 * np.pi, self.realization_num, endpoint=False)
        b_mu_x, b_mu_y = self.belief_to_circular(self.joint_goal_belief)
        m_mu_x, m_mu_y = self.belief_to_circular(self.model_goal_belief)
        x, y = np.cos(self.angles), np.sin(self.angles)

        gv = self.goal_value[self.b]

        fig, ax = plt.subplots(figsize=(5,5))
        ax.add_artist(plt.Circle((0,0), 1, color='k', fill=False))
        ax.plot(x, y, 'o', ms=6, color='k')
        ax.plot(x[gv], y[gv], 'g*', ms=22)
        ax.plot(0, 0, 'k+', ms=10)
        ax.plot(b_mu_x, b_mu_y, 'o', ms=20, mec='k', mew=2, color='C0')
        ax.plot(m_mu_x, m_mu_y, 'o', ms=20, mec='k', mew=2, color='C1')
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal', adjustable='box'); ax.set_xticks([]); ax.set_yticks([]); ax.set_axisbelow(True)
        plt.show()