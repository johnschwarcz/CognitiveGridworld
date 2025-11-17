import seaborn as sns; import numpy as np; import matplotlib.patches as patches;
import matplotlib.colors as mcolors; import pylab as plt

class Plotting_helpers():
   
    ################################
    """ general plotting helpers """ 
    ################################
    def ctx_col(self, c=None):
        cond1 = c is not None and self.is_goal(c)
        cond2 = c is None and self.is_goal(self.k, self.q)
        return self.goal_col if cond1 or cond2 else 'orange'
    
    def is_goal(self, c1, c2 = -1):
        g = self.goal_ind[self.b]
        if (c1 == g) or (c2 == g):
            return True
        return False

    def simple_heat(self, data, ax, cmap='rocket', vmax=1.1):
        sns.heatmap(data, ax=ax, vmin=-.1, vmax=vmax, cmap=cmap,
            cbar=False, yticklabels=False, xticklabels=False)
        ax.invert_yaxis()                 

    def obs_title(self, ax, i):
        title_obs = 1 + self.o
        ax.set_title(f"obs {-title_obs if i else title_obs}", fontsize=20)      
 
    ##############################
    """ plot_likelihood helper """
    ##############################
    def plot_likelihood_helper(self, h, w, L, title): 
        if (title == "naive") or (self.ctx_num == 1):
            fig, self.axes = plt.subplots(1, 2, figsize=(w, h), squeeze = False)
            self.fill_row_heat(L)
            for c in self.ctx_range:
                self.fill_row_patch((self.ctx_vals[self.b][c], c), self.ctx_col(c))              
            self.naive_heat_labeling()
        else:
            fig, self.axes = plt.subplots(self.rows, 2, figsize=(w, h * self.rows), squeeze = False)
            self.fill_joint_heat(L)
        fig.suptitle(f"{title} likelihood (batch {self.b})", fontsize = 30)
        plt.subplots_adjust(hspace=0.5, top = .7)     
        plt.show()    
        
    def fill_joint_heat(self, L, row = 0):
        vals = self.ctx_vals[self.b]
        for self.k in self.ctx_range:
            for self.q in range(self.k + 1, self.ctx_num):
                curr_L = self.avg_over_ctx(L, includes_obs = True)
                x, y = (self.k, self.q) if self.k < self.q else (self.q, self.k)
                self.joint_heat_labeling(self.axes[row, -1], x,y)
                self.fill_row_heat(curr_L, row)
                self.fill_row_patch((vals[self.k], vals[self.q]), self.ctx_col(), row)
                row += 1

    def fill_row_heat(self, L, row=0):     
        L = L[self.b, self.o]
        if L.ndim == 1: 
            L = L[None, :]
        for i, p_x in enumerate([L, 1-L]):        
            ax = self.axes[row, i]       
            self.simple_heat(p_x, ax=ax)                
            self.obs_title(ax, i)
            
    def fill_row_patch(self, val, col, row=0):
        for ax in self.axes[row]:
            ax.add_patch(patches.Rectangle(val, 1, 1, 
               edgecolor = col, facecolor='none', linewidth=2))  

    def joint_heat_labeling(self, ax, x,y):
        ax.text(1.05, 0.5, "ctx\n  " f"{y}",fontsize=21,va='center',ha='left',transform=ax.transAxes)
        ax.text(.5, -.2, f"ctx {x}",fontsize=21,va='bottom',ha='center',transform=ax.transAxes)      
        
    def naive_heat_labeling(self):
        ax = self.axes[0, -1]
        for c in self.ctx_range:
            title = self.ctx_range[c]
            y = (1 + c)/self.ctx_num - .5/self.ctx_num
            ax.text(1.05, y, f"ctx {title}",fontsize=21,va='center',ha='left',transform=ax.transAxes)
            ax.text(.5, -.2, "realizations",fontsize=21,va='bottom',ha='center',transform=ax.transAxes)      

    ##########################
    """ plot_trial helpers """
    ##########################
    def plot_trial_helper(self, ax):
        xlim, ylim = [0, self.step_num], [0, self.realization_num]
        xax = self.step_range[:,None].repeat(self.realization_num, -1)
        yax = self.realization_range[None, :].repeat(self.step_num, 0) + 0.5
        zips = zip(self.agent_beliefs, self.agent_titles, [200, 40], self.agent_col_maps)
        for i, (belief, agent, s, cmap) in enumerate(zips):
            if i == 0:
                self.plot_trial_stim(ax)   
            for c in self.ctx_range:
                self.plot_trial_belief(ax, belief, c, xlim, ylim, s, cmap, xax, yax)

    def plot_trial_stim(self, ax):
        # Plot stimulus for the trial.
        cmap = mcolors.ListedColormap(['black'] + sns.color_palette("cool", self.obs_num))
        P = 1 - (self.pobs__joint[self.b]**2).reshape(-1, 1)
        self.simple_heat(P, ax[0, 1], 'binary')             
        self.simple_heat(self.obs_flat[self.b].T, ax[0, 0], cmap, vmax= self.obs_num)
            
    def plot_trial_belief(self, ax, belief, c, xlim, ylim, s, cmap, xax, yax):
        # Plot belief states for each context in the trial.
        right = ax[c + 1, 1];        left = ax[c + 1, 0]
        left.set_ylim(ylim);         left.set_xlim(xlim)
        right.set_xticks([]);        right.set_yticks([]);        right.set_ylim(ylim)
        
        col = self.ctx_col(c)
        v = self.ctx_vals[self.b, c]
        belief_col = cmap(belief[self.b, :, c].flatten())
        
        right.add_patch(patches.Rectangle((0,v), 1, 1, facecolor=col, edgecolor='k'))
        left.axhline(v + .5, linestyle ='--', c = col, zorder = 10, linewidth = 3)
        left.scatter(xax, yax, marker = 's', c = belief_col, s = s)

    def plot_trial_postprocess(self, fig, ax):
        fig.suptitle(f"batch {self.b}", fontsize = 30)
        ax[-1, 0].set_xlabel("time", fontsize=30)
        fig.tight_layout()
        plt.show()     

    ################################
    """ plot_performance helpers """
    ################################
    def plot_agent_perf(self, axes, samples):
        # Plot single trial and trial average performance. 
        sample_inds = np.random.choice(self.batch_range, size = samples)
        perf_list = [self.agent_accs, self.agent_TPs, self.agent_mses]
        ylims = [[-.1, 1.1], [-.1, 1.1], [-.1, self.realization_num]]
        titles = ["accuracy", "p(correct)", "MSE"]
        alphas = [.6, .4]
        
        for ax, perfs, title, ylim in zip(axes, perf_list, titles, ylims):
            for perf, col, label, alpha in zip(perfs, self.agent_cols,  self.agent_titles, alphas):
                y_samples = perf[sample_inds].T
                y_std = perf.std(0)
                y = perf.mean(0)
                
                ax.plot(y_samples, c = col, zorder = -1, alpha = alpha)
                ax.plot(self.step_range, y, c = col, linewidth = 4, label = label)
                ax.fill_between(self.step_range, y-y_std, y+y_std, color = col, alpha = .1)  
                for y, s, w1 in zip([y, y-y_std, y+y_std], ['-', '--', '--'],[4, .5, .5]):
                    for c, w2, a in zip(['k', col],[2, 1], [.5, 1]):
                        ax.plot(self.step_range, y, linestyle = s, 
                              c = c, linewidth = w1*w2, alpha = a)
                
            ax.set_title(title, fontsize = 20)
            ax.legend(fontsize = 20)
            ax.set_ylim(ylim)
        
    def plot_performance_postprocess(self, fig, ax):
        for j in range(3):
            ax[j].set_xlabel("Time", fontsize = 30)
            ax[j].set_xlim([0,self.step_num])
        fig.suptitle("Performance")
        fig.tight_layout()
        plt.show()
