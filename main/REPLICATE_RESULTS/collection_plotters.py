import numpy as np; import torch; import os; import sys; import inspect
import pylab as plt; from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.transforms import blended_transform_factory as btf
from matplotlib.ticker import MultipleLocator, NullFormatter
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import matplotlib as mpl
from tqdm import tqdm

path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')

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

class Collection_Plotters():
    def __init__(self, Collector_init_vars):
        self.__dict__.update(vars(Collector_init_vars))
 
    def plot_likelihood(self, save = None):
        #######################################
        """ PLOTTING LIKELHOOD """
        #######################################

        # pick a random trial
        i = np.random.randint(self.batch_num)
        o = np.random.randint(self.obs_num)
        ctx = 1

        g_i = self.goal_ind[0,0,ctx,i]
        gv = self.goal_value[0,0,ctx,i]
        jj = self.expert_likelihood[i,o]
        nj = self.naive_likelihood[i,o]
        R = self.realization_num
        c_i = 1 - g_i

        fig = plt.figure(figsize=(10,4))
        gs = fig.add_gridspec(2,5,
            width_ratios  =(4,0.5,1,4,0.5),
            height_ratios =(4,0.4),
            wspace=0.3, hspace=0.3)

        for v,base in zip((0,1),(0,3)):
            ax_j = fig.add_subplot(gs[0,base])    # joint
            ax_r = fig.add_subplot(gs[0,base+1])  # vertical marginal
            ax_h = fig.add_subplot(gs[1,base])    # horizontal marginal

            disp  = jj.T if g_i==0 else jj
            pg,pc = nj[g_i], nj[c_i]
            joint = (1-disp) if v==0 else disp
            right = ((1-pc)[:,None] if v==0 else pc[:,None])
            top   = ((1-pg)[None,:] if v==0 else pg[None,:])

            ax_j.imshow(joint, aspect='auto', vmin=0, vmax=1)
            ax_r.imshow(right, aspect='auto', vmin=0, vmax=1)
            ax_h.imshow(top,   aspect='auto', vmin=0, vmax=1)

            ax_j.xaxis.tick_top()
            ax_j.xaxis.set_label_position('top')
            ax_j.set(xticks=np.arange(R), yticks=np.arange(R), xlabel=r"$r_g$", ylabel=r"$r_c$")
            ax_j.tick_params(which='both', direction='out', length=4)
            ax_j.set_title(rf"$P_{{{{Z}}}}({{o^{o}={v}}}\mid r_c,r_g)$", pad=15, fontsize = 12)
            ax_r.set_xticks([]); ax_r.set_yticks([])
            ax_r.set_ylabel(rf"$P_{{{{Z}}}}({{o^{o}={v}}}\mid r_c)$", rotation=0, labelpad = -8)
            ax_r.yaxis.set_label_coords(1, -0.1)
            ax_h.set_xticks([]); ax_h.set_yticks([])
            ax_h.set_xlabel(rf"$P_{{{{Z}}}}({{o^{o}={v}}}\mid r_g)$", labelpad=12)
            ax_j.set_frame_on(False)
            ax_r.set_frame_on(True)
            ax_h.set_frame_on(True)
            for ax in [ax_r, ax_h]:
                for side in ('left','right','top','bottom'):
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_linewidth(2)
                for spine in ax.spines.values():
                    spine.set_linewidth(2)

        self.finish_plot(save)

    def upsample(self, Z):
        T = self.step_num
        R = self.realization_num
        A = np.empty((R, self.t2.size))
        B = np.empty((self.r2.size, self.t2.size))
        for j in range(R): A[j] = np.interp(self.t2, np.arange(T), Z[j])
        for k in range(self.t2.size): B[:,k] = np.interp(self.r2, np.arange(R), A[:,k])
        return B

    def plot_belief(self, ctx = 1, eps = 1e-3, save = None):
        #######################################
        """ PLOTTING BELIEF """
        #######################################
        i = np.random.randint(self.batch_num)
        R = self.realization_num
        T = self.step_num

        self.t2 = np.linspace(0, T-1, (T-1)* 10 +1)
        self.r2 = np.linspace(0, R-1, (R-1)* 10 +1)
        Times = np.arange(T)
        X2, Y2 = np.meshgrid(self.t2, self.r2)
        for cond in range(self.pairs):
            fig1, ax1 = plt.subplots(1, 1, figsize=(25,25), subplot_kw={'projection':'3d'})
            fig2, ax2 = plt.subplots(1, 1, figsize=(25,25), subplot_kw={'projection':'3d'})

            for ind, ax in enumerate([ax1, ax2]):
                gv = self.goal_value[cond,ind, ctx, i]
                g_i = self.goal_ind[cond, ind, ctx, i]       
                belief = self.belief[cond, ind, ctx, i].T
                belief[:,0] = 0
                B_up = self.upsample(belief)
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
                    xt = Times[-1]
                    bar_x = np.full(2, xt)
                    bar_y = np.full(2, gv)            
                    ax.plot_surface(X2, Y2, B_up, cmap='rocket', edgecolor='none', zorder = 0,
                                    rstride=1, antialiased=False, norm=norm, alpha = alph)
                for v in range(R):
                    ax.plot(Times, np.full(T, v), belief[v], '-', color = b, alpha = .3, lw=20, zorder=100)
                    ax.plot(Times, np.full(T, v), belief[v], '-', color = b, alpha = 1, lw=5, zorder=200)
                ax.plot(Times, np.full(T, gv), belief[gv], '-g', lw=20, alpha=.3, zorder=200)
                ax.plot(Times, np.full(T, gv), belief[gv], '-', color = 'limegreen', lw=5, zorder=300)
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
            ax1.set_title(self.labels[2 * cond])
            ax2.set_title(self.labels[2 * cond + 1])
            fig1.canvas.draw()
            fig2.canvas.draw()
            if save is None:
                save1=save2=save 
            else:
                save1 = "1_" + save
                save2 = "2_" + save
            self.finish_plot(save1, cond, fig = fig1)
            self.finish_plot(save2, cond, fig = fig2)

    def plot_perf(self, save = None):
        #######################################
        """ PLOTTING PERFORMANCE """
        #######################################
        
        B = self.batch_num 
        T = self.step_num
        t = np.arange(T)
        relative_labels = [r"$\frac{\text{Joint}}{\text{Independent}}$", 
                        r"$\frac{\text{Fully Trained}}{\text{Echo State}}$"]
        markers = ['o', '^']
        mecs = ['k', 'r']

        if self.WITHOUT_net:
            fig, axs = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
        if self.WITH_net:
            # fig, axs = plt.subplots(1, 4, figsize=(16, 5), tight_layout=True,
            # gridspec_kw={'width_ratios':  (1, 1, 1, .4)})
            fig, axs = plt.subplots(1, 4, figsize=(16,5), tight_layout=True,
            gridspec_kw={'width_ratios':  (.7, .7, 1, 1)})
        
        for cond in range(self.pairs):
            marker = markers[cond]
            lw = 2.5
            j_acc = self.accs[cond, 0]
            n_acc = self.accs[cond, 1]
            mj = j_acc.mean(1)
            mn = n_acc.mean(1)
            sdn = n_acc.std(1)/np.sqrt(B)
            sdj = j_acc.std(1)/np.sqrt(B)
            ls = self.linestyles[cond]

            col = plt.cm.viridis(np.linspace(0.15, 0.85, self.ctx_num))
            idx = np.linspace(0, T-1, 4, dtype=int)

            if self.WITH_net:
                for ax in axs:
                    ax.grid(alpha=.3)
                    for s in ('top', 'right'):
                        ax.spines[s].set_visible(False)
                ax = axs[0]
                ax.set_ylim(.2, .8)

            if self.WITHOUT_net:
                for ax_ in axs:
                    for ax in ax_:
                        ax.grid(alpha=.3)
                        for s in ('top', 'right'):
                            ax.spines[s].set_visible(False)
                ax = axs[0, 0]
                ax.set_ylim(0, 1.1)

            for c in range(self.ctx_num):
                if self.WITHOUT_net:
                    label = f'$|C|={c+1}$'
                if self.WITH_net:
                    if c > 0:
                        label = None 
                    elif cond == 0:
                        label = "Joint Inference"
                    else:
                        label = "Fully Trained"

                ax.plot(t, mj[c], c=col[c], lw=lw, ls = ls)
                ax.scatter(t[idx], mj[c, idx], s=50, marker=marker, label = label,
                    facecolors=col[c], edgecolors=mecs[cond], zorder=3)
            title = "Joint Inference" if self.WITHOUT_net else "Experts"

            ax.set(title=title, xlabel='Inference Time', ylabel='Accuracy')
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.legend()

            if self.WITH_net:
                ax = axs[1]
                ax.set_ylim(.2, .8)
            if self.WITHOUT_net:
                ax = axs[0, 1]
                ax.set_ylim(0, 1.1)

            for c in range(self.ctx_num):
                if (c > 0) or (self.WITHOUT_net):
                    label = None 
                elif cond == 0:
                    label = "Independent Inference"
                else:
                    label = "Echo State"

                ax.plot(t, mn[c], c=col[c], lw=lw, ls = ls)
                ax.scatter(t[idx], mn[c, idx], s=50, marker=marker, label = label,
                    facecolors=col[c], edgecolors=mecs[cond], zorder=3)               
            title = "Independent Inference" if self.WITHOUT_net else "Baselines"
            ax.set(title=title, xlabel='Inference Time')
            ax.set_yticks(np.linspace(0, 1, 6))

            # 3) Joint vs Naive scatter–curve

            if self.WITH_net:
                ax.legend()
                ax = axs[2]
            if self.WITHOUT_net:
                ax = axs[1, 0]

            ax.plot([0, 1], [0, 1], ls = '-', c='gray', lw=1, alpha=1)
            for c in range(self.ctx_num):
                if (self.WITH_net):
                    label = f'$|C|={c+1}$ ' + (relative_labels[cond] if c == 0 else "")
                else:
                    label = ""
                ax.plot(mn[c], mj[c], c=col[c], lw=lw, ls = ':' if cond == 0 else '--')
                ax.scatter(mn[c, idx], mj[c, idx], s=50, marker=marker, label = label,
                    facecolors=col[c], edgecolors=mecs[cond], zorder=3)
                if (self.WITHOUT_net) or (cond == 1): 
                    # shows net accs if with net
                    text_col = 'k' if self.WITHOUT_net else 'r'
                    ax.annotate(r"$\times$" + f'{mj[c,-1]/mn[c,-1]:.0f}',(mn[c,-1], mj[c,-1]),
                        xytext=(-10, 10), textcoords='offset points', color = text_col)
            if cond == 0:
                ax.annotate('start', (mn[0,0], mj[0,0]), xytext=(-25, -30), textcoords='offset points')
            if self.WITHOUT_net:
                ax.set(xlabel= 'Independent Inference', ylabel='Joint Inference')
                ax.set_ylim(.1, 1.1)
                ax.set_xlim(.1, 1)
            else:
                ax.set(xlabel= 'Accuracy')
                ax.set_ylim(.1, .85)
                ax.set_xlim(.1, .85)

            ax.set_title("Relative Accuracy")               
            ax.grid(alpha=.3)
            ax.set_yticks([.2, .4, .6, .8], labels=[.2, .4, .6, .8])

            # 4) DKL

            if self.WITHOUT_net:
                ax = axs[1, 1]
                dkl = self.joint_naive_DKL
                for c in range(self.ctx_num):
                    ax.plot(t, dkl[c], c=col[c], ls = '--', lw=1.5)
                    ax.scatter(t[idx], dkl[c, idx], s=50, marker='o',
                        facecolors=col[c], edgecolors='k', zorder=3
                    )
                ax.set(ylabel='KL-divergence', xlabel='Inference Time', title='Semantic Interaction Information')
                max_d = dkl.max() * 1.05
                ax.set_ylim(None, max_d)
                ax.set_yticks([0, 4, 8, 12])

            if self.WITH_net:
                c = 1
                ax.legend(loc = 'lower right')
                ax = axs[3]
                if cond == 0:
                    TJ_dkl = self.net_joint_DKL[0,1]
                    RJ_dkl = self.net_joint_DKL[1,1]
                    TN_dkl = self.net_naive_DKL[0,1]
                    RN_dkl = self.net_naive_DKL[1,1]
                    JT_dkl = self.joint_net_DKL[0,1]
                    JR_dkl = self.joint_net_DKL[1,1]
                    NR_dkl = self.naive_net_DKL[1,1]
                    NT_dkl = self.naive_net_DKL[0,1]
                    JN_dkl = self.joint_naive_DKL[1]
                    NJ_dkl = self.naive_joint_DKL[1]
                    JN_sym = (JN_dkl + NJ_dkl)/2
                    TN_sym = (TN_dkl + NT_dkl)/2
                    JR_sym = (RJ_dkl + JR_dkl)/2
                    NR_sym = (RN_dkl + NR_dkl)/2
                    JT_sym = (TJ_dkl + JT_dkl)/2

                    ax.plot(JT_sym, JR_sym, c = '#CBE79D', zorder = 5)
                    ax.plot(TN_sym, NR_sym, c = '#8A9776', zorder = 5)
                    ax.scatter(JT_sym[idx], JR_sym[idx], marker = 'h', s = 75, c = "#CBE79D", edgecolors = 'k', label = r"$\mathcal{D}_{KL}(\cdot || $" + " Joint" + r"$)$", zorder = 10)
                    ax.scatter(TN_sym[idx], NR_sym[idx], marker = 'h', s = 75, c = "#8A9776", edgecolors = 'k', label =r"$\mathcal{D}_{KL}(\cdot || $" + " Independent" + r"$)$", zorder = 10)
                    ax.scatter(JT_sym[-1], JR_sym[-1], marker = 'h', s = 75, c = "#CBE79D", edgecolors = 'r', zorder = 20, linewidths = 1.5)
                    ax.scatter(TN_sym[-1], NR_sym[-1], marker = 'h', s = 75, c = "#8A9776", edgecolors = 'r', zorder = 20, linewidths = 1.5)

                    ax.plot(np.linspace(-5, JN_sym[-1], 2), np.ones(2) * JN_sym[-1], c = 'r', ls = '--', label = r"$\mathcal{D}_{KL}($"+"Joint" + r"$||$" + " Independent" +  r"$)$", zorder= 0, lw = 1)
                    ax.plot(np.ones(2) * JN_sym[-1], np.linspace(-5, JN_sym[-1], 2), c = 'r', ls = '--', zorder= 0, lw = 1)
                    ax.set_xlabel(r"$\mathcal{D}_{KL}($"+"Fully Trained " + r"$|| \cdot)$")
                    ax.set_ylabel(r"$\mathcal{D}_{KL}($"+"Echo State " + r"$|| \cdot)$")
                    ax.set_title("Relative " + r"$\mathcal{D}_{KL}$")
                    ax.set_xlim([-.3, JN_sym[-1] + .1])
                    ax.set_ylim([-1.3, JN_sym[-1] + .1])
                    ax.set_xticks([0,1,2,3])
                    ax.set_yticks([-1, 0,1,2,3])
                    ax.legend(loc = 'lower right')
                    ax.annotate('start', (0,0), xytext=(-15, -15), textcoords='offset points')


        self.finish_plot(save)

    def plot_density_curves(self, bins=60, sigma=0, ymax=1, save=None):
        B=self.batch_num; T=self.step_num
        edges=np.linspace(0,1,bins+1); x=0.5*(edges[:-1]+edges[1:])
        chance=1/self.realization_num
        fig,axses=plt.subplots(2,2,figsize=(8,5.5), tight_layout=True)
        for ax in axses.ravel():
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

        for r,agent in enumerate(("bayes","net")):
            axs=axses[r]; TP=self.goal_TP[0] if agent=="bayes" else self.goal_TP[1]
            prefix=["Joint Inference","Independent Inference"] if agent=="bayes" else ["Fully Trained Network","Echo State Network"]
            expert,baseline=TP[0,1],TP[1,1]
            pj=np.zeros((T,bins)); pn=np.zeros((T,bins))
            for t in range(T):
                cj,_=np.histogram(expert[:,t],bins=edges); sj=cj.sum(); pj[t]=cj/sj if sj>0 else 0.0
                cn,_=np.histogram(baseline[:,t],bins=edges); sn=cn.sum(); pn[t]=cn/sn if sn>0 else 0.0
            pj=self._smooth_axis(pj,sigma,axis=1); pn=self._smooth_axis(pn,sigma,axis=1)

            blue=np.array([31,119,180])/255.0; red=np.array([214,39,40])/255.0
            a=(np.arange(T)/(T-1 if T>1 else 1))[:,None]; colors=(1.0-a)*blue+a*red

            for t in range(T):
                axs[0].plot(x,pj[t],color=colors[t],lw=1.5, zorder = -100)
                axs[1].plot(x,pn[t],color=colors[t],lw=1.5, zorder = -100)
            for t in range(T):
                alph=.3/(1+np.log(t+1))
                axs[0].fill_between(x,0,pj[t],color=colors[t],alpha=alph, zorder = 100)
                axs[1].fill_between(x,0,pn[t],color=colors[t],alpha=alph, zorder = 100)

            for ax,title in zip(axs,prefix):
                ax.set_xlim(0,1); ax.set_ylim(0,ymax); ax.set_title(title)
                ax.axvline(chance,color='k',ls='--', zorder = 1000, alpha = .5)
            axs[0].set_ylabel("PDF")

        for ax in axses.ravel():
            ax.xaxis.set_major_locator(MultipleLocator(0.2))
            ax.yaxis.set_major_locator(MultipleLocator(0.03))
            ax.set_axisbelow(True)
            ax.grid(True,which='major',axis='both',alpha=0.35)

        for r in (0,1):
            for c in (0,1):
                ax=axses[r,c]
                ax.xaxis.set_major_formatter(NullFormatter())
                ax.tick_params(axis='x',length=0)
                ax.set_xlim(.01,.99)

        for r in (0,1):
            axses[r,0].ticklabel_format(axis='y', style='sci', scilimits=(-2, -2), useMathText=True)
            off = axses[r,0].yaxis.get_offset_text()
            off.set_x(-0.1)           

            axses[r,1].yaxis.set_major_formatter(NullFormatter())
            axses[r,1].tick_params(axis='y',length=0)

            # ax=axses[r,1]; 
            # ax.tick_params(axis='y',length=0)
            # axses[r,0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.3f}'))

        # axses[1,0].set_xlabel(r"$B_{tg \mathbf{a}^\star}$",color='k',labelpad=1)
        # axses[1,1].set_xlabel(r"$B_{tg \mathbf{a}^\star}$",color='k',labelpad=1)
        # tx=btf(axses[0,1].transData,axses[0,1].transAxes)
        # axses[0,1].text(chance,-0.05,r"$\mathbf{\mathrm{Chance}}$",transform=tx,ha='center',va='top',color='dimgrey',clip_on=False)
        # tr=btf(axses[1,1].transData,axses[1,1].transAxes)
        # axses[1,1].text(0.0,-0.03,r"$0$",transform=tr,ha='center',va='top',clip_on=False)
        # axses[1,1].text(1.0,-0.03,r"$1$", transform=tr,ha='center',va='top',clip_on=False)        
        # axses[1,1].text(0.0,-0.13,r"$P(\mathrm{Miss})$",transform=tr,ha='center',va='top',clip_on=False)
        # axses[1,1].text(1.0,-0.13,r"$P(\mathrm{Hit})$", transform=tr,ha='center',va='top',clip_on=False)
        # axses[1,0].text(0.0,-0.13,r"$P(\mathrm{Miss})$",transform=tr,ha='center',va='top',clip_on=False)
        # axses[1,0].text(1.0,-0.13,r"$P(\mathrm{Hit})$", transform=tr,ha='center',va='top',clip_on=False)
        sm=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0,vmax=T-1),
            cmap=mpl.colors.LinearSegmentedColormap.from_list("blue_to_red",[blue,red],N=max(T,2)))
        sm.set_array([])
        cb_ax=inset_axes(axses[0,1],width="70%",height="3%",loc="upper right",borderpad=.5)
        cbar=fig.colorbar(sm,cax=cb_ax,orientation='horizontal'); cbar.set_label("Inference Time",labelpad=5,y=2); cbar.set_ticks([])

        self.finish_plot(save)



    def _smooth_axis(self, A, sigma, axis):
        if sigma <= 0: return A
        k, r = self._gauss1d(sigma)
        if axis == 1:  # smooth along Value bins
            T, B = A.shape; P = np.empty_like(A)
            for t in range(T):
                row = np.pad(A[t], r, mode='reflect')
                P[t] = np.convolve(row, k, mode='valid')
            return P
        else:          # smooth along Time
            T, B = A.shape; P = np.empty_like(A)
            for b in range(B):
                col = np.pad(A[:, b], r, mode='reflect')
                P[:, b] = np.convolve(col, k, mode='valid')
            return P

    def _gauss1d(self, s):
        r = int(3*s+0.5); x = np.arange(-r, r+1)
        k = np.exp(-0.5*(x/s)**2); return k/k.sum(), r


    def finish_plot(self, save, cond = "", fig = None):
        if save is not None:
            if fig is not None:
                fig.savefig(f"{cond}_{save}")
            else:
                plt.savefig(f"{cond}_{save}")
        plt.show()

