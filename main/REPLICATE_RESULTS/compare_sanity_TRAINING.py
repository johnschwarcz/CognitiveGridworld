import numpy as np; import torch; import os; import sys; import inspect; import pylab as plt;
import matplotlib.pyplot as plt; from scipy.optimize import curve_fit
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld 

if __name__ == "__main__":
    cuda = 0
    realization_num = 10
    step_num = 30
    hid_dim = 1000
    state_num = 500
    obs_num = 5
    training = False 
    batch_num = 8000 if training else 5000
    episodes = 50000 if training else 1

    echo = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5, 
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 2, 
        'training': training,
        'save_env': "/sanity/reservoir_ctx_2_e5" if training else None,
        'load_env': None if training else "/sanity/reservoir_ctx_2_e5"})
    if training:
        del(echo)

    trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False, 
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .001, 'ctx_num': 2, 
        'training': training,
        'save_env': "/sanity/fully_trained_ctx_2_e5" if training else None,
        'load_env': None if training else "/sanity/fully_trained_ctx_2_e5"})
    if training:
        del(trained)

    echo_1 = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 1,
        'training': training,
        'save_env': "/sanity/reservoir_ctx_1_e5" if training else None,
        'load_env': None if training else "/sanity/reservoir_ctx_1_e5"})
    if training:
        del(echo_1)

    trained_1 = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num, 'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .001, 'ctx_num': 1,
        'training': training,
        'save_env': "/sanity/fully_trained_ctx_1_e5" if training else None,
        'load_env': None if training else "/sanity/fully_trained_ctx_1_e5"})         
    if training:
        del(trained_1)  
    
    # #############################################

    # fig, ax = plt.subplots(2, 2, figsize = (7, 5), tight_layout = True) 
    # ax[0,0].plot(trained.test_acc_through_training[:,-1])
    # ax[0,0].plot(echo.test_acc_through_training[:,-1])
    # ax[0,1].plot(trained.test_SII_coef_through_training)
    # ax[0,1].plot(echo.test_SII_coef_through_training)
    # ax[1,0].plot(trained.test_net_joint_DKL_through_training[:,-1], c= 'C0', lw = 4)
    # ax[1,0].plot(echo.test_net_joint_DKL_through_training[:,-1], c = 'C1', lw = 4)
    # ax[1,1].plot(trained.test_net_naive_DKL_through_training[:,-1], c = 'C0', lw = 4)
    # ax[1,1].plot(echo.test_net_naive_DKL_through_training[:,-1], c = 'C1', lw = 4)
    # plt.show()

    # ############################################# Dimensionality at end of training

    # ms = 5
    # mew = .5
    # rm_PC1 = False

    # fig, ax = plt.subplots(2,2, figsize = (10, 6), tight_layout = True)
    # xax = np.arange(trained.hid_dim)
    # if rm_PC1:
    #     xax = xax[:-1]
    # xax = xax + 1


    # for i in range(2):

    #     var = trained.test_model_input_dim_through_training
    #     if rm_PC1:
    #         var = var[:, 1:]
    #     ax[i, 0].plot(xax, var[0], '-o', c = 'k', ms = ms/2, alpha = .5, label = "init")
    #     ax[i, 0].plot(xax, var[-1], '-o', c = 'g', ms = ms,  mec = 'g', mew = mew, label = "trained")

    #     var = echo.test_model_input_dim_through_training
    #     if rm_PC1:
    #         var = var[:, 1:]
    #     ax[i, 0].plot(xax, var[-1], '-o', c = 'r', alpha = 1,  ms = ms, mec = 'r', mew = mew, label = "random")
    #     ax[i, 0].legend()
        
    #     var = trained.test_model_input_dim_ratio_through_training
    #     if rm_PC1:
    #         var = var[:, 1:]
    #         var = var/var.sum(-1, keepdims = True)
    #     ax[i, 1].plot(xax, var[0], '-o', c = 'k', ms = ms/2, alpha = .5, label = "init")
    #     ax[i, 1].plot(xax, var[-1], '-o', c = 'g', ms = ms,  mec = 'g', mew = .5, label = "trained")
        
    #     var = echo.test_model_input_dim_ratio_through_training
    #     if rm_PC1:
    #         var = var[:, 1:]
    #         var = var/var.sum(-1, keepdims = True)
    #     ax[i, 1].plot(xax, var[-1], '-o', c = 'r', alpha = 1,  ms = ms, mec = 'r', mew = mew, label = "random")

    #     if i == 1:
    #         ax[i, 1].set_yscale('log')
    #         ax[i, 0].set_yscale('log')
    #     ax[i, 0].set_ylabel("var exp (log scale)" if i == 1 else "var exp")
    #     ax[i, 1].set_ylabel("var exp (log scale)" if i == 1 else "var exp")
    #     ax[i, 0].set_title("Abs")
    #     ax[i, 1].set_title("Ratio")
    #     ax[i, 1].set_xscale('log')
    #     ax[i, 0].set_xscale('log')
    #     ax[i, 1].legend()
    # plt.show()

    ############################################# Participation ratio through training

    # def pr(evals):
    #     e = np.asarray(evals)
    #     s1 = e.sum(1)
    #     s2 = (e * e).sum(1)
    #     return (s1 * s1) / s2
    
    # xax = np.arange(len(trained.test_model_update_dim_ratio_through_training)) + 1
    # var = pr(trained.test_model_update_dim_through_training)
    # plt.plot(xax , var , '-', c = 'C0'); 
    # var = pr(echo.test_model_update_dim_through_training)
    # plt.plot(xax , var , '-', c = 'C1')

    # xax = np.arange(len(trained_1.test_model_update_dim_ratio_through_training))  + 1
    # var = pr(trained_1.test_model_update_dim_through_training)
    # plt.plot(xax , var , '-', c = 'C0', alpha = .5); 
    # var = pr(echo_1.test_model_update_dim_through_training)
    # plt.plot(xax , var , '-', c = 'C1' , alpha = .5)

    # plt.yscale('log')
    # plt.xscale('log')
    # plt.show()


    ############################################# Participation ratio and related metrics through training
    # lim = 600
    # fig, ax = plt.subplots(1, 5, figsize=(20, 3), constrained_layout=True)
    # pairs = (("Trained", trained, trained_1, "C0", 1e1),("Echo",    echo,    echo_1,    "C1", 1.0),)

    # for name, run0, run1, c, sii_scale in pairs:
    #     for run, alpha, suffix in ((run0, 1.0, "run 1"), (run1, 0.5, "run 2")):
    #         y = np.asarray(run.test_model_update_dim_through_training)
    #         x =  np.arange(y.shape[0]) + 1
    #         pr_y = pr(y)
    #         s = y.sum(-1)
    #         ax[0].plot(x, pr_y, "-", c=c, alpha=alpha, label=f"{name} ({suffix})")
    #         ax[1].plot(x, s,    "-", c=c, alpha=alpha)
    #         ax[2].plot(x, pr_y / s, "-", c=c, alpha=alpha)
    #         ax[3].plot(x, pr_y + s, "-", c=c, alpha=alpha)
    #         ax[4].plot(x, np.asarray(run.test_SII_coef_through_training) * sii_scale, "-", c=c, alpha=alpha)

    # titles = (
    #     "dimensionality (PR)",
    #     "magnitude (sum)",
    #     "PR / sum",
    #     "PR + sum",
    #     "SII coefficient (trained ×10)")
    # ylabs = ("PR", "sum", "PR/sum", "PR+sum", "SII")
    # for i in range(5):
    #     ax[i].set_title(titles[i])
    #     ax[i].set_xlabel("Epoch")
    #     ax[i].set_ylabel(ylabs[i])
    # for i in (0, 1, 3, 4):
    #     ax[i].set_xlim(1, lim)
    # ax[2].set_xlim(1, lim)
    # h, l = ax[0].get_legend_handles_labels()
    # fig.legend(h, l, ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.15))
    # plt.show()


################################################ 3D Gradient through training

# def plot_evr3d4(trained, trained_1, echo, echo_1, stride=5, max_pc=20, logz=True, elev=25, azim=-60, til = -1, w = 10, h = 5):
#     def draw(ax, R, title):
#         R = np.asarray(R)[::stride, :max_pc].T
#         t = (np.arange(R.shape[1]) * stride + 1).astype(float)
#         pc = (np.arange(R.shape[0]) + 1).astype(float)
#         T, P = np.meshgrid(t, pc)
#         ax.plot_surface(T, P, np.log10(R) if logz else R)
#         ax.view_init(elev=elev, azim=azim)
#         ax.set_title(title)
#         ax.set_xlabel("epoch")
#         ax.set_ylabel("PC")
#         ax.set_zlabel("log10(EVR)" if logz else "EVR")

#     fig = plt.figure(figsize=(w, h))
#     ax1 = fig.add_subplot(221, projection="3d")
#     ax2 = fig.add_subplot(222, projection="3d")
#     ax3 = fig.add_subplot(223, projection="3d")
#     ax4 = fig.add_subplot(224, projection="3d")
#     draw(ax1, trained.test_model_input_dim_through_training[:til], "trained")
#     draw(ax2, trained_1.test_model_input_dim_through_training[:til], "trained_1")
#     draw(ax3, echo.test_model_input_dim_through_training[:til], "echo")
#     draw(ax4, echo_1.test_model_input_dim_through_training[:til], "echo_1")
#     plt.tight_layout()
#     plt.show()

# plot_evr3d4(trained, trained_1, echo, echo_1, stride = 1, max_pc = 999, elev = 30, azim = 30, til = 15)


################################################ FULL FIGURE

def pr(evals, eps=1e-12):
    e = np.asarray(evals, float)
    s1 = e.sum(-1)
    s2 = (e*e).sum(-1)
    return (s1*s1) / (s2 + eps)

def sym_dkl_pair(P, Q, eps=1e-4):
    P = np.clip(P, eps, 1.0 - eps); Q = np.clip(Q, eps, 1.0 - eps)
    P = P / P.sum(-1, keepdims=True); Q = Q / Q.sum(-1, keepdims=True)
    dPQ = np.sum(P * (np.log(P) - np.log(Q)), axis=-1)
    dQP = np.sum(Q * (np.log(Q) - np.log(P)), axis=-1)
    return 0.5 * (dPQ + dQP)

def make_panel_figure(trained, echo,
                      layout=(("ACC","CORR","DIFF"),
                              ("GRAD","PR","DIST"),
                              ("PROUT","PRIN","PRFMS")),
                      smooth_w=100, figsize=(12, 9), lim=None):
    def smooth(y, w=smooth_w):
        y = np.asarray(y, float)
        if w <= 1: return y
        k = np.ones(w, float) / w
        p = w // 2
        yp = np.pad(y, (p, p), mode="edge")
        return np.convolve(yp, k, mode="valid")

    fig, ax = plt.subplots(3, 3, figsize=figsize, tight_layout=True)

    def _panel_ACC(a):
        a.set_title("ACC (final step)")
        y0 = smooth(trained.test_acc_through_training[:, -1]); x0 = np.arange(y0.size) + 1
        y1 = smooth(echo.test_acc_through_training[:, -1]);   x1 = np.arange(y1.size) + 1
        a.plot(x0, y0, c="C0", lw=3); a.plot(x1, y1, c="C1", lw=3)
        if hasattr(trained, "naive_acc"): a.axhline(np.asarray(trained.naive_acc)[:, -1].mean(), c="r", ls="--")
        if hasattr(trained, "joint_acc"): a.axhline(np.asarray(trained.joint_acc)[:, -1].mean(), c="g", ls="--")
        a.set_ylim(0, .8)
        a.legend(("Trained", "Echo", "naive mean", "joint mean") if (hasattr(trained,"naive_acc") and hasattr(trained,"joint_acc"))
                 else ("Trained","Echo"), frameon=False)

    def _panel_CORR(a):
        a.set_title("CORR (SII coef)")
        y0 = smooth(trained.test_SII_coef_through_training); x0 = np.arange(y0.size) + 1
        y1 = smooth(echo.test_SII_coef_through_training);   x1 = np.arange(y1.size) + 1
        a.plot(x0, y0, c="C0", lw=3); a.plot(x1, y1, c="C1", lw=3)
        a.legend(("Trained","Echo"), frameon=False)

    def _panel_DIFF(a):
        a.set_title("DKL(naive) - DKL(joint)  (final step)")
        y0 = smooth((trained.test_net_naive_DKL_through_training - trained.test_net_joint_DKL_through_training)[:, -1])
        y1 = smooth((echo.test_net_naive_DKL_through_training    - echo.test_net_joint_DKL_through_training)[:, -1])
        x0 = np.arange(y0.size) + 1; x1 = np.arange(y1.size) + 1
        a.plot(x0, y0, c="C0", lw=3); a.plot(x1, y1, c="C1", lw=3)
        a.axhline(0, c="k", ls="--")
        a.legend(("Trained","Echo"), frameon=False)

    def _panel_GRAD(a):
        a.set_title("GRAD (|readout| - |readin|)")
        y0 = smooth(trained.readout_grad_log_through_training) - smooth(trained.readin_grad_log_through_training)
        y1 = smooth(echo.readout_grad_log_through_training)    - smooth(echo.readin_grad_log_through_training)
        x0 = np.arange(y0.size) + 1; x1 = np.arange(y1.size) + 1
        a.plot(x0, y0, c="C0", lw=3); a.plot(x1, y1, c="C1", lw=3)
        a.axhline(0, c="k", ls="--")
        a.legend(("Trained","Echo"), frameon=False)

    def _panel_PR(a):
        a.set_title("PR (output) - PR (input)")
        y0 = smooth(pr(trained.test_model_update_stim_dim_through_training) - (pr(trained.test_model_input_stim_dim_through_training) + 1e-12))
        y1 = smooth(pr(echo.test_model_update_stim_dim_through_training)    - (pr(echo.test_model_input_stim_dim_through_training) + 1e-12))
        x0 = np.arange(y0.size) + 1; x1 = np.arange(y1.size) + 1
        a.plot(x0, y0, c="C0", lw=3, ls="--"); a.plot(x1, y1, c="C1", lw=3, ls="--")
        y0 = smooth(pr(trained.test_model_update_dim_through_training) - (pr(trained.test_model_input_dim_through_training) + 1e-12))
        y1 = smooth(pr(echo.test_model_update_dim_through_training)    - (pr(echo.test_model_input_dim_through_training) + 1e-12))
        x0 = np.arange(y0.size) + 1; x1 = np.arange(y1.size) + 1
        a.plot(x0, y0, c="C0", lw=3); a.plot(x1, y1, c="C1", lw=3)
        a.axhline(0, c="k", ls="--")
        m = np.max(np.abs(np.concatenate((y0, y1))))
        if m > 0: a.set_ylim(-1.05*m, 1.05*m)
        a.legend(("Trained (final step)","Echo (final step)","Trained full","Echo full"), frameon=False)

    def _panel_DIST(a):
        a.set_title("DKL(B_0, B_t) (Belief distance from step 1)")
        JGB = trained.joint_goal_belief
        NGB = trained.naive_goal_belief
        FT  = trained.model_goal_belief
        RS  = echo.model_goal_belief
        T = min(int(trained.step_num), int(echo.step_num)) - 1
        means = np.zeros((4, T), float)
        for k in range(T):
            means[0, k] = sym_dkl_pair(JGB[:, 0], JGB[:, k+1]).mean()
            means[1, k] = sym_dkl_pair(FT[:, 0],  FT[:, k+1]).mean()
            means[2, k] = sym_dkl_pair(NGB[:, 0], NGB[:, k+1]).mean()
            means[3, k] = sym_dkl_pair(RS[:, 0],  RS[:, k+1]).mean()
        x = np.arange(T) + 1
        a.plot(x, means[0], "-o", ms=3); a.plot(x, means[1], "-o", ms=3)
        a.plot(x, means[2], "-o", ms=3); a.plot(x, means[3], "-o", ms=3)
        a.set_xscale("log"); a.set_yscale("log")
        a.legend(("joint","FT","naive","echo"), frameon=False)

    # --- NEW: 3rd-row PR panels ---
    def _panel_PROUT(a):
        a.set_title("PR(output)")
        y0s = smooth(pr(trained.test_model_update_stim_dim_through_training))
        y1s = smooth(pr(echo.test_model_update_stim_dim_through_training))
        y0f = smooth(pr(trained.test_model_update_dim_through_training))
        y1f = smooth(pr(echo.test_model_update_dim_through_training))
        x0 = np.arange(y0f.size) + 1; x1 = np.arange(y1f.size) + 1
        a.plot(x0, y0s, c="C0", lw=3, ls="--"); a.plot(x1, y1s, c="C1", lw=3, ls="--")
        a.plot(x0, y0f, c="C0", lw=3);         a.plot(x1, y1f, c="C1", lw=3)
        a.legend(("Trained (final step)","Echo (final step)","Trained full","Echo full"), frameon=False)

    def _panel_PRIN(a):
        a.set_title("PR(input)")
        y0s = smooth(pr(trained.test_model_input_stim_dim_through_training))
        y1s = smooth(pr(echo.test_model_input_stim_dim_through_training))
        y0f = smooth(pr(trained.test_model_input_dim_through_training))
        y1f = smooth(pr(echo.test_model_input_dim_through_training))
        x0 = np.arange(y0f.size) + 1; x1 = np.arange(y1f.size) + 1
        a.plot(x0, y0s, c="C0", lw=3, ls="--"); a.plot(x1, y1s, c="C1", lw=3, ls="--")
        a.plot(x0, y0f, c="C0", lw=3);         a.plot(x1, y1f, c="C1", lw=3)
        a.legend(("Trained (final step)","Echo (final step)","Trained full","Echo full"), frameon=False)

    def _panel_PRFMS(a):
        a.set_title("PR(time) = PR(final step) - PR(full)")
        d0u =  pr(trained.test_model_update_stim_dim_through_training) - pr(trained.test_model_update_dim_through_training) + 1e-12
        d1u =  pr(echo.test_model_update_stim_dim_through_training) - pr(echo.test_model_update_dim_through_training) + 1e-12
        y0u = smooth(d0u)
        y1u = smooth(d1u)
        x0 = np.arange(y0u.size) + 1; x1 = np.arange(y0u.size) + 1
        a.plot(x0, y0u, c="C0", lw=3)
        a.plot(x1, y1u, c="C1", lw=3)
        a.axhline(0, c="k", ls="--")
        m = np.max(np.abs(np.concatenate((y0u, y1u))))
        if m > 0: a.set_ylim(-1.05*m, 1.05*m)
        a.legend(("Trained (output)","Echo (output)"), frameon=False)

    panels = {
        "ACC": _panel_ACC, "CORR": _panel_CORR, "DIFF": _panel_DIFF,
        "GRAD": _panel_GRAD, "PR": _panel_PR, "DIST": _panel_DIST,
        "PROUT": _panel_PROUT, "PRIN": _panel_PRIN, "PRFMS": _panel_PRFMS,
    }

    order = (layout[0][0], layout[0][1], layout[0][2],
             layout[1][0], layout[1][1], layout[1][2],
             layout[2][0], layout[2][1], layout[2][2])

    for i in range(9):
        k = order[i]
        r = i // 3
        c = i - 3*r
        panels[k](ax[r, c])
        if lim is not None: ax[r, c].set_xlim(1, lim)

    return fig, ax


def plot_combined_variances(trained, echo):
    models = (trained, echo)
    model_labels = ("Trained RNN", "Echo State")
    model_colors = ("C0", "C1")
    joint_color = "C2"
    naive_color = "C3"

    R = trained.realization_num
    D = 4
    xax = trained.step_range + 1

    # --- Pre-calculate weights (Same as original) ---
    idx = np.arange(R)
    delta = np.abs(idx[:, None] - idx[None, :])
    circ_dist = np.minimum(delta, R - delta).astype(np.int64)

    weights = np.zeros((R, D, R), dtype=np.float64)
    for r1 in range(R):
        for d in range(D):
            m = (circ_dist[r1] == d)
            cnt = m.sum()
            if cnt > 0:
                weights[r1, d, m] = 1.0 / cnt

    total_cols = 2 + D
    fig, axes = plt.subplots(1, total_cols, figsize=(15, 3.5))
    
    ax_bar = axes[0]      # New Bar plot
    ax_avg = axes[1]      # Old Fig1, Panel 3
    ax_dists = axes[2:]   # Old Fig2

    bar_data = []
    for i, self in enumerate(models):
        c = model_colors[i]

        inp_val = np.zeros((R, R, self.step_num))
        out_val = np.zeros((R, R, self.step_num))
        belief_mu = np.zeros((R, R, self.step_num))
        joint_mu = np.zeros((R, R, self.step_num))
        naive_mu = np.zeros((R, R, self.step_num))
        belief = np.zeros((R, R, self.step_num, D))
        joint = np.zeros((R, R, self.step_num, D))
        naive = np.zeros((R, R, self.step_num, D))

        for r1 in self.realization_range:
            for r2 in self.realization_range:
                inds = (self.goal_ind == 0) & (self.ctx_vals[:, 0] == r1) & (self.ctx_vals[:, 1] == r2)
                if inds.sum() > 1:
                    # Model Inputs/Outputs
                    inp_val[r1, r2] = self.model_input_flat[inds].var(0).mean(-1)
                    out_val[r1, r2] = self.model_update_flat[inds].var(0).mean(-1)

                    # Beliefs
                    v_bel = self.model_goal_belief[inds].var(0)
                    v_jnt = self.joint_goal_belief[inds].var(0)
                    v_nai = self.naive_goal_belief[inds].var(0)
                    belief_mu[r1, r2] = v_bel.mean(-1)
                    joint_mu[r1, r2] = v_jnt.mean(-1)
                    naive_mu[r1, r2] = v_nai.mean(-1)

                    W = weights[r1]
                    belief[r1, r2] = v_bel @ W.T
                    joint[r1, r2]  = v_jnt @ W.T
                    naive[r1, r2]  = v_nai @ W.T
                else:
                    inp_val[r1, r2] = np.nan
                    out_val[r1, r2] = np.nan
                    belief_mu[r1, r2] = np.nan
                    joint_mu[r1, r2] = np.nan
                    naive_mu[r1, r2] = np.nan
                    belief[r1, r2, :, :] = np.nan
                    joint[r1, r2, :, :] = np.nan
                    naive[r1, r2, :, :] = np.nan

        # --- Store Data for Bar Plot ---
        mean_inp = np.nanmean(inp_val)
        mean_out = np.nanmean(out_val)
        bar_data.append((mean_inp, mean_out))

        # --- Plot Column 1: Average Belief Variance ---
        ax_avg.plot(xax, np.nanmean(belief_mu, axis=(0, 1)), c=c, lw=2, label=model_labels[i])
        if i == 0: # Add baselines only once
            ax_avg.plot(xax, np.nanmean(joint_mu, axis=(0, 1)), c=joint_color, ls="--", lw=1.8, label="joint")
            ax_avg.plot(xax, np.nanmean(naive_mu, axis=(0, 1)), c=naive_color, ls="--", lw=1.8, label="naive")
        # --- Plot Columns 2+: Distance Variances ---
        for d in range(D):
            lab_model = model_labels[i] if d == 0 else None
            lab_joint = "joint" if (i == 0 and d == 0) else None
            lab_naive = "naive" if (i == 0 and d == 0) else None

            ax_dists[d].plot(xax, np.nanmean(belief[..., d], axis=(0, 1)), c=c, lw=2.6, label=lab_model, zorder=3)
            ax_dists[d].plot(xax, np.nanmean(joint[..., d], axis=(0, 1)), c=joint_color, ls="--", lw=1.6, label=lab_joint, zorder=2)
            ax_dists[d].plot(xax, np.nanmean(naive[..., d], axis=(0, 1)), c=naive_color, ls="--", lw=1.6, label=lab_naive, zorder=1)
    # --- Draw Bar Plot (Column 0) ---
    # Structure of bar_data: [(tr_in, tr_out), (echo_in, echo_out)]
    bar_vals = [bar_data[0][0], bar_data[0][1], bar_data[1][0], bar_data[1][1]]
    bar_cols = [model_colors[0], model_colors[0], model_colors[1], model_colors[1]]
    bar_x = [0, 1, 2, 3]
    bar_lbls = ["Tr\nIn", "Tr\nOut", "Echo\nIn", "Echo\nOut"]
    
    ax_bar.bar(bar_x, bar_vals, color=bar_cols, alpha=0.8, edgecolor='k')
    ax_bar.set_xticks(bar_x)
    ax_bar.set_xticklabels(bar_lbls)
    ax_bar.set_title("Input/Output\nMean Variance")
    ax_bar.set_ylabel("variance")
    for ax in axes[1:]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("t")
    # Specific Titles
    ax_avg.set_title(r"$B_{r}$ Variance (averaged over $r$)")
    # Share Y axis for the distance plots (columns 2+) 
    first_dist_ax = ax_dists[0]
    for d in range(D):
        ax_dists[d].set_title(f"Distance = {d}")
        if d > 0:
            ax_dists[d].sharey(first_dist_ax)
            plt.setp(ax_dists[d].get_yticklabels(), visible=False)
    h, l = ax_dists[0].get_legend_handles_labels()
    fig.legend(tuple(h), tuple(l), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.suptitle(r"WITHIN-GROUP variance (between batches)", y=1.28)
    plt.show()


def logit_dynamics(trained, echo):
    R, xax, eps = trained.realization_num, trained.step_range + 1, 1e-9
    # Map model names to objects and attribute prefixes
    cfg = {"Trained": (trained, "model"), "Echo": (echo, "model"), 
        "Joint": (trained, "joint"), "Naive": (trained, "naive")}
    colors = {"Trained": "C0", "Echo": "C1", "Joint": "C2", "Naive": "C3"}
    data = {}

    # 1. Extract Data
    for k, (m, attr) in cfg.items():
        vals = np.zeros((R, R, m.step_num))
        bel = getattr(m, f"{attr}_goal_belief")
        for r1 in m.realization_range:
            for r2 in m.realization_range:
                mask = (m.goal_ind == 0) & (m.ctx_vals[:,0]==r1) & (m.ctx_vals[:,1]==r2)
                if mask.sum() > 1:
                    p = np.clip(bel[mask, :, r1], eps, 1-eps)
                    vals[r1, r2] = np.log(p/(1-p)).mean(0)
                else: vals[r1, r2] = np.nan
        data[k] = np.nanmean(vals, axis=(0,1))

    # 2. Fit Theory (Naive Model)
    func = lambda t, a, b, c: a*np.log(t) - b*t + c
    try: popt, _ = curve_fit(func, xax, data["Naive"], p0=[1, 0.05, -2], maxfev=1e4)
    except: popt = [0.5, 0.05, -2]
    a, b = popt[:2]
    log_v, lin_v = a * np.log(xax), b * (xax - 1)
    
    # 3. Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # Panel 1: Performance
    for k, v in data.items(): 
        ax1.plot(xax, v, c=colors[k], lw=3, alpha=0.8, label=k)
    ax1.set(title="Performance", xlabel="t", ylabel="Logit True Positive")
    # Panel 2: Envelopes
    # Note: Added r'' to labels to fix SyntaxWarning with \pm
    ax2.fill_between(xax, -log_v, log_v, color='g', alpha=0.15, label=r'Mutual Info envelope: $\pm \ln t$'+'\nParameter estimation')
    ax2.plot(xax, log_v, 'g:', lw=1); ax2.plot(xax, -log_v, 'g:', lw=1)
    
    ax2.fill_between(xax, -lin_v, lin_v, color='r', alpha=0.15, label=r'Total Correlation envelope: $\pm t$'+'\nInteraction between obs')
    ax2.plot(xax, lin_v, 'r--', lw=1); ax2.plot(xax, -lin_v, 'r--', lw=1)
    ax2.set(title="Contributions to inference scale differently", xlabel="t", ylabel="Relative Magnitude")
    for ax in (ax1, ax2): ax.grid(alpha=0.25); ax.legend(loc=2, frameon=False)
    plt.tight_layout(); plt.show()





if not training:
    make_panel_figure(trained, echo,
        layout=(("ACC","DIFF","CORR"),  ("GRAD","PR","DIST"),  ("PROUT","PRIN","PRFMS")))
    plt.show()

    plot_combined_variances(trained, echo)

    logit_dynamics(trained, echo)