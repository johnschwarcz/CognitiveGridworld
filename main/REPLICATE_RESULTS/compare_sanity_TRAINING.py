import numpy as np; import torch; import os; import sys; import inspect; import pylab as plt
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
    batch_num = 8000 if training else 50

    echo = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 50000, 'checkpoint_every': 5, 
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 2, 
        'training': training,
        'save_env': "/sanity/reservoir_ctx_2_e5" if training else None,
        'load_env': None if training else "/sanity/reservoir_ctx_2_e5"})
    if training:
        del(echo)

    trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 50000, 'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False, 
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .001, 'ctx_num': 2, 
        'training': training,
        'save_env': "/sanity/fully_trained_ctx_2_e5" if training else None,
        'load_env': None if training else "/sanity/fully_trained_ctx_2_e5"})
    if training:
        del(trained)

    echo_1 = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 50000, 'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 1,
        'training': training,
        'save_env': "/sanity/reservoir_ctx_1_e5" if training else None,
        'load_env': None if training else "/sanity/reservoir_ctx_1_e5"})
    if training:
        del(echo_1)

    trained_1 = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 50000, 'checkpoint_every': 5,
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

def make_panel_figure(trained, echo, layout=(("ACC","CORR","DIFF"),("GRAD","PR","DIST")),
                      smooth_w=100, figsize=(12, 6), lim=None):
    def smooth(y, w=smooth_w):
        y = np.asarray(y, float)
        if w <= 1: return y
        k = np.ones(w, float) / w
        p = w // 2
        yp = np.pad(y, (p, p), mode="edge")
        return np.convolve(yp, k, mode="valid")

    fig, ax = plt.subplots(2, 3, figsize=figsize, tight_layout=True)

    def _panel_ACC(a):
        a.set_title("ACC (final step)")
        y0 = smooth(trained.test_acc_through_training[:, -1]); x0 = np.arange(y0.size) + 1
        y1 = smooth(echo.test_acc_through_training[:, -1]);   x1 = np.arange(y1.size) + 1
        a.plot(x0, y0, c="C0", lw=3)
        a.plot(x1, y1, c="C1", lw=3)
        if hasattr(trained, "naive_acc"): a.axhline(np.asarray(trained.naive_acc)[:, -1].mean(), c="r", ls="--")
        if hasattr(trained, "joint_acc"): a.axhline(np.asarray(trained.joint_acc)[:, -1].mean(), c="g", ls="--")
        a.set_ylim(0, .8)
        a.legend(("Trained", "Echo", "naive mean", "joint mean") if (hasattr(trained,"naive_acc") and hasattr(trained,"joint_acc"))
                 else ("Trained","Echo"), frameon=False)

    def _panel_CORR(a):
        a.set_title("CORR (SII coef)")
        y0 = smooth(trained.test_SII_coef_through_training); x0 = np.arange(y0.size) + 1
        y1 = smooth(echo.test_SII_coef_through_training);   x1 = np.arange(y1.size) + 1
        a.plot(x0, y0, c="C0", lw=3)
        a.plot(x1, y1, c="C1", lw=3)
        a.legend(("Trained","Echo"), frameon=False)

    def _panel_DIFF(a):
        a.set_title("DKL DIFF (naive - joint, final step)")
        y0 = smooth((trained.test_net_naive_DKL_through_training - trained.test_net_joint_DKL_through_training)[:, -1])
        y1 = smooth((echo.test_net_naive_DKL_through_training    - echo.test_net_joint_DKL_through_training)[:, -1])
        x0 = np.arange(y0.size) + 1; x1 = np.arange(y1.size) + 1
        a.plot(x0, y0, c="C0", lw=3)
        a.plot(x1, y1, c="C1", lw=3)
        a.axhline(0, c="k", ls="--")
        a.legend(("Trained","Echo"), frameon=False)

    def _panel_GRAD(a):
        a.set_title("GRAD (|readout| - |readin|)")
        y0 = smooth(trained.readout_grad_log_through_training) - smooth(trained.readin_grad_log_through_training)
        y1 = smooth(echo.readout_grad_log_through_training)    - smooth(echo.readin_grad_log_through_training)
        x0 = np.arange(y0.size) + 1; x1 = np.arange(y1.size) + 1
        a.plot(x0, y0, c="C0", lw=3)
        a.plot(x1, y1, c="C1", lw=3)
        a.axhline(0, c="k", ls="--")
        a.legend(("Trained","Echo"), frameon=False)

    def _panel_PR(a):
        a.set_title("PR (output - input)")
        y0 = smooth(pr(trained.test_model_update_dim_through_training) - (pr(trained.test_model_input_dim_through_training) + 1e-12))
        y1 = smooth(pr(echo.test_model_update_dim_through_training)    - (pr(echo.test_model_input_dim_through_training) + 1e-12))
        x0 = np.arange(y0.size) + 1; x1 = np.arange(y1.size) + 1
        a.plot(x0, y0, c="C0", lw=3)
        a.plot(x1, y1, c="C1", lw=3)
        a.axhline(0, c="k", ls="--")
        m = np.max(np.abs(np.concatenate((y0, y1))))
        if m > 0: a.set_ylim(-1.05*m, 1.05*m)
        a.legend(("Trained","Echo"), frameon=False)

    def _panel_DIST(a):
        a.set_title("DKL DIST (sym DKL from step 1)")
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
        a.plot(x, means[0], "-o", ms=3)
        a.plot(x, means[1], "-o", ms=3)
        a.plot(x, means[2], "-o", ms=3)
        a.plot(x, means[3], "-o", ms=3)
        a.set_xscale("log"); a.set_yscale("log")
        a.legend(("joint","FT","naive","echo"), frameon=False)

    panels = {
        "ACC": _panel_ACC,
        "CORR": _panel_CORR,
        "DIFF": _panel_DIFF,
        "GRAD": _panel_GRAD,
        "PR": _panel_PR,
        "DIST": _panel_DIST,
    }

    if isinstance(layout[0], (tuple,)):  # 2x3
        order = (layout[0][0], layout[0][1], layout[0][2], layout[1][0], layout[1][1], layout[1][2])
    else:  # flat len-6 tuple
        order = layout

    for i in range(6):
        k = order[i]
        r = i // 3
        c = i - 3*r
        panels[k](ax[r, c])
        if lim is not None: ax[r, c].set_xlim(1, lim)
    return fig, ax

if not training:
    make_panel_figure(trained, echo, layout=(("ACC","DIFF", "CORR"),("GRAD", "PR", "DIST")))
    plt.show()