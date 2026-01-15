import numpy as np; import torch; import os; import sys; import inspect; import pylab as plt
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld 

def get_DKLS(model):
    MJ = model.DKL(model.model_goal_belief, model.joint_goal_belief, sym = True)    
    NN = model.DKL(model.model_goal_belief, model.naive_goal_belief, sym = True)
    NJ = model.DKL(model.joint_goal_belief, model.naive_goal_belief, sym = True)
    return MJ, NN, NJ

if __name__ == "__main__":
    cuda = 0
    realization_num = 10
    step_num = 30
    hid_dim = 1000
    state_num = 500
    obs_num = 5

    #############################################################################################################
    """
    Collect large batch from trained networks   
    """

    batch_num = 8000

    trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1, 'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False,'ctx_num': 2, 'load_env': "/sanity/fully_trained_ctx_2"})

    echo = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1, 'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True,  'ctx_num': 2, 'load_env': "/sanity/reservoir_ctx_2"})


    #############################################################################################################
    """
    Plot Relative DKLs    
    """
    JT_sym, TN_sym, JN_sym = get_DKLS(trained)
    JR_sym, NR_sym, _ = get_DKLS(echo)

    # Data setup: (x_data, y_data, color, label)
    networks = [
        (JT_sym, TN_sym, "#B9E572", 'Fully Trained'),
        (JR_sym, NR_sym, "#979994", 'Echo State')]

    fig, ax = plt.subplots(figsize=(4, 4))
    idx = np.linspace(0, len(JT_sym)-1, 4, dtype=int)
    style = dict(marker='h', s=75, zorder=10)

    for x, y, color, label in networks:
        ax.plot(x, y, c=color, zorder=5, lw=2, linestyle='dotted')
        ax.scatter(x[idx], y[idx], c=color, edgecolors='k', lw = 1.5, label=label, **style)
        ax.scatter(x[-1], y[-1], c=color, edgecolors='r', linewidths=2, **style)
    if 'JN_sym' in locals():
        ax.axhline(JN_sym[-1], c='r', ls='--', lw=1, zorder=0)
        ax.axvline(JN_sym[-1], c='r', ls='--', lw=1, zorder=0)
    ax.set(xlabel=r"$\mathcal{D}_{KL}(\cdot \ || \ \text{Joint})$", 
        ylabel=r"$\mathcal{D}_{KL}(\cdot \ || \ \text{Naive})$",
        title="Belief-state Divergence")
    ax.set_xlim([-.5, JN_sym[-1] + .1])
    ax.set_ylim([-.5, JN_sym[-1] + .1])

    ax.legend(loc='upper right')
    ax.annotate('start', (networks[0][0][0], networks[0][1][0]), xytext=(-15, -20), 
                textcoords='offset points', arrowprops=dict(arrowstyle='->', color='gray'))
    plt.show()

    #############################################################################################################
    """
    Compare training with small batch
    """

    batch_num = 50

    echo = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1, 'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 2, 'load_env': "/sanity/reservoir_ctx_2_e5"})

    trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1, 'checkpoint_every': 5,   
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False, 
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .001, 'ctx_num': 2, 'load_env': "/sanity/fully_trained_ctx_2_e5"})

    echo_1 = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1,    'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 1, 'load_env': "/sanity/reservoir_ctx_1_e5"})

    trained_1 = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1, 'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num, 'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .001, 'ctx_num': 1, 'load_env': "/sanity/fully_trained_ctx_1_e5"})
    

    #############################################################################################################
    """ 
    plot training comparisons
    """

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

    make_panel_figure(trained, echo, layout=(("ACC","DIFF", "CORR"),("GRAD", "PR", "DIST")))
    plt.show()