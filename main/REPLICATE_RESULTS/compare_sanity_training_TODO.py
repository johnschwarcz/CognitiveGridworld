import numpy as np; import torch; import os; import sys; import inspect; import pylab as plt
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld 

if __name__ == "__main__":
    cuda = 1
    realization_num = 10
    batch_num = 8000
    step_num = 30
    hid_dim = 1000
    state_num = 500
    obs_num = 5

    echo = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 2, 'load_env': "reservoir_ctx_2__"})

    trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False, 
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .001, 'ctx_num': 2, 'load_env': "fully_trained_ctx_2__"})

    echo_1 = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 1, 'load_env': "reservoir_ctx_1__"})

    trained_1 = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num, 'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .001, 'ctx_num': 1, 'load_env': "fully_trained_ctx_1__"})

    #############################################

    fig, ax = plt.subplots(2, 2, figsize = (7, 5), tight_layout = True) 
    ax[0,0].plot(trained.test_acc_through_training[:,-1])
    ax[0,0].plot(echo.test_acc_through_training[:,-1])
    ax[0,1].plot(trained.test_SII_coef_through_training)
    ax[0,1].plot(echo.test_SII_coef_through_training)
    ax[1,0].plot(trained.test_net_joint_DKL_through_training[:,-1], c= 'C0', lw = 4)
    ax[1,0].plot(echo.test_net_joint_DKL_through_training[:,-1], c = 'C1', lw = 4)
    ax[1,1].plot(trained.test_net_naive_DKL_through_training[:,-1], c = 'C0', lw = 4)
    ax[1,1].plot(echo.test_net_naive_DKL_through_training[:,-1], c = 'C1', lw = 4)
    plt.show()

    #############################################

    ms = 5
    mew = .5
    rm_PC1 = False


    fig, ax = plt.subplots(2,2, figsize = (10, 6), tight_layout = True)
    xax = np.arange(trained.hid_dim)
    if rm_PC1:
        xax = xax[:-1]
    xax = xax + 1


    for i in range(2):

        var = trained.test_model_dim_through_training
        if rm_PC1:
            var = var[:, 1:]
        ax[i, 0].plot(xax, var[0], '-o', c = 'k', ms = ms/2, alpha = .5, label = "init")
        ax[i, 0].plot(xax, var[-1], '-o', c = 'g', ms = ms,  mec = 'g', mew = mew, label = "trained")

        var = echo.test_model_dim_through_training
        if rm_PC1:
            var = var[:, 1:]
        ax[i, 0].plot(xax, var[-1], '-o', c = 'r', alpha = 1,  ms = ms, mec = 'r', mew = mew, label = "random")
        ax[i, 0].legend()
        
        var = trained.test_model_dim_ratio_through_training
        if rm_PC1:
            var = var[:, 1:]
            var = var/var.sum(-1, keepdims = True)
        ax[i, 1].plot(xax, var[0], '-o', c = 'k', ms = ms/2, alpha = .5, label = "init")
        ax[i, 1].plot(xax, var[-1], '-o', c = 'g', ms = ms,  mec = 'g', mew = .5, label = "trained")
        
        var = echo.test_model_dim_ratio_through_training
        if rm_PC1:
            var = var[:, 1:]
            var = var/var.sum(-1, keepdims = True)
        ax[i, 1].plot(xax, var[-1], '-o', c = 'r', alpha = 1,  ms = ms, mec = 'r', mew = mew, label = "random")

        if i == 1:
            ax[i, 1].set_yscale('log')
            ax[i, 0].set_yscale('log')
        ax[i, 0].set_ylabel("var exp (log scale)" if i == 1 else "var exp")
        ax[i, 1].set_ylabel("var exp (log scale)" if i == 1 else "var exp")
        ax[i, 0].set_title("Abs")
        ax[i, 1].set_title("Ratio")
        ax[i, 1].set_xscale('log')
        ax[i, 0].set_xscale('log')
        ax[i, 1].legend()
    plt.show()

    #############################################

    def pr(evals):
        e = np.asarray(evals)
        s1 = e.sum(1)
        s2 = (e * e).sum(1)
        return (s1 * s1) / s2
    
    xax = np.arange(len(trained.test_model_dim_ratio_through_training))//5 + 1
    var = pr(trained.test_model_dim_through_training)
    plt.plot(xax[::5], var[::5], '-', c = 'C0'); 
    var = pr(echo.test_model_dim_through_training)
    plt.plot(xax[::5], var[::5], '-', c = 'C1')

    xax = np.arange(len(trained_1.test_model_dim_ratio_through_training))//5  + 1
    var = pr(trained_1.test_model_dim_through_training)
    plt.plot(xax[::5], var[::5], '-', c = 'C0', alpha = .5); 
    var = pr(echo_1.test_model_dim_through_training)
    plt.plot(xax[::5], var[::5], '-', c = 'C1' , alpha = .5)

    plt.yscale('log')
    plt.xscale('log')
    plt.show()


    #############################################

    fig, ax = plt.subplots(1, 5, figsize = (20, 3))

    xax = np.arange(len(trained.test_model_dim_ratio_through_training)) + 1
    var = pr(trained.test_model_dim_through_training)
    ax[0].plot(xax, var, '-', c = 'C0'); 
    var = pr(echo.test_model_dim_through_training)
    ax[0].plot(xax, var, '-', c = 'C1')
    xax = np.arange(len(trained_1.test_model_dim_ratio_through_training))  + 1
    var = pr(trained_1.test_model_dim_through_training)
    ax[0].plot(xax, var, '-', c = 'C0', alpha = .5); 
    var = pr(echo_1.test_model_dim_through_training)
    ax[0].plot(xax, var, '-', c = 'C1' , alpha = .5)

    xax = np.arange(len(trained.test_model_dim_ratio_through_training)) + 1
    var = (trained.test_model_dim_through_training).sum(-1)
    ax[1].plot(xax, var, '-', c = 'C0'); 
    var = (echo.test_model_dim_through_training).sum(-1)
    ax[1].plot(xax, var, '-', c = 'C1')
    xax = np.arange(len(trained_1.test_model_dim_ratio_through_training))  + 1
    var = (trained_1.test_model_dim_through_training).sum(-1)
    ax[1].plot(xax, var, '-', c = 'C0', alpha = .5); 
    var = (echo_1.test_model_dim_through_training).sum(-1)
    ax[1].plot(xax, var, '-', c = 'C1' , alpha = .5)


    xax = np.arange(len(trained.test_model_dim_ratio_through_training)) + 1
    var = pr(trained.test_model_dim_through_training)/(trained.test_model_dim_through_training).sum(-1)
    ax[2].plot(xax, var, '-', c = 'C0'); 
    var = pr(echo.test_model_dim_through_training)/(echo.test_model_dim_through_training).sum(-1)
    ax[2].plot(xax, var, '-', c = 'C1')
    xax = np.arange(len(trained_1.test_model_dim_ratio_through_training)) + 1
    var =  pr(trained_1.test_model_dim_through_training)/(trained_1.test_model_dim_through_training).sum(-1)
    ax[2].plot(xax, var, '-', c = 'C0', alpha = .5); 
    var = pr(echo_1.test_model_dim_through_training)/(echo_1.test_model_dim_through_training).sum(-1)
    ax[2].plot(xax, var, '-', c = 'C1' , alpha = .5)

    xax = np.arange(len(trained.test_model_dim_ratio_through_training)) + 1
    var = pr(trained.test_model_dim_through_training) + (trained.test_model_dim_through_training).sum(-1)
    ax[3].plot(xax, var, '-', c = 'C0'); 
    var = pr(echo.test_model_dim_through_training) + (echo.test_model_dim_through_training).sum(-1)
    ax[3].plot(xax, var, '-', c = 'C1')
    xax = np.arange(len(trained_1.test_model_dim_ratio_through_training)) + 1
    var =  pr(trained_1.test_model_dim_through_training) + (trained_1.test_model_dim_through_training).sum(-1)
    ax[3].plot(xax, var, '-', c = 'C0', alpha = .5); 
    var = pr(echo_1.test_model_dim_through_training) + (echo_1.test_model_dim_through_training).sum(-1)
    ax[3].plot(xax, var, '-', c = 'C1' , alpha = .5)

    xax = np.arange(len(trained.test_model_dim_ratio_through_training)) + 1
    ax[4].plot(xax, trained.test_SII_coef_through_training*1e1)
    ax[4].plot(xax, echo.test_SII_coef_through_training)

    lim = 200
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[0].set_xlim([0,lim])
    ax[1].set_xlim([0,lim])
    ax[3].set_xlim([0,lim])
    ax[4].set_xlim([0,lim])
    plt.show()


################################################

def plot_evr3d4(trained, trained_1, echo, echo_1, stride=5, max_pc=20, logz=True, elev=25, azim=-60, til = -1, w = 10, h = 5):
    def draw(ax, R, title):
        R = np.asarray(R)[::stride, :max_pc].T
        t = (np.arange(R.shape[1]) * stride + 1).astype(float)
        pc = (np.arange(R.shape[0]) + 1).astype(float)
        T, P = np.meshgrid(t, pc)
        ax.plot_surface(T, P, np.log10(R) if logz else R)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("PC")
        ax.set_zlabel("log10(EVR)" if logz else "EVR")

    fig = plt.figure(figsize=(w, h))
    ax1 = fig.add_subplot(221, projection="3d")
    ax2 = fig.add_subplot(222, projection="3d")
    ax3 = fig.add_subplot(223, projection="3d")
    ax4 = fig.add_subplot(224, projection="3d")
    draw(ax1, trained.test_model_dim_through_training[:til], "trained")
    draw(ax2, trained_1.test_model_dim_through_training[:til], "trained_1")
    draw(ax3, echo.test_model_dim_through_training[:til], "echo")
    draw(ax4, echo_1.test_model_dim_through_training[:til], "echo_1")
    plt.tight_layout()
    plt.show()

plot_evr3d4(trained, trained_1, echo, echo_1, stride = 1, max_pc = 999, elev = 30, azim = 30, til = 15)
