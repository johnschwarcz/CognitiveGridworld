import numpy as np; import torch; import os; import sys; import inspect
import matplotlib.pyplot as plt

path = inspect.getfile(inspect.currentframe())
path = os.path.dirname(os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

AGENT_COLORS = {"Trained": "C0", "Echo": "C1", "Joint": "C2", "Naive": "C3"}
V_ALPHA = 0.5
NET_Z0, NET_Z1 = 2, 3
BAYES_Z0, BAYES_Z1 = 5, 6

# ═══════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════

def pr(evals, eps=1e-12):
    e = np.asarray(evals, float)
    return (e.sum(-1) ** 2) / ((e * e).sum(-1) + eps)



def vne(evals, eps=1e-12, norm=True):
    e = np.maximum(np.asarray(evals, float), 0.0)
    p = e / (e.sum(-1, keepdims=True) + eps)
    H = -(p * np.log(p + eps)).sum(-1)
    if norm:
        H = H / (np.log(float(e.shape[-1])) + eps)
    return H

def _smooth(y, w):
    y = np.asarray(y, float)
    if w <= 1:
        return y
    k = np.ones(int(w), float) / float(w)
    p = int(w) // 2
    return np.convolve(np.pad(y, (p, p), mode="edge"), k, mode="valid")

def _ep_xy(y, episode_lim=None):
    y = np.asarray(y, float)
    n = y.size
    if n == 0:
        return np.zeros(0, int), y

    if episode_lim is None:
        idx = np.arange(n)
    elif np.isscalar(episode_lim):
        m = int(episode_lim)
        if m < 1:
            m = 1
        if m > n:
            m = n
        idx = np.arange(m)
    elif isinstance(episode_lim, slice):
        idx = np.arange(n)[episode_lim]
    else:
        t = tuple(episode_lim)
        if len(t) == 2:
            start, stop = t
            step = 1
        elif len(t) == 3:
            start, stop, step = t
        else:
            start, stop, step = 1, n, 1
        if start is None:
            start = 1
        if stop is None:
            stop = n
        if step is None:
            step = 1
        s = int(start) - 1
        e = int(stop)
        st = int(step)
        if s < 0:
            s = 0
        if e > n:
            e = n
        if st < 1:
            st = 1
        if e <= s:
            e = min(n, s + 1)
        idx = np.arange(s, e, st)

    if idx.size == 0:
        idx = np.arange(min(1, n))
    return idx + 1, y[idx]

def _plot_tr_echo(a, x0, y0, x1, y1, title, zline=False):
    a.plot(x0, y0, c=AGENT_COLORS["Trained"], lw=2.5, label="Trained")
    a.plot(x1, y1, c=AGENT_COLORS["Echo"], lw=2.5, label="Echo")
    if zline:
        a.axhline(0, c="k", ls="--")
    a.set_title(title)
    a.legend(frameon=False, fontsize=9)
# ═══════════════════════════════════════════════════════════════════
# Core figure (2×5 grid — training-epoch diagnostics only)
# ═══════════════════════════════════════════════════════════════════

def make_core_figure(
    trained,
    echo,
    smooth_w=100,
    lim=None,
    figsize=(24, 9),
    episode_lim=None,
    layout=(
        ("ACC", "DIFF", "CORR", "GRADD", "ENTD"),
        ("ENTIO", "PR", "PRIO", "GRADO", "GRADI"),
    ),
):
    def ex(y):
        return _ep_xy(_smooth(y, smooth_w), episode_lim)

    fig, ax = plt.subplots(len(layout), len(layout[0]), figsize=figsize, constrained_layout=False)

    def p_ACC(a):
        x0, y0 = ex(trained.test_acc_through_training[:, -1])
        x1, y1 = ex(echo.test_acc_through_training[:, -1])
        _plot_tr_echo(a, x0, y0, x1, y1, "ACC (final step)")
        if hasattr(trained, "naive_acc"):
            a.axhline(np.asarray(trained.naive_acc)[:, -1].mean(), c=AGENT_COLORS["Naive"], ls="--", label="naive mean", zorder=BAYES_Z0)
        if hasattr(trained, "joint_acc"):
            a.axhline(np.asarray(trained.joint_acc)[:, -1].mean(), c=AGENT_COLORS["Joint"], ls="--", label="joint mean", zorder=BAYES_Z1)
        a.set_ylim(0, 0.8)
        a.legend(frameon=False, fontsize=9)

    def p_DIFF(a):
        x0, y0 = ex((trained.test_net_naive_DKL_through_training - trained.test_net_joint_DKL_through_training)[:, -1])
        x1, y1 = ex((echo.test_net_naive_DKL_through_training - echo.test_net_joint_DKL_through_training)[:, -1])
        _plot_tr_echo(a, x0, y0, x1, y1, "DKL(naive) - DKL(joint) (final step)", zline=True)

    def p_CORR(a):
        x0, y0 = ex(trained.test_SII_coef_through_training)
        x1, y1 = ex(echo.test_SII_coef_through_training)
        _plot_tr_echo(a, x0, y0, x1, y1, "CORR (SII coef)")

    def p_GRADD(a):
        x0, y0 = ex(trained.readout_grad_log_through_training - trained.readin_grad_log_through_training)
        x1, y1 = ex(echo.readout_grad_log_through_training - echo.readin_grad_log_through_training)
        _plot_tr_echo(a, x0, y0, x1, y1, "GRAD(|readout|) - GRAD(|readin|)", zline=True)

    def p_GRADO(a):
        x0, y0 = ex(trained.readout_grad_log_through_training)
        x1, y1 = ex(echo.readout_grad_log_through_training)
        _plot_tr_echo(a, x0, y0, x1, y1, "GRAD(|readout|)")

    def p_GRADI(a):
        x0, y0 = ex(trained.readin_grad_log_through_training)
        x1, y1 = ex(echo.readin_grad_log_through_training)
        _plot_tr_echo(a, x0, y0, x1, y1, "GRAD(|readin|)")

    def p_ENTD(a):
        x0, y0 = ex(vne(trained.test_model_update_dim_through_training) - vne(trained.test_model_input_dim_through_training))
        x1, y1 = ex(vne(echo.test_model_update_dim_through_training) - vne(echo.test_model_input_dim_through_training))
        a.plot(x0, y0, c=AGENT_COLORS["Trained"], lw=2.5, label="Trained", zorder=NET_Z0)
        a.plot(x1, y1, c=AGENT_COLORS["Echo"], lw=2.5, label="Echo", zorder=NET_Z1)
        a.axhline(0, c="k", ls="--")
        m = np.max(np.abs(np.concatenate((y0, y1))))
        if m > 0: a.set_ylim(-1.05 * m, 1.05 * m)
        a.set_title("VNE(out) - VNE(in)")
        a.legend(frameon=False, fontsize=9)

    def p_ENTIO(a):
        x0o, y0o = ex(vne(trained.test_model_update_dim_through_training))
        x0i, y0i = ex(vne(trained.test_model_input_dim_through_training))
        x1o, y1o = ex(vne(echo.test_model_update_dim_through_training))
        x1i, y1i = ex(vne(echo.test_model_input_dim_through_training))
        a.plot(x0o, y0o, c=AGENT_COLORS["Trained"], lw=2.5, alpha=1, label="Tr out", zorder=NET_Z0)
        a.plot(x0i, y0i, c=AGENT_COLORS["Trained"], lw=2.5, alpha=V_ALPHA, label="Tr in", zorder=NET_Z0)
        a.plot(x1o, y1o, c=AGENT_COLORS["Echo"], lw=2.5, alpha=1, label="Echo out", zorder=NET_Z1)
        a.plot(x1i, y1i, c=AGENT_COLORS["Echo"], lw=2.5, alpha=V_ALPHA, label="Echo in", zorder=NET_Z1)
        a.set_title("VNE(input/output)")
        a.legend(frameon=False, fontsize=9)

    def p_PR(a):
        x0, y0 = ex(pr(trained.test_model_update_dim_through_training) - (pr(trained.test_model_input_dim_through_training) + 1e-12))
        x1, y1 = ex(pr(echo.test_model_update_dim_through_training) - (pr(echo.test_model_input_dim_through_training) + 1e-12))
        a.plot(x0, y0, c=AGENT_COLORS["Trained"], lw=2.5, label="Trained", zorder=NET_Z0)
        a.plot(x1, y1, c=AGENT_COLORS["Echo"], lw=2.5, label="Echo", zorder=NET_Z1)
        a.axhline(0, c="k", ls="--")
        m = np.max(np.abs(np.concatenate((y0, y1))))
        if m > 0: a.set_ylim(-1.05 * m, 1.05 * m)
        a.set_title("PR(out) - PR(in)")
        a.legend(frameon=False, fontsize=9)

    def p_PRIO(a):
        x0o, y0o = ex(pr(trained.test_model_update_dim_through_training))
        x0i, y0i = ex(pr(trained.test_model_input_dim_through_training))
        x1o, y1o = ex(pr(echo.test_model_update_dim_through_training))
        x1i, y1i = ex(pr(echo.test_model_input_dim_through_training))
        a.plot(x0o, y0o, c=AGENT_COLORS["Trained"], lw=2.5, alpha=1, label="Tr out", zorder=NET_Z0)
        a.plot(x0i, y0i, c=AGENT_COLORS["Trained"], lw=2.5, alpha=V_ALPHA, label="Tr in", zorder=NET_Z0)
        a.plot(x1o, y1o, c=AGENT_COLORS["Echo"], lw=2.5, alpha=1, label="Echo out", zorder=NET_Z1)
        a.plot(x1i, y1i, c=AGENT_COLORS["Echo"], lw=2.5, alpha=V_ALPHA, label="Echo in", zorder=NET_Z1)
        a.set_title("PR(input/output)")
        a.legend(frameon=False, fontsize=9)

    panels = {
        "ACC": p_ACC, "DIFF": p_DIFF, "CORR": p_CORR, "GRADD": p_GRADD, "GRADO": p_GRADO, "GRADI": p_GRADI,
        "ENTD": p_ENTD, "ENTIO": p_ENTIO, "PR": p_PR, "PRIO": p_PRIO,
    }
    nr, nc = len(layout), len(layout[0])
    for r in range(nr):
        for c in range(nc):
            key = layout[r][c]
            a = ax[r, c]
            if key == "NONE" or key not in panels:
                a.axis("off")
            else:
                panels[key](a)
                if lim is not None: a.set_xlim(1, lim)

    fig.suptitle("Training diagnostics (2x5)", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=1.0, w_pad=0.8)
    return fig, ax

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cuda = 0
    realization_num = 10
    step_num = 30
    hid_dim = 1000
    state_num = 500
    obs_num = 5
    training = False
    batch_num = 8000
    episodes = 50000 if training else 1

    echo = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5,
        'realization_num': realization_num, 'hid_dim': hid_dim, 'obs_num': obs_num, 'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 2,
        'training': training,
        'save_env': "/sanity/reservoir_ctx_2_e5" if training else None,
        'load_env': None if training else "/sanity/reservoir_ctx_2_e5"})

    trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5,
        'realization_num': realization_num, 'hid_dim': hid_dim, 'obs_num': obs_num, 'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .001, 'ctx_num': 2,
        'training': training,
        'save_env': "/sanity/fully_trained_ctx_2_e5" if training else None,
        'load_env': None if training else "/sanity/fully_trained_ctx_2_e5"})

    fig, ax = make_core_figure(trained, echo, smooth_w=100)
    plt.show()
