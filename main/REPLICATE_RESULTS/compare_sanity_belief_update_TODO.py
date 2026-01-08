import matplotlib.patches as mpatches
import numpy as np
import torch
import pylab as plt
import numpy as np; import torch; import pylab as plt; import os; import sys; import inspect
path = inspect.getfile(inspect.currentframe()); path = os.path.dirname(os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + "/main"); sys.path.insert(0, path + "/main/model")
from main.CognitiveGridworld import CognitiveGridworld

def make_env(ctx_num, load_env, reservoir, mode="SANITY", cuda=0, batch=5000):
    return CognitiveGridworld(**{
        "episodes": 1, "state_num": 500, "batch_num": batch, "step_num": 30, "obs_num": 5,
        "KQ_dim": 30, "realization_num": 10, "likelihood_temp": 2, "checkpoint_every": 500,
        "showtime": .1, "show_plots": False, "mode": mode, "hid_dim": 1000,
        "classifier_LR": .0005, "controller_LR": .005, "generator_LR": .001,
        "learn_embeddings": False, "reservoir": reservoir, "training": False, "save_env": None,
        "ctx_num": ctx_num, "load_env": load_env, "cuda": cuda
    })

def sym_dkl_pair(P, Q, eps=1e-4):
    P = np.clip(P, eps, 1.0 - eps); Q = np.clip(Q, eps, 1.0 - eps)
    P = P / P.sum(-1, keepdims=True); Q = Q / Q.sum(-1, keepdims=True)
    dPQ = np.sum(P * (np.log(P) - np.log(Q)), axis=-1)
    dQP = np.sum(Q * (np.log(Q) - np.log(P)), axis=-1)
    return 0.5 * (dPQ + dQP)  # (B,)

cuda = 0
mode = "SANITY"
ctxs = (1, 2)
models = ("joint", "fully trained", "naive", "reservoir")
metrics = ("first→second (1 vs 2)", "first→last (1 vs T)")
colors = ("C0", "C1", "C2", "C3")
batch = 5000
means = np.empty((2, 2, 4), float)

for ci, ctx in enumerate(ctxs):
    env_ft = make_env(ctx, f"fully_trained_ctx_{ctx}", reservoir=False, mode=mode, cuda=cuda, batch = batch)
    JGB = env_ft.joint_goal_belief; NGB = env_ft.naive_goal_belief; FT = env_ft.model_goal_belief
    env_rs = make_env(ctx, f"reservoir_ctx_{ctx}", reservoir=True, mode=mode, cuda=cuda, batch = batch)
    RS = env_rs.model_goal_belief

    P = (JGB, FT, NGB, RS)
    for k in range(4):
        Pk = P[k]
        means[ci, 0, k] = sym_dkl_pair(Pk[:, 1, :], Pk[:, 2, :]).mean()
        means[ci, 1, k] = sym_dkl_pair(Pk[:, 1, :], Pk[:, -1, :]).mean()


x = np.arange(4) * 4
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for ci, ctx in enumerate(ctxs):
    off = ctx
    a = 1.0 if ctx == 1 else 0.5
    for mi in range(2):
        ax[mi].bar(x + off, means[ci, mi], color=colors, alpha=a, linewidth = 1, ec = 'k')

for mi in range(2):
    ax[mi].set_xticks(x + 1.5); ax[mi].set_xticklabels(models, rotation=15, ha="center")
    ax[mi].set_title(metrics[mi])
    ax[mi].grid(axis="y", alpha=0.25); ax[mi].set_axisbelow(True)
    ax[mi].spines["top"].set_visible(False); ax[mi].spines["right"].set_visible(False)
ax[0].set_ylim([.07, .2])
ax[1].set_ylim([1.8, 2.3])
ax[0].set_ylabel("symmetric DKL")

handles = (mpatches.Patch(color="k", alpha=1.0, label="ctx = 1"),
           mpatches.Patch(color="k", alpha=0.5, label="ctx = 2"))
ax[0].legend(handles=handles, frameon=False, loc="upper right")
fig.suptitle("Belief dynamics")
plt.tight_layout(); plt.show()