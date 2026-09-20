import os, sys, inspect
import numpy as np
import torch
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
root = os.path.abspath(os.path.join(path, '..', '..'))
sys.path.insert(0, root)

plt.rcParams.update({'font.family': 'serif', 'font.size': 13, 'axes.labelsize': 13,
                     'axes.titlesize': 15, 'legend.fontsize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})

CHANCE = 0.1
RUNS = [("Ablation (no Generator)", "main/DATA/RL_ablation_net.pth", "C0"),
        ("RCC",                     "main/DATA/RL_2_net.pth",        "C3")]
BINS = [.15, .20, .25, .35, .45, .55, .65, .70, .75]
BOUNDS_CACHE = os.path.join(root, "bayes_bounds_ctx2_T30.npy")


def get_bounds(batch_num=8000, step_num=30, ctx_num=2):
    """Final-step Joint and Naive accuracy. Cached: the Bayes run takes a minute."""
    if os.path.exists(BOUNDS_CACHE):
        return tuple(np.load(BOUNDS_CACHE))
    from main.CognitiveGridworld import CognitiveGridworld
    b = CognitiveGridworld(mode=None, show_plots=False, episodes=1, ctx_num=ctx_num,
                           obs_num=5, realization_num=10, state_num=500,
                           batch_num=batch_num, step_num=step_num)
    out = np.array([b.joint_acc.mean(0)[-1], b.naive_acc.mean(0)[-1]])
    np.save(BOUNDS_CACHE, out)
    return tuple(out)


def load(p):
    d = torch.load(p, weights_only=False, map_location="cpu")
    # index 0 is a test episode, so train_accs[0] is never written
    return d["test_acc_through_training"][1:, -1], d["train_acc_through_training"][1:, -1]


joint, naive = get_bounds()
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), tight_layout=True)

for lab, p, c in RUNS:
    te, tr = load(p)
    ep = np.arange(1, len(te) + 1) * 500
    gap = tr - te
    ratio = (te - CHANCE) / (tr - CHANCE)
    ratio[(tr - CHANCE) <= 0.02] = np.nan        # meaningless while train is still at chance

    ax[0].plot(ep, tr, c=c, lw=1.5, ls='--', alpha=.75)
    ax[0].plot(ep, te, c=c, lw=2, label=lab)
    ax[1].plot(ep, ratio, c=c, lw=2, label=lab)

    # binned by train accuracy: controls for learning rate and episode budget
    x, y = [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (tr >= lo) & (tr < hi)
        if m.sum():
            x.append((lo + hi) / 2)
            y.append(((te[m] - CHANCE) / (tr[m] - CHANCE)).mean())
    ax[2].plot(x, y, c=c, lw=2, marker='o', ms=7, mec='k', label=lab)

    print(f"{lab:<26} test {te[-1]:.4f} | gap {gap[-1]:+.4f} | ratio {ratio[-1]:.3f} "
          f"| min ratio {np.nanmin(ratio):.3f} at ep {ep[np.nanargmin(ratio)]}")

for b, t in ((joint, "Joint"), (naive, "Naive")):
    ax[0].axhline(b, c='k', lw=1.5, ls=':')
    ax[0].text(0, b + .012, f"{t} {b:.3f}", fontsize=10, color='k')
ax[0].plot([], [], c='gray', lw=2, label='held-out test')
ax[0].plot([], [], c='gray', lw=1.5, ls='--', label='train')
ax[0].set(xlabel="training episodes", ylabel="accuracy at step 30", ylim=(0.05, 0.85),
          title="Accuracy through training")

for a in (ax[1], ax[2]):
    a.axhline(1.0, c='k', lw=1, ls=':')
    a.set(ylabel="(test $-$ chance) / (train $-$ chance)", ylim=(0, 1.15))
ax[1].set(xlabel="training episodes", title="Transfer through training")
ax[2].set(xlabel="training-set accuracy", title="Transfer at matched performance")

for a in ax:
    a.grid(alpha=.3)
    a.legend(loc='lower right')

plt.savefig("ablation_transfer_ratio.svg", dpi=300, bbox_inches="tight")
print("saved ablation_transfer_ratio.svg")
plt.show()
