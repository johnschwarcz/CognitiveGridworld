import os, sys, glob, re, inspect
import numpy as np
import torch
import matplotlib.pyplot as plt

def _repo_root():
    """Walk up until main/DATA appears. inspect.getfile fails in an interactive window,
    where it returns '<stdin>' and the repo is then resolved against the cwd instead."""
    try:
        start = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    except Exception:
        start = os.getcwd()
    d = start if os.path.isdir(start) else os.getcwd()
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "main", "DATA")):
            return d
        d, prev = os.path.dirname(d), d
        if d == prev:
            break
    raise RuntimeError(f"could not locate the repo root (main/DATA) from {start}")


root = _repo_root()
sys.path.insert(0, root)
from main.utils import fig_path, apply_fig_style

DATA = os.path.join(root, "main", "DATA", "oracle_pilot")
BOUNDS_CACHE = os.path.join(root, "bayes_bounds_ctx2_T30.npy")
SHOW = [("default", "#1a6fae", "Generator"), ("clf_reg", "#d7191c", "Classifier")]


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


def load(cond):
    A = []
    for f in sorted(glob.glob(os.path.join(DATA, f"{cond}_rep*_net.pth")),
                    key=lambda p: int(re.search(r"rep(\d+)", p).group(1))):
        d = torch.load(f, weights_only=False, map_location="cpu")
        A.append(np.asarray(d["test_acc_through_training"], float)[:, -1])
    if not A:
        raise FileNotFoundError(
            f"no {cond}_rep*_net.pth under {DATA}")
    n = min(len(a) for a in A)
    return np.stack([a[:n] for a in A])


fs = apply_fig_style()
joint, naive = get_bounds()
fig, ax = plt.subplots(figsize=(4, 3.4))

ax.axhspan(naive, joint, color="0.5", alpha=.07, lw=0, zorder=1)
for y, t in ((joint, "Joint"), (naive, "Naive")):
    ax.axhline(y, color="0.25", ls=(0, (5, 2)), lw=1.1, zorder=2)
    ax.text(.6, y + .008, t, fontsize=fs["annot"], color="0.35")

for cond, c, lab in SHOW:
    A = load(cond)
    x = np.arange(A.shape[1])
    m, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(A.shape[0])
    ax.fill_between(x, m - se, m + se, color=c, alpha=.30, lw=0, zorder=3)
    ax.plot(x, m, color=c, lw=1, zorder=4, label=lab, ls = ':')
    print(f"{lab:<12} n={A.shape[0]}  final {m[-1]:.4f} +/- {se[-1]:.4f}")

ax.set(xlabel="Testing episode", ylabel="Testing Accuracy", xlim=(0, 40), ylim=(.10, .81))
ax.set_title("RCC Generator Ablation", pad=10, fontsize=fs["title"])
ax.grid(alpha=.22, lw=.6)
leg = ax.legend(loc="lower right", frameon=False, fontsize=fs["legend"],
                title=r"$\hat{\mathcal{E}}$ recieves"+"\ngradients from", title_fontsize=fs["legend"] - .5,
                labelspacing=.45, handlelength=1.8)
leg._legend_box.align = "left"

fig.tight_layout()
plt.savefig(fig_path("embedding_gradient_source.svg"), dpi=300, bbox_inches="tight")
print("saved embedding_gradient_source.svg")
plt.show()
