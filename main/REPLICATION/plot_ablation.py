import os, sys, glob, re, inspect
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

DATA = os.path.join(root, "main", "DATA")
BOUNDS_CACHE = os.path.join(DATA, "cache", "bayes_bounds_ctx2_T30.npy")
# (condition, colour, label, run dir). Both arms use the identical model -- Z-flow generator,
# standard classifier (mode == "oracle") -- so the only difference between them is which
# loss reaches the compositional embeddings.
SHOW = [("default", "#2c7bb6", "Generator",  "oracle_pilot"),
        ("clf_reg", "#d7191c", "Classifier", "oracle_pilot")]
FIGSIZE = (10, 3.6)
# --old DIR adds DIR/default_rep*_net.pth dashed (comparison Generator-arm runs, e.g. ones trained
# with a different classifier) and saves to a separate file, leaving the main figure alone.
OLD = sys.argv[sys.argv.index("--old") + 1] if "--old" in sys.argv else None


def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=4, width=.9)


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


def load(cond, run_dir):
    A = []
    for f in sorted(glob.glob(os.path.join(DATA, run_dir, f"{cond}_rep*_net.pth")),
                    key=lambda p: int(re.search(r"rep(\d+)", p).group(1))):
        d = torch.load(f, weights_only=False, map_location="cpu")
        A.append(np.asarray(d["test_acc_through_training"], float)[:, -1])
    if not A:
        raise FileNotFoundError(
            f"no {cond}_rep*_net.pth under {os.path.join(DATA, run_dir)}")
    n = min(len(a) for a in A)
    return np.stack([a[:n] for a in A])


fs = apply_fig_style()
joint, naive = get_bounds()
fig, ax = plt.subplots(figsize=FIGSIZE)

show = [(cond, c, lab, run_dir, "-") for cond, c, lab, run_dir in SHOW]
if OLD:
    show.insert(1, ("default", "#2c7bb6", "Generator, comparison runs", OLD, "--"))
for cond, c, lab, run_dir, ls in show:
    A = load(cond, run_dir)
    x = np.arange(1, A.shape[1] + 1)
    gap = joint - A
    m, se = gap.mean(0), gap.std(0, ddof=1) / np.sqrt(A.shape[0])
    ax.fill_between(x, m - se, m + se, color=c, alpha=.22, lw=0)
    ax.plot(x, m, color=c, lw=2, ls=ls, label=f"{lab} ($n{{=}}{A.shape[0]}$)")
    print(f"{lab:<12} n={A.shape[0]}  final excess {m[-1]:.4f} +/- {se[-1]:.4f}")

ax.axhline(joint - naive, c='k', lw=2, ls='--', zorder=10)
ax.text(.02, joint - naive, "Naive bound", color='k', fontsize=15, va='center',
        ha='left', transform=ax.get_yaxis_transform(), zorder=11,
        bbox=dict(fc='white', ec='none', alpha=.85, pad=1.5))

ax.set(xlabel="Testing Episode", ylabel="Optimal $-$ Network Accuracy",
       xscale="log", yscale="log", xlim=(1, 41))
ax.set_title("RCC Generator Ablation", fontsize=fs["title"])
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(mticker.LogLocator(subs=(1., 2., 3., 5.)))
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(mticker.NullFormatter())
ax.grid(alpha=.25, lw=.6, which="both")
_despine(ax)
leg = ax.legend(loc="lower left", frameon=False, fontsize=15,
                title=r"$\hat{\mathcal{E}}$ receives"+"\ngradients from", title_fontsize=15,
                labelspacing=.45, handlelength=1.8)
leg._legend_box.align = "left"

fig.tight_layout()
out = "embedding_gradient_source_before_after.svg" if OLD else "embedding_gradient_source.svg"
plt.savefig(fig_path(out), dpi=300, bbox_inches="tight")
print(f"saved {out}")
plt.show()
