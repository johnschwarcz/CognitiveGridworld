"""embedding_gradient_source plus the oracle Z-flow classifier runs.

    python main/REPLICATION/plot_ablation_zflow.py

Exploratory twin of plot_ablation.py, kept separate so that script stays clean of the
Z-flow condition. Same data, same styling, plus:

    oracle_clffix_v3  the classifier given the generator's Z-flow treatment -- per-stream
                      LayerNorm(hid_dim) after the readin projection, readout-aligned Z
                      block, and the skip added after the cumsum (oracle_zflow=True).

Earlier variants (oracle_clffix, oracle_clffix_v2) are superseded and not plotted.

Writes embedding_gradient_source_zflow.svg, so neither script overwrites the other's figure.
"""
import os, sys, glob, re, inspect
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _repo_root():
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
BOUNDS_CACHE = os.path.join(root, "bayes_bounds_ctx2_T30.npy")
FIGSIZE = (10, 3.6)
# (condition, colour, label, linestyle, run dir, live-log prefix or None)
SHOW = [("default", "#2c7bb6", "Generator",          "-",  "oracle_pilot",     None),
        ("clf_reg", "#d7191c", "Classifier",         "-",  "oracle_pilot",     None),
        ("clf_reg", "#7b3294", "Classifier + Z-flow", "--", "oracle_clffix_v3", "v3_rep")]

# Reps still training have no .pth yet; their checkpoints are read from the live stdout
# logs instead and drawn as individual partial lines, never folded into the mean.
LOG_DIR = os.environ.get("ZFLOW_LOG_DIR", "/tmp/claude-1002/"
    "-home-johns-Desktop-CognitiveGridworld/c1b806fc-cd7c-4d16-a547-8bf7ffe7ab21/scratchpad")
LOG_LINE = re.compile(r"ep\s+(\d+)\s+\(\s*\d+M\)\s+train\s+[\d.]+\s+test\s+([\d.]+)")


def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=4, width=.9)


def get_bounds(batch_num=8000, step_num=30, ctx_num=2):
    if os.path.exists(BOUNDS_CACHE):
        return tuple(np.load(BOUNDS_CACHE))
    from main.CognitiveGridworld import CognitiveGridworld
    b = CognitiveGridworld(mode=None, show_plots=False, episodes=1, ctx_num=ctx_num,
                           obs_num=5, realization_num=10, state_num=500,
                           batch_num=batch_num, step_num=step_num)
    out = np.array([b.joint_acc.mean(0)[-1], b.naive_acc.mean(0)[-1]])
    np.save(BOUNDS_CACHE, out)
    return tuple(out)


def load_live(prefix, skip_reps):
    """Partial curves for reps that are still running, keyed by rep index."""
    out = {}
    if prefix is None or not os.path.isdir(LOG_DIR):
        return out
    for f in sorted(glob.glob(os.path.join(LOG_DIR, f"{prefix}*.log"))):
        r = int(re.search(r"rep(\d+)", os.path.basename(f)).group(1))
        if r in skip_reps:
            continue
        rows = LOG_LINE.findall(open(f, errors="ignore").read().replace("\r", "\n"))
        if rows:
            out[r] = np.array([[int(e) // 250 + 1, float(a)] for e, a in rows])
    return out


def load(cond, run_dir):
    A = []
    for f in sorted(glob.glob(os.path.join(DATA, run_dir, f"{cond}_rep*_net.pth")),
                    key=lambda p: int(re.search(r"rep(\d+)", p).group(1))):
        d = torch.load(f, weights_only=False, map_location="cpu")
        A.append(np.asarray(d["test_acc_through_training"], float)[:, -1])
    if not A:
        return None                       # run not collected yet -- skip the curve
    n = min(len(a) for a in A)
    return np.stack([a[:n] for a in A])


fs = apply_fig_style()
joint, naive = get_bounds()
fig, ax = plt.subplots(figsize=FIGSIZE)

for cond, c, lab, ls, run_dir, log_prefix in SHOW:
    A = load(cond, run_dir)
    done = set()
    if A is not None:
        done = {int(re.search(r"rep(\d+)", f).group(1)) for f in
                glob.glob(os.path.join(DATA, run_dir, f"{cond}_rep*_net.pth"))}
    live = load_live(log_prefix, done)
    for i, (r, arr) in enumerate(sorted(live.items())):
        ax.plot(arr[:, 0], joint - arr[:, 1], color=c, ls=ls, lw=.9, alpha=.5,
                label=f"{lab}, in progress" if i == 0 else None)
        print(f"{lab:<12} rep{r} IN PROGRESS to ep {int(arr[-1,0]-1)*250}, "
              f"test {arr[-1,1]:.4f}")
    if A is None:
        print(f"{lab:<12} -- no completed runs under {run_dir}")
        continue
    x = np.arange(1, A.shape[1] + 1)
    gap = joint - A
    m = gap.mean(0)
    if A.shape[0] > 1:
        se = gap.std(0, ddof=1) / np.sqrt(A.shape[0])
        ax.fill_between(x, m - se, m + se, color=c, alpha=.22, lw=0)
        print(f"{lab:<12} n={A.shape[0]}  final excess {m[-1]:.4f} +/- {se[-1]:.4f}")
    else:
        print(f"{lab:<12} n=1  final excess {m[-1]:.4f}")
    ax.plot(x, m, color=c, ls=ls, lw=2, label=f"{lab} ($n{{=}}{A.shape[0]}$)")

ax.axhline(joint - naive, c='k', lw=2, ls='--', zorder=10)
ax.text(.02, joint - naive, "Naive bound", color='k', fontsize=15, va='center',
        ha='left', transform=ax.get_yaxis_transform(), zorder=11,
        bbox=dict(fc='white', ec='none', alpha=.85, pad=1.5))

ax.set(xlabel="Testing Episode", ylabel="Optimal $-$ Network Accuracy",
       xscale="log", yscale="log", xlim=(1, 41))
ax.set_title("RCC Generator Ablation + Z-flow", fontsize=fs["title"])
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(mticker.LogLocator(subs=(1., 2., 3., 5.)))
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(mticker.NullFormatter())
ax.grid(alpha=.25, lw=.6, which="both")
_despine(ax)
leg = ax.legend(loc="lower left", frameon=False, fontsize=13,
                title=r"$\hat{\mathcal{E}}$ receives"+"\ngradients from", title_fontsize=13,
                labelspacing=.4, handlelength=1.8)
leg._legend_box.align = "left"

fig.tight_layout()
out = fig_path("embedding_gradient_source_zflow.svg")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out.replace(".svg", ".png"), dpi=300, bbox_inches="tight")
print("saved", os.path.basename(out))
