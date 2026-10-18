"""Oracle ablation training losses: classifier and generator loss for each condition.

    python main/REPLICATION/claude_plots/plot_oracle_losses.py [--old DIR]

Reads main/DATA/oracle_pilot/{default,clf_reg}_rep*_net.pth: mean +/- SEM over reps of the
classifier loss (pol_loss) and generator loss (gen_loss). Each logged value is the loss of the
last training episode before a checkpoint (every 250 episodes); entry 0 is never written and is
dropped. L_reg is included in whichever loss trains the embeddings (generator loss for
"default", classifier loss for "clf_reg"). --old DIR adds DIR/default_rep*_net.pth dashed,
e.g. Generator-arm runs trained with a different classifier.

Writes oracle_losses.svg/.png in figure_bin.
"""
import os, sys, glob, re, inspect
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter, NullFormatter

root = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
while not os.path.exists(os.path.join(root, 'main', 'CognitiveGridworld.py')):
    root, prev = os.path.dirname(root), root
    if root == prev:
        raise RuntimeError("project root not found")
sys.path.insert(0, root)
from main.utils import fig_path, apply_fig_style

DATA = os.path.join(root, "main", "DATA", "oracle_pilot")
CKPT = 250          # checkpoint_every in train_oracle_ablation.py
# (condition, colour, label, linestyle) as in plot_ablation.py
SHOW = [("default", "#2c7bb6", "Generator", "-"),
        ("clf_reg", "#d7191c", "Classifier", "-")]
LOSSES = [("pol_loss_through_training", "classifier loss", "linear"),
          ("gen_loss_through_training", "generator loss", "log")]


def load(run_dir, cond):
    files = sorted(glob.glob(os.path.join(run_dir, f"{cond}_rep*_net.pth")),
                   key = lambda f: int(re.search(r"rep(\d+)", f).group(1)))
    if not files:
        raise SystemExit(f"no {cond}_rep*_net.pth in {run_dir}")
    runs = [torch.load(f, weights_only = False, map_location = "cpu", mmap = True) for f in files]
    return {key: np.stack([np.ravel(np.asarray(r[key], float))[1:] for r in runs]) for key, _, _ in LOSSES}


def main():
    curves = [(lab, c, ls, load(DATA, cond)) for cond, c, lab, ls in SHOW]
    if "--old" in sys.argv:
        old = sys.argv[sys.argv.index("--old") + 1]
        curves.insert(1, ("Generator, comparison runs", "#2c7bb6", "--", load(old, "default")))

    fs = apply_fig_style()
    fig, axes = plt.subplots(1, 2, figsize = (7.6, 2.6))
    for ax, (key, title, yscale) in zip(axes, LOSSES):
        for lab, c, ls, L in curves:
            A = L[key]
            x = np.arange(1, A.shape[1] + 1) * CKPT / 1000
            m, se = A.mean(0), A.std(0, ddof = 1) / np.sqrt(len(A))
            ax.fill_between(x, m - se, m + se, color = c, alpha = .2, lw = 0)
            ax.plot(x, m, color = c, ls = ls, lw = 1.5, label = f"{lab} (n={len(A)})")
            print(f"{title:<16} {lab:<36} final {m[-1]:.4f} +/- {se[-1]:.4f}")
        ax.set_yscale(yscale)
        if yscale == "log":                     # plain-number ticks across the 0.02-0.4 range
            ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.4])
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, fontsize = fs["label"] + 1)
        ax.set_xlabel("training episodes (thousands)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins = 5, integer = True))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].set_ylabel("loss")
    axes[1].legend(loc = "center left", bbox_to_anchor = (1.02, .5), frameon = False, fontsize = fs["tick"],
                   title = r"$\hat{\mathcal{E}}$ receives gradients from", title_fontsize = fs["tick"])
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(fig_path(f"oracle_losses.{ext}"), bbox_inches = "tight", dpi = 200 if ext == "png" else None)
    print(fig_path("oracle_losses.svg"))


if __name__ == "__main__":
    main()
