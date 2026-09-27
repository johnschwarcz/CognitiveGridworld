"""Oracle Z-flow fixes on the classifier's embedding path.

    python main/REPLICATION/plot_oracle_clffix.py

Left: every oracle-ablation condition against training samples. The three baselines are
the 50000-episode pilot runs in /oracle_ablation; the Z-flow run is 10000 episodes, so it
stops at 100M -- the curves are only comparable left of that mark.
Right: the clf_reg condition alone, matched at 100M.
"""
import os, sys, inspect
import numpy as np
import torch
import matplotlib.pyplot as plt

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

BASE = os.path.join(root, "main", "DATA", "oracle_ablation")
NEW = os.path.join(root, "main", "DATA", "oracle_clffix")
BOUNDS_CACHE = os.path.join(root, "bayes_bounds_ctx2_T30.npy")
SAMPLES_PER_CKPT = 250 * 10000 / 1e6        # checkpoint_every x batch_num, in millions
MATCHED = 100.0                              # where the Z-flow run ends

BLUE, RED, GREY = "#2c7bb6", "#d7191c", "#8c8c8c"


def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=4, width=.9)


def load(path):
    d = torch.load(path, weights_only=False, map_location="cpu")
    a = np.asarray(d["test_acc_through_training"], float)[:, -1]
    return np.arange(len(a)) * SAMPLES_PER_CKPT, a


def main():
    fs = apply_fig_style()
    runs = {
        "Generator": (os.path.join(BASE, "default_rep0_net.pth"), BLUE, "-"),
        "Classifier": (os.path.join(BASE, "clf_reg_rep0_net.pth"), RED, "-"),
        "Both": (os.path.join(BASE, "both_rep0_net.pth"), GREY, "-"),
    }
    data = {k: load(p) for k, (p, _, _) in runs.items()}
    xz, az = load(os.path.join(NEW, "clf_reg_rep0_net.pth"))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.8, 2.4))

    for name, (_, c, ls) in runs.items():
        x, a = data[name]
        ax0.plot(x, a, color=c, ls=ls, label=name)
    ax0.plot(xz, az, color=RED, ls="--", label="Classifier + Z-flow")
    ax0.axvline(MATCHED, color="k", lw=.7, ls=":", zorder=0)
    ax0.set_xlabel("training samples (M)")
    ax0.set_ylabel("test accuracy")
    ax0.set_title("all conditions", fontsize=fs["label"])
    ax0.set_ylim(.02, .82)
    ax0.legend(frameon=False, fontsize=fs["tick"] - 1, loc="lower right",
               bbox_to_anchor=(1.0, .01), labelspacing=.3, handlelength=1.6,
               borderaxespad=.2)

    if os.path.exists(BOUNDS_CACHE):
        joint, naive = np.load(BOUNDS_CACHE)
        for y, lab in ((joint, "Joint"), (naive, "Naive")):
            ax0.axhline(y, color="k", lw=.6, ls=(0, (1, 3)), zorder=0)
            ax0.annotate(lab, (ax0.get_xlim()[0], y), fontsize=fs["tick"] - 1.5,
                         xytext=(3, 2), textcoords="offset points",
                         va="bottom", ha="left", color="k")

    xb, ab = data["Classifier"]
    m = xb <= MATCHED
    ax1.plot(xb[m], ab[m], color=RED, ls="-", label="Classifier")
    ax1.plot(xz, az, color=RED, ls="--", label="+ Z-flow")
    ax1.set_ylim(.07, .45)
    ax1.annotate(f"{az[-1]:.3f}", (xz[-1], az[-1]), fontsize=fs["tick"] - 1,
                 xytext=(-1, 6), textcoords="offset points", ha="right", color=RED)
    ax1.annotate(f"{ab[m][-1]:.3f}", (xb[m][-1], ab[m][-1]), fontsize=fs["tick"] - 1,
                 xytext=(-1, -11), textcoords="offset points", ha="right", color=RED)
    ax1.set_xlabel("training samples (M)")
    ax1.set_ylabel("test accuracy")
    ax1.set_title("classifier path, matched budget", fontsize=fs["label"])
    ax1.legend(frameon=False, fontsize=fs["tick"] - 1, loc="upper left",
               labelspacing=.3, handlelength=1.6, borderaxespad=.2)

    for ax in (ax0, ax1):
        _despine(ax)

    fig.tight_layout()
    out = fig_path("oracle_clffix.svg")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".svg", ".png"), dpi=300, bbox_inches="tight")
    print("saved", out)
    print("saved", out.replace(".svg", ".png"))


if __name__ == "__main__":
    main()
