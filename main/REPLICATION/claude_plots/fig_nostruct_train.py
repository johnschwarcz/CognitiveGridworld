"""Has the no-world-structure pair learned anything on its TRAINING episodes? (John 11-14 07:00 FL)

Every test record of train_cluster_runs.py also holds three numbers measured on the last TRAINING batch before that test
(1,000 training episodes, training variables only): train_net (the Classifier's goal accuracy at T), train_plugin (exact
Bayes with the Generator's learned likelihood, the Generator's world model) and train_kl_env (E_r KL(P(o|r) || P^(o|r)),
summed over channels: how far the Generator's likelihood is from the environment's). If a network memorised its training
environments without generalising, these would rise while the novel-environment tests stay at chance.
    thick solid  no world structure (z_full: full embeddings per channel, no down-scaling, no norm loss), tasks 401-402
    dashed       only the dot products removed (z_raw), tasks 389-390, stopped at 282k
    thin light   the shuffled-channel runs WITH the structure (rccpermb1k1m_lr4_s0-7, rccclfpermb1k1m_lr4_s0-8), reference
    blue = Generator route, red = Classifier route; all batch 1,000, shuffled channels, the RCC's own feedback, lr 1e-4
Every line is a running mean over 5 tests (5 training batches, 1,250 iterations).

NUMBERS (11-14 07:05 FL, no-world-structure pair at ~287k; Generator / Classifier route):
    train_net .130 / .137 (test .121 / .130); train_plugin .123 / .114 (chance .10; reference runs .54-.66 after learning);
    train_kl_env 2.78 / 2.74 nats (start 1.2, peak ~10 at 25k; reference 2.0-2.4 at chance, .07-.13 after learning);
    gen_loss 1.64 -> 1.21 / 1.24 (reference 1.51 -> 1.16 by 50k while still at chance, .16-.25 after); policy entropy 2.24
    of the maximum 2.30 (no collapse). Same at the dot-products-removed pair (282k). Before the reference runs left chance
    their training-batch numbers moved first (Clf s3 at 100k: train_net .19 vs test .14, train_plugin .16 vs .13); nothing
    like it here: no memorisation of the training environments, just no learning.

    python fig_nostruct_train.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_batch_routes as fb
from main.utils import fig_path

KEYS = [("train_net", r"$\mathbf{a}$   Classifier accuracy, training episodes", "goal accuracy at $T$", False),
        ("train_plugin", r"$\mathbf{b}$   Generator's world model (plug-in), training episodes", "Bayes accuracy with $\\hat P$", False),
        ("train_kl_env", r"$\mathbf{c}$   likelihood error, training environments", r"$\mathrm{KL}(P\,\|\,\hat P)$ (nats)", True)]


def series(rs, key):
    e = np.array([r["ep"] for r in rs], float)
    v = np.array([r.get(key) if r.get(key) is not None else np.nan for r in rs], float)
    m = np.convolve(np.nan_to_num(v), np.ones(5) / 5, mode="valid")
    return e[4:], m


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.serif": "cmr10", "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7.5, "axes.linewidth": .6})
    groups = [("ref", [(("rccpermb1k1m_lr4", None), k, fb.BLUE) for k in range(8)] +
                      [(("rccclfpermb1k1m_lr4", None), k, fb.RED) for k in range(9)], dict(lw=.6, alpha=.35, ls="-")),
              ("raw", [(("rccrawpermb1k1m_lr4", None), 0, fb.BLUE), (("rccclfrawpermb1k1m_lr4", None), 0, fb.RED)],
               dict(lw=1.0, alpha=.9, ls="--")),
              ("full", [(("rccfullpermb1k1m_lr4", None), 0, fb.BLUE), (("rccclffullpermb1k1m_lr4", None), 0, fb.RED)],
               dict(lw=1.6, alpha=1, ls="-"))]
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.1))
    n_ref = 0
    for ax, (key, title, ylab, logy) in zip(axs, KEYS):
        for tag, runs, style in groups:
            for arm, k, colr in runs:
                rs = fb.load(*arm, k)
                if len(rs) < 6:
                    continue
                n_ref += tag == "ref" and key == "train_net"
                x, y = series(rs, key)
                ax.plot(x / 1e3, y, color=colr, zorder=3 if tag == "full" else 2, **style)
        ax.set(xlim=(0, 300), xlabel="training iterations (thousands; batch 1,000)", ylabel=ylab)
        if logy:
            ax.set_yscale("log")
        else:
            ax.axhline(.1, color=".6", lw=.6)
            ax.set_ylim(.08, .72)
        ax.set_title(title, fontsize=8, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h = [Line2D([], [], color=".2", lw=1.6), Line2D([], [], color=".2", lw=1.0, ls="--"),
         Line2D([], [], color=".2", lw=.6, alpha=.5), Line2D([], [], color=fb.BLUE, lw=1.6), Line2D([], [], color=fb.RED, lw=1.6)]
    fig.legend(h, ["no world structure (1 + 1)", "only dot products removed (1 + 1)",
                   f"with structure ({n_ref} shuffled-channel runs)", "Generator route", "Classifier route"], ncol=5,
               loc="upper center", frameon=False, bbox_to_anchor=(.5, 1.0))
    fig.text(.5, .895, "measured on the last training batch before each test (1,000 training episodes); running mean over 5 "
                       "tests; shuffled channels, the RCC's own feedback", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, .88))
    out = fig_path("fig_nostruct_train.png")
    fig.savefig(out, dpi=300)
    print(out)


if __name__ == "__main__":
    main()
