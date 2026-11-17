"""Fair version of the imagination test (John 11-13: "for the dreaming with the classifier, we'd need to make it fair, maybe
training the classifier route with a shuffled channels, or whatever makes it most likely to learn the right representation").

Sources = the nets trained with the channel order shuffled in every episode (readin_perm; ORACLE feedback, the only
shuffled-channel nets that exist: train_cluster_runs.py tasks 185-192). With shuffled channels the Classifier route's seed 1
learned AND its own Generator reads its codes (plug-in .70-.71; the default Classifier route's plug-in is .13); seed 0 never
left chance (plug-in .12). Same teaching protocol as fig_imagination.py: the source net is frozen, its Classifier replaced by
fresh weights (the same per seed in every panel), the only module that trains; batch 1,000, lr 1e-4, 15,000 iterations,
held-out test-state variables masked; one test of 1,000 novel-environment episodes every 250 iterations.
    rows     D = dreams only (both-variable cross-entropy on batches the frozen Generator dreams) | R = real labels
    columns  Generator route | Classifier route
    lines    solid = source trained at lr 1e-4 (genperm20k_lr4 / clfperm20k_lr4), dashed = at lr 1e-3 (genperm10k_lr1e3 /
             clfperm10k_lr1e3); dark = seed 1, light = seed 0; thin dotted line in the run's colour = that frozen
             Generator's plug-in accuracy (its ceiling for D); filled dot = still training (last test before 15,000)
Tasks 302-317 (imagine_/teach_p{gen,clf}{4,3}_s{1,0}), cluster CPU array 66685813. Logs DATA/experimental/cluster_runs/jsonl.

    python fig_imagination_fair.py [--pull]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "cluster_runs", "jsonl")
BLUE, RED = "#2c7bb6", "#d7191c"
LIGHT = {BLUE: "#92c5de", RED: "#f4a582"}
LAST = 15061


def load(name):
    f = os.path.join(LOGS, name + ".jsonl")
    if not os.path.exists(f):
        return []
    return [json.loads(l.replace("NaN", "null")) for l in open(f) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/{{imagine,teach}}_p*.jsonl' {LOGS}/",
                       shell=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.serif": "cmr10", "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7.5, "axes.linewidth": .6})
    fig, axs = plt.subplots(2, 2, figsize=(5.5, 4.6), sharex=True, sharey=True)
    jt, nv = [], []
    for row, (cond, rlabel) in enumerate((("imagine", "Dreams only"), ("teach", "Real labels"))):
        for col, (route, clabel, colr) in enumerate((("gen", "Generator route", BLUE), ("clf", "Classifier route", RED))):
            ax, n = axs[row, col], 0
            for lr, ls in (("4", "-"), ("3", "--")):
                for k in (1, 0):
                    r = load(f"{cond}_p{route}{lr}_s{k}")
                    if not r:
                        continue
                    n += 1
                    c = colr if k == 1 else LIGHT[colr]
                    e, y = np.array([q["ep"] for q in r]), np.array([q["net"] for q in r])
                    ax.plot(e, y, color=c, ls=ls, lw=.9)
                    ax.axhline(np.mean([q["plugin"] for q in r]), color=c, ls=":", lw=.7)
                    if e[-1] < LAST:
                        ax.plot(e[-1], y[-1], "o", ms=3.6, mfc=c, mec="k", mew=.8, zorder=5)
                    jt += [q["joint"] for q in r]
                    nv += [q["naive"] for q in r]
                    print(f"{cond}_p{route}{lr}_s{k}: {len(r)} tests to {e[-1]}, last 5 {y[-5:].mean():.3f}, "
                          f"plug-in {np.mean([q['plugin'] for q in r]):.3f}")
            ax.set_title(f"{rlabel}, {clabel} ({n} run{'' if n == 1 else 's'})", fontsize=8)
            if n == 0:
                ax.text(.5, .5, "not run yet", transform=ax.transAxes, ha="center", va="center", fontsize=8)
    for ax in axs.ravel():
        if jt:
            ax.axhline(np.mean(jt), color="k", ls="--", lw=.8)
            ax.axhline(np.mean(nv), color="k", ls=(0, (1, 1)), lw=.9)
        ax.axhline(.1, color=".7", lw=.6)
        ax.set(xlim=(0, 15500), ylim=(0, .82))
        ax.set_xticks([0, 5000, 10000, 15000], ["0", "5k", "10k", "15k"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axs[1]:
        ax.set_xlabel("training iterations (1,000 episodes each)")
    for ax in axs[:, 0]:
        ax.set_ylabel("goal accuracy, final step\n(novel environments)")
    h = [Line2D([], [], color=".3", lw=1), Line2D([], [], color=".3", lw=1, ls="--"),
         Line2D([], [], color="k", ls="--", lw=.8), Line2D([], [], color="k", ls=(0, (1, 1)), lw=.9),
         Line2D([], [], color=".3", ls=":", lw=.8), Line2D([], [], color=".5", ls="", marker="o", ms=3.6, mec="k")]
    fig.legend(h, ["source trained at lr $10^{-4}$", "source trained at lr $10^{-3}$", "Joint Bayes", "Naive Bayes",
                   "frozen Generator's plug-in", "still training"], ncol=3, loc="upper center", frameon=False,
               bbox_to_anchor=(.5, 1.0))
    fig.text(.5, .87, "fresh Classifier on a frozen net trained with SHUFFLED channel order (oracle feedback); dark = seed 1,"
                      " light = seed 0;\nbatch 1,000, lr $10^{-4}$; 1 test of 1,000 novel-environment episodes per 250 "
                      "iterations", ha="center", fontsize=7, linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, .85))
    out = fig_path("fig_imagination_fair.png")
    fig.savefig(out, dpi=300)
    print(out)


if __name__ == "__main__":
    main()
