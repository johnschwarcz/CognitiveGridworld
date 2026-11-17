"""Channel loss and memorization, as pictures (John 11-12 05:30: "i'm really not following. Can you please show me visuals").

Top row: Fig 13's networks (train_cluster_runs.py, |S| 500, batch 10,000), novel goal accuracy of each seed's Classifier
through training, under three conditions that share each seed's start, environments and episodes: oracle labels at lr
1e-4 (gen60k_lr4 / clf60k_lr4, 6 seeds each), oracle labels at lr 1e-3 (gen30k / clf30k, 4 each), the RCC's own answers as
labels at lr 1e-4 (rcc60k_lr4 / rccclf60k_lr4, 6 each). Blue: the Generator route; red: the Classifier route, dashed
where the trained network misses (or barely uses) a whole channel (channel census, fig_channel_census.py: shuffling that
channel's bits costs < .1 accuracy, against .25-.33 for a used channel; 2,000 novel episodes per net). Grey lines: exact
Bayes with all 5 channels on these tests (.768) and with one or two channels missing (minus .071 / .175: exact Bayes with 4
or 3 of 5 channels, scratchpad wf/theory/bayes_subsets.py, 20,000 episodes).
Bottom row: the same Classifier with the shared read-in (main/model/shared_readin.py), its codes learned by the Generator
(genshared20k_lr4, 2 seeds) or by itself (clfshared60k_lr4, 4 seeds): accuracy on novel environments (solid) and on the
training environments (dashed; the last training batch before each test), mean over seeds.

RESULT 11-12 05:45: channel loss in the Classifier route 1 of 6 (lr 1e-4, oracle), 3 of 4 (lr 1e-3, oracle), 5 of 6 (the
RCC's own labels); the Generator route in 0 of 16 (census of all 32 nets, parts 1-3, 11-12 05:30: every channel of every
Generator-route net costs >= .258 when shuffled).
Shared read-in: the Generator route's novel and training accuracy rise together (.48 / .47 at 3,500); the Classifier
route reaches .20-.27 on its training environments while novel stays at chance (.11-.12) through 18-20k: it memorizes.

    python fig_channel_loss.py [--pull]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
from main.utils import fig_path, apply_fig_style

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "cluster_runs", "jsonl")
JOINT, LOSE1, LOSE2 = .768, .071, .175
C_GEN, C_CLF, C_LOST, C_SH = "#08519c", "#a50f15", "#fd8d3c", "#e6550d"
CONDITIONS = (("a   Oracle labels, lr 1e-4", "gen60k_lr4", "clf60k_lr4", 6, 60000),
              ("b   Oracle labels, lr 1e-3", "gen30k", "clf30k", 4, 30000),
              ("c   The RCC's own answers as labels, lr 1e-4", "rcc60k_lr4", "rccclf60k_lr4", 6, 60000))


def pull():
    for pat in ("genshared20k_lr4", "clfshared60k_lr4"):
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/{pat}_s*.jsonl' {LOGS}/ 2>/dev/null",
                       shell=True)


def jl(f):
    return [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []


def lost_channels():
    """{net: [channels whose shuffled bits cost < .1 accuracy]} for every censused net"""
    from fig_channel_census import parse
    return {n: [int(o) for o in np.flatnonzero(d["drop"] < .1)] for n, d in parse().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    L = fs["legend"] - 1
    lost = lost_channels()
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.18, left=.05, right=.99, top=.93, bottom=.07)
    for i, (title, gen, clf, n, xmax) in enumerate(CONDITIONS):
        ax = fig.add_subplot(gs[0, i])
        for y, lab in ((JOINT, "5 ch."), (JOINT - LOSE1, "4 ch."), (JOINT - LOSE2, "3 ch.")):
            ax.axhline(y, color=".55", ls=":", lw=1.2)
            ax.text(xmax * 1.015, y, lab, fontsize=L + 1, color=".3", va="center")
        n_lost = 0
        for k in range(n):
            for arm, is_clf in ((gen, False), (clf, True)):
                r = jl(os.path.join(LOGS, f"{arm}_s{k}.jsonl"))
                if not r:
                    continue
                x = [q["ep"] for q in r]; y = [q["net"] for q in r]
                miss = lost.get(f"{arm}_s{k}", [])
                n_lost += bool(is_clf and miss)
                col, ls, lw = (C_GEN, "-", 1.4) if not is_clf else ((C_LOST, (0, (4, 2)), 2.2) if miss else (C_CLF, "-", 1.4))
                ax.plot(x, y, color=col, ls=ls, lw=lw)
        ax.set(xlim=(0, xmax * 1.13), ylim=(0, .82), xlabel="training iterations (10,000 episodes each)")
        ax.set_title(f"{title}\nClassifier route misses a channel in {n_lost} of {n} runs; Generator route in 0 of {n}",
                     loc="left", fontsize=fs["title"] - 4)
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.set_ylabel("goal accuracy, novel environments")
            ax.plot([], [], color=C_GEN, lw=1.4, label="Generator route (one line per seed)")
            ax.plot([], [], color=C_CLF, lw=1.4, label="Classifier route, uses all 5 channels")
            ax.plot([], [], color=C_LOST, ls=(0, (4, 2)), lw=2.2, label="Classifier route, misses a channel")
            ax.plot([], [], color=".55", ls=":", lw=1.2, label="exact Bayes with 5, 4, 3 channels")
            ax.legend(frameon=False, fontsize=L, loc="lower right")
        print(f"{title[:1]} Classifier route runs missing a channel: {n_lost} of {n}")

    for j, (arm, lab, col, n) in enumerate((("genshared20k_lr4", "Generator route", C_GEN, 2),
                                             ("clfshared60k_lr4", "Classifier route", C_SH, 4))):
        ax = fig.add_subplot(gs[1, j])
        runs = [r for r in (jl(os.path.join(LOGS, f"{arm}_s{k}.jsonl")) for k in range(n)) if r]
        for key, ls, what in (("net", "-", "new environments"), ("train_net", (0, (4, 2)), "its training environments")):
            curves = [{q["ep"]: q[key] for q in r if key in q} for r in runs]
            common = sorted(x for x in set.intersection(*(set(c) for c in curves)) if x <= 20000)
            ax.plot(common, [np.mean([c[x] for c in curves]) for x in common], color=col, ls=ls, lw=2.4, label=what)
            print(f"{arm} {key}: last common {common[-1]}: {np.mean([c[common[-1]] for c in curves]):.3f}")
        ax.axhline(JOINT, color=".55", ls=":", lw=1.2)
        ax.axhline(.1, color=".55", ls=":", lw=1.2)
        ax.text(19800, .1 - .035, "chance", fontsize=L, color=".3", ha="right")
        ax.set(xlim=(0, 20000), ylim=(0, .82), xlabel="training iterations (10,000 episodes each)")
        ax.set_title(f"{'de'[j]}   Shared read-in, codes learned by the {'Generator' if j == 0 else 'Classifier itself'}\n"
                     f"{lab}, mean of {len(runs)} seeds", loc="left", fontsize=fs["title"] - 4)
        ax.legend(frameon=False, fontsize=L, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            ax.set_ylabel("Classifier's goal accuracy")
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(0, .95, "How to read it", fontsize=fs["legend"], va="top")
    ax.text(0, .82, "a-c: a network that misses a channel\nstops one Bayes level down.\n"
                    "Faster steps (b) or noisier labels (c)\nmake the Classifier route miss channels;\n"
                    "the Generator route never does.\n\n"
                    "d-e: with the shared read-in, the\nClassifier route only learns its own\n"
                    "training environments (dashed);\nnew environments stay at chance.",
            fontsize=fs["legend"] - 1, va="top", linespacing=1.45)
    fig.suptitle("$|\\mathcal{S}| = 500$ training environments; every point = 10,000 novel episodes",
                 fontsize=fs["legend"] - 2)
    out = fig_path("fig_channel_loss.png")
    fig.savefig(out, dpi=170)
    print(out)


if __name__ == "__main__":
    main()
