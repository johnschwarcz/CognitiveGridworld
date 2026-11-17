"""What the Generator knows vs what the Classifier does (John 11-11 23:00: "Is it possible that understanding this would
reveal a dramatic shortcut to generator training?").

Fig 13's networks (train_cluster_runs.py, |S| 500, batch 10,000, lr 1e-4, 60,000 iterations, 6 seeds each); every test =
10,000 novel episodes (both variables held out). Per arm (thin = seeds, thick = mean +- SEM where every seed has a test):
    net      the Classifier's goal accuracy at T (what the network does)
    plugin   exact Bayes with the network's OWN Generator likelihood (what its generative model knows)
a: oracle feedback, b: the RCC's own feedback.

RESULT 11-11 23:15 (per seed, first test at >= .65 novel accuracy):
    oracle, Generator route    plugin 1,250-1,750; its Classifier 9,750-11,000 (~7x later)
    oracle, Classifier route   Classifier 19,250-29,250 in 5 of 6 seeds (seed 2 never: .526); its Generator ~.13 throughout
                               (it trains on detached Z and cannot read the Classifier's relabeled code)
    RCC feedback, Gen route    plugin 13,250-15,750; Classifier 25,500-28,750
    RCC feedback, Clf route    Classifier never reaches .65 in 60k (6 seeds end .38-.63, mean .54; seed 4 below Naive)
-> the generative model reaches .65 11-23x sooner than the Classifier route's network (oracle feedback); the slow part of
   the Generator route is its own Classifier learning to do inference, not the embeddings.

    python fig13_plugin_lead.py
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

DATA = os.path.join(ROOT, "main", "DATA")
LOGS = os.path.join(DATA, "experimental", "cluster_runs", "jsonl")
PANELS = (("a   Oracle feedback", (("gen60k_lr4", "plugin", "#08519c", "-", "Generator route: its generative model + Bayes"),
                                     ("gen60k_lr4", "net", "#6baed6", "-", "Generator route: its Classifier"),
                                     ("clf60k_lr4", "net", "#a50f15", "-", "Classifier route: its Classifier"))),
          ("b   The RCC's own feedback", (("rcc60k_lr4", "plugin", "#08519c", "-", "Generator route: its generative model + Bayes"),
                                          ("rcc60k_lr4", "net", "#6baed6", "-", "Generator route: its Classifier"),
                                          ("rccclf60k_lr4", "net", "#a50f15", "-", "Classifier route: its Classifier"))))


def curves(arm, key):
    runs = []
    for k in range(6):
        f = os.path.join(LOGS, f"{arm}_s{k}.jsonl")
        if os.path.exists(f):
            r = [json.loads(l) for l in open(f) if l.strip()]
            runs.append((np.array([q["ep"] for q in r]), np.array([q[key] for q in r], float)))
    L = max(len(e) for e, _ in runs)
    A = np.full((len(runs), L), np.nan)
    for i, (_, y) in enumerate(runs):
        A[i, :len(y)] = y
    x = max(runs, key=lambda r: len(r[0]))[0]
    n = np.isfinite(A).sum(0)
    ok = n == len(runs)                                  # the mean only where every seed has a test
    m = np.nanmean(A[:, ok], 0)
    se = np.nanstd(A[:, ok], 0, ddof=1) / np.sqrt(n[ok])
    return x[ok], m, se, len(runs), A, x


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    joint, naive = np.load(os.path.join(DATA, "cache", "bayes_bounds_ctx2_T30.npy"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, constrained_layout=True)
    for ax, (title, lines) in zip(axes, PANELS):
        for arm, key, col, ls, lab in lines:
            x, m, se, n, A, xall = curves(arm, key)
            keep = x > 0
            for row in A:
                ax.plot(xall[1:], row[1:], color=col, ls=ls, lw=.5, alpha=.35)
            ax.fill_between(x[keep], (m - se)[keep], (m + se)[keep], color=col, alpha=.25, lw=0)
            ax.plot(x[keep], m[keep], color=col, ls=ls, lw=2, label=f"{lab} ({n})")
            first = [xall[np.argmax(row >= .65)] for row in A if (row >= .65).any()]
            print(f"{title[:28]:28s} {lab:48s} n {n}: .65 reached by {len(first)} seeds at "
                  f"{(min(first), max(first)) if first else None}; end {m[-1]:.3f}")
        ax.axhline(joint, color="k", ls="--", lw=1)
        ax.axhline(naive, color="k", ls=":", lw=1)
        ax.text(250, joint + .01, "Joint Bayes", fontsize=fs["legend"] - 3)
        ax.text(250, naive + .01, "Naive Bayes", fontsize=fs["legend"] - 3)
        ax.set_xscale("log")
        ax.set_xlim(250, 61000)
        ax.set_ylim(0, .8)
        ax.set_xlabel("training iterations (batches of 10,000 episodes)")
        ax.set_title(title, loc="left", fontsize=fs["title"] - 3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("goal accuracy, novel environments")
    axes[0].legend(frameon=False, fontsize=fs["legend"] - 3, loc="lower right")
    fig.suptitle("What the generative model knows vs what the Classifier does (Fig 13's networks: $|\\mathcal{S}| = 500$, lr $10^{-4}$; "
                 "thin = seeds, thick = mean $\\pm$ SEM, (n) seeds; 10,000 novel episodes per test)", fontsize=fs["legend"] - 2)
    out = fig_path("fig13_plugin_lead.png")
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
