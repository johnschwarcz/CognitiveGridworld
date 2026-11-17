"""Exploratory (11-13): can the network teach a NEW Classifier from imagination alone?

A trained RCC (the RCC's own feedback) is loaded read-only; its embeddings and Generator are frozen; its Classifier is replaced
by fresh weights (one fresh Classifier per seed, the same in all six panels), the only module that trains
(train_cluster_runs.py tasks imagine_* / teach_* / rl_*, model_backward.frozen_backward). Sources, matched in accuracy on novel
environments (~.69-.71): Generator route rcc60k_lr4_s0-3 (60k iterations; plug-in .69-.71) and Classifier route
rccclf120k_lr4_s0-3 (120k; plug-in .13-.14). Within a seed both routes share the environments and testing episodes.
    D (top row)     dream only: the fresh Classifier's only loss is the both-variable cross-entropy on batches dreamed by
                    the frozen Generator (no real loss, no real-batch gradient)
    (R, real labels = the true values of both variables: run on 11-13 but LEFT OUT of the figure; John 11-13 20:10: no oracle
     feedback in route runs, exact feedback on both variables only from dreams)
    F (bottom row)   the RCC's own feedback: RL_loss on the real batch (one reward bit on one sampled goal answer)
All: batch 1,000, lr 1e-4, entropy .01, 15,000 training iterations, held-out test-state variables left out of the D / R
losses; one test of 1,000 novel-environment episodes (the batch) every 250 iterations. Lines: one per seed (n = 4 per
panel), the fresh Classifier's goal accuracy at T on the tests; black dashed = Joint, grey dotted = Naive (means over seeds and tests); coloured dashed =
the frozen Generator's plug-in accuracy (Bayes with its learned likelihood; mean over seeds and tests); thin grey = chance.
Predictions stated before launch: D-Generator rises toward its plug-in (.69-.71, its ceiling); D-Classifier stays at or
below its plug-in (~.13), near chance (its Generator is at chance, so this outcome follows from its plug-in); R rises in both
routes; F rises more slowly than R.

Printed: per run, the number of tests, the final accuracy (mean of the last 5 tests) and the first iteration of 3 consecutive
tests above that test's Naive.

    python fig_imagination.py [--pull] [--logs DIR] [--out PNG]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "cluster_runs", "jsonl")
BLUE, RED = "#2c7bb6", "#d7191c"
SEEDS = range(4)
ROWS = [("imagine", "Dreams only"), ("rl", "Own feedback (RL)")]   # 11-13 20:10 John: no oracle feedback in route runs:
# the real-label (true values) arm teach_* is left out of the figure (its logs stay in DATA/experimental/cluster_runs/jsonl)
COLS = [("gen", "Generator route", BLUE), ("clf", "Classifier route", RED)]
ENDED = {"imagine_gen_s0",      # stopped by the runner's 13.5 h wall-time budget at episode 14,846 (cluster log), not training
         "rl_clf_s2", "rl_clf_s3"}   # stopped by STOP files 11-14 03:06 IST at ~12k (John: "let's stop them"; they tracked s0-1)


def load(logs, name):
    f = os.path.join(logs, name + ".jsonl")
    if not os.path.exists(f):
        return []
    return sorted((json.loads(l.replace("NaN", "null")) for l in open(f) if l.strip()), key=lambda r: r["ep"])


def iters(ep, every=250):
    """training iterations before the test at episode ep (tests at 0, every, 2 * every, ... and the last episode)"""
    return ep - -(-ep // every)


def crossing(rows, n=3):
    run = 0
    for r in rows:
        run = run + 1 if r["net"] > r["naive"] else 0
        if run == n:
            return iters(r["ep"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true", help="copy the logs from the cluster first")
    ap.add_argument("--logs", default=LOGS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.pull:
        os.makedirs(a.logs, exist_ok=True)
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/{{imagine,teach,rl}}_*_s*.jsonl' {a.logs}/",
                       shell=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fs = apply_fig_style()
    nr = len(ROWS)
    fig, axs = plt.subplots(nr, 2, figsize=(2 * 2.9 + 1.3, nr * 2.0 + 1.5), sharex=True, sharey=True)
    xmax, counts = 0, []
    for i, (cond, rlabel) in enumerate(ROWS):
        for j, (route, clabel, col) in enumerate(COLS):
            ax, joint, naive, plug, n = axs[i, j], [], [], [], 0
            for k in SEEDS:
                name = f"{cond}_{route}_s{k}"
                rows = load(a.logs, name)
                if not rows:
                    continue
                n += 1
                ep = np.array([iters(r["ep"]) for r in rows])
                ax.plot(ep, [r["net"] for r in rows], color=col, lw=1.0, alpha=.85)
                if rows[-1]["ep"] < 15061 and f"{cond}_{route}_s{k}" not in ENDED:   # still training (11-13 status markers)
                    ax.plot(ep[-1], rows[-1]["net"], "o", ms=5, mfc=col, mec="k", mew=.9, zorder=5)
                joint += [r["joint"] for r in rows]
                naive += [r["naive"] for r in rows]
                plug += [r["plugin"] for r in rows if r.get("plugin") is not None]
                xmax = max(xmax, ep[-1])
                fin = np.mean([r["net"] for r in rows[-5:]])
                print(f"{name}: {len(rows)} tests to iteration {ep[-1]} | final (last 5) {fin:.3f} | 3 tests > Naive from iteration "
                      f"{crossing(rows)} | plug-in {np.mean([r['plugin'] for r in rows if r.get('plugin') is not None]):.3f}")
            counts.append(n)
            if n:
                ax.axhline(np.mean(joint), color="k", ls="--", lw=1)
                ax.axhline(np.mean(naive), color="0.45", ls=":", lw=1.3)
                if plug:
                    ax.axhline(np.mean(plug), color=col, ls="--", lw=1, alpha=.6)
            else:
                ax.text(.5, .5, "not run yet", transform=ax.transAxes, ha="center", va="center", fontsize=fs["annot"])
            ax.axhline(.1, color="0.7", lw=.7)
            ax.set_title(f"{rlabel}, {clabel}", fontsize=fs["label"] + 1)
            if i == nr - 1:
                ax.set_xlabel("Training iterations (1,000 episodes each)")
            if j == 0:
                ax.set_ylabel(r"Accuracy at $T$, $P(\hat r_g = r_g)$")
            ax.spines[["top", "right"]].set_visible(False)
    axs[0, 0].set_ylim(0, .85)
    axs[0, 0].set_xlim(0, max(xmax, 1))
    if xmax >= 2000:
        axs[0, 0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    handles = [Line2D([], [], color=BLUE, lw=1), Line2D([], [], color=RED, lw=1), Line2D([], [], color="k", ls="--", lw=1),
               Line2D([], [], color="0.45", ls=":", lw=1.3), Line2D([], [], color="0.4", ls="--", lw=1, alpha=.6),
               Line2D([], [], color="0.5", ls="", marker="o", ms=5, mec="k")]
    fig.legend(handles, ["fresh Classifier, one line per seed (Generator route)", "same (Classifier route)", "Joint",
                         "Naive", "frozen Generator's plug-in", "still training"], loc="lower center", ncol=2, frameon=False,
               fontsize=fs["legend"])
    nseeds = f"{min(counts)}" if min(counts) == max(counts) else f"{min(counts)}-{max(counts)}"
    fig.suptitle(f"New Classifier from a frozen trained RCC: n = {nseeds} seed{'' if nseeds == '1' else 's'} per panel, "
                 "tests of 1,000 episodes on novel environments", fontsize=fs["label"])
    fig.tight_layout(rect=(0, .14 * 2 / nr, 1, .97))
    out = a.out or fig_path("fig_imagination.png")
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
