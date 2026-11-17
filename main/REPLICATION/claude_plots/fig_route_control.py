"""The Classifier-route control as it stands (John 11-12 05:50: "Can you please show me the proper classifier route control
that we have? Do we have updated RCC feedback, generator route vs classifier route?").

The RCC as updated (gen_zflow Generator) with the Classifier's default dense read-in; the only difference between the arms
is which loss trains the embeddings (Generator route: the Generator's; Classifier route: the Classifier's).
train_cluster_runs.py, Fig 13's protocol: |S| 500, batch 10,000, lr 1e-4, 60k iterations, seeds 0-5 (each seed: the same
start, environments and episodes in all four arms); 10,000 novel episodes per test.
    a  oracle labels (gen60k_lr4 / clf60k_lr4)
    b  the RCC's own answers as labels (rcc60k_lr4 / rccclf60k_lr4); continued to 120k when those logs exist
       (rccclf120k_lr4, rcc120k_lr4: RESUME from the 60k nets); in a, clf120k_lr4_s2 (the seed that lost channels 2 and 3,
       continued on the local GPU, task 183; log copied from runs/cluster_runs)
Thin = seeds (dashed: the trained net misses a whole channel, fig_channel_census.py), thick = mean at the tests every seed
has. The Classifier's goal accuracy on novel environments.

RESULT 11-12 22:50 (120k): b the Classifier route's seeds 0-3 end at .697-.715 (mean .707; the Generator route's 6 seeds
end at 60k at .697 mean), so with the RCC's own answers the Classifier route gets there in about twice the iterations;
seeds 4-5 (.38 / .57 at 60k) are queued to 120k (cluster 66675773_2). a the oracle seed that lost 2 channels ends at
.533 at 120k (flat from 30k). The channel census (dashed) was measured on the 60k nets.
RESULT 11-13 12:25 (all 6 Classifier-route seeds at 120k, b): s0-4 .697-.715, s5 .636 (still ignores channel 0, census
66685419); reach .65 at 64-94k in 5 of 6 (s5 never) vs 25.5-28.75k for the Generator route (6/6 .691-.704 at 60k).
RESULT 11-12 12:00: the oracle seed that lost 2 channels stays flat to 90k (.52-.53 from 30k on: no recovery).
RESULT 11-12 05:50 (60k): a Generator route .71-.74, Classifier route .70-.72 in 5 seeds and .52 in the seed that misses
channels 2 and 3; b Generator route .68-.71 (6/6 past .65 by 25.5-28.75k), Classifier route .38-.63 and still rising
(.06-.15 per 10k), 5 of 6 missing or barely using a channel.

    python fig_route_control.py [--pull]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
from main.utils import fig_path, apply_fig_style

DATA = os.path.join(ROOT, "main", "DATA")
LOGS = os.path.join(DATA, "experimental", "cluster_runs", "jsonl")
C_GEN, C_CLF = "#08519c", "#a50f15"
PANELS = (("a   Oracle labels", (("gen60k_lr4", None), ("clf60k_lr4", "clf120k_lr4"))),
          ("b   The RCC's own answers as labels", (("rcc60k_lr4", "rcc120k_lr4"), ("rccclf60k_lr4", "rccclf120k_lr4"))))


def pull():
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/rcc*120k_lr4_s*.jsonl' {LOGS}/ 2>/dev/null",
                   shell=True)
    subprocess.run(f"cp {ROOT}/runs/cluster_runs/clf120k_lr4_s2.jsonl {LOGS}/", shell=True)   # local GPU, task 183


def jl(f):
    return [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fig_channel_census import parse
    fs = apply_fig_style()
    L = fs["legend"] - 1
    census = parse()
    joint, naive = np.load(os.path.join(DATA, "cache", "bayes_bounds_ctx2_T30.npy"))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True, constrained_layout=True)
    xmax = 60000
    for ax, (title, arms) in zip(axes, PANELS):
        for (arm, ext), col, lab in zip(arms, (C_GEN, C_CLF), ("Generator route", "Classifier route")):
            curves, n_miss = [], 0
            for k in range(6):
                r = jl(os.path.join(LOGS, f"{ext}_s{k}.jsonl")) if ext else []
                r = r or jl(os.path.join(LOGS, f"{arm}_s{k}.jsonl"))
                if not r:
                    continue
                c = {q["ep"]: q["net"] for q in r}
                curves.append(c)
                d = census.get(f"{arm}_s{k}")
                miss = d is not None and bool((d["drop"] < .1).any())
                n_miss += miss
                xs = sorted(c)
                xmax = max(xmax, xs[-1])
                ax.plot(xs, [c[x] for x in xs], color=col, lw=.9, alpha=.55, ls=(0, (4, 2)) if miss else "-")
            common = sorted(set.intersection(*(set(c) for c in curves)))
            ax.plot(common, [np.mean([c[x] for c in curves]) for x in common], color=col, lw=3,
                    label=f"{lab}: mean of {len(curves)} seeds ({n_miss} miss a channel at 60k)")
            print(f"{title[:1]} {arm:14s} seeds {len(curves)}, miss {n_miss}, mean at {common[-1]}: "
                  f"{np.mean([c[common[-1]] for c in curves]):.3f}")
        ax.axhline(joint, color="k", ls="--", lw=1)
        ax.axhline(naive, color="k", ls=":", lw=1)
        ax.text(xmax * .99, joint + .012, "Joint Bayes", fontsize=L, ha="right")
        ax.text(xmax * .99, naive + .012, "Naive Bayes", fontsize=L, ha="right")
        ax.set(xlim=(0, xmax), ylim=(0, .82), xlabel="training iterations (10,000 episodes each)")
        ax.set_title(title, loc="left", fontsize=fs["title"] - 3)
        ax.legend(frameon=False, fontsize=L, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Classifier's goal accuracy, novel environments")
    fig.suptitle("Updated RCC, default Classifier; only the route of the embedding gradient differs "
                 "(lr 1e-4; thin = seeds, dashed = misses a channel at 60k; every point = 10,000 novel episodes)", fontsize=fs["legend"] - 1)
    out = fig_path("fig_route_control.png")
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
