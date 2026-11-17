"""The same Classifier, two sources of codes (John 11-12 04:20: "If the classifier cant learn the proper shared readin channel
structure while the generator can ... that could be a way to argue"). The shared read-in (main/model/shared_readin.py: each
step, one layer reads every channel's (bit, Z) and the results are averaged over channels, then the LSTM) is the same in
both arms; only the route of the embedding gradient differs: the Generator route (genshared*: the codes are learned by the
Generator, the Classifier reads them) or the Classifier route (clfshared*: the Classifier must learn the codes itself). Dashed:
the same two routes with the default dense read-in (Fig 13's arms). train_cluster_runs.py, oracle feedback, |S| 500, batch
10,000; novel goal accuracy of the LSTM Classifier, 10,000 novel episodes per test; thin = seeds, thick = mean at the tests
every seed has.

a  lr 1e-4 (Fig 13's): genshared20k_lr4 (2 seeds), clfshared60k_lr4 (4 seeds); gen60k_lr4 / clf60k_lr4 (6 seeds each).
b  lr 1e-3: genshared10k_lr1e3 (2 seeds), clfshared10k_lr1e3 (2 seeds); gen30k / clf30k (4 seeds each).

RESULT 11-12 04:40 (a): with the Generator's codes the shared read-in Classifier is at .38-.39 by 1,750 iterations (the
dense one at .26-.27), while learning its own codes it is at chance (.11-.12) through 17,750 iterations in 4 of 4 seeds.
At lr 1e-3 the Classifier route with the shared read-in is at chance through 3,750 iterations (seed 0; the dense one .35).
So the shared read-in is a working reader: it fails only when classification has to learn the codes through it.

    python fig_shared_control.py [--pull]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

DATA = os.path.join(ROOT, "main", "DATA")
LOGS = os.path.join(DATA, "experimental", "cluster_runs", "jsonl")
C_GEN, C_CLF = "#08519c", "#e6550d"
PANELS = (("a   lr 1e-4 (Fig 13's)", 20000,
           (("genshared20k_lr4", "Generator route, shared read-in", C_GEN, "-", 2.6),
            ("clfshared60k_lr4", "Classifier route, shared read-in", C_CLF, "-", 2.6),
            ("gen60k_lr4", "Generator route, dense read-in", C_GEN, (0, (4, 2)), 1.4),
            ("clf60k_lr4", "Classifier route, dense read-in", "#a50f15", (0, (4, 2)), 1.4))),
          ("b   lr 1e-3", 10000,
           (("genshared10k_lr1e3", "Generator route, shared read-in", C_GEN, "-", 2.6),
            ("clfshared10k_lr1e3", "Classifier route, shared read-in", C_CLF, "-", 2.6),
            ("gen30k", "Generator route, dense read-in", C_GEN, (0, (4, 2)), 1.4),
            ("clf30k", "Classifier route, dense read-in", "#a50f15", (0, (4, 2)), 1.4))))


def pull():
    for pat in ("genshared*", "clfshared*"):
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/{pat}.jsonl' {LOGS}/ 2>/dev/null", shell=True)
        subprocess.run(f"cp {ROOT}/runs/cluster_runs/{pat}.jsonl {LOGS}/ 2>/dev/null", shell=True)        # local GPU runs


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
    fs = apply_fig_style()
    L = fs["legend"] - 3
    joint, naive = np.load(os.path.join(DATA, "cache", "bayes_bounds_ctx2_T30.npy"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for ax, (title, xmax, arms) in zip(axes, PANELS):
        for arm, lab, col, ls, lw in arms:
            runs = [r for r in (jl(os.path.join(LOGS, f"{arm}_s{k}.jsonl")) for k in range(6)) if r]
            if not runs:
                continue
            curves = [{q["ep"]: q["net"] for q in r if q["ep"] <= xmax} for r in runs]
            for c in curves:
                xs = sorted(c)
                ax.plot(xs, [c[x] for x in xs], color=col, ls=ls, lw=.6, alpha=.35)
            common = sorted(set.intersection(*(set(c) for c in curves)))
            ax.plot(common, [np.mean([c[x] for c in curves]) for x in common], color=col, ls=ls, lw=lw,
                    label=f"{lab} ({len(runs)} seed{'s' if len(runs) > 1 else ''})")
            print(f"{title[:1]} {arm:20s} seeds {len(runs)}: last common test {common[-1]}: "
                  f"{np.mean([c[common[-1]] for c in curves]):.3f}")
        ax.axhline(joint, color="k", ls="--", lw=1)
        ax.axhline(naive, color="k", ls=":", lw=1)
        ax.text(xmax * .015, joint + .012, "Joint Bayes", fontsize=L)
        ax.text(xmax * .015, naive + .012, "Naive Bayes", fontsize=L)
        ax.set(xlim=(0, xmax), ylim=(0, .8), xlabel="training iterations (10,000 episodes each)")
        ax.set_title(title, loc="left", fontsize=fs["title"] - 3)
        ax.legend(frameon=False, fontsize=L, loc="upper left", bbox_to_anchor=(0, .9))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Classifier's goal accuracy, novel environments")
    fig.suptitle("The same Classifier, with its codes learned by the Generator or by itself ($|\\mathcal{S}| = 500$, oracle "
                 "feedback; every point = 10,000 novel episodes)", fontsize=fs["legend"] - 1)
    out = fig_path("fig_shared_control.png")
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
