"""Order invariance imposed on the Classifier: by its architecture (shared read-in) or by its training data (each episode's
channels in a random order) (John 11-12 11:40: "the channel swap needs to hold up to a classifier route shared readin
condition as well").

Fig 13's RCC (train_cluster_runs.py; |S| 500, batch 10,000, oracle labels), only the Classifier's read-in changed:
    shared read-in  (readin_shared, main/model/shared_readin.py): mean over channels of relu(W[bit, Z_o] + b), then the
                    LSTM; a swap of channels (bits and codes together) cannot change its output
    shuffled order  (readin_perm, model_forward.RNN_readin): the default dense read-in, but every training and test episode
                    shows its channels in a new random order (bits and codes moved together)
    default         (John 11-12 12:10: "it looks like the shuffled channels rescue the classifier?") the unconstrained dense
                    read-in, Fig 13's arms: gen60k_lr4 / clf60k_lr4 s0-5 (lr 1e-4), gen30k / clf30k s0-3 (lr 1e-3)
Columns: default / shuffled order / shared read-in; rows: lr 1e-4 (x to 20k: the shuffled and the Generator-route shared
runs end there; clfshared60k_lr4 s0-3 run to 60k, at chance throughout) and lr 1e-3 (x to 10k; genshared10k_lr1e3 /
clfshared10k_lr1e3, genperm10k_lr1e3 / clfperm10k_lr1e3, s0-1). Titles count the runs above Naive Bayes on novel
environments at the panel's last test. Blue = Generator route, red =
Classifier route; one line per seed; solid = novel environments, dashed = the training environments (the last training
batch before each test). Every test = 10,000 novel episodes.

RESULT 11-12 12:15 (with the default column): by 20k at lr 1e-4 the default dense read-in Classifier route beats Naive in
6/6, shuffled order 1/2, shared 0/4 (seed 0 learns with the default read-in, not with shuffled order); lr 1e-3: 4/4, 1/2,
0/2. Order invariance costs the Classifier route; the Generator route is 6/6, 2/2, 2/2 and 4/4, 2/2, 2/2.
RESULT 11-12 12:00: shared read-in: Generator route 4/4 learn (.65 at 20k, lr 1e-4; .69-.70 at 10k, lr 1e-3); Classifier
route 6/6 at chance on novel environments (.11-.12 to 54k at lr 1e-4; .12 at 10k at lr 1e-3) while its training
environments reach .22-.31 (memorization). Shuffled order: Generator route 4/4 (.68-.69); Classifier route 2/4: seed 1 learns
(.66-.67, beats Naive at 7k / 2.75k vs the Generator route's 3.5k / 1.5k), seed 0 stays at chance at both lr.

    python fig_order_invariant.py [--pull]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

DATA = os.path.join(ROOT, "main", "DATA")
LOGS = os.path.join(DATA, "experimental", "cluster_runs", "jsonl")
C_GEN, C_CLF = "#08519c", "#a50f15"
PANELS = ((("a   Default dense read-in, lr 1e-4", "gen60k_lr4", 6, "clf60k_lr4", 6, 20000),
           ("b   Shuffled channel order, lr 1e-4", "genperm20k_lr4", 2, "clfperm20k_lr4", 2, 20000),
           ("c   Shared read-in, lr 1e-4", "genshared20k_lr4", 2, "clfshared60k_lr4", 4, 20000)),
          (("d   Default dense read-in, lr 1e-3", "gen30k", 4, "clf30k", 4, 10000),
           ("e   Shuffled channel order, lr 1e-3", "genperm10k_lr1e3", 2, "clfperm10k_lr1e3", 2, 10000),
           ("f   Shared read-in, lr 1e-3", "genshared10k_lr1e3", 2, "clfshared10k_lr1e3", 2, 10000)))


def pull():
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/{{genshared,clfshared,genperm,clfperm}}*.jsonl'"
                   f" {LOGS}/", shell=True)


def jl(n):
    f = os.path.join(LOGS, n + ".jsonl")
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
    L = fs["legend"] - 1
    joint, naive = np.load(os.path.join(DATA, "cache", "bayes_bounds_ctx2_T30.npy"))
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.6), constrained_layout=True)
    for row, prow in zip(axes, PANELS):
        for ax, (title, gen, ng, clf, nc, xmax) in zip(row, prow):
            res = []
            for arm, n, col in ((gen, ng, C_GEN), (clf, nc, C_CLF)):
                ok = 0
                for k in range(n):
                    r = [q for q in jl(f"{arm}_s{k}") if q["ep"] <= xmax]
                    if not r:
                        continue
                    x = [q["ep"] for q in r]
                    ax.plot(x, [q["net"] for q in r], color=col, lw=1.8)
                    xt = [q["ep"] for q in r if "train_net" in q]
                    ax.plot(xt, [q["train_net"] for q in r if "train_net" in q], color=col, lw=1.1, ls=(0, (3, 2)), alpha=.8)
                    ok += r[-1]["net"] > naive
                    print(f"{title[:1]} {arm}_s{k}: ep {r[-1]['ep']} novel {r[-1]['net']:.3f} train {r[-1].get('train_net', np.nan):.3f}")
                res.append(f"{ok} of {n}")
            ax.axhline(joint, color="k", ls="--", lw=1)
            ax.axhline(naive, color="k", ls=":", lw=1)
            ax.text(xmax * .99, joint + .012, "Joint Bayes", fontsize=L, ha="right")
            ax.text(xmax * .99, naive + .012, "Naive Bayes", fontsize=L, ha="right")
            ax.set(xlim=(0, xmax), ylim=(0, .82), xlabel="training iterations (10,000 episodes each)")
            ax.set_title(f"{title}\nbeat Naive Bayes: Generator route {res[0]}, Classifier route {res[1]}", loc="left",
                         fontsize=fs["title"] - 4)
            ax.spines[["top", "right"]].set_visible(False)
        row[0].set_ylabel("Classifier's goal accuracy")
    ax = axes[0, 0]
    ax.plot([], [], color=C_GEN, lw=1.8, label="Generator route (one line per seed)")
    ax.plot([], [], color=C_CLF, lw=1.8, label="Classifier route")
    ax.plot([], [], color="k", lw=1.8, label="novel environments")
    ax.plot([], [], color="k", lw=1.1, ls=(0, (3, 2)), label="its training environments")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=L, loc="outside lower center", ncol=4)
    fig.suptitle("The Classifier with its default read-in (a, d), made order-invariant by shuffled training orders (b, e) "
                 "or by a shared read-in (c, f); oracle labels; every point = 10,000 novel episodes", fontsize=fs["legend"] - 1)
    out = fig_path("fig_order_invariant.png")
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
