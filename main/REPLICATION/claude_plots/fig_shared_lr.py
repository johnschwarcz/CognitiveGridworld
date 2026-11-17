"""The Classifier route with the shared read-in, and the learning rate (John 11-12 03:40: "What about the classifier route
with the shared readin? Have we seen if this one works?"; "My guess is that high lr makes lost channels to be likely. So if
the generator can handle high lr and the classifier route cant handle a shared readin that is a direction"). Redrawn 11-12
04:00 as three like-for-like panels (John: "I dont understand the figure, why is the left classifier failing but the right
succeeding?": the two were different classifiers at different learning rates).

a  The RCC, Fig 13's protocol (train_cluster_runs.py: oracle feedback, |S| 500, batch 10,000, lr 1e-4; 10,000 novel
   episodes per test): novel goal accuracy of the LSTM Classifier in the Generator route (gen60k_lr4, 6 seeds), in the
   Classifier route (clf60k_lr4, 6 seeds) and in the Classifier route with the shared read-in (clfshared60k_lr4,
   main/model/shared_readin.py: each step, one layer reads every channel's (bit, Z) and the results are averaged over
   channels; the LSTM must still integrate the 30 steps; 4 seeds). Thin = seeds, thick = mean at the tests every seed has.
b  No LSTM (train_bayes_generator.py, |S| 500, d_E 30, batch 10,000, embeddings from scratch, the same start, labels and
   trainer; 10,000 novel episodes per test), lr 1e-3, 4 seeds each: the Generator (one shared Z layer, as built) read out by
   exact Bayes, and a classifier handed Bayes' form (--learner reader_shared, transfer_reader.readers() Evidence: each
   channel's [Z, bit frequency over the 30 steps] -> one shared MLP 3-128-128-100 -> evidence for the 100 joint values,
   summed over channels) read out by its soft-max.
c  As b at lr 3e-3 and 1e-2 (seed 0, cluster job 66672305).

RESULT 11-12 03:50
a  the shared read-in Classifier route is at chance in all 4 seeds (.11-.12 from 1k to 11.5-13k iterations); the dense
   Classifier route had left chance by 3-7k iterations in 6 of 6 seeds (.43-.54 at 9k), the Generator route's Classifier
   by 3k in 6 of 6 (.62-.64 at 9k). Not yet known: whether the shared read-in Classifier can read GOOD codes (the
   Generator route with the shared read-in, tasks 159-168, from 11-12 04:00).
b  the Generator .65 at 750-1,000 batches, .70 by 4,000 (4 of 4); the Bayes-form classifier .22-.23 at 4,000 (4 of 4), and
   at 15,000 one seed .55, one still at chance (.12). The b classifier has an easier job than a's (no integration over
   steps to learn, Bayes' form built in) and a 10x larger lr, so a's failure within 13k iterations is not a contradiction.
c  the Generator is stuck at lr 3e-3 and 1e-2: its loss jumps to 11.4 within 100 batches and stays there, identical at
   both rates, at chance through 1,500 batches (its predicted probabilities saturate, and the clipped loss then passes no
   gradient). The Bayes-form classifier learns faster at 3e-3 (.35 at 3,000 batches) than at 1e-3, slower at 1e-2 (.16).
   So here the Generator does NOT tolerate high learning rates better.

    python fig_shared_lr.py [--pull]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

DATA = os.path.join(ROOT, "main", "DATA")
ARM_LOGS = os.path.join(DATA, "experimental", "cluster_runs", "jsonl")
PROBE = os.path.join(DATA, "experimental", "clf_scratch", "lr_probe")
C_GEN, C_CLF, C_SHARED = "#08519c", "#a50f15", "#e6550d"
ARMS = (("gen60k_lr4", "Generator route", C_GEN),
        ("clf60k_lr4", "Classifier route", C_CLF),
        ("clfshared60k_lr4", "Classifier route, shared read-in", C_SHARED))


def pull():
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/clfshared60k_lr4_s*.jsonl' {ARM_LOGS}/",
                   shell=True)
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/bayes_gen/truth_S500_d30_s0_lr[23]*.jsonl' "
                   f"'elsc-cluster:cgw/runs/bayes_gen/truth_S500_d30_s[0-3]_shared.jsonl' "
                   f"'elsc-cluster:cgw/runs/bayes_gen/truth_S500_d30_s[0-3]_reader_shared.jsonl' {PROBE}/", shell=True)


def jl(f):
    return [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []


def probe(k, lr, learner):
    """train_bayes_generator.py names: truth_S500_d30_s<k>[_lr<-round(log10 lr)>][_<zin or learner>]; the lr 1e-3
    Generator runs carry _shared (--zin shared), the lr-probe Generator runs no zin suffix (shared is the default)"""
    f = f"truth_S500_d30_s{k}{lr}" + (("" if lr else "_shared") if learner == "gen" else "_reader_shared") + ".jsonl"
    r = jl(os.path.join(PROBE, f))
    key = "novel_plugin" if learner == "gen" else "novel_reader"
    return np.array([q["ep"] for q in r][1:]), np.array([q[key] for q in r][1:])


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
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=(1.25, 1, .8)),
                                  constrained_layout=True)

    for arm, lab, col in ARMS:
        runs = [r for r in (jl(os.path.join(ARM_LOGS, f"{arm}_s{k}.jsonl")) for k in range(6)) if r]
        if not runs:
            continue
        curves = [{q["ep"]: q["net"] for q in r} for r in runs]
        for cv in curves:
            xs = [x for x in sorted(cv) if x <= 20000]
            a.plot(xs, [cv[x] for x in xs], color=col, lw=.6, alpha=.4)
        common = sorted(x for x in set.intersection(*(set(cv) for cv in curves)) if x <= 20000)
        a.plot(common, [np.mean([cv[x] for cv in curves]) for x in common], color=col, lw=2.6,
               label=f"{lab} ({len(runs)} seeds)")
        print(f"a {arm:18s} seeds {len(runs)}, last common test {common[-1]}: "
              f"mean {np.mean([cv[common[-1]] for cv in curves]):.3f}")
    a.set(xlim=(0, 20000), ylim=(0, .8), xlabel="training iterations (10,000 episodes each)",
          ylabel="goal accuracy, novel environments")
    a.set_title("a   RCC: LSTM Classifier, lr 1e-4", loc="left", fontsize=fs["title"] - 3)
    a.legend(frameon=False, fontsize=L, loc="upper left", bbox_to_anchor=(0, .9))

    for k in range(4):
        for learner, lab, col in (("gen", "Generator + Bayes", C_GEN), ("clf", "Bayes-form classifier", C_CLF)):
            x, y = probe(k, "", learner)
            if len(x):
                b.plot(x, y, color=col, lw=1.6, label=f"{lab} (4 seeds)" if k == 0 else None)
                print(f"b {lab:22s} s{k}: last test {x[-1]}: {y[-1]:.3f}")
    b.set(xscale="log", xlim=(25, 15000), ylim=(0, .8), xlabel="training batches (10,000 episodes each)")
    b.set_title("b   No LSTM, lr 1e-3", loc="left", fontsize=fs["title"] - 3)
    b.legend(frameon=False, fontsize=L, loc="upper left", bbox_to_anchor=(0, .9))

    for lr, lrlab, ls in (("_lr3", "3e-3", (0, (5, 2))), ("_lr2", "1e-2", (0, (1, 1.5)))):
        for learner, lab, col in (("gen", "Generator + Bayes", C_GEN), ("clf", "Bayes-form classifier", C_CLF)):
            x, y = probe(0, lr, learner)
            if not len(x) or (learner == "gen" and lr == "_lr2"):       # identical to the lr 3e-3 curve: drawn once
                continue
            tag = "lr 3e-3 and 1e-2" if learner == "gen" else f"lr {lrlab}"
            c.plot(x, y, color=col, ls=ls, lw=2.2, label=f"{lab}, {tag}")
            print(f"c {lab:22s} lr {lrlab}: last test {x[-1]}: {y[-1]:.3f}")
    c.set(xscale="log", xlim=(25, 4000), ylim=(0, .8), xlabel="training batches (10,000 episodes each)")
    c.set_title("c   No LSTM, higher lr (seed 0)", loc="left", fontsize=fs["title"] - 3)
    c.legend(frameon=False, fontsize=L, loc="upper left", bbox_to_anchor=(0, .9))
    for ax in (a, b, c):
        ax.axhline(joint, color="k", ls="--", lw=1)
        ax.axhline(naive, color="k", ls=":", lw=1)
        ax.spines[["top", "right"]].set_visible(False)
    a.text(300, joint + .012, "Joint Bayes", fontsize=L)
    a.text(300, naive + .012, "Naive Bayes", fontsize=L)
    fig.suptitle("$|\\mathcal{S}| = 500$ training environments; every point = 10,000 novel episodes",
                 fontsize=fs["legend"] - 1)
    out = fig_path("fig_shared_lr.png")
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
