"""Generation vs classification with everything else matched (John 11-11 23:00: "run the classifier with the shared channel
thing"; "the classifier gets stuck more often ... we need a solution").

trainings/train_bayes_generator.py, |S| 500, d_E 30, batch 10,000, lr 1e-3, reg_loss, the true values of both variables,
embeddings learned from scratch with the same start; 10,000 novel episodes per test. Learners:
    Generator, one shared Z layer (as built, aligned start)       --zin shared            (fig13_controls.py)
    Generator, one shared Z layer, default start                  --zin shared_random
    Generator, one Z layer per channel, independent starts        --zin per_channel_random
    classifier, one evidence function for every channel (Bayes)  --learner reader_shared
    classifier, one evidence function per channel                 --learner reader_unshared
    classifier, LSTM over the 30 steps (like the RCC's)           --learner reader_lstm
a: novel goal accuracy (Generator: exact Bayes with its likelihood; classifier: its own read-out); b: at the last test,
how well each channel's learned code copies the true one (linear R^2 of the best slot), one dot per channel and seed.

RESULT 11-12 00:10 (2 seeds each):
    Generator, shared            .65 at 750-1,000 batches, .70 at 3,000; all 5 channels R^2 .99, one frame
    Generator, shared, default   seed 1 as fast (.67 at 1,000), seed 0 never leaves chance
    Generator, per channel       stuck at .23 from 750 to 3,000 batches: 2 of 5 channels R^2 .99, 3 never learned
    classifier, shared           .21-.22 at 3,000-3,500 batches: all 5 channels together, one frame, but R^2 only ~.5
    classifier, per channel      .18-.21: channels one at a time (1 to 3 of 5 partly learned)
    classifier, LSTM             chance at 1,250-1,500 batches, no code
-> sharing one function across channels makes the channels learn together (no missing channels), for either objective;
   with the read-in matched, generation learns the codes far faster than classification.

    python fig_matched_learners.py [--pull]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
from main.utils import fig_path, apply_fig_style
from fig13_controls import analyse

EXP = os.path.join(ROOT, "main", "DATA", "experimental")
GEN_DIR = os.path.join(EXP, "fig13_embeddings", "controls")
CLF_DIR = os.path.join(EXP, "clf_scratch")
LEARNERS = (("Generator, one shared Z layer", GEN_DIR, "shared", "#08519c", "-"),
            ("Generator, shared Z layer, default start", GEN_DIR, "shared_random", "#6baed6", "-"),
            ("Generator, one Z layer per channel", GEN_DIR, "per_channel_random", "#6baed6", (0, (4, 2))),
            ("classifier, one function for every channel", CLF_DIR, "reader_shared", "#a50f15", "-"),
            ("classifier, one function per channel", CLF_DIR, "reader_unshared", "#fb6a4a", (0, (4, 2))),
            ("classifier, LSTM (like the RCC's)", CLF_DIR, "reader_lstm", "#fcae91", ":"))


def pull():
    for d, pat in ((GEN_DIR, "{shared,shared_random,per_channel_random}"), (CLF_DIR, "{reader_shared,reader_unshared,reader_lstm}")):
        os.makedirs(d, exist_ok=True)
        for ext in (".jsonl", "_codes.npz"):
            subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/bayes_gen/truth_S500_d30_s[01]_{pat}{ext}' {d}/ "
                           "2>/dev/null", shell=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    joint, naive = .768, .446                       # Joint / Naive Bayes on these tests (logged per test: novel_joint / _naive)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=(1.5, 1)), constrained_layout=True)
    js, ns = [], []
    for i, (lab, d, tag, col, ls) in enumerate(LEARNERS):
        r2s = []
        for k in (0, 1):
            f = os.path.join(d, f"truth_S500_d30_s{k}_{tag}.jsonl")
            if not os.path.exists(f):
                continue
            r = [json.loads(l) for l in open(f) if l.strip()]
            key = "novel_reader" if "novel_reader" in r[-1] else "novel_plugin"
            x = np.array([q["ep"] for q in r]); y = np.array([q[key] for q in r])
            js += [q["novel_joint"] for q in r]; ns += [q["novel_naive"] for q in r]
            a.plot(x[1:], y[1:], color=col, ls=ls, lw=1.8, label=lab if k == 0 else None)
            c = f[:-6] + "_codes.npz"
            if os.path.exists(c):
                D = np.load(c)
                _, r2, _ = analyse(D["z"], D["ztrue"])
                r2s += list(r2)
            print(f"{lab:45s} s{k}: last test {x[-1]}: {y[-1]:.3f}" + (f" | code R^2 {np.round(r2, 2).tolist()}" if os.path.exists(c) else ""))
        if r2s:
            b.scatter(np.clip(r2s, 0, 1), i + np.linspace(-.15, .15, len(r2s)), color=col, s=22, zorder=3)
    a.axhline(np.mean(js), color="k", ls="--", lw=1)
    a.axhline(np.mean(ns), color="k", ls=":", lw=1)
    a.text(26, np.mean(js) + .01, "Joint Bayes", fontsize=fs["legend"] - 3)
    a.text(26, np.mean(ns) + .01, "Naive Bayes", fontsize=fs["legend"] - 3)
    a.set(xscale="log", xlim=(25, 4200), ylim=(0, .8), xlabel="training batches (10,000 episodes each)",
          ylabel="goal accuracy, novel environments")
    a.legend(frameon=False, fontsize=fs["legend"] - 3, loc="upper left", bbox_to_anchor=(0, .95))
    a.set_title("a   Learning the embeddings from scratch", loc="left", fontsize=fs["title"] - 3)
    b.set_yticks(range(len(LEARNERS)))
    b.set_yticklabels([l[0].replace(", ", ",\n", 1) for l in LEARNERS], fontsize=fs["legend"] - 4)
    b.invert_yaxis()
    b.set(xlim=(-.03, 1.03), xlabel="each channel's code vs the true one (linear $R^2$)")
    b.set_title("b   Which channels were learned (last test)", loc="left", fontsize=fs["title"] - 3)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Generation vs classification with the same embeddings, start, batch, lr and labels ($|\\mathcal{S}| = 500$, "
                 "2 seeds each; b: 5 channels x 2 seeds; 10,000 novel episodes per test)", fontsize=fs["legend"] - 2)
    out = fig_path("fig_matched_learners.png")
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
