"""Why one d_E 20 grid cell took off and its twin did not (John 11-13 03:30: "why did the naive cell take off while the
joint didn't? ... are they running the same seeds?"). train_bayes_generator.py d_E 20 runs (Generator + embeddings
trained on an observer's answers, batch 10,000, lr 1e-3, 8,000 episodes), |S| 250 and 1000, Joint's and Naive's answers.
The twins share seed 0: identical network, world and test episodes (their first tests agree to every digit); only the
answers differ (right 70% / 40% of the time).
    a  Bayes with the learned likelihood on TRAINING environments (10,000 episodes per test)
    b  KL from the true likelihood on NOVEL environments, novel_kl_env (nats; down = learning the shared code)
RESULT 11-13 03:35: all four sit on the same plateau (KL 1.15-1.16) for the first ~500-1,000 episodes. Two leave it by
the shared code (KL falls to .1-.3 within ~500 episodes, accuracy jumps to .45-.64); the other two keep lowering their
training loss while the novel KL RISES (1.16 -> 1.40-1.54) and training accuracy creeps to .19-.24: they fit the training
variables one by one (memorization). Which road a run takes is set by its label stream, not by the label source: Joint
memorizes at |S| 1000, Naive at |S| 250. Seed 1 and the 20k continuations test how often each happens.
RESULT 11-13 12:45 (seed 1, train_bayes_generator.py TASKS 147-154; the two stuck runs continued to 20,000 episodes,
*_e20k): all four seed-1 runs learn the shared code (novel accuracy .50-.73, KL .02-.24), and so do all four at |S| 500
(seeds 0-1): 10 of 12 runs at |S| 250-1000 take off, the two failures are both seed 0, one per label source. Continued
2.5x longer, both stay stuck and keep memorizing (novel accuracy .10-.11, novel KL 1.44-1.69, training accuracy
.22-.25): memorization is a trap, not a delay. |S| 125: 0 of 4 (seeds 0-1) learn the shared code.
Thick = seed 0, thin = seed 1; x log-scaled (tests at 25, 50, 100, 150, 250, ... episodes).

    python fig_grid_d20_paths.py
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "bayes_gen", "jsonl")
RUNS = (("joint", 250, "#08519c", "-"), ("naive", 250, "#08519c", (0, (4, 2))),
        ("joint", 1000, "#a50f15", "-"), ("naive", 1000, "#a50f15", (0, (4, 2))))


def jl(name):
    f = os.path.join(LOGS, name + ".jsonl")
    return [json.loads(l) for l in open(f) if l.strip()][1:] if os.path.exists(f) else []


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    L = fs["legend"] - 1
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    for lab, S, col, ls in RUNS:
        for k, lw, al in ((0, 2.8, 1), (1, 1.3, .8)):
            r = jl(f"{lab}_S{S}_d20_s{k}")
            ext = jl(f"{lab}_S{S}_d20_s{k}_e20k")                        # continued to 20,000 (log starts with the source)
            r = ext if len(ext) > len(r) else r
            if not r:
                continue
            x = [q["ep"] for q in r]
            took = r[-1]["novel_plugin"] > .3
            print(f"{lab} S{S} s{k}: ep {x[-1]} train {r[-1]['train_plugin']:.3f} novel {r[-1]['novel_plugin']:.3f} "
                  f"novel KL {r[-1]['novel_kl_env']:.2f} -> {'shared code' if took else 'memorized'}")
            a.plot(x, [q["train_plugin"] for q in r], color=col, ls=ls, lw=lw, alpha=al)
            b.plot(x, [q["novel_kl_env"] for q in r], color=col, ls=ls, lw=lw, alpha=al)
            if not took:
                b.annotate(f"seed {k}, memorizing", (x[-1], r[-1]["novel_kl_env"]), xytext=(-6, -22 if S == 1000 else 8),
                           textcoords="offset points", ha="right", fontsize=L, color=col)
        a.plot([], [], color=col, ls=ls, lw=2.4,
               label=f"{'Joint' if lab == 'joint' else 'Naive'}'s answers, " + r"$|\mathcal{S}|$" + f" = {S}")
    a.plot([], [], color=".3", lw=2.8, label="seed 0")
    a.plot([], [], color=".3", lw=1.3, label="seed 1")
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set(xlabel="training episodes (batch 10,000)", xlim=(25, 20000))
        ax.spines[["top", "right"]].set_visible(False)
    a.set(ylabel="accuracy, training environments", ylim=(0, .8))
    a.set_title("a   Bayes with the learned likelihood (training environments)", loc="left", fontsize=fs["title"] - 4)
    b.set(ylabel="KL from the true likelihood (nats)", ylim=(0, 1.8))
    b.set_title("b   How far the learned likelihood is from the true one (novel environments)", loc="left",
                fontsize=fs["title"] - 4)
    a.legend(frameon=False, fontsize=L, loc="upper left")
    fig.suptitle(r"$d_E = 20$ twins (same seed: same network, world and test episodes; only the answers differ); "
                 "stuck runs continued to 20,000 episodes; 10,000 episodes per test", fontsize=fs["legend"])
    out = fig_path("fig_grid_d20_paths.png")
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
