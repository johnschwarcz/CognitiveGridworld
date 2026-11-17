"""Three latent variables per environment (C = 3): the RCC's Generator route vs Classifier route (John 11-12 12:00: "Id be
very curious about channel collapse being a much bigger issue for C = 3 so if we can compare the 2 routes with typical RCC
feedback and see the generator route succeed that would be great").

train_cluster_runs.py TASKS 193-200: Fig 13's protocol (|S| 500, batch 10,000, lr 1e-4, 60k iterations, default dense
read-in) with ctx_num = 3 (10 realizations each: 1,000 joint values; likelihood built on the GPU, bayes_likelihood.py
build_likelihood_gpu). Only the route of the embedding gradient differs between the arms.
    a  oracle labels: genC3_60k_lr4 / clfC3_60k_lr4 (local GPUs)
    b  the RCC's own answers as labels (gen_zflow): rccC3_60k_lr4 / rccclfC3_60k_lr4 (cluster)
Solid = the Classifier's goal accuracy on novel environments; thin dashed = Bayes with the Generator's learned likelihood
(the plug-in) on the same episodes. Every point = 10,000 novel episodes. Joint / Naive Bayes = the mean over the tests
drawn (each test scores both on its own episodes).

11-13 05:40: oracle runs stopped (John: "I'm focusing on the default feedback stuff"): clfC3 s0 at 24.8k (.34; the
Generator route s0 had .33-.34 at 25k), genC3 s1 at 31.8k (.42). b: seed 0 continued to 120k (rccC3_120k_lr4_s0 /
rccclfC3_120k_lr4_s0), seed 1 (cluster), seed 2 (local GPUs, tasks 234-235).
RESULT 11-13 12:40 (b): seed 0 at 120k: Generator route .461 (plug-in .509; Naive's .27 at 51.75k, .40 at 75k, slowing:
+.017 per 20k at the end), Classifier route .377 (Naive's level at 83.25k; still rising ~.03 per 10k; its Generator's
plug-in stays at chance, .13). The Generator route reaches the Classifier route's 120k level at ~70k (1.7x sooner).
Generator seed 1 tracks seed 0 (.322 at 60k vs .317). Seed 2 (Generator) and seed 1 (Classifier) run on the local GPUs.
11-13 17:32 IST C = 3 DROPPED (John: "okay yea let's cancel the C = 3 seed"): Classifier-route seed 1 killed at 38.75k (.152,
plug-in .111; seed 0 .133 there, left chance ~40k), seed 2 stopped at its start, the Generator route's seed 2 at 20.5k; no
120k continuations for seeds 1-2. Final data: seed 0 to 120k (both routes), seed 1 Generator route to 60k.
RESULT 11-12 22:50 (seed 0): a oracle, Generator route: the plug-in reaches .51 by 2k and .60 by 60k, the Classifier
climbs slowly (.24 at 10k, .38 at 30k, .49 at 60k, still rising); the oracle Classifier route had just started (2.6k).
b RCC feedback: Generator route at chance to ~24k, then .19 at 30k, .27 at 51k (Naive .27; plug-in .33), still rising;
Classifier route at chance to 40k, .16 at 49k, plug-in at chance (.12).

    python fig_c3_routes.py [--pull]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "cluster_runs", "jsonl")
C_GEN, C_CLF = "#08519c", "#a50f15"
PANELS = (("a   Oracle labels", "genC3_60k_lr4", "clfC3_60k_lr4"),
          ("b   The RCC's own answers as labels", "rccC3_60k_lr4", "rccclfC3_60k_lr4"))


def pull():
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/rcc*C3_*.jsonl' {LOGS}/", shell=True)
    subprocess.run(f"cp {ROOT}/runs/cluster_runs/{{gen,clf,rcc,rccclf}}C3_*_s*.jsonl {LOGS}/ 2>/dev/null", shell=True,
                   executable="/bin/bash")                                  # local GPUs; brace expansion needs bash


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
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2), constrained_layout=True, gridspec_kw=dict(width_ratios=(1, 1.6, 1)))
    axes[1].sharey(axes[0])
    for ax, (title, gen, clf), xmax in zip(axes, PANELS, (60000, 120000)):
        joints, naives, done = [], [], []
        for arm, col in ((gen, C_GEN), (clf, C_CLF)):
            for k in range(3):
                r = jl(f"{arm}_s{k}")
                ext = jl(f"{arm.replace('60k', '120k')}_s{k}")      # continued from its 60k net (RESUME; log starts with it)
                r = ext if len(ext) > len(r) else r
                if len(r) < 5:                                    # skip runs stopped at their start (rccclfC3 s2, 11-13)
                    continue
                x = [q["ep"] for q in r]
                ax.plot(x, [q["net"] for q in r], color=col, lw=2.2, alpha=(1, .7, .5)[k])
                ax.annotate(f"s{k}", (x[-1], r[-1]["net"]), xytext=(4, -4 if arm == clf else 4), textcoords="offset points",
                            fontsize=L, color=col, fontweight="bold", bbox=dict(boxstyle="round,pad=.15", fc="w", ec="none", alpha=.85))
                ax.plot(x, [q["plugin"] for q in r], color=col, lw=1.1, ls=(0, (3, 2)), alpha=.8)
                joints += [q["joint"] for q in r]; naives += [q["naive"] for q in r]
                done.append(f"{'Gen' if arm == gen else 'Clf'} s{k} {r[-1]['ep'] / 1000:.0f}k")
                print(f"{title[:1]} {arm}_s{k}: ep {r[-1]['ep']} net {r[-1]['net']:.3f} plugin {r[-1]['plugin']:.3f}")
        joint, naive = np.mean(joints), np.mean(naives)
        ax.axhline(joint, color="k", ls="--", lw=1)
        ax.axhline(naive, color="k", ls=":", lw=1)
        ax.axhline(.1, color=".6", ls=":", lw=1)
        for y, t in ((joint, "Joint Bayes"), (naive, "Naive Bayes"), (.1, "chance")):
            ax.text(xmax * .99, y + .01, t, fontsize=L, ha="right", color="k" if t != "chance" else ".4")
        ax.set(xlim=(0, xmax), ylim=(0, .72), xlabel="training iterations (10,000 episodes each)")
        ax.set_xticks(range(0, xmax + 1, 20000), [f"{v // 1000}k" for v in range(0, xmax + 1, 20000)])
        ax.set_title(f"{title}\n" + ", ".join(done), loc="left", fontsize=fs["title"] - 5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("goal accuracy, novel environments")
    ax = axes[0]
    ax.plot([], [], color=C_GEN, lw=2.2, label="Generator route")
    ax.plot([], [], color=C_CLF, lw=2.2, label="Classifier route")
    # c  zoom on the Classifier route's start: is a seed failing where seed 0 left chance (~40-45k)?
    zc = axes[2]
    _, gen, clf = PANELS[1]
    for arm, col in ((gen, C_GEN), (clf, C_CLF)):
        for k in range(2):
            r = jl(f"{arm}_s{k}")
            ext = jl(f"{arm.replace('60k', '120k')}_s{k}")
            r = [q for q in (ext if len(ext) > len(r) else r) if q["ep"] <= 60000]
            if len(r) < 2:
                continue
            zc.plot([q["ep"] for q in r], [q["net"] for q in r], color=col, lw=(2.2, 1.3)[k], alpha=(1, .75)[k])
            if arm == clf:                                        # the Generator route (blue) is the reference, labelled in b
                zc.annotate(f"s{k}", (r[-1]["ep"], r[-1]["net"]), xytext=(4, 0), textcoords="offset points",
                            fontsize=L, color=col, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=.15", fc="w", ec="none", alpha=.85))
    c1 = jl(f"{clf}_s1")
    c0 = {q["ep"]: q["net"] for q in jl(f"{clf.replace('60k', '120k')}_s0")}
    if c1:
        e1 = c1[-1]["ep"]
        zc.set_title(f"c   Classifier route, the start (zoom of b)\nseed 1 at {e1 / 1000:g}k: {c1[-1]['net']:.3f};"
                     f" seed 0 there: {c0.get(e1, float('nan')):.3f}", loc="left", fontsize=fs["title"] - 5)
    zc.axhline(.1, color=".6", ls=":", lw=1)
    zc.set(xlim=(0, 60000), ylim=(.09, .25), xlabel="training iterations (10,000 episodes each)",
           ylabel="goal accuracy, novel environments")
    zc.set_xticks(range(0, 60001, 20000), [f"{v // 1000}k" for v in range(0, 60001, 20000)])
    zc.spines[["top", "right"]].set_visible(False)
    ax.plot([], [], color="k", lw=2.2, label="the Classifier (the RCC's answer)")
    ax.plot([], [], color="k", lw=1.1, ls=(0, (3, 2)), label="Bayes with the Generator's learned likelihood")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=L, loc="outside lower center", ncol=4)
    fig.suptitle(r"Three latent variables per environment (C = 3, 1,000 joint values); $|\mathcal{S}|$ = 500, lr 1e-4; "
                 "every point = 10,000 novel episodes", fontsize=fs["legend"])
    out = fig_path("fig_c3_routes.png")
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
