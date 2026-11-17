"""Does online dreaming speed up the Generator route? (John 11-14 01:45: "Can you tell if there is a significant improvement
from generator dreaming at best? If not we may as well kill the dreamers.")

STATUS 11-14 02:04: dreaming STOPPED (net gain ~15-50% of compute, see "cost" below); seed 0 ends at 90,750 iterations.

Runs (train_cluster_runs.py; batch 1,000, lr 1e-4, the RCC's own feedback, gen_zflow, Generator route; one test of 1,000
novel-environment episodes every 250 iterations):
    dreaming   rccdreamb1k1m_lr4_s{0,1,2} (tasks 376/379/382, cluster packs 66686727): from the first test whose train_plugin
               (the Generator's world model, read out by Bayes, on the last training batch) is >= .3, the Classifier also
               learns every iteration from one batch its own Generator dreams (both variables labelled by the dreamer).
    no dreams  rccb1k_lr4_s0-4 and their continuations (fig_batch_routes.PANELS[1]); seed 5 has not left chance.
Alignment: x = iterations since that run's gate test (first test with train_plugin >= .3; for the dreaming run this is the
test before its dream_gate_ep). Same seeds are NOT twins across machines (the dreaming runs use the cluster's A100s, the
no-dream runs the local 3090 Tis; same code and libraries, different GPU arithmetic), so the no-dream runs are a reference
distribution, not paired controls. Checked 11-14 01:50: the dreaming arm's exact config (current code) rerun on a local 3090 Ti
reproduces the no-dream twin rccb1k_lr4_s0 bit for bit (all 43-47 logged numbers at tests 0, 250, 500); the cluster run
(A100) differs from test 250 on (generator loss 1.820 vs 1.683, net .108 vs .119); both machines run torch 2.5.1+cu121,
cuDNN 9.1.0, numpy 1.26.4, scipy 1.15.1. Two runs on ONE GPU stay identical (rccclfpermb1k1m vs rccclfpermdreamb1k1m, 333
tests before the latter's gate). So the seeds fix every random draw but not the hardware's arithmetic: the RCC samples its
answers, a last-digit difference flips a sample, and the runs part.
    a  the Classifier's goal accuracy at T (what the network answers)
    b  the Generator's plug-in accuracy (its learned world model + exact inference): what the dreams can teach
    c  b minus a: what the Generator knows that the Classifier does not use yet

NUMBERS (11-14 01:50 / 02:05, dreaming seed 0 only):
    gate -> past Naive (3 consecutive tests above that test's Naive)   dreaming 8,500 iterations; no dreams 18,500-27,250 (5/5)
    Classifier at gate + 10k / 20k                                      dreaming .481 / .538; no dreams .327-.368 / .420-.436
    plug-in at gate + 10k / 20k                                         dreaming .522 / .561; no dreams .441-.532 / .507-.539
    first iteration after the gate with the 5-test running mean of the Classifier >= .40 / .45 / .50 / .55:
        dreaming 5,750 / 8,000 / 13,250 / 21,500; no dreams 12,000-17,250 / 18,000-24,750 / 30,750-34,500 / 43,250-47,000
        (time saved grows, ~10k -> 24k iterations; the speed-up ratio shrinks, ~2.8x -> 2.2x vs the no-dream median)
    plug-in minus Classifier (c), 5-test means at gate + 10k / 20k / 30k: dreaming +.041 / +.023 / +.010; no dreams
        +.09-.16 / +.07-.11 / +.06-.11, +.04-.05 at +60k; batch-10,000 Generator route at 60k (no dreams): -.005 to -.009
    cost (11-14 02:00, seed-0 pack on one A100): one Classifier step per iteration either way (the dream gradient is added
        before the single step, model_backward.update), and the same 1,000 real episodes per iteration, so iterations saved
        = real episodes saved = weight updates saved. Compute is not: the dreaming run went 14.45 -> 9.18 it/s when dreams
        switched on while its two pack-mates went 14.44 -> 13.46 (total 43.3 -> 36.1 it/s), i.e. a dreaming iteration costs
        ~1.5x (fair-share estimate 13.46/9.18) to ~1.8x (constant-total estimate). Post-gate compute to reach .45 / .50 / .55
        in no-dream iterations: dreaming 12-14k / 20-24k / 32-39k vs median 23k / 34k / 46k without dreams.
So the dreams do not make the Generator learn faster (b, inside the no-dream range); they make the Classifier catch up with
what its Generator already knows (c), then it rises only as fast as the Generator does. That ceiling caps the no-dream
Classifier too (it ends just below its plug-in at batch 10,000), so dreams buy time, not a higher final accuracy.

    python fig_dream_effect.py [--pull]
"""
import os, sys, json, time, argparse, subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_batch_routes as fb
from main.utils import fig_path

J = os.path.join(fb.ROOT, "main", "DATA", "experimental", "cluster_runs", "jsonl")
DARK, LIGHT = "#08306b", "#9ecae1"
GATE = .3
STOPPED = {0: 90750}      # 11-14 02:04 (John: dream compute -> shuffled channels): seed 0 stopped by STOP file, net saved;
                          # the queued seed 1-2 packs (66686727_[1-2]) were cancelled, so seed 0 is the only dreaming run


def rows(name):
    f = os.path.join(J, name + ".jsonl")
    if not os.path.exists(f):
        return []
    return [r for r in (json.loads(l.replace("NaN", "null")) for l in open(f) if l.strip()) if r["ep"] > 0]


def series(rs):
    e = np.array([r["ep"] for r in rs], float)
    tp = np.array([r.get("train_plugin") if r.get("train_plugin") is not None else np.nan for r in rs])
    i = np.flatnonzero(tp >= GATE)
    if not len(i):
        return None
    g = e[i[0]]
    net = np.array([r["net"] for r in rs])
    pl = np.array([r.get("plugin") if r.get("plugin") is not None else np.nan for r in rs])
    nv = np.array([r["naive"] for r in rs])
    run, past = 0, None
    for k in range(len(e)):
        run = run + 1 if net[k] > nv[k] else 0
        if run == 3 and past is None:
            past = e[k] - g
    return e - g, net, pl, nv, past


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/rccdreamb1k1m_lr4_s*.jsonl' {J}/", shell=True,
                       stderr=subprocess.DEVNULL)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.serif": "cmr10", "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7.5, "axes.linewidth": .6})
    _, _, gen, _, _, _ = fb.PANELS[1]
    nodream = []
    for k in range(12):
        rs = fb.load(*gen, k)
        s = series(rs) if rs else None
        if s is not None and s[4] is not None:            # left chance and passed Naive
            nodream.append((k, s))
    dream, last = [], {}
    for k in range(3):
        rs = rows(f"rccdreamb1k1m_lr4_s{k}")
        s = series(rs) if rs else None
        if s is not None:
            dream.append((k, s))
            last[k] = rs[-1]["ep"]
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.0))
    nv_all = []
    for ax, j, title in ((axs[0], 0, r"$\mathbf{a}$   Classifier: goal accuracy at $T$"),
                         (axs[1], 1, r"$\mathbf{b}$   Generator: plug-in accuracy (its world model)"),
                         (axs[2], 2, r"$\mathbf{c}$   gap: plug-in $-$ Classifier")):
        for grp, colr, lw in ((nodream, LIGHT, .8), (dream, DARK, 1.3)):
            for k, (x, net, pl, nv, past) in grp:
                y = (net, pl, pl - net)[j]
                ax.plot(x, y, color=colr, lw=lw, zorder=4 if grp is dream else 2)
                if grp is dream and last[k] < 999_999 and k not in STOPPED:     # still training (runs go to 1M)
                    ax.plot(x[-1], y[-1], "o", ms=4, mfc=DARK, mec="k", mew=.8, zorder=5)
                nv_all += list(nv)
        ax.axvline(0, color=".5", lw=.6, ls="-")
        ax.set(xlim=(-10000, 62000), xlabel=r"iterations since plug-in $\geq$ .3 (dreams on)")
        ax.set_xticks([-10000, 0, 10000, 20000, 30000, 40000, 50000, 60000],
                      ["-10k", "0", "10k", "20k", "30k", "40k", "50k", "60k"])
        ax.set_title(title, fontsize=8, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axs[:2]:
        ax.set_ylim(.08, .68)
        ax.axhline(np.mean(nv_all), color="k", ls=":", lw=.9)
    axs[2].set_ylim(-.06, .24)
    axs[2].axhline(0, color="k", lw=.6)
    axs[2].set_ylabel("plug-in minus Classifier accuracy")
    axs[0].set_ylabel("accuracy, final step\n(novel environments)")
    h = [Line2D([], [], color=DARK, lw=1.3), Line2D([], [], color=LIGHT, lw=1),
         Line2D([], [], color="k", ls=":", lw=.9), Line2D([], [], color=DARK, ls="", marker="o", ms=4, mec="k")]
    fig.legend(h, [f"with dreams ({len(dream)} run{'s' if len(dream) != 1 else ''}: seed {', '.join(str(k) for k, _ in dream)})",
                   f"no dreams ({len(nodream)} runs: seeds {', '.join(str(k) for k, _ in nodream)})", "Naive Bayes",
                   "still training"], ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(.5, 1.0))
    fig.text(.5, .885, "Generator route, batch 1,000, the RCC's own feedback; 1 test = 1,000 novel-environment episodes per 250 "
                       "iterations; status on " + time.strftime("%d-%m at %H:%M"), ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, .87))
    out = fig_path("fig_dream_effect.png")
    fig.savefig(out, dpi=300)
    print(out)
    for tag, group in (("no dreams", nodream), ("dreams", dream)):
        for k, s in group:
            print(f"{tag:9s} seed {k}: gate -> past Naive {s[4]}")


if __name__ == "__main__":
    main()
