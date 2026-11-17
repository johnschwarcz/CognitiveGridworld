"""Supplementary Fig 13, revised (proposed 11-13): Generator route vs Classifier route of the embedding gradient, per-seed
learning curves under the original feedback (a) and oracle feedback (b). Replaces plot_ablation.py's figure: new runs only,
the same network, seeds and hyperparameters in all four conditions; within a seed the four conditions share the initial
network, the environments and the testing episodes (Joint / Naive equal at every test).

Runs (train_cluster_runs.py; C = 2, |S| 500, d_E 30, batch 10,000, lr 1e-4 for every module, beta1 .01, the modified
Generator "gen_zflow", the standard Classifier; one test of 10,000 testing environments every 250 training episodes):
    a  original feedback (mode RL: reward + the Classifier's sampled answers)
       blue  rcc60k_lr4_s0-5        (continued by rcc120k_lr4_s0-5: 11-14 07:28-14:38 IST, paused by STOP files at
                                     ~70-82k (s0-3) / ~62k (s4-5) for the shuffled batch-10,000 pairs; nets saved)
       red   rccclf60k_lr4_s0-5 -> rccclf120k_lr4_s0-5 (resumed at 60k with a fresh Adam; the duplicate ep-60,000 test of
             the resume is dropped)
    b  oracle feedback (true realizations): gen60k_lr4_s0-5 (blue), clf60k_lr4_s0-5 (red), both cut at 60k (the stalled
       seed's continuation clf120k_lr4_s2 is in the caption only: drawing only the worst seed's continuation is selective)
Blue lines that end before the red ones get a dot at their last test (the unmatched-endpoint fallback).
Joint / Naive = mean over seeds and tests of the per-test values (per test .756-.783 / .432-.466).

NUMBERS (11-13 03:40; crossing = first of 3 consecutive tests above that test's Naive Bayes; final = mean of the last 5):
    beats Naive, original feedback  Generator 12.5/11.5/11.5/12.0/13.75/11.75k, Classifier 43.5/48.25/45.0/49.25/62.75/48.0k
                                    (3.5-4.6x per seed)
    beats Naive, oracle             Generator 3.75/3.5/4.0/3.75/3.75/3.75k, Classifier 9.75/7.5/11.0/9.25/9.5/6.5k (1.7-2.8x)
    paired order                    Generator above its Classifier-route twin at every test from leaving chance to 60k, 12/12
    final, original                 Generator .690-.696 at 60k (still +.006-.012 per 10k); Classifier s0-3 .695-.705 at 120k,
                                    s4/s5 .57/.60 at 74k (running, cluster 66683258, ETA ~12:00 11-13)
    final, oracle                   Generator .722-.729; Classifier .701-.710 in 5/6, s2 .524 (2 channels unused; .50-.55 to 120k)
    channels (census at 60k)        Classifier route, original feedback: 5/6 nets have a channel whose shuffling costs <= .055;
                                    every channel of all 12 Generator-route nets >= .258; s0-3 use all 5 at 120k (transient)
Sources and checks: scratchpad routes_fig/RECOMMENDATION.md + factcheck.md of session 79a05095 (recomputed from these logs).

    python fig13_routes.py [--pull] [--paper]     (--paper also writes fig13_routes.pdf + .svg, pdf.fonttype 42)
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "cluster_runs", "jsonl")
BLUE, RED = "#2c7bb6", "#d7191c"          # the submitted figure's colours
SEEDS = range(6)
PANELS = [  # tag, title, Generator route (base, continuation), Classifier route (base, continuation), cut
    ("a", "Original feedback (reward, own answers)", ("rcc60k_lr4", "rcc120k_lr4"), ("rccclf60k_lr4", "rccclf120k_lr4"), None),
    ("b", "Oracle feedback (true realizations)", ("gen60k_lr4", None), ("clf60k_lr4", None), 60000),
]


def load(base, ext, k):
    """One seed's tests (ep > 0), the continuation's log when it exists (it repeats the base run's first rows)."""
    rows = []
    for n in ([f"{ext}_s{k}"] if ext else []) + [f"{base}_s{k}"]:
        f = os.path.join(LOGS, n + ".jsonl")
        if os.path.exists(f):
            rows = [json.loads(l.replace("NaN", "null")) for l in open(f) if l.strip()]
            break
    out, last = [], -10
    for r in sorted(rows, key=lambda r: r["ep"]):
        if r["ep"] > 0 and r["ep"] - last > 1:   # drops the resume's duplicate test (59,999 then 60,000)
            out.append(r)
            last = r["ep"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true", help="copy the live continuation logs from the cluster first")
    ap.add_argument("--paper", action="store_true", help="also save PDF + SVG for the paper")
    ap.add_argument("--own-only", action="store_true", help="panel a only (the RCC's own feedback; John 11-13: no oracle "
                    "feedback in the route comparison) -> fig13_routes_own.png")
    a = ap.parse_args()
    if a.pull:
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/{{rccclf120k_lr4,rcc120k_lr4}}_s*.jsonl' "
                       f"{LOGS}/", shell=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.serif": "cmr10", "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7.5, "axes.linewidth": .6, "pdf.fonttype": 42})
    panels = PANELS[:1] if a.own_only else PANELS
    fig, axs = plt.subplots(1, len(panels), figsize=(4.4, 3.3) if a.own_only else (5.5, 2.4), sharey=True, squeeze=False)
    axs = axs[0]
    lo = 1
    for ax, (tag, title, gen, clf, cut) in zip(axs, panels):
        runs = {c: [load(*arm, k) for k in SEEDS] for c, arm in ((RED, clf), (BLUE, gen))}
        runs = {c: [[r for r in rs if cut is None or r["ep"] <= cut] for rs in v] for c, v in runs.items()}
        red_end = max(rs[-1]["ep"] for rs in runs[RED] if rs)
        jt, nv = [], []
        for c in (RED, BLUE):
            for rs in runs[c]:
                if not rs:
                    continue
                e, y = np.array([r["ep"] for r in rs]), np.array([r["net"] for r in rs])
                ax.plot(e, y, color=c, lw=.8, alpha=.85)
                if c == BLUE and e[-1] < red_end - 1000:
                    ax.plot(e[-1], y[-1], "o", color=c, ms=2.5)
                jt += [r["joint"] for r in rs]
                nv += [r["naive"] for r in rs]
                lo = min(lo, y.min())
        ax.axhline(np.mean(jt), color="k", ls="--", lw=.8)
        ax.axhline(np.mean(nv), color="k", ls=":", lw=.9)
        ax.set_xscale("log")
        ax.set_xlim(250, 1.25e5)
        ax.set_xlabel(r"training episodes (batch $10^4$)")
        ax.set_title(r"$\mathbf{%s}$   %s" % (tag, title), fontsize=8, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylim(lo - .02, .80)
    axs[0].set_yticks([.2, .4, .6, .8])
    axs[0].set_ylabel("goal accuracy, final step\n(testing environments)")
    h = [Line2D([], [], color=BLUE, lw=1.2), Line2D([], [], color=RED, lw=1.2),
         Line2D([], [], color="k", ls="--", lw=.8), Line2D([], [], color="k", ls=":", lw=.9)]
    fig.legend(h, ["Generator route (embeddings trained by the Generator's loss)",
                   "Classifier route (embeddings trained by the Classifier's loss)", "Joint Bayes", "Naive Bayes"],
               ncol=1 if a.own_only else 2, loc="upper center", frameon=False, bbox_to_anchor=(.5, 1.0), handlelength=1.8,
               columnspacing=1.2)
    if a.own_only:
        fig.text(.5, .705, "each line = one network; 6 seeds per route; 1 test = $10^4$ testing environments,\n"
                           "every 250 training episodes; dot = paused before $1.2\\times10^5$ (will continue)",
                 ha="center", fontsize=7, linespacing=1.4)
        fig.tight_layout(rect=(0, 0, 1, .65))
    else:
        fig.text(.5, .775, r"each line = one network; 6 seeds per route; 1 test = $10^4$ testing environments, "
                           r"every 250 training episodes", ha="center", fontsize=7.5)
        fig.tight_layout(rect=(0, 0, 1, .76))
    out = fig_path("fig13_routes_own.png" if a.own_only else "fig13_routes.png")
    fig.savefig(out, dpi=300)
    print(out)
    if a.paper:
        for ext in ("pdf", "svg"):
            fig.savefig(fig_path(f"fig13_routes.{ext}"))


if __name__ == "__main__":
    main()
