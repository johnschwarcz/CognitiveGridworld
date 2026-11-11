"""Is the drop of the |S| threshold (Fig 12: ~500; now 100-125) the Generator fix? (John 11-10: "Has the task structure
changed (big problem) or is this purely the optimized architecture (important to be able to show)?")

train_bayes_generator.py, Generator alone trained on Joint's answers (no Classifier, no LSTM), the paper's Generator
(--arch old: no stream LayerNorms, no skip, default init) vs the fixed one (--arch new, gen_zflow), |S| 100 / 125 / 250 /
500 x seeds 0-2 (+ 25 / 50 / 75, John 11-11: "let's see smaller |S| as well"), d_E 30, batch 10,000, lr 1e-3, 8,000 episodes (cluster array 66657607 + 66660113, CPU). Accuracy =
Bayes inference with the LEARNED likelihood (plug-in), goal at T = 30, 10,000 test episodes per environment set per test.
    a  novel environments at the end of training, one dot per seed
    b  training environments, the same runs
    c  novel environments through training, one line per run
Logs pulled to main/DATA/experimental/bayes_gen_arch/.

RESULT 11-11 00:20 (22 of 24 runs at 8,000 episodes; |S| 125 seed 0 still running at 7,000 / 5,500):
    new  12 of 12 generalize: novel .61-.67 (|S| 100), .69-.72 (125-500); training .71-.74
    old  3 of 12 escape: |S| 100 seed 2 (.43 novel), 250 seed 0 (.71), 500 seed 0 (.70); the other 9 stay at chance on
         novel environments (.10-.12) while partly memorizing training ones (.44 / .36 / .26 / .16 at |S| 100 / 125 /
         250 / 500)
RESULT 11-11 11:20, all 42 runs (|S| 25 / 50 / 75 added):
    below |S| 100 BOTH Generators memorize and neither generalizes: novel .08-.15 (chance) for all 18 runs; training
         .72-.74 / .65-.67 / .52-.57 at |S| 25 / 50 / 75 (fixed and paper's within .05 of each other: no collapse here)
    from |S| 100 the fixed Generator generalizes in 12 of 12 (novel .61-.67 at 100, .69-.72 at 125-500, = training);
         the paper's escapes in 3 of 12 and the other 9 fall below even the memorization level (training .44 -> .16)
So there IS a memorization -> generalization transition, between |S| 75 and 100, for both architectures alike (the
number of training environments, not the architecture, sets it); above it the paper's Generator fails to optimize
in ~3 of 4 runs. Joint / Naive are drawn per |S| (|S| 25's novel set = 3 environments, Joint .57-.63).
(00:20 reading:) So the task is unchanged and the threshold is not a property of |S|: the paper's Generator escapes in about 1 of 4 runs
at every |S| tested (the known channel collapse), and an escaped run generalizes at |S| 250-500 as well as the fixed one.
The fix removes the collapse; Fig 12's |S| effect was the escape probability (400: 1/3, 800: 2/3, n = 3 each).

    python bayes_gen_arch_plot.py [--pull]
"""
import os, sys, json, glob, re, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "bayes_gen_arch")
SIZES, SEEDS, EPISODES = (25, 50, 75, 100, 125, 250, 500), (0, 1, 2), 8000     # 25-75 added 11-11 (66662744)
ARCH = (("new", "fixed Generator", "#08519c", "o"), ("old", "paper's Generator", "#e6550d", "s"))


def pull():
    os.makedirs(LOGS, exist_ok=True)
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/bayes_gen/joint_S*_d30_s*_{{old,new}}.jsonl' {LOGS}/",
                   shell=True)


def load():
    runs = {}
    for f in glob.glob(os.path.join(LOGS, "joint_S*_d30_s*_*.jsonl")):
        S, k, arch = re.search(r"_S(\d+)_d30_s(\d+)_(old|new)", f).groups()
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if recs:
            runs[arch, int(S), int(k)] = recs
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    runs = load()
    # Joint / Naive per |S| and environment set (11-11: at |S| 25 the novel set is 3 environments, Joint .57-.63);
    # panel c uses the mean over |S| >= 100, where the bounds are stable
    ref = {(e, b, S): np.mean([r[-1][f"{e}_{b}"] for (_, S_, _), r in runs.items() if S_ == S])
           for e in ("novel", "train") for b in ("joint", "naive") for S in SIZES}
    joint = np.mean([r[-1]["novel_joint"] for (_, S, _), r in runs.items() if S >= 100])
    naive = np.mean([r[-1]["novel_naive"] for (_, S, _), r in runs.items() if S >= 100])
    fig, ax = plt.subplots(1, 3, figsize=(3 * 3.3 + .8, 3.6), constrained_layout=True,
                           gridspec_kw=dict(width_ratios=(1, 1, 1.3)))
    for p, (key, title) in enumerate((("novel_plugin", "Novel environments, end"), ("train_plugin", "Training environments, end"))):
        a = ax[p]
        for j, (arch, name, col, mk) in enumerate(ARCH):
            for i, S in enumerate(SIZES):
                ys = [runs[arch, S, k][-1][key] for k in SEEDS if (arch, S, k) in runs]
                xs = i + (j - .5) * .32 + np.linspace(-.07, .07, len(ys))
                a.scatter(xs, ys, s=26, color=col, marker=mk, zorder=3, edgecolor="w", lw=.4,
                          label=name if i == 0 else None)
        env = key.split("_")[0]
        for i, S in enumerate(SIZES):                            # the bounds of this |S|'s own test environments
            a.hlines(ref[env, "joint", S], i - .42, i + .42, color="k", lw=1.1, ls="--", zorder=1)
            a.hlines(ref[env, "naive", S], i - .42, i + .42, color="k", lw=1.1, ls=":", zorder=1)
        a.axhline(.1, color="0.6", lw=.8, ls=":", zorder=1)
        a.set_xticks(range(len(SIZES)))
        a.set_xticklabels([str(S) for S in SIZES])
        a.set_xlabel(r"number of variables $|\mathcal{S}|$")
        a.set_ylim(0, .82)
        a.set_title(f"{'ab'[p]}   {title}", fontsize=fs["title"] - 4, loc="left")
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("goal accuracy with the learned likelihood")
    ax[1].set_yticklabels([])
    for y, lab in ((ref["train", "joint", SIZES[-1]], "Joint Bayes"), (ref["train", "naive", SIZES[-1]], "Naive Bayes"),
                   (.1, "chance")):
        ax[1].text(len(SIZES) - .55, y, lab, fontsize=fs["legend"] - 3, va="center", ha="left")
    ax[0].legend(frameon=False, fontsize=fs["legend"] - 2, loc="center right", bbox_to_anchor=(1, .42), handletextpad=.3)
    a = ax[2]
    sty = {0: "-", 1: "-", 2: "-"}
    cols = {S: c for S, c in zip(SIZES, plt.cm.viridis(np.linspace(0, .85, len(SIZES))))}
    for (arch, S, k), recs in sorted(runs.items()):
        e = np.array([r["ep"] for r in recs])
        y = np.array([r["novel_plugin"] for r in recs])
        a.plot(np.maximum(e, 25), y, color=cols[S], lw=1.3 if arch == "new" else 1.0,
               ls="-" if arch == "new" else (0, (3, 1.5)), alpha=.9)
    a.set_xscale("log")
    a.set_xlim(25, EPISODES * 1.1)
    a.set_ylim(0, .82)
    a.axhline(joint, color="k", lw=1, ls="--", zorder=1)
    a.axhline(naive, color="k", lw=1, ls=":", zorder=1)
    a.set_xlabel("training episodes")
    a.set_title("c   Novel environments through training", fontsize=fs["title"] - 4, loc="left")
    a.spines[["top", "right"]].set_visible(False)
    a.legend(handles=[plt.Line2D([], [], color=cols[S], lw=1.5, label=f"$|\\mathcal{{S}}|$ = {S}") for S in SIZES] +
             [plt.Line2D([], [], color="k", lw=1.3, label="fixed"),
              plt.Line2D([], [], color="k", lw=1.0, ls=(0, (3, 1.5)), label="paper's")],
             frameon=True, facecolor="w", edgecolor="none", framealpha=1, fontsize=fs["legend"] - 3, loc="center left",
             bbox_to_anchor=(0, .52), ncol=1,
             handlelength=1.8)
    done = sum(r[-1]["ep"] >= EPISODES for r in runs.values())
    fig.suptitle("Generator alone, trained on Joint's answers: the paper's Generator vs the fixed one (accuracy of Bayes "
                 "with the learned likelihood)\n" + f"$d_E$ = 30, batch 10,000, lr $10^{{-3}}$, {EPISODES:,} episodes, "
                 f"3 seeds per $|\\mathcal{{S}}|$ ({done} of {len(runs)} runs finished); 10,000 test episodes per "
                 "environment set", fontsize=fs["legend"] - 2)
    out = fig_path("bayes_gen_arch.png")
    fig.savefig(out, dpi=200)
    print(out)
    for arch, name, _, _ in ARCH:
        for S in SIZES:
            print(name, S, [(k, round(runs[arch, S, k][-1]["novel_plugin"], 3), round(runs[arch, S, k][-1]["train_plugin"], 3),
                             runs[arch, S, k][-1]["ep"]) for k in SEEDS if (arch, S, k) in runs])


if __name__ == "__main__":
    main()
