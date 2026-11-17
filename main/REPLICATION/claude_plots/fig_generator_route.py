"""Summary figure: why routing the embedding gradient through the Generator matters (John 11-12 01:00: one figure with
our best argument). Print size (5.5 in, 7-8 pt).

a, b  Fig 13's networks (train_cluster_runs.py: |S| 500, batch 10,000, lr 1e-4, 60,000 iterations, 6 seeds; 10,000 novel
      episodes per test), oracle feedback (a) and the RCC's own feedback (b): novel goal accuracy of the Generator route's
      generative model read out by exact Bayes, of the Generator route's Classifier, of the Classifier route, and of the
      Classifier route with the Generator's read-in structure (one function for every channel; readin_shared, tasks
      123-134; drawn once its logs exist). Thin = seeds, thick = mean where every seed has a test.
c     Which channels the networks use (fig_channel_census.py; 2,000 novel episodes per net): accuracy lost when one
      channel's observations are shuffled. Top: nets that learned; bottom: nets that stalled.
d     The same embeddings, start, batch, lr and labels, learned from scratch (fig_matched_learners.py; |S| 500, 10,000
      novel episodes per test): by generation or by classification, with one function for every channel or one per
      channel (mean over seeds: 4 each, LSTM 2; each seed interpolated in log batches, mean where >= 2 seeds).

Caption numbers (11-12 01:10): a, the generative model reaches .65 at 1,250-1,750 iterations, its Classifier at
9,750-11,000, the Classifier route at 19,250-29,250 (5 of 6 seeds); b, the generative model 13,250-15,750, its Classifier
25,500-28,750, the Classifier route never in 60k; c, every stalled net ignores 1-2 whole channels, every net that learned
uses all 5; d, Generator .65 at 750-1,000 batches, Bayes-form classifier .21 at 3,000-3,750 (all channels at R^2 ~.5),
per-channel versions of either stall with channels missing.

    python fig_generator_route.py [--pull]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
from main.utils import fig_path

DATA = os.path.join(ROOT, "main", "DATA")
EXP = os.path.join(DATA, "experimental")
ARM_LOGS = os.path.join(EXP, "cluster_runs", "jsonl")
GEN_DIR = os.path.join(EXP, "fig13_embeddings", "controls")
CLF_DIR = os.path.join(EXP, "clf_scratch")
C_GEN, C_GENCLF, C_CLF, C_SHARED = "#08519c", "#6baed6", "#a50f15", "#e6550d"
NET_LINES = (("gen", "plugin", C_GEN, "-", "Gen. route: Generator + Bayes"),
             ("gen", "net", C_GENCLF, "-", "Gen. route: Classifier"),
             ("clf", "net", C_CLF, "-", "Clf. route"),
             ("shared", "net", C_SHARED, "-", "Clf. route, shared read-in"))
ARMS = {"a": dict(gen="gen60k_lr4", clf="clf60k_lr4", shared="clfshared60k_lr4"),
        "b": dict(gen="rcc60k_lr4", clf="rccclf60k_lr4", shared="rccclfshared60k_lr4")}
MATCHED = (("Gen., shared", GEN_DIR, "shared", C_GEN, "-"),
           ("Gen., per ch.", GEN_DIR, "per_channel_random", C_GEN, (0, (3, 1.5))),
           ("Clf., shared", CLF_DIR, "reader_shared", C_CLF, "-"),
           ("Clf., per ch.", CLF_DIR, "reader_unshared", C_CLF, (0, (3, 1.5))),
           ("Clf., LSTM", CLF_DIR, "reader_lstm", C_CLF, (0, (1, 1))))
CENSUS = (("gen60k_lr4_s2", "Gen. route"), ("clf60k_lr4_s0", "Clf. route"), ("clf30k_s2", "Clf. route, lr 1e-3"),
          ("rccclf60k_lr4_s2", "Clf. route, RCC fb."), ("clf30k_s0", "Clf. route, lr 1e-3"),
          ("clf30k_s1", "Clf. route, lr 1e-3"), ("clf30k_s3", "Clf. route, lr 1e-3"),
          ("rccclf60k_lr4_s3", "Clf. route, RCC fb."), ("rccclf60k_lr4_s0", "Clf. route, RCC fb."),
          ("rccclf60k_lr4_s1", "Clf. route, RCC fb."), ("rccclf60k_lr4_s4", "Clf. route, RCC fb."),
          ("rccclf60k_lr4_s5", "Clf. route, RCC fb."), ("clf60k_lr4_s2", "Clf. route"))


def pull():
    for arm in ("clfshared60k_lr4", "rccclfshared60k_lr4"):
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/{arm}_s*.jsonl' {ARM_LOGS}/ 2>/dev/null",
                       shell=True)
        subprocess.run(f"cp {ROOT}/runs/cluster_runs/{arm}_s*.jsonl {ARM_LOGS}/ 2>/dev/null", shell=True)   # local GPU runs
    for d, pat in ((GEN_DIR, "{shared,per_channel_random}"), (CLF_DIR, "{reader_shared,reader_unshared,reader_lstm}")):
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/bayes_gen/truth_S500_d30_s[0-9]_{pat}.jsonl' {d}/ "
                       "2>/dev/null", shell=True)


def jl(f):
    return [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []


def seeds_mean(runs, xkey, ykey):
    """thin lines per seed; the mean at the tests every seed has"""
    curves = [{q[xkey]: q[ykey] for q in r} for r in runs]
    common = sorted(set.intersection(*(set(c) for c in curves)))
    return curves, np.array(common), np.array([np.mean([c[x] for c in curves]) for x in common])


def interp_mean(runs, key, grid):
    """each seed interpolated (in log iterations) onto the grid inside its own range; the mean where >= 2 seeds"""
    Y = np.full((len(runs), len(grid)), np.nan)
    for i, r in enumerate(runs):
        x = np.array([max(q["ep"], 1) for q in r], float); y = np.array([q[key] for q in r])
        ok = (grid >= x[1]) & (grid <= x[-1])
        Y[i, ok] = np.interp(np.log(grid[ok]), np.log(x[1:]), y[1:])
    n = np.isfinite(Y).sum(0)
    return grid[n >= 2], np.nanmean(Y[:, n >= 2], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fig_channel_census import SRC, parse
    T7, T8 = 7, 8
    plt.rcParams.update({"font.family": "serif", "font.serif": "cmr10", "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "font.size": T8, "axes.labelsize": T8, "axes.titlesize": T8,
                         "xtick.labelsize": T7, "ytick.labelsize": T7, "legend.fontsize": T7, "axes.linewidth": .6,
                         "xtick.major.width": .6, "ytick.major.width": .6, "pdf.fonttype": 42})
    joint, naive = np.load(os.path.join(DATA, "cache", "bayes_bounds_ctx2_T30.npy"))
    fig = plt.figure(figsize=(5.5, 4.8))
    Gt = fig.add_gridspec(1, 2, left=.085, right=.985, top=.955, bottom=.585, wspace=.28)
    Gb = fig.add_gridspec(1, 4, left=.215, right=.985, top=.44, bottom=.085, width_ratios=(1, .05, .42, 1.2), wspace=.06)
    axa, axb = fig.add_subplot(Gt[0]), fig.add_subplot(Gt[1], sharey=None)
    axc, cax, axd = fig.add_subplot(Gb[0]), fig.add_subplot(Gb[1]), fig.add_subplot(Gb[3])
    for ax, key, title in ((axa, "a", "a   Oracle feedback"), (axb, "b", "b   The RCC's own feedback")):
        for arm, ykey, col, ls, lab in NET_LINES:
            runs = [r for r in (jl(os.path.join(ARM_LOGS, f"{ARMS[key][arm]}_s{k}.jsonl")) for k in range(6)) if r]
            if not runs:
                continue
            curves, x, m = seeds_mean(runs, "ep", ykey)
            for c in curves:
                xs = sorted(c)[1:]
                ax.plot(xs, [c[v] for v in xs], color=col, lw=.35, alpha=.35)
            ax.plot(x[1:], m[1:], color=col, ls=ls, lw=1.4, label=f"{lab} ({len(runs)})")
        ax.axhline(joint, color="k", ls="--", lw=.7)
        ax.axhline(naive, color="k", ls=":", lw=.7)
        ax.set(xscale="log", xlim=(250, 61000), ylim=(0, .8), xlabel="training iterations")
        ax.set_title(title, loc="left")
    axa.set_ylabel("goal accuracy, novel environments")
    axa.text(270, joint + .015, "Joint Bayes", fontsize=T7)
    axa.text(270, naive + .015, "Naive Bayes", fontsize=T7)
    axa.legend(frameon=False, loc="lower right", fontsize=T7 - 1, handlelength=1.4, borderaxespad=.1)

    # c: channel census
    D = parse()
    nets = [(n, lab) for n, lab in CENSUS if n in D]
    M = np.array([D[n]["drop"] for n, _ in nets])
    im = axc.imshow(M, cmap="viridis", vmin=0, vmax=.35, aspect="auto")
    axc.set_yticks(range(len(nets)))
    axc.set_yticklabels([f"{lab} ({D[n]['acc']:.2f})" for n, lab in nets], fontsize=T7 - 1)
    axc.set_xticks(range(5)); axc.set_xticklabels(range(5))
    axc.set_xlabel("observation channel")
    axc.axhline(3.5, color="w", lw=1.5)
    axc.set_title("c   Stalled nets ignore whole channels", loc="left", x=-.68)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("accuracy lost, channel shuffled", fontsize=T7)
    cb.ax.tick_params(labelsize=T7 - 1)

    # d: matched learners from scratch
    for lab, d, tag, col, ls in MATCHED:
        runs = [r for r in (jl(os.path.join(d, f"truth_S500_d30_s{k}_{tag}.jsonl")) for k in range(6)) if r]
        if not runs:
            continue
        key = "novel_reader" if "novel_reader" in runs[0][-1] else "novel_plugin"
        x, m = interp_mean(runs, key, np.geomspace(25, 15000, 160))
        axd.plot(x, m, color=col, ls=ls, lw=1.3, label=lab)
        print(f"d  {lab:14s} seeds {len(runs)}: last mean {m[-1]:.3f} at {x[-1]:.0f}")
    axd.axhline(joint, color="k", ls="--", lw=.7)
    axd.axhline(naive, color="k", ls=":", lw=.7)
    axd.set(xscale="log", xlim=(25, 15000), ylim=(0, .8), xlabel="training batches, from scratch")
    axd.set_title("d   Same embeddings and trainer", loc="left")
    axd.legend(frameon=False, loc="upper left", bbox_to_anchor=(.01, .93), fontsize=T7 - 1, handlelength=1.6,
               borderaxespad=0, labelspacing=.25)
    for ax in (axa, axb, axd):
        ax.spines[["top", "right"]].set_visible(False)
    out = fig_path("fig_generator_route.png")
    fig.savefig(out, dpi=300)
    print(out)


if __name__ == "__main__":
    main()
