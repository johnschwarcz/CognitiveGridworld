"""Fig 13 as one panel (John 11-10 09:30: "put all versions on 1 panel"; the question: what does the Generator do under the
original feedback?): feedback (oracle / the RCC's own) x route of the embedding gradient (Generator / Classifier), plus
Fig 13's own 10,000-episode runs. Axes as Fig 13 (plot_ablation.py): Joint's accuracy minus the network's, novel
environments, T = 30, log-log; x = training episodes.

Every new arm: |S| 500, batch 10,000, 30,000 episodes, the updated Generator (gen_zflow), one test of 10,000 novel episodes
(both variables held out) per 250 episodes, seeds 0-5 (John 11-09: same seeds, episodes and batch size for every arm).
  oracle feedback (lr 1e-3, Fig 13's settings): the Generator is given the true values
      Generator   train_cluster_runs.py gen30k_s0-5    embeddings get gradients from the Generator only
      Classifier  clf30k_s0-5                          ... from the Classifier only
      dashed: Fig 13's own runs of the same two routes (DATA/oracle_pilot default_rep0-5 / clf_reg_rep0-5; 10,000 episodes)
  original feedback (lr 1e-4, entropy bonus .01): reward-trained Classifier; the Generator sees the Classifier's sampled
  answers and whether they were rewarded
      Generator   rccdiag_lr4_s0-5                     the RCC
      Classifier  rccdiag_lr4clf_s0-5                  the RCC with the embedding gradient routed through the Classifier
                                                       (the Generator still trains, on detached embeddings)
11-11 00:10: every new arm switched to the 60k batch (John 11-10: "1e-4 for all", 60,000 episodes; train_cluster_runs.py
gen60k_lr4 / clf60k_lr4 (oracle feedback), rcc60k_lr4 (the RCC, fresh) / rccclf60k_lr4 (original feedback, Classifier
route), seeds 0-5, batch 10,000, one test of 10,000 novel episodes per 250 episodes). The 30k versions above (oracle arms
at lr 1e-3, rccdiag_lr4*) are kept in VERSIONS_30K; lr 1e-3 seeds 4-5 held on the cluster (John: plan is to ignore them).
Thick line = mean over the seeds that have a test (n in the legend) while seeds are missing, all 6 once done.
RESULT 11-11 11:20 (12 of 24 runs at 60k; Joint ~.77 on these tests; "left chance" = first test above .2):
    oracle, Generator route      s0-1 .721 / .730 at 60k (s2-3 .72 at 44k); left chance at 1,250-1,500 episodes
    oracle, Classifier route     s0-1 .699 / .713 at 60k; s3-5 .68-.71 at 43-46k, s2 .52 at 46.5k; left at 3,500-5,750
    the RCC (original feedback)  6 of 6 .691-.704 at 60k; left chance at 6,500-10,500
    original, Classifier route   s0-1 .570 / .578 at 60k; s2-3 .21 / .33 at 34k; left chance at 17,500-33,250
  Under oracle feedback the route is mostly a matter of speed at lr 1e-4 (Classifier route ~3x slower, ends .01-.03
  lower in 5 of 6 seeds; the paper's 10k-episode lr 1e-3 runs had stopped it at .31-.61). Under the original feedback
  the Classifier route is 3x slower than the RCC and ends .12-.13 lower at 60k.
11-11 12:10 (John: "a bit concerning, any idea how to rescue it?"), robust measures, every seed:
  episodes to reach novel accuracy .45 / .60 / .68
    oracle lr 1e-4   Generator 3.5-4k / 6.75-7.5k / 15.5-16.5k; Classifier 6.5-10.5k / 13.75-25.75k / 25-34.75k (s2 never
                     reaches .60: .52)  -> 2-2.6x slower, no overlap
    oracle lr 1e-3   (gen30k / clf30k, s0-3, 30k) Generator 1.5k / 2.25-2.75k / 6.25-7.5k, ends .72-.73 in 4/4;
                     Classifier 3.5-5k / 7.25-8.75k / 10.75k in 1 of 4, the other 3 plateau at .63-.64
    original lr 1e-4 RCC 11.25-13.75k / 17.75-20.25k / 41-43.25k; Classifier route 43.5-48.25k / never in 60k (s0-1;
                     s2-3 not .45 by 34.75k) -> ~4x slower to Naive's level
  training minus novel accuracy at the end: Generator route +.001-.004 (oracle), +.004-.011 (RCC); Classifier route
  +.004-.045 (oracle, mean +.016), +.019-.051 (original): the Classifier route memorizes more.
  Panel c: mean +- SEM over seeds (the paper's Fig 13 statistic); the routes' bands never overlap.
Colour = route (blue Generator, red Classifier); dark = oracle feedback, light = original feedback. Thin lines = seeds;
thick line = mean over seeds, drawn only where every seed of the version has a test. Joint and Naive: the cached bounds of
plot_ablation.py (bayes_bounds_ctx2_T30.npy).

11-13 00:20 (John: "weren't we waiting for rcc_arms to collect more episodes?"): the Classifier-route arms continue from
their 60k nets to 120k (train_cluster_runs.py RESUME: rccclf120k_lr4_s0-5, clf120k_lr4_s2 = the oracle seed that lost 2
channels); a seed's thin line follows its continuation when one exists, thick lines / bands stay where every seed has a
test. The Generator-route continuations were cut 11-12 05:45 (John; flat at .69-.73 since 30k).
RESULT 11-13 00:20: original feedback, Classifier route s0-3 at 120k: Joint - net .053-.071 (acc .697-.715; the RCC's
Generator route at 60k: .07); s4-5 (.38 / .57 at 60k) continue on the cluster (66683258). Oracle s2 flat at .53 to 120k.
RESULT 11-13 12:25 (66683258 done): s4 .699, s5 .636 at 120k; mean of the 6 Classifier-route seeds .694 at 120k vs .697 for
the Generator route's 6 at 60k. s5 still ignores channel 0 (census 66685419), the other 5 use every channel.

    python fig13_rcc_arm.py [--pull]
"""
import os, sys, json, glob, re, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

DATA = os.path.join(ROOT, "main", "DATA")
LOGS = os.path.join(DATA, "experimental", "cluster_runs", "jsonl")
SEEDS, EPISODES = 6, 60000
EXT = {"rccclf60k_lr4_s": "rccclf120k_lr4_s", "clf60k_lr4_s": "clf120k_lr4_s"}   # continued to 120k (11-12, RESUME)
# (label, colour, line style, ("cluster", run prefix) or ("pilot", Fig 13's condition))
VERSIONS = (("oracle, Generator route", "#08519c", "-", ("cluster", "gen60k_lr4_s")),
            ("oracle, Generator route, Fig 13 (10k)", "#08519c", (0, (4, 2)), ("pilot", "default")),
            ("oracle, Classifier route", "#a50f15", "-", ("cluster", "clf60k_lr4_s")),
            ("oracle, Classifier route, Fig 13 (10k)", "#a50f15", (0, (4, 2)), ("pilot", "clf_reg")),
            ("original, Generator route (the RCC)", "#4fa3e0", "-", ("cluster", "rcc60k_lr4_s")),
            ("original, Classifier route", "#f4845f", "-", ("cluster", "rccclf60k_lr4_s")))
VERSIONS_30K = (("oracle, Generator route", "#08519c", "-", ("cluster", "gen30k_s")),
            ("oracle, Generator route, Fig 13 (10k)", "#08519c", (0, (4, 2)), ("pilot", "default")),
            ("oracle, Classifier route", "#a50f15", "-", ("cluster", "clf30k_s")),
            ("oracle, Classifier route, Fig 13 (10k)", "#a50f15", (0, (4, 2)), ("pilot", "clf_reg")),
            ("original, Generator route (the RCC)", "#4fa3e0", "-", ("cluster", "rccdiag_lr4_s")),
            ("original, Classifier route", "#f4845f", "-", ("cluster", "rccdiag_lr4clf_s")))

def pull():
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/{{gen60k,clf60k,rcc60k,rccclf60k,rccclf120k}}*.jsonl' "
                   f"{LOGS}/", shell=True)
    subprocess.run(f"cp {ROOT}/runs/cluster_runs/clf120k_lr4_s2.jsonl {LOGS}/", shell=True)   # local GPU, task 183


def pilot(cond):
    """[(episodes, accuracy)] of Fig 13's runs (final-step test accuracy; test k after 250 k episodes)"""
    import torch
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, "oracle_pilot", f"{cond}_rep*_net.pth")),
                    key=lambda p: int(re.search(r"rep(\d+)", p).group(1))):
        a = np.asarray(torch.load(f, weights_only=False, map_location="cpu")["test_acc_through_training"], float)[:, -1]
        out.append((np.arange(len(a)) * 250, a))
    return out


def cluster(prefix):
    """[(episodes, accuracy)] of the seeds of this arm that have a log"""
    out = []
    for k in range(SEEDS):
        for p in ((EXT[prefix],) if prefix in EXT else ()) + (prefix,):     # the continuation's log starts with the source's
            f = os.path.join(LOGS, f"{p}{k}.jsonl")
            recs = [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []
            if recs:
                out.append((np.array([r["ep"] for r in recs]), np.array([r["net"] for r in recs])))
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    fs = apply_fig_style()
    joint, naive = np.load(os.path.join(DATA, "cache", "bayes_bounds_ctx2_T30.npy"))
    # John 11-11 01:00: two panels, the paper's Fig 13 runs (dashed) and the new 60k batch
    # John 11-11 12:00: a third column, mean +- SEM over seeds as in the paper's Fig 13 caption ("mean +- SEM with 6 seeds")
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0), sharey=True, constrained_layout=True,
                           gridspec_kw=dict(width_ratios=(1, 1.35, 1.35)))
    handles = ([], [])
    xmax_new = EPISODES
    for label, col, ls, (kind, src) in VERSIONS:
        p = 0 if kind == "pilot" else 1
        a = ax[p]
        runs = cluster(src) if kind == "cluster" else pilot(src)
        for x, acc in runs:
            a.plot(x[1:], (joint - acc)[1:], color=col, ls=ls, lw=.8, alpha=.35)
        n = min((len(x) for x, _ in runs), default=0)
        if len(runs) >= 2 and n > 1:                             # mean where every seed that has a log has a test
            G = joint - np.array([acc[:n] for _, acc in runs])
            m, se = G.mean(0), G.std(0, ddof=1) / np.sqrt(len(G))
            a.plot(runs[0][0][1:n], m[1:], color=col, ls=ls, lw=2.4)
            ax[2].fill_between(runs[0][0][1:n], (m - se)[1:], (m + se)[1:], color=col, alpha=.25, lw=0)
            ax[2].plot(runs[0][0][1:n], m[1:], color=col, ls=ls, lw=1.8)
        if kind == "pilot":
            state = f"$n = {len(runs)}$"
        else:
            done = sum(x[-1] >= EPISODES - 1 for x, _ in runs)
            state = (f"$n = {SEEDS}$" if done == SEEDS else
                     f"{done} of {SEEDS} done, {len(runs) - done} running (mean of {len(runs)})" if runs else "queued")
            if src in EXT:
                n2 = sum(x[-1] >= 2 * EPISODES - 1 for x, _ in runs)
                state += f"; {n2} continued to 120,000" if n2 else ""
            xmax_new = max([xmax_new] + [x[-1] for x, _ in runs])
        handles[p].append(plt.Line2D([], [], color=col, ls=ls, lw=2.4,
                                     label=f"{label.replace(', Fig 13 (10k)', '')}: {state}"))
    for p, (a, title, xmax) in enumerate(zip(ax, (r"a   Fig 13 (paper): oracle feedback, lr $10^{-3}$, 10,000 episodes",
                                                  r"b   New: lr $10^{-4}$, 60,000 episodes, every arm" +
                                                  (" (thin lines past 60,000: continued)" if xmax_new > EPISODES else ""),
                                                  r"c   Mean $\pm$ SEM over seeds (a and b; as in the paper)"),
                                              (1.1e4, 1.08 * xmax_new, 1.08 * xmax_new))):
        a.axhline(joint - naive, c="k", lw=1.6, ls="--", zorder=10)
        a.axhline(joint - .1, c="k", lw=1.0, ls=":", zorder=10)
        a.set(xscale="log", yscale="log", xlim=(200, xmax), xlabel="training episodes")
        for axis in (a.xaxis, a.yaxis):
            axis.set_major_locator(mticker.LogLocator(subs=(1., 2., 5.)))
            axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(mticker.NullFormatter())
        a.grid(alpha=.25, lw=.6, which="both")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.set_title(title, loc="left", fontsize=fs["title"] - 4)
        if p == 2:
            a.legend(handles=[plt.Line2D([], [], color="0.3", ls=(0, (4, 2)), lw=1.8, label="Fig 13 (paper), $n = 6$"),
                              plt.Line2D([], [], color="0.3", lw=1.8, label="new: seeds with a log (see b)")],
                     loc="upper center", bbox_to_anchor=(.5, -.16), frameon=False, fontsize=fs["legend"] - 2,
                     handlelength=2.6, title="colours as in a and b; band = SEM", title_fontsize=fs["legend"] - 2)
            continue
        a.legend(handles=handles[p], loc="upper center", bbox_to_anchor=(.5, -.16), frameon=False, ncol=1,
                 fontsize=fs["legend"] - 2, handlelength=2.6,
                 title="feedback, route of the embedding gradient", title_fontsize=fs["legend"] - 2)
    for y, lab in ((joint - naive, "Naive Bayes"), (joint - .1, "chance")):
        ax[2].text(1.01, y, lab, transform=ax[2].get_yaxis_transform(), va="center", ha="left", fontsize=fs["legend"] - 1)
    ax[0].set_ylabel("Joint $-$ network accuracy")
    fig.suptitle(r"Novel environments, $|\mathcal{S}| = 500$, batch 10,000; thin = seeds, thick = mean; "
                 "10,000 episodes per test", fontsize=fs["legend"] - 1)
    out = fig_path("fig13_rcc_arm.png")
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
