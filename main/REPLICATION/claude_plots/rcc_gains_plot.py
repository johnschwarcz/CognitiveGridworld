"""Where and when the RCC gains over the factorized observer, and which component leads (11-10).
The full RCC under its original feedback (train_cluster_runs.py, the updated Generator), novel environments only, T = 30.

a  Accuracy by state type through training, main-text protocol (zflow_s0-2: |S| 500, batch 20,000, lr 1e-4; one test of
   20,000 episodes per 500 training episodes). High-FR states = cells where the factorized observer is inconsistent
   (Delta(r) > 0, FR grows linearly; ~47% of test episodes); the rest are the other states. Dashed: Naive Bayes, dotted:
   Joint Bayes, on the same episodes (mean over tests and runs).
   Last three tests, mean of 3 runs: high-FR .614 (Naive .138, Joint .714), other .743 (Naive .731, Joint .816) - the whole
   gain over Naive is on high-FR states. No factorized stage: at the test where overall accuracy equals Naive's (.45) the
   RCC is at .33 on high-FR states and .56 on the others (8 independent runs: .317-.343 / .550-.569). High-FR accuracy
   clears Naive's by .05 at 6k-9k episodes; the other states reach Naive's level at 23k-24k.
b  The Generator's learned likelihood against the Classifier that trained it: accuracy of exact Bayesian inference with the
   learned likelihood (plug-in; train_cluster_runs.plugin) against the Classifier's own accuracy, one line per run, one
   point per test (runs that log the plug-in: rcc30k_S500_s0, rcc30k_S250_s0 at Fig 13's settings; rcc_rl_S400_s0 at the
   main-text protocol; rccdiag_lr4_s* at batch 10,000, lr 1e-4). Above the diagonal = the learned likelihood supports
   better inference than the Classifier performs. Finished or furthest runs 11-10 03:30: plug-in above the Classifier at
   113 of 120 tests while the Classifier is between .15 and .60 (4 runs past .60; mean lead .09-.14).
   11-10 (John: does the lead's shape depend on FR?): b is split into high-FR states (middle) and the other states (right),
   9 runs (rccdiag_lr4 seeds 3-5 added; lr 1e-3 for Fig 13's settings, 1e-4 otherwise). Lead = plug-in - Classifier on that
   state type, mean over tests binned by the Classifier's overall accuracy:
       overall .10-.15 (plateau)  high -.004  other -.023
               .25-.35            high +.165  other +.132
               .35-.45            high +.219  other +.136
               .45-.55            high +.180  other +.087
               .55-.70            high +.072  other +.020
   End (last three tests): high-FR +.025 to +.057 in all 9 runs, other states -.009 to +.012. The lead peaks mid-training on
   both state types, is larger on high-FR states, and outlasts training only there. Runs trained only on high- or low-FR
   episodes (zflow_high / low / half_s0) did not log the plug-in and kept only their final nets, so their lead through
   training cannot be drawn.

11-10 11:50 (John: drop the first panels, one curve per FR bin, several bins): ONE panel, the lead (plug-in - Classifier)
against the stage of training (the Classifier's overall accuracy, bins of .05; each run's mean per bin, then the mean over
runs +- s.e.), one curve per FR group. Until the 60k RCCs (train_cluster_runs.py rcc60k_lr4_s0-5, which log every test
per FR fifth, FR_EDGES .28 / .65 / 1.5 / 4.5 nats) have data, the two state types of the 9 runs above are drawn.

    python rcc_gains_plot.py [--pull]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "cluster_runs", "jsonl")
MAIN = ("zflow_s0", "zflow_s1", "zflow_s2")
LEAD = (("rcc30k_S500_s0", "#999999", r"Fig 13's settings, lr $10^{-3}$"), ("rcc30k_S250_s0", "#999999", None),
        ("rcc_rl_S400_s0", "#e66101", r"main-text protocol, lr $10^{-4}$"),
        ("rccdiag_lr4_s1", "#2c7bb6", r"batch $10^4$, lr $10^{-4}$")) + tuple(
        (f"rccdiag_lr4_s{k}", "#2c7bb6", None) for k in (0, 2, 3, 4, 5))
HI, LO = "#c51b7d", "#276419"
NEW = tuple(f"rcc60k_lr4_s{k}" for k in range(6))      # 11-10: log the five FR fifths


def pull():
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/{{rcc,zflow_s}}*.jsonl' {LOGS}/", shell=True)


def load(name):
    f = os.path.join(LOGS, name + ".jsonl")
    if not os.path.exists(f):
        return None
    R = [json.loads(l) for l in open(f) if l.strip()]
    keys = sorted({k for r in R for k in r})
    return {k: np.array([np.nan if r.get(k) is None else r.get(k, np.nan) for r in R], float) for k in keys}


def groups():
    """(runs, [(label, colour, key suffix)], source text): the five FR fifths of the 60k RCC runs once two of them log
    them through the takeoff (two runs past Classifier accuracy .5; 11-10 14:10 the first two were at .13 / .22: one bin),
    else the two state types (high-FR / other) of the 30k runs"""
    new = [d for d in map(load, NEW) if d is not None and "plugin_fr0" in d and np.nanmax(d["net"], initial=0) > .5]
    if len(new) >= 2:
        cols = plt_cm(np.linspace(0, .9, 5))
        names = ("$<$ .28", ".28-.65", ".65-1.5", "1.5-4.5", "$>$ 4.5")
        return new, [(f"FR {n}", c, f"_fr{q}") for q, (n, c) in enumerate(zip(names, cols))], \
            f"{len(new)} RCCs at batch $10^4$, lr $10^{{-4}}$ (rcc60k), FR fifths of the test episodes (nats)"
    old = [d for d in (load(n) for n, _, _ in LEAD) if d is not None and np.isfinite(d["plugin"]).any()]
    return old, [("high-FR states", HI, "_high"), ("other states", LO, "_low")], \
        f"{len(old)} RCCs (Fig 13's settings at lr $10^{{-3}}$; main-text protocol, batch $10^4$ at lr $10^{{-4}}$)"


def plt_cm(x):
    import matplotlib.pyplot as plt
    return plt.cm.viridis(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    runs, grp, src = groups()
    edges = np.arange(.10, .751, .05)
    mid = (edges[:-1] + edges[1:]) / 2
    fig, a = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
    for lab, col, suf in grp:
        per_run = []                                     # each run's mean lead per stage bin, then the mean over runs
        for d in runs:
            b = np.digitize(d["net"], edges) - 1
            lead = d["plugin" + suf] - d["net" + suf]
            per_run.append([np.nanmean(lead[b == k]) if (b == k).any() else np.nan for k in range(len(mid))])
        P = np.array(per_run)
        n = np.isfinite(P).sum(0)
        m = np.where(n >= 2, np.nanmean(P, 0), np.nan)
        se = np.where(n >= 2, np.nanstd(P, 0) / np.sqrt(np.maximum(n, 1)), np.nan)
        a.plot(mid, m, color=col, lw=2, marker="o", ms=3, label=lab)
        a.fill_between(mid, m - se, m + se, color=col, alpha=.2, lw=0)
    a.axhline(0, color="k", lw=.9, ls="--")
    a.set(xlabel="stage of training: the Classifier's overall accuracy", xlim=(.1, .72),
          ylabel="lead of the learned likelihood\n" + r"$P(\hat r^{\,\mathrm{plug\text{-}in}}_g = r_g) - P(\hat r^{\,\mathrm{Clf}}_g = r_g)$")
    a.set_title(r"Bayes with the learned likelihood $\hat P(o \mid r)$ minus the Classifier, by FR", fontsize=fs["title"] - 4)
    a.legend(frameon=False, fontsize=fs["legend"] - 2, loc="upper right")
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    fig.suptitle(f"Novel environments, {src}\nmean over runs $\\pm$ s.e.; 10,000-20,000 episodes per test",
                 fontsize=fs["legend"] - 3)
    out = fig_path("rcc_gains.png")
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
