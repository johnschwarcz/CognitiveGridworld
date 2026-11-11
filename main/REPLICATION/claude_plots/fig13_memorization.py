"""Does the Classifier route memorize more? (John 11-11: "How do you see the classifier memorizes more? Showing the ratio of
memorization to generalization could be good.")

Training-environment accuracy against novel-environment accuracy through training, one line per seed (Fig 13's 60k
batch, lr 1e-4; logs of fig13_rcc_arm.py). On the diagonal a network does as well on environments it never saw as on
the ones it trains on (all generalization); the height above the diagonal is memorization, read in accuracy units.
Drawn as the pair instead of their ratio: (training - novel) / (novel - chance) divides by ~0 while novel accuracy is
near chance, so it would be dominated by the start of training.
training accuracy = the Classifier on the last training batch before each test (before its update), novel = the test
(10,000 episodes of novel environments).

RESULT 11-11 12:40 (end of training, training - novel):
    oracle feedback     Generator route +.001 to +.004 (4 seeds); Classifier route +.004 to +.045 (6 seeds, mean +.016)
    original feedback   Generator route (the RCC) +.004 to +.011 (6); Classifier route +.019 to +.051 (4)
A small effect at the end (one to five points); larger DURING learning: at novel accuracy .15-.45 the Classifier route
runs .03-.06 above the diagonal in every seed, the Generator route within ~.01 of it.

11-11 13:50 binned at equal competence (novel accuracy bins of .05, mean over seeds):
    training - novel   oracle: Generator route -.003 to +.009, Classifier route +.042 (at .10-.20) falling to +.003 (.70-.75)
                       original: the RCC +.003 to +.025, Classifier route +.06 (.15-.20) falling to +.026 (.55-.60)
    2 novel / (novel + training)   Generator route .98-1.00 (oracle), .93-1.00 (RCC); Classifier route .86 -> 1.00
                       (oracle), .84 -> .98 (original): the Classifier route is below the Generator route at every stage
                       up to ~.6, by more than the seeds' SEM

    python fig13_memorization.py
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "cluster_runs", "jsonl")
PANELS = (("a   Oracle feedback", (("gen60k_lr4", "Generator route", "#08519c"), ("clf60k_lr4", "Classifier route", "#a50f15"))),
          ("b   Original feedback", (("rcc60k_lr4", "Generator route (the RCC)", "#4fa3e0"),
                                     ("rccclf60k_lr4", "Classifier route", "#f4845f"))))


def runs(arm):
    out = []
    for k in range(6):
        f = os.path.join(LOGS, f"{arm}_s{k}.jsonl")
        if os.path.exists(f):
            r = [json.loads(l) for l in open(f) if l.strip()]
            r = [x for x in r if "train_net" in x]
            out.append((np.array([x["net"] for x in r]), np.array([x["train_net"] for x in r]), r[-1]["ep"]))
    return out


def binned(nov, y, edges):
    """mean of y per bin of novel accuracy (nan where the run has no test in the bin)"""
    b = np.digitize(nov, edges) - 1
    return np.array([y[b == k].mean() if (b == k).any() else np.nan for k in range(len(edges) - 1)])


def main():
    """John 11-11 13:30: "This generalization gap is nice but seems very small. Maybe plotting 2 * test / (test + train)
    would show it better or something?" -> 2 x 2: rows = training - novel (accuracy points) and 2 novel / (novel +
    training) (1 = all generalization), columns = oracle / original feedback; x = novel accuracy in bins of .05 (the
    stage of learning, so the routes are compared at equal competence despite their different speeds); thin = seeds
    (bin means of their tests), thick = mean over the seeds that have the bin, band = SEM where >= 2 do;
    bins of .01 (John 11-11 14:30), lines join populated bins"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    edges = np.arange(.10, .7501, .01)                                   # John 11-11: bins of .01
    mid = (edges[:-1] + edges[1:]) / 2
    rows = (("training $-$ novel accuracy", lambda nov, tr: tr - nov, 0),
            (r"$2\,$novel / (novel + training)", lambda nov, tr: 2 * nov / (nov + tr), 1))
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 6.6), sharex=True, sharey="row", constrained_layout=True)
    for j, (title, arms) in enumerate(PANELS):
        for i, (ylab, f, ref) in enumerate(rows):
            a = ax[i, j]
            hs = []
            for arm, name, col in arms:
                R = runs(arm)
                Y = np.array([binned(nov, f(nov, tr), edges) for nov, tr, _ in R])
                for y in Y:                                      # lines join the bins a seed has tests in
                    ok = np.isfinite(y)
                    a.plot(mid[ok], y[ok], color=col, lw=.6, alpha=.4)
                n = np.isfinite(Y).sum(0)
                m = np.where(n >= 1, np.nanmean(Y, 0), np.nan)
                se = np.where(n >= 2, np.nanstd(Y, 0, ddof=1) / np.sqrt(np.maximum(n, 1)), np.nan)
                ok = np.isfinite(se)
                a.fill_between(mid[ok], (m - se)[ok], (m + se)[ok], color=col, alpha=.25, lw=0)
                ok = np.isfinite(m)
                a.plot(mid[ok], m[ok], color=col, lw=1.8, marker="o", ms=2.2)
                hs.append(plt.Line2D([], [], color=col, lw=2.0, label=f"{name} ({len(R)} seeds)"))
            a.axhline(ref, color="k", lw=.8, ls="--", zorder=1)
            a.spines[["top", "right"]].set_visible(False)
            if i == 0:
                a.set_title(title, loc="left", fontsize=fs["title"] - 3)
                a.legend(handles=hs, loc="upper right", frameon=False, fontsize=fs["legend"] - 2)
            if j == 0:
                a.set_ylabel(ylab)
            if i == 1:
                a.set_xlabel("stage of learning: novel-environment accuracy (bins of .01)")
    fig.suptitle("Memorization at equal competence: training against novel accuracy, Fig 13's 60k batch (lr $10^{-4}$, "
                 r"$|\mathcal{S}| = 500$, batch 10,000)" + "\nthin = seeds (bin means of their tests), thick = mean "
                 r"$\pm$ SEM over seeds; lines join populated bins (the Generator route crosses .15-.55 in few tests); "
                 "dashed = none; 10,000 novel episodes per test",
                 fontsize=fs["legend"] - 2)
    out = fig_path("fig13_memorization.png")
    fig.savefig(out, dpi=200)
    print(out)
    for title, arms in PANELS:
        for arm, name, _ in arms:
            R = runs(arm)
            G = np.array([binned(nov, tr - nov, edges) for nov, tr, _ in R])
            Q = np.array([binned(nov, 2 * nov / (nov + tr), edges) for nov, tr, _ in R])
            print(f"{title[4:]:18s} {name:26s} gap by bin:", np.round(np.nanmean(G, 0), 3).tolist())
            print(f"{'':18s} {'':26s} ratio by bin:", np.round(np.nanmean(Q, 0), 3).tolist())


if __name__ == "__main__":
    main()
