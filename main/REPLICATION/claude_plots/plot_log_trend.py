"""Gain over naive Bayes through training, log time, against the best linear-in-counts rules (John 2026-11-09: "does the
trend become linear [in log scale] and we just need tremendous amounts of data?", then "show log_trend with the best linear
rules ... a new top row for the trained, and show all 3 columns"; then "show all ESNs and the fully trained networks that
best match each one's trainable params", no linear rules on the trained; then "go back to showing all networks in each
panel (picking the best matched ones)"). Rows: trained LSTMs, echo-state networks (ESNs).
Top row: every fully trained network (N 50 to 1k); the one whose trainable parameters are nearest on a log scale to each
ESN of the column is tagged "~ Echo ..." in the legend (marginal: Echo 200 / 500 / 1k (7.6k / 19k / 38k) -> Trained 50
(22k; none smaller exists), Echo 10k (380k) -> Trained 200 (329k); joint: Echo 200 (24k) -> Trained 50 (26k), Echo 500 /
1k (59k / 118k) -> Trained 100 (93k), Echo 2k (236k) -> Trained 150 (199k), Echo 10k (1.18M) -> Trained 400 (1.33M)); the
legends give the counts.
Columns as compare_matched_params_loglog_joint_panels: marginal output with goal supervision, joint output with goal
supervision, joint output with joint supervision. Training-dynamics runs, seed 20261029, batch 8,000; every test of a
500-episode block (125 tests x 8,000 episodes), T = 30, the network minus naive Bayes on the same episodes (all runs share
them). Dashed: a(t) = a_inf - c (t / 1k)^-alpha fitted from 2k on (a_inf <= optimal), extrapolated to 1M, drawn only if it
leaves under 1% of the variance of the data it was fitted to unexplained (11-09: drops trained N 50 in every column and
N 100 in both joint ones; no ESN fit); band: the fits from 1k, 2k and 4k that pass the same test. The paper's Echo N = 2k
(marginal) is not a training-dynamics run (one test per block, no paired naive): not drawn. The linear rules (grey) are
on the ESN row only (John: not needed for the trained).

The best linear rules (logits linear in the five counts, fitted per environment):
  joint output: the Bayes posterior's logits are exactly linear in the counts (each cell (g, c): sum_i n_i x_i(g, c) minus
    T sum_i softplus x_i(g, c)), so the best linear rule IS optimal; and under goal supervision every loss-optimal joint rule
    has Bayes's goal marginal, so the same. Exact: optimal - naive = +.3207 on these test episodes.
  marginal output (goal logits linear in the counts): naive Bayes is one such rule, not the best.
    best under cross-entropy (convex, unique optimum): .5049 +- .0013 vs naive .4495 on the environment's episodes, +.0554
      (50,000 episodes per pair, Newton to |grad| < 1e-10; 64,000 vs 16,000 per pair moves it .0008).
    most accurate found: .5552 +- .0013, +.1057 (best of CE-, naive- and restart-started accuracy refinements). Maximising
      accuracy is non-convex: this is a lower bound on the most accurate linear rule (per-pair best of everything, 13
      exactly scored pairs, puts it at .563-.567); no upper bound below optimal is known. NOT DRAWN (John 11-09: "keep the
      provable cross-entropy best linear and remove the other"); so the drawn line is the family's best belief, not an
      accuracy ceiling. Naive Bayes is one rule of the same family (its logits are linear in the counts too).
  Source: DATA/experimental/ff_deep/extrapolation/naive_vs_linear/verify_2026-10-25/ceiling (log_env_score_sub50k.txt,
  report.py).
-> main/REPLICATION/figure_bin/log_trend.png"""
import glob
import json
import os
import sys
import numpy as np
from scipy.optimize import curve_fit

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
import compare_matched_params as cmp                                              # noqa: E402
import plot_matched_params_new as pm                                              # noqa: E402
import plot_matched_params_joint as pj                                            # noqa: E402

D = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
LIN_CE = .5049 - .4495                          # marginal output: the best rule linear in the counts under cross-entropy
GREY = "#555555"
FVU_MAX = .01                                   # extrapolate only fits leaving < 1% of the data variance unexplained
BOX = dict(fc="white", ec="none", alpha=.85, pad=1.5)          # labels stay readable where a curve crosses


def blocks(run, block=500):
    """(training episodes at each complete block's end, accuracy, naive accuracy, optimal accuracy): every test of the
    block, T = 30"""
    d = os.path.join(D, run + "_s20261029")
    every = json.load(open(os.path.join(d, "run.json")))["cfg"]["checkpoint_every"]
    z = [np.load(f) for f in sorted(glob.glob(os.path.join(d, "test_*.npz")))]
    ep = np.concatenate([q["episode"] for q in z])
    a, n, j = (np.concatenate([q[k][:, -1] for q in z]).astype(float) for k in ("acc", "naive_acc", "joint_acc"))
    k = np.ceil(np.maximum(ep - ep // every, 1) / block).astype(int)
    ks = np.unique(k)
    ks = ks[(np.bincount(k)[ks] >= block // (every - 1)) & (ks * block <= pm.XMAX)]   # complete blocks, to 40k
    return (ks * block,) + tuple(np.array([v[k == u].mean() for u in ks]) for v in (a, n, j))


def sat(t, a_inf, c, alpha):
    return a_inf - c * (t / 1e3) ** (-alpha)


def fits(x, acc, top):
    """the saturating fits from 1k, 2k and 4k: (parameters, fraction of the data's variance they leave unexplained, over the
    range each was fitted to)"""
    out = []
    for lo in (1000, 2000, 4000):
        m = x >= lo
        p = curve_fit(sat, x[m], acc[m], p0=(min(acc[m][-1] + .01, top - 1e-3), .05, .5),
                      bounds=([0, 0, .01], [top, 10, 5]), maxfev=20000)[0]
        out.append((p, (acc[m] - sat(x[m], *p)).var() / acc[m].var()))
    return out                                                                    # [from 1k, from 2k, from 4k]


def name_of(kind, n):
    w = f"{n // 1000}k" if n >= 1000 else f"{n}"
    return rf"{kind} ($N{{=}}{w}$)"


def figure():
    from main.utils import fig_path
    cmp.apply_fig_style()
    nets = cmp.resolve_colors([dict(n) for n in cmp.NETWORKS if n["stem"] not in pm.DROP] + [dict(pm.ESN2K)])
    c1k = next(n["color"] for n in nets if n["stem"] == "sanity/reservoir_ctx_2")
    c2k = next((n["color"] for n in nets if n["stem"] == pm.ESN2K["stem"]), cmp.C_ECHO)
    c10k = next(n["color"] for n in nets if "10000N" in n["stem"])
    blues = ((50, pj.C_N50), (100, pm.C_N100), (150, pj.C_N150), (200, cmp.C_TRAIN), (300, pj.C_N300), (400, pj.C_N400),
             (500, pj.C_N500), (1000, pj.C_N1K))
    reds = {200: pj.C_E200, 500: pj.C_E500, 1000: c1k, 2000: c2k, 10000: c10k}
    w_run = lambda prefix, n, suffix: prefix + ("" if n == 1000 else f"_N{n}") + suffix
    cols = (("Marginal output, goal supervision", "trained", "", "echo", "", (200, 500, 1000, 10000), "_b8k"),
            ("Joint output, goal supervision", "joint_trained", "", "joint_echo", "", (200, 500, 1000, 2000, 10000), "_b8k"),
            ("Joint output, joint supervision", "joint_trained", "_jl", "joint_echo", "_jl", (200, 500, 1000, 2000, 10000), "_b8k"))
    fig, axes = plt.subplots(2, 3, figsize=(30, 11.5), sharey=True)
    tt = np.logspace(np.log10(2000), 6, 200)
    opt_edge = []
    wid = lambda n: f"{n // 1000}k" if n >= 1000 else f"{n}"
    for col, (title, tr, tr_sfx, es, es_sfx, es_widths, big) in enumerate(cols):
        esn = [((w_run(es, n, "") + big + es_sfx) if n == 10000 else w_run(es, n, es_sfx), n) for n in es_widths]
        esn_p = {n: pj.params(run) for run, n in esn}
        cand = {n: pj.params(w_run(tr, n, tr_sfx)) for n, _ in blues}
        match = {}                                                                # trained width -> the ESN widths it matches
        for n in es_widths:
            match.setdefault(min(cand, key=lambda m: abs(np.log(cand[m] / esn_p[n]))), []).append(n)
        tag = lambda n: (r" $\approx$ Echo " + ", ".join(wid(e) for e in match[n])) if n in match else ""
        rows = ((axes[0, col], [(w_run(tr, n, tr_sfx), name_of("Trained", n) + f", {pj.count(cand[n])} params" + tag(n), c, n)
                                for n, c in blues]),
                (axes[1, col], [(run, name_of("Echo", n) + f", {pj.count(esn_p[n])} params", reds[n], n) for run, n in esn]))
        for ax, runs in rows:
            for run, name, color, n in runs:
                x, acc, nv, jv = blocks(run)
                opt_edge.append((jv - nv).mean())
                if not pm.finished(run):
                    name += f" (training: {x[-1] / 1e3:.1f}k of 40k)"
                f = fits(x, acc, jv[-20:].mean())
                ax.plot(x, acc - nv, color=color, lw=2.2, label=name, zorder=3 + n / 1e4)
                ok = [p for p, fvu in f if fvu < FVU_MAX]                       # fits that explain the data they were fitted to
                if f[1][1] < FVU_MAX:                                            # the 2k fit is the dashed line
                    curves = np.array([sat(tt, *p) - nv[-20:].mean() for p in ok])
                    ax.plot(tt, sat(tt, *f[1][0]) - nv[-20:].mean(), color=color, lw=1.6, ls="--", zorder=2 + n / 1e4)
                    ax.fill_between(tt, curves.min(0), curves.max(0), color=color, alpha=.15, lw=0, zorder=1)
                print(f"  {run:26s} gain {acc[-1] - nv[-1]:+.4f} at {x[-1]}; a_inf from 1k/2k/4k "
                      + " / ".join(f"{p[0]:.3f} (unexpl. {fvu:.4f})" for p, fvu in f) + ("" if f[1][1] < FVU_MAX else "  -> NOT EXTRAPOLATED"))
            ax.legend(loc="lower right", fontsize=12.5, frameon=False, labelspacing=.22 if len(runs) > 5 else .5, borderaxespad=.3)
        axes[0, col].set_title(title, fontsize=15)
    opt = float(np.mean(opt_edge))
    for col in range(3):
        for row in range(2):
            ax = axes[row, col]
            ax.axhline(0, color="k", lw=2, ls="--", zorder=4)
            ax.axhline(opt, color="k", lw=2, ls="-.", zorder=4)
            lab = row == 1                                                  # the ESN row has room on the left
            if row == 1:                                                    # the best linear-in-counts rule: ESN row only
                ax.axhline(LIN_CE, color=GREY, lw=1.8, ls=":", zorder=4)
            if col == 0 and lab:
                ax.text(1.12e3, LIN_CE, "best linear-in-counts rule for marginal output (cross-entropy optimum)", fontsize=13.5,
                        va="bottom", color=GREY, bbox=BOX, zorder=12)
            if lab:
                ax.text(1.12e3, 0, "naive Bayes", fontsize=13.5, va="bottom", bbox=BOX, zorder=12)
                ax.text(1.12e3, opt, "optimal" if col == 0 else "optimal = best linear rule (joint output, exact)",
                        fontsize=13.5, va="bottom", bbox=BOX, zorder=12)
            ax.set_xscale("log")
            ax.set_xlim(1e3, 1e6)
            ax.set_ylim(-.33, .37)
            ax.grid(alpha=.25, lw=.6, which="both")
            if row == 1:
                ax.set_xlabel("Training Episode")
    axes[0, 0].set_ylabel("Trained: gain over naive,\n" + r"$a_{net}-a_{naive}$")
    axes[1, 0].set_ylabel("Echo: gain over naive,\n" + r"$a_{net}-a_{naive}$")
    fig.text(.5, .005, "solid: data; dashed: saturating fit from 2k, extrapolated to 1M, drawn only where it leaves under 1% of the "
             "variance of the data it was fitted to unexplained; band: the fits from 1k / 2k / 4k that do.\n"
             r"top: every fully trained network; ``$\approx$ Echo'' marks the one nearest in trainable parameters to each ESN of the column.   grey (ESN row): the best marginal-output "
             "rule linear in the counts (cross-entropy; naive Bayes is one such rule), labelled in the first column.   No marginal-output Echo 2k: it has no run with these tests.\n"
             r"1 seed, batch 8,000; each point: 125 tests $\times$ 8,000 episodes per 500-episode block, T = 30",
             ha="center", va="bottom", fontsize=13.5)
    fig.tight_layout(rect=(0, .06, 1, 1))
    path = fig_path("log_trend.png")
    fig.savefig(path, dpi=150)
    print(f"  optimal - naive {opt:+.4f}; best linear-in-counts rule (cross-entropy) {LIN_CE:+.4f}")
    print("saved", path)


if __name__ == "__main__":
    figure()
