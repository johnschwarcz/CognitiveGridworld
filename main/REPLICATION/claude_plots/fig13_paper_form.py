"""Every Generator-route vs Classifier-route run set in the paper's Fig 13 form (John 11-13: "show the plots in the same form
that is done in the paper"): Joint minus network accuracy at the final inference step, mean +- SEM over seeds, against the
testing episode on log-log axes (plot_ablation.py; the same cached Joint / Naive bounds, bayes_bounds_ctx2_T30.npy,
Joint .7685, Naive .4469). Every run tests once per 250 training episodes on 10,000 testing environments, so testing
episode k = 250 k training episodes; all axes shared. The mean +- SEM is drawn only where every seed of the set has a test.

    a  the submitted runs: DATA/oracle_pilot default_rep0-5 (Generator) / clf_reg_rep0-5 (Classifier); oracle feedback,
       lr 1e-3, 10,000 training episodes (41 tests)
    b  the same protocol re-run (Nov, cluster GPUs): gen30k_s0-3 / clf30k_s0-3; oracle feedback, lr 1e-3, the same seeds,
       continued to 30,000 episodes (seeds 4-5 never ran: held on the cluster)
    c  oracle feedback at lr 1e-4: gen60k_lr4_s0-5 / clf60k_lr4_s0-5, 60,000 episodes
    d  the RCC's own answers (original feedback) at lr 1e-4: rcc60k_lr4_s0-5 (60,000) / rccclf60k_lr4 -> rccclf120k_lr4_s0-5
       (continued to 120,000; s4/s5 still running)
Logs: DATA/experimental/cluster_runs/jsonl (--pull refreshes the running continuations from the cluster).

NUMBERS 11-13 (excess = .7685 - accuracy, mean +- SEM, at 10,000 episodes = test 40/41):
    a  Generator .071 +- .001 (n 6), Classifier .334 +- .053 (n 6)      (the paper's figure)
    b  Generator .075 +- .003 (n 4), Classifier .144 +- .017 (n 4)      same seeds as a's first four: submitted Classifier
       .505/.607/.510/.375 vs re-run .598/.610/.675/.615 accuracy; Generator reproduces (.69-.70 both)
    c  Generator .128 +- .001, Classifier .285 +- .016 (n 6 each); at 60,000: .043 +- .002 vs .092 +- .031 (s2 stalled at
       .52; without it .061 +- .003)
    d  at 10,000 both near chance; see fig13_routes.py for per-seed crossings (Generator beats Naive 3.5-4.6x sooner)

    python fig13_paper_form.py [--pull]
"""
import os, sys, json, glob, re, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path

DATA = os.path.join(ROOT, "main", "DATA")
LOGS = os.path.join(DATA, "experimental", "cluster_runs", "jsonl")
BLUE, RED = "#2c7bb6", "#d7191c"
PANELS = [
    ("a", r"Submitted (oracle, lr $10^{-3}$)", ("pth", "default"), ("pth", "clf_reg"), range(6)),
    ("b", r"Re-run, same seeds (oracle, lr $10^{-3}$)", (None, "gen30k"), (None, "clf30k"), range(4)),
    ("c", r"Oracle feedback, lr $10^{-4}$", (None, "gen60k_lr4"), (None, "clf60k_lr4"), range(6)),
    ("d", r"The RCC's own answers, lr $10^{-4}$", (None, "rcc60k_lr4"), ("rccclf60k_lr4", "rccclf120k_lr4"), range(6)),
]


def seed_curve(src, name, k):
    """Final-step accuracy per test; test 1 = before training (ep 0), test k = ep 250 (k - 1), as in the .pth files."""
    if src == "pth":
        import torch
        d = torch.load(os.path.join(DATA, "oracle_pilot", f"{name}_rep{k}_net.pth"), weights_only=False, map_location="cpu")
        return np.asarray(d["test_acc_through_training"], float)[:, -1]   # 41 tests, as plot_ablation.py
    for n in ([f"{name}_s{k}"] + ([f"{src}_s{k}"] if src else [])):
        f = os.path.join(LOGS, n + ".jsonl")
        if os.path.exists(f):
            rows, last = [], -10
            for r in sorted((json.loads(l.replace("NaN", "null")) for l in open(f) if l.strip()), key=lambda r: r["ep"]):
                if r["ep"] - last > 1:                      # drop a resume's duplicate test (59,999 then 60,000)
                    rows.append(r["net"])
                    last = r["ep"]
            return np.array(rows)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/rccclf120k_lr4_s*.jsonl' {LOGS}/",
                       shell=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    plt.rcParams.update({"font.family": "serif", "font.serif": "cmr10", "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": .6, "pdf.fonttype": 42})
    joint, naive = np.load(os.path.join(DATA, "cache", "bayes_bounds_ctx2_T30.npy"))
    fig, axs = plt.subplots(2, 2, figsize=(5.5, 4.3), sharex=True, sharey=True)
    for ax, (tag, title, gen, clf, seeds) in zip(axs.ravel(), PANELS):
        # the Generator line is drawn second, on top
        for (src, name), col, lab in ((clf, RED, "Classifier"), (gen, BLUE, "Generator")):
            A = [a for a in (seed_curve(src, name, k) for k in seeds) if a is not None]
            n = min(len(a) for a in A)
            G = joint - np.stack([a[:n] for a in A])
            x = np.arange(1, n + 1)
            m, se = G.mean(0), G.std(0, ddof=1) / np.sqrt(len(A))
            ax.fill_between(x, m - se, m + se, color=col, alpha=.22, lw=0)
            ax.plot(x, m, color=col, lw=1.3, label=f"{lab} ($n{{=}}{len(A)}$)")
            print(f"{tag} {lab:<10} n={len(A)} tests={n}  excess@10k {m[min(40, n - 1)]:.3f}+-{se[min(40, n - 1)]:.3f}"
                  f"  end {m[-1]:.3f}+-{se[-1]:.3f}")
        ax.axhline(joint - naive, c="k", lw=1, ls="--", zorder=10)
        ax.set(xscale="log", yscale="log", xlim=(1, 481), ylim=(.03, .75))
        ax.set_title(r"$\mathbf{%s}$   %s" % (tag, title), fontsize=8, loc="left")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(mticker.LogLocator(subs=(1., 2., 5.)))
            axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(alpha=.25, lw=.5, which="both")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        leg = ax.legend(loc="lower left", frameon=False, title=r"$\hat{\mathcal{E}}$ receives gradients from",
                        title_fontsize=7, labelspacing=.3, handlelength=1.6)
        leg._legend_box.align = "left"
    axs[0, 0].text(1.2, joint - naive, "Naive bound", fontsize=7, va="center", ha="left", zorder=11,
                   bbox=dict(fc="white", ec="none", alpha=.85, pad=1))
    for ax in axs[1]:
        ax.set_xlabel("Testing Episode (1 per 250 training episodes)")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"Optimal $-$ Network Accuracy")
    fig.tight_layout()
    out = fig_path("fig13_paper_form.png")
    fig.savefig(out, dpi=300)
    print(out)


if __name__ == "__main__":
    main()
