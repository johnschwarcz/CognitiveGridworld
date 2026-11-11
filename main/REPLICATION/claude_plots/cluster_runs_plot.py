"""Follow-up runs on the ELSC cluster (train_cluster_runs.py), original feedback (mode RL), |S| = 500, batch 20,000.

  (a) zflow_s0-2 (oracle-work Generator) against default_s0-2 (same settings and seeds, default Generator), once those
      exist: accuracy on novel environments through training.
  (b, c) the zflow Generator trained on every cell, on low-FR cells only (Delta(r) = 0: the Naive observer converges to
      the true goal value), on high-FR cells only (Delta(r) > 0), or on a fixed random half of each environment's cells;
      tested on all cells of novel environments, split by the test cell's type.

Each point is one test of 20,000 episodes on novel environments (both variables among the 50 held-out variables),
T = 30, taken every 500 episodes; test episodes train nothing. 2026-11-09 morning (mean of the last 3 tests, 30k ep):
                 low-FR cells   high-FR cells
  every cell     .74 / .74        .62 / .62     (zflow_s0 / s1)
  low only           .75              .41
  high only          .56              .60
  random half        .74              .60       (26.5k episodes)
  Joint / Naive  .82 / .73        .71 / .14

    python cluster_runs_plot.py [--pull]          # --pull copies the JSONL logs from the cluster first
"""
import os, sys, json, glob, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "cluster_runs", "jsonl")
# cancelled 11-09 16:10 at chance (John: replace the default seeds with something more valuable once the comparison
# is clear): x marks where they stopped
STOPPED = ("default_s0", "default_s2")


def pull():
    os.makedirs(LOGS, exist_ok=True)
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "elsc-cluster:cgw/runs/cluster_runs/*.jsonl", LOGS + "/"],
                   check=True)


def load():
    runs = {}
    for f in sorted(glob.glob(os.path.join(LOGS, "*.jsonl"))):
        L = [json.loads(l) for l in open(f)]
        if L:
            runs[os.path.basename(f)[:-6]] = {k: np.array([r[k] for r in L], float) for k in L[0]}
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

    def ref(key):
        return np.mean([r[key].mean() for n, r in runs.items() if n.startswith("zflow")])

    fig, ax = plt.subplots(1, 3, figsize=(3 * 2.9 + 1.4, 2.0 + 1.5), constrained_layout=True)

    a = ax[0]
    seeds = [n for n in runs if n[:-1] in ("zflow_s", "default_s")]
    for n in sorted(seeds):
        r = runs[n]
        zf = n.startswith("zflow")
        a.plot(r["ep"] / 1e3, r["net"], color="k" if zf else "#9a9a9a", lw=1.0)
        if not zf and n in STOPPED:
            a.plot(r["ep"][-1] / 1e3, r["net"][-1], "x", color="#9a9a9a", ms=5, mew=1.3)
    a.plot([], [], color="k", label=f"new Generator ({sum(n.startswith('zflow') for n in seeds)} seeds)")
    nd = sum(n.startswith("default") for n in seeds)
    a.plot([], [], color="#9a9a9a", label=f"default Generator ({nd} seeds)" if nd else
           "default Generator: running")
    if any(n in runs for n in STOPPED):
        a.plot([], [], "x", color="#9a9a9a", ms=5, mew=1.3, label="stopped at chance")
    a.set_title("Learning under the original feedback\n" + r"accuracy $\langle[\hat r_g = r_g]\rangle$, novel environments",
                fontsize=fs["title"] - 3)
    for k, lab in (("joint", "Joint"), ("naive", "Naive")):
        a.axhline(ref(k), color="k", lw=.7, ls="--" if k == "joint" else ":")
        a.text(30.5, ref(k), lab, va="center", fontsize=fs["legend"] - 1)
    a.legend(loc="upper left", fontsize=fs["legend"] - 2, frameon=False, handlelength=1.5, borderaxespad=.3)

    cond = [("all cells", "zflow_s", "k"), ("low-FR cells only", "zflow_low_s", "#2a6fdb"),
            ("high-FR cells only", "zflow_high_s", "#d1495b"), ("random half", "zflow_half_s", "#e8a33d")]
    for a, cell, title in ((ax[1], "low", "Low-FR test cells, $\\Delta(\\mathbf{r}) = 0$\n(Naive converges to the true value)"),
                           (ax[2], "high", "High-FR test cells, $\\Delta(\\mathbf{r}) > 0$\n(Naive converges to a wrong value)")):
        for lab, pre, c in cond:
            names = sorted(n for n in runs if n[:-1] == pre)
            for i, n in enumerate(names):
                r = runs[n]
                a.plot(r["ep"] / 1e3, r["net_" + cell], color=c, lw=1.1 if pre != "zflow_s" else .8,
                       label=(f"{lab} ({len(names)} seeds)" if len(names) > 1 else lab) if i == 0 else None)
        for k, lab in (("joint", "Joint"), ("naive", "Naive")):
            y = ref(f"{k}_{cell}")
            a.axhline(y, color="k", lw=.7, ls="--" if k == "joint" else ":")
            a.text(30.5, y, lab, va="center", fontsize=fs["legend"] - 1)
        a.set_title(title, fontsize=fs["title"] - 3)
    ax[2].legend(loc="upper left", fontsize=fs["legend"] - 2, frameon=False, ncol=2, title="trained on",
                 title_fontsize=fs["legend"] - 2, columnspacing=.8, handlelength=1.2)

    for a in ax:
        a.set_xlim(0, 30)
        a.set_ylim(0, 1.0)
        a.axhline(.1, color="k", lw=.5, alpha=.4)
        a.set_xlabel("training episodes (thousands; batch 20,000)")
    ax[0].set_ylabel("accuracy at T = 30")
    fig.suptitle(r"Original feedback, $|\mathcal{S}|$ = 500: one test of 20,000 novel-environment episodes every 500 episodes; "
                 "FR conditions 1 seed each", fontsize=fs["legend"])
    out = fig_path("cluster_runs_zflow_fr.png")
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
