"""Hourly progress of the Var(z) test (train_bayes_generator.py tasks 123-252, run from ~/cgw_zscale on the cluster): normalized
novel-environment transfer n = (plugin - zblind) / (joint - zblind) against training episodes, one panel per (d_E, |S|) cell
in the layout of the d_E x |S| heat maps, one line per run (colour = arm, each seed its own line). n = 0 is a Generator that
ignores z_hat, n = 1 is Joint Bayes; the hairline at .5 is the pre-registered transfer threshold. "H1" / "H2" mark the cell
where each hypothesis puts the first transferring |S| under scaling (claude_plots/zscale_table.py). At d_E 30 the scale
factor is 1, so its unscaled runs stand for every arm.

Prints a status table (latest n per cell and arm, runs started / finished / stalled) for the hourly report.

    python zscale_progress.py [--pull] [--logs DIR] [--out PATH]
--pull: fetch the logs from ~/cgw_zscale/runs/bayes_gen with cluster.sh (the runs live apart from ~/cgw).
"""
import os, re, sys, glob, json, time, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "bayes_gen_zscale")
REMOTE = "cgw_zscale/runs/bayes_gen/./joint_S*_d*_s*_z*.jsonl"
S_GRID, DIMS, N_RUNS, EPISODES = (50, 75, 100, 125, 250, 500, 1000), (10, 20, 30, 90), 130, 8000
ARMS = (("env", "z scaled: environment", "#2a78d6"), ("both", "z scaled: environment + network", "#eb6834"),
        ("none", "unscaled", "#1baf7a"))                       # categorical slots 1-3, validated all-pairs
H1 = {10: 50, 20: 75, 30: 100, 90: 500}
H2 = {10: 100, 20: 100, 30: 100, 90: 250}
STALL_H = 1.5                                                  # no new test for this long: likely dead or preempted
SURFACE, INK, INK2, HAIR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def pull(logs):
    os.makedirs(logs, exist_ok=True)
    r = subprocess.run(["bash", os.path.join(ROOT, "main", "REPLICATION", "trainings", "cluster.sh"), "pull", REMOTE, logs],
                       capture_output=True, text=True)
    if r.returncode:
        print(f"pull failed (using the logs already here): {(r.stderr or r.stdout).strip()[:300]}")


def load(logs):
    """{(arm, d_E, |S|, seed): (episodes, n, mtime)}; a half-written last line (a run mid-write) is skipped"""
    runs = {}
    for f in glob.glob(os.path.join(logs, "joint_S*_d*_s*_z*.jsonl")):
        m = re.search(r"joint_S(\d+)_d(\d+)_s(\d+)_z(none|env|both)\.jsonl$", f)
        if not m:
            continue
        ep, n = [], []
        for line in open(f):
            try:
                r = json.loads(line)
                ep.append(r["ep"])
                n.append((r["novel_plugin"] - r["novel_zblind"]) / max(r["novel_joint"] - r["novel_zblind"], 1e-6))
            except (ValueError, KeyError):
                continue
        if ep:
            runs[m[4], int(m[2]), int(m[1]), int(m[3])] = (np.array(ep), np.array(n), os.path.getmtime(f))
    return runs


def status(runs):
    now = time.time()
    done = [k for k, v in runs.items() if v[0][-1] >= EPISODES]
    stalled = sorted(k for k, v in runs.items() if v[0][-1] < EPISODES and now - v[2] > STALL_H * 3600)
    eps = [v[0][-1] for v in runs.values()]
    print(f"runs started {len(runs)}/{N_RUNS}, finished {len(done)}, median latest episode "
          f"{int(np.median(eps)) if eps else 0}; stalled (no test in {STALL_H} h): "
          + (", ".join(f"{a} d{d} S{S} s{s}" for a, d, S, s in stalled) or "none"))
    print("latest n (mean over seeds; '*' = .5 or more), by arm and d_E, for |S| = " + " ".join(map(str, S_GRID)))
    for arm, _, _ in ARMS:
        for d in DIMS:
            src = "none" if d == 30 else arm
            cells = []
            for S in S_GRID:
                v = [r[1][-1] for k, r in runs.items() if k[:3] == (src, d, S)]
                cells.append(f"{np.mean(v):5.2f}{'*' if np.mean(v) >= .5 else ' '}" if v else "    - ")
            print(f"  {arm:4s} d_E {d:2d}: " + " ".join(cells))


def plot(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import apply_fig_style
    fs = apply_fig_style()
    fig, ax = plt.subplots(len(DIMS), len(S_GRID), figsize=(15.5, 8.2), sharex=True, sharey=True,
                           constrained_layout=True, facecolor=SURFACE)
    for i, d in enumerate(DIMS):
        for j, S in enumerate(S_GRID):
            a = ax[i, j]
            a.set_facecolor(SURFACE)
            for y in (0, .5, 1):
                a.axhline(y, color=INK2 if y == .5 else HAIR, lw=.8 if y == .5 else .6, zorder=0)
            drawn = False
            for arm, _, col in ARMS:
                for (ka, kd, kS, _), (ep, n, _) in sorted(runs.items()):
                    if (ka, kd, kS) == (arm, d, S):
                        keep = ep > 0                          # log axis: the test before training is off-axis
                        a.plot(ep[keep], n[keep], color=col, lw=1.6, solid_capstyle="round")
                        drawn = True
            tags = [h for h, p in (("H1", H1), ("H2", H2)) if p[d] == S]
            if tags:
                a.text(.04, .93, " ".join(tags), transform=a.transAxes, ha="left", va="top", color=INK2,
                       fontsize=fs["annot"])
            if not drawn:
                a.text(.5, .5, "not started", transform=a.transAxes, ha="center", va="center", color=INK2,
                       fontsize=fs["tick"])
            a.set_xscale("log")
            a.set_xlim(20, 10000)
            a.set_ylim(-.2, 1.15)
            a.set_xticks([100, 1000, 8000], ["100", "1k", "8k"])
            a.set_yticks([0, .5, 1])
            a.tick_params(colors=INK2, length=2)
            for s in a.spines.values():
                s.set_color(HAIR)
            if i == 0:
                a.set_title(rf"$|\mathcal{{S}}| = {S}$", fontsize=fs["label"] + 1, color=INK)
            if j == 0:
                a.set_ylabel(rf"$d_E = {d}$" + ("\n(factor 1: all arms)" if d == 30 else "") + "\n$n$", color=INK)
            if i == len(DIMS) - 1:
                a.set_xlabel("training episodes", color=INK2)
    started, done = len(runs), sum(v[0][-1] >= EPISODES for v in runs.values())
    handles = [Line2D([], [], color=c, lw=2, label=l) for _, l, c in ARMS]
    handles.append(Line2D([], [], color=INK2, lw=.8, label="transfer threshold, n = .5"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=fs["legend"])
    fig.suptitle(rf"Var(z) test: novel-environment transfer $n$ = (plug-in $-$ z-blind) / (Joint $-$ z-blind)"
                 f"\n{started}/{N_RUNS} runs started, {done} finished, updated "
                 f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", fontsize=fs["label"] + 2, color=INK)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--logs", default=LOGS)
    ap.add_argument("--out", default=os.path.join(ROOT, "main", "REPLICATION", "figure_bin", "zscale_progress.png"))
    a = ap.parse_args()
    if a.pull:
        pull(a.logs)
    runs = load(a.logs)
    status(runs)
    plot(runs, a.out)


if __name__ == "__main__":
    main()
