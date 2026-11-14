"""Hourly progress of the Var(z) test (train_bayes_generator.py tasks 123-252, run from ~/cgw_zscale on the cluster): normalized
novel-environment transfer n = (plugin - zblind) / (joint - zblind) against training episodes, one panel per (d_E, |S|) cell
in the layout of the d_E x |S| heat maps, one line per run (colour = arm, each seed its own line). n = 0 is a Generator that
ignores z_hat, n = 1 is Joint Bayes; the hairline at .5 is the pre-registered transfer threshold. At d_E 30 the scale
factor is 1, so its unscaled runs stand for every arm.

Analytic thresholds (11-14, John: "put analytical predictions on this figure"). A thick vertical line on a panel's left edge:
the law puts |S|* for that colour's z just left of this panel, so its runs should transfer from here rightwards (on the
last panel's right edge: beyond the grid). A tinted panel lies inside the threshold's uncertainty. The law has one anchor,
the d_E 30 row (no transfer at |S| 75, transfer at 100: 3 seeds, bayes_gen_arch_plot.py), and no other constant:
    |S|*(d_E, Var z) = (d_E / 30) |S|*_30 R(Var z) / R(1/30),      R(v) = (e^{8 pi^2 v} - 1) / (8 pi^2 v) - 1
R = fold / linear variance of a product of two sinusoids of z ~ N(0, v), the likelihood's logit (uniform phases; the exact
likelihood's ratio is within 10% wherever it decides a cell). Scaled runs have Var z = 1/30 at every d_E, so their
|S|* = (d_E / 30) |S|*_30: the theory chat's H1; "H2" marks the cell where its alternative (a Var z-only barrier near 100
plus spanning) puts the first transfer. Unscaled d_E 90 falls below the spanning bound and is FITTED to its own row (partial
at 125, transfers at 250), not predicted. Not covered: an upper limit on |S| (the law says unscaled d_E 20 transfers at
1000; it does not) and the time course (no formula for the onset or the plateau).

Prints a status table (latest n per cell and arm, runs started / finished / stalled, the law's tally) for the hourly report.

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
H2 = {10: 100, 20: 100, 30: 100, 90: 250}
STALL_H = 1.5                                                  # no new test for this long: likely dead or preempted
SURFACE, INK, INK2, HAIR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ANCHOR = (75., 100.)                                           # |S|* at d_E 30: above 75, at most 100
SPAN_90 = (125., 250.)                                         # unscaled d_E 90: spanning-limited, fitted to its row
TALLY_FROM = 3000                                              # episodes before a run counts for or against the law
LAW = (r"analytic threshold, anchored on $d_E = 30$ ($75 < |\mathcal{S}|^* \leq 100$):   "
       r"$|\mathcal{S}|^* = (d_E / 30)\ |\mathcal{S}|^*_{30}\ R(\mathrm{Var}\,z)\,/\,R(1/30)$,   "
       r"$R(v) = (e^{8 \pi^2 v} - 1)\,/\,(8 \pi^2 v) - 1$;   unscaled $d_E = 90$: spanning-limited, fitted to its row")


def fold_ratio(v):
    """fold / linear variance of sin(2 pi z01 - a) sin(2 pi z10 - b), z ~ N(0, v), phases uniform"""
    x = 8 * np.pi ** 2 * v
    return np.expm1(x) / x - 1


def threshold(d, scaled):
    """((lo, hi), kind): the law puts |S|* above lo and at most hi; kind = anchor | predicted | fitted"""
    if d == 30:
        return ANCHOR, "anchor"
    osc = tuple(a * d / 30 * (1 if scaled else fold_ratio(1 / d) / fold_ratio(1 / 30)) for a in ANCHOR)
    if scaled:
        return osc, "predicted"
    span = tuple(a * d / 90 for a in SPAN_90)
    return (span, "fitted") if span[1] >= osc[1] else ((max(osc[0], span[0]), osc[1]), "predicted")


def predicted(d, scaled, S):
    """does the law say this cell transfers? True / False / None (inside the threshold's uncertainty)"""
    (lo, hi), _ = threshold(d, scaled)
    return True if S >= hi else False if S <= lo else None


def band_text(d, scaled, math=False):
    (lo, hi), kind = threshold(d, scaled)
    num = lambda x: f"{x / 1000:.1f}k" if x >= 1000 else f"{x:.0f}"
    who = "" if d == 30 else "scaled " if scaled else "unscaled "
    return (who + (r"$|\mathcal{S}|^*$ " if math else "") + f"{num(lo)}-{num(hi)}"
            + ("" if kind == "predicted" else f" ({kind})"))


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


def tally(runs):
    """runs past TALLY_FROM episodes against the law, by the kind of their row's threshold:
    {kind: [matches, cells]}, the misses, and the runs in cells the law leaves open"""
    score, miss, open_ = {}, [], []
    for (arm, d, S, seed), (ep, n, _) in sorted(runs.items()):
        if ep[-1] < TALLY_FROM:
            continue
        scaled = arm != "none"
        p, kind, name = predicted(d, scaled, S), threshold(d, scaled)[1], f"{arm} d{d} S{S} s{seed} (n {n[-1]:.2f})"
        if p is None:
            open_.append(name)
            continue
        s = score.setdefault(kind, [0, 0])
        s[1] += 1
        if (n[-1] >= .5) == p:
            s[0] += 1
        else:
            miss.append(name + f", law: {'transfer' if p else 'no transfer'}")
    return score, miss, open_


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
    print("analytic |S|* (anchor: d_E 30, 75-100): "
          + "; ".join(f"d_E {d} " + ", ".join(band_text(d, sc) for sc in (True, False)) for d in DIMS if d != 30))
    score, miss, open_ = tally(runs)
    print(f"law against the runs past {TALLY_FROM} episodes: "
          + ", ".join(f"{k} cells {a}/{b}" for k, (a, b) in sorted(score.items()))
          + f"; misses: {'; '.join(miss) or 'none'}; left open by the law: {'; '.join(open_) or 'none'}")


def plot(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from main.utils import apply_fig_style
    fs = apply_fig_style()
    colour = {arm: c for arm, _, c in ARMS}
    fig, ax = plt.subplots(len(DIMS), len(S_GRID), figsize=(15.5, 8.6), sharex=True, sharey=True,
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
            if not drawn:
                a.text(.5, .5, "not started", transform=a.transAxes, ha="center", va="center", color=INK2,
                       fontsize=fs["tick"])
            a.set_xscale("log")
            a.set_xlim(20, 10000)
            a.set_ylim(-.2, 1.25)                              # headroom for the threshold notes
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

    # the law's thresholds: a line on the left edge of the first panel that should transfer (right edge of the last one if
    # none should), a tint where the law cannot say; one line of text per threshold, top left, where no curve runs
    notes = {}
    for i, d in enumerate(DIMS):
        used = {}
        for arm, scaled in ((("none", False),) if d == 30 else (("env", True), ("none", False))):
            (lo, hi), _ = threshold(d, scaled)
            first = next((j for j, S in enumerate(S_GRID) if S >= hi), None)
            j, x = (first, 0.) if first is not None else (len(S_GRID) - 1, 1.)
            k = used.get((j, x), 0)                            # two thresholds on one edge: the second steps inwards
            used[j, x] = k + 1
            xx = x + .035 * k * (1 if x == 0 else -1)
            ax[i, j].add_line(Line2D([xx, xx], [0, 1], transform=ax[i, j].transAxes, color=colour[arm], lw=3.2,
                                     solid_capstyle="butt", clip_on=False, zorder=4))
            notes.setdefault((i, j), []).append(band_text(d, scaled, math=True))
            for jj, S in enumerate(S_GRID):
                if lo < S < hi:
                    ax[i, jj].axvspan(20, 10000, color=colour[arm], alpha=.09, lw=0, zorder=-1)
        notes.setdefault((i, S_GRID.index(H2[d])), []).append("H2")
    for (i, j), lines in notes.items():
        for k, t in enumerate(lines):
            ax[i, j].text(.05, .955 - .105 * k, t, transform=ax[i, j].transAxes, ha="left", va="top", color=INK2,
                          fontsize=fs["tick"])

    started, done = len(runs), sum(v[0][-1] >= EPISODES for v in runs.values())
    handles = [Line2D([], [], color=c, lw=2, label=l) for _, l, c in ARMS]
    handles.append(Line2D([], [], color=INK2, lw=.8, label="transfer threshold, n = .5"))
    handles.append(Line2D([], [], color=INK2, marker="|", ls="none", ms=13, mew=3.2,
                          label=r"analytic $|\mathcal{S}|^*$ for that colour's z: transfer expected from here rightwards"))
    handles.append(Patch(facecolor=INK2, alpha=.16, lw=0, label=r"inside the uncertainty of $|\mathcal{S}|^*$"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=fs["legend"],
               title=LAW, title_fontsize=fs["tick"] + .5)
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
