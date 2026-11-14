"""Hourly progress of the Var(z) test (train_bayes_generator.py tasks 123-252, run from ~/cgw_zscale on the cluster): normalized
novel-environment transfer n = (plugin - zblind) / (joint - zblind) against training episodes, one panel per (d_E, |S|) cell
in the layout of the d_E x |S| heat maps, one line per run (colour = arm, each seed its own line). n = 0 is a Generator that
ignores z_hat (the floor F, logged as novel_zblind), n = 1 is Joint Bayes (the ceiling C). A run TRANSFERS when its novel
accuracy A >= F + CUT (C - F), i.e. n >= CUT = .2 (John 11-14: a floor-dependent threshold; the floor is .10 at unscaled
d_E 10 but .20 at unscaled d_E 90, so twice chance would not do; the pre-registered cut was .5, and no H1/H2 cell differs
between the two). At d_E 30 the scale factor is 1, so its unscaled runs stand for every arm.
Names: z0 = the standard deviation of z (1/sqrt(d_E) unscaled, 1/sqrt(30) in the scaled runs: z_ref_dim = 1/z0^2);
M = |S| - |S| // 10 = a held-out variable's training partners.

Analytic thresholds (11-14, John: "put analytical predictions on this figure"). A thick vertical line on a panel's left edge:
the law puts |S|* for that colour's z just left of this panel, so its runs should transfer from here rightwards (on the
last panel's right edge: beyond the grid). rho(v) = (e^{8 pi^2 v} - 1) / (8 pi^2 v) - 1 is the fold ratio: the variance of
a product of two sinusoids of z ~ N(0, v) (the likelihood's logit, uniform phases) that a linear function of z cannot
carry, over the variance it can. It is not the gridworld's R, the number of realizations. The exact likelihood's ratio is
within 10% wherever it decides a cell. Three laws (--law):
  proxy   (default; the bootstrap rule of the 11-14 theory note) one measured level: a cell transfers when r >= r*,
          .37 < r* <= .45. r (zscale_proxy.py, its table is PROXY_R here) is the correlation with which an idealized linear
          learner predicts the novel environments' z: computed from the exact likelihood, nothing fitted, and it holds
          the fold effect and the spanning effect at once (for M >> d_E, r = 1 - d_E rho_mf(z0^2) / M with
          rho_mf(v) = sinh(8 pi^2 v) / (8 pi^2 v) - 1). The level was read off the seed-0 cells it sorts, so it is a
          description until new cells test it. A tinted panel has r inside the level's uncertainty.
  free    (John 11-14: "can we do without any fitting?") nothing measured: transfer needs |S| >= d_E rho(z0^2),
          the idealized detection bound for a rank-d_E signal in fold noise (constant 1), AND M >= d_E, a held-out
          variable's d_E unknowns against its M partners. Scaled runs have z0^2 = 1/30, so their |S|* = 3.9 d_E.
  anchor  one measured number, the d_E 30 threshold (no transfer at |S| 75, transfer at 100: 3 seeds,
          bayes_gen_arch_plot.py): |S|* = (d_E / 30) |S|*_30 rho(Var z) / rho(1/30). A tinted panel lies inside the anchor's
          uncertainty. Unscaled d_E 90 falls below the spanning bound and is FITTED to its own row, not predicted.
"H2" marks the cell where the theory chat's alternative (a Var z-only barrier near 100 plus spanning) puts the first
transfer under scaling. Covered by no law: an upper limit on |S| (unscaled d_E 20 fails at 1000) and the time
course (no formula for the onset or the plateau).

Prints a status table for the hourly report: latest n per cell and arm, runs started / finished / stalled, the onset
episode per cell, whether replicate seeds agree, and every law's tally.

    python zscale_progress.py [--pull] [--logs DIR] [--out PATH] [--law proxy|free|anchor]
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
CUT = .2                                                       # a run transfers when n >= CUT (floor-dependent rule)
MODE = "proxy"                                                 # the law drawn: "proxy", "free" or "anchor" (--law)
R_STAR = (.37, .45)                                            # the level r*: above the first, at most the second
PROXY_R = {  # {(d_E, 1 / z0^2): {|S|: r}}, printed by zscale_proxy.py (s.e. <= .01); z0^2 = 1/30 scaled, 1/d_E unscaled
    (10, 30): {20: .306, 25: .373, 30: .430, 35: .490, 40: .519, 45: .573, 50: .601, 75: .726, 100: .808, 125: .853,
               250: .931, 500: .965, 1000: .983},
    (20, 30): {50: .362, 55: .393, 60: .413, 65: .442, 70: .488, 75: .499, 100: .581, 125: .656, 250: .847, 500: .928,
               1000: .965},
    (30, 30): {50: .217, 75: .357, 80: .371, 85: .397, 90: .411, 95: .415, 100: .454, 125: .533, 250: .749, 500: .885,
               1000: .943},
    (90, 30): {50: .005, 75: .002, 100: .013, 125: .148, 250: .370, 275: .407, 300: .422, 325: .466, 350: .474, 400: .532,
               500: .613, 1000: .812},
    (20, 20): {50: .100, 75: .158, 100: .206, 125: .243, 150: .296, 175: .353, 200: .400, 225: .451, 250: .477, 500: .735,
               1000: .874},
    (90, 90): {50: .002, 75: .010, 100: .017, 105: .341, 110: .441, 115: .519, 120: .560, 125: .596, 250: .884, 500: .963,
               1000: .984},
    (10, 10): {50: .013, 75: .000, 100: .000, 125: .005, 250: -.001, 500: .006, 1000: .003}}
RHO = r"$\rho(v) = (e^{8 \pi^2 v} - 1)\,/\,(8 \pi^2 v) - 1$"
Z0 = r"$z_0^2 = \mathrm{Var}\,z$: $1/d_E$ unscaled, $1/30$ scaled"
LAW = {"proxy": (r"bootstrap rule, one measured level:   transfer when the linear proxy $r \geq r^*$,  $0.37 < r^* \leq 0.45$;   "
                 r"for $M \gg d_E$:  $r = 1 - d_E\,\rho_{\mathrm{mf}}(z_0^2)\,/\,M$,   "
                 r"$\rho_{\mathrm{mf}}(v) = \sinh(8 \pi^2 v)\,/\,(8 \pi^2 v) - 1$,   "
                 r"$M = |\mathcal{S}| - \lfloor |\mathcal{S}|/10 \rfloor$,   " + Z0),
       "free": (r"threshold with nothing fitted:   $|\mathcal{S}| \geq d_E\,\rho(z_0^2)$  and  "
                r"$M = |\mathcal{S}| - \lfloor |\mathcal{S}|/10 \rfloor \geq d_E$,   " + RHO + ",   " + Z0),
       "anchor": (r"threshold anchored on $d_E = 30$ ($75 < |\mathcal{S}|^* \leq 100$):   "
                  r"$|\mathcal{S}|^* = (d_E / 30)\ |\mathcal{S}|^*_{30}\ \rho(\mathrm{Var}\,z)\,/\,\rho(1/30)$,   "
                  + RHO + r";   unscaled $d_E = 90$: spanning-limited, fitted to its row")}


def fold_ratio(v):
    """fold / linear variance of sin(2 pi z01 - a) sin(2 pi z10 - b), z ~ N(0, v), phases uniform"""
    x = 8 * np.pi ** 2 * v
    return np.expm1(x) / x - 1


def crossing(r, level):
    """|S| at which the proxy first reaches `level`, interpolated between the tabulated |S| (inf: beyond the table)"""
    S = sorted(r)
    for a, b in zip(S, S[1:]):
        if r[a] < level <= r[b]:
            return a + (b - a) * (level - r[a]) / (r[b] - r[a])
    return S[0] if r[S[0]] >= level else float("inf")


def threshold(d, scaled, mode=None):
    """((lo, hi), kind): the law puts |S|* above lo and at most hi (lo = hi: a point). kind: predicted; spanning (free
    law, the d_E partners bind, not rho); anchor / fitted (anchor law: measured, not predicted)"""
    mode = mode or MODE
    if mode == "proxy":
        return tuple(crossing(PROXY_R[d, 30 if scaled else d], x) for x in R_STAR), "predicted"
    if mode == "free":
        osc = d * fold_ratio(1 / 30 if scaled or d == 30 else 1 / d)
        span = next(S for S in range(1, 100 * d) if S - S // 10 >= d)     # the smallest |S| with d training partners
        return (max(osc, span),) * 2, "predicted" if osc >= span else "spanning"
    if d == 30:
        return ANCHOR, "anchor"
    osc = tuple(a * d / 30 * (1 if scaled else fold_ratio(1 / d) / fold_ratio(1 / 30)) for a in ANCHOR)
    if scaled:
        return osc, "predicted"
    span = tuple(a * d / 90 for a in SPAN_90)
    return (span, "fitted") if span[1] >= osc[1] else ((max(osc[0], span[0]), osc[1]), "predicted")


def predicted(d, scaled, S, mode=None):
    """does the law say this cell transfers? True / False / None (inside the threshold's uncertainty)"""
    if (mode or MODE) == "proxy":                              # r is tabulated to s.e. .01: compared at two decimals
        r = round(PROXY_R[d, 30 if scaled else d][S], 2)
        return True if r >= R_STAR[1] else False if r <= R_STAR[0] else None
    (lo, hi), _ = threshold(d, scaled, mode)
    return True if S >= hi else False if S <= lo else None


def band_text(d, scaled, math=False):
    (lo, hi), kind = threshold(d, scaled)
    num = lambda x: f"{x / 1000:.1f}k" if x >= 1000 else f"{x:.0f}"
    who = "" if d == 30 else "scaled " if scaled else "unscaled "
    span = f"beyond {S_GRID[-1]}" if lo == float("inf") else num(hi) if lo == hi else f"{num(lo)}-{num(hi)}"
    return who + (r"$|\mathcal{S}|^*$ " if math else "") + span + ("" if kind == "predicted" else f" ({kind})")


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


def onset(ep, n):
    """the first test from which n stays at or above CUT up to the run's latest test (None: not transferring now)"""
    below = np.flatnonzero(n < CUT)
    i = below[-1] + 1 if len(below) else 0
    return int(ep[i]) if i < len(n) else None


def tally(runs, mode=None):
    """runs past TALLY_FROM episodes against the law, by the kind of their row's threshold:
    {kind: [matches, cells]}, the misses, and the runs in cells the law leaves open"""
    score, miss, open_ = {}, [], []
    for (arm, d, S, seed), (ep, n, _) in sorted(runs.items()):
        if ep[-1] < TALLY_FROM:
            continue
        scaled = arm != "none"
        p, kind = predicted(d, scaled, S, mode), threshold(d, scaled, mode)[1]
        name = f"{arm} d{d} S{S} s{seed} (n {n[-1]:.2f})"
        if p is None:
            open_.append(name)
            continue
        s = score.setdefault(kind, [0, 0])
        s[1] += 1
        if (n[-1] >= CUT) == p:
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
    print(f"latest n (mean over seeds; '*' = transfer, n >= {CUT}), by arm and d_E, for |S| = " + " ".join(map(str, S_GRID)))
    for arm, _, _ in ARMS:
        for d in DIMS:
            src = "none" if d == 30 else arm
            cells = []
            for S in S_GRID:
                v = [r[1][-1] for k, r in runs.items() if k[:3] == (src, d, S)]
                cells.append(f"{np.mean(v):5.2f}{'*' if np.mean(v) >= CUT else ' '}" if v else "    - ")
            print(f"  {arm:4s} d_E {d:2d}: " + " ".join(cells))
    print(f"onset episode (the first test from which n stays >= {CUT}; median over seeds; '-' = not transferring)")
    for arm, _, _ in ARMS:
        for d in DIMS:
            if d == 30 and arm != "none":
                continue
            cells = []
            for S in S_GRID:
                v = [onset(r[0], r[1]) for k, r in runs.items() if k[:3] == (arm, d, S)]
                v = [x for x in v if x is not None]
                cells.append(f"{int(np.median(v)):6d}" if v else "     -")
            print(f"  {arm:4s} d_E {d:2d}: " + " ".join(cells))
    seeds = {}
    for (arm, d, S, _), (ep, n, _) in runs.items():
        if ep[-1] >= EPISODES:
            seeds.setdefault((arm, d, S), []).append(bool(n[-1] >= CUT))
    split = sorted(k for k, v in seeds.items() if len(set(v)) > 1)
    print(f"replicates: {sum(len(v) > 1 for v in seeds.values())} cells with 2 or more finished seeds; seeds disagree on "
          "transfer in: " + ("; ".join(f"{a} d{d} S{S} ({sum(seeds[a, d, S])}/{len(seeds[a, d, S])} transfer)"
                                       for a, d, S in split) or "none"))
    what = {"proxy": f"bootstrap rule, r* in ({R_STAR[0]}, {R_STAR[1]}]", "free": "nothing fitted",
            "anchor": "anchor: d_E 30, 75-100"}
    print(f"analytic |S|* ({what[MODE]}): "
          + "; ".join(f"d_E {d} " + ", ".join(band_text(d, sc) for sc in ((False,) if d == 30 else (True, False)))
                      for d in DIMS if d != 30 or MODE != "anchor"))
    score, miss, open_ = tally(runs)
    print(f"{MODE} law against the runs past {TALLY_FROM} episodes: "
          + ", ".join(f"{k} cells {a}/{b}" for k, (a, b) in sorted(score.items()))
          + f"; misses: {'; '.join(miss) or 'none'}; left open by the law: {'; '.join(open_) or 'none'}")
    for mode in (m for m in ("proxy", "free") if m != MODE):   # the other laws: the count only
        score, miss, open_ = tally(runs, mode)
        print(f"{mode} law ({what[mode]}): {sum(a for a, _ in score.values())}/{sum(b for _, b in score.values())} runs "
              f"on the law's side, {len(open_)} left open")


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
            for y in (0, CUT, 1):
                a.axhline(y, color=INK2 if y == CUT else HAIR, lw=.8 if y == CUT else .6, zorder=0)
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
            a.set_yticks([0, CUT, 1], ["0", f"{CUT:g}", "1"])
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
    notes, tinted = {}, False
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
                if predicted(d, scaled, S) is None:
                    ax[i, jj].axvspan(20, 10000, color=colour[arm], alpha=.09, lw=0, zorder=-1)
                    tinted = True
        notes.setdefault((i, S_GRID.index(H2[d])), []).append("H2")
    for (i, j), lines in notes.items():
        for k, t in enumerate(lines):
            ax[i, j].text(.05, .955 - .105 * k, t, transform=ax[i, j].transAxes, ha="left", va="top", color=INK2,
                          fontsize=fs["tick"])

    started, done = len(runs), sum(v[0][-1] >= EPISODES for v in runs.values())
    handles = [Line2D([], [], color=c, lw=2, label=l) for _, l, c in ARMS]
    handles.append(Line2D([], [], color=INK2, lw=.8, label=f"transfer: n $\\geq$ {CUT:g} (a fifth of the way from the z-blind floor to Joint Bayes)"))
    handles.append(Line2D([], [], color=INK2, marker="|", ls="none", ms=13, mew=3.2,
                          label=r"analytic $|\mathcal{S}|^*$ for that colour's z: transfer expected from here rightwards"))
    if tinted:
        handles.append(Patch(facecolor=INK2, alpha=.16, lw=0, label=r"inside the uncertainty of $|\mathcal{S}|^*$"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=fs["legend"],
               title=LAW[MODE], title_fontsize=fs["tick"] + .5)
    fig.suptitle(rf"Var(z) test: novel-environment transfer $n$ = (plug-in $-$ z-blind) / (Joint $-$ z-blind)"
                 f"\n{started}/{N_RUNS} runs started, {done} finished, updated "
                 f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", fontsize=fs["label"] + 2, color=INK)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"figure -> {out}")


def main():
    global MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--logs", default=LOGS)
    ap.add_argument("--out", default=os.path.join(ROOT, "main", "REPLICATION", "figure_bin", "zscale_progress.png"))
    ap.add_argument("--law", choices=("proxy", "free", "anchor"), default=MODE,
                    help="the analytic threshold drawn (docstring)")
    a = ap.parse_args()
    MODE = a.law
    if a.pull:
        pull(a.logs)
    runs = load(a.logs)
    status(runs)
    plot(runs, a.out)


if __name__ == "__main__":
    main()
