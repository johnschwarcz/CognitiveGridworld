"""The embedding-transfer theory in one inequality, and every Var(z) run against it (John 11-14: "simplify and compress the
theory as much as possible"). Transfer starts when a variable has enough training partners for its embedding dimension and
for the spread of z:

    M  >=  d_E sinh(x) / x,      x = 8 pi^2 z0^2      (nothing fitted)

M = |S| - |S| // 10 training partners; z0^2 = Var z (1 / d_E in the original world, 1/30 or 1/20 in the scaled runs), so x
is twice the variance of the phase 2 pi z. sinh(x) / x = 1 + rho: the variance of a pair's pooled likelihood read-out over
the part of it that is a straight line in z (zscale_proxy.rho_mf). Reading: each partner gives one equation for the d_E
unknown coordinates of a variable's embedding, of which only the fraction x / sinh(x) is usable; the usable equations must
cover the unknowns. It is the point where least squares on M noisy equations predicts a new pair's z better than ignoring z
(error variance rho d_E / (M - d_E) below the signal's). Why the network obeys an out-of-sample criterion is not derived.
In the original world the need is d_E^2 sinh(8 pi^2 / d_E) / (8 pi^2): least at d_E ~ 41 (72 partners, |S| ~ 80), exploding
below (the fold noise) and growing as d_E above (the unknowns). Goal-only training (bayes_gen_plot.py) has usable fraction
1 / 3.44, so d_E 30 needs 103 partners: its runs fail at |S| 100 and transfer at 125.
Status 11-14 21:40 ET: 88 of 92 runs on the law's side. Off: scaled d_E 10 at |S| 30 (2 seeds, 3% over the line), scaled
d_E 60 at |S| 250 (1 run), unscaled d_E 20 at |S| 1000 seed 0 (the other 4 seeds transfer).
The earlier, longer forms (the proxy r with a measured level, zscale_proxy.py / zscale_progress.py --law proxy; the
bracket, zscale_bootstrap.py) give the same thresholds at z0^2 = 1/30 and are not needed here.

Two figures, n = (plugin - zblind) / (joint - zblind) at each run's latest test, transfer = n >= .2:
  zscale_collapse.png  every run against lambda = M / (d_E sinh(x) / x), partners available over partners needed
  zscale_simple.png    the same runs by |S|, one panel per d_E, a vertical line per world at the |S| where lambda = 1
Colours: blue z scaled to z0^2 = 1/30, green unscaled, black d_E 30 (factor 1: both), orange z scaled to z0^2 = 1/20.
Filled = finished or stopped as converged, hollow = still training (--running, or under 3,000 episodes). A run that may only
have needed more training (past 3,000 episodes, between .05 and the line and rising, or over the line only within its
last 2,000 episodes) is ringed in zscale_simple.png and its curve drawn in a lower panel.

    python zscale_law.py [--pull] [--logs DIR] [--out DIR] [--running NAME,NAME,...] [--stamp TEXT]
"""
import os, re, sys, glob, json, time, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "bayes_gen_zscale")
REMOTE = "cgw_zscale/runs/bayes_gen/./joint_S*_d*_s*_z*.jsonl"
CUT, RISE, SETTLED = .2, .03, 3000
SURFACE, INK, INK2, HAIR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GREEN, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"


def need(d, ref):
    """training partners needed: d_E sinh(x) / x with x = 8 pi^2 z0^2, z0^2 = 1 / ref"""
    x = 8 * np.pi ** 2 / ref
    return d * np.sinh(x) / x


def first_size(d, ref):
    """the smallest |S| whose M = |S| - |S| // 10 training partners meet the need"""
    return next(S for S in range(2, 100_000) if S - S // 10 >= need(d, ref))


def colour(d, ref):
    return INK if (d, ref) == (30, 30) else GREEN if ref == d else BLUE if ref == 30 else ORANGE


def world_name(d, ref):
    return "" if (d, ref) == (30, 30) else "unscaled " if ref == d else "scaled "


def pull(logs):
    os.makedirs(logs, exist_ok=True)
    r = subprocess.run(["bash", os.path.join(ROOT, "main", "REPLICATION", "trainings", "cluster.sh"), "pull", REMOTE, logs],
                       capture_output=True, text=True)
    if r.returncode:
        print(f"pull failed (using the logs already here): {(r.stderr or r.stdout).strip()[:300]}")


def load(logs):
    """{(d_E, 1 / z0^2, |S|, seed): (episodes, n, name)}; unscaled runs have 1 / z0^2 = d_E; the dropped both arm is skipped"""
    runs = {}
    for f in glob.glob(os.path.join(logs, "joint_S*_d*_s*_z*.jsonl")):
        m = re.search(r"joint_(S(\d+)_d(\d+)_s(\d+)_z(none|env)(\d*))\.jsonl$", f)
        if not m:
            continue
        d = int(m[3])
        ep, n = [], []
        for line in open(f):
            try:
                r = json.loads(line)
                ep.append(r["ep"])
                n.append((r["novel_plugin"] - r["novel_zblind"]) / max(r["novel_joint"] - r["novel_zblind"], 1e-6))
            except (ValueError, KeyError):
                continue
        if ep:
            runs[d, int(m[6]) if m[6] else 30 if m[5] == "env" else d, int(m[2]), int(m[4])] = (np.array(ep), np.array(n), m[1])
    return runs


def late(ep, n):
    """could more training change the run's verdict? between .05 and CUT but rising over its last 2,000 episodes (three-test
    means), or over CUT only since within them (a run under 3,000 episodes says nothing yet)"""
    if ep[-1] < SETTLED:
        return False
    before = n[ep <= ep[-1] - 2000]
    below = np.flatnonzero(n < CUT)
    onset = ep[below[-1] + 1] if len(below) and below[-1] + 1 < len(n) else None
    creeping = .05 < n[-1] < CUT and len(before) >= 3 and n[-3:].mean() - before[-3:].mean() >= RISE
    return bool(creeping) or bool(onset is not None and onset > ep[-1] - 2000)


def score(runs):
    """(right, [(lambda, key, n, episodes)] of the wrong ones) over runs past SETTLED episodes or already far above the line"""
    right, odd = 0, []
    for (d, ref, S, seed), (ep, n, name) in sorted(runs.items()):
        lam = (S - S // 10) / need(d, ref)
        if ep[-1] >= SETTLED or n[-1] >= .5:
            if (n[-1] >= CUT) == (lam >= 1):
                right += 1
            else:
                odd.append((lam, (d, ref, S, seed), n[-1], ep[-1]))
    return right, odd


def style(a, fs):
    a.set_facecolor(SURFACE)
    for y in (0, CUT, 1):
        a.axhline(y, color=INK2 if y == CUT else HAIR, lw=.9 if y == CUT else .6, zorder=0)
    a.set_yticks([0, CUT, .5, 1], ["0", f"{CUT:g}", ".5", "1"])
    a.tick_params(colors=INK2, length=2)
    for s in a.spines.values():
        s.set_color(HAIR)


def legend(fig, runs, fs, extra=(), ncol=5):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLUE, marker="o", ls="none", label=r"z scaled, $z_0^2 = 1/30$"),
               Line2D([], [], color=GREEN, marker="o", ls="none", label=r"unscaled, $z_0^2 = 1/d_E$"),
               Line2D([], [], color=INK, marker="o", ls="none", label=r"$d_E = 30$ (scaled = unscaled)")]
    if any(k[1] == 20 and k[0] != 20 for k in runs):
        handles.append(Line2D([], [], color=ORANGE, marker="o", ls="none", label=r"z scaled, $z_0^2 = 1/20$"))
    handles.append(Line2D([], [], color=INK2, marker="o", mfc=SURFACE, ls="none", label="still training"))
    fig.legend(handles=handles + list(extra), loc="outside lower center", ncol=ncol, frameon=False, fontsize=fs["legend"])


def collapse(runs, out, running, stamp):
    import matplotlib.pyplot as plt
    from main.utils import apply_fig_style
    fs = apply_fig_style()
    right, odd = score(runs)
    fig, a = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True, facecolor=SURFACE)
    style(a, fs)
    a.axvline(1, color=INK, lw=1.4, zorder=1)
    a.text(1.06, 1.12, "enough partners", ha="left", va="center", color=INK, fontsize=fs["tick"] + 1)
    a.text(.94, 1.12, "too few", ha="right", va="center", color=INK, fontsize=fs["tick"] + 1)
    for (d, ref, S, seed), (ep, n, name) in sorted(runs.items()):
        lam, col = (S - S // 10) / need(d, ref), colour(d, ref)
        live = name in running or (ep[-1] < SETTLED and n[-1] < .5)
        a.plot(lam, n[-1], "o", ms=5.5, mfc=SURFACE if live else col, mec=col, mew=1.2, alpha=.85, zorder=3)
    cells = {}                                               # name the cells with runs on the wrong side, once each
    for lam, (d, ref, S, seed), n, ep in odd:
        cells.setdefault((lam, d, ref, S), []).append(n)
    spots = [((-14, 30), "right"), ((14, -26), "left"), ((0, 24), "center"), ((14, 30), "left")]   # clear of the n = .2 line
    for k, ((lam, d, ref, S), ns) in enumerate(sorted(cells.items())):
        total = sum(key[:3] == (d, ref, S) for key in runs)
        (dx, dy), ha = spots[k % len(spots)]
        a.annotate(rf"{world_name(d, ref)}$d_E$ {d}, $|\mathcal{{S}}|$ {S}: {len(ns)} of {total} run{'s' if total > 1 else ''}",
                   (lam, float(np.mean(ns))), xytext=(dx, dy), textcoords="offset points", ha=ha, color=INK2,
                   fontsize=fs["tick"], arrowprops=dict(arrowstyle="-", color=INK2, lw=.6))
    a.set_xscale("log")
    a.set_xlim(.02, 45)
    a.set_ylim(-.34, 1.2)                                    # room under the floor for the labels
    a.set_xticks([.03, .1, .3, 1, 3, 10, 30], ["0.03", "0.1", "0.3", "1", "3", "10", "30"])
    a.set_xticks([], minor=True)
    a.set_xlabel(r"partners available / partners needed:   $M \,/\, (d_E \sinh x / x)$,   $x = 8\pi^2 z_0^2$", color=INK)
    a.set_ylabel("transfer $n$ at the latest test\n(0 = ignores z, 1 = Joint Bayes)", color=INK)
    legend(fig, runs, fs)
    fig.suptitle("One inequality, nothing fitted: transfer when a variable has enough training partners"
                 f"\n{len(runs)} runs ({stamp}); {right} of {right + len(odd)} on the law's side",
                 fontsize=fs["label"] + 1, color=INK)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"figure -> {out}")
    return right, odd


def by_size(runs, out, running, stamp):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import apply_fig_style
    fs = apply_fig_style()
    right, odd = score(runs)
    wrong = {key for _, key, _, _ in odd}
    dims = sorted({k[0] for k in runs})
    flagged = sorted(k for k, (ep, n, _) in runs.items() if late(ep, n))
    fig = plt.figure(figsize=(3.3 * len(dims), 3.6 + (2.6 if flagged else 0)), constrained_layout=True, facecolor=SURFACE)
    gs = fig.add_gridspec(2 if flagged else 1, len(dims), height_ratios=(3, 2) if flagged else (1,))
    for j, d in enumerate(dims):
        a = fig.add_subplot(gs[0, j])
        style(a, fs)
        for ref in sorted({k[1] for k in runs if k[0] == d}, key=lambda r: (r == d, r)):
            col, S0 = colour(d, ref), first_size(d, ref)
            if S0 <= 1250:
                a.axvline(S0, color=col, lw=2.2, alpha=.55, zorder=1)
            else:
                a.annotate("threshold\nbeyond 1000", (.97, .5), xycoords="axes fraction", ha="right", va="center", color=col,
                           fontsize=fs["tick"])
            mid = []
            for S in sorted({k[2] for k in runs if k[:2] == (d, ref)}):
                cell = sorted((k, v) for k, v in runs.items() if k[:3] == (d, ref, S))
                seen = [v[1][-1] for k, v in cell if v[0][-1] >= 1000]             # the line: median of runs past 1,000
                if seen:
                    mid.append((S, np.median(seen)))
                for i, (k, (ep, n, name)) in enumerate(cell):
                    x = S * (1 + .035 * (i - (len(cell) - 1) / 2))                 # seeds side by side
                    live = name in running or (ep[-1] < SETTLED and n[-1] < .5)
                    a.plot(x, n[-1], "o", ms=5, mfc=SURFACE if live else col, mec=col, mew=1.2, zorder=4)
                    if k in flagged:
                        a.plot(x, n[-1], "o", ms=10, mfc="none", mec=INK, mew=1.1, zorder=5)
                    if k in wrong and not all(c in wrong for c, _ in cell):     # the odd seed of a cell, by name
                        a.annotate(f"seed {k[3]}", (x, n[-1]), xytext=(-5, 7), textcoords="offset points", ha="right",
                                   color=col, fontsize=fs["tick"])
            if mid:
                a.plot(*zip(*mid), color=col, lw=1.2, alpha=.7, zorder=3)
        a.set_xscale("log")
        a.set_xlim(17, 1250)
        a.set_ylim(-.2, 1.08)
        a.set_xticks([25, 50, 100, 250, 500, 1000], ["25", "50", "100", "250", "500", "1000"])
        a.set_xticks([], minor=True)
        a.tick_params(labelleft=j == 0)
        a.set_title(rf"$d_E = {d}$", fontsize=fs["label"] + 1, color=INK)
        a.set_xlabel(r"number of variables $|\mathcal{S}|$", color=INK2)
        if j == 0:
            a.set_ylabel("transfer $n$ at the latest test\n(0 = ignores z, 1 = Joint Bayes)", color=INK)
    if flagged:
        b = fig.add_subplot(gs[1, :])
        style(b, fs)
        y_last = -1.
        for k in sorted(flagged, key=lambda k: runs[k][1][-1]):           # labels bottom-up, kept apart
            ep, n, name = runs[k]
            b.plot(ep[ep > 0], n[ep > 0], color=colour(k[0], k[1]), lw=1.5)
            y_last = max(n[-1], y_last + .075)
            b.annotate(rf"{world_name(k[0], k[1])}$d_E$ {k[0]}, $|\mathcal{{S}}|$ {k[2]}, seed {k[3]}: {n[-1]:.2f} at {ep[-1]}",
                       (ep[-1], n[-1]), xytext=(9500, y_last), textcoords="data", va="center", color=colour(k[0], k[1]),
                       fontsize=fs["tick"], arrowprops=dict(arrowstyle="-", color=HAIR, lw=.6))
        b.set_xscale("log")
        b.set_xlim(100, 30000)
        b.set_ylim(-.2, max(1.08, y_last + .08))
        b.set_xticks([100, 1000, 8000], ["100", "1k", "8k"])
        b.set_xlabel("training episodes", color=INK2)
        b.set_ylabel("$n$", color=INK)
        b.set_title("ringed above: more training could change the verdict (under the line but rising, or over it only lately)",
                    fontsize=fs["tick"] + 1, color=INK2, loc="left")
    extra = [Line2D([], [], color=INK2, lw=2.2, alpha=.55, label=r"the law's threshold: $M = d_E \sinh x / x$"),
             Line2D([], [], color=INK2, lw=.9, label=f"transfer: n $\\geq$ {CUT:g}")]
    if flagged:
        extra.append(Line2D([], [], color=INK, marker="o", mfc="none", ms=10, ls="none",
                            label="more training could change the verdict"))
    legend(fig, runs, fs, extra, ncol=4)
    fig.suptitle("Var(z) test: transfer by number of variables, against the law's thresholds (nothing fitted)"
                 f"\n{len(runs)} runs ({stamp}); {right} of {right + len(odd)} on the law's side",
                 fontsize=fs["label"] + 1, color=INK)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"figure -> {out}")
    return flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--logs", default=LOGS)
    ap.add_argument("--out", default=os.path.join(ROOT, "main", "REPLICATION", "figure_bin"), help="folder for the two PNGs")
    ap.add_argument("--running", default="", help="names of the runs still training, comma-separated (S100_d30_s1_znone)")
    ap.add_argument("--stamp", default="", help="the time of the data, for the titles (default: now, UTC)")
    a = ap.parse_args()
    if a.pull:
        pull(a.logs)
    import matplotlib
    matplotlib.use("Agg")
    runs, running = load(a.logs), set(filter(None, a.running.split(",")))
    if not runs:
        return print(f"no run logs in {a.logs}")
    stamp = a.stamp or time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    os.makedirs(a.out, exist_ok=True)
    right, odd = collapse(runs, os.path.join(a.out, "zscale_collapse.png"), running, stamp)
    flagged = by_size(runs, os.path.join(a.out, "zscale_simple.png"), running, stamp)
    print(f"the law M >= d_E sinh(x)/x: {right} of {right + len(odd)} runs on its side")
    for lam, (d, ref, S, seed), n, ep in sorted(odd):
        print(f"  off: d_E {d}, z0^2 = 1/{ref}, |S| {S}, seed {seed}: {lam:.2f} of the need, n {n:.2f} at episode {ep}")
    for k in flagged:
        print(f"  more training could change the verdict: {runs[k][2]}  n {runs[k][1][-1]:.2f} at episode {runs[k][0][-1]}")
    print("partners needed -> first |S|: " + "; ".join(f"d_E {d}, z0^2 1/{ref}: {need(d, ref):.0f} -> {first_size(d, ref)}"
                                                        for d, ref in sorted({k[:2] for k in runs})))


if __name__ == "__main__":
    main()
