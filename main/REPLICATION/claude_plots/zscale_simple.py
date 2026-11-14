"""The Var(z) test in one simple figure (John 11-14: "simple one since we dont need the full through training curves unless it
looks like a network just needed more training time"): every run's latest n against |S|, one panel per d_E, one colour per
world (blue: z scaled to z0^2 = 1/30, green: unscaled, black: d_E 30, where the two coincide, orange: z scaled to
z0^2 = 1/20), and for each world the shaded
|S| range in which the bootstrap rule puts its threshold (where the proxy r of zscale_proxy.py crosses .37 to .45; the
level was read off seed 0 of the first 49 cells, later cells were called before they ran: zscale_tests.py).
Left of a world's band its runs should sit at n ~ 0, right of it above the line n = .2.
A filled dot is a run that finished or was stopped as converged, a hollow one a run still training (--running). A run may
only have needed more training if, past 3,000 episodes, it sits between .05 and the line and rose by RISE or more over its
last 2,000 episodes (three-test means), or crossed the line only within them: it is ringed, and the lower panel draws those runs' curves. A run far above
the line that still creeps up is not ringed (its verdict cannot change). With no ringed run the figure is the top row alone.

    python zscale_simple.py [--pull] [--logs DIR] [--out PATH] [--running NAME,NAME,...] [--stamp TEXT]
"""
import os, re, sys, glob, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zscale_progress as zp
import zscale_tests as zt

LOGS = zp.LOGS
CUT, R_STAR, EPISODES, RISE, SETTLED = .2, (.37, .45), 8000, .03, 3000
R = {**zt.R, **zp.PROXY_R}                                    # (d_E, 1 / z0^2): {|S|: r}; the finer table wins
SURFACE, INK, INK2, HAIR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GREEN, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"


def colour(d, ref):
    """d_E 30 at z0^2 = 1/30 is black (John 11-14): its scale factor is 1, so it is the scaled and the unscaled world at once"""
    return INK if (d, ref) == (30, 30) else GREEN if ref == d else BLUE if ref == 30 else ORANGE


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
    """could more training change the run's verdict? under CUT but up by RISE over its last 2,000 episodes, or over CUT
    only since within them (a run under 3,000 episodes says nothing yet)"""
    if ep[-1] < SETTLED:
        return False
    before = n[ep <= ep[-1] - 2000]
    below = np.flatnonzero(n < CUT)
    onset = ep[below[-1] + 1] if len(below) and below[-1] + 1 < len(n) else None
    creeping = .05 < n[-1] < CUT and len(before) >= 3 and n[-3:].mean() - before[-3:].mean() >= RISE   # above the noise
    return bool(creeping) or bool(onset is not None and onset > ep[-1] - 2000)


def call(world, S):
    r = round(R[world][S], 2) if S in R.get(world, ()) else None
    return None if r is None else True if r >= R_STAR[1] else False if r <= R_STAR[0] else None


def new(d, ref, S):
    """a cell the rule called before it ran (tasks 253-306): not one of the cells the level was read off"""
    return (ref == 20 and d != 20) or d == 60 or ((d, ref) == (30, 30) and 75 < S < 100) or ((d, ref) == (10, 30) and S < 50)


def plot(runs, out, running, stamp=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from main.utils import apply_fig_style
    fs = apply_fig_style()
    dims = sorted({k[0] for k in runs})
    flagged = sorted(k for k, (ep, n, _) in runs.items() if late(ep, n))
    fig = plt.figure(figsize=(3.3 * len(dims), 3.6 + (2.6 if flagged else 0)), constrained_layout=True, facecolor=SURFACE)
    gs = fig.add_gridspec(2 if flagged else 1, len(dims), height_ratios=(3, 2) if flagged else (1,))
    right = wrong = new_right = new_wrong = 0
    odd = []                                                 # runs past SETTLED episodes on the wrong side of their call
    for j, d in enumerate(dims):
        a = fig.add_subplot(gs[0, j])
        a.set_facecolor(SURFACE)
        for y in (0, CUT, 1):
            a.axhline(y, color=INK2 if y == CUT else HAIR, lw=.9 if y == CUT else .6, zorder=0)
        for ref in sorted({k[1] for k in runs if k[0] == d}, key=lambda r: (r == d, r)):
            col, world = colour(d, ref), (d, ref)
            lo, hi = (zp.crossing(R[world], x) for x in R_STAR) if world in R else (np.nan, np.nan)
            if np.isfinite(lo):
                a.axvspan(lo, hi if np.isfinite(hi) else 1400, color=col, alpha=.18, lw=0, zorder=1)
            elif world in R:
                a.annotate("threshold\nbeyond 1000", (.97, .5), xycoords="axes fraction", ha="right", va="center",
                           color=col, fontsize=fs["tick"])
            sizes = sorted({k[2] for k in runs if k[:2] == world})
            means = []
            for S in sizes:
                cell = [(k, v) for k, v in runs.items() if k[:3] == (d, ref, S)]
                seen = [v[1][-1] for k, v in cell if v[0][-1] >= 1000]             # the line: median of runs past 1,000
                if seen:
                    means.append((S, np.median(seen)))
                c = call(world, S)
                for i, (k, (ep, n, name)) in enumerate(sorted(cell)):
                    x = S * (1 + .035 * (i - (len(cell) - 1) / 2))                 # seeds side by side
                    live = name in running or ep[-1] < SETTLED
                    a.plot(x, n[-1], "o", ms=5, mfc=SURFACE if live else col, mec=col, mew=1.2, zorder=4)
                    if k in flagged:
                        a.plot(x, n[-1], "o", ms=10, mfc="none", mec=INK, mew=1.1, zorder=5)
                    if c is not None and ep[-1] >= SETTLED:                      # counted, still training or not
                        ok = (n[-1] >= CUT) == c
                        right, wrong = right + ok, wrong + (not ok)
                        if new(d, ref, S):
                            new_right, new_wrong = new_right + ok, new_wrong + (not ok)
                        if not ok:
                            odd.append(rf"{'unscaled' if ref == d else 'scaled'} $d_E$ {d}, $|\mathcal{{S}}|$ {S}, seed {k[3]}")
                            a.annotate(f"seed {k[3]}", (x, n[-1]), xytext=(-5, 7), textcoords="offset points", ha="right",
                                       color=col, fontsize=fs["tick"])
            if means:
                a.plot(*zip(*means), color=col, lw=1.2, alpha=.7, zorder=3)
        a.set_xscale("log")
        a.set_xlim(17, 1250)
        a.set_ylim(-.2, 1.08)
        a.set_xticks([25, 50, 100, 250, 500, 1000], ["25", "50", "100", "250", "500", "1000"])
        a.set_xticks([], minor=True)
        a.set_yticks([0, CUT, .5, 1], ["0", f"{CUT:g}", ".5", "1"])
        a.tick_params(colors=INK2, length=2, labelleft=j == 0)
        for s in a.spines.values():
            s.set_color(HAIR)
        a.set_title(rf"$d_E = {d}$", fontsize=fs["label"] + 1, color=INK)
        a.set_xlabel(r"number of variables $|\mathcal{S}|$", color=INK2)
        if j == 0:
            a.set_ylabel("transfer $n$ at the latest test\n(0 = ignores z, 1 = Joint Bayes)", color=INK)
    if flagged:
        b = fig.add_subplot(gs[1, :])
        b.set_facecolor(SURFACE)
        for y in (0, CUT, 1):
            b.axhline(y, color=INK2 if y == CUT else HAIR, lw=.9 if y == CUT else .6, zorder=0)
        y_last = -1.
        for k in sorted(flagged, key=lambda k: runs[k][1][-1]):           # labels bottom-up, kept apart
            ep, n, name = runs[k]
            b.plot(ep[ep > 0], n[ep > 0], color=colour(k[0], k[1]), lw=1.5)
            y_last = max(n[-1], y_last + .075)
            world = "unscaled" if k[1] == k[0] else f"$z_0^2$ = 1/{k[1]}"
            b.annotate(rf"$d_E$ {k[0]}, $|\mathcal{{S}}|$ {k[2]}, seed {k[3]}, {world}: {n[-1]:.2f} at {ep[-1]}",
                       (ep[-1], n[-1]), xytext=(9500, y_last), textcoords="data", va="center", color=colour(k[0], k[1]),
                       fontsize=fs["tick"], arrowprops=dict(arrowstyle="-", color=HAIR, lw=.6))
        b.set_xscale("log")
        b.set_xlim(100, 30000)
        b.set_ylim(-.2, max(1.08, y_last + .08))
        b.set_xticks([100, 1000, 8000], ["100", "1k", "8k"])
        b.set_yticks([0, CUT, 1], ["0", f"{CUT:g}", "1"])
        b.tick_params(colors=INK2, length=2)
        for s in b.spines.values():
            s.set_color(HAIR)
        b.set_xlabel("training episodes", color=INK2)
        b.set_ylabel("$n$", color=INK)
        b.set_title("ringed above: more training could change the verdict (under the line but rising, or over it only lately)",
                    fontsize=fs["tick"] + 1, color=INK2, loc="left")
    handles = [Line2D([], [], color=BLUE, marker="o", lw=1.2, label=r"z scaled, $z_0^2 = 1/30$ (dot: a run, line: median)"),
               Line2D([], [], color=GREEN, marker="o", lw=1.2, label=r"unscaled, $z_0^2 = 1/d_E$"),
               Line2D([], [], color=INK, marker="o", lw=1.2, label=r"$d_E = 30$: scaled and unscaled are the same world"),
               Line2D([], [], color=ORANGE, marker="o", lw=1.2, label=r"z scaled, $z_0^2 = 1/20$"),
               Patch(facecolor=INK2, alpha=.2, lw=0, label="where the rule puts that colour's threshold"),
               Line2D([], [], color=INK2, lw=.9, label=f"transfer: n $\\geq$ {CUT:g}"),
               Line2D([], [], color=INK2, marker="o", mfc=SURFACE, ls="none", label="still training (value so far)"),
               Line2D([], [], color=INK, marker="o", mfc="none", ms=10, ls="none", label="more training could change the verdict")]
    if not any(k[1] == 20 and k[0] != 20 for k in runs):
        handles.pop(3)
    if not flagged:
        handles.pop()
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=fs["legend"])
    fig.suptitle("Var(z) test: transfer by number of variables, against where the rule puts each threshold"
                 f"\n{len(runs)} runs, {stamp or time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}; past {SETTLED:,} episodes: "
                 f"{right} of {right + wrong} runs on the rule's side of their band"
                 + (f" (not: {'; '.join(odd)})" if 0 < len(odd) <= 2 else "")
                 + (f"; in cells called before they ran: {new_right} of {new_right + new_wrong}" if new_right + new_wrong else ""),
                 fontsize=fs["label"] + 1, color=INK)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"figure -> {out}")
    return flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--logs", default=LOGS)
    ap.add_argument("--out", default=os.path.join(ROOT, "main", "REPLICATION", "figure_bin", "zscale_simple.png"))
    ap.add_argument("--running", default="", help="names of the runs still training, comma-separated (S100_d30_s1_znone)")
    ap.add_argument("--stamp", default="", help="the time of the data, for the title (default: now, UTC)")
    a = ap.parse_args()
    if a.pull:
        zp.pull(a.logs)
    runs = load(a.logs)
    flagged = plot(runs, a.out, set(filter(None, a.running.split(","))), a.stamp)
    for k in flagged:
        ep, n, name = runs[k]
        print(f"more training could change the verdict: {name}  n {n[-1]:.2f} at episode {ep[-1]}")
    print(f"{len(runs)} runs; {len(flagged)} whose verdict more training could change")


if __name__ == "__main__":
    main()
