"""The follow-up cells of the Var(z) test (train_bayes_generator.py tasks 253-306; John 11-14: "let's run these unseen cells"),
each against the call the bootstrap rule made before it ran. The rule (theory note of 11-14, zscale_proxy.py): a cell
transfers (n >= .2) when its proxy r >= r*, .37 < r* <= .45; so r <= .37 is called "no transfer", r >= .45 "transfer",
and a cell in between is "open" (its outcome narrows r*). The level was read off the seed-0 cells of tasks 123-171, so
these cells are the rule's first test. Rows:
  A      d_E 30 at |S| 80-95 (tasks 263-274), between the last failing size (75) and the first transferring one (100):
         80 on the band's lower edge, 85-95 open; the first size that transfers pins r*
  B      a second z0, z0^2 = 1/20 (tasks 275-292, logs *_zenv20): d_E 10 and d_E 30
  C      scaled d_E 10 below the grid, |S| 20-40 (tasks 253-260; only 2-4 held-out variables: 3-10 novel environments)
  miss   unscaled d_E 20 at |S| 1000, seeds 0-4 (tasks 150, 235, 242, 261, 262): r = .87, seed 0 fails, unexplained
  extra  d_E 60, scaled and unscaled (tasks 293-306), drawn once a log exists
A panel: n against training episodes, one line per seed; its header gives r and the call. Blue = the rule called transfer,
orange = no transfer, grey = open. A run counts once it is past 3,000 episodes (runs stopped early as converged count).

    python zscale_tests.py [--pull] [--logs DIR] [--out PATH]
"""
import os, re, sys, glob, json, time, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "bayes_gen_zscale")
REMOTE = "cgw_zscale/runs/bayes_gen/./joint_S*_d*_s*_z*.jsonl"
CUT, R_STAR, COUNT_FROM = .2, (.37, .45), 3000
SURFACE, INK, INK2, HAIR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CALL = {True: ("transfer", "#2a78d6"), False: ("no transfer", "#eb6834"), None: ("open", "#52514e")}
# (d_E, 1 / z0^2): {|S|: r}. zscale_proxy.py's table; the d_E 60 rows: its pooling() and cell() with rng seed 6060 (11-14)
R = {(30, 30): {75: .357, 80: .371, 85: .397, 90: .411, 95: .415, 100: .454},
     (10, 20): {50: .209, 75: .303, 100: .390, 125: .443, 250: .716},
     (30, 20): {125: .165, 250: .334, 500: .593, 1000: .805},
     (10, 30): {20: .306, 25: .373, 30: .430, 40: .519, 50: .601},
     (20, 20): {500: .735, 1000: .874},
     (60, 30): {50: .004, 75: .112, 100: .229, 125: .296, 250: .521, 500: .727, 1000: .883},
     (60, 60): {50: .005, 75: .352, 100: .566, 125: .661, 250: .870, 500: .951, 1000: .977}}
ROWS = (("A", r"$d_E = 30$" + "\n(factor 1)", (30, 30)), ("B", r"$d_E = 10$" + "\n" + r"$z_0^2 = 1/20$", (10, 20)),
        ("B", r"$d_E = 30$" + "\n" + r"$z_0^2 = 1/20$", (30, 20)), ("C", r"$d_E = 10$" + "\n" + r"$z_0^2 = 1/30$", (10, 30)),
        ("miss", r"$d_E = 20$" + "\nunscaled", (20, 20)), ("extra", r"$d_E = 60$" + "\n" + r"$z_0^2 = 1/30$", (60, 30)),
        ("extra", r"$d_E = 60$" + "\nunscaled", (60, 60)))
KNOWN = {(30, 30, 75), (30, 30, 100), (10, 30, 50), (20, 20, 500), (20, 20, 1000)}   # cells the level was read off: not tests


def call(world, S):
    """the rule's call for a cell: True (transfer), False (no transfer) or None (open); r compared at two decimals"""
    r = round(R[world][S], 2)
    return True if r >= R_STAR[1] else False if r <= R_STAR[0] else None


def pull(logs):
    os.makedirs(logs, exist_ok=True)
    r = subprocess.run(["bash", os.path.join(ROOT, "main", "REPLICATION", "trainings", "cluster.sh"), "pull", REMOTE, logs],
                       capture_output=True, text=True)
    if r.returncode:
        print(f"pull failed (using the logs already here): {(r.stderr or r.stdout).strip()[:300]}")


def load(logs):
    """{((d_E, 1 / z0^2), |S|, seed): (episodes, n)} for the cells of R; unscaled runs have 1 / z0^2 = d_E"""
    runs = {}
    for f in glob.glob(os.path.join(logs, "joint_S*_d*_s*_z*.jsonl")):
        m = re.search(r"joint_S(\d+)_d(\d+)_s(\d+)_z(none|env)(\d*)\.jsonl$", f)
        if not m:
            continue
        d = int(m[2])
        world, S = (d, int(m[5]) if m[5] else 30 if m[4] == "env" else d), int(m[1])
        if S not in R.get(world, ()):
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
            runs[world, S, int(m[3])] = (np.array(ep), np.array(n))
    return runs


def status(runs):
    right = wrong = 0
    fails, passes = [], []                                   # the r of the counted runs that fail / transfer (not the miss)
    print(f"follow-up cells: {len(runs)} runs with a log; a run counts once past {COUNT_FROM} episodes; '*' = n >= {CUT}")
    for test, _, world in ROWS:
        if not any(k[0] == world for k in runs):
            continue
        print(f"  {test}: d_E {world[0]}, z0^2 = 1/{world[1]}")
        for S in sorted(R[world]):
            c, seeds = call(world, S), sorted((k[2], v) for k, v in runs.items() if k[:2] == (world, S))
            cells = []
            for seed, (ep, n) in seeds:
                cells.append(f"s{seed} {n[-1]:.2f}{'*' if n[-1] >= CUT else ' '}@{ep[-1]}")
                if ep[-1] >= COUNT_FROM:
                    if test != "miss":
                        (passes if n[-1] >= CUT else fails).append(R[world][S])
                    if c is not None and (world[0], world[1], S) not in KNOWN:
                        right, wrong = right + ((n[-1] >= CUT) == c), wrong + ((n[-1] >= CUT) != c)
            known = " (known before)" if (world[0], world[1], S) in KNOWN else ""
            print(f"    |S| {S:4d}  r {R[world][S]:.2f}  call {CALL[c][0]:11s}{known}  " + ("  ".join(cells) or "no log yet"))
    print(f"calls on new cells, runs past {COUNT_FROM} episodes: {right} right, {wrong} wrong")
    if fails or passes:
        lo, hi = max(fails, default=float("nan")), min(passes, default=float("nan"))
        print(f"the level from these runs alone (the miss row aside): the highest r that fails is {lo:.2f}, the lowest "
              f"r that transfers is {hi:.2f}" + ("  (one level cannot separate them)" if lo >= hi else ""))


def plot(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import apply_fig_style
    fs = apply_fig_style()
    rows = [r for r in ROWS if r[0] != "extra" or any(k[0] == r[2] for k in runs)]
    ncol = max(len(R[w]) for _, _, w in rows)
    fig, ax = plt.subplots(len(rows), ncol, figsize=(2.25 * ncol, 1.95 * len(rows) + 1.3), sharex=True, sharey=True,
                           constrained_layout=True, facecolor=SURFACE, squeeze=False)
    for i, (test, label, world) in enumerate(rows):
        for j in range(ncol):
            a = ax[i, j]
            sizes = sorted(R[world])
            if j >= len(sizes):
                a.set_visible(False)
                continue
            S, c = sizes[j], call(world, sizes[j])
            a.set_facecolor(SURFACE)
            for y in (0, CUT, 1):
                a.axhline(y, color=INK2 if y == CUT else HAIR, lw=.8 if y == CUT else .6, zorder=0)
            seeds = sorted((k[2], v) for k, v in runs.items() if k[:2] == (world, S))
            for _, (ep, n) in seeds:
                a.plot(ep[ep > 0], n[ep > 0], color=CALL[c][1], lw=1.5, solid_capstyle="round")
            if not seeds:
                a.text(.5, .4, "not started", transform=a.transAxes, ha="center", va="center", color=INK2,
                       fontsize=fs["tick"])
            known = ", known" if (world[0], world[1], S) in KNOWN else ""
            a.text(.04, .96, rf"$|\mathcal{{S}}| = {S}$   $r = {R[world][S]:.2f}$" + f"\n{CALL[c][0]}{known}",
                   transform=a.transAxes, ha="left", va="top", color=CALL[c][1], fontsize=fs["tick"])
            a.set_xscale("log")
            a.set_xlim(20, 10000)
            a.set_ylim(-.2, 1.45)
            a.set_xticks([100, 1000, 8000], ["100", "1k", "8k"])
            a.set_yticks([0, CUT, 1], ["0", f"{CUT:g}", "1"])
            a.tick_params(colors=INK2, length=2, labelbottom=True)
            for s in a.spines.values():
                s.set_color(HAIR)
            if j == 0:
                a.set_ylabel(f"{test}\n{label}\n$n$", color=INK)
            if i == len(rows) - 1:
                a.set_xlabel("training episodes", color=INK2)
    handles = [Line2D([], [], color=col, lw=2, label=f"the rule called: {name}") for name, col in CALL.values()]
    handles.append(Line2D([], [], color=INK2, lw=.8, label=f"transfer: n $\\geq$ {CUT:g}"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=fs["legend"],
               title=r"call from the proxy $r$, fixed before the runs: $r \leq 0.37$ no transfer, $r \geq 0.45$ transfer, "
                     r"between: open (its outcome narrows $r^*$);  'known' = a cell the level was read off",
               title_fontsize=fs["tick"] + .5)
    fig.suptitle("Var(z) test, follow-up cells against the bootstrap rule's calls"
                 f"\n{len(runs)} runs with a log, updated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
                 fontsize=fs["label"] + 2, color=INK)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--logs", default=LOGS)
    ap.add_argument("--out", default=os.path.join(ROOT, "main", "REPLICATION", "figure_bin", "zscale_tests.png"))
    a = ap.parse_args()
    if a.pull:
        pull(a.logs)
    runs = load(a.logs)
    status(runs)
    plot(runs, a.out)


if __name__ == "__main__":
    main()
