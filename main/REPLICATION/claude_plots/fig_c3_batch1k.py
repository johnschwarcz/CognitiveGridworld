"""C = 3 RCC at batch 1,000 (train_cluster_runs.py tasks 572-573, cluster job 66761476): progress against batch 10,000.
(John 11-17 09:15 ET: "so everything is fixed and the C = 3 run is in progress? (can i get a figure?)")

Data: main/DATA/experimental/cluster_runs/jsonl/ (copied from the cluster's ~/cgw/runs/cluster_runs; --fetch copies again).
    orange  rccC3b1k_lr4_s0-1: own answers, gen_zflow, lr 1e-4, entropy .01, dense read-in, |S| 500, batch 1,000; one test
            of 1,000 novel episodes every 1,000 iterations, drawn as means of 10 tests (10,000 novel episodes per point)
    blue    rccC3_120k_lr4_s0 / rccC3_60k_lr4_s1: the same config at batch 10,000 (fig_c3_ceiling.py); one test of 10,000
            novel episodes per point; seed 0 resumed from its 60k net at 0.6 B episodes (Adam restarted)
Solid = the Classifier; dashed = exact Bayes with the Generator's learned likelihood (the plug-in). Light = seed 1.
Bottom: the batch-1,000 seeds on the ET clock (both started Mon 11-16 23:51 ET; the 44 h budget saves the nets Wed ~19:50 ET).
Bands = a - b exp(-(E - 0.6 B) / tau) fitted to each seed's tests from 0.6 B and from 0.7 B episodes (min to max of the
four fits; fits from 0.8 B do not converge); dash-dot = the last 0.2 B episodes' slope held constant (both seeds pooled).

RESULT 11-17 09:20 ET (0.95 B episodes, 9.5 h, ~100 M episodes per hour on each A100):
    Naive (.27) passed at 0.47 / 0.51 B episodes (batch 10,000: 0.52 / 0.51 B): the escape is not faster.
    .46 (batch 10,000's end) at 0.84 / 0.82 B (batch 10,000: 1.08 B): ~25% fewer episodes after the escape.
    Last 50 tests: Classifier .471 / .475, plug-in .528 / .534 (Joint .646, Naive .269); slope over the last 0.2 B
    +.021 per 0.1 B (pooled), which would reach Joint at 17:06 ET if it held. Fits at the end of the budget: Classifier
    .49-.54, plug-in .54-.59 (batch 10,000's own fits: .49-.54): the same place, ~25% sooner in episodes.

--plain (John 11-17 09:40 ET: "there is way too much going on here"): one panel, the two seeds' Classifier accuracy per
training episode with Joint and Naive only -> fig_c3_rcc.png.

11-18: seed 1 stopped at 3.98 B episodes (STOP file, net saved); seed 0 ran to its 44 h budget (4.45 B) and task 574
rccC3b1k5B_lr4_s0 resumed it to 5.0 B (cluster job 66787324; Adam restarted). That log starts with seed 0's records, so
--plain draws seed 0 from it (EXT). RESULT 11-19 (5.0 B, finished Thu 01:04 ET): last 100 tests Classifier .564 / .554
(seed 0 at 5.0 B / seed 1 at 3.98 B), plug-in .582 / .578 (Joint .647, Naive .269) = 78% of the Naive -> Joint gap.
Plateau: slope per 0.1 B +.0011 (3.0-3.5 B), +.0002 (3.5-4.0 B), +.0002 (4.0-4.45 B), +.0011 (4.45-5.0 B, after the
resume); 50 tests before the resume .562, after .556 / .558. The two-panel default view was the live view during the
44 h budget: once the runs ended its clock panel has nothing to project (the PNG in figure_bin is the 11-17 16:24 ET render).
11-19 (John: "can we keep training it for another day?"): task 575 rccC3b1k5Bmore_lr4_s0 resumes the 5 B net for 24 h
(cluster job 66796632); add it to EXT when its log is in.

    python fig_c3_batch1k.py [--fetch] [--plain]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "cluster_runs", "jsonl")
REMOTE = "elsc-cluster:/ems/elsc-labs/kadmon-j/vladimir.shaidurov/cgw/runs/cluster_runs/"
NEW, OLD = ("rccC3b1k_lr4_s0", "rccC3b1k_lr4_s1"), ("rccC3_120k_lr4_s0", "rccC3_60k_lr4_s1")
EXT = {"rccC3b1k_lr4_s0": "rccC3b1k5B_lr4_s0"}  # continuation (task 574) whose log starts with the source's records
C_NEW, C_OLD = "#d94801", "#08519c"
BLOCK, BUDGET_H, LEVEL = 10, 44, .46          # tests per drawn point at batch 1,000; run budget (h); batch 10,000's end level
CLOCK0 = 23 + 51 / 60                         # launch, Mon 23:51 ET = hour 0


def jl(n):
    return [json.loads(l) for l in open(os.path.join(LOGS, n + ".jsonl")) if l.strip()]


def raw(n):
    """Episodes (billions), Classifier and plug-in accuracy and hours since the start, one entry per test."""
    r = jl(n)
    per = 1000 if n in NEW or n in EXT.values() else 10000
    return (np.array([q["ep"] * per for q in r], float) / 1e9, np.array([q["net"] for q in r]),
            np.array([q["plugin"] for q in r]), np.array([q["t"] for q in r]) / 3600,
            np.mean([q["joint"] for q in r]), np.mean([q["naive"] for q in r]))


def drawn(n):
    """raw() averaged over BLOCK tests at batch 1,000, so every point holds 10,000 novel episodes like batch 10,000's."""
    e, y, p, h, _, _ = raw(n)
    if n not in NEW and n not in EXT.values():
        return e, y, p, h
    k = len(e) // BLOCK * BLOCK
    return tuple(a[:k].reshape(-1, BLOCK).mean(1) for a in (e, y, p, h))


def plain():
    """The default RCC's accuracy only: two seeds, Joint and Naive."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_fig_style()
    plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12})
    runs = [EXT[n] if n in EXT and os.path.exists(os.path.join(LOGS, EXT[n] + ".jsonl")) else n for n in NEW]
    joint, naive = raw(runs[0])[4:]
    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    for k, n in enumerate(runs):
        e, y, _, _ = drawn(n)
        ax.plot(e, y, color=C_NEW, lw=2.2, alpha=1 if k == 0 else .5, label=f"seed {k}")
    ax.axhline(joint, color="k", ls="--", lw=1); ax.axhline(naive, color="k", ls=":", lw=1)
    x_max = np.ceil(max(drawn(n)[0][-1] for n in runs) * 10 + .5) / 10
    ax.text(x_max * .99, joint + .012, f"Joint Bayes {joint:.2f}", fontsize=12, ha="right")
    ax.text(x_max * .99, naive + .012, f"Naive Bayes {naive:.2f}", fontsize=12, ha="right")
    ax.set(xlim=(0, x_max), ylim=(0, .72), xlabel="training episodes (billions)", ylabel="goal accuracy, novel environments")
    ax.set_title("Default RCC (own answers), C = 3\n"
                 r"$|\mathcal{S}|$ = 500, batch 1,000; point = 10,000 novel episodes", loc="left", fontsize=13)
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    out = fig_path("fig_c3_rcc.png")
    fig.savefig(out, dpi=150)
    print(out, {n: (round(float(raw(n)[0][-1]), 3), round(float(raw(n)[1][-100:].mean()), 3)) for n in runs})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="copy the batch-1,000 logs from the cluster first")
    ap.add_argument("--plain", action="store_true", help="only the two seeds' accuracy (fig_c3_rcc.png)")
    args = ap.parse_args()
    if args.fetch:
        for n in NEW + tuple(EXT.values()):
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", REMOTE + n + ".jsonl", LOGS + "/"], check=True)
    if args.plain:
        return plain()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    apply_fig_style()
    T, L, A = 13, 11.5, 11.5
    plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12})
    joint, naive = raw(NEW[0])[4:]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 10.5), constrained_layout=True)

    # (top) per training episode
    for runs, c, b in ((OLD, C_OLD, "10,000"), (NEW, C_NEW, "1,000")):
        for k, n in enumerate(runs):
            e, y, p, _ = drawn(n)
            a = 1 if k == 0 else .5
            ax.plot(e, y, color=c, lw=2.2, alpha=a, label=f"batch {b}: Classifier" if k == 0 else None)
            ax.plot(e, p, color=c, lw=1.3, alpha=a, ls=(0, (3, 2)),
                    label=f"batch {b}: Bayes with its Generator's likelihood" if k == 0 else None)
            if (y >= LEVEL).any():
                i = np.argmax(y >= LEVEL)
                ax.plot(e[i], y[i], "o", ms=7, color=c, alpha=a, mec="w", zorder=5)
                ax.plot([e[i]] * 2, [0, y[i]], color=c, alpha=a, lw=1, ls=":", zorder=1)
    hit = {n: drawn(n)[0][np.argmax(drawn(n)[1] >= LEVEL)] for n in NEW + OLD[:1]}
    ax.text(min(hit[n] for n in NEW) - .01, .03, f"first point at {LEVEL:.2f}:\n{hit[NEW[1]]:.2f} / {hit[NEW[0]]:.2f} B",
            color=C_NEW, fontsize=A, ha="right", va="bottom")
    ax.text(hit[OLD[0]] + .01, .03, f"{hit[OLD[0]]:.2f} B", color=C_OLD, fontsize=A, ha="left", va="bottom")
    ax.axhline(joint, color="k", ls="--", lw=1); ax.axhline(naive, color="k", ls=":", lw=1)
    ax.text(.02, joint + .012, f"Joint Bayes {joint:.2f}", fontsize=A)
    ax.text(.02, naive + .012, f"Naive Bayes {naive:.2f}", fontsize=A)
    ax.set(xlim=(0, 1.25), ylim=(0, .72), xlabel="training episodes (billions)", ylabel="goal accuracy, novel environments")
    ax.set_title("Per training episode: batch 1,000 escapes no sooner, then climbs faster\n"
                 r"C = 3 (1,000 joint values), own answers, $|\mathcal{S}|$ = 500; point = 10,000 novel episodes; light = seed 1",
                 loc="left", fontsize=T)
    ax.legend(frameon=False, fontsize=L, loc="center left", bbox_to_anchor=(0, .66))

    # (bottom) the batch-1,000 seeds on the clock, with projections to the end of the budget
    f = lambda E, a, b, tau: a - b * np.exp(-(E - .6) / tau)
    h_now = max(raw(n)[3][-1] for n in NEW)
    H = np.linspace(h_now, BUDGET_H, 200)
    fits = {"net": [], "plugin": []}
    tail_e, tail_y, rates = [], [], []
    for k, n in enumerate(NEW):
        e, y, p, h, _, _ = raw(n)
        rate = (e[-1] - e[-101]) / (h[-1] - h[-101])                      # billions of episodes per hour
        rates.append(rate)
        E = e[-1] + (H - h[-1]) * rate
        for E0 in (.6, .7):
            m = e >= E0
            for v, key in ((y, "net"), (p, "plugin")):
                par, _ = curve_fit(f, e[m], v[m], p0=(.55, .2, .3), maxfev=50000)
                fits[key].append(f(E, *par))
        m = e >= e[-1] - .2
        tail_e.append(e[m] - e[-1]); tail_y.append(y[m])
        ed, yd, pd, hd = drawn(n)
        a = 1 if k == 0 else .5
        bx.plot(hd, yd, color=C_NEW, lw=2.2, alpha=a, label="Classifier" if k == 0 else None)
        bx.plot(hd, pd, color=C_NEW, lw=1.3, alpha=a, ls=(0, (3, 2)),
                label="Bayes with its Generator's likelihood" if k == 0 else None)
    for key, al, lab in (("net", .28, "Classifier"), ("plugin", .14, "Bayes, Generator's likelihood")):
        lo, hi = np.min(fits[key], 0), np.max(fits[key], 0)
        bx.fill_between(H, lo, hi, color=C_NEW, alpha=al, lw=0)
        bx.text(BUDGET_H - .3, (lo[-1] + hi[-1]) / 2, f"{lab}: {lo[-1]:.2f}-{hi[-1]:.2f}", color=C_NEW, fontsize=A,
                ha="right", va="center")
    sl, ic = np.polyfit(np.concatenate(tail_e), np.concatenate(tail_y), 1)  # per billion episodes, at the last test
    rate = np.mean(rates)
    h_hit = h_now + (joint - ic) / (sl * rate)
    hl = np.linspace(h_now, min(h_hit, BUDGET_H), 50)
    bx.plot(hl, ic + sl * rate * (hl - h_now), color="#6a51a3", lw=2, ls=(0, (4, 2, 1, 2)))
    clock = lambda hh: f"{int((CLOCK0 + hh) % 24):02d}:{int(round((CLOCK0 + hh) % 1 * 60)) % 60:02d}"
    bx.text(h_hit - 1, joint + .035, f"if the last 0.2 B's slope held (+{sl / 10:.3f} per 0.1 B): Joint at {clock(h_hit)}",
            color="#6a51a3", fontsize=A, ha="left", va="center")
    for hh, s in ((h_now, f"now\n{clock(h_now)}"), (BUDGET_H, f"budget ends,\nnets saved")):
        bx.axvline(hh, color=".6", lw=1, zorder=0)
        bx.text(hh - .3, .05, s, fontsize=A, ha="right", va="bottom", color=".3")
    bx.axhline(joint, color="k", ls="--", lw=1); bx.axhline(naive, color="k", ls=":", lw=1)
    bx.text(.3, joint + .012, f"Joint Bayes {joint:.2f}", fontsize=A)
    bx.text(BUDGET_H - .3, naive + .012, f"Naive Bayes {naive:.2f}", fontsize=A, ha="right")
    ticks = np.arange(24 - CLOCK0, BUDGET_H + 1, 6)
    bx.set_xticks(ticks, [("Tue\n" if t < 6 else "Wed\n" if 23 < t < 25 else "") + clock(t) for t in ticks])
    bx.set(xlim=(0, BUDGET_H + .5), ylim=(0, .72), xlabel="Eastern time (both seeds started Mon 23:51)",
           ylabel="goal accuracy, novel environments")
    bx.set_title(f"Batch 1,000 on the clock: one A100 per seed, {rate * 1000:.0f} M episodes per hour\n"
                 r"bands: $a - b\,e^{-E/\tau}$ fitted to both seeds from 0.6 B and from 0.7 B episodes ($E$)",
                 loc="left", fontsize=T)
    bx.legend(frameon=False, fontsize=L, loc="center left", bbox_to_anchor=(.3, .25))
    for x in (ax, bx):
        x.spines[["top", "right"]].set_visible(False)
    out = fig_path("fig_c3_batch1k.png")
    fig.savefig(out, dpi=150)
    print(out, f"joint {joint:.3f} naive {naive:.3f} now {clock(h_now)} slope {sl / 10:+.4f}/0.1B hit {clock(h_hit)} "
          f"(h {h_hit:.1f}); fits net {np.min(fits['net'], 0)[-1]:.3f}-{np.max(fits['net'], 0)[-1]:.3f} "
          f"plugin {np.min(fits['plugin'], 0)[-1]:.3f}-{np.max(fits['plugin'], 0)[-1]:.3f}; hits {hit}")


if __name__ == "__main__":
    main()
