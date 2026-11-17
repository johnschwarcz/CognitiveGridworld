"""Can the C = 3 RCC reach Joint Bayes if trained longer? (John 11-16 23:30 ET: "how about training an RCC with C = 3
until optimality?, do you think we have time for that?")

Data: fig_c3_routes.py's runs (11-12/13; |S| 500, batch 10,000, lr 1e-4, dense read-in, 1,000 joint values).
    blue   the RCC on its own answers (gen_zflow, Generator route): seed 0 to 120k (continued from its 60k net), seed 1 to 60k
    grey   the Generator route on oracle labels: Bayes with its learned likelihood (the plug-in), seeds 0-1 to 60k / 34k
Dotted: seed 0's Classifier extrapolated past 120k by a + b exp(-(t - 60k) / tau) fitted from 60k or from 80k, and by the
slope of its last 20k held constant. Speed of the 60k -> 120k continuation on one cluster GPU: 2.63 it/s (6.3 h).

RESULT 11-16 23:50 ET: seed 0 at 120k: Classifier .461 (mean of the last 5k .468), plug-in .509; Joint .646, Naive .269.
Fit from 60k: ceiling .488 +- .003 (tau 29k); from 80k: .541 +- .033 (tau 63k); plug-in .532 / .556. Last 20k: +.014 per 10k
(plug-in +.010), which reaches Joint at ~245k if it held. Oracle labels: plug-in .576 at 10k, .600 at 50k (KL .02).

    python fig_c3_ceiling.py
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "cluster_runs", "jsonl")
C_GEN, C_ORC = "#08519c", ".55"
RATE = 2.63                                                   # it/s, the 60k -> 120k continuation on the cluster


def jl(n):
    return [json.loads(l) for l in open(os.path.join(LOGS, n + ".jsonl")) if l.strip()]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    fs = apply_fig_style()
    L = fs["legend"]
    s0, s1 = jl("rccC3_120k_lr4_s0"), jl("rccC3_60k_lr4_s1")
    joint = np.mean([q["joint"] for q in s0]); naive = np.mean([q["naive"] for q in s0])
    fig, ax = plt.subplots(figsize=(11, 6.4), constrained_layout=True)
    for r, a, lab in ((s0, 1, "RCC, own answers: the Classifier (seed 0)"), (s1, .45, "seed 1")):
        ax.plot([q["ep"] for q in r], [q["net"] for q in r], color=C_GEN, lw=2.2, alpha=a, label=lab)
    ax.plot([q["ep"] for q in s0], [q["plugin"] for q in s0], color=C_GEN, lw=1.2, ls=(0, (3, 2)),
            label="RCC, own answers: Bayes with its Generator's likelihood")
    for k in range(2):
        r = jl(f"genC3_60k_lr4_s{k}")
        ax.plot([q["ep"] for q in r], [q["plugin"] for q in r], color=C_ORC, lw=1.2, ls=(0, (3, 2)),
                label="oracle labels: Bayes with the Generator's likelihood" if k == 0 else None)
    ep = np.array([q["ep"] for q in s0], float); y = np.array([q["net"] for q in s0])
    f = lambda t, a, b, tau: a - b * np.exp(-(t - 60000) / tau)
    t = np.linspace(120000, 470000, 200)
    for lo, dy in ((60000, -.035), (80000, .02)):
        m = ep >= lo
        p, cov = curve_fit(f, ep[m], y[m], p0=(.5, .2, 2e4), maxfev=20000)
        se = np.sqrt(cov[0, 0])
        ax.plot(t, f(t, *p), color=C_GEN, lw=2, ls=(0, (1, 1.5)))
        ax.text(468000, f(468000, *p) + dy, f"fit from {lo // 1000}k: ceiling {p[0]:.2f}" + (f" $\\pm$ {se:.2f}" if se >= .005 else ""),
                fontsize=L - 1, ha="right", color=C_GEN)
    m = ep >= 100000
    sl, ic = np.polyfit(ep[m], y[m], 1)
    t_hit = (joint - ic) / sl
    tl = np.linspace(120000, t_hit, 50)
    ax.plot(tl, sl * tl + ic, color="#6a51a3", lw=2, ls=(0, (4, 2, 1, 2)))
    ax.text(250000, .6, f"if the last 20k's slope held\n(+{sl * 1e4:.3f} per 10k)", fontsize=L - 1, ha="left",
            color="#6a51a3")
    ax.axhline(joint, color="k", ls="--", lw=1); ax.axhline(naive, color="k", ls=":", lw=1)
    ax.text(450000, joint + .012, f"Joint Bayes {joint:.2f}", fontsize=L, ha="right")
    ax.text(468000, naive + .012, f"Naive Bayes {naive:.2f}", fontsize=L, ha="right")
    for x, s in ((t_hit, f"+{(t_hit - 120000) / RATE / 3600:.0f} h"),
                 (120000 + 36 * 3600 * RATE, f"+36 h")):
        ax.axvline(x, color=".75", lw=1, zorder=0)
        ax.text(x, .69, s, fontsize=L, ha="center", color=".35", bbox=dict(fc="w", ec="none", pad=1))
    ax.text(121000, .655, "resume\n(1 GPU)", fontsize=L - 2, ha="left", va="bottom", color=".35")
    ax.axvline(120000, color=".75", lw=1, zorder=0)
    ax.set(xlim=(0, 470000), ylim=(0, .72), xlabel="training iterations (10,000 episodes each)",
           ylabel="goal accuracy, novel environments")
    ax.set_xticks(range(0, 400001, 100000), [f"{v // 1000}k" for v in range(0, 400001, 100000)])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("RCC with 3 latent variables (C = 3): Joint only if the slope stops falling\n"
                 r"$|\mathcal{S}|$ = 500, batch 10,000; every point = 10,000 novel episodes; 1 cluster GPU = 2.6 it/s",
                 loc="left", fontsize=fs["title"] - 4)
    ax.legend(frameon=False, fontsize=L - 1, loc="lower right", bbox_to_anchor=(1, .06))
    out = fig_path("fig_c3_ceiling.png")
    fig.savefig(out, dpi=150)
    print(out, f"joint {joint:.3f} naive {naive:.3f} slope {sl * 1e4:.4f}/10k hit {t_hit:.0f}")


if __name__ == "__main__":
    main()
