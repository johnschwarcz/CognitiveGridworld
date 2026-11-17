"""Is the Classifier route's learning mediated by factorization regret? (John 11-14 04:00 FL: "With the classifier route, can
you tell if the embedding learning is mediated by Factorization regret?")

Every test of the route runs (train_cluster_runs.py runner) splits its novel-environment episodes into fifths of the forward
factorization regret at T (FR_EDGES .28 / .65 / 1.5 / 4.5 nats; ~1/5 of the episodes each) and logs the network's, Joint's
and Naive's goal accuracy per fifth. If the Classifier route learned the factorizable (low-FR) part first and the interactions
(high-FR) later, then at the SAME overall accuracy it would be further above Naive in the low-FR fifths and less above
Naive in the high-FR fifths than the Generator route.
    x  overall goal accuracy of the network (trailing mean of 5 tests), levels .25, .275, ..., the first test at or above it
    y  network minus Naive accuracy in each FR fifth, same 5 tests; per seed, then the mean over the seeds that reach x
    colour = FR fifth (viridis, bright = high FR); solid = Generator route, dashed = Classifier route
Columns: batch 10,000 (rcc60k_lr4 / rccclf60k_lr4 -> rccclf120k_lr4, 6 + 6 seeds, ~2,000 test episodes per fifth); batch
1,000 (fig_batch_routes.PANELS[1]); batch 1,000 with shuffled channels (rccpermb1k1m_lr4 / rccclfpermb1k1m_lr4). All with the
RCC's own feedback, lr 1e-4. Generator-route runs paused past Naive end their lines early (no level above their pause).

NUMBERS (11-14 11:00 IST): at overall .45, network - Naive per fifth (low -> high FR), mean over seeds:
    batch 10,000  Generator -.207 -.079 +.072 +.113 +.119   Classifier -.229 -.091 +.069 +.120 +.144   (6 + 6)
    batch 1,000   Generator -.198 -.079 +.076 +.115 +.107   Classifier -.231 -.087 +.066 +.124 +.131   (5 + 11)
    shuffled      Generator -.213 -.066 +.044 +.103 +.092   Classifier -.208 -.083 +.072 +.113 +.131   (3 + 4)
Same profile in both routes (largest difference +.03, the Classifier route slightly further above Naive in the top fifth):
the routes travel the same path through FR space, the Classifier route only later. Both are below Naive in the low-FR
fifths and above it in the high-FR fifths already at overall .35, i.e. neither route passes through a factorized stage.

    python fig_route_fr_profile.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_batch_routes as fb
from main.utils import fig_path

LEVELS = np.round(np.arange(.25, .676, .025), 3)
FAMS = [("batch 10,000", [("Generator", ("rcc60k_lr4", "rcc120k_lr4"), range(6)),
                          ("Classifier", ("rccclf60k_lr4", "rccclf120k_lr4"), range(6))]),
        ("batch 1,000", [("Generator", fb.PANELS[1][2], range(12)), ("Classifier", fb.PANELS[1][3], range(12))]),
        ("batch 1,000, shuffled channels", [("Generator", ("rccpermb1k1m_lr4", None), range(8)),
                                            ("Classifier", ("rccclfpermb1k1m_lr4", None), range(9))])]


def profiles(arm, seeds):
    """{level: list over seeds of the (5,) network - Naive per FR fifth at the first test the 5-test mean reaches it}"""
    out = {L: [] for L in LEVELS}
    for k in seeds:
        rs = fb.load(*arm, k)
        if len(rs) < 5:
            continue
        y = np.convolve([r["net"] for r in rs], np.ones(5) / 5, mode="full")[4:len(rs)]
        for L in LEVELS:
            i = np.flatnonzero(y >= L)
            if len(i):
                w = rs[i[0]:i[0] + 5]
                out[L].append([np.mean([r[f"net_fr{q}"] - r[f"naive_fr{q}"] for r in w]) for q in range(5)])
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.serif": "cmr10", "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7.5, "axes.linewidth": .6})
    cols = plt.cm.viridis(np.linspace(0, .92, 5))
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.3), sharey=True)
    for ax, (title, arms) in zip(axs, FAMS):
        counts = []
        for route, arm, seeds in arms:
            pr = profiles(arm, seeds)
            ls = "-" if route == "Generator" else "--"
            xs = [L for L in LEVELS if len(pr[L]) >= 2]
            for q in range(5):
                ax.plot(xs, [np.mean([p[q] for p in pr[L]]) for L in xs], color=cols[q], ls=ls, lw=1.3)
            counts.append(f"{route} {max(len(v) for v in pr.values())}")
        ax.axhline(0, color="k", lw=.6)
        ax.set(xlim=(.24, .68), xlabel="overall goal accuracy (mean of 5 tests)")
        ax.set_title(f"{title}\nseeds: {', '.join(counts)}", fontsize=8, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("network minus Naive accuracy\nwithin each FR fifth")
    h = [Line2D([], [], color=cols[q], lw=1.3) for q in range(5)] + \
        [Line2D([], [], color=".3", lw=1.3), Line2D([], [], color=".3", lw=1.3, ls="--")]
    fig.legend(h, ["FR fifth 1 (lowest)", "2", "3", "4", "5 (highest)", "Generator route", "Classifier route"], ncol=7,
               loc="upper center", frameon=False, bbox_to_anchor=(.5, 1.0))
    fig.text(.5, .895, "the RCC's own feedback; per FR fifth of the test episodes, at the first test the network reaches each "
                       "overall accuracy; mean over seeds (levels reached by at least 2)", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, .88))
    out = fig_path("fig_route_fr_profile.png")
    fig.savefig(out, dpi=300)
    print(out)


if __name__ == "__main__":
    main()
