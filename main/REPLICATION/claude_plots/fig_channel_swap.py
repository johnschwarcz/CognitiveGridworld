"""Channel order swapped at test (John 11-12 04:20: "if the classifier route really cant handle a shared readin, then it
shouldn't be able to Generalize to swapping observation order (whereas the generator is inherently invariant to that)").

Data: main/DATA/experimental/clf_scratch/swap/swap_results.json (workflow 11-12 04:30, scripts in the same folder). Fig 13's
saved nets, seeds 0-1 of each arm (gen60k_lr4 / clf60k_lr4: oracle labels; rcc60k_lr4 / rccclf60k_lr4: the RCC's own
answers; lr 1e-4, 60k iterations). A swap pi relabels the channels consistently: channel o's bits are shown at position
pi(o), and the world's likelihood table and the net's per-channel embedding tables move with them (Joint Bayes unchanged,
.7715); read-in, LSTM and Generator untouched. The same 2,000 novel episodes for every permutation; the identity, the 4
cyclic shifts, the 10 transpositions and 5-15 random permutations per net. x = number of channels moved (0 = identity,
2 = one transposition); dots = permutations, lines = their mean.
    a  the net's own Classifier (LSTM with the default dense read-in)
    b  the net's Generator read out by exact Bayes
    c  (John 11-12 11:40: "the channel swap needs to hold up to a classifier route shared readin condition as well") the
       Classifier with the shared read-in (fig_order_invariant.py): genshared20k_lr4 / genshared10k_lr1e3 (Generator
       route), clfshared60k_lr4 / clfshared10k_lr1e3 (Classifier route), as far as their nets are swap-tested
    d  the dense read-in trained on shuffled channel orders (genperm20k_lr4 / clfperm20k_lr4; the order is also shuffled
       at test, so a swap is one more shuffle)
       c-d: swap_clf.py on the nets pulled to fig13_embeddings/nets/cluster_runs (scratchpad swap_more.sh, 11-12 12:00),
       per-net outputs in out/clf_<net>.json; same 2,000 novel episodes, the first 20 permutations

RESULT 11-13 01:45 (the RCC-feedback Classifier-route nets continued to 120k, rccclf120k_lr4_s0-3, all 5 channels in use
again; out/clf_rccclf120k_lr4_s*.json, local swap_more.sh): identity .709-.713, one transposition .18-.21 (range
.145-.361), 3 or more channels moved .107-.131: the swap failure does not fade with training.
RESULT c at lr 1e-4 (11-12 13:29, clfshared60k_lr4 s0-3 at 60k): exactly invariant at chance, .107 / .108 / .116 / .117;
the Generator route .648 / .660 (lr 1e-4) and .710 / .713 (lr 1e-3).
RESULT c-d (11-12 12:05): shared read-in Classifiers are exactly invariant (same hits for all 20 orders), the Generator
route's at .648 / .6605, the Classifier route's at chance (.1155 / .1165, lr 1e-3); shuffled orders: Generator route
.68-.71, Classifier route s1 .65-.67, s0 chance (.11-.12). Order invariance imposed on the Classifier costs the Classifier
route its learning (shared: 0 of 2 here, 0 of 6 in fig_order_invariant; shuffled: 1 of 2), never the Generator route.
RESULT (agents 11-12 06:00, verified independently on fresh episodes): Generator-route Classifiers .734 -> .727 under swaps
(range .720-.740: unchanged within one paired SE); Classifier-route Classifiers .707 -> .151 (one transposition already
.17-.20; three or more channels moved .11-.12). Generator + Bayes is exactly invariant (one Generator for every channel):
.744 / .691 (Generator route), .142 (Classifier route, whose Generator never learned). Why: Generator-route codes share
one convention in every slot, so a swapped episode looks like an ordinary novel one; Classifier-route codes sit in
shuffled slots with 2-3 different sign / exchange conventions. Caveat (verifier): a fixed per-sensor relabelling of the
Classifier-route codes, applied before the reordering, makes its Classifier order-invariant too (.71 at every order);
a learned map applied after the reordering does not (.72 -> .15).

    python fig_channel_swap.py
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

SRC = os.path.join(ROOT, "main", "DATA", "experimental", "clf_scratch", "swap", "swap_results.json")
STYLE = {"gen60k_lr4": ("#08519c", "-", "Generator route, oracle labels"),
         "rcc60k_lr4": ("#6baed6", "-", "Generator route, RCC's own labels"),
         "clf60k_lr4": ("#a50f15", "-", "Classifier route, oracle labels"),
         "rccclf60k_lr4": ("#fc9272", "-", "Classifier route, RCC's own labels")}


OUT = os.path.join(os.path.dirname(SRC), "out")
EXTRA = (("c   Classifier with the shared read-in", (("genshared20k_lr4", "#08519c", "-", "Generator route, lr 1e-4"),
                                                   ("genshared10k_lr1e3", "#08519c", (0, (4, 2)), "Generator route, lr 1e-3"),
                                                   ("clfshared60k_lr4", "#a50f15", "-", "Classifier route, lr 1e-4"),
                                                   ("clfshared10k_lr1e3", "#a50f15", (0, (4, 2)), "Classifier route, lr 1e-3"))),
         ("d   Dense read-in trained on shuffled channel orders", (("genperm20k_lr4", "#08519c", "-", "Generator route"),
                                                                  ("clfperm20k_lr4", "#a50f15", "-", "Classifier route"))))


def extra_nets(arm):
    """[(net, moved array, accuracy array)] for the swap-tested seeds of one arm (out/clf_<net>.json)"""
    res = []
    for k in range(6):
        f = os.path.join(OUT, f"clf_{arm}_s{k}.json")
        if os.path.exists(f):
            it = json.load(open(f))["items"]
            ps = [it[f"p{i}"] for i in range(20) if f"p{i}" in it]
            if len(ps) == 20:
                res.append((f"{arm}_s{k}", np.array([p["moved"] for p in ps]), np.array([p["clf"] for p in ps])))
    return res


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    L = fs["legend"] - 1
    nets = json.load(open(SRC))["nets"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), sharey=True, constrained_layout=True)
    axes = axes.ravel()
    seen = set()
    for name, d in nets.items():
        arm = name.rsplit("_s", 1)[0]
        col, ls, lab = STYLE[arm]
        for ax, key in zip(axes[:2], ("clf", "plugin")):
            moved = np.array([p["moved"] for p in d["perms"]]); acc = np.array([p[key] for p in d["perms"]])
            xs = np.unique(moved)
            ax.scatter(moved + np.random.default_rng(hash(name) % 2**32).uniform(-.12, .12, len(moved)), acc, s=14,
                       color=col, alpha=.5, lw=0)
            ax.plot(xs, [acc[moved == x].mean() for x in xs], color=col, ls=ls, lw=2.4,
                    label=lab if (arm, key) not in seen else None, marker="o", ms=5)
            seen.add((arm, key))
            print(f"{key:6s} {name:18s} identity {acc[moved == 0][0]:.3f} -> swaps mean {acc[moved > 0].mean():.3f} "
                  f"[{acc[moved > 0].min():.3f}-{acc[moved > 0].max():.3f}]")
    for ax, (title, arms) in zip(axes[2:], EXTRA):
        for arm, col, ls, lab in arms:
            got = extra_nets(arm)
            for i, (name, moved, acc) in enumerate(got):
                xs = np.unique(moved)
                ax.scatter(moved + np.random.default_rng(hash(name) % 2**32).uniform(-.12, .12, len(moved)), acc, s=14,
                           color=col, alpha=.5, lw=0)
                ax.plot(xs, [acc[moved == x].mean() for x in xs], color=col, ls=ls, lw=2.4, marker="o", ms=5,
                        label=f"{lab} ({len(got)} seeds)" if i == 0 else None)
                print(f"clf    {name:22s} identity {acc[moved == 0][0]:.3f} -> swaps mean {acc[moved > 0].mean():.3f} "
                      f"[{acc[moved > 0].min():.3f}-{acc[moved > 0].max():.3f}]")
        ax.set_title(title, loc="left", fontsize=fs["title"] - 3)
        ax.legend(frameon=False, fontsize=L, loc="center right", bbox_to_anchor=(1, .45))
    joint = np.mean([p["joint"] for d in nets.values() for p in d["perms"]])
    for ax, title in zip(axes[:2], ("a   The network's own Classifier (dense read-in)", "b   Its Generator, read out by exact Bayes")):
        ax.set_title(title, loc="left", fontsize=fs["title"] - 3)
    for ax in axes:
        ax.axhline(joint, color="k", ls="--", lw=1)
        ax.axhline(.1, color=".5", ls=":", lw=1)
        ax.text(5.3, joint + .012, "Joint Bayes", fontsize=L, ha="right")
        ax.text(-.3, .1 - .035, "chance", fontsize=L, ha="left", color=".35")
        ax.set(xlim=(-.4, 5.4), ylim=(0, .82), xticks=[0, 2, 3, 4, 5], xlabel="channels moved to another position")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[::2]:
        ax.set_ylabel("goal accuracy, novel environments")
    axes[0].legend(frameon=False, fontsize=L, loc="center right", bbox_to_anchor=(1, .62))
    fig.suptitle("Channels shown in a new order at test (a-b: Fig 13 nets, seeds 0-1 per arm; c-d: order-invariant "
                 "Classifiers; 2,000 novel episodes per point; dots = permutations)", fontsize=fs["legend"] - 1)
    out = fig_path("fig_channel_swap.png")
    fig.savefig(out, dpi=170)
    print(out)


if __name__ == "__main__":
    main()
