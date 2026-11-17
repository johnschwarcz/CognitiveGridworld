"""Generator route vs Classifier route, where it stands (John 11-13 00:50: "where do we stand on the actual generator route
vs classifier route?" ... "can you please show me these updates"). One panel per comparison; the source plotters keep the
details and numbers in their docstrings. Blue = Generator route, red = Classifier route (the route of the embedding
gradient); Fig 13's protocol: |S| 500, d_E 30, batch 10,000, lr 1e-4, default dense read-in unless a panel says otherwise.
    a, b  learning (fig_route_control.py): Classifier's goal accuracy on novel environments, 10,000 episodes per test; thin
          = seeds, thick = mean where every seed has a test; Classifier-route seeds continued from 60k to 120k (RESUME)
    c     C = 3 (fig_c3_routes.py): the same with three latent variables, the RCC's own answers as labels
    d     channel use (fig_channel_census.py): goal accuracy lost when ONE channel's bits are shuffled across 2,000 novel
          episodes (0 = the network ignores that channel); the same seed at 60k and after its continuation
    e     channel swap at test (fig_channel_swap.py): the net's own Classifier, channels relabelled consistently (bits and
          embedding tables moved together), 2,000 novel episodes; mean over permutations and seeds 0-1
    f     a channel never seen in training (fig_new_channel.py): a fresh shared reader trained on channels 0-3 of the
          frozen codes; the change in accuracy when channel 4 is added at test (2,048 novel episodes per point)
    g     shared read-in (fig_order_invariant.py): the Classifier made order-invariant by architecture, lr 1e-4

    python fig_routes_status.py [--pull]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
from main.utils import fig_path, apply_fig_style

DATA = os.path.join(ROOT, "main", "DATA")
LOGS = os.path.join(DATA, "experimental", "cluster_runs", "jsonl")
SWAP = os.path.join(DATA, "experimental", "clf_scratch", "swap", "swap_results.json")
GEN, CLF = "#08519c", "#a50f15"


def jl(n):
    f = os.path.join(LOGS, n + ".jsonl")
    return [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []


def curves(arm, ext=None, seeds=range(6)):
    out = []
    for k in seeds:
        r = (jl(f"{ext}_s{k}") if ext else []) or jl(f"{arm}_s{k}")
        if r:
            out.append((np.array([q["ep"] for q in r]), np.array([q["net"] for q in r])))
    return out


def learn(ax, arms, joint, naive, xmax):
    for (arm, ext), col in zip(arms, (GEN, CLF)):
        cs = curves(arm, ext)
        for x, y in cs:
            ax.plot(x, y, color=col, lw=.8, alpha=.5)
        n = min(len(x) for x, _ in cs)
        ax.plot(cs[0][0][:n], np.mean([y[:n] for _, y in cs], 0), color=col, lw=2.6)
    refs(ax, joint, naive, xmax)
    return [len(curves(a, e)) for a, e in arms]


def refs(ax, joint, naive, xmax, chance=True):
    ax.axhline(joint, color="k", ls="--", lw=1)
    ax.axhline(naive, color="k", ls=":", lw=1.2)
    if chance:
        ax.axhline(.1, color=".6", ls=":", lw=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/{{rccclf120k,rccC3,rccclfC3}}*.jsonl' "
                       f"{LOGS}/", shell=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fig_channel_census import parse
    from fig_new_channel import runs as newch_runs, ROWS as NEWCH_ROWS, BAYES4, BAYES5
    fs = apply_fig_style()
    L, T = fs["legend"] - 1, fs["title"] - 4
    joint, naive = np.load(os.path.join(DATA, "cache", "bayes_bounds_ctx2_T30.npy"))
    fig, axes = plt.subplots(2, 4, figsize=(23, 11), constrained_layout=True)
    ax = axes.ravel()
    xl = "training iterations (10,000 episodes each)"

    # a, b learning, extended to 120k where continued
    na = learn(ax[0], (("gen60k_lr4", None), ("clf60k_lr4", "clf120k_lr4")), joint, naive, 120000)
    nb = learn(ax[1], (("rcc60k_lr4", None), ("rccclf60k_lr4", "rccclf120k_lr4")), joint, naive, 120000)
    nb120 = sum(x[-1] >= 119999 for x, _ in curves("rccclf60k_lr4", "rccclf120k_lr4"))
    ax[0].set_title(f"a   Oracle labels ({na[0]} + {na[1]} seeds)\nthe seed that lost 2 channels continued to 120k",
                    loc="left", fontsize=T)
    ax[1].set_title(f"b   The RCC's own answers as labels ({nb[0]} + {nb[1]} seeds)\nClassifier route: {nb120} of {nb[1]} "
                    f"seeds continued to 120k", loc="left", fontsize=T)
    for a in ax[:2]:
        a.set(xlim=(0, 120000), ylim=(0, .82), xlabel=xl, xticks=[0, 30000, 60000, 90000, 120000],
              xticklabels=["0", "30k", "60k", "90k", "120k"])
        a.text(119000, joint + .012, "Joint Bayes", fontsize=L, ha="right")
        a.text(119000, naive + .012, "Naive Bayes", fontsize=L, ha="right")
    ax[0].set_ylabel("goal accuracy, novel environments")

    # c C = 3
    c3 = []
    for arm in ("rccC3_60k_lr4", "rccclfC3_60k_lr4"):
        for k in range(3):
            r, ext = jl(f"{arm}_s{k}"), jl(f"{arm.replace('60k', '120k')}_s{k}")
            r = ext if len(ext) > len(r) else r                       # continued from its 60k net (log starts with it)
            if len(r) > 1:
                c3.append((arm, k, r))
    for arm, k, r in c3:
        ax[2].plot([q["ep"] for q in r], [q["net"] for q in r], color=CLF if "clf" in arm else GEN, lw=(2.2, 1.4, 1)[k],
                   alpha=(1, .75, .55)[k])
    j3, n3 = (np.mean([q[key] for _, _, r in c3 for q in r]) for key in ("joint", "naive"))
    refs(ax[2], j3, n3, 120000)
    ax[2].set(xlim=(0, 120000), ylim=(0, .82), xlabel=xl, xticks=[0, 30000, 60000, 90000, 120000],
              xticklabels=["0", "30k", "60k", "90k", "120k"])
    ax[2].text(119000, j3 + .012, "Joint Bayes", fontsize=L, ha="right")
    ax[2].text(2000, n3 + .012, "Naive Bayes", fontsize=L, ha="left")
    runs = lambda tag: ", ".join(f"s{k} {r[-1]['ep'] / 1000:.0f}k" for a, k, r in c3 if ("clf" in a) == (tag == "clf"))
    ax[2].set_title(f"c   C = 3 (1,000 joint values), the RCC's own answers\nGenerator {runs('gen')}; Classifier {runs('clf')}",
                    loc="left", fontsize=T)

    # d channel census
    D = parse()
    G = [D[n]["drop"] for n in D if n.startswith(("gen", "rcc60k"))]
    rows = [(f"Generator route, {len(G)} nets (lowest)", np.min(G, 0))]
    for k in range(6):
        rows.append((f"RCC answers s{k}  60k", D[f"rccclf60k_lr4_s{k}"]["drop"]))
        if f"rccclf120k_lr4_s{k}" in D:
            rows.append((f"RCC answers s{k}  120k", D[f"rccclf120k_lr4_s{k}"]["drop"]))
    rows += [("oracle s2  60k", D["clf60k_lr4_s2"]["drop"]), ("oracle s2  120k", D["clf120k_lr4_s2"]["drop"])]
    M = np.clip(np.array([r for _, r in rows]), 0, None)
    im = ax[3].imshow(M, cmap="viridis", vmin=0, vmax=.3, aspect="auto")
    for i, row in enumerate(M):
        for j, v in enumerate(row):
            ax[3].text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=L,
                       color="k" if v > .17 else "w")
    ax[3].set_yticks(range(len(rows)), [r for r, _ in rows], fontsize=L)
    for i, (lab, _) in enumerate(rows):
        if "120k" in lab:
            ax[3].get_yticklabels()[i].set_fontweight("bold")
    ax[3].set_xticks(range(5), [f"ch {c}" for c in range(5)], fontsize=L)
    ax[3].set_title("d   Accuracy lost when one channel is shuffled (0 = unused)\nrows below the first: Classifier route; "
                    "2,000 novel episodes", loc="left", fontsize=T)
    cb = fig.colorbar(im, ax=ax[3], fraction=.05, pad=.02)
    cb.ax.tick_params(labelsize=L - 1)

    # e channel swap, own Classifier
    nets = json.load(open(SWAP))["nets"]
    for arm, col, ls in (("gen60k_lr4", GEN, "-"), ("rcc60k_lr4", GEN, (0, (4, 2))), ("clf60k_lr4", CLF, "-"),
                         ("rccclf60k_lr4", CLF, (0, (4, 2)))):
        P = [p for n, d in nets.items() if n.rsplit("_s", 1)[0] == arm for p in d["perms"]]
        mv = np.array([p["moved"] for p in P]); acc = np.array([p["clf"] for p in P])
        xs = np.unique(mv)
        ax[4].plot(xs, [acc[mv == x].mean() for x in xs], color=col, ls=ls, lw=2.4, marker="o", ms=6)
    sj = np.mean([p["joint"] for d in nets.values() for p in d["perms"]])
    refs(ax[4], sj, np.mean([d["naive"] for d in nets.values()]), 5)
    ax[4].set(xlim=(-.3, 5.3), ylim=(0, .82), xticks=[0, 2, 3, 4, 5], xlabel="channels moved to another position")
    ax[4].set_title("e   Channels shown in a new order at test\n(seeds 0-1 per arm; solid oracle, dashed RCC answers)",
                    loc="left", fontsize=T)
    ax[4].set_ylabel("goal accuracy, novel environments")

    # f new channel, RCC-answer nets
    lab, netsn, _ = NEWCH_ROWS[1]
    for reader, route, col in (("shared", "gen", GEN), ("shared", "clf", CLF), ("shared_adapt", "clf", "#fd8d3c"),
                               ("shared_mlpmap", "clf", "#8c510a")):
        rs = newch_runs(reader, route, netsn)
        n = min(len(x) for x, _, _ in rs)
        for x, a4, a5 in rs:
            ax[5].plot(x[:n], (a5 - a4)[:n], color=col, lw=.7, alpha=.4)
        ax[5].plot(rs[0][0][:n], np.mean([(a5 - a4)[:n] for _, a4, a5 in rs], 0), color=col, lw=2.6)
    ax[5].axhline(0, color="k", lw=.8)
    ax[5].axhline(BAYES5 - BAYES4, color="k", ls="--", lw=1)
    ax[5].text(20.3, BAYES5 - BAYES4 + .008, "exact Bayes' gain", fontsize=L, ha="right")
    ax[5].set(xlim=(0, 20.5), ylim=(-.23, .12), xlabel="reader training episodes (millions)")
    ax[5].set_ylabel("accuracy with channel 4 $-$ without")
    ax[5].set_title("f   A channel never seen in training, added at test\n(RCC-answer nets, 6 per line)", loc="left",
                    fontsize=T)

    # g shared read-in
    for arm, col, n in (("genshared20k_lr4", GEN, 2), ("clfshared60k_lr4", CLF, 4)):
        for x, y in curves(arm, seeds=range(n)):
            ax[6].plot(x, y, color=col, lw=1.6)
    refs(ax[6], joint, naive, 60000)
    ax[6].set(xlim=(0, 60000), ylim=(0, .82), xlabel=xl, xticks=[0, 20000, 40000, 60000], xticklabels=["0", "20k", "40k", "60k"])
    ax[6].text(59000, joint + .012, "Joint Bayes", fontsize=L, ha="right")
    ax[6].text(59000, naive + .012, "Naive Bayes", fontsize=L, ha="right")
    ax[6].set_title("g   Classifier with a shared (order-invariant) read-in\n(oracle labels; 2 + 4 seeds)", loc="left",
                    fontsize=T)

    # h legend
    ax[7].axis("off")
    H = [plt.Line2D([], [], color=GEN, lw=3, label="Generator route"),
         plt.Line2D([], [], color=CLF, lw=3, label="Classifier route"),
         plt.Line2D([], [], color="#fd8d3c", lw=3, label="f: Classifier route + learned linear map"),
         plt.Line2D([], [], color="#8c510a", lw=3, label="f: Classifier route + learned MLP map"),
         plt.Line2D([], [], color="k", ls="--", lw=1.2, label="Joint Bayes"),
         plt.Line2D([], [], color="k", ls=":", lw=1.4, label="Naive Bayes"),
         plt.Line2D([], [], color=".6", ls=":", lw=1.2, label="chance")]
    ax[7].legend(handles=H, loc="center left", frameon=False, fontsize=L + 4, handlelength=3)
    for a in ax[:7]:
        if a is not ax[3]:
            a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(r"Generator route vs Classifier route ($|\mathcal{S}|$ 500, lr $10^{-4}$, batch 10,000; "
                 "learning curves: 10,000 novel episodes per point)", fontsize=fs["legend"] + 1)
    out = fig_path("fig_routes_status.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
