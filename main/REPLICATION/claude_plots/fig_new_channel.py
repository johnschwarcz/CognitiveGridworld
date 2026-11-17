"""A channel never seen in training, added at test (John 11-12 05:00: "for new channels i meant what you showed in this
figure" = fig13_summary panel g; here repeated on all 12 oracle nets).

Data: main/REPLICATION/trainings/transfer_reader.py TASKS 110-133 (cluster array 66673338, 11-12 06:10-06:28), pulled to
main/DATA/experimental/transfer_reader_newch/. Sources: Fig 13's 60k nets (oracle labels, lr 1e-4, seeds 0-5),
gen60k_lr4 (Generator route) and clf60k_lr4 (Classifier route); their embeddings give each environment its code, FROZEN.
A fresh shared reader (one evidence function for every channel, summed over channels: Joint Bayes' form) learns from
channels 0-3 only (40,000 steps x 512 episodes = 20.5M episodes, Adam lr 1e-3, reader seed 0); at every test channel 4
(its bits and its code from the same net) is added. Classifier-route codes are read as learned, or through a learned
linear map of the 10 code numbers (shared_adapt: it can represent the exact slot + frame correction) or an MLP map
(shared_mlpmap). Each test: 2,048 novel-environment episodes, the same with and without channel 4.
    a  accuracy on novel environments: solid = channels 0-3, dashed = channel 4 added (mean of 6 nets)
    b  gain from the new channel, acc(0-4) - acc(0-3), per net (thin) and mean (thick); grey: exact Bayes' gain from a
       5th channel (.071: 5 vs 4 channels, T = 30)

Row 2 (11-12 12:30, transfer_reader TASKS 134-157, cluster array 66677688): the same test on the RCC-feedback nets
(rcc60k_lr4 = Generator route, rccclf60k_lr4 = Classifier route, the RCC's own answers as labels, seeds 0-5); drawn when
their logs exist.
RESULT 11-12 12:00 (mean of the last 5 tests): Generator route +.049 to +.074 (mean +.065; 6/6 > 0); Classifier route as
learned: the reader cannot read it at all (.14-.19), gain -.020 to +.001; with a linear map it reads channels 0-3
(.59-.62; .37 for s2, which lost 2 channels) and the new channel HURTS: -.076 to -.194 (mean -.168); MLP map -.111 to -.193
(mean -.163). 12/12 Classifier-route readers lose accuracy from the new channel.
RESULT rows 3-4 (11-13 01:45, mean of the last 5 tests): shuffled channel order: Generator route +.068 / +.080 (lr 1e-4),
+.070 / +.073 (lr 1e-3); the Classifier route's learned seed (s1) is readable AS LEARNED (.584 / .578) and the new
channel HELPS: +.068 (lr 1e-4) / +.082 (lr 1e-3), like the Generator route (its maps hurt: -.11 to -.16); its seed 0 stays
at chance (.13, gain 0). Classifier route at 120k (RCC answers, s0-3): as learned unreadable (.14-.20, gain -.004 to
-.013); linear map -.10 to -.19, MLP map -.15 to -.19: the failure does not fade with training.
RESULT row 2 (11-12 12:35, RCC-feedback nets): Generator route +.052 to +.076 (mean +.066, 6/6); Classifier route as
learned -.005 (reads .165); linear map -.075 to -.166 (mean -.099), MLP map -.093 to -.156 (mean -.126): 12/12 lose.

    python fig_new_channel.py [--pull]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

DATA = os.path.join(ROOT, "main", "DATA", "experimental", "transfer_reader_newch")
BAYES4, BAYES5 = .704, .775
CONDS = (("shared", "gen", "#08519c", "Generator route, codes as learned"),
         ("shared_adapt", "clf", "#fd8d3c", "Classifier route + learned linear map"),
         ("shared_mlpmap", "clf", "#8c510a", "Classifier route + learned MLP map"),
         ("shared", "clf", "#a50f15", "Classifier route, codes as learned"))


ROWS = (("oracle labels", {"gen": "gen60k_lr4", "clf": "clf60k_lr4"}, None),
        ("the RCC's own answers as labels", {"gen": "rcc60k_lr4", "clf": "rccclf60k_lr4"}, None),
        # 11-13 (transfer_reader TASKS 158-181): shuffled channel order in training (lr 1e-4 nets; the Classifier route's
        # seed 1, the seed that learned; seed 0 never left chance), and the Classifier route continued to 120k
        ("shuffled channel order, oracle; Classifier route: the seed that learned",
         {"gen": "genperm20k_lr4", "clf": "clfperm20k_lr4"}, {"gen": (0, 1), "clf": (1,)}),
        ("the RCC's own answers; Classifier route at 120k (Generator route 60k)",
         {"gen": "rcc60k_lr4", "clf": "rccclf120k_lr4"}, {"gen": range(6), "clf": range(4)}))


def pull():
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/transfer_reader/*-*_s*_frozen_ch4_s0_long.jsonl' "
                   f"{DATA}/", shell=True)


def runs(reader, route, nets, seeds=None):
    out = []
    for k in (seeds or {}).get(route, range(6)):
        f = os.path.join(DATA, f"{reader}_{route}-{nets[route]}_s{k}_frozen_ch4_s0_long.jsonl")
        if os.path.exists(f):
            q = [json.loads(l) for l in open(f) if l.strip()]
            out.append((np.array([r["episodes"] for r in q]) / 1e6, np.array([r["novel"] for r in q]),
                        np.array([r["novel_all5"] for r in q])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    L = fs["legend"] - 1
    rows = [(lab, nets, sd) for lab, nets, sd in ROWS if all(runs(r, rt, nets, sd) for r, rt, _, _ in CONDS)]
    fig, axes = plt.subplots(len(rows), 2, figsize=(15, 1.2 + 5.4 * len(rows)), constrained_layout=True, squeeze=False)
    for i, ((rowlab, nets, sd), (axa, axb)) in enumerate(zip(rows, axes)):
        for reader, route, col, lab in CONDS:
            rs = runs(reader, route, nets, sd)
            n = min(len(x) for x, _, _ in rs)
            x = rs[0][0][:n]
            a4 = np.mean([r[1][:n] for r in rs], 0); a5 = np.mean([r[2][:n] for r in rs], 0)
            axa.plot(x, a4, color=col, lw=2.4, label=lab if i == 0 else None)
            axa.plot(x, a5, color=col, lw=2.4, ls=(0, (4, 2)))
            for _, nv, al in rs:
                axb.plot(x, (al - nv)[:n], color=col, lw=.8, alpha=.45)
            axb.plot(x, a5 - a4, color=col, lw=3)
            g = np.array([np.mean(r[2][-5:] - r[1][-5:]) for r in rs])
            print(f"{rowlab[:10]:10s} {reader:14s} {route}: n {len(rs)}, acc 0-3 {np.mean([r[1][-5:].mean() for r in rs]):.3f}, "
                  f"gain mean {g.mean():+.3f} [{g.min():+.3f}, {g.max():+.3f}]")
        for y, t in ((BAYES5, "exact Bayes, 5 channels"), (BAYES4, "exact Bayes, 4 channels")):
            axa.axhline(y, color=".55", ls=":", lw=1.2)
            axa.text(20.3, y + .01, t, fontsize=L, color=".3", ha="right")
        axa.axhline(.1, color=".55", ls=":", lw=1)
        axa.text(20.3, .1 + .01, "chance", fontsize=L, color=".3", ha="right")
        axa.set(xlim=(0, 20.5), ylim=(0, .82), xlabel="reader training episodes (millions)",
                ylabel="goal accuracy, novel environments")
        axa.set_title(f"{'aceg'[i]}   Reader trained on channels 0-3\n({rowlab})", loc="left",
                      fontsize=fs["title"] - 3)
        axb.axhline(0, color="k", lw=.8)
        axb.axhline(BAYES5 - BAYES4, color=".55", ls=":", lw=1.4)
        axb.set(xlim=(0, 20.5), ylim=(-.23, .12), xlabel="reader training episodes (millions)",
                ylabel="accuracy with channel 4 $-$ without")
        axb.set_title(f"{'bdfh'[i]}   Gain from the new channel: $\\mathrm{{acc}}_{{0-4}} - \\mathrm{{acc}}_{{0-3}}$\n({rowlab})",
                      loc="left", fontsize=fs["title"] - 3)
        for s_ in (axa, axb):
            s_.spines[["top", "right"]].set_visible(False)
    ax = axes[0, 0]
    ax.plot([], [], color="k", lw=2, label="left: channels 0-3 (trained)")
    ax.plot([], [], color="k", lw=2, ls=(0, (4, 2)), label="left: + channel 4, never seen")
    ax.plot([], [], color=".55", ls=":", lw=1.4, label="right: exact Bayes' gain from a 5th channel (.071)")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=L, loc="outside lower center", ncol=4)
    fig.suptitle("A channel never seen in training, added at test (a-d: 6 nets per route; e-f: 2 Generator-route nets, "
                 "1 Classifier-route seed; g-h: 4 Classifier-route nets at 120k); every point = 2,048 novel episodes",
                 fontsize=fs["legend"] - 1)
    out = fig_path("fig_new_channel.png")
    fig.savefig(out, dpi=170)
    print(out)


if __name__ == "__main__":
    main()
