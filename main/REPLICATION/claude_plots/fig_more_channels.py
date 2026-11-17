"""Adding observation channels (John 11-13 02:05: "if it learns the shared readin we can keep adding more and more
observations to keep improving performance? is there a cap, does it saturate?").
    a  exact Bayes against the number of channels k (main/DATA/experimental/bayes_k_channels: bayes_k_channels.py, the
       Fig 13 world with obs_num = k, true embeddings, 10,000 novel episodes per k, T = 30): Joint and Naive
    b  a fresh shared reader on a Fig 13 net's frozen codes (transfer_reader.py; 40,000 steps x 512 episodes, Adam 1e-3),
       trained on channels 0-1 ("frozen_ch2", TASKS 182-191) or 0-3 ("frozen_ch4", TASKS 110-133), then tested with
       the never-seen channels added one at a time; mean of the last 5 tests (2,048 novel episodes each) per net; thin =
       nets, thick = mean. Generator route gen60k_lr4 s0-5, Classifier route clf60k_lr4 s0-1 (as learned / + the linear
       map), oracle labels. Purple: the RCC's own Classifier with the shared read-in, trained reading channels 0-1 only
       (train_cluster_runs.py genshared2ch10k_lr1e3_b4k_s*, batch 4,000, lr 1e-3, 10k iterations; the Generator learned all
       5 codes), tested on 2,000 novel episodes with channels 0..k-1 (read-in averaged over them; swap/channel_up.py).
RESULT 11-13 03:15 (seed 0): the RCC's own Classifier trained on 2: .383 -> .487 -> .545 -> .557 (Bayes .447 / .592 /
.702 / .771): 70%, 53%, 17% of Bayes' gains: it uses a 3rd unseen channel well, then saturates (the LSTM after the
averaged read-in was calibrated on 2 channels; summing at the training weight instead: .487 / .530 / .529, no better).
RESULT 11-13 01:45: Generator route, trained on 2 channels: .390 -> .514 -> .601 -> .661 with 1, 2, 3 never-seen channels
added (Bayes .448 / .596 / .701 / .766: 84%, 83%, 92% of Bayes' gains); trained on 4: .608 -> .673. Extending a
2-channel reader to 4 channels nearly equals training on 4 (.601 vs .608): the extrapolation is close to free; the gap to
Bayes (~.09 at 4-5 channels) is the reader's own ceiling, the same when it was trained on those channels. Classifier route:
as learned at chance (.15-.17); with the linear map every added channel HURTS (trained on 2: .294 -> .228 -> .196 ->
.175; trained on 4: .601 -> .421). Exact Bayes saturates toward 1: .91 at 10 channels, .98 at 20, .99 at 30.

    python fig_more_channels.py [--pull]
"""
import os, sys, json, glob, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

EXP = os.path.join(ROOT, "main", "DATA", "experimental")
LADDER = os.path.join(EXP, "bayes_k_channels", "ladder_T30_B10000.jsonl")
NEWCH = os.path.join(EXP, "transfer_reader_newch")
GEN, CLF, MAP = "#08519c", "#a50f15", "#fd8d3c"


def pull():
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/transfer_reader/*_frozen_ch2_s0_long.jsonl' {NEWCH}/",
                   shell=True)


def tail_mean(f, keys):
    q = [json.loads(l) for l in open(f) if l.strip()][-5:]
    return [np.mean([r[k] for r in q]) for k in keys]


def curves(reader, route, net, train_k):
    """[(k values, accuracies)] per net: the trained prefix, then each larger prefix (never-seen channels added)"""
    out = []
    for f in sorted(glob.glob(os.path.join(NEWCH, f"{reader}_{route}-{net}_s*_frozen_ch{train_k}_s0_long.jsonl"))):
        keys = ["novel"] + ([f"novel_k{j}" for j in range(train_k + 1, 6)] if train_k != 4 else ["novel_all5"])
        out.append((np.arange(train_k, 6), np.array(tail_mean(f, keys))))
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
    lad = [json.loads(l) for l in open(LADDER)]
    k = np.array([r["k"] for r in lad]); J = np.array([r["joint_T30"] for r in lad]); N = np.array([r["naive_T30"] for r in lad])
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    a.plot(k, J, "-o", color="k", lw=2.2, ms=5, label="Joint Bayes")
    a.plot(k, N, ":o", color="k", lw=1.8, ms=4, label="Naive Bayes")
    a.axhline(1, color=".6", lw=1)
    a.axvspan(.5, 5.5, color=".92", zorder=0)
    a.text(5.7, .12, "Fig 13's world: 5 channels", fontsize=L, color=".35")
    a.set(xlim=(0, 31), ylim=(0, 1.02), xlabel="number of observation channels $k$", ylabel="goal accuracy at $T = 30$")
    a.set_title("a   Exact Bayes with $k$ channels (10,000 novel episodes per $k$)", loc="left", fontsize=fs["title"] - 4)
    a.legend(frameon=False, fontsize=L, loc="lower right")
    b.plot(k[k <= 5], J[k <= 5], "-o", color="k", lw=2.2, ms=5)
    b.plot(k[k <= 5], N[k <= 5], ":o", color="k", lw=1.8, ms=4)
    for reader, route, net, col, lab in (("shared", "gen", "gen60k_lr4", GEN, "Generator route"),
                                         ("shared", "clf", "clf60k_lr4", CLF, "Classifier route, codes as learned"),
                                         ("shared_adapt", "clf", "clf60k_lr4", MAP, "Classifier route + learned linear map")):
        for tk, ls in ((2, "-"), (4, (0, (4, 2)))):
            cs = curves(reader, route, net, tk)
            if route == "clf" and tk == 4:
                cs = cs[:2]                                    # the same two nets as the 2-channel readers
            if not cs:
                continue
            for x, y in cs:
                b.plot(x, y, color=col, ls=ls, lw=.8, alpha=.45)
            m = np.mean([y for _, y in cs], 0)
            b.plot(cs[0][0], m, color=col, ls=ls, lw=2.6, marker="o", ms=6, label=f"{lab}, trained on {tk} ({len(cs)} nets)")
            b.plot(cs[0][0][:1], m[:1], "s", color=col, ms=9, mfc="w", mew=2)
            print(f"{lab:40s} trained on {tk}: " + "  ".join(f"k={x} {v:.3f}" for x, v in zip(cs[0][0], m)))
    up = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(EXP, "clf_scratch", "swap", "out",
                                                                   "up_genshared2ch10k_lr1e3_b4k_s*.json")))]
    if up:                                     # the RCC's own Classifier (shared read-in -> LSTM), trained on 2 channels
        ks = np.arange(2, 6)
        Y = np.array([[u["by_k"][str(x)]["mean"] for x in ks] for u in up])
        for y in Y:
            b.plot(ks, y, color="#6a3d9a", lw=.8, alpha=.45)
        b.plot(ks, Y.mean(0), color="#6a3d9a", lw=2.6, marker="o", ms=6,
               label=f"RCC's own Classifier (shared read-in), trained on 2 ({len(up)} net{"s" if len(up) > 1 else ""})")
        b.plot(ks[:1], Y.mean(0)[:1], "s", color="#6a3d9a", ms=9, mfc="w", mew=2)
        print("RCC's own Classifier trained on 2: " + "  ".join(f"k={x} {v:.3f}" for x, v in zip(ks, Y.mean(0))))
    b.text(5.08, J[k == 5][0], "Joint Bayes", fontsize=L, va="center")
    b.text(5.08, N[k == 5][0], "Naive Bayes", fontsize=L, va="center")
    b.set(xlim=(.7, 5.75), ylim=(0, .82), xticks=range(1, 6), xlabel="number of channels read ($k$)")
    b.set_title("b   A shared reader trained on the first channels (open square),\nthen given never-seen channels",
                loc="left", fontsize=fs["title"] - 4)
    b.legend(frameon=False, fontsize=L - 1, loc="upper left")
    for x in (a, b):
        x.spines[["top", "right"]].set_visible(False)
    fig.suptitle(r"More channels: the ceiling and what the learned codes reach ($|\mathcal{S}|$ = 500, $T$ = 30)",
                 fontsize=fs["legend"])
    out = fig_path("fig_more_channels.png")
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
