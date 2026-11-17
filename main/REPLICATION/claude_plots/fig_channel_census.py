"""Which observation channels a trained network uses (workflow memo 11-12, E1; cluster job 66671499,
scratchpad channel_census.py = the agents' ablate2.py run on the cluster's saved nets).

Per net, on 2,000 novel episodes: the goal accuracy lost when ONE true channel's bits are shuffled across episodes (a
channel the network ignores costs nothing), and how well the model slot holding that channel's code copies the true one
(best-slot linear R^2). Nets: Fig 13's runs (train_cluster_runs.py, |S| 500): Generator route (gen60k_lr4_s2), healthy
Classifier-route nets (clf30k_s2, clf60k_lr4_s0, rccclf60k_lr4_s2) and the ones that stalled (clf30k_s0 / s1 / s3 at
lr 1e-3 near .63, clf60k_lr4_s2 near .52, rccclf60k_lr4_s3 near .51).

RESULT 11-12 00:20: every stalled net ignores whole channels, every healthy one uses all 5:
    clf30k_s0 (.63)  channel 0 unused (no slot codes it, R^2 .01-.03)
    clf30k_s1 (.64)  channel 0 unused; two slots both code channel 4 (R^2 .90 / .97)
    clf30k_s3 (.61)  channel 2 unused
    clf60k_lr4_s2 (.52)  channels 2 and 3 unused; their slots code them on TRAINING environments only (R^2 .30 / .35 vs
                     ~.01 novel): the spare slots memorize
    rccclf60k_lr4_s3 (.51)  channel 2 unused, channel 4 half used
    gen60k_lr4_s2, clf30k_s2, clf60k_lr4_s0, rccclf60k_lr4_s2: all 5 channels used, R^2 .99
    11-12 01:45 part 2 (job 66671962, the RCC-feedback seeds at 60k): s0 (.56) channel 0 unused; s1 (.58) channel 4
                     barely used (shuffling it costs .06 vs ~.25); s4 (.38) channels 0 unused, 4 barely, 1 weak; s5
                     (.56) channel 0 unused, two slots both code channel 3. With s3 (channel 2) and s2 (all 5): 5 of 6
                     RCC-feedback Classifier-route nets miss or barely use a channel at 60k.
    Count over protocols: Classifier route 9 of 16 runs (oracle lr 1e-4 1/6, lr 1e-3 3/4, RCC feedback 5/6), Generator
    route 0 of 16 (all end >= .69; gen60k_lr4_s2 checked directly).
The plateau levels match Bayes with k of the 5 channels (the agents' k-channel ladder).

    python fig_channel_census.py

RESULT 11-13 00:40 (the RCC-feedback Classifier-route nets continued to 120k, cluster job 66683124): all 5 channels in
use in s0-3 (shuffle drop .26-.30 per channel); at 60k s0 lacked channel 0 (.006), s1 channel 4 (.055), s3 channel 2
(-.004) -> lost channels came back with training. The oracle lr 1e-4 seed s2 (2 channels lost) stays at .53 to 120k.
RESULT 11-13 12:55 (s4 / s5 at 120k, cluster job 66685419): s4 (.697) uses all 5 channels (drop .25-.28; at 60k channel 0
.01, channel 4 .03, channel 1 .07); s5 (.622 on these episodes, .636 in its log) STILL ignores channel 0 at 120k (drop
.002; no slot codes it, two slots both code channel 3, R^2 .94 / .97, as at 60k). So 5 of 6 RCC-feedback Classifier-route
nets recover every channel by 120k; one does not.
"""
import os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

SRC = [os.path.join(ROOT, "main", "DATA", "experimental", "clf_scratch", f"channel_census_{j}.txt") for j in (66671499, 66671962, 66673037, 66673038, 66673039, 66673040)] + \
      [os.path.join(ROOT, "main", "DATA", "experimental", "clf_scratch", f) for f in   # 120k continuations (11-13)
       ("channel_census_66683124.txt", "channel_census_local_clf120k_s2.txt", "channel_census_66685419.txt")]
ORDER = (("gen60k_lr4_s2", "Generator route (oracle, lr 1e-4)"),
         ("clf60k_lr4_s0", "Classifier route (oracle, lr 1e-4)"),
         ("clf30k_s2", "Classifier route (oracle, lr 1e-3)"),
         ("rccclf60k_lr4_s2", "Classifier route (RCC feedback)"),
         ("clf30k_s0", "Classifier route (oracle, lr 1e-3)"),
         ("clf30k_s1", "Classifier route (oracle, lr 1e-3)"),
         ("clf30k_s3", "Classifier route (oracle, lr 1e-3)"),
         ("rccclf60k_lr4_s3", "Classifier route (RCC feedback)"),
         ("rccclf60k_lr4_s0", "Classifier route (RCC feedback)"),
         ("rccclf60k_lr4_s1", "Classifier route (RCC feedback)"),
         ("rccclf60k_lr4_s4", "Classifier route (RCC feedback)"),
         ("rccclf60k_lr4_s5", "Classifier route (RCC feedback)"),
         ("clf60k_lr4_s2", "Classifier route (oracle, lr 1e-4)"))


def parse():
    out = {}
    for line in (l for f in SRC if os.path.exists(f) for l in open(f)):
        m = re.match(r"(\S+) novel acc ([\d.]+) \| code slot->true (\[.*?\]) R2 (\[.*?\]) R2 matrix rows=slots (\[\[.*?\]\]) "
                     r"\| bits-shuffled \(true ch 0-4\) (\[.*?\])", line)
        if m:
            M = np.array(eval(m.group(5)))                      # rows = model slots, columns = true channels
            out[m.group(1)] = dict(acc=float(m.group(2)), r2=M.max(0), drop=float(m.group(2)) - np.array(eval(m.group(6))))
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    D = parse()
    nets = [n for n, _ in ORDER if n in D]
    drop = np.array([D[n]["drop"] for n in nets])
    r2 = np.array([D[n]["r2"] for n in nets])
    labels = [f"{dict(ORDER)[n]}, seed {n[-1]}: {D[n]['acc']:.2f}" for n in nets]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, M, title, vmax, fmt in ((axes[0], drop, "a   Accuracy lost when one channel is shuffled", .4, "{:.2f}"),
                                   (axes[1], r2, "b   That channel's learned code vs the truth ($R^2$)", 1, "{:.2f}")):
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=fs["legend"] - 4,
                        color="k" if M[i, j] > .6 * vmax else "w")
        ax.set_xticks(range(5)); ax.set_xticklabels([f"channel {o}" for o in range(5)], fontsize=fs["legend"] - 4)
        ax.set_yticks(range(len(nets))); ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=fs["legend"] - 4)
        ax.axhline(3.5, color="w", lw=2)
        ax.set_title(title, loc="left", fontsize=fs["title"] - 4)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle("Which channels Fig 13's networks use (2,000 novel episodes per net; top: nets that learned, bottom: nets "
                 "that stalled; accuracy after the colon)", fontsize=fs["legend"] - 2)
    out = fig_path("fig_channel_census.png")
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
