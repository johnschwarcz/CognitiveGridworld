"""Transfer of learned embeddings to a fresh reader (trainings/transfer_reader.py; John 11-11). Columns = readers
(Bayes-structured: one evidence function shared by the channels; the same unshared; an LSTM like the RCC's Classifier),
y = goal accuracy at T = 30 on novel environments (2,048 episodes per test), x = training episodes of the fresh reader
(batch 512). Lines: embeddings from the Generator route / the Classifier route of Fig 13's 60k networks (seed 0, oracle
feedback) or random, FROZEN (solid) or UPDATED by the reader (dashed); thin = seeds, thick = mean.

RESULT 11-11 (all 38 runs, 4,000 steps = 2.05M episodes; novel accuracy at the end, seeds 0 / 1):
    LSTM          Generator route frozen .49 / .50, Classifier route frozen .50 / .49: EQUAL (the symmetry null); both
                  leave chance only at ~2e5 episodes; updated (lr 1e-3) .14-.16 for both = random .13-.14
    shared        Generator route .53 / .52; Classifier route .15 / .14; slots put back .30 / .30; slots + frames .53 / .53
    unshared      Generator route .49 / .50; Classifier route .14 / .14; slots put back .49 / .48 (= Generator route)
    from scratch  (random embeddings, updated) .12-.14 with every reader
  Channel-wise readers leave chance ~4x sooner than the LSTM (~5e4 vs ~2e5 episodes; .30 vs ~.18 at 2e5) but all three
  end near .5 at 2M episodes. Updating the embeddings with a fresh reader at lr 1e-3 damages them (Generator route
  .41-.45 updated vs .49-.53 frozen; LSTM never leaves chance). The first ~300 steps of lstm_gen_frozen_s0/1 were lost
  (a cancelled mis-indexed duplicate truncated the files): the straight segment in panel c.

NEW CHANNEL (TASKS 38-61; trained on channels 0-3, frozen; end accuracy 4 channels -> channel 4 added, seeds 0 / 1):
    shared    Generator route .467 -> .536 (+.069) / .491 -> .557 (+.066); Classifier route slots + frames put back +.057 /
              +.068; as learned +.024 / -.002 (at chance); slots only +.006 / -.023
    unshared  +.005 to +.024 for every source (channel 4's function is untrained)
    lstm      -.056 to +.001 for every source (its channel-4 weights are untrained)
  Only a reader with one evidence function for every channel, on channel-consistent codes, uses a channel it never saw.
  (With channel 4 masked in training the LSTM on the Classifier route as learned ends lower, .42 vs .46: masking a slot
  removes another channel's code, since codes and data of a channel sit in different slots.)
PLASTICITY (TASKS 62-77; end accuracy, mean of 2 seeds):
    lstm      frozen .497 / .494 (Generator / Classifier route); updated lr 1e-3 .146 / .152; lr 1e-4 .412 / .414;
              lr 1e-3 + the network's own Generator kept training on the embeddings .376 / .145
    shared    Generator route frozen .525, lr 1e-3 .437, lr 1e-4 .519, + Generator .456; Classifier route .13-.15 in every mode
  A gentle embedding lr protects both routes equally; at lr 1e-3 only the Generator route's own Generator partly protects
  its embeddings (its Classifier-route counterpart is the stuck one). The Classifier code is not re-bound by plastic
  training within 2M episodes (channel-wise readers stay at chance in every mode).

    python fig13_transfer.py [--pull]
"""
import os, sys, json, glob, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "fig13_embeddings", "transfer_reader")
READERS = (("shared", "a   Bayes-structured reader:\none evidence function for every channel"),
           ("unshared", "b   the same, a separate function\nper channel"),
           ("lstm", "c   LSTM reader\n(like the RCC's Classifier)"))
LINES = (("gen", "frozen", "#08519c", "-", "Generator-route embeddings, frozen"),
         ("gen", "updated", "#08519c", (0, (4, 2)), "Generator-route embeddings, updated"),
         ("clf", "frozen", "#a50f15", "-", "Classifier-route embeddings, frozen"),
         ("clf", "updated", "#a50f15", (0, (4, 2)), "Classifier-route embeddings, updated"),
         ("random", "updated", "0.45", (0, (1, 1.5)), "random embeddings, updated"),
         ("clf_slots", "frozen", "#7b3294", "-", "Classifier route, slots put back, frozen"),
         ("clf_aligned", "frozen", "#1b9e77", "-", "Classifier route, slots + frames put back, frozen"))
JOINT, NAIVE = .767, .451                                              # the paper ESN dataset's test bounds (|S| 500)


def pull():
    os.makedirs(LOGS, exist_ok=True)
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/transfer_reader/*.jsonl' {LOGS}/", shell=True)


def load(tag):
    f = os.path.join(LOGS, f"{tag}.jsonl")
    if not os.path.exists(f):
        return []
    out = []                                       # lstm_gen_frozen_s0/1: a cancelled duplicate truncated their start
    for l in open(f, "rb").read().replace(b"\x00", b"").decode().splitlines():   # (NUL hole); keep whole records
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return sorted({q["step"]: q for q in out}.values(), key=lambda q: q["step"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True, constrained_layout=True)
    for a, (reader, title) in zip(ax, READERS):
        for src, mode, col, ls, lab in LINES:
            runs = [load(f"{reader}_{src}_{mode}_s{k}") for k in (0, 1)]
            runs = [r for r in runs if r]
            for r in runs:
                a.plot([max(q["episodes"], 256) for q in r], [q["novel"] for q in r], color=col, ls=ls, lw=.7, alpha=.4)
            n = min((len(r) for r in runs), default=0)
            if n:
                m = np.mean([[q["novel"] for q in r[:n]] for r in runs], 0)
                a.plot([max(q["episodes"], 256) for q in runs[0][:n]], m, color=col, ls=ls, lw=2.0,
                       label=f"{lab} ({len(runs)})")
                last = ", ".join(f"{r[-1]['novel']:.2f}" for r in runs)
                print(f"{reader:9s} {src:7s} {mode:8s} step {runs[0][min(n, len(runs[0])) - 1]['step']:5d}: novel {last}")
        a.axhline(JOINT, color="k", lw=1, ls="--"); a.axhline(NAIVE, color="k", lw=1, ls=":"); a.axhline(.1, color="0.7", lw=.8, ls=":")
        a.set_xscale("log")
        a.set_title(title, loc="left", fontsize=fs["title"] - 4)
        a.set_xlabel("episodes seen by the fresh reader (batch 512)")
        a.spines[["top", "right"]].set_visible(False)
    for y, lab in ((JOINT, "Joint Bayes"), (NAIVE, "Naive Bayes")):
        ax[-1].text(1.01, y, lab, transform=ax[-1].get_yaxis_transform(), va="center", fontsize=fs["legend"] - 2)
    ax[0].set_ylabel("goal accuracy, novel environments")
    ax[0].legend(frameon=False, fontsize=fs["legend"] - 3, loc="upper left")
    fig.suptitle("A fresh reader trained on embeddings learned by Fig 13's networks (oracle feedback, 60,000 episodes, "
                 "seed 0): frozen (solid) or updated by the reader (dashed)\nCPU, batch 512, Adam lr $10^{-3}$, target = "
                 "the true joint value; 2,048 novel episodes per test; thin = seeds, thick = mean", fontsize=fs["legend"] - 2)
    out = fig_path("fig13_transfer.png")
    fig.savefig(out, dpi=200)
    print(out)


SRC4 = (("gen", "#08519c", "Generator route"), ("clf", "#a50f15", "Classifier route, as learned"),
        ("clf_slots", "#7b3294", "Classifier route, slots put back"),
        ("clf_aligned", "#1b9e77", "Classifier route, slots + frames put back"))


def mean_curve(tags, key):
    runs = [r for r in (load(t) for t in tags) if r and key in r[-1]]
    n = min((len(r) for r in runs), default=0)
    if not n:
        return None, None, runs
    x = np.array([max(q["episodes"], 256) for q in runs[0][:n]])
    return x, np.mean([[q[key] for q in r[:n]] for r in runs], 0), runs


def figure_ch4():
    """fig13_transfer_ch4.png (John 11-11: extrapolation to new channels): readers trained on channels 0-3 with frozen
    embeddings; row 1 = accuracy with those 4 channels, row 2 = change when channel 4 is added at test (zero-shot)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True, sharey="row", constrained_layout=True)
    for j, (reader, title) in enumerate(READERS):
        for src, col, lab in SRC4:
            tags = [f"{reader}_{src}_frozen_ch4_s{k}" for k in (0, 1)]
            x, m4, runs = mean_curve(tags, "novel")
            if x is None:
                continue
            _, m5, _ = mean_curve(tags, "novel_all5")
            ax[0, j].plot(x, m4, color=col, lw=2, label=lab)
            ax[1, j].plot(x, m5 - m4, color=col, lw=2)
            for r in runs:
                ax[1, j].plot([max(q["episodes"], 256) for q in r], [q["novel_all5"] - q["novel"] for q in r], color=col,
                              lw=.6, alpha=.4)
            print(f"ch4 {reader:9s} {src:12s} end: 4 channels {m4[-1]:.3f}, with channel 4 added {m5[-1]:.3f}")
        ax[0, j].set_title(title, loc="left", fontsize=fs["title"] - 4)
        ax[1, j].axhline(0, color="k", lw=.8)
        ax[1, j].set_xlabel("episodes seen by the fresh reader (batch 512)")
        for i in (0, 1):
            ax[i, j].set_xscale("log")
            ax[i, j].spines[["top", "right"]].set_visible(False)
    ax[0, 0].set_ylabel("accuracy with the 4 trained channels\n(novel environments)")
    ax[1, 0].set_ylabel("change when channel 4 is added\n(zero-shot; accuracy points)")
    ax[0, 0].legend(frameon=False, fontsize=fs["legend"] - 3, loc="upper left")
    fig.suptitle("Extrapolation to a new observation channel: readers trained on channels 0-3 only (frozen embeddings of "
                 "Fig 13's 60k networks, seed 0), tested with channel 4 added\nthin = seeds, thick = mean; 2,048 novel "
                 "episodes per test", fontsize=fs["legend"] - 2)
    out = fig_path("fig13_transfer_ch4.png")
    fig.savefig(out, dpi=200)
    print(out)


def figure_plastic():
    """fig13_plasticity.png (John 11-11: can Generator-learned embeddings stay plastic in a new setting? 17:20: "organize that
    figure better, maybe use more panels"): rows = readers (LSTM, shared evidence reader), columns = what happens to the
    embeddings while the fresh reader learns (frozen; updated at lr 1e-4; at lr 1e-3; at lr 1e-3 with the network's own
    Generator kept training on them); in each panel the Generator route (blue) and the Classifier route (red), thin =
    seeds, thick = mean"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    modes = (("frozen", "frozen"), ("updated_slow", "updated, lr $10^{-4}$"), ("updated", "updated, lr $10^{-3}$"),
             ("updated_gen", "updated, lr $10^{-3}$,\n+ the network's own Generator's loss"))
    rows = (("lstm", "LSTM reader"), ("shared", "Bayes-structured\nreader (shared)"))
    fig, ax = plt.subplots(2, 4, figsize=(14.5, 6.6), sharex=True, sharey=True, constrained_layout=True)
    for i, (reader, rlab) in enumerate(rows):
        for j, (mode, mlab) in enumerate(modes):
            a_ = ax[i, j]
            for src, col, lab in SRC4[:2]:
                tags = [f"{reader}_{src}_{mode}_s{k}" for k in (0, 1)]
                x, m, runs = mean_curve(tags, "novel")
                if x is None:
                    continue
                for r in runs:
                    a_.plot([max(q["episodes"], 256) for q in r], [q["novel"] for q in r], color=col, lw=.6, alpha=.4)
                a_.plot(x, m, color=col, lw=2, label=lab.split(",")[0])
                print(f"plastic {reader:7s} {src:4s} {mode:13s} end {m[-1]:.3f}")
            a_.axhline(JOINT, color="k", lw=.8, ls="--"); a_.axhline(NAIVE, color="k", lw=.8, ls=":")
            a_.set_xscale("log")
            a_.spines[["top", "right"]].set_visible(False)
            if i == 0:
                a_.set_title(mlab, fontsize=fs["legend"] - 1)
            if j == 0:
                a_.set_ylabel(f"{rlab}\naccuracy, novel environments", fontsize=fs["legend"] - 1)
            if i == 1:
                a_.set_xlabel("episodes seen by the fresh reader", fontsize=fs["legend"] - 2)
    ax[0, 0].legend(frameon=False, fontsize=fs["legend"] - 2, loc="upper left")
    fig.suptitle("Embeddings of Fig 13's 60k networks (seed 0) under a fresh reader: frozen or plastic (dashed = Joint "
                 "Bayes, dotted = Naive Bayes); 2,048 novel episodes per test, batch 512", fontsize=fs["legend"] - 1)
    out = fig_path("fig13_plasticity.png")
    fig.savefig(out, dpi=200)
    print(out)


def figure_correction():
    """fig13_correction.png (John 11-11: "what about learning this correction?"): the shared evidence reader with a
    learnable 10 x 10 linear map of the code numbers in front (identity at the start; it can represent the exact slot +
    frame correction). a: all channels; b: trained on channels 0-3, the change when channel 4 is added at test. References
    without the map: the Generator route, and the Classifier route with the correction put in by hand"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    lines = (("shared_gen", "#08519c", "-", "Generator route, no map"),
             ("shared_adapt_gen", "#08519c", (0, (4, 2)), "Generator route + learnable map"),
             ("shared_adapt_clf", "#a50f15", (0, (4, 2)), "Classifier route + learnable map"),
             ("shared_clf_aligned", "#1b9e77", "-", "Classifier route, corrected by hand, no map"),
             ("shared_clf", "#a50f15", "-", "Classifier route, no map"))
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for pre, col, ls, lab in lines:
        x, m, runs = mean_curve([f"{pre}_frozen_s{k}" for k in (0, 1)], "novel")
        if x is not None:
            for r in runs:
                ax[0].plot([max(q["episodes"], 256) for q in r], [q["novel"] for q in r], color=col, ls=ls, lw=.6, alpha=.4)
            ax[0].plot(x, m, color=col, ls=ls, lw=2, label=lab)
            print(f"correction all channels {pre:20s} end {m[-1]:.3f}")
        tags = [f"{pre}_frozen_ch4_s{k}" for k in (0, 1)]
        x, m4, runs = mean_curve(tags, "novel")
        if x is not None:
            _, m5, _ = mean_curve(tags, "novel_all5")
            ax[1].plot(x, m5 - m4, color=col, ls=ls, lw=2, label=lab)
            print(f"correction channel 4    {pre:20s} end 4 ch {m4[-1]:.3f} -> 5 ch {m5[-1]:.3f} ({m5[-1] - m4[-1]:+.3f})")
    ax[0].axhline(JOINT, color="k", lw=.8, ls="--"); ax[0].axhline(NAIVE, color="k", lw=.8, ls=":")
    ax[1].axhline(0, color="k", lw=.8)
    ax[0].set_ylabel("accuracy, novel environments")
    ax[1].set_ylabel("change when channel 4 is added (zero-shot)")
    for a_, t in zip(ax, ("a   All 5 channels", "b   Trained on channels 0-3, channel 4 added at test")):
        a_.set_xscale("log"); a_.set_title(t, loc="left", fontsize=fs["title"] - 3)
        a_.set_xlabel("episodes seen by the fresh reader (batch 512)")
        a_.spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False, fontsize=fs["legend"] - 3, loc="upper left")
    fig.suptitle("Learning the correction: a shared evidence reader with a learnable linear map of the code (identity at "
                 "the start), frozen embeddings of Fig 13's 60k networks (seed 0)\nthin = seeds, thick = mean; 2,048 "
                 "novel episodes per test", fontsize=fs["legend"] - 2)
    out = fig_path("fig13_correction.png")
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
    figure_ch4()
    figure_plastic()
    figure_correction()
