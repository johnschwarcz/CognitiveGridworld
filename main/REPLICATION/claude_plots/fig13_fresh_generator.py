"""Can a fresh Generator use the Classifier route's embeddings? (John 11-11) Plots trainings/fresh_generator.py: a freshly
initialized Generator trained alone (lr 1e-3, batch 2,000, true values) on FROZEN embeddings of a Fig 13 60k network:
the network's own Generator-route embeddings (positive control), the Classifier route's as learned, the same with the
model's channels relabeled to the observation channel each code carries (fig13_embeddings.py), and random embeddings
(negative control). Rows: accuracy of exact Bayes with the learned likelihood (goal, T = 30) and kl_env (summed over
channels, at the episode's state), novel environments, 2,000 episodes per test; columns: oracle / original feedback.

RESULT 11-11 (cluster 66665893, 3,000 steps each, novel environments):
    own Generator-route embeddings   .726 (oracle) / .693 (RCC), kl .015 / .026: reached by 250 steps
    Classifier route, relabeled      .210 / .291 (oracle s0 / s1), .260 (original s1), kl .55-.64: reached by 250 steps
                                     and flat to 3,000 (12x what the own embeddings needed)
    Classifier route, as learned     .140 / .140 / .153, kl 1.05-1.10 = random embeddings .132, kl 1.10
11-11 15:00, tasks 9-12 (Z-level frames): Classifier route with slots AND per-channel frames restored .70 / .72 by 250
steps (oracle s0 / s1; = own), .58-.61 (original s1, a less finished network); Generator-route code with its channels'
frames scrambled (the 4 frames the Classifier route uses) .21-.23 = the Classifier route relabeled only. The per-channel
frame (sign / swap) is the whole difference.
(earlier reading) So the Classifier route's embeddings are unusable for generation as learned (= random) and only partly usable after the
channels are put back: the per-channel code carries the likelihood (k-NN R^2 .98, fig13_embeddings.py) but in a
nonlinear re-encoding the Generator's architecture (Z enters linearly, added to the answer stream before a ReLU) does not
learn to read in 3,000 steps.

    python fig13_fresh_generator.py [--pull]
"""
import os, sys, json, glob, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "fig13_embeddings", "fresh_generator")
STYLE = {"own": ("#08519c", "-", "Generator-route embeddings (own)"),
         "aligned": ("#1b9e77", "-", "Classifier route, slots + per-channel frames restored"),
         "scrambled": ("#08519c", (0, (4, 2)), "Generator route, per-channel frames scrambled"),
         "relabeled": ("#7b3294", "-", "Classifier route, channels relabeled"),
         "as_learned": ("#a50f15", "-", "Classifier route, as learned"),
         "random": ("0.5", ":", "random embeddings")}
PANELS = (("a   Oracle feedback", ("gen_s0_own", "clf_s0_aligned", "clf_s1_aligned", "gen_s0_scrambled",
                                    "clf_s0_relabeled", "clf_s1_relabeled", "clf_s0_as_learned", "clf_s1_as_learned",
                                    "random")),
          ("b   Original feedback", ("rcc_s0_own", "rccclf_s1_aligned", "rccclf_s1_relabeled", "rccclf_s1_as_learned")))


def pull():
    os.makedirs(LOGS, exist_ok=True)
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/fresh_generator/*.jsonl' {LOGS}/", shell=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    fig, ax = plt.subplots(2, 2, figsize=(9.8, 6.4), sharex=True, sharey="row", constrained_layout=True)
    for j, (title, tags) in enumerate(PANELS):
        seen = set()
        for tag in tags:
            f = os.path.join(LOGS, f"{tag}.jsonl")
            if not os.path.exists(f):
                continue
            r = [json.loads(l) for l in open(f) if l.strip()]
            kind = next(k for k in STYLE if tag.endswith(k))
            col, ls, lab = STYLE[kind]
            x = np.maximum([q["step"] for q in r], 10)
            for i, key in enumerate(("novel_plugin", "novel_kl_env")):
                ax[i, j].plot(x, [q[key] for q in r], color=col, ls=ls, lw=1.8, marker="o", ms=3,
                              label=None if kind in seen or i else lab)
            seen.add(kind)
            print(f"{tag:22s} step {r[-1]['step']:5d}: novel {r[-1]['novel_plugin']:.3f} (train {r[-1]['train_plugin']:.3f}),"
                  f" kl {r[-1]['novel_kl_env']:.3f}")
        ax[0, j].set_title(title, loc="left", fontsize=fs["title"] - 3)
        ax[0, j].legend(frameon=True, facecolor="w", edgecolor="none", framealpha=.9, fontsize=fs["legend"] - 3,
                        loc="center right", bbox_to_anchor=(1, .45))
        for i in (0, 1):
            ax[i, j].spines[["top", "right"]].set_visible(False)
            ax[i, j].set_xscale("log")
        ax[1, j].set_yscale("log")
        ax[1, j].set_xlabel("training steps of the fresh Generator (batch 2,000; step 0 drawn at 10)")
    ax[0, 0].set_ylabel("accuracy, Bayes with the\nlearned likelihood (novel)")
    ax[1, 0].set_ylabel(r"$\Sigma_o\,\mathrm{KL}(P(o\,|\,r)\,\|\,\hat P(o\,|\,r))$" + "\n(nats, novel, log scale)")
    ax[0, 0].axhline(.1, color="0.6", lw=.8, ls=":")
    fig.suptitle("A fresh Generator trained on FROZEN embeddings of Fig 13's 60k networks (only the Generator learns; "
                 "true values; lr $10^{-3}$)\nrelabeled = slots put back; restored = slots and each channel's frame (sign / swap) put "
                 "back; scrambled = each channel given a different frame; 2,000 novel episodes per test", fontsize=fs["legend"] - 2)
    out = fig_path("fig13_fresh_generator.png")
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
