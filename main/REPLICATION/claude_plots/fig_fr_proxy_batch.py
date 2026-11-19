"""Batch 8,000 vs 80,000: does a 10x larger batch make the Trained-vs-ESN divergence track FR in fewer updates?
(John 11-18: "i'm wondering if we could see the growth in R^2 within 10 episodes if we trained with a massive
batch?";: "run the 100 episode at batch 80K, though also save all of the first 10 episodes")

Data (fig_fr_proxy_training.py caches, all on the probes' 50 environments x 5,000 trajectories, T = 30):
    batch  8,000  fr_proxy_training_{dense,big,ext}.npz: the training-dynamics pair, episodes 5-2,000
    batch 80,000  fr_proxy_training_b80k.npz: the same pair retrained from the same seed and initial weights at batch
                  80,000 (16,000-trajectory slices, gradients accumulated), episodes 1-10 and every 5 to 100
R^2 = r^2 of the divergence D_KL(B^Trained || B^ESN) with FR (red) and with the observation count (black), per
environment, step and variable, averaged; band = +-1 SEM over the 50 environments. Updates = episodes minus test
episodes (every 5th, from 0). Left: per update; right: per training trajectory (updates x batch).

RESULT 11-18: per update the two batches give the same curves. R^2 with FR (8,000 / 80,000): .082 / .087 at
4 updates, .076 / .080 at 8, .060 / .060 at 20, .055 / .055 at 40, .040 / .058 at 80; with the count: .52 / .67 at 4,
.59 / .56 at 20, .57 / .58 at 40, and both drop at ~60-70 updates (.155 / .164 at 80). Per training trajectory the
80,000 run is ~10x behind. Each update moves every weight by <= ~1e-3 (Adam, lr 1e-3) at either batch (max |dW|
1.00e-3 to 1.03e-3 per update in the batch-80,000 run; test episodes 5 and 10 leave the weights unchanged; AMP loss
scale 65,536 throughout, no skipped steps). Both runs start from bit-identical weights (all 10 tensors, both networks).
So the early dynamics are set by the number of updates, not by gradient noise: a bigger batch does not bring FR's R^2
forward (within 10 episodes it stays at ~.08, count-driven).
Batch 8,000 with 100 updates per batch (the same batch and Bayes targets, the network's forward pass rerun before every
update; 10 batches): per update the count curve matches (R^2 .49 at 8 updates, drop at ~60-70, .158 / .083 at 400 /
800 vs .143 / .083); FR lags somewhat: .081 / .106 at 400 / 800 updates vs .099 / .159 with a fresh batch per update.
So fresh data helps the FR link a little, but 1% of the data gets most of the way per update. The lr 0.1, batch
100,000 run never learned (Trained RNN accuracy .100 at 100 episodes, logits to 350 per step, divergence median 614 nats)
and is not drawn.
ONE batch of 100,000 with 100 updates (slices of 20,000; John 11-18 "Lets do 100 updates with the 100K batch"): per update
the same curves again: R^2 with FR .059 / .056 / .038 at 16 / 32 / 64 updates (fresh batches of 8,000: .059 / .060 /
.038), with the count .59 / .62 / .36 (.51 / .64 / .35); at 100 updates FR .056, count .19. Within the first 100
updates, neither more data per update nor fresh data changes anything: the number of updates sets the pace.
ONE batch of 20,000 x 2,000 updates (--onebatch -> fig_fr_proxy_onebatch.png; slices of 10,000): the Trained RNN
memorises it (accuracy on that batch .63 / .73 / .75 at 400 / 800 / 1,600 updates, Joint .78) while on a second batch
of 20,000 it stays at .24 from ~200 updates (fresh-data run on that batch: .29 / .46 / .65); R^2 with FR on held-out
trajectories stays flat (.06 / .07 / .07 / .08 at 400 / 800 / 1,600 / 2,000 vs .10 / .16 / .30 fresh), the count's
falls as in every run (.07 at 2,000). So without fresh data the FR link never forms: it needs generalisation.

    python fig_fr_proxy_batch.py
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main.utils import fig_path, apply_fig_style
from fig_fr_proxy_training import BIG_CACHE, DENSE_CACHE, EXT_CACHE, B80K_CACHE, K100_CACHE, S100K_CACHE, updates

C_FR, C_COUNT = "#e41a1c", "k"


def r2_curves(paths, e_max=None):
    """E, {measure: (mean, SEM)} of R^2 over the given caches, sorted by episode, duplicates taken once"""
    Es, R = [], {"pearson": [], "count": []}
    seen = set()
    for p in paths:
        Z = np.load(p)
        for i, e in enumerate(Z["E"]):
            if e in seen or (e_max and e > e_max):
                continue
            seen.add(e)
            Es.append(e)
            for k in R:
                R[k].append((Z[k][i] ** 2).mean(-1).mean(-1))              # (env,)
    o = np.argsort(Es)
    E = np.array(Es)[o]
    return E, {k: (np.array(v)[o].mean(1), np.array(v)[o].std(1, ddof=1) / np.sqrt(len(v[0]))) for k, v in R.items()}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_fig_style()
    plt.rcParams.update({"axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})
    # (label, batch, curves, line style, x = updates?, updates per batch); the lr 0.1 run (batch100k_lr1e-1) is left out:
    # it never learned (accuracy .100 at 100 episodes; John 11-18: "we dont need to compare without learning")
    runs = [("8,000, 1 update per batch", 8000, r2_curves([DENSE_CACHE, BIG_CACHE, EXT_CACHE], 2000), "-", False, 1)]
    for lab, batch, cache, ls, is_u, k in (("80,000, 1 update per batch", 80000, B80K_CACHE, (0, (4, 1.5)), False, 1),
                                           ("8,000, 100 updates per batch", 8000, K100_CACHE, (0, (1, 1)), True, 100),
                                           ("100,000, 100 updates on 1 batch", 100000, S100K_CACHE, (0, (5, 1.5, 1, 1.5)),
                                            True, 100)):
        parts = [c for c in (cache, cache.replace(".npz", "_b.npz")) if os.path.exists(c)]
        if parts:
            runs.append((lab, batch, r2_curves(parts), ls, is_u, k))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)
    for j, ax in enumerate(axs):
        for lab, batch, (E, C), ls, is_u, k in runs:
            u = E if is_u else updates(E)
            x = u if j == 0 else np.ceil(u / k) * batch                   # distinct training trajectories
            for k, c in (("pearson", C_FR), ("count", C_COUNT)):
                mu, se = C[k]
                ax.fill_between(x, mu - se, mu + se, color=c, alpha=.15, lw=0)
                ax.plot(x, mu, color=c, ls=ls, lw=1.6)
        ax.set_xscale("log")
        ax.set_xlabel("optimizer updates" if j == 0 else "distinct training trajectories")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel(r"$R^2$ with $D_{KL}(B^{\mathrm{Trained}}\,\|\,B^{\mathrm{ESN}})$")
    axs[0].set_title("Per update", fontsize=12.5)
    axs[1].set_title("Per training trajectory", fontsize=12.5)
    from matplotlib.lines import Line2D
    l1 = axs[0].legend([Line2D([], [], color=C_FR, lw=1.6), Line2D([], [], color=C_COUNT, lw=1.6)], ["FR", "obs. count"],
                       frameon=False, fontsize=10, loc="upper right", title="Divergence vs.", title_fontsize=10)
    axs[0].add_artist(l1)                                                 # the run legend goes under it
    axs[0].legend([Line2D([], [], color=".3", ls=r[3], lw=1.6) for r in runs], [f"batch {r[0]}" for r in runs],
                  frameon=False, fontsize=10, loc="lower left", bbox_to_anchor=(0, .12))
    fig.suptitle("Batch size and updates per batch (lr $10^{-3}$), same seed and initial weights; R$^2$ on 50 environments "
                 "x 5,000 trajectories x 30 steps",
                 fontsize=12.5)
    out = fig_path("fig_fr_proxy_batch.png")
    fig.savefig(out, dpi=150)
    print(out)
    for lab, batch, (E, C), _, is_u, k in runs:
        u = E if is_u else updates(E)
        print(f"batch {lab}:")
        for target in (1, 2, 4, 8, 12, 20, 40, 80, 160, 400, 800, 1600):
            i = np.flatnonzero(u == target)
            if len(i):
                i = i[0]
                print(f"   {target:5d} updates (E {E[i]:4d}): R^2 FR {C['pearson'][0][i]:.3f}, count {C['count'][0][i]:.3f}")


def one_batch():
    """fig_fr_proxy_onebatch.png (John 11-18: one 20K batch, 2K updates; "i dont see the single batch curves in the first
    panel" -> one colour per run, one panel per measure): R^2 with FR, R^2 with the count, and the Trained RNN's goal
    accuracy on the fixed batch's training batch vs a second batch, per update; fresh batches of 8,000 (one update each)
    vs ONE fixed batch (20,000 x 2,000 updates; 100,000 x 100 updates)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fig_fr_proxy_training import S20K_CACHE, S20K_ACC
    apply_fig_style()
    plt.rcParams.update({"axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})
    from fig_fr_proxy_training import SHUF_CACHE, SHUF_ACC, PERSTEP_CACHE, PERSTEP_ACC
    C_FRESH, C_20K, C_100K, C_SHUF, C_PS = "k", "#d95f02", "#2c7fb8", "#7b3294", "#1b9e77"
    runs = [("fresh batch of 8,000 per update", r2_curves([DENSE_CACHE, BIG_CACHE, EXT_CACHE], 2500), C_FRESH, False)]
    for lab, cache, c in (("one batch of 20,000", S20K_CACHE, C_20K), ("one batch of 100,000", S100K_CACHE, C_100K),
                          ("one batch of 20,000, channels shuffled", SHUF_CACHE, C_SHUF),
                          ("one batch of 20,000, an update every step (no BPTT)", PERSTEP_CACHE, C_PS)):
        parts = [p_ for p_ in (cache, cache.replace(".npz", "_b.npz")) if os.path.exists(p_)]
        if parts:
            runs.append((lab, r2_curves(parts), c, True))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    top = max(float(np.nanmax(C[k][0] + C[k][1])) for _, (E, C), _, _ in runs for k in ("pearson", "count"))
    top = np.ceil(top * 1.04 / .05) * .05                                  # y-limit from the data
    for j, (k, title) in enumerate((("pearson", "Divergence vs FR"), ("count", "Divergence vs observation count"))):
        ax = axs[j]
        for lab, (E, C), c, is_u in runs:
            u = E if is_u else updates(E)
            mu, se = C[k]
            ax.fill_between(u, mu - se, mu + se, color=c, alpha=.15, lw=0)
            ax.plot(u, mu, color=c, lw=1.8, label=lab)
        ax.set_xscale("log")
        ax.set(xlabel="optimizer updates (lr $10^{-3}$)", ylim=(0, top))
        ax.set_title(title + "\n(R$^2$ on 50 environments x 5,000 held-out trajectories x 30 steps)", fontsize=11.5)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel(r"$R^2$ with $D_{KL}(B^{\mathrm{Trained}}\,\|\,B^{\mathrm{ESN}})$")
    axs[0].legend(frameon=False, fontsize=10.5, loc="upper left")
    bx = axs[2]
    if os.path.exists(S20K_ACC):                                           # accuracy: overfitting check
        A = np.load(S20K_ACC)
        u = A["u"]
        m = u > 0
        bx.plot(u[m], A["trained_train"][m], color=C_20K, lw=1.8, ls=(0, (4, 1.5)), label="one batch: its batch")
        bx.plot(u[m], A["trained_second"][m], color=C_20K, lw=1.8, label="one batch: second batch")
        bx.plot(updates(A["E_fresh"]), A["trained_fresh_second"], color=C_FRESH, lw=1.8,
                label="fresh data: second batch")
        if os.path.exists(SHUF_ACC):
            Sa = np.load(SHUF_ACC)
            ms = Sa["u"] > 0
            bx.plot(Sa["u"][ms], Sa["trained_train"][ms], color=C_SHUF, lw=1.8, ls=(0, (4, 1.5)),
                    label="shuffled: its batch")
            bx.plot(Sa["u"][ms], Sa["trained_second"][ms], color=C_SHUF, lw=1.8, label="shuffled: second batch")
        if os.path.exists(PERSTEP_ACC):
            Pa = np.load(PERSTEP_ACC)
            mp = Pa["u"] > 0
            bx.plot(Pa["u"][mp], Pa["trained_train"][mp], color=C_PS, lw=1.8, ls=(0, (4, 1.5)), label="no BPTT: its batch")
            bx.plot(Pa["u"][mp], Pa["trained_second"][mp], color=C_PS, lw=1.8, label="no BPTT: second batch")
        for k, ls, lab in (("joint_acc_T", "--", "Joint Bayes"), ("naive_acc_T", ":", "Naive Bayes")):
            bx.axhline(A[f"second_{k}"], color=".5", ls=ls, lw=1, label=f"{lab} ({A[f'second_{k}']:.2f})")
        bx.set_xscale("log")
        top_b = np.ceil((max(max(float(np.nanmax(l.get_ydata())) for l in bx.get_lines()), .0) + .06) / .05) * .05
        bx.set(xlabel="optimizer updates (lr $10^{-3}$)", ylabel="Trained RNN goal accuracy at T", ylim=(0, top_b))
        bx.set_title("Overfitting check: the training batch vs a second batch\n(20,000 trajectories each)", fontsize=11.5)
        from matplotlib.lines import Line2D                                # key: colour = run, line style = batch
        key = [Line2D([], [], color=c, lw=1.8, label=l) for c, l in
               ((C_20K, "one batch"), (C_SHUF, "one batch, shuffled"), (C_PS, "one batch, no BPTT"), (C_FRESH, "fresh data"))]
        key += [Line2D([], [], color=".3", lw=1.8, ls=(0, (4, 1.5)), label="its training batch"),
                Line2D([], [], color=".3", lw=1.8, label="second batch")]
        key += [h for h in bx.get_legend_handles_labels()[0] if "Bayes" in h.get_label()]
        bx.legend(handles=key[:4] + key[4:], ncol=2, frameon=True, framealpha=.92, edgecolor="none", fontsize=9.5,
                  loc="upper left", bbox_to_anchor=(0, (float(A["second_joint_acc_T"]) - .015) / top_b),
                  columnspacing=1.2, handlelength=2.2)                        # in the gap between the Bayes lines
        bx.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fresh data vs one fixed batch, same seed and initial weights", fontsize=13)
    out = fig_path("fig_fr_proxy_onebatch.png")
    fig.savefig(out, dpi=150)
    print(out)
    for lab, (E, C), c, is_u in runs:
        u = E if is_u else updates(E)
        print(lab + ":", " ".join(f"{t}: {C['pearson'][0][np.flatnonzero(u == t)[0]]:.3f}/{C['count'][0][np.flatnonzero(u == t)[0]]:.3f}"
                                  for t in (16, 64, 100, 200, 400, 800, 1200, 1600, 2000) if (u == t).any()))


if __name__ == "__main__":
    one_batch() if "--onebatch" in sys.argv else main()
