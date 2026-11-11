"""The one strong lever on learning, and where the learning signal sits relative to FR (John 11-09: "if you found a strong
lever can you show me the effect of it, and how the learning signal cross-correlates with FR through training?").
-> main/REPLICATION/figure_bin/fig_learn_lever.png
Top row, the lever (train_learn_masks from-scratch runs, seed 20261029, batch 8,000, half of each step's episodes
selected per training batch and their read-out increments detached before the cumulative sum, so no loss term's
gradient reaches the network through the updates made at those steps; replicate 2: a new world, initialization and
episode stream, thin lines):
  a  goal accuracy at T on the runs' test episodes (8,000 per test, one test every 5 raw episodes; mean per 100 raw
     episodes); grey: naive's accuracy on the same tests; linear x (the effect grows late)
  b  on the 10,000 probes where naive is wrong and Bayes right (goal variable at T): the network's accuracy
     (td_tt_block_pilot.behaviour, 10 checkpoints; replicate 1 only, its world is the probes')
  c, d  the goal belief's error at T, KL(O_T || q_T), in the probes' top (c) and bottom (d) half of goal FR_T
Bottom row, the learning signal and FR (td_xcorr_across: correlations ACROSS episodes at each step, then the mean over
steps 17-30; FR changes slowly within an episode, 67% of its variance is between episodes):
  e  FR is accumulated entanglement: Pearson r over the probes between FR_t and the entanglement summed over the last
     w steps (both variables averaged), per step t = 17-30, then the mean over t
  f, g  credit by lag: r between the share of an error's credit (step t) sent k steps back and FR at the landing step
     t - k (td_tt_credit_reach, 2,000 probes, errors at steps 17-30, 32 checkpoints), Trained RNN and ESN
RESULTS (11-09).
  The lever: blocking the highest-FR half slows learning from the start; accuracy at 2,000 raw episodes (1,600 training
  episodes) .554 vs .653 for a random half (replicate .580 vs .644), lowest-FR half .603 (.612), most / least
  entangled half .622 / .628, control .665. On the probes at 2,400 training episodes (b-d; control / random half /
  highest-FR / lowest-FR / most / least entangled; 3,644 of the probes have naive wrong and Bayes right): accuracy where
  naive is wrong .78 / .76 / .52 / .75 / .70 / .74;
  naive's answer given there .10 / .11 / .26 / .11 / .13 / .12; error at high-FR probes .40 / .39 / 2.98 / .38 / .81 /
  .45 nats, at low-FR probes .16 / .15 / .20 / .27 / .21 / .23. The block keeps exactly the blocked states wrong: the
  highest-FR block leaves the high-FR error 7.5x the control's with the low-FR error near normal; the lowest-FR block
  does the mirror image while naive's rule is being learned (low-FR error 3.0 vs 1.6 at 720 training episodes) and
  then catches up; the most entangled half, which overlaps high FR (r .35), doubles the high-FR error.
  (e-g re-measured 11-10 with FR = KL(Bayes || naive), the paper's forward form: probes 'fr' and td_xcorr_across
  were rebuilt; the 11-09 version used the symmetric clipped KL. The top row was forward already: the runs' masks,
  train_learn_masks.goal_fr, and td_tt_block_pilot's halves.)
  FR = accumulated entanglement: r(FR_t, entanglement at t) .35, summed over the episode so far .65 (the same under
  both definitions); with the accumulated entanglement held fixed (partial r) FR adds little to the credit pattern:
  |r| <= .14 in the Trained RNN (at 1,360 training episodes, lag 14) and <= .08 in the ESN, against raw |r| up to .25
  (with the symmetric KL <= .09 / .07), and the across-episode entanglement maps look like FR's.
  The other update measures follow FR less across episodes (td_xcorr_across, lag 0, Trained RNN |r| <= .22: coherence
  +.19 at 400 -> -.06 at the end; dW energy +.05 -> +.09; dW eff. dim. +.04 -> +.05; cross-band share -.02 -> -.16),
  so credit reach is the FR-linked part of the learning signal. Credit by lag (f, g): in the Trained RNN, from ~100 to
  ~3k training episodes, credit sent 8-15 steps back lands on high-FR steps (+.06 to +.24, mean +.18, at 320-1,600)
  and credit sent 1-2 steps back avoids them (-.24 at 320); after ~3k the positive band moves to lags 2-4 (+.20 at lag
  2 at the end) and lag 0 turns negative (-.22). The ESN has the opposite pattern from ~300 on: lag 0 +.24, lags 2-15
  -.17 to -.25 at the end (its credit stays on the step in high-FR episodes).
Run from the repository root (CPU, ~1 min):
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig_learn_lever.py"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.colors import Normalize                                           # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
BP = os.path.join(TD, "through_training", "block_pilot")
RUNS = (("ctrl", "control", "k", "-"), ("det_rand_f50", "random half", "0.55", "-"),
        ("det_hi_fr_f50", "highest-FR half", "#2a9d6e", "-"), ("det_lo_fr_f50", "lowest-FR half", "#2a9d6e", (0, (3, 1.6))),
        ("det_hi_f50", "most entangled half", "#7b4fa6", "-"), ("det_lo_f50", "least entangled half", "#7b4fa6",
                                                                  (0, (3, 1.6))))
REP2 = ("det_rand_f50", "det_hi_fr_f50", "det_lo_fr_f50")
CROSS = 880.


def train_x(raw):
    raw = np.asarray(raw)
    return np.maximum(raw - raw // 5, 10.).astype(float)


def tests(tag, rep=1):
    """mean goal accuracy at T per 100 raw episodes: x (training episodes), acc, naive"""
    d = os.path.join(TD, f"learnmask_{tag}_s0" + ("" if rep == 1 else f"_r{rep}") + "_s20261029")
    z = np.load(os.path.join(d, "tests.npz"))
    b = z["raw"] // 100
    keys = np.unique(b)
    acc = np.array([z["acc"][b == k, -1].mean() for k in keys])
    nai = np.array([z["naive"][b == k, -1].mean() for k in keys])
    return train_x(keys * 100 + 50), acc, nai


def fr_integration():
    """w (1..30) and r(FR_t, sum of entanglement over the last w steps), mean over t = 17-30 (steps with w <= t)"""
    P = np.load(os.path.join(AN, "probes.npz"))
    fr, ent = P["fr"].astype(np.float64).mean(-1), P["ent"].astype(np.float64).mean(-1)
    cs = np.concatenate([np.zeros((len(ent), 1)), ent.cumsum(1)], 1)                     # cs[:, t+1] = sum_{s<=t}

    def rc(a, b):
        a, b = a - a.mean(0), b - b.mean(0)
        return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))
    ws = np.arange(1, 31)
    out = []
    for w in ws:
        t = np.arange(16, 30)
        t = t[t + 1 - w >= 0]
        out.append(rc(fr[:, t], cs[:, t + 1] - cs[:, t + 1 - w]).mean())
    return ws, np.array(out)


def figure():
    import fig1 as f1
    apply_fig_style()
    plt.rcParams.update(f1.RC)
    FT_, FL_, FLE = f1.FS_TICK, f1.FS_LAB, f1.FS_LETTER
    W, H = f1.W_IN, 4.15
    fig = plt.figure(figsize=(W, H))

    def add(x, top, w, h):
        return fig.add_axes([x / W, (H - top - h) / H, w / W, h / H])

    def title2(ax, name, formula):
        ax.annotate(formula, xy=(.5, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                    fontsize=FT_, ha="center", va="baseline")
        ax.annotate(name, xy=(.5, 1), xycoords="axes fraction", xytext=(0, 13.5), textcoords="offset points",
                    fontsize=FL_ + .5, ha="center", va="baseline")

    def letter(ax, s_):
        ax.annotate(r"$\mathbf{%s}$" % s_, xy=(0, 1), xycoords="axes fraction", xytext=(-26, 13.5),
                    textcoords="offset points", fontsize=FLE, ha="left", va="baseline")

    def logx(ax, label=True):                                          # linear: the effect grows late
        ax.set_xticks([0, 1000, 2000])
        ax.set_xticklabels(["0", "1k", "2k"])
        ax.set_xlim(0, 2450)
        ax.axvline(CROSS, color="0.6", lw=.7, ls=(0, (1, 1.5)), zorder=0)
        if label:
            ax.set_xlabel("training episodes", fontsize=FL_, labelpad=1)
    PH_ = 1.15
    tops = (.48, .48 + PH_ + .95)
    A0 = [add(.55 + j * (1.33 + .48), tops[0], 1.33, PH_) for j in range(4)]
    A1 = [add(.55 + j * (1.64 + .74), tops[1], 1.64, PH_) for j in range(3)]
    # a: accuracy
    a = A0[0]
    naive = []
    for tag, nm, col, ls in RUNS:
        x, acc, nai = tests(tag)
        a.plot(x, acc, color=col, ls=ls, lw=1.1)
        naive.append(nai)
        if tag in REP2:
            x2, acc2, _ = tests(tag, 2)
            a.plot(x2, acc2, color=col, ls=ls, lw=.55, alpha=.8)
    a.axhline(np.mean([v.mean() for v in naive]), color="0.6", lw=.6, ls=(0, (4, 2)))
    a.text(60, np.mean([v.mean() for v in naive]) + .01, "naive", fontsize=FT_ - .5, color="0.45", va="bottom")
    a.set_ylim(.1, .72)
    a.set_ylabel("goal accuracy at $T$", fontsize=FL_, labelpad=1)
    title2(a, "Blocking half of each step", "goal accuracy, test episodes")
    logx(a)
    # b: beyond naive
    a = A0[1]
    B = {}
    for tag, nm, col, ls in RUNS:
        p = os.path.join(BP, f"behav_learnmask_{tag}_s0.npz")
        if os.path.exists(p):
            B[tag] = z = np.load(p)
            a.plot(z["x"], z["acc_only"], color=col, ls=ls, lw=1.1, marker="o", ms=1.8)
    a.set_ylim(0, .85)
    a.set_ylabel("accuracy", fontsize=FL_, labelpad=1)
    title2(a, "Beyond naive", "naive wrong, optimal right")
    logx(a)
    # c, d: the error at high- and at low-FR probes
    for a, key, nm, form in ((A0[2], "kl_hi", "Error at high FR", r"top half of FR$_T$"),
                             (A0[3], "kl_lo", "Error at low FR", r"bottom half of FR$_T$")):
        for tag, _, col, ls in RUNS:
            if tag in B:
                a.plot(B[tag]["x"], B[tag][key], color=col, ls=ls, lw=1.1, marker="o", ms=1.8)
        a.set_yscale("log")
        a.set_ylim(.1, 20)
        a.set_ylabel(r"KL$(O_T\Vert q_T)$ (nats)", fontsize=FL_, labelpad=1)
        title2(a, nm, form)
        logx(a)
    for a, s_ in zip(A0, "abcd"):
        letter(a, s_)
    # run key under the top row
    fig.legend(handles=[Line2D([], [], color=c, ls=ls, lw=1.1, label=nm) for _, nm, c, ls in RUNS]
               + [Line2D([], [], color="0.3", lw=.55, label="replicate (new world)")],
               loc="upper center", bbox_to_anchor=(.5, 1 - (tops[0] + PH_ + .36) / H), ncol=7, frameon=False,
               fontsize=FT_ - .3, handlelength=1.8, columnspacing=1.1)
    A = [A0, A1]
    # d: FR integrates entanglement
    a = A1[0]
    ws, r = fr_integration()
    a.plot(ws, r, color="k", lw=1.1, marker="o", ms=1.6)
    a.set_xlim(0, 31)
    a.set_ylim(0, .75)
    a.set_xlabel("entanglement summed over the last $w$ steps", fontsize=FL_, labelpad=1)
    a.set_ylabel("Pearson $r$ with FR$_t$", fontsize=FL_, labelpad=1)
    title2(a, "FR is accumulated entanglement", r"$r(\mathrm{FR}_t,\ \Sigma_{s>t-w}E_s)$, across episodes")
    letter(a, "e")
    # f, g: credit by lag x FR
    nrm = Normalize(-.25, .25)
    for a, run, nm, s_ in ((A1[1], "trained", "Trained RNN", "f"), (A1[2], "echo", "ESN", "g")):
        d = dict(np.load(os.path.join(AN, f"xcorr_across_{run}.npz")))
        x, C = d["credit_x_fr"], d["credit_fr"]
        lx = np.log(x)
        mid = (lx[1:] + lx[:-1]) / 2
        xe = np.exp(np.concatenate([[lx[0] - (mid[0] - lx[0])], mid, [lx[-1] + (lx[-1] - mid[-1])]]))
        a.pcolormesh(xe, np.arange(C.shape[1] + 1) - .5, C.T, cmap="viridis", norm=nrm, shading="flat",
                     rasterized=True)
        a.set_xscale("log")
        a.set_xticks([10, 100, 1000, 10000])
        a.set_xticklabels(["0", "100", "1k", "10k"])
        a.set_xlim(xe[0], xe[-1])
        a.minorticks_off()
        a.axvline(CROSS, color="w", lw=.7, ls=(0, (1, 1.5)))
        a.set_yticks([0, 5, 10, 15])
        a.set_ylabel("lag $k$ (steps back)", fontsize=FL_, labelpad=1)
        a.set_xlabel("training episodes", fontsize=FL_, labelpad=1)
        title2(a, f"Credit vs FR: {nm}", r"$r$(share sent $k$ back, FR$_{t-k}$)")
        letter(a, s_)
    cx = add(.55 + 3 * 1.64 + 2 * .74 + .08, tops[1], .07, PH_)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap="viridis"), cax=cx, extend="both")
    cb.ax.tick_params(labelsize=FT_ - .5, length=1.5, pad=1)
    cb.outline.set_visible(False)
    cb.set_label("Pearson $r$", fontsize=FT_)
    out = fig_path("fig_learn_lever.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    figure()
