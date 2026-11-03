"""The theory session's questions about learning (memory learning-dynamics-phases), answered from the training-dynamics
recordings (John 11-01: "please go through everything you wanted to analyze"). Data: td_questions_data.py and
td_questions_frozen.py (main/DATA/experimental/training_dynamics/questions/), the per-checkpoint files of td_analysis.py
for the cell maps, and the frozen series' summaries (ff_deep/d9_frozen).
-> main/REPLICATION/figure_bin/
  td_questions_learning.png  how the trained LSTM learns:
      a memory carries everything beyond naive: accuracy at T intact and with the memory cut to 1 step (question 1)
      b width sets when it happens: test accuracy for N = 50, 100 (five worlds), 200, 1000 (question 2)
      c memory's benefit against accuracy: the same path at every width (questions 1, 2)
      d naive's wrong answers rise to the step-wise rule's level, then fall, against accuracy (question 4)
      e the state's effective dimension re-expands before the network passes naive (question 7)
      f the local log-log slope of the gap to Bayes against test accuracy: it rises through naive's accuracy without a
        stall and peaks after it (question 10; corrected 11-01, theory_v10)
  td_questions_traps.png  joint read-outs, read-out gradients, frozen layers, decoding:
      a share of probe episodes whose true pair is shut out, q(c*|g*) < e^-10, for the three joint read-outs (question 8)
      b the same per cell (r0 x r1) at three checkpoints, trained and ESN joint read-outs (question 8)
      c read-out row gradient norms up to 3k episodes: marginal vs joint read-out (question 6)
      d frozen layers squeeze their input without losing it: after k frozen layers the input is still linearly
        readable once rescaled, while all inputs' representations converge and accuracy falls (question 5)
      e linear decoders (a side statistic): counts and Bayes's joint posterior from the state (question 3)
      f the shut-out pairs carry the whole cost: accuracy at T on probes whose true pair is alive / shut out (question 8)
Findings (11-01, one seed per condition unless stated; the four other N = 100 worlds agree):
  1 Trained RNN: memory's benefit (intact - cut to 1 step) grows with its accuracy, to .46 at ~3k episodes; cut to 1 step
    it ends at naive (.46). ESN: benefit <= .08.
  2 Passing naive: 790 / 1.9k / 4.3-4.9k (5 worlds) / 14k training episodes for N = 1000 / 200 / 100 / 50. Against
    accuracy the widths follow nearly one path (naive copying, ladder coefficients beta_W, beta_M; theory_v10); width
    sets the speed. Narrower networks lean less on memory at the same accuracy (cut to 1 step: .28 at N = 50 against
    .20 at N = 1000, at accuracy .50; panel c), and weight naive's term more (beta_N .91 against .72 at naive's accuracy).
    Memory coefficient (theory_v10 rounds 2-3: the part of Bayes's memory term that the current observation does not fix,
    regressed with the network's own previous log-belief as a control): .005-.007 at accuracy .25 (the marginal ESN's
    level: no detectable memory term, though the updates already depend on the past in a way the network's own belief
    accounts for) and .04-.07 at naive's accuracy at every width, against .28 (MLP-rescued ESN) and .16 (joint read-out).
  4 Naive's wrong answer on naive-wrong / Bayes-right probes: rises to ~.29 at accuracy ~.35-.40 at every width, then
    falls to .03 (N = 1000); the ESN climbs to .50 and stays. CORRECTED 11-01 (theory_v10): the step-wise rule itself
    gives naive's answer on .318 of these probes, so the climbers' peak (.27-.32) does not single out a naive stage; the
    evidence for naive's term first is the ladder regression (beta = (.44, .07, .00) at accuracy .25 at every width).
  7 The state collapses to PR ~1.2-1.4 within 20-80 episodes and re-expands; PR is back to 10 at 480 / 800 / 3.6k / 6.4k
    episodes, before passing naive (880 / 2.1k / 6.4k / 22k on the probes); final PR 57 / 18 / 13 / 11, ESN 18.
  10 Log-log slope of the gap to Bayes: fitted before / after passing naive it is .13-.32 / .26-.99, but the local slope
    rises through naive's accuracy without a stall, from ~.1 at every width to a peak after naive (1.55 at N = 1000,
    accuracy .56-.64; .74 at N = 200; about .5 at N = 100, .49-.58 in five worlds; N = 50 levels off near .26);
    MLP-rescued ESN .91, joint-loss ESN .52, ESNs <= .2. At N = 200 and below the crossing falls inside a gradual rise;
    at N = 1000 most of the rise falls within a factor of three in updates around it, too steep to tell a kink from a
    smooth rise (CORRECTED 11-01, theory_v10 round-1 check: "no break at naive" was too strong; "no stall" holds).
  8 Joint read-outs on the goal loss: ~70 cells shut out within 20-50 episodes in both. Trained: the share of probes
    shut out falls .77 -> .43 (~800) -> .29 (40k); 29 cells stay dead, 71 alive, none mixed; living cells' gate tracks
    Bayes's (r .75). At the end it is Bayes-accurate where the pair is alive (.771 vs Bayes .770) and .48 where it is
    shut out (Bayes .775): the shut-out pairs carry its whole deficit against the marginal read-out (.69 vs .76). ESN: .70, its gate keeps closing (median log q(c*|g*) -56). Joint loss: none.
  6 Marginal read-out: all 20 rows get similar gradient (no starvation). Joint read-out on the goal loss: 40-60 cells'
    rows below 1e-4 from ~20 episodes; their probes are the shut-out ones (96-99%).
  5 Frozen series (depth 9, k frozen after the trained read-in): the input stays linearly readable after the last frozen
    layer once rescaled (R^2 .99 -> .83 at k = 8; the ridge penalty is relative to the features' spectrum, an absolute
    one fakes total loss), while the mean cosine between two inputs' representations rises .50 -> .999 and accuracy
    falls .73 -> .25: information kept, scale lost (the conditioning account of theory_v9 3.3). RETRACTED 11-01: "the
    interactions are lost at the cliff" - the pairwise-product measure is small at every layer, differs between
    networks (the k = 4 net's trained read-in already shows none) and is dominated by z x z products, which cannot be
    read out across held-out environments from 25 fitting environments.
  3 (decoders) Trained: counts' R^2 dips .85 -> .43 (~1k) -> .90 while the posteriors rise (joint .77, marginals .88);
    ESN: counts ~.8 throughout, posteriors <= .25.
  9 Loss at T only: both versions stayed at chance (power .5 collapsed at update 1; power 1 at .10 to 22.6k): no answer.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_questions.py"""
import os
import sys
import json
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt                                                    # noqa: E402
from matplotlib.lines import Line2D                                                # noqa: E402
from main.utils import fig_path, apply_fig_style                                   # noqa: E402
from td_questions_data import gate                                                 # noqa: E402

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
QD = os.path.join(TD, "questions")
AN = os.path.join(TD, "analysis")
FF = os.path.join(ROOT, "main", "DATA", "experimental", "ff_deep")
X0, XMAX = 10, 5e4
WIDTH = {"trained_N50": ("N = 50", "#c6dbef"), "trained_N100": ("N = 100", "#9ecae1"),
         "trained_N200": ("N = 200", "#4292c6"), "trained": ("N = 1000", "#1f77b4")}
OTHER = {"echo": ("ESN", "#ff7f0e"), "mlp55": ("MLP-rescued ESN", "#8c564b"),
         "joint_trained": ("Trained, joint read-out", "#2ca02c"), "joint_echo": ("ESN, joint read-out", "#e377c2"),
         "joint_echo_jl": ("ESN, joint read-out + joint loss", "#9467bd")}
WORLDS = ["trained_N100_s20261030", "trained_N100_s20261031", "trained_N100_s20261032", "trained_N100_s20261033"]
NAIVE, BAYES, STEPWISE = .464, .771, .608                                          # probe-set references at T
STEPWISE_COPY = .318       # the step-wise rule's answers equal naive's on .318 of the naive-wrong / Bayes-right probes
NAIVE_TEST = .447          # naive's accuracy at T on the runs' own test episodes


def ck(run):
    return dict(np.load(os.path.join(QD, f"ck_{run}.npz")))


def test(run):
    return dict(np.load(os.path.join(QD, f"test_{run}.npz")))


def xs(x, x0=X0):
    return np.where(np.asarray(x) > 0, x, x0).astype(float)


def binned(run, key="acc"):
    """a run's test log in log-spaced bins of training episodes (the same bins for every run): mean accuracy at T and
    mean gap to Bayes on the same test episodes"""
    t = test(run)
    x = (t["ep"] - t["ep"] // 5).astype(float)
    edges = np.concatenate([[0, 1], np.logspace(np.log10(20), np.log10(4e4), 45)])
    xb, ab, gb = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum():
            xb.append(np.exp(np.log(np.maximum(x[m], X0)).mean()))
            ab.append(t["acc_T"][m].mean())
            gb.append((t["bayes_T"] - t["acc_T"])[m].mean())
    return np.array(xb), np.array(ab), np.array(gb)


def slope(x, g, a, b):
    m = (x >= a) & (x <= b) & (g > 0)
    return -np.polyfit(np.log(x[m]), np.log(g[m]), 1)[0]


def crossing(x, a, level=.45):
    i = np.where(a >= level)[0]
    return x[i[0]] if len(i) else None


def time_axis(ax, title, xlabel=True, x0=X0):
    """log axis of training episodes; the initial network is drawn at x0 ("0")"""
    ax.set_xscale("log")
    ax.set_xlim(x0 * .8, XMAX)
    ticks = [x0] + [t for t in (10, 1e2, 1e3, 1e4) if t > x0]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0"] + [{10: "10", 1e2: "100", 1e3: "1k", 1e4: "10k"}[t] for t in ticks[1:]])
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel("training episodes")
    ax.spines[["top", "right"]].set_visible(False)


def plain(ax, title):
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)


def refs(ax, which=("Bayes", "naive")):
    vals = {"Bayes": BAYES, "naive": NAIVE, "step-wise": STEPWISE}
    for name in which:
        ax.axhline(vals[name], color="0.6", lw=.8, ls=":")
        ax.text(XMAX * .95, vals[name] + .008, name, fontsize=8, color="0.4", ha="right")


def fig_learning():
    fs = apply_fig_style()
    fig, axs = plt.subplots(2, 3, figsize=(3 * 4.9, 2 * 3.8))
    ax = axs.ravel()
    D = {r: ck(r) for r in list(WIDTH) + ["echo", "mlp55"]}
    # a: intact vs memory cut to 1 step
    for r, (lab, c) in [("trained", ("Trained RNN", WIDTH["trained"][1])), ("echo", OTHER["echo"])]:
        d = D[r]
        ax[0].plot(xs(d["x"]), d["trunc"][:, 3], color=c, lw=2)
        ax[0].plot(xs(d["x"]), d["trunc"][:, 0], color=c, lw=1.6, ls=":")
    refs(ax[0], ("Bayes", "step-wise", "naive"))
    time_axis(ax[0], "a  Beyond naive is memory\naccuracy at $T$: intact vs memory cut to 1 step")
    ax[0].set_ylabel("accuracy at $T$")
    ax[0].legend(handles=[Line2D([], [], color=WIDTH["trained"][1], lw=2, label="Trained RNN"),
                          Line2D([], [], color=OTHER["echo"][1], lw=2, label="ESN"),
                          Line2D([], [], color="0.35", lw=2, label="intact"),
                          Line2D([], [], color="0.35", lw=1.6, ls=":", label="memory cut to 1 step")],
                 fontsize=8, frameon=False, loc="upper left")
    # b: width sets when (test logs)
    for w in WORLDS:
        x, a, _ = binned(w)
        ax[1].plot(x, a, color=WIDTH["trained_N100"][1], lw=.9, alpha=.8)
    for r, (lab, c) in WIDTH.items():
        x, a, _ = binned(r)
        ax[1].plot(x, a, color=c, lw=2, label=lab)
        xc = crossing(x, a)
        if xc:
            ax[1].plot(xc, .45, "o", color=c, mec="k", mew=.8, ms=6, zorder=5)
    ax[1].axhline(.45, color="0.6", lw=.8, ls=":")
    time_axis(ax[1], "b  Width sets when it happens\ntest accuracy at $T$ (dots: passes naive)")
    ax[1].set_ylabel("accuracy at $T$")
    ax[1].legend(fontsize=8, frameon=False, loc="upper left", title="Trained RNN", title_fontsize=8)
    # c, d, e against accuracy / time
    for r, (lab, c) in list(WIDTH.items()) + [("mlp55", OTHER["mlp55"]), ("echo", OTHER["echo"])]:
        d = D[r]
        ax[2].plot(d["acc"], d["trunc"][:, 3] - d["trunc"][:, 0], color=c, lw=2)
        ax[3].plot(d["acc"], d["copy_naive"], color=c, lw=2)
        ax[3].plot(d["acc"], d["copy_bayes"], color=c, lw=1.6, ls="--")
    ax[3].axhline(STEPWISE_COPY, color="0.45", lw=1, ls="-.")
    ax[3].text(.765, STEPWISE_COPY + .015, "step-wise rule: as naive", fontsize=8, color="0.35", ha="right")
    for r, (lab, c) in list(WIDTH.items()) + [("echo", OTHER["echo"])]:
        d = D[r]
        ax[4].plot(xs(d["x"]), d["pr_u"], color=c, lw=2)
        xc = crossing(d["x"], d["acc"], NAIVE)
        if xc:
            i = list(d["x"]).index(xc)
            ax[4].plot(xs([xc])[0], d["pr_u"][i], "o", color=c, mec="k", mew=.8, ms=6, zorder=5)
    for a in (ax[2], ax[3]):
        a.axvline(NAIVE, color="0.6", lw=.8, ls=":")
        a.set_xlabel("accuracy at $T$ (dotted: naive)")
    plain(ax[2], "c  Memory's benefit grows with accuracy, less when narrow\nmemory's benefit: acc(intact) $-$ acc(cut to 1)")
    ax[2].set_ylabel("accuracy gained from memory")
    ax[2].legend(handles=[Line2D([], [], color=c, lw=2, label=lab) for lab, c in
                          [WIDTH[r] for r in WIDTH] + [OTHER["mlp55"], OTHER["echo"]]],
                 fontsize=8, frameon=False, loc="upper left")
    plain(ax[3], "d  Naive's wrong answers peak at the step-wise rule's level\nprobes where naive errs and Bayes is right")
    ax[3].set_ylabel("share of those probes")
    ax[3].legend(handles=[Line2D([], [], color="0.35", lw=2, label="answers as naive (wrong)"),
                          Line2D([], [], color="0.35", lw=1.6, ls="--", label="answers as Bayes (right)")],
                 fontsize=8, frameon=False, loc="upper left")
    ax[4].set_yscale("log")
    time_axis(ax[4], "e  The state re-expands before passing naive\n"
              r"PR $=(\Sigma_k\lambda_k)^2/\Sigma_k\lambda_k^2$ of $u_t$, mean over steps")
    ax[4].set_ylabel("PR of the state $u_t$")
    ax[4].legend(handles=[Line2D([], [], color="0.35", lw=0, marker="o", mec="k", ms=6, label="passes naive")],
                 fontsize=8, frameon=False, loc="upper left")
    # f: local log-log slope of the gap to Bayes against accuracy (no stall at naive's accuracy)
    cross = {}
    rows = list(WIDTH.items()) + [("mlp55", OTHER["mlp55"]), ("joint_trained", OTHER["joint_trained"]),
                                  ("echo", OTHER["echo"])]
    handles = []
    for r, (lab, c) in rows:
        x, a, g = binned(r)
        ok = (x >= 50) & (g > 0)
        x, a, g = x[ok], a[ok], g[ok]
        loc = np.array([slope(x[max(i - 2, 0):i + 3], g[max(i - 2, 0):i + 3], 0, np.inf) for i in range(len(x))])
        ax[5].plot(a[2:-2], loc[2:-2], color=c, lw=2)
        i = int(np.argmax(loc[2:-2])) + 2
        cross[r] = (crossing(x, a, NAIVE_TEST), float(loc[i]), float(a[i]))
        short = lab.replace("Trained, joint read-out", "joint read-out").replace("MLP-rescued ESN", "MLP-rescued")
        handles.append(Line2D([], [], color=c, lw=2, label=f"{short}: peak {loc[i]:.2f}"))
    ax[5].axvline(NAIVE_TEST, color="0.6", lw=.8, ls=":")
    plain(ax[5], "f  The gap closes faster through naive's accuracy, no stall\n"
          r"local slope $-d\log g\,/\,d\log x$ ($g$: gap to Bayes, $x$: episodes)")
    ax[5].set_xlabel("test accuracy at $T$ (dotted: naive)")
    ax[5].set_ylabel("local log-log slope")
    ax[5].legend(handles=handles, fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout(h_pad=2.2, w_pad=1.5)
    fig.text(.5, -.035, "a, c, d, e: 10,000 probe episodes (50 environments x 100 cells x 2 streams, test-variable pairs; "
             "one world) at every checkpoint (114 per run); b, f: the runs' own test episodes (8,000 per test, every "
             "5th episode), log-spaced bins; N = 100 also four other worlds (thin). Trained RNN = the trained LSTM. "
             "f: slope fitted over 5 neighbouring bins, from 50 episodes.", ha="center", fontsize=8, wrap=True)
    fig.savefig(fig_path("td_questions_learning.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return cross


def dead_map(run, ep, P):
    z = np.load(os.path.join(AN, run, f"ck_{ep:05d}.npz"))
    g = gate(z["cell_lqT"], P["goal"], P["r"])
    cell = P["r"][:, 0] * 10 + P["r"][:, 1]
    return np.array([(g[cell == c] < -10).mean() for c in range(100)]).reshape(10, 10), int(z["episode"])


def fig_traps():
    apply_fig_style()
    P = dict(np.load(os.path.join(AN, "probes.npz")))
    x0 = 2                                   # joint runs were checkpointed every 5 episodes early on: show updates 4-16
    fig = plt.figure(figsize=(3 * 4.9, 2 * 4.0))
    gs = fig.add_gridspec(2, 3, hspace=.62, wspace=.34)
    # a: shut-out share
    ax = fig.add_subplot(gs[0, 0])
    for r in ("joint_trained", "joint_echo", "joint_echo_jl"):
        d = ck(r)
        lab, c = OTHER[r]
        ax.plot(xs(d["x"], x0), d["gate_dead"], color=c, lw=2, label=lab)
    time_axis(ax, "a  The trained net revives shut-out pairs\n"
              r"share of probes with $q(c^*\,|\,g^*) < e^{-10}$ at $T$", x0=x0)
    ax_a = ax
    ax.set_ylabel("share of probe episodes")
    ax.set_ylim(-.02, .85)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right", bbox_to_anchor=(1, .06))
    # b: cell maps
    sub = gs[0, 1].subgridspec(2, 3, wspace=.08, hspace=.3)
    eps = (50, 1000, 49995)
    maps = []
    for i, r in enumerate(("joint_trained", "joint_echo")):
        for j, ep in enumerate(eps):
            a = fig.add_subplot(sub[i, j])
            maps.append(a)
            m, e = dead_map(r, ep, P)
            a.imshow(m, cmap="viridis", vmin=0, vmax=1, origin="lower")
            a.set_xticks([])
            a.set_yticks([])
            if i == 1:
                a.set_xlabel(f"{e - e // 5:,} episodes", fontsize=8)
            if j == 0:
                a.set_ylabel("Trained" if r == "joint_trained" else "ESN", fontsize=8)
    fig.canvas.draw()
    bb = [a.get_position() for a in maps]
    x_mid = (min(b.x0 for b in bb) + max(b.x1 for b in bb)) / 2
    fig.text(x_mid, ax_a.get_position().y1 + .012, "b  The same, cell by cell\n"
             r"$r_0 \times r_1$; bright = shut out", ha="center", va="bottom",
             fontsize=fig.axes[0].title.get_fontsize())
    # c: read-out row gradients
    ax = fig.add_subplot(gs[0, 2])
    for r, c, lw in (("trained", WIDTH["trained"][1], .7), ("joint_trained", OTHER["joint_trained"][1], .4)):
        d = ck(r)
        ax.plot(xs(d["ro_x"], x0), d["ro"], color=c, lw=lw, alpha=.8)
    ax.set_yscale("log")
    time_axis(ax, "c  No starvation with a marginal output\n"
              r"gradient norm per read-out row, $\Vert \partial L / \partial W_{row} \Vert$", x0=x0)
    ax.set_xlim(x0 * .8, 3.2e3)
    ax.set_xticks([x0, 10, 1e2, 1e3])
    ax.set_xticklabels(["0", "10", "100", "1k"])
    ax.set_ylabel("gradient norm")
    ax.legend(handles=[Line2D([], [], color=WIDTH["trained"][1], lw=2, label="Trained RNN, marginal (20 rows)"),
                       Line2D([], [], color=OTHER["joint_trained"][1], lw=2, label="Trained, joint read-out (100 cells)")],
              fontsize=7.5, frameon=False, loc="lower left")
    # d: frozen layers
    ax = fig.add_subplot(gs[1, 0])
    z = np.load(os.path.join(QD, "frozen_decoding.npz"))
    acc = []
    for k in range(9):
        f = (os.path.join(FF, "d9_p1_lr0.001_rep0_summary.json") if k == 0 else
             os.path.join(FF, "d9_frozen", f"d9_p1_lr0.001_frz{''.join(str(i) for i in range(1, k + 1))}_rep0_summary.json"))
        acc.append(json.load(open(f))["acc_last3"])
    ax.plot(z["k"], z["lin"], color="0.2", lw=2, marker="o", ms=4, label="input read out linearly ($R^2$, rescaled)")
    ax.plot(z["k"], z["cos"], color="0.6", lw=2, marker="o", ms=4, label="cosine of two inputs' representations")
    ax.plot(np.arange(9), acc, color="#1f77b4", lw=2, ls="--", marker="s", ms=4, label="network's accuracy at $T$")
    plain(ax, "d  Frozen layers squeeze the input without losing it\n(after layer $k$: readable, but all inputs alike)")
    ax.set_xlabel("frozen layers $k$ (depth 9, after the trained read-in)")
    ax.set_ylabel("$R^2$ / cosine / accuracy")
    ax.set_ylim(-.03, 1.05)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    # e: decoding
    ax = fig.add_subplot(gs[1, 1])
    for r, (lab, c) in (("trained", ("Trained RNN", WIDTH["trained"][1])), ("echo", OTHER["echo"])):
        d = ck(r)
        ax.plot(xs(d["x"]), d["dec"][:, -1, 0], color=c, lw=2)
        ax.plot(xs(d["x"]), d["dec"][:, -1, 1], color=c, lw=1.8, ls="--")
    time_axis(ax, "e  Decoders: the ESN keeps the counts, not\nthe posterior; the trained net gets both")
    ax.set_ylabel("$R^2$, held-out pairs")
    ax.legend(handles=[Line2D([], [], color=WIDTH["trained"][1], lw=2, label="Trained RNN"),
                       Line2D([], [], color=OTHER["echo"][1], lw=2, label="ESN"),
                       Line2D([], [], color="0.35", lw=2, label="counts $n_t$"),
                       Line2D([], [], color="0.35", lw=1.8, ls="--", label="Bayes's joint log posterior")],
              fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(0, .55))
    # f: what the shut-out pairs cost
    ax = fig.add_subplot(gs[1, 2])
    d = ck("trained")
    ax.plot(xs(d["x"], x0), d["acc"], color=WIDTH["trained"][1], lw=2)
    for r in ("joint_trained", "joint_echo"):
        d = ck(r)
        c = OTHER[r][1]
        ax.plot(xs(d["x"], x0), d["acc_split"][:, 0], color=c, lw=2)
        ax.plot(xs(d["x"], x0), d["acc_split"][:, 1], color=c, lw=1.8, ls=":")
    for v in (BAYES, NAIVE):
        ax.axhline(v, color="0.6", lw=.8, ls=":")
    time_axis(ax, "f  The shut-out pairs carry the whole cost\naccuracy at $T$: true pair alive vs shut out", x0=x0)
    ax.set_ylabel("accuracy at $T$ (probes)")
    ax.legend(handles=[Line2D([], [], color=WIDTH["trained"][1], lw=2, label="Trained, marginal (all probes)"),
                       Line2D([], [], color=OTHER["joint_trained"][1], lw=2, label="Trained, joint read-out"),
                       Line2D([], [], color=OTHER["joint_echo"][1], lw=2, label="ESN, joint read-out"),
                       Line2D([], [], color="0.35", lw=2, label="pair alive"),
                       Line2D([], [], color="0.35", lw=1.8, ls=":", label="pair shut out")],
              fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(0, .755))
    fig.text(.5, .015, "a-c, e: 10,000 probe episodes (50 environments x 100 cells x 2 streams; one world) at every "
             "checkpoint (146 per joint run, 114 otherwise; c: to 3k episodes). Shut out: the true pair's within-row "
             "belief $q(c^*|g^*)$ below $e^{-10}$ (Bayes's never is). d: depth-9 feedforward nets on the counts, k "
             "frozen layers, one seed each; read-outs fitted on 25 environments' inputs, tested on 15 others; ridge $R^2$ "
             "in e at step 30, held-out pairs. f: probes as in a; grey dotted lines: Bayes (.77) and naive (.46).", ha="center", fontsize=8, wrap=True)
    fig.savefig(fig_path("td_questions_traps.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    cross = fig_learning()
    print({r: (round(v[1], 2), round(v[2], 2)) for r, v in cross.items()})
    fig_traps()
    print("saved td_questions_learning.png, td_questions_traps.png")
