"""The 1-2 figure explanation of why the trained RNN reaches optimal Bayes and the ESN stays at naive Bayes, updated
with the training-dynamics recordings (John 11-01: "show me the original and updated figures you would make").
-> main/REPLICATION/figure_bin/fig_explanation_dynamics.png   Figure 1: what each network learns, through training
   main/REPLICATION/figure_bin/fig_explanation_mechanism.png  Figure 2: what each network can compute (endpoint)
The original (10-29) is fig_minimal_set.png (fig_minimal_set.py): its mechanism panels a-c become Figure 2 unchanged;
its d (hybrid observer: task-side, why entanglement matters) leaves the minimal set; its e (the locked learning signal
at the final networks) and f (the old runs' test logs) are replaced by Figure 1, which shows the learning itself.

Figure 1 (td_analysis.py on trained_s20261029 and echo_s20261029, 114 checkpoints each, the same 10,000 probe episodes
at every checkpoint; x: training episodes, log; dotted vertical line: where the trained RNN passes naive):
  a accuracy at T against the probe set's Bayes ladder (Bayes, window 5, factorized state, window 2, step-wise, naive)
  b the per-step update on the ladder: the network's per-step logit increment regressed on naive's increment N, the
    within-step part W = step-wise - naive and the memory part M = Bayes - step-wise (an1): coefficients
  c memory use: accuracy at T lost when the network's memory is cut to its last 1, 2, 5 steps (an6: intact - cut)
  d naive's errors: on particles where naive is wrong and Bayes right, how often the network gives naive's answer
  e the state: variance of u at step 30 explained by the latent variables (r0, r1 and their joint; cell means
    pooled over the 50 environments, i.e. the latent code shared by all environments: the environment-specific
    part sits in env x cell, audit 11-01: per environment, cell + env x cell, Trained .61 vs ESN .24) and by the
    environment (the pair of active variables = Z)
  f the learning signal: the locked statistic's r (tau = 0, steps 17-30) of credit coherence and cross-band share
    with entanglement (5,000 of the probes)
Figure 2 (the standard e5 networks; fig_minimal_set's panels): a memory cut to w steps vs windowed Bayes (12,000
  episodes); b grids at T (39 alias-scan environments x 100 cells x 50 streams); c the ESN's frozen states read out
  in other ways.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig_explanation.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT] + [os.path.join(ROOT, "main", "REPLICATION", *p) for p in ((), ("claude_plots",))]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
import plot_training_dynamics as td                                               # noqa: E402
import fig_minimal_set as fm                                                      # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402

COL = {"trained": fm.COL["Trained"], "echo": fm.COL["Echo"]}
NAME = {"trained": "Trained RNN", "echo": "ESN"}
FACTORS = td.FACTORS


def _axis(ax, title, ylabel=None):
    td._axis(ax, "")
    ax.set_title(title, loc="left", fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


def _legend(ax, styles=(), loc="best", ncol=1, nets=True):
    h = [Line2D([], [], color=COL[r], lw=2.4, label=NAME[r]) for r in COL] if nets else []
    h += [Line2D([], [], color="0.35", lw=1.6, ls=ls, label=lab) for lab, ls in styles]
    ax.legend(handles=h, fontsize=8.5, frameon=False, loc=loc, ncol=ncol)


def figure_dynamics():
    apply_fig_style()
    P = td.probes()
    S = {r: td.series(r, P) for r in COL}
    goal = np.arange(len(P["goal"])), P["goal"]
    ladder = [("Bayes", P["right_bayes"][goal].mean()),
              ("step-wise (memory of 1 step)", P["right_window1"][goal].mean()), ("naive", P["right_naive"][goal].mean())]
    naive = ladder[-1][1]
    tr = S["trained"]
    cross = tr["x"][np.flatnonzero(tr["acc"][:, -1] > naive)[0]]
    fig, axs = plt.subplots(2, 3, figsize=(3 * 4.6, 2 * 3.5))
    ax = axs.ravel()
    for name, v in ladder:
        ax[0].axhline(v, color="0.6", lw=.8, ls=":")
        ax[0].text(td.X0 * .9, v + .006, name, fontsize=8, color="0.35")
    cut = (("cut to 1 step", ":"), ("cut to 2 steps", "-."), ("cut to 5 steps", "--"))
    for run, s in S.items():
        c = COL[run]
        ax[0].plot(s["x"], s["acc"][:, -1], color=c, lw=2)
        for j, ls in enumerate(("-", "--", ":")):
            ax[1].plot(s["x"], s["beta"][:, j], color=c, ls=ls, lw=1.6)
        for j, (_, ls) in enumerate(cut):
            ax[2].plot(s["x"], s["trunc"][:, 3] - s["trunc"][:, j], color=c, ls=ls, lw=1.6)
        ax[3].plot(s["x"], s["agree"][:, 2], color=c, lw=2)
        tot = s["fac"][:, 29, 6, FACTORS.index("total"), 0]
        lat = sum(s["fac"][:, 29, 6, FACTORS.index(k), 0] for k in ("r0", "r1", "joint")) / tot
        ax[4].plot(s["x"], lat, color=c, lw=2)
        ax[4].plot(s["x"], s["fac"][:, 29, 6, FACTORS.index("env"), 0] / tot, color=c, lw=1.6, ls="--")
        ax[5].plot(s["x"], s["xc_upd"][:, 0, td.TAU0], color=c, lw=2)
        ax[5].plot(s["x"], s["xc_upd"][:, 3, td.TAU0], color=c, lw=1.6, ls="--")
    for a in ax:
        a.axvline(cross, color="0.5", lw=1, ls=":")
    ax[0].text(cross * 1.08, .795, "Trained RNN passes naive", fontsize=8, color="0.35", va="bottom")
    ax[0].set_ylim(.07, .84)
    for a in (ax[2], ax[5]):
        a.axhline(0, color="0.75", lw=.7)
    _axis(ax[0], "a   Accuracy at $T$", "goal accuracy at $T = 30$")
    _legend(ax[0], loc="lower right")
    _axis(ax[1], "b   Per-step update on the Bayes ladder:\n     "
          "$d_t = \\beta_N N + \\beta_W W + \\beta_M M$", "coefficient")
    _legend(ax[1], ((r"$\beta_N$: naive", "-"), (r"$\beta_W$: within step", "--"), (r"$\beta_M$: memory", ":")),
            loc="upper left")
    _axis(ax[2], "c   Memory use: accuracy lost by cutting memory", "intact $-$ cut accuracy at $T$")
    _legend(ax[2], cut, loc="upper left")
    _axis(ax[3], "d   Copying naive's errors (naive wrong, Bayes right)", "fraction answering naive's answer")
    _legend(ax[3], loc="upper left")
    _axis(ax[4], "e   State: variance of $u$ at step 30", "share of the variance")
    _legend(ax[4], (("latent variables, shared by all environments", "-"), ("environment (Z)", "--")), loc="upper left")
    ax[4].set_ylim(0, .78)
    _axis(ax[5], r"f   Learning signal: r with entanglement ($\tau$ = 0)", "r (within episodes)")
    _legend(ax[5], (("credit coherence", "-"), ("cross-band share", "--")), loc="upper left")
    fig.suptitle("What the trained RNN and the ESN learn, through training", fontsize=14, y=1.01)
    fig.text(.5, -.015, f"{len(P['goal']):,} probe episodes (50 environments x 100 cells x 2 streams) at each of 114 "
             "checkpoints per network; f: 5,000 of them, steps 17-30. Training episodes = episodes with an update.",
             ha="center", fontsize=9)
    fig.tight_layout()
    path = fig_path("fig_explanation_dynamics.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    i = lambda s, e: int(np.argmin(np.abs(s["x"] - e)))                           # noqa: E731
    print(f"saved {path}; trained passes naive at {cross:.0f} training episodes")
    for run, s in S.items():
        k = int(np.nanargmax(s["agree"][:, 2]))
        print(f"  {run}: final acc {s['acc'][-1, -1]:.3f}, beta {np.round(s['beta'][-1], 2)}, naive-error copy max "
              f"{s['agree'][k, 2]:.2f} at {s['x'][k]:.0f}, final {s['agree'][-1, 2]:.2f}; memory loss (w1) at 800: "
              f"{s['trunc'][i(s, 800), 3] - s['trunc'][i(s, 800), 0]:.3f}")
    return path


def figure_mechanism():
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    W_, H_ = 17.0, 5.0
    fig = plt.figure(figsize=(W_, H_))
    fig.text(.5, 1 - .3 / H_, "What each network can compute (the standard networks at the end of training)",
             ha="center", va="center", fontsize=FS["label"] + 3)
    pw, ph, y0 = 4.3, 3.2, .9
    xs = (.9, 6.55, 12.3)
    ax = fig.add_axes([xs[0] / W_, y0 / H_, pw / W_, ph / H_])
    fm.panel_truncation(ax, FS)
    ax.set_title("a   The belief is built from history", fontsize=FS["label"] + .5, loc="left", pad=6)
    sub = fm.panel_grids(fig, [xs[1] / W_, y0 / H_, pw / W_, ph / H_], FS)
    sub[0].text(0, 1.22, "b   Outcome: unfolded vs folded grid at $T$", transform=sub[0].transAxes,
                fontsize=FS["label"] + .5, ha="left", va="bottom")
    ax = fig.add_axes([(xs[2] + 1.4) / W_, y0 / H_, (pw - 1.4) / W_, ph / H_])
    fm.panel_readout(ax, FS)
    ax.set_title("c   The ESN's read-out format", fontsize=FS["label"] + .5, loc="left", pad=6)
    ax.title.set_position((-.75, 1))
    fig.text(.9 / W_, .25 / H_, "a: 12,000 episodes; memory cut to the last w read-in outputs; diamonds: intact "
             "networks.   b: 39 environments x 100 cells x 50 streams, cell means averaged over environments; number: "
             "mean distance from the true lattice.   c: the ESN's frozen state read out in other ways.",
             fontsize=FS["tick"] - 1, ha="left", va="center", color="0.25")
    path = fig_path("fig_explanation_mechanism.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    figure_dynamics()
    figure_mechanism()
