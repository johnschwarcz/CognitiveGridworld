"""Networks against the Bayesian bounds, over the sanity_default_reps replications.

Writes sanity_default_reps_perf.svg: accuracy against inference time for the experts and
the baselines, then the same thing as a trajectory.

The grammar is collection_plotters.plot_perf's, so this figure reads the same way as the
others in the paper: colour is C, marker shape and edge are the agent (circle/black for
exact Bayes, triangle/red for the network), solid Bayes against a dashed network, markers
at a few sampled inference steps. Each curve is a mean over replications and nothing
else -- the spread is named in the keys as n and belongs in the caption as a number.

Showing that spread graphically was tried and dropped. At 5 replications it is 0.5-0.9%
of the y range, so a bar is 0.4-0.6 pt against a marker radius near 2.9 pt: bands, bars
and hybrids all cost a visual channel and reveal nothing. spread_report() prints the
figures for the caption. To actually SEE it, plot the shortfall from the bound instead,
the way compare_matched_params.py does -- that axis spans ~0.03 rather than 0.70.

    python main/REPLICATION/compare_sanity_reps.py             # plot from the cache
    python main/REPLICATION/compare_sanity_reps.py --collect    # re-collect (needs a GPU)

Each replication drew its own world, so the joint and naive bounds are per-rep too and
carry their own spread -- they are not one fixed line to plot the networks against. Each
panel pairs a network with the bound computed from that network's OWN replications, so
the two curves rest on the same n and the same worlds: the trained nets against the joint
bound of their draws, the echo states against the naive bound of theirs. The networks are
read from the final checkpoint of test_acc_through_training, which is already accuracy
against inference step; the bounds are recomputed by loading each rep's stored
environment. Both are cached in CACHE_NPZ.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _repo_root():
    """Walk up from this file (or cwd) until a dir containing main/DATA is found."""
    start = os.path.abspath(globals().get("__file__") or os.getcwd())
    d = start if os.path.isdir(start) else os.path.dirname(start)
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "main", "DATA")):
            return d
        d, prev = os.path.dirname(d), d
        if d == prev:
            break
    raise RuntimeError(f"could not locate the repo root (main/DATA) from {start}")


path = _repo_root()
sys.path.insert(0, path)
from main.utils import fig_path, apply_fig_style

DATA = os.path.join(path, "main", "DATA")
CACHE_NPZ = os.path.join(DATA, "sanity_default_reps.npz")
CKPT_DIR = "/sanity _default_reps/"   # the stray space is in the directory name itself
REPS = range(5)
CTXS = [1, 2]
COLLECT_TOTAL = 32000   # episodes per rep behind the cached bounds
COLLECT_BATCH = 8000
STEPS = 30
HID = 1000

FIGSIZE = (9.4, 3.0)
YLIM = (.15, .85)
RATIO = (r"$\frac{\mathrm{Joint}}{\mathrm{Naive}}$",
         r"$\frac{\mathrm{Fully\ Trained}}{\mathrm{Echo\ State}}$")
# the agents each panel actually holds, named there rather than in one shared key
NAMES = {"Experts": ("Joint", "Fully Trained"), "Baselines": ("Naive", "Echo State")}

# paper grammar: colour says which C, marker shape and edge say which agent, so the two
# crossed factors get one visual channel each and neither has to borrow the other's
MARK_S = 34              # scatter area; the paper uses 50 in a wider figure
MARK_LW = 1.1
MARK_N = 4               # sampled inference steps carrying a marker
MEC_BAYES, MEC_NET = "k", "r"
BAYES_LS, NET_LS = "-", "--"
REL_BAYES_LS = ":"       # panel 3 in the paper is dotted Bayes against the dashed network
# The paper draws every curve at 2.5 in a 12 in figure, which is ~1.35 pt once placed at
# column width; 1.8 in this 9.4 in figure prints to about the same weight.
LW = 1.8


# --------------------------------------------------------------------------- data

def collect(total=COLLECT_TOTAL, batch=COLLECT_BATCH, cuda=0):
    """Bayes bounds and network accuracy for every replication in CKPT_DIR.

    Chunked to a fixed episode total per rep, the way compare_matched_params does it, so
    the batch can shrink without changing the sample behind a bound.
    """
    import time
    import torch
    from main.CognitiveGridworld import CognitiveGridworld

    rounds = max(1, total // batch)
    res = {}
    for ctx in CTXS:
        for kind in ("fully_trained", "reservoir"):
            net, joint, naive = [], [], []
            for r in REPS:
                stem = f"{CKPT_DIR}{kind}_ctx_{ctx}_rep{r}"
                d = torch.load(f"{DATA}{stem}_net.pth", map_location="cpu",
                               weights_only=False)
                net.append(np.asarray(d["test_acc_through_training"], float)[-1])
                acc = {"joint": 0., "naive": 0.}
                t0 = time.time()
                for _ in range(rounds):
                    g = CognitiveGridworld(
                        mode="SANITY", cuda=cuda, episodes=1, checkpoint_every=1,
                        realization_num=10, obs_num=5, show_plots=False, state_num=500,
                        learn_embeddings=False, classifier_LR=.001, ctx_num=ctx,
                        training=False, batch_num=batch, step_num=STEPS, hid_dim=HID,
                        reservoir=(kind == "reservoir"), load_env=stem)
                    for k in acc:
                        acc[k] = acc[k] + np.asarray(getattr(g, f"{k}_acc"), float).sum(0)
                    del g
                    torch.cuda.empty_cache()
                joint.append(acc["joint"] / (batch * rounds))
                naive.append(acc["naive"] / (batch * rounds))
                print(f"  ctx{ctx} {kind:13s} rep{r}: net {net[-1][-1]:.3f}  "
                      f"joint {joint[-1][-1]:.3f}  naive {naive[-1][-1]:.3f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
            res[f"ctx{ctx}_{kind}_net"] = np.array(net)
            res[f"ctx{ctx}_{kind}_joint"] = np.array(joint)
            res[f"ctx{ctx}_{kind}_naive"] = np.array(naive)
    np.savez(CACHE_NPZ, **res)
    print("saved", CACHE_NPZ)
    return res


# --------------------------------------------------------------------------- plots

def plot_perf(D):
    """The 1 x 3 performance figure, in the paper's marker grammar."""
    col = {c: plt.cm.viridis(v)
           for c, v in zip(CTXS, np.linspace(0.15, 0.85, len(CTXS)))}
    T = D["ctx1_fully_trained_net"].shape[1]
    t = np.arange(T)
    idx = np.linspace(0, T - 1, MARK_N, dtype=int)

    # Each panel's bound comes from the same replications as its network, so the pair
    # rests on one n and one set of worlds rather than on a bound pooled across both.
    panel = {"Experts":   lambda c: (D[f"ctx{c}_fully_trained_joint"],
                                     D[f"ctx{c}_fully_trained_net"]),
             "Baselines": lambda c: (D[f"ctx{c}_reservoir_naive"],
                                     D[f"ctx{c}_reservoir_net"])}

    n = D[f"ctx{CTXS[0]}_fully_trained_net"].shape[0]

    def series(ax, x, y, c, ls, marker, mec, z):
        ax.plot(x, y, color=col[c], lw=LW, ls=ls, zorder=z)
        ax.scatter(x[idx], y[idx], s=MARK_S, marker=marker, facecolors=col[c],
                   edgecolors=mec, linewidths=MARK_LW, zorder=z + 10)

    def key(label, n, marker, mec):
        """The paper's key: a hollow marker, carrying the sample behind the mean."""
        return Line2D([], [], color="none", marker=marker, mec=mec, mfc="none",
                      ms=6.4, mew=MARK_LW,
                      label=label if n is None else f"{label} ($n = {n}$)")

    fs = apply_fig_style()
    fig, axs = plt.subplots(1, 3, figsize=FIGSIZE)

    for k, title in enumerate(panel):
        ax = axs[k]
        for c in CTXS:
            bayes, net = panel[title](c)
            series(ax, t, bayes.mean(0), c, BAYES_LS, "o", MEC_BAYES, 3)
            series(ax, t, net.mean(0), c, NET_LS, "^", MEC_NET, 6)
        ax.set(title=title, xlabel="Inference Time", ylim=YLIM)
        ax.set_yticks(np.linspace(.2, .8, 4))
        leg = ax.legend(handles=[key(NAMES[title][0], n, "o", MEC_BAYES),
                                 key(NAMES[title][1], n, "^", MEC_NET)],
                        loc="upper left", frameon=False, fontsize=fs["legend"] - .5,
                        handlelength=1.8, labelspacing=.3, borderpad=.2)
        ax.add_artist(leg)
        if k == 0:
            ax.legend(handles=[Line2D([], [], color=col[c], lw=2.4, label=f"$C = {c}$")
                               for c in CTXS], loc="lower right", frameon=False,
                      fontsize=fs["legend"] - .5, labelspacing=.3, borderpad=.2)
    axs[0].set_ylabel("Accuracy")
    axs[1].tick_params(labelleft=False)

    # panel 3: the same, as a trajectory -- baseline accuracy against expert accuracy
    ax = axs[2]
    ax.plot([0, 1], [0, 1], ls="-", c="gray", lw=1, zorder=1)
    for c in CTXS:
        mxb = D[f"ctx{c}_reservoir_naive"].mean(0)
        myb = D[f"ctx{c}_fully_trained_joint"].mean(0)
        mxn = D[f"ctx{c}_reservoir_net"].mean(0)
        myn = D[f"ctx{c}_fully_trained_net"].mean(0)
        series(ax, mxb, myb, c, REL_BAYES_LS, "o", MEC_BAYES, 3)
        series(ax, mxn, myn, c, NET_LS, "^", MEC_NET, 6)
        # one ratio per context count, offset so the two never land on each other
        ax.annotate(r"$\times$" + f"{myb[-1] / mxb[-1]:.2f}", (mxb[-1], myb[-1]),
                    xytext=(6, -10 if c == 1 else 4), textcoords="offset points",
                    color=col[c], fontsize=fs["annot"], fontweight="bold")
    ax.annotate("start", (D["ctx1_reservoir_naive"].mean(0)[0],
                          D["ctx1_fully_trained_joint"].mean(0)[0]),
                xytext=(10, -12), textcoords="offset points", fontsize=fs["annot"],
                color="0.35", arrowprops=dict(arrowstyle="-", color="0.55", lw=.9))
    ax.set(title="Relative Accuracy", xlabel="Baseline accuracy",
           ylabel="Expert accuracy", xlim=(.15, .9), ylim=(.15, .9))
    ax.set_xticks(np.linspace(.2, .8, 4))
    ax.set_yticks(np.linspace(.2, .8, 4))

    for ax in axs:
        ax.grid(alpha=.3, lw=.6)
        ax.spines[["top", "right"]].set_visible(False)

    # below the diagonal is the only empty region here, and the agents are already named
    # with their n in the first two panels, so these keys carry the ratio labels alone
    axs[2].legend(handles=[key(RATIO[0], None, "o", MEC_BAYES),
                           key(RATIO[1], None, "^", MEC_NET)],
                  loc="lower right", frameon=False, fontsize=fs["legend"] - .5,
                  handlelength=1.8, labelspacing=.9, borderpad=.2)
    fig.tight_layout(w_pad=1.3)
    out = fig_path("sanity_default_reps_perf.svg")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)
    return out


def spread_report(D):
    """The spread the figure does not draw, for the caption."""
    worst = 0.
    for c in CTXS:
        for nm, key in (("trained", "fully_trained"), ("echo", "reservoir")):
            a = np.asarray(D[f"ctx{c}_{key}_net"], float)
            sd = a.std(0, ddof=1)
            sem = sd / np.sqrt(a.shape[0])
            worst = max(worst, sem.max())
            print(f"  C={c} {nm:8s} n={a.shape[0]}  final SEM {sem[-1]:.4f}  "
                  f"final SD {sd[-1]:.4f}  (max SEM over steps {sem.max():.4f})")
    print(f"  caption: mean over replications; SEM <= {np.ceil(worst * 1000) / 1000:.3f} "
          f"at every inference step")


if __name__ == "__main__":
    if "--collect" in sys.argv or not os.path.exists(CACHE_NPZ):
        D = collect()
    else:
        D = dict(np.load(CACHE_NPZ))
    plot_perf(D)
    spread_report(D)
    # the backend is whatever the environment picked, so this opens a window when run
    # interactively. Skipped outright under a file-only backend, which would otherwise
    # warn that its canvas cannot be shown.
    if plt.get_backend().lower() not in ("agg", "pdf", "ps", "svg", "cairo", "template"):
        plt.show()
