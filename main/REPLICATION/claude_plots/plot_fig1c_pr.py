"""Participation ratio of the beliefs and of their updates, with the covariance taken over particles POOLED across
environments (25 Oct 2026) -> main/REPLICATION/figure_bin/fig1c_participation_ratio.png

Particles: the trained network's gradflow harvest, first 20,000 episodes (fig1_movie.beliefs_every_step: all
four agents' goal-ordered beliefs at every step on the same episodes; each episode is one stream of one cell of one
environment, about 1275 environments), episodes of same-variable environments (exactly symmetric tables) left out.
A belief is 20 numbers (goal-variable marginal | contextual-variable marginal), at most 18 independent dimensions.
  B_t  the beliefs after step t;  U_t = B_t - B_{t-1}, B_0 = the uniform prior (0.1)
  C    the covariance over all pooled particles (1/n);  PR = (tr C)^2 / tr(C^2), the effective number of dimensions;
       tr C the total variance (probability units squared)
  (c)  the same in square-root coordinates (sqrt B_t, sqrt B_t - sqrt B_{t-1})
Bands: 95% bootstrap over environments as clusters (resampling environments with all their episodes; 1000 resamples),
from per-environment sums of x and x x^T. Data cached in figdata/round4b/pr_pooled.npz (rebuilt if missing or with
--rebuild; about 1 min, CPU).
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_pr.py [--rebuild]"""
import os
import sys
import numpy as np


def _root():
    for start in (globals().get("__file__"), globals().get("__vsc_ipynb_file__"), os.getcwd()):
        d = os.path.abspath(start) if start and os.path.exists(start) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return d
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError("project root not found")


ROOT = _root()
sys.path[:0] = [p for p in (ROOT, os.path.join(ROOT, "main", "REPLICATION", "claude_plots")) if p not in sys.path]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from main.utils import apply_fig_style, fig_path                                  # noqa: E402

FS = apply_fig_style()
plt.rcParams["mathtext.fontset"] = "cm"
DATA = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression", "figdata", "round4b", "pr_pooled.npz")
AG = ("Joint", "Trained", "Naive", "Echo")
AGENT = {"Joint": ("Optimal", "k"), "Trained": ("Trained RNN", "#1f77b4"), "Naive": ("Naive", "0.5"),
         "Echo": ("ESN", "#ff7f0e")}
STY = {"Trained": dict(lw=3.0, alpha=.45, zorder=3), "Joint": dict(lw=1.1, zorder=5), "Naive": dict(lw=1.5, zorder=4),
       "Echo": dict(lw=1.5, zorder=4)}
NB = 1000


def _pr(s1, s2, n):
    """PR and tr C from sums: s1 (..., 20) = sum x, s2 (..., 20, 20) = sum x x^T, n (...) = counts"""
    mu = s1 / n[..., None]
    C = s2 / n[..., None, None] - mu[..., :, None] * mu[..., None, :]
    tr = np.einsum("...ii->...", C)
    return tr ** 2 / (C ** 2).sum((-1, -2)), tr


def build():
    import fig1_movie as cm
    import fig1 as cf
    BV, rg, rc = cm.beliefs_every_step()                                           # {agent: (N_EP, T, 20)}
    n_ep, T = BV["Joint"].shape[:2]
    ci = np.sort(cf._harvest("trained", ["ep_ctx_inds"])["ep_ctx_inds"][:n_ep].astype(int), 1)
    keep = ci[:, 0] != ci[:, 1]                                                    # no same-variable environments
    env = np.unique(ci[keep], axis=0, return_inverse=True)[1].ravel()
    n_env = env.max() + 1
    cnt = np.bincount(env, minlength=n_env).astype(np.float64)                     # episodes per environment
    from scipy.sparse import csr_matrix
    Et = csr_matrix((np.ones(len(env)), (env, np.arange(len(env)))), shape=(n_env, len(env)))   # sums per env
    rng = np.random.default_rng(0)
    W = rng.multinomial(n_env, np.full(n_env, 1 / n_env), size=NB).astype(np.float64)   # (NB, n_env) multiplicities
    out = dict(n_episodes=int(keep.sum()), n_env=int(n_env), agents=np.array(AG), coords=np.array(["prob", "sqrt"]))
    for name in ("pr", "tr"):
        out[name] = np.zeros((len(AG), 2, 2, T))                                   # agent, coords, (B, U), step
        out[name + "_lo"], out[name + "_hi"] = np.zeros_like(out[name]), np.zeros_like(out[name])
        out[name + "_all"] = np.zeros((len(AG), 2, 2))                             # pooled over steps as well
        out[name + "_all_lo"], out[name + "_all_hi"] = np.zeros_like(out[name + "_all"]), np.zeros_like(out[name + "_all"])
    for a_, a in enumerate(AG):
        Bv = np.asarray(BV[a][keep], np.float64)
        Bv = Bv.reshape(len(Bv), T, 2, 10)
        Bv = (Bv / Bv.sum(-1, keepdims=True)).reshape(len(Bv), T, 20)
        for c_, f in enumerate((lambda x: x, np.sqrt)):
            X = f(Bv)
            for b_, Y in enumerate((X, np.diff(X, axis=1, prepend=f(np.full((len(X), 1, 20), .1))))):
                A1, A2 = np.zeros((n_env, 20)), np.zeros((n_env, 20, 20))                # sums over all steps
                for s in range(T):
                    x = Y[:, s]
                    s1 = Et @ x
                    s2 = (Et @ (x[:, :, None] * x[:, None, :]).reshape(len(x), -1)).reshape(n_env, 20, 20)
                    pr, tr = _pr(s1.sum(0), s2.sum(0), np.array(cnt.sum()))
                    prb, trb = _pr(W @ s1, (W @ s2.reshape(n_env, -1)).reshape(-1, 20, 20), W @ cnt)
                    for k, v, vb in (("pr", pr, prb), ("tr", tr, trb)):
                        out[k][a_, c_, b_, s] = v
                        out[k + "_lo"][a_, c_, b_, s], out[k + "_hi"][a_, c_, b_, s] = np.percentile(vb, [2.5, 97.5])
                    A1 += s1
                    A2 += s2
                pr, tr = _pr(A1.sum(0), A2.sum(0), np.array(cnt.sum() * T))
                prb, trb = _pr(W @ A1, (W @ A2.reshape(n_env, -1)).reshape(-1, 20, 20), (W @ cnt) * T)
                for k, v, vb in (("pr", pr, prb), ("tr", tr, trb)):
                    out[k + "_all"][a_, c_, b_] = v
                    out[k + "_all_lo"][a_, c_, b_], out[k + "_all_hi"][a_, c_, b_] = np.percentile(vb, [2.5, 97.5])
        print(a, "done", flush=True)
    np.savez(DATA, **out)
    print("saved", DATA)


def figure():
    Z = np.load(DATA)
    ag = list(Z["agents"])
    T = np.arange(1, Z["pr"].shape[-1] + 1)
    W, H = 3 * 2.9 + 2 * .95 + .9, 2.0 + .95 + .95 + .3
    fig = plt.figure(figsize=(W, H))
    axs = [fig.add_axes([(.8 + k * (2.9 + .95)) / W, (H - .95 - 2.0) / H, 2.9 / W, 2.0 / H]) for k in range(3)]
    panels = (("pr", 0, "Participation ratio", r"$PR = (\mathrm{tr}\,C)^2 / \mathrm{tr}(C^2)$, $C$ = covariance over all "
               r"environments' particles", "Effective dimensions"),
              ("tr", 0, "Total variance", r"$\mathrm{tr}\,C$ over all environments' particles", r"Total variance (prob.$^2$)"),
              ("pr", 1, "Participation ratio, sqrt coordinates", r"PR of $\sqrt{B_t}$ and of $\sqrt{B_t} - \sqrt{B_{t-1}}$",
               "Effective dimensions"))
    for k, (q, c_, name, formula, ylab) in enumerate(panels):
        ax = axs[k]
        for a in ("Trained", "Joint", "Naive", "Echo"):
            for b_, ls in ((0, "-"), (1, "--")):
                i = ag.index(a)
                ax.plot(T, Z[q][i, c_, b_], c=AGENT[a][1], ls=ls, **{**STY[a], "lw": STY[a]["lw"] * (1 if b_ == 0 else .9)})
                ax.fill_between(T, Z[q + "_lo"][i, c_, b_], Z[q + "_hi"][i, c_, b_], color=AGENT[a][1], alpha=.15, lw=0)
        if q == "tr":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 18.5), ax.set_yticks([0, 6, 12, 18])
        ax.set_xlim(.5, 30.5), ax.set_xticks([1, 10, 20, 30])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Step $t$", labelpad=1)
        ax.set_ylabel(ylab)
        ax.annotate(name, (.5, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points", ha="center",
                    va="baseline", fontsize=FS["label"] + .5)
        ax.annotate(formula, (.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points", ha="center",
                    va="baseline", fontsize=FS["tick"], color="0.25")
        x0, y0, w, h = ax.get_position().bounds
        fig.text(x0 - .62 / W, y0 + h + 17 / 72 / H, r"$\mathbf{%s}$" % "abc"[k], fontsize=FS["label"] + 2, va="baseline")
    hand = [Line2D([], [], c=AGENT[a][1], label=AGENT[a][0], **{k: v for k, v in STY[a].items() if k != "zorder"})
            for a in ("Joint", "Trained", "Naive", "Echo")]
    hand += [Line2D([], [], c="0.3", lw=1.4, label="beliefs $B_t$"),
             Line2D([], [], c="0.3", lw=1.4, ls="--", label="updates $U_t = B_t - B_{t-1}$")]
    fig.legend(handles=hand, loc="upper center", bbox_to_anchor=(.5, 1 - .06 / H), ncol=6, frameon=False,
               fontsize=FS["tick"], handlelength=1.8, columnspacing=1.4)
    fig.text(.8 / W, .42 / H, f"Particles: {int(Z['n_episodes']):,} episodes (one stream each) from {int(Z['n_env']):,} "
             "environments of the held-out variables (same-variable pairs left out), all four agents on the same episodes;"
             " covariance over all of them pooled.\nA belief is 20 numbers (at most 18 dimensions). $B_0$ = uniform prior."
             " Bands: 95% bootstrap over environments (clusters).", fontsize=FS["tick"] - 1.5, color="0.35", va="top",
             linespacing=1.3)
    path = fig_path("fig1c_participation_ratio.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def figure_bars():
    """the participation ratio with the covariance over all environments' particles AND all 30 steps pooled: one
    number per agent, beliefs and updates, in probability and in square-root coordinates -> fig1c_pr_bars.png"""
    Z = np.load(DATA)
    ag = list(Z["agents"])
    order = ("Joint", "Trained", "Naive", "Echo")
    W, H = 2 * 2.9 + .95 + .9 + .3, 2.0 + .75 + .85
    fig = plt.figure(figsize=(W, H))
    for c_, (name, formula) in enumerate((("Participation ratio, pooled over steps",
                                           r"$C$ over all environments' particles and all 30 steps"),
                                          ("The same in sqrt coordinates",
                                           r"PR of $\sqrt{B_t}$ and of $\sqrt{B_t} - \sqrt{B_{t-1}}$, pooled"))):
        ax = fig.add_axes([(.8 + c_ * (2.9 + .95)) / W, .75 / H, 2.9 / W, 2.0 / H])
        for k, a in enumerate(order):
            i = ag.index(a)
            for b_, (dx, lab) in enumerate(((-.19, "beliefs"), (.19, "updates"))):
                v = Z["pr_all"][i, c_, b_]
                lo, hi = Z["pr_all_lo"][i, c_, b_], Z["pr_all_hi"][i, c_, b_]
                ax.bar(k + dx, v, width=.36, color=AGENT[a][1], alpha=1 if b_ == 0 else .45, lw=0,
                       hatch=None if b_ == 0 else "////", edgecolor="white")
                ax.errorbar(k + dx, v, yerr=[[v - lo], [hi - v]], color="0.2", lw=.8, capsize=2)
                ax.text(k + dx, hi + .3, "%.1f" % v, ha="center", va="bottom", fontsize=FS["tick"] - 2, color="0.25")
        ax.set_xticks(range(4), [AGENT[a][0] for a in order])
        ax.set_ylim(0, 19.5), ax.set_yticks([0, 6, 12, 18])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Effective dimensions")
        ax.annotate(name, (.5, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points", ha="center",
                    va="baseline", fontsize=FS["label"] + .5)
        ax.annotate(formula, (.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points", ha="center",
                    va="baseline", fontsize=FS["tick"], color="0.25")
        x0, y0, w, h = ax.get_position().bounds
        fig.text(x0 - .62 / W, y0 + h + 17 / 72 / H, r"$\mathbf{%s}$" % "ab"[c_], fontsize=FS["label"] + 2, va="baseline")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="0.4", label="beliefs $B_t$"),
                        Patch(facecolor="0.4", alpha=.45, hatch="////", edgecolor="white", label="updates $U_t$")],
               loc="upper center", bbox_to_anchor=(.5, 1 - .04 / H), ncol=2, frameon=False, fontsize=FS["tick"])
    fig.text(.8 / W, .2 / H, f"{int(Z['n_episodes']):,} episodes from {int(Z['n_env']):,} environments x 30 steps, one "
             "covariance per agent; error bars: 95% bootstrap over environments", fontsize=FS["tick"] - 1.5,
             color="0.35", va="top")
    path = fig_path("fig1c_pr_bars.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if "--rebuild" in sys.argv[1:] or not os.path.exists(DATA) or "pr_all" not in np.load(DATA).files:
        build()
    print("saved", figure())
    print("saved", figure_bars())
