"""The networks' hidden streams next to the Bayesian beliefs and their derivatives, by participation ratio (25 Oct 2026)
-> main/REPLICATION/figure_bin/fig1c_hidden_vs_bayes.png

Data: figdata/round4b/hidden_pr.npz (plot_fig1c_hidden_pr.py) and hidden_bayes_prs.npz (hidden_bayes_prs.py): per step,
covariance over the 19,607 kept episodes of 1,225 environments (pooled), PR = (tr C)^2 / tr(C^2).
Top row, the output stream u_t = o_t * tanh(c_t) and the belief's derivatives: the networks' read-out is applied every
step and summed, x_t = sum_s CLR(W u_s + b), so the belief's first derivative dx_t = x_t - x_{t-1} is a linear read of
u_t, and the second derivative d2x_t = dx_t - dx_{t-1} a read of u_t - u_{t-1}. (c): the share of u's (and du's)
variance in the 18 directions the read-out reads (centred rows of W).
Bottom row, the cell (memory) stream c_t and the beliefs: marginal beliefs x_t (20, CLR) and the joint belief X_t (100,
CLR over the cells: the Optimal's joint log-posterior; for the Naive, its implied joint log m_g + log m_c), each with
its update. Belief panels: log-probabilities floored at FLOOR nats before centring (where the Trained RNN's log-beliefs
match the Optimal's; without a floor these PRs stay between 2.4 and 5.4 over all steps).
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/plot_fig1c_hidden_vs_bayes.py"""
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
sys.path[:0] = [p for p in (ROOT,) if p not in sys.path]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from main.utils import apply_fig_style, fig_path                                  # noqa: E402

FS = apply_fig_style()
plt.rcParams["mathtext.fontset"] = "cm"
FIG = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression", "figdata", "round4b")
FLOOR = "10"
COLOR = {"Joint": "k", "Trained": "#1f77b4", "Naive": "0.5", "Echo": "#ff7f0e", "trained": "#1f77b4", "echo": "#ff7f0e"}
NAME = {"Joint": "Optimal", "Trained": "Trained RNN (read-out)", "Naive": "Naive", "Echo": "ESN (read-out)",
        "trained": "Trained RNN", "echo": "ESN"}
T = 30


def figure():
    H_ = np.load(os.path.join(FIG, "hidden_pr.npz"))
    Z = np.load(os.path.join(FIG, "hidden_bayes_prs.npz"))
    t = np.arange(1, T + 1)
    pw, ph, gx, gy = 2.9, 2.0, .95, 1.05
    W, H = .8 + 3 * pw + 2 * gx + .3, .95 + 2 * ph + gy + .95
    fig = plt.figure(figsize=(W, H))

    def panel(r, c, name, formula, letter):
        ax = fig.add_axes([(.8 + c * (pw + gx)) / W, (H - .95 - ph - r * (ph + gy)) / H, pw / W, ph / H])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(.5, T + .5), ax.set_xticks([1, 10, 20, 30])
        ax.set_xlabel("Step $t$", labelpad=1)
        ax.annotate(name, (.5, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points", ha="center",
                    va="baseline", fontsize=FS["label"] + .5)
        ax.annotate(formula, (.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                    ha="center", va="baseline", fontsize=FS["tick"] - .5, color="0.25")
        x0, y0, w, h = ax.get_position().bounds
        fig.text(x0 - .62 / W, y0 + h + 17 / 72 / H, r"$\mathbf{%s}$" % letter, fontsize=FS["label"] + 2, va="baseline")
        return ax

    # (a) u and du
    ax = panel(0, 0, "Output stream $u_t$ (hidden)", r"$u_t = o_t \odot \tanh(c_t)$: state (solid), update (dashed)", "a")
    for net in ("trained", "echo"):
        ax.plot(t, H_[f"{net}_u"], c=COLOR[net], lw=1.4)
        ax.plot(t, H_[f"{net}_du"], c=COLOR[net], lw=1.4, ls="--")
    ax.set_ylim(0, None), ax.set_ylabel("Effective dimensions (PR)")
    # (b) dx and d2x
    ax = panel(0, 1, "Belief derivatives", r"$dx_t$ (solid; networks: $\mathrm{CLR}(W u_t + b)$), $d^2x_t$ (dashed)", "b")
    for a in ("Joint", "Trained", "Naive", "Echo"):
        lw, al = (2.6, .45) if a == "Trained" else (1.2, 1)
        ax.plot(t, Z[f"{a}_{FLOOR}_dx"], c=COLOR[a], lw=lw, alpha=al)
        ax.plot(t, Z[f"{a}_{FLOOR}_d2x"], c=COLOR[a], lw=lw, alpha=al, ls="--")
    ax.set_ylim(0, None), ax.set_ylabel("Effective dimensions (PR)")
    # (c) the read-out's share of u
    ax = panel(0, 2, "What the read-out sees of $u_t$", r"share of the variance in the 18 read-out directions of $W$", "c")
    for net in ("trained", "echo"):
        ax.plot(t, 100 * Z[f"{net}_readfrac_u"], c=COLOR[net], lw=1.4)
        ax.plot(t, 100 * Z[f"{net}_readfrac_du"], c=COLOR[net], lw=1.4, ls="--")
    ax.set_ylim(0, 10), ax.set_ylabel("% of variance")
    # (d) c and dc
    ax = panel(1, 0, "Memory stream $c_t$ (cell)", r"$c_t$: state (solid), update (dashed); $\tanh(c_t)$ (dotted)", "d")
    for net in ("trained", "echo"):
        ax.plot(t, H_[f"{net}_c"], c=COLOR[net], lw=1.4)
        ax.plot(t, H_[f"{net}_dc"], c=COLOR[net], lw=1.4, ls="--")
        ax.plot(t, Z[f"{net}_tanhc"], c=COLOR[net], lw=1, ls=":")
    ax.set_ylim(0, None), ax.set_ylabel("Effective dimensions (PR)")
    # (e) marginal beliefs and updates
    ax = panel(1, 1, "Marginal beliefs $x_t$ (20)", r"state (solid), update $dx_t$ (dashed); floor $-%s$ nats" % FLOOR, "e")
    for a in ("Joint", "Trained", "Naive", "Echo"):
        lw, al = (2.6, .45) if a == "Trained" else (1.2, 1)
        ax.plot(t, Z[f"{a}_{FLOOR}_x"], c=COLOR[a], lw=lw, alpha=al)
        ax.plot(t, Z[f"{a}_{FLOOR}_dx"], c=COLOR[a], lw=lw, alpha=al, ls="--")
    ax.set_ylim(0, None), ax.set_ylabel("Effective dimensions (PR)")
    # (f) joint beliefs and updates
    ax = panel(1, 2, "Joint beliefs $X_t$ (100 cells)", r"state (solid), update $dX_t$ (dashed); floor $-%s$ nats" % FLOOR,
               "f")
    for a in ("Joint", "Naive"):
        ax.plot(t, Z[f"{a}joint_{FLOOR}_X"], c=COLOR[a], lw=1.2)
        ax.plot(t, Z[f"{a}joint_{FLOOR}_dX"], c=COLOR[a], lw=1.2, ls="--")
    ax.set_ylim(0, None), ax.set_ylabel("Effective dimensions (PR)")
    hand = [Line2D([], [], c=COLOR[a], lw=2.6 if a == "Trained" else 1.3, alpha=.45 if a == "Trained" else 1,
                   label={"Joint": "Optimal", "Trained": "Trained RNN", "Naive": "Naive (f: its implied joint)",
                          "Echo": "ESN"}[a]) for a in ("Joint", "Trained", "Naive", "Echo")]
    hand += [Line2D([], [], c="0.3", lw=1.3, label="state"), Line2D([], [], c="0.3", lw=1.3, ls="--", label="update")]
    fig.legend(handles=hand, loc="upper center", bbox_to_anchor=(.5, 1 - .04 / H), ncol=6, frameon=False,
               fontsize=FS["tick"], handlelength=1.8, columnspacing=1.4)
    fig.text(.8 / W, .42 / H, "PR = (tr C)$^2$ / tr(C$^2$) of the covariance over episodes at each step, 19,607 episodes "
             "from 1,225 environments, pooled. Belief panels: centred log-probabilities floored at -%s nats; without a "
             "floor they stay between 2.4 and 5.4 for every agent.\nThe networks' read-out is applied each step and summed, so "
             "$dx_t$ is a linear read of $u_t$ (exactly, without a floor) and $d^2x_t$ a read of $u_t - u_{t-1}$." % FLOOR,
             fontsize=FS["tick"] - 1, color="0.35", va="top", linespacing=1.4)
    path = fig_path("fig1c_hidden_vs_bayes.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure())
