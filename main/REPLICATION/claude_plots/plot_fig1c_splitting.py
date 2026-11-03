"""Do nearby trajectories split at entangled steps? (25 Oct 2026) -> main/REPLICATION/figure_bin/fig1c_splitting.png

Data: the alias scan (main/DATA/experimental/fig1c_compression; 39 environments, e = 2 left out; 100 cells x 50 streams x
30 steps; all four agents on the same streams). Position of a particle = its posterior means x = (E[r_g], E[r_c]).
Near pairs: at step t0 (2..24), each particle i and its nearest particle j of a DIFFERENT cell, in the agent's own
positions (two trajectories that are close but should end apart).
Entanglement of the pair's next step: e_ij = (rev_i + rev_j) / 2 at step t0 + 1 (house entanglement, scan_meta).
Within each (environment, step) the pairs are split into thirds of e_ij; high = top third, low = bottom third.
Measures, k = 0..6 steps after t0:
  split          d(t0 + k) - d(t0), d = ||x_i - x_j||   (do the two trajectories move apart?)
  correct split  D(t0 + k) - D(t0), D = [||x_i - c_j|| - ||x_i - c_i||] + [||x_j - c_i|| - ||x_j - c_j||], c the true
                 cells   (does each particle move toward its own cell rather than the partner's?)
Means over pairs and steps per environment, then over environments; bands: 95% bootstrap over environments.
Null: e_ij shuffled within (environment, step). Data cached in figdata/round4b/splitting.npz (--rebuild; about 1 min).
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_splitting.py [--rebuild]"""
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
SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
DATA = os.path.join(SCAN, "figdata", "round4b", "splitting.npz")
AG = ("Joint", "Trained", "Naive", "Echo")
AGENT = {"Joint": ("Optimal", "k"), "Trained": ("Trained RNN", "#1f77b4"), "Naive": ("Naive", "0.5"),
         "Echo": ("ESN", "#ff7f0e")}
STY = {"Trained": dict(lw=3.0, alpha=.45, zorder=3), "Joint": dict(lw=1.1, zorder=5), "Naive": dict(lw=1.5, zorder=4),
       "Echo": dict(lw=1.5, zorder=4)}
T0, K = np.arange(2, 25), 6                                                       # steps t0 (1-based), lags 0..K
ND = 10                                                                           # entanglement deciles (panel c)


def build():
    from scipy.spatial import cKDTree
    m = np.load(os.path.join(SCAN, "scan_meta.npz"))
    cells, cell_of, rev = m["cells"], m["cell_of"], m["rev"]
    envs = [e for e in range(40) if e != 2]
    c_pos = cells[cell_of].astype(np.float64)                                      # (5000, 2) true (r_g, r_c)
    R = np.arange(10)
    rng = np.random.default_rng(0)
    out = {k: np.full((len(AG), len(envs), K + 1), np.nan) for k in
           ("split_hi", "split_lo", "corr_hi", "corr_lo", "corr_hi_null", "corr_lo_null")}
    out["corr_dec"] = np.full((len(AG), len(envs), ND), np.nan)                    # one-step correct split by decile
    out["split_dec"] = np.full((len(AG), len(envs), ND), np.nan)
    for a_, a in enumerate(AG):
        B = np.load(os.path.join(SCAN, f"scan_{a}.npz"))["beliefs"]
        for k_, e in enumerate(envs):
            b = B[e].astype(np.float64).reshape(5000, 30, 2, 10)
            X = (b * R).sum(-1) / b.sum(-1)                                        # (5000, 30, 2) positions
            acc = {k: [] for k in out}
            for t0 in T0:
                s0 = t0 - 1
                tree = cKDTree(X[:, s0])
                _, nb = tree.query(X[:, s0], k=120)
                other = cell_of[nb] != cell_of[:, None]
                has = other.any(1)
                i = np.flatnonzero(has)
                j = nb[i, other[i].argmax(1)]                                      # nearest of a different cell
                ent = (rev[e, i, s0 + 1] + rev[e, j, s0 + 1]) / 2                  # entanglement of step t0 + 1
                xi, xj = X[i, s0:s0 + K + 1], X[j, s0:s0 + K + 1]                  # (n, K + 1, 2)
                d = np.linalg.norm(xi - xj, axis=-1)
                ci, cj = c_pos[i][:, None], c_pos[j][:, None]
                D = (np.linalg.norm(xi - cj, axis=-1) - np.linalg.norm(xi - ci, axis=-1) +
                     np.linalg.norm(xj - ci, axis=-1) - np.linalg.norm(xj - cj, axis=-1))
                dd, dD = d - d[:, :1], D - D[:, :1]
                lo_, hi_ = np.quantile(ent, [1 / 3, 2 / 3])
                hi, lo = ent >= hi_, ent <= lo_
                acc["split_hi"].append(dd[hi].mean(0)), acc["split_lo"].append(dd[lo].mean(0))
                acc["corr_hi"].append(dD[hi].mean(0)), acc["corr_lo"].append(dD[lo].mean(0))
                sh = rng.permutation(ent)                                          # null: shuffled entanglement
                acc["corr_hi_null"].append(dD[sh >= hi_].mean(0)), acc["corr_lo_null"].append(dD[sh <= lo_].mean(0))
                dec = np.minimum((np.argsort(np.argsort(ent)) * ND) // len(ent), ND - 1)
                acc["corr_dec"].append([dD[dec == q, 1].mean() for q in range(ND)])
                acc["split_dec"].append([dd[dec == q, 1].mean() for q in range(ND)])
            for k in out:
                out[k][a_, k_] = np.mean(acc[k], 0)
        print(a, "done", flush=True)
    np.savez(DATA, agents=np.array(AG), envs=np.array(envs), t0=T0, K=K, **out)
    print("saved", DATA)


def band(X, nb=4000):
    X = np.asarray(X, np.float64)
    idx = np.random.default_rng(0).integers(0, len(X), (nb, len(X)))
    bs = np.nanmean(X[idx], 1)
    return np.nanmean(X, 0), np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)


def figure():
    Z = np.load(DATA)
    ag = list(Z["agents"])
    k = np.arange(int(Z["K"]) + 1)
    W, H = 3 * 2.9 + 2 * .95 + .9, 2.0 + .95 + .95 + .3
    fig = plt.figure(figsize=(W, H))
    axs = [fig.add_axes([(.8 + q * (2.9 + .95)) / W, (H - .95 - 2.0) / H, 2.9 / W, 2.0 / H]) for q in range(3)]
    for p_, (key, name, formula) in enumerate((
            ("corr", "Correct splitting after the step", r"$\Delta D$, $D = \Sigma\,(\|x - c_{\mathrm{partner}}\| - "
             r"\|x - c_{\mathrm{own}}\|)$ of the pair"),
            ("split", "Splitting apart after the step", r"$\Delta d$, $d = \|x_i - x_j\|$ of the pair"))):
        ax = axs[p_]
        for a in ("Trained", "Joint", "Naive", "Echo"):
            for g, ls in (("hi", "-"), ("lo", "--")):
                m_, lo, hi = band(Z[f"{key}_{g}"][ag.index(a)])
                ax.plot(k, m_, c=AGENT[a][1], ls=ls, marker="o", ms=2.5, **STY[a])
                if g == "hi":
                    ax.fill_between(k, lo, hi, color=AGENT[a][1], alpha=.15, lw=0)
        if key == "corr":
            m0 = np.nanmean([band(Z["corr_hi_null"][ag.index(a)])[0] - band(Z["corr_lo_null"][ag.index(a)])[0]
                             for a in AG], 0)
        ax.axhline(0, c="k", lw=.5, alpha=.4)
        ax.set_xticks(k), ax.set_xlim(-.2, k[-1] + .2)
        ax.set_xlabel("Steps after $t_0$ (the step's observation at $k = 1$)", labelpad=1)
        ax.set_ylabel("Grid units")
        _deco(fig, ax, name, formula, "ab"[p_], W, H)
    ax = axs[2]
    x = np.arange(1, 11)
    for a in ("Trained", "Joint", "Naive", "Echo"):
        m_, lo, hi = band(Z["corr_dec"][ag.index(a)])
        ax.plot(x, m_, c=AGENT[a][1], marker="o", ms=2.5, **STY[a])
        ax.fill_between(x, lo, hi, color=AGENT[a][1], alpha=.15, lw=0)
    ax.axhline(0, c="k", lw=.5, alpha=.4)
    ax.set_xticks([1, 4, 7, 10]), ax.set_xlim(.6, 10.4)
    ax.set_xlabel("Entanglement of the next step (decile)", labelpad=1)
    ax.set_ylabel("Grid units per step")
    _deco(fig, ax, "Correct splitting vs entanglement", r"one-step $\Delta D$ by the pair's $\bar e_{t_0 + 1}$", "c", W, H)
    hand = [Line2D([], [], c=AGENT[a][1], label=AGENT[a][0], **{q: v for q, v in STY[a].items() if q != "zorder"})
            for a in ("Joint", "Trained", "Naive", "Echo")]
    hand += [Line2D([], [], c="0.3", lw=1.4, label="entangled step (top third)"),
             Line2D([], [], c="0.3", lw=1.4, ls="--", label="calm step (bottom third)")]
    fig.legend(handles=hand, loc="upper center", bbox_to_anchor=(.5, 1 - .06 / H), ncol=6, frameon=False,
               fontsize=FS["tick"], handlelength=1.8, columnspacing=1.4)
    fig.text(.8 / W, .42 / H, "Near pairs: at each step $t_0 = 2..24$, each particle and its nearest particle of a "
             "different cell in the agent's own positions $x = (E[r_g], E[r_c])$ (same environment). Entanglement of the "
             "pair's next step: the mean of the two streams' $e_{t_0+1}$;\nthirds and deciles within (environment, step). "
             "39 scan environments x 5000 streams; bands: 95%% bootstrap over environments. With shuffled entanglement the "
             "entangled-minus-calm difference is %.3f grid units at $k = 6$." % m0[-1],
             fontsize=FS["tick"] - 1.5, color="0.35", va="top", linespacing=1.3)
    path = fig_path("fig1c_splitting.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _deco(fig, ax, name, formula, letter, W, H):
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate(name, (.5, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points", ha="center",
                va="baseline", fontsize=FS["label"] + .5)
    ax.annotate(formula, (.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points", ha="center",
                va="baseline", fontsize=FS["tick"], color="0.25")
    x0, y0, w, h = ax.get_position().bounds
    fig.text(x0 - .62 / W, y0 + h + 17 / 72 / H, r"$\mathbf{%s}$" % letter, fontsize=FS["label"] + 2, va="baseline")


if __name__ == "__main__":
    if "--rebuild" in sys.argv[1:] or not os.path.exists(DATA):
        build()
    print("saved", figure())
    Z = np.load(DATA)
    for i, a in enumerate(Z["agents"]):
        print("%-8s correct split at k = 1 / 6: entangled %.3f / %.3f, calm %.3f / %.3f | split at k = 6: entangled %.3f, "
              "calm %.3f | one-step correct split, deciles 1 -> 10: %.3f -> %.3f" % (
                  a, np.nanmean(Z["corr_hi"][i, :, 1]), np.nanmean(Z["corr_hi"][i, :, 6]), np.nanmean(Z["corr_lo"][i, :, 1]),
                  np.nanmean(Z["corr_lo"][i, :, 6]), np.nanmean(Z["split_hi"][i, :, 6]), np.nanmean(Z["split_lo"][i, :, 6]),
                  np.nanmean(Z["corr_dec"][i, :, 0]), np.nanmean(Z["corr_dec"][i, :, -1])))
