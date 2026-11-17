"""fig_learn_grids_pr.png: fig_learn_grids' belief grids coloured by the state's effective dimension next to the same grids
coloured by Factorization Regret (John 11-13: "can you tell me if theres a within episode correlation between FR and
effective dim throughout training?", then "like for our grid through training figure, if we color the Scatterpoints by
PR, does it match FR?").

Rows: Trained RNN by FR, by effective dimension; ESN the same. Each point is one realization r = (r_1, r_2), placed as
in fig_learn_grids (the network's posterior means at T = 30, 50 test environments x 24 trajectories per realization).
FR rows: its colours (FR_T, the task's, so the same in every column). Effective-dimension rows: ck 'effdim_u' at T = 30
(td_analysis: the projection PR (Sigma_k p_k^2)^2 / Sigma_k p_k^4 of one state, p_k its deviation from the mean on the
checkpoint's k-th PC), mean over the realization's 100 probe trajectories (50 environments x 2), one log scale for both
networks (no per-column scaling). Bottom: Spearman rho over the 100 realizations between effective dimension and FR
(the probes' FR_T, both variables averaged) at all 114 checkpoints; dashed: the partial rho given the realization's
distance from the grid centre max(|r_1 - 4.5|, |r_2 - 4.5|), which sets FR (rho .96; border 5.9 vs interior 1.4 nats).
RESULTS (11-13): Trained rho -.21 (100) / -.59 (300) / -.54 (800) / -.78 (2,000) / -.81 (5,000 and 40k): effective
dimension LOWEST where FR is highest; given the distance ~0 until 800 (+.01 / +.09), -.27 to -.31 from 2,000; inside
the border alone -.2 to -.27 late; it also falls with the state's size (rho with |u_T| -.78 to -.87). ESN: its effective
dimension varies little across realizations (9.8-11.6 at 40k) and rises with FR (+.22 -> +.61), +.1 to +.27 given the
distance. Within episodes (John's first question; not drawn): per probe, Pearson r over the 30 steps of effdim_u and
FR_t, then the mean over probes: Trained -.14 to +.05, ESN +.33 to +.35, each equal to the r with ANOTHER episode's FR
(the shared time course only); with each step's mean over probes removed from both: Trained -.05 to 0, ESN 0 to +.05
(300 / 800 / 2,000 / 5,000 / 40k training episodes). usage:  python fig_learn_grids_pr.py"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402
from matplotlib.colors import LogNorm                                             # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402

AN = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics", "analysis")
R = 10
NETS = (("trained", "Trained", "#1f77b4"), ("echo", "Echo", "#ff7f0e"))
DIM_LABEL = "effective dim."
DIM_FORM = r"$(\Sigma_k p_k^2)^2 / \Sigma_k p_k^4$"


def cell_means(v, cell):
    return np.bincount(cell, v, R * R) / np.bincount(cell, minlength=R * R)


def partial_rho(x, y, z):
    """Spearman rho of x and y with z's rank removed linearly from both ranks"""
    from scipy.stats import rankdata
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)               # noqa: E731
    return np.corrcoef(res(rx), res(ry))[0, 1]


def dims():
    """{run: (training episodes (114,), effective dimension per realization (114, 100))} and the probes' FR_T per
    realization (100,)"""
    import td_tt_intervene as ti
    import td_tt_gap as gap
    P = np.load(os.path.join(AN, "probes.npz"))
    cell = P["r"][:, 0] * R + P["r"][:, 1]
    out = {}
    for run, _, _ in NETS:
        raws = np.array(ti.checkpoints(run))
        out[run] = (gap.train_x(raws), np.stack([cell_means(np.load(os.path.join(AN, run, f"ck_{raw:05d}.npz"))[
            "effdim_u"][:, -1].astype(np.float64), cell) for raw in raws]))
    return out, cell_means(P["fr"][:, -1].astype(np.float64).mean(-1), cell)


def figure():
    import fig_learn_grids as flg
    import plot_fig1c_mesh as pm
    import fig_td_summary as fts
    from scipy.stats import spearmanr
    MG, stages, fr_mesh = flg.data()
    stages = tuple(x for x in stages if x not in (200, 400))                      # the paper figure's columns
    D, fr = dims()
    apply_fig_style()
    plt.rcParams.update({"mathtext.fontset": "cm"})
    nc = len(stages)
    W, pw, gp, lm, rbar = 9.6, 1.0, .08, .95, 1.15                               # inches
    name_h, rg, line_h = .30, .10, 2.3
    H = 4 * (pw + rg) + name_h * 2 + .35 + line_h + .55
    fig = plt.figure(figsize=(W, H))
    fnorm, cmap_fr = LogNorm(np.floor(fr_mesh.min() * 10) / 10, np.ceil(fr_mesh.max())), pm.FR_CMAP
    lo = min(D[r][1][[list(D[r][0]).index(x) for x in stages]].min() for r, _, _ in NETS)
    hi = max(D[r][1][[list(D[r][0]).index(x) for x in stages]].max() for r, _, _ in NETS)
    dnorm = LogNorm(np.floor(lo), np.ceil(hi))
    idx = {(g, c): g * R + c for g in range(R) for c in range(R)}
    edges = np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                      if (g + dg, c + dc) in idx])
    cells = np.array([(g, c) for g in range(R) for c in range(R)], float)

    def grid(x0, top, pos, values, cmap, norm):
        ax = fig.add_axes([x0 / W, 1 - (top + pw) / H, pw / W, pw / H])
        ax.add_collection(LineCollection(cells[edges], colors="0.86", linewidths=.3, linestyles=(0, (3, 2))))
        Pm = pos.mean(0)                                                          # mean over the 50 environments
        ax.add_collection(LineCollection(Pm[edges], colors="0.7", linewidths=.35, zorder=2))
        ax.scatter(Pm[:, 0], Pm[:, 1], c=values, cmap=cmap, norm=norm, s=7, lw=.2, edgecolors="0.3", zorder=3)
        ax.set_xlim(-.7, 9.7), ax.set_ylim(-.7, 9.7), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color("0.7")
            s_.set_linewidth(.4)
        return ax

    xs = [lm + j * (pw + gp) for j in range(nc)]
    top = name_h
    bars = []
    for run, name, col in NETS:
        x_, dm = D[run]
        fig.text((xs[0] + xs[-1] + pw) / 2 / W, 1 - (top - .07) / H, name, ha="center", va="baseline", fontsize=11,
                 color=col)
        for k, (lab, cmap, norm) in enumerate((("FR", cmap_fr, fnorm), (DIM_LABEL, "viridis", dnorm))):
            for j, xv in enumerate(stages):
                vals = fr_mesh if k == 0 else dm[list(x_).index(xv)]
                grid(xs[j], top, MG[run][xv][0], vals, cmap, norm)
            fig.text((xs[0] - .1) / W, 1 - (top + pw / 2) / H, "factorization\nregret" if k == 0 else
                     "effective\ndimension", ha="right", va="center", fontsize=9, color="0.2")
            bars.append((top, cmap, norm, k))
            top += pw + rg
        top += name_h
    for j, xv in enumerate(stages):                                               # training episodes, below the grids
        fig.text((xs[j] + pw / 2) / W, 1 - (top - name_h + .02) / H, f"{xv:,}", ha="center", va="top", fontsize=8.5)
    fig.text((xs[0] + xs[-1] + pw) / 2 / W, 1 - (top - name_h + .2) / H, "Training episodes", ha="center", va="top",
             fontsize=9)
    for (y0, cmap, norm, k), lab, ticks in zip(bars[:2], ("FR (nats)", "effective dim.\n(both networks)"),
                                               ((.5, 1, 3, 10), (3, 10, 30))):    # beside the Trained rows
        cx = fig.add_axes([(xs[-1] + pw + .15) / W, 1 - (y0 + pw) / H, .07 / W, pw / H])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cx)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:g}" for t in ticks])
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=8, length=2, width=.5, pad=1.5)
        cb.outline.set_visible(False)
        cb.set_label(lab, fontsize=8.5, labelpad=3)
    ax = fig.add_axes([(lm + .45) / W, .55 / H, (xs[-1] + pw - lm - .45) / W, (line_h - .45) / H])
    g, c = np.divmod(np.arange(R * R), R)
    dist = np.maximum(np.abs(g - 4.5), np.abs(c - 4.5))
    for run, name, col in NETS:
        x_, dm = D[run]
        ax.plot(x_, [spearmanr(v, fr)[0] for v in dm], c=col, lw=2, label=name, zorder=3)
        ax.plot(x_, [partial_rho(v, fr, dist) for v in dm], c=col, lw=1.6, ls=(0, (3, 1.6)), zorder=3)
    ax.axhline(0, c="k", alpha=.5, lw=.8)
    fts.xaxis(ax, label=False)
    ax.set_xlim(fts.XMIN, 4.6e4)
    ax.set_xlabel("Training episode", fontsize=9)
    ax.set_ylim(-1, 1)
    ax.set_ylabel(r"Spearman $\rho$ over realizations", fontsize=9)
    ax.set_title("Effective dimension vs FR across realizations (T = 30)", fontsize=10.5, pad=20)
    ax.annotate(f"effective dim. = {DIM_FORM}, $p_k$: $u_T - \\bar u$ on PC $k$;   solid: Spearman $\\rho$, dashed: "
                "given the distance from the grid centre", xy=(.5, 1), xycoords="axes fraction", xytext=(0, 6),
                textcoords="offset points", ha="center", va="baseline", fontsize=8.5, color="0.15")
    ax.legend(loc="lower left", frameon=False, fontsize=9, ncols=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, ls="--", alpha=.3)
    fig.text(.995, .005, "grids and FR colours: 50 environments x 24 trajectories per realization; effective dimension "
             "and line panel: 10,000 probes (50 x 2 per realization), 114 checkpoints", ha="right", va="bottom",
             fontsize=8, color="0.4")
    path = fig_path("fig_learn_grids_pr.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)
    for run, name, _ in NETS:
        x_, dm = D[run]
        print(name, " / ".join(f"{xv:,}: {spearmanr(dm[list(x_).index(xv)], fr)[0]:+.2f}" for xv in stages))


if __name__ == "__main__":
    figure()
