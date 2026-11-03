"""Lesion test: does the Trained RNN need its off-read-out hidden dimensions, and its environment-specific ones in
particular, to keep one grid across environments? (25 Oct 2026) -> main/REPLICATION/figure_bin/fig1c_lesion.png

Data: fig1c_lesion_test.py -> main/DATA/experimental/fig1c_compression/figdata/round6_hidden/lesion/. The Trained RNN
replayed on 10 held-out environments of the alias scan (every 4th of the 39; 100 cells x 50 streams each). After
every step its output u_t (what the read-out reads and what drives the next step's gates) is clamped along a subspace
V to the population mean, u_t <- u_t - (u_t - mu_t) V V^T. Every V is orthogonal to the read-out's 18 directions (so
the read-out at the same step is untouched; any effect comes through the dynamics) and to the shared grid's top 10
directions, and comes from the 29 other environments' hidden cell means:
    environment-specific   top-k PCs of the cell x environment interaction (k = 10, 30, 60)
    stream-to-stream       top 30 PCs of the within-cell variability
    whole-grid shift       top 30 PCs of the environment effect
    random                 random subspaces (30, 60, 260, 400 directions)
x axis of b, c: the share of u's stream-level variance (within-cell plus between-cell-mean scatter, training
environments, mean over steps 1, 3, 5, 10, 20, 30) that the subspace holds.
Measures (goal-ordered beliefs): P(true cell) = b_T(r_g) b_T(r_c) per stream, mean over streams; interference share
V_it / (V_ce + V_it) of the cells' mean beliefs across the 10 environments (plot_fig1c_mesh.decompose; not
noise-corrected: at baseline the floor is about 6% of V_it); accuracy over time = the cells' mean belief at the true
values, (b_t(r_g) + b_t(r_c)) / 2.
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_lesion.py"""
import os
import sys
import glob
import json
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402
from matplotlib.colors import LogNorm                                             # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402

SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
LES = os.path.join(SCAN, "figdata", "round6_hidden", "lesion")
K, M, T, R = 100, 50, 30, 10
ENVS = [e for e in range(40) if e != 2]
TEST = [ENVS[p] for p in range(0, 39, 4)]
FAM = {"it": ("environment-specific", "#d7301f", ("it10", "it30", "it60")),
       "w": ("stream-to-stream", "#6a51a3", ("w30",)),
       "ee": ("whole-grid shift", "#238b45", ("ee30",)),
       "rand": ("random", "0.45", ("rand30", "rand60", "rv260", "rv400"))}
SHOWN = (("none", "no lesion", "#1f77b4"), ("it30", "environment-specific\n30 directions", "#d7301f"),
         ("rv260", "random, same variance\n260 directions", "0.45"), ("w30", "stream-to-stream\n30 directions", "#6a51a3"),
         ("ee30", "whole-grid shift\n30 directions", "#238b45"))


def load(net="trained"):
    D = {}
    for e in TEST:
        for fn in sorted(glob.glob(os.path.join(LES, f"{net}_full_env{e:02d}_*.npz"))):
            f = np.load(fn)
            for k in f.files:
                c, q = k.split("__")
                D.setdefault(c, {}).setdefault(q, []).append(f[k])
    D = {c: {q: np.stack(v) for q, v in d.items()} for c, d in D.items() if len(d["mean_q"]) == len(TEST)}
    var = {c: float(np.mean(v)) for c, v in json.load(open(os.path.join(LES, f"{net}_variance_share.json"))).items()}
    var["none"] = 0.
    info = json.load(open(os.path.join(LES, f"{net}_subspace_info.json")))
    return D, var, info


def decompose(X):
    mu = X.mean((0, 1), keepdims=True)
    ce, ee = X.mean(0, keepdims=True) - mu, X.mean(1, keepdims=True) - mu
    it = X - mu - ce - ee
    V = lambda Y: (Y ** 2).sum(-1).mean((0, 1))
    return V(ce), V(ee), V(it)


def measures(D, cells):
    out = {}
    for c, d in D.items():
        q = d["mean_q"]                                                             # (E, K, T, 20)
        vc, _, vi = decompose(q)
        acc_t = (q[:, np.arange(K), :, cells[:, 0]] + q[:, np.arange(K), :, R + cells[:, 1]]) / 2   # (E, K, T)
        out[c] = dict(p_env=d["p_true"].mean(1), p=d["p_true"].mean(), share=vi / (vc + vi), acc_t=acc_t.mean((0, 1)))
    return out


def figure():
    import fig1 as cf
    import fig1_movie as cm
    import plot_fig1c_mesh as pm
    from main.utils import apply_fig_style, fig_path
    D, var, info = load()
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    cells = meta["cells"]
    Ms = measures(D, cells)
    home, _ = cf.targets([tuple(int(v) for v in c) for c in cells])
    frT = np.maximum(meta["FR"][TEST].astype(np.float64).reshape(len(TEST), K, M, T).mean(2)[:, :, -1].mean(0), 1e-6)
    idx = {(int(g), int(c)): k for k, (g, c) in enumerate(cells)}
    edges = np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                      if (g + dg, c + dc) in idx])
    BV, rg, rc = cm.beliefs_every_step()
    S = cm.step_planes(BV["Trained"], rg, rc)
    proj = lambda V, tau: (np.asarray(V, np.float64) - cm._blend(S["m"], tau)) @ cm._polar(cm._blend(S["U"], tau))
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    t = T - 1
    pw, gap = 2.2, .28
    W = .5 + 5 * pw + 4 * gap + .25 + .12 + .9
    H = .75 + pw + .5 + .45 + 2.1 + 1.3
    fig = plt.figure(figsize=(W, H))
    norm = plt.Normalize(0, pm.DIST_MAX)
    hm = proj(home, float(t))
    unit = np.linalg.norm(hm[edges[:, 0]] - hm[edges[:, 1]], axis=1).mean()        # one cell of the targets' grid
    allp = np.vstack([proj(D["none"]["mean_q"][e, :, t], float(t)) for e in range(len(TEST))] + [hm])
    xl, yl = pm._box(allp)
    ytop = H - .75 - pw
    for i, (c, lab, col) in enumerate(SHOWN):
        ax = fig.add_axes([(.5 + i * (pw + gap)) / W, ytop / H, pw / W, pw / H])
        Pt = np.stack([proj(D[c]["mean_q"][e, :, t], float(t)) for e in range(len(TEST))])   # (E, K, 2)
        avg = Pt.mean(0)
        ax.add_collection(LineCollection(hm[edges], colors="0.8", linewidths=.55, linestyles=(0, (3, 2)), zorder=0))
        for e in range(len(TEST)):
            dv = np.linalg.norm(Pt[e] - avg, axis=1)
            lc = LineCollection(Pt[e][edges], cmap=pm.FR_CMAP, norm=norm, linewidths=.5, alpha=.75, zorder=2)
            lc.set_array((dv[edges[:, 0]] + dv[edges[:, 1]]) / 2 / unit)
            ax.add_collection(lc)
        ax.add_collection(LineCollection(avg[edges], colors="k", linewidths=.9, zorder=4))
        ax.set_xlim(*xl), ax.set_ylim(*yl), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(lab, fontsize=FS["label"], pad=3, color=col if c != "none" else "k", linespacing=1.15)
        ax.text(.5, -.03, "P(true cell) %.2f, interference %.0f%%" % (Ms[c]["p"], 100 * Ms[c]["share"][-1]),
                transform=ax.transAxes, fontsize=FS["tick"] - 1, color="0.25", ha="center", va="top")
    cx = fig.add_axes([(.5 + 5 * pw + 4 * gap + .25) / W, ytop / H, .12 / W, pw / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=pm.FR_CMAP, norm=norm), cax=cx, extend="max")
    cb.set_label("distance from the environments'\naverage grid (cells)", fontsize=FS["tick"])
    cb.outline.set_visible(False)
    fig.text(.5 / W, (H - .3) / H, "The Trained RNN's grids at $T$ in 10 held-out environments, with its hidden output "
             "clamped along directions the read-out does not read (black: their average; colour: distance from it)",
             fontsize=FS["label"] + .5, va="center")
    fig.text(.1 / W, (H - .3) / H, r"$\mathbf{a}$", fontsize=FS["label"] + 2, va="center")
    # bottom row
    by, bh, bgap = 1.3, 2.1, .9
    bw = (W - .5 - .3 - .3 - 3 * bgap) / 4
    axs = [fig.add_axes([(.5 + .3 + i * (bw + bgap)) / W, by / H, bw / W, bh / H]) for i in range(4)]
    for key, (lab, col, conds) in FAM.items():
        xs = [0.] + [100 * var[c] for c in conds]
        for j, (ax, get) in enumerate(((axs[0], lambda c: Ms[c]["p"]), (axs[1], lambda c: 100 * Ms[c]["share"][-1]))):
            ys = [get("none")] + [get(c) for c in conds]
            ax.plot(xs, ys, c=col, lw=1.5 if len(conds) > 1 else 0, marker="o", ms=4.2, label=lab, zorder=3)
            if j == 0:
                for c in conds:                                                     # spread over environments
                    pe = Ms[c]["p_env"] - Ms["none"]["p_env"] + Ms["none"]["p"]
                    ax.plot([100 * var[c]] * 2, np.percentile(pe, [10, 90]), c=col, lw=.8, alpha=.6)
    axs[0].plot([0], [Ms["none"]["p"]], marker="o", ms=5, c="#1f77b4", zorder=4)
    axs[1].plot([0], [100 * Ms["none"]["share"][-1]], marker="o", ms=5, c="#1f77b4", zorder=4)
    axs[0].set_ylabel(r"P(true cell) at $T$, $b_T(r_g)\,b_T(r_c)$")
    axs[0].set_title("Accuracy", fontsize=FS["label"] + .5, pad=4)
    axs[1].set_ylabel("interference share at $T$ (%)")
    axs[1].set_title(r"Inter-environment interference, $\frac{V_{it}}{V_{ce} + V_{it}}$", fontsize=FS["label"] + .5, pad=4)
    for ax in axs[:2]:
        ax.set_xlabel("variance of $u$ clamped (%)", labelpad=1)
        ax.set_xlim(-1.5, 40)
    axs[0].legend(frameon=False, fontsize=FS["tick"] - 1.5, loc="upper right", labelspacing=.25, handlelength=1.8)
    steps = np.arange(1, T + 1)
    for c, lab, col in SHOWN:
        ls = "-" if c != "rv260" else (0, (3, 1.6))
        axs[2].plot(steps, Ms[c]["acc_t"], c=col, lw=1.8 if c == "none" else 1.3, ls=ls, label=lab.replace("\n", ", "))
        axs[3].plot(steps, 100 * Ms[c]["share"], c=col, lw=1.8 if c == "none" else 1.3, ls=ls)
    axs[2].set_ylabel(r"$\frac{1}{2}(\bar b_t(r_g) + \bar b_t(r_c))$")
    axs[2].set_title("Accuracy over the episode", fontsize=FS["label"] + .5, pad=4)
    axs[3].set_ylabel("interference share (%)")
    axs[3].set_title("Interference over the episode", fontsize=FS["label"] + .5, pad=4)
    for ax in axs[2:]:
        ax.set_xlim(1, T), ax.set_xticks([1, 10, 20, 30]), ax.set_xlabel("step $t$", labelpad=1)
    axs[2].legend(frameon=False, fontsize=FS["tick"] - 2, loc="lower right", labelspacing=.2, handlelength=1.8)
    for i, ax in enumerate(axs):
        ax.spines[["top", "right"]].set_visible(False)
        x0, y0, w, h = ax.get_position().bounds
        fig.text(x0 - .7 / W, y0 + h + 6 / 72 / H, r"$\mathbf{%s}$" % "bcde"[i], fontsize=FS["label"] + 2, va="baseline")
    import textwrap
    rate = lambda c: (Ms[c]["p"] - Ms["none"]["p"]) / (100 * var[c])
    fig.text(.5 / W, .62 / H, textwrap.fill(
        "Clamping: after every step the hidden output u loses its deviation from the population mean inside the "
        "subspace V (u minus (u minus mean) projected onto V); every V is orthogonal to the read-out's 18 directions "
        "and to the shared grid's top 10, and is estimated on the 29 other environments. "
        "b, c: x = the share of u's stream-level variance the subspace holds; vertical ticks in b: 10th-90th percentile "
        "over the 10 environments of the paired effect (every lesion lowers P(true cell) in 10 of 10). Loss of P(true "
        "cell) per 1%% of variance: environment-specific %.4f / %.4f / %.4f (10 / 30 / 60 directions), stream-to-stream "
        "%.4f, whole-grid shift %.4f, random %.4f to %.4f. Interference: not noise-corrected." % (
            rate("it10"), rate("it30"), rate("it60"), rate("w30"), rate("ee30"),
            min(rate(c) for c in FAM["rand"][2]), max(rate(c) for c in FAM["rand"][2])), 250),
        fontsize=FS["tick"] - 1.5, color="0.35", va="top", linespacing=1.35)
    path = fig_path("fig1c_lesion.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("saved", path)
    for c in ("none",) + tuple(c for _, _, cs in FAM.values() for c in cs):
        print("%-7s variance %.1f%% | P(true cell) %.3f (per 1%% %s) | interference T %.1f%%" % (
            c, 100 * var[c], Ms[c]["p"], "%.4f" % rate(c) if c != "none" else "-", 100 * Ms[c]["share"][-1]))


if __name__ == "__main__":
    figure()
