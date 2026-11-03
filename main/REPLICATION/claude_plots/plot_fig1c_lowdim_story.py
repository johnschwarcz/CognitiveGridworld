"""Where the factorized observers' grid folds, the joint observers move both beliefs together, and the trained network
does so through environment-specific hidden dimensions that keep expanding (25 Oct 2026)
-> main/REPLICATION/figure_bin/fig1c_lowdim_story.png

Data: the alias scan (main/DATA/experimental/fig1c_compression; 39 environments, e = 2 left out; 100 cells x 50
streams; all four agents on the same streams) and the networks' hidden cell means on it
(hidden_scan_cellmeans_<net>.npz: research9/hidden/replay_scan.py; u = o * tanh(c), what the read-out reads; c = the
cell state, the network's memory).
    a  the environments' average grid at T = 30 (each cell's mean belief averaged over the 39 environments; per-step
       plane, as plot_fig1c_mesh.avg_grid_fr_figure), edges coloured by the two cells' factorization regret at T
    b  the same grids, edges coloured by the cells' joint movement so far: sum over t of C_t = Cov over the cell's 50
       streams of (|u_g(t)|, |u_c(t)|) (u_v(t) = E_t[r_v] - E_{t-1}[r_v], posterior means, E_0 = 4.5; grid units^2),
       mean over the environments
    c  per cell: the joint movement so far against the cell's entanglement summed over the steps (mean over streams
       and environments)
    d, e  dimension of the environment-specific hidden component: the cells' hidden means X[e, k, t] (1000 units)
       split additively as in plot_fig1c_mesh.decompose (cell effect = the shared grid; cell x environment
       interaction = the environment-specific distortion); participation ratio (tr C)^2 / tr(C^2) of the interaction's
       covariance over the (environment, cell) pairs, within the fifths of cells by FR at T (20 cells x 39
       environments each), and of the shared grid's covariance over the 100 cells. Cells in the networks' own slot
       labels (the networks never see which slot is the goal; FR mapped to those labels). Raw PRs: the within-cell
       noise is under 3% of the interaction's variance (noise-corrected PR at T 61.9 vs raw 63.8 for the Trained u,
       figdata/round6_hidden/hidden_interference.npz).
Cache: figdata/round6_hidden/lowdim_story.npz (build(), about 3 min CPU).
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg OMP_NUM_THREADS=4 /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_lowdim_story.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize         # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402

SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
CACHE = os.path.join(SCAN, "figdata", "round6_hidden", "lowdim_story.npz")
GPA = os.path.join(SCAN, "figdata", "round6_hidden", "gpa_steps.json")             # build_gpa()
GPA_INPUT = os.path.join(SCAN, "figdata", "round6_hidden", "gpa_pathway_verifier.json")   # inputs' shape share
ENVS = [e for e in range(40) if e != 2]
E, K, M, T, R = 39, 100, 50, 30, 10
ORDER = ("Joint", "Trained", "Naive", "Echo")
NETS = (("trained", "Trained"), ("echo", "Echo"))
GRID_K = 5                                                                        # the hidden grid's subspace
JM_CMAP = LinearSegmentedColormap.from_list("jm", ["#d4d4d4", "#9ecae1", "#4292c6", "#08519c", "#08306b"])


def _pr(X):
    """participation ratio of the covariance of the rows of X (n, H) about 0, via the n x n Gram matrix"""
    X = X.reshape(-1, X.shape[-1]).astype(np.float64)
    G = X @ X.T
    return np.trace(G) ** 2 / (G ** 2).sum()


def _parts(X):
    """cell effect (K, H) and interaction (E, K, H) of cell means X (E, K, H)"""
    mu = X.mean((0, 1), keepdims=True)
    ce = X.mean(0, keepdims=True) - mu
    return ce[0], X - X.mean(1, keepdims=True) - ce


def _grid_share(Xall, k=GRID_K, seed=0):
    """per step: the environment-specific share V_it / (V_ce + V_it) inside the shared grid's own subspace (top-k
    principal directions of the cell effect); in-sample, and split (subspace from 20 environments, decomposition on
    the other 19)"""
    h = np.random.default_rng(seed).permutation(E)
    h1, h2 = h[:20], h[20:]
    sh, sp = [], []
    for t in range(T):
        X = np.asarray(Xall[:, :, t], np.float64)
        ce, it = _parts(X)
        B = np.linalg.svd(ce, full_matrices=False)[2][:k].T
        vc, vi = ((ce @ B) ** 2).sum(-1).mean(), ((it @ B) ** 2).sum(-1).mean()
        sh.append(vi / (vc + vi))
        B1 = np.linalg.svd(_parts(X[h1])[0], full_matrices=False)[2][:k].T
        ce2, it2 = _parts(X[h2])
        vc, vi = ((ce2 @ B1) ** 2).sum(-1).mean(), ((it2 @ B1) ** 2).sum(-1).mean()
        sp.append(vi / (vc + vi))
    return np.array(sh), np.array(sp)


def _gpa_share(Xc, iters=15):
    """environment-specific SHAPE share: generalized orthogonal Procrustes of the environments' centred grids Xc
    (E, K, H) onto their common mean, then V_it / (V_ce + V_it) of the aligned grids (the workflow verifier's
    gpa_fast: each rotation only needs the K-dim row space of its grid)"""
    As = []
    for e in range(len(Xc)):
        U, s, _ = np.linalg.svd(Xc[e], full_matrices=False)
        As.append(U * s)                                                            # (K, K) coordinates in the row space
    Y = Xc.copy()
    G = Y.mean(0)
    for _ in range(iters):
        for e in range(len(Xc)):
            P, _, Qt = np.linalg.svd(As[e].T @ G, full_matrices=False)
            Y[e] = As[e] @ (P @ Qt)
        G = Y.mean(0)
    vce, vit = (G ** 2).sum(-1).mean(), ((Y - G) ** 2).sum(-1).mean()
    return float(vit / (vce + vit))


def build_gpa():
    """per step, network and view: the shape share (aligned) and the coordinate share; a label-permuted null at
    t = 1, 5, 30 (about 4 min per network and view on 2 threads) -> figdata/round6_hidden/gpa_steps.json"""
    import json
    rng = np.random.default_rng(3)
    out = {}
    ar = np.arange(E)[:, None]
    for net, _ in NETS:
        f = np.load(os.path.join(SCAN, f"hidden_scan_cellmeans_{net}.npz"))
        for view in ("u", "c"):
            X = f[view][ar, f["perm_raw"]]
            coord, aligned, null = [], [], {}
            for t in range(T):
                Xt = X[:, :, t].astype(np.float64)
                Xc = Xt - Xt.mean(1, keepdims=True)
                ce = Xc.mean(0)
                coord.append(float(((Xc - ce) ** 2).sum(-1).mean() / (Xc ** 2).sum(-1).mean()))
                aligned.append(_gpa_share(Xc))
                if t in (0, 4, 29):
                    null[t + 1] = _gpa_share(np.stack([Xc[i][rng.permutation(K)] for i in range(E)]))
            out[f"{net}_{view}"] = dict(coord=coord, aligned=aligned, null=null)
    json.dump(out, open(GPA, "w"))


def build():
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    assert (meta["cell_of"] == np.repeat(np.arange(K), M)).all()
    out = dict(fr=meta["FR"][ENVS].astype(np.float64).reshape(E, K, M, T).mean(2),
               ent=meta["rev"][ENVS].astype(np.float64).reshape(E, K, M, T).mean(2))
    Rv = np.arange(R)
    for a in ORDER:
        B = np.load(os.path.join(SCAN, f"scan_{a}.npz"))["beliefs"][ENVS].astype(np.float64)   # (E, 5000, T, 20)
        b = B.reshape(E, K * M, T, 2, R)
        b = b / b.sum(-1, keepdims=True)
        Ex = np.concatenate([np.full((E, K * M, 1, 2), (R - 1) / 2), (b * Rv).sum(-1)], 2)
        u = np.abs(np.diff(Ex, axis=2)).reshape(E, K, M, T, 2)
        uc = u - u.mean(2, keepdims=True)
        out["jm_" + a] = (uc[..., 0] * uc[..., 1]).sum(2) / (M - 1)               # (E, K, T)
        print(a, "joint movement done", flush=True)
    ar = np.arange(E)[:, None]
    for net, _ in NETS:
        f = np.load(os.path.join(SCAN, f"hidden_scan_cellmeans_{net}.npz"))
        perm = f["perm_raw"]                                                        # goal-ordered -> slot labels
        assert (f["env_index"] == ENVS).all()
        out["perm_" + net] = perm
        frT = out["fr"][ar, perm][:, :, -1].mean(0)                                 # FR at T per slot-labelled cell
        fifth = np.searchsorted(np.quantile(frT, [.2, .4, .6, .8]), frT, side="right")
        out["fifth_" + net] = fifth
        Q = f["Q"].astype(np.float32)                                               # (1000, 18) read-out directions
        for view in ("u", "c"):
            X = f[view][ar, perm].astype(np.float32)                                # (E, K, T, 1000)
            it = X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean((0, 1), keepdims=True)
            ce = X.mean(0) - X.mean((0, 1))                                         # (K, T, 1000)
            del X
            out[f"{net}_{view}_pr_all"] = np.array([_pr(it[:, :, t]) for t in range(T)])
            out[f"{net}_{view}_pr_ce"] = np.array([_pr(ce[:, t]) for t in range(T)])
            out[f"{net}_{view}_pr_fifth"] = np.array([[_pr(it[:, fifth == j, t]) for t in range(T)] for j in range(5)])
            out[f"{net}_{view}_in_grid"], out[f"{net}_{view}_in_grid_split"] = _grid_share(f[view][ar, perm])
            out[f"{net}_{view}_vit"] = (it.astype(np.float64) ** 2).sum(-1).mean((0, 1))
            out[f"{net}_{view}_vit_inQ"] = ((it @ Q).astype(np.float64) ** 2).sum(-1).mean((0, 1))
            print(net, view, "PR at T: interaction %.1f (fifths %s), shared grid %.1f" % (
                out[f"{net}_{view}_pr_all"][-1], np.round(out[f"{net}_{view}_pr_fifth"][:, -1], 1),
                out[f"{net}_{view}_pr_ce"][-1]), flush=True)
            del it, ce
    np.savez(CACHE, **out)
    return out


def load():
    return dict(np.load(CACHE)) if os.path.exists(CACHE) else build()


def _rank(x):
    return np.argsort(np.argsort(x))


def _spearman(x, y):
    return np.corrcoef(_rank(x), _rank(y))[0, 1]


def figure():
    import plot_fig1c_mesh as pm
    from main.utils import apply_fig_style, fig_path
    Z = load()
    A, frm, home, edges, cells, cm = pm.avg_data()
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    t = T - 1
    fr_T = np.maximum(Z["fr"][:, :, t].mean(0), 1e-6)                                # per cell (goal labels)
    assert np.allclose(fr_T, np.maximum(frm[:, t], 1e-6))
    jm = {a: Z["jm_" + a].sum(2).mean(0) for a in ORDER}                            # joint movement so far at T
    ent_S = Z["ent"].sum(2).mean(0)                                                 # summed entanglement per cell
    pw, gap, cbw = 2.45, .3, .12
    rowh = pw + .55
    W = .55 + 4 * pw + 3 * gap + .25 + cbw + .95
    H = .2 + 2 * rowh + .95 + 2.05 + 1.25
    fig = plt.figure(figsize=(W, H))
    fr_norm = LogNorm(.05, 30)
    jm_max = np.quantile(np.concatenate([jm[a] for a in ORDER]), .98)
    jm_norm = Normalize(0, jm_max)
    rows = (("a", "Where the factorized observers' grids fold: the edges' factorization regret at $T$",
             lambda: np.sqrt(fr_T[edges[:, 0]] * fr_T[edges[:, 1]]), pm.FR_CMAP, fr_norm,
             "factorization regret\nat $T$ (nats)", "both"),
            ("b", r"Where the beliefs move together: joint movement so far, $\Sigma_t\,\mathrm{Cov}(|u_g(t)|, |u_c(t)|)$ "
                  "of the edges' cells", None, JM_CMAP, jm_norm, "joint movement so far\n(grid units$^2$)", "both"))
    for r_, (letter, title, val, cmap, norm, cblab, ext) in enumerate(rows):
        y0 = H - .2 - (r_ + 1) * rowh + .05
        for c_, a in enumerate(ORDER):
            ax = fig.add_axes([(.55 + c_ * (pw + gap)) / W, y0 / H, pw / W, pw / H])
            pos, hm = A[a]["proj"](A[a]["Xm"][:, t], float(t)), A[a]["proj"](home, float(t))
            xl, yl = pm._box(np.vstack([pos, hm]))
            ax.add_collection(LineCollection(hm[edges], colors="0.8", linewidths=.55, linestyles=(0, (3, 2))))
            lc = LineCollection(pos[edges], cmap=cmap, norm=norm, linewidths=1.6)
            lc.set_array(val() if val else (jm[a][edges[:, 0]] + jm[a][edges[:, 1]]) / 2)
            ax.add_collection(lc)
            ax.set_xlim(*xl), ax.set_ylim(*yl), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
            if r_ == 0:
                ax.set_title(pm.NAME[a], fontsize=FS["label"] + .5, pad=3)
        cx = fig.add_axes([(.55 + 4 * pw + 3 * gap + .25) / W, y0 / H, cbw / W, pw / H])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cx, extend=ext)
        cb.set_label(cblab, fontsize=FS["tick"])
        cb.outline.set_visible(False)
        fig.text(.55 / W, (y0 + pw + .3) / H, title, fontsize=FS["label"] + .5, va="baseline")
        fig.text(.12 / W, (y0 + pw + .3) / H, r"$\mathbf{%s}$" % letter, fontsize=FS["label"] + 2, va="baseline")
    # bottom row
    bh, by, bgap = 2.05, 1.25, .82
    bw = (W - .55 - .3 - .25 - 3 * bgap) / 4
    axs = [fig.add_axes([(.55 + .3 + i * (bw + bgap)) / W, by / H, bw / W, bh / H]) for i in range(4)]
    ax = axs[0]
    o = np.argsort(ent_S)
    for a in ORDER:
        tr = a == "Trained"
        ax.scatter(ent_S, jm[a], s=7, c=pm.COLOR[a], alpha=.45 if tr else .75, lw=0, label=pm.NAME[a])
        k = 15
        ax.plot(np.convolve(ent_S[o], np.ones(k) / k, "valid"), np.convolve(jm[a][o], np.ones(k) / k, "valid"),
                c=pm.COLOR[a], lw=2.6 if tr else 1.3, alpha=.45 if tr else 1)
    ax.axhline(0, c="0.8", lw=.6, zorder=0)
    ax.set_xlabel(r"entanglement of the cell, $\Sigma_t\,\mathcal{E}_t$ (nats)", labelpad=1)
    ax.set_ylabel("joint movement so far\n(grid units$^2$)")
    ax.set_title("Joint movement follows entanglement", fontsize=FS["label"] + .5, pad=4)
    ax.text(.03, .97, "Spearman over cells:\n" + "\n".join("%s %.2f" % (pm.NAME[a], _spearman(ent_S, jm[a]))
                                                          for a in ORDER), transform=ax.transAxes, va="top",
            fontsize=FS["tick"] - 1.5, color="0.3", linespacing=1.25)
    ax.legend(frameon=False, fontsize=FS["tick"] - 1.5, loc="upper left", bbox_to_anchor=(0, .66), markerscale=1.5,
              handletextpad=.3, labelspacing=.2)
    steps = np.arange(1, T + 1)
    NC = {"trained": "#1f77b4", "echo": "#ff7f0e"}
    import json
    HI = np.load(os.path.join(SCAN, "figdata", "round6_hidden", "hidden_interference.npz"))
    GP = json.load(open(GPA))
    GI = json.load(open(GPA_INPUT))
    ax = axs[1]                                                                     # d: shape share
    for net, a in NETS:
        ax.plot(steps, 100 * np.array(GP[f"{net}_u"]["aligned"]), c=NC[net], lw=1.6)
        ax.plot(steps, 100 * np.array(GP[f"{net}_c"]["aligned"]), c=NC[net], lw=1.1, ls=(0, (3, 1.6)))
        ax.plot(steps, 100 * HI[f"beliefs_{a}_goal__share_c"][0], c=NC[net], lw=1.0, ls=":")
    x_in = 100 * np.mean([GI["trained_input_x"]["aligned"], GI["echo_input_x"]["aligned"]])
    ax.axhline(x_in, c="0.55", lw=.8, ls=(0, (1, 1.5)), zorder=0)
    ax.text(29.5, x_in + 1.2, "inputs", ha="right", va="bottom", fontsize=FS["tick"] - 1.5, color="0.45")
    ax.set_xlim(1, T), ax.set_xticks([1, 10, 20, 30]), ax.set_ylim(0, 52), ax.set_xlabel("step $t$", labelpad=1)
    ax.set_ylabel("environment-specific share (%)")
    ax.set_title("Same grid shape in every environment", fontsize=FS["label"] + .5, pad=4)
    ax.legend(handles=[Line2D([], [], c="0.3", lw=1.6, label="hidden $u$, shape (grids rotated)"),
                       Line2D([], [], c="0.3", lw=1.1, ls=(0, (3, 1.6)), label="hidden $c$, shape"),
                       Line2D([], [], c="0.3", lw=1.0, ls=":", label="beliefs")],
              frameon=False, fontsize=FS["tick"] - 1.5, loc="lower right", bbox_to_anchor=(1, .43), handlelength=2.2,
              labelspacing=.25)
    ax = axs[2]                                                                     # e: orientations' dimension
    ip = list(HI["pr_t"])
    for net, a in NETS:
        ax.plot(steps, Z[f"{net}_u_pr_all"], c=NC[net], lw=1.6)
        ax.plot(ip, np.median(HI[f"{net}__u__spec_pr_it_env_corr"], 1), c=NC[net], lw=1.1, ls=(0, (3, 1.6)),
                marker="o", ms=2.5)
        ax.plot(steps, Z[f"{net}_u_pr_ce"], c=NC[net], lw=.9, ls=":")
    ax.set_xlim(1, T), ax.set_xticks([1, 10, 20, 30]), ax.set_ylim(0, None), ax.set_xlabel("step $t$", labelpad=1)
    ax.set_ylabel(r"dimension, $(\mathrm{tr}\,C)^2 / \mathrm{tr}(C^2)$")
    ax.set_title("...in its own orientation (hidden $u$)", fontsize=FS["label"] + .5, pad=4)
    ax.legend(handles=[Line2D([], [], c="0.3", lw=1.6, label="environment-specific part,\nall 39 environments pooled"),
                       Line2D([], [], c="0.3", lw=1.1, ls=(0, (3, 1.6)), marker="o", ms=2.5,
                              label="the same within one environment\n(median)"),
                       Line2D([], [], c="0.3", lw=.9, ls=":", label="shared grid")],
              frameon=False, fontsize=FS["tick"] - 1.5, loc="center right", bbox_to_anchor=(1, .5), handlelength=2.2,
              labelspacing=.35)
    ax = axs[3]                                                                     # f: by FR fifth
    for net, a in NETS:
        P = Z[f"{net}_u_pr_fifth"]
        ax.plot(steps, P[4], c=NC[net], lw=1.6)
        ax.plot(steps, P[0], c=NC[net], lw=1.1, ls=(0, (3, 1.6)))
    ax.set_xlim(1, T), ax.set_xticks([1, 10, 20, 30]), ax.set_ylim(0, None), ax.set_xlabel("step $t$", labelpad=1)
    ax.set_ylabel(r"dimension, $(\mathrm{tr}\,C)^2 / \mathrm{tr}(C^2)$")
    ax.set_title("Most orientations where FR is high", fontsize=FS["label"] + .5, pad=4)
    f_in = Z["trained_u_vit_inQ"][-1] / Z["trained_u_vit"][-1]
    e_in = Z["echo_u_vit_inQ"][-1] / Z["echo_u_vit"][-1]
    ax.text(29.5, 11.5, "environment-specific part in the\nread-out's 18 directions at $T$:\nTrained %.1f%%, ESN %.1f%% "
            "(chance 1.8%%)" % (100 * f_in, 100 * e_in), ha="right", va="bottom", fontsize=FS["tick"] - 1.5, color="0.3",
            linespacing=1.25)
    ax.legend(handles=[Line2D([], [], c="0.3", lw=1.6, label="environment-specific, highest-FR fifth"),
                       Line2D([], [], c="0.3", lw=1.1, ls=(0, (3, 1.6)), label="environment-specific, lowest-FR fifth")],
              frameon=False, fontsize=FS["tick"] - 1.5, loc="center right", bbox_to_anchor=(1, .55), handlelength=2.2,
              labelspacing=.25)
    fig.legend(handles=[Line2D([], [], c=NC["trained"], lw=1.8, label="Trained RNN"),
                        Line2D([], [], c=NC["echo"], lw=1.8, label="ESN")], frameon=False, fontsize=FS["tick"],
               loc="center", ncol=2, handlelength=2.2, columnspacing=2,
               bbox_to_anchor=((axs[1].get_position().x0 + axs[3].get_position().x1) / 2, (by + bh + .62) / H))
    for i, ax in enumerate(axs):
        ax.spines[["top", "right"]].set_visible(False)
        x0, y0_, w, h = ax.get_position().bounds
        fig.text(x0 - .62 / W, y0_ + h + 6 / 72 / H, r"$\mathbf{%s}$" % "cdef"[i], fontsize=FS["label"] + 2,
                 va="baseline")
    import textwrap
    fig.text(.55 / W, .66 / H, textwrap.fill(
        "Alias scan, 39 environments x 100 cells x 50 streams. a, b: the environments' average grid at T (each cell's "
        "mean belief averaged over the environments), dashed: the targets' grid; joint movement: the covariance over "
        "a cell's streams of the two variables' update sizes, summed over the steps, mean over the environments. "
        "c: one point per cell, lines: running means over 15 cells. d-f: the networks' hidden cell means (slot labels) "
        "split into the shared grid (cell effect) and the environment-specific part (cell x environment interaction). "
        "d: environment-specific share of the grid's shape: each environment's centred 100-cell grid rotated onto the "
        "common one (generalized orthogonal Procrustes), then the interaction's share of cell effect plus interaction "
        "(cell labels permuted: 78-90%); beliefs: the coordinate share (noise-corrected); inputs: the LSTM input. "
        "e, f: dimension = participation ratio; pooled: the environment-specific part's covariance over all (environment, "
        "cell) pairs; within one environment: over its 100 cells (median of 39); f: pooled within the fifths of cells by "
        "FR at T (20 cells x 39 environments)." , 250), fontsize=FS["tick"] - 1.5, color="0.35", va="top",
        linespacing=1.35)
    path = fig_path("fig1c_lowdim_story.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("saved", path)
    hi, lo = fr_T >= np.quantile(fr_T, .8), fr_T <= np.quantile(fr_T, .2)
    for a in ORDER:
        print("%-8s joint movement so far: top-FR fifth %.3f vs bottom %.3f | Spearman with summed entanglement %.2f, "
              "with FR_T %.2f" % (a, jm[a][hi].mean(), jm[a][lo].mean(), _spearman(ent_S, jm[a]), _spearman(fr_T, jm[a])))
    for net, _ in NETS:
        for view in ("u", "c"):
            P = Z[f"{net}_{view}_pr_fifth"]
            print("%-7s %s: PR interaction t=1/10/30 all %s | top fifth %s | bottom %s | shared grid %s" % (
                net, view, np.round(Z[f"{net}_{view}_pr_all"][[0, 9, 29]], 1), np.round(P[4, [0, 9, 29]], 1),
                np.round(P[0, [0, 9, 29]], 1), np.round(Z[f"{net}_{view}_pr_ce"][[0, 9, 29]], 1)))


if __name__ == "__main__":
    if "build" in sys.argv[1:]:
        build()
    figure()
