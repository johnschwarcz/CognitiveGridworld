"""The Trained RNN's grid / particles coloured by the stream's evidence and the gradflow figure's measures (27 Oct 2026;
John: a grid or particle movie of the network coloured by the dynamics_interference measures)
-> main/REPLICATION/figure_bin/fig1c_movie_measures{,_envs,_particles}.gif

Measures (nine), all shown in the plane of the Trained RNN's posterior means (E[r_g], E[r_c]) at step t = 1 .. 30
(Catmull-Rom between steps; the true lattice dashed):
    the stream's evidence  entanglement (the step's house entanglement rev, the alignment variable of the gradflow
                           figure); the error delta, the Trained RNN's read-out against the target it was trained on
                           (the Optimal's exact marginals): [KL_sym(b_g^RNN, b_g^Opt) + KL_sym(b_c^RNN, b_c^Opt)] / 2
                           (John 10-28: "the actual error, delta", replacing entanglement minus its mean so far);
                           factorization regret, the symmetric KL divergence
                           [KL(b_v^Opt || b_v^Naive) + KL(b_v^Naive || b_v^Opt)] / 2 between the Optimal's and the
                           Naive's beliefs about variable v, averaged over BOTH variables v = g, c (John 10-28; the
                           scan's stored FR is the goal variable's alone): fr_both()
    state                  |u_t|; the effective dimension of the network's activity u_t (participation ratio
                           (sum E)^2 / sum E^2 of E_k = ((u_t - mu) . v_k)^2 on the network's fixed 1000-d PCA basis,
                           gradflow_basis_<net>); the same for its beliefs (John 10-28, replacing the update's effective
                           dimension): its 20 read-out probabilities b_t (two marginals) on the fixed PCA basis of the
                           Trained RNN's scan beliefs (belief_pr())
    update                 credit coherence; the update's energy |dW|^2; the cross-band share (10% bands)
Colours (John 10-28: raw data, not stream percentiles): each measure's raw values on its own fixed scale, the 2nd -
98th percentile of the values the movie draws (log10 for positive heavy-tailed measures, whose 98th percentile exceeds
20x the 2nd), with a colour bar under each panel; viridis, bright = more. (data() still ranks by default, for the
rings, jumps and settled figures.)
Modes:
    avg        3 x 3 panels, the environment-averaged grid: each cell at its mean over its 50 streams, averaged over the
               39 scan environments; colour: each cell's mean measure over its 50 streams, averaged over the
               environments; corners (0, 0), (9, 9): circles, (0, 9), (9, 0): squares
    envs       3 x 3 panels, environments' own grids on top of each other (John 10-27: to see whether unfolding,
               or a jump across the grid, goes with any measure): each environment's 100 cell means joined to their
               lattice neighbours, every edge coloured by its two cells' mean value, 1-px lines drawn from the
               shortest edge to the longest, so stretched edges (tears, jumps across the grid) lie on top. All 39
               environments (John 10-28, later: "can we add all 39 envs?"; first 10 of them, every 4th), drawn at
               165 dpi
    particles  3 x 3 panels, 15,600 streams (the first 4 of each cell, every environment) as 2-px squares, each at its
               own posterior mean; colour: its own value; one fixed random drawing order; 64 viridis levels
               (GIF < 100 MB)
    envs_vs_typical, particles_vs_typical   copies with the shared pattern removed (John 10-28: "normalizes per
               timestep, so we see dynamics on top of the typical changes through time"): colour = the raw (or log10)
               value minus the mean of the same cell at that step over all 39 environments (particles: over their
               cell's streams in all 39), red above the typical value, blue below, light grey at it (coolwarm,
               +- the 98th percentile of the departures). The avg movie has no such copy: its departure is 0.
Every measure uses all 50 streams per cell (5,000 per environment; John 10-28 "more trajs per env": the update measures
from scan_credit_trained_ns50.npz); each frame notes the environments and streams it shows.
Data: the scan (scan_meta rev, FR; scan_Trained beliefs), round7_jumps/scan_hidden_trained.npz (|u|, effective
dimension), round7_jumps/scan_credit_trained_ns50.npz (the update measures; before step 16 the credit is sent back
over the t steps there are, not 15).
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_measures_movie.py [avg|envs|particles|envs_vs_typical|
        particles_vs_typical]"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402
from PIL import Image                                                             # noqa: E402
from scipy.stats import rankdata                                                  # noqa: E402

SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
R7 = os.path.join(SCAN, "figdata", "round7_jumps")
ENVS = [e for e in range(40) if e != 2]
R, T, M, NS = 10, 30, 50, 50                                                      # NS: streams per cell analysed
NS_SHOW = 4                                                                       # particles drawn per cell and
PX_P = 2                                                                          # environment, their size (px)
ENVS_SHOW = 1                                                                     # envs: every environment (39)
DPI = {"avg": 110, "envs": 165, "particles": 110}                                 # envs: 1-px lines, thinner
LEVELS = {"avg": 128, "envs": 64, "particles": 64}                                # colour levels (GIF < 100 MB)
NET = "trained"
PANELS = (("rev", "Entanglement", r"$\mathrm{rev}_t$"),
          ("err", "Error",
           r"$\delta = \frac{1}{2}\Sigma_{v = g, c}\,\mathrm{KL}_\mathrm{sym}(b^\mathrm{RNN}_v, b^\mathrm{Opt}_v)$"),
          ("fr", "Factorization regret",
           r"$\frac{1}{2}\Sigma_{v = g, c}\,\mathrm{KL}_\mathrm{sym}(b^\mathrm{Opt}_v, b^\mathrm{Naive}_v)$"),
          ("l2", "Norm", r"$\|u_t\|$"),
          ("pr", "Effective dim., activity", r"$(\Sigma_k E_k)^2 / \Sigma_k E_k^2$, $E_k = ((u_t - \bar u) v_k)^2$"),
          ("bpr", "Effective dim., beliefs", r"$(\Sigma_k E_k)^2 / \Sigma_k E_k^2$, $E_k = ((b_t - \bar b) w_k)^2$"),
          ("coh", "Credit coherence", r"$\|\Sigma_k \delta_k\|^2 / \Sigma_k \|\delta_k\|^2$"),
          ("dW", r"$\Delta W$ energy", r"$\|\Sigma_k \delta_k z_k^\top\|^2$"),
          ("band_x", "Cross-band share",
           r"$\Sigma_B (\langle \Delta W_B, \Delta W \rangle - \|\Delta W_B\|^2) / \|\Delta W\|^2$"))
FPS, STEP_S = 25, .5
UNITS = {"rev": "nats", "err": "nats", "fr": "nats", "l2": "norm", "pr": "dimensions", "bpr": "dimensions",
         "coh": "coherence", "dW": "energy", "band_x": "share"}


class _Exact:
    """each pixel to its nearest palette colour, exactly (Pillow's quantize(palette=) matches at about 4 units per
    channel and greys the white background), by a lazily filled 24-bit table"""

    def __init__(self, cols):
        self.cols = np.asarray(cols, np.float32)
        self.flat = np.asarray(cols, np.uint8).ravel().tolist()
        self.lut = np.full(1 << 24, -1, np.int16)

    def __call__(self, im):
        X = np.asarray(im)[..., :3].astype(np.int32)
        code = (X[..., 0] << 16) | (X[..., 1] << 8) | X[..., 2]
        miss = np.unique(code[self.lut[code] < 0])
        for k in range(0, len(miss), 20000):
            u = miss[k:k + 20000]
            rgb = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], 1).astype(np.float32)
            self.lut[u] = ((rgb[:, None] - self.cols[None]) ** 2).sum(-1).argmin(1)
        P = Image.fromarray(self.lut[code].astype(np.uint8), "P")
        P.putpalette(self.flat)
        return P


def _rank(A):
    """percentile along axis 1 (ties: average ranks), 0 .. 1"""
    return (rankdata(A, axis=1, method="average") - 1) / (A.shape[1] - 1)


def fr_both():
    """factorization regret averaged over the two variables, [FR_g + FR_c] / 2, FR_v = fr_sym(the Optimal's marginal of
    v, the Naive's) (plot_fig1c_all.fr_sym, nats), exact in float64 from the scan's observations (exact_bayes,
    naive_marginals, as the scan's goal-only FR) -> (39, 5000, T); cached in round7_jumps/fr_both.npz (about 1 min);
    the first computation checks the goal variable's against the stored FR"""
    path = os.path.join(R7, "fr_both.npz")
    if not os.path.exists(path):
        import plot_fig1c_all as fa
        meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
        out, err = [], 0.
        for e in ENVS:
            obs, L, g = meta["obs"][e], meta["L"][e], int(meta["goal"][e])
            J, Nv = fa.exact_bayes(obs, L)[1], fa.naive_marginals(obs, L)
            fr = np.stack([fa.fr_sym(J[:, :, v], Nv[:, :, v]) for v in (0, 1)])    # raw variable order
            err = max(err, float(np.abs(fr[g] - meta["FR"][e]).max()))
            out.append(fr.mean(0))
        print(f"fr_both: the goal variable's FR matches the stored one to {err:.1e}", flush=True)
        np.savez(path, fr_both=np.array(out, np.float32), check=err)
    with np.load(path) as f:
        return f["fr_both"].astype(np.float64)


def err_both():
    """the Trained RNN's error against its training target, averaged over the two variables, [d_g + d_c] / 2,
    d_v = fr_sym(the network's marginal of v, the Optimal's exact one) (nats; the scan's beliefs are goal-ordered, the
    exact marginals raw-ordered) -> (39, 5000, T); cached in round7_jumps/err_both.npz"""
    path = os.path.join(R7, "err_both.npz")
    if not os.path.exists(path):
        import plot_fig1c_all as fa
        meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
        B = np.load(os.path.join(SCAN, "scan_Trained.npz"))["beliefs"]
        out = []
        for e in ENVS:
            g = int(meta["goal"][e])
            J = fa.exact_bayes(meta["obs"][e], meta["L"][e])[1]                    # (n, T, 2, R) raw order
            b = B[e].astype(np.float64)
            net = {g: b[..., :R], 1 - g: b[..., R:]}                               # goal-ordered -> raw
            out.append(np.mean([fa.fr_sym(net[v] / net[v].sum(-1, keepdims=True), J[:, :, v]) for v in (0, 1)], 0))
        np.savez(path, err_both=np.array(out, np.float32))
    with np.load(path) as f:
        return f["err_both"].astype(np.float64)


def belief_pr(agent="Trained"):
    """the effective dimension of an agent's beliefs, as the activity's: b_t its 20 read-out probabilities (the two
    marginals, each renormalized; the Optimal's exact marginals), E_k = ((b_t - mean) . w_k)^2 on the 20 principal
    directions w_k of that agent's own pooled scan beliefs (300,000 random stream-steps, seed 0), (sum E)^2 / sum E^2
    -> (39, 5000, T); cached in round7_jumps/belief_pr_<agent>.npz"""
    path = os.path.join(R7, f"belief_pr_{agent}.npz")
    if not os.path.exists(path):
        B = np.load(os.path.join(SCAN, f"scan_{agent}.npz"))["beliefs"][ENVS].astype(np.float32)
        B = np.concatenate([B[..., :R] / B[..., :R].sum(-1, keepdims=True),
                            B[..., R:] / B[..., R:].sum(-1, keepdims=True)], -1)
        flat = B.reshape(-1, 2 * R)
        sub = flat[np.random.default_rng(0).choice(len(flat), 300_000, replace=False)].astype(np.float64)
        mu = sub.mean(0)
        lam, V = np.linalg.eigh(np.cov(sub.T))
        out = np.empty(B.shape[:3], np.float32)
        for e in range(len(B)):
            E = ((B[e].astype(np.float64) - mu) @ V) ** 2
            out[e] = E.sum(-1) ** 2 / np.maximum((E ** 2).sum(-1), 1e-30)
        np.savez(path, belief_pr=out, mu=mu, V=V, lam=lam)
    with np.load(path) as f:
        return f["belief_pr"].astype(np.float64)


def data(raw=False):
    """positions and percentiles (raw=True: the raw values): cells (39, K, T, 2), (name -> (39, K, T)); streams
    (39, 5000, T, 2), (name -> ...); each environment's mean FR at T (39,)"""
    rank = (lambda A, axis=None: A) if raw else (lambda A: _rank(A))
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    assert (meta["cell_of"] == np.arange(M * R * R) // M).all()
    B = np.load(os.path.join(SCAN, "scan_Trained.npz"))["beliefs"][ENVS].astype(np.float32)        # (39, 5000, T, 20)
    r = np.arange(R, dtype=np.float32)
    X = np.stack([B[..., :R] @ r, B[..., R:] @ r], -1)                                             # (39, 5000, T, 2)
    hid = np.load(os.path.join(R7, f"scan_hidden_{NET}.npz"))
    cre = np.load(os.path.join(R7, f"scan_credit_{NET}_ns{NS}.npz"))
    rows = cre["rows"]
    E = len(ENVS)
    cell, strm = {}, {}
    rev = meta["rev"][ENVS].astype(np.float64)                                                      # (39, 5000, T)
    fr = fr_both()
    full = {"rev": rev, "err": err_both(), "fr": fr, "bpr": belief_pr()}
    for k in ("l2", "pr"):
        full[k] = np.stack([hid[f"{k}_e{e:02d}"].astype(np.float64) for e in ENVS])
    for k, A in full.items():
        cell[k] = rank(A.reshape(E, R * R, M, T).mean(2))
        strm[k] = rank(A[:, rows])
    for k in ("coh", "dW", "band_x"):
        A = np.stack([cre[f"{k}_e{e:02d}"].astype(np.float64) for e in ENVS])                     # (39, 5000, T)
        cell[k] = rank(A.reshape(E, R * R, NS, T).mean(2))
        strm[k] = rank(A)
    Xc = X.reshape(E, R * R, M, T, 2).mean(2)                                                       # (39, K, T, 2)
    return Xc, cell, X[:, rows], strm, fr[:, :, -1].mean(1)


def _blend(X, tau, axis):
    """Catmull-Rom between the steps (along axis) of X at tau (0 .. T - 1)"""
    s = int(np.clip(np.floor(tau), 0, T - 2))
    f = tau - s
    p = [np.take(X, int(np.clip(s + d, 0, T - 1)), axis=axis) for d in (-1, 0, 1, 2)]
    return .5 * ((2 * p[1]) + (-p[0] + p[2]) * f + (2 * p[0] - 5 * p[1] + 4 * p[2] - p[3]) * f ** 2
                 + (-p[0] + 3 * p[1] - 3 * p[2] + p[3]) * f ** 3)


def _lin(V, tau):
    """linear between steps of V (..., T) at tau"""
    s = int(np.floor(tau))
    if s >= T - 1:
        return V[..., T - 1]
    f = tau - s
    return (1 - f) * V[..., s] + f * V[..., s + 1]


def _grid():
    """the lattice's edges (index pairs) and positions"""
    g = np.arange(R)
    gg, cc = np.meshgrid(g, g, indexing="ij")
    idx = {(a, b): R * a + b for a in g for b in g}
    edges = np.array([(idx[(a, b)], idx[(a + da, b + db)]) for (a, b) in idx for da, db in ((1, 0), (0, 1))
                      if (a + da, b + db) in idx])
    return edges, np.stack([gg.ravel(), cc.ravel()], 1).astype(float)


def _clock(fig, W, H, X0, x1):
    ty = H - .42
    t_txt = fig.text(X0 / W, ty / H, "", fontsize=14, ha="left", va="center")
    bx0 = X0 + 1.0
    bar = fig.add_axes([bx0 / W, (ty - .06) / H, (x1 - bx0) / W, .12 / H])
    bar.set_xlim(.5, T + .5), bar.set_ylim(0, 1), bar.axis("off")
    bar.barh(.5, T, left=.5, height=1, color="0.92", lw=0)
    fill = bar.barh(.5, 0, left=.5, height=1, color="0.35", lw=0)[0]

    def set_(tau):
        t_txt.set_text(f"$t = {int(np.floor(tau + .5)) + 1}$")
        fill.set_width(tau + 1)
    return set_


def _colorbar(fig, cmap, lo, hi, rect, label):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(lo, hi), cmap=cmap), cax=cax)
    cb.set_ticks([lo, (lo + hi) / 2, hi]), cb.ax.tick_params(labelsize=8)
    cb.set_label(label, fontsize=9)
    cb.outline.set_visible(False)


def _taus():
    return list(np.linspace(0, T - 1, int((T - 1) * STEP_S * FPS) + 1)) + [T - 1.] * int(1.2 * FPS)


def _save(frames, cmap, name, levels=128):
    from main.utils import fig_path
    keys = [tuple(int(round(255 * v)) for v in cmap(u)[:3]) for u in np.linspace(0, 1, levels)]
    keys += [(int(round(v)),) * 3 for v in np.linspace(0, 255, 96)]                  # greys: text, lines, white
    quant = _Exact(np.unique(np.array(keys, np.uint8), axis=0)[:255])
    q = [quant(fr) for fr in frames]
    path = fig_path(name)
    q[0].save(path, save_all=True, append_images=q[1:], duration=1000 // FPS, loop=0)
    print(f"saved {path}: {len(q)} frames, {os.path.getsize(path) / 1e6:.1f} MB", flush=True)
    return path


def _grab(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


RC = {"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
      "font.size": 9}


def movie(mode="avg"):
    """avg, envs or particles: 3 x 3 panels, one measure each; envs_vs_typical, particles_vs_typical: the copies
    coloured by the departure from the same cell's mean over the environments at that step"""
    name_out, vs_typ = mode, mode.endswith("_vs_typical")
    mode = mode.replace("_vs_typical", "")
    Xc, cell, Xs, strm, _ = data(raw=True)
    E, n_strm = len(ENVS), Xs.shape[1]
    logs = {}
    for key, *_ in PANELS:                                         # positive, heavy-tailed measures: log10
        v = strm[key]
        logs[key] = bool((v > 0).all()) and np.percentile(v, 98) > 20 * np.percentile(v, 2)
        if logs[key]:
            cell[key], strm[key] = np.log10(cell[key]), np.log10(strm[key])
    if vs_typ:                                                     # minus the same cell's mean over the environments
        cell = {k: v - v.mean(0, keepdims=True) for k, v in cell.items()}
        cid = np.arange(n_strm) // NS
        strm = {k: v - v.reshape(E, R * R, NS, T).mean((0, 2))[cid] for k, v in strm.items()}
    edges, lattice = _grid()
    show = np.arange(Xs.shape[1]) % NS < NS_SHOW                                                   # particles drawn
    Xs, strm = Xs[:, show], {k: v[:, show] for k, v in strm.items()}
    order = np.random.default_rng(0).permutation(E * Xs.shape[1])                                  # particles' order
    cmap = plt.get_cmap("coolwarm" if vs_typ else "viridis")
    drawn = {"avg": {k: v.mean(0) for k, v in cell.items()}, "envs": cell, "particles": strm}[mode]
    lims = {}
    for key, v in drawn.items():                                   # each measure on its own fixed scale
        if vs_typ:
            L_ = np.percentile(np.abs(v), 98)
            lims[key] = (-L_, L_)
        else:
            lims[key] = tuple(np.percentile(v, [2, 98]))
    PW, GX, GY, X0, YTOP = 2.35, .35, 1.4, .45, 1.0
    W, H = X0 + 3 * PW + 2 * GX + 1.45, YTOP + 3 * PW + 2 * GY + .85
    note = {"avg": f"{E} environments,\n{n_strm:,} streams each",
            "envs": (f"all {E}\nenvironments,\n" if ENVS_SHOW == 1 else
                     f"{len(range(0, E, ENVS_SHOW))} of the {E}\nenvironments,\n") + f"{n_strm:,} streams each",
            "particles": f"{E} environments,\n{NS_SHOW * R * R} of each one's\n{n_strm:,} streams"}[mode]
    with plt.rc_context(RC):
        dpi = DPI[mode]
        fig = plt.figure(figsize=(W, H), dpi=dpi, facecolor="white")
        arts = []
        for n, (key, name, formula) in enumerate(PANELS):
            r_, c_ = divmod(n, 3)
            x, y = X0 + c_ * (PW + GX), H - YTOP - PW - r_ * (PW + GY)
            ax = fig.add_axes([x / W, y / H, PW / W, PW / H])
            ax.add_collection(LineCollection(lattice[edges], colors="0.85", linewidths=.6, linestyles=(0, (2, 2)),
                                             zorder=0))
            if mode == "avg":
                lc = ax.add_collection(LineCollection([], colors="0.55", linewidths=.7, zorder=1))
                sc = ax.scatter(*Xc[0, :, 0].T, s=26, c=cmap(np.full(R * R, .5)), lw=0, zorder=2)
                mk = [ax.plot([], [], "o" if k in (0, 99) else "s", ms=8, mfc="none", mec="k", mew=1.1, zorder=3)[0]
                      for k in (0, 99, 9, 90)]
            elif mode == "envs":
                lc = ax.add_collection(LineCollection([], linewidths=72 / dpi, antialiased=False, zorder=1))
                sc, mk = None, []
            else:
                lc = None
                sc = ax.scatter(*Xs[:, :, 0].reshape(-1, 2)[order].T, s=(72 / dpi * PX_P) ** 2, marker="s",
                                c=cmap(np.full(len(order), .5)), lw=0, zorder=2, antialiased=False)
                mk = []
            ax.set_xlim(-.6, 9.6), ax.set_ylim(-.6, 9.6), ax.set_aspect("equal")
            ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
            ax.tick_params(labelsize=8, length=2, pad=1)
            if c_ == 0:
                ax.set_ylabel(r"$E[r_c]$", labelpad=-4)
            if r_ == 2:
                ax.set_xlabel(r"$E[r_g]$", labelpad=-4)
            ax.set_title(name + "\n" + formula, fontsize=9.5, pad=4)
            lo_k, hi_k = lims[key]
            cax = fig.add_axes([x / W, (y - .42) / H, PW / W, .07 / H])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(lo_k, hi_k), cmap=cmap), cax=cax,
                              orientation="horizontal")
            cb.set_ticks([lo_k, (lo_k + hi_k) / 2, hi_k])
            cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2g}"))
            cb.ax.tick_params(labelsize=7, length=2, pad=1)
            cb.outline.set_visible(False)
            cb.set_label(("log10 " if logs[key] else "") + UNITS[key] + (" minus the cell's typical" if vs_typ else ""),
                         fontsize=7.5, labelpad=1)
            arts.append((key, lc, sc, mk))
        clock = _clock(fig, W, H, X0, X0 + 3 * PW + 2 * GX)
        fig.text((X0 + 3 * PW + 2 * GX + .2) / W, (H - .42) / H, note, fontsize=10, ha="left", va="center")
        if mode == "envs":
            Xe, ce = Xc[::ENVS_SHOW], {k: v[::ENVS_SHOW] for k, v in cell.items()}
        avg_pos = Xc.mean(0)                                                                         # (K, T, 2)
        avg_pct = {k: v.mean(0) for k, v in cell.items()}                                           # (K, T)
        frames = []
        for i, tau in enumerate(_taus()):
            if mode == "envs":
                seg = _blend(Xe, tau, 2)[:, edges].reshape(-1, 2, 2)                                # (10 x 180, 2, 2)
                by_len = np.argsort(np.linalg.norm(seg[:, 1] - seg[:, 0], axis=-1), kind="stable")
                seg = seg[by_len]
                for key, lc, _, _ in arts:
                    v = _lin(ce[key], tau)[:, edges].mean(-1).ravel()[by_len]                       # (10 x 180,)
                    lo_k, hi_k = lims[key]
                    lc.set_segments(seg)
                    lc.set_color(cmap((np.clip(v, lo_k, hi_k) - lo_k) / (hi_k - lo_k)))
                clock(tau)
                frames.append(_grab(fig))
                if i % 100 == 0:
                    print(f"{mode}: frame {i + 1}", flush=True)
                continue
            P = _blend(avg_pos, tau, 1) if mode == "avg" else _blend(Xs, tau, 2).reshape(-1, 2)[order]
            for key, lc, sc, mk in arts:
                if mode == "avg":
                    v = _lin(avg_pct[key], tau)
                    lc.set_segments(P[edges])
                    for m_, k in zip(mk, (0, 99, 9, 90)):
                        m_.set_data([P[k, 0]], [P[k, 1]])
                else:
                    v = _lin(strm[key], tau).ravel()[order]
                lo_k, hi_k = lims[key]
                sc.set_offsets(P)
                sc.set_facecolor(cmap((np.clip(v, lo_k, hi_k) - lo_k) / (hi_k - lo_k)))
            clock(tau)
            frames.append(_grab(fig))
            if i % 100 == 0:
                print(f"{mode}: frame {i + 1}", flush=True)
        plt.close(fig)
    return _save(frames, cmap, "fig1c_movie_measures%s.gif" % ("" if name_out == "avg" else "_" + name_out),
                 levels=LEVELS[mode])                                    # (< 100 MB)


if __name__ == "__main__":
    for m in sys.argv[1:] or ("avg",):
        movie(m)
