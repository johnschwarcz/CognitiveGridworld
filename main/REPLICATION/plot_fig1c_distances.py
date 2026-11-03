"""Two figures on how the factorized agents' representation fails, and how factorization regret drives it (panel (c)
analyses of Figure 1, 22-25 Oct 2026), and the data behind them.

    fig1c_pair_distances.png   distance between two particles: estimates vs targets (posterior means and argmax)
    fig1c_fr_distance.png      how a cell's factorization regret shapes convergence to the true cell (3-D)

Run from the repository root (headless):
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/plot_fig1c_distances.py
        plots from the stored data, building any data file that is missing (about 30 s when both exist)
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/plot_fig1c_distances.py --rebuild all --workers 6
        recreates both data files first: fr (about 1 min, CPU) and pairs (network replays; about 15 CPU-minutes
        per worker for 110 environments, so about 5 min with 6 workers on an idle machine)
    --rebuild fr | pairs   recreates one of them;  --n-env N   environments for the pairs data (default 110)
-> figures in main/REPLICATION/figure_bin/; data in main/DATA/experimental/fig1c_compression/figdata/round4b/

Terms: the world has 500 variables; an environment is a pair of active variables (goal variable, value r_g; contextual
variable, value r_c), whose Z sets the joint likelihood table; a cell is (r_g, r_c) (100 cells); a stream is one
30-step episode from one cell; a particle is a stream's belief after step t; its estimated position is the posterior
means (E[r_g], E[r_c]) (or the argmax of each marginal). Agents: Optimal (joint Bayes), Trained RNN, Naive
(factorized Bayes: the table averaged over the other variable), ESN (frozen LSTM with trained read-in and read-out).

Data
  fig1c_distances_fr.npz     build_fr_data(), from the alias scan main/DATA/experimental/fig1c_compression (README.txt
                             there: 40 environments x 100 cells x 50 streams x 30 steps, all four agents on the same
                             streams). Per scan environment (e = 2, an exactly symmetric table, left out), cell and step:
                             the mean over the cell's 50 particles of the Euclidean distance of the estimated position
                             from the true cell (scan beliefs, float16, each marginal renormalized). Each cell's FR =
                             mean over its streams of FR_g + FR_c at T = 30, Jeffreys/2 between the Optimal's and the
                             Naive's marginal of each variable (clip 1e-15), recomputed exactly (float64) from the scan's
                             observations and tables. Cells ranked by FR within their environment into 10 and 20
                             equal-count bins (stable sort).
                             keys: dist_t{10,20} (4 agents, 39 environments, nbin, 30 steps), fr_median{10,20} (nbin,)
                             (median over environments of each bin's median FR, nats), FRb (39, 100), agents
  fig1c_distances_pairs.npz  build_pair_data(), from the trained network's gradflow harvest (main/DATA/cache/
                             gradflow_trained.npz): n_env environments evenly spaced over the 1225 distinct pairs of the
                             50 held-out variables (one harvest episode per pair: its table, goal variable and Z). Per
                             environment all 100 cells x 10 streams (observations drawn with seed 7,000,000 + episode),
                             the Optimal and the Naive exact, the Trained RNN and the ESN replayed on the same streams
                             (fig1._beliefs). For every one of the 499,500 pairs of streams and every step: the
                             Euclidean distance between the two estimates, averaged per discrete true distance between
                             their cells (51 values).
                             Reproducible bit for bit at a fixed thread count (default 1 per worker); other thread
                             counts change the networks' float32 rounding, which can flip a few argmax ties (changes
                             of about 1e-3 in the argmax means).
                             keys: ymean (n_env, 4 agents, 2 estimators [posterior means, argmax], 30 steps, 51),
                             n (51,) pairs per true distance, xs (51,) the true distances, eps (n_env,) harvest episodes,
                             agents, estimators"""
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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap             # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from main.utils import apply_fig_style, fig_path                                  # noqa: E402

FS = apply_fig_style()
plt.rcParams["mathtext.fontset"] = "cm"                                           # math in Computer Modern, as the text
SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
FD = os.path.join(SCAN, "figdata", "round4b")
FR_DATA = os.path.join(FD, "fig1c_distances_fr.npz")
PAIR_DATA = os.path.join(FD, "fig1c_distances_pairs.npz")
AG = ("Joint", "Trained", "Naive", "Echo")
AGENT = {"Joint": ("Optimal", "k"), "Trained": ("Trained RNN", "#1f77b4"), "Naive": ("Naive", "0.5"),
         "Echo": ("ESN", "#ff7f0e")}
NT = 30
T = np.arange(1, NT + 1)
R = np.arange(10, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════
# data
# ═══════════════════════════════════════════════════════════════════

def _jeffreys2(p, q):
    p, q = np.clip(p, 1e-15, None), np.clip(q, 1e-15, None)
    return .5 * ((p - q) * (np.log(p) - np.log(q))).sum(-1)


def _fr_both_T(L, goal, cells, obs):
    """per stream: FR_g + FR_c after all NT steps, exact (float64) from the counts. L (5, 10, 10) raw order, goal (0/1),
    cells (100, 2) goal-ordered (r_g, r_c), obs (n, NT, 5) bool"""
    r0, r1 = (cells[:, 1], cells[:, 0]) if goal == 1 else (cells[:, 0], cells[:, 1])
    P = np.clip(L[:, r0, r1].T, 1e-12, 1 - 1e-12)                                  # (100 cells, 5) goal-ordered
    n = obs.sum(1).astype(np.float64)                                              # (n, 5) counts at T
    ll = n @ np.log(P).T + (NT - n) @ np.log(1 - P).T
    post = np.exp(ll - ll.max(1, keepdims=True))
    post = (post / post.sum(1, keepdims=True)).reshape(-1, 10, 10)                 # [r_g, r_c]
    fr = 0.
    for opt, Pv in ((post.sum(2), P.reshape(10, 10, 5).mean(1)), (post.sum(1), P.reshape(10, 10, 5).mean(0))):
        l = n @ np.log(Pv).T + (NT - n) @ np.log(1 - Pv).T                         # the Naive: averaged table
        q = np.exp(l - l.max(1, keepdims=True))
        fr = fr + _jeffreys2(opt, q / q.sum(1, keepdims=True))
    return fr


def build_fr_data():
    """the cells' distance from the true cell over t, binned by the cells' FR (see the module docstring)"""
    m = np.load(os.path.join(SCAN, "scan_meta.npz"))
    cells, cell_of, L, goal, obs = m["cells"], m["cell_of"], m["L"], m["goal"], m["obs"]
    n_env, n = obs.shape[:2]
    assert (cell_of == np.arange(n) // (n // 100)).all()                           # streams contiguous per cell
    pos = cells[cell_of].astype(np.float64)                                        # (n, 2) true (r_g, r_c)
    DISP = np.zeros((len(AG), n_env, 100, NT))
    for a_, a in enumerate(AG):
        B = np.load(os.path.join(SCAN, f"scan_{a}.npz"))["beliefs"]                # (40, n, NT, 20) float16
        for e in range(n_env):
            b = B[e].astype(np.float64).reshape(n, NT, 2, 10)
            E = (b * R).sum(-1) / b.sum(-1)                                        # (n, NT, 2) posterior means
            DISP[a_, e] = np.linalg.norm(E - pos[:, None], axis=-1).reshape(100, -1, NT).mean(1)
        del B
    FRb = np.stack([_fr_both_T(L[e], int(goal[e]), cells, obs[e]).reshape(100, -1).mean(1) for e in range(n_env)])
    keep = np.arange(n_env) != 2
    DISP, FRb = DISP[:, keep], FRb[keep]
    out = dict(FRb=FRb, agents=np.array(AG))
    for nbin in (10, 20):
        b = np.zeros_like(FRb, dtype=int)
        for e in range(len(FRb)):
            b[e, np.argsort(FRb[e], kind="stable")] = np.arange(100) * nbin // 100
        out[f"dist_t{nbin}"] = np.stack([np.stack([DISP[:, e, b[e] == q].mean(1) for q in range(nbin)], 1)
                                         for e in range(len(FRb))], 1)             # (4, 39, nbin, NT)
        out[f"fr_median{nbin}"] = np.median([[np.median(FRb[e][b[e] == q]) for q in range(nbin)]
                                             for e in range(len(FRb))], 0)
    np.savez(FR_DATA, **out)
    print("saved", FR_DATA)


M_STREAMS = 10                                                                     # streams per cell, pairs data
SQ = np.unique([a * a + b * b for a in range(10) for b in range(10)])               # squared true distances
SQ_IDX = np.full(SQ.max() + 1, -1)
SQ_IDX[SQ] = np.arange(len(SQ))
CELLS = np.array([(rg, rc) for rg in range(10) for rc in range(10)])               # goal-ordered


def _fig1():
    sys.path.insert(0, os.path.join(ROOT, "main", "REPLICATION", "claude_plots"))
    import fig1
    return fig1


def _environments(f):
    """one harvest episode per distinct unordered pair of different active variables (1225 of the 50 held-out ones)"""
    ci = np.sort(f["ep_ctx_inds"].astype(int), 1)
    seen, eps = set(), []
    for ep, (a, b) in enumerate(ci):
        if a != b and (a, b) not in seen:
            seen.add((a, b)), eps.append(ep)
    return np.array(eps)


def _simulate(f, EP, cf):
    """exact Optimal / Naive marginals (n, NT, 2, 10, raw variable order) and the replay inputs of environment EP"""
    g = int(f["ep_goal_ind"][EP])
    L = np.asarray(f["ep_joint_likelihood"][EP], np.float64)
    cell_of = np.repeat(np.arange(100), M_STREAMS)
    r01 = [(rc, rg) if g == 1 else (rg, rc) for rg, rc in CELLS]
    obs = np.random.default_rng(7_000_000 + EP).random((len(cell_of), NT, len(L))) < \
        np.stack([L[:, a, b] for a, b in r01])[cell_of][:, None]
    o = obs.astype(np.float64)
    cum = np.cumsum(np.einsum("nti,ijk->ntjk", o, np.log(L)) + np.einsum("nti,ijk->ntjk", 1 - o, np.log1p(-L)), 1)
    post = np.exp(cum - cum.max((-1, -2), keepdims=True))
    post /= post.sum((-1, -2), keepdims=True)
    Lb = (L.mean(2), L.mean(1))
    naive = np.cumprod(np.stack([np.exp(o @ np.log(Lb[k]) + (1 - o) @ np.log1p(-Lb[k])) for k in (0, 1)], 2), 1)
    B = {"Joint": np.stack([post.sum(-1), post.sum(-2)], 2), "Naive": naive / naive.sum(-1, keepdims=True)}
    D = {"ep_obs_flat": obs.astype(f["ep_obs_flat"].dtype)}
    for k in cf.REPLAY[1:]:
        D[k] = np.repeat(f[k][EP:EP + 1], len(cell_of), 0)
    return B, D, g, cell_of


def _pair_sums(B, g, cell_of):
    """sums of the estimate distance and pair counts per discrete true distance: (4 agents, 2 estimators, NT, 51)"""
    pos = CELLS[cell_of]
    iu = np.triu_indices(len(cell_of), 1)
    q = SQ_IDX[((pos[iu[0]] - pos[iu[1]]) ** 2).sum(1)]
    cnt = np.bincount(q, minlength=len(SQ)).astype(np.float64)
    S = np.zeros((len(AG), 2, NT, len(SQ)))
    for a_, a in enumerate(AG):
        b = B[a].reshape(len(cell_of), NT, 2, 10)
        b = b / b.sum(-1, keepdims=True)
        b = np.stack([b[:, :, g], b[:, :, 1 - g]], 2)                              # goal-ordered
        for e_, E in enumerate(((b * R).sum(-1), b.argmax(-1).astype(np.float64))):  # posterior means, argmax
            for s in range(NT):
                S[a_, e_, s] = np.bincount(q, weights=np.sqrt(((E[iu[0], s] - E[iu[1], s]) ** 2).sum(1)),
                                           minlength=len(SQ))
    return S, cnt


def _pairs_worker(eps):
    """the pair sums of the environments eps (episode indices): simulate, replay the networks, sum"""
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
    cf = _fig1()
    f = cf._harvest("trained", ["ep_goal_ind", "ep_joint_likelihood", "ep_ctx_inds"] + list(cf.REPLAY))
    S_all, cnt = [], None
    for chunk in np.array_split(np.asarray(eps), max(1, len(eps) // 20)):          # 20 environments per replay call
        sims = [_simulate(f, EP, cf) for EP in chunk]
        D = {k: np.concatenate([s_[1][k] for s_ in sims]) for k in sims[0][1]}
        nets = {("Trained" if net == "trained" else "Echo"):
                cf._beliefs(net, D, len(chunk), check=False, steps=range(NT)).reshape(len(chunk), -1, NT, 20)
                for net in ("trained", "echo")}
        for c_, (B, _, g, cell_of) in enumerate(sims):
            S, cnt = _pair_sums({"Joint": B["Joint"].reshape(-1, NT, 20), "Naive": B["Naive"].reshape(-1, NT, 20),
                                 "Trained": nets["Trained"][c_], "Echo": nets["Echo"][c_]}, g, cell_of)
            S_all.append(S)
    return np.array(S_all), cnt


def build_pair_data(n_env=110, workers=1):
    """the estimate distance of every pair of particles per true distance (see the module docstring)"""
    cf = _fig1()
    f = cf._harvest("trained", ["ep_ctx_inds"])
    all_eps = _environments(f)
    eps = all_eps[np.unique(np.linspace(0, len(all_eps) - 1, n_env).round().astype(int))]
    print(f"pairs data: {len(eps)} of {len(all_eps)} environments, {workers} worker(s)", flush=True)
    parts = [p for p in np.array_split(eps, workers) if len(p)]
    if workers > 1:
        import multiprocessing as mp
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        with mp.get_context("fork").Pool(len(parts)) as pool:                      # torch is imported only in workers
            res = pool.map(_pairs_worker, parts)
    else:
        res = [_pairs_worker(p) for p in parts]
    S = np.concatenate([r[0] for r in res])
    cnt = res[0][1]
    np.savez(PAIR_DATA, ymean=(S / cnt).astype(np.float32), n=cnt, xs=np.sqrt(SQ), eps=eps, agents=np.array(AG),
             estimators=np.array(["mean", "argmax"]))
    print("saved", PAIR_DATA, (S / cnt).shape)


# ═══════════════════════════════════════════════════════════════════
# figures
# ═══════════════════════════════════════════════════════════════════

def title2(ax, name, formula):
    """the panel's quantity: its name over its formula (house rule), centred over the axes"""
    ax.annotate(name, (.5, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points",
                ha="center", va="baseline", fontsize=FS["label"] + .5)
    ax.annotate(formula, (.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                ha="center", va="baseline", fontsize=FS["tick"], color="0.25")


def letter(fig, ax, s, dx=.62):
    """the panel letter on the baseline of the title's name line, dx inches left of the axes"""
    W, H = fig.get_size_inches()
    x0, y0, w, h = ax.get_position().bounds
    fig.text(x0 - dx / W, y0 + h + 17 / 72 / H, r"$\mathbf{%s}$" % s, fontsize=FS["label"] + 2, ha="left",
             va="baseline")


def draw_pairs_row(fig, x0, y0, sq, gx, e_, Z=None, letter_s=None, ncol=4, gy=.8, formula=None):
    """one row of the pair-distance figure (e_ 0: posterior means, 1: argmax): the four agents' panels (sq inches
    square, gx apart) with the lower-left corner at (x0, y0) inches, the header (name over formula) above them, the
    step colour bar to their right. ncol: panels per row (2: a 2 x 2 block, rows gy apart, (x0, y0) its lower-left
    corner); formula: the header's second line (None: the standalone figure's)"""
    W, H = fig.get_size_inches()
    Z = np.load(PAIR_DATA) if Z is None else Z
    Y, xs, ag = Z["ymean"].astype(np.float64), Z["xs"], list(Z["agents"])           # Y (E, 4, 2, T, K)
    nE = len(Y)
    vir = plt.get_cmap("viridis")
    est, lab = (("mean", "posterior means"), ("argmax", "argmax"))[e_]
    nrow = -(-len(AG) // ncol)
    top = y0 + nrow * sq + (nrow - 1) * gy                                           # the block's top
    for k, a in enumerate(AG):
        r_, c_k = divmod(k, ncol)
        ax = fig.add_axes([(x0 + c_k * (sq + gx)) / W, (y0 + (nrow - 1 - r_) * (sq + gy)) / H, sq / W, sq / H])
        Ya = Y[:, ag.index(a), e_]                                                   # (E, T, K)
        M, SE = Ya.mean(0), Ya.std(0, ddof=1) / np.sqrt(nE)
        for s_ in range(NT):
            c_ = vir(s_ / (NT - 1))
            ax.vlines(xs, M[s_] - SE[s_], M[s_] + SE[s_], color=c_, lw=.5, alpha=.7, zorder=2)
            ax.plot(xs, M[s_], c=c_, lw=1.0, zorder=3)
        ax.plot([0, 13], [0, 13], c="k", ls="--", lw=.9, zorder=4)
        ax.set_xlim(0, 13), ax.set_ylim(0, 13)
        ax.set_xticks([0, 4, 8, 12]), ax.set_yticks([0, 4, 8, 12])
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(AGENT[a][0], fontsize=FS["label"] - .5, color=AGENT[a][1] if a != "Naive" else "0.35", pad=3)
        ax.set_xlabel("True distance", labelpad=1)
        if c_k == 0:
            ax.set_ylabel("Estimate distance", labelpad=1)
        if k == 0:
            hdr = fig.add_axes([x0 / W, (top + .38) / H, (ncol * sq + (ncol - 1) * gx) / W, .01 / H])
            hdr.axis("off")
            title2(hdr, f"Distance between two particles: estimates ({lab}) vs targets", formula or (
                   (r"$\|\hat r_i - \hat r_j\|$, $\hat r = (E[r_g], E[r_c])$" if est == "mean" else
                    r"$\|\hat r_i - \hat r_j\|$, $\hat r = (\mathrm{argmax}\, b_g, \mathrm{argmax}\, b_c)$") +
                   r" against the true $\|r_i - r_j\|$; one curve per step; %d environments, SEM bars" % nE))
            if letter_s:
                letter(fig, hdr, letter_s, dx=.62)
    cax = fig.add_axes([(x0 + ncol * sq + (ncol - 1) * gx + .2) / W, (y0 + .1) / H, .1 / W, (top - y0 - .2) / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=vir, norm=plt.Normalize(1, NT)), cax=cax, ticks=[1, 10, 20, 30])
    cb.ax.tick_params(labelsize=FS["tick"] - 1, length=2, pad=1)
    cb.set_label("step $t$", fontsize=FS["tick"] - .5, labelpad=2)


def fig_pair_distances():
    """Distance between two particles, estimates vs targets. For every pair of streams of an environment: x = the
    Euclidean distance between their true cells (51 discrete values), y = the Euclidean distance between the agent's
    estimates of the two particles; one curve per step t (colour), mean over environments, SEM bars over
    environments, dashed = the target y = x. (a) estimates = posterior means (E[r_g], E[r_c]); (b) estimates = the
    argmax of each variable's marginal."""
    Z = np.load(PAIR_DATA)
    sq, gx, gy, top = 1.85, .4, 1.45, 1.0
    W, H = .8 + 4 * sq + 3 * gx + .9, top + 2 * sq + gy + .6
    fig = plt.figure(figsize=(W, H))
    for e_ in range(2):
        draw_pairs_row(fig, .8, H - top - (e_ + 1) * sq - e_ * gy, sq, gx, e_, Z, letter_s="ab"[e_])
    path = fig_path("fig1c_pair_distances.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def draw_fr_3d(fig, rect, cax_rect, hdr_rect, legend_xy, nbin=20, elev=22, azim=-58, letter_s=None, zoom=1.2,
               formula=None, label_xz=((-22, 6.15), (19, 1.75))):
    """the 3-D panel of fig_fr_distance drawn into fig: rect, cax_rect (its colour bar; None: none, the FR axis
    carries the values), hdr_rect (the header's line) in inches (x, y, w, h); legend_xy the legend's upper-left corner
    in inches (None: no legend); zoom the 3-D box's; formula: the header's second line (None: the standalone
    figure's); label_xz: the (step axis, height) positions of the two halves' labels (Naive and ESN, Optimal and
    Trained RNN)"""
    W, H = fig.get_size_inches()
    inch = lambda r: [r[0] / W, r[1] / H, r[2] / W, r[3] / H]
    Z = np.load(FR_DATA)
    ag, D_t, frm = list(Z["agents"]), Z[f"dist_t{nbin}"], Z[f"fr_median{nbin}"]
    M = D_t.mean(1)                                                                  # (4, nbin, NT)
    cm = LinearSegmentedColormap.from_list("fr", ["#8c8c8c", "#f29b5e", "#d7301f", "#67000d"])
    col = [cm(q / (nbin - 1)) for q in range(nbin)]
    side = {"Joint": 1, "Trained": 1, "Naive": -1, "Echo": -1}                       # +: right half, -: left half
    theory, net = dict(lw=2.2, alpha=.85, ls=(0, (2.8, 1.4))), dict(lw=1.0, ls="-")
    dark = lambda c: tuple(np.array(c[:3]) * .75) + (1.,)                            # networks slightly darker
    sty = {"Joint": theory, "Naive": theory, "Trained": net, "Echo": net}
    ticks = [1, round(1 + (nbin - 1) / 3), round(1 + 2 * (nbin - 1) / 3), nbin]
    tlab = ["%.2g" % frm[k - 1] if frm[k - 1] < 10 else "%.0f" % frm[k - 1] for k in ticks]
    ax = fig.add_axes(inch(rect), projection="3d")
    ax.computed_zorder = False
    x = T - 1.                                                                       # 0 at t = 1
    for a in ("Naive", "Joint", "Echo", "Trained"):                                  # theory first, networks on top
        for q in range(nbin):
            o_ = slice(None, None, side[a])                                          # left half drawn from t = 30
            ax.plot((side[a] * x)[o_], np.full(NT, q + 1), M[ag.index(a), q][o_], clip_on=False,  # 3-D axes clip
                    c=dark(col[q]) if a in ("Trained", "Echo") else col[q],                       # to a square box
                    zorder=(3 if a in ("Naive", "Joint") else 5), **sty[a])
    ax.plot([0, 0], [.5, nbin + .5], [0, 0], c="0.55", lw=.8, zorder=2)             # the t = 1 seam on the floor
    ax.set_xlim(-29.5, 29.5), ax.set_ylim(.5, nbin + .5), ax.set_zlim(0, 6)
    ax.set_xticks([-29, -19, -9, 0, 9, 19, 29], ["30", "20", "10", "1", "10", "20", "30"])
    ax.set_zticks([0, 2, 4, 6])
    ax.set_yticks(ticks, tlab)
    ax.tick_params(labelsize=FS["tick"] - 1, pad=0)
    ax.set_xlabel(r"Step $t$ (mirrored at $t = 1$)", fontsize=FS["tick"] + .5, labelpad=0)
    ax.set_ylabel("Cell's FR (nats; %d bins)" % nbin, fontsize=FS["tick"] + .5, labelpad=-1)
    ax.set_zlabel("Distance from the true cell", fontsize=FS["tick"] + .5, labelpad=-3)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0)), pane.set_edgecolor("0.85")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(color="0.9", linewidth=.4)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((8, 4, 3), zoom=zoom)
    kw = dict(fontsize=FS["label"], ha="center", va="bottom", zorder=10)
    ax.text(label_xz[0][0], nbin + .6, label_xz[0][1], "Naive and ESN", color="0.3", **kw)
    ax.text(label_xz[1][0], nbin + .6, label_xz[1][1], "Optimal and Trained RNN", color="k", **kw)
    if legend_xy is not None:
        fig.legend(handles=[Line2D([], [], c="0.35", label="Theory", **theory),
                            Line2D([], [], c="0.35", label="Networks", **net)],
                   frameon=False, fontsize=FS["label"], loc="upper left",
                   bbox_to_anchor=(legend_xy[0] / W, legend_xy[1] / H), handlelength=2.6, labelspacing=.3)
    if cax_rect is not None:
        cax = fig.add_axes(inch(cax_rect))
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=ListedColormap(col), norm=plt.Normalize(.5, nbin + .5)), cax=cax,
                          ticks=ticks)
        cb.ax.set_yticklabels(tlab)
        cb.ax.tick_params(labelsize=FS["tick"] - 1, length=2, pad=1)
        cb.set_label("line colour: the cell's FR (median of the bin, nats)", fontsize=FS["tick"] - .5, labelpad=2)
    hdr = fig.add_axes(inch(hdr_rect))
    hdr.axis("off")
    title2(hdr, "How a cell's factorization regret shapes convergence to the true cell", formula or (
           r"$\langle \|(E[r_g], E[r_c]) - (r_g, r_c)\| \rangle_{\mathrm{particles}}$, one line per agent and FR bin of "
           r"the cell; time mirrored at $t = 1$; mean over 39 environments"))
    if letter_s:
        letter(fig, hdr, letter_s, dx=.62)
    return ax


def fig_fr_distance(nbin=20, elev=22, azim=-58):
    """How a cell's factorization regret shapes convergence to the true cell (one 3-D panel, time mirrored at t = 1).
    Distance = mean over the cell's 50 particles of the Euclidean distance of the estimated position from the true
    cell (grid units; per particle, so split clouds do not cancel); cells binned by their FR within each environment
    (nbin 20: 5 of its 100 cells per bin; 10: deciles). The Optimal and the Trained RNN run from t = 1 to 30 to the
    right, the Naive and the ESN from t = 1 to 30 to the left, so each bin's curves meet at t = 1, where all agents
    start alike. Colour = the cell's FR bin; dashed = theory (Naive left, Optimal right), thin solid, slightly
    darkened = networks (ESN left, Trained right), drawn on top of each other. Means over the 39 scan environments."""
    aw = 9.0
    W, H = .2 + aw + 1.0, 5.6
    fig = plt.figure(figsize=(W, H))
    draw_fr_3d(fig, (.2, -.25, aw, H - .3), (.2 + aw + .25, 1.0, .12, 3.2), (.2, H - .35, aw, .01), (.4, H - .5),
               nbin, elev, azim)
    path = fig_path("fig1c_fr_distance.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser(description="fig1c_pair_distances.png and fig1c_fr_distance.png, and their data")
    pa.add_argument("--rebuild", choices=("all", "fr", "pairs"), default=None, help="recreate data before plotting")
    pa.add_argument("--workers", type=int, default=1, help="worker processes for the pairs data")
    pa.add_argument("--n-env", type=int, default=110, help="environments for the pairs data")
    args, _ = pa.parse_known_args()                                                 # tolerant of IDE kernel args
    if args.rebuild in ("all", "fr") or not os.path.exists(FR_DATA):
        build_fr_data()
    if args.rebuild in ("all", "pairs") or not os.path.exists(PAIR_DATA):
        build_pair_data(args.n_env, args.workers)
    print("saved", fig_pair_distances())
    print("saved", fig_fr_distance())
