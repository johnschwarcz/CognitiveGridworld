"""The effective dimension of the beliefs, for the four agents, as particles (28 Oct 2026; John on the measures movies:
"The effective dim of beliefs are very interesting ... I wonder if we see this pattern for the ESN and Naive beliefs as
well. Maybe we can get a gif just showing the effect dim of the beliefs for the 4 agents.")
-> main/REPLICATION/figure_bin/fig1c_movie_belief_dim.gif          the alias scan (39 environments, 400 streams each)
   main/REPLICATION/figure_bin/fig1c_movie_belief_dim_harvest.gif  the harvest (John: "Can we sample fewer streams
       but many more envs? the blobs should disappear then, or is there real structure there?"): the trained network's
       gradflow harvest, its first 20,000 episodes with two distinct variables (19,607), one stream each, from 1,225
       environments (all pairs of the harvest's 50 variables); each agent's principal directions from its own harvest
       beliefs
   main/REPLICATION/figure_bin/fig1c_belief_dim_maps.png           maps(): are the blobs real?

Panels: Optimal, Trained RNN, Naive, ESN. Particles: the first NS_SHOW streams of each cell in all 39 scan environments
(15,600 per agent), each at its agent's own posterior means (E[r_g], E[r_c]) at step t = 1 .. 30 (Catmull-Rom between
steps, as the measures movies). Colour: the agent's belief effective dimension (plot_fig1c_measures_movie.belief_pr:
its 20 marginal probabilities, centred, on the principal directions of its own pooled beliefs, participation ratio
(sum E)^2 / sum E^2), raw, on one scale for the four agents (VLO - VHI: the pooled 2nd - 98th percentiles; viridis,
bright = more). (A first version ranked each stream within its environment and step; John asked whether that skewed
things: the within-environment spread is real and alike for the agents (IQR 1.6 - 2.8), but the ranking removed the
differences between environments, which are 2 - 4x larger for the Naive and the ESN; their maps also agree less from
one environment to the next (r over the 100 cells 0.31 - 0.35 vs 0.49 - 0.61 for the Optimal and the Trained RNN), so
their environments' patterns, overlaid, look like blobs.)
The blobs are real (28 Oct 2026): they stay with 1,225 environments, one stream each (the harvest movie). maps(): the
mean effective dimension in half-cell bins at step 30, for two halves of the harvest with no variable in common (the
300 pairs within each half of the 50 variables, ~4,800 episodes each): r = .95 / .95 / .92 / .95 (Optimal / Trained
RNN / Naive / ESN); 1-cell bins at t = 10, 20, 30: .96 - .98 (20 random partitions). Not the beliefs' spread: each
belief's effective dimension and its posterior SD correlate weakly (r between -0.3 and 0), and the maps with what spread
predicts removed reproduce as well. The scan's 39 environments give the harvest's map once both use one basis (r .97 -
.99; .36 - .87 on each one's own). What sets the Naive's and the ESN's blobs apart is their own principal directions:
on the Optimal's, their maps are the Optimal's (r .93, .97; on their own .55, .57). Their own directions are
concentrated (top 3: 47% and 39% of the variance; Optimal, Trained RNN 31%), and their posterior means pile up at
the corners (step 30: 1,666 and 1,121 of 19,607 beliefs within a quarter cell of a corner; Optimal 413): corner beliefs
put 73% and 61% of their departure from the mean on the top 3 directions (Optimal 22%) and read as low-dimensional
(3.8, 5.0 vs 6.8, 7.6 elsewhere); a belief's top-3 share and its dimension correlate -0.71 and -0.63 (Optimal -0.34).
So on its own directions an agent's dimension is relative to its own repertoire; for a position-by-position comparison
of the four agents, one basis. The X on the Optimal's directions (low along both diagonals) is not the figure 8: the
Optimal's directions come in equal-variance pairs (one pattern on r_g, the same on r_c) that the PCA mixes into sums and
differences with nothing in the beliefs choosing the mixing; each variable's own directions remove it
(plot_fig1c_belief_dim_basis.py).
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_belief_dim_movie.py [scan|harvest|maps]"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
import plot_fig1c_measures_movie as mm                                            # noqa: E402

AGENTS = (("Joint", "Optimal"), ("Trained", "Trained RNN"), ("Naive", "Naive"), ("Echo", "ESN"))
NS_SHOW = 4
VLO, VHI = 1.5, 10.


def _basis(B):
    """beliefs (..., 20) -> the mean, the principal directions and their variances (ascending) of 300,000 random
    ones of them (seed 0)"""
    flat = B.reshape(-1, 2 * mm.R).astype(np.float64)
    sub = flat[np.random.default_rng(0).choice(len(flat), min(300_000, len(flat)), replace=False)]
    lam, V = np.linalg.eigh(np.cov(sub.T))
    return sub.mean(0), V, lam


def _pr(B, mu, V):
    """beliefs (..., 20) -> the effective dimension on the principal directions V (as belief_pr)"""
    E = ((B.astype(np.float64) - mu) @ V) ** 2
    return E.sum(-1) ** 2 / np.maximum((E ** 2).sum(-1), 1e-30)


def harvest_beliefs(n_ep=20000):
    """the harvest's goal-ordered beliefs {agent: (n, T, 20)}, each marginal renormalized, of its first n_ep episodes
    with two distinct variables, and their variable pairs (n, 2)"""
    F = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c")
    rec = np.load(os.path.join(F, "record.npz"))
    ci, g = rec["ep_ctx_inds"][:n_ep].astype(int), rec["ep_goal_ind"][:n_ep].astype(int)
    keep = ci[:, 0] != ci[:, 1]
    ep = np.load(os.path.join(F, "episodes.npz"), mmap_mode="r")
    net = np.load(os.path.join(F, "beliefs_steps.npz"))
    out = {}
    for agent, _ in AGENTS:
        if agent in ("Joint", "Naive"):                                            # raw order -> goal-ordered
            b = np.asarray(ep["ep_joint_belief" if agent == "Joint" else "ep_naive_belief"][:n_ep], np.float32)
            b = np.concatenate([b[np.arange(n_ep), :, g], b[np.arange(n_ep), :, 1 - g]], -1)
        else:
            b = net[agent][:n_ep].astype(np.float32)
        out[agent] = np.concatenate([b[..., :mm.R] / b[..., :mm.R].sum(-1, keepdims=True),
                                     b[..., mm.R:] / b[..., mm.R:].sum(-1, keepdims=True)], -1)[keep]
    return out, ci[keep]


def _means(b):
    r = np.arange(mm.R, dtype=np.float32)
    return np.stack([b[..., :mm.R] @ r, b[..., mm.R:] @ r], -1)


def data_harvest(n_ep=20000):
    """the harvest: {agent: (positions (1, n, T, 2), raw effective dimension (1, n, T), median by step)}, the number
    of environments"""
    Bs, ci = harvest_beliefs(n_ep)
    out = {}
    for agent, _ in AGENTS:
        pr = _pr(Bs[agent], *_basis(Bs[agent])[:2])[None]
        out[agent] = (_means(Bs[agent])[None], pr, np.median(pr, (0, 1)))
    pairs = np.sort(ci, 1)
    return out, len(np.unique(pairs[:, 0] * 1000 + pairs[:, 1]))


def data():
    """{agent: (positions (39, n, T, 2) of the shown streams, raw effective dimension (39, n, T), median by step)}"""
    show = np.arange(5000) % mm.M < NS_SHOW
    r = np.arange(mm.R, dtype=np.float32)
    out = {}
    for agent, _ in AGENTS:
        B = np.load(os.path.join(mm.SCAN, f"scan_{agent}.npz"))["beliefs"][mm.ENVS][:, show].astype(np.float32)
        X = np.stack([B[..., :mm.R] @ r / B[..., :mm.R].sum(-1), B[..., mm.R:] @ r / B[..., mm.R:].sum(-1)], -1)
        pr = mm.belief_pr(agent)
        out[agent] = (X, pr[:, show], np.median(pr, (0, 1)))
    return out


def movie(source="scan"):
    if source == "harvest":
        D, n_env = data_harvest()
    else:
        D = data()
    E, n = D["Joint"][0].shape[:2]
    order = np.random.default_rng(0).permutation(E * n)
    cmap = plt.get_cmap("viridis")
    PW, GX, GY, X0, YTOP = 3.3, .45, .85, .5, 1.2
    W, H = X0 + 2 * PW + GX + 1.35, YTOP + 2 * PW + GY + .4
    with plt.rc_context(mm.RC):
        fig = plt.figure(figsize=(W, H), dpi=110, facecolor="white")
        arts = []
        for i, (agent, name) in enumerate(AGENTS):
            r_, c_ = divmod(i, 2)
            x, y = X0 + c_ * (PW + GX), H - YTOP - PW - r_ * (PW + GY)
            ax = fig.add_axes([x / W, y / H, PW / W, PW / H])
            sc = ax.scatter(*D[agent][0][:, :, 0].reshape(-1, 2)[order].T, s=(72 / 110 * mm.PX_P) ** 2, marker="s",
                            c=cmap(np.full(len(order), .5)), lw=0, antialiased=False)
            ax.set_xlim(-.6, 9.6), ax.set_ylim(-.6, 9.6), ax.set_aspect("equal")
            ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
            ax.tick_params(labelsize=8, length=2, pad=1)
            if c_ == 0:
                ax.set_ylabel(r"$E[r_c]$", labelpad=-4)
            if r_ == 1:
                ax.set_xlabel(r"$E[r_g]$", labelpad=-4)
            lev = D[agent][2]
            ax.set_title(f"{name}\nmedian level {lev[0]:.1f} (t = 1) to {lev[-1]:.1f} (t = 30)", fontsize=10, pad=4)
            arts.append((agent, sc))
        fig.text(.5, 1 - .16 / H, r"Effective dimension of the beliefs: $(\Sigma_k E_k)^2 / \Sigma_k E_k^2$, "
                 r"$E_k = ((b_t - \bar b)\, w_k)^2$ on each agent's own belief principal directions $w_k$",
                 ha="center", va="center", fontsize=10.5)
        clock = mm._clock(fig, W, H, X0, X0 + 2 * PW + GX)
        note = (f"{n:,} episodes,\none stream each,\nfrom {n_env:,}\nenvironments" if source == "harvest" else
                f"{E} environments,\n{n:,} of each one's\n5,000 streams")
        fig.text((X0 + 2 * PW + GX + .2) / W, (H - .55) / H, note, fontsize=9.5, ha="left", va="center")
        mm._colorbar(fig, cmap, VLO, VHI, [(X0 + 2 * PW + GX + .35) / W, (H - YTOP - 2 * PW - GY) / H, .14 / W,
                                           (2 * PW + GY) / H], "belief effective dimension (raw;\none scale for the "
                                                                "four agents)")
        frames = []
        for i, tau in enumerate(mm._taus()):
            for agent, sc in arts:
                X, P, _ = D[agent]
                sc.set_offsets(mm._blend(X, tau, 2).reshape(-1, 2)[order])
                sc.set_facecolor(cmap(np.clip((mm._lin(P, tau).ravel()[order] - VLO) / (VHI - VLO), 0, 1)))
            clock(tau)
            frames.append(mm._grab(fig))
            if i % 100 == 0:
                print(f"frame {i + 1}", flush=True)
        plt.close(fig)
    return mm._save(frames, cmap, "fig1c_movie_belief_dim%s.gif" % ("_harvest" if source == "harvest" else ""),
                    levels=64)


def _binmap(X, v, sel):
    """mean of v in half-cell bins of the positions X (n, 2) (centred on the nodes, the edges' midpoints and the
    squares' centres) over the rows sel -> (19, 19) map, counts"""
    k = np.clip(np.rint(2 * X[sel]).astype(int), 0, 2 * mm.R - 2)
    i = k[:, 0] * (2 * mm.R - 1) + k[:, 1]
    c = np.bincount(i, minlength=(2 * mm.R - 1) ** 2)
    return (np.bincount(i, v[sel], minlength=(2 * mm.R - 1) ** 2) / np.maximum(c, 1)).reshape(2 * mm.R - 1, -1), \
        c.reshape(2 * mm.R - 1, -1)


def maps(t=30, min_n=5):
    """the check figure: are the blobs real? -> fig1c_belief_dim_maps.png"""
    from main.utils import fig_path
    Bs, ci = harvest_beliefs()
    n_env = lambda sel: len(np.unique(np.sort(ci[sel], 1) @ [1000, 1]))           # noqa: E731
    var = np.unique(ci)
    grpA = np.isin(ci, np.random.default_rng(0).permutation(var)[:len(var) // 2])
    every = np.ones(len(ci), bool)
    sels = (grpA.all(1), (~grpA).all(1), every, every)                            # A, B: no variable in common
    mu0, V0, _ = _basis(Bs["Joint"])
    M, C, rAB, top3 = {}, {}, {}, {}
    for agent, _ in AGENTS:
        b = Bs[agent][:, t - 1]
        X = _means(b)
        mu, V, lam = _basis(Bs[agent])
        top3[agent] = lam[::-1][:3].sum() / lam.sum()
        own, opt = _pr(b, mu, V), _pr(b, mu0, V0)
        for j, sel in enumerate(sels):
            M[agent, j], C[agent, j] = _binmap(X, opt if j == 3 else own, sel)
        ok = (C[agent, 0] >= min_n) & (C[agent, 1] >= min_n)
        rAB[agent] = np.corrcoef(M[agent, 0][ok], M[agent, 1][ok])[0, 1]
    lo, hi = np.percentile(np.concatenate([M[k][C[k] >= min_n] for k in M]), [2, 98])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    PW, GX, GY, X0, YTOP = 2.45, .28, .62, 1.45, 1.85
    W, H = X0 + 4 * PW + 3 * GX + 1.25, YTOP + 4 * PW + 3 * GY + .45
    with plt.rc_context(mm.RC):
        fig = plt.figure(figsize=(W, H), dpi=200, facecolor="white")
        fig.text(.5, 1 - .2 / H, f"Are the blobs real?  Belief effective dimension at step {t}, the harvest (one "
                 "stream per episode): mean over the beliefs in half-cell bins of their posterior means (white: fewer "
                 f"than {min_n})", ha="center", va="center", fontsize=11)
        heads = (f"variables of group A only:\n{n_env(sels[0])} environments, {sels[0].sum():,} episodes",
                 f"variables of group B only:\n{n_env(sels[1])} environments, {sels[1].sum():,} episodes",
                 f"all {n_env(every):,} environments,\n{len(ci):,} episodes",
                 f"all {n_env(every):,} environments,\non the Optimal's principal directions")
        for c_, h in enumerate(heads):
            fig.text((X0 + c_ * (PW + GX) + PW / 2) / W, 1 - 1.12 / H, h, ha="center", va="center", fontsize=9.5)
        for x0, x1, h in ((0, 2, "each agent's own principal directions (as the movie)"),
                          (3, 3, "one basis for the four agents")):
            xa, xb = X0 + x0 * (PW + GX), X0 + x1 * (PW + GX) + PW
            fig.text((xa + xb) / 2 / W, 1 - .62 / H, h, ha="center", va="center", fontsize=10)
            fig.add_artist(plt.Line2D([(xa + .1) / W, (xb - .1) / W], [1 - .78 / H] * 2, color="0.4", lw=.8))
        for r_, (agent, name) in enumerate(AGENTS):
            y = H - YTOP - PW - r_ * (PW + GY)
            for c_ in range(4):
                ax = fig.add_axes([(X0 + c_ * (PW + GX)) / W, y / H, PW / W, PW / H])
                A = np.ma.masked_where(C[agent, c_] < min_n, M[agent, c_])
                im = ax.imshow(A.T, origin="lower", extent=(-.25, 9.25, -.25, 9.25), cmap=cmap, vmin=lo, vmax=hi,
                               interpolation="nearest")
                ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
                ax.tick_params(labelsize=8, length=2, pad=1)
                if c_ == 0:
                    ax.set_ylabel(r"$E[r_c]$", labelpad=-4)
                if r_ == 3:
                    ax.set_xlabel(r"$E[r_g]$", labelpad=-4)
                if c_ >= 2 and agent != "Joint":
                    ok = (C[agent, c_] >= min_n) & (C["Joint", c_] >= min_n)
                    rr = np.corrcoef(M[agent, c_][ok], M["Joint", c_][ok])[0, 1]
                    ax.set_title(f"r with the Optimal's map {rr:.2f}", fontsize=9, pad=3)
                    print(f"{name:12s} column {c_ + 1}: r with the Optimal's map {rr:.2f} ({ok.sum()} bins)")
            fig.text((X0 - .62) / W, (y + PW / 2) / H, f"{name}\n(top 3 directions:\n{100 * top3[agent]:.0f}% of the "
                     "variance)", rotation=90, ha="center", va="center", fontsize=10)
            fig.text((X0 + PW + GX / 2) / W, (y + PW + .14) / H, f"group A vs group B: r = {rAB[agent]:.2f}",
                     ha="center", va="center", fontsize=9)
            print(f"{name:12s} A vs B r {rAB[agent]:.2f}; top-3 variance {top3[agent]:.2f}")
        cax = fig.add_axes([(X0 + 4 * PW + 3 * GX + .3) / W, (H - YTOP - 4 * PW - 3 * GY) / H, .14 / W,
                            (4 * PW + 3 * GY) / H])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("belief effective dimension (raw)", fontsize=10)
        cb.ax.tick_params(labelsize=8)
        path = fig_path("fig1c_belief_dim_maps.png")
        fig.savefig(path)
        plt.close(fig)
    return path


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "scan"
    print(maps() if arg == "maps" else movie(arg))
