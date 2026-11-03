"""The read-out plane lifted by each particle's own path covariance (28 Oct 2026; John on the first version, which had
panels at four steps: "show single panels with full trajectories colored by cov (red to blue) and alpha-ed to make
0 cov transparent. That should show our highways above the rest (maybe show 2 angles, one from above and one from
below)"; then "I want to see raw data", and on lifting by the Optimal's covariance: "the height being the optimal
covariance is problematic. The trajectory has a slope, cant we use the slope of the trajectory? Maybe we need to
smoothen over time to remove the jitter or something better?"; then "add the naive ... smoothen the actual trajectories
with a guassian window ... 1 stream per cell and make the trajectories thicker")
-> main/REPLICATION/figure_bin/fig1c_lifted_plane.png

Each line: one particle's (stream's) path over steps 1-30, smoothed by a Gaussian window over steps (weights
w_ts = exp(-(s - t)^2 / 2 SIGMA^2), SIGMA = 1.5 steps, renormalized at the episode's ends): drawn at the window's
weighted mean of its posterior means (E[r_g], E[r_c]) and lifted by the window's weighted covariance of them,
C_t = sum_s w_ts (E[r_g]_s - mean_t)(E[r_c]_s - mean_t) (cells^2). Its sign is the path's slope (+ while moving along
the (0, 0) - (9, 9) diagonal, - along (0, 9) - (9, 0)); its size grows with how far the particle moves along that
diagonal around t, so jitter and standing still stay near 0. The slope itself (dE[r_c] / dE[r_g]) would give a tiny step
the same weight as a jump and diverge on vertical steps. Computed the same way for every agent from its own read-out;
nothing is decoded and nothing comes from another agent.
Each segment (step t to t + 1) takes its two ends' mean C as colour (red +, blue -) and opacity A_MAX (|C| / CMAX)^GAMMA
(0: invisible); heights clipped at +-(CMAX + 1); segments drawn far to near. Columns: the agents (the Naive next to the
ESN); rows: seen from above (elevation ELEV) and from below (-ELEV), orthographic, so the scale bar holds at every
depth. Data: the alias scan's 39 environments x N_PER_CELL stream per cell (3,900 paths per agent).
numbers() prints (all 39 x 5000 streams, 30 steps; measured 28 Oct, SIGMA 1.5): |C| percentiles 90/99/99.9: Optimal
0.15 / 1.60 / 4.35 cells^2, Trained RNN 0.15 / 1.60 / 4.40, Naive 0.13 / 0.71 / 1.73, ESN 0.11 / 0.71 / 1.68; against
the Optimal's C on the same stream: Trained RNN r = 0.97, Naive 0.21, ESN 0.18; the aggregate maps (mean C by the
smoothed position's 1-cell bin, over streams and steps): r with the Optimal's 0.99 / 0.84 / 0.86, RMS 0.94 / 0.77 /
0.75 of the Optimal's; ESN against the Naive: r = 0.73 per stream, 0.98 for the maps (the ESN moves like the Naive).
The Optimal's belief covariance (for the comparison with its C) is computed once into
main/DATA/experimental/fig1c_compression/figdata/round7_jumps/cov_paths.npz (about 30 s; delete it to recompute).
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_lifted_plane.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402

SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
R7 = os.path.join(SCAN, "figdata", "round7_jumps")
CACHE = os.path.join(R7, "cov_paths.npz")
ENVS = [e for e in range(40) if e != 2]
R, T = 10, 30
AGENTS = (("Joint", "Optimal"), ("Trained", "Trained RNN"), ("Naive", "Naive"), ("Echo", "ESN"))
N_PER_CELL = 1
SIGMA = 1.5                                                                       # the Gaussian window (steps)
CMAX, A_MAX, GAMMA, LW = 4., .7, 1.5, .9
ELEV, AZIM, ZOOM, PANEL_W, PANEL_H = 24, -90, 1.3, 3.4, 2.9


def joint_cov(L, g, obs):
    """the Optimal's exact posterior over the 100 cells at every step -> Cov(r_g, r_c) (n, T), goal-ordered"""
    o = obs.astype(np.float64)
    ll = (np.einsum("nto,oab->ntab", o, np.log(L)) + np.einsum("nto,oab->ntab", 1 - o, np.log1p(-L))).cumsum(1)
    p = np.exp(ll - ll.max((2, 3), keepdims=True))
    p /= p.sum((2, 3), keepdims=True)
    if g == 1:
        p = p.transpose(0, 1, 3, 2)
    r = np.arange(R, dtype=np.float64)
    eg, ec = p.sum(3) @ r, p.sum(2) @ r
    return np.einsum("ntab,a,b->nt", p, r, r) - eg * ec


def compute():
    """the Optimal's belief Cov (39, 5000, T) float32"""
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    goal, obs_all, L_all = meta["goal"].astype(int), meta["obs"], meta["L"]
    return {"Joint": np.stack([joint_cov(L_all[e], int(goal[e]), obs_all[e]) for e in ENVS]).astype(np.float32)}


def belief_cov():
    if not os.path.exists(CACHE):
        np.savez(CACHE, **compute())
    with np.load(CACHE) as f:
        return f["Joint"]


def means(agent, pick=slice(None)):
    """an agent's posterior means (E, n, T, 2) over the scan's streams pick"""
    B = np.load(os.path.join(SCAN, f"scan_{agent}.npz"))["beliefs"]
    r = np.arange(R, dtype=np.float32)
    b = B[ENVS][:, pick].astype(np.float32)
    return np.stack([b[..., :R] @ r / b[..., :R].sum(-1), b[..., R:] @ r / b[..., R:].sum(-1)], -1)


def window(X):
    """positions (..., T, 2) -> the Gaussian window's weighted mean (..., T, 2) and weighted covariance (..., T)"""
    t = np.arange(X.shape[-2])
    w = np.exp(-(t[None] - t[:, None]) ** 2 / (2 * SIGMA ** 2))
    w /= w.sum(1, keepdims=True)                                                  # (t, s), renormalized at the ends
    X = X.astype(np.float64)
    m = np.einsum("ts,...sd->...td", w, X)
    return m, np.einsum("ts,...s->...t", w, X[..., 0] * X[..., 1]) - m[..., 0] * m[..., 1]


def segments(agent):
    """one agent's paths lifted by their own path covariance -> segments (m, 2, 3) and their mean C (m,), the
    invisible ones dropped"""
    m, C = window(means(agent, np.flatnonzero(np.arange(5000) % 50 < N_PER_CELL)))
    X = np.concatenate([m, np.clip(C, -CMAX - 1, CMAX + 1)[..., None]], -1).reshape(-1, T, 3)
    seg = np.stack([X[:, :-1], X[:, 1:]], 2).reshape(-1, 2, 3)
    v = seg[:, :, 2].mean(1)
    keep = A_MAX * np.minimum(np.abs(v) / CMAX, 1) ** GAMMA >= 2e-3
    return seg[keep], v[keep]


def numbers():
    """the path covariance's spread per agent; the Optimal's against its belief covariance; every agent's against the
    Optimal's, stream by stream and as aggregate maps (mean C by the smoothed position's 1-cell bin); the ESN's against
    the Naive's (all streams and steps)"""
    S = {agent: window(means(agent)) for agent, _ in AGENTS}
    maps = {}
    for agent, name in AGENTS:
        m, C = S[agent]
        b = np.clip(np.rint(m), 0, R - 1).astype(int)
        i = (b[..., 0] * R + b[..., 1]).ravel()
        maps[agent] = np.bincount(i, C.ravel(), R * R) / np.maximum(np.bincount(i, minlength=R * R), 1)
        q = np.percentile(np.abs(C), [90, 99, 99.9])
        print(f"{name:12s} |path Cov| percentiles 90/99/99.9: " + " / ".join(f"{x:.2f}" for x in q) + " cells^2")
    a, y = S["Joint"][1].ravel(), belief_cov().astype(np.float64).ravel()
    big = np.abs(y) > 5
    print(f"Optimal: path Cov vs its belief Cov: r = {np.corrcoef(a, y)[0, 1]:.2f}; the same sign at "
          f"{100 * np.mean(np.sign(a[big]) == np.sign(y[big])):.1f}% of the stream-steps where |belief Cov| > 5")
    rms = lambda v: np.sqrt((v ** 2).mean())
    for agent, name in AGENTS[1:]:
        print(f"{name:12s} vs the Optimal: per stream r = {np.corrcoef(a, S[agent][1].ravel())[0, 1]:.2f}; maps r = "
              f"{np.corrcoef(maps['Joint'], maps[agent])[0, 1]:.2f}, RMS {rms(maps[agent]) / rms(maps['Joint']):.2f} "
              "of the Optimal's", flush=True)
    print(f"ESN vs the Naive: per stream r = {np.corrcoef(S['Echo'][1].ravel(), S['Naive'][1].ravel())[0, 1]:.2f}; "
          f"maps r = {np.corrcoef(maps['Echo'], maps['Naive'])[0, 1]:.2f}")


def figure():
    import plotting_helpers as ph
    from main.utils import apply_fig_style, fig_path
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    cmap = plt.get_cmap("RdBu_r")
    S, SH, GX, GY, X0, YT, YB = PANEL_W, PANEL_H, .2, .1, 1.0, 1.35, .1
    W, H = X0 + len(AGENTS) * S + (len(AGENTS) - 1) * GX + .15, YT + 2 * SH + GY + YB
    fig = plt.figure(figsize=(W, H))
    fig.text(.5, 1 - .2 / H, "Each line: one particle's path over steps 1-30, its posterior means "
             r"$(E[r_g], E[r_c])$ smoothed by a Gaussian window over steps ($\sigma$ = "
             f"{SIGMA:g}), lifted by the window's covariance of them (sign: the path's slope; size: how far it moves "
             "along that diagonal)", ha="center", va="center", fontsize=FS["label"])
    fig.text(.5, 1 - .44 / H, "Colour and opacity: the same covariance (0: invisible).  Every agent its own read-out.  "
             f"39 environments x {N_PER_CELL} stream per cell = {39 * 100 * N_PER_CELL:,} paths per agent",
             ha="center", va="center", fontsize=FS["label"])
    cax = fig.add_axes([(W / 2 - 2.) / W, 1 - .74 / H, 4. / W, .11 / H])
    v = np.linspace(-CMAX, CMAX, 401)
    rgba = cmap((v + CMAX) / (2 * CMAX))
    a = (np.abs(v) / CMAX) ** GAMMA
    cax.imshow((rgba[:, :3] * a[:, None] + (1 - a[:, None]))[None], aspect="auto", extent=(-CMAX, CMAX, 0, 1))
    cax.set_yticks([]), cax.set_xticks([-CMAX, -CMAX / 2, 0, CMAX / 2, CMAX])
    cax.tick_params(labelsize=FS["tick"], length=2, pad=1)
    cax.text(-CMAX - .2, .5, "moving along (0, 9) - (9, 0)", ha="right", va="center", fontsize=FS["tick"])
    cax.text(CMAX + .2, .5, "moving along (0, 0) - (9, 9)", ha="left", va="center", fontsize=FS["tick"])
    cax.set_xlabel(r"path covariance, cells$^2$", fontsize=FS["tick"], labelpad=1)
    sq = np.array([[0, 0], [9, 0], [9, 9], [0, 9], [0, 0]])
    zt = int(CMAX)
    for c_, (agent, name) in enumerate(AGENTS):
        seg, val = segments(agent)
        col = cmap((np.clip(val, -CMAX, CMAX) + CMAX) / (2 * CMAX))
        col[:, 3] = A_MAX * np.minimum(np.abs(val) / CMAX, 1) ** GAMMA
        mid = seg.mean(1)
        for r_, (elev, view) in enumerate(((ELEV, "from above"), (-ELEV, "from below"))):
            y = H - YT - SH - r_ * (SH + GY)
            ax = fig.add_axes([(X0 + c_ * (S + GX)) / W, y / H, S / W, SH / H], projection="3d")
            ax.set_xlim(0, 9), ax.set_ylim(0, 9), ax.set_zlim(-CMAX - 1, CMAX + 1)
            ax.set_proj_type("ortho")
            ax.view_init(elev=elev, azim=AZIM)
            ax.set_box_aspect(None, zoom=ZOOM)
            ax.set_axis_off()
            ax.set_facecolor((1, 1, 1, 0))
            depth = proj3d.proj_transform(mid[:, 0], mid[:, 1], mid[:, 2], ax.get_proj())[2]
            order = np.argsort(-depth, kind="stable")                              # far first
            ax.plot(sq[:, 0], sq[:, 1], 0, color="0.45", lw=.9)
            for d in ((0, 9), (9, 0)):
                ax.plot([0, 9], d, [0, 0], color="0.6", lw=.7, ls=":")
            ax.add_collection3d(Line3DCollection(seg[order], colors=col[order], linewidths=LW))
            for (gx, gy) in ((0, 0), (9, 9), (0, 9), (9, 0)):                       # offset in depth: off the plane
                ax.text(gx, gy + (-1.6 if gy == 0 else 1.6), 0, f"({gx}, {gy})", ha="center", va="center",
                        fontsize=FS["tick"] + 1, color="0.3")
            if c_ == 0:                                                            # the height scale
                xb, yb = -1.3, 0
                ax.plot([xb, xb], [yb, yb], [-zt, zt], color="0.25", lw=.9, clip_on=False)
                for z in (-zt, 0, zt):
                    ax.plot([xb - .25, xb + .25], [yb, yb], [z, z], color="0.25", lw=.9, clip_on=False)
                    ax.text(xb - .5, yb, z, f"${z:+d}$" if z else "$0$", ha="right", va="center",
                            fontsize=FS["tick"], color="0.25")
            if r_ == 0:
                ax.set_title(name, fontsize=FS["label"] + 2, color=ph.AGENT_COLOR[agent], pad=-6)
            if c_ == 0:
                fig.text(.2 / W, (y + SH / 2) / H, view, rotation=90, ha="center", va="center",
                         fontsize=FS["label"] + 1)
    path = fig_path("fig1c_lifted_plane.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


if __name__ == "__main__":
    numbers()
    print("saved", figure())
