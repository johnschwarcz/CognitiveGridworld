"""Do the measures mark the grid's tears, or a particle's jump, without aligning on any detected event? (27 Oct 2026;
John on the measures movies: "I want to see if unfolding is associated with any of these variables, or if any of these
can identify 'wormholes'")
-> main/REPLICATION/figure_bin/fig1c_measures_jumps.png

Panels (3 x 3, as the movie): one measure each (the Trained RNN's; the alias scan, 39 environments); curves over step t,
each the median over the environments of a Spearman correlation computed per environment and step:
    particle, same step   across the environment's 5,000 streams: the stream's measure percentile at t and its
                          jump |x_t - x_{t-1}| (x: its posterior means), both ranked within the stream's cell (so
                          border vs centre and the cell cannot drive it), partialling out the stream's distance to its
                          true cell before the jump (round 7: unsettled particles are higher on several measures)
    particle, next step   the same with the next jump |x_{t+1} - x_t|, partialling out the distance at t
    grid, edge length     across the 180 lattice edges of the grid of cell means: the edge's measure (its two cells'
                          mean percentile) and its length |X_j - X_i| (long: the grid torn or stretched across), within
                          rings (edges grouped by their outer cell's ring; the groups' correlations averaged, weighted
                          by size)
Dotted: 0. Data: the alias scan's 39 environments x 5,000 streams (all 50 per cell; the data line at the top says so,
John 10-28; the earlier 1,000-stream version's cache is measures_assoc.npz). Computed once into
main/DATA/experimental/fig1c_compression/figdata/round7_jumps/measures_assoc_ns50.npz (from
plot_fig1c_measures_movie.data()); delete it to recompute.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_measures_jumps.py"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from scipy.stats import rankdata, spearmanr                                       # noqa: E402

CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression", "figdata", "round7_jumps",
                     "measures_assoc_ns50.npz")
CURVES = (("now", "particle, same step: jump $|x_t - x_{t-1}|$", "k", "-"),
          ("next", "particle, next step: jump $|x_{t+1} - x_t|$", "#d95f02", "-"),
          ("edge", "grid: edge length $|X_j - X_i|$", "#1b9e77", "--"))


def _pcorr(x, y, z):
    rxy, rxz, ryz = (np.corrcoef(a, b)[0, 1] for a, b in ((x, y), (x, z), (y, z)))
    return (rxy - rxz * ryz) / np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))


def compute():
    """{key: (3, T) medians over environments: now, next, edge}"""
    import plot_fig1c_measures_movie as mm
    Xc, cell, Xs, strm, _ = mm.data()
    E, S, T, _ = Xs.shape
    cid = np.arange(S) // mm.NS                                                   # streams: mm.NS per cell
    true = np.stack([cid // 10, cid % 10], 1).astype(float)

    def wrank(A):                                                                  # ranks within each cell's streams
        out = np.empty_like(A, dtype=float)
        for c in range(S // mm.NS):
            sl = slice(c * mm.NS, (c + 1) * mm.NS)
            out[:, sl] = rankdata(A[:, sl], axis=1)
        return out

    D = wrank(np.linalg.norm(Xs - true[None, :, None], axis=-1))                  # (E, S, T)
    J = wrank(np.linalg.norm(np.diff(Xs, axis=2), axis=-1))                        # (E, S, T-1): index t-1 = t-1 -> t
    edges, lattice = mm._grid()
    ring = np.minimum(np.minimum(lattice[:, 0], lattice[:, 1]), np.minimum(9 - lattice[:, 0], 9 - lattice[:, 1]))
    ering = np.minimum(ring[edges[:, 0]], ring[edges[:, 1]]).astype(int)
    length = np.linalg.norm(Xc[:, edges[:, 1]] - Xc[:, edges[:, 0]], axis=-1)     # (E, 180, T)
    out = {}
    for key in [p[0] for p in mm.PANELS]:
        V = wrank(strm[key])
        Ve = cell[key][:, edges].mean(2)                                           # (E, 180, T)
        r = np.full((3, E, T), np.nan)
        for t in range(T):
            for e in range(E):
                if np.ptp(V[e, :, t]) == 0:
                    continue
                if t >= 1:
                    r[0, e, t] = _pcorr(V[e, :, t], J[e, :, t - 1], D[e, :, t - 1])
                if t <= T - 2:
                    r[1, e, t] = _pcorr(V[e, :, t], J[e, :, t], D[e, :, t])
                rs, ws = [], []
                for g in np.unique(ering):
                    s = ering == g
                    if np.ptp(Ve[e, s, t]) > 0:
                        rs.append(spearmanr(Ve[e, s, t], length[e, s, t])[0]), ws.append(s.sum())
                if ws:
                    r[2, e, t] = np.average(rs, weights=ws)
        out[key] = np.nanmedian(r, 1)
    out["n_streams"] = np.array(S)
    return out


def figure():
    import plot_fig1c_measures_movie as mm
    from main.utils import fig_path
    if not os.path.exists(CACHE):
        np.savez(CACHE, **compute())
    with np.load(CACHE) as f:
        res = {k: f[k] for k in f.files}
    T = mm.T
    t = np.arange(1, T + 1)
    PW, PH, GX, GY, X0, YT = 3.0, 1.9, .75, .95, .85, 1.0
    W, H = X0 + 3 * PW + 2 * GX + .3, YT + 3 * PH + 2 * GY + 1.15
    with plt.rc_context(mm.RC):
        fig = plt.figure(figsize=(W, H), dpi=200, facecolor="white")
        fig.text(.5, 1 - .2 / H, f"Trained RNN, 39 environments x {int(res['n_streams']):,} streams each; grid: the "
                 f"100 cells' means over their {mm.NS} streams", ha="center", va="center", fontsize=11)
        for n, (key, name, formula) in enumerate(mm.PANELS):
            r_, c_ = divmod(n, 3)
            x, y = X0 + c_ * (PW + GX), H - YT - PH - r_ * (PH + GY)
            ax = fig.add_axes([x / W, y / H, PW / W, PH / H])
            ax.axhline(0, color="0.6", lw=.8, ls=":")
            for i, (_, label, col, ls) in enumerate(CURVES):
                ax.plot(t, res[key][i], color=col, ls=ls, lw=1.6, label=label)
            ax.set_xlim(1, T), ax.set_ylim(-.2, .5)
            ax.set_xticks([1, 10, 20, 30]), ax.set_yticks([-.2, 0, .2, .4])
            ax.tick_params(labelsize=9, length=2)
            ax.spines[["top", "right"]].set_visible(False)
            if c_ == 0:
                ax.set_ylabel("Spearman, median\nover 39 environments", fontsize=9.5)
            if r_ == 2:
                ax.set_xlabel("step $t$", fontsize=10)
            ax.set_title(name + "\n" + formula, fontsize=10, pad=4)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10, frameon=False,
                   bbox_to_anchor=(.5, .005))
        path = fig_path("fig1c_measures_jumps.png")
        fig.savefig(path)
        plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure())
