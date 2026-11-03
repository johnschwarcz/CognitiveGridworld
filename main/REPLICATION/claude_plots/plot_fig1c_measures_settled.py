"""Settled vs still-settling particles on the nine measures, step by step (28 Oct 2026; John on the particles movie with
the shared pattern removed: "an interesting contrast between particles that has settled to a location on the grid and
those that are still settling. Can you please show curves for the gap between settled and unsettled particles for our
9 measures through time.")
-> main/REPLICATION/figure_bin/fig1c_measures_settled.png

Particles: the Trained RNN's streams in the alias scan (39 environments x 5,000 streams, all 50 per cell), each at its
posterior means (E[r_g], E[r_c]) at step t. Settled: within SETTLED cells of a lattice node (any node; the movie's
particles sitting on the grid); unsettled: farther than UNSETTLED (between nodes); the ones in between left out. The
distance to the nearest node has no natural break (it spreads evenly from 0 to the 0.71 between nodes), so the margin
is a choice; the read-out's own sharpness (max b_g max b_c >= 0.9) marks almost no particle before step 20.
Curves: per environment and step, the mean over the settled particles minus the mean over the unsettled ones, median
over the 39 environments (band: the middle half). Solid: the percentile with the shared pattern removed (each
stream's percentile within its environment and step minus the mean percentile of its cell's streams over the 39
environments at that step; the particles_vs_typical movie did this on percentiles until 10-28, now on raw values);
dashed: the plain percentile. Top: the
share of particles settled and unsettled at each step. Factorization regret averaged over both variables
(plot_fig1c_measures_movie.fr_both).
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_measures_settled.py"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
import plot_fig1c_measures_movie as mm                                            # noqa: E402

SETTLED, UNSETTLED = .15, .35


def compute():
    """{measure: (2 kinds, 3 quantiles, T)} gaps (kinds: vs typical, plain; quantiles 25/50/75 over environments),
    shares (2, T) settled / unsettled, the particle count"""
    _, _, Xs, strm, _ = mm.data()
    E, S, T = Xs.shape[:3]
    cid = np.arange(S) // mm.NS
    d = np.linalg.norm(Xs - np.rint(Xs), axis=-1)                                  # (E, S, T)
    st, un = d < SETTLED, d > UNSETTLED
    out = {}
    for key, *_ in mm.PANELS:
        P = strm[key]
        dev = P - P.reshape(E, mm.R * mm.R, mm.NS, T).mean((0, 2))[cid]
        g = []
        for V in (dev, P):
            ms = (V * st).sum(1) / np.maximum(st.sum(1), 1)
            mu = (V * un).sum(1) / np.maximum(un.sum(1), 1)
            gap = np.where((st.sum(1) > 0) & (un.sum(1) > 0), ms - mu, np.nan)    # (E, T)
            g.append(np.nanpercentile(gap, [25, 50, 75], axis=0))
        out[key] = np.array(g)
    return out, np.array([st.mean((0, 1)), un.mean((0, 1))]), (E, S)


def figure():
    from main.utils import fig_path
    res, share, (E, S) = compute()
    T = mm.T
    t = np.arange(1, T + 1)
    PW, PH, GX, GY, X0, YT, SH = 3.0, 1.8, .75, .95, .9, 1.0, 1.0
    W, H = X0 + 3 * PW + 2 * GX + .3, YT + SH + .75 + 3 * PH + 2 * GY + .7
    with plt.rc_context(mm.RC):
        fig = plt.figure(figsize=(W, H), dpi=200, facecolor="white")
        fig.text(.5, 1 - .22 / H, f"Trained RNN, {E} environments x {S:,} streams each: particles settled (posterior "
                 f"means within {SETTLED} cells of a lattice node) minus unsettled (farther than {UNSETTLED})",
                 ha="center", va="center", fontsize=11)
        fig.text(.5, 1 - .5 / H, "solid: percentile with the shared pattern removed (minus the cell's mean "
                 "percentile over the 39 environments at that step);  dashed: plain percentile;  band: the middle "
                 "half of the environments", ha="center", va="center", fontsize=9)
        ax = fig.add_axes([X0 / W, (H - YT - SH) / H, (3 * PW + 2 * GX) / W, SH / H])
        ax.plot(t, 100 * share[0], color="#2166ac", lw=1.8, label=f"settled ($<$ {SETTLED} cells from a node)")
        ax.plot(t, 100 * share[1], color="#b2182b", lw=1.8, label=f"unsettled ($>$ {UNSETTLED})")
        ax.set_xlim(1, T), ax.set_ylim(0, 90), ax.set_xticks([1, 10, 20, 30])
        ax.set_ylabel("% of particles", fontsize=10), ax.tick_params(labelsize=9, length=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9.5, frameon=False, loc="upper left", ncol=2)
        y0 = H - YT - SH - .75
        for n, (key, name, formula) in enumerate(mm.PANELS):
            r_, c_ = divmod(n, 3)
            x, y = X0 + c_ * (PW + GX), y0 - PH - r_ * (PH + GY)
            ax = fig.add_axes([x / W, y / H, PW / W, PH / H])
            ax.axhline(0, color="0.6", lw=.8, ls=":")
            g = res[key]
            ax.fill_between(t, 100 * g[0, 0], 100 * g[0, 2], color="k", alpha=.15, lw=0)
            ax.plot(t, 100 * g[0, 1], color="k", lw=1.7)
            ax.plot(t, 100 * g[1, 1], color="k", lw=1.0, ls="--")
            ax.set_xlim(1, T), ax.set_xticks([1, 10, 20, 30])
            ax.tick_params(labelsize=9, length=2)
            ax.spines[["top", "right"]].set_visible(False)
            if c_ == 0:
                ax.set_ylabel("settled - unsettled\n(percentile points)", fontsize=9.5)
            if r_ == 2:
                ax.set_xlabel("step $t$", fontsize=10)
            ax.set_title(name + "\n" + formula, fontsize=10, pad=4)
        path = fig_path("fig1c_measures_settled.png")
        fig.savefig(path)
        plt.close(fig)
    for key, *_ in mm.PANELS:
        g = 100 * res[key][0, 1]
        print(f"{key:7s} settled - unsettled (vs typical), median over envs, t 2 / 5 / 10 / 20 / 30: "
              + " / ".join(f"{g[i]:+.1f}" for i in (1, 4, 9, 19, 29)))
    return path


if __name__ == "__main__":
    print("saved", figure())
