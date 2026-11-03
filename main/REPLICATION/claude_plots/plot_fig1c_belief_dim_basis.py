"""Why the belief effective-dimension maps keep a pattern, and is the X on the Optimal's directions the figure 8?
(28 Oct 2026; John on the harvest movie: "we still get structure in the belief_dim_harvest. do you know why we still see
this structure? Is the optimal direction structure reflecting the figure 8 in marginals' basis?")
-> main/REPLICATION/figure_bin/fig1c_belief_dim_basis.png

Data: the harvest (plot_fig1c_belief_dim_movie.harvest_beliefs: 19,607 episodes, one stream each, 1,225 environments);
each agent's 20 marginal probabilities b = [b_g, b_c]; effective dimension as the movie: E_k = ((b - mean) . w_k)^2 on
a basis w_k, (sum E)^2 / sum E^2.
Row 1: the Optimal's first six principal directions on the grid, each cell's value the projection of a belief settled
on it, w_g(r_g) + w_c(r_c) - mean . w. They come in pairs of equal variance (10.8 / 10.6%, 9.3 / 9.2%, 7.5 / 7.4%; the
two variables are statistically alike), each pair one pattern a on r_g and the same a on r_c, which the PCA returns
mixed into a sum (a, a) and a difference (a, -a) (cos(w_g, w_c) = +-1.00; share on r_g 0.36 - 0.64 in the first four
pairs); each a is even or odd under r -> 9 - r (parity +-0.95 - 1.00).
Rows 2 - 3: the effective dimension at each cell, for a belief settled on it and measured at step 30 (mean over the
beliefs whose posterior means round to the cell) for the four agents, on the Optimal's directions (row 2, mixed pairs)
and on each variable's own directions (row 3, the PCA of b_g and of b_c apart, no mixing); each row coloured over its
median +- 2.5. X contrast = mean over the 80 off-diagonal cells minus the 20 diagonal ones: row 2 settled +1.84,
measured Optimal +1.57, Trained RNN +1.54, Naive +1.28, ESN +1.44; row 3 -0.00, -0.09, -0.11, -0.47, -0.35.
Why the X: a settled belief on a diagonal projects onto only one member of each mixed pair (difference modes vanish on
r_g = r_c; with the patterns' parity one member of each pair vanishes on r_g = 9 - r_c), so its energy sits in half the
directions and its dimension is lower. Unmixed directions remove it, for every agent.
Not the figure 8: nothing in the beliefs sets the mixing. Across beliefs the goal and context marginals are
uncorrelated (same-pattern amplitude correlations |r| <= 0.03 at steps 1-5, 6-15, 16-30; the harvest's true values
independent, r = -0.016, chi2 75 on 81 df); each pair's mixing angle is undetermined (bootstrap over episodes, pair 1:
-46 deg, 90% -79 to +24); the control with the marginals decoupled (each step's context marginals permuted across
episodes) still mixes and draws the X (+1.53); the scan's own basis mixes the same pairs at other angles (pair 2
unmixed) and draws an X too (+1.71). The equal variances come from the two variables' symmetry and the diagonals from
the square's (swap r_g, r_c; flip r -> 9 - r): the same diagonals as the evidence's u and v, for a different reason. The
marginals cannot hold the 8 in any case: a (0, 0) / (9, 9) belief and a (0, 9) / (9, 0) belief have the same marginals
(plot_fig1c_cov_explained).
Why any pattern stays with more environments: a belief's effective dimension is fixed by where it sits as seen by the
basis (measured vs settled-belief maps r = 0.75 / 0.74 / 0.52 / 0.67 here, row 2; 0.93 in half-cell bins, where the
settled templates also carry the lattice motif: 50 / 50 two-cell beliefs at edges, one cell at nodes).
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_belief_dim_basis.py"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
import plot_fig1c_belief_dim_movie as bd                                          # noqa: E402

R, T, T_SHOW = 10, 30, 30
FLIP = np.arange(R)[::-1]


def own(B):
    """the agent's own principal directions, descending: mean, directions (20, 20), variances"""
    mu, V, lam = bd._basis(B)
    return mu, V[:, ::-1], lam[::-1]


def per_variable(B):
    """each variable's own principal directions (the PCA of b_g and of b_c apart), block-diagonal, by variance"""
    flat = B.reshape(-1, 2 * R).astype(np.float64)
    sub = flat[np.random.default_rng(0).choice(len(flat), 300_000, replace=False)]
    V, lam = np.zeros((2 * R, 2 * R)), np.zeros(2 * R)
    for h in range(2):
        l_, v_ = np.linalg.eigh(np.cov(sub[:, h * R:(h + 1) * R].T))
        V[h * R:(h + 1) * R, h * R:(h + 1) * R], lam[h * R:(h + 1) * R] = v_, l_
    o = np.argsort(-lam)
    return sub.mean(0), V[:, o], lam[o]


def decoupled(B, seed=7):
    """the control: each step's context marginals permuted across episodes"""
    out = B.copy()
    rng = np.random.default_rng(seed)
    for t in range(B.shape[1]):
        out[:, t, R:] = out[rng.permutation(len(B)), t, R:]
    return out


def templates():
    """a belief settled on each cell (10, 10, 20): one-hot in both marginals"""
    E = np.eye(R)
    return np.array([[np.concatenate([E[a], E[b]]) for b in range(R)] for a in range(R)])


def cellmap(X, v):
    """mean of v over the beliefs whose posterior means round to each cell (10, 10), counts"""
    k = np.clip(np.rint(X).astype(int), 0, R - 1)
    i = k[:, 0] * R + k[:, 1]
    c = np.bincount(i, minlength=R * R)
    return (np.bincount(i, v, minlength=R * R) / np.maximum(c, 1)).reshape(R, R), c.reshape(R, R)


def xcontrast(M):
    """mean over the 80 off-diagonal cells minus the 20 diagonal cells (both diagonals)"""
    d = np.zeros((R, R), bool)
    d[np.arange(R), np.arange(R)] = d[np.arange(R), FLIP] = True
    return np.nanmean(M[~d]) - np.nanmean(M[d])


def figure(span=2.5):
    from main.utils import fig_path
    Bs, ci = bd.harvest_beliefs()
    n = len(ci)
    n_env = len(np.unique(np.sort(ci, 1) @ [1000, 1]))
    B = Bs["Joint"]
    bases = {"opt": own(B), "var": per_variable(B), "dec": own(decoupled(B))}
    tpl = templates()
    rows = {}
    for key in ("opt", "var"):
        mu, V, _ = bases[key]
        row = [(bd._pr(tpl, mu, V), "a belief settled\non each cell")]
        for agent, name in bd.AGENTS:
            b = Bs[agent][:, T_SHOW - 1]
            M, c = cellmap(bd._means(b), bd._pr(b, mu, V))
            row.append((np.where(c >= 5, M, np.nan), f"{name}, measured\n(step {T_SHOW})"))
        rows[key] = row
    x_dec = xcontrast(bd._pr(tpl, *bases["dec"][:2]))
    mu, V, lam = bases["opt"]
    PW, GX, X0 = 2.25, .42, .55
    W = X0 + 6 * PW + 5 * GX + .35
    H = 11.8
    with plt.rc_context(bd.mm.RC):
        fig = plt.figure(figsize=(W, H), dpi=200, facecolor="white")
        fig.text(.5, 1 - .22 / H, "The Optimal's belief principal directions and the effective-dimension maps they "
                 f"draw (harvest: {n:,} episodes, one stream each, {n_env:,} environments)", ha="center",
                 va="center", fontsize=12)
        fig.text(.5, 1 - .52 / H, "row 1: each direction's value for a belief settled on each cell, "
                 r"$w_g(r_g) + w_c(r_c) - \bar b \cdot w$: pairs of equal variance, one pattern on $r_g$ and the same "
                 "on $r_c$, mixed into a sum and a difference", ha="center", va="center", fontsize=10.5)
        y1 = H - .95 - PW
        for k in range(6):
            w = V[:, k]
            L = w[:R][:, None] + w[R:][None, :] - mu @ w
            lim = np.abs(L).max()
            x = X0 + k * (PW + GX)
            ax = fig.add_axes([x / W, y1 / H, PW / W, PW / H])
            ax.imshow(L.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest",
                      extent=(-.5, 9.5, -.5, 9.5))
            ax.plot([-.5, 9.5], [-.5, 9.5], color="k", lw=.5, ls=":")
            ax.plot([-.5, 9.5], [9.5, -.5], color="k", lw=.5, ls=":")
            ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
            ax.tick_params(labelsize=8, length=2, pad=1)
            ax.set_xlabel(r"$r_g$", labelpad=-6)
            if k == 0:
                ax.set_ylabel(r"$r_c$", labelpad=-6)
            cs = w[:R] @ w[R:] / (np.linalg.norm(w[:R]) * np.linalg.norm(w[R:]))
            ax.set_title(f"direction {k + 1}: {100 * lam[k] / lam.sum():.1f}% of the variance\n"
                         f"{'sum' if cs > 0 else 'difference'}, {100 * np.linalg.norm(w[:R]) ** 2:.0f}% on $r_g$",
                         fontsize=9.5, pad=4)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("white")
        y = y1
        for key, head in (("opt", "rows 2 - 3: effective dimension at each cell (measured: mean over the beliefs whose "
                                  "posterior means round to it; white: fewer than 5). Row 2: on the Optimal's "
                                  "directions (mixed pairs)"),
                          ("var", "row 3: on each variable's own directions (the PCA of $b_g$ and of $b_c$ apart: "
                                  "no mixing)")):
            fig.text(.5, (y - .62) / H, head, ha="center", va="center", fontsize=10.5)
            y = y - 1.35 - PW
            vals = np.concatenate([m[np.isfinite(m)] for m, _ in rows[key]])
            mid = np.median(vals)
            for i, (M, title) in enumerate(rows[key]):
                x = X0 + i * (PW + GX) + (PW + GX) / 2
                ax = fig.add_axes([x / W, y / H, PW / W, PW / H])
                im = ax.imshow(np.ma.masked_invalid(M).T, origin="lower", cmap=cmap, vmin=mid - span,
                               vmax=mid + span, interpolation="nearest", extent=(-.5, 9.5, -.5, 9.5))
                ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
                ax.tick_params(labelsize=8, length=2, pad=1)
                ax.set_xlabel(r"$E[r_g]$" if i else r"$r_g$", labelpad=-6)
                if i == 0:
                    ax.set_ylabel(r"$r_c$", labelpad=-6)
                ax.set_title(f"{title}\nX contrast {xcontrast(M):+.2f}", fontsize=9.5, pad=4)
            cax = fig.add_axes([(X0 + 5 * (PW + GX) + (PW + GX) / 2) / W, y / H, .13 / W, PW / H])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label(f"effective dimension\n(row median {mid:.1f} $\\pm$ {span:g})", fontsize=9)
            cb.ax.tick_params(labelsize=8)
        fig.text(.5, .62 / H, "X contrast: the 80 off-diagonal cells minus the 20 diagonal ones. Across beliefs the "
                 r"goal and context marginals are uncorrelated (same-pattern amplitudes $|r| \leq 0.03$), so nothing "
                 "in the beliefs picks the mixing:", ha="center", va="center", fontsize=10)
        fig.text(.5, .3 / H, f"the control with the two marginals decoupled (each step's context marginals permuted "
                 f"across episodes) draws the X too (settled beliefs: {x_dec:+.2f}).", ha="center", va="center",
                 fontsize=10)
        path = fig_path("fig1c_belief_dim_basis.png")
        fig.savefig(path)
        plt.close(fig)
    for key in ("opt", "var"):
        print(f"{key}: X contrast " + ", ".join(f"{t.splitlines()[0]} {xcontrast(M):+.2f}" for M, t in rows[key]))
    print(f"decoupled control, settled beliefs: X contrast {x_dec:+.2f}")
    for key in ("opt", "var"):
        M0 = rows[key][0][0]
        print(f"{key}: measured vs settled-belief map r: " + ", ".join(
            f"{t.splitlines()[0]} {np.corrcoef(M[np.isfinite(M)], M0[np.isfinite(M)])[0, 1]:+.2f}"
            for M, t in rows[key][1:]))
    for k in range(6):
        w = V[:, k]
        print(f"direction {k + 1}: {100 * lam[k] / lam.sum():.2f}%, on r_g {np.linalg.norm(w[:R]) ** 2:.2f}, "
              f"cos(w_g, w_c) {w[:R] @ w[R:] / (np.linalg.norm(w[:R]) * np.linalg.norm(w[R:])):+.2f}, parity "
              f"{w[:R] @ w[:R][FLIP] / np.linalg.norm(w[:R]) ** 2:+.2f}")
    return path


if __name__ == "__main__":
    print("saved", figure())
