"""Naive Bayes and the linear rule as geometry (John, 2026-10-22): naive_vs_linear_mechanism.png.

One typical test pair (closest to the median pair on the linear rule's gap over naive Bayes, naive's and joint Bayes's
accuracy), T = 30, decoding r0, on plain count axes of the pair's bits with the largest r0 main effect. Naive Bayes and the
linear rule both score linearly in the counts, so both draw straight borders here (flat in 3 bits); joint Bayes does not.
(An earlier version spaced the axes as naive Bayes measures distance, asin(sqrt(n/T)); John asked for straight borders.)
Lines (colour = r0): where an episode's counts land for each of the 10 values of the other context r1, T p(r0, r1). The
other context is drawn once per episode, so an episode lands near one point of its line. Shading: the value a rule
answers from these bits alone. Dot: a line point answered correctly; cross: answered wrongly, in the colour of the value
the rule picks. Plus: a value's average point pbar(r0), naive Bayes's centre.
  top (a-c), 2 bits: naive Bayes, the linear rule and joint Bayes; shading = the value each answers
  bottom (d-f), 3 bits: the two values whose line points naive Bayes confuses most while the linear rule gets them right;
      a flat cut through the plane their lines lie close to (in counts),
      shaded by which of the two the rule prefers, its border between them in black (dot / cross: the actual 3-bit point
      on the right / wrong side); naive Bayes also shows the two centres and the segment joining them
Naive Bayes restricted to bits B answers argmin_r sum_{i in B} KL(n_i/T || pbar_i(r)) (exact KL). The linear rule: logits
linear in the counts of B, fitted by cross-entropy to the joint posterior given those counts over the population of every
count combination (exact enumeration; no sampling).

    python main/REPLICATION/claude_plots/plot_naive_vs_linear.py

Earlier versions (2026-10-21/22) also plotted the analyses in main/DATA/experimental/ff_deep/extrapolation/naive_vs_linear/
(episode-length sweep, logit-ANOVA scalings, A + B + I); John kept only the geometry.
"""
import os, sys, inspect, itertools
import numpy as np


def _repo_root():
    """the folder holding main/CognitiveGridworld.py, above this file; __file__ can be missing or a placeholder that
    does not exist in an interactive window, so the working directory is tried next"""
    here = None
    try:
        here = inspect.getfile(inspect.currentframe())
    except (TypeError, OSError):
        pass
    for start in (globals().get("__file__"), here, globals().get("__vsc_ipynb_file__"), os.getcwd()):
        d = os.path.abspath(start) if start and os.path.exists(start) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return d
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError("project root (the folder holding main/CognitiveGridworld.py) not found")


ROOT = _repo_root()
sys.path.insert(0, ROOT)
NVL = os.path.join(ROOT, "main", "DATA", "experimental", "ff_deep", "extrapolation", "naive_vs_linear")
T = 30
EPS = 1e-9
COLS = None                                                                       # set in figure(): a gradient over r0 = 0..9


def phi(p):
    return np.arcsin(np.sqrt(np.clip(p, 0, 1)))


def kl_bern(q, p):
    q, p = np.clip(q, EPS, 1 - EPS), np.clip(p, EPS, 1 - EPS)
    return q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))


def strongest_bits(u, v):
    """bits ordered by the RMS over r0 of r0's main effect alpha = vbar (u - ubar)"""
    alpha = v.mean(-1)[..., None] * (u - u.mean(-1)[..., None])
    return np.argsort(-np.sqrt((alpha ** 2).mean(-1)), axis = -1)


def typical_pair():
    """the test pair closest to the median pair on three per-pair accuracies at T = 30 (both contexts): sum over (linear rule
    - naive, naive, joint) of |value - median| / interquartile range"""
    d = np.load(os.path.join(NVL, "delta", "s345_T30fam_perpair.npz"))
    k = "env | T=30 | seeds 11,12 | "
    nv, ce, jt = d[k + "naive"], d[k + "CE"], d[k + "joint"]
    iqr = lambda a: np.subtract(*np.quantile(a, [.75, .25]))
    return int(np.argmin(sum(np.abs(a - np.median(a)) / iqr(a) for a in (ce - nv, nv, jt))))


class Pair:
    """one pair's table P (bit, r0, r1), naive Bayes's centres pbar (bit, r0), and both rules restricted to a set of bits"""

    def __init__(self, P):
        self.P, self.pbar = P, P.mean(-1)

    def naive(self, f, bits):
        """naive Bayes from `bits`: f (..., len(bits)) frequencies n/T -> the value it answers"""
        return sum(kl_bern(f[..., k][..., None], self.pbar[b]) for k, b in enumerate(bits)).argmin(-1)

    def joint(self, f, bits):
        """joint Bayes from `bits`: f (..., len(bits)) frequencies -> argmax_r log mean_r1 prod_i p_i^n_i (1 - p_i)^(T - n_i)
        (the binomial coefficients do not depend on r, so continuous counts are fine)"""
        lp = sum(T * f[..., k][..., None, None] * np.log(self.P[b]) + T * (1 - f[..., k][..., None, None]) * np.log1p(-self.P[b])
                 for k, b in enumerate(bits))                                 # (..., r0, r1)
        m = lp.max(-1, keepdims = True)
        return (m[..., 0] + np.log(np.exp(lp - m).sum(-1))).argmax(-1)

    def scores(self, kind, bits, W = None):
        """the rule's per-value scores as a function of f (for pairwise borders): kind naive / linear (W) / joint"""
        if kind == "naive":
            return lambda f: -sum(kl_bern(f[..., k][..., None], self.pbar[b]) for k, b in enumerate(bits))
        if kind == "linear":
            return lambda f: np.concatenate([f, np.ones(f.shape[:-1] + (1,))], -1) @ W
        def js(f):
            lp = sum(T * f[..., k][..., None, None] * np.log(self.P[b]) + T * (1 - f[..., k][..., None, None]) * np.log1p(-self.P[b])
                     for k, b in enumerate(bits))
            m = lp.max(-1, keepdims = True)
            return m[..., 0] + np.log(np.exp(lp - m).sum(-1))
        return js

    def linear_all(self, cache, K = 2000, seed = 0):
        """the linear rule on all five counts, fitted by cross-entropy to the exact joint posterior of K sampled 30-step
        episodes per (r0, r1) cell (200k episodes; seed 0); cached in `cache`. returns the rule f (..., 5) -> value, .W (6, 10)"""
        if os.path.exists(cache):
            W = np.load(cache)["W"]
        else:
            from scipy.optimize import minimize
            rng = np.random.default_rng(seed)
            r0 = np.repeat(np.arange(10), 10 * K)
            r1 = np.tile(np.repeat(np.arange(10), K), 10)
            n = rng.binomial(T, self.P[:, r0, r1].T)
            lp = np.einsum("ni,irs->nrs", n, np.log(self.P)) + np.einsum("ni,irs->nrs", T - n, np.log1p(-self.P))
            post = np.exp(lp - lp.max((1, 2), keepdims = True)).sum(-1)
            post /= post.sum(-1, keepdims = True)
            X = np.concatenate([n / T, np.ones((len(n), 1))], 1)

            def ce(wv):
                z = X @ wv.reshape(6, 10)
                z = z - z.max(1, keepdims = True)
                lsm = z - np.log(np.exp(z).sum(1, keepdims = True))
                return -(post * lsm).sum() / len(X), (X.T @ (np.exp(lsm) - post)).reshape(-1) / len(X)
            W = minimize(ce, np.zeros(60), jac = True, method = "L-BFGS-B", options = dict(maxiter = 5000, gtol = 1e-9)).x.reshape(6, 10)
            os.makedirs(os.path.dirname(cache), exist_ok = True)
            np.savez(cache, W = W, K = K, seed = seed)
        rule = lambda f: (np.concatenate([f, np.ones(f.shape[:-1] + (1,))], -1) @ W).argmax(-1)
        rule.W = W
        return rule

    def linear(self, bits):
        """the linear rule from `bits`: logits W [n/T, 1], fitted by cross-entropy to the joint posterior over every count
        combination weighted by its probability (r0, r1 uniform); returns f (..., len(bits)) -> the value it answers"""
        from scipy.stats import binom
        from scipy.optimize import minimize
        k = len(bits)
        combos = np.array(list(itertools.product(range(T + 1), repeat = k)))
        lik = np.prod([binom.pmf(combos[:, i][:, None, None], T, self.P[b][None]) for i, b in enumerate(bits)], 0)   # (C, r0, r1)
        Pn = lik.mean(-1)                                                    # P(n | r0)
        pall = Pn.mean(-1)                                                    # P(n)
        post = Pn / Pn.sum(-1, keepdims = True)                              # joint posterior over r0 from these bits
        X = np.concatenate([combos / T, np.ones((len(combos), 1))], 1)

        def ce(wv):
            z = X @ wv.reshape(k + 1, 10)
            z = z - z.max(1, keepdims = True)
            lsm = z - np.log(np.exp(z).sum(1, keepdims = True))
            return -(pall[:, None] * post * lsm).sum(), (X.T @ (pall[:, None] * (np.exp(lsm) - post))).reshape(-1)
        W = minimize(ce, np.zeros((k + 1) * 10), jac = True, method = "L-BFGS-B", options = dict(maxiter = 5000, gtol = 1e-10)).x
        W = W.reshape(k + 1, 10)
        rule = lambda f: (np.concatenate([f, np.ones(f.shape[:-1] + (1,))], -1) @ W).argmax(-1)
        rule.W = W
        return rule


def figure():
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import ListedColormap, to_rgb
    from main.utils import apply_fig_style
    global COLS
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    COLS = [tuple(c) for c in plt.cm.viridis(np.linspace(0, .92, 10))]          # r0 = 0 (dark blue) .. 9 (yellow-green)
    W_ = np.load(os.path.join(NVL, "anova", "world_uv.npz"))
    g = typical_pair()
    pair = Pair(np.clip(W_["JL"][g], EPS, 1 - EPS))
    P, pbar = pair.P, pair.pbar
    order = strongest_bits(W_["u"][g], W_["v"][g])
    b2, b3 = [int(b) for b in order[:2]], [int(b) for b in order[:3]]
    lin2, lin3 = pair.linear(b2), pair.linear(b3)
    rules = {("naive", 2): lambda f: pair.naive(f, b2), ("linear", 2): lin2, ("joint", 2): lambda f: pair.joint(f, b2),
             ("naive", 3): lambda f: pair.naive(f, b3), ("linear", 3): lin3, ("joint", 3): lambda f: pair.joint(f, b3)}
    names = {"naive": "Naive Bayes", "linear": "Linear rule", "joint": "Joint Bayes"}
    tint = lambda c, a: tuple(1 - a * (1 - np.array(to_rgb(c))))
    region_cmap = ListedColormap([tint(c, .3) for c in COLS])
    ticks = [0, 5, 10, 15, 20, 25, 30]
    R2SD = 1 / np.sqrt(T)

    def picks(rule, bits):
        """the value the rule answers at every line point: (r0, r1)"""
        return rule(P[bits].transpose(1, 2, 0))

    W_IN, H_IN = 10.45, 6.05
    fig = plt.figure(figsize = (W_IN, H_IN))
    out = {}
    X0, DX = .5, 3.3

    # --------------------------------------------------------------------- top: two bits, shading = the rule's answer
    grid = np.linspace(0, T, 500)
    G1, G2 = np.meshgrid(grid, grid)
    for k, name in enumerate(("naive", "linear", "joint")):
        ax = fig.add_axes([(X0 + k * DX) / W_IN, 3.0 / H_IN, 2.7 / W_IN, 2.7 / H_IN])
        rule = rules[(name, 2)]
        cell = rule(np.stack([G1 / T, G2 / T], -1))
        ax.pcolormesh(grid, grid, cell, cmap = region_cmap, vmin = -.5, vmax = 9.5, shading = "nearest", rasterized = True, zorder = 0)
        ax.contour(grid, grid, cell, levels = np.arange(.5, 9.5, 1), colors = "white", linewidths = .8, zorder = 1)
        pk = picks(rule, b2)
        for r in range(10):
            s_ = T * P[b2][:, r, :]
            ax.plot(s_[0], s_[1], "-", color = COLS[r], lw = 1.2, zorder = 3)
            own = pk[r] == r
            ax.plot(s_[0][own], s_[1][own], "o", color = COLS[r], ms = 3.2, mec = "white", mew = .4, zorder = 4)
            for j in np.where(~own)[0]:
                ax.plot(s_[0][j], s_[1][j], "x", color = COLS[pk[r, j]], ms = 4.6, mew = 1.5, zorder = 5,
                        path_effects = [pe.withStroke(linewidth = 2.6, foreground = "white")])
        right = int((pk == np.arange(10)[:, None]).sum())
        out[(name, 2)] = right
        ax.set_xlim(0, T); ax.set_ylim(0, T)
        ax.set_aspect("equal")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks); ax.set_yticklabels([str(c) for c in ticks] if k == 0 else [])
        ax.tick_params(length = 2.5, pad = 2)
        ax.set_xlabel(f"count of bit {b2[0] + 1}", labelpad = 1)
        if k == 0:
            ax.set_ylabel(f"count of bit {b2[1] + 1}", labelpad = 1)
        ax.set_title(f"{'abc'[k]}   {names[name]}, 2 bits: {right} of 100 right", fontsize = fs["label"], loc = "left", pad = 6)
    cax = fig.add_axes([(X0 + 2 * DX + 2.8) / W_IN, 3.35 / H_IN, .1 / W_IN, 2.0 / H_IN])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap = ListedColormap(COLS), norm = plt.Normalize(-.5, 9.5)), cax = cax, ticks = range(10))
    cb.set_label("$r_0$", labelpad = 2)
    cb.ax.tick_params(labelsize = fs["tick"] - 1.5, length = 2)
    cb.outline.set_linewidth(.4)

    # --------------------------------------------------------------------- bottom: all five bits, one border per flat rule
    b5 = list(range(5))
    lin5 = pair.linear_all(os.path.join(NVL, "figure", f"linear_all5_pair{g}.npz"))
    pts5 = (T * P).transpose(1, 2, 0)                                              # (r0, r1, 5): the lines in counts
    pk5 = {"naive": pair.naive(pts5 / T, b5), "linear": lin5(pts5 / T)}
    for name in pk5:
        out[(name, 5)] = int((pk5[name] == np.arange(10)[:, None]).sum())
    conf = np.zeros((10, 10))                                                      # the two values naive Bayes confuses most
    for r in range(10):                                                            # while the linear rule gets them right
        for j in range(10):
            if pk5["naive"][r, j] != r and pk5["linear"][r, j] == r:
                conf[r, pk5["naive"][r, j]] += 1
    ra, rb = sorted(int(v) for v in np.unravel_index(np.argmax(conf + conf.T), conf.shape))
    # The view: the plane spanned by the two flat rules' directions for this pair (their border normals, per count). Each
    # rule's decision between ra and rb depends on a point only through its position in this plane, so both borders are
    # exact straight lines here and every point sits on its true side; the 3 collapsed directions lie along both borders.
    lg = lambda p: np.log(p) - np.log1p(-p)
    u_n, c_n = lg(pbar[:, ra]) - lg(pbar[:, rb]), T * (np.log1p(-pbar[:, ra]) - np.log1p(-pbar[:, rb])).sum()
    u_l, c_l = (lin5.W[:5, ra] - lin5.W[:5, rb]) / T, lin5.W[5, ra] - lin5.W[5, rb]
    e1 = u_l / np.linalg.norm(u_l)                                                 # vertical: the linear rule's direction
    e2 = u_n - (u_n @ e1) * e1
    e2 /= np.linalg.norm(e2)                                                       # horizontal: the rest of naive's
    proj = lambda x: np.stack([x @ e2, x @ e1], -1)
    pr = proj(np.concatenate([pts5[ra], pts5[rb]]))
    lo, hi = pr.min(0) - 2.5, pr.max(0) + 2.5
    S1, S2 = np.meshgrid(np.linspace(lo[0], hi[0] + 3, 440), np.linspace(lo[1], hi[1], 320))
    d_n = S1 * (u_n @ e2) + S2 * (u_n @ e1) + c_n
    d_l = S2 * (u_l @ e1) + c_l + 0 * S1
    duel_cmap = ListedColormap([tint(COLS[ra], .16), tint(COLS[rb], .42)])          # the two neighbouring colours, lighter / darker
    mk = {ra: "o", rb: "s"}
    for k, (name, dmap) in enumerate((("naive", d_n), ("linear", d_l), ("both", None))):
        ax = fig.add_axes([(X0 + k * DX) / W_IN, .72 / H_IN, 2.7 / W_IN, 1.4 / H_IN])
        if dmap is not None:
            ax.pcolormesh(S1, S2, (dmap < 0).astype(float), cmap = duel_cmap, vmin = 0, vmax = 1, shading = "nearest", rasterized = True, zorder = 0)
            ax.contour(S1, S2, dmap, levels = [0], colors = "#1f1f1e", linewidths = 1.5, zorder = 2)
        else:
            ax.contour(S1, S2, d_n, levels = [0], colors = "#1f1f1e", linewidths = 1.4, linestyles = [(0, (4, 2))], zorder = 2)
            ax.contour(S1, S2, d_l, levels = [0], colors = "#1f1f1e", linewidths = 1.5, zorder = 2)
        right = 0
        for r in (ra, rb):
            q = proj(pts5[r])
            ax.plot(q[:, 0], q[:, 1], "-", color = COLS[r], lw = 1.5, zorder = 3)
            if name == "both":
                ax.plot(q[:, 0], q[:, 1], mk[r], color = COLS[r], ms = 3.6, mec = "white", mew = .4, zorder = 4)
            else:
                d = pts5[r] @ (u_n if name == "naive" else u_l) + (c_n if name == "naive" else c_l)
                own = (d > 0) == (r == ra)
                right += int(own.sum())
                ax.plot(q[own, 0], q[own, 1], mk[r], color = COLS[r], ms = 3.8, mec = "white", mew = .4, zorder = 4)
                ax.plot(q[~own, 0], q[~own, 1], "x", color = COLS[r], ms = 5.4, mew = 1.7, zorder = 5,
                        path_effects = [pe.withStroke(linewidth = 2.6, foreground = "white")])
            if name == "naive":
                ax.plot(*proj(T * pbar[:, r]), "P", color = COLS[r], mec = "k", mew = .5, ms = 7.5, zorder = 6)
            far = int(np.argmax(q[:, 0]))
            ax.annotate(rf"$r_0={r}$", xy = q[far], xytext = (4, 0), textcoords = "offset points", fontsize = fs["tick"] - 1,
                        color = COLS[r], ha = "left", va = "center", zorder = 7, path_effects = [pe.withStroke(linewidth = 2.4, foreground = "white")])
        if name == "naive":
            cpa, cpb = proj(T * pbar[:, ra]), proj(T * pbar[:, rb])
            ax.plot([cpa[0], cpb[0]], [cpa[1], cpb[1]], color = "k", lw = .8, ls = (0, (2, 1.5)), zorder = 5)
        ax.set_xlim(lo[0], hi[0] + 3); ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        if name == "both":
            ax.set_title("f   Both borders (dashed: naive Bayes)", fontsize = fs["label"], loc = "left", pad = 6)
        else:
            ax.set_title(f"{'de'[k]}   {names[name]}, all 5 bits: {right} of 20 right", fontsize = fs["label"], loc = "left", pad = 6)
    fig.text(X0 / W_IN, .08 / H_IN,
             f"One typical pair, $T$ = 30; axes: counts. Each line: where a value of $r_0$ puts the counts, one point per value of the other context. "
             f"Top (2 bits): shading = the rule's answer; dot right,\ncross wrong (in the colour picked). "
             f"Bottom (all 5 bits): the two values naive Bayes confuses most, $r_0$ = {ra} and {rb}, seen from the two directions the flat rules weigh, "
             f"so both borders are\nexact straight lines; naive: its two centres and the line joining them. "
             f"Over all 10 values, line points right: naive {out[('naive', 5)]}, linear {out[('linear', 5)]} of 100.",
             fontsize = fs["tick"] - 1.5, color = "0.3", ha = "left", va = "bottom", linespacing = 1.3)
    return fig, dict(pair = g, bits2 = [b + 1 for b in b2], duel5 = (ra, rb),
                     right = {f"{a} {b} bits": v for (a, b), v in out.items()})


def main():
    from main.utils import fig_path
    fig, rec = figure()
    path = fig_path("naive_vs_linear_mechanism.png")
    fig.savefig(path, dpi = 300)
    print(path, rec)


if __name__ == "__main__":
    main()
