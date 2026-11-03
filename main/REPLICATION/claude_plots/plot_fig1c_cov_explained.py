"""What Cov(r_g, r_c) is: the covariance of the two variables under ONE belief (28 Oct 2026; John on the figure-8
wrap's colour: "by cov(r_g,r_c) do you mean covaraince of the marginal beliefs, covariance of beliefs in specific
realizations? I still need a bit more detail")
-> main/REPLICATION/figure_bin/fig1c_cov_explained.png

A belief is the Optimal's exact joint posterior p_t(r_g, r_c | o_1 .. o_t) over the 100 cells, for one stream (one
realization of the observations) at one step t. Its covariance
    Cov(r_g, r_c) = sum_gc p(g, c) g c - E[r_g] E[r_c],   E[r_g] = sum_gc p(g, c) g,   E[r_c] = sum_gc p(g, c) c
is one number per belief, read from its whole 10 x 10 table: not a covariance across streams or environments, and not
something the two marginals can give (their product has Cov = 0 by construction, so the Naive's beliefs and any
marginal read-out carry none). It is 0 for a belief on one cell or on cells that share a row or column, +1/4 d^2 for
half on each of two cells d apart along the main diagonal's direction, -1/4 d^2 along the other diagonal: +20.25 for
(0, 0) / (9, 9), -20.25 for (0, 9) / (9, 0), the extremes.
Row 1: real beliefs of the alias scan (the first stream-step meeting each description, environments in f8.ENVS order),
the joint table with its two marginals along the edges. Row 2, left: the marginals of the two wormhole beliefs, which
coincide, so only the joint tells the two wormholes apart; right: Cov over every belief the figure-8 wrap uses (39
environments x 5,000 streams, steps 8 - 30), log count.
In the wrap (plot_fig1c_figure8_wrap.figure), each surface point is coloured by the mean of this number over the beliefs
within 1.5 thicknesses of it (the 64 nearest at most).
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_cov_explained.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
import plot_fig1c_figure8 as f8                                                   # noqa: E402
import plot_fig1c_figure8_wrap as fw                                              # noqa: E402

R = 10


def _find(rule):
    """the first (environment, stream, step) meeting rule(P (n, T, 10, 10), cov (n, T)) -> bool (n, T)"""
    for e in f8.ENVS:
        p, cov = f8.posterior(e)
        P = p.reshape(*p.shape[:2], R, R)
        hit = np.argwhere(rule(P, cov))
        if len(hit):
            s, t = hit[0]
            return e, int(s), int(t), P[s, t], float(cov[s, t])
    raise ValueError("no belief meets the rule")


def _top2(P):
    flat = P.reshape(*P.shape[:-2], R * R)
    o = np.argsort(flat, -1)
    return o[..., -1], o[..., -2], np.take_along_axis(flat, o[..., -1:], -1)[..., 0], \
        np.take_along_axis(flat, o[..., -2:-1], -1)[..., 0]


def examples():
    def one_cell(P, cov):
        a, _, pa, _ = _top2(P)
        g, c = a // R, a % R
        ok = (pa >= .97) & (g >= 2) & (g <= 7) & (c >= 2) & (c <= 7)
        ok[:, :14] = False
        return ok

    def neighbours(P, cov):
        a, b, pa, pb = _top2(P)
        return (pa >= .42) & (pb >= .42) & (a // R == b // R) & (np.abs(a % R - b % R) == 1) & (a // R >= 2) \
            & (a // R <= 7)

    def broad(P, cov):
        ok = (1 / (P ** 2).sum((-2, -1)) >= 25) & (np.abs(cov) >= 1.) & (np.abs(cov) <= 2.5)
        ok[:, 1:] = False
        return ok

    def diagonal(P, cov):                                                     # a local spread (two-cell splits
        ne = 1 / (P ** 2).sum((-2, -1))                                        # far apart along a diagonal, other
        corners = P[..., 0, 0] + P[..., 9, 9] + P[..., 0, 9] + P[..., 9, 0]    # than the corners, almost never occur)
        return (cov >= 2) & (cov <= 6) & (ne >= 3) & (ne <= 10) & (corners < .02) & (P.max((-2, -1)) < .6)

    def worm(i, j):
        return lambda P, cov: (P[..., i // R, i % R] >= .44) & (P[..., j // R, j % R] >= .44)

    return [("On one cell", one_cell), ("On two neighbouring cells", neighbours),
            ("Broad (step 1)", broad), ("Spread along a diagonal", diagonal),
            ("Wormhole (0, 0) / (9, 9)", worm(0, 99)), ("Wormhole (0, 9) / (9, 0)", worm(9, 90))]


def figure():
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    ex = [(name, *_find(rule)) for name, rule in examples()]
    r = np.arange(R)
    PW, MG, GX, X0 = 2.05, .38, .55, .55
    W = X0 + 6 * (PW + MG) + 5 * GX + .3
    H = 8.4
    fig = plt.figure(figsize=(W, H))
    fig.text(.5, 1 - .22 / H, r"Cov$(r_g, r_c) = \Sigma_{g,c}\, p(g, c)\, g\, c - E[r_g]\, E[r_c]$ of ONE belief: the "
             r"Optimal's joint posterior $p_t(r_g, r_c \mid o_1, \ldots, o_t)$ of one stream at one step",
             ha="center", va="center", fontsize=FS["label"] + 1)
    fig.text(.5, 1 - .5 / H, "real beliefs of the alias scan (environment e, stream s, step t); edges: the two "
             "marginals; colour: probability (bright = more)", ha="center", va="center", fontsize=FS["label"])
    ytop = H - .95
    worms = []
    for i, (name, e, s, t, P, cv) in enumerate(ex):
        x = X0 + i * (PW + MG + GX)
        y = ytop - MG - PW
        ax = fig.add_axes([x / W, y / H, PW / W, PW / H])
        ax.imshow(P.T, origin="lower", cmap="viridis", vmin=0, vmax=P.max(), interpolation="nearest",
                  extent=(-.5, 9.5, -.5, 9.5))
        ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
        ax.tick_params(labelsize=FS["tick"], length=2, pad=1)
        ax.set_xlabel(r"$r_g$", labelpad=-6, fontsize=FS["label"])
        if i == 0:
            ax.set_ylabel(r"$r_c$", labelpad=-6, fontsize=FS["label"])
        top = fig.add_axes([x / W, (y + PW + .04) / H, PW / W, (MG - .06) / H])
        top.fill_between(r, P.sum(1), step="mid", color="0.35", lw=0)
        top.set_xlim(-.5, 9.5), top.set_ylim(0, 1), top.axis("off")
        side = fig.add_axes([(x + PW + .04) / W, y / H, (MG - .06) / W, PW / H])
        side.fill_betweenx(r, P.sum(0), step="mid", color="0.35", lw=0)
        side.set_ylim(-.5, 9.5), side.set_xlim(0, 1), side.axis("off")
        col = "#b2182b" if cv > 3 else "#2166ac" if cv < -3 else "k"
        fig.text((x + PW / 2) / W, (ytop + .2) / H, name, ha="center", va="center", fontsize=FS["label"])
        fig.text((x + PW / 2) / W, (y - .5) / H, f"Cov = {cv:+.2f}", ha="center", va="center",
                 fontsize=FS["label"] + 1, color=col)
        fig.text((x + PW / 2) / W, (y - .78) / H, f"e {e}, s {s}, t {t + 1}", ha="center", va="center",
                 fontsize=FS["tick"], color="0.35")
        if name.startswith("Wormhole"):
            worms.append((name, P, cv))
    y2, h2 = .75, 2.55
    ax = fig.add_axes([(X0 + .3) / W, y2 / H, 6.6 / W, h2 / H])
    for (name, P, cv), c, ls in zip(worms, ("#b2182b", "#2166ac"), ("-", (0, (4, 3)))):
        ax.plot(r, P.sum(1), color=c, lw=2, ls=ls, label=f"{name}: marginal of $r_g$")
        ax.plot(r, P.sum(0), color=c, lw=1, ls=ls, marker="o", ms=3, label=f"{name}: marginal of $r_c$")
    ax.set_xticks(range(R)), ax.set_ylim(0, .62)
    ax.set_xlabel("value", fontsize=FS["label"]), ax.set_ylabel("probability", fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS["tick"], frameon=False, loc="upper center", ncol=1)
    ax.set_title("the two wormhole beliefs have the same marginals:\nonly the joint (its Cov) tells them apart",
                 fontsize=FS["label"], pad=6)
    Z = fw.pooled()
    CV = Z["CV"][:, :, fw.FIG_T0 - 1:].ravel()
    edges = np.linspace(-20.5, 20.5, 83)
    cnt = np.histogram(CV, edges)[0]
    ax = fig.add_axes([(X0 + 8.6) / W, y2 / H, 8.6 / W, h2 / H])
    ax.step((edges[:-1] + edges[1:]) / 2, np.log10(np.maximum(cnt, 1)), where="mid", color="k", lw=1.4)
    for _, _, _, _, _, cv in ex:
        ax.axvline(cv, color="#b2182b" if cv > 3 else "#2166ac" if cv < -3 else "0.6", lw=.8, ls=":")
    ax.set_xlim(-21, 21)
    ax.set_xlabel(r"Cov$(r_g, r_c)$ of a belief", fontsize=FS["label"])
    ax.set_ylabel(r"log$_{10}$ number of beliefs", fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    far = np.mean(np.abs(CV) > 8)
    ax.set_title(f"every belief the wrap uses: {len(CV):,} (39 environments x 5,000 streams, steps "
                 f"{fw.FIG_T0} - 30);\n{100 * np.mean(np.abs(CV) < 1):.1f}% within $\\pm$1, {100 * far:.2f}% beyond "
                 r"$\pm$8 (dotted: the examples)", fontsize=FS["label"], pad=6)
    path = fig_path("fig1c_cov_explained.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    for name, e, s, t, P, cv in ex:
        print(f"{name:26s} e {e:2d} s {s:4d} t {t + 1:2d}: Cov {cv:+.2f}, E[r_g] {P.sum(1) @ np.arange(R):.2f}, "
              f"E[r_c] {P.sum(0) @ np.arange(R):.2f}, max p {P.max():.2f}")
    return path


if __name__ == "__main__":
    print("saved", figure())
