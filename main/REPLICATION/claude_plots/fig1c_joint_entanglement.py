"""Joint entanglement: the two variables' entanglements, their average and their product, against the grid border, the
crossings, the wormhole and the trained network (29 Oct 2026; John: "maybe entanglement happens when one of the bayes
(Optimal or Naive) has an uncoupled crossing while the other doesn't, hence why it is strongest at the grid border.
Maybe the wormhole is specifically for joint-entanglement (the product of each variable's entanglement, instead of the
average). I wonder if this would seperate border dynamics and wormhole dynamics.")
-> main/REPLICATION/figure_bin/fig1c_joint_entanglement.png

Entanglement per variable. The stored entanglement (fig1.alias_sim) already IS an average over the two variables:
rev_t = (e_g,t + e_c,t) / 2, e_k,t = 0.5 Jeffreys(the Naive's normalized per-step likelihood for variable k, the
table averaged over the other variable; the Optimal's effective update of k's marginal, b_k,t / b_k,t-1 normalized):
how differently the two Bayesian observers update variable k on the same observation. rev_vars() recomputes e_g, e_c
exactly (float64) from the stored observations and tables; their mean equals the stored rev (max relative difference
5e-8). Cached in DATA/experimental/fig1c_compression/figdata/round9_joint_entanglement/rev_vars.npz (40 environments,
goal-ordered [goal, context]); the figure uses the 39 of fig1c_measures. Joint entanglement: the product, taken
as the sum of logs (John: "you may need to use sum of logs instead of multiplication"): per step the product and the
sum of logs rank steps identically, so they differ only where values are averaged (the maps); every quantity is
averaged in log10. "Double-centred log10": log10 x minus the same cell's typical value at that step (its 50 streams),
minus the stream's own mean over steps 2 - 30 (entanglement rises over steps 1 - 8). Crossings as in fig1c_crossings (a
marginal moving >= 0.25 of its mass between its ends); a crossing disagreement: one of the Optimal and the Naive crosses
a variable on a step and the other does not cross it. Wormhole state: the Optimal's |Cov(r_g, r_c)| > 8
(figure8_pooled.npz); entry: not one at t - 1, one at t.
a - d  mean log10 over all streams and steps 2 - 30 by the true cell: e_g, e_c, the average, half the sum of logs
       (= log10 of the geometric mean, so in the same units as the average).
e  the mean log10 average by the true cell's distance to the border (d = 0: either variable at 0 or 9), all steps vs
   without every crossing-disagreement step and its two neighbours.
f  the double-centred sum of logs around wormhole entries, the Optimal's coupled crossings, lone (one-variable)
   crossings of the Optimal only and of the Naive only; thin: the Naive-only crossings' two variables.
g  the trained RNN's activity effective dimension (minus typical), steps 8 - 30, on the plane of the two variables'
   double-centred log10 entanglements (the sum of logs runs along the anti-diagonal); h  the share of those
   stream-steps in a wormhole state.

Measured 29 Oct (39 environments x 5,000 streams x steps 2 - 30):
- Maps (mean log10 per step; corner / edge / interior): e_g is high where the CONTEXT value sits at an end, e_c where
  the goal value does: each variable's Naive update is most wrong when the OTHER variable is extreme (the 10-18 border
  mechanism, per variable). Average -0.481 / -0.591 / -0.864 (corner - edge +0.11, edge - interior +0.27); half the sum
  of logs -0.536 / -0.668 / -0.932 (+0.13, +0.26). The sum of logs draws nearly the same ring; the corners gain only
  0.02 log10 over the edges, because the two variables are a little more balanced there (|log10 e_g - log10 e_c|
  corner 0.36 / edge 0.43 / interior 0.41). In raw units the same (geometric mean corner / edge 1.24, average 1.20).
  log10 e_g and log10 e_c correlate r 0.45 over stream-steps.
- Crossing disagreements: 0.94% of stream-steps; 1.5% of the top-10% entangled steps (top 1%: 0.6%). Without them
  (and their neighbours, 2.2% of steps) the border profile is unchanged (mean log10 average at distance 0 - 4: -0.579
  / -0.739 / -0.895 / -1.026 / -1.100 vs -0.584 / -0.741 / -0.896 / -1.026 / -1.100). They come on entangled updates,
  but they do not make the border.
- Sum of logs around events (double-centred, lags -1 / 0 / +1): Naive-only lone crossings (35,637) +0.028 / +0.333 /
  +0.013, of which the OTHER variable +0.261 and the crossing variable +0.072 at 0; wormhole entries (18,982) +0.107 /
  +0.271 / +0.059 (goal +0.137, context +0.133); the Optimal's coupled crossings (5,528) +0.160 / -0.074 / +0.039;
  the Optimal-only lone crossings (11,631) +0.169 / +0.038 / +0.017. The Naive's lone border crossings are the most
  jointly entangled steps, more than the wormhole entries. At the Naive's lone crossings the Naive moves the crossing
  variable 3.4 cells, the Optimal moves neither (0.41 / 0.21 cells, typical 0.20); true cells mostly on the other
  variable's edge (56 - 60%), corners 12 - 14%.
- Both variables in their own top 10% (an AND; identical for the product and the sum of logs): wormhole entries 3.7%,
  inside a split 2.3%, all steps 3.4%; the Naive's lone crossings 7%. Wormhole entries are not jointly entangled.
- Network (steps 8 - 30): the activity effective dimension follows mostly the LARGER of the two (L-shaped, not along
  the anti-diagonal of the sum of logs): +1.8 - 2.5 dims when one variable's double-centred entanglement is >= 0.5,
  +2.9 when both are, -1.0 to -1.4 when both are low; sub-additive: (both - neither) / (one - neither) = 1.33
  [1.22 - 1.44] for the activity dimension, 1.42 [1.36 - 1.49] for the cross-band share (pure max 1, additive 2).
  The wormhole state is most common at typical values (0.7 - 0.75% of stream-steps there, 0.2 - 0.3% at the extremes).
- VERIFIED 29 Oct (2 independent agents, own float64 code from obs + L; workflow wu23mp2x5): all claims reproduce.
  Corrections: the product / min make the corners 3 - 8% sharper than the average (geometric mean higher corner / edge
  in 33/39 environments), so "the same ring" overstates; the network response is sub-additive, not a pure max. Extra:
  a static table calculation (the Naive's averaged likelihood vs the likelihood given the other variable's TRUE value)
  reproduces the maps at r 0.91 - 0.92: the ring is table geometry. Ring intact on the 62.6% of steps with no
  end-to-end flow at all. No joint definition (product, min, harmonic, AND of top 5/10/25%, balance) beats the
  average for wormhole entries (AUC double-centred .644 average, .645 max, .633 product, .600 min, AND ~.50); Naive-only
  crossings are at least as entangled as entries under every definition (AUC <= .50). What separates border and
  wormhole dynamics is where and when: Naive-only crossings edge 66% / corner 13%, median step 11; wormhole entries
  corner 30%, median step 4.
=> Entanglement is either variable's update disagreement (an OR); the product / sum of logs adds nothing that separates
   the wormhole from the border. What separates them is the belief state (|Cov|, the split) and its timing
   (fig1c_wormhole_timing.py, fig1c_figure8_learning.py), not the update's entanglement.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/fig1c_joint_entanglement.py"""
import os
import sys
import warnings
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
import fig1c_two_processes as tp                                                  # noqa: E402
import fig1c_crossings as fc                                                      # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)
mm = tp.mm
OUT = os.path.join(mm.SCAN, "figdata", "round9_joint_entanglement")
CACHE = os.path.join(OUT, "rev_vars.npz")
R, T = 10, 30
F64 = np.float64
LAGS = np.arange(-3, 4)
EDGES = np.array([-np.inf, -.3, -.15, -.05, .05, .15, .3, .5, np.inf])


def _approx_lik(b, eps=1e-99):
    """supplementary_plotter2.approx_lik: the effective per-step update b_t / b_t-1, normalized (step 0: b_0)"""
    b = np.maximum(b.astype(F64), eps)
    px = np.zeros_like(b)
    px[:, 0], r = b[:, 0], b[:, 1:] / b[:, :-1]
    px[:, 1:] = r / r.sum(-1, keepdims=True)
    return px


def _jeffreys(p, q, eps=1e-99):
    """supplementary_plotter2.step_dkl: 0.5 (KL(p || q) + KL(q || p)) over the last axis"""
    p, q = np.maximum(p.astype(F64), eps), np.maximum(q.astype(F64), eps)
    p, q = p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)
    return .5 * np.sum(p * (np.log(p) - np.log(q)) + q * (np.log(q) - np.log(p)), -1)


def rev_vars():
    """(40, 5000, 30, 2) float32 goal-ordered [e_goal, e_context], recomputed from scan_meta's observations and tables
    as fig1.alias_sim computes rev (whose mean over the two variables it reproduces), cached"""
    if not os.path.exists(CACHE):
        os.makedirs(OUT, exist_ok=True)
        meta = np.load(os.path.join(mm.SCAN, "scan_meta.npz"))
        out = np.zeros(meta["rev"].shape + (2,), np.float32)
        md = []
        for e in range(len(meta["goal"])):
            L, g, o = meta["L"][e].astype(F64), int(meta["goal"][e]), meta["obs"][e].astype(F64)
            cum = np.cumsum(np.einsum("nti,ijk->ntjk", o, np.log(L))
                            + np.einsum("nti,ijk->ntjk", 1 - o, np.log1p(-L)), 1)
            post = np.exp(cum - cum.max((-1, -2), keepdims=True))
            post /= post.sum((-1, -2), keepdims=True)
            joint = np.stack([post.sum(-1), post.sum(-2)], 2)
            Lb = (L.mean(2), L.mean(1))
            px = np.stack([np.exp(o @ np.log(Lb[k]) + (1 - o) @ np.log1p(-Lb[k])) for k in (0, 1)], 2)
            rk = np.stack([_jeffreys(px[:, :, k] / px[:, :, k].sum(-1, keepdims=True), _approx_lik(joint[:, :, k]))
                           for k in (0, 1)], -1)
            md.append(np.abs(rk.mean(-1) - meta["rev"][e]).max() / meta["rev"][e].max())
            out[e] = np.stack([rk[..., g], rk[..., 1 - g]], -1)
        assert max(md) < 1e-5, f"per-variable entanglement does not reproduce rev ({max(md):.1e})"
        np.savez(CACHE, rev_vars=out, maxreldiff=np.array(md))
    return np.load(CACHE)["rev_vars"][mm.ENVS].astype(F64)


def dc(x):
    """double-centred log10: minus the same cell's typical value at that step, minus the stream's own mean (step 1
    NaN)"""
    lx = np.log10(x)
    lx[:, :, 0] = np.nan
    d = fc.departures(lx)
    return d - np.nanmean(d, 2, keepdims=True)


def crossing_flags():
    """per-variable crossing flags (E, S, T, 2) for the Optimal and the Naive"""
    P = fc.primitives()
    return {a: np.stack([fc.at_steps(np.abs(P[a][0][..., k]) >= fc.CYCLE) for k in (0, 1)], -1)
            for a in ("Joint", "Naive")}


def cellmap(x, keep=None):
    """mean over streams and steps 2 - 30 by the true cell -> (10, 10) rows r_g, columns r_c"""
    meta = np.load(os.path.join(mm.SCAN, "scan_meta.npz"))
    x = x if keep is None else np.where(keep, x, np.nan)
    m = np.nanmean(x[:, :, 1:], (0, 2))
    co = meta["cell_of"]
    return (np.bincount(co, m, minlength=R * R) / np.bincount(co, minlength=R * R)).reshape(R, R)


def trig(ev, X):
    """event-triggered mean of X at LAGS (events at arrival steps)"""
    e_, s_, t_ = np.nonzero(ev)
    out = []
    for L in LAGS:
        tt = t_ + L
        ok = (tt >= 1) & (tt < T)
        out.append(np.nanmean(X[e_[ok], s_[ok], tt[ok]]))
    return np.array(out), len(e_)


def plane(Wg, Wc, X, mask, min_n=200):
    """mean of X over EDGES x EDGES bins of (Wg rows, Wc columns) where mask; bins with < min_n NaN"""
    nb = len(EDGES) - 1
    ig = np.clip(np.searchsorted(EDGES, Wg, side="right") - 1, 0, nb - 1)
    ic = np.clip(np.searchsorted(EDGES, Wc, side="right") - 1, 0, nb - 1)
    f = mask & np.isfinite(Wg) & np.isfinite(Wc) & np.isfinite(X)
    idx = ig[f] * nb + ic[f]
    n = np.bincount(idx, minlength=nb * nb)
    M = np.bincount(idx, X[f], minlength=nb * nb) / np.maximum(n, 1)
    M[n < min_n] = np.nan
    return M.reshape(nb, nb), n.reshape(nb, nb)


def figure():
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    rv = rev_vars()
    E, S = rv.shape[:2]
    eg, ec = rv[..., 0], rv[..., 1]
    lg, lc = np.log10(eg), np.log10(ec)
    Wg, Wc = dc(eg), dc(ec)
    fl = crossing_flags()
    O, N = fl["Joint"], fl["Naive"]
    cv = np.load(os.path.join(mm.R7, "figure8_pooled.npz"))["CV"].astype(F64)
    worm = np.abs(cv) > 8
    entry = np.zeros_like(worm)
    entry[:, :, 1:] = ~worm[:, :, :-1] & worm[:, :, 1:]
    meta = np.load(os.path.join(mm.SCAN, "scan_meta.npz"))
    cells = meta["cells"]
    dist = np.minimum(np.minimum(cells[:, 0], 9 - cells[:, 0]), np.minimum(cells[:, 1], 9 - cells[:, 1]))
    corner = np.zeros(R * R, bool)
    corner[[0, 9, 90, 99]] = True
    edge = (dist == 0) & ~corner

    W_, H_ = 17.0, 8.9
    fig = plt.figure(figsize=(W_, H_))

    def ax_at(x, y, w, h):
        return fig.add_axes([x / W_, y / H_, w / W_, h / H_])

    fig.text(.5, 1 - .3 / H_, "Entanglement is mostly either variable's disagreement, not both: the sum of logs draws "
             "nearly the same border ring as the average, and neither the network nor the wormhole follows it",
             ha="center", va="center", fontsize=FS["label"] + 3)
    fig.text(.5, 1 - .64 / H_, f"alias scan, {E} environments x {S:,} streams x steps 2 - 30; $e_k$: how differently "
             r"the Naive and the Optimal update variable $k$ on the same observation, $\frac{1}{2}$Jeffreys (nats); "
             r"the stored entanglement is $\frac{1}{2}(e_g + e_c)$", ha="center", va="center", fontsize=FS["tick"])
    # a - d: maps by the true cell, every quantity in log10 nats
    maps = ((r"a   Goal variable  $\log_{10} e_g$", lg), (r"b   Context variable  $\log_{10} e_c$", lc),
            (r"c   Average  $\log_{10} \frac{1}{2}(e_g + e_c)$", np.log10((eg + ec) / 2)),
            (r"d   Sum of logs  $\frac{1}{2}(\log_{10} e_g + \log_{10} e_c)$", (lg + lc) / 2))
    for j, (title, x) in enumerate(maps):
        M = cellmap(x)
        ax = ax_at(.85 + 3.55 * j, 5.0, 2.55, 2.55)
        im = ax.imshow(M, origin="lower", cmap="viridis", vmin=-1.2, vmax=-.45, interpolation="nearest")
        ax.set_xticks([0, 9])
        ax.set_yticks([0, 9])
        ax.tick_params(labelsize=FS["tick"])
        ax.set_xlabel("true context value $r_c$", fontsize=FS["tick"] + .5)
        if j == 0:
            ax.set_ylabel("true goal value $r_g$", fontsize=FS["tick"] + .5)
        ax.set_title(title, fontsize=FS["label"] + .5, loc="left", pad=5)
        Mr = M.ravel()
        c_, e_, i_ = Mr[corner].mean(), Mr[edge].mean(), Mr[dist > 0].mean()
        ax.text(1.04, .5, f"corner - edge\n{c_ - e_:+.2f}\n\nedge - interior\n{e_ - i_:+.2f}",
                transform=ax.transAxes, ha="left", va="center", fontsize=FS["tick"] - .5)
    cax = ax_at(15.25, 5.0, .14, 2.55)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("mean per step (log$_{10}$ nats)", fontsize=FS["tick"])
    cb.ax.tick_params(labelsize=FS["tick"] - 1)
    # e: border profile with and without crossing disagreements
    dis = ((N & ~O) | (O & ~N)).any(-1)
    near = dis.copy()
    near[:, :, 1:] |= dis[:, :, :-1]
    near[:, :, :-1] |= dis[:, :, 1:]
    lav = np.log10((eg + ec) / 2)
    ax = ax_at(.85, .85, 3.0, 2.85)
    prof = {}
    for lab, keep, col, ls in (("all steps", None, "k", "-"),
                               (f"without crossing disagreements\n(and their neighbours, "
                                f"{100 * near[:, :, 1:].mean():.1f}% of steps)", ~near, "#1b9e77", (0, (4, 2)))):
        M = cellmap(lav, keep).ravel()
        prof[lab] = [M[dist == d].mean() for d in range(5)]
        ax.plot(range(5), prof[lab], color=col, ls=ls, lw=2, marker="o", ms=4, label=lab)
    ax.set_xticks(range(5))
    ax.set_xlabel("true cell's distance to the border (cells)", fontsize=FS["tick"] + .5)
    ax.set_ylabel(r"$\log_{10} \frac{1}{2}(e_g+e_c)$, mean per step", fontsize=FS["tick"] + .5)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="upper right")
    ax.set_title("e   Crossing disagreements do not make the border", fontsize=FS["label"] + .5, loc="left", pad=5)
    # f: the sum of logs (double-centred) around events; the Naive's lone crossings split into their two variables
    Nun = N & ~N[..., ::-1]
    lone_n = Nun & ~O.any(-1, keepdims=True)
    Oun = O & ~O[..., ::-1]
    lone_o = Oun & ~N.any(-1, keepdims=True)
    W2 = np.stack([Wg, Wc], -1)
    Wsum = Wg + Wc

    def split(ev2):
        own, oth, n = np.zeros(len(LAGS)), np.zeros(len(LAGS)), 0
        for k in (0, 1):
            r1, n1 = trig(ev2[..., k], W2[..., k])
            r2, _ = trig(ev2[..., k], W2[..., 1 - k])
            own, oth, n = own + r1 * n1, oth + r2 * n1, n + n1
        return own / n, oth / n, n

    own, oth, n_l = split(lone_n)
    ax = ax_at(4.75, .85, 3.0, 2.85)
    curves = {}
    for lab, ev, col, mk in (("wormhole entry (Optimal)", entry, "#5e3c99", "s"),
                             ("coupled crossing (Optimal)", O[..., 0] & O[..., 1], "k", "D"),
                             ("lone crossing, Optimal only", lone_o.any(-1), "#1f77b4", "^"),
                             ("lone crossing, Naive only", lone_n.any(-1), "0.45", "o")):
        r, n = trig(ev, Wsum)
        curves[lab] = (r, n)
        ax.plot(LAGS, r, color=col, lw=2, marker=mk, ms=3.5, label=f"{lab} ({n:,})")
    ax.plot(LAGS, oth, color="0.45", lw=1, ls="-", label="   its other variable")
    ax.plot(LAGS, own, color="0.45", lw=1, ls=(0, (3, 2)), label="   its crossing variable")
    ax.axhline(0, color="0.6", lw=.8, ls=":")
    ax.axvline(0, color="0.3", lw=.6, ls=":")
    ax.set_xticks(LAGS)
    ax.set_xlabel("steps from the event", fontsize=FS["tick"] + .5)
    ax.set_ylabel(r"$\log_{10} e_g + \log_{10} e_c$, double-centred", fontsize=FS["tick"] + .5)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-.1, .56)
    ax.legend(fontsize=FS["tick"] - 1.5, frameon=False, loc="upper left", ncol=1, handlelength=2.2,
              borderaxespad=.2, labelspacing=.25)
    ax.set_title("f   Sum of logs around the events", fontsize=FS["label"] + .5, loc="left", pad=5)
    # g, h: the plane of the two variables' entanglements
    V = tp.series()
    st = np.zeros(Wg.shape, bool)
    st[:, :, 7:] = True
    Mpr, n_pl = plane(Wg, Wc, tp.dep(V["pr"]), st)
    Mw, _ = plane(Wg, Wc, worm.astype(F64), st)
    labs = ["$<$ -.3", "", "", "0", "", ".3", "", "$>$ .5"]
    panels = (("g   Network activity: mostly the larger one", Mpr, "PiYG", -3, 3,
               "activity effective dimension\nminus typical"),
              ("h   Wormhole states sit at typical values", 100 * Mw, "viridis", 0, .8,
               "% of stream-steps\nin a wormhole state"))
    for j, (title, M, cmap, vmin, vmax, clab) in enumerate(panels):
        ax = ax_at(9.0 + 4.0 * j, .85, 2.75, 2.85)
        im = ax.imshow(M, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, fontsize=FS["tick"] - .5)
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs, fontsize=FS["tick"] - .5)
        ax.set_xlabel("context: $\\log_{10} e_c$, double-centred", fontsize=FS["tick"])
        ax.set_ylabel("goal: $\\log_{10} e_g$, double-centred", fontsize=FS["tick"])
        ax.set_title(title, fontsize=FS["label"] + .5, loc="left", pad=5)
        cb = fig.colorbar(im, ax=ax, fraction=.06, pad=.03)
        cb.set_label(clab, fontsize=FS["tick"] - .5)
        cb.ax.tick_params(labelsize=FS["tick"] - 1)
    fig.text(9.0 / W_, .3 / H_, f"g, h: the Trained RNN, steps 8 - 30, {int(n_pl.sum()):,} stream-steps; the sum of "
             "logs runs along the anti-diagonal; bins with $<$ 200 stream-steps blank", ha="left", va="center",
             fontsize=FS["tick"] - .5)
    path = fig_path("fig1c_joint_entanglement.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("saved", path)
    for lab, v in prof.items():
        print("e", lab.split("\n")[0], " ".join(f"{x:.3f}" for x in v))
    for lab, (r, n) in curves.items():
        print("f", lab, n, " ".join(f"{x:+.3f}" for x in r))
    print("f Naive lone own / other:", " ".join(f"{x:+.3f}" for x in own), "|", " ".join(f"{x:+.3f}" for x in oth))
    print("g pr:\n", np.array2string(Mpr, precision=2, max_line_width=200))
    print("h worm %:\n", np.array2string(100 * Mw, precision=2, max_line_width=200))
    return path


def movie(vs_typical=False):
    """particles and grids coloured by max entanglement, the larger of the two variables' (John 10-29) ->
    fig1c_movie_max_entanglement{,_vs_typical}.gif. Left: 15,600 streams (4 per cell, every environment) as 2-px
    squares at the Trained RNN's posterior means, each its own value; right: all 39 environments' grids on top of each
    other (cell means over all 50 streams), every edge coloured by its two cells' mean, 1-px lines drawn shortest to
    longest. Colour: log10 max(e_g, e_c) on a fixed scale (2nd - 98th percentile of what each panel draws), viridis;
    vs_typical: minus the same cell's mean over the 39 environments at that step (coolwarm). Timing and palette as
    plot_fig1c_measures_movie"""
    import plot_fig1c_measures_movie as pm
    from matplotlib.collections import LineCollection
    rv = rev_vars()
    E = rv.shape[0]
    mx = rv.max(-1)                                                                   # (39, 5000, T)
    B = np.load(os.path.join(mm.SCAN, "scan_Trained.npz"))["beliefs"][mm.ENVS].astype(np.float32)
    r = np.arange(R, dtype=np.float32)
    X = np.stack([B[..., :R] @ r, B[..., R:] @ r], -1)                                # (39, 5000, T, 2)
    del B
    NSC = 50
    Xc = X.reshape(E, R * R, NSC, T, 2).mean(2)                                       # (39, 100, T, 2)
    vc = np.log10(mx.reshape(E, R * R, NSC, T).mean(2))                               # log of each cell's mean
    show = np.arange(X.shape[1]) % NSC < pm.NS_SHOW
    Xs, vs = X[:, show], np.log10(mx[:, show])                                        # (39, 400, T, ...)
    if vs_typical:
        vc = vc - vc.mean(0, keepdims=True)
        cid = np.arange(X.shape[1])[show] // NSC
        typ = np.log10(mx).reshape(E, R * R, NSC, T).mean((0, 2))                     # (100, T)
        vs = vs - typ[cid][None]
    order = np.random.default_rng(0).permutation(E * Xs.shape[1])
    edges, lattice = pm._grid()
    cmap = plt.get_cmap("coolwarm" if vs_typical else "viridis")
    if vs_typical:
        lims = [(-q, q) for q in (np.percentile(np.abs(vs), 98), np.percentile(np.abs(vc), 98))]
    else:
        lims = [tuple(np.percentile(v, [2, 98])) for v in (vs, vc)]
    PW, GX, X0, YTOP = 4.3, .55, .5, 1.25
    W, H = X0 + 2 * PW + GX + 1.9, YTOP + PW + 1.0
    dpi = 150
    unit = "log10 max entanglement" + (" minus the cell's typical" if vs_typical else " (nats)")
    with plt.rc_context(pm.RC):
        fig = plt.figure(figsize=(W, H), dpi=dpi, facecolor="white")
        fig.text(X0 / W, (H - .78) / H, r"Max entanglement $\max(e_g, e_c)$, the larger of the two variables' "
                 "update disagreements" + (", minus each cell's typical value at that step" if vs_typical else ""),
                 fontsize=11, ha="left", va="center")
        axes = []
        for j, title in enumerate(("Particles: each stream at the Trained RNN's posterior mean",
                                   "Grids: all 39 environments, edges by their cells' mean")):
            x = X0 + j * (PW + GX)
            ax = fig.add_axes([x / W, (H - YTOP - PW) / H, PW / W, PW / H])
            ax.add_collection(LineCollection(lattice[edges], colors="0.85", linewidths=.6, linestyles=(0, (2, 2)),
                                             zorder=0))
            ax.set_xlim(-.6, 9.6), ax.set_ylim(-.6, 9.6), ax.set_aspect("equal")
            ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
            ax.tick_params(labelsize=8, length=2, pad=1)
            ax.set_xlabel(r"$E[r_g]$", labelpad=-4)
            if j == 0:
                ax.set_ylabel(r"$E[r_c]$", labelpad=-4)
            ax.set_title(title, fontsize=9.5, pad=4)
            lo_k, hi_k = lims[j]
            cax = fig.add_axes([x / W, (H - YTOP - PW - .55) / H, PW / W, .08 / H])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(lo_k, hi_k), cmap=cmap), cax=cax,
                              orientation="horizontal")
            cb.set_ticks([lo_k, (lo_k + hi_k) / 2, hi_k])
            cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2g}"))
            cb.ax.tick_params(labelsize=7.5, length=2, pad=1)
            cb.outline.set_visible(False)
            cb.set_label(unit, fontsize=8, labelpad=1)
            axes.append(ax)
        sc = axes[0].scatter(*Xs[:, :, 0].reshape(-1, 2)[order].T, s=(72 / dpi * pm.PX_P) ** 2, marker="s",
                             c=cmap(np.full(len(order), .5)), lw=0, zorder=2, antialiased=False)
        lc = axes[1].add_collection(LineCollection([], linewidths=72 / dpi, antialiased=False, zorder=1))
        clock = pm._clock(fig, W, H, X0, X0 + 2 * PW + GX)
        fig.text((X0 + 2 * PW + GX + .2) / W, (H - .42) / H, f"{E} environments;\nparticles: {pm.NS_SHOW * R * R} of "
                 f"each one's\n5,000 streams; grids:\ncell means of all 50", fontsize=9, ha="left", va="center")
        frames = []
        for i, tau in enumerate(pm._taus()):
            P = pm._blend(Xs, tau, 2).reshape(-1, 2)[order]
            v = pm._lin(vs, tau).ravel()[order]
            lo_k, hi_k = lims[0]
            sc.set_offsets(P)
            sc.set_facecolor(cmap((np.clip(v, lo_k, hi_k) - lo_k) / (hi_k - lo_k)))
            seg = pm._blend(Xc, tau, 2)[:, edges].reshape(-1, 2, 2)
            by_len = np.argsort(np.linalg.norm(seg[:, 1] - seg[:, 0], axis=-1), kind="stable")
            ve = pm._lin(vc, tau)[:, edges].mean(-1).ravel()[by_len]
            lo_k, hi_k = lims[1]
            lc.set_segments(seg[by_len])
            lc.set_color(cmap((np.clip(ve, lo_k, hi_k) - lo_k) / (hi_k - lo_k)))
            clock(tau)
            frames.append(pm._grab(fig))
            if i % 100 == 0:
                print(f"frame {i + 1}", flush=True)
        plt.close(fig)
    return pm._save(frames, cmap, "fig1c_movie_max_entanglement%s.gif" % ("_vs_typical" if vs_typical else ""),
                    levels=64)


if __name__ == "__main__":
    args = sys.argv[1:] or ["figure"]
    if "figure" in args:
        figure()
    if "movie" in args:
        movie()
    if "movie_vs_typical" in args:
        movie(vs_typical=True)
