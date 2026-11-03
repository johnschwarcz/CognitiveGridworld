"""Flattening toward uniform vs divergence, in the pairs of the combined chaos figure (John 11-03).

John 11-03 (verbatim): "For the ratios, im wondering if we can seperate out flattening to uniform belief from divergence
to meaningful differences. Maybe we can do this by making a heatmap where the y-axis is the ratio of divergence with a
uniform marginal. So we can see the new ratios with the previous and uniform, as a function of the previous distance
from each."

-> main/REPLICATION/figure_bin/td_tt_chaos_flatten.png (figure()); numbers() prints every number below. CPU, ~1 min.

DEFINITIONS (the pairs of td_tt_chaos_combined.py: two probes of one test environment at step t with the identical
observation vector at t; a case = one variable of a pair-step; end of training; cases inside the 40 bins of the pair
distance before, 0.25-119 nats)
  Pair ratio R_D = d_H(p_t, q_t) / d_H(p_{t-1}, q_{t-1}) (row 2 of td_tt_chaos_pairs.png).
  Distance from uniform U(p) = d_H(p, uniform) = max_r log p_r - min_r log p_r (the belief's own range of log values;
    0 for a uniform belief). Uniform ratio of one belief R_U(p) = U(p_t) / U(p_{t-1}): < 1 it flattens toward uniform,
    > 1 it sharpens. The pair's uniform ratio = the geometric mean of its two beliefs' (y of the heat maps).
  Groups: both beliefs flatten (both R_U < 1), one flattens (mixed), both sharpen.
  Tempering (both log beliefs scaled by the same factor a about their means) moves R_D and R_U equally (both = a);
    a shared translation of the log beliefs (naive's update) leaves R_D = 1 whatever R_U does.
WHAT IS DRAWN
  Row 1 (heat maps): x = the pair distance before (20 log bins, 0.25-119 nats), y = the pair's uniform ratio (log
    bins), colour = the geometric mean of R_D in the cell (red > 1 apart, blue < 1 together; cells with >= 50 cases);
    black line: the geometric mean of the uniform ratio per x bin (where the cases are).
  Row 2: the same with x = the pair's distance from uniform before (geometric mean of the two beliefs' U).
  Row 3 (lines): the geometric mean of R_D per x bin of row 1 for the three groups (5-95% over the 1,000 shared
    environment resamples; drawn where a group holds >= 100 cases), and all cases (grey, = row 2 of
    td_tt_chaos_pairs.png).
RESULTS (numbers(); geometric means [5-95%]; Bayes / Trained RNN / ESN)
  1 Flattening is rare: both beliefs flatten in .015 / .014 / .003 of the cases, one in .153 / .158 / .069; both sharpen
    in .832 / .828 / .928 (naive .822). Beliefs sharpen x1.12 / x1.12 / x1.14 per step on average (naive x1.12).
  2 The pairs separate while both beliefs sharpen: pairs 0.25-1 nat apart (95-98% both sharpening) x1.53 [x1.46, x1.64]
    / x1.58 [x1.50, x1.67] / x1.35 [x1.33, x1.37]; 1-3 nats x1.15 / x1.12 / x1.20. Where both flatten they separate less
    (0.25-1 nat x1.44 / x1.40 / x1.30; 3-10 nats x.95 / x.96 / x1.04, i.e. they come closer). One flattening gives the
    largest factors (0.25-1 nat x2.58 [x2.22, x3.04] / x2.56 [x2.27, x2.89] / x1.47) but holds 1-3% of the close pairs.
  3 Sharpening is mostly shared, not separating: the mean log pair ratio is smaller than the mean log uniform ratio
    (+.030 = +.114 - .084; +.023 = +.109 - .086; +.062 = +.131 - .069), and naive sharpens as much (x1.12) with every
    pair ratio exactly 1 (T1: a shared translation). Per case, corr(ln R_D, ln R_U) +.31 / +.27 / +.54.
  So the pairs' growth is not flattening toward uniform; it happens while both beliefs sharpen, by the part of the
  update the two beliefs do not share.
"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_chaos_combined as cc                                                             # noqa: E402
import td_tt_chaos_pairs as tp                                                                # noqa: E402

AGENTS = ("bayes", "trained", "echo", "naive")
GROUPS = ("both flatten", "one flattens", "both sharpen")
GCOL = ("#2166ac", "#7f7f7f", "#b2182b")
MIN_CELL, MIN_LINE, NX, NY = 50, 100, 20, 24
_D = {}


def data():
    """per agent, the cases inside the 40 bins: ln R_D, ln R_U (pair geometric mean), group, d_H before, U before (pair
    geometric mean), environment"""
    if _D:
        return _D
    P = cc.pairs()
    e40 = P["edges40"]
    with np.load(cc.UNITS) as Z, np.load(tp.LQ_CACHE) as LQ:
        I, J, S = Z["I"], Z["J"], Z["t"]
        env = np.repeat(Z["env"], tp.NV)
        for k in AGENTS:
            U = tp.osc(LQ[k])                                                    # (10,000, 30, 2): range of log p
            up0, up1, uq0, uq1 = U[I, S - 2], U[I, S - 1], U[J, S - 2], U[J, S - 1]
            pre, post = Z[f"pre_{k}"].ravel(), Z[f"post_{k}"].ravel()
            ok = (pre >= e40[0]) & (pre < e40[-1])
            lp, lq = np.log(up1 / up0).ravel(), np.log(uq1 / uq0).ravel()
            g = np.where((lp < 0) & (lq < 0), 0, np.where((lp > 0) & (lq > 0), 2, 1))
            _D[k] = dict(lrd=np.log(post / pre)[ok], lru=((lp + lq) / 2)[ok], grp=g[ok], x1=pre[ok],
                         x2=np.sqrt(up0 * uq0).ravel()[ok], env=env[ok], umin=float(min(up0.min(), uq0.min())))
    return _D


def cells(x, y, v, xe, ye):
    """the mean of v per (x, y) cell (NaN below MIN_CELL cases) and the counts"""
    bx, by = np.digitize(x, xe) - 1, np.digitize(y, ye) - 1
    nx, ny = len(xe) - 1, len(ye) - 1
    ok = (bx >= 0) & (bx < nx) & (by >= 0) & (by < ny)
    key = bx[ok] * ny + by[ok]
    n = np.bincount(key, minlength=nx * ny).reshape(nx, ny)
    s = np.bincount(key, v[ok], nx * ny).reshape(nx, ny)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= MIN_CELL, s / n, np.nan), n


def edges():
    D = data()
    xe1 = np.logspace(np.log10(cc.pairs()["edges40"][0]), np.log10(cc.pairs()["edges40"][-1]), NX + 1)
    x2 = np.concatenate([D[k]["x2"] for k in AGENTS[:3]])
    xe2 = np.logspace(*np.log10(np.percentile(x2, [.5, 99.5])), NX + 1)
    y = np.concatenate([D[k]["lru"] for k in AGENTS])
    ye = np.linspace(*np.percentile(y, [.5, 99.5]), NY + 1)
    return xe1, xe2, ye


def lines(k, xe):
    """per x bin of the pair distance before and group: geometric mean of R_D with the resamples, and all cases"""
    d = data()[k]
    W = cc._W()
    b = np.digitize(d["x1"], xe) - 1
    ok = (b >= 0) & (b < NX)
    r, n = tp.bin_stats(d["lrd"], b, ok, d["env"], W, NX, d["grp"], 3)
    ra, _ = tp.bin_stats(d["lrd"], b, ok, d["env"], W, NX)
    return r, n, ra[:, 0]


def figure(path=None):
    import matplotlib.pyplot as plt
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    D = data()
    xe1, xe2, ye = edges()
    L10 = np.log(10.)
    maps = {(k, row): cells(D[k]["x1" if row == 0 else "x2"], D[k]["lru"], D[k]["lrd"], xe1 if row == 0 else xe2, ye)
            for k in AGENTS for row in (0, 1)}
    c = max(np.nanpercentile(np.abs(maps[(k, r)][0]), 98) for k in AGENTS[:3] for r in (0, 1)) / L10
    fig, axs = plt.subplots(3, 4, figsize=(17.5, 12.5), gridspec_kw=dict(hspace=.5, wspace=.28, left=.05, right=.93,
                                                                          top=.9, bottom=.09))
    for j, k in enumerate(AGENTS):
        for row, (xe, xn) in enumerate(((xe1, r"pair distance before, $d_H(p_{t-1}, q_{t-1})$ (nats)"),
                                        (xe2, r"distance from uniform before, $\sqrt{U(p_{t-1})U(q_{t-1})}$ (nats)"))):
            ax = axs[row, j]
            g, n = maps[(k, row)]
            im = ax.pcolormesh(np.log10(xe), ye / L10, g.T / L10, cmap="RdBu_r", vmin=-c, vmax=c, shading="flat")
            x = D[k]["x1" if row == 0 else "x2"]
            b = np.digitize(x, xe) - 1
            ok = (b >= 0) & (b < NX)
            nb_ = np.bincount(b[ok], minlength=NX)
            gm = np.bincount(b[ok], D[k]["lru"][ok], NX) / np.maximum(nb_, 1)
            ctr = np.log10(np.sqrt(xe[1:] * xe[:-1]))
            ax.plot(ctr[nb_ >= MIN_CELL], gm[nb_ >= MIN_CELL] / L10, "k-", lw=1.6)
            ax.axhline(0, color="k", lw=.6, ls=":")
            cc._ratio_y(ax, ye[0] / L10, ye[-1] / L10, ticks=(.5, .7, 1, 1.5, 2, 3))
            ax.set_xlim(np.log10(xe[0]), np.log10(xe[-1]))
            tk = [t for t in (.3, 1, 3, 10, 30, 100) if xe[0] <= t <= xe[-1]]
            ax.set_xticks(np.log10(tk))
            ax.set_xticklabels([f"{t:g}" for t in tk])
            ax.set_xlabel(xn, fontsize=10)
            if j == 0:
                ax.set_ylabel(r"uniform ratio, $\sqrt{R_U(p) R_U(q)}$" + "\n" + r"($<$1 flatter, $>$1 sharper)", fontsize=10)
            ax.set_title(f"{cc.NAME[k]}: pair ratio by " + ("pair distance" if row == 0 else "distance from uniform") +
                         "\n" + r"colour: $R_D = d_H(p_t, q_t)/d_H(p_{t-1}, q_{t-1})$", fontsize=11, loc="left")
        ax = axs[2, j]
        r, n, ra = lines(k, xe1)
        ctr = np.log10(np.sqrt(xe1[1:] * xe1[:-1]))
        v, lo, hi = (a / L10 for a in cc._iv(ra))
        ax.plot(ctr, v, color="0.6", lw=3, label="all cases", zorder=2)
        for gi in range(3):
            v, lo, hi = (a[gi] / L10 for a in cc._iv(r))
            m = n[gi] >= MIN_LINE
            ax.fill_between(ctr, np.where(m, lo, np.nan), np.where(m, hi, np.nan), color=GCOL[gi], alpha=.2, lw=0)
            share = n[gi].sum() / n.sum()
            ax.plot(ctr, np.where(m, v, np.nan), color=GCOL[gi], lw=2, label=f"{GROUPS[gi]} ({100 * share:.0f}%)")
        ax.axhline(0, color="k", lw=.6, ls=":")
        cc._ratio_y(ax, -.25, .6, ticks=(.6, .8, 1, 1.5, 2, 3))
        ax.set_xlim(np.log10(xe1[0]), np.log10(xe1[-1]))
        ax.set_xticks(np.log10([.3, 1, 3, 10, 30, 100]))
        ax.set_xticklabels(["0.3", "1", "3", "10", "30", "100"])
        ax.set_xlabel(r"pair distance before (nats)", fontsize=10)
        if j == 0:
            ax.set_ylabel(r"pair ratio $R_D$ (geometric mean)", fontsize=10)
        ax.legend(fontsize=8.5, frameon=False, loc="upper right")
        ax.set_title(f"{cc.NAME[k]}: pair ratio, beliefs flattening or sharpening", fontsize=11, loc="left")
    cax = fig.add_axes([.945, .42, .01, .42])
    cb = fig.colorbar(im, cax=cax)
    t = [x for x in (.5, .7, 1, 1.5, 2) if abs(np.log10(x)) <= c]
    cb.set_ticks(np.log10(t))
    cb.set_ticklabels([f"{x:g}" for x in t])
    cb.set_label("pair ratio $R_D$ (geometric mean per cell)")
    fig.suptitle("Pairs given the same observation: is the pair ratio flattening toward uniform, or divergence? (end of "
                 "training; colour = pair ratio, y = how much the two beliefs flatten or sharpen)", fontsize=13)
    n_in = {k: len(D[k]["lrd"]) for k in AGENTS}
    fig.text(.5, .01, "1,710,703 pair-steps x 2 variables per agent (50 test environments, 10,000 probes); cases inside "
             "the bins: " + ", ".join(f"{cc.NAME[k]} {n_in[k]:,}" for k in AGENTS) + ".\n" + rf"Cells with $\geq${MIN_CELL} "
             rf"cases; lines where a group holds $\geq${MIN_LINE}; bands: 5-95% over 1,000 environment resamples. "
             r"$U(p) = \mathrm{max}_r \log p_r - \mathrm{min}_r \log p_r$ = $d_H(p$, uniform$)$; "
             r"$R_U = U_t / U_{t-1}$.", ha="center", fontsize=9, linespacing=1.5)
    path = fig_path("td_tt_chaos_flatten.png") if path is None else path
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)


def numbers():
    D = data()
    xe1, _, _ = edges()
    W = cc._W()
    bands = ((.25, 1), (1, 3), (3, 10), (10, 120))
    for k in AGENTS:
        d = D[k]
        sh = np.bincount(d["grp"], minlength=3) / len(d["grp"])
        print(f"== {cc.NAME[k]}: smallest U before {d['umin']:.3g}; groups both flatten / one / both sharpen "
              f"{sh[0]:.3f} / {sh[1]:.3f} / {sh[2]:.3f}; uniform ratio x{np.exp(d['lru'].mean()):.3f}, pair ratio "
              f"x{np.exp(d['lrd'].mean()):.3f}")
        for lo_, hi_ in bands:
            s = (d["x1"] >= lo_) & (d["x1"] < hi_)
            parts = []
            for gi in range(3):
                m = s & (d["grp"] == gi)
                if m.sum() < MIN_LINE:
                    parts.append(f"{GROUPS[gi]} n={m.sum()}")
                    continue
                v, a, b = cc._iv(tp._ratio(d["lrd"], m, d["env"], W))
                parts.append(f"{GROUPS[gi]} x{np.exp(v):.3f} [x{np.exp(a):.3f}, x{np.exp(b):.3f}] ({m.sum() / s.sum():.2f})")
            vu = np.exp(d["lru"][s].mean())
            print(f"  pair distance {lo_}-{hi_} nats ({s.sum():,} cases; uniform ratio x{vu:.3f}): " + "; ".join(parts))
        # how far tempering explains the pair ratio: mean ln R_D vs mean ln R_U, and their per-case correlation
        print(f"  ln R_D = ln R_U + rest: mean {d['lrd'].mean():+.4f} = {d['lru'].mean():+.4f} + "
              f"{(d['lrd'] - d['lru']).mean():+.4f}; corr(ln R_D, ln R_U) {np.corrcoef(d['lrd'], d['lru'])[0, 1]:+.3f}"
              if k != "naive" else f"  naive: max |ln R_D| {np.abs(d['lrd']).max():.1e} (T1), ln R_U mean "
              f"{d['lru'].mean():+.4f}")


if __name__ == "__main__":
    what = sys.argv[1:] or ["figure", "numbers"]
    if "figure" in what:
        figure()
    if "numbers" in what:
        numbers()
