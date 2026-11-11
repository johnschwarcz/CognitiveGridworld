"""Which hard steps does the Trained RNN fix first: steps that are hard for naive because the OBSERVATION is entangled,
or because the STATE already carries FR? And why can a step's loss go below 0? (John 11-09: "Let's make a new figure
dedicated just to it"; "we only need panels a and b. But I still dont understand how the curve can go below 0 and then
rise? ... Is the rising good? Is the entanglement learning part of the rising?". The versions with one row per band and
with panels c-e (persistence, other bands, other runs) are kept in the session scratchpad.)
Data: td_tt_order.py (cells.npz, carried.npz). -> main/REPLICATION/figure_bin/fig_learn_order_diag.png

THE COMPARISON. Every step t >= 2 of the 10,000 probe episodes (both variables) has two numbers: the entanglement of
its observation, E_t, and the FR its belief carried before it, FR_t-1, each as a decile WITHIN its step (so every step
contributes equally to every decile). Naive's loss per step, FR_t - FR_t-1, grows with both. A band = the cells of the
10 x 10 decile grid where naive loses about the same per step (a diagonal of the grid, a). Inside a band more FR means
less entanglement, so a band has two ends: 'entangled step' (the half of its cells with the highest E - FR decile
difference; purple) and 'high-FR state' (the other half; green; the middle cell of an odd band is left out). The two
ends of a band have the same naive loss and the same mix of steps (0.03-0.05: naive .039 and .039 nats per step, mean
step 16.3 and 15.9; equal step weights move the crossings by <= 1%).
  b-g one row per network (Trained RNN b-d, ESN e-g; John 11-09: "show the trained and ESN in different rows, so we can
     put the KL at t, KL at t - 1 and delta KL as different columns"), at the two ends of the 0.03-0.05 band:
     b, e  the error after the step, KL(O_t || q_t) (log y)
     c, f  the error before it, the one the step INHERITS, KL(O_t-1 || q_t-1); dotted: the part of it made at earlier
           steps in the most entangled fifth
     d, g  the change at the step = after - before, how much further from optimal the step takes the belief (the
           per-step loss; symlog: linear within +-0.01; 0 = optimal)
     dashed: naive's FR_t, FR_t-1 and FR_t - FR_t-1 at the same steps (.039 at both ends in d, g). Bands: 5-95% over
     td_tt_gap's 1,000 resamples of the 50 environments.
RESULTS (11-09).
  The per-step loss (d) is a difference of two errors, so it is negative whenever a step removes more of the inherited
  error than its own update adds; any distance would do this (a symmetric KL does not change it).
  Split of d (td_tt_order_split.py; per item, then the mean): 'own' = the error the network's step-t update
  adds when applied to the OPTIMAL belief, KL(O_t || norm(O_t-1 q_t / q_t-1)) >= 0; 'merge' = what the CORRECT update
  does to the network's inherited error, KL(O_t || norm(q_t-1 O_t / O_t-1)) - KL(O_t-1 || q_t-1) <= 0 mostly. The two
  orders of the split disagree (the inherited error and the update interact), so the update's share is quoted as the
  mean of the two (Shapley). 0.03-0.05 band, Trained RNN:
    inherited error (c, f) 1,200: .62 (high-FR end) vs .36 (entangled end); 2,160: .23 vs .13; 7,800: .053 vs .035; end
                         .011 vs .008. Naive's: .81 vs .12. ESN: 2.1 vs 1.5 at 1,200, 1.2 vs .40 at the end.
    own update error     equal at the two ends at every checkpoint: 1,200 .054 / .054, 2,400 .040 / .040, 7,800 .015 /
                         .015, end .003 / .003 (naive's own: .030 high-FR end, .039 entangled end).
    share of the gap in b (entangled end - high-FR end) that is removal of inherited error: .88 at 1,680, .75 at 2,400,
                         ~.70 later; other bands: .010-.016 and .018-.028 ~.9-1.0, .06-.10 ~.65, .15-.30 ~.5.
    what the step does to the inherited error (dV - own): Trained RNN from 1,040 on negative at both ends (-.024 high-FR
                         end, -.014 entangled end; -.044 / -.023 at 1,680; ~12-20% of the inherited error per step);
                         ESN positive throughout (+.11 / +.06 at 1,040, +.036 / +.009 at the end); naive +.009 / .000.
    where the inherited error was made (V_t-1 = sum of earlier dV_s by the entanglement fifth of s, high-FR end): the
                         most entangled fifth of earlier steps made .32 of .62 at 1,200, .20 of .23 at 2,160, .14 of .13
                         at 3,400 (the low fifths remove some), .008 of .011 at the end. Naive's FR_t-1 .81: .38 from the
                         top fifth, .24 from the next.
    with the inherited error matched (V_t-1 < .02 nats), the high-FR end still costs less: .013 vs .019 at 2,400 (naive
                         .026 vs .022 on the same items), .0096 vs .0141 at 3,400.
  So the early, large difference between the ends in b is mostly the removal of inherited error, which is larger at
  high-FR states because they inherit more (made at earlier entangled steps); a smaller part (the matched comparison)
  is in the step itself. The earlier reading "FR is managed 1.6-2.5x before entanglement" (crossings of half of naive's:
  .010-.016: 1,113 vs 2,817; .018-.028: 1,369 vs 3,096; .03-.05: 1,242 vs 2,151; .06-.10: 1,084 vs 2,188; .15-.30: 878
  vs 1,366) is mostly this arithmetic, not an order of learning.
  Accumulated entanglement (John 11-09: "an accumulating sum of entanglement through time, or the max entanglement so
  far?"): within-step rank correlation with FR_t-1: sum of E_s, s < t: .43; max so far: .30; E_t itself: .26. The two
  ends do not differ in it (sum 3.8 entangled end vs 3.4 high-FR end; max .72 vs .64). Entanglement is persistent at the
  level of the EPISODE: 28% of its variance is between episodes, and a step's rank correlates .35 with any other step of
  the same episode (.35 / .34 / .33 / .32 at 1 / 5 / 10 / 20 steps apart); only the within-episode fluctuation has no
  memory (-.02 after z-scoring within each episode). FR_t-1: 67% between episodes, .94 / .82 / .72 / .51.
  Naming (John 11-09): 'optimal' = the exact joint posterior (called Bayes elsewhere); naive is also Bayesian.
Run: MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig_learn_order_diag.py"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.colors import LogNorm                                             # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from matplotlib.patches import Rectangle                                          # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import td_tt_gap as gap                                                           # noqa: E402
import td_tt_order as to                                                          # noqa: E402

TS, FL, FT = 11, 10, 9
BANDS = ((.010, .016), (.018, .028), (.03, .05), (.06, .10), (.15, .30))
MAIN = 2
CROSS, X0, XMAX = 880., 10., 4.6e4
END_COL = {"E": "#7b4fa6", "FR": "#2a9d6e"}
END_NAME = {"E": "entangled step", "FR": "high-FR state"}
RUNS = ("trained", "echo")
COL = {"trained": "#1f77b4", "echo": "#ff7f0e"}
NAME = {"trained": "Trained RNN", "echo": "ESN"}
LEVEL = .5


def load():
    return dict(np.load(os.path.join(to.OUT, "cells.npz")))


def ends(Z, band):
    """the band's cells split into its 'entangled step' and 'high-FR state' halves (cell index = E decile * 10 + FR)"""
    grid = Z["naive_cell"].sum(0) / Z["n_cell"].sum(0)
    cs = np.flatnonzero((grid >= band[0]) & (grid <= band[1]))
    o = np.argsort((cs % 10) - (cs // 10), kind="stable")
    h = len(cs) // 2
    return {"E": cs[o[:h]], "FR": cs[o[-h:]]}


def series(Z, run, cs, C, key="cell_{}", naive="naive_cell"):
    """the mean over the cells cs (K,) and its resamples (K, B); naive's (scalar, (B,)); key: which per-cell sums"""
    num, n = Z[key.format(run)][:, :, cs].sum(-1), Z["n_cell"][:, cs].sum(-1)              # (K, 50), (50,)
    nv = Z[naive][:, cs].sum(-1)
    return num.sum(1) / n.sum(), (num @ C.T) / (C @ n)[None], nv.sum() / n.sum(), (C @ nv) / (C @ n)


def half(Z, run, cs, C):
    """training episodes until the loss per step stays below LEVEL x naive's: full sample and 5-95% (RESULTS only)"""
    x = Z[f"x_{run}"]
    y, yb, nv, nvb = series(Z, run, cs, C)
    c = gap.crossing(x, y / nv, LEVEL, sign=-1.)
    cb = gap.crossing(x, (yb / nvb[None]).T, LEVEL, sign=-1.)
    ok = np.isfinite(cb).mean() > .95
    return c, (np.nanpercentile(cb, 5) if ok else np.nan), (np.nanpercentile(cb, 95) if ok else np.nan)


def xaxis(ax, label=True):
    ax.set_xscale("log")
    ax.set_xticks([X0, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.set_xlim(8, XMAX)
    ax.minorticks_off()
    ax.axvline(CROSS, color="0.4", lw=.8, ls=":")
    ax.spines[["top", "right"]].set_visible(False)
    if label:
        ax.set_xlabel("training episodes", fontsize=FL, labelpad=1)


def figure():
    """a: the decile grid and the band's two ends; one row per network (John 11-09: "show the trained and ESN in
    different rows, so we can put the KL at t, KL at t - 1 and delta KL as different columns")"""
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({"xtick.labelsize": FT, "ytick.labelsize": FT})
    Z = load()
    Zc = dict(np.load(os.path.join(to.OUT, "carried.npz")))
    Zc["n_cell"] = Z["n_cell"]
    for run in RUNS:                                                    # the error after the step: inherited + change
        Zc[f"Vt_cell_{run}"] = Zc[f"Vp_cell_{run}"] + Z[f"cell_{run}"]
    Zc["FRt_cell"] = Zc["FRp_cell"] + Z["naive_cell"]
    C = gap.boot_counts()
    E2 = ends(Z, BANDS[MAIN])
    W, H = 7.7, 6.95
    fig = plt.figure(figsize=(W, H))

    def add(l, t, w, h):
        return fig.add_axes([l / W, (H - t - h) / H, w / W, h / H])

    # a: the decile grid, naive's loss per step, the main band's two ends
    aa = add(.62, .62, 1.55, 1.55)
    grid = (Z["naive_cell"].sum(0) / Z["n_cell"].sum(0)).reshape(10, 10)
    aa.imshow(np.where(grid > 0, grid, np.nan), origin="lower", cmap="Greys", norm=LogNorm(3e-3, 3),
              extent=(.5, 10.5, .5, 10.5), interpolation="nearest", alpha=.55)
    for k, cs in E2.items():
        for c in cs:
            aa.add_patch(Rectangle((c % 10 + .5 + .06, c // 10 + .5 + .06), .88, .88, fill=False, ec=END_COL[k],
                                   lw=1.6, zorder=4))
    aa.set_xticks([1, 5, 10])
    aa.set_yticks([1, 5, 10])
    aa.set_xlabel("FR before the step (decile)", fontsize=FL, labelpad=1)
    aa.set_ylabel("entanglement of the step\n(decile)", fontsize=FL, labelpad=1)
    aa.set_title(r"$\mathbf{a}$   Steps naive finds equally hard", loc="left", fontsize=TS, pad=6)
    # key, right of a
    kx, ky = 3.0, .72
    fig.text(kx / W, (H - ky) / H, "Outlined in a: steps where naive loses 0.03-0.05 nats (grey: naive's loss per step)",
             fontsize=FT, va="top")
    keys = ((END_COL["E"], "-", 2, END_NAME["E"] + " (high entanglement, low FR)", END_COL["E"]),
            (END_COL["FR"], "-", 2, END_NAME["FR"] + " (high FR, low entanglement)", END_COL["FR"]),
            ("0.3", (0, (4, 2)), 1.3, "naive (the factorized posterior), the same steps", "k"),
            ("0.3", (0, (1, 1.3)), 1.4, "the part of the inherited error made at earlier\nsteps in the most entangled fifth",
             "k"))
    for i, (cl, ls, lw, txt, tc) in enumerate(keys):
        y = ky + .3 + i * .27
        fig.add_artist(Line2D([(kx + .05) / W, (kx + .45) / W], [(H - y - .07) / H] * 2, color=cl, lw=lw, ls=ls))
        fig.text((kx + .55) / W, (H - y) / H, txt, fontsize=FT, va="top", color=tc, linespacing=1.25)
    fig.text(kx / W, (H - ky - 1.62) / H, "Shading: 5-95% over the 50 test environments.  Dotted vertical: the\n"
             "Trained RNN passes naive (880).  0 in the right column: optimal.", fontsize=FT, va="top", color="0.3",
             linespacing=1.35)

    # b-g: one row per network; columns KL at t, KL at t - 1, the change
    COLS = (("Vt_cell_{}", "FRt_cell", "Error after the step", r"$\mathrm{KL}(O_t\Vert q_t)$"),
            ("Vp_cell_{}", "FRp_cell", "Error before the step", r"$\mathrm{KL}(O_{t-1}\Vert q_{t-1})$"),
            ("cell_{}", "naive_cell", "Change at the step", r"$\mathrm{KL}(O_t\Vert q_t)-\mathrm{KL}(O_{t-1}\Vert "
                                                            r"q_{t-1})$"))
    L0, PW_, PH_, GX_, GY_, T0 = .8, 1.86, 1.45, .44, .62, 3.05
    for i, run in enumerate(RUNS):
        top = T0 + i * (PH_ + GY_)
        for j, (key, nkey, name, form) in enumerate(COLS):
            ax = add(L0 + j * (PW_ + GX_), top, PW_, PH_)
            ZZ = Z if key == "cell_{}" else Zc
            xr = ZZ[f"x_{run}"]
            for k in ("E", "FR"):
                y, yb, nv, _ = series(ZZ, run, E2[k], C, key=key, naive=nkey)
                lo, hi = np.percentile(yb, [5, 95], 1)
                ax.fill_between(xr, lo, hi, color=END_COL[k], alpha=.25, lw=0)
                ax.plot(xr, y, color=END_COL[k], lw=1.6)
                ax.axhline(nv, color=END_COL[k] if j < 2 else "k", lw=.9, ls=(0, (4, 2)), zorder=1)
                if j == 1:
                    yt = series(Zc, run, E2[k], C, key="Vtop_cell_{}", naive="FRtop_cell")[0]
                    ax.plot(xr, np.where(yt > 0, yt, np.nan), color=END_COL[k], lw=1.3, ls=(0, (1, 1.3)))
            xaxis(ax, label=i == 1)
            if j < 2:
                ax.set_yscale("log")
                ax.set_ylim(4e-3, 30)
                ax.set_yticks([.01, .1, 1, 10])
                ax.set_yticklabels([".01", ".1", "1", "10"])
            else:
                ax.axhline(0, color="0.6", lw=.8)
                ax.set_yscale("symlog", linthresh=.01, linscale=.6)
                ax.set_ylim(-.02, 2)
                ax.set_yticks([-.01, 0, .01, .1, 1])
                ax.set_yticklabels(["-.01", "0", ".01", ".1", "1"])
            ax.minorticks_off()
            if j == 0:
                ax.set_ylabel("nats", fontsize=FL, labelpad=1)
                fig.text(.2 / W, (H - top - PH_ / 2) / H, NAME[run], rotation=90, ha="center", va="center",
                         fontsize=TS, color=COL[run])
            ax.text(-.2, 1.04, r"$\mathbf{%s}$" % "bcdefg"[3 * i + j], transform=ax.transAxes, fontsize=TS,
                    ha="left", va="bottom")
            if i == 0:
                ax.annotate(form, xy=(.5, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                            fontsize=FT, ha="center", va="baseline")
                ax.annotate(name, xy=(.5, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points",
                            fontsize=FL + .5, ha="center", va="baseline")
    out = fig_path("fig_learn_order_diag.png")
    fig.savefig(out, dpi=250)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    figure()
