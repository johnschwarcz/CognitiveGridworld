#!/usr/bin/env python3
"""
fr_theory_figure.py -- the theory figure of the Factorization Regret paper with EVERY proposed panel (fig_fr_theory_all).

John 11-11: "everything for the theory will eventually be in 1 figure.  Next to do is to create an updated single figure with 4 rows
and as many columns as you need for all of the proposed panels, then we will trim it down to what we want.  But all proposed panels
should already be high quality, clear and polished."  It gathers the panels of fig_fr_theory and fig_fr_averages (fr_theory.py) and
of fig_fr_threshold and fig_fr_candidates (fr_theory_additions.py); the panels of those two were reworked for readability (11-11).

    python fr_theory_figure.py      seconds.  Reads the caches of fr_theory.py and fr_theory_additions.py and never rebuilds one:
                                    a missing or stale cache stops it (run `fr_theory.py all` / `fr_theory_additions.py data` first).
    outputs: figure_bin/fig_fr_theory_all.png + .pdf; the same and 300-dpi crops per row in main/DATA/experimental/fr_theory.

11-11 evening (John: "I still dont understand j, s or t ... they have to go"; f + l -> mean +- SD per lambda; m-o -> two panels, very low
and very high q, theory and simulation for the mean and deviation, red line at 0: "how quickly the bulk converges to being above 0";
r: "show q as in i"; "make the figure a 3 x something"): 3 rows x 6.

TRIM by editing LAYOUT (rows of panel names; the letters follow the reading order).  Every panel is drawn at its print size, the
panel of the 9-in rotated figures (four per row reproduce their scale), so any subset can be laid out again without rescaling.
TITLES holds every panel title.
THE CONVENTION: lines and shaded bands = analytical, markers = simulation (mean +- 2 SEM; the quantiles of m-o: bootstrap 95%).
WHERE THE PANELS CAME FROM (letters of the 3 x 6 version, 11-11 evening)
  a-d   fig_fr_theory a-d; d at q = 0, q*/2, q*, 2q* (was 0, .001, .002, q*, .01): a multiple of q* has one blue in d, e, k
  e     fig_fr_candidates A (q*/2, q*, 1.1 q*, 2 q*): Corollary 2 at the critical step, exact
  f-h   fig_fr_theory e (mean) and h (spread) combined: mean +- s.d. of FR_t, one panel per lambda (theory at the simulated
        episodes' own joint realizations: t E[Delta], sqrt(t^2 Var(Delta) + t E[v]))
  i     fig_fr_theory f                       j   fig_fr_candidates D (tenths of q with equal counts)
  k     fig_fr_candidates B; the key gives the simulated q* (the Gridworld's own q, ~150 times larger, leaves the panel at once)
  l     fig_fr_theory g
  m, n  fig_fr_averages a, c (lowest and highest third of the trapped fraction) as mean +- s.d. of the step of an N-average:
        theory mu +- sigma / sqrt(N) (exact moments of one step), simulation over groups; above 0 once N > (sigma / mu)^2
  o     fig_fr_candidates C (N in roses: orange is delta)          p, q   fig_fr_averages d, e (q of each third, as in k)
  DROPPED (John: not understood): the Gridworld threshold q*_t vs q (its point is in k's key and j's axis), dips of Gridworld
  averages in the first 250 steps, toy dips in 3,000 steps (Corollary 2 stays in e).  Their caches (grid, pool, toy_sim, toy_fine)
  are still on disk.
"""
import os, pickle, sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib                                                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                            # noqa: E402
from matplotlib.colors import to_rgb                                                       # noqa: E402
from matplotlib.lines import Line2D                                                        # noqa: E402
from matplotlib.legend_handler import HandlerTuple                                         # noqa: E402
from matplotlib.patches import Patch                                                       # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator       # noqa: E402
import fr_theory as F                                                                      # noqa: E402
import fr_theory_additions as A                                                            # noqa: E402
import fr_theory_additions_plot as AP                                                      # noqa: E402

OUT = os.path.join(ROOT, "main", "DATA", "experimental", "fr_theory")
NAME = "fig_fr_theory_all"

LAYOUT = [
    ("Toy example (exact)", ["anatomy_k0", "anatomy_k1", "time_scale", "toy_mixture", "toy_critical"]),
    ("Cognitive Gridworld: the expected curve, and the spread of single episodes", ["meansd_1", "meansd_2", "meansd_3", "observables", "rate_q", "grid_mixture"]),
    ("Cognitive Gridworld: single episodes, averages over $N$ episodes, and their dips", ["step_law", "avg_lo", "avg_hi", "dips_step", "dips_T", "dips_q"]),
]
TITLES = dict(
    anatomy_k0=r"$k=0$: no trap", anatomy_k1=r"$k=1$: one trap", time_scale=r"$k=0$: time scale $\propto 1/\delta$",
    toy_mixture="Monotone vs. eventually monotone", toy_critical=r"Dip at the critical step $t^*$",
    meansd_1=r"Mean $\pm$ s.d. of $\mathrm{FR}_t$, $\lambda=1$", meansd_2=r"Mean $\pm$ s.d. of $\mathrm{FR}_t$, $\lambda=2$",
    meansd_3=r"Mean $\pm$ s.d. of $\mathrm{FR}_t$, $\lambda=3$",
    observables="Growth rate vs. observables", rate_q="Growth rate vs. trapped fraction", grid_mixture=r"Expected curve, varying $q$",
    step_law="One step of a trapped episode",
    avg_lo=lambda C: rf"Step of an average, $q={C['D']['fan3'][0]['q']:.2f}$", avg_hi=lambda C: rf"Step of an average, $q={C['D']['fan3'][-1]['q']:.2f}$",
    dips_step=r"Dip at step $t$ vs. its limit", dips_T=r"Dips within $T$ steps", dips_q=r"Dips within 30 steps, by $q$")

# ---------------------------------------------------------------- geometry (inches): the panel of the 9-in two-row theory figures
_W0, _H0 = F.FIGSIZE
PW, PH = (_W0 - 0.43 - 3 * 0.55 - 0.05) / 4, (_H0 - 2 * 0.35 - 0.31 - 0.39 - 0.05) / 2
LM, GX, RM = 0.45, 0.55, 0.15                    # left margin, gap between panels (holds the next panel's y labels), right margin
TOP, BOT1, BOT2, GAP = 0.35, 0.31, 0.39, 0.05    # above a row's axes (row header, panel title), below them (tick labels, axis label), between rows

# ---------------------------------------------------------------- colours (fr_theory.py's ramps; ROSES3 new: same OKLab lightness steps)
ROSES3 = ["#e977bb", "#aa3680", "#5c1143"]                                            # N, the number of episodes averaged (p)
QCOL = {0.0: F.BLUES5[0], 0.5: F.BLUES5[1], 1.0: F.INK, 1.1: F.BLUES5[2], 2.0: F.BLUES5[3], 27.0: F.BLUES5[4]}   # q as a multiple of q*
DASH = dict(ls=(0, (3, 1.6)), lw=F.LW * 1.1)                                          # the threshold q* itself
RED = "#d62839"                                                                        # zero, where an average stops being able to dip (m, n)
LAMS = F.CFG["LAMS_FULL"]
LAM_LABELS = [rf"$\lambda={LAMS[0]:g}$"] + [rf"${lam:g}$" for lam in LAMS[1:]]


# ============================================================================================================
# helpers
# ============================================================================================================
def theory(ax, x, y, color, lw=F.LW, **kw):
    """Analytical curve: always a line."""
    return ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=3, **kw)


def simulated(ax, x, y, yerr, color, filled=True, ms=F.MS, clip=False):
    """Simulation: always a marker (+- 2 SEM bar; the white ring keeps it legible on its own theory line)."""
    if yerr is not None: ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.7, capsize=0, zorder=4)
    return ax.plot(x, y, ls="none", marker="o", ms=ms, mfc=color if filled else "white", mec="white" if filled else color,
                   mew=0.5 if filled else 0.8, zorder=5, clip_on=clip)


def h_both(color, label): return Line2D([], [], color=color, lw=F.LW, marker="o", ms=F.MS, mfc=color, mec="white", mew=0.5, label=label)
def h_line(color, label, **kw): return Line2D([], [], color=color, lw=kw.pop("lw", F.LW), label=label, **kw)
def h_dot(color, label, filled=True):
    return Line2D([], [], ls="none", marker="o", ms=F.MS, mfc=color if filled else "white", mec="white" if filled else color, mew=0.5 if filled else 0.8, label=label)
def pastel(c): return tuple(0.30 * np.array(to_rgb(c)) + 0.70)                        # the anatomy bands, opaque: they meet without a seam
def decimals(ax, d, axis="y"): (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(lambda v, _: f"${v:.{d}f}$"))
def se_p(p, n): return np.sqrt(np.maximum(p * (1 - p), 1.0 / n) / n)                 # binomial SEM (floored at one count)


def qlab(r, first=False):
    """q as a multiple of the threshold q*."""
    s = {0.0: "0", 0.5: r"q^*\!/2", 1.0: r"q^*"}.get(r, f"{r:g}" + r"q^*")
    return f"$q={s}$" if first else f"${s}$"


def prob_axis(ax, top=1.0):
    """A probability on y: ticks 0, 0.5, 1 (headroom above 1 holds the key)."""
    ax.set_ylim(0, top); ax.set_yticks([0, 0.5, 1.0] if top > 1 else [0, 0.2, 0.4]); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.1f}$"))


def log_n_axis(ax, lo, hi, every=1):
    """Episodes averaged, on a log axis with powers of ten (every `every`-th one labelled)."""
    ax.set_xscale("log"); ax.set_xlim(lo, hi); e0, e1 = int(np.ceil(np.log10(lo))), int(np.floor(np.log10(hi)))
    ax.set_xticks([10.0 ** e for e in range(e0, e1 + 1)]); ax.set_xticklabels([rf"$10^{{{e}}}$" if (e - e0) % every == 0 else "" for e in range(e0, e1 + 1)])
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlabel(r"episodes averaged, $N$")


# ============================================================================================================
# data: the caches of fr_theory.py and fr_theory_additions.py, read only
# ============================================================================================================
class _ReadOnly(F.Store):
    """fr_theory.py's cache store, reading only: the figure must not start a build (cf. the --out mishap of 11-11)."""
    def get(self, name, signature, builder, fresh=False):
        path = self.path(name)
        if name == "row1" and os.path.exists(path):
            with open(path, "rb") as fh: sig, data = pickle.load(fh)
            if sig == signature: return data
        raise SystemExit(f"fr_theory cache '{name}' is missing or stale (the derived quantities no longer match the configuration): "
                         "run `python fr_theory.py figure --rederive` (or `all`) in main/DATA/experimental/fr_theory first")


def load():
    cfg = F.CFG; row1, summary, D = F.load_all(cfg, _ReadOnly(OUT, quick=False))
    sig_o = (F.OPT_VERSION, F.env_fingerprint(cfg), tuple(cfg["OPT_RS"]), tuple(cfg["OPT_DOS"]), cfg["OPT_NZ"], cfg["OPT_NEP"], cfg["OPT_T1"],
             cfg["OPT_T2"], tuple(cfg["OPT_TACC"]), cfg["LAM_MAIN"])                  # as in fr_theory.load_all (checked, not trusted)
    with open(os.path.join(OUT, "fr_theory_cache", "optimum.pkl"), "rb") as fh: sig, opt = pickle.load(fh)
    if sig != sig_o: raise SystemExit("fr_theory cache 'optimum' is stale: run fr_theory.py first")
    add = {}
    for name, keys in A.KEYS.items():
        path = os.path.join(A.OUT, f"{name}.pkl"); want = repr([(k, A.CFG[k]) for k in keys])
        if not os.path.exists(path): raise SystemExit(f"fr_theory_additions cache '{name}' is missing: run `python fr_theory_additions.py data`")
        with open(path, "rb") as fh: sig, data = pickle.load(fh)
        if sig != want: raise SystemExit(f"fr_theory_additions cache '{name}' is stale: run `python fr_theory_additions.py data`")
        add[name] = data
    return dict(cfg=cfg, row1=row1, summary=summary, D=D, opt_sets=opt[(10, 5)], A=add)


# ============================================================================================================
# panels: one function per panel, drawing into its axes (titles and letters are added by the layout)
# ============================================================================================================
# ---------------------------------------------------------------- row 1: toy example, exact (environment-independent)
def _anatomy(ax, C, k):
    r = C["row1"][k]; an, st = r["an"], r["st"]; t = np.arange(F.T_ANATOMY + 1); base = -an["Lam"]; mid = base + an["D"]
    ax.axhline(0, color=F.AXIS, lw=0.6, zorder=1)
    ax.fill_between(t, base, mid, color=pastel(F.ORANGE), lw=0, zorder=2)
    if st["k"]: ax.fill_between(t, mid, an["FR"], color=pastel(F.BLUE), lw=0, zorder=2)
    theory(ax, t, base, F.AQUA); theory(ax, t, an["FR"], F.INK, lw=F.LW * 1.15)
    ax.set_xlabel("step $t$"); ax.set_ylabel("nats"); ax.set_xlim(0, F.T_ANATOMY); ax.set_xticks([0, 100, 200, 300]); decimals(ax, 1)
    return an


def p_anatomy_k0(ax, C):
    """E[FR_t] = Joint term + distraction term (Proposition 1(b)) for the trap-free table; inset: E[FR_t] magnified."""
    an = _anatomy(ax, C, "a"); ax.set_ylim(-1.15, 0.95); ax.set_yticks([-1.0, -0.5, 0.0, 0.5])
    ax.legend(handles=[h_line(F.INK, r"$\mathbb{E}[\mathrm{FR}_t]$ = sum of"), h_line(F.AQUA, r"Joint term $-\Lambda_t$"),
                       Patch(color=pastel(F.ORANGE), lw=0, label=r"distraction $\mathbb{E}[\mathcal{D}_t]$")], loc="upper right")
    SQ = 0.50; ins = ax.inset_axes([1 - SQ / PW - 0.03, 0.045, SQ / PW, SQ / PH]); ins.grid(False)       # a square in the empty lower right
    theory(ins, np.arange(F.T_ANATOMY + 1), an["FR"], F.INK)
    ins.set_xlim(0, F.T_ANATOMY); ins.set_ylim(0, 0.065); ins.set_xticks([0, 100, 200, 300]); ins.set_xticklabels([]); ins.set_yticks([0, 0.05])
    ins.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.2f}".lstrip("0")))
    ins.tick_params(length=1.6, pad=1.0, labelsize=F.FS["tick"])
    for sp in ins.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5)


def p_anatomy_k1(ax, C):
    """As (a) for the table with one trapped joint realization: the trap term grows linearly."""
    _anatomy(ax, C, "b"); ax.set_ylim(-1.25, 2.25); ax.set_yticks([-1, 0, 1, 2])
    ax.legend(handles=[Patch(color=pastel(F.BLUE), lw=0, label=r"trap term $\mathbb{E}[\mathcal{T}_t]$")], loc="upper left")


def p_time_scale(ax, C):
    """No trap: E[FR_t] returns to 0 on a time scale set by the pull delta of the weakest distractor."""
    T2 = F.T_DISTRACTORS; t = np.arange(T2 + 1)
    for e, col in zip(C["row1"]["d"], F.ORANGES3): theory(ax, t, e["efr"][:T2 + 1], col)
    ax.legend(handles=[h_line(col, f"{e['st']['delta']:.2f}") for e, col in zip(C["row1"]["d"], F.ORANGES3)], title=r"Pull $\delta$ of the weakest distractor",
              loc="upper right", ncol=3, handlelength=1.1, columnspacing=0.7, alignment="left")
    ax.set_ylim(0, 0.1); ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08]); decimals(ax, 2); ax.set_xlim(0, T2); ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\mathbb{E}[\mathrm{FR}_t]$ (nats)")


def _toy_qstar(C):
    E0, E1 = C["row1"]["a"]["efr"], C["row1"]["b"]["efr"]; share = C["row1"]["b"]["st"]["k"] / C["row1"]["b"]["ell"].size
    w_star, t_star = F.mixture_threshold(E0, E1); q_star = w_star * share
    assert np.isclose(q_star, C["A"]["toy_theory"]["q_star"]) and t_star == C["A"]["toy_theory"]["t_star"], "the two scripts disagree on q*"
    return E0, E1, share, q_star


def p_toy_mixture(ax, C):
    """Corollary 1: a fraction q of the episodes from table (b), the rest from (a); the expected curve dips iff q < q*."""
    E0, E1, share, q_star = _toy_qstar(C); t = np.arange(len(E0)); hnd = []
    labels = {0.0: r"$q=0$: returns to 0", 0.5: r"$q^*\!/2$: dips, then rises", 1.0: r"$q^*$", 2.0: r"$2q^*$: rises from $t=0$"}
    for r in (0.0, 0.5, 1.0, 2.0):
        w = r * q_star / share; kw = DASH if r == 1.0 else {}
        theory(ax, t[1:], ((1 - w) * E0 + w * E1)[1:], QCOL[r], **kw); hnd.append(h_line(QCOL[r], labels[r], **dict(kw)))
    ax.set_xscale("log"); ax.set_xlim(1, F.T_MIXTURE); ax.set_ylim(-0.006, 0.28); ax.set_yticks([0, 0.1, 0.2]); decimals(ax, 1)
    ax.xaxis.set_major_locator(LogLocator(numticks=5)); ax.xaxis.set_minor_formatter(NullFormatter()); ax.axhline(0, color=F.AXIS, lw=0.6, zorder=1)
    ax.legend(handles=hnd, loc="upper left", alignment="left", handlelength=1.6, borderaxespad=0.15)
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\mathbb{E}[\mathrm{FR}_t]$ (nats)")


def p_toy_critical(ax, C):
    """Corollary 2 at the critical step t* (where q*_t is largest): P(an average of N episodes decreases there), exact law of the
    increment (FFT up to 10^6 episodes, saddlepoint beyond): -> 1 below q*, 1/2 at q*, -> 0 above, slower the closer q is to q*."""
    th = C["A"]["toy_theory"]; hnd = []
    for k, r in enumerate((0.5, 1.0, 1.1, 2.0)):
        kw = DASH if r == 1.0 else {}
        theory(ax, th["Ns"], th["p_star"][th["ratios"].index(r)], QCOL[r], **kw); hnd.append(h_line(QCOL[r], qlab(r, k == 0), **dict(kw)))
    log_n_axis(ax, 1e3, 1e8, every=2); prob_axis(ax, 1.36)
    ax.legend(handles=hnd, loc="upper left", ncol=2, columnspacing=0.9, handlelength=1.5, borderaxespad=0.1)
    ax.set_ylabel(r"P(dip at step $t^*$)")


# ---------------------------------------------------------------- row 2: Cognitive Gridworld, the expected curve
def _meansd(ax, C, lam):
    """Propositions 1(c) and 2(c): mean and s.d. of FR_t across episodes.  Theory (tables alone, at the simulated episodes' own joint
    realizations): t E[Delta] and sqrt(t^2 Var(Delta) + t E[v]) (the line omits the bounded remainder of eq. fr_linear, a fraction of a
    nat).  Simulation: mean and mean +- s.d. over 2,000 sets x 50 episodes (2 SEM by resampling sets)."""
    SP = C["D"]["spread"][lam]; col = F.VIOLETS3[LAMS.index(lam)]; t = np.r_[0, SP["t"]]; m = np.r_[0, SP["mean_th"]]; s = np.r_[0, SP["th"]]
    ax.axhline(0, color=F.AXIS, lw=0.6, zorder=1)
    ax.fill_between(t, m - s, m + s, color=col, alpha=0.15, lw=0, zorder=2)
    for e in (m - s, m + s): theory(ax, t, e, col, lw=F.LW * 0.5)
    theory(ax, t, m, col)
    simulated(ax, SP["tm"], SP["mean_m"], 2 * SP["mean_se"], col)
    for v, e in (("lo_m", "lo_se"), ("hi_m", "hi_se")): simulated(ax, SP["tm"], SP[v], 2 * SP[e], col)
    top, bot = max((m + s).max(), SP["hi_m"].max()), min((m - s).min(), SP["lo_m"].min())
    ax.set_xlim(0, C["cfg"]["T"]); ax.set_xticks(np.arange(0, C["cfg"]["T"] + 1, 100)); ax.set_ylim(bot - 0.08 * (top - bot), top + 0.42 * (top - bot))
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\mathrm{FR}_t$ (nats)")
    if lam == LAMS[0]:
        ax.legend(handles=[h_both(col, "mean"), (Patch(facecolor=col, alpha=0.15, edgecolor=col, lw=0.5), h_dot(col, ""))], labels=["mean", r"mean $\pm$ s.d."],
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, loc="upper left", handlelength=2.2, borderaxespad=0.15)


def p_meansd_1(ax, C): _meansd(ax, C, LAMS[0])
def p_meansd_2(ax, C): _meansd(ax, C, LAMS[1])
def p_meansd_3(ax, C): _meansd(ax, C, LAMS[2])


def p_observables(ax, C):
    """E[Delta] (tables) against late simulated slopes, by number of observables d_o, for R = 5, 10, 20 realizations per variable."""
    xs = np.array(C["cfg"]["OPT_DOS"]); OP = C["D"]["opt"]
    for R_, col in zip(C["cfg"]["OPT_RS"], F.GRAYS3):
        theory(ax, xs, [OP[(R_, d)]["ed"][0] for d in xs], col)
        simulated(ax, xs, [OP[(R_, d)]["slope"][0] for d in xs], [2 * OP[(R_, d)]["slope"][1] for d in xs], col)
    ax.axvline(5, color=F.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)                    # the Gridworld's five observables
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 30]); ax.set_xticklabels(["1", "2", "5", "10", "30"]); ax.set_xlim(0.9, 33)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1.32 * max(max(OP[k]["ed"][0], OP[k]["slope"][0]) for k in OP))
    ax.legend(handles=[h_both(col, rf"$R={R_}$") for R_, col in zip(C["cfg"]["OPT_RS"], F.GRAYS3)], loc="upper center", ncol=3, columnspacing=0.8,
              handlelength=1.3, borderaxespad=0.1)
    ax.set_xlabel(r"observables $d_o$"); ax.set_ylabel(r"$\mathbb{E}[\Delta]$ (nats / step)")


def p_rate_q(ax, C):
    """Growth rate against the trapped fraction q of the interaction set (Proposition 1(c): E[Delta] = q Delta-bar_trap), the
    10,000 sets of (g) at R = 10, d_o = 5, lambda = 2, in tenths of q (equal counts): E[Delta] (tables) and late simulated slopes."""
    v = C["opt_sets"]; bins = np.array_split(np.argsort(v["q"]), 10); col = F.VIOLETS3[LAMS.index(C["cfg"]["LAM_MAIN"])]
    mq = np.array([v["q"][b].mean() for b in bins]); th = np.array([v["ed"][b].mean() for b in bins])
    sm = np.array([v["slope"][b].mean() for b in bins]); se = np.array([v["slope"][b].std(ddof=1) / np.sqrt(len(b)) for b in bins])
    theory(ax, mq, th, col); simulated(ax, mq, sm, 2 * se, col)
    ax.set_xlim(0, 0.8); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8]); decimals(ax, 1, "x"); ax.set_ylim(0, 1.15 * max(th.max(), (sm + 2 * se).max()))
    ax.set_xlabel(r"trapped fraction $q$ of the set"); ax.set_ylabel(r"$\mathbb{E}[\Delta]$ (nats / step)")


def p_grid_mixture(ax, C):
    """The Gridworld version of (d): (1 - q) U_t + q W_t from its simulated untrapped (U) and trapped (W) episodes, lambda = 2,
    for q relative to the simulated threshold q* (windows of 3 grid steps) and the Gridworld's own q."""
    G = C["A"]["grid"][C["cfg"]["LAM_MAIN"]]; Cm = A.components(G); TS = G["TS"]; qs = AP.window_threshold(G, AP.WINDOW)["q_star"]; hnd = []
    for k, (r, q) in enumerate([(0.0, 0.0), (0.5, 0.5 * qs), (1.0, qs), (2.0, 2 * qs)]):
        col = QCOL[r]; m = (1 - q) * Cm["U"] + q * Cm["W"]; se = np.hypot((1 - q) * Cm["U_se"], q * Cm["W_se"])
        simulated(ax, TS, m, 2 * se, col, filled=r != 1.0, ms=F.MS * 0.62, clip=True)
        hnd.append(h_dot(col, qlab(r, k == 0) if r != 1.0 else rf"$q^*\approx{qs:.1g}$", filled=r != 1.0))
    ax.set_xscale("log"); ax.set_xlim(1, A.CFG["GRID_TMAX"]); ax.xaxis.set_major_locator(LogLocator(numticks=5)); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.5, 1.0]); decimals(ax, 1)
    ax.legend(handles=[hnd[0], hnd[2], hnd[1], hnd[3]], loc="upper left", ncol=2, columnspacing=0.8, handletextpad=0.25, borderaxespad=0.1)   # row-wise order
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\mathbb{E}[\mathrm{FR}_t]$ (nats)")


# ---------------------------------------------------------------- row 3: one step of single episodes and of averages
def p_step_law(ax, C):
    """Proposition 2(a): the law of one step X of a trapped episode (tables alone) against simulated steps after step 10,000."""
    ST = C["D"]["pop_step"]; ed = ST["edges"]; mid = 0.5 * (ed[1:] + ed[:-1]); col = F.VIOLETS3[LAMS.index(C["cfg"]["LAM_MAIN"])]
    ax.stairs(ST["dens"], ed, fill=True, color=col, alpha=0.14, lw=0, zorder=2); ax.stairs(ST["dens"], ed, color=col, lw=F.LW, zorder=3)
    simulated(ax, mid, ST["sim"], ST["sim_2se"], col, ms=F.MS * 0.85)
    ax.axvline(0, color=F.AXIS, lw=0.6, zorder=1); ax.set_xlim(-ST["xmax"], ST["xmax"]); ax.set_ylim(0, 1.25 * max(ST["dens"].max(), ST["sim"].max()))
    ax.text(0.03, 0.97, f"decrease at a step:\n{100 * ST['p_pred']:.0f}% theory\n{100 * ST['p_sim']:.0f}% simulation", transform=ax.transAxes,
            va="top", ha="left", linespacing=1.15, fontsize=F.FS["key"], color=F.INK)
    ax.set_xlabel(r"$\mathrm{FR}_{t+1}-\mathrm{FR}_t$ (nats)"); ax.set_ylabel("density")


def _avg(ax, C, k):
    """Proposition 2(b): the one-step change X-bar_N of an average of N episodes (from step 10,000), interaction sets in the k-th third
    of the trapped fraction (q = that third's share of trapped joint realizations).  Theory (tables alone): mean +- s.d. of X-bar_N =
    mu +- sigma / sqrt(N), mu and sigma the exact moments of one step X; the bulk is above zero once N > (sigma / mu)^2.  Simulation:
    mean +- s.d. over independent groups of N simulated episodes (2 SEM by bootstrap over groups)."""
    FN = C["D"]["fan3"][k]; col = F.BLUES3[k]; mu, sg = FN["mean"], FN["sigma"]; Nf = np.logspace(0, np.log10(1300), 300)
    ax.axhline(0, color=RED, lw=1.0, zorder=2.6)
    ax.fill_between(Nf, mu - sg / np.sqrt(Nf), mu + sg / np.sqrt(Nf), color=col, alpha=0.16, lw=0, zorder=2)
    for e in (mu - sg / np.sqrt(Nf), mu + sg / np.sqrt(Nf)): theory(ax, Nf, e, col, lw=F.LW * 0.8)
    for j in (0, 1): simulated(ax, FN["Ns"], FN["msd"][:, j], 2 * FN["msd_se"][:, j], col, ms=F.MS * 0.9)
    log_n_axis(ax, 0.8, 1300); ax.set_ylim(-0.5, 0.72); ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6]); decimals(ax, 1)
    ax.set_ylabel(r"$\widehat{\mathrm{FR}}_{t+1}-\widehat{\mathrm{FR}}_t$ (nats)")
    if k == 0:
        ax.legend(handles=[Patch(facecolor=col, alpha=0.16, edgecolor=col, lw=0.8), h_dot(col, "")], labels=[r"theory: mean $\pm$ s.d.", r"simulation: mean $\pm$ s.d."],
                  loc="lower right", handlelength=1.4, borderaxespad=0.2)


def p_avg_lo(ax, C): _avg(ax, C, 0)
def p_avg_hi(ax, C): _avg(ax, C, len(C["D"]["fan3"]) - 1)


# ---------------------------------------------------------------- dips of averaged curves
def p_dips_step(ax, C):
    """P(an average of N episodes decreases at step t) against t, simulated (lambda = 2), against its limiting law (tables alone):
    reached by about step 15; before that dips are rarer."""
    S = C["A"]["steps"]
    for i, (N, col) in enumerate(zip(S["Ns"], ROSES3)):
        theory(ax, [0.5, 2e4], [0.5 * (S["law_lo"][i] + S["law_hi"][i])] * 2, col); simulated(ax, S["TS"], S["sim"][i], 2 * S["se"][i], col, ms=F.MS * 0.85)
    ax.set_xscale("log"); ax.set_xlim(0.8, 1.3e4); ax.xaxis.set_major_locator(LogLocator(numticks=5)); ax.xaxis.set_minor_formatter(NullFormatter())
    prob_axis(ax, 0.5)
    ax.legend(handles=[h_both(col, (r"$N=" if i == 0 else "$") + f"{N}$") for i, (N, col) in enumerate(zip(S["Ns"], ROSES3))], loc="upper left", ncol=3,
              columnspacing=0.8, handlelength=1.3, borderaxespad=0.1)
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"P(dip at step $t$)")


def p_dips_T(ax, C):
    """The probability that an average of N episodes decreases within the next T steps, from step 10,000, all interaction sets
    (lambda = 2): 1 - E_rho[(1 - p_rho)^T] (theory) against groups of simulated episodes."""
    P_ = C["D"]["pop_h"]
    for j, (T_, col) in enumerate(zip(P_["Ts"], F.GREENS4)):
        theory(ax, P_["Nth"], P_["theory"][:, j], col); simulated(ax, P_["Ns"], P_["sim"][:, j], 2 * P_["se"][:, j], col)
    log_n_axis(ax, 8, 2600); prob_axis(ax, 1.24)
    ax.legend(handles=[h_both(col, (r"$T=" if j == 0 else "$") + f"{T_:,}$".replace(",", "{,}")) for j, (T_, col) in enumerate(zip(P_["Ts"], F.GREENS4))],
              loc="upper right", ncol=4, columnspacing=0.55, handlelength=1.0, handletextpad=0.3, borderaxespad=0.1)
    ax.set_ylabel(r"P(dip within $T$ steps)")


def p_dips_q(ax, C):
    """As (q) for T = 30, by third of the trapped fraction of the interaction set (labelled by q, the third's share of trapped joint realizations)."""
    k30 = list(C["cfg"]["TH_TS"]).index(max(C["cfg"]["TH_TS"])); hnd = []
    for i, (MO, (lo, hi), col) in enumerate(zip(C["D"]["mono3"], C["D"]["third_bins"], F.BLUES3)):
        theory(ax, MO["Nth"], MO["theory"][:, k30], col); simulated(ax, MO["Ns"], MO["sim"][:, k30], 2 * MO["se"][:, k30], col)
        hnd.append(h_both(col, (r"$q=" if i == 0 else "$") + f"{MO['q']:.2f}$"))                                   # q of the third, as in (i)
    log_n_axis(ax, 8, 2600); prob_axis(ax, 1.24)
    ax.legend(handles=hnd, loc="upper right", ncol=3, columnspacing=0.55, handlelength=1.0, handletextpad=0.3, borderaxespad=0.1)
    ax.set_ylabel(r"P(dip within $30$ steps)")


PANELS = {k[2:]: v for k, v in globals().items() if k.startswith("p_") and callable(v)}


# ============================================================================================================
# layout, checks, output
# ============================================================================================================
def headroom(fig, ax, artists, pts, pad=0.03):
    """Raise (never lower) the upper y-limit just enough for the data to pass `pad` (axes fraction) below each artist above them."""
    R = fig.canvas.get_renderer(); inv = ax.transAxes.inverted()
    P = inv.transform(ax.transData.transform(np.vstack(pts))); f = 1.0
    for a in artists:
        b = a.get_window_extent(R).transformed(inv); m = (P[:, 0] > b.x0 - 0.015) & (P[:, 0] < b.x1 + 0.015)
        if m.any() and b.y0 - pad > 0.1: f = max(f, P[m, 1].max() / (b.y0 - pad))
    ax.set_ylim(0, ax.get_ylim()[1] * f)


def text_checks(fig, rows):
    """Print text that overlaps other text or leaves the figure, and panel titles that come too close (as fr_theory.finish_figure)."""
    from matplotlib.text import Text
    fig.draw_without_rendering(); R = fig.canvas.get_renderer(); boxes = []; pooled = set(); drawn = []
    for ax in fig.findobj(lambda o: hasattr(o, "xaxis") and hasattr(o, "yaxis")):
        for axis in (ax.xaxis, ax.yaxis):
            pooled |= {id(lb) for tk in axis.majorTicks + axis.minorTicks for lb in (tk.label1, tk.label2)}
            lo_, hi_ = sorted(axis.get_view_interval())
            for locs, ticks in ((axis.get_majorticklocs(), axis.get_major_ticks), (axis.get_minorticklocs(), axis.get_minor_ticks)):
                drawn += [tk.label1 for tk, loc in zip(ticks(len(locs)), locs) if lo_ - 1e-9 <= loc <= hi_ + 1e-9 and tk.label1.get_visible()]
    for t in [t for t in fig.findobj(Text) if id(t) not in pooled] + drawn:
        if t.get_visible() and t.get_text().strip() and t.figure is not None:
            bb = t.get_window_extent(R)
            if bb.width > 0: boxes.append((t.get_text(), bb))
    n = 0
    for s, bb in boxes:
        if bb.x0 < fig.bbox.x0 - 0.5 or bb.y0 < fig.bbox.y0 - 0.5 or bb.x1 > fig.bbox.x1 + 0.5 or bb.y1 > fig.bbox.y1 + 0.5: print(f"   check: text outside the figure: {s!r}"); n += 1
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (s1, b1), (s2, b2) = boxes[i], boxes[j]
            if b1.overlaps(b2) and min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > 1 and min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > 1:
                print(f"   check: texts overlap: {s1!r} / {s2!r}"); n += 1
    for row in rows:
        for l, r in zip(row[:-1], row[1:]):
            gap = (r.title.get_window_extent(R).x0 - l.title.get_window_extent(R).x1) / fig.dpi
            if gap < 0.06: print(f"   check: only {gap:.2f} in between the titles {l.get_title(loc='left')!r} and {r.get_title(loc='left')!r}"); n += 1
    print(f"   text checks: {n} problem(s)")


def make(C, layout=LAYOUT, name=NAME):
    plt.rcParams.update(F.RC)
    ncol = max(len(names) for _, names in layout); nrow = len(layout)
    W = LM + ncol * PW + (ncol - 1) * GX + RM; H = nrow * (TOP + PH) + (nrow - 1) * (BOT1 + GAP) + BOT2
    fig = plt.figure(figsize=(W, H)); rows = []; extra = []; letters = iter("abcdefghijklmnopqrstuvwxyz")
    for i, (header, names) in enumerate(layout):
        y0 = H - (i + 1) * (TOP + PH) - i * (BOT1 + GAP); row = []
        fig.text(0.02 / W, (y0 + PH + TOP - 0.135) / H, header, fontfamily="cmb10", fontsize=F.FS["header"], va="baseline")
        for j, pname in enumerate(names):
            ax = fig.add_axes([(LM + j * (PW + GX)) / W, y0 / H, PW / W, PH / H]); out = PANELS[pname](ax, C) or {}
            tt = TITLES[pname]; ax.set_title(f"({next(letters)}) {tt(C) if callable(tt) else tt}", loc="left"); ax.tick_params(which="both", width=0.6); row.append(ax)
            if "headroom" in out: extra.append((ax, *out["headroom"]))
        rows.append(row)
    key = [Line2D([], [], color=F.INK, lw=F.LW, label="Theory"), h_dot(F.INK, r"Simulation (mean $\pm$ 2 SEM)")]
    if len(layout[0][1]) < ncol:                                                           # an empty slot in row 1 (which has no markers): the key there
        j = len(layout[0][1]); fig.legend(handles=key, loc="center left", bbox_to_anchor=((LM + j * (PW + GX)) / W, (H - TOP - 0.5 * PH) / H), ncol=1,
                                          fontsize=F.FS["label"], handlelength=1.8, labelspacing=0.8, borderpad=0.1, frameon=False)
    else:                                                                                  # else on the header line of row 2
        yk = H - 2 * (TOP + PH) - (BOT1 + GAP) + PH + TOP + 0.02
        fig.legend(handles=key, loc="upper right", bbox_to_anchor=(1 - RM / W, yk / H), ncol=2, fontsize=F.FS["label"], handlelength=1.8, columnspacing=1.4,
                   borderpad=0.1, frameon=False)
    for _ in range(2):                                                                     # second pass: tick labels may have changed the layout
        fig.draw_without_rendering()
        for ax, arts, pts in extra: headroom(fig, ax, arts, pts)
    text_checks(fig, rows)
    stem = os.path.join(OUT, name); fig.savefig(stem + ".pdf"); fig.savefig(stem + ".png", dpi=150)
    try:
        from main.utils import fig_path
        fig.savefig(fig_path(f"{name}.png"), dpi=150); fig.savefig(fig_path(f"{name}.pdf")); print("   also saved in figure_bin")
    except Exception as err:
        print(f"   figure_bin copy skipped ({err})")
    from PIL import Image                                                                  # 300-dpi crops per row, for checking by eye
    insp = os.path.join(OUT, "fr_theory_inspect"); os.makedirs(insp, exist_ok=True); big = os.path.join(insp, f"{name}_300dpi.png"); fig.savefig(big, dpi=300)
    im = Image.open(big); Wp, Hp = im.size
    for i in range(nrow):
        top = int((i * (TOP + PH + BOT1 + GAP) - 0.02) / H * Hp); bot = int(min(H, (i + 1) * (TOP + PH) + i * (BOT1 + GAP) + BOT2 + 0.02) / H * Hp)
        for part, (x0, x1) in (("left", (0, 0.55)), ("right", (0.45, 1.0))):
            im.crop((int(x0 * Wp), max(top, 0), int(x1 * Wp), min(bot, Hp))).save(os.path.join(insp, f"{name}_row{i + 1}_{part}.png"))
    plt.close(fig); print(f"saved {stem}.pdf/.png ({W:.2f} x {H:.2f} in) and crops in {insp}/")


if __name__ == "__main__":
    make(load())
