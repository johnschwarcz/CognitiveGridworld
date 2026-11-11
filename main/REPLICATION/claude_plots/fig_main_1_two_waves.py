"""Main Figure 1: learning beyond factorized inference. Memory is what passes naive; the state compresses and re-expands
first; the answers and the splitting of near beliefs are half-way to Bayes before holding a split and coupling.
-> main/REPLICATION/figure_bin/publication/fig_main_1.png
  a accuracy at T intact and with the memory cut to 1 step (td_questions: ck_*.npz)
  b top: naive's answer on the probes where naive errs and Bayes is right. It peaks at 640 training episodes (.28,
    accuracy .37), before passing naive, at the step-wise rule's level (.318), so the peak does not single out a naive
    stage (td_questions finding 4). bottom: r(FR_T, correct_T), each variable its own (td_summary_data.fr_acc; FR =
    KL(Bayes || naive), the paper's forward form, since 11-10: Trained RNN -.13 at 560 then back to Bayes's own -.05,
    ESN falling to -.27, naive -.32; with the symmetric KL until then: -.15, -.02, -.33, -.43)
  c PR of the state u_t per step, averaged over steps (ck_*.npz); dots: first checkpoint past naive
  d-f gap fraction per measure and crossings (td_tt_timeline; every definition there)
The family order is an ordered sequence, not two clusters (td_tt_timeline): the additive c50 test is 1,000 of 1,000 paired
resamples, the families interleave at f = .25, and on a log scale the separation shrinks (paired test .47-.85).
Run: MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig_main_1_two_waves.py"""
from fig_main_common import *                                                     # noqa: F403
import matplotlib.patheffects as pe                                               # noqa: E402
import matplotlib.transforms as mtr                                               # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from matplotlib.patches import Patch                                              # noqa: E402
import td_tt_timeline as tl                                                       # noqa: E402
import td_summary_data as sd                                                      # noqa: E402
from td_questions import ck, NAIVE, BAYES, STEPWISE, STEPWISE_COPY, crossing      # noqa: E402

PR_RUNS = ("trained_N50", "trained_N100", "trained_N200", "trained")
PR_NAME = {"trained_N50": "N = 50", "trained_N100": "N = 100", "trained_N200": "N = 200", "trained": "N = 1000"}


def a_memory(ax, D):
    for r in ("trained", "echo"):
        ax.plot(xs(D[r]["x"]), D[r]["trunc"][:, 3], color=COL[r], lw=2)
        ax.plot(xs(D[r]["x"]), D[r]["trunc"][:, 0], color=COL[r], lw=1.6, ls=":")
    ref(ax, BAYES, "Bayes")
    ref(ax, STEPWISE, "step-wise Bayes", color="0.35")
    ref(ax, NAIVE, "naive")
    xaxis(ax)
    ax.set_ylim(.07, .84)
    ax.set_ylabel("goal accuracy at $T$", fontsize=FL)
    title(ax, "a", "Passing naive takes memory", "accuracy at $T$, intact against memory cut to 1 step")
    ax.legend(handles=[Line2D([], [], color=COL["trained"], lw=2, label="Trained RNN"),
                       Line2D([], [], color=COL["echo"], lw=2, label="ESN"),
                       Line2D([], [], color="0.35", lw=2, label="intact"),
                       Line2D([], [], color="0.35", lw=1.6, ls=":", label="cut to 1 step")],
              fontsize=FT, frameon=False, loc="lower right", ncol=2, columnspacing=1.2)


def b_copy(ax, D):
    for r in ("trained", "echo"):
        ax.plot(xs(D[r]["x"]), D[r]["copy_naive"], color=COL[r], lw=2)
    ref(ax, STEPWISE_COPY, "step-wise Bayes", color="0.35", ls="-.", dy=.01)
    xaxis(ax, label=False)
    ax.set_ylabel("answers as naive", fontsize=FL)
    title(ax, "b", "Naive's wrong answers; regret against correctness",
          r"top: share where naive errs, Bayes right;  bottom: $r(\mathrm{FR}_T, \mathrm{correct}_T)$")


def b_fr(ax):
    for r in ("trained", "echo"):
        z = sd.fr_acc(r)
        ax.plot(z["x"], z["r"], color=COL[r], lw=2)
    ref(ax, float(z["r_bayes"]), "Bayes", dy=-.075)
    ref(ax, float(z["r_naive"]), "naive", dy=.008)
    xaxis(ax)
    ax.set_ylim(-.47, .03)
    ax.set_ylabel("Pearson $r$", fontsize=FL)


def c_pr(ax, D):
    for r in PR_RUNS + ("echo",):
        d = D[r]
        c = COL[r] if r == "echo" else WIDTH_COL[r]
        ax.plot(xs(d["x"]), d["pr_u"], color=c, lw=2)
        xc = crossing(d["x"], d["acc"], NAIVE)
        if xc:
            ax.plot(xc, d["pr_u"][list(d["x"]).index(xc)], "o", color=c, mec="k", mew=.8, ms=6, zorder=5)
    ax.set_yscale("log")
    xaxis(ax)
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    ax.set_ylabel("PR of the state $u_t$", fontsize=FL)
    title(ax, "c", "The state compresses, then re-expands before passing naive",
          r"PR $=(\Sigma_k\lambda_k)^2/\Sigma_k\lambda_k^2$ of $u_t$, per step, averaged over steps")
    ax.legend(handles=[Line2D([], [], color=WIDTH_COL[r], lw=2, label=PR_NAME[r]) for r in PR_RUNS]
              + [Line2D([], [], color=COL["echo"], lw=2, label="ESN"),
                 Line2D([], [], color="0.35", lw=0, marker="o", mec="k", ms=6, label="passes naive")],
              fontsize=FT, frameon=False, loc="upper left", ncol=2, columnspacing=1.2)


def timeline(axes, cax, S):
    """the three panels of td_tt_timeline on the suite's type sizes"""
    B, acc = S["blocks"], S["acc"]
    GAP = .9
    ypos, y = {}, 0.
    for bi, (name, b) in enumerate(B.items()):
        y += GAP if bi else 0.
        for r, _ in b:
            ypos[(r["topic"], r["name"])] = y
            y += 1.
    ytop, ybot = -.5, y - .5
    xmax = max(float(np.exp(np.log(r["x"][-1]) + (np.log(r["x"][-1]) - np.log(r["x"][-2])) / 2)) for r in S["rows"])
    cmap = plt.cm.viridis.copy()
    cmap.set_under(tl.NAIVE_GREY)
    cmap.set_bad("w")
    halo = [pe.withStroke(linewidth=2.6, foreground="k")]

    def mark(ax, yy, c, color, filled, marker, heat):
        if c is None or not np.isfinite(c["p"]) or c["p"] <= tl.XMIN * 1.3 or c["share"] < tl.SHOWN:
            return
        lo = c["lo"] if np.isfinite(c["lo"]) else c["p"]
        hi = c["hi"] if np.isfinite(c["hi"]) else xmax
        ax.plot([lo, hi], [yy] * 2, color=("w" if heat else color), lw=2.2 if filled else 1.2, zorder=6,
                solid_capstyle="butt", path_effects=halo if heat else None)
        ax.plot([c["p"]], [yy], marker=marker, ms=6.5 if filled else 5.5, mfc=(color if filled else "w"),
                mec=("k" if marker == "o" else color), mew=1.1, zorder=7, ls="none")

    im = None
    for j, run in enumerate(("trained", "echo")):
        ax = axes[j]
        for b in B.values():
            for r, e in b:
                q = r if run == "trained" else e
                if q is None:
                    continue
                yy = ypos[(r["topic"], r["name"])]
                im = ax.pcolormesh(tl._edges(q["x"]), [yy - .5, yy + .5], tl.fracs(q)[0][None], cmap=cmap, vmin=0.,
                                   vmax=1., shading="flat", rasterized=True)
                Lq = q["L"] if "L" in q else {lev: tl.cross(q, lev) for lev in (.1, .5)}
                mark(ax, yy, Lq[.1], tl.COL[run], False, "o" if run == "trained" else "s", True)
                mark(ax, yy, Lq[.5], tl.COL[run], True, "o" if run == "trained" else "s", True)
    n50 = sum(np.isfinite(d["c50"]["p"]) for d in S["esn"])
    title(axes[0], "d", "Trained RNN: share of the gap closed",
          r"$f = (y - y_{\mathrm{naive}})\,/\,(y_{\mathrm{Bayes}} - y_{\mathrm{naive}})$")
    title(axes[1], "e", "ESN: share of the gap closed", f"past $f = 0.5$ for good in {n50} of {len(S['esn'])} rows")
    for b in B.values():
        for r, _ in b:
            lab = tl._tex(r["name"]) + ("" if r["shape"] == "rise" else "  (returns)")
            axes[0].text(-.02, ypos[(r["topic"], r["name"])], lab, transform=axes[0].get_yaxis_transform(),
                         ha="right", va="center", fontsize=FT, color=tl.FAM_COL[r["family"]])
    cb = plt.colorbar(im, cax=cax, extend="both")
    cb.set_ticks([0, .5, 1])
    cb.set_ticklabels(["0 naive", ".5", "1 Bayes"])
    cb.ax.tick_params(labelsize=FT)
    cb.ax.set_title("$f$", fontsize=FT)
    ax = axes[2]
    for b in B.values():
        for i, (r, e) in enumerate(b):
            yy = ypos[(r["topic"], r["name"])]
            if i % 2 == 0:
                ax.axhspan(yy - .5, yy + .5, color="#f7f3e8", lw=0, zorder=0)
            mark(ax, yy, r["L"][.1], tl.FAM_COL[r["family"]], False, "o", False)
            mark(ax, yy, r["L"][.5], tl.FAM_COL[r["family"]], True, "o", False)
    title(ax, "f", "Crossings of the Trained RNN (ESN: none established)",
          r"open: $f > 0.1$ for good (onset);  filled: $f > 0.5$ for good (half-way)")
    for a_ in axes:
        xaxis(a_, passing=False, xmax=xmax)
        a_.set_xlim(tl.XMIN, xmax)
        a_.set_ylim(ybot, ytop)
        a_.set_yticks([])
        a_.spines[["top", "right"]].set_visible(True)
        a_.axvline(PASS, color="w" if a_ is not ax else "0.45", lw=1, ls=":", zorder=5)
    tested = B["tested"]
    yl = max(ypos[(r["topic"], r["name"])] for r, _ in tested if r["family"] in tl.EARLY)
    yf = min(ypos[(r["topic"], r["name"])] for r, _ in tested if r["family"] in tl.LATE)
    tr_ = mtr.blended_transform_factory(ax.transAxes, ax.transData)
    for a_ in axes:
        a_.axhline((yl + yf) / 2, color="k", lw=1.4, ls=(0, (6, 3)), zorder=8,
                   path_effects=[pe.withStroke(linewidth=3.4, foreground="w")])
    e0, e1 = S["ranges"]["early"]
    l0, l1 = S["ranges"]["late"]
    ax.text(1.03, (yl + yf) / 2 - .35, f"answers, splitting:\nhalf-way at\n{e0:,.0f}-{e1:,.0f}", transform=tr_,
            ha="left", va="bottom", fontsize=FT, linespacing=1.3)
    ax.text(1.03, (yl + yf) / 2 + .35, f"holding, coupling:\nhalf-way at\n{l0:,.0f}-{l1:,.0f}", transform=tr_,
            ha="left", va="top", fontsize=FT, linespacing=1.3)
    for name, lab in (("returns", "returns: not in\nthe family test"), ("learning", "own resamples:\nnot in the test")):
        ys = [ypos[(r["topic"], r["name"])] for r, _ in B[name]]
        for a_ in axes:
            a_.axhline(min(ys) - .5 - GAP / 2, color="k", lw=.8, zorder=8)
        ax.text(1.03, min(ys) - .4, lab, transform=tr_, ha="left", va="top", fontsize=FT, linespacing=1.3)


def figure():
    setup()
    D = {r: ck(r) for r in PR_RUNS + ("echo",)}
    S = tl.summary(tl.load_rows())
    W, H = 17.2, 12.6
    fig, add = canvas(W, H)
    TOPY, TOPH = 8.75, 3.0
    a_memory(add(.75, TOPY, 4.5, TOPH), D)
    b_copy(add(6.35, TOPY + 1.62, 4.5, 1.38), D)
    b_fr(add(6.35, TOPY, 4.5, 1.38))
    c_pr(add(11.95, TOPY, 4.5, TOPH), D)
    BOT, PH, PW = 1.75, 5.95, 3.45
    x0 = 3.35
    axes = [add(x0, BOT, PW, PH), add(x0 + PW + .3, BOT, PW, PH), add(x0 + 2 * PW + 1.3, BOT, PW, PH)]
    timeline(axes, add(x0 + 2 * PW + .42, BOT + 2.7, .14, 2.6), S)
    sep, sv = S["sep"], S["sep_variants"]
    lg = [c["p"] for c in S["log50"].values()]
    acc = S["acc"]
    fig.legend(handles=[Line2D([], [], color=tl.FAM_COL[f], lw=5, label=tl.FAM_NAME[f]) for f in tl.FAM_COL],
               loc="center", bbox_to_anchor=(.5, 1.12 / H), ncol=5, frameon=False, fontsize=FT, columnspacing=1.6,
               handlelength=1.2)
    note(fig, .86 / H,
         "a-c: 10,000 probe episodes (50 test environments x 100 latent pairs x 2 trajectories, 30 steps) at each of 114 "
         "checkpoints; one seed; dotted vertical: Trained RNN passes naive (880).\n"
         "d-f: the same probes (wormhole rows: 58 checkpoints; loss rows: 3,000 training episodes). Bars: 5-95% over "
         "1,000 environment resamples. Grey: at or below naive; white: no value.\n"
         f"Dashed line: every row above is half-way before every row below in {sep[.5] * 1000:,.0f} of 1,000 paired "
         f"resamples on the additive scale; rows interleave at f = .25 ({sep[.25]:.2f}), and on a log scale the late rows "
         f"are half-way at {min(lg):,.0f}-{max(lg):,.0f} (paired test "
         f"{sv['every late row with positive references on a log scale']:.2f}-"
         f"{sv['early and late rows on a log scale where defined']:.2f}).")
    return save(fig, "fig_main_1.png")


if __name__ == "__main__":
    figure()
