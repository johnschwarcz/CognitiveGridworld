"""Figures behind the panel (c) analyses of 21-22 Oct 2026 (reports: main/DATA/experimental/fig1c_compression/
report_{1..5}_*.md), drawn from the figure data in main/DATA/experimental/fig1c_compression/figdata (README.md,
round3/*README*, reeval/, round4b/README.md). From the repository root, in the eureka environment, on the CPU:

    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \
        main/REPLICATION/claude_plots/fig1c_mechanisms.py
    -> main/REPLICATION/figure_bin/fig1c_readout_merging.png    "Merging is in the readout, not the state"
       main/REPLICATION/figure_bin/fig1c_entanglement_steps.png "Entanglement, step by step"
       main/REPLICATION/figure_bin/fig1c_reevaluation.png       "Re-evaluation: fast through the warp"
       main/REPLICATION/figure_bin/fig1c_montage_{entanglement,fr}.png

Data: the alias scan (40 environments x 100 cells x 50 streams x 30 steps). Curves are means over environments with
95% percentile-bootstrap bands over environments (4000 resamples); the blurring and read-out panels leave out the
exactly context-symmetric environment e = 2, as their reports do. Entanglement effects compare streams of the same cell
at the same step, so the time course of entanglement cannot produce them; at t = 1 they are off scale (entanglement
holds only its within-step part) and are not drawn."""
import os
import sys
import numpy as np


def _root():
    for start in (globals().get("__file__"), globals().get("__vsc_ipynb_file__"), os.getcwd()):
        d = os.path.abspath(start) if start and os.path.exists(start) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return d
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError("project root not found")


ROOT = _root()
sys.path[:0] = [p for p in (ROOT,) if p not in sys.path]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from main.utils import apply_fig_style, fig_path                                  # noqa: E402

FS = apply_fig_style()
plt.rcParams["mathtext.fontset"] = "cm"                                           # math in Computer Modern, as the text
FD = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression", "figdata")
PANEL = (2.9, 2.0)                                                                # the house panel (main.utils.FIG_PANEL)
AGENT = {"Joint": ("Optimal", "k"), "Trained": ("Trained RNN", "#1f77b4"), "Pert": ("Wrong-table Bayes", "#7a5195"),
         "Naive": ("Naive", "0.5"), "Echo": ("ESN", "#ff7f0e"), "cfF": ("Naive update on Optimal belief", "0.5")}
T = np.arange(1, 31)


def band(X, keep=None, seed=0):
    """mean over environments (axis 0) and 95% percentile bootstrap band (4000 resamples), NaN-aware per step"""
    X = np.asarray(X, np.float64)
    if keep is not None:
        X = X[keep]
    idx = np.random.default_rng(seed).integers(0, len(X), (4000, len(X)))
    bs = np.nanmean(X[idx], 1)
    return np.nanmean(X, 0), np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)


def curve(ax, X, color, keep=None, ls="-", lw=1.4, t0=1, label=None, fill=True):
    m, lo, hi = band(X, keep)
    s = slice(t0 - 1, None)
    ax.plot(T[s], m[s], c=color, ls=ls, lw=lw, label=label, zorder=3)
    if fill:
        ax.fill_between(T[s], lo[s], hi[s], color=color, alpha=.15, lw=0, zorder=2)
    return m


def curve_t(ax, X, tt, color, keep=None, ls="-", lw=1.4, label=None, fill=True, marker=None, ms=2.5):
    """curve() on the steps tt (X: environments x len(tt))"""
    m, lo, hi = band(X, keep)
    ax.plot(tt, m, c=color, ls=ls, lw=lw, label=label, zorder=3, marker=marker, ms=ms)
    if fill:
        ax.fill_between(tt, lo, hi, color=color, alpha=.15, lw=0, zorder=2)
    return m


def curve_s(ax, S, color, ls="-", lw=1.4, label=None, fill=True):
    """a curve over t = 1..30 already summarized over environments as [mean, lo, hi]"""
    ax.plot(T, S[0], c=color, ls=ls, lw=lw, label=label, zorder=3)
    if fill:
        ax.fill_between(T, S[1], S[2], color=color, alpha=.15, lw=0, zorder=2)


def band2(X, pairs, n_var=50, seed=0):
    """mean over held-out environments and the 95% two-way bootstrap band over the 50 held-out variables: a pair of
    variables enters with the product of their multiplicities (pairs share variables, so an environment bootstrap
    would be too narrow)"""
    X = np.asarray(X, np.float64)
    cnt = np.random.default_rng(seed).multinomial(n_var, np.full(n_var, 1 / n_var), size=4000)
    w = cnt[:, pairs[:, 0]] * cnt[:, pairs[:, 1]]
    bs = (w @ X) / w.sum(1, keepdims=True)
    return X.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


def title2(ax, name, formula):
    """the panel's quantity: its name over its formula (house rule), centred over the axes"""
    ax.annotate(name, (.5, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points",
                ha="center", va="baseline", fontsize=FS["label"] + .5)
    ax.annotate(formula, (.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                ha="center", va="baseline", fontsize=FS["tick"], color="0.25")


def letter(fig, ax, s, dx=.62):
    """the panel letter on the baseline of the title's name line, dx inches left of the axes"""
    W, H = fig.get_size_inches()
    x0, y0, w, h = ax.get_position().bounds
    fig.text(x0 - dx / W, y0 + h + 17 / 72 / H, r"$\mathbf{%s}$" % s, fontsize=FS["label"] + 2, ha="left",
             va="baseline")


def clean(ax, xlabel=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(.5, 30.5)
    ax.set_xticks([1, 10, 20, 30])
    if xlabel:
        ax.set_xlabel("Step $t$", labelpad=1)


def grid_axes(fig, W, H, ncol, nrow, x0, y_top, gx, gy):
    """panels of the house size in inches, from the top left"""
    axes = []
    for r in range(nrow):
        for c in range(ncol):
            x, y = x0 + c * (PANEL[0] + gx), y_top - PANEL[1] - r * (PANEL[1] + gy)
            axes.append(fig.add_axes([x / W, y / H, PANEL[0] / W, PANEL[1] / H]))
    return axes


# ═══════════════════════════════════════════════════════════════════
# Figure 1: merging is in the readout, not the state
# ═══════════════════════════════════════════════════════════════════

def fig_readout():
    F = np.load(os.path.join(FD, "fig1_readout.npz"))
    B = np.load(os.path.join(FD, "round3", "panel_blur.npz"))
    R = np.load(os.path.join(FD, "round3", "panel_readout.npz"))
    keep_b = B["include"]                                                         # leaves out e = 2
    keep_r = ~R["env_symmetric"]
    W, H = 3 * PANEL[0] + 2 * .95 + .9, 2 * PANEL[1] + 1.15 + .62 + 1.05
    fig = plt.figure(figsize=(W, H))
    ax = grid_axes(fig, W, H, 3, 2, .8, H - .62, .95, 1.15)
    agents5 = list(B["agents"])                                                   # Joint, Trained, Pert, Naive, Echo
    # (a) blurring: local mixing, probability (solid) and log space (dashed)
    a = ax[0]
    for i, k in enumerate(agents5):
        curve(a, B["mix_prob"][:, i], AGENT[k][1], keep_b, label=AGENT[k][0])
        curve(a, B["mix_log"][:, i], AGENT[k][1], keep_b, ls="--", lw=1., fill=False)
    a.axhline(float(B["chance"]), c="0.6", lw=.6, ls=":")
    a.text(30, float(B["chance"]) - .01, "chance", fontsize=FS["tick"] - 1, color="0.45", ha="right", va="top")
    a.set_ylim(.45, 1.0)
    a.set_ylabel("Share of the 10 nearest\nfrom another cell", linespacing=.95)
    title2(a, "Blurring: local mixing of particles", r"$\langle \#\{k \in \mathrm{NN}_{10}(j): c_k \neq c_j\}/10 \rangle_j$")
    a.legend(handles=[Line2D([], [], c="0.3", ls="-", label="probability (as drawn)"),
                      Line2D([], [], c="0.3", ls="--", lw=1., label="log-odds (state)")], frameon=False,
             fontsize=FS["tick"] - .5, loc="lower left")
    # (b) population information in the answers vs in the data (the Naive's state)
    a = ax[1]
    exact = dict(zip(F["b_exact_agents"], F["b_mi_answer_exact"]))
    for i, k in enumerate(F["b_agents"]):
        X = exact[k] if k in exact else F["b_mi_answer"][i]                       # exact for the Bayesian agents
        curve(a, X, AGENT[k][1], label=AGENT[k][0])
    curve(a, F["b_mi_data"], "#31a354", lw=2.4, label="data = Naive state")
    a.axhline(float(F["b_log100"]), c="0.6", lw=.6, ls=":")
    a.text(1.5, float(F["b_log100"]) - .06, r"$\log 100$", fontsize=FS["tick"] - 1, color="0.45", va="top")
    a.set_ylim(0, 4.8)
    a.set_ylabel("Information (nats)")
    title2(a, "Information about the cell", r"$I(\mathrm{cell};\,\hat a_t)$  vs  $I(\mathrm{cell};\,n_t)$")
    a.legend(frameon=False, fontsize=FS["tick"] - 1, loc="lower right", handlelength=1.4, labelspacing=.25)
    # (c) five readers of the Naive (solid) and the ESN (dashed)
    a = ax[2]
    rc = {"own": "#c7e9c0", "lda_prob": "#74c476", "lda_logbelief": "#31a354", "bayes_on_decoded_counts": "#00441b",
          "bayes_ceiling": "k"}
    rl = {"own": "own answers", "lda_prob": "linear, probabilities", "lda_logbelief": "linear, log-beliefs",
          "bayes_on_decoded_counts": "Bayes on counts from log-odds", "bayes_ceiling": "Bayes ceiling"}
    for j, rd in enumerate(F["c_readers"]):
        for i, ls in ((0, "-"), (1, "--")):
            if rd == "bayes_ceiling" and i == 1:
                continue
            curve(a, F["c_acc"][i, j], rc[rd], ls=":" if rd == "bayes_ceiling" else ls, lw=1.6 if i == 0 else 1.1,
                  fill=i == 0, label=rl[rd] if i == 0 else None)
    a.set_ylim(0, .78)
    a.set_ylabel("100-way accuracy")
    title2(a, "Who reads the information", r"$P(\hat c = c)$:  Naive (solid), ESN (dashed)")
    a.legend(frameon=False, fontsize=FS["tick"] - 1, loc="upper left", handlelength=1.4, labelspacing=.25)
    # (d) merges by rule: the format does not force them at T
    a = ax[3]
    keys = list(R["format_keys"])
    rules = (("Joint", "Optimal", "k", "-"), ("Pert", "Wrong-table Bayes", "#7a5195", "-"), ("Naive", "Naive", "0.5", "-"),
             ("Echo", "ESN (trained read-out)", "#ff7f0e", "-"),
             ("ce_acc", "best rule linear in counts,\nper environment", "#31a354", "--"),
             ("pat_t", "linear in pattern counts", "#31a354", ":"))
    for k, lab, c, ls in rules:
        curve(a, np.maximum(R["format_merged"][:, keys.index(k)], .5), c, keep_r, ls=ls, label=lab)
    a.set_yscale("log")
    a.set_ylim(.4, 4e4)
    a.set_ylabel("Merged cell pairs (of 4950)")
    title2(a, "Merging by read-out rule", r"pairs with $\hat a(c) = \hat a(c') \notin \{c, c'\}$")
    a.legend(frameon=False, fontsize=FS["tick"] - 1.5, loc="upper right", handlelength=1.6, labelspacing=.2)
    # (e) within-cell radius vs lattice spacing in Figure 1's fixed plane
    a = ax[4]
    for i, k in enumerate(F["e_agents"]):
        if k == "Trained":
            continue
        curve(a, F["e_radius"][i], AGENT[k][1], label=AGENT[k][0])
        curve(a, F["e_spacing"][i], AGENT[k][1], ls="--", lw=1.1, fill=False)
    a.set_ylim(0, .25)
    a.set_ylabel("Belief-plane units")
    title2(a, "Cell clouds vs lattice (Fig. 1 plane)", r"RMS radius (solid)  vs  neighbour spacing (dashed)")
    a.legend(frameon=False, fontsize=FS["tick"] - 1, loc="lower right", handlelength=1.4, labelspacing=.25)
    # (f) one typical cell's streams (env 34): Optimal vs ESN at t = 5, 15, 30
    ax[5].remove()
    tag = "typical"
    xy, share = F[f"d_{tag}_xy"], F[f"d_{tag}_share"]
    lat = F["d_centroids_all"]
    cmap = plt.get_cmap("coolwarm")
    x_l = 2 * PANEL[0] + 2 * .95 + .8
    y_b = H - .62 - 2 * PANEL[1] - 1.15
    mw, mh = (PANEL[0] - .2) / 3, (PANEL[1] - .25) / 2
    for r_, (ai, alab) in enumerate(((0, "Optimal"), (1, "ESN"))):
        pts = np.concatenate([lat[ai].reshape(-1, 2), xy[ai].reshape(-1, 2)])
        lo, hi = np.percentile(pts, 1, 0), np.percentile(pts, 99, 0)
        c0, s = (lo + hi) / 2, (hi - lo).max() * .62
        for c_, ti in enumerate((4, 14, 29)):
            m = fig.add_axes([(x_l + c_ * (mw + .1)) / W, (y_b + (1 - r_) * (mh + .25)) / H, mw / W, mh / H])
            m.scatter(*lat[ai, ti].T, s=3, c="0.8", lw=0, zorder=1)
            m.scatter(*xy[ai, :, ti].T, s=7, c=cmap(share[:, ti]), lw=0, zorder=3)
            for cen, mk in ((f"d_{tag}_centroid_home", "o"), (f"d_{tag}_centroid_nb", "s")) + \
                    (((f"d_{tag}_centroid_esn_dest", "^"),) if ai == 1 else ()):
                m.plot(*F[cen][ai, ti], marker=mk, ms=6, mfc="white", mec="k", mew=.8, ls="none", zorder=4)
            m.set_xlim(c0[0] - s, c0[0] + s), m.set_ylim(c0[1] - s, c0[1] + s)
            m.set_aspect("equal", adjustable="box")
            m.set_xticks([]), m.set_yticks([])
            for sp_ in m.spines.values():
                sp_.set_linewidth(.5), sp_.set_color("0.6")
            if r_ == 0:
                m.set_title(f"$t = {ti + 1}$", fontsize=FS["tick"], pad=2)
            if c_ == 0:
                m.set_ylabel(alab, fontsize=FS["tick"], labelpad=2)
    cell, nb = F[f"d_{tag}_cell"], F[f"d_{tag}_nb_cell"]
    fake = fig.add_axes([x_l / W, (y_b + 2 * mh + .25 + .17) / H, (3 * mw + .2) / W, .01 / H])
    fake.axis("off")
    title2(fake, "One cell's 50 streams (env 34)", r"colour: $P(c')\,/\,[P(c) + P(c')]$ given the counts")
    dest = tuple(int(v) for v in F[f"d_{tag}_dest_esn_modal_cell"])
    fig.text(x_l / W, (y_b - .42) / H, "circle: the cell $c = %s$;  square: its neighbour $c' = %s$;\ntriangle: the ESN's "
             "answer %s;  grey: all 100 cell means" % (tuple(int(v) for v in cell), tuple(int(v) for v in nb), dest),
             fontsize=FS["tick"] - 1, color="0.25", va="top", linespacing=1.2)
    cax = fig.add_axes([(x_l + .65) / W, (y_b - .2) / H, 1.4 / W, .07 / H])
    cax.imshow(np.linspace(0, 1, 256)[None], cmap=cmap, aspect="auto", extent=(0, 1, 0, 1))
    cax.set_yticks([]), cax.set_xticks([0, .5, 1], ["0 (cell)", ".5", "1 (neighbour)"])
    cax.tick_params(labelsize=FS["tick"] - 1.5, length=1.5, pad=1)
    for a_, s_ in zip(ax[:5], "abcde"):
        clean(a_)
        letter(fig, a_, s_)
    letter(fig, fake, "f", dx=.3)
    path = fig_path("fig1c_readout_merging.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════
# Figure 2: entanglement, step by step
# ═══════════════════════════════════════════════════════════════════

def fig_entanglement():
    E = np.load(os.path.join(FD, "fig2_entanglement.npz"))
    O = np.load(os.path.join(FD, "round3", "panel_origin.npz"))
    ag, rk = list(E["agents"]), list(E["ranks"])                                  # Joint Trained Pert Naive Echo cfF
    own, part = rk.index("own"), rk.index("partner")
    W, H = 3 * PANEL[0] + 2 * .95 + .9, 2 * PANEL[1] + 1.15 + 1.25
    fig = plt.figure(figsize=(W, H))
    ax = grid_axes(fig, W, H, 3, 2, .8, H - .62, .95, 1.15)
    order = ("Joint", "Trained", "Pert", "Naive", "Echo", "cfF")
    # (a) toward the truth at the latent's own entangled steps
    a = ax[0]
    for k in order:
        curve(a, E["tow"][ag.index(k), own], AGENT[k][1], t0=2, ls="--" if k == "cfF" else "-", label=AGENT[k][0])
    a.axhline(0, c="k", lw=.5, alpha=.4)
    a.set_ylabel("Value units per step")
    title2(a, "Toward the truth at own entangled steps", r"slope of $|x_{t-1} - r^*| - |x_t - r^*|$ on own rank")
    a.legend(frameon=False, fontsize=FS["tick"] - 1.5, loc="lower right", handlelength=1.6, labelspacing=.2)
    # (b) edge mass: partner (solid) vs own (dotted) ranking
    a = ax[1]
    for k in ("Joint", "Trained", "Pert", "Naive", "Echo"):
        curve(a, E["edge"][ag.index(k), part], AGENT[k][1], t0=2)
        if k in ("Joint", "Naive"):
            curve(a, E["edge"][ag.index(k), own], AGENT[k][1], t0=2, ls=":", lw=1.2, fill=False)
    a.axhline(0, c="k", lw=.5, alpha=.4)
    a.set_ylabel("Probability per step")
    title2(a, "Edge mass at entangled steps", r"slope of $\Delta[b(0) + b(9)]$, partner (solid) / own (dotted)")
    # (c) shared information pushes beliefs outward (the clean radial-velocity display)
    a = ax[2]
    m0, lo0, hi0 = band(O["dvr_t_null"][0])
    a.fill_between(T[1:], lo0[1:], hi0[1:], color="0.85", lw=0, zorder=1, label="shuffled null")
    for i, k in enumerate(O["agents"]):
        curve(a, O["dvr_t"][i], AGENT[str(k)][1], t0=2, label=AGENT[str(k)][0])
    a.axhline(0, c="k", lw=.5, alpha=.4)
    a.set_ylabel("Value units per step")
    title2(a, "Shared information pushes beliefs out", r"$\Delta v_r$: top $-$ bottom third of partner entanglement")
    a.legend(frameon=False, fontsize=FS["tick"] - 1.5, loc="upper right", handlelength=1.6, labelspacing=.2)
    # (d) refinement off the local origin, outer side (solid) vs centre side (dashed), own ranking
    a = ax[3]
    for k in ("Joint", "Naive"):
        curve(a, E["ref_outer"][ag.index(k), own], AGENT[k][1], t0=2, label=f"{AGENT[k][0]}, outer side")
        curve(a, E["ref_inner"][ag.index(k), own], AGENT[k][1], t0=2, ls="--", lw=1.1, label=f"{AGENT[k][0]}, centre side")
    a.axhline(0, c="k", lw=.5, alpha=.4)
    a.set_ylabel("Value units per step")
    title2(a, "Refinement off the local origin", r"slope of $\Delta x \cdot \mathrm{sign}(r^* - m_{t-1})$ on own rank")
    a.legend(frameon=False, fontsize=FS["tick"] - 1.5, loc="lower right", handlelength=1.6, labelspacing=.2)
    # (e) the Naive's toward-truth effect by kappa bin (own ranking), with kappa against the other latent's value
    a = ax[4]
    bins = list(E["kappa_bins"])
    blab = {bins[0]: r"$\kappa < 0$ (reversed)", bins[1]: r"$0 \leq \kappa < 1.5$", bins[2]: r"$\kappa \geq 1.5$"}
    bcol = {bins[0]: "#b2182b", bins[1]: "#f4a582", bins[2]: "#2166ac"}
    for j, b in enumerate(bins):
        curve(a, E["tow_by_kappa"][ag.index("Naive"), 1, j], bcol[b], t0=2, label=blab[b])  # rank axis: house, own
    a.axhline(0, c="k", lw=.5, alpha=.4)
    a.set_ylabel("Value units per step")
    title2(a, r"Naive: toward the truth by $\kappa$", r"$\kappa$ = true slope given $r^*_{\mathrm{other}}$ / averaged slope")
    a.legend(frameon=False, fontsize=FS["tick"] - 1.5, loc="lower right", handlelength=1.6, labelspacing=.2)
    ins = a.inset_axes([.55, .37, .41, .30])
    ins.fill_between(range(10), E["kappa_q25_by_other"], E["kappa_q75_by_other"], color="0.85", lw=0)
    ins.plot(range(10), E["kappa_median_by_other"], c="k", lw=1.1)
    ins.axhline(1, c="0.5", lw=.5, ls=":")
    ins.set_xticks([0, 9]), ins.set_yticks([0, 1, 2])
    ins.tick_params(labelsize=FS["tick"] - 2, length=1.5, pad=1)
    ins.set_xlabel(r"other latent's value", fontsize=FS["tick"] - 2, labelpad=0)
    ins.set_ylabel(r"$\kappa$", fontsize=FS["tick"] - 1.5, labelpad=0)
    # (f) where the Naive's drift toward its fold accumulates: by entanglement third (own ranking), Pert dashed
    a = ax[5]
    th = list(E["drift_thirds"])
    tcol = {"low": "#c6dbef", "mid": "#6baed6", "high": "#08306b"}
    for j, q in enumerate(th):
        curve(a, np.cumsum(E["drift_naive_fold"][1, j], -1), tcol[q], label=f"Naive fold, {q} third")
        curve(a, np.cumsum(E["drift_pert"][1, j], -1), tcol[q], ls="--", lw=1.1, fill=False)
    a.set_ylabel("Log-odds toward the fold (nats)")
    title2(a, "Where the fold accumulates", r"$\Sigma_{s \leq t}\,[\ell_s(a) - \ell_s(r^*)]$ by own-entanglement third")
    a.legend(handles=[Line2D([], [], c=tcol[q], label=f"{q} third") for q in th] +
             [Line2D([], [], c="0.3", ls="--", lw=1.1, label="wrong-table Bayes")],
             frameon=False, fontsize=FS["tick"] - 1.5, loc="upper left", handlelength=1.6, labelspacing=.2)
    for a_, s_ in zip(ax, "abcdef"):
        clean(a_)
        letter(fig, a_, s_)
    path = fig_path("fig1c_entanglement_steps.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════
# Montage: is the entanglement / FR gradient in the particle clouds general across environments?
# ═══════════════════════════════════════════════════════════════════

def fig_gradient_montage(colour="entanglement", n_env=6):
    """T = 30 particles (all 100 cells x 50 streams) of n_env scan environments in each agent's Figure 1 plane, coloured
    by each stream's percentile within its environment of the summed entanglement sum_t rev_t ('entanglement') or of
    the factorization regret FR_T ('fr': the scan's stored FR, the goal variable's KL(Optimal || Naive), the paper's
    forward KL, exact, since scan_meta.npz was patched on 2026-11-10; the symmetric clipped KL, now its FR_sym,
    before). Environments: those at evenly spaced ranks of the Naive's T = 30 accuracy
    (the context-symmetric e = 2 left out) -> figure_bin/fig1c_montage_<colour>.png"""
    sys.path.insert(0, os.path.join(ROOT, "main", "REPLICATION", "claude_plots"))
    import fig1 as cf
    from fig1_movie import ENT_CMAP
    scan = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
    meta = np.load(os.path.join(scan, "scan_meta.npz"))
    G = cf.belief_geometry()
    agents = ("Joint", "Trained", "Naive", "Echo")
    A = {a: np.load(os.path.join(scan, f"scan_{a}.npz")) for a in agents}
    cells = meta["cells"][meta["cell_of"]]
    ans = A["Naive"]["answer"][:, :, -1]
    acc = ((ans[..., 0] == cells[None, :, 0]) & (ans[..., 1] == cells[None, :, 1])).mean(1)
    order = [e for e in np.argsort(acc) if e != 2]
    envs = [order[int(round(q))] for q in np.linspace(0, len(order) - 1, n_env)]
    val = meta["rev"].sum(-1) if colour == "entanglement" else meta["FR"][:, :, -1]
    perm = np.random.default_rng(0).permutation(len(meta["cell_of"]))
    pw, gap = 1.25, .12
    W, H = .55 + 4 * (pw + gap), .55 + n_env * (pw + gap) + .35
    fig = plt.figure(figsize=(W, H))
    for r, e in enumerate(envs):
        pct = np.argsort(np.argsort(val[e])) / (len(val[e]) - 1)
        for c, a in enumerate(agents):
            m, C, Q = G[(a, 30)]["proj"]
            X = (A[a]["beliefs"][e][:, -1].astype(np.float64) - m) @ C.T @ Q
            ax = fig.add_axes([(.45 + c * (pw + gap)) / W, (H - .5 - (r + 1) * pw - r * gap) / H, pw / W, pw / H])
            ax.scatter(*X[perm].T, c=ENT_CMAP(pct[perm]), s=.6, lw=0, rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([]), ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            if r == 0:
                ax.set_title(AGENT[a][0], fontsize=FS["tick"], color=AGENT[a][1] if a != "Naive" else "0.4", pad=3)
            if c == 0:
                ax.set_ylabel("env %d\nNaive acc %.2f" % (e, acc[e]), fontsize=FS["tick"] - 1.5, labelpad=2)
    what = r"summed entanglement $\Sigma_t\,\mathrm{rev}_t$" if colour == "entanglement" else r"factorization regret $FR_T$"
    fig.text(.5, (H - .1) / H, f"T = 30 particles of six environments (all 100 cells x 50 streams), each agent's Figure 1"
             f" plane;\ncoloured by each stream's percentile of {what} within its environment (grey low, dark red high)",
             ha="center", va="top", fontsize=FS["tick"] - 1.5, linespacing=1.3)
    path = fig_path(f"fig1c_montage_{colour}.png")
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path, [int(e) for e in envs]


# ═══════════════════════════════════════════════════════════════════
# Figure 3: re-evaluation - fast through the warp, steered by entanglement (report_4_reevaluation.md)
# ═══════════════════════════════════════════════════════════════════

def fig_reevaluation():
    """from figdata/reeval/fig_arrays.npz (its README key): where VALUE zips start late (t 13-30) in the Optimal's
    plane, for the Naive and the Optimal; how much faster the most torn particles move; zip rates in the entangled vs
    calm third of a cell's streams; the toward-truth efficiency of fast-zone zips; the net answer correction at the
    Naive's / ESN's entangled zips for them and for the Optimal / Trained on the same steps"""
    R = np.load(os.path.join(FD, "reeval", "fig_arrays.npz"))
    ag = list(R["agents"])                                                        # Joint, Trained, Pert, Naive, Echo
    W, H = 3 * PANEL[0] + 2 * .95 + .9, 2 * PANEL[1] + 1.15 + .62 + 1.0
    fig = plt.figure(figsize=(W, H))
    ax = grid_axes(fig, W, H, 3, 2, .8, H - 1.0, .95, 1.15)
    hand = [Line2D([], [], c=AGENT[k][1], lw=1.6, label=AGENT[k][0]) for k in ("Joint", "Trained", "Pert", "Naive", "Echo")]
    hand += [Line2D([], [], c="0.3", ls="-", label="entangled third"), Line2D([], [], c="0.3", ls=":", label="calm third")]
    fig.legend(handles=hand, loc="upper center", bbox_to_anchor=(.5, 1 - .12 / H), ncol=7, frameon=False,
               fontsize=FS["tick"], handlelength=1.8, columnspacing=1.4)
    lo, hi = R["plane_range"]
    for k, (key, nm) in enumerate((("naive", "Naive"), ("opt", "Optimal"))):
        a = ax[k]
        n_, z_ = R[f"map_{key}_in_opt_particles"], R[f"map_{key}_in_opt_zipstarts"]
        rate = np.where(n_ >= 20, z_ / np.maximum(n_, 1), np.nan)
        im = a.imshow(rate.T, origin="lower", extent=(lo, hi, lo, hi), cmap="magma", vmin=0, vmax=.4,
                      interpolation="nearest", aspect="equal")
        a.set_xticks([]), a.set_yticks([])
        m_, l_, h_ = band(R[f"conc_late_{key}_in_opt"][:, None])
        a.set_xlabel("zip starts cluster: %.2f [%.2f, %.2f] nats" % (m_[0], l_[0], h_[0]), labelpad=4,
                     fontsize=FS["tick"])
        title2(a, f"Where {nm} zips start (late)", r"zip starts / particles per bin, Optimal's plane (base 0.1)")
        if k == 1:
            cax = fig.add_axes([(a.get_position().x1 * W + .06) / W, a.get_position().y0, .07 / W,
                                a.get_position().height])
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=FS["tick"] - 1, length=1.5, pad=1)
    a = ax[2]
    for k in ("Naive", "Echo", "Trained"):
        X = R[f"torn_excess_{'ESN' if k == 'Echo' else k}"]
        curve(a, X[..., 0], AGENT[k][1], t0=2)
        curve(a, X[..., 1], AGENT[k][1], t0=2, ls="--", lw=1.0, fill=False)
    a.axhline(0, c="k", lw=.5, alpha=.4)
    a.set_ylabel("Value units per step")
    title2(a, "Moving through torn regions is fast", r"$E[|\Delta x|_1\,|\,$most torn 10%$] - E[|\Delta x|_1]$")
    a.text(29.5, a.get_ylim()[1] * .95, "dashed: the Optimal\non the same streams", ha="right", va="top",
           fontsize=FS["tick"] - 1, color="0.3")
    for i, (key, ylab, ttl, frm) in enumerate((
            ("zip_rate_calm_ent", "P(zip)", "Entanglement marks when particles zip",
             r"$P(\mathrm{zip} \mid$ entanglement third within (step, cell)$)$"),
            ("zone_zip_efficiency_calm_ent", "Toward-truth efficiency", "Where fast-zone zips go",
             r"$\Sigma(|x_{t-1} - r^*| - |x_t - r^*|) \,/\, \Sigma |x_t - x_{t-1}|_1$"))):
        a = ax[3 + i]
        a.axvspan(.5, 1.5, color="0.93", lw=0, zorder=0)
        for k in ("Joint", "Trained", "Pert", "Naive", "Echo"):
            X = R[key][:, ag.index(k)]
            curve(a, X[..., 1], AGENT[k][1])
            curve(a, X[..., 0], AGENT[k][1], ls=":", lw=1.1, fill=False)
        a.axhline(.1 if i == 0 else 0, c="k", lw=.5, alpha=.4)
        a.set_ylabel(ylab)
        title2(a, ttl, frm)
    a = ax[5]
    for src, pair, ls in ((0, ("Naive", "Joint"), "-"), (1, ("Echo", "Trained"), "--")):
        for k in pair:
            curve(a, R["same_step_net"][:, src, 1, ag.index(k)], AGENT[k][1], t0=2, ls=ls,
                  label=f"{AGENT[k][0]}, at {'Naive' if src == 0 else 'ESN'}'s entangled zips")
    a.axhline(0, c="k", lw=.5, alpha=.4)
    a.set_ylabel("Net correction per zip")
    title2(a, "Same steps, different landing", r"$P(\mathrm{right}_t) - P(\mathrm{right}_{t-1})$ of the answer pair")
    a.legend(frameon=False, fontsize=FS["tick"] - 1.5, loc="upper right", handlelength=1.6, labelspacing=.2)
    for a_, s_ in zip(ax, "abcdef"):
        if s_ not in "ab":
            clean(a_)
        letter(fig, a_, s_, dx=.62 if s_ not in "ab" else .25)
    path = fig_path("fig1c_reevaluation.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# Figures 4 and 5 (fig1c_pair_distances.png, fig1c_fr_distance.png) moved on 25 Oct 2026 to the tracked plotter
# main/REPLICATION/plot_fig1c_distances.py


if __name__ == "__main__":
    for f in (fig_readout, fig_entanglement, fig_reevaluation):
        print("saved", f())
    for colour in ("entanglement", "fr"):
        print("saved", fig_gradient_montage(colour))
