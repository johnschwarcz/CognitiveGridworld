"""The two supplementary figures of supplementary_plotter2.py that involve Factorization Regret, rendered without running the
plotter's whole main block (which also draws the event-dynamics panels and opens windows):

  figure_bin/diagnostics_panel_early.svg   learning dynamics (revised draft Fig. 15, submission Fig. 13): accuracy, the FR-accuracy
                                           correlation and PR(recurrent) through training (forward FR; sp.diagnostics_runs)
  figure_bin/calibration_by_tier.svg/.png  confidence-accuracy calibration (revised draft Fig. 11): (a) calibration lines,
                                           (b) accuracy / confidence by Factorization Regret (forward FR)
  figure_bin/diagnostics_panel_early_grid.svg   (--grid) the learning-dynamics figure as a 2 x 6: its six panels, naive
                                           errors and the grid decomposition of supp_learning_behaviour (figure_grid)
  figure_bin/diagnostics_panel_log.svg     (--log) a 2 x 3 on log training episodes over all of training: behaviour,
                                           John 11-13's panel order, removing the shared grid in one panel (figure_log)
  figure_bin/diagnostics_panel_log_forward.svg   (--log --forward) the same with (c, f) from the forward FR, exact, at
                                           the 114 checkpoints on fresh episodes like the runs' tests (FR_FRESH)

The calibration data come from harvests of the trained networks (main/DATA/cache/calibration_ctx<C>.npz; C = 2 is the main
supplementary harvest's networks). To rebuild one (CPU; 11-10: C = 2 with 250 x 1,000, C = 3 with 80 x 500 trajectories per
network, ~1.5 h each on one core):

    python fig_supp_diagnostics.py --harvest C EPISODES BATCH

Run from anywhere with the eureka env; MPLBACKEND=Agg. Seconds once the harvests and the main supplementary harvest exist
(this replaces the scratch driver used on 11-10, which wrote the same files)."""
import os
import sys
import time
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
import supplementary_plotter2 as sp                                               # noqa: E402


def harvest(ctx, episodes, batch):
    t0 = time.time(); out = sp.harvest_calibration(ctx, episodes=episodes, batch_num=batch)
    for tag in ("tr", "ec"):
        print(f"C = {ctx} {tag}: n {out[tag + '_fr'].size:,}; accuracy joint {out[tag + '_joint_acc'].mean():.4f} naive "
              f"{out[tag + '_naive_acc'].mean():.4f} net {out[tag + '_net_acc'].mean():.4f}; FR mean {out[tag + '_fr'].mean():.3f} "
              f"max {out[tag + '_fr'].max():.1f}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


def figures():
    # the plotter's own configuration (its main block): the cached main harvest is keyed by it
    CFG = dict(mode="SANITY", cuda=0, episodes=50, checkpoint_every=1, realization_num=10, hid_dim=1000, obs_num=5, show_plots=False,
               batch_num=10000, step_num=30, state_num=500, learn_embeddings=False, classifier_LR=.001, ctx_num=2, training=False)
    trained, echo = sp.collect_or_load(CFG, sp.DYN_PARAMS)
    diag_tr, diag_ec = sp.diagnostics_runs()
    figA2, axesA2 = plt.subplots(2, 3, figsize=sp.PLOT_CFG["figsize_A_early"], squeeze=False)
    sp.plot_diagnostics_early(axesA2, diag_tr, "Trained"); sp.plot_diagnostics_early(axesA2, diag_ec, "Echo")
    sp.mark_bounds((diag_tr, diag_ec), y_axes=(axesA2[0, 0], axesA2[1, 0]), x_axes=(axesA2[1, 1],))
    sp.finalize_layout_early(axesA2); sp.note_counts(figA2, diag_tr, y=-.05)
    figE, axesE = plt.subplots(1, 3, figsize=sp.PLOT_CFG["figsize_E"], gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
    ctxs = sp.plot_calibration_tier(figE, axesE, (trained, echo))
    figA2.savefig(sp.fig_path("diagnostics_panel_early.svg"), format="svg", bbox_inches="tight")
    figE.savefig(sp.fig_path("calibration_by_tier.svg"), format="svg", bbox_inches="tight")
    figE.savefig(sp.fig_path("calibration_by_tier.png"), dpi=200, bbox_inches="tight")
    print(f"written to figure_bin: diagnostics_panel_early.svg, calibration_by_tier.svg, calibration_by_tier.png (calibration columns {ctxs})")


# ---------------------------------------------------------------------------------------------------------------------
# The learning-dynamics figure with naive errors, the grid decomposition and the gradient-FR maps (John 11-12): a 2 x 6
# on testing episodes (as diagnostics_panel_early) and a 3 x 4 with every panel through time on log training episodes
LOGGED_R = os.path.join(ROOT, "main", "DATA", "experimental", "fr_definition", "fr_logged_implied.npz")
FRESH = os.path.join(ROOT, "main", "DATA", "experimental", "fr_definition", "fr_correct_fresh.npz")   # fr_correct_fresh.py
FR_FULL = True          # John 11-12: "temporarily use the previously calculated FR so we can get the full data (not the
                        # downsampled)": the FR correlation at every test, implied by the logged fits (the symmetric FR
                        # of the time; logged_r). False: the forward-FR replay at the 114 saved checkpoints (dots)
FR_FRESH = False        # True (--forward; John 11-13: "can you please show it for the forward version"): the forward FR,
                        # exact, at the 114 checkpoints on 40,000 episodes drawn like the runs' own tests
                        # (fr_correct_fresh.py), not the probe set -> diagnostics_panel_log_forward.svg
LW_NEW = 2.5            # the panels added 11-12 (naive errors, grid decomposition): thicker lines (John 11-12)
PAD, FFS = 21, 11       # title pad (pt; room for the formula under a pair's title), formula and key size
LETTER_FS = 14          # panel letters "(a)", "(b)", ... centred under each panel in sans-serif (John 11-13: "copy my
                        # preferred labeling style"), the size of the panel titles
GRID_LS = {"tot": "-", "g": (0, (3.5, 1.8)), "rest": (0, (1, 1.3))}
GRID_PAIRS = (("cell", "Context-invariant PR", r"$u_T^{\rm grid} = \mathbb{E}[\bar u_T\,|\,\mathbf{r}]$"),
              ("envcell", "Context-dependent PR", r"$u_T^{\rm grid} = \mathbb{E}[\bar u_T\,|\,\mathbf{r},Z] - "
               r"\mathbb{E}[\bar u_T\,|\,\mathbf{r}] - \mathbb{E}[\bar u_T\,|\,Z]$"))
GRID_SH = (("t30_loo_cell", GRID_LS["g"], r"PR$(u_T - \mathbb{E}[u_T\,|\,\mathbf{r}])$"),   # (td_tt_griddim key,
           ("t30_pr_total", "-", r"PR$(u_T)$"))                              # line style, label); no centring bar (John
GRID_SH_TITLE = r"Removing the shared grid $\mathbb{E}[u_T\,|\,\mathbf{r}]$"         # 11-13): PR is of the covariance,
                                                                                    # and the grand mean cancels in
                                                                                    # u_T - E[u_T | r]
NAIVE_Y = "Naive errors repeated (fraction)"
PR_LABEL = r"PR(RNN) $-$ PR(read-in)"
ACC_LABEL = "Accuracy (average over the batch)"
NETS = (("trained", "Trained"), ("echo", "Echo"))


def implied_r(coef, acc, fr, chunk=32):
    """sp.fr_correlation's r between FR_T and correctness at T in each test batch, from the batch's logged logistic
    slope (coef) and accuracy (acc), given FR's distribution (fr), vectorized over tests: the intercept by Newton's
    method (|mean sigmoid - accuracy| < 1e-11), then r = cov(FR, sigmoid(a + b FR)) / (sd(FR) sqrt(p (1 - p))). On the
    supplementary harvest's runs it equals that function's cache to 8e-14 (11-12, every 50th test)."""
    fr = np.asarray(fr, np.float64).ravel()
    m, sd, n = fr.mean(), fr.std(), fr.size
    p = np.clip(np.asarray(acc, np.float64), 1e-3, 1 - 1e-3)
    b = np.asarray(coef, np.float64)
    out = np.full(len(b), np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        for s0 in range(0, len(b), chunk):
            bb, pp = b[s0:s0 + chunk], p[s0:s0 + chunk]
            a = np.log(pp / (1 - pp)) - bb * m
            for _ in range(80):
                s = 1 / (1 + np.exp(-(a[:, None] + bb[:, None] * fr)))
                f = s.mean(1) - pp
                a = np.clip(a - f / np.maximum((s * (1 - s)).mean(1), 1e-300), -30, 30)
                if np.all((np.abs(f) < 1e-11) | np.isnan(f)):
                    break
            s = 1 / (1 + np.exp(-(a[:, None] + bb[:, None] * fr)))
            out[s0:s0 + chunk] = (s * (fr - m)).sum(1) / (n - 1) / (sd * np.sqrt(pp * (1 - pp)))
    return out


def logged_r():
    """{"Trained", "Echo"}: r at every test (10,001) of sp.diagnostics_runs' two runs, implied by each test's logged
    logistic fit of correctness on env.SII (SII_coef: the symmetric KL clipped at 1e-15 when these runs were made, NOT
    the paper's forward FR) and its accuracy (implied_r), with FR's distribution from the supplementary harvest's
    final-step FR of the same network type (250,000 trajectories), as sp.fr_correlation. NaN where a test's fit failed
    (one class only). ~4 min once; cached in LOGGED_R."""
    if os.path.exists(LOGGED_R):
        f = np.load(LOGGED_R)
        return {k: f[k] for k in ("Trained", "Echo")}
    import glob
    Hc = np.load(sp.HARVEST_CACHE)
    out = {}
    for (name, run), tag in zip(sp.DIAG_RUNS.items(), ("tr_", "ec_")):
        A = [np.load(f) for f in sorted(glob.glob(os.path.join(sp.TD_RUNS, f"{run}_s{sp.DIAG_SEED}", "test_*.npz")))]
        out[name] = implied_r(np.concatenate([a["SII_coef"] for a in A]), np.concatenate([a["acc"][:, -1] for a in A]),
                              Hc[tag + "fr"])
        print(f"  {name}: r implied at {len(out[name]):,} tests ({np.isnan(out[name]).sum()} NaN)", flush=True)
    os.makedirs(os.path.dirname(LOGGED_R), exist_ok=True)
    np.savez(LOGGED_R, **out)
    return out


def diag_runs():
    """(trained, echo): sp.diagnostics_runs(), with the FR correlation at every test (logged_r) when FR_FULL; then the
    observers' own r (fr_joint, fr_naive) are those of the logged FR (symmetric, clipped) on episodes drawn like the
    runs' own tests (fr_correct_fresh: 40,000), not the probe set's forward ones"""
    runs = sp.diagnostics_runs()
    if FR_FRESH:
        F, ept = np.load(FRESH), test_episodes()
        for b, run in zip(runs, ("trained", "echo")):
            it = np.searchsorted(ept, F[f"{run}_raw"])
            assert np.array_equal(ept[it], F[f"{run}_raw"]), run
            b.update(fr_test=it, fr_r=F[f"{run}_fwd"].astype(np.float64), fr_joint=float(F["bayes_fwd"]),
                     fr_naive=float(F["naive_fwd"]))
    elif FR_FULL:
        R = logged_r()
        F = np.load(FRESH)
        for b, name in zip(runs, ("Trained", "Echo")):
            b.update(fr_test=np.arange(len(R[name])), fr_r=R[name], fr_source="logged",
                     fr_joint=float(F["bayes_sym_clip"]), fr_naive=float(F["naive_sym_clip"]))
    return runs


def test_episodes():
    """the raw episode of every test of the two runs (one schedule: every 5th episode, and the last)"""
    import glob
    E = [np.concatenate([np.load(f)["episode"] for f in
                         sorted(glob.glob(os.path.join(sp.TD_RUNS, f"{run}_s{sp.DIAG_SEED}", "test_*.npz")))])
         for run in sp.DIAG_RUNS.values()]
    assert np.array_equal(E[0], E[1])
    return E[0]


def naive_errors(P, ept, ptd):
    """{run: (test index, training episodes, fraction)} at every checkpoint (114): the fraction of Naive Bayes's errors
    at T (Naive wrong, Joint right; the 10,000 probe trajectories) on which the network gives Naive's answer
    (ptd.series 'agree'[:, 2], fig_supp_learning's "Naive errors")"""
    out = {}
    for run, _ in NETS:
        S = ptd.series(run, P)
        it = np.searchsorted(ept, S["episode"])
        assert np.array_equal(ept[it], S["episode"]), run
        out[run] = (it, S["x"], S["agree"][:, 2])
    return out


def draw_grid(ax, z, f_, color):
    """PR(ubar_T), PR(u_T^grid), PR(ubar_T - u_T^grid) of one network and grid (td_tt_griddim, T = 30)"""
    ax.plot(z["x"], z["t30_pr_total"], c=color, ls=GRID_LS["tot"], lw=LW_NEW, zorder=3)
    ax.plot(z["x"], z[f"t30_pr_{f_}"], c=color, ls=GRID_LS["g"], lw=LW_NEW, zorder=3)
    ax.plot(z["x"], z[f"t30_loo_{f_}"], c=color, ls=GRID_LS["rest"], lw=LW_NEW, zorder=3)


def grid_ylim(axes, Z):
    """one tight y range for every grid panel, as the source figure"""
    top = max(Z[r][k].max() for r in Z for k in ("t30_pr_total", "t30_pr_cell", "t30_loo_cell", "t30_pr_envcell",
                                                 "t30_loo_envcell"))
    for a in axes:
        a.set_ylim(0, 1.06 * top)
        a.set_yticks([0, 50, 100])


def grid_key(ax, items=None, title="Grid decomposition\n" + r"$\bar u_T = u_T - \mathbb{E}[u_T]$"):
    """a grid decomposition's key, in the empty top of an ESN panel: one white box over grid lines and the crossing.
    items: (line style, label) pairs; default draw_grid's three"""
    from matplotlib.lines import Line2D
    items = items or list(zip((GRID_LS[k] for k in ("tot", "g", "rest")),
                              (r"PR$(\bar u_T)$", r"PR$(u_T^{\rm grid})$", r"PR$(\bar u_T - u_T^{\rm grid})$")))
    leg = ax.legend(handles=[Line2D([], [], color="0.3", ls=ls, lw=LW_NEW) for ls, _ in items],
                    labels=[lab for _, lab in items], title=title, title_fontsize=11.5,
                    alignment="left", loc="upper left", bbox_to_anchor=(.03, .985), frameon=True, fontsize=FFS,
                    handlelength=2.4, handletextpad=.5, labelspacing=.55, borderaxespad=0, borderpad=.4)
    leg.get_frame().set(facecolor="white", edgecolor="none", alpha=.92)
    leg.set_zorder(6)


def log_time(ax, cross, fts, color="0.6"):
    """log training episodes (the initial network at 10 = "0") over all of training; the crossing of Naive Bayes dotted"""
    fts.xaxis(ax, label=False)
    ax.set_xlim(fts.XMIN, 4.6e4)
    ax.set_xlabel("Training episode")
    ax.axvline(cross, color=color, lw=.9, ls=(0, (1, 1.5)), zorder=5 if color == "w" else 1)


def style(axes, grid=True):
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        if grid:
            a.grid(True, ls="--", alpha=0.3)


def place_columns(fig, A, pad, fixed=None, right=.995):
    """one left edge and width per column of A (sp._align_columns): each gutter as wide as its column's widest y axis
    needs plus pad, or fixed[j] exactly (figure fractions)"""
    fixed = fixed or {}
    fig.canvas.draw()
    rr, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()

    def need(a):                                                       # None: no tick labels
        b = a.yaxis.get_tightbbox(rr)
        return 0 if b is None else a.get_position().x0 - b.transformed(inv).x0
    gut = [fixed[j] if j in fixed else max(need(a) for a in col) + pad for j, col in enumerate(A.T)]
    w = (right - sum(gut)) / A.shape[1]
    x = 0
    for g, col in zip(gut, A.T):
        x += g
        for a in col:
            p_ = a.get_position()
            a.set_position([x, p_.y0, w, p_.height])
        x += w


def panel_letters(fig, A, gap=.012):
    """"(a)", "(b)", ... in reading order, each centred under its panel, one height per row just below the row's
    lowest x label (gap in figure fractions)"""
    fig.canvas.draw()
    rr, inv = fig.canvas.get_renderer(), fig.transFigure.inverted()
    k = 0
    for row in A:
        y = min(a.get_tightbbox(rr).transformed(inv).y0 for a in row) - gap
        for a in row:
            p_ = a.get_position()
            fig.text((p_.x0 + p_.x1) / 2, y, f"({chr(97 + k)})", ha="center", va="top", fontsize=LETTER_FS,
                     family="DejaVu Sans")
            k += 1


def center_pair(a, b, form):
    """a's title (and x label, if b has none) centred over a and b, the formula under the title"""
    pa, pb = a.get_position(), b.get_position()
    xc = ((pa.x0 + pb.x1) / 2 - pa.x0) / pa.width                    # in a's axes fraction
    a.title.set_x(xc)
    if not b.get_xlabel():
        a.xaxis.label.set_x(xc)
    put_formula(a, form, xc)


def put_formula(a, form, x=.5):
    """the formula under a's title (x in a's axes fraction)"""
    a.annotate(form, xy=(x, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points", fontsize=FFS,
               ha="center", va="baseline", color="0.15")


def check_text(fig):
    """warn when a text leaves the figure or two texts overlap"""
    fig.canvas.draw()
    rr, fb = fig.canvas.get_renderer(), fig.bbox
    texts = [t for t in fig.texts + [t for ax in fig.axes for t in ax.texts]
             + [t for ax in fig.axes for t in (ax.title, ax.xaxis.label, ax.yaxis.label)] if t.get_text()]
    ext = [t.get_window_extent(rr) for t in texts]
    for t, e in zip(texts, ext):
        if e.x0 < fb.x0 - .5 or e.x1 > fb.x1 + .5 or e.y0 < fb.y0 - .5 or e.y1 > fb.y1 + .5:
            print(f"  WARNING: text {t.get_text()!r} leaves the figure")
    for k in range(len(texts)):
        for m in range(k + 1, len(texts)):
            if ext[k].overlaps(ext[m]):
                print(f"  WARNING: {texts[k].get_text()!r} overlaps {texts[m].get_text()!r}")


def figure_grid():
    """2 x 6 (John 11-12: "create a version with these other panels from the second image in it as a 2 x 5", then
    "temporarily use the previously calculated FR so we can get the full data (not the downsampled). And lets make the
    new panels have thicker lines. Also lets add our naive errors plot as a second column (early and full trial)").
    Columns 1, 3, 4: diagnostics_panel_early.svg's six panels (sp.diagnostics_runs: 1 run per network, 10,001 tests of
    8,000 trajectories), the FR correlation at every test from the logged fits (logged_r, FR_FULL: symmetric FR)
    instead of the forward-FR replay at 114 checkpoints, without the observers' references (forward FR only).
    Column 2: the fraction of Naive Bayes's errors the network repeats (naive_errors: the 10,000 probe trajectories at
    the 114 checkpoints, each at its test), the first 800 tests (49 checkpoints) and all. Columns 5-6:
    supp_learning_behaviour's grid decomposition (td_tt_griddim at T = 30, the same 10,000 probe trajectories = 50
    interaction sets x 100 cells x 2, 13 checkpoints): top the context-invariant grid, bottom the context-dependent one;
    Trained RNN left, ESN right; one y range; grey dotted = the Trained RNN first exceeds Naive Bayes (880 training
    episodes); their key in the top ESN panel, above its curves. The 11-12 panels drawn thicker (LW_NEW). x as the
    sources: testing episodes in columns 1-4 (a test every 4 training episodes), training episodes (log) in 5-6.
    No note in the figure (counts here). 18.7 x 8 in: at the 5.5 in text width, scale .29.
    -> figure_bin/diagnostics_panel_early_grid.svg"""
    import td_tt_griddim as tg
    import fig_td_summary as fts
    import plot_training_dynamics as ptd
    import fig_entanglement_driver as fd
    W, H, PG, CP = 18.7, 8.0, .10, .20                # inches: figure, Trained-ESN gap, column pad
    TFS = sp.PLOT_CFG["diag_title_fs"]
    diag_tr, diag_ec = diag_runs()
    P = ptd.probes()
    cross, ept = fd.passes_naive(P), test_episodes()
    NE = naive_errors(P, ept, ptd)
    Z = {r: np.load(os.path.join(tg.OUT, f"{r}.npz")) for r, _ in NETS}
    fig, A = plt.subplots(2, 6, figsize=(W, H), squeeze=False)
    D = A[:, [0, 2, 3]]                                                # diagnostics_panel_early's 2 x 3
    if FR_FULL:
        D[0, 1]._fr_refs = True                                        # the observers' own r: forward FR only
    sp.plot_diagnostics_early(D, diag_tr, "Trained"); sp.plot_diagnostics_early(D, diag_ec, "Echo")
    sp.mark_bounds((diag_tr, diag_ec), y_axes=(D[0, 0], D[1, 0]), x_axes=(D[1, 1],))
    lim = sp.PLOT_CFG["early_epochs"]
    for run, name in NETS:                                             # column 2: naive errors, at their tests
        it, _, y = NE[run]
        A[0, 1].plot(it[it < lim] + 1, y[it < lim], c=sp.AGENT_COLORS[name], lw=LW_NEW, zorder=3)
        A[1, 1].plot(it + 1, y, c=sp.AGENT_COLORS[name], lw=LW_NEW, zorder=3)
    rows = [(["Early final step acc.", "Early naive errors", "FR-accuracy correlation", "PR(recurrent)"],
             [ACC_LABEL, NAIVE_Y, sp.CORR_LABEL, PR_LABEL], ["Testing episode"] * 4),
            (["Full final step acc.", "Full naive errors", "Corr. vs. avg. accuracy", "Corr. vs. PR(recurrent)"],
             [ACC_LABEL, NAIVE_Y, sp.CORR_LABEL, sp.CORR_LABEL],
             ["Testing episode", "Testing episode", "Test accuracy", PR_LABEL])]
    for r, (titles, ylabels, xlabels) in enumerate(rows):
        for i, ax in enumerate(A[r, :4]):
            ax.set_title(titles[i], pad=PAD, fontsize=TFS)
            ax.set_ylabel(ylabels[i]); ax.set_xlabel(xlabels[i])
    A[0, 1].set_xlim(A[0, 0].get_xlim()); A[1, 1].set_xlim(A[1, 0].get_xlim())
    sp._match_ylim([A[0, 0], A[1, 0]])
    sp._match_ylim([A[0, 1], A[1, 1]])
    sp._match_ylim([A[0, 2], A[1, 2], A[1, 3]])
    A[1, 0].legend(loc="lower right", frameon=False, ncols=2)
    for i, (f_, name, _) in enumerate(GRID_PAIRS):                     # columns 5-6: the grid decomposition
        for j, (run, net) in enumerate(NETS):
            draw_grid(A[i, 4 + j], Z[run], f_, sp.AGENT_COLORS[net])
            log_time(A[i, 4 + j], cross, fts)
        A[i, 4].set_title(name, pad=PAD, fontsize=TFS)                 # centred over the pair below
        A[i, 4].set_ylabel("Participation ratio")
        A[i, 5].set_xlabel("")
    grid_ylim(A[:, 4:].flat, Z)
    for a in A[:, 5]:
        a.tick_params(labelleft=False)
    grid_key(A[0, 5])
    style(A.flat)
    fig.tight_layout(w_pad=0.5, h_pad=4.5)                             # the rows; the columns below
    place_columns(fig, A, CP / W, fixed={5: PG / W})
    for i, (_, _, form) in enumerate(GRID_PAIRS):
        center_pair(A[i, 4], A[i, 5], form)
    check_text(fig)
    path = sp.fig_path("diagnostics_panel_early_grid.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


def figure_log():
    """2 x 3 (John 11-12: "for consistency lets try another version where all plots through time use the log scaled
    training episodes and we show everything through full training (no early training panels)", first a 3 x 4, then
    without the gradient-FR maps and the 'Corr. vs.' panels, then "can we get the corr vs accuracy panel back. and
    let's remove the context dependent grid curve and the all curve, and combine the trained and ESN grid panels", then
    "maybe instead of shared PR we put the default PR to show the default increases when removing the shared", then
    "can you please make the panels wider? ... are we able to have them all with full data?", then "instead of "all"
    and "all minus" can you please write PR(...)", "do we still need the whole bar(u) thing?": no, see GRID_SH, then
    "can we reorder to [acc, naive error, corr vs eps] [pr recur, remove grid, corr vs acc] also we can move the
    definition of the shared grid in the title from the 2nd line to the 1st so it's just 1 line"). The data of
    figure_grid: final step accuracy, naive errors, the FR-accuracy correlation (FR_FULL: every test, symmetric FR);
    PR(recurrent), removing the shared grid, the FR-accuracy correlation against test accuracy. Time on log training episodes over all of training (a test at 4 i training episodes; the
    initial network at 10 = "0"; the tests at 4 and 8 training episodes left out); grey dotted = the Trained RNN first
    exceeds Naive Bayes (880 training episodes). Removing the shared grid: td_tt_griddim at T = 30 (10,000 probe
    trajectories = 50 environments x 100 cells x 2) at all 114 saved checkpoints (run_full; the 13 of run() if absent),
    as the naive errors; the per-test logs keep batch totals only, so neither can have a point per test. PR of all, u_T
    (solid), and of all minus the shared grid E[u_T | r] (dashed, on top, white-rimmed: the ESN's two lines coincide;
    styles swapped 11-13, John: "let's make the grid removed line the dashed one and the other solid").
    At 40k training episodes: Trained 52.7 -> 96.8 (the shared grid: PR 5.7, 25% of the variance), ESN 18.0 -> 17.9
    (PR 2.4, 2%). 14.8 x 8.4 in (wider panels, John 11-13; 7.8 -> 8.4 for the panel letters "(a)"-"(f)" centred under
    the panels in sans-serif, John 11-13: "copy my preferred labeling style"): at the 5.5 in text width, scale .37.
    (c) marks the observers' own r on its right edge (John 11-13: "could you please add a little mark to panel c where
    the joint and naive correlations sit?"), for (c)'s FR and episodes: FR_FULL, the logged symmetric clipped FR on
    40,000 episodes drawn like the runs' tests (fr_correct_fresh: Joint +.001, Naive -.432; the y axes of (c) and (f)
    extend to the Naive mark); else the probe set's forward replay (Joint -.061, Naive -.322).
    Caption (11-13, John: "update the caption from this to the new figure", from the old diagnostics_panel_early
    caption; 1.5x its length; checked against the data: 10,001 tests of 8,000 (every 5th episode), crossing 880, naive
    errors Trained .11 -> .28 (640) -> .03 and ESN -> .49, the grid at 39,996 as above (shared grid 24.7% / 2.2% of the
    variance); (c, f) are still the symmetric FR of the logs (FR_FULL), which the caption does not say):
      \\caption{\\textbf{Managing Factorization Regret involves high-dimensional RNN dynamics ($C=2$).} Panels (a--e)
      span all of training (log axis), with a testing episode of 8,000 trajectories every 4 training episodes; (b) and
      (e) use 10,000 testing trajectories (50 interaction sets $\\times$ 100 realizations $\\times$ 2) at 114
      checkpoints. Dashed lines: Joint and Naive Bayes accuracy; dotted lines: the Trained RNN first exceeds Naive Bayes
      (880 training episodes). \\textbf{(a)} Final step testing accuracy, averaged over the batch. \\textbf{(b)} Fraction
      of Naive Bayes's errors (Joint Bayes correct) that the network repeats: the Trained RNN sheds them, the ESN repeats
      half. \\textbf{(c)} Correlation between Factorization Regret and final step accuracy across the batch; negative
      early in training, it returns to zero only for the Trained RNN. \\textbf{(d)} PR(recurrent): difference in
      Participation Ratio (PR) between the recurrent and read-in activity. \\textbf{(e)} PR of the final RNN output $u_T$
      (solid) and of $u_T$ without the shared grid $\\mathbb{E}[u_T|\\mathbf{r}]$, its mean over interaction sets given
      the realization $\\mathbf{r}$ (dashed). The shared grid is low-dimensional (PR 6) yet holds 25\\% of the Trained
      RNN's variance (ESN: 2\\%); removing it raises the final PR from 53 to 97. \\textbf{(f)} Correlation versus average
      accuracy over the batch.}
    -> figure_bin/diagnostics_panel_log.svg"""
    import td_tt_griddim as tg
    import fig_td_summary as fts
    import plot_training_dynamics as ptd
    import fig_entanglement_driver as fd
    import matplotlib.patheffects as pe
    W, H, CP = 14.8, 8.4, .20                         # inches: figure (taller for the panel letters), column pad
    TFS = sp.PLOT_CFG["diag_title_fs"]
    diag = dict(zip(("Trained", "Echo"), diag_runs()))
    P = ptd.probes()
    cross, ept = fd.passes_naive(P), test_episodes()
    NE = naive_errors(P, ept, ptd)
    gd = tg.OUT_FULL if all(os.path.exists(os.path.join(tg.OUT_FULL, f"{r}.npz")) for r, _ in NETS) else tg.OUT
    Z = {r: np.load(os.path.join(gd, f"{r}.npz")) for r, _ in NETS}
    print(f"  grid: {len(Z['trained']['x'])} checkpoints ({gd})")
    xt = (ept - ept // 5).astype(float)                               # training episodes at each test
    xt[xt == 0] = fts.X0
    keep = xt >= fts.X0
    fig, A = plt.subplots(2, 3, figsize=(W, H), squeeze=False)
    acc, nai, cor, prr, grd, cva = A.flat                                    # John 11-13's order
    for run, name in NETS:
        ag, c = diag[name], sp.AGENT_COLORS[name]
        acc.plot(xt[keep], ag.test_acc_through_training[keep, -1], c=c, label=name, zorder=3)
        nai.plot(NE[run][1], NE[run][2], c=c, lw=LW_NEW, zorder=3)
        k = keep[ag.fr_test]
        cor.plot(xt[ag.fr_test][k], ag.fr_r[k], c=c, zorder=3, **sp._ckpt_kw(ag))
        prr.plot(xt[keep], sp.pr_gain(ag)[keep], c=c, zorder=3)
        for key, ls, _ in GRID_SH:
            top = key == "t30_loo_cell"                                  # dashed on top: visible on solid
            grd.plot(Z[run]["x"], Z[run][key], c=c, ls=ls, lw=LW_NEW, zorder=4 if top else 3,
                     path_effects=[pe.withStroke(linewidth=LW_NEW + 1.8, foreground="w")] if top else None)
        sp._plot_r_vs_acc(cva, ag, name)
    sp.mark_bounds(tuple(diag.values()), y_axes=(acc,), x_axes=(cva,))
    for a in (cor, prr, cva):
        a.axhline(0, c="k", alpha=0.5)
    sp._match_ylim([cor, cva])
    obs = {"Joint": diag["Trained"].fr_joint, "Naive": diag["Trained"].fr_naive}       # (c): the observers' own r,
    lo, hi = cor.get_ylim()                                                            # marked on its right edge
    if min(obs.values()) < lo + .015:
        for a in (cor, cva):
            a.set_ylim(min(obs.values()) - .025, hi)
    for lab, v in obs.items():
        cor.plot([1.015], [v], marker="<", ms=8, ls="none", color=sp.BOUND_COLORS[lab], clip_on=False, zorder=6,
                 transform=cor.get_yaxis_transform())
        cor.annotate(lab, (1.015, v), xycoords=cor.get_yaxis_transform(), xytext=(7, 0), textcoords="offset points",
                     ha="left", va="center", color=sp.BOUND_COLORS[lab])
    grd.set_ylim(0, 1.06 * max(Z[r][k].max() for r in Z for k, _, _ in GRID_SH))
    grd.set_yticks([0, 50, 100])
    for a in (acc, nai, cor, prr, grd):
        log_time(a, cross, fts)
    for a, t, yl in ((acc, "Final step accuracy", ACC_LABEL), (nai, "Naive errors", NAIVE_Y),
                     (cor, "FR-accuracy correlation", sp.CORR_LABEL), (prr, "PR(recurrent)", PR_LABEL),
                     (grd, GRID_SH_TITLE, "Participation ratio"), (cva, "Corr. vs. avg. accuracy", sp.CORR_LABEL)):
        a.set_title(t, pad=15, fontsize=TFS)                                # one line each: no formula under
        a.set_ylabel(yl)
    cva.set_xlabel("Test accuracy")
    grid_key(grd, [(ls, lab) for _, ls, lab in GRID_SH], title=None)
    acc.legend(loc="upper left", bbox_to_anchor=(0, .88), frameon=False)    # on log time the curves fill the
                                                                       # lower right: under Joint's line, left of 880
    style(A.flat)
    fig.tight_layout(w_pad=0.5, h_pad=5.5, rect=(0, .05, 1, 1))          # room under each row for its letters
    place_columns(fig, A, CP / W, right=.955)                                 # room for (c)'s observer marks
    panel_letters(fig, A)
    check_text(fig)
    path = sp.fig_path("diagnostics_panel_log_forward.svg" if FR_FRESH else "diagnostics_panel_log.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    if "--harvest" in sys.argv:
        i = sys.argv.index("--harvest"); harvest(*(int(a) for a in sys.argv[i + 1:i + 4]))
    elif "--grid" in sys.argv or "--log" in sys.argv:
        FR_FRESH = "--forward" in sys.argv                               # --log --forward: the forward-FR version
        with plt.rc_context({"mathtext.fontset": "cm"}):                 # the formulas in Computer Modern, as the text
            figure_grid() if "--grid" in sys.argv else figure_log()
    else:
        figures()
