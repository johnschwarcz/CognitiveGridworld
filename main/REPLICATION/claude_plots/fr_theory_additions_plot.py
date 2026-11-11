"""Figures and printed values of fr_theory_additions.py (see its docstring).  Style, colours and the one convention of the theory
figures come from fr_theory.py: lines = analytical, markers = simulation (+- 2 s.e.), drawn at print size (9 in wide)."""
import os
import numpy as np
import fr_theory as F
import fr_theory_additions as A

DRAW_RATIOS = (0.5, 1.1, 2.0, 27.0)                 # fig_fr_threshold a: trapped fractions drawn, as multiples of q*
STEP_RATIOS = (0.5, 0.9, 1.0, 1.1, 1.5)             # fig_fr_candidates A
WINDOW = 3                                          # Gridworld threshold: grid steps per window (see window_threshold)


def ratio_label(r, first=False):
    s = {0.5: r"\frac{1}{2}", 1.0: ""}.get(r, f"{r:g}" + r"\,")
    return (r"$q=" if first else "$") + s + r"q^*$"


def window_threshold(G, step, rng=None, B=300):
    """The threshold between grid steps `step` apart, pooled over interaction sets, with bootstrap s.e. over sets:
    q*_i = (U_i - U_j)^+ / ((U_i - U_j)^+ + (W_j - W_i)), j = i + step.  The mixture (1 - q) U + q W is lower at step t_j than at
    t_i exactly when q < q*_i, and q*_i never exceeds the largest one-step threshold between t_i and t_j."""
    idx = np.arange(0, len(G["TS"]) - step, step)
    def stat(sel):
        S, c = G["S"][sel].sum(0), G["cnt"][sel].sum(0); U, W = S[0] / c[0], S[1] / c[1]; dU, dW = U[idx] - U[idx + step], W[idx + step] - W[idx]
        return np.maximum(dU, 0) / (np.maximum(dU, 0) + dW), dU, dW
    nz = len(G["q"]); rng = rng or np.random.default_rng(8); qs, dU, dW = stat(np.arange(nz)); bs = [stat(rng.integers(0, nz, nz)) for _ in range(B)]
    return dict(t=np.sqrt(G["TS"][idx] * G["TS"][idx + step]), t0=G["TS"][idx], t1=G["TS"][idx + step], qs=qs, se=np.std([b[0] for b in bs], axis=0),
                dU=dU, dU_se=np.std([b[1] for b in bs], axis=0), dW=dW, q_star=float(qs.max()), boot_max=np.array([b[0].max() for b in bs]))


def numbers(cfg, D):
    th, sm = D["toy_theory"], D["toy_sim"]; b = np.array(th["both"]); b = b[b[:, 1] > 0]       # the populations drawn all have q > 0
    print(f"\nTOY  q* = {th['q_star']:.6f}, attained at step t* = {th['t_star']}; horizon T = {th['T']}")
    print(f"  saddlepoint vs exact FFT at N = {cfg['N_FFT_MAX']:g}: {len(b)} (step, q > 0) cases, largest difference {np.abs(b[:, 3] - b[:, 2]).max():.1e}"
          "  (q = 0 is not compared: late in the trap-free table the regret is below 1e-6 nats and neither method resolves its increments)")
    for r in sm["ratios"]:
        j = sm["ratios"].index(r); row = []; Nb, blo, bhi = A.toy_bounds(D, r)
        for N, d in zip(sm["Ns"], sm["dip"][j]):
            k = int(np.argmin(np.abs(Nb - N))); assert np.isclose(Nb[k], N), "bounds missing at a simulated N"
            lo, hi = blo[k], min(1.0, bhi[k]); se = np.sqrt(max(d * (1 - d), 1 / sm["paths"]) / sm["paths"])
            row.append(f"1e{np.log10(N):.1f}: {d:.2f} in [{lo:.2f}, {hi:.2f}]" + ("" if lo - 2 * se <= d <= hi + 2 * se else " !!"))
        print(f"  q = {r:g} q*: simulated P(dip) and its bounds  " + "; ".join(row))
    for lam, G in D["grid"].items():
        C = A.components(G); print(f"\nGRIDWORLD lambda = {lam:g}: q = {C['q']:.3f} +- {C['q_se']:.3f} (tables); untrapped component peaks at {C['U_peak']:.3f} nats, t = {C['t_peak']}; "
                                   f"U at t = {G['TS'][-1]}: {C['U'][-1]:.3f}; trapped gain per step at the end {(C['W'][-1] - C['W'][-2]) / (G['TS'][-1] - G['TS'][-2]):.4f}")
        for step in (1, 2, 3, 5):
            Wn = window_threshold(G, step); i = int(Wn["qs"].argmax()); lo, hi = np.quantile(Wn["boot_max"], [0.025, 0.975])
            print(f"   windows of {step} grid steps (x{(G['TS'][-1] / G['TS'][-1 - step]):.2f} in t): q* = {Wn['q_star']:.4f} between t = {Wn['t0'][i]} and {Wn['t1'][i]} "
                  f"(s.e. {Wn['se'][i]:.4f}; bootstrap of the maximum {lo:.4f}-{hi:.4f}); q / q* = {C['q'] / Wn['q_star']:.0f}; "
                  f"untrapped loss per step there {Wn['dU'][i] / (Wn['t1'][i] - Wn['t0'][i]):.5f}, trapped gain {Wn['dW'][i] / (Wn['t1'][i] - Wn['t0'][i]):.4f}")
        inc = np.diff(C["U"]); first = np.flatnonzero(inc < 0)
        print(f"   untrapped component rises until t = {G['TS'][first[0]] if len(first) else 'the end'} (no decrease of the expected curve before, whatever q)")
    print("\nASSUMPTIONS in sampled interaction sets (the tex: \"they held in every interaction set we sampled\")")
    for lam, v in A.check_assumptions().items():
        print(f"   lambda = {lam:g}, {v['n']} sets: (A2) smallest difference between the observation laws of joint realizations with different goals {v['a2']:.1e}; "
              f"(A3) smallest |kappa(r, a) - kappa(r, r_g)| {v['tie']:.1e}, smallest gap between the two best Naive models {v['gap']:.1e}: " + ("hold" if v["ok"] else "VIOLATED"))
    print("\nWHY TRAPS ARE COMMON (the tex: \"the contextual variable moves each observable's probability more than the goal separates the Naive models\")")
    for lam, v in A.trap_origin().items():
        print(f"   lambda = {lam:g}, {v['n']} sets: s.d. of an observable's probability across the contextual variable's realizations {v['sd_context']:.3f}; "
              f"s.d. of its Naive marginal across actions {v['sd_naive']:.3f}; mismatch to the optimal action's Naive model {v['kap_true']:.3f} nats, to the nearest other action's {v['kap_best_other']:.3f}")
    P = D["pool"]; print(f"\nGRIDWORLD pool: {P['n_paths']:,} paths of {cfg['T_POOL']} steps, trapped fraction {P['q_nat']:.3f}; {P['groups']} averages per point")
    for q, v in P["curves"].items(): print(f"   q = {q if q is not None else 'Gridworld'}: P(dip) at N = {list(P['Ns'])}: {np.round(v, 3)}")
    S = D["steps"]; print(f"\nGRIDWORLD dips against the step ({S['n_sets']} sets x {S['n_ep']} episodes per step): asymptotic law P(X-bar_N < 0) = {np.round(S['law_lo'], 4)} for N = {list(S['Ns'])}")
    for k, t in enumerate(S["TS"]): print(f"   t = {t:6d}: " + "  ".join(f"N {N}: {S['sim'][i, k]:.3f} ({(S['sim'][i, k] - S['law_lo'][i]) / S['se'][i, k]:+.1f} s.e. from the law)" for i, N in enumerate(S["Ns"])))


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import logging; logging.getLogger("fontTools").setLevel(logging.ERROR)
    plt.rcParams.update(F.RC); return plt


TEXT_WIDTH = 5.5                                    # inches: the manuscript's text width (an upright figure at \linewidth is printed at this size)


def _row(plt, n, lm=0.50, gx=0.62, rm=0.06, W=None):
    """One row of n panels at print size: the panel height of the two-row theory figures.  W: figure width in inches (default: the
    9 in of the rotated theory figures)."""
    W = W or F.FIGSIZE[0]; TOP, BOT = 0.35, 0.39; ph = (F.FIGSIZE[1] - 2 * 0.35 - 0.31 - 0.39 - 0.05) / 2; H = TOP + ph + BOT
    pw = (W - lm - (n - 1) * gx - rm) / n; fig = plt.figure(figsize=(W, H))
    axs = [fig.add_axes([(lm + j * (pw + gx)) / W, BOT / H, pw / W, ph / H]) for j in range(n)]
    return fig, axs, (W, H, TOP, BOT, ph, lm, rm, pw, gx)


def _helpers():
    from matplotlib.lines import Line2D
    LW, MS = F.LW, F.MS
    def theory(ax, x, y, color, lw=LW, **kw): return ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=3, **kw)
    def simulated(ax, x, y, yerr, color, filled=True, ms=MS, clip=False):
        if yerr is not None: ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.7, capsize=0, zorder=4)
        return ax.plot(x, y, ls="none", marker="o", ms=ms, mfc=color if filled else "white", mec="white" if filled else color, mew=0.5 if filled else 0.8,
                       zorder=5, clip_on=clip)
    def h_both(color, label): return Line2D([], [], color=color, lw=LW, marker="o", ms=MS, mfc=color, mec="white", mew=0.5, label=label)
    def h_line(color, label, **kw): return Line2D([], [], color=color, lw=kw.pop("lw", LW), label=label, **kw)
    def h_dot(color, label, filled=True):
        return Line2D([], [], ls="none", marker="o", ms=MS, mfc=color if filled else "white", mec="white" if filled else color, mew=0.5 if filled else 0.8, label=label)
    return theory, simulated, h_both, h_line, h_dot, Line2D


def _key(fig, geo, h_dot, Line2D, sim_only=False):
    W, H, TOP, BOT, ph, lm, rm, pw, gx = geo
    fig.legend(handles=[Line2D([], [], color=F.INK, lw=F.LW, label="Theory"), h_dot(F.INK, r"Simulation ($\pm$ 2 SEM)")], loc="upper right",
               bbox_to_anchor=(1 - rm / W, (BOT + ph + TOP + 0.02) / H), ncol=2, fontsize=F.FS["label"], handlelength=1.8, columnspacing=1.4, borderpad=0.1, frameon=False)


def _save(plt, fig, name, tag):
    from PIL import Image
    stem = os.path.join(A.OUT, f"{name}{tag}"); fig.savefig(stem + ".pdf"); fig.savefig(stem + ".png", dpi=150)
    if not tag:
        from main.utils import fig_path
        fig.savefig(fig_path(f"{name}.png"), dpi=150); fig.savefig(fig_path(f"{name}.pdf"))
    insp = os.path.join(A.OUT, "inspect"); os.makedirs(insp, exist_ok=True); big = os.path.join(insp, f"{name}{tag}_300dpi.png"); fig.savefig(big, dpi=300)
    im = Image.open(big); W_, H_ = im.size
    im.crop((0, 0, int(0.55 * W_), H_)).save(os.path.join(insp, f"{name}{tag}_left.png")); im.crop((int(0.45 * W_), 0, W_, H_)).save(os.path.join(insp, f"{name}{tag}_right.png"))
    plt.close(fig); print(f"saved {stem}.png (+ pdf" + ("" if tag else ", copies in figure_bin") + f"), crops in {insp}/")


def se_p(p, n): return np.sqrt(np.maximum(p * (1 - p), 1.0 / n) / n)


def panel_toy_dips(ax, D, tools, ratios=DRAW_RATIOS):
    """P(an N-episode average decreases somewhere in the T steps), toy: bounds from the exact law (lines) and simulation (markers)."""
    theory, simulated, h_both, h_line, h_dot, Line2D = tools; th, sm = D["toy_theory"], D["toy_sim"]
    cols = [F.BLUES5[0], F.BLUES5[2], F.BLUES5[3], F.BLUES5[4]][:len(ratios)]
    for r, col in zip(ratios, cols):
        j = sm["ratios"].index(r); Nb, blo, bhi = A.toy_bounds(D, r)
        theory(ax, Nb, blo if r < 1 else np.minimum(1.0, bhi), col)                                   # lower bound below q*, upper bound above
        simulated(ax, sm["Ns"], sm["dip"][j], 2 * se_p(sm["dip"][j], sm["paths"]), col)
    ax.set_xscale("log"); ax.set_xlim(6e2, 1.7e8); ax.set_ylim(0, 1.62); ax.set_yticks([0, 0.5, 1.0]); ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    ax.set_xticklabels([r"$10^3$", "", r"$10^5$", "", r"$10^7$", ""])
    ax.legend(handles=[h_both(col, ratio_label(r, True)) for r, col in zip(ratios, cols)], loc="upper left", ncol=2,
              columnspacing=0.7, handlelength=1.1, handletextpad=0.35, borderaxespad=0.1, labelspacing=0.2)
    ax.set_xlabel(r"episodes averaged, $N$"); ax.set_ylabel(r"P(the average dips)")


def panel_grid_threshold(ax, D, cfg, tools):
    """The Gridworld's threshold q*_t (simulation) against its trapped fraction q (tables), log scale."""
    theory, simulated, h_both, h_line, h_dot, Line2D = tools
    for lam, col in zip(sorted(D["grid"]), F.VIOLETS3 if len(D["grid"]) == 3 else [F.VIOLETS3[1]]):
        G = D["grid"][lam]; Wn = window_threshold(G, WINDOW); ok = Wn["qs"] > 2 * Wn["se"]                 # thresholds distinguishable from 0
        ax.axhline(G["q"].mean(), color=col, lw=F.LW, zorder=3)
        ok &= Wn["qs"] > 1e-4                                                                           # inside the panel
        simulated(ax, Wn["t"][ok], Wn["qs"][ok], 2 * Wn["se"][ok], col, ms=F.MS * 0.9, clip=True)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, cfg["GRID_TMAX"]); ax.set_ylim(1e-4, 8.0)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1]); ax.set_xlabel("step $t$"); ax.set_ylabel("trapped fraction")
    lams = sorted(D["grid"]); cols = F.VIOLETS3 if len(lams) == 3 else [F.VIOLETS3[1]]
    ax.legend(handles=[h_both(c, (r"$\lambda=" if k == 0 else "$") + f"{l:g}$") for k, (l, c) in enumerate(zip(lams, cols))], loc="upper left", ncol=3,
              columnspacing=0.8, handlelength=1.3, borderaxespad=0.1)
    ax.text(0.04, 0.70, r"$q$ of the Gridworld", transform=ax.transAxes, ha="left", va="top", fontsize=F.FS["key"], color=F.INK)
    ax.text(0.04, 0.40, r"threshold", transform=ax.transAxes, ha="left", va="bottom", fontsize=F.FS["key"], color=F.INK)


def panel_grid_dips(ax, D, cfg, tools):
    theory, simulated, h_both, h_line, h_dot, Line2D = tools; P = D["pool"]; keys = list(P["curves"]); cols = [F.BLUES5[0], F.BLUES5[2], F.BLUES5[4]][:len(keys)]
    for q, col in zip(keys, cols): simulated(ax, P["Ns"], P["curves"][q], 2 * se_p(P["curves"][q], P["groups"]), col)
    lab = lambda q, k: (r"$q=" if k == 0 else "$") + (f"{q:g}$" if q is not None else f"{P['q_nat']:.2f}$ (Gridworld)")
    ax.set_xscale("log"); ax.set_xlim(6, P["Ns"][-1] * 1.7); ax.set_ylim(0, 1.62); ax.set_yticks([0, 0.5, 1.0])
    ax.legend(handles=[h_dot(col, lab(q, k)) for k, (q, col) in enumerate(zip(keys, cols))], loc="upper left", ncol=2, columnspacing=0.7, handletextpad=0.2, borderaxespad=0.1,
              labelspacing=0.2)
    ax.set_xlabel(r"episodes averaged, $N$"); ax.set_ylabel(r"P(the average dips)")


def panel_grid_mix(ax, D, cfg, tools):
    """Expected curve of populations with trapped fraction q: (1 - q) U_t + q W_t from the simulated Gridworld components."""
    theory, simulated, h_both, h_line, h_dot, Line2D = tools; G = D["grid"][cfg["LAM"]]; C = A.components(G); TS = G["TS"]
    qstar = window_threshold(G, WINDOW)["q_star"]; qs = list(cfg["MIX_QS"][:3]) + [qstar] + list(cfg["MIX_QS"][3:]) + [C["q"]]
    cols = [F.BLUES5[0], F.BLUES5[1], F.BLUES5[2], F.INK, F.BLUES5[4], F.INK]; hnd = []
    for k, (q, col) in enumerate(zip(qs, cols)):
        m = (1 - q) * C["U"] + q * C["W"]; se = np.hypot((1 - q) * C["U_se"], q * C["W_se"]); star, own = k == 3, k == len(qs) - 1
        simulated(ax, TS, m, 2 * se, col, filled=not star, ms=F.MS * (0.62 if not own else 0.8), clip=True)
        hnd.append(h_dot(col, (r"$q^*$" if star else (r"Gridworld" if own else (r"$q=" if k == 0 else "$") + f"{q:g}$")), filled=not star))
    ax.set_xscale("log"); ax.set_xlim(1, cfg["GRID_TMAX"]); ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.5, 1.0])
    ax.legend(handles=hnd, loc="lower right", ncol=2, columnspacing=0.5, handletextpad=0.15, borderaxespad=0.15, labelspacing=0.15)
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\mathbb{E}[\mathrm{FR}_t]$ (nats)")


def panel_toy_step(ax, D, tools, ratios=STEP_RATIOS):
    theory, simulated, h_both, h_line, h_dot, Line2D = tools; th = D["toy_theory"]
    cols = {0.5: F.BLUES5[0], 0.9: F.BLUES5[1], 1.0: F.INK, 1.1: F.BLUES5[3], 1.5: F.BLUES5[4]}; hnd = []
    for k, r in enumerate(ratios):
        kw = dict(ls=(0, (3, 1.6)), lw=F.LW * 1.1) if r == 1.0 else {}
        theory(ax, th["Ns"], th["p_star"][th["ratios"].index(r)], cols[r], **kw); hnd.append(h_line(cols[r], ratio_label(r, k == 0), **kw))
    ax.set_xscale("log"); ax.set_xlim(1e3, 1e8); ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.5, 1.0])
    ax.legend(handles=hnd, loc="upper left", ncol=3, columnspacing=0.7, handlelength=1.2, handletextpad=0.35, borderaxespad=0.1, labelspacing=0.15)
    ax.set_xlabel(r"episodes averaged, $N$"); ax.set_ylabel(r"P(a dip at step $t^*$)")


def panel_steps(ax, D, tools):
    theory, simulated, h_both, h_line, h_dot, Line2D = tools; S = D["steps"]
    for i, (N, col) in enumerate(zip(S["Ns"], F.ORANGES3)):
        ax.axhline(0.5 * (S["law_lo"][i] + S["law_hi"][i]), color=col, lw=F.LW, zorder=3); simulated(ax, S["TS"], S["sim"][i], 2 * S["se"][i], col, ms=F.MS * 0.9)
    ax.set_xscale("log"); ax.set_xlim(0.8, 1.3e4); ax.set_ylim(0, 0.5); ax.set_yticks([0, 0.2, 0.4]); ax.axvline(30, color=F.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.legend(handles=[h_both(col, (r"$N=" if i == 0 else "$") + f"{N}$") for i, (N, col) in enumerate(zip(S["Ns"], F.ORANGES3))], loc="upper left", ncol=3,
              columnspacing=0.8, handlelength=1.3, borderaxespad=0.1)
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"P(a dip at step $t$)")


def panel_growth_q(ax, cfg, tools):
    """Growth rate against the trapped fraction of the interaction set, from the cache of fig_fr_theory f (R = 10, d_o = 5): E[Delta]
    of each set (tables alone; line: mean per bin of q) and its simulated late slope (markers: mean per bin +- 2 s.e.)."""
    import pickle
    theory, simulated, h_both, h_line, h_dot, Line2D = tools
    with open(os.path.join(A.ROOT, "main", "DATA", "experimental", "fr_theory", "fr_theory_cache", "optimum.pkl"), "rb") as fh: _, opt = pickle.load(fh)
    v = opt[(10, 5)]; edges = np.arange(0.0, 1.0001, 0.1); mid, th, sm, se = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (v["q"] >= lo) & (v["q"] < hi + (hi > 0.99) * 0.01)
        if m.sum() >= 30: mid.append(v["q"][m].mean()); th.append(v["ed"][m].mean()); sm.append(v["slope"][m].mean()); se.append(v["slope"][m].std() / np.sqrt(m.sum()))
    theory(ax, mid, th, F.BLUE); simulated(ax, mid, sm, 2 * np.array(se), F.BLUE)
    ax.set_xlim(0, 0.92); ax.set_ylim(0, 1.12 * max(th)); ax.set_xticks([0, 0.25, 0.5, 0.75]); ax.set_xlabel(r"trapped fraction $q$ of the set")
    ax.set_ylabel(r"$\mathbb{E}[\Delta]$ (nats / step)"); return len(v["q"])


def figures(cfg, D, tag=""):
    plt = _style(); tools = _helpers(); theory, simulated, h_both, h_line, h_dot, Line2D = tools
    def title(ax, letter, text): ax.set_title(f"({letter}) {text}", loc="left")
    def headers(fig, geo, items):
        W, H, TOP, BOT, ph, lm, rm, pw, gx = geo
        for j, name in items: fig.text((0.02 if j == 0 else lm + j * (pw + gx) - 0.40) / W, (BOT + ph + TOP - 0.135) / H, name, fontfamily="cmb10", fontsize=F.FS["header"], va="baseline")

    # ---------------------------------------------------------------- fig_fr_threshold: noise or real (upright, at the text width)
    fig, axs, geo = _row(plt, 3, lm=0.44, gx=0.50, rm=0.05, W=TEXT_WIDTH); a, b, c = axs
    panel_toy_dips(a, D, tools); title(a, "a", r"Dips: real or noise")
    panel_grid_threshold(b, D, cfg, tools); title(b, "b", r"Threshold and $q$")
    panel_grid_dips(c, D, cfg, tools); title(c, "c", rf"Dips in {cfg['T_POOL']} steps")
    headers(fig, geo, [(0, "Toy example"), (1, "Cognitive Gridworld")]); _key(fig, geo, h_dot, Line2D)
    for ax in axs: ax.tick_params(which="both", width=0.6); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    _save(plt, fig, "fig_fr_threshold", tag)

    # ---------------------------------------------------------------- fig_fr_candidates: other panels that could be added
    fig, axs, geo = _row(plt, 4, lm=0.43, gx=0.55, rm=0.05); a, b, c, d = axs
    panel_toy_step(a, D, tools); title(a, "A", r"Toy: the critical step $t^*$")
    panel_grid_mix(b, D, cfg, tools); title(b, "B", r"Gridworld version of (d)")
    panel_steps(c, D, tools); title(c, "C", r"How soon the law of dips holds")
    n_sets = panel_growth_q(d, cfg, tools); title(d, "D", r"Growth rate vs. $q$")
    _key(fig, geo, h_dot, Line2D)
    for ax in axs: ax.tick_params(which="both", width=0.6); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    _save(plt, fig, "fig_fr_candidates", tag)
