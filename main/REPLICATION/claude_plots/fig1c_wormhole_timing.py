"""When entanglement comes around the moves and the wormholes, and what the factorized agents do instead (29 Oct 2026;
John: "if the entanglement peaks right before entering a wormhole that could mean there is just a delay between network
and bayes. Maybe we could visualize wormhole particle traversal colored by entanglement with a shift, or a better way of
seeing this. Can we also see a cross correlation of entanglement and coupling (angle) and crossing probability. Also
wrong esn answers being adjacent could mean that the esn SHOULD have used the wormhole, aka coupled crossing, but
instead did an uncoupled crossing. That is evidence that wormhole are important. For the Naive evidence all adjacent
corners touch.")
-> main/REPLICATION/figure_bin/fig1c_wormhole_timing.png

Data: the alias scan (39 environments x 5,000 streams x 30 steps). Entanglement: log10 rev_t (the update into step t)
minus the same cell's typical value at that step (the mean of its 50 streams), then minus the stream's own mean of that
(steps 2 - 30). Both subtractions are needed: entanglement rises over steps 1 - 8 (log10 -0.88 -> -0.76) and the
wormhole events are early (median step 2 - 5), so with the stream's mean alone every early event sits on a negative
trend (the first version of this figure: entry +0.08, exit -1 +0.08, the factorized agents' flips +0.12 - 0.14).
Moves: each agent's posterior-mean jump (fig1c_crossings.jumps), amplitude (cells) and coupling angle (degrees from the
nearest axis: 0 one variable moves alone, 45 both alike). Crossings (fig1c_crossings.labels): a marginal moving >= 0.25
of its mass between its ends, coupled (both) or one-variable. A wormhole state: the Optimal's |Cov(r_g, r_c)| > 8
(a split across opposite corners); entry: not one at t - 1, one at t; the crossing out: one at t - 1, settled at t
(|Cov| < 2).
a - d  entanglement around every move of >= 2 cells, by its coupling angle (rows, 5 deg bins) and the step relative to
       the move (columns; 0 = the move's own update), one map per agent; bins with < 50 moves grey.
e  crossing probability after an entangled update: P(a crossing, either kind, at t + tau | the update into t is in its
   stream's top 10%) / the same environment's crossing rate at step t + tau; bands 5 - 95% over environments resampled.
f  every wormhole traversal (entry at t >= 2, ending on a settled step), grouped by how many steps the belief stays
   split; colour: the group's mean entanglement by step from entry; black boxes: the crossing out.
g  at each of the Optimal's coupled crossings: the share where the agent crosses at the Optimal's step + tau (the
   Trained RNN coupled, the Naive and the ESN one variable); dotted: the agent's own rate of that crossing.
h  corner truths (4 corner cells x 50 streams x 39 environments): each agent's step-30 answer, streams where the
   Optimal held a wormhole split at some step vs streams where it never did.

Measured 29 Oct:
- a - d, moves of >= 2 cells, lags -4 .. 0 (0 = the move's own update), coupling 30 - 45 deg: Optimal +0.028 / +0.039 /
  +0.049 / +0.064 / +0.074, Trained RNN +0.029 / +0.041 / +0.049 / +0.063 / +0.077 (a ramp over 4 updates); 0 - 15
  deg: Optimal +0.035 / +0.015 / +0.013 / -0.002 / +0.122, Trained RNN +0.020 / +0.023 / +0.030 / +0.025 / +0.112 (one
  update). Naive and ESN at every angle one update: Naive 0 - 15 deg +0.006 / +0.013 / +0.018 / +0.030 / +0.139, 30 -
  45 deg +0.021 / +0.017 / +0.018 / +0.021 / +0.080; ESN +0.003 / +0.005 / +0.019 / +0.042 / +0.108 and +0.018 /
  +0.005 / +0.021 / +0.015 / +0.073 (its diagonal moves +0.037 / +0.022 after, at +1 / +2). figure() prints the maps.
- e, P(crossing at t + tau | top-10% update into t) / the rate at that step, tau -1 / 0 / +1 / +2: Optimal 0.98 / 1.24
  / 1.49 / 1.38, Trained RNN 1.02 / 1.23 / 1.46 / 1.35, Naive 1.06 / 2.40 / 1.11 / 1.05, ESN 1.12 / 2.08 / 1.29 / 1.04:
  an entangled update doubles the factorized agents' crossing probability on that step and nothing after; the joint
  agents' rises less, and over the next 1 - 3 steps.
- f, traversals (3,586; 64% last one step, 2,302): the entry is the most entangled update (+0.20 / +0.15 / +0.14 /
  +0.13 for splits of 1 / 2 / 3 / 4 steps, +0.08 - 0.09 for longer ones), built up over the 3 updates before (+0.02 -
  0.08); inside the split +0.04 - 0.08, decaying; the update into the crossing out only +0.04 / +0.04 / +0.02 / -0.00.
  The exit's one-step lead in the first version was mostly the entry itself.
- g, the Optimal's coupled crossings (5,528): the Trained RNN crosses coupled on the same step in 73.4% (-1 / +1: 9.3 /
  8.0%; any crossing on the same step 94.6%): no network delay. The Naive flips one variable on the same step in 12.2%
  (4.2 / 4.5%; own rate 0.66%), the ESN 4.6% (1.5 / 3.1%; own rate 0.26%): 18x their rates, a one-variable flip in
  place of the coupled crossing.
- h, corner truths with / without a wormhole split (n 4,258 / 3,542): Optimal true .83 / .72, adjacent .002 / .000;
  Trained RNN .81 / .69, .004 / .001; Naive true .19 / .22, adjacent .165 / .098, partner .025 / .007; ESN .14 / .16,
  .161 / .085, .013 / .004. Where the Optimal used a wormhole, the joint agents end on the truth more often and the
  factorized agents one variable off twice as often (the Naive as much as the ESN: the factorized format). Corner
  streams with a one-variable split but no wormhole are too rare to compare (145 of 7,800): at the corners the
  Optimal's ambiguity is almost always across the wormhole.
- The one-variable flip on the Optimal's crossing step (g) is real but carries few of the adjacent errors. Corner
  streams with an Optimal coupled crossing (1,646): with the flip the Naive / ESN end adjacent 24% / 28% and on the
  truth 33% / 25%, without a flip within 2 steps adjacent 12% / 13%, truth 12% / 10% (the Trained RNN's own
  one-variable flips at those steps: adjacent 0.7%, truth 88%). Of all their corner adjacent errors 6% (Naive) / 3%
  (ESN) have such a flip (24% / 13% within streams where the Optimal crossed). Most come from the split state the
  factorized agents cannot hold, not from a missed crossing.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/fig1c_wormhole_timing.py"""
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
E, S, T, R = 39, 5000, 30, 10
LAGS = np.arange(-4, 5)
TAUS = np.arange(-3, 4)
AMIN, NPHI, MIN_N = 2., 9, 50
AG_COL = {"Joint": "k", "Trained": "#1f77b4", "Naive": "0.5", "Echo": "#ff7f0e"}
AG_NAME = {"Joint": "Optimal", "Trained": "Trained RNN", "Naive": "Naive", "Echo": "ESN"}


def data():
    """the scan's series, the Optimal's Cov(r_g, r_c) (E, S, T), entanglement minus typical minus the stream's mean
    (E, S, T; step 1 NaN), each agent's crossing kinds per arrival step (0 none, 1 coupled, 2 one variable)"""
    V = tp.series()
    cv = np.load(os.path.join(tp.mm.R7, "figure8_pooled.npz"))["CV"].astype(np.float64)
    lr = np.log10(V["rev"]).astype(np.float64)
    lr[:, :, 0] = np.nan
    d = fc.departures(lr)
    ws = d - np.nanmean(d, 2, keepdims=True)
    P = fc.primitives()
    kinds = {a: fc.at_steps(fc.labels(P[a][0])[0]) for a, _ in fc.AGENTS}
    return V, cv, ws, kinds


def angle_maps(ws):
    """per agent: mean ws around every move of >= AMIN cells by its coupling angle (NPHI bins over 0 - 45 deg) and the
    step relative to the move (LAGS) -> {agent: (map (NPHI, len(LAGS)), moves per bin)}"""
    J = fc.jumps()
    out = {}
    for a, _ in fc.AGENTS:
        amp, phi = (fc.at_steps(x.astype(np.float32)) for x in fc.polar(J[a]))
        e_, s_, t_ = np.nonzero(amp >= AMIN)
        b = np.minimum((phi[e_, s_, t_] * NPHI / 45).astype(int), NPHI - 1)
        M = np.full((NPHI, len(LAGS)), np.nan)
        for i, L in enumerate(LAGS):
            tt = t_ + L
            ok = (tt >= 1) & (tt < T)
            v = ws[e_[ok], s_[ok], tt[ok]]
            f = np.isfinite(v)
            M[:, i] = (np.bincount(b[ok][f], v[f], minlength=NPHI)
                       / np.maximum(np.bincount(b[ok][f], minlength=NPHI), 1))
        n = np.bincount(b, minlength=NPHI)
        M[n < MIN_N] = np.nan
        out[a] = (M, n)
    return out


def _boot(Ob, X, nboot=500, seed=0):
    """ratio of summed observed / expected over environments, and its 5 - 95% band over environments resampled"""
    idx = np.random.default_rng(seed).integers(0, E, (nboot, E))
    return Ob.sum(0) / X.sum(0), np.percentile(Ob[idx].sum(1) / X[idx].sum(1), [5, 95], 0)


def crossing_prob(ws, kinds, q=90):
    """P(a crossing at t + tau | ws at t in its stream's top 100 - q %) / the environment's crossing rate at step
    t + tau -> {agent: (ratio (len(TAUS),), band (2, len(TAUS)))}"""
    thr = np.nanpercentile(ws[:, :, 1:], q, axis=2, keepdims=True)
    e_, s_, t_ = np.nonzero((ws >= thr) & np.isfinite(ws))
    out = {}
    for a, _ in fc.AGENTS:
        ev = kinds[a] > 0
        base = ev.mean(1)
        Ob, X = np.zeros((E, len(TAUS))), np.zeros((E, len(TAUS)))
        for i, L in enumerate(TAUS):
            tt = t_ + L
            ok = (tt >= 1) & (tt < T)
            Ob[:, i] = np.bincount(e_[ok], ev[e_[ok], s_[ok], tt[ok]], minlength=E)
            X[:, i] = np.bincount(e_[ok], base[e_[ok], tt[ok]], minlength=E)
        out[a] = _boot(Ob, X)
    return out


def synced(kinds):
    """at each of the Optimal's coupled crossings, the share where the agent crosses (Trained RNN coupled, Naive and
    ESN one variable) at the Optimal's step + tau -> {agent: (share, band, the agent's rate of that crossing)}, n"""
    e_, s_, t_ = np.nonzero(kinds["Joint"] == 1)
    out = {}
    for a, kk in (("Trained", 1), ("Naive", 2), ("Echo", 2)):
        Ob, X = np.zeros((E, len(TAUS))), np.zeros((E, len(TAUS)))
        for i, L in enumerate(TAUS):
            tt = t_ + L
            ok = (tt >= 1) & (tt < T)
            Ob[:, i] = np.bincount(e_[ok], kinds[a][e_[ok], s_[ok], tt[ok]] == kk, minlength=E)
            X[:, i] = np.bincount(e_[ok], minlength=E)
        out[a] = _boot(Ob, X) + ((kinds[a][:, :, 1:] == kk).mean(),)
    return out, len(e_)


def traversals(cv, ws, before=3, after=14):
    """wormhole traversals (runs of |Cov| > 8 that start at t >= 2 and end with a settled step): rows of ws aligned on
    entry (columns -before .. +after), their durations"""
    worm = np.abs(cv) > 8
    rows, durs = [], []
    for e in range(E):
        for s in np.flatnonzero(worm[e].any(1)):
            w = worm[e, s]
            t = 1
            while t < T:
                if w[t] and not w[t - 1]:
                    u = t
                    while u < T and w[u]:
                        u += 1
                    if u < T and abs(cv[e, s, u]) < 2:
                        r = np.full(before + after + 1, np.nan)
                        lo, hi = max(1, t - before), min(T, t + after + 1)
                        r[lo - (t - before):hi - (t - before)] = ws[e, s, lo:hi]
                        rows.append(r), durs.append(u - t)
                    t = u
                t += 1
    return np.array(rows), np.array(durs)


def answers(V, cv):
    """corner truths: each agent's step-30 answer category, with / without a wormhole split in the Optimal"""
    cell = V["cell"].astype(int)
    corners = {(0, 0): (9, 9), (9, 9): (0, 0), (0, 9): (9, 0), (9, 0): (0, 9)}
    had = (np.abs(cv) > 8).any(2)
    out = {}
    for a in ("Joint", "Trained", "Naive", "Echo"):
        B = np.load(os.path.join(tp.mm.SCAN, f"scan_{a}.npz"))["beliefs"][tp.mm.ENVS][:, :, -1].astype(np.float32)
        mg, mc = B[..., :R].argmax(-1), B[..., R:].argmax(-1)
        cnt = {flag: {"true": 0, "adjacent": 0, "partner": 0, "n": 0} for flag in (True, False)}
        for (g, c), (pg, pc) in corners.items():
            srows = np.flatnonzero((cell[:, 0] == g) & (cell[:, 1] == c))
            G, C, H = mg[:, srows], mc[:, srows], had[:, srows]
            for flag in (True, False):
                m = H == flag
                cnt[flag]["n"] += m.sum()
                cnt[flag]["true"] += ((G == g) & (C == c) & m).sum()
                cnt[flag]["partner"] += ((G == pg) & (C == pc) & m).sum()
                cnt[flag]["adjacent"] += ((((G == pg) & (C == c)) | ((G == g) & (C == pc))) & m).sum()
        out[a] = {flag: {k: v / max(d["n"], 1) for k, v in d.items() if k != "n"} | {"n": d["n"]}
                  for flag, d in cnt.items()}
    return out


def figure():
    from main.utils import apply_fig_style, fig_path
    from matplotlib.lines import Line2D
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    V, cv, ws, kinds = data()
    names = ("Joint", "Trained", "Echo", "Naive")
    W_, H_ = 17.0, 8.9
    fig = plt.figure(figsize=(W_, H_))

    def ax_at(x, y, w, h):
        return fig.add_axes([x / W_, y / H_, w / W_, h / H_])

    fig.text(.5, 1 - .3 / H_, "The joint agents move diagonally after entanglement builds over several updates, the "
             "factorized agents on a single entangled update; the Trained RNN keeps the Optimal's timing",
             ha="center", va="center", fontsize=FS["label"] + 3)
    fig.text(.5, 1 - .64 / H_, "alias scan, 39 environments x 5,000 streams x 30 steps; entanglement: log$_{10}$ "
             r"$\mathrm{rev}_t$ (the update into $t$) minus the same cell's typical value at that step, minus the "
             "stream's own mean (log$_{10}$ nats)", ha="center", va="center", fontsize=FS["tick"])
    cmap = plt.get_cmap("PiYG").copy()
    cmap.set_bad("0.86")
    lim = .1
    # a - d: entanglement around every move of >= 2 cells, by coupling angle x step from the move
    A = angle_maps(ws)
    fig.text(.85 / W_, 1 - 1.02 / H_, r"Entanglement around every move of $\geq$ 2 cells, by the move's coupling "
             "angle and the step relative to it", ha="left", va="center", fontsize=FS["label"] + 1)
    for j, a in enumerate(names):
        M, n = A[a]
        ax = ax_at(.85 + 3.95 * j, 5.0, 3.3, 2.55)
        im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest",
                       extent=(LAGS[0] - .5, LAGS[-1] + .5, 0, 45))
        ax.axvline(-.5, color="k", lw=.7, ls=":")
        ax.axvline(.5, color="k", lw=.7, ls=":")
        ax.set_xticks(LAGS)
        ax.set_yticks([0, 15, 30, 45])
        ax.tick_params(labelsize=FS["tick"])
        ax.set_xlabel("steps relative to the move (0: its own update)", fontsize=FS["tick"] + .5)
        if j == 0:
            ax.set_ylabel("coupling angle (deg)\n0: one variable, 45: both", fontsize=FS["tick"] + .5)
        ax.set_title(f"{'abcd'[j]}   {AG_NAME[a]}  ({n.sum():,} moves)", fontsize=FS["label"] + .5, loc="left",
                     pad=5)
    cax = ax_at(16.05, 5.0, .14, 2.55)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("entanglement", fontsize=FS["tick"])
    cb.ax.tick_params(labelsize=FS["tick"] - 1)
    # e: crossing probability after an entangled update
    C = crossing_prob(ws, kinds)
    ax = ax_at(.85, .85, 3.0, 2.85)
    for a in names:
        m, band = C[a]
        ax.fill_between(TAUS, band[0], band[1], color=AG_COL[a], alpha=.15, lw=0)
        ax.plot(TAUS, m, color=AG_COL[a], lw=1.8, marker="o", ms=3, label=AG_NAME[a])
    ax.axhline(1, color="0.6", lw=.8, ls=":")
    ax.axvline(0, color="0.3", lw=.6, ls=":")
    ax.set_xticks(TAUS)
    ax.set_xlabel(r"$\tau$: steps after the entangled update", fontsize=FS["tick"] + .5)
    ax.set_ylabel("crossing probability at $t+\\tau$\n(x the rate at that step)", fontsize=FS["tick"] + .5)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="upper left")
    ax.set_title("e   Crossings after a top-10% entangled update", fontsize=FS["label"] + .5, loc="left", pad=5)
    # f: traversals grouped by how long the belief stays split
    rows, durs = traversals(cv, ws)
    groups = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 14))
    before = 3
    cols = np.arange(-before, rows.shape[1] - before)
    Mt = np.full((len(groups), len(cols)), np.nan)
    ns = []
    for gi, (lo, hi) in enumerate(groups):
        m = (durs >= lo) & (durs <= hi)
        ns.append(int(m.sum()))
        Mt[gi] = np.nanmean(rows[m], 0)
    ax = ax_at(4.75, .85, 3.55, 2.85)
    im = ax.imshow(Mt, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest",
                   extent=(cols[0] - .5, cols[-1] + .5, len(groups) - .5, -.5))
    for gi, (lo, hi) in enumerate(groups):
        for d in range(lo, hi + 1):
            if d <= cols[-1]:
                ax.add_patch(plt.Rectangle((d - .5, gi - .5), 1, 1, fill=False, ec="k", lw=1.1))
    ax.axvline(-.5, color="k", lw=.7, ls=":")
    ax.axvline(.5, color="k", lw=.7, ls=":")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{lo}" if lo == hi else f"{lo} - {hi}" for lo, hi in groups], fontsize=FS["tick"])
    ax.set_ylabel("steps the belief stays split", fontsize=FS["tick"] + .5)
    ax.set_xlabel("steps from entering (box: the crossing out)", fontsize=FS["tick"] + .5)
    ax.tick_params(labelsize=FS["tick"])
    for gi, n_ in enumerate(ns):
        ax.text(cols[-1] + .4, gi, f"n {n_:,}", ha="right", va="center", fontsize=FS["tick"] - 1.5, color="0.2")
    ax.set_title(f"f   Every wormhole traversal ({len(durs):,}; colour as a - d)", fontsize=FS["label"] + .5,
                 loc="left", pad=5)
    # g: what each agent does at the Optimal's coupled crossings
    G, n_cc = synced(kinds)
    ax = ax_at(9.35, .85, 2.9, 2.85)
    for a, lab in (("Trained", "Trained RNN, coupled"), ("Naive", "Naive, one variable"),
                   ("Echo", "ESN, one variable")):
        m, band, base = G[a]
        ax.fill_between(TAUS, 100 * band[0], 100 * band[1], color=AG_COL[a], alpha=.15, lw=0)
        ax.plot(TAUS, 100 * m, color=AG_COL[a], lw=1.8, marker="o", ms=3, label=lab)
        ax.axhline(100 * base, color=AG_COL[a], lw=.9, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(.008, 100)
    ax.set_yticks([.01, .1, 1, 10, 100])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.axvline(0, color="0.3", lw=.6, ls=":")
    ax.set_xticks(TAUS)
    ax.set_xlabel(r"$\tau$: steps from the Optimal's crossing", fontsize=FS["tick"] + .5)
    ax.set_ylabel(r"% crossing at $\tau$ (dotted: own rate)", fontsize=FS["tick"] + .5)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="lower left", bbox_to_anchor=(0, .02))
    ax.set_title(f"g   At the Optimal's coupled crossings ({n_cc:,})", fontsize=FS["label"] + .5, loc="left", pad=5)
    # h: corner truths' answers, with vs without a wormhole split
    Ans = answers(V, cv)
    ax = ax_at(13.45, .85, 3.35, 2.85)
    order = ("Joint", "Trained", "Naive", "Echo")
    for i, a in enumerate(order):
        for j, (k, mk) in enumerate((("true", "o"), ("adjacent", "s"))):
            y = i + (-.15 if j == 0 else .15)
            v0, v1 = Ans[a][False][k], Ans[a][True][k]
            ax.plot([v0, v1], [y, y], color=AG_COL[a], lw=1.6, alpha=.7)
            ax.scatter([v0], [y], s=30, facecolors="white", edgecolors=AG_COL[a], marker=mk, linewidths=1.3, zorder=3)
            ax.scatter([v1], [y], s=30, color=AG_COL[a], marker=mk, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([AG_NAME[a] for a in order], fontsize=FS["tick"] + .5)
    ax.invert_yaxis()
    ax.set_xlim(-.02, 1)
    ax.set_xlabel("share of corner truths at step 30", fontsize=FS["tick"] + .5)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("h   Where the Optimal used a wormhole", fontsize=FS["label"] + .5, loc="left", pad=5)
    hand = [Line2D([], [], marker="o", ls="", color="0.3", label="on the true corner"),
            Line2D([], [], marker="s", ls="", color="0.3", label="on an adjacent corner"),
            Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", label="no wormhole split"),
            Line2D([], [], marker="o", ls="", color="0.3", label="a wormhole split")]
    ax.legend(handles=hand, fontsize=FS["tick"] - 1, frameon=False, loc="lower right")
    path = fig_path("fig1c_wormhole_timing.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("saved", path)
    print("NUMBERS")
    for a in names:
        M, n = A[a]
        print(AG_NAME[a], "moves per 5-deg bin", n.tolist())
        for k in range(NPHI):
            print(f"  {5 * k:2d}-{5 * k + 5:2d} deg " + " ".join(f"{x:+.3f}" for x in M[k]))
        for lo, hi in ((0, 3), (6, 9)):
            Mb = np.nansum(A[a][0][lo:hi] * n[lo:hi, None], 0) / n[lo:hi].sum()
            print(f"  {5 * lo}-{5 * hi} deg pooled " + " ".join(f"{x:+.3f}" for x in Mb))
    for a in names:
        print(AG_NAME[a], "crossing probability ratio", " ".join(f"{x:.2f}" for x in C[a][0]))
    for a in ("Trained", "Naive", "Echo"):
        print(AG_NAME[a], "at the Optimal's coupled crossings (%)", " ".join(f"{100 * x:.1f}" for x in G[a][0]),
              f"own rate {100 * G[a][2]:.2f}")
    print("traversal groups", ns, "entry-aligned means by group, steps -3 .. 9 from entry:")
    for gi in range(len(groups)):
        print("  ", " ".join(f"{x:+.3f}" for x in Mt[gi][:13]))
    for a in order:
        print(AG_NAME[a], {flag: {k: round(float(v), 3) for k, v in d.items()} for flag, d in Ans[a].items()})
    return path


if __name__ == "__main__":
    figure()
