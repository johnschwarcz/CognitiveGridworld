"""explain_interaction.png: what fig_learn_interaction's measure is, as distributions. John 11-10 ("I really dont
understand the interaction measure"), then, after a scatter version (effect vs interaction for four agents on a subset;
scratchpad copy explain_interaction.scatter_v1.py): "ah okay, so for interaction, the Delta' - Delta is how much the
j'th flip modulates (strengthens or weakens) the i'th flip. Maybe we should collapse the x-axis to turn the cluster into
a distribtion. Then 3 can have just 2 panels, joint, ESN and trained distributions for between and within (2 panels)."

The measure: flip bit i of the observation at step t; the agent's update at t (the change of its answer's log-odds, per
value of a variable, centred over the 10 values) changes by Delta. Flip bit j as well, in an earlier observation o_s
(between steps) or in o_t itself (within a step): bit i's effect becomes Delta'. Delta' - Delta = fig_learn_interaction's
mixed difference D_i D_j inc_t: how much bit j's flip modulates bit i's effect. Drawn along bit i's effect,
    sign(Delta) (Delta' - Delta):  > 0  j's flip strengthens i's effect,  < 0  it weakens it (or reverses it).
Naive Bayes adds a fixed amount of evidence per bit and step: Delta' = Delta always (0, exactly). Joint Bayes reads a bit
about one variable through its belief about the other, which every other bit moves.

Data: fig_learn_interaction's own sets, at the end of training (raw 49,995 = 39,996 training episodes): the 1,000 probe
episodes of td_tt_mixed_diff (50 test environments x 20); between steps the 32 (t, s) pairs (t = 5, ..., 30; t - s = 1,
2, 3, 5, 8, 12) x 25 bit pairs (j of o_s, i of o_t); within a step t = 5, 10, ..., 30 x 10 bit pairs, each in both orders
(i's effect modulated by j, and j's by i); both variables, 10 values each: 16,000,000 values between steps and
2 x 1,200,000 within a step per agent. Computed here (CPU, ~3 min) with td_tt_mixed_diff's observers and network cells
(md.observer_md / md.net_md between steps; td_tt_mixed_diff0's within-step loop, keeping the single-flip effects for
the sign) into through_training/mixed_diff/explain_dist.npz.

RESULTS (11-10, --numbers; mean |Delta' - Delta| per value, nats, % of Joint Bayes's: identical to fig_learn_interaction's
end-of-training values)
  between steps   Joint Bayes .0748, Trained RNN .0542 (72%), ESN .0049 (7%); median .035 / .035 / .0019, 99th
                  percentile .66 / .31 / .035
  within a step   Joint Bayes .0754, Trained RNN .0883 (117%), ESN .1147 (152%); median .036 / .067 / .094
  Naive Bayes     max |Delta' - Delta| 2e-16 between steps, 4e-15 within a step (0 up to rounding)
  Sign: bit j's flip weakens bit i's effect slightly more often than it strengthens it (strengthens 45-49% of the values
  with |Delta' - Delta| > .001, every agent; mean signed value -.014 / -.004 / -.0002 between steps). Joint Bayes has
  the heaviest tails (0.35% / 0.29% of its values beyond +-1.2 nats, not drawn; no network value is).
BY DELAY (John 11-10, second version: one panel per agent, a curve per delay; steps t = 15, 20, 25, 30, which have
every delay; mean |Delta' - Delta|, % of the agent's own same-step value):
  Joint Bayes  same step .0506; 1, 2, 3, 5, 8, 12 steps earlier .056 each (111%): no decay at all
  Trained RNN  .0763; .0525 / .0444 / .0403 / .0363 / .0344 / .0332 (69% -> 44%): decays, levels off
  ESN          .1143; .0133 / .0071 / .0041 / .0016 / .0004 / .0001 (12% -> 0%): gone within a few steps
DRAWN (third version, 11-10): delays 0, 1, 10 (t = 15-30; 10 from explain_dist_lag10.npz: Joint Bayes .0564 (111%),
Trained RNN .0337 (44%), ESN .0002 (0%)); linear densities smoothed with a Gaussian of SD .012 nats, so the ESN's
1- and 10-step curves (true spread ~.013 / ~.0002 nats) are drawn at about the kernel's width: their peak height is set
by the smoothing, not the data.
DRAWN (fourth version, 11-10): panels = delays 0, 1, 10; in each Joint Bayes dashed black, Trained RNN blue, ESN
orange (fig_main_common.COL); each panel its own y range.
READING (two-panel version): between steps the ESN's distribution is a spike at 0: what a new observation does to its answer barely depends
on what it saw before (its update is additive across steps, like naive's), although the bits of one observation do
interact (within a step its distribution is as wide as the others'). Joint Bayes and the Trained RNN read the present
bit in light of the past. A signature of the computation, not shown to be the cause of the ESN's errors
(fig_learn_interaction's NOT CLAIMED).
usage:  python explain_interaction.py [--data] [--numbers]"""
import os
import sys
import time
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import td_tt_mixed_diff as md                                                     # noqa: E402

T, R, NV, NO, TS, PAIRS = md.T, md.R, md.NV, md.NO, md.TS, md.PAIRS
BITS = tuple((i, j) for i in range(NO) for j in range(i + 1, NO))                  # within a step: 10 unordered pairs
RAW = 49995                                                                       # end of training
AGENTS = (("bayes", "Joint Bayes", "k"), ("trained", "Trained RNN", "#1f77b4"), ("echo", "ESN", "#ff7f0e"))
CACHE = os.path.join(md.OUT, "explain_dist.npz")
FS = dict(tick=7, key=7, label=8, title=8, letter=9)                              # points at print size
FIG_W, DPI = 5.5, 400


# ── data ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def observer(kind, S):
    """between steps: MD (32 pairs, 5 j, 5 i, n, 2, R) (md.observer_md); within a step: the single-flip effects
    (6 steps, 5 i, n, 2, R) and MD0 (6 steps, 10 bit pairs, n, 2, R) (td_tt_mixed_diff0.observer_md0's loop)"""
    between = md.observer_md(kind, S)
    obs = S["obs"]
    n = len(obs)
    LLs = np.stack([md.ll_step(S, obs[:, k]) for k in range(T)], 1)
    cum = np.concatenate([np.zeros((n, 1, R, R)), np.cumsum(LLs, 1)], 1)
    eff, md0 = np.zeros((len(TS), NO, n, NV, R)), np.zeros((len(TS), len(BITS), n, NV, R))
    for a, t in enumerate(TS):
        prev, o = cum[:, t - 1], obs[:, t - 1]
        base = md.inc_obs(kind, S, prev, o)
        one = [md.inc_obs(kind, S, prev, md.flip(o, i)) for i in range(NO)]
        eff[a] = np.array(one) - base
        for b, (i, j) in enumerate(BITS):
            md0[a, b] = md.inc_obs(kind, S, prev, md.flip(md.flip(o, i), j)) - one[i] - one[j] + base
    return between, eff, md0


def network(run, S):
    """the same for a network (md.net_md's fast path between steps; td_tt_mixed_diff0.net_md0's loop within a step)"""
    import torch
    cell = md.LSTMCell(run, RAW)
    envt, code = md.codes(S["rows"])
    U, C = cell.replay(envt, code)
    between = md.net_md(cell, envt, code, U, C)
    n = len(envt)
    eff, md0 = np.zeros((len(TS), NO, n, NV, R)), np.zeros((len(TS), len(BITS), n, NV, R))
    with torch.no_grad():
        for a, t in enumerate(TS):
            Rc, c0 = cell.rec(U[:, t - 1]), C[:, t - 1]

            def inc(cd):
                h1, _ = cell.act(cell.table[envt, cd] + Rc, c0)
                return cell.inc(h1).double().numpy()
            cd = code[:, t - 1]
            base = inc(cd)
            one = [inc(cd ^ (1 << i)) for i in range(NO)]
            eff[a] = np.array(one) - base
            for b, (i, j) in enumerate(BITS):
                md0[a, b] = inc(cd ^ (1 << i) ^ (1 << j)) - one[i] - one[j] + base
    return between, eff, md0


def build():
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    S = md.side()
    out = {}
    for key in ("naive",) + tuple(k for k, _, _ in AGENTS):
        t0 = time.time()
        b, e, w = observer(key, S) if key in ("naive", "bayes") else network(key, S)
        if key == "naive":                                                        # exactly 0: kept as a check only
            out["naive_max"] = np.array([np.abs(b).max(), np.abs(w).max()])
        else:
            out.update({f"{key}_between": b.astype(np.float32), f"{key}_eff": e.astype(np.float32),
                        f"{key}_within": w.astype(np.float32)})
        print(f"{key}: {time.time() - t0:.0f} s", flush=True)
    np.savez(CACHE[:-4] + ".tmp.npz", rows=S["rows"], pairs=np.array(PAIRS), ts=np.array(TS), **out)
    os.replace(CACHE[:-4] + ".tmp.npz", CACHE)
    print("wrote", CACHE)


CACHE10 = os.path.join(md.OUT, "explain_dist_lag10.npz")
PAIRS10 = tuple((t, t - 10) for t in md.T_BAL)                                    # a delay of 10 at t = 15-30


def build_lag10():
    """MD for a delay of 10 steps (John 11-10: "we just need curves for 0, 1 step and 10 steps"; the stored sets have
    8 and 12): md.observer_md / md.net_md on PAIRS10, same probes and checkpoint"""
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    S = md.side()
    out = {}
    for key, _, _ in AGENTS:
        if key == "bayes":
            b = md.observer_md(key, S, pairs=PAIRS10)
        else:
            cell = md.LSTMCell(key, RAW)
            envt, code = md.codes(S["rows"])
            U, C = cell.replay(envt, code)
            b = md.net_md(cell, envt, code, U, C, pairs=PAIRS10)
        out[f"{key}_lag10"] = b.astype(np.float32)
        print(key, "lag 10 done", flush=True)
    np.savez(CACHE10[:-4] + ".tmp.npz", rows=S["rows"], pairs=np.array(PAIRS10), **out)
    os.replace(CACHE10[:-4] + ".tmp.npz", CACHE10)


CACHEMAX = os.path.join(md.OUT, "explain_dist_lagmax.npz")
PAIRSMAX = ((T, 1),)                                                              # the maximum gap: 29 steps


def build_lagmax():
    """MD for the maximum gap (John 11-10: "change d to the maximum gap"): bit j in o_1, bit i in o_30 (29 steps),
    md.observer_md / md.net_md on PAIRSMAX, same probes and checkpoint"""
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    S = md.side()
    out = {}
    for key, _, _ in AGENTS:
        if key == "bayes":
            b = md.observer_md(key, S, pairs=PAIRSMAX)
        else:
            cell = md.LSTMCell(key, RAW)
            envt, code = md.codes(S["rows"])
            U, C = cell.replay(envt, code)
            b = md.net_md(cell, envt, code, U, C, pairs=PAIRSMAX)
        out[f"{key}_lagmax"] = b.astype(np.float32)
        print(key, "maximum gap done", flush=True)
    np.savez(CACHEMAX[:-4] + ".tmp.npz", rows=S["rows"], pairs=np.array(PAIRSMAX), **out)
    os.replace(CACHEMAX[:-4] + ".tmp.npz", CACHEMAX)


def load():
    if not os.path.exists(CACHE):
        build()
    if not os.path.exists(CACHE10):
        build_lag10()
    if not os.path.exists(CACHEMAX):
        build_lagmax()
    Z = dict(np.load(CACHE))
    Z.update(dict(np.load(CACHE10)))
    Z.update(dict(np.load(CACHEMAX)))
    return Z


def aligned(Z, key):
    """sign(Delta) (Delta' - Delta), Delta = bit i's own effect at step t: between steps (32 x 25 x n x 2 x R) and
    within a step (both orders of the 10 bit pairs)"""
    eff = Z[f"{key}_eff"].astype(np.float64)                                      # (6 t, 5 i, n, 2, R)
    tix = np.array([list(TS).index(t) for t, s in PAIRS])
    between = (Z[f"{key}_between"] * np.sign(eff[tix])[:, None]).ravel()          # MD (32, 5 j, 5 i, ...): i = axis 2
    I, J = np.array([i for i, j in BITS]), np.array([j for i, j in BITS])
    w = Z[f"{key}_within"].astype(np.float64)
    within = np.concatenate([(w * np.sign(eff[:, I])).ravel(), (w * np.sign(eff[:, J])).ravel()])
    return between, within


def numbers(Z):
    print(f"naive Bayes: max |MD| between steps {Z['naive_max'][0]:.1e}, within a step {Z['naive_max'][1]:.1e}")
    ref = {}
    for kind in ("between", "within"):
        print(f"{kind}: mean |MD| per value (nats), % of Joint Bayes's; signed along Delta: mean, share > 0 (strengthens) "
              f"among |MD| > 1e-3; quantiles of |MD| (50 / 90 / 99%)")
        for key, name, _ in AGENTS:
            a = aligned(Z, key)[0 if kind == "between" else 1]
            m = np.abs(a).mean()
            ref.setdefault(kind, m)
            big = np.abs(a) > 1e-3
            q = np.quantile(np.abs(a), (.5, .9, .99))
            print(f"   {name:12s} {m:.4f} {100 * m / ref[kind]:5.0f}%   mean {a.mean():+.4f}   strengthens "
                  f"{(a[big] > 0).mean():.2f} (of {big.mean():.2f})   |MD| {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}   "
                  f"beyond 1.2 nats {100 * (np.abs(a) > 1.2).mean():.2f}%")


# ── figure ──────────────────────────────────────────────────────────────────────────────────────────────────────────

TB = md.T_BAL                                                                     # steps with every lag: 15, 20, 25, 30
LAGS = (0,) + tuple(md.LAGS)                                                      # 0 = within a step
SHOW = (0, 1, 5, 10)                                                              # John 11-10: "0, 1 step and 10 steps",
                                                                                  # then "thinner curves and add t = 5"
SMOOTH = .012                                                                     # Gaussian kernel SD, nats


def by_lag(Z, key, lags=LAGS):
    """sign(Delta) (Delta' - Delta) per delay t - s at the steps t in TB (the same steps for every delay): 0 = bit j in
    o_t itself (both orders of the 10 bit pairs), d >= 1 = bit j in o_(t - d) (25 bit pairs); d = 10 from CACHE10"""
    eff = Z[f"{key}_eff"].astype(np.float64)
    out = {}
    ti = [list(TS).index(t) for t in TB]
    I, J = np.array([i for i, j in BITS]), np.array([j for i, j in BITS])
    for d in lags:
        if d == 0:
            w = Z[f"{key}_within"][ti].astype(np.float64)
            out[0] = np.concatenate([(w * np.sign(eff[ti][:, I])).ravel(), (w * np.sign(eff[ti][:, J])).ravel()])
        elif d == 10:
            tix = [list(TS).index(t) for t, s_ in PAIRS10]
            out[10] = (Z[f"{key}_lag10"] * np.sign(eff[tix])[:, None]).ravel()
        else:
            ps = [p for p, (t, s_) in enumerate(PAIRS) if t in TB and t - s_ == d]
            tix = [list(TS).index(PAIRS[p][0]) for p in ps]
            out[d] = (Z[f"{key}_between"][ps] * np.sign(eff[tix])[:, None]).ravel()
    return out


def lag_numbers(Z):
    print(f"mean |MD| per value by delay (steps t = {TB}; 0 = within a step), nats; % of the agent's own delay-0 value")
    for key, name, _ in AGENTS:
        L = by_lag(Z, key, LAGS + (10,))
        m = {d: np.abs(v).mean() for d, v in L.items()}
        print(f"   {name:12s} " + "  ".join(f"{d}: {m[d]:.4f} ({100 * m[d] / m[0]:.0f}%)" for d in sorted(m)))


def smooth_density(a, X, bw=.002, sd=SMOOTH):
    """the values' density (per nat) on a fine grid over [-X, X], smoothed with a Gaussian of SD sd (values outside
    the range are counted in the total, not drawn)"""
    edges = np.arange(-X - 4 * sd, X + 4 * sd + bw / 2, bw)
    h = np.histogram(a, edges)[0] / (a.size * bw)
    k = np.arange(-4 * sd, 4 * sd + bw / 2, bw)
    k = np.exp(-.5 * (k / sd) ** 2)
    h = np.convolve(h, k / k.sum(), mode="same")
    x = (edges[:-1] + edges[1:]) / 2
    keep = np.abs(x) <= X
    return x[keep], h[keep]


MAXGAP = T - 1                                                                    # 29 steps: o_1 vs o_30
SHOW = (0, 1, 5, MAXGAP)


def effect_pairs(Z, key, d):
    """(Delta, Delta' - Delta) per value at delay d: 0 = same step (both orders of the 10 bit pairs, t in TB), 1-12 / 10
    at t in TB, MAXGAP = (t, s) = (30, 1) only"""
    eff = Z[f"{key}_eff"].astype(np.float64)                                      # (6 t, 5 i, n, 2, R)
    if d == 0:
        ti = [list(TS).index(t) for t in TB]
        w = Z[f"{key}_within"][ti].astype(np.float64)
        I, J = np.array([i for i, j in BITS]), np.array([j for i, j in BITS])
        return np.concatenate([eff[ti][:, I].ravel(), eff[ti][:, J].ravel()]), np.concatenate([w.ravel(), w.ravel()])
    if d == MAXGAP:
        tix, mdv = [list(TS).index(T)], Z[f"{key}_lagmax"].astype(np.float64)
    elif d == 10:
        tix, mdv = [list(TS).index(t) for t, s_ in PAIRS10], Z[f"{key}_lag10"].astype(np.float64)
    else:
        ps = [p for p, (t, s_) in enumerate(PAIRS) if t in TB and t - s_ == d]
        tix, mdv = [list(TS).index(PAIRS[p][0]) for p in ps], Z[f"{key}_between"][ps].astype(np.float64)
    D = np.broadcast_to(eff[tix][:, None], mdv.shape)                             # Delta does not depend on bit j
    return D.ravel(), mdv.ravel()


XLAB = r"$\mathrm{sign}(\Delta)\,(\Delta' - \Delta)$ (nats)   ($<0$ weakens, $>0$ strengthens bit $i$'s effect)"
MEASURES = (                                                                      # (x range, SD, log density axis)
    ((-.6, .6), .012, False),
    ((-1.5, 1.5), .012, True))                                                    # John 11-10 15:10: log scaling
                                                                                  # for the difference, no ratios


def measure(D, M):
    ok = np.abs(D) > 1e-12
    return np.sign(D[ok]) * M[ok]


def smooth_density2(a, lo, hi, sd, bw=None):
    bw = bw or sd / 6
    edges = np.arange(lo - 4 * sd, hi + 4 * sd + bw / 2, bw)
    h = np.histogram(a, edges)[0] / (a.size * bw)
    k = np.arange(-4 * sd, 4 * sd + bw / 2, bw)
    k = np.exp(-.5 * (k / sd) ** 2)
    h = np.convolve(h, k / k.sum(), mode="same")
    x = (edges[:-1] + edges[1:]) / 2
    keep = (x >= lo) & (x <= hi)
    return x[keep], h[keep]


def figure(Z, path=None):
    """John 11-10 (fourth version, made outside this session): "let's have the 3 panels be step 0, 1 and 10, with a
    dashed black curve for joint, and then our typical coloring for trained and ESN"; fifth: "thinner curves and add
    t = 5"; sixth: "can we change d to the maximum gap and also make the figure much wider. For the reversing sign,
    isnt that easily fixed from using absolute Delta? can you show how this would come out as a second row, so i could
    see? Also another row for the log scaling of the gap that you used before?". Columns: bit j in the same step, 1, 5
    and 29 steps earlier (the maximum: o_1 vs o_30); rows: the difference sign(Delta)(Delta' - Delta) (nats), the
    magnitude ratio |Delta'| / |Delta| and its log2. Seventh (John 11-10 15:10, to the cluster-training session): "i meant
    the log scaling should be for the difference version not the ratio version. Yea let's get rid of the ratio
    versions": rows = the difference on a linear and on a log density axis (x to +-1.5 nats for the tails); ratio rows
    removed (they made the ESN look Bayes-like at 1 step; scratchpad copy explain_interaction.v6_ratio.py). Densities smoothed with a Gaussian (SD per row in MEASURES;
    a spike is drawn at the kernel's width); each panel its own y range."""
    apply_fig_style()
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.linewidth": .6})
    W_ = 9.5
    left, right, gap, ph, rgap = .62, .08, .28, 1.25, .62
    pw = (W_ - left - right - 3 * gap) / 4
    top0 = .52
    H = top0 + len(MEASURES) * ph + (len(MEASURES) - 1) * rgap + .55
    fig = plt.figure(figsize=(W_, H))
    ftext = lambda x, t, s_, **k: fig.text(x / W_, 1 - t / H, s_, **k)
    style = {"bayes": dict(color="k", lw=.9, ls=(0, (3.5, 2)), zorder=3),
             "trained": dict(color=AGENTS[1][2], lw=1.0, zorder=2), "echo": dict(color=AGENTS[2][2], lw=1.0, zorder=1)}
    titles = {0: "$j$: same step", 1: "$j$: 1 step earlier", 5: "$j$: 5 steps earlier",
              MAXGAP: f"$j$: {MAXGAP} steps earlier (maximum)"}
    P = {(key, d): measure(*effect_pairs(Z, key, d)) for key, _, _ in AGENTS for d in SHOW}
    for row, ((lo, hi), sd, logy) in enumerate(MEASURES):
        top = top0 + row * (ph + rgap)
        for k, d in enumerate(SHOW):
            x0 = left + k * (pw + gap)
            ax = fig.add_axes([x0 / W_, 1 - (top + ph) / H, pw / W_, ph / H])
            ymax = 0
            for key, _, _ in AGENTS:
                xx, h = smooth_density2(P[key, d], lo, hi, sd)
                ax.plot(xx, h, solid_capstyle="round", **style[key])
                ymax = max(ymax, h.max())
            ax.axvline(0, color="0.75", lw=.5, zorder=0)
            ax.set_xlim(lo, hi)
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(1e-3, 2 * ymax)
            else:
                ax.set_ylim(0, 1.06 * ymax)
            ax.tick_params(labelsize=FS["tick"], length=2.5, width=.6, pad=1.5)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ftext(x0 + pw / 2, top - .08, titles[d], fontsize=FS["title"], ha="center", va="baseline")
            ftext(x0 - (.45 if k == 0 else .2), top - .08, r"$\mathbf{%s}$" % "abcdefgh"[4 * row + k],
                  fontsize=FS["letter"], va="baseline")
            if k == 0:
                ax.set_ylabel("density (log scale)" if logy else "density", fontsize=FS["label"], labelpad=2)
        ftext(left + (4 * pw + 3 * gap) / 2, top + ph + .2, XLAB, fontsize=FS["label"], ha="center", va="top")
    hs = [Line2D([], [], **{k_: v for k_, v in style[key].items() if k_ != "zorder"}) for key, _, _ in AGENTS]
    fig.legend(hs, [name for _, name, _ in AGENTS], loc="upper center", ncol=3, bbox_to_anchor=(.5, 1 - .04 / H),
               fontsize=FS["key"], frameon=False, handlelength=2.2, columnspacing=1.6, handletextpad=.5, borderaxespad=0)
    path = path or fig_path("explain_interaction.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    if "--data" in sys.argv:
        build()
    Z = load()
    if "--numbers" in sys.argv:
        numbers(Z)
        lag_numbers(Z)
    figure(Z)
