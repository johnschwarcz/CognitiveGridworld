"""The minimal set of figures for why the trained RNN reaches optimal Bayes and the ESN stays at naive Bayes, from data
that already exists (29 Oct 2026; John: "plot everything you suggest which we can currently plot with the existing
data. And also a single figure of your minimal set")
-> main/REPLICATION/figure_bin/fig_mechanism_memory.png       figure_memory()
   main/REPLICATION/figure_bin/fig_mechanism_esn_readout.png  figure_readout()
   main/REPLICATION/figure_bin/fig_minimal_set.png            figure_minimal()

The chain, one panel each in the minimal set:
a  mechanism (memory): what a network does with its history. Each network's memory cut to its last w inputs (the
   state at step t = the LSTM run over only the last w read-in outputs from the start state; its own read-out summed
   over steps), against windowed Bayes, whose belief about the other variable reaches back w steps; paired on the same
   12,000 episodes (XPOOL 48,000 - 59,999). Source: esn_learnability/sources_10-27/run_esn/results.json
   "1g_truncation_vs_windowed_bayes" (and "1g_memoryless": reset every step, by step)
b  outcome: each agent's grid at T = 30 (every cell's mean posterior mean over its 50 streams, averaged over the 39
   alias-scan environments), dashed: the true lattice; number: each environment's grid error (mean distance of its
   cells from their true positions), averaged over the environments (11-01: before, the distance of the averaged grid,
   ~half: Optimal .114 vs .236, Trained .124 vs .249, Naive .878 vs 1.77, ESN .866 vs 1.71)
c  mechanism (format): the ESN's frozen state read in different ways. Sources: esn_learnability/theory/claims.json
   ("per_environment_readouts": within one environment, 20 environments x 100 cells x 200 streams, 2-fold CV; own
   read-out 0.437; "routing_ladder": a shared 2-hidden-layer per-step read-out on its frozen states, 20k updates, fresh
   environments; "refuted_claims": a shared linear joint read-out 0.50) and fig1/scratch_code_2026-10-18_19/alias_B/
   analyze_cross_out.txt (exact Bayes on the 5 counts decoded by ridge from the summed state S, environment split:
   R2 0.999, goal accuracy 0.765)
d  where the joint computation matters: the hybrid observer (plot_fig1c_hybrid_unfold, hybrid_sweep100_timed.npz,
   100 environments x 100 cells x 50 streams): joint updates at p% of the stream-steps of every step, the most or the
   least entangled ones, factorized updates elsewhere; grid error = each environment's grid's mean distance from the
   true lattice (cells), averaged over the environments
e  the learning signal: the locked gradflow cross-correlations with entanglement (gradflow_analysis.figure_series,
   1% bands, 20,000 harvest episodes per network, steps 16 - 30): credit coherence and cross-band share of the weight
   update, trained RNN vs ESN
f  over training (the only training-time data so far: the standard e5 runs' test batches every 5 episodes,
   supplementary_harvest.npz): final-step accuracy (raw, and smoothed over 50 batches), and the dimensionality the
   recurrence adds (PR of the LSTM update minus PR of its input; supplementary_plotter2.pr_gain)
Measured (printed by each figure): see the reply of 29 Oct and the print-outs.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig_minimal_set.py \\
        [memory] [readout] [minimal]"""
import os
import sys
import json
import pickle
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402

DATA = os.path.join(ROOT, "main", "DATA", "experimental")
RESULTS = os.path.join(DATA, "esn_learnability", "sources_10-27", "run_esn", "results.json")
CLAIMS = os.path.join(DATA, "esn_learnability", "theory", "claims.json")
SCAN = os.path.join(DATA, "fig1c_compression")
HYBRID = os.path.join(SCAN, "figdata", "round6_hidden", "hybrid_sweep100_timed.npz")
GRADFLOW = os.path.join(DATA, "gradflow_ctx", "figure_series_{}.pkl")
ENVS = [e for e in range(40) if e != 2]
R = 10
COL = {"Joint": "k", "Trained": "#1f77b4", "Naive": "0.55", "Echo": "#ff7f0e", "MLP": "#2ca02c"}
NAME = {"Joint": "Optimal", "Trained": "Trained RNN", "Naive": "Naive", "Echo": "ESN",
        "MLP": "ESN + trained MLPs around it"}
NAIVE_REF, BAYES_REF = .452, .771                          # reference ladder (claims.json "reference_ladder")
ECHO_DECODED_BAYES = .765                                  # analyze_cross_out.txt, Echo:S ridge, Joint(decoded) goal


def _claims():
    c = json.load(open(CLAIMS))
    return {x["id"]: x for x in c} if isinstance(c, list) else c


def truncation():
    r = json.load(open(RESULTS))
    t, m = r["1g_truncation_vs_windowed_bayes"]["by_net"], r["1g_memoryless"]
    ws = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 15])
    net = {"Trained": "LSTMe5", "Echo": "R1e5", "MLP": "E55"}
    out = {k: (np.array([t[v]["net_acc"][str(w)] for w in ws]), t[v]["intact_acc"]) for k, v in net.items()}
    win = np.array([t["E55"]["winbayes_acc"][str(w)] for w in ws])
    steps = {k: (np.array(m[v]["acc_by_step_intact"]), np.array(m[v]["acc_by_step_memoryless"]))
             for k, v in net.items()}
    return ws, win, t["E55"]["bayes_acc"], out, steps


def readouts():
    c = _claims()
    pe, rl = c["per_environment_readouts"]["numbers"], c["routing_ladder"]["numbers"]
    shared = [("its own: linear, per variable", .437, "Echo"),
              ("linear, joint (100 cells)", c["refuted_claims"]["numbers"]["joint_readout_R1_state"], "Echo"),
              ("trained, 2 hidden layers (still rising)", rl["R1_h2_u_20k"], "Echo")]
    within = [("linear, per variable", pe["within_env_lin20"], "Echo"),
              ("linear, joint (100 cells)", pe["within_env_lin100"], "Echo"),
              ("MLP", pe["within_env_mlp20"], "Echo")]
    check = [("exact Bayes on the counts decoded linearly", ECHO_DECODED_BAYES, "Joint")]
    return shared, within, check, pe["within_env_naive"], pe["within_env_joint"]


def grids():
    """each agent's environment-averaged grid at T: (100, 2) posterior means (drawn), and the grid error: each
    environment's own grid's mean distance from the truth, averaged over the environments (audit 11-01: the distance of
    the averaged grid let displacements cancel between environments, ~half)"""
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    cells = meta["cells"].astype(float)
    out = {}
    r = np.arange(R, dtype=np.float32)
    for a in ("Joint", "Trained", "Naive", "Echo"):
        B = np.load(os.path.join(SCAN, f"scan_{a}.npz"))["beliefs"][ENVS][:, :, -1].astype(np.float32)
        P = np.stack([B[..., :R] @ r / B[..., :R].sum(-1), B[..., R:] @ r / B[..., R:].sum(-1)], -1)
        Pe = P.reshape(len(ENVS), R * R, -1, 2).mean(2)                                # each environment's grid
        out[a] = (Pe.mean(0), float(np.linalg.norm(Pe - cells, axis=-1).mean()))          # error per environment, then mean
    return out, cells


def hybrid():
    Z = np.load(HYBRID)
    cm, sw = Z["cm"], Z["sweep"].astype(float)
    cells = np.array([(g, c) for g in range(R) for c in range(R)], float)
    Rv = np.arange(R)
    P = np.stack([(cm[..., :R] * Rv).sum(-1), (cm[..., R:] * Rv).sum(-1)], -1)          # (2, levels, nE, K, 2)
    d = np.linalg.norm(P - cells, axis=-1).mean((-2, -1))                               # (2, levels), per env first
    return sw, d, cm.shape[2]


def gradflow(pct=1):
    out = {}
    for net, lab in (("trained", "Trained"), ("echo", "Echo")):
        s = pickle.load(open(GRADFLOW.format(net), "rb"))[(None, pct)]["r"]
        out[lab] = {k: np.asarray(s[k]) for k in ("coh", "band_x", "l2", "pr")}
    n = len(out["Trained"]["coh"])
    return np.arange(n) - n // 2, out


def training():
    import supplementary_plotter2 as sp
    H = np.load(sp.HARVEST_CACHE)
    out = {}
    for lab, tag in (("Trained", "tr_"), ("Echo", "ec_")):
        A = sp.Bunch({k[len(tag):]: H[k] for k in H.files if k.startswith(tag)})
        out[lab] = {"acc": A.test_acc_through_training[:, -1].astype(np.float64), "pr": np.asarray(sp.pr_gain(A)),
                    "joint": float(np.mean(A.joint_acc)), "naive": float(np.mean(A.naive_acc))}
    return out


def _smooth(x, k):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    c = np.convolve(np.where(ok, x, 0), np.ones(k), "same") / np.maximum(np.convolve(ok, np.ones(k), "same"), 1)
    return np.where(ok, c, np.nan)


# ═══════════════════════════════════════════════════════════════════
# Panels
# ═══════════════════════════════════════════════════════════════════

def panel_truncation(ax, FS, legend=True):
    ws, win, bayes, out, _ = truncation()
    x = np.log2(ws)
    ax.plot(x, win, color="k", lw=2.2, marker="o", ms=4, label="Bayes whose belief reaches back $w$ steps", zorder=4)
    for k in ("Trained", "MLP", "Echo"):
        acc, intact = out[k]
        ax.plot(x, acc, color=COL[k], lw=2, marker="o", ms=4, label=f"{NAME[k]}, memory cut to $w$ steps")
        ax.plot([np.log2(22)], [intact], marker="D", ms=6, color=COL[k], mec="k", mew=.6, ls="")
    ax.axhline(NAIVE_REF, color=COL["Naive"], lw=1.2, ls=(0, (4, 3)))
    ax.text(x[-1] + .1, NAIVE_REF + .008, "naive", fontsize=FS["tick"] - .5, color="0.35", ha="right", va="bottom")
    ax.axhline(bayes, color="k", lw=.8, ls=":")
    ax.text(x[0], bayes + .006, "Bayes, whole episode", fontsize=FS["tick"] - .5, color="0.25", va="bottom")
    ax.set_xticks(list(x) + [np.log2(22)])
    ax.set_xticklabels([str(w) for w in ws] + ["all"], fontsize=FS["tick"])
    ax.set_xlabel("memory: the last $w$ steps (diamonds: intact)", fontsize=FS["tick"] + .5)
    ax.set_ylabel("goal accuracy at $T = 30$", fontsize=FS["tick"] + .5)
    ax.set_ylim(.33, .81)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    if legend:
        ax.legend(fontsize=FS["tick"] - 1.5, frameon=False, loc="lower right", bbox_to_anchor=(1, .27),
                  handlelength=1.6)


def panel_steps(ax, FS):
    _, _, _, _, steps = truncation()
    t = np.arange(1, 31)
    for k in ("Trained", "MLP", "Echo"):
        a, m = steps[k]
        ax.plot(t, a, color=COL[k], lw=2, label=f"{NAME[k]}, intact")
        ax.plot(t, m, color=COL[k], lw=1.6, ls=(0, (4, 2)), label=f"{NAME[k]}, reset every step")
    ax.axhline(NAIVE_REF, color=COL["Naive"], lw=1.2, ls=(0, (4, 3)))
    ax.set_xlabel("step $t$", fontsize=FS["tick"] + .5)
    ax.set_ylabel("goal accuracy", fontsize=FS["tick"] + .5)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS["tick"] - 1.5, frameon=False, loc="upper left", handlelength=1.8)


def panel_readout(ax, FS):
    shared, within, check, naive_w, joint_w = readouts()
    rows = [("shared by all environments", shared), ("fitted within each environment", within),
            ("is the evidence there?", check)]
    y, labels, heads = 0, [], []
    for head, items in rows:
        heads.append((y, head))
        y += 1
        for lab, v, col in items:
            ax.plot([NAIVE_REF, v], [y, y], color=COL[col], lw=1.2, alpha=.5)
            ax.plot([v], [y], marker="o", ms=7, color=COL[col], mec="k", mew=.6)
            ax.text(v + .008, y, f"{v:.2f}", fontsize=FS["tick"] - .5, va="center")
            labels.append((y, lab))
            y += 1
    ax.axvline(NAIVE_REF, color=COL["Naive"], lw=1.2, ls=(0, (4, 3)))
    ax.axvline(BAYES_REF, color="k", lw=.8, ls=":")
    ax.set_yticks([p for p, _ in labels])
    ax.set_yticklabels([lab for _, lab in labels], fontsize=FS["tick"] - .5)
    for p, h in heads:
        ax.text(.385, p, h, fontsize=FS["tick"] - .5, fontweight="bold", va="center", ha="left",
                transform=ax.get_yaxis_transform(), color="0.2")
    ax.set_ylim(y - .4, -.6)
    ax.set_xlim(.38, .81)
    ax.set_xlabel("goal accuracy at $T = 30$ (dashed: naive, dotted: Bayes)", fontsize=FS["tick"] + .5)
    ax.tick_params(labelsize=FS["tick"], axis="x")
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)


def panel_grids(fig, rect, FS):
    G, cells = grids()
    x0, y0, w, h = rect
    edges = np.array([(R * g + c, R * (g + dg) + c + dc) for g in range(R) for c in range(R)
                      for dg, dc in ((1, 0), (0, 1)) if g + dg < R and c + dc < R])
    sub = []
    for i, a in enumerate(("Joint", "Trained", "Naive", "Echo")):
        r_, c_ = divmod(i, 2)
        ax = fig.add_axes([x0 + c_ * w / 2 + .01 * w, y0 + (1 - r_) * h / 2 + .02 * h, .44 * w, .4 * h])
        ax.add_collection(LineCollection(cells[edges], colors="0.82", linewidths=.5, linestyles=(0, (3, 2))))
        P, d = G[a]
        ax.add_collection(LineCollection(P[edges], colors=COL[a] if a != "Joint" else "k", linewidths=1.1))
        ax.set_xlim(-.7, 9.7), ax.set_ylim(-.7, 9.7), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.75")
        ax.set_title(f"{NAME[a]}  ({d:.2f} cells)", fontsize=FS["tick"], pad=2, color="k" if a == "Joint" else COL[a])
        sub.append(ax)
    return sub


def panel_hybrid(ax, FS):
    sw, d, nE = hybrid()
    for w, lab, col in ((1, "joint updates at the MOST entangled steps", "#d7301f"),
                        (0, "joint updates at the LEAST entangled steps", "#3f88c5")):
        ax.plot(sw, d[w], color=col, lw=2.2, label=lab)
    i25 = int(np.argmin(np.abs(sw - 24)))
    for w, col in ((1, "#d7301f"), (0, "#3f88c5")):
        ax.plot([sw[i25]], [d[w, i25]], marker="o", ms=6, color=col, mec="k", mew=.6)
    ax.annotate(f"{sw[i25]:.0f}% of steps:\n{d[0, i25]:.2f} vs {d[1, i25]:.2f} cells", (sw[i25], d[1, i25]),
                xytext=(sw[i25] + 12, d[1, i25] + .25), fontsize=FS["tick"] - .5,
                arrowprops=dict(arrowstyle="-", color="0.4", lw=.7))
    ax.set_xlabel("% of each step's stream-steps updated jointly (the rest factorized)", fontsize=FS["tick"] + .5)
    ax.set_ylabel("grid error at $T$ (cells)", fontsize=FS["tick"] + .5)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS["tick"] - 1.5, frameon=False, loc="upper right")
    ax.text(1.5, d[0, 0] + .02, "Naive", fontsize=FS["tick"] - 1, ha="left", va="bottom", color="0.4")
    ax.text(100, d[0, -1], " Optimal", fontsize=FS["tick"] - 1, ha="left", va="center", color="0.2")
    return nE


def panel_gradflow(ax, FS):
    lags, G = gradflow()
    for k, ls, lab in (("coh", "-", "credit coherence"), ("band_x", (0, (4, 2)), "cross-band share")):
        for net in ("Trained", "Echo"):
            ax.plot(lags, G[net][k], color=COL[net], ls=ls, lw=2, marker="o", ms=2.5,
                    label=f"{NAME[net]}: {lab}")
    ax.axhline(0, color="0.6", lw=.8, ls=":")
    ax.axvline(0, color="0.3", lw=.6, ls=":")
    ax.set_xlabel(r"lag $\tau$ (steps after the entangled update)", fontsize=FS["tick"] + .5)
    ax.set_ylabel(r"$r$ with entanglement (within episodes)", fontsize=FS["tick"] + .5)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS["tick"] - 1.5, frameon=False, loc="upper right")
    return G


def panel_training(ax, FS):
    T = training()
    for lab in ("Trained", "Echo"):
        ep = (np.arange(len(T[lab]["acc"])) + 1) * 5
        ax.plot(ep, T[lab]["acc"], color=COL[lab], lw=.4, alpha=.25)
        ax.plot(ep, _smooth(T[lab]["acc"], 51), color=COL[lab], lw=2, label=NAME[lab])
    naive = np.mean([T[k]["naive"] for k in T])
    joint = np.mean([T[k]["joint"] for k in T])
    ax.axhline(naive, color=COL["Naive"], lw=1.2, ls=(0, (4, 3)))
    ax.axhline(joint, color="k", lw=.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("training episode", fontsize=FS["tick"] + .5)
    ax.set_ylabel("goal accuracy at $T$ (test batches)", fontsize=FS["tick"] + .5)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="lower right")
    a = _smooth(T["Trained"]["acc"], 51)
    first = (int(np.flatnonzero(a > naive)[0]) + 1) * 5
    ax.annotate(f"passes naive\nat ~{first:,} episodes", (first, naive), xytext=(first * 2.2, naive - .17),
                fontsize=FS["tick"] - 1, arrowprops=dict(arrowstyle="-", color="0.4", lw=.7))
    ax2 = ax.inset_axes([.08, .5, .4, .33])
    for lab in ("Trained", "Echo"):
        ep = (np.arange(len(T[lab]["pr"])) + 1) * 5
        ax2.plot(ep, _smooth(T[lab]["pr"], 51), color=COL[lab], lw=1.5)
    ax2.axhline(0, color="0.6", lw=.6, ls=":")
    ax2.set_xscale("log")
    ax2.tick_params(labelsize=FS["tick"] - 2, length=2, pad=1)
    ax2.set_title("dimensions the recurrence adds", fontsize=FS["tick"] - 1.5, pad=2)
    return T, naive, joint


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

def figure_memory():
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    W_, H_ = 11.0, 4.6
    fig = plt.figure(figsize=(W_, H_))
    fig.text(.5, 1 - .28 / H_, "Beating naive Bayes takes a belief about the other variable built from history: the "
             "trained RNN builds it, the ESN never does", ha="center", va="center", fontsize=FS["label"] + 2)
    fig.text(.5, 1 - .58 / H_, "12,000 test episodes, the same for every curve; the ESN + MLPs: frozen LSTM with "
             "5 trained layers before and after it (one run)", ha="center", va="center", fontsize=FS["tick"])
    ax = fig.add_axes([.75 / W_, .7 / H_, 4.4 / W_, 3.1 / H_])
    panel_truncation(ax, FS)
    ax.set_title("a   Each network with its memory cut to $w$ steps", fontsize=FS["label"] + .5, loc="left", pad=5)
    ax = fig.add_axes([6.25 / W_, .7 / H_, 4.4 / W_, 3.1 / H_])
    panel_steps(ax, FS)
    ax.set_title("b   Intact vs reset at every step, by step", fontsize=FS["label"] + .5, loc="left", pad=5)
    path = fig_path("fig_mechanism_memory.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    ws, win, bayes, out, steps = truncation()
    print("saved", path)
    print("w", ws.tolist(), "windowed Bayes", np.round(win, 3).tolist(), "full Bayes", bayes)
    for k, (a, i) in out.items():
        print(k, np.round(a, 3).tolist(), "intact", round(i, 3), "| reset each step at T", round(steps[k][1][-1], 3))
    return path


def figure_readout():
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    W_, H_ = 8.4, 4.4
    fig = plt.figure(figsize=(W_, H_))
    fig.text(.5, 1 - .28 / H_, "The ESN's frozen state holds the evidence; only its own linear per-variable read-out "
             "is stuck at naive", ha="center", va="center", fontsize=FS["label"] + 2)
    ax = fig.add_axes([3.05 / W_, .7 / H_, 5.0 / W_, 3.15 / H_])
    panel_readout(ax, FS)
    fig.text(.5, .12 / H_, "within-environment fits: 20 environments x 100 cells x 200 streams (naive .444, Bayes "
             ".774 there); shared read-outs: fresh environments", ha="center", va="center", fontsize=FS["tick"] - .5)
    path = fig_path("fig_mechanism_esn_readout.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("saved", path)
    for grp in readouts()[:3]:
        print([(lab, round(v, 3)) for lab, v, _ in grp])
    return path


def figure_minimal():
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    W_, H_ = 17.0, 10.2
    fig = plt.figure(figsize=(W_, H_))
    fig.text(.5, 1 - .3 / H_, "Why the trained RNN reaches optimal Bayes and the ESN stays at naive Bayes: the "
             "minimal set from existing data", ha="center", va="center", fontsize=FS["label"] + 3)
    pw, ph = 4.3, 3.2
    xs = (.9, 6.55, 12.3)
    yt, yb = 5.75, .9

    def title(ax, s):
        ax.set_title(s, fontsize=FS["label"] + .5, loc="left", pad=6)

    ax = fig.add_axes([xs[0] / W_, yt / H_, pw / W_, ph / H_])
    panel_truncation(ax, FS)
    title(ax, "a   Mechanism: the belief is built from history")
    # b: grids (four small axes in the slot)
    slot = [xs[1] / W_, yt / H_, pw / W_, ph / H_]
    sub = panel_grids(fig, slot, FS)
    sub[0].text(0, 1.22, "b   Outcome: unfolded vs folded grid at $T$", transform=sub[0].transAxes,
                fontsize=FS["label"] + .5, ha="left", va="bottom")
    ax = fig.add_axes([(xs[2] + 1.4) / W_, yt / H_, (pw - 1.4) / W_, ph / H_])
    panel_readout(ax, FS)
    title(ax, "c   Mechanism: the ESN's read-out format")
    ax.title.set_position((-.75, 1))
    ax = fig.add_axes([xs[0] / W_, yb / H_, pw / W_, ph / H_])
    nE = panel_hybrid(ax, FS)
    title(ax, "d   Where the joint update matters")
    ax = fig.add_axes([xs[1] / W_, yb / H_, pw / W_, ph / H_])
    G = panel_gradflow(ax, FS)
    title(ax, "e   Learning signal on entangled steps")
    ax = fig.add_axes([xs[2] / W_, yb / H_, pw / W_, ph / H_])
    T, naive, joint = panel_training(ax, FS)
    title(ax, "f   Over training (the old runs' test logs)")
    notes = ("a: 12,000 episodes; memory cut to the last w read-in outputs; diamonds: intact networks.   "
             "b: 39 environments x 100 cells x 50 streams, grids averaged over environments; number: each "
             "environment's grid error, averaged.   c: the ESN's frozen state read out in other ways.",
             f"d: hybrid Bayes observer, {nE} environments x 100 cells x 50 streams.   e: 20,000 test episodes per "
             "network, steps 16 - 30, 1% bands (the locked gradflow figure).   f: test batches every 5 episodes of "
             "the standard runs; inset: PR(update) - PR(input).")
    for i, n in enumerate(notes):
        fig.text(.9 / W_, (.42 - .22 * i) / H_, n, fontsize=FS["tick"] - 1, ha="left", va="center", color="0.25")
    path = fig_path("fig_minimal_set.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("saved", path)
    sw, d, _ = hybrid()
    i25 = int(np.argmin(np.abs(sw - 24)))
    print(f"hybrid at {sw[i25]:.0f}%: least {d[0, i25]:.3f}, most {d[1, i25]:.3f}; Naive {d[0, 0]:.3f}, Optimal "
          f"{d[0, -1]:.3f}")
    lags = np.arange(len(G["Trained"]["coh"])) - len(G["Trained"]["coh"]) // 2
    i0 = int(np.flatnonzero(lags == 0)[0])
    print("gradflow tau 0:", {n: {k: round(float(v[i0]), 3) for k, v in G[n].items()} for n in G})
    for lab in T:
        a = _smooth(T[lab]["acc"], 51)
        first = np.flatnonzero(a > naive)
        print(lab, "final smoothed acc", round(float(a[-200:].mean()), 3), "first above naive at episode",
              int((first[0] + 1) * 5) if len(first) else None)
    print("grids:", {k: round(v[1], 2) for k, v in grids()[0].items()})
    return path


if __name__ == "__main__":
    args = sys.argv[1:] or ["memory", "readout", "minimal"]
    if "memory" in args:
        figure_memory()
    if "readout" in args:
        figure_readout()
    if "minimal" in args:
        figure_minimal()
