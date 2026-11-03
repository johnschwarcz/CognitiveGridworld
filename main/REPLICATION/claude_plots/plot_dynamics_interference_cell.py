"""The activity half of dynamics_interference for the LSTM's long-term (cell) stream c_t, next to its output stream u_t
(25 Oct 2026) -> main/REPLICATION/figure_bin/dynamics_interference_cell_1pct_xcorr.png

The statistic and conventions of gradflow_analysis.plot (the locked dynamics_interference figure): each episode's
series over the error steps 17..30 (gradflow_analysis.SRC) and its entanglement, each z-scored within the episode, the
product a[t + tau] * b[t] averaged within and then over episodes at lags tau = -8..8 (tau > 0: after the entangled
step); nothing normalized per lag. Series, for x = c_t and x = u_t:
    norm ||x_t||; effective dimension (sum_pc E_pc)^2 / sum_pc E_pc^2 with E_pc = proj_pc(x_t - mu)^2 on a fixed PCA
    basis of x; turn 1 - cos(x_t, x_{t-1}) (x_0 = the learned initial state); and each band's share of the step's energy
    sum_{pc in B} E_pc / sum_pc E_pc, bands of 1% of x's variance (gradflow_analysis.cut_segments; a PC holding more
    than 1% leaves the following bands empty, grey).
Data: the hidden states of research8/networks/replay.py (main/DATA/experimental/fig1c_compression/hidden_<net>_<u|c>.npy:
the first 20,000 episodes of the trained network's gradflow harvest, the ESN on the same episodes; the locked figure
used each network's own episodes) and the harvest's entanglement rev. PCA bases: 200,000 random (episode, step)
samples of each array. Check: the trained network's u curves against the locked figure's cache.
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg OMP_NUM_THREADS=6 /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_dynamics_interference_cell.py [--rebuild]"""
import os
import sys
import pickle
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from main.utils import apply_fig_style, fig_path                                  # noqa: E402

FS = apply_fig_style()
plt.rcParams["mathtext.fontset"] = "cm"
BIG = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
DATA = os.path.join(BIG, "figdata", "round4b", "dynamics_interference_cell.npz")
READOUT = ("/tmp/claude-1002/-home-johns-Desktop-CognitiveGridworld/7af6b091-54ff-45a0-b4dd-ef0a3bc9c778/scratchpad/"
           "research8/networks/work/readout_{}.npz")                               # initial states u0, c0
NETS = (("trained", "Trained RNN", "#1f77b4"), ("echo", "ESN", "#ff7f0e"))
N = 20000
NB = 100


def build():
    import gradflow_analysis as ga
    import fig1 as cf
    s = slice(ga.SRC[0], ga.SRC[-1] + 1)
    rev = cf._harvest("trained", ["rev"])["rev"][:N].astype(np.float64)
    b = ga._zs(rev[:, s])
    r = lambda y: ga._lagged(ga._zs(y), b)
    pct = lambda y: 100 * ga._lagged(y - y.mean(1, keepdims=True), b) / np.median(y)   # change per SD of entanglement,
                                                                                         # % of the series' median
    rng = np.random.default_rng(0)
    out = {"tau": ga.TAU}
    for net, _, _ in NETS:
        r0 = np.load(READOUT.format(net))
        for var in ("c", "u"):
            X = np.load(os.path.join(BIG, f"hidden_{net}_{var}.npy"), mmap_mode="r")
            pick = np.sort(rng.choice(N * 30, 200000, replace=False))
            S = np.asarray(X.reshape(N * 30, -1)[pick], np.float64)
            mu = S.mean(0)
            ev, V = np.linalg.eigh(np.cov(S, rowvar=False))
            ev, V = ev[::-1], V[:, ::-1]
            evr = ev / ev.sum()
            sid = ga.cut_segments(evr, NB)
            OH = np.eye(NB, dtype=np.float32)[sid]
            l2, pr, turn = (np.zeros((N, 30)) for _ in range(3))
            Eb = np.zeros((N, 30, NB), np.float32)
            prev = np.broadcast_to(r0[f"{var}0"].astype(np.float32), (N, X.shape[-1]))
            Vf, muf = V.astype(np.float32), mu.astype(np.float32)
            for t in range(30):
                x = np.asarray(X[:, t], np.float32)
                l2[:, t] = np.linalg.norm(x, axis=1)
                turn[:, t] = 1 - (x * prev).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(prev, axis=1))
                E = ((x - muf) @ Vf) ** 2
                pr[:, t] = E.sum(1).astype(np.float64) ** 2 / (E.astype(np.float64) ** 2).sum(1)
                Eb[:, t] = E @ OH
                prev = x
            sh = Eb[:, s].astype(np.float64)
            sh = sh / sh.sum(-1, keepdims=True)
            tag = f"{net}_{var}"
            out[tag + "_l2"], out[tag + "_pr"], out[tag + "_turn"] = r(l2[:, s]), r(pr[:, s]), r(turn[:, s])
            out[tag + "_l2_pct"], out[tag + "_pr_pct"], out[tag + "_turn_pct"] = pct(l2[:, s]), pct(pr[:, s]), pct(turn[:, s])
            out[tag + "_bands"] = r(sh)                                           # (TAU, NB)
            out[tag + "_npc"] = np.bincount(sid, minlength=NB)
            out[tag + "_size"] = np.array([np.median(l2[:, s]), np.median(pr[:, s]), np.median(turn[:, s])])
            out[tag + "_evr_top"] = evr[:5]
            print(net, var, "done", flush=True)
            del X, S, Eb
    # check against the locked figure's cache (trained, u, 1% bands)
    with open(ga.FIG_CACHE.format("trained"), "rb") as f:
        store = pickle.load(f)
    if (None, 1) in store:
        ref = store[(None, 1)]["r"]
        out["check_l2"] = np.abs(out["trained_u_l2"] - ref["l2"]).max()
        out["check_turn"] = np.abs(out["trained_u_turn"] - ref["turn"]).max()
        out["check_pr"] = np.abs(out["trained_u_pr"] - ref["pr"]).max()
    np.savez(DATA, **out)
    print("saved", DATA)


def figure():
    Z = np.load(DATA)
    tau = Z["tau"]
    pw, ph, gx = 3.0, 2.0, .7
    hw, hh = 2.15, 2.4
    W = .8 + 4 * hw + 3 * .45 + .2 + .12 + .55
    H = .95 + ph + 1.05 + hh + .95
    fig = plt.figure(figsize=(W, H))
    tops = (("l2", "Norm", r"$\|x_t\|$"),
            ("pr", "Effective dim.", r"$(\Sigma_{pc} E_{pc})^2 / \Sigma_{pc} E_{pc}^2$, $E_{pc} = \mathrm{proj}_{pc}^2$"),
            ("turn", "Activity turn", r"$1 - \cos(x_t, x_{t-1})$"))
    lo = min(Z[f"{n}_{v}_{k}"].min() for n, _, _ in NETS for v in ("c", "u") for k, _, _ in tops)
    hi = max(Z[f"{n}_{v}_{k}"].max() for n, _, _ in NETS for v in ("c", "u") for k, _, _ in tops)
    span = W - .8 - .55
    tw = (span - 2 * gx) / 3
    for k_, (key, name, math) in enumerate(tops):
        ax = fig.add_axes([(.8 + k_ * (tw + gx)) / W, (H - .95 - ph) / H, tw / W, ph / H])
        for net, lab, col in NETS:
            for var, ls, lw in (("c", "-", 1.6), ("u", (0, (3, 2)), 1.1)):
                y = Z[f"{net}_{var}_{key}"]
                ax.plot(tau[tau <= 0], y[tau <= 0], c=col, ls=":", lw=lw, alpha=.5)
                ax.plot(tau[tau >= 0], y[tau >= 0], c=col, ls=ls, lw=lw)
        ax.axhline(0, c="k", alpha=.3, lw=.8), ax.axvline(0, c="k", ls="--", alpha=.4, lw=.8)
        ax.set_ylim(lo - .08 * (hi - lo), hi + .08 * (hi - lo))
        ax.set_xticks(range(tau[0], tau[-1] + 1, 4))
        ax.set_xlabel(r"Lag $\tau$ (after the entangled step $>$ 0)", labelpad=1)
        if k_ == 0:
            ax.set_ylabel("r (cross-correlation)")
        else:
            ax.tick_params(labelleft=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.annotate(name, (.5, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points", ha="center",
                    va="baseline", fontsize=FS["label"] + .5)
        ax.annotate(math, (.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points", ha="center",
                    va="baseline", fontsize=FS["tick"] - .5, color="0.25")
        i0 = int(np.flatnonzero(tau == 0)[0])
        sizes = "\n".join("%s: $c$ %+.1f%% / %+.1f%%, $u$ %+.1f%% / %+.1f%%" % (
            lab, *Z[f"{net}_c_{key}_pct"][[i0, i0 + 1]], *Z[f"{net}_u_{key}_pct"][[i0, i0 + 1]]) for net, lab, _ in NETS)
        ax.text(.02, .03, r"per SD of entanglement, $\tau$ = 0 / +1 (% of median):" + "\n" + sizes,
                transform=ax.transAxes, fontsize=FS["tick"] - 2.5, color="0.3", va="bottom", linespacing=1.3)
        fig.text((.8 + k_ * (tw + gx) - .55) / W, (H - .95 + 17 / 72) / H, r"$\mathbf{%s}$" % "abc"[k_],
                 fontsize=FS["label"] + 2, va="baseline")
    # band heat maps
    vmax = max(np.abs(Z[f"{n}_{v}_bands"]).max() for n, _, _ in NETS for v in ("c", "u"))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.85")
    y0 = .95
    for k_, (net, var) in enumerate((("trained", "c"), ("echo", "c"), ("trained", "u"), ("echo", "u"))):
        ax = fig.add_axes([(.8 + k_ * (hw + .45)) / W, y0 / H, hw / W, hh / H])
        M = Z[f"{net}_{var}_bands"].T.copy()                                       # (NB, TAU)
        M[Z[f"{net}_{var}_npc"] == 0] = np.nan
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest",
                       extent=[tau[0] - .5, tau[-1] + .5, NB - .5, -.5])
        ax.axvline(0, c="k", ls="--", alpha=.4, lw=.8)
        ax.set_xticks(range(tau[0], tau[-1] + 1, 4))
        ax.set_yticks([0, 49, 99], ["1", "50", "100"])
        ax.set_xlabel(r"Lag $\tau$", labelpad=1)
        if k_ == 0:
            ax.set_ylabel("Band (1% of the variance each; 1 = top PCs)")
        else:
            ax.tick_params(labelleft=False)
        lab = dict((n, l) for n, l, _ in NETS)[net]
        ax.set_title(f"{lab}, {'cell $c_t$' if var == 'c' else 'output $u_t$'}", fontsize=FS["label"], pad=3)
        if k_ == 0:
            fig.text(.8 / W, (y0 + hh + .3) / H, r"Each band's share of the step's energy, $\Sigma_{pc \in B} E_{pc} / "
                     r"\Sigma_{pc} E_{pc}$, cross-correlation with entanglement (grey: no PCs; top PC of $c$: %.0f%% "
                     r"Trained, %.0f%% ESN)" % (100 * Z["trained_c_evr_top"][0], 100 * Z["echo_c_evr_top"][0]),
                     fontsize=FS["label"] + .5, va="baseline")
            fig.text(.18 / W, (y0 + hh + .3) / H, r"$\mathbf{d}$", fontsize=FS["label"] + 2, va="baseline")
    cx = fig.add_axes([(.8 + 4 * hw + 3 * .45 + .2) / W, y0 / H, .12 / W, hh / H])
    cb = fig.colorbar(im, cax=cx)
    cb.set_label("r", labelpad=2)
    cb.outline.set_visible(False)
    hand = [Line2D([], [], c=col, lw=1.4, label=lab) for _, lab, col in NETS]
    hand += [Line2D([], [], c="0.3", lw=1.6, label="cell (long-term) stream $c_t$"),
             Line2D([], [], c="0.3", lw=1.1, ls=(0, (3, 2)), label="output stream $u_t$"),
             Line2D([], [], c="0.3", lw=1, ls=":", alpha=.5, label=r"before the entangled step ($\tau < 0$)")]
    fig.legend(handles=hand, loc="upper center", bbox_to_anchor=(.5, 1 - .04 / H), ncol=5, frameon=False,
               fontsize=FS["tick"], handlelength=1.8, columnspacing=1.4)
    chk = ""
    if "check_l2" in Z.files:
        chk = ("\nCheck vs the locked figure (Trained, $u$): largest difference: norm %.0e, turn %.0e, effective dim. "
               "%.3f (its PCA basis differs)." % (Z["check_l2"], Z["check_turn"], Z["check_pr"]))
    fig.text(.8 / W, .45 / H, "The statistic of the locked dynamics-interference figure: each series and the entanglement z-scored "
             "within the episode over steps 17-30, averaged lagged products; 20,000 episodes of the trained network's "
             "harvest (the ESN on the same episodes)." + chk, fontsize=FS["tick"] - 1.5, color="0.35", va="top",
             linespacing=1.4)
    path = fig_path("dynamics_interference_cell_1pct_xcorr.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if "--rebuild" in sys.argv[1:] or not os.path.exists(DATA):
        build()
    print("saved", figure())
    Z = np.load(DATA)
    i0 = int(np.flatnonzero(Z["tau"] == 0)[0])
    for net, lab, _ in NETS:
        for var in ("c", "u"):
            print("%-11s %s: r at tau -1/0/+1/+2: norm %s | eff. dim %s | turn %s | medians %s" % (
                lab, var, np.round(Z[f"{net}_{var}_l2"][i0 - 1:i0 + 3], 3), np.round(Z[f"{net}_{var}_pr"][i0 - 1:i0 + 3], 3),
                np.round(Z[f"{net}_{var}_turn"][i0 - 1:i0 + 3], 3), np.round(Z[f"{net}_{var}_size"], 3)))
    for k in ("check_l2", "check_turn", "check_pr"):
        if k in Z.files:
            print(k, float(Z[k]))
