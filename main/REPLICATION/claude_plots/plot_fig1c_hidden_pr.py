"""Participation ratio of the networks' hidden states vs that of their updates (25 Oct 2026)
-> main/REPLICATION/figure_bin/fig1c_hidden_pr.png

Hidden states: the Trained RNN and the ESN (an LSTM each, 1000 units) replayed on the trained network's gradflow
harvest, first 20,000 episodes (the ESN on the same episodes), stored per step as float16 (20000, 30, 1000) in
main/DATA/experimental/fig1c_compression/hidden_<net>_<u|c>.npy (research8/networks/replay.py; checks: the read-out of
the stored u reproduces the networks' beliefs to < 2e-5). u_t = o_t * tanh(c_t) is what the read-out reads; c_t is the
cell (memory) state. Updates: du_t = u_t - u_{t-1}, dc_t = c_t - c_{t-1}, with u_0, c_0 the learned initial states
(constant across episodes, so at t = 1 the update's covariance is the state's).
Episodes: the 19,607 of 1,225 distinct-variable environments, pooled over environments.
PR = (tr C)^2 / tr(C^2) of the covariance C over episodes at each step; 'all steps' = the 30 steps' within-step
covariances summed (each step centred on its own mean, so the time course of the mean does not count as a dimension).
Split halves: the same on two random halves of the environments (stability).
Per-step values are stored in figdata/round4b/hidden_pr.npz (rebuilt if missing or with --rebuild, about 2 min).
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg OMP_NUM_THREADS=6 /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_hidden_pr.py [--rebuild]"""
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
sys.path[:0] = [p for p in (ROOT, os.path.join(ROOT, "main", "REPLICATION", "claude_plots")) if p not in sys.path]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from main.utils import apply_fig_style, fig_path                                  # noqa: E402

FS = apply_fig_style()
plt.rcParams["mathtext.fontset"] = "cm"
BIG = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
DATA = os.path.join(BIG, "figdata", "round4b", "hidden_pr.npz")
READOUT = ("/tmp/claude-1002/-home-johns-Desktop-CognitiveGridworld/7af6b091-54ff-45a0-b4dd-ef0a3bc9c778/scratchpad/"
           "research8/networks/work/readout_{}.npz")                               # initial states u0, c0 of the replay
NETS = (("trained", "Trained RNN", "#1f77b4"), ("echo", "ESN", "#ff7f0e"))
T = 30


def _pr(C):
    return np.trace(C) ** 2 / (C * C).sum()


def _covs(X, s0):
    """X (n, T, H) float16 states; s0 (H,) initial state -> per-step covariances of the state and of its update,
    (T, H, H) each, float64"""
    n, T_, H = X.shape
    Cs, Cd = np.zeros((T_, H, H)), np.zeros((T_, H, H))
    prev = np.broadcast_to(s0.astype(np.float32), (n, H))
    for s in range(T_):
        x = X[:, s].astype(np.float32)
        for C, v in ((Cs, x), (Cd, x - prev)):
            v = v - v.mean(0)
            C[s] = (v.T @ v).astype(np.float64) / (n - 1)
        prev = x
    return Cs, Cd


def build():
    import fig1 as cf
    ci = np.sort(cf._harvest("trained", ["ep_ctx_inds"])["ep_ctx_inds"][:20000].astype(int), 1)
    keep = ci[:, 0] != ci[:, 1]
    env = np.unique(ci[keep], axis=0, return_inverse=True)[1].ravel()
    half = np.random.default_rng(0).permutation(env.max() + 1) % 2                # environment halves
    out = {}
    for net, _, _ in NETS:
        r0 = np.load(READOUT.format(net))
        for var, s0 in (("u", r0["u0"]), ("c", r0["c0"])):
            X = np.load(os.path.join(BIG, f"hidden_{net}_{var}.npy"))[keep]
            Cs, Cd = _covs(X, s0)
            out[f"{net}_{var}"] = np.array([_pr(C) for C in Cs])
            out[f"{net}_d{var}"] = np.array([_pr(C) for C in Cd])
            out[f"{net}_{var}_all"], out[f"{net}_d{var}_all"] = _pr(Cs.sum(0)), _pr(Cd.sum(0))
            out[f"{net}_{var}_tr"] = np.array([np.trace(C) for C in Cs])
            out[f"{net}_d{var}_tr"] = np.array([np.trace(C) for C in Cd])
            for h in (0, 1):
                Hs, Hd = _covs(X[half[env] == h], s0)
                out[f"{net}_{var}_half{h}"] = np.array([_pr(C) for C in Hs])
                out[f"{net}_d{var}_half{h}"] = np.array([_pr(C) for C in Hd])
            print(net, var, "done", flush=True)
            del X, Cs, Cd
    out["n_episodes"], out["n_env"] = keep.sum(), env.max() + 1
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez(DATA, **out)
    print("saved", DATA)


def figure():
    Z = np.load(DATA)
    t = np.arange(1, T + 1)
    pw, ph, gap = 2.9, 2.0, .95
    W, H = .8 + 3 * pw + 2 * gap + .3, .95 + ph + .85
    fig = plt.figure(figsize=(W, H))
    for k, (var, name, formula) in enumerate((("u", "Hidden output state", r"$u_t = o_t \odot \tanh(c_t)$ (read by the read-out)"),
                                              ("c", "Cell (memory) state", r"$c_t$"))):
        ax = fig.add_axes([(.8 + k * (pw + gap)) / W, .85 / H, pw / W, ph / H])
        for net, lab, col in NETS:
            for pre, ls in (("", "-"), ("d", "--")):
                y = Z[f"{net}_{pre}{var}"]
                lo = np.minimum(Z[f"{net}_{pre}{var}_half0"], Z[f"{net}_{pre}{var}_half1"])
                hi = np.maximum(Z[f"{net}_{pre}{var}_half0"], Z[f"{net}_{pre}{var}_half1"])
                ax.fill_between(t, lo, hi, color=col, alpha=.15, lw=0)
                ax.plot(t, y, c=col, ls=ls, lw=1.4)
        ax.set_xlim(.5, T + .5), ax.set_xticks([1, 10, 20, 30])
        ax.set_ylim(0, None)
        ax.set_xlabel("Step $t$", labelpad=1)
        ax.set_ylabel("Effective dimensions (PR)")
        ax.spines[["top", "right"]].set_visible(False)
        _deco(fig, ax, f"{name}: state vs update", formula + r"; $PR = (\mathrm{tr}\,C)^2 / \mathrm{tr}(C^2)$", "ab"[k], W, H)
    # (c) all steps together
    bx = fig.add_axes([(.8 + 2 * (pw + gap)) / W, .85 / H, pw / W, ph / H])
    xs, labs = [], []
    for g, var in enumerate(("u", "c")):
        for n_, (net, lab, col) in enumerate(NETS):
            x0 = g * 2.6 + n_ * 1.1
            for b, (pre, alpha, hatch) in enumerate((("", 1, None), ("d", .45, "////"))):
                v = float(Z[f"{net}_{pre}{var}_all"])
                bx.bar(x0 + (b - .5) * .46, v, width=.44, color=col, alpha=alpha, hatch=hatch, edgecolor="white", lw=0)
                bx.text(x0 + (b - .5) * .46, v + 1.5, "%.0f" % v, ha="center", va="bottom", fontsize=FS["tick"] - 2,
                        color="0.25")
            xs.append(x0), labs.append(lab)
    bx.set_xticks(xs, labs, fontsize=FS["tick"] - 1)
    for g, var in enumerate(("$u$", "$c$")):
        bx.text(g * 2.6 + .55, -.2, var, transform=bx.get_xaxis_transform(), ha="center", va="top", fontsize=FS["label"])
    bx.set_ylim(0, None)
    bx.set_ylabel("Effective dimensions (PR)")
    bx.spines[["top", "right"]].set_visible(False)
    _deco(fig, bx, "All 30 steps", "within-step covariances summed", "c", W, H)
    hand = [Line2D([], [], c=col, lw=1.4, label=lab) for _, lab, col in NETS]
    hand += [Line2D([], [], c="0.3", lw=1.4, label="state (bars: solid)"),
             Line2D([], [], c="0.3", lw=1.4, ls="--", label="update, state$_t$ - state$_{t-1}$ (bars: hatched)"),
             plt.Rectangle((0, 0), 1, 1, color="0.6", alpha=.3, label="range over two halves of the environments")]
    fig.legend(handles=hand, loc="upper center", bbox_to_anchor=(.5, 1 - .04 / H), ncol=5, frameon=False,
               fontsize=FS["tick"], handlelength=1.8, columnspacing=1.4)
    fig.text(.8 / W, .2 / H, "Hidden states of the two LSTMs (1000 units) on the same %s episodes from %s environments, "
             "pooled; covariance over episodes at each step." % (f"{int(Z['n_episodes']):,}", f"{int(Z['n_env']):,}"),
             fontsize=FS["tick"] - 1, color="0.35")
    path = fig_path("fig1c_hidden_pr.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _deco(fig, ax, name, formula, letter, W, H):
    ax.annotate(name, (.5, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points", ha="center",
                va="baseline", fontsize=FS["label"] + .5)
    ax.annotate(formula, (.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points", ha="center",
                va="baseline", fontsize=FS["tick"] - .5, color="0.25")
    x0, y0, w, h = ax.get_position().bounds
    fig.text(x0 - .62 / W, y0 + h + 17 / 72 / H, r"$\mathbf{%s}$" % letter, fontsize=FS["label"] + 2, va="baseline")


if __name__ == "__main__":
    if "--rebuild" in sys.argv[1:] or not os.path.exists(DATA):
        build()
    print("saved", figure())
    Z = np.load(DATA)
    for net, lab, _ in NETS:
        for var in ("u", "c"):
            print("%-11s %s: PR state t=1/5/10/20/30 %s | update %s | all steps state %.1f update %.1f" % (
                lab, var, np.round(Z[f"{net}_{var}"][[0, 4, 9, 19, 29]], 1), np.round(Z[f"{net}_d{var}"][[0, 4, 9, 19, 29]], 1),
                Z[f"{net}_{var}_all"], Z[f"{net}_d{var}_all"]))
