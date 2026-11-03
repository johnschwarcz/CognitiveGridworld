"""Training-dynamics figures from td_analysis.py's per-checkpoint files (John 10-31: "lets get the training analyses").
Every quantity is measured on the same 10,000 probe episodes (50 environments x 100 cells x 2 streams, test variable
pairs) at every checkpoint of a run, and drawn against training episodes (episodes with an update: a test episode
trains nothing, so checkpoint E holds E - E // 5) on a log axis; the initial network sits at the left edge ("0").
The per-checkpoint summaries are cached in analysis/series_<run>.npz (recomputed for new checkpoints).
-> main/REPLICATION/figure_bin/ (legends: colour = network, grey line style = measure)
  training_dynamics_overview.png  one page: accuracy at T (Bayes, memory-free Bayes, naive); PR of u, W_hh u, W_ih x (per
                           step, then averaged over steps) and the share of the variance that is movement over steps
                           within episodes for u, W_hh u, c (td_time_within.py); u's variance
                           at step 30 by factor and the PR each factor induces; the locked statistic's r (tau = 0) of
                           the activity and update measures with entanglement and with FR (both variables averaged)
                           (factor panels: colour = factor, line style = network)
  training_dynamics_overview_vs_accuracy.png  the same against each network's own accuracy at T (a: the training
                           episodes it took to reach it)
  td_locked_measures.png   the locked figure's five measures (activity norm and eff. dim.; credit coherence, update
                           energy, cross-band share): medians through training and their r with entanglement (tau = 0)
  td_learning.png        a gap to Bayes at T (log) intact and with the memory cut to 1, 2, 5 steps (an6), with the
                           windowed-Bayes and naive references; b accuracy at T by the goal variable's own
                           factorization regret FR at T (quintiles; Bayes and naive per quintile at the right edge);
                           c the per-step logit increment on the Bayes ladder: coefficients on N (naive), W (within
                           step), M (memory) and R^2 (an1); d forget-gate time constants tau = 1 / (1 - mean f) per
                           unit: median and 99th percentile; e how often the network's answer at T equals naive's,
                           Bayes's, and naive's answer where naive is wrong and Bayes right (copying naive's errors);
                           f accuracy at T
  td_factors.png         a, b variance of u at step 30 explained by each factor (environment = Z, r0, r1, their joint,
                           env x cell, streams), Trained and Echo; c the dimension (PR) each factor induces in u;
                           d the share of the variance that is movement over steps within episodes and e the PR per
                           step (averaged over steps), for the read-in output x, the input drive W_ih x, u, the
                           recurrent drive W_hh u and c; f Trained:
                           the PR each factor induces in the input vs the recurrent drive
  td_entanglement.png    the locked statistic (gradflow_analysis _zs / _lagged, steps 17-30; tau = 0 is the mean
                           within-episode Pearson r with entanglement) through training: a activity (norm, effective
                           dimension, turn), b update (credit coherence, |dW|^2, dW's effective dimension, cross-band
                           share on the checkpoint's own and on the final bands); c the update medians; d Trained: the
                           coherence's r at every lag
  td_readout.png         up to 3,000 episodes: the gradient norm of every read-out row under the run's loss and the
                           loss; values more than 10 nats below their variable's best (theory_v9's exclusion)
  td_decoding.png        ridge R^2 from u and c (side statistic; 1 - sum SSE / sum SST over the target dimensions):
                           counts, Bayes's centred log joint posterior, the marginal posteriors, at steps 5 and 30
Averaging order (audit 11-01, John: no averaging before the statistic): b of td_learning paired the goal's
correctness with FR averaged over both variables (cross terms turned Bayes's and the Trained RNN's FR trend upside
down); the "variance explained by inference step" of the overview / td_factors was the variance of the probe-averaged
trajectory (td_analysis pool_*: each episode's own movement cancels; x read 0) and is now the within-episode share; the
PRs were of one cloud pooled over all steps (the drift of the mean counted as dimensions) and are now per step.
Findings 11-01 (Trained vs Echo, 114 checkpoints each): the memory-cut curves split from intact at ~300-500 training
episodes, when the gap to Bayes starts to fall fast; Echo's do not split (memory does not matter to it). Increments:
Trained learns N first (~50), then W (~200), then M (~500); Echo ends on N alone (beta_N 1.05, W 0.12, M ~0). On the
naive-wrong / Bayes-right particles, Trained gives naive's wrong answer at most 28% (~600 episodes, accuracy ~.33),
then 3%; Echo climbs to 50% and stays. Echo's u at step 30: the environment takes 70% of the variance, the latent
values' share common to all environments ~1% (Trained ~10% each; within each environment, cell + env x cell: Echo
24%, Trained 61%); PR per step ~18 vs ~57 (one cloud pooled over steps read ~10 vs ~60). The entanglement-locked update structure is absent at initialization and grows in
Trained only after ~1.5-2k episodes (coherence r 0 -> .34, dW eff. dim. r +.37 -> -.16); Echo's stays flat.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/plot_training_dynamics.py"""
import os
import sys
import glob
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
import gradflow_analysis as ga                                                    # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402

AN = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics", "analysis")
FACTORS = ("total", "env", "r0", "r1", "joint", "cell", "envcell", "res")
NETS = {"trained": ("Trained", "#2c7bb6"), "echo": ("Echo", "#d7191c")}
X0 = 10                                                                            # where the initial network is drawn
WIN = slice(16, 30)                                                                # the locked window, steps 17-30
TAU0 = list(ga.TAU).index(0)
N_FR = 5
FCOL = {"env": "#8c564b", "r0": "#1f77b4", "r1": "#17becf", "joint": "#9467bd", "cell": "#2ca02c", "envcell": "#ff7f0e",
        "res": "0.55"}
FNAME = {"env": "environment (Z)", "r0": "r0", "r1": "r1", "joint": r"r0 $\times$ r1 (joint)", "cell": "cell (r0, r1)",
         "envcell": r"env $\times$ cell", "res": "streams"}


def probes():
    return dict(np.load(os.path.join(AN, "probes.npz")))


def _xcorr(y, rev):
    return ga._lagged(ga._zs(y.astype(np.float64)), ga._zs(rev))


def summarize(f, P, fr_bin, rev):
    """the scalar summaries of one checkpoint file"""
    z = np.load(f, allow_pickle=True)
    n = len(P["goal"])
    ar = np.arange(n)
    s = {"episode": int(z["episode"]), "acc": z["acc"], "trunc": z["trunc_acc"][:, 0], "tau": z["tau"],
         "reg_xtx": z["reg_xtx"], "reg_xtd": z["reg_xtd"], "reg_dtd": z["reg_dtd"], "fac": z["fac"],
         "pool_x": z["pool_x"], "pool_u": z["pool_u"], "pool_c": z["pool_c"], "dec": z["dec_r2"],
         "kl_qp": z["kl_qp"], "kl_pq": z["kl_pq"], "excl_T": int(z["excl_T"]), "w_dprev": z["w_dprev"],
         "w_dinit": z["w_dinit"], "prev_episode": int(z["prev_episode"]), "n_bands": int(z["n_bands"])}
    right_T = z["right"][ar, -1, P["goal"]]
    s["acc_fr"] = np.array([right_T[fr_bin == k].mean() for k in range(N_FR)])     # fr_bin: the goal's own FR at T
    s["fr_pairing"] = "goal"
    ans = z["logq"][ar, -1, P["goal"]].astype(np.float32).argmax(-1)               # the network's answer at T
    nb_ans, jb_ans = (P[k][ar, -1, P["goal"]].argmax(-1) for k in ("log_nb", "log_jb"))
    truth = P["r"][ar, P["goal"]]
    naive_only = (nb_ans != truth) & (jb_ans == truth)                               # naive wrong, Bayes right
    s["agree"] = np.array([(ans == nb_ans).mean(), (ans == jb_ans).mean(), (ans == nb_ans)[naive_only].mean()])
    s["xc_act"] = np.stack([_xcorr(z[k][:, WIN], rev[:, WIN]) for k in ("norm_u", "effdim_u", "turn_u")])
    s["med_act"] = np.array([np.median(z[k][:, WIN]) for k in ("norm_u", "effdim_u", "turn_u")])
    if "cr_coh" in z.files:
        rows = z["cr_rows"]
        keys = ("cr_coh", "cr_E", "cr_dim", "cr_bandx")
        s["xc_upd"] = np.stack([_xcorr(z[k], rev[rows][:, WIN]) for k in keys]
                               + [_xcorr(z["cr_bandx_fixed"], rev[rows[:len(z["cr_bandx_fixed"])]][:, WIN])])
        s["med_upd"] = np.array([np.median(z[k]) for k in keys] + [np.median(z["cr_bandx_fixed"])])
    if "ro_gnorm" in z.files:
        s["ro_gnorm"], s["loss"] = z["ro_gnorm"], float(z["loss"])
    return s


def series(run, P):
    """the summaries of every checkpoint of a run, cached in analysis/series_<run>.npz"""
    files = sorted(glob.glob(os.path.join(AN, run, "ck_*.npz")))
    cache = os.path.join(AN, f"series_{run}.npz")
    old = dict(np.load(cache, allow_pickle=True))["s"].item() if os.path.exists(cache) else {}
    ar = np.arange(len(P["goal"]))
    fr = P["fr"][ar, -1, P["goal"]]                                                   # the goal's own FR at T
    fr_bin = np.searchsorted(np.quantile(fr, np.linspace(0, 1, N_FR + 1)[1:-1]), fr)
    rev = P["ent"].mean(-1)
    out = {}
    for f in files:
        ep = int(os.path.basename(f)[3:8])
        s = old.get(ep) or summarize(f, P, fr_bin, rev)
        if s.get("fr_pairing") != "goal":           # caches before 11-01: binned by FR averaged over both variables
            right_T = np.load(f)["right"][ar, -1, P["goal"]]
            s["acc_fr"] = np.array([right_T[fr_bin == k].mean() for k in range(N_FR)])
            s["fr_pairing"] = "goal"
        out[ep] = s
    np.savez(cache, s=np.array(out, dtype=object))
    eps = sorted(out)
    S = {"episode": np.array(eps)}
    for k in out[eps[0]]:
        vals = [out[e].get(k) for e in eps]
        if all(v is not None for v in vals):
            S[k] = np.array(vals)
        else:
            S[k] = np.array([np.full(np.shape(next(v for v in vals if v is not None)), np.nan) if v is None else v
                             for v in vals])
    S["x"] = np.where(S["episode"] - S["episode"] // 5 > 0, S["episode"] - S["episode"] // 5, X0).astype(float)
    xtx, xtd, dtd = S["reg_xtx"][:, 1:].sum(1), S["reg_xtd"][:, 1:].sum(1), S["reg_dtd"][:, 1:].sum(1)
    beta = np.array([np.linalg.lstsq(a, b, rcond=None)[0] for a, b in zip(xtx, xtd)])
    S["beta"] = beta
    S["r2"] = 1 - (dtd - 2 * (beta * xtd).sum(-1) + np.einsum("ki,kij,kj->k", beta, xtx, beta)) / dtd
    return S


def time_within(run, s):
    """td_time_within.py's within-episode time shares / PRs [x rows, u rows, c], aligned to the run's checkpoints"""
    z = np.load(os.path.join(AN, f"time_within_{run}.npz"))
    assert np.array_equal(z["episode"], s["episode"]), f"{run}: time_within checkpoints differ"
    return z


def step_pr(s, row):
    """the PR of each step's particle cloud (td_analysis fac, total factor) averaged over the steps"""
    return np.nanmean(s["fac"][:, :, row, 0, 1], 1)


TIME_T = "Movement within episodes (share)\n" + r"$\langle \Vert s_{it} - \bar s_i \Vert^2 \rangle / \langle \Vert s_{it} - \bar s \Vert^2 \rangle$"
PR_T = "PR per step, mean over steps\n" + r"$\langle (\Sigma_k \lambda_k)^2 / \Sigma_k \lambda_k^2 \rangle_t$"


def _axis(ax, title, xlabel=True):
    ax.set_xscale("log")
    ax.set_xlim(X0 * .8, 5e4)
    ax.set_xticks([X0, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel("training episodes")
    ax.spines[["top", "right"]].set_visible(False)


def _note(fig, P, S, extra="", y=-.01):
    counts = ", ".join(f"{NETS[r][0]} {len(s['episode'])} checkpoints" for r, s in S.items())
    fig.text(.5, y, f"{len(P['goal']):,} probe episodes (50 environments x 100 cells x 2 streams) at every "
             f"checkpoint ({counts}){extra}", ha="center", fontsize=9)


def _legend(ax, styles=(), nets=True, extra=(), loc="best", ncol=1):
    """a split legend: colour = network, grey line style = measure"""
    h = [Line2D([], [], color=NETS[r][1], lw=2.2, label=NETS[r][0]) for r in NETS] if nets else []
    h += [Line2D([], [], color="0.35", lw=1.5, ls=ls, label=lab) for lab, ls in styles]
    h += list(extra)
    ax.legend(handles=h, fontsize=8, frameon=False, loc=loc, ncol=ncol)


def fig_learning(S, P):
    apply_fig_style()
    fig, axs = plt.subplots(2, 3, figsize=(3 * 4.4, 2 * 3.3))
    ax = axs.ravel()
    goal = np.arange(len(P["goal"])), P["goal"]
    J = P["right_bayes"][goal].mean()
    refs = {"naive": P["right_naive"][goal].mean(), "step-wise (w = 1)": P["right_window1"][goal].mean(),
            "window 2": P["right_window2"][goal].mean(), "window 5": P["right_window5"][goal].mean()}
    for k, v in refs.items():
        ax[0].axhline(J - v, color="0.6", lw=.8, ls=":")
        ax[0].text(X0 * .9, (J - v) * 1.06, k, fontsize=8, color="0.4", va="bottom", ha="left")
    ws = (("memory cut to 1 step", ":"), ("cut to 2", "-."), ("cut to 5", "--"), ("intact", "-"))
    cm = plt.get_cmap("viridis")
    for run, s in S.items():
        col = NETS[run][1]
        for j, (_, ls) in enumerate(ws):
            ax[0].plot(s["x"], np.clip(J - s["trunc"][:, j], 1e-3, None), color=col, ls=ls, lw=1.4)
        for k in range(N_FR):
            ax[1].plot(s["x"], s["acc_fr"][:, k], color=cm(k / (N_FR - 1)), ls="-" if run == "trained" else "--", lw=1.3)
        for j, ls in enumerate(("-", "--", ":")):
            ax[2].plot(s["x"], s["beta"][:, j], color=col, ls=ls, lw=1.4)
        ax[2].plot(s["x"], s["r2"], color=col, lw=.8, alpha=.5)
        ax[3].plot(s["x"], np.median(s["tau"], 1), color=col, lw=1.4)
        ax[3].plot(s["x"], np.quantile(s["tau"], .99, axis=1), color=col, lw=1.4, ls="--")
        for j, ls in enumerate(("--", "-", ":")):
            ax[4].plot(s["x"], s["agree"][:, j], color=col, ls=ls, lw=1.4)
        ax[5].plot(s["x"], s["acc"][:, -1], color=col, lw=1.6)
    fr = P["fr"][goal[0], -1, goal[1]]                                               # the goal's own FR at T
    fb = np.searchsorted(np.quantile(fr, np.linspace(0, 1, N_FR + 1)[1:-1]), fr)
    for k in range(N_FR):
        for key, mk in (("right_bayes", "o"), ("right_naive", "x")):
            ax[1].plot(4.6e4, P[key][goal][fb == k].mean(), mk, color=cm(k / (N_FR - 1)), ms=5, clip_on=False)
    for k, v in (("Bayes", J), ("naive", refs["naive"])):
        ax[5].axhline(v, color="0.6", lw=.8, ls=":")
        ax[5].text(4.5e4, v + .01, k, fontsize=8, color="0.4", ha="right")
    _axis(ax[0], "Gap to Bayes at T: Bayes $-$ accuracy")
    ax[0].set_yscale("log")
    _legend(ax[0], ws, loc="lower left", ncol=2)
    _axis(ax[1], "Accuracy at T by the goal's own FR")
    _legend(ax[1], (("Trained", "-"), ("Echo", "--")), nets=False, loc="upper left",
            extra=[Line2D([], [], color=cm(k / (N_FR - 1)), lw=2, label=f"FR quintile {k + 1}") for k in range(N_FR)]
            + [Line2D([], [], color="0.35", marker=m, ls="", label=f"{n} per quintile") for m, n in
               (("o", "Bayes"), ("x", "naive"))])
    _axis(ax[2], r"Increments $d_t = \beta_N N + \beta_W W + \beta_M M$" "\n(one fit pooled over steps 2-30)")
    _legend(ax[2], ((r"$\beta_N$ (naive)", "-"), (r"$\beta_W$ (within step)", "--"), (r"$\beta_M$ (memory)", ":"),
                    (r"$R^2$ (thin)", "-")), loc="upper left")
    _axis(ax[3], r"Forget gate $\tau = 1/(1-\bar f)$ per unit")
    ax[3].set_yscale("log")
    _legend(ax[3], (("median", "-"), ("99th percentile", "--")), loc="center right")
    _axis(ax[4], "Network's answer at T agrees with ...")
    _legend(ax[4], (("naive's answer", "--"), ("Bayes's answer", "-"), ("naive's error (naive wrong, Bayes right)", ":")),
            loc="upper left")
    _axis(ax[5], "Accuracy at T (goal)")
    _legend(ax[5], loc="upper left")
    _note(fig, P, S)
    fig.tight_layout()
    return _save(fig, "td_learning.png")


def fig_factors(S, P):
    apply_fig_style()
    fig, axs = plt.subplots(2, 3, figsize=(3 * 4.4, 2 * 3.3))
    ax = axs.ravel()
    for a_i, run in enumerate(("trained", "echo")):
        if run not in S:
            continue
        s = S[run]
        share = s["fac"][:, 29, 6, :, 0] / s["fac"][:, 29, 6, :1, 0]
        for k in ("env", "r0", "r1", "joint", "envcell", "res"):
            ax[a_i].plot(s["x"], share[:, FACTORS.index(k)], color=FCOL[k], lw=1.5)
        _axis(ax[a_i], f"{NETS[run][0]}: share of u's variance at step 30, by factor")
    sigs = (("x (read-in output)", "#1f77b4"), (r"$W_{ih}x$ (input drive)", "#ff7f0e"), ("u", "#2ca02c"),
            (r"$W_{hh}u$ (recurrent drive)", "#d62728"), ("c", "#9467bd"))
    for run, s in S.items():
        ls = "-" if run == "trained" else "--"
        for k in ("env", "r0", "r1", "joint", "cell", "res"):
            ax[2].plot(s["x"], s["fac"][:, 29, 6, FACTORS.index(k), 1], color=FCOL.get(k, "#2ca02c"), ls=ls, lw=1.3)
        tw = time_within(run, s)
        shares = (tw["x"][:, 0, 0], tw["x"][:, 1, 0], tw["u"][:, 0, 0], tw["u"][:, 1, 0], tw["c"][:, 0, 0])
        for (name, col), sh, row in zip(sigs, shares, (0, 1, 6, 7, 12)):
            ax[3].plot(s["x"], sh, color=col, ls=ls, lw=1.3)
            ax[4].plot(s["x"], step_pr(s, row), color=col, ls=ls, lw=1.3)
    if "trained" in S:
        s = S["trained"]
        for row, ls in ((1, "-"), (7, ":")):
            for k in ("env", "cell", "joint"):
                ax[5].plot(s["x"], s["fac"][:, 29, row, FACTORS.index(k), 1], color=FCOL.get(k, "#2ca02c"), ls=ls, lw=1.3)
    _axis(ax[2], "Dimension (PR) each factor induces in u")
    _axis(ax[3], TIME_T)
    _axis(ax[4], PR_T)
    _axis(ax[5], "Trained: PR per factor, input vs recurrent drive")
    for a in (ax[2], ax[4], ax[5]):
        a.set_yscale("log")
    ls_h = [Line2D([], [], color="0.35", lw=1.6, ls=ls, label=lab) for lab, ls in
            (("Trained", "-"), ("Echo (c-e)", "--"), (r"$W_{ih}x$ input drive (f)", "-"), (r"$W_{hh}u$ recurrent (f)", ":"))]
    fac_h = [Line2D([], [], color=FCOL.get(k, "#2ca02c"), lw=2.2, label=FNAME.get(k, "cell (r0, r1)"))
             for k in ("env", "r0", "r1", "joint", "cell", "envcell", "res")]
    sig_h = [Line2D([], [], color=c, lw=2.2, label=n) for n, c in sigs]
    fig.tight_layout(rect=(0, .1, 1, 1))
    for i, (h, t) in enumerate(((ls_h, "line style"), (fac_h, "factors (a-c, f)"), (sig_h, "signals (d, e)"))):
        fig.legend(handles=h, title=t, loc="lower center", bbox_to_anchor=((.16, .5, .84)[i], -.02),
                   ncol=(2, 4, 3)[i], fontsize=8.5, title_fontsize=9, frameon=False)
    _note(fig, P, S, "; factors at step 30", y=-.06)
    return _save(fig, "td_factors.png")


def fig_entanglement(S, P):
    apply_fig_style()
    fig, axs = plt.subplots(2, 2, figsize=(2 * 4.8, 2 * 3.4))
    ax = axs.ravel()
    act = ((r"norm $\Vert u \Vert$", ":"), ("effective dim. of u", "--"), ("turn 1 $-$ cos(u$_t$, u$_{t-1}$)", "-"))
    upd = (("coherence", "-"), (r"$|dW|^2$", "--"), ("dW eff. dim.", ":"), ("cross-band share", "-."),
           ("cross-band, final bands", (0, (1, 3))))
    for run, s in S.items():
        col = NETS[run][1]
        for j, (_, ls) in enumerate(act):
            ax[0].plot(s["x"], s["xc_act"][:, j, TAU0], color=col, ls=ls, lw=1.4)
        if "xc_upd" in s:
            for j, (_, ls) in enumerate(upd):
                ax[1].plot(s["x"], s["xc_upd"][:, j, TAU0], color=col, ls=ls, lw=1.4)
            for j, ls in ((0, "-"), (2, ":")):
                ax[2].plot(s["x"], s["med_upd"][:, j], color=col, ls=ls, lw=1.4)
    if "trained" in S and "xc_upd" in S["trained"]:
        s = S["trained"]
        im = ax[3].pcolormesh(s["x"], ga.TAU, s["xc_upd"][:, 0].T, cmap="viridis", shading="nearest")
        fig.colorbar(im, ax=ax[3], label="r with entanglement")
        ax[3].set_ylabel(r"lag $\tau$ (steps after the entangled step)")
    for a in ax[:2]:
        a.axhline(0, color="0.7", lw=.6)
    _axis(ax[0], r"Activity: r with entanglement ($\tau$ = 0)")
    ax[0].set_ylim(-.62, .42)
    _legend(ax[0], act, loc="lower left", ncol=2)
    _axis(ax[1], r"Update: r with entanglement ($\tau$ = 0)")
    ax[1].set_ylim(-.62, .45)
    _legend(ax[1], upd, loc="lower left", ncol=2)
    _axis(ax[2], "Update: medians")
    ax[2].set_yscale("log")
    _legend(ax[2], (("coherence", "-"), ("dW eff. dim.", ":")), loc="lower left")
    _axis(ax[3], "Trained: coherence r at every lag")
    _note(fig, P, S, "; update measures on 5,000 of them, steps 17-30")
    fig.tight_layout()
    return _save(fig, "td_entanglement.png")


def fig_readout(S, P):
    apply_fig_style()
    fig, axs = plt.subplots(1, 3, figsize=(3 * 4.4, 3.3))
    for run, s in S.items():
        col = NETS[run][1]
        early = ~np.isnan(s["loss"])
        if early.any():
            g = s["ro_gnorm"][early]
            for v in range(g.shape[1]):
                axs[0].plot(s["x"][early], g[:, v], color=col, lw=.6, alpha=.5)
            axs[1].plot(s["x"][early], s["loss"][early], color=col, lw=1.5)
        axs[2].plot(s["x"], s["excl_T"], color=col, lw=1.5)
    for a, t in zip(axs, ("Read-out row gradient norms (20 rows)", "Loss on the probes",
                          "Values $>$ 10 nats below their variable's best")):
        _axis(a, t)
        _legend(a, loc="best")
    axs[0].set_yscale("log")
    _note(fig, P, S, "; rows and loss up to 3,000 episodes")
    fig.tight_layout()
    return _save(fig, "td_readout.png")


def fig_decoding(S, P):
    apply_fig_style()
    fig, axs = plt.subplots(1, 3, figsize=(3 * 4.4, 3.3))
    targets = ("counts $n_t$", "Bayes's centred log joint", "marginal log posteriors")
    styles = (("from u, step 5", "--"), ("from u, step 30", "-"), ("from c, step 30", ":"))
    for run, s in S.items():
        col = NETS[run][1]
        for j in range(3):
            for (step, a), (_, ls) in zip(((3, 0), (9, 0), (9, 1)), styles):          # DEC_STEPS: index 3 = step 5
                axs[j].plot(s["x"], s["dec"][:, step, a, j], color=col, ls=ls, lw=1.4)
    for a, t in zip(axs, targets):
        _axis(a, f"Ridge R$^2$ (all target dims pooled): {t}")
        _legend(a, styles, loc="lower right" if a is not axs[0] else "lower left")
    _note(fig, P, S, "; fitted on 25 environments, tested on the other 25 (side statistic)")
    fig.tight_layout()
    return _save(fig, "td_decoding.png")


def _fr_terms(f, rev_like):
    """the locked statistic's r with FR (both variables averaged) for the activity and update measures of one checkpoint
    file (added to the cached summaries without recomputing them)"""
    z = np.load(f, allow_pickle=True)
    out = {"xc_act_fr": np.stack([_xcorr(z[k][:, WIN], rev_like[:, WIN]) for k in ("norm_u", "effdim_u", "turn_u")])}
    if "cr_coh" in z.files:
        rows = z["cr_rows"]
        out["xc_upd_fr"] = np.stack([_xcorr(z[k], rev_like[rows][:, WIN]) for k in ("cr_coh", "cr_E", "cr_dim", "cr_bandx")])
    return out


def add_fr(S, run, P):
    """S[run] plus "xc_act_fr" / "xc_upd_fr" (cached in analysis/series_fr_<run>.npz)"""
    cache = os.path.join(AN, f"series_fr_{run}.npz")
    old = dict(np.load(cache, allow_pickle=True))["s"].item() if os.path.exists(cache) else {}
    fr = P["fr"].mean(-1)
    out = {}
    for ep in S[run]["episode"]:
        out[ep] = old.get(ep) or _fr_terms(os.path.join(AN, run, f"ck_{ep:05d}.npz"), fr)
    np.savez(cache, s=np.array(out, dtype=object))
    for k in out[S[run]["episode"][0]]:
        S[run][k] = np.array([out[e][k] for e in S[run]["episode"]])
    return S[run]


def fig_overview(S, P, x_acc=False):
    """one page of the main quantities, both networks, against training episodes or (x_acc) against the network's own
    accuracy at T (a: the training episodes it took to get there)"""
    apply_fig_style()
    fig, axs = plt.subplots(3, 3, figsize=(3 * 4.4, 3 * 3.3))
    ax = axs.ravel()
    goal = np.arange(len(P["goal"])), P["goal"]
    refs = {name: P[f"right_{key}"][goal].mean() for name, key in
            (("Bayes", "bayes"), ("memory-free Bayes", "window1"), ("naive", "naive"))}
    act = ((r"norm $\Vert u \Vert$", ":"), ("effective dim.", "--"), ("turn", "-"))
    upd = (("coherence", "-"), (r"$|dW|^2$", "--"), ("dW eff. dim.", ":"), ("cross-band share", "-."))
    pr_sig = (("u", "-"), (r"$W_{hh}u$", "--"), (r"$W_{ih}x$", ":"))
    time_sig = (("u", "-"), (r"$W_{hh}u$", "--"), ("c", ":"))
    for run, s in S.items():
        col = NETS[run][1]
        ls_net = "-" if run == "trained" else "--"
        x = s["acc"][:, -1] if x_acc else s["x"]
        if x_acc:
            ax[0].plot(x, s["x"], color=col, lw=1.6)
        else:
            ax[0].plot(x, s["acc"][:, -1], color=col, lw=1.6)
        for row, (_, ls) in zip((6, 7, 1), pr_sig):
            ax[1].plot(x, step_pr(s, row), color=col, ls=ls, lw=1.4)
        tw = time_within(run, s)
        for p_, (_, ls) in zip((tw["u"][:, 0, 0], tw["u"][:, 1, 0], tw["c"][:, 0, 0]), time_sig):
            ax[2].plot(x, p_, color=col, ls=ls, lw=1.4)
        share = s["fac"][:, 29, 6, :, 0] / s["fac"][:, 29, 6, :1, 0]
        for k in ("env", "r0", "r1", "joint", "envcell", "res"):
            ax[3].plot(x, share[:, FACTORS.index(k)], color=FCOL[k], ls=ls_net, lw=1.3)
        for k in ("env", "r0", "r1", "joint", "envcell"):
            ax[4].plot(x, s["fac"][:, 29, 6, FACTORS.index(k), 1], color=FCOL[k], ls=ls_net, lw=1.3)
        for j, (_, ls) in enumerate(act):
            ax[5].plot(x, s["xc_act"][:, j, TAU0], color=col, ls=ls, lw=1.4)
            ax[7].plot(x, s["xc_act_fr"][:, j, TAU0], color=col, ls=ls, lw=1.4)
        if "xc_upd" in s:
            for j, (_, ls) in enumerate(upd):
                ax[6].plot(x, s["xc_upd"][:, j, TAU0], color=col, ls=ls, lw=1.4)
                ax[8].plot(x, s["xc_upd_fr"][:, j, TAU0], color=col, ls=ls, lw=1.4)
    titles = ("Training episodes to reach the accuracy" if x_acc else "Accuracy at T (goal)",
              PR_T, TIME_T,
              "Share of u's variance at step 30, by factor", "Dimension (PR) per factor, u, step 30",
              r"Entanglement r ($\tau$ = 0): activity" "\n(entanglement: mean of the two variables)",
              r"Entanglement r ($\tau$ = 0): update" "\n(entanglement: mean of the two variables)",
              "Factorization regret r " r"($\tau$ = 0): activity" "\n(FR: mean of the two variables)",
              "Factorization regret r " r"($\tau$ = 0): update" "\n(FR: mean of the two variables)")
    for a, t in zip(ax, titles):
        if x_acc:
            a.set_title(t)
            a.set_xlim(.08, .8)
            a.set_xlabel("accuracy at T (goal)")
            a.spines[["top", "right"]].set_visible(False)
            for name in ("naive", "Bayes"):
                a.axvline(refs[name], color="0.6", lw=.8, ls=":")
        else:
            _axis(a, t)
    if x_acc:
        ax[0].set_yscale("log")
        ax[0].set_ylabel("training episodes")
        for name in ("naive", "Bayes"):
            ax[0].text(refs[name] - .008, .03, name, transform=ax[0].get_xaxis_transform(), fontsize=8, color="0.4",
                       ha="right", va="bottom")
    else:
        for name, v in refs.items():
            ax[0].axhline(v, color="0.6", lw=.8, ls=":")
            ax[0].text(4.5e4, v + .006, name, fontsize=8, color="0.4", ha="right", va="bottom")
    for a in (ax[1], ax[4]):
        a.set_yscale("log")
    for a in ax[5:]:
        a.axhline(0, color="0.7", lw=.6)
    _legend(ax[0], loc="upper left")
    ax[0].get_legend().set(frame_on=True)
    ax[0].get_legend().get_frame().set(facecolor="white", edgecolor="none", alpha=1)
    _legend(ax[1], pr_sig, loc="upper left")
    _legend(ax[2], time_sig, loc="lower left" if not x_acc else "upper right")
    fac_h = [Line2D([], [], color=FCOL[k], lw=2, label=FNAME[k]) for k in ("env", "r0", "r1", "joint", "envcell", "res")]
    net_h = [Line2D([], [], color="0.35", lw=1.5, ls=ls, label=lab) for lab, ls in (("Trained", "-"), ("Echo", "--"))]
    ax[3].set_ylim(0, 1.12)
    ax[3].legend(handles=fac_h + net_h, fontsize=7.5, frameon=False, loc="upper right", ncol=3, columnspacing=1.,
                 handlelength=1.6)
    ax[4].set_ylim(.8, 3e3)
    ax[4].legend(handles=fac_h[:5] + net_h, fontsize=7.5, frameon=False, loc="upper right", ncol=3,
                 columnspacing=1., handlelength=1.6)
    for a, st in ((ax[5], act), (ax[6], upd), (ax[7], act), (ax[8], upd)):
        a.set_ylim(-.66, .46)
        _legend(a, st, loc="lower right" if a is ax[8] else "lower left", ncol=2)
    _note(fig, P, S, "; update measures (g, i) on 5,000 of them, steps 17-30; r: the locked within-episode statistic"
          + ("; dotted lines: naive's and Bayes's accuracy" if x_acc else ""))
    fig.tight_layout()
    return _save(fig, "training_dynamics_overview_vs_accuracy.png" if x_acc else "training_dynamics_overview.png")


def fig_locked_measures(S, P):
    """the locked figure's five measures (gradflow_analysis; activity: steps 17-30 of all probes; update: 5,000 probes'
    errors at steps 17-30) through training: their medians and their within-episode r with entanglement at tau = 0"""
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    meas = ((r"Activity norm", r"$\Vert u_t \Vert$", "act", 0),
            ("Activity eff. dim.", r"$(\Sigma_i a_i^2)^2 / \Sigma_i a_i^4$", "act", 1),
            ("Credit coherence", r"$\Vert \Sigma_k \delta_k \Vert^2 / \Sigma_k \Vert \delta_k \Vert^2$", "upd", 0),
            ("Update energy", r"$\Vert \Delta W \Vert^2$", "upd", 1),
            ("Cross-band share", r"$1 - \Sigma_B \Vert \Delta W_B \Vert^2 / \Vert \Delta W \Vert^2$", "upd", 3))
    fig, axs = plt.subplots(2, 5, figsize=(5 * 3.9, 2 * 3.4))
    for j, (name, formula, kind, idx) in enumerate(meas):
        for run, s in S.items():
            col = NETS[run][1]
            axs[0, j].plot(s["x"], s[f"med_{kind}"][:, idx], color=col, lw=1.6)
            axs[1, j].plot(s["x"], s[f"xc_{kind}"][:, idx, TAU0], color=col, lw=1.6)
        _axis(axs[0, j], f"{name}\n{formula}", xlabel=False)
        _axis(axs[1, j], f"{name}: r with entanglement", xlabel=True)
        axs[1, j].axhline(0, color="0.7", lw=.6)
        axs[1, j].set_ylim(-.45, .45)
        axs[0, j].set_title(axs[0, j].get_title(), fontsize=11)
        axs[1, j].set_title(axs[1, j].get_title(), fontsize=10.5)
    for j in (0, 1, 2, 3):
        axs[0, j].set_yscale("log")
    axs[0, 4].axhline(0, color="0.7", lw=.6)
    axs[0, 0].set_ylabel("median (steps 17-30)")
    axs[1, 0].set_ylabel(r"within-episode r ($\tau$ = 0)")
    _legend(axs[0, 0], loc="lower right")
    _note(fig, P, S, "; activity: all 10,000 probes, update: 5,000 probes' errors, steps 17-30; r: the locked "
          "statistic (z-scored within each episode)", y=-.02)
    fig.tight_layout()
    return _save(fig, "td_locked_measures.png")


ATLAS = ("training_dynamics_overview.png", "training_dynamics_overview_vs_accuracy.png", "td_learning.png",
         "td_factors.png", "td_entanglement.png", "td_readout.png", "td_decoding.png")


def _save(fig, name):
    for i, a in enumerate(a for a in fig.axes if a.get_label() != "<colorbar>"):
        a.set_title(f"{'abcdefghij'[i]}   {a.get_title()}", loc="left", fontsize=a.title.get_fontsize())
        a.set_title("")
    path = fig_path(os.path.join("td_atlas", name) if name in ATLAS else name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    P = probes()
    S = {run: series(run, P) for run in NETS if glob.glob(os.path.join(AN, run, "ck_*.npz"))}
    for run in S:
        add_fr(S, run, P)
    for f in (fig_overview, fig_learning, fig_factors, fig_entanglement, fig_readout, fig_decoding):
        f(S, P)
    fig_overview(S, P, x_acc=True)
    fig_locked_measures(S, P)
