"""Participation ratio of the beliefs and of their updates in LOGIT space, covariance pooled over environments
(25 Oct 2026) -> main/REPLICATION/figure_bin/fig1c_participation_ratio_logit.png

Particles: the trained network's gradflow harvest, first 20,000 episodes, same-variable environments left out (19,607
episodes from 1,225 environments), all four agents on the same episodes.
Logit coordinates: per variable, the centred log-belief x(r) = log b(r) - mean_r' log b(r') (the softmax's free constant
removed), 10 per variable, 20 in all (at most 18 independent dimensions).
  Optimal, Naive  exact, float64, from the episode's observations and table (Naive: the table averaged over the other
                  variable; its log-belief is linear in the 5 observation counts, so within one environment and step its
                  cloud has at most 5 dimensions; pooled over environments the subspaces differ)
  Trained, ESN    the networks' own summed read-out logits r_1 + ... + r_t (what their softmax reads), replayed on CPU
  X_t the logits after step t; U_t = X_t - X_{t-1}, X_0 = 0 (the uniform prior). PR = (tr C)^2 / tr(C^2), C over all
  pooled particles (per step; and pooled over the 30 steps for the bars); tr C in nats^2. Bands: 95% bootstrap over
  environments (clusters, 1000 resamples). Also computed (printed, not drawn): the same with log b floored at -10 nats.
Data: figdata/round4b/pr_logit.npz (the logits themselves cached in the session scratchpad; rebuilt with --rebuild,
about 3 min on the CPU).
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_pr_logit.py [--rebuild]"""
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
DATA = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression", "figdata", "round4b", "pr_logit.npz")
CACHE = ("/tmp/claude-1002/-home-johns-Desktop-CognitiveGridworld/7af6b091-54ff-45a0-b4dd-ef0a3bc9c778/scratchpad/"
         "research7/pr_logit_cache.npz")
AG = ("Joint", "Trained", "Naive", "Echo")
AGENT = {"Joint": ("Optimal", "k"), "Trained": ("Trained RNN", "#1f77b4"), "Naive": ("Naive", "0.5"),
         "Echo": ("ESN", "#ff7f0e")}
STY = {"Trained": dict(lw=3.0, alpha=.45, zorder=3), "Joint": dict(lw=1.1, zorder=5), "Naive": dict(lw=1.5, zorder=4),
       "Echo": dict(lw=1.5, zorder=4)}
N_EP, NB = 20000, 1000


def _lse(x, axis):
    m = x.max(axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.exp(x - m).sum(axis))


def _clr(lb):
    return lb - lb.mean(-1, keepdims=True)


def _net_logits(net, D, batches):
    """fig1._beliefs with the summed read-out logits returned instead of their softmax: (n, T, 2, 10), raw
    variable order"""
    import gc
    import torch
    import fig1 as cf
    from main.CognitiveGridworld import CognitiveGridworld
    ga = cf.ga
    load_env, reservoir = ga.AGENTS[net]
    agent = CognitiveGridworld(**dict(ga.CFG, episodes=1), reservoir=reservoir, load_env=load_env,
                               external_logger=lambda env: {})
    m, (B, T) = agent.model, (ga.CFG["batch_num"], ga.CFG["step_num"])
    dev, out = m.STM.device, []
    for b in range(batches):
        rows = slice(b * B, (b + 1) * B)
        m.obs_flat, m.active_Z, m.goal_ind, m.joint_goal_belief = (torch.as_tensor(D[k][rows], device=dev)
                                                                   for k in cf.REPLAY)
        with torch.no_grad(), torch.autocast(dev.type, enabled=False):
            st, _ = ga.forward_states(m, ga.lstm_input(m))
            r = m.classifier_readout(st["u"])
            out.append(r.double().cumsum(1).reshape(B, T, m.ctx_num, m.realization_num).cpu().numpy())
        print(f"{net}: logit batch {b + 1} of {batches}", flush=True)
    del agent
    gc.collect()
    return np.concatenate(out)


def logits():
    """{agent: centred log-beliefs (n, T, 20), goal-ordered}, the kept episodes' environment index, and the floored
    version; cached"""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return {a: z[a] for a in AG}, {a: z[a + "_f10"] for a in AG}, z["env"]
    import fig1 as cf
    f = cf._harvest("trained", ["ep_obs_flat", "ep_joint_likelihood", "ep_goal_ind", "ep_ctx_inds"] + list(cf.REPLAY[1:]))
    ci = np.sort(f["ep_ctx_inds"][:N_EP].astype(int), 1)
    keep = ci[:, 0] != ci[:, 1]
    env = np.unique(ci[keep], axis=0, return_inverse=True)[1].ravel()
    g = f["ep_goal_ind"][:N_EP].astype(int)
    raw = {a: np.zeros((N_EP, 30, 2, 10)) for a in AG}                              # log b, raw variable order
    for s0 in range(0, N_EP, 2000):
        sl = slice(s0, s0 + 2000)
        o = f["ep_obs_flat"][sl].astype(np.float64)                                 # (n, T, 5)
        L = np.clip(f["ep_joint_likelihood"][sl], 1e-12, 1 - 1e-12)                 # (n, 5, r0, r1)
        cum = np.cumsum(np.einsum("nti,nijk->ntjk", o, np.log(L)) + np.einsum("nti,nijk->ntjk", 1 - o, np.log1p(-L)), 1)
        lp = cum - _lse(cum.reshape(*cum.shape[:2], -1), -1)[..., None, None]       # joint log-posterior
        raw["Joint"][sl] = np.stack([_lse(lp, -1), _lse(lp, -2)], 2)
        for k, Lb in ((0, L.mean(3)), (1, L.mean(2))):                              # the Naive: averaged tables
            ll = np.cumsum(np.einsum("nti,nir->ntr", o, np.log(Lb)) + np.einsum("nti,nir->ntr", 1 - o, np.log1p(-Lb)), 1)
            raw["Naive"][sl, :, k] = ll - _lse(ll, -1)[..., None]
    D = {k: f[k][:N_EP] for k in cf.REPLAY}
    for net, a in (("trained", "Trained"), ("echo", "Echo")):
        lg = _net_logits(net, D, N_EP // cf.ga.CFG["batch_num"])
        raw[a] = lg - _lse(lg, -1)[..., None]
    ar = np.arange(N_EP)
    X, Xf = {}, {}
    for a in AG:
        gb = np.stack([raw[a][ar, :, g], raw[a][ar, :, 1 - g]], 2)[keep]           # goal-ordered (n, T, 2, 10)
        X[a] = _clr(gb).reshape(keep.sum(), 30, 20)
        Xf[a] = _clr(np.maximum(gb, -10.)).reshape(keep.sum(), 30, 20)
    np.savez(CACHE, env=env, **{a: X[a].astype(np.float32) for a in AG}, **{a + "_f10": Xf[a].astype(np.float32) for a in AG})
    return X, Xf, env


def _pr(s1, s2, n):
    mu = s1 / n[..., None]
    C = s2 / n[..., None, None] - mu[..., :, None] * mu[..., None, :]
    tr = np.einsum("...ii->...", C)
    return tr ** 2 / (C ** 2).sum((-1, -2)), tr


FLOORS = (-10., -30., None)                                                      # log-probability floors drawn


def _floored(X, f):
    """centred logits (n, T, 20) -> log-probabilities floored at f, re-centred (f None: unchanged)"""
    if f is None:
        return X.astype(np.float64)
    x = X.astype(np.float64).reshape(*X.shape[:2], 2, 10)
    lp = x - _lse(x, -1)[..., None]
    y = np.maximum(lp, f)
    return (y - y.mean(-1, keepdims=True)).reshape(*X.shape[:2], 20)


def build():
    from scipy.sparse import csr_matrix
    X, _, env = logits()
    n_env = env.max() + 1
    cnt = np.bincount(env, minlength=n_env).astype(np.float64)
    Et = csr_matrix((np.ones(len(env)), (env, np.arange(len(env)))), shape=(n_env, len(env)))
    W = np.random.default_rng(0).multinomial(n_env, np.full(n_env, 1 / n_env), size=NB).astype(np.float64)
    out = dict(n_episodes=len(env), n_env=int(n_env), agents=np.array(AG),
               floors=np.array([np.nan if f is None else f for f in FLOORS]))
    nf = len(FLOORS)
    for key in ("pr", "tr"):
        for sfx, shape in (("", (len(AG), nf, 2, 30)), ("_all", (len(AG), nf, 2))):
            for b in ("", "_lo", "_hi"):
                out[key + sfx + b] = np.zeros(shape)                                # agent, floor, (X, U), [step]
    for a_, a in enumerate(AG):
        for f_, fl in enumerate(FLOORS):
            Z = _floored(X[a], fl)
            for b_, Y in enumerate((Z, np.diff(Z, axis=1, prepend=0.))):
                A1, A2 = np.zeros((n_env, 20)), np.zeros((n_env, 20, 20))
                for s in range(30):
                    x = Y[:, s]
                    s1 = Et @ x
                    s2 = (Et @ (x[:, :, None] * x[:, None, :]).reshape(len(x), -1)).reshape(n_env, 20, 20)
                    A1 += s1
                    A2 += s2
                    pr, tr = _pr(s1.sum(0), s2.sum(0), np.array(cnt.sum()))
                    prb, trb = _pr(W @ s1, (W @ s2.reshape(n_env, -1)).reshape(-1, 20, 20), W @ cnt)
                    for k, v, vb in (("pr", pr, prb), ("tr", tr, trb)):
                        out[k][a_, f_, b_, s] = v
                        out[k + "_lo"][a_, f_, b_, s], out[k + "_hi"][a_, f_, b_, s] = np.percentile(vb, [2.5, 97.5])
                pr, tr = _pr(A1.sum(0), A2.sum(0), np.array(cnt.sum() * 30))
                prb, trb = _pr(W @ A1, (W @ A2.reshape(n_env, -1)).reshape(-1, 20, 20), (W @ cnt) * 30)
                for k, v, vb in (("pr", pr, prb), ("tr", tr, trb)):
                    out[k + "_all"][a_, f_, b_] = v
                    out[k + "_all_lo"][a_, f_, b_], out[k + "_all_hi"][a_, f_, b_] = np.percentile(vb, [2.5, 97.5])
        print(a, "done", flush=True)
    np.savez(DATA, **out)
    print("saved", DATA)


def figure():
    Z = np.load(DATA)
    ag = list(Z["agents"])
    T = np.arange(1, 31)
    fl = [None if np.isnan(v) else v for v in Z["floors"]]
    nf = len(fl)
    W, H = nf * 2.9 + (nf - 1) * .95 + .9, 2 * 2.0 + 1.25 + .95 + .75
    fig = plt.figure(figsize=(W, H))
    lab = lambda f: "log-probabilities floored at %d nats" % f if f is not None else "exact (no floor)"
    for f_, f in enumerate(fl):
        x0 = .8 + f_ * (2.9 + .95)
        ax = fig.add_axes([x0 / W, (H - .95 - 2.0) / H, 2.9 / W, 2.0 / H])
        for a in ("Trained", "Joint", "Naive", "Echo"):
            i = ag.index(a)
            for b_, ls in ((0, "-"), (1, "--")):
                ax.plot(T, Z["pr"][i, f_, b_], c=AGENT[a][1], ls=ls, **STY[a])
                ax.fill_between(T, Z["pr_lo"][i, f_, b_], Z["pr_hi"][i, f_, b_], color=AGENT[a][1], alpha=.15, lw=0)
        ax.set_ylim(0, 18.5), ax.set_yticks([0, 6, 12, 18])
        ax.set_xlim(.5, 30.5), ax.set_xticks([1, 10, 20, 30])
        ax.set_xlabel("Step $t$", labelpad=1)
        ax.set_ylabel("Effective dimensions")
        _deco(fig, ax, "Participation ratio, " + ("floor %d nats" % f if f is not None else "no floor"),
              r"$PR = (\mathrm{tr}\,C)^2 / \mathrm{tr}(C^2)$; " + lab(f), "abc"[f_], W, H)
        bx = fig.add_axes([x0 / W, .75 / H, 2.9 / W, 2.0 / H])
        order = ("Joint", "Trained", "Naive", "Echo")
        for k_, a in enumerate(order):
            i = ag.index(a)
            for b_, dx in ((0, -.19), (1, .19)):
                v, lo, hi = Z["pr_all"][i, f_, b_], Z["pr_all_lo"][i, f_, b_], Z["pr_all_hi"][i, f_, b_]
                bx.bar(k_ + dx, v, width=.36, color=AGENT[a][1], alpha=1 if b_ == 0 else .45, lw=0,
                       hatch=None if b_ == 0 else "////", edgecolor="white")
                bx.errorbar(k_ + dx, v, yerr=[[v - lo], [hi - v]], color="0.2", lw=.8, capsize=2)
                bx.text(k_ + dx, hi + .12, "%.1f" % v, ha="center", va="bottom", fontsize=FS["tick"] - 2, color="0.25")
        bx.set_xticks(range(4), [AGENT[a][0] for a in order], fontsize=FS["tick"] - .5)
        bx.set_ylim(0, 8), bx.set_yticks([0, 2, 4, 6, 8])
        bx.set_ylabel("Effective dimensions")
        _deco(fig, bx, "Pooled over the 30 steps", r"one $C$ per agent: all particles $\times$ all steps", "def"[f_], W, H)
    hand = [Line2D([], [], c=AGENT[a][1], label=AGENT[a][0], **{k: v for k, v in STY[a].items() if k != "zorder"})
            for a in ("Joint", "Trained", "Naive", "Echo")]
    hand += [Line2D([], [], c="0.3", lw=1.4, label="beliefs $x_t$ (bars: solid)"),
             Line2D([], [], c="0.3", lw=1.4, ls="--", label="updates $x_t - x_{t-1}$ (bars: hatched)")]
    fig.legend(handles=hand, loc="upper center", bbox_to_anchor=(.5, 1 - .06 / H), ncol=6, frameon=False,
               fontsize=FS["tick"], handlelength=1.8, columnspacing=1.4)
    fig.text(.8 / W, .32 / H, f"Logit coordinates: per variable, log-probabilities (floored at the panel's floor), centred "
             f"over the 10 values; Optimal and Naive exact, networks their own summed read-out logits.\n"
             f"{int(Z['n_episodes']):,} episodes from {int(Z['n_env']):,} environments, pooled. A floor at $f$ treats values "
             f"with probability below $e^{{f}}$ as ruled out.", fontsize=FS["tick"] - 1.5, color="0.35", va="top")
    path = fig_path("fig1c_participation_ratio_logit.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _deco(fig, ax, name, formula, letter, W, H):
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate(name, (.5, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points", ha="center",
                va="baseline", fontsize=FS["label"] + .5)
    ax.annotate(formula, (.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points", ha="center",
                va="baseline", fontsize=FS["tick"], color="0.25")
    x0, y0, w, h = ax.get_position().bounds
    fig.text(x0 - .62 / W, y0 + h + 17 / 72 / H, r"$\mathbf{%s}$" % letter, fontsize=FS["label"] + 2, va="baseline")


if __name__ == "__main__":
    if "--rebuild" in sys.argv[1:] or not os.path.exists(DATA) or "floors" not in np.load(DATA).files:
        if "--rebuild" in sys.argv[1:] and os.path.exists(CACHE):
            os.remove(CACHE)
        build()
    print("saved", figure())
    Z = np.load(DATA)
    for f_, f in enumerate(Z["floors"]):
        for i, a in enumerate(Z["agents"]):
            print("floor %5s %-8s PR at T beliefs %5.2f updates %5.2f | pooled over steps %5.2f / %5.2f" % (
                f, a, Z["pr"][i, f_, 0, -1], Z["pr"][i, f_, 1, -1], *Z["pr_all"][i, f_]))
