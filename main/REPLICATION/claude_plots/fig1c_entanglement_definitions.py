"""Which entanglement explains the network: combinations of the two variables (min ... max) and other definitions,
against the trained RNN's representation, dynamics and learning, with and without update-size controls (29 Oct 2026;
John: "so maybe the correct measure isn't average or product, but max of the variables??" and "maybe the wormholes
really are not the right story, but maybe this analysis will find a cleaner version of entanglement for explaining
network representation, dynamics and learning.")
-> main/REPLICATION/figure_bin/fig1c_entanglement_definitions.png

Statistic: the locked gradflow cross-correlation at tau 0 (fig1c_two_processes._lagged): the network measure and the
candidate each z-scored within its stream over the window, the mean over streams and steps of their product (= the mean
within-stream Pearson r); late window steps 17 - 30 (the locked figure's), early 3 - 16. "Beyond update size": the
candidate first residualized (one pooled least-squares fit, within-stream centred; fig1c_two_processes._resid) on the
model-free observation novelty and change (the locked figure's own control) AND on the Optimal's information gain
KL(post_t || post_t-1), its Bayesian surprise -log p(o_t | past) and the Naive's information gain.
Candidates (per stream-step, recomputed in float64 from scan_meta's observations and tables; 39 environments x 5,000
streams): from the two per-variable entanglements e_g, e_c (fig1c_joint_entanglement.rev_vars) the generalized means
M_p = ((e_g^p + e_c^p) / 2)^(1/p): p = -inf min, 0 the sum of logs (geometric mean), 1 the average (= the stored
entanglement rev), +inf max; KL(Naive || Optimal) and KL(Optimal || Naive) instead of their average (Jeffreys);
"joint posterior": 0.5 Jeffreys between the Optimal's posterior over the 100 cells after its true update and after a
factorized update of the same prior (the product of the Naive's per-variable likelihoods); "observation only": the same
for a uniform prior (how non-factorizable the observation's likelihood is). Controls (not entanglement): the Optimal's
information gain and surprise, the model-free novelty. Network (the Trained RNN, fig1c_two_processes.series):
representation = activity effective dimension, activity norm; dynamics = turn; learning = weight-update energy, credit
coherence, cross-band share, update effective dimension.
a  the generalized-mean scan, late window: the statistic for each network measure against p.
b - e  every candidate x network measure: late / early window, raw / beyond update size.

Measured 29 Oct:
- a: every network measure peaks at p = 1 - 2 (the average) and stays flat to the max (max vs average within 0.006);
  the sum of logs is 0.01 - 0.02 lower, the min 0.03 - 0.07 lower, either variable alone 0.04 - 0.06 lower. Binned R2
  (steps 8 - 30, double-centred): the average captures 90 - 100% of what any function of (e_g, e_c) explains.
- b: late, raw: the joint posterior is ahead for learning and dynamics (weight-update energy +0.33 vs +0.19, turn
  +0.29 vs +0.27, coherence +0.36 vs +0.34, activity dimension +0.30 vs +0.27) and behind for the activity norm (+0.25
  vs +0.37); 39/39 environments for the weight-update energy. Early (d) its lead is large: cross-band share +0.47 vs
  +0.15, weight-update energy +0.42 vs -0.02, turn +0.43 vs +0.25. The observation-only version explains the activity
  norm best (+0.39 / +0.38) and little else. KL directions change nothing (within 0.02).
- But the joint posterior is mostly update size: within-stream r with the Optimal's information gain 0.65 (late) /
  0.70 (early), and the information gain alone does as well or better (late: weight-update energy +0.36, coherence
  +0.39, cross-band share +0.34; early cross-band share +0.55). The Optimal's surprise and the model-free novelty are
  as strong or stronger for activity dimension and turn (+0.41 - 0.44).
- c, e: beyond update size, the stored average keeps the clearest entanglement-specific links (late: activity norm
  +0.22, coherence +0.11, cross-band share +0.08, update effective dimension -0.09; early: activity norm +0.25,
  coherence +0.14, weight-update energy -0.12); the joint posterior keeps a small lead only for the weight-update
  energy (late +0.09 vs +0.02, early +0.08 vs -0.12) and the early cross-band share (+0.08 vs -0.03). The
  observation-only version is the cleanest link of all, to the activity norm: +0.39 raw / +0.33 beyond update size
  (late), +0.38 / +0.34 (early), against the stored average's +0.37 / +0.22 and +0.32 / +0.25; it does not depend on
  the belief at all (how non-factorizable this observation's likelihood is over the 100 cells).
=> For learning and dynamics no cleaner entanglement: the max equals the average, the product is worse, and the
   definition that looks better (the joint posterior) does so by carrying the update's size. For representation the
   observation's own entanglement is cleaner (activity norm). Beyond the locked figure's novelty control, the
   Optimal's surprise and update size remove about another third of the entanglement - network link (late cross-band
   share +0.12 -> +0.08, coherence +0.15 -> +0.11); the activity norm's link survives best.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/fig1c_entanglement_definitions.py"""
import os
import sys
import warnings
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
import fig1c_two_processes as tp                                                  # noqa: E402
import fig1c_joint_entanglement as je                                             # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)
mm = tp.mm
F64 = np.float64
EPS = 1e-99
CANDS = os.path.join(je.OUT, "candidates.npz")
CTRLS = os.path.join(je.OUT, "controls.npz")
PS = (-np.inf, -4, -2, -1, 0, 1, 2, 4, np.inf)
WINDOWS = {"late": (17, 30), "early": (3, 16)}
NET = (("pr", "activity\ndimension"), ("l2", "activity\nnorm"), ("turn", "turn"), ("dW", "weight-update\nenergy"),
       ("coh", "credit\ncoherence"), ("band_x", "cross-band\nshare"), ("dim", "update eff.\ndimension"))
GROUPS = (("representation", 0, 2), ("dynamics", 2, 3), ("learning", 3, 7))


def _nrm(x, axes):
    x = np.maximum(x, EPS)
    return x / x.sum(axes, keepdims=True)


def _kl(p, q, axes):
    return (p * (np.log(p) - np.log(q))).sum(axes)


def _per_env(e, meta):
    """one environment's exact joint log-likelihoods, posterior, prior and the Naive's per-variable log-likelihoods"""
    L, o = meta["L"][e].astype(F64), meta["obs"][e].astype(F64)
    ll = np.einsum("nti,ijk->ntjk", o, np.log(L)) + np.einsum("nti,ijk->ntjk", 1 - o, np.log1p(-L))
    cum = np.cumsum(ll, 1)
    post = np.exp(cum - cum.max((-1, -2), keepdims=True))
    post /= post.sum((-1, -2), keepdims=True)
    prior = np.concatenate([np.full_like(post[:, :1], 1 / 100), post[:, :-1]], 1)
    Lb = (L.mean(2), L.mean(1))
    lpx = np.stack([o @ np.log(Lb[k]) + (1 - o) @ np.log1p(-Lb[k]) for k in (0, 1)], 2)
    return ll, post, prior, lpx


def candidates():
    """cached per stream-step (40 envs): KL(Naive || Optimal), KL(Optimal || Naive) of the per-variable updates (mean
    over the variables), the joint-posterior and observation-only Jeffreys"""
    if not os.path.exists(CANDS):
        meta = np.load(os.path.join(mm.SCAN, "scan_meta.npz"))
        out = {k: np.zeros(meta["rev"].shape, np.float32)
               for k in ("kl_naive_opt", "kl_opt_naive", "post_jeff", "lik_jeff")}
        for e in range(len(meta["goal"])):
            ll, post, prior, lpx = _per_env(e, meta)
            joint = np.stack([post.sum(-1), post.sum(-2)], 2)
            px = _nrm(np.exp(lpx - lpx.max(-1, keepdims=True)), -1)
            aj = np.stack([_nrm(je._approx_lik(joint[:, :, k]), -1) for k in (0, 1)], 2)
            out["kl_naive_opt"][e] = _kl(px, aj, -1).mean(-1)
            out["kl_opt_naive"][e] = _kl(aj, px, -1).mean(-1)
            lfac = lpx[:, :, 0][..., :, None] + lpx[:, :, 1][..., None, :]
            pf = _nrm(prior * np.exp(lfac - lfac.max((-1, -2), keepdims=True)), (-1, -2))
            pt = _nrm(post, (-1, -2))
            out["post_jeff"][e] = .5 * (_kl(pt, pf, (-1, -2)) + _kl(pf, pt, (-1, -2)))
            lj = _nrm(np.exp(ll - ll.max((-1, -2), keepdims=True)), (-1, -2))
            lf = _nrm(np.exp(lfac - lfac.max((-1, -2), keepdims=True)), (-1, -2))
            out["lik_jeff"][e] = .5 * (_kl(lj, lf, (-1, -2)) + _kl(lf, lj, (-1, -2)))
        np.savez(CANDS, **out)
    with np.load(CANDS) as f:
        return {k: f[k][mm.ENVS].astype(F64) for k in f.files}


def controls():
    """cached per stream-step (40 envs): the Optimal's information gain and surprise, the Naive's information gain"""
    if not os.path.exists(CTRLS):
        meta = np.load(os.path.join(mm.SCAN, "scan_meta.npz"))
        out = {k: np.zeros(meta["rev"].shape, np.float32) for k in ("ig_opt", "surprise_opt", "ig_naive")}
        for e in range(len(meta["goal"])):
            ll, post, prior, lpx = _per_env(e, meta)
            pt, pr = np.maximum(post, EPS), np.maximum(prior, EPS)
            out["ig_opt"][e] = (pt * (np.log(pt) - np.log(pr))).sum((-1, -2))
            out["surprise_opt"][e] = -np.log((prior * np.exp(ll)).sum((-1, -2)))
            cn = np.cumsum(lpx, 1)
            bn = np.exp(cn - cn.max(-1, keepdims=True))
            bn /= bn.sum(-1, keepdims=True)
            bp = np.concatenate([np.full_like(bn[:, :1], 1 / 10), bn[:, :-1]], 1)
            bn, bp = np.maximum(bn, EPS), np.maximum(bp, EPS)
            out["ig_naive"][e] = (bn * (np.log(bn) - np.log(bp))).sum(-1).sum(-1)
        np.savez(CTRLS, **out)
    with np.load(CTRLS) as f:
        return {k: f[k][mm.ENVS].astype(F64) for k in f.files}


def pmean(rv, p):
    """generalized mean of the two variables' entanglements"""
    lg, lc = np.log(rv[..., 0]), np.log(rv[..., 1])
    if p == -np.inf:
        return np.exp(np.minimum(lg, lc))
    if p == np.inf:
        return np.exp(np.maximum(lg, lc))
    if p == 0:
        return np.exp((lg + lc) / 2)
    m = np.maximum(p * lg, p * lc)
    return np.exp((m + np.log((np.exp(p * lg - m) + np.exp(p * lc - m)) / 2)) / p)


def scores():
    """{window: {'raw' / 'beyond': (rows, (n rows, n NET))}}, the p-scan (late, raw) and the row names"""
    V = tp.series()
    rv = je.rev_vars()
    C, K = candidates(), controls()
    rows = {"average (stored)": V["rev"], "max": pmean(rv, np.inf), "sum of logs": pmean(rv, 0),
            "min": pmean(rv, -np.inf), r"KL(Naive $\Vert$ Optimal)": C["kl_naive_opt"],
            r"KL(Optimal $\Vert$ Naive)": C["kl_opt_naive"], "joint posterior": C["post_jeff"],
            "observation only": C["lik_jeff"], "Optimal's information gain": K["ig_opt"],
            "Optimal's surprise": K["surprise_opt"], "novelty (model-free)": V["nov"]}
    n_ent = 8
    out = {}
    for wn, (lo, hi) in WINDOWS.items():
        idx = np.arange(lo - 1, hi)
        nets = [tp._zs(tp._win(V[k], idx)) for k, _ in NET]
        Z = {nm: tp._zs(tp._win(x, idx)) for nm, x in rows.items()}
        ctrl = np.stack([tp._win(V["nov"], idx), tp._win(V["chg"], idx), Z["Optimal's information gain"],
                         Z["Optimal's surprise"], tp._zs(tp._win(K["ig_naive"], idx))], -1)
        raw = np.array([[tp._lagged(a, Z[nm], taus=[0])[0] for a in nets] for nm in rows])
        bey = np.full_like(raw, np.nan)
        for i, nm in enumerate(list(rows)[:n_ent]):
            rb = tp._resid(Z[nm], ctrl)
            bey[i] = [tp._lagged(a, rb, taus=[0])[0] for a in nets]
        out[wn] = {"raw": raw, "beyond": bey}
    idx = np.arange(16, 30)
    nets = [tp._zs(tp._win(V[k], idx)) for k, _ in NET]
    scan = np.array([[tp._lagged(a, tp._zs(tp._win(pmean(rv, p), idx)), taus=[0])[0] for a in nets] for p in PS])
    return out, scan, list(rows)


def figure():
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    out, scan, names = scores()
    W_, H_ = 17.0, 10.4
    fig = plt.figure(figsize=(W_, H_))

    def ax_at(x, y, w, h):
        return fig.add_axes([x / W_, y / H_, w / W_, h / H_])

    fig.text(.5, 1 - .3 / H_, "The max equals the average and the product is worse; the joint-belief version wins only "
             "by carrying the update's size; the observation's own entanglement is cleanest for the activity norm",
             ha="center", va="center", fontsize=FS["label"] + 3)
    fig.text(.5, 1 - .64 / H_, "alias scan, 39 environments x 5,000 streams; the Trained RNN; statistic: within-stream "
             r"Pearson $r$ between the candidate at step $t$ and the network measure at $t$ (the locked gradflow "
             "cross-correlation at lag 0)", ha="center", va="center", fontsize=FS["tick"])
    # a: the generalized-mean scan
    ax = ax_at(.85, 3.75, 4.3, 4.9)
    cols = plt.get_cmap("tab10")
    x = np.arange(len(PS))
    for j, (k, lab) in enumerate(NET):
        ax.plot(x, scan[:, j], color=cols(j), lw=1.9, marker="o", ms=3.5, label=lab.replace("\n", " "))
    ax.axhline(0, color="0.6", lw=.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(["min", "-4", "-2", "-1", "0\nsum of\nlogs", "1\naverage", "2", "4", "max"],
                       fontsize=FS["tick"])
    ax.set_xlabel(r"$p$ of the mean $\left(\frac{e_g^p + e_c^p}{2}\right)^{1/p}$", fontsize=FS["tick"] + .5)
    ax.set_ylabel(r"$r$ with the network (steps 17 - 30)", fontsize=FS["tick"] + .5)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS["tick"] - 1.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, .5))
    ax.set_title("a   Min, sum of logs, average or max of the two variables", fontsize=FS["label"] + .5,
                 loc="left", pad=5)
    # b - e: candidate x network heatmaps
    spots = {("late", "raw"): (8.85, 5.7), ("late", "beyond"): (13.0, 5.7),
             ("early", "raw"): (8.85, .9), ("early", "beyond"): (13.0, .9)}
    letters = {("late", "raw"): "b", ("late", "beyond"): "c", ("early", "raw"): "d", ("early", "beyond"): "e"}
    for (wn, kind), (x0, y0) in spots.items():
        M = out[wn][kind]
        lim = .5 if kind == "raw" else .25
        ax = ax_at(x0, y0, 3.3, 3.55)
        im = ax.imshow(M, cmap="PiYG", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.2f}".replace("+0.", "+.").replace("-0.", "-."), ha="center",
                            va="center", fontsize=FS["tick"] - 2, color="k" if abs(M[i, j]) < .7 * lim else "w")
        ax.axhline(7.5, color="k", lw=1.2)
        ax.axhline(3.5, color="k", lw=.6, ls=":")
        for g, lo, hi in GROUPS:
            if hi < len(NET):
                ax.axvline(hi - .5, color="k", lw=.8)
        ax.set_xticks(range(len(NET)))
        ax.set_xticklabels([lab for _, lab in NET], fontsize=FS["tick"] - 2, rotation=90)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names if x0 < 10 else [], fontsize=FS["tick"] - 1)
        for g, lo, hi in GROUPS:
            ax.text((lo + hi - 1) / 2, -.75, g, ha="center", va="bottom", fontsize=FS["tick"] - 1.5)
        title = (f"{letters[(wn, kind)]}   {'Late' if wn == 'late' else 'Early'} steps "
                 f"{WINDOWS[wn][0]} - {WINDOWS[wn][1]}, " + ("raw" if kind == "raw" else "beyond update size"))
        ax.set_title(title, fontsize=FS["label"] + .5, loc="left", pad=16)
        cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.02)
        cb.ax.tick_params(labelsize=FS["tick"] - 1)
    fig.text(.85 / W_, 2.2 / H_, "b - e: rows above the solid line are entanglement candidates (above the dotted "
             "line:\nthe two variables combined); below it, controls that are not entanglement.\n\nBeyond update size: "
             "each candidate residualized on the model-free novelty\nand change, the Optimal's information gain and "
             "surprise, and the Naive's\ninformation gain (the locked figure controlled novelty and change only).",
             ha="left", va="center", fontsize=FS["tick"], linespacing=1.4)
    path = fig_path("fig1c_entanglement_definitions.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("saved", path)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("p-scan rows", [str(p) for p in PS], "cols", [k for k, _ in NET])
    print(scan)
    for (wn, kind) in spots:
        print(wn, kind)
        for nm, row in zip(names, out[wn][kind]):
            print(f"  {nm:28s} " + " ".join(f"{v:+.3f}" for v in row))
    return path


if __name__ == "__main__":
    figure()
