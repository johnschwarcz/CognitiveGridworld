"""Data for the training-dynamics summary figure (fig_td_summary.py), computed from the repository's own data and cached
in main/DATA/experimental/training_dynamics/summary_data/ (moved 11-01 out of this session's scratch explorers; nothing
here reads the scratch folder):
  sii(run)      every test episode's SII score (logistic fit of correctness at T on factorization regret) and the
                majority rate max(acc_T, 1 - acc_T), averaged in log-spaced bins of training episodes; the run's own
                test_*.npz (8,000 trajectories each)
  readout(run)  the read-out's singular values per checkpoint (analysis/<run>/ck_*.npz: sv_ro)
  fr_acc(run)   per checkpoint, Pearson r between each variable's factorization regret at T and its own correctness
                at T (probes, both variables pooled), with Bayes's and naive's
  belief_spectrum_steps(run)  per checkpoint and step, the PCA spectrum of the network's marginal beliefs
                (probabilities) over the probes at that step and its PR, with Bayes's and naive's (panel f)
  state_spectrum_steps(run)   per checkpoint and step, u_t's PCA spectrum over the probes and its PR (panel g; built
                on GPU 0 once)
  belief_spectrum(run), spectrum(run)  superseded (11-01 draft): one cloud pooled over all steps (the drift of the mean
                counts as dimensions); kept for the comparison numbers in fig_td_summary's docstring
  credit(run)   per checkpoint, the update measures' (cr_coh, cr_E, cr_dim, cr_bandx, cr_bandx_fixed) quantiles and
                histograms over all (probe, error step) values (ck files: 5,000 probes x steps 17-30; fixed bands
                2,000 probes)
  grids()       belief grids (summary_data/grids/fwd_*.npz from td_grids_fwd.py: 50 environments x 100 cells x 24
                fresh trajectories at 19 checkpoints): each environment's grid of cell-mean posterior means; a grid
                triangle is folded when its signed area is negative (162 triangles); % folded at T where naive's grid is
                folded and elsewhere, and % folded at every step
  shares()      end of training, per environment (50) and inference step (30): the share of the cells' variance carried
                by one variable alone (the two-way split of the 100 cell means' variance into r0 main effect, r1 main
                effect and interaction, summed over dimensions, noise-corrected for M = 50 fresh particles per cell)
                for the ESN's and the Trained RNN's state u_t and read-out update (the per-step logit increment, centred
                over each variable's values; GPU forward at the last checkpoint), and for naive's and
                Bayes's beliefs (marginal probabilities, 20 values) and belief updates (the per-step change of the log
                marginal beliefs, centred over each variable's 10 values: what the network's read-out emits each
                step); plus each variable's exact share in the observation table p(r0, r1)
Run from the repository root (shares() and state_spectrum_steps() need GPU 0 once; everything else CPU):
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_summary_data.py"""
import os
import sys
import glob
import numpy as np
from scipy.special import logsumexp

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION")]
TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
OUT = os.path.join(TD, "summary_data")
SEED = 20261029                                     # the runs' seed
X0 = 10                                             # the initial network on log axes
T, R = 30, 10
CR = ("cr_coh", "cr_E", "cr_dim", "cr_bandx", "cr_bandx_fixed")
QS = np.array([.01, .05, .1, .25, .5, .75, .9, .95, .99])
BINS = {"cr_coh": np.geomspace(.05, 20, 81), "cr_E": np.geomspace(1e-3, 1e5, 81), "cr_dim": np.linspace(1, 4, 81),
        "cr_bandx": np.linspace(-6, 1, 81), "cr_bandx_fixed": np.linspace(-6, 1, 81)}
M_SHARE, SHARE_SEED = 50, 20261112


def _cached(name, build):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        return dict(np.load(path))
    out = build()
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    return out


def _ck_files(run):
    return sorted(glob.glob(os.path.join(AN, run, "ck_?????.npz")))


def train_x(ep):
    ep = np.asarray(ep)
    return np.maximum(ep - ep // 5, X0).astype(float)


# ── test records, read-out, update measures ─────────────────────────────────────────────────────────────────────────

def sii(run, n_bins=120, lo=12):
    def build():
        files = sorted(glob.glob(os.path.join(TD, f"{run}_s{SEED}", "test_*.npz")))
        z = [np.load(f) for f in files]
        ep = np.concatenate([q["episode"] for q in z]).astype(int)
        accT = np.concatenate([q["acc"][:, -1] for q in z]).astype(float)
        score = np.concatenate([q["SII_score"] for q in z]).astype(float)
        x = (ep - ep // 5).astype(float)
        edges = np.geomspace(lo, x.max() + 1, n_bins + 1)
        b = np.digitize(x, edges) - 1
        keep = [k for k in range(n_bins) if (b == k).any()]
        xs = np.r_[X0, [np.exp(np.log(x[b == k]).mean()) for k in keep]]
        maj = np.maximum(accT, 1 - accT)
        f = lambda v: np.r_[v[x == 0].mean(), [v[b == k].mean() for k in keep]]  # noqa: E731
        return {"x": xs, "score": f(score), "majority": f(maj), "n_tests": len(ep)}
    return _cached(f"sii_{run}.npz", build)


def readout(run):
    def build():
        fs = _ck_files(run)
        return {"x": train_x([int(np.load(f)["episode"]) for f in fs]), "sv": np.stack([np.load(f)["sv_ro"] for f in fs])}
    return _cached(f"readout_{run}.npz", build)


def credit(run):
    def build():
        fs = _ck_files(run)
        out = {"x": train_x([int(np.load(f)["episode"]) for f in fs])}
        for k in CR:
            q, h = np.zeros((len(fs), len(QS))), np.zeros((len(fs), len(BINS[k]) - 1))
            for c, f in enumerate(fs):
                v = np.load(f)[k].astype(np.float64).ravel()
                q[c] = np.quantile(v, QS)
                h[c] = np.histogram(np.clip(v, BINS[k][0], BINS[k][-1]), BINS[k])[0]
            out[f"{k}_q"], out[f"{k}_hist"] = q, h
        return out
    return _cached(f"credit_{run}.npz", build)


def fr_acc(run):
    """per checkpoint: Pearson r between each variable's factorization regret at T and its own correctness at T, over
    the 10,000 probes x both variables (20,000 judgements; the original fig1 (e) used the goal variable's: the same up
    to noise). Averaging FR and correctness over the two variables first (10-31's version) adds cross terms - one
    variable's FR correlates +.13 to +.17 with the OTHER variable's correctness for every observer - which pushed the
    averaged r up (Bayes +.12 instead of -.02; goal only -.03). "r_bayes" / "r_naive": the same for Bayes and naive (John 11-01)"""
    def build():
        P = np.load(os.path.join(AN, "probes.npz"))
        fr = P["fr"][:, -1].ravel()                                                   # (n * 2,) variable by variable
        r = lambda right: np.corrcoef(fr, right.astype(np.float64).ravel())[0, 1]    # noqa: E731
        fs = _ck_files(run)
        return {"x": train_x([int(np.load(f)["episode"]) for f in fs]),
                "r": np.array([r(np.load(f)["right"][:, -1]) for f in fs]),
                "r_bayes": r(P["right_bayes"]), "r_naive": r(P["right_naive"])}
    return _cached(f"fr_acc_own_{run}.npz", build)


def belief_spectrum(run):
    """per checkpoint: the eigenvalue spectrum (fractions of the variance) of the network's marginal beliefs as
    probabilities (20 numbers: 2 variables x 10 values) over all 10,000 probes x 30 steps, and its participation ratio;
    "pr_bayes" / "pr_naive": the same for Bayes's and naive's beliefs on the same probes"""
    def spec(p):
        X = p.reshape(-1, p.shape[-2] * p.shape[-1]).astype(np.float64)
        ev = np.clip(np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1], 0, None)
        return ev / ev.sum()
    def build():
        P = np.load(os.path.join(AN, "probes.npz"))
        fs = _ck_files(run)
        ev = []
        for f in fs:
            lq = np.load(f)["logq"].astype(np.float64)
            ev.append(spec(np.exp(lq - logsumexp(lq, -1, keepdims=True))))
        ev = np.array(ev)
        pr = lambda e: 1 / (e ** 2).sum(-1)                                           # noqa: E731
        return {"x": train_x([int(np.load(f)["episode"]) for f in fs]), "evr": ev, "pr": pr(ev),
                "pr_bayes": pr(spec(np.exp(P["log_jb"].astype(np.float64)))),
                "pr_naive": pr(spec(np.exp(P["log_nb"].astype(np.float64))))}
    return _cached(f"belief_spectrum_{run}.npz", build)


def _step_fractions(C):
    """(..., D, D) covariances -> (..., D) eigenvalue fractions, descending"""
    ev = np.clip(np.linalg.eigvalsh(C)[..., ::-1], 0, None)
    return ev / ev.sum(-1, keepdims=True)


def belief_spectrum_steps(run):
    """per checkpoint and step t: the PCA spectrum of the network's marginal beliefs as probabilities (20 numbers) over
    the 10,000 probes at that step (fractions of the step's variance) and its PR; "pr_bayes" / "pr_naive" (30,): the
    same for Bayes's and naive's beliefs. The figure averages over the 30 steps AFTER the per-step spectra and PRs
    (John 11-01: no averaging before the statistic): one cloud pooled over the steps (belief_spectrum, the 11-01 draft)
    also counts the drift of the average belief across steps (pooled PR 1-2 higher, initial network 2.2 vs 4.3)"""
    def spec(p):
        X = p.reshape(p.shape[0], p.shape[1], -1).astype(np.float64)                 # (n, T, 20)
        Xc = X - X.mean(0)
        f = _step_fractions(np.einsum("nti,ntj->tij", Xc, Xc) / len(X))
        return f, 1 / (f ** 2).sum(-1)
    def build():
        P = np.load(os.path.join(AN, "probes.npz"))
        fs = _ck_files(run)
        ev, pr = [], []
        for f in fs:
            lq = np.load(f)["logq"].astype(np.float64)
            e, p = spec(np.exp(lq - logsumexp(lq, -1, keepdims=True)))
            ev.append(e)
            pr.append(p)
        prob = lambda key: np.exp(P[key].astype(np.float64) - logsumexp(P[key].astype(np.float64), -1, keepdims=True))  # noqa: E731
        return {"x": train_x([int(np.load(f)["episode"]) for f in fs]), "evr": np.array(ev), "pr": np.array(pr),
                "pr_bayes": spec(prob("log_jb"))[1], "pr_naive": spec(prob("log_nb"))[1]}
    return _cached(f"belief_spectrum_steps_{run}.npz", build)


def state_spectrum_steps(run):
    """per checkpoint and step t: the PCA spectrum of the state u_t (1,000 neurons) over the 10,000 probes at that step
    (fractions of the step's variance, float32) and its PR (= td_analysis fac[t, u, total, PR]; checked). The figure
    averages over the 30 steps AFTER the per-step spectra and PRs: the pooled basis spectrum (spectrum(), 20,000 basis
    episodes x 30 steps in one cloud, the 11-01 draft) counts the drift of the average state across steps as dimensions
    (initial network: pooled PR 1.8, per step 10.7; end: Trained 74 vs 57, ESN 11 vs 18). GPU 0, ~3 s per checkpoint"""
    def build():
        import torch
        import td_analysis as td
        import td_entanglement_factors as tf
        dev = torch.device("cuda:0")
        P = td.probes()
        n = len(P["goal"])
        obs = torch.as_tensor(P["obs"], dtype=torch.float32, device=dev)
        zin = torch.as_tensor(P["Z"][P["env"]], device=dev).reshape(n, -1)
        fs = _ck_files(run)
        eps = [int(np.load(f)["episode"]) for f in fs]
        ev, pr = [], []
        for ep in eps:
            w = torch.load(td.ckpt_path(run, ep), map_location="cpu", weights_only=False)["weights"]
            p = {k: v.to(dev).float() for k, v in w.items()}
            C = []
            with torch.no_grad():
                for t, u in tf.states(p, obs, zin):
                    ud = u.double()
                    ud = ud - ud.mean(0)
                    C.append(ud.T @ ud / n)
                e = torch.linalg.eigvalsh(torch.stack(C)).flip(-1).clamp_min(0)
            f = (e / e.sum(-1, keepdim=True)).cpu().numpy()
            ev.append(f.astype(np.float32))
            pr.append(1 / (f ** 2).sum(-1))
            del C, e, p
            torch.cuda.empty_cache()
        return {"x": train_x(eps), "evr": np.array(ev), "pr": np.array(pr)}
    return _cached(f"state_spectrum_steps_{run}.npz", build)


def spectrum(run):
    """per checkpoint: the eigenvalue spectrum of u's covariance as fractions of the variance (td_analysis basis_evr:
    20,000 basis episodes x 30 steps pooled, 1,000 PCs) and its participation ratio PR = 1 / sum(fraction^2)"""
    def build():
        fs = _ck_files(run)
        ev = np.stack([np.load(f)["basis_evr"] for f in fs]).astype(np.float64)
        return {"x": train_x([int(np.load(f)["episode"]) for f in fs]), "evr": ev, "pr": 1 / (ev ** 2).sum(1)}
    return _cached(f"spectrum_{run}.npz", build)


# ── belief grids ────────────────────────────────────────────────────────────────────────────────────────────────────

S_GRID = 24
_i = lambda g, c: R * g + c                                                          # noqa: E731
TRI = np.array([(_i(g, c), _i(g + 1, c), _i(g + 1, c + 1)) for g in range(R - 1) for c in range(R - 1)] +
               [(_i(g, c), _i(g + 1, c + 1), _i(g, c + 1)) for g in range(R - 1) for c in range(R - 1)])


def folded(G):
    """(..., 162) True where the grid's triangle is turned over (negative signed area); G (..., 100, 2)"""
    a, b, c = G[..., TRI[:, 0], :], G[..., TRI[:, 1], :], G[..., TRI[:, 2], :]
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (c[..., 0] - a[..., 0]) * (b[..., 1] - a[..., 1]) < 0


def _per_env(E, m):
    x = np.zeros((50, R * R) + m.shape[1:])
    np.add.at(x, (E["env"], E["cell"]), m)
    return x / S_GRID


def grids():
    def build():
        d = os.path.join(OUT, "grids")
        E = np.load(os.path.join(d, "fwd_episodes.npz"))
        refs = {a: _per_env(E, E[f"m_{a}"][:, -1]) for a in ("naive", "bayes")}
        fN, fB = folded(refs["naive"]), folded(refs["bayes"])
        out = {"bayes_where_naive": 100 * fB[fN].mean(), "naive_folded": 100 * fN.mean()}
        for a in ("bayes", "naive"):
            out[f"steps_{a}"] = 100 * folded(_per_env(E, E[f"m_{a}"]).transpose(0, 2, 1, 3)).mean((0, 2))
        for run in ("trained", "echo"):
            F = np.load(os.path.join(d, f"fwd_{run}.npz"))
            out[f"x_{run}"] = np.where(F["tr_ep"] > 0, F["tr_ep"], X0).astype(float)
            fold_T = np.stack([folded(_per_env(E, F["m"][k][:, -1].astype(float))) for k in range(len(F["ep"]))])
            out[f"where_naive_{run}"] = 100 * fold_T[:, fN].mean(1)
            out[f"elsewhere_{run}"] = 100 * fold_T[:, ~fN].mean(1)
            out[f"steps_{run}"] = np.stack([100 * folded(_per_env(E, F["m"][k].astype(float)).transpose(0, 2, 1, 3))
                                            .mean((0, 2)) for k in range(len(F["ep"]))])
        return out
    return _cached("grids.npz", build)


# ── share carried by one variable alone ─────────────────────────────────────────────────────────────────────────────

def split(X):
    """(..., 100, H) cell means (r0-major) -> (..., 3): total cell-mean variance, r0 and r1 main effects (summed over
    dimensions)"""
    G = X.reshape(*X.shape[:-2], R, R, X.shape[-1]).astype(np.float64)
    mu = G.mean((-3, -2), keepdims=True)
    a, b = G.mean(-2, keepdims=True) - mu, G.mean(-3, keepdims=True) - mu
    return np.stack([((G - mu) ** 2).sum(-1).mean((-2, -1)), (a ** 2).sum(-1).mean((-2, -1)),
                     (b ** 2).sum(-1).mean((-2, -1))], -1)


def shares_of(sp, M=M_SHARE):
    """noise-corrected shares (..., 3) [r0 main, r1 main, interaction] from rows [total, r0, r1, w] (cell means of M
    particles: E[total] = true + .99 w / M, E[main] = true + .09 w / M; w = the mean within-cell variance)"""
    tot, a, b, w = (sp[..., k] for k in range(4))
    tot, a, b = tot - .99 * w / M, a - .09 * w / M, b - .09 * w / M
    return np.stack([a / tot, b / tot, (tot - a - b) / tot], -1)


def _particles(P, M=M_SHARE, seed=SHARE_SEED):
    rng = np.random.default_rng(seed)
    env = np.repeat(np.arange(50), R * R * M)
    cell = np.tile(np.repeat(np.arange(R * R), M), 50)
    p = P["L"][env, :, cell // R, cell % R]
    return env, cell, rng.random((len(env), T, p.shape[1])) < p[:, None, :]


def _observer_splits(P, obs, M=M_SHARE):
    """per environment and step, [total, r0, r1, w] of naive's / Bayes's beliefs and belief updates"""
    out = {f"{a}_{k}": np.zeros((50, T, 4)) for a in ("naive", "bayes") for k in ("belief", "update")}
    n = R * R * M
    for e in range(50):
        o = obs[e * n:(e + 1) * n].astype(np.float64)
        L = P["L"][e]
        ll = np.einsum("nti,ijk->ntjk", o, np.log(L)) + np.einsum("nti,ijk->ntjk", 1 - o, np.log1p(-L))
        lam = np.cumsum(ll, 1)
        lp = lam - logsumexp(lam.reshape(n, T, -1), -1)[..., None, None]
        lb = np.stack([logsumexp(lp, 3), logsumexp(lp, 2)], 2)                       # Bayes's log marginals
        Lb = np.stack([L.mean(2), L.mean(1)], 1)                                     # naive's tables (5, 2, R)
        cl = np.cumsum(np.einsum("nti,ikr->ntkr", o, np.log(Lb)) + np.einsum("nti,ikr->ntkr", 1 - o, np.log1p(-Lb)), 1)
        ln = cl - logsumexp(cl, -1)[..., None]                                       # naive's log marginals
        for a, lm in (("bayes", lb), ("naive", ln)):
            prev = np.concatenate([np.full_like(lm[:, :1], -np.log(R)), lm[:, :-1]], 1)
            upd = lm - prev
            upd -= upd.mean(-1, keepdims=True)
            for k, X in (("belief", np.exp(lm)), ("update", upd)):
                Xr = X.reshape(R * R, M, T, 2 * R)
                w = Xr.var(1, ddof=1).sum(-1).mean(0)                                # (T,)
                out[f"{a}_{k}"][e] = np.concatenate([split(Xr.mean(1).transpose(1, 0, 2)), w[:, None]], 1)
    return out


def _network_splits(run, P, env, obs, ep=49995, M=M_SHARE):
    import torch
    import td_analysis as td
    import gradflow_analysis as ga
    dev = torch.device("cuda:0")
    m = td.build(run, ep, dev)
    td.load_weights(m, run, ep)
    W_ih, W_hh, b = ga.lstm_weights(m)
    out, ro = np.zeros((50, T, 4)), np.zeros((50, T, 4))                           # state u_t; read-out update
    n5 = 5 * R * R * M
    with torch.no_grad():
        for e0 in range(0, 50, 5):
            sl = slice(e0 * R * R * M, e0 * R * R * M + n5)
            o = torch.as_tensor(obs[sl], dtype=torch.float32, device=dev)
            zin = torch.as_tensor(P["Z"][env[sl]], device=dev).reshape(len(o), -1)
            h, c = td._start(m, len(o))
            for t in range(T):
                h, c, _ = td._cell(td._readin(m, o[:, t], zin) @ W_ih.T + b + h @ W_hh.T, c)
                for dst, Y in ((out, h.double()), (ro, m.classifier_readout(h).double().reshape(len(o), 2, R))):
                    if Y.ndim == 3:                                                  # the per-step logit increment,
                        Y = (Y - Y.mean(-1, keepdim=True)).reshape(len(o), -1)       # centred like Bayes's update
                    X = Y.reshape(5, R * R, M, -1)
                    dst[e0:e0 + 5, t, :3] = split(X.mean(2).cpu().numpy())
                    dst[e0:e0 + 5, t, 3] = X.var(2, unbiased=True).sum(-1).mean(-1).cpu().numpy()
    return out, ro


def obs_shares(P):
    """(50, 2) each variable's exact share of the observation table's variance across the 100 cells"""
    s = split(P["L"].transpose(0, 2, 3, 1).reshape(50, R * R, -1))
    return np.stack([s[:, 1] / s[:, 0], s[:, 2] / s[:, 0]], 1)


def shares():
    def build():
        P = dict(np.load(os.path.join(AN, "probes.npz")))
        env, cell, obs = _particles(P)
        out = {k: shares_of(v) for k, v in _observer_splits(P, obs).items()}
        for run in ("trained", "echo"):
            st, ro = _network_splits(run, P, env, obs)
            out[f"{run}_state"], out[f"{run}_readout_update"] = shares_of(st), shares_of(ro)
        out["obs"] = obs_shares(P)
        out["M"] = M_SHARE
        return out
    return _cached("shares.npz", build)


if __name__ == "__main__":
    for r in ("trained", "echo"):
        print(r, "sii", sii(r)["n_tests"], "tests; readout", readout(r)["sv"].shape, "; credit", credit(r)["x"].shape)
    g = grids()
    print("grids: naive folds", round(float(g["naive_folded"]), 1), "% of its triangles")
    s = shares()
    for k in ("naive_belief", "naive_update", "echo_state", "echo_readout_update", "trained_state",
              "trained_readout_update", "bayes_belief", "bayes_update"):
        v = s[k][..., :2]
        print(f"  {k:14s} share at t = 1 / 30: mean {v[:, 0].mean():.2f} / {v[:, -1].mean():.2f}, SD {v[:, 0].std():.3f} / "
              f"{v[:, -1].std():.3f}, r with the observations at 30 {np.corrcoef(v[:, -1].ravel(), s['obs'].ravel())[0, 1]:.2f}")
