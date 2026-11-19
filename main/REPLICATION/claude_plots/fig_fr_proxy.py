"""Is the divergence between the Trained RNN's and the ESN's marginal beliefs a proxy for Factorization Regret?
(John 11-18 08:10 ET: "can we check if the divergence between trained and ESN marginal beliefs is a good proxy for
Factorization Regret? (and show it visually)")

Data: the fig1c alias scan (main/DATA/experimental/fig1c/scan_*.npz; plot_fig1c_all.py): 40 environments drawn at
random (seed 42) x 100 states (every joint value) x 50 trajectories, T = 30; e = 2 (an exactly symmetric table) left out
as in every fig1c analysis -> 39 environments x 5,000 trajectories. The Trained RNN and the ESN
(sanity/{fully_trained,reservoir}_ctx_2_e5_net.pth) were replayed on the same trajectories; their marginal beliefs are
stored as float16. Every quantity is per trajectory, step and variable, in nats, averaged over the two variables after
it is computed:
    FR        = D_KL(B^Joint || B^Naive), exact (plot_fig1c_all.fr_streams: log space, nothing clipped)
    proxy     = D_KL(B^Trained || B^ESN), both networks replayed in float64 log space on the scan's trajectories
                (replay_log; the stored float16 beliefs agree to 2e-3 but underflow: ESN log probabilities reach -208)
    controls  = D_KL(B^Joint || B^ESN) (the Trained RNN replaced by the exact Joint) and D_KL(B^Trained || B^Naive)
                (the ESN replaced by the exact Naive, in log space)
Rank correlations are taken within one environment, step and variable across its 5,000 trajectories, then averaged.
Cache: main/DATA/experimental/fr_proxy/fr_proxy.npz (~10 min to build at 8 threads; also holds proxy_f16, the same
divergence from the stored float16 beliefs with ESN zeros at 2^-25).

RESULT 11-18 (exact replays; the float16 beliefs give the same ranks, r .700, but cap the divergence near 10 nats):
    Within an environment and step (Spearman, mean of 39 environments and both variables): Trained vs ESN .44 (t = 1),
    .75 (t = 5), .73 (t = 10), .70 (t = 20), .69 (t = 30), .70 over all steps; exact Joint vs ESN .72 over steps (no
    better); Trained vs exact Naive .98 (.82 at t = 1): the ESN, not the Trained RNN, limits the proxy.
    States at T (39 x 100, means of 50 trajectories, log r): .81; the median environment's (e = 31) .88; environment
    means r .84. Size: median divergence / FR at T 1.32 (quartiles .68-2.36; per-environment medians .89-2.52); the
    median curve follows equality from ~0.3 to ~30 nats and lies above it below ~0.1 nats (the ESN's own error).

--big (fig_fr_proxy_big.png, John 11-18): 400 environments (build_big: 2,450 non-symmetric harvest environments, 400
drawn with seed 1) x 5,000 x 30 replayed on a GPU. RESULT: within-step r .698 (.41 at t = 1, .74 at t = 5, .69 at T);
exact Joint vs ESN .712; Trained vs exact Naive .978; state r per environment median .84 (quartiles .75-.89);
divergence / FR at T median 1.30 (quartiles .68-2.42); panel (b) draws 99.6% of the 60 M steps (both >= 10^-2 nats).
Panel (a) comes from fig_fr_proxy_training.py's caches (build_big, build_dense: run those first). Fit in (b): log10 D =
.004 + .807 log10 FR on the drawn steps (>= 10^-3 nats: all but 0.04%), log r .786; it crosses equality at ~1 nat
(slope < 1: the network divergence overestimates FR below ~0.3 nats, the ESN's own error, as the median showed).

    python fig_fr_proxy.py [--rebuild]
    python fig_fr_proxy.py --big [--rebuild] [--vector]
"""
import os, sys, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "main", "REPLICATION"))
from main.utils import fig_path, apply_fig_style

SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c")
CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy.npz")
FLOOR = 2.0 ** -25
KEYS = ("fr", "proxy", "joint_esn", "trained_naive")
C_PROXY, C_JE, C_TN = "k", "#6a51a3", "#1b9e77"


def _kl(p, logq):
    """sum_x p (log p - log q) over the last axis, 0 log 0 = 0"""
    return np.where(p > 0, p * (np.log(np.where(p > 0, p, 1.0)) - logq), 0.0).sum(-1)


def _net(b):
    b = b.astype(np.float64).reshape(b.shape[:-1] + (2, 10))
    return b / b.sum(-1, keepdims=True)


def log_naive(obs, L):
    """the Naive's log marginals (n, T, 2, R), raw order: plot_fig1c_all.naive_marginals without leaving log space"""
    from scipy.special import logsumexp
    o = obs.astype(np.float64)
    Lb = (L.mean(2), L.mean(1))
    cum = np.cumsum(np.stack([o @ np.log(Lb[k]) + (1 - o) @ np.log1p(-Lb[k]) for k in (0, 1)], 2), 1)
    return cum - logsumexp(cum, -1, keepdims=True)


def replay_log(F, net, obs, Z, threads=8):
    """plot_fig1c_all.replay_beliefs in log space: log_softmax of the summed read-out logits in float64 -> (n, T, 2, R),
    raw order (the stored float16 beliefs underflow: the ESN's log probabilities reach -208)"""
    import torch
    torch.set_num_threads(threads)
    W, out = F.checkpoint(net)["W"], []
    with torch.no_grad():
        for b0 in range(0, len(obs), F.BATCH):
            o = torch.from_numpy(obs[b0:b0 + F.BATCH].astype(np.float32))
            z = torch.from_numpy(Z[b0:b0 + F.BATCH].astype(np.float32))
            r = F._logits(W, o, z)
            out.append(r.double().cumsum(1).reshape(len(o), F.N_T, 2, F.N_R).log_softmax(-1).numpy())
    return np.concatenate(out)


def build():
    """~15 s per environment at 8 threads (the replays), ~10 min"""
    import plot_fig1c_all as F
    M = np.load(os.path.join(SCAN, "scan_meta.npz"))
    obs, L, goal, fr_goal, cell_of, eps = M["obs"], M["L"], M["goal"], M["FR"], M["cell_of"], M["eps"]
    envs = [e for e in range(F.SCAN_N_ENV) if e != F.SCAN_LEFT_OUT]
    T16 = np.load(os.path.join(SCAN, "scan_Trained.npz"))["beliefs"]
    E16 = np.load(os.path.join(SCAN, "scan_Echo.npz"))["beliefs"]
    Zall = F.record()["ep_active_Z"]
    out = {k: np.zeros((len(envs),) + obs.shape[1:3] + (2,), np.float32) for k in KEYS + ("proxy_f16",)}
    for i, e in enumerate(envs):
        g = [goal[e], 1 - goal[e]]                                       # raw -> goal-ordered, as the stored beliefs
        fr = F.fr_streams(obs[e], L[e])[..., g]
        assert np.allclose(fr[..., 0], fr_goal[e], rtol=1e-4, atol=1e-6)
        J = F.joint_marginals(obs[e], L[e])[:, :, g]
        logN = log_naive(obs[e], L[e])[:, :, g]
        assert np.allclose(_kl(J, logN), fr, rtol=1e-4, atol=1e-6), "the Naive in log space does not reproduce FR"
        Z = np.repeat(np.asarray(Zall[eps[e]:eps[e] + 1]), len(obs[e]), 0)
        logT, logE = (replay_log(F, n, obs[e], Z)[:, :, g] for n in ("trained", "echo"))
        for lg, b16 in ((logT, T16[e]), (logE, E16[e])):                 # the same streams and networks as the scan
            assert np.abs(np.exp(lg) - _net(b16)).max() < 2e-3
        pT = np.exp(logT)
        out["fr"][i], out["proxy"][i] = fr, _kl(pT, logE)
        out["joint_esn"][i], out["trained_naive"][i] = _kl(J, logE), _kl(pT, logN)
        out["proxy_f16"][i] = _kl(_net(T16[e]), np.log(np.maximum(_net(E16[e]), FLOOR)))
        print(f"environment {e}: done", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, envs=np.array(envs), cell_of=cell_of, **out)


# ---- the larger sample (John 11-18: "expand this with more data so we can see it more smoothly") ----------------
BIG = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_big.npz")
N_BIG, BIG_SEED, BIG_STREAM0 = 400, 1, 3_000_000
_GPU_W = {}


def logits_gpu(W, obs, Z):
    """plot_fig1c_all._logits with its state on the tensors' device (the original allocates it on the CPU); same ops in
    the same order"""
    import torch
    Fn = torch.nn.functional
    B, T = obs.shape[:2]
    Zx = Z.float().reshape(B, 1, -1).expand(-1, T, -1)
    x = torch.relu(Fn.linear(torch.cat((obs.float(), Zx), -1), W["classifier_readin.weight"], W["classifier_readin.bias"]))
    W_ih, W_hh = W["LSTM.weight_ih_l0"], W["LSTM.weight_hh_l0"]
    b = W["LSTM.bias_ih_l0"] + W["LSTM.bias_hh_l0"]
    H = x.shape[-1]
    u = W["STM"].reshape(1, H).expand(B, H)
    c = W["LTM"].reshape(1, H).expand(B, H)
    xa = x @ W_ih.T + b
    U = torch.empty(B, T, H, device=x.device)
    for s in range(T):
        i, f, g, o = (xa[:, s] + u @ W_hh.T).chunk(4, -1)
        i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
        c = f * c + i * g
        u = o * c.tanh()
        U[:, s] = u
    return Fn.linear(U, W["classifier_readout.weight"], W["classifier_readout.bias"])


def replay_log_gpu(F, net, obs, Z, dev):
    """replay_log on a GPU, all streams in one batch -> (n, T, 2, R) float64 log beliefs, raw order"""
    import torch
    if net not in _GPU_W:
        _GPU_W[net] = {k: v.to(dev) for k, v in F.checkpoint(net)["W"].items()}
    with torch.no_grad():
        r = logits_gpu(_GPU_W[net], torch.from_numpy(obs.astype(np.float32)).to(dev),
                       torch.from_numpy(np.asarray(Z, np.float32)).to(dev))
        return r.double().cumsum(1).reshape(len(obs), F.N_T, 2, F.N_R).log_softmax(-1).cpu().numpy()


def environments_big(F, n=N_BIG, seed=BIG_SEED):
    """n harvest environments drawn as plot_fig1c_all.scan_environments draws the scan's (the first episode of every
    distinct (table, goal) among the first N_EP), exactly symmetric tables left out (97 of 2,547), default_rng(seed),
    sorted -> eps, every episode's table (N_EP, 5, R, R)"""
    Lall = np.array(F.episodes("ep_joint_likelihood")[:F.N_EP], np.float64)
    G = F.record()["ep_goal_ind"][:F.N_EP].astype(int)
    _, env = np.unique(np.round(Lall.reshape(F.N_EP, -1), 6), axis=0, return_inverse=True)
    env, first = env.ravel(), {}
    for ep in range(F.N_EP):
        first.setdefault((env[ep], G[ep]), ep)
    pool = [ep for ep in sorted(first.values()) if not np.allclose(Lall[ep], Lall[ep].transpose(0, 2, 1))]
    return np.array(sorted(np.random.default_rng(seed).choice(pool, n, replace=False).tolist())), Lall


def build_big(n=N_BIG):
    """N_BIG environments x 100 states x 50 trajectories (seed BIG_STREAM0 + episode), the networks replayed on a GPU
    (checked first against the CPU replay of the 39-environment cache's first environment)"""
    import time, torch
    import plot_fig1c_all as F
    dev = f"cuda:{torch.cuda.device_count() - 1}"
    R = F.record()
    Zall, G = R["ep_active_Z"], R["ep_goal_ind"].astype(int)
    M, old = np.load(os.path.join(SCAN, "scan_meta.npz")), np.load(CACHE)
    e0 = int(old["envs"][0])
    gi = [M["goal"][e0], 1 - M["goal"][e0]]
    Z = np.repeat(np.asarray(Zall[M["eps"][e0]:M["eps"][e0] + 1]), len(M["obs"][e0]), 0)
    lT, lE = (replay_log_gpu(F, k, M["obs"][e0], Z, dev)[:, :, gi] for k in ("trained", "echo"))
    d = np.abs(_kl(np.exp(lT), lE) - old["proxy"][0])
    print(f"GPU vs CPU replay, scan environment {e0}: proxy max abs diff {d.max():.2e} nats", flush=True)
    assert d.max() < 1e-2 * max(1, old["proxy"][0].max() / 100)
    eps, Lall = environments_big(F, n)
    out = {k: np.zeros((n, len(F.SCAN_CELLS) * F.SCAN_M, F.N_T, 2), np.float32) for k in KEYS}
    t0 = time.time()
    for i, EP in enumerate(eps):
        L, g = Lall[EP], G[EP]
        gi = [g, 1 - g]
        obs, cell_of = F.cell_streams(L, g, F.SCAN_CELLS, F.SCAN_M, BIG_STREAM0 + int(EP))
        fr = F.fr_streams(obs, L)[..., gi]
        J = F.joint_marginals(obs, L)[:, :, gi]
        logN = log_naive(obs, L)[:, :, gi]
        Z = np.repeat(np.asarray(Zall[EP:EP + 1]), len(obs), 0)
        lT, lE = (replay_log_gpu(F, k, obs, Z, dev)[:, :, gi] for k in ("trained", "echo"))
        pT = np.exp(lT)
        out["fr"][i], out["proxy"][i] = fr, _kl(pT, lE)
        out["joint_esn"][i], out["trained_naive"][i] = _kl(J, lE), _kl(pT, logN)
        if i % 20 == 0 or i == n - 1:
            print(f"environment {i + 1}/{n}: {time.time() - t0:.0f} s", flush=True)
    np.savez(BIG, eps=eps, goal=G[eps], cell_of=cell_of, **out)


def spearman_within(x, y):
    """rank correlation across trajectories (axis 1) for every environment, step and variable -> (env, T, 2)"""
    from scipy.stats import rankdata
    rx, ry = rankdata(x, axis=1), rankdata(y, axis=1)
    rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
    return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))


TRAIN_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_big.npz")
DENSE_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_dense.npz")
EXT_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_ext.npz")
E_MAX = 2000                                                           # panel (a)'s last episode (John 11-18)
A_XSCALE = "linear"                                                    # panel (a)'s x axis (John 11-18: "look at panel a
                                                                       # ... with a linear x-axes"; "log" before)
EARLY_R = "pearson"                                                    # which correlation panel (a) draws



D500_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_dense500.npz")
FULL_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_full.npz")  # > 2,000
A_TAG = ""                                                             # output-name tag (e.g. "_full" with E_MAX 50,000)
A_TITLE = "Early learning"                                             # panel (a)'s title
C_DROP = (2493, 329)                                                   # one-row maps: examples left out (John 11-18)
A_LEG = {}                                                             # panel (a) legend placement overrides
A_LEG_TITLE = r"$D_{KL}(B^{\mathrm{Trained}}\,\|\,B^{\mathrm{Echo}})$ correlated with"   # John 11-18: math, one line
A_LABELS = {EARLY_R: "Factorization Regret", "count": "Observation count"}  # spelled out (John 11-18)
A_CURVES = (EARLY_R, "count")                                          # (EARLY_R,): without the count curve
A_YMAX = 1.                                                            # panel (a)'s R^2 axis top
A_ZERO = False                                                         # prepend episode 0 at R^2 = 0 (_with_zero)
A_XMIN = 0                                                             # panel (c)'s left x limit (episodes; ticks from 0)
A_LW = 1.1                                                             # panel (c)'s curve width (pt)
B_FILL = False                                                         # BOX_ALIGN: (a), (b) down to (c)'s x-label line
A_TALL = False                                                         # BOX_ALIGN: (c)'s axis up into the free title line
A_PA = 1.85                                                            # panel (a)'s width (in)
A_TOP = .30                                                            # room above the titles (in)
A_LAST = False                                                         # the R^2 panel last (John 11-18)
A_LEG_FS = None                                                        # legend font size (None: FT - 1.5)
B_SWAP = False                                                         # histogram axes swapped (John 11-18)
B_UNDER = "min"                                                        # empty bins: "min" colour or "black"
B_DARK = 1.                                                            # brightness of the lowest count (1: plain viridis)
B_LO = 0.                                                              # viridis from here up (.15 tried: lighter minimum)
B_VMIN = 10                                                            # counts below share the empty bins' colour
B_NORM = "linear"                                                      # histogram colour: "linear" density or "log" counts
B_LIM = (10 ** -2.25, 10 ** 1.75)                                      # histogram axes: the bins above 1% of the peak (1e-3..1e2 before)
B_BPD = 14                                                             # histogram bins per decade (12 before)
C_SWAP = True                                                          # maps: examples as rows, Bayes | Network columns
GAP_AB, GAP_BC = None, .58                                             # A_LAST gaps (in) before the maps / the R^2 axis
BOX_ALIGN = False                                                      # all three boxes share top and bottom (John 11-18)
A_PH, RCL, LH = 1.51, .16, .16                                         # BOX_ALIGN: box height, r_c baseline, title line step
B_TITLE = "Divergence correspondence"                                  # the histogram's title
SAVE_TIGHT = False                                                     # crop the empty margins when saving
C_TITLE = "Representative contexts"                                    # the maps' title (John 11-18)
REP_CACHES = [os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", f"fr_proxy_training_rep{r}.npz")
              for r in range(1, 9)]                                    # fig_fr_proxy_training.build_replicate(1..8)
A_RUNS = "mean"                                                        # panel (a): "mean" = the original run and its
                                                                       # replicates averaged at their shared checkpoints
                                                                       # (every 25 to 500, every 100 to 2,000), band = +-1
                                                                       # SEM over runs; "one" = the original run alone at
                                                                       # every replayed checkpoint, band = SEM over envs


def _rows(paths):
    """{episode: {measure: (env, T, 2) correlations}} over the caches (first cache wins), episodes 1..E_MAX"""
    rows = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        Z_ = np.load(path)
        for i, e in enumerate(Z_["E"]):
            if 0 < e <= E_MAX and int(e) not in rows:
                rows[int(e)] = {k: Z_[k][i] for k in (EARLY_R, "count")}
    return rows


def _r2(r, final):
    """R^2 per environment, step and variable, averaged over the variables and the steps (or step T) -> (..., env)"""
    rr = r ** 2
    return rr[..., -1, :].mean(-1) if final else rr.mean(-1).mean(-1)


def _lin_ticks(ax, n):
    """episode ticks from 0 at the smallest round step (1, 2, 2.5 or 5 x 10^k) giving at most n intervals to E_MAX,
    labels with thousands separators"""
    k = 10 ** np.floor(np.log10(E_MAX / n))
    step = min(m * k for m in (1, 2, 2.5, 5, 10) if m * k >= E_MAX / n - 1e-9)
    t = np.arange(0, E_MAX * 1.02 + 1, step)
    ax.set_xticks(t, [f"{v:,.0f}" for v in t])


def early_series(final, runs=None):
    """panel (a)'s data, sorted by episode -> E, {measure: (mean R^2, band)}. runs="one": every replayed checkpoint of
    the original run (dense500 both halves, dense, big, ext caches; the re-runs' weights are bit-identical to the
    original's), band = SEM over the 50 environments; runs="mean" (A_RUNS): the original run and the replicates with a
    cache, each run's R^2 averaged over environments first, then over runs, at the checkpoints all of them have; band =
    SEM over runs (falls back to "one" without replicates)"""
    runs = runs or A_RUNS
    orig = _rows((D500_CACHE, D500_CACHE.replace(".npz", "_b.npz"), DENSE_CACHE, TRAIN_CACHE, EXT_CACHE, FULL_CACHE))
    reps = [_rows([p, p.replace(".npz", "_ext.npz")]) for p in REP_CACHES if os.path.exists(p)]   # + 2,100-3,000
    curves = {}
    if runs == "mean" and reps:
        E = np.array(sorted(set(orig).intersection(*map(set, reps))))
        for k in (EARLY_R, "count"):
            M = np.stack([_r2(np.stack([R[e][k] for e in E]), final).mean(-1) for R in [orig] + reps])   # (run, E)
            curves[k] = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
        return _with_zero(E, curves)
    E = np.array(sorted(orig))
    for k in (EARLY_R, "count"):
        rr = _r2(np.stack([orig[e][k] for e in E]), final)                # (E, env)
        curves[k] = rr.mean(1), rr.std(1, ddof=1) / np.sqrt(rr.shape[1])
    return _with_zero(E, curves)


def _with_zero(E, curves):
    """A_ZERO (John 11-18: "add a point for 0, 0"): episode 0 prepended with R^2 = 0 and no band. At episode 0 the
    Trained and Echo networks of every run are identical (weights equal, checked in all nine runs), so the divergence is
    exactly 0 on every trajectory and its correlation undefined: plotted as nothing explained. Linear x only."""
    if not A_ZERO or A_XSCALE != "linear" or (len(E) and E[0] == 0):
        return E, curves
    return np.r_[0, E], {k: (np.r_[0., mu], np.r_[0., se]) for k, (mu, se) in curves.items()}


def big_figure(vector=False, final=False):
    """John 11-18 (several rounds), drawn at the paper's width (5.5 in, 7-7.5 pt):
    (a) "Early learning": the Trained-vs-ESN divergence's Pearson correlation with FR over the first 2,000 training
        episodes (1,000 before), log scale (fig_fr_proxy_training.build_big / build_dense: the training-dynamics pair replayed on 50
        environments x 5,000 trajectories; every 5 episodes to 100 from the re-runs <net>_s20261029_ep105, whose
        weights are bit-identical to the original run's at 0, 25, 50, 75 and 100; every 25 to 500 and every 100 to
        2,000 from the original run (fr_proxy_training_ext.npz: 1,100-2,000); r within an environment, step and variable, averaged; band = +-1 SEM over
        environments; within-environment sampling noise is negligible at 5,000 trajectories, and one training run means
        no seed-to-seed spread); red = with FR, black (John 11-18, "so we can explain the early bump") = the
        same divergence's Pearson r with the observation count (observations equal to 1 so far), which FR also tracks
        weakly (fr_count); lines without markers; since (John: "let's see R^2") the curves are R^2 = r^2 per
        environment, step and variable, averaged (squaring first matters: early correlations change sign between
        environments and steps, so mean r understates them; e.g. count at 75 episodes: mean r -.02, mean R^2 .49);
    (b) every trajectory step of the 400-environment sample, both axes from 10^-3 nats (steps below on either axis are
        not drawn), square, dashed = equality, black = no step (the colour bar's black end); the least-squares line
        (slope .81 / .71 at T) removed (John: distracting); axes labelled by their KL only;
    (c) "Context-specific divergences" (John 11-18: no "Example"): two example environments' 100 states, at
        T with final=True and averaged over all 30 steps otherwise (John: "for the version with all steps please have
        panel c use all steps as well"); the examples = the 7/8 and 3/8 quantiles of the state correlation (log
        values) in the same version (at T: .91 and .80, not printed since 11-18), rows r_g (the goal
        variable's value, upward), columns r_c, Bayes | Network divergence on one colour scale, the maps extended
        below (b)'s axis so the column has no empty strip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    from matplotlib.colors import LogNorm
    FS, FT = 9, 8.5                                                       # 7.3 / 6.9 pt if scaled to 5.5 in
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FT, "ytick.labelsize": FT,
        "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6, "xtick.minor.width": .4,
        "ytick.minor.width": .4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5, "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2, "axes.titlepad": 4,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.facecolor": "white"})
    Z = np.load(BIG)
    fr, px, cell_of = Z["fr"].mean(-1), Z["proxy"].mean(-1), Z["cell_of"]  # both-variable means (env, traj, T)
    n_env = fr.shape[0]
    x, y = (fr[:, :, -1].ravel(), px[:, :, -1].ravel()) if final else (fr.ravel(), px.ravel())
    lo, hi = 1e-3, 1e2                                                     # John 11-18: back out to 10^-3
    bins = np.logspace(np.log10(lo), np.log10(hi), int(round(12 * np.log10(hi / lo))) + 1)
    H, _, _ = np.histogram2d(x, y, [bins, bins])
    def state_maps(per_traj):                                              # (env, 2, 100) state means
        S_ = np.stack([np.stack([np.bincount(cell_of, per_traj(q, e), 100) for q in (fr, px)]) for e in range(n_env)])
        return S_ / np.bincount(cell_of)
    S_T = state_maps(lambda q, e: q[e, :, -1])
    S = S_T if final else state_maps(lambda q, e: q[e].mean(1))           # at T, or the mean over steps
    # examples chosen at T in both versions (John 11-18 asked why (c) changed between them: each version had picked
    # its own environments; within an environment the T and all-step maps agree, log r .99 median over 400)
    r_env = np.array([np.corrcoef(np.log(S_T[e]))[0, 1] for e in range(n_env)])
    ex = [int(np.argsort(r_env)[int(q * (n_env - 1))]) for q in (7 / 8, 3 / 8)]
    E, curves = early_series(final)
    rE, sem = curves[EARLY_R]
    fit_m = (x >= lo) & (y >= lo) & (x <= hi) & (y <= hi)
    lx, ly = np.log10(x[fit_m]), np.log10(y[fit_m])
    slope = np.cov(lx, ly)[0, 1] / lx.var(ddof=1)
    icpt = ly.mean() - slope * lx.mean()

    W = 6.85                                                               # inches (John 11-18: wider, more space)
    LM, PA, G1, PS, CBG, CB, CBL, G2, MG, MV, BOT, TOP, CBOT = .52, 1.0, .80, 1.40, .05, .07, .40, .48, .05, .08, .55, \
        .32, .38
    top_c = BOT + PS - .115                                                # (c)'s title level with (b)'s one-line title
    MS = (top_c - CBOT - MV) / 2
    Hh = BOT + PS + TOP
    fig = plt.figure(figsize=(W, Hh))
    # (a)
    aa = fig.add_axes([LM / W, BOT / Hh, PA / W, PS / Hh])
    lines = []
    for k, c, lab in ((EARLY_R, "#e41a1c", "FR"), ("count", "k", "obs. count")):
        mu, se = curves[k]
        aa.fill_between(E, mu - se, mu + se, color=c, alpha=.2, lw=0)
        lines += aa.plot(E, mu, color=c, lw=1.1, label=lab)
    lo_a = min(min(mu - se) for mu, se in curves.values())
    hi_a = max(max(mu + se) for mu, se in curves.values())
    aa.set_xscale(A_XSCALE)
    if A_XSCALE == "log":
        aa.set(xlim=(E.min() * .8, E_MAX * 1.2))
    else:
        aa.set(xlim=(0, E_MAX * 1.02))
        _lin_ticks(aa, 2)
    aa.set(ylim=(0, hi_a + .3), xlabel="Training episodes", ylabel="$R^2$")
    lg_a = aa.legend(title="Divergence vs.", loc="upper right", frameon=False, fontsize=FT, title_fontsize=FT,
                     handlelength=1.0, handletextpad=.4, borderaxespad=.15, labelspacing=.2, borderpad=.1)
    lg_a._legend_box.align = "left"
    if A_XSCALE == "log":
        aa.xaxis.set_major_locator(mt.LogLocator(numticks=5))
    aa.set_title(A_TITLE)
    # (b)
    xb = LM + PA + G1
    ab = fig.add_axes([xb / W, BOT / Hh, PS / W, PS / Hh])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("k")                                                    # bins with no step: black, below 1
    pm = ab.pcolormesh(bins, bins, np.where(H.T > 0, H.T, .5), cmap=cmap, norm=LogNorm(1, H.max()), rasterized=True)
    ab.plot([lo, hi], [lo, hi], color="k", ls=(0, (3, 2)), lw=.8)

    ab.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ab.set_xlabel(r"$D_{KL}(B^{\mathrm{Joint}}\,\|\,B^{\mathrm{Naive}})$")
    ab.set_ylabel(r"$D_{KL}(B^{\mathrm{Trained}}\,\|\,B^{\mathrm{ESN}})$")
    ab.set_title("Divergence correspondence")                             # John 11-18: one line
    ab.spines[["top", "right"]].set_visible(True)
    xc = xb + PS + CBG
    cax = fig.add_axes([xc / W, BOT / Hh, CB / W, PS / Hh])
    cb = fig.colorbar(pm, cax=cax, extend="min", extendfrac=.06)
    cb.set_label("Bin count", labelpad=2)
    cb.ax.yaxis.set_major_locator(mt.LogLocator(numticks=4))
    cb.ax.tick_params(labelsize=FT, width=.6, length=2)
    cb.outline.set_linewidth(.5)
    fig.text((xc + CB + .05) / W, (BOT - .06 * PS / 2) / Hh, "0", fontsize=FT, ha="left", va="center")
    # (c)
    xm = xc + CB + CBL + G2
    norm = LogNorm(S[ex].min(), S[ex].max())
    for row, e in enumerate(ex):
        y0 = top_c - (row + 1) * MS - row * MV
        for col in range(2):
            a = fig.add_axes([(xm + col * (MS + MG)) / W, y0 / Hh, MS / W, MS / Hh])
            im = a.imshow(S[e, col].reshape(10, 10), cmap="viridis", norm=norm, origin="lower")
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(True); sp.set_linewidth(.4)
            if row == 0:
                a.set_title(("Bayes", "Network")[col], pad=3)
            if row == 1:
                a.set_xlabel("$r_c$", labelpad=1.5)
            if col == 0:
                a.set_ylabel("$r_g$", labelpad=2, rotation=0, va="center", ha="right")
    fig.text((xm + MS + MG / 2) / W, (top_c + .165) / Hh, "Context-specific divergences", ha="center",
             va="bottom", fontsize=FS)
    xc2 = xm + 2 * MS + MG + CBG
    cax2 = fig.add_axes([xc2 / W, CBOT / Hh, CB / W, (top_c - CBOT) / Hh])
    cb2 = fig.colorbar(im, cax=cax2)
    cb2.set_label("Nats", labelpad=2)
    cb2.ax.tick_params(labelsize=FT, width=.6, length=2)
    cb2.outline.set_linewidth(.5)
    for x0, w, let in ((LM, PA, "a"), (xb, PS, "b"), (xm, 2 * MS + MG, "c")):
        fig.text((x0 + w / 2) / W, .03 / Hh, f"({let})", fontfamily="DejaVu Sans", fontsize=8, ha="center", va="baseline")
    out = fig_path("fig_fr_proxy_big_T.png" if final else "fig_fr_proxy_big.png")
    fig.savefig(out, dpi=300)
    if vector:
        for ext in ("pdf", "svg"):
            fig.savefig(out[:-3] + ext)
    print(out, f"right edge {xc2 + CB:.2f} in (+ labels) of {W}; height {Hh:.2f} in; squares {MS:.2f} in; steps drawn "
               f"{H.sum() / x.size:.3f}; examples {[int(Z['eps'][e]) for e in ex]} r {np.round(r_env[ex], 3).tolist()}; "
               f"early r ({EARLY_R}) {rE[0]:.3f} -> {rE[-1]:.3f}, SEM <= {sem.max():.3f}; log-log fit slope {slope:.3f} "
               f"intercept {icpt:.3f} (r {np.corrcoef(lx, ly)[0, 1]:.3f}, {fit_m.mean():.3f} of steps)")
    return Z, r_env


def big_figure2(vector=False, final=False, style=None):
    """John 11-18 ("I didnt realize they were different environments ... can you please try to fit all 4 environments
    in?"): the same three panels in two rows at the paper's width (6.85 in, 9 / 8.5 pt): (a) and (b) on top, (c) below
    with four environments side by side, each a Bayes | Network pair on one shared colour scale: the 7/8 and 3/8
    quantiles of the state correlation (log values) at T and over all steps (the two each version showed before);
    maps at T (final=True) or averaged over steps. -> fig_fr_proxy_big2(_T).png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    from matplotlib.colors import LogNorm
    FS, FT = 9, 8.5
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FT, "ytick.labelsize": FT,
        "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6, "xtick.minor.width": .4,
        "ytick.minor.width": .4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5, "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2, "axes.titlepad": 4,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.facecolor": "white"})
    Z = np.load(BIG)
    fr, px, cell_of = Z["fr"].mean(-1), Z["proxy"].mean(-1), Z["cell_of"]
    n_env = fr.shape[0]
    x, y = (fr[:, :, -1].ravel(), px[:, :, -1].ravel()) if final else (fr.ravel(), px.ravel())
    lo, hi = 1e-3, 1e2
    bins = np.logspace(np.log10(lo), np.log10(hi), int(round(12 * np.log10(hi / lo))) + 1)
    H, _, _ = np.histogram2d(x, y, [bins, bins])

    def state_maps(per_traj):
        S_ = np.stack([np.stack([np.bincount(cell_of, per_traj(q, e), 100) for q in (fr, px)]) for e in range(n_env)])
        return S_ / np.bincount(cell_of)
    S_T, S_A = state_maps(lambda q, e: q[e, :, -1]), state_maps(lambda q, e: q[e].mean(1))
    S = S_T if final else S_A
    pick = lambda S_: [int(np.argsort(r)[int(q * (n_env - 1))]) for r in
                       [np.array([np.corrcoef(np.log(S_[e]))[0, 1] for e in range(n_env)])] for q in (7 / 8, 3 / 8)]
    eT, eA = pick(S_T), pick(S_A)
    ex = [eT[0], eA[0], eT[1], eA[1]]
    style = style or STYLE_C
    POS = example_positions([int(Z["eps"][e]) for e in ex]) if style == "warped" else None
    E, curves = early_series(final)

    W, Hh = 6.85, 4.1                                                      # inches
    LM, PA, G1, PS, CBG, CB = .55, 2.9, .85, 1.75, .05, .07                 # (a) wider: no space right of (b)
    Y1, MS, MG, MGG, Y2 = 2.05, .62, .05, .2, .44                        # row 1 axes bottom; maps; row 2 maps bottom
    fig = plt.figure(figsize=(W, Hh))
    aa = fig.add_axes([LM / W, Y1 / Hh, PA / W, PS / Hh])
    for k, c, lab in ((EARLY_R, "#e41a1c", "FR"), ("count", "k", "obs. count")):
        mu, se = curves[k]
        aa.fill_between(E, mu - se, mu + se, color=c, alpha=.2, lw=0)
        aa.plot(E, mu, color=c, lw=1.1, label=lab)
    hi_a = max(max(mu + se) for mu, se in curves.values())
    aa.set_xscale(A_XSCALE)
    if A_XSCALE == "log":
        aa.set(xlim=(E.min() * .8, E_MAX * 1.2))
        aa.xaxis.set_major_locator(mt.LogLocator(numticks=5))
    else:
        aa.set(xlim=(0, E_MAX * 1.02))
        _lin_ticks(aa, 4)
    aa.set(ylim=(0, hi_a + .1), xlabel="Training episodes", ylabel="$R^2$")
    lg_a = aa.legend(title="Divergence vs.", loc="upper right", frameon=False, fontsize=FT, title_fontsize=FT,
                     handlelength=1.0, handletextpad=.4, borderaxespad=.15, labelspacing=.2, borderpad=.1)
    lg_a._legend_box.align = "left"
    aa.set_title(A_TITLE)
    xb = LM + PA + G1
    ab = fig.add_axes([xb / W, Y1 / Hh, PS / W, PS / Hh])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("k")
    pm = ab.pcolormesh(bins, bins, np.where(H.T > 0, H.T, .5), cmap=cmap, norm=LogNorm(1, H.max()), rasterized=True)
    ab.plot([lo, hi], [lo, hi], color="k", ls=(0, (3, 2)), lw=.8)
    ab.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ab.set_xlabel(r"$D_{KL}(B^{\mathrm{Joint}}\,\|\,B^{\mathrm{Naive}})$")
    ab.set_ylabel(r"$D_{KL}(B^{\mathrm{Trained}}\,\|\,B^{\mathrm{ESN}})$")
    ab.set_title("Divergence correspondence")
    ab.spines[["top", "right"]].set_visible(True)
    xc = xb + PS + CBG
    cax = fig.add_axes([xc / W, Y1 / Hh, CB / W, PS / Hh])
    cb = fig.colorbar(pm, cax=cax, extend="min", extendfrac=.05)
    cb.set_label("Bin count", labelpad=2)
    cb.ax.yaxis.set_major_locator(mt.LogLocator(numticks=4))
    cb.ax.tick_params(labelsize=FT, width=.6, length=2)
    cb.outline.set_linewidth(.5)
    fig.text((xc + CB + .05) / W, (Y1 - .05 * PS / 2) / Hh, "0", fontsize=FT, ha="left", va="center")
    cm, norm = _example_cmap(S[ex])
    x0 = LM
    for i, e in enumerate(ex):
        for col in range(2):
            xm = x0 + i * (2 * MS + MG + MGG) + col * (MS + MG)
            a = fig.add_axes([xm / W, Y2 / Hh, MS / W, MS / Hh])
            if style == "warped":                                          # Naive / ESN geometry at T
                _warped_grid(a, POS[("Naive", "ESN")[col]][i], S[e, col], cm, norm, MS)
            elif style == "edges":
                _edge_grid(a, S[e, col], cm, norm, MS)
            else:
                _dot_lattice(a, S[e, col], cm, norm, MS)
            a.set_title(("Bayes", "Network")[col], pad=3)
            a.set_xlabel("$r_c$", labelpad=1.5)
            if i == 0 and col == 0:
                a.set_ylabel("$r_g$", labelpad=2, rotation=0, va="center", ha="right")
    x_end = x0 + 4 * (2 * MS + MG) + 3 * MGG
    fig.text(((x0 + x_end) / 2) / W, (Y2 + MS + .3) / Hh, "Context-specific divergences", ha="center", va="bottom",
             fontsize=FS)
    cax2 = fig.add_axes([(x_end + CBG) / W, Y2 / Hh, CB / W, MS / Hh])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(cmap=cm, norm=norm), cax=cax2, extend="both", extendfrac=.05)
    cb2.set_label("Nats", labelpad=2)
    cb2.ax.tick_params(labelsize=FT, width=.6, length=2)
    cb2.outline.set_linewidth(.5)
    for xx, w, yy, let in ((LM, PA, Y1 - .5, "a"), (xb, PS, Y1 - .5, "b"), (x0, x_end - x0, .03, "c")):
        fig.text((xx + w / 2) / W, yy / Hh, f"({let})", fontfamily="DejaVu Sans", fontsize=9, ha="center",
                 va="baseline")
    out = fig_path(f"fig_fr_proxy_big2{dict(warped='w', edges='e').get(style, '')}{'_T' if final else ''}{A_TAG}.png")
    fig.savefig(out, dpi=300)
    if vector:
        for ext in ("pdf", "svg"):
            fig.savefig(out[:-3] + ext)
    print(out, "examples (top: 7/8 at T, 7/8 all steps; bottom: 3/8 at T, 3/8 all steps):",
          [int(Z["eps"][e]) for e in ex], f"right edge {x_end + CBG + CB:.2f} in")


def _dot_lattice(ax, vals, cm, norm, size_in):
    """one state map in the lattice-dot style of fig_learn_grids (John 11-18: "maybe we should use one of these styles"):
    the 100 states (r_g up, r_c across) as dots on the true lattice, joined to their neighbours, coloured by divergence"""
    from matplotlib.collections import LineCollection
    R_ = 10
    cells = np.array([(c, g) for g in range(R_) for c in range(R_)], float)              # x = r_c, y = r_g
    idx = {(g, c): g * R_ + c for g in range(R_) for c in range(R_)}
    edges = np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                      if (g + dg, c + dc) in idx])
    ax.add_collection(LineCollection(cells[edges], colors="0.7", linewidths=.3, zorder=2))
    d_pt = .72 * size_in / (R_ + .4) * 72                                          # dot diameter, points
    ax.scatter(cells[:, 0], cells[:, 1], c=vals, cmap=cm, norm=norm, s=d_pt ** 2, lw=.2, edgecolors="0.3", zorder=3)
    ax.set_xlim(-.7, R_ - .3), ax.set_ylim(-.7, R_ - .3), ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("0.7"); sp.set_linewidth(.4)


POS_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_positions.npz")


def example_positions(eps_ex, dev="cuda:0"):
    """the four example environments' state geometry at T (John 11-18: the warped-grid style of fig_manifold_fr): each
    state's posterior means (E[r_g], E[r_c]) at T, averaged over its 50 trajectories (the big sample's streams, seed
    BIG_STREAM0 + episode), for Joint and Naive Bayes (exact) and the Trained RNN and the ESN (replayed) -> cache
    {observer: (4, 100, 2)} with x = E[r_c], y = E[r_g]"""
    if os.path.exists(POS_CACHE):
        Z_ = np.load(POS_CACHE)
        if list(Z_["eps"]) == list(eps_ex):
            return {k: Z_[k] for k in ("Joint", "Naive", "Trained", "ESN")}
    import plot_fig1c_all as F
    R_ = F.record()
    out = {k: np.zeros((len(eps_ex), 100, 2)) for k in ("Joint", "Naive", "Trained", "ESN")}
    vals = np.arange(10)
    for i, EP in enumerate(eps_ex):
        L = np.array(F.episodes("ep_joint_likelihood")[EP], np.float64)
        g = int(R_["ep_goal_ind"][EP])
        gi = [g, 1 - g]
        obs, cell_of = F.cell_streams(L, g, F.SCAN_CELLS, F.SCAN_M, BIG_STREAM0 + int(EP))
        Zin = np.repeat(np.asarray(R_["ep_active_Z"][EP:EP + 1]), len(obs), 0)
        logb = {"Joint": np.log(np.maximum(F.joint_marginals(obs, L), 1e-300))[:, -1, gi],
                "Naive": log_naive(obs, L)[:, -1, gi],
                "Trained": replay_log_gpu(F, "trained", obs, Zin, dev)[:, -1, gi],
                "ESN": replay_log_gpu(F, "echo", obs, Zin, dev)[:, -1, gi]}           # (n, 2, R), goal-ordered
        for k, lb in logb.items():
            m = (np.exp(lb) * vals).sum(-1)                                      # (n, 2): E[r_g], E[r_c]
            pos = np.stack([np.bincount(cell_of, m[:, 1], 100), np.bincount(cell_of, m[:, 0], 100)], 1)
            out[k][i] = pos / np.bincount(cell_of)[:, None]
    np.savez(POS_CACHE, eps=np.array(eps_ex), **out)
    return out


def _warped_grid(ax, pos, vals, cm, norm, size_in):
    """one state map in the warped-grid style of fig_manifold_fr: the states at an observer's posterior means, joined
    to their lattice neighbours; an edge's colour = the mean of its two states' divergence; dotted: the true lattice"""
    from matplotlib.collections import LineCollection
    R_ = 10
    idx = {(g, c): g * R_ + c for g in range(R_) for c in range(R_)}
    edges = np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                      if (g + dg, c + dc) in idx])
    for v in range(R_):
        ax.plot([v, v], [0, R_ - 1], color="0.8", lw=.35, ls=(0, (1, 1.6)), zorder=0)
        ax.plot([0, R_ - 1], [v, v], color="0.8", lw=.35, ls=(0, (1, 1.6)), zorder=0)
    lc = LineCollection(pos[edges], cmap=cm, norm=norm, linewidths=max(.5, 1.5 * size_in), capstyle="round", zorder=3)
    lc.set_array((vals[edges[:, 0]] + vals[edges[:, 1]]) / 2)
    ax.add_collection(lc)
    ax.set_xlim(-.75, R_ - .25), ax.set_ylim(-.75, R_ - .25), ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("0.75"); sp.set_linewidth(.4)


def _edge_grid(ax, vals, cm, norm, size_in):
    """one state map as the edge grid of fig_manifold_fr on the TRUE lattice (John 11-18: "the grid visual of the edges
    but not the warping, it's ambiguous which agent is being shown"): each lattice edge coloured by the mean of its two
    states' divergence"""
    from matplotlib.collections import LineCollection
    R_ = 10
    cells = np.array([(c, g) for g in range(R_) for c in range(R_)], float)              # x = r_c, y = r_g
    idx = {(g, c): g * R_ + c for g in range(R_) for c in range(R_)}
    edges = np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                      if (g + dg, c + dc) in idx])
    lc = LineCollection(cells[edges], cmap=cm, norm=norm, linewidths=max(1., 3.2 * size_in), capstyle="round", zorder=3)   # thick (John 11-18)
    lc.set_array((vals[edges[:, 0]] + vals[edges[:, 1]]) / 2)
    ax.add_collection(lc)
    ax.set_xlim(-.6, R_ - .4), ax.set_ylim(-.6, R_ - .4), ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("0.75"); sp.set_linewidth(.4)


STYLE_C = "edges"                                                       # panel (c): "dots" or "warped"


def _example_cmap(S_ex):
    """the paper's FR colour map (plotting_helpers.FR_CMAP: grey, peach, red, dark red) on a log scale from the 2nd to
    the 98th percentile of the maps shown, the rest in the end triangles (John 11-18: "use this color map")"""
    from matplotlib.colors import LogNorm
    from plotting_helpers import FR_CMAP
    cm = FR_CMAP.copy()
    cm.set_under(FR_CMAP(0.)); cm.set_over(FR_CMAP(1.))
    lo_, hi_ = np.percentile(S_ex, [2, 98])
    return cm, LogNorm(lo_, hi_)


def big_figure3(vector=False, final=False, style=None):
    """John 11-18 ("see if we can fit them back into the top row if we put Bayes and Network as different sub-rows
    instead of sub-columns"): one row, (a) | (b) | (c) with the four environments as columns and Bayes / Network as
    sub-rows, the FR colour map; 7.25 in wide, 9.5 / 9 pt. John 11-18 (his pick of the layouts): (a) wider for the
    legend title "Network Divergence / corresponds with"; (b) smaller (1.05 in), no equality line; the room goes to (c)
    (maps 0.60 in) and the figure is shorter (less empty space around (c)). John 11-18, next round: (a) thin again
    (R^2 axis to 1 so the two-line legend sits above the curves), (b) back to 1.25 in, the first example (environment
    2493) dropped for more space between the panels and larger maps (three columns), colour bars without tick marks,
    "Counts" for "Bin count". Then: legend title "Network divergence / correlation with"; tops and bottoms aligned:
    (b)'s side = the two map rows of (c), (a) as tall, the three titles on one baseline, (a) and (b)'s x labels on one
    line. Then (John: "make the grids larger so they fill this empty space ... wider if needed"): (c)'s maps reach down
    to the x-label line (its r_c labels on (a) and (b)'s x-label baseline), the figure as wide as that needs; colour
    bars without tick labels (the room goes to the gaps between the panels); (b) without ticks or tick labels, then
    larger (its top on the common line, its bottom down to its x label). A_LEG overrides panel (a)'s legend placement
    (the full-training versions). Then (John): (a) as wide as (b), then wider still (1.85 in, gaps narrowed), legend
    "Network divergence correlated with" / "Factorization Regret" / "Observation count"; A_CURVES = (EARLY_R,) drops
    the count curve. B^ESN -> B^Echo (John 11-18). A_PA sets (a)'s width; A_LEG_TITLE = None: no legend (the FR-only
    version names the two divergences in its title, R^2(., .)); A_TOP: room above the titles (two-line title).
    A_LAST (John 11-18): the R^2 panel moved to the end; the letters follow the order on the page. Then: maps titled
    C_TITLE ("Representative contexts"); B_SWAP: the histogram's axes swapped (x = network divergence); A_LEG_FS: the
    legend's font size (FS = the figure's labels, so its math matches the axis labels). B_UNDER = "min" (John 11-18):
    empty bins in the colour of the lowest count (no black, no colour-bar extension); "black": the earlier look.
    B_DARK: the lowest counts darkened (viridis scaled to B_DARK of its brightness at the bottom, back to normal by 15%
    of the range; John: "have the minimum count be a darker"); then the opposite (John: "less dark"): B_LO, viridis
    from B_LO up (its darkest end cut). C_SWAP (John: "swap rows and columns again"): the examples as rows, Bayes |
    Network as columns labelled above the maps (the maps shrink to make room; the labels' tops on the common line).
    B_VMIN (John: "the 0 data region not noticeably different from the low count region"): the colour scale starts at
    B_VMIN counts; empty bins and bins below it share its lowest colour. B_NORM "linear" (John: "this is nicer ...
    normalize and rename Counts to Density"): the colour = the density of trajectory steps per log10-nat^2 (counts /
    (all steps x bin area); a linear scale from 0), axes B_LIM (10^-2.25 to 10^1.75: the bins above 1% of the peak span
    10^-2.3 to 10^1.8), B_BPD bins per decade.
    -> fig_fr_proxy_big3(_T).png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    from matplotlib.colors import LogNorm
    FS, FT = 9.5, 9                                                       # 7.2 / 6.8 pt at 5.5 in
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FT, "ytick.labelsize": FT,
        "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6, "xtick.minor.width": .4,
        "ytick.minor.width": .4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5, "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2, "axes.titlepad": 4,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.facecolor": "white"})
    Z = np.load(BIG)
    fr, px, cell_of = Z["fr"].mean(-1), Z["proxy"].mean(-1), Z["cell_of"]
    n_env = fr.shape[0]
    x, y = (fr[:, :, -1].ravel(), px[:, :, -1].ravel()) if final else (fr.ravel(), px.ravel())
    lo, hi = B_LIM
    bins = np.logspace(np.log10(lo), np.log10(hi), int(round(B_BPD * np.log10(hi / lo))) + 1)
    H, _, _ = np.histogram2d(x, y, [bins, bins])
    if B_NORM == "linear":                                                 # density per log10-nat^2 of all steps
        H = H / (x.size * np.log10(bins[1] / bins[0]) ** 2)

    def state_maps(per_traj):
        S_ = np.stack([np.stack([np.bincount(cell_of, per_traj(q, e), 100) for q in (fr, px)]) for e in range(n_env)])
        return S_ / np.bincount(cell_of)
    S_T, S_A = state_maps(lambda q, e: q[e, :, -1]), state_maps(lambda q, e: q[e].mean(1))
    S = S_T if final else S_A
    pick = lambda S_: [int(np.argsort(r)[int(q * (n_env - 1))]) for r in
                       [np.array([np.corrcoef(np.log(S_[e]))[0, 1] for e in range(n_env)])] for q in (7 / 8, 3 / 8)]
    eT, eA = pick(S_T), pick(S_A)
    ex = [e for e in (eT[0], eA[0], eT[1], eA[1]) if int(Z["eps"][e]) not in C_DROP]
    style = style or STYLE_C
    POS = example_positions([int(Z["eps"][e]) for e in ex]) if style == "warped" else None
    E, curves = early_series(final)

    LM, PA, G1, CBG, CB, CBL, G2, MG, MV, BOT, TOP, TP, XL, RC, XB = (.52, A_PA, .62, .05, .07, .18, .52, .08, .07, .60,
                                                                      A_TOP, .09, .37, .14, .22)
    PS = A_PH if BOX_ALIGN else 1.36                                       # (a)'s height; tops of (a), (b), (c) at BOT + PS
    PB = PS if BOX_ALIGN and not B_FILL else PS + XL - XB                  # (b): no tick labels, so it reaches its x label
    yb0 = BOT + PS - PB
    nC = len(ex)
    ncol, nrow = (2, nC) if C_SWAP else (nC, 2)                            # C_SWAP: examples as rows
    CL = .17 if C_SWAP and not BOX_ALIGN else 0.                           # room for the Bayes | Network labels
    G2 = .34 if C_SWAP else G2                                             # no rotated row labels then
    MS = (PS - CL + (0. if BOX_ALIGN and not B_FILL else XL - (RCL if BOX_ALIGN else RC)) - (nrow - 1) * MV) / nrow
                                                                           # BOX_ALIGN: the maps span the boxes' height;
                                                                           # B_FILL: down to the shared x-label line
    Hh = BOT + PS + TOP
    if A_LAST:                                                             # histogram | maps | R^2 curves
        xb = .42                                                           # room for (b)'s y label
        xc = xb + PB + CBG
        xm = xc + CB + CBL + (G2 if GAP_AB is None else GAP_AB)            # GAP_AB / GAP_BC: John 11-18, tighter gaps
        xa = xm + ncol * MS + (ncol - 1) * MG + CBG + CB + .22 + GAP_BC    # the Nats label, then (a)'s y axis
        W = xa + PA + .22                                                  # (a)'s last tick label
    else:
        fixed = LM + PA + G1 + CBG + CB + CBL + G2 + (ncol - 1) * MG + CBG + CB + .22
        W = fixed + PB + ncol * MS                                         # John 11-18: wider if needed
        xa = LM
        xb = LM + PA + G1
        xc = xb + PB + CBG
        xm = xc + CB + CBL + G2
    fig = plt.figure(figsize=(W, Hh))
    PSa = PS + (LH if A_TALL and BOX_ALIGN else 0.)                        # A_TALL: (c) up into the free title line
    aa = fig.add_axes([xa / W, BOT / Hh, PA / W, PSa / Hh])
    for k, c in ((k, c) for k, c in ((EARLY_R, "#e41a1c"), ("count", "k")) if k in A_CURVES):
        lab = A_LABELS[k]
        mu, se = curves[k]
        aa.fill_between(E, mu - se, mu + se, color=c, alpha=.2, lw=0)
        aa.plot(E, mu, color=c, lw=A_LW, label=lab)
    hi_a = max(max(curves[k][0] + curves[k][1]) for k in A_CURVES)
    aa.set_xscale(A_XSCALE)
    if A_XSCALE == "log":                                                  # every decade (every other if thin)
        aa.set(xlim=(E.min() * .8, E_MAX * 1.2))
        step = 1 if PA >= 1.2 else 2
        aa.xaxis.set_major_locator(mt.FixedLocator([10. ** k for k in range(0, int(np.log10(E_MAX)) + 1, step)]))
        aa.xaxis.set_minor_locator(mt.LogLocator(subs=np.arange(2, 10), numticks=20))
        aa.xaxis.set_minor_formatter(mt.NullFormatter())
    else:
        aa.set(xlim=(A_XMIN, E_MAX * 1.02))                               # A_XMIN < 0: the (0, 0) point off the axis
        _lin_ticks(aa, 3 if E_MAX == 3000 else 2)                         # 0, 1,000, 2,000 (, 3,000)
    aa.set(ylim=(0, A_YMAX), xlabel="Training episodes", ylabel="$R^2$")    # R^2's full range: room for the legend
    aa.yaxis.set_major_locator(mt.MultipleLocator(.2 if A_YMAX > .6 else .1))
    if A_LEG_TITLE is not None:
        lg_kw = dict(loc="upper left", borderaxespad=.6)
        lg_kw.update(A_LEG)
        lfs = A_LEG_FS or FT - 1.5
        lg_a = aa.legend(title=A_LEG_TITLE, frameon=False, fontsize=lfs, title_fontsize=lfs, handlelength=.9,
                         handletextpad=.3, labelspacing=.15, borderpad=.05, **lg_kw)
        lg_a._legend_box.align = "left"
    ab = fig.add_axes([xb / W, yb0 / Hh, PB / W, PB / Hh])
    from matplotlib.colors import ListedColormap
    xs_ = np.linspace(0, 1, 256)
    rgba = plt.get_cmap("viridis")(B_LO + (1 - B_LO) * xs_)                # B_LO > 0: the lowest counts lighter
    rgba[:, :3] *= (B_DARK + (1 - B_DARK) * np.clip(xs_ / .15, 0, 1))[:, None]   # the lowest counts darker
    cmap = ListedColormap(rgba)
    cmap.set_under(cmap(0) if B_UNDER == "min" else "k")
    Hb = H if B_SWAP else H.T                                              # H[FR bin, network bin]
    from matplotlib.colors import Normalize
    nrm = LogNorm(B_VMIN, H.max()) if B_NORM == "log" else Normalize(0, H.max())   # B_NORM "linear": John's comparison
    pm = ab.pcolormesh(bins, bins, np.where(Hb > 0, Hb, B_VMIN / 2) if B_NORM == "log" else Hb, cmap=cmap, norm=nrm,
                       rasterized=True)
    ab.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ab.tick_params(which="both", left=False, bottom=False, labelleft=False, labelbottom=False)  # John 11-18
    lab_fr, lab_net = (r"$D_{KL}(B^{\mathrm{Joint}}\,\|\,B^{\mathrm{Naive}})$",
                       r"$D_{KL}(B^{\mathrm{Trained}}\,\|\,B^{\mathrm{Echo}})$")
    ab.set_xlabel(lab_net if B_SWAP else lab_fr)
    ab.set_ylabel(lab_fr if B_SWAP else lab_net)
    ab.spines[["top", "right"]].set_visible(True)
    for ax_, y_ in ((aa, -XL / PSa), (ab, -XB / PB if BOX_ALIGN else (BOT - XL - yb0) / PB)):   # x-label baselines
        ax_.xaxis.set_label_coords(.5, y_)
        ax_.xaxis.label.set_va("baseline")
    cax = fig.add_axes([xc / W, yb0 / Hh, CB / W, PB / Hh])
    cb = fig.colorbar(pm, cax=cax, extend="neither" if B_UNDER == "min" else "min", extendfrac=.05)
    cb.set_label("Density" if B_NORM == "linear" else "Counts", labelpad=2)
    cb.set_ticks([])                                                       # John 11-18: no tick labels
    cb.minorticks_off()
    cb.ax.tick_params(labelsize=FT, length=0)                              # John 11-18: no tick marks
    cb.outline.set_linewidth(.5)
    cm, norm = _example_cmap(S[ex])
    top_c = BOT + PS - CL                                                  # the maps' top
    for i, e in enumerate(ex):
        for j in range(2):                                                 # j: Bayes, Network
            col, row = (j, i) if C_SWAP else (i, j)
            x0, y0 = xm + col * (MS + MG), top_c - (row + 1) * MS - row * MV
            a = fig.add_axes([x0 / W, y0 / Hh, MS / W, MS / Hh])
            if style == "warped":                                          # Naive / ESN geometry at T
                _warped_grid(a, POS[("Naive", "ESN")[j]][i], S[e, j], cm, norm, MS)
            elif style == "edges":
                _edge_grid(a, S[e, j], cm, norm, MS)
            else:
                _dot_lattice(a, S[e, j], cm, norm, MS)
            if row == nrow - 1:                                            # on (a) and (b)'s x-label baseline
                fig.text((x0 + MS / 2) / W, (BOT - (RCL if BOX_ALIGN and not B_FILL else XL)) / Hh, "$r_c$", ha="center",
                         va="baseline", fontsize=FS)
            if col == 0:
                a.set_ylabel("$r_g$", labelpad=2, rotation=0, va="center", ha="right")
            if C_SWAP and row == 0:                                        # column labels above the maps
                fig.text((x0 + MS / 2) / W, (BOT + PS + TP if BOX_ALIGN else top_c + .04) / Hh, ("Bayes", "Network")[j],
                         ha="center", va="baseline", fontsize=FS)          # BOX_ALIGN: in the title band
            elif not C_SWAP and col == 0:
                fig.text((xm - .2) / W, (y0 + MS / 2) / Hh, ("Bayes", "Network")[j], rotation=90, ha="right",
                         va="center", fontsize=FS)
    x_end = xm + ncol * MS + (ncol - 1) * MG
    titles = ((xa, PA, A_TITLE), (xb, PB, B_TITLE), (xm, x_end - xm, C_TITLE))
    if BOX_ALIGN:                                                          # titles top-aligned, lines LH apart; (b)'s
        NL = max(max(t.count("\n") + 1 for _, _, t in titles), 2 if C_SWAP else 1)   # column labels on the last line
        for x0, w, t in titles:
            for k, line in enumerate(t.split("\n")):
                fig.text((x0 + w / 2) / W, (BOT + PS + TP + (NL - 1 - k) * LH) / Hh, line, ha="center",
                         va="baseline", fontsize=FS)
    else:
        for x0, w, t in titles:
            fig.text((x0 + w / 2) / W, (BOT + PS + TP) / Hh, t, ha="center", va="baseline", fontsize=FS)  # one baseline
    yb = top_c - nrow * MS - (nrow - 1) * MV
    cax2 = fig.add_axes([(x_end + CBG) / W, yb / Hh, CB / W, (top_c - yb) / Hh])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(cmap=cm, norm=norm), cax=cax2, extend="both", extendfrac=.04)
    cb2.set_label("Nats", labelpad=2)
    cb2.set_ticks([])
    cb2.minorticks_off()
    cb2.ax.tick_params(labelsize=FT, length=0)
    cb2.outline.set_linewidth(.5)
    for let, (xx, w) in zip("abc", sorted(((xa, PA), (xb, PB), (xm, x_end - xm)))):   # letters in page order
        fig.text((xx + w / 2) / W, .03 / Hh, f"({let})", fontfamily="DejaVu Sans", fontsize=9, ha="center", va="baseline")
    out = fig_path(f"fig_fr_proxy_big3{dict(warped='w', edges='e').get(style, '')}{'_T' if final else ''}{A_TAG}.png")
    tight = {"bbox_inches": "tight", "pad_inches": .03} if SAVE_TIGHT else {}   # SAVE_TIGHT: no empty margins
    fig.savefig(out, dpi=300, **tight)
    if vector:
        for ext in ("pdf", "svg"):
            fig.savefig(out[:-3] + ext, **tight)
    print(out, f"squares {MS:.2f} in; right edge {x_end + CBG + CB:.2f} in; examples", [int(Z["eps"][e]) for e in ex])


GAP_AB_PAPER, GAP_BC_PAPER = .22, .44                                  # the tighter gaps of paper_figure


def paper_figure(vector=True):
    """John 11-18 ~19:40 ET, the version he picked ("let's go with this version"), then polished until ~21:45 ET:
    (a) the density of every step of the fully trained pair (FR on y against the network divergence on x, linear
    density colour, plain viridis, axes 10^-1.5 to 10^1.75 nats: 3.4% of steps outside), (b) two representative contexts
    as rows (environments 1687 and 4497: the 7/8 and 3/8 quantiles of the log-map correlation over all steps) x Bayes |
    Network labelled in the title band, (c) early learning to 3,000 episodes, the mean of nine runs from (0, 0), the
    legend titled by the network divergence "correlated with" (upper right), R^2 to .65, its axis up to where its
    one-line title ends; (a) and (b) down to (c)'s x-label line, margins cropped -> fig_fr_proxy_big3e_final.png (+ PDF,
    SVG with vector=True)"""
    global E_MAX, A_LAST, B_SWAP, A_PA, A_LEG, A_YMAX, A_TAG, B_LIM, GAP_AB, GAP_BC, BOX_ALIGN, B_TITLE, SAVE_TIGHT, A_TOP, \
        A_ZERO, A_XMIN, A_LW, B_FILL, A_TITLE, A_LEG_TITLE, A_TALL
    E_MAX, A_LAST, B_SWAP, A_PA, A_YMAX, A_TAG = 3000, True, True, 1.85, .65, "_final"
    B_LIM = (10 ** -1.5, 10 ** 1.75)                                       # John 11-18 ~19:50 ET: from 10^-1.5
    GAP_AB, GAP_BC = GAP_AB_PAPER, GAP_BC_PAPER                            # John: "reduce the space between panels"
    BOX_ALIGN, SAVE_TIGHT, A_TOP = True, True, .46                         # John 11-18 ~19:45 ET: boxes line up, no margins
    A_ZERO, A_XMIN = True, -20                                             # John 11-18 ~20:30 ET: the (0, 0) point; x from
                                                                           # -20 (his pick 21:30 ET; -60 tried)
    A_LW, B_FILL = .8, True                                                # John 11-18 21:30 ET: thinner curves; (a), (b)
                                                                           # grown to fill the space below them
    A_TITLE, A_TALL = "Early learning", True                               # John 21:50 ET: back to the one-line title and
    A_LEG_TITLE = r"$D_{KL}(B^{\mathrm{Trained}}\,\|\,B^{\mathrm{Echo}})$ correlated with"   # the legend title (the 21:40
    A_LEG = dict(loc="upper right", borderaxespad=.1)                      # two-line title dropped), R^2 to .65; (c) taller,
                                                                           # up to where its title ends
    B_TITLE = "Correspondence between\nnetwork and Bayes divergences"     # John's title, two lines (order swapped 11-18)
    big_figure3(vector, False, "edges")


def full_training_figure(final=False):
    """John 11-18 ("a version of the R^2 with the full training if we have it for any"): panel (a)'s two curves over
    the whole training of the original run (the replicates stop at 2,005 episodes): every replayed checkpoint, 1 to
    49,995 episodes (fig_fr_proxy_training.build_full and the earlier caches), 50 environments x 5,000 trajectories,
    R^2 per environment, step and variable, averaged; band = +-1 SEM over environments; linear and log x; dotted =
    2,000 episodes (panel (a)'s range). -> fig_fr_proxy_full(_T).png"""
    global E_MAX
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    FS, FT = 9.5, 9
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FT, "ytick.labelsize": FT,
        "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6, "xtick.minor.width": .4,
        "ytick.minor.width": .4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    e_max, E_MAX = E_MAX, 49995
    try:
        E, curves = early_series(final, "one")
    finally:
        E_MAX = e_max
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.3), constrained_layout=True)
    for ax, xs in zip(axs, ("linear", "log")):
        for k, c, lab in ((EARLY_R, "#e41a1c", "FR"), ("count", "k", "obs. count")):
            mu, se = curves[k]
            ax.fill_between(E, mu - se, mu + se, color=c, alpha=.2, lw=0)
            ax.plot(E, mu, color=c, lw=1.1, label=lab)
        ax.axvline(2000, color=".5", ls=":", lw=.8)
        ax.set(xscale=xs, ylim=(0, 1), xlabel="Training episodes", title="Full training" + (", log scale" * (xs == "log")))
        ax.yaxis.set_major_locator(mt.MultipleLocator(.2))
    axs[0].set(xlim=(0, 51000), ylabel="$R^2$")
    axs[0].set_xticks([0, 25000, 50000], ["0", "25,000", "50,000"])
    axs[1].set_xlim(.8, 6e4)
    lg = axs[1].legend(title="Network divergence\ncorrelation with", loc="upper left", frameon=False, fontsize=FT - 1,
                       title_fontsize=FT - 1, handlelength=.9, handletextpad=.3, labelspacing=.15)
    lg._legend_box.align = "left"
    out = fig_path(f"fig_fr_proxy_full{'_T' if final else ''}.png")
    fig.savefig(out, dpi=300)
    print(out, len(E), "checkpoints")


def big_stats(Z):
    """within-step rank correlations on the big sample (per environment, to keep memory low) -> printout"""
    rs = {k: [] for k in KEYS[1:]}
    for e in range(len(Z["eps"])):
        f = Z["fr"][e:e + 1].astype(np.float64)
        for k in KEYS[1:]:
            rs[k].append(spearman_within(Z[k][e:e + 1].astype(np.float64), f).mean(-1)[0])
    for k in KEYS[1:]:
        r = np.array(rs[k])
        print(f"   {k:14s} within-step r at t = 1, 5, 10, 20, 30:", " ".join(f"{r.mean(0)[i]:.3f}" for i in (0, 4, 9, 19, 29)),
              f"| all steps {r.mean():.3f}")
    fr, px = Z["fr"].mean(-1)[..., -1], Z["proxy"].mean(-1)[..., -1]
    rat = px / fr
    print("   divergence / FR at T: median %.2f, quartiles %.2f-%.2f" % (np.median(rat), *np.percentile(rat, [25, 75])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--big", action="store_true", help="the 400-environment figure (fig_fr_proxy_big.png)")
    ap.add_argument("--vector", action="store_true", help="with --big: also PDF + SVG")
    ap.add_argument("--final", action="store_true", help="with --big: every panel at the final step T only")
    ap.add_argument("--paper", action="store_true", help="the version John picked 11-18 (paper_figure)")
    args = ap.parse_args()
    if args.paper:
        return paper_figure()
    if args.big:
        if args.rebuild or not os.path.exists(BIG):
            build_big()
        Z, _ = big_figure(args.vector, args.final)
        return big_stats(Z)
    if args.rebuild or not os.path.exists(CACHE):
        build()
    Z = np.load(CACHE)
    D = {k: Z[k].astype(np.float64) for k in KEYS}
    envs, cell_of = Z["envs"], Z["cell_of"]
    n_env, n_tr, n_t = D["fr"].shape[:3]
    r = {k: spearman_within(D[k], D["fr"]).mean(-1) for k in KEYS[1:]}     # (env, T), mean of the two variables
    both = {k: D[k].mean(-1) for k in KEYS}                                # (env, trajectory, T)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    apply_fig_style()
    plt.rcParams.update({"axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})
    T_, A = 12.5, 11
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 9.6), constrained_layout=True)

    # top: one environment's states at T (the environment with the median correlation over steps)
    e_mid = int(np.argsort(r["proxy"].mean(1))[n_env // 2])
    cells = np.zeros((2, 100))
    for j, k in enumerate(("fr", "proxy")):
        cells[j] = np.bincount(cell_of, both[k][e_mid, :, -1], 100) / np.bincount(cell_of, minlength=100)
    norm = LogNorm(cells.min(), cells.max())
    r_cells = np.corrcoef(np.log(cells))[0, 1]
    for j, (title) in enumerate((r"Factorization Regret, $D_{KL}(B^{Joint}\,\|\,B^{Naive})$",
                                  r"Trained vs ESN, $D_{KL}(B^{Trained}\,\|\,B^{ESN})$")):
        im = ax[0, j].imshow(cells[j].reshape(10, 10), cmap="viridis", norm=norm, origin="lower")
        ax[0, j].set(xlabel="contextual variable's value", ylabel="goal variable's value", xticks=[0, 9], yticks=[0, 9])
        ax[0, j].set_title(title, fontsize=T_)
    cb = fig.colorbar(im, ax=ax[0, :], shrink=.85, pad=.02)
    cb.set_label("nats at T = 30 (mean of 50 trajectories per state)", fontsize=A)
    ax[0, 0].text(0, -.24, f"one environment (median correlation of the 39); its 100 states: r = {r_cells:.2f} (log)",
                  transform=ax[0, 0].transAxes, fontsize=A, ha="left", va="top")

    # bottom left: every trajectory step of every environment
    x, y = both["fr"].ravel(), both["proxy"].ravel()
    hi = 10 ** np.ceil(np.log10(max(x.max(), y.max())))
    bins = np.logspace(-4, np.log10(hi), int(10 * (np.log10(hi) + 4)) + 1)
    H, _, _ = np.histogram2d(np.clip(x, 1e-4, hi * .99), np.clip(y, 1e-4, hi * .99), [bins, bins])
    a = ax[1, 0]
    a.pcolormesh(bins, bins, H.T, cmap="viridis", norm=LogNorm(1, H.max()), rasterized=True)
    idx = np.digitize(x, bins)
    med = np.array([np.median(y[idx == i]) if (idx == i).sum() > 200 else np.nan for i in range(1, len(bins))])
    mid = np.sqrt(bins[1:] * bins[:-1])
    a.plot(mid, med, color="#e41a1c", lw=2.2, label="median divergence")
    a.plot(bins, bins, color="k", ls="--", lw=1.2, label="equal")
    a.set(xscale="log", yscale="log", xlim=(1e-4, hi), ylim=(1e-4, hi),
          xlabel="Factorization Regret (nats)", ylabel="Trained vs ESN divergence (nats)")
    a.set_title(f"Every trajectory step ({n_env} x {n_tr:,} x {n_t}); colour = count", fontsize=T_)
    a.legend(frameon=False, fontsize=A, loc="upper left")

    # bottom right: rank correlation with FR at every step, and the two controls
    a = ax[1, 1]
    t = np.arange(1, n_t + 1)
    for k, c, lab in (("trained_naive", C_TN, r"Trained vs exact Naive"), ("proxy", C_PROXY, "Trained vs ESN"),
                      ("joint_esn", C_JE, r"exact Joint vs ESN")):
        lo, hi = np.percentile(r[k], [25, 75], axis=0)
        a.fill_between(t, lo, hi, color=c, alpha=.15, lw=0)
        a.plot(t, r[k].mean(0), color=c, lw=2.2, label=lab)
    a.set(xlim=(1, n_t), ylim=(0, 1), xlabel="step t", ylabel="rank correlation with FR (Spearman r)")
    a.set_title("Within an environment and step, across trajectories", fontsize=T_)
    a.legend(frameon=False, fontsize=A, loc="lower left")
    for a in ax.ravel():
        a.spines[["top", "right"]].set_visible(False)
    out = fig_path("fig_fr_proxy.png")
    fig.savefig(out, dpi=150)

    from scipy.stats import spearmanr
    rs = lambda k: spearmanr(both["fr"].ravel()[::7], both[k].ravel()[::7])[0]
    print(out)
    print("within-step r (mean over environments) at t = 1, 5, 10, 20, 30:")
    for k in KEYS[1:]:
        print(f"   {k:14s}", " ".join(f"{r[k].mean(0)[i]:.3f}" for i in (0, 4, 9, 19, 29)),
              f"| all steps {r[k].mean():.3f}; pooled (every 7th point) {rs(k):.3f}")
    print(f"median environment e = {envs[e_mid]} (row {e_mid}); its states' log r at T = {r_cells:.3f}")
    if "proxy_f16" in Z.files:
        r16 = spearman_within(Z["proxy_f16"].astype(np.float64), D["fr"]).mean(-1)
        print("the stored float16 beliefs instead (ESN zeros at 2^-25): within-step r over all steps %.3f (exact %.3f)"
              % (r16.mean(), r["proxy"].mean()))
    rat = both["proxy"][..., -1] / both["fr"][..., -1]
    print("proxy / FR at T: median %.2f, quartiles %.2f-%.2f; per-environment medians %.2f-%.2f" %
          (np.median(rat), *np.percentile(rat, [25, 75]), *np.percentile(np.median(rat, 1), [0, 100])))
    st = np.stack([np.bincount(cell_of, both[k][i, :, -1], 100) for i in range(n_env) for k in ("fr", "proxy")])
    st = st.reshape(n_env, 2, 100)
    print("states at T (all environments, log r): %.3f; between-environment means r: %.3f" %
          (np.corrcoef(np.log(st[:, 0].ravel()), np.log(st[:, 1].ravel()))[0, 1],
           np.corrcoef(both["fr"][..., -1].mean(1), both["proxy"][..., -1].mean(1))[0, 1]))


if __name__ == "__main__":
    main()
