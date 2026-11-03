"""Where the unfolding comes from: a hybrid Bayes observer that uses the joint evidence only at a few steps
(25 Oct 2026) -> main/REPLICATION/figure_bin/fig1c_hybrid_unfold.png

The hybrid observer keeps a joint belief over (r_g, r_c). At the selected steps it multiplies in the step's joint
likelihood (as the Optimal does), at all other steps the Naive's factorized likelihood (the table averaged over the
other variable, fig1.alias_sim). All 30 steps joint is the Optimal, none is the Naive (both reproduce the scan's
beliefs to float16 rounding, 2.4e-4). Steps selected per stream, k of 30:
    most entangled    the stream's k steps of highest entanglement (house rev, as the scan stores it)
    random            k random steps
    timed             the k most entangled steps of another stream of the same cell and environment (same timing,
                      not this stream's entanglement)
    least entangled   the stream's k steps of lowest entanglement
Grid: each cell's mean posterior means (E[r_g], E[r_c]) at T over its 50 streams, averaged over the 39 environments
of the alias scan, joined to the grid neighbours; dashed: the true lattice (r_g, r_c). Unfolded share: (d_Naive -
d) / (d_Naive - d_Optimal), d = mean distance of the cells from their true positions (grid units).
Cache: main/DATA/experimental/fig1c_compression/figdata/round6_hidden/hybrid.npz (build: about 9 min CPU).
Time control of the sweep (28 Oct 2026; John: "Most entangled steps are happening later. Can we make sure that both
columns are sampling timesteps equally?"): entanglement is ~0 at step 1 by construction (from the flat prior the
joint's first marginal update is the Naive's) and flat from step 3 (median 0.16 - 0.20 nats), so over all stream-steps
the least-entangled selection over-samples steps 1 - 5 (28% of its steps at 10% vs 9% for the most entangled; mean step
13.3 vs 17.1). build_sweep100(timed=True) -> hybrid_sweep100_timed.npz picks, at EVERY step, the p% least / most
entangled streams of that step (both columns sample every step equally). Unfolded at 10 / 26 / 50% of steps: least
-1 / 4 / 22% -> -2 / 3 / 20%, most 50 / 81 / 94% -> 50 / 81 / 94% (100 environments): the effect is not timing (the
final belief sums the selected steps' evidence, order-free; only the steps' labels changed). Movie:
movie_sweep_combined(cache=CACHE_SWEEP100_TIMED, out="fig1c_movie_hybrid_sweep_combined_timed.gif", text=dict(clock=
"joint evidence at {p:.0f}% of the streams at every step ({n_env} envs x 50 streams per cell)", head_left="least
entangled, every step", head_right="most entangled, every step"); its thresholds are the median of the steps' cuts.
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_hybrid_unfold.py [build]"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402
from matplotlib.colors import LogNorm, to_rgb                                             # noqa: E402

SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
CACHE = os.path.join(SCAN, "figdata", "round6_hidden", "hybrid.npz")
CACHE_ENV = os.path.join(SCAN, "figdata", "round6_hidden", "hybrid_env.npz")      # steps chosen over the environment
CACHE_SWEEP = os.path.join(SCAN, "figdata", "round6_hidden", "hybrid_sweep.npz")  # threshold sweep, beliefs at T only
SWEEP = np.arange(0, 101, 2)                                                      # % of the environment's stream-steps
ENVS = [e for e in range(40) if e != 2]
E, K, M, T, R = 39, 100, 50, 30, 10
KS = (1, 3, 5, 10)                                                                 # per stream: k of 30 steps
PCTS = (5, 25, 50, 75)                                                             # per environment: % of stream-steps
SEL = {"bottom": ("least entangled steps", "#74a9cf"), "random": ("random steps", "0.45"),
       "timed": ("another stream's most entangled steps", "#b8b8b8"), "top": ("most entangled steps", "#d7301f")}


def _terms(L, o):
    """per stream and step: the joint log likelihood (n, T, R, R) and the Naive's factorized one, slot order"""
    lJ = np.einsum("nti,ijk->ntjk", o, np.log(L)) + np.einsum("nti,ijk->ntjk", 1 - o, np.log1p(-L))
    Lb = (L.mean(2), L.mean(1))
    lp = [o @ np.log(Lb[k]) + (1 - o) @ np.log1p(-Lb[k]) for k in (0, 1)]
    return lJ, lp[0][..., :, None] + lp[1][..., None, :]


def _run(lJ, lN, mask, g):
    """hybrid beliefs, goal-ordered marginals (n, T, 20); mask (n, T): True = joint step"""
    cum = np.cumsum(np.where(mask[..., None, None], lJ, lN), 1)
    post = np.exp(cum - cum.max((-1, -2), keepdims=True))
    post /= post.sum((-1, -2), keepdims=True)
    m = np.stack([post.sum(-1), post.sum(-2)], 2)
    return np.concatenate([m[:, :, g], m[:, :, 1 - g]], -1)


def build(seed=0):
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    cells, goal = meta["cells"], meta["goal"].astype(int)
    SJ = np.load(os.path.join(SCAN, "scan_Joint.npz"))["beliefs"]
    SN = np.load(os.path.join(SCAN, "scan_Naive.npz"))["beliefs"]
    rng = np.random.default_rng(seed)
    tg = cells[np.repeat(np.arange(K), M)]
    res = {}
    for ie, e in enumerate(ENVS):
        lJ, lN = _terms(meta["L"][e].astype(np.float64), meta["obs"][e].astype(np.float64))
        n = len(lJ)
        order = np.argsort(-meta["rev"][e].astype(np.float64), 1)
        part = np.empty(n, int)
        for k in range(K):
            idx = np.arange(k * M, (k + 1) * M)
            p = rng.permutation(M)
            while (p == np.arange(M)).any():
                p = rng.permutation(M)
            part[idx] = idx[p]
        conds = {"optimal": np.ones((n, T), bool), "naive": np.zeros((n, T), bool)}
        for kk in KS:
            top = np.zeros((n, T), bool); np.put_along_axis(top, order[:, :kk], True, 1)
            bot = np.zeros((n, T), bool); np.put_along_axis(bot, order[:, -kk:], True, 1)
            rnd = np.zeros((n, T), bool); np.put_along_axis(rnd, np.argsort(rng.random((n, T)), 1)[:, :kk], True, 1)
            conds[f"top{kk}"], conds[f"bottom{kk}"], conds[f"random{kk}"], conds[f"timed{kk}"] = top, bot, rnd, top[part]
        for c, mask in conds.items():
            b = _run(lJ, lN, mask, goal[e])
            if c in ("optimal", "naive"):
                ref = (SJ if c == "optimal" else SN)[e].astype(np.float64)
                assert np.abs(b - ref).max() < 1e-3, (c, e)
            pt = b[np.arange(n), -1, tg[:, 0]] * b[np.arange(n), -1, R + tg[:, 1]]
            d = res.setdefault(c, {"cm": np.zeros((E, K, T, 20), np.float32), "p": np.zeros(E), "frac_steps": np.zeros(E)})
            d["cm"][ie], d["p"][ie], d["frac_steps"][ie] = b.reshape(K, M, T, 20).mean(1), pt.mean(), mask.mean()
        print(f"env {e} ({ie + 1}/39)", flush=True)
    np.savez(CACHE, conds=np.array(list(res)), **{f"{c}__{k}": v for c, d in res.items() for k, v in d.items()})


def build_env(seed=0):
    """the same observer with the joint steps chosen over the whole environment: of all its 5,000 x 30 stream-steps,
    the pct% most entangled (top), least entangled (bottom), a random pct% (random), or per cell as many as the top
    selection gives that cell but at random stream-steps of the cell (cell: the top selection's allocation over cells
    without its choice of streams and steps) -> CACHE_ENV"""
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    cells, goal = meta["cells"], meta["goal"].astype(int)
    rng = np.random.default_rng(seed)
    tg = cells[np.repeat(np.arange(K), M)]
    res = {}
    for ie, e in enumerate(ENVS):
        lJ, lN = _terms(meta["L"][e].astype(np.float64), meta["obs"][e].astype(np.float64))
        n = len(lJ)
        order = np.argsort(-meta["rev"][e].astype(np.float64).ravel())
        conds = {"optimal": np.ones((n, T), bool), "naive": np.zeros((n, T), bool)}
        for kk in PCTS:
            nsel = int(round(n * T * kk / 100))
            top = np.zeros(n * T, bool); top[order[:nsel]] = True; top = top.reshape(n, T)
            bot = np.zeros(n * T, bool); bot[order[-nsel:]] = True; bot = bot.reshape(n, T)
            rnd = np.zeros(n * T, bool); rnd[rng.permutation(n * T)[:nsel]] = True; rnd = rnd.reshape(n, T)
            cel = np.zeros((n, T), bool)
            for k in range(K):
                rows = slice(k * M, (k + 1) * M)
                blk = np.zeros(M * T, bool); blk[rng.permutation(M * T)[:top[rows].sum()]] = True
                cel[rows] = blk.reshape(M, T)
            conds[f"top{kk}"], conds[f"bottom{kk}"], conds[f"random{kk}"], conds[f"cell{kk}"] = top, bot, rnd, cel
        for c, mask in conds.items():
            b = _run(lJ, lN, mask, goal[e])
            pt = b[np.arange(n), -1, tg[:, 0]] * b[np.arange(n), -1, R + tg[:, 1]]
            d = res.setdefault(c, {"cm": np.zeros((E, K, T, 20), np.float32), "p": np.zeros(E),
                                   "frac_steps": np.zeros(E), "cell_frac": np.zeros((E, K))})
            d["cm"][ie], d["p"][ie], d["frac_steps"][ie] = b.reshape(K, M, T, 20).mean(1), pt.mean(), mask.mean()
            d["cell_frac"][ie] = mask.reshape(K, M * T).mean(1)
        print(f"env {e} ({ie + 1}/39)", flush=True)
    np.savez(CACHE_ENV, conds=np.array(list(res)), **{f"{c}__{k}": v for c, d in res.items() for k, v in d.items()})


def build_sweep():
    """beliefs at T of the hybrid observer with the joint evidence at the p% least / most entangled of all the
    environment's stream-steps, p = SWEEP (the sets are nested, so each level adds the next steps' joint - factorized
    log-likelihood difference to every stream's running sum) -> CACHE_SWEEP: cm (2, levels, E, K, 20) cell means of the
    goal-ordered beliefs (0 = least, 1 = most entangled), thr (2, levels, E) the entanglement at the selection's edge"""
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    goal = meta["goal"].astype(int)
    SJ = np.load(os.path.join(SCAN, "scan_Joint.npz"))["beliefs"]
    SN = np.load(os.path.join(SCAN, "scan_Naive.npz"))["beliefs"]
    cm = np.zeros((2, len(SWEEP), E, K, 20), np.float32)
    thr = np.full((2, len(SWEEP), E), np.nan)
    for ie, e in enumerate(ENVS):
        lJ, lN = _terms(meta["L"][e].astype(np.float64), meta["obs"][e].astype(np.float64))
        n = len(lJ)
        Dflat = (lJ - lN).reshape(n * T, R, R)
        base = lN.sum(1)
        rev = meta["rev"][e].astype(np.float64).ravel()
        desc = np.argsort(-rev)
        g = goal[e]
        for w, order in enumerate((desc[::-1], desc)):                               # least, most entangled first
            Ssum, done = base.copy(), 0
            for j, pct in enumerate(SWEEP):
                nsel = int(round(n * T * pct / 100))
                new = order[done:nsel]
                np.add.at(Ssum, new // T, Dflat[new])
                done = nsel
                if nsel:
                    thr[w, j, ie] = rev[order[nsel - 1]]
                post = np.exp(Ssum - Ssum.max((-1, -2), keepdims=True))
                post /= post.sum((-1, -2), keepdims=True)
                m = np.stack([post.sum(-1), post.sum(-2)], 1)                       # (n, 2, R) slot order
                b = np.concatenate([m[:, g], m[:, 1 - g]], -1)
                if pct in (0, 100):
                    ref = (SN if pct == 0 else SJ)[e][:, -1].astype(np.float64)
                    assert np.abs(b - ref).max() < 1e-3, (e, w, pct)
                cm[w, j, ie] = b.reshape(K, M, 20).mean(1)
        print(f"env {e} ({ie + 1}/39)", flush=True)
    np.savez(CACHE_SWEEP, cm=cm, thr=thr, sweep=SWEEP)


CACHE_SWEEP100 = os.path.join(SCAN, "figdata", "round6_hidden", "hybrid_sweep100.npz")


def _environments100(n_env=100, seed=2026):
    """harvest episodes and observation seeds: the alias scan's list of random environment orientations
    (fig1/esn_scan.npz eps, 80; seed 1000 + its index, so its first 40 are the scan's environments), without
    same-variable environments and repeated variable pairs, then random harvest episodes with new pairs (seed 1080 +
    j) up to n_env"""
    import fig1 as cf
    ga = cf.ga
    f = np.load(ga.CACHE.format("trained"), mmap_mode="r")
    pool = np.load(os.path.join(ROOT, "main", "DATA", "experimental", "fig1", "esn_scan.npz"))["eps"]
    ci = np.sort(np.asarray(f["ep_ctx_inds"]), 1)
    eps, seeds, seen = [], [], set()
    for i, ep in enumerate(pool):
        pair = tuple(ci[ep])
        if pair[0] != pair[1] and pair not in seen:
            eps.append(int(ep)), seeds.append(1000 + i), seen.add(pair)
    rng = np.random.default_rng(seed)
    j = 0
    for ep in rng.permutation(len(ci)):
        if len(eps) >= n_env:
            break
        pair = tuple(ci[ep])
        if pair[0] != pair[1] and pair not in seen:
            eps.append(int(ep)), seeds.append(1080 + j), seen.add(pair)
            j += 1
    return np.array(eps), np.array(seeds)


CACHE_SWEEP100_TIMED = os.path.join(SCAN, "figdata", "round6_hidden", "hybrid_sweep100_timed.npz")


def _sweep_env(job):
    """one environment of build_sweep100: job = (EP, seed, timed) -> cm (2, levels, K, 20), thr (2, levels), fr (K,).
    timed False: the p% least / most entangled of all the environment's stream-steps; timed True: at EVERY step the p%
    least / most entangled streams at that step (John 10-28: "Most entangled steps are happening later ... make sure
    that both columns are sampling timesteps equally"), thr then the median over steps of each step's cut"""
    import fig1 as cf
    ga = cf.ga
    EP, sd, timed = job
    f = np.load(ga.CACHE.format("trained"), mmap_mode="r")
    cells = [(g, c) for g in range(R) for c in range(R)]
    cm = np.zeros((2, len(SWEEP), K, 20), np.float32)
    thr = np.full((2, len(SWEEP)), np.nan)
    g = int(f["ep_goal_ind"][EP])
    L = np.asarray(f["ep_joint_likelihood"][EP], np.float64)
    cell_of = np.repeat(np.arange(K), M)
    r01 = [(rc, rg) if g == 1 else (rg, rc) for rg, rc in cells]
    p_on = np.stack([L[:, a, b] for a, b in r01])[cell_of][:, None]
    obs = np.random.default_rng(int(sd)).random((K * M, T, len(L))) < p_on
    o = obs.astype(np.float64)
    lJ, lN = _terms(L, o)
    n = len(lJ)
    cum = np.cumsum(lJ, 1)
    post = np.exp(cum - cum.max((-1, -2), keepdims=True))
    post /= post.sum((-1, -2), keepdims=True)
    joint = np.stack([post.sum(-1), post.sum(-2)], 2)                              # (n, T, 2, R) slot order
    del cum, post
    Lb = (L.mean(2), L.mean(1))
    px = np.exp(np.stack([o @ np.log(Lb[k]) + (1 - o) @ np.log1p(-Lb[k]) for k in (0, 1)], 2))
    naive = np.cumprod(px, 1)
    naive /= naive.sum(-1, keepdims=True)
    rev = ga.sp.step_dkl(px / px.sum(-1, keepdims=True),
                         np.stack([ga.sp.approx_lik(joint[:, :, k]) for k in (0, 1)], 2)).mean(-1)   # (n, T)
    x, y = np.clip(joint[:, -1, g], 1e-15, 1 - 1e-15), np.clip(naive[:, -1, g], 1e-15, 1 - 1e-15)
    frc = (.5 * ((x * np.log(x / y)).sum(-1) + (y * np.log(y / x)).sum(-1))).reshape(K, M).mean(1)
    Dflat = (lJ - lN).reshape(n * T, R, R)
    base = lN.sum(1)
    flat = rev.ravel()
    desc = np.argsort(-flat)
    per_step = np.argsort(-rev, 0)                                                 # (n, T): each step's streams
    for w in (0, 1):                                                               # least, most entangled first
        order = desc[::-1] if w == 0 else desc
        Ssum, done = base.copy(), 0
        for j, pct in enumerate(SWEEP):
            if timed:
                k_cur = int(round(n * pct / 100))
                rank = per_step[::-1] if w == 0 else per_step
                new = (rank[done:k_cur] * T + np.arange(T)).ravel()
                done = k_cur
                if k_cur:
                    thr[w, j] = np.median(rev[rank[k_cur - 1], np.arange(T)])
            else:
                nsel = int(round(n * T * pct / 100))
                new = order[done:nsel]
                done = nsel
                if nsel:
                    thr[w, j] = flat[order[nsel - 1]]
            np.add.at(Ssum, new // T, Dflat[new])
            q = np.exp(Ssum - Ssum.max((-1, -2), keepdims=True))
            q /= q.sum((-1, -2), keepdims=True)
            m = np.stack([q.sum(-1), q.sum(-2)], 1)
            b = np.concatenate([m[:, g], m[:, 1 - g]], -1)
            if pct == 100:
                assert np.abs(b - np.concatenate([joint[:, -1, g], joint[:, -1, 1 - g]], -1)).max() < 1e-9
            cm[w, j] = b.reshape(K, M, 20).mean(1)
    return cm, thr, frc


def build_sweep100(n_env=100, timed=False, workers=1):
    """the sweep of build_sweep on n_env environments (_environments100): each environment's 100 cells x 50 streams
    simulated as alias_sim does (its observations, house entanglement rev and FR reproduce the scan's exactly for the
    scan's environments: checked), the hybrid observer at the p% least / most entangled stream-steps (timed: at every
    step the p% least / most entangled streams) -> CACHE_SWEEP100 (CACHE_SWEEP100_TIMED): cm (2, levels, E, K, 20),
    thr (2, levels, E), fr (E, K) the cells' goal-variable FR at T, eps, seeds"""
    eps, seeds = _environments100(n_env)
    jobs = [(int(EP), int(sd), timed) for EP, sd in zip(eps, seeds)]
    if workers > 1:
        from multiprocessing import get_context
        with get_context("fork").Pool(workers) as pool:
            res = []
            for ie, r_ in enumerate(pool.imap(_sweep_env, jobs)):
                res.append(r_)
                print(f"environment {ie + 1}/{len(eps)}", flush=True)
    else:
        res = []
        for ie, job in enumerate(jobs):
            res.append(_sweep_env(job))
            print(f"environment {ie + 1}/{len(eps)} (episode {job[0]})", flush=True)
    cm = np.stack([r_[0] for r_ in res], 2)
    thr = np.stack([r_[1] for r_ in res], 2)
    frc = np.stack([r_[2] for r_ in res])
    np.savez(CACHE_SWEEP100_TIMED if timed else CACHE_SWEEP100, cm=cm, thr=thr, fr=frc, eps=eps, seeds=seeds,
             sweep=SWEEP)


def movie_sweep():
    """two average grids at T side by side, the share of joint stream-steps sweeping 0 -> 100%: left, the joint evidence
    where entanglement is below a threshold (the least entangled steps first); right, where it is above one (the most
    entangled first) -> figure_bin/fig1c_movie_hybrid_sweep.gif"""
    from PIL import Image
    from main.utils import fig_path
    import plot_fig1c_mesh as pm
    Z = np.load(CACHE_SWEEP)
    cm, thr, sw = Z["cm"], Z["thr"], Z["sweep"].astype(float)
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    cells = meta["cells"]
    fr = np.maximum(meta["FR"][ENVS].astype(np.float64).reshape(E, K, M, T).mean(2)[:, :, -1].mean(0), 1e-6)
    idx = {(int(g), int(c)): k for k, (g, c) in enumerate(cells)}
    edges = np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                      if (g + dg, c + dc) in idx])
    Rv = np.arange(R)
    P = np.stack([(cm[..., :R] * Rv).sum(-1), (cm[..., R:] * Rv).sum(-1)], -1).mean(2)   # (2, levels, K, 2)
    d = np.linalg.norm(P - cells, axis=-1).mean(-1)                                  # (2, levels)
    share = 100 * (d[:, :1] - d) / (d[:, :1] - d[:, -1:])
    th = np.nanmedian(thr, -1)                                                       # (2, levels) nats
    norm = LogNorm(.05, 30)
    ecol = np.sqrt(fr[edges[:, 0]] * fr[edges[:, 1]])
    pw, gap = 3.2, .5
    W, H = .3 + 2 * pw + gap + .35 + .14 + 1.0, .95 + pw + .45 + 1.55 + .8
    fig = plt.figure(figsize=(W, H), facecolor="white", dpi=80)
    COL = ("#3f88c5", "#d7301f")
    TTL = ("least entangled steps first\njoint where entanglement < %.2f nats",
           "most entangled steps first\njoint where entanglement > %.2f nats")
    art = []
    for w in range(2):
        ax = fig.add_axes([(.3 + w * (pw + gap)) / W, (H - .95 - pw) / H, pw / W, pw / H])
        ax.add_collection(LineCollection(cells[edges].astype(float), colors="0.82", linewidths=.5, linestyles=(0, (3, 2))))
        lc = LineCollection(P[w, 0][edges], cmap=pm.FR_CMAP, norm=norm, linewidths=1.5)
        lc.set_array(ecol)
        ax.add_collection(lc)
        ax.set_xlim(-.7, 9.7), ax.set_ylim(-.7, 9.7), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(COL[w]), sp.set_linewidth(1.4)
        ttl = ax.set_title("", fontsize=10, color=COL[w], pad=5, linespacing=1.25)
        pct = ax.text(.97, .03, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color=COL[w])
        art.append((w, lc, ttl, pct))
    cbx = fig.add_axes([(.3 + 2 * pw + gap + .35) / W, (H - .95 - pw) / H, .14 / W, pw / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=pm.FR_CMAP, norm=norm), cax=cbx, extend="both")
    cb.set_label("edge colour: factorization regret\nof the edge's two cells at T (nats)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_visible(False)
    cax = fig.add_axes([.3 / W + .08, .95 / H, (2 * pw + gap) / W - .16, 1.3 / H])
    for w in range(2):
        cax.plot(sw, share[w], c=COL[w], lw=1.2, alpha=.35)
    dots = [cax.plot([], [], "o", c=COL[w], ms=7)[0] for w in range(2)]
    trace = [cax.plot([], [], c=COL[w], lw=2.2)[0] for w in range(2)]
    cax.plot([0, 100], [0, 100], c="0.85", lw=.8, zorder=0)
    cax.set_xlim(-1, 101), cax.set_ylim(-4, 104)
    cax.set_xlabel("joint stream-steps (% of the environment's)", fontsize=9)
    cax.set_ylabel("unfolded share (%)", fontsize=9)
    cax.tick_params(labelsize=8)
    cax.spines[["top", "right"]].set_visible(False)
    clock = fig.text(.3 / W, (H - .22) / H, "", fontsize=13, va="center")
    import textwrap
    fig.text(.3 / W, .28 / H, textwrap.fill("Hybrid Bayes observer, alias scan (39 environments x 100 cells x 50 streams): "
             "joint likelihood at the selected stream-steps, the Naive's factorized one elsewhere; grid: the cells' mean "
             "estimates (E[r_g], E[r_c]) at T, averaged over environments; dashed: true lattice; colour: FR at T. Thresholds: "
             "median over environments.", 150), fontsize=6.5, color="0.35", va="center", linespacing=1.3)

    def update(x):
        j = min(int(np.floor(x)), len(sw) - 2)
        f = x - j
        p_now = (1 - f) * sw[j] + f * sw[j + 1]
        for w, lc, ttl, pct in art:
            pos = (1 - f) * P[w, j] + f * P[w, j + 1]
            lc.set_segments(pos[edges])
            t_now = (1 - f) * th[w, j] + f * th[w, j + 1] if np.isfinite(th[w, j]) else th[w, j + 1]
            ttl.set_text((TTL[w] % t_now) if p_now > 0 else TTL[w].split("\n")[0] + "\nno joint evidence yet (the Naive)")
            sh = (1 - f) * share[w, j] + f * share[w, j + 1]
            pct.set_text("unfolded %.0f%%" % sh)
            k = int(np.searchsorted(sw, p_now, side="right"))
            trace[w].set_data(np.r_[sw[:k], p_now], np.r_[share[w, :k], sh])
            dots[w].set_data([p_now], [sh])
        clock.set_text("joint evidence at %.0f%% of the stream-steps" % p_now)

    xs = np.linspace(0, len(sw) - 1, 4 * (len(sw) - 1) + 1)
    sched = [(xs[0], 1200)] + [(x, 70) for x in xs[1:-1]] + [(xs[-1], 2000)]

    def grab():
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    protos = []
    for x in (0., 6., 20., len(sw) - 1.):
        update(x)
        protos.append(np.asarray(grab()))
    pal = Image.fromarray(np.concatenate(protos, 0)).quantize(colors=255, method=Image.Quantize.MEDIANCUT,
                                                              dither=Image.Dither.NONE)
    frames = []
    for x, _ in sched:
        update(x)
        frames.append(grab().quantize(palette=pal, dither=Image.Dither.NONE))
    plt.close(fig)
    path = fig_path("fig1c_movie_hybrid_sweep.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[ms for _, ms in sched], loop=0)
    print(f"saved {path}: {len(frames)} frames, {os.path.getsize(path) / 1e6:.1f} MB")
    for pc in (5, 10, 25, 50, 75):
        iv = lambda y: np.interp(pc, sw[1:], y[1:])
        print("%3d%%: least entangled (< %.3f nats) unfolded %.0f%% | most entangled (> %.3f nats) %.0f%%" % (
            pc, iv(th[0]), iv(share[0]), iv(th[1]), iv(share[1])))


def movie_sweep_envs(dist_max=3.):
    """the sweep of movie_sweep with every environment's grid drawn (as the envdist movie through time): left, the joint
    evidence at the least entangled stream-steps first; right, the most entangled first. Thin lines: the 39
    environments' grids (the cells' mean estimates (E[r_g], E[r_c]) at T over their 50 streams), each edge coloured by
    the distance of its two cells from the environments' average grid (black), in cells
    -> figure_bin/fig1c_movie_hybrid_sweep_envs.gif"""
    from PIL import Image
    from main.utils import fig_path
    import plot_fig1c_mesh as pm
    Z = np.load(CACHE_SWEEP)
    cm, sw = Z["cm"], Z["sweep"].astype(float)
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    cells = meta["cells"]
    idx = {(int(g), int(c)): k for k, (g, c) in enumerate(cells)}
    edges = np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                      if (g + dg, c + dc) in idx])
    Rv = np.arange(R)
    P = np.stack([(cm[..., :R] * Rv).sum(-1), (cm[..., R:] * Rv).sum(-1)], -1)        # (2, levels, E, K, 2)
    Pa = P.mean(2)                                                                   # average grid
    spread = np.linalg.norm(P - Pa[:, :, None], axis=-1).mean((-1, -2))            # mean distance from the average
    d = np.linalg.norm(Pa - cells, axis=-1).mean(-1)
    share = 100 * (d[:, :1] - d) / (d[:, :1] - d[:, -1:])
    norm = plt.Normalize(0, dist_max)
    pw, gap = 3.2, .5
    W, H = .3 + 2 * pw + gap + .35 + .14 + 1.0, .95 + pw + .7 + 1.55 + .8
    fig = plt.figure(figsize=(W, H), facecolor="white", dpi=72)
    COL = ("#3f88c5", "#d7301f")
    TTL = ("least entangled steps first", "most entangled steps first")
    art = []
    for w in range(2):
        ax = fig.add_axes([(.3 + w * (pw + gap)) / W, (H - .95 - pw) / H, pw / W, pw / H])
        ax.add_collection(LineCollection(cells[edges].astype(float), colors="0.85", linewidths=.5, linestyles=(0, (3, 2)),
                                         zorder=0))
        envs = []
        for e in range(E):
            lc = LineCollection(P[w, 0, e][edges], cmap=pm.FR_CMAP, norm=norm, linewidths=.45, alpha=.7, zorder=2)
            lc.set_array(np.zeros(len(edges)))
            envs.append(ax.add_collection(lc))
        avg = ax.add_collection(LineCollection(Pa[w, 0][edges], colors="k", linewidths=1.1, zorder=4))
        ax.set_xlim(-.9, 9.9), ax.set_ylim(-.9, 9.9), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(COL[w]), sp.set_linewidth(1.4)
        ax.set_title(TTL[w], fontsize=10, color=COL[w], pad=5)
        txt = ax.text(.5, -.03, "", transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=COL[w])
        art.append((w, envs, avg, txt))
    cbx = fig.add_axes([(.3 + 2 * pw + gap + .35) / W, (H - .95 - pw) / H, .14 / W, pw / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=pm.FR_CMAP, norm=norm), cax=cbx, extend="max")
    cb.set_label("edge colour: distance from the\nenvironments' average grid (cells)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_visible(False)
    cax = fig.add_axes([.3 / W + .08, .95 / H, (2 * pw + gap) / W - .16, 1.3 / H])
    for w in range(2):
        cax.plot(sw, spread[w], c=COL[w], lw=1.2, alpha=.35)
    dots = [cax.plot([], [], "o", c=COL[w], ms=7)[0] for w in range(2)]
    trace = [cax.plot([], [], c=COL[w], lw=2.2)[0] for w in range(2)]
    cax.set_xlim(-1, 101), cax.set_ylim(0, spread.max() * 1.08)
    cax.set_xlabel("joint stream-steps (% of the environment's)", fontsize=9)
    cax.set_ylabel("spread across\nenvironments (cells)", fontsize=9)
    cax.tick_params(labelsize=8)
    cax.spines[["top", "right"]].set_visible(False)
    clock = fig.text(.3 / W, (H - .22) / H, "", fontsize=13, va="center")
    import textwrap
    fig.text(.3 / W, .28 / H, textwrap.fill("Hybrid Bayes observer, alias scan: joint likelihood at the selected stream-steps, "
             "the Naive's factorized one elsewhere. Thin: each of the 39 environments' grids (the cells' mean estimates "
             "(E[r_g], E[r_c]) at T over 50 streams); black: their average; dashed: true lattice. Spread: the mean distance "
             "of the environments' cells from the average grid.", 150), fontsize=6.5, color="0.35", va="center",
             linespacing=1.3)

    def update(x):
        j = min(int(np.floor(x)), len(sw) - 2)
        f = x - j
        p_now = (1 - f) * sw[j] + f * sw[j + 1]
        for w, envs, avg, txt in art:
            Pe = (1 - f) * P[w, j] + f * P[w, j + 1]                                # (E, K, 2)
            pa = Pe.mean(0)
            avg.set_segments(pa[edges])
            for e, lc in enumerate(envs):
                dv = np.linalg.norm(Pe[e] - pa, axis=1)
                lc.set_segments(Pe[e][edges])
                lc.set_array((dv[edges[:, 0]] + dv[edges[:, 1]]) / 2)
            sp_ = (1 - f) * spread[w, j] + f * spread[w, j + 1]
            sh = (1 - f) * share[w, j] + f * share[w, j + 1]
            txt.set_text("average grid unfolded %.0f%%, spread %.2f cells" % (sh, sp_))
            k = int(np.searchsorted(sw, p_now, side="right"))
            trace[w].set_data(np.r_[sw[:k], p_now], np.r_[spread[w, :k], sp_])
            dots[w].set_data([p_now], [sp_])
        clock.set_text("joint evidence at %.0f%% of the stream-steps" % p_now)

    xs = np.linspace(0, len(sw) - 1, 4 * (len(sw) - 1) + 1)
    sched = [(xs[0], 1200)] + [(x, 70) for x in xs[1:-1]] + [(xs[-1], 2000)]

    def grab():
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    protos = []
    for x in (0., 6., 20., len(sw) - 1.):
        update(x)
        protos.append(np.asarray(grab()))
    pal = Image.fromarray(np.concatenate(protos, 0)).quantize(colors=255, method=Image.Quantize.MEDIANCUT,
                                                              dither=Image.Dither.NONE)
    frames = []
    for x, _ in sched:
        update(x)
        frames.append(grab().quantize(palette=pal, dither=Image.Dither.NONE))
    plt.close(fig)
    path = fig_path("fig1c_movie_hybrid_sweep_envs.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[ms for _, ms in sched], loop=0)
    print(f"saved {path}: {len(frames)} frames, {os.path.getsize(path) / 1e6:.1f} MB")
    for pc in (0, 10, 26, 50, 76, 100):
        j = list(sw).index(pc)
        print("%3d%%: spread least %.2f / most %.2f cells | unfolded %.0f%% / %.0f%%" % (
            pc, spread[0, j], spread[1, j], share[0, j], share[1, j]))


SWEEP_TEXT = dict(                                       # every text of the combined sweep movie ({}: filled in every frame)
    clock="joint evidence at {p:.0f}% of the stream-steps ({n_env} envs x 50 streams per cell)",
    row_top="the environments' average grid",
    row_bottom="each environment's grid (black: their average; dashed grey: the true lattice)",
    head_left="least entangled steps first",
    head_right="most entangled steps first",
    thr_left="joint where entanglement < {thr:.2f} nats",
    thr_right="joint where entanglement > {thr:.2f} nats",
    naive="no joint evidence yet (the Naive)",
    cbar_top="FR of the edge's cells at T (nats)",
    cbar_bottom="distance from the average grid (cells)",
    curve_title="unfolding",
    curve_x="steps (%)",
    curve_y="grid error")


def movie_sweep_combined(dist_max=3., dpi=100, cache=None, text=None,
                         out="fig1c_movie_hybrid_sweep_combined.gif"):
    """the sweep in one movie: columns, the joint evidence at the least / most entangled stream-steps first, then the
    grid error: the mean distance between the average grid (black) and the true lattice (dashed grey) over the 100
    cells (the full column); top row, the environments' average grid (edges: FR of the edge's
    cells at T); bottom row, every environment's grid (edges: distance from their average, black). cache:
    CACHE_SWEEP100 (100 environments) if it exists, else CACHE_SWEEP (the scan's 39). text: replacements for
    SWEEP_TEXT's entries -> figure_bin/fig1c_movie_hybrid_sweep_combined.gif"""
    from PIL import Image
    from main.utils import fig_path
    import plot_fig1c_mesh as pm
    TX = dict(SWEEP_TEXT, **(text or {}))
    cache = cache or (CACHE_SWEEP100 if os.path.exists(CACHE_SWEEP100) else CACHE_SWEEP)
    Z = np.load(cache)
    cm, thr, sw = Z["cm"], Z["thr"], Z["sweep"].astype(float)
    nE = cm.shape[2]
    cells = np.array([(g, c) for g in range(R) for c in range(R)])
    if "fr" in Z.files:
        fr = np.maximum(Z["fr"].mean(0), 1e-6)
    else:
        meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
        fr = np.maximum(meta["FR"][ENVS].astype(np.float64).reshape(E, K, M, T).mean(2)[:, :, -1].mean(0), 1e-6)
    idx = {(int(g), int(c)): k for k, (g, c) in enumerate(cells)}
    edges = np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                      if (g + dg, c + dc) in idx])
    Rv = np.arange(R)
    P = np.stack([(cm[..., :R] * Rv).sum(-1), (cm[..., R:] * Rv).sum(-1)], -1)        # (2, levels, nE, K, 2)
    Pa = P.mean(2)
    d = np.linalg.norm(Pa - cells, axis=-1).mean(-1)
    th = np.nanmedian(thr, -1)
    frn, dn = LogNorm(.05, 30), plt.Normalize(0, dist_max)
    ecol = np.sqrt(fr[edges[:, 0]] * fr[edges[:, 1]])
    FSZ = dict(clock=15, row=12, title=11.5, num=12, lab=10.5, tick=9.5, note=11)
    pw, gap, rg, cw = 3.0, .45, .8, 4.3
    xcb = .3 + 2 * pw + gap + .3
    xcv = xcb + .16 + .75 + .55
    W = xcv + cw + .3
    H = 1.5 + pw + rg + pw + .55
    fig = plt.figure(figsize=(W, H), facecolor="white", dpi=dpi)
    COL = ("#3f88c5", "#d7301f")
    HEAD = (TX["head_left"], TX["head_right"])
    y_top = H - 1.5 - pw
    y_bot = y_top - rg - pw
    lw_env = .45 if nE <= 40 else .3
    art = []
    for w in range(2):
        x = .3 + w * (pw + gap)
        a1 = fig.add_axes([x / W, y_top / H, pw / W, pw / H])
        a2 = fig.add_axes([x / W, y_bot / H, pw / W, pw / H])
        for ax in (a1, a2):
            ax.add_collection(LineCollection(cells[edges].astype(float), colors="0.85", linewidths=.5,
                                             linestyles=(0, (3, 2)), zorder=0))
            ax.set_xlim(-.9, 9.9), ax.set_ylim(-.9, 9.9), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(COL[w]), sp.set_linewidth(1.4)
        lca = LineCollection(Pa[w, 0][edges], cmap=pm.FR_CMAP, norm=frn, linewidths=1.6, zorder=3)
        lca.set_array(ecol)
        a1.add_collection(lca)
        ttl = a1.set_title("", fontsize=FSZ["title"], color=COL[w], pad=6, linespacing=1.3)
        envs = []
        for e in range(nE):
            lc = LineCollection(P[w, 0, e][edges], cmap=pm.FR_CMAP, norm=dn, linewidths=lw_env, alpha=.65, zorder=2)
            lc.set_array(np.zeros(len(edges)))
            envs.append(a2.add_collection(lc))
        avg = a2.add_collection(LineCollection(Pa[w, 0][edges], colors="k", linewidths=1.2, zorder=4))
        art.append((w, lca, ttl, envs, avg))
    fig.text(.3 / W, (y_top + pw + .75) / H, TX["row_top"].format(n_env=nE),
             fontsize=FSZ["row"], color="0.2", va="bottom")
    fig.text(.3 / W, (y_bot + pw + .12) / H, TX["row_bottom"].format(n_env=nE),
             fontsize=FSZ["row"], color="0.2", va="bottom")
    for y, nm, lab, ext in ((y_top, frn, TX["cbar_top"], "both"), (y_bot, dn, TX["cbar_bottom"], "max")):
        cbx = fig.add_axes([xcb / W, y / H, .16 / W, pw / H])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=pm.FR_CMAP, norm=nm), cax=cbx, extend=ext)
        cb.set_label(lab, fontsize=FSZ["lab"])
        cb.ax.tick_params(labelsize=FSZ["tick"])
        cb.outline.set_visible(False)
    ax = fig.add_axes([xcv / W, (y_bot + .5) / H, cw / W, (y_top + pw - .4 - y_bot - .5) / H])   # the grid error
    ax.set_title(TX["curve_title"], fontsize=FSZ["row"], pad=8)
    for w in range(2):
        ax.plot(sw, d[w], c=COL[w], lw=1.3, alpha=.35)
    trace = [ax.plot([], [], c=COL[w], lw=2.4)[0] for w in range(2)]
    dots = [ax.plot([], [], "o", c=COL[w], ms=7)[0] for w in range(2)]
    ax.set_xlim(-1, 101), ax.set_ylim(0, d.max() * 1.06), ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(TX["curve_x"], fontsize=FSZ["lab"], labelpad=2)
    ax.set_ylabel(TX["curve_y"], fontsize=FSZ["lab"])
    ax.tick_params(labelsize=FSZ["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    import textwrap
    clock = fig.text(.3 / W, (H - .35) / H, "", fontsize=FSZ["clock"], va="center")
    TTL = (TX["thr_left"], TX["thr_right"])

    def update(x):
        j = min(int(np.floor(x)), len(sw) - 2)
        f = x - j
        p_now = (1 - f) * sw[j] + f * sw[j + 1]
        k = int(np.searchsorted(sw, p_now, side="right"))
        for w, lca, ttl, envs, avg in art:
            Pe = (1 - f) * P[w, j] + f * P[w, j + 1]
            pa = Pe.mean(0)
            lca.set_segments(pa[edges])
            avg.set_segments(pa[edges])
            for e, lc in enumerate(envs):
                dv = np.linalg.norm(Pe[e] - pa, axis=1)
                lc.set_segments(Pe[e][edges])
                lc.set_array((dv[edges[:, 0]] + dv[edges[:, 1]]) / 2)
            t_now = (1 - f) * th[w, j] + f * th[w, j + 1] if np.isfinite(th[w, j]) else th[w, j + 1]
            ttl.set_text(HEAD[w] + "\n" + (TTL[w].format(thr=t_now) if p_now > 0 else TX["naive"]))
            dd = (1 - f) * d[w, j] + f * d[w, j + 1]
            trace[w].set_data(np.r_[sw[:k], p_now], np.r_[d[w, :k], dd])
            dots[w].set_data([p_now], [dd])
        clock.set_text(TX["clock"].format(p=p_now, n_env=nE))

    xs = np.linspace(0, len(sw) - 1, 4 * (len(sw) - 1) + 1)
    sched = [(xs[0], 1200)] + [(x, 70) for x in xs[1:-1]] + [(xs[-1], 2000)]

    def grab():
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    protos = []
    for x in (0., 6., 20., len(sw) - 1.):
        update(x)
        protos.append(np.asarray(grab()))
    exact = [(1., 1., 1.), (0., 0., 0.)] + [to_rgb(c) for c in COL] + [(v, v, v) for v in np.linspace(0, 1, 24)]
    ramp = [np.asarray(pm.FR_CMAP(v)[:3]) for v in np.linspace(0, 1, 32)]
    exact += [tuple(c) for c in ramp] + [tuple(.65 * c + .35) for c in ramp]
    exact = np.unique((255 * np.clip(np.array(exact, np.float64), 0, 1)).round().astype(np.uint8), axis=0)
    rest = 255 - len(exact)
    mc = Image.fromarray(np.concatenate(protos, 0)).quantize(colors=rest, method=Image.Quantize.MEDIANCUT,
                                                              dither=Image.Dither.NONE)
    allc = np.concatenate([exact, np.array(mc.getpalette(), np.uint8).reshape(-1, 3)[:rest]], 0)[:255]
    pal = Image.new("P", (1, 1))
    pal.putpalette(allc.ravel().tolist() + [255, 255, 255] * (256 - len(allc)))
    frames = []
    for i, (x, _) in enumerate(sched):
        update(x)
        frames.append(grab().quantize(palette=pal, dither=Image.Dither.NONE))
        if i % 50 == 0:
            print(f"frame {i + 1} of {len(sched)}", flush=True)
    plt.close(fig)
    path = fig_path(out)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[ms for _, ms in sched], loop=0)
    print(f"saved {path}: {len(frames)} frames, {frames[0].size[0]} x {frames[0].size[1]} px, "
          f"{os.path.getsize(path) / 1e6:.1f} MB")


def _pmean(Z, c, envs=slice(None), t=-1):
    cm = Z[f"{c}__cm"][envs][:, :, t]
    return np.stack([(cm[..., :R] * np.arange(R)).sum(-1), (cm[..., R:] * np.arange(R)).sum(-1)], -1).mean(0)


SEL_ENV = {"bottom": ("least entangled steps", "#74a9cf"), "random": ("random steps", "0.45"),
           "cell": ("random steps, as many per cell as the most entangled", "#b8b8b8"),
           "top": ("most entangled steps", "#d7301f")}


def figure(scope="stream"):
    """scope "stream": k steps chosen within each stream (CACHE); "env": pct% of all the environment's stream-steps
    (CACHE_ENV)"""
    from main.utils import apply_fig_style, fig_path
    import plot_fig1c_mesh as pm
    env = scope == "env"
    Z = np.load(CACHE_ENV if env else CACHE)
    sels = SEL_ENV if env else SEL
    levels, full = (PCTS, 100) if env else (KS, T)
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    cells = meta["cells"]
    fr = np.maximum(meta["FR"][ENVS].astype(np.float64).reshape(E, K, M, T).mean(2)[:, :, -1].mean(0), 1e-6)
    hi = fr >= np.quantile(fr, .8)
    idx = {(int(g), int(c)): k for k, (g, c) in enumerate(cells)}
    edges = np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                      if (g + dg, c + dc) in idx])
    dist = lambda c, b=slice(None): np.linalg.norm(_pmean(Z, c, b) - cells, axis=1)
    def share(c, sel, b=slice(None)):
        dn, do, dx = dist("naive", b)[sel].mean(), dist("optimal", b)[sel].mean(), dist(c, b)[sel].mean()
        return (dn - dx) / (dn - do)
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    rows = ("bottom", "top") if env else ("bottom", "random", "top")
    if env:
        sels = {k: v for k, v in sels.items() if k in rows}
    colk = (0,) + levels + (full,) if env else (0, 3, 5, 10, 30)
    pw, gap = 1.75, .12
    W = 1.35 + len(colk) * pw + (len(colk) - 1) * gap + .25 + .12 + .9
    H = .55 + len(rows) * (pw + .12) + .55 + 2.0 + 1.0
    fig = plt.figure(figsize=(W, H))
    norm = LogNorm(.05, 30)
    ecol = np.sqrt(fr[edges[:, 0]] * fr[edges[:, 1]])
    for r, s in enumerate(rows):
        y0 = H - .55 - (r + 1) * (pw + .12)
        for c, k in enumerate(colk):
            cond = "naive" if k == 0 else "optimal" if k == full else f"{s}{k}"
            ax = fig.add_axes([(1.35 + c * (pw + gap)) / W, y0 / H, pw / W, pw / H])
            P = _pmean(Z, cond)
            ax.add_collection(LineCollection(cells[edges].astype(float), colors="0.8", linewidths=.45,
                                             linestyles=(0, (3, 2))))
            lc = LineCollection(P[edges], cmap=pm.FR_CMAP, norm=norm, linewidths=1.2)
            lc.set_array(ecol)
            ax.add_collection(lc)
            ax.set_xlim(-.7, 9.7), ax.set_ylim(-.7, 9.7), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
            if r == 0 and not env:
                ax.set_title({0: "0 joint steps\n(the Naive)", 30: "30 joint steps\n(the Optimal)"}.get(
                    k, "%d joint steps" % k), fontsize=FS["label"], pad=3)
            elif r == 0:
                ax.set_title({0: "no joint steps\n(the Naive)", 100: "all steps joint\n(the Optimal)"}.get(
                    k, "%d%% of the steps joint" % k), fontsize=FS["label"], pad=3)
            if 0 < k < full:
                ax.text(.97, .03, "%.0f%%" % (100 * share(cond, np.ones(K, bool))), transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=FS["tick"] - .5, color=sels[s][1] if s != "random" else "0.3")
            if c == 0:
                ax.text(-.08, .5, "joint at the\n" + sels[s][0].replace(" steps", "\nsteps"), transform=ax.transAxes,
                        rotation=0, ha="right", va="center", fontsize=FS["label"], color=sels[s][1], linespacing=1.15)
    cx = fig.add_axes([(1.35 + len(colk) * pw + (len(colk) - 1) * gap + .25) / W,
                       (H - .55 - len(rows) * (pw + .12)) / H, .12 / W, (len(rows) * (pw + .12) - .12) / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=pm.FR_CMAP, norm=norm), cax=cx, extend="both")
    cb.set_label("factorization regret of the edge's cells at $T$ (nats)", fontsize=FS["tick"])
    cb.outline.set_visible(False)
    fig.text(.15 / W, (H - .25) / H, r"$\mathbf{a}$", fontsize=FS["label"] + 2, va="center")
    fig.text(.5 / W, (H - .25) / H, ("Average grid at $T$ when the joint evidence is used at only a share of all the "
             "environment's stream-steps" if env else "Average grid at $T$ when the joint evidence is used at only $k$ of "
             "the 30 steps") + ": cells at their mean estimates $(E[r_g], E[r_c])$; dashed: true lattice; %: unfolded "
             "share", fontsize=FS["label"] + .5, va="center")
    # curves
    by, bh = 1.0, 2.0
    npan = 1 if env else 3
    bw = (W - 1.35 - .9 - 2 * 1.0) / 3 * (1.3 if env else 1)
    axs = [fig.add_axes([(1.35 + i * (bw + 1.0)) / W, by / H, bw / W, bh / H]) for i in range(npan)]
    xs = np.array((0,) + levels + (full,))
    for s, (lab, col) in sels.items():
        for ax, sel in ((axs[0], np.ones(K, bool)),) + (() if env else ((axs[1], hi),)):
            ys = [0.] + [share(f"{s}{k}", sel) for k in levels] + [1.]
            ax.plot(xs, 100 * np.array(ys), c=col, marker="o", ms=3, lw=1.5 if s == "top" else 1.1,
                    ls=(0, (3, 1.6)) if s in ("timed", "cell") else "-", label=lab)
        if not env:
            ys = [Z["naive__p"].mean()] + [Z[f"{s}{k}__p"].mean() for k in levels] + [Z["optimal__p"].mean()]
            axs[-1].plot(xs, ys, c=col, marker="o", ms=3, lw=1.5 if s == "top" else 1.1,
                         ls=(0, (3, 1.6)) if s in ("timed", "cell") else "-")
    for ax, ttl in ((axs[0], "Unfolding"),) if env else ((axs[0], "Unfolding, all cells"),
                                                         (axs[1], "Unfolding, highest-FR fifth of cells")):
        ax.plot([0, full], [0, 100], c="0.85", lw=.8, zorder=0)
        ax.set_ylabel(r"unfolded share (%), $\frac{d_{Naive} - d}{d_{Naive} - d_{Optimal}}$")
        ax.set_title(ttl, fontsize=FS["label"] + .5, pad=4)
        ax.set_ylim(-3, 103)
    if not env:
        axs[-1].set_ylabel(r"P(true cell) at $T$, $b_T(r_g)\,b_T(r_c)$")
        axs[-1].set_title("Accuracy", fontsize=FS["label"] + .5, pad=4)
    if env:
        axs[0].legend(frameon=False, fontsize=FS["tick"] - 1, loc="lower right", labelspacing=.25)
    else:
        axs[0].legend(frameon=False, fontsize=FS["tick"] - 1.5, loc="lower right", labelspacing=.25)
    for i, ax in enumerate(axs):
        ax.set_xlabel("joint stream-steps (% of the environment's)" if env else "number of joint steps $k$ (of 30)",
                      labelpad=1)
        ax.set_xticks([0, 25, 50, 75, 100] if env else [0, 5, 10, 20, 30])
        ax.spines[["top", "right"]].set_visible(False)
        x0, y0, w, h = ax.get_position().bounds
        fig.text(x0 - .75 / W, y0 + h + 6 / 72 / H, r"$\mathbf{%s}$" % "bcd"[i], fontsize=FS["label"] + 2, va="baseline")
    import textwrap
    fig.text(1.35 / W, .42 / H, textwrap.fill(
        "Alias scan, 39 environments x 100 cells x 50 streams; the hybrid observer keeps a joint belief and multiplies in "
        "the joint likelihood at the selected steps, the Naive's factorized likelihood at the others (none: the "
        "Naive, all: the Optimal, both exact). Entanglement: the scan's per-step house entanglement of the stream. " + (
            "Steps ranked over all 5,000 streams x 30 steps of the environment. Grey line in b: the share proportional to "
            "the steps used." if env else
            "Grey line in b, c: the share proportional to k. d: the most entangled steps buy no more accuracy than random "
            "ones (the least entangled buy less): they move the cells' estimates, not the probability on the true cell."),
        225), fontsize=FS["tick"] - 1.5, color="0.35", va="top", linespacing=1.35)
    path = fig_path("fig1c_hybrid_unfold%s.png" % ("_env" if env else ""))
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    if "build" in sys.argv[1:]:
        build()
    if "build_env" in sys.argv[1:]:
        build_env()
    if {"sweep", "sweep_envs", "sweep_combined"} & set(sys.argv[1:]):
        if not os.path.exists(CACHE_SWEEP):
            build_sweep()
        if "sweep" in sys.argv[1:]:
            movie_sweep()
        if "sweep_envs" in sys.argv[1:]:
            movie_sweep_envs()
        if "sweep_combined" in sys.argv[1:]:
            movie_sweep_combined()
        sys.exit()
    figure("env" if "env" in sys.argv[1:] else "stream")
