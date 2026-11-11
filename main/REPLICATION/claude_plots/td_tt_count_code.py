"""Is the count-decoding dip forgetting, or a change of code? (John 11-08: "I think the R^2 result is interesting because it
implies compression (forgetting) is necessary for crossing Naive. This would be very cool if we can verify is not a
trivial result of regularization or another decoder issues.")
td_summary d: ridge R^2 of the observation counts n_t decoded from u at step 30 (td_analysis.ridge_r2: standardized
features, penalty c * tr(G) / D with c in 1e-4..1 chosen on 5 validation environments, fit on 25 environments, tested
on the other 25) falls from .87 to .43 around passing naive (1,120 training episodes) and recovers to .90 in the
N = 1000 Trained RNN; the chosen c sits at the grid's top (1.0) during the dip.
-> main/DATA/experimental/training_dynamics/through_training/count_code/<run>_EEEEE.npz (per checkpoint), refs.npz
   main/REPLICATION/figure_bin/td_tt_count_code.png   figure()
Episodes: fresh, 1,000 per test environment (the probes' 50 environments: each of the 100 cells x 10 trajectories of
30 steps, observations drawn from the environment's likelihood table L; seed SEED), 50,000 in all. The networks are rerun
on them (td_tt_chaos_ent's drive table and LSTM step); the targets are the counts n_t (5 observables, inclusive of t).
Decoders (features standardized on the fitting rows; R^2 = 1 - SSE / SST pooled over the 5 counts; SST around each
test environment's own mean ("within") and, for the across-environment linear decoder, also around the test set's mean
as td_analysis ("global")):
  lin_x   across environments, linear ridge: fit on the 25 fitting environments (probes.npz dec_train + dec_val),
          penalty c in 10^-6..10^4 (x tr(G) / D) chosen by grouped 5-fold cross-validation over those environments,
          refit, tested on the 25 dec_test environments
  lin_w   within environments, linear ridge: each environment its own decoder, 5-fold over its 1,000 episodes (folds =
          trajectory pairs, every fold holds every cell); one c for all environments, chosen on the fitting
          environments, R^2 on the test environments
  mlp     across environments, MLP (256-256, ReLU, Adam 1e-3, batch 512, early stopping on 5 of the fitting
          environments, at most 200 epochs) on the standardized features; mlp_z: the same with the environment's
          interactions Z (10 numbers) appended
Sources: u (= h, the short-term stream) and [u, c] at step 30; u at step 10 (lin_x only).
References (features = an observer's code of the same episodes, decoded the same way): naive's log marginals (20), Bayes's
log marginals (20), Bayes's log joint posterior (100), each floored 30 nats below its maximum as td_analysis.
RESULTS 11-08 (u at step 30; R^2 "global" across environments, "within" per environment; training episodes)
  Shared code (lin_x, wide grid, grouped CV; the penalty is never at the grid's edge): Trained N = 1000 .88 (40) ->
    .66 (480) -> .50 (800) -> .48 (880) -> .47 (1,040) -> .51 (1,600) -> .66 (3,000) -> .89 (9,600) -> .92 (end); [u, c]
    .95 -> .42 (1,200) -> .91; ESN .86 -> .82 (1,040) -> .80 (end). The original decoder (td_analysis) agrees at every
    checkpoint. Nonlinear control (PCs + random Fourier features, ridge, grouped CV; 200 PCs, 20,000 episodes): .82 ->
    .43 (880) -> .41 (1,200) -> .75 (end); the MLPs (weaker than linear) dip the same way.
  Within environments (lin_w): Trained .93 (init) -> .91 (320-640) -> .92 (880) -> .97 (3,000) -> .99 (end); ESN .93-.94.
    Memory-window equivalents (the last w steps' counts alone, same decoders): across .41 / .69 / .80 / .91 / .98 for
    w = 1 / 3 / 5 / 10 / 20; within .50 / .74 / .82 / .91 / .97. Observer codes: across < 0 (-.15 to -.18 for every
    decoder); within: Bayes's log joint .96, naive's log marginals .91, Bayes's log marginals .75.
  Widths: N = 200 shared .83 -> .57 (1,040) -> .43 (2,080, passing naive) -> .43 (end), within .91-.95 throughout;
    N = 100 shared .81 -> .26 (6,400, passing naive) -> .11 (end), within .89-.92.
  Posteriors (across environments): Trained naive's log marginals .02 -> .36 (320) -> .52 (880) -> .68 (end); Bayes's
    .02 -> .30 (320) -> .53 (880) -> .87 (end): naive's ahead until passing naive, Bayes's after; Bayes - naive ~0 to 320,
    .02 (480), .05 (880), .16 (2,000), .38 (end). ESN: naive's .33, Bayes's .18, Bayes - naive -.01 at the end.
    N = 200: Bayes - naive .01 at its passing naive (2,080), .15 (end; [u, c] .20), with no shared code.
  Reading: the dip is not forgetting (the counts stay decodable within every environment, ~10 steps' worth or more) and
  not the decoder; what goes is the code shared across environments (stimulus-bound, as in the untrained network and
  the ESN), while the state comes to carry environment-specific posteriors (naive's first, then Bayes's and the
  interaction part). Every trained run loses half of its shared code before passing naive; only N = 1000 rebuilds it.
Run from the repository root:
    /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_count_code.py harvest 1
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_count_code.py"""
import os
import sys
import time
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_intervene as ti                                                                  # noqa: E402
import td_tt_chaos as tc                                                                      # noqa: E402
import td_tt_chaos_ent as ce                                                                  # noqa: E402

OUT = os.path.join(gap.TT, "count_code")
SEED, M_PER, T, NO, R = 20261108, 1000, 30, 5, 10
STEPS = (9, 29)                                         # 0-based: steps 10 and 30
CGRID = 10.0 ** np.arange(-6, 4.01, .5)
X_SEL = {"trained": (10, 40, 100, 200, 320, 480, 640, 800, 880, 1040, 1200, 1600, 2000, 3000, 5000, 9600, 20000,
                     39996),
         "trained_N200": (10, 1000, 2080, 5000, 39996), "trained_N100": (10, 2000, 6400, 39996),
         "echo": (10, 1000, 39996)}
FLOOR = 30.


# ── episodes and observers ──────────────────────────────────────────────────────────────────────────────────────────

def episodes():
    """env (n,), cell (n, 2), obs (n, T, 5) bool, counts (n, T, 5): 1,000 per environment (100 cells x 10 trajectories)"""
    P = np.load(os.path.join(ti.AN, "probes.npz"))
    L = P["L"].astype(np.float64)                                               # (50, 5, R, R)
    rng = np.random.default_rng(SEED)
    ne = L.shape[0]
    cells = np.stack(np.meshgrid(np.arange(R), np.arange(R), indexing="ij"), -1).reshape(-1, 2)   # (100, 2)
    env = np.repeat(np.arange(ne), M_PER)
    cell = np.tile(np.repeat(cells, M_PER // len(cells), 0), (ne, 1))
    traj = np.tile(np.tile(np.arange(M_PER // len(cells)), len(cells)), ne)
    p1 = L[env, :, cell[:, 0], cell[:, 1]]                                      # (n, 5)
    obs = rng.random((len(env), T, NO)) < p1[:, None, :]
    return dict(env=env, cell=cell, traj=traj, obs=obs, counts=obs.cumsum(1).astype(np.float64), L=L,
                split_fit=np.concatenate([P["dec_train"], P["dec_val"]]), split_test=P["dec_test"])


def observer_codes(E, t):
    """naive's log marginals (n, 20), Bayes's log marginals (n, 20), Bayes's log joint (n, 100) after t + 1 steps"""
    L, env, n = E["L"], E["env"], E["counts"][:, t]
    k = t + 1
    lL, l1 = np.log(L[env]), np.log1p(-L[env])                                  # (n, 5, R, R)
    lj = np.einsum("ni,nirs->nrs", n, lL) + np.einsum("ni,nirs->nrs", k - n, l1)
    lj = lj - lj.reshape(len(n), -1).max(-1)[:, None, None]
    from scipy.special import logsumexp
    ljn = lj - logsumexp(lj.reshape(len(n), -1), -1)[:, None, None]
    lb = np.stack([logsumexp(ljn, 2), logsumexp(ljn, 1)], 1)                    # (n, 2, R)
    m = np.stack([L[env].mean(3), L[env].mean(2)], 1)                           # (n, 2, 5, R): naive's marginal lik.
    ln = np.einsum("ni,nvir->nvr", n, np.log(m)) + np.einsum("ni,nvir->nvr", k - n, np.log1p(-m))
    ln = ln - logsumexp(ln, -1)[..., None]

    def fl(a):
        a = a.reshape(len(a), -1)
        return np.maximum(a, a.max(-1, keepdims=True) - FLOOR)
    return {"naive_marg": fl(ln), "bayes_marg": fl(lb), "bayes_joint": fl(ljn)}


@torch.no_grad()
def states(run, raw, E, dev, chunk=5000):
    """u and c at STEPS for every episode: {(src, t): (n, H) float32 on dev}"""
    p, table = ce.setup(run, raw, dev)
    Whh = p["LSTM.weight_hh_l0"]
    H = Whh.shape[1]
    code = torch.as_tensor((E["obs"] * (1 << np.arange(NO))).sum(-1).astype(np.int64), device=dev)
    envt = torch.as_tensor(E["env"].astype(np.int64), device=dev)
    n = len(envt)
    out = {(s, t): torch.empty(n, H, device=dev) for s in ("u", "c") for t in STEPS}
    for b0 in range(0, n, chunk):
        sl = slice(b0, min(n, b0 + chunk))
        b = sl.stop - sl.start
        h = p["STM"].reshape(1, H).expand(b, H).contiguous()
        c = p["LTM"].reshape(1, H).expand(b, H).contiguous()
        for t in range(T):
            h, c, _ = tc.lstm_step(table[envt[sl], code[sl, t]], h, c, Whh)
            if t in STEPS:
                out[("u", t)][sl], out[("c", t)][sl] = h, c
    return out


# ── decoders ────────────────────────────────────────────────────────────────────────────────────────────────────────

def _std(X, rows):
    mx, sx = X[rows].mean(0), X[rows].std(0).clamp_min(1e-6)
    return (X - mx) / sx


def _r2(Y, Yh, env, rows, within=True):
    Yr, Yhr, er = Y[rows], Yh[rows], env[rows]
    sse = ((Yr - Yhr) ** 2).sum()
    if within:
        mu = torch.zeros_like(Yr)
        for e in torch.unique(er):
            k = er == e
            mu[k] = Yr[k].mean(0)
        sst = ((Yr - mu) ** 2).sum()
    else:
        sst = ((Yr - Yr.mean(0)) ** 2).sum()
    return float(1 - sse / sst)


def _ridge_fit(Xs, Y, c):
    """primal ridge on standardized features: penalty c * tr(X^T X) / D; returns (B, intercept)"""
    G = Xs.T @ Xs
    lam = c * torch.diagonal(G).sum() / G.shape[0]
    my = Y.mean(0)
    B = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=G.device, dtype=G.dtype), Xs.T @ (Y - my))
    return B, my


def lin_x(X, Y, env, fit_envs, test_envs):
    """across environments: grouped 5-fold CV over the fitting environments for c, refit, test"""
    X, Y = X.double(), Y.double()
    fe = np.asarray(fit_envs)
    folds = np.array_split(fe, 5)
    envn = env.cpu().numpy()
    cv = np.zeros(len(CGRID))
    for f in folds:
        tr = torch.as_tensor(np.isin(envn, np.setdiff1d(fe, f)), device=X.device)
        va = torch.as_tensor(np.isin(envn, f), device=X.device)
        mx, sx = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
        Xt, Xv = (X[tr] - mx) / sx, (X[va] - mx) / sx
        G = Xt.T @ Xt
        ev, V = torch.linalg.eigh(G)
        ev = ev.clamp_min(0)
        my = Y[tr].mean(0)
        P_ = V.T @ (Xt.T @ (Y[tr] - my))
        XvV = Xv @ V
        for j, c in enumerate(CGRID):
            lam = c * ev.sum() / len(ev)
            pred = XvV @ (P_ / (ev + lam)[:, None]) + my
            cv[j] += float(((Y[va] - pred) ** 2).sum())
    cbest = CGRID[int(np.argmin(cv))]
    tr = torch.as_tensor(np.isin(envn, fe), device=X.device)
    te = torch.as_tensor(np.isin(envn, test_envs), device=X.device)
    mx, sx = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
    B, my = _ridge_fit((X[tr] - mx) / sx, Y[tr], cbest)
    Yh = ((X - mx) / sx) @ B + my
    return {"within": _r2(Y, Yh, env, te), "global": _r2(Y, Yh, env, te, within=False), "c": cbest,
            "c_edge": bool(cbest in (CGRID[0], CGRID[-1]))}


def lin_w(X, Y, env, traj, fit_envs, test_envs):
    """within environments: each its own ridge, 5-fold over trajectory pairs; c chosen on the fitting environments"""
    X, Y = X.double(), Y.double()
    sse = {"fit": np.zeros(len(CGRID)), "test": np.zeros(len(CGRID))}
    sst = {"fit": 0., "test": 0.}
    trj = torch.as_tensor(traj, device=X.device)
    for e in np.concatenate([fit_envs, test_envs]):
        part = "fit" if e in fit_envs else "test"
        rows = torch.nonzero(env == int(e)).squeeze(1)
        Xe, Ye, te_ = X[rows], Y[rows], trj[rows]
        sst[part] += float(((Ye - Ye.mean(0)) ** 2).sum())
        for f in range(5):
            va = (te_ // 2) == f
            tr = ~va
            mx, sx = Xe[tr].mean(0), Xe[tr].std(0).clamp_min(1e-6)
            Xt, Xv = (Xe[tr] - mx) / sx, (Xe[va] - mx) / sx
            K = Xt @ Xt.T                                                        # dual form: n_tr x n_tr
            ev, V = torch.linalg.eigh(K)
            ev = ev.clamp_min(0)
            my = Ye[tr].mean(0)
            A = V.T @ (Ye[tr] - my)
            KvV = (Xv @ Xt.T) @ V
            trace = float(ev.sum())                                              # tr(X^T X) = tr(X X^T)
            for j, c in enumerate(CGRID):
                lam = c * trace / Xt.shape[1]
                pred = KvV @ (A / (ev + lam)[:, None]) + my
                sse[part][j] += float(((Ye[va] - pred) ** 2).sum())
    j = int(np.argmin(sse["fit"]))
    return {"within": 1 - sse["test"][j] / sst["test"], "c": CGRID[j], "c_edge": bool(j in (0, len(CGRID) - 1))}


def mlp(X, Y, env, fit_envs, test_envs, Z=None, seed=0, epochs=200, patience=10):
    """across environments: 256-256 ReLU MLP on standardized features (optionally with Z appended), early stopping on
    5 of the fitting environments"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    envn = env.cpu().numpy()
    fe = np.asarray(fit_envs)
    va_e = fe[-5:]
    tr = torch.as_tensor(np.isin(envn, fe[:-5]), device=X.device)
    va = torch.as_tensor(np.isin(envn, va_e), device=X.device)
    te = torch.as_tensor(np.isin(envn, test_envs), device=X.device)
    F = X.float()
    if Z is not None:
        F = torch.cat([F, Z.float()], 1)
    mx, sx = F[tr].mean(0), F[tr].std(0).clamp_min(1e-6)
    F = (F - mx) / sx
    my, sy = Y[tr].mean(0), Y[tr].std(0).clamp_min(1e-6)
    Yn = ((Y - my) / sy).float()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(F.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                              torch.nn.ReLU(), torch.nn.Linear(256, Y.shape[1])).to(X.device)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    itr = torch.nonzero(tr).squeeze(1)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        perm = itr[torch.randperm(len(itr), generator=g).to(X.device)]
        net.train()
        for b0 in range(0, len(perm), 512):
            ix = perm[b0:b0 + 512]
            loss = ((net(F[ix]) - Yn[ix]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(((net(F[va]) - Yn[va]) ** 2).mean())
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        Yh = net(F).double() * sy + my
    return {"within": _r2(Y.double(), Yh, env, te), "global": _r2(Y.double(), Yh, env, te, within=False),
            "epochs": ep + 1}


def decode_all(feats, E, dev, full=True):
    """every decoder on every feature set: {name: {decoder: result}}"""
    env = torch.as_tensor(E["env"], device=dev)
    Zt = torch.as_tensor(np.load(os.path.join(ti.AN, "probes.npz"))["Z"].reshape(50, -1)[E["env"]], device=dev)
    fit, test = E["split_fit"], E["split_test"]
    res = {}
    for name, (X, t) in feats.items():
        Y = torch.as_tensor(E["counts"][:, t], device=dev)
        r = {"lin_x": lin_x(X, Y, env, fit, test)}
        if full:
            r["lin_w"] = lin_w(X, Y, env, E["traj"], fit, test)
            r["mlp"] = mlp(X, Y, env, fit, test)
            r["mlp_z"] = mlp(X, Y, env, fit, test, Z=Zt)
        res[name] = r
    return res


# ── harvest ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def raw_of(run, xv):
    eps = np.array(ti.checkpoints(run))
    return int(eps[np.argmin(np.abs(np.log(gap.train_x(eps)) - np.log(xv)))])


def harvest(gpu=1):
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_per_process_memory_fraction(min(1., 7 * 2 ** 30 / torch.cuda.get_device_properties(dev).total_memory),
                                               dev)
    E = episodes()
    ref = os.path.join(OUT, "refs.npz")
    if not os.path.exists(ref):
        t0 = time.time()
        feats = {}
        for t in STEPS:
            for k, v in observer_codes(E, t).items():
                feats[f"{k}@{t + 1}"] = (torch.as_tensor(v, device=dev), t)
        res = decode_all(feats, E, dev)
        np.savez(ref, res=np.array(res, dtype=object))
        print(f"refs: {time.time() - t0:.0f} s", flush=True)
        for k, v in res.items():
            print("  ", k, {d: round(r["within"], 3) for d, r in v.items()}, flush=True)
    for run, xs in X_SEL.items():
        for xv in xs:
            raw = raw_of(run, xv)
            path = os.path.join(OUT, f"{run}_{raw:05d}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            S = states(run, raw, E, dev)
            feats = {"u@30": (S[("u", 29)], 29), "uc@30": (torch.cat([S[("u", 29)], S[("c", 29)]], 1), 29)}
            res = decode_all(feats, E, dev)
            res.update(decode_all({"u@10": (S[("u", 9)], 9), "c@30": (S[("c", 29)], 29)}, E, dev, full=False))
            np.savez(path, res=np.array(res, dtype=object), x=gap.train_x(raw), raw=raw)
            del S
            torch.cuda.empty_cache()
            u = res["u@30"]
            print(f"{run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s | u@30 lin_x {u['lin_x']['global']:.3f}"
                  f" (within {u['lin_x']['within']:.3f}, c {u['lin_x']['c']:.0e}{' EDGE' if u['lin_x']['c_edge'] else ''})"
                  f" lin_w {u['lin_w']['within']:.3f} mlp {u['mlp']['within']:.3f} mlp_z {u['mlp_z']['within']:.3f} | uc"
                  f" lin_x {res['uc@30']['lin_x']['within']:.3f} lin_w {res['uc@30']['lin_w']['within']:.3f} mlp"
                  f" {res['uc@30']['mlp']['within']:.3f}", flush=True)




# ── reference: what a memory of the last w steps alone can decode ───────────────────────────────────────────────────

WINDOWS = (1, 2, 3, 5, 8, 10, 15, 20, 30)


def window_refs():
    """R^2 of the counts at step 30 decoded (lin_x, lin_w) from the counts of the last w steps only (5 numbers; for a
    linear decoder the sum is sufficient: the w observations are exchangeable) -> OUT/window_refs.npz"""
    path = os.path.join(OUT, "window_refs.npz")
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))["res"].item()
    E = episodes()
    dev = torch.device("cpu")
    env = torch.as_tensor(E["env"])
    Y = torch.as_tensor(E["counts"][:, T - 1])
    res = {}
    for w in WINDOWS:
        Xw = Y - (torch.as_tensor(E["counts"][:, T - 1 - w]) if w < T else 0.)
        res[w] = {"lin_x": lin_x(Xw, Y, env, E["split_fit"], E["split_test"]),
                  "lin_w": lin_w(Xw, Y, env, E["traj"], E["split_fit"], E["split_test"])}
        print(w, {k: round(v["within"], 3) for k, v in res[w].items()}, round(res[w]["lin_x"]["global"], 3), flush=True)
    np.savez(path, res=np.array(res, dtype=object))
    return res


# ── nonlinear control that contains the linear decoder: ridge on [PCs, random Fourier features] ─────────────────────

RFF_X = {"trained": (10, 320, 640, 880, 1040, 1200, 1600, 2000, 5000, 39996), "trained_N200": (10, 2080, 39996),
         "echo": (10, 39996)}
N_PC, N_RFF_IN, N_RFF = 200, 50, 2048
RFF_TRAJ = 4                                               # trajectories per cell kept for the RFF pass (all cells)


def rff_x(X, Y, env, fit_envs, test_envs, seed=0):
    """across environments: PCA of the standardized features on the fitting rows; features = the top N_PC PCs plus
    N_RFF random Fourier features (Gaussian kernel, bandwidth = the median pairwise distance) of the top N_RFF_IN PCs;
    ridge with grouped 5-fold CV over the fitting environments (lin_x). Contains the linear decoder of the top PCs, so a
    lower R^2 than lin_x can only come from the PC truncation"""
    envn = env.cpu().numpy()
    tr = torch.as_tensor(np.isin(envn, fit_envs), device=X.device)
    Xd = X.double()
    mx, sx = Xd[tr].mean(0), Xd[tr].std(0).clamp_min(1e-6)
    Xs = (Xd - mx) / sx
    U_, S_, Vh = torch.linalg.svd(Xs[tr], full_matrices=False)
    P = Xs @ Vh[:N_PC].T                                                      # (n, N_PC)
    g = torch.Generator(device="cpu").manual_seed(seed)
    Pin = P[:, :N_RFF_IN] / P[tr, :N_RFF_IN].std(0).clamp_min(1e-9)
    sub = torch.nonzero(tr).squeeze(1)[torch.randperm(int(tr.sum()), generator=g)[:2000].to(X.device)]
    bw = torch.cdist(Pin[sub], Pin[sub]).median()
    Wr = torch.randn(N_RFF_IN, N_RFF, generator=g, dtype=torch.float64).to(X.device) / bw
    br = torch.rand(N_RFF, generator=g, dtype=torch.float64).to(X.device) * 2 * np.pi
    F = torch.cat([P, np.sqrt(2. / N_RFF) * torch.cos(Pin @ Wr + br)], 1)
    return lin_x(F, Y, env, fit_envs, test_envs)


def harvest_rff(gpu=1):
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_per_process_memory_fraction(min(1., 6 * 2 ** 30 / torch.cuda.get_device_properties(dev).total_memory),
                                               dev)
    E = episodes()
    keep = E["traj"] < RFF_TRAJ                                # 100 cells x RFF_TRAJ trajectories per environment
    E = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in E.items()}
    env = torch.as_tensor(E["env"], device=dev)
    for run, xs in RFF_X.items():
        for xv in xs:
            raw = raw_of(run, xv)
            path = os.path.join(OUT, f"rff_{run}_{raw:05d}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            S = states(run, raw, E, dev)
            Y = torch.as_tensor(E["counts"][:, 29], device=dev)
            res = {"u@30": rff_x(S[("u", 29)], Y, env, E["split_fit"], E["split_test"]),
                   "uc@30": rff_x(torch.cat([S[("u", 29)], S[("c", 29)]], 1), Y, env, E["split_fit"], E["split_test"])}
            np.savez(path, res=np.array(res, dtype=object), x=gap.train_x(raw), raw=raw)
            del S
            torch.cuda.empty_cache()
            print(f"rff {run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s | u {res['u@30']['global']:.3f} "
                  f"(c {res['u@30']['c']:.0e}) uc {res['uc@30']['global']:.3f}", flush=True)


# ── figure ──────────────────────────────────────────────────────────────────────────────────────────────────────────

REP = ("trained", "trained_p1", "trained_N200", "trained_N100", "trained_N50")
REP_STYLE = {"trained": ("#08306b", "N = 1000"), "trained_p1": ("#2171b5", "N = 1000, loss power 1"),
             "trained_N200": ("#4292c6", "N = 200"), "trained_N100": ("#6baed6", "N = 100"),
             "trained_N50": ("#9ecae1", "N = 50")}
NAIVE_ACC = .4641


def cached_series(run):
    """td_analysis's own decoding at every checkpoint (dec_r2 at step 30: [u, c] x [counts, log joint, marginals]),
    goal accuracy at T, training episodes and the run's passing-naive time c0 (stays above naive's accuracy)"""
    import glob
    fs = sorted(glob.glob(os.path.join(ti.AN, run, "ck_*.npz")))
    eps = np.array([int(np.load(f)["episode"]) for f in fs])
    R = np.array([np.load(f)["dec_r2"][9] for f in fs])
    acc = np.array([np.load(f)["acc"][-1] for f in fs])
    x = gap.train_x(eps)
    c0 = next((x[i] for i in range(len(acc)) if (acc[i:] > NAIVE_ACC).all()), np.nan)
    return dict(x=x, R=R, acc=acc, c0=c0)


def harvested(run):
    """this run's per-checkpoint files only (<run>_EEEEE.npz: 'trained_*' would also match trained_N200_*)"""
    import glob
    import re
    fs = sorted(f for f in glob.glob(os.path.join(OUT, f"{run}_*.npz"))
                if re.fullmatch(rf"{re.escape(run)}_\d{{5}}\.npz", os.path.basename(f)))
    rows = [dict(np.load(f, allow_pickle=True)) for f in fs]
    rows = sorted(rows, key=lambda z: float(z["x"]))
    return [float(z["x"]) for z in rows], [z["res"].item() for z in rows]


def figure():
    import matplotlib.pyplot as plt
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    blue, orange = "#1f77b4", "#ff7f0e"
    cross = 880
    W = window_refs()
    refs = dict(np.load(os.path.join(OUT, "refs.npz"), allow_pickle=True))["res"].item()
    xt, Rt = harvested("trained")
    xe, Re = harvested("echo")
    C = {r: cached_series(r) for r in REP + ("echo",)}
    fig, ax = plt.subplots(2, 3, figsize=(17.5, 10))
    fig.subplots_adjust(left=.06, right=.93, top=.9, bottom=.15, wspace=.42, hspace=.55)
    ax = ax.ravel()

    def head(a, text):
        a.set_title(text, loc="left", fontsize=12.5, linespacing=1.35)

    def get(R, src, dec, key="global"):
        return np.array([r[src][dec][key] if dec in r[src] else np.nan for r in R])

    def window_axis(a, kind):
        """right-hand ticks: the R^2 a memory of the last w steps alone would give (lin_x 'global' / lin_w 'within')"""
        a2 = a.twinx()
        a2.set_ylim(a.get_ylim())
        ws = (1, 2, 3, 5, 10, 20)
        a2.set_yticks([W[w][kind[0]][kind[1]] for w in ws])
        a2.set_yticklabels([f"{w}" for w in ws], fontsize=9.5, color=".4")
        a2.set_ylabel("memory of the last w steps alone", fontsize=10, color=".4")
        a2.spines[["top"]].set_visible(False)
    # a: the shared code, td_analysis's decoder at every checkpoint and the wide-grid decoder here
    a = ax[0]
    a.plot(C["trained"]["x"], C["trained"]["R"][:, 0, 0], color=blue, lw=1, alpha=.5)
    a.plot(C["echo"]["x"], C["echo"]["R"][:, 0, 0], color=orange, lw=1, alpha=.5)
    a.plot(xt, get(Rt, "u@30", "lin_x"), "o-", color=blue, lw=2, ms=4, label=r"Trained RNN, $u$")
    a.plot(xt, get(Rt, "uc@30", "lin_x"), "s--", color=blue, lw=1.4, ms=3.5, label=r"Trained RNN, $[u, c]$")
    a.plot(xe, get(Re, "u@30", "lin_x"), "o-", color=orange, lw=2, ms=4, label=r"ESN, $u$")
    a.set_ylim(.3, 1.)
    head(a, "a   The shared count code fades and returns\n" + r"counts at step 30 decoded across environments, R$^2$")
    a.set_ylabel(r"R$^2$ on 25 unseen environments")
    window_axis(a, ("lin_x", "global"))
    # b: decoder controls (u at step 30)
    a = ax[1]
    a.plot(xt, get(Rt, "u@30", "lin_x"), "o-", color=blue, lw=2, ms=4, label="linear (grouped CV)")
    xr, Rr = [], []
    import glob
    import re
    for f in sorted(f for f in glob.glob(os.path.join(OUT, "rff_trained_*.npz"))
                    if re.fullmatch(r"rff_trained_\d{5}\.npz", os.path.basename(f))):
        z = dict(np.load(f, allow_pickle=True))
        xr.append(float(z["x"]))
        Rr.append(z["res"].item()["u@30"]["global"])
    if xr:
        o = np.argsort(xr)
        a.plot(np.array(xr)[o], np.array(Rr)[o], "D-", color="k", lw=1.6, ms=4, label="+ Fourier features\n(200 PCs, 20k episodes)")
    a.plot(xt, get(Rt, "u@30", "mlp"), "^:", color=blue, lw=1.4, ms=4, label="MLP")
    a.plot(xt, get(Rt, "u@30", "mlp_z"), "v:", color=".5", lw=1.2, ms=4, label=r"MLP + $Z$")
    a.set_ylim(.3, 1.)
    head(a, "b   Not the decoder: the dip under every\ndecoder (Trained RNN, " + r"$u$ at step 30)")
    a.set_ylabel(r"R$^2$ on 25 unseen environments")
    # c: within each environment
    a = ax[2]
    a.plot(xt, get(Rt, "u@30", "lin_w", "within"), "o-", color=blue, lw=2, ms=4, label=r"Trained RNN, $u$")
    a.plot(xt, get(Rt, "uc@30", "lin_w", "within"), "s--", color=blue, lw=1.4, ms=3.5, label=r"Trained RNN, $[u, c]$")
    a.plot(xe, get(Re, "u@30", "lin_w", "within"), "o-", color=orange, lw=2, ms=4, label=r"ESN, $u$")
    for k, lab, ls in (("bayes_joint@30", "code: Bayes's log joint", "-"),
                       ("naive_marg@30", "code: naive's log marginals", "--"),
                       ("bayes_marg@30", "code: Bayes's log marginals", ":")):
        a.axhline(refs[k]["lin_w"]["within"], color="k", lw=1, ls=ls, label=lab)
    a.set_ylim(.45, 1.)
    head(a, "c   Within each environment the counts stay\n" + r"decodable (linear, own decoder per environment)")
    a.set_ylabel(r"R$^2$ within environments")
    window_axis(a, ("lin_w", "within"))
    # d: replication on each run's own clock
    a = ax[3]
    for r in REP:
        col, lab = REP_STYLE[r]
        a.plot(C[r]["x"] / C[r]["c0"], C[r]["R"][:, 0, 0], color=col, lw=1.8, label=lab)
    a.axvline(1, color=".55", ls=":", lw=1)
    a.set_xscale("log")
    a.set_xlabel("training episodes / own passing-naive time")
    a.set_ylim(0, 1.)
    head(a, "d   Every trained run loses the shared code\nbefore passing naive; only N = 1000 rebuilds it")
    a.set_ylabel(r"R$^2$ (original decoder, $u$ at step 30)")
    # e: which posterior the state carries (fresh episodes, across environments)
    a = ax[4]

    def post_series(run):
        import glob
        import re
        fs = sorted(f for f in glob.glob(os.path.join(OUT, f"post_{run}_*.npz"))
                    if re.fullmatch(rf"post_{re.escape(run)}_\d{{5}}\.npz", os.path.basename(f)))
        rows = sorted((dict(np.load(f, allow_pickle=True)) for f in fs), key=lambda z: float(z["x"]))
        xx = np.array([float(z["x"]) for z in rows])
        get_ = lambda k: np.array([z["res"].item()[("u", 30, k)]["lin_x"]["global"] for z in rows])  # noqa: E731
        return xx, {k: get_(k) for k in ("naive", "bayes", "bayes_minus_naive")}
    for run, col, mk in (("trained", blue, "o"), ("trained_N200", "#6baed6", "s"), ("echo", orange, "o")):
        xx, Pp = post_series(run)
        a.plot(xx, Pp["bayes"], color=col, lw=1.8, marker=mk, ms=3.5)
        a.plot(xx, Pp["naive"], color=col, lw=1.4, ls=(0, (4, 2)), marker=mk, ms=3)
        a.plot(xx, Pp["bayes_minus_naive"], color=col, lw=2.2, ls=(0, (1, 1.2)), marker=mk, ms=3)
    a.set_ylim(-.05, 1.)
    head(a, "e   The state carries naive's posterior first,\nthen Bayes's and the interaction part (Bayes " + r"$-$" +
         " naive)")
    a.set_ylabel(r"R$^2$ on 25 unseen environments")
    from matplotlib.lines import Line2D
    a.legend(handles=[Line2D([], [], color=".3", ls=(0, (4, 2)), label="naive's log marginals"),
                      Line2D([], [], color=".3", ls="-", label="Bayes's log marginals"),
                      Line2D([], [], color=".3", ls=(0, (1, 1.2)), lw=2.2, label="Bayes's $-$ naive's"),
                      Line2D([], [], color="#6baed6", marker="s", ls="none", label="N = 200")],
             fontsize=9.5, frameon=False, loc="upper left")
    # f: against each run's own accuracy
    a = ax[5]
    for r in REP:
        col, lab = REP_STYLE[r]
        a.plot(C[r]["acc"], C[r]["R"][:, 0, 0], color=col, lw=1.6, label=lab)
    a.plot(C["echo"]["acc"], C["echo"]["R"][:, 0, 0], color=orange, lw=1.8, label="ESN")
    a.axvline(NAIVE_ACC, color="k", ls=":", lw=1)
    a.text(NAIVE_ACC + .005, .03, "naive", fontsize=9.5, color=".3")
    a.set_xlabel("goal accuracy at T (own)")
    a.set_ylim(0, 1.)
    head(a, "f   No run passes naive with the shared code\n" + r"R$^2$ against the run's own accuracy")
    a.set_ylabel(r"R$^2$ (original decoder)")
    a.legend(fontsize=9, frameon=False, loc="lower left")
    import stage_bands as sb
    for a in (ax[0], ax[1], ax[2], ax[4]):
        sb.shade(a)
        a.set_xscale("log")
        a.set_xticks([10, 100, 1e3, 1e4])
        a.set_xticklabels(["0", "100", "1k", "10k"])
        a.set_xlim(8, 4.4e4)
        a.minorticks_off()
        a.set_xlabel("training episodes")
        a.axvline(cross, color=".55", ls=":", lw=1)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    for a, loc in ((ax[0], "lower left"), (ax[1], "lower left"), (ax[2], "lower left"), (ax[3], "lower left")):
        a.legend(fontsize=9, frameon=False, loc=loc)
    import textwrap
    note = ("Seed-20261029 networks. a-c: 50,000 fresh episodes (1,000 per test environment: 100 cells x 10 "
            "trajectories), decoders fit on 25 environments and tested on the other 25 (c: each environment's own "
            "decoder, 5-fold over its episodes); right axes: the R2 that a memory of only the last w steps' counts gives "
            "under the same decoder. a, thin lines: the original decoder at every checkpoint. d-f: the original decoder "
            "(td-analysis: 10,000 probes, penalty chosen on 5 environments) at every checkpoint of five trained runs and "
            "the ESN. Dotted vertical: the Trained RNN first exceeds naive (880). Shaded: the behavioural stages (half-way "
            "spans of answers / splitting, 774-1,575, and of holding / coupling, 2,747-11,103).")
    fig.text(.5, .06, "\n".join(textwrap.wrap(note, 190)), ha="center", va="center", fontsize=10.5, color=".3",
             linespacing=1.5)
    path = fig_path("td_tt_count_code.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)


# ── which posterior does the state carry? naive's, Bayes's, and the interaction's part (Bayes - naive) ──────────────

POST_X = {"trained": (10, 40, 100, 200, 320, 480, 640, 800, 880, 1040, 1200, 1600, 2000, 3000, 5000, 9600, 20000,
                      39996),
          "echo": (10, 1000, 39996), "trained_N200": (10, 1000, 2080, 5000, 39996)}


def post_targets(E, t):
    """naive's and Bayes's log marginals (n, 20) after t + 1 steps (each floored 30 nats below its variable's maximum,
    renormalised per variable after flooring is not needed for R^2) and their difference, Bayes - naive"""
    oc = observer_codes(E, t)
    nb, bb = oc["naive_marg"], oc["bayes_marg"]
    return {"naive": nb, "bayes": bb, "bayes_minus_naive": bb - nb}


def harvest_post(gpu=1):
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_per_process_memory_fraction(min(1., 5 * 2 ** 30 / torch.cuda.get_device_properties(dev).total_memory),
                                               dev)
    E = episodes()
    env = torch.as_tensor(E["env"], device=dev)
    Ys = {t: {k: torch.as_tensor(v, device=dev) for k, v in post_targets(E, t).items()} for t in STEPS}
    for run, xs in POST_X.items():
        for xv in xs:
            raw = raw_of(run, xv)
            path = os.path.join(OUT, f"post_{run}_{raw:05d}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            S = states(run, raw, E, dev)
            res = {}
            for t in STEPS:
                for src, X in (("u", S[("u", t)]), ("uc", torch.cat([S[("u", t)], S[("c", t)]], 1))):
                    for k, Y in Ys[t].items():
                        res[(src, t + 1, k)] = {"lin_x": lin_x(X, Y, env, E["split_fit"], E["split_test"]),
                                                "lin_w": lin_w(X, Y, env, E["traj"], E["split_fit"], E["split_test"])}
            np.savez(path, res=np.array(res, dtype=object), x=gap.train_x(raw), raw=raw)
            del S
            torch.cuda.empty_cache()
            g = lambda k: res[("u", 30, k)]                                                    # noqa: E731
            print(f"post {run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s | u@30 across/within: naive "
                  f"{g('naive')['lin_x']['global']:.3f}/{g('naive')['lin_w']['within']:.3f} bayes "
                  f"{g('bayes')['lin_x']['global']:.3f}/{g('bayes')['lin_w']['within']:.3f} bayes-naive "
                  f"{g('bayes_minus_naive')['lin_x']['global']:.3f}/{g('bayes_minus_naive')['lin_w']['within']:.3f}",
                  flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "harvest":
        harvest(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    elif len(sys.argv) > 1 and sys.argv[1] == "post":
        harvest_post(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    elif len(sys.argv) > 1 and sys.argv[1] == "rff":
        harvest_rff(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    else:
        figure()


def harvest_post_fast(gpu=1):
    """harvest_post without the within-environment decoders (which need 250 eigendecompositions per target): across
    environments only (lin_x), u and [u, c] at step 30; same file names and keys as harvest_post"""
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_per_process_memory_fraction(min(1., 5 * 2 ** 30 / torch.cuda.get_device_properties(dev).total_memory),
                                               dev)
    E = episodes()
    env = torch.as_tensor(E["env"], device=dev)
    Ys = {k: torch.as_tensor(v, device=dev) for k, v in post_targets(E, 29).items()}
    for run, xs in POST_X.items():
        for xv in xs:
            raw = raw_of(run, xv)
            path = os.path.join(OUT, f"post_{run}_{raw:05d}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            S = states(run, raw, E, dev)
            res = {}
            for src, X in (("u", S[("u", 29)]), ("uc", torch.cat([S[("u", 29)], S[("c", 29)]], 1))):
                for k, Y in Ys.items():
                    res[(src, 30, k)] = {"lin_x": lin_x(X, Y, env, E["split_fit"], E["split_test"])}
            np.savez(path, res=np.array(res, dtype=object), x=gap.train_x(raw), raw=raw)
            del S
            torch.cuda.empty_cache()
            g = lambda k: res[("u", 30, k)]["lin_x"]["global"]                                 # noqa: E731
            print(f"post {run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s | u@30 across: naive "
                  f"{g('naive'):.3f} bayes {g('bayes'):.3f} bayes-naive {g('bayes_minus_naive'):.3f} | uc: naive "
                  f"{res[('uc', 30, 'naive')]['lin_x']['global']:.3f} bayes {res[('uc', 30, 'bayes')]['lin_x']['global']:.3f}"
                  f" bayes-naive {res[('uc', 30, 'bayes_minus_naive')]['lin_x']['global']:.3f}", flush=True)


# ── belief before correction, across widths: the posterior pass on every replication run ───────────────────────────

POST_X_RUNS = {"trained_p1": (10, 100, 320, 640, 960, 1280, 1600, 2000, 3000, 5000, 10000, 39996),
               "trained_N200": (100, 320, 640, 1500, 3000, 10000, 20000),
               "trained_N100": (10, 320, 1000, 3000, 4800, 6400, 8000, 12000, 20000, 39996),
               "trained_N50": (10, 1000, 5000, 10000, 16000, 21600, 28000, 39996),
               "trained": (60, 150, 400, 560, 720, 960, 1360)}


def harvest_post_runs(device="cpu"):
    """harvest_post_fast's measures (naive's / Bayes's posterior and their difference, across environments) at the
    POST_X_RUNS checkpoints (CPU by default: the GPUs are busy)"""
    dev = torch.device(device)
    E = episodes()
    env = torch.as_tensor(E["env"], device=dev)
    Ys = {k: torch.as_tensor(v, device=dev) for k, v in post_targets(E, 29).items()}
    for run, xs in POST_X_RUNS.items():
        for xv in xs:
            raw = raw_of(run, xv)
            path = os.path.join(OUT, f"post_{run}_{raw:05d}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            S = states(run, raw, E, dev)
            res = {}
            for src, X in (("u", S[("u", 29)]), ("uc", torch.cat([S[("u", 29)], S[("c", 29)]], 1))):
                for k, Y in Ys.items():
                    res[(src, 30, k)] = {"lin_x": lin_x(X, Y, env, E["split_fit"], E["split_test"])}
            np.savez(path, res=np.array(res, dtype=object), x=gap.train_x(raw), raw=raw)
            g = lambda k: res[("u", 30, k)]["lin_x"]["global"]                                 # noqa: E731
            print(f"post {run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s | naive {g('naive'):.3f} "
                  f"bayes {g('bayes'):.3f} bayes-naive {g('bayes_minus_naive'):.3f}", flush=True)


def belief_before_correction():
    """td_tt_belief_correction.png: the interaction part of the posterior (Bayes - naive, decoded across environments
    from u at step 30) against (a) how well the state carries naive's posterior, (b) the run's own clock, for every
    trained run and the ESN; printed: each run's onset (first checkpoint with interaction R^2 >= .01) and the naive
    posterior's R^2 there"""
    import glob
    import re
    import matplotlib.pyplot as plt
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    fig.subplots_adjust(left=.08, right=.98, top=.82, bottom=.24, wspace=.3)
    out = {}
    for run in REP + ("echo",):
        fs = [f for f in glob.glob(os.path.join(OUT, f"post_{run}_*.npz"))
              if re.fullmatch(rf"post_{re.escape(run)}_\d{{5}}\.npz", os.path.basename(f))]
        rows = sorted((dict(np.load(f, allow_pickle=True)) for f in fs), key=lambda z: float(z["x"]))
        x = np.array([float(z["x"]) for z in rows])
        g = lambda k: np.array([z["res"].item()[("u", 30, k)]["lin_x"]["global"] for z in rows])  # noqa: E731
        nv, ia = g("naive"), g("bayes_minus_naive")
        col, lab = (REP_STYLE[run] if run in REP_STYLE else ("#ff7f0e", "ESN"))
        ax[0].plot(nv, ia, "o-", color=col, lw=1.7, ms=3.5, label=lab)
        if run != "echo":
            c0 = cached_series(run)["c0"]
            ax[1].plot(x / c0, ia, "o-", color=col, lw=1.7, ms=3.5, label=lab)
        on = np.flatnonzero(ia >= .01)
        out[run] = (x[on[0]] if len(on) else np.nan, nv[on[0]] if len(on) else np.nan)
        print(f"{run:13s} onset (interaction R2 >= .01) at {out[run][0]:,.0f} episodes, naive-posterior R2 there "
              f"{out[run][1]:.3f}", flush=True)
    ax[0].axvspan(.40, .47, color="0.85", lw=0, zorder=0)
    ax[0].set_xlabel(r"naive's posterior decoded from $u$ (R$^2$ across environments)")
    ax[0].set_ylabel(r"interaction part, Bayes $-$ naive (R$^2$)")
    ax[0].set_title("a   The interaction part appears once the state\ncarries naive's posterior (grey: .40-.47)",
                    loc="left", fontsize=12.5)
    ax[1].set_xscale("log")
    ax[1].axvline(1, color=".55", ls=":", lw=1)
    ax[1].set_xlabel("training episodes / own passing-naive time")
    ax[1].set_title("b   ... not once the run passes naive\n(the same curves on each run's own clock)", loc="left",
                    fontsize=12.5)
    for a in ax:
        a.axhline(0, color="k", lw=.6)
        a.axhline(.01, color=".6", lw=.7, ls=(0, (2, 2)))
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=9.5, frameon=False, loc="upper left")
    fig.text(.5, .07, "u at step 30 of 50,000 fresh episodes; decoders fit on 25 test environments, tested on the other "
             "25 (linear ridge, grouped CV).\nDashed: R2 = .01, the onset threshold. Five trained runs (one seed each) and "
             "the ESN.", ha="center", va="center", fontsize=10.5, color=".3", linespacing=1.5)
    path = fig_path("td_tt_belief_correction.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)
    return out
