"""Exact read-outs for fixed reservoirs. The question: how sensitive are the ESN results to the reservoir's hyperparameters
(spectral radius, leak rate, sparsity) and to read-out regularization, and can a large reservoir with tuned
hyperparameters beat Naive Bayes? (John 11-09: parts 1-3 below.)

With the reservoir fixed, the best linear read-out is a convex problem, so it is solved exactly instead of by SGD: the
reservoir runs once over a fixed episode set, X^T X and X^T Y accumulate, and one eigendecomposition gives the ridge
read-out at every regularization strength lambda (33 values, 10^-7 .. 10^1 times the mean eigenvalue). Targets: the
one-hot value of each variable (2 x 10, the classical ESN read-out); decision: argmax of the goal variable's 10 outputs at
T = 30. Two feature sets: the final state h_T (the classical read-out) and the time-summed state sum_t h_t (the paper's
ESN, whose per-step read-outs are cumulated). lambda and the settings are chosen on one set of novel environments and
reported on another.

Episodes (harvest, once): the environment of the paper's ESN (DATA/sanity/reservoir_ctx_2_e5: SANITY, |S| 500, d_E 30,
two variables, T = 30, the true Z as input): 200,000 training-environment episodes for the fit, 20,000 novel-environment
episodes (both variables held out) for the choices, 20,000 more novel ones and 20,000 held-out training-environment ones
to report. Inputs per step: the 5 observation bits and the 10 Z values (the networks' read-in input).
  part 1/2  classical leaky-tanh ESN, h <- (1 - a) h + a tanh(W h + W_in x + b), h_0 = 0: W Gaussian with connection
            density p, scaled to spectral radius rho (exactly for N <= 2,000, else by the circular law); W_in and b
            uniform in [-1, 1] times the input scales s_obs (observation bits), s_z (Z values) and s_b.
            Configs 0-3 are textbook settings, 4+ a random search (config(i)).
  part 3    the paper's ESN reservoir (frozen LSTM with its trained read-in and initial state): its own SGD read-out and
            the exact read-out on its states; the same for the trained RNN (fully_trained_ctx_2_e5) as a check of what
            the ridge read-out loses against a trained one.

    python esn_convex.py harvest                               # dataset + part 3, local CPU
    python esn_convex.py sweep --part 0 --of 10 [--n-cfg 300]  # part 1 at N = 1,000: configs part, part + of, ...
    python esn_convex.py scale --config 17 --N 5000 --seed 1   # part 2
"""
import os, sys, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "main", "DATA", "experimental", "esn_convex")
RUNS = os.path.join(ROOT, "runs", "esn_convex")
SPLITS = dict(train=(200_000, False, 1), val=(20_000, True, 2), test=(20_000, True, 3), test_train=(20_000, False, 4))
if os.environ.get("ESN_SMOKE"):                                        # tiny sizes, separate folders
    SPLITS = {k: (v[0] // 100, v[1], v[2]) for k, v in SPLITS.items()}
    DATA, RUNS = DATA + "_smoke", RUNS + "_smoke"
T, R, C = 30, 10, 2
LAMS = 10.0 ** np.arange(-7, 1.01, 0.25)


def dataset_path():
    for d in (RUNS, DATA):
        if os.path.exists(os.path.join(d, "dataset.npz")):
            return os.path.join(d, "dataset.npz")
    raise FileNotFoundError("dataset.npz: run harvest first")


def load_data():
    d = np.load(dataset_path())
    return {s: dict(obs=d[s + "_obs"], Z=d[s + "_Z"], vals=d[s + "_vals"].astype(np.int64), goal=d[s + "_goal"].astype(np.int64),
                    fr_low=d[s + "_fr_low"]) for s in SPLITS}


def joint_post(split, chunk=20_000):
    """(B, C, R) Joint Bayes posterior of each variable at T, rebuilt from Z and the counts as in bayes_likelihood
    (observations are iid given the values; reproduces the stored Joint accuracy exactly): the soft read-out targets"""
    import torch
    Vn = 2 * R + 1
    Vr = 2 * np.sin(np.linspace(-1, 1, Vn + 1)[1:] * np.pi)
    roll = torch.as_tensor(np.stack([np.roll(Vr, r) for r in range(R)]))
    out = []
    for i in range(0, len(split["Z"]), chunk):
        Z = torch.as_tensor(split["Z"][i:i + chunk], dtype=torch.float64).reshape(-1, 5, C)
        Zs = (torch.arange(Vn, dtype=torch.float64) - Z[..., None] * Vn) % Vn
        V = torch.softmax(-torch.minimum(Zs, Vn - Zs) ** 2, -1) @ roll.T                # (B, 5, C, R)
        p = torch.sigmoid(V[:, :, 0, :, None] * V[:, :, 1, None, :])                   # (B, 5, R, R)
        n = torch.as_tensor(split["obs"][i:i + chunk], dtype=torch.float64).sum(1)[:, :, None, None]
        q = torch.softmax((n * p.log() + (T - n) * (1 - p).log()).sum(1).reshape(len(Z), R * R), -1).reshape(-1, R, R)
        out.append(torch.stack([q.sum(2), q.sum(1)], 1).float())
    return torch.cat(out)


class Ridge:
    """X^T X, X^T Y and sums for a ridge read-out of features F (B, N) onto the one-hot values of both variables"""
    def __init__(self, N):
        import torch
        self.G = torch.zeros(N, N, dtype=torch.float64)
        self.Cx = torch.zeros(N, C * R, dtype=torch.float64)
        self.sx = torch.zeros(N, dtype=torch.float64)
        self.sy = torch.zeros(C * R, dtype=torch.float64)
        self.n = 0

    def add(self, F, vals):
        import torch
        Y = torch.zeros(len(F), C * R)
        Y[torch.arange(len(F))[:, None], torch.as_tensor(vals) + torch.arange(C) * R] = 1
        self.G += (F.T @ F).double()
        self.Cx += (F.T @ Y).double()
        self.sx += F.double().sum(0)
        self.sy += Y.double().sum(0)
        self.n += len(F)

    def path(self, tests):
        """{split: (F, vals, goal)} -> {split: accuracy at each lambda}; one eigendecomposition for all lambdas"""
        import torch
        mu, yb = self.sx / self.n, self.sy / self.n
        Gc = self.G - self.n * torch.outer(mu, mu)
        Cc = self.Cx - self.n * torch.outer(mu, yb)
        s, V = torch.linalg.eigh(Gc)
        s = s.clamp(min=0)
        Bv = V.T @ Cc
        lams = LAMS * float(s.mean())
        out = {}
        for k, (F, vals, goal) in tests.items():
            A = (F.double() - mu) @ V
            acc = []
            for lam in lams:
                P = (A @ (Bv / (s + lam)[:, None]) + yb).reshape(len(F), C, R)
                pick = P[torch.arange(len(F)), torch.as_tensor(goal)].argmax(-1).numpy()
                acc.append(float((pick == vals[np.arange(len(F)), goal]).mean()))
            out[k] = acc
        return out


LOG_LAMS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def logistic_path(Ftr, vals, tests, iters=200, soft=None):
    """exact multinomial logistic read-out (convex; L-BFGS to convergence) of each variable's value from standardized
    features, L2 strength lambda (per weight, on the mean cross-entropy), warm-started along LOG_LAMS from the strongest.
    Targets: the true values (one-hot), or with soft = (B, C, R) the Joint posterior at T (still convex)"""
    import torch
    import torch.nn.functional as Fn
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    X = (Ftr - mu) / sd
    y = torch.as_tensor(vals)
    if soft is None:
        soft = torch.zeros(len(y), C, R).scatter_(2, y[..., None], 1.)
    W = torch.zeros(X.shape[1], C * R, requires_grad=True)
    b = torch.zeros(C * R, requires_grad=True)
    out = {k: [None] * len(LOG_LAMS) for k in tests}
    for j in reversed(range(len(LOG_LAMS))):
        lam = LOG_LAMS[j]
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, tolerance_grad=1e-6,
                                tolerance_change=1e-9, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            Zl = (X @ W + b).reshape(len(X), C, R)
            loss = -(soft * Fn.log_softmax(Zl, -1)).sum(-1).mean(0).sum() + lam * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            for k, (Ft, vt, goal) in tests.items():
                P = (((Ft - mu) / sd) @ W + b).reshape(len(Ft), C, R)
                pick = P[torch.arange(len(Ft)), torch.as_tensor(goal)].argmax(-1).numpy()
                out[k][j] = float((pick == vt[np.arange(len(Ft)), goal]).mean())
    return out


def report(path_out):
    """lambda chosen on val (novel environments); accuracies there and on the other splits"""
    i = int(np.argmax(path_out["val"]))
    return dict(lam_i=i, lam_rel=float(LAMS[i]), **{k: v[i] for k, v in path_out.items()}, curves=path_out)


# ---------------------------------------------------------------------------------------------------------------------
# part 1 / 2: classical ESN

def config(i):
    """configs 0-3: textbook ESN settings; 4+: random search"""
    fixed = [dict(rho=.9, alpha=1., density=.1, s_obs=1., s_z=1., s_b=.1), dict(rho=.99, alpha=.3, density=.1, s_obs=1., s_z=1., s_b=.1),
             dict(rho=.9, alpha=.5, density=1., s_obs=.5, s_z=2., s_b=.2), dict(rho=1.1, alpha=.1, density=.01, s_obs=1., s_z=5., s_b=.5)]
    if i < len(fixed):
        return fixed[i]
    r = np.random.default_rng(1000 + i)
    return dict(rho=float(r.uniform(.1, 1.5)), alpha=float(10 ** r.uniform(np.log10(.03), 0)),
                density=float(10 ** r.uniform(np.log10(.005), 0)), s_obs=float(10 ** r.uniform(-1.5, 1)),
                s_z=float(10 ** r.uniform(-1.5, 1.5)), s_b=float(10 ** r.uniform(-2, .5)))


def make_reservoir(cfg, N, seed, readin="dense"):
    """readin "dense": every unit receives all 15 inputs; "local": each unit receives one observation bit and that bit's two
    Z values only (the same weights, masked; the mask is drawn after b, so W, b and the kept weights equal the dense ones)"""
    import torch
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(N, N, generator=g)
    if cfg["density"] < 1:
        W = W * (torch.rand(N, N, generator=g) < cfg["density"])
    if N <= 2000:
        W = W * (cfg["rho"] / float(torch.linalg.eigvals(W).abs().max()))
    else:
        W = W * (cfg["rho"] / np.sqrt(N * cfg["density"]))
    Win = torch.rand(N, 15, generator=g) * 2 - 1
    Win[:, :5] *= cfg["s_obs"]
    Win[:, 5:] *= cfg["s_z"]
    b = (torch.rand(N, generator=g) * 2 - 1) * cfg["s_b"]
    if readin == "local":
        o = torch.randint(0, 5, (N,), generator=g)
        mask = torch.zeros(N, 15)
        for j in (o, 5 + C * o, 6 + C * o):
            mask[torch.arange(N), j] = 1
        Win = Win * mask
    return W, Win, b


def classical_feats(cfg, res, split, chunk):
    """yields (h_T, sum_t h_t, slice) over the split"""
    import torch
    W, Win, b = res
    Wt, a = W.T.contiguous(), cfg["alpha"]
    for i in range(0, len(split["obs"]), chunk):
        o = torch.as_tensor(split["obs"][i:i + chunk], dtype=torch.float32)
        uz = torch.as_tensor(split["Z"][i:i + chunk]) @ Win[:, 5:].T + b
        h = torch.zeros(len(o), W.shape[0])
        s = torch.zeros_like(h)
        for t in range(T):
            h = (1 - a) * h + a * torch.tanh(h @ Wt + o[:, t] @ Win[:, :5].T + uz)
            s += h
        yield h, s, slice(i, i + len(o))


def run_features(feats, data, N, chunk, logistic=True):
    """feats(split, chunk) -> iterator of (final state, summed state, slice); fits the ridge read-out of each feature set
    and, with logistic, the logistic read-out too (keys final / sum and final_log / sum_log)"""
    import torch
    fits = {"final": Ridge(N), "sum": Ridge(N)}
    keep = {"final": [], "sum": []}
    for hT, hs, sl in feats(data["train"], chunk):
        fits["final"].add(hT, data["train"]["vals"][sl])
        fits["sum"].add(hs, data["train"]["vals"][sl])
        if logistic:
            keep["final"].append(hT)
            keep["sum"].append(hs)
    tests = {"final": {}, "sum": {}}
    for k in ("val", "test", "test_train"):
        F = {"final": [], "sum": []}
        for hT, hs, _ in feats(data[k], chunk):
            F["final"].append(hT)
            F["sum"].append(hs)
        for f in F:
            tests[f][k] = (torch.cat(F[f]), data[k]["vals"], data[k]["goal"])
    out = {f: report(fits[f].path(tests[f])) for f in fits}
    if logistic:
        post = joint_post(data["train"])
        for f in fits:
            Ftr = torch.cat(keep[f])
            keep[f] = None
            out[f + "_log"] = report_log(logistic_path(Ftr, data["train"]["vals"], tests[f]))
            out[f + "_soft"] = report_log(logistic_path(Ftr, data["train"]["vals"], tests[f], soft=post))
    return out


def report_log(path_out):
    i = int(np.argmax(path_out["val"]))
    return dict(lam_i=i, lam=LOG_LAMS[i], **{k: v[i] for k, v in path_out.items()}, curves=path_out)


def run_classical(cfg, N, seed, data, chunk, readin="dense"):
    res = make_reservoir(cfg, N, seed, readin)
    return run_features(lambda split, ch: classical_feats(cfg, res, split, ch), data, N, chunk)


def sweep(part, of, n_cfg, N=1000, readin="dense"):
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads())))
    data = load_data()
    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, f"sweep_{'' if readin == 'dense' else readin + '_'}N{N}_p{part}.jsonl")
    done = {json.loads(l)["cfg_id"] for l in open(path)} if os.path.exists(path) else set()
    for i in range(part, n_cfg, of):
        if i in done:
            continue
        t0 = time.time()
        out = run_classical(config(i), N, seed=i, data=data, chunk=5000, readin=readin)
        rec = dict(cfg_id=i, N=N, seed=i, readin=readin, cfg=config(i), secs=round(time.time() - t0, 1), **out)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"cfg {i:3d} {rec['secs']:6.1f}s | ridge final {out['final']['test']:.3f} sum {out['sum']['test']:.3f} | logistic "
              f"final {out['final_log']['test']:.3f} sum {out['sum_log']['test']:.3f} | soft final {out['final_soft']['test']:.3f} "
              f"sum {out['sum_soft']['test']:.3f} | " + ", ".join(f"{k} {v:.3g}" for k, v in config(i).items()), flush=True)


def scale(cfg_id, N, seed, readin="dense"):
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads())))
    data = load_data()
    os.makedirs(RUNS, exist_ok=True)
    t0 = time.time()
    out = run_classical(config(cfg_id), N, seed=seed, data=data, chunk=max(500, min(5000, 2 * 10 ** 7 // N)), readin=readin)
    rec = dict(cfg_id=cfg_id, N=N, seed=seed, readin=readin, cfg=config(cfg_id), secs=round(time.time() - t0, 1), **out)
    with open(os.path.join(RUNS, f"scale_{'' if readin == 'dense' else readin + '_'}c{cfg_id}_N{N}_s{seed}.json"), "w") as fh:
        json.dump(rec, fh)
    print(f"cfg {cfg_id} N {N} seed {seed}: ridge final {out['final']['test']:.3f} sum {out['sum']['test']:.3f} | logistic "
          f"final {out['final_log']['test']:.3f} sum {out['sum_log']['test']:.3f} | soft final {out['final_soft']['test']:.3f} "
          f"sum {out['sum_soft']['test']:.3f} ({rec['secs']:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# part 1b / 2b (John 11-09 21:40: span the read-in gains, the spectrum and the field's standard tricks before concluding):
# still classical (every input and recurrent weight fixed and random, only the read-out trained), but also
#   read-in    dense, or channel-local (each unit: one observation bit and that bit's two Z values; fixed random weights)
#   spectrum   Gaussian with connection density p, Haar-orthogonal (every eigenvalue on the circle of radius rho), or the
#              simple cycle reservoir (a ring with weight rho; Rodan & Tino 2011); rho = 0 in 20% of settings
#   read-out   the field's augmentations: the input next to the state (W_out [1; u; x]; here the summed inputs = the
#              5 counts and T x the 10 Z values) and squared states [x, x^2]; fits on the final state, the summed state
#              and the summed [state, squared state]

def config2(i):
    """0-3: anchors (channel-local without recurrence at the diagnostics' scales; local + Gaussian; local + cycle; the dense
    textbook ESN); 4+: random search"""
    fixed = [dict(readin="local", topology="gaussian", rho=0., alpha=1., density=.1, s_obs=1.35, s_z=2.9, s_b=.62),
             dict(readin="local", topology="gaussian", rho=.9, alpha=.3, density=.1, s_obs=1.35, s_z=2.9, s_b=.62),
             dict(readin="local", topology="cycle", rho=.9, alpha=1., density=1., s_obs=1.35, s_z=2.9, s_b=.62),
             dict(readin="dense", topology="gaussian", rho=.9, alpha=1., density=.1, s_obs=1., s_z=1., s_b=.1)]
    if i < len(fixed):
        return fixed[i]
    r = np.random.default_rng(5000 + i)
    return dict(readin=str(r.choice(["dense", "local"])), topology=str(r.choice(["gaussian", "orthogonal", "cycle"])),
                rho=0. if r.random() < .2 else float(10 ** r.uniform(np.log10(.03), np.log10(1.5))),
                alpha=float(10 ** r.uniform(np.log10(.03), 0)), density=float(10 ** r.uniform(np.log10(.005), 0)),
                s_obs=float(10 ** r.uniform(-1.5, 1)), s_z=float(10 ** r.uniform(-1.5, 2)), s_b=float(10 ** r.uniform(-2, .5)))


def make_reservoir2(cfg, N, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    if cfg["topology"] == "cycle":
        W = torch.zeros(N, N)
        W[torch.arange(N), (torch.arange(N) + 1) % N] = cfg["rho"]
    elif cfg["topology"] == "orthogonal":
        Q, Rq = torch.linalg.qr(torch.randn(N, N, generator=g))
        W = cfg["rho"] * Q * torch.sign(torch.diagonal(Rq))[None, :]
    else:
        W = torch.randn(N, N, generator=g)
        if cfg["density"] < 1:
            W = W * (torch.rand(N, N, generator=g) < cfg["density"])
        if cfg["rho"] > 0:
            W = W * (cfg["rho"] / (float(torch.linalg.eigvals(W).abs().max()) if N <= 2000 else np.sqrt(N * cfg["density"])))
        else:
            W = W * 0
    Win = torch.rand(N, 15, generator=g) * 2 - 1
    Win[:, :5] *= cfg["s_obs"]
    Win[:, 5:] *= cfg["s_z"]
    b = (torch.rand(N, generator=g) * 2 - 1) * cfg["s_b"]
    if cfg["readin"] == "local":
        o = torch.randint(0, 5, (N,), generator=g)
        mask = torch.zeros(N, 15)
        for j in (o, 5 + C * o, 6 + C * o):
            mask[torch.arange(N), j] = 1
        Win = Win * mask
    return W, Win, b


def feats2(cfg, res, split, chunk, square=True):
    """yields (final [x_T; u_T], summed [x; u], summed [x; x^2; u]) per chunk: u_T = the last input (bits, Z), u = the
    summed inputs (counts, T x Z)"""
    import torch
    W, Win, b = res
    Wt, a = W.T.contiguous(), cfg["alpha"]
    for i in range(0, len(split["obs"]), chunk):
        o = torch.as_tensor(split["obs"][i:i + chunk], dtype=torch.float32)
        Z = torch.as_tensor(split["Z"][i:i + chunk])
        uz = Z @ Win[:, 5:].T + b
        h = torch.zeros(len(o), W.shape[0])
        s, s2 = torch.zeros_like(h), torch.zeros_like(h)
        for t in range(T):
            h = (1 - a) * h + a * torch.tanh(h @ Wt + o[:, t] @ Win[:, :5].T + uz)
            s += h
            if square:
                s2 += h * h
        u = torch.cat([o.sum(1), T * Z], 1)
        yield torch.cat([h, o[:, -1], Z], 1), torch.cat([s, u], 1), (torch.cat([s, s2, u], 1) if square else None)


def run2_fit(cfg, N, seed, data, chunk, square=True):
    """ridge read-outs of the feature sets and soft-target logistic read-outs of the summed ones"""
    import torch
    res = make_reservoir2(cfg, N, seed)
    keys = ("final", "sum", "sum_sq") if square else ("final", "sum")
    dims = dict(final=N + 15, sum=N + 15, sum_sq=2 * N + 15)
    fits = {k: Ridge(dims[k]) for k in keys}
    keep = {k: [] for k in keys if k != "final"}
    i0 = 0
    for parts in feats2(cfg, res, data["train"], chunk, square):
        sl = slice(i0, i0 + len(parts[0]))
        i0 += len(parts[0])
        for k, F in zip(keys, parts):
            fits[k].add(F, data["train"]["vals"][sl])
            if k in keep:
                keep[k].append(F)
    tests = {k: {} for k in keys}
    for s in ("val", "test", "test_train"):
        got = {k: [] for k in keys}
        for parts in feats2(cfg, res, data[s], chunk, square):
            for k, F in zip(keys, parts):
                got[k].append(F)
        for k in keys:
            tests[k][s] = (torch.cat(got[k]), data[s]["vals"], data[s]["goal"])
    out = {k: report(fits[k].path(tests[k])) for k in keys}
    post = joint_post(data["train"])
    for k in keep:
        Ftr = torch.cat(keep[k])
        keep[k] = None
        out[k + "_soft"] = report_log(logistic_path(Ftr, data["train"]["vals"], tests[k], soft=post))
    return out


def best2(rec):
    """the read-out with the best validation accuracy: (key, val, test)"""
    k = max((k for k in rec if isinstance(rec[k], dict) and "val" in rec[k]), key=lambda k: rec[k]["val"])
    return k, rec[k]["val"], rec[k]["test"]


def sweep2(part, of, n_cfg, N=1000):
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads())))
    data = load_data()
    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, f"sweep2_N{N}_p{part}.jsonl")
    done = {json.loads(l)["cfg_id"] for l in open(path)} if os.path.exists(path) else set()
    for i in range(part, n_cfg, of):
        if i in done:
            continue
        t0 = time.time()
        out = run2_fit(config2(i), N, seed=i, data=data, chunk=5000)
        rec = dict(cfg_id=i, N=N, seed=i, cfg=config2(i), secs=round(time.time() - t0, 1), **out)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        k, v, t = best2(rec)
        print(f"cfg {i:3d} {rec['secs']:6.0f}s | best {k} val {v:.3f} test {t:.3f} | "
              + " ".join(f"{kk} {out[kk]['test']:.3f}" for kk in out) + " | "
              + ", ".join(f"{kk} {vv:.3g}" if isinstance(vv, float) else f"{kk} {vv}" for kk, vv in config2(i).items()), flush=True)


def scale2(rank, N, seed, N_sweep=1000):
    """the rank-th best sweep2 setting (by its best read-out's validation accuracy) at size N"""
    import torch, glob
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads())))
    recs = [json.loads(l) for f in glob.glob(os.path.join(RUNS, f"sweep2_N{N_sweep}_p*.jsonl")) for l in open(f)]
    cfg_id = sorted(recs, key=lambda r: -best2(r)[1])[rank]["cfg_id"]
    data = load_data()
    square = N <= 8000                  # [x, x^2] read-outs only while they fit in memory (N 8,000: ~13 GB of train features)
    t0 = time.time()
    out = run2_fit(config2(cfg_id), N, seed=seed, data=data, chunk=max(500, min(5000, 2 * 10 ** 7 // N)), square=square)
    rec = dict(cfg_id=cfg_id, rank=rank, N=N, seed=seed, cfg=config2(cfg_id), secs=round(time.time() - t0, 1), **out)
    with open(os.path.join(RUNS, f"scale2_r{rank}_N{N}_s{seed}.json"), "w") as fh:
        json.dump(rec, fh)
    k, v, t = best2(rec)
    print(f"rank {rank} (cfg {cfg_id}) N {N} seed {seed}: best {k} val {v:.3f} test {t:.3f} | "
          + " ".join(f"{kk} {out[kk]['test']:.3f}" for kk in out) + f" ({rec['secs']:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# harvest and part 3

def paper_net(name):
    """the paper's network as a function (obs, Z) -> (states (B, T, N), belief (B, T, C, R)); its own forward pass"""
    import torch
    w = torch.load(os.path.join(ROOT, "main", "DATA", "sanity", name + "_net.pth"), weights_only=False, map_location="cpu")["weights"]
    N = w["STM"].shape[-1]
    lstm = torch.nn.LSTM(N, N, batch_first=True)
    lstm.load_state_dict({k[5:]: v.float() for k, v in w.items() if k.startswith("LSTM.")})
    Wi, bi = w["classifier_readin.weight"].float(), w["classifier_readin.bias"].float()
    Wo, bo = w["classifier_readout.weight"].float(), w["classifier_readout.bias"].float()

    def run(obs, Z):
        B = len(obs)
        x = torch.cat([obs, Z[:, None, :].expand(-1, T, -1)], -1)
        H, _ = lstm(torch.relu(x @ Wi.T + bi), (w["STM"].float().expand(-1, B, -1).contiguous(),
                                                w["LTM"].float().expand(-1, B, -1).contiguous()))
        belief = torch.softmax((H @ Wo.T + bo).cumsum(1).reshape(B, T, C, R), -1)
        return H, belief
    return run, N


def harvest():
    import torch
    from main.CognitiveGridworld import CognitiveGridworld
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    B = min(10_000, min(v[0] for v in SPLITS.values()))
    np.random.seed(0); torch.manual_seed(0)
    env = CognitiveGridworld(mode="SANITY", load_env="/sanity/reservoir_ctx_2_e5", reservoir=True, hid_dim=1000, obs_num=5,
                             learn_embeddings=False, state_num=500, ctx_num=2, realization_num=10, step_num=T, batch_num=B,
                             episodes=1, training=False, checkpoint_every=1, gpu_inference=True, show_plots=False, cuda=0)
    esn, N = paper_net("reservoir_ctx_2_e5")
    # check: our forward pass of the paper's ESN reproduces the model's own beliefs on the episode it just ran
    with torch.no_grad():
        _, mine = esn(torch.as_tensor(env.obs_flat, dtype=torch.float32), torch.as_tensor(env.joint_Z.reshape(B, -1), dtype=torch.float32))
    theirs = env.model.classifier_belief_flat.float().cpu()
    print(f"paper ESN forward check: max |belief diff| = {float((mine - theirs).abs().max()):.2e}", flush=True)
    out = {}
    for split, (D, test_set, seed) in SPLITS.items():
        np.random.seed(100 + seed)
        rec = {k: [] for k in ("obs", "Z", "vals", "goal", "fr_low", "joint_ok", "naive_ok", "ctx")}
        for _ in range(D // B):
            env.test_set = test_set
            env.run_generators()
            g = torch.as_tensor(env.goal_ind).long()
            gv = torch.as_tensor(env.goal_value).long()
            jb = env.gpu_belief(env.joint_likelihood)[:, -1].float()
            nb = env.gpu_belief(env.naive_likelihood, naive=True)[:, -1].float()
            ok, _ = env.naive_consistent_cells()
            cell = torch.as_tensor(np.ravel_multi_index(tuple(env.ctx_vals.T), (R,) * C))
            rec["obs"].append(env.obs_flat.astype(np.uint8))
            rec["Z"].append(env.joint_Z.reshape(B, -1).astype(np.float32))
            rec["vals"].append(env.ctx_vals.astype(np.int8))
            rec["goal"].append(env.goal_ind.astype(np.int8))
            rec["fr_low"].append(ok[torch.arange(B), cell].numpy())
            rec["joint_ok"].append((jb[torch.arange(B), g].argmax(-1) == gv).numpy())
            rec["naive_ok"].append((nb[torch.arange(B), g].argmax(-1) == gv).numpy())
            rec["ctx"].append(env.ctx_inds.astype(np.int16))
        for k, v in rec.items():
            out[f"{split}_{k}"] = np.concatenate(v)
        print(f"{split}: {len(out[split + '_obs'])} episodes | Joint {out[split + '_joint_ok'].mean():.3f} "
              f"Naive {out[split + '_naive_ok'].mean():.3f}", flush=True)
    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(os.path.join(DATA, "dataset.npz"), **out)
    paper()


def paper():
    """part 3: the paper's ESN and trained RNN, their own read-outs and the exact read-outs on their states"""
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    data = load_data()
    d = np.load(dataset_path())
    res = dict(bayes={s: dict(joint=float(d[s + "_joint_ok"].mean()), naive=float(d[s + "_naive_ok"].mean())) for s in SPLITS})
    for name in ("reservoir_ctx_2_e5", "fully_trained_ctx_2_e5"):
        net, N = paper_net(name)
        t0 = time.time()

        beliefs = {}                                                 # the network's own final beliefs, per split

        def feats(split, chunk, net=net):
            for i in range(0, len(split["obs"]), chunk):
                with torch.no_grad():
                    H, belief = net(torch.as_tensor(split["obs"][i:i + chunk], dtype=torch.float32),
                                    torch.as_tensor(split["Z"][i:i + chunk]))
                beliefs.setdefault(id(split), []).append(belief[:, -1])
                yield H[:, -1], H.sum(1), slice(i, i + len(H))
        exact = run_features(feats, data, N, 2000)
        own = {}
        for k in ("val", "test", "test_train"):
            bel = torch.cat(beliefs[id(data[k])])
            pick = bel[torch.arange(len(bel)), torch.as_tensor(data[k]["goal"])].argmax(-1).numpy()
            own[k] = float((pick == data[k]["vals"][np.arange(len(bel)), data[k]["goal"]]).mean())
        res[name] = dict(own=own, exact=exact, secs=round(time.time() - t0, 1))
        print(f"{name}: own read-out test {own['test']:.3f} | exact ridge read-out: final {exact['final']['test']:.3f} "
              f"sum {exact['sum']['test']:.3f} | logistic: final {exact['final_log']['test']:.3f} sum "
              f"{exact['sum_log']['test']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(DATA, "paper.json"), "w") as fh:
        json.dump(res, fh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["harvest", "paper", "sweep", "scale", "sweep2", "scale2"])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--n-cfg", type=int, default=300)
    ap.add_argument("--config", type=int, default=0)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--readin", choices=["dense", "local"], default="dense")
    ap.add_argument("--rank", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "harvest":
        harvest()
    elif a.cmd == "paper":
        paper()
    elif a.cmd == "sweep":
        sweep(a.part, a.of, a.n_cfg, readin=a.readin)
    elif a.cmd == "sweep2":
        sweep2(a.part, a.of, a.n_cfg)
    elif a.cmd == "scale2":
        scale2(a.rank, a.N, a.seed)
    else:
        scale(a.config, a.N, a.seed, readin=a.readin)
