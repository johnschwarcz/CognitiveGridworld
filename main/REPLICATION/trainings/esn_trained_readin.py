"""Classical reservoirs with a TRAINED read-in (John 11-09 22:20: "allow your classical read-in to be trained ... it is
fair to say we needed to use a trained read-in"), for the question of reservoir hyperparameters.

esn_convex.py: a classical ESN (fixed random read-in and reservoir, exact read-outs) stays far below Naive (.22-.26 at N up
to 20,000), and the paper's ESN (a frozen random LSTM) works because its read-in is trained (with a random read-in it also
scores .22). So here the read-in is trained as in the paper's ESN, and the frozen core is a classical leaky-tanh reservoir
whose spectral radius, leak rate and connection density are the knobs in question:

  r_t = relu(W_i x_t + b_i)                            trained read-in, x_t = (5 observation bits, the 10 Z values)
  h_t = (1 - a) h_{t-1} + a tanh(W h_{t-1} + r_t + b)    frozen: W Gaussian with connection density p, scaled to spectral
                                                       radius rho (exact for N <= 2,000, else the circular law); b uniform
                                                       in [-.1, .1]; trainable initial state h_0
  y_t = sum_{s <= t} (W_o h_s + b_o)                   trained read-out, cumulated over steps as in the paper's ESN;
                                                       belief = softmax of y_t per variable
Loss: the paper's SANITY loss at power 1 (symmetric KL between the goal variable's belief and the Joint posterior, every
step); Adam, lr 1e-3; fresh episodes every update from a GPU sampler of the environment (GPUEnv; it reproduces the
environment's likelihood tables exactly, see --check). Tests every TEST_EVERY updates: goal accuracy at T = 30 on fixed sets
of 20,000 novel-environment episodes (both variables held out; val for choices, test to report) and 20,000 held-out
training-environment episodes. At the end the exact soft logistic read-out (esn_convex.logistic_path, Joint posterior at T)
is refit on the trained network's summed states: what read-out regularization changes once the read-in is trained.

The environment: CognitiveGridworld(mode None) at np.random seed 0 -- its own 500 variables at the paper's settings (|S| 500,
d_E 30, two variables, T = 30); Naive and Joint are measured on the same test sets.
    python esn_trained_readin.py --task 3            # TASKS index (GPU)
Tasks 11-13 (John 11-10 08:30: "please queue the larger trained read-in ESNs"): N 4,000 and 8,000 with TF32 matmuls on an
A100 (fp32 would take ~10-20 h at N 8,000), and N 1,000 with TF32 as the precision control against task 0 (same seed).
    python esn_trained_readin.py --check             # GPUEnv against the environment
    python esn_trained_readin.py --rho .9 --alpha 1 --density .1 --N 1000 [--updates 40000] [--smoke]
"""
import os, sys, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "esn_trained_readin")
T, R, C, O = 30, 10, 2, 5
UPDATES, BATCH, TEST_EVERY, LR = 40000, 2000, 1000, 1e-3
# (rho, alpha, density, N): spectral radius at no leak, leak at rho .9, density, then size (default rho .9, a 1, p .1)
# (task 10: the paper's core, a frozen random LSTM, in this harness: what this training budget reaches with it)
# tasks 11-13 (appended 11-10): larger N with TF32 matmuls, + the TF32 control at N 1,000 (TF32_TASKS)
TASKS = ([(rho, 1., .1, 1000) for rho in (.9, 0., .5, 1.2)] + [(.9, a, .1, 1000) for a in (.3, .1)] +
         [(.9, 1., p, 1000) for p in (.01, 1.)] + [(.9, 1., .1, N) for N in (500, 2000)] + [("lstm", 1., 1., 1000)] +
         [(.9, 1., .1, N) for N in (4000, 8000, 1000)])
TF32_TASKS = (11, 12, 13)


def make_env(batch=1000):
    from main.CognitiveGridworld import CognitiveGridworld
    np.random.seed(0)
    return CognitiveGridworld(mode=None, state_num=500, ctx_num=C, realization_num=R, obs_num=O, step_num=T, KQ_dim=30,
                              batch_num=batch, learn_embeddings=False, show_plots=False)


class GPUEnv:
    """the environment's episode sampler (env_generators.default_context / bayes_likelihood / default_gen_observations) in
    torch: training pairs keep at most one held-out variable and never make it the goal; test pairs draw both from the
    held-out variables (repeats allowed); values uniform; observation bits Bernoulli(sigmoid(V_0(r_0) V_1(r_1)))"""
    def __init__(self, env, device):
        import torch
        self.dev = device
        self.K = torch.as_tensor(env.all_K, dtype=torch.float32, device=device)            # (S, O, d)
        self.Q = torch.as_tensor(env.all_Q, dtype=torch.float32, device=device)
        self.S, self.n_test = env.state_num, env.test_states
        Vn = 2 * R + 1
        Vr = env.likelihood_temp * np.sin(np.linspace(-1, 1, Vn + 1)[1:] * np.pi * env.likelihood_freq)
        self.roll = torch.as_tensor(np.stack([np.roll(Vr, r) for r in range(R)]), dtype=torch.float32, device=device)
        self.Vn = Vn

    def likelihood(self, ctx):
        """(B, O, C) Z values (joint_Z) and (B, O, R, R) likelihood of a 1 for pairs ctx (B, C)"""
        import torch
        Z = torch.stack([(self.K[ctx[:, 0]] * self.Q[ctx[:, 1]]).sum(-1), (self.K[ctx[:, 1]] * self.Q[ctx[:, 0]]).sum(-1)], -1)
        Zs = (torch.arange(self.Vn, device=self.dev, dtype=torch.float32) - Z[..., None] * self.Vn) % self.Vn
        V = torch.softmax(-torch.minimum(Zs, self.Vn - Zs) ** 2, -1) @ self.roll.T          # (B, O, C, R)
        return Z, torch.sigmoid(V[:, :, 0, :, None] * V[:, :, 1, None, :])

    def sample(self, B, test_set, gen=None):
        import torch
        kw = dict(device=self.dev, generator=gen)
        if test_set:
            ctx = torch.randint(0, self.n_test, (B, C), **kw)
        else:
            ctx = torch.randint(0, self.S, (B, C), **kw)
            both = (ctx < self.n_test).all(1)
            ctx[both, 1] = torch.randint(self.n_test, self.S, (int(both.sum()),), **kw)
        ctx = ctx.sort(1).values
        rng = torch.rand(B, C, **kw)
        if not test_set:
            rng[ctx < self.n_test] = 0
        goal = rng.argmax(1)
        vals = torch.randint(0, R, (B, C), **kw)
        Z, p = self.likelihood(ctx)
        i = torch.arange(B, device=self.dev)
        pobs = p[i, :, vals[:, 0], vals[:, 1]]                                                # (B, O)
        obs = (pobs[:, None, :] > torch.rand(B, T, O, **kw)).to(torch.uint8)
        return dict(obs=obs, Z=Z.reshape(B, -1), vals=vals, goal=goal)


def posterior(obs, Z, naive=False):
    """(B, T, C, R) Joint posterior of each variable after each step (or the Naive observer's), from Z and the counts"""
    import torch
    Vn = 2 * R + 1
    Vr = 2 * np.sin(np.linspace(-1, 1, Vn + 1)[1:] * np.pi)
    roll = torch.as_tensor(np.stack([np.roll(Vr, r) for r in range(R)]), device=Z.device, dtype=torch.float64)
    Zd = Z.double().reshape(-1, O, C)
    Zs = (torch.arange(Vn, device=Z.device, dtype=torch.float64) - Zd[..., None] * Vn) % Vn
    V = torch.softmax(-torch.minimum(Zs, Vn - Zs) ** 2, -1) @ roll.T                    # (B, O, C, R)
    p = torch.sigmoid(V[:, :, 0, :, None] * V[:, :, 1, None, :])                          # (B, O, R, R)
    n = obs.double().cumsum(1)                                                            # (B, T, O)
    t = torch.arange(1, T + 1, device=Z.device, dtype=torch.float64)[None, :, None]
    if naive:
        return torch.stack([torch.softmax(n @ pc.log() + (t - n) @ (1 - pc).log(), -1)
                            for pc in (p.mean(3), p.mean(2))], 2).float()
    pf = p.reshape(len(Z), O, R * R)
    q = torch.softmax(n @ pf.log() + (t - n) @ (1 - pf).log(), -1).reshape(len(Z), T, R, R)
    return torch.stack([q.sum(3), q.sum(2)], 2).float()


class ESN:
    def __init__(self, rho, alpha, density, N, seed, device):
        """rho = "lstm": the paper's core instead (a frozen nn.LSTM at PyTorch's default init, trainable initial states)"""
        import torch
        g = torch.Generator().manual_seed(seed)
        lin_i, lin_o = torch.nn.Linear(15, N), torch.nn.Linear(N, C * R)                 # PyTorch's default inits
        self.Wi, self.bi = lin_i.weight.detach().to(device).requires_grad_(), lin_i.bias.detach().to(device).requires_grad_()
        self.Wo, self.bo = lin_o.weight.detach().to(device).requires_grad_(), lin_o.bias.detach().to(device).requires_grad_()
        self.h0 = torch.zeros(N, device=device, requires_grad=True)
        self.params = [self.Wi, self.bi, self.Wo, self.bo, self.h0]
        self.lstm = None
        if rho == "lstm":
            self.lstm = torch.nn.LSTM(N, N, batch_first=True).to(device).requires_grad_(False)
            self.c0 = torch.zeros(N, device=device, requires_grad=True)
            self.params.append(self.c0)
            return
        W = torch.randn(N, N, generator=g)
        if density < 1:
            W = W * (torch.rand(N, N, generator=g) < density)
        if rho > 0:
            W = W * (rho / (float(torch.linalg.eigvals(W).abs().max()) if N <= 2000 else np.sqrt(N * density)))
        else:
            W = W * 0
        self.Wt = W.T.contiguous().to(device)
        self.b = ((torch.rand(N, generator=g) * 2 - 1) * .1).to(device)
        self.a = alpha

    def states(self, obs, Z):
        """(B, T, N) core states"""
        import torch
        x = torch.cat([obs.float(), Z[:, None, :].expand(-1, T, -1)], -1)
        r = torch.relu(x @ self.Wi.T + self.bi)
        if self.lstm is not None:
            return self.lstm(r, (self.h0.expand(1, len(x), -1).contiguous(), self.c0.expand(1, len(x), -1).contiguous()))[0]
        h = self.h0.expand(len(x), -1)
        H = []
        for t in range(T):
            h = (1 - self.a) * h + self.a * torch.tanh(h @ self.Wt + r[:, t] + self.b)
            H.append(h)
        return torch.stack(H, 1)

    def logits(self, H):
        return (H @ self.Wo.T + self.bo).cumsum(1).reshape(*H.shape[:2], C, R)


def check():
    """GPUEnv's likelihood tables and Z against the environment's own, on the environment's own pairs; Bayes accuracies of
    both samplers"""
    import torch
    env = make_env(5000)
    genv = GPUEnv(env, "cpu")
    for test_set in (False, True):
        env.test_set = test_set
        env.run_generators()
        Z, p = genv.likelihood(torch.as_tensor(env.ctx_inds))
        print(f"test_set {test_set}: max |Z - env| {float((Z - torch.as_tensor(env.joint_Z)).abs().max()):.1e}, max |p - env| "
              f"{float((p - torch.as_tensor(env.joint_likelihood)).abs().max()):.1e}")
        d_env = dict(obs=torch.as_tensor(env.obs_flat.astype(np.uint8)), Z=torch.as_tensor(env.joint_Z.reshape(5000, -1), dtype=torch.float32),
                     vals=torch.as_tensor(env.ctx_vals), goal=torch.as_tensor(env.goal_ind))
        d_gpu = genv.sample(20000, test_set, torch.Generator().manual_seed(1))
        for name, d in (("environment", d_env), ("GPUEnv", d_gpu)):
            i = torch.arange(len(d["obs"]))
            acc = {k: float((posterior(d["obs"], d["Z"], k == "naive")[i, -1, d["goal"]].argmax(-1) == d["vals"][i, d["goal"]]).float().mean())
                   for k in ("joint", "naive")}
            print(f"   {name:11s}: Joint {acc['joint']:.3f} Naive {acc['naive']:.3f}")
        if not test_set:
            ctx = torch.as_tensor(env.ctx_inds)
            print(f"   environment training pairs with a held-out variable {float((ctx < env.test_states).any(1).float().mean()):.3f} "
                  f"(GPUEnv expects 1 - .9^2 = .19), goal held out {float((ctx[torch.arange(5000), torch.as_tensor(env.goal_ind)] < env.test_states).float().mean()):.3f}")


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--rho", default=.9, help="spectral radius, or lstm for the paper's core")
    ap.add_argument("--alpha", type=float, default=1.)
    ap.add_argument("--density", type=float, default=.1)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=UPDATES)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tf32", action="store_true", help="TF32 matmuls (A100 and newer)")
    a = ap.parse_args()
    if a.check:
        return check()
    if a.task is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        a.task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if a.task is not None:
        a.rho, a.alpha, a.density, a.N = TASKS[a.task]
        a.tf32 = a.tf32 or a.task in TF32_TASKS
    a.rho = a.rho if a.rho == "lstm" else float(a.rho)
    import esn_convex as E
    n_test, n_fit, test_every = (2000, 4000, 20) if a.smoke else (20_000, 200_000, TEST_EVERY)
    if a.smoke:
        a.updates = 40
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.manual_seed(a.seed)
    os.makedirs(RUNS, exist_ok=True)
    name = ("lstm" if a.rho == "lstm" else f"rho{a.rho:g}_a{a.alpha:g}_p{a.density:g}") + f"_N{a.N}_s{a.seed}" + ("_tf32" if a.tf32 else "") + ("_smoke" if a.smoke else "")
    log = open(os.path.join(RUNS, name + ".jsonl"), "w")
    genv = GPUEnv(make_env(), dev)
    gen = torch.Generator(device=dev).manual_seed(1000 + a.seed)
    tests = dict(val=genv.sample(n_test, True, gen), test=genv.sample(n_test, True, gen), test_train=genv.sample(n_test, False, gen))
    bayes = {}
    for s, d in tests.items():
        i = torch.arange(n_test, device=dev)
        for kind in ("joint", "naive"):
            P = posterior(d["obs"], d["Z"], naive=kind == "naive")
            bayes[f"{kind}_{s}"] = float((P[i, -1, d["goal"]].argmax(-1) == d["vals"][i, d["goal"]]).float().mean())
    print(f"=== {name} on {torch.cuda.get_device_name() if dev == 'cuda' else 'cpu'} | Bayes on the test sets:",
          {k: round(v, 3) for k, v in bayes.items()}, flush=True)
    net = ESN(a.rho, a.alpha, a.density, a.N, a.seed, dev)
    opt = torch.optim.Adam(net.params, lr=LR)
    t0 = time.time()

    def accuracy(d):
        hits = 0.
        with torch.no_grad():
            for j in range(0, len(d["obs"]), 5000):
                L = net.logits(net.states(d["obs"][j:j + 5000], d["Z"][j:j + 5000]))
                i = torch.arange(len(L), device=dev)
                hits += float((L[i, -1, d["goal"][j:j + 5000]].argmax(-1) == d["vals"][j:j + 5000][i, d["goal"][j:j + 5000]]).sum())
        return hits / len(d["obs"])

    for u in range(a.updates + 1):
        if u % test_every == 0 or u == a.updates:
            rec = dict(update=u, secs=round(time.time() - t0, 1), **{s: accuracy(d) for s, d in tests.items()})
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"{name} update {u:6d} | novel val {rec['val']:.3f} test {rec['test']:.3f} | training envs "
                  f"{rec['test_train']:.3f} ({rec['secs']:.0f}s)", flush=True)
        if u == a.updates:
            break
        d = genv.sample(a.batch, False, gen)
        i = torch.arange(a.batch, device=dev)
        q = posterior(d["obs"], d["Z"])[i, :, d["goal"]].clamp(min=1e-6)                 # (B, T, R) Joint, goal variable
        lq = torch.nn.functional.log_softmax(net.logits(net.states(d["obs"], d["Z"]))[i, :, d["goal"]], -1)
        loss = ((q * (q.log() - lq)).sum(-1) + (lq.exp() * (lq - q.log())).sum(-1)).mean() / 2
        opt.zero_grad()
        loss.backward()
        opt.step()

    # exact read-out refit on the trained network's summed states (soft targets: the Joint posterior at T)
    with torch.no_grad():
        S, post, fit_vals = {s: [] for s in ("fit", *tests)}, [], []
        for j in range(0, n_fit, 5000):
            d = genv.sample(min(5000, n_fit - j), False, gen)
            S["fit"].append(net.states(d["obs"], d["Z"]).sum(1).cpu())
            post.append(posterior(d["obs"], d["Z"])[:, -1].cpu())
            fit_vals.append(d["vals"].cpu())
        for s, d in tests.items():
            S[s] = [net.states(d["obs"][j:j + 5000], d["Z"][j:j + 5000]).sum(1).cpu() for j in range(0, n_test, 5000)]
        S = {k: torch.cat(v) for k, v in S.items()}
    exact = E.report_log(E.logistic_path(S["fit"], torch.cat(fit_vals).numpy(),
                                         {s: (S[s], d["vals"].cpu().numpy(), d["goal"].cpu().numpy()) for s, d in tests.items()},
                                         soft=torch.cat(post)))
    out = dict(name=name, rho=a.rho, alpha=a.alpha, density=a.density, N=a.N, seed=a.seed, updates=a.updates, batch=a.batch, tf32=a.tf32,
               bayes=bayes, final={k: rec[k] for k in tests}, exact_soft={k: exact[k] for k in ("lam", "val", "test", "test_train")},
               secs=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(RUNS, name + ".json"), "w"))
    print(f"=== {name} done: own read-out test {rec['test']:.3f}; exact soft read-out on its states test {exact['test']:.3f} "
          f"({out['secs'] / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
