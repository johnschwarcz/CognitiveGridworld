"""Per-environment analysis of the trained RCCs (Experiment 2): does what the learned embeddings buy follow FR?

An environment is an ordered pair (goal variable, contextual variable). For each of the four RCCs that learned
(RL_2 |S|=500; RL_state_num_reps 400_2, 800_1, 800_2 -- the other three seeds of RL_state_num_reps never left
chance, train and test), it samples ENVS novel environments (both variables are testing variables, so their
embeddings never received a goal gradient) and ENVS training environments (both training variables), and runs
EPS episodes in each (T = 30). Per episode it stores the final goal beliefs of Joint Bayes, Naive Bayes, the RCC
Classifier, and the Classifier with every learned embedding swapped for another variable's (a random derangement
within the testing and within the training variables, new per batch), plus the Generator's likelihood error
E_r[KL(P_Z(o|r) || P^_Z^(o|r))] (summed over the 5 channels, averaged over the 100 joint realizations, confidence
input 1) with the learned and with the swapped embeddings.

    python rcc_pairs.py --harvest [--cuda 1]      # ~2-4 min per RCC on a shared 3090 Ti
    python rcc_pairs.py                           # figure only, from DATA/experimental/rcc_pairs/*.npz

FR of an environment = mean over its episodes of KL(B_T^Joint || B_T^Naive) at T = 30 (nats). Accuracies are
cross-fitted: FR from the even episodes of an environment, accuracies from the odd ones, so an environment's bin
is not chosen by the same noise that sets its accuracy.
"""
import os, sys, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

OUT = os.path.join(ROOT, "main", "DATA", "experimental", "rcc_pairs")
RUNS = [("RL_2", 500), ("RL_state_num_reps/400_2", 400), ("RL_state_num_reps/800_1", 800),
        ("RL_state_num_reps/800_2", 800)]
ENVS, EPS, BATCH, R = 400, 200, 5000, 10


def derangement(idx, rng):
    order = rng.permutation(idx)
    out = np.empty_like(idx)
    out[order - idx[0]] = np.roll(order, 1)
    return out


def harvest(run, S, cuda):
    import torch
    from main.CognitiveGridworld import CognitiveGridworld
    env = CognitiveGridworld(mode="SANITY", cuda=cuda, episodes=1, realization_num=R, hid_dim=1000, obs_num=5,
                             show_plots=False, batch_num=BATCH, step_num=30, state_num=S, learn_embeddings=True,
                             ctx_num=2, load_env="/" + run, training=False, checkpoint_every=1, gpu_inference=True)
    model, dev = env.model, env.device
    rng = np.random.default_rng(0)
    Ts = env.test_states
    test, train = np.arange(Ts), np.arange(Ts, S)

    def pairs(pool):
        gi = rng.integers(0, len(pool), ENVS)
        ci = rng.integers(0, len(pool) - 1, ENVS)
        ci = np.where(ci >= gi, ci + 1, ci)                                            # distinct variables, uniform
        return np.stack([pool[gi], pool[ci]], 1)

    envs = np.concatenate([pairs(test), pairs(train)])                                 # (2*ENVS, 2): goal, context
    is_test = np.r_[np.ones(ENVS, bool), np.zeros(ENVS, bool)]
    ep_env = np.repeat(np.arange(len(envs)), EPS)
    ep_env = ep_env[rng.permutation(len(ep_env))]
    nb = len(ep_env) // BATCH

    def set_context(e):
        pair = envs[e]
        order = np.argsort(pair, 1, kind="stable")
        env.ctx_inds = np.take_along_axis(pair, order, 1)
        env.goal_ind = np.argmax(order == 0, 1)

    K0, Q0 = model.all_K.data.clone(), model.all_Q.data.clone()
    conf = torch.ones(BATCH, 2, device=dev)

    def gen_error():
        """generator likelihood error averaged over the 100 joint realizations, and at the episode's own realization"""
        Lt = torch.as_tensor(env.joint_likelihood, device=dev).float().clamp(1e-6, 1 - 1e-6)
        own = torch.as_tensor(env.ctx_vals, device=dev).long()
        kl, kl_cell = torch.zeros(BATCH, device=dev), torch.zeros(BATCH, device=dev)
        for r1 in range(R):
            for r2 in range(R):
                model.get_prediction(torch.tensor([r1, r2], device=dev).expand(BATCH, 2).contiguous(), conf)
                p = model.pred_pobs.float().clamp(1e-6, 1 - 1e-6)
                l = Lt[:, :, r1, r2]
                k = (l * torch.log(l / p) + (1 - l) * torch.log((1 - l) / (1 - p))).sum(-1)
                kl += k
                hit = (own[:, 0] == r1) & (own[:, 1] == r2)
                kl_cell[hit] = k[hit]
        return (kl / R ** 2).cpu().numpy(), kl_cell.cpu().numpy()

    def fr_rate():
        """Delta(r) of Proposition 1 at each episode's realization: per-step KL of the true observation distribution to
        the Naive likelihood of the true goal value, minus its minimum over goal values (0 = Naive converges to r_g)"""
        B_, O = BATCH, env.obs_num
        L = torch.as_tensor(env.joint_likelihood, device=dev, dtype=torch.float32).reshape(B_, O, -1).clamp(1e-6, 1 - 1e-6)
        g = torch.as_tensor(env.goal_ind, device=dev).long()
        m = torch.as_tensor(env.naive_likelihood, device=dev, dtype=torch.float32)[torch.arange(B_, device=dev), :, g]
        m = m.clamp(1e-6, 1 - 1e-6)
        cell = torch.as_tensor(np.ravel_multi_index(tuple(env.ctx_vals.T), (R, R)), device=dev)
        l = L[torch.arange(B_, device=dev), :, cell]                                           # (B, O)
        kap = -(torch.einsum('bo,bog->bg', l, m.log()) + torch.einsum('bo,bog->bg', 1 - l, (1 - m).log()))
        rg = torch.as_tensor(env.goal_value, device=dev).long()
        return (kap[torch.arange(B_, device=dev), rg] - kap.min(1).values).cpu().numpy()

    keys = ["env", "joint", "naive", "net", "swap", "gen_kl", "gen_kl_swap", "gen_kl_cell", "gen_kl_cell_swap",
            "delta", "ctx_vals"]
    rec = {k: [] for k in keys}
    t0 = time.time()
    for b in range(nb):
        e = ep_env[b * BATCH:(b + 1) * BATCH]
        env.custom_context = (lambda e=e: (set_context(e), True)[1])
        env.test_set = True
        env.run_generators()
        env.joint_goal_belief = env.forward_inference(env.joint_likelihood)[2]
        jg = env.joint_goal_belief[:, -1]
        ng = env.forward_inference(env.naive_likelihood, naive=True)[2][:, -1]
        env.prep_model()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            model.all_K.data, model.all_Q.data = K0, Q0
            model.forward_compute()
            net = model.classifier_goal_belief[:, -1].float().cpu().numpy()
            gk, gkc = gen_error()
            dl = fr_rate()
            perm = torch.as_tensor(np.r_[derangement(test, rng), derangement(train, rng)], device=dev)
            model.all_K.data, model.all_Q.data = K0[perm], Q0[perm]
            model.forward_compute()
            swap = model.classifier_goal_belief[:, -1].float().cpu().numpy()
            gks, gksc = gen_error()
            model.all_K.data, model.all_Q.data = K0, Q0
        gv = env.goal_value
        for k, v in zip(keys, [e, jg, ng, net, swap, gk, gks, gkc, gksc, dl, env.ctx_vals.copy()]):
            rec[k].append(v)
        rec.setdefault("goal_value", []).append(gv.copy())
        if b == 0:
            print(f"  batch 0 done in {time.time() - t0:.1f}s; {nb} batches", flush=True)
    out = {k: np.concatenate(v) for k, v in rec.items()}
    out.update(envs=envs, is_test=is_test, S=S, run=run)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, run.replace("/", "_") + ".npz"), **out)
    print(f"{run}: {len(out['env'])} episodes in {time.time() - t0:.0f}s", flush=True)
    del env
    torch.cuda.empty_cache()


def kl_rows(p, q):
    p, q = np.clip(p, 1e-12, 1), np.clip(q, 1e-12, 1)
    return (p * (np.log(p) - np.log(q))).sum(-1)


def per_env(d):
    """cross-fitted per-environment table: FR from even episodes, everything else from odd ones"""
    E = len(d["envs"])
    fr_ep = kl_rows(d["joint"], d["naive"])
    cols = {k: (d[k].argmax(-1) == d["goal_value"]).astype(float) for k in ["joint", "naive", "net", "swap"]}
    rank = np.zeros(len(d["env"]), int)
    for e in range(E):                                       # episode rank within its environment
        idx = np.flatnonzero(d["env"] == e)
        rank[idx] = np.arange(len(idx))
    even, odd = rank % 2 == 0, rank % 2 == 1
    T = {"fr": np.bincount(d["env"][even], fr_ep[even], E) / np.bincount(d["env"][even], minlength=E)}
    n_odd = np.bincount(d["env"][odd], minlength=E)
    for k, v in cols.items():
        T[k] = np.bincount(d["env"][odd], v[odd], E) / n_odd
    for k in ["gen_kl", "gen_kl_swap"]:
        T[k] = np.bincount(d["env"], d[k], E) / np.bincount(d["env"], minlength=E)
    T["is_test"] = d["is_test"]
    return T


def plot_env():
    """per-environment version (--by-env): environments only span ~1-5 nats of FR, since ~95% of FR variance is within environments"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    tabs = []
    for run, S in RUNS:
        f = os.path.join(OUT, run.replace("/", "_") + ".npz")
        if os.path.exists(f):
            tabs.append((S, per_env(dict(np.load(f, allow_pickle=True)))))
    if not tabs:
        sys.exit("no harvest found; run with --harvest first")
    n_runs = len(tabs)

    # common FR bins: quantiles of the pooled novel environments
    fr_test = np.concatenate([T["fr"][T["is_test"]] for _, T in tabs])
    edges = np.quantile(fr_test, np.linspace(0, 1, 9))
    edges[0], edges[-1] = -np.inf, np.inf

    def binned(key, test=True, fn=None):
        """per-run bin means, then mean +- SEM over runs"""
        M = []
        for _, T in tabs:
            sel = T["is_test"] == test
            x = T["fr"][sel]
            y = (T[key] if fn is None else fn(T))[sel]
            b = np.digitize(x, edges[1:-1])
            M.append([np.nanmean(y[b == i]) for i in range(len(edges) - 1)])
        M = np.array(M)
        return M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else 0 * M[0]

    xc, _ = binned("fr")
    cols = dict(joint="#1f3b73", net="#d1495b", naive="#8d8d8d", swap="#e8a33d")
    fig, ax = plt.subplots(1, 3, figsize=(3 * 2.9 + 2.0, 2.0 + 1.25), constrained_layout=True)

    a = ax[0]
    lab_xy = dict(joint=("Joint Bayes", 3.3, .815), net=("RCC, learned embeddings", 2.2, .655),
                  naive=("Naive Bayes", 3.4, .445), swap=("RCC, embeddings swapped", 2.2, .135))
    for k in ["joint", "net", "naive", "swap"]:
        m, s = binned(k)
        a.plot(xc, m, "-o", ms=3, color=cols[k])
        a.fill_between(xc, m - s, m + s, color=cols[k], alpha=.25, lw=0)
        lab, x, y = lab_xy[k]
        a.text(x, y, lab, color=cols[k], fontsize=fs["legend"] - 1)
    a.axhline(1 / R, color="k", lw=.6, ls=":")
    a.text(5.3, 1 / R - .045, "chance", fontsize=fs["legend"] - 2, ha="right")
    a.set_ylim(0, .9)
    a.set_xlabel(r"environment FR  $\mathbb{E}[D_{KL}(B^{Joint}_T\,\|\,B^{Naive}_T)]$ (nats)")
    a.set_ylabel("accuracy at T = 30")
    a.set_title("Accuracy on novel environments\n" + r"$\langle[\arg\max_r B_{Tgr}=r_g]\rangle$", fontsize=fs["title"] - 3)

    a = ax[1]
    share = lambda T: (T["net"] - T["naive"])
    for test, ls, lab in [(True, "-", "RCC, novel"), (False, "--", "RCC, training")]:
        m, s = binned(None, test, share)
        a.plot(xc, m, ls + "o", ms=3, color=cols["net"], label=lab)
        a.fill_between(xc, m - s, m + s, color=cols["net"], alpha=.2, lw=0)
    for test, ls, lab in [(True, "-", "Joint Bayes, novel"), (False, "--", "Joint Bayes, training")]:
        m, s = binned(None, test, lambda T: T["joint"] - T["naive"])
        a.plot(xc, m, ls, color=cols["joint"], lw=1.1, label=lab)
    a.axhline(0, color="k", lw=.6, ls=":")
    a.set_xlabel("environment FR (nats)")
    a.set_ylabel("accuracy minus Naive Bayes")
    a.set_title("Gain over Naive Bayes\n" + r"$\mathrm{acc}-\mathrm{acc}^{Naive}$", fontsize=fs["title"] - 3)
    a.legend(fontsize=fs["legend"] - 2, frameon=False, loc="lower right")

    a = ax[2]
    for key, test, ls, lab, c in [("gen_kl", True, "-", "learned, novel", cols["net"]),
                                  ("gen_kl", False, "--", "learned, training", cols["net"]),
                                  ("gen_kl_swap", True, "-", "swapped, novel", cols["swap"])]:
        m, s = binned(key, test)
        a.plot(xc, m, ls + "o", ms=3, color=c, label=lab)
        a.fill_between(xc, m - s, m + s, color=c, alpha=.2, lw=0)
    a.set_yscale("log")
    a.set_ylim(.01, 10)
    a.set_xlabel("environment FR (nats)")
    a.set_ylabel("nats per observation")
    a.set_title("Generator's likelihood error\n" + r"$\mathbb{E}_r D_{KL}(P_Z(o|r)\,\|\,\hat P_{\hat Z}(o|r))$",
                fontsize=fs["title"] - 3)
    a.legend(fontsize=fs["legend"] - 2, frameon=False, loc="center right")

    n_ep = EPS
    fig.suptitle(f"{n_runs} trained RCCs ($|\\mathcal{{S}}|$ = " + ", ".join(str(S) for S, _ in tabs) +
                 f") $\\times$ {ENVS} novel + {ENVS} training environments $\\times$ {n_ep} episodes;  "
                 "bins = octiles of novel-environment FR, mean $\\pm$ SEM over RCCs", fontsize=9)
    out = fig_path("rcc_pairs_fr_env.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)

    # numbers for the caption
    for S, T in tabs:
        t = T["is_test"]
        r = np.corrcoef(T["fr"][t], (T["net"] - T["naive"])[t])[0, 1]
        rj = np.corrcoef(T["fr"][t], (T["joint"] - T["naive"])[t])[0, 1]
        print(f"|S|={S}: novel acc net {T['net'][t].mean():.3f} joint {T['joint'][t].mean():.3f} "
              f"naive {T['naive'][t].mean():.3f} swap {T['swap'][t].mean():.3f} | r(FR, net-naive) {r:.2f} "
              f"r(FR, joint-naive) {rj:.2f} | gen KL novel {T['gen_kl'][t].mean():.4f} training "
              f"{T['gen_kl'][~t].mean():.4f} swapped {T['gen_kl_swap'][t].mean():.4f}")


def plot():
    """per-trial version: x = Delta(r), the exact rate at which FR grows at the trial's environment and realization
    (Proposition 1; 0 = the Naive observer converges to the true goal value). Exact, so no selection on noise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    data = []
    for run, S in RUNS:
        f = os.path.join(OUT, run.replace("/", "_") + ".npz")
        if os.path.exists(f):
            d = dict(np.load(f, allow_pickle=True))
            if "delta" in d:
                d["test"] = d["is_test"][d["env"]]
                for k in ["joint", "naive", "net", "swap"]:
                    d["acc_" + k] = (d[k].argmax(-1) == d["goal_value"]).astype(float)
                data.append((S, d))
    if not data:
        sys.exit("no per-trial harvest found; run with --harvest first")
    zero = 1e-6
    pos = np.concatenate([d["delta"][d["test"] & (d["delta"] > zero)] for _, d in data])
    q = np.quantile(pos, [.2, .4, .6, .8])
    edges = np.r_[-np.inf, zero, q, np.inf]
    labels = ["0", f"$<${q[0]:.2g}"] + [f"{a:.2g}-{b:.2g}" for a, b in zip(q[:-1], q[1:])] + [f"$>${q[-1]:.2g}"]
    x = np.arange(len(labels))
    p0 = np.mean([np.mean(d["delta"][d["test"]] <= zero) for _, d in data])

    def binned(key, test=True):
        M = []
        for _, d in data:
            sel = d["test"] == test
            b = np.digitize(d["delta"][sel], edges[1:-1])
            y = d[key][sel] if not callable(key) else key(d)[sel]
            M.append([y[b == i].mean() for i in range(len(labels))])
        M = np.array(M)
        return M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))

    cols = dict(joint="#1f3b73", net="#d1495b", naive="#8d8d8d", swap="#e8a33d")
    fig, ax = plt.subplots(1, 3, figsize=(3 * 2.9 + 2.0, 2.0 + 1.45), constrained_layout=True)
    xl = r"FR growth rate $\Delta(\mathbf{r})$ of the trial (nats per step)"

    a = ax[0]
    lab_xy = dict(acc_joint=(2.4, .86), acc_net=(2.4, .58), acc_naive=(2.4, .24), acc_swap=(3.25, .125))
    for k, lab in [("acc_joint", "Joint Bayes"), ("acc_net", "RCC, learned embeddings"),
                   ("acc_naive", "Naive Bayes"), ("acc_swap", "RCC, embeddings swapped")]:
        m, s_ = binned(k)
        c = cols[k[4:]]
        a.plot(x, m, "-o", ms=3, color=c)
        a.fill_between(x, m - s_, m + s_, color=c, alpha=.25, lw=0)
        a.text(*lab_xy[k], lab, color=c, fontsize=fs["legend"] - 1)
    a.axhline(1 / R, color="k", lw=.6, ls=":")
    a.set_ylim(0, .95)
    a.set_ylabel("accuracy at T = 30")
    a.set_title("Accuracy on novel environments\n" + r"$\langle[\arg\max_r B_{Tgr}=r_g]\rangle$", fontsize=fs["title"] - 3)

    a = ax[1]
    for test, ls, lab in [(True, "-", "RCC, novel"), (False, "--", "RCC, training")]:
        m, s_ = binned(lambda d: d["acc_net"] - d["acc_naive"], test)
        a.plot(x, m, ls + "o", ms=3, color=cols["net"], label=lab)
        a.fill_between(x, m - s_, m + s_, color=cols["net"], alpha=.2, lw=0)
    for test, ls, lab in [(True, "-", "Joint Bayes, novel"), (False, "--", "Joint Bayes, training")]:
        m, _ = binned(lambda d: d["acc_joint"] - d["acc_naive"], test)
        a.plot(x, m, ls, color=cols["joint"], lw=1.1, label=lab)
    a.axhline(0, color="k", lw=.6, ls=":")
    a.set_ylabel("accuracy minus Naive Bayes")
    a.set_title("Gain over Naive Bayes\n" + r"$\mathrm{acc}-\mathrm{acc}^{Naive}$", fontsize=fs["title"] - 3)
    a.legend(fontsize=fs["legend"] - 2, frameon=False, loc="upper left")

    a = ax[2]
    for key, test, ls, lab, c in [("gen_kl_cell", True, "-", "learned, novel", cols["net"]),
                                  ("gen_kl_cell", False, "--", "learned, training", cols["net"]),
                                  ("gen_kl_cell_swap", True, "-", "swapped, novel", cols["swap"])]:
        m, s_ = binned(key, test)
        a.plot(x, m, ls + "o", ms=3, color=c, label=lab)
        a.fill_between(x, m - s_, m + s_, color=c, alpha=.2, lw=0)
    a.set_yscale("log")
    a.set_ylim(.005, 20)
    a.set_ylabel("nats per observation")
    a.set_title("Generator's likelihood error at the trial\n" + r"$D_{KL}(P_Z(o|\mathbf{r})\,\|\,\hat P_{\hat Z}(o|\mathbf{r}))$",
                fontsize=fs["title"] - 3)
    a.legend(fontsize=fs["legend"] - 2, frameon=False, loc="center right")

    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=35, ha="right", fontsize=fs["tick"] - .5)
        a.set_xlabel(xl)
        a.axvspan(-.4, .4, color="#8d8d8d", alpha=.08, lw=0)
    ax[0].text(0, .56, f"Naive right\nasymptotically\n({p0:.0%} of trials)", ha="center", va="top", fontsize=fs["legend"] - 2.5)
    n_ep = len(data[0][1]["env"])
    fig.suptitle(f"{len(data)} trained RCCs ($|\\mathcal{{S}}|$ = " + ", ".join(str(S) for S, _ in data) +
                 f") $\\times$ {ENVS} novel + {ENVS} training environments $\\times$ {EPS} episodes; "
                 "bins: $\\Delta = 0$, then quintiles of $\\Delta > 0$; mean $\\pm$ SEM over RCCs", fontsize=9)
    out = fig_path("rcc_pairs_fr.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)
    for S, d in data:
        t = d["test"]
        z = t & (d["delta"] <= zero)
        nz = t & (d["delta"] > zero)
        print(f"|S|={S}: novel trials Delta=0 ({z.sum() / t.sum():.2f}): joint {d['acc_joint'][z].mean():.3f} "
              f"naive {d['acc_naive'][z].mean():.3f} net {d['acc_net'][z].mean():.3f} | Delta>0: joint "
              f"{d['acc_joint'][nz].mean():.3f} naive {d['acc_naive'][nz].mean():.3f} net {d['acc_net'][nz].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--cuda", type=int, default=1)
    ap.add_argument("--only", type=int, default=None, help="harvest just RUNS[i]")
    ap.add_argument("--quick", action="store_true", help="timing test: 25+25 environments x 200 episodes")
    ap.add_argument("--by-env", action="store_true", help="also draw the per-environment version")
    a = ap.parse_args()
    if a.quick:
        ENVS, OUT = 25, OUT + "_quick"
    if a.harvest:
        for i, (run, S) in enumerate(RUNS):
            if a.only is None or a.only == i:
                harvest(run, S, a.cuda)
    plot()
    if a.by_env:
        plot_env()
