"""Order dependence of the agents' beliefs (27 Oct 2026; John: a flat state space (plane, torus) and a curved one
(sphere) differ in whether the order of the moves matters) -> main/DATA/experimental/fig1c_compression/figdata/
round7_jumps/order_<net>.npz

The Bayes observers are order-invariant by construction: both multiply per-step likelihoods (the Naive's table is L
averaged over the other latent, fixed), so the log posterior after t steps is a sum over the steps; checked here on the
same permutations. The networks are replayed (as fig1c_lesion_test, no lesion; the original order reproduces the scan's
beliefs) on the same observations in the original order and in other orders: P = 3 random permutations of all 30 steps
(compared at T = 30) and P10 = 2 permutations of the first 10 steps only (compared at t = 10); one permutation per
stream and order (seed 7000 + e). Streams: 10 scan test environments (the lesion test's, ENVS[0::4]) x 100 cells x the
first 10 of each cell's 50 streams (1000 streams, one batch).
Saved per environment: the posterior means x = (E[r_g], E[r_c]) of every agent and order at t = 10 and T = 30, the true
cells. Measures (cells of the grid): the order spread |x^perm - x^orig| beside the distance |x^orig - x^Optimal| and
|x^orig - true cell|.
Run from the repository root (about 3 min per network on 6 CPU threads):
    CUDA_VISIBLE_DEVICES="" /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/fig1c_order_check.py trained echo"""
import os
import sys
import time
import numpy as np
import torch

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.chdir(ROOT)
torch.set_num_threads(6)
import fig1 as cf                                                          # noqa: E402

ga = cf.ga
SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
OUT = os.path.join(SCAN, "figdata", "round7_jumps")
M, R, T, H, NS, P, P10 = 50, 10, 30, 1000, 10, 3, 2
ENVS = [e for e in range(40) if e != 2]
TEST = ENVS[0::4]
AGENT = {"trained": "Trained", "echo": "Echo"}


def bayes(L, g, obs):
    """the Optimal's and the Naive's goal-ordered beliefs (n, T, 20) for observations obs (n, T, 5); L (5, R, R) raw"""
    o = obs.astype(np.float64)
    step = lambda tab: np.einsum("nto,o...->nt...", o, np.log(tab)) + np.einsum("nto,o...->nt...", 1 - o, np.log1p(-tab))
    lj = step(L).cumsum(1)                                                        # (n, T, R, R) log likelihood
    pj = np.exp(lj - lj.max((2, 3), keepdims=True))
    pj /= pj.sum((2, 3), keepdims=True)
    joint = [pj.sum(3), pj.sum(2)]                                                # raw context 0, 1 marginals
    naive = []
    for tab in (L.mean(2), L.mean(1)):                                            # context 0: averaged over r1
        ln = step(tab).cumsum(1)
        pn = np.exp(ln - ln.max(-1, keepdims=True))
        naive.append(pn / pn.sum(-1, keepdims=True))
    order = (0, 1) if g == 0 else (1, 0)
    return {a: np.concatenate([b[i] for i in order], -1) for a, b in (("Joint", joint), ("Naive", naive))}


def build_model(net):
    from main.CognitiveGridworld import CognitiveGridworld
    load_env, reservoir = ga.AGENTS[net]
    agent = CognitiveGridworld(**dict(ga.CFG, episodes=1), reservoir=reservoir, load_env=load_env,
                               external_logger=lambda env: {})
    assert agent.model.batch_num == R * R * NS and agent.model.step_num == T
    return agent, agent.model


def replay(m, W, obs, inp, g, dtype):
    """the network's goal-ordered beliefs (n, T, 20) for observations obs (n, T, 5)"""
    W_ih, W_hh, b, Wr, br = W
    n = len(obs)
    m.obs_flat = torch.as_tensor(obs.astype(dtype))
    m.active_Z, m.goal_ind, m.joint_goal_belief = (torch.as_tensor(np.repeat(v, n, 0)) for v in inp)
    with torch.no_grad(), torch.autocast("cpu", enabled=False):
        xa = ga.lstm_input(m) @ W_ih.T + b
        u = m.STM.detach().float().reshape(1, H).expand(n, H)
        c = m.LTM.detach().float().reshape(1, H).expand(n, H)
        rs = []
        for s in range(T):
            i, f_, g_, o = (xa[:, s] + u @ W_hh.T).chunk(4, -1)
            c = f_.sigmoid() * c + i.sigmoid() * g_.tanh()
            u = o.sigmoid() * c.tanh()
            rs.append(u @ Wr.T + br)
        q = torch.stack(rs, 1).double().cumsum(1).reshape(n, T, 2, R).softmax(-1)
        return torch.stack([q[:, :, g], q[:, :, 1 - g]], 2).reshape(n, T, 2 * R).numpy()


def main(net):
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    eps, goal, L_all = meta["eps"].astype(int), meta["goal"].astype(int), meta["L"]
    rows = (np.arange(R * R)[:, None] * M + np.arange(NS)[None]).ravel()        # the first NS streams of each cell
    truth = meta["cells"][meta["cell_of"][rows]].astype(float)                    # (n, 2) goal-ordered (r_g, r_c)
    with np.load(ga.CACHE.format("trained")) as f:
        inp_all = [f[k][eps] for k in cf.REPLAY[1:]]
        dtype = f["ep_obs_flat"].dtype
    Bscan = np.load(os.path.join(SCAN, f"scan_{AGENT[net]}.npz"))["beliefs"]
    agent, m = build_model(net)
    W = (*ga.lstm_weights(m), m.classifier_readout.weight.detach().float(), m.classifier_readout.bias.detach().float())
    Rv = np.arange(R, dtype=np.float64)
    mean = lambda B: np.stack([B[..., :R] @ Rv, B[..., R:] @ Rv], -1)
    out = {"envs": np.array(TEST), "truth": truth}
    for e in TEST:
        obs0 = meta["obs"][e, rows]
        rng = np.random.default_rng(7000 + e)
        orders = [np.tile(np.arange(T), (len(rows), 1))]
        orders += [np.argsort(rng.random((len(rows), T)), 1) for _ in range(P)]
        orders += [np.concatenate([np.argsort(rng.random((len(rows), 10)), 1), np.tile(np.arange(10, T), (len(rows), 1))],
                                  1) for _ in range(P10)]
        inp = [v[e:e + 1] for v in inp_all]
        X = {a: [] for a in (net, "Joint", "Naive")}
        for k, perm in enumerate(orders):
            obs = np.take_along_axis(obs0, perm[..., None], 1)
            B = replay(m, W, obs, inp, int(goal[e]), dtype)
            if k == 0:
                chk = np.abs(B - Bscan[e, rows].astype(np.float64)).max()
            X[net].append(mean(B)[:, [9, T - 1]])
            for a, Bb in bayes(L_all[e], int(goal[e]), obs).items():
                X[a].append(mean(Bb)[:, [9, T - 1]])
        for a in X:
            out[f"x_{a}_e{e:02d}"] = np.stack(X[a])                               # (1 + P + P10, n, 2 (t = 10, T), 2)
        print(f"{net} env {e}: replay check {chk:.1e}; {time.time() - t0:.0f} s", flush=True)
    np.savez(os.path.join(OUT, f"order_{net}.npz"), **out)
    return out


def report(net, out):
    rows = []
    for a in (net, "Joint", "Naive"):
        X = np.concatenate([out[f"x_{a}_e{e:02d}"] for e in out["envs"]], 1)       # (orders, N, 2, 2)
        Xo = np.concatenate([out[f"x_Joint_e{e:02d}"] for e in out["envs"]], 1)
        tr = np.tile(out["truth"], (len(out["envs"]), 1))
        full = np.linalg.norm(X[1:1 + P, :, 1] - X[0, :, 1], axis=-1)             # (P, N) at T
        f10 = np.linalg.norm(X[1 + P:, :, 0] - X[0, :, 0], axis=-1)               # (P10, N) at t = 10
        rows.append(f"{a:8s} order spread T=30: mean {full.mean():.3f}, median {np.median(full):.3f}, 90% "
                    f"{np.percentile(full, 90):.3f} | t=10 (first 10 permuted): mean {f10.mean():.3f}, 90% "
                    f"{np.percentile(f10, 90):.3f} | distance to the Optimal T=30 {np.linalg.norm(X[0, :, 1] - Xo[0, :, 1], axis=-1).mean():.3f}"
                    f", t=10 {np.linalg.norm(X[0, :, 0] - Xo[0, :, 0], axis=-1).mean():.3f} | to the true cell T=30 "
                    f"{np.linalg.norm(X[0, :, 1] - tr, axis=-1).mean():.3f}")
    print("\n".join(rows))


if __name__ == "__main__":
    for net in sys.argv[1:] or ("trained", "echo"):
        report(net, main(net))
