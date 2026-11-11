"""Within-step interaction across observation dimensions (lag 0), the companion of td_tt_mixed_diff's observation x
past interaction (lags 1-12). John 11-09: "we want to be able to capture entanglement through time across dimensions
of the observation space". The same mixed second difference with both flipped bits in the SAME observation:
    MD0 = inc_t(o_t flipped at i and j) - inc_t(o_t flipped at i) - inc_t(o_t flipped at j) + inc_t(o_t),   i != j
per variable and value, the networks stepped from their intact state at t - 1. MD0 is the part of bit i's effect on
the update that depends on bit j of the same observation. Naive Bayes: 0 (its increment is a sum over bits).
Memory-free Bayes: non-zero (log Sigma_c prod_i p(o_i | r_v, c) couples the bits through c) and the same at every step.
Bayes: non-zero; the past enters through p(c | r_v, o_<t).
Lags 1-12 (td_tt_mixed_diff) are zero for memory-free Bayes and naive. So lag 0 is the within-step entanglement and
lags >= 1 are the entanglement through time, both across the observation's 5 dimensions.
Same probes (1,000 = 50 test environments x 20), steps TS = 5, 10, ..., 30, checkpoints RAWS and statistics as
td_tt_mixed_diff (pooled mean |MD| per value; kappa = Sigma MD MD_ref / Sigma MD_ref^2 against Bayes and against
memory-free Bayes). Bit pairs: the 10 unordered pairs i < j (MD0 is symmetric in i, j).
-> main/DATA/experimental/training_dynamics/through_training/mixed_diff/ck0_<run>_<raw>.npz, obs0.npz
   ck0: s_abs, s_nb, s_nn, s_ns (n, 6 steps, 2 variables): per probe, step and variable the sums over the 10 bit pairs
        and 10 values of |MD0|, MD0 MD0_Bayes, MD0^2, MD0 MD0_memory-free
   obs0: the observers' MD0 sums (bayes / stepwise / naive: s_abs, s_bb, s_ss, s_bs) on the same probes
Run from the repository root:
    /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_mixed_diff0.py harvest | numbers"""
import os
import sys
import time
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_mixed_diff as md                                                                 # noqa: E402

T, R, NV, NO, TS, OUT = md.T, md.R, md.NV, md.NO, md.TS, md.OUT
BITS = tuple((i, j) for i in range(NO) for j in range(i + 1, NO))                  # 10 unordered pairs


def flip2(o, i, j):
    return md.flip(md.flip(o, i), j)


def observer_md0(kind, S):
    """MD0 (6 steps, 10 bit pairs, n, 2, R) of an exact observer"""
    obs = S["obs"]
    n = len(obs)
    LLs = np.stack([md.ll_step(S, obs[:, k]) for k in range(T)], 1)
    cum = np.concatenate([np.zeros((n, 1, R, R)), np.cumsum(LLs, 1)], 1)
    out = np.zeros((len(TS), len(BITS), n, NV, R))
    for a, t in enumerate(TS):
        prev, o = cum[:, t - 1], obs[:, t - 1]
        base = md.inc_obs(kind, S, prev, o)
        one = [md.inc_obs(kind, S, prev, md.flip(o, i)) for i in range(NO)]
        for b, (i, j) in enumerate(BITS):
            out[a, b] = md.inc_obs(kind, S, prev, flip2(o, i, j)) - one[i] - one[j] + base
    return out


@torch.no_grad()
def net_md0(cell, envt, code, U, C):
    """MD0 (6 steps, 10 bit pairs, n, 2, R) of a network stepped from its intact state at t - 1"""
    n = len(envt)
    out = np.zeros((len(TS), len(BITS), n, NV, R))
    for a, t in enumerate(TS):
        Rc = cell.rec(U[:, t - 1])
        c0 = C[:, t - 1]

        def inc(cd):
            h1, _ = cell.act(cell.table[envt, cd] + Rc, c0)
            return cell.inc(h1).double().numpy()
        cd = code[:, t - 1]
        base = inc(cd)
        one = [inc(cd ^ (1 << i)) for i in range(NO)]
        for b, (i, j) in enumerate(BITS):
            out[a, b] = inc(cd ^ (1 << i) ^ (1 << j)) - one[i] - one[j] + base
    return out


def obs_file():
    return os.path.join(OUT, "obs0.npz")


def ck0_file(run, raw):
    return os.path.join(OUT, f"ck0_{run}_{int(raw):05d}.npz")


def refs(S):
    B, W = observer_md0("bayes", S), observer_md0("stepwise", S)
    if not os.path.exists(obs_file()):
        Nv = observer_md0("naive", S)
        np.savez(obs_file(), rows=S["rows"], env=S["env"], ts=np.array(TS), bits=np.array(BITS),
                 **{f"{k}_s_abs": np.abs(x).sum((1, 4)).transpose(1, 0, 2) for k, x in
                    (("bayes", B), ("stepwise", W), ("naive", Nv))},
                 bayes_s_bb=(B ** 2).sum((1, 4)).transpose(1, 0, 2), stepwise_s_ss=(W ** 2).sum((1, 4)).transpose(1, 0, 2),
                 bayes_s_bs=(B * W).sum((1, 4)).transpose(1, 0, 2), naive_max=np.abs(Nv).max())
    return B, W


def harvest(runs=md.RUNS, raws=md.RAWS):
    torch.set_num_threads(md.THREADS)
    S = md.side()
    t0 = time.time()
    B, W = refs(S)
    z = np.load(obs_file())
    print(f"observers: {time.time() - t0:.0f} s; mean |MD0| Bayes {np.abs(B).mean():.4f}, memory-free {np.abs(W).mean():.4f}, "
          f"naive max {float(z['naive_max']):.1e}", flush=True)
    envt, code = md.codes(S["rows"])
    for raw in raws:
        for run in runs:
            f = ck0_file(run, raw)
            if os.path.exists(f):
                continue
            t1 = time.time()
            cell = md.LSTMCell(run, raw)
            with torch.no_grad():
                U, C = cell.replay(envt, code)
            M = net_md0(cell, envt, code, U, C)
            np.savez(f, run=run, raw=raw, x=gap.train_x(raw), rows=S["rows"], secs=time.time() - t1,
                     s_abs=np.abs(M).sum((1, 4)).transpose(1, 0, 2), s_nb=(M * B).sum((1, 4)).transpose(1, 0, 2),
                     s_nn=(M ** 2).sum((1, 4)).transpose(1, 0, 2), s_ns=(M * W).sum((1, 4)).transpose(1, 0, 2))
            print(f"{run:7s} raw {raw:5d} (x {gap.train_x(raw):6.0f}): mean |MD0| {np.abs(M).mean():.5f}  kappa_B "
                  f"{(M * B).sum() / (B ** 2).sum():+.4f}  kappa_mf {(M * W).sum() / (W ** 2).sum():+.4f}  "
                  f"{time.time() - t1:.0f} s", flush=True)


def summary0(runs=md.RUNS):
    """-> mixed_diff/summary0.npz: lag-0 mean |MD0| per value over t = 15-30 (md.T_BAL steps, comparable with
    summary.npz's md_lag_*) and over all six steps, kappa against Bayes and against memory-free Bayes (t = 15-30), each
    with 5-95% over the shared environment resamples; Bayes's and memory-free Bayes's lag-0 values on the same steps"""
    z = np.load(obs_file())
    env = z["env"]
    Cb = gap.boot_counts()
    bal = np.isin(np.array(TS), (15, 20, 25, 30))
    n_terms = len(BITS) * R

    def ratio(num, den, sel):
        """sum(num[:, sel]) / sum(den[:, sel]) pooled, and over the environment resamples"""
        a = np.bincount(env, num[:, sel].sum((1, 2)), gap.N_ENV)
        b = np.bincount(env, den[:, sel].sum((1, 2)) if den is not None else np.full(len(env), sel.sum() * NV * n_terms),
                        gap.N_ENV)
        full, boot = a.sum() / b.sum(), (Cb @ a) / (Cb @ b)
        return full, *np.percentile(boot, [5, 95])

    out = {}
    for k, nm in (("bayes", "bayes"), ("stepwise", "memfree")):
        for sel, tag in ((bal, ""), (np.ones(len(TS), bool), "_allt")):
            out[f"md0_{nm}{tag}"] = np.array(ratio(z[f"{k}_s_abs"], None, sel))
    for run in runs:
        rows = []
        for raw in md.RAWS:
            f = ck0_file(run, raw)
            if not os.path.exists(f):
                continue
            c = np.load(f)
            rows.append([float(c["x"]), *ratio(c["s_abs"], None, bal), *ratio(c["s_abs"], None, np.ones(len(TS), bool)),
                         *ratio(c["s_nb"], z["bayes_s_bb"], bal), *ratio(c["s_ns"], z["stepwise_s_ss"], bal)])
        rows = np.array(rows)
        out[f"x_{run}"] = rows[:, 0]
        for j, nm in enumerate(("md0", "md0_allt", "kappaB0", "kappaMF0")):
            out[f"{nm}_{run}"], out[f"{nm}_{run}_lo"], out[f"{nm}_{run}_hi"] = rows[:, 1 + 3 * j], rows[:, 2 + 3 * j], rows[:, 3 + 3 * j]
    np.savez(os.path.join(OUT, "summary0.npz"), ts=np.array(TS), t_bal=np.array(TS)[bal], bits=np.array(BITS), **out)
    print("saved", os.path.join(OUT, "summary0.npz"))
    for k in ("md0_bayes", "md0_memfree", "md0_bayes_allt", "md0_memfree_allt"):
        print(f"  {k}: {out[k][0]:.4f} [{out[k][1]:.4f}, {out[k][2]:.4f}]")
    for run in runs:
        x = out[f"x_{run}"]
        for i in [j for j, v in enumerate(x) if v in (40, 200, 480, 880, 1040, 2400, 8000, 39996)]:
            print(f"  {run:7s} {x[i]:6.0f}: md0 {out[f'md0_{run}'][i]:.4f} [{out[f'md0_{run}_lo'][i]:.4f}, "
                  f"{out[f'md0_{run}_hi'][i]:.4f}]  all t {out[f'md0_allt_{run}'][i]:.4f}  kappa_B "
                  f"{out[f'kappaB0_{run}'][i]:+.3f}  kappa_MF {out[f'kappaMF0_{run}'][i]:+.3f}")


def numbers():
    z = np.load(obs_file())
    n_terms = len(BITS) * R
    b = z["bayes_s_abs"].sum() / (z["bayes_s_abs"].size * n_terms)
    w = z["stepwise_s_abs"].sum() / (z["stepwise_s_abs"].size * n_terms)
    print(f"mean |MD0| per value: Bayes {b:.4f}, memory-free Bayes {w:.4f}, naive max {float(z['naive_max']):.1e}")
    for run in md.RUNS:
        for raw in md.RAWS:
            f = ck0_file(run, raw)
            if not os.path.exists(f):
                continue
            c = np.load(f)
            print(f"{run:7s} {float(c['x']):7.0f}: |MD0| {c['s_abs'].sum() / (c['s_abs'].size * n_terms):.4f}  "
                  f"kappa_B {c['s_nb'].sum() / z['bayes_s_bb'].sum():+.3f}  kappa_mf {c['s_ns'].sum() / z['stepwise_s_ss'].sum():+.3f}")


if __name__ == "__main__":
    {"harvest": harvest, "numbers": numbers, "summary": summary0}[sys.argv[1]]()
