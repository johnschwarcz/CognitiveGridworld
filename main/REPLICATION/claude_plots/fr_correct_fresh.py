"""r(FR_T, correct_T) through training on FRESH test episodes drawn like the runs' own tests (John 11-13: "just for
comparison, can you please send the with and without of the learning figure using the same amount of data?", then,
on the probe set's replays looking worse than the logged figure, "yes please run it").

Why: fr_definition_check's fr_correct replays the 114 checkpoints on the fixed probe set, 50 test-variable pairs, on
which even Joint Bayes has r = -.03 (symmetric clipped FR) / -.06 (forward exact); over all test environments, which the
runs' own logged tests sample, Joint Bayes has -.001 / -.024 (160,000 fresh episodes, 11-13). So the probe set's
environments, not the clamp or the method, made the Trained RNN end below the logged figure's ~0.

Here: N episodes drawn as the runs' tests are (test variables 0-49 paired WITH replacement, so ~2% pair a variable with
itself; r uniform over the 100 cells; the goal either variable), the world of every seed-20261029 run (td_analysis.
world_tables), one draw (seed SEED) shared by every checkpoint. Bayes side in float64 log space: FR of the goal variable
at T three ways, fwd = KL(Joint || Naive) exact, fwd_clip = the same with both beliefs clipped to [1e-15, 1 - 1e-15],
sym_clip = (fwd_clip + its reverse) / 2 = env.SII before 11-10 (what the runs logged); the observers' own correctness.
Network side: each of the 114 checkpoints of trained / echo replayed (td_analysis's LSTM step, read-in, read-out and
cumulative logits; checked against the stored probe replays: identical correctness) -> correctness at T of the goal.
-> main/DATA/experimental/fr_definition/fr_correct_fresh.npz, keys as fr_correct.npz ({run}_raw, {run}_x, {run}_acc,
{run}_{fwd, fwd_clip, sym_clip}, bayes_*, naive_*, acc_bayes, acc_naive) plus n, n_self, seed.
RESULTS (11-13; 40,000 episodes, 842 self-paired, 1,275 pairs; the lean replay equals the stored probe replays on every
probe at ep 6,000 and 49,995 of both runs), r at T, fwd / fwd_clip / sym_clip:
    Joint Bayes  -.029 / -.028 / +.001 (accuracy .767);  Naive Bayes  -.339 / -.366 / -.432 (accuracy .447)
    Trained, mean of the last 10 checkpoints  -.032 / -.031 / -.001 (minimum -.129 / -.138 / -.142, around 880)
    ESN, the same  -.285 / -.306 / -.334
sym_clip reproduces the logged figure (Trained ~0, ESN -.33): the probe set's lower Trained values were its 50
environments. The clamp alone moves the ESN by .02 and the Trained RNN by nothing; the Trained RNN ends at Joint Bayes's
own level under every definition. Figures: fig_supp_diagnostics.figure_log with these r (scratch renders, 11-13).
usage (GPU, ~25 min for both runs next to a training on the same GPU):  CUDA_VISIBLE_DEVICES=1 python fr_correct_fresh.py"""
import os
import sys
import time
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import torch                                                                       # noqa: E402
import td_analysis as td                                                           # noqa: E402
from main.utils import factorization_regret                                       # noqa: E402

OUT = os.path.join(ROOT, "main", "DATA", "experimental", "fr_definition", "fr_correct_fresh.npz")
N, SEED, CHUNK = 40000, 20261113, 20000
RUNS = ("trained", "echo")


def episodes():
    """fresh test episodes like the runs' own: pairs (sorted), env index, r (N, 2), goal (N,), obs (N, T, 5), L, Z"""
    rng = np.random.default_rng(SEED)
    pairs = np.sort(rng.integers(td.N_TEST_VARS, size=(N, 2)), 1)
    uniq, env = np.unique(pairs, axis=0, return_inverse=True)
    ck = torch.load(td.ckpt_path(td.WORLD, 0), map_location="cpu", weights_only=False)
    L, Z = td.world_tables(ck, uniq)
    r = rng.integers(td.R, size=(N, 2))
    goal = rng.integers(2, size=N)
    p = L[env][np.arange(N), :, r[:, 0], r[:, 1]]
    obs = rng.random((N, td.T, td.NO)) < p[:, None, :]
    return dict(pairs=uniq, env=env.ravel(), r=r, goal=goal, obs=obs, L=L, Z=Z, world_K=np.asarray(ck["K"]),
                world_Q=np.asarray(ck["Q"]), n_self=int((pairs[:, 0] == pairs[:, 1]).sum()))


def bayes(E):
    """goal-variable FR at T (three ways) and the observers' correctness at T, float64 log space"""
    i, g = np.arange(N), E["goal"]
    out = {k: np.zeros(N) for k in ("fwd", "fwd_clip", "rev_clip")}
    out.update(right_bayes=np.zeros(N, bool), right_naive=np.zeros(N, bool))
    for s in range(0, N, CHUNK):
        sl = slice(s, s + CHUNK)
        Le, o = E["L"][E["env"][sl]], E["obs"][sl].astype(np.float64)
        lam = np.einsum("nti,nijk->njk", o, np.log(Le)) + np.einsum("nti,nijk->njk", 1 - o, np.log1p(-Le))
        lp = lam - td._lse(lam.reshape(len(o), -1), -1)[:, None, None]
        ljb = np.stack([td._lse(lp, 2), td._lse(lp, 1)], 1)                                    # (n, 2, R)
        Lb = np.stack([Le.mean(3), Le.mean(2)], 2)
        cl = np.einsum("nti,nikr->nkr", o, np.log(Lb)) + np.einsum("nti,nikr->nkr", 1 - o, np.log1p(-Lb))
        lnb = cl - td._lse(cl, -1)[..., None]
        j, gg = np.arange(len(o)), g[sl]
        lj, ln = ljb[j, gg], lnb[j, gg]
        out["fwd"][sl] = factorization_regret(lj, ln, logs=True)
        out["fwd_clip"][sl] = factorization_regret(np.exp(lj), np.exp(ln), clip=1e-15)
        out["rev_clip"][sl] = factorization_regret(np.exp(ln), np.exp(lj), clip=1e-15)
        out["right_bayes"][sl] = lj.argmax(-1) == E["r"][sl][j, gg]
        out["right_naive"][sl] = ln.argmax(-1) == E["r"][sl][j, gg]
    out["sym_clip"] = (out["fwd_clip"] + out["rev_clip"]) / 2
    return out


@torch.no_grad()
def correct_T(m, obs, Z, goal, r, dev):
    """the network's correctness at T on the goal variable (td_analysis.forward's step, read-out and cumulative logits)"""
    W_ih, W_hh, b = td.ga.lstm_weights(m)
    res = []
    for s in range(0, len(obs), CHUNK):
        o = torch.as_tensor(obs[s:s + CHUNK], dtype=torch.float32, device=dev)
        zin = torch.as_tensor(Z[s:s + CHUNK], device=dev).reshape(len(o), -1)
        h, c = td._start(m, len(o))
        cum = 0.
        for t in range(td.T):
            x = td._readin(m, o[:, t], zin)
            h, c, _ = td._cell(x @ W_ih.T + b + h @ W_hh.T, c)
            cum = cum + m.classifier_readout(h).double()
        lq = td.marginals(cum, m.joint_out)                                                    # (n, 2, R)
        ar = torch.arange(len(o), device=dev)
        gt = torch.as_tensor(goal[s:s + CHUNK], device=dev).long()
        rt = torch.as_tensor(r[s:s + CHUNK], device=dev).long()
        res.append((lq[ar, gt].argmax(-1) == rt[ar, gt]).cpu().numpy())
    return np.concatenate(res)


def main():
    t0 = time.time()
    dev = torch.device("cuda:0")
    E = episodes()
    Bq = bayes(E)
    defs = ("fwd", "fwd_clip", "sym_clip")
    out = dict(n=N, n_self=E["n_self"], seed=SEED, acc_bayes=Bq["right_bayes"].mean(), acc_naive=Bq["right_naive"].mean())
    for k in defs:
        out[f"bayes_{k}"] = np.corrcoef(Bq[k], Bq["right_bayes"])[0, 1]
        out[f"naive_{k}"] = np.corrcoef(Bq[k], Bq["right_naive"])[0, 1]
    print(f"{N:,} episodes ({E['n_self']} self-paired, {len(E['pairs'])} pairs); Joint r " +
          " / ".join(f"{k} {out[f'bayes_{k}']:+.3f}" for k in defs) + f"; {time.time() - t0:.0f} s", flush=True)
    P = td.probes()
    Zf = E["Z"][E["env"]]
    for run in RUNS:
        eps = td.checkpoints(run)
        ck0 = torch.load(td.ckpt_path(run, 0), map_location="cpu", weights_only=False)
        assert np.array_equal(np.asarray(ck0["K"]), E["world_K"]) and np.array_equal(np.asarray(ck0["Q"]), E["world_Q"])
        m = td.build(run, eps[0], dev)
        for ep in (eps[len(eps) // 2], eps[-1]):                       # the lean step against the stored probe replay
            td.load_weights(m, run, ep)
            mine = correct_T(m, P["obs"], P["Z"][P["env"]], P["goal"], P["r"], dev)
            st = np.load(os.path.join(td.OUT, run, f"ck_{ep:05d}.npz"))["right"][np.arange(len(P["goal"])), -1, P["goal"]]
            agree = (mine == st).mean()
            print(f"  check {run} ep {ep}: correctness equal to the stored replay on {agree:.5f} of the probes", flush=True)
            assert agree > .999, (run, ep, agree)
        acc, rr = [], {k: [] for k in defs}
        for ep in eps:
            td.load_weights(m, run, ep)
            right = correct_T(m, E["obs"], Zf, E["goal"], E["r"], dev).astype(np.float64)
            acc.append(right.mean())
            for k in defs:
                rr[k].append(np.corrcoef(Bq[k], right)[0, 1])
        out[f"{run}_raw"] = np.array(eps)
        out[f"{run}_x"] = np.array(eps) - np.array(eps) // 5
        out[f"{run}_acc"] = np.array(acc)
        for k in defs:
            out[f"{run}_{k}"] = np.array(rr[k])
        print(f"{run}: {len(eps)} checkpoints, final acc {acc[-1]:.3f}, final r " +
              " / ".join(f"{k} {rr[k][-1]:+.3f}" for k in defs) + f"; {time.time() - t0:.0f} s", flush=True)
        del m
        torch.cuda.empty_cache()
    np.savez(OUT, **out)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
