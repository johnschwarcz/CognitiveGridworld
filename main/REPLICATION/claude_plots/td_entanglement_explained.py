"""Variance of the network state explained by entanglement and by factorization regret (FR), through training (John
11-01: "a figure that puts them on the same panel, along with var explained by Factorization Regret and entanglement").
-> data main/DATA/experimental/training_dynamics/analysis/ent_explained_<run>.npz (every checkpoint; GPU 0, ~1 s each)

u_it = the LSTM output (1,000 neurons) of probe episode i at step t (td_analysis's 10,000 probes), steps 2-30.
x_it = entanglement (rev, both variables averaged) or FR (factorization regret, KL(Bayes || naive) of a variable's
marginal belief, the paper's forward form, exact: probes.npz 'fr' = main.utils.factorization_regret; both variables
averaged), per particle-step. Until 11-10 'fr' was the symmetric KL clipped at 1e-15, and the caches of 11-01
(ent_explained_<run>.npz without the field fr_def) hold THAT FR's curve: they were not rebuilt on 11-10 (a rebuild
replays every checkpoint: ~1 s each on a GPU, 54 s on the CPU at 3 threads = ~1.7 h per run). Check at the last
Trained-RNN checkpoint (CPU replay, 11-10): explained by FR, 5 / 20 bins, .1943 / .2304 forward against .1918 / .2202
symmetric, of a total of 16.40; entanglement's .1369 / .1669 reproduced. Before rebuilding, rename the old cache to
ent_explained_<run>_sym_superseded.npz.
  total variance       V = < || u_it - m ||^2 >  over all particle-steps (m: their mean)
  explained by x       VE_x = the variance of the state's means across the bins of x, within each step, averaged over
                       steps:  VE_x = (1/29) sum_t (1/K) sum_k || m_kt - m_t ||^2,  m_kt = the mean state of the
                       2,000 (K = 5) particles in bin k of step t (bins: x ranked within its step), m_t = the step's mean
                       state; minus the bins' sampling noise, (K - 1)/K * the average over (t, k) of the within-bin
                       variance / 2,000 (its expected contribution)
Stored per checkpoint: "tot" V, "ve" (variable, K) for K = 5 and 20 bins, "ve_raw" (uncorrected), "within" the
average within-step variance (V minus the variance of the step means).
Run from the repository root (GPU 0):
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_entanglement_explained.py \\
        trained echo"""
import os
import sys
import time
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_entanglement_factors as tf                                              # noqa: E402
import fig_entanglement_driver as fd                                              # noqa: E402

td = tf.td
VARS = ("entanglement", "fr")
BINS = (5, 20)
CHUNK = tf.CHUNK


@torch.no_grad()
def checkpoint(p, P, Bs, dev):
    n, T, H = len(P["goal"]), td.T, p["STM"].shape[-1]
    S1t = torch.zeros(T - 1, H, dtype=torch.float64, device=dev)
    S2t = torch.zeros(T - 1, dtype=torch.float64, device=dev)
    S1 = {key: torch.zeros(T - 1, key[1], H, dtype=torch.float64, device=dev) for key in Bs}
    S2 = {key: torch.zeros(T - 1, key[1], dtype=torch.float64, device=dev) for key in Bs}
    for c0 in range(0, n, CHUNK):
        sl = slice(c0, c0 + CHUNK)
        obs = torch.as_tensor(P["obs"][sl], dtype=torch.float32, device=dev)
        zin = torch.as_tensor(P["Z"][P["env"][sl]], device=dev).reshape(len(obs), -1)
        for t, u in tf.states(p, obs, zin):
            if t == 0:
                continue
            ud = u.double()
            sq = (ud ** 2).sum(-1)
            S1t[t - 1] += ud.sum(0)
            S2t[t - 1] += sq.sum()
            for key, B in Bs.items():
                b = torch.as_tensor(B[sl, t - 1], device=dev)
                S1[key][t - 1].index_add_(0, b, ud)
                S2[key][t - 1].index_add_(0, b, sq)
    m_t = S1t / n
    m = m_t.mean(0)
    tot = float((S2t / n).mean() - (m ** 2).sum())
    within = float(((S2t / n) - (m_t ** 2).sum(-1)).mean())
    ve, ve_raw = np.zeros((len(VARS), len(BINS))), np.zeros((len(VARS), len(BINS)))
    for (var, K), s1 in S1.items():
        nk = n // K
        m_kt = s1 / nk                                                               # (29, K, H)
        between = float(((m_kt - m_t[:, None]) ** 2).sum(-1).mean())
        w_kt = (S2[(var, K)] - nk * (m_kt ** 2).sum(-1)) / (nk - 1)                 # within-bin variance, unbiased
        noise = float((K - 1) / K * (w_kt / nk).mean())
        ve_raw[VARS.index(var), BINS.index(K)] = between
        ve[VARS.index(var), BINS.index(K)] = between - noise
    return {"tot": tot, "within": within, "ve": ve, "ve_raw": ve_raw}


def run(name, dev=torch.device("cuda:0")):
    path = os.path.join(fd.AN, f"ent_explained_{name}.npz")
    P = td.probes()
    x = {"entanglement": P["ent"].mean(-1), "fr": P["fr"].mean(-1)}
    Bs = {(var, K): fd.within_step(x[var], K) for var in VARS for K in BINS}
    eps = td.checkpoints(name)
    res = {}
    t0 = time.time()
    for i, ep in enumerate(eps):
        w = torch.load(td.ckpt_path(name, ep), map_location="cpu", weights_only=False)["weights"]
        o = checkpoint({k: v.to(dev).float() for k, v in w.items()}, P, Bs, dev)
        for k, v in o.items():
            res.setdefault(k, []).append(v)
        if i % 20 == 0:
            print(f"{time.strftime('%H:%M:%S')} {name} ep {ep} ({time.time() - t0:.0f} s): total {o['tot']:.3f}, "
                  f"explained (5 / 20 bins): entanglement {np.round(o['ve'][0], 4)}, FR {np.round(o['ve'][1], 4)}",
                  flush=True)
    np.savez(path, episode=np.array(eps), vars=np.array(VARS), bins=np.array(BINS), fr_def="forward",
             **{k: np.array(v) for k, v in res.items()})          # fr_def: FR = KL(Bayes || naive) (probes 'fr')
    print(f"saved {path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        run(nm)
