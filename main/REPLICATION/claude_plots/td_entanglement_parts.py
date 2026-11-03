"""Variance and dimension (PR) of the four nested parts of the network state, split by entanglement, through training
(John 11-01: "split between episode into all 3, so we can see PR due to time, trajectory stochasticity, latent pair,
environmental interaction ... be very careful, explicit and simple about how we are calculating things").
-> data main/DATA/experimental/training_dynamics/analysis/ent_parts_<run>.npz (every 3rd checkpoint and the last; GPU 0)

Notation: environment e (50), latent pair c = (r0, r1) (100), episode i (a trajectory: its own observations), step t.
u_it = the LSTM output (1,000 neurons). Averages over steps 2-30 carry a bar:
  ubar_i  = episode i's average state
  m_ec    = the average of ubar over 20 REFERENCE trajectories of e and c (new observations, not the probes)
  m_e     = the average of m_ec over the 100 latent pairs;   m = the average of m_e over the 50 environments
Every particle-step's state splits exactly into four parts (each the deviation from its parent group's average):
  time        d = u_it  - ubar_i     the step's deviation from its own episode's average
  trajectory  d = ubar_i - m_ec      the episode's average vs its latent pair's (trajectory stochasticity)
  latent pair d = m_ec  - m_e        the latent pair's average vs its environment's
  environment d = m_e   - m         the environment's average vs the overall
  (total      d = u_it  - m         = the sum of the four)
Bins: each particle-step (steps 2-30) by its own entanglement (rev, both variables averaged), ranked within its step:
5 fifths, each 2,000 particles of every step (fig_entanglement_driver.within_step).
Per part and fifth: M = the average of d d^T over the fifth's particle-steps (1,000 x 1,000; the part's covariance
within its parent groups, pooled over the fifth), then
  variance  tr M                    (the mean squared length of d, summed over neurons)
  PR        (tr M)^2 / tr(M M)      (its dimension: how many directions the variance spreads over)
The reference averages carry sampling noise that adds, in expectation, N = sum_g w_g W_g / 20 to the trajectory and
latent-pair M (W_g: the within-group covariance of the 20 reference trajectory averages of group g = (e, c); w_g: the
fifth's share of particle-steps from g); it is subtracted (the environment's is 100x smaller and left). Stored per
checkpoint: "tr", "pr" (K, part, fifth), parts order PARTS; "noise_tr" (K, fifth) = tr N; "count" (fifth,).
Run from the repository root (GPU 0):
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_entanglement_parts.py \\
        trained echo"""
import os
import sys
import time
import numpy as np
import torch

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_entanglement_factors as tf                                              # noqa: E402
import fig_entanglement_driver as fd                                              # noqa: E402

td = tf.td
PARTS = ("time", "trajectory", "pair", "environment", "total")
N_BIN = 5
CHUNK = tf.CHUNK


@torch.no_grad()
def checkpoint(p, P, ref, B, dev):
    env_r, cell_r, obs_r = ref
    R2, T, H, S = td.R * td.R, td.T, p["STM"].shape[-1], tf.S_REF
    G = P["L"].shape[0] * R2
    # reference: every trajectory's average state over steps 2-30, then the group averages
    Ubar_r = torch.zeros(len(env_r), H, device=dev)
    for c0 in range(0, len(env_r), CHUNK):
        sl = slice(c0, c0 + CHUNK)
        obs = torch.as_tensor(obs_r[sl], dtype=torch.float32, device=dev)
        zin = torch.as_tensor(P["Z"][env_r[sl]], device=dev).reshape(len(obs), -1)
        for t, u in tf.states(p, obs, zin):
            if t >= 1:
                Ubar_r[sl] += u
    Ubar_r /= T - 1
    g_r = torch.as_tensor(env_r * R2 + cell_r, device=dev)
    m_ec = torch.zeros(G, H, dtype=torch.float64, device=dev).index_add_(0, g_r, Ubar_r.double()) / S
    m_e = m_ec.reshape(-1, R2, H).mean(1)
    m = m_e.mean(0)
    # the fifths' composition by group (environment, pair) over their particle-steps, for the noise matrices
    g_p = P["env"] * R2 + P["cell"]
    count = np.array([(B == k).sum() for k in range(N_BIN)])
    w = np.stack([np.bincount(np.repeat(g_p, B.shape[1])[(B == k).ravel()], minlength=G) / count[k]
                  for k in range(N_BIN)])                                          # (fifth, G), each row sums to 1
    D = (Ubar_r.double() - m_ec[g_r]).float()                                       # reference deviations from m_ec
    noise = []
    for k in range(N_BIN):
        Dw = D * torch.as_tensor(np.sqrt(w[k] / (S * (S - 1))), dtype=torch.float32, device=dev)[g_r][:, None]
        noise.append((Dw.T @ Dw).double())                                          # sum_g w_g W_g / S
    del Ubar_r, D
    # probes: accumulate d d^T per part and fifth
    M = torch.zeros(len(PARTS), N_BIN, H, H, dtype=torch.float64, device=dev)
    n = len(P["goal"])
    Bt = torch.as_tensor(B, device=dev)
    for c0 in range(0, n, CHUNK):
        sl = slice(c0, c0 + CHUNK)
        obs = torch.as_tensor(P["obs"][sl], dtype=torch.float32, device=dev)
        e = torch.as_tensor(P["env"][sl], device=dev)
        g = e * R2 + torch.as_tensor(P["cell"][sl], device=dev)
        zin = torch.as_tensor(P["Z"][P["env"][sl]], device=dev).reshape(len(obs), -1)
        U = torch.zeros(len(obs), T, H, device=dev)
        for t, u in tf.states(p, obs, zin):
            U[:, t] = u
        ubar = U[:, 1:].mean(1)
        const = {"trajectory": (ubar.double() - m_ec[g]).float(), "pair": (m_ec[g] - m_e[e]).float(),
                 "environment": (m_e[e] - m).float()}
        for t in range(1, T):
            d = {"time": U[:, t] - ubar, **const, "total": (U[:, t].double() - m).float()}
            bt = Bt[sl, t - 1]
            for k in range(N_BIN):
                rows = bt == k
                for j, part in enumerate(PARTS):
                    X = d[part][rows]
                    M[j, k] += (X.T @ X).double()
        del U
    M /= torch.as_tensor(count, dtype=torch.float64, device=dev)[None, :, None, None]
    for k in range(N_BIN):
        for part in ("trajectory", "pair"):
            M[PARTS.index(part), k] -= noise[k]
    tr = M.diagonal(dim1=-2, dim2=-1).sum(-1)
    pr = tr ** 2 / (M * M).sum((-1, -2))
    return {"tr": tr.cpu().numpy(), "pr": pr.cpu().numpy(),
            "noise_tr": np.array([float(x.trace()) for x in noise]), "count": count}


def run(name, dev=torch.device("cuda:0")):
    path = os.path.join(fd.AN, f"ent_parts_{name}.npz")
    P = td.probes()
    ref = tf.reference(P)
    B = fd.within_step(P["ent"].mean(-1), N_BIN)                                     # (n, 29), steps 2-30
    eps = td.checkpoints(name)
    eps = sorted(set(eps[::tf.EVERY]) | {eps[-1]})
    res = {}
    t0 = time.time()
    for i, ep in enumerate(eps):
        w = torch.load(td.ckpt_path(name, ep), map_location="cpu", weights_only=False)["weights"]
        o = checkpoint({x: v.to(dev).float() for x, v in w.items()}, P, ref, B, dev)
        for k, v in o.items():
            res.setdefault(k, []).append(v)
        if i % 10 == 0:
            print(f"{time.strftime('%H:%M:%S')} {name} ep {ep} ({time.time() - t0:.0f} s): PR by fifth, time "
                  f"{np.round(o['pr'][0], 1)}, pair {np.round(o['pr'][2], 1)}", flush=True)
        torch.cuda.empty_cache()
    np.savez(path, episode=np.array(eps), parts=np.array(PARTS), tr=np.stack(res["tr"]), pr=np.stack(res["pr"]),
             noise_tr=np.stack(res["noise_tr"]), count=res["count"][0])
    print(f"saved {path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        run(nm)
