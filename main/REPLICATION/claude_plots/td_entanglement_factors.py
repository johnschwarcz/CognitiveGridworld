"""Variance of the network state by factor - time, environment, joint realization, episode - per particle-step, for
splitting by entanglement (John 11-01: "Maybe I need to see curves for neural variance across time, environment,
episode and joint realization, split by entanglement.").
-> data main/DATA/experimental/training_dynamics/analysis/ent_factors_<run>.npz (every EVERY-th checkpoint and the last;
   GPU 0, ~6 s per checkpoint)

Each probe state u_it (td_analysis's 10,000 probes: episode i of environment e with latent pair c = (r0, r1), step t; u
= the LSTM output, as for the PR) is split exactly,  u_it - mu = tau_t + eps_et + rho_ect + eta_it,  with means from a
separate REFERENCE set (the same 50 environments x 100 latent pairs, S_REF new observation streams each, seed REF_SEED),
so no probe particle contributes to the means it is measured against:
  tau_t   = mu_t - mu         time: the shared time course (mu_t: mean of the reference states at step t; mu: their mean
                              over steps 2-30)
  eps_et  = mu_et - mu_t      environment: how the environment (the pair of active variables) moves the state at step t
  rho_ect = mu_ect - mu_et    joint realization: how the latent pair (r0, r1) moves it, within its environment
  eta_it  = u_it - mu_ect     episode: this trajectory's own deviation (its particular observations)
Stored per probe particle-step (float16, steps 1-30): "e_tot" |u_it - mu|^2, "e_time" |tau_t|^2, "e_env" |eps_et|^2,
"e_real" |rho_ect|^2, "e_ep" |eta_it|^2 (the four parts plus the cross terms make the total; over all particles the
cross terms vanish up to the reference noise). The reference means' sampling noise adds the same expected amount to
e_real and e_ep: "noise" (K, 5000, T) = the unbiased within-(e, c, t) variance of the reference states / S_REF, stored
per (environment * 100 + pair, step) for subtraction.
Second split, "time_last" (John 11-01: "take the variance across timesteps, and then average over episodes"): time
nested within the episode, u_it - m = (m_e - m) + (m_ec - m_e) + (ubar_i - m_ec) + (u_it - ubar_i), every bar and m an
average over steps 2-30: environment, joint realization, episode (the trajectory's own average state, ubar_i) and time
(the step's deviation from its own episode's average: averaged over a whole episode, its variance across its steps)
-> ent_factors_tl_<run>.npz: e_tot, e_time (K, n, T); e_env, e_real, e_ep (K, n) (constant over an episode's steps);
noise (K, 5000) for e_real and e_ep, as above.
Run from the repository root (GPU 0; mode time_first (default) or time_last):
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_entanglement_factors.py \\
        time_last trained echo"""
import os
import sys
import time
import numpy as np
import torch

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_entanglement_driver as dr                                               # noqa: E402

td = dr.td
S_REF, REF_SEED = 20, 20261105
EVERY = 3                         # every 3rd checkpoint and the last (6 s each; all 114 x 2 networks would take ~23 min)
CHUNK = 5000
KEYS = ("e_tot", "e_time", "e_env", "e_real", "e_ep")


def reference(P):
    """the reference set: environment, latent-pair index (r0 * 10 + r1) and observations of S_REF streams per pair"""
    rng = np.random.default_rng(REF_SEED)
    E, R = P["L"].shape[0], td.R
    env = np.repeat(np.arange(E), R * R * S_REF)
    cell = np.tile(np.repeat(np.arange(R * R), S_REF), E)
    L = P["L"][env, :, cell // R, cell % R]                                         # (n, 5): the cell's 5 probabilities
    obs = rng.random((len(env), td.T, td.NO)) < L[:, None, :]
    return env, cell, obs


@torch.no_grad()
def states(p, obs, zin):
    """yield (t, u_t) for a chunk: the hand-written fp32 LSTM of td_analysis"""
    h, c = dr._start(p, len(obs))
    for t in range(td.T):
        x = torch.relu(torch.cat([obs[:, t], zin], -1) @ p["classifier_readin.weight"].T + p["classifier_readin.bias"])
        pre = x @ p["LSTM.weight_ih_l0"].T + p["LSTM.bias_ih_l0"] + p["LSTM.bias_hh_l0"] + h @ p["LSTM.weight_hh_l0"].T
        h, c, _ = td._cell(pre, c)
        yield t, h


@torch.no_grad()
def checkpoint(p, P, ref, dev):
    env_r, cell_r, obs_r = ref
    G, T, H = P["L"].shape[0] * td.R * td.R, td.T, p["STM"].shape[-1]
    S1 = torch.zeros(G, T, H, device=dev)
    S2 = torch.zeros(G, T, dtype=torch.float64, device=dev)
    for c0 in range(0, len(env_r), CHUNK):
        sl = slice(c0, c0 + CHUNK)
        obs = torch.as_tensor(obs_r[sl], dtype=torch.float32, device=dev)
        zin = torch.as_tensor(P["Z"][env_r[sl]], device=dev).reshape(len(obs), -1)
        g = torch.as_tensor(env_r[sl] * td.R * td.R + cell_r[sl], device=dev)
        for t, u in states(p, obs, zin):
            S1[:, t].index_add_(0, g, u)
            S2[:, t].index_add_(0, g, (u.double() ** 2).sum(-1))
    m_ect = S1 / S_REF                                                               # (G, T, H)
    noise = ((S2 / S_REF - (m_ect.double() ** 2).sum(-1)) * S_REF / (S_REF - 1) / S_REF).float()
    m_et = m_ect.reshape(-1, td.R * td.R, T, H).mean(1)                              # (E, T, H)
    m_t = m_et.mean(0)                                                               # (T, H)
    mu = m_t[1:].mean(0)
    n = len(P["goal"])
    out = {k: np.zeros((n, T), np.float16) for k in KEYS}
    for c0 in range(0, n, CHUNK):
        sl = slice(c0, c0 + CHUNK)
        obs = torch.as_tensor(P["obs"][sl], dtype=torch.float32, device=dev)
        e = torch.as_tensor(P["env"][sl], device=dev)
        zin = torch.as_tensor(P["Z"][P["env"][sl]], device=dev).reshape(len(obs), -1)
        g = e * td.R * td.R + torch.as_tensor(P["cell"][sl], device=dev)
        for t, u in states(p, obs, zin):
            parts = {"e_tot": u - mu, "e_time": (m_t[t] - mu).expand_as(u), "e_env": m_et[e, t] - m_t[t],
                     "e_real": m_ect[g, t] - m_et[e, t], "e_ep": u - m_ect[g, t]}
            for k, v in parts.items():
                out[k][sl, t] = (v.double() ** 2).sum(-1).cpu().numpy()
    out["noise"] = noise.cpu().numpy()
    return out


@torch.no_grad()
def checkpoint_time_last(p, P, ref, dev):
    """John 11-01 ("take the variance across timesteps, and then average over episodes"): time nested within the
    episode, u_it - m = (m_e - m) + (m_ec - m_e) + (ubar_i - m_ec) + (u_it - ubar_i): environment, joint realization,
    episode (the trajectory's own average state) and time (this step's deviation from its episode's average); bars and
    m: averages over steps 2-30 (m_ec over the S_REF reference trajectories of environment e and pair c)"""
    env_r, cell_r, obs_r = ref
    G, T, H = P["L"].shape[0] * td.R * td.R, td.T, p["STM"].shape[-1]
    A1 = torch.zeros(G, H, dtype=torch.float64, device=dev)
    A2 = torch.zeros(G, dtype=torch.float64, device=dev)
    for c0 in range(0, len(env_r), CHUNK):
        sl = slice(c0, c0 + CHUNK)
        obs = torch.as_tensor(obs_r[sl], dtype=torch.float32, device=dev)
        zin = torch.as_tensor(P["Z"][env_r[sl]], device=dev).reshape(len(obs), -1)
        g = torch.as_tensor(env_r[sl] * td.R * td.R + cell_r[sl], device=dev)
        ubar = torch.zeros(len(obs), H, device=dev)
        for t, u in states(p, obs, zin):
            if t >= 1:
                ubar += u
        ubar = ubar.double() / (T - 1)
        A1.index_add_(0, g, ubar)
        A2.index_add_(0, g, (ubar ** 2).sum(-1))
    m_ec = A1 / S_REF                                                                # (G, H)
    noise = (A2 / S_REF - (m_ec ** 2).sum(-1)) * S_REF / (S_REF - 1) / S_REF
    m_e = m_ec.reshape(-1, td.R * td.R, H).mean(1)
    m = m_e.mean(0)
    n = len(P["goal"])
    out = {"e_tot": np.zeros((n, T), np.float16), "e_time": np.zeros((n, T), np.float16),
           "e_env": np.zeros(n, np.float32), "e_real": np.zeros(n, np.float32), "e_ep": np.zeros(n, np.float32)}
    for c0 in range(0, n, CHUNK):
        sl = slice(c0, c0 + CHUNK)
        obs = torch.as_tensor(P["obs"][sl], dtype=torch.float32, device=dev)
        e = torch.as_tensor(P["env"][sl], device=dev)
        zin = torch.as_tensor(P["Z"][P["env"][sl]], device=dev).reshape(len(obs), -1)
        g = e * td.R * td.R + torch.as_tensor(P["cell"][sl], device=dev)
        U = torch.zeros(len(obs), T, H, device=dev)
        for t, u in states(p, obs, zin):
            U[:, t] = u
        U = U.double()
        ubar = U[:, 1:].mean(1)
        out["e_env"][sl] = ((m_e[e] - m) ** 2).sum(-1).cpu().numpy()
        out["e_real"][sl] = ((m_ec[g] - m_e[e]) ** 2).sum(-1).cpu().numpy()
        out["e_ep"][sl] = ((ubar - m_ec[g]) ** 2).sum(-1).cpu().numpy()
        out["e_time"][sl] = ((U - ubar[:, None]) ** 2).sum(-1).cpu().numpy()
        out["e_tot"][sl] = ((U - m) ** 2).sum(-1).cpu().numpy()
        del U
    out["noise"] = noise.float().cpu().numpy()
    return out


def run(name, mode="time_first", dev=torch.device("cuda:0")):
    tag = "" if mode == "time_first" else "_tl"
    path = os.path.join(dr.AN, f"ent_factors{tag}_{name}.npz")
    fn = checkpoint if mode == "time_first" else checkpoint_time_last
    P = td.probes()
    ref = reference(P)
    eps = td.checkpoints(name)
    eps = sorted(set(eps[::EVERY]) | {eps[-1]})
    res = {}
    t0 = time.time()
    for i, ep in enumerate(eps):
        w = torch.load(td.ckpt_path(name, ep), map_location="cpu", weights_only=False)["weights"]
        o = fn({x: v.to(dev).float() for x, v in w.items()}, P, ref, dev)
        for k, v in o.items():
            res.setdefault(k, []).append(v)
        if i % 10 == 0:
            print(f"{time.strftime('%H:%M:%S')} {name} {mode} ep {ep} ({time.time() - t0:.0f} s): mean "
                  + ", ".join(f"{k} {np.asarray(v, np.float64).mean():.3g}" for k, v in o.items() if k != "noise"),
                  flush=True)
    np.savez(path, episode=np.array(eps), s_ref=S_REF, **{k: np.stack(v) for k, v in res.items()})
    print(f"saved {path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args.pop(0) if args and args[0] in ("time_first", "time_last") else "time_first"
    for nm in args:
        run(nm, mode)
