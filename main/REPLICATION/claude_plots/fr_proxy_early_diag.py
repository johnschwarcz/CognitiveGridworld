"""Why is the Trained-vs-ESN divergence correlated with FR after 25 episodes? Partial correlations on 20 of the 50
probe environments x 5,000 trajectories (same streams as fig_fr_proxy_training.build_big)."""
import os, sys, numpy as np, torch
ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, ROOT + "/main/REPLICATION", ROOT + "/main/REPLICATION/claude_plots"]
import plot_fig1c_all as F
from fig_fr_proxy import logits_gpu
from fig_fr_proxy_training import kl, _pearson, AN, RUNS, BIG_M, BIG_SEED0
dev = "cuda:1"
P = np.load(os.path.join(AN, "probes.npz"))
L, Zenv = P["L"], P["Z"]
ENVS = range(20)
obs = {e: F.cell_streams(L[e], 0, F.SCAN_CELLS, BIG_M, BIG_SEED0 + e)[0] for e in ENVS}
fr = {e: F.fr_streams(obs[e], L[e]) for e in ENVS}
lj = {e: np.log(np.maximum(F.joint_marginals(obs[e], L[e]), 1e-300)) for e in ENVS}
ent = lambda lp: -(np.exp(lp) * lp).sum(-1)
strength = {e: np.log(10) - ent(lj[e]) for e in ENVS}                 # KL(B^Joint || uniform): evidence strength
n_on = {e: np.cumsum(obs[e].sum(-1), 1)[..., None].repeat(2, -1).astype(float) for e in ENVS}  # observations on so far

def weights(run, E):
    d = torch.load(os.path.join(RUNS, f"{run}_s20261029", f"ckpt_ep{E:05d}_net.pth"), map_location="cpu", weights_only=False)
    return {k: v.detach().float().to(dev) for k, v in d["weights"].items()}

def replay(W, o, z):
    with torch.no_grad():
        r = logits_gpu(W, torch.from_numpy(o.astype(np.float32)).to(dev),
                       torch.from_numpy(np.repeat(np.asarray(z, np.float32)[None], len(o), 0)).to(dev))
        return r.double().cumsum(1).reshape(len(o), F.N_T, 2, F.N_R).log_softmax(-1).cpu().numpy()

def partial(rxy, rxz, ryz):
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))

print("E      r(D,FR)  r(D,S)  r(FR,S)  partial r(D,FR|S) | r(D,n_on) r(FR,n_on) partial|n_on | trained conf, ESN conf (KL to uniform)")
for E in (25, 50, 75, 100, 150, 250, 500, 1000, 49995):
    WT, WE = weights("trained", E), weights("echo", E)
    acc = {k: [] for k in ("dfr", "ds", "frs", "dn", "frn", "cT", "cE")}
    for e in ENVS:
        lT, lE = replay(WT, obs[e], Zenv[e]), replay(WE, obs[e], Zenv[e])
        D = kl(lT, lE)
        acc["dfr"].append(_pearson(D, fr[e])); acc["ds"].append(_pearson(D, strength[e])); acc["frs"].append(_pearson(fr[e], strength[e]))
        acc["dn"].append(_pearson(D, n_on[e])); acc["frn"].append(_pearson(fr[e], n_on[e]))
        acc["cT"].append((np.log(10) - ent(lT)).mean()); acc["cE"].append((np.log(10) - ent(lE)).mean())
    a = {k: np.array(v) for k, v in acc.items()}
    pr = partial(a["dfr"], a["ds"], a["frs"]); pn = partial(a["dfr"], a["dn"], a["frn"])
    print(f"{E:5d}  {np.nanmean(a['dfr']):.3f}   {np.nanmean(a['ds']):.3f}   {np.nanmean(a['frs']):.3f}    {np.nanmean(pr):.3f}"
          f"          | {np.nanmean(a['dn']):.3f}    {np.nanmean(a['frn']):.3f}     {np.nanmean(pn):.3f}     | {a['cT'].mean():.4f} {a['cE'].mean():.4f}", flush=True)
