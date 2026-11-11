"""Does a blocked half carry more of the error signal? The obvious objection to td_tt_learnmask's from-scratch FR blocks
is "you blocked the biggest gradients". For checkpoints of the from-scratch runs, on the 10,000 probes (50 test
environments x 200), the training loss exactly as model_backward.SANITY_loss computes it: the mean over episodes, steps
and goal values of (S - min S + 1e-8) ** .5, where S is the elementwise symmetric KL between the network's goal belief
and Bayes's, each soft-clipped to [1e-3, 1 - 1e-3]. From it:
  g_bs   the gradient with respect to the goal read-out increment of episode b at step s (what det_* detaches)
  G_bs   |g_bs|^2, summed over the 10 goal values
  dW     the read-out weight gradient sum_{b,s} g_bs (x) h_bs, with h_bs the LSTM output, kept for each half
Each step's probes are ranked as in training: train_learn_masks.goal_fr (KL(Bayes_t || naive_t)) or goal_entanglement,
top / bottom half, and a random half from a fixed generator. Stored per checkpoint:
  share (6, T)  the sum of G over each half (hi_fr, lo_fr, rand, hi_ent, lo_ent, all) at every step
  wnorm (6,)    |dW| of each half;  wcos (6,): its cosine with the whole batch's dW
  over_loss (T,)  the share of the highest-FR half that is also in the network's highest-loss half (loss = the
                  episode-step's summed loss terms), at every step
  G, loss (n, T) float32 per (episode, step), for the control only
-> main/DATA/experimental/training_dynamics/through_training/learnmask_grad/<cond>_EEEEE.npz
Run from the repository root:
    OMP_NUM_THREADS=6 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_learnmask_grad.py <cond> [raw ...]"""
import os
import sys
import time
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main"), os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots"), os.path.join(ROOT, "main", "REPLICATION", "trainings")]
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_intervene as ti                                                                  # noqa: E402
import td_tt_chaos_ent as ce                                                                  # noqa: E402
from train_learn_masks import goal_fr, goal_entanglement                                      # noqa: E402

OUT = os.path.join(gap.TT, "learnmask_grad")
T, R, NV = 30, 10, 2
KINDS = ("hi_fr", "lo_fr", "rand", "hi_ent", "lo_ent", "all")


def soft_clip(x, eps=1e-3):
    """model_backward.soft_clip"""
    ciel = 1. - eps
    x = ciel - (1 / (1 + torch.exp(-(ciel - x) / eps))) * (ciel - x)
    return eps + (1 / (1 + torch.exp(-(x - eps) / eps))) * (x - eps)


def masks(P):
    """(6, n, T) bool: each step's top / bottom half by goal FR and by goal entanglement, a fixed random half, all"""
    n = len(P["goal"])
    i = np.arange(n)
    jb = torch.as_tensor(P["jb"][i, :, P["goal"]], dtype=torch.float32)
    nb = torch.as_tensor(P["nb"][i, :, P["goal"]], dtype=torch.float32)
    fr, ent = goal_fr(jb, nb), goal_entanglement(jb, nb)
    k = n // 2
    out = torch.zeros(6, n, T, dtype=torch.bool)
    gen = torch.Generator().manual_seed(20261109)
    for t in range(T):
        out[0, torch.topk(fr[:, t], k).indices, t] = True
        out[1, torch.topk(-fr[:, t], k).indices, t] = True
        out[2, torch.randperm(n, generator=gen)[:k], t] = True
        out[3, torch.topk(ent[:, t], k).indices, t] = True
        out[4, torch.topk(-ent[:, t], k).indices, t] = True
    out[5] = True
    return out, fr


def one(cond, raw, P, M, chunk=2500):
    p, table = ce.setup(f"learnmask_{cond}", raw, torch.device("cpu"))
    n = len(P["goal"])
    goal = torch.as_tensor(P["goal"])
    B = torch.as_tensor(P["jb"][np.arange(n), :, P["goal"]], dtype=torch.float64)
    inc, U = [], []
    with torch.no_grad():
        for b0 in range(0, n, chunk):
            envt, code = ce.codes(np.arange(b0, min(n, b0 + chunk)), torch.device("cpu"))
            _, u, _, lg = ce.replay(p, table, envt, code)
            sl = torch.arange(b0, min(n, b0 + chunk))
            inc.append(lg.reshape(len(sl), T, NV, R)[torch.arange(len(sl)), :, goal[sl]])
            U.append(u[:, 1:].half())                                                    # (b, T, H); fp16 to fit
    inc = torch.cat(inc).double().requires_grad_(True)                              # (n, T, R) goal increments
    q = torch.softmax(inc.cumsum(1), -1)
    qc, bc = soft_clip(q), soft_clip(B)
    S = .5 * (qc - bc) * torch.log(qc / bc)                                         # elementwise symmetric KL
    terms = (S - S.detach().min() + 1e-8) ** .5
    g, = torch.autograd.grad(terms.mean(), inc)
    G = (g ** 2).sum(-1)                                                            # (n, T)
    loss = terms.detach().sum(-1)                                                   # (n, T)
    share = np.stack([(G * m).sum(0).numpy() for m in M])                           # (6, T)
    dW = torch.zeros(6, NV * R, U[0].shape[-1], dtype=torch.float64)
    for b0, u in zip(range(0, n, chunk), U):
        sl = slice(b0, b0 + len(u))
        for v in range(NV):
            sel = (goal[sl] == v)
            if not sel.any():
                continue
            gv, uv = g[sl][sel], u[sel].double()
            for j, m in enumerate(M):
                dW[j, v * R:(v + 1) * R] += torch.einsum("bsr,bsh->rh", gv * m[sl][sel][..., None], uv)
    flat = dW.reshape(6, -1)
    wnorm = flat.norm(dim=1).numpy()
    wcos = (flat @ flat[-1] / (flat.norm(dim=1) * flat[-1].norm() + 1e-300)).numpy()
    k = n // 2
    hi_loss = torch.zeros(n, T, dtype=torch.bool)
    for t in range(T):
        hi_loss[torch.topk(loss[:, t], k).indices, t] = True
    over = ((hi_loss & M[0]).sum(0).float() / M[0].sum(0).float()).numpy()
    return dict(share=share, wnorm=wnorm, wcos=wcos, over_loss=over, G=G.float().numpy(), loss=loss.float().numpy())


def main(cond, raws):
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 6)))
    P = ti.probes() | {k: v for k, v in np.load(os.path.join(ti.AN, "probes.npz")).items() if k in ("jb", "nb")}
    M, _ = masks(P)
    for raw in raws:
        path = os.path.join(OUT, f"{cond}_{raw:05d}.npz")
        if os.path.exists(path):
            continue
        t0 = time.time()
        r = one(cond, raw, P, M)
        if cond != "ctrl_s0":
            r.pop("G"), r.pop("loss")
        np.savez(path, raw=raw, x=gap.train_x(raw), kinds=np.array(KINDS), **r)
        tot = r["share"][-1].sum()
        print(f"{cond} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s | share of |g|^2 "
              + " ".join(f"{k} {r['share'][j].sum() / tot:.3f}" for j, k in enumerate(KINDS[:5]))
              + f" | |dW| hi/lo/rand {r['wnorm'][0] / r['wnorm'][-1]:.3f}/{r['wnorm'][1] / r['wnorm'][-1]:.3f}/"
                f"{r['wnorm'][2] / r['wnorm'][-1]:.3f} | hi-FR in hi-loss {r['over_loss'].mean():.3f}", flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    rs = [int(x) for x in sys.argv[2:]] or list(range(0, 3001, 200))
    main(c, rs)
