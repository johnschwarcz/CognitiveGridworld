"""Activity dimension and update errors split by entanglement, through training (John 11-01: "Im still not convinced
that entanglement is just a consequence, rather than a driver of the need for memory. Another analyses id be
interested in is splitting steps by entanglement and seeing if the PR grows with entanglement.")
-> data main/DATA/experimental/training_dynamics/analysis/ent_split_<run>.npz (every checkpoint; forward passes on
   GPU 0 over all 10,000 probe episodes, the exact fp32 replay of td_analysis.py)

Entanglement per particle-step: rev = the mean over the two variables of td_analysis's probe entanglement (the locked
definition: Jeffreys/2 between naive's per-step likelihood and Bayes's effective update). Bins: within every step t
(steps 2-30) the particles are split into N_BIN quantiles of rev at that step, so each bin holds the same number of
particles from every step (the early-step rise of entanglement does not sort steps into bins).
Per checkpoint and bin:
  pr_u, var_u   the participation ratio (tr C)^2 / tr(C^2) and the variance tr C of the hidden state u's covariance C over
                the bin's particle-steps, each centred on its own (step, bin) mean, the covariances summed over steps
                2-30 BEFORE the PR (audit 11-01: the PR of the step-averaged within-step covariance; per step, then
                averaged, is 9-12% lower for the Trained RNN at the end, ~1% for the ESN; pooled over the 50
                environments)
  pr_x, var_x   the same for the LSTM input x (the read-in output): the input control
  pr_c, var_c   the same for the cell state c
  dis           the mean disagreement of the network's per-step update with Bayes's: Jeffreys/2 between softmax of the
                network's per-step logit increment and softmax of Bayes's per-step increment, per variable, averaged
                over both variables (for the naive observer this disagreement is the entanglement itself)
  dis_naive     the same for naive (= the bin's mean entanglement)
Run from the repository root (GPU 0):
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_entanglement_split.py \\
        trained echo"""
import os
import sys
import time
import numpy as np
import torch

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_analysis as td                                                          # noqa: E402
import gradflow_analysis as ga                                                    # noqa: E402

AN = os.path.join(td.TD, "analysis")
N_BIN = 5
STEPS = range(1, td.T)                                                             # 0-based: steps 2-30


def bins(P):
    """(n, T) quantile bin of rev within each step (-1 at step 1)"""
    rev = P["ent"].mean(-1)
    b = np.full(rev.shape, -1)
    for t in STEPS:
        q = np.quantile(rev[:, t], np.linspace(0, 1, N_BIN + 1)[1:-1])
        b[:, t] = np.searchsorted(q, rev[:, t])
    return b, rev


def _jeff(a, b):
    """Jeffreys / 2 between softmax(a) and softmax(b) over the last axis"""
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    return .5 * ((la.exp() - lb.exp()) * (la - lb)).sum(-1)


@torch.no_grad()
def checkpoint_stats(m, P, B, dev):
    n = len(P["obs"])
    obs = torch.as_tensor(P["obs"], dtype=torch.float32, device=dev)
    zin = torch.as_tensor(P["Z"][P["env"]], device=dev).reshape(n, -1)
    W_ih, W_hh, b = ga.lstm_weights(m)
    H = W_hh.shape[1]
    Bt = torch.as_tensor(B, device=dev)
    D = torch.as_tensor(P["inc_D"], device=dev).double()                         # Bayes's per-step increments
    acc = {s: [torch.zeros(N_BIN, H, H, dtype=torch.float64, device=dev), torch.zeros(N_BIN, device=dev,
                                                                                         dtype=torch.float64)]
           for s in ("u", "x", "c")}
    dis = torch.zeros(N_BIN, dtype=torch.float64, device=dev)
    cnt = torch.zeros(N_BIN, dtype=torch.float64, device=dev)
    h, c = td._start(m, n)
    prev = None
    for t in range(td.T):
        x = td._readin(m, obs[:, t], zin)
        h, c, _ = td._cell(x @ W_ih.T + b + h @ W_hh.T, c)
        lg = m.classifier_readout(h).double()
        if m.joint_out:                                                            # marginal increments of a joint read-out
            cum = lg if prev is None else prev + lg
            mlam = torch.stack([cum.reshape(n, td.R, td.R).logsumexp(-1), cum.reshape(n, td.R, td.R).logsumexp(-2)], 1)
            pm = (torch.stack([prev.reshape(n, td.R, td.R).logsumexp(-1), prev.reshape(n, td.R, td.R).logsumexp(-2)], 1)
                  if prev is not None else torch.full_like(mlam, np.log(td.R)))
            d, prev = mlam - pm, cum
        else:
            d = lg.reshape(n, td.C, td.R)
        if t == 0:
            continue
        e = _jeff(d, D[:, t]).mean(-1)                                             # (n,) both variables
        bt = Bt[:, t]
        for k in range(N_BIN):
            sel = bt == k
            dis[k] += e[sel].sum()
            cnt[k] += sel.sum()
            for s, X in (("u", h), ("x", x), ("c", c)):
                Xs = X[sel].double()
                Xs = Xs - Xs.mean(0)                                               # centred on its (step, bin) mean
                acc[s][0][k] += Xs.T @ Xs
                acc[s][1][k] += len(Xs)
    out = {"dis": (dis / cnt).cpu().numpy()}
    for s, (S2, N) in acc.items():
        Cm = S2 / N[:, None, None]
        tr = Cm.diagonal(dim1=-2, dim2=-1).sum(-1)
        out["pr_" + s] = (tr ** 2 / (Cm * Cm.transpose(-1, -2)).sum((-1, -2))).cpu().numpy()
        out["var_" + s] = tr.cpu().numpy()
    return out


def run(name, dev=torch.device("cuda:0")):
    path = os.path.join(AN, f"ent_split_{name}.npz")
    P = td.probes()
    B, rev = bins(P)
    eps = td.checkpoints(name)
    done = dict(np.load(path)) if os.path.exists(path) else {}
    have = set(done.get("episode", np.array([], int)).tolist())
    m = td.build(name, eps[0], dev)
    rows = {k: list(done[k]) for k in done} if done else {}
    t0 = time.time()
    for ep in eps:
        if ep in have:
            continue
        td.load_weights(m, name, ep)
        s = checkpoint_stats(m, P, B, dev)
        s["episode"] = ep
        for k, v in s.items():
            rows.setdefault(k, []).append(v)
        print(f"{time.strftime('%H:%M:%S')} {name} ep {ep}: PR u by bin {np.round(s['pr_u'], 1)}, "
              f"dis {np.round(s['dis'], 3)}", flush=True)
    order = np.argsort(rows["episode"])
    out = {k: np.array(v)[order] for k, v in rows.items()}
    sel = B >= 0
    out["dis_naive"] = np.array([rev[sel & (B == k)].mean() for k in range(N_BIN)])
    out["rev_bin_mean"] = out["dis_naive"]
    out["n_bin"] = np.array([(sel & (B == k)).sum() for k in range(N_BIN)])
    np.savez(path, **out)
    print(f"saved {path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        run(name)
