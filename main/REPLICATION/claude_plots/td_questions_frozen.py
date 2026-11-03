"""Question 5 of the theory session (memory learning-dynamics-phases): does a stack of frozen layers keep the additive
(linear) part of its input and lose the interactions? Depth-9 feedforward networks on the counts (ff_deep/d9_frozen, power
1, lr 1e-3, seed 0, width 1000, 20k episodes), the first k of the eight N x N layers after the trained 15 x N read-in
frozen at initialization, k = 0 (fully trained, ff_deep/d9_p1_lr0.001_rep0) to 8. Each network's own read-in and stack
are run (fp32, CPU; the trainings ran fp16 activations) on 10,000 inputs (the five observation means at T = 30 and the 10-number context code z of the
training-dynamics probe set: the same generator, another world; the stack is a fixed function of its input, so what is
asked is what survives of a representative input), and the representation after layer k (after its ReLU; k = 0: after
the read-in) is read out linearly (ridge, penalty chosen on a validation split; fitted on 25 environments' inputs,
tested on 15 others; the penalty is relative to the centred features' spectrum, so the shrinking scale of the
input-dependent part deep in the stack is not penalized for itself):
  "lin"  R^2 of the 15 inputs themselves (pooled over dimensions: 1 - sum SSE / sum SST),
  "int"  R^2 of the 120 pairwise products x_i x_j (i <= j) after removing what the 15 inputs explain linearly (the
         residual of each product on the inputs and a constant, fitted on all inputs): the interaction part,
both also from the last hidden layer (layer 8, what the read-out sees) "lin_last" / "int_last". Also the mean cosine
between two inputs' representations after layer k "cos" (1,000 random pairs).
-> main/DATA/experimental/training_dynamics/questions/frozen_decoding.npz
Run from the repository root (CPU):
    OMP_NUM_THREADS=4 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_questions_frozen.py"""
import os
import numpy as np
import torch

FF = "/home/johns/Desktop/CognitiveGridworld/main/DATA/experimental/ff_deep"
AN = "/home/johns/Desktop/CognitiveGridworld/main/DATA/experimental/training_dynamics/analysis"
OUT = "/home/johns/Desktop/CognitiveGridworld/main/DATA/experimental/training_dynamics/questions/frozen_decoding.npz"


def net_path(k):
    if k == 0:
        return os.path.join(FF, "d9_p1_lr0.001_rep0_net.pth")
    return os.path.join(FF, "d9_frozen", f"d9_p1_lr0.001_frz{''.join(str(i) for i in range(1, k + 1))}_rep0_net.pth")


def layers(path):
    w = torch.load(path, map_location="cpu", weights_only=False)["weights"]
    return [(w[f"classifier_readin.{2 * i}.weight"].float(), w[f"classifier_readin.{2 * i}.bias"].float()) for i in range(9)]


def reps(L, x, k):
    h = x
    out = {}
    for i, (W, b) in enumerate(L):
        h = torch.relu(h @ W.T + b)
        if i == k:
            out["k"] = h.clone()
    out["last"] = h
    return out


def ridge_r2(H, Y, tr, va, te, lams=tuple(10.0 ** np.arange(-10, 3))):
    """ridge from H to Y fitted on tr, penalty by va, R^2 on te pooled over Y's columns. The penalty is relative to the
    mean squared singular value of the centred features (lam * mean(S^2)), so a stack that shrinks the input-dependent
    part of its activations is not penalized for the scale alone"""
    H = H.double()
    Y = torch.as_tensor(Y).double()
    mu_h, mu_y = H[tr].mean(0), Y[tr].mean(0)
    Ht, Yt = H[tr] - mu_h, Y[tr] - mu_y
    U, S, Vt = torch.linalg.svd(Ht, full_matrices=False)
    UtY = U.T @ Yt
    best = None
    for lam in lams:
        B = Vt.T @ ((S / (S ** 2 + lam * (S ** 2).mean()))[:, None] * UtY)
        r2v = 1 - (((H[va] - mu_h) @ B + mu_y - Y[va]) ** 2).sum() / ((Y[va] - Y[va].mean(0)) ** 2).sum()
        if best is None or r2v > best[0]:
            best = (r2v, B)
    B = best[1]
    return float(1 - (((H[te] - mu_h) @ B + mu_y - Y[te]) ** 2).sum() / ((Y[te] - Y[te].mean(0)) ** 2).sum())


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    P = np.load(os.path.join(AN, "probes.npz"))
    x = np.concatenate([P["obs"].mean(1), P["Z"][P["env"]].reshape(len(P["obs"]), -1)], 1).astype(np.float32)
    assert x.shape[1] == 15, x.shape
    env = P["env"]
    rng = np.random.default_rng(0)
    envs = rng.permutation(np.unique(env))
    tr, va, te = (np.where(np.isin(env, e))[0] for e in (envs[:25], envs[25:35], envs[35:]))
    iu = np.triu_indices(15)
    prod = (x[:, :, None] * x[:, None, :])[:, iu[0], iu[1]].astype(np.float64)
    A = np.concatenate([x.astype(np.float64), np.ones((len(x), 1))], 1)
    resid = prod - A @ np.linalg.lstsq(A, prod, rcond=None)[0]       # the interaction part of each product
    xt = torch.as_tensor(x)
    pairs = rng.integers(0, len(x), (1000, 2))
    res = {k: [] for k in ("k", "lin", "int", "lin_last", "int_last", "cos")}
    for k in range(9):
        L = layers(net_path(k))
        with torch.no_grad():
            R = reps(L, xt, k)
        hk, hl = R["k"], R["last"]
        res["k"].append(k)
        res["lin"].append(ridge_r2(hk, x, tr, va, te))
        res["int"].append(ridge_r2(hk, resid, tr, va, te))
        res["lin_last"].append(ridge_r2(hl, x, tr, va, te))
        res["int_last"].append(ridge_r2(hl, resid, tr, va, te))
        a, b = hk[pairs[:, 0]].double(), hk[pairs[:, 1]].double()
        res["cos"].append(float(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-30)).mean()))
        print(k, {q: round(v[-1], 3) for q, v in res.items() if q != "k"}, flush=True)
    np.savez(OUT, **{q: np.array(v) for q, v in res.items()})
