"""Participation ratios to relate the networks' hidden states to Bayesian beliefs (25 Oct 2026)
-> main/DATA/experimental/fig1c_compression/figdata/round4b/hidden_bayes_prs.npz

Per step t = 1..30, covariance over the 19,607 kept episodes (1,225 environments, pooled), PR = (tr C)^2 / tr(C^2):
  networks (hidden_<net>_<u|c>.npy):  tanh(c_t) (to see what the output gate adds: u_t = o_t * tanh(c_t)), and the
      spectrum (top-5 eigenvalue shares) of c_t, tanh(c_t), u_t
  read-out (logit cache, CLR summed read-out logits x_t): x_t, its first derivative dx_t = x_t - x_{t-1} (= the CLR of
      the step's read-out W u_t + b, exactly, without a floor) and second derivative d2x_t = dx_t - dx_{t-1}
      (= CLR of W (u_t - u_{t-1}) for t >= 2); x_0 = 0, dx_0 = 0
  Bayes, marginals (20, CLR): Optimal and Naive x_t, dx_t, d2x_t
  Bayes, joint (100, CLR over the cells, goal-ordered): the Optimal's joint log-posterior X_t and dX_t; the Naive's
      implied joint (log m_g + log m_c) and its update
  each Bayes/read-out quantity without a floor and with log-probabilities floored at -30 and -10 before centring.
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/hidden_bayes_prs.py"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
BIG = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
OUT = os.path.join(BIG, "figdata", "round4b", "hidden_bayes_prs.npz")
SCR = "/tmp/claude-1002/-home-johns-Desktop-CognitiveGridworld/7af6b091-54ff-45a0-b4dd-ef0a3bc9c778/scratchpad"
CACHE = os.path.join(SCR, "research7", "pr_logit_cache.npz")
FLOORS = (None, -30., -10.)
T = 30


def pr_of(C):
    return np.trace(C) ** 2 / (C * C).sum()


def pr_steps(X):
    """X (n, T, d) -> PR per step (T,)"""
    out = []
    for s in range(X.shape[1]):
        v = X[:, s].astype(np.float64)
        v = v - v.mean(0)
        out.append(pr_of(v.T @ v))
    return np.array(out)


def lse(x, axis):
    m = x.max(axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.exp(x - m).sum(axis))


def floored_marg(X, f):
    """centred log-beliefs (n, T, 20) -> log-probabilities floored at f, re-centred per variable (f None: unchanged)"""
    if f is None:
        return X.astype(np.float64)
    x = X.astype(np.float64).reshape(*X.shape[:2], 2, 10)
    y = np.maximum(x - lse(x, -1)[..., None], f)
    return (y - y.mean(-1, keepdims=True)).reshape(*X.shape[:2], 20)


def floored_joint(lp, f):
    """normalized joint log-probabilities (n, T, 100) -> floored at f, centred over the 100 cells"""
    y = lp if f is None else np.maximum(lp, f)
    return y - y.mean(-1, keepdims=True)


def derivs(X):
    d1 = np.diff(np.concatenate([np.zeros_like(X[:, :1]), X], 1), axis=1)
    d2 = np.diff(np.concatenate([np.zeros_like(d1[:, :1]), d1], 1), axis=1)
    return d1, d2


def main():
    import fig1 as cf
    f = cf._harvest("trained", ["ep_obs_flat", "ep_joint_likelihood", "ep_goal_ind", "ep_ctx_inds"])
    n0 = 20000
    ci = np.sort(f["ep_ctx_inds"][:n0].astype(int), 1)
    keep = ci[:, 0] != ci[:, 1]
    out = {}
    # networks: tanh(c), spectra
    for net in ("trained", "echo"):
        for var in ("c", "u"):
            X = np.load(os.path.join(BIG, f"hidden_{net}_{var}.npy"))[keep]
            if var == "c":
                out[f"{net}_tanhc"] = pr_steps(np.tanh(X.astype(np.float32)))
            for name, Y in ((var, X), ("tanhc", None)) if var == "c" else ((var, X),):
                for s in (4, 29):                                                  # spectra at t = 5 and 30
                    v = (np.tanh(X[:, s].astype(np.float64)) if name == "tanhc" else X[:, s].astype(np.float64))
                    v = v - v.mean(0)
                    ev = np.linalg.eigvalsh(v.T @ v)[::-1]
                    out[f"{net}_{name}_top5_t{s + 1}"] = ev[:5] / ev.sum()
            del X
        print(net, "done", flush=True)
    # read-out and Bayes marginals (cache: CLR, goal-ordered, kept episodes)
    z = np.load(CACHE)
    for a in ("Joint", "Naive", "Trained", "Echo"):
        for fl in FLOORS:
            x = floored_marg(z[a], fl)
            d1, d2 = derivs(x)
            tag = f"{a}_{'none' if fl is None else int(-fl)}"
            out[tag + "_x"], out[tag + "_dx"], out[tag + "_d2x"] = pr_steps(x), pr_steps(d1), pr_steps(d2)
        print(a, "marginals done", flush=True)
    # Bayes joint (exact, float64): the Optimal's joint log-posterior and the Naive's implied joint
    g = f["ep_goal_ind"][:n0].astype(int)[keep]
    o = f["ep_obs_flat"][:n0][keep].astype(np.float64)
    L = np.clip(f["ep_joint_likelihood"][:n0][keep], 1e-12, 1 - 1e-12)
    lpJ = np.zeros((keep.sum(), T, 100))
    for s0 in range(0, keep.sum(), 2000):
        sl = slice(s0, s0 + 2000)
        cum = np.cumsum(np.einsum("nti,nijk->ntjk", o[sl], np.log(L[sl]))
                        + np.einsum("nti,nijk->ntjk", 1 - o[sl], np.log1p(-L[sl])), 1)   # raw order (r0, r1)
        lp = cum - lse(cum.reshape(*cum.shape[:2], -1), -1)[..., None, None]
        swap = g[sl] == 1                                                          # goal-ordered: rows = goal value
        lp[swap] = np.swapaxes(lp[swap], -1, -2)
        lpJ[sl] = lp.reshape(*lp.shape[:2], 100)
    xN = z["Naive"].astype(np.float64).reshape(-1, T, 2, 10)
    lmN = xN - lse(xN, -1)[..., None]                                             # Naive's log marginals
    lpN = (lmN[:, :, 0, :, None] + lmN[:, :, 1, None, :]).reshape(-1, T, 100)
    for nm, lp in (("Joint", lpJ), ("Naive", lpN)):
        for fl in FLOORS:
            X = floored_joint(lp, fl)
            d1, _ = derivs(X)
            tag = f"{nm}joint_{'none' if fl is None else int(-fl)}"
            out[tag + "_X"], out[tag + "_dX"] = pr_steps(X), pr_steps(d1)
        print(nm, "joint done", flush=True)
    # check: the Optimal's marginals recomputed from the joint equal the cache
    xm = np.stack([lse(lpJ.reshape(-1, T, 10, 10), -1), lse(lpJ.reshape(-1, T, 10, 10), -2)], 2)
    xm = (xm - xm.mean(-1, keepdims=True)).reshape(-1, T, 20)
    out["check_joint_vs_cache_maxabs"] = np.abs(xm - z["Joint"]).max()
    np.savez(OUT, **out)
    print("saved", OUT, "check", out["check_joint_vs_cache_maxabs"])


if __name__ == "__main__":
    main()
