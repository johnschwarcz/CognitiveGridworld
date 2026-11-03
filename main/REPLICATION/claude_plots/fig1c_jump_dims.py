"""Do the Trained RNN's jumps go through extra hidden dimensions? (27 Oct 2026; exploratory, John's 'wormhole'
hypothesis: a split is a passage through a higher-dimensional space, and the effective dimensionality rises during or
after the jump) -> main/DATA/experimental/fig1c_compression/figdata/round7_jumps/jump_dims_<cell|dist>.npz

Data: the trained harvest's first 20,000 episodes (distinct-variable environments), the Trained RNN's beliefs
(fig1_movie.beliefs_every_step) and its stored hidden output u (hidden_trained_u.npy, float16; its read-out
reproduces the beliefs to < 2e-5), u before step 1 = the learned initial state u0.
Jump: a step that moves the posterior mean x = (E[r_g], E[r_c]) by more than JUMP = 2 cells (the split movie's).
Events: isolated jumps (no other jump within lags -4 .. +6) at steps 6 .. 24 (lag windows inside the episode; steps
1-5 hold the first localization). Controls: for each event, one episode of the same true cell with no jump in the same
window (seed 0), taken at the same steps (CTRL = cell); or (CTRL = dist) one episode of any cell whose distance to its
true cell before the step (lag -1) is within 0.5 cells of the event's, with no jump in the window: a particle as far
from its cell that does not jump. Lags l = -4 .. +10 (l = 0: the jump step; steps 6 .. 20).
Subspaces of u (1000 units): Q, the read-out's 18 directions (the beliefs' only view of u); G, the shared grid's top 10
directions outside Q (the cell effect of the scan's cell means, train environments, as fig1c_lesion_test); IT, the top
30 directions of the environment-specific (cell x environment) interaction outside Q and G (the lesion test's V_it,
the dimensions of the 13 -> 64 expansion); REST, everything else.
Measures per lag, events and controls alike (raw, absolute units):
    move       the belief step |x_t - x_{t-1}| (cells); the Optimal's at the same step
    dE_S       mean squared update |P_S du_t|^2 per subspace S, du_t = u_t - u_{t-1}
    sE_S       mean squared deviation |P_S (u_t - mu_t)|^2 from the population mean at the same step
    PR_du      participation ratio (tr C)^2 / tr(C^2) of the updates' covariance over the events at that lag, all of u
               and outside Q (output-null); bootstrap 2.5-97.5% over events (equal numbers of events and controls)
    PR_u       the same for the states (each centred on its own step's population mean)
Jump kinds: 'toward' (the distance to the true cell falls by more than 2 cells) and 'away' (it rises by more than 2).
Run from the repository root (about 2 min):
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/fig1c_jump_dims.py [cell|dist]"""
import os
import sys
import time
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
OUT = os.path.join(SCAN, "figdata", "round7_jumps")
JUMP, LAGS, K0, K1 = 2., np.arange(-4, 11), 5, 19                                # k: 0-based step of the jump
CTRL = sys.argv[1] if len(sys.argv) > 1 else "cell"                               # the control kind (docstring)
H, NB = 1000, 100
ENVS = [e for e in range(40) if e != 2]
TRAIN_POS = [p for p in range(39) if p not in range(0, 39, 4)]                   # (the lesion test's)


def subspaces():
    """orthonormal bases (H, k) of Q, G, IT (see the docstring); REST is the complement"""
    f = np.load(os.path.join(SCAN, "hidden_scan_cellmeans_trained.npz"))
    assert list(f["env_index"]) == ENVS
    Q = f["Q"].astype(np.float64)
    tr = np.array(TRAIN_POS)
    X = f["u"][tr[:, None], f["perm_raw"][tr]].astype(np.float32)                 # (Etr, K, T, H) slot labels
    ce = (X.mean(0) - X.mean((0, 1))).reshape(-1, H).astype(np.float64)
    Vt = np.linalg.svd(ce, full_matrices=False)[2]
    B, _ = np.linalg.qr(np.concatenate([Q, Vt[:10].T], 1))                        # (H, 28): Q, then the grid
    IT = np.load(os.path.join(SCAN, "figdata", "round6_hidden", "lesion", "trained_subspaces.npz"))["V_it"][:, :30]
    assert np.abs(B.T @ IT).max() < 1e-6
    return {"Q": B[:, :18], "G": B[:, 18:28], "IT": IT.astype(np.float64)}


def _prs(V, W):
    """participation ratios (tr C)^2 / tr(C^2) of the covariance of the rows of V (n, H), one per row of the weights
    W (m, n) (counts: a bootstrap resample; ones: the sample itself), from the Gram K = V V^T: with the weighted mean
    removed, Kc_ij = K_ij - a_i - a_j + b, tr C ~ sum_i w_i Kc_ii and tr C^2 ~ sum_ij w_i w_j Kc_ij^2 (exact)"""
    K = V @ V.T
    out = []
    for w in W:
        N = w.sum()
        a = K @ w / N
        Kc = K - a[:, None] - a[None] + w @ a / N
        out.append((w @ np.diag(Kc)) ** 2 / (w @ (Kc ** 2) @ w))
    return np.array(out)


def main():
    import fig1 as cf
    import fig1_movie as cm
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    BV, rg, rc = cm.beliefs_every_step()
    n = cf.N_EP
    rg, rc = np.asarray(rg)[:n].astype(int), np.asarray(rc)[:n].astype(int)
    ci = np.sort(cf._harvest("trained", ["ep_ctx_inds"])["ep_ctx_inds"][:n].astype(int), 1)
    keep = ci[:, 0] != ci[:, 1]
    Rv = np.arange(10.)
    X = {}
    for a in ("Trained", "Joint"):
        B = np.asarray(BV[a][:n], np.float64)
        X[a] = np.concatenate([np.full((n, 1, 2), 4.5), np.stack([B[..., :10] @ Rv, B[..., 10:] @ Rv], -1)], 1)
    mv = {a: np.linalg.norm(np.diff(X[a], axis=1), axis=-1) for a in X}           # (n, T): the move at step k
    J = mv["Trained"] > JUMP
    truth = np.stack([rg, rc], 1).astype(float)
    dtrue = np.linalg.norm(X["Trained"][:, 1:] - truth[:, None], axis=-1)        # (n, T) distance to the true cell
    win = lambda k: slice(k + LAGS[0], k + LAGS[-1] + 1)
    ev = [(i, k) for i, k in zip(*np.nonzero(J)) if keep[i] and K0 <= k <= K1 and J[i, win(k)].sum() == 1]
    rng = np.random.default_rng(0)
    cell = rg * 10 + rc
    by_cell = {c: np.flatnonzero(keep & (cell == c)) for c in np.unique(cell)}
    E, C = [], []
    calm = {k: np.flatnonzero(keep & ~J[:, win(k)].any(1)) for k in range(K0, K1 + 1)}
    for i, k in ev:
        if CTRL == "cell":
            cand = by_cell[cell[i]]
            cand = cand[(cand != i) & ~J[cand, win(k)].any(1)]
        else:
            cand = calm[k][np.abs(dtrue[calm[k], k - 1] - dtrue[i, k - 1]) < .5]
        if len(cand):
            E.append((i, k)), C.append((int(rng.choice(cand)), k))
    E, C = np.array(E), np.array(C)
    print(f"{len(ev)} isolated jumps at steps {K0 + 1}-{K1 + 1}; {len(E)} with a control; {time.time() - t0:.0f} s",
          flush=True)
    S = subspaces()
    Bfull = np.concatenate([S["Q"], S["G"], S["IT"]], 1)
    U = np.load(os.path.join(SCAN, "hidden_trained_u.npy"), mmap_mode="r")
    u0 = np.load(os.path.join(SCAN, "hidden_scan_cellmeans_trained.npz"))["u0"].astype(np.float64)
    mu = np.zeros((cm.T, H))                                                      # population mean per step
    kept = np.flatnonzero(keep)
    for c0 in range(0, len(kept), 2000):
        mu += U[kept[c0:c0 + 2000]].astype(np.float64).sum(0)
    mu /= len(kept)
    print(f"subspaces and means: {time.time() - t0:.0f} s", flush=True)

    def windows(P):
        """states u at lags L[0] - 1 .. L[-1] for each (episode, step) -> (n, len(L) + 1, H)"""
        out = np.zeros((len(P), len(LAGS) + 1, H))
        for r, (i, k) in enumerate(P):
            st = U[i].astype(np.float64)
            prev = np.concatenate([u0[None], st], 0)                             # index k + 1 = state after step k
            out[r] = prev[k + LAGS[0]:k + LAGS[-1] + 2]
        return out

    res = {"lags": LAGS, "events": E, "controls": C, "n": len(E)}
    kinds = {"all": np.ones(len(E), bool)}
    pre, post = dtrue[E[:, 0], E[:, 1] - 1], dtrue[E[:, 0], E[:, 1]]
    kinds["toward"], kinds["away"] = post < pre - JUMP, post > pre + JUMP
    res.update({f"n_{k}": int(v.sum()) for k, v in kinds.items()})
    for tag, P in (("ev", E), ("ct", C)):
        W = windows(P)                                                            # (n, L + 1, H)
        du = W[:, 1:] - W[:, :-1]                                                 # (n, L, H): update at each lag
        steps = P[:, 1:2] + LAGS[None]                                            # (n, L) absolute 0-based steps
        dev = W[:, 1:] - mu[steps]                                                # deviation from the step's mean
        res[f"{tag}_move"] = mv["Trained"][P[:, :1], steps]
        res[f"{tag}_move_opt"] = mv["Joint"][P[:, :1], steps]
        res[f"{tag}_dtrue"] = dtrue[P[:, :1], steps]
        for name, V in (("Q", S["Q"]), ("G", S["G"]), ("IT", S["IT"])):
            res[f"{tag}_dE_{name}"] = ((du @ V) ** 2).sum(-1)                     # (n, L)
            res[f"{tag}_sE_{name}"] = ((dev @ V) ** 2).sum(-1)
        res[f"{tag}_dE_REST"] = (du ** 2).sum(-1) - ((du @ Bfull) ** 2).sum(-1)
        res[f"{tag}_sE_REST"] = (dev ** 2).sum(-1) - ((dev @ Bfull) ** 2).sum(-1)
        nullQ = lambda Y: Y - (Y @ S["Q"]) @ S["Q"].T
        for kind, sel in kinds.items():
            if sel.sum() < 30:
                continue
            for qty, Y in (("du", du), ("u", dev)):
                for part, f in (("all", lambda y: y), ("null", nullQ)):
                    Wb = np.concatenate([np.ones((1, sel.sum())), rng.multinomial(sel.sum(), np.full(sel.sum(), 1 / sel.sum()),
                                                                                    NB).astype(float)])
                    prs = np.array([_prs(f(Y[sel, l]), Wb) for l in range(len(LAGS))])     # (L, 1 + NB)
                    pr, bs = prs[:, 0], prs[:, 1:].T
                    res[f"{tag}_{kind}_PR_{qty}_{part}"] = pr
                    res[f"{tag}_{kind}_PR_{qty}_{part}_ci"] = np.percentile(bs, [2.5, 97.5], 0)
        print(f"{tag}: done, {time.time() - t0:.0f} s", flush=True)
    np.savez(os.path.join(OUT, f"jump_dims_{CTRL}.npz"), **res)
    return res


def report(res):
    L = list(res["lags"])
    show = [L.index(l) for l in (-4, -2, -1, 0, 1, 2, 4, 6, 8, 10)]
    print("lag                 " + " ".join(f"{l:>7d}" for l in res["lags"][show]))
    rows = [("move (cells)", "move"), ("Optimal's move", "move_opt"), ("distance to truth", "dtrue")]
    rows += [(f"|du|^2 in {s}", f"dE_{s}") for s in ("Q", "G", "IT", "REST")]
    rows += [(f"|u - mu|^2 in {s}", f"sE_{s}") for s in ("Q", "G", "IT", "REST")]
    for lab, key in rows:
        for tag in ("ev", "ct"):
            v = res[f"{tag}_{key}"].mean(0)
            print(f"{lab + (' ev' if tag == 'ev' else ' ct'):24s}" + " ".join(f"{v[i]:7.2f}" for i in show))
    for kind in ("all", "toward", "away"):
        for qty in ("du", "u"):
            for part in ("all", "null"):
                k_ = f"{kind}_PR_{qty}_{part}"
                if f"ev_{k_}" not in res:
                    continue
                for tag in ("ev", "ct"):
                    v, ci = res[f"{tag}_{k_}"], res[f"{tag}_{k_}_ci"]
                    print(f"PR {qty:2s} {part:4s} {kind:6s} {tag}    " + " ".join(
                        f"{v[i]:5.1f}{'*' if tag == 'ev' and (ci[0, i] > res['ct_' + k_ + '_ci'][1, i] or ci[1, i] < res['ct_' + k_ + '_ci'][0, i]) else ' '} "
                        for i in show))
    print({k: int(res[k]) for k in ("n", "n_all", "n_toward", "n_away") if k in res})


if __name__ == "__main__":
    report(main())
