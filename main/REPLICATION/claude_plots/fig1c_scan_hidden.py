"""Replay the networks on the alias scan and keep what the jump / joint-correlation analyses need (27 Oct 2026; John's
hypothesis: the Trained RNN holds the joint correlation of the two variables outside its read-out, entangled evidence
changes it, and the splits use it) -> main/DATA/experimental/fig1c_compression/figdata/round7_jumps/scan_hidden_<net>.npz

Replay: as fig1c_order_check (the networks' own recurrence, no lesion), every environment of the scan (39; e = 2 left
out) x 5000 streams x 30 steps, in batches of 1000; checked against the scan's stored beliefs.
Subspaces of u (1000 units), per network: Q, the read-out's 18 directions; G, the shared grid's top 10 directions outside
Q (the cell effect of the scan's cell means, the lesion test's train environments); IT, the top 30 directions of the
environment-specific (cell x environment) interaction outside Q and G (fig1c_lesion_test.subspaces' rule); NULL, the top
64 principal directions of u outside Q (2000 of the harvest's episodes, every step centred on its own mean; their share
of the output-null variance is saved).
Saved per environment (float16 where large; 5000 streams x 30 steps unless stated):
    dE_<S>, sE_<S>  squared update |P_S du_t|^2 (du_t = u_t - u_{t-1}, u_0 the learned initial state) and squared
                    deviation |P_S (u_t - m_et)|^2 from the environment's mean at step t (the mean of the scan's 100 cell
                    means); S = Q, G, IT, ALL (no projection)
    turn, l2, pr    the gradflow figure's activity measures: 1 - cos(u_t, u_{t-1}); |u_t|; the effective dimension
                    (sum E)^2 / sum E^2, E_k = ((u_t - mu) . v_k)^2 on the network's fixed basis (gradflow_basis_<net>)
    Z               (5000, 30, 82) u_t - m_et on [Q | NULL]
    pair_u          (n_pairs, 2, 30, 1000) the full u of both streams of the split movie's pairs in this environment
    dec_y, dec_<F>  the decoding (steps 2-16, (5000, 15)): the target, the Optimal's exact joint covariance
                    Cov(r_g, r_c) (cells^2; the exact posterior over the 100 cells from the stream's observations and the
                    table), and its 5-fold cross-validated ridge predictions (folds by stream: stream % 5; X and y centred
                    per step; the ridge strength c tr(G)/p, c in 1e-4 .. 1, chosen per feature set by the pooled CV R^2)
                    from feature sets F: marg, the network's own two marginals (20) and their outer product (100, its
                    factorized joint); opt, the same from the Optimal's marginals; null, the output-null part of u,
                    (I - QQ^T)(u - m_et) (1000); marg_null and opt_null, the unions
Run from the repository root (about 25 min per network on 6 CPU threads; the two in parallel):
    CUDA_VISIBLE_DEVICES="" /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/fig1c_scan_hidden.py trained   (and: echo; SMOKE=1: two environments only)"""
import os
import sys
import time
import numpy as np
import torch

NET = sys.argv[1] if len(sys.argv) > 1 else "trained"
SMOKE = os.environ.get("SMOKE") == "1"
torch.set_num_threads(int(os.environ.get("NTHREADS", "6")))
ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.chdir(ROOT)
import fig1 as cf                                                          # noqa: E402

ga = cf.ga
SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
OUT = os.path.join(SCAN, "figdata", "round7_jumps")
AGENT = {"trained": "Trained", "echo": "Echo"}[NET]
M, R, T, H, B, KN = 50, 10, 30, 1000, 1000, 64
DEC_STEPS = np.arange(1, 16)                                                      # 0-based: steps 2-16
CS = (1e-4, 1e-3, 1e-2, 1e-1, 1.)
ENVS = [e for e in range(40) if e != 2]
TRAIN_POS = [p for p in range(39) if p not in range(0, 39, 4)]


def subspaces():
    """Q (H, 18), G (H, 10), IT (H, 30) from the scan's cell means; NULL (H, KN) from the harvest's hidden states"""
    f = np.load(os.path.join(SCAN, f"hidden_scan_cellmeans_{NET}.npz"))
    assert list(f["env_index"]) == ENVS
    Q = f["Q"].astype(np.float64)
    U_cm = f["u"]
    tr = np.array(TRAIN_POS)
    X = U_cm[tr[:, None], f["perm_raw"][tr]].astype(np.float32)                    # (Etr, K, T, H) slot labels
    ce = (X.mean(0) - X.mean((0, 1))).reshape(-1, H).astype(np.float64)
    Bx, _ = np.linalg.qr(np.concatenate([Q, np.linalg.svd(ce, full_matrices=False)[2][:10].T], 1))
    Pc = np.eye(H) - Bx @ Bx.T
    it = (X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean((0, 1), keepdims=True)).reshape(-1, H)
    C = Pc @ (it.T.astype(np.float64) @ it.astype(np.float64)) @ Pc
    IT = np.linalg.eigh((C + C.T) / 2)[1][:, ::-1][:, :30]
    m_env = U_cm.astype(np.float32).mean(1)                                        # (E, T, H) environment means
    U = np.load(os.path.join(SCAN, f"hidden_{NET}_u.npy"), mmap_mode="r")
    sel = np.sort(np.random.default_rng(0).choice(len(U), 2000, replace=False))
    Y = U[sel].astype(np.float64)                                                  # (2000, T, H)
    Y = (Y - Y.mean(0)).reshape(-1, H)
    Y = Y - (Y @ Q) @ Q.T
    w, V = np.linalg.eigh(Y.T @ Y)
    w, V = w[::-1], V[:, ::-1]
    bas = np.load(os.path.join(ROOT, "main", "DATA", "cache", f"gradflow_basis_{NET}.npz"))
    return dict(Q=Q, G=Bx[:, 18:28], IT=IT, NULL=V[:, :KN], null_share=w[:KN].sum() / w.sum(), m_env=m_env,
                u0=f["u0"].astype(np.float64), bmu=bas["mu"].astype(np.float64), bV=bas["V"].astype(np.float64))


def build_model():
    from main.CognitiveGridworld import CognitiveGridworld
    load_env, reservoir = ga.AGENTS[NET]
    agent = CognitiveGridworld(**dict(ga.CFG, episodes=1), reservoir=reservoir, load_env=load_env,
                               external_logger=lambda env: {})
    assert agent.model.batch_num == B and agent.model.step_num == T
    return agent, agent.model


def replay(m, W, obs, inp, g, dtype):
    """u (n, T, H) and the goal-ordered beliefs (n, T, 20) for observations obs (n, T, 5)"""
    W_ih, W_hh, b, Wr, br = W
    n = len(obs)
    m.obs_flat = torch.as_tensor(obs.astype(dtype))
    m.active_Z, m.goal_ind, m.joint_goal_belief = (torch.as_tensor(np.repeat(v, n, 0)) for v in inp)
    with torch.no_grad(), torch.autocast("cpu", enabled=False):
        xa = ga.lstm_input(m) @ W_ih.T + b
        u = m.STM.detach().float().reshape(1, H).expand(n, H)
        c = m.LTM.detach().float().reshape(1, H).expand(n, H)
        us = []
        for s in range(T):
            i, f_, g_, o = (xa[:, s] + u @ W_hh.T).chunk(4, -1)
            c = f_.sigmoid() * c + i.sigmoid() * g_.tanh()
            u = o.sigmoid() * c.tanh()
            us.append(u)
        U = torch.stack(us, 1)
        q = (U @ Wr.T + br).double().cumsum(1).reshape(n, T, 2, R).softmax(-1)
        q = torch.stack([q[:, :, g], q[:, :, 1 - g]], 2).reshape(n, T, 2 * R)
    return U.numpy().astype(np.float64), q.numpy()


def joint_posterior(L, g, obs):
    """the Optimal's exact posterior over the 100 cells (goal-ordered (r_g, r_c)) at every step -> its two marginals
    (n, T, 20) and the covariance Cov(r_g, r_c) (n, T)"""
    o = obs.astype(np.float64)
    ll = (np.einsum("nto,oab->ntab", o, np.log(L)) + np.einsum("nto,oab->ntab", 1 - o, np.log1p(-L))).cumsum(1)
    p = np.exp(ll - ll.max((2, 3), keepdims=True))
    p /= p.sum((2, 3), keepdims=True)
    if g == 1:
        p = p.transpose(0, 1, 3, 2)
    r = np.arange(R, dtype=np.float64)
    mg, mc = p.sum(3), p.sum(2)
    eg, ec = mg @ r, mc @ r
    cov = np.einsum("ntab,a,b->nt", p, r, r) - eg * ec
    return np.concatenate([mg, mc], -1), cov


def decode(feats, y, fold):
    """5-fold ridge with the folds' Grams (X, y centred per step beforehand; X (n, p) per set) -> the CV predictions
    of the best ridge strength per set and the chosen strengths"""
    out, chosen = {}, {}
    for name, X in feats.items():
        G = X.T @ X
        Gf = [X[fold == k].T @ X[fold == k] for k in range(5)]
        bf = [X[fold == k].T @ y[fold == k] for k in range(5)]
        b = X.T @ y
        best = None
        for c in CS:
            pred = np.zeros_like(y)
            for k in range(5):
                Gt, bt = G - Gf[k], b - bf[k]
                lam = c * np.trace(Gt) / len(Gt)
                w = np.linalg.solve(Gt + lam * np.eye(len(Gt)), bt)
                pred[fold == k] = X[fold == k] @ w
            r2 = 1 - ((y - pred) ** 2).sum() / (y ** 2).sum()
            if best is None or r2 > best[0]:
                best = (r2, pred, c)
        out[name], chosen[name] = best[1].astype(np.float32), best[2]
    return out, chosen


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    S = subspaces()
    print(f"{NET}: subspaces in {time.time() - t0:.0f} s; NULL holds {100 * S['null_share']:.1f}% of the output-null "
          f"variance", flush=True)
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    eps, goal, obs_all, L_all = meta["eps"].astype(int), meta["goal"].astype(int), meta["obs"], meta["L"]
    with np.load(ga.CACHE.format("trained")) as f:
        inp_all = [f[k][eps] for k in cf.REPLAY[1:]]
        dtype = f["ep_obs_flat"].dtype
    Bscan = np.load(os.path.join(SCAN, f"scan_{AGENT}.npz"))["beliefs"]
    sp = np.load(os.path.join(SCAN, "figdata", "round4b", "split_pairs.npz"))
    agent, m = build_model()
    W = (*ga.lstm_weights(m), m.classifier_readout.weight.detach().float(), m.classifier_readout.bias.detach().float())
    basis = {"Q": S["Q"], "G": S["G"], "IT": S["IT"]}
    zb = np.concatenate([S["Q"], S["NULL"]], 1)                                    # (H, 82)
    Pn = np.eye(H) - S["Q"] @ S["Q"].T
    out = {k: S[k] for k in ("Q", "G", "IT", "NULL")}
    out["null_share"] = S["null_share"]
    n_all = M * R * R
    fold = np.repeat((np.arange(n_all) % 5)[:, None], len(DEC_STEPS), 1).ravel()
    check, chosen_all = 0., []
    for p, e in enumerate(ENVS[:2] if SMOKE else ENVS):
        te = time.time()
        res = {f"{a}_{k}": np.zeros((n_all, T), np.float32) for a in ("dE", "sE") for k in ("Q", "G", "IT", "ALL")}
        for k in ("turn", "l2", "pr"):
            res[k] = np.zeros((n_all, T), np.float32)
        Z = np.zeros((n_all, T, zb.shape[1]), np.float16)
        pin = sp["env"] == e
        pair_rows = np.stack([sp["i"][pin], sp["j"][pin]], 1)
        pair_u = np.zeros((len(pair_rows), 2, T, H), np.float16)
        X_null = np.zeros((n_all, len(DEC_STEPS), H), np.float32)
        q_net = np.zeros((n_all, len(DEC_STEPS), 2 * R))
        for bt in range(n_all // B):
            rows = np.arange(bt * B, (bt + 1) * B)
            U, q = replay(m, W, obs_all[e, rows], [v[e:e + 1] for v in inp_all], int(goal[e]), dtype)
            check = max(check, float(np.abs(q - Bscan[e, rows].astype(np.float64)).max()))
            prev = np.concatenate([np.broadcast_to(S["u0"], (B, 1, H)), U[:, :-1]], 1)
            du, dev = U - prev, U - S["m_env"][p][None].astype(np.float64)
            for k, V in basis.items():
                res[f"dE_{k}"][rows] = ((du @ V) ** 2).sum(-1)
                res[f"sE_{k}"][rows] = ((dev @ V) ** 2).sum(-1)
            res["dE_ALL"][rows], res["sE_ALL"][rows] = (du ** 2).sum(-1), (dev ** 2).sum(-1)
            nu, npv = np.linalg.norm(U, axis=-1), np.linalg.norm(prev, axis=-1)
            res["turn"][rows], res["l2"][rows] = 1 - (U * prev).sum(-1) / (nu * npv), nu
            Ep = ((U - S["bmu"]) @ S["bV"]) ** 2
            res["pr"][rows] = Ep.sum(-1) ** 2 / np.maximum((Ep ** 2).sum(-1), 1e-12)
            Z[rows] = (dev @ zb).astype(np.float16)
            X_null[rows] = (dev[:, DEC_STEPS] @ Pn).astype(np.float32)
            q_net[rows] = q[:, DEC_STEPS]
            for r_, (i_, j_) in enumerate(pair_rows):
                for k_, s_ in enumerate((i_, j_)):
                    if bt * B <= s_ < (bt + 1) * B:
                        pair_u[r_, k_] = U[s_ - bt * B].astype(np.float16)
        t_rep = time.time() - te
        # the decoding: the Optimal's joint covariance from marginals, with and without the output-null u
        q_opt, cov = joint_posterior(L_all[e], int(goal[e]), obs_all[e])
        q_opt = q_opt[:, DEC_STEPS]
        y = cov[:, DEC_STEPS]
        outer = lambda q: np.einsum("nsa,nsb->nsab", q[..., :R], q[..., R:]).reshape(*q.shape[:2], R * R)
        Fm = np.concatenate([q_net, outer(q_net)], -1)
        Fo = np.concatenate([q_opt, outer(q_opt)], -1)
        cen = lambda A: (A - A.mean(0, keepdims=True)).reshape(-1, A.shape[-1]) if A.ndim == 3 else \
            (A - A.mean(0, keepdims=True)).ravel()
        Fm, Fo, Xn, yc = cen(Fm), cen(Fo), cen(X_null.astype(np.float64)), cen(y)
        feats = {"marg": Fm, "opt": Fo, "null": Xn, "marg_null": np.concatenate([Fm, Xn], 1),
                 "opt_null": np.concatenate([Fo, Xn], 1)}
        preds, chosen = decode(feats, yc, fold)
        chosen_all.append([chosen[k] for k in feats])
        for k_, v in preds.items():
            out[f"dec_{k_}_e{e:02d}"] = v.reshape(n_all, len(DEC_STEPS))
        out[f"dec_y_e{e:02d}"] = yc.reshape(n_all, len(DEC_STEPS)).astype(np.float32)
        for k, v in res.items():
            out[f"{k}_e{e:02d}"] = v.astype(np.float16) if k.startswith(("dE", "sE")) else v
        out[f"Z_e{e:02d}"], out[f"pair_u_e{e:02d}"], out[f"pair_rows_e{e:02d}"] = Z, pair_u, pair_rows
        r2 = {k_: 1 - ((yc - preds[k_]) ** 2).sum() / (yc ** 2).sum() for k_ in preds}
        print(f"{NET} env {e} ({p + 1}/39): replay {t_rep:.0f} s, total {time.time() - te:.0f} s; check {check:.1e}; "
              "CV R^2 " + ", ".join(f"{k_} {v:.3f}" for k_, v in r2.items()) +
              f"; elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    out["check"], out["dec_c"] = check, np.array(chosen_all)
    np.savez(os.path.join(OUT, f"scan_hidden_{NET}{'_smoke' if SMOKE else ''}.npz"), **out)
    print(f"saved; {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
