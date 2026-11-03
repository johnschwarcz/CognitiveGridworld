"""Lesion test (25 Oct 2026; data for plot_fig1c_lesion.py): does the trained network need its environment-specific
hidden dimensions to keep one context-invariant grid? During the recurrence, after every step, the LSTM output u_t
(what the read-out reads and what drives the next step's gates) is clamped along a k-dimensional subspace V to the
population mean:
    u_t <- u_t - (u_t - mu_t) V V^T
V is orthogonal to the read-out's 18 directions Q (the read-out at the same step is untouched: any effect on the
beliefs comes through the dynamics) and to the shared grid's top 10 directions. Subspaces, from the TRAIN environments'
hidden cell means (hidden_scan_cellmeans_<net>.npz, slot labels); the lesioned network runs on the TEST environments:
    it    top-k PCs of the cell x environment interaction (the environment-specific distortion)
    ee    top-k PCs of the environment effect (whole-grid shift per environment)
    w     top-k PCs of the within-cell (stream-to-stream) variability (scatter matrices of research9/hidden/parts_<net>)
    rand  a random k-dimensional subspace (k = 30, 60)
    rv    a second random subspace, k = 260 / 400 directions: matched in variance to it30 / it60 (about 20% / 31% of
          u's stream-level variance)
Saved per condition and test environment: the cell means of the beliefs (goal-ordered, 100 x T x 20), each stream's
belief at its true cell at T, and the cell means of the update sizes |u_g(t)|, |u_c(t)| and their within-cell
covariance (joint movement).
Run from the repository root (about 1 min per condition and environment on 3 CPU threads; the 10 conditions of the
figure ran as 4 + 4 + 1 + 1 in parallel processes, about 10 min):
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/fig1c_lesion_test.py trained none,it10,it30,it60 3 10
    (further condition sets: ee30,w30,rand30,rand60 and rv260 and rv400; add "smoke" as a 5th argument for one batch)"""
import os
import sys
import time
import json
import numpy as np
import torch

NET = sys.argv[1]
CONDS = sys.argv[2].split(",")
NTHR = int(sys.argv[3]) if len(sys.argv) > 3 else 3
NTEST = int(sys.argv[4]) if len(sys.argv) > 4 else 10
SMOKE = len(sys.argv) > 5 and sys.argv[5] == "smoke"
torch.set_num_threads(NTHR)
ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.chdir(ROOT)
import fig1 as cf                                                          # noqa: E402

ga = cf.ga
SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
SCR = "/tmp/claude-1002/-home-johns-Desktop-CognitiveGridworld/7af6b091-54ff-45a0-b4dd-ef0a3bc9c778/scratchpad"
PARTS = os.path.join(SCR, "research9", "hidden", f"parts_{NET}")                 # within-cell scatter (replay_scan.py)
OUT = os.path.join(SCAN, "figdata", "round6_hidden", "lesion")
SUBSPACES = os.path.join(OUT, f"{NET}_subspaces.npz")                             # saved V and mu_t (no PARTS needed)
AGENT = {"trained": "Trained", "echo": "Echo"}[NET]
M, K, T, H, R = 50, 100, 30, 1000, 10
CPB = 20
ENVS = [e for e in range(40) if e != 2]
TEST_POS = list(range(0, 39, 4))[:NTEST]                                          # positions in ENVS
TRAIN_POS = [p for p in range(39) if p not in range(0, 39, 4)]


def subspaces(kmax=60):
    f = np.load(os.path.join(SCAN, f"hidden_scan_cellmeans_{NET}.npz"))
    assert list(f["env_index"]) == ENVS
    perm, Q = f["perm_raw"], f["Q"].astype(np.float64)
    tr = np.array(TRAIN_POS)
    X = f["u"][tr[:, None], perm[tr]].astype(np.float32)                           # (Etr, K, T, H) slot labels
    mu_t = X.mean((0, 1)).astype(np.float64)                                       # (T, H)
    ce = (X.mean(0) - X.mean((0, 1))).reshape(-1, H).astype(np.float64)
    _, _, Vt = np.linalg.svd(ce, full_matrices=False)
    Bx, _ = np.linalg.qr(np.concatenate([Q, Vt[:10].T], 1))                        # (H, 28) excluded directions
    Pc = np.eye(H) - Bx @ Bx.T

    def top(C, k):
        C = Pc @ C @ Pc
        w, V = np.linalg.eigh((C + C.T) / 2)
        return V[:, ::-1][:, :k], w[::-1]

    it = X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean((0, 1), keepdims=True)
    it = it.reshape(-1, H).astype(np.float64)
    Vit, wit = top(it.T @ it, kmax)
    ee = (X.mean(1) - X.mean((0, 1))).reshape(-1, H).astype(np.float64)
    Vee, wee = top(ee.T @ ee, kmax)
    S = sum(np.load(os.path.join(PARTS, f"env_{ENVS[p]:02d}.npz"))["S_u"].astype(np.float64).sum(0) for p in TRAIN_POS)
    Vw, ww = top(S, kmax)
    rng = np.random.default_rng(0)
    Vr, _ = np.linalg.qr(Pc @ rng.standard_normal((H, kmax)))
    Vrv, _ = np.linalg.qr(Pc @ np.random.default_rng(1).standard_normal((H, 400)))   # variance-matched random
    tot = {"it": it, "ee": ee}
    info = {}
    for name, V in (("it", Vit), ("ee", Vee), ("w", Vw), ("rand", Vr), ("rv", Vrv)):
        for k in ((10, 30, 60) if name != "rv" else (260, 400)):
            info[f"{name}{k}"] = {"in_Q": float(np.linalg.norm(Q.T @ V[:, :k]) ** 2 / k),
                                  **{f"share_of_{c}": float(((tot[c] @ V[:, :k]) ** 2).sum() / (tot[c] ** 2).sum())
                                     for c in tot}}
    return dict(it=Vit, ee=Vee, w=Vw, rand=Vr, rv=Vrv), mu_t, info


def build_model():
    from main.CognitiveGridworld import CognitiveGridworld
    load_env, reservoir = ga.AGENTS[NET]
    agent = CognitiveGridworld(**dict(ga.CFG, episodes=1), reservoir=reservoir, load_env=load_env,
                               external_logger=lambda env: {})
    m = agent.model
    assert m.batch_num == M * CPB and m.step_num == T
    return agent, m


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if os.path.exists(SUBSPACES):
        f = np.load(SUBSPACES)
        V, mu_t = {k[2:]: f[k].astype(np.float64) for k in f.files if k.startswith("V_")}, f["mu_t"]
        info = json.load(open(os.path.join(OUT, f"{NET}_subspace_info.json")))
    else:
        V, mu_t, info = subspaces()
    print(f"{NET}: subspaces in {time.time() - t0:.0f} s; " + json.dumps(
        {c: {k: round(v, 4) for k, v in info[c].items()} for c in info if c.endswith("30")}), flush=True)
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    obs_all, eps, goal = meta["obs"], meta["eps"].astype(int), meta["goal"].astype(int)
    cells = meta["cells"]
    with np.load(ga.CACHE.format("trained")) as f:
        inp = {k: f[k][eps] for k in cf.REPLAY[1:]}
        obs_dtype = f["ep_obs_flat"].dtype
    Bscan = np.load(os.path.join(SCAN, f"scan_{AGENT}.npz"))["beliefs"]
    agent, m = build_model()
    dev = m.STM.device
    W_ih, W_hh, b = ga.lstm_weights(m)
    Wr = m.classifier_readout.weight.detach().float()
    br = m.classifier_readout.bias.detach().float()
    mu = torch.as_tensor(mu_t, dtype=torch.float32)
    P = {}
    for c in CONDS:
        if c != "none":
            name, k = c.rstrip("0123456789"), int(c[len(c.rstrip("0123456789")):])
            Vk = torch.as_tensor(np.ascontiguousarray(V[name][:, :k]), dtype=torch.float32)
            P[c] = Vk
    Rv = torch.arange(R, dtype=torch.float64)
    for p in TEST_POS:
        e = ENVS[p]
        g = int(goal[e])
        res = {c: dict(mean_q=np.zeros((K, T, 2 * R)), p_true=np.zeros((K * M,)), map_ok=np.zeros((K * M,)),
                       size_mean=np.zeros((K, T, 2)), jm=np.zeros((K, T)), check=0.) for c in CONDS}
        t_env = time.time()
        for bt in range(K // CPB if not SMOKE else 1):
            rows = slice(bt * M * CPB, (bt + 1) * M * CPB)
            cl = slice(bt * CPB, (bt + 1) * CPB)
            n = M * CPB
            m.obs_flat = torch.as_tensor(obs_all[e, rows].astype(obs_dtype), device=dev)
            m.active_Z, m.goal_ind, m.joint_goal_belief = (torch.as_tensor(np.repeat(inp[k][e:e + 1], n, 0), device=dev)
                                                           for k in cf.REPLAY[1:])
            with torch.no_grad(), torch.autocast(dev.type, enabled=False):
                x = ga.lstm_input(m)
                xa = x @ W_ih.T + b                                                # shared by every condition
                for c in CONDS:
                    u = m.STM.detach().float().reshape(1, H).expand(n, H)
                    cs = m.LTM.detach().float().reshape(1, H).expand(n, H)
                    rs = []
                    for s in range(T):
                        i, f_, g_, o = (xa[:, s] + u @ W_hh.T).chunk(4, -1)
                        cs = f_.sigmoid() * cs + i.sigmoid() * g_.tanh()
                        u = o.sigmoid() * cs.tanh()
                        if c != "none":
                            u = u - ((u - mu[s]) @ P[c]) @ P[c].T
                        rs.append(u @ Wr.T + br)
                    r = torch.stack(rs, 1).double()
                    q = r.cumsum(1).reshape(n, T, 2, R).softmax(-1)               # slot order
                    qg = torch.stack([q[:, :, g], q[:, :, 1 - g]], 2)             # goal-ordered (as the scan)
                    if c == "none":
                        res[c]["check"] = max(res[c]["check"], float(np.abs(qg.reshape(n, T, 2 * R).numpy()
                                                                            - Bscan[e, rows].astype(np.float64)).max()))
                    qn = qg.numpy()
                    res[c]["mean_q"][cl] = qn.reshape(CPB, M, T, 2 * R).mean(1)
                    tg = cells[np.repeat(np.arange(bt * CPB, (bt + 1) * CPB), M)]  # (n, 2) true (r_g, r_c)
                    ar = np.arange(n)
                    pt = qn[ar, -1, 0, tg[:, 0]] * qn[ar, -1, 1, tg[:, 1]]
                    res[c]["p_true"][rows] = pt
                    res[c]["map_ok"][rows] = ((qn[:, -1, 0].argmax(-1) == tg[:, 0]) & (qn[:, -1, 1].argmax(-1) == tg[:, 1]))
                    Ex = torch.cat([torch.full((n, 1, 2), 4.5, dtype=torch.float64), (qg * Rv).sum(-1)], 1)
                    us = (Ex[:, 1:] - Ex[:, :-1]).abs().numpy().reshape(CPB, M, T, 2)
                    res[c]["size_mean"][cl] = us.mean(1)
                    uc = us - us.mean(1, keepdims=True)
                    res[c]["jm"][cl] = (uc[..., 0] * uc[..., 1]).sum(1) / (M - 1)
        out = {f"{c}__{k}": v for c in CONDS for k, v in res[c].items()}
        tag = "smoke" if SMOKE else "full"
        np.savez(os.path.join(OUT, f"{NET}_{tag}_env{e:02d}_{'-'.join(CONDS)}.npz"), **out)
        print(f"{NET} env {e} ({TEST_POS.index(p) + 1}/{len(TEST_POS)}): {time.time() - t_env:.0f} s, elapsed "
              f"{(time.time() - t0) / 60:.1f} min; baseline check {res.get('none', {}).get('check', float('nan')):.2e}; "
              + "; ".join(f"{c}: P(true cell) {res[c]['p_true'][:(n if SMOKE else None)].mean():.3f}" for c in CONDS),
              flush=True)
    fn = os.path.join(OUT, f"{NET}_subspace_info.json")
    old = json.load(open(fn)) if os.path.exists(fn) else {}
    json.dump({**old, **info}, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main()
