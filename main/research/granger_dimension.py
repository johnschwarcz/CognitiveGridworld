"""
Granger causality analysis — PC-to-PC directed information flow.
Stripped-down version: SVD → Granger → 4-panel plot.
"""
import numpy as np, torch, os, sys, inspect, time
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("root:", path)
sys.path.insert(0, path + '/main'); sys.path.insert(0, path + '/main/bayes'); sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld

if __name__ == "__main__":
    cuda = 0; realization_num = 10; step_num = 30; hid_dim = 1000
    state_num = 500; obs_num = 5; batch_num = 8000; episodes = 1
    _common = dict(mode="SANITY", cuda=cuda, episodes=episodes, checkpoint_every=5,
        realization_num=realization_num, hid_dim=hid_dim, obs_num=obs_num, show_plots=False,
        batch_num=batch_num, step_num=step_num, state_num=state_num, learn_embeddings=False,
        classifier_LR=.001, ctx_num=2, training=False)
    echo = CognitiveGridworld(**{**_common, 'reservoir': True, 'load_env': "/sanity/reservoir_ctx_2_e5"})
    trained = CognitiveGridworld(**{**_common, 'reservoir': False, 'load_env': "/sanity/fully_trained_ctx_2_e5"})

    PARAMS = {"K_PCA": None, "K_CAUSAL": None, "LAG_MAX": 29,
        "CAUSAL_RIDGE": 1e-6, "CAUSAL_MAX_ROWS": 60000,
        "FIG_DPI": 120, "MIN_YABS": 0.1, "SHOW_TQDM": True}

    # ── Helpers ──
    def _to_np(x, dtype=None):
        a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        return a if dtype is None else a.astype(dtype, copy=False)

    def _z3(x):
        m = np.nanmean(x, axis=(0, 1), keepdims=True)
        return (x - m) / (np.nanstd(x, axis=(0, 1), keepdims=True) + 1e-12)

    def _min_ylim(ax, min_abs=0.1, pad=0.08):
        y0, y1 = ax.get_ylim()
        lo, hi = min(y0, y1), max(y0, y1)
        lo = min(lo, -min_abs); hi = max(hi, min_abs)
        span = max(hi - lo, 2 * min_abs); d = pad * span
        ax.set_ylim(lo - d, hi + d)

    def _iter(it, use_tqdm, desc=""):
        return tqdm(it, desc=desc, leave=False) if use_tqdm else it

    # ── SVD (cached) ──
    def pca_from_updates(model, k_pca=None):
        upd = _to_np(model.model_update_flat, np.float32)
        B, T, N = upd.shape
        cache = getattr(model, '_svd_cache', None)
        if cache is not None and cache['shape'] == (B, T, N):
            x, s, vt = cache['x'], cache['s'], cache['vt']
        else:
            x = np.ascontiguousarray(upd.reshape(B * T, N), dtype=np.float64)
            x = np.nan_to_num(x, nan=0., posinf=0., neginf=0.)
            x -= x.mean(0, keepdims=True)
            _, s, vt = np.linalg.svd(x, full_matrices=False)
            try: model._svd_cache = {'shape': (B, T, N), 'x': x, 's': s, 'vt': vt}
            except AttributeError: pass
        k = vt.shape[0] if k_pca is None else int(min(max(1, k_pca), vt.shape[0]))
        z = (x @ vt[:k].T).reshape(B, T, k).astype(np.float32, copy=False)
        return z

    # ── Granger causality ──
    def _lag_design(z_bt, L):
        B, T, K = z_bt.shape; M = T - L
        w = sliding_window_view(z_bt, L, axis=1)[:, :M]
        X = w.transpose(0, 1, 3, 2).reshape(B * M, K * L).astype(np.float64)
        Y = z_bt[:, L:L + M, :].reshape(B * M, K).astype(np.float64)
        return X, Y

    def _causal_subsample(z_bt, L, max_rows):
        B, T, _ = z_bt.shape; M = T - L
        if M <= 1: return z_bt[:1]
        b_use = max(1, min(B, max_rows // M))
        if b_use >= B: return z_bt
        return z_bt[np.linspace(0, B - 1, b_use, dtype=np.int64)]

    def granger_matrix(z_bt, L, ridge=1e-6, max_rows=60000, use_tqdm=False):
        z = np.asarray(z_bt, np.float64); _, T, K = z.shape
        if L >= T - 1: return np.zeros((K, K), np.float32)
        z = _causal_subsample(z, L, max_rows)
        X, Y = _lag_design(z, L)
        X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
        Y = (Y - Y.mean(0, keepdims=True)) / (Y.std(0, keepdims=True) + 1e-6)
        GG = X.T @ X; HH = X.T @ Y; YY = (Y * Y).sum(0)
        out = np.zeros((K, K), np.float32)
        base = np.arange(L, dtype=np.int64) * K
        I_L = np.eye(L, dtype=np.float64) * ridge
        # Precompute per-PC diagonal blocks and inverses
        Aii = np.empty((K, L, L), np.float64); Aii_inv = np.empty((K, L, L), np.float64)
        idx_pc = base[:, None] + np.arange(K)  # (L, K)
        for i in range(K):
            Aii[i] = GG[np.ix_(idx_pc[:, i], idx_pc[:, i])] + I_L
            Aii_inv[i] = np.linalg.inv(Aii[i])
        # Batched: for each target j, solve all K-1 sources in one call
        for j in _iter(range(K), use_tqdm, desc="Granger targets"):
            bt = HH[idx_pc[:, j], j]; v = Aii_inv[j] @ bt
            rss_t = YY[j] - bt @ v
            if rss_t <= 1e-12: continue
            srcs = np.concatenate([np.arange(j, dtype=np.int64), np.arange(j + 1, K, dtype=np.int64)])
            ns = srcs.size
            if ns == 0: continue
            idx_j = idx_pc[:, j]; idx_srcs = idx_pc[:, srcs]
            Ait_all = GG[idx_srcs.T[:, :, None], idx_j[None, None, :]]
            h_srcs = HH[idx_srcs, j].T
            u_stack = h_srcs - np.einsum('nij,j->ni', Ait_all, v)
            AiA = np.einsum('nij,jk->nik', Ait_all, Aii_inv[j])
            S_stack = Aii[srcs] - np.einsum('nij,nkj->nik', AiA, Ait_all) + I_L
            try:
                sol = np.linalg.solve(S_stack, u_stack)
            except np.linalg.LinAlgError:
                sol = np.zeros_like(u_stack)
                for si in range(ns):
                    try: sol[si] = np.linalg.solve(S_stack[si], u_stack[si])
                    except np.linalg.LinAlgError: sol[si] = np.linalg.lstsq(S_stack[si], u_stack[si], rcond=None)[0]
            vals = np.einsum('ni,ni->n', u_stack, sol) / (rss_t + 1e-12)
            pos = vals > 0
            out[srcs[pos], j] = vals[pos].astype(np.float32)
        return out

    def causal_totals_by_pc(M):
        K = M.shape[0]; diag = np.diag(M)
        cs_in = np.cumsum(M, 0); cs_out = np.cumsum(M, 1)
        in_lo = np.zeros(K, np.float32); out_lo = np.zeros(K, np.float32)
        if K > 1:
            r = np.arange(K - 1, dtype=np.int64)
            in_lo[1:] = cs_in[r, r + 1]; out_lo[1:] = cs_out[r + 1, r]
        return {"in_from_lower": in_lo, "in_from_higher": (cs_in[-1] - in_lo - diag).astype(np.float32, copy=False),
                "out_to_lower": out_lo, "out_to_higher": (cs_out[:, -1] - out_lo - diag).astype(np.float32, copy=False), "K": K}

    # ── Run + Plot ──
    def run_granger(model, prm):
        t0 = time.time()
        z_all = pca_from_updates(model, prm["K_PCA"])
        print(f"  SVD: {time.time()-t0:.2f}s  K={z_all.shape[2]}")
        k_c = z_all.shape[2] if prm["K_CAUSAL"] is None else int(min(max(2, prm["K_CAUSAL"]), z_all.shape[2]))
        z_c = _z3(z_all[:, :, :k_c]).astype(np.float32, copy=False)
        lag = int(min(max(1, prm["LAG_MAX"]), z_c.shape[1] - 2))
        t1 = time.time()
        M = granger_matrix(z_c, lag, prm["CAUSAL_RIDGE"], prm["CAUSAL_MAX_ROWS"], prm["SHOW_TQDM"])
        print(f"  Granger (K={k_c}, L={lag}): {time.time()-t1:.2f}s")
        print(f"  TOTAL: {time.time()-t0:.2f}s")
        return {"matrix": M, "totals": causal_totals_by_pc(M), "lag": lag, "k": k_c}

    def plot_granger(g_tr, g_ec, prm):
        fig, ax = plt.subplots(1, 4, figsize=(20, 4), dpi=prm["FIG_DPI"])
        keys = ("in_from_lower", "in_from_higher", "out_to_lower", "out_to_higher")
        titles = ("in from lower", "in from higher", "out to lower", "out to higher")
        tr, ec = g_tr["totals"], g_ec["totals"]
        x = np.arange(min(tr["K"], ec["K"]), dtype=np.int64)
        for j in range(4):
            a = ax[j]
            a.plot(x, tr[keys[j]][:x.size], lw=2, color="C0", label="trained")
            a.plot(x, ec[keys[j]][:x.size], lw=2, color="C1", label="echo")
            a.axhline(0, color="k", lw=1, alpha=0.4); a.grid(alpha=0.25)
            a.set_title(titles[j]); a.set_xlabel("PC index (0 = highest-variance PC)")
            if j == 0: a.set_ylabel("total directed ΔR²")
            _min_ylim(a, prm["MIN_YABS"])
        ax[0].legend(frameon=False, fontsize=9)
        fig.tight_layout()
        return fig

    def run_all(trained, echo, params=None):
        prm = dict(PARAMS)
        if params is not None: prm.update(params)
        print("=== Granger: trained ===")
        g_tr = run_granger(trained, prm)
        print("=== Granger: echo ===")
        g_ec = run_granger(echo, prm)
        fig = plot_granger(g_tr, g_ec, prm)
        print(f"done | K tr/ec={g_tr['k']}/{g_ec['k']} lag tr/ec={g_tr['lag']}/{g_ec['lag']}")
        return {"trained": g_tr, "echo": g_ec, "fig": fig, "params": prm}

    # =========================
    out = run_all(trained, echo)
    plt.show()
