import numpy as np; import torch; import os; import sys; import inspect; from matplotlib.colors import PowerNorm
from tqdm import tqdm; 
import matplotlib.pyplot as plt; 
from matplotlib.colors import Normalize; from matplotlib.cm import ScalarMappable
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld

if __name__ == "__main__":
    cuda = 0
    realization_num = 10
    step_num = 30
    hid_dim = 1000
    state_num = 500
    obs_num = 5
    batch_num = 8000
    episodes = 1

    echo = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5, 
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 2, 
        'training': False, 'load_env': "/sanity/reservoir_ctx_2_e5"})

    trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False, 
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .001, 'ctx_num': 2, 
        'training': False, 'load_env': "/sanity/fully_trained_ctx_2_e5"})

    PARAMS = {
        "K_PCA": None,
        "P_BANDS": 100,
        "EVENT_WIN": 15,
        "EVENT_STD_MULT": 1,
        "K_SHOW_PCS": 100,
        "N_SHOW_NEUR": 100,
        "K_CAUSAL": None,
        "LAG_MAX": 29,
        "CAUSAL_RIDGE": 1e-6,
        "CAUSAL_MAX_ROWS": 60000,
        "FIG_DPI": 120,
        "MIN_YABS": 0.1,
        "SHOW_TQDM": True}

    ####### NEW PLOT: AVERAGED DYNAMICS ####### 

    def _to_np(x, dtype=None):
        a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        return a if dtype is None else a.astype(dtype, copy=False)

    def _z2(x):
        m = np.nanmean(x); s = np.nanstd(x)
        return (x - m) / (s + 1e-12)

    def _z3(x):
        m = np.nanmean(x, axis=(0, 1), keepdims=True); s = np.nanstd(x, axis=(0, 1), keepdims=True)
        return (x - m) / (s + 1e-12)

    def _min_ylim(ax, min_abs=0.1, pad=0.08):
        y0, y1 = ax.get_ylim()
        lo, hi = (y0, y1) if y0 < y1 else (y1, y0)
        lo = lo if lo < -min_abs else -min_abs
        hi = hi if hi > min_abs else min_abs
        span = hi - lo
        if span < 2 * min_abs: span = 2 * min_abs
        d = pad * span
        ax.set_ylim(lo - d, hi + d)

    def _iter(it, use_tqdm, desc=""):
        if use_tqdm and (tqdm is not None):
            return tqdm(it, desc=desc, leave=False)
        return it

    def get_logit_entropy(model, eps=1e-9):
        b = _to_np(model.model_goal_belief, np.float64)
        B, T, G = b.shape
        g = _to_np(model.goal_value, None)
        if np.ndim(g) == 0:
            gi = np.full((B,), int(g), dtype=np.int64)
        else:
            gv = np.asarray(g).reshape(-1)
            if gv.size == 1:
                gi = np.full((B,), int(gv[0]), dtype=np.int64)
            elif gv.size == B:
                gi = gv.astype(np.int64, copy=False)
            else:
                raise ValueError("goal_value must be scalar or shape [B].")
        gi = np.clip(gi, 0, G - 1)
        bix = np.arange(B, dtype=np.int64)[:, None]
        tix = np.arange(T, dtype=np.int64)[None, :]
        p = np.clip(b[bix, tix, gi[:, None]], eps, 1.0 - eps)
        logit = np.log(p / (1.0 - p)).astype(np.float32, copy=False)
        bb = np.clip(b, eps, 1.0)
        ent = (-(bb * np.log(bb)).sum(-1) / np.log(G + eps)).astype(np.float32, copy=False)
        return logit, ent

    def get_sii_bt(model, B, T):
        s = getattr(model, "SII", None)
        if s is None:
            return np.full((B, T), np.nan, dtype=np.float32)
        s = _to_np(s, np.float64)
        if s.ndim == 2:
            if s.shape == (B, T):
                out = s
            elif s.shape == (T, B):
                out = s.T
            elif s.size == B * T:
                out = s.reshape(B, T)
            else:
                raise ValueError(f"SII shape {s.shape} incompatible with (B,T)=({B},{T}).")
        elif s.ndim == 1 and s.size == B * T:
            out = s.reshape(B, T)
        else:
            raise ValueError(f"SII shape {s.shape} incompatible with (B,T)=({B},{T}).")
        return out.astype(np.float32, copy=False)

    def pca_from_updates(model, k_pca=None):
        upd = _to_np(model.model_update_flat, np.float32)
        if upd.ndim != 3:
            raise ValueError("model_update_flat must be [B,T,N].")
        B, T, N = upd.shape
        x = np.ascontiguousarray(upd.reshape(B * T, N), dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x - x.mean(0, keepdims=True)
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        kmax = vt.shape[0]
        k = kmax if k_pca is None else int(min(max(1, k_pca), kmax))
        v = vt[:k].T
        z = (x @ v).reshape(B, T, k).astype(np.float32, copy=False)
        ev = s * s
        evr = (ev[:k] / (ev.sum() + 1e-12)).astype(np.float32, copy=False)
        return upd, z, evr

    def metrics_from_scores_A(z):
        B, T, K = z.shape
        e = z * z
        s = np.sum(e, axis=-1, keepdims=True)
        p = e / np.maximum(s, 1e-12)
        pc_norm = np.sqrt(s[..., 0]).astype(np.float32, copy=False)
        if K == 1:
            centralization = np.zeros((B, T), dtype=np.float32)
        else:
            r = np.linspace(0.0, 1.0, K, dtype=np.float32).reshape(1, 1, K)
            centralization = (1.0 - 2.0 * np.sum(p * r, axis=-1)).astype(np.float32, copy=False)
        spectral_entropy = (-(p * np.log(np.maximum(p, 1e-12))).sum(-1) / np.log(K + 1e-12)).astype(np.float32, copy=False)
        return pc_norm, centralization, spectral_entropy, p.astype(np.float32, copy=False)

    def make_bands(p_btk, evr, p_bands):
        K = evr.size
        P = int(max(1, min(p_bands, K)))
        q = np.linspace(0.0, 1.0, P + 1)
        c = np.cumsum(evr)
        bid = np.searchsorted(q[1:-1], c, side="right").astype(np.int64)
        M = np.zeros((K, P), dtype=np.float32)
        M[np.arange(K), bid] = 1.0
        bands = np.tensordot(p_btk, M, axes=(2, 0)).astype(np.float32, copy=False)
        counts = np.zeros((P,), dtype=np.int64)
        for j in range(P):
            counts[j] = np.sum(bid == j)
        return bands, bid, counts

    def event_centers_beneficial(logit_bt, std_mult):
        d = logit_bt[:, 1:] - logit_bt[:, :-1]
        m = logit_bt[:, :-1] < 0
        x = d[m]
        if x.size == 0:
            z = np.zeros((0, 2), dtype=np.int64)
            return z, z.copy()
        th = x.mean() + std_mult * x.std()
        e = np.argwhere(m & (d >= th)).astype(np.int64)
        c = np.argwhere(m & (d < th)).astype(np.int64)
        return e, c

    def match_event_control(e, c, rng):
        if e.shape[0] == 0 or c.shape[0] == 0:
            return e[:0], c[:0]
        n = e.shape[0] if e.shape[0] < c.shape[0] else c.shape[0]
        ie = rng.choice(e.shape[0], size=n, replace=False) if e.shape[0] > n else np.arange(n)
        ic = rng.choice(c.shape[0], size=n, replace=False) if c.shape[0] > n else np.arange(n)
        return e[ie], c[ic]

    def aligned_mean_1d(x_bt, ctr, tau):
        n = ctr.shape[0]
        T = x_bt.shape[1]
        L = tau.size
        mu = np.full((L,), np.nan, dtype=np.float32)
        se = np.full((L,), np.nan, dtype=np.float32)
        if n == 0:
            return mu, se
        b = ctr[:, 0].astype(np.int64, copy=False)
        t0 = ctr[:, 1].astype(np.int64, copy=False)
        for i in range(L):
            tt = t0 + int(tau[i])
            ok = (tt >= 0) & (tt < T)
            nv = int(ok.sum())
            if nv <= 0:
                continue
            v = x_bt[b[ok], tt[ok]].astype(np.float64)
            mu[i] = np.nanmean(v).astype(np.float32)
            se[i] = (np.nanstd(v) / np.sqrt(max(1, nv))).astype(np.float32)
        return mu, se

    def aligned_mean_matrix(x_btf, ctr, tau):
        _, T, F = x_btf.shape
        L = tau.size
        mu = np.full((L, F), np.nan, dtype=np.float32)
        se = np.full((L, F), np.nan, dtype=np.float32)
        n = ctr.shape[0]
        if n == 0:
            return mu, se
        b = ctr[:, 0].astype(np.int64, copy=False)
        t0 = ctr[:, 1].astype(np.int64, copy=False)
        for i in range(L):
            tt = t0 + int(tau[i])
            ok = (tt >= 0) & (tt < T)
            nv = int(ok.sum())
            if nv <= 0:
                continue
            v = x_btf[b[ok], tt[ok], :].astype(np.float64)
            mu[i, :] = np.nanmean(v, axis=0).astype(np.float32)
            se[i, :] = (np.nanstd(v, axis=0) / np.sqrt(max(1, nv))).astype(np.float32)
        return mu, se

    def event_diff_1d(x_bt, e_ctr, c_ctr, tau):
        me, se_e = aligned_mean_1d(x_bt, e_ctr, tau)
        mc, se_c = aligned_mean_1d(x_bt, c_ctr, tau)
        return (me - mc).astype(np.float32, copy=False), np.sqrt(se_e * se_e + se_c * se_c).astype(np.float32, copy=False)

    def event_diff_matrix(x_btf, e_ctr, c_ctr, tau):
        me, se_e = aligned_mean_matrix(x_btf, e_ctr, tau)
        mc, se_c = aligned_mean_matrix(x_btf, c_ctr, tau)
        return (me - mc).astype(np.float32, copy=False), np.sqrt(se_e * se_e + se_c * se_c).astype(np.float32, copy=False)

    def full_trial_mean_matrix(x_btf):
        B = x_btf.shape[0]
        mu = np.nanmean(x_btf, axis=0).astype(np.float32)
        se = (np.nanstd(x_btf, axis=0) / np.sqrt(max(1, B))).astype(np.float32)
        return mu, se

    def _causal_subsample(z_bt, L, max_rows):
        B, T, _ = z_bt.shape
        M = T - L
        if M <= 1:
            return z_bt[:1]
        b_use = max(1, min(B, max_rows // M))
        if b_use >= B:
            return z_bt
        idx = np.linspace(0, B - 1, b_use, dtype=np.int64)
        return z_bt[idx]

    def _lag_design(z_bt, L):
        B, T, K = z_bt.shape
        M = T - L
        n = B * M
        X = np.zeros((n, K * L), dtype=np.float64)
        for lag in range(1, L + 1):
            X[:, (lag - 1) * K:lag * K] = z_bt[:, L - lag:T - lag, :].reshape(n, K)
        Y = z_bt[:, L:, :].reshape(n, K).astype(np.float64)
        return X, Y

    def granger_matrix_fast(z_bt, L, ridge=1e-6, max_rows=60000, use_tqdm=False):
        z = np.asarray(z_bt, dtype=np.float64)
        _, T, K = z.shape
        if L >= T - 1:
            return np.zeros((K, K), dtype=np.float32)
        z = _causal_subsample(z, L, max_rows)
        X, Y = _lag_design(z, L)
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
        Y = (Y - Y.mean(axis=0, keepdims=True)) / (Y.std(axis=0, keepdims=True) + 1e-6)
        G = X.T @ X
        H = X.T @ Y
        YY = np.sum(Y * Y, axis=0)
        out = np.zeros((K, K), dtype=np.float32)
        base = np.arange(L, dtype=np.int64) * K
        I_L = np.eye(L, dtype=np.float64) * ridge
        Aii = np.zeros((K, L, L), dtype=np.float64)
        for i in range(K):
            idx_i = base + i
            Aii[i] = G[np.ix_(idx_i, idx_i)] + I_L
        for j in _iter(range(K), use_tqdm, desc="Granger targets"):
            idx_t = base + j
            At = G[np.ix_(idx_t, idx_t)] + I_L
            At_inv = np.linalg.inv(At)
            bt = H[idx_t, j]
            v = At_inv @ bt
            rss_t = YY[j] - bt @ v
            if rss_t <= 1e-12:
                continue
            for i in range(K):
                if i == j:
                    continue
                idx_i = base + i
                Ait = G[np.ix_(idx_i, idx_t)]
                u = H[idx_i, j] - Ait @ v
                S = Aii[i] - Ait @ At_inv @ Ait.T + I_L
                try:
                    sol = np.linalg.solve(S, u)
                except np.linalg.LinAlgError:
                    sol = np.linalg.lstsq(S, u, rcond=None)[0]
                val = (u @ sol) / (rss_t + 1e-12)
                if val > 0:
                    out[i, j] = np.float32(val)
        return out

    def causal_totals_by_pc(M):
        K = M.shape[0]
        diag = np.diag(M)
        cs_in = np.cumsum(M, axis=0)
        cs_out = np.cumsum(M, axis=1)
        total_in = cs_in[-1, :]
        total_out = cs_out[:, -1]
        in_lo = np.zeros((K,), dtype=np.float32)
        out_lo = np.zeros((K,), dtype=np.float32)
        if K > 1:
            r = np.arange(K - 1, dtype=np.int64)
            in_lo[1:] = cs_in[r, r + 1]
            out_lo[1:] = cs_out[r + 1, r]
        in_hi = (total_in - in_lo - diag).astype(np.float32, copy=False)
        out_hi = (total_out - out_lo - diag).astype(np.float32, copy=False)
        return {
            "in_from_lower": in_lo,
            "in_from_higher": in_hi,
            "out_to_lower": out_lo,
            "out_to_higher": out_hi,
            "K": K
        }

    def prep_model(model, prm, rng):
        logit, entropy = get_logit_entropy(model)
        B, T = logit.shape
        sii = get_sii_bt(model, B, T)
        upd, z_all, evr = pca_from_updates(model, prm["K_PCA"])
        pc_norm, centralization, spectral_entropy, p = metrics_from_scores_A(z_all)
        bands, band_id, band_counts = make_bands(p, evr, prm["P_BANDS"])
        e, c = event_centers_beneficial(logit, prm["EVENT_STD_MULT"])
        e, c = match_event_control(e, c, rng)
        k_show = int(min(prm["K_SHOW_PCS"], z_all.shape[2]))
        n_show = int(min(prm["N_SHOW_NEUR"], upd.shape[2]))
        idx_neur = rng.choice(upd.shape[2], size=n_show, replace=False) if n_show > 0 else np.zeros((0,), dtype=np.int64)
        behavior = np.empty((B, T, 3), dtype=np.float32)
        behavior[:, :, 0] = _z2(logit).astype(np.float32, copy=False)
        behavior[:, :, 1] = _z2(entropy).astype(np.float32, copy=False)
        behavior[:, :, 2] = _z2(sii).astype(np.float32, copy=False)
        k_causal = z_all.shape[2] if prm["K_CAUSAL"] is None else int(min(max(2, prm["K_CAUSAL"]), z_all.shape[2]))
        z_causal = _z3(z_all[:, :, :k_causal]).astype(np.float32, copy=False)
        lag = int(min(max(1, prm["LAG_MAX"]), z_causal.shape[1] - 2))
        M = granger_matrix_fast(z_causal, lag, prm["CAUSAL_RIDGE"], prm["CAUSAL_MAX_ROWS"], prm["SHOW_TQDM"])
        return {
            "logit": logit,
            "entropy": entropy,
            "sii": sii,
            "metrics_z": (
                _z2(pc_norm).astype(np.float32, copy=False),
                _z2(centralization).astype(np.float32, copy=False),
                _z2(spectral_entropy).astype(np.float32, copy=False)
            ),
            "bands_z": _z3(bands).astype(np.float32, copy=False),
            "behavior_z": behavior,
            "z_show_z": _z3(z_all[:, :, :k_show]).astype(np.float32, copy=False),
            "neur_z": _z3(upd[:, :, idx_neur]).astype(np.float32, copy=False) if n_show > 0 else np.zeros((B, T, 0), dtype=np.float32),
            "band_id": band_id.astype(np.int64, copy=False),
            "band_counts": band_counts.astype(np.int64, copy=False),
            "pca_dim": z_all.shape[2],
            "bands_num": bands.shape[2],
            "e_ctr": e,
            "c_ctr": c,
            "causal": {"matrix": M, "totals": causal_totals_by_pc(M), "lag": lag, "k": k_causal}
        }

    def plot_all(d_tr, d_ec, prm):
        fig, ax = plt.subplots(6, 4, figsize=(20, 19), dpi=prm["FIG_DPI"])
        models = ((d_tr, "trained"), (d_ec, "echo"))
        met_cols = ("C0", "C2", "C3")
        met_names = ("pc_norm", "centralization", "spectral_entropy")
        beh_cols = ("C1", "C4", "C5")
        beh_names = ("logit", "entropy", "SII")
        w = int(max(1, prm["EVENT_WIN"]))
        tau = np.arange(-w, w + 1, dtype=np.int64)

        for mi, item in enumerate(models):
            d, label = item
            c_evt = 2 * mi
            c_full = c_evt + 1
            e = d["e_ctr"]
            c = d["c_ctr"]

            a = ax[0, c_evt]
            y0, _ = event_diff_1d(d["metrics_z"][0], e, c, tau)
            y1, _ = event_diff_1d(d["metrics_z"][1], e, c, tau)
            y2, _ = event_diff_1d(d["metrics_z"][2], e, c, tau)
            a.plot(tau, y0, lw=2, color=met_cols[0], label=met_names[0])
            a.plot(tau, y1, lw=2, color=met_cols[1], label=met_names[1])
            a.plot(tau, y2, lw=2, color=met_cols[2], label=met_names[2])
            a.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
            a.axhline(0, color="k", lw=1, alpha=0.5)
            a.grid(alpha=0.25)
            a.set_title(f"{label} | beneficial (event-control)")
            a.set_xlabel("τ around event")
            a.set_ylabel("z")
            if mi == 0:
                a.legend(frameon=False, fontsize=9)
            _min_ylim(a, prm["MIN_YABS"])

            b = ax[0, c_full]
            tt = np.arange(d["metrics_z"][0].shape[1], dtype=np.int64)
            b.plot(tt, np.nanmean(d["metrics_z"][0], axis=0), lw=2, color=met_cols[0])
            b.plot(tt, np.nanmean(d["metrics_z"][1], axis=0), lw=2, color=met_cols[1])
            b.plot(tt, np.nanmean(d["metrics_z"][2], axis=0), lw=2, color=met_cols[2])
            b.axhline(0, color="k", lw=1, alpha=0.5)
            b.grid(alpha=0.25)
            b.set_title(f"{label} | full trial")
            b.set_xlabel("t")
            _min_ylim(b, prm["MIN_YABS"])

            a = ax[1, c_evt]
            P = d["bands_z"].shape[2]
            cmap = plt.get_cmap("viridis")
            for j in range(P):
                y, _ = event_diff_1d(d["bands_z"][:, :, j], e, c, tau)
                a.plot(tau, y, lw=1, alpha=0.9, color=cmap(j / max(P - 1, 1)))
            a.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
            a.axhline(0, color="k", lw=1, alpha=0.5)
            a.grid(alpha=0.25)
            a.set_title(f"{label} | PC-bands | beneficial")
            a.set_xlabel("τ around event")
            a.set_ylabel("z")
            _min_ylim(a, prm["MIN_YABS"])

            b = ax[1, c_full]
            tt = np.arange(d["bands_z"].shape[1], dtype=np.int64)
            for j in range(P):
                b.plot(tt, np.nanmean(d["bands_z"][:, :, j], axis=0), lw=1, alpha=0.9, color=cmap(j / max(P - 1, 1)))
            b.axhline(0, color="k", lw=1, alpha=0.5)
            b.grid(alpha=0.25)
            b.set_title(f"{label} | PC-bands | full trial")
            b.set_xlabel("t")
            _min_ylim(b, prm["MIN_YABS"])

            a = ax[2, c_evt]
            md, sd = event_diff_matrix(d["behavior_z"], e, c, tau)
            for k in range(3):
                a.plot(tau, md[:, k], lw=2, color=beh_cols[k], label=beh_names[k] if mi == 0 else None)
                a.fill_between(tau, md[:, k] - sd[:, k], md[:, k] + sd[:, k], color=beh_cols[k], alpha=0.12, lw=0)
            a.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
            a.axhline(0, color="k", lw=1, alpha=0.5)
            a.grid(alpha=0.25)
            a.set_title(f"{label} | behavior | beneficial")
            a.set_xlabel("τ around event")
            a.set_ylabel("z")
            if mi == 0:
                a.legend(frameon=False, fontsize=9)
            _min_ylim(a, prm["MIN_YABS"])

            b = ax[2, c_full]
            tt = np.arange(d["behavior_z"].shape[1], dtype=np.int64)
            mu_bh, se_bh = full_trial_mean_matrix(d["behavior_z"])
            for k in range(3):
                b.plot(tt, mu_bh[:, k], lw=2, color=beh_cols[k])
                b.fill_between(tt, mu_bh[:, k] - se_bh[:, k], mu_bh[:, k] + se_bh[:, k], color=beh_cols[k], alpha=0.12, lw=0)
            b.axhline(0, color="k", lw=1, alpha=0.5)
            b.grid(alpha=0.25)
            b.set_title(f"{label} | behavior | full trial")
            b.set_xlabel("t")
            _min_ylim(b, prm["MIN_YABS"])

            a = ax[3, c_evt]
            K = d["z_show_z"].shape[2]
            cmap2 = plt.get_cmap("plasma")
            for k in range(K):
                y, _ = event_diff_1d(d["z_show_z"][:, :, k], e, c, tau)
                a.plot(tau, y, lw=1.2, alpha=0.95, color=cmap2(k / max(K - 1, 1)))
            a.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
            a.axhline(0, color="k", lw=1, alpha=0.5)
            a.grid(alpha=0.25)
            a.set_title(f"{label} | explicit PCs | beneficial")
            a.set_xlabel("τ around event")
            a.set_ylabel("z")
            _min_ylim(a, prm["MIN_YABS"])

            b = ax[3, c_full]
            tt = np.arange(d["z_show_z"].shape[1], dtype=np.int64)
            for k in range(K):
                b.plot(tt, np.nanmean(d["z_show_z"][:, :, k], axis=0), lw=1.2, alpha=0.95, color=cmap2(k / max(K - 1, 1)))
            b.axhline(0, color="k", lw=1, alpha=0.5)
            b.grid(alpha=0.25)
            b.set_title(f"{label} | explicit PCs | full trial")
            b.set_xlabel("t")
            _min_ylim(b, prm["MIN_YABS"])

            a = ax[4, c_evt]
            Q = d["neur_z"].shape[2]
            cmap3 = plt.get_cmap("tab20")
            for q in range(Q):
                y, _ = event_diff_1d(d["neur_z"][:, :, q], e, c, tau)
                a.plot(tau, y, lw=1, alpha=0.9, color=cmap3((q % 20) / 19.0))
            a.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
            a.axhline(0, color="k", lw=1, alpha=0.5)
            a.grid(alpha=0.25)
            a.set_title(f"{label} | random neurons | beneficial")
            a.set_xlabel("τ around event")
            a.set_ylabel("z")
            _min_ylim(a, prm["MIN_YABS"])

            b = ax[4, c_full]
            tt = np.arange(d["neur_z"].shape[1], dtype=np.int64)
            for q in range(Q):
                b.plot(tt, np.nanmean(d["neur_z"][:, :, q], axis=0), lw=1, alpha=0.9, color=cmap3((q % 20) / 19.0))
            b.axhline(0, color="k", lw=1, alpha=0.5)
            b.grid(alpha=0.25)
            b.set_title(f"{label} | random neurons | full trial")
            b.set_xlabel("t")
            _min_ylim(b, prm["MIN_YABS"])

        keys = ("in_from_lower", "in_from_higher", "out_to_lower", "out_to_higher")
        titles = ("in from lower", "in from higher", "out to lower", "out to higher")
        tr = d_tr["causal"]["totals"]
        ec = d_ec["causal"]["totals"]
        x = np.arange(min(tr["K"], ec["K"]), dtype=np.int64)

        for j in range(4):
            a = ax[5, j]
            k = keys[j]
            a.plot(x, tr[k][:x.size], lw=2, color="C0", label="trained")
            a.plot(x, ec[k][:x.size], lw=2, color="C1", label="echo")
            a.axhline(0, color="k", lw=1, alpha=0.4)
            a.grid(alpha=0.25)
            a.set_title(titles[j])
            a.set_xlabel("PC index (0 = highest-variance PC)")
            if j == 0:
                a.set_ylabel("total directed ΔR²")
            _min_ylim(a, prm["MIN_YABS"])

        ax[5, 0].legend(frameon=False, fontsize=9)
        fig.tight_layout()
        return fig

    def run_all(trained, echo, params=None):
        prm = dict(PARAMS)
        if params is not None:
            prm.update(params)
        rng = np.random.default_rng()
        d_tr = prep_model(trained, prm, rng)
        d_ec = prep_model(echo, prm, rng)
        fig = plot_all(d_tr, d_ec, prm)
        print(
            f"done | trained B={d_tr['logit'].shape[0]} T={d_tr['logit'].shape[1]} | "
            f"echo B={d_ec['logit'].shape[0]} T={d_ec['logit'].shape[1]} | "
            f"causal K tr/ec={d_tr['causal']['k']}/{d_ec['causal']['k']} lag tr/ec={d_tr['causal']['lag']}/{d_ec['causal']['lag']} | "
            f"bands tr/ec={d_tr['bands_num']}/{d_ec['bands_num']}"
        )
        return {"trained": d_tr, "echo": d_ec, "fig": fig, "params": prm}

    ####### NEW PLOT: VECTOR FIELDS ####### 

    CFG = {
        "K_PCA": None,
        "NX": 40,
        "NY": 40,
        "QLO": 20.0,
        "QHI": 80.0,
        "MIN_COUNT": 10,
        "FLOW_SCALE": 0.5,
        "QUIVER_WIDTH": 0.0038,
        "COLOR_MODE": "level",                # "level" or "delta"
        "COLOR_REF": "value",                 # kept for compatibility
        "PAIR_AVG_WEIGHTED": True,            # kept for compatibility
        "ROBUST_PCT": 98.0,
        "METRIC_NORM": "global",              # "global" or "none"
        "DEFAULT_CMAP": "viridis",
        "BAND_CMAP": "coolwarm",
        "BAND_SPLIT": 0.5,
        "BAND_ACTIVITY_MODE": "signed_ratio", # "winner" or "signed_ratio"
        "BAND_FORCE_SYMMETRIC": False,
        "FIGSIZE": (5, 10),                   # base size for 2 cols; combined plot uses x2
        "DPI": 120,
        "PRINT_SANITY": True,
        "FACECOLOR": "black",
        "GRID_ALPHA": 0.07,
        "COUNT_ALPHA_PCTL": 75.0,

        # Correlation annotation
        "CORR_MODE": "partial_time",          # "sii" | "partial_time" | "both" | "none"
        "PANEL_CORR_FONTSIZE": 7.2,
        "PANEL_CORR_Y": 1.02,
        "PARTIAL_FALLBACK_TO_PEARSON": True,

        # Label styling
        "YLABEL_FONTSIZE": 9,
        "YLABEL_LABELPAD": 6
    }


    def _to_np(x, dtype=None):
        a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        return a if dtype is None else a.astype(dtype, copy=False)

    def _safe_logit(p, eps=1e-9):
        q = np.clip(p, eps, 1.0 - eps)
        return np.log(q / (1.0 - q))

    def _z_global(x):
        m = np.nanmean(x)
        s = np.nanstd(x)
        return (x - m) / (s + 1e-12)

    def _normalize_metric(x, mode):
        if mode == "global":
            return _z_global(x).astype(np.float32, copy=False)
        if mode == "none":
            return x.astype(np.float32, copy=False)
        raise ValueError("METRIC_NORM must be 'global' or 'none'")

    def _time_avg_bt(x):
        m = np.nanmean(x, axis=1, keepdims=True)
        return np.broadcast_to(m, x.shape).astype(np.float32, copy=False)

    def _coerce_bt(x, B, T, fill_value=0.0):
        a = np.squeeze(_to_np(x, np.float32))
        if a.ndim == 0:
            return np.full((B, T), float(a), dtype=np.float32)
        if a.ndim == 1:
            if a.size == B:
                return np.broadcast_to(a[:, None], (B, T)).astype(np.float32, copy=False)
            if a.size == T:
                return np.broadcast_to(a[None, :], (B, T)).astype(np.float32, copy=False)
            return np.full((B, T), fill_value, dtype=np.float32)
        out = np.full((B, T), fill_value, dtype=np.float32)
        b = min(B, a.shape[0])
        t = min(T, a.shape[1])
        out[:b, :t] = a[:b, :t]
        return out

    def _model_name(model):
        nm = getattr(model, "name", None)
        return str(nm) if nm is not None else "model"

    def _goal_index(goal_value, B, G):
        gv = _to_np(goal_value)
        if np.ndim(gv) == 0:
            gi = np.full((B,), int(gv), dtype=np.int64)
        else:
            g = np.asarray(gv).reshape(-1)
            if g.size == 1:
                gi = np.full((B,), int(g[0]), dtype=np.int64)
            elif g.size == B:
                gi = g.astype(np.int64, copy=False)
            else:
                raise ValueError("goal_value must be scalar or shape [B].")
        return np.clip(gi, 0, G - 1)


    def belief_entropy_logit_from_model(model, eps=1e-9):
        b = _to_np(model.model_goal_belief, np.float64)
        b = np.clip(b, eps, 1.0)
        b = b / np.maximum(b.sum(axis=-1, keepdims=True), eps)
        B, T, G = b.shape

        gi = _goal_index(model.goal_value, B, G)
        bix = np.arange(B, dtype=np.int64)[:, None]
        tix = np.arange(T, dtype=np.int64)[None, :]
        p_true = b[bix, tix, gi[:, None]]

        L = _safe_logit(p_true, eps=eps).astype(np.float32, copy=False)
        H = (-(b * np.log(np.maximum(b, eps))).sum(axis=-1)).astype(np.float32, copy=False)
        return H, L


    def pca_scores_from_updates(model, k_pca=None):
        upd = _to_np(model.model_update_flat, np.float32)
        if upd.ndim != 3:
            raise ValueError("model_update_flat must be [B,T,N].")
        B, T, N = upd.shape

        x = np.ascontiguousarray(upd.reshape(B * T, N), dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x -= x.mean(axis=0, keepdims=True)

        _, _, vt = np.linalg.svd(x, full_matrices=False)
        kmax = vt.shape[0]
        k = kmax if k_pca is None else int(min(max(2, k_pca), kmax))
        return (x @ vt[:k].T).reshape(B, T, k).astype(np.float32, copy=False)


    def metrics_from_scores_B(z):
        e = z * z
        s = np.sum(e, axis=-1, keepdims=True)
        p = e / np.maximum(s, 1e-12)
        pc_norm = np.sqrt(s[..., 0]).astype(np.float32, copy=False)
        K = z.shape[-1]

        if K == 1:
            centralization = np.zeros(pc_norm.shape, dtype=np.float32)
            spectral_entropy = np.zeros(pc_norm.shape, dtype=np.float32)
        else:
            r = np.linspace(0.0, 1.0, K, dtype=np.float32).reshape(1, 1, K)
            centroid = np.sum(p * r, axis=-1)
            centralization = (1.0 - 2.0 * centroid).astype(np.float32, copy=False)
            spectral_entropy = (-(p * np.log(np.maximum(p, 1e-12))).sum(axis=-1) / np.log(K)).astype(np.float32, copy=False)

        return pc_norm, spectral_entropy, centralization


    def band_activity_from_scores(z, split_ratio=0.5, mode="winner"):
        e = z * z
        K = z.shape[-1]
        split = int(np.floor(K * float(split_ratio)))
        if split < 1:
            split = 1
        if split > K - 1:
            split = K - 1

        low = np.sum(e[..., :split], axis=-1)
        high = np.sum(e[..., split:], axis=-1)

        if mode == "winner":
            return np.where(high >= low, 1.0, -1.0).astype(np.float32, copy=False)
        if mode == "signed_ratio":
            den = np.maximum(high + low, 1e-12)
            return ((high - low) / den).astype(np.float32, copy=False)
        raise ValueError("BAND_ACTIVITY_MODE must be 'winner' or 'signed_ratio'")


    def build_model_base(model, cfg):
        H, L = belief_entropy_logit_from_model(model)
        Z = pca_scores_from_updates(model, cfg["K_PCA"])
        pc_norm, spectral_entropy, centralization = metrics_from_scores_B(Z)

        upd = _to_np(model.model_update_flat, np.float32)
        B, T, _ = upd.shape
        var_neurons = np.var(upd, axis=-1).astype(np.float32, copy=False)

        sii_attr = getattr(model, "SII", None)
        if sii_attr is None:
            sii_attr = getattr(model, "sii", None)
        if sii_attr is None:
            sii = np.zeros((B, T), dtype=np.float32)
            print(f"Warning [{_model_name(model)}]: no SII attribute; using zeros.")
        else:
            sii = _coerce_bt(sii_attr, B, T, fill_value=0.0)

        time_metric = np.broadcast_to(np.arange(T, dtype=np.float32)[None, :], (B, T)).astype(np.float32, copy=False)
        band_dom = band_activity_from_scores(Z, split_ratio=cfg["BAND_SPLIT"], mode=cfg["BAND_ACTIVITY_MODE"])

        metric_names = (
            "PC norm",
            "Spectral entropy",
            "Centralization",
            "Variance between neurons",
            "SII",
            "Time",
            "Band dominance (high vs low)"
        )
        raw_vals = (
            pc_norm,
            spectral_entropy,
            centralization,
            var_neurons,
            sii,
            time_metric,
            band_dom
        )

        return {
            "name": _model_name(model),
            "H": H,
            "L": L,
            "metric_names": metric_names,
            "raw_vals": raw_vals,
            "sii_raw": sii,
            "time_raw": time_metric
        }


    def _make_edges(x, qlo, qhi, nbin):
        xf = x[np.isfinite(x)]
        if xf.size == 0:
            return np.linspace(-1.0, 1.0, int(nbin) + 1)
        lo, hi = np.nanpercentile(xf, (qlo, qhi))
        if not np.isfinite(lo):
            lo = np.nanmin(xf)
        if not np.isfinite(hi):
            hi = np.nanmax(xf)
        if hi <= lo:
            hi = lo + 1e-9
        return np.linspace(lo, hi, int(nbin) + 1)


    def _binned_flow_from_flat(x0, y0, u, v, c, mask, xb, yb, min_count):
        nx = xb.size - 1
        ny = yb.size - 1
        m = mask & np.isfinite(x0) & np.isfinite(y0) & np.isfinite(u) & np.isfinite(v) & np.isfinite(c)

        xg = x0[m]
        yg = y0[m]
        ug = u[m]
        vg = v[m]
        cg = c[m]

        ix = np.digitize(xg, xb) - 1
        iy = np.digitize(yg, yb) - 1
        inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

        ix = ix[inb]
        iy = iy[inb]
        ug = ug[inb]
        vg = vg[inb]
        cg = cg[inb]
        flat = iy * nx + ix

        count = np.zeros((ny, nx), dtype=np.float64)
        sum_u = np.zeros((ny, nx), dtype=np.float64)
        sum_v = np.zeros((ny, nx), dtype=np.float64)
        sum_c = np.zeros((ny, nx), dtype=np.float64)

        np.add.at(count.reshape(-1), flat, 1.0)
        np.add.at(sum_u.reshape(-1), flat, ug)
        np.add.at(sum_v.reshape(-1), flat, vg)
        np.add.at(sum_c.reshape(-1), flat, cg)

        U = np.divide(sum_u, count, out=np.zeros_like(sum_u), where=count > 0)
        V = np.divide(sum_v, count, out=np.zeros_like(sum_v), where=count > 0)
        C = np.divide(sum_c, count, out=np.zeros_like(sum_c), where=count > 0)
        keep = count >= float(min_count)

        xc = 0.5 * (xb[:-1] + xb[1:])
        yc = 0.5 * (yb[:-1] + yb[1:])
        X, Y = np.meshgrid(xc, yc)
        return X, Y, U, V, C, keep, count


    def _vlims_from_rows(rows, robust_pct, color_mode):
        M = rows.shape[0]
        vlims = np.zeros((M, 2), dtype=np.float64)
        r = 0
        while r < M:
            C = rows[r][4]
            K = rows[r][5]
            a = C[K]
            if a.size == 0:
                vlims[r, 0] = -1.0
                vlims[r, 1] = 1.0
            else:
                if color_mode == "delta":
                    vmax = np.nanpercentile(np.abs(a), robust_pct)
                    vmax = max(vmax, 1e-9)
                    vlims[r, 0] = -vmax
                    vlims[r, 1] = vmax
                else:
                    lo = np.nanpercentile(a, 100.0 - robust_pct)
                    hi = np.nanpercentile(a, robust_pct)
                    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
                        lo = -1.0
                        hi = 1.0
                    vlims[r, 0] = lo
                    vlims[r, 1] = hi
            r += 1
        return vlims


    def _pearson_1d(x, y):
        n = x.size
        if n < 3:
            return np.nan, n
        x0 = x - np.mean(x)
        y0 = y - np.mean(y)
        den = np.sqrt(np.dot(x0, x0) * np.dot(y0, y0))
        if den <= 1e-12:
            return np.nan, n
        return float(np.dot(x0, y0) / den), n

    def _corr_xy(x, y):
        xf = np.asarray(x, dtype=np.float64).reshape(-1)
        yf = np.asarray(y, dtype=np.float64).reshape(-1)
        m = np.isfinite(xf) & np.isfinite(yf)
        n = int(np.count_nonzero(m))
        if n < 3:
            return np.nan, n
        return _pearson_1d(xf[m], yf[m])

    def _partial_corr_xy_t(x, y, t, fallback_to_corr=True):
        xf = np.asarray(x, dtype=np.float64).reshape(-1)
        yf = np.asarray(y, dtype=np.float64).reshape(-1)
        tf = np.asarray(t, dtype=np.float64).reshape(-1)
        m = np.isfinite(xf) & np.isfinite(yf) & np.isfinite(tf)
        n = int(np.count_nonzero(m))
        if n < 4:
            return np.nan, n

        xv = xf[m]
        yv = yf[m]
        tv = tf[m]

        tm = np.mean(tv)
        xc = xv - np.mean(xv)
        yc = yv - np.mean(yv)
        tc = tv - tm
        tss = np.dot(tc, tc)

        if tss <= 1e-12:
            if fallback_to_corr:
                return _corr_xy(xv, yv)
            return np.nan, n

        bx = np.dot(tc, xc) / tss
        by = np.dot(tc, yc) / tss

        rx = xv - (np.mean(xv) + bx * (tv - tm))
        ry = yv - (np.mean(yv) + by * (tv - tm))

        r, nn = _pearson_1d(rx, ry)
        if (not np.isfinite(r)) and fallback_to_corr:
            return _corr_xy(xv, yv)
        return r, nn


    def _format_corr(v):
        return "nan" if not np.isfinite(v) else f"{v:+.2f}"

    def _corr_text(cfg, r_sii, r_partial):
        mode = str(cfg.get("CORR_MODE", "partial_time"))
        if mode == "none":
            return ""
        if mode == "sii":
            return f"r(SII)={_format_corr(r_sii)}"
        if mode == "partial_time":
            return f"r(SII|time)={_format_corr(r_partial)}"
        if mode == "both":
            return f"r(SII)={_format_corr(r_sii)} | r(SII|time)={_format_corr(r_partial)}"
        return f"r(SII|time)={_format_corr(r_partial)}"


    def _pack_from_base(base, cfg, time_avg):
        H = base["H"]
        L = base["L"]
        metric_names = base["metric_names"]
        raw_vals = base["raw_vals"]
        sii_raw = base["sii_raw"]
        time_raw = base["time_raw"]
        M = len(metric_names)

        vals = np.empty((M,), dtype=object)
        i = 0
        while i < M:
            m = raw_vals[i]
            if time_avg:
                m = _time_avg_bt(m)
            vals[i] = _normalize_metric(m, cfg["METRIC_NORM"])
            i += 1

        sii_cmp_src = _time_avg_bt(sii_raw) if time_avg else sii_raw

        if cfg["PRINT_SANITY"]:
            t = vals[M - 2]
            b = vals[M - 1]
            print(
                f"[Sanity {base['name']}] time_avg={time_avg} norm={cfg['METRIC_NORM']} "
                f"| Time std={np.nanstd(t):.5g} | BandDom min/max=({np.nanmin(b):.5g},{np.nanmax(b):.5g})"
            )

        dL = L[:, 1:] - L[:, :-1]
        dH = H[:, 1:] - H[:, :-1]

        x0 = H[:, :-1].reshape(-1)
        y0 = L[:, :-1].reshape(-1)
        u = dH.reshape(-1)
        v = dL.reshape(-1)

        xb = _make_edges(x0, cfg["QLO"], cfg["QHI"], cfg["NX"])
        yb = _make_edges(y0, cfg["QLO"], cfg["QHI"], cfg["NY"])
        mask_all = np.ones(x0.shape, dtype=bool)

        rows = np.empty((M,), dtype=object)
        corr_sii = np.full((M,), np.nan, dtype=np.float64)
        corr_partial = np.full((M,), np.nan, dtype=np.float64)

        pbar = tqdm(range(M), desc=f"Binning [{base['name']}, time_avg={time_avg}]", leave=False)

        r = 0
        while r < M:
            m = vals[r]
            if cfg["COLOR_MODE"] == "delta":
                m_cmp = m[:, 1:] - m[:, :-1]
                s_cmp = sii_cmp_src[:, 1:] - sii_cmp_src[:, :-1]
                t_cmp = time_raw[:, :-1]  # keep un-averaged control variable
            else:
                m_cmp = m[:, :-1]
                s_cmp = sii_cmp_src[:, :-1]
                t_cmp = time_raw[:, :-1]  # keep un-averaged control variable

            rs, _ = _corr_xy(m_cmp, s_cmp)
            rp, _ = _partial_corr_xy_t(
                m_cmp, s_cmp, t_cmp,
                fallback_to_corr=bool(cfg.get("PARTIAL_FALLBACK_TO_PEARSON", True))
            )
            corr_sii[r] = rs
            corr_partial[r] = rp

            rows[r] = _binned_flow_from_flat(
                x0=x0, y0=y0, u=u, v=v, c=m_cmp.reshape(-1), mask=mask_all,
                xb=xb, yb=yb, min_count=cfg["MIN_COUNT"]
            )
            pbar.update(1)
            r += 1

        pbar.close()
        vlims = _vlims_from_rows(rows, cfg["ROBUST_PCT"], cfg["COLOR_MODE"])

        return {
            "rows_plot": rows,
            "row_vlims": vlims,
            "metric_names": metric_names,
            "time_avg": time_avg,
            "corr_sii": corr_sii,
            "corr_partial_time": corr_partial
        }


    def _style_axis(a, cfg):
        a.set_facecolor(cfg.get("FACECOLOR", "black"))
        for sp in a.spines.values():
            sp.set_color("0.5")
        a.grid(alpha=cfg.get("GRID_ALPHA", 0.07), color="white")
        a.tick_params(axis="both", colors="0.9", labelsize=8)

    def _is_band_metric(name):
        return name == "Band dominance (high vs low)"

    def _alpha_from_count(den, pct):
        if den.size == 0:
            return np.empty((0,), dtype=np.float64)
        ref = np.nanpercentile(den, pct) if np.any(np.isfinite(den)) else 1.0
        if (not np.isfinite(ref)) or (ref <= 0):
            ref = 1.0
        return np.clip(den / ref, 0.2, 1.0)


    def _draw_combined_grid(tr_false, tr_true, ec_false, ec_true, cfg):
        metric_names = tr_false["metric_names"]
        M = len(metric_names)

        if tuple(tr_true["metric_names"]) != tuple(metric_names):
            raise ValueError("Metric names mismatch: trained False vs True.")
        if tuple(ec_false["metric_names"]) != tuple(metric_names):
            raise ValueError("Metric names mismatch: trained vs echo.")
        if tuple(ec_true["metric_names"]) != tuple(metric_names):
            raise ValueError("Metric names mismatch: echo False vs True.")

        packs = (tr_false, tr_true, ec_false, ec_true)
        col_titles = (
            "trained | time_avg=False",
            "trained | time_avg=True",
            "echo | time_avg=False",
            "echo | time_avg=True"
        )

        fig_w = float(cfg["FIGSIZE"][0]) * 2.0
        fig_h = float(cfg["FIGSIZE"][1])

        fig, ax = plt.subplots(
            M, 4,
            figsize=(fig_w, fig_h),
            dpi=cfg["DPI"],
            sharex=True,
            sharey=True,
            constrained_layout=True
        )
        if M == 1:
            ax = np.expand_dims(ax, axis=0)

        fig.patch.set_facecolor(cfg.get("FACECOLOR", "black"))

        col = 0
        while col < 4:
            p = packs[col]
            r = 0
            while r < M:
                metric_name = p["metric_names"][r]
                is_band = _is_band_metric(metric_name)
                cmap = plt.get_cmap(cfg["BAND_CMAP"] if is_band else cfg["DEFAULT_CMAP"])

                vmin = p["row_vlims"][r, 0]
                vmax = p["row_vlims"][r, 1]
                if is_band and cfg["BAND_FORCE_SYMMETRIC"]:
                    vv = max(abs(vmin), abs(vmax), 1e-9)
                    vmin = -vv
                    vmax = vv

                norm = Normalize(vmin=vmin, vmax=vmax)
                sm = ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array(())

                a = ax[r, col]
                _style_axis(a, cfg)

                X, Y, U, V, C, keep, N = p["rows_plot"][r]
                if np.any(keep):
                    den = N[keep]
                    rgba = cmap(norm(C[keep]))
                    rgba[:, 3] = _alpha_from_count(den, cfg.get("COUNT_ALPHA_PCTL", 75.0))
                    a.quiver(
                        X[keep], Y[keep], U[keep], V[keep],
                        angles="xy", scale_units="xy", scale=cfg["FLOW_SCALE"],
                        color=rgba, width=cfg["QUIVER_WIDTH"], zorder=2
                    )

                if r == 0:
                    a.set_title(col_titles[col], fontsize=10, color="0.95", pad=16)

                corr_label = _corr_text(cfg, p["corr_sii"][r], p["corr_partial_time"][r])
                if corr_label != "":
                    a.text(
                        0.5, cfg.get("PANEL_CORR_Y", 1.02), corr_label,
                        transform=a.transAxes, ha="center", va="bottom",
                        fontsize=cfg.get("PANEL_CORR_FONTSIZE", 7.2),
                        color="0.95", clip_on=False
                    )

                if col == 0:
                    a.set_ylabel(
                        metric_name,
                        fontsize=cfg.get("YLABEL_FONTSIZE", 9),
                        color="0.95",
                        labelpad=cfg.get("YLABEL_LABELPAD", 6)
                    )
                if r == M - 1:
                    a.set_xlabel("Entropy H", fontsize=9, color="0.95")

                a.tick_params(labelleft=(col == 0), labelbottom=(r == M - 1))

                cb = fig.colorbar(sm, ax=a, fraction=0.045, pad=0.02)
                cb.set_label(metric_name, fontsize=8, color="0.95")
                cb.ax.tick_params(labelsize=8, colors="0.9")
                cb.outline.set_edgecolor("0.6")

                r += 1
            col += 1

        fig.suptitle(
            f"Flow fields ({M}x4) | norm={cfg['METRIC_NORM']} | mode={cfg['COLOR_MODE']} | band={cfg['BAND_ACTIVITY_MODE']} | corr={cfg['CORR_MODE']}",
            fontsize=12.8, color="0.97"
        )
        return fig


    def plot_network_flow_fields(trained, echo, cfg_override=None, metric_norm=None):
        cfg = dict(CFG)
        if cfg_override is not None:
            cfg.update(cfg_override)
        if metric_norm is not None:
            cfg["METRIC_NORM"] = str(metric_norm)

        cfg.setdefault("FACECOLOR", "black")
        cfg.setdefault("GRID_ALPHA", 0.07)
        cfg.setdefault("COUNT_ALPHA_PCTL", 75.0)
        cfg.setdefault("DEFAULT_CMAP", "viridis")
        cfg.setdefault("BAND_CMAP", "coolwarm")
        cfg.setdefault("CORR_MODE", "partial_time")
        cfg.setdefault("PARTIAL_FALLBACK_TO_PEARSON", True)

        if cfg.get("COLOR_REF", "value") == "pair_centered":
            print("Warning: COLOR_REF='pair_centered' is not used in this 4-column no-split layout.")

        print("Building trained base...")
        tr_base = build_model_base(trained, cfg)
        print("Building trained packs...")
        tr_false = _pack_from_base(tr_base, cfg, time_avg=False)
        tr_true = _pack_from_base(tr_base, cfg, time_avg=True)

        print("Building echo base...")
        ec_base = build_model_base(echo, cfg)
        print("Building echo packs...")
        ec_false = _pack_from_base(ec_base, cfg, time_avg=False)
        ec_true = _pack_from_base(ec_base, cfg, time_avg=True)

        fig_all = _draw_combined_grid(tr_false, tr_true, ec_false, ec_true, cfg)

        return {
            "trained": {"time_avg_false": tr_false, "time_avg_true": tr_true},
            "echo": {"time_avg_false": ec_false, "time_avg_true": ec_true},
            "fig_combined": fig_all,
            "cfg": cfg
        }


    # =========================
    # plotting calls
    # =========================
    out = run_all(trained, echo, params={
        "P_BANDS": 100,
        "K_SHOW_PCS": 100,
        "N_SHOW_NEUR": 100,
        "K_CAUSAL": None,
        "LAG_MAX": 29,
        "SHOW_TQDM": True,
        "EVENT_STD_MULT": 1
    })
    plt.show()

    out = plot_network_flow_fields(
        trained,
        echo,
        metric_norm="none",
        cfg_override={
        }
    )
    plt.show()
