"""
HEAVY ANALYSES - Event-aligned dynamics + network flow fields.
Granger causality extracted to event_and_flow_fast.py.
"""
import numpy as np, torch, os, sys, inspect
from matplotlib.colors import PowerNorm, Normalize
from matplotlib.cm import ScalarMappable
from tqdm import tqdm
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("root:", path)
sys.path.insert(0, path + '/main'); sys.path.insert(0, path + '/main/bayes'); sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld

if __name__ == "__main__":
    cuda = 0; realization_num = 10; step_num = 30; hid_dim = 1000
    state_num = 500; obs_num = 5; batch_num = 15000; episodes = 1
    _common = dict(mode="SANITY", cuda=cuda, episodes=episodes, checkpoint_every=5,
        realization_num=realization_num, hid_dim=hid_dim, obs_num=obs_num, show_plots=False,
        batch_num=batch_num, step_num=step_num, state_num=state_num, learn_embeddings=False,
        classifier_LR=.001, ctx_num=2, training=False)
    echo = CognitiveGridworld(**{**_common, 'reservoir': True, 'load_env': "/sanity/reservoir_ctx_2_e5"})
    trained = CognitiveGridworld(**{**_common, 'reservoir': False, 'load_env': "/sanity/fully_trained_ctx_2_e5"})

    PARAMS = {"K_PCA": None, "P_BANDS": 100, "EVENT_WIN": 15, "EVENT_STD_MULT": 1,
        "K_SHOW_PCS": 100, "N_SHOW_NEUR": 100, "FIG_DPI": 120, "MIN_YABS": 0.1}

    ####### SECTION 1: AVERAGED DYNAMICS #######

    def _to_np(x, dtype=None):
        a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        return a if dtype is None else a.astype(dtype, copy=False)

    def _z(x):
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

    def _z3(x):
        m = np.nanmean(x, axis=(0, 1), keepdims=True)
        return (x - m) / (np.nanstd(x, axis=(0, 1), keepdims=True) + 1e-12)

    def _min_ylim(ax, min_abs=0.1, pad=0.08):
        y0, y1 = ax.get_ylim()
        lo, hi = min(y0, y1), max(y0, y1)
        lo = min(lo, -min_abs); hi = max(hi, min_abs)
        span = max(hi - lo, 2 * min_abs); d = pad * span
        ax.set_ylim(lo - d, hi + d)

    def get_logit_entropy(model, eps=1e-9):
        b = _to_np(model.model_goal_belief, np.float64); B, T, G = b.shape
        g = _to_np(model.goal_value, None)
        if np.ndim(g) == 0: gi = np.full((B,), int(g), np.int64)
        else:
            gv = np.asarray(g).reshape(-1)
            if gv.size == 1: gi = np.full((B,), int(gv[0]), np.int64)
            elif gv.size == B: gi = gv.astype(np.int64, copy=False)
            else: raise ValueError("goal_value must be scalar or shape [B].")
        gi = np.clip(gi, 0, G - 1)
        p = np.clip(b[np.arange(B)[:, None], np.arange(T)[None, :], gi[:, None]], eps, 1.0 - eps)
        logit = np.log(p / (1.0 - p)).astype(np.float32, copy=False)
        bb = np.clip(b, eps, 1.0)
        ent = (-(bb * np.log(bb)).sum(-1) / np.log(G + eps)).astype(np.float32, copy=False)
        return logit, ent

    def get_sii_bt(model, B, T):
        s = getattr(model, "SII", None)
        if s is None: return np.full((B, T), np.nan, np.float32)
        s = _to_np(s, np.float64)
        if s.ndim == 2:
            if s.shape == (B, T): out = s
            elif s.shape == (T, B): out = s.T
            elif s.size == B * T: out = s.reshape(B, T)
            else: raise ValueError(f"SII shape {s.shape} incompatible with (B,T)=({B},{T}).")
        elif s.ndim == 1 and s.size == B * T: out = s.reshape(B, T)
        else: raise ValueError(f"SII shape {s.shape} incompatible with (B,T)=({B},{T}).")
        return out.astype(np.float32, copy=False)

    def pca_from_updates(model, k_pca=None):
        upd = _to_np(model.model_update_flat, np.float32); B, T, N = upd.shape
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
        ev = s * s; evr = (ev[:k] / (ev.sum() + 1e-12)).astype(np.float32, copy=False)
        return upd, z, evr

    def metrics_from_scores(z):
        e = z * z; s = e.sum(-1, keepdims=True); K = z.shape[2]
        p = e / np.maximum(s, 1e-12)
        pc_norm = np.sqrt(s[..., 0]).astype(np.float32, copy=False)
        if K == 1: cent = np.zeros(z.shape[:2], np.float32)
        else:
            r = np.linspace(0., 1., K, dtype=np.float32).reshape(1, 1, K)
            cent = (1.0 - 2.0 * (p * r).sum(-1)).astype(np.float32, copy=False)
        sp_ent = (-(p * np.log(np.maximum(p, 1e-12))).sum(-1) / np.log(K + 1e-12)).astype(np.float32, copy=False)
        return pc_norm, cent, sp_ent, p.astype(np.float32, copy=False)

    # ── Vectorized aligned_mean ──
    def aligned_mean_1d(x_bt, ctr, tau):
        n = ctr.shape[0]; T = x_bt.shape[1]; L = tau.size
        mu, se = np.full((L,), np.nan, np.float32), np.full((L,), np.nan, np.float32)
        if n == 0: return mu, se
        b = ctr[:, 0].astype(np.int64); t0 = ctr[:, 1].astype(np.int64)
        tt = t0[:, None] + tau[None, :]; ok = (tt >= 0) & (tt < T)
        vals = x_bt[np.broadcast_to(b[:, None], tt.shape), np.clip(tt, 0, T - 1)].astype(np.float64)
        vals[~ok] = np.nan; nv = ok.sum(0); v = nv > 0
        mu[v] = np.nanmean(vals[:, v], 0).astype(np.float32)
        se[v] = (np.nanstd(vals[:, v], 0) / np.sqrt(np.maximum(nv[v], 1))).astype(np.float32)
        return mu, se

    def aligned_mean_matrix(x_btf, ctr, tau):
        _, T, F = x_btf.shape; L = tau.size; n = ctr.shape[0]
        mu, se = np.full((L, F), np.nan, np.float32), np.full((L, F), np.nan, np.float32)
        if n == 0: return mu, se
        b = ctr[:, 0].astype(np.int64); t0 = ctr[:, 1].astype(np.int64)
        tt = t0[:, None] + tau[None, :]; ok = (tt >= 0) & (tt < T)
        vals = x_btf[np.broadcast_to(b[:, None], tt.shape), np.clip(tt, 0, T - 1), :].astype(np.float64)
        vals[~ok] = np.nan; nv = ok.sum(0)
        for i in range(L):
            if nv[i] > 0:
                mu[i] = np.nanmean(vals[:, i, :], 0).astype(np.float32)
                se[i] = (np.nanstd(vals[:, i, :], 0) / np.sqrt(max(1, nv[i]))).astype(np.float32)
        return mu, se

    def make_bands(p_btk, evr, p_bands):
        K = evr.size; P = int(max(1, min(p_bands, K)))
        c = np.cumsum(evr); bid = np.searchsorted(np.linspace(0., 1., P + 1)[1:-1], c, side="right").astype(np.int64)
        M = np.zeros((K, P), np.float32); M[np.arange(K), bid] = 1.0
        bands = np.tensordot(p_btk, M, axes=(2, 0)).astype(np.float32, copy=False)
        counts = np.array([np.sum(bid == j) for j in range(P)], np.int64)
        return bands, bid, counts

    def event_centers_beneficial(logit_bt, std_mult):
        d = logit_bt[:, 1:] - logit_bt[:, :-1]; m = logit_bt[:, :-1] < 0; x = d[m]
        if x.size == 0:
            z = np.zeros((0, 2), np.int64); return z, z.copy()
        th = x.mean() + std_mult * x.std()
        return np.argwhere(m & (d >= th)).astype(np.int64), np.argwhere(m & (d < th)).astype(np.int64)

    def match_event_control(e, c, rng):
        if e.shape[0] == 0 or c.shape[0] == 0: return e[:0], c[:0]
        n = min(e.shape[0], c.shape[0])
        ie = rng.choice(e.shape[0], n, replace=False) if e.shape[0] > n else np.arange(n)
        ic = rng.choice(c.shape[0], n, replace=False) if c.shape[0] > n else np.arange(n)
        return e[ie], c[ic]

    def event_diff(x, e, c, tau, matrix=False):
        _align = aligned_mean_matrix if matrix else aligned_mean_1d
        me, se_e = _align(x, e, tau); mc, se_c = _align(x, c, tau)
        return (me - mc).astype(np.float32, copy=False), np.sqrt(se_e**2 + se_c**2).astype(np.float32, copy=False)

    def full_trial_mean_matrix(x_btf):
        B = x_btf.shape[0]
        return np.nanmean(x_btf, 0).astype(np.float32), (np.nanstd(x_btf, 0) / np.sqrt(max(1, B))).astype(np.float32)

    def prep_model(model, prm, rng):
        logit, entropy = get_logit_entropy(model); B, T = logit.shape
        sii = get_sii_bt(model, B, T)
        upd, z_all, evr = pca_from_updates(model, prm["K_PCA"])
        pc_norm, cent, sp_ent, p = metrics_from_scores(z_all)
        bands, bid, bcnt = make_bands(p, evr, prm["P_BANDS"])
        e, c = match_event_control(*event_centers_beneficial(logit, prm["EVENT_STD_MULT"]), rng)
        k_show = int(min(prm["K_SHOW_PCS"], z_all.shape[2]))
        n_show = int(min(prm["N_SHOW_NEUR"], upd.shape[2]))
        idx_n = rng.choice(upd.shape[2], n_show, replace=False) if n_show > 0 else np.zeros(0, np.int64)
        beh = np.stack([_z(logit), _z(entropy), _z(sii)], axis=-1).astype(np.float32, copy=False)
        return {"logit": logit, "entropy": entropy, "sii": sii,
            "metrics_z": (_z(pc_norm).astype(np.float32), _z(cent).astype(np.float32), _z(sp_ent).astype(np.float32)),
            "bands_z": _z3(bands).astype(np.float32, copy=False), "behavior_z": beh,
            "z_show_z": _z3(z_all[:, :, :k_show]).astype(np.float32, copy=False),
            "neur_z": _z3(upd[:, :, idx_n]).astype(np.float32) if n_show > 0 else np.zeros((B, T, 0), np.float32),
            "band_id": bid.astype(np.int64, copy=False), "band_counts": bcnt.astype(np.int64, copy=False),
            "pca_dim": z_all.shape[2], "bands_num": bands.shape[2], "e_ctr": e, "c_ctr": c}

    # ── Plotting helpers ──
    def _plot_row_pair(ax_evt, ax_full, data_3d, e, c, tau, label, subtitle, cmap_name, prm, show_se=False, names=None):
        """Plot event-aligned diff (left) and full-trial mean (right) for multi-feature data."""
        F = data_3d.shape[2]; cmap = plt.get_cmap(cmap_name)
        tt = np.arange(data_3d.shape[1], dtype=np.int64)
        for k in range(F):
            col = cmap(k / max(F - 1, 1)) if names is None else cmap_name if isinstance(cmap_name, str) and F <= 10 else cmap(k / max(F - 1, 1))
            if names is not None and F <= 10: col = f"C{k}"
            y, se = event_diff(data_3d[:, :, k], e, c, tau, matrix=False)
            kw = dict(lw=2 if names else 1, alpha=0.9, color=col, label=names[k] if names and k < len(names) else None)
            ax_evt.plot(tau, y, **kw)
            if show_se: ax_evt.fill_between(tau, y - se, y + se, color=col, alpha=0.12, lw=0)
            ax_full.plot(tt, np.nanmean(data_3d[:, :, k], 0), **{**kw, 'label': None})
            if show_se:
                mu_f, se_f = np.nanmean(data_3d[:, :, k], 0), np.nanstd(data_3d[:, :, k], 0) / np.sqrt(data_3d.shape[0])
                ax_full.fill_between(tt, mu_f - se_f, mu_f + se_f, color=col, alpha=0.12, lw=0)
        for a in (ax_evt, ax_full):
            a.axhline(0, color="k", lw=1, alpha=0.5); a.grid(alpha=0.25); _min_ylim(a, prm["MIN_YABS"])
        ax_evt.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
        ax_evt.set_title(f"{label} | {subtitle} | beneficial"); ax_evt.set_xlabel("τ around event"); ax_evt.set_ylabel("z")
        ax_full.set_title(f"{label} | {subtitle} | full trial"); ax_full.set_xlabel("t")

    def plot_all(d_tr, d_ec, prm):
        fig, ax = plt.subplots(5, 4, figsize=(20, 16), dpi=prm["FIG_DPI"])
        met_cols, met_names = ("C0", "C2", "C3"), ("pc_norm", "centralization", "spectral_entropy")
        beh_cols, beh_names = ("C1", "C4", "C5"), ("logit", "entropy", "SII")
        w = int(max(1, prm["EVENT_WIN"])); tau = np.arange(-w, w + 1, dtype=np.int64)
        for mi, (d, label) in enumerate(((d_tr, "trained"), (d_ec, "echo"))):
            c_evt, c_full = 2 * mi, 2 * mi + 1
            e, c = d["e_ctr"], d["c_ctr"]
            # Row 0: metrics (3 lines)
            a, b = ax[0, c_evt], ax[0, c_full]
            for k in range(3):
                y, _ = event_diff(d["metrics_z"][k], e, c, tau, matrix=False)
                a.plot(tau, y, lw=2, color=met_cols[k], label=met_names[k])
                b.plot(np.arange(d["metrics_z"][0].shape[1]), np.nanmean(d["metrics_z"][k], 0), lw=2, color=met_cols[k])
            a.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
            for x in (a, b): x.axhline(0, color="k", lw=1, alpha=0.5); x.grid(alpha=0.25); _min_ylim(x, prm["MIN_YABS"])
            a.set_title(f"{label} | beneficial"); a.set_xlabel("τ around event"); a.set_ylabel("z")
            b.set_title(f"{label} | full trial"); b.set_xlabel("t")
            if mi == 0: a.legend(frameon=False, fontsize=9)
            # Row 1: bands
            _plot_row_pair(ax[1, c_evt], ax[1, c_full], d["bands_z"], e, c, tau, label, "PC-bands", "viridis", prm)
            # Row 2: behavior (with SE bands)
            a2, b2 = ax[2, c_evt], ax[2, c_full]
            md, sd = event_diff(d["behavior_z"], e, c, tau, matrix=True)
            mu_bh, se_bh = full_trial_mean_matrix(d["behavior_z"])
            tt = np.arange(d["behavior_z"].shape[1], dtype=np.int64)
            for k in range(3):
                a2.plot(tau, md[:, k], lw=2, color=beh_cols[k], label=beh_names[k] if mi == 0 else None)
                a2.fill_between(tau, md[:, k] - sd[:, k], md[:, k] + sd[:, k], color=beh_cols[k], alpha=0.12, lw=0)
                b2.plot(tt, mu_bh[:, k], lw=2, color=beh_cols[k])
                b2.fill_between(tt, mu_bh[:, k] - se_bh[:, k], mu_bh[:, k] + se_bh[:, k], color=beh_cols[k], alpha=0.12, lw=0)
            for x in (a2, b2): x.axhline(0, color="k", lw=1, alpha=0.5); x.grid(alpha=0.25); _min_ylim(x, prm["MIN_YABS"])
            a2.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
            a2.set_title(f"{label} | behavior | beneficial"); a2.set_xlabel("τ around event"); a2.set_ylabel("z")
            b2.set_title(f"{label} | behavior | full trial"); b2.set_xlabel("t")
            if mi == 0: a2.legend(frameon=False, fontsize=9)
            # Row 3: explicit PCs
            _plot_row_pair(ax[3, c_evt], ax[3, c_full], d["z_show_z"], e, c, tau, label, "explicit PCs", "plasma", prm)
            # Row 4: neurons
            _plot_row_pair(ax[4, c_evt], ax[4, c_full], d["neur_z"], e, c, tau, label, "random neurons", "tab20", prm)
        fig.tight_layout()
        return fig

    def run_all(trained, echo, params=None):
        prm = dict(PARAMS)
        if params is not None: prm.update(params)
        rng = np.random.default_rng()
        d_tr = prep_model(trained, prm, rng); d_ec = prep_model(echo, prm, rng)
        fig = plot_all(d_tr, d_ec, prm)
        return {"trained": d_tr, "echo": d_ec, "fig": fig, "params": prm}

    ####### SECTION 2: HEATMAP FIELDS #######

    CFG = {"K_PCA": None, "NX": 40, "NY": 40, "QLO": 20., "QHI": 80., "MIN_COUNT": 10, "COLOR_MODE": "level", "ROBUST_PCT": 98., "METRIC_NORM": "global",
        "DEFAULT_CMAP": "viridis", "BAND_CMAP": "coolwarm", "BAND_SPLIT": 0.5, "BAND_ACTIVITY_MODE": "signed_ratio", "BAND_FORCE_SYMMETRIC": False,
        "FIGSIZE": (10, 3), "DPI": 120, "FACECOLOR": "black", "GRID_ALPHA": 0.07, "YLABEL_FONTSIZE": 9, "YLABEL_LABELPAD": 6}

    def _normalize_metric(x, mode):
        if mode == "global": return _z(x).astype(np.float32, copy=False)
        if mode == "none": return x.astype(np.float32, copy=False)
        raise ValueError("METRIC_NORM must be 'global' or 'none'")

    def _coerce_bt(x, B, T, fill_value=0.0):
        a = np.squeeze(_to_np(x, np.float32))
        if a.ndim == 0: return np.full((B, T), float(a), np.float32)
        if a.ndim == 1:
            if a.size == B: return np.broadcast_to(a[:, None], (B, T)).astype(np.float32, copy=False)
            if a.size == T: return np.broadcast_to(a[None, :], (B, T)).astype(np.float32, copy=False)
            return np.full((B, T), fill_value, np.float32)
        out = np.full((B, T), fill_value, np.float32)
        out[:min(B, a.shape[0]), :min(T, a.shape[1])] = a[:min(B, a.shape[0]), :min(T, a.shape[1])]
        return out

    def band_activity_from_scores(z, split_ratio=0.5, mode="winner"):
        e = z * z; K = z.shape[-1]; sp = max(1, min(int(np.floor(K * split_ratio)), K - 1))
        lo, hi = e[..., :sp].sum(-1), e[..., sp:].sum(-1)
        if mode == "winner": return np.where(hi >= lo, 1., -1.).astype(np.float32, copy=False)
        if mode == "signed_ratio": return ((hi - lo) / np.maximum(hi + lo, 1e-12)).astype(np.float32, copy=False)
        raise ValueError("BAND_ACTIVITY_MODE must be 'winner' or 'signed_ratio'")

    def build_model_base(model, cfg):
        L, H = get_logit_entropy(model)
        _, Z, _ = pca_from_updates(model, cfg["K_PCA"])
        pc_norm, cent, sp_ent, _ = metrics_from_scores(Z)
        upd = _to_np(model.model_update_flat, np.float32); B, T, _ = upd.shape
        var_n = np.var(upd, axis=-1).astype(np.float32, copy=False)
        sii_attr = getattr(model, "SII", None)
        sii = _coerce_bt(sii_attr, B, T)
        time_m = np.broadcast_to(np.arange(T, dtype=np.float32)[None, :], (B, T)).astype(np.float32, copy=False)
        band_dom = band_activity_from_scores(Z, cfg["BAND_SPLIT"], cfg["BAND_ACTIVITY_MODE"])
        nm = getattr(model, "name", None); name = str(nm) if nm is not None else "model"
        return {"name": name, "H": H, "L": L,
            "metric_names": ("PC norm", "Spectral entropy", "Centralization", "Neural variance", "SII", "Time", "Band dominance"),
            "raw_vals": (pc_norm, sp_ent, cent, var_n, sii, time_m, band_dom)}

    def _make_edges(x, qlo, qhi, nbin):
        xf = x[np.isfinite(x)]
        if xf.size == 0: return np.linspace(-1., 1., int(nbin) + 1)
        lo, hi = np.nanpercentile(xf, (qlo, qhi))
        if not np.isfinite(lo): lo = np.nanmin(xf)
        if not np.isfinite(hi): hi = np.nanmax(xf)
        if hi <= lo: hi = lo + 1e-9
        return np.linspace(lo, hi, int(nbin) + 1)

    def _binned_heatmap(x0, y0, c, mask, xb, yb, min_count):
        nx, ny = xb.size - 1, yb.size - 1
        m = mask & np.isfinite(x0) & np.isfinite(y0) & np.isfinite(c)
        xg, yg, cg = x0[m], y0[m], c[m]
        ix = np.clip(np.digitize(xg, xb) - 1, 0, nx - 1)
        iy = np.clip(np.digitize(yg, yb) - 1, 0, ny - 1)
        flat = iy * nx + ix; sz = ny * nx
        count = np.bincount(flat, minlength=sz).reshape(ny, nx).astype(np.float64)
        sum_c = np.bincount(flat, cg, minlength=sz).reshape(ny, nx)
        C = np.full((ny, nx), np.nan); nz = count >= min_count
        C[nz] = sum_c[nz] / count[nz]
        return C, xb, yb

    def _vlims(rows, robust_pct, color_mode):
        M = len(rows); vlims = np.zeros((M, 2), np.float64)
        for r in range(M):
            a = rows[r][0][np.isfinite(rows[r][0])]
            if a.size == 0: vlims[r] = [-1., 1.]
            elif color_mode == "delta":
                vmax = max(np.nanpercentile(np.abs(a), robust_pct), 1e-9); vlims[r] = [-vmax, vmax]
            else:
                lo, hi = np.nanpercentile(a, 100. - robust_pct), np.nanpercentile(a, robust_pct)
                vlims[r] = [lo, hi] if np.isfinite(lo) and np.isfinite(hi) and hi > lo else [-1., 1.]
        return vlims

    def _pack_from_base(base, cfg, time_avg):
        M = len(base["metric_names"])
        _tavg = lambda m: np.broadcast_to(np.nanmean(m, 1, keepdims=True), m.shape).astype(np.float32, copy=False)
        vals = [_normalize_metric(_tavg(base["raw_vals"][i]) if time_avg else base["raw_vals"][i], cfg["METRIC_NORM"]) for i in range(M)]
        x0, y0 = base["H"][:, :-1].ravel(), base["L"][:, :-1].ravel()
        xb = _make_edges(x0, cfg["QLO"], cfg["QHI"], cfg["NX"])
        yb = _make_edges(y0, cfg["QLO"], cfg["QHI"], cfg["NY"])
        mask_all = np.ones(x0.shape, bool)
        rows = [None] * M
        for r in tqdm(range(M), desc=f"Binning [{base['name']}, time_avg={time_avg}]", leave=False):
            m_cmp = (vals[r][:, 1:] - vals[r][:, :-1]) if cfg["COLOR_MODE"] == "delta" else vals[r][:, :-1]
            rows[r] = _binned_heatmap(x0, y0, m_cmp.ravel(), mask_all, xb, yb, cfg["MIN_COUNT"])
        return {"rows": rows, "vlims": _vlims(rows, cfg["ROBUST_PCT"], cfg["COLOR_MODE"]),
                "metric_names": base["metric_names"], "time_avg": time_avg}

    def _style_axis(a, cfg):
        a.set_facecolor(cfg.get("FACECOLOR", "black"))
        for sp in a.spines.values(): sp.set_color("0.5")
        a.grid(alpha=cfg.get("GRID_ALPHA", 0.07), color="white"); a.tick_params(axis="both", colors="0.9", labelsize=8)

    def _draw_combined_grid(tr_f, tr_t, ec_f, ec_t, cfg):
        names = tr_f["metric_names"]; M = len(names)
        packs = (tr_f, tr_t, ec_f, ec_t)
        row_titles = ("trained", "trained | time", "echo", "echo | time")
        fig, ax = plt.subplots(4, M, figsize=(cfg["FIGSIZE"][0], cfg["FIGSIZE"][1] * 2),
            dpi=cfg["DPI"], sharex=True, sharey=True, constrained_layout=True)
        if M == 1: ax = np.expand_dims(ax, 1)
        fig.patch.set_facecolor(cfg.get("FACECOLOR", "black"))
        for row in range(4):
            p = packs[row]
            for c in range(M):
                mn = p["metric_names"][c]; is_band = mn == "Band dominance (high vs low)"
                cmap = plt.get_cmap(cfg["BAND_CMAP"] if is_band else cfg["DEFAULT_CMAP"])
                vmin, vmax = p["vlims"][c]
                if is_band and cfg["BAND_FORCE_SYMMETRIC"]:
                    vv = max(abs(vmin), abs(vmax), 1e-9); vmin, vmax = -vv, vv
                a = ax[row, c]; _style_axis(a, cfg)
                C, xb, yb = p["rows"][c]
                a.pcolormesh(xb, yb, C, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
                sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap); sm.set_array(())
                if row == 0: a.set_title(mn, fontsize=9, color="0.95", pad=16)
                if c == 0: a.set_ylabel(row_titles[row], fontsize=cfg.get("YLABEL_FONTSIZE", 9), color="0.95", labelpad=cfg.get("YLABEL_LABELPAD", 6))
                if row == 3: a.set_xlabel("Entropy H", fontsize=9, color="0.95")
                a.tick_params(labelleft=(c == 0), labelbottom=(row == 3))
        fig.suptitle(f"Heatmaps | norm={cfg['METRIC_NORM']} | mode={cfg['COLOR_MODE']} | band={cfg['BAND_ACTIVITY_MODE']}",
            fontsize=12.8, color="0.97")
        return fig

    def plot_network_flow_fields(trained, echo, cfg_override=None, metric_norm=None):
        cfg = {**CFG, **(cfg_override or {})}
        if metric_norm is not None: cfg["METRIC_NORM"] = str(metric_norm)
        print("plotting heatmaps")
        tr_base = build_model_base(trained, cfg)
        tr_f = _pack_from_base(tr_base, cfg, False); tr_t = _pack_from_base(tr_base, cfg, True)
        ec_base = build_model_base(echo, cfg)
        ec_f = _pack_from_base(ec_base, cfg, False); ec_t = _pack_from_base(ec_base, cfg, True)
        return {"trained": {"time_avg_false": tr_f, "time_avg_true": tr_t},
                "echo": {"time_avg_false": ec_f, "time_avg_true": ec_t},
                "fig_combined": _draw_combined_grid(tr_f, tr_t, ec_f, ec_t, cfg), "cfg": cfg}

    # =========================
    # plotting calls
    # =========================
    out = run_all(trained, echo, params={"P_BANDS": 100, "K_SHOW_PCS": 100, "N_SHOW_NEUR": 100, "EVENT_STD_MULT": 1})
    plt.show()
    out = plot_network_flow_fields(trained, echo, metric_norm="none", cfg_override={"NX": 50, "NY": 50, "MIN_COUNT": 1})
    plt.show()
