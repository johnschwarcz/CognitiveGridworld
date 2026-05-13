import numpy as np, torch, os, sys, inspect, matplotlib.pyplot as plt, matplotlib.ticker as ticker
from matplotlib.colors import PowerNorm, Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from tqdm import tqdm

path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
for sub in ('', '/bayes', '/model'): sys.path.insert(0, path + '/main' + sub)
from main.CognitiveGridworld import CognitiveGridworld

# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
EF_PARAMS = {"K_PCA": None, "P_BANDS": 100, "EVENT_WIN": 15, "EVENT_STD_MULT": 1,
    "K_SHOW_PCS": 100, "N_SHOW_NEUR": 100, "FIG_DPI": 120, "MIN_YABS": 0.1}

HM_CFG = {"K_PCA": None, "NX": 40, "NY": 40, "QLO": 20., "QHI": 80., "MIN_COUNT": 10, "COLOR_MODE": "level", "ROBUST_PCT": 98., "METRIC_NORM": "global",
    "DEFAULT_CMAP": "viridis", "BAND_CMAP": "coolwarm", "BAND_SPLIT": 0.5, "BAND_ACTIVITY_MODE": "signed_ratio", "BAND_FORCE_SYMMETRIC": False,
    "FIGSIZE": (15, 3), "DPI": 120, "FACECOLOR": "black", "GRID_ALPHA": 0.07, "YLABEL_FONTSIZE": 9, "YLABEL_LABELPAD": 6}

CFG = dict(DPI=140, FACECOLOR="black", GRID_ALPHA=0.07,
    PLOT_MODE="all", EPS=1e-10, T_PLOT=None, STEP_STRIDE=1,
    POINTS_PER_STEP=0, CONTOUR_BINS=250, CONTOUR_QHI=90,
    CONTOUR_MASS=0.90, CONTOUR_SMOOTH=3, CONTOUR_PAD=1.5,
    FRONT_ANGLE_BINS=250, FRONT_SMOOTH_SIGMA=3, VIEW_ELEV=25, VIEW_AZIM=-55, INTERSECT_LW=1)
plt.rcParams.update({"figure.dpi": CFG["DPI"], "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10})
AGENT_COLORS = {"Trained": "C0", "Echo": "C1", "Joint": "C2", "Naive": "C3"}
NET_Z0, NET_Z1, BAYES_Z0, BAYES_Z1 = 2, 3, 5, 6
_ANGLE_PLANES = ((-90,"red"),(-45,"green"),(0,"lightgreen"),(45,"green"),(90,"red"))
_XTICKS_180 = (-180, -90, -45, 0, 45, 90, 180)

# ═══════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════
def npy(x):
    return None if x is None else x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def _z(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)

def _z3(x):
    m = np.nanmean(x, axis=(0, 1), keepdims=True)
    return (x - m) / (np.nanstd(x, axis=(0, 1), keepdims=True) + 1e-12)

def renorm(P):
    P = np.asarray(P, dtype=np.float64); d = P.sum(-1, keepdims=True)
    return P / np.where(d > 0, d, 1.0)

def logit_arr(p, eps=1e-9):
    """Safe logit for arrays, clips to avoid inf."""
    p = np.clip(np.asarray(p, np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype(np.float64)

def calc_entropy(p):
    p = np.clip(p, 1e-20, 1.0); p = p / p.sum(axis=1, keepdims=True)
    return -(p * np.log(p)).sum(axis=1)

def compute_r2(X, y):
    return Ridge(alpha=1.0).fit(X, y).score(X, y)

def sym_dkl_pair(P, Q, eps=1e-4):
    P, Q = [np.clip(x, eps, 1-eps) for x in (P, Q)]
    P, Q = P / P.sum(-1, keepdims=True), Q / Q.sum(-1, keepdims=True)
    lP, lQ = np.log(P), np.log(Q)
    return 0.5 * ((P * (lP - lQ)).sum(-1) + (Q * (lQ - lP)).sum(-1))

def style(ax, dark=False):
    if dark:
        ax.set_facecolor(CFG["FACECOLOR"])
        for sp in ax.spines.values(): sp.set_color("0.5")
        ax.grid(alpha=CFG["GRID_ALPHA"], color="white"); ax.tick_params(axis="both", colors="0.9", labelsize=8)
    else:
        ax.grid(True, alpha=0.22)
        try: ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        except Exception: pass

def robust_lo_hi(v, qhi):
    u = np.asarray(v)[np.isfinite(v)]
    if u.size == 0: return -1.0, 1.0
    lo, hi = (float(u.min()), float(u.max())) if qhi >= 100 else (
        float(np.percentile(u, 100-qhi)), float(np.percentile(u, qhi)))
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        m = float(np.nanmean(u)) if np.isfinite(np.nanmean(u)) else 0.0
        s = float(np.nanstd(u)) if np.isfinite(np.nanstd(u)) and np.nanstd(u) > 0 else 1.0
        lo, hi = m - 3*s, m + 3*s
    return lo, hi

def mass_level(H, mass, normalize=False):
    H = np.asarray(H, dtype=np.float64); s = H.sum()
    if s <= 0: return np.nan
    平 = (H/s if normalize else H).ravel(); 平 = 平[平 > 0]
    if 平.size == 0: return np.nan
    平 = np.sort(平)[::-1]; c = np.cumsum(平)
    return float(平[np.searchsorted(c, mass * c[-1], side="left")])

def project_pca(X, mode="all"):
    X = npy(X)
    if isinstance(mode, int): X = X[:, mode:mode+1]
    B, T = X.shape[:2]; flat = X.reshape(B*T, -1)
    pca = PCA(n_components=2).fit(flat)
    return pca.transform(flat).reshape(B, T, 2), pca.explained_variance_ratio_ * 100

def coerce_goal_ind(goal_ind, B, T):
    g = npy(goal_ind)
    if g.ndim == 0: return np.full((B, T), int(g), np.int64)
    if g.ndim == 1:
        if g.shape[0] == B: return np.repeat(g[:, None].astype(np.int64), T, 1)
        if g.shape[0] == T: return np.repeat(g[None, :].astype(np.int64), B, 0)
        return np.full((B, T), int(g.reshape(-1)[0]), np.int64)
    if g.ndim == 2 and g.shape == (B, T): return g.astype(np.int64)
    return np.full((B, T), int(np.ravel(g)[0]), np.int64)

def neighbor_table(R, K):
    if K == -1:
        M = max(R-1, 1)
        return np.array([[(r+1+j)%R for j in range(M)] for r in range(R)], np.int64)
    M = 2*int(K); tab = np.empty((R, M), np.int64)
    for r in range(R):
        for d in range(1, int(K)+1):
            j = 2*(d-1); tab[r, j], tab[r, j+1] = (r-d)%R, (r+d)%R
    return tab

def nong_reduce(A_BTS, gBT):
    B, T, S = A_BTS.shape
    return A_BTS[np.arange(B)[:, None], np.arange(T)[None, :], (gBT+1)%S]

def _ctx_mask_iter(m, T):
    for r1 in m.realization_range:
        for r2 in m.realization_range:
            mask = (m.goal_ind == 0) & (m.ctx_vals[:, 0] == r1) & (m.ctx_vals[:, 1] == r2)
            if mask.sum() > 1: yield r1, r2, mask

def _plot_4agents(ax, xax, data, ls_map=None, lw=2.5, **kw):
    ls_map = ls_map or {"Trained":"-","Echo":"-","Joint":"--","Naive":"--"}
    for name, vals in data.items():
        ax.plot(xax, vals, ls=ls_map.get(name, "-"), c=AGENT_COLORS[name], lw=lw, label=name, **kw)

def approximate_likelihood(belief, eps=1e-99):
    belief = np.maximum(belief.astype(np.float64), eps)
    approx_px = np.zeros_like(belief); approx_px[:, 0] = belief[:, 0]
    ratio = belief[:, 1:] / belief[:, :-1]
    approx_px[:, 1:] = ratio / ratio.sum(axis=-1, keepdims=True)
    return approx_px

def compute_stepwise_dkl(p, q, eps=1e-99):
    p, q = np.maximum(p.astype(np.float64), eps), np.maximum(q.astype(np.float64), eps)
    p /= p.sum(axis=-1, keepdims=True); q /= q.sum(axis=-1, keepdims=True)
    d_pq = (p * (np.log(p) - np.log(q))).sum(axis=-1)
    d_qp = (q * (np.log(q) - np.log(p))).sum(axis=-1)
    return 0.5 * (d_pq + d_qp)

# ═══════════════════════════════════════════════════════════════════
# Section 1: Event-aligned dynamics
# ═══════════════════════════════════════════════════════════════════
def _min_ylim(ax, min_abs=0.1, pad=0.08):
    y0, y1 = ax.get_ylim(); lo, hi = min(y0, y1), max(y0, y1)
    lo = min(lo, -min_abs); hi = max(hi, min_abs)
    span = max(hi - lo, 2 * min_abs); d = pad * span
    ax.set_ylim(lo - d, hi + d)

def get_logit_entropy(model, eps=1e-9):
    b = npy(model.model_goal_belief).astype(np.float64); B, T, G = b.shape
    g = npy(model.goal_value)
    if np.ndim(g) == 0: gi = np.full((B,), int(g), np.int64)
    else:
        gv = np.asarray(g).reshape(-1)
        if gv.size == 1: gi = np.full((B,), int(gv[0]), np.int64)
        elif gv.size == B: gi = gv.astype(np.int64, copy=False)
        else: raise ValueError("goal_value must be scalar or shape [B].")
    gi = np.clip(gi, 0, G - 1)
    p = np.clip(b[np.arange(B)[:, None], np.arange(T)[None, :], gi[:, None]], eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p)).astype(np.float64, copy=False)
    bb = np.clip(b, eps, 1.0)
    ent = (-(bb * np.log(bb)).sum(-1) / np.log(G + eps)).astype(np.float64, copy=False)
    return logit, ent

def get_sii_bt(model, B, T):
    s = getattr(model, "SII", None)
    if s is None: return np.full((B, T), np.nan, np.float64)
    s = npy(s).astype(np.float64)
    if s.ndim == 2:
        if s.shape == (B, T): out = s
        elif s.shape == (T, B): out = s.T
        elif s.size == B * T: out = s.reshape(B, T)
        else: raise ValueError(f"SII shape {s.shape} incompatible with (B,T)=({B},{T}).")
    elif s.ndim == 1 and s.size == B * T: out = s.reshape(B, T)
    else: raise ValueError(f"SII shape {s.shape} incompatible with (B,T)=({B},{T}).")
    return out.astype(np.float64, copy=False)

def pca_from_updates(model, k_pca=None):
    upd = npy(model.model_update_flat).astype(np.float64); B, T, N = upd.shape
    cache = getattr(model, '_svd_cache', None)
    if cache is not None and cache['shape'] == (B, T, N):
        x, s, vt = cache['x'], cache['s'], cache['vt']
    else:
        x = np.ascontiguousarray(upd.reshape(B * T, N), dtype=np.float64)
        x = np.nan_to_num(x, nan=0., posinf=0., neginf=0.); x -= x.mean(0, keepdims=True)
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        try: model._svd_cache = {'shape': (B, T, N), 'x': x, 's': s, 'vt': vt}
        except AttributeError: pass
    k = vt.shape[0] if k_pca is None else int(min(max(1, k_pca), vt.shape[0]))
    z = (x @ vt[:k].T).reshape(B, T, k).astype(np.float64, copy=False)
    ev = s * s; evr = (ev[:k] / (ev.sum() + 1e-12)).astype(np.float64, copy=False)
    return upd, z, evr

def metrics_from_scores(z):
    e = z * z; s = e.sum(-1, keepdims=True); K = z.shape[2]
    p = e / np.maximum(s, 1e-12)
    pc_norm = np.sqrt(s[..., 0]).astype(np.float64, copy=False)
    if K == 1: cent = np.zeros(z.shape[:2], np.float64)
    else:
        r = np.linspace(0., 1., K, dtype=np.float64).reshape(1, 1, K)
        cent = (1.0 - 2.0 * (p * r).sum(-1)).astype(np.float64, copy=False)
    sp_ent = (-(p * np.log(np.maximum(p, 1e-12))).sum(-1) / np.log(K + 1e-12)).astype(np.float64, copy=False)
    return pc_norm, cent, sp_ent, p.astype(np.float64, copy=False)

def aligned_mean_1d(x_bt, ctr, tau):
    n = ctr.shape[0]; T = x_bt.shape[1]; L = tau.size
    mu, se = np.full((L,), np.nan, np.float64), np.full((L,), np.nan, np.float64)
    if n == 0: return mu, se
    b = ctr[:, 0].astype(np.int64); t0 = ctr[:, 1].astype(np.int64)
    tt = t0[:, None] + tau[None, :]; ok = (tt >= 0) & (tt < T)
    vals = x_bt[np.broadcast_to(b[:, None], tt.shape), np.clip(tt, 0, T - 1)].astype(np.float64)
    vals[~ok] = np.nan; nv = ok.sum(0); v = nv > 0
    mu[v] = np.nanmean(vals[:, v], 0).astype(np.float64)
    se[v] = (np.nanstd(vals[:, v], 0) / np.sqrt(np.maximum(nv[v], 1))).astype(np.float64)
    return mu, se

def aligned_mean_matrix(x_btf, ctr, tau):
    _, T, F = x_btf.shape; L = tau.size; n = ctr.shape[0]
    mu, se = np.full((L, F), np.nan, np.float64), np.full((L, F), np.nan, np.float64)
    if n == 0: return mu, se
    b = ctr[:, 0].astype(np.int64); t0 = ctr[:, 1].astype(np.int64)
    tt = t0[:, None] + tau[None, :]; ok = (tt >= 0) & (tt < T)
    vals = x_btf[np.broadcast_to(b[:, None], tt.shape), np.clip(tt, 0, T - 1), :].astype(np.float64)
    vals[~ok] = np.nan; nv = ok.sum(0)
    for i in range(L):
        if nv[i] > 0:
            mu[i] = np.nanmean(vals[:, i, :], 0).astype(np.float64)
            se[i] = (np.nanstd(vals[:, i, :], 0) / np.sqrt(max(1, nv[i]))).astype(np.float64)
    return mu, se

def make_bands(p_btk, evr, p_bands):
    K = evr.size; P = int(max(1, min(p_bands, K)))
    c = np.cumsum(evr); bid = np.searchsorted(np.linspace(0., 1., P + 1)[1:-1], c, side="right").astype(np.int64)
    M = np.zeros((K, P), np.float64); M[np.arange(K), bid] = 1.0
    bands = np.tensordot(p_btk, M, axes=(2, 0)).astype(np.float64, copy=False)
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
    return (me - mc).astype(np.float64, copy=False), np.sqrt(se_e**2 + se_c**2).astype(np.float64, copy=False)

def full_trial_mean_matrix(x_btf):
    B = x_btf.shape[0]
    return np.nanmean(x_btf, 0).astype(np.float64), (np.nanstd(x_btf, 0) / np.sqrt(max(1, B))).astype(np.float64)

def prep_model(model, prm, rng):
    logit, entropy = get_logit_entropy(model); B, T = logit.shape
    sii = get_sii_bt(model, B, T)
    
    # ADDED: Compute re-evaluation dynamically for event alignment
    eps = 1e-99
    j_bel = getattr(model, "joint_belief", None)
    j_px = getattr(model, "naive_px", None)
    j_bel = npy(j_bel).astype(np.float64)
    app_px = approximate_likelihood(j_bel, eps)
    reval = compute_stepwise_dkl(j_px, app_px, eps).mean(axis=-1)

    upd, z_all, evr = pca_from_updates(model, prm["K_PCA"])
    pc_norm, cent, sp_ent, p = metrics_from_scores(z_all)
    bands, bid, bcnt = make_bands(p, evr, prm["P_BANDS"])
    e, c = match_event_control(*event_centers_beneficial(logit, prm["EVENT_STD_MULT"]), rng)
    k_show = int(min(prm["K_SHOW_PCS"], z_all.shape[2]))
    n_show = int(min(prm["N_SHOW_NEUR"], upd.shape[2]))
    idx_n = rng.choice(upd.shape[2], n_show, replace=False) if n_show > 0 else np.zeros(0, np.int64)
    
    # UPDATED: Included reval in behavior stack
    beh = np.stack([_z(logit), _z(entropy), _z(sii), _z(reval)], axis=-1).astype(np.float64, copy=False)
    
    return {"logit": logit, "entropy": entropy, "sii": sii, "reval": reval,
        "metrics_z": (_z(pc_norm).astype(np.float64), _z(cent).astype(np.float64), _z(sp_ent).astype(np.float64)),
        "bands_z": _z3(bands).astype(np.float64, copy=False), "behavior_z": beh,
        "z_show_z": _z3(z_all[:, :, :k_show]).astype(np.float64, copy=False),
        "neur_z": _z3(upd[:, :, idx_n]).astype(np.float64) if n_show > 0 else np.zeros((B, T, 0), np.float64),
        "band_id": bid.astype(np.int64, copy=False), "band_counts": bcnt.astype(np.int64, copy=False),
        "pca_dim": z_all.shape[2], "bands_num": bands.shape[2], "e_ctr": e, "c_ctr": c}

# ── Event-aligned plotting ──
def _plot_row_pair(ax_evt, ax_full, data_3d, e, c, tau, label, subtitle, cmap_name, prm, show_se=False, names=None):
    F = data_3d.shape[2]; cmap = plt.get_cmap(cmap_name); tt = np.arange(data_3d.shape[1], dtype=np.int64)
    for k in range(F):
        col = f"C{k}" if names and F <= 10 else cmap(k / max(F - 1, 1))
        y, se = event_diff(data_3d[:, :, k], e, c, tau, matrix=False)
        kw = dict(lw=2 if names else 1, alpha=0.9, color=col, label=names[k] if names and k < len(names) else None)
        ax_evt.plot(tau, y, **kw)
        if show_se: ax_evt.fill_between(tau, y - se, y + se, color=col, alpha=0.12, lw=0)
        ax_full.plot(tt, np.nanmean(data_3d[:, :, k], 0), **{**kw, "label": None})
        if show_se:
            mu_f = np.nanmean(data_3d[:, :, k], 0); se_f = np.nanstd(data_3d[:, :, k], 0) / np.sqrt(data_3d.shape[0])
            ax_full.fill_between(tt, mu_f - se_f, mu_f + se_f, color=col, alpha=0.12, lw=0)
    for a in (ax_evt, ax_full): a.axhline(0, color="k", lw=1, alpha=0.5); a.grid(alpha=0.25); _min_ylim(a, prm["MIN_YABS"])
    ax_evt.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax_evt.set_title(f"{label} | {subtitle} | beneficial"); ax_evt.set_xlabel("tau"); ax_evt.set_ylabel("z")
    ax_full.set_title(f"{label} | {subtitle} | full trial"); ax_full.set_xlabel("t")

def plot_event_dynamics(d_tr, d_ec, prm):
    fig, ax = plt.subplots(5, 4, figsize=(20, 16), dpi=prm["FIG_DPI"])
    met_cols, met_names = ("C0", "C2", "C3"), ("pc_norm", "centralization", "spectral_entropy")
    # UPDATED: Added re-evaluation line to the behavior color scheme and labels list
    beh_cols, beh_names = ("C1", "C4", "C5", "C6"), ("logit", "entropy", "SII", "re-evaluation")
    w = int(max(1, prm["EVENT_WIN"])); tau = np.arange(-w, w + 1, dtype=np.int64)
    for mi, (d, label) in enumerate(((d_tr, "trained"), (d_ec, "echo"))):
        c_evt, c_full = 2 * mi, 2 * mi + 1; e, c = d["e_ctr"], d["c_ctr"]
        # Row 0: metrics
        a, b = ax[0, c_evt], ax[0, c_full]
        for k in range(3):
            y, _ = event_diff(d["metrics_z"][k], e, c, tau, matrix=False)
            a.plot(tau, y, lw=2, color=met_cols[k], label=met_names[k])
            b.plot(np.arange(d["metrics_z"][0].shape[1]), np.nanmean(d["metrics_z"][k], 0), lw=2, color=met_cols[k])
        a.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
        for x in (a, b): x.axhline(0, color="k", lw=1, alpha=0.5); x.grid(alpha=0.25); _min_ylim(x, prm["MIN_YABS"])
        a.set_title(f"{label} | beneficial"); a.set_xlabel("tau"); a.set_ylabel("z")
        b.set_title(f"{label} | full trial"); b.set_xlabel("t")
        if mi == 0: a.legend(frameon=False, fontsize=9)
        # Row 1: bands
        _plot_row_pair(ax[1, c_evt], ax[1, c_full], d["bands_z"], e, c, tau, label, "PC-bands", "viridis", prm)
        # Row 2: behavior
        a2, b2 = ax[2, c_evt], ax[2, c_full]
        md, sd = event_diff(d["behavior_z"], e, c, tau, matrix=True)
        mu_bh, se_bh = full_trial_mean_matrix(d["behavior_z"]); tt = np.arange(d["behavior_z"].shape[1], dtype=np.int64)
        
        # UPDATED: Loop to 4 elements now to handle re-evaluation
        for k in range(4):
            a2.plot(tau, md[:, k], lw=2, color=beh_cols[k], label=beh_names[k] if mi == 0 else None)
            a2.fill_between(tau, md[:, k] - sd[:, k], md[:, k] + sd[:, k], color=beh_cols[k], alpha=0.12, lw=0)
            b2.plot(tt, mu_bh[:, k], lw=2, color=beh_cols[k])
            b2.fill_between(tt, mu_bh[:, k] - se_bh[:, k], mu_bh[:, k] + se_bh[:, k], color=beh_cols[k], alpha=0.12, lw=0)
        
        for x in (a2, b2): x.axhline(0, color="k", lw=1, alpha=0.5); x.grid(alpha=0.25); _min_ylim(x, prm["MIN_YABS"])
        a2.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
        a2.set_title(f"{label} | behavior | beneficial"); a2.set_xlabel("tau"); a2.set_ylabel("z")
        b2.set_title(f"{label} | behavior | full trial"); b2.set_xlabel("t")
        if mi == 0: a2.legend(frameon=False, fontsize=9)
        # Row 3-4: PCs and neurons
        _plot_row_pair(ax[3, c_evt], ax[3, c_full], d["z_show_z"], e, c, tau, label, "explicit PCs", "plasma", prm)
        _plot_row_pair(ax[4, c_evt], ax[4, c_full], d["neur_z"], e, c, tau, label, "random neurons", "tab20", prm)
    fig.tight_layout(); return fig

def run_event_dynamics(trained, echo, params=None):
    prm = dict(EF_PARAMS)
    if params is not None: prm.update(params)
    rng = np.random.default_rng()
    d_tr = prep_model(trained, prm, rng); d_ec = prep_model(echo, prm, rng)
    return {"trained": d_tr, "echo": d_ec, "fig": plot_event_dynamics(d_tr, d_ec, prm), "params": prm}

# ═══════════════════════════════════════════════════════════════════
# Section 2: Heatmap fields
# ═══════════════════════════════════════════════════════════════════
def _normalize_metric(x, mode):
    if mode == "global": return _z(x).astype(np.float64, copy=False)
    if mode == "none": return x.astype(np.float64, copy=False)
    raise ValueError("METRIC_NORM must be 'global' or 'none'")

def _coerce_bt(x, B, T, fill_value=0.0):
    a = np.squeeze(npy(x).astype(np.float64))
    if a.ndim == 0: return np.full((B, T), float(a), np.float64)
    if a.ndim == 1:
        if a.size == B: return np.broadcast_to(a[:, None], (B, T)).astype(np.float64, copy=False)
        if a.size == T: return np.broadcast_to(a[None, :], (B, T)).astype(np.float64, copy=False)
        return np.full((B, T), fill_value, np.float64)
    out = np.full((B, T), fill_value, np.float64)
    out[:min(B, a.shape[0]), :min(T, a.shape[1])] = a[:min(B, a.shape[0]), :min(T, a.shape[1])]
    return out

def band_activity_from_scores(z, split_ratio=0.5, mode="winner"):
    e = z * z; K = z.shape[-1]; sp = max(1, min(int(np.floor(K * split_ratio)), K - 1))
    lo, hi = e[..., :sp].sum(-1), e[..., sp:].sum(-1)
    if mode == "winner": return np.where(hi >= lo, 1., -1.).astype(np.float64, copy=False)
    if mode == "signed_ratio": return ((hi - lo) / np.maximum(hi + lo, 1e-12)).astype(np.float64, copy=False)
    raise ValueError("BAND_ACTIVITY_MODE must be 'winner' or 'signed_ratio'")

def build_model_base(model, cfg):
    L, H = get_logit_entropy(model)
    _, Z, _ = pca_from_updates(model, cfg["K_PCA"])
    pc_norm, cent, sp_ent, _ = metrics_from_scores(Z)
    upd = npy(model.model_update_flat).astype(np.float64); B, T, _ = upd.shape
    var_n = np.var(upd, axis=-1).astype(np.float64, copy=False)
    sii = _coerce_bt(getattr(model, "SII", None), B, T)
    time_m = np.broadcast_to(np.arange(T, dtype=np.float64)[None, :], (B, T)).astype(np.float64, copy=False)
    band_dom = band_activity_from_scores(Z, cfg["BAND_SPLIT"], cfg["BAND_ACTIVITY_MODE"])
    nm = getattr(model, "name", None); name = str(nm) if nm is not None else "model"
    
    # Calculate goal/non-goal metrics
    gBT = coerce_goal_ind(getattr(model, "goal_ind", np.zeros(B)), B, T)
    ctx = npy(getattr(model, "ctx_vals", np.zeros((B,T,2)))).astype(np.float64)
    while ctx.ndim > 3 and ctx.shape[-1] == 1: ctx = ctx[..., 0]
    if ctx.ndim == 2: ctx = np.repeat(ctx[:, None, :], T, axis=1)
    gBT_ext = gBT[:, :, None]
    
    goal_R = np.take_along_axis(ctx, gBT_ext, axis=2).squeeze(2)
    nongoal_R = np.take_along_axis(ctx, 1 - gBT_ext, axis=2).squeeze(2)
    
    eps = 1e-99
    j_px_raw = npy(getattr(model, "joint_px")).astype(np.float64)
    j_px_raw = j_px_raw / np.maximum(j_px_raw.sum(axis=(-1, -2), keepdims=True), eps)
    j_px = np.stack([j_px_raw.sum(axis=-1), j_px_raw.sum(axis=-2)], axis=2).astype(np.float64)
    
    j_bel = npy(getattr(model, "joint_belief")).astype(np.float64)
    app_px = approximate_likelihood(j_bel, eps)
        
    reval = compute_stepwise_dkl(j_px, app_px, eps)
    goal_reval = np.take_along_axis(reval, gBT_ext, axis=2).squeeze(2)
    nongoal_reval = np.take_along_axis(reval, 1 - gBT_ext, axis=2).squeeze(2)

    return {"name": name, "H": H, "L": L,
        "metric_names": ("PC norm", "Spectral entropy", "Centralization", "Neural variance", "SII", "Time", "Band dominance", "goal R", "non-goal R", "goal Re-eval", "non-goal Re-eval"),
        "raw_vals": (pc_norm, sp_ent, cent, var_n, sii, time_m, band_dom, goal_R, nongoal_R, goal_reval, nongoal_reval)}

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
    _tavg = lambda m: np.broadcast_to(np.nanmean(m, 1, keepdims=True), m.shape).astype(np.float64, copy=False)
    vals = [_normalize_metric(_tavg(base["raw_vals"][i]) if time_avg else base["raw_vals"][i], cfg["METRIC_NORM"]) for i in range(M)]
    x0, y0 = base["H"][:, :-1].ravel(), base["L"][:, :-1].ravel()
    xb = _make_edges(x0, cfg["QLO"], cfg["QHI"], cfg["NX"])
    yb = _make_edges(y0, cfg["QLO"], cfg["QHI"], cfg["NY"])
    mask_all = np.ones(x0.shape, bool); rows = [None] * M
    for r in tqdm(range(M), desc=f"Binning [{base['name']}, time_avg={time_avg}]", leave=False):
        m_cmp = (vals[r][:, 1:] - vals[r][:, :-1]) if cfg["COLOR_MODE"] == "delta" else vals[r][:, :-1]
        rows[r] = _binned_heatmap(x0, y0, m_cmp.ravel(), mask_all, xb, yb, cfg["MIN_COUNT"])
    return {"rows": rows, "vlims": _vlims(rows, cfg["ROBUST_PCT"], cfg["COLOR_MODE"]),
            "metric_names": base["metric_names"], "time_avg": time_avg}

def _hm_style_axis(a, cfg):
    a.set_facecolor(cfg.get("FACECOLOR", "black"))
    for sp in a.spines.values(): sp.set_color("0.5")
    a.grid(alpha=cfg.get("GRID_ALPHA", 0.07), color="white"); a.tick_params(axis="both", colors="0.9", labelsize=8)

def _draw_combined_grid(tr_f, tr_t, ec_f, ec_t, cfg):
    names = tr_f["metric_names"]; M = len(names)
    packs = (tr_f, tr_t, ec_f, ec_t); row_titles = ("trained", "trained | time", "echo", "echo | time")
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
            a = ax[row, c]; _hm_style_axis(a, cfg)
            C, xb, yb = p["rows"][c]
            a.pcolormesh(xb, yb, C, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
            if row == 0: a.set_title(mn, fontsize=9, color="0.95", pad=16)
            if c == 0: a.set_ylabel(row_titles[row], fontsize=cfg.get("YLABEL_FONTSIZE", 9), color="0.95", labelpad=cfg.get("YLABEL_LABELPAD", 6))
            if row == 3: a.set_xlabel("Entropy H", fontsize=9, color="0.95")
            a.tick_params(labelleft=(c == 0), labelbottom=(row == 3))
    fig.suptitle(f"Heatmaps | norm={cfg['METRIC_NORM']} | mode={cfg['COLOR_MODE']} | band={cfg['BAND_ACTIVITY_MODE']}",
        fontsize=12.8, color="0.97")
    return fig

def plot_network_flow_fields(trained, echo, cfg_override=None, metric_norm=None):
    cfg = {**HM_CFG, **(cfg_override or {})}
    if metric_norm is not None: cfg["METRIC_NORM"] = str(metric_norm)
    print("plotting heatmaps")
    tr_base = build_model_base(trained, cfg)
    tr_f = _pack_from_base(tr_base, cfg, False); tr_t = _pack_from_base(tr_base, cfg, True)
    ec_base = build_model_base(echo, cfg)
    ec_f = _pack_from_base(ec_base, cfg, False); ec_t = _pack_from_base(ec_base, cfg, True)
    return {"trained": {"time_avg_false": tr_f, "time_avg_true": tr_t},
            "echo": {"time_avg_false": ec_f, "time_avg_true": ec_t},
            "fig_combined": _draw_combined_grid(tr_f, tr_t, ec_f, ec_t, cfg), "cfg": cfg}

# ═══════════════════════════════════════════════════════════════════
# Contour / front helpers
# ═══════════════════════════════════════════════════════════════════
def plot_step_contours(ax, X, Y, steps, ex, ey):
    cx, cy = 0.5*(ex[:-1]+ex[1:]), 0.5*(ey[:-1]+ey[1:])
    Xc, Yc = np.meshgrid(cx, cy, indexing="xy")
    mX, mY = np.full(steps.size, np.nan), np.full(steps.size, np.nan)
    cmap, sigma = plt.cm.coolwarm, CFG["CONTOUR_SMOOTH"]
    pps = int(CFG["POINTS_PER_STEP"]); n = X.shape[0]
    for i, st in enumerate(steps):
        st = int(st)
        if st < 1 or st > steps[-1] or ((st-1)%CFG["STEP_STRIDE"]) != 0: continue
        H, _, _ = np.histogram2d(X[:, i], Y[:, i], bins=(ex, ey))
        if sigma > 0: H = gaussian_filter(H.astype(np.float64), sigma=sigma)
        Hn = H / np.maximum(H.sum(), 1.0)
        lvl = mass_level(Hn, CFG["CONTOUR_MASS"], normalize=False)
        if pps > 0 and n > 0:
            idx = np.linspace(0, n-1, min(n, pps), dtype=np.int64)
            ax.scatter(X[idx,i], Y[idx,i], c=np.full(idx.size, float(st)),
                       cmap=cmap, vmin=1, vmax=steps[-1], s=7, alpha=0.20, linewidths=0, zorder=2)
        if np.isfinite(lvl) and lvl > 0:
            ax.contour(Xc, Yc, Hn.T, levels=(lvl,),
                       colors=(cmap((st-1)/max(steps[-1]-1,1)),), linewidths=1.6, alpha=0.95, zorder=3)
        mX[i], mY[i] = float(np.nanmean(X[:,i])), float(np.nanmean(Y[:,i]))
    ok = np.isfinite(mX) & np.isfinite(mY)
    if np.any(ok):
        ax.plot(mX[ok], mY[ok], lw=1.6, alpha=0.85, zorder=6)
        ax.scatter(mX[ok], mY[ok], c=(steps[ok].astype(np.float64)-1)/max(steps[-1]-1,1),
                   cmap=cmap, s=15, alpha=0.98, linewidths=0, zorder=7)

def calc_true_contour_front(X, Y, ex, ey, step_max):
    ab = CFG["FRONT_ANGLE_BINS"]; te = np.linspace(-np.pi, np.pi, ab+1); tc = 0.5*(te[:-1]+te[1:])
    fronts = np.full((step_max, ab), np.nan, dtype=np.float64)
    mx0, my0 = 0.0, 0.0
    for t in range(step_max):
        x, y = X[:,t], Y[:,t]; ok = np.isfinite(x) & np.isfinite(y)
        H, xed, yed = np.histogram2d(x, y, bins=(ex, ey))
        lvl = mass_level(H, CFG["CONTOUR_MASS"], normalize=False)
        ix = np.clip(np.searchsorted(xed, x, side="right")-1, 0, len(xed)-2)
        iy = np.clip(np.searchsorted(yed, y, side="right")-1, 0, len(yed)-2)
        m = (H[ix,iy] >= lvl) & ok
        if not np.any(m): continue
        xs, ys = x[m], y[m]; r = np.hypot(xs, ys)
        th = (np.arctan2(ys, xs) - np.pi/4 + np.pi) % (2*np.pi) - np.pi
        raw = np.full(ab, np.nan)
        for i in range(ab):
            b = (th >= te[i]) & (th < te[i+1])
            if b.sum() > 3: raw[i] = float(np.percentile(r[b], 95.0))
        okr = np.isfinite(raw)
        if okr.sum() > 3:
            s = np.interp(np.arange(ab), np.where(okr)[0], raw[okr])
            if CFG["FRONT_SMOOTH_SIGMA"] > 0: s = gaussian_filter1d(s, sigma=CFG["FRONT_SMOOTH_SIGMA"], mode="wrap")
            fronts[t] = s
    return fronts, tc

# ═══════════════════════════════════════════════════════════════════
# Figure: Paired PCA grids
# ═══════════════════════════════════════════════════════════════════
def render_paired_pca_grids(models_data, fig_title="PCA Projections"):
    ngr = len(models_data) * 2
    fig = plt.figure(figsize=(20, 2.25*ngr), dpi=CFG["DPI"])
    fig.patch.set_facecolor(CFG["FACECOLOR"]); gs = fig.add_gridspec(ngr, 7, hspace=0.2, wspace=0.1)
    for mi, (data, ev, ctx_raw, title_label) in enumerate(models_data):
        ctx = npy(ctx_raw)
        while ctx.ndim > 3 and ctx.shape[-1] == 1: ctx = ctx[..., 0]
        if ctx.ndim == 3: ctx = ctx[:, 0]
        u_r = np.unique(ctx[:, :2]); nR = len(u_r); den = max(nR-1, 1)
        cmap = plt.get_cmap("plasma")
        px = (data[...,0].max()-data[...,0].min())*0.35; py = (data[...,1].max()-data[...,1].min())*0.35
        xlim = (data[...,0].min()-px, data[...,0].max()+px); ylim = (data[...,1].min()-py, data[...,1].max()+py)
        def plot_traj(ax, mask, color, ls, label):
            if not mask.any(): return
            traj = data[mask].mean(0)
            ax.plot(traj[:,0], traj[:,1], color=color, ls=ls, alpha=0.8, lw=2, label=label)
            ax.scatter(traj[0,0], traj[0,1], color=color, marker="o", s=20)
            ax.scatter(traj[-1,0], traj[-1,1], color=color, marker="X", s=60, edgecolor="black", lw=0.5)
        for rs in range(2):
            gr = mi*2 + rs; r1p = (rs == 0); shift = "R1=(R2+{c})%n" if r1p else "R2=(R1+{c})%n"
            last = (gr == ngr-1)
            for c in range(6):
                ax = fig.add_subplot(gs[gr, c]); style(ax, True); ax.set_xlim(xlim); ax.set_ylim(ylim)
                ax.set_title(shift.format(c=c), color="0.95", fontsize=6)
                if c == 0:
                    lbl = f"PC2 ({ev[1]:.1f}%)\n{title_label}" if rs == 0 else f"PC2 ({ev[1]:.1f}%)"
                    ax.set_ylabel(lbl, color="cyan", fontsize=6, fontweight="bold")
                ax.tick_params(labelleft=False, labelbottom=False)
                if last: ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)", color="0.95", fontsize=6)
                for i, rv in enumerate(u_r):
                    r1, r2 = ((rv+c)%nR, rv) if r1p else (rv, (rv+c)%nR)
                    plot_traj(ax, (ctx[:,0]==r1)&(ctx[:,1]==r2), cmap(i/den), "-", f"({r1},{r2})")
                if rs < 2 and mi == 0: ax.legend(loc="best", fontsize=6, ncol=1, framealpha=0.1, labelcolor="0.95")
            ax = fig.add_subplot(gs[gr, 6]); style(ax, True); ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_title(f"Marginal: {'R2' if r1p else 'R1'}", color="yellow", fontsize=6)
            ax.tick_params(labelleft=False, labelbottom=last)
            for i, r in enumerate(u_r):
                mask = ctx[:,1]==r if r1p else ctx[:,0]==r
                plot_traj(ax, mask, cmap(i/den), "--" if r1p else "-", f"R={r}")
                if rs < 2 and mi == 0: ax.legend(loc="best", fontsize=6, ncol=1, framealpha=0.1, labelcolor="0.95")
    return fig

# ═══════════════════════════════════════════════════════════════════
# Figure: Boundary shape combined
# ═══════════════════════════════════════════════════════════════════
def plot_boundary_shape_combined(trained):
    bix = npy(getattr(trained, "batch_range", None))
    Pj_full, Pn_full = npy(trained.joint_belief), npy(trained.naive_belief)
    bix = bix.astype(np.int64) if bix is not None else np.arange(Pj_full.shape[0], dtype=np.int64)
    Pj, Pn = renorm(Pj_full[bix]), renorm(Pn_full[bix]); B, T, S, R = Pj.shape
    Tp = T if CFG["T_PLOT"] is None else int(np.clip(int(CFG["T_PLOT"]), 1, T))

    gBT = coerce_goal_ind(trained.goal_ind, B, T)
    ctx = npy(trained.ctx_vals)[bix]
    while ctx.ndim > 3 and ctx.shape[-1] == 1: ctx = ctx[..., 0]
    if ctx.ndim == 2: ctx = np.repeat(ctx[:, None, :], T, axis=1)
    ctx = ctx.astype(np.int64)
    bi, ti = np.arange(B, dtype=np.int64)[:, None], np.arange(T, dtype=np.int64)[None, :]
    tr_goal = ctx[bi, ti, gBT]
    tabA = neighbor_table(R, -1)
    pj_st = np.take_along_axis(Pj, ctx[..., None], 3)[..., 0]
    pn_st = np.take_along_axis(Pn, ctx[..., None], 3)[..., 0]
    pj_nr = np.take_along_axis(Pj, tabA[ctx], 3).mean(3)
    pn_nr = np.take_along_axis(Pn, tabA[ctx], 3).mean(3)
    pj_g_nr = np.take_along_axis(Pj[bi,ti,gBT], tabA[tr_goal], 2).mean(2)
    pn_g_nr = np.take_along_axis(Pn[bi,ti,gBT], tabA[tr_goal], 2).mean(2)
    XJ, YJ, XN, YN = (np.zeros((B,T,2), np.float64) for _ in range(4))
    XJ[...,0] = logit_arr(pj_st[bi,ti,gBT]); YJ[...,0] = logit_arr(nong_reduce(pj_st,gBT))
    XN[...,0] = logit_arr(pn_st[bi,ti,gBT]); YN[...,0] = logit_arr(nong_reduce(pn_st,gBT))
    XJ[...,1] = XJ[...,0] - logit_arr(pj_g_nr)
    YJ[...,1] = YJ[...,0] - logit_arr(nong_reduce(pj_nr, gBT))
    XN[...,1] = XN[...,0] - logit_arr(pn_g_nr)
    YN[...,1] = YN[...,0] - logit_arr(nong_reduce(pn_nr, gBT))
    col_titles = ("True", "True - all")
    steps = np.arange(1, Tp+1); cmap = plt.cm.coolwarm
    VIEW2_ELEV, VIEW2_AZIM = 0, 0  # second 3D viewing angle
    _LINE_ANGLES = np.arange(-180, 181, 45)  # every 45°
    _line_cmap = LinearSegmentedColormap.from_list(
        "grn_red", [(0, (.0,.4,.0)), (0.5, (.6,.9,.6)), (0.5, (.9,.6,.6)), (1, (.5,.0,.0))])
    fig = plt.figure(figsize=(10, 7))
    
    # GridSpec hspace and height_ratios adjusted to prevent 3D axes overlap
    outer = fig.add_gridspec(3, 1, hspace=0.1, height_ratios=[1, 1.2, 0.6],
                             left=0.06, right=0.98, top=0.94, bottom=0.06)
    gs0 = outer[0].subgridspec(1, 4, wspace=0.28)
    gs1 = outer[1].subgridspec(1, 4, wspace=0.05)
    gs2 = outer[2].subgridspec(1, 4, wspace=0.28)
    
    def _style_3d(ax3):
        ax3.set_xticklabels([]); ax3.set_yticklabels([]); ax3.set_zticklabels([])
        ax3.tick_params(axis='both', which='both', length=0, pad=0)
        ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis.pane.set_edgecolor((0.8,0.8,0.8,0.3))
            
    def _angle_color(a):
        return _line_cmap(abs(a) / 180.0)
        
    for sc in range(2):
        xj, yj = XJ[:,:Tp,sc], YJ[:,:Tp,sc]; xn, yn = XN[:,:Tp,sc], YN[:,:Tp,sc]
        x0, x1 = robust_lo_hi(np.concatenate((xj.ravel(),xn.ravel())), CFG["CONTOUR_QHI"])
        y0, y1 = robust_lo_hi(np.concatenate((yj.ravel(),yn.ravel())), CFG["CONTOUR_QHI"])
        dx, dy = (x1-x0)*CFG["CONTOUR_PAD"], (y1-y0)*CFG["CONTOUR_PAD"]
        ex = np.linspace(x0-dx, x1+dx, CFG["CONTOUR_BINS"]+1); ey = np.linspace(y0-dy, y1+dy, CFG["CONTOUR_BINS"]+1)
        fJ, deg = calc_true_contour_front(xj, yj, ex, ey, Tp); fN, _ = calc_true_contour_front(xn, yn, ex, ey, Tp)
        dp = deg*180/np.pi; xm, ym = np.meshgrid(dp, steps); zmax = max(np.nanmax(fJ), np.nanmax(fN))*1.05
        yp, zp = np.meshgrid((1, Tp), (0, zmax))
        
        for ji, (x, y, fronts, title) in enumerate([(xj,yj,fJ,"Joint"), (xn,yn,fN,"Naive")]):
            pi = sc*2 + ji
            
            # Row 0: contours
            ax = fig.add_subplot(gs0[pi])
            ax.set_title(f"{title}: {col_titles[sc]}", fontsize=8, fontweight='bold', pad=4)
            plot_step_contours(ax, x, y, steps, ex, ey)
            ax.axvline(0, lw=0.8, alpha=0.4, color='gray'); ax.axhline(0, lw=0.8, alpha=0.4, color='gray')
            style(ax, False); ax.set_aspect('equal', adjustable='datalim')
            
            # Row 1: 3D surface (single panel)
            ax3 = fig.add_subplot(gs1[pi], projection="3d")
            v = np.where(~np.isnan(fronts).all(1))[0]
            if v.size:
                sl = slice(v[0], v[-1]+1)
                ax3.plot_surface(xm[sl], ym[sl], fronts[sl], facecolors=cmap(plt.Normalize(1,Tp)(ym[sl])), lw = 1, ec = 'k',  ccount = 1, rcount = 10, alpha=1)
            for a in [0]:
                ax3.plot(np.full(Tp,a), steps, fronts[:,np.argmin(np.abs(dp-a))], color=_angle_color(a), lw=CFG["INTERSECT_LW"], ls = 'dotted', alpha=1, zorder=10)
            ax3.set_xlim(-180,180); ax3.set_ylim(1,Tp); ax3.set_zlim(0,zmax)
            ax3.view_init(CFG["VIEW_ELEV"], CFG["VIEW_AZIM"]); _style_3d(ax3)
            
            # Row 2: waterfall (front radius vs angle at each time)
            aw = fig.add_subplot(gs2[pi])
            for t in range(Tp):
                if np.any(np.isfinite(fronts[t])):
                    aw.plot(dp, fronts[t], color=cmap(t/max(Tp,1)), lw=3, zorder= t)
                    aw.plot(dp, fronts[t], color='k', lw=.4, zorder= 1000)
            for a in _LINE_ANGLES: aw.axvline(a, color=_angle_color(a), linestyle="-", lw=.5, zorder = 100)
            for a in _LINE_ANGLES: aw.axvline(a, color=_angle_color(a), linestyle="-", lw=1, alpha=1, zorder = -100)
            aw.set_xlim(-180,180); aw.set_xticks(_XTICKS_180); aw.set_xlabel("angle (deg)", fontsize=7, labelpad=2)
            style(aw, False); aw.tick_params(labelsize=6, pad=1)
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# Figure: Manifold evolution (entanglement)
# ═══════════════════════════════════════════════════════════════════
def plot_manifold_evolution(trained, echo, timesteps=[0, 15, -1]):
    agents = {
        "Trained": (trained.model_belief, trained),
        "Joint":   (trained.joint_belief, trained),
        "Naive":   (echo.naive_belief, echo),
        "Echo":    (echo.model_belief, echo)}
    names = list(agents.keys())
    nS = len(timesteps)
    nM = len(names)
    nc = nM * nS + (nS - 1)
    wr = ([1] * nM + [0.3]) * nS
    wr = wr[:-1]
    
    fig, axes = plt.subplots(3, nc, figsize=(8 * nS, 8), gridspec_kw={'width_ratios': wr}, constrained_layout=True)
    fig.suptitle("Representational Manifold Evolution: " + " $\\rightarrow$ ".join(f"Step (t={t})" for t in timesteps), fontsize=20, fontweight='bold')
    for row in axes:
        for s in range(nS - 1): 
            row[(s + 1) * nM + s].axis('off')
    for tl, ti in enumerate(timesteps):
        for mi, mn in enumerate(names):
            bel_raw, model = agents[mn]
            bel = npy(bel_raw)
            ctx = npy(model.ctx_vals)
            t_idx = ti if ti >= 0 else bel.shape[1] + ti
            t_idx = np.clip(t_idx, 0, bel.shape[1] - 1)
            r1, r2 = ctx[:, 0].astype(int), ctx[:, 1].astype(int)
            Xs = bel[:, t_idx]
            Xf = Xs.reshape(len(Xs), -1)
            X3 = Xs.reshape(len(Xs), 2, -1) if Xs.ndim == 2 else Xs
            ent = calc_entropy(X3[:, 0]) + calc_entropy(X3[:, 1])
            pca = PCA(n_components=2)
            Xp = pca.fit_transform(Xf)
            ve = pca.explained_variance_ratio_
            col_idx = tl * (nM + 1) + mi
            rows = [
                (r1 + r2, 'viridis', 'SUM'),
                (r1 - r2, 'inferno', 'DIFF'),
                (ent, 'plasma', 'ENTROPY')]
            for r_idx, (color_data, cm, yl) in enumerate(rows):
                ax = axes[r_idx, col_idx]
                ax.scatter(Xp[:, 0], Xp[:, 1], c=color_data, cmap=cm, s=150, alpha=0.1, edgecolor='none')
                sc = compute_r2(Xp, color_data)
                ax.text(0.95, 0.05, f"$R^2$: {sc:.2f}", transform=ax.transAxes, ha='right', va='bottom', 
                        fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=np.clip(1.5*sc, 0.2, 1), edgecolor='k'))
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(True, linestyle='--', alpha=0.3)
                if r_idx == 0: 
                    ax.set_title(f"{mn} | t={t_idx}\nPC1:{ve[0]:.1%} PC2:{ve[1]:.1%}", fontweight='bold', fontsize=10)
                if col_idx == 0: 
                    ax.set_ylabel(yl, fontweight='bold', fontsize=12)
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# Figure: Post-training diagnostics
# ═══════════════════════════════════════════════════════════════════
def _compute_variance_bundle(trained, echo, D=4):
    models = (trained, echo)
    T, R = min(int(trained.step_num), int(echo.step_num)), int(trained.realization_num)
    idx = np.arange(R); delta = np.abs(idx[:,None]-idx[None,:]); cdist = np.minimum(delta, R-delta).astype(np.int64)
    W = np.zeros((R, D, R), np.float64)
    for r1 in range(R):
        for d in range(D):
            m = cdist[r1]==d; c = m.sum()
            if c > 0: W[r1,d,m] = 1.0/c
    io_bar = np.zeros((2,2), np.float64)
    avg = {k: np.full((2,T), np.nan) for k in ("bel","joint","naive")}
    dist = {k: np.full((2,T,D), np.nan) for k in ("bel","joint","naive")}
    for i, mdl in enumerate(models):
        inp, out = np.full((R,R,T), np.nan), np.full((R,R,T), np.nan)
        mu = {k: np.full((R,R,T), np.nan) for k in ("bel","joint","naive")}
        dd = {k: np.full((R,R,T,D), np.nan) for k in ("bel","joint","naive")}
        for r1, r2, mask in _ctx_mask_iter(mdl, T):
            inp[r1,r2] = mdl.model_input_flat[mask,:T].var(0).mean(-1)
            out[r1,r2] = mdl.model_update_flat[mask,:T].var(0).mean(-1)
            Wr = W[r1]
            for k, bel in (("bel", mdl.model_goal_belief), ("joint", mdl.joint_goal_belief), ("naive", mdl.naive_goal_belief)):
                v = bel[mask,:T].var(0); mu[k][r1,r2] = v.mean(-1); dd[k][r1,r2] = v @ Wr.T
        io_bar[i] = [np.nanmean(inp), np.nanmean(out)]
        for k in avg: avg[k][i] = np.nanmean(mu[k], axis=(0,1)); dist[k][i] = np.nanmean(dd[k], axis=(0,1))
    return np.arange(T)+1, io_bar, avg, dist

def _compute_logit_bundle(trained, echo):
    T = min(int(trained.step_num), int(echo.step_num))
    xax = np.arange(T)+1; R = int(trained.realization_num); eps = 1e-9
    cfg = {"Trained":(trained,"model"), "Echo":(echo,"model"), "Joint":(trained,"joint"), "Naive":(trained,"naive")}
    curves = {}
    for key, (m, attr) in cfg.items():
        vals = np.full((R,R,T), np.nan); bel = getattr(m, f"{attr}_goal_belief")[:,:T]
        for r1, r2, mask in _ctx_mask_iter(m, T):
            p = np.clip(bel[mask,:,r1], eps, 1-eps); vals[r1,r2] = np.log(p/(1-p)).mean(0)
        curves[key] = np.nanmean(vals, axis=(0,1))
    fn = lambda t, a, b, c: a*np.log(t) - b*t + c
    try:    popt, _ = curve_fit(fn, xax, curves["Naive"], p0=(1,0.05,-2), maxfev=10000)
    except: popt = (0.5, 0.05, -2)
    return xax, curves, popt[0]*np.log(xax), popt[1]*(xax-1)

def plot_post_training_diagnostics(trained, echo, D=3):
    T = min(int(trained.step_num), int(echo.step_num))
    xax, io_bar, avg, dist = _compute_variance_bundle(trained, echo, D=D)
    x_logit, lc, log_v, lin_v = _compute_logit_bundle(trained, echo)
    Bs = [trained.joint_goal_belief[:,:T], trained.model_goal_belief[:,:T],
          trained.naive_goal_belief[:,:T], echo.model_goal_belief[:,:T]]
    ncol = 1+D; fig, axes = plt.subplots(2, ncol, figsize=(12, 4), constrained_layout=True); AC = AGENT_COLORS
    # Row 0 col 0: VARBAR
    a = axes[0,0]
    a.bar(range(4), [io_bar[0,0],io_bar[0,1],io_bar[1,0],io_bar[1,1]],
          color=(AC["Trained"],AC["Trained"],AC["Echo"],AC["Echo"]), alpha=0.82, edgecolor="k")
    a.set_xticks(range(4)); a.set_xticklabels(("Tr\nIn","Tr\nOut","Echo\nIn","Echo\nOut"))
    a.set_ylabel("variance"); a.set_title("Input/Output variance")
    # Row 0 col 1: DKL
    a = axes[0,1]; TT = T-1; means = np.zeros((4,TT))
    for k in range(TT):
        for j, B in enumerate(Bs): means[j,k] = sym_dkl_pair(B[:,0], B[:,k+1]).mean()
    xp = np.arange(TT)+1
    for idx, (name, ls) in enumerate([("Trained","-"),("Echo","-"),("Joint","--"),("Naive","--")]):
        order = [1,3,0,2][idx]
        a.plot(xp, means[order], ls+"o", ms=2.2, lw=1.8, c=AC[name], label=name.lower())
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("t"); a.set_ylabel("symDKL"); a.set_title("DKL(B_0, B_t)")
    a.legend(frameon=False, fontsize=9)
    # Row 0 col 2: LOGITPERF
    a = axes[0,2]; _plot_4agents(a, x_logit, lc, lw=2.5)
    a.set_title("True Positive Logit"); a.set_xlabel("t"); a.grid(alpha=0.25); a.legend(frameon=False, loc=2, fontsize=9)
    # Row 0 col 3: LOGITSCALE
    a = axes[0,3]
    a.fill_between(x_logit, -log_v, log_v, color=AC["Joint"], alpha=0.15, label=r"$\pm\ln t$")
    a.plot(x_logit, log_v, ":", c=AC["Joint"], lw=1.1); a.plot(x_logit, -log_v, ":", c=AC["Joint"], lw=1.1)
    a.fill_between(x_logit, -lin_v, lin_v, color=AC["Naive"], alpha=0.15, label=r"$\pm t$")
    a.plot(x_logit, lin_v, "--", c=AC["Naive"], lw=1.1); a.plot(x_logit, -lin_v, "--", c=AC["Naive"], lw=1.1)
    a.set_title("Contributions scale differently"); a.set_xlabel("t"); a.grid(alpha=0.25); a.legend(frameon=False, loc=2, fontsize=9)
    for j in range(4, ncol): axes[0,j].axis("off")
    # Row 1: variance panels
    eps = 1e-12
    def _var_panel(ax, bel_t, bel_e, jt, nv, title):
        ax.plot(xax, np.clip(bel_t, eps, np.inf), c=AC["Trained"], lw=2.5, label="Trained RNN", zorder=NET_Z0)
        ax.plot(xax, np.clip(bel_e, eps, np.inf), c=AC["Echo"], lw=2.5, label="Echo State", zorder=NET_Z1)
        ax.plot(xax, np.clip(jt, eps, np.inf), c=AC["Joint"], ls="--", lw=1.7, label="joint", zorder=BAYES_Z0)
        ax.plot(xax, np.clip(nv, eps, np.inf), c=AC["Naive"], ls="--", lw=1.7, label="naive", zorder=BAYES_Z1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.25)
        ax.set_title(title); ax.set_xlabel("t")
    _var_panel(axes[1,0], avg["bel"][0], avg["bel"][1], avg["joint"][0], avg["naive"][0],
               r"$B_r$ variance (avg over r1,r2,state)")
    axes[1,0].set_ylabel("variance across batches"); axes[1,0].legend(frameon=False, fontsize=9)
    for d in range(D):
        ad = axes[1, d+1]
        _var_panel(ad, dist["bel"][0,:,d], dist["bel"][1,:,d], dist["joint"][0,:,d], dist["naive"][0,:,d], f"Distance = {d}")
        if d > 0: ad.sharey(axes[1,1]); plt.setp(ad.get_yticklabels(), visible=False)
    return fig

# ═══════════════════════════════════════════════════════════════════
# Figure: Re-evaluation
# ═══════════════════════════════════════════════════════════════════
def plot_re_evaluation(trained):
    j_px = n_px = trained.naive_px / trained.naive_px.sum(axis=-1, keepdims=True)
    j_px_raw = trained.joint_px / trained.joint_px.sum(axis=(-1, -2), keepdims=True)
    reval_n = compute_stepwise_dkl(n_px, approximate_likelihood(trained.naive_belief))
    reval_j = compute_stepwise_dkl(j_px, approximate_likelihood(trained.joint_belief))
    time_len = n_px.shape[1]
    fig, ax = plt.subplots(1, 3, figsize=(8, 3), dpi=CFG["DPI"])
    c, t_ax = np.arange(1, time_len), np.arange(1, time_len)
    j_mean = reval_j.mean(axis=(0, 2))
    ax[0].plot(np.arange(1, time_len + 1), reval_n.mean(axis=(0, 2)), 'C2--o', ms=2, lw=1, alpha=0.6, label="Naive")
    ax[0].plot(np.arange(1, time_len + 1), j_mean, 'C1-o', ms=2, lw=1, label="Joint")
    ax[0].set(title="Linear scale", xlabel="Time step (t)", ylabel="Avg Re-evaluation"); ax[0].legend(fontsize=10, frameon=False)
    ax[1].plot(t_ax, j_mean[1:], 'k-', lw=2)
    ax[1].scatter(t_ax, j_mean[1:], c=c, cmap='coolwarm', edgecolor='k', lw=0.5, s=40, zorder=2)
    ax[1].set(title="Log scale", xlabel="Time step (t)", xscale='log', yscale='log')
    ax[1].tick_params(which='both', left=False, labelleft=False)
    cum_j = reval_j[:, 1:]; mu_x, mu_y = cum_j[..., 0].mean(axis=0), cum_j[..., 1].mean(axis=0)
    x_data, y_data = cum_j[..., 0].flatten(), cum_j[..., 1].flatten(); mask = (x_data > 0) & (y_data > 0)
    hb = ax[2].hexbin(x_data[mask], y_data[mask], gridsize=250, cmap='plasma', mincnt=1, bins='log', xscale='log', yscale='log')
    ax[2].set(title="Re-evaluation Covariance", xlabel="R1 (log scale)", ylabel="R2 (log scale)")
    fig.colorbar(hb, ax=ax[2], label="Density")
    ax[2].tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    fmt = ticker.LogFormatterMathtext()
    for i, a in enumerate(ax):
        style(a, dark=False)
        if i in [1, 2]: a.xaxis.set_major_formatter(fmt); a.yaxis.set_major_formatter(fmt)
    fig.suptitle(r"Re-evaluation: DKL(marginal likelihood $\parallel \frac{\text{marginal posterior}}{\text{marginal prior}}$)")
    fig.tight_layout(); plt.show()

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # common = dict(mode="SANITY", cuda=0, episodes=1, checkpoint_every=5,
    #               realization_num=10, hid_dim=1000, obs_num=5, show_plots=False,
    #               batch_num=15000, step_num=30, state_num=500,
    #               learn_embeddings=False, classifier_LR=.001, ctx_num=2, training=False)
    # echo    = CognitiveGridworld(**{**common, 'reservoir': True,  'load_env': "/sanity/reservoir_ctx_2_e5"})
    # trained = CognitiveGridworld(**{**common, 'reservoir': False, 'load_env': "/sanity/fully_trained_ctx_2_e5"})

    # _pca = lambda m, attr, lbl: (*project_pca(getattr(m, attr), mode=CFG["PLOT_MODE"]), npy(m.ctx_vals), lbl)

    """ Event-aligned dynamics """
    # run_event_dynamics(trained, echo, params={"P_BANDS": 100, "K_SHOW_PCS": 100, "N_SHOW_NEUR": 100, "EVENT_STD_MULT": 1})
    # plt.show()

    # """ heatmaps """
    # plot_network_flow_fields(trained, echo, metric_norm="none", cfg_override={"NX": 50, "NY": 50, "MIN_COUNT": 50})
    # plt.show()

    """ Paired PCA grids """
    # render_paired_pca_grids([_pca(trained,"model_belief","TRAINED"), _pca(trained,"joint_belief","JOINT")], fig_title="Trained + Joint")
    # render_paired_pca_grids([_pca(echo,"model_belief","ECHO"), _pca(echo,"naive_belief","NAIVE")], fig_title="Echo + Naive")

    """ Boundary shapes, manifold evolution, diagnostics, re-evaluation """
    # plot_boundary_shape_combined(trained)
    # plot_manifold_evolution(trained, echo, timesteps=[0, 15, -1])
    # plot_post_training_diagnostics(trained, echo)
    # plot_re_evaluation(trained)
