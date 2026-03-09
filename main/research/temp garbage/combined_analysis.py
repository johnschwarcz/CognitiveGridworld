"""
COMBINED ANALYSIS — Representation, Bayes flow-field, and entanglement manifolds.
Merges representation_and_bayes.py + entanglement.py.
"""
import numpy as np, os, sys, inspect, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import matplotlib.ticker as ticker

path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
for sub in ('', '/bayes', '/model'): sys.path.insert(0, path + '/main' + sub)
from main.CognitiveGridworld import CognitiveGridworld

# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════
CFG = dict(
    DPI=140, FACECOLOR="black", GRID_ALPHA=0.07,
    PLOT_MODE="all", EPS=1e-10, T_PLOT=None, STEP_STRIDE=1,
    POINTS_PER_STEP=0, CONTOUR_BINS=100, CONTOUR_QHI=95,
    CONTOUR_MASS=0.90, CONTOUR_SMOOTH=2, CONTOUR_PAD=1,
    FRONT_ANGLE_BINS=100, FRONT_SMOOTH_SIGMA=3, FRONT_SCALE_MODE="none",
    FRONT_CENTROID_MODE="t0",
    VIEW_ELEV=25, VIEW_AZIM=-55, PLANE_ALPHA=0.2, INTERSECT_LW=1.8)
plt.rcParams.update({"figure.dpi": CFG["DPI"], "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10})
AGENT_COLORS = {"Trained": "C0", "Echo": "C1", "Joint": "C2", "Naive": "C3"}
NET_Z0, NET_Z1, BAYES_Z0, BAYES_Z1 = 2, 3, 5, 6
_ANGLE_PLANES = ((-90,"red"),(-45,"green"),(0,"lightgreen"),(45,"green"),(90,"red"))
_XTICKS_180 = (-180, -90, -45, 0, 45, 90, 180)

# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════
def npy(x):
    return None if x is None else x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def renorm(P):
    P = np.asarray(P, dtype=np.float64); d = P.sum(-1, keepdims=True)
    return P / np.where(d > 0, d, 1.0)

def logit(p, clip=False):
    p = np.asarray(p, dtype=np.float64)
    if clip: p = np.clip(p, CFG["EPS"], 1.0 - CFG["EPS"])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(p / (1.0 - p)).astype(np.float32)

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
        ax.grid(alpha=CFG["GRID_ALPHA"], color="white")
        ax.tick_params(axis="both", colors="0.9", labelsize=8)
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
    flat = (H/s if normalize else H).ravel(); flat = flat[flat > 0]
    if flat.size == 0: return np.nan
    flat = np.sort(flat)[::-1]; c = np.cumsum(flat)
    return float(flat[np.searchsorted(c, mass * c[-1], side="left")])

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
    M = 2*int(K)
    tab = np.empty((R, M), np.int64)
    for r in range(R):
        for d in range(1, int(K)+1):
            j = 2*(d-1); tab[r, j], tab[r, j+1] = (r-d)%R, (r+d)%R
    return tab

def nong_reduce(A_BTS, gBT):
    B, T, S = A_BTS.shape
    return A_BTS[np.arange(B)[:, None], np.arange(T)[None, :], (gBT+1)%S]

def _ctx_mask_iter(m, T):
    """Yield (r1, r2, mask) for each context pair with >1 sample."""
    for r1 in m.realization_range:
        for r2 in m.realization_range:
            mask = (m.goal_ind == 0) & (m.ctx_vals[:, 0] == r1) & (m.ctx_vals[:, 1] == r2)
            if mask.sum() > 1: yield r1, r2, mask

def _plot_4agents(ax, xax, data, ls_map=None, lw=2.5, **kw):
    """Plot Trained/Echo/Joint/Naive on one axis with standard colors."""
    ls_map = ls_map or {"Trained":"-","Echo":"-","Joint":"--","Naive":"--"}
    for name, vals in data.items():
        ax.plot(xax, vals, ls=ls_map.get(name, "-"), c=AGENT_COLORS[name], lw=lw,
                label=name, **kw)

# --- Added specifically for the structural friction plot ---
def approximate_likelihood(belief, eps=1e-99):
    """Numerically stable Bayes approximation (operates on full N-D arrays natively)."""
    belief = np.maximum(belief.astype(np.float64), eps)
    approx_px = np.zeros_like(belief)
    approx_px[:, 0] = belief[:, 0]
    
    ratio = belief[:, 1:] / belief[:, :-1]
    approx_px[:, 1:] = ratio / ratio.sum(axis=-1, keepdims=True)
    return approx_px

def compute_stepwise_dkl(p, q, eps=1e-99):
    """Computes symmetric DKL directly for t=t', averaged over batch and features."""
    p, q = np.maximum(p.astype(np.float64), eps), np.maximum(q.astype(np.float64), eps)
    p /= p.sum(axis=-1, keepdims=True)
    q /= q.sum(axis=-1, keepdims=True)
    
    d_pq = (p * (np.log(p) - np.log(q))).sum(axis=-1)
    d_qp = (q * (np.log(q) - np.log(p))).sum(axis=-1)
    
    # Return mean over Batch (axis 0) and Features (axis 2)
    return 0.5 * (d_pq + d_qp)

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
                   cmap=cmap, s=34, alpha=0.98, linewidths=0, zorder=7)

def calc_true_contour_front(X, Y, ex, ey, step_max):
    ab = CFG["FRONT_ANGLE_BINS"]
    te = np.linspace(-np.pi, np.pi, ab+1)
    tc = 0.5*(te[:-1]+te[1:])
    fronts = np.full((step_max, ab), np.nan, dtype=np.float64)
    mx0, my0 = 0.0, 0.0
    if CFG["FRONT_CENTROID_MODE"] == "t0":
        ok0 = np.isfinite(X[:,0]) & np.isfinite(Y[:,0])
        if np.any(ok0): mx0, my0 = float(X[ok0,0].mean()), float(Y[ok0,0].mean())
    for t in range(step_max):
        x, y = X[:,t], Y[:,t]; ok = np.isfinite(x) & np.isfinite(y)
        if not np.any(ok): continue
        cm = CFG["FRONT_CENTROID_MODE"]
        mx, my = ((x[ok].mean(), y[ok].mean()) if cm == "per_step"
                  else (mx0, my0) if cm == "t0" else (0.0, 0.0))
        H, xed, yed = np.histogram2d(x, y, bins=(ex, ey))
        lvl = mass_level(H, CFG["CONTOUR_MASS"], normalize=False)
        ix = np.clip(np.searchsorted(xed, x, side="right")-1, 0, len(xed)-2)
        iy = np.clip(np.searchsorted(yed, y, side="right")-1, 0, len(yed)-2)
        m = (H[ix,iy] >= lvl) & ok
        if not np.any(m): continue
        xs, ys = x[m]-mx, y[m]-my
        r = np.hypot(xs, ys)
        th = (np.arctan2(ys, xs) - np.pi/4 + np.pi) % (2*np.pi) - np.pi
        raw = np.full(ab, np.nan)
        for i in range(ab):
            b = (th >= te[i]) & (th < te[i+1])
            if b.sum() > 3: raw[i] = float(np.percentile(r[b], 95.0))
        okr = np.isfinite(raw)
        if okr.sum() > 3:
            s = np.interp(np.arange(ab), np.where(okr)[0], raw[okr])
            if CFG["FRONT_SMOOTH_SIGMA"] > 0:
                s = gaussian_filter1d(s, sigma=CFG["FRONT_SMOOTH_SIGMA"], mode="wrap")
            sm = CFG["FRONT_SCALE_MODE"]
            if sm in ("mean","max"):
                denom = (np.nanmean if sm == "mean" else np.nanmax)(s)
                if denom > 1e-8: s /= denom
            fronts[t] = s
    return fronts, tc

# ═══════════════════════════════════════════════════════════════════
# Figure: Paired PCA grids
# ═══════════════════════════════════════════════════════════════════
def render_paired_pca_grids(models_data, fig_title="PCA Projections"):
    """PCA grid for a PAIR of models: 4 rows (2 per model: R1/R2-primary) × 7 columns."""
    ngr = len(models_data) * 2
    fig = plt.figure(figsize=(20, 2.25*ngr), dpi=CFG["DPI"])
    fig.patch.set_facecolor(CFG["FACECOLOR"])
    gs = fig.add_gridspec(ngr, 7, hspace=0.2, wspace=0.1)

    for mi, (data, ev, ctx_raw, title_label) in enumerate(models_data):
        ctx = npy(ctx_raw)
        while ctx.ndim > 3 and ctx.shape[-1] == 1: ctx = ctx[..., 0]
        if ctx.ndim == 3: ctx = ctx[:, 0]
        u_r = np.unique(ctx[:, :2]); nR = len(u_r); den = max(nR-1, 1)
        cmap = plt.get_cmap("plasma")
        px = (data[...,0].max()-data[...,0].min())*0.35
        py = (data[...,1].max()-data[...,1].min())*0.35
        xlim = (data[...,0].min()-px, data[...,0].max()+px)
        ylim = (data[...,1].min()-py, data[...,1].max()+py)

        def plot_traj(ax, mask, color, ls, label):
            if not mask.any(): return
            traj = data[mask].mean(0)
            ax.plot(traj[:,0], traj[:,1], color=color, ls=ls, alpha=0.8, lw=2, label=label)
            ax.scatter(traj[0,0], traj[0,1], color=color, marker="o", s=20)
            ax.scatter(traj[-1,0], traj[-1,1], color=color, marker="X", s=60, edgecolor="black", lw=0.5)

        for rs in range(2):
            gr = mi*2 + rs; r1p = (rs == 0)
            shift = "R1=(R2+{c})%n" if r1p else "R2=(R1+{c})%n"
            last = (gr == ngr-1)
            for c in range(6):
                ax = fig.add_subplot(gs[gr, c]); style(ax, True)
                ax.set_xlim(xlim); ax.set_ylim(ylim)
                ax.set_title(shift.format(c=c), color="0.95", fontsize=6)
                if c == 0:
                    lbl = f"PC2 ({ev[1]:.1f}%)\n{title_label}" if rs == 0 else f"PC2 ({ev[1]:.1f}%)"
                    ax.set_ylabel(lbl, color="cyan", fontsize=6, fontweight="bold")
                ax.tick_params(labelleft=False, labelbottom=False)
                if last: ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)", color="0.95", fontsize=6)
                for i, rv in enumerate(u_r):
                    r1, r2 = ((rv+c)%nR, rv) if r1p else (rv, (rv+c)%nR)
                    plot_traj(ax, (ctx[:,0]==r1)&(ctx[:,1]==r2), cmap(i/den), "-", f"({r1},{r2})")
                if rs < 2 and mi == 0:
                    ax.legend(loc="best", fontsize=6, ncol=1, framealpha=0.1, labelcolor="0.95")
            # Marginal column
            ax = fig.add_subplot(gs[gr, 6]); style(ax, True)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_title(f"Marginal: {'R2' if r1p else 'R1'}", color="yellow", fontsize=6)
            ax.tick_params(labelleft=False, labelbottom=last)
            for i, r in enumerate(u_r):
                mask = ctx[:,1]==r if r1p else ctx[:,0]==r
                plot_traj(ax, mask, cmap(i/den), "--" if r1p else "-", f"R={r}")
                if rs < 2 and mi == 0:
                    ax.legend(loc="best", fontsize=6, ncol=1, framealpha=0.1, labelcolor="0.95")
    return fig

# ═══════════════════════════════════════════════════════════════════
# Figure: Boundary shape combined
# ═══════════════════════════════════════════════════════════════════
def plot_boundary_shape_combined(trained):
    """3 rows × 6 columns (Contours / 3D / Waterfall)."""
    bix = npy(getattr(trained, "batch_range", None))
    Pj_full, Pn_full = npy(trained.joint_belief), npy(trained.naive_belief)
    bix = bix.astype(np.int64) if bix is not None else np.arange(Pj_full.shape[0], dtype=np.int64)
    Pj, Pn = renorm(Pj_full[bix]), renorm(Pn_full[bix])
    B, T, S, R = Pj.shape
    Tp = T if CFG["T_PLOT"] is None else int(np.clip(int(CFG["T_PLOT"]), 1, T))
    gBT = coerce_goal_ind(trained.goal_ind, B, T)
    ctx = npy(trained.ctx_vals)[bix]
    while ctx.ndim > 3 and ctx.shape[-1] == 1: ctx = ctx[..., 0]
    if ctx.ndim == 2: ctx = np.repeat(ctx[:, None, :], T, axis=1)
    ctx = ctx.astype(np.int64)
    bi, ti = np.arange(B, dtype=np.int64)[:, None], np.arange(T, dtype=np.int64)[None, :]
    tr_goal = ctx[bi, ti, gBT]
    tab1, tabA = neighbor_table(R, 1), neighbor_table(R, -1)
    pj_st = np.take_along_axis(Pj, ctx[..., None], 3)[..., 0]
    pn_st = np.take_along_axis(Pn, ctx[..., None], 3)[..., 0]

    pj_nr, pn_nr = np.zeros((B,T,S,2)), np.zeros((B,T,S,2))
    pj_g_nr, pn_g_nr = np.zeros((B,T,2)), np.zeros((B,T,2))
    for k, tab in enumerate((tab1, tabA)):
        pj_nr[...,k] = np.take_along_axis(Pj, tab[ctx], 3).mean(3)
        pn_nr[...,k] = np.take_along_axis(Pn, tab[ctx], 3).mean(3)
        pj_g_nr[...,k] = np.take_along_axis(Pj[bi,ti,gBT], tab[tr_goal], 2).mean(2)
        pn_g_nr[...,k] = np.take_along_axis(Pn[bi,ti,gBT], tab[tr_goal], 2).mean(2)

    XJ, YJ, XN, YN = (np.zeros((B,T,3), np.float32) for _ in range(4))
    XJ[...,0] = logit(pj_st[bi,ti,gBT], clip=True); YJ[...,0] = logit(nong_reduce(pj_st,gBT), clip=True)
    XN[...,0] = logit(pn_st[bi,ti,gBT], clip=True); YN[...,0] = logit(nong_reduce(pn_st,gBT), clip=True)
    for k in range(2):
        c = k+1
        XJ[...,c] = XJ[...,0] - logit(pj_g_nr[...,k], clip=True)
        YJ[...,c] = YJ[...,0] - logit(nong_reduce(pj_nr[...,k], gBT), clip=True)
        XN[...,c] = XN[...,0] - logit(pn_g_nr[...,k], clip=True)
        YN[...,c] = YN[...,0] - logit(nong_reduce(pn_nr[...,k], gBT), clip=True)

    col_titles = ("True", "True − near (K=1)", "True − near (all)")
    steps = np.arange(1, Tp+1); cmap = plt.cm.coolwarm
    fig = plt.figure(figsize=(20, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.1)

    for sc in range(3):
        xj, yj = XJ[:,:Tp,sc], YJ[:,:Tp,sc]
        xn, yn = XN[:,:Tp,sc], YN[:,:Tp,sc]
        x0, x1 = robust_lo_hi(np.concatenate((xj.ravel(),xn.ravel())), CFG["CONTOUR_QHI"])
        y0, y1 = robust_lo_hi(np.concatenate((yj.ravel(),yn.ravel())), CFG["CONTOUR_QHI"])
        dx, dy = (x1-x0)*CFG["CONTOUR_PAD"], (y1-y0)*CFG["CONTOUR_PAD"]
        ex = np.linspace(x0-dx, x1+dx, CFG["CONTOUR_BINS"]+1)
        ey = np.linspace(y0-dy, y1+dy, CFG["CONTOUR_BINS"]+1)
        fJ, deg = calc_true_contour_front(xj, yj, ex, ey, Tp)
        fN, _ = calc_true_contour_front(xn, yn, ex, ey, Tp)
        dp = deg*180/np.pi; xm, ym = np.meshgrid(dp, steps)
        zmax = max(np.nanmax(fJ), np.nanmax(fN))*1.05
        yp, zp = np.meshgrid((1, Tp), (0, zmax))

        for ji, (x, y, fronts, title) in enumerate([
            (xj,yj,fJ,"Joint"), (xn,yn,fN,"Naive")]):
            gc = sc*2 + ji
            # Contours
            ax = fig.add_subplot(gs[0,gc]); ax.set_title(f"{title}: {col_titles[sc]}", fontsize=10)
            plot_step_contours(ax, x, y, steps, ex, ey)
            ax.axvline(0, lw=1, alpha=0.5); ax.axhline(0, lw=1, alpha=0.5); style(ax, False)
            # 3D
            ax3 = fig.add_subplot(gs[1,gc], projection="3d")
            ax3.set_title(f"{title}: {col_titles[sc]}", fontsize=9)
            v = np.where(~np.isnan(fronts).all(1))[0]
            if v.size:
                sl = slice(v[0], v[-1]+1)
                ax3.plot_surface(xm[sl], ym[sl], fronts[sl],
                    facecolors=cmap(plt.Normalize(1,Tp)(ym[sl])), shade=True, lw=0, alpha=0.9)
            for a, cl in _ANGLE_PLANES:
                ax3.plot_surface(np.full_like(yp,a), yp, zp, color=cl, alpha=CFG["PLANE_ALPHA"], shade=False)
                ax3.plot(np.full(Tp,a), steps, fronts[:,np.argmin(np.abs(dp-a))],
                         color=cl, lw=CFG["INTERSECT_LW"], zorder=10)
            ax3.set_xlim(-180,180); ax3.set_ylim(1,Tp); ax3.set_zlim(0,zmax)
            ax3.set_xticks(_XTICKS_180); ax3.view_init(CFG["VIEW_ELEV"], CFG["VIEW_AZIM"])
            # Waterfall
            aw = fig.add_subplot(gs[2,gc]); aw.set_title(f"{title}: {col_titles[sc]}", fontsize=9)
            for t in range(Tp):
                if np.any(np.isfinite(fronts[t])):
                    aw.plot(dp, fronts[t], color=cmap(t/max(Tp,1)), lw=1.2, alpha=0.8)
            for a, cl in _ANGLE_PLANES: aw.axvline(a, color=cl, linestyle="--")
            aw.set_xlim(-180,180); aw.set_ylim(0,zmax); aw.set_xticks(_XTICKS_180); style(aw, False)

    fig.suptitle(f"Boundary Shapes (Centroid: {CFG['FRONT_CENTROID_MODE']}, Scale: {CFG['FRONT_SCALE_MODE']})", fontsize=14)
    return fig

# ═══════════════════════════════════════════════════════════════════
# Figure: Manifold evolution (entanglement)
# ═══════════════════════════════════════════════════════════════════
def plot_manifold_evolution(model_pairs, R_val, timesteps=[0, 5, 10, -1]):
    """PCA manifold colored by R1, R2, SUM, DIFF, ENTROPY across models & timesteps."""
    names = ["Trained", "Joint", "Naive", "Echo"]
    nM, nS = len(names), len(timesteps)
    nc = nM*nS + (nS-1)
    wr = ([1]*nM + [0.3])*nS; wr = wr[:-1]
    fig, axes = plt.subplots(5, nc, figsize=(8*nS, 12),
                             gridspec_kw={'width_ratios': wr}, constrained_layout=True)
    fig.suptitle("Representational Manifold Evolution: " +
                 " $\\rightarrow$ ".join(f"Step (t={t})" for t in timesteps),
                 fontsize=20, fontweight='bold')
    for row in axes:
        for s in range(nS-1): row[(s+1)*nM+s].axis('off')

    for tl, ti in enumerate(timesteps):
        for mi, mn in enumerate(names):
            if mn not in model_pairs: continue
            bel, model = model_pairs[mn]
            ctx = np.asarray(model.ctx_vals)
            r1, r2 = ctx[:,0].astype(int), ctx[:,1].astype(int)
            Xs = bel[:,ti]; Xf = Xs.reshape(len(Xs),-1)
            X3 = Xs.reshape(len(Xs),2,-1) if Xs.ndim == 2 else Xs
            ent = calc_entropy(X3[:,0]) + calc_entropy(X3[:,1])
            pca = PCA(n_components=2); Xp = pca.fit_transform(Xf); ve = pca.explained_variance_ratio_
            col = tl*(nM+1) + mi
            for row, (d, cm, yl) in enumerate([
                (r1,'viridis','R1'),(r2,'viridis','R2'),
                (r1+r2,'inferno','SUM'),(r1-r2,'inferno','DIFF'),(ent,'plasma','ENTROPY')]):
                ax = axes[row, col]
                ax.scatter(Xp[:,0], Xp[:,1], c=d, cmap=cm, s=150, alpha=0.1, edgecolor='none')
                sc = compute_r2(Xp, d)
                ax.text(0.95, 0.05, f"$R^2$: {sc:.2f}", transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  alpha=np.clip(1.5*sc,0.2,1), edgecolor='k'))
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(True, linestyle='--', alpha=0.3)
                if row == 0:
                    ax.set_title(f"{mn} | t={ti}\nPC1:{ve[0]:.1%} PC2:{ve[1]:.1%}",
                                 fontweight='bold', fontsize=10)
                if col == 0: ax.set_ylabel(yl, fontweight='bold', fontsize=12)
    return fig

# ═══════════════════════════════════════════════════════════════════
# Figure: Post-training diagnostics
# ═══════════════════════════════════════════════════════════════════
def _compute_variance_bundle(trained, echo, D=4):
    models = (trained, echo)
    T, R = min(int(trained.step_num), int(echo.step_num)), int(trained.realization_num)
    idx = np.arange(R); delta = np.abs(idx[:,None]-idx[None,:])
    cdist = np.minimum(delta, R-delta).astype(np.int64)
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
            for k, bel in (("bel", mdl.model_goal_belief), ("joint", mdl.joint_goal_belief),
                           ("naive", mdl.naive_goal_belief)):
                v = bel[mask,:T].var(0)
                mu[k][r1,r2] = v.mean(-1); dd[k][r1,r2] = v @ Wr.T
        io_bar[i] = [np.nanmean(inp), np.nanmean(out)]
        for k in avg:
            avg[k][i] = np.nanmean(mu[k], axis=(0,1))
            dist[k][i] = np.nanmean(dd[k], axis=(0,1))

    return np.arange(T)+1, io_bar, avg, dist

def _compute_logit_bundle(trained, echo):
    T = min(int(trained.step_num), int(echo.step_num))
    xax = np.arange(T)+1; R = int(trained.realization_num); eps = 1e-9
    cfg = {"Trained":(trained,"model"), "Echo":(echo,"model"),
           "Joint":(trained,"joint"), "Naive":(trained,"naive")}
    curves = {}
    for key, (m, attr) in cfg.items():
        vals = np.full((R,R,T), np.nan)
        bel = getattr(m, f"{attr}_goal_belief")[:,:T]
        for r1, r2, mask in _ctx_mask_iter(m, T):
            p = np.clip(bel[mask,:,r1], eps, 1-eps)
            vals[r1,r2] = np.log(p/(1-p)).mean(0)
        curves[key] = np.nanmean(vals, axis=(0,1))
    fn = lambda t, a, b, c: a*np.log(t) - b*t + c
    try:    popt, _ = curve_fit(fn, xax, curves["Naive"], p0=(1,0.05,-2), maxfev=10000)
    except: popt = (0.5, 0.05, -2)
    return xax, curves, popt[0]*np.log(xax), popt[1]*(xax-1)

def plot_post_training_diagnostics(trained, echo, D=3):
    """VARBAR, DIST0, LOGITPERF, LOGITSCALE + avg/per-distance belief variance."""
    T = min(int(trained.step_num), int(echo.step_num))
    xax, io_bar, avg, dist = _compute_variance_bundle(trained, echo, D=D)
    x_logit, lc, log_v, lin_v = _compute_logit_bundle(trained, echo)
    Bs = [trained.joint_goal_belief[:,:T], trained.model_goal_belief[:,:T],
          trained.naive_goal_belief[:,:T], echo.model_goal_belief[:,:T]]
    ncol = 1+D
    fig, axes = plt.subplots(2, ncol, figsize=(12, 4), constrained_layout=True)
    AC = AGENT_COLORS

    # Row 0 col 0: VARBAR
    a = axes[0,0]
    a.bar(range(4), [io_bar[0,0],io_bar[0,1],io_bar[1,0],io_bar[1,1]],
          color=(AC["Trained"],AC["Trained"],AC["Echo"],AC["Echo"]), alpha=0.82, edgecolor="k")
    a.set_xticks(range(4)); a.set_xticklabels(("Tr\nIn","Tr\nOut","Echo\nIn","Echo\nOut"))
    a.set_ylabel("variance"); a.set_title("Input/Output variance")

    # Row 0 col 1: DKL
    a = axes[0,1]; TT = T-1
    means = np.zeros((4,TT))
    for k in range(TT):
        for j, B in enumerate(Bs): means[j,k] = sym_dkl_pair(B[:,0], B[:,k+1]).mean()
    xp = np.arange(TT)+1
    for idx, (name, ls) in enumerate([("Trained","-"),("Echo","-"),("Joint","--"),("Naive","--")]):
        order = [1,3,0,2][idx]
        a.plot(xp, means[order], ls+"o", ms=2.2, lw=1.8, c=AC[name], label=name.lower())
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("t"); a.set_ylabel("symDKL"); a.set_title("DKL(B_0, B_t)")
    a.legend(frameon=False, fontsize=9)

    # Row 0 col 2: LOGITPERF
    a = axes[0,2]
    _plot_4agents(a, x_logit, lc, lw=2.5)
    a.set_title("True Positive Logit"); a.set_xlabel("t"); a.grid(alpha=0.25)
    a.legend(frameon=False, loc=2, fontsize=9)

    # Row 0 col 3: LOGITSCALE
    a = axes[0,3]
    a.fill_between(x_logit, -log_v, log_v, color=AC["Joint"], alpha=0.15, label=r"$\pm\ln t$")
    a.plot(x_logit, log_v, ":", c=AC["Joint"], lw=1.1)
    a.plot(x_logit, -log_v, ":", c=AC["Joint"], lw=1.1)
    a.fill_between(x_logit, -lin_v, lin_v, color=AC["Naive"], alpha=0.15, label=r"$\pm t$")
    a.plot(x_logit, lin_v, "--", c=AC["Naive"], lw=1.1)
    a.plot(x_logit, -lin_v, "--", c=AC["Naive"], lw=1.1)
    a.set_title("Contributions scale differently"); a.set_xlabel("t"); a.grid(alpha=0.25)
    a.legend(frameon=False, loc=2, fontsize=9)
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
        _var_panel(ad, dist["bel"][0,:,d], dist["bel"][1,:,d],
                   dist["joint"][0,:,d], dist["naive"][0,:,d], f"Distance = {d}")
        if d > 0: ad.sharey(axes[1,1]); plt.setp(ad.get_yticklabels(), visible=False)
    return fig

# ═══════════════════════════════════════════════════════════════════
# Figure: re-evaluation
# ═══════════════════════════════════════════════════════════════════
def plot_re_evaluation(trained):
    n_px = trained.naive_px / trained.naive_px.sum(axis=-1, keepdims=True)
    j_px_raw = trained.joint_px / trained.joint_px.sum(axis=(-1, -2), keepdims=True)
    j_px = np.stack([j_px_raw.sum(axis=-1), j_px_raw.sum(axis=-2)], axis=2)
    reval_n = compute_stepwise_dkl(n_px, approximate_likelihood(trained.naive_belief))
    reval_j = compute_stepwise_dkl(j_px, approximate_likelihood(trained.joint_belief))
    time_len = n_px.shape[1]
    fig, ax = plt.subplots(1, 4, figsize=(10, 3), dpi=CFG["DPI"])
    c, t_ax = np.arange(1, time_len), np.arange(1, time_len)
    j_mean = reval_j.mean(axis=(0, 2))
    ax[0].plot(np.arange(1, time_len + 1), reval_n.mean(axis=(0, 2)), 'C2--o', ms=2, lw=1, alpha=0.6, label="Naive")
    ax[0].plot(np.arange(1, time_len + 1), j_mean, 'C1-o', ms=2, lw=1, label="Joint")    
    ax[0].set(title="Linear scale", xlabel="Time step (t)", ylabel="Avg Re-evaluation")
    ax[0].legend(fontsize=10, frameon=False)
    ax[1].plot(t_ax, j_mean[1:], 'k-', lw=2)
    ax[1].scatter(t_ax, j_mean[1:], c=c, cmap='coolwarm', edgecolor='k', lw=0.5, s=40, zorder=2)
    ax[1].set(title="Log scale", xlabel="Time step (t)", xscale='log', yscale='log')
    ax[1].tick_params(which='both', left=False, labelleft=False)
    cum_j = reval_j[:, 1:]
    mu_x, mu_y = cum_j[..., 0].mean(axis=0), cum_j[..., 1].mean(axis=0)
    ax[2].plot(mu_x, mu_y, 'k-', lw=1.5, zorder=1)
    sc = ax[2].scatter(mu_x, mu_y, c=c, cmap='coolwarm', edgecolor='k', lw=0.5, s=40, zorder=2)
    ax[2].set(title="Re-evaluation Covariance", xlabel="R1 (log scale)", ylabel="R2 (log scale)", xscale='log', yscale='log')
    ax[2].tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    fig.colorbar(sc, ax=ax[2], label="Time step (t)")
    x_data, y_data = cum_j[..., 0].flatten(), cum_j[..., 1].flatten()
    mask = (x_data > 0) & (y_data > 0)
    hb = ax[3].hexbin(x_data[mask], y_data[mask], gridsize=250, cmap='plasma', mincnt=1, bins='log', xscale='log', yscale='log')
    ax[3].set(title="Re-evaluation Covariance", xlabel="R1 (log scale)", ylabel="R2 (log scale)")
    fig.colorbar(hb, ax=ax[3], label="Density")
    ax[3].tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    fmt = ticker.LogFormatterMathtext()
    for i, a in enumerate(ax):
        style(a, dark=False)
        if i in [1, 3]:  
            a.xaxis.set_major_formatter(fmt)
            a.yaxis.set_major_formatter(fmt)
    fig.suptitle(r"Re-evaluation: DKL(marginal likelihood $\parallel \frac{\text{marginal posterior}}{\text{marginal prior}}$)")
    fig.tight_layout()
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    common = dict(mode="SANITY", cuda=0, episodes=1, checkpoint_every=5,
                  realization_num=10, hid_dim=1000, obs_num=5, show_plots=False,
                  batch_num=15000, step_num=30, state_num=500,
                  learn_embeddings=False, classifier_LR=.001, ctx_num=2, training=False)

    # echo    = CognitiveGridworld(**{**common, 'reservoir': True,  'load_env': "/sanity/reservoir_ctx_2_e5"})
    # trained = CognitiveGridworld(**{**common, 'reservoir': False, 'load_env': "/sanity/fully_trained_ctx_2_e5"})

    _pca = lambda m, attr, lbl: (*project_pca(getattr(m, attr), mode=CFG["PLOT_MODE"]), npy(m.ctx_vals), lbl)

    # Paired PCA grids
    # render_paired_pca_grids([_pca(trained,"model_belief_flat","TRAINED"), _pca(trained,"joint_belief","JOINT")], fig_title="Trained + Joint")
    # render_paired_pca_grids([_pca(echo,"model_belief_flat","ECHO"), _pca(echo,"naive_belief","NAIVE")], fig_title="Echo + Naive")
    plot_boundary_shape_combined(trained)
    # Manifold evolution
    plot_manifold_evolution({"Trained":(trained.model_belief_flat,trained), "Joint":(trained.joint_belief,trained),
                             "Naive":(echo.naive_belief,echo), "Echo":(echo.model_belief_flat,echo)}, trained.realization_num, timesteps=[0, 15,-1])
    # Post-training diagnostics
    plot_post_training_diagnostics(trained, echo)
    # Structural Friction (New Plot)
    plot_re_evaluation(trained)
