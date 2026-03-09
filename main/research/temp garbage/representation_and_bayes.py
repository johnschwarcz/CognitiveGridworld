"""
REPRESENTATION & BAYES - Single-episode representation + Bayesian flow-field.
Consolidates representation_analysis.py + compare_bayes.py.
"""

import numpy as np
import os, sys, inspect
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from sklearn.decomposition import PCA

path = inspect.getfile(inspect.currentframe())
path = os.path.dirname(os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld

CFG = dict(
    DPI=140, FACECOLOR="black", GRID_ALPHA=0.07,
    PLOT_MODE="all",        # 0, ... 29, all
    EPS=1e-10, T_PLOT=None, STEP_STRIDE=1,
    POINTS_PER_STEP=0,      # remove scatter entirely — let contours carry the info
    CONTOUR_BINS=100,         # fewer bins, smoother with less noise
    CONTOUR_QHI=95,          # wider bin range to capture more outliers
    CONTOUR_MASS=0.90,       # tighter contours around high-density core
    CONTOUR_SMOOTH=2,
    CONTOUR_PAD=1,

    FRONT_ANGLE_BINS=100, FRONT_SMOOTH_SIGMA=3, FRONT_SCALE_MODE="none",
    FRONT_CENTROID_MODE="t0",
    VIEW_ELEV=25, VIEW_AZIM=-55, PLANE_ALPHA=0.2, INTERSECT_LW=1.8
)

plt.rcParams.update({
    "figure.dpi": CFG["DPI"], "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
})

# Shared utilities + representation analysis

def npy(x):
    return None if x is None else x.detach().cpu().numpy() if hasattr(x,"detach") else np.asarray(x)

def renorm(P):
    P=np.asarray(P,dtype=np.float64)
    d=P.sum(-1,keepdims=True)
    d=np.where(d>0,d,1.0)
    return P/d

def logit(p,clip=False):
    p=np.asarray(p,dtype=np.float64)
    if clip: p=np.clip(p,CFG["EPS"],1.0-CFG["EPS"])
    with np.errstate(divide="ignore",invalid="ignore"):
        return np.log(p/(1.0-p)).astype(np.float32)

def style(ax,dark=False):
    if dark:
        ax.set_facecolor(CFG["FACECOLOR"])
        for sp in ax.spines.values(): sp.set_color("0.5")
        ax.grid(alpha=CFG["GRID_ALPHA"],color="white")
        ax.tick_params(axis="both",colors="0.9",labelsize=8)
    else:
        ax.grid(True,alpha=0.22)
        try:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        except Exception:
            pass

def robust_lo_hi(v,qhi):
    u=np.asarray(v)[np.isfinite(v)]
    if u.size==0: return -1.0,1.0
    if qhi>=100.0: lo,hi=float(u.min()),float(u.max())
    else: lo,hi=float(np.percentile(u,100.0-qhi)),float(np.percentile(u,qhi))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi<=lo:
        m=float(np.nanmean(u)) if np.isfinite(np.nanmean(u)) else 0.0
        s=float(np.nanstd(u)) if np.isfinite(np.nanstd(u)) and np.nanstd(u)>0 else 1.0
        lo,hi=m-3.0*s,m+3.0*s
    return lo,hi

def mass_level(H,mass,normalize=False):
    H=np.asarray(H,dtype=np.float64)
    s=H.sum()
    if s<=0: return np.nan
    A=H/s if normalize else H
    flat=A.reshape(-1)
    flat=flat[flat>0]
    if flat.size==0: return np.nan
    flat=np.sort(flat)[::-1]
    c=np.cumsum(flat)
    return float(flat[np.searchsorted(c,mass*c[-1],side="left")])

def project_pca(X,mode="all"):
    X=npy(X)
    if isinstance(mode,int): X=X[:,mode:mode+1]
    B,T=X.shape[:2]
    Y=PCA(n_components=2).fit_transform(X.reshape(B*T,-1)).reshape(B,T,2)
    ev=PCA(n_components=2).fit(X.reshape(B*T,-1)).explained_variance_ratio_*100
    return Y,ev

def coerce_goal_ind(goal_ind,B,T):
    g=npy(goal_ind)
    if g.ndim==0: return np.full((B,T),int(g),np.int64)
    if g.ndim==1:
        if g.shape[0]==B: return np.repeat(g[:,None].astype(np.int64),T,1)
        if g.shape[0]==T: return np.repeat(g[None,:].astype(np.int64),B,0)
        return np.full((B,T),int(g.reshape(-1)[0]),np.int64)
    if g.ndim==2 and g.shape==(B,T): return g.astype(np.int64)
    return np.full((B,T),int(np.ravel(g)[0]),np.int64)

def neighbor_table(R,K):
    if K==-1:
        M=max(R-1,1)
        tab=np.empty((R,M),np.int64)
        for r in range(R):
            for j in range(M): tab[r,j]=(r+1+j)%R
        return tab
    M=2*int(K)
    tab=np.empty((R,M),np.int64)
    for r in range(R):
        for d in range(1,int(K)+1):
            j=2*(d-1)
            tab[r,j],tab[r,j+1]=(r-d)%R,(r+d)%R
    return tab

def nong_reduce(A_BTS,gBT):
    B,T,S=A_BTS.shape
    bix=np.arange(B,dtype=np.int64)[:,None]
    tix=np.arange(T,dtype=np.int64)[None,:]
    sng=(gBT+1)%S
    return A_BTS[bix,tix,sng]




def plot_step_contours(ax,X,Y,steps,ex,ey):
    cx,cy=0.5*(ex[:-1]+ex[1:]),0.5*(ey[:-1]+ey[1:])
    Xc,Yc=np.meshgrid(cx,cy,indexing="xy")
    mX,mY=np.full(steps.size,np.nan),np.full(steps.size,np.nan)
    cmap=plt.cm.coolwarm
    sigma=CFG["CONTOUR_SMOOTH"]
    for i,st in enumerate(steps):
        st=int(st)
        if st<1 or st>steps[-1] or ((st-1)%CFG["STEP_STRIDE"])!=0: continue
        H,_,_=np.histogram2d(X[:,i],Y[:,i],bins=(ex,ey))
        if sigma>0:
            H=gaussian_filter(H.astype(np.float64),sigma=sigma)
        Hn=H/np.maximum(H.sum(),1.0)
        lvl=mass_level(Hn,CFG["CONTOUR_MASS"],normalize=False)
        n=X.shape[0]
        pps=int(CFG["POINTS_PER_STEP"])
        if pps>0 and n>0:
            idx=np.linspace(0,n-1,min(n,pps),dtype=np.int64)
            ax.scatter(X[idx,i],Y[idx,i],c=np.full(idx.size,float(st)),cmap=cmap,vmin=1,vmax=steps[-1],s=7,alpha=0.20,linewidths=0,zorder=2)
        if np.isfinite(lvl) and lvl>0:
            ax.contour(Xc,Yc,Hn.T,levels=(lvl,),colors=(cmap((st-1)/max(steps[-1]-1,1)),),linewidths=1.6,alpha=0.95,zorder=3)
        mX[i],mY[i]=float(np.nanmean(X[:,i])),float(np.nanmean(Y[:,i]))
    ok=np.isfinite(mX)&np.isfinite(mY)
    if np.any(ok):
        ax.plot(mX[ok],mY[ok],lw=1.6,alpha=0.85,zorder=6)
        frac=(steps[ok].astype(np.float64)-1.0)/float(max(steps[-1]-1,1))
        ax.scatter(mX[ok],mY[ok],c=frac,cmap=cmap,s=34,alpha=0.98,linewidths=0,zorder=7)

def calc_true_contour_front(X,Y,ex,ey,step_max):
    angle_bins=CFG["FRONT_ANGLE_BINS"]
    theta_edges=np.linspace(-np.pi,np.pi,angle_bins+1)
    theta_centers=0.5*(theta_edges[:-1]+theta_edges[1:])
    fronts=np.full((step_max,angle_bins),np.nan,dtype=np.float64)
    mx0,my0=0.0,0.0
    if CFG["FRONT_CENTROID_MODE"]=="t0":
        x0,y0=X[:,0],Y[:,0]
        ok0=np.isfinite(x0)&np.isfinite(y0)
        if np.any(ok0): mx0,my0=float(np.mean(x0[ok0])),float(np.mean(y0[ok0]))
    for t in range(step_max):
        x,y=X[:,t],Y[:,t]
        ok=np.isfinite(x)&np.isfinite(y)
        if not np.any(ok): continue
        if CFG["FRONT_CENTROID_MODE"]=="per_step": mx,my=float(np.mean(x[ok])),float(np.mean(y[ok]))
        elif CFG["FRONT_CENTROID_MODE"]=="t0": mx,my=mx0,my0
        else: mx,my=0.0,0.0
        H,xed,yed=np.histogram2d(x,y,bins=(ex,ey))
        lvl=mass_level(H,CFG["CONTOUR_MASS"],normalize=False)
        ix=np.clip(np.searchsorted(xed,x,side="right")-1,0,len(xed)-2)
        iy=np.clip(np.searchsorted(yed,y,side="right")-1,0,len(yed)-2)
        m=(H[ix,iy]>=lvl)&ok
        if not np.any(m): continue
        xs,ys=x[m]-mx,y[m]-my
        r=np.sqrt(xs*xs+ys*ys)
        th=(np.arctan2(ys,xs)-np.pi/4.0+np.pi)%(2*np.pi)-np.pi
        raw=np.full(angle_bins,np.nan)
        for i in range(angle_bins):
            b=(th>=theta_edges[i])&(th<theta_edges[i+1])
            if np.sum(b)>3: raw[i]=float(np.percentile(r[b],95.0))
        okr=np.isfinite(raw)
        if np.sum(okr)>3:
            s=np.interp(np.arange(angle_bins),np.where(okr)[0],raw[okr])
            if CFG["FRONT_SMOOTH_SIGMA"]>0: s=gaussian_filter1d(s,sigma=CFG["FRONT_SMOOTH_SIGMA"],mode="wrap")
            if CFG["FRONT_SCALE_MODE"]=="mean": s/=(np.nanmean(s) if np.nanmean(s)>1e-8 else 1.0)
            elif CFG["FRONT_SCALE_MODE"]=="max": s/=(np.nanmax(s) if np.nanmax(s)>1e-8 else 1.0)
            fronts[t]=s
    return fronts,theta_centers


def render_paired_pca_grids(models_data, fig_title="PCA Projections"):
    """
    PCA projection grid for a PAIR of models: 4 rows (2 per model: R1/R2-primary) × 7 columns.
    models_data: list of exactly 2 tuples: (data_2d, ev_2d, ctx_vals, label).
    Returns one figure.
    """
    n_grid_rows = len(models_data) * 2
    fig = plt.figure(figsize=(20, 2.25 * n_grid_rows), dpi=CFG["DPI"])
    fig.patch.set_facecolor(CFG["FACECOLOR"])
    gs = fig.add_gridspec(n_grid_rows, 7, hspace=0.2, wspace=0.1)

    for model_idx, (data, ev, ctx_raw, title_label) in enumerate(models_data):
        ctx = npy(ctx_raw)
        while ctx.ndim > 3 and ctx.shape[-1] == 1: ctx = ctx[..., 0]
        if ctx.ndim == 3: ctx = ctx[:, 0]
        u_r = np.unique(ctx[:, :2])
        n_R = len(u_r)
        den = max(n_R - 1, 1)
        cmap = plt.get_cmap("plasma")
        pad_x = (data[..., 0].max() - data[..., 0].min()) * 0.35
        pad_y = (data[..., 1].max() - data[..., 1].min()) * 0.35
        xlim = (data[..., 0].min() - pad_x, data[..., 0].max() + pad_x)
        ylim = (data[..., 1].min() - pad_y, data[..., 1].max() + pad_y)

        def plot_traj(ax, mask, color, ls, label):
            if not mask.any(): return
            traj = data[mask].mean(0)
            ax.plot(traj[:, 0], traj[:, 1], color=color, ls=ls, alpha=0.8, lw=2, label=label)
            ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o", s=20)
            ax.scatter(traj[-1, 0], traj[-1, 1], color=color, marker="X", s=60, edgecolor="black", lw=0.5)

        for row_sub in range(2):
            grid_row = model_idx * 2 + row_sub
            is_r1_primary = (row_sub == 0)
            shift_type = "R1=(R2+{c})%n" if is_r1_primary else "R2=(R1+{c})%n"
            is_last_row = (grid_row == n_grid_rows - 1)

            for c in range(6):
                ax = fig.add_subplot(gs[grid_row, c]); style(ax, True)
                ax.set_xlim(xlim); ax.set_ylim(ylim)
                ax.set_title(shift_type.format(c=c), color="0.95", fontsize=6)
                if c == 0:
                    lbl = f"PC2 ({ev[1]:.1f}%)\n{title_label}" if row_sub == 0 else f"PC2 ({ev[1]:.1f}%)"
                    ax.set_ylabel(lbl, color="cyan", fontsize=6, fontweight="bold")
                ax.tick_params(labelleft=False)
                if is_last_row:
                    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)", color="0.95", fontsize = 6)
                ax.tick_params(labelbottom=False)
                for i, r_val in enumerate(u_r):
                    if is_r1_primary:
                        r1, r2 = (r_val + c) % n_R, r_val
                    else:
                        r1, r2 = r_val, (r_val + c) % n_R
                    plot_traj(ax, (ctx[:, 0] == r1) & (ctx[:, 1] == r2), cmap(i / den), "-", f"({r1},{r2})")
                if row_sub < 2 and model_idx == 0:
                    ax.legend(loc="best", fontsize=6, ncol=1, framealpha=0.1, labelcolor="0.95")

            ax = fig.add_subplot(gs[grid_row, 6]); style(ax, True)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            marg_label = "R2" if is_r1_primary else "R1"
            ax.set_title(f"Marginal: {marg_label}", color="yellow", fontsize=6)
            ax.tick_params(labelleft=False, labelbottom=is_last_row)
            for i, r in enumerate(u_r):
                mask = ctx[:, 1] == r if is_r1_primary else ctx[:, 0] == r
                plot_traj(ax, mask, cmap(i / den), "--" if is_r1_primary else "-", f"R={r}")
                if row_sub < 2 and model_idx == 0:
                    ax.legend(loc="best", fontsize=6, ncol=1, framealpha=0.1, labelcolor="0.95")
    return fig


def plot_boundary_shape_combined(trained):
    """Combined boundary shape figure: 3 rows × 6 columns.
    Rows: Contours, 3D manifolds, Waterfall.
    Columns: J-True, N-True, J-near(K=1), N-near(K=1), J-near(all), N-near(all).
    """
    bix = npy(getattr(trained, "batch_range", None))
    Pj_full, Pn_full = npy(trained.joint_belief), npy(trained.naive_belief)
    bix = bix.astype(np.int64) if bix is not None else np.arange(Pj_full.shape[0], dtype=np.int64)
    Pj, Pn = renorm(Pj_full[bix]), renorm(Pn_full[bix])
    B, T, S, R = Pj.shape
    Tplot = T if CFG["T_PLOT"] is None else int(np.clip(int(CFG["T_PLOT"]), 1, T))
    gBT = coerce_goal_ind(trained.goal_ind, B, T)
    ctx = npy(trained.ctx_vals)[bix]
    while ctx.ndim > 3 and ctx.shape[-1] == 1: ctx = ctx[..., 0]
    if ctx.ndim == 2: ctx = np.repeat(ctx[:, None, :], T, axis=1)
    ctx = ctx.astype(np.int64)
    bix2 = np.arange(B, dtype=np.int64)[:, None]
    tix2 = np.arange(T, dtype=np.int64)[None, :]
    tr_goal = ctx[bix2, tix2, gBT]
    tab1, tabA = neighbor_table(R, 1), neighbor_table(R, -1)
    pj_st = np.take_along_axis(Pj, ctx[..., None], 3)[..., 0]
    pn_st = np.take_along_axis(Pn, ctx[..., None], 3)[..., 0]

    pj_nr, pn_nr = np.zeros((B, T, S, 2)), np.zeros((B, T, S, 2))
    pj_g_nr, pn_g_nr = np.zeros((B, T, 2)), np.zeros((B, T, 2))
    for k, tab in enumerate((tab1, tabA)):
        pj_nr[..., k] = np.take_along_axis(Pj, tab[ctx], 3).mean(3)
        pn_nr[..., k] = np.take_along_axis(Pn, tab[ctx], 3).mean(3)
        pj_g_nr[..., k] = np.take_along_axis(Pj[bix2, tix2, gBT], tab[tr_goal], 2).mean(2)
        pn_g_nr[..., k] = np.take_along_axis(Pn[bix2, tix2, gBT], tab[tr_goal], 2).mean(2)

    XJ, YJ, XN, YN = (np.zeros((B, T, 3), dtype=np.float32) for _ in range(4))
    XJ[..., 0], YJ[..., 0] = logit(pj_st[bix2, tix2, gBT], clip=True), logit(nong_reduce(pj_st, gBT), clip=True)
    XN[..., 0], YN[..., 0] = logit(pn_st[bix2, tix2, gBT], clip=True), logit(nong_reduce(pn_st, gBT), clip=True)
    for k in range(2):
        c = k + 1
        XJ[..., c] = XJ[..., 0] - logit(pj_g_nr[..., k], clip=True)
        YJ[..., c] = YJ[..., 0] - logit(nong_reduce(pj_nr[..., k], gBT), clip=True)
        XN[..., c] = XN[..., 0] - logit(pn_g_nr[..., k], clip=True)
        YN[..., c] = YN[..., 0] - logit(nong_reduce(pn_nr[..., k], gBT), clip=True)

    col_titles = ("True", "True − near (K=1)", "True − near (all)")
    steps = np.arange(1, Tplot + 1)
    cmap = plt.cm.coolwarm

    # 3 rows × 6 cols: each subtraction mode gets 2 columns (Joint, Naive)
    fig = plt.figure(figsize=(20, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.1)

    for sub_c in range(3):  # subtraction modes: True, near-K1, near-all
        xj, yj = XJ[:, :Tplot, sub_c], YJ[:, :Tplot, sub_c]
        xn, yn = XN[:, :Tplot, sub_c], YN[:, :Tplot, sub_c]
        x0, x1 = robust_lo_hi(np.concatenate((xj.ravel(), xn.ravel())), CFG["CONTOUR_QHI"])
        y0, y1 = robust_lo_hi(np.concatenate((yj.ravel(), yn.ravel())), CFG["CONTOUR_QHI"])
        dx, dy = (x1 - x0) * CFG["CONTOUR_PAD"], (y1 - y0) * CFG["CONTOUR_PAD"]
        ex = np.linspace(x0 - dx, x1 + dx, CFG["CONTOUR_BINS"] + 1)
        ey = np.linspace(y0 - dy, y1 + dy, CFG["CONTOUR_BINS"] + 1)

        fJ, deg = calc_true_contour_front(xj, yj, ex, ey, Tplot)
        fN, _ = calc_true_contour_front(xn, yn, ex, ey, Tplot)
        deg_plot = deg * 180 / np.pi
        xm, ym = np.meshgrid(deg_plot, steps)
        z_max = max(np.nanmax(fJ), np.nanmax(fN)) * 1.05
        yp, zp = np.meshgrid((1, Tplot), (0, z_max))

        for jn_idx, (x, y, fronts, title) in enumerate([
            (xj, yj, fJ, "Joint"), (xn, yn, fN, "Naive")
        ]):
            gc = sub_c * 2 + jn_idx  # grid column

            # Row 0: Contours
            ax = fig.add_subplot(gs[0, gc])
            ax.set_title(f"{title}: {col_titles[sub_c]}", fontsize=10)
            plot_step_contours(ax, x, y, steps, ex, ey)
            ax.axvline(0, lw=1, alpha=0.5); ax.axhline(0, lw=1, alpha=0.5)
            style(ax, False)

            # Row 1: 3D
            ax3d = fig.add_subplot(gs[1, gc], projection="3d")
            ax3d.set_title(f"{title}: {col_titles[sub_c]}", fontsize=9)
            v = np.where(~np.isnan(fronts).all(1))[0]
            if v.size:
                sl = slice(v[0], v[-1] + 1)
                ax3d.plot_surface(xm[sl], ym[sl], fronts[sl], facecolors=cmap(plt.Normalize(1, Tplot)(ym[sl])), shade=True, lw=0, alpha=0.9)
            for a, cl in zip((-90, -45, 0, 45, 90), ("red", "green", "lightgreen", "green", "red")):
                ax3d.plot_surface(np.full_like(yp, a), yp, zp, color=cl, alpha=CFG["PLANE_ALPHA"], shade=False)
                idx = np.argmin(np.abs(deg_plot - a))
                ax3d.plot(np.full(Tplot, a), steps, fronts[:, idx], color=cl, lw=CFG["INTERSECT_LW"], zorder=10)
            ax3d.set_xlim(-180, 180); ax3d.set_ylim(1, Tplot); ax3d.set_zlim(0, z_max)
            ax3d.set_xticks((-180, -90, -45, 0, 45, 90, 180)); ax3d.view_init(CFG["VIEW_ELEV"], CFG["VIEW_AZIM"])

            # Row 2: Waterfall
            ax2d = fig.add_subplot(gs[2, gc])
            ax2d.set_title(f"{title}: {col_titles[sub_c]}", fontsize=9)
            for t in range(Tplot):
                if np.any(np.isfinite(fronts[t])):
                    ax2d.plot(deg_plot, fronts[t], color=cmap(t / max(Tplot, 1)), lw=1.2, alpha=0.8)
            for a, cl in zip((-90, -45, 0, 45, 90), ("red", "green", "lightgreen", "green", "red")):
                ax2d.axvline(a, color=cl, linestyle="--")
            ax2d.set_xlim(-180, 180); ax2d.set_ylim(0, z_max)
            ax2d.set_xticks((-180, -90, -45, 0, 45, 90, 180))
            style(ax2d, False)

    fig.suptitle(f"Boundary Shapes (Centroid: {CFG['FRONT_CENTROID_MODE']}, Scale: {CFG['FRONT_SCALE_MODE']})", fontsize=14)
    return fig

def sym_dkl_pair(P, Q, eps=1e-4):
    P = np.clip(P, eps, 1.0 - eps)
    Q = np.clip(Q, eps, 1.0 - eps)
    P = P / P.sum(-1, keepdims=True)
    Q = Q / Q.sum(-1, keepdims=True)
    dPQ = np.sum(P * (np.log(P) - np.log(Q)), axis=-1)
    dQP = np.sum(Q * (np.log(Q) - np.log(P)), axis=-1)
    return 0.5 * (dPQ + dQP)

def _compute_variance_bundle(trained, echo, D=4):
    models = (trained, echo)
    T = min(int(trained.step_num), int(echo.step_num))
    R = int(trained.realization_num)

    idx = np.arange(R)
    delta = np.abs(idx[:, None] - idx[None, :])
    circ_dist = np.minimum(delta, R - delta).astype(np.int64)

    W = np.zeros((R, D, R), dtype=np.float64)
    for r1 in range(R):
        for d in range(D):
            m = circ_dist[r1] == d
            c = m.sum()
            if c > 0:
                W[r1, d, m] = 1.0 / c

    io_bar = np.zeros((2, 2), dtype=np.float64)
    avg_bel = np.full((2, T), np.nan)
    avg_joint = np.full((2, T), np.nan)
    avg_naive = np.full((2, T), np.nan)
    dist_bel = np.full((2, T, D), np.nan)
    dist_joint = np.full((2, T, D), np.nan)
    dist_naive = np.full((2, T, D), np.nan)

    for i, m in enumerate(models):
        inp = np.full((R, R, T), np.nan)
        out = np.full((R, R, T), np.nan)
        bmu = np.full((R, R, T), np.nan)
        jmu = np.full((R, R, T), np.nan)
        nmu = np.full((R, R, T), np.nan)
        bd = np.full((R, R, T, D), np.nan)
        jd = np.full((R, R, T, D), np.nan)
        nd = np.full((R, R, T, D), np.nan)

        for r1 in m.realization_range:
            for r2 in m.realization_range:
                mask = (m.goal_ind == 0) & (m.ctx_vals[:, 0] == r1) & (m.ctx_vals[:, 1] == r2)
                if mask.sum() > 1:
                    xin = m.model_input_flat[mask, :T]
                    xout = m.model_update_flat[mask, :T]
                    b = m.model_goal_belief[mask, :T]
                    j = m.joint_goal_belief[mask, :T]
                    n = m.naive_goal_belief[mask, :T]

                    inp[r1, r2] = xin.var(0).mean(-1)
                    out[r1, r2] = xout.var(0).mean(-1)

                    vb = b.var(0)
                    vj = j.var(0)
                    vn = n.var(0)

                    bmu[r1, r2] = vb.mean(-1)
                    jmu[r1, r2] = vj.mean(-1)
                    nmu[r1, r2] = vn.mean(-1)

                    Wr = W[r1]
                    bd[r1, r2] = vb @ Wr.T
                    jd[r1, r2] = vj @ Wr.T
                    nd[r1, r2] = vn @ Wr.T

        io_bar[i, 0] = np.nanmean(inp)
        io_bar[i, 1] = np.nanmean(out)
        avg_bel[i] = np.nanmean(bmu, axis=(0, 1))
        avg_joint[i] = np.nanmean(jmu, axis=(0, 1))
        avg_naive[i] = np.nanmean(nmu, axis=(0, 1))
        dist_bel[i] = np.nanmean(bd, axis=(0, 1))
        dist_joint[i] = np.nanmean(jd, axis=(0, 1))
        dist_naive[i] = np.nanmean(nd, axis=(0, 1))

    xax = np.arange(T) + 1
    return xax, io_bar, avg_bel, avg_joint, avg_naive, dist_bel, dist_joint, dist_naive

# ═══════════════════════════════════════════════════════════════════
# Logit bundle
# ═══════════════════════════════════════════════════════════════════

def _compute_logit_bundle(trained, echo):
    T = min(int(trained.step_num), int(echo.step_num))
    xax = np.arange(T) + 1
    R = int(trained.realization_num)
    eps = 1e-9
    cfg = {
        "Trained": (trained, "model"),
        "Echo": (echo, "model"),
        "Joint": (trained, "joint"),
        "Naive": (trained, "naive")
    }
    curves = {}

    for key in cfg:
        m, attr = cfg[key]
        vals = np.full((R, R, T), np.nan)
        bel = getattr(m, f"{attr}_goal_belief")[:, :T]
        for r1 in m.realization_range:
            for r2 in m.realization_range:
                mask = (m.goal_ind == 0) & (m.ctx_vals[:, 0] == r1) & (m.ctx_vals[:, 1] == r2)
                if mask.sum() > 1:
                    p = np.clip(bel[mask, :, r1], eps, 1.0 - eps)
                    vals[r1, r2] = np.log(p / (1.0 - p)).mean(0)
        curves[key] = np.nanmean(vals, axis=(0, 1))

    fn = lambda t, a, b, c: a * np.log(t) - b * t + c
    try:
        popt, _ = curve_fit(fn, xax, curves["Naive"], p0=(1.0, 0.05, -2.0), maxfev=10000)
    except Exception:
        popt = (0.5, 0.05, -2.0)
    a, b = popt[0], popt[1]
    return xax, curves, a * np.log(xax), b * (xax - 1.0)

AGENT_COLORS = {"Trained": "C0", "Echo": "C1", "Joint": "C2", "Naive": "C3"}
NET_Z0, NET_Z1 = 2, 3
BAYES_Z0, BAYES_Z1 = 5, 6

def plot_post_training_diagnostics(trained, echo, D=3):
    """Post-training diagnostics: VARBAR, DIST0, LOGITPERF, LOGITSCALE,
    avg belief variance, and per-distance variance panels."""
    from scipy.optimize import curve_fit

    T = min(int(trained.step_num), int(echo.step_num))
    vb = _compute_variance_bundle(trained, echo, D=D)
    io_bar = vb[1]
    x_logit, logit_curves, log_v, lin_v = _compute_logit_bundle(trained, echo)

    B0 = trained.joint_goal_belief[:, :T]
    B1 = trained.model_goal_belief[:, :T]
    B2 = trained.naive_goal_belief[:, :T]
    B3 = echo.model_goal_belief[:, :T]

    n_cols = 1 + D  # avg + per-distance
    fig, axes = plt.subplots(2, n_cols, figsize=(12, 4), constrained_layout=True)

    # ── Row 0: VARBAR, DIST0, LOGITPERF, LOGITSCALE ──
    # VARBAR
    a = axes[0, 0]
    vals = np.array((io_bar[0, 0], io_bar[0, 1], io_bar[1, 0], io_bar[1, 1]), float)
    x = np.arange(4)
    a.bar(x, vals, color=(AGENT_COLORS["Trained"], AGENT_COLORS["Trained"], AGENT_COLORS["Echo"], AGENT_COLORS["Echo"]), alpha=0.82, edgecolor="k")
    a.set_xticks(x); a.set_xticklabels(("Tr\nIn", "Tr\nOut", "Echo\nIn", "Echo\nOut"))
    a.set_ylabel("variance"); a.set_title("Input/Output variance")

    # DIST0
    a = axes[0, 1]
    TT = T - 1
    means = np.zeros((4, TT), float)
    for k in range(TT):
        t1 = k + 1
        means[0, k] = sym_dkl_pair(B0[:, 0], B0[:, t1]).mean()
        means[1, k] = sym_dkl_pair(B1[:, 0], B1[:, t1]).mean()
        means[2, k] = sym_dkl_pair(B2[:, 0], B2[:, t1]).mean()
        means[3, k] = sym_dkl_pair(B3[:, 0], B3[:, t1]).mean()
    xp = np.arange(TT) + 1
    a.plot(xp, means[1], "-o", ms=2.2, lw=1.8, c=AGENT_COLORS["Trained"], label="trained")
    a.plot(xp, means[3], "-o", ms=2.2, lw=1.8, c=AGENT_COLORS["Echo"], label="echo")
    a.plot(xp, means[0], "--o", ms=2.2, lw=1.8, c=AGENT_COLORS["Joint"], label="joint")
    a.plot(xp, means[2], "--o", ms=2.2, lw=1.8, c=AGENT_COLORS["Naive"], label="naive")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("t"); a.set_ylabel("symDKL"); a.set_title("DKL(B_0, B_t)")
    a.legend(frameon=False, fontsize=9)

    # LOGITPERF
    a = axes[0, 2]
    a.plot(x_logit, logit_curves["Trained"], c=AGENT_COLORS["Trained"], lw=2.5, label="Trained")
    a.plot(x_logit, logit_curves["Echo"], c=AGENT_COLORS["Echo"], lw=2.5, label="Echo")
    a.plot(x_logit, logit_curves["Joint"], c=AGENT_COLORS["Joint"], lw=2.5, ls="--", label="Joint")
    a.plot(x_logit, logit_curves["Naive"], c=AGENT_COLORS["Naive"], lw=2.5, ls="--", label="Naive")
    a.set_title("True Positive Logit"); a.set_xlabel("t"); a.grid(alpha=0.25)
    a.legend(frameon=False, loc=2, fontsize=9)

    # LOGITSCALE
    a = axes[0, 3]
    a.fill_between(x_logit, -log_v, log_v, color=AGENT_COLORS["Joint"], alpha=0.15, label=r"$\pm\ln t$")
    a.plot(x_logit, log_v, ":", c=AGENT_COLORS["Joint"], lw=1.1)
    a.plot(x_logit, -log_v, ":", c=AGENT_COLORS["Joint"], lw=1.1)
    a.fill_between(x_logit, -lin_v, lin_v, color=AGENT_COLORS["Naive"], alpha=0.15, label=r"$\pm t$")
    a.plot(x_logit, lin_v, "--", c=AGENT_COLORS["Naive"], lw=1.1)
    a.plot(x_logit, -lin_v, "--", c=AGENT_COLORS["Naive"], lw=1.1)
    a.set_title("Contributions scale differently"); a.set_xlabel("t"); a.grid(alpha=0.25)
    a.legend(frameon=False, loc=2, fontsize=9)

    # Turn off extra top-row axes if n_cols > 4
    for j in range(4, axes.shape[1]):
        axes[0, j].axis("off")

    # ── Row 1: Avg belief variance + per-distance panels ──
    xax = vb[0]
    avg_bel, avg_joint, avg_naive = vb[2], vb[3], vb[4]
    dist_bel, dist_joint, dist_naive = vb[5], vb[6], vb[7]
    eps = 1e-12

    # Panel 0: average across all distances
    a = axes[1, 0]
    a.plot(xax, np.clip(avg_bel[0], eps, np.inf), c=AGENT_COLORS["Trained"], lw=2.5, label="Trained RNN", zorder=NET_Z0)
    a.plot(xax, np.clip(avg_bel[1], eps, np.inf), c=AGENT_COLORS["Echo"], lw=2.5, label="Echo State", zorder=NET_Z1)
    a.plot(xax, np.clip(avg_joint[0], eps, np.inf), c=AGENT_COLORS["Joint"], ls="--", lw=1.7, label="joint", zorder=BAYES_Z0)
    a.plot(xax, np.clip(avg_naive[0], eps, np.inf), c=AGENT_COLORS["Naive"], ls="--", lw=1.7, label="naive", zorder=BAYES_Z1)
    a.set_xscale("log"); a.set_yscale("log"); a.grid(True, which="both", alpha=0.25)
    a.set_title(r"$B_r$ variance (avg over r1,r2,state)")
    a.set_xlabel("t"); a.set_ylabel("variance across batches")
    a.legend(frameon=False, fontsize=9)

    # Panels 1..D: per circular distance
    for d in range(D):
        ad = axes[1, d + 1]
        ad.plot(xax, np.clip(dist_bel[0, :, d], eps, np.inf), c=AGENT_COLORS["Trained"], lw=2.5, label="Trained RNN", zorder=NET_Z0)
        ad.plot(xax, np.clip(dist_bel[1, :, d], eps, np.inf), c=AGENT_COLORS["Echo"], lw=2.5, label="Echo State", zorder=NET_Z1)
        ad.plot(xax, np.clip(dist_joint[0, :, d], eps, np.inf), c=AGENT_COLORS["Joint"], ls="--", lw=1.6, label="joint", zorder=BAYES_Z0)
        ad.plot(xax, np.clip(dist_naive[0, :, d], eps, np.inf), c=AGENT_COLORS["Naive"], ls="--", lw=1.6, label="naive", zorder=BAYES_Z1)
        ad.set_xscale("log"); ad.set_yscale("log"); ad.grid(True, which="both", alpha=0.25)
        ad.set_title(f"Distance = {d}")
        ad.set_xlabel("t")
        if d > 0:
            ad.sharey(axes[1, 1])
            plt.setp(ad.get_yticklabels(), visible=False)
    return fig



# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # cuda = 0
    # realization_num = 10
    # step_num = 30
    # hid_dim = 1000
    # state_num = 500
    # obs_num = 5
    # batch_num = 8000
    # episodes = 1

    # echo = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes,
    #     'checkpoint_every': 5, 'realization_num': realization_num, 'hid_dim': hid_dim,
    #     'obs_num': obs_num, 'show_plots': False, 'batch_num': batch_num,
    #     'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
    #     'reservoir': True, 'classifier_LR': .001, 'ctx_num': 2,
    #     'training': False, 'load_env': "/sanity/reservoir_ctx_2_e5"})

    # trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes,
    #     'checkpoint_every': 5, 'realization_num': realization_num, 'hid_dim': hid_dim,
    #     'obs_num': obs_num, 'show_plots': False, 'batch_num': batch_num,
    #     'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
    #     'reservoir': False, 'classifier_LR': .001, 'ctx_num': 2,
    #     'training': False, 'load_env': "/sanity/fully_trained_ctx_2_e5"})

    # PCA: build per-model data
    def _pca(model, attr, label):
        d, ev = project_pca(getattr(model, attr), mode=CFG["PLOT_MODE"])
        return (d, ev, npy(model.ctx_vals), label)

    # # Figure PCA
    # render_paired_pca_grids(
    #     [_pca(trained, "model_belief_flat", "TRAINED MODEL"),
    #      _pca(trained, "joint_belief", "JOINT")],
    #     fig_title="Trained + Joint")
    # render_paired_pca_grids(
    #     [_pca(echo, "model_belief_flat", "ECHO MODEL"),
    #      _pca(echo, "naive_belief", "NAIVE")],
    #     fig_title="Echo + Naive")

    # Combined boundary shape figure
    plot_boundary_shape_combined(trained)

    # Post-training diagnostics
    # plot_post_training_diagnostics(trained, echo)

    plt.show()
