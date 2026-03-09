import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

CFG=dict(
    DPI=140,FACECOLOR="black",GRID_ALPHA=0.07,QLO=0,QHI=99,PAD=0.25,NX=120,NY=120,HDR_MASS=0.9,SMOOTH=1,N_SPLITS=20,
    PLOT_MODE="all",SAVE_PDF=False,
    EPS=1e-12,NON_GOAL_MODE="mean",T_PLOT=None,STEP_STRIDE=1,POINTS_PER_STEP=6,CONTOUR_BINS=70,CONTOUR_QHI=100.0,CONTOUR_MASS=0.95,
    FRONT_ANGLE_BINS=100,FRONT_SMOOTH_SIGMA=3,FRONT_SCALE_MODE="max",FRONT_CENTROID_MODE="t0",
    VIEW_ELEV=25,VIEW_AZIM=-55,PLANE_ALPHA=0.12,INTERSECT_LW=1.8
)

plt.rcParams.update({
    "figure.dpi":CFG["DPI"],
    "font.size":11,
    "axes.titlesize":12,
    "axes.labelsize":11,
    "legend.fontsize":10,
    "xtick.labelsize":10,
    "ytick.labelsize":10,
})

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

def edges_from_quantiles(x,qlo,qhi,n,pad):
    xf=np.asarray(x)[np.isfinite(x)]
    if xf.size==0: return np.linspace(-1,1,int(n)+1,dtype=np.float32)
    lo,hi=np.percentile(xf,(qlo,qhi))
    hi=max(hi,lo+1e-6)
    d=(hi-lo)*float(pad)
    return np.linspace(lo-d,hi+d,int(n)+1,dtype=np.float32)

def box_smooth2d(Z,r):
    if r<=0: return Z
    k=2*r+1
    P=np.pad(Z,r,mode="edge")
    I=np.zeros((P.shape[0]+1,P.shape[1]+1),dtype=np.float64)
    I[1:,1:]=P.cumsum(0).cumsum(1)
    return ((I[k:,k:]-I[:-k,k:]-I[k:,:-k]+I[:-k,:-k])/(k*k)).astype(np.float32)

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

def nong_reduce(A_BTS,gBT,mode):
    B,T,S=A_BTS.shape
    bix=np.arange(B,dtype=np.int64)[:,None]
    tix=np.arange(T,dtype=np.int64)[None,:]
    if mode=="single":
        sng=(gBT+1)%S
        return A_BTS[bix,tix,sng]
    sm=A_BTS.sum(2)
    g=A_BTS[bix,tix,gBT]
    return (sm-g)/float(max(S-1,1))

def get_goal_vs_nongoal_logits(model,belief_attr):
    P=renorm(npy(getattr(model,belief_attr))).astype(np.float32)
    rtrue=npy(model.ctx_vals).astype(np.int64)
    B,T=P.shape[:2]
    if rtrue.ndim==2: idx=np.broadcast_to(rtrue[:,None,:,None],(B,T,rtrue.shape[1],1))
    elif rtrue.ndim==3: idx=rtrue[...,None]
    else: raise ValueError(f"Unexpected ctx_vals shape: {rtrue.shape}")
    g=coerce_goal_ind(model.goal_ind,B,T)
    p_true=np.take_along_axis(P,idx,axis=3)[...,0]
    bix=np.arange(B,dtype=np.int64)[:,None]
    tix=np.arange(T,dtype=np.int64)[None,:]
    p_goal=p_true[bix,tix,g]
    p_ng=(p_true.sum(2)-p_goal)/np.float32(max(P.shape[2]-1,1))
    return logit(p_goal,clip=False),logit(p_ng,clip=False)

def plot_goal_vs_nongoal_hdr(trained,echo):
    Lg_tr,Ln_tr=get_goal_vs_nongoal_logits(trained,"model_belief_flat")
    Lg_ec,Ln_ec=get_goal_vs_nongoal_logits(echo,"model_belief_flat")
    Te=Lg_tr.shape[1]-1
    xb=edges_from_quantiles(np.concatenate((Ln_tr[:,:Te].ravel(),Ln_ec[:,:Te].ravel())),CFG["QLO"],CFG["QHI"],CFG["NX"],CFG["PAD"])
    yb=edges_from_quantiles(np.concatenate((Lg_tr[:,:Te].ravel(),Lg_ec[:,:Te].ravel())),CFG["QLO"],CFG["QHI"],CFG["NY"],CFG["PAD"])
    cmap=plt.get_cmap("coolwarm")
    cols=cmap(np.linspace(0.0,1.0,CFG["N_SPLITS"]))
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.0),dpi=CFG["DPI"],constrained_layout=True,sharex=True,sharey=True)
    fig.patch.set_facecolor(CFG["FACECOLOR"])
    for ax,title in zip(axes,("trained model | 95% HDR","echo model | 95% HDR")):
        style(ax,True)
        ax.set_title(title,color="0.95",fontsize=11)
        ax.set_xlabel("ℓ_non-goal",color="0.95")
    axes[0].set_ylabel("ℓ_goal",color="0.95")
    edges=np.linspace(0,Te,CFG["N_SPLITS"]+1).astype(np.int64)
    for si in range(CFG["N_SPLITS"]):
        lo,hi=int(edges[si]),min(max(int(edges[si])+1,int(edges[si+1])),Te)
        for ax,Ln,Lg in zip(axes,(Ln_tr,Ln_ec),(Lg_tr,Lg_ec)):
            x,y=Ln[:,lo:hi].ravel(),Lg[:,lo:hi].ravel()
            m=np.isfinite(x)&np.isfinite(y)
            if not m.any(): continue
            H,_,_=np.histogram2d(y[m],x[m],bins=(yb,xb))
            H=box_smooth2d(H.astype(np.float32),int(CFG["SMOOTH"]))
            lvl=mass_level(H,float(CFG["HDR_MASS"]),normalize=False)
            if np.isfinite(lvl) and H.max()>0:
                ax.contour(H,levels=(lvl,),colors=(cols[si],),linewidths=2.0,extent=(xb[0],xb[-1],yb[0],yb[-1]))
    if CFG["SAVE_PDF"]: fig.savefig("figure_hdr.pdf",bbox_inches="tight")
    return fig

def render_projection_grid(data,ev,ctx,title_label):
    ctx=npy(ctx)
    while ctx.ndim>3 and ctx.shape[-1]==1: ctx=ctx[...,0]
    if ctx.ndim==3: ctx=ctx[:,0]
    u_r=np.unique(ctx[:,:2])
    n_R=len(u_r)
    den=max(n_R-1,1)
    cmap=plt.get_cmap("plasma")
    pad_x=(data[...,0].max()-data[...,0].min())*0.35
    pad_y=(data[...,1].max()-data[...,1].min())*0.35
    xlim=(data[...,0].min()-pad_x,data[...,0].max()+pad_x)
    ylim=(data[...,1].min()-pad_y,data[...,1].max()+pad_y)
    fig=plt.figure(figsize=(22,11),dpi=CFG["DPI"])
    fig.patch.set_facecolor(CFG["FACECOLOR"])
    gs=fig.add_gridspec(2,7,hspace=0.3,wspace=0.15)

    def plot_traj(ax,mask,color,ls,label):
        if not mask.any(): return
        traj=data[mask].mean(0)
        ax.plot(traj[:,0],traj[:,1],color=color,ls=ls,alpha=0.8,lw=2,label=label)
        ax.scatter(traj[0,0],traj[0,1],color=color,marker="o",s=20)
        ax.scatter(traj[-1,0],traj[-1,1],color=color,marker="X",s=60,edgecolor="black",lw=0.5)

    for row_idx in range(2):
        is_r1_primary=row_idx==0
        shift_type="R1=(R2+c)%n" if is_r1_primary else "R2=(R1+c)%n"
        for c in range(6):
            ax=fig.add_subplot(gs[row_idx,c]); style(ax,True)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_title(shift_type.replace("c",str(c)),color="0.95",fontsize=9)
            if c==0: ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)\n{title_label}",color="cyan",fontsize=10,fontweight="bold")
            if row_idx==1: ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)",color="0.95")
            else: ax.tick_params(labelbottom=False)
            if c>0: ax.tick_params(labelleft=False)
            for i,r_val in enumerate(u_r):
                r1,r2=((r_val+c)%n_R,r_val) if is_r1_primary else (r_val,(r_val+c)%n_R)
                plot_traj(ax,(ctx[:,0]==r1)&(ctx[:,1]==r2),cmap(i/den),"-",f"({r1},{r2})")
            ax.legend(loc="best",fontsize=6,ncol=2,framealpha=0.4,labelcolor="0.95")
        ax=fig.add_subplot(gs[row_idx,6]); style(ax,True)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_title(f"Marginal: {'R2' if is_r1_primary else 'R1'}",color="yellow",fontsize=10)
        ax.tick_params(labelleft=False)
        if row_idx==0: ax.tick_params(labelbottom=False)
        for i,r in enumerate(u_r):
            plot_traj(ax,ctx[:,1]==r if is_r1_primary else ctx[:,0]==r,cmap(i/den),"--" if is_r1_primary else "-",f"R={r}")
        ax.legend(loc="best",fontsize=7,ncol=2,framealpha=0.5,labelcolor="0.95")
    mode_str=f"TimeStep {CFG['PLOT_MODE']}" if isinstance(CFG["PLOT_MODE"],int) else "All TimeSteps"
    fig.suptitle(f"PCA Projection ({mode_str}): {title_label}",color="white",fontsize=20,y=0.98)
    if CFG["SAVE_PDF"]: fig.savefig(f"figure_{title_label.lower().replace(' ','_')}.pdf",bbox_inches="tight")
    return fig

def plot_step_contours(ax,X,Y,steps,ex,ey):
    cx,cy=0.5*(ex[:-1]+ex[1:]),0.5*(ey[:-1]+ey[1:])
    Xc,Yc=np.meshgrid(cx,cy,indexing="xy")
    mX,mY=np.full(steps.size,np.nan),np.full(steps.size,np.nan)
    cmap=plt.cm.coolwarm
    for i,st in enumerate(steps):
        st=int(st)
        if st<1 or st>steps[-1] or ((st-1)%CFG["STEP_STRIDE"])!=0: continue
        H,_,_=np.histogram2d(X[:,i],Y[:,i],bins=(ex,ey))
        lvl=mass_level(H,CFG["CONTOUR_MASS"],normalize=True)
        n=X.shape[0]
        if n>0:
            idx=np.linspace(0,n-1,min(n,CFG["POINTS_PER_STEP"]),dtype=np.int64)
            ax.scatter(X[idx,i],Y[idx,i],c=np.full(idx.size,float(st)),cmap=cmap,vmin=1,vmax=steps[-1],s=7,alpha=0.20,linewidths=0,zorder=2)
        if np.isfinite(lvl) and lvl>0:
            ax.contour(Xc,Yc,(H/np.maximum(H.sum(),1.0)).T,levels=(lvl,),colors=(cmap((st-1)/max(steps[-1]-1,1)),),linewidths=1.1,alpha=0.95,zorder=3)
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

def plot_boundary_shape_figures(trained):
    bix=npy(getattr(trained,"batch_range",None))
    Pj_full,Pn_full=npy(trained.joint_belief),npy(trained.naive_belief)
    bix=bix.astype(np.int64) if bix is not None else np.arange(Pj_full.shape[0],dtype=np.int64)
    Pj,Pn=renorm(Pj_full[bix]),renorm(Pn_full[bix])
    B,T,S,R=Pj.shape
    Tplot=T if CFG["T_PLOT"] is None else int(np.clip(int(CFG["T_PLOT"]),1,T))
    gBT=coerce_goal_ind(trained.goal_ind,B,T)
    ctx=npy(trained.ctx_vals)[bix]
    while ctx.ndim>3 and ctx.shape[-1]==1: ctx=ctx[...,0]
    if ctx.ndim==2: ctx=np.repeat(ctx[:,None,:],T,axis=1)
    ctx=ctx.astype(np.int64)
    bix2=np.arange(B,dtype=np.int64)[:,None]
    tix2=np.arange(T,dtype=np.int64)[None,:]
    tr_goal=ctx[bix2,tix2,gBT]
    tab1,tabA=neighbor_table(R,1),neighbor_table(R,-1)
    pj_st=np.take_along_axis(Pj,ctx[...,None],3)[...,0]
    pn_st=np.take_along_axis(Pn,ctx[...,None],3)[...,0]

    pj_nr,pn_nr=np.zeros((B,T,S,2)),np.zeros((B,T,S,2))
    pj_g_nr,pn_g_nr=np.zeros((B,T,2)),np.zeros((B,T,2))
    for k,tab in enumerate((tab1,tabA)):
        pj_nr[...,k]=np.take_along_axis(Pj,tab[ctx],3).mean(3)
        pn_nr[...,k]=np.take_along_axis(Pn,tab[ctx],3).mean(3)
        pj_g_nr[...,k]=np.take_along_axis(Pj[bix2,tix2,gBT],tab[tr_goal],2).mean(2)
        pn_g_nr[...,k]=np.take_along_axis(Pn[bix2,tix2,gBT],tab[tr_goal],2).mean(2)

    XJ,YJ,XN,YN=(np.zeros((B,T,3),dtype=np.float32) for _ in range(4))
    XJ[...,0],YJ[...,0]=logit(pj_st[bix2,tix2,gBT],clip=True),logit(nong_reduce(pj_st,gBT,CFG["NON_GOAL_MODE"]),clip=True)
    XN[...,0],YN[...,0]=logit(pn_st[bix2,tix2,gBT],clip=True),logit(nong_reduce(pn_st,gBT,CFG["NON_GOAL_MODE"]),clip=True)
    for k in range(2):
        c=k+1
        XJ[...,c]=XJ[...,0]-logit(pj_g_nr[...,k],clip=True)
        YJ[...,c]=YJ[...,0]-logit(nong_reduce(pj_nr[...,k],gBT,CFG["NON_GOAL_MODE"]),clip=True)
        XN[...,c]=XN[...,0]-logit(pn_g_nr[...,k],clip=True)
        YN[...,c]=YN[...,0]-logit(nong_reduce(pn_nr[...,k],gBT,CFG["NON_GOAL_MODE"]),clip=True)

    col_titles=("True","True − near (K=1)","True − near (all)")
    steps=np.arange(1,Tplot+1)
    cmap=plt.cm.coolwarm

    fig1,axs1=plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
    fig1.suptitle("Goal vs non-goal evidence contours")
    fig2,axs2=plt.subplots(2,3,figsize=(15,10),subplot_kw={"projection":"3d"},constrained_layout=True)
    fig2.suptitle(f"3D Boundary Shape Manifolds (Centroid: {CFG['FRONT_CENTROID_MODE']}, Scale: {CFG['FRONT_SCALE_MODE']})")
    fig3,axs3=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
    fig3.suptitle("2D Boundary Shape Waterfall")

    for c in range(3):
        xj,yj,xn,yn=XJ[:,:Tplot,c],YJ[:,:Tplot,c],XN[:,:Tplot,c],YN[:,:Tplot,c]
        x0,x1=robust_lo_hi(np.concatenate((xj.ravel(),xn.ravel())),CFG["CONTOUR_QHI"])
        y0,y1=robust_lo_hi(np.concatenate((yj.ravel(),yn.ravel())),CFG["CONTOUR_QHI"])
        ex,ey=np.linspace(x0,x1,CFG["CONTOUR_BINS"]+1),np.linspace(y0,y1,CFG["CONTOUR_BINS"]+1)

        for ax,x,y,t in zip(axs1[:,c],(xj,xn),(yj,yn),("Joint","Naive")):
            ax.set_title(f"{t}: {col_titles[c]}")
            plot_step_contours(ax,x,y,steps,ex,ey)
            ax.axvline(0,lw=1,alpha=0.5)
            ax.axhline(0,lw=1,alpha=0.5)
            style(ax,False)

        fJ,deg=calc_true_contour_front(xj,yj,ex,ey,Tplot)
        fN,_=calc_true_contour_front(xn,yn,ex,ey,Tplot)
        deg_plot=deg*180/np.pi
        xm,ym=np.meshgrid(deg_plot,steps)
        z_max=max(np.nanmax(fJ),np.nanmax(fN))*1.05
        yp,zp=np.meshgrid((1,Tplot),(0,z_max))

        for row_idx,fronts,title in zip((0,1),(fJ,fN),("Joint","Naive")):
            ax3d=axs2[row_idx,c]
            ax2d=axs3[row_idx,c]
            ax3d.set_title(f"{title}: {col_titles[c]}")
            ax2d.set_title(f"{title}: {col_titles[c]}")
            v=np.where(~np.isnan(fronts).all(1))[0]
            if v.size:
                sl=slice(v[0],v[-1]+1)
                ax3d.plot_surface(xm[sl],ym[sl],fronts[sl],facecolors=cmap(plt.Normalize(1,Tplot)(ym[sl])),shade=True,lw=0,alpha=0.9)
            for t in range(Tplot):
                if np.any(np.isfinite(fronts[t])): ax2d.plot(deg_plot,fronts[t],color=cmap(t/max(Tplot,1)),lw=1.2,alpha=0.8)
            for a,cl in zip((-90,-45,0,45,90),("red","green","gray","green","red")):
                ax3d.plot_surface(np.full_like(yp,a),yp,zp,color=cl,alpha=CFG["PLANE_ALPHA"],shade=False)
                idx=np.argmin(np.abs(deg_plot-a))
                ax3d.plot(np.full(Tplot,a),steps,fronts[:,idx],color=cl,lw=CFG["INTERSECT_LW"],zorder=10)
                ax2d.axvline(a,color=cl,linestyle=":",alpha=0.6)
            ax3d.set_xlim(-180,180); ax3d.set_ylim(1,Tplot); ax3d.set_zlim(0,z_max)
            ax3d.set_xticks((-180,-90,-45,0,45,90,180)); ax3d.view_init(CFG["VIEW_ELEV"],CFG["VIEW_AZIM"])
            ax2d.set_xlim(-180,180); ax2d.set_ylim(0,z_max)
            ax2d.set_xticks((-180,-90,-45,0,45,90,180))
            style(ax2d,False)
    return fig1,fig2,fig3

if __name__=="__main__":
    plot_goal_vs_nongoal_hdr(trained,echo)

    for model,attr,label in (
        (trained,"model_belief_flat","TRAINED MODEL BELIEF"),
        (echo,"model_belief_flat","ECHO MODEL BELIEF"),
        (trained,"joint_belief","JOINT BELIEF"),
        (echo,"naive_belief","NAIVE BELIEF"),
    ):
        data_2d,ev_2d=project_pca(getattr(model,attr),mode=CFG["PLOT_MODE"])
        render_projection_grid(data_2d,ev_2d,npy(model.ctx_vals),label)

    plot_boundary_shape_figures(trained)
    plt.show()

