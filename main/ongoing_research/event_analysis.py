import numpy as np, os, sys, inspect, warnings; import matplotlib.pyplot as plt
[sys.path.insert(0, os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + f'/main{s}') for s in ('', '/bayes', '/model')]
from main.CognitiveGridworld import CognitiveGridworld

F64, I64 = np.float64, np.int64
EF_PARAMS = {"K_PCA": None, "P_BANDS": 100, "EVENT_WIN": 15, "EVENT_STD_MULT": 1, "K_SHOW_PCS": 100, "N_SHOW_NEUR": 100, "FIG_DPI": 120, "MIN_YABS": 0.1, "E_START": None, "E_END": None, "T_START": 0, "T_END": None}

def _z(x, axes=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning); m, s = (np.mean, np.std) if not np.isnan(x).any() else (np.nanmean, np.nanstd)
        return (x - m(x, axis=axes, keepdims=True)) / (s(x, axis=axes, keepdims=True) + 1e-12)

def approx_lik(b, eps=1e-30):
    b = np.maximum(b.astype(F64), eps); 
    px = np.zeros_like(b)
    px[:, 0] = b[:, 0]
    r = b[:, 1:] / b[:, :-1]
    px[:, 1:] = r / r.sum(-1, keepdims=True)
    return px

def step_dkl(p, q, eps=1e-30):
    p, q = np.maximum(p.astype(F64), eps), np.maximum(q.astype(F64), eps)
    p /= p.sum(-1, keepdims=True); q /= q.sum(-1, keepdims=True); return 0.5 * np.sum(p*np.log(p/q) + q*np.log(q/p), -1)

def step_angle(x, eps=1e-12):
    x = x.astype(F64); v1, v2 = x[:, :-1], x[:, 1:]; ang = np.full((x.shape[0], x.shape[1]), np.nan, F64)
    dot = np.einsum('bti,bti->bt', v1, v2); n1, n2 = np.linalg.norm(v1, axis=-1), np.linalg.norm(v2, axis=-1)
    ang[:, 1:] = np.degrees(np.arccos(np.clip(dot / np.maximum(n1*n2, eps), -1., 1.))); return ang

def _min_ylim(ax, m=0.1, p=0.08):
    y0, y1 = ax.get_ylim(); lo, hi = min(min(y0, y1), -m), max(max(y0, y1), m)
    d = p * max(hi - lo, 2 * m); ax.set_ylim(lo - d, hi + d)

def logit_ent(m, eps=1e-9):
    b = m.model_goal_belief.astype(F64); B, T, G = b.shape; gi = np.broadcast_to(m.goal_value.reshape(-1), B).astype(I64)
    p = np.clip(b[np.arange(B)[:, None], np.arange(T)[None, :], np.clip(gi, 0, G-1)[:, None]], eps, 1.-eps); bb = np.clip(b, eps, 1.)
    return np.log(p/(1.-p)).astype(F64), (-(bb*np.log(bb)).sum(-1)/np.log(G+eps)).astype(F64)

def get_sii(m, B, T):
    return step_dkl(m.joint_belief, m.naive_belief).mean(-1)

def pca_upd(m, k_pca=None):
    upd = m.model_update_flat.astype(F64); B, T, N = upd.shape; c = getattr(m, '_svd_cache', {})
    if c.get('shape') == (B, T, N): x, s, vt = c['x'], c['s'], c['vt']
    else:
        x = np.nan_to_num(upd.reshape(B*T, N).copy(), 0., 0., 0.); x -= x.mean(0, keepdims=True); _, s, vt = np.linalg.svd(x, False)
        m._svd_cache = {'shape': (B,T,N), 'x': x, 's': s, 'vt': vt}
    k = vt.shape[0] if k_pca is None else min(max(1, k_pca), vt.shape[0]); ev = s*s
    return upd, (x @ vt[:k].T).reshape(B, T, k).astype(F64), (ev[:k]/(ev.sum()+1e-12)).astype(F64)

def met_scores(z):
    e = z*z; s = np.maximum(e.sum(-1, keepdims=True), 1e-12); K, p = z.shape[2], e/s
    ct = 0. if K==1 else 1.-2.*(p*np.linspace(0., 1., K, dtype=F64).reshape(1,1,K)).sum(-1)
    se = -(p*np.log(np.maximum(p, 1e-12))).sum(-1)/np.log(K+1e-12); pr = (e.sum(-1)**2) / np.maximum((e*e).sum(-1), 1e-12)
    return np.sqrt(s[...,0]), ct, se, pr, p

def align_mean(x, ctr, tau):
    if not len(ctr): return np.full((tau.size,) + x.shape[2:], np.nan, F64), np.full((tau.size,) + x.shape[2:], np.nan, F64)
    v = x[np.broadcast_to(ctr[:, :1], (ctr.shape[0], tau.size)), ctr[:, 1:] + tau].astype(F64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning);
        return np.nanmean(v, 0).astype(F64), (np.nanstd(v, 0)/np.sqrt(np.maximum(1, v.shape[0]))).astype(F64)

def m_bands(p, evr, P):
    P = min(max(1, P), evr.size); bid = np.searchsorted(np.linspace(0., 1., P+1)[1:-1], np.cumsum(evr), "right")
    return np.tensordot(p, np.eye(P, dtype=F64)[bid], (2, 0)), bid, np.bincount(bid, minlength=P)

def ev_ctrs(log, mul):
    d = log[:, 1:] - log[:, :-1]; m = log[:, :-1] < 0; x = d[m]
    if not x.size: return np.zeros((0,2), I64), np.zeros((0,2), I64)
    th = x.mean() + mul * x.std(); return np.argwhere(m & (d >= th)), np.argwhere(m & (d < th))

def ev_diff(x, e, c, tau):
    me, se = align_mean(x, e, tau); mc, sc = align_mean(x, c, tau); return me-mc, np.sqrt(se**2 + sc**2)

def prep_model(m, prm, rng):
    lg, ent = logit_ent(m)
    B, T = lg.shape
    sii = get_sii(m, B, T)
    jp, jb = getattr(m, "joint_px", None), getattr(m, "joint_belief", None)
    jp = jp.astype(F64)
    jp /= np.maximum(jp.sum((-1,-2), keepdims=True), 1e-30)
    rev = step_dkl(np.stack([jp.sum(-1), jp.sum(-2)], 2), approx_lik(jb)).mean(-1)

    ang = step_angle(m.model_update_flat); upd, z, evr = pca_upd(m, prm["K_PCA"])
    pn, ct, se, pr, pp = met_scores(z); bnd, bid, bcnt = m_bands(pp, evr, prm["P_BANDS"])
    e_all, c_all = ev_ctrs(lg, prm["EVENT_STD_MULT"]); nm = min(len(e_all), len(c_all))
    e, c = (e_all[rng.choice(len(e_all), nm, False)], c_all[rng.choice(len(c_all), nm, False)]) if nm > 0 else (e_all, c_all)
    return {"mz": (_z(pn), _z(ct), _z(se), _z(ang), _z(pr)), "bz": _z(bnd, (0,1)), "sz": _z(z[:, :, :prm["K_SHOW_PCS"]], (0,1)),
            "bh": np.stack([_z(lg), _z(ent), _z(sii), _z(rev)], -1), "nz": _z(upd[:, :, rng.choice(upd.shape[2], min(prm["N_SHOW_NEUR"], upd.shape[2]), False)], (0,1)), 
            "e": e, "c": c, "e_all": e_all, "c_all": c_all}

def plt_row(ae, af, d, e, c, tau, lbl, sub, prm, cmap="viridis", nms=None, cls=None, pop=False):
    ts, te = prm.get("T_START", 0), prm.get("T_END", None); y, se = ev_diff(d, e, c, tau); dv, tt = d[:, ts:te], np.arange(d.shape[1])[ts:te]
    with warnings.catch_warnings(): warnings.simplefilter("ignore", RuntimeWarning); mf, sf = np.nanmean(dv, 0), np.nanstd(dv, 0)/np.sqrt(dv.shape[0])
    cls = cls or ([plt.get_cmap(cmap)(k/max(dv.shape[2]-1, 1)) for k in range(dv.shape[2])] if not (nms and dv.shape[2]<=10) else [plt.get_cmap("tab10")(k) for k in range(dv.shape[2])])
    ae.set_prop_cycle(color=cls); af.set_prop_cycle(color=cls); lns = ae.plot(tau, y, lw=2 if nms else 1, alpha=0.9); af.plot(tt, mf, lw=2 if nms else 1, alpha=0.9)
    if nms: [l.set_label(nms[k]) for k, l in enumerate(lns) if k < len(nms)]; (ae.legend(frameon=False, fontsize=9, loc='upper left') if lbl == "trained" else None)
    for k in range(dv.shape[2]): ae.fill_between(tau, y[:,k]-se[:,k], y[:,k]+se[:,k], color=cls[k], alpha=0.12, lw=0); af.fill_between(tt, mf[:,k]-sf[:,k], mf[:,k]+sf[:,k], color=cls[k], alpha=0.12, lw=0)
    if pop:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning); my, sy, mf_, sf_ = np.nanmean(y, 1), np.nanstd(y, 1), np.nanmean(mf, 1), np.nanstd(mf, 1)
            ae.fill_between(tau, my-sy, my+sy, color='r', alpha=0.35, zorder=10, lw=0); af.fill_between(tt, mf_-sf_, mf_+sf_, color='r', alpha=0.35, zorder=10, lw=0)
    for a in (ae, af): a.axhline(0, c="k", lw=1, alpha=0.5); a.grid(alpha=0.25); _min_ylim(a, prm["MIN_YABS"])
    ae.axvline(0, c="k", ls="--", lw=1, alpha=0.5); ae.set(title=f"{lbl} | {sub} | ben", xlabel="tau", ylabel="z"); af.set(title=f"{lbl} | {sub} | full", xlabel="t")

def plot_event_dynamics(d_tr, d_ec, prm):
    fig, ax = plt.subplots(6, 4, figsize=(20, 19), dpi=prm["FIG_DPI"])
    es, ee, ts, te = prm.get("E_START") or -prm["EVENT_WIN"], prm.get("E_END") or prm["EVENT_WIN"], prm.get("T_START", 0), prm.get("T_END", None)
    tau = np.arange(es, ee + 1)
    mc, mn = ("tab:blue", "tab:green", "tab:red", "tab:olive", "tab:cyan"), ("pc_norm", "centralization", "spectral_entropy", "angular_change", "participation_ratio")
    bc, bn = ("tab:orange", "tab:purple", "tab:brown", "tab:pink"), ("logit", "entropy", "SII", "re-evaluation")
    for mi, (d, lbl) in enumerate(((d_tr, "trained"), (d_ec, "echo"))):
        tev = te if te is not None else d["bh"].shape[1]; er, cr = d["e"], d["c"]
        e, c = er[(er[:,1]+es >= ts) & (er[:,1]+ee < tev)], cr[(cr[:,1]+es >= ts) & (cr[:,1]+ee < tev)]
        ce, cf = 2*mi, 2*mi+1; a, b = ax[0, ce], ax[0, cf]
        for k in range(len(mn)):
            y, se = ev_diff(d["mz"][k], e, c, tau)
            with warnings.catch_warnings(): warnings.simplefilter("ignore", RuntimeWarning); mf = np.nanmean(d["mz"][k][:, ts:te], 0)
            a.plot(tau, y, lw=2, c=mc[k], label=mn[k], alpha=0.7); b.plot(np.arange(d["mz"][k].shape[1])[ts:te], mf, lw=2, c=mc[k], alpha=0.7)
        for x in (a, b): x.axhline(0, c="k", lw=1, alpha=.5); x.grid(alpha=.25); _min_ylim(x, prm["MIN_YABS"])
        a.axvline(0, c="k", ls="--", lw=1, alpha=.5); a.set(title=f"{lbl} | metrics | ben", xlabel="tau", ylabel="z"); b.set(title=f"{lbl} | metrics | full", xlabel="t")
        if mi == 0: a.legend(frameon=False, fontsize=9)
        plt_row(ax[1, ce], ax[1, cf], d["bz"], e, c, tau, lbl, "PC-bands", prm)
        plt_row(ax[2, ce], ax[2, cf], d["bh"], e, c, tau, lbl, "behavior", prm, nms=bn, cls=bc)
        plt_row(ax[3, ce], ax[3, cf], d["sz"], e, c, tau, lbl, "explicit PCs", prm, cmap="plasma")
        plt_row(ax[4, ce], ax[4, cf], d["nz"], e, c, tau, lbl, "random neurons", prm, cmap="tab20", pop=True)
        
        ah, bh, ea = ax[5, ce], ax[5, cf], d["e_all"]
        bh.hist(e[:,1], bins=np.arange(ts, tev+2)-0.5, color='tab:blue', alpha=0.7)
        rt = [r for b_idx, t_anc in e for r in ea[ea[:,0]==b_idx][:,1]-t_anc if es <= r <= ee]
        ah.hist(rt, bins=np.arange(es, ee+2)-0.5, color='tab:blue', alpha=0.7); ah.axvline(0, c="k", ls="--", lw=1, alpha=0.5)
        ah.set(title=f"{lbl} | event auto-correlation | ben", xlabel="tau", ylabel="Count"); bh.set(title=f"{lbl} | valid anchors dist | full", xlabel="t", ylabel="Count")
        for x in (ah, bh): x.grid(alpha=0.25)
    fig.tight_layout(); return fig

def run_event_dynamics(trained, echo, params=None):
    prm = {**EF_PARAMS, **(params or {})}; rng = np.random.default_rng(); d_tr, d_ec = prep_model(trained, prm, rng), prep_model(echo, prm, rng)
    return {"trained": d_tr, "echo": d_ec, "fig": plot_event_dynamics(d_tr, d_ec, prm), "params": prm}

if __name__ == "__main__":
    # cm = dict(mode="SANITY", cuda=0, episodes=1, checkpoint_every=5, realization_num=10, hid_dim=1000, 
    #           obs_num=5, show_plots=False, batch_num=15000, step_num=30, state_num=500, 
    #           learn_embeddings=False, classifier_LR=.001, ctx_num=2, training=False)
    # echo = CognitiveGridworld(**{**cm, 'reservoir': True, 'load_env': "/sanity/reservoir_ctx_2_e5"})
    # trained = CognitiveGridworld(**{**cm, 'reservoir': False, 'load_env': "/sanity/fully_trained_ctx_2_e5"})
    run_event_dynamics(trained, echo, params={"E_START": -5, "E_END": 5, "T_START": 5, "T_END": 30, "P_BANDS": 100, "K_SHOW_PCS": 100, "N_SHOW_NEUR": 100, "EVENT_STD_MULT": 1})
    plt.show()