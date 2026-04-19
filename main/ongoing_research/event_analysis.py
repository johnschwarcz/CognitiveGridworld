import numpy as np
import os
import sys
import inspect
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

[sys.path.insert(0, os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + f'/main{s}') for s in ('', '/bayes', '/model')]
from main.CognitiveGridworld import CognitiveGridworld

F64, I64 = np.float64, np.int64
EF_PARAMS = {
    "K_PCA": None, "P_BANDS": 100, "EVENT_WIN": 15, "EVENT_STD_MULT": 1, 
    "K_SHOW_PCS": 100, "N_SHOW_NEUR": 100, "FIG_DPI": 120, "MIN_YABS": 0.1, 
    "E_START": None, "E_END": None, "T_START": 0, "T_END": None, "CTX_MODE": "average",
    "HEATMAP_MODE": False
}

def _z(x, axes=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu, sd = np.nanmean(x, axis=axes, keepdims=True), np.nanstd(x, axis=axes, keepdims=True)
        return (x - mu) / (sd + 1e-9)

def approx_lik(b, eps=1e-99):
    b = np.maximum(b.astype(F64), eps)
    px = np.zeros_like(b)
    px[:, 0], r = b[:, 0], b[:, 1:] / b[:, :-1]
    px[:, 1:] = r / r.sum(-1, keepdims=True)
    return px

def step_dkl(p, q, eps=1e-99):
    p, q = np.maximum(p.astype(F64), eps), np.maximum(q.astype(F64), eps)
    p, q = p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)
    return 0.5 * np.sum(p * (np.log(p) - np.log(q)) + q * (np.log(q) - np.log(p)), -1)

def step_angle(x, eps=1e-99):
    x = x.astype(F64)
    v1, v2 = x[:, :-1], x[:, 1:]
    ang = np.full((x.shape[0], x.shape[1]), np.nan, F64)
    dot = np.einsum('bti,bti->bt', v1, v2)
    n1, n2 = np.linalg.norm(v1, axis=-1), np.linalg.norm(v2, axis=-1)
    ang[:, 1:] = np.degrees(np.arccos(np.clip(dot / np.maximum(n1 * n2, eps), -1., 1.)))
    return ang

def _min_ylim(ax, m=0.1, p=0.08):
    y0, y1 = ax.get_ylim()
    lo, hi = min(min(y0, y1), -m), max(max(y0, y1), m)
    d = p * max(hi - lo, 2 * m)
    ax.set_ylim(lo - d, hi + d)

def logit_ent(m, eps=1e-9):
    mb, cv = m.model_belief.astype(F64), m.ctx_vals.astype(I64)
    B, T, C, R = mb.shape
    b_idx, t_idx, c_idx = np.arange(B)[:, None, None], np.arange(T)[None, :, None], np.arange(C)[None, None, :]
    r_idx = np.clip(cv[:, None, :], 0, R-1)
    p = np.clip(mb[b_idx, t_idx, c_idx, r_idx], eps, 1. - eps)
    logits = (np.log(p) - np.log(1. - p)).astype(F64)
    ent = (-(mb * np.log(np.clip(mb, eps, 1.))).sum(axis=-1) / np.log(R + eps)).astype(F64)
    return logits, ent

def pca_upd(m, k_pca=None):
    upd = m.model_update_flat.astype(F64)
    B, T, N = upd.shape
    c = getattr(m, '_svd_cache', {})
    if c.get('shape') == (B, T, N):
        x, s, vt = c['x'], c['s'], c['vt']
    else:
        x = np.nan_to_num(upd.reshape(B * T, N).copy(), 0., 0., 0.)
        x -= x.mean(0, keepdims=True)
        _, s, vt = np.linalg.svd(x, False)
        m._svd_cache = {'shape': (B, T, N), 'x': x, 's': s, 'vt': vt}
    k = k_pca if k_pca is not None else vt.shape[0]
    ev = s * s
    return upd, (x @ vt[:k].T).reshape(B, T, k).astype(F64), (ev[:k] / (ev.sum() + 1e-99)).astype(F64)

def met_scores(z):
    e, K = z * z, z.shape[2]
    s = np.maximum(e.sum(-1, keepdims=True), 1e-99)
    p = e / s
    ct = 0. if K == 1 else 1. - 2. * (p * np.linspace(0., 1., K, dtype=F64).reshape(1, 1, K)).sum(-1)
    se = -(p * np.log(np.maximum(p, 1e-99))).sum(-1) / np.log(K + 1e-99)
    pr = (e.sum(-1)**2) / np.maximum((e * e).sum(-1), 1e-99)
    return np.sqrt(s[..., 0]), ct, se, pr, p

def align_mean(x, ctr, tau):
    v = x[np.broadcast_to(ctr[:, :1], (ctr.shape[0], tau.size)), ctr[:, 1:] + tau].astype(F64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(v, 0).astype(F64), (np.nanstd(v, 0) / np.sqrt(max(1, v.shape[0]))).astype(F64)

def m_bands(p, evr, P):
    bid = np.searchsorted(np.linspace(0., 1., P + 1)[1:-1], np.cumsum(evr), "right")
    return np.tensordot(p, np.eye(P, dtype=F64)[bid], (2, 0)), bid, np.bincount(bid, minlength=P)

def ev_ctrs(log, mul):
    d = log[:, 1:] - log[:, :-1]
    th = d.mean() + mul * d.std()
    return np.argwhere(d >= th), np.argwhere(d < th)

def ev_diff(x, e, c, tau):
    me, se = align_mean(x, e, tau)
    mc, sc = align_mean(x, c, tau)
    return me - mc, np.sqrt(se**2 + sc**2)

def get_sii_ctx(m, B, T):
    return step_dkl(m.joint_belief, m.naive_belief)

def get_xcorr(a, b, lag_min, lag_max):
    B, T = a.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        a_n, b_n = (a - np.nanmean(a, 1, keepdims=True)) / (np.nanstd(a, 1, keepdims=True) + 1e-9), (b - np.nanmean(b, 1, keepdims=True)) / (np.nanstd(b, 1, keepdims=True) + 1e-9)
    lags = np.arange(lag_min, lag_max + 1)
    xc = np.zeros((B, len(lags)))
    for i, lag in enumerate(lags):
        if lag < 0: xc[:, i] = np.nanmean(a_n[:, :lag] * b_n[:, -lag:], axis=1)
        elif lag > 0: xc[:, i] = np.nanmean(a_n[:, lag:] * b_n[:, :-lag], axis=1)
        else: xc[:, i] = np.nanmean(a_n * b_n, axis=1)
    return np.nanmean(xc, 0), np.nanstd(xc, 0) / np.sqrt(B)

def prep_model(m, prm, rng):
    logits_all, ent_all = logit_ent(m)
    B, T, C = logits_all.shape
    px_s = m.naive_px / m.naive_px.sum(-1, keepdims=True)
    rev_all, rev_n, rev_m = [step_dkl(px_s, approx_lik(x.astype(F64))) for x in (m.joint_belief, m.naive_belief, m.model_belief)]
    
    obs = m.obs_flat.astype(F64)
    running_sum = np.zeros_like(obs)
    running_sum[:, 1:] = np.cumsum(obs[:, :-1, :], axis=1)
    p_feat_is_1 = (running_sum + 1.0) / (np.arange(T).reshape(1, T, 1) + 2.0)
    p_curr = np.where(obs == 1, p_feat_is_1, 1.0 - p_feat_is_1)
    emp_prob, surprisal = p_curr.mean(-1), -np.log(p_curr + 1e-99).mean(-1)
    accuracy = m.model_acc.astype(F64)

    upd_flat = m.model_update_flat.astype(F64)
    ang, upd_norm = step_angle(upd_flat), np.linalg.norm(upd_flat, axis=-1)
    upd, z, evr = pca_upd(m, prm["K_PCA"])
    pn, ct, se, pr, pp = met_scores(z)
    dz = np.zeros_like(z); dz[:, 1:] = z[:, 1:] - z[:, :-1]
    ang_mom = np.sqrt(np.maximum(0, np.sum(z**2, -1) * np.sum(dz**2, -1) - np.sum(z * dz, -1)**2))

    bnd, bid, bcnt = m_bands(pp, evr, prm["P_BANDS"])
    idx = rng.choice(upd.shape[2], prm["N_SHOW_NEUR"], replace=False)
    upd_s, z_s = upd[:, :, idx], z[:, :, :prm["K_SHOW_PCS"]]

    mode = prm.get("CTX_MODE", "average")
    if mode in ["same", "opposite", "both"] and C >= 2:
        lg_event = np.transpose(logits_all, (0, 2, 1)).reshape(B * C, T)
        tile = lambda x: np.repeat(x, C, axis=0)
        upd_norm, ang, pn, ct, se, pr, ang_mom, emp_prob, surprisal, accuracy = map(tile, (upd_norm, ang, pn, ct, se, pr, ang_mom, emp_prob, surprisal, accuracy))
        upd_s, z_s, bnd = map(tile, (upd_s, z_s, bnd))
        bh_3d = [ent_all, get_sii_ctx(m, B, T), rev_all, rev_n, rev_m]
        bh_flat = [np.transpose(x[:, :, ::-1] if mode=="opposite" else x, (0, 2, 1)).reshape(B * C, T) for x in bh_3d]
        measures = [np.transpose(logits_all[:, :, ::-1] if mode=="opposite" else logits_all, (0, 2, 1)).reshape(B * C, T)] + bh_flat + [emp_prob, surprisal, accuracy]
        e_all, c_all = ev_ctrs(lg_event, prm["EVENT_STD_MULT"])
    else:
        measures = [x.mean(-1) if x.ndim == 3 else x for x in (logits_all, ent_all, get_sii_ctx(m, B, T), rev_all, rev_n, rev_m, emp_prob, surprisal, accuracy)]
        e_all, c_all = ev_ctrs(measures[0], prm["EVENT_STD_MULT"])

    nm = min(len(e_all), len(c_all))
    e, c = e_all[rng.choice(len(e_all), nm, replace=False)], c_all[rng.choice(len(c_all), nm, replace=False)]
    
    return {"raw_mz": (pn, ct, se, ang, pr, upd_norm, ang_mom), "raw_bh": measures, "bz": bnd, "sz": z_s, "nz": _z(upd_s, (0, 1)), "e": e, "c": c, "e_all": e_all, "c_all": c_all}

def plt_row(ae, af, d_stack, e, c, tau, lbl, sub, prm, cmap="viridis", nms=None, cls=None, pop=False, center_pop=False, heatmap=False):
    ts, te = prm.get("T_START", 0), prm.get("T_END", None)
    y, se = ev_diff(d_stack, e, c, tau)
    dv, tt = d_stack[:, ts:te], np.arange(d_stack.shape[1])[ts:te]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mf, sf = np.nanmean(dv, 0), np.nanstd(dv, 0) / np.sqrt(max(1, dv.shape[0]))
        
    if pop and center_pop:
        y, mf = y - np.nanmean(y, 0, keepdims=True), mf - np.nanmean(mf, 0, keepdims=True)
        
    if heatmap:
        y_p, mf_p = [x[:, np.argsort(np.argmax(np.nan_to_num(x, nan=-np.inf), axis=0))].T for x in (y, mf)]
        ae.imshow(y_p / (np.nanmax(np.abs(y_p), 1, keepdims=True) + 1e-99), aspect='auto', origin='lower', extent=[tau[0], tau[-1], 0, y.shape[1]], cmap="coolwarm", vmin=-1, vmax=1)
        af.imshow(mf_p / (np.nanmax(np.abs(mf_p), 1, keepdims=True) + 1e-99), aspect='auto', origin='lower', extent=[tt[0], tt[-1], 0, mf.shape[1]], cmap="coolwarm", vmin=-1, vmax=1)
    else:
        cls = cls or [plt.get_cmap(cmap)(k / max(d_stack.shape[2] - 1, 1)) for k in range(d_stack.shape[2])]
        for k in range(d_stack.shape[2]):
            ae.plot(tau, y[:, k], lw=2, color=cls[k], alpha=0.9, label=nms[k] if nms else None); af.plot(tt, mf[:, k], lw=2, color=cls[k], alpha=0.9)
            ae.fill_between(tau, y[:, k] - se[:, k], y[:, k] + se[:, k], color=cls[k], alpha=0.1, lw=0); af.fill_between(tt, mf[:, k] - sf[:, k], mf[:, k] + sf[:, k], color=cls[k], alpha=0.1, lw=0)
        
        if pop and not center_pop:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                my, sy, mf_, sf_ = np.nanmean(y, 1), np.nanstd(y, 1), np.nanmean(mf, 1), np.nanstd(mf, 1)
            ae.fill_between(tau, my - sy, my + sy, color='r', alpha=0.35, zorder=10, lw=0)
            af.fill_between(tt, mf_ - sf_, mf_ + sf_, color='r', alpha=0.35, zorder=10, lw=0)
            
        if nms and lbl == "trained": ae.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
        
    for a in (ae, af): a.axhline(0, c="k", lw=1, alpha=0.5); a.grid(alpha=0.15); a.axvline(0, c="k", ls="--", lw=1, alpha=0.5)
    ae.set(title=f"{lbl} | {sub} | ben", xlabel="tau", ylabel="z"); af.set(title=f"{lbl} | {sub} | full", xlabel="t")

def plot_event_dynamics(d_tr, d_ec, prm):
    plt.style.use('default')
    fig_m, ax_m = plt.subplots(6, 4, figsize=(20, 19), dpi=prm["FIG_DPI"], facecolor='white')
    # fig_x: 2 rows (Trained, Echo) x 4 columns (Metrics, Re-evals, PCs, Bands)
    fig_x, ax_x = plt.subplots(2, 4, figsize=(24, 10), dpi=prm["FIG_DPI"], facecolor='white')
    
    es, ee, ts, te = prm.get("E_START") or -prm["EVENT_WIN"], prm.get("E_END") or prm["EVENT_WIN"], prm.get("T_START", 0), prm.get("T_END", None)
    tau = np.arange(es, ee + 1)
    mn, mc = ("pc_norm", "centralization", "spectral_entropy", "angular_change", "participation_ratio", "upd_magnitude", "angular_momentum"), ("tab:blue", "tab:green", "tab:red", "tab:olive", "tab:cyan", "tab:gray", "black")
    bn, bc = ("logit", "entropy", "SII", "joint_rev", "naive_rev", "model_rev", "bern_prob", "surprisal", "accuracy"), ("tab:orange", "tab:purple", "tab:brown", "tab:pink", "crimson", "teal", "navy", "goldenrod", "black")
    
    for mi, (d, lbl) in enumerate(((d_tr, "trained"), (d_ec, "echo"))):
        tev, er, cr = te if te is not None else d["raw_bh"][0].shape[1], d["e"], d["c"]
        e, c = er[(er[:, 1] + es >= ts) & (er[:, 1] + ee < tev)], cr[(cr[:, 1] + es >= ts) & (cr[:, 1] + ee < tev)]
        ce, cf = 2 * mi, 2 * mi + 1
        
        # Main Dashboard Layout
        plt_row(ax_m[0, ce], ax_m[0, cf], np.stack([_z(x) for x in d["raw_mz"]], -1), e, c, tau, lbl, "neural", prm, nms=mn, cls=mc)
        plt_row(ax_m[1, ce], ax_m[1, cf], np.stack([_z(x) for x in d["raw_bh"]], -1), e, c, tau, lbl, "behavior", prm, nms=bn, cls=bc)
        plt_row(ax_m[2, ce], ax_m[2, cf], _z(d["bz"], (0, 1)), e, c, tau, lbl, "PC-bands", prm, heatmap=prm["HEATMAP_MODE"])
        plt_row(ax_m[3, ce], ax_m[3, cf], _z(d["sz"], (0, 1)), e, c, tau, lbl, "explicit PCs", prm, cmap="plasma", heatmap=prm["HEATMAP_MODE"])
        plt_row(ax_m[4, ce], ax_m[4, cf], d["nz"], e, c, tau, lbl, "random neurons", prm, cmap="tab20", pop=True, heatmap=prm["HEATMAP_MODE"])
        ah, bh = ax_m[5, ce], ax_m[5, cf]
        bh.hist(e[:, 1], bins=np.arange(ts, tev + 2) - 0.5, color='tab:blue', alpha=0.3, edgecolor='k')
        rt = [r for b_idx, t_anc in e for r in d["e_all"][d["e_all"][:, 0] == b_idx][:, 1] - t_anc if es <= r <= ee]
        ah.hist(rt, bins=np.arange(es, ee + 2) - 0.5, color='tab:blue', alpha=0.3, edgecolor='k'); ah.set_title(f"{lbl} | auto-corr"); bh.set_title(f"{lbl} | anchor dist"); [x.grid(alpha=0.15) for x in (ah, bh)]
        
        # --- Cross-Correlation Dashboard ---
        tr = d["raw_bh"][5][:, ts:te] # target is model_rev
        ax_reevals = ax_x[mi, 0]
        ax_metrics = ax_x[mi, 1]
        ax_pc      = ax_x[mi, 2]
        ax_bd      = ax_x[mi, 3]
        
        # 1. Panel: Consolidated Neural + Behavior (excluding revs)
        beh_indices = [0, 1, 2, 6, 7, 8]
        for i in range(len(mn)):
            feat = d["raw_mz"][i][:, ts:te]
            mx, sx = get_xcorr(feat, tr, es, ee); ax_metrics.plot(tau, mx, c=mc[i], label=mn[i], alpha=0.8, lw=3)
        for i in beh_indices:
            feat = d["raw_bh"][i][:, ts:te]
            mx, sx = get_xcorr(feat, tr, es, ee); ax_metrics.plot(tau, mx, c=bc[i], label=bn[i], alpha=0.8, lw=3)

        # 2. Panel: Re-evaluations (Correlate Joint and Naive against Model re-eval)
        rev_indices = [3, 4] # Joint, Naive
        for i in rev_indices:
            feat = d["raw_bh"][i][:, ts:te]
            mx, sx = get_xcorr(feat, tr, es, ee); ax_reevals.plot(tau, mx, c=bc[i], label=bn[i], alpha=0.8, lw=3)
        # Also plot Model Re-eval self-correlation
        feat_self = d["raw_bh"][5][:, ts:te]
        mx, sx = get_xcorr(feat_self, tr, es, ee); ax_reevals.plot(tau, mx, c=bc[5], label=bn[5], alpha=0.4, lw=3)

        # 3. PCs and 4. Bands
        num_pcs, num_bands = d["sz"].shape[2], d["bz"].shape[2]
        pc_cls = [plt.get_cmap("plasma")(k / max(num_pcs - 1, 1)) for k in range(num_pcs)]
        bd_cls = [plt.get_cmap("viridis")(k / max(num_bands - 1, 1)) for k in range(num_bands)]
        
        for i in range(num_pcs):
            feat = d["sz"][:, ts:te, i]
            mx, sx = get_xcorr(feat, tr, es, ee); ax_pc.plot(tau, mx, c=pc_cls[i], alpha=0.4, lw=3)
        for i in range(num_bands):
            feat = d["bz"][:, ts:te, i]
            mx, sx = get_xcorr(feat, tr, es, ee); ax_bd.plot(tau, mx, c=bd_cls[i], alpha=0.4, lw=3)

        # Formatting all panels
        for a in ax_x[mi, :]:
            a.axhline(0, c='k', lw=1, alpha=0.5); a.axvline(0, c='k', ls='--', alpha=0.5); a.grid(alpha=0.15)
            a.set(xlabel="tau", ylabel="R")
        
        ax_metrics.set_title(f"{lbl} | All Metrics x Re-eval"); ax_metrics.legend(frameon=False, fontsize=6, loc='upper right', ncol=2)
        ax_reevals.set_title(f"{lbl} | Re-evaluations x Re-eval"); ax_reevals.legend(frameon=False, fontsize=7, loc='upper right')
        ax_pc.set_title(f"{lbl} | PCs ({num_pcs}) x Re-eval")
        ax_bd.set_title(f"{lbl} | Bands ({num_bands}) x Re-eval")
        
    for r in range(5): [ax_m[r, ci].set_xlabel("") for ci in range(4)]; [ax_m[r, ci].tick_params(labelbottom=False) for ci in range(4)]
    for c in range(4): ax_x[0, c].set_xlabel(""); ax_x[0, c].tick_params(labelbottom=False)
    
    fig_m.tight_layout(); fig_x.tight_layout()
    return fig_m, fig_x

def run_event_dynamics(trained, echo, params=None):
    prm = {**EF_PARAMS, **(params or {})}
    rng = np.random.default_rng()
    d_tr, d_ec = prep_model(trained, prm, rng), prep_model(echo, prm, rng)
    fig_m, fig_x = plot_event_dynamics(d_tr, d_ec, prm)
    return {"trained": d_tr, "echo": d_ec, "main": fig_m, "xc": fig_x, "params": prm}

if __name__ == "__main__":
    # cm = dict(mode="SANITY", cuda=0, episodes=1, checkpoint_every=5, realization_num=10, hid_dim=1000, 
    #           obs_num=5, show_plots=False, batch_num=15000, step_num=30, state_num=500, 
    #           learn_embeddings=False, classifier_LR=.001, ctx_num=2, training=False)
    # echo = CognitiveGridworld(**{**cm, 'reservoir': True, 'load_env': "/sanity/reservoir_ctx_2_e5"})
    # trained = CognitiveGridworld(**{**cm, 'reservoir': False, 'load_env': "/sanity/fully_trained_ctx_2_e5"})
        
    run_event_dynamics(trained, echo, params={"E_START": -10, "E_END": 10, "T_START": 3, "T_END": 30, "EVENT_STD_MULT": 1, "CTX_MODE": "both", "HEATMAP_MODE": False})
    plt.show()
    
    run_event_dynamics(trained, echo, params={"E_START": -1, "E_END": 25, "T_START": 3, "T_END": 30, "EVENT_STD_MULT": 1, "CTX_MODE": "both", "HEATMAP_MODE": False})
    plt.show()

    # CTX_MODE: "average", "same", "opposite", "both"
