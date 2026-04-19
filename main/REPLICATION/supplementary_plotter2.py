# import numpy as np
# import torch
# import os
# import sys
# import inspect
# import gc
# import warnings
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from sklearn.decomposition import PCA
# from numpy.lib.stride_tricks import sliding_window_view
# from tqdm import tqdm

# path = inspect.getfile(inspect.currentframe())
# path = os.path.dirname(os.path.abspath(path))
# sys.path.insert(0, path + '/main')
# sys.path.insert(0, path + '/main/bayes')
# sys.path.insert(0, path + '/main/model')
# from main.CognitiveGridworld import CognitiveGridworld

# F64, I64 = np.float64, np.int64

# # ═══════════════════════════════════════════════════════════════════
# # Configuration
# # ═══════════════════════════════════════════════════════════════════

# PLOT_CFG = {
#     "figsize": (26, 12),
#     "early_epochs": 400,             
#     "smooth_w": 1,                   
#     "line_width": 2.5,
#     "title_fs": 16,        
#     "label_fs": 14,
#     "tick_fs": 12,
#     "row_hspace": 0.3 
# }

# DYN_PARAMS = {
#     "K_PCA": None, 
#     "P_BANDS": 1000, 
#     "K_SHOW_PCS": 1000, 
#     "E_START": -8, 
#     "E_END": 8,
#     "T_START": 20, 
#     "T_END": 30
# }

# AGENT_COLORS = {"Trained": "#1f77b4", "Echo": "#ff7f0e", "Joint": "#2ca02c", "Naive": "#d62728"}

# # Publication Defaults
# plt.rcParams.update({
#     'font.size': PLOT_CFG["label_fs"],
#     'axes.labelsize': PLOT_CFG["label_fs"],
#     'axes.titlesize': PLOT_CFG["title_fs"],
#     'axes.titleweight': 'bold',
#     'legend.fontsize': 12,
#     'xtick.labelsize': PLOT_CFG["tick_fs"],
#     'ytick.labelsize': PLOT_CFG["tick_fs"],
#     'axes.linewidth': 1.5,
#     'lines.linewidth': PLOT_CFG["line_width"],
#     'figure.dpi': 300, 
#     'font.family': 'sans-serif',
# })

# # ═══════════════════════════════════════════════════════════════════
# # Logic: Diagnostics
# # ═══════════════════════════════════════════════════════════════════

# def _smooth(y, w):
#     y = np.asarray(y, float)
#     if w <= 1: return y
#     k = np.ones(int(w)) / float(w)
#     return np.convolve(np.pad(y, (int(w)//2, int(w)//2), mode="edge"), k, mode="valid")

# def _ep_xy(y, lim=None):
#     y = np.asarray(y, float)
#     idx = np.arange(y.size) if lim is None else np.arange(min(y.size, int(lim)))
#     return idx + 1, y[idx]

# def pr(evals):
#     e = np.asarray(evals, float)
#     return (e.sum(-1)**2) / ((e*e).sum(-1) + 1e-12)

# # ═══════════════════════════════════════════════════════════════════
# # Logic: Event Dynamics
# # ═══════════════════════════════════════════════════════════════════

# def approx_lik(b, eps=1e-99):
#     b = np.maximum(b.astype(F64), eps)
#     px = np.zeros_like(b)
#     px[:, 0], r = b[:, 0], b[:, 1:] / b[:, :-1]
#     px[:, 1:] = r / r.sum(-1, keepdims=True)
#     return px

# def step_dkl(p, q, eps=1e-99):
#     p, q = np.maximum(p.astype(F64), eps), np.maximum(q.astype(F64), eps)
#     p, q = p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)
#     return 0.5 * np.sum(p * (np.log(p) - np.log(q)) + q * (np.log(q) - np.log(p)), -1)

# def step_angle(x, eps=1e-99):
#     x = x.astype(F64)
#     v1, v2 = x[:, :-1], x[:, 1:]
#     ang = np.full((x.shape[0], x.shape[1]), np.nan, F64)
#     dot = np.einsum('bti,bti->bt', v1, v2)
#     n1, n2 = np.linalg.norm(v1, axis=-1), np.linalg.norm(v2, axis=-1)
#     ang[:, 1:] = np.degrees(np.arccos(np.clip(dot / np.maximum(n1 * n2, eps), -1., 1.)))
#     return ang

# def step_centralization(x, eps=1e-99):
#     x = x.astype(F64)
#     l1 = np.linalg.norm(x, ord=1, axis=-1)
#     l2 = np.linalg.norm(x, ord=2, axis=-1)
#     n = x.shape[-1]
#     sqrt_n = np.sqrt(n)
#     return (sqrt_n - (l1 / np.maximum(l2, eps))) / (sqrt_n - 1.0)

# def pca_upd(m, k_pca=None):
#     upd = m.model_update_flat.astype(F64)
#     B, T, N = upd.shape
#     c = getattr(m, '_svd_cache', {})
#     if c.get('shape') == (B, T, N):
#         x, s, vt = c['x'], c['s'], c['vt']
#     else:
#         x = np.nan_to_num(upd.reshape(B * T, N).copy(), 0., 0., 0.)
#         x -= x.mean(0, keepdims=True)
#         _, s, vt = np.linalg.svd(x, False)
#         m._svd_cache = {'shape': (B, T, N), 'x': x, 's': s, 'vt': vt}
#     k = k_pca if k_pca is not None else vt.shape[0]
#     ev = s * s
#     return upd, (x @ vt[:k].T).reshape(B, T, k).astype(F64), (ev[:k] / (ev.sum() + 1e-99)).astype(F64)

# def met_scores(z):
#     e = z * z
#     return e / np.maximum(e.sum(-1, keepdims=True), 1e-99)

# def m_bands(p, evr, P):
#     bid = np.searchsorted(np.linspace(0., 1., P + 1)[1:-1], np.cumsum(evr), "right")
#     return np.tensordot(p, np.eye(P, dtype=F64)[bid], (2, 0)), bid, np.bincount(bid, minlength=P)

# def get_xcorr(a, b, lag_min, lag_max):
#     B, T = a.shape
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=RuntimeWarning)
#         a_n, b_n = (a - np.nanmean(a, 1, keepdims=True)) / (np.nanstd(a, 1, keepdims=True) + 1e-9), (b - np.nanmean(b, 1, keepdims=True)) / (np.nanstd(b, 1, keepdims=True) + 1e-9)
#     lags = np.arange(lag_min, lag_max + 1)
#     xc = np.zeros((B, len(lags)))
#     for i, lag in enumerate(lags):
#         if lag < 0: xc[:, i] = np.nanmean(a_n[:, :lag] * b_n[:, -lag:], axis=1)
#         elif lag > 0: xc[:, i] = np.nanmean(a_n[:, lag:] * b_n[:, :-lag], axis=1)
#         else: xc[:, i] = np.nanmean(a_n * b_n, axis=1)
#     return np.nanmean(xc, 0), np.nanstd(xc, 0) / np.sqrt(B)

# def prep_model(m, prm):
#     upd_flat = m.model_update_flat.astype(F64)
#     ang = step_angle(upd_flat)
#     upd_norm = np.linalg.norm(upd_flat, axis=-1)
#     cent = step_centralization(upd_flat)

#     px_s = m.naive_px / m.naive_px.sum(-1, keepdims=True)
#     joint_rev = step_dkl(px_s, approx_lik(m.joint_belief.astype(F64))).mean(-1)

#     upd, z, evr = pca_upd(m, prm["K_PCA"])
#     pp = met_scores(z)
#     bnd, bid, bcnt = m_bands(pp, evr, prm["P_BANDS"])
#     sz = z[:, :, :prm["K_SHOW_PCS"]]

#     return {
#         "ang": ang, "upd_norm": upd_norm, "cent": cent,
#         "joint_rev": joint_rev, "upd_flat": upd_flat, "sz": sz, "bz": bnd
#     }

# def analyze_event_dynamics(d, prm):
#     l_min, l_max = prm.get("E_START", -8), prm.get("E_END", 8)
#     ts, te = prm.get("T_START", 0), prm.get("T_END", None)
    
#     tr = d["joint_rev"][:, ts:te]
    
#     # Neural Core
#     ang_mx, _ = get_xcorr(d["ang"][:, ts:te], tr, l_min, l_max)
#     upd_norm_mx, _ = get_xcorr(d["upd_norm"][:, ts:te], tr, l_min, l_max)
#     cent_mx, _ = get_xcorr(d["cent"][:, ts:te], tr, l_min, l_max)
    
#     # Matrices
#     num_pcs = d["sz"].shape[2]
#     num_bands = d["bz"].shape[2]
#     pc_mxs = np.array([get_xcorr(d["sz"][:, ts:te, i], tr, l_min, l_max)[0] for i in range(num_pcs)])
#     bd_mxs = np.array([get_xcorr(d["bz"][:, ts:te, i], tr, l_min, l_max)[0] for i in range(num_bands)])
    
#     valid_b_idx = np.where(~np.all(np.isnan(bd_mxs) | (bd_mxs == 0), axis=1))[0]
#     bd_mxs_v = bd_mxs[valid_b_idx]
    
#     return {
#         "ang_mx": ang_mx, "upd_norm_mx": upd_norm_mx, "cent_mx": cent_mx,
#         "pc_mxs": pc_mxs, "bd_mxs_v": bd_mxs_v, "valid_b_idx": valid_b_idx,
#         "num_pcs": num_pcs, "num_bands": num_bands
#     }

# # ═══════════════════════════════════════════════════════════════════
# # Plotting Helpers
# # ═══════════════════════════════════════════════════════════════════

# def apply_dynamic_ylim(ax_target, max_x=None, pad_factor=0.3):
#     valid_y = []
#     for line in ax_target.get_lines():
#         if line.get_linestyle() == '--': continue 
#         x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
#         mask = (x <= max_x) if max_x else np.ones_like(x, dtype=bool)
#         if np.any(mask): valid_y.extend(y[mask])
#     if valid_y:
#         y_min, y_max = np.nanmin(valid_y), np.nanmax(valid_y)
#         pad = (y_max - y_min) * pad_factor
#         ax_target.set_ylim(y_min - pad, y_max + pad)

# def add_internal_colorbar(fig, ax, CM, A=None, B=None):
#     """Embeds a small colorbar legend directly inside the top-left of the line plots."""
#     cax = ax.inset_axes([0.03, 0.55, 0.02, 0.40]) 
#     sm = plt.cm.ScalarMappable(cmap=CM, norm=plt.Normalize(vmin=0, vmax=1))
#     sm.set_array([])
#     cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
#     cbar.set_ticks([0, 1])
#     cbar.set_ticklabels([A, B])
#     cbar.ax.tick_params(colors='black', labelsize=9, length=0, pad=4)
#     cbar.outline.set_edgecolor('#aaaaaa')
#     cbar.outline.set_linewidth(0.5)
#     cax.invert_yaxis()

# def plot_diagnostics(ax_main, ax_early, agent, name):
#     def ex(y, lim=None): return _ep_xy(_smooth(y, PLOT_CFG["smooth_w"]), lim)
    
#     x, y = ex(agent.test_acc_through_training[:, -1])
#     xi, yi = ex(agent.test_acc_through_training[:, -1], PLOT_CFG["early_epochs"])
#     ax_main[0].plot(x, y, c=AGENT_COLORS[name], label=name, zorder=3)
#     ax_early[0].plot(xi, yi, c=AGENT_COLORS[name], zorder=3)
    
#     if name == "Trained":
#         if hasattr(agent, "joint_acc"):
#             j_acc = float(np.mean(agent.joint_acc[:, -1]))
#             ax_main[0].axhline(j_acc, c=AGENT_COLORS["Joint"], ls="--", label="Joint", zorder=5)
#         if hasattr(agent, "naive_acc"):
#             n_acc = float(np.mean(agent.naive_acc[:, -1]))
#             ax_main[0].axhline(n_acc, c=AGENT_COLORS["Naive"], ls="--", label="Naive", zorder=5)

#     x, y = ex(agent.test_SII_coef_through_training)
#     xi, yi = ex(agent.test_SII_coef_through_training, PLOT_CFG["early_epochs"])
#     ax_main[1].plot(x, y, c=AGENT_COLORS[name], zorder=3)
#     ax_early[1].plot(xi, yi, c=AGENT_COLORS[name], zorder=3)

#     y_raw = pr(agent.test_model_update_dim_through_training) - (pr(agent.test_model_input_dim_through_training) + 1e-12)
#     x, y = ex(y_raw)
#     xi, yi = ex(y_raw, PLOT_CFG["early_epochs"])
#     ax_main[2].plot(x, y, c=AGENT_COLORS[name], zorder=3)
#     ax_early[2].plot(xi, yi, c=AGENT_COLORS[name], zorder=3)

# def plot_event_dynamics_column(fig, ax_list, d_ana, prm):
#     """Plots Neural Core, PCs, and Bands into a given list of 3 axes (Light Mode)."""
#     ax_mz, ax_pc, ax_bd = ax_list
    
#     l_min, l_max = prm.get("E_START", -8), prm.get("E_END", 8)
#     tau = np.arange(l_min, l_max + 1)
#     past = tau <= 0
#     future = tau >= 0
    
#     # 1. Neural Core Lines (Deeper colors for light bg)
#     core_metrics = [
#         (d_ana["ang_mx"], "Rotation", "#d62728", "o"),       # Deep Red
#         (d_ana["upd_norm_mx"], "Magnitude", "#9467bd", "o"), # Deep Purple
#         # (d_ana["cent_mx"], "Centralization", "#009E73", "^")#"2ca02c") # Deep Green
#     ]
#     for mx, name, color, ms in core_metrics:
#         ax_mz.plot(tau[past], mx[past], c=color, alpha=0.5, lw=2.0, ls=':', marker=ms, mec='white', mew=1, ms=4, zorder=3)
#         ax_mz.plot(tau[future], mx[future], c=color, alpha=1, lw=2.5, ls='-', marker=ms, mec='k', mew=1.5, ms=7, label=name, zorder=4)

#     # 2 & 3. PCs and Bands
#     num_pcs, num_bands = d_ana["num_pcs"], d_ana["num_bands"]
#     pc_mxs, bd_mxs_v = d_ana["pc_mxs"], d_ana["bd_mxs_v"]
#     valid_b_idx = d_ana["valid_b_idx"]
#     num_valid_bands = len(valid_b_idx)

#     CM = "managua"
#     # CM = "inferno" 
#     # CM = "plasma"
#     CM = plt.get_cmap(CM).reversed()
#     pc_cls = [CM(k / max(num_pcs - 1, 1)) for k in range(num_pcs)]
#     bd_cls_v = [CM(k / max(num_bands - 1, 1)) for k in valid_b_idx]

#     for i in range(num_pcs): 
#         ax_pc.plot(tau[past], pc_mxs[i][past], c=pc_cls[i], alpha=0.4, lw=1, ls=':', zorder=3)
#         ax_pc.plot(tau[future], pc_mxs[i][future], c=pc_cls[i], alpha=1, lw=1, ls='-', zorder=4)
#     ax_pc.set_ylim([-.25, .25])
    
#     for i in reversed(range(num_valid_bands)): 
#         ax_bd.plot(tau[past], bd_mxs_v[i][past], c=bd_cls_v[i], alpha=0.4, lw=1, ls=':', zorder=3)
#         ax_bd.plot(tau[future], bd_mxs_v[i][future], c=bd_cls_v[i], alpha=1, lw=1, ls='-', zorder=4)
#     ax_bd.set_ylim([-.15, .25])
    
#     # Internal Colorbars
#     add_internal_colorbar(fig, ax_pc, CM, "High Variance PCs", "Low Variance PCs")
#     add_internal_colorbar(fig, ax_bd, CM, "High Variance Clusters", "Low Variance Clusters")

#     # Styling for Event Dynamics
#     for ax in ax_list:
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
#         ax.axhline(0, color='black', alpha=0.5, lw=1.2, zorder=1)
#         ax.axvline(0, color='black', alpha=0.4, ls='--', lw=1.5, zorder=1)
        
#     ax_mz.legend(frameon=False, loc='upper right')
#     ax_mz.set(title="RNN Dynamics", ylabel="Cross-Correlation (R)")
#     ax_pc.set(title=f"PCs ({num_pcs})", ylabel="Cross-Correlation (R)")
#     ax_bd.set(title=f"Bands ({num_valid_bands})", ylabel="Correlation (R)")

# def finalize_layout(ax_main, ax_early, ax_dyn_tr, ax_dyn_ec):
#     titles = ["Accuracy", "Correlation (SII, Accuracy)", "PR(RNN) - PR(Read-in)"]
#     ylabels = ["Accuracy", "Correlation Coeff.", "PR(RNN) - PR(Read-in)"]
    
#     st_cfg = dict(xy=(0.5, 1.15), xycoords='axes fraction', ha='center', va='bottom', 
#                   fontsize=PLOT_CFG["title_fs"]+6, fontweight='bold', alpha=0.75)
    
#     # Column Supertitles
#     ax_main[0].annotate("Full Training", **st_cfg)
#     ax_early[0].annotate("Early Training", **st_cfg)
#     ax_dyn_tr[0].annotate("Post Training", **st_cfg)
#     ax_dyn_ec[0].annotate("Post Training", **st_cfg)

#     # Style Columns 1 & 2 (Diagnostics)
#     for i in range(3):
#         for ax in [ax_main[i], ax_early[i]]:
#             ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
#             ax.grid(True, ls='--', alpha=0.3)
#             ax.set_title(titles[i], pad=12)
#             ax.set_ylabel(ylabels[i])
        
#         apply_dynamic_ylim(ax_main[i], pad_factor=0.4)
#         ax_early[i].set_xlim(0, PLOT_CFG["early_epochs"])
#         apply_dynamic_ylim(ax_early[i], max_x=PLOT_CFG["early_epochs"], pad_factor=0.2)
        
#         if i > 0:
#             ax_main[i].axhline(0, c="k", lw=1.2, alpha=0.6, zorder=1)
#             ax_early[i].axhline(0, c="k", lw=1.2, alpha=0.6, zorder=1)

#     ax_main[0].legend(loc='lower right', frameon=False)
#     for ax_list in [ax_main, ax_early, ax_dyn_tr, ax_dyn_ec]:
#         for ax in ax_list[:2]: 
#             ax.tick_params(labelbottom=False)
#             ax.set_xlabel("")
            
#     ax_main[2].set_xlabel("Testing Episode")
#     ax_early[2].set_xlabel("Episode")
#     ax_dyn_tr[2].set_xlabel("Lag ($\\tau$)")
#     ax_dyn_ec[2].set_xlabel("Lag ($\\tau$)")


# # ═══════════════════════════════════════════════════════════════════
# # Main Execution Block
# # ═══════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     cuda, realization_num, step_num, hid_dim =1, 10, 30, 1000
#     state_num, obs_num, batch_num = 500, 5, 20000

#     # trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1, 'checkpoint_every': 5, 
#     #                                 'realization_num': realization_num, 'hid_dim': hid_dim, 'obs_num': obs_num, 
#     #                                 'show_plots': False, 'batch_num': batch_num, 'step_num': step_num, 
#     #                                 'state_num': state_num, 'learn_embeddings': False, 'reservoir': False, 
#     #                                 'classifier_LR': .001, 'ctx_num': 2, 'training': False, 
#     #                                 'load_env': "/sanity/fully_trained_ctx_2_e5"})       
#     # echo = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1, 'checkpoint_every': 5, 
#     #                              'realization_num': realization_num, 'hid_dim': hid_dim, 'obs_num': obs_num, 
#     #                              'show_plots': False, 'batch_num': batch_num, 'step_num': step_num, 
#     #                              'state_num': state_num, 'learn_embeddings': False, 'reservoir': True, 
#     #                              'classifier_LR': .001, 'ctx_num': 2, 'training': False, 
#     #                              'load_env': "/sanity/reservoir_ctx_2_e5"})    

#     d_tr = prep_model(trained, DYN_PARAMS)
#     a_tr = analyze_event_dynamics(d_tr, DYN_PARAMS)
#     d_ec = prep_model(echo, DYN_PARAMS)
#     a_ec = analyze_event_dynamics(d_ec, DYN_PARAMS)

#     fig_diag = plt.figure(figsize=PLOT_CFG["figsize"])
#     gs_base = GridSpec(1, 4, figure=fig_diag, width_ratios=[1.2, 1, 1.2, 1.2], wspace=0.35)
#     ax_main = [fig_diag.add_subplot(gs_base[0, 0].subgridspec(3, 1, hspace=PLOT_CFG["row_hspace"])[i, 0]) for i in range(3)]
#     ax_early = [fig_diag.add_subplot(gs_base[0, 1].subgridspec(3, 1, hspace=PLOT_CFG["row_hspace"])[i, 0]) for i in range(3)]
#     ax_dyn_tr = [fig_diag.add_subplot(gs_base[0, 2].subgridspec(3, 1, hspace=PLOT_CFG["row_hspace"])[i, 0]) for i in range(3)]
#     ax_dyn_ec = [fig_diag.add_subplot(gs_base[0, 3].subgridspec(3, 1, hspace=PLOT_CFG["row_hspace"])[i, 0]) for i in range(3)]
#     plot_diagnostics(ax_main, ax_early, trained, "Trained")
#     plot_diagnostics(ax_main, ax_early, echo, "Echo")
#     plot_event_dynamics_column(fig_diag, ax_dyn_tr, a_tr, DYN_PARAMS)
#     plot_event_dynamics_column(fig_diag, ax_dyn_ec, a_ec, DYN_PARAMS)
#     finalize_layout(ax_main, ax_early, ax_dyn_tr, ax_dyn_ec)
#     plt.savefig("diagnostics_with_event_dynamics.svg", format="svg", bbox_inches="tight", facecolor='white')
#     plt.show()


import numpy as np
import torch
import os
import sys
import inspect
import gc
import warnings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

path = inspect.getfile(inspect.currentframe())
path = os.path.dirname(os.path.abspath(path))
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld

F64, I64 = np.float64, np.int64

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

PLOT_CFG = {
    "figsize_A": (14, 5),   # 1 x 4 Layout
    "figsize_B": (16, 5),   # 1 x 3 Layout
    "early_epochs": 400,             
    "smooth_w": 1,                   
    "line_width": 2.5,
    "title_fs": 16,        
    "label_fs": 14,
    "tick_fs": 12,
}

DYN_PARAMS = {
    "E_START": -8, 
    "E_END": 8,
    "T_START": 20,    
    "T_END": 30
}

AGENT_COLORS = {"Trained": "#1f77b4", "Echo": "#ff7f0e", "Joint": "#2ca02c", "Naive": "#d62728"}

# Publication Defaults
plt.rcParams.update({
    'font.size': PLOT_CFG["label_fs"],
    'axes.labelsize': PLOT_CFG["label_fs"],
    'axes.titlesize': PLOT_CFG["title_fs"],
    'axes.titleweight': 'bold',
    'legend.fontsize': 12,
    'xtick.labelsize': PLOT_CFG["tick_fs"],
    'ytick.labelsize': PLOT_CFG["tick_fs"],
    'axes.linewidth': 1.5,
    'lines.linewidth': PLOT_CFG["line_width"],
    'figure.dpi': 300, 
    'font.family': 'sans-serif',
})

# ═══════════════════════════════════════════════════════════════════
# Logic: Diagnostics
# ═══════════════════════════════════════════════════════════════════

def _smooth(y, w):
    y = np.asarray(y, float)
    if w <= 1: return y
    k = np.ones(int(w)) / float(w)
    return np.convolve(np.pad(y, (int(w)//2, int(w)//2), mode="edge"), k, mode="valid")

def _ep_xy(y, lim=None):
    y = np.asarray(y, float)
    idx = np.arange(y.size) if lim is None else np.arange(min(y.size, int(lim)))
    return idx + 1, y[idx]

def pr(evals):
    e = np.asarray(evals, float)
    return (e.sum(-1)**2) / ((e*e).sum(-1) + 1e-12)

# ═══════════════════════════════════════════════════════════════════
# Logic: Event Dynamics
# ═══════════════════════════════════════════════════════════════════

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

def step_participation_ratio(x, eps=1e-99):
    x = x.astype(F64)
    x_sq = x ** 2
    return (np.sum(x_sq, axis=-1) ** 2) / np.maximum(np.sum(x_sq ** 2, axis=-1), eps)

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

def prep_model(m, prm):
    upd_flat = m.model_update_flat.astype(F64)
    ang = step_angle(upd_flat)
    upd_norm = np.linalg.norm(upd_flat, axis=-1)
    part_ratio = step_participation_ratio(upd_flat)

    px_s = m.naive_px / m.naive_px.sum(-1, keepdims=True)
    joint_rev = step_dkl(px_s, approx_lik(m.joint_belief.astype(F64))).mean(-1)

    return {
        "ang": ang, "upd_norm": upd_norm, "part_ratio": part_ratio,
        "joint_rev": joint_rev, "upd_flat": upd_flat
    }

def analyze_event_dynamics(d, prm):
    l_min, l_max = prm.get("E_START", -8), prm.get("E_END", 8)
    ts, te = prm.get("T_START", 0), prm.get("T_END", None)
    
    tr = d["joint_rev"][:, ts:te]
    
    # Neural Core
    ang_mx, _ = get_xcorr(d["ang"][:, ts:te], tr, l_min, l_max)
    upd_norm_mx, _ = get_xcorr(d["upd_norm"][:, ts:te], tr, l_min, l_max)
    pr_mx, _ = get_xcorr(d["part_ratio"][:, ts:te], tr, l_min, l_max)
    
    return {
        "ang_mx": ang_mx, "upd_norm_mx": upd_norm_mx, "pr_mx": pr_mx
    }

# ═══════════════════════════════════════════════════════════════════
# Plotting Helpers
# ═══════════════════════════════════════════════════════════════════

def apply_dynamic_ylim(ax_target, max_x=None, pad_factor=0.3):
    valid_y = []
    for line in ax_target.get_lines():
        if line.get_linestyle() == '--': continue 
        x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        mask = (x <= max_x) if max_x else np.ones_like(x, dtype=bool)
        if np.any(mask): valid_y.extend(y[mask])
    if valid_y:
        y_min, y_max = np.nanmin(valid_y), np.nanmax(valid_y)
        pad = (y_max - y_min) * pad_factor
        ax_target.set_ylim(y_min - pad, y_max + pad)

def plot_diagnostics_1x4(axes, agent, name):
    def ex(y, lim=None): return _ep_xy(_smooth(y, PLOT_CFG["smooth_w"]), lim)
    
    # 0. Full Accuracy
    x, y = ex(agent.test_acc_through_training[:, -1])
    axes[0].plot(x, y, c=AGENT_COLORS[name], label=name, zorder=3)
    if name == "Trained":
        j_acc = float(np.mean(agent.joint_acc[:, -1]))
        axes[0].axhline(j_acc, c=AGENT_COLORS["Joint"], ls="--", label="Joint", zorder=5)
    else:
        n_acc = float(np.mean(agent.naive_acc[:, -1]))
        axes[0].axhline(n_acc, c=AGENT_COLORS["Naive"], ls="--", label="Naive", zorder=5)

    # 1. Early Accuracy
    xi, yi = ex(agent.test_acc_through_training[:, -1], PLOT_CFG["early_epochs"])
    axes[1].plot(xi, yi, c=AGENT_COLORS[name], zorder=3)

    # 2. Early Correlation
    xi, yi = ex(agent.test_SII_coef_through_training, PLOT_CFG["early_epochs"])
    axes[2].plot(xi, yi, c=AGENT_COLORS[name], zorder=3)

    # 3. Early PR
    y_raw = pr(agent.test_model_update_dim_through_training) - (pr(agent.test_model_input_dim_through_training) + 1e-12)
    xi, yi = ex(y_raw, PLOT_CFG["early_epochs"])
    axes[3].plot(xi, yi, c=AGENT_COLORS[name], zorder=3)

def plot_event_dynamics_comparison(ax_list, a_tr, a_ec, prm):
    """Plots Neural Core comparison between Trained and Echo models."""
    l_min, l_max = prm.get("E_START", -8), prm.get("E_END", 8)
    tau = np.arange(l_min, l_max + 1)
    past = tau <= 0
    future = tau >= 0
    
    metrics = [
        ("ang_mx", "Rotation", ax_list[0]),
        ("upd_norm_mx", "Magnitude", ax_list[1]),
        ("pr_mx", "PR(RNN)", ax_list[2])
    ]
    
    for key, name, ax in metrics:
        for d_ana, model_name in [(a_tr, "Trained"), (a_ec, "Echo")]:
            color = AGENT_COLORS[model_name]
            mx = d_ana[key]
            
            ax.plot(tau[past], mx[past], c=color, alpha=0.5, lw=2.0, ls=':', marker='o', mec='white', mew=1, ms=4, zorder=3)
            ax.plot(tau[future], mx[future], c=color, alpha=1, lw=2.5, ls='-', marker='o', mec='k', mew=1.5, ms=7, label=model_name, zorder=4)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.axhline(0, color='black', alpha=0.5, lw=1.2, zorder=1)
        ax.axvline(0, color='black', alpha=0.4, ls='--', lw=1.5, zorder=1)
        ax.set(title=name, ylabel="r")
        
    ax_list[0].legend(frameon=False, loc='upper right')

def finalize_layout_A(axes):
    titles = ["Final Step Accuracy", "Final Step Accuracy", "Correlation(SII, Acc.)", "PR(RNN) - PR(Read-in)"]
    ylabels = ["Accuracy", "Accuracy", "r", r"$\Delta$ PR"]
    
    for i, ax in enumerate(axes):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, ls='--', alpha=0.3)
        ax.set_title(titles[i], pad=12)
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel("Testing Episode")
        
        if i == 0:
            apply_dynamic_ylim(ax, pad_factor=0.4)
        else:
            ax.set_xlim(0, PLOT_CFG["early_epochs"])
            apply_dynamic_ylim(ax, max_x=PLOT_CFG["early_epochs"], pad_factor=0.2)
            if i > 1: # Corr and PR have a zero-line baseline
                ax.axhline(0, c="k", lw=1.2, alpha=0.6, zorder=1)

    axes[0].legend(loc='lower right', frameon=False, ncols=2)
    plt.tight_layout()

def finalize_layout_B(axes):
    for ax in axes:
        ax.set_xlabel(r"Lag ($\tau$)")
    plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════
# Main Execution Block
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cuda, realization_num, step_num, hid_dim = 0, 10, 30, 1000
    state_num, obs_num, batch_num = 500, 5, 20000

    # trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1, 'checkpoint_every': 5, 
    #                                 'realization_num': realization_num, 'hid_dim': hid_dim, 'obs_num': obs_num, 
    #                                 'show_plots': False, 'batch_num': batch_num, 'step_num': step_num, 
    #                                 'state_num': state_num, 'learn_embeddings': False, 'reservoir': False, 
    #                                 'classifier_LR': .001, 'ctx_num': 2, 'training': False, 
    #                                 'load_env': "/sanity/fully_trained_ctx_2_e5"})       
    # echo = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1, 'checkpoint_every': 5, 
    #                              'realization_num': realization_num, 'hid_dim': hid_dim, 'obs_num': obs_num, 
    #                              'show_plots': False, 'batch_num': batch_num, 'step_num': step_num, 
    #                              'state_num': state_num, 'learn_embeddings': False, 'reservoir': True, 
    #                              'classifier_LR': .001, 'ctx_num': 2, 'training': False, 
    #                              'load_env': "/sanity/reservoir_ctx_2_e5"})    

    d_tr = prep_model(trained, DYN_PARAMS)
    a_tr = analyze_event_dynamics(d_tr, DYN_PARAMS)
    d_ec = prep_model(echo, DYN_PARAMS)
    a_ec = analyze_event_dynamics(d_ec, DYN_PARAMS)

    # ---------------------------------------------------------
    # Plot A: 1x4 Diagnostics 
    # ---------------------------------------------------------
    figA, axesA = plt.subplots(1, 4, figsize=PLOT_CFG["figsize_A"])
    plot_diagnostics_1x4(axesA, trained, "Trained")
    plot_diagnostics_1x4(axesA, echo, "Echo")
    finalize_layout_A(axesA)
    
    figA.savefig("plot_A_diagnostics.svg", format="svg", bbox_inches="tight", facecolor='white')

    # ---------------------------------------------------------
    # Plot B: 1x3 Cross-Correlations
    # ---------------------------------------------------------
    figB, axesB = plt.subplots(1, 3, figsize=PLOT_CFG["figsize_B"])
    plot_event_dynamics_comparison(axesB, a_tr, a_ec, DYN_PARAMS)
    figB.suptitle("Cross-Correlation(Dis-entanglement, " + r"$\cdot$" + " )")
    finalize_layout_B(axesB)
    figB.savefig("plot_B_cross_correlations.svg", format="svg", bbox_inches="tight", facecolor='white')
    plt.show()

