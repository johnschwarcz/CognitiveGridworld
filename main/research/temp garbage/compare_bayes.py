""" COMPARE JOINT VS NAIVE BAYESIAN BELIEF INFERENCE """

# %matplotlib auto
# %matplotlib inline
import numpy as np; import torch; import pylab as plt; import os; import sys; import inspect
from matplotlib.colors import Normalize; import math

try:
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld

if __name__ == "__main__":
    mode = None  # [None, "SANITY", "RL"]  
    cuda = 0

    self = CognitiveGridworld(**{
        'episodes': 1,
        'state_num': 500, 
        'batch_num': 15000, 
        'step_num': 30, 
        'obs_num': 5, 
        'ctx_num': 2, 
        'KQ_dim': 30, 
        'realization_num': 10,
        'likelihood_temp': 2,
        'show_plots': True,

        'mode': mode,
        'hid_dim': 1000,
        'classifier_LR': .0005, 
        'controller_LR': .005, 
        'generator_LR': .001,
        'learn_embeddings': True,   
        'reservoir': False,
        'save_env': None,
        'load_env': None,
        'cuda': cuda})

    JGB = self.joint_goal_belief
    NGB = self.naive_goal_belief
    Jent = (-JGB * np.log(JGB)).sum(-1)
    Nent = (-NGB * np.log(NGB)).sum(-1)
    ent_diff = Jent - Nent
    Jlogit = np.log(self.joint_TP / (1-self.joint_TP))
    Nlogit = np.log(self.naive_TP / (1-self.naive_TP))
    belief_diff = Jlogit - Nlogit

    #################### PLOTTING ####################

    """ INITIAL ENTROPY AND LOGIT HISTOGRAMS / 3D SCATTER PLOTS """
    fig = plt.figure(figsize=(8, 10)); views = ((90,-90), (0, -90), (0, 0), (5, -60))
    for i, (elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(1, 4, i, projection='3d')
        for t in range(self.step_num):
            c = plt.cm.coolwarm(t / max(1, self.step_num - 1))
            ax.plot(t, Nent[:, t], Nlogit[:, t], 'o', color=c, alpha=.05, ms=3)
        ax.set_xlabel("t"); ax.set_ylabel("Nent"); ax.set_zlabel("NLLR");    ax.view_init(elev=elev, azim=azim)
        ax.set_zlim(-20,20)

        ax = fig.add_subplot(2, 4, i, projection='3d')
        for t in range(self.step_num):
            c = plt.cm.coolwarm(t / max(1, self.step_num - 1))
            ax.plot(t, Jent[:, t], Jlogit[:, t], 'o', color=c, alpha= .05, ms=3)
        ax.set_xlabel("t"); ax.set_ylabel("Jent"); ax.set_zlabel("JLLR");    ax.view_init(elev=elev, azim=azim)
        ax.set_zlim(-20,20)
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, self.step_num - 1))
    sm.set_array([]);plt.tight_layout();plt.show()

    t = np.arange(self.step_num, dtype=float)
    n = Jent.shape[0]
    Jent_m = Jent.mean(0); Jent_se = Jent.std(0, ddof=1)
    Nent_m = Nent.mean(0); Nent_se = Nent.std(0, ddof=1) 
    Jlog_m = Jlogit.mean(0); Jlog_se = Jlogit.std(0, ddof=1) 
    Nlog_m = Nlogit.mean(0); Nlog_se = Nlogit.std(0, ddof=1)

    fig, ax = plt.subplots(1, 3, figsize=(8, 3), tight_layout=True)
    ax[0].plot(t, Jent_m); ax[0].fill_between(t, Jent_m - Jent_se, Jent_m + Jent_se, alpha=.4)
    ax[0].plot(t, Nent_m); ax[0].fill_between(t, Nent_m - Nent_se, Nent_m + Nent_se, alpha=.4)
    ax[0].set_title("Entropy"); ax[0].set_xlabel("t"); ax[0].set_ylabel("entropy")
    ax[1].plot(t, Jlog_m); ax[1].fill_between(t, Jlog_m - Jlog_se, Jlog_m + Jlog_se, alpha=.4)
    ax[1].plot(t, Nlog_m); ax[1].fill_between(t, Nlog_m - Nlog_se, Nlog_m + Nlog_se, alpha=.4)
    ax[1].set_title("True Positive logit"); ax[1].set_xlabel("t"); ax[1].set_ylabel("log(TP/(1-TP))")
    ax[2].plot(t, Jlog_m)
    ax[2].plot(t, Nlog_m)
    ax[2].set_title("True Positive logit"); ax[2].set_xlabel("t"); ax[2].set_ylabel("log(TP/(1-TP))")
    plt.show()

    ### NEW PLOT ###

    """ PLOTTING CONFIDENCE ACCURACY VECTOR FIELDS """

    def preprocess_belief(GB, goal_value, eps=1e-12):
        """ Normalize goal beliefs and compute entropy + true-positive logit """
        GB = np.asarray(GB, dtype=float)
        GB = np.clip(GB, eps, 1.0 - eps)
        GB = GB / GB.sum(-1, keepdims=True)
        B, T, _ = GB.shape
        bix = np.arange(B)[:, None]
        tix = np.arange(T)[None, :]
        ent = -(GB * np.log(GB)).sum(-1)
        gv = np.asarray(goal_value)
        if gv.ndim == 0:
            gv = np.full(B, int(gv), dtype=np.int64)
        else:
            gv = gv.astype(np.int64).reshape(B)
        p_true = GB[bix, tix, gv[:, None]]
        logit = np.log(p_true / (1.0 - p_true))
        return ent, logit, T


    def bins_single(ent, logit, nx=42, ny=34, qlo=0.5, qhi=99.5):
        """ Compute bin edges for entropy/logit based on percentiles """
        x = ent[:, :-1].reshape(-1)
        y = logit[:, :-1].reshape(-1)
        xlo, xhi = np.nanpercentile(x, (qlo, qhi))
        ylo, yhi = np.nanpercentile(y, (qlo, qhi))
        if not np.isfinite(xlo) or not np.isfinite(xhi):
            xlo, xhi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(ylo) or not np.isfinite(yhi):
            ylo, yhi = np.nanmin(y), np.nanmax(y)
        if xhi <= xlo:
            xhi = xlo + 1e-9
        if yhi <= ylo:
            yhi = ylo + 1e-9
        xb = np.linspace(xlo, xhi, nx + 1)
        yb = np.linspace(ylo, yhi, ny + 1)
        return xb, yb


    def signed_mask(dx, dy, sx=0, sy=0):
        """ Create boolean mask for transitions with specified sign in dx/dy """
        mx = np.ones(dx.shape, dtype=bool) if sx == 0 else (dx > 0 if sx > 0 else dx < 0)
        my = np.ones(dy.shape, dtype=bool) if sy == 0 else (dy > 0 if sy > 0 else dy < 0)
        return mx & my


    def corr_stats(x, y):
        """ Compute Pearson correlation, p-value, and sample count """
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        n = x.size
        if n < 3:
            return np.nan, np.nan, n
        sx = np.std(x)
        sy = np.std(y)
        if sx <= 0 or sy <= 0:
            return np.nan, np.nan, n
        if HAS_SCIPY:
            r, p = pearsonr(x, y)
            return float(r), float(p), int(n)
        r = float(np.corrcoef(x, y)[0, 1])
        if (not np.isfinite(r)) or (abs(r) >= 1.0):
            return r, np.nan, n
        z = 0.5 * np.log((1.0 + r) / (1.0 - r)) * np.sqrt(max(n - 3.0, 1.0))
        p = math.erfc(abs(z) / np.sqrt(2.0))
        return r, float(p), n


    def p_to_stars(p):
        """ Convert p-value to significance stars (*/**/***) """
        if not np.isfinite(p):
            return "n/a"
        if p < 1e-3:
            return "***"
        if p < 1e-2:
            return "**"
        if p < 5e-2:
            return "*"
        return "n.s."


    def p_str(p):
        """ Format p-value as a readable string """
        if not np.isfinite(p):
            return "nan"
        if p == 0.0:
            return "<1e-300"
        if p < 1e-3:
            return f"{p:.1e}"
        return f"{p:.3f}"


    def subtitle_with_stats(label, share, r, p):
        """ Generate subtitle string with share %, correlation, and p-value """
        rtxt = f"{r:+.2f}" if np.isfinite(r) else "nan"
        if np.isfinite(p):
            ptxt = "<1e-99" if p < 1e-99 else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")
            st = p_to_stars(p)
        else:
            ptxt = "nan"
            st = "n/a"
        return f"{label} [{share:.1f}%] | r={rtxt}, p={ptxt} ({st})"


    """ FLOW FIELD COMPUTATION """

    def compute_signed_median_trajectory(ent, logit, sx=0, sy=0):
        """ Compute median trajectory for transitions matching sign constraints """
        B, T = ent.shape
        xm = np.full(T, np.nan, dtype=float)
        ym = np.full(T, np.nan, dtype=float)
        if T < 2:
            return xm, ym
        dx = ent[:, 1:] - ent[:, :-1]
        dy = logit[:, 1:] - logit[:, :-1]
        m = signed_mask(dx, dy, sx=sx, sy=sy)
        fin = (
            np.isfinite(ent[:, :-1]) & np.isfinite(logit[:, :-1]) &
            np.isfinite(ent[:, 1:]) & np.isfinite(logit[:, 1:])
        )
        m = m & fin
        t = 0
        while t < T - 1:
            mt = m[:, t]
            if np.any(mt):
                xm[t] = np.median(ent[mt, t])
                ym[t] = np.median(logit[mt, t])
            t += 1
        mend = m[:, T - 2]
        if np.any(mend):
            xm[T - 1] = np.median(ent[mend, T - 1])
            ym[T - 1] = np.median(logit[mend, T - 1])
        return xm, ym


    def compute_flow_raw(ent, logit, xb, yb, min_count=12, sx=0, sy=0, with_median=True):
        """ Compute binned flow vectors (U, V) from belief transitions """
        x0 = ent[:, :-1].reshape(-1)
        y0 = logit[:, :-1].reshape(-1)
        x1 = ent[:, 1:].reshape(-1)
        y1 = logit[:, 1:].reshape(-1)
        dx = x1 - x0
        dy = y1 - y0

        nx = xb.size - 1
        ny = yb.size - 1
        ix = np.digitize(x0, xb) - 1
        iy = np.digitize(y0, yb) - 1

        inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        fin = np.isfinite(dx) & np.isfinite(dy)
        sgn = signed_mask(dx, dy, sx=sx, sy=sy)
        good = inb & fin & sgn

        ixg = ix[good]
        iyg = iy[good]
        dxg = dx[good]
        dyg = dy[good]
        flat = iyg * nx + ixg

        count = np.zeros((ny, nx), dtype=float)
        sum_u = np.zeros((ny, nx), dtype=float)
        sum_v = np.zeros((ny, nx), dtype=float)

        np.add.at(count.reshape(-1), flat, 1.0)
        np.add.at(sum_u.reshape(-1), flat, dxg)
        np.add.at(sum_v.reshape(-1), flat, dyg)

        U = np.divide(sum_u, count, out=np.zeros_like(sum_u), where=count > 0)
        V = np.divide(sum_v, count, out=np.zeros_like(sum_v), where=count > 0)
        keep = count >= min_count

        xc = 0.5 * (xb[:-1] + xb[1:])
        yc = 0.5 * (yb[:-1] + yb[1:])
        X, Y = np.meshgrid(xc, yc)

        if with_median:
            xm, ym = compute_signed_median_trajectory(ent, logit, sx=sx, sy=sy)
        else:
            xm, ym = np.array([]), np.array([])

        total = float(count.sum())
        r, p, n_corr = corr_stats(x0[good], y0[good])

        return {
            "count": count,
            "X": X,
            "Y": Y,
            "U": U,
            "V": V,
            "keep": keep,
            "xm": xm,
            "ym": ym,
            "total": total,
            "r": r,
            "p": p,
            "n_corr": n_corr
        }


    def full_flow_reference(flow):
        """ Get total count from flow for normalization reference """
        total = flow["count"].sum()
        return total if total > 0 else 1.0


    def make_pair_maps(flow_a, flow_b, ref_total, eps=1e-12):
        """ Compute signed dominance maps between two flow fields """
        ca = flow_a["count"]
        cb = flow_b["count"]
        den = ref_total if ref_total > 0 else 1.0
        pa = ca / den
        pb = cb / den
        support = pa + pb
        sig_a = np.divide(pa - pb, support + eps)  # signed dominance ratio in [-1,1]
        sig_b = -sig_a
        vmax = 1.0
        return sig_a, sig_b, support, vmax


    """ DRAWING / VISUALIZATION FUNCTIONS """

    def draw_median_main(ax, xm, ym, cmap, norm):
        """ Draw median trajectory line with time-colored markers """
        if xm.size == 0:
            return
        valid = np.isfinite(xm) & np.isfinite(ym)
        if np.sum(valid) < 2:
            return
        xv = xm[valid]
        yv = ym[valid]
        tv = np.arange(xm.size, dtype=float)[valid]
        ax.plot(xv, yv, c="0.15", lw=1.8, zorder=4)
        ax.scatter(xv, yv, c=tv, cmap=cmap, norm=norm, s=22, zorder=5)
        i0 = np.flatnonzero(valid)[0]
        i1 = np.flatnonzero(valid)[-1]
        ax.scatter(xm[i0], ym[i0], c="blue", s=18, edgecolors="k", linewidths=0.25, zorder=6)
        ax.scatter(xm[i1], ym[i1], c="red", s=20, marker="X", edgecolors="k", linewidths=0.25, zorder=6)


    def draw_trajectories(ax, ent, logit, n_examples, cmap, norm, title):
        """ Draw individual belief trajectories in entropy-logit space """
        B, T = ent.shape
        n = min(n_examples, B)
        sel = np.random.permutation(B)[:n]
        tt = np.arange(T, dtype=float)
        i = 0
        while i < n:
            bi = sel[i]
            ax.plot(ent[bi], logit[bi], c="0.80", lw=0.9, alpha=0.75, zorder=1)
            ax.scatter(ent[bi], logit[bi], c=tt, cmap=cmap, norm=norm, s=12, linewidths=0, zorder=2)
            ax.scatter(ent[bi, 0], logit[bi, 0], c="blue", s=15, marker="o", edgecolors="k", linewidths=0.25, zorder=3)
            ax.scatter(ent[bi, -1], logit[bi, -1], c="red", s=17, marker="X", edgecolors="k", linewidths=0.25, zorder=3)
            i += 1
        ax.set_xscale("symlog")
        ax.set_title(title)
        ax.set_xlabel("Belief entropy")
        ax.set_ylabel("True-positive logit")
        ax.grid(alpha=0.2)


    def draw_flow_main(ax, flow, xb, yb, title, dens_vmax, time_cmap, norm_time, density_cmap="viridis", flow_scale=1.0):
        """ Draw main flow field with density heatmap and quiver arrows """
        count = flow["count"]
        X = flow["X"]
        Y = flow["Y"]
        U = flow["U"]
        V = flow["V"]
        keep = flow["keep"]

        dens = np.log10(count + 1.0)
        ax.pcolormesh(xb, yb, dens, shading="auto", alpha=0.34, vmin=0.0, vmax=dens_vmax, cmap=density_cmap)

        self_total = count.sum() if count.sum() > 0 else 1.0
        frac = count / self_total
        self_mean = frac[keep].mean() if np.any(keep) else 1.0
        if self_mean <= 0:
            self_mean = 1.0

        n_ar = int(np.sum(keep))
        rgba = np.zeros((n_ar, 4), dtype=float)
        if n_ar > 0:
            a = np.clip(frac[keep] / self_mean, 0.0, 1.0)
            rgba[:, 3] = a
            ax.quiver(
                X[keep], Y[keep], U[keep], V[keep],
                angles="xy", scale_units="xy", scale=flow_scale,
                color=rgba, width=0.003, zorder=2
            )

        draw_median_main(ax, flow["xm"], flow["ym"], time_cmap, norm_time)
        ax.set_xscale("symlog")
        ax.set_title(title)
        ax.set_xlabel("Belief entropy")
        ax.set_ylabel("True-positive logit")
        ax.grid(alpha=0.2)


    def mini_rgba_from_maps(signal_map, support_map, keep, cmap, vmax, eps=1e-12):
        """ Compute RGBA colors for mini arrows based on signal and support """
        sgn = signal_map[keep]
        sup = support_map[keep]

        cm = plt.get_cmap(cmap)
        norm = Normalize(vmin=-vmax, vmax=vmax)
        rgba = cm(norm(sgn))

        pos = sup > 0
        sref = np.nanmean(sup[pos]) if np.any(pos) else 1.0
        if (not np.isfinite(sref)) or (sref <= 0):
            sref = 1.0

        a_support = np.sqrt(np.divide(sup, sref + eps))
        a_support = np.clip(a_support, 0.0, 1.0)
        rgba[:, 3] = a_support
        return rgba


    def draw_flow_mini_arrows(ax, flow, title_label, signal_map, support_map, vmax, ref_total, mini_cmap="coolwarm", flow_scale=1.0):
        """ Draw mini subplot with colored quiver arrows for signed flow """
        count = flow["count"]
        X = flow["X"]
        Y = flow["Y"]
        U = flow["U"]
        V = flow["V"]
        keep = flow["keep"]

        ax.set_facecolor("black")

        rgba = mini_rgba_from_maps(signal_map=signal_map, support_map=support_map, keep=keep, cmap=mini_cmap, vmax=vmax)
        if rgba.shape[0] > 0:
            ax.quiver(
                X[keep], Y[keep], U[keep], V[keep],
                angles="xy", scale_units="xy", scale=flow_scale,
                color=rgba, width=0.0035, zorder=2
            )

        share = 100.0 * (count.sum() / ref_total) if ref_total > 0 else 0.0
        ttl = subtitle_with_stats(title_label, share, flow["r"], flow["p"])

        ax.text(
            0.01, 0.99, ttl,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=15, color="white",
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.65, pad=1.3),
            clip_on=False, zorder=10
        )

        ax.set_xscale("symlog")
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        for k in ax.spines:
            ax.spines[k].set_color("0.35")


    def vmax_main(flow):
        """ Get max log-density value for colorbar scaling """
        return np.nanmax(np.log10(flow["count"] + 1.0))


    """ MAIN PLOTTING FUNCTION: JOINT VS NAIVE COMPARISON """

    def plot_joint_naive_4mini(
        joint_GB,
        naive_GB,
        goal_value,
        n_examples=100,
        nx=42,
        ny=34,
        min_count=12,
        min_count_signed=4,
        qlo=0.5,
        qhi=99.5,
        cmap="coolwarm",          # fallback for mini cmap
        time_cmap=None,           # trajectories/main median time colors
        mini_cmap=None,           # mini arrows
        density_cmap="viridis",   # main flow background
        flow_scale=1.0,
        flow_scale_signed=1.0,
        figsize=(22, 10)
    ):
        """
        Create full comparison figure: trajectories, flow fields, and 
        signed mini-panels for both Joint and Naive belief inference.
        """
        if time_cmap is None:
            time_cmap = "coolwarm"   # keep trajectories as before
        if mini_cmap is None:
            mini_cmap = cmap

        ent_j, logit_j, Tj = preprocess_belief(joint_GB, goal_value)
        ent_n, logit_n, Tn = preprocess_belief(naive_GB, goal_value)

        xb_j, yb_j = bins_single(ent_j, logit_j, nx=nx, ny=ny, qlo=qlo, qhi=qhi)
        xb_n, yb_n = bins_single(ent_n, logit_n, nx=nx, ny=ny, qlo=qlo, qhi=qhi)

        full_j = compute_flow_raw(ent_j, logit_j, xb_j, yb_j, min_count=min_count, sx=0, sy=0, with_median=True)
        full_n = compute_flow_raw(ent_n, logit_n, xb_n, yb_n, min_count=min_count, sx=0, sy=0, with_median=True)

        j_dLp = compute_flow_raw(ent_j, logit_j, xb_j, yb_j, min_count=min_count_signed, sx=0, sy=1, with_median=False)
        j_dLn = compute_flow_raw(ent_j, logit_j, xb_j, yb_j, min_count=min_count_signed, sx=0, sy=-1, with_median=False)
        j_dHp = compute_flow_raw(ent_j, logit_j, xb_j, yb_j, min_count=min_count_signed, sx=1, sy=0, with_median=False)
        j_dHn = compute_flow_raw(ent_j, logit_j, xb_j, yb_j, min_count=min_count_signed, sx=-1, sy=0, with_median=False)

        n_dLp = compute_flow_raw(ent_n, logit_n, xb_n, yb_n, min_count=min_count_signed, sx=0, sy=1, with_median=False)
        n_dLn = compute_flow_raw(ent_n, logit_n, xb_n, yb_n, min_count=min_count_signed, sx=0, sy=-1, with_median=False)
        n_dHp = compute_flow_raw(ent_n, logit_n, xb_n, yb_n, min_count=min_count_signed, sx=1, sy=0, with_median=False)
        n_dHn = compute_flow_raw(ent_n, logit_n, xb_n, yb_n, min_count=min_count_signed, sx=-1, sy=0, with_median=False)

        ref_total_j = full_flow_reference(full_j)
        ref_total_n = full_flow_reference(full_n)

        j_dl_pos, j_dl_neg, j_dl_support, j_dl_v = make_pair_maps(j_dLp, j_dLn, ref_total_j)
        j_dh_pos, j_dh_neg, j_dh_support, j_dh_v = make_pair_maps(j_dHp, j_dHn, ref_total_j)

        n_dl_pos, n_dl_neg, n_dl_support, n_dl_v = make_pair_maps(n_dLp, n_dLn, ref_total_n)
        n_dh_pos, n_dh_neg, n_dh_support, n_dh_v = make_pair_maps(n_dHp, n_dHn, ref_total_n)

        norm_time = Normalize(vmin=0, vmax=max(Tj, Tn) - 1)

        fig = plt.figure(figsize=figsize, constrained_layout=False)
        gs = fig.add_gridspec(2, 3, width_ratios=(1.16, 1.16, 2.30), wspace=0.08, hspace=0.18)

        ax_tj = fig.add_subplot(gs[0, 0])
        ax_fj = fig.add_subplot(gs[0, 1])
        gs_jm = gs[0, 2].subgridspec(2, 2, wspace=0.03, hspace=0.08)
        ax_j_tl = fig.add_subplot(gs_jm[0, 0])
        ax_j_bl = fig.add_subplot(gs_jm[1, 0])
        ax_j_tr = fig.add_subplot(gs_jm[0, 1])
        ax_j_br = fig.add_subplot(gs_jm[1, 1])

        ax_tn = fig.add_subplot(gs[1, 0])
        ax_fn = fig.add_subplot(gs[1, 1])
        gs_nm = gs[1, 2].subgridspec(2, 2, wspace=0.03, hspace=0.08)
        ax_n_tl = fig.add_subplot(gs_nm[0, 0])
        ax_n_bl = fig.add_subplot(gs_nm[1, 0])
        ax_n_tr = fig.add_subplot(gs_nm[0, 1])
        ax_n_br = fig.add_subplot(gs_nm[1, 1])

        draw_trajectories(ax_tj, ent_j, logit_j, n_examples, time_cmap, norm_time, "Joint: trajectories")
        draw_flow_main(ax_fj, full_j, xb_j, yb_j, "Joint: flow field", vmax_main(full_j), time_cmap, norm_time, density_cmap=density_cmap, flow_scale=flow_scale)

        draw_flow_mini_arrows(ax_j_tl, j_dLp, "dL>0", j_dl_pos, j_dl_support, j_dl_v, ref_total_j, mini_cmap=mini_cmap, flow_scale=flow_scale_signed)
        draw_flow_mini_arrows(ax_j_bl, j_dLn, "dL<0", j_dl_neg, j_dl_support, j_dl_v, ref_total_j, mini_cmap=mini_cmap, flow_scale=flow_scale_signed)
        draw_flow_mini_arrows(ax_j_tr, j_dHp, "dH>0", j_dh_pos, j_dh_support, j_dh_v, ref_total_j, mini_cmap=mini_cmap, flow_scale=flow_scale_signed)
        draw_flow_mini_arrows(ax_j_br, j_dHn, "dH<0", j_dh_neg, j_dh_support, j_dh_v, ref_total_j, mini_cmap=mini_cmap, flow_scale=flow_scale_signed)

        draw_trajectories(ax_tn, ent_n, logit_n, n_examples, time_cmap, norm_time, "Naive: trajectories")
        draw_flow_main(ax_fn, full_n, xb_n, yb_n, "Naive: flow field", vmax_main(full_n), time_cmap, norm_time, density_cmap=density_cmap, flow_scale=flow_scale)

        draw_flow_mini_arrows(ax_n_tl, n_dLp, "dL>0", n_dl_pos, n_dl_support, n_dl_v, ref_total_n, mini_cmap=mini_cmap, flow_scale=flow_scale_signed)
        draw_flow_mini_arrows(ax_n_bl, n_dLn, "dL<0", n_dl_neg, n_dl_support, n_dl_v, ref_total_n, mini_cmap=mini_cmap, flow_scale=flow_scale_signed)
        draw_flow_mini_arrows(ax_n_tr, n_dHp, "dH>0", n_dh_pos, n_dh_support, n_dh_v, ref_total_n, mini_cmap=mini_cmap, flow_scale=flow_scale_signed)
        draw_flow_mini_arrows(ax_n_br, n_dHn, "dH<0", n_dh_neg, n_dh_support, n_dh_v, ref_total_n, mini_cmap=mini_cmap, flow_scale=flow_scale_signed)

        fig.subplots_adjust(left=0.03, right=0.995, top=0.95, bottom=0.075)
        plt.show()


    """ RUN THE VISUALIZATION """

    plot_joint_naive_4mini(
    self.joint_goal_belief,
    self.naive_goal_belief,
    self.goal_value,
    n_examples=200,
    nx=100,
    ny=100,
    min_count=1,
    min_count_signed=1,
    qlo=0.5,
    qhi=99.5,
    cmap="coolwarm",
    time_cmap="coolwarm",
    mini_cmap="viridis",     # change mini colormap here
    density_cmap="viridis",
    flow_scale=1.0,
    flow_scale_signed=1.0,
    figsize=(22, 20)
)



