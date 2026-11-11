"""Through-training heat maps: for every measure its RAW data (rows 1-2), its CROSS-CORRELATION with entanglement
(rows 3-4) and with FR (rows 5-6), the Trained RNN and the ESN (John 11-09: "one row for trained, one row for ESN"; then
"1 row be the raw data and another row be cross-corrs, but show raw and cross corrs for all columns", "bring back
effective dim"; then "put the credit vs FR into fig_learn maps in a new row where we see everything vs FR").
-> main/REPLICATION/figure_bin/fig_learn_maps.png
Columns (sources and definitions in their docstrings):
  a  state spectrum. Raw: lambda_k / Sigma lambda of u_t's PCA per step (mean over steps), white: its participation
     ratio (fig_td_summary g). Cross-correlation: r at lag 0 between u_t's energy share in each 10%-variance band of
     the checkpoint's own PCA basis (td_analysis band_u: band 1 = the top PCs holding the first 10% of the variance,
     band 10 = the tail) and the entanglement at t (td_xcorr_lags._xc_band; empty when a band is empty early on)
  b  credit by lag. Raw: ||delta_k||^2 / Sigma_k ||delta_k||^2 of the errors at steps 17-30 sent k steps back, per probe,
     then the mean (td_tt_credit_reach). Cross-correlation: for each lag k, r between the share an error at step t sends k
     steps back and the entanglement of the step it lands on, t - k (both z-scored within each episode over the 14 error
     steps; credit_reach/full_<run>_*.npz: does credit sent k steps back land preferentially on entangled steps?)
  c-i  activity norm ||u_t||, effective dimension (projection PR on the checkpoint's basis), activity turn
     1 - cos(u_t, u_t-1), credit coherence, dW energy, cross-band share, dW effective dimension (f-i: the update measures
     of each step's error; the ESN's: what its frozen recurrent weights would receive). Raw: the mean over probes at
     each step (c-e: steps 1-30 of 10,000 probes; f-i: error steps 17-30 of 5,000). Cross-correlation: r with
     entanglement at lag tau (td_xcorr_lags, the raw locks: the measure at t + tau against the entanglement at t, both
     z-scored within each episode over steps 17-30).
  Rows 5-6, FR (td_xcorr_across; FR_t = KL(Bayes_t || naive_t), the paper's forward form, exact, both variables
     averaged; until 11-10 the symmetric clipped KL): the same measures,
     but r ACROSS episodes at each step t, then the mean over steps 17-30 (FR's variance is 67% between episodes and it
     drifts slowly within one, autocorrelation .94 at 1 step, so a within-episode z-score would remove most of it);
     a: band share at t vs FR_t; b: the credit share an error at t sends k steps back vs FR at the landing step t - k;
     c-i: the measure at t + tau vs FR_t.
Colour (John 11-09): raw, every panel on its own scale (its 1st-99th percentile); cross-correlations, one scale for all
panels (-.2 to .2, John 11-09: the credit-by-lag correlations, |r| <= .17, need the range; the strong lag-0 locks
saturate after the crossing, arrows on the bar); bright = more.
a vs d: a is the spectrum of the CLOUD of states (u_t's PCA eigenvalues over all probes at a step: how many directions
the population varies along); d is the spread of ONE state (how many of those principal axes one state's deviation from
the mean loads on, the projection PR, then the mean over probes). For a Gaussian cloud d ~ PR(a) / 3, so the raw maps
move together; their cross-correlations differ: a asks which variance bands gain energy at entangled steps, d whether
an entangled step's state spreads over more axes.
h is negative where the updates sent through different variance bands cancel (sum of the parts' energies > the whole's).
Per-panel scales make near-constant maps look structured: read the bar (e.g. the ESN's i spans 1.006-1.009). Dotted: the Trained RNN passes naive (880 training episodes); strips above the maps: the behavioural stages.
RESULTS (11-09) b raw: effective number of lags per probe Trained RNN 1.8 -> 7.0 (80) -> 7.7 (640) -> 3.1 (end); ESN
  1.8-2.0 throughout (lag 0 holds .67-.71 of the credit).
  b cross-correlation (weak everywhere, |r| <= .17): both networks start at +.05 to +.11 at every lag (the shared initial
  network); Trained RNN at the end -.11 / -.17 at lags 0 / 1 (credit sent 0-1 steps back lands less on entangled steps)
  and +.04 to +.07 at lags 3-15; ESN at the end -.06 to +.03.
  FR rows (re-measured 11-10 with the forward KL; td_xcorr_across; r across episodes, steps 17-30; the colour scale
  saturates for c; in brackets the 11-09 values with the symmetric clipped KL where they moved by > .03): Trained RNN
  c norm -.19 (the initial network) -> +.22 (240) -> +.42 (end), the same at every lag (FR drifts slowly); d effective
  dim -.04 (240) -> -.19 (end) [-.28]; e turn -.13 at most; a top band +.16 (9,600) / +.10 (end) [+.31 / +.24], bands
  2-7 down to -.25; b credit sent 8-15 steps back lands on high-FR steps from 160 to 1,600 (+.14 to +.21 on average,
  +.23 at lag 14 at 320) while lags 1-3 avoid them (-.24 at 320); after ~3,000 lags 2-4 +.10 to +.20 and lag 0 -.22;
  f-i |r| <= .23 anywhere, 95% of the entries <= .14. ESN: norm -.17 to -.20, effective dim +.08, turn +.10, dW dim
  +.12 throughout; b lag 0 +.11 (640) -> +.24 (10k) and every other lag -.17 to -.25; f credit coherence down to -.27
  late. The reverse term of the symmetric KL is what tied FR to the top variance band and to the effective dimension
  as strongly as the 11-09 numbers said; norm and credit by lag are the same under both.
CAUSE OR CONSEQUENCE (John 11-09: "which of them are causal and which of them are consequences? ... learning or
inference?"). Lag-0 r with entanglement at 10 (the shared initial network) / 880 / 2,400 / 9,600 / end:
  Trained RNN  norm -.16 / .01 / .04 / .27 / .36;  turn .31 / .13 / .13 / .19 / .27;  dW dim .38 / .00 / -.02 / -.14 /
               -.16;  credit coherence -.06 / .00 / .02 / .22 / .34;  dW energy -.09 / .00 / .01 / .12 / .18;  cross-band
               .00 / .00 / .03 / .19 / .28;  effective dim .01 / .09 / .14 / .19 / .25
  ESN          norm -.16 -> -.22;  turn .31 -> .32;  dW dim .38 -> .40;  the others within +-.05 throughout
  -> the turn and dW-dimension locks are the INPUT's (the untrained network has them, the frozen ESN keeps them). By
  880 the Trained RNN has LOST them and the input's negative norm lock (-.16 -> .01), and its effective-dimension lock
  has started (.09); its other lag-0 locks grow after the crossing. Timing alone cannot decide cause (John 11-09:
  "'losing' a lock when crossing naive could also be a meaningful change that requires 'undo-ing' input statistics";
  "locks can continue growing after crossing naive, which could hide their importance for the initial crossing").
  Inference test, read at the manipulated step (td_tt_dose.py, learn_dose_pilot.png; one step per episode, entangled
  vs random steps, 12 checkpoints): until the crossing the network's dependence on its norm is the same at entangled
  and random steps (its steps are too large everywhere: halving one LOWERS the error); entangled steps need their extra
  norm only from ~2,400 (halving: +.020 vs +.014 nats; end +.028 vs +.015), when the norm lock forms. More turn lowers
  the error at a step from 200 to ~2,400, more at entangled steps (doubling, 880: -.021 vs -.016): the lost turn lock
  was worth keeping, and the loss pushes it back. Manipulating the state 2 steps after an entangled step = a random
  step: the after-step response is not used at inference. All effects at the step are small in accuracy (<= 3 points).
  After ~2k the Trained RNN corrects a perturbed step (24% of the effect left at T at the end; ESN never), so reading
  at T (the earlier clamp: KL_T .0152 -> .0179 vs .0158, accuracy -.002) hid most of it.
  Every activity contender (td_tt_dose second pass, learn_dose_contenders.png: norm, turn, effective dimension and
  top-band share halved / doubled at one entangled, high-FR or random step): all effects at the step are small (a few %
  of the step's error before ~2k, accuracy <= 1.2 points); entangled-specific parts <= .01 nats except the turn; high-FR
  steps are the most sensitive to norm, dimension and band, and so are the ESN's (not learned). Learning side: the
  update measures (f-i) belong to the errors at entangled steps; dropping those errors' loss terms (train_learn_masks
  mask_hi, 20% per step) or detaching the updates made there (det_hi) leaves learning unchanged (accuracy at 1,600 raw
  .610 / .605 vs control .614, random .614 / .604), so whatever those updates' structure, they are not needed; blocking
  the high-FR half from scratch is the one strong lever found (.554 vs .653 random at 2,000 raw; replicate .580 vs
  .644). train_learn_causal (cut / boost / norm / turn / dim at entangled, high-FR or random steps) is ready, not run.
Run: MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig_learn_maps.py"""
import os
import sys
import glob
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.colors import LogNorm, Normalize                                  # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import td_xcorr_lags as xl                                                        # noqa: E402
import td_xcorr_across as xa                                                      # noqa: E402
import td_tt_credit_spread as cs                                                  # noqa: E402
import td_tt_credit_reach as cr                                                   # noqa: E402
import fig_td_summary as fts                                                      # noqa: E402
import stage_bands as sb                                                          # noqa: E402
import td_summary_data as sd                                                      # noqa: E402
import gradflow_analysis as ga                                                    # noqa: E402

RUNS = ("trained", "echo")
COL = {"trained": "#1f77b4", "echo": "#ff7f0e"}
NAME = {"trained": "Trained RNN", "echo": "ESN"}
TS, FL, FT = 12, 11, 10
AN = xl.AN
# (raw key in raw_data, xcorr key in td_xcorr_lags.data, group, title, formula, raw colour scale)
MEAS = (("norm_u", ("act", 0), "Activity", "Activity norm", r"$\Vert u_t\Vert_2$", "lin"),
        ("effdim_u", ("act", 1), "Activity", "Effective dim.", r"$(\Sigma_{pc} proj^2)^2/\Sigma_{pc} proj^4$", "lin"),
        ("turn_u", ("act", 2), "Activity", "Activity turn", r"$1-\cos(u_t,u_{t-1})$", "lin"),
        ("cr_coh", ("upd", 0), "Update", "Credit coherence", r"$\Vert\Sigma_k\delta_k\Vert^2/\Sigma_k\Vert\delta_k\Vert^2$",
         "lin"),
        ("cr_E", ("upd", 1), "Update", r"$\Delta W$ energy", r"$\Vert\Delta W\Vert^2$", "log"),
        ("cr_bandx", ("upd", 3), "Update", "Cross-band share", r"$1-\Sigma_B\Vert\Delta W_B\Vert^2/\Vert\Delta W\Vert^2$",
         "lin"),
        ("cr_dim", ("upd", 2), "Update", r"$\Delta W$ effective dim.", r"$(\Sigma_i\sigma_i^2)^2/\Sigma_i\sigma_i^4$",
         "lin"))
PW, PH, GX, GY, LM, RM = 1.8, 1.45, .62, .55, 1.6, .3


def raw_data(run):
    """per checkpoint: the per-step means of the activity (30 steps) and update (14 error steps) measures, and the
    lag-0 r of u's band energy shares with entanglement (10 bands); cached in analysis/learn_maps_raw_<run>.npz"""
    cache = os.path.join(AN, f"learn_maps_raw_{run}.npz")
    files = sorted(glob.glob(os.path.join(AN, run, "ck_*.npz")))
    eps = np.array([int(os.path.basename(f)[3:8]) for f in files])
    if os.path.exists(cache):
        z = dict(np.load(cache))
        if np.array_equal(z["episode"], eps):
            return z
    P = np.load(os.path.join(AN, "probes.npz"))
    rev = P["ent"].astype(np.float64).mean(-1)
    out = {k: [] for k, *_ in MEAS}
    out["band_r"] = []
    i0 = int(np.flatnonzero(xl.TAU == 0)[0])
    for f in files:
        z = np.load(f)
        for k, *_ in MEAS:
            out[k].append(np.nanmean(z[k].astype(np.float64), 0))
        b = z["band_u"]
        out["band_r"].append([xl._xc_band(b[..., j], rev)[i0] for j in range(b.shape[-1])])
    out = {k: np.array(v) for k, v in out.items()}
    out["episode"], out["x"] = eps, np.maximum(eps - eps // 5, 10.).astype(float)       # td_tt_gap.train_x
    out["cr_steps"] = np.load(files[0])["cr_steps"]
    np.savez(cache, **out)
    return out


def credit_xcorr(run):
    """lag k x checkpoint: r between the credit share an error at step t sends k steps back and the entanglement of the
    landing step t - k (z-scored within each episode over the 14 error steps; the probes' mean entanglement over both
    variables); from credit_reach/full_<run>_<raw>.npz"""
    files = sorted(glob.glob(os.path.join(cr.OUT, f"full_{run}_*.npz")))
    if not files:
        return None, None
    P = np.load(os.path.join(AN, "probes.npz"))
    rev = P["ent"].astype(np.float64).mean(-1)                                  # (n_all, T)
    xs, R = [], []
    for f in files:
        z = np.load(f)
        E, rows = z["E"].astype(np.float64), z["rows"]                          # (n, 14, 16)
        sh = E / E.sum(-1, keepdims=True)
        t_idx = np.asarray(cr.SRC)                                              # 0-based error steps (16..29)
        r = []
        for k in range(sh.shape[-1]):
            a = ga._zs(sh[:, :, k])
            b = ga._zs(rev[rows][:, t_idx - k])
            r.append((a * b).mean())
        xs.append(float(z["x"]))
        R.append(r)
    o = np.argsort(xs)
    return np.array(xs)[o], np.array(R)[o]                                      # (C,), (C, 16)


XNORM = Normalize(-.2, .2)          # every cross-correlation panel (John 11-09: "all cbars use the same range"; then
                                    # "+/- .3", then "+/-.2" for the credit-by-lag structure; stronger values saturate)


def own_norm(v, log=False):
    """a raw panel's own colour scale (John 11-09: "each panel's cbar normalized separately"): its 1st-99th percentile"""
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & ((v > 0) if log else True)]
    lo, hi = np.percentile(v, [1, 99])
    return LogNorm(lo, hi) if log else Normalize(lo, hi)


def figure():
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({"xtick.labelsize": FT, "ytick.labelsize": FT})
    D, _, cross = xl.data()
    RD = {r: raw_data(r) for r in RUNS}
    nc = 2 + len(MEAS)
    W = LM + nc * PW + (nc - 1) * GX + RM
    GR = .78                                                                     # under each raw row: its colour bars
    tops = [1.2, 1.2 + PH + GR]
    tops += [tops[1] + PH + GR + .95, tops[1] + PH + GR + .95 + PH + GY]
    tops += [tops[3] + PH + 1.05, tops[3] + PH + 1.05 + PH + GY]                 # rows 5-6: FR
    H = tops[5] + PH + 1.95
    fig = plt.figure(figsize=(W, H))

    def add(j, top):
        return fig.add_axes([(LM + j * (PW + GX)) / W, (H - top - PH) / H, PW / W, PH / H])
    ax = np.array([[add(j, t) for j in range(nc)] for t in tops])               # (6, nc): raw, x entanglement, x FR

    def cbar(a, nrm, top, lab=None):
        """a colour bar under one raw panel"""
        x0 = LM + list(ax[0 if top == tops[0] else 1]).index(a) * (PW + GX) + .12
        cax = fig.add_axes([x0 / W, (H - top - PH - .36) / H, (PW - .24) / W, .07 / H])
        cbr = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap="viridis"), cax=cax, orientation="horizontal")
        cbr.ax.tick_params(labelsize=FT - 2, length=2, pad=1)
        cbr.ax.xaxis.set_major_locator(plt.MaxNLocator(3)) if not isinstance(nrm, LogNorm) else None
        cbr.outline.set_linewidth(.5)
        if lab:
            cbr.set_label(lab, fontsize=FT - 1.5, labelpad=1)

    # a raw: the state spectrum (fig_td_summary g's own code), then each panel on its own scale
    fts.g_spectrum([ax[0, 0], ax[1, 0]])
    for i, a in enumerate(ax[:2, 0]):
        a.set_title("")
        xs_ = sd.state_spectrum_steps(RUNS[i])["x"]
        xl.xaxis(a, xl.log_edges(np.asarray(xs_, float)), label=False)
        for t in [t for t in a.texts if t.get_text().startswith("PR:")]:
            v = t.get_text().replace("PR: ", "").replace(" at the start, ", r" $\rightarrow$ ").replace(" at the end", "")
            t.set_text("PR " + v)
        qm = a.collections[0]
        nrm = own_norm(qm.get_array(), log=True)
        qm.set_norm(nrm)
        cbar(a, nrm, tops[i])
    # a xcorr: band energy shares x entanglement
    for i, r in enumerate(RUNS):
        d = RD[r]
        a = ax[2 + i, 0]
        a.pcolormesh(xl.log_edges(d["x"]), np.arange(.5, 11), np.ma.masked_invalid(d["band_r"]).T, cmap="viridis",
                     norm=XNORM, shading="flat", rasterized=True)
        a.set_ylim(10.5, .5)
        a.set_yticks([1, 5, 10])
        a.set_ylabel("variance band (1 = top PCs)", fontsize=FT, labelpad=1)
        xl.xaxis(a, xl.log_edges(d["x"]), label=i == 1)
    # b raw: credit share by lag; b xcorr: credit landing x entanglement
    for i, r in enumerate(RUNS):
        x, S, _ = cs.load(r)
        a = ax[i, 1]
        nrm = own_norm(S.mean(1), log=True)
        a.pcolormesh(xl.log_edges(x), np.arange(cs.K + 1) - .5, S.mean(1).T, cmap="viridis", norm=nrm,
                     shading="flat", rasterized=True)
        cbar(a, nrm, tops[i])
        xc, Rc = credit_xcorr(r)
        a2 = ax[2 + i, 1]
        if xc is not None and len(xc) > 1:
            a2.pcolormesh(xl.log_edges(xc), np.arange(cs.K + 1) - .5, Rc.T, cmap="viridis", norm=XNORM, shading="flat",
                          rasterized=True)
        for aa in (a, a2):
            aa.set_yticks([0, 5, 10, 15])
            aa.set_ylabel("lag $k$ (steps back)", fontsize=FT, labelpad=1)
            xl.xaxis(aa, xl.log_edges(x), label=False)
    # c-i raw: per-step means, each panel on its own scale; c-i xcorr: the entanglement locks
    te = np.concatenate([xl.TAU - .5, [xl.TAU[-1] + .5]])
    for j, (rk, xk, grp, title, form, sc) in enumerate(MEAS):
        c = 2 + j
        upd = rk.startswith("cr_")
        steps = (np.asarray(RD["trained"]["cr_steps"]) + 1) if upd else np.arange(1, 31)
        for i, r in enumerate(RUNS):
            d = RD[r]
            a = ax[i, c]
            ye = np.concatenate([steps - .5, [steps[-1] + .5]])
            nrm = own_norm(d[rk], log=sc == "log")
            a.pcolormesh(xl.log_edges(d["x"]), ye, d[rk].T, cmap="viridis", norm=nrm, shading="flat", rasterized=True)
            cbar(a, nrm, tops[i])
            a.set_ylabel("step $t$", fontsize=FT, labelpad=1) if j == 0 else None
            xl.xaxis(a, xl.log_edges(d["x"]), label=False)
            dx = D[r]
            a2 = ax[2 + i, c]
            a2.pcolormesh(xl.log_edges(dx["x"]), te, np.ma.masked_invalid(dx[xk]).T, cmap="viridis", norm=XNORM,
                          shading="flat", rasterized=True)
            a2.set_yticks([-8, -4, 0, 4, 8])
            if j == 0:
                a2.set_ylabel(r"lag $\tau$ (steps after)", fontsize=FT, labelpad=1)
            xl.xaxis(a2, xl.log_edges(dx["x"]), label=False)
        ax[0, c].set_title(f"{chr(ord('c') + j)}   {title}\n{form}", loc="left", fontsize=TS, pad=10, linespacing=1.4)
    # rows 5-6: every measure against FR, r across episodes at each step (td_xcorr_across)
    for i, r in enumerate(RUNS):
        d = xa.data(r)
        a = ax[4 + i, 0]
        a.pcolormesh(xl.log_edges(d["x"]), np.arange(.5, 11), np.ma.masked_invalid(d["band_fr"]).T, cmap="viridis",
                     norm=XNORM, shading="flat", rasterized=True)
        a.set_ylim(10.5, .5)
        a.set_yticks([1, 5, 10])
        a.set_ylabel("variance band (1 = top PCs)", fontsize=FT, labelpad=1)
        xl.xaxis(a, xl.log_edges(d["x"]), label=False)
        a = ax[4 + i, 1]
        a.pcolormesh(xl.log_edges(d["credit_x_fr"]), np.arange(cs.K + 1) - .5, d["credit_fr"].T, cmap="viridis",
                     norm=XNORM, shading="flat", rasterized=True)
        a.set_yticks([0, 5, 10, 15])
        a.set_ylabel("lag $k$ (steps back)", fontsize=FT, labelpad=1)
        xl.xaxis(a, xl.log_edges(d["credit_x_fr"]), label=False)
        a.set_xlim(ax[2 + i, 1].get_xlim())
        for j, (rk, xk, *_) in enumerate(MEAS):
            a = ax[4 + i, 2 + j]
            v = d["act_fr" if xk[0] == "act" else "upd_fr"][:, xk[1]]
            a.pcolormesh(xl.log_edges(d["x"]), te, np.ma.masked_invalid(v).T, cmap="viridis", norm=XNORM,
                         shading="flat", rasterized=True)
            a.set_yticks([-8, -4, 0, 4, 8])
            if j == 0:
                a.set_ylabel(r"lag $\tau$ (steps after)", fontsize=FT, labelpad=1)
            xl.xaxis(a, xl.log_edges(d["x"]), label=False)
    # one colour bar for every cross-correlation panel
    x0 = LM + 3 * (PW + GX)
    cax = fig.add_axes([x0 / W, (H - tops[5] - PH - .72) / H, (3 * PW + 2 * GX) / W, .1 / H])
    cbr = fig.colorbar(plt.cm.ScalarMappable(norm=XNORM, cmap="viridis"), cax=cax, orientation="horizontal",
                       extend="both")
    cbr.ax.tick_params(labelsize=FT, length=2, pad=1)
    cbr.outline.set_linewidth(.5)
    cbr.set_label("$r$ with entanglement (rows 3-4) or with FR (rows 5-6)", fontsize=FT, labelpad=2)
    ax[0, 0].set_title("a   State spectrum\n" r"$\lambda_k/\Sigma\lambda$ of $u_t$; white: PR", loc="left", fontsize=TS,
                       pad=10, linespacing=1.4)
    ax[0, 1].set_title("b   Credit by lag\n" r"$\Vert\delta_k\Vert^2/\Sigma_k\Vert\delta_k\Vert^2$", loc="left",
                       fontsize=TS, pad=10, linespacing=1.4)
    for i in range(len(tops)):
        for j in range(nc):
            a = ax[i, j]
            a.axvline(cross, color="w", lw=.9, ls=":", zorder=5)
            sb.strip(a)
            a.set_xlabel("training episodes", fontsize=FT, labelpad=1) if i == len(tops) - 1 else a.set_xlabel("")
        fig.text(.18 / W, (H - tops[i] - PH / 2) / H, NAME[RUNS[i % 2]], rotation=90, ha="center", va="center",
                 fontsize=TS, color=COL[RUNS[i % 2]])
    fig.text(LM / W, (H - tops[0] + .75) / H, "RAW DATA (each panel on its own colour scale)", fontsize=TS, ha="left",
             va="bottom", color="0.25")
    fig.text(LM / W, (H - tops[2] + .4) / H, "CROSS-CORRELATION WITH ENTANGLEMENT (z-scored within each episode; one "
             "colour scale)", fontsize=TS, ha="left", va="bottom", color="0.25")
    fig.text(LM / W, (H - tops[4] + .4) / H, "CROSS-CORRELATION WITH FR (r across episodes at each step: FR varies "
             "mostly between episodes; same colour scale)", fontsize=TS, ha="left", va="bottom", color="0.25")
    fig.text(.5, .12 / H, "a: 10,000 probes, 114 checkpoints;  b: 2,000 probes, errors at steps 17-30, 32 checkpoints;  "
             "c-e: 10,000 probes;  f-i: 5,000 probes, error steps 17-30;  cross-correlations: steps 17-30, 114 "
             "checkpoints.  ESN b, f-i: what its frozen recurrent weights would receive.  Dotted: the Trained RNN "
             "passes naive (880).\nRows 3-4: the measure and entanglement z-scored within each episode over steps 17-30."
             "  Rows 5-6: r over the probes at each step, then the mean over steps 17-30 (b: FR at the step the credit "
             "lands on, t - k).", ha="center", va="bottom", fontsize=FT - 1, color="0.3", linespacing=1.5)
    out = fig_path("fig_learn_maps.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    figure()
