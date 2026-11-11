"""Short-term (u = h, 'stm') vs long-term (c, 'ltm') stream: where does the step expansion at entangled observations live?
(John 11-08: "please also check if the chaotic dynamics differs between the short and long term stream, it would be good
if we could stick to short term memory dynamics".)
-> main/REPLICATION/figure_bin/td_tt_chaos_streams.png   figure()
   numbers()                                               the values quoted below
   harvest [gpu]                                           caches main/DATA/experimental/training_dynamics/
                                                           through_training/chaos_streams/<run>_EEEEE.npz
Same probes, entanglement fifths, flips and Arnoldi as td_tt_chaos_ent (1,000 probes = 20 per test environment; units =
(probe, step t); fifths of observation t's entanglement ranked within each step; 5-95% over the 1,000 shared environment
resamples). Measures per unit, at the intact states:
  full   log rho(J_t), J_t = d(u_t, c_t) / d(u_{t-1}, c_{t-1}) (td_tt_chaos_ent's cached ext_EEEEE.npz)
  u      log rho(J_uu), J_uu = d u_t / d u_{t-1} with c_{t-1} held: the short-term loop (u -> gates -> c_t -> u_t) without
         the carried cell state; Arnoldi as td_tt_chaos_ent.arnoldi_rho on the 1,000-dim map
  c      log rho(J_cc) = log max_i f_i,t: with u_{t-1} held, c_t = f_t * c_{t-1} + (terms of u_{t-1}, x_t), so J_cc =
         diag(f_t) exactly (no peepholes): the long-term stream alone always contracts (f < 1)
  realised growth per stream: one flipped observation (steps 1, 3, 5, 10; each of the 5 observations), the network rerun
         in difference form (td_tt_chaos_ent.flips' identities); g_u = log ||du_t|| - log ||du_{t-1}||, g_c likewise
         for dc; unit value = the mean over its flips at lag 1 / lags 1-4 / lags 5-20 (lag = t - flip step)
Structure (exact): J = [[J_uu, E D], [C, D]] with D = diag(f_t), E = diag(o_t (1 - tanh^2 c_t)), C = d c_t / d u_{t-1}.
RESULTS 11-08 (Trained RNN, 16 checkpoints; [5-95%]; nats / step; at 880 = passing naive unless named)
  The expansion at entangled observations is a property of the coupled step, not of either stream:
    full    most entangled fifth > 0 at 560-1,360 (+.0123 [+.0088, +.0157] at 880; contrast +.034), all steps < 0
            (-.004); the recomputed full log rho equals the cached td_tt_chaos_ent values exactly (max |d| 0 at 800)
    u       contracts strongly at every checkpoint: -.237 [-.244, -.229] (all), at most 2.8% of steps > 0 (800); no
            entanglement contrast at 560-2,000 (-.006 [-.018, +.006] at 880); late, entangled steps contract less (end
            -.227 vs -.299, contrast +.072)
    c       log max f < 0 at every step (-.038 at 880; -.006 at the end): some units' forget gates sit at ~1 from ~500
            episodes on (an integrator), never above
  Realised growth of a flip's difference: on the step after the flip the cell state grows from 720 episodes on (+.097
    [+.088, +.105] at 880, peak +.134 at 1,200), the short-term state only from 1,040 and less (+.002 [-.009, +.011] at
    880, +.064 at 1,200); over lags 1-4 and 5-20 both shrink at every checkpoint (u -.146 / -.146 at 880, end -.222 /
    -.067), less at entangled observations (u contrast +.040 / +.022 at 880).
  So the expansion needs the short-term loop feeding the cell state's near-unit carry; the short-term stream's own
  dynamics are contraction, slower at entangled observations.
  ESN (4 checkpoints: 0, 880, 4,800, end): everything contracts and nothing is learned in either stream. u alone stays
    at -1.93 (no contrast); c alone -.48 -> -.34 (its largest forget gate ~.85-.89: no integrator units); the full step
    -.42 -> -.34, the most entangled steps contracting slightly MORE (contrast -.005 to -.013); a flip's difference
    shrinks by ~.65 nats on the next step in both streams and ~.41-.46 per step at lags 5-20. Long-lag flip effects (td_tt_flips' lags 18-21) can only
  be carried by c (u alone decays ~.24 nats / step: e^-4.8 over 20 steps).
Run from the repository root:
    /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_chaos_streams.py harvest 1
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_chaos_streams.py"""
import os
import sys
import time
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_chaos as tc                                                                      # noqa: E402
import td_tt_chaos_ent as ce                                                                  # noqa: E402

T, NO = ce.T, ce.NO
OUT = os.path.join(gap.TT, "chaos_streams")
CKPTS = {"trained": (0, 125, 300, 500, 700, 800, 900, 1000, 1100, 1300, 1500, 1700, 2500, 6000, 15000, 49995),
         "echo": (0, 1100, 6000, 49995)}
LAGS = {"1": (1, 1), "1-4": (1, 4), "5-20": (5, 20)}
MEAS = ("full", "u", "c", "full_pos", "u_pos") + tuple(f"g{s}:{k}" for s in "uc" for k in LAGS)
CHUNK, SEED, GPU_CAP_GB = 100, 20261108, 6


# ── harvest (GPU) ───────────────────────────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def lrho_u(p, A, C, seed):
    """log rho(J_uu) (b, T) and the relative residual (b, T): the u -> u block of the step Jacobian, c_{t-1} held"""
    Whh = p["LSTM.weight_hh_l0"]
    H = Whh.shape[1]
    dev = Whh.device
    b = A.shape[0]
    N = b * T
    gall = tuple(x.reshape(N, H) for x in ce._gates(A)) + (torch.tanh(C[:, 1:]).reshape(N, H),)
    call = C[:, :-1].reshape(N, H)

    def mv(v, ids):
        it = torch.as_tensor(ids, device=dev)
        dh, _ = tc.tangent(v[:, None], torch.zeros_like(v[:, None]), call[it], tuple(x[it] for x in gall), Whh)
        return dh[:, 0]
    th, res, _, _, _ = ce.arnoldi_rho(mv, N, H, dev, seed)
    return np.log(np.abs(th)).reshape(b, T), res.reshape(b, T)


@torch.no_grad()
def flips_streams(p, table, envt, code, A, C):
    """td_tt_chaos_ent.flips' state part, per stream: log ||du_t||, log ||dc_t|| (b, nS, 5, T) float64"""
    Whh = p["LSTM.weight_hh_l0"]
    H = Whh.shape[1]
    b, dev = len(envt), Whh.device
    nS = len(ce.FLIP_S)
    lu = torch.full((b, nS, NO, T), float("nan"), device=dev, dtype=torch.float64)
    lc = torch.full((b, nS, NO, T), float("nan"), device=dev, dtype=torch.float64)
    bits = torch.as_tensor(1 << np.arange(NO), device=dev).repeat(b)
    eb = envt.repeat_interleave(NO)
    for k, s in enumerate(ce.FLIP_S):
        cd = code[:, s - 1].repeat_interleave(NO)
        dA = table[eb, cd ^ bits] - table[eb, cd]
        dh = torch.zeros(b * NO, H, device=dev, dtype=Whh.dtype)
        dc = torch.zeros(b * NO, H, device=dev, dtype=Whh.dtype)
        for t in range(s - 1, T):
            if t > s - 1:
                dA = dh @ Whh.T
            a = A[:, t].repeat_interleave(NO, 0)
            i, f, g, o = ce._gates(a)
            di, df, dg, do = dA.chunk(4, -1)
            ai, af, ag, ao = a.chunk(4, -1)
            Di, Df, Dg, Do = ce.dsig(ai, di), ce.dsig(af, df), ce.dtanh(ag, dg), ce.dsig(ao, do)
            cp = C[:, t].repeat_interleave(NO, 0)
            cn = C[:, t + 1].repeat_interleave(NO, 0)
            dc = Df * cp + (f + Df) * dc + Di * (g + Dg) + i * Dg
            dh = Do * torch.tanh(cn + dc) + o * ce.dtanh(cn, dc)
            lu[:, k, :, t] = torch.log(dh.double().norm(dim=-1)).reshape(b, NO)
            lc[:, k, :, t] = torch.log(dc.double().norm(dim=-1)).reshape(b, NO)
    return lu, lc


def growth_units(logn):
    """per unit (probe, step t) the mean one-step growth log n_t - log n_{t-1} over the flips whose lag t - (s - 1)
    falls in each lag group: {group: (n, T)} (NaN where no flip reaches)"""
    n = logn.shape[0]
    out = {}
    for name, (l0, l1) in LAGS.items():
        acc, cnt = np.zeros((n, T)), np.zeros((n, T))
        for k, s in enumerate(ce.FLIP_S):
            for t in range(s, T):                                          # 0-indexed step t; flip at s - 1
                lag = t - (s - 1)
                if l0 <= lag <= l1:
                    g = logn[:, k, :, t] - logn[:, k, :, t - 1]            # (n, 5)
                    ok = np.isfinite(g)
                    acc[:, t] += np.where(ok, g, 0.).sum(-1)
                    cnt[:, t] += ok.sum(-1)
        out[name] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return out


def ck_path(run, raw):
    return os.path.join(OUT, f"{run}_{int(raw):05d}.npz")


@torch.no_grad()
def harvest_one(run, raw, dev, check_full=False):
    rws = ce.rows()
    p, table = ce.setup(run, raw, dev)
    envt, code = ce.codes(rws, dev)
    n = len(rws)
    out = dict(lr_u=np.zeros((n, T)), lr_u_res=np.zeros((n, T), np.float32), lr_c=np.zeros((n, T)),
               lu=np.zeros((n, len(ce.FLIP_S), NO, T)), lc=np.zeros((n, len(ce.FLIP_S), NO, T)))
    if check_full:
        out["lr_full_check"] = np.zeros((n, T))
    for b0 in range(0, n, CHUNK):
        sl = slice(b0, min(n, b0 + CHUNK))
        A, U, C, lg = ce.replay(p, table, envt[sl], code[sl])
        out["lr_u"][sl], out["lr_u_res"][sl] = lrho_u(p, A, C, SEED + b0)
        f = ce._gates(A)[1]                                                    # (b, T, H) forget gates
        out["lr_c"][sl] = torch.log(f.amax(-1)).double().cpu().numpy()
        lu, lc = flips_streams(p, table, envt[sl], code[sl], A, C)
        out["lu"][sl], out["lc"][sl] = lu.cpu().numpy(), lc.cpu().numpy()
        if check_full:
            out["lr_full_check"][sl] = ce.lrho_units(p, A, C, ce.LR_SEED + b0)[0]
        del A, U, C, lg
    out["lu"], out["lc"] = out["lu"].astype(np.float32), out["lc"].astype(np.float32)
    return out


def harvest(gpu=1):
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_per_process_memory_fraction(
        min(1., GPU_CAP_GB * 2 ** 30 / torch.cuda.get_device_properties(dev).total_memory), dev)
    for run, eps in CKPTS.items():
        for raw in eps:
            path = ck_path(run, raw)
            if os.path.exists(path):
                continue
            t0 = time.time()
            o = harvest_one(run, raw, dev, check_full=(run == "trained" and raw == 1000))
            np.savez(path + ".tmp.npz", **o)
            os.replace(path + ".tmp.npz", path)
            msg = (f"{run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s; log rho u {o['lr_u'].mean():+.4f}"
                   f" (> 0 {(o['lr_u'] > 0).mean():.3f}, res max {o['lr_u_res'].max():.1e}); c {o['lr_c'].mean():+.4f}"
                   f" (max {o['lr_c'].max():+.4f})")
            if "lr_full_check" in o:
                ext = np.load(ce.ext_file(run, raw))["lrho"]
                msg += f"; full vs cache max |d| {np.abs(o['lr_full_check'] - ext).max():.2e}"
            print(msg, flush=True)


# ── statistics ──────────────────────────────────────────────────────────────────────────────────────────────────────

def unit_measures(run, raw):
    z = np.load(ck_path(run, raw))
    full = np.load(ce.ext_file(run, raw))["lrho"]
    U = {"full": full, "u": z["lr_u"], "c": z["lr_c"], "full_pos": (full > 0).astype(float),
         "u_pos": (z["lr_u"] > 0).astype(float)}
    for s, key in (("u", "lu"), ("c", "lc")):
        for k, v in growth_units(z[key].astype(np.float64)).items():
            U[f"g{s}:{k}"] = v
    return U


def series(run):
    """x (K,), and per measure the full-sample and resampled (1,000) means of the least / most entangled fifth, all
    steps, and the contrast most - least: dict measure -> {"least", "most", "all", "contrast"} (full (K,), boot (B, K))"""
    cov = ce.covariates()
    q = ce.fifths(cov["ent"])
    env = np.broadcast_to(cov["env"][:, None], q.shape)
    Cb = gap.boot_counts()
    eps = [e for e in CKPTS[run] if os.path.exists(ck_path(run, e))]
    K = len(eps)
    out = {m: {k: [np.zeros(K), np.zeros((len(Cb), K))] for k in ("least", "most", "all", "contrast")} for m in MEAS}
    for j, raw in enumerate(eps):
        U = unit_measures(run, raw)
        for m in MEAS:
            v = U[m]
            ok = np.isfinite(v)
            S, C = np.zeros((gap.N_ENV, ce.NQ)), np.zeros((gap.N_ENV, ce.NQ))
            np.add.at(S, (env[ok], q[ok]), v[ok])
            np.add.at(C, (env[ok], q[ok]), 1.)
            for name, num, den in (("least", S[:, 0], C[:, 0]), ("most", S[:, -1], C[:, -1]),
                                   ("all", S.sum(1), C.sum(1))):
                out[m][name][0][j] = num.sum() / den.sum()
                out[m][name][1][:, j] = (Cb @ num) / (Cb @ den)
            out[m]["contrast"][0][j] = out[m]["most"][0][j] - out[m]["least"][0][j]
            out[m]["contrast"][1][:, j] = out[m]["most"][1][:, j] - out[m]["least"][1][:, j]
    return gap.train_x(eps), out


def numbers():
    for run in CKPTS:
        x, S = series(run)
        print(f"===== {run}")
        for m in MEAS:
            print(f"--- {m}")
            for name in ("all", "least", "most", "contrast"):
                full, boot = S[m][name]
                lo, hi = np.percentile(boot, [5, 95], 0)
                print(f"  {name:9s} " + "  ".join(f"{xv:,.0f}:{f:+.4f}[{a:+.4f},{b:+.4f}]"
                                                   for xv, f, a, b in zip(x, full, lo, hi)))


# ── figure ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def figure():
    """td_tt_chaos_streams.png: a-c the full step at every checkpoint (Trained RNN and ESN), d-e the streams at the 16
    checkpoints of this module, f the tracking of factorized-state Bayes's expansion"""
    import matplotlib.pyplot as plt
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    xs, St = series("trained")
    F = {r: full_series(r) for r in ("trained", "echo")}
    A = {r: adf_tracking(r) for r in ("trained", "echo")}
    cross = 880                             # the Trained RNN's accuracy first exceeds naive's (fd.passes_naive)
    blue, orange, purple = "#1f77b4", "#ff7f0e", "#9467bd"
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.6))
    fig.subplots_adjust(left=.06, right=.99, top=.9, bottom=.16, wspace=.3, hspace=.62)
    ax = ax.ravel()

    def band(a, x, full, boot, color, ls="-", label=None, lw=1.8, ms=0):
        lo, hi = np.percentile(boot, [5, 95], 0)
        a.fill_between(x, lo, hi, color=color, alpha=.18, lw=0)
        a.plot(x, full, color=color, ls=ls, lw=lw, marker="o" if ms else None, ms=ms, label=label)

    def head(a, text):
        a.set_title(text, loc="left", fontsize=12.5, linespacing=1.35)

    for r, c_ in (("trained", blue), ("echo", orange)):
        band(ax[0], F[r]["x"], F[r]["all"], F[r]["all_boot"], c_, label={"trained": "Trained RNN", "echo": "ESN"}[r])
    head(ax[0], "a   Edge of stability\n" + r"$\log\rho(J_t)$ of the full step $(u, c)$, all steps")
    ax[0].set_ylabel("nats / step")
    ax[0].text(150, -.16, "Bayes, naive: exactly 0 (their log-\nbelief dynamics are translations)", fontsize=9.5,
               color=".35", va="center")
    for name, ls, l_ in (("most", "-", "most entangled fifth"), ("all", (0, (4, 2)), "all steps"),
                         ("least", (0, (1, 1.5)), "least entangled fifth")):
        band(ax[1], F["trained"]["x"], F["trained"][name], F["trained"][name + "_boot"], blue, ls, l_)
    ax[1].set_ylim(-.035, .025)
    head(ax[1], "b   Trained RNN, zoom: entangled steps cross\n" + r"$\log\rho(J_t)$ by entanglement fifth; $>0$: "
         "the step map expands")
    band(ax[2], F["trained"]["x"], F["trained"]["pos_most"], F["trained"]["pos_most_boot"], blue, "-",
         "most entangled fifth")
    band(ax[2], F["trained"]["x"], F["trained"]["pos_least"], F["trained"]["pos_least_boot"], blue, (0, (1, 1.5)),
         "least entangled fifth")
    band(ax[2], F["echo"]["x"], F["echo"]["pos_most"], F["echo"]["pos_most_boot"], orange, "-", "ESN (both)")
    head(ax[2], "c   Share of steps that expand\n" + r"$\log\rho(J_t)>0$, Trained RNN by fifth; ESN")
    ax[2].set_ylabel("share of steps")
    col = {"full": "k", "u": blue, "c": purple}
    lab = {"full": r"full step $(u, c)$", "u": r"short-term stream alone: $J_{uu}$", "c": r"long-term stream alone: "
           r"$J_{cc}=\mathrm{diag}(f_t)$"}
    for m in ("full", "u", "c"):
        band(ax[3], xs, St[m]["all"][0], St[m]["all"][1], col[m], label=lab[m], ms=3)
    head(ax[3], "d   Neither stream alone is at the edge\n" + r"$\log\rho$ of each block of $J_t$, all steps "
         "(Trained RNN)")
    ax[3].set_ylabel("nats / step")
    for s_, c_ in (("u", blue), ("c", purple)):
        band(ax[4], xs, *St[f"g{s_}:1"]["all"], c_, "-", f"{s_}: the step after the flip", ms=3)
        band(ax[4], xs, *St[f"g{s_}:5-20"]["all"], c_, (0, (1, 1.5)), f"{s_}: lags 5-20", ms=3)
    head(ax[4], "e   Realised growth of a flipped\nobservation's difference, per stream")
    ax[4].set_ylabel(r"$\log\Vert\Delta_t\Vert-\log\Vert\Delta_{t-1}\Vert$")
    for r, c_ in (("trained", blue), ("echo", orange)):
        band(ax[5], A[r]["x"], A[r]["r_adf"], A[r]["r_adf_boot"], c_, "-",
             {"trained": "Trained RNN", "echo": "ESN"}[r])
        band(ax[5], A[r]["x"], A[r]["r_adf_partial"], A[r]["r_adf_partial_boot"], c_, (0, (2, 1.5)),
             {"trained": "beyond entanglement", "echo": None}[r], lw=1.4)
    head(ax[5], "f   Expanding where factorized-state Bayes\nexpands: " + r"$r(\log\rho_{net}, \log(1+\rho_1^{FS}))$"
         " per step")
    ax[5].set_ylabel("Pearson r over probes")
    import stage_bands as sb
    for a in ax:
        sb.shade(a)
        a.axhline(0, color="k", lw=.7)
        a.axvline(cross, color=".55", ls=":", lw=1)
        a.set_xscale("log")
        a.set_xticks([10, 100, 1e3, 1e4])
        a.set_xticklabels(["0", "100", "1k", "10k"])
        a.set_xlim(8, 4.4e4)
        a.minorticks_off()
        a.set_xlabel("training episodes")
        a.spines[["top", "right"]].set_visible(False)
    for a, loc in ((ax[0], "lower right"), (ax[1], "lower right"), (ax[2], "upper right"), (ax[3], "lower right"),
                   (ax[4], "lower right"), (ax[5], "upper left")):
        a.legend(fontsize=9.5, frameon=False, loc=loc, handlelength=2.2)
    fig.text(.5, .05, "Seed-20261029 networks; 1,000 probes (20 per test environment) x 30 steps; a-c, f: all 114 "
             "checkpoints; d-e: 16. Fifths of observation t's entanglement within each step. Bands: 5-95% over 1,000 "
             "environment resamples.\nFactorized-state Bayes (FS): the exact observer restricted to marginal beliefs; its "
             "step Jacobian's eigenvalues are 1 +- the canonical correlations of its one-step joint, so it expands at "
             "every step. Dotted vertical: the Trained RNN first exceeds naive (880); shaded: the behavioural stages.", ha="center", va="center",
             fontsize=10.5, color=".3", linespacing=1.5)
    path = fig_path("td_tt_chaos_streams.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "harvest":
        harvest(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    elif len(sys.argv) > 1 and sys.argv[1] == "numbers":
        numbers()
    else:
        figure()


# ── does the network expand where factorized-state Bayes expands? ───────────────────────────────────────────────────

def adf_tracking(run="trained", nboot=1000):
    """per checkpoint: the Pearson r over probes between the network's log rho(J_t) (td_tt_chaos_ent ext cache) and
    factorized-state Bayes's log rho at the same probe-step (td_tt_chaos_ent observers_upd 'lrho_adf_eig': its exact step
    Jacobian's eigenvalues 1 +- canonical correlations, so log rho = log(1 + rho_1)), per step 2-30 and then averaged
    over the steps; 'partial': both regressed on the step's entanglement first; and r(network, entanglement). 5-95% over
    the shared environment resamples (td_tt_gap.boot_counts: each resample weights each environment's 20 probes by its
    count). -> OUT/adf_tracking_<run>.npz"""
    path = os.path.join(OUT, f"adf_tracking_{run}.npz")
    import glob
    files = sorted(glob.glob(os.path.join(ce.OUT, run, "ext_*.npz")))
    raws = [int(os.path.basename(f)[4:9]) for f in files]
    if os.path.exists(path):
        Z = dict(np.load(path))
        if list(Z["raw"]) == raws:
            return Z
    O = np.load(os.path.join(ce.OUT, "observers_upd.npz"))
    assert np.array_equal(O["rows"], ce.rows())
    adf = O["lrho_adf_eig"]
    cov = ce.covariates()
    ent, env = cov["ent"], cov["env"]
    W = np.concatenate([np.ones((1, gap.N_ENV)), gap.boot_counts()[:nboot]], 0)[:, env]      # (1 + B, n) unit weights

    def wcorr(a, b, w):
        """weighted Pearson r of a, b (n,) under each weight row of w (B, n) -> (B,)"""
        sw = w.sum(1)
        ma, mb = (w @ a) / sw, (w @ b) / sw
        da, db = a[None] - ma[:, None], b[None] - mb[:, None]
        return (w * da * db).sum(1) / np.sqrt((w * da * da).sum(1) * (w * db * db).sum(1))

    def resid(a, c, w):
        """a minus its weighted least-squares fit on [1, c] under each weight row: (B, n)"""
        sw = w.sum(1)
        mc, ma = (w @ c) / sw, (w @ a) / sw
        dc, da = c[None] - mc[:, None], a[None] - ma[:, None]
        beta = (w * dc * da).sum(1) / (w * dc * dc).sum(1)
        return da - beta[:, None] * dc

    def wcorr2(A, Bm, w):
        sw = w.sum(1)
        ma, mb = (w * A).sum(1) / sw, (w * Bm).sum(1) / sw
        da, db = A - ma[:, None], Bm - mb[:, None]
        return (w * da * db).sum(1) / np.sqrt((w * da * da).sum(1) * (w * db * db).sum(1))
    out = {"raw": np.array(raws), "x": gap.train_x(raws)}
    rr, rp, re = [], [], []
    for f in files:
        L = np.load(f)["lrho"]
        a_, p_, e_ = [], [], []
        for t in range(1, ce.T):
            a_.append(wcorr(L[:, t], adf[:, t], W))
            e_.append(wcorr(L[:, t], ent[:, t], W))
            p_.append(wcorr2(resid(L[:, t], ent[:, t], W), resid(adf[:, t], ent[:, t], W), W))
        rr.append(np.mean(a_, 0))
        rp.append(np.mean(p_, 0))
        re.append(np.mean(e_, 0))
    for k, v in (("r_adf", rr), ("r_adf_partial", rp), ("r_ent", re)):
        v = np.array(v)                                                         # (K, 1 + B)
        out[k], out[k + "_boot"] = v[:, 0], v[:, 1:].T
    np.savez(path, **out)
    return out


def full_series(run):
    """log rho(J_t) of the full step at EVERY stored checkpoint (td_tt_chaos_ent ext cache): x, and for 'all', 'most',
    'least' (entanglement fifths within each step) the full-sample mean (K,) and resampled (B, K); 'pos_most' the share
    of the most entangled fifth's steps with log rho > 0 -> OUT/full_series_<run>.npz"""
    import glob
    path = os.path.join(OUT, f"full_series_{run}.npz")
    files = sorted(glob.glob(os.path.join(ce.OUT, run, "ext_*.npz")))
    raws = [int(os.path.basename(f)[4:9]) for f in files]
    if os.path.exists(path):
        Z = dict(np.load(path))
        if list(Z["raw"]) == raws and "pos_least" in Z:
            return Z
    cov = ce.covariates()
    q = ce.fifths(cov["ent"])
    env = np.broadcast_to(cov["env"][:, None], q.shape)
    Cb = gap.boot_counts()
    out = {"raw": np.array(raws), "x": gap.train_x(raws)}
    acc = {k: ([], []) for k in ("all", "most", "least", "pos_most", "pos_least")}
    for f in files:
        L = np.load(f)["lrho"]
        for name, sel, v in (("all", np.ones_like(q, bool), L), ("most", q == ce.NQ - 1, L), ("least", q == 0, L),
                             ("pos_most", q == ce.NQ - 1, (L > 0).astype(float)),
                             ("pos_least", q == 0, (L > 0).astype(float))):
            S_, C_ = np.zeros(gap.N_ENV), np.zeros(gap.N_ENV)
            np.add.at(S_, env[sel], v[sel])
            np.add.at(C_, env[sel], 1.)
            acc[name][0].append(S_.sum() / C_.sum())
            acc[name][1].append((Cb @ S_) / (Cb @ C_))
    for k, (a, b) in acc.items():
        out[k], out[k + "_boot"] = np.array(a), np.array(b).T
    np.savez(path, **out)
    return out
