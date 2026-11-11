"""How far back does the learning signal reach, through training? (John 11-08: the per-lag credit analysis offered as
"whether the memory horizon extends when credit at that lag appears".)
-> main/DATA/experimental/training_dynamics/through_training/credit_reach/<run>_EEEEE.npz
   main/REPLICATION/figure_bin/td_tt_credit_reach.png   figure()
Credit (gradflow_analysis's convention, recomputed from td_tt_chaos_ent.replay's states): the error at step t,
e_t = dL_t/du_t with L_t the step-t term of SANITY_loss (DKL_sym with the model's soft clip, minus the batch's detached
minimum, power .5) for each variable as the goal (both variables' errors summed: SANITY draws the goal at random), past
read-outs held fixed, is carried back through the cell and the recurrence (gradflow_analysis.backward's equations)
k = 0..15 steps; delta_k = its gradient on the gate pre-activations at step t - k, and E_k = ||delta_k||^2. Error steps
17-30 (every one has the 16 lags). Per probe: the credit share s_k = Sum_t E_k / Sum_t Sum_k E_k (statistic per probe,
then averaged), and the decay of log E_k over k.
Probes: 40 per test environment (2,000 of analysis/probes.npz, seed SUB_SEED). Checkpoints: X_SEL.
Run from the repository root (CPU):
    /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_credit_reach.py harvest"""
import os
import sys
import time
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_intervene as ti                                                                  # noqa: E402
import td_tt_chaos_ent as ce                                                                  # noqa: E402

OUT = os.path.join(gap.TT, "credit_reach")
T, R, NV, KMAX = 30, 10, 2, 15
SRC = np.arange(KMAX + 1, T)                                     # gradflow: SRC = arange(KMAX + 1, 30): steps 17-30
SUB_SEED, PER_ENV = 20261113, 40
X_SEL = {"trained": (10, 20, 40, 60, 80, 100, 160, 240, 320, 400, 480, 560, 640, 720, 800, 880, 960, 1040, 1200, 1360,
                     1600, 2000, 2400, 3000, 4000, 5000, 7000, 10000, 15000, 20000, 30000, 39996)}


def soft_clip(x, eps=1e-3):
    ciel = 1. - eps
    x = ciel - torch.sigmoid((ciel - x) / eps) * (ciel - x)
    return eps + torch.sigmoid((x - eps) / eps) * (x - eps)


def dkl_sym(q, p):
    q, p = soft_clip(q), soft_clip(p)
    return ((q * torch.log(q / p)).sum(-1) + (p * torch.log(p / q)).sum(-1)) / 2


def rows():
    P = ti.probes()
    rng = np.random.default_rng(SUB_SEED)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(P["env"] == e), PER_ENV, replace=False)
                                   for e in range(gap.N_ENV)]))


def credit(run, raw, rws, dev=torch.device("cpu")):
    """E (n, len(SRC), KMAX + 1): the credit energy of each error step at each lag; coh (n, len(SRC)): the credit
    coherence ||Sigma_k delta_k||^2 / Sigma_k ||delta_k||^2 (td_analysis 'cr_coh', for the check)"""
    p, table = ce.setup(run, raw, dev, dtype=torch.float64)
    envt, code = ce.codes(rws, dev)
    Pz = np.load(os.path.join(ti.AN, "probes.npz"))
    jb = torch.as_tensor(Pz["jb"][rws].astype(np.float64), device=dev)          # (n, T, 2, R) Bayes's marginals
    with torch.no_grad():
        A, U, C, lg = ce.replay(p, table, envt, code)
    Wro = p["classifier_readout.weight"].double()
    Whh = p["LSTM.weight_hh_l0"].double()
    n = len(rws)
    cum = lg.double().cumsum(1)                                                 # (n, T, 20) cumulative logits
    gates = [x.double() for x in ce._gates(A.double())]                         # i, f, g, o (n, T, H)
    # the batch's detached minimum of DKL over (episode, step), for each variable as the goal (as SANITY_loss's)
    with torch.no_grad():
        q_all = torch.softmax(cum.reshape(n, T, NV, R), -1)
        mins = [dkl_sym(q_all[:, :, v], jb[:, :, v]).min() for v in range(NV)]
    E = np.zeros((n, len(SRC), KMAX + 1))
    coh = np.zeros((n, len(SRC)))
    for j, t in enumerate(SRC):
        z = cum[:, t].clone().requires_grad_(True)                              # (n, 20): logits at t, past held fixed
        q = torch.softmax(z.reshape(n, NV, R), -1)
        Lt = sum(((dkl_sym(q[:, v], jb[:, t, v]) - mins[v] + 1e-8) ** .5).sum() for v in range(NV))
        gz, = torch.autograd.grad(Lt, z)
        du = gz @ Wro                                                           # e_t = dL_t / du_t   (n, H)
        dc = torch.zeros_like(du)
        tot = None
        with torch.no_grad():
            for k in range(KMAX + 1):
                s = t - k
                i, f, g, o = (x[:, s] for x in gates)
                cp, tcs = C[:, s].double(), torch.tanh(C[:, s + 1].double())
                dct = dc + du * o * (1 - tcs * tcs)
                delta = torch.cat([dct * g * i * (1 - i), dct * cp * f * (1 - f), dct * i * (1 - g * g),
                                   du * tcs * o * (1 - o)], -1)
                E[:, j, k] = (delta * delta).sum(-1).cpu().numpy()
                tot = delta if tot is None else tot + delta
                dc, du = dct * f, delta @ Whh
            coh[:, j] = ((tot * tot).sum(-1).cpu().numpy()) / E[:, j].sum(-1)
    return E, coh


def harvest():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    rws = rows()
    env = ti.probes()["env"][rws]
    for run, xs in X_SEL.items():
        eps = np.array(ti.checkpoints(run))
        for xv in xs:
            raw = int(eps[np.argmin(np.abs(np.log(gap.train_x(eps)) - np.log(xv)))])
            path = os.path.join(OUT, f"{run}_{raw:05d}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            E, coh = credit(run, raw, rws)
            share = E.sum(1) / E.sum((1, 2))[:, None]                           # (n, K + 1) per probe
            np.savez(path, x=gap.train_x(raw), raw=raw, env=env, share=share.astype(np.float32),
                     logE=np.log(E.mean(1) + 1e-300).astype(np.float32), coh=coh.astype(np.float32))
            print(f"{run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s | share lag 0 {share[:, 0].mean():.3f}"
                  f" 1-4 {share[:, 1:5].sum(1).mean():.3f} 5-15 {share[:, 5:].sum(1).mean():.3f} | coh median {np.median(coh):.3f}",
                  flush=True)


# ── figure ──────────────────────────────────────────────────────────────────────────────────────────────────────────

BANDS = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 15))


def series(run="trained"):
    """x (K,), per-environment means of the credit share per lag (K, 50, 16), of the decay slope of log E_k over
    k = 1..15 (K, 50) and of the coherence (K, 50)"""
    import glob
    import re
    fs = sorted(f for f in glob.glob(os.path.join(OUT, f"{run}_*.npz"))
                if re.fullmatch(rf"{run}_\d{{5}}\.npz", os.path.basename(f)))
    Z = sorted((dict(np.load(f)) for f in fs), key=lambda z: float(z["x"]))
    x = np.array([float(z["x"]) for z in Z])
    env = Z[0]["env"]
    cnt = np.bincount(env, minlength=gap.N_ENV).astype(float)
    k = np.arange(1, KMAX + 1)

    def per_env(v):                                                             # (n, ...) -> (50, ...)
        flat = v.reshape(len(v), -1)
        out = np.stack([np.bincount(env, flat[:, j], gap.N_ENV) for j in range(flat.shape[1])], -1)
        return (out / cnt[:, None]).reshape((gap.N_ENV,) + v.shape[1:])
    share = np.array([per_env(z["share"].astype(np.float64)) for z in Z])
    slope = np.array([per_env(np.polyfit(k, z["logE"][:, 1:].astype(np.float64).T, 1)[0]) for z in Z])
    coh = np.array([per_env(z["coh"].astype(np.float64).mean(1)) for z in Z])
    return x, share, slope, coh, cnt


def figure():
    import matplotlib.pyplot as plt
    from main.utils import fig_path, apply_fig_style
    import stage_bands as sb
    import td_tt_memlen as tm
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    x, share, slope, coh, cnt = series()
    Cb = gap.boot_counts()
    w_ = Cb * cnt[None]                                                          # resample weights per environment
    env_mean = lambda v: ((w_ @ v.reshape(len(v), -1) if v.ndim > 1 else w_ @ v) / w_.sum(1)[:, None]  # noqa: E731
                          if v.ndim > 1 else (w_ @ v) / w_.sum(1))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    fig.subplots_adjust(left=.06, right=.99, top=.8, bottom=.24, wspace=.32)
    a = ax[0]
    cols = plt.cm.viridis(np.linspace(0, .9, len(BANDS)))
    for (k0, k1), c in zip(BANDS, cols):
        v = share[:, :, k0:k1 + 1].sum(-1)                                       # (K, 50)
        full = (v * cnt).sum(1) / cnt.sum()
        boot = np.array([(w_ @ v[j]) / w_.sum(1) for j in range(len(x))]).T
        lo, hi = np.percentile(boot, [5, 95], 0)
        a.fill_between(x, lo, hi, color=c, alpha=.18, lw=0)
        a.plot(x, full, "o-", color=c, lw=1.8, ms=3, label=f"lag {k0}" if k0 == k1 else f"lags {k0}-{k1}")
    a.set_ylabel("share of the error's credit")
    a.set_title("a   Where an error's credit lands\n" + r"$\Vert\delta_k\Vert^2/\Sigma_k\Vert\delta_k\Vert^2$, errors at "
                "steps 17-30", loc="left", fontsize=12.5)
    a.legend(fontsize=9.5, frameon=False, loc="center left")
    a = ax[1]
    full = (slope * cnt).sum(1) / cnt.sum()
    boot = np.array([(w_ @ slope[j]) / w_.sum(1) for j in range(len(x))]).T
    lo, hi = np.percentile(boot, [5, 95], 0)
    a.fill_between(x, lo, hi, color="#1f77b4", alpha=.18, lw=0)
    a.plot(x, full, "o-", color="#1f77b4", lw=1.8, ms=3)
    a.axhline(0, color="k", lw=.7)
    a.set_ylabel(r"slope of $\log\Vert\delta_k\Vert^2$ over lags 1-15 (per lag)")
    a.set_title("b   How fast credit decays going back\n(0: it reaches every lag undiminished)", loc="left",
                fontsize=12.5)
    a = ax[2]                                                                    # credit arrival vs memory onset, per lag
    d = tm.loss("trained")
    s_r = tm.SOURCES.index("random")
    o = tm.onset_core(d["x"], d["net"][:, s_r], d["net_b"][:, s_r], d["bayes"][s_r], d["bayes_b"][s_r], .5)
    arr, arr_lo, arr_hi = [], [], []
    for w in range(1, KMAX + 1):
        v = share[:, :, w]                                                       # (K, 50)
        full_w = (v * cnt).sum(1) / cnt.sum()
        lvl = .5 * full_w.max()
        above = full_w >= lvl
        i = int(np.argmax(above))
        arr.append(x[i] if above.any() else np.nan)
    arr = np.array(arr)
    ws = np.arange(1, KMAX + 1)
    a.plot(arr, o["p"][:KMAX], "o", color="#1f77b4", ms=6)
    for w, xa, yo in zip(ws, arr, o["p"][:KMAX]):
        if w in (1, 2, 4, 5, 6, 10, 15):
            a.annotate(f"w = {w}", (xa, yo), textcoords="offset points", xytext=(6, -2), fontsize=9.5)
    lim = [50, 2e4]
    a.plot(lim, lim, color="0.6", lw=.8, ls=":")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(*lim)
    a.set_ylim(*lim)
    a.set_xlabel("credit at lag w reaches half its peak share (training episodes)")
    a.set_ylabel("answers depend on the state w steps back\n(memory-horizon onset, td-tt-memlen)")
    a.set_title("c   Does memory follow the credit?\n(each point a lag / window w)", loc="left", fontsize=12.5)
    for a in ax[:2]:
        sb.shade(a)
        a.set_xscale("log")
        a.set_xticks([10, 100, 1e3, 1e4])
        a.set_xticklabels(["0", "100", "1k", "10k"])
        a.set_xlim(8, 4.4e4)
        a.minorticks_off()
        a.axvline(880, color=".55", ls=":", lw=1)
        a.set_xlabel("training episodes")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .06, "Trained RNN (seed 20261029), 2,000 probes (40 per test environment) at 32 checkpoints. Credit: "
             "both variables' step errors sent back through the cell and the recurrence (gradflow-analysis's equations, "
             "the coherence reproduced within .002 of the stored values).\nBands 5-95% over 1,000 environment "
             "resamples. Dotted vertical: the Trained RNN first exceeds naive (880); shaded: the behavioural stages.",
             ha="center", va="center", fontsize=10.5, color=".3", linespacing=1.5)
    path = fig_path("td_tt_credit_reach.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)
    return dict(arrival=arr, onset=o["p"][:KMAX])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "harvest":
        harvest()
    else:
        figure()
