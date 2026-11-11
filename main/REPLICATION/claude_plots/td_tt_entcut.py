"""Causal test: does the network's accuracy depend on re-evaluating the past at ENTANGLED observations? (John 11-08:
"We really want causal account of learnability"; the analysis offered as "memory cut only at the most vs least
entangled observations (matched counts), Trained RNN vs Bayes".)
-> main/DATA/experimental/training_dynamics/through_training/entcut/<run>_EEEEE.npz, bayes.npz
   main/REPLICATION/figure_bin/td_tt_entcut.png   figure()
Probes: analysis/probes.npz (10,000 episodes = 50 test environments x 100 cells x 2 trajectories, 30 steps). Donor:
one fixed probe per probe from the same environment and another cell (seed DONOR_SEED) - the random-cell swap
(memory memcut-random-donor-preferred).
Cut at step s (s = 2..30): the update at s is computed from the DONOR's memory before s and the probe's own observation
at s; every other update is the intact one (the network's belief is the cumulative sum of its updates, so only the
update at the cut steps changes; the state after s continues intact).
  network  the cell is stepped once from the donor's (h, c) after s - 1 steps with the probe's input at s; the cut
           update = the read-out of that state
  Bayes    the change of its marginals from adding observation s to the donor's first s - 1 observations (exact, from
           the environment's likelihood table)
  naive    needs no memory: a cut changes nothing
Steps cut: those whose observation's entanglement (rev, the mean of the two variables' 'ent') is in fifth k (f0 least ..
f4 most) or decile k (d0 .. d9) of its step (over the probes), or a random fifth / tenth (seed SEL_SEED), always at
steps 2-30, so every fifth (decile) cuts the same number of updates as the random fifth (tenth); 'all' cuts every step
2-30. Measure: goal
accuracy at T, intact minus cut (the accuracy the memory at those steps is worth); 5-95% over the 1,000 shared
environment resamples (td_tt_gap.boot_counts), per-environment sums.
Run from the repository root (CPU):
    /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_entcut.py harvest
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_entcut.py"""
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
import td_tt_chaos as tc                                                                      # noqa: E402
import td_tt_chaos_ent as ce                                                                  # noqa: E402

OUT = os.path.join(gap.TT, "entcut")
T, R, NV, NO = 30, 10, 2, 5
DONOR_SEED, SEL_SEED = 20261111, 20261112
CONDS = tuple(f"f{k}" for k in range(5)) + tuple(f"d{k}" for k in range(10)) + ("random", "random10", "all")
X_SEL = {"trained": (10, 100, 320, 480, 640, 800, 880, 1040, 1200, 1600, 2000, 3000, 5000, 10000, 20000, 39996),
         "echo": (10, 880, 4800, 39996)}


def setup():
    """probes, donors (n,), the selection masks {cond: (n, T) bool}"""
    P = np.load(os.path.join(ti.AN, "probes.npz"))
    env, cell = P["env"], P["cell"]
    n = len(env)
    rng = np.random.default_rng(DONOR_SEED)
    donor = np.empty(n, np.int64)
    for e in np.unique(env):
        ix = np.flatnonzero(env == e)
        for i in ix:
            pool = ix[cell[ix] != cell[i]]
            donor[i] = rng.choice(pool)
    rev = P["ent"].astype(np.float64).mean(-1)                                   # (n, T)
    q5, q10 = ce.fifths(rev), ce.fifths(rev, nb=10)                              # within each step
    rs = np.random.default_rng(SEL_SEED)
    rnd, rnd10 = np.zeros((n, T), bool), np.zeros((n, T), bool)
    for t in range(T):
        perm = rs.permutation(n)
        rnd[perm[:n // 5], t] = True
        rnd10[perm[:n // 10], t] = True
    sel = {**{f"f{k}": q5 == k for k in range(5)}, **{f"d{k}": q10 == k for k in range(10)},
           "random": rnd, "random10": rnd10, "all": np.ones((n, T), bool)}
    for k in sel:
        sel[k][:, 0] = False                                                     # step 1: no memory to cut
    return P, donor, sel


def _log_marg(n_cnt, k, L):
    """Bayes's log marginals (m, 2, R) from per-observable counts n_cnt (m, 5) over k steps, tables L (m, 5, R, R)"""
    from scipy.special import logsumexp
    lj = np.einsum("mi,mirs->mrs", n_cnt, np.log(L)) + np.einsum("mi,mirs->mrs", k - n_cnt, np.log1p(-L))
    lj = lj - logsumexp(lj.reshape(len(lj), -1), -1)[:, None, None]
    return np.stack([logsumexp(lj, 2), logsumexp(lj, 1)], 1)


def bayes_increments(P, donor):
    """intact and cut Bayes increments (n, T, 2, R) of the log marginals: intact = log B_t - log B_t-1; cut at t = the
    change from adding observation t to the donor's first t - 1 observations"""
    path = os.path.join(OUT, "bayes.npz")
    if os.path.exists(path):
        Z = np.load(path)
        return Z["inc"], Z["inc_cut"]
    obs = P["obs"].astype(np.float64)
    L = P["L"].astype(np.float64)[P["env"]]                                      # (n, 5, R, R)
    cnt = np.concatenate([np.zeros_like(obs[:, :1]), obs.cumsum(1)], 1)          # (n, T + 1, 5): counts after t steps
    n = len(obs)
    lm = np.stack([_log_marg(cnt[:, t], t, L) for t in range(T + 1)], 1)         # (n, T + 1, 2, R)
    inc = lm[:, 1:] - lm[:, :-1]
    inc_cut = np.zeros_like(inc)
    inc_cut[:, 0] = inc[:, 0]
    for t in range(1, T):
        prev = cnt[donor, t]                                                     # donor's counts after t steps
        before = _log_marg(prev, t, L)
        after = _log_marg(prev + obs[:, t], t + 1, L)
        inc_cut[:, t] = after - before
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, inc=inc, inc_cut=inc_cut)
    return inc, inc_cut


@torch.no_grad()
def net_increments(run, raw, P, donor, dev=torch.device("cpu"), chunk=2500):
    """intact and cut read-out increments (n, T, 20) float64 of the network"""
    p, table = ce.setup(run, raw, dev)
    rws = np.arange(len(P["env"]))
    envt, code = ce.codes(rws, dev)
    Whh = p["LSTM.weight_hh_l0"]
    n = len(rws)
    lg_all, U_all, C_all = [], [], []
    for b0 in range(0, n, chunk):
        sl = slice(b0, min(n, b0 + chunk))
        A, U, C, lg = ce.replay(p, table, envt[sl], code[sl])
        lg_all.append(lg.cpu())
        U_all.append(U.cpu())
        C_all.append(C.cpu())
        del A
    lg, U, C = torch.cat(lg_all), torch.cat(U_all), torch.cat(C_all)
    cut = lg.clone()
    d = torch.as_tensor(donor)
    for t in range(1, T):                                                        # 0-based step t: state before = index t
        for b0 in range(0, n, chunk):
            ix = torch.arange(b0, min(n, b0 + chunk))
            h0, c0 = U[d[ix], t].to(dev), C[d[ix], t].to(dev)
            h1, _, _ = tc.lstm_step(table[envt[ix.to(dev)], code[ix.to(dev), t]], h0, c0, Whh)
            cut[ix, t] = torch.nn.functional.linear(h1, p["classifier_readout.weight"],
                                                    p["classifier_readout.bias"]).double().cpu()
    return lg.double().numpy(), cut.numpy()


def accuracy(inc, inc_cut, sel, P, kind):
    """goal correct at T (n,) under each condition: the cumulative updates with the cut ones at the selected steps"""
    goal, truth = P["goal"], P["r"][np.arange(len(P["goal"])), P["goal"]]
    out = {}
    for c in ("intact",) + CONDS:
        use = inc if c == "intact" else np.where(sel[c][..., None, None] if kind == "bayes" else sel[c][..., None],
                                                  inc_cut, inc)
        tot = use.sum(1)                                                         # (n, 2, R) or (n, 20)
        if kind != "bayes":
            tot = tot.reshape(len(tot), NV, R)
        out[c] = tot[np.arange(len(goal)), goal].argmax(-1) == truth
    return out


def env_sums(right, env):
    return {c: np.bincount(env, v.astype(float), gap.N_ENV) for c, v in right.items()}


def harvest():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    P, donor, sel = setup()
    t0 = time.time()
    binc, bcut = bayes_increments(P, donor)
    bs = env_sums(accuracy(binc, bcut, sel, P, "bayes"), P["env"])
    np.savez(os.path.join(OUT, "bayes_acc.npz"), **bs)
    print(f"bayes: {time.time() - t0:.0f} s | " + " ".join(f"{c} {v.sum() / len(P['env']):.4f}" for c, v in bs.items()),
          flush=True)
    for run, xs in X_SEL.items():
        eps = np.array(ti.checkpoints(run))
        for xv in xs:
            raw = int(eps[np.argmin(np.abs(np.log(gap.train_x(eps)) - np.log(xv)))])
            path = os.path.join(OUT, f"{run}_{raw:05d}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            inc, cut = net_increments(run, raw, P, donor)
            s = env_sums(accuracy(inc, cut, sel, P, "net"), P["env"])
            np.savez(path, x=gap.train_x(raw), raw=raw, **s)
            print(f"{run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s | " +
                  " ".join(f"{c} {v.sum() / len(P['env']):.4f}" for c, v in s.items()), flush=True)




# ── figure ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def load(run):
    import glob
    fs = sorted(glob.glob(os.path.join(OUT, f"{run}_*.npz")))
    Z = sorted((dict(np.load(f)) for f in fs), key=lambda z: float(z["x"]))
    return Z


def losses(Z, conds, Cb, n_env):
    """intact minus cut accuracy: full (K, C) and resampled (B, K, C)"""
    full = np.array([[(z["intact"].sum() - z[c].sum()) / n_env.sum() for c in conds] for z in Z])
    boot = np.array([[(Cb @ (z["intact"] - z[c])) / (Cb @ n_env) for c in conds] for z in Z]).transpose(2, 0, 1)
    return full, boot


def figure():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from main.utils import fig_path, apply_fig_style
    import stage_bands as sb
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    P = np.load(os.path.join(ti.AN, "probes.npz"))
    n_env = np.bincount(P["env"], minlength=gap.N_ENV).astype(float)
    Cb = gap.boot_counts()
    B = dict(np.load(os.path.join(OUT, "bayes_acc.npz")))
    dec = [f"d{k}" for k in range(10)]
    bl = np.array([(B["intact"].sum() - B[c].sum()) / n_env.sum() for c in dec])
    bl_b = np.array([(Cb @ (B["intact"] - B[c])) / (Cb @ n_env) for c in dec]).T
    Zt, Ze = load("trained"), load("echo")
    xt = np.array([float(z["x"]) for z in Zt])
    Lt, Lt_b = losses(Zt, dec + ["random10", "all"], Cb, n_env)
    Le, Le_b = losses(Ze, dec + ["random10", "all"], Cb, n_env)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    fig.subplots_adjust(left=.06, right=.99, top=.8, bottom=.24, wspace=.3)
    blues = plt.cm.Blues
    stages = (320, 640, 880, 1200, 2000, 5000, 39996)
    snorm = LogNorm(stages[0], stages[-1])
    a = ax[0]
    dx = np.arange(10)
    for xv in stages:
        k = int(np.argmin(np.abs(np.log(xt) - np.log(xv))))
        lo, hi = np.percentile(Lt_b[:, k, :10], [5, 95], 0)
        c_ = blues(.35 + .65 * snorm(xt[k]))
        a.fill_between(dx, lo, hi, color=c_, alpha=.15, lw=0)
        a.plot(dx, Lt[k, :10], "o-", color=c_, lw=1.8, ms=3.5, label=f"{xt[k]:,.0f}")
    lo, hi = np.percentile(bl_b, [5, 95], 0)
    a.fill_between(dx, lo, hi, color="k", alpha=.1, lw=0)
    a.plot(dx, bl, "s-", color="k", lw=1.8, ms=3.5, label="Bayes")
    lo, hi = np.percentile(Le_b[:, -1, :10], [5, 95], 0)
    a.plot(dx, Le[-1, :10], "^-", color="#ff7f0e", lw=1.6, ms=3.5, label="ESN, end")
    a.set_xticks(dx)
    a.set_xticklabels([str(k + 1) for k in dx])
    a.set_xlabel("decile of the observation's entanglement (within its step)")
    a.set_ylabel("accuracy at T lost by the cut")
    a.set_title("a   Cutting the memory at 10% of the steps\n(each decile of entanglement; Trained RNN at stages)",
                loc="left", fontsize=12.5)
    a.legend(title="training episodes", fontsize=9, title_fontsize=9.5, frameon=False, ncol=2, loc="upper left")
    a = ax[1]
    slope = lambda L: np.polyfit(dx, L.T, 1)[0]                                   # noqa: E731
    st, st_b = slope(Lt[:, :10]), np.array([slope(b[:, :10]) for b in Lt_b])
    se, se_b = slope(Le[:, :10]), np.array([slope(b[:, :10]) for b in Le_b])
    sbay, sbay_b = np.polyfit(dx, bl, 1)[0], np.polyfit(dx, bl_b.T, 1)[0]
    xe = np.array([float(z["x"]) for z in Ze])
    for x_, y, yb, col in ((xt, st, st_b, "#1f77b4"), (xe, se, se_b, "#ff7f0e")):
        lo, hi = np.percentile(yb, [5, 95], 0)
        a.fill_between(x_, lo, hi, color=col, alpha=.15, lw=0)
        a.plot(x_, y, "o-", color=col, lw=1.8, ms=3.5)
    a.axhline(sbay, color="k", lw=1.2)
    a.axhspan(*np.percentile(sbay_b, [5, 95]), color="k", alpha=.08, lw=0)
    a.text(12, sbay, "Bayes", fontsize=10, va="bottom")
    a.set_ylabel("loss per decile of entanglement (slope)")
    a.set_title("b   Memory matters more where observations\nare entangled: slope of a over the deciles", loc="left",
                fontsize=12.5)
    a = ax[2]
    for x_, L, Lb, col, lab in ((xt, Lt, Lt_b, "#1f77b4", "Trained RNN"), (xe, Le, Le_b, "#ff7f0e", "ESN")):
        for j, ls, l2 in ((10, "-", "a random 10% of the steps"), (11, (0, (3, 1.5)), "every step (memory-free)")):
            lo, hi = np.percentile(Lb[:, :, j], [5, 95], 0)
            a.fill_between(x_, lo, hi, color=col, alpha=.12, lw=0)
            a.plot(x_, L[:, j], color=col, ls=ls, lw=1.8, marker="o", ms=3, label=f"{lab}: {l2}")
    for c_, ls in (("random10", "-"), ("all", (0, (3, 1.5)))):
        v = (B["intact"].sum() - B[c_].sum()) / n_env.sum()
        a.axhline(v, color="k", lw=1, ls=ls)
    a.set_ylabel("accuracy at T lost by the cut")
    a.set_title("c   How much the answers rest on memory\n(black: Bayes under the same cuts)", loc="left",
                fontsize=12.5)
    a.legend(fontsize=9, frameon=False, loc="center left", bbox_to_anchor=(0, .55))
    for a in ax[1:]:
        sb.shade(a)
        a.set_xscale("log")
        a.set_xticks([10, 100, 1e3, 1e4])
        a.set_xticklabels(["0", "100", "1k", "10k"])
        a.set_xlim(8, 4.4e4)
        a.minorticks_off()
        a.axvline(880, color=".55", ls=":", lw=1)
        a.set_xlabel("training episodes")
    for a in ax:
        a.axhline(0, color="k", lw=.6)
        a.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .06, "10,000 probes (50 test environments x 100 cells x 2 trajectories). Cut at step s: the update at s "
             "computed from another cell's memory (same environment, one fixed donor per probe) and the probe's own "
             "observation; every other update intact.\nBayes: its exact update from the donor's history plus the own "
             "observation. Bands 5-95% over 1,000 environment resamples. Dotted: the Trained RNN first exceeds naive "
             "(880); shaded: the behavioural stages. The deciles (not a random 10%) are the matched comparison: "
             "entanglement is correlated within an episode, so a decile's cuts share episodes and a random 10% does not.", ha="center", va="center", fontsize=10.5, color=".3",
             linespacing=1.5)
    path = fig_path("td_tt_entcut.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "harvest":
        harvest()
    else:
        figure()
