"""Causal test of learnability: what does going beyond naive need to be learned from? Analysis of the
train_learn_masks.py branches (every branch continues the shared trunk at raw episode 600 = 480 training episodes, with
the trunk's Adam state and the same episode stream; see that script for the conditions).
-> main/DATA/experimental/training_dynamics/through_training/learnmask/<cond>_EEEEE.npz (per checkpoint)
   main/REPLICATION/figure_bin/td_tt_learnmask.png   figure()
Measures on the 10,000 probes (analysis/probes.npz) at every stored checkpoint (raw 600, 700, ..., 1,600):
  accuracy  goal answer at T right
  ladder    td_analysis's regression of the network's per-step logit increment (centred over values) on naive's
            increment N, the within-step correction W = memory-free Bayes - naive and the memory correction M = Bayes -
            memory-free Bayes (probes.npz inc_N / inc_SW / inc_D), steps 2-30, both variables, no intercept: the
            coefficients (beta_N, beta_W, beta_M); per-environment normal-equation sums, so the 5-95% intervals resample
            the 50 test environments (td_tt_gap.boot_counts)
and, densely, the run's own test episodes (every 5th episode, 8,000 test-environment episodes each: tests.npz).
RESULTS 11-08 (branch - control at 1,280 training episodes; paired 5-95% over the 50 test environments; ONE run per
condition, so the intervals are test sampling only, not run-to-run variability; control: accuracy .617, beta N .786,
W .283, M .097). Gradient blocked through the most / least entangled / a random share of each step's updates:
  share  block   accuracy               beta_N          beta_W          beta_M
  5%     most    +.000 [-.008, +.008]   +.041           -.010           -.007 (-7%)
         least   -.018 [-.026, -.010]   +.001           +.007           +.004
         random  -.013 [-.022, -.004]   +.008           +.006           +.002
  20%    most    -.022 [-.033, -.012]   +.106           -.034           -.023 (-24%)
         least   -.015 [-.023, -.006]   -.000           +.023           +.011 (+11%)
         random  -.015 [-.024, -.006]   -.001           -.003           -.002
  50%    most    -.049 [-.059, -.040]   +.092           -.080           -.048 (-49%)
         least   -.069 [-.080, -.057]   -.095           +.043           +.027 (+28%)
         random  -.042 [-.052, -.033]   -.012           -.019           -.012
  Read: what the update is made of is ordered by the entanglement of the blocked updates at every share, dose-dependently:
  without the entangled updates the network learns less of Bayes's corrections and leans on naive's increment; without
  the least entangled ones (at 50%) it learns less of naive's part and more of the corrections. Accuracy is not ordered
  that way: at 5-20% every block costs about the same (random blocks alone swing by +-.02 between checkpoints); at 50%
  the least-entangled block costs the most (-.069 vs random -.042).
  Loss dropped instead (20%, mask_*): the same beta ordering at about half the size (beta_M -.011 / random +.002 / least
  +.009); accuracy -.009 / -.004 / -.008.
Run from the repository root:
    /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_learnmask.py harvest [conds]
    /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_learnmask.py table
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_learnmask.py"""
import os
import re
import sys
import glob
import time
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_intervene as ti                                                                  # noqa: E402
import td_tt_chaos_ent as ce                                                                  # noqa: E402

OUT = os.path.join(gap.TT, "learnmask")
TD = os.path.dirname(gap.TT)
CONDS = ("ctrl", "det_hi", "det_lo", "det_rand", "mask_hi", "mask_lo", "mask_rand",
         "det_hi_f50", "det_lo_f50", "det_rand_f50", "det_hi_f05", "det_lo_f05", "det_rand_f05")
FRACS = (("_f05", "5%"), ("", "20%"), ("_f50", "50%"))                         # gradient blocks, John 11-08: more extreme
SCRATCH = (("ctrl_s0", "k", "-", "control"),                                      # John 11-09: from episode 0, ranked by FR
           ("det_rand_f50_s0", "0.5", "-", "random half blocked"),
           ("det_hi_fr_f50_s0", "#d62728", "-", "highest-FR half blocked"),
           ("det_lo_fr_f50_s0", "#1f77b4", "-", "lowest-FR half blocked"),
           ("det_hi_f50_s0", "#d62728", (0, (3, 1.5)), "most entangled half blocked"),
           ("det_lo_f50_s0", "#1f77b4", (0, (3, 1.5)), "least entangled half blocked"))
STYLE = {"ctrl": ("k", "-", "control (no selection)"),
         "det_hi": ("#d62728", "-", "no gradient via the most entangled 20%"),
         "det_lo": ("#1f77b4", "-", "no gradient via the least entangled 20%"),
         "det_rand": ("0.5", "-", "no gradient via a random 20%"),
         "mask_hi": ("#d62728", (0, (3, 1.5)), "loss dropped at the most entangled 20%"),
         "mask_lo": ("#1f77b4", (0, (3, 1.5)), "loss dropped at the least entangled 20%"),
         "mask_rand": ("0.5", (0, (3, 1.5)), "loss dropped at a random 20%")}
for _f, _pc in FRACS[::2]:
    for _k, _w in (("hi", "the most entangled"), ("lo", "the least entangled"), ("rand", "a random")):
        STYLE[f"det_{_k}{_f}"] = (STYLE[f"det_{_k}"][0], "-", f"no gradient via {_w} {_pc}")
T, R, NV = 30, 10, 2


def run_dir(cond):
    return os.path.join(TD, f"learnmask_{cond}_s20261029")


@torch.no_grad()
def measure(cond, raw, P, X, dev=torch.device("cpu"), chunk=2500):
    """per-environment sums: correct (50,), XtX (50, 3, 3), XtD (50, 3)"""
    p, table = ce.setup(f"learnmask_{cond}", raw, dev)
    n = len(P["env"])
    envt, code = ce.codes(np.arange(n), dev)
    lg = []
    for b0 in range(0, n, chunk):
        sl = slice(b0, min(n, b0 + chunk))
        _, _, _, l_ = ce.replay(p, table, envt[sl], code[sl])
        lg.append(l_.cpu())
    lg = torch.cat(lg).double().reshape(n, T, NV, R)
    goal, truth = P["goal"], P["r"][np.arange(n), P["goal"]]
    right = (lg.sum(1)[np.arange(n), goal].argmax(-1).numpy() == truth).astype(float)
    d = lg - lg.mean(-1, keepdim=True)
    xtx = torch.einsum("ntvjk,ntvjl->nkl", X[:, 1:], X[:, 1:]).numpy()           # steps 2-30
    xtd = torch.einsum("ntvjk,ntvj->nk", X[:, 1:], d[:, 1:]).numpy()
    env = P["env"]
    return (np.bincount(env, right, gap.N_ENV),
            np.stack([np.bincount(env, xtx[:, a, b], gap.N_ENV) for a in range(3) for b in range(3)], -1).reshape(-1, 3, 3),
            np.stack([np.bincount(env, xtd[:, a], gap.N_ENV) for a in range(3)], -1))


def harvest(conds=CONDS):
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    P = dict(np.load(os.path.join(ti.AN, "probes.npz")))
    Nn, SW, D = (torch.as_tensor(P["inc_" + k]).double() for k in ("N", "SW", "D"))
    X = torch.stack([Nn, SW - Nn, D - SW], -1)                                   # (n, T, 2, R, 3)
    for cond in conds:
        for f in sorted(glob.glob(os.path.join(run_dir(cond), "ckpt_ep*_net.pth"))):
            raw = int(os.path.basename(f)[7:12])
            path = os.path.join(OUT, f"{cond}_{raw:05d}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            c, xtx, xtd = measure(cond, raw, P, X)
            np.savez(path, raw=raw, x=gap.train_x(raw), correct=c, xtx=xtx, xtd=xtd, n_env=np.bincount(P["env"]))
            beta = np.linalg.solve(xtx.sum(0), xtd.sum(0))
            print(f"{cond} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s | acc {c.sum() / len(P['env']):.4f} "
                  f"beta N {beta[0]:.3f} W {beta[1]:.3f} M {beta[2]:.3f}", flush=True)


def series(cond):
    pat = re.compile(rf"{re.escape(cond)}_\d{{5}}\.npz")                      # exact: det_hi_* would also match det_hi_f50_*
    fs = sorted(f for f in glob.glob(os.path.join(OUT, f"{cond}_*.npz")) if pat.fullmatch(os.path.basename(f)))
    Z = [dict(np.load(f)) for f in fs]
    Cb = gap.boot_counts()
    x = np.array([float(z["x"]) for z in Z])
    acc = np.array([z["correct"].sum() / z["n_env"].sum() for z in Z])
    acc_b = np.array([(Cb @ z["correct"]) / (Cb @ z["n_env"]) for z in Z]).T
    beta = np.array([np.linalg.solve(z["xtx"].sum(0), z["xtd"].sum(0)) for z in Z])
    beta_b = np.array([np.linalg.solve(np.einsum("be,ekl->bkl", Cb, z["xtx"]), np.einsum("be,ek->bk", Cb, z["xtd"]))
                       for z in Z]).transpose(1, 0, 2)                           # (B, K, 3)
    return dict(x=x, acc=acc, acc_b=acc_b, beta=beta, beta_b=beta_b)


def tests(cond):
    z = np.load(os.path.join(run_dir(cond), "tests.npz"))
    return gap.train_x(z["raw"]), z["acc"][:, -1], z["naive"][:, -1], z["joint"][:, -1]


def paired(S, c, key, i=None, ctrl="ctrl"):
    """branch minus control at the shared checkpoints: (x, full, resampled (B, K)); the same environment resamples
    for both, so the interval is paired"""
    a, b = S[c], S[ctrl]
    xs = np.intersect1d(a["x"], b["x"])
    ia, ib = np.searchsorted(a["x"], xs), np.searchsorted(b["x"], xs)
    if key == "acc":
        return xs, a["acc"][ia] - b["acc"][ib], a["acc_b"][:, ia] - b["acc_b"][:, ib]
    return xs, a["beta"][ia, i] - b["beta"][ib, i], a["beta_b"][:, ia, i] - b["beta_b"][:, ib, i]


def figure():
    """gradient blocks only (John 11-08: "we can stick to the blocking"): the control's and the half blocks' levels on top,
    then branch - control for 5%, 20% and 50% of each step's updates; one y range per column"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import fig_path, apply_fig_style
    import stage_bands as sb
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    have = [c for c in CONDS if c.startswith(("ctrl", "det_")) and glob.glob(os.path.join(OUT, f"{c}_*.npz"))]
    S = {c: series(c) for c in have}
    cols = (("acc", None, "accuracy at T", r"$P(\hat g_T = g)$"),
            ("beta", 1, "within-step correction", r"$\beta_W$"),
            ("beta", 2, "memory correction", r"$\beta_M$"))
    KIND = (("hi", "#d62728"), ("lo", "#1f77b4"), ("rand", "0.5"))
    fig, ax = plt.subplots(4, 3, figsize=(15.5, 16))
    fig.subplots_adjust(left=.075, right=.985, top=.88, bottom=.05, wspace=.3, hspace=.5)

    def band(a, x, full, boot, col, ls="-", lw=2):
        lo, hi = np.percentile(boot, [5, 95], 0)
        a.fill_between(x, lo, hi, color=col, alpha=.13, lw=0)
        a.plot(x, full, color=col, ls=ls, lw=lw, marker="o", ms=3.5)
    for j, (key, i, name, sym) in enumerate(cols):
        a = ax[0, j]
        for c in ("ctrl", "det_hi_f50", "det_lo_f50", "det_rand_f50"):
            if c in S:
                y, yb = (S[c]["acc"], S[c]["acc_b"]) if key == "acc" else (S[c]["beta"][:, i], S[c]["beta_b"][:, :, i])
                band(a, S[c]["x"], y, yb, "k" if c == "ctrl" else STYLE[c][0])
        a.set_title(f"{name}: {sym}\ncontrol and the 50% blocks", loc="left", fontsize=13.5)
        lim = []
        for r, (f, pc) in enumerate(FRACS, 1):
            a = ax[r, j]
            a.axhline(0, color="k", lw=1)
            for k, col in KIND:
                c = f"det_{k}{f}"
                if c in S:
                    x, d, db = paired(S, c, key, i)
                    band(a, x, d, db, col)
                    lim += [np.percentile(db, 5, 0).min(), np.percentile(db, 95, 0).max()]
            a.set_title(f"{name}, {pc} of each step blocked:\n{sym}(branch) $-$ {sym}(control)", loc="left",
                        fontsize=13.5)
        if lim:
            pad = .05 * (max(lim) - min(lim))
            for r in range(1, 4):
                ax[r, j].set_ylim(min(lim) - pad, max(lim) + pad)
    for a in ax.flat:
        sb.shade(a)
        a.set_xlim(450, 1310)
        a.set_xticks([480, 640, 800, 960, 1120, 1280])
        a.set_xlabel("training episodes (branch point 480)", fontsize=12)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=11)
    h = [Line2D([], [], color="k", lw=2.2, label="control (no block)")] + \
        [Line2D([], [], color=col, lw=2.2, label=f"no gradient via the {w} share of each step")
         for (k, col), w in zip(KIND, ("most entangled", "least entangled", "random"))]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(.5, .997))
    fig.text(.5, .962, "Each branch continues the same 480-episode network (same Adam state, same episodes); at every "
             "step the same share of the batch is blocked, ranked by goal entanglement at that step (no bias to late "
             "steps).\nOne run per condition. Bands: 5-95% over the 50 test environments (paired for differences), i.e. "
             "test sampling only, not run-to-run variability. 10,000 test episodes (50 environments x 200) per "
             "checkpoint.", ha="center", va="top", fontsize=11.5, color="0.25")
    path = fig_path("td_tt_learnmask.png")
    fig.savefig(path, dpi=125)
    plt.close(fig)
    print("saved", path)


def scratch_figure():
    """from-scratch blocks (train_learn_masks.py ... 3001 .5 <sel> 1): performance first. a: accuracy at T on the runs'
    own test episodes (every 5th episode, 8,000 test-environment episodes each, identical across runs), running mean of 9
    tests; b: the same minus the control's (paired: the same test episodes); c: the 10,000 probes with 5-95% intervals
    over environments; d-f: the update ladder"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import fig_path, apply_fig_style
    import stage_bands as sb
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    have = [c for c in SCRATCH if os.path.exists(os.path.join(run_dir(c[0]), "tests.npz"))]
    fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
    fig.subplots_adjust(left=.06, right=.985, top=.8, bottom=.08, wspace=.27, hspace=.45)
    k = 9
    sm = lambda y: np.convolve(y, np.ones(k) / k, mode="valid")                    # noqa: E731
    T0 = {}
    naive = bayes = None
    for c, col, ls, lab in have:
        z = np.load(os.path.join(run_dir(c), "tests.npz"))
        x, y = gap.train_x(z["raw"]), z["acc"][:, -1]
        T0[c] = (x, y)
        naive, bayes = z["naive"][:, -1].mean(), z["joint"][:, -1].mean()
        ax[0, 0].plot(sm(x), sm(y), color=col, ls=ls, lw=1.8, label=lab)
    for a in (ax[0, 0],):
        for v, nm in ((naive, "naive"), (bayes, "Bayes")):
            if v is not None:
                a.axhline(v, color="0.5", lw=.9, ls=":")
                a.text(12, v + .006, nm, fontsize=10, color="0.35")
    if "det_rand_f50_s0" in T0:                                                   # reference: the random half (both worlds)
        xc, yc = T0["det_rand_f50_s0"]
        ax[0, 1].axhline(0, color="0.5", lw=1.8)
        for c, col, ls, lab in have:
            if c == "det_rand_f50_s0":
                continue
            x, y = T0[c]
            n = min(len(x), len(xc))
            ax[0, 1].plot(sm(x[:n]), sm(y[:n] - yc[:n]), color=col, ls=ls, lw=1.8 if c != "ctrl_s0" else 1.2)
    W2 = {}
    for c in ("det_rand_f50_s0", "det_hi_fr_f50_s0", "det_lo_fr_f50_s0"):        # world 2 (rep 2): new seed and world
        f = os.path.join(run_dir(c + "_r2"), "tests.npz")
        if os.path.exists(f):
            z = np.load(f)
            W2[c] = (gap.train_x(z["raw"]), z["acc"][:, -1])
    for c, col in (("det_rand_f50_s0", "0.5"), ("det_hi_fr_f50_s0", "#d62728"), ("det_lo_fr_f50_s0", "#1f77b4")):
        if c in W2:
            ax[0, 0].plot(sm(W2[c][0]), sm(W2[c][1]), color=col, ls=":", lw=1.4)
            if c != "det_rand_f50_s0" and "det_rand_f50_s0" in W2:
                (x, y), (xr, yr) = W2[c], W2["det_rand_f50_s0"]
                n = min(len(x), len(xr))
                ax[0, 1].plot(sm(x[:n]), sm(y[:n] - yr[:n]), color=col, ls=":", lw=1.8)
    S = {c: series(c) for c, *_ in have if glob.glob(os.path.join(OUT, f"{c}_*.npz"))}
    for c, col, ls, lab in have:
        if c not in S:
            continue
        s = S[c]
        lo, hi = np.percentile(s["acc_b"], [5, 95], 0)
        ax[0, 2].fill_between(s["x"], lo, hi, color=col, alpha=.12, lw=0)
        ax[0, 2].plot(s["x"], s["acc"], color=col, ls=ls, lw=1.8, marker="o", ms=2.5)
        for j in range(3):
            lo, hi = np.percentile(s["beta_b"][:, :, j], [5, 95], 0)
            ax[1, j].fill_between(s["x"], lo, hi, color=col, alpha=.12, lw=0)
            ax[1, j].plot(s["x"], s["beta"][:, j], color=col, ls=ls, lw=1.8, marker="o", ms=2.5)
    heads = (("a", "Accuracy at T", r"$P(\hat g_T = g)$, own test episodes (8,000 each)"),
             ("b", "Accuracy minus the random-half block's", "same test episodes, running mean of 9 tests"),
             ("c", "Accuracy at T on the probes", "10,000 probes, 5-95% over environments"),
             ("d", "Naive's part", r"$\beta_N$ of the update ladder"),
             ("e", "Within-step correction", r"$\beta_W$"),
             ("f", "Memory correction", r"$\beta_M$"))
    for a, (l, n, sub) in zip(ax.flat, heads):
        a.set_title(f"{l}   {n}\n{sub}", loc="left", fontsize=13)
        sb.shade(a)
        a.set_xscale("log")
        a.set_xlim(10, 2600)
        a.set_xticks([10, 100, 1000])
        a.set_xticklabels(["0", "100", "1k"])
        a.minorticks_off()
        a.set_xlabel("training episodes", fontsize=12)
        a.spines[["top", "right"]].set_visible(False)
    h = [Line2D([], [], color=col, ls=ls, lw=2.2, label=lab) for c, col, ls, lab in SCRATCH]
    band = sb.legend_handles()[0]
    band.set_label("original run: answers half-way to Bayes")
    h.append(Line2D([], [], color="0.3", ls=":", lw=1.8, label="FR halves and random half, world 2 (new seed and world)"))
    fig.legend(handles=h + [band], loc="upper center", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(.5, .998))
    fig.text(.5, .925, "Trained RNN from initialization (seed 20261029, one episode stream for every run). At every step, half of "
             "the batch's updates get no gradient,\nranked by the goal belief's factorization regret "
             r"$KL(B_t\Vert N_t)$ or by its entanglement at that step. One run per condition;"
             "\nbands: 5-95% over the 50 test environments (test sampling only).", ha="center", va="top", fontsize=11,
             color="0.25", linespacing=1.5)
    path = fig_path("td_tt_learnmask_scratch.png")
    fig.savefig(path, dpi=125)
    plt.close(fig)
    print("saved", path)


def scratch_summary():
    """-> learnmask/scratch_summary.npz, the from-scratch FR blocks for the main figure. Per world (w1: seed 20261029,
    w2: rep 2, a new world and episode stream, train_learn_masks rep): the dense test accuracy at T (every 5th episode,
    8,000 test episodes shared by the runs of a world) of each FR-ranked half block minus the random-half block of the
    same world, running mean of 9 tests (x, d_hi, d_lo; w1 also control - random); end = the mean over the last 20
    tests (raw 2,905-3,000) with its standard error over those tests. The gradient shares (td_tt_learnmask_grad.py,
    control run, probes): the share of the error signal |dL/d increment|^2 carried by the highest-FR half, per checkpoint."""
    k = 9
    sm = lambda y: np.convolve(y, np.ones(k) / k, mode="valid")                    # noqa: E731
    out = {}
    for w, suf in (("w1", ""), ("w2", "_r2")):
        T_ = {}
        for c in ("ctrl_s0", "det_rand_f50_s0", "det_hi_fr_f50_s0", "det_lo_fr_f50_s0"):
            f = os.path.join(run_dir(c + suf), "tests.npz")
            if os.path.exists(f):
                z = np.load(f)
                T_[c] = (z["raw"], z["acc"][:, -1], z["naive"][:, -1].mean(), z["joint"][:, -1].mean())
        if "det_rand_f50_s0" not in T_:
            continue
        rr, ra = T_["det_rand_f50_s0"][:2]
        out[f"{w}_naive"], out[f"{w}_bayes"] = T_["det_rand_f50_s0"][2:]
        for c, nm in (("det_hi_fr_f50_s0", "hi"), ("det_lo_fr_f50_s0", "lo"), ("ctrl_s0", "ctrl")):
            if c not in T_:
                continue
            r, a = T_[c][:2]
            n = min(len(r), len(rr))
            assert (r[:n] == rr[:n]).all(), (w, c)
            d = a[:n] - ra[:n]
            out[f"{w}_x_{nm}"], out[f"{w}_d_{nm}"] = sm(gap.train_x(r[:n])), sm(d)
            if r[n - 1] == 3000:
                out[f"{w}_end_{nm}"] = np.array([d[-20:].mean(), d[-20:].std(ddof=1) / 20 ** .5])
        out[f"{w}_acc_rand_end"] = ra[-20:].mean() if rr[-1] == 3000 else np.nan
    G = sorted(glob.glob(os.path.join(gap.TT, "learnmask_grad", "ctrl_s0_*.npz")))
    if G:
        Z = [np.load(f) for f in G]
        out["grad_x"] = np.array([float(z["x"]) for z in Z])
        out["grad_share_hi"] = np.array([z["share"][0].sum() / z["share"][-1].sum() for z in Z])
        out["grad_share_lo"] = np.array([z["share"][1].sum() / z["share"][-1].sum() for z in Z])
        out["grad_share_rand"] = np.array([z["share"][2].sum() / z["share"][-1].sum() for z in Z])
    path = os.path.join(OUT, "scratch_summary.npz")
    np.savez(path, **out)
    print("saved", path)
    for w in ("w1", "w2"):
        for nm in ("hi", "lo", "ctrl"):
            if f"{w}_end_{nm}" in out:
                e = out[f"{w}_end_{nm}"]
                print(f"  {w} {nm} - random, end (last 20 tests): {e[0]:+.3f} +- {e[1]:.3f}")
            elif f"{w}_d_{nm}" in out:
                print(f"  {w} {nm} - random, so far (x {out[f'{w}_x_{nm}'][-1]:.0f}): {out[f'{w}_d_{nm}'][-1]:+.3f}")
    if "grad_x" in out:
        print(f"  control: highest-FR half's share of the error signal {out['grad_share_hi'].min():.3f}-"
              f"{out['grad_share_hi'].max():.3f} over {len(G)} checkpoints (random {out['grad_share_rand'].min():.3f}-"
              f"{out['grad_share_rand'].max():.3f})")


def scratch_table():
    """per from-scratch run: when the dense test accuracy (running mean of 9 tests) passes naive for good, and the end
    values on the probes (branch - control, paired 5-95% over environments)"""
    k = 9
    sm = lambda y: np.convolve(y, np.ones(k) / k, mode="valid")                    # noqa: E731
    S = {}
    for c, *_ in SCRATCH:
        f = os.path.join(run_dir(c), "tests.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        x, y, nv = sm(gap.train_x(z["raw"])), sm(z["acc"][:, -1]), z["naive"][:, -1].mean()
        above = y > nv
        cross = x[np.flatnonzero(~above)[-1] + 1] if above[-1] and (~above).any() else np.nan
        line = f"{c:18s} tests to raw {int(z['raw'][-1])}: end acc {y[-1]:.3f}, passes naive ({nv:.3f}) for good at {cross:,.0f}"
        if glob.glob(os.path.join(OUT, f"{c}_*.npz")):
            S[c] = series(c)
            b = S[c]["beta"][-1]
            line += f" | probes x {S[c]['x'][-1]:.0f}: acc {S[c]['acc'][-1]:.3f}, beta N {b[0]:.3f} W {b[1]:.3f} M {b[2]:.3f}"
        print(line)
    if "ctrl_s0" in S:
        for c in S:
            if c == "ctrl_s0":
                continue
            out = []
            for key, i, nm in (("acc", None, "acc"), ("beta", 0, "bN"), ("beta", 1, "bW"), ("beta", 2, "bM")):
                x, d, db = paired(S, c, key, i, "ctrl_s0")
                lo, hi = np.percentile(db, [5, 95], 0)
                out.append(f"{nm} {d[-1]:+.3f} [{lo[-1]:+.3f}, {hi[-1]:+.3f}]")
            print(f"  {c} - control at {x[-1]:.0f}: " + "; ".join(out))


def table():
    """branch - control at every checkpoint, with paired 5-95% intervals"""
    have = [c for c in CONDS if glob.glob(os.path.join(OUT, f"{c}_*.npz"))]
    S = {c: series(c) for c in have}
    for c in have[1:]:
        print(c)
        for key, i, nm in (("acc", None, "acc"), ("beta", 1, "bW"), ("beta", 2, "bM")):
            x, d, db = paired(S, c, key, i)
            lo, hi = np.percentile(db, [5, 95], 0)
            print(f"  {nm:4s}" + " ".join(f"{xx:5.0f}:{v:+.3f}[{l:+.3f},{h:+.3f}]" for xx, v, l, h in zip(x, d, lo, hi)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "harvest":
        harvest(tuple(sys.argv[2:]) or CONDS)
    elif len(sys.argv) > 1 and sys.argv[1] == "table":
        table()
    elif len(sys.argv) > 1 and sys.argv[1] == "scratch":
        scratch_figure()
        scratch_table()
        scratch_summary()
    else:
        figure()
