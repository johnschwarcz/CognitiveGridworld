"""Single-step dose tests of the entanglement-locked activity features (John 11-09: "make sure that you're looking at
the accuracy drop at the step of manipulation, otherwise the effects could be hidden by the fact that the network
recovers. Let's do a better job with the pilot before we do a full training run"). Replaces td_tt_clamp_norm's
every-step clamp read at T.
Design. At a checkpoint, replay the 10,000 probes; in every episode change the state at ONE step s only, and compare
with the intact replay of the same episode (paired):
  feature  norm: u_s -> d * u_s (the step's read-out increment W u_s scales by d; the next step's recurrent drive too);
           turn: the part of u_s orthogonal to u_s-1 scaled by d, then rescaled to ||u_s|| (more or less turn, same
           norm)
  dose     d in DOSE (1 = intact)
  where    'ent': s drawn among the episode's steps in the most entangled fifth of their step (rev = mean of the two
           variables' entanglement, ranked within each step), s in 2..23 (1-based); 'after': s = that step + 2 (the
           after-step response that tracked accuracy across the blocking runs, td_tt_block_pilot); 'rand': the step
           index of another episode's 'ent' step (the same step distribution, typical entanglement)
  outcome  at s, s+1, ..., s+5 and at T: the belief error KL(O || q) (nats, mean of the two variables) and the accuracy
           (each variable's argmax == its true value, mean of the two), manipulated - intact, mean over the 10,000
           episodes. Step s itself is the immediate effect (no chance to recover).
-> through_training/dose/<run>.npz: raw (C,), dkl, dacc (C, feature, where, dose, 7: lags 0-5 and T), their SEs,
   base_kl / base_acc (C, where, 7) of the intact replay, feat_ent / feat_rand (C, 2): the intact feature at the 'ent' /
   'rand' steps.
-> figure_bin/learn_dose_pilot.png (figure()): a the effect over the steps after s (state x 2 at an entangled step); b
   at s and at T through training; c halving the state at s, d doubling the turn at s, entangled vs random steps.
RESULTS (11-09; 'hurt' = the belief error KL(O || q) rises; td_tt_clamp_norm's '4.5x' was KL at T after clamping
the whole entangled fifth: +.0027 vs +.0006 nats, accuracy at T -.002).
  Reading at T hides most of the effect, once the network corrects (a, b). State x 2 at an entangled step, Trained RNN,
    KL at s / at T: 640 training episodes .119 / .222 (the error grows after s), 880 .079 / .096, 2,400 .037 / .031,
    8,000 .063 / .026, end .083 / .020; part left at T: 1.86, 1.21, 1.21 (1,200), 1.02 (1,600), .83 (2,400), .56,
    .41, .31, .24: the Trained RNN learns to correct a perturbed step only AFTER passing naive (from ~2k; the same
    period in which fig_learn_order_diag's steps start removing inherited error). ESN: 1.3-1.7 at every checkpoint,
    never. Halving: part left at T .45 (2,400) -> .12 (end).
  Accuracy at the step (points, 0.2-0.3 SE): small for every dose; largest state x 2 at the end, entangled -2.9 vs random
    -1.3 (at T -1.0 / -0.7); state x .5 -0.7 / -0.6.
  Norm (c). Until the crossing (880) halving the state at a step LOWERS the error at that step (-.099 at 200, -.034
    at 640, -.009 at 880: the young network's steps are too large) equally at entangled and random steps (-.034 /
    -.036 at 640); from ~1,600 halving hurts, and from 2,400 more at entangled steps (+.020 vs +.014; end
    +.028 vs +.015): entangled steps come to need their extra norm when the lag-0 norm lock forms (after the crossing).
    The young network had lost the input's negative norm lock (-.16 -> .01 by 880) with no entanglement-specific cost.
  Turn (d). From 200 to ~2,400 training episodes more turn LOWERS the error at the step, more at entangled steps (x 2:
    -.028 vs -.023 at 640, -.021 vs -.016 at 880, -.012 vs -.007 at 1,600, -.008 vs -.002 at 2,400); less turn costs
    more there (x .5: +.031 vs +.023 at 880); calibrated from ~4,000 (both directions cost). So the turn lock the
    Trained RNN loses by 880 (input's .31 -> .13) was worth keeping at inference, and the loss pushes it back (lag-0
    turn lock .27 at the end). ESN at the end: doubling the turn hurts at entangled steps (+.013) and helps at random
    ones (-.007): it keeps the input's lock (.32).
  After-step (s = entangled step + 2): no different from random steps (state x .5 at the end +.011 vs +.015; turn x 2
    at 640 -.022 vs -.023): the after-step norm response that tracked accuracy across the blocking runs
    (td_tt_block_pilot) is not used at inference.
EVERY CONTENDER (11-09, second pass; John: "explore causal manipulations for every contender in learn_maps"). Features
  norm, turn (as above), dim: the deviation of u_s from its step's mean, in the step's PCs (eigenvectors of the 10,000
  probes' states at that step), y -> sign(y) |y|^(1/d) rescaled to ||y|| (d = .5 concentrates it, effective dimension
  x ~.2; d = 2 spreads it, x ~9); band: the deviation's part in the top band (the PCs holding the first 10% of the
  step's variance) x d, then the deviation rescaled to its old norm (top-band share x .3 / x 3). Where adds 'fr': one of
  the episode's steps in the highest fifth of FR_s = KL(O_s || N_s) (mean of the two variables) of its step. Doses .5
  and 2. -> learn_dose_contenders.png (figure_contenders); the first pass (4 doses, no 'fr') is in the session
  scratchpad (td_tt_dose_v1.py, dose_*_v1.npz). Map columns f-i (update measures) and b (credit) have no inference
  side; their learning tests are train_learn_masks (mask_*, det_*) and train_learn_causal.
  RESULTS (dKL at the step, nats, entangled / high-FR / random; SE ~.001; the intact error at the step is .94 at 880,
  .24 at 2,400, .011 at the end):
  - Every contender's effect is small: before ~2k a few % of the step's error; accuracy at the step moves <= 1.2 points
    (norm x 2 at the end: -2.9 / -1.7 / -1.3).
  - Entanglement-specific parts are small: turn x .5 at 880 .031 / .033 / .023, turn x 2 -.021 / -.024 / -.015; norm
    x .5 at the end .027 / .028 / .015; dim x .5 at 880 .042 / .056 / .038; band x 2 at 1,200 .032 / .039 / .030.
  - High-FR steps are the most sensitive to norm, dim and band (dim x .5 at 2,400 .042 / .057 / .033), and the ESN has
    the same order (norm x 2 at the end .148 / .183 / .116): this sensitivity is not learned.
  - Until ~1.2k less norm, more turn, a spread state and less top-band energy LOWER the error at every kind of step
    (the young network's states are too large and too concentrated: dim x 2 at 200 -.072 / -.065 / -.076); from ~2.4k
    every change hurts.
Run from the repository root (GPU 0, ~25 s per checkpoint; the second pass ~46 s):
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_dose.py trained
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_dose.py figure"""
import os
import sys
import time
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import td_analysis as td                                                                      # noqa: E402

OUT = os.path.join(td.TD, "through_training", "dose")
RAW = {"trained": (0, 250, 500, 800, 1100, 1500, 2000, 3000, 5000, 10000, 20000, 49995),
       "echo": (0, 1100, 3000, 10000, 49995)}
FEATS = ("norm", "turn", "dim", "band")
WHERE = ("ent", "after", "fr", "rand")
DOSE = (.5, 2.)
TOP = .1                                                                                      # band 1: the top PCs
                                                                                              # holding 10% of the variance
LAG = 6                                                                                       # s .. s+5
S_LO, S_HI, AFTER = 1, 22, 2                                                                  # 0-based 'ent' steps


def plan(P, seed=0):
    """(4, n) 0-based manipulation step per episode for 'ent', 'after', 'fr', 'rand'; 'fr': one of the episode's steps
    in the highest fifth of FR_s = KL(O_s || N_s) (mean of the two variables) of its step"""
    import td_tt_order as to
    rng = np.random.default_rng(seed)

    def top_fifth(v):
        n = len(v)
        rk = np.argsort(np.argsort(v, 0), 0) / (n - 1.)                                       # within-step rank
        out = np.empty(n, int)
        for e in range(n):
            cand = np.flatnonzero(rk[e, S_LO:S_HI + 1] >= .8) + S_LO
            out[e] = rng.choice(cand) if len(cand) else S_LO + int(np.argmax(rk[e, S_LO:S_HI + 1]))
        return out
    s_ent = top_fifth(P["ent"].astype(np.float64).mean(-1))
    n = len(s_ent)
    s_rand = s_ent[rng.permutation(n)]
    s_fr = top_fifth(to.refs()["FR"].mean(-1))
    return np.stack([s_ent, s_ent + AFTER, s_fr, s_rand])


def _pcs(hn, B, t):
    """the deviation of u_t from the step's mean in the step's PCs (n, H), and its norm (n, 1)"""
    y = (hn - B["mu"][t]) @ B["V"][t]
    return y, y.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def feats_of(h, hn, B, t):
    """norm, turn, effective dimension (projection PR of the deviation on the step's PCs, ck 'effdim_u') and the top
    band's share of the deviation's energy, per episode (n, 4)"""
    y, yn = _pcs(hn, B, t)
    p2 = y ** 2
    return torch.stack([hn.norm(dim=-1),
                        1 - (hn * h).sum(-1) / (hn.norm(dim=-1) * h.norm(dim=-1)).clamp_min(1e-12),
                        p2.sum(-1) ** 2 / (p2 ** 2).sum(-1).clamp_min(1e-30),
                        (p2 * B["top"][t]).sum(-1) / p2.sum(-1).clamp_min(1e-30)], -1)


def manipulate(h, hn, feat, dose, B, t):
    """the changed state for every episode (the caller keeps it only at the chosen step)
    norm: u -> d u.  turn: the part of u orthogonal to u_t-1 scaled by d, then rescaled to ||u||.
    dim: the deviation from the step's mean, in the step's PCs y, -> sign(y) |y|^(1/d) rescaled to ||y|| (d > 1 spreads
    it over more PCs, d < 1 concentrates it; the norm of the deviation and the mean are kept).
    band: the deviation's part in the top band (the top PCs holding 10% of the step's variance) scaled by d, then the
    whole deviation rescaled to its old norm (energy moved between the top band and the rest)."""
    if feat == "norm":
        return hn * dose
    if feat == "turn":
        u = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        par = (hn * u).sum(-1, keepdim=True) * u
        hm = par + dose * (hn - par)
        return hm * hn.norm(dim=-1, keepdim=True) / hm.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    y, yn = _pcs(hn, B, t)
    if feat == "dim":
        ym = y.sign() * y.abs().clamp_min(1e-30) ** (1. / dose)
    else:
        ym = y * (1 + (dose - 1) * B["top"][t])
    ym = ym * yn / ym.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return B["mu"][t] + ym @ B["V"][t].T


@torch.no_grad()
def replay(m, P, dev, s=None, feat=None, dose=1., B=None):
    """log marginal beliefs (n, T, 2, R), the features feats_of (n, T, 4) of the (possibly changed) states and, for the
    intact replay (s None), the steps' PCA basis B = {mu (T, H), V (T, H, H) descending, top (T, H) band-1 indicator};
    s (n,): the step to change"""
    n = len(P["obs"])
    obs = torch.as_tensor(P["obs"], dtype=torch.float32, device=dev)
    zin = torch.as_tensor(P["Z"][P["env"]], device=dev).reshape(n, -1)
    W_ih, W_hh, b = td.ga.lstm_weights(m)
    h, c = td._start(m, n)
    st = None if s is None else torch.as_tensor(s, device=dev)
    if B is None:                                                                             # the intact pass: basis
        hs = []
        with torch.no_grad():
            hh, cc = h, c
            for t in range(td.T):
                hh, cc, _ = td._cell(td._readin(m, obs[:, t], zin) @ W_ih.T + b + hh @ W_hh.T, cc)
                hs.append(hh.clone())
        mu, V, top = [], [], []
        for t in range(td.T):
            x = hs[t].double()
            m_ = x.mean(0)
            ev, vec = torch.linalg.eigh(torch.cov(x.T))
            ev, vec = ev.flip(0).clamp_min(0), vec.flip(1)
            frac = ev.cumsum(0) / ev.sum()
            k = int((frac < TOP).sum()) + 1                                                   # PCs up to the one
            tp = torch.zeros_like(ev)                                                         # crossing 10%
            tp[:k] = 1
            mu.append(m_.float()), V.append(vec.float()), top.append(tp.float())
        del hs
        B = {"mu": torch.stack(mu), "V": torch.stack(V), "top": torch.stack(top)}
    lg, F = [], []
    for t in range(td.T):
        x = td._readin(m, obs[:, t], zin)
        hn, c, _ = td._cell(x @ W_ih.T + b + h @ W_hh.T, c)
        if st is not None:
            sel = (st == t)[:, None]
            if bool(sel.any()):
                hn = torch.where(sel, manipulate(h, hn, feat, dose, B, t), hn)
        F.append(feats_of(h, hn, B, t))
        lg.append(m.classifier_readout(hn))
        h = hn
    cum = torch.stack(lg, 1).double().cumsum(1)
    lq = td.marginals(cum, m.joint_out).double()
    return lq, torch.stack(F, 1).double().cpu().numpy(), B


def outcome(lq, lB, r, s):
    """KL(O || q) and accuracy at s + k (k = 0..5) and at T, per episode: (n, 7) each"""
    n = len(s)
    ar = torch.arange(n, device=lq.device)
    kl = (lB.exp() * (lB - lq)).sum(-1).mean(-1)                                              # (n, T)
    acc = (lq.argmax(-1) == r[:, None, :]).double().mean(-1)                                  # (n, T)
    idx = torch.as_tensor(np.minimum(s[:, None] + np.arange(LAG)[None], td.T - 1), device=lq.device)
    idx = torch.cat([idx, torch.full((n, 1), td.T - 1, device=lq.device)], 1)
    return (kl[ar[:, None], idx].cpu().numpy(), acc[ar[:, None], idx].cpu().numpy())


def main(run="trained", gpu=0):
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device(f"cuda:{gpu}")
    P = np.load(os.path.join(td.OUT, "probes.npz"))
    lB = torch.as_tensor(P["log_jb"], device=dev).double()
    lB = lB - lB.logsumexp(-1, keepdim=True)
    r = torch.as_tensor(P["r"], device=dev)
    S = plan(P)
    n = S.shape[1]
    raws = RAW[run]
    sh = (len(raws), len(FEATS), len(WHERE), len(DOSE), LAG + 1)
    res = {k: np.zeros(sh) for k in ("dkl", "dacc", "dkl_se", "dacc_se")}
    base_kl, base_acc = np.zeros((len(raws), len(WHERE), LAG + 1)), np.zeros((len(raws), len(WHERE), LAG + 1))
    feat_at = np.zeros((len(raws), len(WHERE), 4))                    # the intact norm, turn, eff. dim, top-band share
    feat_ratio = np.zeros(sh[:4] + (4,))                              # changed / intact, at the step, mean over episodes
    m = td.build(run, raws[0], dev)
    ar = np.arange(n)
    for ci, ep in enumerate(raws):
        t0 = time.time()
        td.load_weights(m, run, ep)
        lq0, F0, Bas = replay(m, P, dev)
        for wi in range(len(WHERE)):
            k0, a0 = outcome(lq0, lB, r, S[wi])
            base_kl[ci, wi], base_acc[ci, wi] = k0.mean(0), a0.mean(0)
            f0 = F0[ar, S[wi]]
            feat_at[ci, wi] = f0.mean(0)
            for fi, f in enumerate(FEATS):
                for di, d in enumerate(DOSE):
                    lq, F1, _ = replay(m, P, dev, S[wi], f, d, Bas)
                    feat_ratio[ci, fi, wi, di] = (F1[ar, S[wi]] / np.maximum(f0, 1e-12)).mean(0)
                    k1, a1 = outcome(lq, lB, r, S[wi])
                    for key, v in (("dkl", k1 - k0), ("dacc", a1 - a0)):
                        res[key][ci, fi, wi, di] = v.mean(0)
                        res[key + "_se"][ci, fi, wi, di] = v.std(0) / np.sqrt(n)
        i_e, i_f, i_r = WHERE.index("ent"), WHERE.index("fr"), WHERE.index("rand")
        print(f"{run} {ep}: ({time.time() - t0:.0f} s) intact KL at s {base_kl[ci, i_e, 0]:.4f}; dKL at s, ent / fr / "
              "rand: " + "; ".join(
                  f"{f} x{d}: {res['dkl'][ci, fi, i_e, di, 0]:+.4f} / {res['dkl'][ci, fi, i_f, di, 0]:+.4f} / "
                  f"{res['dkl'][ci, fi, i_r, di, 0]:+.4f}" for fi, f in enumerate(FEATS) for di, d in enumerate(DOSE)),
              flush=True)
        del Bas
        torch.cuda.empty_cache()
        np.savez(os.path.join(OUT, f"{run}.npz"), raw=np.array(raws[:ci + 1]), base_kl=base_kl[:ci + 1],
                 base_acc=base_acc[:ci + 1], feat_at=feat_at[:ci + 1], feat_ratio=feat_ratio[:ci + 1], plan=S,
                 dose=np.array(DOSE), feats=np.array(FEATS), where=np.array(WHERE),
                 **{k: v[:ci + 1] for k, v in res.items()})




def figure():
    """-> main/REPLICATION/figure_bin/learn_dose_pilot.png"""
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    TS, FL, FT = 11, 10, 9
    plt.rcParams.update({"xtick.labelsize": FT, "ytick.labelsize": FT})
    COL, NAME = {"trained": "#1f77b4", "echo": "#ff7f0e"}, {"trained": "Trained RNN", "echo": "ESN"}
    D = {r: dict(np.load(os.path.join(OUT, f"{r}.npz"))) for r in ("trained", "echo")}
    for r in D:
        D[r]["x"] = np.maximum(D[r]["raw"] - D[r]["raw"] // 5, 10).astype(float)
    W, H = 7.6, 5.55
    fig = plt.figure(figsize=(W, H))

    def add(l, t, w, h):
        return fig.add_axes([l / W, (H - t - h) / H, w / W, h / H])

    def xaxis(ax, label=True):
        ax.set_xscale("log")
        ax.set_xticks([10, 1e2, 1e3, 1e4])
        ax.set_xticklabels(["0", "100", "1k", "10k"])
        ax.set_xlim(8, 5e4)
        ax.minorticks_off()
        ax.axvline(880, color="0.5", lw=.8, ls=":")
        if label:
            ax.set_xlabel("training episodes", fontsize=FL, labelpad=1)

    fi_n, fi_t, w_e, w_r = FEATS.index("norm"), FEATS.index("turn"), WHERE.index("ent"), WHERE.index("rand")
    d2, d05 = DOSE.index(2.), DOSE.index(.5)
    A = [[add(.62 + j * 3.75, .55 + i * 2.45, 2.95, 1.7) for j in range(2)] for i in range(2)]
    # a: the effect over the steps after the manipulation
    a = A[0][0]
    xs = list(range(LAG)) + [LAG + .8]
    tr = D["trained"]
    shades = plt.cm.Blues(np.linspace(.4, 1., 5))
    for k, raw in enumerate((800, 1100, 3000, 10000, 49995)):
        ci = list(tr["raw"]).index(raw)
        a.plot(xs, tr["dkl"][ci, fi_n, w_e, d2], "-o", color=shades[k], ms=2.5, lw=1.3,
               label="40k" if raw == 49995 else f"{raw - raw // 5:,}")
    ci = len(D["echo"]["raw"]) - 1
    a.plot(xs, D["echo"]["dkl"][ci, fi_n, w_e, d2], "-o", color=COL["echo"], ms=2.5, lw=1.3, label="ESN, 40k")
    a.set_xticks(xs)
    a.set_xticklabels(["s", "+1", "+2", "+3", "+4", "+5", "T"])
    a.set_xlabel("step after the manipulated step $s$", fontsize=FL, labelpad=1)
    a.set_ylabel(r"$\Delta\,\mathrm{KL}(O\Vert q)$ (nats)", fontsize=FL, labelpad=1)
    a.set_ylim(0, .33)
    a.legend(title="Trained RNN at (training episodes)", title_fontsize=FT - .5, fontsize=FT - .5, frameon=False,
             loc="upper center", ncol=3, handlelength=1.4, columnspacing=.8, labelspacing=.2, borderaxespad=.1)
    a.set_title(r"$\mathbf{a}$   The network corrects a one-step perturbation", loc="left", fontsize=TS, pad=14)
    a.text(0, 1.02, r"state $\times 2$ at one entangled step; error vs the intact replay", transform=a.transAxes,
           fontsize=FT - .5)
    # b: at the step and at T, through training
    a = A[0][1]
    for r in D:
        a.plot(D[r]["x"], D[r]["dkl"][:, fi_n, w_e, d2, 0], "-o", color=COL[r], ms=2.5, lw=1.4)
        a.plot(D[r]["x"], D[r]["dkl"][:, fi_n, w_e, d2, -1], "--o", color=COL[r], ms=2.5, lw=1.4, mfc="w")
    a.set_ylim(0, .33)
    a.legend(ncol=2, handles=[Line2D([], [], color="0.3", lw=1.4, marker="o", ms=2.5, label="at the manipulated step"),
                      Line2D([], [], color="0.3", lw=1.4, ls="--", marker="o", ms=2.5, mfc="w", label="at T"),
                      Line2D([], [], color=COL["trained"], lw=2, label=NAME["trained"]),
                      Line2D([], [], color=COL["echo"], lw=2, label=NAME["echo"])],
             fontsize=FT - .5, frameon=False, loc="upper center", labelspacing=.2, borderaxespad=.1, columnspacing=.8,
             handlelength=1.8)
    xaxis(a, label=False)
    a.set_title(r"$\mathbf{b}$   Reading at T hides most of the effect", loc="left", fontsize=TS, pad=14)
    a.text(0, 1.02, r"as a, through training", transform=a.transAxes, fontsize=FT - .5)
    # c, d: at the step, entangled vs random steps
    for a, fi, di, ttl, sub in ((A[1][0], fi_n, d05, r"$\mathbf{c}$   Halving the state at one step",
                                 "error at that step; below 0: the step was too large"),
                                (A[1][1], fi_t, d2, r"$\mathbf{d}$   Doubling the turn at one step",
                                 "error at that step; below 0: more turn would help")):
        for r in D:
            a.plot(D[r]["x"], D[r]["dkl"][:, fi, w_e, di, 0], "-o", color=COL[r], ms=2.5, lw=1.4)
            a.plot(D[r]["x"], D[r]["dkl"][:, fi, w_r, di, 0], ":o", color=COL[r], ms=2.5, lw=1.4, mfc="w")
        a.axhline(0, color="0.6", lw=.8)
        xaxis(a)
        a.set_ylabel(r"$\Delta\,\mathrm{KL}(O_s\Vert q_s)$ (nats)", fontsize=FL, labelpad=1)
        a.set_title(ttl, loc="left", fontsize=TS, pad=14)
        a.text(0, 1.02, sub, transform=a.transAxes, fontsize=FT - .5)
    A[1][0].set_yscale("symlog", linthresh=.02, linscale=1)
    A[1][0].set_ylim(-.12, .045)
    A[1][0].set_yticks([-.1, -.02, 0, .02])
    A[1][0].set_yticklabels(["-.1", "-.02", "0", ".02"])
    A[1][1].set_ylim(-.036, .026)
    A[1][1].legend(handles=[Line2D([], [], color="0.3", lw=1.4, marker="o", ms=2.5, label="entangled step"),
                            Line2D([], [], color="0.3", lw=1.4, ls=":", marker="o", ms=2.5, mfc="w",
                                   label="random step")], fontsize=FT - .5, frameon=False, loc="lower right",
                   labelspacing=.25, borderaxespad=.1)
    fig.text(.5, .08 / H, "10,000 probe episodes per checkpoint, one manipulated step each (entangled: one of the "
             "episode's steps 2-23 in the most entangled fifth of its step;\nrandom: the same step indices in other "
             "episodes), paired with the intact replay.  Dotted vertical: the Trained RNN passes naive (880).",
             ha="center", va="bottom", fontsize=FT - 1.5, color="0.3", linespacing=1.4)
    out = fig_path("learn_dose_pilot.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    figure() if a == ["figure"] else main(a[0] if a else "trained", int(a[1]) if len(a) > 1 else 0)


def figure_contenders():
    """-> main/REPLICATION/figure_bin/learn_dose_contenders.png: every activity contender of fig_learn_maps changed at
    one step per episode, the change of the belief error at that step through training (Trained RNN solid, ESN dashed),
    at entangled (purple), high-FR (green) and random (grey) steps; top row: less of the feature, bottom row: more"""
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    TS, FL, FT = 10, 9, 8
    plt.rcParams.update({"xtick.labelsize": FT, "ytick.labelsize": FT})
    D = {r: dict(np.load(os.path.join(OUT, f"{r}.npz"))) for r in ("trained", "echo")}
    for r in D:
        D[r]["x"] = np.maximum(D[r]["raw"] - D[r]["raw"] // 5, 10).astype(float)
    WC = {"ent": "#7b4fa6", "fr": "#2a9d6e", "rand": "0.55"}
    WN = {"ent": "entangled step", "fr": "high-FR step", "rand": "random step"}
    COLS = (("norm", "a", "Activity norm", r"$\Vert u_s\Vert$", (r"state $\times\,0.5$", r"state $\times\,2$")),
            ("turn", "b", "Activity turn", r"$1-\cos(u_s,u_{s-1})$",
             (r"turn part $\times\,0.5$", r"turn part $\times\,2$")),
            ("dim", "c", "Effective dimension", r"PR of $u_s$ over the PCs",
             (r"concentrated (dim. $\times\,0.2$)", r"spread (dim. $\times\,9$)")),
            ("band", "d", "Top-band share", r"share of $u_s$ in top PCs",
             (r"top band $\times\,0.5$ (share $\times\,0.3$)", r"top band $\times\,2$ (share $\times\,3$)")))
    W, H = 7.7, 5.2
    fig = plt.figure(figsize=(W, H))
    PW, PH, L0, GX, T0, GY = 1.42, 1.2, .62, .42, 1.12, .78

    def add(j, i):
        return fig.add_axes([(L0 + j * (PW + GX)) / W, (H - T0 - i * (PH + GY) - PH) / H, PW / W, PH / H])
    for j, (f, let, name, form, rows) in enumerate(COLS):
        fi = list(D["trained"]["feats"]).index(f)
        for i, d in enumerate((.5, 2.)):
            a = add(j, i)
            di = list(D["trained"]["dose"]).index(d)
            for r, ls, mk in (("trained", "-", None), ("echo", (0, (3, 1.5)), "o")):
                wl = list(D[r]["where"])
                for w in ("rand", "fr", "ent"):
                    a.plot(D[r]["x"], D[r]["dkl"][:, fi, wl.index(w), di, 0], color=WC[w], ls=ls, lw=1.3, marker=mk,
                           ms=2.5, mfc="w")
            a.axhline(0, color="0.75", lw=.6, zorder=0)
            a.axvline(880, color="0.5", lw=.7, ls=":")
            a.set_xscale("log")
            a.set_xticks([10, 1e2, 1e3, 1e4])
            a.set_xticklabels(["0", "100", "1k", "10k"])
            a.set_xlim(8, 5e4)
            a.minorticks_off()
            a.text(0, 1.03, rows[i], transform=a.transAxes, fontsize=FT - .5, va="bottom", color="0.25")
            if j == 0:
                a.set_ylabel(r"$\Delta\,\mathrm{KL}(O_s\Vert q_s)$ (nats)", fontsize=FL, labelpad=1)
            if i == 1:
                a.set_xlabel("training episodes", fontsize=FL, labelpad=1)
            if i == 0:
                a.set_title(r"$\mathbf{%s}$   %s" % (let, name) + "\n" + form, loc="left", fontsize=TS, pad=15,
                            linespacing=1.3)
    fig.legend(handles=[Line2D([], [], color=WC[w], lw=2, label=WN[w]) for w in ("ent", "fr", "rand")]
               + [Line2D([], [], color="0.3", lw=1.3, label="Trained RNN"),
                  Line2D([], [], color="0.3", lw=1.3, ls=(0, (3, 1.5)), marker="o", ms=2.5, mfc="w", label="ESN")],
               loc="upper center", bbox_to_anchor=(.5, 1 - .05 / H), ncol=5, frameon=False, fontsize=FL,
               handlelength=2.2, columnspacing=1.4)
    fig.text(.5, .06 / H, "10,000 probe episodes per checkpoint; one manipulated step per episode (steps 2-23: in the "
             "most entangled / highest-FR fifth of its step, or a random step),\npaired with the intact replay; error "
             "read at the manipulated step; above 0: the change hurts.  Dotted: the Trained RNN passes naive (880).\n"
             "Dimension and top band on the PCA of each step's states over the 10,000 probes; the deviation's norm "
             "kept.",
             ha="center", va="bottom", fontsize=FT - 1, color="0.3", linespacing=1.4)
    out = fig_path("learn_dose_contenders.png")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print("saved", out)
