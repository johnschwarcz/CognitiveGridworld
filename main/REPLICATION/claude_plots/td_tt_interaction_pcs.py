"""Do the latent interactions live in high- or low-variance components of the short-term memory stream? (John 11-03)

John 11-03 (verbatim): "The interaction direction analyses is interesting. I dont think we need to constrain it to only
top dimensions of interaction, but it is probably important to use the same directions at every step of inference. It
would be interesting to see whether the interactions come from high or low variance components of the short term memory
stream. Can we compute this without an expensive recollection?"

-> main/REPLICATION/figure_bin/td_tt_interaction_pcs.png (figure()); numbers(); caches through_training/interaction_pcs/
   <run>_EEEEE.npz (harvest(): one clean forward pass per checkpoint, statistics accumulated on the GPU, ~4 s each; no
   lesion runs; the stored analysis files hold only per-step traces and a top-100 basis, not the directions needed).

DEFINITIONS (Trained RNN, 10,000 probes = 50 test environments x 100 cells (r0, r1) x 2 streams, steps 1-30)
  u_s = the LSTM's short-term memory h after step s (1,000 units). Fixed directions: the eigenvectors v_k of
    C_w = (1/30) Sum_s C_s, the within-step covariance of u averaged over the steps (the same directions at every step),
    ordered by variance lambda_k (k = 1 the largest). Robustness: the eigenvectors of the pooled covariance (with the
    drift of the step means).
  Factors per step (two-way, balanced, td_analysis.factor_covs): cell means M(r0, r1) of u (pooled over environments and
    streams); r0 / r1 main effects = row / column means - grand mean; interaction = M - row - column + grand mean (all of
    it, not its top dimensions). Its variance along v_k: mean over the 100 cells of (E . v_k)^2.
  Noise-free estimate: the cell means of two disjoint environment halves A, B (25 + 25 environments, 50 probes per cell
    each) give mean_cells (E_A . v_k)(E_B . v_k), unbiased for the interaction shared across environments (environment x
    cell and within-cell noise cancel). Two splits: halves 0-24 / 25-49 and even / odd environments; drawn: their mean.
WHAT IS DRAWN: a cumulative shares over the fixed directions (rank on a log axis) at the end of training: total variance,
  main effects (r0 + r1), interaction (each split thin); b through training: the variance-ranked direction below which
  half (and 90%) of each lies; c the interaction's share of each direction's variance, log-binned ranks, 4 stages;
  d at the end, per step: the median rank of the interaction and of the total variance.
RESULTS (numbers(); all 114 checkpoints; checks: the full-sample interaction and total traces equal the stored
  analysis traces, ck fac, to <= 1.9e-7 relative)
  The interaction lives in HIGH-variance directions, more concentrated than the variance itself. End of training: half
  of it in the top 5.5 of 1,000 directions (total variance 26.6; main effects 1.5), 90% in 77 (total 222, main effects
  27); the top 10 directions hold .58 of the interaction and .30 of the variance; its share of a direction's variance
  peaks at ranks 4-6 (~.23; main effects dominate ranks 1-2). The two environment splits agree (5.5 / 5.5); the basis
  with the step drift gives 5.9. Through training: up to ~100 episodes all three sit in direction 1; from ~300 the
  variance spreads (half in 2.5 directions at 320, 13 at 880, 21 at 1,680, 27 at the end) and the interaction less
  (2.1, 3.6, 5.1, 5.5). Per step at the end, from step ~4 on, half the interaction lies in the top ~5-7 directions.
  The interaction is .02-.04 of u's variance (noise-free; the same-half estimate adds ~.01 of noise).
"""
import os
import re
import sys
import glob
import time
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402

OUT = os.path.join(gap.TT, "interaction_pcs")
T, R, H = 30, 10, 1000
STAGES = (100, 880, 4200, 39996)                       # training episodes for the per-stage curves


def _file(run, raw):
    return os.path.join(OUT, f"{run}_{raw:05d}.npz")


def harvest_one(run, raw, dev):
    import torch
    import td_tt_intervene as ti
    import td_tt_lesions as tl
    t0 = time.time()
    w = ti.load(run, raw)
    P = ti.probes()
    cell, env = np.asarray(P["cell"], np.int64), np.asarray(P["env"], np.int64)
    r = np.asarray(P["r"], np.int64)
    assert np.array_equal(cell, r[:, 0] * R + r[:, 1])                             # cell = (r0, r1), row-major
    side = np.stack([env >= gap.N_ENV // 2, env % 2 == 1])                           # (2 splits, n): half B
    for sp in range(2):
        for b in (False, True):
            assert (np.bincount(cell[side[sp] == b], minlength=R * R) == 50).all()
    f64 = torch.float64
    cell_t, side_t = torch.as_tensor(cell, device=dev), torch.as_tensor(side, device=dev)
    S1 = torch.zeros(T + 1, H, dtype=f64, device=dev)
    S2 = torch.zeros(T + 1, H, H, dtype=f64, device=dev)
    CS = torch.zeros(T + 1, 2, 2, R * R, H, dtype=f64, device=dev)

    def rec(s, rows, st):
        rr = torch.as_tensor(rows, device=dev)
        u = st["u"].double()
        S1[s] += u.sum(0)
        S2[s] += u.T @ u
        for sp in range(2):
            b = side_t[sp, rr]
            CS[s, sp, 0].index_add_(0, cell_t[rr][~b], u[~b])
            CS[s, sp, 1].index_add_(0, cell_t[rr][b], u[b])
    tl.lforward(w, None, None, dev=dev, record=rec, budget=5000)
    n = len(cell)
    m = S1[1:] / n                                                                   # (T, H)
    C = S2[1:] / n - m[:, :, None] * m[:, None, :]                                   # (T, H, H)
    Cw = C.mean(0)
    lam, V = torch.linalg.eigh(Cw)
    lam, V = lam.flip(0), V.flip(1)
    md = m - m.mean(0)
    lamp, Vp = torch.linalg.eigh(Cw + md.T @ md / T)
    lamp, Vp = lamp.flip(0), Vp.flip(1)
    lam_s = torch.einsum("hk,thg,gk->tk", V, C, V)                                    # per-step variance along v_k

    def effects(M):
        """two-way effects of cell means (100, H): r0 rows (R, H), r1 columns (R, H), interaction (100, H)"""
        Mg = M.reshape(R, R, H)
        g = Mg.mean((0, 1))
        a0, a1 = Mg.mean(1) - g, Mg.mean(0) - g
        return a0, a1, (Mg - a0[:, None] - a1[None] - g).reshape(R * R, H)
    out = {k: np.zeros((2, T, H), np.float32) for k in ("int_x", "int_same", "r0_x", "r1_x")}
    out["intp_x"] = np.zeros((2, H))
    tr_full = np.zeros(T)
    for s in range(1, T + 1):
        for sp in range(2):
            ea, eb = effects(CS[s, sp, 0] / 50), effects(CS[s, sp, 1] / 50)
            for key, i in (("r0_x", 0), ("r1_x", 1), ("int_x", 2)):
                out[key][sp, s - 1] = ((ea[i] @ V) * (eb[i] @ V)).mean(0).cpu().numpy()
            out["int_same"][sp, s - 1] = (((ea[2] @ V) ** 2 + (eb[2] @ V) ** 2) / 2).mean(0).cpu().numpy()
            out["intp_x"][sp] += ((ea[2] @ Vp) * (eb[2] @ Vp)).mean(0).cpu().numpy() / T
        ef = effects((CS[s, 0, 0] + CS[s, 0, 1]) / 100)[2]
        tr_full[s - 1] = float((ef ** 2).sum(1).mean())
    ckf = os.path.join(ti.AN, run, f"ck_{raw:05d}.npz")                           # optional check (analysis cache)
    chk_joint = chk_total = np.nan
    if os.path.exists(ckf):
        with np.load(ckf) as z:
            fr, fa = list(z["fac_rows"]), list(z["factors"])
            ref = z["fac"][:, fr.index("u"), fa.index("joint"), 0]
            reft = z["fac"][:, fr.index("u"), fa.index("total"), 0]
        chk_joint = float(np.abs(tr_full - ref).max() / ref.max())
        chk_total = float(np.abs(C.diagonal(dim1=1, dim2=2).sum(1).cpu().numpy() - reft).max() / reft.max())
    o = dict(out, raw=raw, x=float(gap.train_x(raw)), lam=lam.cpu().numpy(), lamp=lamp.cpu().numpy(),
             lam_s=lam_s.cpu().numpy().astype(np.float32), chk_joint=chk_joint, chk_total=chk_total,
             seconds=time.time() - t0)
    os.makedirs(OUT, exist_ok=True)
    tmp = _file(run, raw)[:-4] + ".tmp.npz"
    np.savez(tmp, **o)
    os.replace(tmp, _file(run, raw))
    return o


def harvest(part=0, nparts=1, run="trained"):
    import td_tt_intervene as ti
    dev = ti._device()
    todo = [r for r in ti.checkpoints(run)[part::nparts] if not os.path.exists(_file(run, r))]
    print(f"{run}: {len(todo)} checkpoints (part {part} of {nparts})", flush=True)
    for raw in todo:
        if os.path.exists(_file(run, raw)):
            continue
        o = harvest_one(run, raw, dev)
        print(f"{time.strftime('%H:%M:%S')} {raw} ({o['x']:,.0f}): {o['seconds']:.1f} s; joint trace vs ck fac rel "
              f"{o['chk_joint']:.1e}, total {o['chk_total']:.1e}", flush=True)
    print("done", flush=True)


def load(run="trained"):
    pat = re.compile(rf"^{re.escape(run)}_(\d{{5}})\.npz$")
    raws = sorted(int(m_.group(1)) for m_ in (pat.match(os.path.basename(f)) for f in glob.glob(os.path.join(OUT, "*")))
                  if m_)
    keys = ("lam", "lamp", "lam_s", "int_x", "int_same", "r0_x", "r1_x", "intp_x", "chk_joint", "chk_total")
    Z = {k: [] for k in keys}
    for raw in raws:
        with np.load(_file(run, raw)) as f:
            for k in keys:
                Z[k].append(f[k])
    Z = {k: np.stack(v) for k, v in Z.items()}
    Z.update(raw=np.array(raws), x=gap.train_x(np.array(raws)))
    return Z


def profiles(Z):
    """per checkpoint, over the fixed directions (K, H): total variance, main effects, interaction (mean of the two
    splits' noise-free estimates), each averaged over the steps; and the interaction per split"""
    tot = Z["lam"]
    main = (Z["r0_x"] + Z["r1_x"]).mean((1, 2))
    inter = Z["int_x"].mean((1, 2))
    return tot, main, inter, Z["int_x"].mean(2)


def rank_at(p, q):
    """the direction rank (1-based, fractional by linear interpolation) below which the share q of p's sum lies"""
    c = np.cumsum(p, -1) / p.sum(-1, keepdims=True)
    k = np.arange(1, p.shape[-1] + 1)
    return np.array([np.interp(q, cc, k) for cc in c.reshape(-1, p.shape[-1])]).reshape(p.shape[:-1])


def figure(path=None):
    import matplotlib.pyplot as plt
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    Z = load()
    tot, main, inter, isp = profiles(Z)
    x = Z["x"]
    k = np.arange(1, H + 1)
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.8), gridspec_kw=dict(wspace=.3, left=.05, right=.98, top=.8, bottom=.16))
    i = len(x) - 1
    for p, col, lb in ((tot[i], "0.5", "total variance"), (main[i], "#2166ac", "main effects (r0 + r1)"),
                       (inter[i], "#b2182b", "interaction (r0 x r1)")):
        ax[0].plot(k, np.cumsum(p) / p.sum(), color=col, lw=2.2, label=lb)
    for sp in range(2):
        ax[0].plot(k, np.cumsum(isp[i, sp]) / isp[i, sp].sum(), color="#b2182b", lw=.8, ls=":")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("fixed direction, ranked by variance (1 = largest)")
    ax[0].set_ylabel("cumulative share")
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    ax[0].set_title(f"a  End of training ({x[i]:,.0f} episodes)\ncumulative share over the fixed directions", loc="left",
                    fontsize=11)
    lx = np.log10(x)
    for p, col in ((tot, "0.5"), (main, "#2166ac"), (inter, "#b2182b")):
        ax[1].plot(lx, rank_at(p, .5), color=col, lw=2)
        ax[1].plot(lx, rank_at(p, .9), color=col, lw=1.2, ls="--")
    ax[1].set_yscale("log")
    ax[1].set_xticks([1, 2, 3, 4])
    ax[1].set_xticklabels(["0", "100", "1k", "10k"])
    ax[1].set_xlabel("training episodes")
    ax[1].set_ylabel("direction rank holding 50% (solid) / 90% (dashed)")
    ax[1].set_title("b  Through training: where each part lies\n(grey total, blue main effects, red interaction)",
                    loc="left", fontsize=11)
    edges = np.unique(np.round(np.logspace(0, 3, 16)).astype(int))
    cols = plt.cm.viridis(np.linspace(0, .85, len(STAGES)))
    for c_, st in zip(cols, STAGES):
        j = int(np.argmin(np.abs(x - st)))
        sh = [inter[j, a - 1:b - 1].sum() / tot[j, a - 1:b - 1].sum() for a, b in zip(edges[:-1], edges[1:])]
        ax[2].plot(np.sqrt(edges[:-1] * (edges[1:] - 1).clip(1)), sh, "o-", color=c_, ms=3, lw=1.6,
                   label=f"{x[j]:,.0f} episodes")
    ax[2].set_xscale("log")
    ax[2].axhline(0, color="k", lw=.6)
    ax[2].set_xlabel("fixed direction rank (log bins)")
    ax[2].set_ylabel("interaction / total variance")
    ax[2].legend(frameon=False, fontsize=9)
    ax[2].set_title("c  The interaction's share of each\ndirection's variance", loc="left", fontsize=11)
    st_ = np.arange(1, T + 1)
    ax[3].plot(st_, rank_at(Z["lam_s"][i], .5), color="0.5", lw=2, label="total variance")
    ax[3].plot(st_, rank_at(Z["int_x"][i].mean(0), .5), color="#b2182b", lw=2, label="interaction")
    ax[3].plot(st_, rank_at((Z["r0_x"][i] + Z["r1_x"][i]).mean(0), .5), color="#2166ac", lw=2, label="main effects")
    ax[3].set_yscale("log")
    ax[3].set_xlabel("step of the episode")
    ax[3].set_ylabel("direction rank holding 50%")
    ax[3].legend(frameon=False, fontsize=9)
    ax[3].set_title("d  End of training, per step\n(the same directions at every step)", loc="left", fontsize=11)
    fig.suptitle("Is the latent interaction in high- or low-variance directions of the short-term memory (h)? Trained "
                 "RNN; fixed directions = PCs of the within-step covariance, the same at every step", fontsize=12.5)
    fig.text(.5, .01, f"10,000 probes (50 test environments x 100 cells x 2 streams) x 30 steps per checkpoint; "
             f"{len(x)} checkpoints. Interaction and main effects: noise-free cross-products of two disjoint "
             "environment halves (two splits; dotted in a).", ha="center", fontsize=9)
    path = fig_path("td_tt_interaction_pcs.png") if path is None else path
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)


def numbers():
    Z = load()
    tot, main, inter, isp = profiles(Z)
    x = Z["x"]
    print(f"{len(x)} checkpoints; checks: joint trace vs ck fac rel max {Z['chk_joint'].max():.1e}, total "
          f"{Z['chk_total'].max():.1e}")
    for st in (20, 40, 100, 320, 880, 1680, 4200, 39996):
        j = int(np.argmin(np.abs(x - st)))
        r50 = [rank_at(p[j], .5) for p in (tot, main, inter)]
        r90 = [rank_at(p[j], .9) for p in (tot, main, inter)]
        top = [p[j, :k_].sum() / p[j].sum() for p in (tot, inter) for k_ in (10, 100)]
        sp = [rank_at(isp[j, s], .5) for s in range(2)]
        pr = rank_at(Z["intp_x"][j].mean(0), .5)
        print(f"  {x[j]:7,.0f}: rank holding 50% total / main / interaction {r50[0]:.1f} / {r50[1]:.1f} / {r50[2]:.1f} "
              f"(splits {sp[0]:.1f}, {sp[1]:.1f}; pooled-drift basis {pr:.1f}); 90% {r90[0]:.0f} / {r90[1]:.0f} / "
              f"{r90[2]:.0f}; top 10 / 100 directions hold total {top[0]:.2f} / {top[1]:.2f}, interaction {top[2]:.2f} / "
              f"{top[3]:.2f}; interaction / total {inter[j].sum() / tot[j].sum():.3f} (same-half, noisy "
              f"{Z['int_same'][j].mean((0, 1)).sum() / tot[j].sum():.3f})")


if __name__ == "__main__":
    what = sys.argv[1:] or ["figure", "numbers"]
    if "one" in what:
        import td_tt_intervene as ti
        harvest_one("trained", int(what[what.index("one") + 1]), ti._device())
    if "harvest" in what:
        i = what.index("harvest")
        harvest(int(what[i + 1]), int(what[i + 2]))
    if "figure" in what:
        figure()
    if "numbers" in what:
        numbers()
