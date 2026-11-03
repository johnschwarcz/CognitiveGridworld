"""Entanglement and the expansion of small differences (John 11-03: "Can you please try to show what you are saying
about the entanglement dependent amplification of small divergences.")

-> main/REPLICATION/figure_bin/td_tt_chaos_ent_small.png; CPU, ~1 min.

Close pairs only: two beliefs 0.25-1 nat apart (d_H before) given the same observation (the pairs of
td_tt_chaos_combined.py, end of training), split into fifths of the observation's entanglement. Every value is matched on
step and distance bin (td_tt_chaos_pairs.matched: cells (step, bin) with >= 10 cases in every fifth, weighted by their
counts) with 5-95% over the 1,000 environment resamples.
  Row 1: the distance ratio d_H after / d_H before (geometric mean): actual (solid) and with the update difference of
    another case from the same fifth, with a random sign (dashed; td_tt_chaos_pairs's shuffle, 5 seeds).
  Row 2: how different the two beliefs' updates are: d_H of the update difference (log p_t - log p_{t-1}) - (log q_t -
    log q_{t-1}) (nats; 0 for naive, whose update is the same for both).
RESULTS (cells cover .59 / .73 / .90 of the close cases; Bayes / Trained RNN / ESN)
  Close pairs grow more after entangled observations in Bayes (x1.39 least -> x1.89 most entangled fifth) and the
  Trained RNN (x1.38 -> x2.04), barely in the ESN (x1.32 -> x1.39). The two beliefs' updates differ more after entangled
  observations, and a random-direction update difference of the same size reproduces most of the rise (shuffled x1.19
  -> x1.51, x1.28 -> x1.75). Beyond the shuffle the factor hardly depends on entanglement: x1.17 -> x1.25 [x.96, x1.23
  for most / least], x1.08 -> x1.16 [x.97, x1.19]; the ESN's excess (x1.21 -> x1.19) is the same at every level.
"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_chaos_combined as cc                                                             # noqa: E402
import td_tt_chaos_pairs as tp                                                                # noqa: E402

AGENTS = ("bayes", "trained", "echo")
BAND = (.25, 1.)


def compute():
    """per agent: matched fifth means (5, 1 + N_BOOT) of ln r (actual), ln r (shuffled), ln r actual - shuffled, and the
    update difference's d_H"""
    P = cc.pairs()
    e40, nb, W = P["edges40"], 40, cc._W()
    out = {}
    with np.load(cc.UNITS) as Z, np.load(tp.SHUF40_CACHE) as SH, np.load(tp.LQ_CACHE) as LQ:
        I, J, S = Z["I"], Z["J"], Z["t"]
        m = len(I)
        env, t_u = np.repeat(Z["env"], 2), np.repeat(S, 2)
        g5 = np.repeat(SH["g5"].astype(np.int64), 2)
        for k in AGENTS:
            pre, post = Z[f"pre_{k}"].ravel(), Z[f"post_{k}"].ravel()
            b = tp.unit_bins(pre, e40)
            band = (b >= 0) & (b < nb) & (pre >= BAND[0]) & (pre < BAND[1])
            lq = LQ[k]
            D = lq[I, S - 2] - lq[J, S - 2]
            dl = (lq[I, S - 1] - lq[I, S - 2]) - (lq[J, S - 1] - lq[J, S - 2])
            gid = ((g5 * tp.OTH_K + SH[f"cond_{k}"].ravel().astype(np.int64)) * (nb + 2) + b + 1).reshape(m, 2)
            acc = np.zeros((m, 2))
            for sd in tp.SHUF40_SEED + np.arange(tp.N_SHUF):
                r_ = np.random.default_rng(int(sd))
                for v in range(2):
                    perm = tp._within_perm(gid[:, v], r_)
                    sgn = r_.choice([-1., 1.], m)
                    acc[:, v] += np.log(np.maximum(tp.osc(D[:, v] + sgn[:, None] * dl[perm, v]), 1e-300) /
                                        pre.reshape(m, 2)[:, v])
            lr, ls = np.log(post / pre), (acc / tp.N_SHUF).ravel()
            cell = t_u * nb + b
            out[k] = {nm: tp.matched(v, g5, cell, band, env, W, n_min=10, K=5) for nm, v in
                      (("actual", lr), ("shuffled", ls), ("excess", lr - ls), ("update", tp.osc(dl).ravel()))}
    return out


def figure(path=None):
    import matplotlib.pyplot as plt
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    M = compute()
    xs = np.arange(1, 6)
    L10 = np.log(10.)
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), gridspec_kw=dict(hspace=.42, wspace=.25, left=.07, right=.98,
                                                                     top=.86, bottom=.1))

    def iv(a):
        return a[:, 0], np.percentile(a[:, 1:], 5, 1), np.percentile(a[:, 1:], 95, 1)
    for j, k in enumerate(AGENTS):
        ax = axs[0, j]
        for nm, ls, col, lab in (("actual", "-", cc.COL[k], f"{cc.NAME[k]}"),
                                 ("shuffled", "--", cc.SHUF_C, "random-direction update of the same size")):
            v, lo, hi = (a / L10 for a in iv(M[k][nm]["std"]))
            ax.fill_between(xs, lo, hi, color=col, alpha=.15, lw=0)
            ax.plot(xs, v, ls, color=col, lw=2.2, marker="o", ms=5, label=lab)
        e = iv(M[k]["excess"]["std"])
        ax.text(.03, .97, f"actual / random: x{np.exp(e[0][0]):.2f} (least) " + r"$\rightarrow$" +
                f" x{np.exp(e[0][4]):.2f} (most)",
                transform=ax.transAxes, ha="left", va="top", fontsize=10)
        cc._ratio_y(ax, np.log10(1.), np.log10(2.5), ticks=(1, 1.25, 1.5, 2, 2.5))
        ax.set_xticks(xs)
        ax.set_xticklabels(["1\nleast", "2", "3", "4", "5\nmost"])
        ax.set_xlabel("entanglement of the observation (fifth)")
        if j == 0:
            ax.set_ylabel("distance ratio, after / before")
        ax.legend(frameon=False, fontsize=9, loc="center right" if k == "echo" else "lower right")
        ax.set_title(f"{cc.NAME[k]}: how much close pairs grow", loc="left", fontsize=12)
        ax = axs[1, j]
        v, lo, hi = iv(M[k]["update"]["std"])
        ax.fill_between(xs, lo, hi, color=cc.COL[k], alpha=.2, lw=0)
        ax.plot(xs, v, "-o", color=cc.COL[k], lw=2.2, ms=5)
        ax.set_ylim(0, None)
        ax.set_xticks(xs)
        ax.set_xticklabels(["1\nleast", "2", "3", "4", "5\nmost"])
        ax.set_xlabel("entanglement of the observation (fifth)")
        if j == 0:
            ax.set_ylabel(r"$d_H$ of the update difference (nats)")
        ax.set_title(f"{cc.NAME[k]}: how different the two updates are", loc="left", fontsize=12)
    fig.suptitle("Close pairs (0.25-1 nat apart) given the same observation: after entangled observations the two beliefs' "
                 "updates differ more,\nso close pairs grow more; a random-direction update of the same size grows them "
                 "almost as much (naive: updates identical, ratio 1)", fontsize=12.5)
    fig.text(.5, .01, "End of training; pairs: two probes of one test environment with the same observation at step t; "
             "matched on step and distance; bands: 5-95% over 1,000 environment resamples; cases 0.25-1 nat: Bayes 8,301, "
             "Trained RNN 12,696, ESN 47,402.", ha="center", fontsize=9)
    path = fig_path("td_tt_chaos_ent_small.png") if path is None else path
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)
    for k in AGENTS:
        print(k, {nm: [round(float(np.exp(M[k][nm]["std"][g, 0])) if nm != "update" else
                             float(M[k][nm]["std"][g, 0]), 3) for g in range(5)] for nm in M[k]})


if __name__ == "__main__":
    figure()
