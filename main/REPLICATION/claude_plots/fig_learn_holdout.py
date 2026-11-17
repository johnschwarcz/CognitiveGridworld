"""fig_learn_holdout.png: does learning transfer across a factorization-regret (FR) threshold, and in which direction?
John 11-10: "Could we show that blocking learning when FR is above .1 does not Generalize (the network fails at high
FR) but blocking when FR is below .1 does (the network succeeds at low FR). With balanced number of blocked steps of
course." Then: "lets keep just panels a and b, but turn them into 2 rows and have N columns for different thresholds of
FR that make significant differences (remember to balance the number of steps and episodes etc). also lets span a
greater range of FR on the x axis." After the .1 and 1 columns (12:25): "I think the only other theta we need is a
smaller one, maybe 0.01 nats" (the 3 / .3 / 10 runs, queue6, stopped; .01 and the .1 random block: learnmask_queue7.sh).

Networks: Trained RNNs trained from initialization in the first world (train_learn_masks.py, scratch, seed 20261029,
batch 8,000, excl_*), two per FR threshold theta (THRS, nats). Blocked = held out of training: the update's loss term is
dropped and the gradient is stopped at the update (read-out increment), so no gradient comes from, or passes through,
a blocked update (detaching alone would let the blocked steps' loss terms still train earlier updates). FR_t =
KL(Bayes_t || naive_t) of the goal belief at step t of each training episode. With a(t) of the batch's 8,000 updates at
step t above theta, both networks of a column block m(t) = max(a(t), 8,000 - a(t)) updates at that step:
  row 1, blocked above theta   every update with FR_t > theta, plus random ones below where those are the majority
  row 2, blocked below theta   every update with FR_t < theta, plus random ones above where those are the majority
so the two networks of a column learn from the same number of updates at every step of every batch, min(a(t), 8,000 -
a(t)), all on their own side of theta, for the same number of episodes (3,001 raw = 2,400 training episodes; the .1
runs continue to 12,001 raw). The trained share differs between columns (it is largest where theta splits the updates
evenly: printed by --numbers and noted under each column title).

Panels: accuracy at the last step (T = 30) by the probe state's FR_T (half-decade bins, .003 to 30 nats, plus the two
open ends; drawn at the bins' mean FR_T), at the stages of training (colour); thin black lines: Bayes (solid) and naive
Bayes (dashed); shaded red: the blocked side of theta (John 11-10: "keep blocked region as red and we dont need no block
or chance"; the unblocked Trained RNN, analysis/trained, stays in --numbers).

Data: the runs' checkpoints replayed on the probe set (analysis/probes.npz: 50 test environments x 100 cells x 2
trajectories = 10,000 episodes x 2 variables = 20,000 states; CPU, ~10 s per checkpoint) into
through_training/fr_holdout/<run>_<raw>.npz (ans (n, 30, 2) = each variable's answer at every step, logit_T (n, 2, 10)).
EQUAL-CHUNK RESULTS (figure_bands, 11-10 19:54; (.1 | 1), each network learns the same 20.4% of updates at every
step; accuracy at T on probe states with FR_T < .1 / .1-1 / > 1; 2,400 -> 4,800 -> 7,200 -> 9,600 training episodes):
  learned low FR    low .772 .795 .807 .807 | mid .583 .662 .693 .713 | high .225 .309 .388 .437 (still rising)
  learned high FR   low .573 .716 .754 .774 | mid .560 .666 .695 .707 | high .593 .656 .676 .688
  (.03 | 3), 7.8% learned at every step; FR_T < .03 / .03-3 / > 3:
  learned low FR    low .772 .815 .825 .831 | mid .442 .513 .542 .558 | high .079 .079 .084 .084 (flat: none up)
  learned high FR   low .321 .537 .614 .675 | mid .355 .478 .549 .596 | high .524 .613 .639 .666 (low still rising)
  random, the (.1 | 1) count, 2,400 / 4,800 / 7,200: below .1 .810 .839 .843 | .1-1 .682 .727 .739 | above 1 .596 .655
  .674 (learning from every FR range beats both chunks at low FR and matches the high-FR chunk at high FR).
  END OF TRAINING (end_panel; mean over 13 checkpoints, raw 9,600-12,000, the last 1,920 training episodes; SD over
  checkpoints <= .016): (.03 | 3) learned low: below .833, above .084; learned high: below .655, above .656;
  (.1 | 1) learned low: below .806, above .410; learned high: below .765, above .684; random at the (.1 | 1) count:
  below .03 .840, above 3 .681, below .1 .809, above 1 .698. The random line is drawn in the (.1 | 1) column only.
  (naive .814 / .584 / .065, Bayes .826 / .765 / .736). Learning from high FR carries down to low FR (.774 vs .807);
  learning from low FR carries up slowly (.437 vs .688 at 9,600, the gap closing ~.05 per 2,400 episodes).
RESULTS (11-10, --numbers; accuracy at T at 2,400 training episodes, below / above theta, bootstrap 95% CI over the
50 environments; no block = the unblocked Trained RNN at 2,400 episodes):
  theta .01 (4.1% of updates learned; 1,093 probe states below)
    blocked above   below .794 [.773, .815]   above .319 [.298, .341]   (naive .899 / .435: worse than naive above)
    blocked below   below .801 [.768, .832]   above .588 [.565, .612]   no block .866 / .689
  theta .1 (26.5%)
    blocked above   below .767 [.751, .782]   above .411 [.390, .431]   (9,600 episodes: .806 / .557, figure_long)
    blocked below   below .740 [.718, .759]   above .647 [.625, .669]   no block .785 / .671
  theta 1 (25.3%; nothing learned at step 1, 3.5% at step 2)
    blocked above   below .725 [.708, .740]   above .514 [.484, .544]
    blocked below   below .612 [.588, .634]   above .619 [.589, .646]   no block .729 / .640 (still rising steeply:
                    1,280 episodes .36 / .44)
  random block at .1's count (excl_rand_thr10_s0: m(t) random updates blocked per step; 71% of the learned ones above
    .1, the batches' own mix): below .765 [.748, .780], above .646 [.623, .668]: the count alone costs .02-.03 against
    no block; learning only above .1 costs nothing above (.647) and .025 below; learning only below .1 costs nothing
    below (.767) and .235 above. (Not drawn; numbers only.)
  At .01 and .1 the two networks are equal on the low-FR states while the network that learned only from high-FR
  updates is far ahead on the high-FR ones (+.27, +.24): learning transfers from high FR down, not up. At 1 both are
  ~.12 behind the unblocked network on their blocked side at 2,400 episodes.
usage:  python fig_learn_holdout.py [--data [run ...]] [--numbers]"""
import os
import sys
import time
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.cm import ScalarMappable                                          # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from matplotlib.patches import Patch                                              # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import fig_learn_profile as flp                                                   # noqa: E402

TD, AN = flp.TD, flp.AN
OUT = os.path.join(TD, "through_training", "fr_holdout")
T, NO, R, N_ENV = flp.T, flp.NO, flp.R, flp.N_ENV
THRS = (.01, .1, 1)                                                               # nats (John 11-10, below)
EDGES = np.array([0, .003, .01, .03, .1, .3, 1, 3, 10, 30, np.inf])               # FR_T bins: every threshold an edge
BLOCK = "#d62728"                                                                 # the blocked FR range (both rows)
ROWS = (("hi", "blocked above", BLOCK), ("lo", "blocked below", BLOCK))
RAND = "excl_rand_thr10_s0"                                                       # random block, the .1 count (numbers)
FULL = ("trained", "no block", "#1f77b4")                                         # the Trained RNN (analysis/trained)
STAGES = (100, 300, 600, 1100, 1600, 3000)                                        # raw; 3,000 = 2,400 training episodes
EXTRA = (6000, 9000, 12000)                                                       # the .1 runs' later stages (numbers)
CONT = tuple(range(12400, 37501, 400)) + (37500,)                                 # the (.1 | 1) runs continued (11-11):
                                                                                  # every 4th checkpoint to 30,000 episodes
DRAW = STAGES + (6000, 12000, 18800, 37500)                                       # stages drawn in figure_bands
TS, FL, FT, DPI, CMAP, NORM = flp.TS, flp.FL, flp.FT, flp.DPI, flp.CMAP, flp.NORM
FIG_W = 7.0
train_x = flp.train_x


def tag(th):
    return f"{round(th * 100):02d}"


def run_name(side, th):
    return f"excl_{side}_thr{tag(th)}_s0"


def fmt(th):
    return f"{th:g}"


# ── data ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def cache(run, raw):
    return os.path.join(OUT, f"{run}_{int(raw):05d}.npz")


def replay(run, raw, chunk=2500):
    """the network's answers on the 10,000 probes: ans (n, 30, 2) int8 = argmax of each variable's summed read-out at
    every step, logit_T (n, 2, 10) float32 = the summed read-out at the last step (td_tt_learnmask_fr.answers' replay;
    agrees with its stored ctrl_s0 replays to 2e-5 in the logits, every answer equal)"""
    import torch
    import td_tt_intervene as ti
    import td_tt_chaos as tc
    P = ti.probes()
    with torch.no_grad():
        w = ti.load(f"learnmask_{run}", raw)
        table = ti._table(w, P["Z"], torch.device("cpu"))
        code = torch.as_tensor((P["obs"] * (1 << np.arange(NO))).sum(-1).astype(np.int64))
        envt = torch.as_tensor(P["env"].astype(np.int64))
        Whh, H = w["LSTM.weight_hh_l0"], w["LSTM.weight_hh_l0"].shape[1]
        ans, last = [], []
        for b0 in range(0, len(envt), chunk):
            e, cd = envt[b0:b0 + chunk], code[b0:b0 + chunk]
            b = len(e)
            h = w["STM"].reshape(1, H).expand(b, H).contiguous()
            c = w["LTM"].reshape(1, H).expand(b, H).contiguous()
            lg = torch.zeros(b, 2 * R, dtype=torch.float64)
            a = torch.empty(b, T, 2, dtype=torch.int8)
            for t in range(T):
                h, c, _ = tc.lstm_step(table[e, cd[:, t]], h, c, Whh)
                lg += (h @ w["classifier_readout.weight"].T + w["classifier_readout.bias"]).double()
                a[:, t] = lg.reshape(b, 2, R).argmax(-1).to(torch.int8)
            ans.append(a)
            last.append(lg.reshape(b, 2, R).float())
    return torch.cat(ans).numpy(), torch.cat(last).numpy()


BANDS = ((.03, 3), (.1, 1))                                                       # equal FR chunks (lo, hi), nats;
                                                                                  # each learns its own per-step count
BROWS = (("lo", "learned low FR"), ("hi", "learned high FR"))
LEARN = "#2ca02c"                                                                 # the learned chunk


def band_name(side, lo, hi):
    return f"excl_band{side}_b{tag(lo)}_{tag(hi)}_c{tag(lo)}_{tag(hi)}_s0"            # own count (John 11-10 16:20)


BRAND = band_name("rand", .1, 1)                                                  # the same count, random updates


def all_runs():
    return ([run_name(s, th) for th in THRS for s, _, _ in ROWS] + [RAND]
            + [band_name(s, lo, hi) for lo, hi in BANDS for s, _ in BROWS] + [BRAND])


END = tuple(range(9600, 12001, 200))                                              # the last ~1,920 training episodes


def harvest(runs=None, raws=None):
    """replay every stored checkpoint of raws (default STAGES / EXTRA) that has no file yet (atomic writes)"""
    import torch
    import td_tt_intervene as ti
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    os.makedirs(OUT, exist_ok=True)
    for run in runs or all_runs():
        for raw in raws or STAGES + EXTRA:
            f = cache(run, raw)
            if os.path.exists(f) or not os.path.exists(ti.ckpt_path(f"learnmask_{run}", raw)):
                continue
            t0 = time.time()
            ans, last = replay(run, raw)
            np.savez(f[:-4] + ".tmp.npz", ans=ans, logit_T=last)
            os.replace(f[:-4] + ".tmp.npz", f)
            print(f"{run} {raw}: {time.time() - t0:.0f} s", flush=True)


def load():
    """the probe states' FR and truth, the observers' answers, every replayed stage of every run (answers at T), and the
    unblocked Trained RNN at the same stages"""
    P = np.load(os.path.join(AN, "probes.npz"))
    lj, ln = P["log_jb"], P["log_nb"]
    fr = flp.forward_fr(lj, ln)                                                   # (n, 30, 2)
    Z = dict(fr=fr, fr_T=fr[:, -1], r=P["r"], env=P["env"], naive=ln[:, -1].argmax(-1), bayes=lj[:, -1].argmax(-1))
    Z["bin"] = np.searchsorted(EDGES, Z["fr_T"], side="right") - 1
    Z["x"] = np.array([Z["fr_T"][Z["bin"] == k].mean() for k in range(len(EDGES) - 1)])
    for run in all_runs():
        raws = [raw for raw in sorted(set(STAGES + EXTRA + CONT)) if os.path.exists(cache(run, raw))]
        Z[f"raw_{run}"] = np.array(raws, int)
        Z[f"ans_{run}"] = np.stack([np.load(cache(run, raw))["ans"][:, -1] for raw in raws]) if raws else None
    f = os.path.join(OUT, "trained_answers.npz")                                  # the Trained RNN's answers, read once
    if not os.path.exists(f):
        raws = [raw for raw in STAGES + EXTRA if os.path.exists(os.path.join(AN, FULL[0], f"ck_{raw:05d}.npz"))]
        ans = np.stack([np.load(os.path.join(AN, FULL[0], f"ck_{raw:05d}.npz"))["logq"].astype(np.float32).argmax(-1)
                        for raw in raws]).astype(np.int8)
        os.makedirs(OUT, exist_ok=True)
        np.savez(f[:-4] + ".tmp.npz", raw=np.array(raws), ans=ans)
        os.replace(f[:-4] + ".tmp.npz", f)
    z = np.load(f)
    Z[f"raw_{FULL[0]}"] = z["raw"]
    Z[f"ans_{FULL[0]}"] = z["ans"] if z["ans"].ndim == 3 else z["ans"][:, :, -1]
    return Z


def profile(hit, b):
    return np.array([hit[b == k].mean() for k in range(len(EDGES) - 1)])


def at_T(Z, run, raw):
    """the run's answers at the last step at raw episode raw: (n, 2), or None"""
    if Z.get(f"ans_{run}") is None:
        return None
    i = np.flatnonzero(Z[f"raw_{run}"] == raw)
    return Z[f"ans_{run}"][i[0]] if len(i) else None


def sel_stats(run):
    """the training batches' blocked share and share above the threshold (means over the run), per step too"""
    f = os.path.join(TD, f"learnmask_{run}_s20261029", "sel.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return dict(blocked=z["sel"][:, 0].mean(), fr_blocked=z["sel"][:, 1].mean(), fr_trained=z["sel"][:, 2].mean(),
                above=z["sel"][:, 3].mean(), above_trained=z["sel"][:, 4].mean(), steps=z["steps"].mean(0), n=len(z["sel"]))


def columns(Z):
    """the thresholds whose two networks both reached 2,400 training episodes"""
    return [th for th in THRS if all(at_T(Z, run_name(s, th), STAGES[-1]) is not None for s, _, _ in ROWS)]


# ── numbers ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def numbers(Z):
    f = lambda a: " ".join(f"{v:5.3f}" for v in a)
    b, r, fr = Z["bin"], Z["r"], Z["fr_T"]
    W = flp.boot_weights()
    g = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    print(f"{b.size:,} probe states; states per bin {np.bincount(b.ravel(), minlength=len(EDGES) - 1).tolist()}")
    print(f"{'bin mean FR_T':34s}", " ".join(f"{v:5.3g}" for v in Z["x"]))
    for name, a in (("naive", Z["naive"]), ("Bayes", Z["bayes"])):
        print(f"{name:34s}", f(profile(a == r, b)))
    i = np.flatnonzero(Z[f"raw_{FULL[0]}"] == STAGES[-1])[0]
    print(f"{'no block, 2,400 episodes':34s}", f(profile(Z[f'ans_{FULL[0]}'][i] == r, b)))
    for th in THRS:
        hi = fr > th
        ab = Z["fr"] > th
        never, always = ~ab.any(1), ab.all(1)
        nb = Z[f"ans_{FULL[0]}"][i]
        print(f"\n== threshold {fmt(th)} nats: FR_T above {hi.sum():,}, below {(~hi).sum():,}; all 30 updates above "
              f"{always.sum():,}, all below {never.sum():,}")
        print(f"   {'':26s} below            above            | all below  all above")
        for name, a in (("naive", Z["naive"]), ("Bayes", Z["bayes"]), ("no block (2,400)", nb)):
            print(f"   {name:26s} {(a == r)[~hi].mean():.3f}            {(a == r)[hi].mean():.3f}            | "
                  f"{(a == r)[never].mean():.3f}      {(a == r)[always].mean():.3f}")
        for side, rname, _ in ROWS + (("rand", "random", None),) if th == .1 else ROWS:
            run = run_name(side, th)
            s = sel_stats(run)
            if s is not None:
                print(f"   -- {rname} {fmt(th)} ({run}): {s['n']:,} training episodes logged; trained "
                      f"{1 - s['blocked']:.3f} of the updates (per step: t=1 {1 - s['steps'][0, 0]:.3f}, t=2 "
                      f"{1 - s['steps'][0, 1]:.3f}, t=10 {1 - s['steps'][0, 9]:.3f}, t=30 {1 - s['steps'][0, 29]:.3f}); "
                      f"share above theta among the trained {s['above_trained']:.3f}; mean FR_t blocked "
                      f"{s['fr_blocked']:.3f} / trained {s['fr_trained']:.3f}")
            if Z.get(f"ans_{run}") is None:
                print(f"   -- {rname} {fmt(th)}: no replayed stage yet")
                continue
            for raw, a in zip(Z[f"raw_{run}"], Z[f"ans_{run}"]):
                print(f"      {train_x(raw):6,d}  {f(profile(a == r, b))} | below {g(flp.boot(Z['env'], ~hi, a == r, W=W))}"
                      f" above {g(flp.boot(Z['env'], hi, a == r, W=W))} | all below {(a == r)[never].mean():.3f} all "
                      f"above {(a == r)[always].mean():.3f} | naive's answers above {(a == Z['naive'])[hi].mean():.3f}")


# ── figure ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def axes(ax, left, bottom):
    ax.set_xscale("log")
    ax.set_xlim(4e-4, 120)
    ax.set_xticks([1e-3, 1e-1, 10])
    ax.set_xticklabels([r"$10^{-3}$", "0.1", "10"] if bottom else [])
    ax.set_xticks([1e-2, 1], minor=True)
    ax.set_xticklabels([], minor=True)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, .5, 1])
    ax.set_yticks([.25, .75], minor=True)
    ax.set_yticklabels(["0", "0.5", "1"] if left else [])
    ax.tick_params(labelsize=FT - .5, pad=2, length=3)
    ax.tick_params(which="minor", length=1.8)
    ax.spines[["top", "right"]].set_visible(False)


def figure(Z, cols=None, path=None):
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    b, x, r = Z["bin"], Z["x"], Z["r"]
    naive, bayes = profile(Z["naive"] == r, b), profile(Z["bayes"] == r, b)
    cols = columns(Z) if cols is None else cols
    nc = len(cols)

    left, right, gap, ph, rgap = .66, .08, .10, 1.12, .16
    pw = (FIG_W - left - right - (nc - 1) * gap) / max(nc, 4)
    top0 = .58
    tick, xlab, strip, note = .20, .22, .48, .78
    H = top0 + 2 * ph + rgap + tick + xlab + strip + note
    fig = plt.figure(figsize=(FIG_W, H))
    ftext = lambda xx, tp, s, **k: fig.text(xx / FIG_W, 1 - tp / H, s, **k)
    x0 = [left + k * (pw + gap) for k in range(nc)]
    for k, th in enumerate(cols):
        s = sel_stats(run_name("hi", th))
        ftext(x0[k] + pw / 2, top0 - .25, rf"$\theta$ = {fmt(th)} nats", fontsize=TS, ha="center", va="bottom")
        if s is not None:
            ftext(x0[k] + pw / 2, top0 - .06, f"{100 * (1 - s['blocked']):.0f}% of updates learned", fontsize=FT - 1.5,
                  ha="center", va="bottom", color="0.3")
    for row, (side, name, c) in enumerate(ROWS):
        top = top0 + row * (ph + rgap)
        ftext(.13, top + ph / 2, rf"{name} $\theta$", rotation=90, fontsize=FL, ha="center", va="center")
        for k, th in enumerate(cols):
            ax = fig.add_axes([x0[k] / FIG_W, 1 - (top + ph) / H, pw / FIG_W, ph / H])
            axes(ax, k == 0, row == 1)
            span = (th, 200) if side == "hi" else (1e-4, th)
            ax.axvspan(*span, color=c, alpha=.10, lw=0, zorder=0)
            ax.axvline(th, color=c, lw=.8, zorder=1)
            run = run_name(side, th)
            st = [raw for raw in STAGES if at_T(Z, run, raw) is not None]
            flp.stages(ax, x, np.stack([profile(at_T(Z, run, raw) == r, b) for raw in st]), st)
            flp.observers(ax, x, naive, bayes)
            if k == 0:
                ax.set_ylabel("accuracy", fontsize=FL - .5, labelpad=2)
    ftext(left + (nc * pw + (nc - 1) * gap) / 2 if nc >= 3 else left - .3, top0 + 2 * ph + rgap + tick + .02,
          r"$\mathrm{FR}_T = D_{\mathrm{KL}}(B_T^{\mathrm{Bayes}}\,\Vert\,B_T^{\mathrm{Naive}})$ of the probe state (nats)",
          fontsize=FL, ha="center" if nc >= 3 else "left", va="top")

    # legend strip: stage colour bar, observers, the unblocked network, the blocked side
    y0 = top0 + 2 * ph + rgap + tick + xlab + .10
    hs = [Line2D([], [], color="k", lw=.9), Line2D([], [], color="k", lw=.9, ls=(0, (4, 2))),
          Patch(fc=BLOCK, alpha=.22, lw=0)]
    lb = ["Bayes", "naive", "blocked (not learned)"]
    fig.legend(hs, lb, loc="upper left", bbox_to_anchor=((left + 2.62) / FIG_W, 1 - (y0 + .07) / H), ncol=3,
               fontsize=FT - .5, frameon=False, handlelength=1.6, columnspacing=1.2, handletextpad=.4, borderpad=0,
               borderaxespad=0)
    cw = 1.25
    cax = fig.add_axes([(left + .78) / FIG_W, 1 - (y0 + .16) / H, cw / FIG_W, .075 / H])
    cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal")
    cax.set_xlim(train_x(STAGES[0]), train_x(STAGES[-1]))
    cb.set_ticks([100, 1000])
    cb.set_ticklabels(["100", "1k"])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=FT - 1, pad=1.5, length=2)
    cb.outline.set_linewidth(.5)
    ftext(left + .72, y0 + .125, "episodes", fontsize=FT - .5, ha="right", va="center")
    ftext(left + .78 + cw + .05, y0 + .125, f"{train_x(STAGES[-1]):,}", fontsize=FT - .5, ha="left", va="center")

    hi = Z["fr_T"]
    ftext(.12, y0 + strip, f"{N_ENV} test environments $\\times$ 200 trajectories $\\times$ 2 variables = {b.size:,} "
          f"states, accuracy at the last step ($T$ = {T}); FR$_T$ bins of\nhalf a decade (states per bin "
          f"{np.bincount(b.ravel()).min():,} to {np.bincount(b.ravel()).max():,}); every network trained from "
          f"initialization for {train_x(STAGES[-1]):,} episodes; one run per network;\nthe two networks of a column "
          f"learn from the same number of updates at every step, each only from its unblocked side",
          fontsize=FT - 1, ha="left", va="top", color="0.15", linespacing=1.35)
    path = path or fig_path("fig_learn_holdout.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("saved", path)


def figure_long(Z, th=.1, path=None):
    """fig_learn_holdout_long.png: the theta = .1 pair through all its stages (to 9,600 training episodes): does the
    network blocked above theta catch up at high FR with 4x the training, or does the deficit stay? Same panels as one
    column of fig_learn_holdout.
    RESULTS (11-10, 9,600 training episodes; accuracy below / above .1 nats, bootstrap 95% CI over environments):
      blocked above .1   below .806 [.792, .817]   above .557 [.533, .579]   (2,400: .767 / .411; 4,800 .490; 7,200 .522)
      blocked below .1   below .802 [.786, .815]   above .713 [.693, .731]   (2,400: .740 / .647)
      no block           below .816                above .732                 (2,400: .785 / .671)
      naive .814 / .348, Bayes .826 / .752; the 5.6 / 17 / 44-nat bins: blocked above .35 / .16 / .02 (naive .03 / .02 /
      .02), blocked below .69 / .68 / .60, no block .71 / .69 / .64. Transfer down is complete; transfer up is slow and
      partial, and at the highest FR there is none in 9,600 episodes (a delay is not ruled out: say "not learned")."""
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    b, x, r = Z["bin"], Z["x"], Z["r"]
    naive, bayes = profile(Z["naive"] == r, b), profile(Z["bayes"] == r, b)
    st_all = STAGES + EXTRA
    last = max(raw for raw in st_all if all(at_T(Z, run_name(s, th), raw) is not None for s, _, _ in ROWS))

    left, pw, ph, rgap, top0, tick, xlab = .66, 2.6, 1.2, .16, .52, .20, .24
    lx = left + pw + .45                                                          # legend column
    W = lx + 2.05
    H = top0 + 2 * ph + rgap + tick + xlab + .72
    fig = plt.figure(figsize=(W, H))
    ftext = lambda xx, tp, s, **k: fig.text(xx / W, 1 - tp / H, s, **k)
    s = sel_stats(run_name("hi", th))
    ftext(left + pw / 2, top0 - .25, rf"$\theta$ = {fmt(th)} nats", fontsize=TS, ha="center", va="bottom")
    if s is not None:
        ftext(left + pw / 2, top0 - .06, f"{100 * (1 - s['blocked']):.0f}% of updates learned", fontsize=FT - 1.5,
              ha="center", va="bottom", color="0.3")
    for row, (side, name, c) in enumerate(ROWS):
        top = top0 + row * (ph + rgap)
        ftext(.13, top + ph / 2, rf"{name} $\theta$", rotation=90, fontsize=FL, ha="center", va="center")
        ax = fig.add_axes([left / W, 1 - (top + ph) / H, pw / W, ph / H])
        axes(ax, True, row == 1)
        ax.axvspan(*((th, 200) if side == "hi" else (1e-4, th)), color=c, alpha=.10, lw=0, zorder=0)
        ax.axvline(th, color=c, lw=.8, zorder=1)
        run = run_name(side, th)
        st = [raw for raw in st_all if raw <= last and at_T(Z, run, raw) is not None]
        flp.stages(ax, x, np.stack([profile(at_T(Z, run, raw) == r, b) for raw in st]), st)
        flp.observers(ax, x, naive, bayes)
        ax.set_ylabel("accuracy", fontsize=FL - .5, labelpad=2)
    ftext(left - .3, top0 + 2 * ph + rgap + tick + .02,
          r"$\mathrm{FR}_T = D_{\mathrm{KL}}(B_T^{\mathrm{Bayes}}\,\Vert\,B_T^{\mathrm{Naive}})$ of the probe state (nats)",
          fontsize=FL, ha="left", va="top")

    hs = [Line2D([], [], color="k", lw=.9), Line2D([], [], color="k", lw=.9, ls=(0, (4, 2))),
          Patch(fc=BLOCK, alpha=.22, lw=0)]
    lb = ["Bayes", "naive", "blocked (not learned)"]
    fig.legend(hs, lb, loc="upper left", bbox_to_anchor=(lx / W, 1 - (top0 + .02) / H), ncol=1, fontsize=FT - .5,
               frameon=False, handlelength=1.6, handletextpad=.4, borderpad=0, borderaxespad=0, labelspacing=.45)
    cw, cy = 1.45, top0 + 1.62
    ftext(lx, cy - .07, "training episodes", fontsize=FT - .5, ha="left", va="bottom")
    cax = fig.add_axes([lx / W, 1 - (cy + .075) / H, cw / W, .075 / H])
    cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal")
    cax.set_xlim(train_x(STAGES[0]), train_x(last))
    cb.set_ticks([100, 1000])
    cb.set_ticklabels(["100", "1k"])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=FT - 1, pad=1.5, length=2)
    cb.outline.set_linewidth(.5)
    ftext(lx + cw + .05, cy + .04, f"{train_x(last):,}", fontsize=FT - .5, ha="left", va="center")

    ftext(.12, H - .58, f"{N_ENV} test environments $\\times$ 200 trajectories $\\times$ 2 variables = {b.size:,} states, "
          f"accuracy at $T$ = {T};\nFR$_T$ in half-decade bins; both networks trained from initialization for "
          f"{train_x(last):,} episodes (one run each),\neach learning from the same number of updates at every step, only "
          f"from its unblocked side", fontsize=FT - 1, ha="left", va="top", color="0.15", linespacing=1.35)
    path = path or fig_path("fig_learn_holdout_long.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("saved", path)


def band_columns(Z):
    return [b for b in BANDS if all(at_T(Z, band_name(s, *b), STAGES[-1]) is not None for s, _ in BROWS)]


def band_numbers(Z):
    """accuracy at T below lo / between / above hi for each band run at every stage (+ the random control)"""
    b, r, fr = Z["bin"], Z["r"], Z["fr_T"]
    W = flp.boot_weights()
    g = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    for lo, hi in BANDS:
        low, high = fr < lo, fr > hi
        print(f"\n== chunks FR < {fmt(lo)} | > {fmt(hi)}: probe states {low.sum():,} / {(~low & ~high).sum():,} / "
              f"{high.sum():,}")
        for name, a in (("naive", Z["naive"]), ("Bayes", Z["bayes"])):
            print(f"   {name:24s} low {(a == r)[low].mean():.3f}  mid {(a == r)[~low & ~high].mean():.3f}  "
                  f"high {(a == r)[high].mean():.3f}")
        for run in [band_name(sd_, lo, hi) for sd_, _ in BROWS] + [BRAND]:
            st = sel_stats(run)
            if st is not None:
                print(f"   -- {run}: {st['n']:,} training episodes logged; learned {1 - st['blocked']:.4f}; mean FR "
                      f"learned {st['fr_trained']:.3f}")
            if Z.get(f"ans_{run}") is None:
                continue
            for raw, a in zip(Z[f"raw_{run}"], Z[f"ans_{run}"]):
                print(f"      {train_x(raw):6,d}  low {g(flp.boot(Z['env'], low, a == r, W=W))}  mid "
                      f"{g(flp.boot(Z['env'], ~low & ~high, a == r, W=W))}  high {g(flp.boot(Z['env'], high, a == r, W=W))}")


GEN_LS = (0, (3, 1.6))                                                            # generalization: outside the chunk


def split_curve(ax, x, y, th, side, color, last):
    """one stage's accuracy profile, solid where FR is in the learned chunk and dashed outside it, joined at the
    threshold (log-linear interpolation)"""
    lx, lt = np.log10(x), np.log10(th)
    yb = float(np.interp(lt, lx, y))
    inside = x < th if side == "lo" else x > th
    xin, yin = x[inside], y[inside]
    xout, yout = x[~inside], y[~inside]
    if side == "lo":
        xin, yin = np.r_[xin, th], np.r_[yin, yb]
        xout, yout = np.r_[th, xout], np.r_[yb, yout]
    else:
        xin, yin = np.r_[th, xin], np.r_[yb, yin]
        xout, yout = np.r_[xout, th], np.r_[yout, yb]
    kw = dict(color=color, lw=.8, zorder=3, solid_capstyle="round")              # one width for every stage (John
                                                                                  # 11-10: "why is the last episode line
                                                                                  # so thick?": only fig_learn_profile's
                                                                                  # highlight of the latest stage)
    ax.plot(xin, yin, **kw)
    ax.plot(xout, yout, ls=GEN_LS, **kw)
    ax.plot(x, y, ls="none", marker="o", ms=1.4, mew=0, color=color, zorder=kw["zorder"])


END_COL = {"hi": "#2ca02c", "lo": "#d62728"}                                      # learned at high / low FR
RND_STYLE = dict(color="0.4", lw=1.1, ls=(0, (1, 1.5)))                           # the random control: dotted (as the GIF)
END_SHADE = {(.03, 3): 1.0, (.1, 1): .55}                                         # narrower chunks darker


def end_profile(Z, run, last=12000):
    """accuracy per FR_T bin at each replayed checkpoint in the last 2,400 raw (1,920 training) episodes up to raw
    last, then the mean over the checkpoints (and their SD), or None"""
    have = [raw for raw in range(last - 2400, last + 1, 100) if os.path.exists(cache(run, raw))]
    if len(have) < 5:
        return None
    P = np.stack([profile(np.load(cache(run, raw))["ans"][:, -1] == Z["r"], Z["bin"]) for raw in have])
    return P.mean(0), P.std(0), len(have)


def end_panel(fig, Z, W_, H, xl, w, top, h, cols, lasts):
    """John 11-11 00:40: "a panel for the end of training results, just the last however many training episodes (1k
    or 2k, however many we can get for a stable batch of data), with all the different conditions, green curves for
    training at high FR, red curves for training at low FR. Again with solid curves for the training range and dashed
    curves for the testing range": accuracy at T by FR_T bin, the mean over each run's replayed checkpoints in its last
    1,920 training episodes (end_profile; lasts: each column's last common checkpoint, so both networks of a column, and
    the random control of (.1 | 1), end at the same episode); dotted: the random control"""
    from matplotlib.colors import to_rgb
    ax = fig.add_axes([xl / W_, 1 - (top + h) / H, w / W_, h / H])
    axes(ax, True, True)
    ax.set_xlim(10 ** -3.5, 10 ** 2.5)
    x = Z["x"]
    hs, lb = [], []
    for (lo, hi), last in zip(cols, lasts):
        for side in ("lo", "hi"):
            e = end_profile(Z, band_name(side, lo, hi), last)
            if e is None:
                continue
            k = END_SHADE.get((lo, hi), .75)
            c = tuple(1 - k * (1 - v) for v in to_rgb(END_COL[side]))
            split_curve(ax, x, e[0], lo if side == "lo" else hi, side, c, False)
            hs.append(Line2D([], [], color=c, lw=1.2))
            lb.append(("FR $<$ " if side == "lo" else "FR $>$ ") + fmt(lo if side == "lo" else hi) + f" ({train_x(last):,})")
        if (lo, hi) == (.1, 1):
            e = end_profile(Z, BRAND, last)
            if e is not None:
                ax.plot(x, e[0], zorder=2, **RND_STYLE)
                hs.append(Line2D([], [], **RND_STYLE))
                lb.append(rf"random steps ({train_x(last):,})")
    ax.set_ylabel("accuracy", fontsize=FL - .5, labelpad=2)
    ax.set_xlabel(r"FR$_T$ of the probe state (nats)", fontsize=FL, labelpad=3)
    ax.annotate("End of training", xy=(.5, 1), xycoords="axes fraction", xytext=(0, 16), textcoords="offset points",
                ha="center", va="bottom", fontsize=TS)
    ax.annotate("mean over each run's last 1,920 training episodes", xy=(.5, 1), xycoords="axes fraction",
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=FT - 1.5, color="0.3")
    ax.legend(hs, lb, title="learned from (training episodes)", title_fontsize=FT - 1, loc="lower left",
              fontsize=FT - 1, frameon=False, handlelength=1.6, labelspacing=.3, borderaxespad=.2)


def figure_bands(Z, cols=None, path=None):
    """John 11-10 (13:58): "Does learn_holdout control for the amount of training data and weight updates? Everything
    should be balanced. Also maybe switching from red for blocked, let's go green for the learned chunk, and order the
    panels so the columns have an equal chunk of FR (i.e. equal size of green chunk). Also adding theta = 10 could be
    good." Columns: a pair of equal FR chunks (lo, hi), lo * hi = .1, so on the log axis (centred on .32 nats) the two
    green chunks have the same width; rows: the network that learned only from updates with FR < lo / FR > hi
    (train_learn_masks band modes): every network learns exactly n(t) = min(#FR < .01, #FR > 10) updates at step t of
    every batch (2.9% of all), the same in every panel, for the same 2,400 training episodes (2,400 weight updates; the
    loss is the mean over the learned terms). Grey: the same count of random updates (excl_bandrand) at 2,400.
    Since 11-11 17:10 (John: "update everything with the most recent data"): each column runs to its own last checkpoint
    that all its networks have reached (the (.1 | 1) networks continue to 30,000 training episodes, train_learn_masks
    cont; the (.03 | 3) ones end at 9,600), so the networks of a column always compare at equal training; the random
    control dotted; the key in the GIF's wording (Training set / Evaluation set / Training at random steps)."""
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    b, x, r = Z["bin"], Z["x"], Z["r"]
    naive, bayes = profile(Z["naive"] == r, b), profile(Z["bayes"] == r, b)
    cols = band_columns(Z) if cols is None else cols
    nc = len(cols)
    if not nc:
        print("no band pair has reached 2,400 episodes yet: nothing drawn")
        return
    def col_last(lo, hi):                                                         # each column's last checkpoint that
        runs_ = [band_name(s_, lo, hi) for s_, _ in BROWS] + ([BRAND] if (lo, hi) == (.1, 1) else [])   # every one of
        return max(raw for raw in sorted(set(STAGES + EXTRA + CONT))              # its networks has reached: within a
                   if all(at_T(Z, r_, raw) is not None for r_ in runs_))          # column the comparison stays matched
    lasts = [col_last(*b) for b in cols]
    last = max(lasts)
    left, right, gap, ph, rgap, pw = .66, .08, .14, 1.12, .16, 2.0
    egap, ew = .55, 2.6                                                           # the end-of-training panel
    W_ = max(left + nc * pw + (nc - 1) * gap + egap + ew + right, 4.9)
    top0, tick, xlab, strip = .58, .20, .22, .48
    H = top0 + 2 * ph + rgap + tick + xlab + strip + .58                          # room for the 4-line legend
    fig = plt.figure(figsize=(W_, H))
    ftext = lambda xx, tp, s_, **k: fig.text(xx / W_, 1 - tp / H, s_, **k)
    x0 = [left + k * (pw + gap) for k in range(nc)]
    XL = (10 ** -3.5, 10 ** 2.5)                                                  # symmetric in log about .32 nats
    for k, (lo, hi) in enumerate(cols):
        st = sel_stats(band_name("lo", lo, hi))
        ftext(x0[k] + pw / 2, top0 - .25, rf"FR $<$ {fmt(lo)}  or  $>$ {fmt(hi)} nats", fontsize=TS, ha="center",
              va="bottom")
        if st is not None:
            ftext(x0[k] + pw / 2, top0 - .06, f"{100 * (1 - st['blocked']):.1f}% of updates, "
                  f"{train_x(lasts[k]):,} episodes", fontsize=FT - 1.5, ha="center", va="bottom", color="0.3")
    for row, (side, name) in enumerate(BROWS):
        top = top0 + row * (ph + rgap)
        ftext(.13, top + ph / 2, name, rotation=90, fontsize=FL, ha="center", va="center")
        for k, (lo, hi) in enumerate(cols):
            ax = fig.add_axes([x0[k] / W_, 1 - (top + ph) / H, pw / W_, ph / H])
            axes(ax, k == 0, row == 1)
            ax.set_xlim(*XL)
            th = lo if side == "lo" else hi
            ax.axvspan(*((XL[0], lo) if side == "lo" else (hi, XL[1])), color=LEARN, alpha=.13, lw=0, zorder=0)
            ax.axvline(th, color=LEARN, lw=.8, zorder=1)
            rnd = at_T(Z, BRAND, lasts[k])
            if rnd is not None and (lo, hi) == (.1, 1):                  # its count matches (.1 | 1) only
                ax.plot(x, profile(rnd == r, b), zorder=2, **RND_STYLE)
            run = band_name(side, lo, hi)
            sts = sorted({raw for raw in DRAW if raw <= lasts[k]} | {lasts[k]})
            sts = [raw for raw in sts if at_T(Z, run, raw) is not None]
            for n_, raw in enumerate(sts):                                # solid on the learned chunk, dashed where
                split_curve(ax, x, profile(at_T(Z, run, raw) == r, b), th, side,  # it generalizes (John 11-10)
                            CMAP(NORM(train_x(raw))), n_ == len(sts) - 1)
                                                                          # no naive / Bayes lines (John 11-10: "keep it
                                                                          # simpler")
            if k == 0:
                ax.set_ylabel("accuracy", fontsize=FL - .5, labelpad=2)
    ftext(left - .3, top0 + 2 * ph + rgap + tick + .02,
          r"$\mathrm{FR}_T = D_{\mathrm{KL}}(B_T^{\mathrm{Bayes}}\,\Vert\,B_T^{\mathrm{Naive}})$ of the probe state (nats)",
          fontsize=FL, ha="left", va="top")
    y0 = top0 + 2 * ph + rgap + tick + xlab + .22
    hs = [Patch(fc=LEARN, alpha=.26, lw=0), Line2D([], [], color="0.35", lw=1.2),
          Line2D([], [], color="0.35", lw=1.2, ls=GEN_LS), Line2D([], [], **RND_STYLE)]
    lb = ["training FR range", "Training set", "Evaluation set", r"Training at random steps (.1 $|$ 1)"]   # the GIF's
                                                                                  # wording (John 11-11)
    fig.legend(hs, lb, loc="upper left", bbox_to_anchor=((left + 2.05) / W_, 1 - (y0 + .02) / H), ncol=1,
               fontsize=FT - .5, frameon=False, handlelength=1.6, columnspacing=1.2, handletextpad=.4, borderpad=0,
               borderaxespad=0)
    cw = 1.25
    cax = fig.add_axes([(left + .05) / W_, 1 - (y0 + .16) / H, cw / W_, .075 / H])
    cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal")
    cax.set_xlim(train_x(STAGES[0]), train_x(last))
    tk = [v for v in (100, 1000, 10000) if v < train_x(last) * .8]
    cb.set_ticks(tk)
    cb.set_ticklabels([{100: "100", 1000: "1k", 10000: "10k"}[v] for v in tk])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=FT - 1, pad=1.5, length=2)
    cb.outline.set_linewidth(.5)
    ftext(left + .05, y0 + .06, "training episodes", fontsize=FT - .5, ha="left", va="bottom")
    ftext(left + .05 + cw + .05, y0 + .125, f"{train_x(last):,}", fontsize=FT - .5, ha="left", va="center")
    end_panel(fig, Z, W_, H, x0[-1] + pw + egap, ew, top0, 2 * ph + rgap, cols, lasts)
    path = path or fig_path("fig_learn_holdout.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("saved", path)


def figure_saturation(Z, chunk=(.1, 1), path=None):
    """fig_learn_holdout_saturation.png: has learning saturated? (John 11-11: "Are you sure the red is levelling off?")
    Accuracy at T of the (.1 | 1) networks at every replayed checkpoint (every 80 training episodes to 9,600, then
    every 320: the continued runs, train_learn_masks cont), against training episodes on a LINEAR axis; left: the probe
    states with FR_T > 1 (the learned-high network's training set, the learned-low network's evaluation set), right:
    FR_T < .1. Thin: every checkpoint; thick: a running mean over 1,000 training episodes. Dotted vertical: 15,040
    episodes, where the runs were stopped and continued (a fresh episode stream, the episode-9,600 Adam state).
    RESULTS (11-11 22:56, all three to 30,000; mean over the checkpoints near 15,040 / 20,000 / 25,000 / 30,000 training
    episodes, FR_T < .1 | FR_T > 1): learned FR < .1 .813|.508 -> .817|.565 -> .817|.593 -> .817|.609 (its FR > 1 slope
    per 1,000 episodes .015 at 10-15k, .008, .006, .003 at 25-30k; last five checkpoints .604-.614: levelled off);
    learned FR > 1 .788|.702 -> .799|.707 -> .801|.714 -> .805|.716; random steps .814|.712 -> .817|.718 -> .815|.720 ->
    .818|.722. At 30,000 the learned-low network stays .11 below on FR > 1, the learned-high one .012 below on FR < .1."""
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    lo, hi = chunk
    fr, r = Z["fr_T"], Z["r"]
    sets = ((fr > hi, rf"probe states with FR$_T$ $>$ {fmt(hi)}"), (fr < lo, rf"probe states with FR$_T$ $<$ {fmt(lo)}"))
    runs = ((band_name("lo", lo, hi), END_COL["lo"], rf"learned FR $<$ {fmt(lo)}", "lo"),
            (band_name("hi", lo, hi), END_COL["hi"], rf"learned FR $>$ {fmt(hi)}", "hi"),
            (BRAND, "0.4", "random steps", None))
    raws_all = sorted(set(range(100, 12001, 100)) | set(CONT))
    W_, H_ = 7.2, 2.75
    fig = plt.figure(figsize=(W_, H_))
    for j, (sel, ttl) in enumerate(sets):
        ax = fig.add_axes([(.62 + j * 3.35) / W_, .58 / H_, 2.95 / W_, 1.75 / H_])
        for run, c, nm, side in runs:
            have = [raw for raw in raws_all if os.path.exists(cache(run, raw))]
            t = np.array([train_x(raw) for raw in have], float)
            a = np.array([(np.load(cache(run, raw))["ans"][:, -1] == r)[sel].mean() for raw in have])
            train_side = (side == "hi") == (j == 0)                              # this panel is the run's training set
            ls = "-" if (side is not None and train_side) else (GEN_LS if side is not None else (0, (1, 1.5)))
            ax.plot(t, a, color=c, lw=.5, alpha=.45, ls="-")
            sm = np.array([a[np.abs(t - ti) <= 500].mean() for ti in t])          # running mean, 1,000 episodes
            ax.plot(t, sm, color=c, lw=1.6, ls=ls, label=nm)
        ax.axvline(15040, color="0.6", lw=.7, ls=(0, (1, 2)), zorder=0)
        ax.set_xlim(0, 30500)
        ax.set_xticks([0, 10000, 20000, 30000])
        ax.set_xticklabels(["0", "10k", "20k", "30k"])
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .5, 1])
        ax.tick_params(labelsize=FT - .5, pad=2, length=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(ttl, fontsize=FL, pad=4)
        ax.set_xlabel("training episodes", fontsize=FL - .5, labelpad=2)
        if j == 0:
            ax.set_ylabel("accuracy", fontsize=FL - .5, labelpad=2)
            ax.legend(loc="lower right", fontsize=FT - 1, frameon=False, handlelength=2.0, labelspacing=.3)
            ax.text(15400, .98, "continued", fontsize=FT - 1.5, color="0.45", ha="left", va="top")
    path = path or fig_path("fig_learn_holdout_saturation.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("saved", path)


def figure_saturation_one(Z, chunk=(.1, 1), path=None, logx=False, logy=False, random=False):
    """fig_learn_holdout_saturation_1panel.png: figure_saturation's curves in ONE panel (John 11-13: "i wonder if we can
    put it on 1 panel. with color indicating "Trained on" and linestyle indicating "Evaluated on", and a log-log scaling
    maybe?", then "maybe let's drop the log scale x axis here and the random steps lines"): colour = the updates a
    network learned from (red FR_t < .1, green FR_t > 1; random=True adds the grey random-steps network), line = the
    probe states it is evaluated on (solid FR_T > 1, dashed FR_T < .1); linear axes (logx / logy: log axes, the 11-13
    first draft). Thin: every replayed checkpoint; thick: a running mean over 1,000 training episodes (log x: +-.06
    decades). No 'continued' marker: the restart at 15,040 episodes (a fresh episode stream, the episode-9,600 Adam
    state) leaves no step in any curve. Numbers: figure_saturation's RESULTS."""
    from matplotlib.lines import Line2D as L2
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    lo, hi = chunk
    fr, r = Z["fr_T"], Z["r"]
    evals = ((fr > hi, "-", rf"FR$_T$ $>$ {fmt(hi)}"), (fr < lo, (0, (4, 2)), rf"FR$_T$ $<$ {fmt(lo)}"))
    runs = ((band_name("lo", lo, hi), END_COL["lo"], rf"FR$_t$ $<$ {fmt(lo)}"),
            (band_name("hi", lo, hi), END_COL["hi"], rf"FR$_t$ $>$ {fmt(hi)}"))
    runs += ((BRAND, "0.45", "random steps"),) if random else ()
    raws_all = sorted(set(range(100, 12001, 100)) | set(CONT))
    W_, H_ = 4.2, 3.0
    fig = plt.figure(figsize=(W_, H_))
    ax = fig.add_axes([.62 / W_, .55 / H_, 2.4 / W_, 2.2 / H_])
    for run, c, _ in runs:
        have = [raw for raw in raws_all if os.path.exists(cache(run, raw))]
        t = np.array([train_x(raw) for raw in have], float)
        ans = [np.load(cache(run, raw))["ans"][:, -1] == r for raw in have]
        lt = np.log10(t)
        for sel, ls, _ in evals:
            a = np.array([x[sel].mean() for x in ans])
            ax.plot(t, a, color=c, lw=.4, alpha=.3, ls="-", zorder=2)
            sm = np.array([a[np.abs(lt - v) <= .06].mean() for v in lt] if logx else
                          [a[np.abs(t - v) <= 500].mean() for v in t])
            ax.plot(t, sm, color=c, lw=1.7, ls=ls, zorder=3)
    if logx:
        ax.set_xscale("log")
        ax.set_xlim(70, 3.3e4)
        ax.set_xticks([100, 1000, 10000])
        ax.set_xticklabels(["100", "1k", "10k"])
    else:
        ax.set_xlim(0, 30500)
        ax.set_xticks([0, 10000, 20000, 30000])
        ax.set_xticklabels(["0", "10k", "20k", "30k"])
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(.08, 1)
        ax.set_yticks([.1, .2, .5, 1])
        ax.set_yticklabels([".1", ".2", ".5", "1"])
    else:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .5, 1])
    ax.minorticks_off()
    ax.tick_params(labelsize=FT - .5, pad=2, length=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("training episodes", fontsize=FL - .5, labelpad=2)
    ax.set_ylabel("accuracy", fontsize=FL - .5, labelpad=2)
    leg1 = ax.legend([L2([], [], color=c, lw=1.7) for _, c, _ in runs], [nm for _, _, nm in runs],
                     title="trained on", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=FT - 1,
                     title_fontsize=FT - .5, frameon=False, handlelength=1.8, labelspacing=.3, alignment="left")
    ax.add_artist(leg1)
    ax.legend([L2([], [], color="0.2", lw=1.7, ls=ls) for _, ls, _ in evals], [nm for _, _, nm in evals],
              title="evaluated on", loc="upper left", bbox_to_anchor=(1.02, .52), fontsize=FT - 1,
              title_fontsize=FT - .5, frameon=False, handlelength=1.8, labelspacing=.3, alignment="left")
    path = path or fig_path("fig_learn_holdout_saturation_1panel.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def figure_saturation_cols(Z, chunks=((.03, 3), (.1, 1)), path=None, xunit="updates", gap=True):
    """fig_learn_holdout_saturation_2col.png: figure_saturation_one with one column per FR band (John 11-13: "do we have an
    equivalent for the other threshold that we could put as another column?", then "and if the x axis is episdes *
    updates? Also can we add another panel that shows all of their Eval - Training"): (.03 | 3), continued 11-13/14
    from 12,000 raw to 66,750 to match the learned updates (John: "we'd want to match the learning updates"; last
    checkpoint raw 66,700 = 53,360 training episodes), and (.1 | 1), to 30,000. x: learned updates = training episodes
    x the band's learned share (sel.npz: 1 - the blocked share per episode; .0780 +- .0017 and .2044 +- .0023, constant
    through training) x 8,000 trajectories x 30 steps, i.e. 18,720 and 49,056 learned updates per episode; totals
    9.99e8 and 1.47e9 (xunit "episodes": training episodes, each column its own axis). Colour = trained on (red low FR,
    green high FR), line = evaluated on (solid high-FR probe states, dashed low-FR ones), the band's own edges. Thin:
    every replayed checkpoint ((.03 | 3): 270 per run, every 25 raw to 500, every 100 to 12,000, every 400 from 12,400,
    and the last, 66,700); thick: a running mean over 1,000 training episodes. Last panel: evaluation minus training
    accuracy of every network (red: high-FR minus low-FR probe states, green: the reverse), both bands on one axis of
    learned updates ((.03 | 3) dotted, (.1 | 1) solid); triangles: Joint Bayes's own difference (it too is better on
    low-FR states: low / high .854 / .726 for (.03 | 3), .826 / .736 for (.1 | 1), so -.128 / +.128 and -.090 / +.090;
    open (.03 | 3), filled (.1 | 1)); its key sits under the panel (the dotted red curve ran into it). At the end
    (running means): (.03 | 3) at 9.99e8 learned low .849 / .243 -> gap -.61 (its high-FR accuracy at 1 / 2 / 4 / 6 / 8
    / 10e8: .081 / .080 / .112 / .138 / .188 / .243, still rising), learned high .799 / .702 -> +.10 (its low-FR
    accuracy .545 / .692 / .767 / .781 / .796 / .799, levelled off); (.1 | 1) learned low .817 / .609 -> -.21, learned
    high .805 / .716 -> +.09 (= Joint Bayes's +.090). The matched-range version: figure_holdout_matched."""
    import glob as _g
    from matplotlib.lines import Line2D as L2
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    fr, r = Z["fr_T"], Z["r"]
    ok_b = Z["bayes"] == r
    upd = xunit == "updates"
    SC = 1e8                                                                       # x in units of 10^8 learned updates
    BLS = {(.03, 3): (0, (1, 1.2)), (.1, 1): "-"}                                  # the gap panel: band line style

    def share(run):
        f = os.path.join(TD, f"learnmask_{run}_s20261029", "sel.npz")
        return 1 - float(np.load(f)["sel"][:, 0].mean())

    def series(run, sel_eval):
        raws = sorted(int(os.path.basename(f)[len(run) + 1:-4]) for f in _g.glob(os.path.join(OUT, f"{run}_*.npz"))
                      if os.path.basename(f)[len(run) + 1:-4].isdigit())
        t = np.array([train_x(raw) for raw in raws], float)
        ans = [np.load(cache(run, raw))["ans"][:, -1] == r for raw in raws]
        out = []
        for sel in sel_eval:
            a = np.array([x[sel].mean() for x in ans])
            out.append((a, np.array([a[np.abs(t - v) <= 500].mean() for v in t])))
        return t, out

    n_col = len(chunks) + bool(gap)
    PW, GAP_ = 2.05, .62
    W_, H_ = .55 + n_col * PW + (n_col - 1) * GAP_ + .1, 2.7
    fig = plt.figure(figsize=(W_, H_))
    axes = [fig.add_axes([(.55 + j * (PW + GAP_)) / W_, .78 / H_, PW / W_, 1.55 / H_]) for j in range(n_col)]
    gax = axes[-1] if gap else None
    gmax = 0
    for j, (lo, hi) in enumerate(chunks):
        ax = axes[j]
        evals = ((fr > hi, "-"), (fr < lo, (0, (4, 2))))
        xmax = tmax = 0
        for side in ("lo", "hi"):
            run = band_name(side, lo, hi)
            t, curves = series(run, [e[0] for e in evals])
            x = t * share(run) * 8000 * 30 / SC if upd else t
            xmax, tmax = max(xmax, x.max()), max(tmax, t.max())
            for (a, sm), (_, ls) in zip(curves, evals):
                ax.plot(x, a, color=END_COL[side], lw=.4, alpha=.3, ls="-", zorder=2)
                ax.plot(x, sm, color=END_COL[side], lw=1.6, ls=ls, zorder=3)
            if gax is not None:                                                   # evaluation - training, smoothed
                (_, hi_sm), (_, lo_sm) = curves
                g_ = hi_sm - lo_sm if side == "lo" else lo_sm - hi_sm
                gax.plot(x, g_, color=END_COL[side], lw=1.5, ls=BLS[(lo, hi)], zorder=3)
                gmax = max(gmax, x.max())
        if gax is not None:                                                       # Joint Bayes's own differences
            d_b = ok_b[fr > hi].mean() - ok_b[fr < lo].mean()
            for side, v in (("lo", d_b), ("hi", -d_b)):
                gax.plot([1.03], [v], marker="<", ms=6, ls="none", clip_on=False, zorder=6,
                         transform=gax.get_yaxis_transform(), markeredgecolor=END_COL[side], markeredgewidth=1,
                         markerfacecolor=END_COL[side] if (lo, hi) == (.1, 1) else "white")
        ax.set_xlim(0, xmax * 1.02)
        if upd:
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        else:
            ticks = [0, 5000, 10000] if xmax < 12000 else [0, 10000, 20000, 30000]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{v // 1000}k" if v else "0" for v in ticks])
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .5, 1])
        ax.set_title(rf"low FR $<$ {fmt(lo)}, high FR $>$ {fmt(hi)} nats", fontsize=FL, pad=14)
        ax.annotate(f"{int(round(tmax)):,} episodes", (.5, 1), xycoords="axes fraction", xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=FT - 1, color="0.4")
        if j == 0:
            ax.set_ylabel("accuracy", fontsize=FL - .5, labelpad=2)
    xl = r"learned updates ($\times 10^8$)" if upd else "training episodes"
    for ax in axes:
        ax.tick_params(labelsize=FT - .5, pad=2, length=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel(xl, fontsize=FL - .5, labelpad=2)
    if gax is not None:
        gax.axhline(0, color="0.5", lw=.7, zorder=1)
        gax.set_xlim(0, gmax * 1.02)
        if upd:
            gax.xaxis.set_major_locator(plt.MaxNLocator(4))
        gax.set_ylim(-.85, .3)
        gax.set_yticks([-.75, -.5, -.25, 0, .25])
        gax.set_ylabel("accuracy difference", fontsize=FL - .5, labelpad=2)
        gax.set_title("evaluation $-$ training", fontsize=FL, pad=14)
        gax.annotate("both bands", (.5, 1), xycoords="axes fraction", xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=FT - 1, color="0.4")
    fig.canvas.draw()                                                              # the legends right under the x labels
    yb = min(a.xaxis.label.get_window_extent().transformed(fig.transFigure.inverted()).y0 for a in axes) - .015
    xc = (axes[0].get_position().x0 + axes[len(chunks) - 1].get_position().x1) / 2
    leg1 = fig.legend([L2([], [], color=END_COL[s], lw=1.6) for s in ("lo", "hi")], ["low FR", "high FR"],
                      title="trained on", loc="upper right", bbox_to_anchor=(xc - .02, yb), ncols=2, fontsize=FT - 1,
                      title_fontsize=FT - .5, frameon=False, handlelength=1.8, columnspacing=1.2)
    fig.add_artist(leg1)
    fig.legend([L2([], [], color="0.2", lw=1.6, ls=ls) for ls in ("-", (0, (4, 2)))], ["high FR", "low FR"],
               title="evaluated on", loc="upper left", bbox_to_anchor=(xc + .02, yb), ncols=2, fontsize=FT - 1,
               title_fontsize=FT - .5, frameon=False, handlelength=1.8, columnspacing=1.2)
    if gax is not None:                                                            # under the gap panel: no curve crosses it
        fig.legend([L2([], [], color="0.2", lw=1.5, ls=BLS[b]) for b in chunks] +
                   [L2([], [], color="0.2", marker="<", ls="none", ms=5)],
                   [rf"$<$ {fmt(lo)} $|$ $>$ {fmt(hi)}" for lo, hi in chunks] + ["Joint Bayes"],
                   title="FR band", loc="upper center", bbox_to_anchor=((gax.get_position().x0 + gax.get_position().x1) / 2, yb),
                   ncols=2, fontsize=FT - 1, title_fontsize=FT - .5, frameon=False, handlelength=1.8, columnspacing=1.2)
    path = path or fig_path("fig_learn_holdout_saturation_2col.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def holdout_matched_primitives(Z):
    """every number figure_holdout_matched draws, as a flat dict (savable with np.savez; a drawing function needs nothing
    else): cap (the matched range, 1e8 learned updates), bands (2, 2) = (.03, 3), (.1, 1); per band bi (0, 1) and
    learner side (lo = trained on low FR, hi = trained on high FR): x_{bi}_{side} learned updates / 1e8, t_{bi}_{side}
    training episodes, acc_{bi}_{side}_{high|low} the running mean (1,000 training episodes, on the full run) of the
    accuracy on the band's high- / low-FR test answers and raw_{bi}_{side}_{high|low} the per-checkpoint value, all cut
    at cap; jb_{bi}_{high|low} Joint Bayes's accuracy on the same answers, n_{bi}_{high|low} their counts; col_lo /
    col_hi the colours (red / green)"""
    import glob as _g
    fr, r = Z["fr_T"], Z["r"]
    ok_b = Z["bayes"] == r
    bands = ((.03, 3), (.1, 1))
    P = {"bands": np.array(bands, float), "col_lo": END_COL["lo"], "col_hi": END_COL["hi"]}
    xs = {}
    for bi, (lo, hi) in enumerate(bands):
        sets = (("high", fr > hi), ("low", fr < lo))
        for name, sel in sets:
            P[f"jb_{bi}_{name}"] = float(ok_b[sel].mean())
            P[f"n_{bi}_{name}"] = int(sel.sum())
        for side in ("lo", "hi"):
            run = band_name(side, lo, hi)
            sh = 1 - float(np.load(os.path.join(TD, f"learnmask_{run}_s20261029", "sel.npz"))["sel"][:, 0].mean())
            raws = sorted(int(os.path.basename(f)[len(run) + 1:-4]) for f in _g.glob(os.path.join(OUT, f"{run}_*.npz"))
                          if os.path.basename(f)[len(run) + 1:-4].isdigit())
            t = np.array([train_x(raw) for raw in raws], float)
            ans = [np.load(cache(run, raw))["ans"][:, -1] == r for raw in raws]
            xs[(bi, side)] = (t, t * sh * 8000 * 30 / 1e8)
            for name, sel in sets:
                a = np.array([x[sel].mean() for x in ans])
                P[f"raw_{bi}_{side}_{name}"] = a
                P[f"acc_{bi}_{side}_{name}"] = np.array([a[np.abs(t - v) <= 500].mean() for v in t])
    cap = min(x[-1] for _, x in xs.values())
    P["cap"] = float(cap)
    for (bi, side), (t, x) in xs.items():
        k = x <= cap * (1 + 1e-9)
        P[f"x_{bi}_{side}"], P[f"t_{bi}_{side}"] = x[k], t[k]
        for name in ("high", "low"):
            for q in ("acc", "raw"):
                P[f"{q}_{bi}_{side}_{name}"] = P[f"{q}_{bi}_{side}_{name}"][k]
    return P


def draw_holdout_matched(P, path=None, return_fig=False, opposite=False, key="top"):
    """draw fig_learn_holdout_matched.png from P = holdout_matched_primitives(Z) alone (the design: see
    figure_holdout_matched); opposite: its two-panel version, fig_learn_holdout_opposite.png (see
    figure_holdout_opposite), its key outside the panels: above them (key="top") or in a column right of them
    ("right"); asserts that every key inside a panel clears the curves and markers by >= 4 pt; return_fig: return the
    open figure instead of closing it"""
    import matplotlib.transforms as mtr
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["pdf.fonttype"] = 42                                                # a PDF copy embeds TrueType
    n = 2 if opposite else 3                                                         # panels
    W, L, G1, G2, R = 5.5, .36, .15, .46, .12                                        # inches, equal panels
    gaps = [G2] if opposite else [G1, G2]                                            # G1: no tick labels between
    B, AH, TOP = .56, 2.0, .2
    KG, KR, KT, KM = .2, .04, .1, .04                                                # outside key: gap, right / top
    F7, F8 = 7, 8                                                                    # points
    col = {"lo": P["col_lo"], "hi": P["col_hi"]}
    K, JB, ZERO = "0.15", "0.45", "0.6"
    LW = {"high": 2.0, "low": .8}                                                    # 3 panels: width = evaluated on
    LWC, SPINE = 1.2, .6                                                             # gap-styled curves; spines, ticks
    BLS = {0: (0, (1, 1.3)), 1: "-"}                                                 # line style per gap
    OPEN = {0: True, 1: False}                                                       # Joint Bayes: open = 2-decade gap
    MS, MEW = 5, .9
    CLEAR, SPAD = 4, 3                                                               # key clearance (asserted), pt
    XR = 10.0 if opposite else 11.0                                                  # x range; x > 10: Joint Bayes
    cap = float(P["cap"])
    edge = {(0, "lo"): r"3{\cdot}10^{-2}", (0, "hi"): r"3{\cdot}10^{0}", (1, "lo"): r"10^{-1}", (1, "hi"): r"10^{0}"}
    dec = {0: 2, 1: 1}

    fig = plt.figure(figsize=(W, B + AH + TOP))
    rend = fig.canvas.get_renderer()
    pt = fig.dpi / 72                                                                # pixels per point

    def table(ax, head, rows, row=9.6, line=8.4, gap=None):
        """build a key in points (origin: its header's baseline), drawn through an offset that is translated later:
        the header (if any), then one row per entry with its handles left of its label, whose further lines (split
        at newlines) run under the first, line pt apart; gap: {row index: extra pt above it}; handles are dicts of
        kind "chip" (colour), "line" (width / style) or "marker" """
        off = mtr.Affine2D()
        tr = off + mtr.Affine2D().scale(1 / 72) + fig.dpi_scale_trans + mtr.ScaledTranslation(0, 0, ax.transAxes)
        CH, GAP, MID = 4.2, 3, 2.3                                                   # chip height, gap, handle height
        cw = {"chip": 7, "line": 12, "marker": 6}                                    # handle widths
        arts = [ax.text(0, 0, head, fontsize=F7, transform=tr, ha="left", va="baseline")] if head else []
        yb = -row if head else 0
        for i, (hs, lab) in enumerate(rows):
            yb, u = yb - (row + (gap or {}).get(i, 0) if i else 0), 0
            for kw in hs:
                kw = dict(kw)
                kind = kw.pop("kind")
                u0, u = u, u + cw[kind]
                if kind == "chip":
                    arts.append(ax.add_patch(Rectangle((u0, yb + MID - CH / 2), u - u0, CH, transform=tr, lw=0,
                                                       clip_on=False, **kw)))
                elif kind == "marker":
                    arts += ax.plot([(u0 + u) / 2], [yb + MID], ls="none", transform=tr, clip_on=False, label="_key",
                                    **kw)
                else:
                    arts += ax.plot([u0, u], [yb + MID] * 2, transform=tr, clip_on=False, label="_key",
                                    solid_capstyle="butt", dash_capstyle="butt", **kw)
                u += GAP
            for k, t in enumerate(lab.split("\n")):
                arts.append(ax.text(u + .5, yb - k * line, t, fontsize=F7, transform=tr, ha="left", va="baseline"))
            yb -= lab.count("\n") * line
        return off, arts

    def boxes(arts):
        """display boxes (x0, y0, x1, y1) of a key's parts as printed: text extents, strokes with their width"""
        out = []
        for a in arts:
            if isinstance(a, Text) or isinstance(a, Rectangle):
                b = a.get_window_extent(rend)
                out.append((b.x0, b.y0, b.x1, b.y1))
                continue
            d = a.get_transform().transform(np.column_stack([np.asarray(v, float) for v in a.get_data()]))
            if len(d) == 1:
                h = (a.get_markersize() + a.get_markeredgewidth()) / 2 * pt
                out.append((d[0, 0] - h, d[0, 1] - h, d[0, 0] + h, d[0, 1] + h))
            else:
                h = a.get_linewidth() / 2 * pt
                out.append((d[:, 0].min(), d[:, 1].min() - h, d[:, 0].max(), d[:, 1].max() + h))
        return np.array(out)

    def ink(ax):
        """display points along every curve / marker / reference line of ax, each with its half-width (px)"""
        pts, hw = [], []
        for ln in ax.get_lines():
            if ln.get_label() == "_key":
                continue
            d = ln.get_transform().transform(np.column_stack([np.asarray(v, float) for v in ln.get_data()]))
            d = d[np.isfinite(d).all(1)]
            if ln.get_linestyle() not in ("None", "none", "", " ") and len(d) > 1:
                n_ = np.maximum(np.ceil(np.hypot(*(d[1:] - d[:-1]).T) / pt).astype(int), 1)  # <= 1 pt apart
                seg = np.concatenate([p + (q - p) * np.arange(k)[:, None] / k for p, q, k in zip(d[:-1], d[1:], n_)]
                                     + [d[-1:]])
                pts.append(seg)
                hw.append(np.full(len(seg), ln.get_linewidth() / 2 * pt))
            if ln.get_marker() not in (None, "None", "none", "", " "):
                h = (ln.get_markersize() + ln.get_markeredgewidth()) / 2 * pt
                g = np.linspace(-h, h, 7)
                sq = (d[:, None, None, :] + np.stack(np.meshgrid(g, g), -1)[None]).reshape(-1, 2)
                pts.append(sq)
                hw.append(np.zeros(len(sq)))
        return np.concatenate(pts), np.concatenate(hw)

    def clearance(bx, P_, hw):
        """min over the key's boxes bx (k, 4) of the distance to the ink P_ (n, 2) minus its half-width, in pt"""
        best = np.inf
        for x0_, y0_, x1_, y1_ in bx:
            dx = np.maximum(np.maximum(x0_ - P_[:, 0], 0), P_[:, 0] - x1_)
            dy = np.maximum(np.maximum(y0_ - P_[:, 1], 0), P_[:, 1] - y1_)
            best = min(best, (np.hypot(dx, dy) - hw).min())
        return best / pt

    def search(ax, bx, xs, avoid=None):
        """the shift (clearance in pt, sx, sy in px) of a key with boxes bx that puts its right edge at one of the data
        x in xs (None: its left edge SPAD pt right of the axis line), its height anywhere that keeps it SPAD pt inside
        both axis lines and under the axes top (away from the title), and clears the data (and the boxes avoid) most"""
        P_, hw = ink(ax)
        abb = ax.bbox
        xlo, ylo = ax.transData.transform((0, ax.get_ylim()[0])) + (SPAD + SPINE / 2) * pt
        best = (-np.inf, 0, 0)
        for xr in xs:
            sx = xlo - bx[:, 0].min() if xr is None else ax.transData.transform((xr, 0))[0] - bx[:, 2].max()
            if bx[:, 0].min() + sx < xlo - 1e-6:
                continue
            for sy in np.arange(ylo - bx[:, 1].min(), abb.y1 - SPAD * pt - bx[:, 3].max(), pt / 4):
                b = bx + [sx, sy, sx, sy]
                c = clearance(b, P_, hw)
                if avoid is not None:                                                # box to box, in pt
                    dx = np.maximum(np.maximum(b[:, None, 0] - avoid[None, :, 2], avoid[None, :, 0] - b[:, None, 2]), 0)
                    dy = np.maximum(np.maximum(b[:, None, 1] - avoid[None, :, 3], avoid[None, :, 1] - b[:, None, 3]), 0)
                    c = min(c, np.hypot(dx, dy).min() / pt)
                if c > best[0]:
                    best = (c, sx, sy)
        return best

    def place(ax, keys, xs, name):
        """slide the keys [(offset, artists), ...] (one or two) of ax into place: for each order, the first where it
        clears the data most, the second clear of the data and the first; keep the order whose worse key clears more"""
        fig.canvas.draw()
        bxs = [boxes(arts) for _, arts in keys]
        best = (-np.inf, None)
        for order in ([0], ) if len(keys) == 1 else ([0, 1], [1, 0]):
            sh, avoid, worst = {}, None, np.inf
            for k in order:
                c, sx, sy = search(ax, bxs[k], xs, avoid)
                sh[k], worst = (sx, sy), min(worst, c)
                avoid = bxs[k] + [sx, sy, sx, sy]
            if worst > best[0]:
                best = (worst, sh)
        c, sh = best
        for k, (off, _) in enumerate(keys):
            off.translate(sh[k][0] / pt, sh[k][1] / pt)
        assert c >= CLEAR, f"the {name} key clears the data by only {c:.1f} pt"

    def jb_marker(ax, x, y, edgec, open_):
        ax.plot([x], [y], marker="<", ms=MS, mew=MEW, ls="none", mec=edgec, mfc="white" if open_ else edgec,
                clip_on=False, zorder=6)

    KW = KH = 0                                                                      # the outside key (inches)
    if opposite:                                                                     # colour above line style
        kax = fig.add_axes([0, 0, 1, 1])
        kax.axis("off")
        brk = "\n" if key == "right" else " "
        rows = [([dict(kind="chip", color=col[s])], f"Trained on {a},{brk}evaluated on {b}")
                for s, a, b in (("lo", "low", "high"), ("hi", "high", "low"))]
        rows += [([dict(kind="line", color=K, lw=LWC, ls=BLS[bi])], f"{dec[bi]}-decade gap{brk}"
                  rf"$(\mathrm{{Low}} < {edge[(bi, 'lo')]}\ |\ \mathrm{{High}} > {edge[(bi, 'hi')]})$") for bi in (0, 1)]
        koff, karts = table(kax, None, rows, gap={2: 4})
        fig.canvas.draw()
        kb = boxes(karts)
        KW, KH = (kb[:, 2].max() - kb[:, 0].min()) / fig.dpi, (kb[:, 3].max() - kb[:, 1].min()) / fig.dpi
    kright = opposite and key == "right"
    H = B + AH + TOP + (KT + KH + KM if opposite and not kright else 0)
    fig.set_size_inches(W, H)
    rend = fig.canvas.get_renderer()
    PW = (W - L - sum(gaps) - (KG + KW + KR if kright else R)) / n
    x0s = [L + j * PW + sum(gaps[:j]) for j in range(n)]
    axes = [fig.add_axes([x0 / W, B / H, PW / W, AH / H]) for x0 in x0s]
    upt = PW * 72 / XR                                                               # points per x unit
    XA = cap + 7 / upt                                                               # one gap per panel: 7 pt past
    XM = {0: cap + 4.5 / upt, 1: cap + (12.5 if opposite else 11) / upt}             # both gaps: side by side
    if opposite:                                                                     # key: right column / centred above
        if kright:
            kax.set_position([(W - KR - KW) / W, B / H, KW / W, AH / H])
        else:
            kax.set_position([L / W, (B + AH + TOP + KT) / H, (x0s[-1] + PW - L) / W, KH / H])
        fig.canvas.draw()
        kb, bb = boxes(karts), kax.bbox
        dx = bb.x0 - kb[:, 0].min() if kright else (bb.x0 + bb.x1 - kb[:, 0].min() - kb[:, 2].max()) / 2
        dy = (bb.y0 + bb.y1 - kb[:, 1].min() - kb[:, 3].max()) / 2
        koff.translate(dx / pt, dy / pt)

    if opposite:                                                                     # a: each learner on the other set
        ax = axes[0]
        for bi in (0, 1):
            for side, name in (("lo", "high"), ("hi", "low")):
                ax.plot(P[f"x_{bi}_{side}"], P[f"acc_{bi}_{side}_{name}"], color=col[side], lw=LWC, ls=BLS[bi],
                        zorder=3)
        ax.set_title("evaluated on the other FR set", fontsize=F8, pad=4)
    else:                                                                            # a, b: accuracy, one gap each
        for bi in (0, 1):
            ax = axes[bi]
            for side in ("lo", "hi"):
                for name in ("low", "high"):
                    ax.plot(P[f"x_{bi}_{side}"], P[f"acc_{bi}_{side}_{name}"], color=col[side], lw=LW[name],
                            solid_capstyle="butt", zorder=3)
            for name in ("high", "low"):
                jb_marker(ax, XA, P[f"jb_{bi}_{name}"], JB, OPEN[bi])
            ax.set_title(f"{dec[bi]}-decade gap", fontsize=F8, pad=4)
        axes[1].tick_params(labelleft=False)
    for ax in axes[:-1]:
        ax.set_ylim(0, 1)
        ax.set_yticks(np.round(np.linspace(0, 1, 6), 6))
    axes[0].set_ylabel("accuracy", fontsize=F8, labelpad=2)

    gax = axes[-1]                                                                   # last: evaluation - training
    gax.plot([0, 10], [0, 0], color=ZERO, lw=.6, zorder=1)                           # zero, over the x axis's extent
    ys = []
    for bi in (0, 1):
        for side in ("lo", "hi"):
            hi_, lo_ = P[f"acc_{bi}_{side}_high"], P[f"acc_{bi}_{side}_low"]
            ys.append(hi_ - lo_ if side == "lo" else lo_ - hi_)
            gax.plot(P[f"x_{bi}_{side}"], ys[-1], color=col[side], lw=LWC, ls=BLS[bi], zorder=3)
        if not opposite:                                                             # Joint Bayes's own differences
            d = P[f"jb_{bi}_high"] - P[f"jb_{bi}_low"]
            jb_marker(gax, XM[bi], d, col["lo"], OPEN[bi])
            jb_marker(gax, XM[bi], -d, col["hi"], OPEN[bi])
            ys.append(np.array([d, -d]))
    y0 = round(np.floor(min(y.min() for y in ys) / .2) * .2, 6)                      # 1.0 wide: the accuracies' scale
    assert max(y.max() for y in ys) < y0 + 1, "the difference panel's data exceed the 1.0-wide range"
    gax.set_title(r"evaluation $-$ training", fontsize=F8, pad=4)
    gax.set_ylim(y0, y0 + 1)
    gax.set_yticks(np.round(np.linspace(y0, y0 + 1, 6), 6))
    gax.set_ylabel("accuracy difference", fontsize=F8, labelpad=2)

    for ax in axes:
        ax.set_xlim(0, XR)
        ax.set_xticks(np.arange(0, 10.1, 2))
        ax.spines["bottom"].set_bounds(0, 10)
        ax.spines[["left", "bottom"]].set_linewidth(SPINE)
        ax.tick_params(labelsize=F7, pad=2, length=2.5, width=SPINE)
        ax.spines[["top", "right"]].set_visible(False)

    if not opposite:                                                                 # keys, each slid into place
        for bi in (0, 1):
            lab = {"low": rf"low FR $(\!<\!{edge[(bi, 'lo')]})$", "high": rf"high FR $(\!>\!{edge[(bi, 'hi')]})$"}
            rows = [([dict(kind="chip", color=col[s]), dict(kind="line", color=K, lw=LW[nm])], lab[nm])
                    for s, nm in (("lo", "low"), ("hi", "high"))]
            place(axes[bi], [table(axes[bi], "trained on / evaluated on", rows)], [10], "abc"[bi])
        tri = lambda open_: dict(kind="marker", marker="<", ms=MS, mew=MEW, mec=JB, mfc="white" if open_ else JB)
        place(gax, [table(gax, "Joint Bayes / RNNs",
                          [([tri(OPEN[bi]), dict(kind="line", color=K, lw=LWC, ls=BLS[bi])], f"{dec[bi]}-decade gap")
                           for bi in (0, 1)])], np.arange(7, 10.01, .25), "c")

    for j in (range(n) if opposite else [1]):                                        # x labels: one per panel / under b
        fig.text((x0s[j] + PW / 2) / W, (B - .3) / H, r"learned updates ($\times 10^8$)", ha="center",
                 va="baseline", fontsize=F8)
    for j, x0 in enumerate(x0s):                                                     # panel letters, the paper's style
        fig.text((x0 + PW / 2) / W, .05 / H, f"({'abc'[j]})", ha="center", va="baseline", fontsize=F8,
                 family="DejaVu Sans")
    path = path or fig_path("fig_learn_holdout_opposite.png" if opposite else "fig_learn_holdout_matched.png")
    fig.savefig(path, dpi=DPI)
    if return_fig:
        return fig
    plt.close(fig)
    print("saved", path)


def figure_holdout_matched(Z, path=None):
    """fig_learn_holdout_matched.png, at print size (5.5 x 2.76 in, >= 7 pt): the two FR-band hold-outs over the range
    of learned updates both bands reach (John 11-14: "show the figure with an xlim for where we have matched updates";
    polish 1: "make high and low FR, thick and thin lines, so we can reserve dashed and dotted linestyles for panel
    c ... clarify why the 4 bayes triangles differ ... put the legends into the panels ... show the structure of 0.03
    and 3, and 0.1 and 1"; polish 2: "put the legends from panel a and b into both panels a and b (without the joint
    bayes line or scatters) ... put the actual decade into 'low FR (< decade)' ... remove the bars ... fit titles into
    1 line"; polish 3: "ooof this is a little rough", redesigned from four candidate designs, two judges and a
    synthesis; polish 4 (11-15): "put both the color and size legend on the right or left of the text"; drawn by
    draw_holdout_matched from holdout_matched_primitives' numbers alone; two-panel version: figure_holdout_opposite).
    Columns a, b: one band each, titled by the FR gap between its two sets (.03 | 3: 2 decades, .1 | 1: 1 decade; both
    gaps centred at ~.3 nats, .03 x 3 = .1 x 1 = .1; the probes' median FR_T is .45), the edges in powers of ten in each
    panel's key. Colour = trained on, line width = evaluated on (thick: the band's high-FR test answers, thin: its
    low-FR ones); lines: running means over 1,000 training episodes, computed on the full runs, then cut at the matched
    range (the (.03 | 3) runs' last checkpoint, 9.99e8 learned updates = 53,360 training episodes; (.1 | 1): 20,160 of
    its 30,000). Grey triangles: Joint Bayes's own accuracy on the low-FR (upper) and high-FR (lower) answers: .854 /
    .726 (a, open) and .826 / .736 (b, filled). c: evaluation minus training accuracy of each network (red: high-FR
    minus low-FR answers, green: the reverse), line style = gap (dotted: 2 decades, solid: 1 decade); triangles: the
    same two differences for Joint Bayes (the differences of the grey triangles in a and b: -.128 / +.128 open, -.090 /
    +.090 filled, coloured by learner; Joint Bayes is less accurate on high-FR answers, the more so the wider the gap),
    i.e. where a learner matching Joint Bayes on both sets would sit.
    Layout: one row of three equally wide axes (1.47 x 2.0 in) on one baseline, all at one vertical scale (1.0 per axes
    height, ticks every .2 that line up across the row; c spans the 1.0-wide window from its data's minimum floored to
    .2, -.8 to .2), one x label (under b), the letters (8 pt, the titles' size) centred under the panels. The x axis line (and c's zero line)
    stops at 10; the column right of it holds Joint Bayes's triangles, centred 7 pt past the matched end in a, b and in
    c side by side per gap (open 4.5 pt, filled 11 pt past). Keys: one table per panel in the room its own curves leave,
    both handles left of each label, in the order the header names them; a, b: "trained on / evaluated on" = colour
    chip, line width; rows = the panel's two FR sets with its own edges, no Joint Bayes entry; c: "Joint Bayes / RNNs"
    = grey open / filled triangle (as in a, b), dotted / solid line; rows = the two gaps. Each table is right-aligned
    (a, b at x = 10; c at the best x in 7-10, now 8.75) and slid, at least 3 pt inside the axis lines and under the
    axes top, to where it clears every curve and marker most, handles included (asserted >= 4 pt; now a 7.4 pt,
    between the green curves and the thick red one; b 5.5 pt, under the thick red one; c 5.3 pt, between zero and the
    solid red one).
    Test set: the 10,000 probe trajectories (50 environments x 200; both variables, n answers in the band: 2,255 low /
    3,751 high for (.03 | 3), 4,817 / 6,898 for (.1 | 1)). At the matched end (9.99e8 / 9.89e8): trained on high FR
    -> low-FR answers .799 / .798; trained on low FR -> high-FR answers .243 / .566."""
    return draw_holdout_matched(holdout_matched_primitives(Z), path)


def figure_holdout_opposite(Z, path=None):
    """fig_learn_holdout_opposite.png, the two-panel version of figure_holdout_matched and the one John keeps (11-15:
    "maybe let's try showing only the trained on evaluated on opposites, reducing panels a and b into 1 panel where the
    linestyle indicated decade"; then "let's stick with the 2 panel ... legend style 'Trained on high, evaluated on
    low' (etc) and with both legends just the first panel. Also let's get rid of the bayes markers, I think it's still
    too confusing"): same size, data, smoothing and matched range (x 0-10; no Joint Bayes column). a: each learner's
    accuracy on the OTHER set only (red: trained on low FR, evaluated on its band's high-FR answers; green: the
    reverse), line style = gap (dotted: 2 decades, solid: 1 decade, as in b). b: the last panel of
    figure_holdout_matched (evaluation - training) without the Joint Bayes markers. One key, centred above both
    panels (John 11-15: the gap legend as "X-decade gap (Low < Y | High > Z)", "put it below the other legend. we can
    adjust the height / width of the figure if necessary, you can also move both legends around"): colour chips
    "Trained on low, evaluated on high" / "Trained on high, evaluated on low" (FR is named in a's title, "evaluated on
    the other FR set"), then, 4 pt lower, line styles "2-decade gap (Low < 3e-2 | High > 3e0)" / "1-decade gap (Low <
    1e-1 | High > 1e0)". The stacked key (about 141 x 40 pt) fits in neither panel without stretching its axes (a's
    room above the green curves is 29 pt), so it sits outside: 0.1 in above the titles, figure 5.5 x 3.46 in, two
    equal axes 2.28 x 2.0 in, one x label each. draw_holdout_matched(..., key="right") puts it in a column right of
    the panels instead (labels wrapped after "Trained on low," and "gap"; panels 1.49 in wide; 5.5 x 2.76 in). For the
    caption, Joint
    Bayes: .854 / .826 on the low-FR answers, .726 / .736 on the high-FR ones (2- / 1-decade gap), so its own
    evaluation - training is +.128 / +.090 (green's direction) and -.128 / -.090 (red's). At the matched end (9.99e8 /
    9.89e8 learned updates): green .799 / .798, red .243 / .566; evaluation - training green +.097 / +.091, red
    -.605 / -.253."""
    return draw_holdout_matched(holdout_matched_primitives(Z), path, opposite=True, key="top")


if __name__ == "__main__":
    if "--end" in sys.argv:                                                       # the end-of-training checkpoints
        harvest([band_name(sd_, *b) for b in BANDS for sd_, _ in BROWS] + [BRAND], END)
    if "--data" in sys.argv:
        harvest([a for a in sys.argv[sys.argv.index("--data") + 1:] if not a.startswith("--")] or None)
    Z = load()
    if "--numbers" in sys.argv:
        numbers(Z)
    if "--numbers" in sys.argv:
        band_numbers(Z)
    if "--thresholds" in sys.argv:                                                # the 11-10 morning design
        figure(Z, path=fig_path("fig_learn_holdout_thresholds.png"))
    if "--one" in sys.argv:                                                       # only the one-panel saturation figure
        figure_saturation_one(Z)
        sys.exit()
    if "--primitives" in sys.argv:                                                # the matched figure's numbers -> npz
        out_ = sys.argv[sys.argv.index("--primitives") + 1]
        np.savez(out_, **holdout_matched_primitives(Z))
        print("saved", out_)
        sys.exit()
    if "--matched" in sys.argv:                                                   # the matched-updates figures (11-14)
        figure_holdout_matched(Z)
        figure_holdout_opposite(Z)
        sys.exit()
    if "--cols" in sys.argv:                                                      # only the one-column-per-band one
        figure_saturation_cols(Z)
        sys.exit()
    figure_bands(Z)
    figure_saturation(Z)
    figure_saturation_one(Z)
    if "--long" in sys.argv:
        figure_long(Z)
