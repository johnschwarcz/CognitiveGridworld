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
        raws = [raw for raw in STAGES + EXTRA if os.path.exists(cache(run, raw))]
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
END_SHADE = {(.03, 3): 1.0, (.1, 1): .55}                                         # narrower chunks darker


def end_profile(Z, run):
    """accuracy per FR_T bin at each END checkpoint, then the mean over the checkpoints (and their SD), or None"""
    have = [raw for raw in END if os.path.exists(cache(run, raw))]
    if len(have) < len(END) // 2:
        return None
    P = np.stack([profile(np.load(cache(run, raw))["ans"][:, -1] == Z["r"], Z["bin"]) for raw in have])
    return P.mean(0), P.std(0), len(have)


def end_panel(fig, Z, W_, H, xl, w, top, h, cols):
    """John 11-11 00:40: "a panel for the end of training results, just the last however many training episodes (1k
    or 2k, however many we can get for a stable batch of data), with all the different conditions, green curves for
    training at high FR, red curves for training at low FR. Again with solid curves for the training range and dashed
    curves for the testing range": accuracy at T by FR_T bin, the mean over the END checkpoints (raw 9,600 - 12,000
    every 200: 13 checkpoints over the last ~1,920 training episodes); grey: the random control"""
    from matplotlib.colors import to_rgb
    ax = fig.add_axes([xl / W_, 1 - (top + h) / H, w / W_, h / H])
    axes(ax, True, True)
    ax.set_xlim(10 ** -3.5, 10 ** 2.5)
    x = Z["x"]
    hs, lb = [], []
    for lo, hi in cols:
        for side in ("lo", "hi"):
            e = end_profile(Z, band_name(side, lo, hi))
            if e is None:
                continue
            k = END_SHADE.get((lo, hi), .75)
            c = tuple(1 - k * (1 - v) for v in to_rgb(END_COL[side]))
            split_curve(ax, x, e[0], lo if side == "lo" else hi, side, c, False)
            hs.append(Line2D([], [], color=c, lw=1.2))
            lb.append(("FR $<$ " if side == "lo" else "FR $>$ ") + fmt(lo if side == "lo" else hi))
    e = end_profile(Z, BRAND)
    if e is not None:
        ax.plot(x, e[0], color="0.55", lw=1.0, zorder=2)
        hs.append(Line2D([], [], color="0.55", lw=1.2))
        lb.append(r"random (count of .1 $|$ 1)")
    ax.set_ylabel("accuracy", fontsize=FL - .5, labelpad=2)
    ax.set_xlabel(r"FR$_T$ of the probe state (nats)", fontsize=FL, labelpad=3)
    ax.annotate("End of training", xy=(.5, 1), xycoords="axes fraction", xytext=(0, 16), textcoords="offset points",
                ha="center", va="bottom", fontsize=TS)
    ax.annotate(f"mean over the last {train_x(END[-1]) - train_x(END[0]):,} training episodes", xy=(.5, 1),
                xycoords="axes fraction", xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                fontsize=FT - 1.5, color="0.3")
    ax.legend(hs, lb, title="learned from", title_fontsize=FT - 1, loc="lower left", fontsize=FT - 1, frameon=False,
              handlelength=1.6, labelspacing=.3, borderaxespad=.2)


def figure_bands(Z, cols=None, path=None):
    """John 11-10 (13:58): "Does learn_holdout control for the amount of training data and weight updates? Everything
    should be balanced. Also maybe switching from red for blocked, let's go green for the learned chunk, and order the
    panels so the columns have an equal chunk of FR (i.e. equal size of green chunk). Also adding theta = 10 could be
    good." Columns: a pair of equal FR chunks (lo, hi), lo * hi = .1, so on the log axis (centred on .32 nats) the two
    green chunks have the same width; rows: the network that learned only from updates with FR < lo / FR > hi
    (train_learn_masks band modes): every network learns exactly n(t) = min(#FR < .01, #FR > 10) updates at step t of
    every batch (2.9% of all), the same in every panel, for the same 2,400 training episodes (2,400 weight updates; the
    loss is the mean over the learned terms). Grey: the same count of random updates (excl_bandrand) at 2,400."""
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    b, x, r = Z["bin"], Z["x"], Z["r"]
    naive, bayes = profile(Z["naive"] == r, b), profile(Z["bayes"] == r, b)
    cols = band_columns(Z) if cols is None else cols
    nc = len(cols)
    if not nc:
        print("no band pair has reached 2,400 episodes yet: nothing drawn")
        return
    allst = STAGES + EXTRA                                                        # the long runs: every stage to 9,600
    last = max(raw for raw in allst if all(at_T(Z, band_name(s_, *b), raw) is not None for b in cols for s_, _ in BROWS))
    rnd = at_T(Z, BRAND, last)
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
            ftext(x0[k] + pw / 2, top0 - .06, f"{100 * (1 - st['blocked']):.1f}% of updates learned", fontsize=FT - 1.5,
                  ha="center", va="bottom", color="0.3")
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
            if rnd is not None and (lo, hi) == (.1, 1):                  # its count matches (.1 | 1) only
                ax.plot(x, profile(rnd == r, b), color="0.6", lw=1.0, zorder=2, solid_capstyle="round")
            run = band_name(side, lo, hi)
            sts = [raw for raw in allst if raw <= last and at_T(Z, run, raw) is not None]
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
          Line2D([], [], color="0.35", lw=1.2, ls=GEN_LS), Line2D([], [], color="0.6", lw=1.0)]
    lb = ["learned FR range", "accuracy there", "generalization (outside it)", r"random, same count (.1 $|$ 1)"]
    fig.legend(hs, lb, loc="upper left", bbox_to_anchor=((left + 2.05) / W_, 1 - (y0 + .02) / H), ncol=1,
               fontsize=FT - .5, frameon=False, handlelength=1.6, columnspacing=1.2, handletextpad=.4, borderpad=0,
               borderaxespad=0)
    cw = 1.25
    cax = fig.add_axes([(left + .05) / W_, 1 - (y0 + .16) / H, cw / W_, .075 / H])
    cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal")
    cax.set_xlim(train_x(STAGES[0]), train_x(last))
    cb.set_ticks([100, 1000] + ([] if train_x(last) < 5000 else []))
    cb.set_ticklabels(["100", "1k"])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=FT - 1, pad=1.5, length=2)
    cb.outline.set_linewidth(.5)
    ftext(left + .05, y0 + .06, "training episodes", fontsize=FT - .5, ha="left", va="bottom")
    ftext(left + .05 + cw + .05, y0 + .125, f"{train_x(last):,}", fontsize=FT - .5, ha="left", va="center")
    end_panel(fig, Z, W_, H, x0[-1] + pw + egap, ew, top0, 2 * ph + rgap, cols)
    path = path or fig_path("fig_learn_holdout.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("saved", path)


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
    figure_bands(Z)
    if "--long" in sys.argv:
        figure_long(Z)
