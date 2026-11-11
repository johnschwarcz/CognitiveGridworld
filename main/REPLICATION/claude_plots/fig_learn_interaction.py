"""fig_learn_interaction.png: do the networks combine observations, or add them up? Through training, for the Trained
RNN, the ESN and the ESN + MLPs (the ESN's frozen LSTM between 5 trained layers before and 5 after it).

Measure: the mixed second difference of the read-out increment at step t (the update of the answer's logits, centred
per variable) under two flipped observation bits,
    D_i D_j inc_t = inc_t(i and j flipped) - inc_t(i flipped) - inc_t(j flipped) + inc_t(intact),
the part of bit i's effect on the update that depends on bit j. Mean absolute value per value of a variable, in nats.
  a  within a step: both bits in the observation o_t (10 bit pairs, t = 5, 10, ..., 30), the network stepped from its
     intact state at t - 1 (td_tt_mixed_diff0).
  b  through time: bit i in o_t, bit j in an earlier observation o_s, s = t - 1, 2, 3, 5, 8, 12 (32 (t, s) pairs x 25
     bit pairs), the network replayed from s (td_tt_mixed_diff).
Naive Bayes is 0 in both (its update is a sum over bits and steps). Bayes (dashed) is not: the evidence a bit carries
about one variable depends on what the other bits, now and earlier, say about the other variable. "Bayes without
memory" (dotted in a; td_analysis' window 1) combines the five bits of each observation jointly, with a uniform belief
about the other variable before every observation: 0 through time, six times Bayes's size within a step. Any linear
reservoir read out linearly is 0 through time. Bands: 5-95% over 2,000 resamples of the 50 test environments. The two
panels have their own y scales (a reaches 8.5 times Bayes's).

RESULTS (mean |MD| per value, nats; fig_learn_interaction.py --numbers)
  References: Bayes .0754 within a step, .0748 through time; Bayes without memory .4706 within a step, 0 through time;
  naive Bayes 0 in both.
  Within a step, at 240 / 880 / 2,400 / 8,000 / 39,996 training episodes [5-95% at the end]; % of Bayes's:
      Trained RNN  .0480 / .0730 / .0739 / .0853 / .0883 [.0862, .0904];  64 /  97 /  98 / 113 / 117%
      ESN          .0159 / .0259 / .0424 / .0694 / .1147 [.1100, .1191];  21 /  34 /  56 /  92 / 152%
      ESN + MLPs   .2331 / .6443 / .4665 / .2638 / .2049 [.1998, .2107]; 309 / 855 / 619 / 350 / 272% (peak at 880)
  Through time:
      Trained RNN  .0065 / .0199 / .0356 / .0540 / .0542 [.0529, .0554];   9 /  27 /  48 /  72 /  72% (peak at 24,000)
      ESN          .0002 / .0004 / .0009 / .0020 / .0049 [.0048, .0050];   0 /   1 /   1 /   3 / 6.6%
      ESN + MLPs   .0319 / .0947 / .1112 / .0896 / .0704 [.0688, .0721];  43 / 127 / 149 / 120 /  94% (peak at 2,000)
  Through time at the end: Trained RNN / ESN 11.0, ESN + MLPs / ESN 14.3.
  Range at the end (t = 15-30), lag 1 / 2 / 3 / 5 / 8 / 12: Trained RNN .0525 .0444 .0403 .0363 .0344 .0332; ESN .0133
  .0071 .0041 .0016 .0004 .0001 (falling by 30-50% per step of lag); ESN + MLPs .0834 .0682 .0635 .0561 .0447 .0325.
  Regression of the network's mixed difference on Bayes's at the end, kappa (cosine): Trained RNN +.193 (+.40), ESN
  +.002 (+.03), ESN + MLPs +.222 (+.38).

READING
  The ESN's update is additive across steps, like naive's: what an observation adds to its answer barely depends on
  the observations before it (6.6% of Bayes's interaction through time, falling by 30-50% per step of lag), although
  the bits within one observation do interact (152% of Bayes's size at the end). The two networks that end on Bayes's
  accuracy at high factorization regret (fig_learn_profile a, c, d) both make the present observation's effect depend
  on the past (72% and 94% of Bayes's), by different routes: the Trained RNN with Bayes-sized interactions on both
  axes, the ESN + MLPs with 2.7 to 8.5 times Bayes's within the step.

NOT CLAIMED
  - that the missing through-time interaction is why the ESN is wrong at high FR. A rule that is additive through time
    can be right there: fitted per environment and variable, a rule linear in the counts of the 32 observation patterns
    (0 through time by construction) reaches .60 in the highest FR tenth and .67 over all states (Bayes .72 / .77,
    Bayes without memory .30 / .61, ESN .045 / .43, naive .015 / .46; fr_profile/checks/additive_fit.txt; fitted
    numbers are lower bounds for their family). So at most .12 of the ESN's .67 deficit in the highest tenth is owed to
    through-time combination as such; the rest is in what it reads from each observation.
  - that the size of the interaction is what makes a network right at high FR. It is a signature of a non-additive
    update, not a sufficient cause: the Trained RNN with the gradient blocked at the highest-FR half keeps 71% of the
    unblocked network's through-time interaction at 2,400 episodes (.0250 vs .0352) while its accuracy in the highest
    FR tenth is .22 vs .62; and the ESN + MLPs overshoots Bayes's size (149% at 2,400 episodes) before it is accurate
    there (.50).
  - that the networks' interaction is Bayes's, probe by probe: the regression of a network's mixed difference on
    Bayes's is ~.2 at the end for both (kappa; cosine ~.4), 0 for the ESN.
  - that the Trained RNN computes as Bayes does: sizes only (117% of Bayes's within a step, 72% through time; Bayes
    without memory would be 0 through time and five times the network's size within a step).
  - where the ESN + MLPs makes its interaction (its read-out is non-linear, so it can be made after the reservoir).
  - one training run per network.

Data (read only): through_training/mixed_diff/{summary, summary0, summary_deep}.npz (td_tt_mixed_diff,
td_tt_mixed_diff0, td_tt_mixed_diff_deep).
usage:  python fig_learn_interaction.py [--numbers]"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402

MD_DIR = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics", "through_training", "mixed_diff")
NETS = (("trained", "Trained RNN", "#1f77b4"), ("echo", "ESN", "#ff7f0e"), ("mlp55", "ESN + MLPs", "#8c564b"))
X0, XMIN, XMAX = 10, 8, 5.2e4                    # the initial network is drawn at X0 ("0")
TS, FL, FT = 10.5, 10, 9.5                       # title, axis label, tick / annotation sizes (points, at print size)
FIG_W, DPI = 5.5, 300                            # print width (text width), inches


def load():
    """per network: x (training episodes), and mean |MD| with its 5-95% band within a step and through time"""
    Z, Z0, D = (np.load(os.path.join(MD_DIR, f)) for f in ("summary.npz", "summary0.npz", "summary_deep.npz"))
    out = dict(bayes=float(Z["md_bayes"]), bayes0=float(Z0["md0_bayes_allt"][0]), memfree0=float(Z0["md0_memfree_allt"][0]))
    assert abs(float(D["md_bayes"]) - out["bayes"]) < 1e-9 and abs(float(D["md0_bayes"]) - out["bayes0"]) < 1e-9
    for run in ("trained", "echo"):
        assert np.allclose(Z[f"x_{run}"], Z0[f"x_{run}"])
        out[run] = dict(x=Z[f"x_{run}"], time=Z[f"md_{run}"], time_lo=Z[f"md_{run}_lo"], time_hi=Z[f"md_{run}_hi"],
                        step=Z0[f"md0_allt_{run}"], step_lo=Z0[f"md0_allt_{run}_lo"], step_hi=Z0[f"md0_allt_{run}_hi"],
                        kappa=Z[f"kappa_{run}"], energy=Z[f"energy_{run}"], lag=Z[f"md_lag_{run}"])
    out["mlp55"] = dict(x=D["x"], time=D["md"], time_lo=D["md_lo"], time_hi=D["md_hi"], step=D["md0"],
                        step_lo=D["md0_lo"], step_hi=D["md0_hi"], kappa=D["kappa"], energy=D["energy"], lag=D["md_lag"])
    for run, _, _ in NETS:
        d = out[run]
        d.update({k: np.asarray(v, float) for k, v in d.items()})
        d["x"] = np.where(d["x"] > 0, d["x"], X0)
        assert np.array_equal(d["x"], out["trained"]["x"])
    out["lags"] = np.asarray(Z["lags"])
    return out


def numbers(I):
    x = I["trained"]["x"]
    at = [int(np.argmin(np.abs(x - v))) for v in (240, 880, 2400, 8000, 39996)]
    print(f"Bayes: within a step {I['bayes0']:.4f}, through time {I['bayes']:.4f}; memory-free Bayes within a step "
          f"{I['memfree0']:.4f} (0 through time); naive 0 in both")
    for key, name, ref in (("step", "within a step", I["bayes0"]), ("time", "through time", I["bayes"])):
        print(f"{name}, training episodes " + " / ".join(f"{x[i]:,.0f}" for i in at) + "  [5-95% at the end]; % of Bayes's")
        for run, nm, _ in NETS:
            d = I[run]
            print(f"  {nm:12s} " + " / ".join(f"{d[key][i]:.4f}" for i in at)
                  + f"  [{d[key + '_lo'][-1]:.4f}, {d[key + '_hi'][-1]:.4f}]; "
                  + " / ".join(f"{100 * d[key][i] / ref:.0f}%" for i in at) + f"; peak {d[key].max():.4f} at "
                  f"{x[int(d[key].argmax())]:,.0f}")
    print("through time at the end: kappa (cosine = kappa / sqrt(energy)); by lag " + ", ".join(map(str, I["lags"]))
          + " (t = 15-30)")
    for run, nm, _ in NETS:
        d = I[run]
        print(f"  {nm:12s} {d['kappa'][-1]:+.3f} ({d['kappa'][-1] / np.sqrt(d['energy'][-1]):+.2f}); "
              + " ".join(f"{v:.4f}" for v in d["lag"][-1]))
    print("ratio Trained RNN / ESN through time at the end: "
          f"{I['trained']['time'][-1] / I['echo']['time'][-1]:.1f}; ESN + MLPs / ESN "
          f"{I['mlp55']['time'][-1] / I['echo']['time'][-1]:.1f}")


def panel(ax, I, key, ref, ymax, ticks):
    for run, _, c in NETS:
        d = I[run]
        ax.fill_between(d["x"], d[key + "_lo"], d[key + "_hi"], color=c, alpha=.22, lw=0, zorder=2)
        ax.plot(d["x"], d[key], color=c, lw=1.7, marker="o", ms=2.0, mew=0, zorder=5 if run == "echo" else 4,
                solid_capstyle="round")
    ax.axhline(ref, color="k", lw=1.0, ls=(0, (4, 2)), zorder=3)
    ax.axhline(0, color="0.35", lw=.7, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks([X0, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.minorticks_off()
    ax.set_ylim(-.03 * ymax, ymax)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:g}" for v in ticks])
    ax.tick_params(labelsize=FT, pad=2, length=3)
    ax.spines[["top", "right"]].set_visible(False)


def title(ax, letter, name, formula):
    for text, dy, fs, dx in ((formula, 5, FL, 0), (letter, 21, TS, 0), (name, 21, TS, 13)):
        ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points", ha="left",
                    va="baseline", fontsize=fs, annotation_clip=False)


def figure(I):
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    left, right, gap, ph = .58, .08, .52, 1.50
    pw = (FIG_W - left - right - gap) / 2
    head, ttl, tick, xlab, strip, note = .21, .42, .21, .19, .27, .40
    H = .03 + head + ttl + ph + tick + xlab + strip + note + .02
    fig = plt.figure(figsize=(FIG_W, H))
    top = .03 + head + ttl
    add = lambda col: fig.add_axes([(left + col * (pw + gap)) / FIG_W, 1 - (top + ph) / H, pw / FIG_W, ph / H])
    ftext = lambda x, t, s, **k: fig.text(x / FIG_W, 1 - t / H, s, **k)
    note_kw = dict(textcoords="offset points", fontsize=FT)

    ftext(left - .44, .03 + head - .02,
          r"Interaction of two observation bits in the answer's update,  mean $|\Delta_i \Delta_j\, \mathrm{inc}_t|$",
          fontsize=FL, ha="left", va="bottom")
    for col, (key, ref, letter, name, formula, ymax, ticks) in enumerate((
            ("step", I["bayes0"], "a", "within a step", r"bits $i \neq j$ of the same observation $o_t$", .78, (0, .2, .4, .6)),
            ("time", I["bayes"], "b", "through time", r"bit $i$ of $o_t$, bit $j$ of an earlier $o_s$", .16, (0, .05, .1, .15)))):
        ax = add(col)
        panel(ax, I, key, ref, ymax, ticks)
        title(ax, letter, name, formula)
        ax.annotate("Bayes", xy=(XMIN, ref), xytext=(4, 2.5), ha="left", va="bottom", **note_kw)
        if key == "step":                       # what a rule without memory needs within the step: 6x Bayes's
            ax.axhline(I["memfree0"], color="k", lw=.9, ls=(0, (1, 1.6)), zorder=3)
            ax.annotate("Bayes without\nmemory", xy=(XMIN, I["memfree0"]), xytext=(4, 2.5), ha="left", va="bottom",
                        linespacing=1.1, **note_kw)
        ax.annotate("naive Bayes: 0" if key == "step" else "naive Bayes: 0\nBayes without memory: 0", xy=(0, 1),
                    xycoords="axes fraction", xytext=(4, -2), ha="left", va="top", color="0.35", linespacing=1.15, **note_kw)
    fig.text(.012, 1 - (top + ph / 2) / H, "interaction (nats)", rotation=90, fontsize=FL, ha="left", va="center")
    ftext(FIG_W / 2, top + ph + tick + .02, "training episodes", fontsize=FL, ha="center", va="top")

    y0 = top + ph + tick + xlab + .07
    hs = [Line2D([], [], color=c, lw=1.7, marker="o", ms=2.0, mew=0) for _, _, c in NETS]
    fig.legend(hs, [n for _, n, _ in NETS], loc="upper center", bbox_to_anchor=(.5, 1 - y0 / H), ncol=3,
               fontsize=FT, frameon=False, handlelength=1.6, columnspacing=1.6, handletextpad=.45, borderpad=0,
               borderaxespad=0)
    k = len(I["trained"]["x"])
    ftext(FIG_W / 2, y0 + strip,
          f"50 test environments $\\times$ 20 trajectories = 1,000 probes; {k} checkpoints per network;\n"
          "one training run each; bands: 5-95% over resampled environments; two y scales",
          fontsize=FT, ha="center", va="top", color="0.15", linespacing=1.35)
    path = fig_path("fig_learn_interaction.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    I = load()
    if "--numbers" in sys.argv:
        numbers(I)
    figure(I)
