"""Figure for the ESN question: how sensitive are the ESN results to the reservoir's hyperparameters (spectral radius,
leak rate, sparsity) and to read-out regularization, and can a large reservoir with tuned hyperparameters beat Naive
Bayes? One panel per item of the question; y = accuracy on novel environments at T = 30 in every panel
(argmax of the goal variable's read-out). Reference levels: Joint and Naive Bayes; the trained RNN and the paper's ESN (a
frozen LSTM with a trained read-in and read-out), both N = 1,000 with their own read-outs. John 11-10 15:10: "we dont
need to show the trained RNN or horizontal lines besides joint and naive bayes" -> the paper's ESN is one point in the
size panel; the legend says what a dot / line / diamond is (trained read-in: one run per point, no hidden scatter).

John 11-10 09:30: show only the trained read-in and the channel-local read-in (the dense, classical read-in, .10-.26
at every setting and N up to 20,000, is still loaded and printed, not drawn).
Three kinds of echo-state network (leaky-tanh reservoir, never trained; N = 1,000 except in the last panel):
  dense          the classical ESN, every unit reads all 15 inputs through random weights: esn_convex.py sweep (294 random
                 settings) + the dense half of sweep2 (120 random settings; Gaussian / orthogonal / cycle reservoirs)
  channel-local  a fixed random read-in in which a unit reads one observation bit and that channel's two Z values (the
                 other half of sweep2)
  trained        esn_trained_readin.py: read-in and read-out trained by SGD around the frozen reservoir (40,000 updates of
                 2,000 fresh episodes), one knob varied at a time from rho .9, alpha 1, p .1, N 1,000; drawn as runs finish
Only classical leaky-tanh ESNs are drawn (John 11-11 01:10: "keep it simple, just classic ESNs"; the two marks below were
added at 00:55 and removed). For the record (John 11-11: "is this an LSTM ESN like in
the paper or everything in this figure is with classic simple ESNs?"): the open diamond = esn_trained_readin.py task 10,
the paper's core (a frozen random LSTM, N 1,000) with the same trained read-in / read-out and budget as the green line
(own read-out .395, exact .402; classical core at N 1,000: .389 / .399), and the grey triangle = the paper's ESN itself
(frozen LSTM, its own training: 400M episodes vs 80M here; .430).
A dot = one setting with the read-out that scores best on the validation set (ridge or multinomial logistic on the final or
the time-summed state; sweep2 adds squared states and soft targets = Joint's posterior at T); thin line = the best setting
within each part of the knob's range. Read-outs are fit on 200,000 training-environment episodes; settings and lambda are
chosen on 20,000 novel-environment episodes, the accuracy is that on another 20,000.
  panels 1-3  accuracy against spectral radius, leak rate, connection density (density: Gaussian reservoirs only)
  panel 4     accuracy against the read-out's regularization strength for the best dense and the best channel-local
              setting of sweep2 on summed [x; x^2; u]: logistic with soft targets (lambda per weight, standardized
              features) and ridge (lambda / mean eigenvalue of X'X)
  panel 5     accuracy against reservoir size N: dense = the first sweep's best three settings x 2 seeds (line = mean);
              channel-local = sweep2's best two settings with the summed [x; x^2; u] soft logistic read-out (N <= 8,000);
              trained read-in at the sizes that have finished

11-10 05:30 (the script prints these): dense 357 settings .10-.24 (median .19); channel-local 57 settings .16-.46 (median
.27; best = setting 80: rho .08, alpha .30, p .007); with size, dense .21 (N 500) -> .26 (20,000), channel-local setting 80
.458 / .465 / .478 / .483 at N 1,000 / 2,000 / 4,000 / 8,000 and setting 116 .452 / .478 / .488 at 1,000 / 2,000 / 4,000;
trained read-in rho .9: .399, rho 0: .422 (2 of 11 runs finished). Naive .45, Joint .77, trained RNN .76, paper's ESN .43.
So a tuned fixed reservoir passes Naive Bayes by at most .04 and only with the channel-local read-in and squared states;
the trained RNN is .27 above that.
RESULT 11-13 21:40 IST (trained read-in, larger N; esn_trained_readin.py tasks 11-13 = N 4,000 / 8,000 with TF32 matmuls on
an A100, + the TF32 control at N 1,000): the control matches fp32 (exact read-out .400 vs .399, own .392 vs .390), so it
is printed, not drawn. Exact read-out vs N: .377 / .399 / .418 / .439 at N 500 / 1,000 / 2,000 / 4,000 (own read-outs
.365 / .390 / .396 / .421), about +.02 per doubling; Naive Bayes .451. N 8,000 (11-14 00:07 IST): exact .455 (val .452,
lambda 1e-4; own .422) = Naive Bayes' level (+.004, about one standard error of a 20,000-episode test); its CPU read-out
fit alone took ~4 h. Channel-local N 500 (scale2 rank 0, seed 1, local CPU): .433, so both lines span N 500-8,000.
RESULT 11-14 03:06 IST (John 11-13: trained networks in panel 4 and every series spanning the channel-local range;
esn_trained_readin.py tasks 14-17, local GPUs, fp32, which now store their read-out paths and nets): rho 1.5 .392,
alpha .03 .406, p .006 .400 (exact read-outs; own .373 / .399 / .392). The rho 1.2 rerun (best N 1,000 setting by
validation) reproduces the cluster run: exact .420 vs .423, own .396 vs .404 (old file kept as old_cluster_rho1.2_...json);
panel 4 draws its paths: logistic (soft targets) .420 / .421 / .418 / .393 / .351 at lambda 1e-6 ... 1e-2; ridge on its
summed states .353 at the smallest lambda, best .357 (lambda 1.8e-3 x the mean eigenvalue), .241 at the largest. Every knob leaves the trained read-in at .39-.42 at N 1,000; only N moves it.

11-14 03:40 IST REDESIGN (John: "Can we do all y-axes ... gap to joint bayes?"; "If it's all about the readout, let's
remove the channel local readin points and just put the data we have for the trained readin"): only the trained read-in
is drawn; y = Joint Bayes' goal accuracy minus the network's, on the same 20,000 novel-environment test episodes (Joint
.772 there; Naive .453), log scale, inverted (up = closer). Why: the channel-local points' read-out used summed
[x; x^2; u] (u = counts and T x Z); with summed [x; u] the channel-local scored .365 / .377 / .391 / .398 / .403 at N 500-
8,000, below the trained read-in's summed-x read-out (.377 / .399 / .418 / .439 / .455) at every N. Size: gap .392 / .369
/ .351 / .333 / .317 at N 500-8,000 = a power law, gap ~ N^-0.076 (halving it would take ~10,000x more units if the trend
held; against the dataset's Joint .767 instead the exponent is -0.081: the extrapolation is fragile, the fit is not).
The pre-redesign plotter is in the session scratchpad (esn_convex_plot_before_gap.py).
11-14 03:55 IST FINAL (John: "no local readin or fancy readout and just wrap it up"). For the reply's text: the best FIXED
reservoir found (channel-local random read-in, 57 random settings, summed [x; x^2; u] read-out) does pass Naive slightly:
.458 at N 1,000, .483 at N 8,000 (Naive .451): +.03 at most, .28 below Joint. Not drawn, but "no reservoir beats Naive"
would contradict it. Only 4 of its 57 settings beat the best trained read-in (.427 with the same read-out).
John 11-14 03:58 IST: "squared readout is already too complex" -> the reply keeps to LINEAR read-outs of the reservoir
state (summed states, + inputs). Then no reservoir exceeds Naive: trained read-in .455 at N 8,000 (= Naive .451 within
~1 SE), channel-local fixed read-in at most .403 (N 8,000, summed [x; u]).
John 11-14 ~13:30 ET: linear y up to gap 0 (= Joint Bayes), notes moved to the caption, panel letters. CAPTION (draft):
Echo-state networks with a trained read-in stay far from Joint Bayes. Gap = Joint Bayes' goal accuracy at T = 30 minus the
network's, on 20,000 novel-environment episodes (0 = Joint Bayes, top). One network per point, one setting varied at a time
from rho = 0.9, gamma = 1, p = 0.1, N = 1,000 units (11-15, John: the leak rate is gamma in figure and text; alpha is the
manuscript's loss exponent). (a-c) Spectral radius, leak rate and connection density of the frozen
leaky-tanh reservoir. (d) Read-out regularization at the best setting (rho = 1.2): logistic and ridge read-outs (ridge lambda
relative to the mean eigenvalue). (e) Reservoir size; dashed: power-law fit, gap ~ N^-0.08. Read-outs fit on 200,000
training-environment episodes, lambda chosen on another 20,000 novel-environment episodes. Dotted: Naive Bayes and a linear
rule on the observation counts (cross-entropy fit).

    python esn_convex_plot.py [--pull] [--pdf]
"""
import os, sys, json, glob, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

EXP = os.path.join(ROOT, "main", "DATA", "experimental")
DATA = os.path.join(EXP, "esn_convex")
TRAINED = os.path.join(EXP, "esn_trained_readin")
sys.path.insert(0, os.path.join(ROOT, "main", "REPLICATION", "trainings"))
from esn_convex import LAMS, LOG_LAMS

DENSE, LOCAL, TRAIN, RNN, PAPER = "#1f3b73", "#e66101", "#1b9e77", "#2a6fdb", "#7a7a7a"
NEUTRAL = "0.4"                                                  # panels (b) and (c) of the 3-panel figure (John 11-15)
DEFAULT = dict(rho=.9, alpha=1., density=.1, N=1000)             # the trained-read-in runs vary one knob from here
SQ = "sum_sq_soft"                                               # summed [x; x^2; u], logistic on Joint's posterior


def pull():
    os.makedirs(TRAINED, exist_ok=True)
    for pat in ("sweep*_N1000_p*.jsonl", "scale*.json"):
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/esn_convex/{pat}' {DATA}/", shell=True)
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/esn_trained_readin/{{rho,lstm}}*.json' {TRAINED}/", shell=True)
    # 11-13: tasks 14-17 ran on the local GPUs (the cluster's rho 1.2 result is kept as old_cluster_rho1.2_...json)
    subprocess.run(f"cp -p {ROOT}/runs/esn_trained_readin/rho*_N1000_s0.json {TRAINED}/ 2>/dev/null", shell=True)


# John 11-10 16:00 ("maybe esn_convex is where the best linear correction should go?"), then 16:20: "the best linear
# readout of the counts needs to be fit over all environments ... Maybe it can have Z as a regressor, but it shouldnt be
# separate per environment, that's cheating": trainings/esn_shared_linear.py, ONE logistic read-out for all environments
# (soft Joint targets, lambda on val, same splits as the ESNs): counts + Z .134; + count x Z products .188; + count x
# Fourier(Z) products .240 (val-chosen, drawn); control: the same fit on Naive's own log-likelihoods .440 (Naive .451).
# Lower bounds on the best shared count rule (richer Z codes could go higher), never an upper bound. John 11-10 16:50:
# draw the per-environment CE fit instead ("best linear function of counts", .505 = Naive + .0554, paired edge exact):
# weights any function of Z (fit per environment from the exact tables), logits linear in the counts; Naive Bayes is one
# member, so it is never below Naive. The shared fits stay in the caption only.
LIN_EDGE = .0554
SHARED_LIN = os.path.join(ROOT, "main", "DATA", "experimental", "esn_convex", "shared_linear", "result.json")


def best(rec):
    """test accuracy of the read-out with the highest validation accuracy"""
    return max((v for v in rec.values() if isinstance(v, dict) and "val" in v), key=lambda v: v["val"])["test"]


def best_val(rec):
    """the validation accuracy that best() picks by"""
    return max(v["val"] for v in rec.values() if isinstance(v, dict) and "val" in v)


def json_files(d, pat):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, pat)))]


def load():
    lines = lambda pat: [json.loads(l) for f in sorted(glob.glob(os.path.join(DATA, pat))) for l in open(f) if l.strip()]
    files = lambda d, pat: [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, pat)))]
    sweep2 = lines("sweep2_N1000_p*.jsonl")
    dense = [dict(r, cfg=dict(r["cfg"], readin="dense", topology="gaussian")) for r in lines("sweep_N1000_p*.jsonl")]
    dense += [r for r in sweep2 if r["cfg"]["readin"] == "dense"]
    local = [r for r in sweep2 if r["cfg"]["readin"] == "local"]
    return (dense, local, files(TRAINED, "rho*.json"), files(DATA, "scale_c*.json"), files(DATA, "scale2_r*.json"),
            json.load(open(os.path.join(DATA, "paper.json"))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--pdf", action="store_true", help="also save a PDF (TrueType fonts)")
    ap.add_argument("--linear", action="store_true",     # John 11-14: "without any log scaling (or the linear counts line)"
                    help="linear x axes -> esn_convex_linear.png")
    ap.add_argument("--five", action="store_true", help="the earlier 5-panel layout -> esn_convex_5panel.png")
    args = ap.parse_args()
    if args.pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    dense, local, trained, scale, scale2, paper = load()
    control = [r for r in trained if r.get("tf32") and r["N"] == 1000]        # the TF32 precision control: not drawn
    trained = [r for r in trained if r not in control]                        # (it would double every knob's default point)
    for r in control:
        print(f"TF32 control {r['name']}: own {r['final']['test']:.3f}, exact {r['exact_soft']['test']:.3f}")
    naive, joint = paper["bayes"]["test"]["naive"], paper["bayes"]["test"]["joint"]
    rnn, esn = (paper[n]["own"]["test"] for n in ("fully_trained_ctx_2_e5", "reservoir_ctx_2_e5"))
    lin = naive + LIN_EDGE

    # John 11-14 ("can we do all y-axes ... gap to joint bayes?" and "If it's all about the readout, let's remove the channel
    # local readin points and just put the data we have for the trained readin"): the trained read-in only; y = Joint Bayes'
    # accuracy minus the network's on the same 20,000 novel-environment test episodes (each run's own Bayes), log scale,
    # inverted so that up = closer to Joint. (The channel-local points' read-out had squared states and the inputs, which
    # the trained runs' did not: esn_trained_sqreadout.py.)
    gap = lambda r, acc: r["bayes"]["joint_test"] - acc
    b0 = next(r for r in trained if all(r[k] == v for k, v in DEFAULT.items()))["bayes"]
    g_naive = b0["joint_test"] - b0["naive_test"]
    g_lin = b0["joint_test"] - (b0["naive_test"] + LIN_EDGE)
    fig, ax = plt.subplots(1, 5, figsize=(5 * 2.9 + 1.7, 2.0 + 2.3), sharey=True, constrained_layout=True)

    def one_knob(a, k, marker="D", color=TRAIN, ms=7, y=gap):
        """the trained-read-in runs that differ from DEFAULT in this knob only"""
        runs = sorted((r for r in trained if all(r[j] == v for j, v in DEFAULT.items() if j != k)), key=lambda r: r[k])
        a.plot([r[k] for r in runs], [y(r, best(r)) for r in runs], "-", marker=marker, color=color, ms=ms, lw=1.4,
               zorder=6, markeredgecolor="k", markeredgewidth=.7)  # John 11-15: black outlines, bigger markers
        return runs

    if not args.five:
        # John 11-14: "can we put panels 1-3 together with a per-param x-axis? and reduce the height of the figure": (a) rho,
        # alpha and p in one panel, each scaled to the panel width and read on its own x-axis in its colour; (b) lambda,
        # (c) N; no linear-rule line. Later: no lines between (a)'s points; the alpha / p axes inside the panel's empty lower
        # part (one row of tick labels, so (b) and (c) keep their height); then "back to log scaling x axis and squeeze the 3
        # axes closer" (rho stays linear: it includes 0); no fit line in (c) (its dashed black read as (b)'s ridge read-out)
        logx = not args.linear
        # John 11-15 ("the y axis is still gap to bayes, can we make it raw accuracy?"): y = the network's goal accuracy at T
        # on its own 20,000 test episodes; Joint / Naive Bayes lines from the default network's test episodes (the runs'
        # three test draws differ by <= .006 in Joint accuracy). The printed N slope stays the gap's.
        raw = lambda r, acc: acc
        plt.close(fig)
        fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True, layout="constrained")
        fig.get_layout_engine().set(rect=(0, .08, 1, .92))           # room for the panel letters below
        knobs = (("rho", "spectral radius", r"$\rho$", "#1baf7a", "D", False),
                 ("alpha", "leak rate", r"$\gamma$", "#2a78d6", "o", True),
                 ("density", "connection density", r"$p$", "#eb6834", "s", True))
        a = ax[0]
        for i, (k, name, sym, col, mk, log) in enumerate(knobs):
            runs = sorted((r for r in trained if all(r[j] == v for j, v in DEFAULT.items() if j != k)), key=lambda r: r[k])
            vals = np.array([r[k] for r in runs], float)
            if log and logx:                    # the panel's position u in [0, 1] <-> the knob's value, log-spaced
                lo, hi = np.log10(vals.min()), np.log10(vals.max())
                fwd = lambda u, lo=lo, hi=hi: 10 ** (lo + np.asarray(u, float) * (hi - lo))
                inv = lambda v, lo=lo, hi=hi: (np.log10(np.clip(np.asarray(v, float), 1e-12, None)) - lo) / (hi - lo)
            else:
                top = vals.max()
                fwd = lambda u, t=top: np.asarray(u, float) * t
                inv = lambda v, t=top: np.asarray(v, float) / t
            # John 11-15: "make the marker outlines black and increase the size of the circle and diamond": per-marker
            # sizes (at equal size a diamond has half a square's area, a circle .79 of it)
            a.plot(inv(vals), [raw(r, best(r)) for r in runs], ls="none", marker=mk, color=col,
                   ms={"D": 10.5, "o": 8, "s": 5}[mk], markeredgecolor="k", markeredgewidth=.7,
                   zorder={"D": 6, "o": 7, "s": 8}[mk],         # John 11-15: "the markers in front smaller than the ones
                   label=f"{name} {sym}")                 # behind": diamond at the back, circle, square in front
            loc = .12 * i
            sec = a.secondary_xaxis(loc, functions=(fwd, inv))
            if log and logx:
                sec.xaxis.set_major_locator(matplotlib.ticker.LogLocator())
                sec.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=np.arange(2, 10)))
                sec.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                sec.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
            elif k == "rho":
                sec.set_xticks([0, .5, 1, 1.5])
            sec.tick_params(axis="x", which="both", colors=col, labelsize=fs["tick"])
            sec.spines["bottom"].set_color(col)
            a.text(1.015, loc, sym, transform=a.transAxes, ha="left", va="center", color=col, fontsize=fs["label"])
        a.tick_params(axis="x", bottom=False, labelbottom=False)
        a.spines["bottom"].set_visible(False)
        a.set_xlim(-.04, 1.04)
        a.set_title("Reservoir parameters", fontsize=fs["title"] - 3)
        # John 11-15: the legend above the Joint Bayes line, so its markers cannot be read as data points
        a.legend(loc="upper center", bbox_to_anchor=(.5, 1.02), frameon=False, fontsize=fs["legend"] - 1, ncol=1,
                 handlelength=1.4, handletextpad=.7, labelspacing=.3, borderaxespad=0.)

        a = ax[1]                               # the best N 1,000 run by validation among those that store their paths
        r = max((r for r in trained if "paths" in r and r["N"] == 1000), key=best_val)
        # John 11-15 ("should we have different markers for panels 2 and 3?"): (b) and (c) get their own markers in a
        # neutral grey, so they do not read as (a)'s green spectral-radius diamonds
        a.plot(LOG_LAMS, r["paths"]["logistic_soft"]["test"], "-p", ms=9, color=NEUTRAL, lw=1.5,   # 11-15: (b) / (c) markers swapped
               zorder=5, markeredgecolor="k", markeredgewidth=.7)
        a.plot(LAMS, r["paths"]["ridge"]["test"], "--", color=NEUTRAL, lw=1.3, zorder=5)
        if logx:
            a.set_xscale("log")
            a.set_xticks([1e-6, 1e-4, 1e-2, 1])
        a.set_title(r"Read-out regularization $\lambda$", fontsize=fs["title"] - 3)
        a.legend(handles=[plt.Line2D([], [], color=NEUTRAL, marker="p", ms=9, lw=1.5, markeredgecolor="k",
                                     markeredgewidth=.7, label="logistic read-out"),
                          plt.Line2D([], [], color=NEUTRAL, ls="--", lw=1.3, label="ridge read-out")],
                 fontsize=fs["legend"] - 1, frameon=False, loc="upper right", handlelength=1.8)

        a = ax[2]
        runs = one_knob(a, "N", marker="^", color=NEUTRAL, ms=8, y=raw)
        Nn, G = np.array([r["N"] for r in runs]), np.array([gap(r, best(r)) for r in runs])
        slope = np.polyfit(np.log(Nn), np.log(G), 1)[0]
        print(f"size: gap {np.round(G, 3)} at N {Nn}: power-law slope {slope:.3f} (caption; the fit line is not drawn)")
        if logx:
            a.set_xscale("log")
            a.set_xlim(400, 1.2e4)
        else:
            a.set_xlim(0, 8800)
        a.set_title(r"Reservoir size $N$", fontsize=fs["title"] - 3)

        y_max = 1.                                           # John 11-15: "make the y-axis 0 to 1"
        for a in ax:
            a.axhline(b0["joint_test"], color="k", lw=1.0, zorder=2)
            a.axhline(b0["naive_test"], color="k", lw=1.0, ls=":", zorder=2)
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
        ax[0].set_ylim(0, y_max)                             # raw accuracy, up = better
        ax[0].set_yticks(np.arange(0, y_max + 1e-9, .2))
        ax[0].set_ylabel("Accuracy")                         # John 11-15
        ax[1].set_xlabel(r"$\lambda$", fontsize=fs["label"])   # John 11-15: x labels on (b) and (c) too, symbols as (a)'s
        ax[2].set_xlabel(r"$N$", fontsize=fs["label"])
        t = ax[2].get_yaxis_transform()
        for y, lab in ((b0["joint_test"], "Joint Bayes"), (b0["naive_test"], "Naive Bayes")):
            ax[2].text(1.02, y, lab, transform=t, ha="left", va="center", fontsize=fs["legend"] - 1)
        fig.canvas.draw()                                    # the layout, then the paper's letters on one line below
        for a, letter in zip(ax, "abc"):
            p = a.get_position()
            fig.text((p.x0 + p.x1) / 2, .035, f"({letter})", ha="center", va="center", family="sans-serif",
                     fontsize=fs["label"])
        out = fig_path("esn_convex_linear.png" if args.linear else "esn_convex.png")
        fig.savefig(out, dpi=200)
        print(out)
        if args.pdf:
            with plt.rc_context({"pdf.fonttype": 42}):
                fig.savefig(out[:-4] + ".pdf")
            print(out[:-4] + ".pdf")
        return

    # John 11-14: "take the gap all the way up to 0", "simplify the text, remove the note if this can be in a caption":
    # linear y from 0 (= Joint Bayes) down; the run settings, sets and the size law moved to the caption
    for a, (k, title, sym, log) in zip(ax, (("rho", "Spectral radius", r"$\rho$", False), ("alpha", "Leak rate", r"$\gamma$", True),
                                            ("density", "Connection density", r"$p$", True))):
        one_knob(a, k)
        if log and not args.linear:
            a.set_xscale("log")
        a.set_title(f"{title} {sym}", fontsize=fs["title"] - 3)

    a = ax[3]                                   # the best N 1,000 run by validation among those that store their paths
    r = max((r for r in trained if "paths" in r and r["N"] == 1000), key=best_val)
    a.plot(LOG_LAMS, [gap(r, v) for v in r["paths"]["logistic_soft"]["test"]], "-D", ms=4, color=TRAIN, lw=1.5, zorder=5,
           markeredgecolor="w", markeredgewidth=.5)
    a.plot(LAMS, [gap(r, v) for v in r["paths"]["ridge"]["test"]], "--", color=TRAIN, lw=1.3, zorder=5)
    print(f"panel 4: {r['name']}")
    if not args.linear:
        a.set_xscale("log")
        a.set_xticks([1e-6, 1e-4, 1e-2, 1])
    a.set_title(r"Read-out regularization $\lambda$", fontsize=fs["title"] - 3)
    a.legend(handles=[plt.Line2D([], [], color="k", marker="D", ms=4, lw=1.5, label="logistic read-out"),
                      plt.Line2D([], [], color="k", ls="--", lw=1.3, label="ridge read-out")],
             fontsize=fs["legend"] - 1, frameon=False, loc="lower left", handlelength=1.8)

    a = ax[4]
    runs = one_knob(a, "N")
    Nn, G = np.array([r["N"] for r in runs]), np.array([gap(r, best(r)) for r in runs])
    slope, icpt = np.polyfit(np.log(Nn), np.log(G), 1)
    xx = np.geomspace(Nn.min(), Nn.max(), 50)
    a.plot(xx, np.exp(icpt) * xx ** slope, color="k", lw=1.0, ls=(0, (4, 2)), zorder=7)
    print(f"size: gap {np.round(G, 3)} at N {Nn}: slope {slope:.3f}, halving the gap needs x{2 ** (-1 / slope):.0f} units")
    if args.linear:
        a.set_xlim(0, 8800)
    else:
        a.set_xscale("log")
        a.set_xlim(400, 1.2e4)
    a.set_title(r"Reservoir size $N$", fontsize=fs["title"] - 3)

    y_max = np.ceil(10 * max(max(np.max(l.get_ydata()) for a in ax for l in a.get_lines()), g_naive)) / 10
    for a, letter in zip(ax, "abcde"):
        a.axhline(0, color="k", lw=1.0, zorder=2)
        a.axhline(g_naive, color="k", lw=1.0, ls=":", zorder=2)
        if not args.linear:
            a.axhline(g_lin, color="0.5", lw=1.0, ls=(0, (1, 1.5)), zorder=2)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.set_xlabel(rf"$\mathsf{{({letter})}}$", fontsize=fs["label"])          # the paper's panel letters, sans-serif
    ax[0].set_ylim(y_max, 0)                                 # linear, inverted: 0 = Joint Bayes on top
    ax[0].set_yticks(np.arange(0, y_max + 1e-9, .1))
    t = ax[4].get_yaxis_transform()
    for y, lab, col in ((0, "Joint Bayes", "k"), (g_naive, "Naive Bayes", "k"), (g_lin, "linear rule on counts", "0.4"))[
            :2 if args.linear else 3]:
        ax[4].text(1.02, y, lab, transform=t, color=col, ha="left", va="center", fontsize=fs["legend"] - 1)
    ax[0].set_ylabel("gap to Joint Bayes")
    out = fig_path("esn_convex_5panel" + ("_linear" if args.linear else "") + ".png")
    fig.savefig(out, dpi=200)
    print(out)
    if args.pdf:                                                  # John 11-14: "send ... esn_convex as pdfs"
        with plt.rc_context({"pdf.fonttype": 42}):
            fig.savefig(out[:-4] + ".pdf")
        print(out[:-4] + ".pdf")

    acc = lambda recs: np.array([best(r) for r in recs])
    for name, recs in (("dense", dense), ("channel-local", local)):
        v = acc(recs)
        print(f"{name}: {len(recs)} settings, {v.min():.3f}-{v.max():.3f}, median {np.median(v):.3f}")
    print("dense by N:", {n: round(float(np.mean([best(r) for r in scale if r["N"] == n])), 3) for n in sorted({r["N"] for r in scale})})
    print("channel-local by N:", {(r["cfg_id"], r["N"]): round(r[SQ]["test"], 3) for r in local + scale2
                                  if SQ in r and r["cfg_id"] in {s["cfg_id"] for s in scale2}})
    print("trained read-in:", {r["name"]: round(best(r), 3) for r in trained})
    print(f"Naive {naive:.3f}  Joint {joint:.3f}  trained RNN {rnn:.3f}  paper's ESN {esn:.3f}")


if __name__ == "__main__":
    main()
