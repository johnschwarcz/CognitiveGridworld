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

    python esn_convex_plot.py [--pull]
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
DEFAULT = dict(rho=.9, alpha=1., density=.1, N=1000)             # the trained-read-in runs vary one knob from here
SQ = "sum_sq_soft"                                               # summed [x; x^2; u], logistic on Joint's posterior


def pull():
    os.makedirs(TRAINED, exist_ok=True)
    for pat in ("sweep*_N1000_p*.jsonl", "scale*.json"):
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/esn_convex/{pat}' {DATA}/", shell=True)
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/esn_trained_readin/{{rho,lstm}}*.json' {TRAINED}/", shell=True)


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
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    dense, local, trained, scale, scale2, paper = load()
    naive, joint = paper["bayes"]["test"]["naive"], paper["bayes"]["test"]["joint"]
    rnn, esn = (paper[n]["own"]["test"] for n in ("fully_trained_ctx_2_e5", "reservoir_ctx_2_e5"))
    lin = naive + LIN_EDGE

    fig, ax = plt.subplots(1, 5, figsize=(5 * 2.9 + 1.7, 2.0 + 2.3), sharey=True, constrained_layout=True)

    def one_knob(a, k):
        """the trained-read-in runs that differ from DEFAULT in this knob only"""
        runs = sorted((r for r in trained if all(r[j] == v for j, v in DEFAULT.items() if j != k)), key=lambda r: r[k])
        if runs:
            a.plot([r[k] for r in runs], [best(r) for r in runs], "-D", color=TRAIN, ms=5, lw=1.4, zorder=6,
                   markeredgecolor="w", markeredgewidth=.5)

    for a, (k, title, log) in zip(ax, (("rho", r"Spectral radius $\rho(W)$", False), ("alpha", r"Leak rate $\alpha$", True),
                                       ("density", r"Connection density $p$", True))):
        for recs, col, alpha in ((local, LOCAL, .8),):           # John 11-10: no dense (classical) read-in
            if k == "density":                                   # a cycle or an orthogonal reservoir has no density
                recs = [r for r in recs if r["cfg"]["topology"] == "gaussian"]
            x, y = np.array([r["cfg"][k] for r in recs]), np.array([best(r) for r in recs])
            a.scatter(x, y, s=8, color=col, alpha=alpha, lw=0, zorder=3)   # John 11-10 16:00: no best-per-range curve
        one_knob(a, k)
        if log:
            a.set_xscale("log")
        a.set_title(title, fontsize=fs["title"] - 3)

    a = ax[3]
    for recs, col in ((local, LOCAL),):
        r = max((r for r in recs if SQ in r), key=lambda r: max(r[SQ]["val"], r["sum_sq"]["val"]))
        a.plot(LOG_LAMS, r[SQ]["curves"]["test"], "-o", ms=3.5, color=col, lw=1.5, zorder=4)
        a.plot(LAMS, r["sum_sq"]["curves"]["test"], "--", color=col, lw=1.3, zorder=4)
    a.set_xscale("log")
    a.set_xticks([1e-6, 1e-4, 1e-2, 1])
    a.set_title(r"Read-out regularization $\lambda$" + "\n(the best channel-local setting)", fontsize=fs["title"] - 3)
    a.legend(handles=[plt.Line2D([], [], color="k", marker="o", ms=3.5, lw=1.5, label="logistic read-out"),
                      plt.Line2D([], [], color="k", ls="--", lw=1.3, label=r"ridge read-out ($\lambda$ / mean eigenvalue)")],
             fontsize=fs["legend"] - 1, frameon=False, loc="center", bbox_to_anchor=(.5, .7), handlelength=1.8)

    a = ax[4]
    Ns = sorted({r["N"] for r in scale})
    c = max({r["cfg_id"] for r in scale2},                       # one curve (John 11-10): the setting best at N 1,000
            key=lambda c: max(r[SQ]["val"] for r in local if r["cfg_id"] == c and SQ in r))
    rs = sorted((r for r in local + scale2 if r["cfg_id"] == c and SQ in r), key=lambda r: r["N"])
    a.plot([r["N"] for r in rs], [r[SQ]["test"] for r in rs], "-o", ms=3.5, color=LOCAL, lw=1.5, zorder=4)
    one_knob(a, "N")
    # John 11-11 01:10: "let's keep it simple, just classic ESNs": no paper-ESN triangle, no LSTM-core diamond
    a.set_xscale("log")
    a.set_xlim(400, 1.2e4)
    a.set_title(r"Reservoir size $N$ (units)" + "\n(channel-local: the best setting)", fontsize=fs["title"] - 3)

    for i, a in enumerate(ax):
        a.axhline(joint, color="k", lw=1.0, ls="--", zorder=2)
        a.axhline(naive, color="k", lw=1.0, ls=":", zorder=2)
        a.axhline(lin, color="0.5", lw=1.0, ls=(0, (1, 1.5)), zorder=2)
        a.set_ylim(0, .85)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    t = ax[4].get_yaxis_transform()
    for y, lab, col, va in ((joint, "Joint Bayes", "k", "bottom"), (naive, "Naive Bayes", "k", "top"),
                            (lin, "best linear\nfunction of counts", "0.4", "bottom")):
        ax[4].text(1.02, y, lab, transform=t, color=col, ha="left", va=va, fontsize=fs["legend"] - 1)
    ax[0].set_ylabel("accuracy, novel environments")

    dot = lambda col, lab: plt.Line2D([], [], marker="o", ls="", ms=5, color=col, label=lab)
    fig.legend(handles=[dot(LOCAL, f"channel-local random read-in: one dot = one of {len(local)} random settings "
                                   "(all knobs drawn together)"),
                        plt.Line2D([], [], marker="D", ms=5, lw=1.4, color=TRAIN, markeredgecolor="w", markeredgewidth=.5,
                                   label=r"trained read-in, frozen reservoir: one run per point, one knob varied from "
                                         r"$\rho$ .9, $\alpha$ 1, $p$ .1, $N$ 1,000")],
               loc="outside lower center", ncol=2, frameon=False, fontsize=fs["legend"], columnspacing=1.6,
               handletextpad=.4)
    fig.suptitle(r"Echo-state networks with exact read-outs: accuracy $\langle[\hat r_g = r_g]\rangle$ at $T = 30$, "
                 r"$\hat r_g = \mathrm{argmax}_r\,\hat y_g(r)$; $N = 1{,}000$ except in the last panel" + "\n"
                 "read-outs fit on 200,000 episodes of training environments; settings and $\\lambda$ chosen on 20,000 "
                 "episodes of novel environments, accuracy on another 20,000", fontsize=fs["legend"])
    out = fig_path("esn_convex.png")
    fig.savefig(out, dpi=200)
    print(out)

    acc = lambda recs: np.array([best(r) for r in recs])
    for name, recs in (("dense", dense), ("channel-local", local)):
        v = acc(recs)
        print(f"{name}: {len(recs)} settings, {v.min():.3f}-{v.max():.3f}, median {np.median(v):.3f}")
    print("dense by N:", {n: round(float(np.mean([best(r) for r in scale if r["N"] == n])), 3) for n in Ns})
    print("channel-local by N:", {(r["cfg_id"], r["N"]): round(r[SQ]["test"], 3) for r in local + scale2
                                  if SQ in r and r["cfg_id"] in {s["cfg_id"] for s in scale2}})
    print("trained read-in:", {r["name"]: round(best(r), 3) for r in trained})
    print(f"Naive {naive:.3f}  Joint {joint:.3f}  trained RNN {rnn:.3f}  paper's ESN {esn:.3f}")


if __name__ == "__main__":
    main()
