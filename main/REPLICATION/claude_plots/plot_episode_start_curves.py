"""Does a fresh random start state per episode help? Accuracy through training, per-episode random start against the
same network's other start states, in four settings (2026-10-27 learnability batch). Four panels.

    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/plot_episode_start_curves.py

CPU only, a few seconds. Writes figure_bin/episode_start_curves.png.

Start states (colour = start, the same in every panel):
  shared start      R3 / R1's trained start state (still essentially its N(0, 1) draw), or a fixed N(0, 0.25^2) feature
                    added at step 1 in the no-reservoir network: the same vector in every episode
  start 0           start state 0 (no start feature in the no-reservoir network)
  per-episode start a fresh N(0, 1) start state for every episode (N(0, 0.25^2) feature in the no-reservoir network),
                    never trained, test episodes included
Panels:
  (a) joint-output ESN trained from scratch (frozen LSTM, trained read-in / read-out, the networks' own goal loss, batch
      8,000, Adam 1e-3, fp16). Per-episode start: run_fullesn A (4,839 updates, a first look). Start 0: joint_scratch_zero
      (same seed, initial weights and test episodes as A). Shared start: the original joint-output ESN training.
      Tests: 8,000 test-state episodes of the network's environment.
  (b) marginal-output ESN, the same: per-episode start run_fullesn B (4,708 updates); shared start the original ESN.
  (c) joint read-out alone, retrained from zero on the trained joint ESN's frozen read-in + reservoir (R3), batch 8,000,
      own loss, Adam 1e-3, TF32; one pool of 60,000 episodes, held-out episodes 54,000-59,999. Per-episode start: a
      fresh draw at every presentation (B2f).
  (d) no reservoir: one ReLU layer (the ESN read-in) + linear joint read-out per step, logits summed over steps, own
      loss, batch 2,000 fresh environments, Adam 1e-3; held-out 8,000 episodes. Seed 0 solid, seed 1 thin.
Dashed lines: optimal (joint) Bayes and naive Bayes on each panel's test episodes.
Data: main/DATA/experimental/esn_learnability/ (copied from the session scratchpad; README.txt there) and the original
networks in main/DATA/sanity and main/DATA/experimental/interaction_bounds/analyses/joint_esn_extension.
"""
import os, sys, json, inspect
import numpy as np


def _repo_root():
    """the folder holding main/CognitiveGridworld.py, above this file; __file__ can be missing or a placeholder that
    does not exist in an interactive window, so the working directory is tried next"""
    here = None
    try:
        here = inspect.getfile(inspect.currentframe())
    except (TypeError, OSError):
        pass
    for start in (globals().get("__file__"), here, globals().get("__vsc_ipynb_file__"), os.getcwd()):
        d = os.path.abspath(start) if start and os.path.exists(start) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return d
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError("project root (the folder holding main/CognitiveGridworld.py) not found")


ROOT = _repo_root()
sys.path.insert(0, ROOT)
import matplotlib.pyplot as plt                                                  # noqa: E402
from main.utils import fig_path, apply_fig_style                                 # noqa: E402

SANITY = os.path.join(ROOT, "main", "DATA", "sanity")
EXT = os.path.join(ROOT, "main", "DATA", "experimental", "interaction_bounds", "analyses", "joint_esn_extension")
EL = os.path.join(ROOT, "main", "DATA", "experimental", "esn_learnability")
C_SH, C_ZERO, C_EP = "#7b3294", "#1b9e77", "#e66101"          # shared start, start 0, per-episode start
C_OPT, C_NAIVE = "k", "0.5"


def original(stem):
    """(updates before each test, final-step test accuracy) of an original network's training"""
    import torch
    d = torch.load(os.path.join(SANITY, stem + "_net.pth"), map_location="cpu", weights_only=False)
    acc = np.asarray(d["test_acc_through_training"], np.float64)[:, -1]
    return np.arange(len(acc)) * 499, acc                          # a test replaces an update: test k after 499 k


def jsonl(path):
    with open(path) as f:
        L = [json.loads(line) for line in f if line.strip()]
    get = lambda k: np.array([r[k] for r in L], np.float64)
    return get("update"), get("acc_T"), get("opt_acc_T"), get("naive_acc_T")


def readout(name):
    m = json.load(open(os.path.join(EL, "readout_only", name, "metrics.json")))
    return (np.array([r["update"] for r in m["metrics"]], np.float64),
            np.array([r["acc_T30"] for r in m["metrics"]], np.float64))


def noreservoir(folder, tag):
    d = json.load(open(os.path.join(EL, "no_reservoir", folder, tag + ".json")))
    return (np.array([c["update"] for c in d["curve"]], np.float64),
            np.array([c["acc_t30"] for c in d["curve"]], np.float64), d["heldout_refs"])


def main():
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    W, H = 7.4, 5.95
    L0, CW, GAP, B0, RH, VG = .62, 2.9, .78, .5, 2.0, .95
    axs = {"a": [L0, B0 + RH + VG], "b": [L0 + CW + GAP, B0 + RH + VG], "c": [L0, B0], "d": [L0 + CW + GAP, B0]}
    fig = plt.figure(figsize=(W, H))
    ax = {k: fig.add_axes([x / W, y / H, CW / W, RH / H]) for k, (x, y) in axs.items()}
    fl, lw = fs["tick"] - 1, 1.3
    acc_math = r"accuracy $\Pr(\hat{r}_g = r_g)$ at $t = 30$"

    def bounds(a, opt, naive):
        a.axhline(opt, color=C_OPT, ls=(0, (4, 2)), lw=.9, zorder=1)
        a.axhline(naive, color=C_NAIVE, ls=(0, (4, 2)), lw=.9, zorder=1)
        x0 = .02 * a.get_xlim()[1]                                  # labels at the left, clear of the curves
        a.text(x0, opt + .012, f"optimal Bayes {opt:.2f}", ha="left", va="bottom", fontsize=fl - .5, color="0.25")
        a.text(x0, naive + .012, f"naive Bayes {naive:.2f}", ha="left", va="bottom", fontsize=fl - .5, color="0.45")

    def finish(a, title, xmax, xlabel):
        a.set_xlim(0, xmax)
        a.set_xticks(np.arange(0, xmax + .1, 2 if xmax <= 10 else 5))
        a.set_ylim(0, .82)
        a.set_title(title + "\n" + acc_math, fontsize=fs["label"], loc="left")
        a.set_xlabel(xlabel, fontsize=fl)
        a.set_ylabel("Accuracy", fontsize=fl + .5)
        a.spines[["top", "right"]].set_visible(False)

    # (a) joint ESN from scratch
    a = ax["a"]
    uA, yA, oA, nA = jsonl(os.path.join(EL, "full_esn_episode_start", "A_joint_episode_start_log.jsonl"))
    u4, y4, o4, n4 = jsonl(os.path.join(EXT, "runs", "joint_scratch_zero", "log.jsonl"))
    x3, y3 = original("joint_reservoir_ctx_2")
    k3, k4 = x3 <= 10000, u4 <= 10000
    a.plot(x3[k3] / 1e3, y3[k3], color=C_SH, lw=lw, label=f"shared start (trained): {np.interp(4500, x3, y3):.3f}")
    a.plot(u4[k4] / 1e3, y4[k4], color=C_ZERO, lw=lw, label=f"start 0: {y4[u4 == 4500][0]:.3f}")
    a.plot(uA / 1e3, yA, color=C_EP, lw=lw + .3, label=f"per-episode start: {yA[uA == 4500][0]:.3f}")
    a.plot(uA[-1] / 1e3, yA[-1], ">", ms=3, color=C_EP)
    finish(a, "(a) Joint-output ESN, trained from scratch", 10, "Training updates (thousands; 8,000 episodes)")
    bounds(a, float(np.mean(np.r_[oA, o4])), float(np.mean(np.r_[nA, n4])))
    a.legend(loc="lower right", frameon=False, fontsize=fl, title="values at 4,500 updates", title_fontsize=fl,
             bbox_to_anchor=(1.0, .08), handlelength=1.6, alignment="left")

    # (b) marginal ESN from scratch
    a = ax["b"]
    uB, yB, oB, nB = jsonl(os.path.join(EL, "full_esn_episode_start", "B_marginal_episode_start_log.jsonl"))
    x1, y1 = original("reservoir_ctx_2")
    k1 = x1 <= 10000
    a.plot(x1[k1] / 1e3, y1[k1], color=C_SH, lw=lw, label=f"shared start (trained): {np.interp(4500, x1, y1):.3f}")
    a.plot(uB / 1e3, yB, color=C_EP, lw=lw + .3, label=f"per-episode start: {yB[uB == 4500][0]:.3f}")
    a.plot(uB[-1] / 1e3, yB[-1], ">", ms=3, color=C_EP)
    finish(a, "(b) Marginal-output ESN, trained from scratch", 10, "Training updates (thousands; 8,000 episodes)")
    bounds(a, float(np.mean(oB)), float(np.mean(nB)))
    a.legend(loc="lower right", frameon=False, fontsize=fl, title="values at 4,500 updates", title_fontsize=fl,
             bbox_to_anchor=(1.0, .08), handlelength=1.6, alignment="left")

    # (c) joint read-out alone on R3's frozen front end
    a = ax["c"]
    for name, c, lab in (("A1_r3_own_adam", C_SH, "shared start (R3's)"), ("C4_zero_own_adam", C_ZERO, "start 0"),
                         ("B2f_epr_fresh_own_adam", C_EP, "per-episode start")):
        u, y = readout(name)
        a.plot(u / 1e3, y, color=c, lw=lw + (.3 if c == C_EP else 0), label=f"{lab}: {y[-1]:.3f}")
    ref = json.load(open(os.path.join(EL, "readout_only", "bayes_ref.json")))
    finish(a, "(c) Joint read-out alone, retrained on the\njoint ESN's frozen read-in + reservoir", 20,
           "Read-out updates (thousands; 8,000 episodes)")
    bounds(a, ref["bayes_acc_t"][-1], ref["naive_acc_t"][-1])
    a.legend(loc="lower right", frameon=False, fontsize=fl, title="values at 20,000 updates", title_fontsize=fl,
             bbox_to_anchor=(1.0, .08), handlelength=1.6, alignment="left")

    # (d) no reservoir
    a = ax["d"]
    hs = []
    for tag, c, lab in (("j2s", C_SH, "shared start feature"), ("j2", C_ZERO, "no start"),
                        ("j2r", C_EP, "per-episode start feature")):
        u0, y0, refs = noreservoir("e2e_d1_x", tag)
        u1, y1_, _ = noreservoir("x_s1", tag)
        h, = a.plot(u0 / 1e3, y0, color=c, lw=lw + (.3 if c == C_EP else 0),
                    label=f"{lab}: {y0[u0 == 16000][0]:.3f} / {y1_[u1 == 16000][0]:.3f}")
        a.plot(u1 / 1e3, y1_, color=c, lw=.7, alpha=.8)
        hs.append(h)
    finish(a, "(d) No reservoir: one ReLU layer per step,\njoint read-out, logits summed over steps", 20,
           "Training updates (thousands; 2,000 episodes)")
    bounds(a, refs["bayes_acc_t30"], refs["naive_acc_t30"])
    a.legend(handles=hs, loc="lower right", frameon=False, fontsize=fl,
             title="values at 16,000 updates, seed 0 / seed 1 (thin)", title_fontsize=fl, bbox_to_anchor=(1.0, .08),
             handlelength=1.6, alignment="left")

    out = fig_path("episode_start_curves.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
