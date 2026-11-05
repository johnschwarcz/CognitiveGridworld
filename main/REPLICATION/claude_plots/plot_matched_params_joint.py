"""Two rows of three panels (John 11-04: "add the width 50 LSTMs to the left panel and move all of the ESNs to a 2nd row"): top
the trained LSTMs, bottom the ESNs, the columns the three conditions below, one y-axis for all six; the width-50 LSTM
(trained_N50, one seed; palest blue) on the top left, sampled one test per 500 episodes as its left neighbours, and
joint_trained_N50 / _jl (26k, launched 11-04) in the other two. Bottom row (John 11-04): the ESNs at N = 500 (echo_N500, palest red; joint_echo_N500 / _jl) and
N = 2k (joint_echo_N2000 / _jl, in the Echo N = 2k's colour), drawn once they have tests. Before that: a three-panel variant of plot_matched_params_new.py (John 2026-11-04: "a new figure ... dont overwrite the existing
one"): (left) its "Controlling for number of trainable parameters" panel, retitled "Marginal output, goal supervision"
(John 11-04, this figure only), its curves unchanged; (middle) joint output, goal
supervision; (right) joint output, joint supervision (John 11-04: "joint", the old figure's "full supervision"), each with the parameter-matched pair: the ESN at N = 1k (joint_echo /
joint_echo_jl, 118k trainable parameters) and the trained LSTM at N = 100 (joint_trained_N100 / joint_trained_N100_jl,
93k; train_training_dynamics.py, launched 11-04 08:07, trained_N100's configuration with the joint read-out, seed
20261029, the joint ESNs' seed), and at N = 200 (joint_trained_N200 / _jl, 345k, launched 11-04 11:34; solid blue as on
the left), and at N = 1k (John 11-04: trained, joint_trained, joint_trained_jl; navy, in all three panels, the left's too,
pooled over 5 tests per point as plot_matched_params_new.py's right panel draws these runs: at gaps of ~.01 a single test
has about the gap's own noise). Every line solid, each in its same-width counterpart's colour on the left (Echo N = 1k
light red, Trained N = 100 light blue); one y-axis for all three panels. Output and supervision go to the panel titles,
widths and trainable parameters to the legends. The right-hand curves are sampled as in plot_matched_params_new.py's
right panel (paper_style_pooled: the first 5 tests of each 500-training-episode block, 40,000 test episodes per point);
a run still training is drawn up to its last complete block, its legend entry saying how far it has got (computed from
the data). Trainable parameters as compare_matched_params counts them (a reservoir's LSTM excluded), from the run's
saved weights (final_net.pth, or its first checkpoint while it trains).
n1 (John 11-05): one run per curve, seed 20261029, all from the training-dynamics runs at batch 8,000 and 5 tests per point
(the batch-2,000 reps of Trained N = 200 and the five seeds of N = 100 replaced by single runs, and Echo N = 1k by echo), except the
marginal Echo N = 2k and 10k, the paper's runs (one test per point; 10k at batch 2,000), and the joint Echo N = 10k at batch 2,000.
-> main/REPLICATION/figure_bin/compare_matched_params_loglog_joint_panels.png, _n1.png, _no1k.png
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/plot_matched_params_joint.py"""
import os
import re
import sys
import glob
import json
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import torch                                                                      # noqa: E402
import matplotlib.pyplot as plt                                                   # noqa: E402
import compare_matched_params as cmp                                              # noqa: E402
import plot_matched_params_new as pm                                              # noqa: E402

PANELS = (("Joint output, goal supervision", ("joint_echo_N200", "joint_echo_N500", "joint_echo", "joint_echo_N2000", "joint_echo_N10000"),
           ("joint_trained_N50", "joint_trained_N100", "joint_trained_N150", "joint_trained_N200", "joint_trained_N300", "joint_trained_N400", "joint_trained_N500",
            "joint_trained")),
          ("Joint output, joint supervision", ("joint_echo_N200_jl", "joint_echo_N500_jl", "joint_echo_jl", "joint_echo_N2000_jl", "joint_echo_N10000_jl"),
           ("joint_trained_N50_jl", "joint_trained_N100_jl", "joint_trained_N150_jl", "joint_trained_N200_jl", "joint_trained_N300_jl", "joint_trained_N400_jl", "joint_trained_N500_jl",
            "joint_trained_jl")))
C_N300, C_N400, C_N500 = "#2368a3", "#1f5e9a", "#1a5490"   # between the N = 200's blue and the N = 1k's navy
C_N150 = "#5fa2cb"                                    # between the N = 100's light blue and the N = 200's blue
C_E200, C_E500 = "#fddcd3", "#fcc9bf"                 # paler reds: fewer trainable parameters than the Echo N = 1k
C_N50 = "#c6dbef"                                     # palest blue: the fewest trainable parameters
C_N1K = "#08306b"                                     # navy: darker than the N = 200's blue, more trainable parameters


def count(n):
    """a parameter count as the legends state it"""
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def params(run):
    """trainable parameters (a reservoir's LSTM excluded) from the final weights, or the first checkpoint's while training"""
    d = pm._dir(run)
    f = os.path.join(d, "final_net.pth")
    f = f if os.path.exists(f) else sorted(glob.glob(os.path.join(d, "ckpt_ep*_net.pth")))[0]
    w = torch.load(f, map_location="cpu", weights_only=False)["weights"]
    res = json.load(open(os.path.join(d, "run.json")))["cfg"].get("reservoir", False)
    return sum(v.numel() for k, v in w.items() if not (res and k.startswith("LSTM")))


def pooled(run, block=500, n_tests=pm.TEST_POOL):
    """pm.paper_style_pooled for a run that may still be training: the complete blocks only; and the training episodes
    reached"""
    d = pm._dir(run)
    every = json.load(open(os.path.join(d, "run.json")))["cfg"]["checkpoint_every"]
    z = [np.load(f) for f in sorted(glob.glob(os.path.join(d, "test_*.npz")))]
    ep = np.concatenate([q["episode"] for q in z]).astype(int)
    acc = np.concatenate([q["acc"][:, -1] for q in z]).astype(float)
    trained = ep - ep // every
    at = {t: i for i, t in enumerate(trained)}
    x, y = [], []
    for kk in range(1, pm.XMAX // block + 1):
        keys = [(kk - 1) * block + j * (every - 1) for j in range(n_tests)]
        if not all(k in at for k in keys):
            break
        x.append(kk * block)
        y.append(acc[[at[k] for k in keys]].mean())
    return np.array(x), np.array(y), int(trained.max())


def width_of(label):
    m = re.search(r"N\{=\}(\d+)(k?)", label)
    return int(m[1]) * (1000 if m[2] else 1)


def by_width(ax):
    """the wider curves over the narrower, under the naive bound; the legend by width, the largest at the bottom (John 11-05)"""
    lines = sorted((ln for ln in ax.get_lines() if not ln.get_label().startswith("_")), key=lambda ln: width_of(ln.get_label()))
    for ln in lines:
        ln.set_zorder(2 + np.log10(width_of(ln.get_label())))
    ax.legend(lines, [ln.get_label() for ln in lines], loc="lower left", frameon=False, fontsize=12)


def figure(step=cmp.STEP, path=None, width_1k=True, n1=False):
    from main.utils import fig_path

    def draw(ax, run, name, col, n_tests=pm.TEST_POOL):
        """one run's curve to its last complete block, if it has tests yet; the legend says how far a running one has got"""
        if not glob.glob(os.path.join(pm._dir(run), "test_*.npz")):
            print(f"  {run}: no test yet")
            return
        x, acc, reached = pooled(run, n_tests=n_tests)
        done = pm.finished(run)
        note = "" if done else f" (training: {min(reached, pm.XMAX) / 1e3:.1f}k of {pm.XMAX / 1e3:.0f}k)"
        ax.plot(x, pm._pos(J - acc), color=col, lw=2, ls="-", label=f"{name}, {count(params(run))} trainable params{note}")
        print(f"  {run:24s} {acc[-1] if len(acc) else np.nan:.3f} at training episode {x[-1] if len(x) else 0}"
              f"{'' if done else ' (still training)'}")

    cmp.apply_fig_style()
    b = dict(np.load(cmp.BOUNDS_NPZ))
    J = b[f"{cmp.BOUND_KEY}_joint"][step - 1]
    fig, axes = plt.subplots(2, 3, figsize=(30, 10.4), sharex=True, sharey=True)
    (t1, t2, t3), (e1, e2, e3) = axes                                             # rows: trained LSTMs, ESNs
    extra = [dict(pm.ESN2K)] if os.path.exists(os.path.join(cmp.DATA, pm.ESN2K["stem"] + "_net.pth")) else []
    nets = cmp.resolve_colors([dict(n) for n in cmp.NETWORKS if n["stem"] not in pm.DROP] + extra)
    if n1:                                                                        # one run per curve, seed 20261029, batch 8,000
        c1k = next(n["color"] for n in nets if n["stem"] == "sanity/reservoir_ctx_2")
        for run, name, col in (("echo_N200", r"Echo ($N{=}200$)", C_E200), ("echo_N500", r"Echo ($N{=}500$)", C_E500), ("echo", r"Echo ($N{=}1k$)", c1k)):
            draw(e1, run, name, col)
        for net in nets:                                                          # no training-dynamics run: the paper's
            if "reservoir" in net["stem"] and net["stem"] != "sanity/reservoir_ctx_2":
                pm.paper(e1, J, net, step)
        for run, name, col in (("trained_N50", r"Trained ($N{=}50$)", C_N50), ("trained_N100", r"Trained ($N{=}100$)", pm.C_N100),
                               ("trained_N150", r"Trained ($N{=}150$)", C_N150), ("trained_N200", r"Trained ($N{=}200$)", cmp.C_TRAIN),
                               ("trained_N300", r"Trained ($N{=}300$)", C_N300), ("trained_N400", r"Trained ($N{=}400$)", C_N400),
                               ("trained_N500", r"Trained ($N{=}500$)", C_N500), ("trained", r"Trained ($N{=}1k$)", C_N1K)):
            draw(t1, run, name, col)
    else:
        for net in nets:                                                              # the left panels, as plot_matched_params_new
            if "reservoir" in net["stem"]:
                pm.paper(e1, J, net, step)
        draw(e1, "echo_N200", r"Echo ($N{=}200$)", C_E200, n_tests=1)                 # one test per 500 episodes, as its neighbours
        draw(e1, "echo_N500", r"Echo ($N{=}500$)", C_E500, n_tests=1)
        x, acc = pm.paper_style_curve("trained_N50")                                  # one seed, one test per 500 episodes
        t1.plot(x, pm._pos(J - acc), color=C_N50, lw=2, label=rf"Trained ($N{{=}}50$), {count(params('trained_N50'))} trainable params")
        print(f"  trained_N50              {acc[-1]:.3f} at training episode {x[-1]}")
        pm.n100(t1, J)
        draw(t1, "trained_N150", r"Trained ($N{=}150$)", C_N150, n_tests=1)
        for net in nets:
            if "reservoir" not in net["stem"]:
                pm.paper(t1, J, net, step)
        draw(t1, "trained_N300", r"Trained ($N{=}300$)", C_N300)                      # 5 tests per point, as the N = 1k (John 11-05)
        draw(t1, "trained_N400", r"Trained ($N{=}400$)", C_N400)
        draw(t1, "trained_N500", r"Trained ($N{=}500$)", C_N500)
        if width_1k:
            x, acc, _ = pooled("trained")                                             # the N = 1k: as drawn in pm's right panel
            t1.plot(x, pm._pos(J - acc), color=C_N1K, lw=2, label=rf"Trained ($N{{=}}1k$), {count(params('trained'))} trainable params")
            print(f"  trained                  {acc[-1]:.3f} at training episode {x[-1]}")
    c_echo = next(n["color"] for n in nets if n["stem"] == "sanity/reservoir_ctx_2")   # Echo N = 1k on the left
    c_2k = next((n["color"] for n in nets if n["stem"] == pm.ESN2K["stem"]), cmp.C_ECHO)
    c_10k = next(n["color"] for n in nets if "10000N" in n["stem"])
    for at, ae, (title, echoes, trained) in zip((t2, t3), (e2, e3), PANELS):
        for run, name, col in zip(echoes, (r"Echo ($N{=}200$)", r"Echo ($N{=}500$)", r"Echo ($N{=}1k$)", r"Echo ($N{=}2k$)",
                                           r"Echo ($N{=}10k$)"), (C_E200, C_E500, c_echo, c_2k, c_10k)):
            draw(ae, run, name, col)
        for run, name, col in zip(trained if width_1k else trained[:-1], (r"Trained ($N{=}50$)", r"Trained ($N{=}100$)", r"Trained ($N{=}150$)", r"Trained ($N{=}200$)",
                                                                           r"Trained ($N{=}300$)", r"Trained ($N{=}400$)", r"Trained ($N{=}500$)", r"Trained ($N{=}1k$)"),
                                  (C_N50, pm.C_N100, C_N150, cmp.C_TRAIN, C_N300, C_N400, C_N500, C_N1K)):
            draw(at, run, name, col)
        at.set_title(title, fontsize=15)
    for ax in (t1, e1):
        pm._style(ax, J, b, step)
        ax.set_ylabel("Optimal $-$ Network Accuracy")
    for ax in (t2, t3, e2, e3):
        pm._style(ax, J, b, step, label=False)
    for ax in (t1, t2, t3, e1, e2, e3):
        by_width(ax)
    for ax in (t1, t2, t3):
        ax.set_xlabel("")
    t1.set_title("Marginal output, goal supervision", fontsize=15)
    fig.tight_layout()
    fig.canvas.draw()
    for ax in axes.flat:
        for name in pm._collisions(ax):
            print(f"  WARNING: a curve passes within 3 pt of the text {name!r}")
    print(f"  y-axis {t1.get_ylim()[0]:.4f} to {t1.get_ylim()[1]:.3f}")
    path = path or fig_path("compare_matched_params_loglog_joint_panels.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    figure()
    from main.utils import fig_path
    figure(path=fig_path("compare_matched_params_loglog_joint_panels_n1.png"), n1=True)   # John 11-05: one run per curve, batch 8,000
    figure(path=fig_path("compare_matched_params_loglog_joint_panels_no1k.png"), width_1k=False)   # John 11-04: without the Trained N = 1k
