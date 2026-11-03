"""The paper's Figure 9 (compare_matched_params.plot_loglog, titled there "Controlling for potential confounds") with a
second panel for the training-dynamics runs of 29 Oct - 1 Nov 2026. John's requests (10-31, in turn): add the new
trainings; a second column with the new trained N = 1k marginal and joint outputs; the joint-loss ESN in red on the
right, all new curves solid, paper-ready legends, no note; the joint-output ESN on the marginal loss on the right,
dashed, the trained joint output blue dashed; left title "Controlling for number of trainable parameters", no parameter
counts on the right; the five width-100 reps on the left in light blue, the joint-output ESN off the left, every right
curve stating its supervision ("goal" / "full supervision"); then "clip the x-axis so both panels match" (the same
amount of data for every curve), an arrow from the solid to the dashed blue curve reading "expanding output space", and
no output type or supervision in the left legend (every left network has marginal belief updates, goal supervision);
then no "N = 1k" and no "Naive bound" label on the right (its line stays) and an arrow from the dashed to the solid red
curve reading "expanding supervision"; then the full-supervision ESN dotted ("so people dont think solid is expanded
supervision": solid = marginal output + goal supervision, dashed = joint output, dotted = joint output + full); then
the 1k marginal-output ESN on the right (solid red; from the same seed as the other right-panel runs: echo_s20261029),
a fresh 2k ESN on the left once trained (train_esn_widths.py, launched 10-31 22:17), and each legend in the order of
its curves; then (11-01) the left legend by increasing performance, top to bottom (the curve's mean over all its points
on the axis: Echo 1k, 2k, 10k, then Trained N = 100, 200 - light to solid red, light to solid blue, as John expected;
the three ESNs end within 0.02 of each other, at the naive bound); then the right panel's
marginal ESN in the light red of the left's Echo N = 1k, and a shorter right legend: the paper's "network -> output"
with "belief updates" left to the panel title and goal supervision the unstated default (only full supervision named);
then (John: "the legend misses information now ... indicate network type, output type and feedback type instead of each
individual line") a factorized right legend: network by colour (Echo red, Trained blue), output and supervision by
line style (grey handles, unlike the black naive bound): solid = marginal output, goal supervision; dashed = joint
output, goal supervision; dotted = joint output, full supervision; then the marginal ESN off the right again.
-> main/REPLICATION/figure_bin/compare_matched_params_loglog_new_runs.png

Supervision: goal supervision = KL to Bayes's marginal posterior of the goal variable (the paper's loss); full
supervision = KL to Bayes's joint posterior over the 100 value pairs, at every step.
Left: the paper's networks without its joint-output ESN, drawn as compare_matched_params draws them (one test batch
every 500 episodes, taken at the start of each 500-episode block and plotted at its end; the trained N = 200 with its
+-1 SE band over 5 reps; the naive bound with its +-1 SD band over the cached environment draws), and the trained
N = 100 (light blue, mean +-1 SE over 5 reps: seed 20261029, Bayes on the CPU, and 20261030-33 with gpu_inference, the
same exact posterior, launched 10-31 21:47; drawn only once all five have finished), sampled like the paper's runs
(John 11-01: "it looks too smooth to be real data"): one test batch per 500 training episodes, the one at the start of
each block (training episode (k - 1) * 500, i.e. episode (k - 1) * 625), plotted at the block's end; its first version
averaged all ~125 test batches of each block, ~15x less noise per point than its neighbours (single test batch SD
0.006 in all runs). Right: four runs from seed 20261029
(the same environment, initial weights and test episodes), all N = 1k: the trained LSTM (blue) with marginal (solid)
and joint (dashed) belief updates, and the ESN (red) with joint belief updates under goal (dashed) and full (dotted)
supervision (the marginal ESN, echo_s20261029, final 0.431, lies within 0.5-4% of the dashed one throughout); the
arrows stand at ARROW_OUT (output space, label on the left) and ARROW_SUP (supervision, label on the right) training
episodes, each from one curve to the other.
The new runs (train_training_dynamics.py) test every 5th episode, and a test episode trains nothing (env_model_manager:
no backward pass), so their 50,000 episodes hold 40,000 training episodes; the paper's runs test every 500th (49,900).
The new runs are plotted against training episodes (episode minus the test episodes before it). Every curve is sampled
as the paper's runs were tested (John 2026-11-01: the same test sampling in one figure): the test at the start of each
500-episode block, plotted at the block's end. Test episodes per point differ by data: left, Echo N = 10k 2,000 (one
rep, batch 2,000), Echo N = 1k and 2k 8,000, Trained N = 200 10,000 (5 reps x 2,000), Trained N = 100 40,000 (5 seeds
x 8,000); right, 40,000: the first TEST_POOL = 5 tests of each block, within 16 training episodes (paper_style_pooled;
John: "batch together testing episodes on the right"). The right panel's noise is set mostly by its gaps' size: one
8,000-episode test has SD 0.006, about the trained marginal run's late gap (.008); with single tests 4 of its 80 points
fell at or below the bound. Until 11-01 the right panel showed 500-episode block means (~125 tests, at the block's end,
half a block late); a moving average over 5 points for every curve was tried and dropped (it cut each curve's ends).
Non-positive gaps are left out (_pos), not clipped. Both x-axes end at XMAX =
40,000, where the new runs end, so every curve shows the same amount of training. One y-scale; every curve is
subtracted from the same cached optimal bound (matched_params_bounds.npz, 0.7677 at T), which the new runs' own test
episodes reproduce exactly (naive 0.4470 vs the cache's 0.4488). At 40,000 training episodes (the last 500-episode block's mean; the plotted single tests: .753, .692, .433, .666): trained
marginal 0.762, trained joint 0.689, ESN joint under goal supervision 0.433 (trapped at the naive bound), ESN joint
under full supervision 0.666; trainable parameters (counted as compare_matched_params counts them, a reservoir's LSTM
excluded): 8.0M, 8.1M, 118k, 118k. 20 x 5.2 in: two of the paper's 10-in panels. figure() prints any text (legend
entry or label) a curve passes within 3 pt of.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_matched_params_new.py"""
import os
import sys
import json
import glob
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION")]
import torch                                                                      # noqa: E402
import matplotlib.pyplot as plt                                                   # noqa: E402
import matplotlib.ticker as mticker                                               # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
import compare_matched_params as cmp                                              # noqa: E402

RUNS = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
SEED = 20261029
FULL = "full supervision"                            # goal supervision: the unstated default
DASH = (0, (6, 2))                                    # longer dashes than the naive bound's, which one curve lies on
DOT = (0, (1, 1.5))                                   # full supervision: not solid, which reads as the baseline
# labels name the curves for _collisions only: the right legend is factorized (_legend_factors)
NEW = (("trained", r"Trained $\to$ marginal", cmp.C_TRAIN, "-"),
       ("joint_trained", r"Trained $\to$ joint", cmp.C_TRAIN, DASH),
       ("joint_echo", r"Echo $\to$ joint", cmp.C_ECHO, DASH),
       ("joint_echo_jl", rf"Echo $\to$ joint, {FULL}", cmp.C_ECHO, DOT))
DROP = ("sanity/joint_reservoir_ctx_2",)              # the paper's joint-output ESN, off the left panel (John 10-31)
ESN2K = dict(label=r"Echo ($N{=}2k$) $\to$ marginal belief updates", reps=None, color=None, ls="-",
             stem="experimental/esn_widths/reservoir_ctx_2_2000N")  # train_esn_widths.py: the paper's 1k protocol
N100 = ("trained_N100", (20261029, 20261030, 20261031, 20261032, 20261033))
C_N100 = "#92c5de"                                    # light blue: fewer trainable parameters than the N = 200
XMAX = 40_000                                         # training episodes in the new runs (50,000, every 5th a test)
XTICKS = (500, 1e3, 2e3, 3e3, 5e3, 1e4, 2e4, 3e4, 4e4)
ARROW_OUT, ARROW_SUP = 30_000, 12_000                 # block ends; labels left / right, clear of curves and legend
YMIN = None                                           # shared y floor (None: automatic)
LABEL_OUT = None                                      # "expanding output space" height (None: midway between the curves)
TEST_POOL = 5                                         # right panel: tests pooled per point (5 x 8,000 = 40,000 episodes)
SMOOTH_W = 1                                          # every curve: a centred moving average over this many tests (1: single)


def _dir(run, seed=SEED):
    return os.path.join(RUNS, f"{run}_s{seed}")


def finished(run, seed=SEED):
    p = os.path.join(_dir(run, seed), "progress.log")
    return os.path.exists(p) and open(p).read().rstrip().endswith("DONE")


def new_curve(run, seed=SEED, block=500):
    """(x, accuracy at T): test logs averaged over each block of `block` training episodes; a test episode at episode
    e (every checkpoint_every-th, no update) follows e - e // checkpoint_every training episodes"""
    d = _dir(run, seed)
    every = json.load(open(os.path.join(d, "run.json")))["cfg"]["checkpoint_every"]
    z = [np.load(f) for f in sorted(glob.glob(os.path.join(d, "test_*.npz")))]
    ep = np.concatenate([q["episode"] for q in z]).astype(int)
    acc = np.concatenate([q["acc"][:, -1] for q in z]).astype(float)
    k = np.ceil(np.maximum(ep - ep // every, 1) / block).astype(int)           # (k-1)*block < trained <= k*block
    n = np.bincount(k)
    s = np.bincount(k, acc)
    keep = np.flatnonzero(n[1:] > 0) + 1
    return keep * block, s[keep] / n[keep]


def params(run, seed=SEED):
    """trainable parameters as compare_matched_params counts them (a reservoir's LSTM excluded)"""
    d = _dir(run, seed)
    w = torch.load(os.path.join(d, "final_net.pth"), map_location="cpu", weights_only=False)["weights"]
    res = json.load(open(os.path.join(d, "run.json")))["cfg"].get("reservoir", False)
    return sum(v.numel() for k, v in w.items() if not (res and k.startswith("LSTM")))


def paper_style_curve(run, seed=SEED, block=500):
    """(x, accuracy at T) sampled like the paper's runs: the single test batch at the start of each block of `block`
    training episodes (training episode (k - 1) * block), plotted at the block's end (k * block)"""
    d = _dir(run, seed)
    every = json.load(open(os.path.join(d, "run.json")))["cfg"]["checkpoint_every"]
    z = [np.load(f) for f in sorted(glob.glob(os.path.join(d, "test_*.npz")))]
    ep = np.concatenate([q["episode"] for q in z]).astype(int)
    acc = np.concatenate([q["acc"][:, -1] for q in z]).astype(float)
    trained = ep - ep // every
    k = np.arange(1, XMAX // block + 1)
    at = {t: i for i, t in enumerate(trained)}                                     # first test at each training count
    idx = np.array([at[(kk - 1) * block] for kk in k])
    assert (trained[idx] == (k - 1) * block).all()
    return k * block, acc[idx]


def paper_style_pooled(run, seed=SEED, block=500, n_tests=None):
    """paper_style_curve with each point pooled over the first n_tests tests of its block (a test every 4 training
    episodes, so within 4 * (n_tests - 1) of the block's start), plotted at the block's end: n_tests x 8,000 episodes per
    point (John 2026-11-01: "batch together testing episodes on the right to make the noise more manageable")"""
    n_tests = TEST_POOL if n_tests is None else n_tests
    d = _dir(run, seed)
    every = json.load(open(os.path.join(d, "run.json")))["cfg"]["checkpoint_every"]
    z = [np.load(f) for f in sorted(glob.glob(os.path.join(d, "test_*.npz")))]
    ep = np.concatenate([q["episode"] for q in z]).astype(int)
    acc = np.concatenate([q["acc"][:, -1] for q in z]).astype(float)
    trained = ep - ep // every
    at = {t: i for i, t in enumerate(trained)}
    k = np.arange(1, XMAX // block + 1)
    step = every - 1                                                               # training episodes between tests
    idx = np.array([[at[(kk - 1) * block + j * step] for j in range(n_tests)] for kk in k])
    return k * block, acc[idx].mean(1)


def n100(ax, J):
    """the five width-100 reps, mean +-1 SE, once all have finished; one test batch per 500 training episodes, as the
    paper's runs"""
    run, seeds = N100
    done = [s for s in seeds if finished(run, s)]
    if len(done) < len(seeds):
        print(f"  {run}: {len(done)} of {len(seeds)} reps finished; drawn when all are")
        return
    curves = [paper_style_curve(run, s) for s in seeds]
    x = curves[0][0][:min(len(c[0]) for c in curves)]
    a = np.stack([c[1][:len(x)] for c in curves])
    m, se = a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a))
    x, m, se = _smooth(x, m, se)
    ax.fill_between(x, J - m - se, J - m + se, color=C_N100, alpha=.22, lw=0)
    ax.plot(x, _pos(J - m), color=C_N100, lw=2,
            label=rf"Trained ($N{{=}}100$), {params(run, seeds[0]) / 1e3:.0f}k trainable params ($n{{=}}{len(a)}$)")
    print(f"  {run} (n={len(a)}): {m[-1]:.3f} +- {se[-1]:.3f} at training episode {x[-1]}")


def paper(ax, J, net, step):
    """one of the paper's networks, as compare_matched_params draws it; legend: name and trainable parameters"""
    a = cmp.step_acc(net, step)
    m = a.mean(0)
    x = (np.arange(len(m)) + 1) * cmp.CKPT_EVERY
    if a.shape[0] > 1:
        se = a.std(0, ddof=1) / np.sqrt(a.shape[0])
        _, se = _smooth(x, se)
        ax.fill_between(_smooth(x, m)[0], J - _smooth(x, m)[1] - se, J - _smooth(x, m)[1] + se, color=net["color"], alpha=.22, lw=0)
    x, m = _smooth(x, m)
    tail = rf" ($n{{=}}{a.shape[0]}$)" if a.shape[0] > 1 else ""
    name = net["label"].split(r" $\to$")[0]
    ax.plot(x, _pos(J - m), color=net["color"], lw=2,
            label=f"{name}, {cmp.trainable_params(net) / 1e3:.0f}k trainable params{tail}")


def _arrow(ax, J, frm, to, x0, text, side, gap, label_y=None):
    """an arrow at training episode x0 from curve frm to curve to ((x, accuracy) pairs), stopping the factors gap =
    (tail, head) short of them, labelled on its side ("left" / "right") midway between the curves (log scale)"""
    y0 = J - frm[1][np.flatnonzero(frm[0] == x0)[0]]
    y1 = J - to[1][np.flatnonzero(to[0] == x0)[0]]
    up = y1 > y0
    ax.annotate("", xy=(x0, y1 / gap[1] if up else y1 * gap[1]), xytext=(x0, y0 * gap[0] if up else y0 / gap[0]),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5, mutation_scale=14, shrinkA=0, shrinkB=0))
    left = side == "left"
    ax.text(x0 / 1.08 if left else x0 * 1.08, label_y or np.sqrt(y0 * y1), text, ha="right" if left else "left", va="center",
            fontsize=13)
    print(f"  arrow '{text}' at {x0}: {y0:.4f} -> {y1:.4f}")


def _collisions(ax, pad=3):
    """texts (legend entries, labels) whose box, padded by pad points (a curve's half-width: 1), a curve passes through;
    the naive bound's own line is skipped (its label sits on it by design)"""
    r = ax.figure.canvas.get_renderer()
    px = pad * ax.figure.dpi / 72
    texts = [t for t in list(ax.get_legend().get_texts()) + list(ax.texts) if t.get_text()]
    boxes = [(t.get_text(), t.get_window_extent(r).padded(px)) for t in texts]
    hit = []
    for ln in ax.get_lines():
        if ln.get_label().startswith("_"):
            continue
        xy = ax.transData.transform(np.column_stack([np.asarray(v, float) for v in ln.get_data()]))
        if len(xy) < 2:
            continue
        f = np.linspace(0, 1, 25)[:, None, None]                                  # points along every segment
        xy = (xy[:-1] * (1 - f) + xy[1:] * f).reshape(-1, 2)
        hit += [name for name, bb in boxes if name not in hit and np.any((xy[:, 0] >= bb.x0) & (xy[:, 0] <= bb.x1)
                                                                         & (xy[:, 1] >= bb.y0) & (xy[:, 1] <= bb.y1))]
    return hit


def _smooth(x, *ys):
    """every curve the same way (2026-11-01): a centred moving average over SMOOTH_W consecutive tests (the paper's own
    knob, compare_matched_params.plot_loglog's SMOOTH, 1 there), x at the windows' centres (on the 500-episode grid)"""
    k = np.ones(SMOOTH_W) / SMOOTH_W
    return (np.convolve(np.asarray(x, float), k, "valid"),) + tuple(np.convolve(np.asarray(y, float), k, "valid") for y in ys)


def _pos(gap):
    """a gap curve for the log axis: values at or below 0 (single test batches at or above the optimal bound) left
    out, not clipped to the axis floor"""
    gap = np.asarray(gap, float)
    return np.where(gap > 0, gap, np.nan)


def _mean_gap(line):
    """a curve's mean gap to optimal over all its points on the axis (lower = better)"""
    x, y = (np.asarray(v, float) for v in line.get_data())
    return np.nanmean(y[x <= XMAX])


def _style(ax, J, b, step, label=True):
    """the paper's axes: naive bound with its +-1 SD band over the cached environment draws (labelled on the line at the
    left end, as in the paper, if label), log-log"""
    fact = J - b[f"{cmp.BOUND_KEY}_naive"][step - 1]
    draws = [k[:-6] for k in b if k.endswith("_joint") and b[k].size == step]
    sd = np.std([b[f"{t}_joint"][step - 1] - b[f"{t}_naive"][step - 1] for t in draws], ddof=1)
    ax.axhspan(fact - sd, fact + sd, color="k", alpha=.13, lw=0, zorder=9)
    ax.axhline(fact, c="k", lw=2, ls="--", zorder=10)
    if label:
        ax.text(.01, fact, "Naive bound", color="k", fontsize=15, va="center", ha="left",
                transform=ax.get_yaxis_transform(), zorder=11, bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5))
    ax.set(xscale="log", yscale="log", xlim=(cmp.CKPT_EVERY, XMAX), xlabel="Training Episode")
    ax.xaxis.set_major_locator(mticker.FixedLocator(XTICKS))
    ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1., 2., 3., 5.)))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
        axis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(alpha=.25, lw=.6, which="both")
    cmp._despine(ax)


def _legend_curves(ax):
    """one entry per curve, by increasing performance top to bottom (the mean gap over the whole run, largest first)"""
    lines = [ln for ln in ax.get_lines() if not ln.get_label().startswith("_")]
    lines.sort(key=_mean_gap, reverse=True)
    for ln in lines:
        print(f"  legend: {ln.get_label()[:40]:40s} mean gap {_mean_gap(ln):.4f}")
    ax.legend(lines, [ln.get_label() for ln in lines], loc="lower left", frameon=False, fontsize=12)


def _legend_factors(ax):
    """network by colour (column 1), output and supervision by line style (column 2, grey)"""
    def seg(**k):
        return Line2D([], [], lw=2, **k)
    handles = [seg(color=cmp.C_ECHO), seg(color=cmp.C_TRAIN), seg(color="none"),
               seg(color="0.35"), seg(color="0.35", ls=DASH), seg(color="0.35", ls=DOT)]
    labels = ["Echo", "Trained", "", "marginal output, goal supervision", "joint output, goal supervision",
              f"joint output, {FULL}"]
    ax.legend(handles, labels, ncol=2, loc="lower left", frameon=False, fontsize=12, handlelength=3, columnspacing=2.5)


def figure(step=cmp.STEP, ymin=YMIN, path=None):
    from main.utils import fig_path
    cmp.apply_fig_style()
    b = dict(np.load(cmp.BOUNDS_NPZ))
    J = b[f"{cmp.BOUND_KEY}_joint"][step - 1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 5.2), sharey=True)
    extra = [dict(ESN2K)] if os.path.exists(os.path.join(cmp.DATA, ESN2K["stem"] + "_net.pth")) else []
    if not extra:
        print(f"  {ESN2K['stem']}: not trained yet; drawn when it is")
    nets = cmp.resolve_colors([dict(n) for n in cmp.NETWORKS if n["stem"] not in DROP] + extra)  # copies: it colours
    for net in nets:                                                              # echoes, light to dark
        if "reservoir" in net["stem"]:
            paper(a1, J, net, step)
    n100(a1, J)                                                                   # then trained, by width
    for net in nets:
        if "reservoir" not in net["stem"]:
            paper(a1, J, net, step)
    curves = {}
    for run, name, col, ls in NEW:
        curves[run] = x, acc = _smooth(*paper_style_pooled(run))   # 40,000 test episodes per point (2026-11-01)
        a2.plot(x, _pos(J - acc), color=col, lw=2, ls=ls, label=name)
        print(f"  {run:14s} {acc[-1]:.3f} at training episode {x[-1]}")
    _arrow(a2, J, curves["trained"], curves["joint_trained"], ARROW_OUT, "expanding output space", "left", (1.25, 1.15),
           label_y=LABEL_OUT)
    _arrow(a2, J, curves["joint_echo"], curves["joint_echo_jl"], ARROW_SUP, "expanding supervision", "right",
           (1.06, 1.12))
    _style(a1, J, b, step)
    _style(a2, J, b, step, label=False)
    _legend_curves(a1)
    _legend_factors(a2)
    if ymin is not None:
        a1.set_ylim(bottom=ymin)
    a1.set_ylabel("Optimal $-$ Network Accuracy")
    a1.set_title("Controlling for number of trainable parameters", fontsize=15)
    a2.set_title("Marginal vs. joint belief updates", fontsize=15)
    fig.tight_layout()
    fig.canvas.draw()
    for ax in (a1, a2):
        for name in _collisions(ax):
            print(f"  WARNING: a curve passes within 3 pt of the text {name!r}")
    print(f"  y-axis {a1.get_ylim()[0]:.4f} to {a1.get_ylim()[1]:.3f}")
    path = path or fig_path("compare_matched_params_loglog_new_runs.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    figure()
