"""Feedforward (mode FF) and LSTM runs at loss power 1 from train_ff_deep.py: what depth, width and frozen layers buy.

    python main/REPLICATION/claude_plots/plot_ff_deep.py              # every figure below, from the saved curves
    python main/REPLICATION/claude_plots/plot_ff_deep.py --collect    # first re-evaluate the saved nets (GPU 0, ~10 min)

Data: FF at width 1000 (depths 1-9, two seeds; depth 4 and 9 with layers frozen at initialization), width 2000 (the
same, one seed), depth 4 at width 4000 (lr 3e-4), and the power-1 LSTMs (1.3M, 2.9M, 8.05M trainable parameters).
All lr 1e-3 unless noted, 20k episodes, final-step KL(joint || net) (mean of the last 3 checkpoints for final values).
FF sees all 30 steps' observation means at once; the LSTM sees the steps one by one and trains on all of them.

ff_deep_power1.png           training curves, fully trained FF by depth at each width beside the LSTMs (curves_figure)
ff_deep_power1_scatter.png   every finished run, KL against trainable parameters, labelled (power1_scatter_figure)
ff_deep_frozen.png           depth 9 and depth 4 with k layers frozen from either end, widths 1000 and 2000, through
                             training and against k beside removing the layers (frozen_figure)
ff_deep_widths.png           fully trained FF at each width and the LSTMs against trainable parameters (widths_figure)
ff_deep_widths_frozen.png    removed vs frozen layers against the number that train, per width (widths_frozen_figure)
ff_deep_heatmaps.png         raw KL as heatmaps, final and through training (heatmaps_figure)
ff_deep_relations.png        KL ratios: cost of freezing, what width buys, every net vs the best FFs (relations_figure)
ff_deep_disagreement.png     FF, LSTMs and reservoirs against joint and naive Bayes: whose answer where they disagree,
                             KL from each, accuracy against confidence
ff_deep_frozen_widths.png    FF with only readin and readout trained (depths 1-4) against width, beside the reservoirs
                             (ff_deep_frozen_widths_all_depths.png: every depth, with the shared-readin net)
                             (frozen_widths_figure; --collect-missing adds new nets to the disagreement data)
                             (disagreement_figure; data from --collect or --collect-disagreement, GPU 0, ~10 min)

--collect evaluates every saved power-1 FF net (20k) and LSTM (50k) on fresh test episodes from its own
stored world and writes eval_final_step.json: final-step accuracy, joint and naive accuracy, and KL in both
directions against the joint posterior and between joint and naive.
"""
import os, re, sys, glob, json, inspect
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter

root = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
while not os.path.exists(os.path.join(root, 'main', 'CognitiveGridworld.py')):
    root, prev = os.path.dirname(root), root
    if root == prev:
        raise RuntimeError("project root not found")
sys.path.insert(0, root)
from main.utils import fig_path, apply_fig_style

DATA = os.path.join(root, "main", "DATA")
OUT = os.path.join(DATA, "experimental", "ff_deep")
DEPTHS = list(range(1, 10))
RAMP = LinearSegmentedColormap.from_list("depth", ["#86b6ef", "#0d366b"])
NORM = BoundaryNorm(np.arange(.5, 10), RAMP.N)
NAME = re.compile(r"d(\d+)_p([0-9.]+)_lr([0-9.e-]+)_rep(\d+)")
EVAL = os.path.join(OUT, "eval_final_step.json")
LSTM1_COLOR = "#e08214"                     # LSTM, power 1 (train_ff_deep.py lstm)
EVAL_ROUNDS, EVAL_BATCH = 2, 8000


def collect():
    """Final-step evaluation of the saved power-1 FF nets and the LSTMs on fresh test episodes -> EVAL."""
    from main.CognitiveGridworld import CognitiveGridworld
    from main.REPLICATION.trainings.train_ff_deep import BASE

    def kl(p, q):
        p, q = np.clip(p, 1e-12, 1), np.clip(q, 1e-12, 1)
        return (p * np.log(p / q)).sum(-1).mean()

    def evaluate(make):
        rows = []
        for _ in range(EVAL_ROUNDS):
            grab = {}
            def logger(self):
                grab.update({k: np.asarray(getattr(self, a), float)[:, -1] for k, a in
                             (("q", "model_goal_belief"), ("P", "joint_goal_belief"), ("N", "naive_goal_belief"))})
                with torch.no_grad():                   # the power-1 training loss, final step only
                    D = self.model.DKL_sym(self.model.classifier_goal_belief[:, -1].float(), self.model.joint_goal_belief[:, -1].float(), PM = False)
                    grab["loss_final_p1"] = float((D - D.min() + 1e-8).mean())
                return {}
            g = make(logger)
            q, P, N = grab["q"], grab["P"], grab["N"]
            rows.append(dict(acc_net = g.test_accs[0, -1], acc_joint = g.joint_acc.mean(0)[-1], acc_naive = g.naive_acc.mean(0)[-1],
                             kl_joint_net = kl(P, q), kl_net_joint = kl(q, P), kl_joint_naive = kl(P, N), kl_naive_joint = kl(N, P),
                             loss_final_p1 = grab["loss_final_p1"]))
            del g; torch.cuda.empty_cache()
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    res = {}
    for f in sorted(glob.glob(os.path.join(OUT, "d*_p1_*_net.pth"))):
        stem = os.path.relpath(f, DATA)[:-len("_net.pth")]
        name = os.path.basename(stem)
        depth = int(NAME.fullmatch(name).group(1))
        res[name] = evaluate(lambda lg: CognitiveGridworld(**dict(BASE, training = False, episodes = 1, checkpoint_every = 1,
            batch_num = EVAL_BATCH, readin_depth = depth, classifier_LR = 1e-3, load_env = stem, external_logger = lg)))
        print(name, res[name], flush = True)
    for r in range(5):
        stem = f"sanity_default_reps/fully_trained_ctx_2_rep{r}"
        res[f"lstm_rep{r}"] = evaluate(lambda lg: CognitiveGridworld(mode = "SANITY", cuda = 0, episodes = 1, checkpoint_every = 1,
            realization_num = 10, obs_num = 5, show_plots = False, state_num = 500, learn_embeddings = False, classifier_LR = .001,
            ctx_num = 2, training = False, batch_num = EVAL_BATCH, step_num = 30, hid_dim = 1000, reservoir = False,
            gpu_inference = True, load_env = stem, external_logger = lg))
        print(f"lstm_rep{r}", res[f"lstm_rep{r}"], flush = True)
    res["lstm_p1_rep0"] = evaluate(lambda lg: CognitiveGridworld(**dict(BASE, mode = "SANITY", training = False, episodes = 1,
        checkpoint_every = 1, batch_num = EVAL_BATCH, classifier_LR = 1e-3, load_env = "experimental/ff_deep/lstm_p1_lr0.001_rep0",
        external_logger = lg)))
    print("lstm_p1_rep0", res["lstm_p1_rep0"], flush = True)
    json.dump(res, open(EVAL, "w"), indent = 1)
    print("saved", EVAL)


def KL(p, q):
    """KL(p || q) label; cmr10 has no '|' glyph (it draws a dash), so the bars come from mathtext."""
    return rf"KL({p}$\,\|\,${q})"


IN_COLOR, OUT_COLOR, ALL_COLOR = "#1b9e77", "#7570b3", "#4d4d4d"    # frozen from the input end / output end / all frozen
WIDE_COLOR = "#e7298a"


def load_curve(f):
    return dict(np.load(f)) if os.path.exists(f) else None


def trainable(depth, n, k = 0):
    """Trainable parameters of an FF of the given depth and width with k of its N x N layers frozen."""
    return 16 * n + (depth - 1 - k) * (n * n + n) + 20 * n + 20




def frozen_block(fig, axes, depth, letters, fs):
    """One depth's two rows (KL, accuracy) x five columns of frozen_figure; returns the k colour ramp and handles."""
    L = depth - 1
    sets = {"input": lambda k: list(range(1, k + 1)), "output": lambda k: list(range(L - k + 1, L + 1))}
    lstms, base_ref = lstm_runs(), baselines()
    KRAMP = LinearSegmentedColormap.from_list("k", ["#8fd18a", "#00441b"])
    panels = [("input", 1000), ("input", 2000), ("output", 1000), ("output", 2000)]
    for row, (key, ylabel, rowname) in enumerate((("kl_joint", KL("joint", "net") + " (nats)", KL("joint", "net")),
                                                  ("acc", "final-step test accuracy", "accuracy"))):
        for col, (side, n) in enumerate(panels):
            ax = axes[row, col]
            for k in range(1, L + 1):
                c = width_curve(depth, n, sets[side](k))
                if c is not None:
                    ax.plot(c["ep"] / 1000, c[key], color = KRAMP((k - 1) / max(L - 1, 1)), lw = 1.1)
            base = width_curve(depth, n)
            if base is not None:
                ax.plot(base["ep"] / 1000, base[key], color = "#1f1f1e", ls = "--", lw = 1.3, label = "fully trained (k = 0)")
            for label, c, lc in lstms:
                ax.plot(c["ep"] / 1000, c[key], color = lc, lw = 1, label = label)
            ax.set_xlim(0, 20)
            ax.set_xticks(range(0, 21, 5))
            ax.set_xlabel("training episodes (thousands)")
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(f"({letters[5 * row + col]}) depth {depth}, width {n}, {'first' if side == 'input' else 'last'} k frozen",
                         fontsize = fs["label"])
            refs = (((base_ref["kl_naive"], "naive", "--", NAIVE_COLOR),) if key == "kl_joint" else
                    ((base_ref["acc_naive"], "naive", "--", NAIVE_COLOR), (base_ref["acc_joint"], "joint", ":", JOINT_COLOR)))
            if key == "kl_joint":
                ax.set_yscale("log")
                ax.set_ylim(.02, 3.5)
            else:
                ax.set_ylim(.1, .8)
            for y, lab, ls, col_ in refs:
                if y is not None:
                    ax.axhline(y, ls = ls, color = col_, lw = 1)
                    ax.text(.99, y, lab, transform = ax.get_yaxis_transform(), ha = "right", va = "top" if lab == "naive" else "bottom",
                            fontsize = fs["tick"] - 1.5, color = col_)
        ax = axes[row, 4]                                        # against k, both widths
        end = lambda c: widths_value(c, key)
        for n, ls, mfc in ((1000, "-", None), (2000, "--", "white")):
            fams = (("removed: depth - k, all trained", RAMP(.67), "o", lambda k: width_curve(depth - k, n)),
                    ("first k frozen (input end)", IN_COLOR, "s", lambda k: width_curve(depth, n) if k == 0 else width_curve(depth, n, sets["input"](k))),
                    ("last k frozen (output end)", OUT_COLOR, "D", lambda k: width_curve(depth, n) if k == 0 else width_curve(depth, n, sets["output"](k))))
            for label, col_, mk, get in fams:
                xy = [(k, end(get(k))) for k in range(L + 1)]
                xy = [q for q in xy if q[1] is not None]
                if xy:
                    ax.plot(*zip(*xy), color = col_, ls = ls, marker = mk, ms = 4, lw = 1.3, mfc = mfc or col_, label = f"{label}, width {n}")
        if key == "kl_joint":
            ax.set_yscale("log")
        ref_lines(ax, [(end(c), label, lc, "-") for label, c, lc in lstms]
                  + ([(base_ref["kl_naive"], "naive Bayes", NAIVE_COLOR, "--")] if key == "kl_joint" else
                     [(base_ref["acc_naive"], "naive Bayes", NAIVE_COLOR, "--"), (base_ref["acc_joint"], "joint Bayes", JOINT_COLOR, ":")]), fs)
        ax.set_xticks(range(L + 1))
        ax.set_xlim(-.3, L + .4)
        ax.set_xlabel("k: N x N layers removed or frozen")
        ax.set_title(f"({letters[5 * row + 4]}) depth {depth}: {rowname} at 20k against k", fontsize = fs["label"])
        if key == "kl_joint":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return KRAMP


def frozen_figure():
    """Depth 9 (top two rows) and depth 4 (bottom two rows) with k of their N x N layers frozen at initialization, at
    widths 1000 and 2000 (seed 0, power 1). In each block the first row is KL(joint || net), the second accuracy, final
    step. Columns 1-4: through training, one panel per side (first k / last k frozen) and width, shaded by k, with the
    fully trained network of that depth and width (k = 0, dashed) and the 8M LSTM. Column 5: at 20k (mean of the last
    3 checkpoints) against k: k layers removed (the fully trained depth - k, same trainable parameters), frozen from the
    input end, frozen from the output end; width 1000 solid, 2000 dashed. Writes ff_deep_frozen.png."""
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, axes = plt.subplots(4, 5, figsize = (19.5, 13.2), gridspec_kw = dict(width_ratios = [1, 1, 1, 1, 1.15]))
    letters = "abcdefghijklmnopqrst"
    ramps = {9: frozen_block(fig, axes[:2], 9, letters[:10], fs), 4: frozen_block(fig, axes[2:], 4, letters[10:], fs)}
    axes[0, 0].legend(frameon = False, fontsize = fs["tick"] - 2, loc = "lower left")
    h, l = axes[1, 4].get_legend_handles_labels()
    fig.legend(h, l, loc = "lower center", ncol = 3, frameon = False, fontsize = fs["tick"] - 1.5)
    fig.tight_layout(rect = (0, .05, .955, 1), w_pad = 1, h_pad = 1.4)
    for depth, rows in ((9, (0, 1)), (4, (2, 3))):              # one k colour bar per depth, beside its two rows
        L = depth - 1
        top, bottom = axes[rows[0], 4].get_position(), axes[rows[1], 4].get_position()
        cax = fig.add_axes([.962, bottom.y0 + .02, .008, top.y1 - bottom.y0 - .04])
        fig.colorbar(ScalarMappable(norm = BoundaryNorm(np.arange(.5, L + 1), ramps[depth].N), cmap = ramps[depth]), cax = cax,
                     ticks = range(1, L + 1), label = f"depth {depth}: k frozen")
    fig.savefig(fig_path("ff_deep_frozen.png"), dpi = 150)
    print(fig_path("ff_deep_frozen.png"))


def curves_figure():
    """Power 1 through training: final-step accuracy and KL(joint || net) of the fully trained FF, depths 1-9, at width
    1000 (top; seed 0 thick, seed 1 thin) and width 2000 (bottom; with depth 4 at width 4000, lr 3e-4, dashed), each
    beside the three power-1 LSTMs (1.3M, 2.9M and 8.05M trainable parameters). Writes ff_deep_power1.png."""
    from main.REPLICATION.trainings.train_ff_deep import LSTM_WIDTHS, lstm_params
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    lstms, base_ref = lstm_runs(), baselines()
    fig, axes = plt.subplots(2, 2, figsize = (10.6, 7.2))
    for row, n in enumerate((1000, 2000)):
        for col, key in enumerate(("acc", "kl_joint")):
            ax = axes[row, col]
            for d in DEPTHS:
                c = width_curve(d, n)
                if c is not None:
                    ax.plot(c["ep"] / 1000, c[key], color = RAMP(NORM(d)), lw = 1.3)
                c1 = load_curve(os.path.join(OUT, f"d{d}_p1_lr0.001_rep1_curve.npz")) if n == 1000 else None
                if c1 is not None:
                    ax.plot(c1["ep"] / 1000, c1[key], color = RAMP(NORM(d)), lw = .6, alpha = .55)
            if n == 2000:
                c = width_curve(4, 4000)
                if c is not None:
                    ax.plot(c["ep"] / 1000, c[key], color = WIDE_COLOR, lw = 1.3, ls = "--", label = "depth 4, width 4000 (lr 3e-4)")
            for label, c, lc in lstms:
                if c is not None:
                    ax.plot(c["ep"] / 1000, c[key], color = lc, lw = 1.7, label = label)
            ax.set_xlim(0, 20)
            ax.set_xticks(range(0, 21, 5))
            ax.set_xlabel("training episodes (thousands)")
            ax.set_title(f"({'abcd'[2 * row + col]}) width {n}: " + ("accuracy" if key == "acc" else KL("joint", "net")),
                         fontsize = fs["label"])
            if key == "acc":
                ax.set_ylim(0, .8)
                ax.set_ylabel("final-step test accuracy")
                refs = ((base_ref["acc_naive"], "naive", "--", NAIVE_COLOR), (base_ref["acc_joint"], "joint", ":", JOINT_COLOR))
            else:
                ax.set_yscale("log")
                ax.set_ylabel(KL("joint", "net") + ", final step (joint Bayes: 0)")
                refs = ((base_ref["kl_naive"], "naive", "--", NAIVE_COLOR),)
            for y, lab, ls, col_ in refs:
                if y is not None:
                    ax.axhline(y, ls = ls, color = col_, lw = 1)
                    ax.text(.99, y, lab, transform = ax.get_yaxis_transform(), ha = "right", va = "top" if lab == "naive" else "bottom",
                            fontsize = fs["tick"] - 1.5, color = col_)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    axes[0, 1].legend(frameon = False, fontsize = fs["tick"] - 1.5, loc = "lower left")
    axes[1, 1].legend(frameon = False, fontsize = fs["tick"] - 1.5, loc = "lower left")
    fig.tight_layout(rect = (0, 0, .92, 1), h_pad = 1.5)
    cax = fig.add_axes([.935, .25, .012, .5])
    fig.colorbar(ScalarMappable(norm = NORM, cmap = RAMP), cax = cax, ticks = DEPTHS, label = "FF depth (width 1000: thin = seed 1)")
    fig.savefig(fig_path("ff_deep_power1.png"), dpi = 150)
    print(fig_path("ff_deep_power1.png"))




def width_curve(depth, n, frozen = ()):
    """Curve of the seed-0 power-1 run of this depth ("lstm" for the LSTM), width and frozen set, wherever it lives:
    ff_deep/widths first (other widths, and width-1000 configurations rerun there because their chosen rate moved off
    1e-3), else the original width-1000 runs (ff_deep, d4_variants, d9_frozen) at 1e-3."""
    frz = f"_frz{''.join(map(str, frozen))}" if frozen else ""
    name = "lstm" if depth == "lstm" else f"d{depth}"
    width = f"_N{n}" if n != 1000 else ""
    pat = re.compile(rf"{name}{width}_p1_lr[0-9.e-]+{frz}_rep0_curve\.npz")
    fs = [f for f in glob.glob(os.path.join(OUT, "widths", f"{name}{width}_p1_*_rep0_curve.npz")) if pat.fullmatch(os.path.basename(f))]
    fixed = [f for f in fs if "_lr0.001" in os.path.basename(f)]
    if len(fixed) == 1 or len(fs) == 1:  # the fixed-rate (1e-3) run when there are two, e.g. width 4000 depth 4 at 3e-4 too
        return load_curve((fixed or fs)[0])
    if n == 1000:
        folder = OUT if not frozen else os.path.join(OUT, "d4_variants" if depth == 4 else "d9_frozen")
        return load_curve(os.path.join(folder, f"{name}_p1_lr0.001{frz}_rep0_curve.npz"))
    return None


def widths_value(c, key, at_ep = None):
    """Final value (mean of the last 3 checkpoints of a finished run) or, with at_ep, the value at that episode for any
    run that has reached it."""
    if c is None:
        return None
    if at_ep is None:
        return float(np.mean(c[key][-3:])) if len(c["ep"]) >= 41 else None
    i = np.flatnonzero(c["ep"] >= at_ep)
    return float(c[key][i[0]]) if len(i) else None


NAIVE_COLOR, JOINT_COLOR = "#8a8984", "#1f1f1e"


def baselines():
    """Joint- and naive-Bayes references in the seed-0 world: final-step accuracy of each (from the width-1000 LSTM's
    test batches) and KL(joint || naive), averaged over the seed-0 FF evaluations in EVAL (about 2.7 nats). On KL
    axes joint Bayes is KL = 0, which a log axis cannot show."""
    c = width_curve("lstm", 1000)
    kl = None
    if os.path.exists(EVAL):
        vals = [v["kl_joint_naive"] for k, v in json.load(open(EVAL)).items() if k.startswith("d") and k.endswith("rep0")]
        kl = float(np.mean(vals)) if vals else None
    return dict(acc_joint = float(np.mean(c["joint"])), acc_naive = float(np.mean(c["naive"])), kl_naive = kl)


def lstm_runs():
    """[(label, curve, colour)] of the power-1 LSTMs, smallest first, light to dark orange."""
    from main.REPLICATION.trainings.train_ff_deep import LSTM_WIDTHS, lstm_params
    ns = sorted(LSTM_WIDTHS + [1000])
    ramp = LinearSegmentedColormap.from_list("lstm", ["#fdbe85", "#a63603"])
    return [(f"LSTM {lstm_params(n) / 1e6:.1f}M", width_curve("lstm", n), ramp(i / max(len(ns) - 1, 1))) for i, n in enumerate(ns)]


def ref_lines(ax, refs, fs):
    """Horizontal reference lines [(y, label, colour, linestyle)], labelled just outside the right edge of the axis,
    labels pushed apart so none overlap. The y range is widened to include every reference; call after the data."""
    refs = [r for r in refs if r[0] is not None]
    if not refs:
        return
    log = ax.get_yscale() == "log"
    y0, y1 = ax.get_ylim()
    lo, hi = min(r[0] for r in refs), max(r[0] for r in refs)
    y0, y1 = (min(y0, lo / 1.15), max(y1, hi * 1.15)) if log else (min(y0, lo - .02), max(y1, hi + .02))
    ax.set_ylim(y0, y1)
    f = np.log10 if log else (lambda v: v)
    items = sorted(refs, key = lambda r: r[0])
    for y, _, col, ls in items:
        ax.axhline(y, color = col, ls = ls, lw = 1, zorder = 1)
    at = [(f(y) - f(y0)) / (f(y1) - f(y0)) for y, *_ in items]   # each line's height in axis fractions
    pos, gap = list(at), .055
    for i in range(1, len(pos)):                                 # push up from the bottom, then down from the top: labels
        pos[i] = max(pos[i], pos[i - 1] + gap)                   # move only as far as their neighbours force them to
    pos[-1] = min(pos[-1], 1.)
    for i in range(len(pos) - 2, -1, -1):
        pos[i] = min(pos[i], pos[i + 1] - gap)
    for (y, label, col, _), a_, p_ in zip(items, at, pos):
        ax.plot([1., 1.03, 1.05], [a_, p_, p_], transform = ax.transAxes, color = col, lw = .6, clip_on = False)   # leader
        ax.text(1.06, p_, label, transform = ax.transAxes, color = col, fontsize = fs["tick"] - 2.5, va = "center",
                ha = "left", clip_on = False)


def relations_figure():
    """The relationships behind the scatter, as KL ratios on one diverging log scale (red: worse than the reference,
    blue: better; x factors on the colour bar). Width-1000 fully trained references are the geometric mean of both seeds.
    (a) final cost of freezing: KL of a frozen net / KL of the fully trained net with the same trainable N x N layers
    (depth m + 1) and width, depth 9 and 4, by side and k, at widths 1000 (two-seed reference) and 2000 (one seed);
    cells within the largest seed difference (x1.4) in grey type.
    (b) the same cost through training (3-checkpoint means). (c) what doubling the width buys each configuration:
    KL at width 2000 / KL at width 1000, through training (unfinished runs as far as they have got). (d) every fully
    trained FF and the LSTMs relative to the best-FF band (geometric mean of width-1000 depths 5-8, both seeds)."""
    from matplotlib.colors import TwoSlopeNorm
    from main.REPLICATION.trainings.train_ff_deep import LSTM_WIDTHS, lstm_params
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    LN10, NOISE = np.log(10), .33                      # largest seed difference in ln KL (depth 9, width 1000)
    norm, cmap = TwoSlopeNorm(0, -1.2, 1.2), plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f1efe8")
    lnkl = lambda c: None if c is None else np.log(np.asarray(c["kl_joint"], float))
    trail = lambda a: np.array([a[max(0, i - 2):i + 1].mean() for i in range(len(a))])

    def full(d, n = 1000):
        """ln KL of the fully trained depth d through training: both seeds' mean at width 1000."""
        cs = [width_curve(d, n)] + ([load_curve(os.path.join(OUT, f"d{d}_p1_lr0.001_rep1_curve.npz"))] if n == 1000 else [])
        arrs = [lnkl(c) for c in cs if c is not None]
        if not arrs:
            return None
        L = min(len(a) for a in arrs)
        return np.mean([a[:L] for a in arrs], 0)

    def ratio(a, b):                                    # log10 of the trailing-3 KL ratio over the shared checkpoints
        if a is None or b is None:
            return None
        L = min(len(a), len(b))
        return (trail(a[:L]) - trail(b[:L])) / LN10

    def configs(depth):
        """(row label, frozen set or None for fully trained, m) in a fixed order: input end k = L..1, output end 1..L-1."""
        L = depth - 1
        rows = [(f"d{depth}: all {L} frozen" if L > 1 else f"d{depth}: layer frozen", list(range(1, L + 1)), 0)]
        rows += [(f"d{depth}: first {k} frozen", list(range(1, k + 1)), L - k) for k in range(L - 1, 0, -1)]
        rows += [(f"d{depth}: last {k} frozen", list(range(L - k + 1, L + 1)), L - k) for k in range(1, L)]
        return rows

    fig = plt.figure(figsize = (17.5, 13))
    gs = fig.add_gridspec(2, 3, height_ratios = [.62, 1], hspace = .24, wspace = .42)

    def matrix_panel(ax, M, T, rows, cols, title):
        ax.imshow(np.ma.masked_invalid(M), cmap = cmap, norm = norm, aspect = "auto")
        for (i, j), v in np.ndenumerate(M):
            if np.isfinite(v):
                ax.text(j, i, f"x{10 ** v:.2g}", ha = "center", va = "center", fontsize = fs["tick"] - 2,
                        color = "#888780" if abs(v * LN10) < NOISE else ("white" if abs(norm(v) - .5) > .3 else "black"))
        ax.set_xticks(range(len(cols)), cols, fontsize = fs["tick"] - 1)
        ax.set_yticks(range(len(rows)), rows, fontsize = fs["tick"] - 1.5)
        ax.set_title(title, fontsize = fs["label"])
        ax.tick_params(length = 0)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # (a) final cost of freezing, by side and k
    rows_a, M = [], []
    for n in (1000, 2000):
        for side in ("input", "output"):
            for depth in (9, 4):
                L = depth - 1
                rows_a.append(f"w{n}: depth {depth}, {'first' if side == 'input' else 'last'} k frozen")
                row = []
                for k in range(1, 9):
                    fr = list(range(1, k + 1)) if side == "input" else list(range(L - k + 1, L + 1))
                    c = width_curve(depth, n, fr) if k <= L else None
                    fin = None if c is None or len(c["ep"]) < 41 else np.mean(lnkl(c)[-3:])
                    base = full(depth - k, n) if k <= L else None
                    row.append(np.nan if fin is None or base is None else (fin - np.mean(base[-3:])) / LN10)
                M.append(row)
        c = width_curve(2, n, [1])                               # depth 2: its one N x N layer frozen (k = 1)
        fin = None if c is None or len(c["ep"]) < 41 else np.mean(lnkl(c)[-3:])
        rows_a.append(f"w{n}: depth 2, layer frozen")
        M.append([np.nan if fin is None else (fin - np.mean(full(1, n)[-3:])) / LN10] + [np.nan] * 7)
    ax = fig.add_subplot(gs[0, :])
    matrix_panel(ax, np.array(M), None, rows_a, [str(k) for k in range(1, 9)],
                 "(a) cost of freezing at 20k: KL / KL of the fully trained net with the same trainable layers and width")
    ax.set_xlabel("k frozen N x N layers (k = depth - 1: all frozen, compared with depth 1)")
    for y, lw in ((1.5, 2), (3.5, 2), (4.5, 5), (6.5, 2), (8.5, 2)):   # thick line between the widths
        ax.axhline(y, color = "white", lw = lw)

    def time_panel(ax, rows, vals, title, breaks = ()):
        T = np.full((len(rows), 41), np.nan)
        for i, v in enumerate(vals):
            if v is not None:
                T[i, :min(len(v), 41)] = v[:41]
        ax.imshow(np.ma.masked_invalid(T), cmap = cmap, norm = norm, aspect = "auto", interpolation = "nearest",
                  extent = (-.25, 20.25, len(rows) - .5, -.5))
        ax.set_yticks(range(len(rows)), rows, fontsize = fs["tick"] - 2)
        ax.set_xticks(range(0, 21, 5))
        ax.set_xlabel("training episodes (thousands)")
        ax.set_title(title, fontsize = fs["label"])
        ax.tick_params(length = 0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for b in breaks:
            ax.axhline(b, color = "white", lw = 2.5)

    # (b) cost of freezing through training, width 1000
    rows, vals, breaks = [], [], []
    for depth in (9, 4, 2):
        if rows:
            breaks.append(len(rows) - .5)
        for label, fr, m in configs(depth):
            rows.append(label); vals.append(ratio(lnkl(width_curve(depth, 1000, fr)), full(m + 1)))
    time_panel(fig.add_subplot(gs[1, 0]), rows, vals, "(b) cost of freezing through training, width 1000", breaks)

    # (c) what doubling the width buys
    rows, vals, breaks = [], [], []
    for depth in (9, 4, 2):
        if rows:
            breaks.append(len(rows) - .5)
        rows.append(f"d{depth}: fully trained"); vals.append(ratio(lnkl(width_curve(depth, 2000)), lnkl(width_curve(depth, 1000))))
        for label, fr, m in configs(depth):
            rows.append(label); vals.append(ratio(lnkl(width_curve(depth, 2000, fr)), lnkl(width_curve(depth, 1000, fr))))
    time_panel(fig.add_subplot(gs[1, 1]), rows, vals, "(c) KL at width 2000 / KL at width 1000, same configuration", breaks)

    # (d) every fully trained FF and the LSTMs against the best-FF band
    band_runs = [full(d) for d in (5, 6, 7, 8)]
    L = min(len(a) for a in band_runs)
    band = np.mean([a[:L] for a in band_runs], 0)
    rows, vals = [f"FF w1000 d{d}" for d in DEPTHS], [ratio(full(d), band) for d in DEPTHS]
    rows += [f"FF w2000 d{d}" for d in DEPTHS]; vals += [ratio(lnkl(width_curve(d, 2000)), band) for d in DEPTHS]
    rows += ["FF w4000 d4 (lr 3e-4)"]; vals += [ratio(lnkl(width_curve(4, 4000)), band)]
    lstms = sorted(LSTM_WIDTHS + [1000])
    rows += [f"LSTM {lstm_params(n) / 1e6:.1f}M" for n in lstms]; vals += [ratio(lnkl(width_curve("lstm", n)), band) for n in lstms]
    time_panel(fig.add_subplot(gs[1, 2]), rows, vals, "(d) KL / best-FF band (width-1000 depths 5-8)", (8.5, 17.5, 18.5))

    fig.subplots_adjust(left = .08, right = .9, top = .95, bottom = .06)
    cax = fig.add_axes([.925, .1, .009, .8])
    cb = fig.colorbar(ScalarMappable(norm = norm, cmap = cmap), cax = cax)
    ticks = [.1, .25, .5, 1, 2, 4, 10]
    cb.set_ticks(np.log10(ticks), labels = [f"x{t:g}" for t in ticks])
    cb.set_label("KL ratio (log scale): red = worse than the reference, blue = better")
    fig.savefig(fig_path("ff_deep_relations.png"), dpi = 150)
    print(fig_path("ff_deep_relations.png"))


def heatmaps_figure():
    """KL(joint || net) heatmaps on one log colour scale (bright = closer to Bayes). Top: final values (mean of the last
    3 checkpoints, finished runs only, written in the cells): (a, b) at widths 1000 and 2000, for depth 9 and depth 4,
    rows compare k N x N layers removed (the fully trained depth - k, same trainable parameters), frozen from the output
    end and frozen from the input end; (c) fully trained FF, depth against width. Bottom: KL through training, (d, e)
    every depth-9 and depth-4 condition at widths 1000 and 2000, (f) the fully trained depths at widths 1000 and 2000,
    depth 4 at width 4000 and the three LSTMs. All seed 0; width 4000 is lr 3e-4."""
    from matplotlib.colors import LogNorm
    from main.REPLICATION.trainings.train_ff_deep import NEW_WIDTHS, LSTM_WIDTHS, lstm_params
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    norm, cmap = LogNorm(.02, 3.2), plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#f1efe8")
    final = lambda c: widths_value(c, "kl_joint")
    fig = plt.figure(figsize = (16.5, 10.6))
    gs = fig.add_gridspec(2, 3, width_ratios = [1.25, 1.25, .62], height_ratios = [1, 1.45], hspace = .25, wspace = .22)

    def cells(ax, M, rows, cols, title):
        ax.imshow(np.ma.masked_invalid(M), cmap = cmap, norm = norm, aspect = "auto")
        for (i, j), v in np.ndenumerate(M):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}" if v < 1 else f"{v:.1f}", ha = "center", va = "center", fontsize = fs["tick"] - 2.5,
                        color = "black" if norm(v) < .55 else "white")
        ax.set_xticks(range(len(cols)), cols, fontsize = fs["tick"] - 1)
        ax.set_yticks(range(len(rows)), rows, fontsize = fs["tick"] - 1.5)
        ax.set_title(title, fontsize = fs["label"])
        ax.tick_params(length = 0)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # (a, b) removed vs frozen, per depth and side
    for col, n in enumerate([1000] + NEW_WIDTHS[:1]):
        rows, M = [], []
        for depth in (9, 4, 2):
            D = depth - 1
            base = final(width_curve(depth, n))
            fams = (("removed", lambda k: final(width_curve(depth - k, n))),
                    ("frozen, output end", lambda k: final(width_curve(depth, n, list(range(D - k + 1, D + 1))))),
                    ("frozen, input end", lambda k: final(width_curve(depth, n, list(range(1, k + 1))))))
            if depth == 2:                                       # one N x N layer: both ends are the same frozen net
                fams = (fams[0], ("layer frozen", fams[1][1]))
            for name, get in fams:
                rows.append(f"depth {depth}: {name}")
                M.append([base if k == 0 else get(k) if k <= D else None for k in range(9)])
        M = np.array([[np.nan if v is None else v for v in r] for r in M], float)
        ax = fig.add_subplot(gs[0, col])
        cells(ax, M, rows, [str(k) for k in range(9)], f"({'ab'[col]}) width {n}: k layers removed or frozen")
        ax.set_xlabel("k (removed / frozen N x N layers); 0 = fully trained")
        ax.axhline(2.5, color = "white", lw = 3)
        ax.axhline(5.5, color = "white", lw = 3)
        if col > 0:                                              # same rows as (a)
            ax.set_yticklabels([])

    # (c) fully trained: depth x width
    widths = [1000] + NEW_WIDTHS + [4000]
    M = np.array([[np.nan if final(width_curve(d, n)) is None else final(width_curve(d, n)) for n in widths] for d in DEPTHS])
    if 4000 in widths:
        M[:, widths.index(4000)] = [final(width_curve(4, 4000)) if d == 4 else np.nan for d in DEPTHS]
    ax = fig.add_subplot(gs[0, 2])
    cells(ax, M, [f"depth {d}" for d in DEPTHS], [f"{n // 1000}k" for n in widths], "(c) fully trained: depth x width")
    ax.set_xlabel("width (4k: lr 3e-4)")

    # (d, e) depth 9 through training
    def dynamics(ax, rows, curves, title, breaks = ()):
        T = np.full((len(rows), 41), np.nan)
        for i, c in enumerate(curves):
            if c is not None:
                T[i, :len(c["kl_joint"])] = c["kl_joint"][:41]
        im = ax.imshow(np.ma.masked_invalid(T), cmap = cmap, norm = norm, aspect = "auto", interpolation = "nearest",
                       extent = (-.25, 20.25, len(rows) - .5, -.5))
        ax.set_yticks(range(len(rows)), rows, fontsize = fs["tick"] - 1.5)
        ax.set_xticks(range(0, 21, 5))
        ax.set_xlabel("training episodes (thousands)")
        ax.set_title(title, fontsize = fs["label"])
        ax.tick_params(length = 0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for b in breaks:                                         # white lines between groups of rows
            ax.axhline(b, color = "white", lw = 2.5)
        return im
    for col, n in enumerate([1000] + NEW_WIDTHS[:1]):
        rows, curves, breaks = [], [], []
        for depth in (9, 4, 2):                                  # depth 9, then 4, then 2, same row order
            L = depth - 1
            if rows:
                breaks.append(len(rows) - .5)
            rows += [f"d{depth}: all {L} frozen" if L > 1 else f"d{depth}: layer frozen"]
            curves += [width_curve(depth, n, list(range(1, L + 1)))]
            rows += [f"d{depth}: first {k} frozen" for k in range(L - 1, 0, -1)]
            curves += [width_curve(depth, n, list(range(1, k + 1))) for k in range(L - 1, 0, -1)]
            breaks += [len(rows) - L + .5, len(rows) - .5]          # after "all"; before "fully trained"
            rows += [f"d{depth}: fully trained"]; curves += [width_curve(depth, n)]
            breaks.append(len(rows) - .5)
            rows += [f"d{depth}: last {k} frozen" for k in range(1, L)]
            curves += [width_curve(depth, n, list(range(L - k + 1, L + 1))) for k in range(1, L)]
        im = dynamics(fig.add_subplot(gs[1, col]), rows, curves, f"({'de'[col]}) depths 9, 4 and 2, width {n}: through training",
                      breaks = sorted(set(breaks)))
        if col > 0:
            plt.gca().set_yticklabels([])
    # (f) fully trained depths and LSTMs
    lstms = sorted(LSTM_WIDTHS + [1000])
    rows = ([f"FF w1000 d{d}" for d in DEPTHS] + [f"FF w2000 d{d}" for d in DEPTHS] + ["FF w4000 d4"]
            + [f"LSTM {lstm_params(n) / 1e6:.1f}M" for n in lstms])
    curves = ([width_curve(d, 1000) for d in DEPTHS] + [width_curve(d, 2000) for d in DEPTHS] + [width_curve(4, 4000)]
              + [width_curve("lstm", n) for n in lstms])
    im = dynamics(fig.add_subplot(gs[1, 2]), rows, curves, "(f) fully trained FF, LSTMs",
                  breaks = (8.5, 17.5, 18.5))
    cax = fig.add_axes([.905, .12, .009, .76])
    fig.subplots_adjust(left = .1, right = .89, top = .95, bottom = .07)
    cb = fig.colorbar(ScalarMappable(norm = norm, cmap = cmap), cax = cax)
    cb.set_label(KL("joint", "net") + ", final step (log scale; bright = closer to Bayes)", labelpad = 52)   # clear of the marks
    cb.set_ticks([.02, .03, .05, .1, .2, .3, .5, 1, 2])
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    cb.ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    marks = [(final(c), label, lc) for label, c, lc in lstm_runs()] + [(baselines()["kl_naive"], "naive Bayes", NAIVE_COLOR)]
    for y, label, lc in marks:                                   # final KL of each LSTM and of naive Bayes on the colour bar
        if y is not None:
            cb.ax.axhline(y, color = lc, lw = 2)
            cb.ax.text(1.6, y, label, transform = cb.ax.get_yaxis_transform(), color = lc, va = "center", fontsize = fs["tick"] - 2.5)
    fig.savefig(fig_path("ff_deep_heatmaps.png"), dpi = 150)
    print(fig_path("ff_deep_heatmaps.png"))


def scatter_markers(axis, points, size = 1.):
    """points: (x, y, color, marker, filled, label); open markers when not filled."""
    for x, y, col, mk, filled, _ in points:
        axis.scatter(x, y, marker = mk, s = (70 if mk == "*" else 26) * size, zorder = 3, color = col,
                     facecolor = col if filled else "white", edgecolor = col, lw = 1)


def scatter_labels(fig, axis, labelled, points, fs, size = 1.):
    """Label each point right of it unless the text would cover another point there; then just above or below on the
    right; only then to its left (a label flipped left can sit beside a neighbour's marker and read as that
    neighbour's). Near-coincident points in one column part vertically. Screen units: call after the layout is final."""
    fsz = (fs["tick"] - 3) * size ** .5
    px = fig.dpi / 72
    screen = lambda q: axis.transData.transform((q[0], q[1]))
    col_of = lambda x: float(f"{x:.2g}")
    for column in sorted({col_of(q[0]) for q in labelled}):
        col_pts = sorted([q for q in labelled if col_of(q[0]) == column], key = lambda q: q[1])
        for i, q in enumerate(col_pts):
            x, y, col, _, _, label = q
            below = i + 1 < len(col_pts) and abs(np.log10(col_pts[i + 1][1] / y)) < .05
            above = i > 0 and abs(np.log10(y / col_pts[i - 1][1])) < .05
            dy, va = (-4, "top") if below and not above else (4, "bottom") if above else (0, "center")
            x0, y0 = screen(q)
            reach = (6 + .52 * fsz * len(label)) * px

            def covers(sign, dy, va):
                yc = y0 + (dy + (-fsz / 2 if va == "top" else fsz / 2 if va == "bottom" else 0)) * px
                return any(abs(oy - yc) < .75 * fsz * px and 0 < sign * (ox - x0) < reach
                           for o in points if o is not q for ox, oy in [screen(o)])
            options = [(1, dy, va), (1, 7, "bottom"), (1, -7, "top"), (-1, dy, va)]
            sign, dy, va = next((o for o in options if not covers(*o)), options[0])
            axis.annotate(label, (x, y), xytext = (5 * sign, dy), textcoords = "offset points", va = va,
                          ha = "left" if sign > 0 else "right", fontsize = fsz, color = col)


def zoom_inset(ax, pts, zoom, fs, bounds = (.06, .05, .42, .38)):
    """Inset magnifying zoom = (x0, x1, y0, y1) on log axes: the axis' lines (faint) and the points inside, marked on
    the main axis. Returns the inset and a test for points inside."""
    inside = lambda q: zoom[0] <= q[0] <= zoom[1] and zoom[2] <= q[1] <= zoom[3]
    ins = ax.inset_axes(list(bounds))
    for line in ax.get_lines():
        ins.plot(*line.get_data(), color = line.get_color(), ls = line.get_linestyle(), lw = 1, alpha = .35)
    scatter_markers(ins, [q for q in pts if inside(q)], 1.3)
    ins.set_xscale("log"); ins.set_yscale("log")
    ins.set_xlim(zoom[0], zoom[1] * 1.12); ins.set_ylim(zoom[2], zoom[3])
    ins.set_xticks([4e6, 6e6, 8e6], ["4M", "6M", "8M"])
    ins.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ins.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ins.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if round(v * 100) in (6, 8, 10, 12) else ""))
    ins.tick_params(labelsize = fs["tick"] - 3, which = "both")
    ax.indicate_inset_zoom(ins, edgecolor = "#888780")
    return ins, inside


def power1_scatter_figure():
    """Every finished power-1 run at 20k episodes (mean of the last 3 checkpoints), KL(joint || net) against trainable
    parameters, one panel per width (1000, 2000), each point labelled: FF fully trained (d<depth>), depth 4 and 9 with k
    hidden N x N layers frozen from the input end (in<k>), the output end (out<k>) or all of them (all). Both panels
    also show the power-1 LSTMs (by width), depth 4 at width 4000 (lr 3e-4) and depth 30 at width 4000 with all 29 N x N
    layers frozen, without and with residual skips (open markers); the width-1000 panel the fully trained
    second seed as small open markers. Faint lines join each family; insets magnify the crowded regions. Labelled points
    are seed 0: one world."""
    from main.REPLICATION.trainings.train_ff_deep import LSTM_WIDTHS, ff_params, lstm_params
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    end = lambda c: widths_value(c, "kl_joint")
    FULL_COLOR = RAMP(.67)
    ZOOMS = {1000: (3.6e6, 8.7e6, .052, .125), 2000: (1.0e7, 3.5e7, .04, .125)}
    fig, axes = plt.subplots(1, 2, figsize = (17.2, 6.8), sharey = True)
    lstm = [(lstm_params(n), end(width_curve("lstm", n)), LSTM1_COLOR, "^", True, f"LSTM {n}") for n in [1000] + LSTM_WIDTHS]
    lstm = [q for q in lstm if q[1] is not None]
    w4 = end(width_curve(4, 4000))
    shared = lstm + ([(ff_params(4, 4000), w4, WIDE_COLOR, "*", True, "d4, width 4000")] if w4 is not None else [])
    for res_, mk, col, lab in ((False, "s", ALL_COLOR, "d30 all frozen, w4000"), (True, "D", WIDE_COLOR, "d30 all frozen, w4000, residual")):
        c30 = load_curve(os.path.join(OUT, "widths", f"d30_N4000_p1_lr0.001_frz1-29{'_res' if res_ else ''}_rep0_curve.npz"))
        if end(c30) is not None:                                 # depth 30, width 4000: in both panels, like the other width-4000 run
            shared.append((ff_params(30, 4000, 29), end(c30), col, mk, False, lab))
    layout = []
    for ax, n in zip(axes, (1000, 2000)):
        pts = []                                                 # (x, y, color, marker, filled, label)

        def family(xy, col):
            xy = sorted((x, y) for x, y in xy if y is not None)
            if len(xy) > 1:
                ax.plot(*zip(*xy), color = col, lw = 1, alpha = .35, zorder = 1)

        full = {d: end(width_curve(d, n)) for d in DEPTHS}
        pts += [(ff_params(d, n), y, FULL_COLOR, "o", True, f"d{d}") for d, y in full.items() if y is not None]
        family([(ff_params(d, n), y) for d, y in full.items()], FULL_COLOR)
        for depth in (2, 4, 9):
            D = depth - 1
            for side, col, mk, sets in (("in", IN_COLOR, "s", lambda k: list(range(1, k + 1))),
                                        ("out", OUT_COLOR, "D", lambda k: list(range(D - k + 1, D + 1)))):
                xy = [(ff_params(depth, n), full.get(depth))]
                for k in range(1, D + 1):
                    y = end(width_curve(depth, n, sets(k)))
                    xy.append((ff_params(depth, n, k), y))
                    if y is None or (k == D and side == "out"):  # the all-frozen net is one point for both sides
                        continue
                    pts.append((ff_params(depth, n, k), y, ALL_COLOR if k == D else col, "s" if k == D else mk, True,
                                f"d{depth} all" if k == D else f"d{depth} {side}{k}"))
                family(xy, col)
        family([(q[0], q[1]) for q in lstm], LSTM1_COLOR)
        pts += shared
        scatter_markers(ax, pts)
        seed1 = []
        if n == 1000:
            seed1 = [(ff_params(d, 1000), end(load_curve(os.path.join(OUT, f"d{d}_p1_lr0.001_rep1_curve.npz")))) for d in DEPTHS]
            seed1 = [q for q in seed1 if q[1] is not None]
            ax.scatter(*zip(*seed1), s = 12, facecolor = "none", edgecolor = FULL_COLOR, lw = .8, alpha = .7, zorder = 2)
        zoom = ZOOMS[n]
        ins, inside = zoom_inset(ax, pts, zoom, fs)
        if n == 2000:
            ins.set_xticks([1e7, 2e7, 3e7], ["10M", "20M", "30M"])
        z1 = [q for q in seed1 if zoom[0] <= q[0] <= zoom[1] and zoom[2] <= q[1] <= zoom[3]]
        if z1:
            ins.scatter(*zip(*z1), s = 16, facecolor = "none", edgecolor = FULL_COLOR, lw = .8, alpha = .7, zorder = 2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(2e4, 8e7)
        kn = baselines()["kl_naive"]
        if kn is not None:
            ax.axhline(kn, color = NAIVE_COLOR, ls = "--", lw = 1, zorder = 0)
            ax.text(2.4e4, kn, "naive Bayes", color = NAIVE_COLOR, fontsize = fs["tick"] - 2, va = "bottom", ha = "left")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}"[-1] in "2357" and v < 1 else ""))
        ax.set_xlabel("trainable parameters")
        ax.set_title(f"({'ab'[n // 2000]}) FF width {n}: all finished runs", fontsize = fs["label"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        layout.append((ax, ins, inside, pts))
    axes[0].set_ylabel(KL("joint", "net") + " at 20k episodes, mean of the last 3 checkpoints (joint Bayes: 0)")
    H = lambda **k: plt.Line2D([], [], lw = 0, **{"ms": 6, **k})
    handles = [H(marker = "o", color = FULL_COLOR, label = "FF fully trained, d = depth"),
               H(marker = "s", color = IN_COLOR, label = "frozen from the input end:\nin k = first k hidden layers"),
               H(marker = "D", color = OUT_COLOR, label = "frozen from the output end:\nout k = last k"),
               H(marker = "s", color = ALL_COLOR, label = "all hidden layers frozen"),
               H(marker = "^", color = LSTM1_COLOR, label = "LSTM, by width (both panels)"),
               H(marker = "*", color = WIDE_COLOR, ms = 9, label = "depth 4, width 4000, lr 3e-4\n(both panels)"),
               H(marker = "s", color = ALL_COLOR, mfc = "white", label = "depth 30, width 4000, all 29 frozen\n(open; both panels)"),
               H(marker = "D", color = WIDE_COLOR, mfc = "white", label = "the same with residual skips"),
               H(marker = "o", color = FULL_COLOR, mfc = "none", ms = 4, label = "width 1000, seed 1 (unlabelled)")]
    fig.legend(handles = handles, frameon = False, fontsize = fs["tick"] - 2, loc = "center left", bbox_to_anchor = (.86, .5))
    fig.tight_layout(rect = (0, 0, .86, 1), w_pad = 2)
    for ax, ins, inside, pts in layout:
        scatter_labels(fig, ax, [q for q in pts if not inside(q)], pts, fs)
        scatter_labels(fig, ins, [q for q in pts if inside(q)], [q for q in pts if inside(q)], fs, 1.3)
    fig.savefig(fig_path("ff_deep_power1_scatter.png"), dpi = 150)
    print(fig_path("ff_deep_power1_scatter.png"))


def widths_figure(at_ep = None):
    """Fully trained FF, depths 1-9 at widths 1000 and 2000 (and depth 4 at width 4000, lr 3e-4), and the power-1 LSTMs
    (1.3M, 2.9M, 8.05M): (a) KL(joint || net) and (b) accuracy at 20k episodes (mean of the last 3 checkpoints;
    finished runs only) against trainable parameters. One line per FF width through depths 1-9 (labelled at depth 1
    and 9; seed 0), with the width-1000 second seed as small open markers; the LSTMs as one line."""
    from main.REPLICATION.trainings.train_ff_deep import NEW_WIDTHS, LSTM_WIDTHS, ff_params, lstm_params
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    end = lambda c, key: widths_value(c, key, at_ep)
    when = "20k episodes" if at_ep is None else f"{at_ep / 1000:g}k episodes (runs in progress)"
    fig, axes = plt.subplots(1, 2, figsize = (10.4, 4.0))
    WIDTH_COLORS = {1000: "#6baed6", 2000: "#08306b"}                 # only these two widths exist
    for ax, key, title in ((axes[0], "kl_joint", "(a) " + KL("joint", "net")), (axes[1], "acc", "(b) accuracy")):
        for n in [1000] + NEW_WIDTHS:
            xy = [(ff_params(d, n), end(width_curve(d, n), key), d) for d in DEPTHS]
            xy = [q for q in xy if q[1] is not None]
            if xy:
                ax.plot([q[0] for q in xy], [q[1] for q in xy], color = WIDTH_COLORS[n], marker = "o", ms = 3.5, lw = 1.3,
                        label = f"FF, width {n}")
                for x, y, d in (xy[0], xy[-1]):
                    ax.annotate(f"d{d}", (x, y), xytext = (4, 4), textcoords = "offset points", fontsize = fs["tick"] - 2.5,
                                color = WIDTH_COLORS[n])
            if n == 1000:                                        # the second seed of the width-1000 depths
                r1 = [(ff_params(d, n), end(load_curve(os.path.join(OUT, f"d{d}_p1_lr0.001_rep1_curve.npz")), key)) for d in DEPTHS]
                r1 = [q for q in r1 if q[1] is not None]
                ax.scatter(*zip(*r1), s = 14, facecolor = "white", edgecolor = WIDTH_COLORS[n], lw = .9, zorder = 3,
                           label = "FF, width 1000, seed 1")
        w4 = end(width_curve(4, 4000), key)
        if w4 is not None:
            ax.scatter(ff_params(4, 4000), w4, marker = "*", s = 60, color = WIDE_COLOR, zorder = 3, label = "FF depth 4, width 4000 (lr 3e-4)")
        xy = sorted((lstm_params(n), end(width_curve("lstm", n), key)) for n in LSTM_WIDTHS + [1000])   # by size
        xy = [q for q in xy if q[1] is not None]
        if xy:
            ax.plot(*zip(*xy), color = LSTM1_COLOR, marker = "^", ms = 4.5, lw = 1.4, label = "LSTM")
        ax.set_xscale("log")
        ax.set_xlabel("trainable parameters")
        ax.set_title(f"{title}, {when}", fontsize = fs["label"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}"[-1] in "2357" and v < 1 else ""))
    axes[0].set_ylabel(KL("joint", "net") + ", final step (joint Bayes: 0)")
    axes[1].set_ylabel("final-step test accuracy")
    base_ref = baselines()
    ref_lines(axes[0], [(base_ref["kl_naive"], "naive Bayes", NAIVE_COLOR, "--")], fs)
    ref_lines(axes[1], [(base_ref["acc_naive"], "naive Bayes", NAIVE_COLOR, "--"), (base_ref["acc_joint"], "joint Bayes", JOINT_COLOR, ":")], fs)
    axes[1].legend(frameon = False, fontsize = fs["tick"] - 1.5, loc = "lower right")
    fig.tight_layout(w_pad = 2)
    fig.savefig(fig_path("ff_deep_widths.png"), dpi = 150)
    print(fig_path("ff_deep_widths.png"))


def widths_frozen_figure():
    """Frozen versus removed layers at widths 1000 and 2000, at 20k episodes (mean of the last 3 checkpoints): top
    KL(joint || net), bottom accuracy, against m = N x N layers that train. At a given m every network has the same
    trainable parameters as the fully trained depth m + 1 (drawn too); depth 9 and depth 4 frozen from either end start
    at their fully trained depth (m = depth - 1) and end with every N x N layer frozen (m = 0). References: the 8M LSTM
    and depth 4 at width 4000 (lr 3e-4)."""
    from main.REPLICATION.trainings.train_ff_deep import NEW_WIDTHS
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    widths = [1000] + NEW_WIDTHS
    fig, axes = plt.subplots(2, len(widths), figsize = (4.6 * len(widths) + .6, 6.8), sharey = "row")
    refs = [(label, c, lc) for label, c, lc in lstm_runs()] + [("d4, width 4000", width_curve(4, 4000), WIDE_COLOR)]
    base_ref = baselines()
    for row, (key, ylabel) in enumerate((("kl_joint", KL("joint", "net") + ", final step, 20k episodes"),
                                         ("acc", "final-step test accuracy, 20k episodes"))):
        end = lambda c: widths_value(c, key)
        for col, n in enumerate(widths):
            ax = axes[row, col]
            lines = [("fully trained, depth m + 1", RAMP(.67), "-", "o", [(d - 1, width_curve(d, n)) for d in DEPTHS])]
            for depth, ls in ((9, "-"), (4, "--")):
                D = depth - 1
                for side, c_, mk, sets in (("output", OUT_COLOR, "D", lambda k: list(range(D - k + 1, D + 1))),
                                           ("input", IN_COLOR, "s", lambda k: list(range(1, k + 1)))):
                    pts = [(D - k, width_curve(depth, n) if k == 0 else width_curve(depth, n, sets(k))) for k in range(D + 1)]
                    lines.append((f"depth {depth}, frozen from the {side} end", c_, ls, mk, pts))
            lines.append(("depth 2, its one N x N layer frozen", ALL_COLOR, ":", "s", [(1, width_curve(2, n)), (0, width_curve(2, n, [1]))]))
            for label, c_, ls, mk, pts in lines:
                xy = [(m, end(c)) for m, c in pts if end(c) is not None]
                if xy:
                    ax.plot(*zip(*xy), color = c_, ls = ls, marker = mk, ms = 4 if ls == "-" else 3.5, lw = 1.3 if ls == "-" else 1,
                            mfc = c_ if ls == "-" else "white", label = label)
            if key == "kl_joint":
                ax.set_yscale("log")
            ref_lines(ax, [(end(c), lab, rc, "-") for lab, c, rc in refs]
                      + ([(base_ref["kl_naive"], "naive Bayes", NAIVE_COLOR, "--")] if key == "kl_joint" else
                         [(base_ref["acc_naive"], "naive Bayes", NAIVE_COLOR, "--"), (base_ref["acc_joint"], "joint Bayes", JOINT_COLOR, ":")]), fs)
            ax.set_xticks(range(9))
            ax.set_xlim(-.3, 8.4)
            ax.set_xlabel("m: N x N layers that train")
            ax.set_title(f"({'ab'[col] if row == 0 else 'cd'[col]}) width {n}", fontsize = fs["label"])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if key == "kl_joint":
                ax.set_yscale("log")
        axes[row, 0].set_ylabel(ylabel)
    axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0, 0].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}"[-1] in "2357" and v < 1 else ""))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = "lower center", ncol = 3, frameon = False, fontsize = fs["tick"] - 1.5)
    fig.tight_layout(w_pad = 1, h_pad = 1.2, rect = (0, .08, 1, 1))
    fig.savefig(fig_path("ff_deep_widths_frozen.png"), dpi = 150)
    print(fig_path("ff_deep_widths_frozen.png"))


DISAGREE = os.path.join(OUT, "eval_disagreement.json")


def disagreement_nets():
    """(label, family, load config) of every net in the joint-vs-naive test. Families: lstm (trained), reservoir,
    ff (fully trained), frozen, residual. FF and power-1 LSTMs are seed 0 (one world); the sanity_default_reps
    LSTMs and reservoirs (loss power 0.5) carry their own worlds."""
    from main.REPLICATION.trainings.train_ff_deep import BASE
    def ff(stem, depth, hid = 1000, **kw):
        return dict(BASE, training = False, episodes = 1, checkpoint_every = 1, batch_num = 8000, readin_depth = depth, hid_dim = hid,
                    classifier_LR = 1e-3, load_env = "experimental/ff_deep/" + stem, **kw)
    def rnn(stem, hid, reservoir, **kw):
        return dict(mode = "SANITY", cuda = 0, episodes = 1, checkpoint_every = 1, realization_num = 10, obs_num = 5, show_plots = False,
                    state_num = 500, learn_embeddings = False, classifier_LR = .001, ctx_num = 2, training = False,
                    batch_num = 8000 if hid <= 1000 else 1000, step_num = 30, gpu_inference = True, hid_dim = hid, reservoir = reservoir,
                    load_env = stem, **kw)
    W = "widths/"
    return ([("LSTM 8.0M (power 0.5)", "lstm", rnn("sanity_default_reps/fully_trained_ctx_2_rep0", 1000, False)),
             ("LSTM 0.33M (power 0.5)", "lstm", rnn("sanity_default_reps/fully_trained_ctx_2_200N_rep0", 200, False)),
             ("LSTM 8.0M", "lstm", rnn("experimental/ff_deep/lstm_p1_lr0.001_rep0", 1000, False))]
            + [(f"LSTM {m}", "lstm", rnn(f"experimental/ff_deep/{W}lstm_N{n}_p1_lr0.001_rep0", n, False)) for n, m in ((600, "2.9M"), (400, "1.3M"), (200, "0.33M"))]
            + [(f"reservoir 1000, seed {r}", "reservoir", rnn(f"sanity_default_reps/reservoir_ctx_2_rep{r}", 1000, True)) for r in range(5)]
            + [("reservoir 10000", "reservoir", rnn("sanity_default_reps/reservoir_ctx_2_10000N_rep0", 10000, True)),
               ("reservoir, joint output", "reservoir", rnn("sanity/joint_reservoir_ctx_2", 1000, True, output_joint = True))]
            + [(f"reservoir {n} (power 1)", "reservoir", rnn(f"experimental/ff_deep/{reservoir_stem(n)}", n, True)) for n in RESERVOIR_WIDTHS
               if os.path.exists(os.path.join(OUT, reservoir_stem(n) + "_summary.json"))]
            + [(f"FF depth {d}", "ff", ff(f"d{d}_p1_lr0.001_rep0", d)) for d in (1, 2, 3, 5, 9)]
            + [("FF d2, layer frozen", "frozen", ff(W + "d2_p1_lr0.001_frz1_rep0", 2)),
               ("FF d4, all 3 frozen", "frozen", ff("d4_variants/d4_p1_lr0.001_frz123_rep0", 4)),
               ("FF d9, all 8 frozen", "frozen", ff("d9_frozen/d9_p1_lr0.001_frz12345678_rep0", 9)),
               ("FF d9, first 4 frozen", "frozen", ff("d9_frozen/d9_p1_lr0.001_frz1234_rep0", 9)),
               ("FF d9, first 6 frozen", "frozen", ff("d9_frozen/d9_p1_lr0.001_frz123456_rep0", 9)),
               ("FF d9, last 3 frozen", "frozen", ff("d9_frozen/d9_p1_lr0.001_frz678_rep0", 9)),
               ("FF d9, last 6 frozen", "frozen", ff("d9_frozen/d9_p1_lr0.001_frz345678_rep0", 9)),
               ("FF d30, all 29 frozen, w4000", "frozen", ff(W + "d30_N4000_p1_lr0.001_frz1-29_rep0", 30, 4000)),
               ("FF d30, all 29 frozen, w4000, residual", "residual", ff(W + "d30_N4000_p1_lr0.001_frz1-29_res_rep0", 30, 4000, readin_residual = True))]
            + frozen_width_nets(ff))


FROZEN_WIDTH_SERIES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]   # 10-50: train_ff_deep.py small_widths
RESERVOIR_WIDTHS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]    # loss power 1 and the FF nets' data: 20k episodes x
                                        # batch 8000, seed 0 (train_ff_deep.py matched_a; reservoirs_wide from N=1000 up)


LSTM_STEMS = {**{n: f"widths/lstm_N{n}_p1_lr0.001_rep0" for n in (10, 20, 50, 200, 400, 600)},   # fully trained, power 1,
              1000: "lstm_p1_lr0.001_rep0"}                                                     # the FF nets' data


def tff1_stem(n):
    """Depth 30 with all 29 N x N layers frozen at nn.Linear's default draws x sqrt(6) (train_ff_deep.py tff_series ...
    tff1: main/model/time_ff.py with input at layer 1 only, fp64-identical to the standard depth-30 FF with those weights).
    The time-injected nets (tff, tffs) left the analysis (John, 2026-10-16); tffs is back in its own panel (tffs_stem)."""
    return f"widths/tff1{'' if n == 1000 else f'_N{n}'}_p1_lr0.001_rep0"


def tffs_stem(n):
    """The shared-readin net (train_ff_deep.py tff_series ... tffs, time_ff.py shared): tff1's 29 frozen layers (the same
    seeds), with input at every layer through one shared readin, the n-th layer from the readout getting the mean of the
    first n observations with Z; 36 N + 20 trainable parameters, as depth 1 and tff1. Widths 100-5000 (2026-10-15/16),
    back in ff_deep_frozen_widths_all_depths.png (John, 2026-10-22; its own figure until 10-25). Width 10000, 60k episodes
    (TRAINED_ON): at lr 1e-3 its read-out overshot (10-28 run killed at 33.5k: *_killed_ep33500; with a 100-update lr
    warm-up, later: *_ramp100_*_killed_ep1000, 10-29), so lr 2.5e-4 with that warm-up (John 2026-10-29; TFFS_LR)."""
    return f"widths/tffs{'' if n == 1000 else f'_N{n}'}_p1_lr{TFFS_LR.get(n, 1e-3):g}_rep0"


TFFS_LR = {10000: 2.5e-4}       # the shared-readin widths not at lr 1e-3 (tffs_stem)


def lr_text(lr):
    """2.5e-4, not 0.00025 or 2.5e-04"""
    return ("%.1e" % lr).replace("e-0", "e-").replace(".0e", "e")



def reservoir_stem(n):
    """N=10000 trained at lr 5e-4 (John 2026-10-18): at 1e-3 its readout's Adam steps, ~lr x N on logits summed over 30
    steps, overshot in the first ~500 updates and pinned some values' logits near -100 (final-step KL ~9; kept on disk as
    lstm_N10000_p1_lr0.001_reservoir_rep0)."""
    lr = "0.0005" if n == 10000 else "0.001"
    return f"widths/lstm{'' if n == 1000 else f'_N{n}'}_p1_lr{lr}_reservoir_rep0"


def frozen_width_stem(d, n):
    """Run stem of depth d at width n with every N x N layer frozen (train_ff_deep.py frozen_widths, reservoir_compare)."""
    if (d, n) == (1, 1000):
        return "d1_p1_lr0.001_rep0"
    if (d, n) == (4, 1000):
        return "d4_variants/d4_p1_lr0.001_frz123_rep0"
    tag = "" if d == 1 else "_frz" + "".join(map(str, range(1, d)))
    return f"widths/d{d}{'' if n == 1000 else f'_N{n}'}_p1_lr0.001{tag}_rep0"


def frozen_width_nets(ff):
    """The frozen_widths series as (label, family, config), family 'frozen_width'; only runs that have finished."""
    out = []
    for d in (1, 2, 3, 4):
        for n in FROZEN_WIDTH_SERIES:
            stem = frozen_width_stem(d, n)
            if os.path.exists(os.path.join(OUT, stem + "_summary.json")):
                out.append((f"width series: depth {d}, width {n}", "frozen_width", ff(stem, d, n)))
    return out


def collect_disagreement(only_missing = False):
    """Final-step comparison of every net in disagreement_nets() with joint and naive Bayes, on fresh test batches
    (~16k trials per net in its own world) -> DISAGREE. On trials where the joint and naive answers (argmax) differ:
    the share of the net's answers that are joint's or naive's. On all trials: accuracy (argmax against the true value)
    and confidence (mean probability of the net's own top answer) of the net, joint and naive; KL(joint || net),
    KL(naive || net), and the same for naive, joint and a uniform belief as references (clipped at 1e-12)."""
    from main.CognitiveGridworld import CognitiveGridworld

    def kl(p, q):
        p, q = np.clip(p, 1e-12, 1), np.clip(q, 1e-12, 1)
        return float((p * np.log(p / q)).sum(-1).mean())

    res = json.load(open(DISAGREE)) if only_missing and os.path.exists(DISAGREE) else {}
    for label, family, cfg in disagreement_nets():
        if label in res:
            continue
        rounds, rows = 8 if cfg["batch_num"] == 1000 else 2, []
        for _ in range(rounds):
            g = {}
            def lg(self):
                g.update({k: np.asarray(getattr(self, a), float)[:, -1] for k, a in
                          (("q", "model_goal_belief"), ("P", "joint_goal_belief"), ("N", "naive_goal_belief"))})
                g["v"] = np.asarray(self.goal_value)
                return {}
            CognitiveGridworld(**cfg, external_logger = lg)
            q, P, N, v = g["q"], g["P"], g["N"], g["v"]
            a, j, n = q.argmax(-1), P.argmax(-1), N.argmax(-1)
            d = j != n
            U = np.full_like(q, 1 / q.shape[-1])
            rows.append(dict(n_dis = d.sum(), pick_joint = (a[d] == j[d]).sum(), pick_naive = (a[d] == n[d]).sum(), n = len(a),
                             acc = (a == v).mean(), conf = q.max(-1).mean(), acc_joint = (j == v).mean(), conf_joint = P.max(-1).mean(),
                             acc_naive = (n == v).mean(), conf_naive = N.max(-1).mean(),
                             kl_joint = kl(P, q), kl_naive = kl(N, q), kl_joint_naive = kl(P, N), kl_naive_joint = kl(N, P),
                             kl_joint_uniform = kl(P, U), kl_naive_uniform = kl(N, U)))
            torch.cuda.empty_cache()
        r = {k: float(np.mean([x[k] for x in rows])) for k in rows[0] if k not in ("n_dis", "pick_joint", "pick_naive", "n")}
        n_dis = sum(x["n_dis"] for x in rows)
        r.update(family = family, joint = sum(x["pick_joint"] for x in rows) / n_dis, naive = sum(x["pick_naive"] for x in rows) / n_dis,
                 disagree_frac = n_dis / sum(x["n"] for x in rows))
        res[label] = r
        print(label, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush = True)
    json.dump(res, open(DISAGREE, "w"), indent = 1)
    print("saved", DISAGREE)


def place_labels(fig, ax, pts, fs, size = 1., avoid = ()):
    """Label points [(x, y, colour, marker, filled, label)] so that no label overlaps another label or another point:
    each label tries positions around its point, nearest first, then further out with a leader line. Call after the
    layout is final."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fsz = (fs["tick"] - 3) * size ** .5
    merged = []                                                  # points within a few pixels share one label
    for q in pts:                                                # (points with an empty label are obstacles only)
        sq = ax.transData.transform((q[0], q[1]))
        for k, m in enumerate(merged):
            if q[5] and m[5] and q[2] == m[2] and np.hypot(*(ax.transData.transform((m[0], m[1])) - sq)) < 4 * fig.dpi / 72:
                merged[k] = m[:5] + (m[5] + " / " + q[5],)
                break
        else:
            merged.append(tuple(q))
    pts = merged
    screen = [ax.transData.transform((q[0], q[1])) for q in pts]
    marker_r = 6 * fig.dpi / 72
    placed, frame = [a_.get_window_extent(r) for a_ in avoid], ax.get_window_extent(r)   # avoid: e.g. an inset
    near = [(6, 0, "left", "center"), (6, 5, "left", "bottom"), (6, -5, "left", "top"), (-6, 0, "right", "center"),
            (-6, 5, "right", "bottom"), (-6, -5, "right", "top"), (0, 8, "center", "bottom"), (0, -8, "center", "top")]
    far = [(dx * k, dy * k, ha, va) for k in (2.5, 4, 6) for dx, dy, ha, va in near]
    diag = [(5, 7, "left", "bottom"), (5, -7, "left", "top"), (-5, 7, "right", "bottom"), (-5, -7, "right", "top")]
    farther = [(dx * k, dy * k, ha, va) for k in (2.5, 4, 6, 9, 13) for dx, dy, ha, va in diag] + \
              [(dx * k, dy * k, ha, va) for k in (9, 13) for dx, dy, ha, va in near]           # tried only after the original set

    def overlap(bb, i):
        """How badly a label box collides: area over placed boxes, plus points it covers, plus area outside the axes."""
        area = sum(max(0, min(bb.x1, p.x1) - max(bb.x0, p.x0)) * max(0, min(bb.y1, p.y1) - max(bb.y0, p.y0)) for p in placed)
        hits = sum(bb.x0 - marker_r < sx < bb.x1 + marker_r and bb.y0 - marker_r < sy < bb.y1 + marker_r
                   for j, (sx, sy) in enumerate(screen) if j != i)
        outside = bb.width * bb.height - max(0, min(bb.x1, frame.x1) - max(bb.x0, frame.x0)) * max(0, min(bb.y1, frame.y1) - max(bb.y0, frame.y0))
        return area + 400 * hits + 10 * outside

    for i in sorted(range(len(pts)), key = lambda i: -pts[i][1]):
        x, y, col, _, _, label = pts[i]
        if not label:
            continue
        best = None
        for dx, dy, ha, va in near + far + farther:
            lead = abs(dx) > 8 or abs(dy) > 10
            t = ax.annotate(label, (x, y), xytext = (dx, dy), textcoords = "offset points", ha = ha, va = va, fontsize = fsz, color = col,
                            arrowprops = dict(arrowstyle = "-", color = col, lw = .5, shrinkA = 0, shrinkB = 4) if lead else None)
            bb = t.get_window_extent(r).expanded(1.03, 1.15)
            score = overlap(bb, i)
            if score == 0:
                placed.append(bb)
                break
            if best is None or score < best[0]:
                if best is not None:
                    best[1].remove()
                best = (score, t, bb)
            else:
                t.remove()
        else:                                                    # nowhere free: the least-overlapping candidate
            placed.append(best[2])
            continue
        if best is not None:
            best[1].remove()


def disagreement_figure():
    """Every net in disagreement_nets() against joint and naive Bayes, final step, from DISAGREE (--collect). (a) Where
    joint and naive Bayes disagree (about half the trials): share picking naive's answer against share picking joint's;
    the rest pick neither; below the diagonal leans naive. (b) KL(joint || net) against KL(naive || net) on the goal
    marginal, log axes; joint and naive Bayes are 0 from themselves, so they sit on the axes as edge markers; uniform is
    the input-blind reference. (c) Final-step accuracy against confidence (mean probability of the net's own top answer):
    on the diagonal = calibrated, below = overconfident. The five width-1000 reservoir seeds are one point with their
    range. Reference values are averaged over the nets' worlds."""
    if not os.path.exists(DISAGREE):
        print("no", DISAGREE, "(run with --collect)")
        return
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    labels = {label for label, _, _ in disagreement_nets()}               # entries for nets no longer in the test are ignored
    res = {k: v for k, v in json.load(open(DISAGREE)).items() if k in labels}
    style = {"lstm": (LSTM1_COLOR, "^"), "reservoir": ("#b2182b", "v"), "ff": (RAMP(.67), "o"), "frozen": (ALL_COLOR, "s"),
             "residual": (WIDE_COLOR, "D")}
    ref = {k: np.mean([v[k] for v in res.values() if k in v]) for k in ("kl_naive_joint", "kl_joint_naive", "kl_joint_uniform", "kl_naive_uniform",
                                                                "acc_joint", "conf_joint", "acc_naive", "conf_naive")}
    seeds = [v for k, v in res.items() if k.startswith("reservoir 1000, seed")]
    nets = []                                                    # (label, family, record, seed records or None)
    for label, v in res.items():
        if v["family"] == "frozen_width":                       # drawn in frozen_widths_figure
            continue
        if label.startswith("reservoir 1000, seed"):
            if label.endswith("seed 0"):
                nets.append((f"reservoir 1000 ({len(seeds)} seeds)", "reservoir", {k: np.mean([s_[k] for s_ in seeds]) for k in
                             ("joint", "naive", "kl_joint", "kl_naive", "acc", "conf")}, seeds))
            continue
        nets.append((label, v["family"], v, None))

    fig, axes = plt.subplots(1, 3, figsize = (20.5, 7.4))
    panels = (("naive", "joint", "share picking naive Bayes's answer", "share picking joint Bayes's answer",
               "(a) trials where joint and naive Bayes disagree"),
              ("kl_joint", "kl_naive", KL("joint", "net"), KL("naive", "net"), "(b) distance of the marginal belief from each Bayes"),
              ("conf", "acc", "confidence: mean probability of the top answer", "accuracy", "(c) final-step accuracy against confidence"))
    for ax, (xk, yk, xlabel, ylabel, title) in zip(axes, panels):
        pts = []
        for label, family, r, group in nets:
            col, mk = style[family]
            if group:
                xs, ys = [g_[xk] for g_ in group], [g_[yk] for g_ in group]
                ax.errorbar(r[xk], r[yk], xerr = [[r[xk] - min(xs)], [max(xs) - r[xk]]], yerr = [[r[yk] - min(ys)], [max(ys) - r[yk]]],
                            color = col, lw = .8, zorder = 2)
            pts.append((r[xk], r[yk], col, mk, True, label))
        scatter_markers(ax, pts, 1.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize = fs["label"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        refs = []
        if xk == "naive":
            ax.plot([0, .62], [0, .62], color = "#b4b2a9", lw = 1, ls = ":", zorder = 0)
            ax.text(.6, .57, "no lean", color = "#8a8984", fontsize = fs["tick"] - 2, ha = "right", va = "top", rotation = 45)
            refs = [(0, 1, "#1f1f1e", "*", True, "joint Bayes"), (.1, .1, "#1f1f1e", "*", True, "random guessing")]
            ax.set_xlim(-.03, .62); ax.set_ylim(-.03, 1.05)
            ax.annotate("naive Bayes at (1, 0)", xy = (.62, .01), xytext = (.45, .06), fontsize = fs["tick"] - 1, color = "#1f1f1e",
                        arrowprops = dict(arrowstyle = "->", color = "#1f1f1e", lw = .8))
        elif xk == "kl_joint":
            ax.set_xscale("log"); ax.set_yscale("log")
            x0, x1, y0, y1 = .008, 6, .5, 7
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            refs = [(ref["kl_joint_uniform"], ref["kl_naive_uniform"], "#1f1f1e", "*", True, "uniform (input-blind)")]
            ax.scatter(x0, ref["kl_naive_joint"], marker = "<", s = 90, color = "#1f1f1e", clip_on = False, zorder = 4)
            ax.annotate("joint Bayes (0 from joint)", (x0, ref["kl_naive_joint"]), xytext = (8, 6), textcoords = "offset points",
                        fontsize = fs["tick"] - 1)
            ax.scatter(ref["kl_joint_naive"], y0, marker = "v", s = 90, color = "#1f1f1e", clip_on = False, zorder = 4)
            ax.annotate("naive Bayes (0 from naive)", (ref["kl_joint_naive"], y0), xytext = (-6, 8), textcoords = "offset points",
                        ha = "right", fontsize = fs["tick"] - 1)
            for a_ in (ax.xaxis, ax.yaxis):
                a_.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                a_.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}"[0] in "25" else ""))
        else:
            ax.plot([0, .85], [0, .85], color = "#b4b2a9", lw = 1, ls = ":", zorder = 0)
            ax.text(.2, .23, "calibrated", color = "#8a8984", fontsize = fs["tick"] - 2, ha = "left", va = "bottom", rotation = 45)
            refs = [(ref["conf_joint"], ref["acc_joint"], "#1f1f1e", "*", True, "joint Bayes"),
                    (ref["conf_naive"], ref["acc_naive"], "#1f1f1e", "*", True, "naive Bayes"), (.1, .1, "#1f1f1e", "*", True, "uniform")]
            ax.set_xlim(0, .85); ax.set_ylim(0, .85)
        for x, y, *_ in refs:
            ax.scatter(x, y, marker = "*", s = 110, color = "#1f1f1e", zorder = 3)
        ax._place = pts + refs
        if xk == "conf":                                         # the trained nets crowd near joint Bayes: magnify them
            box = (.715, .80, .675, .785)
            inside = lambda q: box[0] <= q[0] <= box[1] and box[2] <= q[1] <= box[3]
            # the inset goes where it hides no point: lower right if free, else upper left (axes 0-.85, .02 margin)
            covers = lambda b, q: (b[0] * .85 - .02 <= q[0] <= (b[0] + b[2]) * .85 + .02
                                   and b[1] * .85 - .02 <= q[1] <= (b[1] + b[3]) * .85 + .02)
            spot = min(([.5, .03, .48, .33], [.03, .55, .45, .33]), key = lambda b: sum(covers(b, q) for q in pts + refs))
            if any(covers(spot, q) for q in pts + refs):
                print("warning: the panel (c) inset hides", [q[5] for q in pts + refs if covers(spot, q)])
            ins = ax.inset_axes(spot)
            ins.plot([0, 1], [0, 1], color = "#b4b2a9", lw = 1, ls = ":", zorder = 0)
            zoomed = [q for q in pts + refs if inside(q)]
            for q in zoomed:
                if q[3] == "*":
                    ins.scatter(q[0], q[1], marker = "*", s = 140, color = "#1f1f1e", zorder = 3)
            scatter_markers(ins, [q for q in zoomed if q[3] != "*"], 1.5)
            ins.set_xlim(box[0], box[1]); ins.set_ylim(box[2], box[3])
            ins.tick_params(labelsize = fs["tick"] - 3)
            ax.indicate_inset_zoom(ins, edgecolor = "#888780")
            ax._place = [q for q in pts + refs if not inside(q)]
            ax._inset = (ins, zoomed)
    H = lambda col, mk, lab: plt.Line2D([], [], lw = 0, marker = mk, color = col, ms = 6, label = lab)
    axes[0].legend(handles = [H(*style["lstm"], "LSTM, trained"), H(*style["reservoir"], "reservoir (LSTM frozen; power 0.5 unless labelled power 1)"),
                              H(*style["ff"], "FF, fully trained"), H(*style["frozen"], "FF, layers frozen"),
                              H(*style["residual"], "FF, frozen with residual skips")], frameon = False, fontsize = fs["tick"] - 1.5,
                   loc = "upper right")
    fig.tight_layout(w_pad = 2.5)
    for ax in axes:
        place_labels(fig, ax, ax._place, fs, 1.1, avoid = [ax._inset[0]] if hasattr(ax, "_inset") else ())
        if hasattr(ax, "_inset"):
            place_labels(fig, *ax._inset, fs, 1.1)
    fig.savefig(fig_path("ff_deep_disagreement.png"), dpi = 150)
    print(fig_path("ff_deep_disagreement.png"))


CONTINUED = os.path.join(OUT, "widths", "continued")


def continued_curve(stem):
    """the run of `stem` continued from its saved weights past 20k episodes (train_ff_deep.py continue_20k,
    continue_tff1_ramp; 2026-10-21/22): (episodes in thousands, accuracy, finished, joint Bayes's accuracy on the same test
    batches) for its checkpoints past 20k, or None"""
    base = os.path.join(CONTINUED, os.path.basename(stem))
    c = load_curve(base + "_curve.npz")
    if c is None or not (c["ep"] > 20000).any():
        return None
    m = c["ep"] > 20000
    return c["ep"][m] / 1000, c["acc"][m], os.path.exists(base + "_summary.json"), c["joint"][m]


TRAINED_ON = {tffs_stem(10000): 60000}   # runs trained past 20k in one go, with their planned episodes (John 2026-10-27:
                                         # the shared readin at width 10000 for 60k episodes, checkpointed every 5000)


def late_curve(stem):
    """the run of `stem` past 20k episodes: its continuation (continued_curve) or, for a run trained past 20k in one go
    (TRAINED_ON), its own checkpoints past 20k; (episodes in thousands, accuracy, finished, joint Bayes's accuracy), or None"""
    cc = continued_curve(stem)
    if cc is not None:
        return cc
    c = load_curve(os.path.join(OUT, stem + "_curve.npz"))
    if c is None or not (c["ep"] > 20000).any():
        return None
    m = c["ep"] > 20000
    return c["ep"][m] / 1000, c["acc"][m], os.path.exists(os.path.join(OUT, stem + "_summary.json")), c["joint"][m]


SANITY_DIR = os.path.join(DATA, "sanity")
ESN_RUNS = os.path.join(DATA, "experimental", "interaction_bounds", "analyses", "joint_esn_extension", "runs")
ESN_ENTROPY = os.path.join(DATA, "experimental", "esn_learnability", "full_esn_entropy", "continued")
ESN_MLP = os.path.join(DATA, "experimental", "esn_mlp_rescue")
TD = os.path.join(DATA, "experimental", "training_dynamics")     # train_training_dynamics.py runs (another session, 10-29 - 11-01)


def td_curve(run, every_n = 500):
    """a training_dynamics run (a test every 5th episode, which trains nothing; 50,000 episodes = 40,000 training
    episodes), sampled as every other curve in these figures is tested (John 2026-11-01: "different curves in the same
    figure show the same test sampling"): one 8,000-episode test batch per `every_n` training episodes, the one nearest
    each multiple (within 4); (thousands of training episodes before it, its final-step accuracy, joint Bayes's accuracy
    on the same batch). Until 11-01 the tests were averaged in 500-episode blocks (smoother curves, plotted first at the
    blocks' ends, a 2x shift of the first block on log axes, then at their means)."""
    d = os.path.join(TD, run)
    every = json.load(open(os.path.join(d, "run.json")))["cfg"]["checkpoint_every"]
    z = [np.load(f) for f in sorted(glob.glob(os.path.join(d, "test_*.npz")))]
    ep = np.concatenate([q["episode"] for q in z]).astype(int)
    acc, J = (np.concatenate([q[k][:, -1] for q in z]).astype(float) for k in ("acc", "joint_acc"))
    o = np.argsort(ep)
    trained, acc, J = (ep - ep // every)[o], acc[o], J[o]                   # training episodes before each test
    marks = np.arange(0, trained[-1] + every_n / 2, every_n)
    i = np.clip(np.searchsorted(trained, marks), 1, len(trained) - 1)
    i = np.where(np.abs(trained[i - 1] - marks) <= np.abs(trained[i] - marks), i - 1, i)   # the nearest test
    ok = np.abs(trained[i] - marks) <= 4
    return trained[i][ok] / 1000, acc[i][ok], J[i][ok]


TD_N100 = [f"trained_N100_s{s}" for s in range(20261029, 20261034)]


def lstm_n100():
    """the fully trained LSTM at N = 100, the one width the ff_deep LSTMs lack (train_training_dynamics.py, another
    session, 10-31 - 11-01): 5 seeds, SANITY loss (power 0.5), each seed in its own world, 40k updates (a test every 5th
    episode). Per 500-update block, the seeds' mean accuracy and mean joint Bayes accuracy (td_curve); None unless all
    five are DONE. Same-protocol runs at the figure's widths sit +.024 (N=50), +.010 (N=200), +.005 (N=1000) above the
    ff_deep LSTMs at 20k (inventory 2026-11-01). Per 500 training episodes, the mean of the five seeds' single tests."""
    runs = []
    for r in TD_N100:
        p = os.path.join(TD, r, "progress.log")
        if not (os.path.exists(p) and open(p).read().rstrip().endswith("DONE")):
            return None
        runs.append(td_curve(r))
    n = min(len(x) for x, _, _ in runs)
    return runs[0][0][:n], np.mean([a[:n] for _, a, _ in runs], 0), np.mean([J[:n] for _, _, J in runs], 0)


def esn_rescues():
    """The ESNs that beat naive Bayes (John 2026-10-29: "the rescued ESNs (joint output and MLP layers)"; on every
    frozen-widths figure from 11-01): the MLP read-in and read-out, the joint output trained from scratch with the joint
    loss (td_curve("joint_echo_jl_s20261029"): train_training_dynamics.py, another session 10-30, seed 20261029's world,
    joint Bayes .7677 per test batch; its 500-training-episode block means; until 11-01 the joint ESN continued with the
    joint loss from 50k, joint_esn_extension/runs/joint_trained, drawn here; the joint output with the entropy bonus
    was removed 11-01, John: "now that we have the joint-loss"); and as references the unrescued ESNs (John 2026-10-29:
    "include the joint output no rescue cases as well"): Figure 1's marginal ESN and the joint-output ESN (both on their
    own goal loss, 50k; the joint one's goal-loss continuation to 104k, .474, left out; the entropy run's control, the
    same start without the bonus, 10k, dropped by John the same day as redundant with the joint-output ESN). All are a
    frozen LSTM-1000 trained in mode SANITY (batch 8000, Adam 1e-3, symmetric KL
    to joint Bayes at power 0.5, a test every 500 episodes on 8,000 test-state episodes), each in its own world, so its
    gap is to its own world's joint Bayes: paired per test batch where the run logged joint Bayes (log.jsonl
    opt_acc_T), else that world's joint Bayes accuracy (the SANITY ESNs': the mean over their continuations' logs, run in
    the same environments, .7683 marginal / .7679 joint; the MLP rescue's: esn_mlp_rescue/bayes_reference_2026-10-29.json,
    .7664 on 200,000 episodes). Per net: label (curve panel), a_label ((a)), colour, marker, rescue (False for the
    references), segments [(thousands of episodes, accuracy, joint Bayes's accuracy, line style)] and, where its 20k
    value comes from an earlier, undrawn stretch, start (thousands of episodes, accuracy, joint Bayes's accuracy)."""
    def pth(path):
        a = np.asarray(torch.load(path, map_location = "cpu", weights_only = False)["test_acc_through_training"], float)[:, -1]
        return np.arange(len(a)) * .5, a                                   # a test every 500 episodes
    def log(path):
        L = [json.loads(l) for l in open(path) if l.strip()]
        return tuple(np.array([r[k] for r in L], float) for k in ("update", "acc_T", "opt_acc_T"))
    const = lambda a, J: np.full_like(a, J)
    def n_params(path):                                                    # trainable: all but the frozen LSTM
        w = torch.load(path, map_location = "cpu", weights_only = False)["weights"]
        return sum(v.numel() for k, v in w.items() if not k.startswith("LSTM"))
    out = []
    x, a = pth(os.path.join(SANITY_DIR, "reservoir_ctx_2_net.pth"))
    J = float(np.mean(log(os.path.join(ESN_RUNS, "marginal_both_trained", "log.jsonl"))[2]))
    out.append(dict(label = "no rescue, marginal output", col = "#9e9e9e", mk = None, rescue = False, lw = 1.,
                    segments = [(x, a, const(a, J), "-")]))
    xj, aj = pth(os.path.join(SANITY_DIR, "joint_reservoir_ctx_2_net.pth"))   # joint output, its own (goal) loss
    Jj = float(np.mean(np.concatenate([log(os.path.join(ESN_RUNS, r, "log.jsonl"))[2] for r in ("orig_trained", "joint_trained")])))
    out.append(dict(label = "no rescue, joint output", col = "#4d4d4d", mk = None, rescue = False, lw = 1.2,
                    segments = [(xj, aj, const(aj, Jj), "--")]))
    x, a = pth(os.path.join(ESN_MLP, "esn_5in_5out_matched_ctx_2_net.pth"))
    J = json.load(open(os.path.join(ESN_MLP, "bayes_reference_2026-10-29.json")))["joint"]
    out.append(dict(label = "5-layer MLP read-in, read-out", a_label = "ESN + MLP read-in, read-out (at 50k)",
                    col = "#c51b7d", mk = "X", rescue = True, lw = 1.4, segments = [(x, a, const(a, J), "-")],
                    params = n_params(os.path.join(ESN_MLP, "esn_5in_5out_matched_ctx_2_net.pth"))))
    x, a, J = td_curve("joint_echo_jl_s20261029")                          # from scratch, 40k training episodes
    out.append(dict(label = "joint output, joint loss", a_label = "ESN, joint output, joint loss (at 40k)",
                    col = "#00a6a6", mk = "p", rescue = True, lw = 1.4, segments = [(x, a, J, "-")],
                    params = n_params(os.path.join(TD, "joint_echo_jl_s20261029", "final_net.pth"))))
    return out


def frozen_widths_figure(running = (), running_d30 = False, depths = (1, 2), name = "ff_deep_frozen_widths.png", ncols = None,
                         shared_readin = False, running_shared = True, loglog = False, rescues = False, params_panel = True,
                         upto = None, side_legend = True):
    """FF with only readin and readout trained (every N x N layer frozen) at the given depths, against width (train_ff_deep.py
    frozen_widths, reservoir_compare, matched_a, d4_w20000, small_widths; depth 1 has no N x N layer, so it is the
    ordinary depth 1), with depth 30 and beside the reservoirs (an LSTM with only readin and readout trained) and the
    fully trained LSTMs. Every net here used loss power 1 and saw the same data: 20k episodes x batch 8000, seed 0 (one
    world); finished runs only. Reservoirs from N=1000 up ran on the wide path (fp16 frozen weights, deduplicating frozen
    LSTM, micro-batches with accumulated gradients): the same model and updates, fp64-tested; N=10000 at lr 5e-4 (see
    reservoir_stem). (a) final-step accuracy at 20k episodes (mean of the last 3 checkpoints) against width, legends to its
    right. Below, final-step test accuracy through training, line thickness growing with width: one panel per depth, then
    depth 30 with all 29 N x N layers frozen (purple, tff1_stem; its frozen layers are the default draws x sqrt(6),
    Kaiming-uniform, because at the default scale 29 layers erase the input: 3e-11 of layer 30 depends on it at init and
    the Oct 12 deep30 runs sat at chance), then the LSTMs: frozen (reservoirs, red) and fully trained (orange, the best
    case). ncols: curve panels per row (default: all in one row). John 2026-10-21: no KL panel; versions, see
    frozen_widths_figures. running: reservoir widths still training to include anyway, at their latest 3 checkpoints
    (open markers, dotted link; their curve ends where training has got to); running_d30: the same for every depth-30
    width past its first 500 episodes. shared_readin: also the shared-readin net (green, tffs_stem) in (a) and in its
    own panel after depth 30's. running_shared (John 2026-10-29, "show it on the plot"): its running widths too, in its
    curve panel past their first 500 episodes and in (a) once past 20k, when their 20k value is final (before, the
    mean of their latest 3 tests would sit among 20k values; the width-10000 run, 60k episodes at lr 2.5e-4 with a
    100-update lr warm-up, restarted 10-29: see tffs_stem; at lr 1e-3 it overshot as the lr-1e-3 reservoir did). A run
    trained past 20k in one go (TRAINED_ON) enters (a) at its 20k value, its latest value an open marker as a continued
    run's (moved right when the two would overlap); the curve panels then run to 40k or 60k. Reference lines: joint and naive Bayes, chance, and the
    rule linear in the counts fitted by cross-entropy to the joint posterior, as the nets are trained (LINEAR_CE, John
    2026-10-25). It is not a ceiling of linear rules: accuracy-tuned ones reach at least ~.55 (verify_2026-10-25).
    loglog (John 2026-10-29: "a copy ... in log log scaling, or however is the canonical x-y scaling for this kind of
    plot"): every accuracy replaced by its gap to joint Bayes on the same test batch, acc_joint - acc (the reducible error:
    joint Bayes is the optimal decision rule, so the gap goes to 0 for an optimal net; the scaling-law convention plots
    such an excess error on log axes, where a power law in width or in training is a straight line), against log width
    and log training episodes (from the first test after training starts, 500 episodes). (a) takes the mean of the last
    3 checkpoints' gaps; joint Bayes itself (gap 0) is off the log axis. No gap here is <= 0 (smallest .0087, the fully
    trained LSTM at N = 1000); any would be left out. rescues (John 2026-10-29 log-log copy, 11-01 all four figures):
    the ESNs that beat naive Bayes (esn_rescues: MLP read-in and read-out; joint output, joint loss), in (a) at N = 1000
    at their end only (white: a longer run; drawn over the markers there), and in
    their own panel with the unrescued ESNs, on its own x range (their runs go to 104k). They trained in their own
    worlds, SANITY loss (power 0.5), not the seed-0 world at power 1 of the rest: compare gaps to each world's joint
    Bayes (log-log copies). One marker per width (John 2026-11-01): a run trained past 20k shows its latest value,
    white, in its 20k value's place on its family's line; no horizontal nudges. params_panel (John 2026-11-01): (b), the
    same against trainable parameters (each run's summary count, else its family's formula: 36 N + 20 for the frozen FFs,
    38 N + 20 for the reservoirs with their start state, 8 N^2 + 46 N + 20 for the LSTMs; the ESNs counted from their
    weights, the frozen LSTM excluded: 8.05M with the MLPs, 118k with the joint output), sharing (a)'s y axis; the
    reference labels and legends move right of (b), the curve panels' letters one on. upto (thousands of episodes; John
    2026-11-01, "everything shown up until 20K"): every run at its value at upto in (a) and (b), filled (no longer-run
    values, the ESNs and the N = 100 LSTM included), every curve cut there and the x axes ending there. side_legend False
    (John 2026-11-01, the 20k copy): no family legend (the curve panels' titles name the families; the N = 100 LSTM's
    protocol goes into the LSTM panel's legend), the top panels take its column, the line-width key runs along the
    bottom."""
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    shade = {1: RAMP(.15), 2: RAMP(.45), 3: RAMP(.75), 4: RAMP(1.)}
    tffs_col = "#1b7837"                                        # the green partner of depth 30's purple (PRGn)
    names = {1: "depth 1 (readin + readout)", 2: "depth 2, its layer frozen", 3: "depth 3, both layers frozen",
             4: "depth 4, all 3 layers frozen"}
    titles = {1: "depth 1 (readin + readout)", 2: "depth 2, 1 layer frozen", 3: "depth 3, 2 layers frozen",
              4: "depth 4, 3 layers frozen"}
    mark = {1: "o", 2: "s", 3: "D", 4: "h"}                    # neighbouring depths nearly coincide at small widths
    res_col, d30_col = "#b2182b", "#762a83"

    def X(ep):
        """episodes in thousands; with loglog the test at episode 0 is left out"""
        ep = np.asarray(ep, float)
        return np.where(ep > 0, ep / 1000, np.nan) if loglog else ep / 1000

    def Y(acc, joint):
        """the plotted quantity per checkpoint: accuracy, or with loglog its gap to joint Bayes on the same batch (> 0)"""
        if not loglog:
            return np.asarray(acc, float)
        g = np.asarray(joint, float) - np.asarray(acc, float)
        return np.where(g > 0, g, np.nan)

    def val(acc, joint):
        """(a)'s value: the mean of the last 3 checkpoints' accuracies, or of their gaps to joint Bayes"""
        if not loglog:
            return float(np.mean(acc[-3:]))
        g = float(np.mean(np.asarray(joint[-3:], float) - np.asarray(acc[-3:], float)))
        return g if g > 0 else np.nan

    def cut(x, y):
        """a curve only up to `upto` thousand episodes (John 2026-11-01: a copy with everything up to 20k); all of it if None"""
        x, y = np.asarray(x, float), np.asarray(y, float)
        return (x, y) if upto is None else (x, np.where(x <= upto + 1e-9, y, np.nan))

    def series(stem_of, widths_, run_ok):
        """(width, KL, accuracy, final, last episode) per run at 20k episodes, and its whole curve: finished runs, plus
        the running ones that run_ok admits, at their latest 3 checkpoints up to 20k (final once the run is past 20k)."""
        rows, crv = [], {}
        for n in widths_:
            c = load_curve(os.path.join(OUT, stem_of(n) + "_curve.npz"))
            done = os.path.exists(os.path.join(OUT, stem_of(n) + "_summary.json"))
            if c is not None and (done or run_ok(n, c)):
                m = c["ep"] <= 20000                                  # a run trained past 20k: its value at 20k
                rows.append((n, float(np.mean(c["kl_joint"][m][-3:])), val(c["acc"][m], c["joint"][m]),
                             done or bool(c["ep"][-1] > 20000), int(c["ep"][m][-1])))
                crv[n] = (X(c["ep"]), Y(c["acc"], c["joint"]), done)
        return rows, crv
    reservoir, curves = series(reservoir_stem, RESERVOIR_WIDTHS, lambda n, c: n in running)
    d30, d30_curves = series(tff1_stem, FROZEN_WIDTH_SERIES, lambda n, c: running_d30 and c["ep"][-1] >= 500)   # not a stale start
    lstm, lstm_curves = series(lambda n: LSTM_STEMS[n], sorted(LSTM_STEMS), lambda n, c: False)
    n100 = lstm_n100()                                           # the N = 100 LSTM (5 seeds, SANITY loss, 40k; John 11-01)
    if n100 is not None:
        x100, a100, J100 = n100
        m100 = x100 <= 20 + 1e-9
        lstm = sorted(lstm + [(100, np.nan, val(a100[m100], J100[m100]), True, 20000)])
        lstm_curves[100] = (x100, Y(a100, J100), True)
    shared, shared_curves = series(tffs_stem, FROZEN_WIDTH_SERIES, lambda n, c: running_shared and c["ep"][-1] >= 500) \
        if shared_readin else ([], {})
    shared = [r for r in shared if r[3]]                         # (a): finished, or past 20k (its 20k value); curves: all
    base = baselines()
    resc = esn_rescues() if rescues else []
    widths = sorted(set(FROZEN_WIDTH_SERIES) | set(curves) | set(LSTM_STEMS))
    thick = lambda n: .3 + .75 * np.log10(n / 10)               # line width encodes the network width

    # layout in inches: (a) against width and, with params_panel, (b) against trainable parameters (John 2026-11-01),
    # sharing (a)'s y axis, the legends right of the last; then the curve panels in rows of ncols
    n_curves = len(depths) + 2 + bool(shared_readin) + bool(rescues)
    ncols = ncols or n_curves
    nrows = -(-n_curves // ncols)
    L, R, PW, PH, GW, GH = .7, .78, 2.15, 2.4, .16, .8         # margins, curve panel width / height, gaps
    AH, LEG, GAP = 2.6, (3.1 if side_legend else 0), .3        # top panels' height; legend column right of the reference
    W = L + ncols * PW + (ncols - 1) * GW + R                  # labels; between (a) and (b)
    H = .35 + AH + .85 + nrows * PH + (nrows - 1) * GH + .55 + (0 if side_legend else .42)  # without: the line key below
    AW = W - L - R - LEG
    if params_panel:
        AW = (AW - GAP) / 2
    fig = plt.figure(figsize = (W, H))
    ax_acc = fig.add_axes([L / W, (H - .35 - AH) / H, AW / W, AH / H])
    ax_par = fig.add_axes([(L + AW + GAP) / W, (H - .35 - AH) / H, AW / W, AH / H], sharey = ax_acc) if params_panel else None
    top_right = ax_par if params_panel else ax_acc             # the reference labels and the legends go right of it
    letters = "cdefghijk" if params_panel else "bcdefghij"     # the curve panels'
    curve_axes = []
    for k in range(n_curves):
        r, c = divmod(k, ncols)
        y = H - .35 - AH - .85 - (r + 1) * PH - r * GH
        curve_axes.append(fig.add_axes([(L + c * (PW + GW)) / W, y / H, PW / W, PH / H]))

    # top row: accuracy at 20k episodes against width ((a)) and trainable parameters ((b)); a run trained longer
    # (continued, or one long run) shows its latest value instead, white, in its 20k value's place on its family's line
    # (John 2026-11-01: one marker per width)
    fams = ([(lambda n, d = d: frozen_width_stem(d, n), f"depth {d}", FROZEN_WIDTH_SERIES) for d in depths]
            + [(tff1_stem, "depth 30", FROZEN_WIDTH_SERIES), (reservoir_stem, "reservoir", RESERVOIR_WIDTHS)]
            + ([(tffs_stem, "shared readin", sorted(shared_curves))] if shared_readin else [])
            + [(lambda n: LSTM_STEMS[n], "lstm", sorted(LSTM_STEMS))])   # fully trained (continue_lstm, John 2026-11-02)
    late, cont_running, long_done = {}, [], []                    # (family, width) -> its longer run's latest value
    for stem_of, fam, ns in (fams if upto is None else []):     # cut at upto: every run at its 20k value
        for n in ns:
            cc = late_curve(stem_of(n))
            if cc is None or load_curve(os.path.join(OUT, stem_of(n) + "_curve.npz")) is None:
                continue
            late[(fam, n)] = val(cc[1], cc[3])
            if not cc[2]:
                cont_running.append((fam, n, cc[0][-1], TRAINED_ON.get(stem_of(n))))
            elif stem_of(n) in TRAINED_ON:                       # finished, trained past 20k in one go: say how far
                long_done.append(f"{fam} at N = {n:g}: one run to {TRAINED_ON[stem_of(n)] // 1000}k")
    if n100 is not None and upto is None:
        late[("lstm", 100)] = val(a100, J100)
        long_done.append("LSTM at N = 100: 5 seeds, one run each to 40k")
    stems = {**{f"depth {d}": (lambda n, d = d: frozen_width_stem(d, n)) for d in depths}, "depth 30": tff1_stem,
             "shared readin": tffs_stem, "reservoir": reservoir_stem, "lstm": lambda n: LSTM_STEMS[n]}

    def n_params(fam, n):
        """trainable parameters of the family's run at width n: its summary's count, else the family's formula (a
        frozen FF: read-in 16 N + read-out 20 N + 20; a reservoir also its 2 N start state; an LSTM 8 N^2 + 46 N + 20)"""
        try:
            f = os.path.join(OUT, stems[fam](n) + "_summary.json")
        except KeyError:                                          # the N = 100 LSTM: no ff_deep run
            f = ""
        t = json.load(open(f)).get("trainable") if os.path.exists(f) else None
        return t or (8 * n * n + 46 * n + 20 if fam == "lstm" else 38 * n + 20 if fam == "reservoir" else 36 * n + 20)

    def draw_top(ax, by_params):
        """one top panel: every family against width or trainable parameters; the legend handles of (a) only"""
        X = (lambda fam, n: n_params(fam, n)) if by_params else (lambda fam, n: n)
        lab = (lambda t: "_nolegend_") if by_params else (lambda t: t)

        def open_marks(pts, col, mk, ms):
            """white markers over a family's line at its longer runs' points (the filled marker there is covered)"""
            if pts:
                ax.plot([q[0] for q in pts], [q[1] for q in pts], marker = mk, ms = ms, color = col, mfc = "white", mew = 1.2,
                        ls = "none", zorder = 5)

        for d in depths:
            rows = []
            for n in FROZEN_WIDTH_SERIES:
                c = load_curve(os.path.join(OUT, frozen_width_stem(d, n) + "_curve.npz"))
                if os.path.exists(os.path.join(OUT, frozen_width_stem(d, n) + "_summary.json")) and c is not None:
                    rows.append((n, late.get((f"depth {d}", n), val(c["acc"], c["joint"]))))
            if rows:
                ax.plot([X(f"depth {d}", r[0]) for r in rows], [r[1] for r in rows], color = shade[d], marker = mark[d], ms = 5, lw = 1.4,
                        label = lab(names[d]))
                open_marks([(X(f"depth {d}", r[0]), r[1]) for r in rows if (f"depth {d}", r[0]) in late], shade[d], mark[d], 5)

        def draw_family(rows, col, mk, name, fam):
            """Finished widths as a solid line (a longer run's latest value white, in its 20k value's place); running ones
            as open markers on one dotted path from the last finished width."""
            rows = [(r[0], r[1], late.get((fam, r[0]), r[2])) + tuple(r[3:]) for r in rows]
            fin, run = [r for r in rows if r[3]], [r for r in rows if not r[3]]
            ms = 8 if mk == "*" else 5.5                     # a star reads small at the others' size
            h = ax.plot([X(fam, r[0]) for r in fin], [r[2] for r in fin], color = col, marker = mk, ms = ms, lw = 1.4, label = lab(name)) if fin else []
            open_marks([(X(fam, r[0]), r[2]) for r in fin if (fam, r[0]) in late], col, mk, ms)
            if run:
                path = [q for q in fin if q[0] < run[0][0]][-1:] + run
                ax.plot([X(fam, q[0]) for q in path], [q[2] for q in path], color = col, lw = 1.1, ls = ":")
                eps = sorted({r[4] for r in run})
                when = f"episode {eps[0]:,}" if len(eps) == 1 else f"episodes {eps[0]:,}-{eps[-1]:,}"
                who = f" N = {run[0][0]:g}" if len(run) == 1 and f"N = {run[0][0]:g}" not in name else ""
                h = h + ax.plot([X(fam, r[0]) for r in run], [r[2] for r in run], ls = "none", marker = mk, ms = ms, color = col,
                                mfc = "white", label = lab(f"{name}{who},\n    still training ({when})"))
            return h
        family = draw_family(sorted(d30), d30_col, "*", r"depth 30, all 29 layers frozen ($\times\sqrt{6}$)", "depth 30")
        if shared_readin:
            family += draw_family(sorted(shared), tffs_col, "P", r"depth 30, the same layers ($\times\sqrt{6}$), input at" + "\n    every layer through one shared readin"
                                  + "".join(f"\n    (lr {lr_text(TFFS_LR[r[0]])} at N = {r[0]:g}, 100-update warm-up)" for r in shared if r[0] in TFFS_LR),
                                  "shared readin")
        family += draw_family(sorted(reservoir), res_col, "v", "reservoir (LSTM frozen, power 1; lr 5e-4 at N = 10000)", "reservoir")
        family += draw_family(lstm, LSTM1_COLOR, "^", "LSTM, fully trained (power 1)" + ("\n    (N = 100: SANITY loss, power 0.5, 5 seeds)"
                                                                                         if n100 is not None else ""), "lstm")
        for r in (r for r in resc if r["rescue"]):                # their end only (white: a longer run), at N = 1000
            if upto is None:
                y_r, fill, tag = val(r["segments"][-1][1], r["segments"][-1][2]), "white", r["a_label"]
            else:                                                 # cut: their value at upto, filled like everyone's
                x0, a0, J0 = r.get("start", r["segments"][0][:3])
                m0 = x0 <= upto + 1e-9
                y_r, fill, tag = val(a0[m0], J0[m0]), r["col"], re.sub(r"\(at \d+k\)", f"(at {upto:g}k)", r["a_label"])
            ax.plot([r["params"] if by_params else 1000], [y_r], marker = r["mk"], ms = 6.5, color = r["col"], mfc = fill, mew = 1.2,
                    ls = "none", zorder = 6.5)
            family.append(plt.Line2D([], [], color = r["col"], marker = r["mk"], ms = 6, mfc = fill, mew = 1.2, lw = 1.4, label = tag))
        return family

    running_text = []                                             # one line per width: "depths 1, 2 at N = 10000: ..."
    for n in sorted({n for _, n, _, _ in cont_running}):
        fams_n, ks = [f for f, m, _, _ in cont_running if m == n], [k for _, m, k, _ in cont_running if m == n]
        one = "".join(f"one {t // 1000}k run, " for t in sorted({t for _, m, _, t in cont_running if m == n and t}))
        deps = [f.split(" ", 1)[1] for f in fams_n if f.startswith("depth ")]
        who = ", ".join(([("depths " if len(deps) > 1 else "depth ") + ", ".join(deps)] if deps else [])
                        + [{"lstm": "LSTM"}.get(f, f) for f in fams_n if not f.startswith("depth ")])
        when = f"{min(ks):.1f}k" if max(ks) - min(ks) < .05 else f"{min(ks):.1f}-{max(ks):.1f}k"
        line = f"{who} at N = {n:g}: {one}still training ({when})"
        running_text.append(line if len(line) <= 48 else f"{who} at N = {n:g}:\n     {one}still training ({when})")
    cont_handles = [] if upto is not None else [plt.Line2D([], [], color = "#5f5e5a", marker = "o", ms = 5.5, mfc = "white", mew = 1.2, ls = "none",
                               label = "the same run trained longer (continued to 40k)" + "".join(f"\n    ({t})" for t in running_text + long_done))]
    family = draw_top(ax_acc, False)
    if params_panel:
        draw_top(ax_par, True)
    if resc:
        family.append(plt.Line2D([], [], color = "none", label = "(rescued ESNs: frozen LSTM, N = 1000; SANITY loss,\n    power 0.5, each in its own world; see "
                                 + f"({letters[len(depths) + 2 + bool(shared_readin)]}))"))         # the ESN panel's letter
    quantity = ("gap to joint Bayes after" + ("\n20k episodes, " if params_panel else " 20k episodes, ") + r"$\mathrm{acc}_\mathrm{joint} - \mathrm{acc}$"
                if loglog else "accuracy after 20k episodes")
    if loglog:
        ax_acc.set_ylabel("gap to joint Bayes")
        ax_acc.set_yscale("log")
        ax_acc.set_yticks([.01, .02, .05, .1, .2, .5], ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5"])
        ax_acc.set_yticks([], minor = True)
    else:
        ax_acc.set_ylabel("final-step accuracy")
    ax_acc.set_title(f"(a) {quantity}" + (",\nagainst width" if params_panel and not loglog else "")
                     + ("" if params_panel else " (open: the same run, trained longer)"), fontsize = fs["label"])
    ax_acc.set_xscale("log")
    shown = {10, 100, 1000, 10000} if AW < 5 else set(widths) - {400, 600}   # a narrow (a) labels decades only
    ax_acc.set_xticks(widths, [f"{n:g}" if n in shown else "" for n in widths])
    ax_acc.set_xticks([], minor = True)
    ax_acc.tick_params(axis = "x", labelsize = fs["tick"] - 1)
    ax_acc.set_xlim(7, 28000)
    ax_acc.set_xlabel("width N")
    ax_acc.spines['top'].set_visible(False)
    ax_acc.spines['right'].set_visible(False)
    if params_panel:
        ax_par.set_xscale("log")
        decades = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
        ax_par.set_xticks(decades, ["100", "1k", "10k", "100k", "1M", "10M"])
        ax_par.set_xticks([], minor = True)
        ax_par.tick_params(axis = "x", labelsize = fs["tick"] - 1)
        ax_par.tick_params(axis = "y", labelleft = False)
        ax_par.set_xlim(200, 2e7)
        ax_par.set_xlabel("trainable parameters")
        ax_par.set_title("(b) the same, against\ntrainable parameters", fontsize = fs["label"])
        ax_par.spines['top'].set_visible(False)
        ax_par.spines['right'].set_visible(False)
    xleg = 1 + (R + .05) / AW                                   # just right of the reference-line labels
    depth_lines = [l for l in ax_acc.get_lines() if l.get_label().startswith("depth") and l not in family]
    line_key = [plt.Line2D([], [], color = "#5f5e5a", lw = thick(n), label = f"{n:g}") for n in widths]
    if side_legend:
        families = top_right.legend(handles = depth_lines + family + cont_handles, frameon = False, fontsize = fs["tick"] - 1.5, loc = "upper left",
                                    bbox_to_anchor = (xleg, 1.04))
        top_right.add_artist(families)
        y_key = (-.24 if shared_readin else -.14) if rescues else -.06
        key = top_right.legend(handles = line_key,
                               title = "line width in the training curves = width N", title_fontsize = fs["tick"] - 1.5, frameon = False,
                               fontsize = fs["tick"] - 2, loc = "lower left", bbox_to_anchor = (xleg, y_key), ncol = 5, columnspacing = .9,
                               handlelength = 1.6, alignment = "left")
        # the families' legend grows downwards with its notes (runs still training, runs trained longer; 2026-11-02 the
        # all-depths figure's ran into the key): keep them 3 pt apart, measured at the saved resolution (savefig below,
        # dpi 150: the legends' text is ~8% taller there than at the figure's 300 dpi), by raising the families' legend
        # into the room above it, and only then lowering the key (below it are the curve panels' titles)
        dpi0 = fig.dpi
        fig.set_dpi(150)
        r = fig.canvas.get_renderer()
        fam, kb, pt = families.get_window_extent(r), key.get_window_extent(r), fig.dpi / 72
        clash = kb.y1 - (fam.y0 - 3 * pt)                                                        # pixels at 150 dpi
        if clash > 0:
            up = min(clash, max(0., fig.bbox.y1 - 3 * pt - fam.y1))
            families.set_bbox_to_anchor((xleg, 1.04 + up / top_right.bbox.height))
            if clash > up:
                key.set_bbox_to_anchor((xleg, y_key - (clash - up) / top_right.bbox.height))
        fig.set_dpi(dpi0)
    else:                                                        # the families named by the curve panels' titles (John
        fig.legend(handles = line_key, title = "line width in the training curves = width N",   # 2026-11-01); the key below
                   title_fontsize = fs["tick"] - 1, frameon = False, fontsize = fs["tick"] - 1.5, loc = "lower center",
                   bbox_to_anchor = (.5, .01), ncol = len(line_key), columnspacing = 1.2, handlelength = 1.8)

    # training curves: one panel per depth, then depth 30, then the LSTMs (frozen and fully trained)
    for i, d in enumerate(depths):
        ax = curve_axes[i]
        for n in FROZEN_WIDTH_SERIES:
            c = load_curve(os.path.join(OUT, frozen_width_stem(d, n) + "_curve.npz"))
            if c is not None and os.path.exists(os.path.join(OUT, frozen_width_stem(d, n) + "_summary.json")):   # finished runs only
                ax.plot(*cut(X(c["ep"]), Y(c["acc"], c["joint"])), color = shade[d], lw = thick(n), alpha = .9)
                cc = continued_curve(frozen_width_stem(d, n))
                if cc is not None and upto is None:
                    ax.plot(np.r_[c["ep"][-1] / 1000, cc[0]], Y(np.r_[c["acc"][-1], cc[1]], np.r_[c["joint"][-1], cc[3]]),
                            color = shade[d], lw = thick(n), alpha = .9)
        ax.set_title(f"({letters[i]}) {titles[d]}", fontsize = fs["label"] - .5, color = shade[d] if d > 1 else RAMP(.35))
    ax = curve_axes[len(depths)]                                 # depth 30, all 29 layers frozen x sqrt(6)
    for n, (x, y, done) in sorted(d30_curves.items()):
        ax.plot(*cut(x, y), color = d30_col, lw = thick(n), alpha = .85)
        cc = continued_curve(tff1_stem(n))
        if cc is not None and upto is None:
            ax.plot(np.r_[x[-1], cc[0]], np.r_[y[-1], Y(cc[1], cc[3])], color = d30_col, lw = thick(n), alpha = .85)
    ax.set_title(rf"({letters[len(depths)]}) depth 30, all frozen ($\times\sqrt{{6}}$)", fontsize = fs["label"] - .5, color = d30_col)
    k_lstm = len(depths) + 1
    if shared_readin:                                            # the same 29 frozen layers, input at every layer
        ax = curve_axes[k_lstm]
        for n, (x, y, done) in sorted(shared_curves.items()):
            ax.plot(*cut(x, y), color = tffs_col, lw = thick(n), alpha = .85)
        ax.set_title(rf"({letters[k_lstm]}) depth 30, shared readin ($\times\sqrt{{6}}$)", fontsize = fs["label"] - .5, color = tffs_col)
        still_shared = "".join(f"\nN = {n:g}" + (f", lr {lr_text(TFFS_LR[n])}" if n in TFFS_LR else "")
                               + ((": still training" + (f" (to {TRAINED_ON[tffs_stem(n)] // 1000}k)" if tffs_stem(n) in TRAINED_ON else ""))
                                  if not done else f": {TRAINED_ON[tffs_stem(n)] // 1000}k episodes")
                               for n, (_, _, done) in sorted(shared_curves.items()) if not done or tffs_stem(n) in TRAINED_ON)
        for a, note in ((curve_axes[len(depths)], "input at layer 1 only"), (ax, "input at every layer" + still_shared)):
            a.text(.05, .2 if loglog else .8, note, transform = a.transAxes, fontsize = fs["tick"] - 1.5, color = "#5f5e5a",
                   va = "top")                                   # log-log: the curves start top left
        k_lstm += 1
    ax = curve_axes[k_lstm]                                      # the LSTM frozen (reservoir) and fully trained, power 1
    for crv, col in ((curves, res_col), (lstm_curves, LSTM1_COLOR)):
        for n, (x, y, done) in sorted(crv.items()):
            ax.plot(*cut(x, y), color = col, lw = thick(n), alpha = .85)
            cc = (None if upto is not None else continued_curve(reservoir_stem(n)) if crv is curves else
                  continued_curve(LSTM_STEMS[n]) if n in LSTM_STEMS else None)        # the N = 100 LSTM: its own 40k runs
            if cc is not None:
                ax.plot(np.r_[x[-1], cc[0]], np.r_[y[-1], Y(cc[1], cc[3])], color = col, lw = thick(n), alpha = .85)
    still = [n for n, (_, _, done) in curves.items() if not done]
    ax.legend(handles = [plt.Line2D([], [], color = LSTM1_COLOR, lw = 1.5, label = "fully trained" + (
                             "\n(N = 100: SANITY loss, 5 seeds)" if n100 is not None and not side_legend else "")),
                         plt.Line2D([], [], color = res_col, lw = 1.5, label = "frozen (reservoir)" +
                                    "".join(f"; N = {n:g} running" for n in still))],
              frameon = False, fontsize = fs["tick"] - 2.5,
              **(dict(loc = "lower left") if loglog else                  # log-log: empty, early training is far from joint
                 dict(loc = "center right", bbox_to_anchor = (1, .755))))  # between the linear bound and the trained LSTMs
    ax.set_title(f"({letters[k_lstm]}) LSTM, frozen and trained", fontsize = fs["label"] - .5, color = res_col)
    if rescues:                                                  # the rescued ESNs and the unrescued one (SANITY, own worlds)
        ax = curve_axes[k_lstm + 1]
        for r in resc:
            for x, a, J, ls in r["segments"]:
                ax.plot(*cut(np.where(x > 0, x, np.nan) if loglog else x, Y(a, J)), color = r["col"], ls = ls, lw = r["lw"], alpha = .9)
        ax.legend(handles = [plt.Line2D([], [], color = r["col"], marker = r["mk"], ms = 5, lw = r["lw"], ls = r["segments"][0][3],
                                        label = r["label"])
                             for r in resc], frameon = False, fontsize = fs["tick"] - 2.5,
                  **(dict(loc = "lower left") if loglog else dict(loc = "lower right", bbox_to_anchor = (1, .1))))   # the empty corner
        ax.set_title(f"({letters[k_lstm + 1]}) rescued ESNs (N = 1000)", fontsize = fs["label"] - .5, color = "#c51b7d")

    acc_refs = [(base["acc_naive"], "naive Bayes", NAIVE_COLOR, "--"), (base["acc_joint"], "joint Bayes", JOINT_COLOR, ":"),
                (LINEAR_CE, "linear rule\n(CE fit)", LINEAR_COLOR, "-."), (.1, "chance", "#b4b2a9", ":")]
    if loglog:                                                   # their gaps to joint Bayes; joint Bayes itself is 0
        acc_refs = [(base["acc_joint"] - y, label, col, ls) for y, label, col, ls in acc_refs if label != "joint Bayes"]
    longest = max([20.] + [late_curve(st)[0][-1] for st in
                           [frozen_width_stem(d, n) for d in depths for n in FROZEN_WIDTH_SERIES] + [tff1_stem(n) for n in FROZEN_WIDTH_SERIES]
                           + [reservoir_stem(n) for n in RESERVOIR_WIDTHS] + [tffs_stem(n) for n in shared_curves] + list(LSTM_STEMS.values())
                           if late_curve(st) is not None])
    span = lambda top: next(x for x in (20, 40, 60, 80, 120) if top <= x + .01)   # thousands of episodes
    xmax = span(longest) if upto is None else upto
    xmax_esn = (span(max(float(np.max(x)) for r in resc for x, *_ in r["segments"])) if resc else xmax) if upto is None else upto
    k_esn = len(depths) + 2 + bool(shared_readin) if rescues else None             # the ESN panel: its own x range
    gap_ticks = [.01, .02, .05, .1, .2, .5]
    for k, ax in enumerate(curve_axes):
        xm = xmax_esn if k == k_esn else xmax
        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(.42, xm * 1.1)
            ticks = [t for t in ((.5, 1, 2, 5, 10, 20, 50, 100) if xm > 60 else (.5, 1, 2, 5, 10, 20, 40, 60)) if t <= xm]
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
            ax.set_xticks([], minor = True)
            ax.set_ylim(.007, .8)
            ax.set_yticks(gap_ticks, [f"{t:g}" for t in gap_ticks])
            ax.set_yticks([], minor = True)
        else:
            ax.set_xlim(0, xm + .5)
            ax.set_xticks([0, 5, 10, 15, 20] if xm == 20 else list(range(0, xm + 1, 10 if xm <= 60 else 20)))
            ax.set_ylim(.05, .8)
        if xm > 20:
            ax.axvline(20, color = "#d3d1c7", lw = .7, zorder = 0)
        ax.set_xlabel("training episodes (thousands" + (", log)" if loglog else ")"), fontsize = fs["label"] - .5)
        if k % ncols == 0:
            ax.set_ylabel("gap to joint Bayes" if loglog else "final-step test accuracy", fontsize = fs["label"] - .5)
        else:
            ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if k % ncols == ncols - 1 or k == n_curves - 1:
            ref_lines(ax, acc_refs, fs)                          # labelled once per row, at its right end
        else:
            for y, _, col, ls in acc_refs:
                ax.axhline(y, color = col, ls = ls, lw = 1, zorder = 1)
    if params_panel:                                             # labelled once, right of (b); (a) shares its y range
        for y, _, col, ls in acc_refs:
            ax_acc.axhline(y, color = col, ls = ls, lw = 1, zorder = 1)
    ref_lines(top_right, acc_refs, fs)
    fig.savefig(fig_path(name), dpi = 150)
    print(fig_path(name))


def frozen_widths_figures(**kw):
    """The two versions John asked for (2026-10-21): depths 1 and 30 (ff_deep_frozen_widths.png; depth 2 dropped later the
    same day), and every depth of this setting, 1-4 and 30, with the shared-readin net (John 2026-10-25; it had its own
    version, ff_deep_frozen_widths_shared_readin.png, from 10-22), curve panels in two rows of four
    (ff_deep_frozen_widths_all_depths.png), each with a log-log copy (*_loglog.png; John 2026-10-29 the all-depths one,
    11-01 the minimal one; see loglog in frozen_widths_figure: the gap to joint Bayes on log axes), and all four with the
    rescued ESNs and their references (esn_rescues; John 2026-11-01); and the minimal log-log one cut at 20k episodes
    (ff_deep_frozen_widths_loglog_20k.png, upto = 20; John 2026-11-01)."""
    for loglog, tail in ((False, ""), (True, "_loglog")):          # the ESNs on all four (John 2026-11-01)
        frozen_widths_figure(depths = (1,), name = f"ff_deep_frozen_widths{tail}.png", loglog = loglog, rescues = True, **kw)
        frozen_widths_figure(depths = (1, 2, 3, 4), name = f"ff_deep_frozen_widths_all_depths{tail}.png", ncols = 4, shared_readin = True,
                             loglog = loglog, rescues = True, **kw)
    frozen_widths_figure(depths = (1,), name = "ff_deep_frozen_widths_loglog_20k.png", loglog = True, rescues = False, upto = 20,
                         side_legend = False, **kw)                                    # everything only to 20k episodes, no ESN panel (John 2026-11-01:
                                                                  # "super minimal")


# ---- extrapolation to infinite training and width (John, 2026-10-21) ----
EXTRAP = os.path.join(OUT, "extrapolation")


def _limit_form(t, a, b, c, beta):
    """Test accuracy after t thousand updates as a power-law approach to the limit a. Backtested 2026-10-21 on all 58
    finished runs: fitted on 1k-10k updates it predicts the 19k-20k accuracy with mean error 0.007 and no bias (noise of
    that target 0.003); an exponential approach under-predicts by 0.019 on average, logarithmic growth over-predicts by
    0.019, a power law without the offset c is slightly worse (0.008)."""
    return a - b * (t + c) ** (-beta)


def extrapolation_runs():
    """(family, width, stem) for every run of the frozen-widths figures."""
    out = [(f"depth {d}", n, frozen_width_stem(d, n)) for d in (1, 2, 3, 4) for n in FROZEN_WIDTH_SERIES]
    out += [("depth 30", n, tff1_stem(n)) for n in FROZEN_WIDTH_SERIES]
    out += [("reservoir", n, reservoir_stem(n)) for n in RESERVOIR_WIDTHS]
    out += [("LSTM trained", n, LSTM_STEMS[n]) for n in sorted(LSTM_STEMS)]
    return out


JOINT_CAP = .7663          # joint Bayes's argmax accuracy: the optimal decision rule, so no predictor beats it on this measure


def fit_training_limits(refresh = False, draws = 300, cap = JOINT_CAP):
    """Per finished run: the limit of test accuracy at infinite training (_limit_form fitted on checkpoints >= 1k
    updates), a 95% interval from a moving-block residual bootstrap (blocks of 4 checkpoints, `draws` refits), and the
    limit refitted on 1k-10k updates only. identified = interval narrower than 0.1 with its top below 0.85, i.e. the curve
    has bent enough within 20k updates to pin its limit. Cached in EXTRAP/training_limits.json by stem; a run is refitted
    when its curve file is newer than the cache entry."""
    from scipy.optimize import curve_fit
    os.makedirs(EXTRAP, exist_ok = True)
    path = os.path.join(EXTRAP, "training_limits.json" if cap is None else "training_limits_capped.json")
    cache = json.load(open(path)) if os.path.exists(path) and not refresh else {}
    rng = np.random.default_rng(0)
    top = 1.5 if cap is None else cap
    p0, bounds = [min(.6, top - .01), .3, .5, .7], ([0, 0, 0, .01], [top, 5, 20, 5])

    def fit(t, y, start = p0):
        try:
            return curve_fit(_limit_form, t, y, p0 = start, bounds = bounds, maxfev = 20000)[0]
        except Exception:
            return None
    for fam, n, stem in extrapolation_runs():
        f = os.path.join(OUT, stem)
        if not (os.path.exists(f + "_summary.json") and os.path.exists(f + "_curve.npz")):
            continue
        mtime = os.path.getmtime(f + "_curve.npz")
        if stem in cache and cache[stem]["mtime"] >= mtime:
            continue
        c = np.load(f + "_curve.npz")
        t, y = c["ep"] / 1000., c["acc"].astype(float)
        keep = t >= 1
        t, y = t[keep], y[keep]
        par = fit(t, y)
        if par is None:
            continue
        early = t <= 10.001
        par10 = fit(t[early], y[early])
        fitted = _limit_form(t, *par)
        res = y - fitted
        boots = []
        for _ in range(draws):
            idx = np.concatenate([np.arange(s0, s0 + 4) for s0 in rng.integers(0, len(res) - 3, size = len(res) // 4 + 1)])[:len(res)]
            q = fit(t, fitted + res[idx], start = par)
            if q is not None:
                boots.append(q[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        cache[stem] = dict(family = fam, N = int(n), mtime = mtime, acc20k = float(np.mean(c["acc"][-3:])),
                           params = [float(v) for v in par], limit = float(par[0]), lo = float(lo), hi = float(hi),
                           limit_10k = None if par10 is None else float(par10[0]),
                           identified = bool(hi - lo < .1 and hi < (.85 if cap is None else cap - .005)))
    json.dump(cache, open(path, "w"), indent = 1)
    return cache


# reference levels (numbers kept here so the figure stands alone)
BANDS = dict(naive = (.436, .469), joint = (.755, .775))   # 95% of single test batches, 2545 batches of the FF/LSTM curves
LINEAR_READOUT = (.497, .538)   # a fixed linear readout of the five counts (naive_vs_linear, 2026-10-21): cross-entropy fit to the
                                # joint posterior .497 (.504 with more data), fitted for accuracy >= .538; naive Bayes is one such
                                # readout (.449). Beating naive by up to this much needs no inference over the other context.
                                # Neither is a ceiling (verify_2026-10-25): .538 is one local optimum of the non-convex accuracy
                                # fit; the best of 3 starts per pair reaches ~.550, and linear rules are bounded above only by
                                # the joint. The extrapolation figure still shades .497-.538.
LINEAR_CE = .503                # the linear rule fitted by cross-entropy to the joint posterior, the nets' own objective (John
                                # 2026-10-25): convex, so one optimum; .497/.501/.503 with 1500/4000/16000 training episodes
                                # per pair, .501-.503 on 8M of the env's own episodes (verify_2026-10-25/audit). A reference
                                # for a linear readout trained like the nets, not an upper bound of linear rules.
LINEAR_COLOR = "#6b6b47"
JM = os.path.join(EXTRAP, "joint_models")   # the joint (width, training-time) models and the depth model, 2026-10-21 workflow


def _extrap_models():
    """the judged joint models (joint_models/judge/extrap_final.py: predict, interval, ...) and their parameter file"""
    sys.path.insert(0, os.path.join(JM, "judge"))
    import extrap_final
    return extrap_final, json.load(open(os.path.join(JM, "judge", "final_extrapolation.json")))


def extrapolation_figure(name = "ff_deep_extrapolated.png"):
    """Extrapolated accuracy from the joint models (John, 2026-10-21, third version). Every family is fitted jointly over its
    widths and training curves (joint_models/judge: depths 1-4 and 30 'kl-acc', reservoir 'smooth-max', trained LSTM
    'shape'), so its curves cannot cross and share one saturation; the depth model (joint_models/depth) extends the
    critical-init stack (frozen default draws x sqrt 6, depths 1 and 30 measured on the GPU, 10 and 100 as CPU replicas) to
    any depth. Argmax accuracy (the env's measure). Bands: naive and joint Bayes (95% of single test batches), a fixed
    linear readout of the counts (LINEAR_READOUT), chance.
    (a) acc(N, t = inf) against width for every family; markers on the line where the width was trained (the t = inf value
        is still an extrapolation in time), open beyond; from N = 10^4 up and at N = inf: 95% bootstrap (thick) and the
        structural envelope, i.e. other forms that fit the data as well (thin). Depth 1 at lr 1e-3 dies at N = 10^5 and
        2x10^5 (probes 2026-10-21), so the wide end describes nets whose training stays stable.
    (b) acc(N, D, t = inf) of the critical-init stack: depth acts through an effective width, so width must outgrow depth;
        paths N = 1000 D^gamma; dots: trained.
    (c) very wide nets (N = 10^8) against depth, by training budget; D = inf in both orders of the limits.
    (d-g) training curves (line width = N) with the joint model's continuation to convergence; grey: trained range;
        lighter: the 4x reach the backtests checked; t = inf limit at the right. (e, f): the width-20000 runs continued
        (dots) against the model's prediction at 40k updates made before the continuation (diamond, 95%)."""
    from matplotlib.colors import LinearSegmentedColormap as LSC
    X, J = _extrap_models()
    PM = np.load(os.path.join(JM, "depth", "phase_map.npz"), allow_pickle = True)
    D100 = np.load(os.path.join(JM, "depth", "depth100_curves.npz"), allow_pickle = True)
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    fams = [("depth 1", RAMP(.15), "o", "depth 1 (readin + readout)"), ("depth 2", RAMP(.45), "s", "depth 2, 1 layer frozen"),
            ("depth 3", RAMP(.75), "D", "depth 3, 2 layers frozen"), ("depth 4", RAMP(1.), "h", "depth 4, 3 layers frozen"),
            ("depth 30", "#762a83", "*", r"depth 30, 29 frozen ($\times\sqrt{6}$)"), ("reservoir", "#b2182b", "v", "reservoir (LSTM frozen)"),
            ("LSTM trained", LSTM1_COLOR, "^", "LSTM, fully trained")]
    col = {f: c for f, c, _, _ in fams}
    mk = {f: m for f, _, m, _ in fams}
    ms = lambda f, s = 4.2: s * 1.45 if f == "depth 30" else s
    thick = lambda n: .3 + .55 * np.log10(n / 10)
    grey, lgrey = "#5f5e5a", "#b4b2a9"

    def refs(ax, labels = False, x = 1.02):
        for (lo, hi), c_ in ((BANDS["naive"], NAIVE_COLOR), (BANDS["joint"], JOINT_COLOR)):
            ax.axhspan(lo, hi, color = c_, alpha = .15, lw = 0, zorder = 0)
        ax.axhspan(*LINEAR_READOUT, facecolor = "none", edgecolor = "#a8a6a0", hatch = "////", lw = 0, zorder = 0)
        ax.axhline(.1, color = lgrey, ls = ":", lw = .9, zorder = 0)
        if labels:
            tr = ax.get_yaxis_transform()
            for y, t_ in ((np.mean(BANDS["joint"]), "joint Bayes"), (np.mean(LINEAR_READOUT), "linear readout\nof the counts"),
                          (np.mean(BANDS["naive"]) - .01, "naive Bayes"), (.1, "chance")):
                ax.text(x, y, t_, transform = tr, fontsize = fs["tick"] - 1.5, color = grey, va = "center", ha = "left",
                        clip_on = False, linespacing = .95)

    def clean(ax, ylim = (.05, .8)):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length = 2.5, pad = 2)
        ax.set_ylim(*ylim)

    def title2(ax, letter, name_, formula, x = .5):
        ax.annotate(formula, xy = (x, 1), xycoords = "axes fraction", xytext = (0, 4), textcoords = "offset points",
                    fontsize = fs["tick"] - .5, ha = "center", va = "baseline", color = "#3d3d3a")
        ax.annotate(name_, xy = (x, 1), xycoords = "axes fraction", xytext = (0, 16), textcoords = "offset points",
                    fontsize = fs["label"], ha = "center", va = "baseline")
        ax.annotate(r"$\mathbf{%s}$" % letter, xy = (0, 1), xycoords = "axes fraction", xytext = (-28, 16),
                    textcoords = "offset points", fontsize = fs["label"], ha = "left", va = "baseline")

    W, H = 13.9, 6.75
    fig = plt.figure(figsize = (W, H))
    ax_ = lambda x, y, w, h, **k: fig.add_axes([x / W, y / H, w / W, h / H], **k)
    Y1, Y2, PH = 3.85, .62, 2.25                                  # row bottoms, panel height (in)

    # ------------------------------------------------------------------------------- (a) infinite training against width
    ax = ax_(.62, Y1, 3.95, PH)
    axi = ax_(4.62, Y1, .62, PH, sharey = ax)
    big = [1e4, 1e5, 1e6, 1e7, 1e8]
    for k, (f, c_, m_, _) in enumerate(fams):
        L = J["families"][f]["limits"]
        Ns = np.array([np.inf if s == "inf" else float(s) for s in L["N"]])
        pt, lo, hi = (np.array([np.nan if v is None else v for v in L[q]], float) for q in ("point", "lo", "hi"))
        elo, ehi = (np.array([np.nan if v is None else v for v in L[q]], float) for q in ("envelope_lo", "envelope_hi"))
        meas = np.array(J["families"][f]["measured_widths"], float)
        fin = np.isfinite(Ns) & (Ns >= meas.min())
        trained = fin & np.isin(Ns, meas)
        last = meas.max()
        ax.plot(Ns[fin & (Ns <= last)], pt[fin & (Ns <= last)], color = c_, lw = 1.3, zorder = 3)
        ax.plot(Ns[fin & (Ns >= last)], pt[fin & (Ns >= last)], color = c_, lw = 1.1, ls = (0, (3, 1.6)), zorder = 3)
        ax.plot(Ns[trained], pt[trained], ls = "none", marker = m_, ms = ms(f), color = c_, mec = "white", mew = .4, zorder = 4)
        dx = 10 ** (.034 * (k - 3))                               # side by side at the extrapolated widths
        for n in big:
            i = np.where(Ns == n)[0]
            if not len(i):
                continue
            i = i[0]
            x = n * dx
            ax.plot([x, x], [elo[i], ehi[i]], color = c_, lw = .7, alpha = .8, zorder = 2, solid_capstyle = "butt")
            ax.plot([x, x], [lo[i], hi[i]], color = c_, lw = 2.2, alpha = .85, zorder = 3, solid_capstyle = "butt")
            if n > last:
                ax.plot([x], [pt[i]], marker = m_, ms = ms(f) * .9, color = c_, mfc = "white", mew = .8, zorder = 5)
        i = np.where(np.isinf(Ns))[0][0]                          # the N = inf column
        x = -.6 + 1.2 * k / (len(fams) - 1)
        axi.plot([x, x], [elo[i], ehi[i]], color = c_, lw = .8, zorder = 2, solid_capstyle = "butt")
        axi.plot([x, x], [lo[i], hi[i]], color = c_, lw = 2.6, alpha = .9, zorder = 3, solid_capstyle = "butt")
        axi.plot([x], [pt[i]], marker = m_, ms = ms(f), color = c_, mfc = "white", mew = .9, zorder = 4)
    ax.plot([1e5, 2e5], [.1, .1], ls = "none", marker = "x", ms = 5, mew = 1.3, color = col["depth 1"], zorder = 6, clip_on = False)
    ax.annotate("depth 1 trained at lr 1e-3:\ndead at $N$ = 10$^5$, 2$\\times$10$^5$", xy = (1.5e5, .11), xytext = (2e3, .155),
                fontsize = fs["tick"] - 1.5, color = col["depth 1"], ha = "center", va = "bottom",
                arrowprops = dict(arrowstyle = "-", color = col["depth 1"], lw = .6, shrinkA = 1, shrinkB = 3))
    ax.set_xscale("log")
    ax.set_xticks([10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8], ["10", "100", "1k", "10k", "100k", "1M", "10M", "100M"])
    ax.set_xticks([], minor = True)
    ax.set_xlim(7, 1.6e8)
    ax.set_xlabel("width $N$", labelpad = 1)
    ax.set_ylabel("accuracy at infinite training", labelpad = 2)
    clean(ax)
    refs(ax)
    clean(axi)
    refs(axi, labels = True, x = 1.06)
    axi.set_xlim(-.85, .85)
    axi.set_xticks([0], [r"$N=\infty$"])
    axi.spines["left"].set_visible(False)
    axi.tick_params(axis = "y", length = 0, labelleft = False)
    title2(ax, "a", "Infinite training, against width",
           r"$a(N,t\!=\!\infty)$: point, 95% bootstrap (thick), forms that fit as well (thin)", x = .55)
    ax.legend(handles = [plt.Line2D([], [], color = c_, marker = m_, ms = ms(f), lw = 1.2, mec = "white", mew = .4, label = lab)
                         for f, c_, m_, lab in fams],
              frameon = False, fontsize = fs["tick"] - 1.6, loc = "lower right", bbox_to_anchor = (1.0, .17),
              handlelength = 1.6, labelspacing = .25, borderaxespad = 0)

    # ------------------------------------------------------------------------------- (b) width against depth, t = inf
    axb = ax_(6.55, Y1, 2.55, PH)
    lN, dd = PM["log10N_fine"].astype(float), PM["depth_fine"].astype(float)
    okd = np.isfinite(dd)
    Z = PM["fine_point_tinf"][okd][:, np.isfinite(lN)]
    lNf, ddf = lN[np.isfinite(lN)], dd[okd]
    cmap = LSC.from_list("acc", ["#f4f3ee", "#c9dcf2", "#7fb0e8", "#2a78d6", "#0d366b"])
    pc = axb.pcolormesh(lNf, ddf, Z, cmap = cmap, vmin = .1, vmax = .77, shading = "nearest", rasterized = True, zorder = 0)
    cs = axb.contour(lNf, ddf, Z, levels = [np.mean(BANDS["naive"]), .75], colors = [NAIVE_COLOR, JOINT_COLOR],
                     linewidths = [1.1, 1.1], linestyles = ["--", "-"], zorder = 2)
    axb.clabel(cs, fmt = {cs.levels[0]: "naive", cs.levels[1]: ".75"}, fontsize = fs["tick"] - 2, inline_spacing = 2)
    for key, c_, lab in (("t20000", "#7fe3ff", r"2$\times$10$^4$"), ("t1e+06", "#ffb000", r"10$^6$"), ("t1e+09", "#e8590c", r"10$^9$")):
        Zt = PM[f"fine_point_{key}"][okd][:, np.isfinite(lN)]          # where nets beat naive Bayes at a finite budget
        ct = axb.contour(lNf, ddf, np.nan_to_num(Zt, nan = .1), levels = [np.mean(BANDS["naive"])], colors = [c_], linewidths = 1.3,
                         linestyles = "-", zorder = 3)
        axb.clabel(ct, fmt = {ct.levels[0]: lab}, fontsize = fs["tick"] - 2.2, inline_spacing = 1, manual = False)
    D_ = np.geomspace(1, 1000, 60)
    for g, ls in ((0, (0, (1, 1.4))), (.5, (0, (4, 1.5))), (1, "-")):
        axb.plot(np.log10(1000 * D_ ** g), D_, color = "white", lw = 1.5, zorder = 3)
        axb.plot(np.log10(1000 * D_ ** g), D_, color = "#3d3d3a", lw = .9, ls = ls, zorder = 4)
    for g, d0 in ((0, 300), (.5, 200), (1, 60)):
        axb.text(np.log10(1000 * d0 ** g) + .1, d0, rf"$\gamma$={g:g}", fontsize = fs["tick"] - 2, color = "#3d3d3a", va = "center",
                 ha = "left", zorder = 5, path_effects = [__import__("matplotlib.patheffects", fromlist = ["x"]).withStroke(linewidth = 2, foreground = "white")])
    meas = [(1, n) for n in FROZEN_WIDTH_SERIES] + [(30, n) for n in FROZEN_WIDTH_SERIES] + [(10, 100), (100, 50), (100, 100), (100, 200), (100, 400)]
    axb.plot([np.log10(n) for _, n in meas], [d for d, _ in meas], ls = "none", marker = "o", ms = 2.2, color = "#1f1f1e", mec = "white",
             mew = .3, zorder = 6)
    axb.set_yscale("log")
    axb.set_ylim(1, 1000)
    axb.set_xlim(1, 8)
    axb.set_xticks([1, 2, 3, 4, 5, 6, 7, 8], ["10", "", "1k", "", "100k", "", "10M", ""])
    axb.set_xlabel("width $N$", labelpad = 1)
    axb.set_ylabel("depth $D$", labelpad = 1)
    axb.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    axb.tick_params(length = 2.5, pad = 2)
    title2(axb, "b", r"Width against depth ($\times\sqrt{6}$ stack)",
           r"colour: $t=\infty$; lines: naive level at $t$ = 2$\times$10$^4$, 10$^6$, 10$^9$")
    cax = ax_(9.17, Y1 + .25, .09, PH - .5)
    cb = fig.colorbar(pc, cax = cax, ticks = [.1, .3, .45, .6, .766])
    cb.ax.set_yticklabels(["0.10", "0.30", "0.45", "0.60", "0.77"], fontsize = fs["tick"] - 2)
    cb.outline.set_linewidth(.5)
    cb.ax.tick_params(length = 2)

    # ------------------------------------------------------------------------------- (c) very wide nets against depth
    axc = ax_(10.25, Y1, 2.5, PH)
    dep = PM["depth"].astype(float)
    j8 = int(np.where(PM["log10N"] == 8)[0][0])
    fin_d = np.isfinite(dep)
    budgets = [("t20000", r"$t$ = 2$\times$10$^4$ (trained)", "#c6dbef"), ("t1e+06", r"$t$ = 10$^6$", "#7fb0e8"),
               ("t1e+09", r"$t$ = 10$^9$", "#2a78d6"), ("tinf", r"$t=\infty$", "#0d366b")]
    for key, lab, c_ in budgets:
        y = PM[f"point_{key}"][fin_d, j8]
        axc.fill_between(dep[fin_d], PM[f"lo95_{key}"][fin_d, j8], PM[f"hi95_{key}"][fin_d, j8], color = c_, alpha = .3, lw = 0, zorder = 1)
        axc.plot(dep[fin_d], y, color = c_, lw = 1.6, zorder = 3, label = lab)
    axc.plot(dep[fin_d], PM["env_lo_tinf"][fin_d, j8], color = "#0d366b", lw = .8, ls = (0, (2, 1.5)), zorder = 3,
             label = r"$t=\infty$, lowest form that fits")
    xinf = 3000
    axc.plot([xinf], [.1], marker = "o", ms = 4, color = grey, zorder = 5, clip_on = False)
    axc.plot([xinf], [float(PM["B_any_scalar"])], marker = "_", ms = 8, mew = 1.4, color = grey, zorder = 5, clip_on = False)
    axc.annotate("$D\\to\\infty$ before $N$: chance at any\nfinite $N$, one channel $\\leq$ .21", xy = (xinf, .105), xytext = (1.1, .13),
                 fontsize = fs["tick"] - 2, color = grey, ha = "left", va = "bottom",
                 arrowprops = dict(arrowstyle = "-", color = grey, lw = .5, shrinkA = 1, shrinkB = 3, connectionstyle = "arc3,rad=-0.15"))
    axc.set_xscale("log")
    axc.set_xlim(.85, 4500)
    axc.set_xticks([1, 10, 100, 1000, xinf], ["1", "10", "100", "1000", r"$\infty$"])
    axc.set_xticks([], minor = True)
    axc.set_xlabel("depth $D$", labelpad = 1)
    clean(axc)
    refs(axc, labels = True)
    axc.tick_params(axis = "y", labelleft = True)
    axc.legend(frameon = False, fontsize = fs["tick"] - 2, loc = "lower left", bbox_to_anchor = (0, .2), handlelength = 1.5,
               labelspacing = .2, borderaxespad = 0)
    title2(axc, "c", r"Very wide ($N$ = 10$^8$) against depth", r"$a(10^8, D, t)$ by training budget $t$; median, 95%")

    # ------------------------------------------------------------------------------- (d-g) curves to convergence
    T = np.geomspace(1e3, 1e16, 240)
    panels = [("d", "LSTMs: frozen (red), trained (orange)", ("reservoir", "LSTM trained")),
              ("e", "depth 1 (readin + readout)", ("depth 1",)),
              ("f", r"depth 30, 29 frozen ($\times\sqrt{6}$)", ("depth 30",)),
              ("g", r"depth 100, 99 frozen ($\times\sqrt{6}$)", ())]
    PW, GAP = 2.75, .47
    curve_axes = []
    for k, (letter, ttl, fl) in enumerate(panels):
        a_ = ax_(.62 + k * (PW + GAP), Y2, PW, PH)
        curve_axes.append(a_)
        a_.axvspan(1e3, 2e4, color = "#e8e7e2", lw = 0, zorder = 0)
        a_.axvspan(2e4, 8e4, color = "#f3f2ee", lw = 0, zorder = 0)
        for f in fl:
            meas_n = J["families"][f]["measured_widths"]
            stems = {n: s for f_, n, s in extrapolation_runs() if f_ == f}
            for n in meas_n:
                c = load_curve(os.path.join(OUT, stems[n] + "_curve.npz"))
                if c is None:
                    continue
                m = c["ep"] >= 1000
                a_.plot(c["ep"][m], c["acc"][m], color = col[f], lw = thick(n), alpha = .9, zorder = 3)
                y = np.ravel(X.predict(f, float(n), T))
                a_.plot(T[T >= 2e4], y[T >= 2e4], color = col[f], lw = max(thick(n) * .6, .35), ls = (0, (3, 1.4)), alpha = .85, zorder = 2)
                lim_ = float(np.ravel(X.predict(f, float(n), np.inf))[0])
                a_.plot([1.6e16], [lim_], marker = "<", ms = 3, color = col[f], clip_on = False, zorder = 4)
        if letter in ("e", "f"):                                  # the width-20000 run, continued, against the prediction at 40k
            f = fl[0]
            stem = {"depth 1": "d1_N20000_p1_lr0.001_rep0", "depth 30": "tff1_N20000_p1_lr0.001_rep0"}[f]
            cc = load_curve(os.path.join(OUT, "widths", "continued", stem + "_curve.npz"))
            p, lo, hi = (float(np.ravel(v)[0]) for v in X.interval(f, 2e4, 4e4))
            a_.plot([4e4, 4e4], [lo, hi], color = "#1f1f1e", lw = 1.1, zorder = 6)
            a_.plot([4e4], [p], marker = "D", ms = 3.6, color = "white", mec = "#1f1f1e", mew = .9, zorder = 7)
            if cc is not None and len(cc["ep"]) > 2:
                a_.plot(cc["ep"][2:], cc["acc"][2:], ls = "none", marker = "o", ms = 1.9, color = "#1f1f1e", zorder = 7)
                if f == "depth 30" and cc["acc"][2:].min() < .42:
                    i = int(np.argmin(cc["acc"][2:])) + 2
                    a_.annotate("dip after resuming\n(being checked)", xy = (cc["ep"][i], cc["acc"][i]), xytext = (3e5, .2),
                                fontsize = fs["tick"] - 2.2, color = "#1f1f1e", ha = "left", va = "center",
                                arrowprops = dict(arrowstyle = "-", color = "#1f1f1e", lw = .5, shrinkA = 1, shrinkB = 2))
        if letter == "g":                                         # depth 100: CPU replicas and the depth model
            for i, n in enumerate(D100["N"]):
                n = float(n)
                a_.fill_between(D100["t"], D100["lo95"][i], D100["hi95"][i], color = "#3d6fb6", alpha = .12, lw = 0, zorder = 1)
                a_.plot(D100["t"], D100["median"][i], color = "#3d6fb6", lw = thick(n) * .8, zorder = 2)
            a_.plot(D100["t"], D100["env_lo"][-1], color = "#3d6fb6", lw = .8, ls = (0, (2, 1.5)), zorder = 2)
            for key in [k_ for k_ in D100.files if k_.startswith("measured_") and k_.endswith("_acc")]:
                ep = D100[key.replace("_acc", "_ep")]
                a_.plot(ep, D100[key], ls = "none", marker = "o", ms = 1.7, color = "#c2185b", zorder = 4)
            a_.text(.03, .97, "dots: CPU replicas, $N$ = 50-400\nlines: model at $N$ = 100, 10$^3$, 5$\\times$10$^3$, 2$\\times$10$^4$\n"
                    "(median, 95%); dashed: lowest form\nthat fits ($N$ = 2$\\times$10$^4$); converges\nonly after $\\sim$10$^{28}$ updates",
                    transform = a_.transAxes, fontsize = fs["tick"] - 2.2, color = "#3d3d3a", ha = "left", va = "top", linespacing = 1.1)
        a_.set_xscale("log")
        a_.set_xlim(1e3, 1.6e16)
        a_.set_xticks([1e3, 1e6, 1e9, 1e12, 1e15], ["10$^3$", "10$^6$", "10$^9$", "10$^{12}$", "10$^{15}$"])
        a_.set_xticks([], minor = True)
        a_.set_xlabel("training updates (line width = $N$)", labelpad = 1)
        clean(a_)
        refs(a_, labels = k == 3)
        if k == 0:
            a_.set_ylabel("accuracy", labelpad = 2)
        else:
            a_.tick_params(axis = "y", labelleft = False)
        title2(a_, letter, ttl, {"d": r"converge within $\sim$10$^6$ updates",
                                 "e": r"converges after $\sim$10$^9$ updates (wide)",
                                 "f": r"converges after $\sim$10$^{15}$ updates",
                                 "g": r"$N\leq$ 400 trained on CPU; wider: depth model"}[letter])
    a_ = curve_axes[0]
    a_.text(1.15e3, .065, "trained", fontsize = fs["tick"] - 2.4, color = grey, va = "bottom", rotation = 90)
    a_.text(2.2e4, .065, "backtested", fontsize = fs["tick"] - 2.4, color = grey, va = "bottom", rotation = 90)
    a_.text(1.2e5, .065, "model only", fontsize = fs["tick"] - 2.4, color = grey, va = "bottom")
    curve_axes[1].legend(handles = [plt.Line2D([], [], color = "#1f1f1e", marker = "o", ms = 2.2, ls = "none",
                                               label = "$N$ = 20000 continued (10-21)"),
                                    plt.Line2D([], [], color = "#1f1f1e", marker = "D", ms = 3.6, mfc = "white", mew = .9, lw = 1.1,
                                               label = "model's prediction at 40k, 95%")],
                         frameon = False, fontsize = fs["tick"] - 2, loc = "lower right", bbox_to_anchor = (1, .12), borderaxespad = 0)
    fig.text(.62 / W, .05 / H,
             "Every value at $t=\\infty$ or beyond $2\\times10^4$ updates is the model's continuation; the backtests checked it to 4x the "
             "trained range (intervals there covered 77-88% at nominal 95%: widen them about 1.3-1.5x). Single seed per width. "
             "Depths 2-4: PyTorch's default init; depth 30 and 100: $\\times\\sqrt{6}$.",
             fontsize = fs["tick"] - 1.6, color = "#5f5e5a", ha = "left", va = "bottom")
    fig.savefig(fig_path(name), dpi = 200)
    print(fig_path(name))


if __name__ == "__main__":
    if "--collect" in sys.argv:
        collect()
        collect_disagreement()
    if "--collect-disagreement" in sys.argv:
        collect_disagreement()
    if "--collect-missing" in sys.argv:                         # evaluate only nets not yet in DISAGREE
        collect_disagreement(only_missing = True)
    curves_figure()
    power1_scatter_figure()
    frozen_figure()
    widths_figure()
    widths_frozen_figure()
    heatmaps_figure()
    relations_figure()
    disagreement_figure()
    frozen_widths_figures()
