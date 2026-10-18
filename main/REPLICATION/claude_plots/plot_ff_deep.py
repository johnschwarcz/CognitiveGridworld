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
    The time-injected nets (tff, tffs) left the analysis (John, 2026-10-16)."""
    return f"widths/tff1{'' if n == 1000 else f'_N{n}'}_p1_lr0.001_rep0"


def reservoir_stem(n):
    return f"widths/lstm{'' if n == 1000 else f'_N{n}'}_p1_lr0.001_reservoir_rep0"


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


def frozen_widths_figure(running = (), running_d30 = False):
    """FF with only readin and readout trained (every N x N layer frozen) at depths 1-3, against width (train_ff_deep.py
    frozen_widths, reservoir_compare, matched_a, d4_w20000; depth 1 has no N x N layer, so it is the ordinary depth 1),
    beside the reservoirs (an LSTM with only readin and readout trained) at RESERVOIR_WIDTHS. Every net here, FF and
    reservoir, used loss power 1 and saw the same data: 20k episodes x batch 8000, seed 0 (one world); finished runs only.
    Reservoirs from N=1000 up ran on the wide path (fp16 frozen weights, deduplicating frozen LSTM, micro-batches with
    accumulated gradients): the same model and updates, fp64-tested. The older sanity_default_reps reservoirs (power 0.5;
    width 10000 at batch 2000) are left out. Top: (a) KL(joint || net) and (b) accuracy at 20k episodes (mean of the last
    3 checkpoints). Bottom: final-step test accuracy through training, line thickness growing with width; (c-e) one panel
    per depth (depth 4 dropped for simplicity, John 2026-10-18), (f) depth 30 with all 29 N x N layers frozen (purple,
    tff1_stem): its frozen layers are the default draws
    x sqrt(6), Kaiming-uniform, because at the default scale 29 layers erase the input (3e-11 of layer 30 depends on it at
    init; the Oct 12 deep30 runs sat at chance), (g) the LSTMs: frozen (reservoirs, red) and fully trained (LSTM_STEMS,
    orange, the best case). Widths 10-50 come from train_ff_deep.py small_widths. running: reservoir widths still
    training to include anyway, at their latest 3 checkpoints (open markers, dotted link; their curve ends where training
    has got to); running_d30: the same for every depth-30 width past its first 500 episodes."""
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    depths = (1, 2, 3)
    shade = {1: RAMP(.15), 2: RAMP(.45), 3: RAMP(.75), 4: RAMP(1.)}
    names = {1: "depth 1 (readin + readout)", 2: "depth 2, its layer frozen", 3: "depth 3, both layers frozen",
             4: "depth 4, all 3 layers frozen"}
    mark = {1: "o", 2: "s", 3: "D", 4: "h"}                    # neighbouring depths nearly coincide at small widths
    res_col, d30_col = "#b2182b", "#762a83"

    def series(stem_of, widths_, run_ok):
        """(width, KL, accuracy, finished, last episode) per run, and its curve: finished runs, plus the running ones that
        run_ok admits (at their latest 3 checkpoints)."""
        rows, crv = [], {}
        for n in widths_:
            c = load_curve(os.path.join(OUT, stem_of(n) + "_curve.npz"))
            done = os.path.exists(os.path.join(OUT, stem_of(n) + "_summary.json"))
            if c is not None and (done or run_ok(n, c)):
                rows.append((n, float(np.mean(c["kl_joint"][-3:])), float(np.mean(c["acc"][-3:])), done, int(c["ep"][-1])))
                crv[n] = (c["ep"] / 1000, c["acc"], done)
        return rows, crv
    reservoir, curves = series(reservoir_stem, RESERVOIR_WIDTHS, lambda n, c: n in running)
    d30, d30_curves = series(tff1_stem, FROZEN_WIDTH_SERIES, lambda n, c: running_d30 and c["ep"][-1] >= 500)   # not a stale start
    lstm, lstm_curves = series(lambda n: LSTM_STEMS[n], sorted(LSTM_STEMS), lambda n, c: False)
    base = baselines()
    widths = sorted(set(FROZEN_WIDTH_SERIES) | set(curves) | set(LSTM_STEMS))
    thick = lambda n: .3 + .75 * np.log10(n / 10)               # line width encodes the network width

    fig = plt.figure(figsize = (18.5, 9.2))
    top = fig.add_gridspec(1, 2, left = .05, right = .95, top = .95, bottom = .58, wspace = .25)
    bottom = fig.add_gridspec(1, 5, left = .05, right = .95, top = .47, bottom = .07, wspace = .08)
    ax_kl, ax_acc = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    curve_axes = [fig.add_subplot(bottom[0, i]) for i in range(5)]

    # (a, b) against width
    for d in depths:
        rows = []
        for n in FROZEN_WIDTH_SERIES:
            c = load_curve(os.path.join(OUT, frozen_width_stem(d, n) + "_curve.npz"))
            if os.path.exists(os.path.join(OUT, frozen_width_stem(d, n) + "_summary.json")) and c is not None:
                rows.append((n, float(np.mean(c["kl_joint"][-3:])), float(np.mean(c["acc"][-3:]))))
        if rows:
            ax_kl.plot([r[0] for r in rows], [r[1] for r in rows], color = shade[d], marker = mark[d], ms = 5.5, lw = 1.5, label = names[d])
            ax_acc.plot([r[0] for r in rows], [r[2] for r in rows], color = shade[d], marker = mark[d], ms = 5.5, lw = 1.5, label = names[d])
    def draw_family(rows, col, mk, name):
        """Finished widths as a solid line; running ones as open markers on one dotted path from the last finished width."""
        handles = []
        fin, run = [r for r in rows if r[3]], [r for r in rows if not r[3]]
        for ax, i in ((ax_kl, 1), (ax_acc, 2)):
            ms = 8.5 if mk == "*" else 6                     # a star reads small at the others' size
            h = ax.plot([r[0] for r in fin], [r[i] for r in fin], color = col, marker = mk, ms = ms, lw = 1.5, label = name) if fin else []
            if run:
                path = [q for q in fin if q[0] < run[0][0]][-1:] + run
                ax.plot([q[0] for q in path], [q[i] for q in path], color = col, lw = 1.2, ls = ":")
                eps = sorted({r[4] for r in run})
                when = f"episode {eps[0]:,}" if len(eps) == 1 else f"episodes {eps[0]:,}-{eps[-1]:,}"
                who = f" N = {run[0][0]:g}" if len(run) == 1 else ""
                h = h + ax.plot([r[0] for r in run], [r[i] for r in run], ls = "none", marker = mk, ms = ms, color = col,
                                mfc = "white", label = f"{name}{who},\n    still training ({when})")
            if ax is ax_kl:
                handles += h
        return handles
    res_line = draw_family(sorted(reservoir), res_col, "v", "reservoir (LSTM frozen, power 1)")
    draw_family(sorted(d30), d30_col, "*", r"depth 30, all 29 layers frozen ($\times\sqrt{6}$)")   # in (b)'s depth legend
    res_line += draw_family(lstm, LSTM1_COLOR, "^", "LSTM, fully trained (power 1)")
    ax_kl.set_yscale("log")
    ax_kl.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax_kl.set_ylabel(KL("joint", "net") + ", final step (joint Bayes: 0)")
    ax_kl.set_title("(a) distance from joint Bayes", fontsize = fs["label"])
    ax_acc.set_ylabel("final-step accuracy")
    ax_acc.set_title("(b) accuracy", fontsize = fs["label"])
    for ax in (ax_kl, ax_acc):
        ax.set_xscale("log")
        ax.set_xticks(widths, [f"{n:g}" if n not in (400, 600) else "" for n in widths])   # LSTM-only widths: no room
        ax.set_xticks([], minor = True)
        ax.set_xlim(7, 28000)
        ax.set_xlabel("width N")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax_kl.legend(handles = res_line, frameon = False, fontsize = fs["tick"] - 1.5, loc = "lower left")    # empty there
    ax_acc.legend(handles = [l for l in ax_acc.get_lines() if l.get_label().startswith("depth")], frameon = False,
                  fontsize = fs["tick"] - 1.5, loc = "lower right", bbox_to_anchor = (1, .05))     # clear of the chance line

    # (c-f) training curves, one panel per depth; (g) the reservoirs, in a panel of their own
    for i, (ax, d) in enumerate(zip(curve_axes, depths)):
        for n in FROZEN_WIDTH_SERIES:
            c = load_curve(os.path.join(OUT, frozen_width_stem(d, n) + "_curve.npz"))
            if c is not None and os.path.exists(os.path.join(OUT, frozen_width_stem(d, n) + "_summary.json")):   # finished runs only
                ax.plot(c["ep"] / 1000, c["acc"], color = shade[d], lw = thick(n), alpha = .9)
        ax.set_title(f"({'cdef'[i]}) {names[d]}", fontsize = fs["label"], color = shade[d] if d > 1 else RAMP(.35))
    ax = curve_axes[3]                                           # depth 30, all 29 layers frozen x sqrt(6)
    for n, (x, y, done) in sorted(d30_curves.items()):
        ax.plot(x, y, color = d30_col, lw = thick(n), alpha = .85)
    ax.set_title(r"(f) depth 30, all 29 layers frozen ($\times\sqrt{6}$)", fontsize = fs["label"], color = d30_col)
    ax = curve_axes[4]                                           # the LSTM frozen (reservoir) and fully trained, power 1
    for crv, col in ((curves, res_col), (lstm_curves, LSTM1_COLOR)):
        for n, (x, y, done) in sorted(crv.items()):
            ax.plot(x, y, color = col, lw = thick(n), alpha = .85)
    still = [n for n, (_, _, done) in curves.items() if not done]
    ax.legend(handles = [plt.Line2D([], [], color = LSTM1_COLOR, lw = 1.5, label = "fully trained"),
                         plt.Line2D([], [], color = res_col, lw = 1.5, label = "frozen (reservoir)" +
                                    "".join(f"; N = {n:g} running" for n in still))],
              frameon = False, fontsize = fs["tick"] - 2, loc = "lower right", bbox_to_anchor = (1, .07))
    ax.set_title("(g) LSTM, frozen and fully trained", fontsize = fs["label"], color = res_col)

    for i, ax in enumerate(curve_axes):
        ax.set_xlim(0, 20.5)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_ylim(.05, .8)
        ax.set_xlabel("training episodes (thousands)")
        if i == 0:
            ax.set_ylabel("final-step test accuracy")
        else:
            ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    curve_axes[0].legend(handles = [plt.Line2D([], [], color = "#5f5e5a", lw = thick(n), label = f"width {n:g}") for n in widths],
                         frameon = False, fontsize = fs["tick"] - 2.5, loc = "upper left", bbox_to_anchor = (0, .95), ncol = 3,
                         columnspacing = .8, handlelength = 1.6)     # above the curves: the small widths run along the bottom

    ref_lines(ax_kl, [(base["kl_naive"], "naive Bayes", NAIVE_COLOR, "--"), (base.get("kl_joint_uniform", 1.76), "uniform", "#b4b2a9", ":")], fs)
    lo, hi = ax_kl.get_ylim()                                    # after the reference lines, which can extend the range
    ticks = [v for v in (.01, .02, .03, .05, .1, .2, .3, .5, .7, 1, 2, 3, 5, 7, 10) if lo <= v <= hi]   # within the range
    ax_kl.set_yticks(ticks, [f"{v:g}" for v in ticks])
    acc_refs = [(base["acc_naive"], "naive Bayes", NAIVE_COLOR, "--"), (base["acc_joint"], "joint Bayes", JOINT_COLOR, ":"),
                (.1, "chance", "#b4b2a9", ":")]
    ref_lines(ax_acc, acc_refs, fs)
    for i, ax in enumerate(curve_axes):
        if i == len(curve_axes) - 1:
            ref_lines(ax, acc_refs, fs)                          # labelled once, at the right of the row
        else:
            for y, _, col, ls in acc_refs:
                ax.axhline(y, color = col, ls = ls, lw = 1, zorder = 1)
    fig.savefig(fig_path("ff_deep_frozen_widths.png"), dpi = 150)
    print(fig_path("ff_deep_frozen_widths.png"))


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
    frozen_widths_figure()
