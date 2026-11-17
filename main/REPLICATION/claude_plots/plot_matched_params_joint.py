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
Echo N = 10k (John 11-08): the batch-8,000 runs in every version (echo_N10000_b8k, joint_echo_N10000_b8k here; joint_echo_N10000_b8k_jl
from John's other machine, its test files copied in, no weights: NO_WEIGHTS), in place of the batch-2,000 ones.
Best linear rule (John 11-09: "add this best linear to the log-log panels"): grey dotted on the ESN row, the marginal
output's best rule linear in the counts under cross-entropy (one optimum; fitted per environment, .5049 vs naive .4495 on
the same episodes, verify_2026-10-25/ceiling): drawn LIN_CE_EDGE = .0554 below the naive bound's gap, so the paired edge
is exact. For joint output the best linear rule is optimal (gap 0, off the log axis): a note says so.
Submission version, _n1 only (John 11-10: "remove notes, make sure text is large enough to be read, remove the best linear
count and indicate 'trained' and 'echo' in a way that doesnt need repeating every line of the legend ... 2 columns 'width'
and 'trainable params', and put the network title in the y-axis"): drawn at its print size, SUB_SIZE (5.5 in, the paper's
text width, so it goes in at \\textwidth unscaled), type SUB_FS points; no best-linear line, no notes; "Trained" / "Echo" as
row titles beside one shared y label. Colours evenly spaced by width rank along Blues / Reds (the exploratory versions'
N = 300 / 400 / 500 blues and N = 200 / 500 reds could not be told apart in print). Later the same day (John 11-10: "since
the trainable params differ from column to column, let's put the width legend in the first column and the trainable
params legend in each column, let's also make the figure a bit wider and ... the trained LSTM only at widths 50, 100,
200, 400, 1k"): a key in every panel's empty lower left, the first column's with the widths, every one with that
column's trainable parameters (marginal output in the first, joint in the other two: the same networks, so the same
counts); 4.6 in -> 4.0 in tall, wider at the fixed print width; Trained N = 150 / 300 / 500 left out (SUB_DROP). Then
(John 11-10: "let's make the figure much wider so trainable params can be 1 line and add width 150 fully trained"): 8.0 x
4.0 in, the headers on one line, the numbers centred under them; Trained N = 150 back. At 8 in it is wider than the text
width: placed at \\textwidth it scales by 5.5 / 8 (7 pt type -> 4.8 pt); full size needs a rotated page (9 in). Then (John
11-10: "let's add width 300 as well, make the figure a bit wider, and let's lower the ylim a bit so we can move trainable
params to the bottom right and keep width on the left. Let's also change 'goal' to 'marginal' and capitalize all Joint
and Marginal"): Trained N = 300 back (only N = 500 left out); 8.5 x 4.0 in (at \\textwidth 7 pt -> 4.5 pt); the y-axis
from the data's minimum / 4 (.6 decade lower); the widths alone at the first column's lower left, each panel's
trainable parameters at its bottom, as far right as the curves leave room: the lower right corner in four panels, but
not in Trained marginal / joint supervision, where the widest networks' curves end there (no lower limit frees it: at
the minimum / 100 they still cross it); titles Marginal / Joint output, Marginal / Joint supervision. Then (John 11-10:
"can we get the y-axis down to 0.0001 to fix the legend to the right?" -- not with one-column keys: two Trained panels
still crossed -- "or maybe we can split the trained network legend lists into 2 columns?"): the y-axis from 1e-4
(a fixed floor), the Trained keys in two blocks: every trainable-parameter key at the lower right corner but
Trained joint supervision's, just left of the widest network's dip at 20k (to ~6e-4; even at 5e-5 it crosses the key).
Then (John 11-10: "can you also please make another version without the fully trained 1K?"): _n1_no1k.png, the same
without the Trained N = 1k. Then (John 11-10: "Maybe for the legends we can give it a title 'Widths / Trainable params'
and then show each as 'number / number'? This way we can do a single column legend on the left of each panel. Also
let's change 'output' to 'belief updating'"): one key per panel at its lower left, its own networks as width / trainable
parameters, the slashes aligned; titles Marginal / Joint belief updating. Then (John 11-10: "let's bolden Joint and Marginal in the
titles"): mathtext bold, Computer Modern bold as the row titles. Then (John 11-10: "save the figure as a pdf and svg
titled param_control_b"): the same drawing also as param_control_b.pdf (TrueType fonts embedded) and .svg (text as paths).
Then (John 11-10: "also adjust the ylims if you can"): _n1_no1k.png from 3e-3, the highest floor that clears the keys
(4e-3 crosses one; its lowest point .0064, joint supervision N = 400 at 34k); _n1.png stays at 1e-4: its Trained N = 1k
dips to 6e-4 (joint supervision, 21k) and crosses a key's header at every floor tried from 2e-4 to 2e-3.
Echo N = 3k (John 11-11: "a new batch of 3K ESNs to include"): echo_N3000_b8k, joint_echo_N3000_b8k and
joint_echo_N3000_b8k_jl in the three Echo panels of every version; from John's other machine (his
Downloads/N3000_b8k_test_accuracies: test files, run.json and progress.log, copied in unchanged; no weights:
NO_WEIGHTS), batch 8,000 in slices, seed 20261029, 5 tests per point as the other training-dynamics runs, finished 11-11.
Submission version's keys and y-axis (John 11-11: "make the 'trainable params' 2 lines and reduce the shared ylim as much
as possible"; 11-13: "move 'Widths' down so it's on the second line"): the keys' header on two lines, 'Trainable' over
'Widths / params', and no fixed floor: the shared y-axis ends SUB_YPAD points beyond the data's extremes, its lower end
dropping further only while a curve passes within 2 pt of a key's text. _n1_no1k.png and the PDF: .0056 to .77 (2.1
decades; before .003 to .84, 2.4), both ends set by the data (the lowest point .0064, no key in a curve's way); _n1.png:
.00027 to .83 (before 1e-4 to .84), the floor set by the Trained N = 1k's curve beside the header of the
joint-supervision key.
-> main/REPLICATION/figure_bin/compare_matched_params_loglog_joint_panels.png, _n1.png, _n1_no1k.png, _no500_no1k.png,
param_control_b.pdf, param_control_b.svg (= _n1_no1k.png; John 11-10: "remove the 1k fully trained from the pdf and svg")
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
import matplotlib.ticker as mticker                                               # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from matplotlib.transforms import Affine2D, ScaledTranslation                     # noqa: E402
import compare_matched_params as cmp                                              # noqa: E402
import plot_matched_params_new as pm                                              # noqa: E402

PANELS = (("Joint output, goal supervision", ("joint_echo_N200", "joint_echo_N500", "joint_echo", "joint_echo_N2000", "joint_echo_N3000_b8k", "joint_echo_N10000_b8k"),
           ("joint_trained_N50", "joint_trained_N100", "joint_trained_N150", "joint_trained_N200", "joint_trained_N300", "joint_trained_N400", "joint_trained_N500",
            "joint_trained")),
          ("Joint output, joint supervision", ("joint_echo_N200_jl", "joint_echo_N500_jl", "joint_echo_jl", "joint_echo_N2000_jl", "joint_echo_N3000_b8k_jl", "joint_echo_N10000_b8k_jl"),
           ("joint_trained_N50_jl", "joint_trained_N100_jl", "joint_trained_N150_jl", "joint_trained_N200_jl", "joint_trained_N300_jl", "joint_trained_N400_jl", "joint_trained_N500_jl",
            "joint_trained_jl")))
C_N300, C_N400, C_N500 = "#2368a3", "#1f5e9a", "#1a5490"   # between the N = 200's blue and the N = 1k's navy
C_N150 = "#5fa2cb"                                    # between the N = 100's light blue and the N = 200's blue
C_E200, C_E500 = "#fddcd3", "#fcc9bf"                 # paler reds: fewer trainable parameters than the Echo N = 1k
C_E3K = "#ea6c6b"                                     # between the Echo N = 2k's and 10k's reds: cmp.resolve_colors' ramp
#                                                       (log-spaced in trainable parameters, N = 1k to 10k) at the N = 3k
C_N50 = "#c6dbef"                                     # palest blue: the fewest trainable parameters
C_N1K = "#08306b"                                     # navy: darker than the N = 200's blue, more trainable parameters
LIN_CE_EDGE = .5049 - .4495                           # the best linear-in-counts rule (cross-entropy) over naive, marginal output
GREY = "#555555"
SUB_SIZE = (8.5, 4.)                                  # the submission version, inches (docstring: print size)
SUB_DROP = (500,)                                     # its trained widths left out (John 11-10): 50-400 and 1k drawn
SUB_YPAD = 3.                                         # its shared y-axis ends this many points beyond the data's extremes
SUB_YSTEP = 1.02                                      # (John 11-11); the lower end / SUB_YSTEP while a curve is on a key
SUB_FS = dict(tick=7, key=7, label=8, title=8, row=9)   # its type, points in print
SUB_LW = 1.                                           # its curves and naive bound, points
SUB_RAMPS = (("Blues", .35, 1.), ("Reds", .3, .9))    # its rows' colours (Trained, Echo): evenly spaced by width rank


def count(n):
    """a parameter count as the legends state it"""
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


NO_WEIGHTS = {"joint_echo_N10000_b8k_jl": 118 * 10000 + 100,   # imported test files only (other machine, 11-08): 118 N + 100
              "echo_N3000_b8k": 38 * 3000 + 20,                # the same, 11-11; marginal output: 38 N + 20 (the counts of
              "joint_echo_N3000_b8k": 118 * 3000 + 100,        # every ESN with weights here follow the two formulas exactly)
              "joint_echo_N3000_b8k_jl": 118 * 3000 + 100}


def params(run):
    """trainable parameters (a reservoir's LSTM excluded) from the final weights, or the first checkpoint's while training"""
    if run in NO_WEIGHTS and not glob.glob(os.path.join(pm._dir(run), "*_net.pth")):
        return NO_WEIGHTS[run]
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


def params_of(label):
    """the trainable-parameter count a curve's label states"""
    return re.search(r", ([\d.]+[kM]) trainable", label)[1]


def key_table(ax, rows, head, fs, at=(.03, .02)):
    """a key as a small table with its lower-left corner at `at` (axes fraction): a line in the curve's colour, then
    'width / params', the slashes aligned (widths right-aligned before them, params left-aligned after), under the header
    'head[0] / head[1]' on the same slash (John 11-10), head[1] a text line per line break in it, the last beside head[0]
    and the others above it at the params' left edge (John 11-11: a narrower key; 11-13: head[0] on the last line);
    returns its artists; rows = (colour, width, params), the first on top; offsets in points, so the table keeps its
    print size"""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    pt = Affine2D().scale(1 / 72) + fig.dpi_scale_trans + ScaledTranslation(*at, ax.transAxes)

    def wd(s):
        t = fig.text(0, 0, s, fontsize=fs)
        w = t.get_window_extent(r).width * 72 / fig.dpi
        t.remove()
        return w

    hl, sp, sl = 10, wd("x") * .6, wd("/")                                        # handle length, space, slash width
    xs = max(hl + 4 + max(wd(w) for _, w, _ in rows), wd(head[0]))                # the widths end here, the slash follows
    cols = ((xs, "right"), (xs + sp + sl / 2, "center"), (xs + 2 * sp + sl, "left"))
    dy, base = 1.15 * fs, .25 * fs
    kw = dict(transform=pt, fontsize=fs, va="baseline", clip_on=False)
    arts = []
    for i, (c, w, n) in enumerate(rows):                                          # the widest network last
        y = (len(rows) - 1 - i) * dy + base
        arts.append(ax.add_artist(Line2D([0, hl], [y + .3 * fs] * 2, transform=pt, color=c, lw=SUB_LW,
                                         solid_capstyle="butt", label="_key", clip_on=False)))
        arts += [ax.text(x, y, t, ha=ha, **kw) for (x, ha), t in zip(cols, (w, "/", n))]
    lines = head[1].split("\n")
    low = len(rows) * dy + base + 1                                               # the header's last line
    arts += [ax.text(x, low, t, ha=ha, **kw) for (x, ha), t in zip(cols, (head[0], "/", lines[-1]))]
    arts += [ax.text(cols[2][0], low + i * dy, t, ha="left", **kw) for i, t in enumerate(lines[-2::-1], 1)]
    return arts


def _hits(ax, pad=2, texts=None):
    """texts in ax (the keys, the naive bound's label; or those given) whose box, padded by pad points, a curve passes
    through"""
    r = ax.figure.canvas.get_renderer()
    px = pad * ax.figure.dpi / 72
    boxes = [(t.get_text(), t.get_window_extent(r).padded(px)) for t in (ax.texts if texts is None else texts) if t.get_text()]
    hit = []
    for ln in ax.get_lines():
        if ln.get_label().startswith("_"):
            continue
        xy = ax.transData.transform(np.column_stack([np.asarray(v, float) for v in ln.get_data()]))
        xy = xy[np.isfinite(xy).all(1)]
        f = np.linspace(0, 1, 25)[:, None, None]
        xy = (xy[:-1] * (1 - f) + xy[1:] * f).reshape(-1, 2)
        hit += [s for s, bb in boxes if s not in hit and np.any((xy[:, 0] >= bb.x0) & (xy[:, 0] <= bb.x1)
                                                                  & (xy[:, 1] >= bb.y0) & (xy[:, 1] <= bb.y1))]
    return hit


def _finish_submission(fig, axes, path):
    """the submission version's look (docstring): print-size layout and type, rows' colours, row titles, shared axis
    labels, a key per panel, the tightest shared y-axis that keeps the curves off the keys; checks that no curve crosses a
    text and that every text lies inside the figure"""
    fs = SUB_FS
    W, H = fig.get_size_inches()
    LM, RM, TM, BM, GX, GY = .68, .10, .34, .40, .10, .10                         # margins and gaps, inches
    pw, ph = (W - LM - RM - 2 * GX) / 3, (H - TM - BM - GY) / 2
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.set_position([(LM + j * (pw + GX)) / W, (BM + (1 - i) * (ph + GY)) / H, pw / W, ph / H])
    curves = {ax: [ln for ln in ax.get_lines() if not ln.get_label().startswith("_")] for ax in axes.flat}
    keys = []
    for row, (cmap, lo, hi) in zip(axes, SUB_RAMPS):
        widths = sorted({width_of(ln.get_label()) for ax in row for ln in curves[ax]})
        col = dict(zip(widths, plt.get_cmap(cmap)(np.linspace(lo, hi, len(widths)))))
        for ax in row:
            for ln in curves[ax]:
                w = width_of(ln.get_label())
                ln.set(color=col[w], lw=SUB_LW, zorder=2 + np.log10(w))
        for ax in row:                                                            # each panel its own networks: width /
            n = {width_of(ln.get_label()): params_of(ln.get_label()) for ln in curves[ax]}   # trainable parameters
            keys.append((ax, [(col[w], f"{w // 1000}k" if w >= 1000 else str(w), n[w]) for w in sorted(n)]))
    for ax in axes.flat:
        for ln in ax.get_lines():
            if ln.get_label().startswith("_") and ln.get_linestyle() == "--":   # the naive bound
                ln.set_linewidth(SUB_LW)
        for t in list(ax.texts):
            if t.get_text() == "Naive bound":                                     # pm._style's label on the line, at the left:
                t.remove()                                                        # curves cross it there
        ax.set(xlabel="", ylabel="", title="")
        ax.xaxis.set_major_locator(mticker.FixedLocator((1e3, 2e3, 5e3, 1e4, 2e4, 4e4)))
        ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1., 3.)))
        ax.tick_params(which="major", labelsize=fs["tick"], length=3, width=.6, pad=2)
        ax.tick_params(which="minor", length=1.8, width=.5)
        ax.grid(True, alpha=.3, lw=.4, which="major")
        ax.grid(False, which="minor")
        for s in ax.spines.values():
            s.set_linewidth(.6)
    bm, bj = r"$\mathbf{Marginal}$", r"$\mathbf{Joint}$"                        # bold (John 11-10): cmb10, as the text's cmr10
    for ax, t in zip(axes[0], (f"{bm} belief updating\n{bm} supervision", f"{bj} belief updating\n{bm} supervision",
                               f"{bj} belief updating\n{bj} supervision")):
        ax.set_title(t, fontsize=fs["title"], pad=4, linespacing=1.15)
    ys = np.concatenate([ln.get_ydata() for ax in axes.flat for ln in curves[ax]]).astype(float)
    ys = ys[np.isfinite(ys) & (ys > 0)]
    for ax, rows in keys:                                                         # at each panel's lower left (John 11-10)
        ax.set_autoscale_on(False)                                                # the keys' lines must not move the limits
        key_table(ax, rows, ("Widths", "Trainable\nparams"), fs["key"])
    h = 72 * ph                                                                   # a panel's height, points
    lo = ys.min() * (ys.min() / ys.max()) ** (SUB_YPAD / (h - 2 * SUB_YPAD))      # the shared y-axis (John 11-11): the data's
    while True:                                                                   # extremes SUB_YPAD points inside its ends;
        hi = ys.max() * (ys.max() / lo) ** (SUB_YPAD / (h - SUB_YPAD))            # the lower end drops while a curve passes
        for ax in axes.flat:                                                      # within 2 pt of a key's text
            ax.set_ylim(lo, hi)
        if lo < ys.min() / 100 or not any(_hits(ax) for ax in axes.flat):
            break
        lo /= SUB_YSTEP
    naive = next(ln for ln in axes[1, 0].get_lines() if ln.get_label().startswith("_") and ln.get_linestyle() == "--")
    axes[1, 0].text(.02, naive.get_ydata()[0] / 1.12, "Naive bound", fontsize=fs["key"], ha="left", va="top",   # once, under
                    transform=axes[1, 0].get_yaxis_transform())                   # the line where no curve is (Echo, early)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    x0 = min(inv.transform((t.get_window_extent(r).x0, 0))[0] for ax in axes[:, 0] for t in ax.get_yticklabels() if t.get_text())
    for ax, name in zip(axes[:, 0], ("Trained", "Echo")):
        p = ax.get_position()
        fig.text(x0 - 3 / 72 / W, p.y0 + p.height / 2, name, rotation=90, ha="right", va="center", fontsize=fs["row"],
                 fontfamily="cmb10")
    top, bot = axes[0, 0].get_position(), axes[1, 0].get_position()
    fig.text(2 / 72 / W, (bot.y0 + top.y1) / 2, "Optimal $-$ Network Accuracy", rotation=90, ha="left", va="center",
             fontsize=fs["label"])
    y0 = min(inv.transform((0, t.get_window_extent(r).y0))[1] for ax in axes[1] for t in ax.get_xticklabels() if t.get_text())
    fig.text((bot.x0 + axes[1, 2].get_position().x1) / 2, y0 - 2 / 72 / H, "Training Episode", ha="center", va="top",
             fontsize=fs["label"])
    fig.canvas.draw()
    for ax in axes.flat:
        for name in _hits(ax):
            print(f"  WARNING: a curve passes within 2 pt of the text {name!r}")
        bb = [(t.get_text(), t.get_window_extent(r)) for t in ax.texts if t.get_text()]
        for i, (s, b1) in enumerate(bb):
            for s2, b2 in bb[i + 1:]:
                if b1.overlaps(b2):
                    print(f"  WARNING: the texts {s!r} and {s2!r} overlap")
    fb = fig.bbox
    shown = [t for ax in axes.flat for t in [ax.title] + ax.texts] + fig.texts
    for ax in axes.flat:                                                          # tick labels drawn: inside the view limits
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            shown += [tk.label1 for tk in axis.get_major_ticks() if lim[0] <= tk.get_loc() <= lim[1] and tk.label1.get_visible()]
    for t in shown:
        if not t.get_text():
            continue
        bb = t.get_window_extent(r)
        if bb.x0 < fb.x0 - .5 or bb.x1 > fb.x1 + .5 or bb.y0 < fb.y0 - .5 or bb.y1 > fb.y1 + .5:
            print(f"  WARNING: {t.get_text()!r} reaches outside the figure")
    print(f"  y-axis {axes[0, 0].get_ylim()[0]:.4f} to {axes[0, 0].get_ylim()[1]:.3f}; panels {pw:.2f} x {ph:.2f} in")
    for p in (path,) if isinstance(path, str) else path:                         # one path, or several (one per format)
        fig.savefig(p, dpi=400)
        print("saved", p)
    plt.close(fig)
    return path


@plt.rc_context()                                                                 # the submission version's mathtext, undone
def figure(step=cmp.STEP, path=None, width_1k=True, n1=False, drop_trained=(), submission=False):
    from main.utils import fig_path

    def draw(ax, run, name, col, n_tests=pm.TEST_POOL):
        """one run's curve to its last complete block, if it has tests yet; the legend says how far a running one has got"""
        if not glob.glob(os.path.join(pm._dir(run), "test_*.npz")):
            print(f"  {run}: no test yet")
            return
        x, acc, reached = pooled(run, n_tests=n_tests)
        done = pm.finished(run)
        note = "" if done or submission else f" (training: {min(reached, pm.XMAX) / 1e3:.1f}k of {pm.XMAX / 1e3:.0f}k)"
        ax.plot(x, pm._pos(J - acc), color=col, lw=2, ls="-", label=f"{name}, {count(params(run))} trainable params{note}")
        print(f"  {run:24s} {acc[-1] if len(acc) else np.nan:.3f} at training episode {x[-1] if len(x) else 0}"
              f"{'' if done else ' (still training)'}")

    cmp.apply_fig_style()
    if submission:
        plt.rcParams["mathtext.fontset"] = "cm"                                   # math in the text's typeface
        plt.rcParams["pdf.fonttype"] = 42                                         # PDF: TrueType fonts embedded, not Type 3
    b = dict(np.load(cmp.BOUNDS_NPZ))
    J = b[f"{cmp.BOUND_KEY}_joint"][step - 1]
    fig, axes = plt.subplots(2, 3, figsize=SUB_SIZE if submission else (30, 10.4), sharex=True, sharey=True)
    (t1, t2, t3), (e1, e2, e3) = axes                                             # rows: trained LSTMs, ESNs
    extra = [dict(pm.ESN2K)] if os.path.exists(os.path.join(cmp.DATA, pm.ESN2K["stem"] + "_net.pth")) else []
    nets = cmp.resolve_colors([dict(n) for n in cmp.NETWORKS if n["stem"] not in pm.DROP] + extra)
    if n1:                                                                        # one run per curve, seed 20261029, batch 8,000
        c1k = next(n["color"] for n in nets if n["stem"] == "sanity/reservoir_ctx_2")
        for run, name, col in (("echo_N200", r"Echo ($N{=}200$)", C_E200), ("echo_N500", r"Echo ($N{=}500$)", C_E500), ("echo", r"Echo ($N{=}1k$)", c1k)):
            draw(e1, run, name, col)
        for net in nets:                                                          # no training-dynamics run: the paper's
            if "reservoir" in net["stem"] and net["stem"] not in ("sanity/reservoir_ctx_2",) and "10000N" not in net["stem"]:
                pm.paper(e1, J, net, step)
        draw(e1, "echo_N3000_b8k", r"Echo ($N{=}3k$)", C_E3K)
        draw(e1, "echo_N10000_b8k", r"Echo ($N{=}10k$)", next(n["color"] for n in nets if "10000N" in n["stem"]))
        for run, name, col in (("trained_N50", r"Trained ($N{=}50$)", C_N50), ("trained_N100", r"Trained ($N{=}100$)", pm.C_N100),
                               ("trained_N150", r"Trained ($N{=}150$)", C_N150), ("trained_N200", r"Trained ($N{=}200$)", cmp.C_TRAIN),
                               ("trained_N300", r"Trained ($N{=}300$)", C_N300), ("trained_N400", r"Trained ($N{=}400$)", C_N400),
                               ("trained_N500", r"Trained ($N{=}500$)", C_N500), ("trained", r"Trained ($N{=}1k$)", C_N1K)):
            draw(t1, run, name, col)
    else:
        for net in nets:                                                              # the left panels, as plot_matched_params_new
            if "reservoir" in net["stem"] and "10000N" not in net["stem"]:
                pm.paper(e1, J, net, step)
        draw(e1, "echo_N10000_b8k", r"Echo ($N{=}10k$)", next(n["color"] for n in nets if "10000N" in n["stem"]), n_tests=1)
        draw(e1, "echo_N200", r"Echo ($N{=}200$)", C_E200, n_tests=1)                 # one test per 500 episodes, as its neighbours
        draw(e1, "echo_N500", r"Echo ($N{=}500$)", C_E500, n_tests=1)
        draw(e1, "echo_N3000_b8k", r"Echo ($N{=}3k$)", C_E3K, n_tests=1)
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
                                           r"Echo ($N{=}3k$)", r"Echo ($N{=}10k$)"), (C_E200, C_E500, c_echo, c_2k, C_E3K, c_10k)):
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
    if not submission:                                                            # John 11-10: not in the submission version
        lin = J - b[f"{cmp.BOUND_KEY}_naive"][step - 1] - LIN_CE_EDGE                # the naive bound's gap minus the paired edge
        box = dict(fc="white", ec="none", alpha=.85, pad=1.5)
        for ax in (e1, e2, e3):                                                   # ESN row only (John 11-09)
            ax.axhline(lin, color=GREY, lw=2, ls=":", zorder=10)
        e1.text(.2, lin * .97, "Best linear-in-counts rule, marginal output (cross-entropy)", color=GREY, fontsize=15, va="top",
                ha="left", transform=e1.get_yaxis_transform(), zorder=11, bbox=box)
        for ax in (e2, e3):
            ax.text(.98, .04, "Joint output: best linear rule = optimal", color=GREY, fontsize=15, va="bottom", ha="right",
                    transform=ax.transAxes, zorder=11, bbox=box)
    for ax in (t1, t2, t3):                                                       # trained widths left out (John 11-08: comparable
        for ln in [ln for ln in ax.get_lines() if not ln.get_label().startswith("_")]:   # trainable parameters to the ESNs)
            if width_of(ln.get_label()) in drop_trained:
                ln.remove()
    for ax in (t1, t2, t3, e1, e2, e3):
        ax.relim()
        ax.autoscale_view()
    if submission:
        return _finish_submission(fig, axes, path or fig_path("compare_matched_params_loglog_joint_panels_n1.png"))
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
    figure(path=fig_path("compare_matched_params_loglog_joint_panels_n1.png"), n1=True, submission=True,   # John 11-05: one run per
           drop_trained=SUB_DROP)                                                 # curve, batch 8,000; 11-10: the submission version
    figure(path=(fig_path("compare_matched_params_loglog_joint_panels_n1_no1k.png"), fig_path("param_control_b.pdf"),
                 fig_path("param_control_b.svg")), n1=True, submission=True,     # John 11-10: also the paper's PDF and SVG
           width_1k=False, drop_trained=SUB_DROP + (1000,))                       # the same without the Trained N = 1k
    figure(path=fig_path("compare_matched_params_loglog_joint_panels_no500_no1k.png"), width_1k=False, drop_trained=(500,))   # John 11-08:
    # the trained LSTMs up to N = 400 (1.3M trainable parameters), against the ESNs up to N = 10k (1.2M)   # John 11-04: without the Trained N = 1k
