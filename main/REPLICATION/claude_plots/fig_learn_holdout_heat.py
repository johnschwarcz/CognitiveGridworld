"""fig_learn_holdout_heat.png / .pdf: after learning from one FR range only, when does each FR range leave chance?
The FR hold-out runs (fig_learn_holdout.py) as heat maps (John 11-15; prototypes in the 11-15 polish rounds).

Columns: the 2-decade gap (the networks learned only from updates with FR < .03, or only from FR > 3) and the 1-decade
gap (< .1 or > 1); rows: trained on high FR (top), trained on low FR (bottom). The two networks of a column learn from
the same number of updates at every step of every batch (train_learn_masks.py band modes with their own counts: n(t) =
min(#FR < lo, #FR > hi) of the batch's 8,000 updates at step t, picked at random within the trained range): 7.80% of all
updates for the 2-decade gap, 20.44% for the 1-decade gap (per-batch SD <= .0023). Time-balanced (checked 11-15 from the
sel*.npz 'steps' logs): the two runs of a column saw the same batches (same episode stream) and learned identical counts
at every inference step t of all 53,400 (2-decade) / 30,000 (1-decade) batches, so their learned updates have the same
distribution over t (almost none before t = 3 / t = 2, then ~3-4.5% of them per step). One batch (8,000 trajectories x 30
steps) is one optimizer step; the x axis counts the updates (trajectory x step) whose loss term reached the weights,
summed exactly from the runs' per-batch selection logs (sel_to*.npz, sel.npz), log scale, the four runs' common range.
Heat: accuracy at the last step (T = 30) on the probe set (analysis/probes.npz: 50 test environments x 100 cells x 2
trajectories = 10,000 episodes x 2 variables = 20,000 answers) in bins of the answer's FR at T (forward KL, nats; edges
1e-3, 3e-3, ..., 1e2; 228-4,626 answers per bin; 312 answers below 1e-3 are not drawn, none lie above 1e2). Cells: 6 per
decade of updates, a sixth of a decade without a checkpoint merges into the next one, each cell the mean over its
checkpoints (every checkpoint replayed on the probe set, fig_learn_holdout.harvest).
Red: the trained FR range (box; the y ticks inside it, plus an unlabelled tick at the box edge .03 / 3). Dotted white,
"Left chance": the contour at accuracy .2 = 2 x chance (10 answer options), drawn through the cells' log-space centres
(the grid padded to the frame). Strips: Joint Bayes (top) and Naive Bayes (bottom) per bin. No panel letters.
NUMBERS (--numbers, 11-15; where the line crosses each bin's centre row, i.e. accuracy first passes .2, interpolated in
log updates between two cell centres; common range 9.8e5-1.0e9 updates):
  trained on high FR, every bin leaves chance within a factor of 2.5: 2-decade 1.3e7 (FR 10-30) to 3.1e7 (FR < .003),
    the trained bins (FR > 3) first (1.3e7) and FR < .03 last (2.7-3.1e7); 1-decade 2.4e7 (FR 3-10) to 4.2e7 (FR < .003).
  trained on low FR, the line climbs one range at a time: 2-decade FR < .03 above chance from the first cell, .03-.1 at
    2.8e6, .1-.3 5.1e6, .3-1 1.2e7, 1-3 3.0e7, 3-10 5.5e8, never above 10 (FR 10-30 ends at .125, 30-100 at .022);
    1-decade FR < .01 from the first cell, .01-.03 2.0e6, .03-.1 4.5e6, .1-.3 7.4e6, .3-1 1.7e7, 1-3 3.4e7, 3-10 1.3e8,
    10-30 4.1e8, never above 30 (ends at .089).
  end accuracy (last cell) on the other side: trained high -> FR < .03 .865 / .780 / .757 (2-decade; Joint .890 / .868 /
    .808), trained low -> FR > 3 .280 / .125 / .022 (Naive .029 / .015 / .019).
Layout (John 11-15): 5.5 x 2.2 in (print width, no scaling), ticks / keys 7 pt, labels / titles 8 pt, Computer Modern;
"Trained on" in red outside each box's outer-left corner (no row titles: John 11-15 removed "Trained on high / low
FR" from the y axes, the caption names the rows); key "Left chance" (John) on a dark swatch; x label "Training
steps" (John 11-15, after "Weight updates" and "Trained trajectory steps": the simplest; the caption defines them as the
trajectory steps trained on, one optimizer step per batch of 8,000 x 30, so they are not read as optimizer steps). PDF: fonts embedded (Type 42), the meshes rasterized at 400 dpi without edges (no seams).
usage: python fig_learn_holdout_heat.py [--recompute] [--numbers]"""
import os
import sys
import glob
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
import matplotlib.transforms as mtr                                               # noqa: E402
import contourpy                                                                  # noqa: E402
from matplotlib.patches import Rectangle                                          # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, LogLocator             # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import fig_learn_holdout as h                                                     # noqa: E402
import fig_learn_grids as flg                                                     # noqa: E402  (paper style constants)

PRIM = os.path.join(h.OUT, "heat_primitives.npz")
EDGES = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30, 100])        # FR_T bins (nats)
BANDS = ((.03, 3), (.1, 1))                                                       # (lo, hi): the 2- and 1-decade gaps
SIDES = ("hi", "lo")                                                              # rows: trained on high / low FR
UPD = 8000 * h.T                                                                  # updates per batch (8,000 x 30)
CHANCE = .1                                                                       # 10 answer options
LEVEL = 2 * CHANCE                                                                # "Left chance"
PER = 6                                                                           # cells per decade of updates
RED = "#d62728"
COLT = ["2-decade gap", "1-decade gap"]
OBS = {"hi": ("jb", "Joint", "k"), "lo": ("nb", "Naive", "0.5")}
DOT = dict(color="white", lw=1.1, ls=(0, (.1, 1.9)), dash_capstyle="round", solid_capstyle="round", zorder=5)
MESH = dict(cmap="viridis", vmin=0, vmax=1, shading="flat", rasterized=True, edgecolors="none", linewidth=0,
            antialiased=False)


def run_dir(run):
    return os.path.join(h.TD, f"learnmask_{run}_s20261029")


def learned_share(run):
    """the learned share of every training batch, in order: the continuation segments (sel_to<raw>.npz, oldest first)
    then sel.npz; column 0 of a row is the blocked share of that batch"""
    segs = sorted(glob.glob(os.path.join(run_dir(run), "sel_to*.npz")),
                  key=lambda f: int(os.path.basename(f)[6:-4])) + [os.path.join(run_dir(run), "sel.npz")]
    return np.concatenate([1 - np.load(f)["sel"][:, 0] for f in segs])


def primitives(recompute=False):
    """per run: the replayed checkpoints' training episodes t, the updates learned from u (exact cumulative count), the
    accuracy at T in every FR bin (bins, checkpoints); Joint / Naive Bayes per bin; answers per bin (cached in PRIM)"""
    if os.path.exists(PRIM) and not recompute:
        z = np.load(PRIM)
        return {k: z[k] for k in z.files}
    pr = np.load(os.path.join(h.AN, "probes.npz"))
    lj, ln, r = pr["log_jb"], pr["log_nb"], pr["r"]
    fr = h.flp.forward_fr(lj, ln)[:, -1].ravel()                                  # (n x 2,): FR at T, both variables
    okb, okn = (lj[:, -1].argmax(-1) == r).ravel(), (ln[:, -1].argmax(-1) == r).ravel()
    inb = [(fr >= a) & (fr < b) for a, b in zip(EDGES[:-1], EDGES[1:])]
    P = dict(edges=EDGES, bands=np.array(BANDS), n=np.array([b.sum() for b in inb]),
             jb=np.array([okb[b].mean() for b in inb]), nb=np.array([okn[b].mean() for b in inb]),
             n_below=np.array((fr < EDGES[0]).sum()), n_above=np.array((fr >= EDGES[-1]).sum()))
    for bi, (lo, hi) in enumerate(BANDS):
        for side in SIDES:
            run = h.band_name(side, lo, hi)
            raws = sorted(int(os.path.basename(f)[len(run) + 1:-4]) for f in glob.glob(os.path.join(h.OUT, f"{run}_*.npz"))
                          if os.path.basename(f)[len(run) + 1:-4].isdigit())
            sh = learned_share(run)
            cum = np.concatenate([[0.], np.cumsum(sh)])
            t = np.array([h.train_x(raw) for raw in raws])
            assert t[-1] <= len(sh), f"{run}: selection log shorter than the checkpoints"
            P[f"t_{bi}_{side}"], P[f"u_{bi}_{side}"] = t, cum[t] * UPD
            P[f"share_{bi}_{side}"] = np.array([sh.mean(), sh.std()])
            P[f"acc_{bi}_{side}"] = np.array([[(np.load(h.cache(run, raw))["ans"][:, -1] == r).ravel()[b].mean()
                                               for b in inb] for raw in raws]).T
    np.savez(PRIM[:-4] + ".tmp.npz", **P)
    os.replace(PRIM[:-4] + ".tmp.npz", PRIM)
    return P


def x_range(P):
    """the four runs' common range of updates: the latest first nonzero checkpoint, the earliest last one"""
    xs = max(P[f"u_{b}_{s}"][P[f"u_{b}_{s}"] > 0][0] for b in range(len(BANDS)) for s in SIDES)
    xe = min(P[f"u_{b}_{s}"][-1] for b in range(len(BANDS)) for s in SIDES)
    return xs, xe


def log_cells(X, acc, xs, xe):
    """cells of PER per decade on the lattice 10^(k / PER); a lattice bin without a checkpoint merges into the next one;
    the first / last cell stretched to the common range; a cell = the mean of its checkpoints"""
    keep = (X >= xs * (1 - 1e-9)) & (X <= xe * (1 + 1e-9))
    X, acc = X[keep], acc[:, keep]
    e = 10 ** np.arange(np.floor(np.log10(X[0]) * PER) / PER, np.log10(X[-1]) + 1 / PER + 1e-9, 1 / PER)
    k = np.clip(np.searchsorted(e, X, side="right") - 1, 0, len(e) - 2)
    K = np.unique(k)
    edges = np.concatenate([[e[K[0]]], e[K + 1]])
    edges[0], edges[-1] = xs, xe
    return edges, np.array([[row[k == kk].mean() for kk in K] for row in acc])


def chance_line(xe, cell):
    """the contour at LEVEL through the cells' log-space centres, the grid padded to the frame with the edge cells"""
    xg = np.log10(np.r_[xe[0], np.sqrt(xe[:-1] * xe[1:]), xe[-1]])
    yg = np.log10(np.r_[EDGES[0], np.sqrt(EDGES[:-1] * EDGES[1:]), EDGES[-1]])
    lines = contourpy.contour_generator(xg, yg, np.pad(cell - LEVEL, 1, mode="edge")).lines(0)
    return [(10 ** q[:, 0], 10 ** q[:, 1]) for q in lines]


def crossings(xe, cell):
    """per FR bin: where accuracy first passes LEVEL going up, interpolated in log updates between cell centres (the
    contour's crossing of the bin's centre row); 0 = above from the first cell, nan = never"""
    lx = np.log10(np.sqrt(xe[:-1] * xe[1:]))
    out = []
    for v in cell:
        if v[0] >= LEVEL:
            out.append(0.)
            continue
        k = np.flatnonzero((v[:-1] < LEVEL) & (v[1:] >= LEVEL))
        out.append(np.nan if not len(k) else 10 ** (lx[k[0]] + (LEVEL - v[k[0]]) / (v[k[0] + 1] - v[k[0]])
                                                     * (lx[k[0] + 1] - lx[k[0]])))
    return np.array(out)


def numbers(P):
    xs, xe_ = x_range(P)
    print(f"common range of updates learned from: {xs:.4g} .. {xe_:.4g}; answers per bin {P['n'].tolist()}, "
          f"below 1e-3 {int(P['n_below'])}, above 1e2 {int(P['n_above'])}")
    print("Joint per bin", np.round(P["jb"], 3).tolist(), "\nNaive per bin", np.round(P["nb"], 3).tolist())
    for bi in range(len(BANDS)):
        for side in SIDES:
            m, sd = P[f"share_{bi}_{side}"]
            xe, cell = log_cells(P[f"u_{bi}_{side}"], P[f"acc_{bi}_{side}"], xs, xe_)
            c = crossings(xe, cell)
            print(f"{COLT[bi]}, trained {side} (learned share {m:.4f}, per-batch SD {sd:.4f}; {cell.shape[1]} cells): "
                  "accuracy passes .2 at (updates; 0 = from the first cell, nan = never), end accuracy:")
            for k in range(len(c)):
                print(f"   FR [{EDGES[k]:g}, {EDGES[k + 1]:g}): {c[k]:9.3g}   end {cell[k, -1]:.3f}")


def fr_axis(ax, extra=()):
    ax.set_yscale("log")
    ax.set_ylim(EDGES[0], EDGES[-1])
    dec = np.arange(np.ceil(np.log10(EDGES[0]) - 1e-9), np.floor(np.log10(EDGES[-1]) + 1e-9) + 1)
    ax.yaxis.set_major_locator(FixedLocator(10.0 ** dec[dec % 2 == 0]))
    ax.yaxis.set_minor_locator(FixedLocator(np.sort(np.r_[10.0 ** dec[dec % 2 != 0], list(extra)])))
    ax.yaxis.set_minor_formatter(NullFormatter())


def red_ticks(ax, a, b):
    """y ticks (mark and label) with a <= FR <= b in red; ticks paired with their locations as Axis._update_ticks does"""
    for locs, ticks in ((ax.yaxis.get_majorticklocs(), ax.yaxis.get_major_ticks),
                        (ax.yaxis.get_minorticklocs(), ax.yaxis.get_minor_ticks)):
        for tk, v in zip(ticks(len(locs)), locs):
            if a * (1 - 1e-9) <= v <= b * (1 + 1e-9):
                tk.tick1line.set_markeredgecolor(RED)
                tk.label1.set_color(RED)


def figure(P, path=None):
    FS, W = flg.FS, flg.W
    apply_fig_style()
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.linewidth": .5, "pdf.fonttype": 42})
    HX0 = .45                                                                     # FR label and ticks
    SW, SG, CB, CG = .12, .06, .58, .1                                            # strip, gap, colour bar column, col gap
    sx0 = W - CB - SW
    NC = len(BANDS)
    HW = (sx0 - SG - HX0 - CG * (NC - 1)) / NC
    TOPM, TT, RH, RG, BOT = .02, .3, .62, .22, .42                                # inches
    h_tops = [TOPM + TT, TOPM + TT + RH + RG]
    H = h_tops[1] + RH + BOT
    INS = .9                                                                      # box inset at the frame (pt)
    xs, xe_ = x_range(P)
    fig = plt.figure(figsize=(W, H))
    for kr, side in enumerate(SIDES):
        top = h_tops[kr]
        for bi, (lo, hi) in enumerate(BANDS):
            ax = fig.add_axes([(HX0 + bi * (HW + CG)) / W, 1 - (top + RH) / H, HW / W, RH / H])
            xe, cell = log_cells(P[f"u_{bi}_{side}"], P[f"acc_{bi}_{side}"], xs, xe_)
            m = ax.pcolormesh(xe, EDGES, cell, **MESH)
            ax.set_xscale("log")
            a, b = (EDGES[0], lo) if side == "lo" else (hi, EDGES[-1])
            thr = b if side == "lo" else a                                        # the box's inner edge
            fr_axis(ax, extra=() if np.isclose(np.log10(thr), np.round(np.log10(thr))) else (thr,))
            ax.set_xlim(xs, xe_)
            for xs_, ys_ in chance_line(xe, cell):
                ax.plot(xs_, ys_, **DOT)
            f = 10 ** (INS / 72 * np.log10(EDGES[-1] / EDGES[0]) / RH)
            a_, b_ = (a * f, b) if side == "lo" else (a, b / f)
            dx = INS / 72 / HW
            ax.add_patch(Rectangle((dx, a_), 1 - 2 * dx, b_ - a_, fill=False, ec=RED, lw=1.2, zorder=4,
                                   transform=mtr.blended_transform_factory(ax.transAxes, ax.transData)))
            if side == "hi":
                ax.text(dx, 1 + 2 / 72 / RH, "Trained on", transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=FS["key"], color=RED)
            else:
                ax.text(dx, -4 / 72 / RH, "Trained on", transform=ax.transAxes, ha="left", va="top",
                        fontsize=FS["key"], color=RED)
            ax.xaxis.set_major_locator(FixedLocator([1e7, 1e8]))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 1.0, numticks=20))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(which="both", labelsize=FS["tick"], pad=2, width=.5)
            ax.tick_params(which="major", length=2.5)
            ax.tick_params(which="minor", length=1.5)
            ax.tick_params(axis="y", which="minor", length=2.5)                  # odd decades and the box edge
            if kr == 0:
                ax.tick_params(labelbottom=False)
                ax.set_title(COLT[bi], fontsize=FS["title"], pad=13)
            if bi > 0:
                ax.tick_params(labelleft=False)
            red_ticks(ax, a, b)
        sx = fig.add_axes([sx0 / W, 1 - (top + RH) / H, SW / W, RH / H])
        sx.pcolormesh([0, 1], EDGES, P[OBS[side][0]][:, None], **MESH)
        fr_axis(sx)
        sx.set_xticks([])
        sx.tick_params(which="both", left=False, labelleft=False)
        sx.set_title(OBS[side][1], fontsize=FS["title"], pad=3, color=OBS[side][2])
    fig.text(.08 / W, 1 - (h_tops[0] + RH + RG / 2) / H, "factorization regret", rotation=90, ha="center",
             va="center", fontsize=FS["label"])
    KY = 1 - (h_tops[1] + RH + .3) / H                                            # x-label baseline
    fig.text((HX0 + (NC * HW + (NC - 1) * CG) / 2) / W, KY, "Training steps", ha="center", va="baseline",
             fontsize=FS["label"])
    hcx = fig.add_axes([(sx0 + SW + .12) / W, 1 - (h_tops[1] + RH) / H, .06 / W, (h_tops[1] + RH - h_tops[0]) / H])
    hcb = fig.colorbar(m, cax=hcx, ticks=np.round(np.linspace(0, 1, 6), 6))
    hcb.solids.set_rasterized(True)
    hcb.solids.set_edgecolor("face")
    hcb.ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 6)])
    hcb.ax.tick_params(labelsize=FS["tick"], length=2, width=.5, pad=1.5)
    hcb.outline.set_visible(False)
    hcb.set_label("accuracy", fontsize=FS["label"], labelpad=3)
    kt = fig.text((W - .03) / W, KY, "Left chance", ha="right", va="baseline", fontsize=FS["key"])
    fig.canvas.draw()
    kb = kt.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    KW, KH = .22, .08                                                             # key: the line on a dark swatch
    kax = fig.add_axes([(kb.x0 - 3 / 72 - KW) / W, ((kb.y0 + kb.y1) / 2 - KH / 2) / H, KW / W, KH / H])
    kax.set_facecolor(plt.cm.viridis(.1))
    kax.plot([.12, .88], [.5, .5], **DOT)
    kax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    kax.spines[:].set_visible(False)
    path = path or fig_path("fig_learn_holdout_heat.png")
    fig.savefig(path, dpi=flg.DPI)
    fig.savefig(path[:-4] + ".pdf", dpi=flg.DPI)
    print("saved", path, "and .pdf", "(%.2f x %.2f in)" % (W, H))
    return fig


if __name__ == "__main__":
    P = primitives(recompute="--recompute" in sys.argv)
    if "--numbers" in sys.argv:
        numbers(P)
    plt.close(figure(P))
