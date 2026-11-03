"""The factorized observers' fold, in two summary figures (27 Oct 2026)
-> main/REPLICATION/figure_bin/fig1c_summary_fold.png and fig1c_summary_updates.png

    fig1c_summary_fold.png (one row: a as a 2 x 2 block, b beside it)
        a     distance between two particles, estimates (posterior means) vs targets     plot_fig1c_distances.draw_pairs_row
        b     how a cell's factorization regret shapes convergence to the true cell (3-D) plot_fig1c_distances.draw_fr_3d
    fig1c_summary_updates.png (one row of five)
        a, b  update sizes of the two beliefs, joint minus factorized agent              plot_fig1c_update_sizes.draw
        c-e   joint movement of the two beliefs by FR, entanglement and path length      plot_fig1c_update_corr.draw
Each part is drawn by its own figure's code (the standalone figures are unchanged), from the same data:
figdata/round4b/fig1c_distances_pairs.npz, fig1c_distances_fr.npz and updates_50k.npz. The data behind each panel are
in the docstrings of those plotters (no captions in the figures).
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/plot_fig1c_fold_summary.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
import plot_fig1c_distances as pd                                                 # noqa: E402
import plot_fig1c_update_corr as pc                                               # noqa: E402
import plot_fig1c_update_sizes as pu                                              # noqa: E402

FS = pd.FS


def _header(fig, x, y, w, name, formula, letter):
    W, H = fig.get_size_inches()
    hdr = fig.add_axes([x / W, y / H, w / W, .01 / H])
    hdr.axis("off")
    pd.title2(hdr, name, formula)
    pd.letter(fig, hdr, letter, dx=.62)


def _fit_3d(target, zoom, passes=3, dpi=60, **kw):
    """the axes rect (x, y, w, h; inches) that makes the 3-D panel's drawn content (curves, box, ticks, labels: its
    inked pixels, measured on a large scratch canvas so nothing is clipped) fill target (x0, y0, x1, y1), centred"""
    rw, rh = 8., 5.
    Tw, Th = target[2] - target[0], target[3] - target[1]
    for _ in range(passes):
        tf = plt.figure(figsize=(3 * rw, 3 * rh), dpi=dpi)
        ax = pd.draw_fr_3d(tf, (rw, rh, rw, rh), None, (0, 0, .1, .01), None, zoom=zoom, **kw)
        for a_ in tf.axes:
            if a_ is not ax:
                a_.set_visible(False)
        for lg in tf.legends:
            lg.set_visible(False)
        tf.canvas.draw()
        ink = (np.asarray(tf.canvas.buffer_rgba())[..., :3] < 250).any(-1)
        plt.close(tf)
        rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
        cx0, cx1 = cols[0] / dpi - rw, (cols[-1] + 1) / dpi - rw                    # from the axes' left edge
        cy0, cy1 = 2 * rh - (rows[-1] + 1) / dpi, 2 * rh - rows[0] / dpi            # from the axes' bottom edge
        sc = min(Tw / (cx1 - cx0), Th / (cy1 - cy0))
        rect = (target[0] + (Tw - sc * (cx1 - cx0)) / 2 - sc * cx0, target[1] + (Th - sc * (cy1 - cy0)) / 2 - sc * cy0,
                sc * rw, sc * rh)
        if abs(sc - 1) < .01:
            break
        rw, rh = sc * rw, sc * rh
    return rect


def figure_fold(zoom=1.2):
    """a: pair distances (posterior means; the four agents in a 2 x 2 block); b, beside it: the 3-D FR-convergence plot
    (no colour bar: the FR axis carries the values; the line styles named in its header), scaled to fill the rest"""
    sq, gx, gy, xa, ya = 2.55, .45, .85, .8, .62
    top = ya + 2 * sq + gy                                                            # a's block
    hdr_y = top + .38                                                                 # both headers' line
    xb = xa + 2 * sq + gx + .2 + .1 + .75                                             # after a's step colour bar
    W, H = xb + 10.6, hdr_y + 17 / 72 + .35
    fig = plt.figure(figsize=(W, H))
    pd.draw_pairs_row(fig, xa, ya, sq, gx, 0, letter_s="a", ncol=2, gy=gy,
                      formula=r"$\|\hat r_i - \hat r_j\|$ against the true $\|r_i - r_j\|$, $\hat r = (E[r_g], E[r_c])$; "
                              r"one curve per step; SEM bars")
    kw = dict(label_xz=((-22, 6.15), (13, 3.3)))
    rect = _fit_3d((xb, .08, W - .08, hdr_y - .12), zoom, **kw)
    pd.draw_fr_3d(fig, rect, None, (xb + .45, hdr_y, W - xb - .55, .01), None, letter_s="b", zoom=zoom,
                  formula=r"$\langle \|(E[r_g], E[r_c]) - (r_g, r_c)\| \rangle_{\mathrm{particles}}$; "
                          r"mirrored at $t = 1$; dashed: theory, solid: networks", **kw)
    path = pd.fig_path("fig1c_summary_fold.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def figure_updates():
    """one row of five: a, b the update-size difference maps (Optimal - Naive, Trained RNN - ESN); c-e joint movement by
    FR, entanglement and path length. The curves are taller than the maps by the maps' notes, so every panel's title
    and letter sit on one line"""
    Zu = np.load(pu.DATA)
    P, y0 = 2.6, .62                                                                  # map side, bottom of every panel
    Pc = P + 30 / 72 - 6 / 72                                                         # curves: titles on the maps' line
    xa, gap_ab, wc, gxc = .85, .35, 2.75, .85
    xb = xa + P + gap_ab
    xcb = xb + P + .3                                                                 # the maps' colour bar
    xc = xcb + 1.95                                                                   # the first curve
    W = xc + 3 * wc + 2 * gxc + .25
    top = y0 + Pc + 6 / 72                                                            # the titles' baseline
    hdr_y = top + .55
    H = hdr_y + 17 / 72 + .38
    fig = plt.figure(figsize=(W, H))
    pu.draw(fig, [(xa, y0, P, P), (xb, y0, P, P)], (xcb, y0 + .45, .13, P - .9), Zu, notes="short")
    for x, off, s_ in ((xa, .62, "a"), (xb, .3, "b")):
        fig.text((x - off) / W, top / H, r"$\mathbf{%s}$" % s_, fontsize=FS["label"] + 2, va="baseline")
    C, med = pc.joint_movement(Zu)
    pc.draw(fig, xc, y0, wc, Pc, gxc, C, med, letters="cde", formulas=False,
            legend_xy=(xc + 1.5 * wc + gxc, top + .43))
    for x0, w, name, formula in (
            (xa, xb + P - xa, "How the two beliefs move",
             r"share of steps per bin of $(|\Delta x|, |\Delta y|)$, joint $-$ factorized agent; every step of 48,968 "
             r"trajectories"),
            (xc, 3 * wc + 2 * gxc, "Joint movement of the two beliefs",
             r"by step: $\mathrm{Cov}(|u_g(t)|, |u_c(t)|)$, mean over $t$")):
        hdr = fig.add_axes([x0 / W, hdr_y / H, w / W, .01 / H])
        hdr.axis("off")
        pd.title2(hdr, name, formula)
    path = pd.fig_path("fig1c_summary_updates.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure_fold())
    print("saved", figure_updates())
