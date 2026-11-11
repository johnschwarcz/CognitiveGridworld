"""Flipping gradients through training (John 11-01: "The flipping gradient in the trained around when it surpasses Naive
is exactly the kind of thing we want to see, do you see these kinds of flipping gradients anywhere else (in other
measures, other splits, other analyses, etc)").
-> main/REPLICATION/figure_bin/td_flips.png, td_flips_smoothed.png, td_flips_relative.png
The flips found by td_flip_scan.py (14 measures x 4 split variables x 2 rankings, both networks, all 114 checkpoints;
a flip = the top-minus-bottom fifth difference reliably one sign for >= 5 consecutive checkpoints, then reliably the
other; both environment halves agree), drawn as panel e of td_summary is: the 5 fifths of the split variable (ranked
within each step 2-30: 2,000 particles of every step per fifth; "within": the variable minus its episode's and step's
means) through training, light = lowest fifth, dark = highest; grey band: between the last checkpoint of the first
sign and the first of the other; dotted: the Trained RNN passes naive (880 training episodes).
Pairing (td_flip_scan.py, flip_scan_paired.npz): a per-variable measure (accuracy, belief error, memory use) is split
by the SAME variable's entanglement / confidence / FR / update size, both variables' values ranked together within
each step (4,000 values of every step per fifth: panels a-e); whole-network measures by the split variable averaged
over the two variables (f, h: 2,000 particles of every step per fifth; g, i: the 5,000 update-half probes, about 1,000
per step). Versions: the first (11-01) averaged both sides over the two variables; its accuracy-by-FR flip (- to + at
140-880) came from the cross terms (paired: high-FR steps less accurate at every checkpoint, -.019 to -.255), and its
accuracy-by-entanglement flip (140-360) moves to 480-720 when paired and no longer replicates (halves agree 81%).
Panels a and b are now the belief error by confidence and the accuracy by update size within the episode.
Selection: flips visible in the fans (both halves agree on >= 95% of the checkpoints of both signs), and the ESN's
memory-use flip (opposite in direction, ~5x earlier). Scan totals (halves agree >= 95%): Trained RNN 36 flips in 31
of 112 measure x split combinations, 23 of them with geometric midpoints between 300 and 3,000 training episodes
(around passing naive), 3 earlier, 10 later; ESN 17 flips in 16 combinations, among them memory use (140-260
episodes), and the belief error by Bayes's confidence (+1.20 -> -1.23 nats, 1,600-4,400) and by update size (-.41 ->
+.45, 720-1,120). (Totals re-scanned 11-10 with FR = KL(Bayes || naive), the paper's forward form: probes.npz 'fr'.
With the symmetric clipped FR of 11-01 they were 38 in 33 (24 / 4 / 10) and 14 in 13; only the FR splits' series
changed (td_flip_scan.py), and no panel here is split by FR, so the figures are as they were.)
Flips found but not drawn: |u| by Bayes's update size (-.28 -> +.11 between 880 and 1,360), memory use by own
entanglement within the episode (-.002 -> +.050, 400-720); the update energy by update size (-10.6 -> +10.4 between
1,040 and 1,680) is left out: its halves agree on only 83%.
Views (John 11-01, on the training-time jitter: "smoothing over neighbouring checkpoints" and "the panels relative to
the checkpoint's means"):
  smoothed  each fifth's running mean over 5 neighbouring checkpoints (+-2; the window shrinks at the ends, the initial
            network alone), the raw curves faint underneath. The jitter is independent from one checkpoint to the next
            (td_summary notes: no memory beyond ~4 updates), so 5 checkpoints cut it by sqrt(5) ~ 2.2x; the window
            spans x3 in training episodes at 100, x1.5 at 500, about -5% to +20% from 3,000 on
  relative  each fifth minus the mean of the five at the same checkpoint (log-scaled panels: in log10 units, the
            vertical distance on the raw panel's log axis); the common swing of all fifths, most of the jitter, cancels:
            the jitter (second-difference SD from 40 training episodes on, median over the fifths; jitter_table())
            drops 2.3x (accuracy, update dimension) to 8.9x (cell-state norm), median 3.5x, with no blur; the
            smoothed view's 2.2x costs the early phase (e.g. d's real dip at 20 training episodes, -.068, reads
            -.026). (An earlier table, 1.7-44x, included the transient at 20 training episodes, which is not jitter.)
Run from the repository root (CPU, ~3 min the first time; fan data cached in analysis/flip_fans_paired.npz):
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig_flips.py"""
import os
import sys
import glob
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm                        # noqa: E402
from matplotlib.cm import ScalarMappable                                          # noqa: E402
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter            # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import td_flip_scan as fs                                                         # noqa: E402
import fig_entanglement_driver as fd                                              # noqa: E402

AN = fd.AN
KCOL = plt.cm.Purples(np.linspace(.3, 1., 5))
X0, XMIN, XMAX = 10, 8, 7e4
COL = {"trained": "#1f77b4", "echo": "#ff7f0e"}
NAME = {"trained": "Trained RNN", "echo": "ESN"}
# (run, measure, split, y label, log y)
PANELS = (("trained", "kl_bayes_net", "confidence | across", r"KL(Bayes $\Vert$ net) (nats)", True),
          ("trained", "correct", "update size | within", "correct at step t", False),
          ("trained", "kl_bayes_net", "entanglement | within", r"KL(Bayes $\Vert$ net) (nats)", True),
          ("trained", "memory_cost", "entanglement | across", r"cut $-$ intact update error", False),
          ("echo", "memory_cost", "entanglement | across", r"cut $-$ intact update error", False),
          ("trained", "norm_c", "entanglement | across", r"$\Vert c_t \Vert$", False),
          ("trained", "cr_dim", "entanglement | across", "update eff. dimension", False),
          ("trained", "cr_bandx", "confidence | across", "cross-band share", False),
          ("trained", "cr_coh", "entanglement | within", "credit coherence", True))
TITLE = {"correct": "Accuracy", "kl_bayes_net": "Belief error", "memory_cost": "Memory use",
         "cr_dim": "Update eff. dimension", "norm_u": "Activity norm", "cr_bandx": "Cross-band share",
         "cr_E": "Update energy", "norm_c": "Cell-state norm", "cr_coh": "Credit coherence"}
SPLIT = {"entanglement": "entanglement", "FR": "factorization regret", "update size": "Bayes's update size",
         "confidence": "Bayes's confidence"}
PER_VAR = ("correct", "kl_bayes_net", "entropy", "update_error", "memory_cost")   # split by the variable's own value


def fans():
    """fifth means per checkpoint for every panel (cached)"""
    path = os.path.join(AN, "flip_fans_paired.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    P = dict(np.load(os.path.join(AN, "probes.npz")))
    B = fs.bins(P)
    out = {}
    for run in sorted({p[0] for p in PANELS}):
        files = sorted(glob.glob(os.path.join(AN, run, "ck_?????.npz")))
        memcut = np.load(os.path.join(AN, f"ent_memcut_var_{run}.npz"))
        want = [(m, s) for r, m, s, _, _ in PANELS if r == run]
        rows = {key: [] for key in want}
        eps = []
        for k, f in enumerate(files):
            ck = np.load(f)
            eps.append(int(ck["episode"]))
            M = fs.measures(run, k, ck, P, memcut)
            for m, s in want:
                v, r_, st = M[m]
                b = fs.bins_for(B[s], v, r_, st)
                rows[(m, s)].append([v[b == q].mean() for q in range(5)])
        out[f"{run}|episode"] = np.array(eps)
        for (m, s), v in rows.items():
            out[f"{run}|{m}|{s}"] = np.array(v)
        print(run, "done", flush=True)
    np.savez(path, **out)
    return out


def flip_window(run, m, s):
    z = np.load(os.path.join(AN, "flip_scan_paired.npz"))
    g = z[f"{run}|{m}|{s}"]
    found = fs.flips(g[:, 0], g[:, 1], g[:, 2])
    x = fd.train_x(z[f"{run}|episode"])
    return [(x[fl["end"]], x[fl["start"]]) for fl in found if fl["rep"] >= .95]               # replicated flips only


def xaxis(ax, label):
    ax.set_xscale("log")
    ax.set_xticks([X0, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.set_xlim(XMIN, XMAX)
    ax.minorticks_off()
    if label:
        ax.set_xlabel("training episodes", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)


K_SMOOTH = 2                                                                      # +-2 neighbouring checkpoints
OUT = {"raw": "td_flips.png", "smoothed": "td_flips_smoothed.png", "relative": "td_flips_relative.png"}
HEAD = {"raw": "", "smoothed": "Smoothed: running mean over 5 neighbouring checkpoints (raw faint). ",
        "relative": "Relative: each fifth minus the mean of the five at the same checkpoint (log panels: in log10 units). "}


def smooth(Y, k=K_SMOOTH):
    """running mean over the 2k + 1 neighbouring checkpoints; the window shrinks symmetrically at both ends (the
    initial network stays alone)"""
    out = np.empty(Y.shape)
    for i in range(len(Y)):
        h = min(k, i, len(Y) - 1 - i)
        out[i] = Y[i - h:i + h + 1].mean(0)
    return out


def relative(Y, logy):
    """each fifth minus the mean of the five at its checkpoint (log10 units for log-scaled panels)"""
    Z = np.log10(Y) if logy else Y
    return Z - Z.mean(1, keepdims=True)


def jitter(Y):
    """each curve's checkpoint-to-checkpoint jitter SD from second differences (white noise: var = 6 sigma^2)"""
    return (Y[2:] - 2 * Y[1:-1] + Y[:-2]).std(0) / np.sqrt(6)


def figure(view="raw"):
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    F = fans()
    cross = fd.passes_naive(dict(np.load(os.path.join(AN, "probes.npz"))))
    fig, axs = plt.subplots(2, 5, figsize=(26, 10.5))
    key = axs.ravel()[len(PANELS)]
    for ax, (run, m, s, ylab, logy) in zip(axs.ravel(), PANELS):
        x = fd.train_x(F[f"{run}|episode"])
        Y = F[f"{run}|{m}|{s}"]
        for lo, hi in flip_window(run, m, s):
            ax.axvspan(lo, hi, color="0.88", lw=0, zorder=0)
        if view == "relative":
            Y = relative(Y, logy)
            ax.axhline(0, color="0.6", lw=.8)
            ylab = (r"$\log_{10}$ " + ylab.split(" (")[0] if logy else ylab) + "\n" + r"$-$ mean of the fifths"
        for q in range(5):
            if view == "smoothed":
                ax.plot(x, Y[:, q], color=KCOL[q], lw=.8, alpha=.35)
            ax.plot(x, smooth(Y[:, q:q + 1])[:, 0] if view == "smoothed" else Y[:, q], color=KCOL[q], lw=2)
        if logy and view != "relative":
            ax.set_yscale("log")
            if np.nanmax(Y) / np.nanmin(Y) < 100:                                     # less than two decades: label 1-2-5
                ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
        ax.axvline(cross, color="0.5", lw=1, ls=":")
        xaxis(ax, label=ax in axs[-1] or ax is axs[0, -1])
        var, rank = s.split(" | ")
        own = "own " if m in PER_VAR else ""
        ax.set_title(f"{TITLE[m]}, by {own}{SPLIT[var]}" + (" (within episode)" if rank == "within" else "")
                     + ("" if m in PER_VAR else "\n(mean of the two variables)"), loc="left",
                     fontsize=12.5, color=COL[run] if run == "echo" else "k")
        ax.set_ylabel(ylab, fontsize=11)
        if run == "echo":
            ax.text(.03, .95, "ESN", transform=ax.transAxes, ha="left", va="top", fontsize=11, color="w",
                    bbox=dict(fc=COL["echo"], ec="none", pad=1.5))
    for i, ax in enumerate(axs.ravel()[:len(PANELS)]):
        ax.text(-.14, 1.06, "abcdefghi"[i], transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")
    key.axis("off")
    h = [Line2D([], [], color="0.85", lw=8, label="flip: between the last checkpoint\nof one sign and the first of the other"),
         Line2D([], [], color="0.5", lw=1, ls=":", label=f"Trained RNN passes naive\n({cross:,.0f} training episodes)")]
    key.legend(handles=h, loc="upper left", frameon=False, fontsize=11.5, borderaxespad=0)
    cax = key.inset_axes([0, .3, .95, .07])
    sm = ScalarMappable(norm=BoundaryNorm(np.linspace(0, 100, 6), 5), cmap=ListedColormap(KCOL))
    fig.colorbar(sm, cax=cax, orientation="horizontal").set_label(
        "fifth of the split variable\n(percentile, ranked within each step)", fontsize=11.5)
    fig.text(.5, .985, HEAD[view] + "Trained RNN except e (ESN); 10,000 probe episodes (50 environments x 100 latent "
             "pairs x 2 trajectories), steps 2-30 (g-i: 5,000 of them, error steps 17-30), 114 checkpoints per network",
             ha="center", va="top", fontsize=11, color="0.3")
    fig.tight_layout(rect=(0, 0, 1, .965), h_pad=3, w_pad=2.2)
    path = fig_path(OUT[view])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


def jitter_table():
    """per panel: the median over the fifths of the raw jitter SD and of the relative view's (same units: log10 for the
    log-scaled panels), and the common mode's (the mean of the fifths)"""
    F = fans()
    for run, m, s, _, logy in PANELS:
        Y = F[f"{run}|{m}|{s}"][2:]                    # from 40 training episodes: the transient at 20 is not jitter
        Z = np.log10(Y) if logy else Y
        raw, rel, com = np.median(jitter(Z)), np.median(jitter(relative(Y, logy))), jitter(Z.mean(1, keepdims=True))[0]
        print(f"  {run:7s} {m:13s} {s:24s} jitter raw {raw:.3g}, relative {rel:.3g} ({raw / rel:.1f}x less), "
              f"common mode {com:.3g}")


if __name__ == "__main__":
    for v in OUT:
        figure(v)
    jitter_table()
