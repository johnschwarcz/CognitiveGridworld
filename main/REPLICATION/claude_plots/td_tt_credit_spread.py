"""Is the error's credit spread evenly over the past steps when the Trained RNN passes naive? (John 11-09, on
td_tt_credit_reach's share-by-lag curves: "does this image imply (and can you verify) that at crossing naive the error
is spread more uniformly across all steps? Maybe we could show the distribution of error spread across lags, through
training and see this uniform spreading emerge?")
Data: td_tt_credit_reach (read only): per probe (2,000 = 40 per test environment) and checkpoint (32), the credit share
s_k of lags k = 0..15 = the share of the gate-pre-activation gradient energy ||delta_k||^2 that the errors at steps 17-30
send k steps back (summed over those error steps, then normalised per probe).
  a  mean share per lag through training (every lag, no binning; log colour, bright = more); uniform = 1/16
  b  effective number of lags per probe, PR = 1 / Sigma_k s_k^2 (16 = perfectly even over the 16 lags, 1 = one lag),
     statistic per probe, then the mean; band: 10-90% of probes; dotted: the Trained RNN passes naive (880)
  c  the share profile at five checkpoints (mean over probes), log y; dashed: uniform 1/16
-> main/REPLICATION/figure_bin/td_tt_credit_spread.png
RESULTS (11-09): PR 1.8 (initial) -> 7.0 (80) -> peak 8.2-8.6 at 560-800 -> 7.7 (880) -> 5.4 (2,400) -> 3.1 (end).
  Not uniform at any stage: at the peak the credit is even over the last ~4 steps (lags 0-3: .16 / .17 / .16 / .13 at
  640) and decays beyond (lags 9-15: .01-.02 each, against 1/16 = .06). Before the peak lag 0 dominates (.71 initial,
  .37 at 80); after, lags 0-1 take it back (.39 / .38 at the end). The binned figure (lags 3-4, 5-8, 9-15 pooled) makes
  the bands meet near 880 because pooled bins hold 2-7 lags each.
Run: MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_credit_spread.py"""
import os
import sys
import glob
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.colors import LogNorm                                             # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import td_tt_gap as gap                                                           # noqa: E402
import td_tt_credit_reach as cr                                                   # noqa: E402

TS, FL, FT = 11, 10, 10
CROSS = 880
STAGES = (10, 80, 640, 2400, 39996)
K = cr.KMAX + 1


def load(run="trained"):
    fs = sorted(glob.glob(os.path.join(cr.OUT, f"{run}_*.npz")))
    x, S, env = [], [], None
    for f in fs:
        z = np.load(f)
        s = z["share"].astype(np.float64)
        S.append(s / s.sum(1, keepdims=True))
        x.append(float(z["x"]))
        env = z["env"]
    o = np.argsort(x)
    return np.array(x)[o], np.stack(S)[o], env                                  # (C,), (C, n, 16), (n,)


def head(ax, letter, name, formula):
    for text, dy, fs in ((formula, 6, FL), (f"{letter}   {name}", 21, TS)):
        ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, dy), textcoords="offset points",
                    ha="left", va="baseline", fontsize=fs, annotation_clip=False)


def taxis(ax):
    ax.set_xscale("log")
    ax.set_xticks([10, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.minorticks_off()
    ax.set_xlabel("training episodes", fontsize=FL, labelpad=1)


def figure():
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({"xtick.labelsize": FT, "ytick.labelsize": FT})
    x, S, env = load()
    pr = 1 / (S ** 2).sum(-1)                                                   # (C, n) per probe
    C = gap.boot_counts()
    pe = np.stack([np.bincount(env, p, gap.N_ENV) for p in pr])                # (C, 50) per-environment sums
    ne = np.bincount(env, minlength=gap.N_ENV)
    boot = (pe @ C.T) / (C @ ne)                                                # (C, B)
    m = S.mean(1)                                                               # (C, 16)
    Wf, Hf = 5.5, 5.0
    fig = plt.figure(figsize=(Wf, Hf))

    def add(l, top, w, h):
        return fig.add_axes([l / Wf, (Hf - top - h) / Hf, w / Wf, h / Hf])

    # a: lag x training map
    ax = add(.55, .6, 2.05, 1.45)
    lx = np.log(x)
    xe = np.exp(np.concatenate([[lx[0] - (lx[1] - lx[0]) / 2], (lx[1:] + lx[:-1]) / 2, [lx[-1] + (lx[-1] - lx[-2]) / 2]]))
    nrm = LogNorm(5e-3, .7)
    ax.pcolormesh(xe, np.arange(K + 1) - .5, m.T, cmap="viridis", norm=nrm, shading="flat", rasterized=True)
    ax.axvline(CROSS, color="w", lw=.9, ls=":")
    taxis(ax)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_ylabel("lag $k$ (steps back)", fontsize=FL, labelpad=2)
    cax = add(.55 + 2.05 + .07, .6, .07, 1.45)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap="viridis"), cax=cax)
    cb.set_ticks([.01, .0625, .1, .5])
    cb.set_ticklabels([".01", "1/16", ".1", ".5"])
    cb.ax.tick_params(labelsize=FT - 1, length=2, pad=1.5)
    cb.outline.set_linewidth(.5)
    head(ax, "a", "Credit share by lag", r"$\Vert\delta_k\Vert^2/\Sigma_k\Vert\delta_k\Vert^2$, mean over probes")

    # b: effective number of lags per probe
    ax = add(3.6, .6, 1.8, 1.45)
    lo, hi = np.percentile(pr, [10, 90], 1)
    ax.fill_between(x, lo, hi, color="#1f77b4", alpha=.2, lw=0, label="10-90% of probes")
    blo, bhi = np.percentile(boot, [5, 95], 1)
    ax.fill_between(x, blo, bhi, color="#1f77b4", alpha=.45, lw=0)
    ax.plot(x, pr.mean(1), color="#1f77b4", lw=1.6, marker="o", ms=2.2, label="mean")
    ax.axhline(K, color="k", lw=.9, ls=(0, (4, 2)))
    ax.text(12, K - .4, "even over all 16 lags", fontsize=FT - 1, va="top")
    ax.axvline(CROSS, color="0.4", lw=.9, ls=":")
    taxis(ax)
    ax.set_ylim(0, 17)
    ax.set_ylabel("lags", fontsize=FL, labelpad=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1, .86), frameon=False, fontsize=FT - 1, handlelength=1.2)
    head(ax, "b", "Effective number of lags", r"$1/\Sigma_k s_k^2$ per probe")

    # c: profiles at stages
    ax = add(.55, 3.05, 4.85, 1.35)
    cols = plt.cm.Blues(np.linspace(.35, 1, len(STAGES)))
    for s, c in zip(STAGES, cols):
        i = int(np.argmin(np.abs(np.log(x) - np.log(s))))
        ax.plot(np.arange(K), m[i], color=c, lw=1.6, marker="o", ms=2.5,
                label=f"{x[i]:,.0f}")
    ax.axhline(1 / K, color="k", lw=.9, ls=(0, (4, 2)))
    ax.text(.2, 1 / K * .85, "even: 1/16", fontsize=FT - 1, ha="left", va="top")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 5)
    ax.set_xlim(-.4, 15.4)
    ax.set_xticks(range(0, 16, 3))
    ax.set_xlabel("lag $k$ (steps back from the error)", fontsize=FL, labelpad=1)
    ax.set_ylabel("credit share", fontsize=FL, labelpad=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=FT - 1, handlelength=1.2, ncol=5, columnspacing=.8,
              title="training episodes (Trained RNN; it passes naive at 880)", title_fontsize=FT - 1)
    head(ax, "c", "Credit profile at five stages", "mean share per lag")
    fig.text(.5, .08 / Hf, "2,000 probes, errors at steps 17-30, 32 checkpoints; "
             "b dark band: 5-95% over environments", ha="center", fontsize=FT - 2, color="0.3")
    out = fig_path("td_tt_credit_spread.png")
    fig.savefig(out, dpi=300)
    print("saved", out)


if __name__ == "__main__":
    figure()
