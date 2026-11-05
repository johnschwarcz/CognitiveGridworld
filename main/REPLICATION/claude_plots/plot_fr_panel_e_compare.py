"""Why the 'FR against accuracy' panel differs between the two figures, one change at a time:
    left    supplementary_plotter2's diagnostics_panel_early (e): the logged logistic slope, first 800 tests
    middle  the same slope over all 10,001 tests
    right   fig1's (e): r implied by that slope (supplementary_plotter2.fr_correlation), all tests, smoothed accuracy on x
Reads supplementary_harvest.npz and fig1's fr_corr_implied.npz -> figure_bin/fr_panel_e_compare.png"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.dirname(os.path.dirname(os.path.dirname(HERE)))]
import supplementary_plotter2 as sp                                              # noqa: E402
from main.utils import fig_path, apply_fig_style                                 # noqa: E402

apply_fig_style()
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "DATA")
H = np.load(sp.HARVEST_CACHE)
F = np.load(os.path.join(DATA, "experimental", "fig1", "fr_corr_implied.npz"))
NETS = (("Trained", "tr_", "trained", "tab:blue"), ("ESN", "ec_", "echo", "tab:orange"))
EARLY = sp.PLOT_CFG["early_epochs"]

fig, axes = plt.subplots(1, 3, figsize=(8.7, 2.9), sharex=True, sharey=True)
for lab, tag, key, c in NETS:
    acc = H[tag + "test_acc_through_training"][:, -1].astype(np.float64)
    b = H[tag + "test_SII_coef_through_training"].astype(np.float64)
    axes[0].plot(acc[:EARLY], b[:EARLY], c=c, lw=.8, label=lab)
    axes[1].plot(acc, b, c=c, lw=.5)
    axes[2].plot(sp._smooth(acc, 11), F[key], c=c, lw=.5)
    fr, net = H[tag + "fr"].astype(np.float64), H[tag + "net_acc"].astype(np.float64)
    exact = np.mean([np.corrcoef(f, a)[0, 1] for f, a in zip(fr, net)])
    print(f"{lab}: end of training  slope {b[-1000:].mean():.3f}  implied r {F[key][-1000:].mean():.3f}  "
          f"exact r on the final network {exact:.3f}  acc SD {acc[-1000:].std():.4f}")
titles = (("Logistic slope, first 800 tests", r"$\beta$:  logit $p(\mathrm{correct}) = \alpha + \beta\,\mathrm{FR}$"),
          ("Logistic slope, all tests", r"$\beta$:  logit $p(\mathrm{correct}) = \alpha + \beta\,\mathrm{FR}$"),
          ("Correlation, all tests", r"$r(\mathrm{FR}, \mathrm{correct})$"))
for ax, (name, math) in zip(axes, titles):
    ax.set_title(f"{name}\n{math}", fontsize=10)
    ax.axhline(0, c="k", lw=.5, alpha=.5)
    for v, ls in ((float(np.mean(H["tr_naive_acc"])), "--"), (float(np.mean(H["tr_joint_acc"])), ":")):
        ax.axvline(v, c="0.45", ls=ls, lw=.8)
    ax.set_xlabel("Final-step accuracy")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_xlim(.05, .82)
axes[0].set_ylabel("Value of the plotted quantity")
axes[0].legend(frameon=False, loc="lower right")
fig.text(.995, .005, "1 run per network; 1 point per test", ha="right", va="bottom", fontsize=8, color="0.3")
fig.tight_layout(rect=(0, .03, 1, 1))
fig.savefig(fig_path("fr_panel_e_compare.png"), dpi=200, bbox_inches="tight")
