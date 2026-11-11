"""The network state's variance by part - time, trajectory, latent pair, environment - on one panel, with the variance
explained by factorization regret (FR) and by entanglement, through training (John 11-01: "a figure that puts them on
the same panel, along with var explained by Factorization Regret and entanglement").
-> main/REPLICATION/figure_bin/td_variance_parts.png  (rows: Trained RNN, ESN; a: variance, b: share of the total)
u_it = the LSTM output (1,000 neurons) of probe episode i at step t (10,000 probes: 50 environments x 100 latent pairs
x 2 trajectories), steps 2-30; squared lengths are summed over neurons; <.> = the average over all particle-steps.
Parts (td_entanglement_factors.py time_last; each the deviation from its parent group's average; bars: averages over
steps 2-30; mu_ec over 20 reference trajectories of environment e and latent pair c, not the probes):
  time         d = u_it - ubar_i       trajectory   d = ubar_i - mu_ec
  latent pair  d = mu_ec - mu_e        environment  d = mu_e - mu
  plotted <||d||^2> (trajectory and latent pair minus the reference noise); the four add up to the total
  <||u_it - mu||^2> (cross terms ~0); every 3rd checkpoint and the last.
Explained by x (td_entanglement_explained.py; every checkpoint): the variance of the state's means across 20 bins of x
at the same step (x ranked within its step, 500 particles per bin), averaged over steps, minus the bins' sampling noise.
x = FR or entanglement (both variables averaged). These are not further parts: they overlap the four. Fifths instead of
20 bins give ~20% less (they resolve x more coarsely). FR = KL(Bayes || naive) of a variable's marginal belief (the
paper's forward form: probes.npz 'fr') in caches that carry fr_def = "forward"; the caches of 11-01 do not: their FR
curve was computed with the symmetric clipped KL (td_entanglement_explained.py; not rebuilt on 11-10, it needs the
networks) and is then labelled so in the figure. At the last Trained-RNN checkpoint the two differ by 5% (.2304
forward against .2202, 20 bins).
Run from the repository root (CPU, seconds):
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig_variance_parts.py"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter            # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import fig_entanglement_driver as fd                                              # noqa: E402

LINES = (("e_time", "time", r"$d = u_{it} - \bar u_i$", "#d62728", "-"),
         ("e_ep", "trajectory", r"$d = \bar u_i - \bar\mu_{ec}$", "0.5", "-"),
         ("e_real", "latent pair", r"$d = \bar\mu_{ec} - \bar\mu_e$", "#2ca02c", "-"),
         ("e_env", "environment", r"$d = \bar\mu_e - \bar\mu$", "#8c564b", "-"),
         ("fr", "explained by FR", "20 bins of FR (mean of the two variables) per step", "#9467bd", "--"),
         ("entanglement", "explained by entanglement", "20 bins of entanglement (mean of the two variables) per step",
          "#17becf", "--"))
K20 = 1                                                                            # index of the 20-bin estimate


def parts(run, P):
    """the four parts' and the total's <||d||^2> over all particle-steps, every 3rd checkpoint"""
    z = np.load(os.path.join(fd.AN, f"ent_factors_tl_{run}.npz"))
    g = P["env"] * 100 + P["cell"]
    out = {"x": fd.train_x(z["episode"])}
    for key in ("e_tot", "e_time", "e_ep", "e_real", "e_env"):
        v = z[key].astype(np.float64)
        out[key] = v[:, :, fd.S].mean((1, 2)) if v.ndim == 3 else v.mean(1)
    noise = np.array([z["noise"][k][g].mean() for k in range(len(z["episode"]))])
    out["e_ep"] = out["e_ep"] - noise
    out["e_real"] = out["e_real"] - noise
    return out


def explained(run):
    """td_entanglement_explained.py's cache; "fr_forward": does its FR curve use the forward KL(Bayes || naive) (caches
    written since 11-10 say so; older ones used the symmetric clipped KL)"""
    z = np.load(os.path.join(fd.AN, f"ent_explained_{run}.npz"))
    names = list(z["vars"])
    return {"x": fd.train_x(z["episode"]), "tot": z["tot"],
            "fr_forward": "fr_def" in z.files and str(z["fr_def"]) == "forward",
            **{v: z["ve"][:, names.index(v), K20] for v in names}}


def _log_ticks(ax):
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def figure():
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    P = dict(np.load(os.path.join(fd.AN, "probes.npz")))
    cross = fd.passes_naive(P)
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    fwd = all(explained(run)["fr_forward"] for run in fd.RUNS)
    if not fwd:
        print("NOTE: ent_explained_<run>.npz predates 11-10: its FR curve uses the symmetric clipped KL (labelled so)")
    for i, run in enumerate(fd.RUNS):
        pa, ex = parts(run, P), explained(run)
        for j, share in enumerate((False, True)):
            ax = axs[i, j]
            if not share:
                ax.plot(pa["x"], pa["e_tot"], color="k", lw=2.4)
            for key, _, _, col, ls in LINES:
                src = pa if key.startswith("e_") else ex
                y = src[key] / (src["e_tot"] if src is pa else src["tot"]) if share else src[key]
                ax.plot(src["x"], y, color=col, lw=2.2, ls=ls)
            ax.set_yscale("log")
            _log_ticks(ax)
            ax.axvline(cross, color="0.55", lw=1, ls=":")
            fd._time_axis(ax, xlabel=i == 1)
            ax.set_ylabel("share of the total variance" if share else r"variance $\langle \Vert d \Vert^2 \rangle$",
                          fontsize=10)
        axs[i, 0].text(-.17, .5, fd.NAME[run], transform=axs[i, 0].transAxes, rotation=90, va="center", ha="center",
                       fontsize=14, color=fd.COL[run])
        s, e = pa, ex
        print(f"{run}, end of training: total {s['e_tot'][-1]:.2f}; " + ", ".join(
            f"{name} {(s if key.startswith('e_') else e)[key][-1]:.3f} "
            f"({(s if key.startswith('e_') else e)[key][-1] / (s['e_tot'][-1] if key.startswith('e_') else e['tot'][-1]):.1%})"
            for key, name, _, _, _ in LINES))
    axs[0, 0].set_title(r"a   Variance of the state, by part  $\langle \Vert d \Vert^2 \rangle$", loc="left", fontsize=12.5)
    axs[0, 1].set_title(r"b   Share of the total  $\langle \Vert d \Vert^2 \rangle\, /\, \langle \Vert u_{it} - \bar\mu "
                        r"\Vert^2 \rangle$", loc="left", fontsize=12.5)
    h = [Line2D([], [], color="k", lw=2.4, label=r"total  $u_{it} - \bar\mu$")]
    fr_note = r" = KL(Bayes $\Vert$ naive)" if fwd else " (symmetric KL: cache of 11-01)"
    h += [Line2D([], [], color=col, lw=2.2, ls=ls, label=f"{name}{fr_note if key == 'fr' else ''}  {formula}")
          for key, name, formula, col, ls in LINES]
    h += [Line2D([], [], color="0.55", lw=1, ls=":", label="Trained RNN passes naive")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(.5, -.075), ncol=3, frameon=False, fontsize=10)
    fig.text(.5, -.1, "10,000 probe episodes (50 environments x 100 latent pairs x 2 trajectories), steps 2-30; squared "
             r"lengths summed over the 1,000 neurons; $\langle\cdot\rangle$: average over all particle-steps." "\n"
             r"Bars: averages over steps 2-30 ($\bar u_i$: episode $i$'s; $\bar\mu_{ec}$: 20 reference trajectories of "
             r"environment $e$ and latent pair $c$; $\bar\mu_e$, $\bar\mu$: over pairs, environments). The four parts "
             "add up to the total.\nExplained by FR / entanglement: the variance of the mean state across 20 bins of FR / "
             "entanglement at the same step, averaged over steps; it overlaps the four parts.",
             ha="center", va="top", fontsize=9.5, color="0.3", linespacing=1.6)
    fig.tight_layout(h_pad=2.5, w_pad=3)
    path = fig_path("td_variance_parts.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    figure()
