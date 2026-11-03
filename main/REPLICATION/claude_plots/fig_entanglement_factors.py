"""Variance of the network state by factor - time, environment, joint realization, episode - split by entanglement,
through training (John 11-01: "Maybe I need to see curves for neural variance across time, environment, episode and
joint realization, split by entanglement."; then: "take the variance across timesteps, and then average over episodes.
That is what I would think is the contribution of time").
-> main/REPLICATION/figure_bin/td_entanglement_factors.png             time within the episode (John's definition)
   main/REPLICATION/figure_bin/td_entanglement_factors_time_first.png  time = the course shared by all episodes
Data: td_entanglement_factors.py (ent_factors_tl_<run>.npz / ent_factors_<run>.npz). Every probe state u_it (episode i
of environment e with latent pair c = (r0, r1), step t; u = the LSTM output) is split exactly into parts measured
against means from a separate reference set (20 new trajectories per environment and pair), squared lengths summed
over the 1,000 neurons:
  time within the episode (main):  u_it - m = (u_it - ubar_i) + (ubar_i - m_ec) + (m_ec - m_e) + (m_e - m)
    time = the step's deviation from its own episode's average state (averaged over a whole episode: its variance
    across its own steps); trajectory, latent pair and environment compare time-averaged states (bars and m: means
    over steps 2-30; m_ec over the reference trajectories of e and c); the dimension (PR) of each part is in
    td_entanglement_driver.png (c-f);
  time first:  u_it - mu = (mu_t - mu) + (mu_et - mu_t) + (mu_ect - mu_et) + (u_it - mu_ect)
    time = the time course shared by all episodes; every other part is defined at each step.
Bins: each particle-step by its own entanglement (rev, both variables averaged), ranked within its step: 5 fifths of
2,000 particles from every step 2-30. Plotted: the mean over the fifth's particle-steps of each part's squared length
(trajectory and latent pair minus the reference means' sampling noise, which adds the same expected amount to both;
the parts that are constant over an episode count once per step); total = the four parts + the cross terms (column f).
Rows: Trained RNN, ESN through training (every 3rd checkpoint and the last), and both at the end against the percentile.
Run from the repository root (CPU, ~1 min):
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig_entanglement_factors.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from matplotlib.cm import ScalarMappable                                          # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm                        # noqa: E402
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter            # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import fig_entanglement_driver as fd                                              # noqa: E402

N_BIN = 5
MODES = {
    "time_last": dict(
        tag="_tl", out="td_entanglement_factors.png",
        parts=(("e_tot", "Total", r"$\Vert u_{it} - \bar\mu \Vert^2$"),
               ("e_time", "Time (within the episode)", r"$\Vert u_{it} - \bar u_i \Vert^2$"),
               ("e_ep", "Trajectory", r"$\Vert \bar u_i - \bar\mu_{ec} \Vert^2$"),
               ("e_real", "Latent pair", r"$\Vert \bar\mu_{ec} - \bar\mu_e \Vert^2$"),
               ("e_env", "Environment", r"$\Vert \bar\mu_e - \bar\mu \Vert^2$")),
        note=r"$\bar u_i$: episode $i$'s own average state over steps 2-30;  $\bar\mu_{ec}$: the average over 20 "
             r"reference trajectories (not the probes) of environment $e$ and latent pair $c = (r_0, r_1)$;  "
             r"$\bar\mu_e$, $\bar\mu$: its means over pairs, environments."),
    "time_first": dict(
        tag="", out="td_entanglement_factors_time_first.png",
        parts=(("e_tot", "Total", r"$\Vert u_{it} - \mu \Vert^2$"),
               ("e_time", "Time (shared by all episodes)", r"$\Vert \mu_t - \mu \Vert^2$"),
               ("e_env", "Environment", r"$\Vert \mu_{et} - \mu_t \Vert^2$"),
               ("e_real", "Latent pair", r"$\Vert \mu_{ect} - \mu_{et} \Vert^2$"),
               ("e_ep", "Trajectory", r"$\Vert u_{it} - \mu_{ect} \Vert^2$")),
        note=r"$\mu_{ect}$: mean state of environment $e$, latent pair $c = (r_0, r_1)$, step $t$ over 20 reference "
             r"trajectories (not the probes);  $\mu_{et}$, $\mu_t$, $\mu$: its means over pairs, environments, steps 2-30."),
}


def series(run, B, P, mode):
    M = MODES[mode]
    z = np.load(os.path.join(fd.AN, f"ent_factors{M['tag']}_{run}.npz"))
    g = P["env"] * 100 + P["cell"]                                                  # each probe's (environment, pair)
    K, nT = len(z["episode"]), B.shape[1]
    out = {"x": fd.train_x(z["episode"])}
    for key, _, _ in M["parts"]:
        out[key] = np.zeros((K, N_BIN))
    out["cross"] = np.zeros((K, N_BIN))
    for k in range(K):
        noise = z["noise"][k][g]
        noise = noise[:, fd.S] if noise.ndim == 2 else noise[:, None]
        parts = {}
        for key, _, _ in M["parts"]:
            v = z[key][k].astype(np.float64)
            parts[key] = v[:, fd.S] if v.ndim == 2 else np.broadcast_to(v[:, None], (len(v), nT))
        parts["e_real"] = parts["e_real"] - noise
        parts["e_ep"] = parts["e_ep"] - noise
        for key in parts:
            out[key][k] = fd._mean_by(np.ascontiguousarray(parts[key]), B, N_BIN)
        out["cross"][k] = out["e_tot"][k] - sum(out[key][k] for key, _, _ in M["parts"][1:])
    return out


def _log_ticks(ax):
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def figure(mode="time_last", I=None):
    M = MODES[mode]
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    I = I or fd.inputs()
    P = I["P"]
    B = fd.within_step(I["rev"], N_BIN)
    D = {run: series(run, B, P, mode) for run in fd.RUNS}
    cross = fd.passes_naive(P)
    kcol = plt.cm.Purples(np.linspace(.3, 1., N_BIN))
    pct = 100 * (np.arange(N_BIN) + .5) / N_BIN
    cols = M["parts"] + (("cross", "Cross terms", "total $-$ the four parts"),)
    fig, axs = plt.subplots(3, len(cols), figsize=(4.0 * len(cols), 11), gridspec_kw={"height_ratios": [1, 1, .9]})
    for j, (key, name, formula) in enumerate(cols):
        log = key != "cross"
        for i, run in enumerate(fd.RUNS):
            ax = axs[i, j]
            d = D[run]
            for k in range(N_BIN):
                ax.plot(d["x"], d[key][:, k], color=kcol[k], lw=2)
            if log:
                ax.set_yscale("log")
            else:
                ax.axhline(0, color="0.6", lw=.8)
            ax.axvline(cross, color="0.55", lw=1, ls=":")
            fd._time_axis(ax, xlabel=i == 1)
            if i == 0:
                ax.set_title(f"{'abcdef'[j]}   {name}\n{formula}", loc="left", fontsize=12, linespacing=1.6)
            if key == "e_time" and mode == "time_first":
                ax.text(.04, .04, "the same in every fifth\n(each fifth holds the same steps)", transform=ax.transAxes,
                        fontsize=9, color="0.3", va="bottom")
        ax = axs[2, j]
        for run in fd.RUNS:
            ax.plot(pct, D[run][key][-1], color=fd.COL[run], lw=2, marker="o", ms=4)
        if log:
            ax.set_yscale("log")
        else:
            ax.axhline(0, color="0.6", lw=.8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("entanglement percentile", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    for i in range(3):
        axs[i, 0].set_ylabel("mean squared length (variance units)", fontsize=9.5)
    for i, run in enumerate(fd.RUNS):
        axs[i, 0].text(-.32, .5, fd.NAME[run], transform=axs[i, 0].transAxes, rotation=90, va="center", ha="center",
                       fontsize=14, color=fd.COL[run])
    axs[2, 0].text(-.32, .5, "end of training", transform=axs[2, 0].transAxes, rotation=90, va="center", ha="center",
                   fontsize=13, color="0.25")
    for j in range(len(M["parts"])):                                               # one y-range per network and column
        for i, run in enumerate(fd.RUNS):
            v = D[run][M["parts"][j][0]]
            v = v[v > 0]
            axs[i, j].set_ylim(v.min() / 1.5, v.max() * 1.5)
        for i in range(3):
            _log_ticks(axs[i, j])
    if mode == "time_first":
        axs[2, 1].set_ylim(.3, 1.2)
    h = [Line2D([], [], color=fd.COL[r], lw=2, marker="o", ms=4, label=fd.NAME[r]) for r in fd.RUNS]
    h += [Line2D([], [], color="0.55", lw=1, ls=":", label="Trained RNN passes naive")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(.38, -.035), ncol=3, frameon=False, fontsize=10)
    fig.text(.5, -.075, "10,000 probe episodes (50 environments x 100 latent pairs x 2 streams), steps 2-30; lines: fifths "
             "of the particle-step's entanglement (rev: the mean of the two variables), ranked within its step (2,000 particles "
             "of every step per fifth); "
             "values: means over the fifth.\n" + M["note"] + "  Trajectory and latent pair: minus the reference "
             "means' sampling noise.", ha="center", va="top", fontsize=9.5, color="0.3", linespacing=1.6)
    fig.tight_layout(h_pad=2.0, w_pad=1.6)
    cax = fig.add_axes([.66, -.02, .16, .012])
    sm = ScalarMappable(norm=BoundaryNorm(np.linspace(0, 100, N_BIN + 1), N_BIN), cmap=ListedColormap(kcol))
    fig.colorbar(sm, cax=cax, orientation="horizontal").set_label("entanglement percentile (rows 1-2)", fontsize=9.5)
    path = fig_path(M["out"])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    for run in fd.RUNS:
        d = D[run]
        print(f"  {run}, end of training, by fifth (least -> most entangled):")
        for key, name, _ in M["parts"]:
            print(f"    {name:30s} {np.round(d[key][-1], 3)}  (all: {d[key][-1].mean():.3f})")
        print(f"    {'cross terms':30s} {np.round(d['cross'][-1], 3)}")
    return path


if __name__ == "__main__":
    I = fd.inputs()
    figure("time_last", I)
    figure("time_first", I)
