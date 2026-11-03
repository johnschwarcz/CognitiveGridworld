"""Entanglement through training: where the networks' updates are wrong, where they use memory, and the dimension of
each part of their state, all split the same way (John 11-01: "Im still not convinced that entanglement is just a
consequence, rather than a driver of the need for memory ... splitting steps by entanglement and seeing if the PR grows
with entanglement"; then: equal steps per bin, 5 splits, variance first then average, and the state split into time,
trajectory stochasticity, latent pair and environment, "very careful, explicit and simple").
-> main/REPLICATION/figure_bin/td_entanglement_driver.png
Split: a, b each variable's update error / memory use by that variable's own entanglement (step_dkl of the variable;
both variables' values ranked together within each step: 5 fifths, each 4,000 values of every step; audit 11-01: the
first version split the errors averaged over the two variables by rev, their mean); c-f each particle-step's state by
rev (the whole state; 2,000 particles of every step per fifth). Rows: Trained RNN, ESN through training, and both at
the end of training against the percentile.
  a update error: the network's step-t update applied to Bayes's belief at t-1, Jeffreys/2 against Bayes's belief at t
    (per variable); right edge: naive (squares), memory-free Bayes (diamonds), per fifth
  b memory use: the same error with the memory cut to the last step (the cell rerun from its start state over input t)
    minus intact; diamonds: what memory-free Bayes lacks
  c-f the dimension of each part of the state (td_entanglement_parts.py; every 3rd checkpoint and the last). The state
    u_it (1,000 neurons) splits exactly into four parts, each the deviation from its parent group's average (bars:
    averages over steps 2-30; mu_ec over 20 reference trajectories of environment e and latent pair c, not the probes):
      c time         d = u_it - ubar_i       the step vs its own episode's average
      d trajectory   d = ubar_i - mu_ec      the episode's average vs its latent pair's (trajectory stochasticity)
      e latent pair  d = mu_ec - mu_e        the pair's average vs its environment's
      f environment  d = mu_e - mu           the environment's average vs the overall
    M = the average of d d^T over the fifth's particle-steps (trajectory and latent pair minus the reference noise);
    PR = (tr M)^2 / tr(M M). The variances tr M are in td_entanglement_factors.png (same parts, same split).
Data: td_entanglement_memcut.py (a, b), td_entanglement_parts.py (c-f).
Run from the repository root (CPU, ~1 min):
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig_entanglement_driver.py"""
import os
import sys
import glob
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION")]
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from matplotlib.cm import ScalarMappable                                          # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm                        # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402

AN = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics", "analysis")
RUNS = ("trained", "echo")
COL = {"trained": "#1f77b4", "echo": "#ff7f0e"}
NAME = {"trained": "Trained RNN", "echo": "ESN"}
X0, XR1, XR2 = 10, 7.5e4, 1.25e5                                                   # initial network; right-edge columns
S = slice(1, 30)                                                                   # steps 2-30
N_BIN = 5                                                                          # John 11-01: 5 splits
PARTS = (("time", "Time (within the episode)", r"$d = u_{it} - \bar u_i$"),
         ("trajectory", "Trajectory", r"$d = \bar u_i - \bar\mu_{ec}$"),
         ("pair", "Latent pair", r"$d = \bar\mu_{ec} - \bar\mu_e$"),
         ("environment", "Environment", r"$d = \bar\mu_e - \bar\mu$"))


def _jeff2(la, lb):
    from scipy.special import log_softmax
    la, lb = log_softmax(la, -1), log_softmax(lb, -1)
    return .5 * ((np.exp(la) - np.exp(lb)) * (la - lb)).sum(-1)


def inputs():
    """per particle-step: entanglement rev (n, T) and per variable "ent" (n, T, 2), and naive's and memory-free
    (step-wise) Bayes's update errors in the measure of a, per variable (n, T, 2)"""
    P = dict(np.load(os.path.join(AN, "probes.npz")))
    lj = P["log_jb"].astype(np.float64)
    prior = np.concatenate([np.full_like(lj[:, :1], -np.log(lj.shape[-1])), lj[:, :-1]], 1)
    return {"rev": P["ent"].mean(-1), "ent": P["ent"].astype(np.float64), "P": P,
            "naive": _jeff2(prior + P["inc_N"].astype(np.float64), lj),
            "stepwise": _jeff2(prior + P["inc_SW"].astype(np.float64), lj)}


def _rank_bins(x, nb, axis=0):
    """bin of each value by its rank along axis: exactly len / nb values per bin (ties broken by order)"""
    r = np.argsort(np.argsort(x, axis, kind="stable"), axis, kind="stable")
    return r * nb // x.shape[axis]


def within_step(x, nb):
    """(n, T) -> (n, 29) bin of x by its rank within each step 2-30: every bin holds exactly n / nb particles of every
    step"""
    b = _rank_bins(x[:, S], nb)
    counts = np.stack([np.bincount(b[:, t], minlength=nb) for t in range(b.shape[1])])
    assert (counts == x.shape[0] // nb).all(), "unequal step counts"
    return b


def per_var_bins(x, nb):
    """(n, T, 2) -> (n, 29, 2): fifths of both variables' values ranked together within each step 2-30"""
    n = x.shape[0]
    b = within_step(np.concatenate([x[..., 0], x[..., 1]], 0), nb)
    return np.stack([b[:n], b[n:]], -1)


def _mean_by(v, b, nb):
    return np.bincount(b.ravel(), v.ravel(), nb) / np.bincount(b.ravel(), minlength=nb)


def train_x(ep):
    ep = np.asarray(ep)
    return np.maximum(ep - ep // 5, X0).astype(float)


def series(run, I):
    """a, b per fifth of each variable's own entanglement at every checkpoint (per-variable memory-cut pass); c-f per
    fifth of rev at every 3rd checkpoint (parts pass)"""
    B = per_var_bins(I["ent"], N_BIN)
    m = np.load(os.path.join(AN, f"ent_memcut_var_{run}.npz"))
    out = {"x_ab": train_x(m["episode"]), "a": [], "b": []}
    for k in range(len(m["episode"])):
        ei, ec = m["err_int"][k][:, S].astype(np.float64), m["err_cut1"][k][:, S].astype(np.float64)
        out["a"].append(_mean_by(ei, B, N_BIN))
        out["b"].append(_mean_by(ec - ei, B, N_BIN))
    out["a"], out["b"] = np.array(out["a"]), np.array(out["b"])
    out["naive"] = _mean_by(I["naive"][:, S], B, N_BIN)
    out["stepwise"] = _mean_by(I["stepwise"][:, S], B, N_BIN)
    z = np.load(os.path.join(AN, f"ent_parts_{run}.npz"))
    names = list(z["parts"])
    out["x_parts"] = train_x(z["episode"])
    for key, _, _ in PARTS:
        out[key] = z["pr"][:, names.index(key)]
        out["var_" + key] = z["tr"][:, names.index(key)]
    return out


def passes_naive(P):
    """training episodes at which the trained RNN's goal accuracy at T first exceeds naive's"""
    naive = P["right_naive"][np.arange(len(P["goal"])), P["goal"]].mean()
    for f in sorted(glob.glob(os.path.join(AN, "trained", "ck_*.npz"))):
        z = np.load(f)
        if z["acc"][-1] > naive:
            return float(train_x(int(z["episode"])))


def _time_axis(ax, xlabel=True):
    ax.set_xscale("log")
    ax.set_xlim(X0 * .8, 1.7e5)
    ax.set_xticks([X0, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.minorticks_off()
    ax.axvline(5.2e4, color="0.85", lw=.8)
    if xlabel:
        ax.set_xlabel("training episodes", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)


def figure(I=None):
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    I = I or inputs()
    D = {run: series(run, I) for run in RUNS}
    cross = passes_naive(I["P"])
    kcol = plt.cm.Purples(np.linspace(.3, 1., N_BIN))
    pct = 100 * (np.arange(N_BIN) + .5) / N_BIN
    cols = (("a", "a   Update error", r"$J(\pi^B_{t-1} \cdot \ell^{net}_t,\ \pi^B_t)$"),
            ("b", "b   Memory use", "error with memory cut to the last step $-$ intact")) \
        + tuple((key, f"{'cdef'[j]}   {name}", formula) for j, (key, name, formula) in enumerate(PARTS))
    fig, axs = plt.subplots(3, 6, figsize=(25, 11), gridspec_kw={"height_ratios": [1, 1, .9]})
    for j, (key, title, formula) in enumerate(cols):
        xk = "x_ab" if key in "ab" else "x_parts"
        for i, run in enumerate(RUNS):
            ax = axs[i, j]
            d = D[run]
            for k in range(N_BIN):
                ax.plot(d[xk], d[key][:, k], color=kcol[k], lw=2)
            if key in "ab":
                for k in range(N_BIN):
                    if key == "a":
                        ax.plot(XR1, d["naive"][k], "s", ms=5, color=kcol[k], mec="0.45", mew=.8)
                    ax.plot(XR2, d["stepwise"][k], "D", ms=4.5, color=kcol[k], mec="k", mew=.8)
            if key == "b":
                ax.axhline(0, color="0.6", lw=.8)
            ax.axvline(cross, color="0.55", lw=1, ls=":")
            _time_axis(ax, xlabel=i == 1)
            if i == 0:
                ax.set_title(f"{title}\n{formula}", loc="left", fontsize=12, linespacing=1.6)
        ax = axs[2, j]
        for run in RUNS:
            ax.plot(pct, D[run][key][-1], color=COL[run], lw=2, marker="o", ms=4)
        if key == "a":
            ax.plot(pct, D["trained"]["naive"], color="0.55", lw=1.5, marker="s", ms=3)
        if key in "ab":
            ax.plot(pct, D["trained"]["stepwise"], color="k", lw=1.5, marker="D", ms=3)
        if key == "b":
            ax.axhline(0, color="0.6", lw=.8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("entanglement percentile", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    axs[2, 0].text(-.3, .5, "end of training", transform=axs[2, 0].transAxes, rotation=90, va="center", ha="center",
                   fontsize=13, color="0.25")
    for i, run in enumerate(RUNS):
        axs[i, 0].text(-.3, .5, NAME[run], transform=axs[i, 0].transAxes, rotation=90, va="center", ha="center",
                       fontsize=14, color=COL[run])
    for i in range(3):
        axs[i, 0].set_yscale("log")
        axs[i, 0].set_ylim(1.5e-3, .3)
        axs[i, 0].set_ylabel("error of the step's belief update (nats)", fontsize=9.5)
        axs[i, 1].set_ylabel("cut $-$ intact error (nats)", fontsize=9.5)
        for j in range(2, 6):
            axs[i, j].set_ylabel(r"PR of $\langle d\, d^\top \rangle$ (dimensions)", fontsize=9.5)
    h = [Line2D([], [], color=COL[r], lw=2, marker="o", ms=4, label=NAME[r]) for r in RUNS]
    h += [Line2D([], [], color="0.55", lw=1.5, marker="s", ms=4, label="naive"),
          Line2D([], [], color="k", lw=1.5, marker="D", ms=4, label="memory-free Bayes"),
          Line2D([], [], color="0.55", lw=1, ls=":", label="Trained RNN passes naive")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(.42, -.035), ncol=5, frameon=False, fontsize=10)
    fig.text(.5, -.075, "10,000 probe episodes (50 environments x 100 latent pairs x 2 trajectories), steps 2-30. a, b: each "
             "variable's error by its own entanglement (both variables ranked together within each step, 4,000 per "
             "fifth); c-f: particle-steps by rev, the mean of the two variables' entanglement (2,000 of every step per "
             "fifth); values: averages over the fifth.\n"
             r"$\bar u_i$: episode $i$'s average state;  $\bar\mu_{ec}$: the average over 20 reference trajectories of "
             r"environment $e$ and latent pair $c$;  $\bar\mu_e$, $\bar\mu$: its averages over pairs, environments.  "
             r"$\langle d\,d^\top\rangle$: over the fifth's particle-steps;  PR $= (\mathrm{tr}\,M)^2 / \mathrm{tr}\,M^2$.",
             ha="center", va="top", fontsize=9.5, color="0.3", linespacing=1.6)
    fig.tight_layout(h_pad=2.0, w_pad=1.6)
    cax = fig.add_axes([.72, -.02, .16, .012])
    sm = ScalarMappable(norm=BoundaryNorm(np.linspace(0, 100, N_BIN + 1), N_BIN), cmap=ListedColormap(kcol))
    fig.colorbar(sm, cax=cax, orientation="horizontal").set_label("entanglement percentile (rows 1-2)", fontsize=9.5)
    path = fig_path("td_entanglement_driver.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path, "; trained passes naive at", cross)
    for run in RUNS:
        d = D[run]
        print(f"  {run}, end of training, by fifth: a {np.round(d['a'][-1], 4)}, b {np.round(d['b'][-1], 4)}")
        for key, name, _ in PARTS:
            print(f"    {name:26s} PR {np.round(d[key][-1], 1)}   variance {np.round(d['var_' + key][-1], 3)}")
    return path


if __name__ == "__main__":
    figure()
