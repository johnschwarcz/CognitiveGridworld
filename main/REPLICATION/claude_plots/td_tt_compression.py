"""Does low FR allow greater compression? (John 11-13: "I'm trying to understand if there is something about compression,
i.e. low FR allows greater compression"; then "was this checking for compression across different Z?" (yes: i below is
how a realization's code changes across Z), "are you sure measuring corrs of log scaled measures was correct?", "was
this for all steps or final steps? does it matter?", "can you also show the raw departure split by FR percentiles").
RESULTS (11-13, 8 checkpoints 300-40k): within each realization the Trained RNN's context-dependent part grows with FR:
||i||^2 from the lowest to the highest FR decile 7.9 -> 11.2 (300), 14.4 -> 17.4 (2,000), 5.8 -> 7.2 (40k), steepest in
the top two deciles; the ESN's is flat to U-shaped (.58 / .48 / .61 at 300). Within-realization Spearman rho at every
step: Trained +.14-.22 (300) -> +.25-.35 (40k), largest at steps 5-10, +.15-.25 at step 30; ESN +.25 at step 1, ~+.05
from step 7, the same at every checkpoint (input-driven). The transform does not matter for the Trained RNN (step 30:
ranks +.16 -> +.25, raw with environment and realization means out +.18 -> +.34, logs +.17 -> +.28); the ESN's level
does (+.06 / +.18 / +.10). The trajectory residual rises with FR in both networks (the ESN's too: input-driven). Not more
dimensions: PR of i in the high / low / random FR thirds 98 / 89 / 100 at 40k, no consistent sign. Parts' mean squared
norms = td_tt_griddim's variances (1e-15); step 30 of steps_* = the final-step harvest exactly.
The grid decomposition of td_tt_griddim, per state: u_T of the 10,000 probes
(50 environments e x 100 cells c x 2 streams; T = 30) splits exactly (balanced design) into
    u_T = mu + a(c) [shared grid] + b(e) [environment offset] + i(e, c) [context-dependent grid] + eps [trajectory]
The compressed code mu + a + b is low-dimensional (PR ~6 and ~23 at 40k training episodes, td_tt_griddim); i and eps are
high-dimensional (~105, ~109). Test: within each cell (its 50 environments; the cell's place on the grid, which sets FR
across cells (rho .96 with the distance from the centre), held fixed), Spearman rho between FR(e, c) (the probes' FR_T,
both variables averaged, mean of the two streams) and the size of each high-dimensional part, ||i(e, c)||^2 and
||eps||^2 (= ||u_1 - u_2||^2 / 4), then the mean over the 100 cells (average after the statistic); and the same parts'
mean size in each cell's highest- and lowest-FR third of environments (absolute units). Stores u_T (float16, (50, 100, 2,
H)) per checkpoint for follow-ups.
-> main/DATA/experimental/training_dynamics/through_training/compression/<run>_<raw>.npz
Run from the repository root (CPU, 5 workers x 3 threads, ~1 min per checkpoint each):
    /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_compression.py [stats | steps | figure]
(steps: every step, 7 workers, ~5 min; figure: td_tt_compression.png from both harvests)"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "3"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np                                                                            # noqa: E402

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import td_tt_gap as gap                                                                       # noqa: E402

OUT = os.path.join(gap.TT, "compression")
AN = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics", "analysis")
XS = (300, 560, 800, 1040, 2000, 5000, 10000, 39996)                                          # training episodes
NE, NC = 50, 100


def raws_for(run, xs):
    """the saved checkpoints (raw episodes) nearest the training episodes xs, once each"""
    import td_tt_intervene as ti
    raws = np.array(ti.checkpoints(run))
    return sorted({int(raws[np.argmin(np.abs(gap.train_x(raws) - x))]) for x in xs})


def arrange(v, env, cell):
    """v (10,000, ...) -> (50, 100, 2, ...) by environment, cell, stream"""
    X = np.zeros((NE, NC, 2) + v.shape[1:], v.dtype)
    cnt = np.zeros((NE, NC), int)
    for k in range(len(v)):
        X[env[k], cell[k], cnt[env[k], cell[k]]] = v[k]
        cnt[env[k], cell[k]] += 1
    assert (cnt == 2).all()
    return X


def harvest(job):
    """one (run, raw): u_T of the probes (as td_tt_griddim.run_full), its parts' squared norms -> OUT/<run>_<raw>.npz"""
    import torch
    import td_tt_chaos_ent as ce
    import td_tt_intervene as ti
    torch.set_num_threads(3)
    nm, raw = job
    P = ti.probes()
    env, cell = P["env"], P["r"][:, 0] * 10 + P["r"][:, 1]
    p, table = ce.setup(nm, raw, torch.device("cpu"))
    U = []
    with torch.no_grad():
        for b0 in range(0, len(env), 2500):
            envt, code = ce.codes(np.arange(b0, b0 + 2500), torch.device("cpu"))
            U.append(ce.replay(p, table, envt, code)[1][:, 30].double().numpy())
    X = arrange(np.concatenate(U), env, cell)                                                 # (50, 100, 2, H)
    g = X.mean((0, 1, 2))
    b = X.mean((1, 2)) - g
    a = X.mean((0, 2)) - g
    m = X.mean(2)
    i = m - g - b[:, None] - a[None]
    eps = X[:, :, 0] - m
    path = os.path.join(OUT, f"{nm}_{raw:05d}.npz")
    np.savez(path, raw=raw, x=float(gap.train_x(raw)), u=X.astype(np.float16), i2=(i ** 2).sum(-1),
             e2=(eps ** 2).sum(-1), a2=(a ** 2).sum(-1), b2=(b ** 2).sum(-1))
    return path


def harvest_steps(job):
    """one (run, raw), every step s = 1..30 (John 11-13: "was this for all steps or final steps? does it matter?"):
    the parts' squared norms i2, e2 (30, 50, 100), a2 (30, 100), b2 (30, 50) and the PR of i over each cell's
    highest / lowest / a random third of environments by FR_s (pr_hi, pr_lo, pr_rnd (30,)) -> OUT/steps_<run>_<raw>.npz"""
    import torch
    import td_tt_chaos_ent as ce
    import td_tt_intervene as ti
    torch.set_num_threads(3)
    nm, raw = job
    P = ti.probes()
    env, cell = P["env"], P["r"][:, 0] * 10 + P["r"][:, 1]
    p, table = ce.setup(nm, raw, torch.device("cpu"))
    U = []
    with torch.no_grad():
        for b0 in range(0, len(env), 2500):
            envt, code = ce.codes(np.arange(b0, b0 + 2500), torch.device("cpu"))
            U.append(ce.replay(p, table, envt, code)[1][:, 1:].numpy())          # after 1 .. 30 observations
    X = arrange(np.concatenate(U), env, cell)                                     # (50, 100, 2, 30, H) float32
    FR = fr_steps()
    rng = np.random.default_rng(0)
    rnd = np.stack([rng.permutation(NE)[:17] for _ in range(NC)], 1)
    cols = np.arange(NC)
    out = {k: [] for k in ("i2", "e2", "a2", "b2", "pr_hi", "pr_lo", "pr_rnd")}
    for s in range(X.shape[3]):
        Xs = X[:, :, :, s].astype(np.float64)
        g = Xs.mean((0, 1, 2))
        b = Xs.mean((1, 2)) - g
        a = Xs.mean((0, 2)) - g
        m = Xs.mean(2)
        i = m - g - b[:, None] - a[None]
        out["i2"].append((i ** 2).sum(-1))
        out["e2"].append(((Xs[:, :, 0] - m) ** 2).sum(-1))
        out["a2"].append((a ** 2).sum(-1))
        out["b2"].append((b ** 2).sum(-1))
        order = np.argsort(FR[s], 0)
        for k, e in (("hi", order[-17:]), ("lo", order[:17]), ("rnd", rnd)):
            V = i[e, cols[None, :]].reshape(-1, i.shape[-1])
            V = V - V.mean(0)
            ev = np.clip(np.linalg.eigvalsh(V.T @ V / len(V)), 0, None)
            out["pr_" + k].append(ev.sum() ** 2 / (ev ** 2).sum())
    path = os.path.join(OUT, f"steps_{nm}_{raw:05d}.npz")
    np.savez(path, raw=raw, x=float(gap.train_x(raw)), **{k: np.array(v) for k, v in out.items()})
    return path


def fr_steps():
    """FR_s (both variables averaged) per step, environment and cell, the mean of the two streams (30, 50, 100)"""
    P = np.load(os.path.join(AN, "probes.npz"))
    cell = P["r"][:, 0] * 10 + P["r"][:, 1]
    return np.moveaxis(arrange(P["fr"].astype(np.float64).mean(-1), P["env"], cell).mean(2), -1, 0)


def fr_cells():
    """FR_T (both variables averaged) per (environment, cell), the mean of the two streams (50, 100)"""
    P = np.load(os.path.join(AN, "probes.npz"))
    cell = P["r"][:, 0] * 10 + P["r"][:, 1]
    return arrange(P["fr"][:, -1].astype(np.float64).mean(-1), P["env"], cell).mean(2)


def stats():
    from scipy.stats import spearmanr
    import glob
    fr = fr_cells()
    lo, hi = (np.argsort(fr, 0)[s] for s in (slice(0, 17), slice(-17, None)))             # per cell: environments
    cols = np.arange(NC)
    print("within each cell (50 environments), mean over cells: rho(FR, ||part||^2); part size in the cell's high- /"
          " low-FR third")
    for nm in ("trained", "echo"):
        for f in sorted(glob.glob(os.path.join(OUT, f"{nm}_*.npz")), key=lambda s: int(s[-9:-4])):
            z = np.load(f)
            row = [f"{nm:7s} {int(z['x']):6d}:"]
            for k in ("i2", "e2"):
                v = z[k]
                rho = np.mean([spearmanr(fr[:, c], v[:, c])[0] for c in cols])
                row.append(f"{k} rho {rho:+.2f} (hi {v[hi, cols].mean():6.2f} / lo {v[lo, cols].mean():6.2f})")
            row.append(f"| ||a||^2 {z['a2'].mean():6.2f} ||b||^2 {z['b2'].mean():6.2f}")
            print(" ".join(row))


def by_decile(v, fr):
    """v, fr (50, 100): within each realization (column) its 50 environments ranked by FR, 10 bins of 5; the mean of v
    in each bin over the bin's environments and the 100 realizations (10,)"""
    order = np.argsort(fr, 0)
    return np.array([np.take_along_axis(v, order[5 * k:5 * k + 5], 0).mean() for k in range(10)])


def rho_within(fr, v):
    """Spearman rho between fr and v over each realization's 50 environments, then the mean over the 100 realizations"""
    from scipy.stats import spearmanr
    return np.nanmean([spearmanr(fr[:, c], v[:, c])[0] for c in range(NC)])


def figure():
    """-> figure_bin/td_tt_compression.png (John 11-13: "can you also show the raw departure split by FR percentiles",
    "was this for all steps or final steps? does it matter?"). Rows Trained RNN, ESN; lines the 8 checkpoints (light to
    dark through training). a ||i(e, c)||^2 and b ||eps||^2 at step 30 by FR decile within each realization (its 50
    environments ranked by FR, 5 per decile; mean over the decile and the 100 realizations; raw units); c the
    within-realization Spearman rho of FR_s and ||i_s||^2 at every step s (steps_*.npz)"""
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from main.utils import fig_path, apply_fig_style
    import supplementary_plotter2 as sp
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    fr, FR = fr_cells(), fr_steps()
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.2))
    dec = np.arange(1, 11)
    for row, (run, name, cmap) in enumerate((("trained", "Trained", plt.cm.Blues), ("echo", "Echo", plt.cm.Oranges))):
        fs = [os.path.join(OUT, f"{run}_{raw:05d}.npz") for raw in raws_for(run, XS)]          # the 8 of XS
        cols = cmap(np.linspace(.35, 1, len(fs)))
        for f, c in zip(fs, cols):
            z = np.load(f)
            lab = f"{int(z['x']):,}"
            axs[row, 0].plot(dec, by_decile(z["i2"], fr), c=c, lw=2, marker="o", ms=3.5, label=lab)
            axs[row, 1].plot(dec, by_decile(z["e2"], fr), c=c, lw=2, marker="o", ms=3.5)
            st = np.load(os.path.join(OUT, f"steps_{run}_{int(z['raw']):05d}.npz"))
            axs[row, 2].plot(np.arange(1, 31), [rho_within(FR[k], st["i2"][k]) for k in range(30)], c=c, lw=2)
        axs[row, 0].set_ylabel(f"{name}\nmean squared norm", fontsize=11)
        axs[row, 0].yaxis.label.set_color(sp.AGENT_COLORS[name])
        axs[row, 2].legend(*axs[row, 0].get_legend_handles_labels(), title="training episodes", frameon=False,
                           loc="lower center" if row == 0 else "upper right", ncols=4 if row == 0 else 2,
                           fontsize=8.5, title_fontsize=9, handlelength=1.6, columnspacing=.8)   # empty there
        axs[row, 1].set_ylabel("mean squared norm")
        axs[row, 2].set_ylabel(r"Spearman $\rho$, mean over realizations")
        axs[row, 2].axhline(0, c="k", alpha=.5, lw=.8)
    titles = (("Context-dependent part by FR decile", r"$\Vert i(e,c)\Vert^2$ at step 30, "
               r"$i = \mathbb{E}[u|\mathbf{r},Z] - \mathbb{E}[u|\mathbf{r}] - \mathbb{E}[u|Z] + \mathbb{E}[u]$"),
              ("Trajectory residual by FR decile", r"$\Vert \epsilon \Vert^2$ at step 30, "
               r"$\epsilon = u - \mathbb{E}[u|\mathbf{r},Z]$"),
              ("FR and the context-dependent part by step", r"$\rho(\mathrm{FR}_s, \Vert i_s\Vert^2)$ over each "
               "realization's 50 environments"))
    for j, (t, f) in enumerate(titles):
        axs[0, j].set_title(t, fontsize=12.5, pad=19)
        axs[0, j].annotate(f, xy=(.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                           ha="center", va="baseline", fontsize=9.5, color="0.15")
        for a in axs[:, j]:
            a.spines[["top", "right"]].set_visible(False)
            a.grid(True, ls="--", alpha=.3)
            if j < 2:
                a.set_xticks(dec)
                a.set_xlabel("FR decile within the realization (1 = lowest)")
            else:
                a.set_xlim(1, 30)
                a.set_xlabel("step $s$")
    lo = min(a.get_ylim()[0] for a in axs[:, 2])
    hi = max(a.get_ylim()[1] for a in axs[:, 2])
    for a in axs[:, 2]:
        a.set_ylim(lo, hi)
    fig.text(.995, .005, "10,000 probes = 50 environments x 100 realizations x 2 trajectories; deciles: each "
             "realization's environments by its FR (5 each); 8 checkpoints", ha="right", va="bottom", fontsize=8.5,
             color="0.4")
    fig.tight_layout(rect=(0, .02, 1, 1), h_pad=2.5)
    path = fig_path("td_tt_compression.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("saved", path)


def figure_training():
    """-> figure_bin/td_tt_compression_training.png (John 11-13: "I mean seeing this splits by FR through ... training"):
    the raw departure from the compressed code at step 30 through training, one line per FR decile within the
    realization (each realization's 50 environments ranked by FR, 5 per decile; mean over the decile and the 100
    realizations), grey (lowest FR) to red (highest), the paper's FR colours. Rows Trained RNN, ESN; columns the
    context-dependent part ||i||^2 and the trajectory residual ||eps||^2. The 8 checkpoints of XS (a denser fill was
    started 11-13 and stopped, John: "I don't think we need a denser version"; its 24 finished checkpoints are in OUT)"""
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from main.utils import fig_path, apply_fig_style
    import fig_td_summary as fts
    import plot_fig1c_mesh as pm
    import supplementary_plotter2 as sp
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    fr = fr_cells()
    cols = pm.FR_CMAP(np.linspace(.2, 1, 10))
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8))
    n_ck = 0
    for row, (run, name) in enumerate((("trained", "Trained"), ("echo", "Echo"))):
        fs = [os.path.join(OUT, f"{run}_{raw:05d}.npz") for raw in raws_for(run, XS)]          # the 8 of XS
        n_ck = len(fs)
        Zs = [np.load(f) for f in fs]
        x = np.array([float(z["x"]) for z in Zs])
        for j, key in enumerate(("i2", "e2")):
            D = np.array([by_decile(z[key], fr) for z in Zs])              # (checkpoints, 10)
            for k in range(10):
                axs[row, j].plot(x, D[:, k], c=cols[k], lw=2.2 if k in (0, 9) else 1.5, marker="o",
                                 ms=3.5 if k in (0, 9) else 2.5, zorder=3 + k)
            a = axs[row, j]
            fts.xaxis(a, label=False)
            a.set_xlim(200, 4.6e4)
            a.set_xlabel("Training episode")
            a.axvline(880, color="0.6", lw=.9, ls=(0, (1, 1.5)), zorder=1)
            a.spines[["top", "right"]].set_visible(False)
            a.grid(True, ls="--", alpha=.3)
            a.set_ylabel(f"{name}\nmean squared norm" if j == 0 else "mean squared norm", fontsize=11 if j == 0 else 10)
        axs[row, 0].yaxis.label.set_color(sp.AGENT_COLORS[name])
    titles = (("Context-dependent part", r"$\Vert i(e,c)\Vert^2$ at step 30, "
               r"$i = \mathbb{E}[u|\mathbf{r},Z] - \mathbb{E}[u|\mathbf{r}] - \mathbb{E}[u|Z] + \mathbb{E}[u]$"),
              ("Trajectory residual", r"$\Vert \epsilon \Vert^2$ at step 30, $\epsilon = u - \mathbb{E}[u|\mathbf{r},Z]$"))
    for j, (t, f) in enumerate(titles):
        axs[0, j].set_title(t, fontsize=12.5, pad=19)
        axs[0, j].annotate(f, xy=(.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                           ha="center", va="baseline", fontsize=9.5, color="0.15")
    fig.tight_layout(rect=(0, .03, .9, 1), h_pad=2.5)
    cax = fig.add_axes([.915, .3, .015, .4])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=ListedColormap(cols), norm=BoundaryNorm(np.arange(.5, 11), 10)),
                      cax=cax, ticks=np.arange(1, 11))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, length=2)
    cb.set_label("FR decile within the realization (1 = lowest)", fontsize=10)
    fig.text(.995, .005, f"10,000 probes = 50 environments x 100 realizations x 2 trajectories; deciles: each "
             f"realization's environments by its FR (5 each); {n_ck} checkpoints", ha="right", va="bottom",
             fontsize=8.5, color="0.4")
    path = fig_path("td_tt_compression_training.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    if sys.argv[1:] == ["figure"]:
        figure()
        sys.exit()
    if sys.argv[1:] == ["training"]:
        figure_training()
        sys.exit()
    if sys.argv[1:] == ["steps"]:
        import multiprocessing as mp
        import td_tt_intervene as ti
        jobs = []
        for nm in ("trained", "echo"):
            raws = np.array(ti.checkpoints(nm))
            for x in XS:
                raw = int(raws[np.argmin(np.abs(gap.train_x(raws) - x))])
                if not os.path.exists(os.path.join(OUT, f"steps_{nm}_{raw:05d}.npz")):
                    jobs.append((nm, raw))
        with mp.get_context("spawn").Pool(7) as pool:
            for path in pool.imap_unordered(harvest_steps, jobs):
                print("saved", path, flush=True)
        sys.exit()
    if sys.argv[1:] != ["stats"]:
        import multiprocessing as mp
        import td_tt_intervene as ti
        os.makedirs(OUT, exist_ok=True)
        jobs = []
        for nm in ("trained", "echo"):
            raws = np.array(ti.checkpoints(nm))
            for x in XS:
                raw = int(raws[np.argmin(np.abs(gap.train_x(raws) - x))])
                if not os.path.exists(os.path.join(OUT, f"{nm}_{raw:05d}.npz")):
                    jobs.append((nm, raw))
        with mp.get_context("spawn").Pool(5) as pool:
            for path in pool.imap_unordered(harvest, jobs):
                print("saved", path, flush=True)
    stats()
