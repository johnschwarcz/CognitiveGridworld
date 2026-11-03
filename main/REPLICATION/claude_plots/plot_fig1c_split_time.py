"""When do nearby trajectories split, and how coupled are the two variables' updates, step by step (25 Oct 2026)
-> main/REPLICATION/figure_bin/fig1c_split_time.png

(a, b) Splits, on the alias scan (main/DATA/experimental/fig1c_compression; 39 environments, e = 2 left out; 100 cells x
50 streams x 30 steps; all four agents on the same streams). Position = posterior means x = (E[r_g], E[r_c]).
At each step t, in each agent's own positions and environment: a stream is merged if its nearest stream of a FAR cell
(true cells at least D_CELL grid units apart: the two should end far apart) is within R_NEAR grid units; such a pair
(counted once) splits if it is at least D_SPLIT further apart two steps later, and splits correctly if, after the
split, each stream is nearer its own true cell than the partner's.
    (a) share of streams that are merged at t
    (b) P(split within 2 steps | merged at t) (solid) and P(correct split | merged at t) (dashed)
No condition on earlier steps (the split movie's pairs had to stay together from step 2, which selects early splits).
(c) Coupling per step, on the gradflow harvest's 19,607 kept episodes (fig1_movie beliefs; figdata/round4b/
update_corr.npz): at each step t, Spearman's rank correlation over trajectories between the two variables' update
sizes |u_g(t)| and |u_c(t)| (u_v(t) = E_t[r_v] - E_{t-1}[r_v]): do both beliefs move a lot in the same steps?
Null: |u_c(t)| taken from another trajectory of the same environment (keeps environment-wide size differences).
Bands: 95% bootstrap over environments. Data cached in figdata/round4b/split_time.npz (--rebuild: about 2 min).
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_split_time.py [--rebuild]"""
import os
import sys
import numpy as np


def _root():
    for start in (globals().get("__file__"), globals().get("__vsc_ipynb_file__"), os.getcwd()):
        d = os.path.abspath(start) if start and os.path.exists(start) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return d
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError("project root not found")


ROOT = _root()
sys.path[:0] = [p for p in (ROOT,) if p not in sys.path]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from main.utils import apply_fig_style, fig_path                                  # noqa: E402

FS = apply_fig_style()
plt.rcParams["mathtext.fontset"] = "cm"
SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
FIG = os.path.join(SCAN, "figdata", "round4b")
DATA = os.path.join(FIG, "split_time.npz")
AG = ("Joint", "Trained", "Naive", "Echo")
NAME = {"Joint": "Optimal", "Trained": "Trained RNN", "Naive": "Naive", "Echo": "ESN"}
COLOR = {"Joint": "k", "Trained": "#1f77b4", "Naive": "0.5", "Echo": "#ff7f0e"}
STY = {"Trained": dict(lw=2.6, alpha=.45), "Joint": dict(lw=1.2), "Naive": dict(lw=1.2), "Echo": dict(lw=1.2)}
R_NEAR, D_SPLIT, K, D_CELL = .5, 3., 2, 4.
T = 30


def _ranks(x):
    return np.argsort(np.argsort(x, 0), 0).astype(np.float64)


def build():
    from scipy.spatial import cKDTree
    m = np.load(os.path.join(SCAN, "scan_meta.npz"))
    cells, cell_of = m["cells"], m["cell_of"]
    c_pos = cells[cell_of].astype(np.float64)
    envs = [e for e in range(40) if e != 2]
    R = np.arange(10)
    steps = np.arange(1, T - K + 1)                                               # t = 1..28 (t + K <= 30)
    out = {"steps": steps, "envs": np.array(envs)}
    for a in AG:
        B = np.load(os.path.join(SCAN, f"scan_{a}.npz"))["beliefs"]
        b = B.astype(np.float32).reshape(*B.shape[:3], 2, 10)
        X = (b * R).sum(-1) / b.sum(-1)                                           # (40, 5000, T, 2)
        merged, pairs, split, correct = (np.zeros((len(envs), len(steps))) for _ in range(4))
        for k_, e in enumerate(envs):
            for j_, t in enumerate(steps):
                P = X[e, :, t - 1].astype(np.float64)
                dd, jj = cKDTree(P).query(P, k=60)
                other = np.linalg.norm(c_pos[jj] - c_pos[:, None], axis=-1) >= D_CELL   # neighbours of a far cell
                first = np.where(other.any(1), other.argmax(1), -1)
                ok = first >= 0
                dn = np.full(len(P), np.inf)
                dn[ok] = dd[ok, first[ok]]
                near = dn < R_NEAR
                merged[k_, j_] = near.mean()
                i = np.flatnonzero(near)
                jn = jj[i, first[i]]
                key = np.unique(np.sort(np.stack([i, jn], 1), 1), axis=0)          # each pair once
                if not len(key):
                    continue
                pi, pj = key[:, 0], key[:, 1]
                d0 = np.linalg.norm(X[e, pi, t - 1] - X[e, pj, t - 1], axis=-1)
                xi, xj = X[e, pi, t - 1 + K].astype(np.float64), X[e, pj, t - 1 + K].astype(np.float64)
                sp_ = np.linalg.norm(xi - xj, axis=-1) - d0 >= D_SPLIT
                own = ((np.linalg.norm(xi - c_pos[pi], axis=-1) < np.linalg.norm(xi - c_pos[pj], axis=-1))
                       & (np.linalg.norm(xj - c_pos[pj], axis=-1) < np.linalg.norm(xj - c_pos[pi], axis=-1)))
                pairs[k_, j_], split[k_, j_], correct[k_, j_] = len(key), sp_.sum(), (sp_ & own).sum()
        out.update({f"{a}_merged": merged, f"{a}_pairs": pairs, f"{a}_split": split, f"{a}_correct": correct})
        print(a, "splits done", flush=True)
    # coupling per step on the harvest
    Z = np.load(os.path.join(FIG, "update_corr.npz"))
    env = Z["env"]
    rng = np.random.default_rng(0)
    order = np.argsort(env, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(env))]
    perm = np.empty(len(env), int)                                                # another trajectory, same environment
    for e in range(env.max() + 1):
        idx = order[bounds[e]:bounds[e + 1]]
        perm[idx] = np.roll(rng.permutation(idx), 1) if len(idx) > 1 else idx
    for a in AG:
        u = np.abs(Z["u_" + a].astype(np.float64))                                # (n, T, 2)
        rg, rc, rc0 = _ranks(u[..., 0]), _ranks(u[..., 1]), _ranks(u[perm, :, 1])
        rg, rc, rc0 = (r - r.mean(0) for r in (rg, rc, rc0))
        out[f"{a}_cov"], out[f"{a}_cov0"] = rg * rc, rg * rc0                     # per-trajectory products (n, T)
        out[f"{a}_var"] = np.stack([(rg ** 2), (rc ** 2), (rc0 ** 2)], -1).mean(0)  # (T, 3)
    out["env"] = env
    np.savez_compressed(DATA, **out)
    print("saved", DATA)


def _boot(num, den, nb=1000, seed=0):
    """ratio of sums over environments (rows) with a 95% bootstrap over environments"""
    rng = np.random.default_rng(seed)
    est = num.sum(0) / np.maximum(den.sum(0), 1e-12)
    idx = rng.integers(0, len(num), (nb, len(num)))
    bs = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return est, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


def _spearman(Z, a, null=False, nb=500, seed=0):
    """per step: Spearman's rho from the stored rank products, with a bootstrap over environments"""
    cov = Z[f"{a}_cov0" if null else f"{a}_cov"]
    v = Z[f"{a}_var"]
    den = np.sqrt(v[:, 0] * v[:, 2 if null else 1])
    env = Z["env"]
    ne = env.max() + 1
    S = np.zeros((ne, cov.shape[1]))
    np.add.at(S, env, cov)
    n = np.bincount(env, minlength=ne)
    rng = np.random.default_rng(seed)
    est = S.sum(0) / n.sum() / den
    idx = rng.integers(0, ne, (nb, ne))
    bs = S[idx].sum(1) / n[idx].sum(1)[:, None] / den
    return est, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


def figure():
    Z = np.load(DATA)
    steps = Z["steps"]
    pw, ph, gx = 2.9, 2.0, .95
    W, H = .8 + 3 * pw + 2 * gx + .3, .95 + ph + .9
    fig = plt.figure(figsize=(W, H))

    def panel(c, name, formula, letter):
        ax = fig.add_axes([(.8 + c * (pw + gx)) / W, .9 / H, pw / W, ph / H])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(.5, T + .5), ax.set_xticks([1, 10, 20, 30])
        ax.set_xlabel("Step $t$", labelpad=1)
        ax.annotate(name, (.5, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points", ha="center",
                    va="baseline", fontsize=FS["label"] + .5)
        ax.annotate(formula, (.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                    ha="center", va="baseline", fontsize=FS["tick"] - .5, color="0.25")
        x0, y0, w, h = ax.get_position().bounds
        fig.text(x0 - .62 / W, y0 + h + 17 / 72 / H, r"$\mathbf{%s}$" % letter, fontsize=FS["label"] + 2, va="baseline")
        return ax

    ax = panel(0, "Merged streams", r"share with a stream of a far cell ($\geq %g$ apart) within %.1f" % (D_CELL, R_NEAR),
               "a")
    for a in AG:
        m = Z[f"{a}_merged"]
        e, lo, hi = _boot(m, np.ones_like(m))
        ax.fill_between(steps, 100 * lo, 100 * hi, color=COLOR[a], alpha=.15, lw=0)
        ax.plot(steps, 100 * e, c=COLOR[a], **STY[a])
    ax.set_ylim(0, None), ax.set_ylabel("% of streams")
    ax = panel(1, "Probability of splitting", r"P(apart by $\geq %g$ within %d steps $\mid$ merged at $t$); dashed: "
               r"correct splits" % (D_SPLIT, K), "b")
    for a in AG:
        for key, ls in (("split", "-"), ("correct", "--")):
            e, lo, hi = _boot(Z[f"{a}_{key}"], Z[f"{a}_pairs"])
            ok = Z[f"{a}_pairs"].sum(0) >= 50
            ax.fill_between(steps[ok], 100 * lo[ok], 100 * hi[ok], color=COLOR[a], alpha=.12, lw=0)
            ax.plot(steps[ok], 100 * e[ok], c=COLOR[a], ls=ls, **STY[a])
    ax.set_ylim(0, None), ax.set_ylabel("% of merged pairs")
    ax = panel(2, "Coupling of the update sizes", r"Spearman $\rho_t(|u_g(t)|, |u_c(t)|)$ over trajectories", "c")
    t = np.arange(1, T + 1)
    for a in AG:
        e, lo, hi = _spearman(Z, a)
        ax.fill_between(t, lo, hi, color=COLOR[a], alpha=.15, lw=0)
        ax.plot(t, e, c=COLOR[a], **STY[a])
        e0, _, _ = _spearman(Z, a, null=True)
        ax.plot(t, e0, c=COLOR[a], lw=.8, ls=(0, (3, 2)), alpha=.7)
    ax.axhline(0, c="0.6", lw=.6)
    ax.set_ylabel(r"Spearman $\rho$")
    hand = [Line2D([], [], c=COLOR[a], label=NAME[a], **STY[a]) for a in AG]
    hand += [Line2D([], [], c="0.3", lw=.8, ls=(0, (3, 2)), label="c: null (another trajectory of the environment)")]
    fig.legend(handles=hand, loc="upper center", bbox_to_anchor=(.5, 1 - .04 / H), ncol=5, frameon=False,
               fontsize=FS["tick"], handlelength=1.8, columnspacing=1.4)
    n_pairs = {a: int(Z[f"{a}_pairs"].sum()) for a in AG}
    fig.text(.8 / W, .42 / H, "a, b: alias scan, 39 environments x 5,000 streams, each agent in its own positions "
             "$(E[r_g], E[r_c])$; merged pairs over all steps: %s;\nb drawn where a step has at least 50 pairs. "
             "c: 19,607 trajectories from 1,225 environments. Bands: 95%% bootstrap over environments." % ", ".join(
                 f"{NAME[a]} {n_pairs[a]:,}" for a in AG), fontsize=FS["tick"] - 1.5, color="0.35", va="top",
             linespacing=1.4)
    path = fig_path("fig1c_split_time.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


if __name__ == "__main__":
    if "--rebuild" in sys.argv[1:] or not os.path.exists(DATA):
        build()
    print("saved", figure())
    Z = np.load(DATA)
    for a in AG:
        m = Z[f"{a}_merged"].mean(0)
        ps, pc = (Z[f"{a}_{k}"].sum(0) / np.maximum(Z[f"{a}_pairs"].sum(0), 1) for k in ("split", "correct"))
        rho = _spearman(Z, a)[0]
        print("%-8s merged %% t=1/5/10/20/28: %s | P(split) %s | P(correct) %s | coupling t=1/5/10/20/30 %s" % (
            a, np.round(100 * m[[0, 4, 9, 19, 27]], 1), np.round(100 * ps[[0, 4, 9, 19, 27]], 1),
            np.round(100 * pc[[0, 4, 9, 19, 27]], 1), np.round(rho[[0, 4, 9, 19, 29]], 2)))
