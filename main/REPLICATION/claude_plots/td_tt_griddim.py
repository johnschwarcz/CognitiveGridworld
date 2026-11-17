"""Is the rise of effective dimension through training the learning of a context-invariant grid code (John 11-09: "is
the increase in effective dim through training explained by learning a context-invariant grid code? (meaning the
dimensionality is handling the Z-interference?)"). Hidden state u_t of the 10,000 probes (50 test environments x 100
cells x 2 streams, balanced and crossed) replayed at checkpoints of the Trained RNN and the ESN, split exactly into
  env      the environment means (the context Z's own code)
  cell     the cell means over all environments: the CONTEXT-INVARIANT grid code (r0, r1)
  envcell  environment x cell: the cell code's part that differs between environments
  res      within (environment, cell): the two streams' differences (trajectory-specific)
Covariances (H x H) add exactly: total = env + cell + envcell + res. Stored per checkpoint and step t (15, 30): each
factor's variance and participation ratio PR = (tr S)^2 / tr(S^2), the total's PR, the total's PR with one factor
removed (leave-one-out), and the share of each factor's variance inside the top-k principal subspace of env + envcell
(k = 5, 10, 20).
(k = 5, 10, 20); 11-09: the same with the context subspace holding 50 / 80 / 90% of its variance (inZq*, kZ*),
the shared grid's purity along its own axes holding 50 / 80 / 90% of its variance, cell / (cell + env + envcell) there
(pureG*, kG*), and the shared grid's share of all the variance (share_cell).
-> main/DATA/experimental/training_dynamics/through_training/griddim/<run>.npz
Run from the repository root (CPU, ~1 min per checkpoint):
    OMP_NUM_THREADS=8 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_griddim.py
11-13 (John: "are we able to have them all with full data?"): at T = 30 only, at all 114 saved checkpoints per run
(RAWS are 13 of them), the eigenvalue spectra of the total, each factor and the total without each factor, and run()'s
PR / variance keys from them (run_full; ~55 core-s per checkpoint, the replay). In shards, then joined:
    for i in 0 1 2 3 4 5 6; do OMP_NUM_THREADS=3 nohup python td_tt_griddim.py full $i 7 & done; python td_tt_griddim.py merge
-> main/DATA/experimental/training_dynamics/through_training/griddim_full/<run>.npz"""
import os
import sys
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_intervene as ti                                                                  # noqa: E402
import td_tt_chaos_ent as ce                                                                  # noqa: E402

OUT = os.path.join(gap.TT, "griddim")
OUT_FULL = os.path.join(gap.TT, "griddim_full")
RAWS = (0, 100, 250, 500, 800, 1100, 1500, 2000, 3000, 6000, 10000, 25000, 49995)
STEPS = (15, 30)
FACT = ("env", "cell", "envcell", "res")
KS = (5, 10, 20)
QS = (.5, .8, .9)


def pr(S):
    ev = np.clip(np.linalg.eigvalsh(S), 0, None)
    return ev.sum() ** 2 / (ev ** 2).sum()


def pr_ev(ev):
    """the participation ratio from an eigenvalue spectrum (as pr)"""
    ev = np.clip(ev, 0, None)
    return ev.sum() ** 2 / (ev ** 2).sum()


def split(u, env, cell):
    """u (n, H) of a balanced env x cell x 2 design -> factor covariances {name: (H, H)} (sums of squares / n)"""
    n, H = u.shape
    X = np.zeros((50, 100, 2, H))
    cnt = np.zeros((50, 100), int)
    for i in range(n):
        X[env[i], cell[i], cnt[env[i], cell[i]]] = u[i]
        cnt[env[i], cell[i]] += 1
    assert (cnt == 2).all()
    g = X.mean((0, 1, 2))
    me = X.mean((1, 2)) - g                                        # (50, H)
    mc = X.mean((0, 2)) - g                                        # (100, H)
    mec = X.mean(2) - g - me[:, None] - mc[None]                   # (50, 100, H)
    res = X - X.mean(2, keepdims=True)
    parts = {"env": np.repeat(me, 200, 0), "cell": np.tile(np.repeat(mc, 2, 0), (50, 1)),
             "envcell": np.repeat(mec.reshape(-1, H), 2, 0), "res": res.reshape(-1, H)}
    return {k: v.T @ v / n for k, v in parts.items()}


def run(name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.npz")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    P = ti.probes()
    env, r = P["env"], P["r"]
    cell = r[:, 0] * 10 + r[:, 1]
    rows = []
    for raw in RAWS:
        p, table = ce.setup(name, raw, torch.device("cpu"))
        U = []
        with torch.no_grad():
            for b0 in range(0, len(env), 2500):
                envt, code = ce.codes(np.arange(b0, b0 + 2500), torch.device("cpu"))
                _, u, _, _ = ce.replay(p, table, envt, code)
                U.append(u[:, [t for t in STEPS]].double().numpy())       # U index t = after t observations
        U = np.concatenate(U)
        row = {"x": float(gap.train_x(raw))}
        for j, t in enumerate(STEPS):
            S = split(U[:, j], env, cell)
            tot = sum(S.values())
            row[f"t{t}_pr_total"] = pr(tot)
            Z = S["env"] + S["envcell"]
            w, V = np.linalg.eigh(Z)
            V = V[:, ::-1]
            for f in FACT:
                row[f"t{t}_var_{f}"] = np.trace(S[f])
                row[f"t{t}_pr_{f}"] = pr(S[f])
                row[f"t{t}_loo_{f}"] = pr(tot - S[f])
                for k in KS:
                    Vk = V[:, :k]
                    row[f"t{t}_inZ{k}_{f}"] = np.trace(Vk.T @ S[f] @ Vk) / np.trace(S[f])
            # 11-09 (John: "top 10 PCs seems arbitrary"): subspaces set by a share of variance, not a count
            wz = np.clip(w[::-1], 0, None)
            for q in QS:
                k = int(np.searchsorted(np.cumsum(wz) / wz.sum(), q)) + 1
                Vk = V[:, :k]
                row[f"t{t}_kZ{int(q * 100)}"] = k
                row[f"t{t}_inZq{int(q * 100)}_cell"] = np.trace(Vk.T @ S["cell"] @ Vk) / np.trace(S["cell"])
            wc, Vc = np.linalg.eigh(S["cell"])
            wc, Vc = np.clip(wc[::-1], 0, None), Vc[:, ::-1]
            for q in QS:                                          # along the shared grid's own axes
                k = int(np.searchsorted(np.cumsum(wc) / wc.sum(), q)) + 1
                Vk = Vc[:, :k]
                g, z = np.trace(Vk.T @ S["cell"] @ Vk), np.trace(Vk.T @ Z @ Vk)
                row[f"t{t}_kG{int(q * 100)}"] = k
                row[f"t{t}_pureG{int(q * 100)}"] = g / (g + z)    # the grid's share of grid + context variance there
            row[f"t{t}_share_cell"] = np.trace(S["cell"]) / np.trace(tot)
        rows.append(row)
        print(name, raw, {k: round(v, 3) for k, v in row.items() if k.startswith("t30") and ("pr" in k or "loo" in k)},
              flush=True)
    np.savez(path, **{k: np.array([rw[k] for rw in rows]) for k in rows[0]})



def run_full(shard=0, nshards=1):
    """every nshards-th (run, checkpoint) pair from shard, of both runs' saved checkpoints (ti.checkpoints): u_30 of the
    10,000 probes (as run()), split; per row the spectra t30_ev_{total, <factor>, loo_<factor>} (H each) and t30_pr_total,
    t30_pr_<factor>, t30_loo_<factor>, t30_var_<factor>, t30_share_cell computed from them -> OUT_FULL/<run>_shard<i>of<n>"""
    os.makedirs(OUT_FULL, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 3)))
    P = ti.probes()
    env, r = P["env"], P["r"]
    cell = r[:, 0] * 10 + r[:, 1]
    pairs = [(nm, raw) for nm in ("trained", "echo") for raw in ti.checkpoints(nm)][shard::nshards]
    rows = {}
    for nm, raw in pairs:
        p, table = ce.setup(nm, raw, torch.device("cpu"))
        U = []
        with torch.no_grad():
            for b0 in range(0, len(env), 2500):
                envt, code = ce.codes(np.arange(b0, b0 + 2500), torch.device("cpu"))
                _, u, _, _ = ce.replay(p, table, envt, code)
                U.append(u[:, 30].double().numpy())                    # after 30 observations
        S = split(np.concatenate(U), env, cell)
        tot = sum(S.values())
        ev = {"total": np.linalg.eigvalsh(tot)}
        for f in FACT:
            ev[f], ev["loo_" + f] = np.linalg.eigvalsh(S[f]), np.linalg.eigvalsh(tot - S[f])
        row = {"raw": raw, "x": float(gap.train_x(raw)), "t30_pr_total": pr_ev(ev["total"]),
               "t30_share_cell": np.trace(S["cell"]) / np.trace(tot)}
        for f in FACT:
            row.update({f"t30_pr_{f}": pr_ev(ev[f]), f"t30_loo_{f}": pr_ev(ev["loo_" + f]), f"t30_var_{f}": np.trace(S[f])})
        row.update({f"t30_ev_{k}": v for k, v in ev.items()})
        rows.setdefault(nm, []).append(row)
        print(nm, raw, round(row["t30_pr_total"], 2), round(row["t30_loo_cell"], 2), flush=True)
    for nm, rw in rows.items():
        np.savez(os.path.join(OUT_FULL, f"{nm}_shard{shard}of{nshards}.npz"), **{k: np.array([x[k] for x in rw]) for k in rw[0]})


def merge_full():
    """the shards of run_full -> OUT_FULL/<run>.npz, in checkpoint order (all 114 checked), shards removed"""
    import glob
    for nm in ("trained", "echo"):
        fs = sorted(glob.glob(os.path.join(OUT_FULL, f"{nm}_shard*of*.npz")))
        parts = [np.load(f) for f in fs]
        D = {k: np.concatenate([q[k] for q in parts]) for k in parts[0].files}
        o = np.argsort(D["raw"])
        D = {k: v[o] for k, v in D.items()}
        assert np.array_equal(D["raw"], ti.checkpoints(nm)), nm
        np.savez(os.path.join(OUT_FULL, f"{nm}.npz"), **D)
        for f in fs:
            os.remove(f)
        print(nm, len(D["raw"]), "checkpoints ->", os.path.join(OUT_FULL, f"{nm}.npz"))


def figure():
    """-> figure_bin/td_tt_griddim.png: a PR of each factor (Trained RNN solid, ESN dashed); b total PR and the total
    with one factor removed; c the shared grid's separation from the context: share of its variance inside the top-10
    principal subspace of env + envcell, and the shared fraction cell / (cell + envcell). Step 30."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    TS, FL, FT = 11, 10, 9
    Z = {r: np.load(os.path.join(OUT, f"{r}.npz")) for r in ("trained", "echo")}
    FC = {"env": "#7f7f7f", "cell": "#d62728", "envcell": "#9467bd", "res": "#2ca02c"}
    NM = {"env": "context (env)", "cell": "shared grid (cell)", "envcell": "context-specific grid", "res": "trajectory"}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    fig.subplots_adjust(left=.07, right=.97, bottom=.36, top=.78, wspace=.45)
    for run, ls in (("trained", "-"), ("echo", (0, (3, 1.5)))):
        z = Z[run]
        x = z["x"]
        for f in FACT:
            axs[0].plot(x, z[f"t30_pr_{f}"], color=FC[f], ls=ls, lw=1.5)
            axs[1].plot(x, z[f"t30_loo_{f}"], color=FC[f], ls=ls, lw=1.5)
        for a in axs[:2]:
            a.plot(x, z["t30_pr_total"], color="k", ls=ls, lw=1.8)
        axs[2].plot(x, z["t30_inZ10_cell"], color="#d62728", ls=ls, lw=1.6)
        axs[2].plot(x, z["t30_var_cell"] / (z["t30_var_cell"] + z["t30_var_envcell"]), color="#ff9896", ls=ls, lw=1.6)
    for a in axs:
        a.set_xscale("log")
        a.set_xticks([10, 1e2, 1e3, 1e4])
        a.set_xticklabels(["0", "100", "1k", "10k"])
        a.minorticks_off()
        a.axvline(880, color="0.4", lw=.8, ls=":")
        a.set_xlabel("training episodes", fontsize=FL)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=FT)
    for a in axs[:2]:
        a.set_yscale("log")
        a.set_ylim(1, 200)
        a.set_ylabel("participation ratio", fontsize=FL)
    axs[2].set_ylim(0, 1)
    axs[2].set_ylabel("share", fontsize=FL)
    titles = (("a   Dimension of each part", "the four parts of $u_{30}$ (they add up)"),
              ("b   Total, one part removed", "black: all; colour: that part left out"),
              ("c   Shared grid vs context", "dark: in context PCs; light: shared"))
    for a, (t, f) in zip(axs, titles):
        a.annotate(t, xy=(0, 1), xycoords="axes fraction", xytext=(0, 19), textcoords="offset points", fontsize=TS)
        a.annotate(f, xy=(0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points", fontsize=FT - 1)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=FC[f], lw=1.5) for f in FACT] + [Line2D([], [], color="k", lw=1.8),
          Line2D([], [], color="0.3", lw=1.5), Line2D([], [], color="0.3", lw=1.5, ls=(0, (3, 1.5)))]
    fig.legend(hs, [NM[f] for f in FACT] + ["total", "Trained RNN", "ESN"], loc="lower center", ncol=7, frameon=False,
               fontsize=FT - 1, handlelength=1.6, columnspacing=.9)
    out = fig_path("td_tt_griddim.png")
    fig.savefig(out, dpi=250)
    print("saved", out)


if __name__ == "__main__":
    if sys.argv[1:] == ["figure"]:
        figure()
    elif sys.argv[1:2] == ["full"]:
        run_full(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1:] == ["merge"]:
        merge_full()
    else:
        for nm in sys.argv[1:] or ("trained", "echo"):
            run(nm)
