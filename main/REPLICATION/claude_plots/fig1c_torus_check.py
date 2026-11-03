"""Is the evidence (and the representation) a torus? (27 Oct 2026) -> figdata/round6_hidden/torus_check.npz and printed
numbers; plot_fig1c_torus.py draws them

Why it could be: the generator (main/env/bayes_likelihood.py, env_preprocessing.py) gives each value r of a variable
the feature V_o(r) = a smoothed sine on a circle of V_num = 21 positions, value r at position r (a circular
convolution of one full sine period, so exactly A_o sin(w r + phi_o), w = 2 pi / 21), and each observation's logit is
the product V1_o(r0) V2_o(r1) = A [cos(w (r0 - r1) + a_o) - cos(w (r0 + r1) + b_o)] / 2: a function of u = r0 + r1
and v = r0 - r1 modulo 21 only. The 10 x 10 grid therefore sits on the torus Z_21 x Z_21 in these diagonal
coordinates, covering 19 of the 21 residues of each: opposite corners (0, 0) / (9, 9) and (0, 9) / (9, 0) are 3
residues apart across the wrap (torus distance 2.1 cells) though 12.7 apart on the grid.
Checks (alias scan, 39 environments, e = 2 left out):
    1  each observation's logit table is rank 1 and its two factors are sinusoids of period 21 (R^2 of the fit)
    2  evidence geometry: Jeffreys divergence between two cells' observation distributions (5 Bernoullis, one step);
       the 10 wrap pairs (torus distance <= 3, grid distance >= 8: the opposite-corner region) against non-wrapping
       pairs matched to their torus distance (near), the 20 axis-reflection pairs (0, c)-(9, c), (r, 0)-(r, 9) (grid
       and torus 9: far on a torus, close under a per-variable fold r <-> 9 - r) and the other farthest non-wrapping
       pairs (grid 8-9.1; far: a plane puts the wrap pairs, 12-12.7 apart, at least this far)
    3  the Optimal's split pairs (plot_fig1c_split_movie, 743 pairs): torus and grid distance of their true cells,
       against all cell pairs at the same grid distance (+-0.25)
    4  the Optimal's exact joint posterior of each split pair's streams, aligned on the split step t_s: the mass near
       the stream's own true cell and near its partner's (cells within 1.5 on the grid)
    5  representation geometry per agent and step: the cells' mean beliefs (20 numbers, the scan's) and the networks'
       hidden u (hidden_scan_cellmeans), distance of wrap pairs against the near and far controls
Torus distance: sqrt((du^2 + dv^2) / 2), du, dv the circular differences of u, v modulo 21 (= the grid distance for
pairs that do not wrap).
Run from the repository root (about 3 min, CPU):
    CUDA_VISIBLE_DEVICES="" /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig1c_torus_check.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
OUT = os.path.join(SCAN, "figdata", "round6_hidden", "torus_check.npz")
ENVS = [e for e in range(40) if e != 2]
E, K, M, T, R, P21 = 39, 100, 50, 30, 10, 21
AG = ("Joint", "Trained", "Naive", "Echo")


def geometry():
    """grid and torus distances between the 100 cells k = 10 a + b (either variable order: both are symmetric in it)"""
    a, b = np.divmod(np.arange(K), R)
    u, v = a + b, a - b
    circ = lambda x: np.minimum(np.abs(x) % P21, P21 - np.abs(x) % P21)
    du, dv = circ(u[:, None] - u[None]), circ(v[:, None] - v[None])
    d_grid = np.hypot(a[:, None] - a[None], b[:, None] - b[None])
    d_torus = np.sqrt((du ** 2 + dv ** 2) / 2.)
    return d_grid, d_torus


CATS = ("wrap", "near", "axis", "far")                                            # the contrast's order


def categories(d_grid, d_torus, tol=.5):
    """wrap pairs (torus <= 3, grid >= 8), per wrap pair the non-wrapping pairs matched to its torus distance (near),
    the axis-reflection pairs (0, c)-(9, c) and (r, 0)-(r, 9) (axis), and the other farthest non-wrapping pairs (grid
    >= 8, i.e. 8-9.1: far); indices into the upper triangle"""
    iu = np.triu_indices(K, 1)
    dg, dt = d_grid[iu], d_torus[iu]
    plain = np.isclose(dg, dt)
    wrap = np.flatnonzero((dt <= 3) & (dg >= 8))
    near = [np.flatnonzero(plain & (np.abs(dg - dt[w]) < tol)) for w in wrap]
    a0, b0 = np.divmod(iu[0], R)
    a1, b1 = np.divmod(iu[1], R)
    axis = np.flatnonzero(((a0 + a1 == 9) & (b0 == b1) & np.isin(a0, (0, 9))) |
                          ((b0 + b1 == 9) & (a0 == a1) & np.isin(b0, (0, 9))))
    far = np.setdiff1d(np.flatnonzero(plain & (dg >= 8)), axis)
    return iu, wrap, near, (axis, far)


def contrast(D, iu, wrap, near, rest):
    """D (K, K) distances -> mean distance of the wrap pairs, of their matched near pairs, of the axis-reflection
    pairs and of the far pairs (CATS)"""
    x = D[iu]
    axis, far = rest
    return np.array([x[wrap].mean(), np.mean([x[n].mean() for n in near]), x[axis].mean(), x[far].mean()])


def main():
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    Ls, goal, cells, cell_of, obs = meta["L"], meta["goal"].astype(int), meta["cells"], meta["cell_of"], meta["obs"]
    d_grid, d_torus = geometry()
    iu, wrap, near, far = categories(d_grid, d_torus)
    print("wrap pairs (torus <= 3, grid >= 8): %d of 4950; e.g. %s" % (len(wrap), [
        (tuple(divmod(iu[0][w], R)), tuple(divmod(iu[1][w], R)), round(d_grid[iu][w], 1), round(d_torus[iu][w], 1))
        for w in wrap[:6]]))
    out = dict(d_grid=d_grid, d_torus=d_torus, wrap=wrap)
    # 1: rank 1 and period-21 sinusoids
    w = 2 * np.pi / P21
    basis = np.stack([np.cos(w * np.arange(R)), np.sin(w * np.arange(R))], 1)          # (10, 2)
    r1, r2_fit = [], []
    for e in ENVS:
        for o in range(Ls.shape[1]):
            lg = np.log(Ls[e, o]) - np.log1p(-Ls[e, o])                                # (r0, r1) logits
            U, s, Vt = np.linalg.svd(lg)
            r1.append(s[0] ** 2 / (s ** 2).sum())
            for f in (U[:, 0] * s[0], Vt[0]):
                coef, res, *_ = np.linalg.lstsq(basis, f, rcond=None)
                r2_fit.append(1 - ((f - basis @ coef) ** 2).sum() / ((f - f.mean()) ** 2).sum())
    print("1  logit tables: rank-1 share of variance min %.6f; sinusoid (period 21) fit R^2 of the factors min %.6f" % (
        np.min(r1), np.min(r2_fit)))
    out["rank1_min"], out["sine_r2_min"] = np.min(r1), np.min(r2_fit)
    # 2: evidence geometry (raw slot order: cell k = 10 r0 + r1 of the table)
    ev = []
    for e in ENVS:
        p = Ls[e].reshape(Ls.shape[1], K).T                                             # (K, 5)
        lg = np.log(p) - np.log1p(-p)
        J = ((p[:, None] - p[None]) * (lg[:, None] - lg[None])).sum(-1)                 # Jeffreys, one step
        ev.append(contrast(J, iu, wrap, near, far))
    ev = np.array(ev)
    out["evidence"] = ev
    print("2  evidence (Jeffreys per step, mean over envs): wrap %.3f | near (same torus distance) %.3f | axis reflection "
          "%.3f | far %.3f; wrap < far in %d/39 envs, wrap < 2 x near in %d/39" % (
              *ev.mean(0), (ev[:, 0] < ev[:, 3]).sum(), (ev[:, 0] < 2 * ev[:, 1]).sum()))
    out["evidence_J"] = np.stack([(lambda p: (lambda lg: ((p[:, None] - p[None]) * (lg[:, None] - lg[None])).sum(-1))(
        np.log(p) - np.log1p(-p)))(Ls[e].reshape(Ls.shape[1], K).T) for e in ENVS])    # (E, K, K) raw slot cells
    # 3: split pairs
    Zs = np.load(os.path.join(SCAN, "figdata", "round4b", "split_pairs.npz"))
    ki, kj = cell_of[Zs["i"]], cell_of[Zs["j"]]
    sp_g, sp_t = d_grid[ki, kj], d_torus[ki, kj]
    allg, allt = d_grid[iu], d_torus[iu]
    null_t = np.array([allt[np.abs(allg - g_) <= .25].mean() for g_ in sp_g])          # same grid distance
    out.update(split_grid=sp_g, split_torus=sp_t, split_null_torus=null_t, all_grid=allg, all_torus=allt)
    print("3  split pairs (%d): grid distance median %.1f; torus distance median %.1f vs %.1f for cell pairs at the same "
          "grid distance; below that null in %.0f%%; opposite-corner region (wrap pairs): %.0f%% (%.1f%% of pairs >= 4)" % (
              len(sp_g), np.median(sp_g), np.median(sp_t), np.median(null_t), 100 * (sp_t < null_t).mean(),
              100 * np.isin(np.minimum(ki, kj) * K + np.maximum(ki, kj), iu[0][wrap] * K + iu[1][wrap]).mean(),
              100 * len(wrap) / (allg >= 4).sum()))
    # 4: the Optimal's joint posterior of the split pairs' streams around the split
    TAU = np.arange(-4, 5)
    own, partner = np.full((len(ki), 2, len(TAU)), np.nan), np.full((len(ki), 2, len(TAU)), np.nan)
    nb = d_grid <= 1.5                                                                  # (K, K) grid neighbourhoods
    for e_ in np.unique(Zs["env"]):
        L = Ls[e_].astype(np.float64)
        rows = np.flatnonzero(Zs["env"] == e_)
        for s_, col in ((0, "i"), (1, "j")):
            st = Zs[col][rows]
            o = obs[e_, st].astype(np.float64)                                          # (n, T, 5)
            lJ = np.einsum("nti,ijk->ntjk", o, np.log(L)) + np.einsum("nti,ijk->ntjk", 1 - o, np.log1p(-L))
            cum = np.cumsum(lJ, 1).reshape(len(st), T, K)                               # raw slot cells
            post = np.exp(cum - cum.max(-1, keepdims=True))
            post /= post.sum(-1, keepdims=True)
            g = goal[e_]
            raw = lambda k: (cells[k, 0] * R + cells[k, 1]) if g == 0 else (cells[k, 1] * R + cells[k, 0])
            k_own = raw(cell_of[st])
            k_par = raw(cell_of[Zs["j" if col == "i" else "i"][rows]])
            for n_, r_ in enumerate(rows):
                for t_, tau in enumerate(TAU):
                    t = Zs["ts"][r_] - 1 + tau                                            # tau = 0: step t_s
                    if 0 <= t < T:
                        own[r_, s_, t_] = post[n_, t, nb[k_own[n_]]].sum()
                        partner[r_, s_, t_] = post[n_, t, nb[k_par[n_]]].sum()
    out.update(tau=TAU, post_own=own, post_partner=partner)
    print("4  Optimal posterior, split pairs, both streams (median): steps from t_s %s" % TAU)
    print("     near own cell:     %s" % np.round(np.nanmedian(own, (0, 1)), 3))
    print("     near partner cell: %s" % np.round(np.nanmedian(partner, (0, 1)), 3))
    print("     sum:               %s" % np.round(np.nanmedian(own + partner, (0, 1)), 3))
    # 5: representation geometry over steps
    rep = {}
    for a in AG:
        B = np.load(os.path.join(SCAN, f"scan_{a}.npz"))["beliefs"][ENVS].astype(np.float32)
        Cm = B.reshape(E, K, M, T, 20).mean(2).astype(np.float64)                     # (E, K, T, 20) goal order
        res = np.zeros((E, T, len(CATS)))
        for ie in range(E):
            for t in range(T):
                X = Cm[ie, :, t]
                D = np.sqrt(np.maximum(((X[:, None] - X[None]) ** 2).sum(-1), 0))
                res[ie, t] = contrast(D, iu, wrap, near, far)
        rep["belief_" + a] = res
        m = res.mean(0)
        print("5  %-8s beliefs, wrap / near / axis / far, t=1, 5, 10, 30: %s" % (
            a, "  ".join("%.2f/%.2f/%.2f/%.2f" % tuple(m[t]) for t in (0, 4, 9, 29))))
        del B
    for net in ("trained", "echo"):
        f = np.load(os.path.join(SCAN, f"hidden_scan_cellmeans_{net}.npz"))
        res = np.zeros((E, T, len(CATS)))
        U = f["u"]
        for ie in range(E):
            X = U[ie].astype(np.float64)                                                # (K, T, 1000) goal order
            for t in range(T):
                Xt = X[:, t]
                sq = (Xt ** 2).sum(-1)
                D = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * Xt @ Xt.T, 0))
                res[ie, t] = contrast(D, iu, wrap, near, far)
        rep["hidden_u_" + net] = res
        m = res.mean(0)
        print("5  %-8s hidden u, wrap / near / axis / far, t=1, 5, 10, 30: %s" % (
            net, "  ".join("%.2f/%.2f/%.2f/%.2f" % tuple(m[t]) for t in (0, 4, 9, 29))))
    out.update(rep)
    np.savez(OUT, **out)
    print("saved", OUT)


if __name__ == "__main__":
    main()
