"""The linear proxy r of the Var(z) theory (11-14; no network, nothing fitted): what an idealized LINEAR learner knows about the
novel environments' z in a world of |S| variables with d_E-dimensional unit embeddings and Var z = z0^2 (z0^2 = 1/d_E in the
original world, 1/30 in the scaled runs of train_bayes_generator.py tasks 123-252).
  1. Every ordered pair's 100 likelihood cells (the exact likelihood: sigmoid of a product of two sinusoids of z01 and z10,
     the 10 realization phases) are read out linearly into estimates of (z01, z10) by the matched filter W = (B'B)^-1 B',
     B = the cells' slopes on (z01, z10): Y = Z + fold residual. rho_mf = Var(residual) / Var(z), the fold ratio after
     pooling; rho_avg = the same ratio with every cell on its own. Closed forms for z ~ N(0, z0^2) and a linearized
     sigmoid, x = 8 pi^2 z0^2: rho_mf = sinh(x) / x - 1, rho_avg = (e^x - 1) / x - 1.
  2. The M x M training block of Y (M = |S| - |S| // 10, a held-out variable's training partners) is cut to rank d_E: what a
     rank-d_E bilinear model can fit.
  3. Held-out variables get their factors by least squares against their M training partners.
  4. r = the correlation of predicted and true z over the held-out x held-out block: the novel environments.
r holds the fold effect (rho) and the spanning effect (M against d_E) at once. For M >> d_E it is 1 - d_E rho_mf / M, the
squared overlap of a rank-d_E spike in noise (BBP); near M = d_E it collapses (least squares amplifies errors by about
d_E / (M - d_E)); below it nothing is determined.

The bootstrap rule (theory note of 11-14): a cell transfers (n >= .2) when r >= r*, with .37 < r* <= .45, the one measured
number. It was read off the 49 seed-0 cells it sorts (48 right; the miss is unscaled d_E 20 at |S| 1000, where r = .87), so
it is a description until new cells test it: test A (d_E 30 at |S| 80-95) pins r*, test B (z0^2 = 1/20, the last two rows
here) moves the threshold. zscale_progress.py holds this script's table (PROXY_R) for its --law proxy.

    python zscale_proxy.py [--out proxy.json] [--pairs 12000]       # about 2 minutes on 4 cores; each r has s.e. <= .01
"""
import json, time, argparse
import numpy as np

rng = np.random.default_rng(777)
REAL, VN = 10, 21                                              # realizations per variable; soft_roll's phase grid
AMP = 2 * np.exp(-np.pi ** 2 / 441)                            # soft_roll's amplitude (bayes_likelihood.soft_roll: 1.95574)
PHASE = 2 * np.pi * (np.arange(REAL) - 1) / VN
GRID = (50, 75, 100, 125, 250, 500, 1000)
R_STAR = (.37, .45)                                            # no seed-0 cell at or below .37 transfers; all from .45 up do
ROWS = (  # d_E, 1 / z0^2, the |S| added to the grid around the row's crossing
    (10, 30, (20, 25, 30, 35, 40, 45)), (20, 30, (55, 60, 65, 70)), (30, 30, (80, 85, 90, 95)),
    (90, 30, (275, 300, 325, 350, 400)), (20, 20, (150, 175, 200, 225)), (90, 90, (105, 110, 115, 120)), (10, 10, ()),
    (10, 20, (85, 110, 150, 175, 200)), (30, 20, (300, 350, 400)))        # the last two: test B's worlds
LAST = ((10, 30), (20, 30), (30, 30), (90, 30), (90, 90))      # rows whose |S| 1000 cell is drawn after all the others
#                                                                (the order only fixes the random stream)


def rho_mf(v):
    """closed-form fold ratio after matched-filter pooling of the cells, z ~ N(0, v)"""
    x = 8 * np.pi ** 2 * v
    return np.sinh(x) / x - 1


def rho_avg(v):
    """closed-form fold ratio with every cell on its own (zscale_progress.fold_ratio)"""
    x = 8 * np.pi ** 2 * v
    return np.expm1(x) / x - 1


def V(z):
    return -AMP * np.sin(2 * np.pi * z[..., None] - PHASE)


def sphere(d, n):
    """n draws of <k, q> for independent unit vectors in d dimensions"""
    g = rng.normal(size=n)
    return g / np.sqrt(g ** 2 + rng.chisquare(d - 1, size=n))


def pooling(d, scale, n=300_000):
    """W (2, 100): the matched-filter read-out of (z01, z10) from a pair's 100 cells; the mean cells; rho_mf, rho_avg"""
    z01, z10 = sphere(d, n) * scale, sphere(d, n) * scale
    P = (1 / (1 + np.exp(-V(z01)[:, :, None] * V(z10)[:, None, :]))).reshape(n, -1)
    Pbar = P.mean(0)
    Pc, X = P - Pbar, np.stack([z01 - z01.mean(), z10 - z10.mean()], 1)
    B = np.linalg.lstsq(X, Pc, rcond=None)[0].T                # (100, 2): each cell's slopes
    N = Pc - X @ B.T                                           # each cell's fold residual
    W = np.linalg.solve(B.T @ B, B.T)
    v = X.var(0).mean()
    return W, Pbar, float((N @ W.T).var(0).mean() / v), float(N.var(0).sum() / ((B ** 2).sum() * v))


def pooled_matrix(Z, W, Pbar, chunk=100_000):
    """Y[i, j]: the read-out estimate of Z[i, j] from the pair's cells (a pair is presented sorted: slot 1 = lower index)"""
    S = Z.shape[0]
    Y = np.zeros_like(Z)
    W1, W2, c1, c2 = W[0].reshape(REAL, REAL), W[1].reshape(REAL, REAL), W[0] @ Pbar, W[1] @ Pbar
    I, J = np.triu_indices(S, 1)
    for lo in range(0, len(I), chunk):
        i, j = I[lo:lo + chunk], J[lo:lo + chunk]
        P = 1 / (1 + np.exp(-V(Z[i, j])[:, :, None] * V(Z[j, i])[:, None, :]))
        Y[i, j] = np.einsum('nab,ab->n', P, W1) - c1
        Y[j, i] = np.einsum('nab,ab->n', P, W2) - c2
    return Y


def one_world(d, S, scale, W, Pbar):
    """predicted and true z over the held-out x held-out block of one random world"""
    K = rng.normal(size=(S, d)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    Q = rng.normal(size=(S, d)); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    Z = scale * K @ Q.T
    Y = pooled_matrix(Z, W, Pbar)
    T = S // 10
    h, t = np.arange(T), np.arange(T, S)
    U, s, Vt = np.linalg.svd(Y[np.ix_(t, t)], full_matrices=False)
    k = min(d, S - T)
    Zhat = (Y[np.ix_(h, t)] @ Vt[:k].T / s[:k]) @ (U[:, :k].T @ Y[np.ix_(t, h)])
    off = ~np.eye(T, dtype=bool)
    return Zhat[off], Z[np.ix_(h, h)][off]


def cell(d, S, scale, W, Pbar, pairs):
    """r and its standard error from at least `pairs` held-out pairs (two worlds or more)"""
    T = S // 10
    a, b = [], []
    for _ in range(max(2, int(np.ceil(pairs / max(T * (T - 1), 1))))):
        zh, z = one_world(d, S, scale, W, Pbar)
        a.append(zh); b.append(z)
    a, b = np.concatenate(a), np.concatenate(b)
    r = float(np.corrcoef(a, b)[0, 1])
    return r, float((1 - r * r) / np.sqrt(len(a)))


def crossing(cells, level):
    """|S| at which r first reaches `level`, by linear interpolation between the tabulated |S| (inf: not on the table)"""
    S = sorted(cells)
    for a, b in zip(S, S[1:]):
        if cells[a] < level <= cells[b]:
            return a + (b - a) * (level - cells[a]) / (cells[b] - cells[a])
    return S[0] if cells[S[0]] >= level else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--pairs", type=int, default=12_000)
    a = ap.parse_args()
    t0, out, late = time.time(), {}, []
    for d, d0, extra in ROWS:
        scale = float(np.sqrt(d / d0))
        W, Pbar, mf, avg = pooling(d, scale)
        out[d, d0] = {"rho_mf": mf, "rho_avg": avg, "cells": {}}
        for S in sorted(set(GRID) | set(extra)):
            if S > 500 and (d, d0) in LAST:
                late.append((d, d0, S, scale, W, Pbar))
            else:
                out[d, d0]["cells"][S] = cell(d, S, scale, W, Pbar, a.pairs)
        print(f"d_E {d:2d} z0^2 1/{d0}: {time.time() - t0:.0f}s", flush=True)
    for d, d0, S, scale, W, Pbar in late:
        out[d, d0]["cells"][S] = cell(d, S, scale, W, Pbar, a.pairs)
    print(f"all cells: {time.time() - t0:.0f}s\n")
    for (d, d0), row in out.items():
        r = {S: row["cells"][S][0] for S in sorted(row["cells"])}
        lo, hi = (crossing(r, x) for x in R_STAR)
        print(f"d_E {d:2d}, z0^2 = 1/{d0}: rho_mf {row['rho_mf']:.3g} (closed form {rho_mf(1 / d0):.3g}), rho_avg "
              f"{row['rho_avg']:.3g} ({rho_avg(1 / d0):.3g}); r crosses {R_STAR[0]} at |S| {lo:.0f}, {R_STAR[1]} at {hi:.0f}")
        print("    " + "  ".join(f"{S}: {v:.3f}" for S, v in r.items()))
    print("\nPROXY_R = {" + "\n           ".join(
        f"({d}, {d0}): {{" + ", ".join(f"{S}: {round(row['cells'][S][0], 3) + 0:.3f}" for S in sorted(row["cells"])) + "},"
        for (d, d0), row in out.items()) + "}")
    if a.out:
        json.dump({f"{d},{d0}": row for (d, d0), row in out.items()}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
