"""Where the level r* of the bootstrap rule comes from (11-14; no network, nothing fitted): a map that brackets every
observed threshold of the Var(z) test (theory note of 11-14, "Where the level comes from"). zscale_proxy.py is round 0.

One step. If the embeddings predict z with correlation m on the pairs they are trained on, a Generator that has learned
E[cells | z_hat] leaves a smaller fold residual: what is still unknown about z has variance z0^2 (1 - m^2), and for the
idealized likelihood (a product of two sinusoids, Gaussian z) the remaining problem is exactly the original one with z0^2
replaced by z0^2 (1 - m^2) (the matched filter of the residual, sin(da) cos(db), is what the gradient through a trained
Generator pools). Relative to the full signal the noise ratio becomes
    G(e) = e rho(x e),   e = 1 - m^2,   x = 8 pi^2 z0^2:   G_mf(e) = (sinh(x e) - x e) / x,   G_avg(e) = (e^{x e} - 1 - x e) / x.
The map. r <- r_lin(M, d_E, G(1 - r^2)) from e = 1, where r_lin(M, d_E, rho) is the linear proxy for a rank-d_E signal in
independent noise of ratio rho (zscale_proxy.py with Gaussian noise in place of the cells: the same r within .04). The
alignment stalls at a weak fixed point or runs away to 1. rho_mf (every pair's 100 cells pooled perfectly) gives the
lowest threshold, rho_avg (no pooling) a high one. Runs away from |S| (11-14, this script):
    row                 rho_mf   rho_avg   observed (last failure, first transfer)
    scaled d_E 20          35       70     50, 75
    d_E 30                 60      115     75, 100
    scaled d_E 90         200      400     250, 500
    unscaled d_E 20        95      250     125, 250
    unscaled d_E 90       110      150     100, 125
Every observed bracket overlaps its range; in proxy units the bounds are about .2-.3 and .5 where the fold ratio limits,
against the measured .37 < r* <= .45. Heuristics: what is left after a round is treated as fresh independent noise, and m
is taken to be the novel-pair correlation (the training-pair fit also holds absorbed noise). Thresholds carry Monte Carlo
noise of about one step of the |S| list.

--goal-only: the proxy for 'True answer, goal only' training (bayes_gen_plot.py; d_E 30, existing runs: chance up to |S|
100, transfer from 125). The Generator is given the goal's realization alone, so only marginal cells exist: 20 for a pair of
training variables (either can be the goal), 10 for a pair with a held-out partner (never the goal). r = .21 at |S| 100
and .27 at 125: the level .37-.45 belongs to training on both answers, it is not one number for every condition.

    python zscale_bootstrap.py [--curves cache.json] [--goal-only]      # curves: about 7 minutes on 4 cores
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zscale_proxy as zp

rng = np.random.default_rng(4242)
RHOS = np.concatenate([[0.], np.geomspace(1e-3, 3e3, 70)])     # noise ratios at which r_lin is tabulated
SIZES = sorted(set(zp.GRID) | {20, 25, 30, 35, 40, 45, 55, 60, 65, 70, 80, 85, 90, 95, 105, 110, 115, 120, 150, 175, 200, 225,
                               275, 300, 325, 350, 400})
ROWS = {(10, 30): 1.42, (20, 30): 1.53, (30, 30): 1.55, (90, 30): 1.60, (20, 20): 5.58, (90, 90): .135, (10, 10): 500.}
#       (d_E, 1 / z0^2): rho_mf of the exact likelihood (zscale_proxy.py); the closed forms are rescaled to it
OBSERVED = {(10, 30): (None, 50), (20, 30): (50, 75), (30, 30): (75, 100), (90, 30): (250, 500), (20, 20): (125, 250),
            (90, 90): (100, 125), (10, 10): (1000, None)}      # seed 0: last failing, first transferring |S| (n >= .2)


def curves(d, S, draws):
    """the training-pair fit's and the novel pairs' correlation with z, on RHOS (signal of unit variance, Gaussian noise)"""
    T = S // 10
    h, t = np.arange(T), np.arange(T, S)
    off = ~np.eye(T, dtype=bool)
    num, den_a, den_b = np.zeros((2, len(RHOS))), np.zeros((2, len(RHOS))), np.zeros(2)
    for _ in range(draws):
        K = rng.normal(size=(S, d)); K /= np.linalg.norm(K, axis=1, keepdims=True)
        Q = rng.normal(size=(S, d)); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        Z = np.sqrt(d) * K @ Q.T
        N = rng.normal(size=(S, S))
        Ztt, Zhh = Z[np.ix_(t, t)], Z[np.ix_(h, h)][off]
        den_b += (Ztt ** 2).sum(), (Zhh ** 2).sum()
        for i, rho in enumerate(RHOS):
            Y = Z + np.sqrt(rho) * N
            U, s, Vt = np.linalg.svd(Y[np.ix_(t, t)], full_matrices=False)
            k = min(d, S - T)
            fit = (U[:, :k] * s[:k]) @ Vt[:k]
            Zhat = ((Y[np.ix_(h, t)] @ Vt[:k].T / s[:k]) @ (U[:, :k].T @ Y[np.ix_(t, h)]))[off]
            num[0, i] += (fit * Ztt).sum(); den_a[0, i] += (fit ** 2).sum()
            num[1, i] += (Zhat * Zhh).sum(); den_a[1, i] += (Zhat ** 2).sum()
    return num / np.sqrt(den_a * den_b[:, None])


def all_curves():
    out, t0 = {}, time.time()
    for d in (10, 20, 30, 90):
        for S in SIZES:
            T = S // 10
            if T < 2 or (d == 90 and S < 50):
                continue
            draws = min(max(2, int(np.ceil(3000 / (T * (T - 1))))), 40) if S < 500 else 1
            out[f"{d},{S}"] = curves(d, S, draws).tolist()
        print(f"r_lin curves, d_E {d}: {time.time() - t0:.0f}s", flush=True)
    return out


def G(e, x, kind):
    u = x * e
    return (np.sinh(u) - u) / x if kind == "mf" else (np.expm1(u) - u) / x


def at(curve, rho):
    """a curve on RHOS at rho (linear in log rho above the first positive tabulated value)"""
    if rho <= RHOS[1]:
        return curve[0] + (curve[1] - curve[0]) * rho / RHOS[1]
    return float(np.interp(np.log(rho), np.log(RHOS[1:]), curve[1:]))


def evolve(curve, x, kind, scale, rounds=3000):
    """the map r <- r_lin(G(1 - r^2)) from e = 1: the alignment after each round"""
    e, hist = 1., []
    for _ in range(rounds):
        hist.append(at(curve, scale * G(e, x, kind)))
        e_new = 1 - min(max(hist[-1], 0.), 1.) ** 2
        if abs(e_new - e) < 1e-9:
            break
        e = e_new
    return hist


def goal_only(d=30, sizes=(50, 75, 100, 110, 125, 150, 250, 500), n=300_000, pairs=12_000):
    zp.rng = np.random.default_rng(99)
    cells = lambda z01, z10: 1 / (1 + np.exp(-zp.V(z01)[:, :, None] * zp.V(z10)[:, None, :]))   # [slot-0, slot-1 realization]

    def fit(F, X):
        """least-squares read-out of X (n, 2) from features F (n, k): W, the mean features, the noise ratio"""
        Fbar = F.mean(0)
        B = np.linalg.lstsq(X, F - Fbar, rcond=None)[0].T
        W = np.linalg.solve(B.T @ B, B.T)
        return W, Fbar, float((((F - Fbar) - X @ B.T) @ W.T).var(0).mean() / X.var(0).mean())

    z01, z10 = zp.sphere(d, n), zp.sphere(d, n)
    P, X = cells(z01, z10), np.stack([z01 - z01.mean(), z10 - z10.mean()], 1)
    (Wb, Fb, nb), (W1, F1, n1), (Wf, Ff, nf) = (fit(np.concatenate([P.mean(2), P.mean(1)], 1), X), fit(P.mean(1), X),
                                                fit(P.reshape(n, -1), X))
    print(f"read-out noise ratio, d_E {d}: goal only {nb:.2f} (a pair of training variables, 20 marginal cells), {n1:.2f} "
          f"(a held-out partner, 10 cells); both answers {nf:.2f} (100 cells)")
    for mode in ("both answers", "goal only"):
        row = []
        for S in sizes:
            T = S // 10
            h, t = np.arange(T), np.arange(T, S)
            a, b = [], []
            for _ in range(max(2, int(np.ceil(pairs / (T * (T - 1)))))):
                K = zp.rng.normal(size=(S, d)); K /= np.linalg.norm(K, axis=1, keepdims=True)
                Q = zp.rng.normal(size=(S, d)); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
                Z = K @ Q.T
                Y = np.zeros_like(Z)
                I, J = np.triu_indices(S, 1)               # i < j: slot 0 = i; the held-out variables are the first T
                for lo in range(0, len(I), 100_000):
                    i, j = I[lo:lo + 100_000], J[lo:lo + 100_000]
                    C = cells(Z[i, j], Z[j, i])
                    if mode == "both answers":
                        est = (C.reshape(len(i), -1) - Ff) @ Wf.T
                    else:
                        est = (np.concatenate([C.mean(2), C.mean(1)], 1) - Fb) @ Wb.T
                        est[i < T] = (C[i < T].mean(1) - F1) @ W1.T          # held-out variable in slot 0: goal = slot 1
                    Y[i, j], Y[j, i] = est[:, 0], est[:, 1]
                U, s, Vt = np.linalg.svd(Y[np.ix_(t, t)], full_matrices=False)
                k = min(d, S - T)
                Zhat = (Y[np.ix_(h, t)] @ Vt[:k].T / s[:k]) @ (U[:, :k].T @ Y[np.ix_(t, h)])
                off = ~np.eye(T, dtype=bool)
                a.append(Zhat[off]); b.append(Z[np.ix_(h, h)][off])
            row.append(f"{S}: {np.corrcoef(np.concatenate(a), np.concatenate(b))[0, 1]:.3f}")
        print(f"  proxy r, {mode:12s}: " + "  ".join(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", default="", help="JSON cache of the r_lin curves (written if absent)")
    ap.add_argument("--goal-only", action="store_true")
    a = ap.parse_args()
    if a.goal_only:
        return goal_only()
    if a.curves and os.path.exists(a.curves):
        C = json.load(open(a.curves))["curves"]
    else:
        C = all_curves()
        if a.curves:
            json.dump({"rhos": RHOS.tolist(), "curves": C}, open(a.curves, "w"))
    C = {k: np.array(v) for k, v in C.items()}
    for kind in ("mf", "avg"):
        print(f"\nthe map with rho_{kind}: round-0 r -> final r (rounds), per |S|")
        for (d, d0), rho in ROWS.items():
            x = 8 * np.pi ** 2 / d0
            scale = rho / zp.rho_mf(1 / d0)
            row, first = [], None
            for S in sorted(int(k.split(",")[1]) for k in C if k.startswith(f"{d},")):
                h = evolve(C[f"{d},{S}"][1], x, kind, scale)
                row.append(f"{S}: {h[0]:.2f}->{h[-1]:.2f}({len(h)})")
                first = S if first is None and h[-1] >= .9 else first
            print(f"  d_E {d}, z0^2 = 1/{d0}: runs away from |S| {first}; observed: last failure {OBSERVED[d, d0][0]}, "
                  f"first transfer {OBSERVED[d, d0][1]}\n     " + "  ".join(row))


if __name__ == "__main__":
    main()
