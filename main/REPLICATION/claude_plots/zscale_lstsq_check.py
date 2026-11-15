"""Support for the law M >= d_E sinh(x) / x (zscale_law.py; John 11-14: "can you support this from literature or
derivations?"): does the fold residual of the REAL likelihood act as plain noise in a least-squares fit of one variable's
embedding? No network, nothing fitted.

A held-out variable h has an unknown embedding k_h (d_E numbers). Its M partners have known unit embeddings q_j. Each
pair's 100 cells are pooled by the matched filter into y_j = z_j + fold residual, z_j = scale <k_h, q_j> (zscale_proxy.py's
pooling; the residual is a function of z, uncorrelated with it, variance rho z0^2). Least squares on (q_j, y_j) gives k_hat.
Score: error / signal of scale <k_hat, q_new> against z_new for fresh partners.

The classical result for INDEPENDENT noise of ratio rho and a Gaussian design is error / signal = rho d_E / (M - d_E - 1),
which crosses 1 at M = d_E (1 + rho) + 1: the law, with the exact likelihood's rho in place of sinh(x) / x - 1. The check is
whether a residual that is a function of z behaves the same. Result (11-14, this script): it does, within about 10% at
the predicted M in all six worlds.

    python zscale_lstsq_check.py            # about 2 minutes on 4 cores
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zscale_proxy as zp

WORLDS = ((30, 30), (20, 20), (90, 90), (10, 30), (60, 60), (90, 30))      # (d_E, 1 / z0^2)
FRACTIONS = (.8, .9, 1., 1.1, 1.25, 1.5, 2.)                               # M as a fraction of d_E (1 + rho)


def unit(n, d):
    g = zp.rng.normal(size=(n, d))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def readout(z01, z10, W, Pbar):
    """the matched-filter estimate of z01 from a pair's 100 cells"""
    P = 1 / (1 + np.exp(-zp.V(z01)[:, :, None] * zp.V(z10)[:, None, :]))
    return (P.reshape(len(z01), -1) - Pbar) @ W[0]


def error_over_signal(d, scale, M, W, Pbar, draws=300, fresh=400):
    num = den = 0.
    for _ in range(draws):
        k, kq = unit(1, d)[0], unit(1, d)[0]                   # h's K (slot 0) and Q (slot 1: sets the partner-side z10)
        Qp, Kp = unit(M, d), unit(M, d)                        # the partners' Q (known regressors) and K
        y = readout(scale * Qp @ k, scale * Kp @ kq, W, Pbar)
        khat = np.linalg.lstsq(scale * Qp, y, rcond=None)[0]
        Qn = unit(fresh, d)
        zt, zh = scale * Qn @ k, scale * Qn @ khat
        num += ((zh - zt) ** 2).sum()
        den += (zt ** 2).sum()
    return num / den


def main():
    zp.rng = np.random.default_rng(31415)
    print("error / signal of the least-squares embedding on fresh partners; in brackets the classical rho d / (M - d - 1)")
    for d, ref in WORLDS:
        scale = float(np.sqrt(d / ref))
        W, Pbar, rho, _ = zp.pooling(d, scale, n=200_000)
        x = 8 * np.pi ** 2 / ref
        need = d * (1 + rho)
        row = []
        for frac in FRACTIONS:
            M = int(round(frac * need))
            if M <= d + 2:
                row.append(f"{frac:g}x: M {M} <= d_E + 2")
                continue
            row.append(f"{frac:g}x: M {M} {error_over_signal(d, scale, M, W, Pbar):.2f} [{rho * d / (M - d - 1):.2f}]")
        print(f"d_E {d}, z0^2 = 1/{ref}: rho {rho:.3f} (closed form {np.sinh(x) / x - 1:.3f}), d_E (1 + rho) = {need:.0f}\n    "
              + "  ".join(row), flush=True)


if __name__ == "__main__":
    main()
