#!/usr/bin/env python3
"""
fr_theory_popdip.py -- the expected curve of the Cognitive Gridworld ITSELF can dip (short theory text, "Is the expected curve
monotone?": "With few realizations and many observables trapped joint realizations are rare, and the expected curve of the generator
of the main text itself dips").

Every episode samples its own interaction set (fr_theory.make_env: the paper's generator, two distinct variables), as in the main text;
both observers are exact.  Two populations at lambda = 2, steps 1-40:
    R = 3,  d_o = 15   few realizations, many observables: traps are rare (growth rate E[Delta] small, panel l of the theory figure)
    R = 10, d_o = 5    the main text
PRE-SPECIFIED TEST (fixed from an independent first run by the checker of the text, 11-14, other seeds): E[FR_4] - E[FR_14] > 0, paired
over interaction sets.
RESULT (seeds below; ~1 min on 4 threads):
    R = 3, d_o = 15 (1,000 sets x 400 episodes): the mean rises to .272 at step 4, decreases at steps 4-15 to .223 at step 16 and
        rises again; E[FR_4] - E[FR_14] = .047 +- .006 (z = 8.5).  Other seeds: .045 +- .006 (z = 7.9); the checker's runs .051 +-
        .006 (z = 9.3), and .274 -> .236 in its first run.
    R = 10, d_o = 5 (300 sets x 200 episodes): no step at which the mean decreases (E[FR_30] = 2.88).

    python fr_theory_popdip.py
"""
import os, sys, time
os.environ.setdefault("MPLBACKEND", "Agg"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fr_theory as F

LAM, STEPS = 2.0, np.arange(1, 41)


def lse(a, axis):
    m = a.max(axis=axis, keepdims=True); return np.squeeze(m, axis) + np.log(np.exp(a - m).sum(axis=axis))


def population(R, d_o, n_sets, n_ep, seed):
    """Mean FR_t per interaction set (rows) at the steps STEPS: n_ep episodes per set, joint realizations uniform."""
    rng, erng = np.random.default_rng(seed), np.random.default_rng(seed + 1); out = np.zeros((n_sets, len(STEPS)))
    for z in range(n_sets):
        ell = F.checked_table(F.make_env(rng, LAM, R=R, d_o=d_o), LAM); m = ell.mean(2)                     # ell[i, r_g, r_c]; Naive marginals m[i, a]
        rg, rc = erng.integers(0, R, n_ep), erng.integers(0, R, n_ep); p = ell[:, rg, rc].T; n = np.zeros((n_ep, d_o))
        lE, l1E, lm, l1m = np.log(ell), np.log1p(-ell), np.log(m), np.log1p(-m)
        for i, t in enumerate(STEPS):
            n += erng.random((n_ep, d_o)) < p                                                                # counts of ones per observable
            LJ = np.einsum("ei,igc->egc", n, lE) + np.einsum("ei,igc->egc", t - n, l1E); lB = lse(LJ, 2); lB -= lse(lB, 1)[:, None]
            LN = n @ lm + (t - n) @ l1m; lN = LN - lse(LN, 1)[:, None]
            out[z, i] = np.sum(np.exp(lB) * (lB - lN), axis=1).mean()
    return out


if __name__ == "__main__":
    for R, d_o, n_sets, n_ep, seed in ((3, 15, 1000, 400, 20261114), (10, 5, 300, 200, 20261115)):
        t0 = time.time(); P = population(R, d_o, n_sets, n_ep, seed); mean = P.mean(0); inc = np.diff(P, axis=1)
        z = inc.mean(0) / (inc.std(0) / np.sqrt(n_sets)); d = P[:, 3] - P[:, 13]; down = STEPS[:-1][inc.mean(0) < 0]
        print(f"R = {R}, d_o = {d_o}, lambda = {LAM:g}: {n_sets} interaction sets x {n_ep} episodes  [{time.time() - t0:.0f} s]")
        print("   E[FR_t], t = 1..40:", np.array2string(mean, precision=3, max_line_width=220))
        print(f"   steps at which the mean decreases: {list(down) if len(down) else 'none'}" + (f" (z from {z[inc.mean(0) < 0].min():.1f} to {z[inc.mean(0) < 0].max():.1f})" if len(down) else ""))
        print(f"   pre-specified test E[FR_4] - E[FR_14] = {d.mean():+.4f} +- {d.std() / np.sqrt(n_sets):.4f} (paired over sets; z = {d.mean() / (d.std() / np.sqrt(n_sets)):+.1f})")
