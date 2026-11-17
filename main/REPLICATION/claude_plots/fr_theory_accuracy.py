#!/usr/bin/env python3
"""
fr_theory_accuracy.py -- the accuracy lost by acting on the Naive belief against its bounds through Factorization Regret
(Proposition 4 of the theory text), and the three terms of the slope of expected Factorization Regret, in the main text's setting
(R = 10 realizations per variable, d_o = 5 observables, likelihood temperature lambda = 2, two distinct variables).  It prints the
numbers that the theory text quotes in "The generator of the main text, beyond the figure"; no figure, no cache.

    python fr_theory_accuracy.py            1,000 interaction sets x 40 episodes of 300 steps, about 1 min on 2 threads
    python fr_theory_accuracy.py 200        fewer interaction sets

THE STATEMENTS
  Proposition 4.  With a_J = argmax B_t and a_N = argmax Bn_t the actions selected from the Joint and the Naive belief,
        P(a_J = r_g) - P(a_N = r_g)  =  E[B_t(a_J) - B_t(a_N)]  <=  E[min(1, sqrt(2 FR_t))]  <=  min(1, sqrt(2 E[FR_t])),
      because B_t(a_J) - B_t(a_N) <= 2 TV(B_t, Bn_t) <= sqrt(2 FR_t) on every episode (Pinsker) and by Jensen's inequality.
      Both the gap and the first bound tend to the trapped share q.  The pathwise inequality is asserted on every episode and step.
  The slope.      E_r[Delta] = sum_i I(o^i; r_c | r_g) - E_r[min_a kappa(r, a)],   I(o^i; r_c | r_g) = I(o^i; r_c) - I(r_g; r_c; o^i):
      E_r[kappa(r, r_g)] is computed from the tables and, independently, from entropies (asserted equal).

RESULT (seed 2024, 1,000 x 40):  trapped share .475;  sum_i I(o^i; r_c | r_g) = .680, E[min_a kappa] = .593, E[Delta] = .087 nats a step;
  step   gap    E[min(1, sqrt(2 FR))]   min(1, sqrt(2 E[FR]))
     1   .023   .363                    .422
    30   .320   .742                    1      (the averaged bound is 1 from step 5 on)
   300   .468   .548                    1
The bound through single episodes is loose early and closes on the gap late (both tend to q).
"""
import os, sys, time
import numpy as np
from scipy.special import logsumexp, xlogy

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import fr_theory as F                                                                      # noqa: E402

LAM, N_EP, T, SEED = 2.0, 40, 300, 2024
STEPS = (1, 2, 3, 4, 5, 6, 10, 30, 100, 300)


def run(nz):
    rng = np.random.default_rng(SEED); t0 = time.time(); hb = lambda p: -(xlogy(p, p) + xlogy(1 - p, 1 - p))
    acc_j, acc_n, gap_b, bound, fr = (np.zeros(T) for _ in range(5)); worst = -np.inf; ceil, edelta, emin, qs = [], [], [], []
    for _ in range(nz):
        ell = F.checked_table(F.make_env(rng, LAM), LAM); d_o, R, Rc = ell.shape; tab = F.one_step_table(ell); m, kap = tab["m"], tab["kap"]
        kg = np.take_along_axis(kap, np.broadcast_to(np.arange(R)[:, None, None], (R, Rc, 1)), 2)[:, :, 0]         # kappa(r, r_g)
        ceil.append(kg.mean()); edelta.append(tab["Delta"].mean()); emin.append(kap.min(2).mean()); qs.append(tab["trapped"].mean())
        assert abs((hb(m).mean(1) - hb(ell).mean((1, 2))).sum() - kg.mean()) < 1e-10, "E_r[kappa(r, r_g)] is not sum_i I(o^i; r_c | r_g)"
        assert abs(kg.mean() - kap.min(2).mean() - tab["Delta"].mean()) < 1e-12
        l1, l0, lm1, lm0 = np.log(ell), np.log1p(-ell), np.log(m), np.log1p(-m)
        rg, rc = rng.integers(0, R, N_EP), rng.integers(0, Rc, N_EP); idx = np.arange(N_EP)
        obs = (rng.random((T, N_EP, d_o)) < ell[:, rg, rc].T[None]).astype(float); lj = np.zeros((N_EP, R, Rc)); ln = np.zeros((N_EP, R))
        for t in range(T):
            o = obs[t]; lj += np.einsum("ei,ijk->ejk", o, l1) + np.einsum("ei,ijk->ejk", 1 - o, l0); ln += o @ lm1 + (1 - o) @ lm0
            bj = np.exp(lj - lj.max(axis=(1, 2), keepdims=True)); bj /= bj.sum(axis=(1, 2), keepdims=True); bg = bj.sum(2)   # Joint goal belief
            lbn = ln - logsumexp(ln, axis=1, keepdims=True); f = np.maximum(np.sum(xlogy(bg, bg) - bg * lbn, axis=1), 0.0)     # Naive log belief; FR
            a_j, a_n = bg.argmax(1), lbn.argmax(1); g = bg[idx, a_j] - bg[idx, a_n]
            worst = max(worst, (g - np.sqrt(2 * f)).max())
            acc_j[t] += (a_j == rg).mean(); acc_n[t] += (a_n == rg).mean(); gap_b[t] += g.mean(); bound[t] += np.minimum(1.0, np.sqrt(2 * f)).mean(); fr[t] += f.mean()
    for a in (acc_j, acc_n, gap_b, bound, fr): a /= nz
    assert worst <= 1e-9, "B_t(a_J) - B_t(a_N) <= sqrt(2 FR_t) fails on an episode"
    print(f"{nz:,} interaction sets x {N_EP} episodes of {T} steps, lambda = {LAM:g}; trapped share {np.mean(qs):.3f}; {time.time() - t0:.0f} s")
    print(f"slope, nats a step: sum_i I(o^i; r_c | r_g) = {np.mean(ceil):.4f};  E[min_a kappa] = {np.mean(emin):.4f};  E[Delta] = {np.mean(edelta):.4f}")
    print("step   Joint   Naive   gap     E[B(a_J) - B(a_N)]   E[min(1, sqrt(2 FR))]   min(1, sqrt(2 E[FR]))   E[FR]")
    for t in STEPS:
        k = t - 1; print(f"{t:4d}   {acc_j[k]:.3f}   {acc_n[k]:.3f}   {acc_j[k] - acc_n[k]:.3f}   {gap_b[k]:.3f}                {bound[k]:.3f}                   "
                         f"{min(1.0, np.sqrt(2 * fr[k])):.3f}                   {fr[k]:.3f}")
    hit = np.sqrt(2 * fr) >= 1; print(f"sqrt(2 E[FR_t]) reaches 1 at step {int(np.argmax(hit)) + 1}" if hit.any() else f"sqrt(2 E[FR_t]) stays below 1 up to step {T}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
