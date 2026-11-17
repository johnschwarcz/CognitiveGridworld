#!/usr/bin/env python3
"""
fr_theory_verify_proof.py -- exact numerical check (no sampling) of the proof of Proposition 1 of the theory text ("Time course of
expected Factorization Regret") and of the short statements built on it.  Random likelihood tables with one binary observable and
uniform priors: given the joint realization r a history of t steps is one count n ~ Binomial(t, ell(r)), and both observers depend
on n only, so every expectation is a finite sum.
  PROOF OF PROPOSITION 1 (one concentration argument, Hoeffding's inequality, used for both observers)
  1. pathwise identity   FR_t - S^ahat_t = -H(B_t) - ln Bn_t(ahat) - sum_{a != r_g} B_t(a) S^a_t            (every r, every history)
  2. expected identity   E[FR_t] = Lam~_t - Lam_t = t E_r[Delta] + N~_t - Lam_t,  and  Lam_t = E[H(B_t)]    (Proposition 1(a))
  3. odds                1 / B_t(r_g) <= 1 + sum_{r': r'_g != r_g} e^{L_t(r') - L_t(r)};   1 / Bn_t(ahat) = 1 + sum_{a != ahat} e^{S^a - S^ahat}
  4. Hoeffding           every alternative's walk V_t (steps in [-K, K], mean -m < 0 per step) has  P(V_t > -m t / 2 | r) <= e^{-m^2 t / (8 K^2)},
                         for the Joint walks V = L(r') - L(r)  (mean -KL(P_r || P_r') < 0: the goal is identifiable)
                         and the Naive walks V = S^a - S^ahat  (mean kappa(r, ahat) - kappa(r, a) < 0: ahat is unique)
  5. hence               E[-ln B_t(r_g) | r]  and  E[-ln Bn_t(ahat) | r]  are at most the sum over the alternatives of
                         e^{-m t / 2} + (ln(1 + n) + K t) e^{-m^2 t / (8 K^2)}   (n alternatives; the terms are at most ln(1 + sum e^V) and at
                         most ln(1 + n) + K t): both vanish exponentially
  6. Proposition 1(b)    |sum_{a != r_g} B_t(a) S^a_t| <= K t (1 - B_t(r_g)) <= -K t ln B_t(r_g)   (pathwise),
                         i.e. the display of the proof  |FR_t - S^ahat_t| <= H(B_t) - ln Bn_t(ahat) + K t (1 - B_t(r_g));
                         E|FR_t - S^ahat_t| <= (1 + K t) Lam_t + N~_t,   and given r at most R^C times as much
  7. (no step 7: the later numbers are kept as they were)
  STATEMENTS BUILT ON IT
  8. the increment       E[FR_t+1] - E[FR_t] = E_r[Delta] - (N~_t - N~_t+1) + (Lam_t - Lam_t+1),  Lam_t - Lam_t+1 >= 0;
                         at every decreasing step  N~_t - N~_t+1 > E_r[Delta]                       ("Transient decreases")
  9. three log losses    Lam~_t >= Lam_t >= H(r_g | o_1:t, r_c);  Syn_t = Lam_t - H(r_g | o_1:t, r_c) = I(r_g; r_c | o_1:t)   (Proposition 2)
 10. late steps          Syn_t <= Lam_t <= ln R  and  E[FR_t] - Syn_t >= t E_r[Delta] - 2 ln R       ("When does ... bound synergy?")
 11. the step X        at every trapped r:  sum_o Pn(o | r_g) e^{X(o)} = 1,  E[X | r] = Delta(r),  hence  P(X < 0 | r) > 0
                         ("single trajectories keep decreasing")
 12. the slope         E_r[Delta] = I(o; r_c | r_g) - E_r[min_a kappa(r, a)],  with  I(o; r_c | r_g) = I(o; r_c) + I(r_g; r_c | o)
                         (= I(o; r_c) - I(r_g; r_c; o): Interaction Information is symmetric), each computed from entropies
 13. Proposition 4     a_J = argmax B_t, a_N = argmax Bn_t:  P(a_J = r_g) - P(a_N = r_g) = E[B_t(a_J) - B_t(a_N)]
                         <= E[min(1, sqrt(2 FR_t))] <= min(1, sqrt(2 E[FR_t])),  and on every history
                         B_t(a_J) - B_t(a_N) <= [B_t(a_J) - Bn_t(a_J)] + [Bn_t(a_N) - B_t(a_N)] <= ||B_t - Bn_t||_1 <= sqrt(2 FR_t)
                         (Pinsker; the limits -> q are printed, not asserted)
A violation larger than 1e-9 raises (1e-7 for the two bounds of 13 that take a square root of FR_t: an FR_t that is zero up to
rounding, 1e-16, has a square root that is off by 1e-8).  240 tables (R x R_c = 2x2, 3x3, 4x3, 3x5), t = 1 ... 1,000; about 10 s.
NOTATION.  The theory text writes the three expected log losses out; here Lam_t = E[-ln B_t(r_g)] (minus its "Joint term"), N~_t =
E[-ln Bn_t(atilde)] (its "Naive term"), Lam~_t = E[-ln Bn_t(r_g)], and ahat is its atilde (the action the Naive observer converges to).
The other statements of the theory text are checked by fr_theory_verify.py.

    python fr_theory_verify_proof.py
"""
import numpy as np
from scipy.special import gammaln, logsumexp, xlogy

rng = np.random.default_rng(20); TOL = 1e-9
worst = dict(id1=0.0, id2=0.0, odds=-np.inf, hoefJ=-np.inf, hoefN=-np.inf, conc=-np.inf, path=-np.inf, l1=-np.inf, l1r=-np.inf,
             inc=0.0, incJ=-np.inf, dipN=-np.inf, chain=-np.inf, syn=0.0, late=-np.inf, stepX=0.0, slope=0.0, acc=0.0, accB=-np.inf)
n_dec = 0; n_trap = 0
T_LIST = (1, 2, 3, 5, 6, 20, 21, 80, 81, 320, 321, 1000)                                          # consecutive pairs: the increment (8)
def dkl(p, q): return xlogy(p, p / q) + xlogy(1 - p, (1 - p) / (1 - q))


def table(R, Rc):
    while True:
        ell = rng.uniform(0.05, 0.95, (R, Rc)); m = ell.mean(1)
        kap = dkl(ell[:, :, None], m[None, None, :]); srt = np.sort(kap, axis=2)                  # kap[rg, rc, a]
        if (srt[:, :, 1] - srt[:, :, 0]).min() > 1e-3: return ell, m, kap                         # (A3) with a margin


def check(R, Rc):
    global n_dec, n_trap
    ell, m, kap = table(R, Rc); ahat = kap.argmin(2); rg = np.arange(R)[:, None] * np.ones((1, Rc), int)
    Delta = np.take_along_axis(kap, rg[:, :, None], 2)[:, :, 0] - kap.min(2); EDelta = Delta.mean(); RC = R * Rc
    K = max(np.abs(np.log(ell)).max(), np.abs(np.log1p(-ell)).max(), np.abs(np.log(m)).max(), np.abs(np.log1p(-m)).max())
    out = []; terms = {}
    hb = lambda q_: -(xlogy(q_, q_) + xlogy(1 - q_, 1 - q_)); po = ell.mean(); post1, post0 = ell / ell.sum(), (1 - ell) / (1 - ell).sum()   # 12
    mi = lambda P: (xlogy(P, P / (P.sum(1, keepdims=True) * P.sum(0, keepdims=True)))).sum()                                   # I(r_g; r_c) under a joint table P
    I_c_given_g = (hb(m) - hb(ell).mean(1)).mean(); I_c = hb(po) - hb(ell.mean(0)).mean(); I_gc_given_o = po * mi(post1) + (1 - po) * mi(post0)
    worst["slope"] = max(worst["slope"], abs(EDelta - (I_c_given_g - kap.min(2).mean())), abs(I_c_given_g - (I_c + I_gc_given_o)))
    for g, c in zip(*np.nonzero(ahat != rg)):                                                                                  # 11
        a_ = ahat[g, c]; X1, X0 = np.log(m[a_] / m[g]), np.log((1 - m[a_]) / (1 - m[g])); n_trap += 1                          # X at o = 1 and at o = 0
        worst["stepX"] = max(worst["stepX"], abs(m[g] * np.exp(X1) + (1 - m[g]) * np.exp(X0) - 1), abs(ell[g, c] * X1 + (1 - ell[g, c]) * X0 - Delta[g, c]))
        assert min(X1, X0) < 0 < max(X1, X0)                                                                                   # each outcome has probability > 0: P(X < 0 | r) > 0
    for t in T_LIST:
        n = np.arange(t + 1); logC = gammaln(t + 1) - gammaln(n + 1) - gammaln(t - n + 1)
        L = n[:, None, None] * np.log(ell)[None] + (t - n)[:, None, None] * np.log1p(-ell)[None]        # joint log-likelihood [n, rg, rc]
        Ln = n[:, None] * np.log(m)[None] + (t - n)[:, None] * np.log1p(-m)[None]                       # naive log-likelihood [n, a]
        lpost = L - logsumexp(L.reshape(t + 1, -1), axis=1)[:, None, None]; post = np.exp(lpost)        # joint posterior of (rg, rc) given n
        lB = logsumexp(lpost, axis=2); B = np.exp(lB)                                                   # joint goal marginal
        lBn = Ln - logsumexp(Ln, axis=1)[:, None]                                                       # naive belief
        FR = (B * (lB - lBn)).sum(1); H = -(xlogy(B, B)).sum(1); Pn = np.exp(logC[:, None, None] + L)   # P(n | r)
        prc = post.sum(1); Hc = -(xlogy(post, post)).sum((1, 2)) + (xlogy(prc, prc)).sum(1)             # H(r_g | n, r_c), averaged over r_c given n
        lam_r = np.einsum("ngc,ng->gc", Pn, -lB); H_r = np.einsum("ngc,n->gc", Pn, H)
        ntil_r = np.zeros((R, Rc)); lamn_r = np.zeros((R, Rc)); l1_r = np.zeros((R, Rc))
        for g in range(R):
            S = Ln - Ln[:, [g]]; mask = np.arange(R) != g
            for c in range(Rc):
                w = Pn[:, g, c]; a_ = ahat[g, c]; oth = np.arange(R) != a_
                third = (B[:, mask] * S[:, mask]).sum(1)
                worst["id1"] = max(worst["id1"], np.abs(FR - S[:, a_] - (-H - lBn[:, a_] - third)).max())                      # 1
                V = L - L[:, [g], :][:, :, [c]]                                                                                # 3
                worst["odds"] = max(worst["odds"], (-lB[:, g] - np.logaddexp(0, logsumexp(V[:, mask, :].reshape(t + 1, -1), axis=1))).max(),
                                    np.abs(-lBn[:, a_] - np.logaddexp(0, logsumexp(S[:, oth] - S[:, [a_]], axis=1))).max())
                nJ = int(mask.sum()) * Rc; bound_J = 0.0                                                                       # 4, 5: Joint (alternatives r' with another goal)
                for g2 in np.nonzero(mask)[0]:
                    for c2 in range(Rc):
                        mJ = dkl(ell[g, c], ell[g2, c2]); assert mJ > 0; tail = np.exp(-mJ ** 2 * t / (8 * K ** 2))
                        worst["hoefJ"] = max(worst["hoefJ"], w[V[:, g2, c2] > -mJ * t / 2].sum() - tail)
                        bound_J += np.exp(-mJ * t / 2) + (np.log1p(nJ) + K * t) * tail
                nN = int(oth.sum()); bound_N = 0.0                                                                             # 4, 5: Naive (alternatives a != ahat)
                for a in np.nonzero(oth)[0]:
                    mN = kap[g, c, a] - kap[g, c, a_]; assert mN > 0; tail = np.exp(-mN ** 2 * t / (8 * K ** 2))
                    worst["hoefN"] = max(worst["hoefN"], w[S[:, a] - S[:, a_] > -mN * t / 2].sum() - tail)
                    bound_N += np.exp(-mN * t / 2) + (np.log1p(nN) + K * t) * tail
                ntil_r[g, c] = w @ -lBn[:, a_]; lamn_r[g, c] = w @ -lBn[:, g]; l1_r[g, c] = w @ np.abs(FR - S[:, a_])
                worst["conc"] = max(worst["conc"], lam_r[g, c] - bound_J, ntil_r[g, c] - bound_N)                                                              # 5
                worst["path"] = max(worst["path"], (np.abs(third) - K * t * (1 - B[:, g])).max(), ((1 - B[:, g]) + lB[:, g]).max(),                             # 6
                                    (np.abs(FR - S[:, a_]) - (H - lBn[:, a_] + K * t * (1 - B[:, g]))).max())
        Pm = Pn.mean((1, 2)); Lam, LamN, Nt, EFR, EH, L1 = lam_r.mean(), lamn_r.mean(), ntil_r.mean(), Pm @ FR, H_r.mean(), l1_r.mean()
        aJ, aN = B.argmax(1), lBn.argmax(1); nn = np.arange(t + 1); gB = B[nn, aJ] - B[nn, aN]; fr0 = np.maximum(FR, 0.0)        # 13
        accJ = np.mean([Pn[aJ == g_, g_, :].sum(0) for g_ in range(R)]); accN = np.mean([Pn[aN == g_, g_, :].sum(0) for g_ in range(R)])
        Bn = np.exp(lBn); l1n = np.abs(B - Bn).sum(1); mid = (B[nn, aJ] - Bn[nn, aJ]) + (Bn[nn, aN] - B[nn, aN])
        b1 = Pm @ np.minimum(1.0, np.sqrt(2 * fr0)); b2 = min(1.0, np.sqrt(2 * max(EFR, 0.0)))
        worst["acc"] = max(worst["acc"], abs((accJ - accN) - Pm @ gB))
        worst["accB"] = max(worst["accB"], (gB - mid).max(), (mid - l1n).max(), (l1n ** 2 - 2 * FR).max(), Pm @ gB - b1 - 1e-7, b1 - b2 - 1e-7)   # Pinsker squared: a square
                                                                                  # root of an FR that is zero up to rounding (1e-16) is off by 1e-8
        worst["id2"] = max(worst["id2"], abs(EFR - (LamN - Lam)), abs(EFR - (t * EDelta + Nt - Lam)), abs(EH - Lam))           # 2
        bnd = (1 + K * t) * Lam + Nt
        worst["l1"] = max(worst["l1"], L1 - bnd); worst["l1r"] = max(worst["l1r"], (l1_r - RC * bnd).max(), (H_r - RC * Lam).max())   # 6
        LamC = Pm @ Hc; Syn = Lam - LamC; I_direct = Pm @ (H - Hc)                                                             # 9, 10
        worst["chain"] = max(worst["chain"], Lam - LamN, LamC - Lam); worst["syn"] = max(worst["syn"], abs(Syn - I_direct))
        worst["late"] = max(worst["late"], Syn - Lam, Lam - np.log(R), (t * EDelta - 2 * np.log(R)) - (EFR - Syn))
        terms[t] = (EFR, Nt, Lam); out.append((t, EFR, t * EDelta, Nt, Lam, L1, bnd, Syn, accJ - accN, b1))
    for t in T_LIST:                                                                                                           # 8
        if t + 1 in terms:
            (e0, n0, l0), (e1, n1, l1) = terms[t], terms[t + 1]
            worst["inc"] = max(worst["inc"], abs((e1 - e0) - (EDelta - (n0 - n1) + (l0 - l1)))); worst["incJ"] = max(worst["incJ"], l1 - l0)
            if e1 < e0 - 1e-12: n_dec += 1; worst["dipN"] = max(worst["dipN"], EDelta - (n0 - n1))
    return out, int((Delta > 0).sum())


for R, Rc in ((2, 2), (3, 3), (4, 3), (3, 5)):
    for rep in range(60):
        out, k = check(R, Rc)
        if rep == 0:
            print(f"R = {R}, Rc = {Rc}, trapped joint realizations k = {k}")
            for t, e, lin, Nn, Lam, l1, b, syn, gap, b1 in out:
                print(f"   t {t:5d}: E[FR] {e:9.5f} = t E[Delta] {lin:9.5f} + N~ {Nn:8.5f} - Lam {Lam:8.5f};  Syn {syn:8.5f};  E|FR - walk| {l1:9.2e} <= (1 + K t) Lam + N~ = {b:9.2e};  "
                      f"accuracy lost {gap:.4f} <= E[min(1, sqrt(2 FR))] {b1:.4f}  (q = {k / (R * Rc):.4f})")
print("largest violations:", {k: float(f"{v:.2e}") for k, v in worst.items()})
print(f"decreasing steps among the checked pairs (t, t + 1): {n_dec}; trapped joint realizations: {n_trap}")
assert all(v < TOL for v in worst.values())
print("all checks passed (240 tables)")
