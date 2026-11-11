#!/usr/bin/env python3
"""fr_theory_verify.py -- the identities and inequalities of the theory block (FR_theory_block_v13.tex), checked numerically.

For John (11-11: "I will need to be able to verify everything"): the proofs are in the tex; this file is an independent check.
It draws random likelihood tables (R goal x Rc context realizations, d_o = 1 or 2 binary observables, uniform priors), computes
every quantity EXACTLY (sums over the counts of ones: no sampling, so a failure cannot be noise), and prints, per claim, the
largest violation over all tables and steps.  A claim passes when that number is zero up to rounding (tolerance 1e-9).

    python fr_theory_verify.py            ~1 min
    python fr_theory_verify.py --tables 1000

WHAT IS CHECKED EXACTLY (each of these checks can fail)
  Prop 1  (a) identity and increment, (b) anatomy and its two monotone terms, the bound on decreases, D_0; (c) the linear-growth
          identity; the same identities on tables WITH TIES (a wrong action fitting exactly as well as the optimal one, two
          distractors tying), where the tie counts as a distractor
  growth  E[Delta] in terms of Interaction Information; the decomposition E[FR] = TC - KL and the ceiling on E[FR]
  Prop 2  mean and variance of the step X; the Chebyshev bound of (b) in BOTH directions (mu_t > 0 and mu_t < 0), with the
          exact law of the increment
  Cor 1   the mixture decreases at step t exactly when q < q*_t
  Prop 3  the three informations and their identities;  Prop 4  the accuracy bound;  Prop 5  (a) and (b)
WHAT IS ONLY SHOWN ALONG t (limits cannot be exact at a finite step; printed for the two tables of the paper's figure and
asserted at the last step): Lambda_t, N_t, the offset from the line, U_t, the trapped growth rate, the accuracy gap -> q,
Var(FR_t), E[eps_t^2] of Prop 2(a), and the single-episode dip probability -> P(X < 0 | r).
WHAT IS NOT CHECKED HERE: Corollary 2 over a set of steps (a union bound over the per-step bounds checked above; simulated
averages are compared with it in fig_fr_threshold a, `python fr_theory_additions.py numbers`); the liminf / limsup chain of
Prop 2(b) and Chernoff's bound (fig_fr_averages compares the law of X-bar_N with simulation); anything about sampled
interaction sets (a continuum of Z), which the tex states separately.

Notation as in the tex: B = Joint goal marginal, Bn = Naive belief, kappa(r, a) = KL(P(o | r) || Naive model of a), Delta(r) =
kappa(r, r_g) - min_a kappa(r, a); a wrong action a is a TRAP at r if kappa(r, a) < kappa(r, r_g), a DISTRACTOR otherwise.
"""
import itertools, sys
import numpy as np
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import norm

TOL = 1e-9


def dkl_bern(p, q): return xlogy(p, p / q) + xlogy(1 - p, (1 - p) / (1 - q))


def entropy(p): return -xlogy(p, p).sum(axis=-1)


class Table:
    """Exact quantities of one likelihood table ell[i, r_g, r_c] at every step t = 0..T (uniform prior over (r_g, r_c))."""
    def __init__(self, ell, T, steps=None):
        self.ell = ell = np.asarray(ell, float); self.d, self.R, self.Rc = ell.shape; self.T = T
        self.m = ell.mean(axis=2)                                                     # Naive marginals m[i, a]
        self.kap = dkl_bern(ell[:, :, :, None], self.m[:, None, None, :]).sum(0)      # kappa[r_g, r_c, a]
        rg = np.arange(self.R)[:, None]; self.kg = np.take_along_axis(self.kap, np.broadcast_to(rg[:, :, None], (self.R, self.Rc, 1)), 2)[:, :, 0]
        self.Delta = self.kg - self.kap.min(axis=2); self.trapped = self.kap.min(axis=2) < self.kg           # a trap fits strictly better
        self.ahat = np.where(self.trapped, self.kap.argmin(axis=2), rg)                # a minimizer of kappa; r_g whenever r_g is one
        self.is_trap = self.kap < self.kg[:, :, None]                                 # [r_g, r_c, a]
        self.is_dist = ~self.is_trap & (np.arange(self.R)[None, None, :] != rg[:, :, None])
        self.k = int(self.trapped.sum()); self.q = self.trapped.mean()
        self.steps = {t: self._step(t) for t in (range(T + 1) if steps is None else steps)}

    def _step(self, t):
        d, R, Rc, ell = self.d, self.R, self.Rc, self.ell
        counts = np.array(list(itertools.product(range(t + 1), repeat=d)))            # count vectors n (M, d)
        logC = (gammaln(t + 1) - gammaln(counts + 1) - gammaln(t - counts + 1)).sum(1)
        logL = np.einsum("mi,ijk->mjk", counts, np.log(ell)) + np.einsum("mi,ijk->mjk", t - counts, np.log1p(-ell))    # log P(sequence | r)
        P = np.exp(logC[:, None, None] + logL)                                        # P(n | r)
        logJ = logL - logsumexp(logL, axis=(1, 2), keepdims=True)                     # Joint posterior over r
        logB = logsumexp(logJ, axis=2)                                                # Joint goal marginal
        logLn = counts @ np.log(self.m) + (t - counts) @ np.log1p(-self.m); logBn = logLn - logsumexp(logLn, axis=1, keepdims=True)
        B, Bn = np.exp(logB), np.exp(logBn); FR = np.sum(B * (logB - logBn), axis=1)   # FR(n)
        W = P / (R * Rc); Pn = W.sum(axis=(1, 2))                                     # joint weight of (n, r); P(n)
        out = dict(counts=counts, P=P, FR=FR, B=B, Bn=Bn, logB=logB, logBn=logBn, EFR=Pn @ FR, Pn=Pn, VFR=Pn @ FR ** 2 - (Pn @ FR) ** 2)
        out["Lam"] = -np.einsum("mjk,mj->", W, logB); out["LamN"] = -np.einsum("mjk,mj->", W, logBn)
        out["Ntil"] = -np.sum(W * logBn[:, self.ahat])                                # Naive log loss on the action it converges to
        S = logLn[:, None, :] - logLn[:, :, None]                                     # S[n, r_g, a] = Naive log-odds of a against r_g
        St = np.where(self.is_trap[None], S[:, :, None, :], -np.inf); Sd = np.where(self.is_dist[None], S[:, :, None, :], -np.inf)
        Tt = np.logaddexp(0.0, logsumexp(St, axis=3)); Dt = np.log1p(np.exp(logsumexp(Sd, axis=3) - Tt))      # trap and distraction terms [n, r_g, r_c]
        out["ET"], out["ED"] = np.sum(W * Tt), np.sum(W * Dt)
        post_c = np.exp(logJ - logB[:, :, None])                                      # P(r_c | r_g, n)
        out["Syn"] = Pn @ np.sum(B * (entropy(np.exp(logsumexp(logJ, axis=1)))[:, None] - entropy(post_c)), axis=1) if t else 0.0   # I(r_g; r_c | n)
        logLc = logL - logsumexp(logL, axis=1, keepdims=True)                         # posterior of r_g given r_c (the context-informed observer)
        out["LamC"] = -np.sum(W * logLc)
        out["accJ"] = Pn @ B[np.arange(len(B)), B.argmax(1)]; out["accN"] = Pn @ B[np.arange(len(B)), Bn.argmax(1)]
        out["sqrt"] = Pn @ np.sqrt(2 * np.maximum(FR, 0))
        # total correlation of the atoms given the goal and the offset: E[FR] = TC - KL(P(x) || Ptilde(x)) (the paper's decomposition)
        Hc = Pn @ np.sum(B * entropy(post_c), axis=1); out["I_xc_g"] = np.log(Rc) - Hc    # I(x_{1:n}; r_c | r_g)
        lp = logsumexp(logL, axis=(1, 2)) - np.log(R * Rc); lpn = logsumexp(logLn, axis=1) - np.log(R)       # log P(sequence), log Ptilde(sequence)
        out["KLoff"] = Pn @ (lp - lpn)
        Sx = np.where(self.trapped[None], logLn[:, self.ahat] - logLn[:, np.arange(R)][:, :, None], 0.0)     # sum of the steps X: S^ahat (0 if untrapped)
        out["eps2"] = np.sum(W * (FR[:, None, None] - Sx) ** 2)                       # E[(FR_t - sum_s X_s)^2]
        U = np.einsum("mjk,m->jk", P, FR)                                             # E[FR_t | r]
        out["U"] = U[~self.trapped].mean() if (~self.trapped).any() else np.nan; out["W"] = U[self.trapped].mean() if self.k else np.nan
        return out

    def series(self, key): return np.array([self.steps[t][key] for t in sorted(self.steps)])


def interaction_information(ell_i):
    """I(r_g; r_c; o) = -I(r_g; r_c | o) for one binary observable with uniform independent priors; also I(o; r_c) and I(o; r_c | r_g)."""
    R, Rc = ell_i.shape; pj = np.stack([ell_i, 1 - ell_i]) / (R * Rc)                  # p(o, r_g, r_c)
    po = pj.sum(axis=(1, 2)); cmi = 0.0
    for o in range(2):
        pc = pj[o] / po[o]; cmi += po[o] * np.sum(xlogy(pc, pc / (pc.sum(1, keepdims=True) * pc.sum(0, keepdims=True))))
    p_oc = pj.sum(axis=1); I_oc = np.sum(xlogy(p_oc, p_oc / (po[:, None] / Rc)))
    I_oc_g = np.mean(dkl_bern(ell_i, ell_i.mean(1, keepdims=True)))
    return -cmi, I_oc, I_oc_g


def random_table(rng, d, R, Rc, kind="generic"):
    if kind == "goal_only": return np.broadcast_to(rng.uniform(0.1, 0.9, (d, R, 1)), (d, R, Rc)).copy()
    if kind == "context_only": return np.broadcast_to(rng.uniform(0.1, 0.9, (d, 1, Rc)), (d, R, Rc)).copy()
    if kind == "cyclic":
        f = rng.uniform(0.1, 0.9, (d, R)); return f[:, (np.arange(R)[:, None] + np.arange(R)[None, :]) % R]
    a = rng.uniform(0.5, 2.5); return 1 / (1 + np.exp(-a * rng.normal(size=(d, R, Rc))))


def dip_exact(x, w, N):
    """P(sum of N draws of the law (x, w) < 0) by FFT (as dip_fft in fr_theory_additions.py)."""
    mu = w @ x; sd = np.sqrt(w @ x ** 2 - mu ** 2); n = int(2 ** max(16, np.ceil(np.log2(270 * np.sqrt(N)))))
    ring = 2.0 * (N * abs(mu) + 12 * np.sqrt(N) * sd + np.abs(x).max()); h = ring / n
    j = x / h; j0 = np.floor(j).astype(int); f = j - j0
    pmf = np.bincount(j0 % n, weights=w * (1 - f), minlength=n) + np.bincount((j0 + 1) % n, weights=w * f, minlength=n)
    pN = np.fft.irfft(np.fft.rfft(pmf) ** N, n); return float(np.clip(pN[n // 2:].sum(), 0, 1))


def toy_tables():
    """The two tables of the paper's figure (fig_fr_theory a: no trap; b: one trapped joint realization of 9)."""
    u = np.array([-1.0, 0.0, 1.0]); lg = 1.2 * u[:, None] + 0.72 * u[None, :] / 2; lgb = lg.copy(); lgb[0, 2] += 0.6
    return 1 / (1 + np.exp(-lg))[None], 1 / (1 + np.exp(-lgb))[None]


def limits():
    """The limits (t -> infinity) cannot be checked exactly at a finite step: print them along t for the two tables of the paper's
    figure, where every expectation is an exact sum, and require each to be near its limit at the last step.  Returns the number
    of limits that are not."""
    ts = (30, 100, 300, 1000, 3000, 10_000, 30_000, 100_000); steps = sorted(set(ts) | {t + 1 for t in ts}); bad = 0
    print("\nLimits along t, exact, in the two tables of the figure (each column approaches the value in its header; * = not within tolerance at the last step):")
    for name, ell in zip(("table a (k = 0)", "table b (k = 1)"), toy_tables()):
        tb = Table(ell, None, steps=steps); ED_ = tb.Delta.mean(); dtrap = tb.Delta[tb.trapped].mean() if tb.k else 0.0; varD = tb.Delta.var()
        lm, l1m = np.log(tb.m / (1 - tb.m)), np.log1p(-tb.m)
        Ev = np.mean([tb.ell[0, g, c] * (1 - tb.ell[0, g, c]) * (lm[0, tb.ahat[g, c]] - lm[0, g]) ** 2 for g in range(3) for c in range(3)])
        px = None
        if tb.k:                                                                          # the trapped joint realization: P(X < 0 | r)
            g, c = np.argwhere(tb.trapped)[0]; a = tb.ahat[g, c]; th = lm[0, a] - lm[0, g]; c0 = l1m[0, a] - l1m[0, g]; p1 = tb.ell[0, g, c]
            px = p1 * (th + c0 < 0) + (1 - p1) * (c0 < 0)
        print(f"  {name}: q = {tb.q:.3f}, E[Delta] = {ED_:.5f}" + (f", mean Delta of the trapped = {dtrap:.4f}, P(X < 0 | r) = {px:.4f}" if tb.k else ""))
        print("        t  Lambda_t  N_t       E[FR_t]-tE[Delta]  U_t       E[eps_t^2]  gap-q     " + ("W_t+1-W_t  Var ratio  P(dip | trapped r)" if tb.k else ""))
        for t in ts:
            a, b = tb.steps[t], tb.steps[t + 1]; row = [max(a["Lam"], 0.0), a["Ntil"], a["EFR"] - t * ED_, a["U"], a["eps2"], a["accJ"] - a["accN"] - tb.q]
            line = f"  {t:7d}  {row[0]:.1e}   {row[1]:.1e}   {row[2]:+.1e}           {row[3]:.1e}   {row[4]:.1e}     {row[5]:+.4f}   "
            if tb.k:
                P0 = a["P"][:, g, c]; d1 = b["FR"][1:] - a["FR"]; d0 = b["FR"][:-1] - a["FR"]; pdip = P0 @ (p1 * (d1 < 0) + (1 - p1) * (d0 < 0))
                extra = [b["W"] - a["W"], a["VFR"] / (t ** 2 * varD + t * Ev), pdip]; line += f"{extra[0]:.4f}     {extra[1]:.4f}     {extra[2]:.4f}"
            print(line)
        tol = [1e-3] * 6; fails = [abs(v) > tl for v, tl in zip(row, tol)]
        if tb.k: fails += [abs(extra[0] - dtrap) > 1e-3, abs(extra[1] - 1) > 1e-2, abs(extra[2] - px) > 1e-2]
        bad += sum(fails); print("  -> " + ("all limits reached at the last step" if not sum(fails) else f"* {sum(fails)} limit(s) NOT reached at the last step"))
    return bad


def ties(check, rng):
    """Proposition 1 on tables WITH TIES (only (A1) holds): two goal realizations with the same likelihoods (a wrong action that fits
    exactly as well as the optimal one, at every joint realization of either) and the symmetric table of fig_fr_theory a (two
    distractors tie at the middle joint realization).  A tie counts as a distractor; a-hat = r_g whenever r_g is a minimizer."""
    n_tie = 0
    for it in range(40):
        d = 1 + it % 2; R = 3; Rc = 2 + it % 2; ell = random_table(rng, d, R, Rc); ell[:, 1] = ell[:, 0]          # goal realizations 0 and 1 identical
        tabs = [ell] + ([toy_tables()[0]] if it == 0 else [])
        for e in tabs:
            tb = Table(e, 20); tt = np.arange(21); FR, Lam, LamN, Nt, ET, ED = (tb.series(k) for k in ("EFR", "Lam", "LamN", "Ntil", "ET", "ED"))
            n_tie += int(np.sum(np.isclose(tb.kap, tb.kg[:, :, None], atol=1e-14) & (np.arange(tb.R)[None, None, :] != np.arange(tb.R)[:, None, None])))
            check("ties       Prop 1(a) identity", np.abs(FR - (LamN - Lam)).max())
            check("ties       Prop 1(b) E[FR_t] = trap term + distraction term - Joint log loss", np.abs(FR - (ET + ED - Lam)).max())
            check("ties       Prop 1(b) trap term never decreases", max(0.0, -np.diff(ET).min()))
            check("ties       Prop 1(b) a decrease of E[FR] is at most the decrease of the distraction term", max(0.0, ((FR[:, None] - FR[None, :]) - (ED[:, None] - ED[None, :]))[np.triu_indices(21, 1)].max()))
            check("ties       Prop 1(c) E[FR_t] = t E[Delta] + N_t - Lambda_t", np.abs(FR - (tt * tb.Delta.mean() + Nt - Lam)).max())
    return n_tie


def chebyshev_decreasing(check, cheb):
    """Proposition 2(b) where the expected curve DECREASES (mu_t < 0): the trap-free table of the figure after its peak, exact law of
    one episode's increment; the average of N episodes fails to decrease with probability at most sigma^2 / (N mu^2)."""
    ell = toy_tables()[0]; ts = (120, 150, 200, 300, 400, 600, 800); tb = Table(ell, None, steps=sorted(set(ts) | {t + 1 for t in ts})); p = ell[0]
    for t in ts:
        s0, s1 = tb.steps[t], tb.steps[t + 1]; Pn = s0["P"]
        x = np.r_[s1["FR"][1:] - s0["FR"], s1["FR"][:-1] - s0["FR"]]; w = np.r_[(Pn * p[None]).mean((1, 2)), (Pn * (1 - p[None])).mean((1, 2))]
        mu = w @ x; var = w @ x ** 2 - mu ** 2; assert mu < 0
        for N in (10 ** 4, 3 * 10 ** 4, 10 ** 5, 3 * 10 ** 5, 10 ** 6):
            if var / (N * mu ** 2) >= 1: continue
            cheb[False] += 1; check("Prop 2(b)  mu_t < 0: P(average does not decrease) <= sigma^2 / (N mu^2)", max(0.0, 1 - dip_exact(x, w, N) - var / (N * mu ** 2) - 1e-6))


def main():
    n_tab = int(sys.argv[sys.argv.index("--tables") + 1]) if "--tables" in sys.argv else 300
    rng = np.random.default_rng(2026); worst = {}; cheb = {True: 0, False: 0}
    def check(name, violation): worst[name] = max(worst.get(name, 0.0), float(violation))
    for it in range(n_tab):
        d = 1 + it % 2; R = 2 + (it // 2) % 2; Rc = 2 + (it // 4) % 2; T = 60 if d == 1 else 24
        tb = Table(random_table(rng, d, R, Rc), T); tt = np.arange(T + 1)
        FR, Lam, LamN, Nt, ET, ED = (tb.series(k) for k in ("EFR", "Lam", "LamN", "Ntil", "ET", "ED"))
        Syn, LamC, accJ, accN, sq = (tb.series(k) for k in ("Syn", "LamC", "accJ", "accN", "sqrt"))
        ED_ = tb.Delta.mean()
        # ---- Proposition 1
        check("Prop 1(a)  E[FR_t] = Naive log loss - Joint log loss", np.abs(FR - (LamN - Lam)).max())
        # information gain of the Joint observer at step t + 1 = expected KL between its consecutive beliefs, I(r_g; o_{t+1} | o_{1:t})
        for t in (0, 1, 2, 5, 12):
            s0, s1 = tb.steps[t], tb.steps[t + 1]; idx = {tuple(c): i for i, c in enumerate(s1["counts"])}; g = 0.0
            for mi, c in enumerate(s0["counts"]):
                for o in itertools.product((0, 1), repeat=tb.d):
                    po_r = np.prod(np.where(np.array(o)[:, None, None] == 1, tb.ell, 1 - tb.ell), axis=0)    # P(o | r)
                    pj = (s0["P"][mi] * po_r).sum() / (tb.R * tb.Rc); b1 = s1["B"][idx[tuple(np.array(c) + o)]]
                    g += pj * np.sum(xlogy(b1, b1 / s0["B"][mi]))
            check("Prop 1(a)  increment: Joint information gain - fall of the Naive log loss", abs((FR[t + 1] - FR[t]) - (g - (LamN[t] - LamN[t + 1]))))
        check("Prop 1(b)  E[FR_t] = trap term + distraction term - Joint log loss", np.abs(FR - (ET + ED - Lam)).max())
        check("Prop 1(b)  trap term never decreases", max(0.0, -np.diff(ET).min()))
        check("Prop 1(b)  Joint log loss never increases", max(0.0, np.diff(Lam).max()))
        drop = FR[:, None] - FR[None, :]; cap = ED[:, None] - ED[None, :]; iu = np.triu_indices(T + 1, 1)
        check("Prop 1(b)  a decrease of E[FR] is at most the decrease of the distraction term", max(0.0, (drop - cap)[iu].max()))
        d0 = np.log1p(tb.is_dist.sum(2) / (1 + tb.is_trap.sum(2))).mean()
        check("Prop 1(b)  D_0 = ln(1 + #Dist / (1 + #Trap)) <= ln R", max(abs(ED[0] - d0), d0 - np.log(tb.R)))
        check("Prop 1(c)  E[FR_t] = t E[Delta] + N_t - Lambda_t", np.abs(FR - (tt * ED_ + Nt - Lam)).max())
        ii = [interaction_information(e) for e in tb.ell]
        check("slope      E[Delta] = sum_i [I(o;r_c) - I(r_g;r_c;o)] - E[min kappa]", abs(ED_ - (sum(i[1] - i[0] for i in ii) - tb.kap.min(2).mean())))
        check("slope      I(o;r_c|r_g) = I(o;r_c) - I(r_g;r_c;o)", max(abs(i[2] - (i[1] - i[0])) for i in ii))
        # ---- Proposition 2: the step X at each joint realization (mean Delta, variance v)
        lm, l1m = np.log(tb.m / (1 - tb.m)), np.log1p(-tb.m)
        for rg in range(tb.R):
            for rc in range(tb.Rc):
                a = tb.ahat[rg, rc]; th = lm[:, a] - lm[:, rg]; c0 = (l1m[:, a] - l1m[:, rg]).sum(); p = tb.ell[:, rg, rc]
                outs = np.array(list(itertools.product((0, 1), repeat=tb.d))); x = outs @ th + c0; w = np.prod(np.where(outs == 1, p, 1 - p), axis=1)
                check("Prop 2     the step X has mean Delta(r)", abs(w @ x - tb.Delta[rg, rc]))
                check("Prop 2     the step X has variance v(r)", abs(w @ x ** 2 - (w @ x) ** 2 - np.sum(p * (1 - p) * th ** 2)))
        # ---- Corollary 1: the mixture (1 - q') U + q' W decreases at step t exactly when q' < q*_t
        if 0 < tb.k < tb.R * tb.Rc:
            U, W = tb.series("U"), tb.series("W"); u, w_ = U[:-1] - U[1:], W[1:] - W[:-1]
            check("Cor 1      E[FR_t] = (1 - q) U_t + q W_t", np.abs(FR - ((1 - tb.q) * U + tb.q * W)).max())
            # every step at which W does not decrease, with the conventions of the corollary (0/0 = 0; u > 0 = w gives q*_t = 1);
            # steps where both components are flat to rounding (the first step with one observable) are exact zeros
            u = np.where(np.abs(u) < 1e-13, 0.0, u); w_ = np.where(np.abs(w_) < 1e-13, 0.0, w_); ok = w_ >= 0; up = np.maximum(u, 0)
            qs = np.where(up > 0, up / np.where(up + w_ > 0, up + w_, 1.0), 0.0)
            for qq in (0.0, 0.01, 0.05, 0.2, 0.5, 0.9):
                dec = (1 - qq) * u > qq * w_ + 1e-13; pred = qq < qs - 1e-12; amb = np.abs(qq - qs) < 1e-9
                check("Cor 1      the mixture decreases at step t exactly when q < q*_t (all steps with W non-decreasing)", float(np.any((dec != pred) & ok & ~amb)))
                check("Cor 1      criterion (1 - q) u_t > q w_t = decrease of the mixture", float(np.any((np.diff((1 - qq) * U + qq * W) < -1e-12) != ((1 - qq) * (U[:-1] - U[1:]) > qq * (W[1:] - W[:-1]) + 1e-12))))
        # ---- Corollary 2 / Proposition 2(b): Chebyshev bounds on the dip of an N-episode average (d_o = 1: exact law of the increment)
        if tb.d == 1 and it % 6 == 0:
            for t in (3, 10, 30):
                s0, s1 = tb.steps[t], tb.steps[t + 1]; p = tb.ell[0]; Pn = s0["P"]                    # P[n, rg, rc]
                x = np.r_[s1["FR"][1:] - s0["FR"], s1["FR"][:-1] - s0["FR"]]; w = np.r_[(Pn * p[None]).mean((1, 2)), (Pn * (1 - p[None])).mean((1, 2))]
                mu = w @ x; var = w @ x ** 2 - mu ** 2
                check("Prop 2(b)  mean increment = E[FR_{t+1}] - E[FR_t]", abs(mu - (FR[t + 1] - FR[t])))
                for N in (10, 100, 1000, 10_000):
                    if abs(mu) < 1e-9 or var < 1e-14 or var / (N * mu ** 2) >= 1: continue          # a bound >= 1 says nothing
                    pd = dip_exact(x, w, N); bound = var / (N * mu ** 2); cheb[mu > 0] += 1
                    check("Prop 2(b)  mu_t > 0: P(average decreases) <= sigma^2 / (N mu^2)" if mu > 0 else "Prop 2(b)  mu_t < 0: P(average does not decrease) <= sigma^2 / (N mu^2)",
                          max(0.0, (pd if mu > 0 else 1 - pd) - bound - 1e-6))
        # ---- Propositions 3 and 4
        H0 = np.log(tb.R); IJ, IN, IC = H0 - Lam, H0 - LamN, H0 - LamC
        check("Prop 3     I_Naive <= I_Joint <= I_Context", max(0.0, (IN - IJ).max(), (IJ - IC).max()))
        check("Prop 3     Syn_t = I_Context - I_Joint = I(r_g; r_c | o)", np.abs(Syn - (IC - IJ)).max())
        # 1e-7 allowed below: the square root turns the rounding error of an FR that is exactly 0 (about 1e-16) into 1e-8
        check("Prop 4     accuracy gap <= E[sqrt(2 FR)] <= sqrt(2 E[FR])", max(0.0, (accJ - accN - sq).max() - 1e-7, (sq - np.sqrt(2 * np.maximum(FR, 0))).max() - 1e-7))
        ceil_ = tt * sum(i[2] for i in ii); TC = ceil_ - tb.series("I_xc_g")              # TC(x_{1:n} | r_g) = sum_j I(x_j; r_c | r_g) - I(x_{1:n}; r_c | r_g)
        check("ceiling    E[FR_t] = TC - KL(P || Ptilde)  (the paper's decomposition, over atoms)", np.abs(FR - (TC - tb.series("KLoff"))).max())
        check("ceiling    E[FR_t] <= TC <= t sum_i I(o^i; r_c | r_g)", max(0.0, (FR - TC).max(), (TC - ceil_).max()))
        if tb.d == 1: check("atoms      one binary observable: FR_1 = 0", abs(FR[1]))
        # ---- Proposition 5
        for kind in ("goal_only", "context_only"):
            t5 = Table(random_table(rng, d, R, Rc, kind), 8)
            check("Prop 5(a)  no Interaction Information => FR_t = 0 for every observation sequence", max(np.abs(s["FR"]).max() for s in t5.steps.values()))
            check("Prop 5(a)  table depending on one variable alone => Interaction Information = 0", max(abs(interaction_information(e)[0]) for e in t5.ell))
        check("Prop 5(a)  table depending on both variables => Interaction Information < 0", max(0.0, max(interaction_information(e)[0] for e in tb.ell) + 1e-12))
        t5 = Table(random_table(rng, d, R, R, "cyclic"), 8)
        check("Prop 5(b)  cyclic table: FR_t = 0 for every observation sequence", max(np.abs(s["FR"]).max() for s in t5.steps.values()))
        check("Prop 5(b)  cyclic table: Interaction Information < 0 and Syn_t > 0", max(0.0, max(interaction_information(e)[0] for e in t5.ell) + 1e-12, 1e-12 - t5.series("Syn")[1:].min()))
    n_tie = ties(check, rng); chebyshev_decreasing(check, cheb)
    print(f"{n_tab} random tables (d_o = 1, 2; R, Rc = 2, 3) and 41 tables with ties ({n_tie} exact ties), exact sums.  Largest violation per claim (pass: <= {TOL:g}):\n")
    bad = 0
    for name, v in worst.items():
        ok = v <= TOL; bad += not ok; print(f"  {'pass' if ok else 'FAIL'}  {v:9.2e}   {name}")
    print(f"\n  Chebyshev bound of Prop 2(b): {cheb[True]} informative cases with mu_t > 0 and {cheb[False]} with mu_t < 0 (bound < 1) were tested.")
    bad += limits()
    print("\nALL CLAIMS PASS" if not bad else f"\n{bad} CLAIM(S) FAIL"); sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
