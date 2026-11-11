"""learnability_section.tex, Lemma 1(ii): the check behind "In the Cognitive Gridworld the condition fails" (11-10; run from the
repository root with the eureka python, ~3 min).
Does the Joint goal belief separate into per-step terms?  log B_t(r_g) = sum_s h_s(o_s, r_g) + c(o_{1:t}) for all sequences
holds iff the log-likelihood ratios between goal realizations separate over steps.  For t = 2 that is exact: the matrix
L[o_1, o_2] = ln P(o_1, o_2 | r_g) - ln P(o_1, o_2 | r'_g) must be additive (row term + column term).  Double-centring removes
the additive part; what remains is the non-separable part.  Reported: its share of L's variance (uniform over the 32 x 32
observation pairs), over all goal pairs and sampled interaction sets at the paper's likelihood temperature.  The same test on
the Naive likelihood must give exactly 0 (it separates by construction)."""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["MPLBACKEND"] = "Agg"
import fr_theory as ft

rng = np.random.default_rng(20261110)
lam, nZ = 2.0, 100
obs = np.array(list(itertools.product([0, 1], repeat=5)), float)          # (32, 5)


def step_lik(ell):
    """P(o | r_g, r_c) for every observation: (32, R, R_c)"""
    return np.exp(obs @ np.log(ell).reshape(5, -1) + (1 - obs) @ np.log1p(-ell).reshape(5, -1)).reshape(32, *ell.shape[1:])


def nonsep_share(P2):
    """P2[o1, o2, r_g]: two-step likelihood; the non-separable share of the log-ratio, per goal pair"""
    L = np.log(P2)
    out = []
    for a, b in itertools.combinations(range(P2.shape[-1]), 2):
        D = L[:, :, a] - L[:, :, b]
        I = D - D.mean(1, keepdims=True) - D.mean(0, keepdims=True) + D.mean()
        out.append(I.var() / D.var())
    return np.array(out)


shares, shares_naive = [], []
for k in range(nZ):
    ell = ft.make_env(rng, lam)                                               # (5, R, R_c), goal on axis 1
    p = step_lik(ell)                                                          # (32, R, R_c)
    P2 = np.einsum("irc,jrc->ijr", p, p) / ell.shape[2]                        # (32, 32, R): sum over r_c of the product
    shares.append(nonsep_share(P2))
    m = ell.mean(2)                                                            # Naive marginal likelihood m^i(r_g)
    pn = np.exp(obs @ np.log(m) + (1 - obs) @ np.log1p(-m))                    # (32, R)
    shares_naive.append(nonsep_share(pn[:, None, :] * pn[None, :, :]))
S, SN = np.array(shares), np.array(shares_naive)
print(f"Joint, lambda={lam}, {nZ} interaction sets x {S.shape[1]} goal pairs: non-separable share of the two-step log-ratio "
      f"min {S.min():.3f}  median {np.median(S):.3f}  max {S.max():.3f}; sets with min share < 1e-6: {(S.min(1) < 1e-6).sum()}")
print(f"Naive (control): max {SN.max():.2e}")
