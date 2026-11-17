#!/usr/bin/env python3
"""fr_theory_horizon.py -- how P(an average of N episodes dips within T steps) scales with the horizon T (fig_fr_theory_all p).

John 11-11: "Is this apparent step at t = 1000 because of the ratio of t/N? ... does P(dip within t) -> 0 as t/N -> 0? Is there a
phase transition here?"  Theory only (tables alone): 1 - E_rho[(1 - p_rho)^T] (fr_theory.monotone_theory, 1,500 draws of rho per N) on
the population of panel p (2,000 interaction sets, lambda = 2, q = .474); ~1 min on 10 cores.

RESULTS (11-11)
  T        N at P = .9 / .5 / .1    width (.9 -> .1)
  10            15 /  67 /  173         159
  100           99 / 182 /  303         204
  1,000        202 / 303 /  443         241
  10^4         311 / 433 /  582         271
  10^5         434 / 562 /  722         289
  10^6         553 / 691 /  864         311
  10^7         670 / 821 / 1002         332
  10^8         785 / 952 / 1140         355
  The midpoint grows like ln T, NOT like T / N: dN_1/2 / d ln T = 50-57 episodes (-> 1/I_q = 58.5, I_q = .0171 the large-deviation
  rate of one step for a typical group; annealed rate I_a = .0154 is the one in the guarantee P <= T exp(-N I_a), Chernoff + union
  bound).  So P -> 0 as soon as N - ln(T) / I -> infinity (in particular as T / N -> 0).  In the variable N / ln T the drop becomes a
  step at 1 / I (a sharp threshold), but slowly: the width grows like sqrt(ln T) (composition of the groups) while the location
  grows like ln T, relative width 2.4 (T = 10) -> 0.37 (T = 10^8).  The step look of T = 1,000 in the panel is mostly the log axis.
"""

import os, sys, pickle, time, numpy as np
sys.path[:0] = ["/home/johns/Desktop/CognitiveGridworld", "/home/johns/Desktop/CognitiveGridworld/main/REPLICATION/claude_plots"]
os.environ["MPLBACKEND"] = "Agg"
import fr_theory as F
from scipy.optimize import minimize_scalar
with open("/home/johns/Desktop/CognitiveGridworld/main/DATA/experimental/fr_theory/fr_theory_cache/population.pkl", "rb") as fh: _, pop = pickle.load(fh)
pool = pop["pool"]; tr = pool["trapped"]; P, TH, C = pool["p"][tr].astype(float), pool["theta"][tr].astype(float), pool["const"][tr].astype(float)
q = tr.mean(); print(f"population: {tr.size} joint realizations, q = {q:.4f}")
def lnM(s):                                   # ln E[exp(-s X) | r] for every trapped r (X = C + sum_i theta_i o_i, o_i ~ Bernoulli(p_i))
    return -s * C + np.log1p(P * np.expm1(-s * TH)).sum(1)
fq = lambda s: q * lnM(s).mean()                                  # quenched: E_r ln M_r(s)   (untrapped r: ln M = 0)
fa = lambda s: np.log((1 - q) + q * np.exp(lnM(s)).mean())        # annealed: ln E_r M_r(s)
for name, f in (("quenched", fq), ("annealed", fa)):
    r = minimize_scalar(f, bounds=(0, 20), method="bounded"); print(f"  {name} Chernoff rate I = {-r.fun:.5f} at s = {r.x:.3f}")
Iq = -minimize_scalar(fq, bounds=(0, 20), method="bounded").fun; Ia = -minimize_scalar(fa, bounds=(0, 20), method="bounded").fun
x, w, qq = F.step_atoms(pool); mu, sd = F.step_moments(x, w); print(f"  mean step {mu:.4f}, s.d. {sd:.4f}, SNR {mu/sd:.4f}, SNR^2/2 = {(mu/sd)**2/2:.5f}")
Ts = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10**4, 10**5, 10**6, 10**7, 10**8]   # 11-11 night: the T of the
                                                                  # simulated markers of w (fr_theory_dipsim.py) too; the rest unchanged
Ns = np.unique(np.round(np.logspace(np.log10(5), np.log10(8000), 70)).astype(int))
t0 = time.time(); m, se = F.monotone_theory(pool, Ns, Ts, 1500, 9191, workers=int(os.environ.get("FR_WORKERS", 10)))
print(f"theory computed in {time.time() - t0:.0f} s"); Pd = 1 - m
np.savez("/home/johns/Desktop/CognitiveGridworld/main/DATA/experimental/fr_theory/additions/horizon_scaling.npz", Ns=Ns, Ts=Ts, Pd=Pd, se=se, Iq=Iq, Ia=Ia, mu=mu, sd=sd, q=q)
def cross(p):
    out = []
    for j in range(len(Ts)):
        y = Pd[:, j]; k = np.flatnonzero((y[:-1] >= p) & (y[1:] < p))
        out.append(np.nan if not len(k) else float(np.exp(np.interp(p, [y[k[-1] + 1], y[k[-1]]], [np.log(Ns[k[-1] + 1]), np.log(Ns[k[-1]])]))))
    return np.array(out)
h, a, b = cross(0.5), cross(0.9), cross(0.1)
print(" T        N(P=.9)  N(P=.5)  N(P=.1)   width(.9->.1)  N_half*Iq/ln(T)   dN_half/dlnT")
for j, T in enumerate(Ts):
    slope = (h[j] - h[j - 1]) / np.log(10) if j else np.nan
    print(f" {T:<8g} {a[j]:8.0f} {h[j]:8.0f} {b[j]:8.0f}   {b[j] - a[j]:8.0f}        {h[j] * Iq / np.log(T) if T > 1 else np.nan:6.3f}           {slope:7.1f}")
print(f" 1/Iq = {1/Iq:.1f}, 1/Ia = {1/Ia:.1f}  (predicted dN_half/dlnT -> 1/I)")
