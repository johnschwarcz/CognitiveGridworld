"""The rule M u >= d_E (zscale_law.py) for goal-only training (truth1 runs of train_bayes_generator.py; John 11-15: "do not
leave anything unsettled"). No network, nothing fitted.

In goal-only training the Generator sees the goal's realization alone, so a pair gives 10 marginal cells
c[r_goal] = mean over the other variable's realization of the likelihood. A held-out variable h is never the goal in
training (env_generators.default_context) and is always slot 0 (its index is the smaller one), so its two embeddings are
read differently:
  first   Q_h, from the goal's scalar z10 = <K_goal, Q_h>, by one straight line through the cells
  second  K_h, from z01 = <K_h, Q_goal>, by a straight line whose weights may depend on z10, which the first step has made
          known. Novel pairs need the goal's K, so this step is necessary.
Each step has a noise ratio rho (zscale_proxy.pooling's read-out: the pseudo-inverse of the cells' slopes), a usable share
u = 1 / (1 + rho) and a need d_E / u; the larger need binds. Result (11-15) against the truth1 runs (seed 0, 8,000 episodes):
  d_E 30   need 102 partners, |S| >= 113   runs: chance up to |S| 100, transfer from 125
  d_E 90   need 149 partners, |S| >= 165   runs: .19-.23 up to |S| 125, .38 from 250 (150 and 200 were not run)
  d_E 10   need far above 1000             runs: chance at every |S|
The reading in turn was worked out after the runs. With one straight line for both embeddings the second need is 268
partners at d_E 30, which the runs contradict.

    python zscale_goal_only.py            # about 3 minutes on 4 cores
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zscale_proxy as zp

WORLDS = ((30, 30), (90, 90), (10, 10))                      # (d_E, 1 / z0^2): the truth1 runs are original worlds
N, BINS = 600_000, 150


def ratio(F, x):
    """noise ratio of the straight-line read-out of x from the cells F (pseudo-inverse of the cells' slopes on x)"""
    Fc, xc = F - F.mean(0), x - x.mean()
    b = Fc.T @ xc / (xc @ xc)
    return ((Fc - np.outer(xc, b)) @ (b / (b @ b))).var() / xc.var()


def ratio_given(F, x, known):
    """the same with weights that may depend on `known` (quantile bins); the bins are pooled by their precision"""
    order = np.argsort(known)
    return 1 / np.mean([1 / ratio(F[i], x[i]) for i in np.array_split(order, BINS)])


def first_size(need):
    return next(S for S in range(2, 100_000) if S - S // 10 >= need)


def main():
    zp.rng = np.random.default_rng(2026)
    for d, ref in WORLDS:
        scale = np.sqrt(d / ref)
        z01, z10 = zp.sphere(d, N) * scale, zp.sphere(d, N) * scale
        P = 1 / (1 + np.exp(-zp.V(z01)[:, :, None] * zp.V(z10)[:, None, :]))        # [slot-0, slot-1 realization]
        cells = P.mean(1)                                                             # goal = slot 1: 10 marginal cells
        first, second, single = ratio(cells, z10), ratio_given(cells, z01, z10), ratio(cells, z01)
        need = d * (1 + max(first, second))
        print(f"d_E {d}, z0^2 = 1/{ref}: first step u {1 / (1 + first):.2f} (need {d * (1 + first):.0f}), second step u "
              f"{1 / (1 + second):.2f} (need {d * (1 + second):.0f}) -> {int(np.ceil(need))} partners, |S| >= {first_size(need)}"
              f"   [second embedding by one straight line: need {d * (1 + single):.0f}]", flush=True)


if __name__ == "__main__":
    main()
