"""Pairs of marginal beliefs given the same observation vector (John 11-02): does the distance between them grow, and
how does that depend on the observation's entanglement, on the other variable's distance and on factorization regret?

John 11-02 (the first figures): "look at the divergence of similar marginal beliefs given the same observation ...
marginal belief divergence before an observation on the x-axes (20 bins), and the distribution of divergence after the
observation on the y-axes (a violin plot), using pairs of pre-observation beliefs where the observation (the full
vector) is identical ... for all 4 agents" (Bayes, naive, Trained RNN, ESN), "... split by Factorization Regret".
Round 3: "the chaos pairs is nice, can you please split it from 20 to 40 bins. Also do you do this for both variable
marginals seperately or use the average distance? It would be nice to try different ways of doing it. It would be
interesting to see if this changes when both variable marganals are close versus only one. Also instead of splitting
by fr, can you please split by the entanglement of the observation. And let's combine the chaos pairs and chaos pairs fr
(now entanglement) into 1 figure (for the entanglement split we can just show different averages as curves on a single
panel per agent)." This was measure M4 of td_tt_chaos.py, moved here on 11-02; the theory (T1-T3) and the Lyapunov
analysis stay in td_tt_chaos.py.

STATUS (11-02, round 4): done at the end of training (one checkpoint per network); corrected after three adversarial
checks (CORRECTIONS).
-> main/REPLICATION/figure_bin/td_tt_chaos_pairs_detail.png (command figure; the first figure, 20 bins: d_H after
   against before; its change, with the shuffle; the change of the symmetric KL, with the common-mode update and naive)
-> through_training/chaos/pairs_combined.npz (command combined): the data of John's combined figure (40 bins; three
   definitions of the distance; the split by the observation's entanglement); layout and drawing notes under COMBINED
   CACHE. The combined figure, figure_bin/td_tt_chaos_pairs.png, is drawn by td_tt_chaos_combined.py (with
   td_tt_chaos.py's sweep panel). The FR figure (td_tt_chaos_pairs_fr.png) and its command fr were dropped on 11-02
   (John: split by the observation's entanglement instead of FR); numbers() still prints its panel texts.
   td_tt_chaos.py's M4 code was removed (11-02).

DEFINITIONS
  Pairs: every two probes of the same environment and step t (2-30) with the identical 5-observation vector at t and
    different observation counts before t (Bayes's and naive's beliefs depend on the observations only through the
    counts, so theirs differ too). A unit is one variable of one pair-step. The pairs are selected for a shared
    observation, not for close beliefs, and most are far apart (RESULTS).
  d_H(p, q) = max_r log(p_r / q_r) - min_r log(p_r / q_r), per variable, between the pair's marginal beliefs over the 10
    values: the range of the log-ratio (Hilbert's projective metric). Before = after step t - 1, after = after step t,
    Delta d_H = after - before. Multiplying both beliefs by the same vector leaves d_H unchanged, and naive's update
    for a shared observation is such a vector (td_tt_chaos.py T1). d_H is set by the two values with the extreme
    log-ratios, which can be values that both beliefs treat as near-impossible.
  Bulk divergences (values weighted by their probability): SKL(p, q) = [KL(p||q) + KL(q||p)] / 2 = Sigma_r (p_r - q_r)
    log(p_r / q_r) / 2 (FR's convention); JS(p, q) = [KL(p||m) + KL(q||m)] / 2 with m = (p + q) / 2 (at most log 2).
  Common mode: both log beliefs shifted by the pair's mean log-update (Delta log p + Delta log q) / 2, then
    renormalised. This update changes both beliefs alike and leaves d_H unchanged. "Beyond the common mode" = after
    minus this counterfactual's after. Naive's update is its own common mode. "Beyond naive" = the agent's change minus
    naive's change on the same pair-steps.
  Shuffled (the control for d_H): delta = Delta log p - Delta log q (per value) is the step's difference. It is taken
    from another unit of the same group, with a random orientation (sign) per unit: osc(D_before + sign x
    delta_permuted); per unit, the mean of 5 seeds. Groups of the first figures: variable, FR third and bin of d_H
    before (units outside the 20 bins form two bins of their own). This is what the change would be if the step's
    difference were unrelated to the existing difference D_before. osc is convex and the randomly oriented delta has
    mean 0, so by Jensen the shuffled change is >= 0 on average: it is what the step's difference does through its size
    alone. The actual change beyond it is close to the part aligned with D_before (the sign flip below; RESULTS: +.483 /
    +.265 / +.975 vs +.483 / +.266 / +.988). The donors come from all environments, so for small subsets the shuffle is
    noisy (RESULTS, the closest pairs).
  Sign flip (no permutation, no seed): each unit's change splits exactly into a symmetric part, (osc(D + delta) +
    osc(D - delta)) / 2 - osc(D) >= 0 (convexity; the expected change if the unit's own delta had a random orientation),
    and an aligned part, (osc(D + delta) - osc(D - delta)) / 2.
  Product of marginals (Bayes): each probe's joint belief before step t is replaced by the product of its own
    marginals (same marginals, no coupling between the variables). It is updated with the same observation's joint
    likelihood (one step of factorized-state Bayes), and its marginals give d_H after. Joint d_H: between the pair's
    joint posteriors over the 100 cells. The shared observation conserves it (T1), and the marginal d_H never exceeds
    it (T2).
  FR: the pair's factorization regret before the observation, per variable: [KL(Bayes||naive) + KL(naive||Bayes)] / 2
    of that variable's marginals, the mean of the two probes. Thirds are pooled over all units, the same for every
    agent.
  Matched on step and distance before (the matched estimator, matched()): cells are (step t, bin of the agent's own d_H
    before), using the figure's bins and the units inside them; only cells with at least 20 units in every group (FR
    third, entanglement fifth, ...) are used. Per cell, the group's mean; cells are averaged with their unit counts as
    weights (direct standardisation). A cell missing a group in a resample is left out of that resample. "Raw" = the
    groups' means without matching. Each agent's contrast is standardised to its own (step, bin) units; "Trained RNN -
    Bayes on one population" uses the cells usable for both, each weighted by both agents' counts (common_hilo()).
  Intervals: [5-95%] over the 1,000 shared environment resamples (td_tt_gap.boot_counts; ratios of count-weighted
    environment sums; a pair's environment = its probes'). Each architecture has one trained network (seed 20261029), so
    the intervals do not include variation across training seeds.
  Combined data (John, round 3):
  40 bins: log-spaced over the pooled (four agents, all units) 0.1st-99th percentiles of d_H before (0.250-118.9 nats,
    each bin x1.167). The pooled minimum (0.006 nats, naive's) would leave the lowest 10 / 9 / 8 of 40 bins empty for
    Bayes / the Trained RNN / the ESN; with the 0.1st percentile every bin holds >= 130 units of every agent (fewest:
    Bayes 130, naive 2,105, Trained RNN 189, ESN 930). Outside the 40 bins (below / above): Bayes 102 / 31,116, naive
    10,811 / 22,168, Trained RNN 289 / 431, ESN 2,480 / 83,142.
  Three definitions of the distance:
    (i) per variable (as in the first figures): a unit = one variable of a pair-step, x = its d_H before.
    (ii) the pair's average: a unit = a pair-step, x = the mean of its two variables' d_H before, y = the mean of their
      changes (and of every other measure). It is exactly half the d_H between the pair's products of marginals over
      the 100 cells (the range of f(r0) + g(r1) is the sum of the two ranges; check). A small x means both variables
      are close. The mean over all pair-steps equals the per-variable mean over all units; only the binning differs.
    (iii) both close vs only one close: for a unit (pair-step, v), the other variable's d_H before, cut into thirds
      within each cell (step t, bin of the unit's own d_H before): the cell's values ranked and cut into three parts
      whose sizes differ by at most one unit. Where the own variable is close, the lower third is "both close" (median
      other distance 0.75-1.3 nats at own 0.5-0.9 nats) and the upper third "only this one close" (4.6-6.7 nats). The
      pooled (absolute) thirds first proposed (10.6 and 27.3 nats, over the four agents' units) cannot show this: the
      two distances are correlated, so their upper third is empty where the own variable is close (bins with >= 50
      upper-third units start at 2.3 / 1.7 / 0.8 nats for Bayes / Trained RNN / ESN). The within-cell thirds are
      populated in every cell and are matched on step and own distance by construction (their raw per-bin means are
      their step-standardised lines).
  Entanglement of the observation (whole belief): probes.npz 'ent' at step t (td_analysis: per variable, the symmetric
    KL between naive's step likelihood and Bayes's effective step update jb_t / jb_{t-1}, each normalised over the 10
    values; the locked definition), the mean of the two variables, then of the two probes: one value per pair-step,
    used for both of its units. Fifths pooled over the pair-steps (edges .006 / .051 / .103 / .177 / .313 / 2.51), the
    same for every agent. It is computed from Bayes's beliefs, so for Bayes the dependence is partly built in: if both
    probes' entanglement were 0, Bayes's step update would equal naive's and its d_H could not change (T1). Controls:
    thirds (edges .006 / .084 / .211 / 2.51); the unit's own variable's entanglement (mean of the two probes); the other
    variable's; matched also on the FR third, on the within-cell third of the other variable's distance, or on step band
    (2-10, 11-20, 21-30), bin and environment.
  Step-standardised lines (the entanglement fifths per bin): per bin, the cells (step t, bin) with >= 20 units in every
    fifth; each cell's fifth means weighted by the cell's unit count; a line is drawn where its used cells hold >= 500
    units. Averaged over the bins with these weights, they reproduce the matched fifths' means (check). The raw per-bin
    means are confounded by step: in every bin centred 3.18-32.08 nats (16 bins) the ESN's raw fifths differ by +.12 to
    +.34 within a bin, its step-standardised ones by +.02 to +.06.
  Shuffles of the combined data (5 seeds each): the donor is another unit of the same variable, entanglement fifth,
    within-cell third of the other variable's distance and bin of 40 (out-of-range units in two bins of their own); for
    the pooled thirds the pooled third replaces the within-cell third; for the pair average the donor is another
    pair-step of the same entanglement fifth and bin of the pair average (one donor per variable). Each split's shuffle
    thus draws donors from its own groups.
RESULTS (numbers(); end of training; 1,710,703 pair-steps x 2 variables = 3,421,406 units per agent; all units unless
  stated; [5-95%]; values in the order Bayes / Trained RNN / ESN unless named)
  Pairs and distances: 13,321 same-observation pairs are left out for equal counts; 610,261 distinct probe pairs.
    d_H before, median (quartiles): Bayes 21.9 nats (10.6-40.5), naive 13.6 (6.0-28.7), Trained 16.7 (8.4-30.9), ESN
    17.8 (7.5-38.3). Minimum .077 / .006 / .067 / .049 nats, share below 1 nat .002 / .020 / .004 / .015 and above 10
    nats .766 / .602 / .695 / .677 (Bayes / naive / Trained / ESN). Mean d_H before by FR third: Bayes 21.5 / 29.4 /
    37.3, naive 18.2 / 21.7 / 24.6, Trained 16.6 / 22.2 / 28.0, ESN 23.7 / 29.3 / 33.3. Share set by near-impossible
    values (at both extremes of the log-ratio the larger belief is < 1e-3): Bayes .174, naive .080, Trained .100, ESN
    .135. FR thirds 0 / .219 / .869 / 34.3 nats (1,140,469 / 1,140,468 / 1,140,469 units); mean step per third 12.7 /
    16.3 / 19.4.
  Checks:
    - float64 Bayes / naive vs the stored float32: 7.6e-6.
    - The pair set and the networks' forwards reproduce the old pairs.npz exactly (max |d| 0).
    - Naive's d_H after - before: at most 8.5e-14 (T1).
    - The common mode keeps d_H to 8.5e-14.
    - The product-of-marginals code, run with the true joint, reproduces Bayes's d_H after to 2.3e-13.
  The range of the log-ratio grows (Delta d_H). Share of units where it grows (shrinks) and mean change:
    Bayes .706 [.695-.718] (.293), +.604 [+.581, +.628] nats; Trained .606 [.594-.618] (.394), +.347 [+.316, +.379];
    ESN .839 [.836-.842] (.161), +1.005 [+.977, +1.036]; naive 0.
  Beyond the shuffle:
    - Shuffled change +.120 [+.116, +.124] / +.082 [+.080, +.084] / +.030 [+.028, +.031] (seed means +.120 to +.121 /
      +.081 to +.083 / +.028 to +.031).
    - Actual - shuffled +.483 [+.457, +.511] / +.265 [+.232, +.298] / +.975 [+.947, +1.006].
    - By FR third (raw): shuffled Bayes +.145 / +.117 / +.100, Trained +.106 / +.075 / +.065, ESN +.039 / +.027 / +.023;
      actual - shuffled Bayes +.373 / +.488 / +.589, Trained +.064 [+.030, +.096] / +.226 / +.506, ESN +1.025 / +.980 /
      +.921.
    - Other shuffle groups (bin; env x bin; step x bin; env x step x bin; env x observation x bin; one seed each): Bayes
      +.482 to +.493, Trained +.265 to +.270, ESN +.974 to +.982.
    - Sign flip: aligned part +.483 [+.470, +.496] / +.266 [+.235, +.296] / +.988 [+.961, +1.018]; symmetric part
      +.121 [+.110, +.134] / +.081 [+.078, +.085] / +.017 [+.016, +.017]. The aligned part is positive in .678 / .576 /
      .832 of units.
    - The old oriented null (pairs.npz): shuffled +.159 / +.060 / +.249, actual - shuffled +.444 / +.287 / +.756.
    - The closest pairs, d_H before < 1 nat (8,403 / 12,985 / 49,882 units; .0025 / .0038 / .0146 of units; grows in
      .828 / .831 / .849): change +.564 [+.488, +.670] / +.581 [+.509, +.667] / +.273 [+.259, +.287]; aligned part +.051
      [+.039, +.062] / +.028 [+.011, +.046] / +.148 [+.138, +.158] (positive in .585 / .521 / .747 of units); symmetric
      part +.513 [+.434, +.620] / +.553 [+.487, +.634] / +.125 [+.118, +.132]; shuffled change +.517 [+.508, +.527] /
      +.549 [+.540, +.558] / +.140 [+.138, +.143]; actual - shuffled +.047 [-.028, +.147] / +.032 [-.039, +.115] / +.133
      [+.119, +.147].
    - 1-3 nats (112,776 / 154,082 / 255,529 units): change +.409 [+.374, +.453] / +.351 [+.319, +.386] / +.448 [+.430,
      +.466]; aligned +.073 [+.064, +.083] / +.065 [+.053, +.077] / +.380 [+.366, +.396] (positive in .605 / .547 /
      .783); symmetric +.335 / +.286 / +.067; shuffled change +.336 / +.291 / +.096; actual - shuffled +.073 [+.038,
      +.115] / +.060 [+.028, +.095] / +.352 [+.333, +.371].
    - Actual - shuffled = aligned + (symmetric - shuffled change), exactly (to 1.8e-15). The second term averages about
      0 but varies across environments, because the donors come from all environments while a unit's own symmetric
      part is specific to its environment: below 1 nat -.004 [-.080, +.099] / +.004 [-.062, +.085] / -.015 [-.023,
      -.008]; 1-3 nats -.001 [-.036, +.046] / -.005 [-.031, +.027] / -.029 [-.033, -.023]. So the shuffle cannot resolve
      the small aligned parts of Bayes and the Trained RNN at close range; the sign flip can.
    - The Trained RNN's growth depends on the distance. In the lowest FR third, the per-bin mean change falls from
      +.16 (30 nats) to +.11 (42), +.04 (58), -.10 (81) and -.45 (113 nats; 92,823 / 73,041 / 48,267 / 19,540 / 2,058
      units). In the highest third below 3 nats (13,196 units) it equals the shuffle (+.172 vs +.170).
  Bulk divergences (Bayes / Trained / ESN, then naive where shown; SKL before: mean 7.21 / 6.47 / 5.82, naive 4.67):
    - Raw SKL change: -.044 [-.054, -.035] (grows in .368 of units) / -.030 [-.044, -.015] (.400) / +.016 [+.011,
      +.021] (.515); naive -.309 [-.321, -.297] (.110).
    - Raw JS change: -.0053 / -.0048 / -.0002 [-.0004, +.0000]; naive -.0150.
    - Beyond naive: SKL +.264 [+.254, +.274] / +.279 [+.268, +.290] / +.325 [+.313, +.338]; JS +.0097 / +.0101 /
      +.0148.
    - Beyond the agent's own common mode: SKL +.247 [+.241, +.254] / +.206 [+.193, +.220] / +.340 [+.330, +.351] (above
      it in .682 / .675 / .816 of units); JS +.0078 / +.0085 / +.0122. Naive equals its common mode (SKL 1.7e-13).
    - SKL by FR third (raw), beyond naive: Bayes +.170 / +.253 / +.370, Trained +.146 / +.251 / +.441, ESN +.305 /
      +.325 / +.344. Beyond the common mode: +.192 / +.254 / +.295, +.110 / +.188 / +.320, +.324 / +.341 / +.355.
  Bayes's coupling:
    - Product of marginals (no coupling): change +.927 [+.862, +1.003] (grows in .678), by FR third +.535 / +.767 /
      +1.478.
    - Actual - product (the coupling's part): -.323 [-.390, -.265], by third -.018 [-.057, +.021] / -.162 [-.204,
      -.122] / -.790 [-.911, -.678]; r(actual change, product change) over units .257.
    - Where it acts, by d_H before: < 1 nat (.0025 of units) +.091 [+.048, +.135]; < 2 nats (.0139) +.035 [+.007,
      +.066]; 2-10 (.2201) -.195 [-.223, -.168]; 10-50 (.5913) -.286 [-.354, -.227]; >= 50 (.1746) -.641 [-.755, -.534].
      Units whose d_H before is set by near-impossible values (.174): +.291 [+.245, +.336]; the others -.452 [-.534,
      -.379]. Per FR third, near-impossible-set / others: +.458 [+.389, +.536] / -.107 [-.150, -.066]; +.316 [+.265,
      +.370] / -.263 [-.318, -.214]; +.130 [+.056, +.197] / -1.005 [-1.148, -.871].
    - T2: the marginal d_H after never exceeds the joint d_H (max -1.7e-6). But the joint d_H averages 69.9 nats
      (median 52.3), leaving 39.8 nats of headroom on average (median 25.6); .0070 of units are within 1 nat of it.
  FR, highest - lowest third of the change, matched on step and d_H before (raw second):
    - Delta d_H: Bayes +.081 [+.063, +.101] (raw +.171 [+.149, +.191]); Trained +.402 [+.367, +.436] (raw +.401 [+.369,
      +.431]); ESN -.012 [-.023, -.001] (raw -.121 [-.173, -.069]); cells cover .968 of the units.
    - Other bins: with the 1st-99th percentile bins +.079 / +.402 / -.010 [-.021, +.001]; with 40 bins (0.1st-99.9th)
      +.081 / +.402 / -.009 [-.020, +.002].
    - Matched on d_H before only: -.024 / +.331 / -.349.
    - Matched third means: Bayes +.573 / +.606 / +.654, Trained +.133 / +.299 / +.536, ESN +.992 / +.993 / +.980.
    - The confounds: mean d_H before by third (above); mean step 12.7 / 16.3 / 19.4.
    - Actual - shuffled: +.085 [+.067, +.104] / +.397 [+.361, +.431] / -.010 [-.022, +.001] (raw +.216 / +.442 / -.104).
    - SKL beyond naive: +.121 [+.088, +.159] / +.219 [+.186, +.253] / +.084 [+.070, +.098] (matched thirds Bayes +.210
      / +.257 / +.331, Trained +.171 / +.256 / +.390, ESN +.275 / +.320 / +.358; raw +.201 / +.296 / +.040).
    - SKL beyond the common mode: +.036 [+.024, +.049] / +.167 [+.152, +.182] / +.078 [+.069, +.088] (thirds +.233 /
      +.255 / +.269, +.121 / +.188 / +.288, +.296 / +.332 / +.374; raw +.103 / +.210 / +.031).
    - JS beyond naive: +.0062 / +.0067 / +.0034 (thirds +.0064 / +.0098 / +.0126, +.0065 / +.0104 / +.0132, +.0119 /
      +.0144 / +.0153; raw +.0034 / +.0034 / -.0058).
    - JS beyond the common mode: +.0013 [+.0010, +.0018] / +.0028 [+.0023, +.0033] / +.0020 [+.0016, +.0025] (raw
      -.0032 / -.0020 / -.0064).
    - Other matchings, in the order Delta d_H / SKL beyond naive / SKL beyond the common mode / JS beyond naive / JS
      beyond the common mode:
      on step and the agent's own SKL or JS before (20 bins; the four bulk measures only, SKL beyond naive / SKL beyond
        the common mode / JS beyond naive / JS beyond the common mode): Bayes +.127 [+.093, +.164] / +.022 [+.010,
        +.035] / +.0068 / +.0014; Trained +.221 [+.187, +.255] / +.158 [+.141, +.174] / +.0075 / +.0031; ESN +.022
        [+.008, +.036] / +.006 [+.000, +.011] / +.0024 [+.0018, +.0030] / +.0010 [+.0007, +.0013].
      on step band (2-10, 11-20, 21-30), d_H bin and environment (cover .980-.987): Bayes +.057 [+.040, +.076] / +.169 /
        +.037 / +.0062 / -.0001 [-.0006, +.0003]; Trained +.327 [+.305, +.348] / +.231 / +.128 / +.0062 / +.0008
        [+.0003, +.0013]; ESN -.135 [-.146, -.125] / +.045 [+.031, +.058] / +.018 [+.009, +.025] / -.0008 [-.0013,
        -.0003] / -.0021 [-.0024, -.0018].
      on step, d_H bin and environment (at least 5 units per third; cover .917-.934): Bayes +.095 [+.077, +.114] / +.192
        / +.061 / +.0074 / +.0014; Trained +.361 [+.339, +.382] / +.253 / +.145 / +.0075 / +.0025; ESN -.010 [-.018,
        -.000] / +.101 [+.087, +.114] / +.075 [+.067, +.084] / +.0038 [+.0032, +.0043] / +.0019 [+.0016, +.0023].
    - Trained RNN - Bayes. The two agents' units differ (median d_H before 16.7 vs 21.9 nats), so the difference of
      their own-standardised contrasts depends on the population; its sign does not. Delta d_H: own +.321 [+.283,
      +.356]; one population +.393 [+.348, +.434]; Bayes's weights +.451 [+.399, +.498]; environment-matched (band, bin,
      env) +.269 [+.248, +.289], (step, bin, env) +.266 [+.243, +.287]: +.27 to +.45. Actual - shuffled +.311 / +.380 /
      +.439 (own / one population / Bayes's weights). SKL beyond naive +.098 [+.083, +.111] / +.126 / +.151,
      environment-matched +.062 / +.061; SKL beyond the common mode +.131 [+.114, +.146] / +.166 / +.200, +.091 /
      +.085; JS beyond naive +.0005 [+.0003, +.0007] / +.0006 / +.0006, -.0001 [-.0003, +.0002] / +.0001 [-.0002,
      +.0003]; JS beyond the common mode +.0015 [+.0012, +.0017] / +.0015 / +.0015, +.0009 / +.0010.
  Figures: the units inside the 20 bins (0.25-185 nats) are Bayes 3,420,946 (102 below, 358 above), naive 3,409,350
    (10,811 / 1,245), Trained 3,421,117 (289 / 0) and ESN 3,406,843 (2,480 / 12,083); td_tt_chaos_pairs_detail.png
    draws them, and its panel texts use them.
    Panel texts (Bayes / Trained / ESN unless named):
    - d_H before, median: Bayes 21.9, naive 13.7, Trained 16.7, ESN 17.7 nats.
    - Delta d_H: grows .706 / .606 / .838; mean +.604 / +.347 / +.994; shuffled +.120 / +.082 / +.030.
    - SKL change: -.044 / -.030 / +.016 (naive -.309); beyond the common mode +.247 / +.206 / +.332; beyond naive
      +.264 / +.279 / +.319.
    FR thirds, units inside the 20 bins (the panel texts of the FR figure, dropped 11-02), lowest / middle / highest:
      Bayes grows .716 / .710 / .693, change +.517 / +.605 / +.688, shuffled +.145 / +.117 / +.100 (1,140,333 /
      1,140,330 / 1,140,283 units); Trained .564 / .599 / .656, +.169 / +.301 / +.571, +.105 / +.075 / +.065 (1,140,192
      / 1,140,458 / 1,140,467); ESN .854 / .837 / .824, +1.054 / +.994 / +.932, +.039 / +.027 / +.023 (1,134,275 /
      1,135,895 / 1,136,673); naive unchanged (1,131,614 / 1,138,035 / 1,139,701). Matched on step and bin +.081 / +.402
      / -.012 (unmatched +.171 / +.401 / -.122; cells cover .968); Trained RNN minus Bayes on one population +.393
      [+.348, +.434].
  Combined data (John, round 3; pairs_combined.npz; 40 bins):
    - Checks: the stored pair entanglement = probes.npz 'ent' at step t (max |d| 0); within-cell thirds differ in size
      by at most one unit; the pair average = half the d_H between the products of marginals (2.8e-14 on 20,000
      pair-steps); the step-standardised lines reproduce the matched fifths' means (2.2e-16); every bin of 40 holds >=
      130 units of every agent; bins with < 50 units in some fifth: Bayes 6, Trained 5, ESN 0 (some third of the other
      variable's distance: 1 / 0 / 0); the step-standardised fifths are drawn from 0.68 / 0.58 / 0.32 nats; highest -
      lowest fifth per bin, raw vs step-standardised, at 3.18 / 9.35 / 32.08 nats: Bayes +.374 vs +.363 / +.380 vs
      +.281 / +.563 vs +.403, Trained +.396 vs +.364 / +.419 vs +.351 / +.831 vs +.881, ESN +.121 vs +.030 / +.214 vs
      +.023 / +.341 vs +.039; range over every bin centred 3.18-32.08 nats (16 bins): Bayes +.368 to +.563 vs +.274 to
      +.420, Trained +.388 to +.831 vs +.332 to +.881, ESN +.121 to +.341 vs +.019 to +.063. With the pooled minimum
      (0.0062 nats) as the lowest edge, the lowest 10 / 9 / 8 bins would be empty (naive 0).
    - Shuffles (mean change over all units, range over the 5 seeds): main +.119 to +.121 / +.082 to +.083 / +.029 to
      +.030; pooled-third groups +.119 to +.120 / +.083 to +.084 / +.028 to +.029; pair average +.118 to +.121 / +.081
      to +.082 / +.043 to +.045; naive 0. With the main shuffle: shuffled +.120 [+.115, +.126] / +.083 [+.080, +.086] /
      +.029 [+.028, +.030], actual - shuffled +.483 [+.462, +.506] / +.264 [+.233, +.295] / +.976 [+.947, +1.007].
    - Definitions, the closest pairs (per variable: units; pair average: pair-steps). Below 1 nat: per variable
      (8,403 / 12,985 / 49,882) change +.564 / +.581 / +.273 (above); pair average (1,083 / 1,332 / 7,012) +.155 [+.096,
      +.253] / +.304 [+.236, +.392] / +.343 [+.312, +.374], of which aligned +.018 [-.001, +.035] / +.024 [-.002, +.054]
      / +.233 [+.205, +.261] and symmetric +.138 [+.076, +.248] / +.280 [+.221, +.357] / +.110 [+.100, +.121];
      pair-level shuffled +.136 [+.105, +.183] / +.285 [+.252, +.321] / +.148 [+.143, +.153]. 1-3 nats: pair average
      (44,558 / 61,647 / 97,422) +.279 [+.251, +.315] / +.259 [+.224, +.297] / +.568 [+.545, +.595], aligned +.022
      [+.008, +.036] / +.017 [+.000, +.034] / +.513 [+.491, +.538], symmetric +.257 / +.242 / +.055, shuffled +.254 /
      +.239 / +.127.
      Against each variable's own (step, bin of 40) cell (predicted: per unit, the mean change of all units in its
      cell; re-estimated in every resample): pair average below 1 nat, predicted +.416 [+.320, +.550] / +.541 [+.448,
      +.645] / +.309 [+.292, +.326], actual - predicted -.260 [-.360, -.176] / -.237 [-.294, -.186] / +.034 [+.013,
      +.054]; mean own d_H before .72 / .75 / .77 nats (per variable below 1 nat: .73 / .72 / .66; actual - predicted
      about 0 by construction, -.000 / +.001 / +.000).
    - Per bin (Delta d_H at the bins centred 0.50 / 0.93 / 3.18 / 9.35 / 32.08 / 94.33 nats): per variable Bayes +.62
      [+.48, +.79] / +.60 / +.31 / +.35 / +.63 / +1.41, Trained +.63 [+.48, +.80] / +.58 / +.24 / +.20 / +.48 / +.47,
      ESN +.25 / +.32 / +.50 / +.60 / +1.11 / +2.29; pair average Bayes +.10 [+.07, +.14] / +.27 [+.11, +.50] / +.26 /
      +.33 / +.62 / +1.58, Trained +.17 [+.10, +.26] / +.35 / +.21 / +.17 / +.43 / +1.84, ESN +.28 / +.42 / +.66 / +.68
      / +1.06 / +2.09.
    - Both close vs only one close (within-cell thirds of the other variable's d_H before). Median other distance in
      the lower / upper third at own bins 0.50 / 0.93 / 3.18 / 9.35 / 32.08 nats: Bayes 0.75 / 4.84, 1.31 / 5.34, 2.48 /
      7.23, 6.41 / 18.9, 18.2 / 51.5; Trained 0.94 / 4.66, 1.31 / 5.49, 2.37 / 8.35, 6.17 / 20.6, 15.0 / 46.8; ESN 1.13
      / 4.59, 1.20 / 6.73, 2.26 / 14.5, 5.26 / 32.7, 10.8 / 61.9.
      Upper - lower third, matched on step and own distance (cover .999): Delta d_H +.429 [+.401, +.460] / +.709 [+.676,
      +.743] / +.015 [+.001, +.031] (thirds Bayes +.415 / +.523 / +.845, Trained +.008 / +.318 / +.717, ESN +.945 /
      +.941 / +.960); actual - shuffled +.344 [+.314, +.377] / +.633 [+.601, +.666] / +.003 [-.011, +.018]; aligned
      +.338 / +.646 / +.011 [-.003, +.026]; symmetric +.091 / +.063 / +.005; Delta SKL +.245 [+.225, +.267] / +.340
      [+.318, +.362] / +.027 [+.017, +.037] (thirds Bayes -.151 / -.081 / +.094, Trained -.187 / -.055 / +.154, ESN
      +.005 / +.010 / +.032); SKL beyond naive +.341 / +.446 / +.121 [+.107, +.136]; SKL beyond the common mode +.149 /
      +.272 / +.112 [+.094, +.130].
      Also matched on the entanglement fifth: +.348 [+.321, +.377] / +.610 [+.579, +.641] / +.007 [-.006, +.021] (cover
      .986-.988); on the FR third: +.417 [+.387, +.450] / +.662 [+.630, +.694] / +.017 [+.003, +.031].
      Per bin, lower / upper third: at 0.50 nats +.27 [+.12, +.50] / +.98 [+.71, +1.38] (Bayes), +.35 [+.20, +.52] /
      +.90 [+.70, +1.15] (Trained), +.25 [+.22, +.28] / +.24 [+.21, +.27] (ESN); at 0.93 nats +.44 / +.92, +.40 / +.91,
      +.34 / +.31; at 9.35 nats +.21 / +.55, +.03 [+.00, +.05] / +.42, +.59 / +.62; at 94.33 nats +1.15 / +1.64, -.67
      [-.75, -.59] / +1.73, +2.27 / +2.31.
      The Trained RNN's lower third: matched mean +.008 [-.021, +.035] (middle +.318 [+.288, +.347], upper +.717 [+.687,
      +.746]); per bin +.147 to +.582 below 3 nats (bins 0.27-2.73), +.003 to +.059 at 3.18-37.42 nats (at 9.35 nats
      +.029 [+.004, +.055]) and -.992 to -.054 at 43.66-110.05 nats (at 94.33 nats -.674 [-.754, -.594]).
      Trained RNN - Bayes on one population (726 cells): +.400 [+.353, +.445] (Bayes's weights +.490 [+.432, +.548]);
      actual - shuffled +.420; aligned +.437; symmetric -.037 [-.043, -.030]; Delta SKL +.199; SKL beyond naive +.214;
      beyond the common mode +.197.
      Bayes's product of marginals (no coupling; the same cells, cover .999): upper - lower third +1.202 [+1.141,
      +1.276] (thirds +.326 / +.877 / +1.527); the coupling's part (actual - product) -.772 [-.841, -.711] (thirds +.090
      [+.047, +.129] / -.354 [-.427, -.288] / -.683 [-.774, -.603]). Product - Trained RNN on one population +.227
      [+.177, +.287] (Bayes's weights +.252 [+.198, +.316]).
      Pooled (absolute) thirds, upper - lower: raw +.548 [+.527, +.567] / +.590 [+.561, +.623] / +.189 [+.123, +.253];
      matched on step and own bin +.529 [+.498, +.561] / +.837 [+.796, +.878] / +.023 [+.005, +.042] (cover .822 / .839
      / .901).
    - The entanglement of the observation (whole belief), highest - lowest fifth. Delta d_H: raw +.758 [+.697, +.820] /
      +.745 [+.702, +.795] / +.560 [+.495, +.622]; matched on bin +.570 / +.663 / +.261; on step and bin +.460 [+.395,
      +.530] / +.651 [+.600, +.713] / +.035 [+.016, +.056] (cover .984 / .987 / .992; matched fifths Bayes +.378 / +.451
      / +.495 / +.606 / +.838, Trained +.081 / +.136 / +.204 / +.362 / +.732, ESN +.914 / +.933 / +.939 / +.940 /
      +.949); also on the FR third +.427 [+.363, +.499] / +.529 [+.486, +.586] / +.042 [+.024, +.062] (cover .937 /
      .937 / .951); also on the within-cell third of the other variable's distance +.340 [+.276, +.411] / +.427 [+.385,
      +.484] / +.036 [+.017, +.053]; on step band, bin and environment +.562 [+.513, +.613] / +.626 [+.584, +.669] /
      +.094 [+.077, +.112] (cover .810 / .802 / .785). Thirds +.352 / +.496 / +.026 [+.010, +.043]; the unit's own
      variable's entanglement fifths +.590 / +.636 / +.030; the other variable's +.205 / +.512 / +.037.
      On step and bin, the other measures: actual - shuffled +.221 [+.159, +.289] / +.518 [+.468, +.580] / +.021 [+.002,
      +.040]; aligned +.222 / +.536 / +.027; symmetric +.238 / +.115 / +.009; Delta SKL +.348 [+.315, +.383] / +.345
      [+.317, +.373] / +.034 [+.021, +.049] (fifths Bayes -.203 / -.177 / -.123 / -.036 / +.145, Trained -.177 / -.155 /
      -.112 / -.029 / +.168, ESN -.003 / +.001 / +.009 / +.021 / +.031); SKL beyond naive +.422 / +.407 / +.098 [+.082,
      +.116]; SKL beyond the common mode +.209 / +.256 / +.105 [+.091, +.120]. Raw: actual - shuffled +.565 / +.659 /
      +.559; Delta SKL +.326 / +.327 / +.041; beyond naive +.653 / +.671 / +.329; beyond the common mode +.406 / +.379 /
      +.351.
      Trained RNN - Bayes on one population (594 cells): +.334 [+.265, +.402] (Bayes's weights +.364 [+.289, +.440]);
      actual - shuffled +.467; aligned +.463; symmetric -.129 [-.146, -.111]; Delta SKL +.113 [+.089, +.138]; beyond
      naive +.079 [+.060, +.101]; beyond the common mode +.131 [+.107, +.152].
      Bayes's product of marginals (no coupling; the same cells, cover .984): highest - lowest fifth +1.404 [+1.239,
      +1.587] (fifths +.370 / +.436 / +.574 / +.907 / +1.774); the coupling's part -.944 [-1.142, -.774] (fifths +.008
      [-.062, +.071] / +.014 [-.031, +.059] / -.079 [-.124, -.033] / -.301 [-.388, -.230] / -.936 [-1.121, -.780]).
      Product - Trained RNN on one population +.498 [+.321, +.692] (Bayes's weights +.531 [+.351, +.730]).
      Per bin (step-standardised), lowest / highest fifth: at 0.93 nats +.39 / +1.33 [+.96, +1.78] (Bayes), +.23 / +1.12
      [+.87, +1.38] (Trained), +.30 / +.35 (ESN); at 9.35 nats +.25 / +.54, +.09 / +.44, +.58 / +.61; at 94.33 nats +.48
      / +1.83, -.63 [-.81, -.37] / +1.10, +2.22 / +2.21.
    - FR thirds at 40 bins, matched on step and bin: +.079 [+.060, +.098] / +.401 [+.366, +.435] / -.008 [-.019,
      +.002]; also matched on the entanglement fifth -.023 [-.045, -.000] / +.271 [+.246, +.296] / -.019 [-.028, -.010]
      (cover .913-.923).
READING
  1. After an identical observation vector, the range of the log-ratio (d_H) between two marginal beliefs grows in
     Bayes, the Trained RNN and the ESN, beyond an orientation-free shuffle of the step's difference (every grouping
     tried). Naive's never changes (T1). Most of the growth is aligned with the existing difference. Similar beliefs
     (< 1 nat apart) grow apart by +.564 / +.581 / +.273 nats; for Bayes and the Trained RNN almost all of it is the
     symmetric (size) part (+.513 / +.553), for the ESN about half (+.125). The part aligned with the existing
     difference is small but positive for Bayes (+.051 [+.039, +.062]) and the Trained RNN (+.028 [+.011, +.046]), and
     about half of the ESN's change (+.148 [+.138, +.158]). The shuffle is too noisy to resolve the small parts (+.047
     [-.028, +.147], +.032 [-.039, +.115]).
  2. "Grows" holds for d_H only. Weighted by probability (SKL, JS), Bayes's and the Trained RNN's pairs on average come
     slightly closer, and the ESN's barely move. They separate only relative to naive, whose shared update brings the
     pairs together (SKL -.309), and relative to their own common-mode update.
  3. Bayes: each variable's update averages the joint likelihood over the other variable, weighted by the probe's own
     belief about it (conditional on this variable's value). Two probes with different beliefs about the other
     variable therefore receive different likelihoods for this one. Naive's likelihood for a variable does not depend
     on its belief about the other (it averages each observation's table uniformly over the other variable), so it is
     the same for both probes and its d_H cannot change. On average the coupling restrains the divergence (actual -
     product of marginals -.323 [-.390, -.265]). It does so for pairs more than about 2 nats apart whose range is not
     set by near-impossible values, mostly in the highest FR third. For the closest pairs (< 1 nat: +.091 [+.048,
     +.135]) and for the 17% of units whose d_H is set by near-impossible values (+.291 [+.245, +.336]) it adds to the
     divergence. In the lowest FR third its net part is about 0 (-.018 [-.057, +.021]). The T2 bound holds but is
     loose here, so it does not set the size.
  4. FR: at matched step and distance before, the Trained RNN's change grows with FR (+.40 nats, highest - lowest
     third; +.33 to +.36 with the environment matched as well). Bayes's grows slightly (+.08; +.06 to +.10). The ESN's
     d_H barely depends on FR (-.012 [-.023, -.001]; with the other bins the interval includes 0). The ESN's bulk
     changes depend little on FR: its SKL change beyond naive grows slightly in every step-matched comparison (+.02 to
     +.10); its SKL change beyond the common mode and its JS slopes are small and change size or sign with the
     matching. The raw trends of Bayes (+.17) and the ESN (-.12) mostly reflect that high-FR pairs come later and
     start farther apart. Each agent's contrast is standardised to its own (step, d_H before) units; on a common
     population the Trained RNN grows more steeply than Bayes by +.27 to +.45 in d_H, +.06 to +.15 in SKL beyond naive
     and +.085 to +.20 beyond the common mode. In JS beyond naive the difference nearly vanishes (-.0001 to +.0006,
     against Bayes's own +.0062); in JS beyond the common mode the Trained RNN's slope is larger than Bayes's by +.0015
     [+.0012, +.0017] (Bayes's own +.0013 [+.0010, +.0018]). Matched on the observation's entanglement as well (40
     bins), Bayes's FR trend vanishes and turns slightly negative (+.079 [+.060, +.098] -> -.023 [-.045, -.000]): its
     divergence follows the entanglement of the observation, not the regret before it. The Trained RNN keeps part of
     its FR trend (+.271 [+.246, +.296] of +.401).
  5. Chaos is a separate question with two separate results. (a) No network is chaotic by td_tt_chaos.py's criterion,
     the asymptotic Lyapunov exponent of the state (M1(b)): lambda_1 < 0 at the end of training (Trained RNN -.0129
     [-.0133, -.0126], ESN -.362 [-.368, -.355] per step). This does not mean that the state contracts within an
     episode: the Trained RNN's finite-time amplification over the 30 steps is positive (+.016 per step, M1(a)). (b) The
     divergence here is a one-step divergence of marginal beliefs, and Bayes shows it too, although its joint filter
     conserves d_H between joint beliefs (T1: every exponent exactly 0 in log coordinates). So it is not evidence of
     sensitive dependence of a state.
  6. The entanglement of the observation (John, round 3). At matched step and distance before, Bayes's and the Trained
     RNN's pairs separate most after the most entangled observations (highest - lowest fifth +.460 [+.395, +.530] and
     +.651 [+.600, +.713] nats). The ESN's barely depend on it: +.035 [+.016, +.056], at most +.094 with the other
     step-matched variants (+.261 [+.221, +.300] matched on distance alone; its matched fifths' means are +.91 to
     +.95). The Trained RNN separates mostly after the most entangled observations: +.081 nats after the least
     entangled fifth (Bayes +.378), +.732 after the most entangled (Bayes +.838), mostly in the part aligned with the
     existing difference (+.536 of +.651). The dependence survives matching on the FR third (+.427 / +.529 / +.042) and
     on the other variable's distance (+.340 / +.427 / +.036). On one population it is steeper for the Trained RNN than
     for Bayes (+.334 [+.265, +.402]). The ESN's raw trend (+.560) comes from the distance before and the step. For
     Bayes the dependence is partly built in (with zero entanglement its update would equal naive's); for the networks
     it is measured.
  7. Both close vs only one close (John, round 3). At the same own distance and step, a variable's beliefs separate
     more when the pair's beliefs about the other variable are farther apart: upper - lower third of the other
     variable's distance +.429 [+.401, +.460] (Bayes), +.709 [+.676, +.743] (Trained RNN), +.015 [+.001, +.031] (ESN).
     For Bayes this comes from the mechanism of reading 3 (each probe averages the joint likelihood over its own belief
     about the other variable), not from the coupling between the variables: without the coupling (product of
     marginals) the dependence is larger, +1.202 [+1.141, +1.276], and the coupling's part works against it, -.772
     [-.841, -.711]. By third (+.090 / -.354 / -.683) the coupling adds slightly where the beliefs about the other
     variable are closest and restrains the divergence most where they differ most. The Trained RNN's dependence lies
     between Bayes's and the no-coupling one, also on one population (product - Trained RNN +.227 [+.177, +.287],
     Trained RNN - Bayes +.400 [+.353, +.445]), so being larger than Bayes's is no sign of more coupling. When the
     other variable is relatively close (lower third), the Trained RNN's d_H is about unchanged on average (+.008
     [-.021, +.035]), but this averages growth below 3 nats (+.15 to +.58 per bin), little change at 3.2-37 nats (+.00
     to +.06; at 9.35 nats +.029 [+.004, +.055]) and shrinkage beyond 40 nats (-.05 to -.99; at 94.33 nats -.674
     [-.754, -.594]). When the other variable is far (upper third), it grows (+.717). Near 0.5 nats, with both close /
     only this one close: +.27 / +.98 (Bayes), +.35 / +.90 (Trained RNN), +.25 / +.24 (ESN). Bayes's and the Trained
     RNN's dependence survives matching on the entanglement fifth (+.348 / +.610) and on the FR third (+.417 / +.662).
     The ESN's divergence barely depends on the other variable: +.015 [+.001, +.031] against +.429 / +.709 (+.017
     [+.003, +.031] with the FR third matched); with the entanglement fifth matched it is not resolved, +.007 [-.006,
     +.021].
  8. Definitions of the distance (John, round 3). Per variable and as the pair's average, the overall mean is the same
     (the same changes); the bins differ, and a small pair average means both variables are close (a pair average
     below 1 nat puts both within 2 nats). Pairs close in both variables (pair average < 1 nat) separate much less than
     single close variables in Bayes (+.155 vs +.564) and the Trained RNN (+.304 vs +.581), and less than predicted
     from each variable's own distance and step, by -.260 [-.360, -.176] and -.237 [-.294, -.186] (their own distances
     are like the per-variable sets', .72 / .75 vs .73 / .72 nats). This is reading 7 seen from the other side. Their
     growth is mostly symmetric (size) growth, +.138 of +.155 and +.280 of +.304; the aligned part is not resolved from
     0 (+.018 [-.001, +.035], +.024 [-.002, +.054]). The ESN's pair-average change is larger than its per-variable
     change (+.343 vs +.273) and mostly aligned (+.233). Part of that +.070 excess is composition: the change
     predicted from each variable's own distance and step is +.309 for the pair-average set, against +.273 for the
     per-variable set; +.034 [+.013, +.054] remains.
  9. What survives matching on step and distance before (intervals excluding 0) and what does not:
     - Survive: the entanglement dependence of Bayes and the Trained RNN (also with FR, the other variable's distance or
       the environment matched); the other-variable dependence of Bayes and the Trained RNN (also with entanglement or
       FR matched); the Trained RNN's FR dependence (also with the environment matched; partly with entanglement
       matched, +.271).
     - Shrink to small effects (still resolved from 0 unless noted): Bayes's FR dependence once entanglement is also
       matched (raw +.171 -> +.081 -> -.023 [-.045, -.000], slightly negative); the ESN's raw entanglement trend (+.560
       -> +.035 [+.016, +.056]), its raw other-variable trend (pooled thirds +.189 -> +.023 [+.005, +.042]; within-cell
       thirds +.015 [+.001, +.031], not resolved with entanglement matched) and its raw FR trend (-.121 -> -.012
       [-.023, -.001]; not resolved with the other bins, -.010 to -.008).
CORRECTIONS
  11-02, round 4 (the third adversarial check):
  - Wrong attribution: "For Bayes this is the coupling of reading 3; the Trained RNN shows the same signature more
    strongly" (READING 7). Without the coupling (product of marginals) Bayes's dependence on the other variable's
    distance is larger (+1.202 vs +.429), and the coupling's part works against it (-.772); the Trained RNN lies
    between the two. The dependence comes from each probe averaging the joint likelihood over its own belief about the
    other variable.
  - Over-claimed nulls: "the ESN's divergence ignores the other variable" (+.015 [+.001, +.031] is small but resolved);
    Bayes's FR trend with entanglement matched "vanishes" (-.023 [-.045, -.000] is slightly negative).
  - READING 8 compared the two definitions' closest pairs unmatched. Matched on each variable's own distance and step,
    pairs close in both variables separate less by -.260 (Bayes) and -.237 (Trained RNN); part of the ESN's raw excess
    (+.070) is composition (+.034 remains). Dropped: "their growth is what the step's difference does through its
    size" (an equivalence read from an unresolved interval).
  - The Trained RNN's lower third "does not grow" (+.008) averaged growth below 3 nats with shrinkage beyond 40 nats;
    the profile is now given (READING 7). READING 1's "almost all symmetric" also covered the ESN (about half).
  - Numbers: the pooled minimum leaves the lowest 10 / 9 / 8 bins empty (not 10 for both Bayes and the Trained RNN);
    the ESN's step-standardised fifths differ by +.02 to +.06 over every bin from 3.18 to 32.08 nats (not +.02 to
    +.04, which held at three bins); "at most +.094 with the other matchings" left out +.261 (distance alone).
  - "About twice Bayes's" (a ratio to a small baseline) is now the difference (READING 4). "Separates almost only
    after entangled observations" is now "mostly after the most entangled" (middle fifths +.136 to +.362; READING 6).
  - Figures: `figure` now writes td_tt_chaos_pairs_detail.png, with the panel texts of rows 2-3 moved above the axes
    (the violins' tails ran under them); td_tt_chaos_pairs.png is the combined figure (td_tt_chaos_combined.py). The
    FR figure and its command `fr` were dropped.
  11-02, round 3 (the second adversarial check):
  - Wrong: "For the closest pairs (< 1 nat), Bayes's and the Trained RNN's growth is no more than the shuffle's; the
    ESN's is more." The exact sign-flip split shows a small positive aligned part for both (+.051 [+.039, +.062], +.028
    [+.011, +.046]). The shuffle (+.047 [-.028, +.147], +.032 [-.039, +.115]) is too noisy there because its donors come
    from all environments. All three align, the ESN most (READING 1).
  - Over-general: "the coupling restrains the divergence" holds on average only (READING 3).
  - Over-stated: the ESN's "SKL and JS changes beyond naive or its common mode grow with FR" (READING 4).
  - Trained - Bayes was the difference of two contrasts standardised to different populations. This is now stated,
    with the range over common populations (+.27 to +.45); the FR figure's panel o shows the one-population value
    (+.393 instead of +.321).
  - The Figures paragraph referred to per-third numbers that were not in the docstring; they are listed now.
  - "No network is chaotic" now names its criterion, the asymptotic exponent (READING 5).
  11-02 (the first check, of td_tt_chaos.py's M4):
  - The old shuffle was not orientation-free. Pairs come out i < j by row and rows are cell-ordered (r0_i <= r0_j in
    every pair), so a permuted delta shared the unit's orientation. It gave shuffled +.159 / +.060 / +.249 and actual -
    shuffled +.444 / +.287 / +.756, and the ESN's dashed line was mostly this artifact. Orientation-free, they are now
    +.120 / +.082 / +.030 and +.483 / +.265 / +.975.
  - Retracted: "Bayes's divergence is bounded by the joint's conserved distance (the coupling the marginals do not
    show)". Removing the coupling increases it (+.927 vs +.604).
  - Corrected: Bayes's divergence "grows with FR" (raw +.171) mostly through the distance before and the step;
    matched on both, the trend is +.081. Retracted: the ESN's "shrinks with FR, so it is not the coupling's"; matched,
    its trend is -.012 [-.023, -.001] (-.010 [-.021, +.001] with the old bins), and the inference did not follow.
  - "Nearby marginal beliefs ... move apart": the pairs are not nearby (median d_H before 22 nats for Bayes), and
    "apart" holds for d_H only.
  - "Neither network's state is chaotic, so this divergence is history-dependent updating": the inference did not
    follow (READING 5).
  - Figures: the panel texts used all units while the panels drew the units inside the bins; the bins started at the
    pooled 1st percentile (0.99 nats), so the closest pairs were not drawn; the footnote's count was Bayes's only;
    Bayes's mean line and the shuffle line were both black.
COMBINED CACHE (pairs_combined.npz, written by combined(). Axes: A = 4 agents in the order of `agents` (bayes, naive,
  trained, echo), B = 40 bins, M = 9 measures, N_BOOT = 1,000. Every _mean / _lo / _hi triple holds the full-sample
  value and the 5% / 95% over the shared environment resamples; NaN where a bin has no units)
  Labels: agents (A,), names (A,) = Bayes, naive, Trained RNN, ESN; colors (A,) = k, 0.55, #1f77b4, #ff7f0e; edges40
    (B + 1,) nats and centres40 (B,) (geometric centres); meas (M,) = dh (Delta d_H), shuf (shuffled Delta d_H), sym and
    aligned (the sign-flip parts, dh = sym + aligned), skl (Delta SKL), skl_cm (the common-mode update's Delta SKL),
    skl_naive (naive's Delta SKL on the same pair-steps), skl_bn (skl - skl_naive), skl_bcm (SKL after - SKL after the
    common-mode update); ent5_edges (6,), ent3_edges (4,), oth_edges (4,) (pooled thirds of the other variable's d_H
    before, nats), fr_edges (4,); n_pairsteps, n_units, n_boot, shuf_seeds, n_min (20), line_min (500), edge_pct.
  Per variable: var_mean / var_lo / var_hi (A, M, B); var_n (A, B) units per bin; var_out (A, 2) units below / above.
  Pair average: avg_mean / avg_lo / avg_hi (A, M, B) over the bins of the pair's average d_H before (shuf = the
    pair-level shuffle); avg_n (A, B) pair-steps; avg_out (A, 2).
  Both close vs only one close: cond_mean / cond_lo / cond_hi (A, M, 3, B) (third 0 = the other variable closest;
    matched on step by construction); cond_n (A, 3, B); cond_med_other (A, 3, B) the median other d_H before per third
    and bin (bins with >= 50 units).
  Pooled thirds of the other variable's distance: oth_mean / oth_lo / oth_hi (A, M, 3, B) raw per-bin means (shuf = the
    pooled-third shuffle); oth_n (A, 3, B).
  Entanglement fifths: ent_mean / ent_lo / ent_hi (A, M, 5, B) the step-standardised lines (fifth 0 = least entangled;
    NaN where the used cells hold < 500 units); ent_used (A, B) units in the used cells; entraw_mean / entraw_lo /
    entraw_hi (A, M, 5, B) the raw per-bin means (step-confounded); ent_n = entraw_n (A, 5, B) units.
  Map: map_sum / map_n (A, B, B) the sum and count of Delta d_H by (own bin, other variable's bin), for units with both
    distances inside the bins; mean = map_sum / map_n.
  Violins (the first figure's rows at 40 bins): vio_meas (3,) = log_after (log10 of d_H after), dh, skl; vio_y / vio_d
    (A, 3, B, 100) the KDE grid and density (matplotlib's violin statistics: Scott's rule on at most 20,000 units per
    bin, seeded subsample, 100 points from the subsample's minimum to its maximum; scale each violin's density to its
    maximum for the width); vio_med (A, 3, B) the median of all units in the bin (naive's dh is NaN: d_H never changes).
  Contrasts and summaries: stat_names (S,), stat_boot (S, 1 + N_BOOT) (row 0 the full sample, then the resamples),
    stat_aux (S,) (matched contrasts: the share of the units in the used cells; 'close': the number of units or
    pair-steps; Trained - Bayes: the number of common cells; raw: 1; else NaN). Names 'split|measure|matching|agent':
    split = ent5 / ent3 (entanglement fifths / thirds), ent5own / ent5other (fifths of the unit's own / the other
    variable's entanglement), cond3 (within-cell thirds), oth3 (pooled thirds), fr3 (FR thirds); measure = dh, dh-shuf,
    aligned, sym, skl, skl_bn, skl_bcm; matching = raw, bin, step+bin, step+bin+fr, step+bin+cond3, step+bin+ent5,
    band+bin+env; agent = bayes, trained, echo, trained-bayes (one population, summed weights) or trained-bayes-Bweights
    (Bayes's weights). Summaries 'all|measure|agent' and 'close|<definition><range>|measure|agent' with definition var
    or avg and range <1 or 1-3 (nats). cmeas (7,), ent5_std (A, 7, 5, 1 + N_BOOT) and cond3_std (A, 7, 3, 1 + N_BOOT):
    the matched group means (step and bin); shuf_seed_means (A, 3, N_SHUF) (main, pooled-third, pair-average shuffle);
    chk_names, chk_values.
  pairs_shuf40.npz: per unit (m, 2) the three shuffles' d_H after (shuf_, shufo_, shufa_<agent>), the within-cell
    thirds (cond_<agent>, int8), and the entanglement fifth per pair-step (g5, int8).
  Drawing notes: draw ent_* (step-standardised), not entraw_* (the ESN's raw fifths separate by step alone), and cond_*
    for "both close vs only one close", not oth_* (empty where the own variable is close). Naive's Delta d_H is 0 in
    every split (T1), so its split panels carry no result. The step-standardised fifths start at 0.68 / 0.58 / 0.32 nats
    (Bayes / Trained RNN / ESN); the intervals of the lowest bins are wide (few units).
DATA
  Probes: analysis/probes.npz (10,000 = 50 test environments x 100 cells x 2 streams of 30 steps). Networks: the
  seed-20261029 runs at raw checkpoint 49995 (39,996 training episodes), td_tt_intervene's fp32 forward. Caches
  (through_training/chaos/): pairs_lq.npz holds the four agents' marginal log beliefs on all probes (float64);
  pairs_v2.npz holds the pair set and every per-unit measure; pairs_combined.npz and pairs_shuf40.npz hold the combined
  data (above). pairs.npz (424 MB) is the old M4 cache (oriented shuffle); it is read only for the comparison and can
  be deleted once John confirms (DATA is not in git); build() and numbers() then skip it (they check that it exists).

Run from the repository root:
    CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
        /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_chaos_pairs.py \\
        build | figure | combined | numbers
    (build: the GPU for the two network forwards, about 2.5 min; figure: CPU, under 2 min; combined: CPU, about 4 min;
    numbers: CPU, about 3 min, and it also prints the combined numbers when pairs_combined.npz exists)"""
import os
import sys
import time
import numpy as np
from scipy.special import logsumexp

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
OUT = os.path.join(gap.TT, "chaos")
CACHE = os.path.join(OUT, "pairs_v2.npz")              # per-unit measures (build)
LQ_CACHE = os.path.join(OUT, "pairs_lq.npz")           # the four agents' marginal log beliefs on all probes (build)
OLD_CACHE = os.path.join(OUT, "pairs.npz")             # td_tt_chaos.py's old M4 cache (oriented shuffle), compared only
COMB_CACHE = os.path.join(OUT, "pairs_combined.npz")   # the combined figure's data (combined)
SHUF40_CACHE = os.path.join(OUT, "pairs_shuf40.npz")   # the combined analyses' per-unit shuffle and groups (combined)
T, R, NV, NO = 30, 10, 2, 5
END_RAW = 49995                                        # raw checkpoint episode = 39,996 training episodes
RUNS = ("trained", "echo")
AGENTS = (("bayes", "Bayes", "k"), ("naive", "naive", "0.55"), ("trained", "Trained RNN", "#1f77b4"),
          ("echo", "ESN", "#ff7f0e"))
PAIR_BINS, EDGE_PCT, KDE_MAX = 20, (.1, 99.9), 20000
PAIR_SEED, N_SHUF = 20261107, 5
N_MIN = 20                                             # matched estimator: units per group in a cell
SHUF_C, CM_C = "#a50f15", "#54278f"                    # shuffled null (dark red), common-mode update (dark purple)
NBIN40, EDGE40_PCT = 40, (.1, 99.)                     # the combined figure's bins (John 11-02: 20 -> 40 bins)
ENT_K, OTH_K = 5, 3                                    # entanglement fifths; thirds of the other variable's distance
SHUF40_SEED = PAIR_SEED + 1000                         # seeds of the combined analyses' shuffle
LINE_MIN = 500                                         # a step-standardised line needs >= 500 units in its used cells
COMB_AGENTS = tuple(k for k, _, _ in AGENTS)           # the agent axis of pairs_combined.npz
MEAS = ("dh", "shuf", "sym", "aligned", "skl", "skl_cm", "skl_naive", "skl_bn", "skl_bcm")
VIO = ("log_after", "dh", "skl")
_P = {}


# ═══════════════════════════════════════════════════════════════════
# Observers (float64; copied from td_tt_chaos.py so that this module does not depend on it)
# ═══════════════════════════════════════════════════════════════════

def probes():
    """analysis/probes.npz: 10,000 probes = 50 test environments x 100 cells x 2 streams of 30 steps
    (environment-major)"""
    if not _P:
        Z = np.load(os.path.join(AN, "probes.npz"))
        _P.update({k: Z[k] for k in ("obs", "env", "log_jb", "log_nb")})
        _P.update({k: Z[k].astype(np.float64) for k in ("L", "fr", "ent")})
    return _P


def osc(a, axis=-1):
    return a.max(axis) - a.min(axis)


def step_loglik(obs, L):
    """(n, T, R, R) log p(x_t | r0, r1) for obs (n, T, 5) and per-episode tables L (n, 5, R, R)"""
    o = obs.astype(np.float64)
    return np.einsum("nti,nijk->ntjk", o, np.log(L)) + np.einsum("nti,nijk->ntjk", 1 - o, np.log1p(-L))


def marg_log(u):
    """unnormalised log marginals (..., 2, R) of joint log beliefs u (..., R, R)"""
    return np.stack([logsumexp(u, -1), logsumexp(u, -2)], -2)


def exact_observers():
    """float64 log marginals (10,000, T, 2, R) of joint Bayes and naive on all probes (td_analysis.bayes_side's
    formulas, recomputed in float64; uniform prior over the 100 cells)"""
    P = probes()
    n = len(P["env"])
    out = {"bayes": np.zeros((n, T, NV, R)), "naive": np.zeros((n, T, NV, R))}
    for a in range(0, n, 1000):
        sl = slice(a, a + 1000)
        Le = P["L"][P["env"][sl]]
        u = np.cumsum(step_loglik(P["obs"][sl], Le), 1)
        lm = marg_log(u)
        out["bayes"][sl] = lm - logsumexp(lm, -1)[..., None]
        Lb = np.stack([Le.mean(3), Le.mean(2)], 2)
        o = P["obs"][sl].astype(np.float64)
        cl = np.cumsum(np.einsum("nti,nikr->ntkr", o, np.log(Lb)) + np.einsum("nti,nikr->ntkr", 1 - o, np.log1p(-Lb)),
                       1)
        out["naive"][sl] = cl - logsumexp(cl, -1)[..., None]
    return out


def network_beliefs(dev=None):
    """the Trained RNN's and the ESN's marginal log beliefs (10,000, T, 2, R) at the end of training (td_tt_intervene's
    fp32 engine forward, GPU 0), as float64"""
    import td_tt_intervene as ti
    dev = ti._device(dev)
    return {run: ti.forward(ti.load(run, END_RAW), device=dev)["logq"].astype(np.float64) for run in RUNS}


# ═══════════════════════════════════════════════════════════════════
# Pairs and their per-unit measures (a unit = one variable of one pair-step)
# ═══════════════════════════════════════════════════════════════════

def pair_set():
    """John's pairs: per environment and step t = 2..30, every two probes (i < j) with the same observation vector at t
    and different observation counts before t. Returns I, J (probe rows), t (1-based step of the shared observation),
    and the number of same-observation pairs left out for equal counts"""
    P = probes()
    obs, env = P["obs"], P["env"]
    code = (obs * (1 << np.arange(NO))).sum(-1)
    cnt = np.concatenate([np.zeros((len(env), 1, NO), np.int64), np.cumsum(obs, 1)], 1)    # cnt[:, t - 1]: before t
    I, J, S, n_eq = [], [], [], 0
    for e in range(gap.N_ENV):
        rows = np.flatnonzero(env == e)
        a, b = np.triu_indices(len(rows), 1)
        ra, rb = rows[a], rows[b]
        for t in range(2, T + 1):
            same = code[ra, t - 1] == code[rb, t - 1]
            k = same & (cnt[ra, t - 1] != cnt[rb, t - 1]).any(-1)
            n_eq += int(same.sum() - k.sum())
            I.append(ra[k])
            J.append(rb[k])
            S.append(np.full(int(k.sum()), t, np.int64))
    return np.concatenate(I), np.concatenate(J), np.concatenate(S), n_eq


def _norm(lg):
    return lg - logsumexp(lg, -1)[..., None]


def _skl(a, b):
    """symmetric KL [KL(p||q) + KL(q||p)] / 2 of log beliefs a, b (..., R)"""
    return .5 * ((np.exp(a) - np.exp(b)) * (a - b)).sum(-1)


def _js(a, b):
    """Jensen-Shannon divergence [KL(p||m) + KL(q||m)] / 2, m = (p + q) / 2 (nats, at most log 2)"""
    m = np.logaddexp(a, b) - np.log(2.)
    return .5 * ((np.exp(a) * (a - m)).sum(-1) + (np.exp(b) * (b - m)).sum(-1))


def pair_measures(lq, I, J, S, chunk=200000):
    """per unit (m, 2): d_H before / after / after with the step's difference reversed (flip = osc(D - delta)); SKL and
    JS before / after / after the common-mode update (both log beliefs shifted by the pair's mean log-update, which
    leaves d_H unchanged). Also returns max |d_H(common mode) - d_H(before)| (a check)"""
    keys = ("pre", "post", "flip", "skl_pre", "skl_post", "skl_cm", "js_pre", "js_post", "js_cm")
    out = {k: np.zeros((len(I), NV)) for k in keys}
    cm_dev = 0.
    for a in range(0, len(I), chunk):
        sl = slice(a, a + chunk)
        i, j, t = I[sl], J[sl], S[sl]
        D = lq[i, t - 2] - lq[j, t - 2]
        dl = (lq[i, t - 1] - lq[i, t - 2]) - (lq[j, t - 1] - lq[j, t - 2])
        out["pre"][sl], out["post"][sl] = osc(D), osc(lq[i, t - 1] - lq[j, t - 1])
        out["flip"][sl] = osc(D - dl)
        ai, aj, bi, bj = _norm(lq[i, t - 2]), _norm(lq[j, t - 2]), _norm(lq[i, t - 1]), _norm(lq[j, t - 1])
        c = ((bi - ai) + (bj - aj)) / 2
        ci, cj = _norm(ai + c), _norm(aj + c)
        cm_dev = max(cm_dev, float(np.abs(osc(ci - cj) - osc(ai - aj)).max()))
        out["skl_pre"][sl], out["skl_post"][sl], out["skl_cm"][sl] = _skl(ai, aj), _skl(bi, bj), _skl(ci, cj)
        out["js_pre"][sl], out["js_post"][sl], out["js_cm"][sl] = _js(ai, aj), _js(bi, bj), _js(ci, cj)
    return out, cm_dev


def make_edges(Z):
    """20 log-spaced bins of d_H before, shared by the agents: the pooled 0.1st-99.9th percentiles (all units)"""
    pooled = np.concatenate([Z[f"pre_{k}"].ravel() for k, _, _ in AGENTS])
    lo, hi = np.percentile(pooled, EDGE_PCT)
    return np.logspace(np.log10(lo), np.log10(hi), PAIR_BINS + 1)


def unit_bins(pre, edges):
    """bin of d_H before: 0..19 inside the edges, -1 below, PAIR_BINS above"""
    return np.digitize(pre, edges) - 1


def fr_thirds(fr):
    """FR third edges (pooled over all units) and the third of each unit (same shape as fr)"""
    fe = np.quantile(fr.ravel(), [0, 1 / 3, 2 / 3, 1])
    return fe, np.clip(np.searchsorted(fe, fr, side="right") - 1, 0, 2)


def _within_perm(gid, rng):
    """a random permutation that maps every unit to a unit of the same group"""
    a = np.lexsort((np.arange(len(gid)), gid))
    b = np.lexsort((rng.random(len(gid)), gid))
    perm = np.empty(len(gid), np.int64)
    perm[a] = b
    return perm


def pair_shuffle(lq, I, J, S, gid, seeds):
    """the orientation-free shuffle: per variable, the step's difference delta = (change of log p) - (change of log q)
    of another unit of the same group gid (m, 2), with a random sign (orientation) per unit: osc(D_before + sign x
    delta_permuted). Returns the per-unit mean over the seeds (m, 2) and, per seed, the mean shuffled change over all
    units"""
    D = lq[I, S - 2] - lq[J, S - 2]
    dl = (lq[I, S - 1] - lq[I, S - 2]) - (lq[J, S - 1] - lq[J, S - 2])
    pre = osc(D)
    acc, per_seed = np.zeros((len(I), NV)), []
    for sd in seeds:
        rng = np.random.default_rng(int(sd))
        sh = np.zeros((len(I), NV))
        for v in range(NV):
            perm = _within_perm(gid[:, v], rng)
            sgn = rng.choice([-1., 1.], len(I))
            sh[:, v] = osc(D[:, v] + sgn[:, None] * dl[perm, v])
        acc += sh
        per_seed.append(float(np.mean(sh - pre)))
    return acc / len(seeds), np.array(per_seed)


def bayes_counterfactual(lqb, I, J, S, chunk=100000):
    """Bayes only. (1) Product of marginals: each probe's joint before step t replaced by the product of its own Bayes
    marginals (no coupling), the same observation's joint likelihood applied, new marginals -> d_H after (m, 2).
    (2) The pair's joint d_H (count-based joint log posteriors; conserved by the shared observation, T1) (m,).
    (3) The same code path run with the true joint: its marginal d_H after (m, 2), which must equal the actual one"""
    P = probes()
    L, obs, env = P["L"], P["obs"].astype(np.float64), P["env"]
    cum = np.cumsum(obs, 1)                                                                # cum[:, t - 2]: steps 1..t-1
    prod, joint, true = np.zeros((len(I), NV)), np.zeros(len(I)), np.zeros((len(I), NV))
    for a in range(0, len(I), chunk):
        sl = slice(a, a + chunk)
        i, j, t = I[sl], J[sl], S[sl]
        Le = L[env[i]]
        lL, l1L = np.log(Le), np.log1p(-Le)
        x = obs[i, t - 1]
        ll = np.einsum("no,nojk->njk", x, lL) + np.einsum("no,nojk->njk", 1 - x, l1L)
        mp, mt, u = [], [], []
        for r_ in (i, j):
            mg = lqb[r_, t - 2]
            mp.append(marg_log(mg[:, 0, :, None] + mg[:, 1, None, :] + ll))
            n1 = cum[r_, t - 2]
            uu = np.einsum("no,nojk->njk", n1, lL) + np.einsum("no,nojk->njk", (t - 1)[:, None] - n1, l1L)
            u.append(uu)
            mt.append(marg_log(uu + ll))
        prod[sl], true[sl] = osc(mp[0] - mp[1]), osc(mt[0] - mt[1])
        joint[sl] = osc((u[0] - u[1]).reshape(len(i), -1))
    return prod, joint, true


def build(dev=None):
    """the four agents' log beliefs (-> pairs_lq.npz), the pair set and every per-unit measure (-> pairs_v2.npz)"""
    t0 = time.time()
    P = probes()
    lq = exact_observers()
    checks = {f"float64 vs stored float32, {k}": float(np.abs(lq[k] - P[f"log_{s}"]).max())
              for k, s in (("bayes", "jb"), ("naive", "nb"))}
    lq.update(network_beliefs(dev))
    np.savez(LQ_CACHE, **lq)
    print(f"log beliefs ({time.time() - t0:.0f} s)", flush=True)
    I, J, S, n_eq = pair_set()
    out = dict(I=I, J=J, t=S, env=P["env"][I], fr=(P["fr"][I, S - 2] + P["fr"][J, S - 2]) / 2,
               ent=(P["ent"][I, S - 1] + P["ent"][J, S - 1]) / 2, n_equal_counts=n_eq)
    for k in lq:
        ms, cm_dev = pair_measures(lq[k], I, J, S)
        out.update({f"{nm}_{k}": v for nm, v in ms.items()})
        checks[f"common mode keeps d_H, {k}"] = cm_dev
    print(f"measures ({time.time() - t0:.0f} s)", flush=True)
    edges = make_edges(out)
    fe, fg = fr_thirds(out["fr"])
    for k in lq:
        gid = fg * (PAIR_BINS + 2) + unit_bins(out[f"pre_{k}"], edges) + 1
        out[f"shuf_{k}"], out[f"shufseed_{k}"] = pair_shuffle(lq[k], I, J, S, gid, PAIR_SEED + np.arange(N_SHUF))
    print(f"shuffles ({time.time() - t0:.0f} s)", flush=True)
    out["prod_bayes"], out["joint_bayes"], true = bayes_counterfactual(lq["bayes"], I, J, S)
    checks["true-joint path vs actual d_H after, bayes"] = float(np.abs(true - out["post_bayes"]).max())
    if os.path.exists(OLD_CACHE):
        Old = np.load(OLD_CACHE)
        same = all(np.array_equal(Old[a], b) for a, b in (("I", I), ("J", J), ("t", S)))
        checks["old pair set identical"] = float(same)
        if same:
            for k in lq:
                checks[f"vs old pairs.npz, {k}"] = float(max(np.abs(out[f"pre_{k}"] - Old[f"pre_{k}"]).max(),
                                                             np.abs(out[f"post_{k}"] - Old[f"post_{k}"]).max()))
    out.update(edges=edges, fr_edges=fe, check_names=np.array(list(checks)),
               check_values=np.array(list(checks.values())))
    np.savez(CACHE, **out)
    print(f"pairs: {len(I):,} pair-steps ({time.time() - t0:.0f} s); checks {checks}")


def load():
    return dict(np.load(CACHE))


# ═══════════════════════════════════════════════════════════════════
# Statistics: environment resamples, the FR comparison matched on step and distance before
# ═══════════════════════════════════════════════════════════════════

def _W():
    """(1 + N_BOOT, 50): the full sample (row 0), then td_tt_gap's shared environment resamples"""
    return np.vstack([np.ones((1, gap.N_ENV)), gap.boot_counts()])


def _ratio(v, m, env, W):
    """count-weighted mean of v over the units m (environment sums): (1 + N_BOOT,)"""
    s = np.bincount(env[m], v[m], gap.N_ENV)
    c = np.bincount(env[m], None, gap.N_ENV).astype(np.float64)
    return (W @ s) / (W @ c)


def _f(x, d=3):
    """value [5%, 95%] of a (1 + N_BOOT,) array"""
    return f"{x[0]:+.{d}f} [{np.percentile(x[1:], 5):+.{d}f}, {np.percentile(x[1:], 95):+.{d}f}]"


def _fs(x, d=3):
    """a share: value [5-95%]"""
    return f"{x[0]:.{d}f} [{np.percentile(x[1:], 5):.{d}f}-{np.percentile(x[1:], 95):.{d}f}]"


def _sg(x, d=2):
    """a signed number in mathtext (a true minus sign in figure text)"""
    return f"${x:+.{d}f}$"


def _fi(x, d=3):
    """value [5%, 95%] of a (1 + N_BOOT,) array, signed, in mathtext"""
    return f"{_sg(x[0], d)} [{_sg(np.percentile(x[1:], 5), d)}, {_sg(np.percentile(x[1:], 95), d)}]"


def matched(v, third, cell, ok_unit, env, W, n_min=N_MIN, need=None, K=3):
    """K groups (FR thirds by default) compared within cells (direct standardisation): per cell the mean of v in each
    group; cells with >= n_min units in each group of `need` (default: all K; full sample); cell means averaged over
    cells with the cell's unit count (all groups) as weight; in a resample, a cell missing one of those groups is left
    out of that resample.
    Returns dict(std (K, 1 + N_BOOT) standardised group means, hilo (1 + N_BOOT,) highest - lowest group, cells (G,)
    cell ids, ok (G,), mean (K, G) and weight (G,) of the full sample, share = units in the used cells / units, sw / cw
    (K, 1 + N_BOOT, G) the resampled sums and counts)"""
    c, th, e, vv = cell[ok_unit], third[ok_unit], env[ok_unit], v[ok_unit]
    cells, ci = np.unique(c, return_inverse=True)
    G = len(cells)
    S_, C_ = np.zeros((K, gap.N_ENV * G)), np.zeros((K, gap.N_ENV * G))
    for j in range(K):
        s = th == j
        S_[j] = np.bincount(e[s] * G + ci[s], vv[s], gap.N_ENV * G)
        C_[j] = np.bincount(e[s] * G + ci[s], None, gap.N_ENV * G)
    sw = np.einsum("be,jeg->jbg", W, S_.reshape(K, gap.N_ENV, G))
    cw = np.einsum("be,jeg->jbg", W, C_.reshape(K, gap.N_ENV, G))
    nd = list(range(K) if need is None else need)
    ok = (cw[nd, 0] >= n_min).all(0)
    good = ok[None] & (cw[nd] > 0).all(0)
    wt = np.where(good, cw.sum(0), 0.)
    with np.errstate(invalid="ignore", divide="ignore"):
        mn = np.where(cw > 0, sw / cw, 0.)
    std = (mn * wt[None]).sum(-1) / wt.sum(-1)[None]
    return dict(std=std, hilo=std[K - 1] - std[0], cells=cells, ok=ok, mean=mn[:, 0], weight=wt[0],
                share=float(wt[0].sum() / cw[:, 0].sum()), sw=sw, cw=cw)


def common_hilo(mA, mB, weights="sum"):
    """agent A's highest - lowest contrast minus agent B's, both standardised to one population: the cells usable for
    both (matched() with the same cells, groups and n_min), each weighted by the two agents' summed unit counts
    ('sum') or by B's ('B'), per resample. Returns (1 + N_BOOT,) and the number of common cells"""
    common = np.intersect1d(mA["cells"][mA["ok"]], mB["cells"][mB["ok"]])
    iA, iB = np.searchsorted(mA["cells"], common), np.searchsorted(mB["cells"], common)
    swA, cwA, swB, cwB = mA["sw"][:, :, iA], mA["cw"][:, :, iA], mB["sw"][:, :, iB], mB["cw"][:, :, iB]
    wt = cwB.sum(0) if weights == "B" else cwA.sum(0) + cwB.sum(0)
    wt = np.where((cwA > 0).all(0) & (cwB > 0).all(0), wt, 0.)
    out = []
    for sw, cw in ((swA, cwA), (swB, cwB)):
        with np.errstate(invalid="ignore", divide="ignore"):
            mn = np.where(cw > 0, sw / cw, 0.)
        std = (mn * wt[None]).sum(-1) / wt.sum(-1)[None]
        out.append(std[-1] - std[0])
    return out[0] - out[1], len(common)


def _units(Z, k):
    """flattened units (pair-step x variable): before, after"""
    return Z[f"pre_{k}"].ravel(), Z[f"post_{k}"].ravel()


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

def _violins(ax, groups, pos, width, color, rng, alpha=.55):
    """violins of the values in groups (list of arrays) at positions pos; KDE on at most KDE_MAX points per violin
    (seeded subsample); medians (all points) as white dots"""
    data, P_, med = [], [], []
    for g, x in zip(groups, pos):
        if len(g) < 50:
            continue
        data.append(g if len(g) <= KDE_MAX else rng.choice(g, KDE_MAX, replace=False))
        P_.append(x)
        med.append(np.median(g))
    if not data:
        return
    v = ax.violinplot(data, positions=P_, widths=width, showextrema=False, showmedians=False, bw_method="scott")
    for b_ in v["bodies"]:
        b_.set_facecolor(color)
        b_.set_edgecolor("none")
        b_.set_alpha(alpha)
    ax.plot(P_, med, "o", ms=3.2, mfc="w", mec="k", mew=.6, zorder=4)


def _mean_style(k, color):
    """the mean line: the agent's colour with a dark outline (Bayes: black with a white outline)"""
    import matplotlib.patheffects as pe
    return dict(color=color, lw=2., solid_capstyle="round", zorder=6,
                path_effects=[pe.Stroke(linewidth=3.6, foreground="w" if k == "bayes" else "k"), pe.Normal()])


def _null_style(color):
    import matplotlib.patheffects as pe
    return dict(color=color, lw=1.6, ls=(0, (3.5, 2)), zorder=7,
                path_effects=[pe.Stroke(linewidth=3., foreground="w"), pe.Normal()])


def _bin_means(v, b, sel, n_min=50):
    """per bin: the mean of v over the units sel in the bin (NaN for bins with < n_min units)"""
    s = np.bincount(b[sel], v[sel], PAIR_BINS)
    c = np.bincount(b[sel], None, PAIR_BINS)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c >= n_min, s / np.maximum(c, 1), np.nan)


def _xaxis(a_, le):
    from matplotlib.ticker import FixedLocator
    a_.set_xlim(le[0] - .05, le[-1] + .05)
    ticks = [x for x in (-1, 0, 1, 2) if le[0] - .05 <= x <= le[-1] + .05]
    a_.xaxis.set_major_locator(FixedLocator(ticks))
    a_.set_xticklabels([f"{10 ** x:g}" for x in ticks])
    a_.xaxis.set_minor_locator(FixedLocator([np.log10(m * 10 ** e) for e in (-1, 0, 1, 2) for m in range(2, 10)
                                             if le[0] - .05 <= np.log10(m * 10 ** e) <= le[-1] + .05]))
    a_.spines[["top", "right"]].set_visible(False)


def _mean_handle():
    """legend handle: the mean lines of the four agents side by side"""
    from matplotlib.lines import Line2D
    return tuple(Line2D([], [], **_mean_style(k, c)) for k, _, c in AGENTS)


def figure_stats(Z=None):
    """the numbers drawn in td_tt_chaos_pairs_detail.png's panel texts (units inside the 20 bins)"""
    Z = load() if Z is None else Z
    edges = Z["edges"]
    st = {}
    skn = (Z["skl_post_naive"] - Z["skl_pre_naive"]).ravel()
    for k, _, _ in AGENTS:
        pre, post = _units(Z, k)
        b = unit_bins(pre, edges)
        inb = (b >= 0) & (b < PAIR_BINS)
        dsk = (Z[f"skl_post_{k}"] - Z[f"skl_pre_{k}"]).ravel()
        st[k] = dict(n=int(inb.sum()), below=int((b < 0).sum()), above=int((b >= PAIR_BINS).sum()),
                     med=float(np.median(pre[inb])), q1=float(np.percentile(pre[inb], 25)),
                     q3=float(np.percentile(pre[inb], 75)), grow=float(np.mean(post[inb] > pre[inb] + 1e-9)),
                     shrink=float(np.mean(post[inb] < pre[inb] - 1e-9)), mean=float(np.mean((post - pre)[inb])),
                     shuf=float(np.mean((Z[f"shuf_{k}"].ravel() - pre)[inb])), skl=float(np.mean(dsk[inb])),
                     skl_cm=float(np.mean((Z[f"skl_post_{k}"] - Z[f"skl_cm_{k}"]).ravel()[inb])),
                     skl_naive=float(np.mean((dsk - skn)[inb])))
    return st


def figure():
    """John 11-02: the distance between two marginal beliefs before an identical observation (x, 20 bins) and its
    distribution after (violins), for Bayes, naive, the Trained RNN and the ESN; row 2: the change of d_H with the
    orientation-free shuffle; row 3: the change of the symmetric KL with the common-mode update
    (-> td_tt_chaos_pairs_detail.png; td_tt_chaos_pairs.png is John's combined figure, td_tt_chaos_combined.py)"""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    FL, TS = fs["label"], 12.
    Z = load()
    st = figure_stats(Z)
    edges = Z["edges"]
    le = np.log10(edges)
    cen, wid = (le[1:] + le[:-1]) / 2, (le[1] - le[0]) * .85
    rng = np.random.default_rng(PAIR_SEED)
    fig, ax = plt.subplots(3, 4, figsize=(19.5, 14.2), gridspec_kw=dict(hspace=.74, wspace=.26, left=.05, right=.985,
                                                                       top=.875, bottom=.165))
    lims = {1: [], 2: []}
    above = dict(xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points", ha="left", va="bottom",
                 fontsize=FL)                                  # rows 2-3: the panel text between the title and the axes
    pad = 36                                                   # (the violins' tails reach the top of these axes)
    skn = (Z["skl_post_naive"] - Z["skl_pre_naive"]).ravel()
    for col, (k, name, color) in enumerate(AGENTS):
        pre, post = _units(Z, k)
        b = unit_bins(pre, edges)
        inb = (b >= 0) & (b < PAIR_BINS)
        alpha = .7 if k == "bayes" else .55
        ch = post - pre
        dsk = (Z[f"skl_post_{k}"] - Z[f"skl_pre_{k}"]).ravel()
        cm = (Z[f"skl_cm_{k}"] - Z[f"skl_pre_{k}"]).ravel()
        s = st[k]
        # row 1: d_H after against d_H before
        a0 = ax[0, col]
        a0.plot([le[0], le[-1]], [le[0], le[-1]], color="k", lw=.9, ls=(0, (4, 3)), zorder=1)
        _violins(a0, [np.log10(post[inb & (b == i)]) for i in range(PAIR_BINS)], cen, wid, color, rng, alpha)
        a0.set_title(f"{'abcd'[col]}   {name}: distance after the observation\n"
                     r"$d_H(p_t, q_t)$ against $d_H(p_{t-1}, q_{t-1})$", loc="left", fontsize=TS)
        a0.text(.03, .97, f"median before {s['med']:.1f} nats\n(quartiles {s['q1']:.1f}-{s['q3']:.1f})",
                transform=a0.transAxes, ha="left", va="top", fontsize=FL)
        # row 2: change of d_H
        a1 = ax[1, col]
        a1.axhline(0, color="k", lw=.7, zorder=1)
        if k != "naive":
            _violins(a1, [ch[inb & (b == i)] for i in range(PAIR_BINS)], cen, wid, color, rng, alpha)
            a1.plot(cen, _bin_means(Z[f"shuf_{k}"].ravel() - pre, b, inb), **_null_style(SHUF_C))
            txt = (f"$d_H$ grows in {100 * s['grow']:.0f}%, shrinks in {100 * s['shrink']:.0f}%\n"
                   f"mean {_sg(s['mean'])} nats (shuffled {_sg(s['shuf'])})")
        else:
            txt = "unchanged in every pair (T1: the same\nobservation shifts both log beliefs alike)"
        a1.plot(cen, _bin_means(ch, b, inb), **_mean_style(k, color))
        a1.annotate(txt, **above)
        a1.set_title(f"{'efgh'[col]}   {name}: change of the distance\n"
                     r"$\Delta d_H = d_H(\mathrm{after}) - d_H(\mathrm{before})$", loc="left", fontsize=TS, pad=pad)
        lims[1].append(np.percentile(ch[inb], [1, 99]) if k != "naive" else np.zeros(2))
        # row 3: change of the symmetric KL
        a2 = ax[2, col]
        a2.axhline(0, color="k", lw=.7, zorder=1)
        _violins(a2, [dsk[inb & (b == i)] for i in range(PAIR_BINS)], cen, wid, color, rng, alpha)
        a2.plot(cen, _bin_means(cm, b, inb), **_null_style(CM_C))
        if k != "naive":
            a2.plot(cen, _bin_means(skn, b, inb), color="0.55", lw=1.5, zorder=5)
            txt = (f"mean {_sg(s['skl'])} nats; beyond the\ncommon mode {_sg(s['skl_cm'])}, "
                   f"beyond naive {_sg(s['skl_naive'])}")
        else:
            txt = f"mean {_sg(s['skl'])} nats (its update is\nits common mode: T1)"
        a2.plot(cen, _bin_means(dsk, b, inb), **_mean_style(k, color))
        a2.annotate(txt, **above)
        a2.set_title(f"{'ijkl'[col]}   {name}: change of the symmetric KL\n"
                     r"$\Delta\mathrm{SKL}$, $\mathrm{SKL}(p,q) = \Sigma_r\,(p_r - q_r)\,\log(p_r/q_r)\,/\,2$",
                     loc="left", fontsize=TS, pad=pad)
        lims[2].append(np.percentile(dsk[inb], [1, 99]))
    for r_ in (1, 2):
        lo, hi = min(x[0] for x in lims[r_]), max(x[1] for x in lims[r_])
        for col in range(4):
            ax[r_, col].set_ylim(lo - .1 * (hi - lo), hi + .1 * (hi - lo))
    for col in range(4):
        for r_ in range(3):
            _xaxis(ax[r_, col], le)
            ax[r_, col].set_xlabel(r"distance before, $d_H(p_{t-1}, q_{t-1})$ (20 bins)", fontsize=FL)
        ax[0, col].set_ylim(le[0] - .35, le[-1] + .55)
        yt = [x for x in (-1, 0, 1, 2) if le[0] - .35 <= x <= le[-1] + .55]
        ax[0, col].yaxis.set_major_locator(FixedLocator(yt))
        ax[0, col].set_yticklabels([f"{10 ** x:g}" for x in yt])
        ax[0, col].yaxis.set_minor_locator(NullLocator())
    ax[0, 0].set_ylabel(r"distance after, $d_H$ (nats)", fontsize=FL)
    ax[1, 0].set_ylabel(r"$\Delta d_H$ (nats)", fontsize=FL)
    ax[2, 0].set_ylabel(r"$\Delta\mathrm{SKL}$ (nats)", fontsize=FL)
    hd = [_mean_handle(), Line2D([], [], ls="none", marker="o", ms=4.5, mfc="w", mec="k", mew=.6),
          Line2D([], [], color="k", lw=.9, ls=(0, (4, 3))), Line2D([], [], **_null_style(SHUF_C)),
          Line2D([], [], **_null_style(CM_C)), Line2D([], [], color="0.55", lw=1.5)]
    lb = ["mean over the units in the bin (Bayes, naive, Trained RNN, ESN)", "median",
          "row 1: unchanged (after = before)",
          "row 2: shuffled: the step's difference of another unit\n(same variable, FR third and bin), random "
          "orientation",
          "row 3: common mode: both log beliefs shifted by the pair's\nmean log-update (leaves $d_H$ unchanged)",
          "row 3: naive on the same pairs"]
    fig.legend(hd, lb, loc="lower center", bbox_to_anchor=(.5, .075), ncol=3, frameon=False, fontsize=FL + .5,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=.4)}, handlelength=5.5, columnspacing=2.5)
    fig.suptitle("Pairs of marginal beliefs receive the same observation vector: does the range of their log-ratio "
                 "grow, and do they separate in bulk?\n"
                 r"$d_H(p, q) = \max_r\,\log(p_r / q_r) \;-\; \min_r\,\log(p_r / q_r)$ per variable (the range of "
                 "the log-ratio; it can be set by values that both beliefs treat as near-impossible)",
                 fontsize=TS + 1, y=.985, linespacing=1.6)
    n_ = len(Z["I"]) * NV
    cnt = ", ".join(f"{name} {st[k]['n']:,}" for k, name, _ in AGENTS)
    fig.text(.5, .012,
             f"Data: 50 test environments x 200 probes (100 cells x 2 streams of 30 steps). Pairs: the same "
             f"environment and step t = 2-30, the identical 5-observation vector at t, different observation counts "
             f"before t:\n"
             f"{len(Z['I']):,} pair-steps x 2 variables = {n_:,} units per agent. Drawn, and in the panel texts: the "
             f"units inside the 20 bins (pooled 0.1st-99.9th percentiles of the distance before, "
             f"{edges[0]:.2f}-{edges[-1]:.0f} nats):\n{cnt}. Violins: KDE on at most {KDE_MAX:,} units per bin "
             f"(seeded subsample); medians and means: all units in the bin; shuffled: mean of {N_SHUF} seeds.\n"
             "Bayes = the joint posterior's marginals; Trained RNN and ESN: one training seed each, end of training "
             "(39,996 episodes), fp32 replays.", ha="center", va="bottom", fontsize=FL, linespacing=1.5)
    path = fig_path("td_tt_chaos_pairs_detail.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


def fr_stats(Z=None):
    """per agent and FR third, the units inside the 20 bins (grows, mean change, shuffled, count); per agent, the
    matched comparison (step and bin cells). These were the panel texts of the FR figure (td_tt_chaos_pairs_fr.png),
    dropped on 11-02 (John: split by the observation's entanglement instead); numbers() still prints them"""
    Z = load() if Z is None else Z
    edges = Z["edges"]
    _, fg = fr_thirds(Z["fr"])
    fg = fg.ravel()
    t_u, env = np.repeat(Z["t"], NV), np.repeat(Z["env"], NV)
    W = _W()
    st = {}
    for k, _, _ in AGENTS:
        pre, post = _units(Z, k)
        b = unit_bins(pre, edges)
        inb = (b >= 0) & (b < PAIR_BINS)
        ch, sh = post - pre, Z[f"shuf_{k}"].ravel() - pre
        for r_ in range(3):
            sel = inb & (fg == r_)
            st[(k, r_)] = dict(grow=float(np.mean(post[sel] > pre[sel] + 1e-9)), mean=float(np.mean(ch[sel])),
                               shuf=float(np.mean(sh[sel])), n=int(sel.sum()))
        mt = matched(ch, fg, t_u * PAIR_BINS + b, inb, env, W)
        raw = [_ratio(ch, inb & (fg == j), env, W) for j in (0, 2)]
        st[k] = dict(hilo=mt["hilo"], raw=raw[1] - raw[0], share=mt["share"])
    return st


# ═══════════════════════════════════════════════════════════════════
# The combined figure's data (John 11-02): 40 bins, three definitions of the distance, the split by the observation's
# entanglement (-> pairs_combined.npz)
# ═══════════════════════════════════════════════════════════════════

def edges40(Z):
    """NBIN40 log-spaced bins of d_H before, shared by the agents: the pooled (four agents, all units) 0.1st-99th
    percentiles"""
    pooled = np.concatenate([Z[f"pre_{k}"].ravel() for k in COMB_AGENTS])
    lo, hi = np.percentile(pooled, EDGE40_PCT)
    return np.logspace(np.log10(lo), np.log10(hi), NBIN40 + 1)


def qgroups(x, K):
    """quantile edges (K + 1,) of x and the group 0..K-1 of each value"""
    e = np.quantile(x, np.linspace(0, 1, K + 1))
    return e, np.clip(np.searchsorted(e, x, side="right") - 1, 0, K - 1)


def cond_groups(x, cell, K=OTH_K):
    """the group 0..K-1 of each value of x within its cell: the values of a cell ranked and cut into K near-equal parts
    (sizes differ by at most one)"""
    order = np.lexsort((x, cell))
    cs = cell[order]
    starts = np.r_[0, np.flatnonzero(np.diff(cs)) + 1]
    sizes = np.diff(np.r_[starts, len(cs)])
    rank = np.arange(len(cs)) - np.repeat(starts, sizes)
    g = np.empty(len(x), np.int64)
    g[order] = (K * rank) // np.repeat(sizes, sizes)
    return g


def unit_measures(Z, k, sh):
    """per unit (flattened (m * 2,)) agent k's measures MEAS: dh = Delta d_H; shuf = the shuffled Delta d_H (sh: its
    shuffled d_H after, (m, 2)); sym / aligned = the sign-flip parts; skl = Delta SKL; skl_cm = the common-mode update's
    Delta SKL; skl_naive = naive's Delta SKL on the same pair-steps; skl_bn = skl - skl_naive (beyond naive); skl_bcm =
    SKL after - SKL after the common-mode update (beyond the common mode)"""
    pre, post, fl = (Z[f"{a}_{k}"].ravel() for a in ("pre", "post", "flip"))
    s0, s1, scm = (Z[f"skl_{a}_{k}"].ravel() for a in ("pre", "post", "cm"))
    skn = (Z["skl_post_naive"] - Z["skl_pre_naive"]).ravel()
    return dict(dh=post - pre, shuf=sh.ravel() - pre, sym=(post + fl) / 2 - pre, aligned=(post - fl) / 2,
                skl=s1 - s0, skl_cm=scm - s0, skl_naive=skn, skl_bn=(s1 - s0) - skn, skl_bcm=s1 - scm)


def bin_stats(v, b, ok, env, W, nb, grp=None, K=1):
    """per group and bin: the count-weighted mean of v over the units ok in the full sample and every resample
    (1 + N_BOOT, K, nb), and the full-sample counts (K, nb)"""
    g = np.zeros(len(v), np.int64) if grp is None else grp
    key = (env[ok] * K + g[ok]) * nb + b[ok]
    S = W @ np.bincount(key, v[ok], gap.N_ENV * K * nb).reshape(gap.N_ENV, K * nb)
    C = W @ np.bincount(key, None, gap.N_ENV * K * nb).reshape(gap.N_ENV, K * nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = S / C
    return r.reshape(-1, K, nb), C[0].reshape(K, nb)


def step_lines(v, b, grp, ok, env, t_u, W, K, nb, n_min=N_MIN):
    """per bin: the K groups' means standardised over steps within the bin (the matched estimator per bin): cells
    (t, bin) with >= n_min units in every group (full sample); each cell's group means weighted by the cell's unit
    count (all groups; per resample, a cell missing a group there is left out of it). Returns (1 + N_BOOT, K, nb) and
    the units in the used cells per bin (nb,)"""
    n_c = (T + 1) * nb * K
    key = ((env[ok] * (T + 1) + t_u[ok]) * nb + b[ok]) * K + grp[ok]
    S = (W @ np.bincount(key, v[ok], gap.N_ENV * n_c).reshape(gap.N_ENV, n_c)).reshape(-1, T + 1, nb, K)
    C = (W @ np.bincount(key, None, gap.N_ENV * n_c).reshape(gap.N_ENV, n_c)).reshape(-1, T + 1, nb, K)
    use = (C[0] >= n_min).all(-1)
    wt = np.where(use[None] & (C > 0).all(-1), C.sum(-1), 0.)
    with np.errstate(invalid="ignore", divide="ignore"):
        mn = np.where(C > 0, S / np.maximum(C, 1e-300), 0.)
        lines = (mn * wt[..., None]).sum(1) / wt.sum(1)[..., None]
    return lines.transpose(0, 2, 1), wt[0].sum(0)


def violin_stats(x, rng, n_pts=100):
    """matplotlib's violin statistics: a Gaussian KDE (Scott's rule) on at most KDE_MAX values (seeded subsample),
    evaluated at n_pts points from their minimum to their maximum; the median of all values. NaN below 50 values"""
    from scipy.stats import gaussian_kde
    if len(x) < 50 or np.ptp(x) < 1e-9:
        return np.full(n_pts, np.nan), np.full(n_pts, np.nan), np.nan
    xs = x if len(x) <= KDE_MAX else rng.choice(x, KDE_MAX, replace=False)
    y = np.linspace(xs.min(), xs.max(), n_pts)
    return y, gaussian_kde(xs, "scott")(y), float(np.median(x))


def _pct(r):
    """full sample, 5% and 95% over the resamples of (1 + N_BOOT, ...) (NaN where every resample is NaN)"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return r[0], np.nanpercentile(r[1:], 5, 0), np.nanpercentile(r[1:], 95, 0)


def combined():
    """John 11-02: the data of the combined pairs figure (-> pairs_combined.npz; the per-unit shuffle and groups ->
    pairs_shuf40.npz): 40 bins of d_H before; three definitions of the distance (per variable, the pair's average,
    the other variable's distance); the split by the observation's entanglement; matched contrasts and checks"""
    t0 = time.time()
    Z, LQ = load(), np.load(LQ_CACHE)
    P = probes()
    I, J, S = Z["I"], Z["J"], Z["t"]
    m = len(I)
    W = _W()
    env, t_u = np.repeat(Z["env"], NV), np.repeat(S, NV)
    e40 = edges40(Z)
    nb = NBIN40
    ep = Z["ent"].mean(1)                                                  # whole belief: variables, then probes
    e5, g5p = qgroups(ep, ENT_K)
    e3, g3p = qgroups(ep, 3)
    g5, g3 = np.repeat(g5p, NV), np.repeat(g3p, NV)
    own5 = qgroups(Z["ent"].ravel(), ENT_K)[1]
    oth5 = qgroups(Z["ent"][:, ::-1].ravel(), ENT_K)[1]
    fe, fg = fr_thirds(Z["fr"])
    fg = fg.ravel()
    oe = np.quantile(np.concatenate([Z[f"pre_{k}"].ravel() for k in COMB_AGENTS]), [0, 1 / 3, 2 / 3, 1])
    band = np.digitize(t_u, [11, 21])
    chk = {"ent: stored pair means vs probes.npz 'ent' at step t": float(np.abs(
        Z["ent"] - (P["ent"][I, S - 1] + P["ent"][J, S - 1]) / 2).max())}
    A, M_, G3, G5 = len(COMB_AGENTS), len(MEAS), 3, ENT_K
    out = dict(edges40=e40, centres40=np.sqrt(e40[1:] * e40[:-1]), agents=np.array(COMB_AGENTS),
               names=np.array([nm for _, nm, _ in AGENTS]), colors=np.array([c for _, _, c in AGENTS]),
               meas=np.array(MEAS), vio_meas=np.array(VIO), ent5_edges=e5, ent3_edges=e3, oth_edges=oe,
               fr_edges=fe, n_pairsteps=m, n_units=m * NV, n_boot=gap.N_BOOT,
               shuf_seeds=SHUF40_SEED + np.arange(N_SHUF), n_min=N_MIN, line_min=LINE_MIN,
               edge_pct=np.array(EDGE40_PCT))
    for d in ("var", "avg"):
        out.update({f"{d}_{s}": np.full((A, M_, nb), np.nan) for s in ("mean", "lo", "hi")})
        out[f"{d}_n"] = np.zeros((A, nb), np.int64)
        out[f"{d}_out"] = np.zeros((A, 2), np.int64)
    for d, K in (("oth", G3), ("cond", G3), ("ent", G5), ("entraw", G5)):
        out.update({f"{d}_{s}": np.full((A, M_, K, nb), np.nan) for s in ("mean", "lo", "hi")})
        out[f"{d}_n"] = np.zeros((A, K, nb), np.int64)
    out["ent_used"] = np.zeros((A, nb))
    out["cond_med_other"] = np.full((A, G3, nb), np.nan)
    out["map_sum"], out["map_n"] = np.zeros((A, nb, nb)), np.zeros((A, nb, nb), np.int64)
    out.update({f"vio_{s}": np.full((A, len(VIO), nb, 100), np.nan) for s in ("y", "d")})
    out["vio_med"] = np.full((A, len(VIO), nb), np.nan)
    CMEAS = ("dh", "dh-shuf", "aligned", "sym", "skl", "skl_bn", "skl_bcm")
    out["cmeas"] = np.array(CMEAS)
    out["ent5_std"] = np.full((A, len(CMEAS), G5, 1 + gap.N_BOOT), np.nan)
    out["cond3_std"] = np.full((A, len(CMEAS), G3, 1 + gap.N_BOOT), np.nan)
    out["shuf_seed_means"] = np.full((A, 3, N_SHUF), np.nan)                 # (main, pooled other third, pair average)
    stats, covers, shuf_store = {}, {}, {}
    mt_keep = {}

    def put(name, x, cover=np.nan):
        stats[name], covers[name] = np.asarray(x, np.float64), float(cover)

    rng = np.random.default_rng(PAIR_SEED + 2000)
    for ia, k in enumerate(COMB_AGENTS):
        pre, post = _units(Z, k)
        oth = Z[f"pre_{k}"][:, ::-1].ravel()
        b = unit_bins(pre, e40)
        inb = (b >= 0) & (b < nb)
        cg = cond_groups(oth, t_u * (nb + 2) + b + 1)
        gid = ((g5 * OTH_K + cg) * (nb + 2) + b + 1).reshape(m, NV)
        sh, per_seed = pair_shuffle(LQ[k], I, J, S, gid, SHUF40_SEED + np.arange(N_SHUF))
        og = np.clip(np.searchsorted(oe, oth, side="right") - 1, 0, 2)
        pa = Z[f"pre_{k}"].mean(1)
        ba = unit_bins(pa, e40)
        inba = (ba >= 0) & (ba < nb)
        # each split's shuffle draws its donors from its own groups: the pooled thirds of the other variable's distance;
        # the pair average (donor pair-steps of the same entanglement fifth and bin of the pair average)
        sho, seed_o = pair_shuffle(LQ[k], I, J, S, ((g5 * OTH_K + og) * (nb + 2) + b + 1).reshape(m, NV),
                                   SHUF40_SEED + 10 + np.arange(N_SHUF))
        sha, seed_a = pair_shuffle(LQ[k], I, J, S, np.repeat((g5p * (nb + 2) + ba + 1)[:, None], NV, 1),
                                   SHUF40_SEED + 20 + np.arange(N_SHUF))
        shuf_store.update({f"shuf_{k}": sh, f"shufo_{k}": sho, f"shufa_{k}": sha,
                           f"cond_{k}": cg.reshape(m, NV).astype(np.int8)})
        out["shuf_seed_means"][ia] = per_seed, seed_o, seed_a
        csz = np.bincount((t_u * (nb + 2) + b + 1) * OTH_K + cg,
                          minlength=(T + 1) * (nb + 2) * OTH_K).reshape(-1, OTH_K)
        chk[f"conditional thirds: max size spread within a cell, {k}"] = float((csz.max(1) - csz.min(1)).max())
        U = unit_measures(Z, k, sh)
        sh_o, sh_a = sho.ravel() - pre, sha.ravel() - pre
        out["var_n"][ia] = np.bincount(b[inb], minlength=nb)
        out["var_out"][ia] = [(b < 0).sum(), (b >= nb).sum()]
        out["avg_n"][ia] = np.bincount(ba[inba], minlength=nb)
        out["avg_out"][ia] = [(ba < 0).sum(), (ba >= nb).sum()]
        for im, nm in enumerate(MEAS):
            v = U[nm]
            va = (sh_a if nm == "shuf" else v).reshape(m, NV).mean(1)
            for d, r in (("var", bin_stats(v, b, inb, env, W, nb)[0][:, 0]),
                         ("avg", bin_stats(va, ba, inba, Z["env"], W, nb)[0][:, 0])):
                out[f"{d}_mean"][ia, im], out[f"{d}_lo"][ia, im], out[f"{d}_hi"][ia, im] = _pct(r)
            for d, grp, K in (("oth", og, G3), ("cond", cg, G3), ("entraw", g5, G5)):
                r, n = bin_stats(sh_o if (d == "oth" and nm == "shuf") else v, b, inb, env, W, nb, grp, K)
                out[f"{d}_mean"][ia, im], out[f"{d}_lo"][ia, im], out[f"{d}_hi"][ia, im] = _pct(r)
                out[f"{d}_n"][ia] = n
            r, used = step_lines(v, b, g5, inb, env, t_u, W, G5, nb)
            if nm == "dh":
                with np.errstate(invalid="ignore"):
                    lines_check = (np.nan_to_num(r[0]) * used).sum(-1) / used.sum()   # all bins, before the mask
            r[:, :, used < LINE_MIN] = np.nan
            out["ent_mean"][ia, im], out["ent_lo"][ia, im], out["ent_hi"][ia, im] = _pct(r)
            out["ent_used"][ia] = used
        out["ent_n"][ia] = out["entraw_n"][ia]
        for g in range(G3):
            for i in range(nb):
                s = inb & (b == i) & (cg == g)
                if s.sum() >= 50:
                    out["cond_med_other"][ia, g, i] = np.median(oth[s])
        bo = unit_bins(oth, e40)
        both = inb & (bo >= 0) & (bo < nb)
        out["map_sum"][ia] = np.bincount(b[both] * nb + bo[both], U["dh"][both], nb * nb).reshape(nb, nb)
        out["map_n"][ia] = np.bincount(b[both] * nb + bo[both], None, nb * nb).reshape(nb, nb)
        for iv, x in enumerate((np.log10(np.maximum(post, 1e-300)), U["dh"], U["skl"])):
            if VIO[iv] == "dh" and k == "naive":
                continue
            for i in range(nb):
                out["vio_y"][ia, iv, i], out["vio_d"][ia, iv, i], out["vio_med"][ia, iv, i] = \
                    violin_stats(x[inb & (b == i)], rng)
        print(f"  {k}: curves ({time.time() - t0:.0f} s)", flush=True)
        if k == "naive":
            continue
        # summaries with intervals (all units; the closest pairs under two definitions)
        allm = np.ones(m * NV, bool)
        for nm in ("dh", "shuf", "sym", "aligned"):
            put(f"all|{nm}|{k}", _ratio(U[nm], allm, env, W))
        put(f"all|dh-shuf|{k}", _ratio(U["dh"] - U["shuf"], allm, env, W))
        for lab, lo_, hi_ in (("<1", 0, 1), ("1-3", 1, 3)):
            sv = (pre >= lo_) & (pre < hi_)
            sa = (pa >= lo_) & (pa < hi_)
            for nm in ("dh", "shuf", "sym", "aligned"):
                put(f"close|var{lab}|{nm}|{k}", _ratio(U[nm], sv, env, W), sv.sum())
                put(f"close|avg{lab}|{nm}|{k}", _ratio((sh_a if nm == "shuf" else U[nm]).reshape(m, NV).mean(1), sa,
                                                       Z["env"], W), sa.sum())
        # matched contrasts: highest - lowest group
        tb = t_u * nb + b
        cmeas = {"dh": U["dh"], "dh-shuf": U["dh"] - U["shuf"], "aligned": U["aligned"], "sym": U["sym"],
                 "skl": U["skl"], "skl_bn": U["skl_bn"], "skl_bcm": U["skl_bcm"]}
        for im, (nm, v) in enumerate(cmeas.items()):
            raw = [_ratio(v, inb & (g5 == j), env, W) for j in (0, G5 - 1)]
            put(f"ent5|{nm}|raw|{k}", raw[1] - raw[0], 1.)
            for lab, cell in (("bin", b), ("step+bin", tb), ("step+bin+fr", tb * 3 + fg)):
                mt = matched(v, g5, cell, inb, env, W, K=G5)
                put(f"ent5|{nm}|{lab}|{k}", mt["hilo"], mt["share"])
                if lab == "step+bin":
                    out["ent5_std"][ia, im] = mt["std"]
                    if nm == "dh":
                        chk[f"step lines reproduce the matched fifths' means, {k}"] = float(
                            np.abs(lines_check - mt["std"][:, 0]).max())
                    if k in ("bayes", "trained"):
                        mt_keep[("ent5", nm, k)] = mt
            mt = matched(v, cg, tb, inb, env, W, K=G3)
            put(f"cond3|{nm}|step+bin|{k}", mt["hilo"], mt["share"])
            out["cond3_std"][ia, im] = mt["std"]
            if k in ("bayes", "trained"):
                mt_keep[("cond", nm, k)] = mt
        v = U["dh"]
        for lab, grp, K, cell, nmin in (("ent3|dh|step+bin", g3, 3, tb, N_MIN),
                                        ("ent5own|dh|step+bin", own5, G5, tb, N_MIN),
                                        ("ent5other|dh|step+bin", oth5, G5, tb, N_MIN),
                                        ("ent5|dh|band+bin+env", g5, G5, (band * nb + b) * gap.N_ENV + env, N_MIN),
                                        ("fr3|dh|step+bin", fg, 3, tb, N_MIN),
                                        ("fr3|dh|step+bin+ent5", fg, 3, tb * G5 + g5, N_MIN),
                                        ("ent5|dh|step+bin+cond3", g5, G5, tb * 3 + cg, N_MIN),
                                        ("cond3|dh|step+bin+ent5", cg, 3, tb * G5 + g5, N_MIN),
                                        ("cond3|dh|step+bin+fr", cg, 3, tb * 3 + fg, N_MIN),
                                        ("oth3|dh|step+bin", og, 3, tb, N_MIN)):
            mt = matched(v, grp, cell, inb, env, W, n_min=nmin, K=K)
            put(f"{lab}|{k}", mt["hilo"], mt["share"])
        raw = [_ratio(v, inb & (og == j), env, W) for j in (0, 2)]
        put(f"oth3|dh|raw|{k}", raw[1] - raw[0], 1.)
        print(f"  {k}: contrasts ({time.time() - t0:.0f} s)", flush=True)
    for split in ("ent5", "cond"):
        for nm in ("dh", "dh-shuf", "aligned", "sym", "skl", "skl_bn", "skl_bcm"):
            x, nc = common_hilo(mt_keep[(split, nm, "trained")], mt_keep[(split, nm, "bayes")])
            put(f"{'ent5' if split == 'ent5' else 'cond3'}|{nm}|step+bin|trained-bayes", x, nc)
        x, nc = common_hilo(mt_keep[(split, "dh", "trained")], mt_keep[(split, "dh", "bayes")], "B")
        put(f"{'ent5' if split == 'ent5' else 'cond3'}|dh|step+bin|trained-bayes-Bweights", x, nc)
    # check: the pair average = half the d_H between the products of the two marginals (100 cells), on 20,000 pair-steps
    sel = np.random.default_rng(PAIR_SEED + 3000).choice(m, 20000, replace=False)
    for k in ("bayes", "trained"):
        a, b_ = LQ[k][I[sel], S[sel] - 2], LQ[k][J[sel], S[sel] - 2]
        D = a - b_
        prod = osc((D[:, 0, :, None] + D[:, 1, None, :]).reshape(len(sel), -1))
        chk[f"pair average = d_H(products) / 2, {k}"] = float(np.abs(prod / 2 - Z[f"pre_{k}"][sel].mean(1)).max())
    names = sorted(stats)
    out.update(stat_names=np.array(names), stat_boot=np.stack([stats[n] for n in names]),
               stat_aux=np.array([covers[n] for n in names]), chk_names=np.array(list(chk)),
               chk_values=np.array(list(chk.values()), np.float64))
    np.savez(COMB_CACHE, **out)
    np.savez(SHUF40_CACHE, g5=g5p.astype(np.int8), **shuf_store)
    print(f"saved {COMB_CACHE} and {SHUF40_CACHE} ({time.time() - t0:.0f} s); checks {chk}")


# ═══════════════════════════════════════════════════════════════════
# Numbers (every number of the docstring)
# ═══════════════════════════════════════════════════════════════════

def numbers():
    Z = load()
    LQ = np.load(LQ_CACHE)
    W = _W()
    m = len(Z["I"])
    env = np.repeat(Z["env"], NV)
    t_u = np.repeat(Z["t"], NV)
    fe, fg = fr_thirds(Z["fr"])
    fg = fg.ravel()
    edges = Z["edges"]
    allm = np.ones(m * NV, bool)
    third = [fg == j for j in range(3)]
    print(f"== data: {m:,} pair-steps x 2 variables = {m * NV:,} units per agent; same-observation pairs left out for "
          f"equal counts {int(Z['n_equal_counts']):,}; distinct probe pairs "
          f"{len(np.unique(Z['I'] * 10000 + Z['J'])):,}")
    for k, name, _ in AGENTS:
        p = Z[f"pre_{k}"].ravel()
        print(f"  d_H before, {name:11s}: median {np.median(p):.1f} (quartiles {np.percentile(p, 25):.1f}-"
              f"{np.percentile(p, 75):.1f}), min {p.min():.3f}, max {p.max():.1f}; share < 1 nat {np.mean(p < 1):.3f}, "
              f"> 10 nats {np.mean(p > 10):.3f}; mean by FR third " +
              " / ".join(f"{p[th].mean():.1f}" for th in third))
    _tails(Z, LQ)
    print(f"  bins: {edges[0]:.3f}-{edges[-1]:.1f} nats (pooled percentiles {EDGE_PCT[0]} to {EDGE_PCT[1]})")
    print(f"  FR thirds {np.round(fe, 3)} nats, units {[int(th.sum()) for th in third]}; mean step per third " +
          " / ".join(f"{t_u[th].mean():.1f}" for th in third))
    print("== checks")
    for nm, v in zip(Z["check_names"], Z["check_values"]):
        print(f"  {nm}: {v:.2g}")
    print(f"  naive: max |d_H after - before| {np.abs(Z['post_naive'] - Z['pre_naive']).max():.2g}")
    print("== change of d_H (all units)")
    for k, name, _ in AGENTS:
        if k == "naive":
            continue
        pre, post = _units(Z, k)
        close = pre < 1
        print(f"  {name:11s}: grows {_fs(_ratio((post > pre + 1e-9).astype(float), allm, env, W))}, shrinks "
              f"{np.mean(post < pre - 1e-9):.3f}; mean change {_f(_ratio(post - pre, allm, env, W))}; closest pairs "
              f"(before < 1 nat, {close.mean():.4f} of units): mean change {_f(_ratio(post - pre, close, env, W))}, "
              f"grows {np.mean(post[close] > pre[close] + 1e-9):.3f}, shuffled "
              f"{np.mean((Z[f'shuf_{k}'].ravel() - pre)[close]):+.3f}, actual - shuffled "
              f"{_f(_ratio(post - Z[f'shuf_{k}'].ravel(), close, env, W))}")
    _closest(Z, W, env)
    print(f"== shuffled (orientation-free; groups: variable x FR third x bin, out-of-range units in 2 own bins; "
          f"{N_SHUF} seeds)")
    for k, name, _ in AGENTS:
        if k == "naive":
            continue
        pre, post = _units(Z, k)
        sh = Z[f"shuf_{k}"].ravel()
        fl = Z[f"flip_{k}"].ravel()
        print(f"  {name:11s}: shuffled change {_f(_ratio(sh - pre, allm, env, W))} (seeds "
              f"{Z[f'shufseed_{k}'].min():+.3f} to {Z[f'shufseed_{k}'].max():+.3f}); actual - shuffled "
              f"{_f(_ratio(post - sh, allm, env, W))}; by FR third: shuffled " +
              " / ".join(f"{_ratio(sh - pre, th, env, W)[0]:+.3f}" for th in third) + ", actual - shuffled " +
              " / ".join(_f(_ratio(post - sh, th, env, W)) for th in third))
        sym = (post + fl) / 2 - pre
        print(f"  {'':11s}  sign flip (an exact split of each unit's change; no permutation): aligned part "
              f"(osc(D + delta) - osc(D - delta)) / 2 {_f(_ratio((post - fl) / 2, allm, env, W))}, symmetric part "
              f"(osc(D + delta) + osc(D - delta)) / 2 - osc(D) {_f(_ratio(sym, allm, env, W))} (min {sym.min():+.1e});"
              f" share osc(D + delta) > osc(D - delta) {np.mean(post > fl + 1e-9):.3f}")
    if os.path.exists(OLD_CACHE):
        Old = np.load(OLD_CACHE)
        print("  the old oriented null (td_tt_chaos M4, pairs.npz: no random orientation, units outside the 1st-99th "
              "percentile bins in the end bins): " + "; ".join(
                  f"{name} shuffled {np.mean(Old[f'shuf_{k}'] - Old[f'pre_{k}']):+.3f}, actual - shuffled "
                  f"{np.mean(Old[f'post_{k}'] - Old[f'shuf_{k}']):+.3f}" for k, name, _ in AGENTS if k != "naive"))
    _robust_nulls(Z, W, env, LQ)
    print("== bulk divergences (SKL, JS): change; beyond naive (same pairs); beyond the agent's own common mode")
    for meas in ("skl", "js"):
        d = 3 if meas == "skl" else 4
        cn = (Z[f"{meas}_post_naive"] - Z[f"{meas}_pre_naive"]).ravel()
        for k, name, _ in AGENTS:
            a0, a1, acm = (Z[f"{meas}_{x}_{k}"].ravel() for x in ("pre", "post", "cm"))
            ch = a1 - a0
            line = (f"  {meas.upper():3s} {name:11s}: before mean {a0.mean():.{d}f} (median {np.median(a0):.{d}f}); "
                    f"change {_f(_ratio(ch, allm, env, W), d)}, grows {np.mean(a1 > a0):.3f}")
            if k == "naive":
                print(line + f"; naive minus its common mode: max |d| {np.abs(a1 - acm).max():.2g}")
                continue
            print(line + "; by FR third " + " / ".join(f"{_ratio(ch, th, env, W)[0]:+.{d}f}" for th in third))
            for lab, v in (("beyond naive", ch - cn), ("beyond common mode", a1 - acm)):
                print(f"      {lab:18s} {_f(_ratio(v, allm, env, W), d)}; above {np.mean(v > 0):.3f}; by FR third " +
                      " / ".join(f"{_ratio(v, th, env, W)[0]:+.{d}f}" for th in third))
    print("== Bayes: the coupling counterfactual and the joint distance (T2)")
    pre, post = _units(Z, "bayes")
    pp = Z["prod_bayes"].ravel()
    print(f"  product of marginals (same marginals, no coupling): change {_f(_ratio(pp - pre, allm, env, W))}, grows "
          f"{np.mean(pp > pre + 1e-9):.3f}; by FR third " + " / ".join(f"{_ratio(pp - pre, th, env, W)[0]:+.3f}"
                                                                    for th in third))
    print(f"  actual - product (the coupling's part) {_f(_ratio(post - pp, allm, env, W))}; by FR third " +
          " / ".join(f"{_ratio(post - pp, th, env, W)[0]:+.3f}" for th in third) +
          f"; r(actual change, product change) over units {np.corrcoef(post - pre, pp - pre)[0, 1]:.3f}")
    for lab, mk in (("d_H before < 1 nat", pre < 1), ("< 2 nats", pre < 2), ("2-10 nats", (pre >= 2) & (pre < 10)),
                    ("10-50 nats", (pre >= 10) & (pre < 50)), (">= 50 nats", pre >= 50)):
        print(f"  the coupling's part, {lab} ({mk.mean():.4f} of units): {_f(_ratio(post - pp, mk, env, W))}")
    tail = tail_flags(Z, LQ, "bayes").ravel()
    print(f"  the coupling's part, d_H before set by near-impossible values ({tail.mean():.3f} of units): "
          f"{_f(_ratio(post - pp, tail, env, W))}; the other units {_f(_ratio(post - pp, ~tail, env, W))}")
    for j, th in enumerate(third):
        print(f"    FR third {j + 1}: all {_f(_ratio(post - pp, th, env, W))}; set by near-impossible values "
              f"{_f(_ratio(post - pp, th & tail, env, W))}; others {_f(_ratio(post - pp, th & ~tail, env, W))}")
    jd = Z["joint_bayes"]
    gapj = jd[:, None] - Z["post_bayes"]
    print(f"  T2: max (marginal after - joint) {-gapj.min():+.2g}; joint d_H mean {jd.mean():.1f} (median "
          f"{np.median(jd):.1f}); headroom joint - marginal after: mean {gapj.mean():.1f}, median "
          f"{np.median(gapj):.1f} nats; within 1 nat of the bound {np.mean(gapj < 1):.4f}")
    _fr_numbers(Z, W, env, t_u, fg)
    print("== figure panel texts (td_tt_chaos_pairs_detail.png; units inside the 20 bins)")
    for k, s in figure_stats(Z).items():
        print(f"  {k:8s} units {s['n']:,} (below {s['below']:,}, above {s['above']:,}); median before {s['med']:.1f} "
              f"({s['q1']:.1f}-{s['q3']:.1f}); grows {s['grow']:.3f}, shrinks {s['shrink']:.3f}; mean change "
              f"{s['mean']:+.3f}, shuffled {s['shuf']:+.3f}; SKL change {s['skl']:+.3f}, beyond the common mode "
              f"{s['skl_cm']:+.3f}, beyond naive {s['skl_naive']:+.3f}")
    print("== FR thirds, units inside the 20 bins (the panel texts of the FR figure, dropped 11-02)")
    fst = fr_stats(Z)
    for k, name, _ in AGENTS:
        print(f"  {name:11s} " + "; ".join(f"third {r_ + 1}: units {fst[(k, r_)]['n']:,}, grows "
                                           f"{fst[(k, r_)]['grow']:.3f}, change {fst[(k, r_)]['mean']:+.3f}, shuffled "
                                           f"{fst[(k, r_)]['shuf']:+.3f}" for r_ in range(3)) +
              f"; matched hi - lo {_f(fst[k]['hilo'])} (raw {fst[k]['raw'][0]:+.3f}, cells cover "
              f"{fst[k]['share']:.3f} of the inside-bin units)")


def _combined_numbers():
    """the numbers of pairs_combined.npz (combined())"""
    C = np.load(COMB_CACHE)
    st = dict(zip(C["stat_names"], C["stat_boot"]))
    aux = dict(zip(C["stat_names"], C["stat_aux"]))
    nm = dict(zip(C["agents"], C["names"]))
    AG3 = ("bayes", "trained", "echo")
    e40, cen = C["edges40"], C["centres40"]

    def g(key, d=3):
        return _f(st[key], d)

    print(f"== combined (pairs_combined.npz): {NBIN40} bins {e40[0]:.3f}-{e40[-1]:.1f} nats (pooled percentiles "
          f"{EDGE40_PCT[0]} to {EDGE40_PCT[1]}; bin width x{e40[1] / e40[0]:.3f}); units below / above: " +
          "; ".join(f"{nm[k]} {C['var_out'][i, 0]:,} / {C['var_out'][i, 1]:,}" for i, k in enumerate(C["agents"])) +
          "; pair average below / above: " +
          "; ".join(f"{nm[k]} {C['avg_out'][i, 0]:,} / {C['avg_out'][i, 1]:,}" for i, k in enumerate(C["agents"])))
    print(f"  entanglement of the observation (pair-step; fifths) edges {np.round(C['ent5_edges'], 3)}, thirds "
          f"{np.round(C['ent3_edges'], 3)}; pooled thirds of the other variable's d_H before "
          f"{np.round(C['oth_edges'], 2)} nats; bins where some group has fewer than 50 units: " + "; ".join(
              f"{nm[k]} {int((C['entraw_n'][i].min(0) < 50).sum())} (fifths) / "
              f"{int((C['cond_n'][i].min(0) < 50).sum())} (other thirds)" for i, k in enumerate(C["agents"])
              if k != "naive"))
    print("  fewest units in a bin: " + "; ".join(f"{nm[k]} {C['var_n'][i].min():,}"
                                                   for i, k in enumerate(C["agents"])))
    idh_ = list(C["meas"]).index("dh")
    first = [cen[np.argmax(np.isfinite(C["ent_mean"][i, idh_, 0]))] for i in range(len(C["agents"]))]
    print("  step-standardised fifths: first drawn bin " + "; ".join(
        f"{nm[k]} {first[i]:.2f} nats" for i, k in enumerate(C["agents"]) if k != "naive") +
        "; highest - lowest fifth per bin, raw vs step-standardised, at the bins centred " + " / ".join(
        f"{cen[int(np.argmin(np.abs(np.log(cen / x))))]:.2f}" for x in (3, 10, 30)) + " nats: " + "; ".join(
        f"{nm[k]} " + " / ".join(
            f"{C['entraw_mean'][i, idh_, 4, b] - C['entraw_mean'][i, idh_, 0, b]:+.3f} vs "
            f"{C['ent_mean'][i, idh_, 4, b] - C['ent_mean'][i, idh_, 0, b]:+.3f}"
            for b in [int(np.argmin(np.abs(np.log(cen / x)))) for x in (3, 10, 30)])
        for i, k in enumerate(C["agents"]) if k != "naive"))
    s33 = (cen >= 3) & (cen <= 33)
    print(f"  highest - lowest fifth in every bin centred {cen[s33][0]:.2f}-{cen[s33][-1]:.2f} nats ({int(s33.sum())} "
          "bins), range over the bins, raw vs step-standardised: " + "; ".join(
              f"{nm[k]} {np.nanmin(dr):+.3f} to {np.nanmax(dr):+.3f} vs {np.nanmin(ds):+.3f} to {np.nanmax(ds):+.3f}"
              for i, k in enumerate(C["agents"]) if k != "naive"
              for dr, ds in [(C["entraw_mean"][i, idh_, 4, s33] - C["entraw_mean"][i, idh_, 0, s33],
                              C["ent_mean"][i, idh_, 4, s33] - C["ent_mean"][i, idh_, 0, s33])]))
    for n_, v in zip(C["chk_names"], C["chk_values"]):
        print(f"  check: {n_}: {v:.2g}")
    for j, lab in enumerate(("main (variable x entanglement fifth x within-cell third of the other variable's "
                             "distance x bin)",
                             "pooled other thirds (variable x entanglement fifth x pooled third x bin)",
                             "pair average (pair-steps of the same entanglement fifth x bin of the pair average)")):
        print(f"  shuffle, {lab}: mean change over all units per seed " + "; ".join(
            f"{nm[k]} {C['shuf_seed_means'][i, j].min():+.3f} to {C['shuf_seed_means'][i, j].max():+.3f}"
            for i, k in enumerate(C["agents"])))
    for k in AG3:
        print(f"  {nm[k]:11s} all units: change {g(f'all|dh|{k}')}, shuffled {g(f'all|shuf|{k}')}, actual - shuffled "
              f"{g(f'all|dh-shuf|{k}')}; sign flip: aligned {g(f'all|aligned|{k}')}, symmetric {g(f'all|sym|{k}')}")
    print("== definitions of the distance: per variable (a unit = one variable of a pair-step) vs the pair's average "
          "over the two variables (a unit = a pair-step); the closest pairs")
    for k in AG3:
        for lab in ("<1", "1-3"):
            print(f"  {nm[k]:11s} d_H before {lab} nat(s): " + "; ".join(
                f"{'per variable' if d == 'var' else 'pair average'} ({int(aux[f'close|{d}{lab}|dh|{k}']):,}): change "
                f"{g(f'close|{d}{lab}|dh|{k}')}, aligned {g(f'close|{d}{lab}|aligned|{k}')}, symmetric "
                f"{g(f'close|{d}{lab}|sym|{k}')}, shuffled {g(f'close|{d}{lab}|shuf|{k}')}" for d in ("var", "avg")))
    cm = list(C["cmeas"])
    lab_m = {"dh": "Delta d_H", "dh-shuf": "actual - shuffled", "aligned": "aligned part", "sym": "symmetric part",
             "skl": "Delta SKL", "skl_bn": "SKL beyond naive", "skl_bcm": "SKL beyond the common mode"}
    print("== entanglement of the observation (whole belief, fifths): highest - lowest fifth (raw; matched on d_H bin; "
          "on step and d_H bin; on step, d_H bin and FR third), matched fifths' means")
    for k in AG3:
        ia = list(C["agents"]).index(k)
        for im, mm in enumerate(cm):
            d = 3
            print(f"  {nm[k]:11s} {lab_m[mm]:27s} raw {g(f'ent5|{mm}|raw|{k}', d)}; bin {g(f'ent5|{mm}|bin|{k}', d)}; "
                  f"step+bin {g(f'ent5|{mm}|step+bin|{k}', d)} (fifths " +
                  " / ".join(f"{x:+.3f}" for x in C["ent5_std"][ia, im, :, 0]) +
                  f"; cover {aux[f'ent5|{mm}|step+bin|{k}']:.3f}); + FR third {g(f'ent5|{mm}|step+bin+fr|{k}', d)} "
                  f"(cover {aux[f'ent5|{mm}|step+bin+fr|{k}']:.3f})")
        print(f"  {nm[k]:11s} Delta d_H, other groupings (step + bin): thirds {g(f'ent3|dh|step+bin|{k}')}; own "
              f"variable's entanglement fifths {g(f'ent5own|dh|step+bin|{k}')}; other variable's "
              f"{g(f'ent5other|dh|step+bin|{k}')}; fifths on step band, bin and environment "
              f"{g(f'ent5|dh|band+bin+env|{k}')} (cover {aux[f'ent5|dh|band+bin+env|{k}']:.3f}); fifths on step, bin "
              f"and the within-cell third of the other variable's distance {g(f'ent5|dh|step+bin+cond3|{k}')} (cover "
              f"{aux[f'ent5|dh|step+bin+cond3|{k}']:.3f})")
        print(f"  {nm[k]:11s} FR thirds (40 bins), highest - lowest: step+bin {g(f'fr3|dh|step+bin|{k}')}; + "
              f"entanglement fifth {g(f'fr3|dh|step+bin+ent5|{k}')} (cover {aux[f'fr3|dh|step+bin+ent5|{k}']:.3f})")
    for mm in cm:
        print(f"  Trained RNN - Bayes, {lab_m[mm]:27s} one population "
              f"({int(aux[f'ent5|{mm}|step+bin|trained-bayes'])} cells, summed weights) "
              f"{g(f'ent5|{mm}|step+bin|trained-bayes')}")
    print(f"  Trained RNN - Bayes, Delta d_H, Bayes's weights {g('ent5|dh|step+bin|trained-bayes-Bweights')}")
    print("== the other variable's distance before: within each (step, own bin) cell its lower / middle / upper third; "
          "upper - lower third (matched on step and own d_H by construction), matched thirds' means")
    for k in AG3:
        ia = list(C["agents"]).index(k)
        meds = []
        for x in (.5, 1, 3, 10, 30):
            i = int(np.argmin(np.abs(np.log(cen / x))))
            meds.append(f"{cen[i]:.2f}: {C['cond_med_other'][ia, 0, i]:.2f} / {C['cond_med_other'][ia, 2, i]:.2f}")
        print(f"  {nm[k]:11s} median other d_H in the lower / upper third, own bin centre: " + "; ".join(meds))
        for im, mm in enumerate(cm):
            print(f"  {nm[k]:11s} {lab_m[mm]:27s} upper - lower {g(f'cond3|{mm}|step+bin|{k}')} (thirds " +
                  " / ".join(f"{x:+.3f}" for x in C["cond3_std"][ia, im, :, 0]) +
                  f"; cover {aux[f'cond3|{mm}|step+bin|{k}']:.3f})")
        print(f"  {nm[k]:11s} Delta d_H, upper - lower third also matched on the entanglement fifth "
              f"{g(f'cond3|dh|step+bin+ent5|{k}')} (cover {aux[f'cond3|dh|step+bin+ent5|{k}']:.3f}); on the FR third "
              f"{g(f'cond3|dh|step+bin+fr|{k}')} (cover {aux[f'cond3|dh|step+bin+fr|{k}']:.3f})")
        print(f"  {nm[k]:11s} pooled thirds of the other's d_H (absolute; asked first): raw {g(f'oth3|dh|raw|{k}')}; "
              f"step+bin {g(f'oth3|dh|step+bin|{k}')} (cover {aux[f'oth3|dh|step+bin|{k}']:.3f}); own bins with >= 50 "
              f"units in the upper third start at {cen[np.argmax(C['oth_n'][ia, 2] >= 50)]:.1f} nats")
    for mm in cm:
        print(f"  Trained RNN - Bayes, {lab_m[mm]:27s} one population "
              f"({int(aux[f'cond3|{mm}|step+bin|trained-bayes'])} cells, summed weights) "
              f"{g(f'cond3|{mm}|step+bin|trained-bayes')}")
    print(f"  Trained RNN - Bayes, Delta d_H, Bayes's weights {g('cond3|dh|step+bin|trained-bayes-Bweights')}")
    print("  the other variable's lower third (relatively close), Delta d_H: the matched thirds' means with [5-95%]; "
          "per bin (raw means of the within-cell third), the range over the bins of three distance bands, then every "
          "bin")
    bands = (("below 3 nats", cen < 3), ("3-40 nats", (cen >= 3) & (cen < 40)), ("40 nats and above", cen >= 40))
    for k in AG3:
        ia = list(C["agents"]).index(k)
        lmn, llo, lhi = (C[f"cond_{s}"][ia, idh_, 0] for s in ("mean", "lo", "hi"))
        print(f"  {nm[k]:11s} matched thirds " + " / ".join(_f(x) for x in C["cond3_std"][ia, cm.index("dh")]) +
              "; lower third per bin: " + "; ".join(
                  f"{lab} (bins {cen[bm][0]:.2f}-{cen[bm][-1]:.2f}) {np.nanmin(lmn[bm]):+.3f} to "
                  f"{np.nanmax(lmn[bm]):+.3f}" for lab, bm in bands))
        print(f"  {nm[k]:11s} lower third, every bin (centre: mean [5-95%]): " + "; ".join(
            f"{c_:.2f}: {a_:+.3f} [{l_:+.3f}, {h_:+.3f}]" for c_, a_, l_, h_ in zip(cen, lmn, llo, lhi)
            if np.isfinite(a_)))
    # the curves at a few distances (Delta d_H; [5-95%] over the resamples; the cache holds every bin)
    sel = [int(np.argmin(np.abs(np.log(cen / x)))) for x in (.5, 1, 3, 10, 30, 100)]
    idh = list(C["meas"]).index("dh")

    def cv(arr, ia, *ix):
        mean, lo, hi = (C[f"{arr}_{s}"][(ia, idh) + ix] for s in ("mean", "lo", "hi"))
        return f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]" if np.isfinite(mean) else "n/a"
    print("== the curves at bins centred at " + " / ".join(f"{cen[i]:.2f}" for i in sel) + " nats (Delta d_H, [5-95%])")
    for k in AG3:
        ia = list(C["agents"]).index(k)
        for lab, arr, ix in (("per variable", "var", ()), ("pair average", "avg", ()),
                             ("other variable's distance, lower third", "cond", (0,)),
                             ("other variable's distance, upper third", "cond", (2,)),
                             ("entanglement, lowest fifth (step-standardised)", "ent", (0,)),
                             ("entanglement, highest fifth (step-standardised)", "ent", (4,))):
            print(f"  {nm[k]:11s} {lab:48s} " + " / ".join(cv(arr, ia, *(ix + (i,))) for i in sel))


def _combined_more():
    """the combined data's numbers that need the per-unit caches (11-02, round 4): the pooled-minimum edges; Bayes's
    product of marginals (no coupling) under the other-variable and entanglement splits (combined()'s estimator and
    cells); the pair-average closest pairs against each variable's own (step, bin) cell mean"""
    Z, C, SH = load(), np.load(COMB_CACHE), np.load(SHUF40_CACHE)
    W = _W()
    env, t_u = np.repeat(Z["env"], NV), np.repeat(Z["t"], NV)
    e40, nb = C["edges40"], NBIN40
    g5 = np.repeat(SH["g5"].astype(np.int64), NV)
    pooled = np.concatenate([Z[f"pre_{k}"].ravel() for k in COMB_AGENTS])
    emin = np.logspace(np.log10(pooled.min()), np.log10(np.percentile(pooled, EDGE40_PCT[1])), nb + 1)

    def lowest_empty(p):
        b_ = unit_bins(p, emin)
        return int(np.argmax(np.bincount(b_[(b_ >= 0) & (b_ < nb)], minlength=nb) > 0))
    print(f"== combined, more. {nb} bins from the pooled minimum ({emin[0]:.4f} nats) to the pooled {EDGE40_PCT[1]}th "
          "percentile instead: lowest bins left empty " + "; ".join(
              f"{name} {lowest_empty(Z[f'pre_{k}'].ravel())}" for k, name, _ in AGENTS))
    print("== Bayes's product of marginals (no coupling between the variables; DEFINITIONS) under the combined splits, "
          "matched on step and own d_H bin (40 bins; combined()'s cells)")
    pre, post = _units(Z, "bayes")
    prod = Z["prod_bayes"].ravel()
    b = unit_bins(pre, e40)
    inb = (b >= 0) & (b < nb)
    tb = t_u * nb + b
    cg = SH["cond_bayes"].ravel().astype(np.int64)
    mts = {}
    for lab, v in (("actual change", post - pre), ("product-of-marginals change", prod - pre),
                   ("the coupling's part (actual - product)", post - prod)):
        m3, m5 = matched(v, cg, tb, inb, env, W), matched(v, g5, tb, inb, env, W, K=ENT_K)
        mts[lab] = (m3, m5)
        print(f"  {lab}: other variable's within-cell thirds, upper - lower {_f(m3['hilo'])} (thirds " +
              " / ".join(_f(x) for x in m3["std"]) + f"; cover {m3['share']:.3f})")
        print(f"  {'':{len(lab)}s}  entanglement fifths, highest - lowest {_f(m5['hilo'])} (fifths " +
              " / ".join(_f(x) for x in m5["std"]) + f"; cover {m5['share']:.3f})")
    pt, qt = _units(Z, "trained")
    bt = unit_bins(pt, e40)
    okt = (bt >= 0) & (bt < nb)
    mt3 = matched(qt - pt, SH["cond_trained"].ravel().astype(np.int64), t_u * nb + bt, okt, env, W)
    mt5 = matched(qt - pt, g5, t_u * nb + bt, okt, env, W, K=ENT_K)
    p3, p5 = mts["product-of-marginals change"]
    print("  product of marginals - Trained RNN on one population (the cells usable for both; summed weights / "
          f"Bayes's weights): other-variable thirds {_f(common_hilo(p3, mt3)[0])} / "
          f"{_f(-common_hilo(mt3, p3, 'B')[0])}; entanglement fifths {_f(common_hilo(p5, mt5)[0])} / "
          f"{_f(-common_hilo(mt5, p5, 'B')[0])}")
    print("== definitions: the closest pairs against each variable's own (step, bin) cell: predicted = the mean change "
          "of all units in the unit's cell (step t, bin of 40; 2 out-of-range bins), re-estimated in every resample")
    ncell = (T + 1) * (nb + 2)
    for k, name in (("bayes", "Bayes"), ("trained", "Trained RNN"), ("echo", "ESN")):
        pv = Z[f"pre_{k}"]
        ch = (Z[f"post_{k}"] - pv).ravel()
        key = env * ncell + t_u * (nb + 2) + unit_bins(pv.ravel(), e40) + 1
        Sw = W @ np.bincount(key, ch, gap.N_ENV * ncell).reshape(gap.N_ENV, ncell)
        Cw = W @ np.bincount(key, None, gap.N_ENV * ncell).reshape(gap.N_ENV, ncell)
        with np.errstate(invalid="ignore", divide="ignore"):
            cmean = np.nan_to_num(Sw / Cw)                       # (1 + N_BOOT, cells); 0 only where a cell is empty
        res = []
        for lab, sel, unit in (("pair average < 1 nat", np.repeat(pv.mean(1) < 1, NV), "pair-steps"),
                               ("per variable < 1 nat", pv.ravel() < 1, "units")):
            Nw = W @ np.bincount(key[sel], None, gap.N_ENV * ncell).reshape(gap.N_ENV, ncell)
            n_ = W @ np.bincount(env[sel], None, gap.N_ENV).astype(np.float64)
            act, pred = (W @ np.bincount(env[sel], ch[sel], gap.N_ENV)) / n_, (Nw * cmean).sum(1) / n_
            cnt = int(sel.sum()) // (NV if unit == "pair-steps" else 1)
            res.append(f"{lab} ({cnt:,} {unit}; mean own d_H before {pv.ravel()[sel].mean():.2f}): actual {_f(act)}, "
                       f"predicted {_f(pred)}, actual - predicted {_f(act - pred)}")
        print(f"  {name:11s} " + "; ".join(res))


def tail_flags(Z, LQ, k):
    """per unit (m, 2): d_H before is set at near-impossible values: at both extremes of the log-ratio (its argmax and
    argmin value), the larger of the two beliefs is below 1e-3"""
    I, J, S = Z["I"], Z["J"], Z["t"]
    a, b = LQ[k][I, S - 2], LQ[k][J, S - 2]
    D = a - b
    hi_, lo_ = D.argmax(-1)[..., None], D.argmin(-1)[..., None]

    def at(x, ix):
        return np.take_along_axis(x, ix, -1)[..., 0]
    return np.exp(np.maximum(np.maximum(at(a, hi_), at(b, hi_)), np.maximum(at(a, lo_), at(b, lo_)))) < 1e-3


def _tails(Z, LQ):
    """share of units whose d_H before is set at near-impossible values (tail_flags)"""
    print("  d_H before set by near-impossible values (the larger belief < 1e-3 at both extremes of the log-ratio): " +
          ", ".join(f"{name} {tail_flags(Z, LQ, k).mean():.3f}" for k, name, _ in AGENTS))


def _closest(Z, W, env):
    """the closest pairs: each unit's change split exactly (sign flip) into the part aligned with the existing
    difference and the symmetric part, against the shuffle; actual - shuffled = aligned + (symmetric - shuffled
    change) exactly"""
    print("== closest pairs: the exact sign-flip split vs the shuffle (variable x FR third x bin; donors from all "
          "environments)")
    for k, name, _ in AGENTS:
        if k == "naive":
            continue
        pre, post = _units(Z, k)
        fl, sh = Z[f"flip_{k}"].ravel(), Z[f"shuf_{k}"].ravel()
        sym = (post + fl) / 2 - pre
        for lab, mk in (("< 1 nat", pre < 1), ("1-3 nats", (pre >= 1) & (pre < 3))):
            ident = np.abs((post - sh) - ((post - fl) / 2 + sym - (sh - pre)))[mk].max()
            print(f"  {name:11s} {lab:8s} ({int(mk.sum()):,} units): change {_f(_ratio(post - pre, mk, env, W))}; "
                  f"aligned {_f(_ratio((post - fl) / 2, mk, env, W))} (positive in "
                  f"{np.mean(post[mk] > fl[mk] + 1e-9):.3f}); "
                  f"symmetric {_f(_ratio(sym, mk, env, W))}; shuffled change {_f(_ratio(sh - pre, mk, env, W))}; "
                  f"actual - shuffled {_f(_ratio(post - sh, mk, env, W))}; symmetric - shuffled change "
                  f"{_f(_ratio(sym - (sh - pre), mk, env, W))} (identity to {ident:.1e})")


def _robust_nulls(Z, W, env, LQ):
    """actual - shuffled under other groupings (orientation-free, one seed each)"""
    I, J, S = Z["I"], Z["J"], Z["t"]
    m = len(I)
    P = probes()
    code = (P["obs"] * (1 << np.arange(NO))).sum(-1)[I, S - 1]
    envp = Z["env"]
    allm = np.ones(m * NV, bool)
    out = []
    for k, name, _ in AGENTS:
        if k == "naive":
            continue
        lq = LQ[k]
        D = lq[I, S - 2] - lq[J, S - 2]
        dl = (lq[I, S - 1] - lq[I, S - 2]) - (lq[J, S - 1] - lq[J, S - 2])
        post = Z[f"post_{k}"].ravel()
        res = []
        rng = np.random.default_rng(PAIR_SEED + 100)
        for lab in ("bin", "env x bin", "step x bin", "env x step x bin", "env x observation x bin"):
            sh = np.zeros((m, NV))
            for v in range(NV):
                b = unit_bins(Z[f"pre_{k}"][:, v], Z["edges"]) + 1
                gid = {"bin": b, "env x bin": envp * 22 + b, "step x bin": S * 22 + b,
                       "env x step x bin": (envp * 31 + S) * 22 + b, "env x observation x bin": (envp * 32 + code) * 22
                       + b}[lab]
                perm = _within_perm(gid, rng)
                sh[:, v] = osc(D[:, v] + rng.choice([-1., 1.], m)[:, None] * dl[perm, v])
            res.append(f"{lab} {_ratio(post - sh.ravel(), allm, env, W)[0]:+.3f}")
        out.append(f"  {name:11s}: actual - shuffled with other groups (one seed each): " + "; ".join(res))
    print("\n".join(out))


def _fr_numbers(Z, W, env, t_u, fg):
    """the FR comparison: raw thirds, matched on d_H before, matched on step and d_H before; Trained - Bayes"""
    print("== FR: change of d_H, highest - lowest third (raw; matched on d_H before; matched on step and d_H before)")
    third = [fg == j for j in range(3)]
    mt, mobj, alt = {}, {}, {}
    band = np.digitize(t_u, [11, 21])
    for k, name, _ in AGENTS:
        if k == "naive":
            continue
        pre, post = _units(Z, k)
        b = unit_bins(pre, Z["edges"])
        inb = (b >= 0) & (b < PAIR_BINS)
        res = {}
        for lab, v in (("change", post - pre), ("actual - shuffled", post - Z[f"shuf_{k}"].ravel()),
                       ("SKL beyond naive", ((Z[f"skl_post_{k}"] - Z[f"skl_pre_{k}"]) -
                                             (Z["skl_post_naive"] - Z["skl_pre_naive"])).ravel()),
                       ("SKL beyond common mode", (Z[f"skl_post_{k}"] - Z[f"skl_cm_{k}"]).ravel()),
                       ("JS beyond naive", ((Z[f"js_post_{k}"] - Z[f"js_pre_{k}"]) -
                                            (Z["js_post_naive"] - Z["js_pre_naive"])).ravel()),
                       ("JS beyond common mode", (Z[f"js_post_{k}"] - Z[f"js_cm_{k}"]).ravel())):
            raw = [_ratio(v, th, env, W) for th in third]
            m_b = matched(v, fg, b, inb, env, W)
            m_tb = mobj[(k, lab)] = matched(v, fg, t_u * PAIR_BINS + b, inb, env, W)
            res[lab] = m_tb["hilo"]
            d = 4 if lab.startswith("JS") else 3
            print(f"  {name:11s} {lab:22s} raw thirds " + " / ".join(f"{x[0]:+.{d}f}" for x in raw) +
                  f", hi - lo {_f(raw[2] - raw[0], d)}; matched on d_H before {_f(m_b['hilo'], d)}; matched on step "
                  f"and d_H before: thirds " + " / ".join(f"{x:+.{d}f}" for x in m_tb["std"][:, 0]) +
                  f", hi - lo {_f(m_tb['hilo'], d)} (cells cover {m_tb['share']:.3f})")
            if lab == "actual - shuffled":
                continue
            # other matchings: the agent's own bulk measure before (instead of d_H), environment as well
            line = f"  {'':11s} {'':22s} "
            if lab != "change":
                meas = "skl" if lab.startswith("SKL") else "js"
                bef = Z[f"{meas}_pre_{k}"].ravel()
                lo_, hi_ = np.percentile(bef[bef > 0], EDGE_PCT)
                b2 = unit_bins(bef, np.logspace(np.log10(lo_), np.log10(hi_), PAIR_BINS + 1))
                ok2 = (b2 >= 0) & (b2 < PAIR_BINS)
                h2 = matched(v, fg, t_u * PAIR_BINS + b2, ok2, env, W)["hilo"]
                line += f"matched on step and own {meas.upper()} before (20 bins) {_f(h2, d)}; "
            h3 = matched(v, fg, (band * PAIR_BINS + b) * gap.N_ENV + env, inb, env, W)
            h4 = matched(v, fg, (t_u * PAIR_BINS + b) * gap.N_ENV + env, inb, env, W, n_min=5)
            alt[(k, lab)] = (h3["hilo"], h4["hilo"])
            print(line + f"on step band (2-10, 11-20, 21-30), d_H bin and environment {_f(h3['hilo'], d)} (cover "
                  f"{h3['share']:.3f}); on step, d_H bin and environment (n_min 5) {_f(h4['hilo'], d)} (cover "
                  f"{h4['share']:.3f})")
        mt[k] = res
        ch = post - pre
        sens = []
        pooled = np.concatenate([Z[f"pre_{a}"].ravel() for a, _, _ in AGENTS])
        for lab, (p0, p1, nb) in (("1st-99th percentile bins (20)", (1, 99, 20)), ("40 bins", (*EDGE_PCT, 40))):
            e2 = np.logspace(*np.log10(np.percentile(pooled, [p0, p1])), nb + 1)
            b2 = unit_bins(pre, e2)
            ok2 = (b2 >= 0) & (b2 < nb)
            hl = matched(ch, fg, t_u * nb + b2, ok2, env, W)["hilo"]
            sens.append(f"{lab}, {e2[0]:.2f}-{e2[-1]:.1f} nats: {_f(hl)}")
        print(f"  {name:11s} change, matched on step and d_H before, other bins: " + "; ".join(sens))
    print("  Trained RNN - Bayes, matched on step and d_H before. Own: each agent's contrast standardised to its own "
          "(step, bin) units; one population: the cells usable for both, weighted by both agents' counts (summed) or "
          "by Bayes's; environment-matched: the difference of the two own-standardised contrasts")
    for lab in mt["bayes"]:
        d = 4 if lab.startswith("JS") else 3
        line = (f"  Trained RNN - Bayes, {lab:22s} own {_f(mt['trained'][lab] - mt['bayes'][lab], d)}; summed "
                f"{_f(common_hilo(mobj[('trained', lab)], mobj[('bayes', lab)])[0], d)}; Bayes's "
                f"{_f(common_hilo(mobj[('trained', lab)], mobj[('bayes', lab)], 'B')[0], d)}")
        if ("bayes", lab) in alt:
            line += (f"; (band, bin, env) {_f(alt[('trained', lab)][0] - alt[('bayes', lab)][0], d)}; (step, bin, env) "
                     f"{_f(alt[('trained', lab)][1] - alt[('bayes', lab)][1], d)}")
        print(line)
    pre, post = _units(Z, "trained")
    sh = Z["shuf_trained"].ravel()
    b = unit_bins(pre, Z["edges"])
    cen = np.sqrt(Z["edges"][1:] * Z["edges"][:-1])
    near = third[2] & (pre < 3)
    lo_bins = [(cen[i], np.mean((post - pre)[third[0] & (b == i)]), int((third[0] & (b == i)).sum()))
               for i in range(PAIR_BINS) if (third[0] & (b == i)).sum() >= 50]
    print("  Trained RNN, lowest FR third, mean change per bin of d_H before (centre: change, units), the bins above "
          "30 nats: " + "; ".join(f"{c:.0f}: {v:+.2f} ({n:,})" for c, v, n in lo_bins if c > 30))
    print(f"  Trained RNN, highest FR third, d_H before < 3 nats ({near.sum():,} units): mean change "
          f"{np.mean((post - pre)[near]):+.3f}, shuffled {np.mean((sh - pre)[near]):+.3f}")


if __name__ == "__main__":
    what = sys.argv[1:] or ["numbers"]
    if "build" in what:
        build()
    if "figure" in what:
        figure()
    if "fr" in what:
        print("fr: the FR figure was dropped on 11-02; `figure` draws this module's only figure")
    if "combined" in what:
        combined()
    if "numbers" in what:
        numbers()
        if os.path.exists(COMB_CACHE):
            _combined_numbers()
            _combined_more()
