"""Expansion at entangled observations (John 11-02, on the growth of one flipped observation's impact, td_tt_chaos_sweep
panel e / td_tt_flips: "we expect the dynamics the be converging on average, but chaotic during entangled observations.
Did we check that detail?"). Every measure is split by the entanglement of the step's observation, at every stored
checkpoint of the Trained RNN and the ESN, with the exact observers as references. Revised 11-02/03 after two
independent checks (CORRECTIONS).
-> main/REPLICATION/figure_bin/td_tt_chaos_ent.png

ANSWER (Trained RNN and ESN, all 114 checkpoints each; fifth 5 - fifth 1 = most - least entangled; [5-95%] over 1,000
environment resamples; nats / step)
  Excluded: chaos as divergence. A flipped observation's state difference shrinks on average at every checkpoint, in
    every entanglement fifth and in the most entangled twentieth, at lags 1-4, 5-20 and 21-29 alike (every fifth < 0 at
    all 114 checkpoints in each; the most entangled fifth at most -.057, -.035 and -.024). Only the first step after a
    flip grows, transiently, around passing naive ((3) below). The flip's belief difference (d_H) accumulates on
    average at every checkpoint, as Bayes's marginal difference does (the flipped evidence is remembered; bounded for
    Bayes): at every level up to 31,200 episodes (every fifth > 0 at 10-31,200 and 32,800); late, at lags 5-20, it no
    longer grows (all units > 0 only to 34,400; end +.0006 [-.0015, +.0030]). The step's entangled observation adds at
    most +.007 to it (own part; during learning it grows faster where observations are entangled, with the state:
    below). The ESN contracts everywhere: no ESN step ever grows, and its step map has no expanding eigen-direction.
  Present, in three weaker senses.
    (1) Potential expansion: around passing naive, the step map at the most entangled observations expands along an
    eigen-direction while the average step contracts. log rho(J_t), the log of the step Jacobian's largest |eigenvalue|
    (coordinate-free), is > 0 in the most entangled fifth at 560-1,200 and 1,360 episodes. While all steps average < 0
    (both bounds: 560-640, 880-1,200 and 1,360) it reaches +.0125 [+.0081, +.0168] (640; +.0123 at 880); its peak,
    +.0156 [+.0113, +.0198] at 800, is where the average is flat (neither bound at 720-800). At 880, 43% of the
    most entangled steps expand, against 21% of the least entangled. About a third of the contrast is the step's own
    observation (own part +.007 to +.013 at 360-1,200 episodes, of +.022 to +.032), the rest goes with the states at
    which observations are entangled; late the own part carries it (+.004 of +.005 at the end). Factorized-state Bayes
    expands at every step, more at entangled observations and because of them (log rho = log(1 + rho_1), +.067 ->
    +.118 by fifth; own +.044 of +.051); Bayes and naive never (0).
    (2) Update sensitivity: the flip's difference in the step's belief update shrinks on average but grows at the most
    entangled observations at lags 5-20, as in exact Bayes (Bayes: all -.062, most entangled fifth +.293; contrast
    +.705, all of it the observation's own part). The Trained RNN acquires this late: its most entangled fifth grows at
    9,600-39,996 episodes (end +.077 [+.064, +.089], all units -.082); contrast +.075 at 4,200, +.221 at 12,000, +.305
    at the end, the observation's own part +.087, +.227, +.309. The ESN never (contrast < 0). This is the update
    depending more on the belief at entangled observations (in Bayes all of it the observation's own part), not
    beliefs diverging.
    (3) Realised one-step growth, transient: the first step after a fresh flip (lag 1; figure panel e) grows from 720
    episodes on, first at the more entangled observations (720: fifths -.0102 / +.0036 / +.0022 / +.0131 / +.0131,
    fifths 4-5 > 0, fifth 1 not; the lag-1 contrast is > 0 at 120-800, +.0233 [+.0113, +.0332] at 720), state-level
    ('state' contrast > 0 at 120-800, +.031 at 720; 'obs' +.0009 [-.0068, +.0083] at 720, > 0 only from 3,000 on),
    then at every level alike (every fifth > 0 at 800-4,800, the contrast not separated from 0 at 880-5,600; all units
    up to +.111 at 1,200). Its onset falls inside the window of (1); it is gone by 7,000 episodes (every fifth < 0 from
    7,000 on), and averaged over lags 1-4 or later lags every fifth shrinks (Excluded). The ESN never (P(g > 0) = 0).
  During learning (160-1,700 episodes) the state difference shrinks less (+.034 at 560) and the belief difference grows
    more (+.070 at 240) where observations are entangled. This is not because of the step's own observation (own part
    -.0047 to -.0004 for g, at most +.0027 for d_H, up to 1,680 episodes; fifths of the observation's own entanglement
    within state fifths: +.0004 at 560, +.0016 at 240): it goes with the states at which observations are entangled
    (fifths of the state's expected entanglement: +.058 at 560, +.106 at 240). It survives matching on belief
    confidence (+.021 at 560, +.051 at 240).

DEFINITIONS
  Probes: td_tt_chaos.subset() (20 per test environment x 50 environments = 1,000 episodes of 30 steps; the chaos
    flips' probes). Unit = (probe, step t). Entanglement of observation t = analysis/probes.npz 'ent' (the locked
    definition: SKL between naive's per-step likelihood and Bayes's effective per-step marginal update, given Bayes's
    belief before t), the mean of the two variables (whole belief).
  Fifths: per step t, the 1,000 probes ranked by entanglement (ties share their average rank) and cut into five groups
    of 200, so every fifth holds the same steps and step trends cannot drive the split. Fifth 1 = least, fifth 5 =
    most entangled; contrast = fifth 5 - fifth 1; v20 = the same ranking in twentieths (a threshold at the very top?).
  (1) One-step expansion, coordinate-free (the headline measure): log rho(J_t) = the log of the largest |eigenvalue|
    of the step Jacobian J_t = d(u_t, c_t) / d(u_{t-1}, c_{t-1}) (2,000 dims) at the intact state and observation t
    (td_tt_chaos.tangent), steps 1-30. Invariant to any fixed linear change of coordinates (S^-1 J S has J's
    eigenvalues). > 0: the step's map expands an eigen-direction (it would grow under repetitions of this step):
    potential expansion, not realised growth (consecutive steps are different maps; (2) is the realised growth).
    Share = the units with log rho > 0. Method: Arnoldi on batched JVPs (GPU, fp32; one random start per unit; full
    reorthogonalisation, classical Gram-Schmidt twice), the small Hessenberg eigenproblems in float64 (2 CPU threads),
    checked at Krylov dimensions 60, 90 and 120 (a check skipped in later chunks where it stopped < 10% of the units):
    a unit stops when the top-modulus Ritz pair's relative residual |h_(m+1,m) y_m| / |theta| <= 1e-5; units above it
    restart from that Ritz vector (real + imaginary part), at most 5 times (the last value kept; CHECKS).
    Exact observers: joint Bayes and naive are translations in log coordinates (T1), log rho = 0; factorized-state
    Bayes (ADF): its exact step Jacobian in the zero-sum chart has eigenvalues 1 +- the canonical correlations of the
    one-step joint (T3), so log rho = log(1 + rho_1) >= 0 (its explicit eigenvalues agree within 1.9e-9).
  Euclidean (u, c) stretch (chart-dependent; the first version's measure (1), numbers only): log sigma_max(J_t), the
    largest singular value in Euclidean (u, c) (Lanczos on J^T J, within 1e-4). It is free of a random frame but not of
    the coordinates: its entanglement contrast and the ESN's level change sign with the scaling of u and c (check
    11-02: at 560 episodes -.0096 Euclidean, +.0051 [-.0018, +.0118] block-RMS, +.077 whitened; ESN at the end -.080
    Euclidean, +.104 whitened), so no sign claim is made from it. Its +2 nats at the end are non-normal amplification
    (the eigenvalues stay near 1).
  (2) Realised growth of the state difference: flip one observation o at step s (s in 1, 3, 5, 10; each of the 5
    observations), rerun the network from its intact state with the intact observations after; Delta_t = (u, c)_flipped
    - (u, c)_intact at t >= s; lag = t - s. One-step growth g_t = log ||Delta_t|| - log ||Delta_{t-1}|| at lags >= 1.
    A unit's value = the mean of g_t over its flips in a lag group: lag 1 (the step after the flip), lags 1-4 and 5-20
    (John's short / long split), 21-29 (only the flips at steps 1-5 reach them) and all lags; P(g_t > 0) = the share of
    those flips with g_t > 0. Delta is propagated in difference form (dsig, dtanh: exact identities, no subtraction of
    nearly equal states), so it keeps fp32 relative precision however small it gets (CHECKS).
  (3) Growth of the belief difference: chi_t = d_H(intact, flipped marginal beliefs) per variable (Hilbert's metric:
    the range of the log-ratio), the mean of the two variables; its one-step change chi_t - chi_{t-1}, units and lag
    groups as in (2). It accumulates in every agent with memory, Bayes included: the network's read-out keeps adding
    the remembered state difference; Bayes's marginal d_H is bounded by the joint's, which the flip fixes (T2). Also
    the symmetric KL, SKL = Sigma_r (p_r - q_r) log(p_r / q_r) / 2 (numbers only: chart-confounded, even naive with a
    constant d_H has a contrast). Networks: the cumulative read-out of the flipped minus intact increments (float64).
  (4) Update sensitivity: s_t = the d_H size of the flip's difference in the step's marginal belief update, the mean
    over the 2 variables of osc((log m'_t - log m_t) - (log m'_{t-1} - log m_{t-1})) (at the flip step osc(log m'_s -
    log m_s)); its one-step growth log s_t - log s_{t-1}, units and lag groups as in (2). Networks: osc of W_ro du_t,
    the flipped minus intact read-out of u_t (the network's log marginals move at t by the read-out of u_t, up to a
    constant per variable). After the flip step Bayes's two runs see the same likelihoods, so s_t measures how much
    the step's update depends on the belief. For a product-form belief (ADF) a factorized likelihood gives s_t = 0;
    for the joint posterior a factorized observation still acts through the belief's correlations (the flip changes
    p(r_1 | r_0)), and the observation's entanglement adds to it (Bayes: contrast +.705, all of it own). Naive's
    update ignores its belief: s_t = 0 exactly (T1), left out. Sensitivity of the update, not divergence: the
    beliefs' own difference is (3).
  Exact observers (float64, CPU; the same probes, flips, units and fifths): Bayes = the joint posterior's marginals;
    naive; factorized-state Bayes (ADF, td_tt_chaos.adf_run: keeps only its marginals). Joint Bayes and naive have no
    state growth (g = 0, T1); naive's d_H change is exactly 0 too. ADF's state = its log marginals in the orthonormal
    zero-sum chart (td_tt_chaos.adf_lyap's), with its own g.
  Averaging (the statistic at the finest unit, then means): per flip -> per unit (the mean over its flips in the lag
    group; shares over its flips) -> per-environment sums and unit counts per fifth -> the pooled ratio of sums over the
    environments (every unit weighs the same); 5-95% intervals over the 1,000 shared environment resamples
    (td_tt_gap.boot_counts()); contrasts paired in every resample. Equal-weight environment means (each environment's
    mean first) agree: numbers prints both for every contrast (e.g. g at 560 +.0291 vs +.0338 pooled, log rho at 800
    +.0311 vs +.0353, the update measure at the end +.3302 vs +.3054, Bayes's d_H change -.0337 vs -.0327). No
    normalisation: every value is in nats (per step) or a share.
  Counterfactual control (numbers; the figure's dots): step t rerun for every unit under each of the 32 observation
    patterns, from the same intact state and (for the flip measures) the same flip difference Delta_{t-1} (difference
    form; the flips' later steps keep their actual observations). expected = the mean over the patterns weighted by
    p(pattern | the probe's cell) (the distribution the actual observations are drawn from); own = actual - expected =
    what the step's actual observation adds at that state and history; the fifth contrast splits exactly into
    expected + own. g and the d_H change (all lags) and the update measure (each lag group) at 10 Trained RNN and 4 ESN
    checkpoints (CF_X); log rho at 6 Trained RNN and 2 ESN checkpoints (CF_LR_X) on 200 of the probes (the first 4 of
    each environment; the main fifth labels); the observers: Bayes's d_H change and update measure, ADF's g, d_H change,
    update measure and log rho (all probes, float64).
  Controls (numbers): 'env' = fifths ranked within each step AND environment; 'usize' = entanglement fifths within
    each step and fifth of Bayes's marginal update size at t (matched on update size); 'usizeq' = the update-size
    fifths themselves; 'var' = per variable (each variable's d_H / SKL change with its own entanglement); 'hist' =
    entanglement fifths within fifths of what the step receives from the flip (the unit's mean log||Delta_{t-1}||,
    chi_{t-1}, SKL_{t-1}); 'conf' = entanglement fifths within fifths of Bayes's marginal entropy before t (matched on
    belief confidence); 'confq' = the entropy fifths themselves (most uncertain - most confident); 'state' = fifths of
    the state's expected entanglement sum_k w_k ent_k (the entanglement an observation has on average at the same
    belief: a state-level grouping); 'obs' = fifths of the observation's own part ent - sum_k w_k ent_k within the
    state fifths (an observational counterpart of the counterfactual, at every checkpoint).

RESULTS (`numbers`; nats per step unless a share; all = the all-unit mean; "> 0 at" = 5% bound > 0, "< 0 at" = 95%
bound < 0, as printed with the number of checkpoints; [5-95%]; equal-weight environment means in braces)
  Exact observers (the same probes and flips, float64; constant through training):
    Bayes: log rho = g = 0 (T1). d_H change all +.0173 [+.0155, +.0189], fifths +.036 / +.021 / +.017 / +.009 / +.004,
      contrast -.0327 [-.0378, -.0275] {-.0337} (env -.034, usize -.034, var -.020, hist -.034, conf -.034, state -.011,
      obs -.033; usizeq +.019); own part -.0272 [-.0323, -.0217] (expected -.0055). Update growth, lags 5-20: all
      -.0622 [-.0639, -.0601], fifths -.411 / -.214 / -.075 / +.097 / +.293, contrast +.7047 [+.6638, +.7471] {+.744}
      (conf +.725, state +.015, obs +.858), P(> 0) .22 -> .63; own part +.7050 [+.6654, +.7453] (expected -.0003);
      lags 1-4: all -.538, fifths -.888 -> -.203, contrast +.685; lag 1: all -1.713; lags 21-29: contrast +.693; all
      lags: fifths -.518 -> +.171, contrast +.688 [+.649, +.729], all -.175. SKL change: all -.0110, contrast +.0018
      [+.0005, +.0031].
    Naive: log rho = g = 0, d_H change and update difference exactly 0 (T1); SKL change all -.0064, contrast +.0033
      [+.0024, +.0042].
    Factorized-state Bayes (ADF): log rho = log(1 + rho_1): all +.0911 [+.0885, +.0935], fifths +.067 / +.080 / +.089 /
      +.103 / +.118, contrast +.0507 [+.0431, +.0576] {+.047} (conf +.066, state +.010, obs +.053), > 0 at every unit;
      own part +.0444 [+.0414, +.0476] (expected +.0063 [-.0001, +.0122]). g: all +.0268 [+.0253, +.0283] (its flip
      difference grows on average), contrast +.0055 [+.0020, +.0091], own -.0018 [-.0037, +.0001]. d_H change: all
      +.186, fifths +.128 -> +.277, contrast +.1498 [+.0899, +.2140] (state +.236, obs +.008), own +.0037 [-.0125,
      +.0208]. Update growth, lags 5-20: all -.0828, fifths -.253 -> +.178, contrast +.4310 [+.3989, +.4642], own
      +.4608. Euclidean stretch of its zero-sum chart .211 -> .325 (no sign claims).
  Trained RNN (114 checkpoints; passes naive at 835 episodes):
    (1) log rho(J_t): all -.416 (initial), -.0695 (20), +.0022 (60), -.0096 (120), -.0184 (180), -.0104 (240), -.0163
      (360), -.0105 (560), -.0029 (720), -.0038 (880), -.0056 (1,200), -.0110 (1,680), -.0059 (4,200), -.0101
      [-.0106, -.0095] (end); > 0 at 60-80 (2; max +.0060 [+.0047, +.0073] at 80, every fifth > 0 at 80: an early
      expansion, least entangled most), < 0 at 10-40 (3), 100-640 (19), 880-39,996 (88). Most entangled fifth: > 0 at
      80 (1), 560-1,200 (9), 1,360 (1), max +.0156 [+.0113, +.0198] at 800, where the all-unit mean is flat (neither
      bound at 720-800; -.0011 [-.0044, +.0019] at 800, check 11-03); fifths at 880 -.0215 / -.0088 / -.0037 / +.0027
      / +.0123; > 0 while the all-unit mean < 0 at 560-640 (2), 880-1,200 (5), 1,360 (1), up to +.0125 [+.0081,
      +.0168] (640, check 11-03; the figure's suptitle prints +0.012); < 0 at 1,680-2,800 (12), 4,000 (1), 6,200-39,996
      (50); end -.0062. Contrast > 0 at 180-39,996 (105), < 0 at 10-140 (8); max +.0353 [+.0298, +.0407] at 800
      {+.0311}; end +.0074 [+.0059, +.0089]. Controls at 800: env +.0305, usize +.0350, conf +.0289 [+.0234, +.0344],
      state +.0403 [+.0338, +.0462], obs +.0125 [+.0094, +.0155]; usizeq -.0022, confq -.0195.
      Top twentieth max +.026 [+.019, +.034] at 800. Share of steps with log rho > 0: all max .456 at 80, .108 at the
      end; fifths at 880 .210 / .299 / .329 / .371 / .431, at the end .062 -> .169; contrast max +.239 at 560.
      Counterfactual (200 probes; actual = expected + own): 360 +.0319 = +.0249 + .0070 [+.0041, +.0102]; 560 +.0311 =
      +.0223 + .0088 [+.0051, +.0126]; 880 +.0324 = +.0195 + .0129 [+.0103, +.0155]; 1,200 +.0219 = +.0139 + .0080
      [+.0055, +.0109]; 1,680 +.0113 = +.0034 [-.0019, +.0083] + .0079 [+.0053, +.0111]; end +.0054 = +.0014 [-.0005,
      +.0031] + .0040 [+.0028, +.0052]. Leading eigenvalue complex at 15.5% of the units.
      Euclidean (u, c) stretch (chart-dependent): log sigma_max -.149 (initial) -> +2.08 (end), > 0 at every unit of
      steps 2-30 from 20 episodes on; its contrasts are not interpreted.
    (2) g: all lags: all -.484 (initial), max -.0565 [-.0574, -.0556] at 16,800, end -.0577; most entangled fifth max
      -.0515 [-.0536, -.0491] (39,200), top twentieth max -.0478 (36,800); never > 0 in any fifth. Lag 1: most
      entangled fifth > 0 at 720-5,000 (35), all > 0 at 800-5,000 (34), max +.1111 [+.1028, +.1191] at 1,200, every
      fifth > 0 at 800-4,800 (33), P(g > 0) up to .82 (1,200). At 720 (all lags -.102): all +.0044 [-.0048, +.0125],
      fifths -.0102 / +.0036 / +.0022 / +.0131 / +.0131 (check 11-03: fifth 1 [-.0241, +.0037], fifth 4 [+.0019,
      +.0225], fifth 5 [+.0043, +.0213]), contrast +.0233 [+.0113, +.0332]; its controls (check 11-03) conf +.0124
      [+.0027, +.0216], state +.0307 [+.0165, +.0432], obs +.0009 [-.0068, +.0083] (`numbers`: conf > 0 at 140-720,
      state at 120-800, obs only at 3,000-39,996). The lag-1 contrast is > 0 at 120-800 (20; max +.050 at 260; +.0114
      [+.0011, +.0204] at 800, check 11-03) and from 5,800 on (end +.025), not separated from 0 at 880-5,600, where
      every level grows. Lags 1-4: every fifth < 0 at all 114, the largest -.0571 [95% -.0521] (fifth
      5, 1,360); lags 5-20: every fifth < 0 at all 114, largest -.0269 (fifth 1, 60); lags 21-29: every fifth < 0 at all
      114, largest -.0159 (fifth 1, 80). Contrast (all lags) > 0 at 160-39,996 (106), < 0 at 10-140 (8); max +.0338
      [+.0274, +.0400] at 560 (fifths -.129 / -.116 / -.113 / -.102 / -.095) {+.0291}; end +.0110 [+.0088, +.0136]
      {+.0134}. Controls at 560: env +.0277, usize +.0325, hist +.0060 [+.0003, +.0113], conf +.0207 [+.0151, +.0264]
      (max +.0216 at 240), state +.0583 [+.0507, +.0659] (max +.0596 at 360), obs +.0004 [-.0022, +.0031] (> 0 only from
      5,200 on, max +.0066 at the end); usizeq -.0007, confq -.0381. Incoming difference log||Delta_{t-1}||: contrast
      +.50 [+.42, +.58] at 560 (fifths -.52 -> -.02), +.096 at the end. Counterfactual own part: -.0031 (60), -.0047
      (120), -.0029 (240), -.0015 (360), -.0004 (560), -.0005 (880), -.0008 (1,680), +.0009 [+.0002, +.0017] (4,200),
      +.0032 [+.0025, +.0039] (12,000), +.0053 [+.0044, +.0061] (end).
    (3) d_H change: all lags: all > 0 at every checkpoint (+.0015 initial, max +.1215 at 300, +.0072 at the end); every
      fifth > 0 at 10-31,200 and 32,800. Contrast > 0 at 120-10,400 (71), < 0 at 34,400-39,996 (8); max +.0702 [+.0505,
      +.0932] at 240 (fifths +.075 -> +.146) {+.0591}; end -.0095 [-.0163, -.0024]. Controls at 240: env +.0530, usize
      +.0653, var +.0577, hist +.0256 [+.0125, +.0403], conf +.0510 [+.0352, +.0698], state +.1062 [+.0728, +.1440]
      (max +.1172 at 640), obs +.0016 [-.0050, +.0080]; usizeq +.0178, confq -.0547. Lag groups: lag 1 all max +.676
      (1,200); lags 1-4 all > 0 at 10-35,200 (108); lags 5-20 > 0 at 10-34,400 (107; end +.0006 [-.0015, +.0030];
      every fifth > 0 at 10-320 and 360-20,800, so late it no longer grows); contrasts max +.116 (lag 1,
      5,800), +.106 (1-4, 560), +.065 (5-20, 640). Counterfactual own part: +.0000 (60), +.0004 (120), +.0005 (240),
      +.0002 (360), +.0007 (560), +.0018 (880), +.0027 (1,680), +.0067 [+.0038, +.0100] (4,200), +.0001 (12,000), -.0107
      [-.0158, -.0051] (end). SKL change: all > 0 at 10-1,600 (36), < 0 at 2,240-39,996 (71); its most entangled fifth
      > 0 while the mean < 0 at 2,240-3,200 (7), but SKL's contrast is chart-confounded (naive with a constant d_H
      shows +.0033); contrast max +.0103 [+.0063, +.0150] at 340, end +.0029.
    (4) Update growth: lags 5-20: all < 0 at all 114 (max -.0818 at 39,200, end -.0819); most entangled fifth > 0 at
      9,600-39,996 (39), max +.0768 [+.0647, +.0889] at 39,200, end +.0766 [+.0642, +.0888], with the all-unit mean < 0
      there; contrast > 0 at 120-39,996 (108): +.084 (240), +.069 (560), +.035 (880), +.013 (1,680), +.075 (4,200),
      +.221 (12,000), end +.3054 [+.2860, +.3246] {+.3302}; P(> 0) at the end .41 (fifth 5 .55). Controls at the end:
      conf +.3265, state -.0045 [-.0087, +.0005] (state-level fifths > 0 only at 160-1,600, max +.119 at 360), obs
      +.3744. Lags 21-29: most entangled fifth > 0 at 6,600-39,996 (48), max +.103. Lags 1-4: every fifth < 0 at all 114
      (largest -.2255 at 300), contrast > 0 from 100 on (end +.286). All lags pooled: every fifth < 0 at all 114 (lags
      1-4 dominate), contrast end +.304. Counterfactual own part, lags 5-20: +.0044 (60), +.0123 (120), +.0186 (240, of
      +.084), +.0112 (360), +.0079 (560), +.0105 (880), +.0089 (1,680), +.0869 [+.0772, +.0969] (4,200, of +.075),
      +.2266 (12,000, of +.221), +.3091 [+.2928, +.3252] (end, of +.305; expected -.004); lags 1-4: +.0018 (60), +.0071
      (120), +.0145 (240), +.0066 (360), +.0022 (560), +.0027 (880), +.0057 (1,680), +.0480 (4,200), +.1455 (12,000),
      +.2437 (end).
  ESN (114): (1) log rho all -.4176 (40) -> -.3399 (end), < 0 at every unit of every checkpoint (largest -.115);
    contrast < 0 at 10-20 and 180-39,996 (min -.0167 at 11,200, end -.0115 [-.0177, -.0053]); counterfactual own part at
    880 -.0004 [-.0013, +.0006], at the end -.0023 [-.0045, -.0000]. Leading eigenvalue complex at 47% of the units.
    Euclidean stretch: > 0 at step 1 (share .996-1.000) and at under 1% of later steps (at most 246 of 29,000 units, at
    23,200; at 56 checkpoints; largest +.079) - chart-dependent, not interpreted. (2) g all -.485 (40) -> -.442 (end);
    P(g > 0) = 0 at every checkpoint and lag group; contrast < 0 at all 114 (end -.0106 [-.0139, -.0074]); own part
    -.0013 (240), -.0017 (880), -.0029 (4,200), -.0040 (end). (3) d_H change all +.0015 -> +.092 (its integrating
    read-out adds the decaying difference); contrast within -.002..+.001. (4) Update growth: every fifth < 0 at every
    checkpoint and lag group; lags 5-20 contrast < 0 at 140-39,996 (end -.0134 [-.0185, -.0087]), own part < 0 (end
    -.0072).
  Not supported: divergence (g > 0 on average) at entangled observations at any stage, in any fifth, twentieth or lag
    group except lag 1 (sense (3): first at the more entangled observations at 720 episodes, then at every level alike
    at 880-4,800); a threshold at the most entangled steps (twentieths behave like fifths); the learning-phase
    contrasts of g and d_H as effects of the step's own observation; an expanding eigen-direction at the most
    entangled steps after 1,360 episodes (their log rho < 0 at 1,680-2,800, 4,000 and from 6,200 on; neither at
    3,000-3,800 and 4,200-6,000). The 'hist' control alone (matching on the incoming difference's size) leaves +.006 /
    +.026 that the counterfactual removes, so size is not the whole history.

CHECKS (checks, cf_check -> chaos_ent/checks.json; 8 checkpoints: Trained RNN at 0, 75, 400, 2,100 and 49,995 raw
episodes, ESN at 300, 2,100 and 49,995)
  Difference-form flips vs float64 reruns, all 1,000 probes: log||Delta|| within 1.3e-5, g 4.9e-6, chi 1.1e-4, its
    change 1.7e-5, SKL 6.3e-4, its change 1.2e-4, the update size log s 6.2e-5 and its growth 4.6e-5 (all largest at
    400 raw episodes; the 11-02 check found 3.0e-5 / 5.7e-4 / 1.5e-3 for log||Delta|| / chi / SKL at 240 episodes, also
    on all probes). The direct fp32 subtraction it replaces is off by up to .15 in log||Delta||, .089 in g and 3.4 in
    log s (initial networks: differences fall to e^-14).
  log rho (Arnoldi) vs the exact eigenvalues of the explicit 2,000 x 2,000 Jacobian (float64), 36 units per checkpoint:
    24 random units median |error| <= 2.1e-7, max 7.2e-4 (the initial network), <= 6.1e-5 elsewhere; the 6 slowest
    units max 4.3e-4; the 6 with the largest residual max 2.7e-3 (ESN at 300). A second random start (100 probes =
    3,000 units per checkpoint): mean difference <= 1.8e-6, max 4.6e-3, share > 1e-4 from 0 to 2.6% (initial network).
    Units still above the 1e-5 tolerance after 5 restarts: at most 1.6% of a checkpoint (the initial Trained RNN) and
    3.2% (the ESN at 200 episodes: a random reservoir has no dominant eigenvalue), relative residual <= 3.9e-2. Before
    the harvest, on 150 random units of the last checkpoint: median 1.1e-7, max 5.3e-5, while the check's power
    iteration (400 steps) was off by up to 2.7e-3 there (clustered spectra: up to 87 eigenvalues with |lambda| > .9).
    A residual alone can accept the second-largest eigenvalue (a looser 1e-4 tolerance did so with errors up to 3.8e-3),
    and the stored 1e-5 tolerance does too: the check 11-03 (an independent float64 Arnoldi with a fixed Krylov
    dimension of 200 on all 30,000 units, arbitrated by exact eigenvalues; Trained RNN at 560, 800, 880 and 1,200
    episodes and the end, ESN at the end) found 6-168 units per checkpoint off by > 1e-4 (0.02-0.6%), at most 2.2e-2
    (1,200 episodes), 2 with the wrong sign (560, 800); among the 25 most disagreeing units at 800, 880 and the end the
    stored value is the second-largest |eigenvalue| at 10, about 20 and 11 (gaps up to 1e-2, not only near-ties).
    Every fifth mean moves by <= 8e-6; fifths, contrasts and intervals agree to 4 decimals.
  Lanczos log sigma_max vs the exact SVD within 9.8e-8. The intact forward vs the engine (ti.forward): log q within
    3.8e-5. The extension pass's flips vs the first pass's cache: log||Delta|| within 5.4e-5 (fp32 batch sizes).
  Observers vs td_tt_chaos's caches: Bayes's chi 2.8e-14, ADF's chi 0, ADF's log sigma_max 0; naive's d_H change <=
    5.7e-14; ADF's log rho from its explicit Jacobian vs log(1 + rho_1) 1.9e-9. The stored network chi vs the chaos
    sweep: within 1 fp16 ulp at every entry (max .51 ulp).
  Counterfactual at the actual observation vs the main values: networks g 7.1e-7, d_H change 5.6e-6, update growth
    7.4e-7; Bayes d_H change 1.2e-14, update growth 7.1e-12; ADF g and d_H change 2.9e-15, log rho 1.9e-9; its
    entanglement vs probes.npz 'ent' 1.2e-15; pattern weights sum to 1 within 4e-16. log rho's pattern pass vs the
    stored values: median <= 3.3e-7; units differing by > 1e-4: 1-35 of 6,000 (Trained RNN), 9 and 99 (ESN), at most
    2.2e-2 (lock-ins on the second-largest eigenvalue, above).
  Fifth counts per step: main 199-201, env 191-214, usize 197-202, conf 198-203, state 199-201, obs 199-201 (var
    399-401: two variables). confq is degenerate at step 1 (every probe has the uniform belief), so it rests on steps
    2-30.

CORRECTIONS (to the first version of 11-02, found by the first independent check)
  - "Not chaotic during entangled observations ... in any measure" overclaimed: the coordinate-free log rho(J_t) shows
    John's pattern around passing naive (most entangled fifth > 0 while the average step contracts), and the update
    measure (4) grows at entangled observations (exact Bayes too; the Trained RNN late). What holds is the narrower
    statement: realised perturbations shrink on average at every entanglement level over lags 1-4, 5-20 and 21-29;
    only the first step after a flip grows, transiently (ANSWER (3)).
  - The one-step stretch log sigma_max was called frame-free: it is free of a random frame, not of the coordinates. Its
    claims "entangled steps stretch less around passing naive", "> 0 from 2,320 on" and "ESN < 0" depend on the chart
    and are withdrawn; log rho replaces it as the headline and in the figure.
  - "Entanglement does not cause this" (the learning-phase contrasts of g and the d_H change) overreached: the
    counterfactual removes only the step's own observation. The contrasts go with the states at which observations are
    entangled, a state-level form of entanglement ('state' grouping).
  - "ESN: log sigma_max < 0 at every step but the first" was false even in the Euclidean chart: its Euclidean stretch is
    > 0 at step 1 (share .996-1.000) and at under 1% of later steps (at most 246 of 29,000 units; 56 checkpoints;
    largest +.079).
  - Pooling all lags hid the one-step growth right after a flip (lag 1: first at the more entangled observations,
    then at every entanglement level; ANSWER (3)).
  - Belief confidence was not controlled: 'conf' added. Matched on belief confidence the learning-phase contrasts
    remain > 0 and are each reduced (61%, 73%, 82% remain: g +.021 of +.034 at 560, d_H +.051 of +.070 at 240, log
    rho +.029 of +.035 at 800; main - conf, paired in every resample: +.0131 [+.0103, +.0158], +.0192 [+.0104, +.0291],
    +.0064 [+.0046, +.0081]; final check 11-03).
  - The difference-form check used 100 probes and understated the fp32 error; now all 1,000 (CHECKS).
  - Figure: the off-scale note over a band, the flat share panel, ADF's zero-sum-chart stretch against the networks'
    Euclidean one, the missing ESN-dot legend entry, overlapping dots and the ESN panels without units: all replaced or
    fixed.
  - Ranges are now quoted as `numbers` prints them; the averaging is the pooled ratio of sums, and equal-weight
    environment means agree.
CORRECTIONS (to the revision of 11-03, found by a second independent check; every number reproduced, wording only)
  - "The actual perturbations never realise it" and "the first step after a flip grows at every level alike" claimed a
    null the numbers contradict: that step grows first at the more entangled observations (720 episodes, state-level),
    then at every level alike (880-4,800). It is now the third sense, ANSWER (3).
  - log rho: "+.016 while the average step contracts" joined two checkpoints. At the +.0156 peak (800) the all-unit
    mean is flat; while it contracts, the most entangled fifth reaches +.0125 (640).
  - "The flip's belief difference accumulates at every level" was unscoped: at every level only up to 31,200
    episodes; late, at lags 5-20, it no longer grows.
  - "A factorized likelihood gives s_t = 0" holds for product-form beliefs (ADF) only, not for the joint posterior.
  - "conf keeps 61-82%" lacked the paired intervals (corrected above: all three are reduced, paired).
  - The log rho CHECKS understated the stored Arnoldi's lock-ins on the second eigenvalue (now quoted on the full
    set); smax()'s docstring still called the Euclidean stretch frame-free; the figure had an orphan 'naive' key.

DATA
  Checkpoints: all 114 stored checkpoints of the seed-20261029 Trained RNN (trained) and ESN (echo); training episodes
  = E - E // 5, the initial network drawn at 10. Caches main/DATA/experimental/training_dynamics/through_training/
  chaos_ent/: <run>/ck_EEEEE.npz (rows, smax / smax_res / smax_steps (1000, 30), logd (1000, 4 flip steps,
  5 observations, 30), chi / skl (1000, 4, 5, 30, 2 variables), float32, NaN before the flip; ~10 MB each),
  <run>/ext_EEEEE.npz (lrho (1000, 30) float64, lrho_res / lrho_steps / lrho_restarts / lrho_cx, lupd (1000, 4, 5, 30)
  float32 = log s, logd_vs_cache, stage_stats), observers.npz and observers_upd.npz (float64), series_<run>.npz (the
  per-environment sums, rebuilt when a checkpoint is added), cf/<run>_EEEEE.npz (Gk, Xk (1000, 30, 32)),
  cf/<run>_EEEEE_upd.npz (Uk (1000, 30, 32, 4 lag groups) sums, cnt), cf/<run>_EEEEE_lrho.npz (L (200, 30, 32)),
  cf/observers.npz, cf/observers_upd.npz, cf/observers_lrho.npz, checks.json, logs. 11 of the Trained RNN's
  extension files (8,000-32,000 episodes) and the ESN's log rho counterfactuals were computed by a helper on GPU 0
  with the same code.

Run from the repository root (CUDA_VISIBLE_DEVICES=1; OMP/MKL/OPENBLAS_NUM_THREADS=2; PYTHONDONTWRITEBYTECODE=1):
    /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_chaos_ent.py <command>
    commands: harvest <run> [reverse], ext [reverse] (STOP file chaos_ent/STOP), observers, cf, cfupd,
    cflrho [reverse], checks, cf_check, figure, numbers (default)"""
import os
import sys
import json
import time
import warnings
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import rankdata

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_intervene as ti                                                                  # noqa: E402
import td_tt_chaos as tc                                                                      # noqa: E402

T, R, NV, NO = ti.T, ti.R, ti.NV, ti.NO
OUT = os.path.join(gap.TT, "chaos_ent")
RUNS = ("trained", "echo")
FLIP_S = (1, 3, 5, 10)                       # flip steps (as td_tt_chaos.flip_decay); each of the 5 observations
NQ = 5                                       # entanglement fifths
LZ_M, LZ_TOL, LZ_RESTARTS, LZ_CHUNK, LZ_EVERY, SMAX_SEED = 40, 1e-4, 4, 1500, 10, 20261121
CHECK_SEED = 20261122


# ═══════════════════════════════════════════════════════════════════
# Probes, entanglement fifths
# ═══════════════════════════════════════════════════════════════════

def rows():
    """the probe subset of the chaos flips (td_tt_chaos.subset(): 20 per test environment, 1,000 episodes)"""
    return tc.subset()


def covariates(rws=None):
    """per probe-step covariates on the Bayes side (float64): ent (n, T) whole-belief entanglement of observation t =
    the mean over the 2 variables of probes.npz 'ent'; ent_v (n, T, 2) per variable; upd (n, T) Bayes's marginal update
    size at t = the mean over the 2 variables of d_H(marginal after t, marginal before t) (before step 1: uniform)"""
    rws = rows() if rws is None else rws
    Z = np.load(os.path.join(ti.AN, "probes.npz"))
    ent_v = Z["ent"][rws].astype(np.float64)
    lj = Z["log_jb"][rws].astype(np.float64)                                    # (n, T, 2, R) float32 -> float64
    prev = np.concatenate([np.full_like(lj[:, :1], -np.log(R)), lj[:, :-1]], 1)
    upd = tc.osc(lj - prev).mean(-1)
    out = dict(ent=ent_v.mean(-1), ent_v=ent_v, upd=upd, env=ti.probes()["env"][rws])
    pu, pc = os.path.join(OUT, "observers_upd.npz"), os.path.join(OUT, "cf", "observers.npz")
    if np.array_equal(rws, rows()):
        if os.path.exists(pu):
            out["hb"] = np.load(pu)["hb"]                                       # Bayes's marginal entropy before t
        if os.path.exists(pc):
            O = np.load(pc)
            out["ent_state"] = (O["ent_k"] * O["w"][:, None, :]).sum(-1)        # the state's expected entanglement
    return out


def fifths(x, groups=None, nb=None):
    """(n, T) fifth 0..4 (or bin 0..nb - 1) of x ranked within each step over the probes (ties share their average
    rank); groups (n,) or (n, T): rank within each step AND group"""
    nb = NQ if nb is None else nb
    n, S = x.shape
    q = np.zeros((n, S), np.int64)
    G = np.zeros((n, S), np.int64) if groups is None else np.broadcast_to(
        np.asarray(groups).reshape(n, -1), (n, S))
    for t in range(S):
        for gv in np.unique(G[:, t]):
            ix = np.flatnonzero(G[:, t] == gv)
            rk = rankdata(x[ix, t], method="average")
            q[ix, t] = np.minimum(nb - 1, np.floor(nb * (rk - 1) / len(ix)).astype(np.int64))
    return q


# ═══════════════════════════════════════════════════════════════════
# Networks (GPU): intact run, one-step stretch, flips in difference form
# ═══════════════════════════════════════════════════════════════════

def dsig(a, d):
    """sigma(a + d) - sigma(a) without cancellation: tanh(d / 2) (1 - tanh(a / 2) tanh((a + d) / 2)) / 2"""
    return .5 * torch.tanh(.5 * d) * (1. - torch.tanh(.5 * a) * torch.tanh(.5 * (a + d)))


def dtanh(a, d):
    """tanh(a + d) - tanh(a) without cancellation: tanh(d) (1 - tanh(a) tanh(a + d))"""
    return torch.tanh(d) * (1. - torch.tanh(a) * torch.tanh(a + d))


def setup(run, raw, dev, dtype=torch.float32):
    """weights on the device, the drive table (NE, 32, 4H) and the probes' environment / observation-code tensors"""
    w = ti.load(run, raw)
    p = {k: v.to(dev, dtype) for k, v in w.items()}
    table = ti._table({k: v.to(dev, torch.float32) for k, v in w.items()}, ti.probes()["Z"], dev).to(dtype)
    return p, table


def codes(rws, dev):
    P = ti.probes()
    code = torch.as_tensor((P["obs"][rws] * (1 << np.arange(NO))).sum(-1).astype(np.int64), device=dev)
    envt = torch.as_tensor(P["env"][rws].astype(np.int64), device=dev)
    return envt, code


def replay(p, table, envt, code):
    """the intact run (td_tt_chaos.lstm_step): pre-activations A (b, T, 4H), states U, C (b, T + 1, H) (index 0 = the
    start state), per-step read-out increments lg (b, T, 20) float64"""
    Whh = p["LSTM.weight_hh_l0"]
    H = Whh.shape[1]
    b = len(envt)
    dt = Whh.dtype
    h = p["STM"].reshape(1, H).expand(b, H).contiguous()
    c = p["LTM"].reshape(1, H).expand(b, H).contiguous()
    A = torch.empty(b, T, 4 * H, device=Whh.device, dtype=dt)
    U = torch.empty(b, T + 1, H, device=Whh.device, dtype=dt)
    C = torch.empty(b, T + 1, H, device=Whh.device, dtype=dt)
    U[:, 0], C[:, 0] = h, c
    for t in range(T):
        xin = table[envt, code[:, t]]
        A[:, t] = xin + h @ Whh.T
        h, c, _ = tc.lstm_step(xin, h, c, Whh)
        U[:, t + 1], C[:, t + 1] = h, c
    lg = torch.nn.functional.linear(U[:, 1:], p["classifier_readout.weight"], p["classifier_readout.bias"]).double()
    return A, U, C, lg


def _gates(a):
    ai, af, ag, ao = a.chunk(4, -1)
    return ai.sigmoid(), af.sigmoid(), ag.tanh(), ao.sigmoid()


def flips(p, table, envt, code, A, C, lg, flip_s=FLIP_S, upd=False):
    """one flipped observation (each of the 5 at each step of flip_s), the network rerun from its intact state with the
    intact observations after; the flipped-minus-intact difference Delta = (du, dc) is propagated in difference form
    (dsig, dtanh: exact identities, so Delta keeps fp32 relative precision however small it gets; checks() compares it
    with a float64 rerun). Returns logd (b, nS, 5, T) = log ||Delta(u, c)_t||, chi / skl (b, nS, 5, T, 2) per variable:
    d_H and the symmetric KL between the intact and flipped marginal beliefs at t (NaN before the flip step); upd: also
    lupd (b, nS, 5, T) = log of the size of the flip's difference in the step's belief update, the mean over the 2
    variables of osc(W_ro du_t) (the network's log marginals at t move by the read-out of u_t, up to a constant per
    variable, so W_ro du_t is the flipped minus intact update; osc = max - min = its d_H size)"""
    Whh, Wro = p["LSTM.weight_hh_l0"], p["classifier_readout.weight"]
    H = Whh.shape[1]
    b = len(envt)
    dev = Whh.device
    nS = len(flip_s)
    logd = torch.full((b, nS, NO, T), float("nan"), device=dev, dtype=torch.float64)
    chi = torch.full((b, nS, NO, T, NV), float("nan"), device=dev, dtype=torch.float64)
    skl = torch.full((b, nS, NO, T, NV), float("nan"), device=dev, dtype=torch.float64)
    lupd = torch.full((b, nS, NO, T), float("nan"), device=dev, dtype=torch.float64)
    Lam = lg.cumsum(1).reshape(b, T, NV, R)                                       # intact cumulative logits (float64)
    lq = torch.log_softmax(Lam, -1)
    bits = torch.as_tensor(1 << np.arange(NO), device=dev).repeat(b)
    eb = envt.repeat_interleave(NO)
    for k, s in enumerate(flip_s):
        cd = code[:, s - 1].repeat_interleave(NO)
        dA = table[eb, cd ^ bits] - table[eb, cd]
        dh = torch.zeros(b * NO, H, device=dev, dtype=Whh.dtype)
        dc = torch.zeros(b * NO, H, device=dev, dtype=Whh.dtype)
        D = torch.zeros(b * NO, NV, R, device=dev, dtype=torch.float64)
        for t in range(s - 1, T):
            if t > s - 1:
                dA = dh @ Whh.T
            a = A[:, t].repeat_interleave(NO, 0)
            i, f, g, o = _gates(a)
            di, df, dg, do = dA.chunk(4, -1)
            ai, af, ag, ao = a.chunk(4, -1)
            Di, Df, Dg, Do = dsig(ai, di), dsig(af, df), dtanh(ag, dg), dsig(ao, do)
            cp = C[:, t].repeat_interleave(NO, 0)
            cn = C[:, t + 1].repeat_interleave(NO, 0)
            dc = Df * cp + (f + Df) * dc + Di * (g + Dg) + i * Dg
            dh = Do * torch.tanh(cn + dc) + o * dtanh(cn, dc)
            logd[:, k, :, t] = torch.log(torch.cat([dh, dc], -1).double().norm(dim=-1)).reshape(b, NO)
            inc = (dh @ Wro.T).double().reshape(-1, NV, R)
            D += inc
            if upd:
                lupd[:, k, :, t] = torch.log((inc.amax(-1) - inc.amin(-1)).mean(-1)).reshape(b, NO)
            chi[:, k, :, t] = (D.amax(-1) - D.amin(-1)).reshape(b, NO, NV)
            L0 = lq[:, t].repeat_interleave(NO, 0)                                # (b NO, NV, R) intact log beliefs
            dl = D - (torch.logsumexp(L0 + D, -1, keepdim=True))                  # log q' - log q (L0 is normalised)
            skl[:, k, :, t] = (.5 * (L0.exp() * torch.expm1(dl) * dl).sum(-1)).reshape(b, NO, NV)
    return (logd, chi, skl, lupd) if upd else (logd, chi, skl)


def _lanczos(mv, q0, m, tol, every=LZ_EVERY):
    """Lanczos on the symmetric PSD maps mv (one per row of q0 (n, d), unit vectors), full reorthogonalisation (two
    passes per step), at most m steps; every `every` steps the top Ritz value theta of the tridiagonal (float64, CPU)
    and its residual r = beta_j |y_j| are checked and a row stops when r <= tol theta. Returns theta, r (n,) float64,
    the steps (n,) and the top Ritz vectors (n, d)"""
    n, d = q0.shape
    dev, dt = q0.device, q0.dtype
    Q = torch.zeros(m + 1, n, d, device=dev, dtype=dt)                          # step-major: Q[:j + 1] contiguous
    Q[0] = q0
    al = torch.zeros(n, m, device=dev, dtype=torch.float64)
    be = torch.zeros(n, m, device=dev, dtype=torch.float64)
    th_o, r_o, st_o = np.zeros(n), np.zeros(n), np.zeros(n, np.int64)
    y_o = torch.zeros(n, d, device=dev, dtype=dt)
    ids = np.arange(n)
    for j in range(m):
        v = Q[j]
        w = mv(v, ids)
        a_ = (w * v).sum(-1)
        w = w - a_[:, None] * v
        if j > 0:
            w = w - be[:, j - 1, None].to(dt) * Q[j - 1]
        for _ in range(2):
            Qj = Q[:j + 1]
            w = w - torch.einsum("kn,knd->nd", torch.einsum("knd,nd->kn", Qj, w), Qj)
        bt = w.norm(dim=-1)
        al[:, j], be[:, j] = a_.double(), bt.double()
        Q[j + 1] = w / bt[:, None].clamp_min(1e-30)
        if (j + 1) % every and j < m - 1:
            continue
        Tm = (torch.diag_embed(al[:, :j + 1]) + torch.diag_embed(be[:, :j], 1) + torch.diag_embed(be[:, :j], -1)).cpu()
        ev, evec = torch.linalg.eigh(Tm)
        th, r = ev[:, -1], be[:, j].cpu() * evec[:, -1, -1].abs()
        done = ((r <= tol * th) | (j == m - 1)).numpy()
        if done.any():
            k_ = ids[done]
            th_o[k_], r_o[k_], st_o[k_] = th[done].numpy(), r[done].numpy(), j + 1
            dm = torch.as_tensor(done, device=dev)
            y_o[torch.as_tensor(k_, device=dev)] = torch.einsum("nk,knd->nd", evec[done, :, -1].to(dev, dt),
                                                                Q[:j + 1, dm])
            keep = ~done
            if not keep.any():
                break
            ids, km = ids[keep], torch.as_tensor(keep, device=dev)
            Q, al, be = Q[:, km], al[km], be[km]
    return th_o, r_o, st_o, y_o


def smax(p, A, C, m=LZ_M, tol=LZ_TOL, restarts=LZ_RESTARTS, seed=SMAX_SEED, chunk=LZ_CHUNK):
    """log sigma_max (b, T) of every step's Jacobian J_t of (u, c) (Euclidean, 2H dims; td_tt_chaos.tangent / adjoint at
    the intact states), free of a random frame (chart-dependent): Lanczos on J_t^T J_t (_lanczos) from one random unit
    vector, m steps, restarted from the top Ritz vector (up to `restarts` times) where the residual is still above tol
    theta. |theta - lambda| <= r for an eigenvalue lambda of J^T J, so log sigma_max = log(theta) / 2 is within tol / 2
    of an eigenvalue's (theta is a Rayleigh quotient: a lower bound of sigma_max^2). Returns the estimate (b, T), the
    relative residual r / theta (b, T) and the Lanczos steps used (b, T)"""
    Whh = p["LSTM.weight_hh_l0"]
    H = Whh.shape[1]
    dev = Whh.device
    b = A.shape[0]
    N = b * T
    gall = tuple(x.reshape(N, H) for x in _gates(A)) + (torch.tanh(C[:, 1:]).reshape(N, H),)
    call = C[:, :-1].reshape(N, H)
    est, res, nst = np.zeros(N), np.zeros(N), np.zeros(N, np.int64)
    gen = torch.Generator(device=dev).manual_seed(seed)
    for c0 in range(0, N, chunk):
        sel = np.arange(c0, min(N, c0 + chunk))
        selt = torch.as_tensor(sel, device=dev)
        gts, cp = tuple(x[selt] for x in gall), call[selt]

        def mv(v, ids):
            it = torch.as_tensor(ids, device=dev)
            g_, c_ = tuple(x[it] for x in gts), cp[it]
            dh, dc = tc.tangent(v[:, None, :H], v[:, None, H:], c_, g_, Whh)
            dh, dc = tc.adjoint(dh, dc, c_, g_, Whh)
            return torch.cat([dh, dc], -1)[:, 0]
        q = torch.randn(len(sel), 2 * H, device=dev, generator=gen, dtype=Whh.dtype)
        q = q / q.norm(dim=-1, keepdim=True)
        todo = np.arange(len(sel))
        for rs in range(restarts + 1):
            th, r, st, y = _lanczos(lambda v, ids: mv(v, todo[ids]), q, m, tol)
            est[sel[todo]] = .5 * np.log(th)
            res[sel[todo]] = r / th
            nst[sel[todo]] += st
            bad = r > tol * th
            if not bad.any():
                break
            todo, q = todo[bad], y[torch.as_tensor(bad, device=dev)]
            q = q / q.norm(dim=-1, keepdim=True)
    return est.reshape(b, T), res.reshape(b, T), nst.reshape(b, T)


@torch.no_grad()
def one(run, raw, dev, rws=None, chunk=250):
    """every network measure of one checkpoint on the probes rws (rows() if None)"""
    rws = rows() if rws is None else rws
    p, table = setup(run, raw, dev)
    envt, code = codes(rws, dev)
    n = len(rws)
    out = dict(smax=np.zeros((n, T), np.float32), smax_res=np.zeros((n, T), np.float32),
               smax_steps=np.zeros((n, T), np.int16),
               logd=np.zeros((n, len(FLIP_S), NO, T), np.float32),
               chi=np.zeros((n, len(FLIP_S), NO, T, NV), np.float32),
               skl=np.zeros((n, len(FLIP_S), NO, T, NV), np.float32))
    for b0 in range(0, n, chunk):
        sl = slice(b0, min(n, b0 + chunk))
        A, U, C, lg = replay(p, table, envt[sl], code[sl])
        out["smax"][sl], out["smax_res"][sl], out["smax_steps"][sl] = smax(p, A, C)
        ld, chi, skl = flips(p, table, envt[sl], code[sl], A, C, lg)
        out["logd"][sl], out["chi"][sl], out["skl"][sl] = (x.float().cpu().numpy() for x in (ld, chi, skl))
        del A, U, C, lg
    return out


# ═══════════════════════════════════════════════════════════════════
# One-step expansion, coordinate-free: log rho(J_t), the spectral radius of the step Jacobian, by Arnoldi on batched
# JVPs (GPU, fp32) with the small eigenproblems on 2 CPU threads
# ═══════════════════════════════════════════════════════════════════

LR_STAGES, LR_TOL, LR_RESTARTS, LR_CHUNK, LR_SEED, LR_SKIP = (60, 90, 120), 1e-5, 5, 100, 20261124, .1
_POOL = []


def _eig(Hm):
    """np.linalg.eig of a batch of small matrices on 2 threads (MKL runs one thread per small matrix: 2x faster)"""
    if len(Hm) < 64:
        return np.linalg.eig(Hm)
    if not _POOL:
        from concurrent.futures import ThreadPoolExecutor
        _POOL.append(ThreadPoolExecutor(2))
    parts = list(_POOL[0].map(np.linalg.eig, np.array_split(Hm, 2)))
    return np.concatenate([a for a, _ in parts]), np.concatenate([v for _, v in parts])


def _ritz_top(Hh, m):
    """the top-modulus Ritz value theta (n,) complex of the m-step Arnoldi Hessenberg matrices Hh (n, M + 1, M) (numpy
    float64), its relative residual |h_{m+1,m} y_m| / |theta| (y = its unit eigenvector of H_m; = ||J x - theta x|| /
    |theta| for the Ritz vector x = Q_m y) and y (n, m)"""
    ev, V = _eig(Hh[:, :m, :m])
    ar = np.arange(len(ev))
    k = np.argmax(np.abs(ev), -1)
    th, y = ev[ar, k], V[ar, :, k]
    return th, np.abs(Hh[:, m, m - 1] * y[:, -1]) / np.maximum(np.abs(th), 1e-300), y


def arnoldi_rho(mv, n, d, dev, seed=LR_SEED, stages=LR_STAGES, tol=LR_TOL, restarts=LR_RESTARTS, skip=()):
    """the top-modulus eigenvalue of n linear maps (mv(v, ids) = J v for the units ids; v (len(ids), d), fp32) by
    Arnoldi from one random unit vector each with full reorthogonalisation (classical Gram-Schmidt twice), checked at
    the Krylov dimensions `stages` (those in `skip` not checked, except the last): a unit stops when its top-modulus Ritz
    value's relative residual is <= tol; units above it at the last stage restart from the real plus imaginary part of
    that Ritz vector, at most `restarts` times (the last value is kept whatever its residual). Returns theta (n,)
    complex, the relative residual (n,), the Krylov steps (n,), the restarts used (n,) and {stage: [stopped, checked]}
    of the first pass"""
    gen = torch.Generator(device=dev).manual_seed(seed)
    q0 = torch.randn(n, d, device=dev, generator=gen)
    q0 = q0 / q0.norm(dim=-1, keepdim=True)
    th_o, r_o = np.zeros(n, complex), np.full(n, np.inf)
    st_o, rs_o = np.zeros(n, np.int64), np.zeros(n, np.int64)
    stat = {m: [0, 0] for m in stages}
    todo = np.arange(n)
    M = stages[-1]
    for rs in range(restarts + 1):
        ids = todo
        Q = torch.zeros(M + 1, len(ids), d, device=dev, dtype=q0.dtype)              # step-major: Q[:j + 1] contiguous
        Hh = torch.zeros(len(ids), M + 1, M, device=dev, dtype=torch.float64)
        Q[0] = q0
        j = 0
        for m in stages:
            if m in skip and m != M:
                continue
            j0 = j
            while j < m:
                w = mv(Q[j], ids)
                Qj = Q[:j + 1]
                h = torch.einsum("knd,nd->kn", Qj, w)
                w = w - torch.einsum("kn,knd->nd", h, Qj)
                h2 = torch.einsum("knd,nd->kn", Qj, w)
                w = w - torch.einsum("kn,knd->nd", h2, Qj)
                bt = w.norm(dim=-1)
                Hh[:, :j + 1, j] = (h + h2).T.double()
                Hh[:, j + 1, j] = bt.double()
                Q[j + 1] = w / bt.clamp_min(1e-30)[:, None]
                j += 1
            th, r, y = _ritz_top(Hh.cpu().numpy(), m)
            st_o[ids] += m - j0
            last = m == M
            fin = (r <= tol) | (last & (rs == restarts))
            if rs == 0:
                stat[m][0] += int((r <= tol).sum())
                stat[m][1] += len(ids)
            th_o[ids[fin]], r_o[ids[fin]], rs_o[ids[fin]] = th[fin], r[fin], rs
            if fin.all():
                return th_o, r_o, st_o, rs_o, stat
            if last:
                bad = torch.as_tensor(~fin, device=dev)
                yb = torch.as_tensor(y[~fin], device=dev)
                z = torch.einsum("nk,knd->nd", (yb.real + yb.imag).to(Q.dtype), Q[:m, bad])
                q0 = z / z.norm(dim=-1, keepdim=True)
                todo = ids[~fin]
                break
            km = torch.as_tensor(~fin, device=dev)
            ids = ids[~fin]
            Q = Q[:, km].contiguous()
            Hh = Hh[km].contiguous()
    return th_o, r_o, st_o, rs_o, stat


@torch.no_grad()
def lrho_units(p, A, C, seed=LR_SEED, skip=()):
    """log rho(J_t) (b, T): the log spectral radius of every step's Jacobian J_t = d(u_t, c_t) / d(u_{t-1}, c_{t-1})
    (td_tt_chaos.tangent at the intact states and observations of replay()), by arnoldi_rho; also the relative
    residual, Krylov steps, restarts (b, T), whether the top eigenvalue is complex (b, T) and the stage counts"""
    Whh = p["LSTM.weight_hh_l0"]
    H = Whh.shape[1]
    dev = Whh.device
    b = A.shape[0]
    N = b * T
    gall = tuple(x.reshape(N, H) for x in _gates(A)) + (torch.tanh(C[:, 1:]).reshape(N, H),)
    call = C[:, :-1].reshape(N, H)

    def mv(v, ids):
        it = torch.as_tensor(ids, device=dev)
        dh, dc = tc.tangent(v[:, None, :H], v[:, None, H:], call[it], tuple(x[it] for x in gall), Whh)
        return torch.cat([dh, dc], -1)[:, 0]
    th, res, st, rs, stat = arnoldi_rho(mv, N, 2 * H, dev, seed, skip=skip)
    return (np.log(np.abs(th)).reshape(b, T), res.reshape(b, T), st.reshape(b, T), rs.reshape(b, T),
            (np.abs(th.imag) > 0).reshape(b, T), stat)


GPU_CAP_GB = 8                                   # per-process GPU memory cap (the caching allocator stays below it)


def _gpu_cap(dev):
    if dev.type == "cuda":
        tot = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., GPU_CAP_GB * 2 ** 30 / tot), dev)
    return dev


def ext_file(run, raw):
    return os.path.join(OUT, run, f"ext_{int(raw):05d}.npz")


@torch.no_grad()
def ext_one(run, raw, dev, rws=None, chunk=LR_CHUNK):
    """the extension measures of one checkpoint on the probes rws: lrho (n, T) float64 and its convergence records
    (lrho_res, lrho_steps, lrho_restarts, lrho_cx), lupd (n, nS, 5, T) float32 (flips(upd=True)); stages whose checks
    stopped < LR_SKIP of the units in a chunk are skipped in the later chunks (more Krylov steps, never less accurate).
    Also the largest |logd - the cached logd| (the same flips) and the stage counts"""
    rws = rows() if rws is None else rws
    p, table = setup(run, raw, dev)
    envt, code = codes(rws, dev)
    n = len(rws)
    out = dict(lrho=np.zeros((n, T)), lrho_res=np.zeros((n, T), np.float32), lrho_steps=np.zeros((n, T), np.int16),
               lrho_restarts=np.zeros((n, T), np.int8), lrho_cx=np.zeros((n, T), bool),
               lupd=np.zeros((n, len(FLIP_S), NO, T), np.float32))
    ck = np.load(ck_file(run, raw))["logd"] if os.path.exists(ck_file(run, raw)) else None
    dlogd, skip, stats = 0., set(), {m: [0, 0] for m in LR_STAGES}

    def part(b0, b1):
        nonlocal dlogd
        sl = slice(b0, b1)
        A, U, C, lg = replay(p, table, envt[sl], code[sl])
        lr, res, st, rs, cx, stat = lrho_units(p, A, C, LR_SEED + b0, skip)
        out["lrho"][sl], out["lrho_res"][sl], out["lrho_steps"][sl] = lr, res, st
        out["lrho_restarts"][sl], out["lrho_cx"][sl] = rs, cx
        for m, (s_, c_) in stat.items():
            stats[m][0] += s_
            stats[m][1] += c_
            if m != LR_STAGES[-1] and c_ and s_ / c_ < LR_SKIP:
                skip.add(m)
        ld, _, _, lu = flips(p, table, envt[sl], code[sl], A, C, lg, upd=True)
        out["lupd"][sl] = lu.float().cpu().numpy()
        if ck is not None:
            a_, b_ = ld.float().cpu().numpy(), ck[sl]
            ok = np.isfinite(a_) & np.isfinite(b_)
            dlogd = max(dlogd, float(np.abs(a_[ok] - b_[ok]).max()))

    def run_part(b0, b1):
        oom = False
        try:
            part(b0, b1)
        except torch.cuda.OutOfMemoryError:                                    # below the cap: halve the chunk
            oom = True
        if oom:                                         # outside the except block: the traceback's tensors are freed
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            assert b1 - b0 > 10, "GPU memory: chunk of 10 probes still too large"
            mid = (b0 + b1) // 2
            print(f"  out of GPU memory at probes {b0}-{b1}: halves", flush=True)
            run_part(b0, mid)
            run_part(mid, b1)
    for b0 in range(0, n, chunk):
        run_part(b0, min(n, b0 + chunk))
    return out, dlogd, stats


def ext_jobs():
    """(run, raw) of both networks: every third checkpoint and the last of each first, then the rest"""
    first, rest = [], []
    for run in RUNS:
        eps = ti.checkpoints(run)
        for i, e in enumerate(eps):
            (first if i % 3 == 0 or i == len(eps) - 1 else rest).append((run, e))
    return first + rest


def ext_harvest(reverse=False, dev=None):
    """the extension measures of every checkpoint of both networks -> chaos_ent/<run>/ext_EEEEE.npz (resumable; a second
    process takes the job list from the other end (reverse); a job being computed holds <file>.lock; stops before the
    next job when chaos_ent/STOP exists; GPU memory capped at GPU_CAP_GB)"""
    dev = _gpu_cap(ti._device(dev))
    for run in RUNS:
        os.makedirs(os.path.join(OUT, run), exist_ok=True)
    rws = rows()
    jobs = ext_jobs()
    for run, raw in (jobs[::-1] if reverse else jobs):
        if os.path.exists(os.path.join(OUT, "STOP")):
            print("STOP file: stopping", flush=True)
            return
        f = ext_file(run, raw)
        if os.path.exists(f):
            continue
        try:
            fd = os.open(f + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.close(fd)
        t0 = time.time()
        o, dl, stats = ext_one(run, raw, dev, rws)
        tmp = f[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, rows=rws, raw=raw, x=gap.train_x(raw), logd_vs_cache=dl,
                            stage_stats=np.array([stats[m] for m in LR_STAGES]), **o)
        os.replace(tmp, f)
        os.remove(f + ".lock")
        res = o["lrho_res"]
        print(f"{run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s; log rho {o['lrho'].mean():+.4f} (> 0 "
              f"{(o['lrho'] > 0).mean():.3f}); res max {res.max():.1e} (> tol {(res > LR_TOL).mean():.4f}); steps mean "
              f"{o['lrho_steps'].mean():.0f}; stages " + " ".join(f"{m}:{s_}/{c_}" for m, (s_, c_) in stats.items())
              + f"; |logd - cache| {dl:.1e}", flush=True)


# ═══════════════════════════════════════════════════════════════════
# Exact observers (CPU, float64): Bayes's marginals, naive, factorized-state Bayes (ADF)
# ═══════════════════════════════════════════════════════════════════

def _nrm(lx):
    """normalise log beliefs over the last axis"""
    return lx - logsumexp(lx, -1, keepdims=True)


def _dh_skl(lp, lq):
    """d_H and the symmetric KL (..., ) between normalised log beliefs lp (intact) and lq (flipped) over the last axis:
    osc(lq - lp) and Sigma_r p_r expm1(delta_r) delta_r / 2 with delta = lq - lp (= Sigma (q - p)(log q - log p) / 2)"""
    dl = lq - lp
    return tc.osc(dl), .5 * (np.exp(lp) * np.expm1(dl) * dl).sum(-1)


def observers(rws=None):
    """the exact observers under the same flips (float64): chi_<o>, skl_<o> (n, nS, 5, T, 2) for o in bayes (the joint
    posterior's marginals), naive, adf (NaN before the flip step); logd_adf (n, nS, 5, T) = log of the Euclidean norm
    of ADF's centred log-marginal difference (its state in the orthonormal zero-sum chart, td_tt_chaos.adf_lyap's);
    smax_adf (n, T) = log sigma_max of ADF's step Jacobian in that chart (exact SVD), lrho_adf (n, T) = log(1 + rho_1),
    its stretch in the step's own metric (T3). Joint Bayes and naive: their state difference is the flip's own
    log-likelihood change, constant after the flip (T1), so their state growth and one-step stretch are exactly 0"""
    rws = rows() if rws is None else rws
    P = ti.probes()
    L, _ = tc.tables()
    obs, env = P["obs"][rws], P["env"][rws]
    Le = L[env]
    n = len(rws)
    ll = tc.step_loglik(obs, Le)                                                # (n, T, R, R)
    u = np.cumsum(ll, 1)
    lm = _nrm(tc.marg_log(u))                                                   # (n, T, 2, R)
    logit = np.log(Le) - np.log1p(-Le)                                          # (n, 5, R, R)
    Lb = np.stack([Le.mean(3), Le.mean(2)], 2)                                  # naive tables (n, 5, 2, R)
    logit_n = np.log(Lb) - np.log1p(-Lb)
    o = obs.astype(np.float64)
    lpx = np.einsum("nti,nikr->ntkr", o, np.log(Lb)) + np.einsum("nti,nikr->ntkr", 1 - o, np.log1p(-Lb))
    cl = np.cumsum(lpx, 1)
    ln = _nrm(cl)
    la = tc.adf_run(ll)                                                         # (n, T, 2, R), normalised
    nS = len(FLIP_S)
    out = {f"{k}_{a}": np.full((n, nS, NO, T, NV), np.nan) for k in ("chi", "skl") for a in ("bayes", "naive", "adf")}
    out["logd_adf"] = np.full((n, nS, NO, T), np.nan)
    for k, s in enumerate(FLIP_S):
        sign = 1. - 2. * obs[:, s - 1].astype(np.float64)                       # (n, 5)
        for j in range(NO):
            d = sign[:, j, None, None] * logit[:, j]                            # change of log p(x_s | r0, r1)
            lm2 = _nrm(tc.marg_log(u[:, s - 1:] + d[:, None]))
            out["chi_bayes"][:, k, j, s - 1:], out["skl_bayes"][:, k, j, s - 1:] = _dh_skl(lm[:, s - 1:], lm2)
            dn = sign[:, j, None, None] * logit_n[:, j]                         # (n, 2, R)
            ln2 = _nrm(cl[:, s - 1:] + dn[:, None])
            out["chi_naive"][:, k, j, s - 1:], out["skl_naive"][:, k, j, s - 1:] = _dh_skl(ln[:, s - 1:], ln2)
            ll2 = ll[:, s - 1:].copy()
            ll2[:, 0] += d
            la2 = tc.adf_run(ll2, None if s == 1 else np.exp(la[:, s - 2]))
            out["chi_adf"][:, k, j, s - 1:], out["skl_adf"][:, k, j, s - 1:] = _dh_skl(la[:, s - 1:], la2)
            dl = la2 - la[:, s - 1:]
            dl = dl - dl.mean(-1, keepdims=True)
            out["logd_adf"][:, k, j, s - 1:] = .5 * np.log((dl ** 2).sum((-1, -2)))
    la_prev = np.concatenate([np.full((n, 1, 2, R), -np.log(R)), la[:, :-1]], 1)
    J, rho = tc.adf_jacobians(ll, la_prev)
    out["smax_adf"] = np.log(np.linalg.svd(J, compute_uv=False)[..., 0])
    out["lrho_adf"] = np.log1p(rho[..., 0])
    out["rows"] = rws
    return out


def observers_cached():
    path = os.path.join(OUT, "observers.npz")
    if not os.path.exists(path):
        os.makedirs(OUT, exist_ok=True)
        t0 = time.time()
        O = observers()
        tmp = path[:-4] + ".tmp.npz"
        np.savez(tmp, **O)
        os.replace(tmp, path)
        print(f"observers: {time.time() - t0:.0f} s", flush=True)
    return dict(np.load(path))


def _upd_size(dl):
    """log of the size of the step's update difference from the flipped-minus-intact log marginals dl (n, S, 2, R)
    from the flip step on: the mean over the 2 variables of osc(dl_t - dl_{t-1}) (dl before the flip = 0)"""
    du = np.diff(np.concatenate([np.zeros_like(dl[:, :1]), dl], 1), axis=1)
    return np.log(tc.osc(du).mean(-1))


def observers_upd(rws=None):
    """the exact observers' update measure and references (float64, CPU; the probes and flips of observers()): lupd_bayes
    / lupd_adf (n, nS, 5, T) = log of the size of the flip's difference in the step's marginal belief update, the mean
    over the 2 variables of osc((log m'_t - log m_t) - (log m'_{t-1} - log m_{t-1})) (at the flip step osc(log m'_s -
    log m_s)), NaN before the flip; naive's is exactly 0 after the flip step (its update does not depend on its belief,
    T1) and is left out. lrho_adf_eig (n, T) = log of the largest |eigenvalue| of ADF's exact step Jacobian
    (td_tt_chaos.adf_jacobians, zero-sum chart; = log(1 + rho_1), T3: lrho_adf_cc); hb (n, T) = Bayes's marginal
    entropy before t, the mean of the 2 variables (uniform before step 1)"""
    rws = rows() if rws is None else rws
    P = ti.probes()
    L, _ = tc.tables()
    obs, env = P["obs"][rws], P["env"][rws]
    Le = L[env]
    n = len(rws)
    ll = tc.step_loglik(obs, Le)
    u = np.cumsum(ll, 1)
    lm = _nrm(tc.marg_log(u))
    logit = np.log(Le) - np.log1p(-Le)
    la = tc.adf_run(ll)
    out = {k: np.full((n, len(FLIP_S), NO, T), np.nan) for k in ("lupd_bayes", "lupd_adf")}
    for k, s in enumerate(FLIP_S):
        sign = 1. - 2. * obs[:, s - 1].astype(np.float64)
        for j in range(NO):
            d = sign[:, j, None, None] * logit[:, j]
            out["lupd_bayes"][:, k, j, s - 1:] = _upd_size(_nrm(tc.marg_log(u[:, s - 1:] + d[:, None])) - lm[:, s - 1:])
            ll2 = ll[:, s - 1:].copy()
            ll2[:, 0] += d
            la2 = tc.adf_run(ll2, None if s == 1 else np.exp(la[:, s - 2]))
            out["lupd_adf"][:, k, j, s - 1:] = _upd_size(la2 - la[:, s - 1:])
    la_prev = np.concatenate([np.full((n, 1, 2, R), -np.log(R)), la[:, :-1]], 1)
    J, rho = tc.adf_jacobians(ll, la_prev)
    out["lrho_adf_eig"] = np.log(np.abs(np.linalg.eigvals(J)).max(-1))
    out["lrho_adf_cc"] = np.log1p(rho[..., 0])
    lp = np.concatenate([np.full((n, 1, 2, R), -np.log(R)), lm[:, :-1]], 1)
    out["hb"] = -(np.exp(lp) * lp).sum(-1).mean(-1)
    out["rows"] = rws
    return out


def observers_upd_cached():
    path = os.path.join(OUT, "observers_upd.npz")
    if not os.path.exists(path):
        t0 = time.time()
        O = observers_upd()
        tmp = path[:-4] + ".tmp.npz"
        np.savez(tmp, **O)
        os.replace(tmp, path)
        print(f"observers_upd: {time.time() - t0:.0f} s", flush=True)
    return dict(np.load(path))


# ═══════════════════════════════════════════════════════════════════
# Float64 reference of the flips (checks)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def flips_direct(p, table, envt, code, U, C, lg, flip_s=FLIP_S):
    """the flips by rerunning the network and subtracting (td_tt_chaos.network's way), in the dtype of p (float64:
    the reference for checks(); float32: the direct fp32 subtraction the difference form replaces). Same outputs as
    flips(upd=True)"""
    Whh, Wro, bro = p["LSTM.weight_hh_l0"], p["classifier_readout.weight"], p["classifier_readout.bias"]
    b = len(envt)
    dev = Whh.device
    nS = len(flip_s)
    logd = torch.full((b, nS, NO, T), float("nan"), device=dev, dtype=torch.float64)
    chi = torch.full((b, nS, NO, T, NV), float("nan"), device=dev, dtype=torch.float64)
    skl = torch.full((b, nS, NO, T, NV), float("nan"), device=dev, dtype=torch.float64)
    lupd = torch.full((b, nS, NO, T), float("nan"), device=dev, dtype=torch.float64)
    lq = torch.log_softmax(lg.cumsum(1).reshape(b, T, NV, R), -1)
    bits = torch.as_tensor(1 << np.arange(NO), device=dev).repeat(b)
    eb = envt.repeat_interleave(NO)
    for k, s in enumerate(flip_s):
        hb, cb = U[:, s - 1].repeat_interleave(NO, 0), C[:, s - 1].repeat_interleave(NO, 0)
        D = torch.zeros(b * NO, NV, R, device=dev, dtype=torch.float64)
        for t in range(s - 1, T):
            cd = code[:, t].repeat_interleave(NO)
            if t == s - 1:
                cd = cd ^ bits
            hb, cb, _ = tc.lstm_step(table[eb, cd], hb, cb, Whh)
            d = torch.cat([hb - U[:, t + 1].repeat_interleave(NO, 0), cb - C[:, t + 1].repeat_interleave(NO, 0)], -1)
            logd[:, k, :, t] = torch.log(d.double().norm(dim=-1)).reshape(b, NO)
            inc = torch.nn.functional.linear(hb, Wro, bro).double() - lg[:, t].repeat_interleave(NO, 0)
            inc = inc.reshape(-1, NV, R)
            D += inc
            lupd[:, k, :, t] = torch.log((inc.amax(-1) - inc.amin(-1)).mean(-1)).reshape(b, NO)
            chi[:, k, :, t] = (D.amax(-1) - D.amin(-1)).reshape(b, NO, NV)
            L0 = lq[:, t].repeat_interleave(NO, 0)
            dl = D - torch.logsumexp(L0 + D, -1, keepdim=True)
            skl[:, k, :, t] = (.5 * (L0.exp() * torch.expm1(dl) * dl).sum(-1)).reshape(b, NO, NV)
    return logd, chi, skl, lupd


# ═══════════════════════════════════════════════════════════════════
# Counterfactual observations (control): the same state and flip difference, step t under each of the 32 patterns
# ═══════════════════════════════════════════════════════════════════

NPAT = 1 << NO
CF_X = {"trained": (60, 120, 240, 360, 560, 880, 1680, 4200, 12000, 39996), "echo": (240, 880, 4200, 39996)}
CF_KG = 8                                                                       # patterns per GPU pass


def cf_raws(run):
    """the stored checkpoints nearest (log scale) to CF_X[run]"""
    eps = np.array(ti.checkpoints(run))
    x = gap.train_x(eps)
    return [int(eps[np.argmin(np.abs(np.log(x) - np.log(w)))]) for w in CF_X[run]]


@torch.no_grad()
def cf_net(run, raw, dev, rws=None, chunk=100):
    """for every unit (probe, step t >= 2) and every observation pattern k (32), the one-step growth g^(k) of the
    flip's state difference and the change of its belief impact (d_H, the mean of the 2 variables) dchi^(k), had step
    t received pattern k instead of its own observation: the same intact state before t and the same flip difference
    Delta_{t-1} (difference form), the mean over the flips before t (as unit_values) -> Gk, Xk (n, T, 32) float64
    (NaN at step 1). The flips' own trajectories keep their actual observations"""
    rws = rows() if rws is None else rws
    p, table = setup(run, raw, dev)
    envt, code = codes(rws, dev)
    Whh, Wro = p["LSTM.weight_hh_l0"], p["classifier_readout.weight"]
    H = Whh.shape[1]
    n = len(rws)
    Gk, Xk = np.full((n, T, NPAT), np.nan), np.full((n, T, NPAT), np.nan)
    bits = torch.as_tensor(1 << np.arange(NO), device=dev)
    for b0 in range(0, n, chunk):
        sl = slice(b0, min(n, b0 + chunk))
        b = sl.stop - sl.start
        A, U, C, lg = replay(p, table, envt[sl], code[sl])
        eb = envt[sl].repeat_interleave(NO)
        gs = torch.zeros(b, T, NPAT, device=dev, dtype=torch.float64)
        xs = torch.zeros(b, T, NPAT, device=dev, dtype=torch.float64)
        cnt = torch.zeros(b, T, device=dev, dtype=torch.float64)
        for s in FLIP_S:
            cd = code[sl, s - 1].repeat_interleave(NO)
            dA = table[eb, cd ^ bits.repeat(b)] - table[eb, cd]
            dh = torch.zeros(b * NO, H, device=dev, dtype=Whh.dtype)
            dc = torch.zeros(b * NO, H, device=dev, dtype=Whh.dtype)
            D = torch.zeros(b * NO, NV, R, device=dev, dtype=torch.float64)
            for t in range(s - 1, T):
                if t > s - 1:
                    dA = dh @ Whh.T
                    rec = (U[:, t] @ Whh.T).repeat_interleave(NO, 0)              # intact recurrent drive
                    cp = C[:, t].repeat_interleave(NO, 0)
                    ln0 = torch.log(torch.cat([dh, dc], -1).double().norm(dim=-1))
                    r0 = (D.amax(-1) - D.amin(-1)).mean(-1)
                    di, df, dg, do = dA.chunk(4, -1)
                    for k0 in range(0, NPAT, CF_KG):
                        ks = torch.arange(k0, k0 + CF_KG, device=dev)
                        a = table[eb[None], ks[:, None].expand(-1, b * NO)] + rec[None]    # (KG, b NO, 4H)
                        ai, af, ag, ao = a.chunk(4, -1)
                        i, f, g, o = ai.sigmoid(), af.sigmoid(), ag.tanh(), ao.sigmoid()
                        Di, Df, Dg, Do = dsig(ai, di), dsig(af, df), dtanh(ag, dg), dsig(ao, do)
                        cn = f * cp + i * g
                        dck = Df * cp + (f + Df) * dc + Di * (g + Dg) + i * Dg
                        dhk = Do * torch.tanh(cn + dck) + o * dtanh(cn, dck)
                        gk = torch.log(torch.cat([dhk, dck], -1).double().norm(dim=-1)) - ln0
                        Dk = D[None] + (dhk @ Wro.T).double().reshape(CF_KG, -1, NV, R)
                        xk = (Dk.amax(-1) - Dk.amin(-1)).mean(-1) - r0
                        gs[:, t, k0:k0 + CF_KG] += gk.reshape(CF_KG, b, NO).sum(-1).T
                        xs[:, t, k0:k0 + CF_KG] += xk.reshape(CF_KG, b, NO).sum(-1).T
                    cnt[:, t] += NO
                a = A[:, t].repeat_interleave(NO, 0)
                i, f, g, o = _gates(a)
                di, df, dg, do = dA.chunk(4, -1)
                ai, af, ag, ao = a.chunk(4, -1)
                Di, Df, Dg, Do = dsig(ai, di), dsig(af, df), dtanh(ag, dg), dsig(ao, do)
                cp = C[:, t].repeat_interleave(NO, 0)
                cn = C[:, t + 1].repeat_interleave(NO, 0)
                dc = Df * cp + (f + Df) * dc + Di * (g + Dg) + i * Dg
                dh = Do * torch.tanh(cn + dc) + o * dtanh(cn, dc)
                D += (dh @ Wro.T).double().reshape(-1, NV, R)
        with np.errstate(invalid="ignore"):
            c_ = cnt.cpu().numpy()[..., None]
            Gk[sl] = np.where(c_ > 0, gs.cpu().numpy() / np.maximum(c_, 1), np.nan)
            Xk[sl] = np.where(c_ > 0, xs.cpu().numpy() / np.maximum(c_, 1), np.nan)
    return Gk, Xk


def _pat_bits():
    return ((np.arange(NPAT)[:, None] >> np.arange(NO)) & 1).astype(np.float64)   # (32, 5)


def cf_observers(rws=None):
    """the Bayes side of the counterfactual control (float64): ent_k (n, T, 32) = the entanglement step t would have
    with pattern k (the locked definition, probes.npz 'ent' for the actual pattern: checked), w (n, 32) = p(pattern k |
    the probe's cell) (the distribution the actual observations are drawn from), and under each pattern with the
    same history: Bayes's d_H change Xk_bayes, ADF's state growth Gk_adf and d_H change Xk_adf (n, T, 32), the mean over
    the flips before t (NaN at step 1)"""
    import supplementary_plotter2 as sp
    rws = rows() if rws is None else rws
    P = ti.probes()
    L, _ = tc.tables()
    obs, env = P["obs"][rws], P["env"][rws]
    Le = L[env]                                                                 # (n, 5, R, R)
    n = len(rws)
    PB = _pat_bits()
    llk = np.einsum("ki,nijl->nkjl", PB, np.log(Le)) + np.einsum("ki,nijl->nkjl", 1 - PB, np.log1p(-Le))  # (n,32,R,R)
    rr = P["r"][rws]
    w = np.exp(llk[np.arange(n), :, rr[:, 0], rr[:, 1]])                        # (n, 32)
    ll = tc.step_loglik(obs, Le)
    u = np.cumsum(ll, 1)
    u0 = np.concatenate([np.zeros((n, 1, R, R)), u[:, :-1]], 1)                 # joint log belief before t
    Lb = np.stack([Le.mean(3), Le.mean(2)], 2)                                  # (n, 5, 2, R)
    lpk = np.einsum("ki,nivr->nkvr", PB, np.log(Lb)) + np.einsum("ki,nivr->nkvr", 1 - PB, np.log1p(-Lb))
    npx = np.exp(lpk - logsumexp(lpk, -1, keepdims=True))                       # (n, 32, 2, R)
    ent_k = np.zeros((n, T, NPAT))
    for t in range(T):
        jt = u0[:, t, None] + llk                                               # (n, 32, R, R)
        jt = jt - logsumexp(jt.reshape(n, NPAT, -1), -1)[..., None, None]
        pt = np.exp(jt)
        jb = np.maximum(np.stack([pt.sum(3), pt.sum(2)], 2), 1e-99)             # (n, 32, 2, R)
        if t == 0:
            px = jb
        else:
            pp = np.exp(u0[:, t] - logsumexp(u0[:, t].reshape(n, -1), -1)[:, None, None])
            jp = np.maximum(np.stack([pp.sum(2), pp.sum(1)], 1), 1e-99)        # (n, 2, R)
            ratio_ = jb / jp[:, None]
            px = ratio_ / ratio_.sum(-1, keepdims=True)
        ent_k[:, t] = sp.step_dkl(npx, px).mean(-1)
    logit = np.log(Le) - np.log1p(-Le)
    la = tc.adf_run(ll)
    out = {k: np.zeros((n, T, NPAT)) for k in ("Xk_bayes", "Gk_adf", "Xk_adf")}
    cnt = np.zeros((n, T))

    def mlog(x):
        return _nrm(tc.marg_log(x))

    def adf_step(la_prev, llt):
        j = la_prev[..., 0, :, None] + la_prev[..., 1, None, :] + llt
        j = j - logsumexp(j.reshape(j.shape[:-2] + (-1,)), -1)[..., None, None]
        return _nrm(tc.marg_log(j))
    for s in FLIP_S:
        sign = 1. - 2. * obs[:, s - 1].astype(np.float64)
        for o in range(NO):
            d = sign[:, o, None, None] * logit[:, o]
            ll2 = ll[:, s - 1:].copy()
            ll2[:, 0] += d
            la2 = tc.adf_run(ll2, None if s == 1 else np.exp(la[:, s - 2]))     # flipped ADF from step s
            for t in range(s, T):
                ui, uf = u0[:, t, None] + llk, u0[:, t, None] + d[:, None] + llk
                chi_prev = tc.osc(mlog(u0[:, t] + d) - mlog(u0[:, t])).mean(-1)
                out["Xk_bayes"][:, t] += tc.osc(mlog(uf) - mlog(ui)).mean(-1) - chi_prev[:, None]
                ai, af = la[:, t - 1], la2[:, t - s]                            # ADF marginals before t
                ni, nf = adf_step(ai[:, None], llk), adf_step(af[:, None], llk)
                dl0 = af - ai
                dl0 = dl0 - dl0.mean(-1, keepdims=True)
                dl1 = nf - ni
                dl1 = dl1 - dl1.mean(-1, keepdims=True)
                out["Gk_adf"][:, t] += (.5 * np.log((dl1 ** 2).sum((-1, -2)))
                                        - .5 * np.log((dl0 ** 2).sum((-1, -2)))[:, None])
                out["Xk_adf"][:, t] += tc.osc(nf - ni).mean(-1) - tc.osc(af - ai).mean(-1)[:, None]
                cnt[:, t] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in out:
            out[k] = np.where(cnt[..., None] > 0, out[k] / cnt[..., None], np.nan)
    code = (obs * (1 << np.arange(NO))).sum(-1)
    return dict(rows=rws, ent_k=ent_k, w=w, code=code, **out)


LAGG = ((1, 1), (2, 4), (5, 20), (21, T - 1))           # the disjoint lag groups of the update counterfactual's sums
LAGSET = {"1": (0,), "1-4": (0, 1), "5-20": (2,), "21-29": (3,), "all": (0, 1, 2, 3)}


def _lag_group(lag):
    return next(i for i, (a, b) in enumerate(LAGG) if a <= lag <= b)


@torch.no_grad()
def cf_net_upd(run, raw, dev, rws=None, chunk=100):
    """the counterfactual of the update measure for a network checkpoint: for every unit (probe, step t) and pattern k,
    the growth log s_t^(k) - log s_{t-1} of the flip's update-difference size (flips(upd=True)'s measure) had step t
    received pattern k, from the same intact state and flip difference Delta_{t-1} (difference form, as cf_net) ->
    Uk (n, T, 32, 4) float64 = its sums over the flips before t in each disjoint lag group (LAGG), cnt (T, 4) = their
    counts (the same for every probe); the flips' trajectories keep their actual observations"""
    rws = rows() if rws is None else rws
    p, table = setup(run, raw, dev)
    envt, code = codes(rws, dev)
    Whh, Wro = p["LSTM.weight_hh_l0"], p["classifier_readout.weight"]
    H = Whh.shape[1]
    n = len(rws)
    Uk = np.zeros((n, T, NPAT, len(LAGG)))
    cnt = np.zeros((T, len(LAGG)))
    for s in FLIP_S:
        for t in range(s, T):
            cnt[t, _lag_group(t - (s - 1))] += NO
    bits = torch.as_tensor(1 << np.arange(NO), device=dev)
    for b0 in range(0, n, chunk):
        sl = slice(b0, min(n, b0 + chunk))
        b = sl.stop - sl.start
        A, U, C, lg = replay(p, table, envt[sl], code[sl])
        eb = envt[sl].repeat_interleave(NO)
        us = torch.zeros(b, T, NPAT, len(LAGG), device=dev, dtype=torch.float64)
        for s in FLIP_S:
            cd = code[sl, s - 1].repeat_interleave(NO)
            dA = table[eb, cd ^ bits.repeat(b)] - table[eb, cd]
            dh = torch.zeros(b * NO, H, device=dev, dtype=Whh.dtype)
            dc = torch.zeros(b * NO, H, device=dev, dtype=Whh.dtype)
            for t in range(s - 1, T):
                if t > s - 1:
                    dA = dh @ Whh.T
                    gi = _lag_group(t - (s - 1))
                    rec = (U[:, t] @ Whh.T).repeat_interleave(NO, 0)
                    cp = C[:, t].repeat_interleave(NO, 0)
                    inc0 = (dh @ Wro.T).double().reshape(-1, NV, R)
                    r0 = torch.log((inc0.amax(-1) - inc0.amin(-1)).mean(-1))       # log s_{t-1}
                    di, df, dg, do = dA.chunk(4, -1)
                    for k0 in range(0, NPAT, CF_KG):
                        ks = torch.arange(k0, k0 + CF_KG, device=dev)
                        a = table[eb[None], ks[:, None].expand(-1, b * NO)] + rec[None]
                        ai, af, ag, ao = a.chunk(4, -1)
                        i, f, g, o = ai.sigmoid(), af.sigmoid(), ag.tanh(), ao.sigmoid()
                        Di, Df, Dg, Do = dsig(ai, di), dsig(af, df), dtanh(ag, dg), dsig(ao, do)
                        cn = f * cp + i * g
                        dck = Df * cp + (f + Df) * dc + Di * (g + Dg) + i * Dg
                        dhk = Do * torch.tanh(cn + dck) + o * dtanh(cn, dck)
                        inck = (dhk @ Wro.T).double().reshape(CF_KG, -1, NV, R)
                        uk = torch.log((inck.amax(-1) - inck.amin(-1)).mean(-1)) - r0
                        us[:, t, k0:k0 + CF_KG, gi] += uk.reshape(CF_KG, b, NO).sum(-1).T
                a = A[:, t].repeat_interleave(NO, 0)
                i, f, g, o = _gates(a)
                di, df, dg, do = dA.chunk(4, -1)
                ai, af, ag, ao = a.chunk(4, -1)
                Di, Df, Dg, Do = dsig(ai, di), dsig(af, df), dtanh(ag, dg), dsig(ao, do)
                cp = C[:, t].repeat_interleave(NO, 0)
                cn = C[:, t + 1].repeat_interleave(NO, 0)
                dc = Df * cp + (f + Df) * dc + Di * (g + Dg) + i * Dg
                dh = Do * torch.tanh(cn + dc) + o * dtanh(cn, dc)
        Uk[sl] = us.cpu().numpy()
    return Uk, cnt


def cf_observers_upd(rws=None):
    """the counterfactual of the update measure for the exact observers (float64, CPU): for every unit and pattern k the
    growth log s_t^(k) - log s_{t-1} of Bayes's and ADF's update-difference size (observers_upd) had step t received
    pattern k, at the same beliefs before t (intact and flipped) -> Uk_bayes, Uk_adf (n, T, 32, 4) sums over the flips
    before t per disjoint lag group, cnt (T, 4)"""
    rws = rows() if rws is None else rws
    P = ti.probes()
    L, _ = tc.tables()
    obs, env = P["obs"][rws], P["env"][rws]
    Le = L[env]
    n = len(rws)
    PB = _pat_bits()
    llk = np.einsum("ki,nijl->nkjl", PB, np.log(Le)) + np.einsum("ki,nijl->nkjl", 1 - PB, np.log1p(-Le))
    ll = tc.step_loglik(obs, Le)
    u0 = np.concatenate([np.zeros((n, 1, R, R)), np.cumsum(ll, 1)[:, :-1]], 1)  # joint log belief before t
    logit = np.log(Le) - np.log1p(-Le)
    la = tc.adf_run(ll)
    Ob = observers_upd_cached()
    out = {k: np.zeros((n, T, NPAT, len(LAGG))) for k in ("Uk_bayes", "Uk_adf")}
    cnt = np.zeros((T, len(LAGG)))

    def mlog(x):
        return _nrm(tc.marg_log(x))

    def adf_step(la_prev, llt):
        j = la_prev[..., 0, :, None] + la_prev[..., 1, None, :] + llt
        j = j - logsumexp(j.reshape(j.shape[:-2] + (-1,)), -1)[..., None, None]
        return _nrm(tc.marg_log(j))
    for k_s, s in enumerate(FLIP_S):
        sign = 1. - 2. * obs[:, s - 1].astype(np.float64)
        for o in range(NO):
            d = sign[:, o, None, None] * logit[:, o]
            ll2 = ll[:, s - 1:].copy()
            ll2[:, 0] += d
            la2 = tc.adf_run(ll2, None if s == 1 else np.exp(la[:, s - 2]))
            for t in range(s, T):
                gi = _lag_group(t - (s - 1))
                b0i, b0f = mlog(u0[:, t]), mlog(u0[:, t] + d)                  # Bayes's marginals before t
                bi = mlog(u0[:, t, None] + llk)
                bf = mlog(u0[:, t, None] + d[:, None] + llk)
                sz = tc.osc((bf - b0f[:, None]) - (bi - b0i[:, None])).mean(-1)          # (n, 32)
                out["Uk_bayes"][:, t, :, gi] += np.log(sz) - Ob["lupd_bayes"][:, k_s, o, t - 1][:, None]
                ai, af = la[:, t - 1], la2[:, t - s]                            # ADF's marginals before t
                ni, nf = adf_step(ai[:, None], llk), adf_step(af[:, None], llk)
                sza = tc.osc((nf - af[:, None]) - (ni - ai[:, None])).mean(-1)
                out["Uk_adf"][:, t, :, gi] += np.log(sza) - Ob["lupd_adf"][:, k_s, o, t - 1][:, None]
                cnt[t, gi] += 1
    return dict(rows=rws, cnt=cnt, **out)


def cf_upd_file(run, raw):
    return os.path.join(OUT, "cf", f"{run}_{int(raw):05d}_upd.npz")


CF_LR_X = {"trained": (360, 560, 880, 1200, 1680, 39996), "echo": (880, 39996)}
CF_LR_PER_ENV = 4                                # the log rho counterfactual's probes: the first 4 of each environment


def cf_lrho_rows():
    """the log rho counterfactual's probes: the first CF_LR_PER_ENV of rows()'s 20 per environment (200 probes)"""
    rws = rows()
    env = ti.probes()["env"][rws]
    return np.sort(np.concatenate([rws[env == e][:CF_LR_PER_ENV] for e in range(gap.N_ENV)]))


def cf_lrho_file(run, raw):
    return os.path.join(OUT, "cf", f"{run}_{int(raw):05d}_lrho.npz")


@torch.no_grad()
def cf_lrho(run, raw, dev, rws=None, chunk=10, kb=10):
    """the counterfactual of log rho(J_t): for every unit (probe, step t) of rws and pattern k, the log spectral radius
    of the step Jacobian had step t received pattern k at the same intact state (u_{t-1}, c_{t-1}) (arnoldi_rho, as
    ext_one) -> L (n, T, 32)"""
    rws = cf_lrho_rows() if rws is None else rws
    p, table = setup(run, raw, dev)
    envt, code = codes(rws, dev)
    Whh = p["LSTM.weight_hh_l0"]
    H = Whh.shape[1]
    n = len(rws)
    L = np.zeros((n, T, NPAT))
    for b0 in range(0, n, chunk):
        sl = slice(b0, min(n, b0 + chunk))
        b = sl.stop - sl.start
        A, U, C, lg = replay(p, table, envt[sl], code[sl])
        rec = U[:, :-1] @ Whh.T                                                 # (b, T, 4H): the drive of the state
        cp = C[:, :-1]
        for k0 in range(0, NPAT, kb):
            ks = torch.arange(k0, min(NPAT, k0 + kb), device=dev)
            a = table[envt[sl]][:, None][:, :, ks] + rec[:, :, None]            # (b, T, K, 4H)
            i, f, g, o = _gates(a)
            cpk = cp[:, :, None].expand(-1, -1, len(ks), -1)
            tcn = torch.tanh(f * cpk + i * g)
            N = b * T * len(ks)
            gall = tuple(x.reshape(N, H) for x in (i, f, g, o, tcn))
            call = cpk.reshape(N, H)

            def mv(v, ids):
                it = torch.as_tensor(ids, device=dev)
                dh, dc = tc.tangent(v[:, None, :H], v[:, None, H:], call[it], tuple(x[it] for x in gall), Whh)
                return torch.cat([dh, dc], -1)[:, 0]
            th = arnoldi_rho(mv, N, 2 * H, dev, LR_SEED + 1000 * b0 + k0)[0]
            L[sl, :, k0:k0 + len(ks)] = np.log(np.abs(th)).reshape(b, T, len(ks))
    return L


def cf_lrho_harvest(reverse=False, dev=None):
    """the log rho counterfactual at CF_LR_X of both networks -> cf/<run>_EEEEE_lrho.npz (resumable; reverse: the job
    list from the other end, for a second process; a job being computed holds <file>.lock)"""
    dev = _gpu_cap(ti._device(dev))
    eps = {run: np.array(ti.checkpoints(run)) for run in RUNS}
    jobs = [(run, int(eps[run][np.argmin(np.abs(np.log(gap.train_x(eps[run])) - np.log(w)))]))
            for run in RUNS for w in CF_LR_X[run]]
    rws = cf_lrho_rows()
    for run, raw in (jobs[::-1] if reverse else jobs):
        f = cf_lrho_file(run, raw)
        if os.path.exists(f):
            continue
        try:
            fd = os.open(f + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.close(fd)
        t0 = time.time()
        Lk = cf_lrho(run, raw, dev, rws)
        tmp = f[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, L=Lk, rows=rws, raw=raw, x=gap.train_x(raw))
        os.replace(tmp, f)
        os.remove(f + ".lock")
        print(f"cf lrho {run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s", flush=True)


def cf_observers_lrho(rws=None):
    """ADF's log rho under each pattern k at the same marginals before t: log(1 + rho_1) of the one-step joint (= the
    log of its exact step Jacobian's largest |eigenvalue|, T3; checked by observers_upd's lrho_adf_eig) -> Lk_adf
    (n, T, 32); joint Bayes and naive: exactly 0 under every pattern (T1)"""
    rws = rows() if rws is None else rws
    P = ti.probes()
    L, _ = tc.tables()
    obs, env = P["obs"][rws], P["env"][rws]
    Le = L[env]
    n = len(rws)
    PB = _pat_bits()
    llk = np.einsum("ki,nijl->nkjl", PB, np.log(Le)) + np.einsum("ki,nijl->nkjl", 1 - PB, np.log1p(-Le))
    la = tc.adf_run(tc.step_loglik(obs, Le))
    la_prev = np.concatenate([np.full((n, 1, 2, R), -np.log(R)), la[:, :-1]], 1)
    Lk = np.zeros((n, T, NPAT))
    for t in range(T):
        _, rho = tc.adf_jacobians(llk, np.broadcast_to(la_prev[:, t, None], (n, NPAT, 2, R)))
        Lk[:, t] = np.log1p(rho[..., 0])
    return dict(rows=rws, Lk_adf=Lk)


def cf_observers_lrho_cached():
    path = os.path.join(OUT, "cf", "observers_lrho.npz")
    if not os.path.exists(path):
        tmp = path[:-4] + ".tmp.npz"
        np.savez(tmp, **cf_observers_lrho())
        os.replace(tmp, path)
    return dict(np.load(path))


def cf_adf_lrho_stats(gname="main"):
    """ADF's log rho counterfactual: {part: (value, 5%, 95%)} of the fifth 5 - fifth 1 contrast (all probes), own by
    fifth, and the largest |actual (pattern pass) - observers_upd's lrho_adf_eig|"""
    Z = cf_observers_lrho_cached()
    O = dict(np.load(os.path.join(OUT, "cf", "observers.npz")))
    Lk = Z["Lk_adf"]
    act = np.take_along_axis(Lk, O["code"][..., None], -1)[..., 0]
    exp_ = (Lk * O["w"][:, None, :]).sum(-1)
    cov = covariates()
    q = groupings(cov)[gname]
    out, fv = {}, None
    for part, v in (("actual", act), ("expected", exp_), ("own", act - exp_)):
        S, C = group_sums({"lrho": v}, {gname: q}, cov["env"])[("lrho", gname)]
        (cv, cb), _ = contrast(S, C)
        out[part] = (float(cv), *(float(x_) for x_ in iv(cb)))
        if part == "own":
            fv = ratio(S, C)[0]
    chk = float(np.abs(act - observers_upd_cached()["lrho_adf_eig"]).max())
    return out, fv, chk


def cf_lrho_parts(run, raw):
    """the log rho counterfactual per unit (n_sub, T): actual (the stored ext_ value of the unit), the value at its own
    pattern in the counterfactual pass (actual_cf), expected = sum_k w_k log rho^(k) (p(pattern | the probe's cell), as
    cf_stats), own = actual_cf - expected; and the index of the subsample's probes in rows()"""
    Z = np.load(cf_lrho_file(run, raw))
    sub = np.searchsorted(rows(), Z["rows"])
    O = dict(np.load(os.path.join(OUT, "cf", "observers.npz")))
    Lk = Z["L"]
    act_cf = np.take_along_axis(Lk, O["code"][sub][..., None], -1)[..., 0]
    exp_ = (Lk * O["w"][sub][:, None, :]).sum(-1)
    act = np.load(ext_file(run, raw))["lrho"][sub] if os.path.exists(ext_file(run, raw)) else act_cf
    return {"actual": act, "actual_cf": act_cf, "expected": exp_, "own": act_cf - exp_}, sub


def cf_lrho_stats(run, raw, gname="main"):
    """the fifth 5 - fifth 1 contrast of each part of cf_lrho_parts on the subsample (main fifths of all probes, so the
    same labels as everywhere): {part: (value, 5%, 95%)} and the per-fifth values of 'own'"""
    parts, sub = cf_lrho_parts(run, raw)
    cov = covariates()
    q = groupings(cov)[gname][sub]
    out, fv = {}, None
    for part, v in parts.items():
        S, C = group_sums({"lrho": v}, {gname: q}, cov["env"][sub])[("lrho", gname)]
        (cv, cb), _ = contrast(S, C)
        out[part] = (float(cv), *(float(x_) for x_ in iv(cb)))
        if part == "own":
            fv = ratio(S, C)[0]
    return out, fv


def cf_upd_harvest(dev=None):
    """the update measure's counterfactual at cf_raws() of both networks -> cf/<run>_EEEEE_upd.npz (Uk float32, cnt) and
    the observers' -> cf/observers_upd.npz"""
    path = os.path.join(OUT, "cf", "observers_upd.npz")
    if not os.path.exists(path):
        t0 = time.time()
        tmp = path[:-4] + ".tmp.npz"
        np.savez(tmp, **cf_observers_upd())
        os.replace(tmp, path)
        print(f"cf observers (update): {time.time() - t0:.0f} s", flush=True)
    dev = _gpu_cap(ti._device(dev))
    for run in RUNS:
        for raw in cf_raws(run):
            f = cf_upd_file(run, raw)
            if os.path.exists(f):
                continue
            t0 = time.time()
            Uk, cnt = cf_net_upd(run, raw, dev)
            tmp = f[:-4] + ".tmp.npz"
            np.savez_compressed(tmp, Uk=Uk.astype(np.float32), cnt=cnt, raw=raw, x=gap.train_x(raw))
            os.replace(tmp, f)
            print(f"cf upd {run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s", flush=True)


def cf_stats(run=None, raw=None, agent=None):
    """the counterfactual decomposition of the fifth 5 - fifth 1 contrast (main grouping) of g and of the d_H change
    for a network checkpoint (run, raw) or an exact observer (agent 'bayes' | 'adf'): actual = the unit's value with
    its own observation (from the counterfactual pass: = unit_values'), expected = the mean over the 32 patterns
    weighted by p(pattern | the probe's cell) (the same state and flip history, any observation), own = actual -
    expected (what the step's actual observation adds). actual = expected + own, so the contrasts add exactly.
    Returns {(measure, part): (value, 5%, 95%)} and the per-fifth values of 'own'"""
    O = dict(np.load(os.path.join(OUT, "cf", "observers.npz")))
    cov = covariates()
    q = groupings(cov)["main"]
    if agent is None:
        Z = np.load(cf_file(run, raw))
        arrs = {"g": Z["Gk"].astype(np.float64), "dchi": Z["Xk"].astype(np.float64)}
    elif agent == "bayes":
        arrs = {"dchi": O["Xk_bayes"]}
    else:
        arrs = {"g": O["Gk_adf"], "dchi": O["Xk_adf"]}
    out, fifth_vals = {}, {}
    for m, K in arrs.items():
        act = np.take_along_axis(K, O["code"][..., None], -1)[..., 0]
        exp_ = (K * O["w"][:, None, :]).sum(-1)
        parts = {"actual": act, "expected": exp_, "own": act - exp_}
        for part, v in parts.items():
            S, C = group_sums({m: v}, {"main": q}, cov["env"])[(m, "main")]
            (cv, cb), _ = contrast(S, C)
            out[(m, part)] = (float(cv), *(float(x_) for x_ in iv(cb)))
            if part == "own":
                fifth_vals[m] = ratio(S, C)[0]
    return out, fifth_vals


def cf_upd_parts(run=None, raw=None, agent=None, lags="all"):
    """the update measure's counterfactual per unit (n, T): actual (the unit's value with its own observation, over the
    flips whose lag lies in LAGSET[lags]), expected (the pattern-weighted mean at the same state and flip history) and
    own = actual - expected; network checkpoint (run, raw) or exact observer (agent 'bayes' | 'adf')"""
    O = dict(np.load(os.path.join(OUT, "cf", "observers.npz")))
    if agent is None:
        Z = np.load(cf_upd_file(run, raw))
        Uk = Z["Uk"].astype(np.float64)
    else:
        Z = np.load(os.path.join(OUT, "cf", "observers_upd.npz"))
        Uk = Z[f"Uk_{agent}"]
    sel = list(LAGSET[lags])
    num, den = Uk[..., sel].sum(-1), Z["cnt"][:, sel].sum(-1)                  # (n, T, 32), (T,)
    with np.errstate(invalid="ignore", divide="ignore"):
        K = np.where(den[None, :, None] > 0, num / den[None, :, None], np.nan)
    act = np.take_along_axis(K, O["code"][..., None], -1)[..., 0]
    exp_ = (K * O["w"][:, None, :]).sum(-1)
    return {"actual": act, "expected": exp_, "own": act - exp_}


def cf_upd_stats(run=None, raw=None, agent=None, lags="all", gname="main"):
    """the fifth 5 - fifth 1 contrast of each part of cf_upd_parts (actual = expected + own; grouping gname):
    {part: (value, 5%, 95%)} and the per-fifth values of 'own'"""
    cov = covariates()
    q = groupings(cov)[gname]
    out = {}
    fv = None
    for part, v in cf_upd_parts(run, raw, agent, lags).items():
        S, C = group_sums({"upd": v}, {gname: q}, cov["env"])[("upd", gname)]
        (cv, cb), _ = contrast(S, C)
        out[part] = (float(cv), *(float(x_) for x_ in iv(cb)))
        if part == "own":
            fv = ratio(S, C)[0]
    return out, fv


def cf_check():
    """CPU: the counterfactual pass at the actual observation vs the main values (networks: unit g and d_H change at
    every cf checkpoint; observers: Bayes's d_H change, ADF's g and d_H change), its entanglement vs probes.npz 'ent',
    the pattern weights summing to 1 -> merged into chaos_ent/checks.json"""
    O = dict(np.load(os.path.join(OUT, "cf", "observers.npz")))
    cov = covariates()
    ea = np.take_along_axis(O["ent_k"], O["code"][..., None], -1)[..., 0]
    out = {"cf_ent_actual_vs_probes_maxabs": float(np.abs(ea - cov["ent"]).max()),
           "cf_weights_sum_min_max": [float(O["w"].sum(1).min()), float(O["w"].sum(1).max())]}
    Ob = observers_cached()

    def act(K):
        return np.take_along_axis(K, O["code"][..., None], -1)[..., 0]
    ub = unit_values(None, None, Ob["chi_bayes"], None)
    ua = unit_values(None, Ob["logd_adf"], Ob["chi_adf"], None)
    out["cf_bayes_dchi_vs_main_maxabs"] = float(np.nanmax(np.abs(act(O["Xk_bayes"]) - ub["dchi"])))
    out["cf_adf_g_dchi_vs_main_maxabs"] = [float(np.nanmax(np.abs(act(O["Gk_adf"]) - ua["g"]))),
                                           float(np.nanmax(np.abs(act(O["Xk_adf"]) - ua["dchi"])))]
    worst = [0., 0.]
    for run in RUNS:
        for raw in cf_raws(run):
            Z, C = np.load(ck_file(run, raw)), np.load(cf_file(run, raw))
            u = unit_values(None, Z["logd"], Z["chi"], None)
            worst[0] = max(worst[0], float(np.nanmax(np.abs(act(C["Gk"].astype(np.float64)) - u["g"]))))
            worst[1] = max(worst[1], float(np.nanmax(np.abs(act(C["Xk"].astype(np.float64)) - u["dchi"]))))
    out["cf_networks_g_dchi_vs_main_maxabs"] = worst
    # the update measure: the counterfactual at the actual observation vs the main unit values, every lag set
    O2 = observers_upd_cached()
    w_ob = 0.
    for a in ("bayes", "adf"):
        u = unit_values(lupd=O2[f"lupd_{a}"])
        for lags in LAGSET:
            key = "upd" if lags == "all" else f"upd:{lags}"
            w_ob = max(w_ob, float(np.nanmax(np.abs(cf_upd_parts(agent=a, lags=lags)["actual"] - u[key]))))
    out["cf_upd_observers_vs_main_maxabs"] = w_ob
    w_net = 0.
    for run in RUNS:
        for raw in cf_raws(run):
            if not (os.path.exists(cf_upd_file(run, raw)) and os.path.exists(ext_file(run, raw))):
                continue
            u = unit_values(lupd=np.load(ext_file(run, raw))["lupd"])
            for lags in LAGSET:
                key = "upd" if lags == "all" else f"upd:{lags}"
                w_net = max(w_net, float(np.nanmax(np.abs(cf_upd_parts(run, raw, lags=lags)["actual"] - u[key]))))
    out["cf_upd_networks_vs_main_maxabs"] = w_net
    out["adf_lrho_eig_vs_log1p_rho1_maxabs"] = float(np.abs(O2["lrho_adf_eig"] - O2["lrho_adf_cc"]).max())
    path = os.path.join(OUT, "checks.json")
    allc = json.load(open(path)) if os.path.exists(path) else {}
    allc.update(out)
    with open(path, "w") as fh:
        json.dump(allc, fh, indent=1)
    for k_, v in out.items():
        print(f"  {k_}: {v}", flush=True)
    return out


def cf_file(run, raw):
    return os.path.join(OUT, "cf", f"{run}_{int(raw):05d}.npz")


def cf_harvest(dev=None):
    """the counterfactual control at cf_raws() of both runs -> chaos_ent/cf/<run>_EEEEE.npz (Gk, Xk) and
    cf/observers.npz"""
    dev = _gpu_cap(ti._device(dev))
    os.makedirs(os.path.join(OUT, "cf"), exist_ok=True)
    path = os.path.join(OUT, "cf", "observers.npz")
    if not os.path.exists(path):
        t0 = time.time()
        np.savez(path, **cf_observers())
        print(f"cf observers: {time.time() - t0:.0f} s", flush=True)
    for run in RUNS:
        for raw in cf_raws(run):
            f = cf_file(run, raw)
            if os.path.exists(f):
                continue
            t0 = time.time()
            Gk, Xk = cf_net(run, raw, dev)
            tmp = f[:-4] + ".tmp.npz"
            np.savez_compressed(tmp, Gk=Gk.astype(np.float32), Xk=Xk.astype(np.float32), raw=raw,
                                x=gap.train_x(raw))
            os.replace(tmp, f)
            print(f"cf {run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s", flush=True)


# ═══════════════════════════════════════════════════════════════════
# Harvest: every stored checkpoint of the Trained RNN and the ESN (resumable; STOP file chaos_ent/STOP)
# ═══════════════════════════════════════════════════════════════════

def ck_file(run, raw):
    return os.path.join(OUT, run, f"ck_{int(raw):05d}.npz")


def harvest(run, dev=None, reverse=False):
    """one file per checkpoint -> chaos_ent/<run>/ck_EEEEE.npz (rows, raw, x and one()'s arrays, float32); every third
    checkpoint and the last first, then the rest (reverse: the whole order reversed, for a second process on the same
    run; the two meet in the middle, a checkpoint written twice is replaced atomically by an identical one); stops
    before the next checkpoint when chaos_ent/STOP exists"""
    dev = _gpu_cap(ti._device(dev))
    os.makedirs(os.path.join(OUT, run), exist_ok=True)
    eps = ti.checkpoints(run)
    first = [e for i, e in enumerate(eps) if i % 3 == 0 or i == len(eps) - 1]
    rws = rows()
    print(f"{run}: {len(eps)} checkpoints", flush=True)
    order = first + [e for e in eps if e not in first]
    for raw in order[::-1] if reverse else order:
        if os.path.exists(os.path.join(OUT, "STOP")):
            print("STOP file: stopping", flush=True)
            return
        f = ck_file(run, raw)
        if os.path.exists(f):
            continue
        t0 = time.time()
        chunk = 250
        while True:
            try:
                o = one(run, raw, dev, rws, chunk=chunk)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if chunk > 50:
                    chunk //= 2
                else:
                    print("GPU full: waiting 60 s", flush=True)
                    time.sleep(60)
        tmp = f[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, rows=rws, raw=raw, x=gap.train_x(raw), **o)
        os.replace(tmp, f)
        g = np.diff(o["logd"].astype(np.float64), axis=-1)
        print(f"{run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.1f} s; smax {o['smax'].mean():+.3f} "
              f"(res max {o['smax_res'].max():.1e}, steps max {o['smax_steps'].max()}); mean g {np.nanmean(g):+.4f}",
              flush=True)


# ═══════════════════════════════════════════════════════════════════
# Statistics: per probe-step unit -> entanglement fifths -> environment sums -> the shared environment resamples
# ═══════════════════════════════════════════════════════════════════

LAGS = {"1": (1, 1), "1-4": (1, 4), "5-20": (5, 20), "21-29": (21, T - 1)}  # lag = t - s (flip at s, unit at t)
MEAS = ("lrho", "lrho_pos", "smax", "smax_pos",
        "g", "g_pos", "g:1", "g:1_pos", "g:1-4", "g:1-4_pos", "g:5-20", "g:5-20_pos", "g:21-29", "g:21-29_pos",
        "dchi", "dchi_pos", "dchi:1", "dchi:1-4", "dchi:5-20", "dchi:21-29", "dskl", "dskl_pos",
        "upd", "upd_pos", "upd:1", "upd:1_pos", "upd:1-4", "upd:1-4_pos", "upd:5-20", "upd:5-20_pos", "upd:21-29",
        "upd:21-29_pos")
NV20 = 20                                    # ventiles of the main grouping
SERIES_VERSION = 3                           # series_<run>.npz layout (rebuilt when it differs)
GROUPS = ("main", "env", "usize", "usizeq", "var", "hist", "conf", "confq", "state", "obs")


def _step_diff(a, axis):
    """one-step differences along the step axis with a NaN in front (same shape as a)"""
    d = np.diff(a, axis=axis)
    pad = list(d.shape)
    pad[axis] = 1
    return np.concatenate([np.full(pad, np.nan), d], axis)


def _flip_mean(v, axes=(1, 2)):
    """mean over the flips available at each unit and the share of them > 0 (NaN where no flip precedes the step)"""
    ok = np.isfinite(v)
    k = ok.sum(axes)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, v, 0.).sum(axes) / k, ((v > 0) & ok).sum(axes) / k


def _lag_mask(name):
    """(nS, T) True where the lag t - s of a flip at step s (FLIP_S) and the unit at step t lies in LAGS[name]"""
    lag = np.arange(T)[None, :] - (np.array(FLIP_S)[:, None] - 1)
    lo, hi = LAGS[name]
    return (lag >= lo) & (lag <= hi)


def _with_lags(out, key, per_flip, pos=True):
    """unit values of the per-flip one-step values per_flip (n, nS, 5, T): all lags (key, key_pos) and each lag group of
    LAGS (key:<lags>, key:<lags>_pos)"""
    out[key], p_ = _flip_mean(per_flip)
    if pos:
        out[key + "_pos"] = p_
    for name in LAGS:
        m = np.broadcast_to(_lag_mask(name)[None, :, None, :], per_flip.shape)
        out[f"{key}:{name}"], out[f"{key}:{name}_pos"] = _flip_mean(np.where(m, per_flip, np.nan))


def unit_values(smax=None, logd=None, chi=None, skl=None, zero_dh=False, lrho=None, lupd=None):
    """per probe-step unit values (n, T) (step t: the unit of observation t), from one agent's per-flip arrays:
    lrho / lrho_pos = log rho(J_t) (the log spectral radius of the step Jacobian) and 1[log rho(J_t) > 0]; smax /
    smax_pos = the Euclidean (u, c) stretch log sigma_max(J_t) and 1[> 0]; g / g_pos = the mean over the flips at steps
    s < t (each of the 5 observations at s in FLIP_S) of g_t = log||Delta_t|| - log||Delta_{t-1}|| and the share of them
    with g_t > 0; dchi / dchi_pos = the same for the one-step change of the flip's belief impact chi_t - chi_{t-1}
    (chi = d_H of the marginals, the mean of the 2 variables); dskl / dskl_pos for the symmetric KL; upd / upd_pos for
    the one-step growth of the flip's update difference, log s_t - log s_{t-1} (lupd = log s); <m>:<lags> the same over
    the flips whose lag t - s lies in LAGS[lags] only; dchi_v / dskl_v (n, T, 2) per variable. zero_dh: d_H changes are
    exactly 0 (naive, T1): set them to 0"""
    out = {}
    for key, arr in (("lrho", lrho), ("smax", smax)):
        if arr is not None:
            sm = np.asarray(arr, np.float64)
            out[key], out[key + "_pos"] = sm, (sm > 0).astype(np.float64)
    if logd is not None:
        _with_lags(out, "g", _step_diff(np.asarray(logd, np.float64), -1))
    if lupd is not None:
        _with_lags(out, "upd", _step_diff(np.asarray(lupd, np.float64), -1))
    for key, arr in (("dchi", chi), ("dskl", skl)):
        if arr is None:
            continue
        a = np.asarray(arr, np.float64)
        d = _step_diff(a.mean(-1), -1)
        dv = _step_diff(a, -2)
        if zero_dh and key == "dchi":
            d, dv = np.where(np.isfinite(d), 0., d), np.where(np.isfinite(dv), 0., dv)
        if key == "dchi":
            _with_lags(out, key, d)
        else:
            out[key], out[key + "_pos"] = _flip_mean(d)
        out[key + "_v"] = _flip_mean(dv)[0]
    return out


def history(logd, chi, skl):
    """per unit (n, T) the size of what step t receives: the mean over the flips before t of log||Delta_{t-1}||, of
    chi_{t-1} (d_H, the mean of the 2 variables) and of SKL_{t-1}: the covariates of the 'hist' control"""
    def prev(a):
        return np.concatenate([np.full(a.shape[:-1] + (1,), np.nan), a[..., :-1]], -1)
    out = {"g": _flip_mean(prev(np.asarray(logd, np.float64)))[0]}
    for k, arr in (("dchi", chi), ("dskl", skl)):
        out[k] = _flip_mean(prev(np.asarray(arr, np.float64).mean(-1)))[0]
    return out


def hist_groups(H, ent):
    """'hist': entanglement fifths within each step and fifth of the unit's history covariate (history()); units
    without a history (step 1) get fifth 0 and drop out (their values are NaN)"""
    out = {}
    for k, h in H.items():
        hq = fifths(np.where(np.isfinite(h), h, -np.inf))
        out[k] = fifths(ent, groups=hq)
    return out


def groupings(cov=None):
    """fifth indices (n, T) of each grouping: main = entanglement fifths within each step (all probes pooled over
    environments); env = within each step and environment; usize = entanglement fifths within each step and fifth of
    Bayes's marginal update size (matched on update size); usizeq = update-size fifths within each step (the comparison
    split: big vs small updates, whatever their entanglement); var = per variable (n, T, 2): that variable's own
    entanglement fifths within each step; conf = entanglement fifths within each step and fifth of Bayes's marginal
    entropy before t (matched on belief confidence); confq = the entropy fifths themselves (most uncertain - most
    confident); state = fifths of the state's expected entanglement sum_k w_k ent_k (the entanglement the observation
    at t would have on average at the same belief, cf_observers: a state-level grouping); obs = fifths of the
    observation's own part ent - sum_k w_k ent_k within the state fifths"""
    cov = covariates() if cov is None else cov
    uq = fifths(cov["upd"])
    G = dict(main=fifths(cov["ent"]), env=fifths(cov["ent"], groups=cov["env"]),
             usize=fifths(cov["ent"], groups=uq), usizeq=uq,
             var=np.stack([fifths(cov["ent_v"][..., v]) for v in range(NV)], -1))
    if "hb" in cov:
        hq = fifths(cov["hb"])
        G["conf"], G["confq"] = fifths(cov["ent"], groups=hq), hq
    if "ent_state" in cov:
        sq = fifths(cov["ent_state"])
        G["state"], G["obs"] = sq, fifths(cov["ent"] - cov["ent_state"], groups=sq)
    return G


def group_sums(U, G, env, nb=NQ):
    """per-environment sums and counts (50, nb) of every measure in U (unit_values) under every grouping in G: dict
    (measure, grouping) -> (S, C); the per-variable grouping pairs each variable's change with its own entanglement
    (dchi_v, dskl_v) and is skipped for the whole-state measures"""
    out = {}
    for m in MEAS:
        if m not in U:
            continue
        for gname, q in G.items():
            if gname == "var":
                if m not in ("dchi", "dskl"):
                    continue
                val = U[m + "_v"]
            else:
                val = U[m]
            e = np.broadcast_to(env.reshape((-1,) + (1,) * (val.ndim - 1)), val.shape)
            ok = np.isfinite(val)
            S, C = np.zeros((gap.N_ENV, nb)), np.zeros((gap.N_ENV, nb))
            np.add.at(S, (e[ok], q[ok]), val[ok])
            np.add.at(C, (e[ok], q[ok]), 1.)
            out[(m, gname)] = (S, C)
    return out


def summarize(run):
    """the per-environment sums of every stored checkpoint of run -> chaos_ent/series_<run>.npz (rebuilt when a
    checkpoint was added): x (K,), raw (K,), S, C (K, len(MEAS), len(GROUPS), 50, NQ) (NaN where a pair is not
    defined), S20, C20 (K, len(MEAS), 50, NV20): the main grouping in twentieths (ventiles: is there a threshold at
    the very most entangled steps?), and the convergence records of the one-step stretch (res_max, steps_max)"""
    eps = [e for e in ti.checkpoints(run) if os.path.exists(ck_file(run, e))]
    ext = [e for e in eps if os.path.exists(ext_file(run, e))]
    path = os.path.join(OUT, f"series_{run}.npz")
    if os.path.exists(path):
        Z = dict(np.load(path))
        if (list(Z["raw"]) == eps and list(Z.get("ext_raw", [])) == ext and int(Z.get("version", 0)) == SERIES_VERSION
                and list(Z["groups"]) == list(GROUPS) and list(Z["meas"]) == list(MEAS)):
            return Z
    cov = covariates()
    G = groupings(cov)
    G20 = {"main": fifths(cov["ent"], nb=NV20)}
    K = len(eps)
    S = np.full((K, len(MEAS), len(GROUPS), gap.N_ENV, NQ), np.nan)
    C = np.full_like(S, np.nan)
    S20 = np.full((K, len(MEAS), gap.N_ENV, NV20), np.nan)
    C20 = np.full_like(S20, np.nan)
    res_max, steps_max = np.zeros(K), np.zeros(K, np.int64)
    lr_conv = np.full((K, 5), np.nan)            # log rho: residual max, share > LR_TOL, steps mean / max, restarts max
    for k, e in enumerate(eps):
        Z = np.load(ck_file(run, e))
        assert np.array_equal(Z["rows"], rows())
        X = np.load(ext_file(run, e)) if e in ext else None
        U = unit_values(Z["smax"], Z["logd"], Z["chi"], Z["skl"], lrho=None if X is None else X["lrho"],
                        lupd=None if X is None else X["lupd"])
        for (m, gname), (s_, c_) in group_sums(U, G, cov["env"]).items():
            S[k, MEAS.index(m), GROUPS.index(gname)], C[k, MEAS.index(m), GROUPS.index(gname)] = s_, c_
        HG = hist_groups(history(Z["logd"], Z["chi"], Z["skl"]), cov["ent"])
        for base, q in HG.items():
            for m in (base, base + "_pos"):
                s_, c_ = group_sums({m: U[m]}, {"hist": q}, cov["env"])[(m, "hist")]
                S[k, MEAS.index(m), GROUPS.index("hist")], C[k, MEAS.index(m), GROUPS.index("hist")] = s_, c_
        for (m, _), (s_, c_) in group_sums(U, G20, cov["env"], nb=NV20).items():
            S20[k, MEAS.index(m)], C20[k, MEAS.index(m)] = s_, c_
        res_max[k], steps_max[k] = Z["smax_res"].max(), Z["smax_steps"].max()
        if X is not None:
            assert np.array_equal(X["rows"], rows())
            lr_conv[k] = (X["lrho_res"].max(), (X["lrho_res"] > LR_TOL).mean(), X["lrho_steps"].mean(),
                          X["lrho_steps"].max(), X["lrho_restarts"].max())
    out = dict(x=gap.train_x(np.array(eps)), raw=np.array(eps), ext_raw=np.array(ext, np.int64), S=S, C=C, S20=S20,
               C20=C20, res_max=res_max, steps_max=steps_max, lr_conv=lr_conv, version=SERIES_VERSION,
               groups=np.array(GROUPS), meas=np.array(MEAS))
    np.savez_compressed(path, **out)
    return out


def observer_sums():
    """the same sums for the exact observers on the same units: (dict agent -> {(measure, grouping): (S, C)}, the same
    for the ventiles of the main grouping); agents bayes (the joint posterior's marginals), naive, adf (lrho = the log
    spectral radius of its exact step Jacobian in the zero-sum chart, smax = its log sigma_max there); bayes and naive
    have no state measures here (exactly 0 by T1), naive's d_H change is exactly 0 (T1) and its update difference is
    exactly 0 after the flip step (no update measure)"""
    O = observers_cached()
    O2 = observers_upd_cached()
    cov = covariates()
    G = groupings(cov)
    U = {"bayes": unit_values(None, None, O["chi_bayes"], O["skl_bayes"], lupd=O2["lupd_bayes"]),
         "naive": unit_values(None, None, O["chi_naive"], O["skl_naive"], zero_dh=True),
         "adf": unit_values(O["smax_adf"], O["logd_adf"], O["chi_adf"], O["skl_adf"], lrho=O2["lrho_adf_eig"],
                            lupd=O2["lupd_adf"])}
    G20 = {"main": fifths(cov["ent"], nb=NV20)}
    out = {a: group_sums(u, G, cov["env"]) for a, u in U.items()}
    for a in ("bayes", "naive", "adf"):
        HG = hist_groups(history(O["logd_adf"] if a == "adf" else np.zeros_like(O["logd_adf"]), O[f"chi_{a}"],
                                 O[f"skl_{a}"]), cov["ent"])
        for base, q in HG.items():
            for m in (base, base + "_pos"):
                if m in U[a] and not (a != "adf" and base == "g"):
                    out[a][(m, "hist")] = group_sums({m: U[a][m]}, {"hist": q}, cov["env"])[(m, "hist")]
    return (out,
            {a: group_sums(u, G20, cov["env"], nb=NV20) for a, u in U.items()})


def ratio(S, C, B=None):
    """pooled ratio of sums over the environments: S, C (..., 50, NQ) -> value (..., NQ), the 1,000 resampled values
    (1,000, ..., NQ) (td_tt_gap.boot_counts: the shared resamples)"""
    B = gap.boot_counts() if B is None else B
    S, C = np.asarray(S, np.float64), np.asarray(C, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = S.sum(-2) / C.sum(-2)
        bt = np.einsum("be,...eq->b...q", B, S) / np.einsum("be,...eq->b...q", B, C)
    return val, bt


def contrast(S, C, B=None):
    """the most minus the least entangled fifth (value (...), resamples (1,000, ...)), paired in every resample, and
    the all-unit mean (value, resamples)"""
    v, b = ratio(S, C, B)
    vo, bo = ratio(np.asarray(S).sum(-1, keepdims=True), np.asarray(C).sum(-1, keepdims=True), B)
    return (v[..., -1] - v[..., 0], b[..., -1] - b[..., 0]), (vo[..., 0], bo[..., 0])


def iv(bt):
    """5-95% interval over the resamples (axis 0)"""
    return np.percentile(bt, 5, 0), np.percentile(bt, 95, 0)



# ═══════════════════════════════════════════════════════════════════
# Checks (GPU for 1-3, CPU for 4-6) -> chaos_ent/checks.json
# ═══════════════════════════════════════════════════════════════════

CHECK_CKS = (("trained", 0), ("trained", 75), ("trained", 400), ("trained", 2100), ("trained", 49995), ("echo", 300),
             ("echo", 2100), ("echo", 49995))


def _maxabs(a, b):
    """the largest |a - b| where both are finite; where only one is finite (e.g. an exact 0 difference of the fp32
    subtraction, log 0), the count of such entries is appended as a list [maxabs, count]"""
    a, b = (x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x) for x in (a, b))
    ok = np.isfinite(a) & np.isfinite(b)
    mis = int((np.isfinite(a) != np.isfinite(b)).sum())
    v = float(np.abs(a[ok] - b[ok]).max())
    return v if mis == 0 else [v, mis]


@torch.no_grad()
def checks(dev=None):
    """1. the difference-form flips (fp32) vs float64 reruns (and the direct fp32 subtraction it replaces): log||Delta||,
    its one-step growth g, chi, its one-step change, SKL and its change, the update difference log s and its growth,
    all 1,000 probes, CHECK_CKS; 2. Lanczos log sigma_max vs the exact SVD of the explicit 2,000 x 2,000 step Jacobian
    (float64), 8 random (probe, step) units and the 4 that took the most Lanczos steps, 200 probes, CHECK_CKS; 2b. the
    stored Arnoldi log rho vs the exact eigenvalues of the explicit Jacobian (24 random units of the first 200 probes,
    the 6 slowest, the 6 with the largest residual); 2c. log rho from a second random start (first 100 probes) vs the
    stored values; 3. the intact forward vs the engine (ti.forward)
    log beliefs, 200 probes; 4. the observers vs td_tt_chaos's caches (pilot/bayes_side.npz: Bayes's and ADF's chi at
    flip steps 1, 3, 5, 10; lyap/adf.npz: ADF's log sigma_max at steps 6-30 on its 500 probes) and naive's d_H change
    (exactly 0 by T1); 5. the stored network chi vs the chaos sweep (fp16 means over the 5 observations); 6. the fifth
    counts per step. -> chaos_ent/checks.json"""
    dev = _gpu_cap(ti._device(dev))
    out = {}
    rws = rows()
    for run, raw in CHECK_CKS:
        p32, t32 = setup(run, raw, dev)
        p64, t64 = setup(run, raw, dev, torch.float64)
        parts = {"diff_form": [], "fp32_direct": [], "fp64": []}
        for b0 in range(0, len(rws), 250):                                    # all probes
            envt, code = codes(rws[b0:b0 + 250], dev)
            A, U, C, lg = replay(p32, t32, envt, code)
            A6, U6, C6, lg6 = replay(p64, t64, envt, code)
            for lab, fx in (("diff_form", flips(p32, t32, envt, code, A, C, lg, upd=True)),
                            ("fp32_direct", flips_direct(p32, t32, envt, code, U, C, lg)),
                            ("fp64", flips_direct(p64, t64, envt, code, U6, C6, lg6))):
                parts[lab].append([x.cpu().numpy() for x in fx])
        F = {lab: [np.concatenate([c[j] for c in v]) for j in range(4)] for lab, v in parts.items()}
        l6, c6, s6, u6 = F["fp64"]
        res = {}
        for lab in ("diff_form", "fp32_direct"):
            ld, chi, skl, lu = F[lab]
            res[lab] = dict(logd=_maxabs(ld, l6), g=_maxabs(np.diff(ld, axis=-1), np.diff(l6, axis=-1)),
                            chi=_maxabs(chi, c6), dchi=_maxabs(np.diff(chi.mean(-1), axis=-1),
                                                               np.diff(c6.mean(-1), axis=-1)),
                            skl=_maxabs(skl, s6), dskl=_maxabs(np.diff(skl.mean(-1), axis=-1),
                                                               np.diff(s6.mean(-1), axis=-1)),
                            lupd=_maxabs(lu, u6), upd_growth=_maxabs(np.diff(lu, axis=-1), np.diff(u6, axis=-1)))
        res["fp64_logd_min"] = float(np.nanmin(l6))
        res["fp64_median_abs_g"] = float(np.nanmedian(np.abs(np.diff(l6, axis=-1))))
        res["fp64_median_abs_dchi"] = float(np.nanmedian(np.abs(np.diff(c6.mean(-1), axis=-1))))
        res["n_probes"] = len(rws)
        out[f"flips_vs_fp64_{run}_{raw}"] = res
        # 2. Lanczos vs exact SVD
        envt, code = codes(rws[:200], dev)
        A, U, C, lg = replay(p32, t32, envt, code)
        est, rr, ns = smax(p32, A, C)
        rng = np.random.default_rng(CHECK_SEED)
        pick = [(int(rng.integers(200)), int(rng.integers(T))) for _ in range(8)]
        pick += [(int(k // T), int(k % T)) for k in np.argsort(-ns.ravel(), kind="stable")[:4]]
        H = p32["LSTM.weight_hh_l0"].shape[1]
        Whh = p32["LSTM.weight_hh_l0"].double()
        ex, gp = [], []
        for i, t in pick:
            gts = _gates(A[i:i + 1, t].double()) + (torch.tanh(C[i:i + 1, t + 1].double()),)
            I_ = torch.eye(2 * H, device=dev, dtype=torch.float64)[None]
            dh, dc = tc.tangent(I_[..., :H], I_[..., H:], C[i:i + 1, t].double(), gts, Whh)
            sv = np.linalg.svd(torch.cat([dh, dc], -1)[0].cpu().numpy(), compute_uv=False)
            ex.append(np.log(sv[0]))
            gp.append(np.log(sv[0]) - np.log(sv[1]))
        err = np.abs(np.array(ex) - est[tuple(np.array(pick).T)])
        out[f"lanczos_vs_svd_{run}_{raw}"] = dict(maxabs=float(err.max()), gap_min=float(min(gp)),
                                                  steps_max=int(ns.max()), res_max=float(rr.max()))
        # 2b. Arnoldi log rho (the stored values) vs the exact eigenvalues of the explicit Jacobian: 24 random units of
        # the first 200 probes, the 6 that took the most Krylov steps and the 6 with the largest residual
        if os.path.exists(ext_file(run, raw)):
            X = np.load(ext_file(run, raw))
            lr, st_, rs_ = X["lrho"][:200], X["lrho_steps"][:200], X["lrho_res"][:200]
            pk = [(int(rng.integers(200)), int(rng.integers(T))) for _ in range(24)]
            pk += [(int(k // T), int(k % T)) for k in np.argsort(-st_.ravel(), kind="stable")[:6]]
            pk += [(int(k // T), int(k % T)) for k in np.argsort(-rs_.ravel(), kind="stable")[:6]]
            exr, gap12 = [], []
            for i, t in pk:
                gts = _gates(A[i:i + 1, t].double()) + (torch.tanh(C[i:i + 1, t + 1].double()),)
                I_ = torch.eye(2 * H, device=dev, dtype=torch.float64)[None]
                dh, dc = tc.tangent(I_[..., :H], I_[..., H:], C[i:i + 1, t].double(), gts, Whh)
                lev = np.sort(np.log(np.abs(np.linalg.eigvals(torch.cat([dh, dc], -1)[0].cpu().numpy()))))[::-1]
                exr.append(lev[0])
                gap12.append(lev[0] - lev[1])
            e_ = np.array(exr) - lr[tuple(np.array(pk).T)]
            out[f"lrho_vs_exact_{run}_{raw}"] = dict(
                random_median_abs=float(np.median(np.abs(e_[:24]))), random_maxabs=float(np.abs(e_[:24]).max()),
                slowest_maxabs=float(np.abs(e_[24:30]).max()), largest_res_maxabs=float(np.abs(e_[30:]).max()),
                n_above_1e4=int((np.abs(e_) > 1e-4).sum()), n=len(pk), gap12_min=float(min(gap12)),
                stored_res_max=float(X["lrho_res"].max()),
                stored_share_above_tol=float((X["lrho_res"] > LR_TOL).mean()))
            # 2c. a second, independent random start (first 100 probes) vs the stored values
            e1, c1 = codes(rws[:100], dev)
            A1, U1, C1, _ = replay(p32, t32, e1, c1)
            lr2 = lrho_units(p32, A1, C1, LR_SEED + 7777)[0]
            d2 = lr2 - X["lrho"][:100]
            out[f"lrho_second_start_{run}_{raw}"] = dict(maxabs=float(np.abs(d2).max()), mean=float(d2.mean()),
                                                         share_above_1e4=float((np.abs(d2) > 1e-4).mean()),
                                                         n_units=int(d2.size))
        # 3. intact forward vs the engine
        eng = ti.forward(ti.load(run, raw), rows=rws[:200], device=dev)["logq"].astype(np.float64)
        lq = torch.log_softmax(lg.cumsum(1).reshape(200, T, NV, R), -1).cpu().numpy()
        out[f"forward_vs_engine_{run}_{raw}"] = float(np.abs(lq - eng).max())
        print(run, raw, json.dumps({k: v for k, v in out.items() if k.endswith(f"{run}_{raw}")}), flush=True)
        del p32, p64, t32, t64
        torch.cuda.empty_cache()
    # 4. observers vs td_tt_chaos's caches
    O = observers_cached()
    pil = dict(np.load(os.path.join(tc.OUT, "pilot", "bayes_side.npz")))
    assert np.array_equal(pil["rows"], rws)
    ix = [list(tc.FLIP_S).index(s_) for s_ in FLIP_S]
    for a in ("bayes", "adf"):
        out[f"observers_chi_{a}_vs_chaos_pilot"] = _maxabs(O[f"chi_{a}"], pil[f"chi_{a}"][:, ix])
    A_ = dict(np.load(os.path.join(tc.OUT, "lyap", "adf.npz")))
    sub = np.searchsorted(rws, A_["rows"])
    assert np.array_equal(rws[sub], A_["rows"])
    out["observers_adf_smax_vs_chaos_lyap"] = _maxabs(O["smax_adf"][sub, tc.SMAX_T0:], A_["smax"])
    out["naive_dH_change_maxabs"] = float(np.nanmax(np.abs(np.diff(O["chi_naive"], axis=-2))))
    # 5. stored network chi vs the chaos sweep (fp16 means over the observations)
    for run, raw in (("trained", 2100), ("trained", 49995), ("echo", 49995)):
        f = ck_file(run, raw)
        sw = os.path.join(tc.OUT, "sweep", f"{run}_{raw:05d}.npz")
        if not (os.path.exists(f) and os.path.exists(sw)):
            continue
        mine = np.load(f)["chi"].astype(np.float64).mean(2)                    # (n, nS, T, 2)
        theirs = np.load(sw)["chi"].astype(np.float64)[:, ix]
        ulp = np.spacing(np.abs(theirs).astype(np.float16)).astype(np.float64)
        ok = np.isfinite(theirs)
        d = np.abs(mine - theirs)[ok]
        out[f"chi_vs_chaos_sweep_{run}_{raw}"] = dict(maxabs=float(d.max()),
                                                      share_within_1ulp=float((d <= ulp[ok]).mean()),
                                                      max_in_ulps=float((d / ulp[ok]).max()))
    # 6. fifth counts per step
    G = groupings()
    for gname, q in G.items():
        q2 = q.reshape(q.shape[0], T, -1)
        cnt = np.stack([np.bincount(q2[:, t].ravel(), minlength=NQ) for t in range(T)])
        out[f"fifth_counts_{gname}_min_max"] = [int(cnt.min()), int(cnt.max())]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "checks.json")
    allc = json.load(open(path)) if os.path.exists(path) else {}
    allc.update(out)
    with open(path, "w") as fh:
        json.dump(allc, fh, indent=1)
    for k_, v in out.items():
        print(f"  {k_}: {v}", flush=True)
    return out


# ═══════════════════════════════════════════════════════════════════
# Numbers (every number of the docstring)
# ═══════════════════════════════════════════════════════════════════

BASE_LAB = {"lrho": "log rho(J_t) (one-step expansion, coordinate-free)",
            "smax": "Euclidean (u, c) stretch log sigma_max(J_t) (chart-dependent)",
            "g": "g_t (growth of the flip's state difference)", "dchi": "d_H change (of the flip's belief difference)",
            "dskl": "SKL change", "upd": "update-difference growth log s_t - log s_(t-1)"}
SHOW_X = (10, 20, 60, 120, 180, 240, 360, 480, 560, 720, 880, 1200, 1680, 2400, 4200, 7800, 12000, 20000, 39996)


def mlab(m):
    """the printed name of measure m"""
    pos = m.endswith("_pos")
    core = m[:-4] if pos else m
    base, _, lags = core.partition(":")
    lag_txt = f", lags {lags}" if lags else (", all lags" if base in ("g", "dchi", "upd") else "")
    return ("P(> 0): " if pos else "") + BASE_LAB[base] + lag_txt


def x_naive():
    """training episodes at which the Trained RNN's goal accuracy passes naive's for good (td_tt_jumps, c0)"""
    acc = gap.load("jumps")
    return float(acc["c0"][list(acc["names"]).index("goal accuracy at T")])


def _f(v, lo=None, hi=None, f="+.4f"):
    return f"{v:{f}}" + ("" if lo is None else f" [{lo:{f}}, {hi:{f}}]")


def _pct(b, q):
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(b, q, 0)


def run_curves(Z, m, gname="main", eq=False):
    """fifths (K, NQ) with 5-95% (K, NQ) x 2, the contrast (K,) with its interval, the all-unit mean (K,) with its
    interval, for measure m under grouping gname of a series file (eq: equal-weight environment means instead of the
    pooled ratio of sums)"""
    S, C = Z["S"][:, MEAS.index(m), GROUPS.index(gname)], Z["C"][:, MEAS.index(m), GROUPS.index(gname)]
    rf = ratio_eq if eq else ratio
    v, b = rf(S, C)
    vo, bo = rf(np.asarray(S).sum(-1, keepdims=True), np.asarray(C).sum(-1, keepdims=True))
    cb = b[..., -1] - b[..., 0]
    return dict(v=v, lo=_pct(b, 5), hi=_pct(b, 95), c=v[:, -1] - v[:, 0], clo=_pct(cb, 5), chi=_pct(cb, 95),
                o=vo[:, 0], olo=_pct(bo[..., 0], 5), ohi=_pct(bo[..., 0], 95))


def ratio_eq(S, C, B=None):
    """equal-weight environment means: S / C per environment, then the mean over the environments (a resample weights
    each environment by how often it is drawn): value (..., NQ), resamples (1,000, ..., NQ)"""
    B = gap.boot_counts() if B is None else B
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.asarray(S, np.float64) / np.asarray(C, np.float64)
        ok = np.isfinite(m)
        mz = np.where(ok, m, 0.)
        val = mz.sum(-2) / ok.sum(-2)
        bt = np.einsum("be,...eq->b...q", B, mz) / np.einsum("be,...eq->b...q", B, ok.astype(np.float64))
    return val, bt


def _ranges(x, mask, counts=True):
    """the checkpoint ranges (training episodes) where mask holds, as text (counts: the number of checkpoints)"""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append(f"{x[i]:,.0f}" + ("" if j == i else f"-{x[j]:,.0f}") + (f" ({j - i + 1})" if counts else ""))
            i = j + 1
        else:
            i += 1
    return ", ".join(out) if out else "none"


def delta_size(run, raw, cov=None, G=None):
    """is the growth contrast a size effect? the mean log||Delta_{t-1}|| (the flip's state difference entering step t;
    unit = the mean over its flips) per entanglement fifth of observation t (main grouping): value (NQ,) and the
    fifth 5 - fifth 1 contrast with its interval"""
    cov = covariates() if cov is None else cov
    G = groupings(cov) if G is None else G
    ld = np.load(ck_file(run, raw))["logd"].astype(np.float64)
    prev = np.concatenate([np.full(ld.shape[:-1] + (1,), np.nan), ld[..., :-1]], -1)  # log||Delta_{t-1}|| at t
    U = {"g": _flip_mean(prev)[0]}
    S, C = group_sums(U, {"main": G["main"]}, cov["env"])[("g", "main")]
    (cv, cb), _ = contrast(S, C)
    return ratio(S, C)[0], (float(cv), *(float(x_) for x_ in iv(cb)))


def _argmax(a):
    return int(np.nanargmax(a)) if np.isfinite(a).any() else 0


def _argmin(a):
    return int(np.nanargmin(a)) if np.isfinite(a).any() else 0


def _curve_summary(x, r, pad="    "):
    """the summary lines of one measure's curves r (run_curves) through training"""
    o, f5, f1 = r["o"], r["v"][:, -1], r["v"][:, 0]
    ko, kn, k5, k1, kc, kd = _argmax(o), _argmin(o), _argmax(f5), _argmax(f1), _argmax(r["c"]), _argmin(r["c"])
    allf_pos = np.all(r["lo"] > 0, -1)
    allf_neg = np.all(r["hi"] < 0, -1)
    kf = np.unravel_index(_argmax(r["v"].ravel()), r["v"].shape)
    print(f"{pad}all-unit mean: max {_f(o[ko], r['olo'][ko], r['ohi'][ko])} at {x[ko]:,.0f}, min {o[kn]:+.4f} at "
          f"{x[kn]:,.0f}, end {_f(o[-1], r['olo'][-1], r['ohi'][-1])}; > 0 (5% bound) at {_ranges(x, r['olo'] > 0)}; "
          f"< 0 (95% bound) at {_ranges(x, r['ohi'] < 0)}")
    print(f"{pad}most entangled fifth: max {_f(f5[k5], r['lo'][k5, -1], r['hi'][k5, -1])} at {x[k5]:,.0f}, end "
          f"{_f(f5[-1], r['lo'][-1, -1], r['hi'][-1, -1])}; > 0 (5% bound) at {_ranges(x, r['lo'][:, -1] > 0)}; < 0 "
          f"(95% bound) at {_ranges(x, r['hi'][:, -1] < 0)}; > 0 while the all-unit mean < 0 (both bounds) at "
          f"{_ranges(x, (r['lo'][:, -1] > 0) & (r['ohi'] < 0))}")
    print(f"{pad}least entangled fifth: max {_f(f1[k1], r['lo'][k1, 0], r['hi'][k1, 0])} at {x[k1]:,.0f}; > 0 (5% "
          f"bound) at {_ranges(x, r['lo'][:, 0] > 0)}; every fifth > 0 (each 5% bound) at {_ranges(x, allf_pos)}; "
          f"every fifth < 0 (each 95% bound) at {_ranges(x, allf_neg)}; largest fifth value {r['v'][kf]:+.4f} [95% "
          f"{r['hi'][kf]:+.4f}] (fifth {kf[1] + 1} at {x[kf[0]]:,.0f})")
    print(f"{pad}contrast: > 0 (5% bound) at {_ranges(x, r['clo'] > 0)}; < 0 (95% bound) at "
          f"{_ranges(x, r['chi'] < 0)}; max {_f(r['c'][kc], r['clo'][kc], r['chi'][kc])} at {x[kc]:,.0f}; min "
          f"{_f(r['c'][kd], r['clo'][kd], r['chi'][kd])} at {x[kd]:,.0f}; end {_f(r['c'][-1], r['clo'][-1], r['chi'][-1])}")
    return kc


def numbers():
    """every number of the docstring"""
    path = os.path.join(OUT, "checks.json")
    if os.path.exists(path):
        print("== checks (checks.json)")
        for k_, v in json.load(open(path)).items():
            print(f"  {k_}: {v}")
    O, O20 = observer_sums()
    O2 = observers_upd_cached()
    xn = x_naive()
    print(f"== data: {len(rows())} probes (20 per test environment x 50), units = (probe, step): log rho and the "
          f"Euclidean stretch at steps 1-30, flips (each of the 5 observations at steps {FLIP_S}) at the steps after the "
          f"flip (lag = t - s); the Trained RNN passes naive at {xn:,.0f} episodes. ADF: log of its step Jacobian's "
          f"largest |eigenvalue| vs log(1 + rho_1): max |diff| {np.abs(O2['lrho_adf_eig'] - O2['lrho_adf_cc']).max():.1e}")
    print("== exact observers (fifths 1..5 = least .. most entangled; contrast = fifth 5 - fifth 1 [5-95%]; all = the "
          "all-unit mean; v20 = top - bottom twentieth; controls: the contrast under each other grouping)")
    for a in ("bayes", "naive", "adf"):
        for m in MEAS:
            if (m, "main") not in O[a]:
                continue
            S, C = O[a][(m, "main")]
            v, b = ratio(S, C)
            (cv, cb), (ov, ob) = contrast(S, C)
            v20 = ratio(*O20[a][(m, "main")])[0]
            ctl = "; ".join(f"{g_} {_f(*(lambda r: (r[0][0], *iv(r[0][1])))(contrast(*O[a][(m, g_)])))}"
                            for g_ in GROUPS[1:] if (m, g_) in O[a])
            veq, beq = ratio_eq(S, C)
            ceq = beq[..., -1] - beq[..., 0]
            print(f"  {a:6s} {mlab(m)}: fifths {' '.join(f'{x_:+.4f}' for x_ in v)}; contrast {_f(cv, *iv(cb))} "
                  f"(equal-weight environments {_f(veq[-1] - veq[0], *iv(ceq))}); all {_f(ov, *iv(ob))}; v20 "
                  f"{v20[-1] - v20[0]:+.4f}; controls: {ctl}")
    if os.path.exists(os.path.join(OUT, "cf", "observers.npz")):
        print("== counterfactual control (fifth 5 - fifth 1 [5-95%]; actual = expected over the 32 observations at the "
              "same state and flip history + own = what the step's actual observation adds)")
        for a in ("bayes", "adf"):
            st, fv = cf_stats(agent=a)
            for m in ("g", "dchi"):
                if (m, "own") in st:
                    print(f"  {a:7s} {mlab(m)}: " + "; ".join(f"{p_} {_f(*st[(m, p_)])}" for p_ in
                                                               ("actual", "expected", "own"))
                          + "; own by fifth " + " ".join(f"{v_:+.4f}" for v_ in fv[m]))
            if os.path.exists(os.path.join(OUT, "cf", "observers_upd.npz")):
                for lags in LAGSET:
                    st, fv = cf_upd_stats(agent=a, lags=lags)
                    print(f"  {a:7s} update-difference growth, lags {lags}: " + "; ".join(
                        f"{p_} {_f(*st[p_])}" for p_ in ("actual", "expected", "own"))
                        + "; own by fifth " + " ".join(f"{v_:+.4f}" for v_ in fv))
        st, fv, chk = cf_adf_lrho_stats()
        print(f"  adf     {mlab('lrho')}: " + "; ".join(f"{p_} {_f(*st[p_])}" for p_ in ("actual", "expected", "own"))
              + "; own by fifth " + " ".join(f"{v_:+.4f}" for v_ in fv)
              + f"; |pattern pass at the actual observation - lrho_adf_eig| max {chk:.1e}; Bayes and naive: 0")
        for run in RUNS:
            for raw in ti.checkpoints(run):
                if not os.path.exists(cf_lrho_file(run, raw)):
                    continue
                st, fv = cf_lrho_stats(run, raw)
                parts, _ = cf_lrho_parts(run, raw)
                d_ = np.abs(parts["actual_cf"] - parts["actual"])
                print(f"  {run:7s} {gap.train_x(raw):7,.0f} {mlab('lrho')} ({len(np.load(cf_lrho_file(run, raw))['rows'])}"
                      f" probes): " + "; ".join(f"{p_} {_f(*st[p_])}" for p_ in ("actual", "actual_cf", "expected", "own"))
                      + "; own by fifth " + " ".join(f"{v_:+.4f}" for v_ in fv)
                      + f"; |pattern pass - stored| median {np.median(d_):.1e}, max {d_.max():.1e}")
        for run in RUNS:
            for raw in cf_raws(run):
                if os.path.exists(cf_file(run, raw)):
                    st, fv = cf_stats(run, raw)
                    for m in ("g", "dchi"):
                        print(f"  {run:7s} {gap.train_x(raw):7,.0f} {mlab(m)}: " + "; ".join(
                            f"{p_} {_f(*st[(m, p_)])}" for p_ in ("actual", "expected", "own"))
                              + "; own by fifth " + " ".join(f"{v_:+.4f}" for v_ in fv[m]))
                if os.path.exists(cf_upd_file(run, raw)):
                    for lags in LAGSET:
                        st, fv = cf_upd_stats(run, raw, lags=lags)
                        print(f"  {run:7s} {gap.train_x(raw):7,.0f} update-difference growth, lags {lags}: " + "; ".join(
                            f"{p_} {_f(*st[p_])}" for p_ in ("actual", "expected", "own"))
                              + "; own by fifth " + " ".join(f"{v_:+.4f}" for v_ in fv))
    for run in RUNS:
        Z = summarize(run)
        x = Z["x"]
        ks = sorted({int(np.argmin(np.abs(np.log(x) - np.log(w)))) for w in SHOW_X})
        lc = Z["lr_conv"]
        ok = np.isfinite(lc[:, 0])
        print(f"== {run}: {len(x)} of {len(ti.checkpoints(run))} checkpoints, log rho at {ok.sum()}; Arnoldi: relative "
              f"residual max {np.nanmax(lc[:, 0]):.1e}, share of units above {LR_TOL:.0e} max {np.nanmax(lc[:, 1]):.4f} "
              f"(at {x[_argmax(lc[:, 1])]:,.0f}; > 0 at {_ranges(x, lc[:, 1] > 0)}), mean Krylov steps "
              f"{np.nanmin(lc[:, 2]):.0f}-{np.nanmax(lc[:, 2]):.0f}, max {np.nanmax(lc[:, 3]):.0f}, restarts max "
              f"{np.nanmax(lc[:, 4]):.0f}; Euclidean stretch Lanczos residual max {Z['res_max'].max():.1e}")
        sm = np.stack([np.load(ck_file(run, e))["smax"] for e in Z["raw"]])            # (K, n, T) Euclidean stretch
        pos1, posl = (sm[:, :, 0] > 0).mean(1), (sm[:, :, 1:] > 0).mean((1, 2))
        nl = (sm[:, :, 1:] > 0).sum((1, 2))
        print(f"  Euclidean (u, c) stretch > 0: share at step 1 {pos1.min():.3f}-{pos1.max():.3f}; at steps 2-30 max "
              f"{posl.max():.4f} ({nl.max()} of {sm[0, :, 1:].size} units, at {x[_argmax(posl)]:,.0f}), > 0 at any of "
              f"steps 2-30 at {int((nl > 0).sum())} checkpoints; largest value at steps 2-30 {sm[:, :, 1:].max():+.3f}")
        if np.isfinite(lc[:, 0]).any():
            Xs = [np.load(ext_file(run, e)) for e in Z["raw"] if os.path.exists(ext_file(run, e))]
            lr_all = np.stack([X_["lrho"] for X_ in Xs])
            cx = np.mean([X_["lrho_cx"].mean() for X_ in Xs])
            print(f"  log rho > 0: share of all units over the checkpoints {(lr_all > 0).mean():.4f}; largest value "
                  f"{lr_all.max():+.4f}; complex leading eigenvalue: share {cx:.3f}")
        for m in MEAS:
            r = run_curves(Z, m)
            if not np.isfinite(r["o"]).any():
                continue
            print(f"  -- {mlab(m)}")
            for k in ks:
                if np.isfinite(r["o"][k]):
                    print(f"    {x[k]:7,.0f}: all {_f(r['o'][k], r['olo'][k], r['ohi'][k])}; fifths "
                          + " ".join(f"{v_:+.4f}" for v_ in r["v"][k])
                          + f"; contrast {_f(r['c'][k], r['clo'][k], r['chi'][k])}")
            kc = _curve_summary(x, r)
            re_ = run_curves(Z, m, eq=True)
            print(f"    equal-weight environment means: contrast at the max ({x[kc]:,.0f}) "
                  f"{_f(re_['c'][kc], re_['clo'][kc], re_['chi'][kc])}, end "
                  f"{_f(re_['c'][-1], re_['clo'][-1], re_['chi'][-1])}"
                  f"; all-unit mean at the end {re_['o'][-1]:+.4f}")
            S20, C20 = Z["S20"][:, MEAS.index(m)], Z["C20"][:, MEAS.index(m)]
            v20, b20 = ratio(S20, C20)
            t20 = v20[:, -1]
            k2 = _argmax(t20)
            print(f"    top twentieth: max {t20[k2]:+.4f} [{_pct(b20[:, k2, -1], 5):+.4f}, {_pct(b20[:, k2, -1], 95):+.4f}] "
                  f"at {x[k2]:,.0f}; > 0 (5% bound) at {_ranges(x, _pct(b20[..., -1], 5) > 0)}; top - bottom "
                  f"twentieth max {np.nanmax(v20[:, -1] - v20[:, 0]):+.4f} at {x[_argmax(v20[:, -1] - v20[:, 0])]:,.0f}")
            if m == "g":
                for k_ in sorted({kc, len(x) - 1}):
                    v_, c_ = delta_size(run, Z["raw"][k_])
                    print(f"    log||Delta_(t-1)|| entering step t at {x[k_]:,.0f}: fifths "
                          + " ".join(f"{q_:+.3f}" for q_ in v_) + f"; contrast {_f(*c_)}")
            for g_ in GROUPS[1:]:
                if not np.isfinite(Z["S"][:, MEAS.index(m), GROUPS.index(g_)]).any():
                    continue
                rc = run_curves(Z, m, g_)
                km_ = _argmax(rc["c"])
                print(f"    control {g_:6s}: contrast > 0 (5% bound) at {_ranges(x, rc['clo'] > 0)}; < 0 at "
                      f"{_ranges(x, rc['chi'] < 0)}; max {_f(rc['c'][km_], rc['clo'][km_], rc['chi'][km_])} at "
                      f"{x[km_]:,.0f}; end {rc['c'][-1]:+.4f}; at the main max ({x[kc]:,.0f}) "
                      f"{_f(rc['c'][kc], rc['clo'][kc], rc['chi'][kc])}; most entangled fifth max "
                      f"{np.nanmax(rc['v'][:, -1]):+.4f}")


# ═══════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════

COL = {"trained": "#1f77b4", "echo": "#ff7f0e", "bayes": "k", "naive": "0.55"}
ADF_LS = (0, (6, 2, 1.5, 2))                                                    # factorized-state Bayes: dash-dot
RUN_NAME = {"trained": "Trained RNN", "echo": "ESN"}
# rows of the figure: (measure family, [panels]); a panel = (kind, run, measure); kind fifths: the entanglement fifths
# of one network through training; contrast: fifth 5 - fifth 1 of both networks with the exact observers' contrasts
FIG_ROWS = (("lrho", (("fifths", "trained", "lrho"), ("fifths", "trained", "lrho_pos"), None,
                      ("fifths", "echo", "lrho"), ("contrast", None, "lrho"))),
            ("g", (("fifths", "trained", "g:1"), ("fifths", "trained", "g:1-4"), ("fifths", "trained", "g:5-20"),
                   ("fifths", "echo", "g"), ("contrast", None, "g"))),
            ("dchi", (("fifths", "trained", "dchi:1"), ("fifths", "trained", "dchi:1-4"),
                      ("fifths", "trained", "dchi:5-20"), ("fifths", "echo", "dchi"), ("contrast", None, "dchi"))),
            ("upd", (("fifths", "trained", "upd:1"), ("fifths", "trained", "upd:1-4"),
                     ("fifths", "trained", "upd:5-20"), ("fifths", "echo", "upd:5-20"),
                     ("contrast", None, "upd:5-20"))))


def _obs_value(O, a, m, part="all"):
    """an observer's all-unit mean ('all') or its fifth 5 - fifth 1 contrast ('contrast') of measure m (main
    grouping); bayes and naive state measures (lrho, smax, g, their shares and lag groups) are exactly 0 (T1); None
    where not defined"""
    if (m, "main") not in O.get(a, {}):
        base = m.split(":")[0].replace("_pos", "")
        if a in ("bayes", "naive") and base in ("lrho", "smax", "g"):
            return 0.
        return None
    (cv, _), (ov, _) = contrast(*O[a][(m, "main")])
    return float(ov if part == "all" else cv)


def _own_points(m, run=None, agent=None):
    """the counterfactual own part [value, 5%, 95%] of measure m (lrho; g, dchi all lags; upd[:<lags>]) at a network's
    counterfactual checkpoints ((x, values) arrays) or for an exact observer (values; None where not computed)"""
    base, _, lags = m.partition(":")
    if base == "lrho":
        if agent is not None:
            return cf_adf_lrho_stats()[0]["own"] if agent == "adf" else None
        pts = [(gap.train_x(raw), cf_lrho_stats(run, raw)[0]["own"]) for raw in ti.checkpoints(run)
               if os.path.exists(cf_lrho_file(run, raw))]
    elif base == "upd":
        if agent is not None:
            return cf_upd_stats(agent=agent, lags=lags or "all")[0]["own"]
        pts = [(gap.train_x(raw), cf_upd_stats(run, raw, lags=lags or "all")[0]["own"]) for raw in cf_raws(run)
               if os.path.exists(cf_upd_file(run, raw))]
    else:
        if agent is not None:
            return cf_stats(agent=agent)[0].get((base, "own"))
        pts = [(gap.train_x(raw), cf_stats(run, raw)[0][(base, "own")]) for raw in cf_raws(run)
               if os.path.exists(cf_file(run, raw))]
    return np.array([p_[0] for p_ in pts]), np.array([p_[1] for p_ in pts]).reshape(-1, 3)


def figure(path=None):
    """four measures through training (rows: one-step expansion log rho, the flip's state-difference growth, its
    belief-difference growth, its update-difference growth), by entanglement fifth (Trained RNN by lag group, ESN), and
    the most minus least entangled fifth of both networks with the exact observers' contrasts and the counterfactual
    own parts"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    FL, TS = fs["label"], 11.
    O, _ = observer_sums()
    Z = {r: summarize(r) for r in RUNS}
    xn = x_naive()
    qc = plt.cm.viridis(np.linspace(.08, .95, NQ))                               # bright = more entangled
    have = {"g": os.path.exists(os.path.join(OUT, "cf", "observers.npz")),
            "upd": os.path.exists(os.path.join(OUT, "cf", "observers_upd.npz"))}
    have["dchi"] = have["g"]
    have["lrho"] = any(os.path.exists(cf_lrho_file(r, e)) for r in RUNS for e in ti.checkpoints(r))
    fig = plt.figure(figsize=(18.4, 17.6))
    gsp = fig.add_gridspec(4, 5, hspace=.92, wspace=.34, left=.045, right=.988, top=.80, bottom=.085)
    header = {"lrho": r"One-step expansion, coordinate-free:  $\log\rho(J_t)$,  $\rho$ = the largest $|$eigenvalue$|$ "
                      r"of the step Jacobian  $J_t = \partial(u_t, c_t)\,/\,\partial(u_{t-1}, c_{t-1})$   ($> 0$: an "
                      "eigen-direction that the step map expands)",
              "g": r"Growth of a flipped observation's state difference:  $g_t = \log\|\Delta_t\| - \log\|\Delta_{t-1}\|$,"
                   r"  $\Delta = (u, c)_{\rm flipped} - (u, c)_{\rm intact}$   (lag = steps since the flip)",
              "dchi": r"Growth of its belief difference:  $\Delta\chi_t = \chi_t - \chi_{t-1}$,  $\chi = d_H$(intact, "
                      "flipped marginal beliefs)   (it accumulates in every agent with memory, Bayes too; Bayes's "
                      "is bounded by its constant joint difference)",
              "upd": r"Growth of its difference in the step's belief update (update sensitivity):  $\log s_t - "
                     r"\log s_{t-1}$,  $s_t = d_H$-size of $\Delta\log m_t - \Delta\log m_{t-1}$"}
    sym = {"lrho": r"$\log\rho(J_t)$", "g": r"$g_t$", "dchi": r"$\Delta\chi_t$", "upd": r"$\log s_t - \log s_{t-1}$"}

    def xaxis(ax):
        ax.set_xscale("log")
        ax.set_xlim(8, 6e4)
        ax.set_xticks([10, 1e2, 1e3, 1e4])
        ax.set_xticklabels(["0", "100", "1k", "10k"])
        ax.minorticks_off()
        ax.set_xlabel("training episodes", fontsize=FL)
        ax.axvline(xn, color="k", lw=1, ls=":", zorder=1)
        ax.spines[["top", "right"]].set_visible(False)

    def lags_of(m):
        return m.replace("_pos", "").partition(":")[2] or "all"

    def set_ylim(ax, vals, pad=.07):
        vals = np.concatenate([np.ravel(v) for v in vals])
        vals = vals[np.isfinite(vals)]
        lo, hi = min(vals.min(), 0.), max(vals.max(), 0.)
        sp = hi - lo
        ax.set_ylim(lo - pad * sp, hi + pad * sp)

    def refs(ax, m, part):
        """the exact observers' values (all-unit mean or contrast) as horizontal lines where they fall on the axis;
        Bayes and naive left out where exactly 0 (the zero line stands for them)"""
        lo_, hi_ = ax.get_ylim()
        for a, col, ls in (("bayes", "k", "-"), ("naive", "0.55", "-"), ("adf", "k", ADF_LS)):
            v = _obs_value(O, a, m, part)
            if v is not None and not (a in ("bayes", "naive") and v == 0.) and lo_ <= v <= hi_:
                ax.axhline(v, color=col, lw=1.4, ls=ls, zorder=1.5)

    def fifths_panel(ax, run, m, title):
        z = Z[run]
        r = run_curves(z, m)
        x = z["x"]
        ok = np.isfinite(r["o"])
        for q in range(NQ):
            ax.fill_between(x[ok], r["lo"][ok, q], r["hi"][ok, q], color=qc[q], alpha=.2, lw=0, zorder=2)
            ax.plot(x[ok], r["v"][ok, q], color=qc[q], lw=1.6, zorder=3 + q)
        ax.plot(x[ok], r["o"][ok], color=COL[run], lw=1.5, ls=(0, (4, 1.5)), zorder=9)
        if m.endswith("_pos"):
            ax.set_ylim(-.02, 1.02)
            ax.set_ylabel("share of steps", fontsize=FL)
        else:
            sel = ok & (x >= 15)                                                 # the initial network may be off scale
            set_ylim(ax, [r["v"][sel], r["lo"][sel], r["hi"][sel], r["o"][sel]])
            ax.axhline(0, color="k", lw=.7, zorder=1)
            base = m.split(":")[0]
            ax.set_ylabel(f"{sym[base]} (nats / step)", fontsize=FL)
        refs(ax, m, "all")
        xaxis(ax)
        ax.set_title(title, loc="left", fontsize=TS)

    def contrast_panel(ax, m, title):
        base = m.split(":")[0]
        vals = []
        for run in RUNS:
            r = run_curves(Z[run], m)
            x = Z[run]["x"]
            ok = np.isfinite(r["c"])
            ax.fill_between(x[ok], r["clo"][ok], r["chi"][ok], color=COL[run], alpha=.25, lw=0, zorder=2)
            ax.plot(x[ok], r["c"][ok], color=COL[run], lw=2, zorder=3)
            sel = ok & (x >= 15)
            vals += [r["c"][sel], r["clo"][sel], r["chi"][sel]]
        ax.axhline(0, color="k", lw=.7, zorder=1)
        if have[base]:
            for run, dx in (("trained", 1.), ("echo", 1.15)):                     # ESN dots offset to the right
                xx, vv = _own_points(m, run=run)
                if len(xx):
                    ax.errorbar(xx * dx, vv[:, 0], yerr=[vv[:, 0] - vv[:, 1], vv[:, 2] - vv[:, 0]], fmt="o", ms=5.5,
                                color=COL[run], mec="k", mew=.7, elinewidth=1.2, capsize=0, zorder=8)
                    vals.append(vv)
            for a, mk, xo in (("bayes", "o", 7.2e4), ("adf", "D", 1.05e5)):      # the observers' own parts
                v_ = _own_points(m, agent=a)
                if v_ is not None:
                    ax.errorbar([xo], [v_[0]], yerr=[[v_[0] - v_[1]], [v_[2] - v_[0]]], fmt=mk, ms=6, color="k",
                                mfc="k" if a == "bayes" else "w", mec="k", mew=1., elinewidth=1.2, capsize=0, zorder=8)
                    vals.append(np.array(v_))
        for a in ("bayes", "naive", "adf"):
            v = _obs_value(O, a, m, "contrast")
            if v is not None:
                vals.append(np.array([v]))
        set_ylim(ax, vals)
        refs(ax, m, "contrast")
        xaxis(ax)
        ax.set_xlim(8, 1.35e5)                                  # room for the observers' own parts at the right edge
        ax.set_ylabel("fifth 5 $-$ fifth 1 (nats / step)", fontsize=FL)
        ax.set_title(title, loc="left", fontsize=TS)

    letters = iter("abcdefghijklmnopqrstuvwxyz")
    first_axes = []
    for i, (fam, panels) in enumerate(FIG_ROWS):
        for j, spec in enumerate(panels):
            if spec is None:
                lax = fig.add_subplot(gsp[i, j])
                lax.axis("off")
                continue
            ax = fig.add_subplot(gsp[i, j])
            if j == 0:
                first_axes.append(ax)
            kind, run, m = spec
            L = next(letters)
            base = m.split(":")[0].replace("_pos", "")
            lg = lags_of(m)
            lag_txt = "" if base == "lrho" else (" all lags" if lg == "all" else " lag 1 (the step after the flip)"
                                                 if lg == "1" else f" lags {lg}")
            if kind == "fifths":
                if m.endswith("_pos"):
                    t = f"{L}   {RUN_NAME[run]}: expanding steps\nshare of steps with {sym['lrho']} $> 0$"
                else:
                    t = f"{L}   {RUN_NAME[run]}{',' if lag_txt else ''}{lag_txt}\n{sym[base]} by entanglement fifth"
                fifths_panel(ax, run, m, t)
            else:
                t = f"{L}   Most $-$ least entangled fifth\n{sym[base]}{',' if lag_txt else ''}{lag_txt}: fifth 5 $-$ 1"
                contrast_panel(ax, m, t)
    for i, ((fam, _), ax) in enumerate(zip(FIG_ROWS, first_axes)):
        y1 = ax.get_position().y1
        fig.text(.012, y1 + .043, f"{i + 1}   {header[fam]}", ha="left", va="bottom", fontsize=TS + 1.5)
    # keys: fifths and all-unit means in the free slot of row 1, the observers and the right column at the top
    h1 = [Line2D([], [], color=qc[q], lw=2.5, label=lab) for q, lab in
          enumerate(("fifth 1 (least entangled)", "fifth 2", "fifth 3", "fifth 4", "fifth 5 (most entangled)"))]
    h1 += [Line2D([], [], color=COL["trained"], lw=1.5, ls=(0, (4, 1.5)), label="Trained RNN, all units"),
           Line2D([], [], color=COL["echo"], lw=1.5, ls=(0, (4, 1.5)), label="ESN, all units")]
    slot = fig.add_subplot(gsp[0, 2])
    slot.axis("off")
    slot.legend(handles=h1, loc="center", frameon=False, fontsize=FL + .5, handlelength=2.2,
                title="entanglement fifth of observation $t$\n(ranked within each step)", title_fontsize=FL + .5)
    # no naive key: naive is exactly 0 (the zero line) or undefined in every panel
    h2 = [Line2D([], [], color="k", lw=1.4, label="Bayes (exact)"),
          Line2D([], [], color="k", lw=1.4, ls=ADF_LS, label="factorized-state Bayes"),
          Line2D([], [], color="k", lw=.7, label="0 (Bayes and naive where exactly 0)"),
          Line2D([], [], color="k", lw=1, ls=":", label=f"Trained RNN passes naive ({xn:,.0f})")]
    h3 = [Patch(color=COL["trained"], alpha=.5, label="Trained RNN (band 5-95%)"),
          Patch(color=COL["echo"], alpha=.5, label="ESN"),
          Line2D([], [], color=COL["trained"], marker="o", mec="k", ls="none", ms=6,
                 label="own part of the step's observation: Trained RNN"),
          Line2D([], [], color=COL["echo"], marker="o", mec="k", ls="none", ms=6, label="ESN"),
          Line2D([], [], color="k", marker="o", ls="none", ms=6, label="Bayes"),
          Line2D([], [], color="k", marker="D", mfc="w", ls="none", ms=6, label="factorized-state Bayes")]
    fig.legend(handles=h2, loc="upper center", bbox_to_anchor=(.5, .932), ncol=4, frameon=False, fontsize=FL + 1,
               handlelength=2.6, columnspacing=1.8)
    fig.legend(handles=h3, loc="upper center", bbox_to_anchor=(.5, .91), ncol=6, frameon=False, fontsize=FL + 1,
               handlelength=1.6, columnspacing=1.4)
    fig.text(.5, .887, "Right column: own part = actual $-$ expected over the 32 possible observations at the same "
             "state and flip history (what the step's own observation adds; bars 5-95%)", ha="center", va="top",
             fontsize=FL + 1)
    # the conclusion, from the curves
    zt, xt = Z["trained"], Z["trained"]["x"]
    rl = run_curves(zt, "lrho")
    win = (rl["lo"][:, -1] > 0) & (rl["ohi"] < 0)
    k5 = _argmax(np.where(win, rl["v"][:, -1], np.nan))
    gmax = max(np.nanmax(run_curves(zt, m)["v"][xt >= 15]) for m in ("g:1-4", "g:5-20"))
    g1 = run_curves(zt, "g:1")
    g1w = np.all(g1["lo"] > 0, -1)
    ru = run_curves(zt, "upd:5-20")
    own_end = _own_points("upd:5-20", run="trained")[1][-1, 0] if have["upd"] else np.nan
    cb = _obs_value(O, "bayes", "upd:5-20", "contrast")
    lines = ["Converging on average, and no divergence at entangled observations: a flipped observation's state "
             "difference shrinks at every entanglement level and stage",
             f"(row 2: every fifth $\\leq$ {np.ceil(gmax * 1000) / 1000:+.3f} nats / step at lags 1-4 and 5-20; only the "
             f"first step after a flip grows: from 720 episodes, first at the more entangled observations,",
             f"then at every level at {_ranges(xt, g1w, False)} episodes). But the step map expands along an "
             f"eigen-direction at the most entangled observations around passing naive",
             f"(row 1: {_ranges(xt, win, False)} episodes, up to {rl['v'][k5, -1]:+.3f} while the average step contracts), "
             f"and late in training the update becomes more sensitive to the belief at entangled observations,",
             f"as in Bayes, because of the observation itself (row 4: {ru['c'][-1]:+.2f} at the end, own part "
             f"{own_end:+.2f}; Bayes {cb:+.2f})."]
    fig.text(.5, .992, "\n".join(lines), ha="center", va="top", fontsize=TS + 2, linespacing=1.4)
    n_ck = {r: len(z["x"]) for r, z in Z.items()}
    n_lr = {r: int(np.isfinite(z["lr_conv"][:, 0]).sum()) for r, z in Z.items()}
    n_cf = {r: sum(os.path.exists(cf_file(r, e)) for e in cf_raws(r)) for r in RUNS}
    n_cl = {r: sum(os.path.exists(cf_lrho_file(r, e)) for e in ti.checkpoints(r)) for r in RUNS}
    fig.text(.5, .036, f"Data: {len(rows()):,} probes (20 per test environment x 50 environments) x 30 steps; "
             f"checkpoints: Trained RNN {n_ck['trained']} ({n_lr['trained']} with $\\log\\rho$), ESN {n_ck['echo']} "
             f"({n_lr['echo']}). Unit = (probe, step $t$); fifths of the entanglement of observation $t$ (the mean of "
             "the two variables), 200 units per fifth and step.\n"
             r"$\log\rho$: Arnoldi on the step Jacobian (fp32, Krylov dimension $\leq$ 120 with restarts; checked "
             "against exact eigenvalues). Flips: each of the 5 observations at steps 1, 3, 5, 10, the network rerun "
             "from its intact state; a unit's value = the mean over its flips in the panel's lag group.\n"
             f"Own-part dots: Trained RNN {n_cf['trained']} and ESN {n_cf['echo']} checkpoints (row 1: "
             f"{n_cl['trained']} and {n_cl['echo']}, on 200 of the probes); exact observers at the right edge. Bands "
             "and bars: 5-95% over 1,000 environment resamples (pooled ratio of sums). Observer levels beyond a "
             "panel's range are not drawn.\nExact observers (float64), the same probes and flips: Bayes and naive are "
             r"translations in log coordinates ($\log\rho = g = 0$); naive's $d_H$ change and update difference are "
             "exactly 0. The initial network (0 episodes) can lie off scale.", ha="center", va="center", fontsize=FL,
             linespacing=1.55)
    path = path or fig_path("td_tt_chaos_ent.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    what = sys.argv[1:] or ["numbers"]
    if what[0] == "harvest":
        harvest(what[1], reverse="reverse" in what)
    if what[0] == "ext":
        ext_harvest(reverse="reverse" in what)
    if what[0] == "cflrho":
        cf_lrho_harvest(reverse="reverse" in what)
    if "observers" in what:
        observers_cached()
        observers_upd_cached()
    if "cf" in what:
        cf_harvest()
    if "cfupd" in what:
        cf_upd_harvest()
    if "checks" in what:
        checks()
    if "cf_check" in what:
        cf_check()
    if "figure" in what:
        figure()
    if "numbers" in what:
        numbers()
