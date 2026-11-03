"""Chaos in the belief dynamics, through training (John 11-02: "I'm also wondering about chaotic dynamics, with regards
to neural and marginal belief dynamics. Some of our behaviors, like splitting and such, point in this direction and I'm
wondering if we can explore rigorous and formally precise measures of chaos throughout learning and between optimal and
naive bayes' marginal belief dynamics.")

STATUS (11-02, revised after three adversarial checks): theory checked (check(), theory_checks()). The state exponents
of the first version (a 30-step QR from a random frame) were biased and unconverged; they are replaced by converged
estimators (M1): lyap_sweep() -> chaos/lyap/ (Trained RNN and ESN at all 114 checkpoints, the per-step log sigma_max
at every third and the last; the other four runs at every sixth and the last, 20), lam_long() -> chaos/lyap/long/
(the Trained RNN and power 1 at every checkpoint: their top exponents are nearly degenerate late in training),
adf_lyap() (chaos/lyap/adf.npz), lyap_check(), lam_recheck(), lam_frames(), bound_check() and horizon_check()
(chaos/lyap/checks.json), lam_decide() (chaos/lyap/decide/: the chaos test's decision, 64 draws per candidate).
The flips (M2) cover every checkpoint of the 6 runs (chaos/sweep/: chi valid, its QR fields superseded); the pilot
(chaos/pilot/, 6 checkpoints per network) is kept for the flips and as the record of the superseded QR.
-> main/REPLICATION/figure_bin/td_tt_chaos_sweep.png.

THEORY (exact statements; check() and theory_checks() verify each one numerically on the probes)
  Beliefs with full support are compared in Hilbert's projective metric d_H(p, q) = max_i log(p_i / q_i) - min_i log(p_i
  / q_i), the range of the log-ratio (for softmax beliefs: the range of the logit difference). Multiplying both beliefs
  by the same positive vector (a likelihood) leaves d_H unchanged, and a positive linear map never increases it
  (Birkhoff-Hopf).
  T1 Joint Bayes and naive are neutral in log coordinates (d_H). Joint Bayes adds the step's log-likelihood log p(x_t |
    r) to its log posterior over the 100 cells; naive adds its per-variable log-likelihood to each variable's log
    belief. Both are translations in log coordinates (Jacobian = identity on the quotient by constants; the likelihoods
    are strictly positive, L in [.021, .979], and the latent is static), so for any two starting beliefs and any
    observations d_H(b_t, b'_t) = d_H(b_0, b'_0). In log coordinates every finite-time and asymptotic Lyapunov exponent
    is exactly 0: neither filter is chaotic or contracting there; both are perfect integrators. Chart caveat: in
    probability charts the finite-time exponents are not 0 and their sign depends on the chart, because the posterior
    concentrates (Checks: 30 steps from the uniform start, joint Bayes +.025 in the Euclidean chart of p, -.021 in the
    Fisher-Rao chart of sqrt(p)). So a comparison with the networks' exponents (Euclidean (u, c)) must state each side's
    chart. d_H is the chart of these statements because a likelihood update is a d_H isometry and d_H is the range of
    the logit difference, the coordinate of the networks' read-out; it is still a choice, one that weights
    negligible-probability values equally.
  T2 Bayes's marginals are bounded by the conserved joint distance, but can grow from 0. m(r0) / m'(r0) is a weighted
    mean of b(r) / b'(r) over r1, so d_H(m_t, m'_t) <= d_H(b_t, b'_t) = d_H(b_0, b'_0): no marginal distance ever exceeds
    the conserved joint distance. Relative to the starting marginal distance the growth is unbounded, because the
    marginals are not a closed system: a joint b and the product of its own marginals m0 x m1 have the same marginals
    now (distance 0) and separate under the same observations, by at most d_H(b, m0 x m1) = the range of b's pointwise
    mutual information (the coupling the marginals do not show). Naive's state is its marginals, so naive is closed.
  T3 Factorized-state Bayes expands on coupling steps. ADF (factorized-state Bayes, td_analysis) keeps only the
    marginals a0, a1 and updates both simultaneously through the one-step joint pi_t ∝ a0 x a1 x p(x_t | .), a <-
    marginals of pi_t. In log coordinates its Jacobian on the quotient is [[I, A], [B, I]] with A = pi_t(r1 | r0) and B =
    pi_t(r0 | r1). Its eigenvalues are 1 +- rho_k, where rho_k are the canonical correlations of pi_t: the singular
    values of D0^-1/2 pi_t D1^-1/2 below the trivial 1, with D0, D1 the diagonal matrices of pi_t's two marginals. In the
    step's own metric (L2(D0) + L2(D1) on mean-zero log changes) a step that couples the variables (rho_1 > 0)
    stretches the co-moving direction by exactly 1 + rho_1 (at most x2) and squeezes the opposite one by 1 - rho_1;
    other metrics can show more stretch (the orthonormal zero-sum chart: up to x2.6). Each step contracts volume (det =
    prod_k (1 - rho_k^2) <= 1, < 1 whenever rho_1 > 0), and the coupling is front-loaded (mean rho_1 .60 at step 1, .020
    at step 30). So a marginal-only observer that reads each step through its own beliefs amplifies small differences
    on coupling steps: a candidate formal analogue of committing / splitting (not tested here). Joint Bayes holds the
    coupling instead (T1, T2).
MEASURES
  M1 State exponents of a network: its state (u, c) in Euclidean coordinates (2H dims), the exact tangent map of the
    LSTM step (tangent(): analytic JVP, checked against finite differences; adjoint(): its transpose, checked against
    torch.func.vjp), 500 probes (lyap_rows(): 10 per test environment), from the trained start state.
    (a) Finite-time amplification over the episode: (1/T) log sigma_1(Phi), Phi = J_T ... J_1 the propagator of steps
      1-30 (also 6-30): the largest growth rate of any infinitesimal state perturbation over the episode. Frame-free:
      block power iteration on Phi^T Phi (4 vectors, Rayleigh-Ritz) until no probe's log sigma_1 moves by 5e-4. Not a
      chaos criterion: a contracting system can amplify transiently (non-normal growth), and the value depends on the
      coordinates.
    (b) Asymptotic (conditional) Lyapunov exponents lambda_1 >= lambda_2 along the inputs of the probe's own world: an
      observation sequence sampled i.i.d. from the probe's environment and cell, QR of 2 tangent vectors from a random
      orthonormal frame, lambda_j = the mean of log |R_jj| after a burn-in that is meant to align the frame. The state is
      chaotic iff lambda_1 > 0 (two copies driven by the same observations separate); lambda_1 < 0: they converge, the
      state forgets its initial condition (the echo-state property). Unlike (a), it does not depend on a rescaling of
      the coordinates. Estimators:
      300 steps, the mean over steps 101-300 (lyap_sweep(), every run). An unaligned frame grows at a weighted mean of
        the exponents, so the estimate is biased low; 100 steps align it only where lambda_1 - lambda_2 is large. The
        Trained RNN's and power 1's top exponents are close late in training (lambda_1 - lambda_2 from 4,000 episodes:
        Trained RNN .0024-.0064, power 1 falling from .0156 to .0034), and there the 300-step estimate is biased low by
        up to .0032 (Trained RNN) and .0048 (power 1) (RESULTS), while 5 frames on the same sequences agree within
        .0009 (lam_frames()): all random frames carry the same bias, so their agreement cannot detect it. Power 1's
        300-step estimate is also biased low by .0063 at 360 episodes, without a small top gap there (lambda_1 -
        lambda_2 .0125; at 240 episodes the gap is .0085 and the bias .0007): cause not identified.
      1,500 steps, the mean over steps 501-1,500, 4 frames on the same new draw (lam_long(); the Trained RNN and power
        1 at every checkpoint, the other runs at the end): frames agree within .00025. Drawn and quoted for the
        Trained RNN and power 1 (lam_curve()); for the ESN and N = 200 / 100 / 50 the 300-step estimate (their gaps
        are large, or the bias is negligible against lambda_1: at the end the two estimators agree within .0004).
      Frame-free upper estimate (lam_long(), prop_growth()): (1/n) log sigma_1 of the propagator of steps 501-1,500
        (n = 1,000) on the same sequences. E log sigma_1(Phi_n) is subadditive in n (||AB|| <= ||A|| ||B||), so for
        stationary inputs (1/n) E log sigma_1(Phi_n) >= lambda_1 for every n and converges to it from above (Fekete;
        Furstenberg-Kesten): no frame to align. The QR estimate is biased low (an unaligned vector carries weight on
        slower directions), the bound high, so together they bracket lambda_1 up to sampling noise.
      Chaos test (lam_recheck(), lam_decide()): every probe-checkpoint whose 300-step estimate is positive, or whose
        frame-free bound is >= 0 where computed (a candidate), is re-estimated with 64 independent draws of 4,000
        steps, each with its own frame (the mean of log |R_11| over steps 1,001-4,000), and decided by the mean over
        the draws and its SEM: chaotic if mean - 2 SEM > 0 (at lambda_1 = 0 a candidate passes with probability .025),
        contracting if mean + 2 SEM < 0, otherwise undecided. The earlier rule (lam_recheck(): 8 draws of 2,000 steps,
        steps 501-2,000, each with its own frame; chaotic only if all 8 are positive, ~1/256 by chance) is kept as the
        strict count. It has little power where lambda_1 is small: the 8 draws' SD is .0044-.0055 per 1,500 steps at
        N = 200 / 100 / 50, so a probe at lambda_1 = +.005 passes with probability .20-.33 (normal approximation), and
        it missed 14 of the 16 chaotic candidates with lambda_1 below +.005 (RESULTS). The counts are lower bounds:
        the 300-step screen can miss probes with lambda_1 just above 0, and undecided candidates can be weakly chaotic
        (N = 100's 720-episode undecided probe passes in the float64 control; a checker's longer draws, steps
        1,001-8,000, also put two undecided N = 200 probes slightly above 0, +.0009 +- .0003 and +.0003 +- .0001).
      Horizon: the networks only ever run 30-step episodes. (b), its bound and tau_1 describe the state driven far
        beyond them (steps 101-300, 501-1,500, 501-2,000 or 1,001-4,000 of sampled sequences), not the trained regime.
        The extension is mild (horizon_check(), RESULTS: the state's scale and the 30-step growth on the sampled
        sequences stay near their values late in the episode), but within episodes the memory measure is tau_flip.
    (c) Time constants: tau_1 = -1 / lambda_1, the state's asymptotic decay time when driven beyond the trained horizon
      (the slowest decay of a state perturbation, in steps; from (b)); tau_flip = -1 / kappa, measured within episodes:
      kappa = the decay rate of the state difference after one flipped observation, (log ||Delta(u, c)|| at lag 20 -
      log ||Delta(u, c)|| at lag 5) / 15 per flip (steps 1, 3, 5, 10 x the 5 observations: lags up to step 30),
      averaged over the flips, then over probes and environments.
    (d) One-step expansion: log sigma_max of each step's Jacobian (power iteration, one vector; steps 6-30).
    The networks' beliefs add 18 exactly neutral directions: their read-out is cumulative (log q_t = log softmax of
    Sigma_{tau <= t} (W u_tau + b)), so the Jacobian of (u, c, cumulative logits) is block triangular with an identity
    block. Like Bayes's, their beliefs integrate exactly; (a)-(d) are the exponents of the state that feeds them.
    Superseded (the first version; the QR fields of pilot/ and sweep/): the top K = 32 local exponents by QR from a
    random frame over the 30 steps ('FTLE'). Its top value is (1/30) log ||Phi q_1|| of one random unit vector q_1, below
    (a) by about log(sqrt(2H)) / 30 (measured: RESULTS), so the bias grows with the width (and distorted the width
    comparison); its columns are neither ordered nor converged in 30 steps, so they gave no time constants.
    ADF: (a), (b) and (d) in its 18-dim log-marginal quotient (orthonormal zero-sum coordinates, exact Jacobians, exact
    SVD). Joint Bayes and naive: all exponents 0 in log coordinates (T1).
  M2 Belief susceptibility to one observation. Flip observation o at step s; chi(t, s) = d_H of the marginal beliefs
    (intact vs flipped) at t >= s, per variable, then the mean of the two variables. Predictions: naive constant in t
    (T1); joint Bayes's marginals at most the flipped observation's joint range (T2); ADF can grow (T3); networks are
    measured. Growth: the paired rise chi(s + lag, s) - chi(s, s), averaged over the flips at steps 1, 3, 5, 10 (each
    has every lag up to 20; pooling all flips per lag would mix flip steps whose chi(s, s) differ).
  M3 One-step expansion vs entanglement: per step, log sigma_max of the step Jacobian (M1 d; ADF in its zero-sum chart)
    against the probes' entanglement (mean of the two variables), r within each episode over steps 6-30, then averaged
    over episodes. One measure and one window for every agent; the ADF contrast depends on the measure (RESULTS).
  M4 (pairs: nearby marginal beliefs given the same observation, John 11-02): moved to td_tt_chaos_pairs.py and
    td_tt_chaos_combined.py (code, results, figures).
RESULTS
  Checks (check() and theory_checks(): chaos/checks.json; lyap_check(), lam_recheck(), lam_frames(), bound_check() and
  horizon_check(): chaos/lyap/checks.json; lam_decide(): chaos/lyap/decide/):
    T1 d_H conserved to 5e-14 (joint Bayes) and 5e-14 (naive) over 30 steps from random priors (400 probes); from the
      uniform start the d_H exponent is 0 (5-95% within 1.4e-10, the rounding of a 1e-6 change). Probability charts
      (1,000 probes, 30 steps from the uniform start): joint Bayes +.025 (the probes' 5-95% range -.045 to +.064, not an
      environment-resample interval) in the Euclidean chart of p (positive for 82% of probes), -.021 (probes' range -.064
      to +.007) in the Fisher-Rao chart (15%); naive -.011 (49%) and -.023 (10%). The likelihood tables lie in [.021,
      .979].
    T2 marginal d_H minus joint d_H at most -.029 (never above); a joint at step 10 and the product of its own marginals
      (their marginals 0 apart) separate to 11.2 nats on average at T, always below the PMI range (57.7 on average).
    T3 ADF Jacobian eigenvalues = 1 +- rho_k to 1.4e-8, the Jacobian vs finite differences 6.6e-9 (relative), mean rho_1
      .103 (.60 at step 1, .020 at step 30); sigma_max in the step's own metric = 1 + rho_1 to 1.6e-15; in the
      orthonormal zero-sum chart sigma_max is 1.27 (median), 1.71 (95%), 2.59 (max; above 2 at 0.8% of steps), .18 above
      1 + rho_1 (median); log det = sum_k log(1 - rho_k^2) to 1e-12, det <= 1 at every step (to rounding).
    Networks: the LSTM tangent map vs float64 finite differences 6.4e-10 (relative); adjoint() vs torch.func.vjp
      (float64) 2.5e-16; the network forward = td_tt_intervene's (max |d log q| 0). M1 (a) vs the exact SVD of the
      explicit 2,000 x 2,000 propagator (2 probes x 3 checkpoints x 2 windows): within 4e-7; tolerance 5e-4 vs 1e-7
      (125 probes): within 2.2e-6; at most 15 iterations (Trained RNN) and 23 (ESN). M1 (d) with one vector vs a 3-vector
      block (125 probes): .0004 (Trained RNN) and .0028 (ESN) lower on average, r with entanglement unchanged to .002.
      M1 (b), the 300-step estimate: a second frame and a second observation draw (500 probes, 7 checkpoints;
      lyap_check) moved the means by at most .0006 (Trained RNN end -.0129 / -.0133 / -.0130), per-probe differences SD
      .001-.005 (N = 50 across draws .019); 4 more frames on the same sequences (lam_frames) spread .0006 / .0009 / .0008
      / .0006 at the Trained RNN's 1,680 / 4,200 / 12,000 / 39,996 episodes and .0007 at power 1's end (end: -.0128 to
      -.0134); over steps 301-1,000 (burn-in 300) vs 101-300 (125 probes): -.0452 vs -.0466 (training 1,680), -.0110 vs
      -.0128 (end), -.363 vs -.364 (ESN), no probe above 0. These agreements are within the frames' shared bias, which
      lam_long measures (State). The frame-free bound (prop_growth) vs the exact SVD of the explicit 1,000-step
      propagator (N = 50 and N = 200 at the end, 5 probes each; bound_check): within 3e-8; its power iteration
      converged to 2e-4 in log sigma_1 in 3-6 iterations at 39 of its 40 checkpoints (N = 50 at the end: 1e-2 after
      20, i.e. 1e-5 per step). The chaos decision in float64 (lam_decide, N = 200 / 100 / 50: the same draws and
      frames, step and tangent maps in float64): the same decisions for 162 of the 163 candidates (the exception: an
      N = 100 probe at 720 episodes, t 1.74 in float32, 2.00 in float64; its draws spread from -.024 to +.020); the
      candidates' 64-draw means change by .00000 +- .00001 / -.00004 +- .00005 / +.00003 +- .00003 on average. Single
      draws can differ (48-89% agree within 1e-4): the two precisions' trajectories separate on expanding stretches,
      also of some contracting candidates; on average the means do not move (per candidate by up to .0012, 2.9 SEM;
      the one decision that changes is the N = 100 probe above). The superseded QR: CholeskyQR2 vs torch.linalg.qr
      6e-7; its two random frames agree (training 1,680, top-5 means -.130 / -.141 / -.143 / -.147 / -.154 vs -.131 /
      -.139 / -.147 / -.158 / -.156; per probe within .076) because both carry the frame bias, so the agreement cannot
      detect it.
  State (M1, M3; 500 probes; nats per step; [5-95%] over the 1,000 environment resamples, for lam_long's widened to
  the union of its 4 frames; figure a-d, h):
    Chaos (b): none at N = 1,000; N = 200, 100 and 50 are weakly chaotic at single probes in mid-training (figure a,
      dots). Every run's mean lambda_1 is < 0 at every checkpoint (N = 200 / 100 / 50: at most -.062 / -.066 / -.060).
      Candidates (186, each decided by its 64 draws, M1 b): the Trained RNN has 18 positive 300-step estimates among
      its 57,000 probe-checkpoints (14 checkpoints, 60-32,000 episodes, max +.030) and one probe whose frame-free bound
      is >= 0 at 5 checkpoints (16,000-25,600 episodes, up to +.0006); all 23 are contracting (64-draw means at most
      -.0020, the near-neutral probe's -.0030 to -.0020; the 18's 8 earlier draws average at most -.0095, and none of
      the 23 has a positive draw among its 8 earlier ones); power 1 and the ESN have none. Frame-free: the bound is
      below 0 for every other probe at every checkpoint where it was computed (Trained RNN 25 checkpoints, 80-39,996
      episodes; power 1 11, 4,200-39,996; ESN and N = 200 / 100 / 50 at the end), so those states are not chaotic,
      whatever the frame.
      N = 200: at least 1 of 43 candidates is chaotic, probe 7402 at 720 episodes: lambda_1 +.0036 +- .0004 (t = 8.2; 57 of
      its 64 draws positive, 6 of its 8 earlier ones); 3 are undecided (means -.0008 to +.0006), 39 contracting.
      N = 100: 19 of 49 chaotic at 360-3,000 episodes (9 at 720, 4 at 1,200), means +.0014 to +.0167. N = 50: 19 of
      71 at 720-7,800 episodes (5 at 1,200, 5 at 2,160; one probe still at 5,400 and 7,800, +.0040 / +.0041), means
      +.0013 to +.0245. The chaotic candidates' t = mean / SEM is 2.9-36.0 (N = 100) and 3.5-49.5 (N = 50); mean - 3
      SEM > 0 for 1 / 18 / 19 (N = 200 / 100 / 50). Undecided: 4 (N = 100) and 3 (N = 50), all |mean| <= .0023. At
      lambda_1 = 0 a candidate passes with probability .025: even if every chaotic and undecided candidate were at
      0, .10 / .57 / .55 would pass by chance. The strict rule (all 8 earlier draws positive) counts 0 / 9 / 10 for
      N = 200 / 100 / 50 (0 for the Trained RNN): every candidate it passed is chaotic by the decision, but it found
      only 19 of the 39 chaotic candidates, 2 of the 16 with lambda_1 below +.005. These counts are lower bounds (the
      300-step screen can miss probes just above 0, and undecided candidates can be weakly chaotic: M1 (b)).
    Late lambda_1 (b): the 300-step estimate is biased low where the top exponents are close: lam_long's exceeds it by
      +.0001 to +.0032 for the Trained RNN (median +.0022; +.0020 to +.0032 from 4,000 episodes, where lambda_1 -
      lambda_2 is .0024-.0064) and by -.0001 to +.0063 for power 1 (+.0012 to +.0048 from 4,000; largest at 360
      episodes, where lambda_1 - lambda_2 is .0125, not small: cause not identified), while lam_long's 4 frames agree
      within .00025 and the frame-free bound lies .0006-.0017 above it. At the end: Trained RNN lambda_1 -.0106
      [-.0109, -.0103] (frames -.0106 to -.0106), bound -.0090 [-.0093, -.0087], so tau_1 lies between 94 and 111 steps
      (the 300-step estimate said 77 [75.5, 79.3]); power 1 -.0114 [-.0119, -.0108], bound -.0102: 88-98 steps
      (300-step: 62). The ESN and N = 200 / 100 / 50 at the end: the two estimators agree within .0004 (frames within
      .00003).
    Horizon (horizon_check, 500 probes): on the sampled sequences the state stays near its late-episode regime.
      Trained RNN end: mean |c| .307 after 30 episode steps vs .335 / .343 after 100 / 300 sampled steps (|u| .076 vs
      .078 / .079); the frame-free 30-step growth is +.051 over steps 6-30 of the episodes vs +.055 / +.059 / +.060
      over steps 31-60 / 101-130 / 271-300 of the sampled sequences (every probe positive). At 1,680 episodes: |c| .363
      vs .375 / .374; growth +.010 (steps 6-30 of the episodes) vs -.001 / -.000 / +.001. ESN end: unchanged (|c|
      .181 / .181 / .180; growth -.319 vs -.324). So lambda_1 and tau_1 extrapolate the late-episode state mildly,
      but they are measured beyond the trained horizon, not within it.
    Trained RNN: lambda_1 -.43 (initial, = the ESN's) -> -.075 (20 episodes) -> -.018 (80) -> -.055 (880, where it
      passes naive at 835) -> -.045 (1,680) -> -.018 (4,200) -> -.0106 at the end (-.0105 at 39,200, its value
      closest to 0). tau_1 = -1/lambda_1, the decay time of its state driven beyond the 30 trained steps: 2.3
      (initial), 57 (80), 18 (880), 22 (1,680), 55 (4,200), 94-111 at the end. It jumps within 100 episodes, is
      shortest at 880 episodes (18.2), as the network passes naive, and lengthens after. tau_flip (within episodes) is
      shortest earlier (7.2 at 340 episodes) and lengthens from about 500 episodes: 2.2 -> 33 (80) -> 7.6 (480) -> 8.9
      (880) -> 10.0 (1,680) -> 26 at the end (kappa -.038). ESN: lambda_1 -.431 -> -.362 [-.368, -.355], tau_1 2.3 ->
      2.8, tau_flip 2.2 -> 2.4: its reservoir stays strongly contracting.
    Width: at the end lambda_1 = -.0106 / -.0114 / -.062 / -.136 / -.202 (tau_1 94 / 88 / 16 / 7.4 / 4.9 steps;
      frame-free bounds -.0090 / -.0102 / -.060 / -.133 / -.200) for N = 1,000 / 1,000 power 1 / 200 / 100 / 50.
      N = 50 and 100 come closest to 0 at 240 episodes (-.060 / -.066) and end more contracting (-.202 / -.136; minima
      -.211 at 31,200 / -.154 at 7,800): the reversal. N = 200 instead dips (-.125 at 5,400) and recovers to -.062 at
      the end, its value closest to 0: the shape of the N = 1,000 runs (a mid-training dip, then closest to 0 at the
      end: Trained RNN -.055 at 880, power 1 -.079 at 2,160), later and deeper.
    Finite time (a): the Trained RNN's state grows along some direction within an episode although it contracts
      asymptotically. (1/T) log sigma_1(Phi) is -.389 at the start, positive at 60 / 80 episodes (+.004 [+.001, +.007]
      / +.012 [+.010, +.013]), +.002 [-.000, +.004] at 100, -.036 at 480, positive at every checkpoint from 2,800
      episodes (+.001; +.003 [+.001, +.005] at 3,000) and +.016 [+.015, +.018] at the end (positive for 90% of probes;
      steps 6-30 +.051 [+.049, +.053], 100%): a transient growth of about x1.6 over the 30 steps (x3.6 over steps
      6-30); one step can stretch some direction by e^2 (mean log sigma_max(J_t) +2.07). ESN: -.389 -> -.324, never
      positive. At the end: power 1 +.007, N = 200 +.002, N = 100 -.049, N = 50 -.114. Factorized-state Bayes: +.073
      [+.070, +.075] (every probe positive; steps 6-30 +.074), but its asymptotic exponent is +.0008 [+.0007, +.0009]
      (lambda_2 +.0004; over steps 101-300) because its coupling fades as its marginals concentrate (mean rho_1 .59 at
      step 1, .022 at 30, .002 at 100, .000 at 300): expansive transiently, neutral in the limit.
    One-step expansion vs entanglement (M3, d): r(log sigma_max(J_t), entanglement_t) within episodes, steps 6-30, lies
      between -.068 and +.089 for the Trained RNN through training (highest at 180 episodes; -.068 [-.092, -.043] at the
      end) and between -.070 and -.005 for the ESN (-.067 [-.116, -.018] at the end), against +.35 for factorized-state
      Bayes with the same measure and window. The contrast depends on the measure: ADF's log(1 + rho_1) (its stretch in
      its own metric) gives +.27 (+.26 on the 1,000 flip probes, +.13 over steps 2-30), its QR local exponent -.02.
    Like Bayes's beliefs, the networks' beliefs carry 18 exactly neutral directions whatever the state does: the
      cumulative read-out integrates its increments exactly (M1). The first version's "the Trained RNN's state contracts
      ever less (toward the neutral integrators of T1)" mixed the state with the beliefs; the statements above replace
      it.
  Flips (M2; 1,000 probes; rises over the flips at steps 1, 3, 5, 10; figure e-g):
    naive: chi 1.778, rise 0 at every lag (T1). Bayes's marginals: chi(s, s) 3.03 (2.68 at s = 1 to 3.17 at s = 25),
      rise +.06 / +.10 / +.15 / +.19 / +.24 at lags 1 / 2 / 5 / 10 / 20, below the flipped observation's joint range
      6.35 (T2). ADF: chi(s, s) 3.18, rise +.25 / +.49 / +1.10 / +1.87 / +3.19 (T3).
    Trained RNN: chi(s, s) .02 (initial) -> .67 (320) -> 1.20 (880) -> 1.44 (1,680) -> 1.92 (4,200) -> 2.35 (end); the
      rise at lag 20 is +1.65 (320), +1.52 (880), +1.30 (1,680), +1.01 (4,200) and +.02 at the end (lags 1-10 +.05 /
      +.04 / -.03 / -.05): while it learns a flip keeps moving its beliefs for 20 steps, at the end it does not, although
      the flip's state difference persists (tau_flip 26 steps).
    ESN: chi(s, s) 1.42 at the end, rise +.68 / +1.02 / +1.34 / +1.40 / +1.40: a contracting reservoir whose remembered
      flip keeps changing the integrated read-out for a few steps (its state difference decays at kappa = -.41 per step,
      so the rise saturates by lag 10).
  Superseded (the first version's state exponents: 30-step QR from a random frame; pilot/ and the sweep/ QR fields):
    The top 'FTLE' was the growth of one random vector, below (a) by +.088 (initial), +.131 (Trained RNN at 320 and 880),
      +.116 (Trained RNN end), +.113 (power 1), +.108 / +.105 / +.082 (N = 200 / 100 / 50) and +.097 (ESN end), so its
      "-.098 at the end" and its "contracts within episodes" were frame artifacts, and its width comparison was
      distorted (the same order, compressed gaps: -.197 / -.154 / -.106 / -.105 / -.100 vs the frame-free -.114 / -.049
      / +.002 / +.007 / +.016 for N = 50 / 100 / 200 / 1,000 power 1 / 1,000); its columns gave no time constants
      (-1/-.098 = 10 steps at the end vs tau_1 94-111, tau_flip 26). Its "no positive exponent at
      any probe over steps 6-30" was false (a rounded mean count): positive late QR exponents at 32 / 6 / 8 / 6 / 2 of
      the 1,000 probes at training 320 / 880 / 1,680 / 4,200 / 39,996 (max +.108 / +.054 / +.030 / +.032 / +.019), none
      for the initial network or the ESN. ADF's QR exponents depend on the frame (top +.022 to +.028, 2nd +.015 to +.020
      over 4 frames; the sum of the 18, -.062, does not). The pooled chi lag profiles mixed flip steps; the paired rises
      above replace them.
    The second version's late lambda_1 (300 steps, the first 100 discarded: Trained RNN end -.0129, tau_1 77; power 1
      -.016, tau_1 62) was biased low (Late lambda_1 above). Its recheck of the positive estimates (one new draw each:
      10 / 19 / 25 "stay positive over 1,000 steps" at N = 200 / 100 / 50, max +.004 / +.023 / +.035;
      lam1_positive_recheck_* in chaos/lyap/checks.json) rested on a single noisy redraw (across-draw SD .004-.006 per
      1,500 steps); the 64-draw decision above replaces it. Its "the small networks come closest to 0 at 240 episodes
      and then contract more" was false for N = 200 (closest to 0 at the end).
    The third version decided chaos by the strict rule alone (all 8 draws of 2,000 steps positive). With power .20-.33
      at lambda_1 = +.005 it counted N = 200 as 0 of 43 and concluded that N = 200 is never chaotic (false: probe 7402
      at 720 episodes, +.0036 +- .0004), and it undercounted N = 100 / 50 (9 / 10 vs 19 / 19, Chaos above). Its
      counts are kept as the strict count.
DATA
  Probes (analysis/probes.npz): a fixed subset of 20 probes per environment (seed 20261104) = 1,000 episodes x 30 steps
  (flips, M2); its first 10 per environment = 500 episodes (state exponents, M1, M3); sampled sequences per probe from
  its environment and cell: 300 steps (seed 20261108; frame seed 20261109; lam_frames: 20261111-14 on the same
  sequences), lam_long 1,500 steps (seed 20261110; frames 20261111-14), lam_recheck 8 x 2,000 steps (draw d: seed
  20261120 + d, frame 20261140 + d), lam_decide 64 x 4,000 steps per candidate (its draws from numpy's generator
  seeded [20261160, raw checkpoint, probe row], its frames from [20261161, raw, row]). Flips at steps 1, 3, 5, 10, 15,
  20, 25, all 5 observations. Checkpoints of the seed-20261029 runs (trained, echo, trained_p1 = N = 1,000 with loss
  power 1, trained_N200, trained_N100, trained_N50); pilot at training 10 / 320 / 880 / 1,680 / 4,200 / 39,996.
  Caches: through_training/chaos/ (pilot/, sweep/, lyap/, lyap/long/, lyap/decide/, checks.json, lyap/checks.json).

Run from the repository root:
    CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
        /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_chaos.py <command>
    commands: check, theory_checks, pilot, sweep (flips), adf_lyap, lyap_sweep (STOP file chaos/lyap/STOP), lyap_check,
    lam_recheck, lam_decide, lam_decide_float64, lam_long (STOP file chaos/lyap/STOP), lam_frames, bound_check,
    horizon_check, sweep_figure, numbers (default)"""
import os
import sys
import json
import time
import numpy as np
import torch
from scipy.special import logsumexp

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_intervene as ti                                                                  # noqa: E402

T, R, NV, NO = ti.T, ti.R, ti.NV, ti.NO
OUT = os.path.join(gap.TT, "chaos")
SUB_SEED, FRAME_SEED, CHECK_SEED = 20261104, 20261105, 20261106
PER_ENV_SUB = 20
K_LYAP = 32
FLIP_S = (1, 3, 5, 10, 15, 20, 25)
PILOT_RAW = (0, 400, 1100, 2100, 5250, 49995)
RUNS = ("trained", "echo")


# ═══════════════════════════════════════════════════════════════════
# Probes and the Bayes-side observers (float64, CPU)
# ═══════════════════════════════════════════════════════════════════

def subset():
    """the fixed probe subset: PER_ENV_SUB rows per environment (sorted)"""
    P = ti.probes()
    rng = np.random.default_rng(SUB_SEED)
    rows = [rng.choice(np.flatnonzero(P["env"] == e), PER_ENV_SUB, replace=False) for e in range(gap.N_ENV)]
    return np.sort(np.concatenate(rows))


def tables():
    """L (50, 5, R, R) = p(x_o = 1 | r0, r1) per environment; ent (10,000, 30, 2)"""
    Z = np.load(os.path.join(ti.AN, "probes.npz"))
    return Z["L"].astype(np.float64), Z["ent"].astype(np.float64)


def osc(a, axis=-1):
    return a.max(axis) - a.min(axis)


def step_loglik(obs, L):
    """(n, T, R, R) log p(x_t | r0, r1) for obs (n, T, 5) and per-episode tables L (n, 5, R, R)"""
    o = obs.astype(np.float64)
    return np.einsum("nti,nijk->ntjk", o, np.log(L)) + np.einsum("nti,nijk->ntjk", 1 - o, np.log1p(-L))


def marg_log(u):
    """unnormalised log marginals (..., 2, R) of joint log beliefs u (..., R, R)"""
    return np.stack([logsumexp(u, -1), logsumexp(u, -2)], -2)


def adf_run(ll, a0=None):
    """factorized-state Bayes (td_analysis's ADF) on per-step log-likelihoods ll (n, S, R, R) from marginals a0 (n, 2, R)
    (uniform if None): log marginals (n, S, 2, R) after each step"""
    n, S = ll.shape[:2]
    la = np.full((n, 2, R), -np.log(R)) if a0 is None else np.log(a0)
    out = np.zeros((n, S, 2, R))
    for t in range(S):
        j = la[:, 0, :, None] + la[:, 1, None, :] + ll[:, t]
        j -= logsumexp(j.reshape(n, -1), -1)[:, None, None]
        la = marg_log(j)
        out[:, t] = la
    return out


def _zero_sum_basis():
    """(R, R - 1) orthonormal basis of the zero-sum subspace"""
    return np.linalg.svd(np.eye(R) - 1. / R)[0][:, :R - 1]


def adf_jacobians(ll, la_prev):
    """the exact quotient Jacobians (n, S, 2R - 2, 2R - 2) of ADF's log-marginal update, and the canonical correlations
    rho (n, S, R - 1) of the one-step joint pi_t ∝ a0 x a1 x p(x_t | .); la_prev (n, S, 2, R) = log marginals before
    each step. Conditionals and M in log space (long sequences drive marginal values below float64's range)"""
    n, S = ll.shape[:2]
    j = la_prev[:, :, 0, :, None] + la_prev[:, :, 1, None, :] + ll
    j -= logsumexp(j.reshape(n, S, -1), -1)[..., None, None]
    l0, l1 = logsumexp(j, -1), logsumexp(j, -2)                                # log marginals of pi
    A = np.exp(j - l0[..., :, None])                                           # pi(r1 | r0), rows r0
    B = np.swapaxes(np.exp(j - l1[..., None, :]), -1, -2)                      # pi(r0 | r1), rows r1
    E = _zero_sum_basis()
    I = np.eye(R - 1)
    EAE = np.einsum("ia,nsij,jb->nsab", E, A, E)
    EBE = np.einsum("ia,nsij,jb->nsab", E, B, E)
    J = np.zeros((n, S, 2 * R - 2, 2 * R - 2))
    J[..., :R - 1, :R - 1] = I
    J[..., R - 1:, R - 1:] = I
    J[..., :R - 1, R - 1:] = EAE
    J[..., R - 1:, :R - 1] = EBE
    M = np.exp(j - (l0[..., :, None] + l1[..., None, :]) / 2)                  # D0^-1/2 pi D1^-1/2
    sv = np.linalg.svd(M, compute_uv=False)                                    # (n, S, R), sv[..., 0] = 1
    return J, sv[..., 1:]


def qr_lyap(J, k=None, seed=FRAME_SEED):
    """local exponents (n, S, k) by QR along the Jacobians J (n, S, d, d) from a fixed random frame"""
    n, S, d, _ = J.shape
    k = d if k is None else k
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, k)))[0]
    Q = np.broadcast_to(Q, (n, d, k)).copy()
    loc = np.zeros((n, S, k))
    for t in range(S):
        Q, Rm = np.linalg.qr(J[:, t] @ Q)
        loc[:, t] = np.log(np.abs(np.diagonal(Rm, axis1=-2, axis2=-1)))
    return loc


def bayes_side(rows):
    """chi (n, len(FLIP_S), 5, T, 2) of joint Bayes's marginals, naive and ADF (NaN before the flip), ADF's local
    exponents (n, T, 18) and log(1 + rho_1) (n, T), the joint flip ranges osc(logit L) (n, len(FLIP_S), 5)"""
    P = ti.probes()
    L, ent = tables()
    obs, env = P["obs"][rows], P["env"][rows]
    Le = L[env]                                                                 # (n, 5, R, R)
    n = len(rows)
    ll = step_loglik(obs, Le)
    u = np.cumsum(ll, 1)                                                        # joint log posterior (unnormalised)
    lm = marg_log(u)                                                            # (n, T, 2, R)
    logit = np.log(Le) - np.log1p(-Le)                                          # (n, 5, R, R)
    Lb = np.stack([Le.mean(3), Le.mean(2)], 2)                                  # naive tables (n, 5, 2, R)
    logit_n = np.log(Lb) - np.log1p(-Lb)
    la = adf_run(ll)                                                            # (n, T, 2, R)
    nS = len(FLIP_S)
    chi = {k: np.full((n, nS, NO, T, NV), np.nan) for k in ("bayes", "naive", "adf")}
    jrange = np.zeros((n, nS, NO))
    for a, s in enumerate(FLIP_S):
        x = obs[:, s - 1].astype(np.float64)                                    # (n, 5)
        sign = 1. - 2. * x
        for o in range(NO):
            d = sign[:, o, None, None] * logit[:, o]                            # (n, R, R) change of log p(x_s | r)
            jrange[:, a, o] = osc(d.reshape(n, -1))
            lm2 = marg_log(u[:, s - 1:] + d[:, None])
            chi["bayes"][:, a, o, s - 1:] = osc(lm2 - lm[:, s - 1:])
            chi["naive"][:, a, o, s - 1:] = osc(logit_n[:, o])[:, None, :]
            ll2 = ll[:, s - 1:].copy()
            ll2[:, 0] += d
            a0 = None if s == 1 else np.exp(la[:, s - 2])
            la2 = adf_run(ll2, a0)
            chi["adf"][:, a, o, s - 1:] = osc(la2 - la[:, s - 1:])
    la_prev = np.concatenate([np.full((n, 1, 2, R), -np.log(R)), la[:, :-1]], 1)
    J, rho = adf_jacobians(ll, la_prev)
    loc = qr_lyap(J)
    return dict(rows=rows, chi_bayes=chi["bayes"], chi_naive=chi["naive"], chi_adf=chi["adf"], jrange=jrange,
                adf_local=loc, adf_rho=rho, ent=ent[rows].mean(-1), right_adf=None)


# ═══════════════════════════════════════════════════════════════════
# Networks: exact tangent map of the LSTM step, QR Lyapunov spectrum, observation flips (GPU 0)
# ═══════════════════════════════════════════════════════════════════

def lstm_step(xin, h, c, Whh):
    pre = xin + h @ Whh.T
    i, f, g, o = pre.chunk(4, -1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    c2 = f * c + i * g
    tc = c2.tanh()
    return o * tc, c2, (i, f, g, o, tc)


def tangent(dh, dc, c_prev, gts, Whh):
    """the LSTM step's JVP for K tangent vectors: dh, dc (b, K, H), c_prev (b, H), gts the step's (i, f, g, o, tanh c)"""
    i, f, g, o, tc = (a[:, None] for a in gts)
    di, df, dg, do = (dh @ Whh.T).chunk(4, -1)
    dc2 = df * (f * (1 - f)) * c_prev[:, None] + f * dc + di * (i * (1 - i)) * g + i * dg * (1 - g * g)
    dh2 = do * (o * (1 - o)) * tc + o * (1 - tc * tc) * dc2
    return dh2, dc2


def cholqr2(Y):
    """QR of a batch of tall matrices Y (b, d, k) by CholeskyQR2 (Gram matrices and Cholesky in float64): Q (b, d, k),
    the diagonal of R (b, k) (positive). 7x faster than torch.linalg.qr here (the pilot's QR); the local exponents agree
    with torch.linalg.qr's to ~1e-6 (lyap_check: pilot_cholqr2_vs_torch_qr_local_maxabs)"""
    G = Y.transpose(1, 2).double() @ Y.double()
    R1 = torch.linalg.cholesky(G).transpose(1, 2)
    Q = torch.linalg.solve_triangular(R1.float(), Y, upper=True, left=False)
    R2 = torch.linalg.cholesky(Q.transpose(1, 2).double() @ Q.double()).transpose(1, 2)
    Q = torch.linalg.solve_triangular(R2.float(), Q, upper=True, left=False)
    return Q, (R2.diagonal(dim1=-2, dim2=-1) * R1.diagonal(dim1=-2, dim2=-1))


@torch.no_grad()
def network(run, raw, rows, dev, k=K_LYAP, chunk=500, frame_seed=FRAME_SEED, qr="chol"):
    """local Lyapunov exponents (n, T, k) of (u, c), chi (n, len(FLIP_S), 5, T, 2) of the network's marginal beliefs,
    the intact log beliefs (n, T, 2, R)"""
    w = ti.load(run, raw)
    p = {kk: v.to(dev, torch.float32) for kk, v in w.items()}
    H = p["LSTM.weight_hh_l0"].shape[1]
    Whh, Wro, bro = p["LSTM.weight_hh_l0"], p["classifier_readout.weight"], p["classifier_readout.bias"]
    P = ti.probes()
    table = ti._table(p, P["Z"], dev)
    obs, env = P["obs"][rows], P["env"][rows]
    code = torch.as_tensor((obs * (1 << np.arange(NO))).sum(-1).astype(np.int64), device=dev)
    envt = torch.as_tensor(env.astype(np.int64), device=dev)
    n = len(rows)
    g = torch.Generator().manual_seed(frame_seed)
    Q0 = torch.linalg.qr(torch.randn(2 * H, k, generator=g, dtype=torch.float64))[0].float().to(dev)
    loc = np.zeros((n, T, k), np.float32)
    chi = np.full((n, len(FLIP_S), NO, T, NV), np.nan, np.float32)
    logq = np.zeros((n, T, NV, R), np.float32)
    for b0 in range(0, n, chunk):
        sl = slice(b0, min(n, b0 + chunk))
        b = sl.stop - sl.start
        h = p["STM"].reshape(1, H).expand(b, H).contiguous()
        c = p["LTM"].reshape(1, H).expand(b, H).contiguous()
        Q = Q0.T[None].expand(b, k, 2 * H).contiguous()                          # (b, k, 2H) rows = tangent vectors
        hs, cs, lg = [h], [c], []
        lc = torch.zeros(b, T, k, device=dev, dtype=torch.float64)
        for t in range(T):
            xin = table[envt[sl], code[sl, t]]
            h2, c2, gts = lstm_step(xin, h, c, Whh)
            dh2, dc2 = tangent(Q[..., :H], Q[..., H:], c, gts, Whh)
            Y = torch.cat([dh2, dc2], -1).transpose(1, 2)                         # (b, 2H, k)
            if qr == "chol":
                Qm, dg = cholqr2(Y)
            else:
                Qm, Rm = torch.linalg.qr(Y)
                dg = torch.diagonal(Rm, dim1=-2, dim2=-1).abs()
            lc[:, t] = torch.log(dg)
            Q = Qm.transpose(1, 2).contiguous()
            h, c = h2, c2
            hs.append(h)
            cs.append(c)
            lg.append(torch.nn.functional.linear(h, Wro, bro))
        loc[sl] = lc.float().cpu().numpy()
        lg = torch.stack(lg, 1).double()                                          # (b, T, 20)
        logq[sl] = torch.log_softmax(lg.cumsum(1).reshape(b, T, NV, R), -1).float().cpu().numpy()
        for a, s in enumerate(FLIP_S):                                            # flip observation o at step s
            hb = hs[s - 1].repeat_interleave(NO, 0)
            cb = cs[s - 1].repeat_interleave(NO, 0)
            eb = envt[sl].repeat_interleave(NO)
            flipbit = torch.as_tensor(1 << np.arange(NO), device=dev).repeat(b)
            d = torch.zeros(b * NO, T - s + 1, NV * R, device=dev, dtype=torch.float64)
            for j, t in enumerate(range(s - 1, T)):
                cd = code[sl, t].repeat_interleave(NO)
                if t == s - 1:
                    cd = cd ^ flipbit
                hb, cb, _ = lstm_step(table[eb, cd], hb, cb, Whh)
                d[:, j] = torch.nn.functional.linear(hb, Wro, bro).double() - lg[:, t].repeat_interleave(NO, 0)
            D = d.cumsum(1).reshape(b, NO, T - s + 1, NV, R)
            chi[sl, a, :, s - 1:] = (D.amax(-1) - D.amin(-1)).float().cpu().numpy()
    return dict(local=loc, chi=chi, logq=logq, H=H)


# ═══════════════════════════════════════════════════════════════════
# Checks of the theory and of the tangent map
# ═══════════════════════════════════════════════════════════════════

def check(dev=None):
    """numerical checks of T1-T3 and of the network tangent map; prints and saves through_training/chaos/checks.json"""
    dev = ti._device(dev)
    out = {}
    P = ti.probes()
    L, _ = tables()
    rng = np.random.default_rng(CHECK_SEED)
    rows = np.sort(rng.choice(len(P["env"]), 400, replace=False))
    Le = L[P["env"][rows]]
    ll = step_loglik(P["obs"][rows], Le)
    n = len(rows)
    # T1: joint Bayes and naive conserve d_H between two random priors
    u0, u1 = rng.normal(0, 1, (n, R, R)), rng.normal(0, 1, (n, R, R))
    ua, ub = u0[:, None] + np.cumsum(ll, 1), u1[:, None] + np.cumsum(ll, 1)
    dj = osc((ua - ub).reshape(n, T, -1))
    out["T1_joint_max_change"] = float(np.abs(dj - osc((u0 - u1).reshape(n, -1))[:, None]).max())
    Lb = np.stack([Le.mean(3), Le.mean(2)], 2)
    o = P["obs"][rows].astype(np.float64)
    lpx = np.einsum("nti,nikr->ntkr", o, np.log(Lb)) + np.einsum("nti,nikr->ntkr", 1 - o, np.log1p(-Lb))
    v0, v1 = rng.normal(0, 1, (n, 2, R)), rng.normal(0, 1, (n, 2, R))
    dn = osc(v0[:, None] + np.cumsum(lpx, 1) - v1[:, None] - np.cumsum(lpx, 1))
    out["T1_naive_max_change"] = float(np.abs(dn - osc(v0 - v1)[:, None]).max())
    # T2: marginal distance <= joint distance; joint vs product of its marginals: <= osc(PMI)
    dm = osc(marg_log(ua) - marg_log(ub))                                         # (n, T, 2)
    out["T2_marginal_over_joint_max"] = float((dm - dj[..., None]).max())
    u = np.cumsum(ll, 1)
    t0 = 10
    lm = marg_log(u[:, t0 - 1])
    lm -= logsumexp(lm, -1)[..., None]
    lj = u[:, t0 - 1] - logsumexp(u[:, t0 - 1].reshape(n, -1), -1)[:, None, None]
    pmi = lj - lm[:, 0, :, None] - lm[:, 1, None, :]
    prod = lm[:, 0, :, None] + lm[:, 1, None, :]
    fa = marg_log(lj[:, None] + np.cumsum(ll[:, t0:], 1))
    fb = marg_log(prod[:, None] + np.cumsum(ll[:, t0:], 1))
    steer = osc(fa - fb)                                                          # (n, T - t0, 2)
    out["T2_steer_minus_pmi_range_max"] = float((steer - osc(pmi.reshape(n, -1))[:, None, None]).max())
    out["T2_steer_mean_at_T"] = float(steer[:, -1].mean())
    out["T2_pmi_range_mean"] = float(osc(pmi.reshape(n, -1)).mean())
    # T3: ADF Jacobian eigenvalues = 1 +- rho_k; Jacobian vs finite differences
    la = adf_run(ll)
    la_prev = np.concatenate([np.full((n, 1, 2, R), -np.log(R)), la[:, :-1]], 1)
    J, rho = adf_jacobians(ll, la_prev)
    ev = np.sort(np.linalg.eigvals(J[:50, ::5].reshape(-1, 2 * R - 2, 2 * R - 2)).real, -1)
    pred = np.sort(np.concatenate([1 + rho[:50, ::5], 1 - rho[:50, ::5]], -1).reshape(-1, 2 * R - 2), -1)
    out["T3_eig_vs_1pm_rho_max"] = float(np.abs(ev - pred).max())
    E = _zero_sum_basis()
    eps, errs = 1e-6, []
    for i in range(5):
        for t in (0, 4, 14, 29):
            d = rng.normal(0, 1, 2 * R - 2)
            dl = np.stack([E @ d[:R - 1], E @ d[R - 1:]])
            fp = adf_run(ll[i:i + 1, t:t + 1], np.exp(la_prev[i:i + 1, t] + eps * dl))[0, 0]
            fm = adf_run(ll[i:i + 1, t:t + 1], np.exp(la_prev[i:i + 1, t] - eps * dl))[0, 0]
            num = np.concatenate([E.T @ (fp[0] - fm[0]), E.T @ (fp[1] - fm[1])]) / (2 * eps)
            errs.append(np.abs(num - J[i, t] @ d).max() / np.abs(num).max())
    out["T3_jacobian_vs_finite_diff_rel_max"] = float(max(errs))
    out["T3_rho1_mean"] = float(rho[..., 0].mean())
    # tangent map of the LSTM step vs finite differences (float64), trained network at training 1,680
    w = ti.load("trained", 2100)
    p = {k: v.to(dev, torch.float64) for k, v in w.items()}
    H = p["LSTM.weight_hh_l0"].shape[1]
    Whh = p["LSTM.weight_hh_l0"]
    table = ti._table({k: v.float() for k, v in p.items()}, P["Z"], dev).double()
    code = (P["obs"][rows[:20]] * (1 << np.arange(NO))).sum(-1)
    envt = torch.as_tensor(P["env"][rows[:20]], device=dev)
    h = p["STM"].reshape(1, H).expand(20, H).contiguous()
    c = p["LTM"].reshape(1, H).expand(20, H).contiguous()
    g = torch.Generator().manual_seed(CHECK_SEED)
    errs = []
    for t in range(T):
        xin = table[envt, torch.as_tensor(code[:, t], device=dev)]
        h2, c2, gts = lstm_step(xin, h, c, Whh)
        if t in (0, 5, 15, 29):
            dh = torch.randn(20, 3, H, generator=g, dtype=torch.float64).to(dev)
            dc = torch.randn(20, 3, H, generator=g, dtype=torch.float64).to(dev)
            th, tcc = tangent(dh, dc, c, gts, Whh)
            e = 1e-6
            hp, cp, _ = lstm_step(xin[:, None], h[:, None] + e * dh, c[:, None] + e * dc, Whh)
            hm, cm, _ = lstm_step(xin[:, None], h[:, None] - e * dh, c[:, None] - e * dc, Whh)
            num = torch.cat([hp - hm, cp - cm], -1) / (2 * e)
            ana = torch.cat([th, tcc], -1)
            errs.append(float(((num - ana).abs().amax() / num.abs().amax()).cpu()))
        h, c = h2, c2
    out["tangent_vs_finite_diff_rel_max"] = max(errs)
    # frame dependence of the network exponents and the fp32 forward vs the engine
    sub = subset()[:200]
    a = network("trained", 2100, sub, dev, frame_seed=FRAME_SEED)
    b = network("trained", 2100, sub, dev, frame_seed=FRAME_SEED + 1)
    fa, fb = a["local"].mean(1), b["local"].mean(1)
    out["frame_ftle_top5_mean_a"] = [float(x) for x in fa[:, :5].mean(0)]
    out["frame_ftle_top5_mean_b"] = [float(x) for x in fb[:, :5].mean(0)]
    out["frame_ftle1_per_episode_max_abs_diff"] = float(np.abs(fa[:, 0] - fb[:, 0]).max())
    eng = ti.forward(ti.load("trained", 2100), rows=sub, device=dev)
    out["logq_vs_engine_max"] = float(np.abs(a["logq"] - eng["logq"]).max())
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "checks.json")
    allc = json.load(open(path)) if os.path.exists(path) else {}                 # keeps theory_checks()'s entries
    allc.update(out)
    with open(path, "w") as fh:
        json.dump(allc, fh, indent=1)
    for k_, v in out.items():
        print(f"  {k_}: {v}")
    return out


# ═══════════════════════════════════════════════════════════════════
# Pilot
# ═══════════════════════════════════════════════════════════════════

def pilot(dev=None):
    dev = ti._device(dev)
    os.makedirs(os.path.join(OUT, "pilot"), exist_ok=True)
    rows = subset()
    path = os.path.join(OUT, "pilot", "bayes_side.npz")
    if not os.path.exists(path):
        t0 = time.time()
        B = bayes_side(rows)
        np.savez(path, **{k: v for k, v in B.items() if v is not None})
        print(f"bayes side {time.time() - t0:.0f} s")
    have = ti.checkpoints("trained")
    for run in RUNS:
        for raw in PILOT_RAW:
            assert raw in have, raw
            f = os.path.join(OUT, "pilot", f"{run}_{raw:05d}.npz")
            if os.path.exists(f):
                continue
            t0 = time.time()
            N = network(run, raw, rows, dev)
            np.savez(f, rows=rows, local=N["local"], chi=N["chi"], logq=N["logq"])
            print(f"{run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.1f} s", flush=True)


SWEEP_RUNS = ("trained", "echo", "trained_p1", "trained_N200", "trained_N100", "trained_N50")


def sweep(dev=None):
    """every stored checkpoint of SWEEP_RUNS on the probe subset, one file per checkpoint in chaos/sweep/ (trained and
    echo first, every third checkpoint before the rest; then the other runs the same way); stops before the next
    checkpoint when chaos/STOP exists. Stored per checkpoint: ftle (n, K) float32 = the mean of the random-frame QR
    local exponents over steps 1-30, ftle_late (n, K) float32 = their mean over steps 6-30, loc1 / locsum (n, T) the
    leading local exponent and the sum of the K local exponents per step, loc_mean (T, K) the probe mean per step (all
    of these are the superseded random-frame QR measure, see M1; not drawn), chi (n, len(FLIP_S), T, 2) float16 = d_H
    after a flip, mean over the 5 observations (drawn: panels e-g)"""
    dev = ti._device(dev)
    rows = subset()
    d = os.path.join(OUT, "sweep")
    os.makedirs(d, exist_ok=True)
    jobs = []
    for base, runs in ((0, SWEEP_RUNS[:2]), (2, SWEEP_RUNS[2:])):
        for run in runs:
            for i, raw in enumerate(ti.checkpoints(run)):
                jobs.append((base + (0 if i % 3 == 0 else 1), run, raw))
    jobs.sort(key=lambda j: j[0])
    print(f"{len(jobs)} checkpoints", flush=True)
    for _, run, raw in jobs:
        if os.path.exists(os.path.join(OUT, "STOP")):
            print("STOP file: stopping", flush=True)
            return
        f = os.path.join(d, f"{run}_{raw:05d}.npz")
        if os.path.exists(f):
            continue
        t0 = time.time()
        N = network(run, raw, rows, dev)
        loc = N["local"].astype(np.float64)
        tmp = f[:-4] + ".tmp.npz"
        np.savez(tmp, rows=rows, H=N["H"], ftle=loc.mean(1).astype(np.float32),
                 ftle_late=loc[:, 5:].mean(1).astype(np.float32), loc1=loc[..., 0].astype(np.float32),
                 locsum=loc.sum(-1).astype(np.float32), loc_mean=loc.mean(0),
                 chi=np.nanmean(N["chi"], 2).astype(np.float16))
        os.replace(tmp, f)
        print(f"{run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.1f} s", flush=True)


# ═══════════════════════════════════════════════════════════════════
# Converged state exponents (M1): finite-time amplification, asymptotic exponents, flip decay (GPU 0)
# ═══════════════════════════════════════════════════════════════════

LYAP_PER_ENV, AMP_K, LONG_K = 10, 4, 2
AMP_WINDOWS = ((0, T), (5, T))                                  # the propagators of steps 1-30 and 6-30
SMAX_T0 = 5                                                     # per-step log sigma_max at steps 6-30
LONG_BURN, LONG_AVG = 100, 200
LONG_SEED, LYAP_SEED = 20261108, 20261109
DECAY_S, DECAY_LAGS = (1, 3, 5, 10), (5, 20)
LYAP_RUNS = ("trained", "echo", "trained_p1", "trained_N200", "trained_N100", "trained_N50")


def lyap_rows():
    """the probes of the state measures: the first LYAP_PER_ENV of subset()'s 20 per environment (500 episodes)"""
    rows = subset()
    env = ti.probes()["env"][rows]
    return np.sort(np.concatenate([rows[env == e][:LYAP_PER_ENV] for e in range(gap.N_ENV)]))


def adjoint(lh, lc, c_prev, gts, Whh):
    """the transpose of tangent() (the exact VJP of lstm_step in (u, c)): cotangents lh, lc (b, K, H) of the state after
    the step -> the cotangents of the state before it"""
    i, f, g, o, tc = (a[:, None] for a in gts)
    mu = lc + o * (1 - tc * tc) * lh                                            # total cotangent of the new c
    pre = torch.cat([mu * (i * (1 - i)) * g, mu * (f * (1 - f)) * c_prev[:, None], mu * i * (1 - g * g),
                     lh * (o * (1 - o)) * tc], -1)                                # cotangents of the gate inputs
    return pre @ Whh, f * mu


def mgs2(Y):
    """orthonormal rows Q (b, k, d) of Y (b, k, d) by modified Gram-Schmidt run twice, and log R_jj (b, k) float64
    (R_jj = the product of the two passes' norms)"""
    Q, lr = [], []
    for j in range(Y.shape[1]):
        v, acc = Y[:, j], 0.
        for _ in range(2):
            for q in Q:
                v = v - (v * q).sum(-1, keepdim=True) * q
            nrm = v.norm(dim=-1)
            v = v / nrm[:, None]
            acc = acc + torch.log(nrm.double())
        Q.append(v)
        lr.append(acc)
    return torch.stack(Q, 1), torch.stack(lr, 1)


def _ritz_top(W):
    """log of the largest singular value (b,) of W (b, k, d), the images of an orthonormal block (Rayleigh-Ritz)"""
    M = torch.einsum("bkd,bld->bkl", W.double(), W.double()).cpu().numpy()
    return .5 * np.log(np.maximum(np.linalg.eigvalsh(M)[:, -1], 1e-300))


def _replay(p, table, envt, code):
    """the probes' forward pass with what the tangent maps need: gates G[t] of step t + 1, states U[t], C[t] after t
    steps (t = 0: the start state)"""
    Whh = p["LSTM.weight_hh_l0"]
    H = Whh.shape[1]
    b = len(envt)
    h = p["STM"].reshape(1, H).expand(b, H).contiguous()
    c = p["LTM"].reshape(1, H).expand(b, H).contiguous()
    G, U, C = [], [h], [c]
    for t in range(code.shape[1]):
        h, c, gts = lstm_step(table[envt, code[:, t]], h, c, Whh)
        G.append(gts)
        U.append(h)
        C.append(c)
    return G, U, C


def amplification(G, C, Whh, t0, t1, k=AMP_K, tol=5e-4, n_max=60, seed=LYAP_SEED):
    """frame-free finite-time amplification of the propagator Phi of steps t0 + 1 .. t1 (the state after t0 steps ->
    the state after t1): (1 / (t1 - t0)) log sigma_1(Phi) per probe, by block power iteration on Phi^T Phi (k vectors;
    tangent() forward, adjoint() back, MGS2, Rayleigh-Ritz). Stops when no probe's log sigma_1 moved by more than tol in
    an iteration. Returns the rates (b,), the last change of log sigma_1 (b,), the iterations"""
    b, H = C[0].shape
    gen = torch.Generator(device=Whh.device).manual_seed(seed)
    V = mgs2(torch.randn(b, k, 2 * H, device=Whh.device, generator=gen))[0]
    prev = None
    for it in range(n_max):
        dh, dc = V[..., :H], V[..., H:]
        for t in range(t0, t1):
            dh, dc = tangent(dh, dc, C[t], G[t], Whh)
        ls = _ritz_top(torch.cat([dh, dc], -1))
        for t in reversed(range(t0, t1)):
            dh, dc = adjoint(dh, dc, C[t], G[t], Whh)
        Z = torch.cat([dh, dc], -1)
        V = mgs2(Z / Z[:, :1].norm(dim=-1, keepdim=True))[0]
        ch = np.full(b, np.inf) if prev is None else np.abs(ls - prev)
        if ch.max() < tol:
            break
        prev = ls
    return ls / (t1 - t0), ch, it + 1


def step_smax(G, C, Whh, t0=SMAX_T0, tol=1e-4, n_max=60, seed=LYAP_SEED):
    """log sigma_max of each step's Jacobian of (u, c) (Euclidean) at steps t0 + 1 .. T, all steps at once: power
    iteration on J_t^T J_t with one vector (the estimate log ||J_t v|| rises to log sigma_max from below). Returns
    (b, T - t0), the last change (b, T - t0), the iterations"""
    b, H = C[0].shape
    S = len(G) - t0
    gts = tuple(torch.cat([G[t][j] for t in range(t0, len(G))]) for j in range(5))        # (S b, H), step-major
    cp = torch.cat([C[t] for t in range(t0, len(G))])
    gen = torch.Generator(device=Whh.device).manual_seed(seed)
    v = torch.randn(S * b, 1, 2 * H, device=Whh.device, generator=gen)
    v = v / v.norm(dim=-1, keepdim=True)
    prev = None
    for it in range(n_max):
        dh, dc = tangent(v[..., :H], v[..., H:], cp, gts, Whh)
        ls = torch.log(torch.cat([dh, dc], -1)[:, 0].norm(dim=-1).double()).cpu().numpy()
        dh, dc = adjoint(dh, dc, cp, gts, Whh)
        v = torch.cat([dh, dc], -1)
        v = v / v.norm(dim=-1, keepdim=True)
        ch = np.full(S * b, np.inf) if prev is None else np.abs(ls - prev)
        if ch.max() < tol:
            break
        prev = ls
    return ls.reshape(S, b).T, ch.reshape(S, b).T, it + 1


def long_codes(rows, n, seed=LONG_SEED):
    """(len(rows), n) observation codes sampled i.i.d. from each probe's environment and cell, x_o ~ Bernoulli(L[env, o,
    r0, r1]) (bit o of the code = x_o)"""
    P = ti.probes()
    L, _ = tables()
    pr = L[P["env"][rows], :, P["r"][rows, 0], P["r"][rows, 1]]                             # (n, 5)
    ob = np.random.default_rng(seed).random((len(rows), n, NO)) < pr[:, None, :]
    return (ob * (1 << np.arange(NO))).sum(-1).astype(np.int64)


def asymptotic(p, table, envt, codes, k=LONG_K, burn=LONG_BURN, seed=LYAP_SEED, Q0=None):
    """conditional Lyapunov exponents (b, k) of (u, c) along the sampled sequences codes (b, n) from the trained start
    state: QR (MGS2) of k tangent vectors from a random orthonormal frame (one frame from seed for all sequences, or Q0
    (b, k, 2H): one per sequence), the mean of log R_jj over steps burn + 1 .. n (nats per step)"""
    Whh = p["LSTM.weight_hh_l0"]
    H = Whh.shape[1]
    b, n = codes.shape
    h = p["STM"].reshape(1, H).expand(b, H).contiguous()
    c = p["LTM"].reshape(1, H).expand(b, H).contiguous()
    if Q0 is None:
        g = torch.Generator().manual_seed(seed)
        Q = torch.linalg.qr(torch.randn(2 * H, k, generator=g, dtype=torch.float64))[0].T.float().to(Whh.device)
        Q = Q[None].expand(b, k, 2 * H).contiguous()
    else:
        Q = Q0.to(Whh.device, Whh.dtype).contiguous()
    acc = torch.zeros(b, k, device=Whh.device, dtype=torch.float64)
    for t in range(n):
        h2, c2, gts = lstm_step(table[envt, codes[:, t]], h, c, Whh)
        dh, dc = tangent(Q[..., :H], Q[..., H:], c, gts, Whh)
        Q, lr = mgs2(torch.cat([dh, dc], -1))
        if t >= burn:
            acc += lr
        h, c = h2, c2
    return (acc / (n - burn)).cpu().numpy()


def flip_decay(table, envt, code, U, C, Whh, flips=DECAY_S, lags=DECAY_LAGS):
    """the state difference after one flipped observation: log ||(u, c)_flipped - (u, c)_intact|| (b, 21) at lags 0-20
    (mean over the flips: steps DECAY_S x the 5 observations), and the decay rate (b,) = the mean over the flips of
    (log d_20 - log d_5) / 15 (nats per step)"""
    b = len(envt)
    flipbit = torch.as_tensor(1 << np.arange(NO), device=Whh.device).repeat(b)
    eb = envt.repeat_interleave(NO)
    logd = torch.zeros(b, len(flips), NO, lags[1] + 1, device=Whh.device, dtype=torch.float64)
    for a, s in enumerate(flips):
        hb, cb = U[s - 1].repeat_interleave(NO, 0), C[s - 1].repeat_interleave(NO, 0)
        for j in range(lags[1] + 1):
            t = s - 1 + j
            cd = code[:, t].repeat_interleave(NO)
            if j == 0:
                cd = cd ^ flipbit
            hb, cb, _ = lstm_step(table[eb, cd], hb, cb, Whh)
            d = torch.cat([hb - U[t + 1].repeat_interleave(NO, 0), cb - C[t + 1].repeat_interleave(NO, 0)], -1)
            logd[:, a, :, j] = torch.log(d.double().norm(dim=-1)).reshape(b, NO)
    rate = (logd[..., lags[1]] - logd[..., lags[0]]) / (lags[1] - lags[0])
    return logd.mean((1, 2)).cpu().numpy(), rate.mean((1, 2)).cpu().numpy()


@torch.no_grad()
def lyap(run, raw, dev, smax=True, chunk=250, long_seed=LONG_SEED, frame_seed=LYAP_SEED, n_long=None, burn=LONG_BURN,
         parts=("amp", "lam", "decay"), rows=None):
    """the converged state measures of one checkpoint on lyap_rows(): amp (n, 2) = the amplification over steps 1-30 and
    6-30 (amp_change, amp_iters: convergence), smax (n, 25) = log sigma_max of the step Jacobians at steps 6-30 (if
    smax), lam (n, LONG_K) = the asymptotic exponents (sampled sequences of n_long = LONG_BURN + LONG_AVG steps, the
    first burn discarded), decay_logd (n, 21) / decay_rate (n,) = the flip decay; rows: lyap_rows() if None"""
    rows = lyap_rows() if rows is None else rows
    w = ti.load(run, raw)
    p = {kk: v.to(dev, torch.float32) for kk, v in w.items()}
    Whh = p["LSTM.weight_hh_l0"]
    H = Whh.shape[1]
    P = ti.probes()
    table = ti._table(p, P["Z"], dev)
    code = torch.as_tensor((P["obs"][rows] * (1 << np.arange(NO))).sum(-1).astype(np.int64), device=dev)
    envt = torch.as_tensor(P["env"][rows].astype(np.int64), device=dev)
    lcode = torch.as_tensor(long_codes(rows, n_long or LONG_BURN + LONG_AVG, long_seed), device=dev)
    n = len(rows)
    out = dict(rows=rows, H=H)
    if "amp" in parts:
        out.update(amp=np.zeros((n, 2)), amp_change=np.zeros((n, 2)), amp_iters=np.zeros(2, np.int64))
    if smax:
        out.update(smax=np.zeros((n, T - SMAX_T0)), smax_change=np.zeros((n, T - SMAX_T0)),
                   smax_iters=np.zeros(1, np.int64))
    if "lam" in parts:
        out["lam"] = np.zeros((n, LONG_K))
    if "decay" in parts:
        out.update(decay_logd=np.zeros((n, DECAY_LAGS[1] + 1)), decay_rate=np.zeros(n))
    for b0 in range(0, n, chunk):
        sl = slice(b0, min(n, b0 + chunk))
        if "amp" in parts or smax or "decay" in parts:
            G, U, C = _replay(p, table, envt[sl], code[sl])
            if "amp" in parts:
                for j, (t0, t1) in enumerate(AMP_WINDOWS):
                    a, ch, it = amplification(G, C, Whh, t0, t1)
                    out["amp"][sl, j], out["amp_change"][sl, j] = a, ch
                    out["amp_iters"][j] = max(out["amp_iters"][j], it)
            if smax:
                s_, ch, it = step_smax(G, C, Whh)
                out["smax"][sl], out["smax_change"][sl] = s_, ch
                out["smax_iters"][0] = max(out["smax_iters"][0], it)
            if "decay" in parts:
                out["decay_logd"][sl], out["decay_rate"][sl] = flip_decay(table, envt[sl], code[sl], U, C, Whh)
            del G, U, C
        if "lam" in parts:
            out["lam"][sl] = asymptotic(p, table, envt[sl], lcode[sl], burn=burn, seed=frame_seed)
    return out


def lyap_sweep(dev=None, runs=LYAP_RUNS):
    """the converged state measures through training -> chaos/lyap/{run}_{raw:05d}.npz, in phases: (1) trained and
    echo, every third checkpoint and the last, with the per-step log sigma_max; (2) the other runs, every sixth
    checkpoint and the last; (3), (4) trained and echo, the remaining checkpoints (i % 3 = 1, then 2). Stops before the
    next checkpoint when chaos/lyap/STOP exists"""
    dev = ti._device(dev)
    d = os.path.join(OUT, "lyap")
    os.makedirs(d, exist_ok=True)
    jobs = []
    for r_, run in enumerate(runs):
        eps = ti.checkpoints(run)
        for i, raw in enumerate(eps):
            last = i == len(eps) - 1
            if run in RUNS:
                ph = 1 if (i % 3 == 0 or last) else 3 if i % 3 == 1 else 4
            else:
                ph = 2 if (i % 6 == 0 or last) else None
            if ph is not None:
                jobs.append((ph, i, r_, run, raw))
    jobs.sort()
    print(f"{len(jobs)} checkpoints", flush=True)
    for ph, _, _, run, raw in jobs:
        if os.path.exists(os.path.join(d, "STOP")):
            print("STOP file: stopping", flush=True)
            return
        f = os.path.join(d, f"{run}_{raw:05d}.npz")
        if os.path.exists(f):
            continue
        t0 = time.time()
        chunk = 125
        while True:                                                     # GPU 0 is shared: back off on a full card
            try:
                N = lyap(run, raw, dev, smax=ph == 1, chunk=chunk)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if chunk > 25:
                    chunk //= 2
                else:
                    print("GPU 0 full: waiting 60 s", flush=True)
                    time.sleep(60)
        tmp = f[:-4] + ".tmp.npz"
        np.savez(tmp, **N)
        os.replace(tmp, f)
        print(f"phase {ph} {run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.1f} s; amp {N['amp'].mean(0).round(4)}"
              f" (iters {N['amp_iters']}), lam1 {N['lam'][:, 0].mean():+.4f}, decay {N['decay_rate'].mean():+.4f}",
              flush=True)


def adf_lyap():
    """ADF's counterparts of the state measures on lyap_rows() (float64, CPU), in its log-marginal quotient chart
    (orthonormal zero-sum coordinates, as qr_lyap): amp (n, 2) = (1 / (t1 - t0)) log sigma_1 of the propagators of steps
    1-30 and 6-30 (exact SVD); smax (n, 25) = log sigma_max of the step Jacobians at steps 6-30; lam (n, LONG_K) = the
    asymptotic exponents along the networks' sampled sequences (long_codes; the same burn-in, window and frame seed);
    rho1 (n, T), the QR local exponent 1 (n, T; frame FRAME_SEED), entanglement (n, T, mean of the 2 variables)
    -> chaos/lyap/adf.npz"""
    rows = lyap_rows()
    P = ti.probes()
    L, ent = tables()
    Le = L[P["env"][rows]]
    n, d = len(rows), 2 * R - 2

    def jac(obs):
        ll = step_loglik(obs, Le)
        la = adf_run(ll)
        la_prev = np.concatenate([np.full((n, 1, 2, R), -np.log(R)), la[:, :-1]], 1)
        return adf_jacobians(ll, la_prev)
    J, rho = jac(P["obs"][rows])
    amp = []
    for t0, t1 in AMP_WINDOWS:
        Pm = np.broadcast_to(np.eye(d), (n, d, d)).copy()
        for t in range(t0, t1):
            Pm = J[:, t] @ Pm
        amp.append(np.log(np.linalg.svd(Pm, compute_uv=False)[:, 0]) / (t1 - t0))
    smax = np.log(np.linalg.svd(J[:, SMAX_T0:], compute_uv=False)[..., 0])
    loc1 = qr_lyap(J)[..., 0]
    codes = long_codes(rows, LONG_BURN + LONG_AVG)
    Jl, rho_l = jac(((codes[..., None] >> np.arange(NO)) & 1).astype(bool))
    lam = qr_lyap(Jl, k=LONG_K, seed=LYAP_SEED)[:, LONG_BURN:].mean(1)
    out = dict(rows=rows, amp=np.stack(amp, 1), smax=smax, lam=lam, rho1=rho[..., 0], loc1=loc1,
               ent=ent[rows].mean(-1), rho1_long=rho_l[..., 0].mean(0))
    os.makedirs(os.path.join(OUT, "lyap"), exist_ok=True)
    np.savez(os.path.join(OUT, "lyap", "adf.npz"), **out)
    print(f"ADF: amp {out['amp'].mean(0).round(4)}, lam {lam.mean(0).round(4)}, mean smax {smax.mean():.3f}")
    return out


def theory_checks():
    """CPU checks of the theory's qualifiers (11-02), merged into chaos/checks.json: T1 in probability charts (joint
    Bayes and naive from their uniform start, an infinitesimal zero-sum change of the log prior, 30 steps, 1,000 probes);
    T3's metric (sigma_max = 1 + rho_1 in the step's own pi-weighted metric, 200 probes x 30 steps), its Euclidean
    zero-sum chart (sigma_max quantiles), the volume (log det = sum_k log(1 - rho_k^2)) and the front-loaded coupling"""
    P = ti.probes()
    L, _ = tables()
    rows = subset()
    n = len(rows)
    Le = L[P["env"][rows]]
    ll = step_loglik(P["obs"][rows], Le)
    rng = np.random.default_rng(CHECK_SEED + 1)
    eps, out = 1e-6, {}
    nrm = np.linalg.norm

    def charts(lp0, lq0, lpT, lqT):
        """30-step exponents of the change lq - lp (log beliefs, (n, ...)) in three charts"""
        p0, q0, pT, qT = (np.exp(x).reshape(n, -1) for x in (lp0, lq0, lpT, lqT))
        eu = np.log(nrm(qT - pT, axis=1) / nrm(q0 - p0, axis=1)) / T
        fr = np.log(nrm(np.sqrt(qT) - np.sqrt(pT), axis=1) / nrm(np.sqrt(q0) - np.sqrt(p0), axis=1)) / T
        dh = np.log(osc(lqT - lpT).reshape(n, -1).max(1) / osc(lq0 - lp0).reshape(n, -1).max(1)) / T
        return eu, fr, dh
    u = np.cumsum(ll, 1).reshape(n, T, R * R)
    dl = rng.normal(0, 1, (n, R * R))
    dl -= dl.mean(1, keepdims=True)
    lp0 = np.full((n, R * R), -2 * np.log(R))
    lq0 = eps * dl - logsumexp(eps * dl, -1)[:, None]
    lpT = u[:, -1] - logsumexp(u[:, -1], -1)[:, None]
    lqT = u[:, -1] + eps * dl - logsumexp(u[:, -1] + eps * dl, -1)[:, None]
    for k_, v in zip(("euclid", "fisher_rao", "dH"), charts(lp0, lq0, lpT, lqT)):
        out[f"T1_joint_{k_}_30step_mean_p5_p95"] = [float(v.mean()), float(np.percentile(v, 5)), float(np.percentile(v, 95))]
        out[f"T1_joint_{k_}_share_positive"] = float((v > 0).mean())
    Lb = np.stack([Le.mean(3), Le.mean(2)], 2)
    o = P["obs"][rows].astype(np.float64)
    cl = np.cumsum(np.einsum("nti,nikr->ntkr", o, np.log(Lb)) + np.einsum("nti,nikr->ntkr", 1 - o, np.log1p(-Lb)), 1)
    dv = rng.normal(0, 1, (n, 2, R))
    dv -= dv.mean(-1, keepdims=True)
    lp0 = np.full((n, 2, R), -np.log(R))
    lq0 = eps * dv - logsumexp(eps * dv, -1)[..., None]
    lpT = cl[:, -1] - logsumexp(cl[:, -1], -1)[..., None]
    lqT = cl[:, -1] + eps * dv - logsumexp(cl[:, -1] + eps * dv, -1)[..., None]
    for k_, v in zip(("euclid", "fisher_rao", "dH"), charts(lp0, lq0, lpT, lqT)):
        out[f"T1_naive_{k_}_30step_mean_p5_p95"] = [float(v.mean()), float(np.percentile(v, 5)), float(np.percentile(v, 95))]
        out[f"T1_naive_{k_}_share_positive"] = float((v > 0).mean())
    # T3: the step's own metric, the Euclidean chart, the volume, the front-loaded coupling
    la = adf_run(ll)
    la_prev = np.concatenate([np.full((n, 1, 2, R), -np.log(R)), la[:, :-1]], 1)
    J, rho = adf_jacobians(ll, la_prev)
    sv = np.linalg.svd(J, compute_uv=False)[..., 0]
    out["T3_euclid_sigma_max_p50_p95_max"] = [float(np.percentile(sv, 50)), float(np.percentile(sv, 95)), float(sv.max())]
    out["T3_euclid_share_sigma_max_above_2"] = float((sv > 2).mean())
    out["T3_euclid_sigma_max_minus_1_plus_rho1_p50_max"] = [float(np.percentile(sv - 1 - rho[..., 0], 50)),
                                                           float((sv - 1 - rho[..., 0]).max())]
    sg, ld = np.linalg.slogdet(J)
    out["T3_logdet_minus_sum_log_1_minus_rho2_maxabs"] = float(np.abs(ld - np.log1p(-rho ** 2).sum(-1)).max())
    out["T3_det_sign_min"] = float(sg.min())
    out["T3_det_max"] = float((sg * np.exp(ld)).max())
    out["T3_rho1_mean_step1_step30"] = [float(rho[:, 0, 0].mean()), float(rho[:, -1, 0].mean())]
    sub = slice(0, 200)
    j = la_prev[sub, :, 0, :, None] + la_prev[sub, :, 1, None, :] + ll[sub]
    j -= logsumexp(j.reshape(200, T, -1), -1)[..., None, None]
    l0, l1 = logsumexp(j, -1), logsumexp(j, -2)
    p0, p1 = np.exp(l0), np.exp(l1)                                                         # (200, T, R)
    A = np.exp(j - l0[..., :, None])
    B = np.swapaxes(np.exp(j - l1[..., None, :]), -1, -2)
    one = np.ones(R)
    Jf = np.zeros((200, T, 2 * R, 2 * R))                                                   # full log-coordinate Jacobian
    Jf[..., :R, :R] = np.eye(R) - one[:, None] * p0[..., None, :]
    Jf[..., :R, R:] = A - one[:, None] * p1[..., None, :]
    Jf[..., R:, :R] = B - one[:, None] * p0[..., None, :]
    Jf[..., R:, R:] = np.eye(R) - one[:, None] * p1[..., None, :]
    d_ = 2 * R - 2
    Xb, Yb = np.zeros((200, T, 2 * R, d_)), np.zeros((200, T, d_, 2 * R))
    for v_, pv in ((0, p0), (1, p1)):
        w = np.sqrt(pv)
        U = np.linalg.svd(np.eye(R) - w[..., :, None] * w[..., None, :])[0][..., :R - 1]   # orthonormal basis of sqrt(p)^perp
        Xb[..., v_ * R:(v_ + 1) * R, v_ * (R - 1):(v_ + 1) * (R - 1)] = U / w[..., :, None]
        Yb[..., v_ * (R - 1):(v_ + 1) * (R - 1), v_ * R:(v_ + 1) * R] = np.swapaxes(U, -1, -2) * w[..., None, :]
    Jw = Yb @ Jf @ Xb                                                                       # in L2(p0) + L2(p1)
    sw = np.linalg.svd(Jw, compute_uv=False)[..., 0]
    out["T3_pi_metric_sigma_max_minus_1_plus_rho1_maxabs"] = float(np.abs(sw - 1 - rho[sub, :, 0]).max())
    path = os.path.join(OUT, "checks.json")
    allc = json.load(open(path)) if os.path.exists(path) else {}
    allc.update(out)
    with open(path, "w") as fh:
        json.dump(allc, fh, indent=1)
    for k_, v in out.items():
        print(f"  {k_}: {v}")
    return out


@torch.no_grad()
def lyap_check(dev=None):
    """checks of the converged estimators -> chaos/lyap/checks.json: adjoint() vs torch.func.vjp (float64); the
    amplification vs the exact SVD of the explicit 30-step propagator (2 probes x 3 checkpoints); the amplification at
    tol 5e-4 vs 1e-7 (125 probes); the per-step log sigma_max with 1 vector vs a 3-vector block (125 probes); the
    asymptotic exponent with a second frame and a second observation draw (500 probes, 7 checkpoints) and over a longer
    horizon (1,000 steps, burn-in 300; 125 probes, 3 checkpoints)"""
    dev = ti._device(dev)
    out = {}
    P = ti.probes()
    rows = lyap_rows()
    code_all = (P["obs"] * (1 << np.arange(NO))).sum(-1).astype(np.int64)

    def setup(run, raw, dt=torch.float32):
        w = ti.load(run, raw)
        p = {kk: v.to(dev, dt) for kk, v in w.items()}
        table = ti._table({kk: v.float() for kk, v in p.items()}, P["Z"], dev).to(dt)
        return p, table, p["LSTM.weight_hh_l0"]

    def tens(sel):
        return (torch.as_tensor(P["env"][sel].astype(np.int64), device=dev),
                torch.as_tensor(code_all[sel], device=dev))
    # adjoint vs autograd, float64
    p, table, Whh = setup("trained", 49995, torch.float64)
    H = Whh.shape[1]
    envt, code = tens(rows[:20])
    G, U, C = _replay(p, table, envt, code)
    g = torch.Generator(device=dev).manual_seed(CHECK_SEED)
    errs = []
    for t in (0, 9, 29):
        lh = torch.randn(20, 1, H, device=dev, dtype=torch.float64, generator=g)
        lc = torch.randn(20, 1, H, device=dev, dtype=torch.float64, generator=g)
        ah, ac = adjoint(lh, lc, C[t], G[t], Whh)
        _, fn = torch.func.vjp(lambda hh, cc: lstm_step(table[envt, code[:, t]], hh, cc, Whh)[:2], U[t], C[t])
        rh, rc = fn((lh[:, 0], lc[:, 0]))
        errs.append(float(torch.cat([ah[:, 0] - rh, ac[:, 0] - rc], -1).abs().max() / torch.cat([rh, rc], -1).abs().max()))
    out["adjoint_vs_vjp_float64_rel_max"] = max(errs)
    # the superseded QR: CholeskyQR2 vs torch.linalg.qr inside network()
    qa = network("trained", 2100, subset()[:100], dev, qr="chol")["local"]
    qb = network("trained", 2100, subset()[:100], dev, qr="torch")["local"]
    out["pilot_cholqr2_vs_torch_qr_local_maxabs"] = float(np.abs(qa - qb).max())
    # amplification vs the exact SVD of the explicit propagator
    sel = subset()[[0, 666]]
    for run, raw in (("trained", 49995), ("trained", 2100), ("echo", 49995)):
        p, table, Whh = setup(run, raw)
        H = Whh.shape[1]
        envt, code = tens(sel)
        G, U, C = _replay(p, table, envt, code)
        res = []
        for t0, t1 in AMP_WINDOWS:
            ex = []
            for i in range(2):
                Gi = [tuple(a[i:i + 1] for a in gt) for gt in G]
                Ci = [c[i:i + 1] for c in C]
                Q = torch.eye(2 * H, device=dev)[None]
                dh, dc = Q[..., :H], Q[..., H:]
                for t in range(t0, t1):
                    dh, dc = tangent(dh, dc, Ci[t], Gi[t], Whh)
                Phi = torch.cat([dh, dc], -1)[0].double()                                    # rows = images of e_j
                ex.append(float(torch.linalg.svdvals(Phi)[0].log()) / (t1 - t0))
            a, _, _ = amplification(G, C, Whh, t0, t1)
            res.append([ex, [float(x) for x in a]])
        out[f"amp_vs_exact_svd_{run}_{raw}"] = res
    # amplification tolerance; per-step sigma_max with one vector vs a 3-vector block
    for run, raw in (("trained", 49995), ("echo", 49995)):
        p, table, Whh = setup(run, raw)
        envt, code = tens(rows[:125])
        G, U, C = _replay(p, table, envt, code)
        a1 = amplification(G, C, Whh, 0, T)[0]
        a2 = amplification(G, C, Whh, 0, T, tol=1e-7, n_max=300)[0]
        out[f"amp_tol_5e-4_vs_1e-7_maxabs_{run}_{raw}"] = float(np.abs(a1 - a2).max())
        s1 = step_smax(G, C, Whh)[0]
        H = Whh.shape[1]
        gts = tuple(torch.cat([G[t][j] for t in range(SMAX_T0, T)]) for j in range(5))
        cp = torch.cat([C[t] for t in range(SMAX_T0, T)])
        V = mgs2(torch.randn(len(cp), 3, 2 * H, device=dev, generator=g))[0]
        for _ in range(100):
            dh, dc = tangent(V[..., :H], V[..., H:], cp, gts, Whh)
            ls = _ritz_top(torch.cat([dh, dc], -1))
            dh, dc = adjoint(dh, dc, cp, gts, Whh)
            V = mgs2(torch.cat([dh, dc], -1))[0]
        s3 = ls.reshape(T - SMAX_T0, -1).T
        ent = tables()[1][rows[:125]].mean(-1)[:, SMAX_T0:]
        out[f"smax_1vec_minus_3block_mean_maxabs_{run}_{raw}"] = [float((s1 - s3).mean()), float(np.abs(s1 - s3).max())]
        out[f"smax_r_ent_1vec_3block_{run}_{raw}"] = [_corr_within(s1, ent), _corr_within(s3, ent)]
    # asymptotic exponent: frames, draws, horizon
    cks = (("trained", 0), ("trained", 400), ("trained", 2100), ("trained", 49995), ("echo", 49995),
           ("trained_N50", 49995), ("trained_p1", 49995))
    for run, raw in cks:
        a = lyap(run, raw, dev, smax=False, chunk=125, parts=("lam",))["lam"][:, 0]
        b = lyap(run, raw, dev, smax=False, chunk=125, parts=("lam",), frame_seed=LYAP_SEED + 1)["lam"][:, 0]
        c = lyap(run, raw, dev, smax=False, chunk=125, parts=("lam",), long_seed=LONG_SEED + 1)["lam"][:, 0]
        out[f"lam1_frameA_frameB_drawB_{run}_{raw}"] = dict(
            mean=[float(a.mean()), float(b.mean()), float(c.mean())], positive=[int((x > 0).sum()) for x in (a, b, c)],
            max=[float(x.max()) for x in (a, b, c)], frame_diff_sd_maxabs=[float((a - b).std()), float(np.abs(a - b).max())],
            draw_diff_sd_maxabs=[float((a - c).std()), float(np.abs(a - c).max())])
    for run, raw in (("trained", 2100), ("trained", 49995), ("echo", 49995)):
        a = lyap(run, raw, dev, smax=False, chunk=125, parts=("lam",), rows=rows[:125])["lam"][:, 0]
        b = lyap(run, raw, dev, smax=False, chunk=125, parts=("lam",), rows=rows[:125], n_long=1000, burn=300)["lam"][:, 0]
        out[f"lam1_steps101_300_vs_301_1000_{run}_{raw}"] = [float(a.mean()), float(b.mean()), int((b > 0).sum()),
                                                             float(b.max())]
    os.makedirs(os.path.join(OUT, "lyap"), exist_ok=True)
    path = os.path.join(OUT, "lyap", "checks.json")
    allc = json.load(open(path)) if os.path.exists(path) else {}
    allc.update(out)
    with open(path, "w") as fh:
        json.dump(allc, fh, indent=1)
    for k_, v in out.items():
        print(f"  {k_}: {v}", flush=True)
    return out


RECHECK_DRAWS, RECHECK_N, RECHECK_BURN = 8, 2000, 500           # lam_recheck: 8 draws x steps 501-2,000
RECHECK_SEED, RECHECK_FRAME = 20261120, 20261140                # draw d: observations RECHECK_SEED + d, frame RECHECK_FRAME + d
DECIDE_DRAWS, DECIDE_N, DECIDE_BURN = 64, 4000, 1000            # lam_decide: 64 draws x steps 1,001-4,000
DECIDE_SEED, DECIDE_FRAME = 20261160, 20261161                  # per candidate: rng([seed, raw, row])
LONG2_BURN, LONG2_AVG = 500, 1000                               # lam_long: steps 501-1,500 of a new 1,500-step draw
LONG2_SEED, LONG2_FRAMES = 20261110, (20261111, 20261112, 20261113, 20261114)
LONG2_FROM = 4000.                                              # lam_long: every checkpoint from 4,000 episodes on


def _merge_checks(out):
    """merge out into chaos/lyap/checks.json"""
    path = os.path.join(OUT, "lyap", "checks.json")
    allc = json.load(open(path)) if os.path.exists(path) else {}
    allc.update(out)
    with open(path, "w") as fh:
        json.dump(allc, fh, indent=1)


@torch.no_grad()
def lam_recheck(dev=None, runs=LYAP_RUNS):
    """the candidates of M1 (b)'s chaos test and its strict count: every probe-checkpoint of chaos/lyap/ with a positive
    300-step estimate lambda_1 > 0, or with a frame-free bound >= 0 where lam_long() computed one (the candidates),
    re-estimated with RECHECK_DRAWS = 8 independent observation draws of 2,000 steps, each with its own frame, the mean
    of log |R_11| over steps 501-2,000. Strict count: all 8 estimates > 0 (at lambda_1 = 0 an estimate is positive with
    probability ~1/2, the frame bias being negative, so ~1/256 of the candidates pass by chance; little power at
    lambda_1 ~ +.005, so the decision is lam_decide()'s) -> chaos/lyap/checks.json, lam1_recheck_{run}: per candidate
    x, row, orig (the 300-step estimate), bound (NaN if none), est (8). Replaces the first recheck
    (lam1_positive_recheck_{run}: one draw of 300 and one of 1,000 steps, a single noisy redraw)"""
    dev = ti._device(dev)
    out = {}
    for run in runs:
        z = lyap_load(run)
        if z is None:
            continue
        cand = z["lam"][..., 0] > 0                                                         # (K, n)
        bnd = np.full(cand.shape, np.nan)
        zl = long_load(run)
        for j, x in enumerate([] if zl is None else zl["x"]):
            if np.isfinite(zl["upper"][j]).all():
                k = int(np.flatnonzero(z["x"] == x)[0])
                bnd[k] = zl["upper"][j]
        cand |= bnd >= 0
        hit = np.argwhere(cand)                                                             # (m, 2): checkpoint, probe
        res = dict(n=int(len(hit)), draws=RECHECK_DRAWS, n_long=RECHECK_N, burn=RECHECK_BURN, x=[], row=[], orig=[],
                   bound=[], est=[])
        t0 = time.time()
        for k in np.unique(hit[:, 0]) if len(hit) else []:
            pr = hit[hit[:, 0] == k, 1]
            rows = z["rows"][pr]
            est = np.stack([lyap(run, int(z["raw"][k]), dev, smax=False, chunk=125, parts=("lam",), rows=rows,
                                 long_seed=RECHECK_SEED + d_, frame_seed=RECHECK_FRAME + d_, n_long=RECHECK_N,
                                 burn=RECHECK_BURN)["lam"][:, 0] for d_ in range(RECHECK_DRAWS)], 1)    # (m, 8)
            res["x"] += [float(z["x"][k])] * len(pr)
            res["row"] += [int(r_) for r_ in rows]
            res["orig"] += [float(v) for v in z["lam"][k, pr, 0]]
            res["bound"] += [float(v) for v in bnd[k, pr]]
            res["est"] += est.tolist()
        e = np.array(res["est"]).reshape(-1, RECHECK_DRAWS)
        res["robust"] = int((e > 0).all(1).sum())
        out[f"lam1_recheck_{run}"] = res
        print(f"{run}: {res['n']} candidates, positive in all {RECHECK_DRAWS} draws: {res['robust']}"
              + (f"; 8-draw means max {e.mean(1).max():+.4f}" if len(e) else "") + f" ({time.time() - t0:.0f} s)",
              flush=True)
    _merge_checks(out)
    return out


@torch.no_grad()
def lam_decide(dev=None, runs=LYAP_RUNS, dtype=torch.float32):
    """the decision of M1 (b)'s chaos test: every candidate of lam_recheck() (every run) re-estimated with DECIDE_DRAWS =
    64 independent observation draws of DECIDE_N = 4,000 steps from the trained start state, each with its own random
    orthonormal frame (asymptotic(): QR of 2 tangent vectors, the mean of log |R_jj| over steps 1,001-4,000). A
    candidate's draws and frames come from its own generators (seeds [DECIDE_SEED, raw, row] and [DECIDE_FRAME, raw,
    row]), so they do not depend on the other candidates. Decided in _recheck_stats(): chaotic if the 64-draw mean - 2
    SEM > 0 -> chaos/lyap/decide/{run}.npz: x, raw, row (m,) in lam1_recheck_{run}'s order, est (m, 64, 2) = lambda_1,
    lambda_2 per draw. dtype float64: the same draws and frames with the recurrent step and tangent maps in float64 (the
    input drive table computed in float32, as bound_check) -> decide/{run}_float64.npz (the precision control). Skips
    runs whose file exists"""
    dev = ti._device(dev)
    d = os.path.join(OUT, "lyap", "decide")
    os.makedirs(d, exist_ok=True)
    chk = json.load(open(os.path.join(OUT, "lyap", "checks.json")))
    P = ti.probes()
    D = DECIDE_DRAWS
    for run in runs:
        v = chk.get(f"lam1_recheck_{run}")
        f = os.path.join(d, f"{run}{'' if dtype == torch.float32 else '_float64'}.npz")
        if v is None or os.path.exists(f):
            continue
        z = lyap_load(run)
        x, row = np.array(v["x"], np.float64), np.array(v["row"], np.int64)
        raw = np.array([z["raw"][np.flatnonzero(z["x"] == x_)[0]] for x_ in x], np.int64)
        est = np.zeros((len(x), D, LONG_K))
        t0 = time.time()
        for r_ in np.unique(raw):
            sel = np.flatnonzero(raw == r_)
            w = ti.load(run, int(r_))
            table = ti._table({kk: vv.to(dev, torch.float32) for kk, vv in w.items()}, P["Z"], dev).to(dtype)
            p = {kk: vv.to(dev, dtype) for kk, vv in w.items()}
            H = p["LSTM.weight_hh_l0"].shape[1]
            codes = np.concatenate([long_codes(np.repeat(row[i], D), DECIDE_N, [DECIDE_SEED, int(r_), int(row[i])])
                                    for i in sel])                                              # (len(sel) D, n)
            Q0 = np.concatenate([np.linalg.qr(np.random.default_rng([DECIDE_FRAME, int(r_), int(row[i])])
                                              .standard_normal((D, 2 * H, LONG_K)))[0] for i in sel])   # (len(sel) D, 2H, k)
            envt = torch.as_tensor(P["env"][np.repeat(row[sel], D)].astype(np.int64), device=dev)
            est[sel] = asymptotic(p, table, envt, torch.as_tensor(codes, device=dev), burn=DECIDE_BURN,
                                  Q0=torch.as_tensor(np.ascontiguousarray(Q0.transpose(0, 2, 1)), dtype=dtype)
                                  ).reshape(len(sel), D, LONG_K)
            del p, table
        tmp = f[:-4] + ".tmp.npz"
        np.savez(tmp, x=x, raw=raw, row=row, est=est, draws=D, n_long=DECIDE_N, burn=DECIDE_BURN, seed=DECIDE_SEED,
                 frame_seed=DECIDE_FRAME)
        os.replace(tmp, f)
        m, sem = est[..., 0].mean(1), est[..., 0].std(1, ddof=1) / np.sqrt(D)
        print(f"{run}: {len(x)} candidates; {D}-draw mean - 2 SEM > 0: {int((m - 2 * sem > 0).sum())}"
              + (f"; means max {m.max():+.4f}" if len(x) else "") + f" ({time.time() - t0:.0f} s)", flush=True)


def decide_load(run):
    """lam_decide()'s result for the run (chaos/lyap/decide/{run}.npz as a dict), None if not computed"""
    f = os.path.join(OUT, "lyap", "decide", f"{run}.npz")
    return dict(np.load(f)) if os.path.exists(f) else None


@torch.no_grad()
def prop_growth(p, table, envt, codes, t0, t1, k=3, tol=2e-4, n_max=20, seed=LYAP_SEED):
    """frame-free growth (1 / (t1 - t0)) log sigma_1 (b,) of the propagator of steps t0 + 1 .. t1 along the observation
    codes (b, >= t1) from the trained start state (the states before t0 are not stored): block power iteration on Phi^T
    Phi (k vectors; tangent() forward, adjoint() back, both rescaled every step so that long windows cannot underflow;
    MGS2, Rayleigh-Ritz) until no probe's log sigma_1 moves by more than tol. For stationary Jacobians E log sigma_1(Phi_n)
    is subadditive in n (||AB|| <= ||A|| ||B||), so E (1/n) log sigma_1(Phi_n) >= lambda_1 for every n and decreases to it
    (Fekete; Furstenberg-Kesten): after a burn-in it is an upper estimate of lambda_1 that needs no frame alignment.
    Returns the rates (b,), the last change of log sigma_1 (b,), the iterations"""
    Whh = p["LSTM.weight_hh_l0"]
    H = Whh.shape[1]
    b = len(envt)
    h = p["STM"].reshape(1, H).expand(b, H).contiguous()
    c = p["LTM"].reshape(1, H).expand(b, H).contiguous()
    G, C = [], []
    for t in range(t1):
        if t >= t0:
            C.append(c)
        h, c, gts = lstm_step(table[envt, codes[:, t]], h, c, Whh)
        if t >= t0:
            G.append(gts)
    gen = torch.Generator(device=Whh.device).manual_seed(seed)
    V = mgs2(torch.randn(b, k, 2 * H, device=Whh.device, generator=gen))[0]
    prev = None
    for it in range(n_max):
        dh, dc = V[..., :H], V[..., H:]
        logs = torch.zeros(b, device=Whh.device, dtype=torch.float64)
        for j in range(t1 - t0):
            dh, dc = tangent(dh, dc, C[j], G[j], Whh)
            nrm = torch.cat([dh, dc], -1).norm(dim=-1).amax(-1)
            dh, dc = dh / nrm[:, None, None], dc / nrm[:, None, None]
            logs += torch.log(nrm.double())
        ls = _ritz_top(torch.cat([dh, dc], -1)) + logs.cpu().numpy()
        for j in reversed(range(t1 - t0)):
            dh, dc = adjoint(dh, dc, C[j], G[j], Whh)
            nrm = torch.cat([dh, dc], -1).norm(dim=-1).amax(-1)
            dh, dc = dh / nrm[:, None, None], dc / nrm[:, None, None]
        V = mgs2(torch.cat([dh, dc], -1))[0]
        ch = np.full(b, np.inf) if prev is None else np.abs(ls - prev)
        if ch.max() < tol:
            break
        prev = ls
    return ls / (t1 - t0), ch, it + 1


@torch.no_grad()
def lam_frames(dev=None):
    """the 300-step estimate's frame dependence (M1 b): lambda_1 on the stored sampled sequences (LONG_SEED, steps
    101-300) from 4 more frames (LYAP_SEED + 2 .. + 5), 500 probes, at the Trained RNN's 1,680 / 4,200 / 12,000 / end
    and power 1's end -> chaos/lyap/checks.json, lam1_frames300_{run}_{raw}: the probe means, the stored frame first"""
    dev = ti._device(dev)
    out = {}
    for run, raw in (("trained", 2100), ("trained", 5250), ("trained", 15000), ("trained", 49995), ("trained_p1", 49995)):
        z = dict(np.load(os.path.join(OUT, "lyap", f"{run}_{raw:05d}.npz")))
        m = [float(z["lam"][:, 0].mean())]
        for fs in range(2, 6):
            m.append(float(lyap(run, raw, dev, smax=False, chunk=500, parts=("lam",),
                                frame_seed=LYAP_SEED + fs)["lam"][:, 0].mean()))
        out[f"lam1_frames300_{run}_{raw}"] = m
        print(run, raw, np.round(m, 5), flush=True)
    _merge_checks(out)
    return out


@torch.no_grad()
def bound_check(dev=None, n_probe=5):
    """prop_growth() vs the exact SVD of the explicit propagator of steps 501-1,500 (float64 tangent maps applied to the
    identity, renormalised every 50 steps), N = 50 and N = 200 at the end, the first n_probe probes of lyap_rows(), the
    lam_long sequences -> chaos/lyap/checks.json, bound_vs_exact_svd_{run}: (exact, power iteration) per probe"""
    dev = ti._device(dev)
    P = ti.probes()
    rows = lyap_rows()[:n_probe]
    out = {}
    for run in ("trained_N50", "trained_N200"):
        raw = ti.checkpoints(run)[-1]
        w = ti.load(run, raw)
        envt = torch.as_tensor(P["env"][rows].astype(np.int64), device=dev)
        codes = torch.as_tensor(long_codes(rows, LONG2_BURN + LONG2_AVG, LONG2_SEED), device=dev)
        p32 = {kk: v.to(dev, torch.float32) for kk, v in w.items()}
        pw = prop_growth(p32, ti._table(p32, P["Z"], dev), envt, codes, LONG2_BURN, LONG2_BURN + LONG2_AVG)[0]
        p = {kk: v.to(dev, torch.float64) for kk, v in w.items()}
        table = ti._table(p32, P["Z"], dev).double()
        Whh = p["LSTM.weight_hh_l0"]
        H = Whh.shape[1]
        h = p["STM"].reshape(1, H).expand(len(rows), H).contiguous()
        c = p["LTM"].reshape(1, H).expand(len(rows), H).contiguous()
        Q = torch.eye(2 * H, device=dev, dtype=torch.float64)[None].repeat(len(rows), 1, 1)    # rows = images of e_j
        logs = torch.zeros(len(rows), device=dev, dtype=torch.float64)
        for t in range(LONG2_BURN + LONG2_AVG):
            h2, c2, gts = lstm_step(table[envt, codes[:, t]], h, c, Whh)
            if t >= LONG2_BURN:
                dh, dc = tangent(Q[..., :H], Q[..., H:], c, gts, Whh)
                Q = torch.cat([dh, dc], -1)
                if (t - LONG2_BURN) % 50 == 49:
                    nrm = Q.flatten(1).norm(dim=1)
                    Q, logs = Q / nrm[:, None, None], logs + torch.log(nrm)
            h, c = h2, c2
        ex = ((torch.linalg.svdvals(Q)[:, 0].log() + logs) / LONG2_AVG).cpu().numpy()
        out[f"bound_vs_exact_svd_{run}"] = [ex.tolist(), pw.tolist()]
        print(run, "exact", ex.round(6), "power iteration", pw.round(6), "max |diff|", f"{np.abs(ex - pw).max():.1e}",
              flush=True)
    _merge_checks(out)
    return out


def long_jobs(part="all"):
    """lam_long's checkpoints (run, raw, with the frame-free bound). 'late': the Trained RNN at every stored checkpoint
    from LONG2_FROM episodes (the bound at every third and the last) and at 80 / 880 / 1,680 / 3,000 episodes (with the
    bound); power 1 at every stored checkpoint from LONG2_FROM (with the bound); the ESN and N = 200 / 100 / 50 at the
    end (with the bound: the comparison with their 300-step estimates). 'early': the other checkpoints of the Trained
    RNN and power 1 (no bound), so that their lines in a and c use one estimator. 'all': both. Ends first"""
    late, early = [], []
    for run in ("trained", "trained_p1"):
        have = [e for e in ti.checkpoints(run) if os.path.exists(os.path.join(OUT, "lyap", f"{run}_{e:05d}.npz"))]
        lt = [e for e in have if gap.train_x(e) >= LONG2_FROM]
        for i, e in enumerate(lt):
            late.append((0 if i == len(lt) - 1 else 2, run, e, run != "trained" or i % 3 == 0 or i == len(lt) - 1))
        if run == "trained":
            for xw in (80., 880., 1680., 3000.):
                late.append((1, run, min(have, key=lambda e_: abs(gap.train_x(e_) - xw)), True))
        done = {j[2] for j in late if j[1] == run}
        early += [(3, run, e, False) for e in have if e not in done]
    for run in ("echo", "trained_N200", "trained_N100", "trained_N50"):
        late.append((1, run, ti.checkpoints(run)[-1], True))
    jobs = dict(late=late, early=early, all=late + early)[part]
    jobs.sort(key=lambda j: j[0])
    return [j[1:] for j in jobs]


@torch.no_grad()
def lam_long(dev=None, part="all"):
    """the asymptotic exponents where the top ones are nearly degenerate (M1 b): one new 1,500-step draw per probe
    (LONG2_SEED), QR of 2 tangent vectors from each of 4 frames (LONG2_FRAMES), the mean of log |R_jj| over steps
    501-1,500 (asymptotic()); and the frame-free upper estimate (1/1,000) log sigma_1 of the propagator of steps
    501-1,500 on the same sequences (prop_growth()), at long_jobs(part)'s checkpoints, 500 probes -> chaos/lyap/long/
    {run}_{raw:05d}.npz: lam (4 frames, 500, 2), upper (500,), upper_change, upper_iters (where computed). Stops before
    the next checkpoint when chaos/lyap/STOP exists"""
    dev = ti._device(dev)
    d = os.path.join(OUT, "lyap", "long")
    os.makedirs(d, exist_ok=True)
    rows = lyap_rows()
    P = ti.probes()
    jobs = long_jobs(part)
    print(f"{len(jobs)} checkpoints", flush=True)
    for run, raw, with_upper in jobs:
        if os.path.exists(os.path.join(OUT, "lyap", "STOP")):
            print("STOP file: stopping", flush=True)
            return
        f = os.path.join(d, f"{run}_{raw:05d}.npz")
        if os.path.exists(f):
            continue
        t0 = time.time()
        lam = np.stack([lyap(run, raw, dev, smax=False, chunk=500, parts=("lam",), rows=rows, long_seed=LONG2_SEED,
                             frame_seed=fs, n_long=LONG2_BURN + LONG2_AVG, burn=LONG2_BURN)["lam"] for fs in LONG2_FRAMES])
        out = dict(rows=rows, raw=raw, x=gap.train_x(raw), lam=lam, frames=np.array(LONG2_FRAMES), seed=LONG2_SEED,
                   burn=LONG2_BURN, avg=LONG2_AVG)
        msg = ""
        if with_upper:
            w = ti.load(run, raw)
            p = {kk: v.to(dev, torch.float32) for kk, v in w.items()}
            table = ti._table(p, P["Z"], dev)
            envt = torch.as_tensor(P["env"][rows].astype(np.int64), device=dev)
            codes = torch.as_tensor(long_codes(rows, LONG2_BURN + LONG2_AVG, LONG2_SEED), device=dev)
            ch_ = 100 if p["LSTM.weight_hh_l0"].shape[1] > 200 else len(rows)
            up, upc, upi = np.zeros(len(rows)), np.zeros(len(rows)), 0
            for b0 in range(0, len(rows), ch_):
                sl = slice(b0, min(len(rows), b0 + ch_))
                up[sl], upc[sl], it = prop_growth(p, table, envt[sl], codes[sl], LONG2_BURN, LONG2_BURN + LONG2_AVG)
                upi = max(upi, it)
            out.update(upper=up, upper_change=upc, upper_iters=upi)
            msg = f"; bound {up.mean():+.5f} (max {up.max():+.5f}, change {upc.max():.0e}, iters {upi})"
            del p, table
            torch.cuda.empty_cache()
        tmp = f[:-4] + ".tmp.npz"
        np.savez(tmp, **out)
        os.replace(tmp, f)
        m = lam[..., 0].mean(1)
        print(f"{run} {raw} ({gap.train_x(raw):,.0f}): lam1 frames {m.round(5)}, lam2 {lam[..., 1].mean():+.5f}{msg}; "
              f"{time.time() - t0:.0f} s", flush=True)


def long_load(run):
    """the run's chaos/lyap/long/ files in checkpoint order: x, raw, lam (K, 4 frames, n, 2), upper (K, n; NaN where not
    computed), rows"""
    d = os.path.join(OUT, "lyap", "long")
    eps = [e for e in ti.checkpoints(run) if os.path.exists(os.path.join(d, f"{run}_{e:05d}.npz"))]
    if not eps:
        return None
    Z = [dict(np.load(os.path.join(d, f"{run}_{e:05d}.npz"))) for e in eps]
    n = len(Z[0]["rows"])
    return dict(x=np.array([gap.train_x(e) for e in eps]), raw=np.array(eps), rows=Z[0]["rows"],
                lam=np.stack([z["lam"] for z in Z]),
                upper=np.stack([z["upper"] if "upper" in z else np.full(n, np.nan) for z in Z]))


@torch.no_grad()
def horizon_check(dev=None):
    """is the asymptotic regime the episode's? Trained RNN at the end and at 1,680, ESN at the end, 500 probes: the
    state's scale (mean |u|, mean |c| over the units and probes) after 1, 10, 30 (episodes) and 1, 10, 30, 100, 300
    steps (the 300-step sampled sequences of M1 b), and the frame-free 30-step amplification (amplification()) over
    steps 1-30 and 6-30 of the episodes and 1-30, 6-30, 31-60, 101-130, 271-300 of the sampled sequences ->
    chaos/lyap/checks.json, horizon_{run}_{raw}"""
    dev = ti._device(dev)
    P = ti.probes()
    rows = lyap_rows()
    out = {}
    for run, raw in (("trained", 49995), ("trained", 2100), ("echo", 49995)):
        w = ti.load(run, raw)
        p = {kk: v.to(dev, torch.float32) for kk, v in w.items()}
        Whh = p["LSTM.weight_hh_l0"]
        table = ti._table(p, P["Z"], dev)
        envt = torch.as_tensor(P["env"][rows].astype(np.int64), device=dev)
        ep = torch.as_tensor((P["obs"][rows] * (1 << np.arange(NO))).sum(-1).astype(np.int64), device=dev)
        lc = torch.as_tensor(long_codes(rows, LONG_BURN + LONG_AVG), device=dev)
        res = {}
        for lab, codes, steps, wins in (("episodes", ep, (1, 10, 30), ((0, 30), (5, 30))),
                                        ("sampled", lc, (1, 10, 30, 100, 300),
                                         ((0, 30), (5, 30), (30, 60), (100, 130), (270, 300)))):
            su, sc = np.zeros((len(steps), len(rows))), np.zeros((len(steps), len(rows)))
            amp = np.zeros((len(wins), len(rows)))
            for b0 in range(0, len(rows), 125):
                sl = slice(b0, min(len(rows), b0 + 125))
                G, U, C = _replay(p, table, envt[sl], codes[sl])
                for i, s in enumerate(steps):
                    su[i, sl] = U[s].abs().mean(-1).cpu().numpy()
                    sc[i, sl] = C[s].abs().mean(-1).cpu().numpy()
                for i, (a0, a1) in enumerate(wins):
                    amp[i, sl] = amplification(G, C, Whh, a0, a1)[0]
                del G, U, C
            res[lab] = dict(steps=list(steps), mean_abs_u=su.mean(1).tolist(), mean_abs_c=sc.mean(1).tolist(),
                            windows=[f"{a0 + 1}-{a1}" for a0, a1 in wins], amp=amp.mean(1).tolist(),
                            amp_share_positive=(amp > 0).mean(1).tolist())
        out[f"horizon_{run}_{raw}"] = res
        print(run, raw, json.dumps(res), flush=True)
    _merge_checks(out)
    return out


RUN_STYLE = {"trained": ("Trained RNN (N = 1000)", "#1f77b4"), "trained_p1": ("N = 1000, power 1", "#6baed6"),
             "trained_N200": ("N = 200", "#2ca02c"), "trained_N100": ("N = 100", "#9467bd"),
             "trained_N50": ("N = 50", "#8c564b"), "echo": ("ESN", "#ff7f0e")}


def sweep_load(run):
    """the run's sweep files in checkpoint order: x (training episodes), and the stored arrays stacked over checkpoints"""
    d = os.path.join(OUT, "sweep")
    eps = [e for e in ti.checkpoints(run) if os.path.exists(os.path.join(d, f"{run}_{e:05d}.npz"))]
    if not eps:
        return None
    Z = [dict(np.load(os.path.join(d, f"{run}_{e:05d}.npz"))) for e in eps]
    out = {k: np.stack([z[k] for z in Z]) for k in ("ftle", "ftle_late", "loc1", "locsum", "loc_mean", "chi")}
    out.update(x=np.array([gap.train_x(e) for e in eps]), rows=Z[0]["rows"], n_total=len(ti.checkpoints(run)))
    return out


def _env_boot(v, rows):
    """environment means of per-probe values v (..., n) -> value (...), 5 / 95% over the shared resamples"""
    env = ti.probes()["env"][rows]
    boot = gap.boot_counts().astype(np.float64)
    S_ = np.stack([v[..., env == e].sum(-1) for e in range(gap.N_ENV)], -1)                 # (..., 50)
    C_ = np.bincount(env, minlength=gap.N_ENV).astype(np.float64)
    val = S_.sum(-1) / C_.sum()
    bt = (S_ @ boot.T) / (boot @ C_)                                                        # (..., 1,000)
    return val, np.percentile(bt, 5, -1), np.percentile(bt, 95, -1)


LAGS = (1, 2, 5, 10, 20)
FIXED_FLIPS = tuple(a for a, s in enumerate(FLIP_S) if s <= 10)                 # flips at steps 1, 3, 5, 10: every lag
ADF_LS = (0, (6, 2, 1.5, 2))                                                    # ADF's reference lines (dash-dot)


def lyap_load(run):
    """the run's chaos/lyap/ files in checkpoint order: x (training episodes), raw, the stored arrays stacked over
    checkpoints, smax where stored (x_smax), rows, n_total"""
    d = os.path.join(OUT, "lyap")
    eps = [e for e in ti.checkpoints(run) if os.path.exists(os.path.join(d, f"{run}_{e:05d}.npz"))]
    if not eps:
        return None
    Z = [dict(np.load(os.path.join(d, f"{run}_{e:05d}.npz"))) for e in eps]
    out = {k: np.stack([z[k] for z in Z]) for k in ("amp", "amp_change", "amp_iters", "lam", "decay_rate",
                                                     "decay_logd")}
    sm = [i for i, z in enumerate(Z) if "smax" in z]
    x = np.array([gap.train_x(e) for e in eps])
    out.update(x=x, raw=np.array(eps), rows=Z[0]["rows"], n_total=len(ti.checkpoints(run)),
               smax=np.stack([Z[i]["smax"] for i in sm]) if sm else None, x_smax=x[sm],
               smax_change=np.stack([Z[i]["smax_change"] for i in sm]) if sm else None)
    return out


def lam_curve(run, z=None):
    """lambda_1 through training as drawn (figure a, c) and quoted: for the Trained RNN and power 1 lam_long's estimate
    (per probe the mean over its 4 frames) at every checkpoint where it exists (their top exponents are nearly degenerate
    late in training), for the other runs the 300-step estimate (chaos/lyap/). Value and 5-95% over the environment
    resamples; for lam_long's the interval is widened to the union of the 4 frames' intervals. Returns x, val, lo, hi,
    is_long"""
    z = lyap_load(run) if z is None else z
    val, lo, hi = _env_boot(z["lam"][..., 0], z["rows"])
    is_long = np.zeros(len(z["x"]), bool)
    zl = long_load(run) if run in ("trained", "trained_p1") else None
    for j, x in enumerate([] if zl is None else zl["x"]):
        assert np.array_equal(zl["rows"], z["rows"])
        k = int(np.flatnonzero(z["x"] == x)[0])
        v_, l_, h_ = _env_boot(zl["lam"][j, :, :, 0].mean(0), zl["rows"])
        fr = [_env_boot(zl["lam"][j, f, :, 0], zl["rows"]) for f in range(zl["lam"].shape[1])]
        val[k], lo[k], hi[k] = v_, min([l_] + [a[1] for a in fr]), max([h_] + [a[2] for a in fr])
        is_long[k] = True
    return z["x"], val, lo, hi, is_long


def _recheck_stats(v, dz=None):
    """lam_recheck's result for one run (a lam1_recheck_{run} entry): the candidates, those positive in all draws (robust,
    the strict rule) with their checkpoints and draw means, mean - 2 SEM > 0, the median across-draw SD, the chance
    expectation n / 2^draws. With lam_decide's draws (dz = decide_load(run)) the decision (dec): per candidate the mean
    and SEM over its 64 draws; chaotic (chaos: mean - 2 SEM > 0) with their checkpoints and means, contracting (mean + 2
    SEM < 0), undecided (neither), and the stricter mean - 3 SEM > 0 (n_chaos3)"""
    D = v["draws"]
    e = np.array(v["est"], dtype=np.float64).reshape(-1, D)
    x, n = np.array(v["x"]), v["n"]
    orig = np.array(v["orig"])
    bnd = np.array(v.get("bound", [np.nan] * n), dtype=np.float64)
    bo = (orig <= 0) & (bnd >= 0)                                                           # candidates by the bound only
    out = dict(n=n, robust=0, x_robust=np.zeros(0), mean_robust=np.zeros(0), mean_max=np.nan, sd_median=np.nan,
               two_sem=0, mean_pos=0, chance=n / 2 ** D, x=x, orig=orig, bound=bnd, n_bound_only=int(bo.sum()),
               bound_only=bo, means=e.mean(1) if n else np.zeros(0), robust_mask=(e > 0).all(1), npos=(e > 0).sum(1),
               row=np.array(v["row"], dtype=np.int64), draws=D, dec=dz is not None)
    if n:
        m = e.mean(1)
        sem = e.std(1, ddof=1) / np.sqrt(D)
        rb = (e > 0).all(1)
        out.update(robust=int(rb.sum()), x_robust=x[rb], mean_robust=m[rb], mean_max=float(m.max()),
                   sd_median=float(np.median(e.std(1, ddof=1))), two_sem=int((m - 2 * sem > 0).sum()),
                   mean_pos=int((m > 0).sum()), sem_robust=sem[rb])
    if dz is not None:
        assert np.array_equal(dz["x"], x) and np.array_equal(dz["row"], np.array(v["row"], dtype=np.int64))
        e1 = dz["est"][..., 0]                                                              # (n, 64) lambda_1 per draw
        m1, s1 = e1.mean(1), e1.std(1, ddof=1) / np.sqrt(e1.shape[1])
        ch, co = m1 - 2 * s1 > 0, m1 + 2 * s1 < 0
        out.update(D_dec=e1.shape[1], est_dec=e1, m_dec=m1, sem_dec=s1, sd_dec=e1.std(1, ddof=1), npos_dec=(e1 > 0).sum(1),
                   lam2_dec=dz["est"][..., 1].mean(1), chaos=ch, contracting=co, n_chaos=int(ch.sum()),
                   n_contracting=int(co.sum()), n_undecided=int((~ch & ~co).sum()), n_chaos3=int((m1 - 3 * s1 > 0).sum()),
                   x_chaos=x[ch], mean_chaos=m1[ch], sem_chaos=s1[ch])
    return out


def _decide_numbers(s, run):
    """numbers()'s lines for lam_decide's decision of one run (s = _recheck_stats(v, decide_load(run))) and for its
    float64 control (decide/{run}_float64.npz, where computed)"""
    from scipy.stats import t as t_dist, norm
    D, n = s["D_dec"], s["n"]
    head = f"    decision (lam_decide: {D} draws x steps {DECIDE_BURN + 1:,}-{DECIDE_N:,}, each with its own frame): "
    if not n:
        return head + "no candidates"
    m, se, ch, co, rb = s["m_dec"], s["sem_dec"], s["chaos"], s["contracting"], s["robust_mask"]
    un = ~ch & ~co
    p0 = t_dist.sf(2., D - 1)
    head = (f"    strict rule's power at lambda_1 = +.005 (normal approximation, the {s['draws']} draws' median SD "
            f"{s['sd_median']:.4f}): {norm.cdf(.005 / s['sd_median']) ** s['draws']:.2f}\n" + head)
    out = [head + f"chaotic (mean - 2 SEM > 0) {s['n_chaos']} of {n}, mean - 3 SEM > 0: {s['n_chaos3']}; contracting "
           f"(mean + 2 SEM < 0) {s['n_contracting']} (means up to {m[co].max():+.4f}); undecided {int(un.sum())}"
           + (" (" + ", ".join(f"{x_:,.0f}/{r_}: {a:+.4f} +- {b:.4f}, draws {lo:+.3f} to {hi:+.3f}" for x_, r_, a, b, lo, hi
                               in zip(s["x"][un], s["row"][un], m[un], se[un], s["est_dec"][un].min(1),
                                      s["est_dec"][un].max(1))) + ")" if un.any() else "")
           + f"; per-draw SD median {np.median(s['sd_dec']):.4f}, SEM median {np.median(se):.4f}; chance at lambda_1 = 0: "
           f"{p0:.3f} per candidate"
           + (f", {p0 * (ch | un).sum():.2f} expected if all {int((ch | un).sum())} chaotic and undecided candidates "
              "were at 0" if (ch | un).any() else "")
           + f"; candidates without a positive draw among the {s['draws']} earlier ones: {int((s['npos'] == 0).sum())}"
           + (f"; the bound-only candidates' 64-draw means {m[s['bound_only']].min():+.4f} to "
              f"{m[s['bound_only']].max():+.4f}" if s["n_bound_only"] else "")]
    if s["n_chaos"]:
        xs = np.unique(s["x_chaos"])
        weak = m < .005
        out.append(f"      chaotic at {len(xs)} checkpoints, training {xs.min():,.0f}-{xs.max():,.0f} ("
                   + ", ".join(f"{x_:,.0f}: {int((s['x_chaos'] == x_).sum())}" for x_ in xs) + f"); means "
                   f"{m[ch].min():+.4f} to {m[ch].max():+.4f}, t {(m / se)[ch].min():.1f} to {(m / se)[ch].max():.1f}, "
                   f"lambda_2 {s['lam2_dec'][ch].min():+.4f} to {s['lam2_dec'][ch].max():+.4f}; the strict rule (all "
                   f"{s['draws']} earlier draws positive) passed {int((rb & ch).sum())} of them and "
                   f"{int((rb & ~ch).sum())} others; chaotic with mean < +.005: "
                   f"{int((ch & weak).sum())}, of which the strict rule passed {int((ch & weak & rb).sum())}")
        out.append(f"      chaotic (x/row: mean +- SEM, t, draws > 0; earlier {s['draws']} draws > 0): " + "; ".join(
            f"{x_:,.0f}/{r_}: {a:+.4f} +- {b:.4f}, t {a / b:.1f}, {k}/{D}; {k8}/{s['draws']}" for x_, r_, a, b, k, k8 in
            zip(s["x"][ch], s["row"][ch], m[ch], se[ch], s["npos_dec"][ch], s["npos"][ch])))
    f64 = decide_load(f"{run}_float64")
    if f64 is not None:
        assert np.array_equal(f64["x"], s["x"]) and np.array_equal(f64["row"], s["row"])
        e64 = f64["est"][..., 0]
        m64, se64 = e64.mean(1), e64.std(1, ddof=1) / np.sqrt(D)
        ch64 = m64 - 2 * se64 > 0
        dm = m64 - m
        dif = np.flatnonzero(ch64 != ch)
        out.append(f"      float64 control (the same draws and frames): chaotic {int(ch64.sum())} of {n}; decisions that "
                   "differ: " + (", ".join(f"{s['x'][i]:,.0f}/{s['row'][i]} (t {m[i] / se[i]:.2f} -> {m64[i] / se64[i]:.2f})"
                                           for i in dif) if len(dif) else "none")
                   + f"; change of the candidates' means (float64 - float32) {dm.mean():+.5f} +- "
                   f"{dm.std(ddof=1) / np.sqrt(n):.5f} (|change| max {np.abs(dm).max():.4f}; in units of the candidate's "
                   f"SEM max {np.abs(dm / se).max():.1f}); single draws within 1e-4: "
                   f"{np.mean(np.abs(e64 - s['est_dec']) < 1e-4):.2f}")
    return "\n".join(out)


def _r_rows(a, b):
    """the within-episode Pearson r over the last axis (steps), per row"""
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))


def chi_rise(chi, lag):
    """the paired rise chi(s + lag, s) - chi(s, s), averaged over the fixed flips (steps 1, 3, 5, 10) per probe: chi
    (..., len(FLIP_S), T) -> (...)"""
    return np.mean([chi[..., a, FLIP_S[a] - 1 + lag] - chi[..., a, FLIP_S[a] - 1] for a in FIXED_FLIPS], 0)


def sweep_figure():
    """the chaos measures through training (partial data are drawn as they are; the footnote counts the checkpoints)"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    FL, TS = fs["label"], 12.
    Ly = {r: lyap_load(r) for r in LYAP_RUNS}
    Ly = {r: v for r, v in Ly.items() if v is not None}
    S = {r: sweep_load(r) for r in RUNS}
    S = {r: v for r, v in S.items() if v is not None}
    B = dict(np.load(os.path.join(OUT, "pilot", "bayes_side.npz")))
    A = dict(np.load(os.path.join(OUT, "lyap", "adf.npz")))
    chk_path = os.path.join(OUT, "lyap", "checks.json")
    chk = json.load(open(chk_path)) if os.path.exists(chk_path) else {}
    rc = {r: chk.get(f"lam1_recheck_{r}") for r in Ly}
    st = {r: _recheck_stats(v, decide_load(r)) for r, v in rc.items() if v}
    acc = gap.load("jumps")
    acc0 = float(acc["c0"][list(acc["names"]).index("goal accuracy at T")])
    fig = plt.figure(figsize=(19.5, 12.2))
    gsp = fig.add_gridspec(2, 4, hspace=.62, wspace=.36, left=.05, right=.985, top=.83, bottom=.19)
    order = [r for r in ("trained_N50", "trained_N100", "trained_N200", "trained_p1", "echo", "trained") if r in Ly]
    short = {"trained": "Trained RNN", "echo": "ESN"}

    def xaxis(ax):
        ax.set_xscale("log")
        ax.set_xlim(8, 7e4)
        ax.set_xticks([10, 1e2, 1e3, 1e4])
        ax.set_xticklabels(["0", "100", "1k", "10k"])
        ax.minorticks_off()
        ax.set_xlabel("training episodes", fontsize=FL)
        ax.axvline(acc0, color="k", lw=1, ls=":")
        ax.spines[["top", "right"]].set_visible(False)

    def band(ax, x, v, rows, color, lw=2., ls="-", label=None, z=3):
        val, lo, hi = _env_boot(v, rows)
        ax.fill_between(x, lo, hi, color=color, alpha=.22, lw=0, zorder=z - 1)
        ax.plot(x, val, color=color, lw=lw, ls=ls, label=label, zorder=z)
        return val, lo, hi

    def lw_(run):
        return 2.2 if run in RUNS else 1.5
    # a: the asymptotic top exponent (the chaos criterion)
    ax = fig.add_subplot(gsp[0, 0])
    for run in order:
        x_, v_, lo_, hi_, _ = lam_curve(run, Ly[run])
        ax.fill_between(x_, lo_, hi_, color=RUN_STYLE[run][1], alpha=.22, lw=0, zorder=2)
        ax.plot(x_, v_, color=RUN_STYLE[run][1], lw=lw_(run), zorder=3, label=RUN_STYLE[run][0])
    ax.axhline(0, color="k", lw=1.2, zorder=1)
    ax.text(9, .135, "chaotic above 0; Bayes, naive: exactly 0", fontsize=FL - 1, va="center", zorder=6,
            bbox=dict(facecolor="w", edgecolor="none", pad=1.))
    dots = [r for r in order if r in st and st[r]["dec"] and st[r]["n_chaos"]]
    dodge = {"trained_N200": .87, "trained_N100": 1., "trained_N50": 1.15}        # sideways by network, same checkpoints
    for r in dots:                                    # single chaotic probes at their 64-draw means
        ax.scatter(st[r]["x_chaos"] * dodge.get(r, 1.), st[r]["mean_chaos"], s=14, color=RUN_STYLE[r][1], edgecolor="k",
                   lw=.4, zorder=5)
    if dots:
        ax.text(6.5e4, .072, "dots: single chaotic probes, shifted\nsideways by network; "
                r"64 draws: mean $-$ 2 SEM $>$ 0", fontsize=FL - 1.5,
                ha="right", va="center", linespacing=1.3)
    ax.set_ylim(-.66, .165)
    ax.set_yticks([-.4, -.3, -.2, -.1, 0.])
    xaxis(ax)
    ax.set_ylabel(r"$\lambda_1$ (nats / step)", fontsize=FL)
    ax.set_title("a   Asymptotic top Lyapunov exponent\n"
                 r"$\lambda_1$ = mean $\log|R_{t,11}|$ (QR) along sampled" "\n"
                 "sequences, beyond the 30 trained steps", loc="left", fontsize=TS)
    h_, l_ = ax.get_legend_handles_labels()
    idx = [l_.index(RUN_STYLE[r][0]) for r in ("trained", "trained_p1", "echo", "trained_N200", "trained_N100",
                                               "trained_N50") if RUN_STYLE[r][0] in l_]
    lg = ax.legend([h_[i] for i in idx], [l_[i] for i in idx], ncol=2, loc="lower center", bbox_to_anchor=(.5, -.01),
                   frameon=True, facecolor="w", edgecolor="none", framealpha=1., fontsize=FL - 1.5, columnspacing=1.2,
                   handlelength=1.5, labelspacing=.3)
    lg.set_zorder(6)
    # b: the finite-time amplification over an episode (frame-free)
    ax = fig.add_subplot(gsp[0, 1])
    for run in order:
        z = Ly[run]
        band(ax, z["x"], z["amp"][..., 0], z["rows"], RUN_STYLE[run][1], lw=lw_(run), label=RUN_STYLE[run][0])
        if run in RUNS:
            band(ax, z["x"], z["amp"][..., 1], z["rows"], RUN_STYLE[run][1], lw=1.3, ls=(0, (3, 1.6)))
    ax.axhline(0, color="k", lw=1.2, zorder=1)
    a_adf = A["amp"].mean(0)
    ax.axhline(a_adf[0], color="k", lw=1.1, ls=ADF_LS, zorder=1)
    ax.text(9, a_adf[0] + .012, f"factorized-state Bayes: {a_adf[0]:+.3f}", fontsize=FL - 1, va="bottom")
    ax.set_ylim(-.5, .16)
    xaxis(ax)
    ax.set_ylabel(r"$(1/T)\,\log\sigma_1(\Phi)$ (nats / step)", fontsize=FL)
    ax.set_title("b   Largest growth over an episode\n"
                 r"$\frac{1}{T}\log\sigma_1(\Phi)$, $\Phi = J_T\cdots J_1$; dashed: steps 6-30", loc="left", fontsize=TS)
    h_, l_ = ax.get_legend_handles_labels()
    keep = [i for i, lab in enumerate(l_) if lab not in (RUN_STYLE["trained"][0], RUN_STYLE["echo"][0])][::-1]
    ax.legend([h_[i] for i in keep], [l_[i] for i in keep], loc="center right", bbox_to_anchor=(1., .45),
              frameon=False, fontsize=FL - 1)
    # c: decay times (tau_1 beyond the trained horizon, its frame-free bound; tau_flip within episodes)
    ax = fig.add_subplot(gsp[0, 2])
    for run in ("echo", "trained"):
        if run not in Ly:
            continue
        z = Ly[run]
        col = RUN_STYLE[run][1]
        x_, v, lo, hi, _ = lam_curve(run, z)
        ax.fill_between(x_, -1. / lo, -1. / np.minimum(hi, -1e-9), color=col, alpha=.22, lw=0)
        ax.plot(x_, -1. / v, color=col, lw=2.2)
        v, lo, hi = _env_boot(z["decay_rate"], z["rows"])
        ax.fill_between(z["x"], -1. / lo, -1. / np.minimum(hi, -1e-9), color=col, alpha=.22, lw=0)
        ax.plot(z["x"], -1. / v, color=col, lw=1.6, ls=(0, (3, 1.6)))
    zl = long_load("trained")
    hb = []
    if zl is not None:
        m = (zl["x"] >= LONG2_FROM) & np.isfinite(zl["upper"]).all(1)
        ub = _env_boot(zl["upper"][m], zl["rows"])[0]
        ax.plot(zl["x"][m], -1. / ub, color=RUN_STYLE["trained"][1], lw=1.5, ls=":", zorder=4)
        hb = [Line2D([], [], color=RUN_STYLE["trained"][1], lw=1.5, ls=":",
                     label=r"upper bound on $\tau_1$ (frame-free)")]
    ax.set_yscale("log")
    yt = [2, 5, 10, 20, 50, 100]
    ax.set_ylim(1.8, 230)
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.set_yticklabels([str(t) for t in yt])
    ax.yaxis.set_minor_locator(NullLocator())
    xaxis(ax)
    ax.set_ylabel("decay time (steps)", fontsize=FL)
    ax.set_title("c   Decay times of a state perturbation\n"
                 r"solid: $\tau_1 = -1/\lambda_1$ (a), beyond the 30 trained steps" "\n"
                 r"dashed: $\tau_{\mathrm{flip}} = -1/\kappa$ (h), within episodes", loc="left", fontsize=TS)
    lg = ax.legend(handles=[Line2D([], [], color=RUN_STYLE[r][1], lw=2.2, label=short[r]) for r in RUNS] + hb,
                   loc="upper left", frameon=True, facecolor="w", edgecolor="none", framealpha=1., fontsize=FL - 1.5,
                   borderaxespad=.1)
    lg.set_zorder(6)
    # d: one-step expansion vs entanglement (one measure, one window for every line)
    ax = fig.add_subplot(gsp[0, 3])
    ent_l = A["ent"][:, SMAX_T0:]
    for run in RUNS:
        if run not in Ly or Ly[run]["smax"] is None:
            continue
        z = Ly[run]
        band(ax, z["x_smax"], _r_rows(z["smax"], ent_l[None]), z["rows"], RUN_STYLE[run][1], label=short[run])
    radf = float(np.nanmean(_r_rows(A["smax"], ent_l)))
    ax.axhline(radf, color="k", lw=1.1, ls=ADF_LS)
    ax.text(9, radf - .015, f"factorized-state Bayes: {radf:+.2f}", fontsize=FL - 1, va="top")
    ax.axhline(0, color="k", lw=.8)
    xaxis(ax)
    ax.set_ylabel("r within episodes (steps 6-30)", fontsize=FL)
    ax.set_title("d   One-step expansion vs entanglement\n"
                 r"r($\log\sigma_{\max}(J_t)$, entanglement$_t$) in episodes", loc="left", fontsize=TS)
    ax.legend(loc="center right", frameon=False, fontsize=FL - 1.5)
    # e, f: the paired rise of one flipped observation's impact (fixed flips)
    lag_cols = plt.cm.viridis(np.linspace(.1, .9, len(LAGS)))
    refs = {nm: chi_rise(B[f"chi_{nm}"].mean(-1).mean(-2), 20).mean() for nm in ("bayes", "adf")}
    ref_h = [Line2D([], [], color="k", lw=1.1, label=f"Bayes, lag 20 ({refs['bayes']:+.2f})"),
             Line2D([], [], color="k", lw=1.1, ls=ADF_LS, label=f"factorized-state Bayes, lag 20 ({refs['adf']:+.2f})")]
    for j, run in enumerate(RUNS):
        if run not in S:
            continue
        ax = fig.add_subplot(gsp[1, j])
        z = S[run]
        chi = z["chi"].astype(np.float64).mean(-1)                                          # (K, n, nS, T)
        for c_, L_ in zip(lag_cols, LAGS):
            band(ax, z["x"], chi_rise(chi, L_), z["rows"], c_, lw=1.8, label=f"lag {L_}")
        ax.axhline(refs["bayes"], color="k", lw=1.1)
        ax.axhline(refs["adf"], color="k", lw=1.1, ls=ADF_LS)
        ax.axhline(0, color="k", lw=.8)
        ax.set_ylim(-.35, 3.75)
        xaxis(ax)
        ax.set_ylabel(r"$\chi(s + \mathrm{lag}, s) - \chi(s, s)$ (nats)", fontsize=FL)
        ax.set_title(f"{'ef'[j]}   {short[run]}: growth of one flip's impact\n"
                     r"$\chi = d_H$(intact, flipped marginals); flips at steps 1-10", loc="left", fontsize=TS)
        h_, l_ = ax.get_legend_handles_labels()
        if j == 0:
            leg = ax.legend(h_, l_, loc="upper left", bbox_to_anchor=(0., .87), frameon=False, fontsize=FL - 1)
            ax.add_artist(leg)
        lg = ax.legend(handles=ref_h, loc="upper right", bbox_to_anchor=(1., .87), frameon=True, fontsize=FL - 1,
                       facecolor="w", edgecolor="none", framealpha=1.)
        lg.set_zorder(6)
    # g: the immediate impact through training
    ax = fig.add_subplot(gsp[1, 2])
    for run in RUNS:
        if run not in S:
            continue
        z = S[run]
        chi = z["chi"].astype(np.float64).mean(-1)
        d = np.stack([chi[:, :, a, s - 1] for a, s in enumerate(FLIP_S)], -1).mean(-1)
        band(ax, z["x"], d, z["rows"], RUN_STYLE[run][1], label=short[run])
    for nm, lab, col, ls, side in (("bayes", "Bayes", "k", "-", "below"), ("naive", "naive", "0.5", "-", "left"),
                                   ("adf", "factorized-state Bayes", "k", ADF_LS, "above")):
        c0 = B[f"chi_{nm}"].mean(-1)
        d = np.mean([np.nanmean(c0[:, a, :, s - 1]) for a, s in enumerate(FLIP_S)])
        ax.axhline(d, color=col, lw=1.1, ls=ls)
        x_, ha = (9, "left") if side == "left" else (6.5e4, "right")
        ax.text(x_, d + (-.05 if side == "below" else .04), f"{lab} {d:.2f}", fontsize=FL - 1.5, ha=ha,
                va="top" if side == "below" else "bottom", color="0.35" if nm == "naive" else "k")
    ax.set_ylim(-.1, 3.6)
    xaxis(ax)
    ax.set_ylabel(r"$\chi(s, s)$ (nats)", fontsize=FL)
    ax.set_title("g   Immediate impact of one flipped observation\n"
                 r"$\chi(s, s) = d_H$(intact, flipped marginals), 7 flips", loc="left", fontsize=TS)
    ax.legend(loc="lower right", frameon=False, fontsize=FL - 1)
    # h: the flip's state difference by lag (raw decay curves)
    ax = fig.add_subplot(gsp[1, 3])
    show = []
    if "trained" in Ly:
        z = Ly["trained"]
        want = (10., 320., 1680., z["x"][-1])
        ks = [int(np.argmin(np.abs(np.log(z["x"]) - np.log(w)))) for w in want]
        blues = ("#9ecae1", "#4292c6", "#2166ac", "#08306b")
        show += [("trained", k, c_) for k, c_ in zip(ks, blues)]
    if "echo" in Ly:
        show.append(("echo", len(Ly["echo"]["x"]) - 1, RUN_STYLE["echo"][1]))
    lagx = np.arange(DECAY_LAGS[1] + 1)
    for run, k, c_ in show:
        z = Ly[run]
        v, lo, hi = _env_boot(np.moveaxis(z["decay_logd"][k], 0, -1), z["rows"])
        ax.fill_between(lagx, np.exp(lo), np.exp(hi), color=c_, alpha=.25, lw=0)
        ax.plot(lagx, np.exp(v), color=c_, lw=2)
        xl = f"{z['x'][k]:,.0f}" if z["x"][k] > 10 else "0"
        ax.text(lagx[-1] + .4, np.exp(v[-1]), f"{short[run]} {xl}", fontsize=FL - 1, va="center",
                color="#6baed6" if c_ == "#9ecae1" else c_)
    ax.set_yscale("log")
    ax.set_xlim(0, 28)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("lag after the flipped observation (steps)", fontsize=FL)
    ax.set_ylabel(r"$\|\Delta(u, c)\|$ (geometric mean)", fontsize=FL)
    ax.axvspan(DECAY_LAGS[0], DECAY_LAGS[1], color="k", alpha=.06, lw=0)
    ax.text(12.5, .97, r"lags of $\kappa$", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=FL - 1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("h   A flip's state difference decays\n"
                 r"$\|(u,c)_{\mathrm{flipped}} - (u,c)_{\mathrm{intact}}\|$ by lag", loc="left", fontsize=TS)
    lam_end = {r: float(lam_curve(r, Ly[r])[1][-1]) for r in RUNS if r in Ly}
    chaotic = [RUN_STYLE[r][0] for r in ("trained_N200", "trained_N100", "trained_N50")
               if r in st and st[r]["dec"] and st[r]["n_chaos"]]
    who = ", ".join(chaotic[:-1]) + " and " + chaotic[-1] if len(chaotic) > 1 else "".join(chaotic)
    fig.suptitle("Chaos through training. The N = 1000 networks are never chaotic: their state's asymptotic Lyapunov "
                 f"exponent stays below 0 (end: Trained RNN {lam_end.get('trained', np.nan):+.3f}, ESN "
                 f"{lam_end.get('echo', np.nan):+.3f} nats / step).\n"
                 + (f"{who}: a few probes are weakly chaotic in mid-training (a). " if chaotic else "")
                 + "The Trained RNN's state still grows along some direction within an episode (b).\nDriven beyond the 30 "
                 "trained steps, its state's decay time jumps, shortens while it passes naive, then lengthens (c). A flip's "
                 "impact on its beliefs grows after the flip during learning, not at the end (e)",
                 fontsize=TS + 1, y=.985, linespacing=1.5)
    cnt = ", ".join(f"{RUN_STYLE[r][0]}: {len(z['x'])}" for r, z in Ly.items())
    cs = ", ".join(f"{short.get(r, r)}: {len(z['x_smax'])}" for r, z in Ly.items() if z["smax"] is not None)
    rec = ""
    small = [r for r in ("trained_N200", "trained_N100", "trained_N50") if r in st]
    if len(st) == len(Ly) and all(s["dec"] for s in st.values()) and "trained" in st:
        lab = {"trained": "Trained RNN", "trained_p1": "power 1", "echo": "ESN", "trained_N200": "N = 200",
               "trained_N100": "N = 100", "trained_N50": "N = 50"}

        def part(r):
            s = st[r]
            t = f"{lab[r]}: {s['n_chaos']} of {s['n']}"
            if s["n_chaos"]:
                lo_, hi_ = s["mean_chaos"].min(), s["mean_chaos"].max()
                xs_ = f"{s['x_chaos'].min():,.0f}-{s['x_chaos'].max():,.0f}" if s["n_chaos"] > 1 else \
                    f"{s['x_chaos'][0]:,.0f}"
                t += (f" ($\\lambda_1$ {lo_:+.3f}" + (f" to {hi_:+.3f}" if s["n_chaos"] > 1 else "")
                      + f", {xs_} episodes)")
            return t
        none = [lab[r] for r in ("trained_p1", "echo") if r in st and not st[r]["n"]]
        mc = np.concatenate([st[r]["mean_chaos"] for r in st])
        rbc = np.concatenate([st[r]["robust_mask"][st[r]["chaos"]] for r in st])
        weak = mc < .005
        rec = ("\na, chaos test: each candidate (a positive 300-step estimate of a probe's $\\lambda_1$, or a frame-free "
               f"bound $\\geq$ 0) was re-estimated with {DECIDE_DRAWS} independent {DECIDE_N:,}-step draws (steps "
               f"{DECIDE_BURN + 1:,}-{DECIDE_N:,}), each with its own frame;\nchaotic if their mean $-$ 2 SEM $>$ 0 "
               "(dots): " + part("trained") + (f" ({', '.join(none)}: no candidates)" if none else "") + ", "
               + ", ".join(part(r) for r in small[:-1]) + ",\n" + part(small[-1])
               + f". The first rule (all {RECHECK_DRAWS} of {RECHECK_DRAWS} earlier {RECHECK_N:,}-step draws positive) "
               "counted " + " / ".join(str(st[r]["robust"]) for r in ["trained"] + small) + " (Trained RNN / "
               + " / ".join(lab[r] for r in small).replace(" / N = ", " / ") + "), missing "
               f"{int((weak & ~rbc).sum())} of the {int(weak.sum())} chaotic candidates with $\\lambda_1$ below +0.005.")
    fig.text(.5, .08, "Data, a-d and h: 500 probes (10 per test environment x 50 environments) from the trained start "
             "state; b, d, h: their 30-step episodes. a, c: $\\lambda_1$ along one sampled observation sequence per "
             "probe (its own cell), beyond the 30 trained steps:\nTrained RNN and power 1: 1,500 steps, the first 500 "
             "discarded, mean of 4 frames (band: the union of the frames' bands); the others: 300 steps, the first 100 "
             "discarded. c, dotted: (1/1,000) $\\log\\sigma_1$ of the propagator of steps 501-1,500 (no frame; "
             "$\\geq\\lambda_1$).\n"
             f"Checkpoints drawn (of 114): {cnt}; d: {cs}. e-g: 1,000 probes (20 per environment), every checkpoint, "
             "flips of each of the 5 observations at steps 1, 3, 5, 10 (e, f) and also 15, 20, 25 (g).\n"
             "Bands: 5-95% over 1,000 environment resamples; dotted vertical: the Trained RNN passes naive. Bayes and "
             "naive: all exponents exactly 0 in log coordinates (T1); networks: Euclidean (u, c); factorized-state "
             "Bayes: its log marginals." + rec, ha="center", va="center", fontsize=FL - 1, linespacing=1.6)
    path = fig_path("td_tt_chaos_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


def _corr_within(a, b):
    """mean over episodes of the within-episode Pearson r over steps (a, b: (n, S))"""
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return float(np.nanmean(r))


def _fmt(v, lo, hi, f="+.3f"):
    return f"{v:{f}} [{lo:{f}}, {hi:{f}}]"


def numbers():
    """every number of the docstring's RESULTS"""
    for name in ("checks.json", os.path.join("lyap", "checks.json")):
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            print(f"== {name}")
            for k_, v in json.load(open(path)).items():
                if k_.startswith("lam1_positive_recheck_"):
                    o, a, b = (np.array(v[q]) for q in ("orig", "draw", "long"))
                    print(f"  {k_} (the first recheck, superseded: one redraw): {v['n']} probe-checkpoints with "
                          f"lambda_1 > 0 at {v['checkpoints']} checkpoints"
                          + (f" (training {min(v['x']):,.0f}-{max(v['x']):,.0f}; max +{o.max():.4f}); another draw, "
                             f"steps 101-300: {v['still_positive_draw']} stay > 0 (max {a.max():+.4f}); 1,000 steps, "
                             f"steps 301-1,000: {v['still_positive_long']} stay > 0 (max {b.max():+.4f}); both: "
                             f"{int(((a > 0) & (b > 0)).sum())}" if v["n"] else ""))
                elif k_.startswith("lam1_recheck_"):
                    run = k_[len("lam1_recheck_"):]
                    s = _recheck_stats(v, decide_load(run))
                    line = (f"  {k_}: {s['n']} candidates (300-step estimate > 0, or a frame-free bound >= 0: "
                            f"{s['n_bound_only']} by the bound only)")
                    if s["n"]:
                        line += (f" at {len(np.unique(s['x']))} checkpoints (training {s['x'].min():,.0f}-"
                                 f"{s['x'].max():,.0f}; max {s['orig'].max():+.4f}); {v['draws']} draws x steps "
                                 f"{v['burn'] + 1}-{v['n_long']}, each with its own frame: positive in all "
                                 f"{v['draws']}: {s['robust']} (chance at lambda_1 = 0: {s['chance']:.2f})")
                        if s["robust"]:
                            xs = np.unique(s["x_robust"])
                            line += (f" at {len(xs)} checkpoints, training {xs.min():,.0f}-{xs.max():,.0f} "
                                     f"({', '.join(f'{x_:,.0f}' for x_ in xs)}); their draw means "
                                     f"{s['mean_robust'].min():+.4f} to {s['mean_robust'].max():+.4f} (SEM of the "
                                     f"largest {s['sem_robust'][np.argmax(s['mean_robust'])]:.4f})")
                        line += (f"; draw mean - 2 SEM > 0: {s['two_sem']}; draw mean > 0: {s['mean_pos']}; draw means "
                                 f"max {s['mean_max']:+.4f}; across-draw SD median {s['sd_median']:.4f}")
                        if s["n_bound_only"]:
                            bo = s["bound_only"]
                            line += (f"; the {int((~bo).sum())} 300-step candidates' draw means max "
                                     f"{s['means'][~bo].max():+.4f}; the bound-only candidates (training {s['x'][bo].min():,.0f}-"
                                     f"{s['x'][bo].max():,.0f}, rows {sorted(set(np.array(v['row'])[bo].tolist()))}): "
                                     f"bounds {s['bound'][bo].min():+.5f} to {s['bound'][bo].max():+.5f}, draw means "
                                     f"{s['means'][bo].min():+.4f} to {s['means'][bo].max():+.4f}, positive draws "
                                     f"{int((np.array(v['est'])[bo] > 0).sum())} of {int(bo.sum()) * v['draws']}")
                    print(line)
                    if s["dec"]:
                        print(_decide_numbers(s, run))
                elif k_.startswith("horizon_"):
                    print(f"  {k_}:")
                    for lab in ("episodes", "sampled"):
                        h = v[lab]
                        print(f"    {lab:8s} mean |u| / |c| after " + ", ".join(
                            f"{s_}: {u_:.3f} / {c_:.3f}" for s_, u_, c_ in zip(h["steps"], h["mean_abs_u"],
                                                                             h["mean_abs_c"]))
                              + "; 30-step amplification " + ", ".join(
                                  f"steps {w_}: {a_:+.4f} ({100 * p_:.0f}% > 0)"
                                  for w_, a_, p_ in zip(h["windows"], h["amp"], h["amp_share_positive"])))
                elif k_.startswith("amp_vs_exact_svd_"):
                    dif = max(abs(e - p_) for w in v for e, p_ in zip(w[0], w[1]))
                    print(f"  {k_}: (exact, power iteration) per window {v}; max |difference| {dif:.1e}")
                else:
                    print(f"  {k_}: {v}")
    lc = os.path.join(OUT, "lyap", "checks.json")
    lchk = json.load(open(lc)) if os.path.exists(lc) else {}
    S = {r: _recheck_stats(lchk[f"lam1_recheck_{r}"], decide_load(r)) for r in LYAP_RUNS if f"lam1_recheck_{r}" in lchk}
    S = {r: s for r, s in S.items() if s["dec"]}
    if S:
        ch = np.concatenate([s["chaos"] for s in S.values()])
        rb = np.concatenate([s["robust_mask"] for s in S.values()])
        weak = np.concatenate([s["m_dec"] < .005 for s in S.values()])
        print(f"== chaos decision, all runs ({' / '.join(S)}): {len(ch)} candidates; chaotic (64-draw mean - 2 SEM > 0) "
              f"{int(ch.sum())} ({' / '.join(str(s['n_chaos']) for s in S.values())}); the strict rule (all 8 earlier "
              f"draws positive) {int(rb.sum())} ({' / '.join(str(s['robust']) for s in S.values())}), of which chaotic "
              f"{int((rb & ch).sum())}; chaotic with mean < +.005: {int((ch & weak).sum())}, of which the strict rule "
              f"passed {int((ch & weak & rb).sum())}")
    B = dict(np.load(os.path.join(OUT, "pilot", "bayes_side.npz")))
    print("== observers, flips (1,000 probes; chi = d_H of the marginals after one flipped observation, mean of the 2 "
          "variables; rises over the flips at steps 1, 3, 5, 10)")
    for nm in ("naive", "bayes", "adf"):
        c0 = B[f"chi_{nm}"].mean(-1).mean(-2)                                                  # (n, nS, T)
        imm = np.mean([np.nanmean(c0[:, a, s - 1]) for a, s in enumerate(FLIP_S)])
        print(f"  {nm:6s} chi(s, s) {imm:.3f} (mean of the 7 flips); chi(s + lag, s) - chi(s, s): "
              + ", ".join(f"lag {L_} {chi_rise(c0, L_).mean():+.3f}" for L_ in LAGS))
    print(f"  joint Bayes: the flipped observation's joint range osc(d) {B['jrange'].mean():.3f}; Bayes's chi(s, s) by "
          "flip step " + " / ".join(f"{np.nanmean(B['chi_bayes'].mean(-1)[:, a, :, s - 1]):.2f}"
                                    for a, s in enumerate(FLIP_S)))
    P = ti.probes()
    L, _ = tables()
    print(f"  the likelihood tables: L = p(x_o = 1 | r0, r1) in [{L.min():.3f}, {L.max():.3f}]")
    rows = B["rows"]
    ll = step_loglik(P["obs"][rows], L[P["env"][rows]])
    la = adf_run(ll)
    J, _ = adf_jacobians(ll, np.concatenate([np.full((len(rows), 1, 2, R), -np.log(R)), la[:, :-1]], 1))
    fr_ = np.array([qr_lyap(J, seed=FRAME_SEED + i).mean(1).mean(0)[:2] for i in range(4)])
    print(f"  ADF random-frame 30-step QR exponents (superseded), 4 frames: top {fr_[:, 0].min():+.3f} to "
          f"{fr_[:, 0].max():+.3f}, 2nd {fr_[:, 1].min():+.3f} to {fr_[:, 1].max():+.3f}; sum of the 18 "
          f"{B['adf_local'].mean(1).sum(-1).mean():+.3f} (= mean sum_k log(1 - rho_k^2), frame-free); mean log(1 + "
          f"rho_1) {np.log1p(B['adf_rho'][..., 0]).mean():.3f}")
    print(f"  ADF r within episodes with entanglement (1,000 probes): log(1 + rho_1) steps 2-30 "
          f"{_corr_within(np.log1p(B['adf_rho'][:, 1:, 0]), B['ent'][:, 1:]):+.3f}, steps 6-30 "
          f"{_corr_within(np.log1p(B['adf_rho'][:, 5:, 0]), B['ent'][:, 5:]):+.3f}; QR local exponent 1, steps 6-30 "
          f"{_corr_within(B['adf_local'][:, 5:, 0], B['ent'][:, 5:]):+.3f}")
    A = dict(np.load(os.path.join(OUT, "lyap", "adf.npz")))
    ra = A["rows"]
    ent_l = A["ent"][:, SMAX_T0:]
    print("== ADF, state measures (500 probes; log-marginal quotient, orthonormal zero-sum coordinates)")
    for j, w in enumerate(("1-30", "6-30")):
        v = _env_boot(A["amp"][:, j], ra)
        print(f"  amplification steps {w}: {_fmt(*v)} (positive {np.mean(A['amp'][:, j] > 0):.3f})")
    v = _env_boot(A["lam"][:, 0], ra)
    print(f"  asymptotic lambda_1 {_fmt(*v, f='+.4f')}, lambda_2 {A['lam'][:, 1].mean():+.4f}; probes with lambda_1 > 0 "
          f"{int((A['lam'][:, 0] > 0).sum())}, max {A['lam'][:, 0].max():+.4f}; mean rho_1 along the sampled sequences at "
          "steps 1 / 30 / 100 / 300: " + " / ".join(f"{A['rho1_long'][t]:.3f}" for t in (0, 29, 99, 299)))
    print(f"  per-step log sigma_max (steps 6-30) mean {A['smax'].mean():.3f}; r within episodes, steps 6-30: log "
          f"sigma_max {np.nanmean(_r_rows(A['smax'], ent_l)):+.3f}, log(1 + rho_1) "
          f"{np.nanmean(_r_rows(np.log1p(A['rho1'][:, SMAX_T0:]), ent_l)):+.3f}, QR local exponent 1 "
          f"{np.nanmean(_r_rows(A['loc1'][:, SMAX_T0:], ent_l)):+.3f}")
    print("== networks, pilot (1,000 probes): the superseded random-frame QR exponents, and the flips")
    for run in RUNS:
        for raw in PILOT_RAW:
            f = os.path.join(OUT, "pilot", f"{run}_{raw:05d}.npz")
            if not os.path.exists(f):
                continue
            N = dict(np.load(f))
            late = N["local"].astype(np.float64)[:, 5:].mean(1)                              # (n, K)
            c0 = N["chi"].astype(np.float64).mean(-1).mean(-2)
            imm = np.mean([np.nanmean(c0[:, a, s - 1]) for a, s in enumerate(FLIP_S)])
            print(f"  {run:7s} {gap.train_x(raw):7,.0f}: QR top 'FTLE' steps 1-30 {N['local'][..., 0].mean():+.3f}; "
                  f"steps 6-30: probes with any of the 32 positive {int((late > 0).any(1).sum())}, with the top positive "
                  f"{int((late[:, 0] > 0).sum())}, max {late.max():+.3f}; chi(s, s) {imm:.3f}; rise "
                  + ", ".join(f"lag {L_} {chi_rise(c0, L_).mean():+.3f}" for L_ in LAGS))
    print("== networks, state measures (500 probes; [5-95%] over the environment resamples; lambda_1, tau_1: the 300-step "
          "estimate, see the lam_long section for the Trained RNN and power 1)")
    sub = np.searchsorted(subset(), lyap_rows())
    for run in LYAP_RUNS:
        z = lyap_load(run)
        if z is None:
            continue
        zs = sweep_load(run)
        picks = [i for i, e in enumerate(z["raw"]) if (run in RUNS and e in PILOT_RAW) or i == len(z["raw"]) - 1]
        for i in picks:
            r_ = z["rows"]
            a1, a6 = _env_boot(z["amp"][i, :, 0], r_), _env_boot(z["amp"][i, :, 1], r_)
            l1 = _env_boot(z["lam"][i, :, 0], r_)
            kd = _env_boot(z["decay_rate"][i], r_)
            line = (f"  {run:12s} {z['x'][i]:7,.0f}: amp 1-30 {_fmt(*a1)} (>0 {np.mean(z['amp'][i, :, 0] > 0):.3f}), "
                    f"6-30 {_fmt(*a6)} (>0 {np.mean(z['amp'][i, :, 1] > 0):.3f}); lambda_1 (300 steps) {_fmt(*l1, f='+.4f')} "
                    f"(>0: {int((z['lam'][i, :, 0] > 0).sum())}, max {z['lam'][i, :, 0].max():+.4f}), lambda_2 "
                    f"{z['lam'][i, :, 1].mean():+.4f}; tau_1 {-1 / l1[0]:.1f} [{-1 / l1[1]:.1f}, {-1 / l1[2]:.1f}]; "
                    f"kappa {_fmt(*kd, f='+.4f')}, tau_flip {-1 / kd[0]:.1f}")
            ks = [] if zs is None else np.flatnonzero(zs["x"] == z["x"][i])
            if len(ks):
                qr1 = zs["ftle"][ks[0]][sub, 0].astype(np.float64)
                line += (f"; QR top 'FTLE' (same probes) {qr1.mean():+.3f}, amp - QR "
                         f"{np.mean(z['amp'][i, :, 0] - qr1):+.3f}")
            if z["smax"] is not None and z["x"][i] in z["x_smax"]:
                k_ = list(z["x_smax"]).index(z["x"][i])
                rr = _env_boot(_r_rows(z["smax"][k_], ent_l), r_)
                line += f"; log sigma_max(J_t) mean {z['smax'][k_].mean():+.3f}, r with entanglement {_fmt(*rr)}"
            print(line)
    print("== late asymptotic exponents (lam_long: one new 1,500-step draw per probe, steps 501-1,500, 4 frames; 500 "
          "probes; [5-95%] of the frame-mean over the environment resamples; the bound: (1/1,000) log sigma_1 of the "
          "propagator of steps 501-1,500, frame-free, >= lambda_1)")
    for run in LYAP_RUNS:
        zl, z = long_load(run), lyap_load(run)
        if zl is None:
            continue
        for j, x in enumerate(zl["x"]):
            r_ = zl["rows"]
            fm = zl["lam"][j, :, :, 0].mean(1)                                                 # (4,) frame means
            l1 = _env_boot(zl["lam"][j, :, :, 0].mean(0), r_)
            fr = [_env_boot(zl["lam"][j, f, :, 0], r_) for f in range(len(fm))]
            k = int(np.flatnonzero(z["x"] == x)[0])
            s1 = _env_boot(z["lam"][k, :, 0], r_)
            line = (f"  {run:12s} {x:7,.0f}: lambda_1 {_fmt(*l1, f='+.5f')}, frames {fm.min():+.5f} to {fm.max():+.5f} "
                    f"(spread {fm.max() - fm.min():.5f}), union of the frames' bands [{min(a[1] for a in fr):+.5f}, "
                    f"{max(a[2] for a in fr):+.5f}]; lambda_2 {zl['lam'][j, :, :, 1].mean():+.5f}; probes > 0 (frame "
                    f"mean) {int((zl['lam'][j, :, :, 0].mean(0) > 0).sum())}; tau_1 {-1 / l1[0]:.1f} (frames "
                    f"{-1 / fm.max():.1f}-{-1 / fm.min():.1f}); 300-step estimate {s1[0]:+.5f} (tau_1 {-1 / s1[0]:.1f})")
            if np.isfinite(zl["upper"][j]).all():
                u = _env_boot(zl["upper"][j], r_)
                line += (f"; bound {_fmt(*u, f='+.5f')} (max {zl['upper'][j].max():+.5f}, > 0: "
                         f"{int((zl['upper'][j] > 0).sum())}) -> tau_1 <= {-1 / u[0]:.1f}")
            print(line)
        if len(zl["x"]) > 1:
            ks = [int(np.flatnonzero(z["x"] == x)[0]) for x in zl["x"]]
            lm = zl["lam"][..., 0].mean((1, 2))                                                # (K,) frame-mean means
            bias = lm - z["lam"][ks, :, 0].mean(1)
            gap_ = lm - zl["lam"][..., 1].mean((1, 2))
            spread = np.ptp(zl["lam"][..., 0].mean(2), 1)
            hb = np.isfinite(zl["upper"]).all(1)
            late = zl["x"] >= LONG2_FROM
            early = "" if (~late).sum() > 12 else (
                f"; before {LONG2_FROM:,.0f} episodes (bias / lambda_1 - lambda_2) "
                + ", ".join(f"{x_:,.0f}: {b_:+.4f} / {g_:.4f}" for x_, b_, g_ in zip(zl["x"][~late], bias[~late], gap_[~late])))
            print(f"  {run}: {len(zl['x'])} checkpoints; long - 300-step estimate {bias.min():+.4f} to {bias.max():+.4f} "
                  f"(median {np.median(bias):+.4f}; max at {zl['x'][int(np.argmax(bias))]:,.0f}, where lambda_1 - "
                  f"lambda_2 is {gap_[int(np.argmax(bias))]:.4f}; from "
                  f"{LONG2_FROM:,.0f} episodes {bias[late].min():+.4f} to {bias[late].max():+.4f}{early}); lambda_1 - "
                  f"lambda_2 from {LONG2_FROM:,.0f} episodes {gap_[late].min():.4f} to {gap_[late].max():.4f}; frame "
                  f"spread max {spread.max():.5f} (at {zl['x'][int(np.argmax(spread))]:,.0f}); bound at "
                  f"{int(hb.sum())} checkpoints, per-probe max {np.nanmax(zl['upper']):+.5f}, bound - lambda_1 "
                  f"{(zl['upper'][hb].mean(1) - lm[hb]).min():+.4f} to {(zl['upper'][hb].mean(1) - lm[hb]).max():+.4f}")
    acc = gap.load("jumps")
    print("== through training (the drawn checkpoints; the values at training x); the Trained RNN passes naive at "
          f"{float(acc['c0'][list(acc['names']).index('goal accuracy at T')]):,.0f} episodes (td_tt_jumps, c0); "
          "lambda_1 and tau_1 as drawn (lam_curve: lam_long's for the Trained RNN and power 1, the 300-step estimate for "
          "the others)")
    show_x = (10, 20, 60, 80, 100, 120, 240, 360, 480, 720, 880, 1200, 1680, 3000, 4200, 7800, 12000, 20000, 39996)
    for run in LYAP_RUNS:
        z = lyap_load(run)
        if z is None:
            continue
        _, l1, _, _, isl = lam_curve(run, z)
        a1 = _env_boot(z["amp"][..., 0], z["rows"])
        a6 = _env_boot(z["amp"][..., 1], z["rows"])[0]
        kd = _env_boot(z["decay_rate"], z["rows"])[0]
        ks = [int(np.flatnonzero(z["x"] == x)[0]) for x in show_x if x in z["x"]]
        print(f"  {run} ({len(z['x'])} checkpoints, {int(isl.sum())} from lam_long; amp iterations max "
              f"{int(z['amp_iters'].max())}, last change max {z['amp_change'].max():.1e})")
        for lab, v, f_ in (("lambda_1", l1, "+.4f"), ("tau_1", -1 / l1, ".1f"), ("amp 1-30", a1[0], "+.4f"),
                           ("amp 6-30", a6, "+.3f"), ("tau_flip", -1 / kd, ".1f")):
            print(f"    {lab:9s} " + ", ".join(f"{z['x'][k]:,.0f}: {v[k]:{f_}}" for k in ks))
        print("    amp 1-30 [5-95%] " + ", ".join(f"{z['x'][k]:,.0f}: {_fmt(a1[0][k], a1[1][k], a1[2][k], '+.4f')}"
                                                for k in ks if z["x"][k] <= 120 or z["x"][k] in (3000, 39996)))
        i_late = z["x"] >= 1000
        print(f"    lambda_1 max {l1.max():+.4f} at {z['x'][int(np.argmax(l1))]:,.0f} (excluding the initial network: "
              f"{l1[1:].max():+.4f} at {z['x'][1 + int(np.argmax(l1[1:]))]:,.0f}), end {l1[-1]:+.4f}; after 1,000 "
              f"episodes min {l1[i_late].min():+.4f} at {z['x'][i_late][int(np.argmin(l1[i_late]))]:,.0f}; tau_1 min "
              f"after 100 episodes {(-1 / l1)[z['x'] > 100].min():.1f} at "
              f"{z['x'][z['x'] > 100][int(np.argmin((-1 / l1)[z['x'] > 100]))]:,.0f}; tau_flip min after 100 episodes "
              f"{(-1 / kd)[z['x'] > 100].min():.1f} at {z['x'][z['x'] > 100][int(np.argmin((-1 / kd)[z['x'] > 100]))]:,.0f}")
        neg = np.flatnonzero(a1[0] <= 0)
        print(f"    amp 1-30 > 0 at {int((a1[0] > 0).sum())} of {len(a1[0])} checkpoints"
              + (f", first at {z['x'][int(np.argmax(a1[0] > 0))]:,.0f}, at every checkpoint from "
                 + (f"{z['x'][neg.max() + 1]:,.0f} ({a1[0][neg.max() + 1]:+.4f})" if neg.max() + 1 < len(a1[0])
                    else "- (never)") + f"; lower 5% > 0 at {int((a1[1] > 0).sum())}"
                 if (a1[0] > 0).any() else f" (max {a1[0].max():+.3f})"))
        if z["smax"] is not None:
            rr = _env_boot(_r_rows(z["smax"], ent_l[None]), z["rows"])[0]
            print(f"    r(log sigma_max(J_t), entanglement) over {len(z['x_smax'])} checkpoints: {rr.min():+.3f} to "
                  f"{rr.max():+.3f} (max at {z['x_smax'][int(np.argmax(rr))]:,.0f}), end {rr[-1]:+.3f}; last "
                  f"power-iteration change max {z['smax_change'].max():.1e}")


if __name__ == "__main__":
    what = sys.argv[1:] or ["numbers"]
    if "check" in what:
        check()
    if "pilot" in what:
        pilot()
    if "sweep" in what:
        sweep()
    if "theory_checks" in what:
        theory_checks()
    if "adf_lyap" in what:
        adf_lyap()
    if "lyap_sweep" in what:
        lyap_sweep()
    if "lyap_check" in what:
        lyap_check()
    if "lam_recheck" in what:
        lam_recheck()
    if "lam_decide" in what:
        lam_decide()
    if "lam_decide_float64" in what:
        lam_decide(runs=("trained_N200", "trained_N100", "trained_N50"), dtype=torch.float64)
    if "lam_long" in what:
        lam_long()
    if "horizon_check" in what:
        horizon_check()
    if "lam_frames" in what:
        lam_frames()
    if "bound_check" in what:
        bound_check()
    if "sweep_figure" in what:
        sweep_figure()
    if "numbers" in what:
        numbers()
