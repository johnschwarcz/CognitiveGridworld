"""Exploration of the latent grid through training: how many of the 100 latent pairs (r0, r1) the networks' answers use,
whether the answers are folded (distant latent pairs given one answer), and how the beliefs move, from the initial
network to the end of training, against Bayes, memory-free Bayes, naive and chance (random answers) on the same
episodes. Revised 11-01 after an independent verification (chance levels drawn; every crossing has an environment-
bootstrap interval; single-checkpoint coincidences and the "global coverage" claim retracted) and 11-02 after a second
and a third one: every crossing is marked by whether noise alone could produce it; the unfolding rests on far errors
(truth-referenced: every noise tested raises them). Distant pairs merged falling below chance (1/100) is out of reach
of uniform noise but not of local answer jitter, so it counts only together with the falling far errors; its
naive-level crossing can come from noise (FAILED OR NULL); synthesis export added.
-> main/REPLICATION/figure_bin/td_tt_latent.png       (a-e: answers at T; f-h: within the episode; i, j: movement)
-> main/REPLICATION/figure_bin/td_tt_latent_fold.png  (a: one environment's map from true pair to answer at 5
   checkpoints; b-g: where the answers land, all environments)
Caches: main/DATA/experimental/training_dynamics/through_training/latent/ (refs.npz, probes_trained.npz,
probes_echo.npz, modal.npz, same_as_naive.npz; timeline.npz = this topic's half-way export; synthesis.npz = the
td_tt_gap export for the cross-topic synthesis figure).

DATA (read only)
  Probes: analysis/probes.npz, 50 environments (pairs of test variables) x 100 latent pairs x 2 trajectories = 10,000
    episodes x 30 steps; the networks' stored marginal log beliefs "logq" (float16) at all 114 checkpoints per network
    (analysis/<run>/ck_*.npz; trained = Trained RNN, echo = ESN; seed 20261029); both right from the stored float64
    "right". fp16 argmax ties: at most 0.19% (ESN 0.20%) of (probe, step, variable), 0.06% at T; the right-pair share
    from fp16 argmax differs from the stored correctness by at most 0.0007.
  Grid replays: summary_data/grids (td_grids_fwd.py), the same 50 environments x 100 pairs x 24 fresh trajectories at
    19 checkpoints per network, answers at T, and Bayes's and naive's answers on the same observations.
  References: Bayes = argmax of each exact marginal (log_jb); its joint MAP (100-cell argmax) gives the same numbers
    within 1% (distinct 93.2 vs 93.0, e^H 84.4 vs 83.9, both right .600 vs .595). Memory-free Bayes = softmax of the
    summed per-step exact marginal likelihoods (cumsum of inc_SW = windowed Bayes, w = 1). Naive = log_nb. Their
    argmax at T agrees with the probe set's stored right_bayes / right_window1 / right_naive in 100% of probes.
  x axis: training episodes = E - E // 5 (every 5th episode is a test episode), initial network at 10 labelled 0;
    dotted line: the landmark, the Trained RNN's goal accuracy at T passes naive's (.464) between the 800 (.441) and
    880 (.493) checkpoints, drawn at 880 (re-timed with the stays-past rule in CHECKS: 835 [798-876]).

DEFINITIONS (answer of variable v at step t: a_v = argmax of the observer's marginal belief; answer pair c^ = (a_0,
  a_1); posterior mean m_t = (E_q r0, E_q r1) under the observer's own marginals, in cells; "far" = 2+ cells off)
  Chance = random answers: every probe's answer pair drawn uniformly from the 100 cells, independently (h: at every
    step). A crossing can come from uniform noise when random answers lie past its level in the crossing's direction
    (noise_can_cross): replacing answers by random ones can then carry a curve across it (open dots). Local noise acts
    differently: jittering answers to nearby cells (what noise on the beliefs produces) lowers distant pairs merged
    (c) and the modal maps' far-apart merges and can carry them below chance; like uniform noise, it raises far errors
    and lowers both right. NUMBERS (noise tests) checks this with the references' answers and the networks' own
    answers (probes and grid replays): a share of them replaced by random answers, jittered locally, or taken from
    log beliefs with Gaussian noise added.
  td_tt_latent.png (a-h: per environment (its 200 probes), then averaged over the 50; i, j: per episode, then averaged)
  a  both variables right at T: P(c^ = c). Chance .010; noise lowers it, so its crossings cannot come from noise.
  b  far errors at T: P(max_v |a_v - r_v| >= 2), a variable 2+ cells off (= one far + both far in the fold figure).
     Chance .922. Truth-referenced, not fold-specific (measured against the true pair: a fold raises it, so does any
     large error); noise raises it, so its falls below naive's and memory-free's levels cannot come from noise.
  c  distant pairs merged at T: over the 18,432 pairs of probes of one environment whose latent pairs are far apart
     (max_v |r_vi - r_vj| >= 2; 4,608 cell pairs x 2 x 2 trajectories), the share given the same answer pair; log
     axis. A fold merges distant pairs. Random answers merge them at exactly 1/100 and the truth at 0. Uniform noise
     (a share p of the answers replaced by random ones) moves it toward 1/100 from either side, an exact mixture
     (1 - p)^2 m + (1 - (1 - p)^2) / 100: falling below naive's .0184 can come from noise (naive with 10% of its
     answers random: .0167, while its far errors rise to .500; open dot). Local answer jitter (what noise on the
     beliefs produces) lowers it for a partly folded observer and can cross 1/100: the Trained RNN's own answers at
     800 (.0120) give .0097 with +-1 jitter at P .5, while their far errors rise from .293 to .398 (naive's answers
     stay at .0119 or above under every noise tested). Every noise tested raises far errors, so falling below 1/100
     (866) is noise-proof only together with b's falling far errors (.293 at 800, .241 at 880): half-filled dot.
  d  answer cells used at T, D_{e,T}: the number of distinct c^ among environment e's 200 probes at T. Truth 100.
     Chance 86.6 = 100 (1 - .99^200), above memory-free Bayes (77.1) and naive (57.3); the truth plus independent +-1
     jitter (P .3 per variable, no folds by construction) scores 91.9. Noise raises D (naive with 10% of its answers
     random: 63.7), so its crossings of naive's and memory-free's levels can come from noise (open dots).
  e  effective number e^H, H = -Sigma_k p_k log p_k, p_k = share of e's 200 answers at T at cell k (plug-in). Truth
     100; chance 75.5 (simulated, 40,000 sets of 200); jitter (as in d) 82.4. Noise raises it (naive with 10% of its
     answers random: 46.5): open dots.
  f, g  D_{e,t} at every step, averaged over environments; white line: the step at which this environment average
     peaks (a statistic of the average, labelled so); right strips: Bayes / memory-free Bayes / naive (their peaks at
     30 / 7 / 7, white circles).
  h  answer cells lost later in the episode: per environment, max_t D_{e,t} - D_{e,30}, then averaged (Bayes's 1.9 is
     the noise floor of a maximum over 30 noisy counts). Chance (random answers at every step): 5.550, exact (E[max of
     30 iid D] - E[D], D = distinct cells among 200 uniform draws, from the occupancy distribution) = naive's level
     (5.56), so falling below naive's can come from noise (open dot). Noise moves it toward 5.55 or above (naive 5.56
     -> 6.05-6.90, memory-free 5.04 -> 5.38-6.46, Bayes 1.90 -> 2.77-4.64), so falling below the naive-Bayes midpoint
     (3.73) cannot.
  i  path of the posterior mean, Sigma_{t=2..30} |m_t - m_{t-1}| per episode; dotted: the path of the probe-averaged
     trajectory (the drift shared by all episodes; Bayes .12).
  j  back-and-forth: the path (i) minus the net move |m_30 - m_1| (an additive part of i). Noise in the beliefs raises
     it (iid Gaussian noise of SD 1 on Bayes's log beliefs: 6.45 -> 17.3), so its rise is not noise-proof.
  Dots with bars (a-e, h; fold figure f): the crossing = the training episode after which the curve stays past the
     level to the end of training, log-interpolated between checkpoints, with its 5-95% interval over 1,000 resamples
     of the 50 environments (references recomputed on the same resampled environments, i.e. paired). Two events are
     ordered only when their intervals do not overlap. Filled: uniform random answers cannot produce the crossing,
     and no other noise tested does; half-filled (c, below 1/100; JITTER_CROSSES): uniform random answers cannot, local
     jitter can, so it counts only together with b's falling far errors; open: random answers lie past the level, so
     noise alone can produce the crossing.
  td_tt_latent_fold.png
  a  environment 14 (variables 7 and 45), chosen by rule as naive's median number of modal answers (45; quartiles
     over the 50 environments 39.25-50; not chosen by eye): each pair's modal answer at T (most frequent over its 24
     trajectories; ties, at most 7.6% of pairs (ESN 6.4%, Bayes 2.8%, naive 6.7%), go to the lowest cell index); grey
     dot = the true pair, arrow = to its answer, circle area = pairs answered there. Titles: distinct modal answers
     here and their mean over all 50 environments. Bayes's map beside the Trained RNN, naive's beside the ESN.
  b-e  where each probe's answer at T lands: the true pair; a neighbour (max_v |a_v - r_v| = 1); one variable far
     (max_v >= 2, min_v <= 1); both far (min_v >= 2). They add up to 1 (b = a and d + e = b of the first figure).
     Chance profile [.010 .068 .403 .518] (exact for uniform answers and uniform truth).
  f  distinct modal answers per environment (of the 100 pairs; 24 trajectories per pair), averaged over the 50.
     Chance: a random map (each pair's answer uniform) gives 63.4 = 100 (1 - .99^100). Random answers through this
     pipeline give 57.4 distinct modal answers and 64.0 far-apart merges (100 x 50 maps; the re-verifier's 300 maps:
     57.8 / 62.7), because ties (67% of pairs for random answers, at most 7.6% for the observers) go to the lowest
     cell; with ties broken at random they give 63.4 / 46.0: the random map is the tie-free level. Open dot: naive's
     level (44.3), which random answers lie past; filled: the random map's (no noise tested carries distinct modal
     answers past it: 62.5 at most).
  g  on the 3,590 (environment, pair) cells where naive's modal answer is wrong and Bayes's right: P(network's modal
     answer = naive's); dotted control: naive's wrong answer for the same pair in another environment (20 random
     environment permutations, counts pooled over them; keeps pair-specific biases such as the pull to the grid's
     edge, drops the environment's own fold).
  timeline.npz (this topic's own half-way export, kept for the record; the synthesis figure uses synthesis.npz):
     x (114,) training episodes (initial network at 10), episode (114,) raw; one row per measure and network (5
     measures x Trained RNN, ESN): names, run, family, value / lo / hi (114,) = the environment-mean curve and its
     5-95% band over the 1,000 resamples, bayes, naive, chance, level, half / half_lo / half_hi (the half-way crossing
     by the stays-past rule above; NaN = never), p_never, scale, half_rule, note. Half-way level: half of the curve's
     own change from the initial network to the end of training (mean of the last 3 checkpoints); on the log scale
     for distant pairs merged; for cells lost later (a hump), half of the way back from its maximum to its end.
     Measures: both right at T, far errors at T, distant pairs merged at T, answer cells lost later (family
     'answers'); back-and-forth (family 'movement'). note: three of the five half-way levels can be crossed by noise
     alone and are not noise-proof events: distant pairs merged (levels .0325 / .107, above naive's .0184; uniform
     noise and local jitter both lower it), cells lost later (10.77 / 14.7, above chance 5.55, toward which noise moves
     it) and back-and-forth (noise raises it).

NUMBERS (python td_tt_latent.py numbers; training episodes; crossings as 'episode [5-95%]')
  Chance: D 86.60, e^H 75.46, categories [.0100 .0684 .4032 .5184], far errors .922, both right .010, distant pairs
    merged .010, cells lost later 5.550 (random answers at every step; simulated 5.54 +- .07); random map: 63.4
    distinct modal answers, 46.1 far-apart merges, modal answer 2+ off .922 (random answers through the modal pipeline:
    57.4 / 64.0; ties broken at random: 63.4 / 46.0); truth + +-1 jitter (no folds): D 91.9, e^H 82.4.
  Noise tests (means of 5 draws; back-and-forth 2): naive's answers at T with 10% / 30% of them random: far errors
    .500 / .590 (naive .452), distant pairs merged .0167 / .0141 (.0184), D 63.7 / 73.2 (57.3), e^H 46.5 / 55.4
    (41.8), both right .155 / .122 (.171); jittered locally (+-1 at P .3 / .9, uniform -2..2 / -3..3): merged .0148 /
    .0123 / .0119 / .0122, never below 1/100, far errors .494-.797. Memory-free Bayes (.0085): .0082-.0102 (the widest
    jitter takes it just past 1/100 from below); Bayes (.0015): .0028-.0090. Random answers on the probe set: far
    errors .9215, merged .0100, D 86.6, e^H 75.4. Cells lost later with noise at every step (10% / 30% random, jitter
    +-1 at P .3): naive 5.56 -> 6.90 / 6.80 / 6.05; memory-free 5.04 -> 5.38 / 5.72 / 6.46; Bayes 1.90 -> 2.77 / 3.86
    / 4.64. Back-and-forth with iid Gaussian noise (SD .5 / 1 / 2) on the log beliefs at every step: Bayes 6.45 ->
    10.6 / 17.3 / 30.3, naive 8.05 -> 12.6 / 20.2 / 35.0.
  Noise tests on the networks' own answers at T (5 draws, one seed per checkpoint and kind; 10% / 30% random, jitter
    +-1 at P .3 / .5 / .9, triangular -2..2, uniform -2..2 / -3..3, iid Gaussian noise of SD 1 / 2 on the log beliefs
    at T; noise-free far errors and merged = the cached curves, max |diff| 0): every kind raises far errors and lowers
    both right at every checkpoint tested. Uniform replacement follows the exact mixture toward 1/100 within Monte
    Carlo error (800, 10% / 30%: .0117 / .0110 vs .0116 / .0110; the largest gap, 880 at 10%, .00952 vs .00967, is 1.9
    SE of the 5-draw mean, SE .00008: printed with every kind). Trained RNN, distant pairs merged (far errors)
    without noise -> range over the 10 kinds:
    640 .0171 (.393) -> .0106-.0158, never below 1/100; 720 .0137 (.335) -> .0099-.0129, below 1/100 only with uniform
    -2..2 jitter (.0099, far errors .649); 800 .0120 (.293) -> .0095-.0117, below 1/100 with +-1 jitter at P .5,
    triangular and uniform -2..2 jitter (.0097 / .0095 / .0096; far errors .398 or more); 880 .0096 (.241) ->
    .0086-.0100 (lowest .0086 with +-1 jitter at P .5, far errors .348; .0088 at P .3). Gaussian noise on the log
    beliefs lowers it too (800: .0109 / .0103 at SD 1 / 2; 880: .0091 / .0086). ESN at 20,000: .0189 (.455) ->
    .0121-.0174, far errors .468-.798. The Trained RNN's grid replays (every stream's answer; 5 draws; 10 / 30 / 60 /
    90% random, jitter +-1 at P .3 / .5 / .9, uniform -2..2): far-apart merges at 880 (47.6) 47.0 / 47.5 / 47.2 /
    59.0 with replacement (all answers random: 64.0), 43.6 / 38.5 / 44.9 / 51.2 with jitter, below the random map's
    46.1 at P .3 / .5 / .9; every kind raises the modal answer 2+ off at 720-1,200 (880: .198 -> .199-.677); distinct
    modal answers stay at 62.5 or below (random map 63.4).
  References, Bayes / memory-free Bayes / naive: both right at T .595 / .336 / .171; far errors .052 / .226 / .452;
    distant pairs merged .0015 / .0085 / .0184 (naive merges distant pairs more often than random answers, .0100);
    answer cells used 93.0 / 77.1 / 57.3; e^H 83.9 / 62.5 / 41.8; cells lost later 1.9 / 5.0 / 5.6; peak step 30 /
    7 / 7; path 9.12 / 10.65 / 10.80 (net move 2.66 / 2.64 / 2.75, back-and-forth 6.45 / 8.01 / 8.05; moves into steps
    2-5 3.75 / 3.89 / 3.81, into steps 11-30 3.33 / 4.41 / 4.58). Where answers land at T [right, neighbour, one far,
    both far]: Bayes [.595 .354 .049 .003], memory-free [.336 .438 .220 .006], naive [.171 .377 .418 .034]; against
    chance, naive's errors are near misses (.377 vs .068) and one-variable errors (.418, chance .403) with few
    both-far (.034 vs .518). One-variable mirror flips (9 - r, >= 2 away) .001 / .015 / .023. Modal answers per
    environment: Bayes 94.4, naive 44.3; far-apart modal merges .5 / 89.8; modal answer 2+ off .005 / .432; merged
    pairs (median per environment) 5 / 142.
  Trained RNN
    initial network: at the chance profile ([.009 .052 .398 .541]); 3.1 answer cells per environment and 13 over all
      10,000 probes (pooled e^H 2.9); distant pairs merged .668.
    folded phase (20 to ~500): far errors .879 (20), .792 (80), .715 (200), .554 (400), against naive's .452 and
      chance .922, with both far .389, .309, .224, .134 (naive .034); distant pairs merged .431, .154, .052, .037
      (naive .018, chance .010); answer cells 3.2, 14.7, 31.1, 40.8 (chance 86.6). At 200 the answers are concentrated
      even pooled over environments (all 10,000 probes: e^H 56.3; Bayes 99.6, naive 88.7) and twice more within one
      environment: within D 31.1, e^H 20.1 vs 200 probes drawn across environments D 62.1, e^H 44.5 (naive 57.3 /
      41.8 within vs 82.0 / 67.3 across: naive's folds are environment-specific too).
    unfolding, naive's level: far errors fall below naive's at 582 [531-620], which noise cannot produce (it raises
      far errors). In the same window distant pairs merged (620 [579-666]), far-apart modal merges (636 [591-684], 19
      checkpoints), answer cells (631 [591-686]) and e^H (639 [600-697]) pass naive's level, which noise alone could
      do (open dots: random answers lie past naive's level of each); these five cannot be ordered among themselves.
      Modal answers pass naive's at 706 [655-781] (open). Both right passes naive's at 746 [678-810] (noise-proof),
      after far errors and distant pairs merged (intervals do not overlap) but not orderable against answer cells, e^H
      or modal answers. Paired differences from naive (mean +- SE over environments): far errors +.071 +- .026 (480),
      +.023 +- .024 (560), -.059 +- .025 (640), -.117 +- .025 (720); distant pairs merged +.0042 +- .0018 (560), -.0013
      +- .0017 (640), -.0048 +- .0014 (720); D -5.2 +- 1.7 (560), +.64 +- 1.9 (640), +4.76 +- 2.0 (720), +7.98 +- 1.9
      (800); e^H -5.6 +- 1.6 (560), +.04 +- 1.8 (640), +4.1 +- 1.9 (720); both right -.034 +- .013 (640), -.009 +- .014
      (720), +.018 +- .014 (800), +.068 +- .017 (880). All of these except both right come before the Trained RNN
      passes naive's goal accuracy at T, 835 [798-876] (re-timed in CHECKS; upper bounds 620-781, below its 798); both
      right overlaps it.
    beyond naive: distant pairs merged falls below random answers' 1/100 at 866 [807-931]. Uniform noise cannot
      produce this, local jitter can (the answers at 800, .0120: .0097 with +-1 jitter at P .5), but every kind of
      noise whose 5-draw mean falls below 1/100 has far errors of .398 or more (each single draw below it, 17 of 50,
      .352 or more), while the network's own keep falling (.293 at 800, .241 at 880): a fold event only together
      with b (half-filled dot). Far-apart modal merges
      fall below the random map's at 897 [810-1,016]: replacing answers by random ones barely moves them (880, 47.6:
      47.0-47.5 with 10-60% random), local jitter takes them below 46.1 (+-1 at P .3 / .5: 43.6 / 38.5), so this
      crossing too counts only together with the modal answers' far share, which every noise tested raises and which
      keeps falling (.258 at 800, .198 at 880, .159 at 1,040). Memory-free Bayes's level: far errors 935 [851-1,042]
      (noise-proof) and distant pairs merged 935 [851-1,045] (out of reach of uniform noise, as .0085 lies below
      1/100; local jitter of the answers at 880, .0096, gives .0088 / .0086 at P .3 / .5, close to .0085: it also
      counts only together with far errors), both before both right (1,199 [1,116-1,319]); e^H 1,181
      [1,057-1,372] and answer cells 1,235 [1,097-1,418] (open: noise could produce them; paired D - memory-free -.24
      +- 1.7 at 1,200, +.30 +- 1.8 at 1,280, +4.18 +- 1.6 at 1,600); modal answers past the random map's 1,230
      [1,125-1,353]; answer cells past random answers' only at 2,638 [2,351-3,302] (e^H 2,549 [2,323-3,109]). At the
      end far errors reach Bayes's level within the SE (.054 vs .052; paired +.0021 +- .0015, per environment over the
      last 10 checkpoints +.0017 +- .0012; 8% of resamples end below it for good), while both right, D and e^H stay
      below it (-.018 +- .003, -1.5 +- .4, -1.6 +- .4: .577 vs .595, 91.5 vs 93.0, 82.3 vs 83.9); distant pairs merged
      ends just above it (+.000075 +- .000045; over the last 10, +.000091 +- .000037).
    curves: both right .009 (0), .044 (200), .100 (560), .162 (720), .239 (880), .336 (1,200), .484 (2,400), .577
      (end); far errors .939, .715 (200), .475 (560), .393 (640), .335 (720), .241 (880), .165 (1,200), .054 (end);
      distant pairs merged .668, .052 (200), .0226 (560), .0171 (640), .0137 (720), .0096 (880), .0056 (1,200), .0016
      (end); answer cells 3.1, 31.1 (200), 52.1 (560), 62.1 (720), 68.7 (880), 76.9 (1,200), 91.5 (end); e^H 2.1,
      20.1 (200), 45.9 (720), 53.6 (880), 63.1 (1,200), 82.3 (end); modal answers 2.6 (0), 15.2 (120), 22.3 (240),
      45.0 (720), 53.2 (880), 62.7 (1,200), 79.4 (2,400), 91.7 (end), environment 14: 2, 28, 67, 91, 100 at 0, 240,
      880, 2,400, end; far-apart modal merges 3,160 (0), 273 (240), 88.4 (640), 47.6 (880), 1.5 (end).
    where errors land: [.044 .241 .491 .224] (200), [.089 .388 .407 .116] (480), [.162 .503 .282 .052] (720), [.336
      .499 .152 .013] (1,200), [.577 .369 .052 .002] (end); neighbours .51-.52 at 800-1,040. At matched joint accuracy
      (720: .162 vs naive's .171) its far errors are .335 vs .452: its errors are near misses (.503 vs .377), with as
      many modal answers (45.0 vs 44.3).
    within the episode: cells lost later 1.5 (0), max 18.9 at 260 (naive 5.6), 16.3 (400), 9.5 (560), 6.0 (720), 5.3
      (880), 4.0 (1,200), 2.4 (end); half-way back 512 [474-568]; equal to naive's within the SE from 720 to 1,120
      (paired +.40 +- .68 at 720, -.22 +- .64 at 880, -.74 +- .59 at 1,120), below it from 1,200 (-1.58 +- .51; -1.92
      +- .48 at 1,600); past naive's for good 844 [701-1,370] (open: random answers at every step sit at naive's
      level, 5.55), memory-free's 1,088 [708-1,944]. Peak step of D_t: 3-7 from 20 to 480, 9 (560), 13-14 (640-880),
      23 (1,200), 30 at the end; in the second half of the episode for good from 946 [660-1,139].
    same wrong modal answer as naive (fold g): .062 (240), max .109 at 800 (the peak's checkpoint over resamples
      640-880), .014 at the end (controls .028, .044, .006). Conditional on the network's modal answer being wrong:
      .066 vs control .029 (240), .138 vs .056 (800), .160 vs .074 (1,200), .228 [.179-.280] vs .096 [.087-.105]
      (end): 2.0-2.7x its control at every checkpoint from 120 on. The fall to .014 is fewer errors (wrong on 78.5% of
      these cells at 800, 6.1% at the end), not less naive-like errors; its errors are mostly its own throughout.
    movement: path 1.46 (0), ~5.0 (20-80), 6.9 (200), 9.57 (560), 9.72 (880), max 10.20 at 1,200, 9.25 (end); drift
      path 1.31-1.99 at 0-40, .4-1.0 at 200-1,200, .15 at the end; net move 3.1-3.5 at 200-1,200, 2.68 at the end;
      back-and-forth .08 (0), 3.51 (200), 6.11 (560), 6.55 (880), max 6.97 at 1,200, 6.57 (end; Bayes 6.45), half-way
      193 [188-200]. The extra travel around 1k is early: moves into steps 2-5 4.45 at 1,200 (Bayes 3.75), into steps
      11-30 3.39 (Bayes 3.33), unlike naive's late restlessness (4.58).
  ESN
    one answer for all 10,000 probes at 20 training episodes (D 1.0 in every environment, 1 cell pooled; distant
      pairs merged 1.0; drift 5.37 of a 5.53 path).
    answers: both right .023 (200), .043 (880), .070 (2,400), .152 (end), never past naive's (in 98% of resamples;
      end -.019 +- .009, paired); far errors .813 (200), .722 (880), .609 (2,400), .420 (end), past naive's only at
      20,197 [11,809-37,642] (never in 3% of resamples; end -.032 +- .016); distant pairs merged .204 (200), .076
      (880), .0375 (2,400), .0177 (end; naive .0184, -.0007 +- .0015; past naive's for good in 72% of resamples, never
      below random answers'); answer cells 10.0 (200), 24.8 (880), 40.4 (2,400), 55.4 (end; -1.9 +- 1.3); e^H 41.0
      (-.77 +- 1.3); modal answers 8.8 (240), 18.4 (880), 28.7 (2,400), 42.1 (end; -2.3 +- 1.1; never past naive's),
      environment 14: 11, 21, 34, 46; far-apart modal merges 82.9 (naive 89.8). Categories at the end [.152 .428 .385
      .034] vs naive [.171 .377 .418 .034] (differences -.019 +.051 -.033 +.001, SE .009 .011 .015 .003). It ends at
      naive's level of folding.
    within the episode: cells lost later max 19.5 at 400, 9.9 at the end (naive 5.6; +4.3 +- .8), half-way back 1,387
      [1,167-1,618]; peak step 1-8 from 20 on (4 at the end), never in the second half.
    same wrong modal answer as naive .037 (240), .087 (880), .203 (2,400), max .330 at 16,000 (the peak's checkpoint
      over resamples 16,000-39,996), .319 at the end (control .058); conditional on being wrong .360 [.328-.391] vs
      .065 [.060-.070] at the end; this ratio to the control grows from 2.1x (240) to 3.0x (880) and 5.5x (end). The
      ESN converges onto naive's specific folds.
    movement: path 5.53 (20, almost all drift), 6.20 (880), 6.98 (end); net move 2.77; back-and-forth 4.21, below
      every reference (-3.84 +- .13 vs naive); moves into steps 2-5 2.90, into steps 11-30 2.59.
  Half-way crossings (timeline.npz; half of each curve's own change), Trained RNN / ESN: both right 1,068 [972-1,129]
    / 3,277 [2,907-4,053]; far errors 523 [449-574] / 1,276 [1,130-1,505]; distant pairs merged (log scale) 447
    [389-485] / 530 [353-755]; cells lost later (back from the hump) 512 [474-568] / 1,387 [1,167-1,618];
    back-and-forth 193 [188-200] / 213 [198-249]. Noise alone could produce the last three (levels .0325 / .107 above
    naive's .0184; 10.77 / 14.7 above chance 5.55; back-and-forth rises with noise): not noise-proof, unlike both
    right and far errors.
  Landmarks: passing naive's goal accuracy at T 835 [798-876] (re-timed in CHECKS: stays-past rule on
    td_tt_gap.boot_counts(), as in the brief; the first checkpoint above naive's runs 800-880 over resamples); the
    memory-use flip by entanglement 800-1,040 (a checkpoint bracket without an interval; not re-derived); the belief
    grid's unfolding 551 [391-649] / 770 [703-858] / 1,262 [1,097-1,523] (25 / 50 / 75%; not re-derived). Far errors
    below naive's (582 [531-620]) come before passing naive (intervals do not overlap) and before the flip's
    bracket; so do the naive-level crossings that noise could produce (distant pairs merged, answer cells, e^H, modal
    measures; upper bounds 666-781). These naive-level crossings start at the beginning of the belief grid's
    unfolding window, not inside it: far errors' interval (531-620) straddles its 25% point (551 [391-649]) and ends
    before its 50% point (770 [703-858]). Both right past naive's (746 [678-810]) overlaps passing naive and the
    flip's bracket. Distant pairs merged below chance (807-931), far errors at memory-free's level (851-1,042) and the
    peak step moving to the second half (660-1,139) overlap passing naive and the flip and cannot be ordered against
    them; both right at memory-free's level (1,116-1,319) comes after passing naive and after the flip's bracket. The
    timing is correlational, with one seed per network.
  Dropped panels (no separate conclusion; numbers kept): grid cells visited per episode (distinct round(m_t)): Bayes /
    memory-free / naive 5.72 / 6.29 / 6.36; Trained 2.33 (0), ~6 (20-80), max 7.14 at 560, 6.86 (880), 5.79 (end); ESN
    7.71 (20, drift), 6.14 (880), 5.77 (end). Pooled grid coverage of the belief cloud (e^H of round(m_t) over all
    10,000 probes, mean over steps): 93.4 / 93.7 / 85.7 (at T 99.4 / 98.7 / 90.5; within an environment at T 81.2 /
    64.2 / 46.3); Trained 3.1 (0), 34 (80), 73.8 (200), 86.5 (560), 89.6 (880), 93.5 (end); ESN 2.5 (20), 32.3
    (200), 67.7 (880), 88.9 (end). Pooled spread Var(m_0) + Var(m_1), mean over steps: 14.24 / 14.62 / 15.07 cells^2;
    Trained .19 (0), 12.6 (200), 14.3 (end); ESN .21 (20), 14.4 (end).
  Ladder (the Trained RNN's logit-increment regression on N, W = SW - N, M = D - SW; ck files' per-step normal
    equations; no environment resampling possible): fitted within each step where X^T X is invertible (steps 2-30;
    step 1 has M = 0; X^T X is the same at every checkpoint), then averaged over those steps: N / W / M .119 / .001 /
    .001 (60), .258 / .015 / -.002 (100), .364 / .027 / -.003 (200), .520 / .097 / .008 (400), .615 / .135 / .024
    (560), .726 / .226 / .062 (880), .760 / .682 / .423 (end); half the final value for good (stays-past,
    log-interpolated) at 215 / 1,895 / 3,762 (first checkpoint at it: 180 / 1,920 / 3,800). Over steps 3-30 only (the
    re-verifier's set): end .754 / .674 / .406, 208 / 1,757 / 3,594 (first: 180 / 1,760 / 3,600). Pooled over steps
    (one regression on the summed normal equations, the 11-01 version; marked pooled): end .768 / .686 / .410, 225 /
    2,054 / 3,690 (first: 180 / 2,080 / 3,800). No intervals: nothing is ordered against the rungs.

SYNTHESIS EXPORT (python td_tt_latent.py synthesis; prints the table and its checks) -> synthesis.npz via
  td_tt_gap.export('latent', x, rows): the gap fraction f = (y - naive) / (Bayes - naive) at the 114 checkpoints (x =
  td_tt_gap.train_x, the initial network at 10); c0 = passes naive's level for good, c50 = past the naive-Bayes
  midpoint for good, each with 5-95% over the shared resamples td_tt_gap.boot_counts() (1,000 x 50, seed 20261103),
  paired. Rows, family 'answers', Trained RNN and ESN each; all probe-based (environment e = probe row // 200, asserted
  for all 10,000 probes; it indexes every cached per-environment value):
    goal and context both right at T (rise; noise lowers it);
    far errors at T (rise: f rises as far errors fall toward Bayes's; truth-referenced, noise raises them);
    distant pairs merged at T (rise; neither crossing is noise-proof on its own: c0 can come from uniform noise; c50
      = .00996, just below 1/100, is out of reach of uniform noise but not of local jitter (the Trained RNN's answers
      at 800: .0120 -> .0097 with +-1 jitter at P .5, quoted in the note), so it counts only together with far errors);
    answer cells lost later in the episode (return: the hump rises far above naive's 5.56 and comes back; c0 can come
      from noise, chance 5.55; c50 = 3.73 cannot).
  Each is a per-environment statistic averaged over the 50 environments (distant pairs merged: 18,432 pairs in every
  environment, so its mean of shares is the ratio of sums), so each resample is the count-weighted mean of the
  per-environment values (td_tt_gap.env_mean), with naive's and Bayes's levels resampled on the same environments; with
  all counts 1 it reproduces the plotted curves (max |diff| 3.3e-16).
  Trained RNN, c0 / c50: both right 746 [683-809] / 1,403 [1,317-1,475]; far errors 582 [535-619] / 863 [811-950];
  distant pairs merged 620 [580-663] / 867 [807-931]; cells lost later 844 [703-1,379] / 1,998 [1,711-28,000]. ESN:
  far errors c0 20,197 [11,611-37,779] (98% of resamples cross), distant pairs merged c0 20,250 [11,233-open] (71%);
  no other crossing (both right c0 in 2% of resamples; c50 in none).
  Agreement with this module's code: every c0 point equals the naive-level crossing of numbers (whose intervals use
  this module's own resamples, seed 20261102: e.g. far errors [531-620] vs [535-619]); every c50 equals stay_cross at
  the naive-Bayes midpoint on the same resamples (point, interval and every resample), except 6 resamples of cells
  lost later in which the curve sits exactly on the midpoint at a checkpoint (integer counts) and float rounding
  decides "past" (lower bound 1,711 vs 1,712). The c50s differ from timeline.npz's half-way crossings by definition
  (there: half of each curve's own change): both right 1,403 vs 1,068, far errors 863 vs 523, distant pairs merged
  867 vs 447 (log scale), cells lost later 1,998 vs 512 (half way back from the hump).
  Left out: answer cells used and e^H (noise raises them: random answers' 86.6 / 75.5 lie past naive's and
  memory-free Bayes's levels); back-and-forth (the network approaches from outside the naive-Bayes interval, rising
  from .08 to at most 6.97 against Bayes 6.45 and naive 8.05, so its gap fraction starts above 1 and is not a timing
  measure; noise raises it too); the modal-answer measures (19 checkpoints of a separate episode set).

CHECKS (python td_tt_latent.py checks)
  the truth as an observer: distinct 100, e^H 100, both right 1, categories [1 0 0 0], far errors 0, distant pairs
  merged 0, pooled coverage 100; random answers through the same code: D 86.6, e^H 75.4, categories [.0098 .0685
  .4034 .5182], distant pairs merged .0100 (= chance); distinct answers by np.unique in 3 environments (ck 1,100 raw
  episodes): 73 / 70 / 65 = cache; distant pairs merged by an explicit double loop (environment 17): 241 / 18,432 =
  .01308 = cache; categories sum to 1 (2e-16), far errors = one far + both far (3e-16), the environment mean of both
  right = the pooled share (1e-16); one probe's path (11.07) and visited cells (7) by explicit loops, its
  environment's means 10.4632 / 6.9100 = cache; references' argmax at T = the probe set's stored correctness (100%);
  spread at T 15.052 vs half the mean squared distance of 200,000 random probe pairs 15.051; Bayes's modal answers in
  environment 18 recounted by hand: 96 distinct = cache, modal = truth in 96% of pairs; the crossing helper on known
  curves (316.23 = exact; a dip and recovery handled; never = NaN); the bootstrap with equal weights reproduces the
  curve (2e-16). The rebuilt caches equal the builder's on every shared field (the control in fold g now pools counts
  over the 20 permutations: changes in the 4th decimal). 11-02: the noise-test helpers (t_stats, lost_later,
  back_forth) reproduce the cached reference values exactly (max |diff| 0); the passing-naive landmark re-timed
  (stays-past rule on td_tt_gap.boot_counts(); goal accuracy recomputed from the stored correctness = the stored
  accuracy within 1e-16): 835 [798-876], first checkpoint above naive's 800 (5%) - 880 (95%) over resamples (the
  re-verifier's own resamples: 835 [791-876]); the exact chance level of cells lost later (5.5501, E[D] 86.6020 = 100
  (1 - .99^200)) against simulation (5.54 +- .07; numbers); synthesis: see its paragraph.

FAILED OR NULL
  - "One variable far" is not a fold signature: its chance level (.403) is naive's (.418). The builder's unfolding
    event "one-far errors below naive's for good from 480" is retracted: at 480 far errors were .523 vs naive's .452
    (+.071 +- .026) and both-far .116 vs .034; far errors pass naive's at 582 [531-620].
  - "Around 200 the answers cover the whole grid globally (97 of 100 cells pooled vs 31 within)" is retracted: it set
    10,000 answers against 200, and the pooled answers are concentrated (e^H 56.3 vs Bayes 99.6); sample-matched, 62.1
    across vs 31.1 within (e^H 44.5 vs 20.1).
  - Single-checkpoint coincidences retracted: "cells lost later fall below naive's at 880 = passing naive" (equal
    within the SE from 720 to 1,120; crossing 844 [701-1,370]); "answer cells pass naive's at 640 and memory-free's at
    1,280" (ties at those checkpoints, +.64 +- 1.9 and +.30 +- 1.8; crossings 631 [591-686], 1,235 [1,097-1,418]).
  - Answer cells used and e^H do not mark unfolding by themselves: random answers (86.6 / 75.5) beat memory-free Bayes
    and naive, and the truth plus +-1 jitter (91.9 / 82.4) matches the Trained RNN's end (91.5 / 82.3).
  - Grid cells visited barely separates observers (5.7-6.4) and is inflated at initialization by rounding jitter
    (2.33); pooled coverage saturates and is not sample-matched; the pooled spread does not separate observers
    (14.2-15.1). All three were dropped from the figure.
  - Mirror flips (9 - r) explain almost nothing (naive .023 of its .418 one-far answers).
  - Subtracting the probe-averaged log belief before taking answers is not a usable control: Bayes's answer cells go
    93.0 -> 65.0 (naive 57.3 -> 25.4, memory-free 77.1 -> 41.5).
  - The Trained RNN's intermediate folds are not naive's folds: same wrong answer at most .109 (control .044), and,
    given it is wrong, 2.0-2.7x its control at every stage, while the ESN's ratio grows to 5.5x.
  - 11-02: "noise cannot move distant pairs merged past naive's level" (11-01) is retracted. Uniform noise moves it
    toward 1/100 from either side, and naive with 10% / 30% of its answers random gives .0167 / .0141, below its .0184
    (far errors .500 / .590). Its naive-level crossing (620 [579-666]) and timeline.npz's half-way (447, level .0325)
    are not noise-proof; the ~600 unfolding now rests on far errors (582 [531-620]), and the fall below 1/100 (866
    [807-931]) counts only together with them (next entry). Far errors were called fold-specific; they are
    truth-referenced.
  - 11-02, third check: "falling below 1/100 cannot come from noise" (the second check's fix: "the noise-proof fold
    event", filled dot) holds for uniform noise only. Local answer jitter lowers distant-pair merging of a partly
    folded observer and can cross 1/100: the Trained RNN's answers at 800 go .0120 -> .0097 (+-1 jitter at P .5; far
    errors .293 -> .398); memory-free Bayes .0085 -> .0082 (+-1 at P .3). So the fall below 1/100 (866), far-apart
    modal merges below the random map's (897; jitter at 880: 47.6 -> 43.6 / 38.5, below 46.1) and distant pairs
    merged at memory-free's level (935; jitter at 880: .0096 -> .0086) are not noise-proof on their own; each counts
    only together with falling far errors (b, or the modal answers' far share), which every noise tested raises.
    Neither crossing of the synthesis export's distant-pairs-merged rows is noise-proof on its own (local jitter
    crosses the midpoint .00996).
  - 11-02: answer cells lost later falling below naive's (844 [701-1,370]) can come from noise: random answers at
    every step sit at 5.550, naive's level.
  - 11-02: "Bayes's level is never reached" was too strong: far errors reach it within the SE at the end (+.0021 +-
    .0015).
  - 11-02: the ladder was one regression pooled over steps (averaging order); fitted per step, then averaged, its W
    and M rungs move from 2,054 / 3,690 to 1,895 / 3,762 (stays-past rule; first checkpoint 2,080 / 3,800 -> 1,920 /
    3,800).
  - Ladder rung times depend on the definition (onsets ~50 / ~200 / ~500 from the brief vs half the final coefficient
    for good at 215 / 1,895 / 3,762) and have no intervals: no event here is ordered against them.

CAVEATS
  one seed per network (20261029), one probe world; the intervals resample the 50 environments only (not seeds, not
  trajectories within an environment); modal answers exist at 19 checkpoints only (crossings log-interpolated between
  them); the passing-naive line is drawn at the first checkpoint past naive (880), its crossing (835 [798-876]) is
  re-timed only as a check; answers are pairs of per-variable argmaxes (the networks have marginal read-outs only);
  the arrow maps show one rule-chosen environment; the peak-step line is a statistic of the environment-averaged
  profile; the early path includes shared drift (dotted in i); a peak-then-fall of answer counts within the episode is
  expected from any observer with a folded asymptote, so it marks folding, not a mechanism. 11-02: the passing-naive
  landmark's interval uses td_tt_gap.boot_counts() (seed 20261103) and this module's crossings use their own resamples
  (seed 20261102); intervals are compared only as intervals, so no ordering depends on this. The noise tests cover
  uniform replacement and local jitter of the references' and the networks' own answers (at 640-880 and the ESN's
  20,000; grid replays at 720-1,200) and Gaussian noise on log beliefs, not every kind of noise; "noise-proof" means
  under these. noise_can_cross compares levels: in h, chance 5.550 and naive's 5.56 differ by .01, far below the SD
  (.43) of one 50-environment mean at chance, so the reading is "at naive's level". td_tt_gap.crossing compares float
  gap fractions strictly, so exact ties at a level (integer counts) are broken by rounding (6 of 1,000 resamples of
  the cells-lost c50; its lower bound moves by 1 episode).

Run from the repository root (CPU only; data ~5 min; with the caches built numbers ~0.5-2 min, checks, figures and
synthesis each under ~15 s):
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg \\
        /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_latent.py data
    (then the same with 'figure', 'numbers', 'checks', 'timeline' or 'synthesis')"""
import os
import sys
import glob
import time
import numpy as np
from scipy.special import logsumexp

ROOT = "/home/johns/Desktop/CognitiveGridworld"
HERE = os.path.join(ROOT, "main", "REPLICATION", "claude_plots")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION")]
os.environ.setdefault("MPLBACKEND", "Agg")
TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
GRIDS = os.path.join(TD, "summary_data", "grids")
OUT = os.path.join(TD, "through_training", "latent")
RUNS = ("trained", "echo")
REFS = ("bayes", "memfree", "naive")
T, R, NE = 30, 10, 50
V = np.arange(R, dtype=np.float64)
X0 = 10                                                          # the initial network on log axes
NBOOT, BOOT_SEED, CHANCE_SEED, MATCH_SEED, NOISE_SEED = 1000, 20261102, 20261103, 20261104, 20261105


def train_x(ep):
    ep = np.asarray(ep)
    return np.maximum(ep - ep // 5, X0).astype(float)


def _ck_files(run):
    return sorted(glob.glob(os.path.join(AN, run, "ck_?????.npz")))


def _probs(lq):
    lq = np.asarray(lq, dtype=np.float64)
    return np.exp(lq - logsumexp(lq, -1, keepdims=True))


def _entropy_eff(counts):
    """exp(plug-in entropy) of count vectors along the last axis"""
    p = counts / counts.sum(-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(p), 0.).sum(-1)
    return np.exp(h)


def _distinct_rows(a):
    """number of distinct values in every row of the integer array a (n, k)"""
    s = np.sort(a, 1)
    return 1 + (np.diff(s, axis=1) != 0).sum(1)


_FAR_CACHE = {}


def far_pairs(r, env):
    """index pairs (i < j) of probes of the same environment whose latent pairs are far apart
    (max_v |r_vi - r_vj| >= 2), and each pair's environment; 18,432 pairs per environment (4,608 far-apart cell
    pairs x 2 x 2 trajectories)"""
    key = (len(r), int(np.asarray(r).sum()), int(np.asarray(env).sum()))
    if key not in _FAR_CACHE:
        I, J, E_ = [], [], []
        for e in range(NE):
            rows = np.flatnonzero(env == e)
            i, j = np.triu_indices(len(rows), 1)
            far = np.abs(r[rows[i]] - r[rows[j]]).max(-1) >= 2
            I.append(rows[i][far])
            J.append(rows[j][far])
            E_.append(np.full(int(far.sum()), e))
        _FAR_CACHE[key] = tuple(np.concatenate(v) for v in (I, J, E_))
    return _FAR_CACHE[key]


def answer_stats(lq, r, env, right=None):
    """everything about one observer's answers and posterior means on the probe set (see the module docstring);
    lq (n, T, 2, R) log beliefs (any normalisation), r (n, 2) true values, env (n,), right (n, T, 2) correctness
    (default: argmax == r)"""
    n = len(lq)
    ans = np.asarray(lq).argmax(-1)                                                   # (n, T, 2)
    if right is None:
        right = ans == r[:, None, :]
    acell = ans[..., 0] * R + ans[..., 1]                                            # (n, T) answer cell
    cell = r[:, 0] * R + r[:, 1]
    tt = np.arange(T)[None, :]
    cnt_e = np.bincount(env, minlength=NE).astype(np.float64)

    def per_env(v):                                                                  # (n, T) -> (T, NE) means
        return np.stack([np.bincount(env, v[:, t], NE) for t in range(T)]) / cnt_e

    out = {"acc_var": right.mean(0), "acc_joint": right.all(-1).mean(0), "right_env": per_env(right.all(-1))}
    # (1) answer cells used per environment and step (the environment's 200 probes)
    cnt = np.bincount(((tt * NE + env[:, None]) * R * R + acell).ravel(), minlength=T * NE * R * R)
    cnt = cnt.reshape(T, NE, R * R)
    out["distinct"] = (cnt > 0).sum(-1)                                              # (T, NE)
    out["eff"] = _entropy_eff(cnt.astype(np.float64))                                # (T, NE)
    out["usage"] = cnt.sum(1)                                                        # (T, 100) pooled answer counts
    both = right.all(-1)
    out["cell_acc"] = np.stack([np.bincount(cell, both[:, t], R * R) / np.bincount(cell, minlength=R * R)
                                for t in range(T)])                                  # (T, 100) per true cell
    # (2) far-apart latent pairs given the same answer (same environment): the folds
    I, J, Ep = far_pairs(r, env)
    same = acell[I] == acell[J]                                                      # (pairs, T)
    npair = np.bincount(Ep, minlength=NE).astype(np.float64)
    out["coll_env"] = np.stack([np.bincount(Ep, same[:, t], NE) for t in range(T)]) / npair    # (T, NE)
    # (3) where the answer lands relative to the truth
    d = ans - r[:, None, :]                                                          # (n, T, 2) in -9..9
    ad = np.abs(d)
    lo, hi = ad.min(-1), ad.max(-1)
    out["cat"] = np.stack([(hi == 0).mean(0), (hi == 1).mean(0), ((hi >= 2) & (lo <= 1)).mean(0),
                           (lo >= 2).mean(0)], -1)                                   # (T, 4)
    out["far_env"] = per_env(hi >= 2)                                                # (T, NE) one or both far
    hT, lT = hi[:, -1], lo[:, -1]
    out["cat_env"] = np.stack([np.bincount(env, w, NE) for w in (hT == 0, hT == 1, (hT >= 2) & (lT <= 1), lT >= 2)],
                              -1) / cnt_e[:, None]                                   # (NE, 4) at T
    mir = 9 - r                                                                      # one-variable mirror (9 - r)
    far_mirror = np.abs(mir - r) >= 2                                                # (n, 2) the mirror is >= 2 away
    is_m0 = (ans[..., 0] == mir[:, None, 0]) & (ans[..., 1] == r[:, None, 1]) & far_mirror[:, None, 0]
    is_m1 = (ans[..., 1] == mir[:, None, 1]) & (ans[..., 0] == r[:, None, 0]) & far_mirror[:, None, 1]
    out["mirror"] = (is_m0 | is_m1).mean(0)                                          # (T,)
    out["disp"] = np.bincount(((tt * 19 + d[..., 0] + 9) * 19 + d[..., 1] + 9).ravel(),
                              minlength=T * 19 * 19).reshape(T, 19, 19)
    out["conf_T"] = np.bincount(cell * R * R + acell[:, -1], minlength=R ** 4).reshape(R * R, R * R)
    # (4) posterior means: trajectories and the cloud per step
    m = (_probs(lq) * V).sum(-1)                                                     # (n, T, 2)
    mv = np.linalg.norm(np.diff(m, axis=1), axis=-1)                                 # (n, T - 1)
    out["move"] = mv.mean(0)
    out["ms_move"] = (mv ** 2).mean(0)                     # mean squared step = |step of the average|^2 + variance
    out["mbar"] = m.mean(0)                                # (T, 2) the probe-averaged posterior mean (common drift)
    path = mv.sum(1)
    rc = np.clip(np.floor(m + .5), 0, R - 1).astype(int)
    rcell = rc[..., 0] * R + rc[..., 1]                                              # (n, T) rounded mean's cell
    vis = _distinct_rows(rcell)
    out["path_env"] = np.bincount(env, path, NE) / cnt_e
    net = np.linalg.norm(m[:, -1] - m[:, 0], axis=-1)                                # net displacement, step 1 -> T
    out["net_env"] = np.bincount(env, net, NE) / cnt_e
    out["early_env"] = np.bincount(env, mv[:, :4].sum(1), NE) / cnt_e               # moves into steps 2-5
    out["late_env"] = np.bincount(env, mv[:, 9:].sum(1), NE) / cnt_e                # moves into steps 11-30
    out["visited_env"] = np.bincount(env, vis, NE) / cnt_e
    out["visited_hist"] = np.bincount(vis, minlength=T + 1)
    occ = np.bincount((tt * R * R + rcell).ravel(), minlength=T * R * R).reshape(T, R * R)
    out["occ"] = occ
    out["cover"] = _entropy_eff(occ.astype(np.float64))                              # (T,) pooled over environments
    out["occupied"] = (occ > 0).sum(-1)
    out["spread"] = m.var(0).sum(-1)                                                 # (T,) cells^2, pooled
    out["dist_truth"] = np.linalg.norm(m - r[:, None, :], axis=-1).mean(0)           # (T,)
    cov_env = np.bincount(((tt * NE + env[:, None]) * R * R + rcell).ravel(), minlength=T * NE * R * R)
    out["cover_env"] = _entropy_eff(cov_env.reshape(T, NE, R * R).astype(np.float64))
    assert (env == np.repeat(np.arange(NE), n // NE)).all(), "probes must be environment-major"
    out["spread_env"] = m.reshape(NE, n // NE, T, 2).var(1).sum(-1).T                # (T, NE)
    return out


def _probe_set():
    P = np.load(os.path.join(AN, "probes.npz"))
    return {k: P[k] for k in ("r", "env", "cell", "goal", "log_jb", "log_nb", "inc_SW", "logpost", "right_bayes",
                              "right_naive", "right_window1")}


def ref_beliefs(P):
    """log beliefs (n, T, 2, R) of Bayes (exact marginals), memory-free Bayes (each step's own marginal likelihood
    summed over steps = windowed Bayes w = 1) and naive"""
    return {"bayes": P["log_jb"].astype(np.float64), "naive": P["log_nb"].astype(np.float64),
            "memfree": np.cumsum(P["inc_SW"].astype(np.float64), 1)}


def build_refs():
    path = os.path.join(OUT, "refs.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    P = _probe_set()
    out = {}
    for k, lq in ref_beliefs(P).items():
        for key, v in answer_stats(lq, P["r"], P["env"]).items():
            out[f"{k}__{key}"] = v
    # checks against the probe set's stored correctness at T
    for k, key in (("bayes", "right_bayes"), ("naive", "right_naive"), ("memfree", "right_window1")):
        a = ref_beliefs(P)[k][:, -1].argmax(-1)
        out[f"check_{k}"] = (a == P["r"]) == P[key]
    # Bayes's joint MAP answer at T (the 100-cell argmax), for comparison with the marginal pair
    amap = P["logpost"][:, -1].argmax(-1)
    cnt = np.bincount(P["env"] * R * R + amap, minlength=NE * R * R).reshape(NE, R * R)
    out.update(map__distinct=(cnt > 0).sum(-1), map__eff=_entropy_eff(cnt.astype(np.float64)),
               map__acc_joint=(amap == P["cell"]).mean())
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    return out


def build_run(run):
    path = os.path.join(OUT, f"probes_{run}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    P = _probe_set()
    fs = _ck_files(run)
    rows = {}
    ties = []
    t0 = time.time()
    for f in fs:
        z = np.load(f)
        lq = z["logq"]
        st = answer_stats(lq, P["r"], P["env"], right=z["right"])
        mx = lq.max(-1, keepdims=True)
        ties.append([((lq == mx).sum(-1) > 1).mean(), ((lq[:, -1] == mx[:, -1]).sum(-1) > 1).mean()])
        st["episode"] = int(z["episode"])
        st["acc_goal"] = float(z["acc"][-1])                                         # goal accuracy at T (stored)
        for k, v in st.items():
            rows.setdefault(k, []).append(v)
    out = {k: np.array(v) for k, v in rows.items()}
    out["x"] = train_x(out["episode"])
    out["ties"] = np.array(ties)                                                     # fp16 argmax ties (all steps, T)
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    print(f"{run}: {len(fs)} checkpoints, {time.time() - t0:.0f} s", flush=True)
    return out


_TR = np.arange(R * R)
_CELL_DIST = np.maximum(np.abs(_TR[:, None] // R - _TR[None] // R), np.abs(_TR[:, None] % R - _TR[None] % R))
FAR_CELL_PAIRS = np.triu(_CELL_DIST >= 2, 1)                                       # 4,608 far-apart cell pairs


def modal_stats(a, env, cell):
    """per environment and true cell, the modal answer cell over the cell's fresh streams (summary_data/grids: 24 per
    cell) and what it says about merging; a (n, 2) answers at T"""
    acell = a[:, 0].astype(int) * R + a[:, 1].astype(int)
    cnt = np.bincount((env * R * R + cell) * R * R + acell, minlength=NE * R ** 4).reshape(NE, R * R, R * R)
    modal = cnt.argmax(-1)                                                           # (NE, 100) ties: lowest cell
    top = cnt.max(-1)
    ties = (cnt == top[..., None]).sum(-1) > 1
    size = np.stack([np.bincount(modal[e], minlength=R * R) for e in range(NE)])     # true cells per modal answer
    d = np.stack([modal // R - _TR // R, modal % R - _TR % R], -1)
    ad = np.abs(d)
    lo, hi = ad.min(-1), ad.max(-1)
    cat = np.stack([(hi == 0), (hi == 1), (hi >= 2) & (lo <= 1), (lo >= 2)], -1).mean((0, 1))
    far_merges = np.array([((modal[e][:, None] == modal[e][None]) & FAR_CELL_PAIRS).sum() for e in range(NE)])
    return {"modal": modal.astype(np.int8), "consistency": top / cnt.sum(-1), "modal_ties": ties.mean(),
            "distinct_modal": (size > 0).sum(-1), "merged_pairs": (size * (size - 1) // 2).sum(-1),
            "cells_per_answer": size, "cat_modal": cat, "modal_far": (hi >= 2).mean(-1), "far_merges": far_merges}


def build_fwd():
    """modal answers per (environment, true cell) at the 19 checkpoints of summary_data/grids (50 environments x 100
    cells x 24 fresh streams; td_grids_fwd.py), and the same for Bayes and naive on those streams"""
    path = os.path.join(OUT, "modal.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    E = np.load(os.path.join(GRIDS, "fwd_episodes.npz"))
    env, cell = E["env"], E["cell"]
    out = {}
    for k in ("bayes", "naive"):
        for key, v in modal_stats(E[f"a_{k}"], env, cell).items():
            out[f"{k}__{key}"] = v
    for run in RUNS:
        F = np.load(os.path.join(GRIDS, f"fwd_{run}.npz"))
        rows = {}
        for k in range(len(F["ep"])):
            for key, v in modal_stats(F["a"][k], env, cell).items():
                rows.setdefault(key, []).append(v)
        for key, v in rows.items():
            out[f"{run}__{key}"] = np.array(v)
        out[f"{run}__episode"] = F["ep"]
        out[f"{run}__x"] = train_x(F["ep"])
    out["streams_per_cell"] = int(np.bincount(env * R * R + cell).mean())
    np.savez(path, **out)
    return out


def build_same_as_naive():
    """modal maps (summary_data/grids): on the (environment, pair) cells where naive's modal answer is wrong and Bayes's
    right, how often the network's modal answer is naive's wrong answer; control: naive's wrong answer for the same
    pair in another environment (20 random environment permutations; pairs where that environment's naive is also
    wrong), which keeps any pair-specific bias (e.g. towards the grid's edge) and drops the environment's own fold.
    Per-environment counts (for the environment bootstrap), unconditional and conditional on the network's modal answer
    being wrong."""
    path = os.path.join(OUT, "same_as_naive.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    M = build_fwd()
    nm, bm = M["naive__modal"].astype(int), M["bayes__modal"].astype(int)
    cell = np.broadcast_to(_TR, nm.shape)
    sel = (nm != cell) & (bm == cell)
    rng = np.random.default_rng(20261102)
    perms = [rng.permutation(NE) for _ in range(20)]
    out = {"n_cells": int(sel.sum()), "n_sel_env": sel.sum(1)}
    for run in RUNS:
        mod = M[f"{run}__modal"].astype(int)
        K = len(mod)
        acc = {k: np.zeros((K, NE)) for k in ("hit", "wrong", "c_hit", "c_n", "cw_hit", "cw_n")}
        for k in range(K):
            netw = mod[k] != cell                                                    # the network's modal answer wrong
            acc["hit"][k] = ((mod[k] == nm) & sel).sum(1)
            acc["wrong"][k] = (netw & sel).sum(1)
            for p in perms:
                ok = sel & (nm[p] != cell) & (p != np.arange(NE))[:, None]
                h = (mod[k] == nm[p]) & ok
                acc["c_hit"][k] += h.sum(1)
                acc["c_n"][k] += ok.sum(1)
                acc["cw_hit"][k] += (h & netw).sum(1)
                acc["cw_n"][k] += (ok & netw).sum(1)
        for key, v in acc.items():
            out[f"{run}__{key}_env"] = v
        out[f"{run}__same"] = acc["hit"].sum(1) / sel.sum()
        out[f"{run}__control"] = acc["c_hit"].sum(1) / acc["c_n"].sum(1)
        out[f"{run}__same_w"] = acc["hit"].sum(1) / acc["wrong"].sum(1)
        out[f"{run}__control_w"] = acc["cw_hit"].sum(1) / acc["cw_n"].sum(1)
        out[f"{run}__wrong"] = acc["wrong"].sum(1) / sel.sum()
        out[f"{run}__x"] = M[f"{run}__x"]
    np.savez(path, **out)
    return out


# ═══════════════════════════════════════════════════════════════════
# Chance levels, environment bootstrap, crossings
# ═══════════════════════════════════════════════════════════════════

def chance_levels():
    """random answers: every probe's answer pair uniform over the 100 cells, independently. Exact where possible;
    D and e^H of 200 answers by simulation (40,000 sets). Also a noisy observer without folds: the truth plus
    independent +-1 jitter (P .3 per variable, clipped to the grid), 200 answers = every cell twice."""
    rng = np.random.default_rng(CHANCE_SEED)
    a = rng.integers(R * R, size=(40000, 200))
    cnt = np.stack([np.bincount(v, minlength=R * R) for v in a]).astype(np.float64)
    p0, p1 = .1, .18                                                  # P(|a - r| = 0), P(|a - r| = 1) per variable
    cat = np.array([p0 * p0, (p0 + p1) ** 2 - p0 * p0, 0., (1 - p0 - p1) ** 2])
    cat[2] = 1 - cat.sum()
    tr = np.stack(np.divmod(np.repeat(_TR, 2), R), 1)
    jit = rng.choice([-1, 0, 1], size=(2000, 200, 2), p=[.15, .7, .15])
    aj = np.clip(tr[None] + jit, 0, R - 1)
    cj = np.stack([np.bincount(v, minlength=R * R) for v in aj[..., 0] * R + aj[..., 1]]).astype(np.float64)
    pD = occupancy(200, R * R)
    F = np.cumsum(pD)
    emax = float((np.arange(R * R + 1) * (F ** T - np.r_[0., F[:-1]] ** T)).sum())
    eD = float((np.arange(R * R + 1) * pD).sum())
    return {"D": R * R * (1 - .99 ** 200), "D_sim": (cnt > 0).sum(-1).mean(), "eH": _entropy_eff(cnt).mean(),
            "cat": cat, "far": cat[2] + cat[3], "both": cat[0], "coll": 1 / (R * R),
            "modal": R * R * (1 - .99 ** 100), "far_merges": FAR_CELL_PAIRS.sum() / (R * R),
            "modal_far": cat[2] + cat[3], "jitter_D": (cj > 0).sum(-1).mean(), "jitter_eH": _entropy_eff(cj).mean(),
            "lost": emax - eD, "D_occ": eD}


def occupancy(n, m):
    """P(k distinct cells) for n uniform draws over m cells, k = 0..m (exact: C(m, k) k! S(n, k) / m^n, Stirling
    numbers of the second kind in integer arithmetic)"""
    from math import comb, factorial
    S = [1] + [0] * m                                                  # S(i, k) for i = 0, updated in place to i = n
    for i in range(1, n + 1):
        for k in range(min(i, m), 0, -1):
            S[k] = k * S[k] + S[k - 1]
        S[0] = 0
    return np.array([comb(m, k) * factorial(k) * S[k] / m ** n for k in range(m + 1)])


def t_stats(a, r, env):
    """the answer measures at T from answers a (n, 2) alone (the definitions of answer_stats): far errors, distant
    pairs merged, answer cells D, e^H and both right, each per environment, then averaged"""
    ac = a[:, 0] * R + a[:, 1]
    cnt_e = np.bincount(env, minlength=NE).astype(np.float64)
    I, J, Ep = far_pairs(r, env)
    cnt = np.bincount(env * R * R + ac, minlength=NE * R * R).reshape(NE, R * R).astype(np.float64)
    return {"far": (np.bincount(env, (np.abs(a - r).max(-1) >= 2).astype(float), NE) / cnt_e).mean(),
            "coll": (np.bincount(Ep, (ac[I] == ac[J]).astype(float), NE) / np.bincount(Ep, minlength=NE)).mean(),
            "D": (cnt > 0).sum(-1).mean(), "eH": _entropy_eff(cnt).mean(),
            "both": (np.bincount(env, (a == r).all(-1).astype(float), NE) / cnt_e).mean()}


def lost_later(ans, env):
    """answer cells lost later in the episode (max_t D_{e,t} - D_{e,30}, per environment, then averaged) of answers
    ans (n, T, 2)"""
    ac = ans[..., 0] * R + ans[..., 1]
    tt = np.arange(T)[None, :]
    cnt = np.bincount(((tt * NE + env[:, None]) * R * R + ac).ravel(), minlength=T * NE * R * R)
    d = (cnt.reshape(T, NE, R * R) > 0).sum(-1)
    return float((d.max(0) - d[-1]).mean())


def back_forth(lq, env):
    """back-and-forth of the posterior mean (path - net move, per episode, then per environment, then averaged) of log
    beliefs lq (n, T, 2, R)"""
    m = (_probs(lq) * V).sum(-1)                                                     # (n, T, 2)
    path = np.linalg.norm(np.diff(m, axis=1), axis=-1).sum(1)
    net = np.linalg.norm(m[:, -1] - m[:, 0], axis=-1)
    return float((np.bincount(env, path - net, NE) / np.bincount(env, minlength=NE)).mean())


JITTERS = (("+-1, P .3", (-1, 0, 1), (.15, .7, .15)), ("+-1, P .9", (-1, 0, 1), (.45, .1, .45)),
           ("uniform -2..2", tuple(range(-2, 3)), None), ("uniform -3..3", tuple(range(-3, 4)), None))
# noise on the networks' own answers (own_noise, modal_noise_tests): (label, kind, parameter); see add_noise
_PM = {"P .3": (.15, .7, .15), "P .5": (.25, .5, .25), "P .9": (.45, .1, .45)}
OWN_NOISE = (("10% random", "replace", .1), ("30% random", "replace", .3)) + tuple(
    (f"jitter +-1, {p}", "jitter", ((-1, 0, 1), q)) for p, q in _PM.items()) + (
    ("jitter triangular -2..2", "jitter", (tuple(range(-2, 3)), (1 / 9, 2 / 9, 3 / 9, 2 / 9, 1 / 9))),
    ("jitter uniform -2..2", "jitter", (tuple(range(-2, 3)), None)),
    ("jitter uniform -3..3", "jitter", (tuple(range(-3, 4)), None)),
    ("Gaussian SD 1 on the log beliefs", "belief", 1.), ("Gaussian SD 2 on the log beliefs", "belief", 2.))
OWN_CK = (("trained", 800), ("trained", 900), ("trained", 1000), ("trained", 1100), ("echo", 25000))   # raw episodes
MODAL_NOISE = tuple((f"{p}% random", "replace", p / 100) for p in (10, 30, 60, 90)) + tuple(
    (f"jitter +-1, {p}", "jitter", ((-1, 0, 1), q)) for p, q in _PM.items()) + (
    ("jitter uniform -2..2", "jitter", (tuple(range(-2, 3)), None)),)
JITTER_EXAMPLE = ("trained", 1000, "jitter +-1, P .5", "+-1 jitter at P .5")   # quoted in the synthesis note


def _fmt_t(s):
    return (f"far errors {s['far']:.4f}, distant pairs merged {s['coll']:.5f}, D {s['D']:.2f}, e^H {s['eH']:.2f}, "
            f"both right {s['both']:.4f}")


def add_noise(rng, a, lq, kind, par):
    """answers a (..., 2) with noise: 'replace' a share par of the answer pairs by uniform random pairs; 'jitter' every
    variable's answer moved by a draw from par = (values, probabilities), clipped to the grid (local noise); 'belief'
    the argmax of the log beliefs lq (..., 2, R) plus iid Gaussian noise of SD par"""
    if kind == "replace":
        a = a.copy()
        sw = rng.random(a.shape[:-1]) < par
        a[sw] = rng.integers(0, R, (int(sw.sum()), 2))
        return a
    if kind == "jitter":
        return np.clip(a + rng.choice(par[0], size=a.shape, p=par[1]), 0, R - 1)
    return (lq + rng.normal(0, par, lq.shape)).argmax(-1)


def own_noise(P, run, E, reps=5, keep_draws=False):
    """a network's own answers at T (ck file of raw episode E) without noise and under every kind of OWN_NOISE: t_stats,
    means over reps draws, one seed per (E, kind), so numbers and synthesis print the same values (keep_draws: also
    return {kind: [t_stats of each draw]})"""
    lq = np.load(os.path.join(AN, run, f"ck_{E:05d}.npz"))["logq"][:, -1].astype(np.float64)   # (n, 2, R) at T
    a = lq.argmax(-1)
    out = {"none": t_stats(a, P["r"], P["env"])}
    draws = {}
    for i, (lab, kind, par) in enumerate(OWN_NOISE):
        rng = np.random.default_rng([NOISE_SEED, E, i])
        v = [t_stats(add_noise(rng, a, lq, kind, par), P["r"], P["env"]) for _ in range(reps)]
        out[lab] = {k: float(np.mean([o[k] for o in v])) for k in v[0]}
        draws[lab] = v
    return (out, draws) if keep_draws else out


def own_noise_tests(P, D, reps=5):
    """prints: the networks' own answers at T under every kind of OWN_NOISE (own_noise) at the checkpoints OWN_CK; per
    checkpoint the range over the kinds, which kinds take distant pairs merged below 1/100 and their far errors"""
    print(f"noise tests on the networks' own answers at T (mean of {reps} draws, SE over the draws; one seed per "
          f"checkpoint and kind):")
    for run, E in OWN_CK:
        o, dr = own_noise(P, run, E, reps, keep_draws=True)
        s0 = o.pop("none")
        k = list(D[run]["episode"]).index(E)
        dev = max(abs(s0["far"] - D[run]["far_env"][k, -1].mean()), abs(s0["coll"] - D[run]["coll_env"][k, -1].mean()))
        devb = abs(s0["both"] - D[run]["right_env"][k, -1].mean())
        print(f"  {run} {train_x(E):,.0f} itself: {_fmt_t(s0)} (far errors and merged = the cached curve within "
              f"{dev:.0e}; both right within {devb:.0e}, fp16 argmax ties)")
        for (lab, kind, par), s in zip(OWN_NOISE, o.values()):
            mix = (f" (merged, exact mixture toward 1/100: (1 - p)^2 m + (1 - (1 - p)^2) / 100 = "
                   f"{(1 - par) ** 2 * s0['coll'] + (1 - (1 - par) ** 2) / (R * R):.5f})" if kind == "replace" else "")
            se = np.std([d["coll"] for d in dr[lab]], ddof=1) / np.sqrt(reps)
            print(f"    {lab}: {_fmt_t(s)} (merged SE {se:.5f}){mix}")
        col, far, both = (np.array([s[q] for s in o.values()]) for q in ("coll", "far", "both"))
        low = min(o, key=lambda q: o[q]["coll"])
        if s0["coll"] >= 1 / (R * R):
            below = [lab for lab, s in o.items() if s["coll"] < 1 / (R * R)]
            cross = "from above 1/100 to below it (means): " + (", ".join(f"{q} {o[q]['coll']:.5f}" for q in below) + (
                f"; their far errors >= {min(o[q]['far'] for q in below):.4f}") if below else "none")
            one = [(d["far"], q) for q, v in dr.items() for d in v if d["coll"] < 1 / (R * R)]
            cross += (f"; single draws below 1/100: {len(one)} of {sum(len(v) for v in dr.values())}, far errors >= "
                      f"{min(one)[0]:.4f} ({min(one)[1]})" if one else "; no single draw below 1/100")
        else:
            cross = f"already below 1/100; lowest {low} {o[low]['coll']:.5f}"
        print(f"    over the {len(o)} kinds: distant pairs merged {col.min():.5f}-{col.max():.5f}; far errors "
              f"{far.min():.4f}-{far.max():.4f} (raised by every kind: {bool((far > s0['far']).all())}); both right "
              f"{both.min():.4f}-{both.max():.4f} (lowered by every kind: {bool((both < s0['both']).all())}); {cross}")


def modal_noise_tests(M, C, reps=5):
    """prints: the Trained RNN's grid replays (summary_data/grids) with every stream's answer at T replaced or jittered
    (MODAL_NOISE), through modal_stats: distinct modal answers, far-apart merges, modal answer 2+ off; means over reps
    draws, one seed per (checkpoint, kind). Unlike distant pairs merged on the probes (an exact mixture toward 1/100
    under replacement), the modal measures are not mixtures: replacing a share of the streams changes a pair's modal
    answer only where its mode is weak."""
    E = np.load(os.path.join(GRIDS, "fwd_episodes.npz"))
    env, cell = E["env"].astype(int), E["cell"].astype(int)
    F = np.load(os.path.join(GRIDS, "fwd_trained.npz"))
    x = train_x(F["ep"])
    print(f"noise tests on the Trained RNN's grid replays (every stream's answer at T; mean of {reps} draws): distinct "
          f"modal answers / far-apart merges / modal answer 2+ off")
    for xx in (720, 880, 1040, 1200):
        k = int(np.flatnonzero(x == xx)[0])
        a = F["a"][k].astype(np.int64)
        s0 = modal_stats(a, env, cell)
        dev = max(abs(s0["distinct_modal"].mean() - M["trained__distinct_modal"][k].mean()),
                  abs(s0["far_merges"].mean() - M["trained__far_merges"][k].mean()))
        res = {}
        for i, (lab, kind, par) in enumerate(MODAL_NOISE):
            rng = np.random.default_rng([NOISE_SEED, int(F["ep"][k]), 100 + i])
            v = [modal_stats(add_noise(rng, a, None, kind, par), env, cell) for _ in range(reps)]
            res[lab] = [np.mean([s[q].mean() for s in v]) for q in ("distinct_modal", "far_merges", "modal_far")]
        print(f"  {xx:,.0f} itself {s0['distinct_modal'].mean():.1f} / {s0['far_merges'].mean():.1f} / "
              f"{s0['modal_far'].mean():.3f} (= cache within {dev:.0e}); " +
              "; ".join(f"{lab} {d:.1f} / {f:.1f} / {m:.3f}" for lab, (d, f, m) in res.items()))
        d, f, m = np.array(list(res.values())).T
        below = [lab for lab, v in res.items() if v[1] < C["far_merges"] <= s0["far_merges"].mean()]
        print(f"    over the {len(res)} kinds: distinct modal answers at most {d.max():.1f} (random map "
              f"{C['modal']:.1f}); far-apart merges {f.min():.1f}-{f.max():.1f}, from above the random map's "
              f"({C['far_merges']:.1f}) to below it: " + (", ".join(below) if below else "none") +
              f"; modal answer 2+ off raised by every kind: {bool((m > s0['modal_far'].mean()).all())}")


def noise_tests(P, C, reps=5, seed=NOISE_SEED):
    """what noise does to the answer measures (prints): the references' answers with a share of them replaced by
    uniform random answers, or jittered locally (clipped to the grid); at T (t_stats) and, for answer cells lost later,
    at every step (lost_later). Means over `reps` draws."""
    rng = np.random.default_rng(seed)
    r, env = P["r"], P["env"]
    B = ref_beliefs(P)
    ans = {k: lq.argmax(-1) for k, lq in B.items()}                                  # (n, T, 2)

    def replace(a, p):
        a = a.copy()
        sw = rng.random(a.shape[:-1]) < p
        a[sw] = rng.integers(0, R, (int(sw.sum()), 2))
        return a

    def jitter(a, vals, prob):
        return np.clip(a + rng.choice(vals, size=a.shape, p=prob), 0, R - 1)

    def avg(fn):
        out = [fn() for _ in range(reps)]
        return {k: np.mean([o[k] for o in out]) for k in out[0]}

    fmt = _fmt_t
    print(f"noise tests (answers replaced by uniform random answers, or jittered locally; mean of {reps} draws):")
    for k in ("naive", "memfree", "bayes"):
        aT = ans[k][:, -1]
        print(f"  {k} at T itself: {fmt(t_stats(aT, r, env))}")
        for p in (.1, .3):
            print(f"    {100 * p:.0f}% of answers random: {fmt(avg(lambda: t_stats(replace(aT, p), r, env)))}")
        for lab, vals, prob in JITTERS:
            print(f"    jitter {lab}: {fmt(avg(lambda: t_stats(jitter(aT, vals, prob), r, env)))}")
    print(f"  random answers at T: {fmt(avg(lambda: t_stats(rng.integers(0, R, r.shape), r, env)))}")
    print("  answer cells lost later, noise at every step:")
    for k in ("naive", "memfree", "bayes"):
        a = ans[k]
        print(f"    {k} {lost_later(a, env):.2f}; " + ", ".join(
            f"{100 * p:.0f}% random {np.mean([lost_later(replace(a, p), env) for _ in range(reps)]):.2f}"
            for p in (.1, .3)) + f"; jitter +-1, P .3 "
            f"{np.mean([lost_later(jitter(a, (-1, 0, 1), (.15, .7, .15)), env) for _ in range(reps)]):.2f}")
    sim = [lost_later(rng.integers(0, R, ans["naive"].shape), env) for _ in range(8 * reps)]
    print(f"    random answers at every step (simulated, {8 * reps} x 50 environments): {np.mean(sim):.2f} +- "
          f"{np.std(sim, ddof=1) / np.sqrt(len(sim)):.2f} (exact {C['lost']:.3f}); SD of one 50-environment mean "
          f"{np.std(sim, ddof=1):.2f}")
    print("  back-and-forth of the posterior mean, iid Gaussian noise (SD s) added to the log beliefs at every step:")
    for k in ("bayes", "naive"):
        print(f"    {k} {back_forth(B[k], env):.3f}; " + ", ".join(
            f"s {s:g}: {np.mean([back_forth(B[k] + rng.normal(0, s, B[k].shape), env) for _ in range(2)]):.3f}"
            for s in (.5, 1, 2)))


def modal_random(reps=100, seed=NOISE_SEED):
    """random answers (uniform, independent) through the modal pipeline of fold a / f (24 streams per pair, modal_stats:
    ties to the lowest cell), and the same with ties broken at random: distinct modal answers and far-apart merges per
    environment, averaged over reps x 50 maps; and the share of pairs with tied modes"""
    rng = np.random.default_rng(seed)
    E = np.load(os.path.join(GRIDS, "fwd_episodes.npz"))
    env, cell = E["env"].astype(int), E["cell"].astype(int)
    low, rnd = [], []
    for _ in range(reps):
        a = rng.integers(0, R, (len(env), 2))
        s = modal_stats(a, env, cell)
        low.append([s["distinct_modal"].mean(), s["far_merges"].mean(), s["modal_ties"]])
        cnt = np.bincount((env * R * R + cell) * R * R + a[:, 0] * R + a[:, 1], minlength=NE * R ** 4)
        modal = (cnt.reshape(NE, R * R, R * R) + rng.random((NE, R * R, R * R)) * .5).argmax(-1)
        rnd.append([np.mean([len(np.unique(m)) for m in modal]),
                    np.mean([((m[:, None] == m[None]) & FAR_CELL_PAIRS).sum() for m in modal])])
    return np.mean(low, 0), np.mean(rnd, 0)


def boot_weights(nb=NBOOT, seed=BOOT_SEED):
    """(nb, NE) weights of the 50 environments resampled with replacement (rows sum to 1)"""
    idx = np.random.default_rng(seed).integers(0, NE, (nb, NE))
    return np.stack([np.bincount(i, minlength=NE) for i in idx]).astype(np.float64) / NE


def stay_cross(x, y, level, below):
    """the training episode (log-interpolated between checkpoints) after which y stays past level (below it if below,
    else above it) to the end of training; nan = not past at the end; x[0] = past from the start"""
    y = np.asarray(y, dtype=np.float64)
    past = (y < level) if below else (y > level)
    if not past[-1]:
        return np.nan
    notp = np.flatnonzero(~past)
    if len(notp) == 0:
        return float(x[0])
    k = notp[-1]
    f = (level - y[k]) / (y[k + 1] - y[k])
    return float(np.exp(np.log(x[k]) + f * (np.log(x[k + 1]) - np.log(x[k]))))


def _q(v):
    """5 and 95% of bootstrap crossings ('never' = later than any time) and the share of resamples that never cross"""
    v = np.where(np.isnan(v), np.inf, np.asarray(v, dtype=np.float64))
    s = np.sort(v)
    return s[int(round(.05 * (len(s) - 1)))], s[int(round(.95 * (len(s) - 1)))], float(np.isinf(v).mean())


def cross(x, Y, ref, below, W):
    """(crossing, 5%, 95%, share never) of the environment-mean curve of Y (K, NE) past a reference: per-environment
    values (NE,) (their mean; resampled with the same environments, i.e. paired) or a fixed level (float)"""
    if np.ndim(ref):
        L, Lb = float(np.mean(ref)), W @ ref
    else:
        L, Lb = float(ref), np.full(len(W), float(ref))
    Yb = W @ np.asarray(Y, dtype=np.float64).T
    c = stay_cross(x, Y.mean(-1), L, below)
    return (c, *_q(np.array([stay_cross(x, Yb[b], Lb[b], below) for b in range(len(W))])))


def half_level(y, kind):
    """the level of the half-way crossing and its direction: half of the curve's own change from the initial network
    (y[0]) to the end of training (mean of the last 3 checkpoints); 'log': the same on the log scale (geometric mean);
    'hump': half of the way back from the curve's maximum to its end"""
    y1 = float(np.mean(y[-3:]))
    if kind == "log":
        return float(np.exp(.5 * (np.log(y[0]) + np.log(y1)))), y1 < y[0]
    if kind == "hump":
        return float(np.max(y) - .5 * (np.max(y) - y1)), True
    return float(y[0] + .5 * (y1 - y[0])), y1 < y[0]


def half_cross(x, Y, kind, W):
    """level, (crossing, 5%, 95%, share never) of the half-way crossing (half_level recomputed in every resample)"""
    y = Y.mean(-1)
    lev, below = half_level(y, kind)
    Yb = W @ np.asarray(Y, dtype=np.float64).T
    cb = [stay_cross(x, yb, *half_level(yb, kind)) for yb in Yb]
    return lev, (stay_cross(x, y, lev, below), *_q(np.array(cb)))


def boot_band(Y, W):
    """5-95% of the environment-mean curve of Y (K, NE) at every checkpoint"""
    Yb = W @ np.asarray(Y, dtype=np.float64).T
    return np.percentile(Yb, 5, 0), np.percentile(Yb, 95, 0)


# ═══════════════════════════════════════════════════════════════════
# Series
# ═══════════════════════════════════════════════════════════════════

def env_curve(d, key):
    """per-environment values of quantity key at T: (K, NE) for a run's cache, (NE,) for a reference's"""
    return {"acc": lambda: d["right_env"][..., -1, :],
            "far": lambda: d["far_env"][..., -1, :],
            "coll": lambda: d["coll_env"][..., -1, :],
            "distinct": lambda: d["distinct"][..., -1, :].astype(np.float64),
            "eff": lambda: d["eff"][..., -1, :],
            "merged": lambda: (d["distinct"].max(-2) - d["distinct"][..., -1, :]).astype(np.float64),
            "path": lambda: d["path_env"],
            "net": lambda: d["net_env"],
            "back": lambda: d["path_env"] - d["net_env"],
            "early": lambda: d["early_env"],
            "late": lambda: d["late_env"],
            "visited": lambda: d["visited_env"]}[key]()


def series(d, key):
    """the plotted per-checkpoint value of quantity key from a run's (or a reference's) cache"""
    if key == "drift":
        return np.linalg.norm(np.diff(d["mbar"], axis=-2), axis=-1).sum(-1)
    if key == "cover":
        return d["cover"].mean(-1)
    if key == "spread":
        return d["spread"].mean(-1)
    if key == "peak":
        return d["distinct"].mean(-1).argmax(-1) + 1
    return env_curve(d, key).mean(-1)


def _ref_view(ref, k):
    return {key.split("__", 1)[1]: v for key, v in ref.items() if key.startswith(k + "__")}


def passes_naive(D=None):
    """first checkpoint at which the Trained RNN's goal accuracy at T exceeds naive's (= fig_entanglement_driver's
    passes_naive) and the checkpoint before it; naive's goal accuracy"""
    P = _probe_set()
    naive_goal = P["right_naive"][np.arange(len(P["goal"])), P["goal"]].mean()
    d = (D or {"trained": build_run("trained")})["trained"]
    k = int(np.argmax(d["acc_goal"] > naive_goal))
    return d["x"][k], d["x"][k - 1], naive_goal, d["acc_goal"][k - 1], d["acc_goal"][k]


def _gap():
    """td_tt_gap: the shared conventions of the cross-topic synthesis (resamples, crossings, export)"""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import td_tt_gap
    return td_tt_gap


def landmark():
    """the passing-naive landmark re-timed (a check, not a new landmark): the Trained RNN's goal accuracy at T per
    environment at every checkpoint (ck files' stored correctness), the training episode after which it stays above
    naive's (stays-past rule, log-interpolated) and its 5-95% interval over td_tt_gap.boot_counts() (naive resampled
    on the same environments); also the first checkpoint above naive's in every resample"""
    gap = _gap()
    P = _probe_set()
    n = len(P["goal"])
    rows, env = np.arange(n), P["env"]
    cnt = np.bincount(env, minlength=NE).astype(np.float64)
    nv = np.bincount(env, P["right_naive"][rows, P["goal"]].astype(np.float64), NE) / cnt
    Y, eps, dif = [], [], []
    for f in _ck_files("trained"):
        z = np.load(f)
        rt = z["right"][:, -1][rows, P["goal"]].astype(np.float64)
        Y.append(np.bincount(env, rt, NE) / cnt)
        eps.append(int(z["episode"]))
        dif.append(abs(rt.mean() - float(z["acc"][-1])))
    x, C = gap.train_x(np.array(eps)), gap.boot_counts()
    full, boot = gap.env_mean(np.array(Y).T, C)
    nf, nb = gap.env_mean(nv, C)
    cb = gap.crossing(x, boot, nb)
    first = x[(boot > nb[:, None]).argmax(1)]
    return {"cross": gap.crossing(x, full, nf), "interval": gap.interval(cb), "naive": float(nf),
            "first": (np.percentile(first, 5), np.percentile(first, 95)), "max_dif": max(dif)}


def peak_step_cross(d, W, level=15.5):
    """the training episode after which the peak step of the environment-averaged D_t stays past level (the second
    half of the episode), with its 5-95% interval (the profile recomputed in every resample)"""
    prof = d["distinct"].astype(np.float64)                                         # (K, T, NE)
    pk = prof.mean(-1).argmax(-1) + 1.
    pkb = np.einsum("bn,ktn->bkt", W, prof).argmax(-1) + 1.
    return (stay_cross(d["x"], pk, level, False),
            *_q(np.array([stay_cross(d["x"], v, level, False) for v in pkb])))


CROSS_KEYS = (("acc", False), ("far", True), ("coll", True), ("distinct", False), ("eff", False), ("merged", True))


def all_crossings(D, R_, C, W):
    """every reference / chance crossing of the headline curves (probes, 114 checkpoints), per run"""
    out = {}
    for run in RUNS:
        x = D[run]["x"]
        for key, below in CROSS_KEYS:
            Y = env_curve(D[run], key)
            for k in REFS:
                out[run, key, k] = cross(x, Y, env_curve(R_[k], key), below, W)
            if key in ("acc", "far", "coll", "distinct", "eff"):
                lev = {"acc": C["both"], "far": C["far"], "coll": C["coll"], "distinct": C["D"], "eff": C["eH"]}[key]
                out[run, key, "chance"] = cross(x, Y, lev, below, W)
    return out


TL = (("acc", "both right at T", "answers", "rise"),
      ("far", "far errors at T", "answers", "fall"),
      ("coll", "distant pairs merged at T", "answers", "log"),
      ("merged", "answer cells lost later in the episode", "answers", "hump"),
      ("back", "back-and-forth of the posterior mean", "movement", "rise"))


def build_timeline():
    """timeline.npz for the cross-topic synthesis figure (see the module docstring)"""
    ref = build_refs()
    D = {run: build_run(run) for run in RUNS}
    R_ = {k: _ref_view(ref, k) for k in REFS}
    C = chance_levels()
    W = boot_weights()
    x = D["trained"]["x"]
    assert all((D[run]["x"] == x).all() for run in RUNS)
    rows = {k: [] for k in ("names", "run", "family", "value", "lo", "hi", "bayes", "naive", "level", "half",
                            "half_lo", "half_hi", "p_never", "chance", "scale", "half_rule", "note")}
    note = {"coll": f"the half-way level lies above naive's {float(series(R_['naive'], 'coll')):.4f}: uniform noise "
                    f"alone can carry the curve below it (it moves this measure toward 1/100), and so can local "
                    f"answer jitter (it lowers this measure; noise tests in numbers): not a noise-proof event",
            "merged": "the half-way level lies above chance (random answers at every step: 5.55), toward which noise "
                      "moves this measure: not a noise-proof event",
            "back": "noise in the beliefs raises it (noise tests in numbers): not a noise-proof event"}
    rule = {"rise": "half of its own change, initial network to end (mean of last 3 checkpoints)",
            "fall": "half of its own change, initial network to end (mean of last 3 checkpoints)",
            "log": "half of its own change on the log scale (geometric mean of initial and end)",
            "hump": "half of the way back from its maximum to its end"}
    for run in RUNS:
        for key, name, fam, kind in TL:
            Y = env_curve(D[run], key)
            lo, hi = boot_band(Y, W)
            lev, (c, clo, chi, pn) = half_cross(x, Y, kind, W)
            rows["names"].append(name)
            rows["run"].append(run)
            rows["family"].append(fam)
            rows["value"].append(Y.mean(-1))
            rows["lo"].append(lo)
            rows["hi"].append(hi)
            rows["bayes"].append(float(series(R_["bayes"], key)))
            rows["naive"].append(float(series(R_["naive"], key)))
            rows["level"].append(lev)
            rows["half"].append(c)
            rows["half_lo"].append(clo if np.isfinite(clo) else np.nan)
            rows["half_hi"].append(chi if np.isfinite(chi) else np.nan)
            rows["p_never"].append(pn)
            rows["chance"].append({"acc": C["both"], "far": C["far"], "coll": C["coll"]}.get(key, np.nan))
            rows["scale"].append("log" if kind == "log" else "linear")
            rows["half_rule"].append(rule[kind])
            rows["note"].append(note.get(key, ""))
    out = {k: np.array(v) for k, v in rows.items()}
    out["x"], out["episode"] = x, D["trained"]["episode"]
    path = os.path.join(OUT, "timeline.npz")
    np.savez(path, **out)
    print("saved", path)
    return out


# the measures exported for the cross-topic synthesis (td_tt_gap): (key, name, shape); all family 'answers'
SYN = (("acc", "goal and context both right at T", "rise"),
       ("far", "far errors at T", "rise"),
       ("coll", "distant pairs merged at T", "rise"),
       ("merged", "answer cells lost later in the episode", "return"))


def synthesis():
    """-> through_training/latent/synthesis.npz (td_tt_gap.export: the gap fraction toward Bayes, its crossings of 0
    = passes naive's level for good and .5 = past the naive-Bayes midpoint for good, on the shared environment
    resamples td_tt_gap.boot_counts(), paired): SYN for the Trained RNN and the ESN. Every row is a per-environment
    statistic averaged over the 50 environments (both right, far errors: shares of the environment's 200 probes;
    distant pairs merged: the share of its 18,432 far-apart probe pairs, the same number in every environment, so the
    mean of shares = the ratio of sums; cells lost later: its own statistic), so every resample is the count-weighted
    mean of the per-environment values (td_tt_gap.env_mean), with naive's and Bayes's levels resampled on the same
    environments. Then checks the export against this module's own crossing code."""
    gap = _gap()
    P = _probe_set()
    n = len(P["env"])
    assert np.array_equal(P["env"], np.arange(n) // gap.PER_ENV), "environment e = probe row // 200"
    npair = np.bincount(far_pairs(P["r"], P["env"])[2], minlength=NE)
    assert (npair == npair[0]).all()
    ref = build_refs()
    D = {run: build_run(run) for run in RUNS}
    R_ = {k: _ref_view(ref, k) for k in REFS}
    C = chance_levels()
    x = gap.train_x(D["trained"]["episode"])
    assert all(np.array_equal(gap.train_x(D[run]["episode"]), x) and np.array_equal(D[run]["x"], x) for run in RUNS)
    CB = gap.boot_counts()
    print(f"environment index = probe row // {gap.PER_ENV} for all {n:,} probes (the cache's per-environment values "
          f"are indexed by it); far-apart probe pairs per environment: {npair[0]:,} in every environment")
    jr, jE, jlab, jtxt = JITTER_EXAMPLE
    J = own_noise(P, jr, jE)                                     # the same draws as the noise tests of numbers
    jit = (f"the {NAME[jr]}'s own answers at {train_x(jE):,.0f} go from {J['none']['coll']:.4f} to "
           f"{J[jlab]['coll']:.4f} with {jtxt}, while their far errors rise from {J['none']['far']:.3f} to "
           f"{J[jlab]['far']:.3f}")
    sample = {"coll": f"probe pairs: 50 environments x {npair[0]:,} pairs of probes whose latent pairs are 2+ apart, "
                      f"answers at T",
              "merged": "probes: 50 environments x 200 episodes (100 latent pairs x 2 trajectories), answers at all "
                        "30 steps"}
    rows = []
    for run in RUNS:
        for key, name, shape in SYN:
            Y = env_curve(D[run], key).T                                             # (NE, K)
            value, boot = gap.env_mean(Y, CB)
            nv, nb = gap.env_mean(env_curve(R_["naive"], key), CB)
            bv, bb = gap.env_mean(env_curve(R_["bayes"], key), CB)
            ident = gap.env_mean(Y, np.ones((1, NE)))[1][0]
            print(f"  {run:7s} {name:40s} all counts 1 vs the full-sample curve: max |diff| "
                  f"{np.abs(ident - value).max():.1e}; full sample vs the plotted curve: max |diff| "
                  f"{np.abs(value - series(D[run], key)).max():.1e}")
            note = {"acc": f"noise lowers it (random answers {C['both']:.3f}): neither crossing can come from noise",
                    "far": f"truth-referenced; the gap fraction rises as far errors fall toward Bayes's ({bv:.3f}); "
                           f"noise raises far errors (random answers {C['far']:.3f}): neither crossing can come from "
                           f"noise",
                    "coll": f"neither crossing is noise-proof on its own: c0 (below naive's {nv:.4f}) can come from "
                            f"uniform noise, which moves this measure toward chance (1/100); c50 (the naive-Bayes "
                            f"midpoint {(nv + bv) / 2:.5f}, just below 1/100) is out of reach of uniform noise but not "
                            f"of local answer jitter, which lowers this measure ({jit}). c50 counts only together "
                            f"with far errors that keep falling (every noise tested raises them)",
                    "merged": f"a hump: rises far above naive's level ({nv:.2f}) and comes back toward Bayes's "
                              f"({bv:.2f}); c0 can come from noise (random answers at every step sit at naive's "
                              f"level, {C['lost']:.2f}); c50 (below the naive-Bayes midpoint {(nv + bv) / 2:.2f}) "
                              f"cannot: noise moves this measure toward {C['lost']:.2f} or above it (noise tests in "
                              f"numbers)"}[key]
            rows.append(dict(name=name, run=run, family="answers", shape=shape, paired=True, note=note,
                             sample=sample.get(key, "probes: 50 environments x 200 episodes (100 latent pairs x 2 "
                                                    "trajectories), answers at T"),
                             value=value, boot=boot, naive=float(nv), bayes=float(bv), naive_boot=nb,
                             bayes_boot=bb))
    path = gap.export("latent", x, rows)
    # the export against this module's crossing code (stay_cross): c50 at the naive-Bayes midpoint on the same
    # resamples (weights = counts / 50); c0 against the naive-level crossings of numbers (own resamples)
    S = gap.load("latent")
    W, Wown = CB / NE, boot_weights()
    print("check against this module's crossing code:")
    for i, (run, (key, name, shape)) in enumerate((run, s) for run in RUNS for s in SYN):
        Y = env_curve(D[run], key)                                                   # (K, NE)
        ne, be = env_curve(R_["naive"], key), env_curve(R_["bayes"], key)
        below = be.mean() < ne.mean()
        mid = (ne + be) / 2
        Yb, Lb = W @ Y.T, W @ mid
        cb = np.array([stay_cross(x, Yb[b], Lb[b], below) for b in range(len(W))])
        c = stay_cross(x, Y.mean(-1), mid.mean(), below)
        lo, hi, _ = gap.interval(cb)
        qlo, qhi, _ = _q(cb)
        own = cross(x, Y, ne, below, Wown)
        e = S["c50_boot"][i]
        diff = ~((np.isnan(cb) & np.isnan(e)) | np.isclose(cb, e, rtol=1e-9, atol=0))
        tie = np.isclose(Yb, Lb[:, None], rtol=0, atol=1e-12).any(1)              # the curve exactly at the level
        print(f"  {run:7s} {name:40s} c50 export {S['c50'][i]:,.1f} [{S['c50_lo'][i]:,.0f}-{S['c50_hi'][i]:,.0f}] | "
              f"stay_cross at the midpoint {c:,.1f} [{lo:,.0f}-{hi:,.0f}] (this module's percentile rule "
              f"[{qlo:,.0f}-{qhi:,.0f}]); resamples that differ {diff.sum()} (with the curve exactly at the midpoint "
              f"at a checkpoint: {(diff & tie).sum()}) | c0 export {S['c0'][i]:,.1f} [{S['c0_lo'][i]:,.0f}-"
              f"{S['c0_hi'][i]:,.0f}] vs naive-level crossing (own resamples) {_fmt_cross(own)}")
    return path


# ═══════════════════════════════════════════════════════════════════
# Numbers and checks
# ═══════════════════════════════════════════════════════════════════

def _fmt_cross(c):
    t, lo, hi, never = c
    if np.isnan(t):
        return f"never (in {100 * never:.0f}% of resamples)" if never > .95 else \
            f"not for good at the end ({100 * never:.0f}% of resamples never)"
    hs = "never" if np.isinf(hi) else f"{hi:,.0f}"
    s = f"{t:,.0f} [{lo:,.0f}-{hs}]"
    return s + (f" (never in {100 * never:.0f}%)" if never > 0 else "")


def _at(x, v, pick):
    ii = [int(np.argmin(np.abs(x - p))) for p in pick]
    return " ".join(f"{x[i]:.0f}:{v[i]:.3g}" for i in ii)


def _paired(Y, ref):
    """mean and SE over environments of the per-environment difference"""
    d = Y - ref
    return d.mean(-1), d.std(-1, ddof=1) / np.sqrt(d.shape[-1])


def _matched_pooled(aT, env, rng, ndraw=2000):
    """answer cells within an environment (200 probes) vs 200 probes drawn at random across environments, vs all
    10,000 probes pooled: D and e^H"""
    ac = aT[:, 0] * R + aT[:, 1]
    n = len(ac)

    def de(c):
        k = np.bincount(c, minlength=R * R).astype(np.float64)
        return (k > 0).sum(), _entropy_eff(k)

    w = np.mean([de(ac[env == e]) for e in range(NE)], 0)
    p = np.mean([de(ac[rng.choice(n, 200, replace=False)]) for _ in range(ndraw)], 0)
    return w, p, de(ac)


def numbers():
    ref, M, S = build_refs(), build_fwd(), build_same_as_naive()
    D = {run: build_run(run) for run in RUNS}
    R_ = {k: _ref_view(ref, k) for k in REFS}
    C = chance_levels()
    W = boot_weights()
    P = _probe_set()
    xc, xp, ng, a0, a1 = passes_naive(D)
    print(f"landmark: Trained RNN goal accuracy at T passes naive's ({ng:.3f}) between the {xp:.0f} ({a0:.3f}) and "
          f"{xc:.0f} ({a1:.3f}) checkpoints (drawn at {xc:.0f})")
    print(f"chance (random answers): D {C['D']:.2f} (exact; simulated {C['D_sim']:.2f}), e^H {C['eH']:.2f}, categories "
          f"{np.round(C['cat'], 4)} (far errors {C['far']:.4f}), both right {C['both']:.3f}, distant pairs merged "
          f"{C['coll']:.3f}; random map: distinct modal answers {C['modal']:.1f}, far-apart merges "
          f"{C['far_merges']:.1f}, modal far share {C['modal_far']:.3f}; truth + +-1 jitter (P .3, no folds): D "
          f"{C['jitter_D']:.1f}, e^H {C['jitter_eH']:.1f}")
    print(f"chance, answer cells lost later (random answers at every step, independent; exact from the occupancy "
          f"distribution): {C['lost']:.4f} (E[D] {C['D_occ']:.4f}, = 100 (1 - .99^200))")
    keys = ("acc", "far", "coll", "distinct", "eff", "merged", "peak", "path", "drift", "net", "back", "early",
            "late", "visited", "cover", "spread")
    print("references (Bayes / memory-free Bayes / naive):")
    for key in keys:
        print(f"  {key:9s} " + " / ".join(f"{float(series(R_[k], key)):.4g}" for k in REFS))
    print(f"  Bayes joint MAP: distinct {ref['map__distinct'].mean():.1f}, eff {ref['map__eff'].mean():.1f}, "
          f"both right {float(ref['map__acc_joint']):.3f}")
    print("  categories at T [right, neighbour, one variable far, both far]: " +
          "; ".join(f"{k} {np.round(R_[k]['cat'][-1], 3)}" for k in REFS))
    print("  one-variable mirror (9 - r, >= 2 away) at T: " + ", ".join(f"{k} {R_[k]['mirror'][-1]:.3f}" for k in REFS))
    print("  pooled coverage at T: " + ", ".join(f"{k} {R_[k]['cover'][-1]:.1f}" for k in REFS) +
          "; within an environment at T: " + ", ".join(f"{k} {R_[k]['cover_env'][-1].mean():.1f}" for k in REFS))
    print(f"  modal answers per environment (24 streams): Bayes {M['bayes__distinct_modal'].mean():.1f}, naive "
          f"{M['naive__distinct_modal'].mean():.1f}; far-apart merges Bayes {M['bayes__far_merges'].mean():.1f}, naive "
          f"{M['naive__far_merges'].mean():.1f}; modal answer 2+ off: Bayes {M['bayes__modal_far'].mean():.3f}, naive "
          f"{M['naive__modal_far'].mean():.3f}; merged pairs median Bayes {np.median(M['bayes__merged_pairs']):.0f}, "
          f"naive {np.median(M['naive__merged_pairs']):.0f}")
    pick = (10, 20, 40, 80, 200, 320, 400, 480, 560, 640, 720, 800, 880, 960, 1040, 1200, 1600, 2400, 4000, 8000,
            16000, 39996)
    for run in RUNS:
        d = D[run]
        x = d["x"]
        print(f"{run}: checkpoints {len(x)}; fp16 argmax ties at most {d['ties'][:, 0].max():.4f} of (probe, step, "
              f"variable), {d['ties'][:, 1].max():.4f} at T")
        for key in keys:
            v = series(d, key)
            print(f"  {key:9s} " + _at(x, v, pick))
            j = int(np.argmax(v))
            print(f"            max {v[j]:.4g} at {x[j]:.0f}; end {v[-1]:.4g}")
        cat = d["cat"][:, -1]
        for c, name in enumerate(("right", "neighbour", "one far", "both far")):
            print(f"  cat {name:9s} " + _at(x, cat[:, c], pick))
    # crossings
    print("crossings (training episode after which the curve stays past the level, log-interpolated; [5-95%] over "
          f"{NBOOT} resamples of the 50 environments, references resampled with them):")
    CR = all_crossings(D, R_, C, W)
    for (run, key, k), c in CR.items():
        print(f"  {run:7s} {key:8s} past {k:8s}: {_fmt_cross(c)}")
    for run in RUNS:
        print(f"  {run:7s} peak step of D_t in the second half of the episode (> 15.5): "
              f"{_fmt_cross(peak_step_cross(D[run], W))}")
    print("half-way crossings (timeline):")
    for run in RUNS:
        for key, name, fam, kind in TL:
            lev, c = half_cross(D[run]["x"], env_curve(D[run], key), kind, W)
            print(f"  {run:7s} {name:40s} level {lev:.4g} ({kind}): {_fmt_cross(c)}")
    # paired per-environment differences at selected checkpoints
    print("paired per-environment differences, network minus reference (mean +- SE over the 50 environments):")
    for run, key, k, xs in (("trained", "distinct", "naive", (560, 640, 720, 800)),
                            ("trained", "eff", "naive", (560, 640, 720, 800)),
                            ("trained", "far", "naive", (480, 560, 640, 720)),
                            ("trained", "coll", "naive", (560, 640, 720)),
                            ("trained", "acc", "naive", (640, 720, 800, 880)),
                            ("trained", "merged", "naive", (720, 800, 880, 960, 1120, 1200, 1600)),
                            ("trained", "merged", "memfree", (880, 1120, 1200, 1600)),
                            ("trained", "distinct", "memfree", (1200, 1280, 1600))):
        x = D[run]["x"]
        mu, se = _paired(env_curve(D[run], key), env_curve(R_[k], key))
        ii = [int(np.argmin(np.abs(x - p))) for p in xs]
        print(f"  {run} {key} - {k}: " + ", ".join(f"{x[i]:.0f}: {mu[i]:+.4g} +- {se[i]:.2g}" for i in ii))
    for key in ("acc", "far", "coll", "distinct", "eff", "merged", "back", "path"):
        mu, se = _paired(env_curve(D["echo"], key)[-1], env_curve(R_["naive"], key))
        print(f"  echo end {key} - naive: {mu:+.4g} +- {se:.2g}")
    mu, se = _paired(D["echo"]["cat_env"][-1].T, R_["naive"]["cat_env"].T)
    print(f"  echo end categories - naive: {np.round(mu, 3)} +- {np.round(se, 3)}")
    xt = D["trained"]["x"]
    for key in ("far", "acc", "distinct", "eff", "coll"):
        Y, rb = env_curve(D["trained"], key), env_curve(R_["bayes"], key)
        mu, se = _paired(Y[-1], rb)
        d10 = (Y[-10:] - rb).mean(0)                                                 # per environment, last 10
        print(f"  trained end {key} - bayes: {mu:+.4g} +- {se:.2g}; per environment over the last 10 checkpoints "
              f"({xt[-10]:,.0f}-{xt[-1]:,.0f}): {d10.mean():+.4g} +- {d10.std(ddof=1) / np.sqrt(NE):.2g}")
    noise_tests(P, C)
    own_noise_tests(P, D)
    low, rnd = modal_random()
    print(f"random answers through the modal pipeline (24 streams per pair, 100 x 50 maps): ties to the lowest cell "
          f"{low[0]:.1f} distinct modal answers, {low[1]:.1f} far-apart merges ({100 * low[2]:.0f}% of pairs tied); "
          f"ties broken at random {rnd[0]:.1f} / {rnd[1]:.1f}; random map {C['modal']:.1f} / {C['far_merges']:.1f}")
    modal_noise_tests(M, C)
    # sample-matched pooled vs within
    rng = np.random.default_rng(MATCH_SEED)
    print("answer cells within an environment vs 200 probes drawn across environments (2,000 draws) vs all 10,000:")
    for k, lq in ref_beliefs(P).items():
        w, p, a = _matched_pooled(lq[:, -1].argmax(-1), P["env"], rng)
        print(f"  {k:8s}: within D {w[0]:.1f} e^H {w[1]:.1f} | across D {p[0]:.1f} e^H {p[1]:.1f} | all D {a[0]} "
              f"e^H {a[1]:.1f}")
    for run, eps in (("trained", (0, 100, 250, 500, 700, 1100, 49995)), ("echo", (25, 500, 1100, 49995))):
        for E in eps:
            z = np.load(os.path.join(AN, run, f"ck_{E:05d}.npz"))
            w, p, a = _matched_pooled(z["logq"][:, -1].argmax(-1), P["env"], rng)
            print(f"  {run} {train_x(E):.0f}: within D {w[0]:.1f} e^H {w[1]:.1f} | across D {p[0]:.1f} e^H "
                  f"{p[1]:.1f} | all D {a[0]} e^H {a[1]:.1f}")
    # a failed control: subtracting the probe-averaged log belief of each step before taking answers
    for k, lq in ref_beliefs(P).items():
        lq = lq - logsumexp(lq, -1, keepdims=True)
        st = answer_stats(lq - lq.mean(0, keepdims=True), P["r"], P["env"])
        print(f"  {k}: answer cells at T after subtracting the probe-averaged log belief "
              f"{st['distinct'][-1].mean():.1f} (raw {float(series(R_[k], 'distinct')):.1f})")
    # modal answers and same-as-naive (19 checkpoints)
    for run in RUNS:
        mx = M[f"{run}__x"]
        Y = M[f"{run}__distinct_modal"].astype(np.float64)
        print(f"{run} modal answers per env: " + " ".join(f"{a:.0f}:{b:.1f}" for a, b in zip(mx, Y.mean(-1))))
        print("  far-apart merges: " + " ".join(f"{a:.0f}:{b:.1f}" for a, b in zip(mx, M[f'{run}__far_merges']
                                                                                      .mean(-1))))
        print("  modal answer 2+ off: " + " ".join(f"{a:.0f}:{b:.3f}" for a, b in zip(mx, M[f'{run}__modal_far']
                                                                                         .mean(-1))))
        print("  environment 14: " + " ".join(f"{a:.0f}:{b}" for a, b in zip(mx, M[f'{run}__distinct_modal'][:, 14])))
        print(f"  modal answers past naive's ({M['naive__distinct_modal'].mean():.1f}): "
              f"{_fmt_cross(cross(mx, Y, M['naive__distinct_modal'].astype(np.float64), False, W))}; past the random "
              f"map's ({C['modal']:.1f}): {_fmt_cross(cross(mx, Y, C['modal'], False, W))}")
        Fm = M[f"{run}__far_merges"].astype(np.float64)
        print(f"  far-apart merges below naive's ({M['naive__far_merges'].mean():.1f}): "
              f"{_fmt_cross(cross(mx, Fm, M['naive__far_merges'].astype(np.float64), True, W))}; below the random "
              f"map's ({C['far_merges']:.1f}): {_fmt_cross(cross(mx, Fm, C['far_merges'], True, W))}")
        mu, se = _paired(Y[-1], M["naive__distinct_modal"].astype(np.float64))
        print(f"  end minus naive (paired): {mu:+.2f} +- {se:.2f}")
        sx = S[f"{run}__x"]
        print("  same wrong modal answer as naive / control: " + " ".join(
            f"{a:.0f}:{b:.3f}/{c:.3f}" for a, b, c in zip(sx, S[f'{run}__same'], S[f'{run}__control'])))
        print("  ... given the network wrong / control: " + " ".join(
            f"{a:.0f}:{b:.3f}/{c:.3f}" for a, b, c in zip(sx, S[f'{run}__same_w'], S[f'{run}__control_w'])))
        print("  ... network wrong on these cells: " + " ".join(f"{a:.0f}:{b:.3f}" for a, b in
                                                                 zip(sx, S[f'{run}__wrong'])))
        hw = np.array([S[f"{run}__hit_env"][-1], S[f"{run}__wrong_env"][-1], S[f"{run}__cw_hit_env"][-1],
                       S[f"{run}__cw_n_env"][-1]])                                   # (4, NE) at the end
        bw = W @ hw.T                                                                # (B, 4)
        sw, cw = bw[:, 0] / bw[:, 1], bw[:, 2] / bw[:, 3]
        rr = S[f"{run}__same_w"] / S[f"{run}__control_w"]
        k1 = int(np.argmin(np.abs(sx - 120)))
        print(f"  end, given the network wrong: {S[f'{run}__same_w'][-1]:.3f} [{np.percentile(sw, 5):.3f}-"
              f"{np.percentile(sw, 95):.3f}] vs control {S[f'{run}__control_w'][-1]:.3f} [{np.percentile(cw, 5):.3f}-"
              f"{np.percentile(cw, 95):.3f}]; ratio to control from 120 on: {rr[k1:].min():.2f}-{rr[k1:].max():.2f} "
              f"(" + " ".join(f"{a:.0f}:{b:.2f}" for a, b in zip(sx, rr)) + ")")
        hb = (W @ S[f"{run}__hit_env"].T) / (W @ S["n_sel_env"])[:, None]
        kb = sx[hb.argmax(1)]
        k0 = int(np.argmax(S[f"{run}__same"]))
        print(f"  peak of same-as-naive {S[f'{run}__same'][k0]:.3f} at {sx[k0]:.0f}; the peak checkpoint over "
              f"resamples [5-95%]: {np.percentile(kb, 5):.0f}-{np.percentile(kb, 95):.0f}")
    print(f"cells where naive's modal answer is wrong and Bayes's right: {S['n_cells']} of {NE * R * R}")
    nd = M["naive__distinct_modal"]
    env = int(np.argsort(nd, kind="stable")[len(nd) // 2])
    print(f"example environment (naive's median number of modal answers): {env}, naive {nd[env]}, quartiles "
          f"{np.percentile(nd, 25):.2f}-{np.percentile(nd, 75):.2f}; modal-answer ties at most: " +
          ", ".join(f"{k} {np.max(M[k + '__modal_ties']):.3f}" for k in ("trained", "echo", "bayes", "naive")))
    for run in RUNS:
        pk = series(D[run], "peak")
        print(f"{run}: peak step of D_t from 20 training episodes on: {pk[1:].min()}-{pk[1:].max()}; before 560: "
              f"{pk[1:][D[run]['x'][1:] < 560].min()}-{pk[1:][D[run]['x'][1:] < 560].max()}")
    # the ladder: the logit-increment regression on N, W = SW - N, M = D - SW (ck files: each step's normal equations)
    for name, fit in ladder().items():
        x = D["trained"]["x"]
        print(f"ladder N / W / M, {name}: " + ", ".join(
            f"{x[i]:.0f}: " + " / ".join(f"{v:.3f}" for v in fit[i])
            for i in [int(np.argmin(np.abs(x - p))) for p in (60, 100, 200, 400, 560, 880, 39996)]))
        print("  half the final value for good (stays-past, log-interpolated): " + " / ".join(
            f"{stay_cross(x, fit[:, j], fit[-1, j] / 2, False):,.0f}" for j in range(3)) + "; first checkpoint at "
              "it: " + " / ".join(f"{x[int(np.argmax(fit[:, j] >= fit[-1, j] / 2))]:,.0f}" for j in range(3)))


def ladder():
    """the ladder's coefficients (K, 3) at every checkpoint of the Trained RNN: fitted within each step where X^T X is
    invertible (step 1 is not: M = 0 there), then averaged over those steps; the same over steps 3-30 only; and the
    pooled fit (one regression on the normal equations summed over the 30 steps)"""
    A, B = [], []
    for f in _ck_files("trained"):
        z = np.load(f)
        A.append(z["reg_xtx"])
        B.append(z["reg_xtd"])
    A, B = np.array(A), np.array(B)                                                  # (K, T, 3, 3), (K, T, 3)
    ok = np.linalg.matrix_rank(A[0]) == 3                                            # (T,)
    assert all(np.array_equal(np.linalg.matrix_rank(a) == 3, ok) for a in A)
    step = np.arange(1, T + 1)[ok]
    per = np.linalg.solve(A[:, ok], B[:, ok][..., None])[..., 0]                     # (K, steps, 3)
    return {f"per step, then averaged (steps {step[0]}-{step[-1]})": per.mean(1),
            "per step, then averaged (steps 3-30)": per[:, step >= 3].mean(1),
            "pooled (summed normal equations)": np.linalg.solve(A.sum(1), B.sum(1)[..., None])[..., 0]}


def checks():
    """independent recomputations of the cached statistics (prints)"""
    P = _probe_set()
    ref = build_refs()
    D = {run: build_run(run) for run in RUNS}
    C = chance_levels()
    # 1 the true cells themselves: every environment uses all 100 cells, 2 probes each
    tr = np.zeros((len(P["r"]), T, 2, R)) - 50.
    tr[np.arange(len(P["r"])), :, 0, P["r"][:, 0]] = 0.
    tr[np.arange(len(P["r"])), :, 1, P["r"][:, 1]] = 0.
    st = answer_stats(tr, P["r"], P["env"])
    print(f"truth as an observer: distinct {st['distinct'].mean():.2f}, eff {st['eff'].mean():.2f}, both right "
          f"{st['acc_joint'].mean():.3f}, categories {st['cat'][-1]}, far errors {st['far_env'].mean():.3f}, distant "
          f"pairs merged {st['coll_env'].mean():.4f}, pooled cover {st['cover'][-1]:.2f}")
    # 1b random answers through answer_stats: the chance levels
    rng = np.random.default_rng(5)
    rl = rng.standard_normal((len(P["r"]), T, 2, R))
    st = answer_stats(rl, P["r"], P["env"])
    print(f"random answers through answer_stats: D {st['distinct'].mean():.1f} (chance {C['D']:.1f}), e^H "
          f"{st['eff'].mean():.1f} ({C['eH']:.1f}), categories {np.round(st['cat'].mean(0), 4)} "
          f"({np.round(C['cat'], 4)}), distant pairs merged {st['coll_env'].mean():.4f} ({C['coll']:.4f})")
    # 2 distinct answers per environment by np.unique, one checkpoint and environment
    z = np.load(os.path.join(AN, "trained", "ck_01100.npz"))
    k = list(D["trained"]["episode"]).index(1100)
    a = z["logq"][:, -1].argmax(-1)
    for e in (0, 17, 49):
        rows = P["env"] == e
        u = len(np.unique(a[rows, 0] * R + a[rows, 1]))
        print(f"env {e}: distinct answers at T by np.unique {u}, cached {D['trained']['distinct'][k, -1, e]}")
    # 2b distant pairs merged by an explicit double loop, environment 17
    rows = np.flatnonzero(P["env"] == 17)
    hit = tot = 0
    for ii, i in enumerate(rows):
        for j in rows[ii + 1:]:
            if np.abs(P["r"][i] - P["r"][j]).max() >= 2:
                tot += 1
                hit += int((a[i] == a[j]).all())
    print(f"env 17: distant pairs merged by explicit loop {hit}/{tot} = {hit / tot:.5f}, cached "
          f"{D['trained']['coll_env'][k, -1, 17]:.5f}")
    # 3 categories add up to 1; joint accuracy = the 'right' category; far = one far + both far
    for run in RUNS:
        c = D[run]["cat"]
        print(f"{run}: categories sum to 1 (max |dev| {np.abs(c.sum(-1) - 1).max():.1e}); right category vs both "
              f"variables right (stored float64 argmax): max |diff| "
              f"{np.abs(c[..., 0] - D[run]['acc_joint']).max():.4f}; far errors = one far + both far: max |diff| "
              f"{np.abs(D[run]['far_env'].mean(-1) - c[..., 2] - c[..., 3]).max():.1e}; env-mean of both right = "
              f"pooled: max |diff| {np.abs(D[run]['right_env'].mean(-1) - D[run]['acc_joint']).max():.1e}")
    # 4 path length and visited cells of one probe by an explicit loop
    lq = z["logq"].astype(np.float64)
    i = 1234
    p = np.exp(lq[i] - logsumexp(lq[i], -1, keepdims=True))
    m = (p * V).sum(-1)
    path = sum(np.hypot(*(m[t] - m[t - 1])) for t in range(1, T))
    cells = {(int(np.floor(m[t, 0] + .5)), int(np.floor(m[t, 1] + .5))) for t in range(T)}
    paths, visits = [], []
    for j in np.flatnonzero(P["env"] == P["env"][i]):
        pj = np.exp(lq[j] - logsumexp(lq[j], -1, keepdims=True))
        mj = (pj * V).sum(-1)
        paths.append(sum(np.hypot(*(mj[t] - mj[t - 1])) for t in range(1, T)))
        visits.append(len({(int(np.floor(mj[t, 0] + .5)), int(np.floor(mj[t, 1] + .5))) for t in range(T)}))
    print(f"probe {i}: path {path:.4f}, visited {len(cells)}; its environment's mean path / visited: cache "
          f"{D['trained']['path_env'][k, P['env'][i]]:.4f} / {D['trained']['visited_env'][k, P['env'][i]]:.4f}, "
          f"loop {np.mean(paths):.4f} / {np.mean(visits):.4f}")
    # 5 references: stored correctness at T agrees with the cached argmax
    for kk in REFS:
        print(f"{kk}: argmax at T agrees with the probe set's stored correctness in {ref['check_' + kk].mean():.4f}")
    # 6 the pooled spread decomposes: mean squared distance between two probes' means = 2 x spread
    mm = (np.exp(lq - logsumexp(lq, -1, keepdims=True)) * V).sum(-1)[:, -1]
    j1, j2 = rng.integers(len(mm), size=(2, 200000))
    print(f"spread at T: cache {D['trained']['spread'][k, -1]:.3f}, half the mean squared distance of random probe "
          f"pairs {(((mm[j1] - mm[j2]) ** 2).sum(-1) / 2).mean():.3f}")
    # 7 modal maps: Bayes's modal answer equals the truth in most cells; recount one environment by hand
    E = np.load(os.path.join(GRIDS, "fwd_episodes.npz"))
    M = build_fwd()
    e = 18
    rows = E["env"] == e
    ac = E["a_bayes"][rows, 0].astype(int) * R + E["a_bayes"][rows, 1].astype(int)
    modal = [np.bincount(ac[E["cell"][rows] == c], minlength=R * R).argmax() for c in range(R * R)]
    print(f"env {e}: Bayes's modal answers recounted {len(set(modal))} distinct vs cached "
          f"{M['bayes__distinct_modal'][e]}; modal = truth in {np.mean(np.array(modal) == np.arange(R * R)):.2f}")
    # 8 crossing helper on a known curve, and the bootstrap's identity resample
    xx = np.array([10., 100., 1000., 10000.])
    print(f"stay_cross on y = 0, 1, 3, 4 past 2: {stay_cross(xx, np.array([0., 1., 3., 4.]), 2., False):.2f} (exact "
          f"{np.exp(np.log(100) + .5 * np.log(10)):.2f}); y = 0, 3, 1, 4 past 2: "
          f"{stay_cross(xx, np.array([0., 3., 1., 4.]), 2., False):.2f}; never: "
          f"{stay_cross(xx, np.array([0., 3., 4., 1.]), 2., False)}")
    Y = env_curve(D["trained"], "far")
    Wid = np.full((1, NE), 1. / NE)
    print(f"bootstrap with equal weights reproduces the curve: max |diff| {np.abs(Wid @ Y.T - Y.mean(-1)).max():.1e}")
    # 9 the noise-test helpers (t_stats at T, lost_later) reproduce the cached reference values
    for kk, lq in ref_beliefs(P).items():
        a = lq.argmax(-1)
        s = t_stats(a[:, -1], P["r"], P["env"])
        Rk = _ref_view(ref, kk)
        dev = max(abs(s["far"] - Rk["far_env"][-1].mean()), abs(s["coll"] - Rk["coll_env"][-1].mean()),
                  abs(s["D"] - Rk["distinct"][-1].mean()), abs(s["eH"] - Rk["eff"][-1].mean()),
                  abs(s["both"] - Rk["right_env"][-1].mean()), abs(lost_later(a, P["env"]) - series(Rk, "merged")),
                  abs(back_forth(lq, P["env"]) - series(Rk, "back")))
        print(f"{kk}: noise-test helpers vs cache (far errors, merged, D, e^H, both right, cells lost later, "
              f"back-and-forth): max |diff| {dev:.1e}")
    # 10 the passing-naive landmark re-timed on the shared resamples (td_tt_gap.boot_counts)
    L = landmark()
    lo, hi, p = L["interval"]
    print(f"passing-naive landmark re-timed: goal accuracy at T (stored = recomputed within {L['max_dif']:.0e}) above "
          f"naive's ({L['naive']:.4f}) for good from {L['cross']:,.0f} [{lo:,.0f}-{hi:,.0f}] (crossing in {p:.0%} of "
          f"resamples); first checkpoint above it over resamples {L['first'][0]:,.0f} (5%) - {L['first'][1]:,.0f} "
          f"(95%)")


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

COL = {"trained": "#1f77b4", "echo": "#ff7f0e"}
NAME = {"trained": "Trained RNN", "echo": "ESN"}
RSTY = {"bayes": dict(color="k", ls="--", lw=1.3), "memfree": dict(color="k", ls="-.", lw=1.2),
        "naive": dict(color="0.6", ls="--", lw=1.4)}                   # grey for naive only; Bayes variants black
CSTY = dict(color="#9467bd", ls=(0, (1, 1.4)), lw=1.8)                     # chance: random answers / random map
RNAME = {"bayes": "Bayes", "memfree": "memory-free Bayes", "naive": "naive"}
XMIN, XMAX = 8, 7e4
TS, FL, FT = 12.5, 10.5, 10                                  # title, axis-label, tick / legend sizes


def xaxis(ax, label=True):
    ax.set_xscale("log")
    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks([X0, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"], fontsize=FT)
    ax.minorticks_off()
    ax.tick_params(labelsize=FT)
    if label:
        ax.set_xlabel("training episodes", fontsize=FL)
    ax.spines[["top", "right"]].set_visible(False)


def title(ax, name, formula, letter):
    ax.set_title(f"{letter}   {name}\n{formula}", loc="left", fontsize=TS, linespacing=1.5)


def noise_can_cross(chance, level, below):
    """True when random answers lie past `level` in the crossing's direction, so noise alone can carry a curve across
    it (drawn as an open dot)"""
    if chance is None:
        return False
    return chance < level if below else chance > level


# crossings that uniform random answers cannot produce but local answer jitter can (own_noise_tests in numbers):
# distant pairs merged below chance (1/100); drawn half-filled, they count only together with falling far errors
JITTER_CROSSES = {("coll", "chance")}


def mark(ax, c, level, color, hollow=False, half=False):
    """a crossing (training episode, 5%, 95%, share never) drawn at its level: dot + 5-95% interval bar; hollow: noise
    alone can cross this level (noise_can_cross); half (top half filled): uniform random answers cannot cross it, but
    local jitter of the answers can (JITTER_CROSSES)"""
    t, lo, hi, _ = c
    if np.isnan(t) or not np.isfinite(hi):
        return
    if half:
        ax.errorbar([t], [level], xerr=[[t - lo], [hi - t]], fmt="o", ms=8.5, color=color, mfc=color, mfcalt="w",
                    fillstyle="top", mec=color, mew=1.8, elinewidth=2.2, capsize=3.5, capthick=1.6, zorder=6)
        return
    ax.errorbar([t], [level], xerr=[[t - lo], [hi - t]], fmt="o", ms=7 if hollow else 6, color=color,
                mfc="w" if hollow else color, mec=color if hollow else "k", mew=2 if hollow else .7, elinewidth=2.2,
                capsize=3.5, capthick=1.6, zorder=6)


def _legend(fig, y, extra=()):
    from matplotlib.lines import Line2D
    xc = passes_naive()[0]
    h = [Line2D([], [], color=COL[r], lw=2, label=NAME[r]) for r in RUNS]
    h += [Line2D([], [], **RSTY[k], label=RNAME[k]) for k in REFS]
    h += [Line2D([], [], **CSTY, label="chance: " + " / ".join(("random answers",) + tuple(extra)))]
    h += [Line2D([], [], color="0.55", lw=1, ls=":", label=f"Trained RNN passes naive's goal accuracy (drawn at "
                                                           f"{xc:.0f})")]
    h += [Line2D([], [], color="0.3", marker="o", mec="k", ms=6, lw=2.2,
                 label="passes the level for good (5-95% over environments); uniform random answers cannot cross this "
                       "level")]
    h += [Line2D([], [], color="0.3", marker="o", mfc="w", mec="0.3", mew=2, ms=7, lw=2.2,
                 label="the same, but random answers lie past this level: noise alone can cross it")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(.5, y), ncol=3, frameon=False, fontsize=FL + .5,
               columnspacing=2.2)


def figure_curves():
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    ref = build_refs()
    D = {run: build_run(run) for run in RUNS}
    R_ = {k: _ref_view(ref, k) for k in REFS}
    C = chance_levels()
    W = boot_weights()
    CR = all_crossings(D, R_, C, W)
    xc = passes_naive(D)[0]
    fig = plt.figure(figsize=(23, 11.4))
    gs = fig.add_gridspec(2, 5, left=.045, right=.975, top=.885, bottom=.205, wspace=.36, hspace=.6)
    chance = {"acc": C["both"], "far": C["far"], "coll": C["coll"], "distinct": C["D"], "eff": C["eH"],
              "merged": C["lost"]}
    below_of = dict(CROSS_KEYS)
    marks = {"acc": ("naive", "memfree"), "far": ("naive", "memfree"), "coll": ("naive", "chance"),
             "distinct": ("naive", "memfree"), "eff": ("naive", "memfree"), "merged": ("naive",)}
    panels = [
        ("acc", (0, 0), "a", "Both variables right at T", r"$P(\hat r_{0} = r_{0},\ \hat r_{1} = r_{1})$",
         "share of probes"),
        ("far", (0, 1), "b", "Far errors at T (2+ cells off)", r"$P(\mathrm{max}_v\, |\hat r_v - r_v| \geq 2)$",
         "share of probes"),
        ("coll", (0, 2), "c", "Distant pairs merged at T",
         r"$P(\hat c_i = \hat c_j\ |\ \mathrm{max}_v\, |r_{v,i} - r_{v,j}| \geq 2)$, same env.",
         "share of probe pairs"),
        ("distinct", (0, 3), "d", "Answer cells used at T, per environment",
         r"$D_{e,T}$: distinct $\hat c$ at T among its 200 probes", "cells (of 100)"),
        ("eff", (0, 4), "e", "Effective number of answer cells at T",
         r"$e^{H},\ H = -\Sigma_k\, p_k \log p_k$, its 200 answers at T", "cells (of 100)"),
        ("merged", (1, 2), "h", "Answer cells lost later in the episode",
         r"per environment: $\mathrm{max}_t\, D_{e,t} - D_{e,30}$, then averaged", "cells"),
        ("path", (1, 3), "i", "Path of the posterior mean, per episode",
         r"$\Sigma_t\, \|m_t - m_{t-1}\|,\ m_t = (\mathrm{E}_q r_0, \mathrm{E}_q r_1)$", "cells"),
        ("back", (1, 4), "j", "Back-and-forth of the posterior mean",
         r"$\Sigma_t\, \|m_t - m_{t-1}\| - \|m_{30} - m_1\|$ (path $-$ net move)", "cells")]
    for key, (i, j), letter, name, formula, ylab in panels:
        ax = fig.add_subplot(gs[i, j])
        for run in RUNS:
            ax.plot(D[run]["x"], series(D[run], key), color=COL[run], lw=2)
            if key == "path":
                ax.plot(D[run]["x"], series(D[run], "drift"), color=COL[run], lw=1.2, ls=(0, (1.2, 1.6)))
        for k in REFS:
            ax.axhline(float(series(R_[k], key)), **RSTY[k])
        if key in chance:
            ax.axhline(chance[key], **CSTY)
        ax.axvline(xc, color="0.55", lw=1, ls=":")
        for run in RUNS:
            for k in marks.get(key, ()):
                lev = chance[key] if k == "chance" else float(series(R_[k], key))
                mark(ax, CR[run, key, k], lev, COL[run], hollow=noise_can_cross(chance.get(key), lev, below_of[key]),
                     half=(key, k) in JITTER_CROSSES)
        if key == "coll":
            ax.set_yscale("log")
            ax.set_ylim(8e-4, 1.6)
            ax.text(.97, .97, "chance: 1/100.\nHalf dot: uniform\nnoise cannot cross\n1/100, but jitter to\nnearby "
                    "cells can; it\nraises far errors,\nwhich instead keep\nfalling (b)", transform=ax.transAxes,
                    ha="right", va="top", fontsize=FT, color="#6a3d9a")
        if key == "distinct":
            ax.text(.97, .04, f"random answers: {chance[key]:.1f};\nnoise raises D, so its\ncrossings can come\nfrom "
                    "noise (open dots)", transform=ax.transAxes, ha="right", va="bottom", fontsize=FT,
                    color="#6a3d9a")
        if key == "merged":
            ax.set_ylim(0, 25)
            ax.text(.97, .97, f"random answers\n(every step): {chance[key]:.2f},\nat naive's level\n(open dot)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=FT, color="#6a3d9a")
        if key == "acc":
            ax.set_ylim(-.015, .65)
        if key == "far":
            ax.set_ylim(-.02, 1.0)
        xaxis(ax, label=i == 1)
        ax.set_ylabel(ylab, fontsize=FL)
        title(ax, name, formula, letter)
        if key == "path":
            ax.text(.97, .1, "dotted: path of the probe-\naveraged trajectory (drift\nshared by all episodes)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=FT, color="0.3")
    # f, g: answer cells used at every step (heat maps) with the references' per-step profiles at the right
    norm = Normalize(0, 95)
    for c, run in enumerate(RUNS):
        sub = gs[1, c].subgridspec(1, 3, width_ratios=[7.2, 1.1, .38], wspace=.08)
        ax, axr = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        d = D[run]
        x = d["x"]
        lx = np.log(x)
        ed = np.exp(np.r_[lx[0] - (lx[1] - lx[0]) / 2, (lx[1:] + lx[:-1]) / 2, lx[-1] + (lx[-1] - lx[-2]) / 2])
        prof = d["distinct"].mean(-1)                                                # (K, T) environment mean
        im = ax.pcolormesh(ed, np.arange(.5, T + 1), prof.T, cmap="viridis", norm=norm, shading="flat",
                           rasterized=True)
        ax.plot(x, series(d, "peak"), color="w", lw=1.8)
        ax.axvline(xc, color="w", lw=1, ls=":")
        xaxis(ax)
        ax.set_xlim(ed[0], ed[-1])
        ax.set_ylim(.5, T + .5)
        ax.set_yticks([1, 10, 20, 30])
        ax.set_ylabel("step t", fontsize=FL)
        title(ax, f"{NAME[run]}: answer cells, each step",
              r"$D_{e,t}$ averaged over env.; white: its peak step", "fg"[c])
        prof_r = np.stack([R_[k]["distinct"].mean(-1) for k in REFS], 1)            # (T, 3)
        axr.pcolormesh(np.arange(4), np.arange(.5, T + 1), prof_r, cmap="viridis", norm=norm, shading="flat")
        for q, k in enumerate(REFS):
            axr.plot(q + .5, prof_r[:, q].argmax() + 1, "o", color="w", ms=4.5, mec="k", mew=.6)
        axr.set_xticks(np.arange(3) + .5)
        axr.set_xticklabels(["Bayes", "mem.-free", "naive"], rotation=90, fontsize=FT)
        axr.set_yticks([])
        axr.set_ylim(.5, T + .5)
        if c == 1:
            cb = fig.colorbar(im, cax=fig.add_subplot(sub[2]))
            cb.set_label("answer cells used", fontsize=FT)
            cb.ax.tick_params(labelsize=FT - 1)
    _legend(fig, .066)
    fig.text(.5, .008, "10,000 probe episodes (50 environments x 100 latent pairs x 2 trajectories) x 30 steps at each "
             "of 114 checkpoints per network (seed 20261029). Answers: each variable's argmax. a-h: per environment "
             "(its 200 probes; c: its 18,432 probe pairs with latent pairs 2+ apart), then averaged over the 50 "
             "environments;\ni, j: per episode, then averaged. Bayes = exact marginals; memory-free Bayes = each "
             "step's own exact marginal likelihood, summed over steps; naive = factorized. Chance: every answer drawn "
             "uniformly from the 100 cells (h: at every step). Dots with bars: the training episode after which the "
             "curve\nstays past the level (log-interpolated between checkpoints), with its 5-95% interval over 1,000 "
             "resamples of the 50 environments (references resampled with them).", ha="center", va="bottom",
             fontsize=FT, color="0.3", linespacing=1.5)
    fig.suptitle("The latent grid through training: the Trained RNN makes fewer far errors than naive from about 580 "
                 "training episodes (b) and merges distant pairs less often than chance from about 870 (c); the ESN "
                 "ends at naive's level", x=.045, ha="left", fontsize=TS + 2.5, y=.985)
    path = fig_path("td_tt_latent.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


def _arrow_map(ax, modal, col, sink_col):
    """one environment's map from each true pair (grey dot) to its modal answer (arrow; circle area = pairs answered
    there)"""
    g = np.arange(R * R)
    r0, r1 = g // R, g % R
    a0, a1 = modal // R, modal % R
    ax.scatter(r0, r1, s=7, color="0.75", zorder=1, linewidths=0)
    mv = modal != g
    ax.quiver(r0[mv], r1[mv], (a0 - r0)[mv], (a1 - r1)[mv], angles="xy", scale_units="xy", scale=1, width=.007,
              headwidth=3.5, headlength=4, color=col, alpha=.55, zorder=2)
    size = np.bincount(modal, minlength=R * R)
    u = size > 0
    ax.scatter(g[u] // R, g[u] % R, s=14 * size[u], color=sink_col, alpha=.8, zorder=3, linewidths=.5, edgecolors="k")
    ax.set_xlim(-.8, 9.8)
    ax.set_ylim(-.8, 9.8)
    ax.set_aspect("equal")
    ax.set_xticks([0, 9])
    ax.set_yticks([0, 9])
    ax.tick_params(labelsize=FT)
    for s in ax.spines.values():
        s.set_color("0.7")


ARROW_X = (10, 240, 880, 2400, 39996)                     # training episodes (summary_data/grids checkpoints)


def figure_fold():
    import matplotlib.pyplot as plt
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    ref, M, S = build_refs(), build_fwd(), build_same_as_naive()
    D = {run: build_run(run) for run in RUNS}
    R_ = {k: _ref_view(ref, k) for k in REFS}
    C = chance_levels()
    W = boot_weights()
    xc = passes_naive(D)[0]
    nd = M["naive__distinct_modal"]
    env = int(np.argsort(nd, kind="stable")[len(nd) // 2])                       # naive's median environment
    pairs = np.load(os.path.join(AN, "probes.npz"))["pairs"][env]
    fig = plt.figure(figsize=(23, 14.8))
    gs = fig.add_gridspec(3, 6, left=.05, right=.985, top=.905, bottom=.155, wspace=.3, hspace=.42,
                          height_ratios=[1.12, 1.12, 1])
    for i, run in enumerate(RUNS):
        xs = M[f"{run}__x"]
        for j, xx in enumerate(ARROW_X):
            k = int(np.argmin(np.abs(xs - xx)))
            ax = fig.add_subplot(gs[i, j])
            _arrow_map(ax, M[f"{run}__modal"][k, env].astype(int), COL[run], COL[run])
            n_here = int(M[f"{run}__distinct_modal"][k, env])
            n_all = M[f"{run}__distinct_modal"][k].mean()
            lab = "0" if xs[k] <= X0 else f"{xs[k]:,.0f}"
            ax.set_title(f"{lab} training episodes\n{n_here} answers (all 50 env.: {n_all:.0f})", fontsize=FL + 1,
                         loc="left", linespacing=1.4)
            if j == 0:
                ax.set_ylabel(f"{NAME[run]}\n" + r"$r_1$", fontsize=TS + 1, color=COL[run])
            if i == 1:
                ax.set_xlabel(r"$r_0$", fontsize=FL + 1)
        kref = ("bayes", "naive")[i]                     # Bayes beside the Trained RNN, naive beside the ESN
        ax = fig.add_subplot(gs[i, 5])
        _arrow_map(ax, M[f"{kref}__modal"][env].astype(int), "0.45" if kref == "naive" else "k",
                   "0.6" if kref == "naive" else "0.15")
        ax.set_title(f"{RNAME[kref]} (same trajectories)\n{int(M[f'{kref}__distinct_modal'][env])} answers "
                     f"(all 50 env.: {M[f'{kref}__distinct_modal'].mean():.0f})", fontsize=FL + 1, loc="left",
                     linespacing=1.4)
        if i == 1:
            ax.set_xlabel(r"$r_0$", fontsize=FL + 1)
    fig.text(.05, .935, f"a   One environment (variables {pairs[0]} and {pairs[1]}; naive's median number of answers): "
             "each latent pair's answer at T. Grey dot: the true pair; arrow: to the answer given most often over its "
             "24 trajectories; circle area: pairs answered there.", fontsize=TS, ha="left")
    # curves
    cats = (("right", "Answer = the true pair", r"$\hat c = c$"),
            ("neighbour", "Answer = a neighbour", r"$\mathrm{max}_v\, |\hat r_v - r_v| = 1$"),
            ("one far", "One variable far, the other near",
             r"$\mathrm{max}_v\, |\hat r_v - r_v| \geq 2,\ \mathrm{min}_v\, |\hat r_v - r_v| \leq 1$"),
            ("both far", "Both variables far", r"$\mathrm{min}_v\, |\hat r_v - r_v| \geq 2$"))
    for c, (key, name, formula) in enumerate(cats):
        ax = fig.add_subplot(gs[2, c])
        for run in RUNS:
            ax.plot(D[run]["x"], D[run]["cat"][:, -1, c], color=COL[run], lw=2)
        for k in REFS:
            ax.axhline(R_[k]["cat"][-1, c], **RSTY[k])
        ax.axhline(C["cat"][c], **CSTY)
        ax.axvline(xc, color="0.55", lw=1, ls=":")
        xaxis(ax)
        ax.set_ylim(0, .62)
        ax.set_ylabel("share of probes at T", fontsize=FL)
        title(ax, name, formula, "bcde"[c])
    ax = fig.add_subplot(gs[2, 4])
    for run in RUNS:
        ax.plot(M[f"{run}__x"], M[f"{run}__distinct_modal"].mean(-1), color=COL[run], lw=2, marker="o", ms=3.5)
    for k in ("bayes", "naive"):
        ax.axhline(M[f"{k}__distinct_modal"].mean(), **RSTY[k])
    ax.axhline(C["modal"], **CSTY)
    ax.axvline(xc, color="0.55", lw=1, ls=":")
    mx = M["trained__x"]
    Y = M["trained__distinct_modal"].astype(np.float64)
    lev = M["naive__distinct_modal"].mean()
    mark(ax, cross(mx, Y, M["naive__distinct_modal"].astype(np.float64), False, W), lev, COL["trained"],
         hollow=noise_can_cross(C["modal"], lev, False))
    mark(ax, cross(mx, Y, C["modal"], False, W), C["modal"], COL["trained"])
    xaxis(ax)
    ax.set_ylabel("answers (of 100 pairs)", fontsize=FL)
    title(ax, "Different answers per environment", r"distinct modal $\hat c$ over the 100 pairs (as a)", "f")
    ax = fig.add_subplot(gs[2, 5])
    for run in RUNS:
        ax.plot(S[f"{run}__x"], S[f"{run}__same"], color=COL[run], lw=2, marker="o", ms=3.5)
        ax.plot(S[f"{run}__x"], S[f"{run}__control"], color=COL[run], lw=1.3, ls=":")
    ax.axvline(xc, color="0.55", lw=1, ls=":")
    xaxis(ax)
    ax.set_ylim(0, None)
    ax.set_ylabel("share of pairs", fontsize=FL)
    title(ax, "Same wrong answer as naive", r"$P(\hat c = \hat c^{\,naive} \mid$ naive wrong, Bayes right$)$", "g")
    ax.text(.03, .8, "dotted: naive's answer\nfor the same pair in\nanother environment", transform=ax.transAxes,
            ha="left", va="top", fontsize=FT, color="0.3")
    _legend(fig, .07, extra=("random map (f)",))
    fig.text(.5, .006, f"a, f, g: 50 environments x 100 latent pairs x 24 fresh trajectories at 19 checkpoints per "
             f"network (the belief-grid replays); the answer of a pair = its most frequent answer at T. g: the "
             f"{S['n_cells']:,} (environment, pair) cells where naive's answer is wrong and Bayes's right.\nb-e: the "
             f"10,000 probe episodes (50 environments x 100 pairs x 2 trajectories) at each of 114 checkpoints, each "
             f"episode's own answer at T; b-e add up to 1. Chance: answers drawn uniformly from the 100 cells (b-e: "
             f"[{C['cat'][0]:.3f} {C['cat'][1]:.3f} {C['cat'][2]:.3f} {C['cat'][3]:.3f}]; f: a random map,\n"
             f"{C['modal']:.1f} distinct answers, the tie-free level). Dots with bars in f: the training episode after "
             f"which the curve stays past the level (log-interpolated), 5-95% over 1,000 resamples of the 50 "
             f"environments.",
             ha="center", va="bottom", fontsize=FT, color="0.3", linespacing=1.5)
    fig.suptitle("The folding of the latent grid as it unfolds: where each latent pair's answer lands", x=.05,
                 ha="left", fontsize=TS + 2.5, y=.99)
    path = fig_path("td_tt_latent_fold.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path, "; environment", env, "pair", pairs)
    return path


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "data"
    if what == "data":
        build_refs()
        for run in RUNS:
            build_run(run)
        build_fwd()
        build_same_as_naive()
        build_timeline()
    elif what == "numbers":
        numbers()
    elif what == "checks":
        checks()
    elif what == "timeline":
        build_timeline()
    elif what == "synthesis":
        synthesis()
    elif what == "figure":
        figure_curves()
        figure_fold()
