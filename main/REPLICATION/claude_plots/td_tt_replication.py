"""Does the through-training order replicate in the other trained runs of the same world, and do the neural
candidates keep their place relative to each run's own behaviour? (John 11-02: "Can we identify the neural changes that
precede [the behavioural developments]? How does the network get to the correct answer before learning optimal bayes
dynamics? ... link the neural properties to the clear behavioral developments through training".) Topic: replication.
Questions: (1) does each trained run show 'answers and splitting, then holding and coupling' on its own clock? (2) do
the neural candidates keep their position relative to each run's behavioural crossings ("width sets the pace, not the
order")? (3) natural experiment: N = 100 and N = 50 never form long-tau units or credit cancellation: do they hold
splits or couple?
-> main/REPLICATION/figure_bin/td_tt_replication.png   figure()
   numbers()  every number below;  check()  the verification lines;  synthesis()  the export
   caches main/DATA/experimental/training_dynamics/through_training/replication/: prims_<run>.npz (per-checkpoint
   primitives, built with the topic modules' own functions), trunc_<run>.npz (memory cut, both variables),
   synthesis.npz (per run and row: the curve, its 1,000 resampled curves (float64), naive / Bayes levels and their
   resamples, every event with its resampled crossings and a censored flag)

RUNS (seed 20261029: one world, the same probes and episode stream; the N = 1000 runs share their initial weights; 114
checkpoints each; x = training episodes E - E // 5, the initial network drawn at 10): trained (Trained RNN, N = 1000,
loss power .5), trained_p1 (N = 1000, loss power 1), trained_N200 / N100 / N50 (power .5), echo (ESN, the null).
Keys: logq, right, trunc_right, effdim_u, norm_u, tau, rho_hh in every run; the credit keys cr_* in trained, echo, N200,
N100, N50 only. CANNOT BE COMPUTED here: for trained_p1 the credit rows (coherence, its entanglement lock, cross-band
share, dW-dimension lock; td_analysis's StepLoss hard-codes power .5); for the four runs the forward-set rows of
td_tt_wormholes (held shares on 5,028 replayed grid episodes, distance / angle / same diagonal out of corner splits,
corner answer = truth: they need CPU replays), the learning rows of td_tt_entangled_learning (per-item losses at every
checkpoint), td_tt_neural's replay rows (per-environment tau, exact PR resampling: trained / echo only) and anything
from snapshots (none for these runs). The split gradient is not slow (td_tt_splitting.split_stats ~1 s per
checkpoint): computed. The probe-based held shares (720 / 365 var-steps) stand in for holding, thinly.

DEFINITIONS
  Behaviour rows (the topics' own functions; td_tt_gap conventions): goal accuracy at T (td_tt_jumps); far errors at T,
  distant pairs merged at T (td_tt_latent.answer_stats); split gradient = most minus least entangled tenth of the
  one-step correct split of near pairs (td_tt_splitting.split_stats); opposite corners, fold index at T (extra early
  rows, not tested); jump excess angle over the matched independence null, long-path coupling G, angle of jumps >= 4
  cells (td_tt_jumps.stats / extra_stats); end-mass slope and direction slope within steps (td_tt_wormholes maps,
  probes); held per variable at corner-split ends / one-variable end splits on the probes (extra late rows, not tested).
  Gap fraction f = (y - y_naive) / (y_Bayes - y_naive), naive's and Bayes's values on the same probes. c_L = the training
  episode after which f stays above L to the end (log-interpolated). [5-95%]: td_tt_gap.boot_counts() (1,000 x 50
  environments, seed 20261103), ONE matrix for every row of every run: every run is measured on the same 50
  environments, so crossings are paired across rows and across runs.
  Families: early = answers (goal accuracy, far errors, merged pairs) + splitting (split gradient); late = holding
  (end-mass slope) + coupling (excess angle, G, >= 4-cell angle, direction slope). Family test: share of resamples in
  which every early c_L precedes every late c_L (never = after the end). Censored: a row without a value at the first
  checkpoints (< 50 jumps of >= 4 cells) that is already past L at its first value has no rise to time: its crossing
  is an upper bound (the rise happened before the row became defined); it is left out of that run's test, and the test
  with it included is reported as a check (trained_p1's >= 4-cell angle: < 50 large jumps until 2,320, f .69 at its
  first value). Matched accuracy for it: matched_at() (its value at a checkpoint against the Trained RNN's where the
  Trained RNN's accuracy passes the same level for good, per resample).
  Own clock: a crossing / the same run's passing naive (c0 of goal accuracy), per resample. Accuracy at an event: the
  run's goal-accuracy f at that crossing, per resample. Matched accuracy: a row's value when the run's accuracy passes
  f = L for good, and its paired difference from the Trained RNN's at the Trained RNN's accuracy c_L.
  Neural rows (ck keys and statistics as td_tt_neural reads them; statistic per probe or environment, then averaged):
    memory benefit (cut to 1 step) = goal accuracy at T intact minus with the memory cut to the last step (ck
      trunc_right, an6), per environment. Natural references: naive 0 (its increment depends on the current observation
      only), Bayes = right_bayes - right_window1 = .163 (resampled). Events: half of Bayes's level; Bayes's level.
      Both-variable version (td_tt_neural 'ben') as a check.
    memory benefit (cut to 5 steps): Bayes's own .031. A rise and fall: timed on its rise (half its peak, first reached)
      and its relaxation (half-way back from the peak to the end value, after which it stays below); timed only if the
      peak exceeds Bayes's own in 95% of resamples.
    effective dimension of u: ck effdim_u averaged over steps 2-30 per probe. No natural reference: 10% / 50% of its own
      change (initial network -> mean of the last 3 checkpoints, both per resample).
    r(X, entanglement): the locked statistic (within-episode Pearson r over steps 17-30 with rev, the mean of the two
      variables' entanglement; = td_tt_credit_lags r(0) and the plot_training_dynamics series at tau 0) for X = |u| (all
      probes) and credit coherence / dW dimension (5,000 credit probes, ratio of environment sums). Reference 0. Events:
      r > .1; half its end value (only when the end value is > .1); for dW dimension (+.37 at init) half-way from its
      initial value to 0, and r < 0.
    credit coherence: per environment the median of cr_coh over its probe-steps (error steps 17-30), then averaged.
      Natural reference 1 (lag pieces neither add nor cancel): below 1 for good. Cross-band share on the final bands
      (cr_bandx_fixed, 2,000 probes): below -.1 (the ESN's 0-crossings sit at -.003).
    long-memory units: share of units with tau_j > 5, tau_j = 1 / (1 - g_j), g_j = unit j's forget gate averaged over
      all probes and steps 2-30 (ck key tau, td_analysis forward()). FLAGGED DEVIATIONS: (i) a statistic of an average
      (the averaging-order rule wants tau per probe, then averaged; only this pooled tau is stored, and a per-probe tau
      needs replays: not done). 1 / (1 - g) is convex, so each unit's probe-averaged tau is at least its pooled tau:
      the share with probe-averaged tau > 5 is at least the pooled share at every checkpoint, and the pooled
      crossings of a fixed level are late bounds of that version's (at checkpoint resolution). (ii) Pooled: no
      environment interval; its ratio's interval is passing naive's, and its orders compare its point with the other
      event's interval (no interval on the tau side). (iii) Levels: > 3% (30 of 1,000 units) and > 5% are one absolute
      amount for every run, chosen ad hoc (no natural reference); the rule's own-change events (10% / 50% of the change
      from the initial network (0) to the mean of the last 3 checkpoints) are timed too. N100 / N50 have none (their
      share ends at 0). W_hh radius: the smallest of the i, f, o blocks' spectral radii > 1 (weights; pooled the same
      way).

RESULTS ([5-95%]; training episodes; runs in the order N1000, p1, N200, N100, N50; every number printed by numbers())
  Pace. Passing naive (goal accuracy at T above naive's .464 for good): 835 [798-876], 1,256 [1,207-1,356], 2,070
  [1,966-2,424], 6,271 [5,444-7,662], 21,072 [14,871-30,271]; ESN never. End accuracy .761 / .761 / .717 / .631 / .507
  (f .97 / .97 / .82 / .54 / .14); ESN .434 (f -.10). Accuracy half-way: 1,394 / 2,010 / 6,311 / 33,623 (89% of
  resamples cross) / never.
  (1) ORDER ON EACH RUN'S OWN CLOCK. Width sets the pace, not the order: in every power-.5 run that can be tested, the
    answers and splitting are past the level before holding and coupling (interval rule on the composite crossings:
    N1000 and N200 at f = .5, N100 at f = .25). The power-1 loss moves holding and the large-jump angle forward (below).
    Family test at f = .25 / .5 / .75 (paired shares, supplementary):
    N1000 .948 / 1.000 / 1.000: latest early c50 1,394 [1,285-1,489] (x1.67 its passing naive) before the earliest late
      c50 3,855 [3,363-4,408] (x4.62).
    N200 .997 / 1.000 / 1.000: 6,311 [5,511-7,817] (x3.05) before 27,894 [23,114-35,339] (x13.47); at f = .25 3,444
      [3,067-3,923] before 9,990 [7,801-12,444].
    N100: no late row reaches f = .5 (end f .11-.27). At f = .25 (.987): 12,678 [10,264-16,902] before 35,358
      [21,515-never] (93% cross).
    N50: not testable: accuracy never reaches f = .25 (end .14); far errors / merged pairs / split gradient reach .25 at
      9,666 / 9,969 / 1,860 and .5 at 29,691 (88% cross) / 25,037 (93%) / 28,371 (98%); no late row reaches .25 (end f
      .01-.07, G .22).
    p1 (loss power 1): .019 / .918 / .994 (.019 / .701 / .994 with the censored >= 4-cell angle included). Holding moves
      early: the end-mass slope's c10 721 [500-853] is before passing naive 1,256 [1,207-1,356] (x0.57 [0.40-0.67];
      N1000 x1.56 [1.28-1.83]) and its c25 1,005 [873-1,518] cannot be ordered against it (x0.80 [0.69-1.19], paired
      share earlier .80; N1000 x2.31 [2.04-2.93]); its c50 3,060 [2,383-3,822] (x2.44) comes at accuracy f .71 [.61,
      .77] (N1000 .88 [.85, .90]), after goal accuracy's c50 2,010 [1,835-2,120] but not ordered against the split
      gradient's c50 2,121 [1,782-3,019] (= the latest early c50, whose composite interval is [1,925-3,019]; paired
      share .918). The large-jump angle moves early too: p1 has < 50 jumps of >= 4 cells until 2,320 and is already at
      f .69 there (censored), so its c50 is <= 2,320 (x1.85 of passing naive, at accuracy f .59; N1000's c50 x4.62
      [4.06-5.61] at f .87). N1000 passes p1's accuracy at 2,320 (f .59) at 1,611, where its >= 4-cell f is .29; paired
      at matched accuracy (matched_at), p1 minus N1000 is +.36 [+.24, +.48] at 2,400 (defined in 89% of resamples;
      +.40 [+.32, +.51] at 2,320, 53%) and +.13 [+.08, +.29] at accuracy c75. The excess angle is slightly ahead at
      every matched accuracy (+.10 [+.07, +.14] / +.08 [+.04, +.12] / +.07 [+.03, +.09] / +.04 [+.01, +.07] at accuracy
      c10 / c25 / c50 / c75) but half-way late, 8,814 [8,178-9,843] (x7.01, accuracy f .91); G 7,645 [6,452-8,258]
      (x6.08) and the direction slope 12,647 [11,160-13,895] (x10.07) stay late and about level at matched accuracy (G
      -.01 to +.08, every interval includes 0; direction -.00 to +.05, only c75's +.05 [+.01, +.10] excludes 0).
    Coupling c50 / passing naive (uncensored rows): N1000 x4.62 (>= 4 cells), x6.78 (G), x7.49 (excess), x12.02
      (direction); p1 x6.08 / 7.01 / 10.07 (G, excess, direction); N200 x13.47 [11.19-16.24] (>= 4 cells) and x18.72
      [14.78-inf] (G; 57% of resamples cross), excess and direction never: x4.6-18.7, plus p1's censored >= 4-cell
      angle at <= x1.85.
    The late c50s come at accuracy f .87-.94 (N1000: >= 4 cells .87, end-mass .88, G .90, excess .92, direction .94),
      .90-.92 in p1's coupling (the censored >= 4-cell angle .59, holding .71) and .78-.82 in N200 (>= 4 cells .78,
      end-mass .81, G .82). At f = .1 the families interleave in every run (.298 / .000 / .649 / .098 / .012). ESN: no
      behaviour row half-way, accuracy never past naive.
  (2) NEURAL CANDIDATES RELATIVE TO EACH RUN'S BEHAVIOUR.
    Memory carries the early answers. The memory benefit (cut to 1 step) reaches Bayes's own (.163) before passing naive
      in all five runs: 552 [484-603] (x0.66 [0.58-0.73]), 826 [538-943] (x0.66 [0.43-0.74]), 1,102 [971-1,375] (x0.53
      [0.45-0.64]), 3,483 [2,835-4,828] (x0.56 [0.41-0.74]), 9,591 [6,872-14,291] (x0.46 [0.29-0.74]); before by the
      interval rule in all five; accuracy f there -.44 / -.38 / -.39 / -.24 / -.21. Its meaning: at that point the intact
      network scores .328 / .346 / .343 / .390 / .399 and the memory-cut network .165 / .183 / .180 / .227 / .236 (naive
      .464, step-wise Bayes .608): the memoryless part stays far below naive, so the event is close to an accuracy level
      and says the early answers come from the state, not that the memory is Bayes-like. Half of Bayes's level: 305 / 173
      / 385 / 522 / 883, at intact accuracy .236 / .182 / .237 / .239 / .237. Both-variable version: 554 / 825 / 1,089 /
      3,174 / 9,484 (x0.66 / 0.66 / 0.53 / 0.51 / 0.45). It peaks at .465 / .440 / .381 / .316 / .228 and ends at .303 /
      .314 / .345 / .306 / .222; at matched accuracy narrower nets lean slightly less on it (N200 -.04 to -.08, N100 -.08
      to -.10 at accuracy c25-c75, paired).
    Start-state control (the verifier's check, 11-02, not printed by numbers(); completed here for p1): the w = 1 cut
      restarts every step from the learned start state. CPU replays that restart each step instead from (a) another
      probe's state of the same environment and step or (b) the probe-average state of step t-1 (the replay equals the
      stored right / trunc_right in 100% of probes in (b)'s script; (a) runs the same forward pass and gives the same
      intact accuracy) score .13-.20 (a) and .15-.25 (b), against .17-.26 for the stored cut and .33-.48 intact, at two
      checkpoints per run near its memory event and its passing naive (N1000 560 / 800, p1 800 / 1,280, N200 1,120 /
      2,000, N100 3,400 / 6,200, N50 9,600 / 20,800; the verifier's own runs, without p1's (b): .13-.22). Scripts
      /tmp/claude-1002/-home-johns-Desktop-CognitiveGridworld/7af6b091-54ff-45a0-b4dd-ef0a3bc9c778/
      scratchpad/tt_tmp/verify_replication/swapreset.py (a) and meanreset.py (b; logs log_meanreset_narrow.txt,
      log_meanreset_wide.txt there; all (a) and p1's (b): log_swapreset_all.txt, log_meanreset_p1.txt in
      tt_tmp/fix_replication/). So the low cut accuracy is not a start-state artefact: the probe's own state history
      carries the early answers. The event's exact time depends on the cut, though: in both wide runs these cuts
      bracket it only by the two replay checkpoints. N1000 with (b): .14 at 560, below Bayes's .163, and past it at
      800, so (560, 800] = x0.67-0.96 of passing naive, whose upper end lies inside passing naive's [798-876]. p1: at
      800 the stored cut (.157; its stored event is 826), (a) and (b) are all below .163, and all three are past it at
      1,280, so (800, 1,280] = x0.64-1.02. By the interval rule 'before passing naive' is therefore not established with
      these cuts in either wide run (log-interpolated between the two replay checkpoints, no interval: N1000 (b) about
      603 = x0.72; p1 stored about 820, (a) about 850 = x0.68, (b) about 994 = x0.79). In the narrow nets the benefit at
      the event checkpoint is .20-.26 with both.
    The transient long-memory reliance (cut to 5 steps; Bayes .031): peak .160 / .147 / .126 / .075 / .055 at 1,760 /
      2,240 / 5,000 / 13,600 / 5,000; half its peak at 841 [798-909] / 1,201 [1,159-1,394] / 1,293 [1,104-1,494] / 2,999
      [1,257-5,158] / 2,503 [633-2,839] (x1.01 / 0.96 / 0.62 / 0.48 / 0.12); relaxed half-way back at 5,988 [5,490-7,250]
      / 9,463 [7,714-9,998] / 18,270 [11,661-21,560] / 16,156 [13,378-39,207] / 5,434 [5,013-39,253] (N50 noisy), at
      accuracy f .91 / .91 / .74 / .33 / -.36; ends .038 / .038 / .059 / .064 / .037. ESN: no transient above Bayes's own.
      Against the late window (from the earliest to the latest late c50, composite): the relaxation is inside it in
      N1000 (after 3,855 [3,363-4,408], before 10,034 [8,646-11,979]) and p1 (after 3,060 [2,383-3,822], before 12,647
      [11,160-13,895]), but before it in N200 (before the earliest late c50 27,894 [23,114-35,339]; paired share
      earlier .990); N100 / N50 have no late c50.
    Effective dimension: 10% of its change before passing naive in every run (x0.60 / 0.67 / 0.29 / 0.22 / 0.06), 50%
      after it at N = 1000 (1,038 [1,018-1,076], x1.24; 1,662 [1,589-1,779], x1.32) but before it in the narrow nets
      (x0.69 / 0.69 / 0.13); accuracy f at the 50% point +.24 / +.35 / -.25 / -.16 / -.50: not tied to the behaviour.
      End 37.7 / 34.9 / 13.0 / 9.6 / 7.4 (initial 5.8, 5.4 at N100, 4.3 at N50); the ESN's too rises (5.8 -> 10.0).
    W_hh radius > 1 (all of i, f, o; pooled: its point against passing naive's interval): 633 / 990 / 1,350 / 757 / 2,376
      (x0.76 / 0.79 / 0.65 / 0.12 / 0.11): always before passing naive, at very different points of it (permissive, not
      timed).
    r(dW dimension, entanglement) half-way from +.37 to 0: 262 [225-282] / - / 97 / 108 / 132 (x0.31 / 0.05 / 0.02 /
      0.01; every run with credit keys, p1 has none): an early change that does not move with width; r < 0 for good
      only at N1000 (861 [808-1,066]).
    Late candidates (after the latest early c50 by the interval rule in N1000, p1 and N200, except the long-memory
    share's 10% own-change point in N1000, its first wave; N100's cross-band share cannot be ordered against N100's
    latest early c50, 33,623 [29,072-never], paired share earlier .58):
      credit coherence < 1: N1000 2,213 [2,145-2,426] (x2.65), N200 21,397 [19,630-22,733] (x10.34), at accuracy f .74
        [.70-.78] and .77 [.73-.80] (difference +.035 [-.007, +.053]: matched within its interval), before the earliest
        late c50 in both (interval rule); never in N100 / N50 (minimum 1.47 / 1.55). At matched accuracy coherence barely
        depends on width: 1.43 (N1000), 1.49 (N200, +.07 [+.00, +.12]), 1.51 (N100, +.09 [+.02, +.17]) at accuracy c50;
        .99 vs 1.02 (+.03 [-.07, +.10]) at c75; 1.72 / 1.84 / 1.90 at c25.
      units with tau > 5 above 3% (pooled: no interval on this side): 2,498 / 6,333 / 36,800 (x2.99 / 5.04 / 17.78) at
        accuracy f .78 / .87 / .81 (from N1000's: p1 +.093 [+.069, +.116], N200 +.029 [+.002, +.055]); before every
        late c50 at N1000; in p1 after the end-mass slope's and the censored >= 4-cell angle's c50s, before G's 7,645
        [6,452-8,258]; in N200 after the >= 4-cell angle's c50 27,894 [23,114-35,413], not ordered against the end-mass
        slope's 34,483 [27,626-38,669] or G's 38,752 [30,814-never] (pooled share, flagged; a probe-averaged tau crosses
        3% no later, so these after orders hold for the pooled share only). > 5%: 5,000 / 8,258 / never. Own change (the
        rule's levels; N100 / N50 have none): half-way at 4,866 / 9,600 / 25,331 (x5.83 / 7.64 / 12.24; accuracy f .89 /
        .91 / .79, from N1000's: p1 +.017 [+.001, +.035], N200 -.105 [-.134, -.082]); in N1000 after the >= 4-cell
        angle's c50 and not ordered against the end-mass slope's, in p1 after G's and not ordered against the excess
        angle's, in N200 not ordered against the >= 4-cell angle's and before the other late c50s. Its 10% point marks
        the first wave in N1000 (11) but the second wave in p1 (3,919) and N200 (16,583). N100: no unit with tau > 5 at
        any checkpoint; N50: one unit (2%) at 200-1,120 only (20 checkpoints). Below the second wave N1000 keeps 2% (20
        units, from the first wave at 20-160 episodes) at accuracy c25-c50, p1 0.4-0.8%, the narrow nets 0.
      r(|u|, entanglement) > .1: 3,693 [3,336-4,038] / 5,651 [5,321-6,022] / 22,197 [19,639-26,441] at accuracy f .85 /
        .83 / .77 (p1 -.024 [-.040, +.016], N200 -.083 [-.123, -.057]); never in N100 (end .055) / N50 (-.004).
      r(coherence, entanglement) > .1: 4,423 [3,947-4,733] (f .88), N200 29,802 [27,391-34,050] (f .79, -.085 [-.122,
        -.062]); half its end value 6,885 [6,347-7,168] / 20,775 [19,158-22,923]; never in N100 / N50 (end .018 / .044).
      cross-band share (final bands) < -.1: 3,827 [3,583-4,055] (f .86), N200 10,352 [9,479-11,013] (f .64), N100 32,861
        [29,740-34,313] (f .49): NOT accuracy-locked (-.22 [-.26, -.19], -.37 [-.43, -.32]); N100 forms it without
        coherence < 1.
    So, apart from the cross-band share, the late markers (coherence < 1, long-memory units at 3%, the two r > .1 locks)
      come at far more similar accuracies than training times: in N1000, p1 and N200 at accuracy f .74-.88 but at
      x2.65-17.8 of passing naive (coherence x2.65 -> x10.34, long-memory units x2.99 -> x17.78). The cross-band share
      < -.1 is the reverse case: it keeps its own-clock place (x4.59 / 5.00 / 5.24 in N1000 / N200 / N100; paired x1.09
      [0.95-1.13] and x1.14 [0.94-1.29] of N1000's) but not its accuracy (f .86 / .64 / .49). Across widths only
      coherence < 1 is accuracy-matched within its interval (N200 +.035 [-.007, +.053]); between the two N = 1000 runs
      r(|u|, ent) > .1 is too (p1 -.024 [-.040, +.016]). The others differ: long-memory units at 3% (p1 +.093, N200
      +.029 [+.002, +.055]) and at half their own change (N200 -.105), r(|u|, ent) > .1 and r(coherence, ent) > .1 in
      N200 (-.083, -.085), the cross-band share in N200 / N100 (-.22, -.37). The late behaviour also sits at similar
      accuracies (late c50s at f .78-.94; in p1 holding at .71 and the censored >= 4-cell angle at .59). Their timing
      cannot tell whether they drive the late behaviour or follow the same progress.
  (3) NATURAL EXPERIMENT. N100 and N50 form neither long-memory units nor coherence < 1 (nor r(|u|, ent) > .1), and never
    reach the accuracy at which N1000 / N200 form them (best f .55 / .15). They still start holding and coupling: N100's
    c10 G 3,132 [1,556-9,865], >= 4 cells 16,682 [12,424-26,854], end-mass 18,313 [14,100-28,840], excess 18,504
    [10,055-28,497], direction 36,167 (62%); end f .11-.27, held shares .14 / .14. N50 barely (end f .01-.07, G .22,
    held shares 0). At matched accuracy they are about level with N1000 on most rows, not all (paired differences of f;
    the N100 c50 and N50 c10 intervals are over the 89% / 82% of resamples in which those accuracy crossings exist): N100
    at its accuracy c50 (33,623) excess +.02 [-.01, +.06], G +.08 [-.01, +.13], >= 4 cells -.02 [-.09, +.07], end-mass
    +.05 [+.01, +.11], direction +.19 [+.10, +.23], held corner / one-variable -.02 [-.07, +.04] / +.04; at c25 (12,678)
    -.02 / +.12 / -.10 [-.19, -.02] / -.01 / +.06 / -.03 / -.01; at c10 (8,182) the >= 4-cell angle -.13 [-.21, -.05];
    N50 at its c10 (35,275) -.01 / +.13 / -.07 / -.03 [-.05, -.01] / +.15 / -.04 [-.06, -.02] / .00. So G and the
    direction slope are level or ahead, the excess angle level, the >= 4-cell angle behind at N100's c10 / c25, and N50
    holds slightly less at its c10. N200 (which forms both, later) at c10 / c25 / c50 / c75: excess -.03 / -.05 / -.04 /
    +.05, G -.02 / +.04 / +.03 / +.01, >= 4 cells -.13 [-.22, -.07] / -.16 [-.25, -.10] / -.11 [-.18, -.05] / +.01,
    end-mass -.03 / -.05 / -.04 / +.07, direction -.00 / -.04 / +.03 / +.19.
    WHAT IT TESTS: at the accuracies N100 / N50 reach, N1000 itself has no coherence < 1 (1.72 at its accuracy c25, 1.43
    at c50) and only its first-wave long-memory units (2.0% at c25 and c50: 20 units, formed at 20-160 episodes); its
    second wave and its cancellation come later (accuracy f .78 / .74). So the comparison tests only the first-wave
    units: lacking them does not delay the start of holding and coupling relative to accuracy on most rows. Whether the
    second-wave units or credit cancellation are needed for the late half (c50s at accuracy f .78-.94) cannot be tested:
    no run without them gets there.
    p1 (same width and initial weights, loss power 1) holds MORE at matched accuracy (end-mass slope +.29 [+.23, +.34] /
    +.25 [+.21, +.32] / +.29 [+.23, +.36] / +.24 [+.21, +.31] at accuracy c10 / c25 / c50 / c75; held shares +.17 to
    +.35), its >= 4-cell angle is far ahead where it is first defined (+.36 [+.24, +.48] at 2,400; +.13 [+.08, +.29] at
    accuracy c75) and its excess angle slightly ahead (+.04 to +.10, every interval above 0), while G and the direction
    slope are about level (G -.01 to +.08, direction -.00 to +.05). So the power-1 loss moves holding and the large-jump
    angle earlier and slightly raises the excess angle; G and the direction slope stay late (x6.1-10.1 with the excess
    angle). Holding and coupling do not simply dissociate: the large-jump angle moves with holding.

FAILED, NULL, QUALIFIED
  - "Every neural candidate keeps its place relative to passing naive": FAILED for coherence < 1, the long-memory share
    and the two r > .1 locks (x2.65-5.30 at N1000 vs x10.34-17.78 at N200), the effective dimension (50% point
    x1.24-1.32 at N = 1000, x0.13-0.69 narrower), the W_hh radius (x0.76 -> x0.11) and the dW-dimension lock (x0.31 ->
    x0.01). With the stored start-state cut the memory benefit stays before passing naive (x0.46-0.66), close to an
    accuracy level (the memory-cut network at .17-.24 when it happens); with the start-state controls its event is
    bracketed only by the two replay checkpoints in both wide runs (N1000 (560, 800], x0.67-0.96, the upper end inside
    passing naive's [798-876]; p1 (800, 1,280], x0.64-1.02), so there 'before passing naive' is not established by the
    interval rule (see (2)). The cross-band share < -.1 keeps its own-clock place (x4.59 / 5.00 / 5.24 in N1000 / N200 /
    N100; paired x1.09 [0.95-1.13], x1.14 [0.94-1.29] of N1000's) but not its accuracy (f .86 / .64 / .49).
  - "Width and loss set the pace, not the order" (the first headline): only width does. Under the power-1 loss "holding
    and coupling come late" does not replicate: the end-mass slope passes 10% of its gap before passing naive (x0.57
    [0.40-0.67]), cannot be ordered against it at 25% (x0.80 [0.69-1.19]) and is half-way at accuracy f .71 (N1000 .88;
    family test .918 at f = .5); the >= 4-cell angle is past half-way when it becomes defined (<= x1.85, accuracy f .59;
    family test .701 with it). G, the excess angle and the direction slope are late in every run that reaches them
    (x6.1-18.7).
  - "Holding and coupling dissociate under the loss power" (an earlier reading): withdrawn; the large-jump angle moves
    forward with holding and the excess angle is slightly ahead at matched accuracy.
  - "The late markers occur at a fixed accuracy, not a fixed training time": overstated. Apart from the cross-band
    share they occur at far more similar accuracies (f .74-.88) than times (x2.65-17.8), but across widths only
    coherence < 1 is accuracy-matched within its interval (and r(|u|, ent) > .1 between the two N = 1000 runs). The
    cross-band share is the reverse case: not accuracy-locked (f .86 / .64 / .49; N100 forms it at accuracy f .49
    without coherence < 1), but it keeps its own-clock place (x4.59 / 5.00 / 5.24; paired x1.09 [0.95-1.13], x1.14
    [0.94-1.29] of N1000's).
  - Long-memory units do not consistently precede the late c50s: at 3%, before them all at N1000, after holding and the
    >= 4-cell angle in p1, after the >= 4-cell angle (and not ordered against holding or G) in N200 (pooled share,
    flagged; a probe-averaged tau crosses 3% no later, so these after orders hold for the pooled share only); at half
    their own change, after the >= 4-cell angle at N1000 and after G in p1, but before most late c50s in N200.
  - The transient long-memory reliance does not relax inside every run's late window: before it in N200.
  - At f = .1 the families interleave in every run; at f = .25 p1's test fails (.019, holding's early rise).
  - N50 cannot test the order (its accuracy never reaches 25% of the gap).
  - The natural experiment tests only the first-wave long-memory units, and not every row is level (see (3)).
CHECKS (check()): every primitive of trained / echo at all 114 checkpoints equals the topic caches (24 primitives, max
  |diff| 0, NaN patterns equal); every exported row (curve, resampled curves, naive / Bayes and their resamples) equals
  the topic exports (max |diff| 4e-14) and their c0 / c50 per resample (max |diff| 2e-10 training episodes); the maps rows
  on the wormholes export's 58 checkpoints are identical, and on all 114 checkpoints the end-mass slope's c50 is 4,062
  [3,618-4,927] (export 3,936) and the direction slope's 10,034 [8,646-11,979] (export 10,113); the locks equal td_tt_
  credit_lags r(0) and the plot_training_dynamics series (max |diff| 1e-16); the memory cut's intact column equals the
  stored correctness at T in 100% of probes; Bayes's own memory benefit .1634 (cut to 1) / .0306 (cut to 5).
  11-02 fix round (verifier's issues): synthesis.npz re-exported with the tau row's own-change events; of the 5,317
  arrays the previous export held, 5,314 are identical; the tau rows' label lists (N1000, p1, N200) gained the two
  own-change labels, old labels unchanged at the same indices; the verifier's own code had reproduced every curve,
  resample and event of that export.
CAVEATS: one seed per run and one world: intervals resample environments, not seeds (the 4 extra N = 100 worlds have no
  ck files). Ratios inherit passing naive's interval (N50: [14,871-30,271]). Pooled rows (tau, radius) have no
  environment interval, and tau is a statistic of probe-and-step-averaged forget gates (DEFINITIONS). Censored rows
  give upper bounds only. Matched accuracy is a re-expression: rows that follow accuracy can do so for any reason. The
  probe-based held shares are thin (720 / 365 var-steps). Crossings are limited by the checkpoint spacing (every 20
  training episodes to 400, 80 to 2,400, 200 to 8,000, 800 after). The start-state control has two checkpoints per run.

Run from the repository root (CPU; data ~15 min per run, up to 4 runs in parallel; synthesis ~4 min; numbers, figure,
check ~1 min each):
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg \\
        PYTHONDONTWRITEBYTECODE=1 /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/td_tt_replication.py data <run> [<run> ...]
    ... td_tt_replication.py [synthesis] [check] [numbers] [figure]"""
import os
import sys
import glob
import time
import textwrap
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
TT = os.path.join(TD, "through_training")
OUT = os.path.join(TT, "replication")
RUNS = ("trained", "trained_p1", "trained_N200", "trained_N100", "trained_N50", "echo")
T, R, NE, PER = 30, 10, 50, 200
WIN = slice(16, 30)                                       # steps 17-30: the locked window of the credit keys
STEPS_ED = slice(1, 30)                                   # steps 2-30: effective dimension averaged per probe


# ═══════════════════════════════════════════════════════════════════
# Data: per checkpoint, the primitives of every row (the topic modules' own functions)
# ═══════════════════════════════════════════════════════════════════

def episodes(run):
    return sorted(int(os.path.basename(f)[3:8]) for f in glob.glob(os.path.join(AN, run, "ck_?????.npz")))


def _zs(a):
    """z-score within episode (axis 1) = gradflow_analysis._zs (the locked statistic)"""
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-12)


def _r_within(y, e):
    """per episode: the within-episode Pearson r of y and e over their steps (= the locked statistic at lag 0)"""
    return (_zs(np.asarray(y, np.float64)) * _zs(np.asarray(e, np.float64))).mean(1)


def _env_sum(env, v):
    return np.bincount(env, np.asarray(v, np.float64), NE)


def _env_median(env, v):
    """per environment, the median over all its probe-steps (v (n, steps))"""
    return np.array([np.median(v[env == e]) for e in range(NE)])


def build(run):
    """per checkpoint primitives of one run -> OUT/prims_<run>.npz (skips an existing cache)"""
    import td_tt_latent as tl
    import td_tt_jumps as tj
    import td_tt_splitting as ts
    import td_tt_wormholes as tw
    path = os.path.join(OUT, f"prims_{run}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    os.makedirs(OUT, exist_ok=True)
    P = np.load(os.path.join(AN, "probes.npz"))
    n = len(P["goal"])
    ar, goal, env, r = np.arange(n), P["goal"], P["env"], P["r"]
    assert np.array_equal(env, ar // PER), "probes are environment-major"
    rev = P["ent"].astype(np.float64).mean(-1)                                   # (n, T) entanglement, both variables
    perm = tj._null_perm(n, T - 1)
    C = tj.boot_counts()
    D = tw.probe_side()
    eps = episodes(run)
    out = {}
    t0 = time.time()

    def put(k, v):
        out.setdefault(k, []).append(np.asarray(v))
    for i, ep in enumerate(eps):
        z = np.load(os.path.join(AN, run, f"ck_{ep:05d}.npz"))
        lq = z["logq"]
        right_T = z["right"][ar, -1, goal]
        # latent (td_tt_latent.answer_stats): both right, far errors, distant pairs merged at T, per environment
        st = tl.answer_stats(lq, r, env, right=z["right"])
        for k in ("right_env", "far_env", "coll_env"):
            put("lat_" + k, st[k][-1])
        # jumps (td_tt_jumps.stats / extra_stats): G's per-episode primitives, the excess and band sums, accuracy
        s1 = tj.stats(lq, perm)
        put("jmp_path", s1["path"])
        put("jmp_rho", s1["rho"])
        s2 = tj.extra_stats(lq, right_T, C)
        for k in ("e_es", "ephi_es", "n_es", "nphi_es", "band_cnt", "band_sphi", "acc_env"):
            put("jmp_" + k, s2[k])
        # splitting (td_tt_splitting.split_stats, fold_tables at T)
        X = ts.positions(lq)
        s3 = ts.split_stats(X, env, r, rev, np.random.default_rng(1))
        put("spl_corr", s3["corr"])
        put("spl_corr_g", s3["corr_g"])
        F, _ = ts.fold_tables(X, env, r, np.array([T - 1]))
        put("spl_fold_T", F[:, 0].astype(np.float32))
        # wormholes, probes (td_tt_wormholes.emap_stats / jump_stats / split_stats): the maps and the held shares
        b = tw.norm(lq)
        em = tw.emap_stats(b, D["bB"], D["env"])
        js = tw.jump_stats(b @ tw.V, D["J"], D["env"])
        sp = tw.split_stats(b, D["bB"], D["env"], D["r"], D["S"])
        for k in ("e_st", "e_dot_st"):
            put("wh_" + k, em[k])
        for k in ("phi_st", "phi_dot_st"):
            put("wh_" + k, js[k])
        for k in ("joint_held", "held_two", "held_one", "held_one_m"):
            put("wh_" + k, sp[k])
        # neural: memory cut (an6) at T per environment, columns [w1, w2, w5, intact], goal variable
        tr = z["trunc_right"][ar, :, goal].astype(np.float64)                   # (n, 4)
        put("trunc_env", np.stack([_env_sum(env, tr[:, j]) for j in range(4)], 1) / PER)
        put("trunc_intact_is_right", float((tr[:, 3] == right_T).mean()))
        # activity: effective dimension per probe (steps 2-30), and its / the norm's locked r with entanglement
        put("effdim_env", _env_sum(env, z["effdim_u"][:, STEPS_ED].astype(np.float64).mean(1)) / PER)
        put("r_effdim_env", _env_sum(env, _r_within(z["effdim_u"][:, WIN], rev[:, WIN])) / PER)
        put("r_norm_env", _env_sum(env, _r_within(z["norm_u"][:, WIN], rev[:, WIN])) / PER)
        # credit (CREDIT runs only): coherence, its lock, the cross-band share on the final bands, dW dimension's lock
        if "cr_coh" in z.files:
            rows = z["cr_rows"]
            ec = rows // PER
            e2 = rows[:len(z["cr_bandx_fixed"])] // PER
            put("cr_cnt", np.bincount(ec, minlength=NE).astype(np.float64))
            put("cr_coh_med", _env_median(ec, z["cr_coh"]))
            put("cr_coh_rsum", _env_sum(ec, _r_within(z["cr_coh"], rev[rows][:, WIN])))
            put("cr_dim_rsum", _env_sum(ec, _r_within(z["cr_dim"], rev[rows][:, WIN])))
            put("cr_bxf_med", _env_median(e2, z["cr_bandx_fixed"]))
            put("cr_bxf_cnt", np.bincount(e2, minlength=NE).astype(np.float64))
        # pooled / weight keys (no environment resampling)
        put("tau", z["tau"])
        put("rho_hh", z["rho_hh"][:, 0])
        put("pr_u", np.nanmean(z["fac"][:, 6, 0, 1]))
        put("acc_stored", z["acc"][-1])
        if i % 10 == 0 or i == len(eps) - 1:
            print(f"{run} {i + 1}/{len(eps)} ep {ep} ({time.time() - t0:.0f} s)", flush=True)
    D_ = {k: np.stack(v) for k, v in out.items()}
    D_["episode"] = np.array(eps)
    D_["x"] = gap.train_x(eps)
    np.savez(path, **D_)
    print(f"saved {path} ({time.time() - t0:.0f} s)", flush=True)
    return D_


def build_trunc(run):
    """per checkpoint and environment, the memory-cut correctness at T of BOTH variables (td_tt_neural's 'ben'
    convention): (K, 50, 4 [w1, w2, w5, intact]) means over the environment's 200 probes x 2 variables ->
    OUT/trunc_<run>.npz"""
    path = os.path.join(OUT, f"trunc_{run}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    P = np.load(os.path.join(AN, "probes.npz"))
    env = P["env"]
    eps = episodes(run)
    out = []
    for ep in eps:
        tr = np.load(os.path.join(AN, run, f"ck_{ep:05d}.npz"))["trunc_right"].astype(np.float64)   # (n, 4, 2)
        out.append(np.stack([_env_sum(env, tr[:, j].sum(-1)) for j in range(4)], 1) / (2 * PER))
    D = dict(trunc_both=np.array(out), episode=np.array(eps))
    np.savez(path, **D)
    return D


# ═══════════════════════════════════════════════════════════════════
# References: Bayes's and naive's primitives on the same probes (the topic caches, read only)
# ═══════════════════════════════════════════════════════════════════

def references():
    """{'bayes': prims, 'naive': prims} (leading checkpoint axis of 1, the same keys as build()), 'counts' (the probe
    counts of td_tt_wormholes' maps and held shares), 'mem' {w: Bayes's own memory benefit per environment (50,)}"""
    L = np.load(os.path.join(TT, "latent", "refs.npz"))
    J = np.load(os.path.join(TT, "jumps", "jumps_references.npz"))
    JX = np.load(os.path.join(TT, "jumps", "jumps_extra_references.npz"))
    S = np.load(os.path.join(TT, "splitting", "probes_refs.npz"))
    W = np.load(os.path.join(TT, "wormholes", "probe_refs.npz"))
    P = np.load(os.path.join(AN, "probes.npz"))
    out = {}
    for nm in ("bayes", "naive"):
        d = {f"lat_{k}": L[f"{nm}__{k}"][-1] for k in ("right_env", "far_env", "coll_env")}
        d.update(jmp_path=J[f"{nm}_path"], jmp_rho=J[f"{nm}_rho"])
        d.update({f"jmp_{k}": JX[f"{nm}_{k}"] for k in ("e_es", "ephi_es", "n_es", "nphi_es", "band_cnt", "band_sphi",
                                                       "acc_env")})
        d.update(spl_corr=S[f"{nm}|corr"], spl_corr_g=S[f"{nm}|corr_g"], spl_fold_T=S[f"{nm}|fold"][:, -1])
        d.update({f"wh_{k}": W[f"{nm}_{k}"] for k in ("e_st", "e_dot_st", "phi_st", "phi_dot_st", "joint_held",
                                                     "held_two", "held_one", "held_one_m")})
        out[nm] = {k: np.asarray(v)[None] for k, v in d.items()}
    out["counts"] = {k: W[k] for k in W.files if k.startswith("n_")}
    n = len(P["goal"])
    ar, g, env = np.arange(n), P["goal"], P["env"]
    rb = P["right_bayes"][ar, g].astype(np.float64)
    out["mem"] = {w: _env_sum(env, rb - P[f"right_window{w}"][ar, g]) / PER for w in (1, 5)}
    rb2 = P["right_bayes"].astype(np.float64)
    out["mem"]["1b"] = _env_sum(env, (rb2 - P["right_window1"]).sum(-1)) / (2 * PER)
    return out


# ═══════════════════════════════════════════════════════════════════
# Rows: curves in every shared resample (td_tt_gap.boot_counts), with the topics' definitions
# ═══════════════════════════════════════════════════════════════════

# behaviour rows: key, name (the topic export's name), family, topic
BEH = (("acc", "goal accuracy at T", "answers", "jumps"),
       ("far", "far errors at T", "answers", "latent"),
       ("coll", "distant pairs merged at T", "answers", "latent"),
       ("split", "split gradient (most minus least entangled tenth)", "splitting", "splitting"),
       ("corners", "opposite corners", "answers", "splitting"),
       ("fold", "fold index (corners - edges)", "answers", "splitting"),
       ("excess", "jump excess angle over independent", "coupling", "jumps"),
       ("G", "long-path coupling G", "coupling", "jumps"),
       ("band4", "angle of jumps >= 4 cells", "coupling", "jumps"),
       ("emap", "end-mass slope within steps (maps)", "holding", "wormholes"),
       ("dmap", "direction slope within steps (maps)", "coupling", "wormholes"),
       ("held_two", "held per variable, corner-split ends (probes)", "holding", "wormholes"),
       ("held_one", "held per variable, one-variable end splits (probes)", "holding", "wormholes"))
EARLY, LATE = ("answers", "splitting"), ("holding", "coupling")
TESTED = ("acc", "far", "coll", "split", "excess", "G", "band4", "emap", "dmap")     # the family test's rows
EXTRA = ("corners", "fold", "held_two", "held_one")                                   # reported, not in the test
MIN_N = 50                                                                            # td_tt_jumps.MIN_N


def _rat(num, den, C):
    """ratio of sums: num, den (K, NE) -> full (K,), resampled (B, K)"""
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num.sum(1) / den.sum(1), (C @ num.T) / (C @ den.T)


def _env(Y, C):
    """per-environment values Y (K, NE) -> env_mean (full (K,), resampled (B, K))"""
    return gap.env_mean(np.asarray(Y, np.float64).T, C)


def beh_curve(d, key, Cn, C):
    """(full (K,), resampled (B, K)) of behaviour measure key from primitives d (leading checkpoint axis); Cn = the
    wormholes probe counts"""
    import td_tt_jumps as tj
    import td_tt_wormholes as tw
    if key in ("acc", "far", "coll"):
        return _env({"acc": d["jmp_acc_env"], "far": d["lat_far_env"], "coll": d["lat_coll_env"]}[key], C)
    if key == "split":
        c = np.nanmean(d["spl_corr"].astype(np.float64), -2)                        # (K, NE, 10): mean over t0
        return _env(c[..., -1] - c[..., 0], C)
    if key in ("corners", "fold"):
        F = d["spl_fold_T"].astype(np.float64)
        y = F[..., R - 1, R - 1] - (0. if key == "corners" else (F[..., R - 1, 0] + F[..., 0, R - 1]) / 2)
        return _env(y, C)
    if key == "excess":
        a, ab = _rat(d["jmp_ephi_es"].sum(-1), d["jmp_e_es"].sum(-1), C)
        b, bb = _rat(d["jmp_nphi_es"].sum(-1), d["jmp_n_es"].sum(-1), C)
        return a - b, ab - bb
    if key == "G":
        K = len(d["jmp_path"])
        f = np.stack([tj.rho_fifths(d["jmp_path"][k], d["jmp_rho"][k]) for k in range(K)])
        return f[:, -1] - f[:, 0], np.stack([tj._G_boot(d["jmp_path"][k], d["jmp_rho"][k], C) for k in range(K)], 1)
    if key == "band4":
        cnt, sp = d["jmp_band_cnt"][..., 0].astype(np.float64), d["jmp_band_sphi"][..., 0].astype(np.float64)
        cf, cb = cnt.sum(1), C @ cnt.T
        with np.errstate(invalid="ignore", divide="ignore"):
            return (np.where(cf >= MIN_N, sp.sum(1) / np.maximum(cf, 1), np.nan),
                    np.where(cb >= MIN_N, (C @ sp.T) / np.maximum(cb, 1), np.nan))
    if key in ("emap", "dmap"):
        Z = {k: d["wh_" + k] for k in ("e_st", "e_dot_st", "phi_st", "phi_dot_st")}
        kind = "e" if key == "emap" else "phi"
        return tw.maps_slope(Z, Cn, kind), tw.maps_slope(Z, Cn, kind, C)
    if key in ("held_two", "held_one"):
        den = Cn["n_two_v" if key == "held_two" else "n_one_v"].astype(np.float64)
        num = d["wh_" + key].astype(np.float64)
        return num.sum(1) / den.sum(), (C @ num.T) / (C @ den)[:, None]
    raise KeyError(key)


# neural rows: key, name, family ('early' / 'late' candidate, by the Trained RNN's established timing), needs
NEU = (("mem1", "memory benefit, cut to 1 step (intact - w1)", "neural early", "trunc"),
       ("mem1b", "memory benefit, cut to 1 step, both variables", "neural early", "trunc"),
       ("mem5", "memory benefit, cut to 5 steps (intact - w5)", "neural early", "trunc"),
       ("effdim", "effective dimension of u (steps 2-30)", "neural early", "effdim"),
       ("r_dim", "r(dW dimension, entanglement)", "neural early", "credit"),
       ("coh", "credit coherence (environment median)", "neural late", "credit"),
       ("r_coh", "r(credit coherence, entanglement)", "neural late", "credit"),
       ("bxf", "cross-band share, final bands (median)", "neural late", "credit"),
       ("r_norm", "r(|u|, entanglement)", "neural late", "activity"),
       ("tau", "share of units with tau > 5", "neural late", "pooled"),
       ("rho", "W_hh spectral radius, smallest of the i, f, o blocks", "neural early", "pooled"))


def neu_curve(d, key, C):
    """(full (K,), resampled (B, K) or None for pooled keys) of a neural measure; None if the run lacks the keys"""
    if key in ("mem1", "mem5"):
        tr = d["trunc_env"].astype(np.float64)                                     # (K, NE, [w1, w2, w5, intact])
        return _env(tr[..., 3] - tr[..., 0 if key == "mem1" else 2], C)
    if key == "mem1b":
        tr = d["trunc_both"]
        return _env(tr[..., 3] - tr[..., 0], C)
    if key == "effdim":
        return _env(d["effdim_env"], C)
    if key == "r_norm":
        return _env(d["r_norm_env"], C)
    if key in ("coh", "r_coh", "r_dim", "bxf"):
        if "cr_coh_med" not in d:
            return None
        if key == "coh":
            return _env(d["cr_coh_med"], C)
        if key == "bxf":
            return _env(d["cr_bxf_med"], C)
        return _rat(d["cr_coh_rsum" if key == "r_coh" else "cr_dim_rsum"], d["cr_cnt"], C)
    if key == "tau":
        return (np.asarray(d["tau"], np.float64) > 5).mean(1), None
    if key == "rho":
        return np.asarray(d["rho_hh"], np.float64)[:, [0, 1, 3]].min(1), None
    raise KeyError(key)


# ═══════════════════════════════════════════════════════════════════
# Crossings
# ═══════════════════════════════════════════════════════════════════

def _first(x, y, lev, sign):
    """first training episode (log-interpolated) at which y (..., K) reaches lev (sign +1: at or above); NaN if never"""
    y = np.atleast_2d(np.asarray(y, np.float64))
    lev = np.broadcast_to(np.asarray(lev, np.float64), y.shape[:1])
    out = np.full(len(y), np.nan)
    lx = np.log(np.asarray(x, np.float64))
    for i in range(len(y)):
        with np.errstate(invalid="ignore"):
            past = sign * (y[i] - lev[i]) >= 0
        if not past.any():
            continue
        k = int(np.argmax(past))
        if k == 0:
            out[i] = x[0]
            continue
        f = (lev[i] - y[i, k - 1]) / (y[i, k] - y[i, k - 1])
        out[i] = np.exp(lx[k - 1] + np.clip(f, 0, 1) * (lx[k] - lx[k - 1]))
    return out


def event(x, y, yb, lev, levb, sign=1., first=False, label=""):
    """one timed event of a curve: dict(label, level, p, lo, hi, share, boot); stays past the level to the end
    (td_tt_gap.crossing) unless first (first reached); yb None = a pooled curve (no resamples: boot = the point)"""
    if first:
        p = float(_first(x, y[None], lev, sign)[0])
        cb = _first(x, yb, levb, sign) if yb is not None else None
    else:
        p = float(gap.crossing(x, y, lev, sign))
        cb = gap.crossing(x, yb, levb, sign) if yb is not None else None
    if cb is None:
        return dict(label=label, level=float(lev), p=p, lo=np.nan, hi=np.nan, share=float(np.isfinite(p)),
                    boot=np.full(gap.N_BOOT, p), pooled=True)
    lo, hi, share = gap.interval(cb)
    return dict(label=label, level=float(lev), p=p, lo=lo, hi=hi, share=share, boot=cb, pooled=False)


LEVELS = (0., .1, .25, .5, .75)


def beh_events(x, r):
    """c_L of a behaviour row's gap fraction (td_tt_gap conventions) for LEVELS: {L: event}. censored: the curve has
    no value (NaN: fewer than 50 jumps of >= 4 cells) at the first checkpoints and is already past L at its first
    value, so the crossing marks when the measure becomes defined, not a rise (left out of the family test)"""
    f = gap.gap_fraction(r["value"], r["naive"], r["bayes"])
    fb = gap.gap_fraction(r["boot"], r["naive_boot"], r["bayes_boot"])
    ok = np.flatnonzero(np.isfinite(f))
    k0 = int(ok[0]) if len(ok) else 0
    out = {}
    for L in LEVELS:
        out[L] = event(x, f, fb, L, np.full(gap.N_BOOT, L), label=f"c{int(L * 100)}")
        out[L]["level"] = float(r["naive"] + L * (r["bayes"] - r["naive"]))
        out[L]["censored"] = bool(k0 > 0 and f[k0] > L and np.isfinite(out[L]["p"])
                                  and out[L]["p"] <= x[k0] * (1 + 1e-9))          # exp(log(x)) rounding
    return out


def _end(y):
    return np.asarray(y, np.float64)[..., -3:].mean(-1)


def neu_events(x, key, y, yb, ref):
    """the timed events of a neural row: [event] (labels say which reference each uses)"""
    ev = []
    if key in ("mem1", "mem5", "mem1b"):
        bv, bb = ref                                      # Bayes's own benefit (full, resampled); naive's is 0
        fv, fb = y / bv, yb / bb[:, None]
        if key in ("mem1", "mem1b"):
            ev.append(event(x, fv, fb, .5, np.full(gap.N_BOOT, .5), label="half of Bayes's level"))
            ev.append(event(x, fv, fb, 1., np.ones(gap.N_BOOT), label="Bayes's level"))
        else:
            pk, pkb = np.nanmax(y), np.nanmax(yb, 1)
            if np.percentile(pkb - bb, 5) <= 0:            # no transient beyond Bayes's own: nothing to time
                return ev
            ev.append(event(x, y, yb, pk / 2, pkb / 2, first=True, label="rise: half its peak, first reached"))
            k, kb = int(np.nanargmax(y)), np.nanargmax(yb, 1)
            ev.append(dict(label="its peak", level=float(pk), p=float(x[k]), lo=float(np.percentile(x[kb], 5)),
                           hi=float(np.percentile(x[kb], 95)), share=1., boot=x[kb].astype(np.float64), pooled=False))
            lev, levb = pk + .5 * (_end(y) - pk), pkb + .5 * (_end(yb) - pkb)
            ev.append(event(x, y, yb, lev, levb, sign=-1., label="relaxation: half-way back from its peak"))
    elif key == "effdim":
        y0, y1, y0b, y1b = y[0], _end(y), yb[:, 0], _end(yb)
        f, fb = (y - y0) / (y1 - y0), (yb - y0b[:, None]) / (y1b - y0b)[:, None]
        for L in (.1, .5):
            e = event(x, f, fb, L, np.full(gap.N_BOOT, L), label=f"{int(L * 100)}% of its own change")
            e["level"] = float(y0 + L * (y1 - y0))
            ev.append(e)
    elif key in ("r_coh", "r_norm"):
        ev.append(event(x, y, yb, .1, np.full(gap.N_BOOT, .1), label="r > .1"))
        if _end(y) > .1:                                  # a lock that ends below .1 is not timed by its half
            ev.append(event(x, y, yb, _end(y) / 2, _end(yb) / 2, label="half its end value (reference 0)"))
    elif key == "r_dim":
        ev.append(event(x, y, yb, y[0] / 2, yb[:, 0] / 2, sign=-1.,
                        label="half-way from its initial value to 0"))
        ev.append(event(x, y, yb, 0., np.zeros(gap.N_BOOT), sign=-1., label="r < 0"))
    elif key == "coh":
        ev.append(event(x, y, yb, 1., np.ones(gap.N_BOOT), sign=-1., label="below 1 (pieces cancel)"))
    elif key == "bxf":
        ev.append(event(x, y, yb, -.1, np.full(gap.N_BOOT, -.1), sign=-1., label="below -.1"))
    elif key == "tau":
        for lev in (.03, .05):                            # one absolute amount for every run (ad hoc levels)
            ev.append(event(x, y, None, lev, None, label=f"> {lev:.0%} of units"))
        y0, y1 = float(y[0]), float(_end(y))
        if y1 > y0:                                       # the rule-conform version: c10 / c50 of its own change
            for L in (.1, .5):                            # (none where the share ends at its initial 0)
                e = event(x, y, None, y0 + L * (y1 - y0), None, label=f"{int(L * 100)}% of its own change")
                ev.append(e)
    elif key == "rho":
        ev.append(event(x, y, None, 1., None, label="> 1 (expansive)"))
    return ev


# ═══════════════════════════════════════════════════════════════════
# Synthesis: every row and event of every run -> OUT/synthesis.npz
# ═══════════════════════════════════════════════════════════════════

def run_rows(run, d, Rf, C):
    """the rows of one run: {key: row dict} with value / boot and (behaviour) naive / Bayes levels, plus events"""
    x = np.asarray(d["x"], np.float64)
    Cn = Rf["counts"]
    out = {}
    for key, name, fam, topic in BEH:
        v, vb = beh_curve(d, key, Cn, C)
        nv, nb = beh_curve(Rf["naive"], key, Cn, C)
        bv, bb = beh_curve(Rf["bayes"], key, Cn, C)
        r = dict(key=key, name=name, family=fam, topic=topic, kind="behaviour", value=v, boot=vb,
                 naive=float(nv[0]), bayes=float(bv[0]), naive_boot=nb[:, 0], bayes_boot=bb[:, 0])
        r["events"] = beh_events(x, r)
        out[key] = r
    for key, name, fam, need in NEU:
        c = neu_curve(d, key, C)
        if c is None:
            continue
        v, vb = c
        ref = (gap.env_mean(Rf["mem"][{"mem1": 1, "mem5": 5, "mem1b": "1b"}[key]], C)
               if key in ("mem1", "mem5", "mem1b") else None)
        r = dict(key=key, name=name, family=fam, topic=need, kind="neural", value=v, boot=vb,
                 ref=(np.nan if ref is None else float(ref[0])))
        r["events"] = {e["label"]: e for e in neu_events(x, key, v, vb, ref)}
        out[key] = r
    return out


def synthesis():
    """-> OUT/synthesis.npz: per run, x, every row's curve, its resamples (float64) and references, and every event
    (point, 5-95%, share crossing, resampled crossings)"""
    Rf = references()
    C = gap.boot_counts()
    store = {"runs": np.array(RUNS), "n_boot": gap.N_BOOT, "seed": gap.BOOT_SEED}
    for run in RUNS:
        t0 = time.time()
        d = build(run)
        assert np.array_equal(d["episode"], build("trained")["episode"]), "every run on the same 114 checkpoints"
        tb = build_trunc(run)
        assert np.array_equal(tb["episode"], d["episode"])
        d = dict(d, trunc_both=tb["trunc_both"])
        rows = run_rows(run, d, Rf, C)
        store[f"{run}|x"] = np.asarray(d["x"], np.float64)
        store[f"{run}|keys"] = np.array(list(rows))
        for key, r in rows.items():
            p = f"{run}|{key}|"
            for k in ("name", "family", "topic", "kind"):
                store[p + k] = np.array(r[k])
            store[p + "value"] = np.asarray(r["value"], np.float64)
            if r["boot"] is not None:
                store[p + "boot"] = np.asarray(r["boot"], np.float64)
            for k in ("naive", "bayes", "naive_boot", "bayes_boot", "ref"):
                if k in r:
                    store[p + k] = np.asarray(r[k], np.float64)
            labels = list(r["events"])
            store[p + "events"] = np.array([str(lab) for lab in labels])
            for j, lab in enumerate(labels):
                e = r["events"][lab]
                for k in ("level", "p", "lo", "hi", "share"):
                    store[f"{p}ev{j}|{k}"] = np.float64(e[k])
                store[f"{p}ev{j}|censored"] = np.bool_(e.get("censored", False))
                store[f"{p}ev{j}|boot"] = np.asarray(e["boot"], np.float64)
                store[f"{p}ev{j}|pooled"] = np.bool_(e["pooled"])
        print(f"  {run}: {len(rows)} rows ({time.time() - t0:.0f} s)", flush=True)
    path = os.path.join(OUT, "synthesis.npz")
    np.savez(path, **store)
    print("saved", path)
    return path


def load():
    """synthesis.npz -> {run: {'x': x, 'rows': {key: row with 'events' {label: event}}}}; behaviour events are keyed by
    their level L (0, .1, .25, .5, .75; stored as str(L))"""
    with np.load(os.path.join(OUT, "synthesis.npz"), allow_pickle=False) as z:
        Z = {k: z[k] for k in z.files}
    S = {}
    for run in Z["runs"]:
        run = str(run)
        rows = {}
        for key in Z[f"{run}|keys"]:
            key, p = str(key), f"{run}|{key}|"
            r = {k: str(Z[p + k]) for k in ("name", "family", "topic", "kind")}
            r["value"] = Z[p + "value"]
            r["boot"] = Z.get(p + "boot")
            for k in ("naive", "bayes", "naive_boot", "bayes_boot", "ref"):
                if p + k in Z:
                    r[k] = Z[p + k] if Z[p + k].ndim else float(Z[p + k])
            ev = {}
            for j, lab in enumerate(Z[p + "events"]):
                e = {k: float(Z[f"{p}ev{j}|{k}"]) for k in ("level", "p", "lo", "hi", "share")}
                e.update(boot=Z[f"{p}ev{j}|boot"], pooled=bool(Z[f"{p}ev{j}|pooled"]), label=str(lab),
                         censored=bool(Z[f"{p}ev{j}|censored"]))
                ev[float(lab) if r["kind"] == "behaviour" else str(lab)] = e
            r["events"] = ev
            rows[key] = r
        S[run] = {"x": Z[f"{run}|x"], "rows": rows}
    return S


# ═══════════════════════════════════════════════════════════════════
# Derived statistics: own clock, matched accuracy, order, family test
# ═══════════════════════════════════════════════════════════════════

def _inf(v):
    return np.where(np.isnan(v), np.inf, v)


def passes_naive(S, run):
    """the run's passing-naive event: goal accuracy at T past naive's level for good (c0)"""
    return S[run]["rows"]["acc"]["events"][0.]


def ratio(e, ref):
    """paired ratio of two crossings: point, 5-95% over the shared resamples in which ref crosses (a resample in which
    e never crosses counts as infinitely late: an open upper bound is inf), share of resamples in which both cross"""
    ok = np.isfinite(ref["boot"])
    rb = _inf(e["boot"][ok]) / ref["boot"][ok]
    lo, hi = ((np.percentile(rb, 5, method="lower"), np.percentile(rb, 95, method="higher")) if ok.any()
              else (np.nan, np.nan))
    p = e["p"] / ref["p"] if (np.isfinite(ref["p"]) and np.isfinite(e["p"])) else np.nan
    return dict(p=p, lo=float(lo), hi=float(hi), both=float((np.isfinite(e["boot"]) & np.isfinite(ref["boot"])).mean()))


def ratio_of_ratios(S, run, key, lab, base="trained"):
    """an event's own-clock ratio (event / the run's passing naive) over the base run's, paired per resample: point,
    5-95% over the resamples in which all four crossings exist, and that share (x1 = the same own-clock place)"""
    e, e0 = _ev(S, run, key, lab), _ev(S, base, key, lab)
    a = e["boot"] / passes_naive(S, run)["boot"]
    b = e0["boot"] / passes_naive(S, base)["boot"]
    ok = np.isfinite(a) & np.isfinite(b)
    lo, hi = np.percentile(a[ok] / b[ok], [5, 95]) if ok.sum() > 10 else (np.nan, np.nan)
    p = (e["p"] / passes_naive(S, run)["p"]) / (e0["p"] / passes_naive(S, base)["p"])
    return dict(p=float(p), lo=float(lo), hi=float(hi), share=float(ok.mean()))


def acc_frac(S, run):
    """the run's goal-accuracy gap fraction: full (K,), resampled (B, K)"""
    r = S[run]["rows"]["acc"]
    return (gap.gap_fraction(r["value"], r["naive"], r["bayes"]),
            gap.gap_fraction(r["boot"], r["naive_boot"], r["bayes_boot"]))


def acc_at(S, run, e):
    """the run's accuracy gap fraction at an event's crossing (log-interpolated): point and 5-95% over the paired
    resamples (each resample's accuracy curve at that resample's crossing; NaN where it never crosses)"""
    x = np.log(S[run]["x"])
    f, fb = acc_frac(S, run)
    p = float(np.interp(np.log(e["p"]), x, f)) if np.isfinite(e["p"]) else np.nan
    v = np.array([np.interp(np.log(t), x, fb[b]) if np.isfinite(t) else np.nan for b, t in enumerate(e["boot"])])
    ok = np.isfinite(v)
    lo, hi = np.percentile(v[ok], [5, 95]) if ok.sum() > 10 else (np.nan, np.nan)
    return dict(p=p, lo=lo, hi=hi, share=float(ok.mean()), boot=v)


def frac_at(S, run, key, e):
    """a behaviour row's gap fraction at an event's crossing (e.g. the late rows when accuracy is half-way): point
    and 5-95% over the paired resamples"""
    r = S[run]["rows"][key]
    x = np.log(S[run]["x"])
    f = gap.gap_fraction(r["value"], r["naive"], r["bayes"])
    fb = gap.gap_fraction(r["boot"], r["naive_boot"], r["bayes_boot"])
    p = float(np.interp(np.log(e["p"]), x, f)) if np.isfinite(e["p"]) else np.nan
    v = np.array([np.interp(np.log(t), x, fb[b]) if np.isfinite(t) else np.nan for b, t in enumerate(e["boot"])])
    ok = np.isfinite(v)
    lo, hi = np.nanpercentile(v[ok], [5, 95]) if ok.sum() > 10 else (np.nan, np.nan)
    return dict(p=p, lo=lo, hi=hi, share=float(ok.mean()), boot=v)


def value_at(S, run, key, e):
    """a neural row's raw value at an event's crossing (log-interpolated): point and 5-95% over the paired resamples
    (a pooled row: its full curve at each resample's crossing)"""
    r = S[run]["rows"][key]
    x = np.log(S[run]["x"])
    yb = r["boot"] if r["boot"] is not None else np.broadcast_to(r["value"], (gap.N_BOOT, len(x)))
    p = float(np.interp(np.log(e["p"]), x, r["value"])) if np.isfinite(e["p"]) else np.nan
    v = np.array([np.interp(np.log(t), x, yb[b]) if np.isfinite(t) else np.nan for b, t in enumerate(e["boot"])])
    ok = np.isfinite(v)
    lo, hi = np.percentile(v[ok], [5, 95]) if ok.sum() > 10 else (np.nan, np.nan)
    return dict(p=p, lo=lo, hi=hi, share=float(ok.mean()), boot=v)


def matched(S, run, key, L, base="trained"):
    """a behaviour row's gap fraction when the run's accuracy passes f = L for good (accuracy c_L), and its paired
    difference from the base run's at the base run's accuracy c_L (same environment resamples; NaN where either
    accuracy never crosses)"""
    a = frac_at(S, run, key, S[run]["rows"]["acc"]["events"][L])
    b = frac_at(S, base, key, S[base]["rows"]["acc"]["events"][L])
    dv = a["boot"] - b["boot"]
    ok = np.isfinite(dv)
    lo, hi = np.percentile(dv[ok], [5, 95]) if ok.sum() > 10 else (np.nan, np.nan)
    return a, dict(p=a["p"] - b["p"], lo=lo, hi=hi, share=float(ok.mean()))


def matched_at(S, run, key, xv, base="trained"):
    """a behaviour row's gap fraction at the run's checkpoint xv, against the base run's at the point where the base
    run's accuracy passes (for good) the run's accuracy at xv, per resample (paired; NaN where the row is undefined at
    xv or the base run's accuracy never passes that level): dict(acc, f, base_x, base_f, p, lo, hi, share)"""
    x, xb = S[run]["x"], S[base]["x"]
    k = int(np.flatnonzero(x == xv)[0])
    fa, fab = acc_frac(S, run)
    ga, gab = acc_frac(S, base)
    r, rb = S[run]["rows"][key], S[base]["rows"][key]
    f = gap.gap_fraction(r["value"], r["naive"], r["bayes"])
    fb = gap.gap_fraction(r["boot"], r["naive_boot"], r["bayes_boot"])
    g = gap.gap_fraction(rb["value"], rb["naive"], rb["bayes"])
    gb = gap.gap_fraction(rb["boot"], rb["naive_boot"], rb["bayes_boot"])
    t = gap.crossing(xb, ga, fa[k])
    tb = gap.crossing(xb, gab, fab[:, k])
    g_t = float(np.interp(np.log(t), np.log(xb), g)) if np.isfinite(t) else np.nan
    ok = np.isfinite(tb) & np.isfinite(fb[:, k])
    d = np.full(len(tb), np.nan)
    d[ok] = fb[ok, k] - np.array([np.interp(np.log(tb[b]), np.log(xb), gb[b]) for b in np.flatnonzero(ok)])
    lo, hi = np.percentile(d[ok], [5, 95]) if ok.sum() > 10 else (np.nan, np.nan)
    return dict(acc=float(fa[k]), f=float(f[k]), base_x=float(t), base_f=g_t, p=float(f[k]) - g_t, lo=lo, hi=hi,
                share=float(ok.mean()))


def order(a, b):
    """'before' / 'after' / 'not ordered' (interval rule: 5-95% intervals disjoint; never = after the end) and the
    paired share P(a < b) over the shared resamples"""
    if not (np.isfinite(a["p"]) or np.isfinite(b["p"])):
        return "neither crosses", np.nan
    ahi = a["hi"] if np.isfinite(a["hi"]) else np.inf
    bhi = b["hi"] if np.isfinite(b["hi"]) else np.inf
    alo = a["lo"] if np.isfinite(a["lo"]) else (a["p"] if a["pooled"] else np.inf)
    blo = b["lo"] if np.isfinite(b["lo"]) else (b["p"] if b["pooled"] else np.inf)
    if a["pooled"]:
        ahi = a["p"] if np.isfinite(a["p"]) else np.inf
    if b["pooled"]:
        bhi = b["p"] if np.isfinite(b["p"]) else np.inf
    rel = "before" if ahi < blo else "after" if alo > bhi else "not ordered"
    return rel, float((_inf(a["boot"]) < _inf(b["boot"])).mean())


def family_test(S, run, level=.5, keys=TESTED, censored=False):
    """share of the paired resamples in which every early row's crossing (answers, splitting) precedes every late
    row's (holding, coupling) at gap fraction `level` (never = after the end; both never = not ordered); censored
    rows are left out unless censored=True (then their crossing = when they become defined: a sensitivity check)"""
    rows = S[run]["rows"]
    keys = [k for k in keys if censored or not rows[k]["events"][level]["censored"]]
    A = [_inf(rows[k]["events"][level]["boot"]) for k in keys if rows[k]["family"] in EARLY]
    B = [_inf(rows[k]["events"][level]["boot"]) for k in keys if rows[k]["family"] in LATE]
    a, b = np.max(A, 0), np.min(B, 0)
    return float(((a < b) & np.isfinite(a)).mean())


# ═══════════════════════════════════════════════════════════════════
# Checks: the primitives and rows of the Trained RNN and the ESN against the topics' caches and exports
# ═══════════════════════════════════════════════════════════════════

def _mx(a, b):
    """max |a - b| over finite entries, and whether the NaN patterns agree"""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    same = np.array_equal(np.isnan(a), np.isnan(b))
    d = np.abs(a - b)
    return (float(np.nanmax(d)) if np.isfinite(d).any() else 0.), same


def check():
    """prints: (1) every primitive of trained / echo at all 114 checkpoints = the topic caches; (2) every exported row
    (curve, resampled curves, naive / Bayes levels and their resamples, resampled c0 / c50) = the topic exports (the
    wormholes maps on its 58 checkpoints); (3) the credit lock = td_tt_credit_lags' r(0) and the activity locks =
    plot_training_dynamics' series (tau 0); (4) the memory-cut intact column = the stored correctness at T"""
    print("(1) primitives vs the topic caches (all 114 checkpoints): max |diff|, NaN patterns equal")
    for run in ("trained", "echo"):
        d = build(run)
        L = np.load(os.path.join(TT, "latent", f"probes_{run}.npz"))
        J = np.load(os.path.join(TT, "jumps", f"jumps_{run}.npz"))
        JX = np.load(os.path.join(TT, "jumps", f"jumps_extra_{run}.npz"))
        Sp = np.load(os.path.join(TT, "splitting", f"probes_{run}.npz"))
        W = np.load(os.path.join(TT, "wormholes", f"probe_{run}.npz"))
        CL = np.load(os.path.join(TT, "credit_lags", "credit_lags.npz"))
        pairs = [(f"lat_{k}", L[k][:, -1]) for k in ("right_env", "far_env", "coll_env")]
        pairs += [("jmp_path", J["path"]), ("jmp_rho", J["rho"])]
        pairs += [(f"jmp_{k}", JX[k]) for k in ("e_es", "ephi_es", "n_es", "nphi_es", "band_cnt", "band_sphi", "acc_env")]
        pairs += [("spl_corr", Sp["corr"]), ("spl_corr_g", Sp["corr_g"]), ("spl_fold_T", Sp["fold"][:, :, -1])]
        pairs += [(f"wh_{k}", W[k]) for k in ("e_st", "e_dot_st", "phi_st", "phi_dot_st", "joint_held", "held_two",
                                              "held_one", "held_one_m")]
        pairs += [("cr_coh_rsum", CL[f"{run}_sum"][:, :, 8])]
        worst = max(_mx(d[k], v)[0] for k, v in pairs)
        nan_ok = all(_mx(d[k], v)[1] for k, v in pairs)
        print(f"  {run}: {len(pairs)} primitives, max |diff| {worst:.1e}, NaN patterns equal {nan_ok}; checkpoints "
              f"{np.array_equal(d['episode'], L['episode'])}; memory-cut intact column = stored correctness at T in "
              f"{d['trunc_intact_is_right'].min():.4f} of probes (min over checkpoints)")
    print("(2) rows vs the topic exports (value, resampled curves, naive / Bayes and their resamples; c0 / c50 per "
          "resample)")
    S = load()
    for topic in ("jumps", "latent", "splitting", "wormholes"):
        E = gap.load(topic)
        xe = E["x"]
        for run in ("trained", "echo"):
            x = S[run]["x"]
            k_ = np.searchsorted(x, xe)
            assert np.array_equal(x[k_], xe)
            for key, r in S[run]["rows"].items():
                if r["topic"] != topic or r["kind"] != "behaviour":
                    continue
                i = [j for j in range(len(E["names"])) if str(E["names"][j]) == r["name"] and str(E["run"][j]) == run]
                if not i:
                    continue
                i = i[0]
                dv = _mx(r["value"][k_], E["value"][i])[0]
                db = _mx(r["boot"][:, k_], E["boot"][i])[0]
                dr = max(abs(r["naive"] - E["naive"][i]), abs(r["bayes"] - E["bayes"][i]),
                         _mx(r["naive_boot"], E["naive_boot"][i])[0], _mx(r["bayes_boot"], E["bayes_boot"][i])[0])
                f = gap.gap_fraction(r["boot"][:, k_], r["naive_boot"], r["bayes_boot"])
                dc = max(_mx(gap.crossing(xe, f, L), E[f"c{int(L * 100)}_boot"][i])[0] for L in (0., .5))
                full = (f"; own 114 checkpoints: c50 {r['events'][.5]['p']:,.0f} vs the export's {E['c50'][i]:,.0f}"
                        if len(xe) < len(x) else "")
                print(f"  {topic:9s} {run:7s} {r['name'][:46]:46s} value {dv:.1e} boot {db:.1e} refs {dr:.1e} "
                      f"c0/c50 per resample {dc:.1e}{full}")
    print("(3) locks")
    for run in ("trained", "echo"):
        s = dict(np.load(os.path.join(AN, f"series_{run}.npz"), allow_pickle=True))["s"].item()
        d = build(run)
        eps = d["episode"]
        a = np.array([s[e]["xc_act"][:, 8] for e in eps])
        u = np.array([s[e]["xc_upd"][:, 8] for e in eps])
        print(f"  {run}: r(|u|, ent) mean over probes vs series: {np.abs(d['r_norm_env'].mean(1) - a[:, 0]).max():.1e};"
              f" r(effdim, ent) {np.abs(d['r_effdim_env'].mean(1) - a[:, 1]).max():.1e}; r(coherence, ent) "
              f"{np.abs(d['cr_coh_rsum'].sum(1) / d['cr_cnt'].sum(1) - u[:, 0]).max():.1e}; r(dW dim, ent) "
              f"{np.abs(d['cr_dim_rsum'].sum(1) / d['cr_cnt'].sum(1) - u[:, 2]).max():.1e}")
    R_ = references()
    print(f"(4) Bayes's own memory benefit (goal at T): cut to 1 step {R_['mem'][1].mean():.4f}, to 5 steps "
          f"{R_['mem'][5].mean():.4f}")


# ═══════════════════════════════════════════════════════════════════
# Numbers
# ═══════════════════════════════════════════════════════════════════

NAME = {"trained": "Trained RNN (N = 1000)", "trained_p1": "power-1 loss (N = 1000)", "trained_N200": "N = 200",
        "trained_N100": "N = 100", "trained_N50": "N = 50", "echo": "ESN"}
SHORT = {"trained": "N1000", "trained_p1": "p1", "trained_N200": "N200", "trained_N100": "N100", "trained_N50": "N50",
         "echo": "ESN"}


def _fe(e, dec=0):
    """'point [lo-hi]' of an event ('never' / open bounds; the share of resamples crossing when below 99.5%)"""
    if e is None:
        return "n/a"
    if not np.isfinite(e["p"]):
        return f"never ({e['share']:.1%} of resamples cross)" if not e.get("pooled") else "never"
    if e.get("pooled"):
        return f"{e['p']:,.{dec}f} (pooled: no interval)"
    hi = f"{e['hi']:,.{dec}f}" if np.isfinite(e["hi"]) else "never"
    lo = f"{e['lo']:,.{dec}f}" if np.isfinite(e["lo"]) else "never"
    return f"{e['p']:,.{dec}f} [{lo}-{hi}]" + ("" if e["share"] >= .995 else f" ({e['share']:.0%} cross)")


def _fr(q, dec=2):
    if not np.isfinite(q["p"]):
        return "-"
    hi = f"{q['hi']:.{dec}f}" if np.isfinite(q["hi"]) else "inf"
    return f"x{q['p']:.{dec}f} [{q['lo']:.{dec}f}-{hi}]"


def _ff(q, dec=2):
    if not np.isfinite(q["p"]):
        return "-"
    return f"{q['p']:+.{dec}f} [{q['lo']:+.{dec}f}, {q['hi']:+.{dec}f}]"


def composite(S, run, level, fams, how, keys=TESTED):
    """a family's latest (how = max) or earliest (min) crossing of gap fraction `level` over the tested rows, per
    resample (never = after the end): an event"""
    rows = S[run]["rows"]
    sel = [rows[k] for k in keys if rows[k]["family"] in fams and not rows[k]["events"][level]["censored"]]
    B = np.stack([_inf(r["events"][level]["boot"]) for r in sel])
    P_ = np.array([np.inf if not np.isfinite(r["events"][level]["p"]) else r["events"][level]["p"] for r in sel])
    cb = B.max(0) if how == "max" else B.min(0)
    p = P_.max() if how == "max" else P_.min()
    cb = np.where(np.isinf(cb), np.nan, cb)
    lo, hi, share = gap.interval(cb)
    return dict(label=f"{how} c{int(level * 100)}", level=level, p=float(p) if np.isfinite(p) else np.nan, lo=lo,
                hi=hi, share=share, boot=cb, pooled=False)


def end_frac(r):
    """a behaviour row's gap fraction at the end (mean of the last 3 checkpoints) and its 5-95% band"""
    f = gap.gap_fraction(r["value"], r["naive"], r["bayes"])
    fb = gap.gap_fraction(r["boot"], r["naive_boot"], r["bayes_boot"])
    e = _end(fb)
    return float(_end(f)), np.nanpercentile(e, 5), np.nanpercentile(e, 95), float(np.nanmax(f))


def numbers(S=None):
    S = S or load()
    print("Runs (seed 20261029, the same world, probes and episode stream; 114 checkpoints each; x = training "
          "episodes, the initial network at 10). [5-95%] over the 1,000 shared environment resamples "
          "(td_tt_gap.boot_counts, paired across rows and runs).")
    print("== 1 accuracy (goal at T) and passing naive")
    for run in RUNS:
        r = S[run]["rows"]["acc"]
        ev = r["events"]
        f0, flo, fhi, fmax = end_frac(r)
        print(f"  {NAME[run]:24s} end {_end(r['value']):.3f} (f {f0:+.2f} [{flo:+.2f}, {fhi:+.2f}]); passes naive "
              f"{_fe(ev[0.])}; c10 {_fe(ev[.1])}; c25 {_fe(ev[.25])}; c50 {_fe(ev[.5])}; c75 {_fe(ev[.75])}")
    print("== 2 behaviour rows: crossings of the gap fraction (c25, c50), the end f, c50 / passing naive (paired), "
          "accuracy f at c50")
    for run in RUNS:
        pn = passes_naive(S, run)
        print(f"  -- {NAME[run]}")
        for key in TESTED + EXTRA:
            r = S[run]["rows"][key]
            ev = r["events"]
            f0, flo, fhi, fmax = end_frac(r)
            print(f"    {r['family']:9s} {r['name'][:50]:50s} c10 {_fe(ev[.1])}; c25 {_fe(ev[.25])}; c50 {_fe(ev[.5])}")
            print(f"    {'':9s} {'':50s} end f {f0:+.2f} [{flo:+.2f}, {fhi:+.2f}] (max {fmax:+.2f}); c50 / passes "
                  f"naive {_fr(ratio(ev[.5], pn))}; c25 / passes naive {_fr(ratio(ev[.25], pn))}; accuracy f at c50 "
                  f"{_ff(acc_at(S, run, ev[.5]))}")
    print("== 2b the power-1 run: its holding and its angle of jumps >= 4 cells, against passing naive and against the "
          "Trained RNN at matched accuracy")
    rp, r0 = S["trained_p1"]["rows"], S["trained"]["rows"]
    pn = passes_naive(S, "trained_p1")
    for L in (.1, .25):
        e = rp["emap"]["events"][L]
        rel, sh = order(e, pn)
        print(f"  end-mass slope c{int(L * 100)} {_fe(e)} vs passing naive {_fe(pn)}: {rel} (paired share earlier "
              f"{sh:.3f}); ratio {_fr(ratio(e, pn))} (Trained RNN {_fr(ratio(r0['emap']['events'][L], passes_naive(S, 'trained')))})")
    e5 = rp["emap"]["events"][.5]
    print(f"  end-mass slope c50 {_fe(e5)} at accuracy f {_ff(acc_at(S, 'trained_p1', e5))} (Trained RNN "
          f"{_ff(acc_at(S, 'trained', r0['emap']['events'][.5]))}); vs the split gradient's c50 {_fe(rp['split']['events'][.5])}: "
          f"{order(e5, rp['split']['events'][.5])[0]} (paired share split earlier "
          f"{1 - order(e5, rp['split']['events'][.5])[1]:.3f}); vs goal accuracy's c50 {_fe(rp['acc']['events'][.5])}: "
          f"{order(e5, rp['acc']['events'][.5])[0]}")
    xp = S["trained_p1"]["x"]
    fb4 = gap.gap_fraction(rp["band4"]["boot"], rp["band4"]["naive_boot"], rp["band4"]["bayes_boot"])
    k0 = int(np.flatnonzero(np.isfinite(gap.gap_fraction(rp["band4"]["value"], rp["band4"]["naive"],
                                                         rp["band4"]["bayes"])))[0])
    e = rp["band4"]["events"][.5]
    print(f"  >= 4-cell angle: no value (< {MIN_N} jumps of >= 4 cells) before {xp[k0]:,.0f}; censored at c50: "
          f"{e['censored']}: c50 <= {e['p']:,.0f} ({_fr(ratio(e, pn))} of passing naive, accuracy f "
          f"{_ff(acc_at(S, 'trained_p1', e))}); share of resamples defined at "
          + ", ".join(f"{xp[k]:,.0f}: {np.isfinite(fb4[:, k]).mean():.3f}" for k in range(k0 - 2, k0 + 3)))
    for xv in xp[k0:k0 + 2]:
        q = matched_at(S, "trained_p1", "band4", xv)
        print(f"  at {xv:,.0f}: power-1 accuracy f {q['acc']:+.3f}, >= 4-cell f {q['f']:+.3f}; the Trained RNN passes "
              f"that accuracy at {q['base_x']:,.0f}, where its >= 4-cell f is {q['base_f']:+.3f}; paired difference "
              f"{_ff(q)} over the {q['share']:.1%} of resamples defined")
    print("== 3 family test: share of paired resamples in which every early row (answers, splitting) crosses before "
          "every late row (holding, coupling); tested rows: " + ", ".join(TESTED))
    for run in RUNS:
        print(f"  {NAME[run]:24s} " + "  ".join(f"f = {L}: {family_test(S, run, L):.3f}" for L in (.1, .25, .5, .75)))
        cens = [(k, L) for k in TESTED for L in (.1, .25, .5, .75) if S[run]["rows"][k]["events"][L]["censored"]]
        if cens:
            print(f"  {'':24s} censored rows {sorted(set(k for k, _ in cens))} INCLUDED (crossing = first defined): "
                  + "  ".join(f"f = {L}: {family_test(S, run, L, censored=True):.3f}" for L in (.1, .25, .5, .75)))
    print("== 4 the families as composite events (latest early / earliest late crossing per resample) and their order; "
          "the late window = the earliest to the latest late c50")
    for run in RUNS:
        pn = passes_naive(S, run)
        for L in (.25, .5):
            ea, la = composite(S, run, L, EARLY, "max"), composite(S, run, L, LATE, "min")
            rel, sh = order(ea, la)
            print(f"  {NAME[run]:24s} f = {L}: latest early {_fe(ea)} (x{ea['p'] / pn['p']:.2f} passing naive); "
                  f"earliest late {_fe(la)} (x{la['p'] / pn['p']:.2f}); early {rel} late (paired share early < late "
                  f"{sh:.3f})")
        lz = composite(S, run, .5, LATE, "max")
        print(f"  {NAME[run]:24s} latest late c50 {_fe(lz)} (x{lz['p'] / pn['p']:.2f} passing naive)")
    print("== 5 matched accuracy: each late row's gap fraction when the run's accuracy passes f = L for good, and its "
          "paired difference from the Trained RNN's at the Trained RNN's accuracy c_L (minus = behind it at matched "
          "accuracy)")
    for L in (.1, .25, .5, .75):
        print(f"  -- accuracy c{int(L * 100)}")
        for run in RUNS:
            ea = S[run]["rows"]["acc"]["events"][L]
            if not np.isfinite(ea["p"]):
                print(f"    {NAME[run]:24s} accuracy never past f = {L} ({ea['share']:.0%} of resamples)")
                continue
            parts = []
            for k in TESTED + EXTRA:
                if S[run]["rows"][k]["family"] not in LATE:
                    continue
                a, dlt = matched(S, run, k, L)
                parts.append(f"{S[run]['rows'][k]['name'][:20]} {a['p']:+.2f} (diff {_ff(dlt)})" if run != "trained"
                             else f"{S[run]['rows'][k]['name'][:20]} {_ff(a)}")
            print(f"    {NAME[run]:24s} at {ea['p']:,.0f} ({ea['share']:.0%} of resamples cross; intervals over "
                  f"those): " + "; ".join(parts))
    print("== 5b matched accuracy, neural candidates: the raw value when the run's accuracy passes f = L for good, and "
          "(other runs) its paired difference from the Trained RNN's at the Trained RNN's accuracy c_L")
    for L in (.25, .5, .75):
        for run in RUNS:
            ea = S[run]["rows"]["acc"]["events"][L]
            if not np.isfinite(ea["p"]):
                continue
            e0 = S["trained"]["rows"]["acc"]["events"][L]
            parts = []
            for k in ("coh", "r_coh", "bxf", "r_norm", "tau", "mem1", "mem5", "effdim"):
                if k not in S[run]["rows"]:
                    continue
                a = value_at(S, run, k, ea)
                if run == "trained":
                    parts.append(f"{S[run]['rows'][k]['name'][:24]} {_ff(a, 3)}")
                    continue
                dv = a["boot"] - value_at(S, "trained", k, e0)["boot"]
                ok = np.isfinite(dv)
                lo, hi = np.percentile(dv[ok], [5, 95]) if ok.sum() > 50 else (np.nan, np.nan)
                parts.append(f"{S[run]['rows'][k]['name'][:24]} {a['p']:+.3f} (diff {a['p'] - value_at(S, 'trained', k, e0)['p']:+.3f} "
                             f"[{lo:+.3f}, {hi:+.3f}])")
            print(f"  c{int(L * 100)} {NAME[run]:24s} at {ea['p']:,.0f}: " + "; ".join(parts))
    print("== 6 neural rows: events, / passing naive (paired), accuracy f at the event, order (interval rule; paired "
          "share earlier) vs passing naive, the latest early c50, the earliest late c50 and the latest late c50 (the late "
          "window: earliest to latest late c50). Pooled rows: their point against the other event's interval (no "
          "interval on their side)")
    for run in RUNS:
        pn = passes_naive(S, run)
        ea, la = composite(S, run, .5, EARLY, "max"), composite(S, run, .5, LATE, "min")
        lz = composite(S, run, .5, LATE, "max")
        print(f"  -- {NAME[run]} (passes naive {_fe(pn)}; latest early c50 {_fe(ea)}; earliest late c50 {_fe(la)}; "
              f"latest late c50 {_fe(lz)})")
        for key, name, fam, need in NEU:
            if key not in S[run]["rows"]:
                print(f"    {name[:46]:46s} no keys ({need})")
                continue
            r = S[run]["rows"][key]
            for lab, e in r["events"].items():
                rel = [order(e, q) for q in (pn, ea, la, lz)]
                print(f"    {name[:46]:46s} {lab[:40]:40s} {_fe(e)}; / passes naive {_fr(ratio(e, pn))}; accuracy f "
                      f"{_ff(acc_at(S, run, e))}; vs passes naive: {rel[0][0]} ({rel[0][1]:.3f}), early c50: "
                      f"{rel[1][0]} ({rel[1][1]:.3f}), late c50: {rel[2][0]} ({rel[2][1]:.3f}), latest late c50: "
                      f"{rel[3][0]} ({rel[3][1]:.3f}){' [pooled]' if e['pooled'] else ''}")
    print("== 6b accuracy f at each neural event: paired difference from the Trained RNN's (same environment resamples; "
          "+ = the run reaches the event at a higher accuracy)")
    for key, name, fam, need in NEU:
        r0 = S["trained"]["rows"].get(key)
        if r0 is None:
            continue
        for lab, e0 in r0["events"].items():
            a0 = acc_at(S, "trained", e0)
            parts = []
            for run in TRAINED[1:]:
                e = _ev(S, run, key, lab)
                if e is None:
                    parts.append(f"{SHORT[run]} no keys")
                    continue
                a = acc_at(S, run, e)
                dv = a["boot"] - a0["boot"]
                ok = np.isfinite(dv)
                if ok.sum() < 50:
                    parts.append(f"{SHORT[run]} -")
                    continue
                lo, hi = np.percentile(dv[ok], [5, 95])
                parts.append(f"{SHORT[run]} {a['p'] - a0['p']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            print(f"  {name[:40]:40s} {lab[:38]:38s} N1000 at f {a0['p']:+.2f}: " + "; ".join(parts))
    print("== 6b2 the late markers (" + ", ".join(f"{k} {lab}" for k, lab in LATE_MARKERS) + "): accuracy f at the "
          "event and the event / passing naive, per run")
    for run in TRAINED:
        parts = []
        for k, lab in LATE_MARKERS:
            e = _ev(S, run, k, lab)
            if e is not None and np.isfinite(e["p"]):
                parts.append(f"{k} f {acc_at(S, run, e)['p']:+.2f} x{ratio(e, passes_naive(S, run))['p']:.2f}")
        print(f"  {NAME[run]:24s} " + ("; ".join(parts) or "none crosses"))
    print("  own-clock place against N1000's: paired ratio of ratios (event / passing naive) / (N1000's event / N1000's "
          "passing naive) per resample [5-95%] (x1 = the same place on the run's own clock; [pooled]: only passing "
          "naive is resampled)")
    for run in TRAINED[1:]:
        parts = []
        for k, lab in LATE_MARKERS:
            e, e0 = _ev(S, run, k, lab), _ev(S, "trained", k, lab)
            if e is None or not (np.isfinite(e["p"]) and np.isfinite(e0["p"])):
                continue
            q = ratio_of_ratios(S, run, k, lab)
            parts.append(f"{k} {_fr(q)}" + (" [pooled]" if e["pooled"] else "")
                         + ("" if q["share"] >= .995 else f" ({q['share']:.0%} of resamples)"))
        print(f"  {NAME[run]:24s} " + ("; ".join(parts) or "none crosses"))
    print("== 6b3 the long-memory share (pooled) against each late row's c50 (its point vs the row's interval)")
    for run in TRAINED:
        for lab in ("> 3% of units", "50% of its own change"):
            e = _ev(S, run, "tau", lab)
            if e is None or not np.isfinite(e["p"]):
                print(f"  {NAME[run]:24s} {lab:22s} {'never' if e is not None else 'no own change'}")
                continue
            parts = []
            for k in TESTED:
                if S[run]["rows"][k]["family"] not in LATE:
                    continue
                q = S[run]["rows"][k]["events"][.5]
                rel, sh = order(e, q)
                parts.append(f"{k} {_fe(q)}{' (censored: an upper bound)' if q['censored'] else ''}: {rel} ({sh:.3f})")
            print(f"  {NAME[run]:24s} {lab:22s} {e['p']:,.0f}: " + "; ".join(parts))
    print("== 6c what the memory-benefit event means: goal accuracy at T intact and with the memory cut to 1 step at the "
          "event (log-interpolated, full sample); probes: naive .464, step-wise Bayes (Bayes cut to 1 step) .608")
    for run in TRAINED:
        d = build(run)
        x = np.log(S[run]["x"])
        tr = d["trunc_env"].astype(np.float64).mean(1)                                # (K, 4) env means
        for lab in ("half of Bayes's level", "Bayes's level"):
            e = _ev(S, run, "mem1", lab)
            v = [float(np.interp(np.log(e["p"]), x, tr[:, j])) for j in (3, 0)]
            print(f"  {NAME[run]:24s} {lab:22s} at {e['p']:,.0f}: intact {v[0]:.3f}, cut to 1 step {v[1]:.3f}")
    print("== 6d sizes: the memory benefits' peaks and ends (full sample; end = mean of the last 3 checkpoints), the "
          "effective dimension's initial and end values")
    for run in RUNS:
        rows, x = S[run]["rows"], S[run]["x"]
        parts = []
        for k in ("mem1", "mem5"):
            v = rows[k]["value"]
            parts.append(f"{k} peak {np.nanmax(v):.3f} at {x[int(np.nanargmax(v))]:,.0f}, end {_end(v):.3f}")
        v = rows["effdim"]["value"]
        print(f"  {NAME[run]:24s} " + "; ".join(parts) + f"; effective dimension initial {v[0]:.1f}, end {_end(v):.1f}")
    cn = references()["counts"]
    print(f"  probe-based held shares: {cn['n_two_v'].sum():,.0f} corner-split / {cn['n_one_v'].sum():,.0f} one-variable "
          f"var-steps")
    print("== 7 natural experiment: the late candidates' size per run (end = mean of the last 3 checkpoints)")
    for run in RUNS:
        d = S[run]["rows"]
        x = S[run]["x"]
        tau = d["tau"]["value"]
        nz = np.flatnonzero(tau > 0)
        runs_ = np.split(nz, np.flatnonzero(np.diff(nz) > 1) + 1) if len(nz) else []
        spans = ", ".join(f"{x[g[0]]:,.0f}-{x[g[-1]]:,.0f} ({len(g)})" for g in runs_) or "none"
        s = (f"  {NAME[run]:24s} tau > 5 share: end {_end(tau):.3f}, max after 160 episodes {tau[x > 160].max():.3f}, "
             f"max at <= 160 {tau[x <= 160].max():.3f}; any unit at checkpoints {spans}")
        if "coh" in d:
            s += (f"; coherence median end {_end(d['coh']['value']):.2f} (min {np.nanmin(d['coh']['value']):.2f}); "
                  f"r(coherence, ent) end {_end(d['r_coh']['value']):+.3f}; cross-band (final bands) end "
                  f"{_end(d['bxf']['value']):+.3f}")
        s += f"; r(|u|, ent) end {_end(d['r_norm']['value']):+.3f}"
        print(s)
        late = [f"{d[k]['name'][:24]} {end_frac(d[k])[0]:+.2f} (max {end_frac(d[k])[3]:+.2f})"
                for k in TESTED + EXTRA if d[k]["family"] in LATE]
        print(f"  {'':24s} late rows' end f: " + "; ".join(late))
    return S



# ═══════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════

COLR = {"trained": "#1f77b4", "trained_p1": "#4d004b", "trained_N200": "#17becf", "trained_N100": "#00441b",
        "trained_N50": "#8c564b", "echo": "#ff7f0e"}           # CVD-checked (OKLab dE >= 11 under all three CVDs)
MARK = {"trained": "o", "trained_p1": "D", "trained_N200": "s", "trained_N100": "^", "trained_N50": "v", "echo": "X"}
TRAINED = ("trained", "trained_p1", "trained_N200", "trained_N100", "trained_N50")
LATE_MARKERS = (("coh", "below 1 (pieces cancel)"), ("tau", "> 3% of units"), ("r_norm", "r > .1"),
                ("bxf", "below -.1"), ("r_coh", "r > .1"))                 # the late neural candidates' events
# the crossing chart: (group, [(row key, event key, label)]); rows ordered by the Trained RNN's crossing
CHART = (
    ("neural, early candidates", (
        ("r_dim", "half-way from its initial value to 0", "r(dW dimension, ent.): half-way to 0"),
        ("mem1", "Bayes's level", "memory benefit (cut to 1 step) = Bayes's"),
        ("rho", "> 1 (expansive)", r"$W_{hh}$ radius $>$ 1 (i, f, o blocks)"),
        ("effdim", "50% of its own change", "eff. dimension of u: half its change"))),
    ("answers, splitting: half-way to Bayes", (
        ("split", .5, "split gradient"),
        ("far", .5, "far errors at T"),
        ("coll", .5, "distant pairs merged at T"),
        ("acc", .5, "goal accuracy at T"))),
    ("holding, coupling: half-way to Bayes", (
        ("band4", .5, r"angle of jumps $\geq$ 4 cells"),
        ("emap", .5, "end-mass slope (holding)"),
        ("G", .5, "long-path coupling G"),
        ("excess", .5, "jump excess angle"),
        ("dmap", .5, "direction slope"))),
    ("neural, late candidates", (
        ("coh", "below 1 (pieces cancel)", r"credit coherence $<$ 1"),
        ("tau", "> 3% of units", r"$>$ 3% of units with $\tau > 5$"),
        ("tau", "50% of its own change", r"units with $\tau > 5$: half their own change"),
        ("r_norm", "r > .1", r"r($|u|$, entanglement) $>$ 0.1"),
        ("bxf", "below -.1", r"cross-band share $<$ $-$0.1"),
        ("r_coh", "r > .1", r"r(coherence, entanglement) $>$ 0.1"),
        ("mem5", "relaxation: half-way back from its peak", "memory benefit (cut to 5): relaxed"))))


def _ev(S, run, key, lab):
    rows = S[run]["rows"]
    if key not in rows:
        return None
    return rows[key]["events"].get(lab)


def _yval(S, run, key):
    """the plotted curve of a row: the gap fraction (behaviour), the memory benefit over Bayes's own, or the raw value"""
    r = S[run]["rows"][key]
    if r["kind"] == "behaviour":
        return gap.gap_fraction(r["value"], r["naive"], r["bayes"])
    if key in ("mem1", "mem1b", "mem5"):
        return r["value"] / r["ref"]
    return r["value"]


def figure(S=None):
    """-> figure_bin/td_tt_replication.png"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.transforms import offset_copy
    from main.utils import fig_path, apply_fig_style
    S = S or load()
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    FL, FK, TS = fs["label"], fs["tick"], 12.
    W, H = 19.5, 10.
    fig = plt.figure(figsize=(W, H))
    PW, PH, L0, GX = 2.6, 2.3, .8, .8
    ys = [H - 2.3 - PH, H - 2.3 - PH - 1.4 - PH]
    pos = [(L0 + c * (PW + GX), ys[r]) for r in range(2) for c in range(3)]
    axs = [fig.add_axes([x0 / W, y0 / H, PW / W, PH / H]) for x0, y0 in pos]
    f100 = float(np.nanmax(acc_frac(S, "trained_N100")[0]))           # the best accuracy N = 100 reaches
    MS = 7.

    def place(ax, pts):
        """draw the runs' event markers [(run, accuracy f, level)] at their level (after the axes' limits are set); a
        marker that would overlap one already drawn moves up or down by whole marker heights, so none hides another"""
        s = 1.1 * MS * fig.dpi / 72
        done = []
        for run, xv, yv in pts:
            X, Y = ax.transData.transform((xv, yv))
            k = next(k for k in (0, 1, -1, 2, -2, 3, -3)                  # (.99: one height apart is no overlap)
                     if all(abs(X - a) >= .99 * s or abs(Y + k * s - b) >= .99 * s for a, b in done))
            done.append((X, Y + k * s))
            ax.plot(xv, yv, marker=MARK[run], color=COLR[run], ms=MS, mec="w", mew=.7, zorder=6, ls="none",
                    transform=offset_copy(ax.transData, fig=fig, y=k * 1.1 * MS, units="points"))

    # a: the pace
    ax = axs[0]
    for run in RUNS:
        f, _ = acc_frac(S, run)
        ax.plot(S[run]["x"], f, color=COLR[run], lw=1.6, zorder=3)
        pn = passes_naive(S, run)
        if np.isfinite(pn["p"]):
            ax.plot([pn["lo"], pn["hi"]], [0, 0], color=COLR[run], lw=3.2, solid_capstyle="butt", zorder=4)
            ax.plot(pn["p"], 0, marker=MARK[run], color=COLR[run], mec="w", mew=.6, ms=7.5, zorder=5, ls="none")
    ax.axhline(0, color="0.6", lw=1, ls="--", zorder=1)
    ax.axhline(1, color="k", lw=1, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(8, 5e4)
    ax.set_xticks([10, 100, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.minorticks_off()
    ax.set_ylim(-1.3, 1.12)
    ax.set_xlabel("training episodes", fontsize=FL)
    ax.set_ylabel(r"$f$  (0 = naive, 1 = Bayes)", fontsize=FL)
    ax.set_title("a   Goal accuracy at T: the pace\n"
                 r"$f = (y - y_{\mathrm{naive}})\,/\,(y_{\mathrm{Bayes}} - y_{\mathrm{naive}})$",
                 loc="left", fontsize=TS, linespacing=1.5)
    ax.text(.03, .8, "markers: passes naive\nfor good, bars 5-95%", transform=ax.transAxes, fontsize=FK,
            va="top", ha="left", linespacing=1.3)
    # b-f: matched accuracy
    spec = (
        (1, "mem1", "b   Memory benefit over Bayes's own",
         r"$(y - y_{\mathrm{cut}})\,/\,(y_{\mathrm{B}} - y_{\mathrm{B,cut}})$, cut to 1 step",
         (1., "Bayes's own (0.163)", .98, "top", "right"), "Bayes's level", 1., (-.15, 3.3), "benefit / Bayes's",
         "markers: reaches Bayes's own", (.03, .97, "top")),
        (2, "excess", "c   Coupling: jump excess angle",
         r"$f$ of $\bar\phi - \bar\phi_{\mathrm{indep}}$", None, .5, .5, (-.35, 1.12), r"$f$",
         "markers: half-way for good", (.03, .97, "top")),
        (3, "emap", "d   Holding: end-mass slope",
         r"$f$ of $\beta = \Sigma_t\,\mathrm{cov}_t(e_N, e_B)\,/\,\Sigma_t\,\mathrm{var}_t(e_B)$", None, .5, .5,
         (-.35, 1.12), r"$f$", "markers: half-way for good", (.03, .97, "top")),
        (4, "coh", "e   Credit coherence (median)",
         r"$\|\Sigma_k \delta_k\|^2\,/\,\Sigma_k \|\delta_k\|^2$  (log axis)",
         (1., "1: lag pieces neither add\nnor cancel", .03, "top", "left"), "below 1 (pieces cancel)", 1., (.3, 12.),
         "coherence", "markers: below 1 for good", (.03, .03, "bottom")),
        (5, "tau", r"f   Units with long memory, $\tau > 5$",
         r"share of units, $\tau_j = 1\,/\,(1 - \bar g^{\,\mathrm{forget}}_j)$",
         (.03, "3% of units", .55, "bottom", "center"),
         "> 3% of units", .03, (-.004, .13), "share of units", "markers: above 3% for good", (.03, .97, "top")))
    for j, key, ttl, frm, ref, evk, yev, ylim, ylab, mtxt, mpos in spec:
        ax = axs[j]
        if j >= 2:
            ax.axvspan(f100, 1.03, color="#efe9da", lw=0, zorder=0)
        pts = []
        for run in RUNS:
            if key not in S[run]["rows"]:
                continue
            fa, _ = acc_frac(S, run)
            ax.plot(fa, _yval(S, run, key), color=COLR[run], lw=1.4, zorder=3)
            e = _ev(S, run, key, evk)
            if e is not None and np.isfinite(e["p"]):
                pts.append((run, acc_at(S, run, e)["p"], yev))
        ax.axvline(0, color="0.6", lw=1, ls="--", zorder=1)
        if S["trained"]["rows"][key]["kind"] == "behaviour":
            ax.axhline(0, color="0.6", lw=1, ls="--", zorder=1)
            ax.axhline(1, color="k", lw=1, ls="--", zorder=1)
        if ref is not None:
            ax.axhline(ref[0], color="k", lw=1, ls="--" if key == "mem1" else ":", zorder=1)
            ax.text(ref[2], ref[0] * (.93 if key == "coh" else 1.) + (.0015 if key == "tau" else 0.), ref[1],
                    fontsize=FK, va=ref[3], ha=ref[4], transform=ax.get_yaxis_transform(), zorder=7,
                    linespacing=1.2)
        if key == "coh":
            ax.set_yscale("log")
            ax.set_yticks([.5, 1, 2, 5, 10])
            ax.set_yticklabels(["0.5", "1", "2", "5", "10"])
            ax.minorticks_off()
        ax.set_xlim(-1.25, 1.03)
        ax.set_ylim(*ylim)
        place(ax, pts)
        ax.set_xlabel(r"goal accuracy at T, $f$ (matched accuracy)", fontsize=FL)
        ax.set_ylabel(ylab, fontsize=FL)
        ax.set_title(f"{ttl}\n{frm}", loc="left", fontsize=TS, linespacing=1.5)
        ax.text(mpos[0], mpos[1], mtxt, transform=ax.transAxes, fontsize=FK, va=mpos[2], ha="left", zorder=7)
    axs[2].text((f100 + 1.03) / 2, -.3, "beyond\nN = 100's\nbest", fontsize=FK, ha="center", va="bottom",
                color="0.25", linespacing=1.15, zorder=7)
    # g: crossings on each run's own clock
    LAB = L0 + 3 * PW + 2 * GX + 3.6
    rows_y, y, gy = [], 0., []
    for gi, (gname, items) in enumerate(CHART):
        y += 1. if gi else 0.
        gy.append((gname, y))
        for key, lab, label in items:
            rows_y.append((key, lab, label, y))
            y += 1.
    ytop, ybot = -.95, y - .4
    CB, CT = ys[1], H - 1.95
    NW = .95
    axc = fig.add_axes([LAB / W, CB / H, (W - LAB - NW - .3) / W, (CT - CB) / H])
    axn = fig.add_axes([(W - NW - .25) / W, CB / H, NW / W, (CT - CB) / H], sharey=axc)
    XL, XR = .004, 30.
    offs = {run: o for run, o in zip(TRAINED, np.linspace(-.34, .34, len(TRAINED)))}
    nx = {run: v for run, v in zip(TRAINED, np.linspace(.14, .86, len(TRAINED)))}
    for i, (key, lab, label, yy) in enumerate(rows_y):
        if i % 2 == 0:
            for a_ in (axc, axn):
                a_.axhspan(yy - .5, yy + .5, color="#f4f1e8", lw=0, zorder=0)
        axc.text(-.012, yy, label, transform=axc.get_yaxis_transform(), ha="right", va="center", fontsize=FL)
        for run in TRAINED:
            e = _ev(S, run, key, lab)
            if e is None:
                continue
            q = ratio(e, passes_naive(S, run))
            yo = yy + offs[run]
            if np.isfinite(q["p"]):
                hi = q["hi"] if np.isfinite(q["hi"]) else XR
                axc.plot([max(q["lo"], XL), min(hi, XR)], [yo, yo], color=COLR[run], lw=1.2, zorder=3,
                         solid_capstyle="butt")
                cen = e.get("censored", False)
                axc.plot(q["p"], yo, marker=MARK[run], mfc="w" if cen else COLR[run], mec=COLR[run] if cen else "w",
                         mew=1.1 if cen else .5, ms=5.5, zorder=4, ls="none")
            else:
                axn.plot(nx[run], yy, marker=MARK[run], mfc="w", mec=COLR[run], mew=1.1, ms=5.5, zorder=4, ls="none")
    for gname, yy in gy:
        axc.text(-.012, yy - .85, gname, transform=axc.get_yaxis_transform(), ha="right", va="bottom",
                 fontsize=FL + 1.5)
    axc.axvline(1, color="k", lw=1.2, ls="--", zorder=2)
    axc.text(1.08, ytop + .1, "passes naive", fontsize=FK, va="top", ha="left")
    axc.set_xscale("log")
    axc.set_xlim(XL, XR)
    axc.set_ylim(ybot, ytop)
    axc.set_yticks([])
    tk = [.01, .03, .1, .3, 1, 3, 10, 30]
    axc.set_xticks(tk)
    axc.set_xticklabels([f"{t:g}" for t in tk])
    axc.minorticks_off()
    axc.set_xlabel("crossing / the run's own passing-naive crossing (paired ratio, log axis)", fontsize=FL)
    axc.set_title("g   Each change on its run's own clock: crossing / passing naive [5-95%]",
                  loc="left", fontsize=TS)
    axn.set_xlim(0, 1)
    axn.set_xticks([.5])
    axn.set_xticklabels(["never"], fontsize=FL)
    axn.tick_params(left=False, labelleft=False)
    h = [Line2D([], [], color=COLR[r], marker=MARK[r], ms=7, mec="w", lw=1.5, label=NAME[r]) for r in RUNS]
    h += [Line2D([], [], color="0.6", ls="--", lw=1, label="naive"), Line2D([], [], color="k", ls="--", lw=1,
                                                                             label="Bayes")]
    fig.legend(handles=h, loc="center", bbox_to_anchor=(.5, (H - 1.49) / H), ncol=8, frameon=False, fontsize=FL + 1,
               columnspacing=1.8, handlelength=2.3)
    # the title states what the panels show (numbers from the data; the order claims are asserted)
    for r, L in (("trained", .5), ("trained_N200", .5), ("trained_N100", .25)):
        assert order(composite(S, r, L, EARLY, "max"), composite(S, r, L, LATE, "min"))[0] == "before", (r, L)
    assert order(S["trained_p1"]["rows"]["emap"]["events"][.25], passes_naive(S, "trained_p1"))[0] == "not ordered"
    m1 = [ratio(_ev(S, r, "mem1", "Bayes's level"), passes_naive(S, r))["p"] for r in TRAINED]
    ch = [acc_at(S, r, _ev(S, r, "coh", "below 1 (pieces cancel)"))["p"] for r in ("trained", "trained_N200")]
    la_f, la_x = [], []                         # the late markers in N1000, p1, N200, the cross-band share apart
    for r in ("trained", "trained_p1", "trained_N200"):
        for k, lab in LATE_MARKERS:
            e = _ev(S, r, k, lab)
            if k != "bxf" and e is not None and np.isfinite(e["p"]):
                la_f.append(acc_at(S, r, e)["p"])
                la_x.append(ratio(e, passes_naive(S, r))["p"])
    BX = ("trained", "trained_N200", "trained_N100")     # the cross-band share: same own-clock place, other accuracy
    bx_x = [ratio(_ev(S, r, "bxf", "below -.1"), passes_naive(S, r))["p"] for r in BX]
    bx_f = [acc_at(S, r, _ev(S, r, "bxf", "below -.1"))["p"] for r in BX]
    for r in BX[1:]:
        q = ratio_of_ratios(S, r, "bxf", "below -.1")
        assert q["lo"] < 1 < q["hi"], (r, q)
    e0 = [S["trained"]["rows"]["acc"]["events"][L] for L in (.25, .5)]
    tw = [value_at(S, "trained", "tau", e)["p"] for e in e0]                  # first-wave share at accuracy c25, c50
    cw = [value_at(S, "trained", "coh", e)["p"] for e in e0]
    assert min(cw) > 1
    rest = (f"In all five trained runs, cutting memory to one step (stored start-state cut) costs as much as it costs "
            f"Bayes (0.163) before the run passes naive (x{min(m1):.2f}-{max(m1):.2f} of its passing time; b, g): "
            f"memory carries the early answers. Late neural markers other than the cross-band share come at similar "
            f"accuracies (f = {min(la_f):.2f}-{max(la_f):.2f}) but at x{min(la_x):.2f}-{max(la_x):.1f} of passing "
            f"naive; the cross-band share is the reverse (x{bx_x[0]:.2f}, {bx_x[1]:.2f}, {bx_x[2]:.2f} of passing "
            f"naive at f = {bx_f[0]:.2f}, {bx_f[1]:.2f}, {bx_f[2]:.2f} in N = 1000, 200, 100; g). Across widths only "
            f"credit cancellation is accuracy-matched (f = {ch[0]:.2f}, {ch[1]:.2f}; e). N = 100 and N = 50 form "
            f"neither credit cancellation nor long-memory units, yet hold and couple about level with N = 1000 at "
            f"matched accuracy (c, d; the large-jump angle and N = 50's holding lag slightly); there N = 1000 has no "
            f"cancellation either (coherence {min(cw):.2f}-{max(cw):.2f}) and only first-wave long-memory units "
            f"({max(tw):.0%}; panel f), so only these are tested.")
    fig.suptitle(
        "Width sets the pace, not the order (power-0.5 runs: N = 1000 and N = 200 at f = 0.5, N = 100 at f = 0.25; g); "
        "the power-1 loss moves holding (d) and the large-jump angle (g) forward.\n"
        + textwrap.fill(rest.replace("N = ", "N\0=\0"), 230).replace("\0", " "),         # 'N = 1000' stays on one line
        x=.012, y=1 - .12 / H, ha="left", va="top", fontsize=TS + 1, linespacing=1.45)
    note = (
        "Data (seed 20261029; every run on the same world, probes and episode stream): 10,000 probe episodes per "
        "checkpoint (50 test environments x 100 latent pairs x 2 trajectories, 30 steps), 114 checkpoints per run; "
        "credit keys (e, coherence, cross-band, r(coherence)): 5,000 probes x 14 error steps\n(cross-band on the final "
        r"bands: 2,000), not recorded for the power-1 run; maps (d, direction slope): 580,000 variable-steps / 4,823 of "
        r"Bayes's moves of $\geq$ 2 cells. Intervals: 5-95% over 1,000 resamples of the 50 environments, one matrix for "
        "every row and run (paired). Crossings stay past their level to the end of training.\n"
        r"g: open marker = censored, an upper bound (the power-1 run has $<$ 50 jumps of $\geq$ 4 cells before 2,320 and "
        r"is past half-way at its first value); pooled rows (radius; units with $\tau > 5$, from forget gates averaged "
        "over probes and steps) have no environment interval (their bars are passing naive's);\n"
        r"N = 100 and N = 50 have no units with $\tau > 5$ at the end (no own change); the ESN never passes naive (no "
        "own clock, so it is not drawn in g) and reaches no behaviour row half-way.\nb-f: each curve is one run through "
        "its 114 checkpoints against its own goal accuracy; a marker that would hide another is shifted up or down by "
        r"one marker height; the bumps at $f < -0.7$ are the first 20-160 training episodes; shaded: accuracy beyond "
        "N = 100's best.")
    fig.text(.012, .008, note, ha="left", va="bottom", fontsize=FK, linespacing=1.45)
    path = fig_path("td_tt_replication.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    what = sys.argv[1:] or ["synthesis", "numbers", "figure"]
    if what[0] == "data":
        for run in what[1:] or RUNS:
            build(run)
        sys.exit()
    if "synthesis" in what:
        synthesis()
    if "check" in what:
        check()
    S_ = load() if {"numbers", "figure"} & set(what) else None
    if "numbers" in what:
        numbers(S_)
    if "figure" in what:
        figure(S_)
