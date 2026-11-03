"""Splitting and unfolding through training (John 11-01, while away: "further 'through training' analyses ... wormholes,
jumps, couplings, splitting, unfolding ..."; fixed the same night after an independent verification, and again (11-02)
after an independent re-verification: heat-map line, ESN statements, one ordering, labels; synthesis export; and after
a second re-check: the heat-map line's precision, one ordering, two labels). The end-of-training findings followed
through the 114 checkpoints of the seed-20261029 runs (trained, echo):
  (1) near pairs of particles from DIFFERENT cells split CORRECTLY after entangled steps for the joint agents
      (plot_fig1c_splitting.py: one-step correct split .12 -> .35 grid units from entanglement decile 1 -> 10; naive /
      ESN split apart without moving toward their own cells);
  (2) the estimate-distance vs true-distance curves FOLD BACK for naive / ESN (opposite corners ~4.5 apart vs true 12.7,
      two-variable separations collapse, one-variable ones kept; plot_fig1c_distances.py) and reach y = x for Bayes /
      the Trained RNN.
-> main/REPLICATION/figure_bin/td_tt_splitting.png (splitting), td_tt_splitting_foldback.png (fold-back, and in panel e
   the timing of every half-way point)
Caches: main/DATA/experimental/training_dynamics/through_training/splitting/ (probes_trained.npz, probes_echo.npz,
probes_refs.npz: per checkpoint and environment; grids_check.npz: the density check; grid_folds.npz: the grid folds per
environment; timeline.npz: the headline measures for the cross-topic synthesis; synthesis.npz: the td_tt_gap export on
the shared environment resamples, see "Synthesis export").

DATA. The probe set (analysis/probes.npz): 50 environments (test variable pairs) x 100 cells (r0, r1) x 2 streams =
10,000 episodes of 30 steps; the networks' marginal log beliefs logq at all 114 checkpoints (analysis/<run>/ck_*.npz),
Bayes's and naive's exact log marginals (log_jb, log_nb), goal correctness at T (ck right; right_bayes / right_naive).
Position of a particle = its posterior means x = (E[r0], E[r1]) (raw variable order; the goal does not enter the
network's input, so both streams of a cell are alike). Every measure is computed within an environment and then averaged
over the 50 environments (never across environments first). x axis: training episodes = E - E // 5 (every 5th episode
is a test episode), the initial network drawn at 10. Density check and grid folds: td_grids_fwd's fresh streams
(summary_data/grids/: the same 50 environments x 100 cells x 24 streams, 19 checkpoints; positions stored as float16
for the networks, float32 for Bayes / naive).

DEFINITIONS
(1) Splitting (plot_fig1c_splitting.py's definitions, with ties handled as below): at each step t0 = 2..24, every
  particle i is paired with its nearest particle(s) j of ANOTHER cell in the same environment (in that agent's own
  positions at t0; each observer finds its own pairs). Ties: Bayes's and naive's positions depend only on the
  observation counts and a network's only on the observation sequence, so equidistant nearest partners are common at
  early steps. Every partner within 1e-9 grid units of the nearest distance is kept, at weight 1 / n_tied (each particle
  weighs 1); ties in the data are exact (distance gap 0; no non-zero gap below 1e-9 in 8 agents x 230,000
  particle-steps). Tenths and thirds are by weighted rank (each row placed by the midpoint of its weight on the
  cumulative weight axis); means per tenth are weighted. c = the true cells.
    d = ||x_i - x_j||                                                     (the pair's separation)
    D = ||x_i - c_j|| - ||x_i - c_i|| + ||x_j - c_i|| - ||x_j - c_j||    (how much each is nearer its own cell)
    one-step correct split dD = D(t0 + 1) - D(t0); apart dd = d(t0 + 1) - d(t0); six steps on: the same at t0 + 6
  The pair's entanglement e_bar = (rev_i + rev_j) / 2 of step t0 + 1, rev = the MEAN of the two variables' entanglement
  (step_dkl; the pair's split is a joint 2-D quantity, so the whole-belief rev, labelled on the figure). Tenths and
  thirds of e_bar within (environment, step); per environment the mean per tenth over t0, then over environments.
  Entanglement gradient = tenth 10 - tenth 1 (per environment, then averaged). Controls: null = e_bar shuffled within
  (environment, step); matched = tenths within thirds of the pair's distance to its cells g = (||x_i - c_i|| + ||x_j -
  c_j||) / 2 at t0, averaged over the thirds (round7: always match distance to truth); density = the same on the
  24-stream set, all 24 streams per cell vs 2 of them.
  Heat-map line (d, e): per environment, the tenth where ITS OWN profile (mean over t0 of the tenth means; 3-tenth
  running mean, 2 tenths at the ends) peaks; then the mean over the 50 environments (the statistic per environment,
  averaged after), with its 5-95% band over the shared resamples (td_tt_gap.boot_counts(): count-weighted means of the
  per-environment peaks). Bayes's and naive's marks (B, n columns) the same way. Drawn where the beliefs split at all:
  mean one-step dD of all near pairs >= .005 grid units per step (c's x axis; this leaves out the initial network,
  .0004, and the ESN's 20 and 40, <= .0037, where each environment's argmax is that of noise). The drawn statistic is
  coarse: paired with Bayes by environment, its SE is .18-.45 tenths from 880. Two more precise versions, reported but
  not drawn (each a per-environment statistic, averaged after): the dD-weighted mean tenth, sum_k k max(dD_k, 0) /
  sum_k max(dD_k, 0) (paired SE .02-.06; a compressed range, since every tenth splits and the weighted mean stays near
  the middle), and the peak tenth per (environment, step) (the same running-mean argmax on each step t0's own profile),
  averaged over the 23 steps (paired SE .07-.15; compressed too, since single-step argmaxes spread). They resolve what
  the drawn line cannot (NUMBERS d, e).
(2) Fold-back at T = 30: for every pair of particles of an environment (19,900), the estimate distance ||x_i - x_j||,
  averaged per separation (|dr0|, |dr1|) of their true cells within the environment (fold tables, all 30 steps cached),
  then over environments. Opposite corners = (9, 9) (true 12.73; 8 particle pairs per environment); opposite edges =
  (9, 0) and (0, 9) (true 9; 80). Fold index = corners - edges (true +3.73; uniform shrinkage keeps it > 0). The curves
  in foldback a, b: cells differing in both latents, in 1-unit bins of true distance (pair-weighted; the distinct
  distances alone give jagged curves because one distance mixes separations of different shapes). Cell-mean variant
  (the distance between the two cells' mean positions, a statistic of averages) cached as fold_cm. Grid folds
  (foldback d, td_summary_data.grids()): each cell's mean position over its 24 fresh streams, the grid of these 100
  points cut into 162 triangles; a triangle is folded where its signed area A < 0 (its orientation flips); % folded at T
  among naive's folded triangles (24% of all) and among the others, pooled over environments.
(3) Timing. Intervals: 5-95% over 1,000 resamples of the 50 environments (this module's resamples, seed 20261101: the
  crossings and the bands of a, the verified numbers; the heat-map band of d, e and the synthesis export use the shared
  resamples of td_tt_gap, seed 20261103); Bayes / naive are resampled with the same environments (paired). A crossing =
  the training episode (log-interpolated between checkpoints) after which the curve stays past the level to the end of
  training (NaN = never; 'first reaches' where stated); the level is the naive-Bayes midpoint unless stated, 'past' = on
  Bayes's side. Order: two events are ordered only when their 5-95% intervals are disjoint (touching or overlapping
  intervals: not ordered at this resolution); paired ratios of two crossing times on the same resamples are given as
  supplementary numbers, never as the ordering rule. Over-splitting plateau = the span over which the 3-checkpoint
  running mean of the six-step apart contrast stays >= 80% of its maximum (first reaches .. after which it stays below).
  Landmarks (quoted, not re-derived): passes naive = the first checkpoint whose goal accuracy at T beats naive's (.441
  at 800, .493 at 880; naive .464, Bayes .771): 880, the dotted line (the log-interpolated crossing, 835, and its
  interval are also given; foldback e notes the difference); memory-use flip = td_flip_scan.flips on
  analysis/flip_scan_paired.npz (memory use by own entanglement, across): 800-1,040, the grey band (no environment
  bootstrap here); grid unfolding = 25 / 50 / 75% of the drop of the % folded where naive folds, from the initial
  network to the end (mean of the last 3 checkpoints), first covered: 551 / 770 / 1,262 (the landmark's stays-past rule,
  td_tt_wormholes.unfolding, gives the same points: 551 [394, 649] / 770 [703, 860] / 1,262 [1,097, 1,523]); the
  interval from grid_folds.npz (td_summary_data's functions per environment).

NUMBERS (training episodes; Trained RNN = "T", ESN = "E"; [a, b] = 5-95% environment-bootstrap interval)
Ties (particles with >= 2 equidistant nearest partners), probes t0 = 2 / 3 / 4: Bayes .615 / .333 / .187, naive .618 /
  .335 / .188, T end .386 / .111 / .033, E end .397 / .110 / .035 (all t0: .063, .063, .024, .024); 24 streams: Bayes
  .978 / .868 / .698, naive .978 / .870 / .704, T end .845 / .387 / .149 (all t0: Bayes .206, naive .209).
End of training reproduces the end-of-training analyses (different environments, 2 streams per cell):
  one-step correct split, tenth 1 -> 10: Bayes .125 -> .330, naive .078 -> .055, T .122 -> .320, E .070 -> .059 (scan:
  Optimal .12 -> .35, naive falls to .06). Profiles, tenths 1-10: Bayes .125 .197 .229 .273 .308 .310 .333 .335 .323
  .330; naive .078 .127 .146 .168 .178 .171 .142 .135 .091 .055 (inverted U); E end .070 .102 .123 .124 .128 .127 .115
  .116 .074 .059 (naive's inverted U). Gradient (tenth 10 - 1): Bayes +.205, naive -.023, T end +.198 [.180, .218], E
  end -.012 [-.026, +.003]. Early steps t0 = 2-10 carry most of it: Bayes +.357 vs +.108 for t0 = 11-24, T end +.350 /
  +.101. Six-step raw thirds (entangled / calm): Bayes apart .785 / .669, toward 1.145 / .904; naive apart .994 / .623,
  toward .376 / .478.
  At T: corners Bayes 11.63, naive 4.31, T 11.57, E 4.58; edges 8.49, 9.07, 8.49, 8.46 (end-of-training analysis:
  corners ~4.5, edges 9.1).
Splitting through training (td_tt_splitting.png):
  a gradient: T within -.004 to +.003 of 0 up to 160 training episodes, +.031 [.020, .043] (240), +.041 (400), +.063
    (560), +.074 (720), +.093 [.071, .116] (800), +.101 [.081, .125] (880), +.117 (960), +.150 (1,600), +.169 (2,400),
    +.205 (5,400), end +.198. Half-way (midpoint .091): 793 [617, 1,060]; early steps t0 = 2-10 (midpoint .144): 731
    [542, 971]; late steps t0 = 11-24 (midpoint .057): 1,090 [778, 2,016]. The environment halves' curves correlate r =
    .98 over checkpoints. E: -.022 to +.007 at every checkpoint (halves r = .09): no gradient at any checkpoint,
    slightly negative late as naive's; it never crosses.
    Matched distance to the cells: Bayes +.195, naive -.046 (midpoint .075); T +.057 (800), +.067 (880), +.080 (960),
    +.081 (1,040), +.084 (1,120), +.076 (1,200), +.095 (1,280), end +.197 (all of the end gradient is
    entanglement-specific). At 880, 34% of the raw gradient goes with entangled pairs being farther from their cells
    (raw - matched +.035 +- .007 SE over environments). Half-way: first reaches its midpoint at 927 [713, 1,260] and
    stays past it from 927 [769, 1,296] (the 1,200 checkpoint is only .001 above the midpoint). E end -.019.
    Null (shuffled entanglement): |gradient| <= .023 (T), .011 (E) at every checkpoint; Bayes +.012, naive +.010.
    Density (24 fresh streams per cell): the near pairs are 4x nearer (Bayes .081 vs .324 grid units; naive .068 vs
    .259) and the gradient is unchanged: Bayes +.229 vs +.224 (2 of the 24), naive -.037 vs -.031, T at 880 +.105 vs
    +.108 (probes +.101), end +.224 vs +.224 (probes +.198), E end -.013 vs -.016 (probes -.012); max |24 - 2 streams|
    over the 19 checkpoints .024 (T), .005 (E); r = .99 (T). The circles' own Bayes / naive (+.229 / -.037) are marked
    at the right edge of a (the fresh 2-stream set also runs ~.02 above the probes: Bayes +.224 vs +.205).
  b six steps on (entangled - calm third, k = 6): apart: Bayes .116, naive .371 (midpoint .244). T over-splits like
    naive first: it reaches naive's side of the midpoint at 194 [142, 211] and holds a plateau (.29-.39 over 220-1,120)
    from 216 [194, 389] to 1,567 [1,204, 2,254]; its peak cannot be placed within it (800 - 560 = +.021 +- .026 paired
    SE). Back on Bayes's side for good at 2,935 [1,758, 4,464]; end .123. E over-splits like naive from about 1,100 and
    never returns: on naive's side of the midpoint for good from 1,137 [826, 1,386] (it first reaches the midpoint at
    1,137 [94, 1,364]; the low bound is an early excursion, before 200, in 13.6% of resamples); its plateau starts at
    1,341 [1,159, 1,644] and has not ended by the end of training; maximum .353 at 2,400, below naive's .371; end .311.
    Toward the own cells: Bayes +.241, naive -.102 (midpoint .069): T half-way at 219 [201, 319] (the six-step window
    accumulates the later steps), +.341 (800), maximum +.415 at 1,920 (overshoots Bayes), end +.254; E <= +.021
    throughout, end -.065.
  c overall level (mean one-step dD of all near pairs): Bayes .276, naive .129, T end .279, E end .104. At a matched
    level T already has the gradient: mean .109 at 480 with +.036 [.021, .051] vs E's end (.104) -.012 [-.026, +.003];
    mean .140 at 560 with +.063 vs naive (.129) -.023.
  d, e heat-map line (each environment's peak tenth, then the mean over environments; [a, b] = 5-95% over the shared
    resamples): Bayes 8.16 [7.80, 8.54], naive 5.26 [4.88, 5.64]. T rises from naive's middle toward Bayes's level:
    5.36-5.92 over 20-160 training episodes, 6.20 (240), 6.52 (400), 6.66 (560), 7.10 (720), 7.30 (800), 7.68
    [7.14, 8.20] (880), then 7.42-8.50 at every checkpoint from 880 to the end (8.10 at 1,600, 8.22 at 5,400, end 7.94
    [7.58, 8.34]); past the naive-Bayes midpoint (6.71) for good from 626 [340, 831] (shared resamples). Paired by
    environment, T minus Bayes: -.48 +- .39 (880), -.06 +- .35 (1,600), -.42 +- .33 (2,400), +.06 +- .29 (5,400),
    +.28 +- .25 (9,600), +.02 +- .21 (20,000), -.22 +- .19 (end). This statistic cannot resolve the remaining gap
    (paired SE .32-.45 tenths over 880-2,000; > 2 SE below at 1 of those 15 checkpoints; averaged per environment over
    880-2,000, -.38 +- .27). The two more precise versions (definition 1) can. T minus Bayes, paired by environment:
    dD-weighted mean tenth -.24 +- .05 (880), -.19 +- .06 (1,040), -.10 +- .04 (1,600), -.09 +- .04 (2,400), -.03 +-
    .03 (5,400), +.01 +- .02 (20,000), -.00 +- .02 (end); peak per (environment, step) (Bayes 6.92, naive 5.92) -.41
    +- .14 (880), -.51 +- .15 (1,040), -.27 +- .14 (1,600), -.35 +- .11 (2,400), -.06 +- .12 (5,400), -.00 +- .08
    (20,000), -.02 +- .07 (end). Both put T below Bayes at every checkpoint over 880-2,000 (each > 2 SE: 15 of 15)
    and at Bayes's level from ~3,000 (dD-weighted / per step): still > 2 SE below at 4 / 6 of the 7 checkpoints over
    2,080-2,800 (the last at 2,400 / 2,800), at none of the 66 from 3,000; averaged per environment over 3,000-4,800,
    -.00 +- .03 / -.10 +- .10, from 5,000 +.01 +- .02 / +.01 +- .07. No overshoot in any version: averaged from 5,000
    the drawn line gives -.05 +- .19, and the only checkpoint > 2 SE above Bayes is the dD-weighted 9,600 (+.048 +-
    .024).
    E stays at naive's middle: 5.30-6.18 over 60-160, 4.68-6.24 from 880 (4.68-5.90 from 1,520), end 4.96 [4.52, 5.42];
    never past the midpoint; 2.3-3.3 tenths below Bayes's (paired, at the seven checkpoints above). The initial network
    (not drawn) gives 6.54, the argmax of noise. The dD-weighted mean tenth, compressed toward the middle: Bayes 6.12,
    naive 5.31; T 5.30-5.66 over 20-160 and 5.89-6.17 from 880; E 5.22-5.66 from 880 (5.22-5.47 from 1,520).
  f profile when T passes naive (880): .119 .161 .186 .203 .211 .207 .227 .213 .229 .220: Bayes-shaped (rise, then
    saturation above tenth 5; scaled to its own plateau .42 .68 .85 .93 at tenths 2-5 vs Bayes's .36 .52 .75 .93) at
    about two-thirds of Bayes's height (tenths 5-10: .218 vs .323) and half its rise from tenth 1 (.099 vs .199); no
    longer naive's inverted U.
Fold-back through training (td_tt_splitting_foldback.png):
  the initial network (shared by both runs): corners .38, edges .94 (all beliefs near the centre).
  T folds like naive first: corners 1.95 vs edges 5.43 at 20 training episodes (fold index -3.48; naive -4.76; Bayes
    +3.14), most folded -4.34 at 180; corners past naive's 4.31 for good at 342 [306, 423]; fold index half-way (-.81)
    at 774 [721, 845]; corners overtake edges (fold index 0) at 912 [799, 1,122] (7.72 vs 7.80 at 880); corners half-way
    (7.97) at 946 [843, 1,129]; corners 9.80 (1,600), 10.79 (2,400), 11.45 (5,400), end 11.57 (Bayes 11.63); edges end
    8.49 (Bayes 8.49).
  E stays folded: corners 2.00 vs edges 5.32 at 80 (-3.32), most folded -4.58 at 4,000, end 4.58 vs 8.46 (-3.88). Its
    corners hover at naive's level (4.31): 3.9-4.6 from about 1,500 (3.92-4.59; first 3.9 at 1,120; still below it at
    1,500-2,300, 3.92-4.26); first above it at the 2,400 checkpoint (log-interpolated 2,389 [1,099, never in 8%]); above
    it at 40% of the 80 checkpoints from 1,500; above it to the end only from 25,034 [4,258, never in 17% of resamples]
    (every checkpoint from 25,600), by .04-.28.
  Cell-mean variant, same picture: Bayes 11.61 / 8.46 (corners / edges), naive 4.22 / 8.94, T at 880 7.54 / 7.64, T end
    11.54 / 8.45, E end 4.49 / 8.36.
  d grid folds (% folded at T among naive's folded triangles): T 59.4 (240) -> 23.4 (880) -> 5.8 (2,400) -> 2.5 (end;
    Bayes 2.2); E 50.7 (240) -> 76.2 (8,000) -> 72.0. T's unfolding (the landmark): 25 / 50 / 75% of its drop at 551
    [391, 649] / 770 [703, 858] / 1,262 [1,097, 1,523] (the landmark's values, here with environment intervals); E never
    unfolds.
  e TIMING (T; order = disjoint 5-95% intervals). The half-way points of the splitting gradient (793 [617, 1,060]), the
    fold index (774 [721, 845]), the grid folds (770 [703, 858]), passing naive (landmark 800-880; log-interpolated 835
    [796, 874]) and the memory-use flip (800-1,040) all fall within ~620-1,060 training episodes and cannot be ordered
    at this resolution (supplementary paired ratios of crossing times: gradient / fold index x1.03 [x0.80, x1.33];
    gradient / passing naive x0.96 [x0.74, x1.28]; fold index / passing naive x0.93 [x0.86, x1.01]; gradient / grid
    folds x1.04 [x0.81, x1.39]; fold index / grid folds x1.01 [x0.89, x1.13]). The corners overtaking the edges (912
    [799, 1,122]), the corners' half-way (946 [843, 1,129]) and the distance-matched gradient (927 [769, 1,296]) overlap
    the same window (matched / raw x1.31 [x1.03, x1.62]: suggestive of a lag, but the intervals overlap, so not
    ordered). Before the window (disjoint from every interval in it): corners past naive's level (342 [306, 423]), the
    six-step toward-own-cells half-way (219 [201, 319]) and the start of the over-splitting plateau (216 [194, 389]).
    The accuracy half-way to Bayes (.618; 1,394 [1,277, 1,485]) comes after the gradient / fold / grid / passing-naive
    cluster (and after the flip, the early-step gradient, 731 [542, 971], the corners overtaking the edges and the
    corners' half-way), but is NOT ordered against the distance-matched gradient (927 [769, 1,296]: the intervals touch;
    supplementary paired ratio x1.23 [x1.05, x1.78], later in 99%), nor against the late-step gradient (1,090 [778,
    2,016]) or the grid folds' 75% point (1,262 [1,097, 1,523]). The over-splitting back on Bayes's side (2,935 [1,758,
    4,464]) comes after all of these (the accuracy half-way included) except the late-step gradient (1,090 [778,
    2,016]): those intervals overlap, so they are not ordered.
    ESN: it reaches none of the Trained RNN's Bayes-side levels (every half-way crossing above: never; its corners pass
    naive's own level for good only at 25,034, by .04-.28, hovering at it); it over-splits like naive from about 1,100
    and never returns (b: 1,137 [826, 1,386]).

FAILED, NULL, RETRACTED
  ESN: no entanglement gradient of correct splitting at any checkpoint (-.022 to +.007, halves uncorrelated) and no
    unfolding (fold index -3.3 to -4.6 from 80 training episodes on; corners at naive's level). Null, as expected.
  Retracted after verification: "the gradient passes its midpoint at 800-880, right where it passes naive" (also for the
    early steps) and the step-by-step order (steepest grid unfolding = fold index 720-800, six-step peak 800, gradient
    800-880, passes naive 880, corners overtake edges 880-960). The gradient's 800-880 came from cKDTree's return order
    among equidistant partners, which is not neutral: it put Bayes at +.212 and naive at -.020 (all tied partners: +.205
    / -.023), raising the midpoint from .091 to .096; with a 5-consecutive-checkpoint rule the crossing fell one
    checkpoint later. With ties averaged and environment intervals these events cannot be ordered.
  Retracted: "the matched gradient lags, passing its midpoint only at 1,200-1,280" (one checkpoint .001-.002 below the
    midpoint plus the 5-checkpoint rule); "~40% of the gradient at 880 is distance to the cells" is 34% after the tie
    fix.
  Retracted after re-verification: the heat-map line drawn as the argmax of the ENVIRONMENT-MEAN profile (+-2
    checkpoints; a statistic of an average). On the flat-topped profile it ran to the edge (10 at most checkpoints over
    720-7,000, mostly 9 over 7,200-24,000, 8 from 24,800), so it read as T peaking two tenths above Bayes's 8 and then
    settling. Per environment and then averaged (now drawn), T reaches Bayes's level without overshoot (d, e above:
    from ~3,000 on the precise versions). The check that claimed the two agree was wrong: they differed by about 2
    tenths.
  Corrected after re-verification: "the ESN reaches none of these levels" (it does reach and keep the over-splitting
    level of b, from 1,137); "its corners reach naive's level by ~1,500 ... no lasting crossing" (they are first above
    naive's level only at 2,400, and by the stays-past rule they do cross for good, at 25,034, by .04-.28); "after it:
    the accuracy half-way" (its interval touches the distance-matched gradient's, so those two are not ordered).
  Corrected after the second re-check: "at Bayes's level within noise from 880" (the drawn statistic cannot resolve the
    gap; the precise versions put T below Bayes at every checkpoint over 880-2,000 and at its level from ~3,000); "the
    over-splitting back on Bayes's side comes after all of these" (not after the late-step gradient: the intervals
    overlap).
  Not used: the six-step apart 'peak at 800' (a plateau; see b); the 'steepest grid unfolding' (one 5.8-point drop on a
    19-point series; replaced by the landmark's half-way point with its interval); the steepest-change timing of the
    114-checkpoint series (checkpoint-to-checkpoint jitter decides it).
  The six-step 'toward own cells' contrast is not a clean timing marker: the six-step window accumulates the steps after
    the entangled one, so it passes its midpoint already at 219.
  Unexplained: T's six-step toward-own-cells contrast overshoots Bayes (.415 at 1,920 vs .241) before settling (.254).

CHECKS: near_partners reproduces the verifier's all-tied-partners gradients exactly (Bayes +.2051, naive -.0229, T at
800 / 880 +.0927 / +.1015, E end -.0116) and its tie shares; the k-d tree path (2,400 particles per environment) equals
brute force in 40 of 40 tests (agents x environments x steps); the fold and cell-mean tables are bit-identical to the
builder's (ties do not enter them); the per-environment accuracy averages reproduce the stored accuracy (2e-16); passes
naive (880) and the flip (800-1,040) recomputed; ent_of reproduces the stored entanglement to 3e-15 (check); both runs
give identical values at the shared initial network; every crossing interval contains its point estimate; the heat-map
line IS the per-environment statistic averaged: with every resample count 1 the resampled line equals the drawn one
exactly (0.0), and it equals the mean of the per-environment peaks exactly; the per-(environment, step) peak equals an
independent running-mean argmax exactly, and both precise heat-map versions reproduce the second re-check's values; the
grid landmark's two rules (first covered / stays past) give the same points; the end-of-training values reproduce the
scan / harvest analyses (above).

Synthesis export (command 'synthesis' -> through_training/splitting/synthesis.npz, td_tt_gap.export; numbers prints the
same table): six measures x {Trained RNN 'trained', ESN 'echo'}, all from the probes (sample: 50 environments x 200
episodes), paired = True on gap.boot_counts() (1,000 x 50, seed 20261103). Environment indices verified: probes.npz env
equals probe row // 200 for all 10,000 rows (asserted), and every cache indexes environments by that env. Each measure
is a per-environment statistic averaged over environments, so each resampled curve is the count-weighted mean of the
per-environment values (gap.env_mean; series()'s own resampling code gives the same numbers on these counts, <= 1e-14),
naive's and Bayes's levels likewise on the same resamples; with every count 1 the resampled curve equals the drawn one
(<= 2e-16). x = training episodes, the initial network at 10. Rows (name: family, shape; c0 = passes naive's level for
good, c50 = past the naive-Bayes midpoint for good; T then E):
  'split gradient (most minus least entangled tenth)' (a; movement, rise): T c0 10 [10, 145], c50 793 [615, 1,070];
    E c0 10 [10, never; crosses in 81%], c50 never.
  'split gradient, distance-matched' (movement, rise): T c0 10, c50 927 [770, 1,305]; E c0 10 [10, 37,207], c50 never.
  'over-splitting, six steps apart' (b; movement, return; naive .371, Bayes .117): T c0 827 [10, 1,445] (it exceeds
    naive's level only at 800, .385), c50 2,935 [1,697, 4,454]; E c0 10 [10, 4,019] (never beyond naive's level: max
    .353, so no return to time), c50 never (on naive's side of the midpoint for good from 1,137).
  'six-step move toward own cells' (b; movement, rise; naive and Bayes on the same probes): T c0 10, c50 219 [201, 312];
    E c0 10 [10, 39,702], c50 never.
  'fold index (corners - edges)' (foldback c; answers, rise): T c0 10 [10, 189], c50 774 [706, 846]; E c0 10 [10,
    17,193], c50 never.
  'opposite corners' (foldback c; answers, rise; naive 4.31, Bayes 11.63): T c0 342 [304, 429], c50 946 [838, 1,127];
    E c0 25,034 [2,244, never; crosses in 84%], c50 never.
  c0 = 10 for the gradient, the matched gradient, the toward contrast and the fold index is not a 'passing naive' event:
  naive's levels there lie on the far side of an untrained network (the network starts between naive and Bayes and never
  goes below naive's level; the row notes say so). The grid-folds landmark is left out (its naive level is 100% of
  naive's own folds by definition, so the gap fraction is not defined the same way); it stays the landmark it is.
  Agreement with this module's crossings (same definition: the naive-Bayes midpoint, stays past): every c50 has the
  same point as grad_half / matched_half / apart_back / toward_half / fold_half / corners_half, and the corners' c0 the
  same as corners_naive (342; E 25,034). The intervals differ only through the resamples (shared seed 20261103 vs this
  module's 20261101; e.g. gradient [615, 1,070] vs [617, 1,060], matched [770, 1,305] vs [769, 1,296], over-splitting
  back [1,697, 4,454] vs [1,758, 4,464]; the ESN's corner crossing is the most volatile, since its corners hover at the
  level: low bound 2,244 vs 4,258): run on the shared resamples, this module's crossing code gives the export's c50
  (and the corners' c0) in all 1,000 resamples of every row.

Run from the repository root (CPU; probes ~12 min, grids ~30 min, folds ~1 min, figures / timeline / synthesis / numbers
~10 s each; each build step skips existing caches unless 'force' is given):
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg \\
        /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_splitting.py \\
        [force] [check] [probes] [grids] [folds] [figures] [timeline] [synthesis] [numbers]"""
import os
import sys
import glob
import time
import argparse
import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
GRIDS = os.path.join(TD, "summary_data", "grids")
OUT = os.path.join(TD, "through_training", "splitting")
RUNS = ("trained", "echo")
REFS = ("bayes", "naive")
NE, T, R = 50, 30, 10
V = np.arange(R, dtype=np.float64)
T0 = np.arange(2, 25)                     # 1-based steps t0 of the near pairs (plot_fig1c_splitting.py)
K = 6                                     # lags k = 0..K after t0
ND = 10                                   # deciles of the pair's entanglement
NG = 3                                    # thirds of the pair's distance to its cells (matched control)
X0 = 10                                   # the initial network on the log axis
TOL = 1e-9                                # distance tie tolerance (grid units; ties in the data are exact)
NB, BOOT_SEED = 1000, 20261101            # environment bootstrap: resamples of the 50 environments, seed


def train_x(ep):
    ep = np.asarray(ep)
    return np.maximum(ep - ep // 5, X0).astype(float)


def positions(lq):
    """posterior means (E[r0], E[r1]) (..., 2) from log beliefs (..., 2, R), renormalised"""
    lq = np.asarray(lq, np.float64)
    return np.exp(lq - logsumexp(lq, -1, keepdims=True)) @ V


# ═══════════════════════════════════════════════════════════════════
# measures
# ═══════════════════════════════════════════════════════════════════

def fold_tables(X, env, r, steps):
    """mean particle-pair estimate distance per environment, step and separation: (NE, len(steps), R, R), entry
    [e, s, a, b] = mean over all pairs of particles of environment e whose true cells differ by |dr0| = a, |dr1| = b
    of ||x_i - x_j|| at step steps[s] (0-based); and the same for the cell means (each cell's particles averaged
    first: a distance of averages) at the last of steps"""
    out = np.zeros((NE, len(steps), R * R))
    cm = np.zeros((NE, R * R))
    for e in range(NE):
        ix = np.flatnonzero(env == e)
        x, rr = X[ix][:, steps].astype(np.float64), r[ix]
        i, j = np.triu_indices(len(ix), 1)
        ab = np.abs(rr[i, 0] - rr[j, 0]) * R + np.abs(rr[i, 1] - rr[j, 1])
        cnt = np.bincount(ab, minlength=R * R)
        d = np.linalg.norm(x[i] - x[j], axis=-1)                                    # (pairs, steps)
        for s in range(len(steps)):
            out[e, s] = np.bincount(ab, d[:, s], R * R) / cnt
        cell = rr[:, 0] * R + rr[:, 1]
        xm = np.stack([np.bincount(cell, x[:, -1, q], R * R) for q in range(2)], -1) / np.bincount(cell, minlength=R * R)[:, None]
        ci, cj = np.triu_indices(R * R, 1)
        ab2 = np.abs(ci // R - cj // R) * R + np.abs(ci % R - cj % R)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm[e] = np.bincount(ab2, np.linalg.norm(xm[ci] - xm[cj], axis=-1), R * R) / np.bincount(ab2, minlength=R * R)
    return out.reshape(NE, len(steps), R, R), cm.reshape(NE, R, R)


def near_partners(p, cell, tol=TOL):
    """every particle's nearest particle(s) of ANOTHER cell at positions p (n, 2): rows (i, j, w) for ALL other-cell
    partners within tol of particle i's nearest other-cell distance, each weighted 1 / (number of tied partners), so
    every particle weighs 1 (exact distance ties are common at early steps: identical observation counts give Bayes /
    naive identical positions, identical observation sequences give a network identical outputs). Brute force up to
    500 particles; above that a k-d tree, with brute force for the particles whose tie set may reach past its k
    neighbours."""
    n = len(p)
    if n <= 500:
        dm = np.sqrt((p[:, None, 0] - p[None, :, 0]) ** 2 + (p[:, None, 1] - p[None, :, 1]) ** 2)
        dm[cell[:, None] == cell[None, :]] = np.inf
        I, J = np.nonzero(dm <= dm.min(1, keepdims=True) + tol)
    else:
        kq = min(n, 96)
        dist, idx = cKDTree(p).query(p, k=kq)
        other = cell[idx] != cell[:, None]
        dmin = np.where(other, dist, np.inf).min(1)
        full = ~np.isfinite(dmin) | (dist[:, -1] <= dmin + tol)                      # tie set may extend past kq
        cand = other & (dist <= dmin[:, None] + tol) & ~full[:, None]
        rows, cols = np.nonzero(cand)
        I, J = [rows], [idx[rows, cols]]
        for i in np.flatnonzero(full):
            di = np.sqrt((p[i, 0] - p[:, 0]) ** 2 + (p[i, 1] - p[:, 1]) ** 2)
            di[cell == cell[i]] = np.inf
            jj = np.flatnonzero(di <= di.min() + tol)
            I.append(np.full(len(jj), i)), J.append(jj)
        I, J = np.concatenate(I), np.concatenate(J)
        o = np.lexsort((J, I))
        I, J = I[o], J[o]
    nt = np.bincount(I, minlength=n)
    return I, J, 1. / nt[I], nt


def wbins(v, w, nb):
    """bins 0..nb-1 of v by weighted rank: rows sorted by v (stable), each row placed by the midpoint of its weight
    on the cumulative weight axis (unit weights: equal-count bins as rank * nb // n, up to the boundary rounding)"""
    o = np.argsort(v, kind="stable")
    cw = np.cumsum(w[o]) - w[o] / 2
    b = np.empty(len(v), int)
    b[o] = np.minimum((cw / w.sum() * nb).astype(int), nb - 1)
    return b


def _wmean(b, w, y, nb):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.bincount(b, w * y, nb) / np.bincount(b, w, nb)


def split_stats(X, env, r, rev, rng):
    """the near-pair splitting measures of plot_fig1c_splitting.py per environment and step t0 (see the docstring),
    with every equidistant other-cell partner kept at weight 1 / n_tied (near_partners) and weighted tenths / thirds
    (wbins). X (n, T, 2) positions, env (n,), r (n, 2) true cells, rev (n, T) entanglement (mean of the two
    variables). Also returns the share of particles with >= 2 equidistant partners (tied) per environment and step."""
    nt = len(T0)
    o = {k: np.full((NE, nt, ND), np.nan) for k in ("corr", "split", "corr_null", "corr_g", "g", "d0")}
    o.update({k: np.full((NE, nt, 2, K + 1), np.nan) for k in ("corr_lag", "split_lag")})
    o["tied"] = np.zeros((NE, nt))
    for e in range(NE):
        ix = np.flatnonzero(env == e)
        x, c, rv = X[ix].astype(np.float64), r[ix].astype(np.float64), rev[ix]
        cell = r[ix, 0] * R + r[ix, 1]
        for k_, t0 in enumerate(T0):
            s0 = t0 - 1
            i, j, w, ntie = near_partners(x[:, s0], cell)
            o["tied"][e, k_] = np.mean(ntie > 1)
            ent = (rv[i, s0 + 1] + rv[j, s0 + 1]) / 2                                # the pair's next-step rev
            xi, xj = x[i, s0:s0 + K + 1], x[j, s0:s0 + K + 1]
            ci, cj = c[i][:, None], c[j][:, None]
            d = np.linalg.norm(xi - xj, axis=-1)
            D = (np.linalg.norm(xi - cj, axis=-1) - np.linalg.norm(xi - ci, axis=-1) +
                 np.linalg.norm(xj - ci, axis=-1) - np.linalg.norm(xj - cj, axis=-1))
            dd, dD = d - d[:, :1], D - D[:, :1]
            dec = wbins(ent, w, ND)
            o["corr"][e, k_] = _wmean(dec, w, dD[:, 1], ND)
            o["split"][e, k_] = _wmean(dec, w, dd[:, 1], ND)
            sh = wbins(rng.permutation(ent), w, ND)                                  # null: entanglement shuffled
            o["corr_null"][e, k_] = _wmean(sh, w, dD[:, 1], ND)
            g = (np.linalg.norm(xi[:, 0] - c[i], axis=-1) + np.linalg.norm(xj[:, 0] - c[j], axis=-1)) / 2
            o["g"][e, k_] = _wmean(dec, w, g, ND)
            o["d0"][e, k_] = _wmean(dec, w, d[:, 0], ND)
            th = wbins(ent, w, 3)
            for q, sel in enumerate((th == 2, th == 0)):
                o["corr_lag"][e, k_, q] = w[sel] @ dD[sel] / w[sel].sum()
                o["split_lag"][e, k_, q] = w[sel] @ dd[sel] / w[sel].sum()
            gs = wbins(g, w, NG)                                                     # matched: within thirds of g
            acc = np.zeros((NG, ND))
            for s in range(NG):
                m = np.flatnonzero(gs == s)
                acc[s] = _wmean(wbins(ent[m], w[m], ND), w[m], dD[m, 1], ND)
            o["corr_g"][e, k_] = acc.mean(0)
    return {k: v.astype(np.float32) for k, v in o.items()}


def ent_of(obs, L):
    """entanglement per variable (n, T, 2) exactly as td_analysis.bayes_side: step_dkl(npx, approx_lik(jb)) in float64
    for observations obs (n, T, 5) of one environment with table L (5, R, R)"""
    o = obs.astype(np.float64)
    n = len(o)
    ll = np.einsum("nti,ijk->ntjk", o, np.log(L)) + np.einsum("nti,ijk->ntjk", 1 - o, np.log1p(-L))
    lam = np.cumsum(ll, 1)
    post = np.exp(lam - logsumexp(lam.reshape(n, T, -1), -1)[..., None, None])
    jb = np.stack([post.sum(3), post.sum(2)], 2)                                     # (n, T, 2, R)
    Lb = np.stack([L.mean(2), L.mean(1)], 1)                                         # (5, 2, R) naive tables
    lpx = np.einsum("nti,ikr->ntkr", o, np.log(Lb)) + np.einsum("nti,ikr->ntkr", 1 - o, np.log1p(-Lb))
    npx = np.exp(lpx - logsumexp(lpx, -1, keepdims=True))
    b = np.maximum(jb, 1e-99)                                                        # supplementary_plotter2.approx_lik
    px = np.zeros_like(b)
    px[:, 0], q = b[:, 0], b[:, 1:] / b[:, :-1]
    px[:, 1:] = q / q.sum(-1, keepdims=True)
    p, q = np.maximum(npx, 1e-99), np.maximum(px, 1e-99)                             # supplementary_plotter2.step_dkl
    p, q = p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)
    return .5 * np.sum(p * (np.log(p) - np.log(q)) + q * (np.log(q) - np.log(p)), -1)


# ═══════════════════════════════════════════════════════════════════
# data
# ═══════════════════════════════════════════════════════════════════

def _probe_inputs():
    P = np.load(os.path.join(AN, "probes.npz"))
    return {k: P[k] for k in ("env", "cell", "r", "goal", "ent", "log_jb", "log_nb", "right_naive", "right_bayes")}


def _env_acc(right_goal, env):
    """goal accuracy at T per environment (NE,) from per-episode correctness (n,)"""
    return np.bincount(env, right_goal.astype(float), NE) / np.bincount(env, minlength=NE)


def _one(X, P, rng):
    F, CM = fold_tables(X, P["env"], P["r"], np.arange(T))
    out = dict(fold=F.astype(np.float32), fold_cm=CM.astype(np.float32))
    out.update(split_stats(X, P["env"], P["r"], P["ent"].mean(-1), rng))
    return out


def build_probes(runs=RUNS):
    """per checkpoint (all 114 of each run) and for Bayes / naive: the fold tables, the near-pair measures and the
    goal accuracy at T per environment on the probes -> OUT/probes_<run>.npz, OUT/probes_refs.npz"""
    os.makedirs(OUT, exist_ok=True)
    P = _probe_inputs()
    rows = np.arange(len(P["goal"]))
    path = os.path.join(OUT, "probes_refs.npz")
    if not os.path.exists(path):
        t0 = time.time()
        res = {a: _one(positions(P[f"log_{k}"]), P, np.random.default_rng(1)) for a, k in (("bayes", "jb"), ("naive", "nb"))}
        for a in REFS:
            res[a]["acc_env"] = _env_acc(P[f"right_{a}"][rows, P["goal"]], P["env"])
        np.savez(path, **{f"{a}|{k}": v for a, d in res.items() for k, v in d.items()})
        print(f"refs ({time.time() - t0:.0f} s)", flush=True)
    for run in runs:
        path = os.path.join(OUT, f"probes_{run}.npz")
        if os.path.exists(path):
            continue
        files = sorted(glob.glob(os.path.join(AN, run, "ck_?????.npz")))
        res, eps, acc, acc_env = {}, [], [], []
        t0 = time.time()
        for k, f in enumerate(files):
            z = np.load(f)
            eps.append(int(z["episode"]))
            acc.append(float(z["acc"][-1]))
            acc_env.append(_env_acc(z["right"][rows, -1, P["goal"]], P["env"]))
            for key, v in _one(positions(z["logq"]), P, np.random.default_rng(1)).items():
                res.setdefault(key, []).append(v)
            if k % 10 == 0:
                print(f"{run} {k + 1}/{len(files)} ep {eps[-1]} ({time.time() - t0:.0f} s)", flush=True)
        np.savez(path, episode=np.array(eps), acc_T=np.array(acc), acc_env=np.array(acc_env),
                 **{k: np.stack(v) for k, v in res.items()})
        print(f"saved {path} ({time.time() - t0:.0f} s)", flush=True)


def build_grid_folds():
    """per environment, the belief-grid folds of td_summary_data.grids() (its functions, unchanged): at T, the number
    of naive's folded triangles (of 162 per environment) and how many of them are folded in each network's grid at
    each of the 19 checkpoints -> OUT/grid_folds.npz (the environment bootstrap of the grid-unfolding landmark)"""
    path = os.path.join(OUT, "grid_folds.npz")
    if os.path.exists(path):
        return
    import td_summary_data as tsd
    E = np.load(os.path.join(GRIDS, "fwd_episodes.npz"))
    fN = tsd.folded(tsd._per_env(E, E["m_naive"][:, -1]))                           # (NE, 162)
    fB = tsd.folded(tsd._per_env(E, E["m_bayes"][:, -1]))
    out = {"naive_n": fN.sum(1), "bayes_where_naive": (fB & fN).sum(1)}
    for run in RUNS:
        F = np.load(os.path.join(GRIDS, f"fwd_{run}.npz"))
        out[f"x_{run}"] = np.where(F["tr_ep"] > 0, F["tr_ep"], X0).astype(float)
        fold = np.stack([tsd.folded(tsd._per_env(E, F["m"][k][:, -1].astype(float))) for k in range(len(F["ep"]))])
        out[f"where_naive_{run}"] = (fold & fN).sum(-1)                              # (19, NE)
    np.savez(path, **out)
    print(f"saved {path}", flush=True)


def build_grids():
    """density check of the near-pair measures on td_grids_fwd's fresh streams (the probes' 50 environments x 100
    cells x 24 streams, 19 checkpoints): all 24 streams per cell vs 2 of them (streams 0-1: the probes' density)
    -> OUT/grids_check.npz"""
    path = os.path.join(OUT, "grids_check.npz")
    if os.path.exists(path):
        return
    t0 = time.time()
    E = np.load(os.path.join(GRIDS, "fwd_episodes.npz"))
    L = np.load(os.path.join(AN, "probes.npz"))["L"]
    env, cell, r, obs = E["env"], E["cell"], E["r"], E["obs"]
    rev = np.zeros((len(env), T))
    for e in range(NE):
        ix = np.flatnonzero(env == e)
        rev[ix] = ent_of(obs[ix], L[e]).mean(-1)
    print(f"grids: entanglement ({time.time() - t0:.0f} s)", flush=True)
    stream = np.zeros(len(env), int)                                                 # stream index within the cell
    key = env * 100 + cell
    order = np.argsort(key, kind="stable")
    first = np.r_[0, np.flatnonzero(np.diff(key[order])) + 1]
    stream[order] = np.arange(len(env)) - np.repeat(first, np.diff(np.r_[first, len(env)]))
    two = stream < 2
    out = {}
    agents = [("bayes", E["m_bayes"]), ("naive", E["m_naive"])]
    for run in RUNS:
        F = np.load(os.path.join(GRIDS, f"fwd_{run}.npz"))
        out[f"{run}|episode"] = F["ep"]
        M = F["m"]                                                                   # (19, n, T, 2) float16, once
        agents += [(f"{run}|{k}", M[k]) for k in range(len(M))]
    for name, X in agents:
        for sub, sel in (("all", slice(None)), ("two", two)):
            s = split_stats(X[sel], env[sel], r[sel], rev[sel], np.random.default_rng(1))
            for k in ("corr", "split", "corr_g", "d0", "tied"):
                out[f"{name}|{sub}|{k}"] = s[k]
        print(f"grids: {name} ({time.time() - t0:.0f} s)", flush=True)
    stacked = {}
    for run in RUNS:
        for sub in ("all", "two"):
            for k in ("corr", "split", "corr_g", "d0", "tied"):
                stacked[f"{run}|{sub}|{k}"] = np.stack([out.pop(f"{run}|{q}|{sub}|{k}") for q in range(len(out[f"{run}|episode"]))])
    np.savez(path, **out, **stacked)
    print(f"saved {path} ({time.time() - t0:.0f} s)", flush=True)


def check_ent(n_env=3):
    """ent_of reproduces the probes' stored entanglement"""
    P = np.load(os.path.join(AN, "probes.npz"))
    for e in range(n_env):
        ix = np.flatnonzero(P["env"] == e)
        print(f"env {e}: max |ent_of - stored| = {np.abs(ent_of(P['obs'][ix], P['L'][e]) - P['ent'][ix]).max():.2e}")


# ═══════════════════════════════════════════════════════════════════
# series (per environment first, then averaged) and timing references
# ═══════════════════════════════════════════════════════════════════

def pair_counts():
    """particle pairs per environment for each separation (|dr0|, |dr1|) (R, R): the probes' 2 streams per cell"""
    cells = np.stack(np.divmod(np.arange(R * R), R), 1)
    rr = np.repeat(cells, 2, 0)
    i, j = np.triu_indices(len(rr), 1)
    return np.bincount(np.abs(rr[i, 0] - rr[j, 0]) * R + np.abs(rr[i, 1] - rr[j, 1]), minlength=R * R).reshape(R, R)


def curves_two_var(F):
    """(..., NE, R, R) per-environment tables -> x (12,), (..., 12): the two-variable separations (|dr0|, |dr1| >= 1)
    in 1-unit bins of their true distance [1, 2), ..., [12, 13): x = the bin's pair-weighted mean true distance, y =
    the mean estimate distance over the bin's pairs within each environment, then averaged over environments (the
    distinct distances alone give jagged curves: one true distance mixes separations of different shapes)"""
    a, b = np.meshgrid(np.arange(R), np.arange(R), indexing="ij")
    cnt = pair_counts().astype(np.float64)
    two = (a >= 1) & (b >= 1)
    dist = np.sqrt(a ** 2 + b ** 2)[two]
    grp = np.floor(dist).astype(int) - 1                                             # 0..11
    c = cnt[two]
    v = F[..., two]                                                                  # (..., NE, n_two)
    xs = np.array([(dist * c)[grp == g].sum() / c[grp == g].sum() for g in range(12)])
    num = np.stack([(v[..., grp == g] * c[grp == g]).sum(-1) for g in range(12)], -1)
    return xs, (num / np.array([c[grp == g].sum() for g in range(12)])).mean(-2)


def boot_weights():
    """(NB, NE) environment-bootstrap weights: row b = how often each of the 50 environments is drawn (with
    replacement) in resample b, / 50"""
    idx = np.random.default_rng(BOOT_SEED).integers(0, NE, (NB, NE))
    W = np.zeros((NB, NE))
    np.add.at(W, (np.repeat(np.arange(NB), NE), idx.ravel()), 1.)
    return W / NE


def series():
    """everything the figures and numbers use, from the caches. Per agent: key_env = the per-environment values
    (environment axis last; 'dec' (.., NE, ND)), key = their mean over the 50 environments, key_b = the environment
    bootstrap means (NB, ..); Bayes / naive are resampled on the same environments (paired)"""
    W = boot_weights()
    S = {"W": W}
    Rf = np.load(os.path.join(OUT, "probes_refs.npz"))
    early = T0 <= 10
    for a in REFS + RUNS:
        Z = Rf if a in REFS else np.load(os.path.join(OUT, f"probes_{a}.npz"))
        g = lambda k: Z[f"{a}|{k}"] if a in REFS else Z[k]                              # noqa: E731
        d, e = {}, {}
        ct = g("corr").astype(np.float64)                                             # (.., NE, t0, ND)
        corr = np.nanmean(ct, -2)                                                     # (.., NE, ND): mean over t0
        e["con"] = corr[..., -1] - corr[..., 0]                                       # most - least entangled tenth
        for k, sel in (("early", early), ("late", ~early)):
            c = np.nanmean(ct[..., sel, :], -2)
            e[f"con_{k}"] = c[..., -1] - c[..., 0]
        for k, key in (("con_g", "corr_g"), ("con_null", "corr_null")):
            c = np.nanmean(g(key).astype(np.float64), -2)
            e[k] = c[..., -1] - c[..., 0]
        e["mean"] = corr.mean(-1)                                                     # all near pairs (equal tenths)
        L6 = {k: np.nanmean(g(k)[..., K].astype(np.float64), -2) for k in ("corr_lag", "split_lag")}  # (.., NE, 2)
        e["toward6"] = L6["corr_lag"][..., 0] - L6["corr_lag"][..., 1]                # entangled - calm third
        e["apart6"] = L6["split_lag"][..., 0] - L6["split_lag"][..., 1]
        F = g("fold")[..., -1, :, :].astype(np.float64)                               # (.., NE, R, R) at T
        e["corners"] = F[..., R - 1, R - 1]
        e["edges"] = (F[..., R - 1, 0] + F[..., 0, R - 1]) / 2
        e["fold_idx"] = e["corners"] - e["edges"]
        e["acc"] = g("acc_env").astype(np.float64)
        for k, v in e.items():
            d[k + "_env"], d[k] = v, v.mean(-1)
            d[k + "_b"] = np.einsum("be,...e->b...", W, v)
        d["dec_env"] = corr
        d["peak_step_env"] = peak_tenth(ct).mean(-1)                # (.., NE): peak tenth per (environment, step), mean
        d["dec"] = corr.mean(-2)                                                      # (.., ND)
        d["dec_b"] = np.einsum("be,...ed->b...d", W, corr)
        d["six_raw"] = {k: v.mean(-2) for k, v in L6.items()}                         # (.., 2): entangled, calm
        d["split_dec"] = np.nanmean(g("split"), -2).mean(-2)
        d["tied"] = g("tied").mean(-2)                                                # (.., t0)
        cm = g("fold_cm").astype(np.float64)
        d["corners_cm"] = cm[..., R - 1, R - 1].mean(-1)
        d["edges_cm"] = ((cm[..., R - 1, 0] + cm[..., 0, R - 1]) / 2).mean(-1)
        d["xs"], d["curve"] = curves_two_var(F)
        d["con_se"] = d["con_env"].std(-1, ddof=1) / np.sqrt(NE)
        if a in RUNS:
            d["episode"], d["x"], d["acc_all"] = Z["episode"], train_x(Z["episode"]), Z["acc_T"]
            d["con_A"], d["con_B"] = d["con_env"][:, :25].mean(1), d["con_env"][:, 25:].mean(1)
        S[a] = d
    path = os.path.join(OUT, "grids_check.npz")
    if os.path.exists(path):
        G = np.load(path)
        for a in REFS + RUNS:
            for sub in ("all", "two"):
                c = np.nanmean(G[f"{a}|{sub}|corr"].astype(np.float64), -2)
                S[a][f"grid_con_{sub}"] = (c[..., -1] - c[..., 0]).mean(-1)
                S[a][f"grid_d0_{sub}"] = np.nanmean(G[f"{a}|{sub}|d0"], (-1, -2, -3))
                cg = np.nanmean(G[f"{a}|{sub}|corr_g"].astype(np.float64), -2)
                S[a][f"grid_con_g_{sub}"] = (cg[..., -1] - cg[..., 0]).mean(-1)
                S[a][f"grid_tied_{sub}"] = G[f"{a}|{sub}|tied"].mean(-2)
            if a in RUNS:
                S[a]["grid_x"] = train_x(G[f"{a}|episode"])
    path = os.path.join(OUT, "grid_folds.npz")
    if os.path.exists(path):                                                         # td_summary h, per environment
        Gf = np.load(path)
        n = Gf["naive_n"].astype(np.float64)
        S["bayes"]["gridfold"] = 100 * Gf["bayes_where_naive"].sum() / n.sum()
        for a in RUNS:
            c = Gf[f"where_naive_{a}"].astype(np.float64)                             # (19, NE)
            S[a]["gridfold_x"], S[a]["gridfold"] = Gf[f"x_{a}"], 100 * c.sum(1) / n.sum()
            S[a]["gridfold_b"] = 100 * (W @ c.T) / (W @ n)[:, None]                   # (NB, 19)
    return S


def references():
    """passing naive (training episodes; the landmark: first checkpoint whose goal accuracy at T beats naive's), the
    memory-use flip interval (td_flip_scan) and the belief-grid folds (td_summary_data.grids())"""
    P = np.load(os.path.join(AN, "probes.npz"))
    naive = P["right_naive"][np.arange(len(P["goal"])), P["goal"]].mean()
    bayes = P["right_bayes"][np.arange(len(P["goal"])), P["goal"]].mean()
    Z = np.load(os.path.join(OUT, "probes_trained.npz"))
    x = train_x(Z["episode"])
    k = int(np.argmax(Z["acc_T"] > naive))
    out = {"naive_acc": naive, "bayes_acc": bayes, "passes": float(x[k]), "passes_prev": float(x[k - 1]),
           "acc_pass": (float(Z["acc_T"][k - 1]), float(Z["acc_T"][k]))}
    try:                                                                             # td_flip_scan's paired scan
        import td_flip_scan as fs
        z = np.load(os.path.join(AN, "flip_scan_paired.npz"))
        g = z["trained|memory_cost|entanglement | across"]
        fl = fs.flips(g[:, 0], g[:, 1], g[:, 2])[0]
        xf = train_x(z["trained|episode"])
        out["flip"] = (float(xf[fl["end"]]), float(xf[fl["start"]]))
    except Exception as err:                                                         # the documented interval
        print("flip scan unavailable:", err)
        out["flip"] = (800., 1040.)
    G = np.load(os.path.join(TD, "summary_data", "grids.npz"))
    out["grids"] = {k: G[k] for k in G.files}
    return out


def _interp(x, y, lev, k0, k1):
    """the log-interpolated x where y crosses lev between checkpoints k0 and k1 (arrays)"""
    lx = np.log(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.clip((lev - y[0]) / (y[1] - y[0]), 0, 1)
    return np.exp(lx[k0] + f * (lx[k1] - lx[k0]))


def cross_last(x, y, level, sign=1):
    """the crossing used throughout: the training episode (log-interpolated between checkpoints) after which
    sign * (y - level) > 0 holds to the end of training; x[0] if it holds from the start, NaN (never) if it does not
    hold at the last checkpoint. y (K,) or (B, K), level scalar or (B,) -> scalar or (B,)"""
    y = np.asarray(y, np.float64)
    one = y.ndim == 1
    y = np.atleast_2d(y)
    lev = np.broadcast_to(np.asarray(level, np.float64), y.shape[:1])
    past = sign * (y - lev[:, None]) > 0
    K = y.shape[1]
    out = np.full(len(y), np.nan)
    notp = ~past
    anyn = notp.any(1)
    out[past[:, -1] & ~anyn] = x[0]
    sel = np.flatnonzero(past[:, -1] & anyn)
    k = K - 1 - np.argmax(notp[sel, ::-1], 1)                                        # last checkpoint not past
    out[sel] = _interp(x, (y[sel, k], y[sel, k + 1]), lev[sel], k, k + 1)
    return out[0] if one else out


def cross_first(x, y, level, sign=1):
    """the training episode (log-interpolated) at which sign * (y - level) > 0 FIRST holds (NaN: never)"""
    y = np.asarray(y, np.float64)
    one = y.ndim == 1
    y = np.atleast_2d(y)
    lev = np.broadcast_to(np.asarray(level, np.float64), y.shape[:1])
    past = sign * (y - lev[:, None]) > 0
    out = np.full(len(y), np.nan)
    k = np.argmax(past, 1)
    out[past.any(1) & (k == 0)] = x[0]
    sel = np.flatnonzero(past.any(1) & (k > 0))
    k = k[sel]
    out[sel] = _interp(x, (y[sel, k - 1], y[sel, k]), lev[sel], k - 1, k)
    return out[0] if one else out


def ci(v):
    """5-95% interval of bootstrap crossings (NaN = never counts as later than any training episode) and the share
    of resamples that never cross"""
    v = np.where(np.isnan(v), np.inf, np.asarray(v, np.float64))
    lo, hi = np.quantile(v, [.05, .95], method="nearest")
    return float(lo), float(hi), float(np.mean(np.isinf(v)))


def crossing(S, a, key, level="mid", sign=1, rule="last"):
    """a crossing of agent a's series key with its environment-bootstrap interval: level 'mid' (the naive-Bayes
    midpoint), 'naive' (naive's level) or a number; Bayes / naive levels resampled with the same environments.
    -> dict(x = the crossing, lo, hi, never = share of resamples that never cross, level)"""
    d, B, N = S[a], S["bayes"], S["naive"]
    if level == "mid":
        lev, lev_b = (B[key] + N[key]) / 2, (B[key + "_b"] + N[key + "_b"]) / 2
    elif level == "naive":
        lev, lev_b = N[key], N[key + "_b"]
    else:
        lev, lev_b = float(level), np.full(NB, float(level))
    f = cross_last if rule == "last" else cross_first
    xb = f(d["x"], d[key + "_b"], lev_b, sign)
    lo, hi, never = ci(xb)
    return dict(x=float(f(d["x"], d[key], lev, sign)), lo=lo, hi=hi, never=never, level=float(lev), b=xb)


def grid_unfolding(S, frac=.5, rule="first"):
    """the grid-unfolding landmark (td_summary_data.grids(); 25 / 50 / 75%: 551 / 770 / 1,262): the training episode
    at which the Trained RNN's % folded where naive folds first covers frac of its drop from the initial network to
    the end (mean of the last 3 checkpoints), log-interpolated; with the environment-bootstrap interval. rule 'last':
    the landmark's own rule (td_tt_wormholes.unfolding: stays past to the end), which gives the same points"""
    d = S["trained"]
    x, y, yb = d["gridfold_x"], d["gridfold"], d["gridfold_b"]
    f = cross_first if rule == "first" else cross_last

    def one(v):
        lev = v[..., :1] + frac * (v[..., -3:].mean(-1, keepdims=True) - v[..., :1])
        return f(x, v, lev[..., 0], -1)
    xb = one(yb)
    lo, hi, never = ci(xb)
    lev = y[0] + frac * (y[-3:].mean() - y[0])
    return dict(x=float(one(y)), lo=lo, hi=hi, never=never, b=xb, level=float(lev))


def timing(S, ref):
    """every timing marker of the Trained RNN (and the ESN where it crosses) with its 5-95% interval"""
    tm = {}
    for a in RUNS:
        tm[a] = {
            "grad_half": crossing(S, a, "con"),
            "grad_first": crossing(S, a, "con", rule="first"),
            "early_half": crossing(S, a, "con_early"),
            "late_half": crossing(S, a, "con_late"),
            "matched_first": crossing(S, a, "con_g", rule="first"),
            "matched_half": crossing(S, a, "con_g"),
            "apart_up": crossing(S, a, "apart6", rule="first"),
            "apart_over": crossing(S, a, "apart6"),                    # stays on naive's side of the midpoint
            "apart_back": crossing(S, a, "apart6", sign=-1),
            "toward_half": crossing(S, a, "toward6"),
            "fold_half": crossing(S, a, "fold_idx"),
            "fold_zero": crossing(S, a, "fold_idx", level=0.),
            "corners_naive": crossing(S, a, "corners", level="naive"),
            "corners_naive_first": crossing(S, a, "corners", level="naive", rule="first"),
            "corners_half": crossing(S, a, "corners"),
            "acc_naive": crossing(S, a, "acc", level="naive"),
            "acc_half": crossing(S, a, "acc"),
        }
    if "gridfold_b" in S["trained"]:
        tm["trained"]["grid_half"] = grid_unfolding(S)
        tm["trained"]["grid_25"], tm["trained"]["grid_75"] = grid_unfolding(S, .25), grid_unfolding(S, .75)
        for f in (.25, .5, .75):                                    # the landmark's stays-past rule: same points
            tm["trained"][f"grid_{100 * f:.0f}_stays"] = grid_unfolding(S, f, rule="last")
    return tm


def plateau(x, y, frac=.8):
    """the span of the over-splitting plateau: from the first checkpoint at which the 3-checkpoint running mean of y
    reaches frac of its maximum to the last one at which it is still there (log-interpolated); (B, K) or (K,)"""
    y = np.atleast_2d(np.asarray(y, np.float64))
    ym = np.concatenate([y[:, :1], (y[:, :-2] + y[:, 1:-1] + y[:, 2:]) / 3, y[:, -1:]], 1)
    lev = frac * ym.max(1)
    up = cross_first(x, ym, lev, 1)
    down = cross_last(x, ym, lev, -1)
    return up, down


# ═══════════════════════════════════════════════════════════════════
# figures
# ═══════════════════════════════════════════════════════════════════

COL = {"trained": "#1f77b4", "echo": "#ff7f0e", "bayes": "k", "naive": "0.5"}
NAME = {"trained": "Trained RNN", "echo": "ESN", "bayes": "Bayes", "naive": "naive"}
XMIN, XMAX = 8, 1e5
XSEP, XB, XN = 4.5e4, (4.8e4, 6.8e4), (6.8e4, 9.6e4)                                 # right-edge reference columns
CMAPS = {"trained": "Blues", "echo": "Oranges"}
DASH = (0, (3, 1.6))


def _style():
    import matplotlib.pyplot as plt
    from main.utils import apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    return plt, fs


def _edges(x):
    lx = np.log(x)
    mid = (lx[1:] + lx[:-1]) / 2
    return np.exp(np.concatenate([[lx[0] - (mid[0] - lx[0])], mid, [lx[-1] + (lx[-1] - mid[-1])]]))


def _xaxis(ax, fs, label=True):
    ax.set_xscale("log")
    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks([X0, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.minorticks_off()
    if label:
        ax.set_xlabel("training episodes", fontsize=fs["label"], labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)


def _marks(ax, ref, heat=False):
    """passing naive (dotted) and, on line panels, the memory-use flip interval (band)"""
    if heat:
        ax.axvline(ref["passes"], color="w", lw=1.4, ls=":", zorder=4)
    else:
        ax.axvspan(*ref["flip"], color="0.88", lw=0, zorder=0)
        ax.axvline(ref["passes"], color="0.4", lw=1.1, ls=":", zorder=1)


def _title(ax, fs, letter, name, formula, dy=0):
    ax.annotate(name, (0, 1), xycoords="axes fraction", xytext=(0, 19 + dy), textcoords="offset points", ha="left",
                va="baseline", fontsize=fs["label"] + 1.5)
    ax.annotate(formula, (0, 1), xycoords="axes fraction", xytext=(0, 6 + dy), textcoords="offset points", ha="left",
                va="baseline", fontsize=fs["tick"] + .5, color="0.25")
    ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-40, 19 + dy), textcoords="offset points",
                ha="left", va="baseline", fontsize=fs["label"] + 3, fontweight="bold")


def _hline(ax, fs, v, name, color, x=XMIN * 1.15, dy=.0, va="bottom"):
    ax.axhline(v, color=color, lw=1., ls=":", zorder=1)
    ax.text(x, v + dy, name, fontsize=fs["tick"], color=color if color != "k" else "0.15", va=va, ha="left")


def _edge_refs(ax, fs, S, keys):
    """reference values at the right edge (call after set_ylim): keys [(series key, line style, line width)], Bayes
    black, naive grey, each labelled B / n just outside the axes (labels nudged apart when values are close)"""
    ax.axvline(XSEP, color="0.85", lw=.8)
    marks = []
    for k, ls, lw in keys:
        for q in REFS:
            ax.plot([XB[0], XMAX], [S[q][k]] * 2, color=COL[q], lw=lw + .4, ls=ls, zorder=5, solid_capstyle="butt",
                    clip_on=False)
            marks.append([float(S[q][k]), "B" if q == "bayes" else "n", COL[q]])
    lo, hi = ax.get_ylim()
    gap = .075 * (hi - lo)
    marks.sort()
    ys = [m[0] for m in marks]
    for _ in range(50):                                                              # push close labels apart
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < gap:
                mid = (ys[i] + ys[i - 1]) / 2
                ys[i - 1], ys[i] = mid - gap / 2, mid + gap / 2
    for (v, lab, c), y in zip(marks, ys):
        ax.text(XMAX * 1.12, y, lab, fontsize=fs["tick"], color=c if c != "k" else "0.1", ha="left", va="center",
                clip_on=False)


def figure_split(S, ref):
    """td_tt_splitting.png: correct splitting after entangled steps, through training (see the docstring)"""
    plt, fs = _style()
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    pw, ph, lm, top, bot = 3.0, 2.1, .85, 1.15, 2.1
    gx = (1.15, 1.6)                                                                 # gaps after columns 1 and 2
    gy = 1.45
    cx = [lm, lm + pw + gx[0], lm + 2 * pw + gx[0] + gx[1]]
    W, H = cx[2] + pw + .3, top + 2 * ph + gy + bot
    fig = plt.figure(figsize=(W, H))
    ax = {(r_, c_): fig.add_axes([cx[c_] / W, (bot + (1 - r_) * (ph + gy)) / H, pw / W, ph / H])
          for r_ in range(2) for c_ in range(3)}
    # (a) one-step contrast
    a = ax[0, 0]
    _marks(a, ref)
    a.axhline(0, color="0.8", lw=.7)
    for r in RUNS:
        d = S[r]
        lo, hi = np.quantile(d["con_b"], [.05, .95], axis=0)
        a.fill_between(d["x"], lo, hi, color=COL[r], alpha=.22, lw=0, zorder=2)
        a.plot(d["x"], d["con"], color=COL[r], lw=1.8, zorder=3)
        a.plot(d["x"], d["con_g"], color=COL[r], lw=1., ls=DASH, zorder=3)
        if "grid_x" in d:
            a.plot(d["grid_x"], d["grid_con_all"], "o", mfc="w", mec=COL[r], mew=1.1, ms=3.6, zorder=4)
    _hline(a, fs, S["bayes"]["con"], "Bayes", "k")
    _hline(a, fs, S["naive"]["con"], "naive", "0.4", dy=-.004, va="top")
    _xaxis(a, fs)
    a.set_ylim(-.06, .335)
    if "grid_con_all" in S["bayes"]:                                                 # the circles' own references
        a.axvline(XSEP, color="0.85", lw=.8)
        for q in REFS:
            v = S[q]["grid_con_all"]
            a.plot([np.sqrt(XB[0] * XN[1])], [v], "o", mfc="w", mec=COL[q], mew=1.2, ms=4.2, zorder=5, clip_on=False)
            a.text(XMAX * 1.12, v, "B" if q == "bayes" else "n", fontsize=fs["tick"], ha="left", va="center",
                   color="0.1" if q == "bayes" else "0.4", clip_on=False)
    a.set_ylabel("grid units per step", fontsize=fs["label"])
    a.legend(handles=[Line2D([], [], color="0.3", lw=1.8, label="all near pairs"),
                      Line2D([], [], color="0.3", lw=1, ls=DASH, label="matched to the cells"),
                      Line2D([], [], color="0.3", marker="o", mfc="w", ls="none", ms=3.6,
                             label="24 streams per cell")],
             fontsize=fs["tick"] - .5, frameon=False, loc="upper left", bbox_to_anchor=(-.02, 1.03),
             handlelength=1.3, handletextpad=.5, labelspacing=.3)
    _title(a, fs, "a", "Correct splitting after entangled steps",
           r"$\langle\Delta D\rangle_{10} - \langle\Delta D\rangle_{1}$, tenths of $\bar e$ (the pair's mean rev)")
    # (b) six steps on: correct vs apart, entangled - calm third
    a = ax[0, 1]
    _marks(a, ref)
    a.axhline(0, color="0.8", lw=.7)
    for r in RUNS:
        d = S[r]
        a.plot(d["x"], d["toward6"], color=COL[r], lw=2, zorder=3)
        a.plot(d["x"], d["apart6"], color=COL[r], lw=1.3, ls=DASH, zorder=3)
    _xaxis(a, fs)
    a.set_ylim(-.14, .5)
    _edge_refs(a, fs, S, (("toward6", "-", 2), ("apart6", DASH, 1.3)))
    a.set_ylabel("grid units", fontsize=fs["label"])
    a.legend(handles=[Line2D([], [], color="0.3", lw=2, label=r"toward own cells, $\Delta D$"),
                      Line2D([], [], color="0.3", lw=1.3, ls=DASH, label=r"apart, $\Delta d$")],
             fontsize=fs["tick"] - .5, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02), handlelength=1.8,
             labelspacing=.25)
    _title(a, fs, "b", "Six steps on: toward the own cells, or just apart",
           r"entangled $-$ calm third of $\bar e$, $k = 6$ steps;  $d = \|x_i - x_j\|$")
    # (c) gradient vs overall splitting
    a = ax[0, 2]
    a.axhline(0, color="0.8", lw=.7)
    for r in RUNS:
        d = S[r]
        a.plot(d["mean"], d["con"], color=COL[r], lw=1.6, zorder=3)
        for xv, lab in ((100, "100"), (1000, "1k"), (10000, "10k")):
            k = int(np.argmin(np.abs(np.log(d["x"] / xv))))
            a.plot(d["mean"][k], d["con"][k], "o", color=COL[r], ms=5, zorder=7, mec="w", mew=.7)   # above the stars
            if xv == 100:
                continue
            if r == "trained" and xv == 10000:     # beside the Bayes star: label above-right (empty), leader on top
                a.annotate(lab, (d["mean"][k], d["con"][k]), xytext=(12, 14), textcoords="offset points",
                           fontsize=fs["tick"] - .5, color=COL[r], ha="left", va="center", zorder=8,
                           arrowprops=dict(arrowstyle="-", color=COL[r], lw=.6, shrinkA=1, shrinkB=3.5))
                continue
            off, ha = ((-5, 5), "right") if r == "trained" else ((3, -12), "left")
            a.annotate(lab, (d["mean"][k], d["con"][k]), xytext=off, textcoords="offset points",
                       fontsize=fs["tick"] - .5, color=COL[r], ha=ha)
    tx = (.035, .045)                                                                # one label, a leader to each dot
    for r in RUNS:
        k = int(np.argmin(np.abs(np.log(S[r]["x"] / 100))))
        a.annotate("", (S[r]["mean"][k], S[r]["con"][k]), xytext=tx, textcoords="data",
                   arrowprops=dict(arrowstyle="-", color="0.5", lw=.6, shrinkA=0, shrinkB=3))
    a.text(*tx, "both at 100", fontsize=fs["tick"] - .5, color="0.3", ha="center", va="bottom")
    for q in REFS:
        a.plot(S[q]["mean"], S[q]["con"], "*", color=COL[q], ms=13, zorder=6, mec="w", mew=.6)
        a.annotate(NAME[q], (S[q]["mean"], S[q]["con"]), xytext=(7, -4), textcoords="offset points",
                   fontsize=fs["tick"], color="0.1" if q == "bayes" else "0.4", va="top")
    a.set_xlim(-.01, .32), a.set_ylim(-.06, .27)
    a.set_xlabel(r"$\langle\Delta D\rangle$ of all near pairs (grid units per step)", fontsize=fs["label"],
                 labelpad=1)
    a.set_ylabel(r"$\langle\Delta D\rangle_{10} - \langle\Delta D\rangle_{1}$ (a)", fontsize=fs["label"])
    a.spines[["top", "right"]].set_visible(False)
    a.text(.97, .04, "dots: 100, 1k, 10k\ntraining episodes", transform=a.transAxes, ha="right", va="bottom",
           fontsize=fs["tick"] - .5, color="0.35", linespacing=1.2)
    _title(a, fs, "c", "Entanglement gradient vs overall splitting",
           r"$\langle\Delta D\rangle_{10} - \langle\Delta D\rangle_{1}$ vs $\langle\Delta D\rangle$ of all near pairs")
    # (d, e) heat maps; line: each environment's peak tenth, averaged over environments, with its 5-95% band
    import td_tt_gap as gap
    C = gap.boot_counts()
    halo = [pe.withStroke(linewidth=3.4, foreground="0.1")]
    norm = Normalize(0, .36)
    for c_, r in ((0, "trained"), (1, "echo")):
        a = ax[1, c_]
        d = S[r]
        a.pcolormesh(_edges(d["x"]), np.arange(.5, ND + 1), d["dec"].T, cmap="viridis", norm=norm, shading="flat",
                     rasterized=True)
        for q, (x0, x1) in (("bayes", XB), ("naive", XN)):
            a.pcolormesh([x0, x1], np.arange(.5, ND + 1), S[q]["dec"][:, None], cmap="viridis", norm=norm,
                         shading="flat", rasterized=True)
            a.text(np.sqrt(x0 * x1), ND + .6, "B" if q == "bayes" else "n", ha="center", va="bottom",
                   fontsize=fs["tick"], color="k")
            pf, pb = peak_env(S[q], C)
            xs_ = [x0 * 1.04, x1 / 1.04]
            a.fill_between(xs_, *np.percentile(pb, [5, 95]), color="w", alpha=.35, lw=0, zorder=5)
            a.plot(xs_, [pf, pf], color="w", lw=1.8, solid_capstyle="butt", zorder=6, path_effects=halo)
        pf, pb = peak_env(d, C)
        k0 = int(np.argmax(d["mean"] >= LINE_FLOOR))                                  # beliefs split at all
        assert (d["mean"][k0:] >= LINE_FLOOR).all()
        lo, hi = np.percentile(pb, [5, 95], axis=0)
        a.fill_between(d["x"][k0:], lo[k0:], hi[k0:], color="w", alpha=.35, lw=0, zorder=5)
        a.plot(d["x"][k0:], pf[k0:], color="w", lw=1.8, zorder=6, path_effects=halo)
        a.axvline(XSEP, color="w", lw=2.5)
        _marks(a, ref, heat=True)
        _xaxis(a, fs)
        a.set_ylim(.5, ND + .5)
        a.set_yticks([1, 4, 7, 10])
        a.set_ylabel(r"tenth of $\bar e$ (1: calm, 10: entangled)", fontsize=fs["label"] - .5)
        _title(a, fs, "de"[c_], f"{NAME[r]}: correct split by entanglement",
               r"one-step $\langle\Delta D\rangle$ per tenth $k$; line: $\langle\arg\max_k\,\Delta D_k\rangle_{\rm env}$",
               dy=6)
    cax = fig.add_axes([(cx[1] + pw + .15) / W, (bot + .1) / H, .1 / W, (ph - .2) / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax, ticks=[0, .1, .2, .3])
    cb.set_label(r"$\langle\Delta D\rangle$ (grid units per step)", fontsize=fs["tick"], labelpad=3)
    cb.ax.tick_params(labelsize=fs["tick"] - .5)
    # (f) profiles
    a = ax[1, 2]
    x = np.arange(1, ND + 1)
    for q, sty in (("trained", dict(lw=3.2, alpha=.45)), ("bayes", dict(lw=1.2)), ("naive", dict(lw=1.6)),
                   ("echo", dict(lw=1.6))):
        y = S[q]["dec"][-1] if q in RUNS else S[q]["dec"]
        a.plot(x, y, color=COL[q], marker="o", ms=2.5, **sty)
    k880 = int(np.argmin(np.abs(S["trained"]["x"] - ref["passes"])))
    a.plot(x, S["trained"]["dec"][k880], color=COL["trained"], lw=1.3, ls=DASH, marker="o", ms=2)
    a.text(10.3, S["trained"]["dec"][k880][-1], f"{ref['passes']:,.0f}", color=COL["trained"],
           fontsize=fs["tick"] - .5, va="center")
    a.axhline(0, color="0.8", lw=.7)
    a.set_xticks([1, 4, 7, 10]), a.set_xlim(.6, 11.5)
    a.set_ylim(-.005, .36)
    a.set_xlabel(r"tenth of $\bar e$ (1: calm, 10: entangled)", fontsize=fs["label"] - .5, labelpad=1)
    a.set_ylabel("grid units per step", fontsize=fs["label"])
    a.spines[["top", "right"]].set_visible(False)
    _title(a, fs, "f", "Profiles at the end of training",
           r"one-step $\langle\Delta D\rangle$ per tenth; dashed: Trained RNN at 880", dy=6)
    hand = [Line2D([], [], color=COL[q], lw=2.4, label=NAME[q]) for q in ("trained", "echo", "bayes", "naive")]
    hand += [Line2D([], [], color="0.4", lw=1.1, ls=":", label=f"Trained RNN passes naive ({ref['passes']:,.0f})"),
             plt.Rectangle((0, 0), 1, 1, color="0.88", label="its memory use flips by entanglement "
                           f"({ref['flip'][0]:,.0f}-{ref['flip'][1]:,.0f})")]
    fig.legend(handles=hand, loc="upper center", bbox_to_anchor=(.5, 1 - .08 / H), ncol=6, frameon=False,
               fontsize=fs["tick"] + .5, handlelength=1.8, columnspacing=1.3)
    nul = max(np.abs(S[r]["con_null"]).max() for r in RUNS)
    fig.text(lm / W, 1.27 / H,
             "Near pairs (as in the end-of-training splitting figure): at each step $t_0$ = 2..24, every particle "
             "(posterior means $x = (E[r_0], E[r_1])$) with its nearest particle of another cell, same environment "
             "(equidistant partners: all, weight 1/n each).\n"
             r"$D = \|x_i - c_j\| - \|x_i - c_i\| + \|x_j - c_i\| - \|x_j - c_j\|$ ($c$: true cells), "
             r"$\Delta D$ its change over the next step (b: six steps), $d = \|x_i - x_j\|$.  "
             r"$\bar e = (e_i + e_j)/2$ at $t_0 + 1$, $e$ = rev, the MEAN of the two variables' entanglement" "\n"
             "(the pair's split is a joint 2-D quantity); tenths and thirds within (environment, step); per environment "
             "averaged over $t_0$, then over environments. Matched: tenths within thirds of the pair's distance to its "
             "cells.\nShuffled entanglement: "
             r"$|$a$|$ $\leq$ %.3f at every checkpoint.  Bands: 5-95%% over 1,000 resamples of the 50 environments.  "
             "a: open circles at the right = the 24-stream circles' own Bayes (B) and naive (n).\n"
             "d, e line: per environment the tenth where its own profile (3-tenth running mean) peaks, then the mean "
             "over the 50 environments (band: 5-95%%; B, n: the same for Bayes, naive); drawn where "
             r"$\langle\Delta D\rangle \geq$ .005." "\n"
             "Data: 10,000 probe episodes = 50 environments x 100 cells x 2 streams (200 particles per environment and "
             "step), 114 checkpoints per network; circles: 50 environments x 100 cells x 24 fresh streams at 19 "
             "checkpoints." % nul,
             fontsize=fs["tick"] - .5, color="0.3", va="top", linespacing=1.5)
    path = fig_path_("td_tt_splitting.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def timing_rows(S, ref, tm):
    """the rows of the timing panel: (label, list of (crossing dict, marker style), extra span or None)"""
    T_ = tm["trained"]
    up, down = plateau(S["trained"]["x"], S["trained"]["apart6"])
    rows = [("split gradient half-way (splitting figure a)", [(T_["grad_half"], "o")], None),
            ("distance-matched gradient half-way (splitting figure a)", [(T_["matched_half"], "o")], None),
            ("toward own cells, six steps, half-way (splitting figure b)", [(T_["toward_half"], "o")], None),
            ("over-splitting plateau, six steps apart (splitting figure b)", [], (float(up[0]), float(down[0]))),
            ("over-splitting ends: half-way back (splitting figure b)", [(T_["apart_back"], "o")], None),
            ("fold index half-way (c)", [(T_["fold_half"], "o")], None),
            ("opposite corners overtake edges: fold index 0 (c)", [(T_["fold_zero"], "o")], None),
            ("opposite corners half-way (c)", [(T_["corners_half"], "o")], None)]
    if "grid_half" in T_:
        rows.append(("grid folds: 25 / 50 / 75% of the drop gone (d)", [(T_["grid_half"], "o")],
                     (T_["grid_25"]["x"], T_["grid_75"]["x"])))
    rows += [("goal accuracy passes naive's level", [(T_["acc_naive"], "o")], None),
             ("goal accuracy half-way to Bayes's", [(T_["acc_half"], "o")], None)]
    return rows


def figure_fold(S, ref):
    """td_tt_splitting_foldback.png: the fold-back of the estimate distances through training, and the timing of every
    half-way point (see the docstring)"""
    plt, fs = _style()
    from matplotlib.lines import Line2D
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    tm = timing(S, ref)
    rows = timing_rows(S, ref, tm)
    sq, pw, ph, lm, top, bot = 2.55, 3.0, 2.1, .85, 1.1, 1.78
    gx, gy = 1.75, 1.3
    th = .27 * len(rows) + .2                                                        # timing panel height
    cx = [lm, lm + pw + gx]
    W = cx[1] + pw + .35
    H = top + sq + gy + ph + gy + th + bot
    yb_ = bot + th + gy                                                              # bottom of row 2
    fig = plt.figure(figsize=(W, H))
    tnorm = LogNorm(X0, 4e4)
    for c_, r in enumerate(RUNS):
        a = fig.add_axes([cx[c_] / W, (yb_ + ph + gy) / H, sq / W, sq / H])
        d = S[r]
        cmap = plt.get_cmap(CMAPS[r])
        for k in range(len(d["x"])):
            a.plot(d["xs"], d["curve"][k], color=cmap(.22 + .78 * tnorm(d["x"][k])), lw=.9, zorder=2)
        a.plot(S["naive"]["xs"], S["naive"]["curve"], color="0.4", lw=2, ls=DASH, zorder=4)
        a.plot(S["bayes"]["xs"], S["bayes"]["curve"], color="k", lw=1.6, ls=DASH, zorder=5)
        a.plot([0, 13], [0, 13], color="0.6", lw=.8, ls=":", zorder=1)
        a.set_xlim(0, 13), a.set_ylim(0, 13), a.set_aspect("equal")
        a.set_xticks([0, 4, 8, 12]), a.set_yticks([0, 4, 8, 12])
        a.set_xlabel("true distance of the two cells", fontsize=fs["label"], labelpad=1)
        a.set_ylabel("estimate distance at T", fontsize=fs["label"])
        a.spines[["top", "right"]].set_visible(False)
        a.legend(handles=[Line2D([], [], color="k", lw=1.6, ls=DASH, label="Bayes"),
                          Line2D([], [], color="0.4", lw=2, ls=DASH, label="naive"),
                          Line2D([], [], color="0.6", lw=.8, ls=":", label="estimate = true")],
                 fontsize=fs["tick"] - .5, frameon=False, loc="upper left", handlelength=2.2, labelspacing=.25)
        _title(a, fs, "ab"[c_], f"{NAME[r]}: cells differing in both latents",
               r"$\langle\|x_i - x_j\|\rangle$ at T per true $\|c_i - c_j\|$ (1-unit bins)")
        cax = fig.add_axes([(cx[c_] + sq + .15) / W, (yb_ + ph + gy + .1) / H, .09 / W, (sq - .2) / H])
        sm = plt.cm.ScalarMappable(norm=tnorm, cmap=LinearSegmentedColormap.from_list(
            "c", [cmap(.22 + .78 * v) for v in np.linspace(0, 1, 64)]))
        cb = fig.colorbar(sm, cax=cax, ticks=[X0, 100, 1000, 10000])
        cb.ax.set_yticklabels(["0", "100", "1k", "10k"])
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=fs["tick"] - .5)
        cb.set_label("training episodes (line colour)", fontsize=fs["tick"], labelpad=2)
    # (c) corners vs edges through training
    a = fig.add_axes([cx[0] / W, yb_ / H, pw / W, ph / H])
    _marks(a, ref)
    for r in RUNS:
        a.plot(S[r]["x"], S[r]["corners"], color=COL[r], lw=2, zorder=3)
        a.plot(S[r]["x"], S[r]["edges"], color=COL[r], lw=1.3, ls=DASH, zorder=3)
    _xaxis(a, fs)
    a.set_ylim(0, 13)
    _edge_refs(a, fs, S, (("corners", "-", 2), ("edges", DASH, 1.3)))
    a.set_ylabel("estimate distance at T", fontsize=fs["label"])
    a.legend(handles=[Line2D([], [], color="0.3", lw=2, label="corners (true 12.7)"),
                      Line2D([], [], color="0.3", lw=1.3, ls=DASH, label="edges (true 9)")],
             fontsize=fs["tick"] - .5, frameon=False, loc="upper left", handlelength=1.8, labelspacing=.25)
    _title(a, fs, "c", "Opposite corners vs opposite edges",
           r"$\langle\|x_i - x_j\|\rangle$ for $(|\Delta r_0|, |\Delta r_1|) = (9, 9)$ and $(9, 0), (0, 9)$")
    # (d) grid folds (td_summary_data.grids())
    a = fig.add_axes([cx[1] / W, yb_ / H, pw / W, ph / H])
    _marks(a, ref)
    G = ref["grids"]
    for r in RUNS:
        a.plot(G[f"x_{r}"], G[f"where_naive_{r}"], color=COL[r], lw=2, marker="o", ms=3, zorder=3)
        a.plot(G[f"x_{r}"], G[f"elsewhere_{r}"], color=COL[r], lw=1.3, ls=DASH, marker="o", ms=2.5, zorder=3)
    _hline(a, fs, float(G["bayes_where_naive"]), "Bayes, where naive folds", "k", dy=1)
    _xaxis(a, fs)
    a.set_ylim(0, 85)
    a.set_ylabel("% of the grid folded at T", fontsize=fs["label"])
    leg = a.legend(handles=[Line2D([], [], color="0.3", lw=2, marker="o", ms=3, label="naive's folded triangles"),
                            Line2D([], [], color="0.3", lw=1.3, ls=DASH, marker="o", ms=2.5, label="the other triangles")],
                   fontsize=fs["tick"] - .5, frameon=False, loc="upper left", bbox_to_anchor=(.01, 1.02),
                   handlelength=1.3, handletextpad=.5, labelspacing=.25, borderaxespad=.2)
    fig.canvas.draw()                                       # the legend must end left of the flip band and 880 line
    bb = leg.get_window_extent().transformed(a.transData.inverted())
    if bb.x1 >= min(ref["flip"][0], ref["passes"]):
        print(f"WARNING: panel d legend reaches x = {bb.x1:,.0f} (flip band from {ref['flip'][0]:,.0f})")
    _title(a, fs, "d", "Belief-grid folds at T",
           r"% of the 162 grid triangles with signed area $A_\triangle < 0$")
    # (e) timing of the half-way points
    lx0 = 3.75                                                                       # room for the row labels
    a = fig.add_axes([lx0 / W, bot / H, (W - lx0 - .55) / W, th / H])
    a.axvspan(*ref["flip"], color="0.88", lw=0, zorder=0)
    a.axvline(ref["passes"], color="0.4", lw=1.1, ls=":", zorder=1)
    for i, (lab, marks, span) in enumerate(rows):
        y = len(rows) - 1 - i
        if span is not None:
            dy = .22 if marks else 0
            a.plot(span, [y + dy] * 2, color=COL["trained"], lw=1.6, solid_capstyle="butt", zorder=3)
            a.plot(span, [y + dy] * 2, "|", color=COL["trained"], ms=7, mew=1.6, zorder=3)
        for c, m in marks:
            hi = c["hi"] if np.isfinite(c["hi"]) else 5e4
            a.plot([c["lo"], hi], [y, y], color=COL["trained"], lw=4.5, alpha=.35, solid_capstyle="butt", zorder=2)
            a.plot([c["x"]], [y], "o", color=COL["trained"], mfc="w" if m == "open" else COL["trained"], ms=5.5,
                   mew=1.3, zorder=4)
        a.text(-.01, y, lab, transform=a.get_yaxis_transform(), ha="right", va="center", fontsize=fs["tick"],
               color="0.15", linespacing=1.1)
    a.text(9.8e3, len(rows) - .75, "the intervals around the grey\nband overlap: these events\ncannot be ordered",
           ha="right", va="top", fontsize=fs["tick"] + .5, color="0.2", linespacing=1.3)
    a.text(9.8e3, .5, "after the events around the band,\nexcept the distance-matched gradient\n"
           "and the grid folds' 75% point\n(intervals overlap)", ha="right", va="center", fontsize=fs["tick"],
           color="0.2", linespacing=1.3)
    a.text(760, 1, f"{tm['trained']['acc_naive']['x']:,.0f}: log-interpolated; dotted line:\nthe first checkpoint "
           f"past naive ({ref['passes']:,.0f})", ha="right", va="center", fontsize=fs["tick"] - .5, color="0.3",
           linespacing=1.25)
    a.set_xscale("log")
    a.set_xlim(100, 1e4)
    a.set_xticks([100, 300, 1000, 3000, 10000])
    a.set_xticklabels(["100", "300", "1k", "3k", "10k"])
    a.minorticks_off()
    a.set_ylim(-.6, len(rows) - .4)
    a.set_yticks([])
    a.spines[["top", "right", "left"]].set_visible(False)
    a.set_xlabel("training episodes", fontsize=fs["label"], labelpad=1)
    x_t = -lx0 * 72 + lm * 72                                                        # titles start at the margin
    a.annotate("Half-way points of the Trained RNN", (0, 1), xycoords="axes fraction", xytext=(x_t, 32),
               textcoords="offset points", ha="left", va="baseline", fontsize=fs["label"] + 1.5)
    a.annotate(r"$t_{1/2}$ = stays past the naive-Bayes midpoint $(y_{\rm naive} + y_{\rm Bayes})/2$ to the end of "
               "training (log-interpolated), unless the row names another level",
               (0, 1), xycoords="axes fraction", xytext=(x_t, 19), textcoords="offset points", ha="left",
               va="baseline", fontsize=fs["tick"] + .5, color="0.25")
    a.annotate("dot: the full sample; bar: 5-95% over 1,000 resamples of the 50 environments; thin bars: spans "
               "(see below)",
               (0, 1), xycoords="axes fraction", xytext=(x_t, 6), textcoords="offset points", ha="left",
               va="baseline", fontsize=fs["tick"] + .5, color="0.25")
    a.annotate("e", (0, 1), xycoords="axes fraction", xytext=(x_t - 40, 32), textcoords="offset points",
               ha="left", va="baseline", fontsize=fs["label"] + 3, fontweight="bold")
    hand = [Line2D([], [], color=COL[q], lw=2.4, label=NAME[q]) for q in ("trained", "echo", "bayes", "naive")]
    hand += [Line2D([], [], color="0.4", lw=1.1, ls=":", label=f"Trained RNN passes naive ({ref['passes']:,.0f})"),
             plt.Rectangle((0, 0), 1, 1, color="0.88", label="memory-use flip "
                           f"({ref['flip'][0]:,.0f}-{ref['flip'][1]:,.0f})")]
    fig.legend(handles=hand, loc="upper center", bbox_to_anchor=(.5, 1 - .06 / H), ncol=3, frameon=False,
               fontsize=fs["tick"] + .5, handlelength=1.8, columnspacing=1.3)
    fig.text(lm / W, (bot - .47) / H,
             "a-c: every pair of particles (posterior means $x = (E[r_0], E[r_1])$ at T = 30) of an environment, grouped "
             r"by the true separation of their cells $(|\Delta r_0|, |\Delta r_1|)$;" "\ndistances per environment, "
             "then averaged over environments. 10,000 probe episodes = 50 environments x 100 cells x 2 streams "
             "(19,900 pairs per environment),\n114 checkpoints per network. d: each cell's mean position over 24 fresh "
             "streams (50 environments x 100 cells, 19 checkpoints); the grid of these 100 points\ncut into 162 "
             r"triangles, folded where the orientation flips ($A_\triangle < 0$). e: plateau (thin bar) = 3-checkpoint "
             r"running mean $\geq$ 80% of its maximum;" "\ngrid folds (thin bar): 25 / 50 / 75% of the drop from the "
             "initial network to the end (the landmark), first covered.\nThe ESN reaches none of the Trained RNN's "
             "Bayes-side levels; it over-splits like naive from about 1,100 and never returns (splitting figure b),\n"
             "and its corners hover at naive's level (c).",
             fontsize=fs["tick"] - .5, color="0.3", va="top", linespacing=1.5)
    path = fig_path_("td_tt_splitting_foldback.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_path_(name):
    from main.utils import fig_path
    return fig_path(name)


def _fmt(c):
    """a crossing dict -> 'x [lo, hi]' (training episodes; 'never' where it does not cross)"""
    f = lambda v: "never" if not np.isfinite(v) else f"{v:,.0f}"                     # noqa: E731
    s = f"{f(c['x'])} [{f(c['lo'])}, {f(c['hi'])}]"
    return s + (f" (never in {100 * c['never']:.0f}% of resamples)" if 0 < c["never"] < 1 else "")


def _pd(b1, b2):
    """paired bootstrap difference of two crossings (log10 of their ratio): 5-95% interval and share > 0"""
    ok = np.isfinite(b1) & np.isfinite(b2)
    r = np.log10(b1[ok] / b2[ok])
    lo, hi = np.quantile(r, [.05, .95])
    return f"x{10 ** np.median(r):.2f} [x{10 ** lo:.2f}, x{10 ** hi:.2f}], later in {100 * np.mean(r > 0):.0f}%"


def peak_tenth(dec, axis=-1):
    """the tenth (1-10) where the profile's 3-tenth running mean (2 tenths at the ends) peaks"""
    y = np.moveaxis(np.asarray(dec, np.float64), axis, -1)
    pad = np.concatenate([y[..., :1] * np.nan, y, y[..., -1:] * np.nan], -1)
    rm = np.nanmean(np.stack([pad[..., :-2], pad[..., 1:-1], pad[..., 2:]]), 0)
    return np.argmax(rm, -1) + 1


LINE_FLOOR = .005    # heat-map line drawn where the mean one-step dD of all near pairs >= this (grid units per step)


def peak_env(d, C=None):
    """the heat-map line (d, e): each environment's peak tenth (peak_tenth of its OWN profile), then the mean over
    the 50 environments (the statistic per environment, averaged after) -> (full-sample mean (..,), resampled means
    (NB, ..)); resamples = the shared td_tt_gap.boot_counts() (count-weighted means of the per-environment values;
    with every count 1 this is the full-sample mean)"""
    import td_tt_gap as gap
    return gap.env_mean(np.moveaxis(peak_tenth(d["dec_env"]).astype(np.float64), -1, 0), C)


def mean_tenth_env(dec_env):
    """the alternative heat-map summary, per environment (.., NE): the dD-weighted mean tenth
    sum_k k max(dD_k, 0) / sum_k max(dD_k, 0) (NaN where no tenth splits correctly)"""
    w = np.clip(np.asarray(dec_env, np.float64), 0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w * np.arange(1, ND + 1)).sum(-1) / w.sum(-1)


def paired_versions(d, B):
    """the heat-map line's three per-environment versions, agent minus Bayes paired by environment, (K, NE) each: the
    drawn peak tenth (peak_env), the dD-weighted mean tenth (mean_tenth_env; NaN where no tenth splits correctly) and
    the peak tenth per (environment, step) averaged over the steps (series()'s peak_step_env)"""
    return {"drawn line (peak tenth)": (peak_tenth(d["dec_env"]) - peak_tenth(B["dec_env"])).astype(np.float64),
            "dD-weighted mean tenth": mean_tenth_env(d["dec_env"]) - mean_tenth_env(B["dec_env"]),
            "peak per (environment, step)": d["peak_step_env"] - B["peak_step_env"]}


def _paired(v):
    """mean and SE over environments (the last axis) of a paired per-environment difference (NaN-safe)"""
    v = np.asarray(v, np.float64)
    return np.nanmean(v, -1), np.nanstd(v, -1, ddof=1) / np.sqrt(np.isfinite(v).sum(-1))


def _ck_window(y, win=2):
    """the mean over checkpoints k - win .. k + win (truncated at the ends): only for the record of the retracted
    heat-map line (the argmax of the environment-mean profile, averaged over +-2 checkpoints)"""
    y = np.asarray(y, np.float64)
    return np.stack([y[max(0, k - win):k + win + 1].mean(0) for k in range(len(y))])


def _runs(x, v, start):
    """'value at x0-x1' runs of a per-checkpoint integer series from x >= start (the retracted line's record)"""
    k = np.flatnonzero(x >= start)
    out, j = [], k[0]
    for i in list(k[1:]) + [None]:
        if i is None or v[i] != v[j]:
            last = (i if i is not None else len(x)) - 1
            out.append(f"{v[j]:.0f} at {x[j]:,.0f}" + (f"-{x[last]:,.0f}" if last > j else ""))
            j = i
    return "; ".join(out)


def numbers(S=None, ref=None):
    """the docstring's numbers, from the caches"""
    S = S if S is not None else series()
    ref = ref if ref is not None else references()
    tm = timing(S, ref)
    B, N, Tr, E = S["bayes"], S["naive"], S["trained"], S["echo"]
    at = lambda d, k, xv: d[k][int(np.argmin(np.abs(d["x"] - xv)))]                    # noqa: E731
    ix = lambda d, xv: int(np.argmin(np.abs(d["x"] - xv)))                             # noqa: E731
    print(f"passes naive (landmark): between {ref['passes_prev']:.0f} and {ref['passes']:.0f} (accuracy at T "
          f"{ref['acc_pass'][0]:.3f} / {ref['acc_pass'][1]:.3f}, naive {ref['naive_acc']:.3f}, Bayes "
          f"{ref['bayes_acc']:.3f}); memory-use flip {ref['flip'][0]:,.0f}-{ref['flip'][1]:,.0f}")
    print("TIES (share of particles with >= 2 equidistant nearest other-cell partners), probes t0 = 2 / 3 / 4, all t0:")
    for q, d in (("Bayes", B), ("naive", N), ("Trained end", {"tied": Tr["tied"][-1]}),
                 ("ESN end", {"tied": E["tied"][-1]})):
        print(f"   {q:12s} {np.round(d['tied'][:3], 3)} all {d['tied'].mean():.3f}")
    if "grid_tied_all" in B:
        for q, d in (("Bayes", B), ("naive", N)):
            print(f"   24 streams {q}: {np.round(d['grid_tied_all'][:3], 3)} all {d['grid_tied_all'].mean():.3f}")
        print(f"   24 streams Trained end: {np.round(Tr['grid_tied_all'][-1][:3], 3)}")
    print("END OF TRAINING")
    for q, d in (("Bayes", B), ("naive", N), ("Trained end", {k: v[-1] for k, v in Tr.items() if np.ndim(v)}),
                 ("ESN end", {k: v[-1] for k, v in E.items() if np.ndim(v)})):
        print(f"   {q:12s} tenths {np.round(d['dec'], 3)} gradient {float(d['con']):+.4f} matched "
              f"{float(d['con_g']):+.4f} mean {float(d['mean']):.3f} early/late {float(d['con_early']):+.3f} / "
              f"{float(d['con_late']):+.3f} | apart6 {float(d['apart6']):.3f} toward6 {float(d['toward6']):+.3f} | "
              f"corners {float(d['corners']):.2f} edges {float(d['edges']):.2f} | acc {float(d['acc']):.3f} | peak "
              f"tenth {int(peak_tenth(d['dec']))}")
    for q, d in (("Bayes", B), ("naive", N)):
        print(f"   {q} six-step raw thirds (entangled / calm): apart {np.round(d['six_raw']['split_lag'], 3)} toward "
              f"{np.round(d['six_raw']['corr_lag'], 3)}; cell means corners / edges {d['corners_cm']:.2f} / "
              f"{d['edges_cm']:.2f}")
    print(f"   Trained end cell means {Tr['corners_cm'][-1]:.2f} / {Tr['edges_cm'][-1]:.2f}; at 880 "
          f"{at(Tr, 'corners_cm', 880):.2f} / {at(Tr, 'edges_cm', 880):.2f}; ESN end {E['corners_cm'][-1]:.2f} / "
          f"{E['edges_cm'][-1]:.2f}")
    k = ix(Tr, 880)
    y, yb = Tr["dec"][k], B["dec"]
    print(f"PROFILE at 880: {np.round(y, 3)}; height of tenths 5-10 {y[4:].mean():.3f} vs Bayes {yb[4:].mean():.3f} "
          f"({y[4:].mean() / yb[4:].mean():.2f}); rise tenth 1 -> tenths 5-10 {y[4:].mean() - y[0]:.3f} vs "
          f"{yb[4:].mean() - yb[0]:.3f} ({(y[4:].mean() - y[0]) / (yb[4:].mean() - yb[0]):.2f}); shape "
          f"(rise / own plateau) {np.round((y - y[0]) / (y[4:].mean() - y[0]), 2)} vs Bayes "
          f"{np.round((yb - yb[0]) / (yb[4:].mean() - yb[0]), 2)}")
    for r, d in (("Trained", Tr), ("ESN", E)):
        print(f"{r.upper()}: gradient end {d['con'][-1]:+.4f} +- {d['con_se'][-1]:.4f}; halves r "
              f"{np.corrcoef(d['con_A'], d['con_B'])[0, 1]:.2f}; null max |{np.abs(d['con_null']).max():.3f}| "
              f"(at {d['x'][int(np.argmax(np.abs(d['con_null'])))]:.0f}); range {d['con'].min():+.4f} to "
              f"{d['con'].max():+.4f}; up to 160: {d['con'][d['x'] <= 160].min():+.4f} to "
              f"{d['con'][d['x'] <= 160].max():+.4f}")
        lo, hi = np.quantile(d["con_b"], [.05, .95], axis=0)
        pf = peak_env(d)[0]
        for xv in (20, 80, 160, 180, 240, 400, 480, 560, 640, 720, 800, 880, 960, 1040, 1120, 1200, 1280, 1600, 1920,
                   2400, 2800, 3000, 5400, 39996):
            j = ix(d, xv)
            print(f"   {xv:6d}: gradient {d['con'][j]:+.4f} [{lo[j]:+.3f}, {hi[j]:+.3f}] matched {d['con_g'][j]:+.4f} "
                  f"early {d['con_early'][j]:+.3f} mean {d['mean'][j]:.3f} | apart6 {d['apart6'][j]:.3f} toward6 "
                  f"{d['toward6'][j]:+.3f} | corners {d['corners'][j]:.2f} edges {d['edges'][j]:.2f} fold index "
                  f"{d['fold_idx'][j]:+.2f} | acc {d['acc'][j]:.3f} | peak tenth (per environment, mean) {pf[j]:.2f}")
        for key, c in tm[r.lower() if r == "Trained" else "echo"].items():
            print(f"   crossing {key:14s} level {c.get('level', np.nan):+.3f}: {_fmt(c)}")
        k = int(np.argmax(d["apart6"]))
        print(f"   six-step apart max {d['apart6'][k]:.3f} at {d['x'][k]:.0f}; over 220-1,120 "
              f"{d['apart6'][(d['x'] >= 220) & (d['x'] <= 1120)].min():.3f}-"
              f"{d['apart6'][(d['x'] >= 220) & (d['x'] <= 1120)].max():.3f}; toward6 max {d['toward6'].max():+.3f} at "
              f"{d['x'][int(np.argmax(d['toward6']))]:.0f}; fold index min {d['fold_idx'].min():+.2f} at "
              f"{d['x'][int(np.argmin(d['fold_idx']))]:.0f}")
        up, down = plateau(d["x"], d["apart6"])
        ub, db = plateau(d["x"], d["apart6_b"])
        print(f"   over-splitting plateau (3-checkpoint running mean >= 80% of its max): {up[0]:,.0f} "
              f"[{np.nanquantile(ub, .05):,.0f}, {np.nanquantile(ub, .95):,.0f}] to {down[0]:,.0f} "
              f"[{np.nanquantile(db, .05):,.0f}, {np.nanquantile(db, .95):,.0f}]")
        c, lev, x_ = d["corners"], N["corners"], d["x"]
        late, mid = x_ >= 1500, (x_ >= 1500) & (x_ < 2400)
        print(f"   corners from 1,500 on: {c[late].min():.2f}-{c[late].max():.2f} (naive {lev:.3f}); above naive's level "
              f"at {np.mean(c[late] > lev):.0%} of the {late.sum()} checkpoints from 1,500; 1,500-2,300: "
              f"{c[mid].min():.2f}-{c[mid].max():.2f}; first >= 3.9 at {x_[int(np.argmax(c >= 3.9))]:.0f}; first above "
              f"naive's level at the {x_[int(np.argmax(c > lev))]:,.0f} checkpoint")
        kl = int(np.flatnonzero(c <= lev)[-1]) if (c <= lev).any() else -1
        if kl < len(c) - 1:
            print(f"   corners above naive's level at every checkpoint from {x_[kl + 1]:,.0f} (the last one not above: "
                  f"{x_[kl]:,.0f}, {c[kl]:.2f}), by {(c[kl + 1:] - lev).min():+.3f} to {(c[kl + 1:] - lev).max():+.3f}")
        print(f"   six-step apart vs its midpoint: share of resamples whose first crossing comes before 200 training "
              f"episodes {np.mean(tm[r.lower() if r == 'Trained' else 'echo']['apart_up']['b'] < 200):.1%}")
        if "grid_con_all" in d:
            g = d["grid_con_all"]
            print(f"   24 vs 2 streams per cell: max |difference| {np.abs(g - d['grid_con_two']).max():.3f}, r "
                  f"{np.corrcoef(g, d['grid_con_two'])[0, 1]:.3f}; 24 streams at 880 {g[np.argmin(np.abs(d['grid_x'] - 880))]:+.3f}"
                  f" (2 of them {d['grid_con_two'][np.argmin(np.abs(d['grid_x'] - 880))]:+.3f}), end {g[-1]:+.3f} (2: "
                  f"{d['grid_con_two'][-1]:+.3f}); matched 24 end {d['grid_con_g_all'][-1]:+.3f}")
    print(f"NULL (shuffled entanglement) gradient: Bayes {B['con_null']:+.4f}, naive {N['con_null']:+.4f}")
    import td_tt_gap as gap
    C = gap.boot_counts()
    print("HEAT-MAP LINE (d, e): each environment's peak tenth (3-tenth running mean of its own profile), then the mean "
          "over the 50 environments; [5-95%] over the shared resamples (td_tt_gap.boot_counts()); w = the alternative, "
          "the dD-weighted mean tenth per environment, then averaged")
    lines = {}
    for q, d in (("bayes", B), ("naive", N)):
        pf, pb = peak_env(d, C)
        lines[q] = pf, pb
        pk = peak_tenth(d["dec_env"])
        print(f"   {NAME[q]}: {float(pf):.2f} [{np.percentile(pb, 5):.2f}, {np.percentile(pb, 95):.2f}] (+- "
              f"{pk.std(ddof=1) / np.sqrt(NE):.2f} SE); w {np.nanmean(mean_tenth_env(d['dec_env'])):.2f}; argmax of the "
              f"environment-mean profile {int(peak_tenth(d['dec']))}")
    for r, d in (("trained", Tr), ("echo", E)):
        pf, pb = peak_env(d, C)
        lo, hi = np.percentile(pb, [5, 95], axis=0)
        wt = mean_tenth_env(d["dec_env"])
        w = np.nanmean(wt, -1)
        k0 = int(np.argmax(d["mean"] >= LINE_FLOOR))
        print(f"   {NAME[r]}: drawn from {d['x'][k0]:,.0f} (mean one-step dD {d['mean'][k0]:.4f}; before it at most "
              f"{d['mean'][:k0].max():.4f}); the initial network: {pf[0]:.2f} (w {w[0]:.2f}, undefined in "
              f"{int(np.isnan(wt[0]).sum())} environments)")
        print("      " + "  ".join(f"{d['x'][j]:,.0f}: {pf[j]:.2f} [{lo[j]:.2f}, {hi[j]:.2f}] w {w[j]:.2f}" for j in
                                    [ix(d, v) for v in (20, 60, 120, 160, 240, 400, 560, 720, 800, 880, 1040, 1600, 2000,
                                                        2400, 5400, 10000, 20000, 39996)]))
        for a_, b_ in ((k0, ix(d, 160)), (ix(d, 880), len(pf) - 1), (ix(d, 1500), len(pf) - 1)):
            print(f"      {d['x'][a_]:,.0f}-{d['x'][b_]:,.0f}: line {pf[a_:b_ + 1].min():.2f}-{pf[a_:b_ + 1].max():.2f}; "
                  f"w {w[a_:b_ + 1].min():.2f}-{w[a_:b_ + 1].max():.2f}")
        lev, lev_b = (lines["bayes"][0] + lines["naive"][0]) / 2, (lines["bayes"][1] + lines["naive"][1]) / 2
        xb = cross_last(d["x"], pb, lev_b)
        print(f"      past the naive-Bayes midpoint ({lev:.2f}) for good from "
              f"{_fmt(dict(x=cross_last(d['x'], pf, lev), **dict(zip(('lo', 'hi', 'never'), ci(xb)))))} "
              f"(shared resamples)")
        dif = peak_tenth(d["dec_env"]) - peak_tenth(B["dec_env"])                    # paired by environment
        print("      minus Bayes, paired by environment (mean +- SE): " + "; ".join(
            f"{d['x'][j]:,.0f} {dif[j].mean():+.2f} +- {dif[j].std(ddof=1) / np.sqrt(NE):.2f}" for j in
            [ix(d, v) for v in (880, 1600, 2400, 5400, 9600, 20000, 39996)]))
        levs = {"drawn line (peak tenth)": (peak_tenth(B["dec_env"]), peak_tenth(N["dec_env"])),
                "dD-weighted mean tenth": (mean_tenth_env(B["dec_env"]), mean_tenth_env(N["dec_env"])),
                "peak per (environment, step)": (B["peak_step_env"], N["peak_step_env"])}
        late = d["x"] >= 880
        xl = d["x"][late]
        w1, w2, w3 = xl <= 2000, (xl > 2000) & (xl < 3000), xl >= 3000
        shown = [int(np.argmin(np.abs(xl - v_)))
                 for v_ in (880, 1040, 1600, 2000, 2400, 2800, 5400, 9600, 20000, 39996)]
        print("      the three per-environment versions from 880, minus Bayes paired by environment (mean +- SE; "
              "'> 2 SE' = |mean| > 2 SE at a checkpoint; 'averaged' = per environment over the window's checkpoints, "
              "then mean +- SE over environments):")
        for name, v in paired_versions(d, B).items():
            v = v[late]
            mm, ss = _paired(v)
            below, above = mm < -2 * ss, mm > 2 * ss
            av = lambda s: _paired(np.nanmean(v[s], 0))                                # noqa: E731
            print(f"         {name} (Bayes {np.mean(levs[name][0]):.2f}, naive {np.mean(levs[name][1]):.2f}): "
                  f"paired SE {ss.min():.2f}-{ss.max():.2f} from 880, {ss[w1].min():.2f}-{ss[w1].max():.2f} over "
                  "880-2,000")
            print("            " + "; ".join(f"{xl[j]:,.0f} {mm[j]:+.2f} +- {ss[j]:.2f}" for j in shown))
            ups = ", ".join(f"{xl[j]:,.0f} ({mm[j]:+.3f} +- {ss[j]:.3f})" for j in np.flatnonzero(above))
            print(f"            > 2 SE below Bayes at {below[w1].sum()} of the {w1.sum()} checkpoints over 880-2,000, "
                  f"{below[w2].sum()} of {w2.sum()} over {xl[w2][0]:,.0f}-{xl[w2][-1]:,.0f}, {below[w3].sum()} of "
                  f"{w3.sum()} from 3,000 (the last at {xl[below][-1] if below.any() else np.nan:,.0f}); > 2 SE "
                  f"above Bayes: {ups or 'none'}")
            print("            averaged: " + "; ".join(
                f"{xl[s][0]:,.0f}-{xl[s][-1]:,.0f} {av(s)[0]:+.2f} +- {av(s)[1]:.2f}"
                for s in (w1, w2, w3 & (xl < 5000), xl >= 5000)))
        b1 = peak_env(d, np.ones((1, NE)))[1][0]
        print(f"      check: every count 1 reproduces the drawn line (max |diff| {np.abs(b1 - pf).max():.1e}); it is the "
              f"mean of the per-environment peaks (max |diff| {np.abs(pf - peak_tenth(d['dec_env']).mean(-1)).max():.1e})")
        old = peak_tenth(_ck_window(d["dec"]))
        print(f"      RETRACTED line (argmax of the environment-mean profile, +-2 checkpoints), from 720: "
              f"{_runs(d['x'], old, 720)}")
    T_ = tm["trained"]
    print("PAIRED (same resamples) ratio of crossing times, Trained RNN (supplementary; the ordering rule is disjoint "
          "5-95% intervals):")
    for k1, k2 in (("grad_half", "fold_half"), ("grad_half", "acc_naive"), ("fold_half", "acc_naive"),
                   ("grad_half", "grid_half"), ("fold_half", "grid_half"), ("matched_half", "grad_half"),
                   ("early_half", "grad_half"), ("fold_zero", "grad_half"), ("acc_half", "grad_half"),
                   ("acc_half", "matched_half")):
        if k1 in T_ and k2 in T_:
            print(f"   {k1} / {k2}: {_pd(T_[k1]['b'], T_[k2]['b'])}")
    print("ORDER by the marginal rule (disjoint 5-95% intervals) against the accuracy half-way "
          f"{_fmt(T_['acc_half'])}:")
    for k in ("grad_half", "early_half", "late_half", "matched_half", "fold_half", "fold_zero", "corners_half",
              "acc_naive", "grid_half", "grid_75", "apart_back"):
        c = T_[k]
        rel = ("before" if c["hi"] < T_["acc_half"]["lo"] else "after" if c["lo"] > T_["acc_half"]["hi"] else
               "NOT ORDERED (intervals overlap or touch)")
        print(f"   {k:13s} {_fmt(c):34s} {rel}")
    ab = T_["apart_back"]
    ub = plateau(Tr["x"], Tr["apart6_b"])[0]
    start = dict(x=float(plateau(Tr["x"], Tr["apart6"])[0][0]), lo=float(np.nanquantile(ub, .05)),
                 hi=float(np.nanquantile(ub, .95)), never=float(np.mean(np.isnan(ub))))
    print(f"ORDER by the marginal rule against the over-splitting return (back on Bayes's side) {_fmt(ab)}:")
    events = ("grad_half", "early_half", "late_half", "matched_half", "toward_half", "fold_half", "fold_zero",
              "corners_naive", "corners_half", "acc_naive", "acc_half", "grid_half", "grid_75")
    for k, c in [(k, T_[k]) for k in events] + [("plateau start", start)]:
        rel = ("before" if c["hi"] < ab["lo"] else "after" if c["lo"] > ab["hi"] else
               "NOT ORDERED (intervals overlap or touch)")
        print(f"   {k:13s} {_fmt(c):34s} {rel}")
    print(f"   memory-use flip {ref['flip'][0]:,.0f}-{ref['flip'][1]:,.0f} (no environment interval): "
          f"{'before' if ref['flip'][1] < ab['lo'] else 'NOT ORDERED'}")
    if "grid_con_all" in B:
        print("DENSITY (24 fresh streams per cell vs 2 of them):")
        for q, d in (("Bayes", B), ("naive", N)):
            print(f"   24 streams {q}: gradient {d['grid_con_all']:+.4f} (2 of them {d['grid_con_two']:+.4f}); matched "
                  f"{d['grid_con_g_all']:+.4f}; near-pair distance {d['grid_d0_all']:.3f} vs {d['grid_d0_two']:.3f}")
    if "gridfold" in Tr:
        print(f"GRID FOLDS where naive folds (pooled over environments): Trained {np.round(Tr['gridfold'], 1)} at "
              f"{Tr['gridfold_x']}; ESN {np.round(E['gridfold'], 1)}; Bayes {B['gridfold']:.1f}")
    m880 = ix(Tr, 880)
    print(f"MATCHED share at 880: raw {Tr['con'][m880]:+.4f} matched {Tr['con_g'][m880]:+.4f} -> "
          f"{1 - Tr['con_g'][m880] / Tr['con'][m880]:.2f} of the raw gradient goes with distance to the cells; paired "
          f"per environment raw - matched {np.mean(Tr['con_env'][m880] - Tr['con_g_env'][m880]):+.4f} +- "
          f"{np.std(Tr['con_env'][m880] - Tr['con_g_env'][m880], ddof=1) / np.sqrt(NE):.4f}")
    for xv in (800, 960, 1040, 1120, 1200, 1280):
        print(f"   matched at {xv}: {at(Tr, 'con_g', xv):+.4f} (midpoint {(B['con_g'] + N['con_g']) / 2:+.4f})")
    # overall level (panel c): Trained checkpoints at the ESN's end level and at naive's
    for q, lvl, other in (("ESN end", E["mean"][-1], E["con"][-1]), ("naive", N["mean"], N["con"])):
        j = int(np.argmin(np.abs(Tr["mean"][Tr["x"] <= 3000] - lvl)))
        print(f"   overall level {q} {lvl:.3f} (gradient {other:+.4f}): Trained at {Tr['x'][j]:.0f} mean "
              f"{Tr['mean'][j]:.3f} gradient {Tr['con'][j]:+.4f}")
    print(f"FOLD: initial corners {Tr['corners'][0]:.2f} edges {Tr['edges'][0]:.2f}; ESN at 80 {at(E, 'corners', 80):.2f} "
          f"/ {at(E, 'edges', 80):.2f}; Trained at 20 {at(Tr, 'corners', 20):.2f} / {at(Tr, 'edges', 20):.2f} "
          f"(index {at(Tr, 'fold_idx', 20):+.2f}); Bayes index {B['fold_idx']:+.2f}, naive {N['fold_idx']:+.2f}")
    synthesis_table(S, tm)


TIMELINE = (("split gradient", "con", "movement", 1),
            ("split gradient, distance-matched", "con_g", "movement", 1),
            ("over-splitting, six steps apart", "apart6", "movement", -1),
            ("fold index (corners - edges)", "fold_idx", "answers", 1))


def write_timeline(S=None):
    """OUT/timeline.npz for the cross-topic synthesis: the headline measures of both runs per checkpoint with their
    5-95% environment-bootstrap bands, Bayes / naive levels, and the half-way crossing (the naive-Bayes midpoint;
    stays on Bayes's side to the end; NaN = never) with its interval. x = training episodes E - E // 5 with the
    initial network at 10."""
    S = S if S is not None else series()
    x = S["trained"]["x"]
    assert np.array_equal(x, S["echo"]["x"])
    out = {k: [] for k in ("names", "run", "family", "value", "lo", "hi", "bayes", "naive", "level", "half",
                           "half_lo", "half_hi")}
    for name, key, fam, sign in TIMELINE:
        for r in RUNS:
            c = crossing(S, r, key, sign=sign)
            lo, hi = np.quantile(S[r][key + "_b"], [.05, .95], axis=0)
            fin = lambda v: v if np.isfinite(v) else np.nan                            # noqa: E731
            for k, v in (("names", name), ("run", r), ("family", fam), ("value", S[r][key]), ("lo", lo), ("hi", hi),
                         ("bayes", S["bayes"][key]), ("naive", S["naive"][key]), ("level", c["level"]),
                         ("half", fin(c["x"])), ("half_lo", fin(c["lo"])), ("half_hi", fin(c["hi"]))):
                out[k].append(v)
    out = {k: np.array(v) for k, v in out.items()}
    path = os.path.join(OUT, "timeline.npz")
    np.savez(path, x=x, **out)
    print(f"saved {path}")
    for i in range(len(out["names"])):
        print(f"   {out['names'][i]:34s} {out['run'][i]:8s} {out['family'][i]:9s} level {out['level'][i]:+.3f} half "
              f"{out['half'][i]:,.0f} [{out['half_lo'][i]:,.0f}, {out['half_hi'][i]:,.0f}]")
    return path


# the synthesis export (td_tt_gap): (series key, row name, family, shape, this module's crossing with c50's definition)
SYNTHESIS = (("con", "split gradient (most minus least entangled tenth)", "movement", "rise", "grad_half"),
             ("con_g", "split gradient, distance-matched", "movement", "rise", "matched_half"),
             ("apart6", "over-splitting, six steps apart", "movement", "return", "apart_back"),
             ("toward6", "six-step move toward own cells", "movement", "rise", "toward_half"),
             ("fold_idx", "fold index (corners - edges)", "answers", "rise", "fold_half"),
             ("corners", "opposite corners", "answers", "rise", "corners_half"))
SAMPLE = "probes: 50 environments x 200 episodes (environment = probe row // 200)"


def _synthesis_note(S, run, key):
    """what the row's c0 / c50 mean where the gap fraction's conventions need a reading (numbers from the data)"""
    d, nv, bv = S[run], float(S["naive"][key]), float(S["bayes"][key])
    y, x = d[key], d["x"]
    f = (y - nv) / (bv - nv)
    if key == "corners":
        if run == "trained":
            return "c0 = corners past naive's level for good (corners_naive), c50 = corners_half"
        kl = int(np.flatnonzero(f <= 0)[-1])
        return (f"hovers at naive's level ({y[x >= 1500].min():.2f}-{y[x >= 1500].max():.2f} from 1,500; naive "
                f"{nv:.2f}): first above it at the {x[int(np.argmax(f > 0))]:,.0f} checkpoint, above it at every "
                f"checkpoint only from {x[kl + 1]:,.0f} (c0 = {cross_last(x, y, nv):,.0f}), by "
                f"{(y[kl + 1:] - nv).min():.2f}-{(y[kl + 1:] - nv).max():.2f}")
    if key == "apart6":
        if run == "trained":
            return (f"shape return: f < 0 while it over-splits beyond naive's {nv:.3f} (up to {y.max():.3f} at "
                    f"{x[int(np.argmax(y))]:,.0f}); c0 = below naive's level for good, c50 = back on Bayes's side of the "
                    f"midpoint (apart_back)")
        return (f"never beyond naive's {nv:.3f} (max {y.max():.3f} at {x[int(np.argmax(y))]:,.0f}), so c0 = {x[0]:.0f}: "
                f"not a return; past the midpoint toward naive for good from "
                f"{cross_last(x, y, (nv + bv) / 2):,.0f} (over-splits like naive, never returns), so c50 never")
    if (f > 0).all():
        lo = f"min {y.min():+.3f} at {x[int(np.argmin(y))]:,.0f}" if bv > nv else f"max {y.max():+.3f}"
        s = (f"never on the far side of naive's level ({nv:+.3f}; {lo}): the untrained network already sits between "
             f"naive and Bayes, so c0 = {x[0]:.0f} is not 'passing naive'")
        if key == "fold_idx":
            s += (f"; the initial network ({y[0]:+.2f}: beliefs near the centre) is above the midpoint "
                  f"({(nv + bv) / 2:+.2f}), so c50 is the return after the early fold" if run == "trained" else
                  f"; the initial network ({y[0]:+.2f}: beliefs near the centre) is above the midpoint "
                  f"({(nv + bv) / 2:+.2f}); it folds below it and never returns (c50 never)")
        return s
    return ""


def synthesis_rows(S):
    """the rows of the synthesis export (td_tt_gap.export fields): the six headline measures x {Trained RNN, ESN}.
    Each is a per-environment statistic averaged over the 50 environments, so every resampled curve is the
    count-weighted mean of the per-environment values over gap.boot_counts() (gap.env_mean), and naive's / Bayes's
    levels are resampled on the same environments (paired)"""
    import td_tt_gap as gap
    C = gap.boot_counts()
    rows = []
    for run in RUNS:
        for key, name, fam, shape, own in SYNTHESIS:
            v, vb = gap.env_mean(np.moveaxis(S[run][key + "_env"], -1, 0), C)          # (K,), (NB, K)
            nf, nb = gap.env_mean(S["naive"][key + "_env"], C)
            bf, bb = gap.env_mean(S["bayes"][key + "_env"], C)
            rows.append(dict(name=name, run=run, family=fam, shape=shape, sample=SAMPLE, paired=True,
                             note=_synthesis_note(S, run, key), value=v, boot=vb, naive=float(nf), bayes=float(bf),
                             naive_boot=nb, bayes_boot=bb, key=key, own=own))
    return rows


def synthesis_crossings(x, r):
    """c0 / c50 of one export row as td_tt_gap.export computes them (its public functions):
    {name: (point, lo, hi, share crossing, resampled crossings)}"""
    import td_tt_gap as gap
    f = gap.gap_fraction(r["value"], r["naive"], r["bayes"])
    fb = gap.gap_fraction(r["boot"], r["naive_boot"], r["bayes_boot"])
    out = {}
    for name, lev in (("c0", 0.), ("c50", .5)):
        cb = gap.crossing(x, fb, lev)
        out[name] = (gap.crossing(x, f, lev),) + gap.interval(cb) + (cb,)
    return out


def synthesis_table(S, tm=None, rows=None):
    """prints the export's crossings next to this module's own (same definition: the naive-Bayes midpoint, stays
    past) and checks that they agree on the same resamples"""
    import td_tt_gap as gap
    tm = tm if tm is not None else timing(S, None)
    rows = rows if rows is not None else synthesis_rows(S)
    x = S["trained"]["x"]
    W1 = np.ones((1, NE))
    f0 = lambda v: "never" if not np.isfinite(v) else f"{v:,.0f}"                    # noqa: E731
    print(f"SYNTHESIS EXPORT (td_tt_gap; shared resamples seed {gap.BOOT_SEED}; c0 = passes naive's level for good, "
          "c50 = past the naive-Bayes midpoint for good; [5-95%] (share of resamples that cross)):")
    for r in rows:
        cx = synthesis_crossings(x, r)
        own = tm[r["run"]][r["own"]]
        sign = 1. if r["bayes"] > r["naive"] else -1.
        mine = cross_last(x, r["boot"], (r["naive_boot"] + r["bayes_boot"]) / 2, sign)    # this module, same resamples
        same = np.allclose(np.nan_to_num(mine, nan=-1.), np.nan_to_num(cx["c50"][4], nan=-1.), rtol=1e-9, atol=1e-6)
        v1 = gap.env_mean(np.moveaxis(S[r["run"]][r["key"] + "_env"], -1, 0), W1)[1][0]
        own_b = np.einsum("be,...e->b...", gap.boot_counts() / NE, S[r["run"]][r["key"] + "_env"])   # series()'s code
        print(f"   {NAME[r['run']]:11s} {r['name'][:50]:50s} naive {r['naive']:+.3f} Bayes {r['bayes']:+.3f} | c0 "
              f"{f0(cx['c0'][0])} [{f0(cx['c0'][1])}, {f0(cx['c0'][2])}] ({cx['c0'][3]:.0%}) | c50 {f0(cx['c50'][0])} "
              f"[{f0(cx['c50'][1])}, {f0(cx['c50'][2])}] ({cx['c50'][3]:.0%})")
        print(f"      own {r['own']} {_fmt(own)} (seed {BOOT_SEED}); own code on the shared resamples: identical to c50 in "
              f"{'all' if same else 'NOT all'} {len(mine)} resamples; curve = series() {np.abs(r['value'] - S[r['run']][r['key']]).max():.1e}, "
              f"counts 1 {np.abs(v1 - r['value']).max():.1e}, series()'s resampling on these counts "
              f"{np.abs(own_b - r['boot']).max():.1e}")
        if r["key"] == "corners":                                  # c0 has this module's definition too
            mine0 = cross_last(x, r["boot"], r["naive_boot"], sign)
            same0 = np.allclose(np.nan_to_num(mine0, nan=-1.), np.nan_to_num(cx["c0"][4], nan=-1.), rtol=1e-9,
                                atol=1e-6)
            print(f"      own corners_naive {_fmt(tm[r['run']]['corners_naive'])} (seed {BOOT_SEED}); own code on the "
                  f"shared resamples: identical to c0 in {'all' if same0 else 'NOT all'} {len(mine0)} resamples")
        if r["note"]:
            print(f"      note: {r['note']}")
    return rows


def synthesis(S=None):
    """-> through_training/splitting/synthesis.npz (td_tt_gap.export; see the docstring's 'Synthesis export'):
    the six headline measures of both runs on the shared environment resamples, paired"""
    import td_tt_gap as gap
    S = S if S is not None else series()
    P = np.load(os.path.join(AN, "probes.npz"))
    assert np.array_equal(P["env"], np.arange(len(P["env"])) // gap.PER_ENV), "probes are not environment-major"
    x = S["trained"]["x"]
    assert np.array_equal(x, S["echo"]["x"]) and np.array_equal(x, gap.train_x(S["trained"]["episode"]))
    rows = synthesis_rows(S)
    path = gap.export("splitting", x, rows)
    z = gap.load("splitting")
    for i, r in enumerate(rows):                                  # the export's crossings = synthesis_crossings
        cx = synthesis_crossings(x, r)
        for name in ("c0", "c50"):
            assert np.allclose(np.nan_to_num(z[f"{name}_boot"][i], nan=-1.), np.nan_to_num(cx[name][4], nan=-1.))
            assert np.isclose(np.nan_to_num(z[name][i], nan=-1.), np.nan_to_num(cx[name][0], nan=-1.))
    synthesis_table(S, rows=rows)
    return path


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("what", nargs="*", default=["probes", "grids", "folds", "figures", "timeline", "synthesis"])
    a = pa.parse_args()
    if "force" in a.what:                                                            # rebuild the requested caches
        names = {"probes": ["probes_refs.npz"] + [f"probes_{r}.npz" for r in RUNS], "grids": ["grids_check.npz"],
                 "folds": ["grid_folds.npz"]}
        for step, files in names.items():
            for f in files:
                if step in a.what and os.path.exists(os.path.join(OUT, f)):
                    os.remove(os.path.join(OUT, f))
    if "check" in a.what:
        check_ent()
    if "probes" in a.what:
        build_probes()
    if "grids" in a.what:
        build_grids()
    if "folds" in a.what:
        build_grid_folds()
    if {"figures", "numbers", "timeline", "synthesis"} & set(a.what):
        S, ref = series(), references()
        if "figures" in a.what:
            print("saved", figure_split(S, ref))
            print("saved", figure_fold(S, ref))
        if "timeline" in a.what:
            write_timeline(S)
        if "synthesis" in a.what:
            synthesis(S)
        if "numbers" in a.what:
            numbers(S, ref)
