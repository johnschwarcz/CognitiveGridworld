"""Lesions through training: which neural feature does each behavioural family need, and from when? (John 11-02: "How
does the network get to the correct answer before learning optimal bayes dynamics?"; approved inference-time causal
experiment (3), lesions through training on the stored checkpoints, no new trainings.) One forward per (checkpoint,
lesion) on the probes and the forward set, scored with td_tt_intervene's cheap rows and forward-set rows. Fix round
11-02 (after the adversarial verification): a third pass (g-free random drive rows, |df|-matched forget-gate units,
both-latent / main-effect-free / top-PC / top-PC-span subspace controls with factor splits) and a fourth (a second seed
for the keep-PC controls); the claims are restated per row, per checkpoint and against these controls (RESULTS), and the
earlier 'less training', 'from their onset', 'answers first' and 'f / i less than generic rows' claims are withdrawn
(FAILED OR NULL). Final round 11-02 (after the last check; no new forwards, every earlier number kept): the path test
redone at matched training stage, every 'most', 'special' and 'equal shares' claim stated per control, the answers split
into their four core rows and corner answers (numbers_final), variance ranges over each claim's own checkpoints, and two
lean figures that keep only the results that survived. Label round 11-02 (after the final check): wording and figure
text only, no number changed.
-> main/REPLICATION/figure_bin/td_tt_lesions.png       fig_main(): through training, per panel (core answers, corner
                                                      answers, holding, coupling), the intact network and two lesions:
                                                      5 latent-interaction dims clamped (against both latents'
                                                      main-effect dims and random dims of the same variance) and one
                                                      gate block's recurrent drive resampled (against random rows of
                                                      the i, f and o blocks)
   main/REPLICATION/figure_bin/td_tt_lesions_map.png   fig_map(): the same comparisons, plus each latent's own
                                                      main-effect dims and the main-effect-free part, as specific
                                                      damage per checkpoint and panel (dots: 5-95% excludes 0)
   through_training/lesions/<run>/ck_EEEEE.npz (first pass), extra_EEEEE.npz (second), third_EEEEE.npz (third),
   fourth_EEEEE.npz (fourth), refs.npz, checks.json, checks_self.json, checks_third.json, trained_effects.tsv (table():
   every condition x checkpoint x row, change from intact, 5-95%)

HYPOTHESES (from the task)
  H1 'late dynamics come from belief-dependent gating / long-tau units': forget / input gate and long-tau lesions remove
    holding and coupling at >= 2,400 training episodes, cost the answers little, and do little at <= 1,600.
  H2 'one general capacity': every lesion damages all families in proportion. Tested two ways: equal shares of each
    family's gain removed, and whether the lesioned network lies on the intact network's own developmental path (late
    families vs answers), i.e. looks like a less-trained network.

DATA
  Trained RNN: 20 checkpoints, training episodes 80, 160, 320, 480, 640, 800, 960, 1,120, 1,280, 1,600, 1,920, 2,400,
  3,200, 4,000, 5,600, 8,000, 12,000, 16,000, 24,000, 39,996 (raw 100 ... 49,995; td_tt_gap.train_x). ESN: 800, 1,600,
  4,000, 12,000, 39,996. Every condition: the 10,000 probes (50 test environments x 100 cells x 2 streams; cheap score,
  18 rows) and the 5,028-episode forward set of the same environments (score_fwd, 8 rows), plus the belief change
  KL(intact || lesioned) of the marginals (mean over steps and both variables).

FORWARD (lforward)
  td_tt_intervene.forward's loop copied without memory cuts and generalised (the engine's hooks are hard-wired to whole
  gate blocks and one fixed clamp basis): row sets of the recurrent drive mean- or resample-ablated, forget gates of a
  unit set set to a value (the median, the unit's own mean, or given values), c of a unit set clamped, and a clamp or
  keep basis per step. Hook order within a step as the engine: drive -> gates -> f units -> c = f c + i g -> c units ->
  u = o tanh(c) -> clamp u. Every constant comes from the clean run of the same weights on the 10,000 probes
  (ti.reference), also on the forward set; the resample donors are the engine's (one fixed random probe of the same
  environment with another cell, seed 20261203), their CLEAN drive.

LESIONS (target) and their CONTROLS
  (a) Long-tau units. tau_j = 1 / (1 - mean f_j) (ck 'tau': probes x steps 2-30). Sets: the top 1 / 2 / 5 / 10% (10 /
    20 / 50 / 100 units) at the checkpoint ('own') or at the end of training tracked backward ('final').
    f clamp: f_j(s) := f_med(s), the median over units of the clean per-unit probe-mean forget gate at step s (shortens
    a long time constant to the typical one, and removes its state and input dependence). c clamp: c_j(s) := its clean
    per-step probe mean. Second pass, f at own mean: f_j(s) := the unit's own clean per-step probe mean (keeps its
    average time constant, removes only the state and input dependence of its forgetting). Controls, all from
    non-long-tau units (outside the top 10% at the checkpoint and at the end): count-matched random units (3 seeds; 2
    for f at own mean); units matched one by one on c variance v_j = Sum_s var_s(c_j) (greedy nearest log v, largest
    first; it cannot match: see the numbers); second pass, units with the same TOTAL c variance (random order until Sum
    v equals the target's; more units). Third pass, a perturbation-matched control: the median clamp moves the long-tau
    units' forget gates far more than the random units', which sit near the median (|f - assigned value| per unit and
    step on the clean run: NUMBERS), so the random units control the count only. The same random units (same seeds)
    get f_j(s) := clip(f_mean_j(s) + delta(s), 0, .99), delta(s) = mean_{j in target} f_mean_j(s) - f_med(s), the
    target set's mean shift: the same mean |df| (NUMBERS), applied as a constant per step like the target's (no state
    or input dependence), but upward (the non-long-tau units' forget gates are too low to move down by delta); for the
    own top 5% and 10% and the final top 10%.
  (b) Gate state-dependence. For gate block G in i, f, g, o (and f + i together): the recurrent drive W_hh,G u(s-1) is
    replaced by its clean per-step probe mean (mean-ablation; leaves the state manifold) or by the clean drive of the
    donor probe (resample-ablation; keeps the drive's distribution). Controls: count-matched random drive rows (1,000
    of the 4,000; 2,000 for f + i; 3 seeds; drawn from all four blocks) and, per block, rows of the other three blocks
    with the same total drive variance Sum_s=2..30 var_s(W_hh,r u(s-1)) (random order). Third pass, the main gate
    control: count-matched random rows WITHOUT g rows (1,000 or 2,000 of the 3,000 i, f and o rows; 3 seeds; the same
    sets for both ablation types). A quarter of the first pass's random rows are g rows, and g's drive is by far the
    most necessary block, so against those rows every other block looked unnecessary.
  (c) Latent-interaction subspace. At every step s, the r0 x r1 interaction of the 100 cell means of u (cell mean - r0
    mean - r1 mean + grand mean; td_analysis.factor_covs 'joint', trace checked against ck fac), its covariance
    projected off the 20 read-out rows (Pc C Pc, Pc = I - Q Q^T), top k = 3 or 5 eigenvectors V_s; after every step u_s
    <- u_s - (u_s - mu_s) V_s V_s^T (the read-out of the same step is untouched: every effect goes through the
    dynamics). Controls: random subspaces orthogonal to the read-out rows with the same clean variance of u per step (2
    seeds: the leading m_s columns of a fixed random basis, m_s matched per step); the additive r0 and r1 subspaces (top
    k of the projected r0 / r1 main-effect covariances); second pass, cross-fitted subspaces (estimated on environments
    0-24 and clamped on 25-49, and vice versa), for the interaction and the r0 / r1 controls. Third pass, at the 14
    window checkpoints (800-1,600 and >= 2,400; all 20 would take ~75 min of GPU time): add5 = top 5 of Pc (C_r0 + C_r1)
    Pc (both latents' main effects, no interaction; more variance than joint5: NUMBERS), pure5 = top 5 of Pp C_joint Pp
    (Pp also projects out the 18-dim span of the main-effect cell means, so pure5 holds no main-effect variance), pc5 =
    top 5 of Pc C_u Pc, each with variance-matched random dims (2 seeds), and pcspan5 = 5 random dims inside the top-K
    PC span with joint5's clean variance (K per step; 2 seeds); and the factor split of the clean variance inside each
    subspace and in all of Pc u: environment, r0, r1, r0 x r1 (cell means pooled over environments), environment x cell,
    within cell (the two streams of a cell); additive (balanced probes).
  (d) Dimension. Per step, keep the span of the top k = 5, 10, 20, 50, 100, 200 eigenvectors of Pc C_u(s) Pc plus the
    read-out rows (u_s <- mu_s + B_s B_s^T (u_s - mu_s)); everything else is clamped to the step's mean. Control: random
    removal of the same clean variance per step (random subspaces orthogonal to the read-out rows); fourth pass, a
    second random seed for keep 20 and keep 100 (the first pass's second random basis).

SCORES
  Families = the timeline's rise-shaped rows that the cheap and forward-set scores contain (td_tt_timeline): answers
  (goal accuracy at T, both right at T, far errors, distant pairs merged, corner answer = truth); holding (corner split
  held (joint), held per variable at corner-split ends and at one-variable end splits, distance out of a corner split,
  end-mass slope); coupling (excess angle, G, angle of jumps >= 4 cells, coupling angle out, same-diagonal share,
  direction slope). The splitting rows, fold index and opposite corners need the 'full' score and are left out.
  Gap fraction per row f = (y - y_naive) / (y_Bayes - y_naive) (ti.refs: naive and Bayes on the same episodes; a fixed
  affine map per row), family f = mean over its rows (those defined in both conditions). Change dF = f_lesioned -
  f_intact. Specific damage = f_control - f_lesioned (> 0: the target removes more of the family than its matched
  control; seeds averaged first). Windows: early = 800-1,600 training episodes (5 checkpoints), late = >= 2,400 (9).
  Path residual: late-family f (mean of holding and coupling f) of the lesioned network minus the intact network's at
  the same answers f (linear interpolation along the intact checkpoints >= 640, running maximum of answers f).
  Rewind: the intact checkpoint (log-interpolated) whose answers f equals the lesioned network's. Onset: the training
  episode after which the specific damage stays above 0.1 gap units to the end (td_tt_gap.crossing), with [5-95%] and
  the share of resamples that cross; compared with the families' c10 / c50 (td_tt_timeline, medians over the rows).
  Fix round: per row, the specific damage row by row (row_specific; 'core 4' = the answers rows without corner answers,
  averaged per checkpoint); rewind signatures, a row's raw value of the lesioned network minus the intact network's at
  the rewind point (same answers f; path >= 640, running maximum, linear interpolation; rewind_diff), for corner
  commitment (commit_fwd: share of Bayes's corner-split stream-steps whose answer is a corner while the split is not
  held jointly; naive .554, Bayes 0), held per variable, held jointly and corner answers; the path residual pooled
  over checkpoints >= 640 in bins of the lesioned network's answers f; the seed spread, each seed's own window-mean
  specific damage ([min, max]); share differences, the share of holding's (coupling's) late gain removed minus the
  answers' share. Intervals: 5-95% over the 1,000 shared environment resamples (td_tt_gap.boot_counts, seed
  20261103), the same counts for every condition, checkpoint and reference (all differences paired).
  Final round (numbers_final): panels (PANEL_ROWS; the figures) = core answers (acc_goal_T, acc_both_T, far_T,
  merged_T), corner answers (corner_truth_fwd: answer = truth at T on the forward set's 2,400 unselected corner-cell
  episodes), holding and coupling, each scored like a family. Path test at matched training stage (stage_matched): a
  condition's late path residual minus joint5's at the same late checkpoint, over the late checkpoints where the two
  lesioned networks' answers f differ by <= .05 (at least 3 of the 9) or <= .10 (at least 1); pooling checkpoints into
  answers-f bins (the fix round's binned residuals) mixes training stages.

VALIDATION (checks -> checks.json, check_self -> checks_self.json, check_third -> checks_third.json)
  The lean forward against td_tt_intervene.forward on 3 networks (trained raw 2,100 and 49,995, echo 2,100; intact,
  mean-ablated f and all four blocks, resampled i and o, 100-unit f and c clamps, a 50-dim u clamp; intact and
  resampled i also on the forward set): max |d log q| 3e-05 to 1e-04 (different chunk shapes), answers identical
  except 0-6 near-ties of 6,003,360 (probe, step, variable) entries per network. Definitions against the stored ck
  statistics: the r0 x r1 factor trace (and r0, r1, total) per step to 1e-07 relative, the per-block recurrent-drive
  variance to 1e-07, tau from the engine's forget gates to 6e-08. Clamps: |(u - mu) V| <= 1e-06 after every step, the
  kept-span residual <= 8e-07, the step-1 read-out unchanged to 2e-06 (|Q^T V| ~ 1e-16); repeated calls bit-identical.
  Forget gates at own mean: |f - own mean| = 0 on the clamped units, a clamp of no unit = intact (0). The second
  pass's intact network equals the first pass's in every row (0). Third pass (trained raw 5,000, echo 5,000): forget
  gates at given values exact (0), the own mean given as values reproduces 'self' exactly (0), the g-free sets hold no
  g row, the new subspaces orthonormal and off the read-out rows, held by the clamp, pure5 without main-effect
  variance, the factor split adding up (NUMBERS, VALIDATION third pass); the third and fourth passes' intact networks
  equal the first pass's (0). td_tt_intervene.captured_variance (fixed 11-02 to measure the clamp's actual basis) was
  never used here: every clamp basis is built off the read-out rows, so its clamped variance is tr(V^T C V), equal to
  the fixed function's to <= 1e-13 relative for the third pass's subspaces (checks_third.json); no lesion number
  changes.

RESULTS (Trained RNN unless stated; gap units = fractions of the naive-Bayes gap; window means; [5-95%]; specific damage
> 0 = the target removes more than its control, < 0 = the control removes more; random controls averaged over their
seeds, each seed's own value under SEED SPREAD; panels as in SCORES)
  1. The latent-interaction subspace (5 dims per step). Over the 14 window checkpoints it holds .088-.170 of u's
  variance (early .118-.170, late .088-.111; .22-.44 at 80-640) and .81-.88 of the r0 x r1 factor (itself .022-.039 of
  u's variance; split-half cos^2 .62-.79), but most of the clean variance inside it is not interaction variance: of it,
  r0 x r1 is .11 / .17 / .26 / .37 at 800 / 1,600 / 4,000 / 39,996, the r0 + r1 main effects .28 / .37 / .36 / .24,
  environment x cell .37 / .28 / .24 / .26, within cell .10-.12. Clamping it removes, late, answers -.34 [-.35,-.33],
  holding -.51 [-.54,-.48], coupling -.37 [-.39,-.35] (goal accuracy -.100, held per variable -.497, excess angle -5.87
  deg); early -.29 / -.07 / -.11; shares of the late gain .37 / .82 / .64 (holding - answers +.45 [+.43,+.48]).
  Per unit of removed variance it is special for every family. Late, which family is hit most depends on the control;
  early, the answers are hit most in all three comparisons below, while holding and coupling have barely developed
  (intact f early .44 / .03 / .11). Against random dims of the same variance: holding and coupling (late +.17 / +.25 /
  +.24, answers / holding / coupling; early +.14 / +.03 / +.07). Against 5 random dims inside the top-K PC span with the
  same variance (K 14-36): coupling least (late +.22 / +.19 / +.13, seeds [+.22,+.22] / [+.18,+.20] / [+.12,+.13]; early
  +.14 / +.02 / +.05); the answers' and holding's intervals overlap (+.22 [+.20,+.24] / +.19 [+.16,+.22]), and the
  answers' edge is corner answers (+.45 [+.37,+.53]; the core answers' +.16 [+.15,+.18] overlaps holding's). For its
  main-effect-free part (pure5: no r0 / r1 cell-mean variance; .043-.060 of u, r0 x r1 .19 / .35 / .49 / .55 of it):
  holding (beyond its own random dims late +.19 / +.30 / +.18, coupling no more than the answers; early +.12 / +.04 /
  +.01). Cross-fitted subspaces give the same (late +.14 / +.24 / +.23 beyond random dims). The top 5 PCs (.213-.270 of
  u, against joint5's .088-.170) remove as much holding, a little more coupling and more of the answers (joint5 - pc5,
  late: -.22 / +.01 / -.05).
  Per panel (the figures). Against random dims of the same variance the core answers lose more at every checkpoint from
  the first (80: +.15 [+.08,+.24], where the 5 dims hold .44 of u's variance; peak +.24 at 1,600; +.07 at 39,996; early
  +.17 [+.14,+.19], late +.12 [+.10,+.13]), corner answers from 1,600 on (and at 1,120; +.12 to +.52, late +.40
  [+.32,+.48]), holding from 640 and coupling from 960 on, growing as they develop (late +.25 [+.22,+.28] / +.24
  [+.22,+.26]). Against the latents' main effects it is special late for corner answers and holding, for coupling beyond
  each latent's own dims but barely beyond both latents', and not for the core answers. Beyond each latent's own 5 dims
  (r0 / r1, the two lesions averaged), late: corner +.69 [+.62,+.77], holding +.29 [+.27,+.32], coupling +.12
  [+.10,+.13], core -.01 [-.03,+.01] (goal accuracy -.03 [-.05,-.00], both right +.00 [-.02,+.02], far errors +.01
  [-.00,+.03], merged -.03 [-.05,-.01]; cross-fit -.04 [-.05,-.02]; holding rows +.16 to +.48, coupling rows -.03 to
  +.38); the core answers exceed r0 / r1 only at 320-3,200 (+.04 to +.16; early +.13 [+.11,+.15]) and fall below them
  from 8,000 on (-.03 to -.06), corner answers exceed them from 320 on (+.02 to +.83). Late, corner answers fall to f
  .22 with the interaction dims clamped, against .62 with random dims and .92 / .96 with one / both latents' main-effect
  dims clamped (intact .87: clamping the main effects leaves corner answering at or above the intact level). Beyond a
  5-dim subspace of both latents' main effects (add5; .165-.191 of u, 1.08-2.04x joint5's variance): corner answers at
  every window checkpoint (late +.74 [+.66,+.82], early +.48 [+.41,+.55]), holding at every window checkpoint (late +.22
  [+.18,+.26], early +.03 [+.02,+.04]), coupling only slightly and from 8,000 on (late +.03 [+.01,+.06]; rows -.14 to
  +.29), and not the core answers, of which add5 removes more (late -.17 [-.21,-.14], early -.03 [-.05,-.00]): in the
  early window joint5's edge over add5 on the 5-row answers family (+.08 [+.05,+.10]) is corner answers alone. Per unit
  of variance add5 is special only for coupling (late -.00 / -.01 / +.18 beyond its random dims). The main-effect-free
  part removes beyond its random dims core answers +.08 [+.07,+.10] early and +.06 [+.04,+.07] late, corner answers +.27
  [+.22,+.33] and +.74 [+.68,+.80].
  From when (the 5-row answers family). Against random dims the answers' specific damage is about +.1 from the first
  checkpoint (80: +.12 [+.06,+.19]; 160-960: +.05 to +.11), before the answers pass naive (c10 838), and doubles at
  1,120-1,600 (+.15 to +.22); against r0 / r1 it appears at 320-640 (+.05 [+.01,+.09] at 320, +.12 [+.09,+.15] at 640).
  Its crossing of .1 (915 [311-1,029]) marks where a curve near the threshold stays above it: it is not an onset and is
  not compared with c10. Holding's and coupling's specific damage grows as they develop (holding +.00 to +.02 through
  1,280, +.37 at 39,996); it stays above .1 from 2,842 [2,543-3,277] (holding) and 1,358 [1,252-2,735] (coupling), both
  after the answers' crossing; holding vs coupling is unresolved (overlapping intervals).
  Not 'less training'. In the plane of answers vs late families the lesioned network lies on the intact path only at
  2,400-4,000: its late-family f minus the path's at the same answers f is -.05 to -.03 at 640-1,920 (below the path),
  -.01 to +.01 at 2,400-4,000 and +.05 to +.12 from 5,600 on (above it; late window +.05 [+.03,+.08]; rewind 2,400 /
  4,000 -> 1,044 / 1,256 episodes, late families +.04 vs path +.05, +.09 vs +.08), so 'less training' also fails,
  modestly, inside the plane after 4,000. Matched on training stage (the same late checkpoint, answers f within .05 of
  joint5's, at least 3 of the 9 late checkpoints) it is the closest lesion to the path, not on it: the cross-fitted
  subspace lies as close (-.01 [-.01,+.00]), and the other 24 matched conditions lie +.11 to +.36 further above the
  path, among them both latents' main effects (add5 +.11 [+.09,+.14]), the main-effect-free part (pure5 +.12
  [+.11,+.13]), resampled i (+.16 [+.14,+.18]), g-free random rows (+.19 to +.24), random dims removing keep-100's
  variance (+.23 / +.27) and the |df|-shifted units (+.28 to +.36). The main controls are not among them: the
  same-variance random dims and r0 / r1 keep more of the answers (late-window answers f .77 / .74 and .70 / .73, joint5
  .58). With tolerance .10 (at least 1 checkpoint) the 3-dim subspaces (-.02 [-.03,-.01]) and the cross-fit lie as
  close; every other condition lies +.07 or more above, the closest mean-ablated o (+.07 [+.05,+.09], 3 checkpoints),
  two variance-matched drive-row controls (+.08) and pure5 (+.09 [+.08,+.10]). Matched and drawn on the core answers' f
  instead, the same (joint5's late residual +.03; only joint3 and the cross-fit within -.01; the 27 others +.13 or
  more). The fix round's pooled-bin comparison ('subspace clamps that leave the answers at the same level lie as close')
  mixed training stages: its [.45,.60) bin held joint5 at 3,200-5,600 but the random dims and r0 / r1 at 1,600-2,400,
  where the late families have barely developed and the test has no power; it is withdrawn. Outside the plane the
  lesioned network is unlike the earlier stage with the same answers: at
  its rewind point the intact network commits to a corner at corner splits in .61-.68 of stream-steps, the lesioned
  network in .34-.46 (-.27 [-.33,-.20] late; -.16 at 2,400 to -.34 at 39,996; cross-fit -.26 [-.32,-.19]; at the core
  answers' rewind point -.27 [-.34,-.20]), while holding per variable like it (-.00 [-.03,+.02]) and holding jointly
  about as rarely (.01-.05 vs .02-.06; -.01 [-.03,-.00]): the deficit is the corner conjunction; its corner answers are
  -.08 [-.12,-.04] below the rewind point. Of the intact network's other stages, the earliest (80-480: .01-.39) and the
  latest (12,000-39,996: .38-.17) commit as little as the lesioned network (late mean .386) or less, those in between
  more (640-8,000: .45-.68). The main-effect-free interaction subspace gives the same deficit (commitment -.30
  [-.34,-.24], at the core answers' rewind point -.30 [-.35,-.25]; corner answers -.25 [-.28,-.23]), the both-latent
  main-effect subspace the opposite (+.11 [+.08,+.16], +.35 [+.32,+.38]). The other lesions stay within -.11 to +.04 of
  their rewind points in commitment (random dims -.05 / -.05, r0 / r1 +.00 / +.04, keep 100 PCs -.08, its random dims
  -.02 / -.02, resampled i -.10, long-tau clamp -.05), and most hold more per variable than the rewound network (+.11 to
  +.28; the top 5 PCs +.06). The random 5-dim clamps inside the top-K PC span (joint5's variance; a mild lesion, late
  answers f .80) come closest to an earlier stage (late path residual +.02 / +.02; rewind differences within +-.04,
  though held jointly is lower in both seeds: -.03 [-.06,-.01] / -.04 [-.07,-.02]).
  2. Gate state-dependence: a late block ranking, g > o > i >= generic non-g rows > f (i is above generic rows for the
  answers and holding; for coupling only at 8 of 20 checkpoints when resampled, and not when mean-ablated; per panel
  below; early, holding's o, i and f effects are +.00 to +.01 and only g's is larger, +.04). Against count-matched
  random drive rows without g rows (3 seeds), resampled drive, late: g +.76 [+.72,+.81] / +.22 [+.20,+.24] / +.34
  [+.32,+.36], o +.25 [+.24,+.27] / +.12 [+.10,+.13] / +.17 [+.15,+.18], i +.05 [+.05,+.06] / +.02 [+.01,+.03] / +.02
  [+.01,+.03], f -.19 [-.20,-.18] / -.17 [-.19,-.16] / -.13 [-.14,-.12] (answers / holding / coupling); early g +.57 /
  +.04 / +.10, o +.16 / +.01 / +.05, i +.03 / +.01 / +.00, f -.05 / +.00 / -.03. Per checkpoint i's is > 0 (5-95%) for
  the answers at every checkpoint from 640 on (< 0 at 80), for holding from 2,400 on (and at 4 earlier checkpoints,
  never < 0), for coupling at 8 of 20; f's is < 0 for the answers from 640 on, holding from 1,920, coupling from 480;
  o's > 0 from 480 / 800 / 480; g's from 80 / 640 / 480. f + i together remove less than 2,000 generic non-g rows (late
  -.13 / -.07 / -.09, early -.09 / -.00 / -.04), because f's drive matters least.
  Mean-ablation gives the same late ranking except i for coupling (late +.01 [-.00,+.01], > 0 at 1 of 20 checkpoints;
  and for holding o's and i's intervals overlap, +.08 [+.07,+.11] / +.04 [+.02,+.07]): late g +.88 / +.25 / +.33, o +.14
  / +.08 / +.16, i +.05 / +.04 / +.01, f -.15 / -.19 / -.12. Against the first pass's all-block random rows (a quarter g
  rows) every block but g looked unnecessary (late f -.48 / -.27 / -.27, i -.24 / -.08 / -.12, f + i -.42 / -.13 / -.22,
  o -.04 / +.02 / +.03, g +.47 / +.12 / +.20): that comparison measured mostly g.
  Changes, late: f -.15 / -.11 / -.07, i -.39 / -.31 / -.22, f + i -.50 / -.34 / -.26, o -.59 / -.40 / -.37, g -1.10 /
  -.51 / -.54 (goal accuracy -.413, below naive).
  Per panel, the answers' ranking is the core rows'; corner answers rank differently. Resampled, late (early): core
  answers g +.89 [+.83,+.95] (+.66), o +.33 [+.31,+.35] (+.20), i +.01 [-.00,+.01] (+.04; > 0 only at 480-2,400 and
  39,996), f -.19 [-.19,-.18] (-.09); corner answers g +.26 [+.23,+.30] (+.24; > 0 from 160), i +.25 [+.22,+.27] (-.03;
  > 0 from 1,920), o -.05 [-.08,-.02] (-.00; < 0 from 12,000), f -.19 [-.23,-.16] (+.08 [+.05,+.10]; > 0 at 320-1,920, <
  0 from 3,200). Late gap fractions (intact; g / o / i / f resampled; random rows): core answers .93; -.25 / .31 / .63 /
  .82; .64; corner answers .87; .13 / .44 / .14 / .58; .39. So the 5-row family's late i effect is the corner row: late,
  i's drive matters beyond generic rows for corner answers as much as g's, and o's not at all. Mean-ablated, the same
  split (corner i +.19 [+.15,+.23], o -.22 [-.28,-.16]; core i +.02 [+.01,+.03], o +.23 [+.21,+.25]).
  Shares of the late gain, resampled: f + i .54 / .55 / .45, i .42 / .50 / .39, f .16 / .18 / .12, o .64 / .65 / .64
  (holding - answers +.01 [-.03,+.05], +.08 [+.04,+.11], +.02 [+.00,+.04], +.02 [-.03,+.05]; coupling - answers -.09
  [-.14,-.05], -.04 [-.07,+.00], -.03 [-.05,-.02], +.00 [-.04,+.05]); mean-ablated i .39 / .60 / .32 (holding - answers
  +.21 [+.15,+.26]), f + i (+.11 [+.04,+.16]; coupling - answers -.19 [-.25,-.14]), o .48 / .67 / .59 (+.18 [+.14,+.22]
  / +.11 [+.05,+.15]). Only resampled o removes equal shares of all three families; resampled f + i removes equal shares
  of holding and the answers but less of coupling; i alone is partly holding-selective, more so when mean-ablated (which
  also moves the drive off the state manifold); coupling's share is below the answers' for every f / i lesion (-.03 to
  -.19; resampled i's -.04 [-.07,+.00] only marginally). Off the path +.22 to +.27. g's specific damage starts at the
  first checkpoint for the answers only: onsets 80 [80-400] / 2,201 [1,767-2,606] / 1,092 [969-1,422] against g-free
  rows (80 [80-493] / 4,715 [4,193-5,371] / 2,286 [1,774-2,794] against all-block rows).
  3. Long-tau units. Setting the top 10%'s forget gates to the median costs the answers early (-.10 [-.11,-.09]; holding
  +.00, coupling -.02) and all families late (-.24 / -.17 / -.16; shares .26 / .27 / .28). But the clamp moves their
  forget gates by .40 / .43 / .56 / .69 per unit and step (|f - assigned|, clean run; 800 / 1,600 / 4,000 / 39,996) and
  the count-matched random units' by only .12 / .14 / .16 / .18 (they sit near the median), so 'beyond random units'
  (late +.22 / +.14 / +.14) mostly measured the size of the perturbation. Random units shifted up by the targets' mean
  shift (|df| .39 / .43 / .54 / .63; it lengthens their time constants from 1.35-1.41 to a median of 2.7 / 2.9 / 4.2 /
  6.3, while the clamp shortens the targets' from 3.5 / 3.8 / 6.3 / 10.5 to 1.24-1.35): specific damage early +.06
  [+.04,+.07] / -.01 [-.02,+.00] / +.01 [+.00,+.02], late -.01 [-.03,+.00] / -.04 [-.07,-.02] / +.04 [+.03,+.06]. Per
  checkpoint: coupling's is > 0 at every checkpoint from 1,280 on (+.02 to +.09); holding's is never > 0 and < 0 (the
  shifted random units remove more holding) at 80 and at 9 of the 11 checkpoints from 1,600 on; the answers' is > 0
  through 1,920 (+.03 to +.11 at 640-1,920; larger before 640, where the network is below naive) and at 24,000 / 39,996
  (+.04, +.11), not at 2,400-16,000 (< 0 at 5 of those 7). Seeds, late: [-.02,-.00] / [-.08,-.02] / [+.03,+.06]. So the
  long time constants are necessary, but beyond a perturbation of the same size they are special only for coupling (from
  1,280) and for the answers early and at the end, never for holding; 'only late, for all families alike, answers first'
  is withdrawn. The final checkpoint's long-tau set tracked backward does little early (-.03 / -.00 / -.02) and about as
  much as the own set late (-.21 / -.17 / -.16), but not in equal shares: it removes slightly larger shares of the late
  families (.23 / .27 / .27; holding - answers +.04 [+.00,+.07], coupling - answers +.04 [+.01,+.07]). Forget gates held
  at each unit's own per-step mean (time constant kept, state and input dependence removed): top 10% -.04 / -.03 / -.01
  late (|df| .112 / .111 / .096 / .046, random units at own mean .078 / .096 / .102 / .097); all 1,000 units -.20 / -.15
  / -.07. c clamped (top 10%): -.90 / -.30 / -.28 late; special per unit for the answers and coupling, not for holding
  (beyond count-matched random units +.56 [+.52,+.60] / .00 [-.02,+.03] / +.14 [+.12,+.15], whose summed c variance is
  only .02-.50 of the targets'), and not per unit of c variance: units with the same total c variance remove more (-.86
  / -.31 / -.07), but they are 208-876 units instead of 100 (until 4,000 they reach only .16-.84 of the targets'
  variance), and count-matched units matched one by one on c variance (.11-1.01 of it, 1.01 only at 39,996) remove less
  of the answers and coupling but more holding (+.24 / -.05 [-.08,-.02] / +.07 late).
  4. Dimension: the top-100 span is mildly privileged. Keeping 100 PCs (removing .17-.20 of u's variance late, .09-.15
  early): -.28 / -.26 / -.11 late, .03-.06 less than removing the same variance at random (-.06 [-.08,-.04] / -.03
  [-.05,-.01] / -.03 [-.04,-.02]; every family and both seeds: [-.07,-.05] / [-.05,-.02] / [-.04,-.03]; early -.00 /
  +.00 / -.01). Keeping 20 (removing .48-.52 late): -.98 / -.51 / -.31, the answers +.12 [+.09,+.16] worse than random
  removal (seeds +.11 / +.14: they use directions outside the top 20).
  5. ESN (5 checkpoints; answers f -1.22 to -.02, no holding or coupling): its r0 x r1 factor is .006-.007 of u's
  variance, and removing that subspace costs its answers no more than random dims or r0 / r1 (-.03 to +.04). g's drive
  is needed beyond g-free random rows (+.13 / +.03 / +.05 / +.08 / +.09, answers per checkpoint); its i, f and o drives
  do nothing (dF -.01 to +.00), so the first pass's 'f + i less than random rows' (-.08 to -.01) was the random rows'
  g share. Long-tau units: the median clamp moves their forget gates by .058-.088 per unit and step, random units' by
  .031-.048; against random units shifted by the same mean (.055-.086) the answers lose more only late (+.04
  [+.02,+.06] at 12,000, +.03 [+.01,+.05] at 39,996; earlier -.03 to -.00; max tau 2.5-3.1).
  Answer to "how does the network get the answers before Bayes-like dynamics", as far as lesions show it: early, its
  answers depend, through the recurrent dynamics, on a few dims of u aligned with the latent pair's interaction. The
  core answers lose more when these dims are clamped than when random dims of the same variance are, from the first
  checkpoint on (when the 5 dims hold .22-.44 of u's variance), and more than when either latent's own main-effect dims
  are at 320-3,200, but not more than when a 5-dim subspace of both latents' main effects is (early -.03 [-.05,-.00];
  joint5's early edge over that subspace is corner answers, +.48). The answers also depend most on the state-dependent
  cell input g, then on the output gate o, on the input gate's drive slightly more than on generic drive rows (the core
  answers at 480-2,400), and least on state-dependent forgetting. Late, the same dims are needed beyond both latents'
  main effects for corner answers and holding (coupling barely, +.03), not for the core answers; and late, corner
  answers need i's drive as much as g's. In the plane of answers vs late families the interaction lesion is the closest
  lesion to the developmental path at the same checkpoint and answers level: every other matched condition except the
  cross-fit lies +.11 to +.36 further above (tolerance .05; +.07 or more at .10, apart from the 3-dim subspaces), and
  the same-variance random dims and r0 / r1 keep more of the answers and are not matched at tolerance .05. It lies on
  the path only at 2,400-4,000: below it at 640-1,920 (-.05 to -.03), above it from 5,600 on (+.05 to +.12; late +.05
  [+.03,+.08]; on the core answers' f +.03). Outside that plane it does not look like an earlier stage: it commits to a
  corner far less than the intact network did at the same answers level (.34-.46 vs .61-.68; -.27 [-.33,-.20]) while
  holding per variable like it (-.00 [-.03,+.02]): the deficit is the corner conjunction. The main-effect-free part of
  the interaction gives the same deficit.

FAILED OR NULL
  H1 fails: the gate and long-tau lesions are not late-selective. Resampled f + i removes equal shares of holding and
  the answers (.55 / .54) but less of coupling (.45; -.09 [-.14,-.05]); i is partly holding-selective (holding - answers
  +.08 resampled, +.21 mean-ablated) but costs the answers .42 / .39 of their late gain; f's drive matters less than
  generic non-g rows (late -.19 / -.17 / -.13), i's slightly more (+.05 / +.02 / +.02; on the core answers late only
  +.01 [-.00,+.01], on corner answers +.25 [+.22,+.27]). The own top-10% long-tau forget clamp acts proportionally
  (.26 / .27 / .28; the final top-10% set's not quite, .23 / .27 / .27) and, against a perturbation of the same size, is
  never special for holding and only slightly for coupling. Holding the long-tau units' forgetting at its own mean
  removes only .02-.03 beyond random units.
  H2 as 'equal shares' holds only for resampled o (.64 / .65 / .64; holding - answers +.02 [-.03,+.05], coupling -
  answers +.00 [-.04,+.05]), the own top-10% long-tau forget clamp (.26 / .27 / .28; +.01 / +.02; the final top-10%
  set's clamp removes slightly larger shares of the late families, +.04 [+.00,+.07] / +.04 [+.01,+.07]) and holding vs
  answers under resampled f + i (+.01 [-.03,+.05]); i alone is partly holding-selective, coupling's share is below the
  answers' for every f / i lesion (-.03 to -.19), mean-ablated o removes more of the late families (+.18 / +.11), and
  the networks with equal shares lie off the developmental path (late residual: resampled o +.18, the own top-10%
  forget clamp +.28). H2 as 'less training' fails for the interaction lesion outside the plane of answers vs late
  families, and modestly inside it after 4,000: in that plane, at matched training stage, it is the closest lesion to
  the path (at tolerance .05 every other matched condition but the cross-fit lies +.11 to +.36 further above), but it
  lies on the path only at 2,400-4,000 (above it from 5,600 on, +.05 to +.12; late +.05 [+.03,+.08]); and it commits to
  a corner far less than the intact network at the same answers level (-.27 [-.33,-.20]). The mild random 5-dim clamps
  inside the top-PC span come closest to less training (path residual +.02; rewind differences within +-.04, held
  jointly -.03 / -.04).
  Withdrawn after the verification (11-02): 'the interaction subspace is needed by every family ... from each family's
  onset on' (the answers depend on it before their onset; the .1 crossing at 915 is a threshold artefact); 'removing it
  looks like less training' and 'sends the network back along its own developmental path'; 'answers from ~0.9k, coupling
  from ~1.4k, holding from ~2.8k' (holding and coupling overlap); '+.13 beyond the additive r0 / r1 subspaces' for the
  answers late (one row, corner answers; the four core rows -.01); 'f / i state-dependence matters less than generic
  drive rows in every family and window' (true only against rows that include g; i's drive matters more than generic
  non-g rows); 'damage proportional across families' for mean-ablated i; 'g is necessary for everything from the first
  checkpoint' (only the answers' specific damage starts then); 'long time constants matter only late, for all families
  alike, answers first' (not against a perturbation-matched control; the answers lose -.10 early); 'long-tau units' c
  content is not special' (it is per unit for the answers and coupling, not per unit of c variance).
  Withdrawn after the last check (11-02, final round): 'its closeness to the path is not specific to the interaction;
  subspace clamps at the same answers level lie as close' (the pooled bin mixed training stages; matched on stage it is
  the closest lesion to the path, on it only at 2,400-4,000); 'per unit of removed variance it is special, most for
  holding and coupling' (only late and only against random dims anywhere: within the top-PC span coupling is hit least
  and the answers' and holding's intervals overlap, and for the main-effect-free part coupling no more than the answers;
  early the answers are hit most in all three per-unit comparisons); 'commits to corners far less than any earlier
  stage' and 'unlike any earlier stage' (the intact network commits as little at 80-480 and at 12,000-39,996; the
  deficit is against the stage with the same answers); 'needed beyond r0 / r1 on every answer row through 1,600' (the
  core rows only at 320-3,200, and not beyond both latents' main effects); 'H2 as equal shares holds for the general
  features (f / i / o drive, long time constants)' (only resampled o, the own top-10% forget clamp, and holding vs
  answers under resampled f + i); the unqualified ranking 'g > o > i > generic non-g rows > f' (i is not robustly above
  generic rows for coupling and not at all when mean-ablated, on the core answers only at 480-2,400 and 39,996; corner
  answers rank g ~ i > generic rows > o > f late); 'special per unit' for the c clamp's holding (.00 [-.02,+.03]) and
  'c-matched units remove less' for holding (they remove more, -.05 [-.08,-.02]); 'the random 5-dim clamps inside the
  top-PC span look like an earlier stage in all four signatures' (they come closest; held jointly is lower in both
  seeds); 'null: the top PCs are not privileged' (with the second seed the top-100 span is mildly privileged).
  Withdrawn in the label round (11-02, after the final check): 'at the same checkpoint and answers level it is the only
  lesion that stays on the developmental path' and 'in that plane it looks like an earlier stage' (it is the closest
  lesion there, on the path only at 2,400-4,000); 'within the top-PC span the answers are hit most' (coupling least;
  the answers' edge over holding is corner answers).
  Unit-by-unit c-variance matching failed (the long-tau units hold the most c variance: .11-1.01 of it); the
  same-total-c-variance sets need 208-876 units. Null: the ESN's interaction subspace is not special for its answers.

CAVEATS
  One seed and one world; 20 checkpoints, so onsets are coarse (log-interpolated); window means pool checkpoints.
  Subspaces and every hook constant come from the clean run on the same probes; only the interaction and r0 / r1
  subspaces were cross-fitted (not pure5, add5, pc5 or pcspan5). The third pass's subspace controls ran at the 14 window
  checkpoints only (80-640 and 1,920 left out: all 20 would have taken ~75 min of GPU time); the gate and forget-gate
  controls at all 20. The |df|-matched control shifts the random units' forget gates up (their time constants lengthen
  to a median 2.7-6.3) while the target clamp shortens the long-tau units' to ~1.3: same size, opposite direction (a
  downward shift of that size does not fit their low forget gates); clipping at f = .99 (0-26% of the units) leaves its
  |df| at .63 against the targets' .69 at 39,996. add5 and pc5 hold more variance than joint5 (.165-.191 and .213-.270
  of u against .088-.170; add5 1.08-2.04x): their direct comparisons with joint5 are not variance-matched, so each is
  also compared with its own random dims. Clamps hold directions at the clean per-step mean, off the state manifold for
  large clamps (keep <= 20 PCs, the c clamps, mean-ablation). The holding rows depend on calibration (an unconfident
  network can hold more); the interaction lesion lowers holding and commitment together. The path test is informative
  only where the intact path is shallow: late the answers saturate (f .79-.98 at >= 2,400) while the late families still
  grow, so a lesion that costs the answers a little lands far above the path (forget gates at own mean: +.16), and it
  has little power below answers f ~ .6, where the path's late families are ~0; compare lesions at the same checkpoint
  and answers level (stage_matched), not pooled over checkpoints. The family f averages rows of different noise; the
  answers family includes corner answers (the row the interaction lesion moves most), so the figures and numbers_final
  split it into the four core rows and corner answers, whose single row rests on 2,400 forward-set episodes (wider
  intervals). The path test and the rewind point use the 5-row answers f (on the core answers' f the same:
  numbers_final). Before 640 the network is below naive and gap fractions are large (the |df|-matched answers values
  +.10 to +2.15). Resample donors are other cells of the same environment (their clean drive). The splitting rows were
  not scored. Not done here: the relayed request to start the cumulation plot from lag -8 concerns another figure (not
  touched).

COSTS
  First pass (109 conditions, probes + forward set each): 385-923 s per checkpoint with three workers on GPU 0 beside
  other jobs (definitions 41-156 s); peak GPU memory allocated 1.4 GB (definitions) and 0.9 GB (conditions). Second
  pass (31 conditions, cross-fits run twice): 103-311 s per checkpoint. Third pass (33 conditions with the subspace
  controls, 22 without; ESN 22): 417-556 s per window checkpoint with two workers on a shared GPU 0 (definitions
  230-339 s), 279 s at the last two; the six other Trained-RNN checkpoints (80-640, 1,920) were run by a GPU-1 helper
  process started by another agent with this module's run_third (57-60 s each; their intact rows equal the first
  pass's exactly); peak 1.4 GB allocated. Fourth pass (3 conditions): 18-22 s per checkpoint. numbers / figure /
  table: about 1-2 min of CPU.
  Final round: no forwards; numbers (with numbers_final) and both figures about 1 min of CPU.

RUN (from the repository root; GPU 0 only; 2 CPU threads per process; start only when the load average is < 18)
    CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
        /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_lesions.py checks
    ... td_tt_lesions.py run trained 100,200,400,600,800,1000,1200,1400,1600,2000,2400,3000,4000,5000,7000,10000,15000,
        20000,30000,49995          (first pass; split the list over up to 3 workers; existing files are skipped)
    ... td_tt_lesions.py run echo 1000,2000,5000,15000,49995
    ... td_tt_lesions.py check_self
    ... td_tt_lesions.py extra trained <the same list>      (second pass)
    ... td_tt_lesions.py extra echo 1000,2000,5000,15000,49995
    ... td_tt_lesions.py check_third
    ... td_tt_lesions.py third trained <the same list>      (third pass; subspace controls at W3_SUB only)
    ... td_tt_lesions.py third echo 1000,2000,5000,15000,49995
    ... td_tt_lesions.py fourth trained <the same list>     (fourth pass)
    ... td_tt_lesions.py fourth echo 1000,2000,5000,15000,49995
    MPLBACKEND=Agg ... td_tt_lesions.py numbers | figure | table      (CPU)

NUMBERS (td_tt_lesions.py numbers, verbatim, wrapped)
  trained: 20 checkpoints (80-39,996 training episodes), 174 conditions; second-pass intact = first-pass intact to 0e+00
  VALIDATION trained_2100: lean forward vs engine max |d log q| 1e-04, answers differ at 1 of 6,003,360; joint-factor
      trace vs ck fac rel 1e-07, drive variance rel 1e-07, tau rel 6e-08; clamp residual 1e-06, kept-span residual
      8e-07, step-1 log q 2e-06
  VALIDATION trained_49995: lean forward vs engine max |d log q| 3e-05, answers differ at 0 of 6,003,360; joint-factor
      trace vs ck fac rel 1e-07, drive variance rel 1e-07, tau rel 6e-08; clamp residual 8e-07, kept-span residual
      8e-07, step-1 log q 2e-06
  VALIDATION echo_2100: lean forward vs engine max |d log q| 2e-05, answers differ at 6 of 6,003,360; joint-factor trace
      vs ck fac rel 8e-08, drive variance rel 1e-07, tau rel 6e-08; clamp residual 4e-07, kept-span residual 2e-07,
      step-1 log q 2e-06
  VALIDATION f at own mean: trained_5000 |f - own mean| 0e+00, no unit = intact 0e+00; echo_5000 |f - own mean| 0e+00,
      no unit = intact 0e+00
  VALIDATION third pass trained_5000: f at given values |f - given| 0e+00, given = own mean vs 'self' max |d log q|
      0e+00; g rows in the g-free sets 0; add5 / pure5 / pc5 / pcspan5: orthonormal to 2e-13, |Q^T V| 2e-15, variance vs
      the engine's captured_variance (clamp basis) rel 1e-13; clamp residual 1e-06; pure5 main-effect share -1e-19
      (main-effect span 18-18 dims); factor split vs subspace variance rel 5e-08, all of Pc u vs tr(Pc C Pc) rel 2e-08
  VALIDATION third pass echo_5000: f at given values |f - given| 0e+00, given = own mean vs 'self' max |d log q| 0e+00;
      g rows in the g-free sets 0
  VALIDATION third pass: its intact network = the first pass's to 0e+00
  VALIDATION fourth pass: its intact network = the first pass's to 0e+00
  DEFINITIONS (clean probe run of the checkpoint)
    800: tau > 5 / > 10: 17 / 1 units; top 5% mean tau 4.7 (final set here 2.6, 16 of 50 shared); c-matched / target c
        variance 0.25 (5%) 0.25 (10%); same-total-c-variance sets 841 / 841 units (0.73 / 0.53 of the variance)
       u variance 32.7 per step; r0 x r1 factor 0.023 of it; 5 dims hold 0.170 of u, 0.81 of the factor (3 dims 0.140,
           0.77); split-half cos^2 0.62; cross-fit on held-out half 0.161; r0 / r1 5 dims 0.131 / 0.129; random match
           211 dims
       drive variance i / f / g / o 0.35 / 0.18 / 0.10 / 0.37; keep 20 / 50 / 100 / 200 PCs remove 0.33 / 0.16 / 0.09 /
           0.04 of u's variance
    4,000: tau > 5 / > 10: 43 / 11 units; top 5% mean tau 9.1 (final set here 7.5, 31 of 50 shared); c-matched / target
        c variance 0.46 (5%) 0.41 (10%); same-total-c-variance sets 624 / 869 units (1.01 / 0.84 of the variance)
       u variance 38.6 per step; r0 x r1 factor 0.031 of it; 5 dims hold 0.102 of u, 0.87 of the factor (3 dims 0.086,
           0.80); split-half cos^2 0.73; cross-fit on held-out half 0.092; r0 / r1 5 dims 0.109 / 0.109; random match
           108 dims
       drive variance i / f / g / o 0.35 / 0.16 / 0.09 / 0.40; keep 20 / 50 / 100 / 200 PCs remove 0.50 / 0.31 / 0.19 /
           0.10 of u's variance
    39,996: tau > 5 / > 10: 99 / 38 units; top 5% mean tau 14.2 (final set here 14.2, 50 of 50 shared); c-matched /
        target c variance 1.00 (5%) 1.01 (10%); same-total-c-variance sets 57 / 208 units (0.87 / 0.98 of the variance)
       u variance 15.9 per step; r0 x r1 factor 0.039 of it; 5 dims hold 0.091 of u, 0.87 of the factor (3 dims 0.077,
           0.78); split-half cos^2 0.79; cross-fit on held-out half 0.082; r0 / r1 5 dims 0.108 / 0.103; random match 94
           dims
       drive variance i / f / g / o 0.27 / 0.22 / 0.09 / 0.41; keep 20 / 50 / 100 / 200 PCs remove 0.51 / 0.30 / 0.17 /
           0.08 of u's variance
  EFFECTS, gap units, answers / holding / coupling (window means; early 800-1,600, late >= 2,400): change dF of the
      lesioned network, then specific damage = control f - lesioned f per control
    joint5       early dF -.29 [-.32,-.27] / -.07 [-.08,-.05] / -.11 [-.13,-.09]
                       vs matched           +.14 [+.12,+.17] / +.03 [+.01,+.04] / +.07 [+.05,+.08]
                       vs r0/r1             +.17 [+.15,+.19] / +.03 [+.02,+.04] / +.04 [+.02,+.05]
                       vs r0+r1 5 dims      +.08 [+.05,+.10] / +.03 [+.02,+.04] / +.01 [-.01,+.04]
                       vs random in top PCs +.14 [+.12,+.17] / +.02 [+.01,+.03] / +.05 [+.03,+.06]
                       vs top-5 PCs         -.13 [-.16,-.10] / +.00 [-.00,+.01] / -.01 [-.02,+.00]
    joint5       late  dF -.34 [-.35,-.33] / -.51 [-.54,-.48] / -.37 [-.39,-.35]
                       vs matched           +.17 [+.16,+.19] / +.25 [+.22,+.28] / +.24 [+.22,+.26]
                       vs r0/r1             +.13 [+.11,+.16] / +.29 [+.27,+.32] / +.12 [+.10,+.13]
                       vs r0+r1 5 dims      +.01 [-.03,+.05] / +.22 [+.18,+.26] / +.03 [+.01,+.06]
                       vs random in top PCs +.22 [+.20,+.24] / +.19 [+.16,+.22] / +.13 [+.11,+.15]
                       vs top-5 PCs         -.22 [-.27,-.18] / +.01 [-.01,+.02] / -.05 [-.07,-.03]
    joint5_xfit  early dF -.27 [-.28,-.25] / -.07 [-.08,-.05] / -.10 [-.12,-.08]
                       vs r0/r1             +.15 [+.13,+.17] / +.03 [+.02,+.04] / +.03 [+.01,+.04]
                       vs matched           +.12 [+.09,+.14] / +.03 [+.01,+.04] / +.05 [+.04,+.06]
    joint5_xfit  late  dF -.30 [-.32,-.29] / -.50 [-.53,-.47] / -.36 [-.38,-.34]
                       vs r0/r1             +.11 [+.08,+.13] / +.29 [+.26,+.32] / +.12 [+.10,+.13]
                       vs matched           +.14 [+.12,+.16] / +.24 [+.21,+.27] / +.23 [+.21,+.25]
    add5         early dF -.22 [-.24,-.19] / -.04 [-.05,-.03] / -.10 [-.12,-.08]
                       vs matched           +.03 [+.01,+.06] / -.00 [-.01,+.00] / +.06 [+.04,+.08]
    add5         late  dF -.33 [-.37,-.29] / -.29 [-.32,-.26] / -.34 [-.36,-.32]
                       vs matched           -.00 [-.03,+.03] / -.01 [-.04,+.02] / +.18 [+.16,+.20]
    pure5        early dF -.18 [-.20,-.16] / -.05 [-.06,-.04] / -.03 [-.04,-.01]
                       vs matched           +.12 [+.10,+.14] / +.04 [+.03,+.05] / +.01 [-.00,+.03]
    pure5        late  dF -.28 [-.30,-.27] / -.44 [-.47,-.41] / -.23 [-.25,-.21]
                       vs matched           +.19 [+.17,+.21] / +.30 [+.27,+.33] / +.18 [+.16,+.20]
    pc5          early dF -.43 [-.46,-.40] / -.06 [-.08,-.05] / -.12 [-.14,-.10]
                       vs matched           +.15 [+.12,+.18] / +.01 [+.00,+.01] / +.06 [+.05,+.08]
    pc5          late  dF -.56 [-.61,-.52] / -.50 [-.53,-.48] / -.43 [-.45,-.40]
                       vs matched           +.17 [+.14,+.20] / +.12 [+.10,+.14] / +.21 [+.19,+.24]
    res_i        early dF -.19 [-.21,-.17] / -.02 [-.02,-.01] / -.05 [-.06,-.03]
                       vs matched           -.44 [-.47,-.41] / -.02 [-.03,-.01] / -.07 [-.09,-.05]
                       vs random            -.23 [-.25,-.21] / -.01 [-.02,-.01] / -.03 [-.05,-.02]
                       vs random non-g rows +.03 [+.01,+.04] / +.01 [+.00,+.01] / +.00 [-.00,+.01]
    res_i        late  dF -.39 [-.40,-.38] / -.31 [-.33,-.29] / -.22 [-.24,-.21]
                       vs matched           -.49 [-.52,-.47] / -.15 [-.17,-.14] / -.21 [-.23,-.19]
                       vs random            -.24 [-.25,-.22] / -.08 [-.09,-.06] / -.12 [-.14,-.11]
                       vs random non-g rows +.05 [+.05,+.06] / +.02 [+.01,+.03] / +.02 [+.01,+.03]
    res_f        early dF -.11 [-.12,-.10] / -.01 [-.02,-.01] / -.01 [-.02,-.00]
                       vs matched           -.22 [-.24,-.20] / -.02 [-.03,-.01] / -.07 [-.08,-.06]
                       vs random            -.31 [-.33,-.29] / -.02 [-.03,-.01] / -.07 [-.08,-.06]
                       vs random non-g rows -.05 [-.06,-.04] / +.00 [-.00,+.00] / -.03 [-.04,-.03]
    res_f        late  dF -.15 [-.16,-.14] / -.11 [-.12,-.10] / -.07 [-.08,-.07]
                       vs matched           -.41 [-.43,-.39] / -.28 [-.30,-.26] / -.26 [-.28,-.25]
                       vs random            -.48 [-.50,-.46] / -.27 [-.29,-.26] / -.27 [-.29,-.26]
                       vs random non-g rows -.19 [-.20,-.18] / -.17 [-.19,-.16] / -.13 [-.14,-.12]
    res_fi       early dF -.31 [-.33,-.29] / -.03 [-.03,-.02] / -.05 [-.06,-.03]
                       vs random            -.30 [-.33,-.28] / -.02 [-.02,-.01] / -.08 [-.10,-.06]
                       vs random non-g rows -.09 [-.10,-.07] / -.00 [-.01,+.00] / -.04 [-.06,-.03]
    res_fi       late  dF -.50 [-.52,-.48] / -.34 [-.36,-.32] / -.26 [-.28,-.24]
                       vs random            -.42 [-.44,-.40] / -.13 [-.14,-.12] / -.22 [-.24,-.21]
                       vs random non-g rows -.13 [-.15,-.12] / -.07 [-.07,-.06] / -.09 [-.10,-.08]
    res_g        early dF -.74 [-.78,-.69] / -.05 [-.06,-.03] / -.14 [-.16,-.12]
                       vs matched           +.69 [+.65,+.74] / +.04 [+.03,+.06] / +.12 [+.10,+.14]
                       vs random            +.32 [+.29,+.35] / +.01 [+.01,+.02] / +.06 [+.05,+.08]
                       vs random non-g rows +.57 [+.53,+.62] / +.04 [+.02,+.05] / +.10 [+.08,+.12]
    res_g        late  dF -1.10 [-1.15,-1.05] / -.51 [-.53,-.48] / -.54 [-.57,-.52]
                       vs matched           +1.00 [+.95,+1.05] / +.37 [+.34,+.39] / +.45 [+.42,+.47]
                       vs random            +.47 [+.43,+.51] / +.12 [+.11,+.13] / +.20 [+.18,+.21]
                       vs random non-g rows +.76 [+.72,+.81] / +.22 [+.20,+.24] / +.34 [+.32,+.36]
    res_o        early dF -.33 [-.35,-.30] / -.02 [-.03,-.02] / -.09 [-.11,-.08]
                       vs matched           -.31 [-.34,-.29] / -.02 [-.03,-.01] / -.02 [-.03,-.01]
                       vs random            -.09 [-.11,-.08] / -.01 [-.01,-.01] / +.02 [+.01,+.03]
                       vs random non-g rows +.16 [+.14,+.18] / +.01 [+.01,+.02] / +.05 [+.04,+.06]
    res_o        late  dF -.59 [-.61,-.57] / -.40 [-.42,-.38] / -.37 [-.39,-.35]
                       vs matched           -.43 [-.46,-.40] / -.09 [-.11,-.07] / -.14 [-.15,-.13]
                       vs random            -.04 [-.05,-.03] / +.02 [+.01,+.03] / +.03 [+.02,+.04]
                       vs random non-g rows +.25 [+.24,+.27] / +.12 [+.10,+.13] / +.17 [+.15,+.18]
    mean_i       early dF -.20 [-.22,-.18] / -.03 [-.04,-.02] / -.05 [-.06,-.03]
                       vs matched           -.57 [-.62,-.53] / -.02 [-.03,-.00] / -.08 [-.10,-.05]
                       vs random            -.24 [-.26,-.21] / -.01 [-.02,-.00] / -.03 [-.04,-.01]
                       vs random non-g rows +.05 [+.03,+.06] / +.01 [+.00,+.03] / +.00 [-.01,+.01]
    mean_i       late  dF -.36 [-.39,-.33] / -.37 [-.40,-.34] / -.19 [-.20,-.17]
                       vs matched           -.55 [-.59,-.52] / -.15 [-.18,-.12] / -.22 [-.24,-.20]
                       vs random            -.21 [-.23,-.19] / -.09 [-.11,-.07] / -.14 [-.16,-.13]
                       vs random non-g rows +.05 [+.04,+.07] / +.04 [+.02,+.07] / +.01 [-.00,+.01]
    mean_f       early dF -.13 [-.15,-.12] / -.02 [-.02,-.01] / -.02 [-.03,-.00]
                       vs matched           -.19 [-.21,-.16] / -.01 [-.02,-.01] / -.05 [-.07,-.04]
                       vs random            -.30 [-.33,-.27] / -.02 [-.03,-.01] / -.05 [-.07,-.04]
                       vs random non-g rows -.02 [-.03,-.00] / +.00 [-.00,+.01] / -.02 [-.03,-.02]
    mean_f       late  dF -.16 [-.18,-.14] / -.14 [-.15,-.12] / -.06 [-.06,-.05]
                       vs matched           -.35 [-.37,-.33] / -.32 [-.34,-.29] / -.28 [-.30,-.27]
                       vs random            -.41 [-.43,-.39] / -.33 [-.35,-.30] / -.27 [-.29,-.25]
                       vs random non-g rows -.15 [-.16,-.14] / -.19 [-.21,-.17] / -.12 [-.14,-.11]
    mean_fi      early dF -.38 [-.41,-.35] / -.04 [-.05,-.03] / -.05 [-.07,-.03]
                       vs random            -.35 [-.38,-.31] / -.01 [-.02,+.00] / -.09 [-.11,-.06]
                       vs random non-g rows -.02 [-.03,+.00] / +.01 [-.00,+.01] / -.04 [-.05,-.03]
    mean_fi      late  dF -.54 [-.59,-.51] / -.43 [-.46,-.40] / -.23 [-.25,-.21]
                       vs random            -.37 [-.40,-.34] / -.12 [-.15,-.09] / -.25 [-.27,-.23]
                       vs random non-g rows -.05 [-.07,-.04] / -.03 [-.05,-.02] / -.12 [-.14,-.11]
    mean_g       early dF -.94 [-1.01,-.89] / -.05 [-.07,-.03] / -.13 [-.16,-.10]
                       vs matched           +.90 [+.85,+.97] / +.05 [+.03,+.06] / +.11 [+.08,+.14]
                       vs random            +.51 [+.46,+.56] / +.01 [-.00,+.02] / +.06 [+.04,+.09]
                       vs random non-g rows +.79 [+.74,+.85] / +.04 [+.02,+.05] / +.09 [+.06,+.12]
    mean_g       late  dF -1.19 [-1.25,-1.12] / -.58 [-.61,-.55] / -.51 [-.54,-.48]
                       vs matched           +1.11 [+1.05,+1.17] / +.44 [+.41,+.46] / +.43 [+.41,+.46]
                       vs random            +.62 [+.57,+.67] / +.12 [+.10,+.14] / +.18 [+.16,+.20]
                       vs random non-g rows +.88 [+.82,+.94] / +.25 [+.22,+.28] / +.33 [+.31,+.35]
    mean_o       early dF -.24 [-.26,-.22] / -.01 [-.02,+.01] / -.08 [-.10,-.07]
                       vs matched           -.54 [-.59,-.49] / -.05 [-.08,-.03] / -.02 [-.05,+.00]
                       vs random            -.19 [-.22,-.17] / -.03 [-.05,-.02] / +.01 [-.00,+.02]
                       vs random non-g rows +.09 [+.07,+.11] / -.01 [-.02,+.00] / +.04 [+.03,+.05]
    mean_o       late  dF -.45 [-.48,-.41] / -.41 [-.44,-.39] / -.34 [-.37,-.32]
                       vs matched           -.58 [-.62,-.53] / -.16 [-.18,-.14] / -.15 [-.17,-.13]
                       vs random            -.12 [-.14,-.11] / -.05 [-.07,-.03] / +.01 [-.00,+.03]
                       vs random non-g rows +.14 [+.12,+.16] / +.08 [+.07,+.11] / +.16 [+.14,+.18]
    fclamp_own10 early dF -.10 [-.11,-.09] / +.00 [-.01,+.01] / -.02 [-.03,-.01]
                       vs matched           +.04 [+.02,+.05] / -.01 [-.02,+.01] / +.02 [+.01,+.03]
                       vs random            +.08 [+.07,+.10] / -.00 [-.01,+.00] / +.01 [+.01,+.02]
                       vs same total c var. -.07 [-.09,-.05] / -.01 [-.03,+.01] / -.01 [-.03,+.00]
                       vs random, same |df| +.06 [+.04,+.07] / -.01 [-.02,+.00] / +.01 [+.00,+.02]
    fclamp_own10 late  dF -.24 [-.26,-.22] / -.17 [-.18,-.15] / -.16 [-.17,-.15]
                       vs matched           +.19 [+.17,+.21] / +.13 [+.11,+.15] / +.14 [+.13,+.15]
                       vs random            +.22 [+.20,+.24] / +.14 [+.13,+.16] / +.14 [+.13,+.15]
                       vs same total c var. +.12 [+.10,+.13] / +.07 [+.05,+.08] / +.07 [+.06,+.09]
                       vs random, same |df| -.01 [-.03,+.00] / -.04 [-.07,-.02] / +.04 [+.03,+.06]
    fclamp_fin10 early dF -.03 [-.04,-.02] / -.00 [-.01,+.00] / -.02 [-.03,-.01]
                       vs matched           -.02 [-.03,-.01] / +.00 [-.00,+.01] / +.02 [+.01,+.03]
                       vs random            +.02 [+.01,+.02] / +.00 [-.01,+.01] / +.02 [+.01,+.02]
                       vs random, same |df| +.02 [+.01,+.03] / -.00 [-.01,+.01] / +.02 [+.01,+.02]
    fclamp_fin10 late  dF -.21 [-.23,-.19] / -.17 [-.18,-.15] / -.16 [-.17,-.14]
                       vs matched           +.17 [+.15,+.19] / +.13 [+.11,+.15] / +.14 [+.13,+.15]
                       vs random            +.19 [+.18,+.21] / +.14 [+.12,+.16] / +.14 [+.13,+.15]
                       vs random, same |df| -.00 [-.02,+.01] / -.02 [-.04,+.00] / +.06 [+.04,+.07]
    fself_own10  early dF -.03 [-.04,-.03] / -.01 [-.01,-.00] / +.00 [-.00,+.01]
                       vs random            +.02 [+.02,+.03] / +.00 [+.00,+.01] / -.00 [-.01,+.00]
    fself_own10  late  dF -.04 [-.05,-.04] / -.03 [-.04,-.02] / -.01 [-.01,-.01]
                       vs random            +.03 [+.02,+.03] / +.02 [+.01,+.02] / +.00 [+.00,+.01]
    fself_all    early dF -.18 [-.20,-.16] / -.02 [-.03,-.01] / -.02 [-.03,-.01]
    fself_all    late  dF -.20 [-.22,-.18] / -.15 [-.18,-.13] / -.07 [-.08,-.07]
    cclamp_own10 early dF -.61 [-.66,-.56] / -.01 [-.03,+.00] / -.03 [-.05,-.02]
                       vs matched           +.10 [+.06,+.14] / -.03 [-.05,-.02] / -.01 [-.03,+.02]
                       vs random            +.42 [+.37,+.47] / -.02 [-.03,+.00] / +.02 [+.00,+.04]
                       vs same total c var. -1.00 [-1.17,-.85] / -.09 [-.11,-.07] / +.12 [+.09,+.15]
    cclamp_own10 late  dF -.90 [-.94,-.85] / -.30 [-.33,-.28] / -.28 [-.30,-.26]
                       vs matched           +.24 [+.19,+.28] / -.05 [-.08,-.02] / +.07 [+.05,+.10]
                       vs random            +.56 [+.52,+.60] / +.00 [-.02,+.03] / +.14 [+.12,+.15]
                       vs same total c var. -.86 [-.96,-.76] / -.31 [-.33,-.28] / -.07 [-.10,-.05]
    keep20       early dF -.61 [-.66,-.57] / -.06 [-.07,-.04] / -.06 [-.08,-.04]
                       vs matched           +.13 [+.10,+.16] / -.00 [-.01,+.00] / +.01 [-.00,+.03]
    keep20       late  dF -.98 [-1.03,-.93] / -.51 [-.55,-.47] / -.31 [-.33,-.28]
                       vs matched           +.12 [+.09,+.16] / -.01 [-.04,+.01] / -.03 [-.05,-.00]
    keep100      early dF -.13 [-.14,-.11] / -.02 [-.03,-.02] / -.01 [-.02,-.00]
                       vs matched           -.00 [-.02,+.01] / +.00 [-.00,+.01] / -.01 [-.02,-.00]
    keep100      late  dF -.28 [-.30,-.26] / -.26 [-.28,-.23] / -.11 [-.12,-.10]
                       vs matched           -.06 [-.08,-.04] / -.03 [-.05,-.01] / -.03 [-.04,-.02]
    intact f, early / late: answers 0.44 / 0.92; holding 0.03 / 0.62; coupling 0.11 / 0.58
    intact f per checkpoint (80, 160, 320, 480, 640, 800, 960, 1,120, 1,280, 1,600, 1,920, 2,400, 3,200, 4,000, 5,600,
        8,000, 12,000, 16,000, 24,000, 39,996):
      answers  -2.09 -1.05 -.67 -.33 -.04 +.15 +.35 +.47 +.54 +.67 +.72 +.79 +.87 +.89 +.93 +.95 +.96 +.96 +.98 +.98
      holding  -.08 -.07 -.06 -.06 -.03 -.02 -.00 +.03 +.04 +.11 +.18 +.23 +.36 +.43 +.59 +.68 +.75 +.79 +.86 +.89
      coupling +.09 -.01 +.00 +.05 +.05 +.08 +.07 +.11 +.11 +.16 +.20 +.25 +.33 +.39 +.51 +.59 +.69 +.75 +.82 +.89
  ABSOLUTE UNITS, late window, change from intact: goal accuracy / held per variable (corner ends) / excess angle (deg);
      belief change KL(intact || lesioned)
    joint5       -0.100 [-0.106,-0.094] / -0.497 [-0.527,-0.463] / -5.87 [-6.38,-5.35]; KL 0.316
    joint5_xfit  -0.086 [-0.092,-0.080] / -0.487 [-0.518,-0.453] / -5.68 [-6.19,-5.16]; KL 0.275
    add5         -0.147 [-0.156,-0.138] / -0.296 [-0.323,-0.267] / -4.25 [-4.59,-3.91]; KL 0.634
    pure5        -0.056 [-0.062,-0.051] / -0.436 [-0.464,-0.404] / -3.25 [-3.62,-2.86]; KL 0.164
    pc5          -0.211 [-0.222,-0.200] / -0.489 [-0.519,-0.457] / -5.67 [-6.21,-5.12]; KL 0.861
    res_i        -0.130 [-0.136,-0.124] / -0.294 [-0.312,-0.275] / -2.97 [-3.27,-2.64]; KL 0.299
    res_f        -0.050 [-0.054,-0.046] / -0.105 [-0.114,-0.095] / -1.03 [-1.17,-0.88]; KL 0.084
    res_fi       -0.176 [-0.183,-0.169] / -0.323 [-0.342,-0.302] / -3.30 [-3.64,-2.94]; KL 0.474
    res_g        -0.413 [-0.426,-0.399] / -0.452 [-0.474,-0.429] / -5.62 [-6.23,-5.03]; KL 6.098
    res_o        -0.231 [-0.241,-0.221] / -0.373 [-0.396,-0.351] / -4.95 [-5.39,-4.50]; KL 1.026
    mean_i       -0.126 [-0.136,-0.117] / -0.344 [-0.373,-0.315] / -2.48 [-2.80,-2.18]; KL 0.261
    mean_f       -0.051 [-0.057,-0.045] / -0.133 [-0.150,-0.118] / -0.66 [-0.79,-0.53]; KL 0.092
    mean_fi      -0.201 [-0.213,-0.189] / -0.402 [-0.431,-0.371] / -2.87 [-3.23,-2.53]; KL 0.509
    mean_g       -0.415 [-0.431,-0.400] / -0.533 [-0.563,-0.500] / -4.75 [-5.30,-4.19]; KL 4.169
    mean_o       -0.181 [-0.191,-0.172] / -0.385 [-0.418,-0.352] / -4.95 [-5.45,-4.48]; KL 0.514
    fclamp_own10 -0.113 [-0.119,-0.107] / -0.156 [-0.171,-0.142] / -2.30 [-2.50,-2.09]; KL 0.204
    fclamp_fin10 -0.101 [-0.107,-0.095] / -0.163 [-0.179,-0.147] / -2.26 [-2.47,-2.04]; KL 0.185
    fself_own10  -0.012 [-0.014,-0.010] / -0.032 [-0.039,-0.025] / -0.22 [-0.29,-0.15]; KL 0.016
    fself_all    -0.072 [-0.081,-0.064] / -0.161 [-0.182,-0.140] / -1.27 [-1.48,-1.07]; KL 0.129
    cclamp_own10 -0.330 [-0.341,-0.319] / -0.296 [-0.320,-0.271] / -3.37 [-3.69,-3.06]; KL 1.213
    keep20       -0.373 [-0.387,-0.359] / -0.472 [-0.508,-0.434] / -1.68 [-2.08,-1.28]; KL 1.868
    keep100      -0.114 [-0.121,-0.108] / -0.250 [-0.281,-0.223] / -0.56 [-0.76,-0.38]; KL 0.236
  SHARES, late window: share of each family's intact f removed, -mean dF / mean f_intact, answers / holding / coupling
    joint5       +.37 [+.35,+.39] / +.82 [+.80,+.84] / +.64 [+.61,+.67]
    joint5_xfit  +.33 [+.31,+.35] / +.80 [+.78,+.83] / +.62 [+.59,+.65]
    add5         +.36 [+.31,+.40] / +.47 [+.42,+.52] / +.59 [+.54,+.64]
    pure5        +.31 [+.29,+.32] / +.71 [+.68,+.73] / +.40 [+.37,+.42]
    pc5          +.61 [+.56,+.66] / +.81 [+.79,+.83] / +.74 [+.69,+.78]
    res_i        +.42 [+.40,+.44] / +.50 [+.46,+.53] / +.39 [+.36,+.41]
    res_f        +.16 [+.15,+.17] / +.18 [+.16,+.20] / +.12 [+.12,+.13]
    res_fi       +.54 [+.51,+.57] / +.55 [+.52,+.59] / +.45 [+.42,+.48]
    res_g        +1.19 [+1.13,+1.25] / +.82 [+.80,+.84] / +.93 [+.89,+.99]
    res_o        +.64 [+.61,+.66] / +.65 [+.62,+.68] / +.64 [+.60,+.68]
    mean_i       +.39 [+.36,+.43] / +.60 [+.54,+.65] / +.32 [+.29,+.35]
    mean_f       +.17 [+.15,+.19] / +.22 [+.18,+.25] / +.10 [+.09,+.11]
    mean_fi      +.59 [+.55,+.64] / +.70 [+.64,+.74] / +.40 [+.37,+.43]
    mean_g       +1.29 [+1.21,+1.37] / +.94 [+.90,+.97] / +.88 [+.83,+.93]
    mean_o       +.48 [+.44,+.52] / +.67 [+.63,+.71] / +.59 [+.55,+.63]
    fclamp_own10 +.26 [+.23,+.28] / +.27 [+.23,+.30] / +.28 [+.25,+.30]
    fclamp_fin10 +.23 [+.21,+.25] / +.27 [+.23,+.30] / +.27 [+.24,+.29]
    fself_own10  +.05 [+.04,+.05] / +.05 [+.04,+.06] / +.02 [+.01,+.02]
    fself_all    +.22 [+.19,+.24] / +.25 [+.20,+.30] / +.12 [+.11,+.14]
    cclamp_own10 +.97 [+.92,+1.03] / +.49 [+.46,+.52] / +.48 [+.45,+.52]
    keep20       +1.07 [+1.01,+1.12] / +.83 [+.77,+.87] / +.53 [+.49,+.56]
    keep100      +.30 [+.28,+.33] / +.41 [+.38,+.44] / +.19 [+.17,+.21]
  PER ROW joint5, late dF: acc_goal_T -.32 [-.34,-.31]; acc_both_T -.35 [-.37,-.33]; far_T -.23 [-.25,-.20]; merged_T
      -.15 [-.17,-.13]; corner_truth_fwd -.65 [-.71,-.58]; held_joint_fwd -.46 [-.49,-.43]; held_two_fwd -.52
      [-.55,-.48]; held_one_fwd -.43 [-.48,-.39]; dist_out_fwd -.63 [-.67,-.60]; emass_slope -.51 [-.55,-.47]; excess
      -.52 [-.55,-.49]; G -.30 [-.33,-.27]; band4 -.33 [-.37,-.29]; angle_out_fwd -.41 [-.45,-.36]; same_diag_fwd -.44
      [-.48,-.40]; dir_slope -.24 [-.26,-.22]
  PER ROW joint5_vm_s0, late dF: acc_goal_T -.19 [-.21,-.17]; acc_both_T -.19 [-.21,-.18]; far_T -.08 [-.10,-.07];
      merged_T -.06 [-.08,-.05]; corner_truth_fwd -.23 [-.28,-.19]; held_joint_fwd -.26 [-.29,-.23]; held_two_fwd -.25
      [-.28,-.22]; held_one_fwd -.27 [-.30,-.24]; dist_out_fwd -.25 [-.28,-.22]; emass_slope -.22 [-.25,-.19]; excess
      -.13 [-.15,-.11]; G -.09 [-.12,-.07]; band4 -.09 [-.11,-.07]; angle_out_fwd -.21 [-.24,-.18]; same_diag_fwd -.08
      [-.10,-.06]; dir_slope -.13 [-.16,-.11]
  COMMITTED to a corner at corner splits (naive .554, Bayes 0), late: intact 0.449; joint5 0.386; joint5_vm_s0 0.620;
      r0_5 0.674; res_fi 0.528; fclamp_own10 0.620
  PATH RESIDUAL, late window: late-family f minus the intact path's at equal answers f; late / holding / coupling
    joint5           +.05 [+.03,+.08] / +.04 [+.00,+.06] / +.07 [+.04,+.10]
    joint5_xfit      +.05 [+.02,+.08] / +.03 [-.01,+.06] / +.07 [+.04,+.10]
    add5             +.16 [+.14,+.20] / +.24 [+.20,+.27] / +.09 [+.06,+.13]
    pure5            +.14 [+.11,+.16] / +.08 [+.04,+.11] / +.19 [+.17,+.22]
    pc5              +.08 [+.07,+.10] / +.11 [+.09,+.12] / +.06 [+.03,+.10]
    res_i            +.25 [+.23,+.27] / +.27 [+.24,+.29] / +.24 [+.21,+.27]
    res_f            +.27 [+.25,+.29] / +.27 [+.25,+.30] / +.26 [+.24,+.29]
    res_fi           +.24 [+.22,+.27] / +.26 [+.24,+.29] / +.23 [+.19,+.26]
    res_g            +.07 [+.05,+.08] / +.15 [+.13,+.16] / -.01 [-.04,+.02]
    res_o            +.18 [+.15,+.20] / +.22 [+.19,+.24] / +.13 [+.10,+.17]
    mean_i           +.23 [+.20,+.26] / +.19 [+.16,+.23] / +.27 [+.24,+.30]
    mean_f           +.27 [+.25,+.29] / +.26 [+.24,+.28] / +.28 [+.26,+.31]
    mean_fi          +.22 [+.20,+.25] / +.18 [+.15,+.21] / +.27 [+.23,+.30]
    mean_g           +.05 [+.03,+.07] / +.07 [+.05,+.09] / +.02 [-.01,+.06]
    mean_o           +.16 [+.13,+.18] / +.18 [+.15,+.20] / +.13 [+.10,+.17]
    fclamp_own10     +.28 [+.25,+.31] / +.31 [+.28,+.34] / +.24 [+.21,+.27]
    fclamp_fin10     +.26 [+.23,+.28] / +.29 [+.26,+.31] / +.22 [+.20,+.25]
    fself_own10      +.16 [+.13,+.18] / +.15 [+.12,+.17] / +.17 [+.15,+.19]
    fself_all        +.29 [+.27,+.31] / +.29 [+.26,+.31] / +.30 [+.28,+.32]
    cclamp_own10     +.29 [+.27,+.32] / +.34 [+.31,+.37] / +.24 [+.20,+.28]
    keep20           +.18 [+.16,+.21] / +.14 [+.11,+.16] / +.22 [+.19,+.26]
    keep100          +.29 [+.26,+.31] / +.26 [+.22,+.29] / +.31 [+.28,+.34]
    joint5_vm_s0     +.18 [+.15,+.20] / +.14 [+.11,+.17] / +.21 [+.19,+.24]
    joint5_vm_s1     +.18 [+.15,+.21] / +.15 [+.11,+.18] / +.22 [+.19,+.24]
    r0_5             +.19 [+.16,+.22] / +.25 [+.21,+.28] / +.14 [+.11,+.17]
    r1_5             +.16 [+.13,+.19] / +.22 [+.18,+.26] / +.11 [+.08,+.14]
    r0_5_xfit        +.20 [+.17,+.23] / +.25 [+.22,+.28] / +.15 [+.11,+.18]
    pcspan5_s0       +.02 [-.01,+.04] / -.01 [-.04,+.02] / +.04 [+.01,+.06]
    pcspan5_s1       +.02 [-.01,+.04] / -.00 [-.03,+.03] / +.04 [+.01,+.06]
    res_rand_s0      +.20 [+.18,+.22] / +.24 [+.21,+.26] / +.16 [+.12,+.20]
    res_nog_s0       +.25 [+.23,+.28] / +.26 [+.23,+.29] / +.25 [+.22,+.28]
    fclamp_own10_vt  +.18 [+.16,+.19] / +.19 [+.16,+.21] / +.17 [+.15,+.18]
    fclamp_rand10_s0 +.04 [+.03,+.06] / +.04 [+.03,+.06] / +.04 [+.03,+.07]
    fshift_own10_s0  +.31 [+.28,+.34] / +.31 [+.27,+.34] / +.32 [+.28,+.35]
    cclamp_own10_vt  +.11 [+.09,+.13] / +.04 [+.02,+.05] / +.17 [+.14,+.21]
    keep100_vm       +.29 [+.26,+.32] / +.27 [+.24,+.30] / +.31 [+.28,+.34]
  REWIND: checkpoint -> the intact checkpoint with the lesioned network's answers f (log-interpolated, path >= 640)
      (late-family f: lesioned vs path there)
    joint5       2.4k -> 1,044 (+.04 vs +.05); 4k -> 1,256 (+.09 vs +.08); 12k -> 1,509 (+.20 vs +.12); 40k -> 1,665
        (+.27 vs +.15)
    joint3       2.4k -> 1,155 (+.05 vs +.07); 4k -> 1,368 (+.10 vs +.10); 12k -> 1,747 (+.22 vs +.16); 40k -> 2,027
        (+.28 vs +.20)
    joint5_xfit  2.4k -> 1,104 (+.05 vs +.07); 4k -> 1,332 (+.10 vs +.09); 12k -> 1,612 (+.21 vs +.14); 40k -> 1,847
        (+.29 vs +.18)
    r0_5         2.4k -> 1,364 (+.10 vs +.10); 4k -> 1,561 (+.22 vs +.13); 12k -> 2,058 (+.46 vs +.20); 40k -> 2,370
        (+.58 vs +.24)
    joint5_vm_s0 2.4k -> 1,465 (+.13 vs +.11); 4k -> 1,960 (+.26 vs +.19); 12k -> 2,661 (+.52 vs +.28); 40k -> 2,948
        (+.65 vs +.32)
    res_fi       2.4k -> 962 (+.13 vs +.04); 4k -> 1,045 (+.22 vs +.05); 12k -> 1,079 (+.35 vs +.06); 40k -> 1,053 (+.42
        vs +.06)
    fclamp_own10 2.4k -> 1,726 (+.21 vs +.16); 4k -> 2,049 (+.32 vs +.20); 12k -> 1,747 (+.52 vs +.16); 40k -> 1,346
        (+.59 vs +.09)
    keep100      2.4k -> 1,266 (+.17 vs +.08); 4k -> 1,450 (+.27 vs +.11); 12k -> 1,580 (+.50 vs +.13); 40k -> 1,969
        (+.66 vs +.19)
  ONSET of the specific damage (stays above .1 to the end) [5-95%] (share crossing); family c10 / c50 (timeline medians
      [rows]): answers 838 [621-885] / 1,394 [863-1,575]; holding 1,547 [1,166-2,127] / 4,983 [3,021-9,245]; coupling
      1,039 [213-3,255] / 5,952 [2,747-11,103]
    joint5       vs matched           915 [311-1,029] (100%) / 2,842 [2,543-3,277] (100%) / 1,358 [1,252-2,735] (100%)
    joint5       vs r0/r1             591 [536-32,071] (98%) / 2,284 [1,873-2,560] (100%) / 6,263 [4,503-10,150] (100%)
    joint5_xfit  vs r0/r1             29,249 [652-never] (82%) / 2,414 [1,859-2,612] (100%) / 6,031 [4,266-8,247] (100%)
    joint5_xfit  vs matched           1,035 [961-1,377] (98%) / 2,931 [2,623-3,417] (100%) / 2,442 [1,424-3,077] (100%)
    res_g        vs matched           80 [80-405] (100%) / 1,735 [1,589-1,928] (100%) / 984 [675-1,147] (100%)
    res_g        vs random            80 [80-493] (100%) / 4,715 [4,193-5,371] (100%) / 2,286 [1,774-2,794] (100%)
    res_g        vs random non-g rows 80 [80-400] (100%) / 2,201 [1,767-2,606] (100%) / 1,092 [969-1,422] (100%)
    res_i        vs random non-g rows never / never / never
    fclamp_own10 vs matched           4,136 [3,282-4,707] (100%) / 4,845 [3,985-6,758] (100%) / 4,762 [3,804-6,342]
        (100%)
    fclamp_own10 vs random            2,673 [2,374-4,035] (100%) / 5,071 [4,472-5,988] (100%) / 4,471 [3,789-5,642]
        (100%)
    fclamp_own10 vs same total c var. 8,629 [7,716-9,473] (100%) / 11,388 [10,014-14,311] (100%) / 20,931
        [18,815-23,374] (100%)
    fclamp_own10 vs random, same |df| 37,032 [30,987-never] (76%) / never / never
    fclamp_fin10 vs matched           5,039 [4,663-5,588] (100%) / 4,781 [4,392-8,252] (100%) / 4,631 [3,943-5,906]
        (100%)
    fclamp_fin10 vs random            4,466 [4,064-4,993] (100%) / 4,908 [4,449-5,669] (100%) / 4,705 [3,980-5,739]
        (100%)
    fclamp_fin10 vs random, same |df| 36,902 [30,643-never] (76%) / never / never
  FIX ROUND 11-02
    PER ROW specific damage of joint5 [5-95%], early | late (answers rows; 'core 4' = their first four rows averaged per
        checkpoint, no corner answers)
      vs r0/r1         acc_goal_T +.12 [+.10,+.14] | -.03 [-.05,-.00]; acc_both_T +.10 [+.08,+.11] | +.00 [-.02,+.02];
          far_T +.18 [+.16,+.20] | +.01 [-.00,+.03]; merged_T +.12 [+.09,+.15] | -.03 [-.05,-.01]; corner_truth_fwd +.35
          [+.28,+.41] | +.69 [+.62,+.77]; core 4 +.13 [+.11,+.15] | -.01 [-.03,+.01]
      vs random dims   acc_goal_T +.12 [+.09,+.15] | +.12 [+.10,+.14]; acc_both_T +.12 [+.10,+.14] | +.14 [+.12,+.15];
          far_T +.24 [+.21,+.27] | +.14 [+.12,+.16]; merged_T +.18 [+.15,+.22] | +.08 [+.07,+.09]; corner_truth_fwd +.06
          [-.00,+.12] | +.40 [+.32,+.48]; core 4 +.17 [+.14,+.19] | +.12 [+.10,+.13]
      vs r0+r1 5 dims  acc_goal_T +.01 [-.02,+.03] | -.15 [-.19,-.12]; acc_both_T -.01 [-.03,+.00] | -.13 [-.15,-.11];
          far_T +.02 [-.00,+.04] | -.13 [-.16,-.10]; merged_T -.12 [-.18,-.06] | -.28 [-.35,-.21]; corner_truth_fwd +.48
          [+.41,+.55] | +.74 [+.66,+.82]; core 4 -.03 [-.05,-.00] | -.17 [-.21,-.14]
    PER ROW specific damage of joint5, late [5-95%] (holding and coupling rows)
      vs r0/r1         held_joint_fwd +.16 [+.14,+.18]; held_two_fwd +.29 [+.26,+.31]; held_one_fwd +.22 [+.17,+.28];
          dist_out_fwd +.48 [+.45,+.52]; emass_slope +.31 [+.27,+.35]; excess +.19 [+.16,+.21]; G +.03 [+.01,+.05];
          band4 -.03 [-.07,+.01]; angle_out_fwd +.07 [+.05,+.10]; same_diag_fwd +.38 [+.34,+.41]; dir_slope +.06
          [+.05,+.08]
      vs random dims   held_joint_fwd +.19 [+.16,+.21]; held_two_fwd +.25 [+.22,+.28]; held_one_fwd +.17 [+.12,+.22];
          dist_out_fwd +.37 [+.32,+.40]; emass_slope +.28 [+.23,+.32]; excess +.37 [+.35,+.40]; G +.18 [+.16,+.21];
          band4 +.23 [+.19,+.27]; angle_out_fwd +.19 [+.15,+.23]; same_diag_fwd +.35 [+.30,+.40]; dir_slope +.12
          [+.09,+.14]
      vs r0+r1 5 dims  held_joint_fwd +.11 [+.09,+.13]; held_two_fwd +.21 [+.18,+.24]; held_one_fwd +.16 [+.09,+.26];
          dist_out_fwd +.36 [+.31,+.41]; emass_slope +.25 [+.20,+.30]; excess +.14 [+.11,+.17]; G -.14 [-.17,-.10];
          band4 -.12 [-.17,-.07]; angle_out_fwd -.02 [-.05,+.00]; same_diag_fwd +.29 [+.25,+.34]; dir_slope +.02
          [+.00,+.04]
      joint5_xfit vs r0/r1 cross-fit, late: acc_goal_T -.05 [-.07,-.03]; acc_both_T -.03 [-.05,-.01]; far_T -.01
          [-.03,+.00]; merged_T -.05 [-.07,-.03]; corner_truth_fwd +.68 [+.61,+.76]; core 4 -.04 [-.05,-.02]
    joint5 answers specific damage per checkpoint vs matched: 80 +.12 [+.06,+.19]; 160 +.05 [+.00,+.11]; 320 +.11
        [+.05,+.19]; 480 +.10 [+.06,+.14]; 640 +.10 [+.06,+.14]; 800 +.09 [+.05,+.13]; 960 +.10 [+.07,+.14]; 1,120 +.16
        [+.13,+.20]; 1,280 +.15 [+.11,+.18]; 1,600 +.22 [+.18,+.25]; 1,920 +.19 [+.16,+.22]; 2,400 +.19 [+.16,+.21];
        3,200 +.19 [+.16,+.22]; 4,000 +.18 [+.15,+.20]; 5,600 +.18 [+.16,+.21]; 8,000 +.18 [+.15,+.20]; 12,000 +.17
        [+.15,+.19]; 16,000 +.16 [+.14,+.17]; 24,000 +.17 [+.15,+.19]; 39,996 +.16 [+.14,+.18]
    joint5 answers specific damage per checkpoint vs r0/r1: 80 -.00 [-.04,+.04]; 160 +.03 [-.01,+.08]; 320 +.05
        [+.01,+.09]; 480 +.05 [+.02,+.09]; 640 +.12 [+.09,+.15]; 800 +.13 [+.10,+.16]; 960 +.13 [+.11,+.16]; 1,120 +.19
        [+.16,+.22]; 1,280 +.21 [+.18,+.23]; 1,600 +.21 [+.18,+.24]; 1,920 +.21 [+.18,+.24]; 2,400 +.17 [+.14,+.19];
        3,200 +.14 [+.11,+.17]; 4,000 +.13 [+.10,+.16]; 5,600 +.15 [+.11,+.18]; 8,000 +.12 [+.09,+.15]; 12,000 +.12
        [+.09,+.15]; 16,000 +.11 [+.08,+.13]; 24,000 +.11 [+.09,+.14]; 39,996 +.13 [+.11,+.16]
    joint5 holding specific damage per checkpoint vs matched: 80:+.00 160:-.00 320:+.00 480:+.00 640:+.01 800:+.01
        960:+.02 1,120:+.02 1,280:+.02 1,600:+.05 1,920:+.06 2,400:+.07 3,200:+.12 4,000:+.16 5,600:+.22 8,000:+.27
        12,000:+.32 16,000:+.34 24,000:+.37 39,996:+.37
    joint5 coupling specific damage per checkpoint vs matched: 80:-.01 160:+.00 320:+.01 480:+.01 640:+.03 800:+.02
        960:+.04 1,120:+.06 1,280:+.09 1,600:+.13 1,920:+.11 2,400:+.11 3,200:+.15 4,000:+.15 5,600:+.18 8,000:+.23
        12,000:+.30 16,000:+.32 24,000:+.36 39,996:+.36
    GATES vs count-matched random rows without g (i, f, o blocks; 3 seeds; blocks i / f / g / o of the sets:
        340/348/0/312, 324/345/0/331, 325/334/0/341, 661/684/0/655, 676/675/0/649, 664/662/0/674)
      per checkpoint (80 ... 39,996): + / - / 0 = the specific damage's 5-95% above 0 / below 0 / neither; '> 0 from' =
          the first checkpoint from which every later one is +
      res_i    answers -000++++++++++++++++ (> 0 from 640, < 0 from never); holding 00000+0+++0+++++++++ (> 0 from
          2,400, < 0 from never); coupling +00000000+000++++++0 (> 0 from never, < 0 from never)
      res_f    answers --+0---------------- (> 0 from never, < 0 from 640); holding 00+00+0000---------- (> 0 from
          never, < 0 from 1,920); coupling +00----------------- (> 0 from never, < 0 from 480)
      res_o    answers ++0+++++++++++++++++ (> 0 from 480, < 0 from never); holding 00000+++++++++++++++ (> 0 from 800,
          < 0 from never); coupling -00+++++++++++++++++ (> 0 from 480, < 0 from never)
      res_g    answers ++++++++++++++++++++ (> 0 from 80, < 0 from never); holding -0+0++++++++++++++++ (> 0 from 640, <
          0 from never); coupling 000+++++++++++++++++ (> 0 from 480, < 0 from never)
      res_fi   answers --00---------------- (> 0 from never, < 0 from 640); holding 0-00-0000----------- (> 0 from
          never, < 0 from 1,600); coupling +0------------------ (> 0 from never, < 0 from 320)
      mean_i   answers ---0++++++++++++++++ (> 0 from 640, < 0 from never); holding 000+0++0+0++++++++++ (> 0 from
          1,920, < 0 from never); coupling 0-00000000000+000000 (> 0 from never, < 0 from never)
      mean_f   answers -0000000------------ (> 0 from never, < 0 from 1,280); holding 00+00++000---------- (> 0 from
          never, < 0 from 1,920); coupling +000---------------- (> 0 from never, < 0 from 640)
      mean_o   answers ++++++++++++++++++++ (> 0 from 80, < 0 from never); holding 000000000000++++++++ (> 0 from 3,200,
          < 0 from never); coupling -000++++++++++++++++ (> 0 from 640, < 0 from never)
      mean_g   answers ++++++++++++++++++++ (> 0 from 80, < 0 from never); holding -0+000++++++++++++++ (> 0 from 960, <
          0 from never); coupling 00++++++++++++++++++ (> 0 from 320, < 0 from never)
      mean_fi  answers -----0-000---------- (> 0 from never, < 0 from 1,920); holding 0000000000000-----00 (> 0 from
          never, < 0 from never); coupling +00----------------- (> 0 from never, < 0 from 480)
      res_i vs random non-g rows, answers: 80:-.02 160:-.01 320:-.00 480:+.01 640:+.03 800:+.03 960:+.04 1,120:+.03
          1,280:+.02 1,600:+.01 1,920:+.03 2,400:+.04 3,200:+.03 4,000:+.04 5,600:+.03 8,000:+.05 12,000:+.06
          16,000:+.07 24,000:+.08 39,996:+.09
      res_i vs random non-g rows, holding: 80:-.00 160:+.00 320:+.00 480:+.00 640:+.00 800:+.01 960:+.00 1,120:+.01
          1,280:+.01 1,600:+.01 1,920:+.01 2,400:+.02 3,200:+.01 4,000:+.02 5,600:+.01 8,000:+.02 12,000:+.03
          16,000:+.03 24,000:+.03 39,996:+.03
      res_i vs random non-g rows, coupling: 80:+.00 160:-.00 320:-.01 480:-.01 640:-.01 800:-.00 960:-.01 1,120:+.00
          1,280:+.01 1,600:+.01 1,920:+.01 2,400:+.01 3,200:+.01 4,000:+.03 5,600:+.02 8,000:+.02 12,000:+.03
          16,000:+.03 24,000:+.02 39,996:+.00
      mean_i vs random non-g rows, answers: 80:-.03 160:-.02 320:-.03 480:+.00 640:+.03 800:+.05 960:+.05 1,120:+.05
          1,280:+.04 1,600:+.04 1,920:+.03 2,400:+.06 3,200:+.03 4,000:+.04 5,600:+.05 8,000:+.05 12,000:+.06
          16,000:+.07 24,000:+.06 39,996:+.08
      mean_i vs random non-g rows, holding: 80:+.00 160:-.00 320:+.00 480:+.00 640:+.00 800:+.01 960:+.01 1,120:+.01
          1,280:+.02 1,600:+.02 1,920:+.02 2,400:+.03 3,200:+.03 4,000:+.04 5,600:+.04 8,000:+.03 12,000:+.04
          16,000:+.06 24,000:+.04 39,996:+.06
      mean_i vs random non-g rows, coupling: 80:+.00 160:-.00 320:-.01 480:+.01 640:-.01 800:-.00 960:-.01 1,120:+.00
          1,280:+.02 1,600:+.01 1,920:+.01 2,400:+.01 3,200:+.01 4,000:+.02 5,600:+.01 8,000:+.00 12,000:+.01
          16,000:+.00 24,000:-.00 39,996:+.01
    SHARE DIFFERENCES, late: holding - answers / coupling - answers (shares of each family's intact f removed) [5-95%]
      res_i        +.08 [+.04,+.11] / -.04 [-.07,+.00]
      res_f        +.02 [+.00,+.04] / -.03 [-.05,-.02]
      res_fi       +.01 [-.03,+.05] / -.09 [-.14,-.05]
      res_o        +.02 [-.03,+.05] / +.00 [-.04,+.05]
      mean_i       +.21 [+.15,+.26] / -.07 [-.12,-.03]
      mean_f       +.05 [+.01,+.08] / -.07 [-.10,-.05]
      mean_fi      +.11 [+.04,+.16] / -.19 [-.25,-.14]
      mean_o       +.18 [+.14,+.22] / +.11 [+.05,+.15]
      fclamp_own10 +.01 [-.02,+.04] / +.02 [-.01,+.05]
      fclamp_fin10 +.04 [+.00,+.07] / +.04 [+.01,+.07]
      joint5       +.45 [+.43,+.48] / +.27 [+.24,+.31]
    FORGET-GATE PERTURBATION, clean run: mean over probes, steps 1-30 and units of |f - assigned value|
      800: long-tau top 10% to median 0.40; random units to median 0.12; c-matched units to median 0.17;
          same-total-c-variance units to median 0.12; random units shifted by the targets' mean shift 0.39 (shift 0.39);
          long-tau at own mean 0.112, random at own mean 0.078; final top 10% to median 0.24, its shifted control 0.22
      1,600: long-tau top 10% to median 0.43; random units to median 0.14; c-matched units to median 0.18;
          same-total-c-variance units to median 0.14; random units shifted by the targets' mean shift 0.43 (shift 0.43);
          long-tau at own mean 0.111, random at own mean 0.096; final top 10% to median 0.30, its shifted control 0.28
      4,000: long-tau top 10% to median 0.56; random units to median 0.16; c-matched units to median 0.21;
          same-total-c-variance units to median 0.16; random units shifted by the targets' mean shift 0.54 (shift 0.55);
          long-tau at own mean 0.096, random at own mean 0.102; final top 10% to median 0.47, its shifted control 0.47
      39,996: long-tau top 10% to median 0.69; random units to median 0.18; c-matched units to median 0.23;
          same-total-c-variance units to median 0.19; random units shifted by the targets' mean shift 0.63 (shift 0.69);
          long-tau at own mean 0.046, random at own mean 0.097; final top 10% to median 0.69, its shifted control 0.63
    TIME CONSTANTS tau = 1 / (1 - mean f) (steps 2-30) under the median clamp (targets) and the shift (its |df|-matched
        random units, 3 seeds): 800: top 10% 3.5 -> 1.35; random units 1.35 -> median 2.7 (clipped at f = .99: 0%);
        1,600: top 10% 3.8 -> 1.33; random units 1.35 -> median 2.9 (clipped at f = .99: 0%); 4,000: top 10% 6.3 ->
        1.30; random units 1.34 -> median 4.2 (clipped at f = .99: 8%); 39,996: top 10% 10.5 -> 1.24; random units 1.41
        -> median 6.3 (clipped at f = .99: 26%)
      fclamp_own10 vs random, same |df|, answers +++++++++++-0---0-++ (> 0 from 24,000, < 0 from never): 80:+2.15
          160:+.56 320:+.37 480:+.20 640:+.10 800:+.11 960:+.05 1,120:+.05 1,280:+.04 1,600:+.03 1,920:+.03 2,400:-.03
          3,200:+.01 4,000:-.08 5,600:-.06 8,000:-.06 12,000:-.02 16,000:-.04 24,000:+.04 39,996:+.11
      fclamp_own10 vs random, same |df|, holding -00000000------0-0-- (> 0 from never, < 0 from 24,000): 80:-.00
          160:+.00 320:-.00 480:-.00 640:+.00 800:+.00 960:-.01 1,120:-.00 1,280:-.01 1,600:-.03 1,920:-.02 2,400:-.02
          3,200:-.04 4,000:-.04 5,600:-.05 8,000:-.03 12,000:-.05 16,000:-.03 24,000:-.05 39,996:-.08
      fclamp_own10 vs random, same |df|, coupling +00-0-00++++++++++++ (> 0 from 1,280, < 0 from never): 80:+.02
          160:+.00 320:-.01 480:-.02 640:-.01 800:-.01 960:-.00 1,120:+.01 1,280:+.02 1,600:+.04 1,920:+.03 2,400:+.04
          3,200:+.04 4,000:+.03 5,600:+.03 8,000:+.04 12,000:+.04 16,000:+.03 24,000:+.07 39,996:+.09
      fclamp_own5 vs random, same |df|, answers +++++000------------ (> 0 from never, < 0 from 1,280): 80:+1.00 160:+.28
          320:+.30 480:+.13 640:+.05 800:+.02 960:+.01 1,120:-.01 1,280:-.03 1,600:-.05 1,920:-.06 2,400:-.08 3,200:-.09
          4,000:-.10 5,600:-.07 8,000:-.04 12,000:-.05 16,000:-.05 24,000:-.07 39,996:-.09
      fclamp_own5 vs random, same |df|, holding -0000000-----0000--- (> 0 from never, < 0 from 16,000): 80:-.00 160:-.00
          320:-.00 480:+.00 640:-.00 800:+.00 960:+.00 1,120:+.00 1,280:-.02 1,600:-.02 1,920:-.02 2,400:-.02 3,200:-.02
          4,000:+.01 5,600:-.00 8,000:+.00 12,000:+.01 16,000:-.09 24,000:-.05 39,996:-.10
      fclamp_own5 vs random, same |df|, coupling +000000+0++++++++++- (> 0 from never, < 0 from 39,996): 80:+.02
          160:-.01 320:-.01 480:-.01 640:-.00 800:-.01 960:-.00 1,120:+.02 1,280:+.01 1,600:+.03 1,920:+.02 2,400:+.03
          3,200:+.02 4,000:+.05 5,600:+.05 8,000:+.03 12,000:+.08 16,000:+.03 24,000:+.06 39,996:-.02
      fclamp_fin10 vs random, same |df|, answers ++++++0+0+0-0---0-++ (> 0 from 24,000, < 0 from never): 80:+.62
          160:+.31 320:+.19 480:+.06 640:+.02 800:+.04 960:+.01 1,120:+.03 1,280:+.01 1,600:+.02 1,920:+.00 2,400:-.02
          3,200:+.02 4,000:-.04 5,600:-.04 8,000:-.05 12,000:-.02 16,000:-.04 24,000:+.04 39,996:+.11
      fclamp_fin10 vs random, same |df|, holding -00000000000-000000- (> 0 from never, < 0 from 39,996): 80:-.00
          160:+.00 320:-.00 480:-.00 640:-.00 800:+.00 960:-.01 1,120:+.01 1,280:-.00 1,600:-.00 1,920:+.01 2,400:+.01
          3,200:-.02 4,000:-.01 5,600:-.00 8,000:+.00 12,000:-.02 16,000:-.03 24,000:-.04 39,996:-.08
      fclamp_fin10 vs random, same |df|, coupling +000+00+++++++++++++ (> 0 from 1,120, < 0 from never): 80:+.02
          160:+.00 320:-.01 480:-.00 640:+.01 800:+.01 960:+.01 1,120:+.03 1,280:+.01 1,600:+.03 1,920:+.04 2,400:+.05
          3,200:+.04 4,000:+.05 5,600:+.05 8,000:+.05 12,000:+.05 16,000:+.04 24,000:+.08 39,996:+.09
    C CLAMP of the top 10%: controls' units and c variance (sum over steps of var_s(c)) relative to the targets', per
        checkpoint (training episodes: same-total-c-variance units n / variance ratio; c-matched units variance ratio;
        count-matched random units variance ratio)
      80: 842 / 0.16; 0.11; 0.02; 160: 846 / 0.46; 0.29; 0.05; 320: 843 / 0.72; 0.44; 0.08; 480: 845 / 0.66; 0.36; 0.07;
          640: 840 / 0.61; 0.32; 0.07; 800: 841 / 0.53; 0.25; 0.06; 960: 843 / 0.52; 0.24; 0.05; 1,120: 844 / 0.48;
          0.21; 0.06; 1,280: 847 / 0.55; 0.25; 0.06; 1,600: 850 / 0.58; 0.26; 0.07; 1,920: 853 / 0.62; 0.29; 0.07;
          2,400: 859 / 0.68; 0.32; 0.07; 3,200: 866 / 0.77; 0.38; 0.07; 4,000: 869 / 0.84; 0.41; 0.09; 5,600: 876 /
          0.97; 0.47; 0.10; 8,000: 816 / 1.00; 0.53; 0.12; 12,000: 575 / 1.00; 0.61; 0.15; 16,000: 498 / 1.02; 0.61;
          0.17; 24,000: 398 / 0.98; 0.71; 0.25; 39,996: 208 / 0.98; 1.01; 0.50
    SUBSPACES (third pass): share of u's clean variance (pooled over steps) and the factor split of the variance inside
        (environment / r0 / r1 / r0 x r1 / environment x cell / within cell)
      800: joint5 0.170; r0_5 0.131; r1_5 0.129; add5 0.185; pure5 0.060; pc5 0.270; pcspan5_s0 0.171; pcspan5_s1 0.170;
          top-K span K 14 / 14; random dims matched to add5 / pure5 / pc5 225 / 68 / 325; main-effect (r0 + r1
          cell-mean) variance in pure5 -4.6e-21 of u
         split joint5     0.12 / 0.14 / 0.14 / 0.11 / 0.37 / 0.12 (r0 + r1 0.28)
         split r0_5       0.14 / 0.30 / 0.04 / 0.03 / 0.35 / 0.14 (r0 + r1 0.34)
         split add5       0.12 / 0.21 / 0.23 / 0.02 / 0.31 / 0.11 (r0 + r1 0.44)
         split pure5      0.16 / 0.00 / 0.00 / 0.19 / 0.49 / 0.16 (r0 + r1 0.00)
         split pc5        0.36 / 0.12 / 0.13 / 0.06 / 0.25 / 0.08 (r0 + r1 0.25)
         split pcspan5_s0 0.42 / 0.08 / 0.07 / 0.04 / 0.29 / 0.11 (r0 + r1 0.15)
         split all        0.34 / 0.04 / 0.05 / 0.02 / 0.37 / 0.17 (r0 + r1 0.09)
      1,600: joint5 0.118; r0_5 0.108; r1_5 0.107; add5 0.175; pure5 0.044; pc5 0.227; pcspan5_s0 0.117; pcspan5_s1
          0.118; top-K span K 21 / 22; random dims matched to add5 / pure5 / pc5 189 / 46 / 250; main-effect (r0 + r1
          cell-mean) variance in pure5 -1.1e-20 of u
         split joint5     0.08 / 0.18 / 0.19 / 0.17 / 0.28 / 0.10 (r0 + r1 0.37)
         split r0_5       0.10 / 0.48 / 0.03 / 0.02 / 0.26 / 0.11 (r0 + r1 0.51)
         split add5       0.08 / 0.29 / 0.32 / 0.02 / 0.21 / 0.08 (r0 + r1 0.61)
         split pure5      0.08 / 0.00 / 0.00 / 0.35 / 0.41 / 0.16 (r0 + r1 0.00)
         split pc5        0.30 / 0.20 / 0.21 / 0.05 / 0.18 / 0.06 (r0 + r1 0.41)
         split pcspan5_s0 0.45 / 0.09 / 0.10 / 0.04 / 0.22 / 0.10 (r0 + r1 0.19)
         split all        0.33 / 0.06 / 0.06 / 0.03 / 0.36 / 0.17 (r0 + r1 0.12)
      4,000: joint5 0.102; r0_5 0.109; r1_5 0.109; add5 0.189; pure5 0.045; pc5 0.229; pcspan5_s0 0.102; pcspan5_s1
          0.102; top-K span K 27 / 25; random dims matched to add5 / pure5 / pc5 192 / 45 / 236; main-effect (r0 + r1
          cell-mean) variance in pure5 9.5e-21 of u
         split joint5     0.05 / 0.18 / 0.18 / 0.26 / 0.24 / 0.10 (r0 + r1 0.36)
         split r0_5       0.06 / 0.62 / 0.03 / 0.02 / 0.19 / 0.08 (r0 + r1 0.65)
         split add5       0.05 / 0.35 / 0.37 / 0.02 / 0.15 / 0.06 (r0 + r1 0.72)
         split pure5      0.05 / 0.00 / 0.00 / 0.49 / 0.33 / 0.14 (r0 + r1 0.00)
         split pc5        0.20 / 0.26 / 0.27 / 0.08 / 0.14 / 0.05 (r0 + r1 0.53)
         split pcspan5_s0 0.43 / 0.13 / 0.11 / 0.05 / 0.20 / 0.08 (r0 + r1 0.25)
         split all        0.30 / 0.07 / 0.07 / 0.03 / 0.36 / 0.16 (r0 + r1 0.14)
      39,996: joint5 0.091; r0_5 0.108; r1_5 0.103; add5 0.178; pure5 0.051; pc5 0.213; pcspan5_s0 0.091; pcspan5_s1
          0.091; top-K span K 36 / 35; random dims matched to add5 / pure5 / pc5 176 / 53 / 211; main-effect (r0 + r1
          cell-mean) variance in pure5 1.9e-20 of u
         split joint5     0.03 / 0.14 / 0.10 / 0.37 / 0.26 / 0.11 (r0 + r1 0.24)
         split r0_5       0.02 / 0.69 / 0.01 / 0.02 / 0.18 / 0.08 (r0 + r1 0.69)
         split add5       0.02 / 0.38 / 0.37 / 0.02 / 0.14 / 0.06 (r0 + r1 0.76)
         split pure5      0.04 / 0.00 / 0.00 / 0.55 / 0.29 / 0.12 (r0 + r1 0.00)
         split pc5        0.20 / 0.28 / 0.25 / 0.10 / 0.13 / 0.04 (r0 + r1 0.53)
         split pcspan5_s0 0.44 / 0.12 / 0.10 / 0.06 / 0.22 / 0.06 (r0 + r1 0.22)
         split all        0.33 / 0.08 / 0.07 / 0.04 / 0.36 / 0.13 (r0 + r1 0.15)
    REWIND SIGNATURES, late: lesioned value minus the intact network's at the rewind point (same answers f), raw units,
        late-window mean [5-95%]: committed to a corner (commit_fwd) / held per variable at corner ends (held_two_fwd) /
        corner split held jointly (held_joint_fwd) / corner answer = truth (corner_truth_fwd)
      joint5           -.27 [-.33,-.20] / -.00 [-.03,+.02] / -.01 [-.03,-.00] / -.08 [-.12,-.04]
      joint5_xfit      -.26 [-.32,-.19] / -.01 [-.04,+.01] / -.02 [-.03,-.01] / -.09 [-.13,-.06]
      joint5_vm_s0     -.05 [-.09,-.01] / +.12 [+.09,+.15] / +.11 [+.08,+.14] / +.03 [+.01,+.04]
      joint5_vm_s1     -.05 [-.08,-.00] / +.11 [+.08,+.14] / +.09 [+.06,+.12] / +.03 [+.01,+.05]
      r0_5             +.00 [-.04,+.05] / +.20 [+.17,+.24] / +.11 [+.08,+.14] / +.26 [+.24,+.29]
      r1_5             +.04 [+.00,+.07] / +.18 [+.14,+.21] / +.09 [+.06,+.11] / +.23 [+.21,+.25]
      add5             +.11 [+.08,+.16] / +.18 [+.15,+.22] / +.09 [+.06,+.11] / +.35 [+.32,+.38]
      pure5            -.30 [-.34,-.24] / +.03 [+.00,+.06] / +.02 [-.00,+.03] / -.25 [-.28,-.23]
      pc5              -.09 [-.14,-.02] / +.06 [+.04,+.08] / +.01 [+.00,+.02] / +.17 [+.13,+.20]
      pcspan5_s0       -.02 [-.04,+.01] / -.03 [-.06,-.00] / -.03 [-.06,-.01] / +.00 [-.01,+.01]
      pcspan5_s1       +.03 [+.00,+.05] / -.02 [-.05,+.00] / -.04 [-.07,-.02] / +.01 [-.01,+.02]
      keep100          -.08 [-.12,-.04] / +.21 [+.18,+.24] / +.16 [+.14,+.18] / +.10 [+.07,+.12]
      keep100_vm       -.02 [-.06,+.03] / +.24 [+.21,+.27] / +.17 [+.14,+.20] / +.04 [+.01,+.06]
      keep100_vm_s1    -.02 [-.07,+.03] / +.22 [+.19,+.25] / +.15 [+.13,+.18] / +.04 [+.01,+.07]
      res_i            -.10 [-.15,-.06] / +.23 [+.20,+.25] / +.15 [+.13,+.17] / -.08 [-.12,-.05]
      res_fi           -.09 [-.14,-.04] / +.22 [+.19,+.25] / +.13 [+.11,+.15] / -.12 [-.15,-.09]
      fclamp_own10     -.05 [-.09,-.01] / +.28 [+.25,+.31] / +.19 [+.17,+.22] / +.25 [+.22,+.27]
      fshift_own10_s0  -.11 [-.14,-.06] / +.28 [+.24,+.30] / +.21 [+.18,+.23] / +.11 [+.08,+.15]
      joint5 per late checkpoint (lesioned commit / held jointly; intact at the rewind point commit / held jointly):
          2,400: 0.46 / 0.01; 0.61 / 0.02; 3,200: 0.45 / 0.01; 0.63 / 0.02; 4,000: 0.38 / 0.01; 0.63 / 0.02; 5,600: 0.39
          / 0.02; 0.64 / 0.03; 8,000: 0.39 / 0.02; 0.66 / 0.04; 12,000: 0.35 / 0.03; 0.67 / 0.05; 16,000: 0.35 / 0.03;
          0.68 / 0.06; 24,000: 0.37 / 0.04; 0.68 / 0.06; 39,996: 0.34 / 0.05; 0.68 / 0.06
      intact commit_fwd per checkpoint: 80:0.01 160:0.08 320:0.31 480:0.39 640:0.49 800:0.52 960:0.59 1,120:0.63
          1,280:0.63 1,600:0.68 1,920:0.68 2,400:0.67 3,200:0.67 4,000:0.61 5,600:0.53 8,000:0.45 12,000:0.38
          16,000:0.34 24,000:0.22 39,996:0.17
    SEED SPREAD: specific damage against each seed of a random control alone, window means, answers / holding /
        coupling: [min, max] over the seeds
      joint5       vs matched           (2 seeds) early [+.14,+.15] / [+.02,+.03] / [+.06,+.07]; late [+.16,+.19] /
          [+.24,+.26] / [+.23,+.25]
      joint5       vs random in top PCs (2 seeds) early [+.14,+.15] / [+.02,+.02] / [+.04,+.05]; late [+.22,+.22] /
          [+.18,+.20] / [+.12,+.13]
      pure5        vs matched           (2 seeds) early [+.12,+.12] / [+.03,+.04] / [+.01,+.02]; late [+.19,+.20] /
          [+.29,+.31] / [+.18,+.18]
      add5         vs matched           (2 seeds) early [+.03,+.04] / [-.00,-.00] / [+.05,+.06]; late [-.02,+.01] /
          [-.03,+.01] / [+.17,+.19]
      pc5          vs matched           (2 seeds) early [+.14,+.15] / [+.01,+.01] / [+.06,+.07]; late [+.16,+.19] /
          [+.11,+.13] / [+.21,+.22]
      res_i        vs random non-g rows (3 seeds) early [+.01,+.04] / [+.00,+.01] / [-.01,+.01]; late [+.04,+.08] /
          [+.01,+.03] / [+.01,+.02]
      res_f        vs random non-g rows (3 seeds) early [-.07,-.04] / [-.00,+.00] / [-.05,-.03]; late [-.20,-.16] /
          [-.19,-.16] / [-.14,-.13]
      res_g        vs random non-g rows (3 seeds) early [+.56,+.59] / [+.03,+.04] / [+.09,+.11]; late [+.75,+.79] /
          [+.21,+.23] / [+.33,+.34]
      res_o        vs random non-g rows (3 seeds) early [+.14,+.17] / [+.01,+.02] / [+.04,+.06]; late [+.24,+.28] /
          [+.10,+.13] / [+.16,+.17]
      res_fi       vs random non-g rows (3 seeds) early [-.09,-.08] / [-.00,-.00] / [-.05,-.04]; late [-.15,-.13] /
          [-.08,-.06] / [-.10,-.08]
      mean_i       vs random non-g rows (3 seeds) early [+.04,+.06] / [+.01,+.02] / [-.01,+.01]; late [+.03,+.08] /
          [+.03,+.06] / [-.00,+.01]
      res_i        vs random            (3 seeds) early [-.27,-.18] / [-.01,-.01] / [-.04,-.02]; late [-.27,-.21] /
          [-.08,-.07] / [-.13,-.12]
      res_fi       vs random            (3 seeds) early [-.31,-.30] / [-.02,-.01] / [-.08,-.07]; late [-.44,-.40] /
          [-.13,-.13] / [-.23,-.22]
      fclamp_own10 vs random, same |df| (3 seeds) early [+.05,+.06] / [-.02,-.00] / [+.01,+.01]; late [-.02,-.00] /
          [-.08,-.02] / [+.03,+.06]
      fclamp_own5  vs random, same |df| (3 seeds) early [-.03,-.00] / [-.01,-.00] / [+.01,+.01]; late [-.10,-.05] /
          [-.05,+.00] / [+.02,+.06]
      fclamp_fin10 vs random, same |df| (3 seeds) early [+.02,+.02] / [-.00,+.00] / [+.01,+.02]; late [-.01,+.01] /
          [-.05,-.00] / [+.04,+.07]
      fclamp_own10 vs random            (3 seeds) early [+.08,+.09] / [-.01,-.00] / [+.01,+.02]; late [+.21,+.22] /
          [+.14,+.15] / [+.14,+.14]
      cclamp_own10 vs random            (3 seeds) early [+.41,+.43] / [-.02,-.01] / [+.01,+.02]; late [+.52,+.60] /
          [-.03,+.04] / [+.11,+.15]
      keep100      vs matched           (2 seeds) early [-.00,-.00] / [+.00,+.00] / [-.01,-.01]; late [-.07,-.05] /
          [-.05,-.02] / [-.04,-.03]
      keep20       vs matched           (2 seeds) early [+.13,+.13] / [-.00,-.00] / [+.01,+.02]; late [+.11,+.14] /
          [-.02,-.01] / [-.04,-.01]
    PATH RESIDUAL BY THE LESIONED NETWORK'S ANSWERS f (checkpoints >= 640 pooled; bins [0.45,0.60), [0.60,0.70),
        [0.70,0.80), [0.80,0.90)): mean [5-95%] (n checkpoints)
      joint5           +.02 [-.00,+.04] (3)  +.09 [+.05,+.12] (5)  -  -; late-window answers f 0.58
      joint5_xfit      +.01 [-.01,+.02] (4)  +.07 [+.03,+.10] (3)  +.10 [+.07,+.14] (2)  -; late-window answers f 0.62
      joint5_vm_s0     +.01 [-.01,+.02] (2)  +.03 [+.01,+.05] (2)  +.12 [+.09,+.15] (3)  +.29 [+.26,+.33] (4);
          late-window answers f 0.77
      joint5_vm_s1     +.01 [+.00,+.02] (3)  +.07 [+.04,+.09] (2)  +.19 [+.16,+.23] (3)  +.30 [+.26,+.34] (3);
          late-window answers f 0.74
      r0_5             -.01 [-.02,+.01] (3)  +.10 [+.08,+.14] (3)  +.28 [+.24,+.32] (5)  -; late-window answers f 0.70
      r1_5             -.01 [-.02,+.00] (3)  +.08 [+.05,+.10] (2)  +.20 [+.16,+.23] (4)  +.26 [+.21,+.30] (2);
          late-window answers f 0.73
      add5             +.11 [+.08,+.14] (3)  +.21 [+.17,+.25] (3)  +.24 [+.20,+.28] (2)  -; late-window answers f 0.59
      pure5            +.05 [+.03,+.07] (3)  +.15 [+.11,+.17] (4)  +.25 [+.21,+.29] (2)  -; late-window answers f 0.64
      pcspan5_s0       -.02 [-.04,-.01] (1)  -.04 [-.06,-.02] (1)  -.01 [-.03,+.01] (3)  +.05 [+.01,+.08] (4);
          late-window answers f 0.80
      pcspan5_s1       -.03 [-.04,-.02] (1)  -.03 [-.05,-.00] (1)  +.02 [-.00,+.04] (3)  +.06 [+.02,+.09] (3);
          late-window answers f 0.80
      keep100          +.06 [+.05,+.07] (3)  +.26 [+.24,+.29] (6)  +.45 [+.41,+.49] (2)  -; late-window answers f 0.64
      keep100_vm       +.14 [+.12,+.16] (7)  +.43 [+.39,+.47] (4)  -  -; late-window answers f 0.60
      keep100_vm_s1    +.20 [+.18,+.22] (6)  +.44 [+.40,+.47] (3)  -  -; late-window answers f 0.57
      res_i            +.25 [+.23,+.27] (9)  -  -  -; late-window answers f 0.53
      fclamp_own10     +.18 [+.16,+.19] (3)  +.26 [+.24,+.29] (5)  +.18 [+.15,+.21] (4)  -; late-window answers f 0.69
      fshift_own10_s0  +.00 [-.01,+.01] (1)  +.27 [+.25,+.30] (10)  +.09 [+.06,+.13] (1)  -; late-window answers f 0.66
  FINAL ROUND 11-02 (after the last check)
    VARIANCE of the interaction subspace (5 dims) by the checkpoints a claim uses: share of u's clean variance; share of
        the r0 x r1 factor; the factor's share of u; split-half cos^2
      all 20: 0.088-0.441 / 0.426-0.878 / 0.022-0.043 / 0.615-0.804; 80-640: 0.223-0.441 / 0.426-0.799 / 0.026-0.043 /
          0.626-0.804; 14 window checkpoints: 0.088-0.170 / 0.811-0.878 / 0.022-0.039 / 0.615-0.790; early: 0.118-0.170
          / 0.811-0.842 / 0.022-0.024 / 0.615-0.681; late: 0.088-0.111 / 0.859-0.878 / 0.027-0.039 / 0.713-0.790
      per checkpoint, share of u: 80:0.441 160:0.341 320:0.351 480:0.275 640:0.223 800:0.170 960:0.154 1,120:0.136
          1,280:0.132 1,600:0.118 1,920:0.115 2,400:0.111 3,200:0.108 4,000:0.102 5,600:0.101 8,000:0.097 12,000:0.092
          16,000:0.090 24,000:0.088 39,996:0.091
      third-pass subspaces over their 14 checkpoints (800-39,996), share of u: joint5 0.088-0.170; add5 0.165-0.191;
          pure5 0.043-0.060; pc5 0.213-0.270; pcspan5_s0 0.088-0.171; pcspan5_s1 0.088-0.170
      add5's variance / joint5's: 1.08-2.04 (800:1.08 960:1.15 1,120:1.22 1,280:1.28 1,600:1.48 2,400:1.71 3,200:1.78
          4,000:1.85 5,600:1.88 8,000:1.96 12,000:2.01 16,000:2.04 24,000:2.04 39,996:1.95)
      top-K span of pcspan5 (mean K over steps, both seeds): 14-36
      keep 20 PCs removes of u's variance: late 0.48-0.52, early 0.33-0.46
      keep 100 PCs removes of u's variance: late 0.17-0.20, early 0.09-0.15
    PANELS (core answers = acc_goal_T, acc_both_T, far_T, merged_T; corner answers = corner_truth_fwd; holding;
        coupling): specific damage [5-95%], early | late, and per checkpoint (+ / - / 0: 5-95% above / below 0 /
        neither; '.': no control there)
      joint5 vs random dims       core answers   +.17 [+.14,+.19] | +.12 [+.10,+.13] ++++++++++++++++++++ (> 0 from 80,
          < 0 from never)
      joint5 vs random dims       corner answers +.06 [-.00,+.12] | +.40 [+.32,+.48] 0-00000+0+++++++++++ (> 0 from
          1,600, < 0 from never)
      joint5 vs random dims       holding        +.03 [+.01,+.04] | +.25 [+.22,+.28] 00+0++++++++++++++++ (> 0 from 640,
          < 0 from never)
      joint5 vs random dims       coupling       +.07 [+.05,+.08] | +.24 [+.22,+.26] -000+0++++++++++++++ (> 0 from 960,
          < 0 from never)
      joint5 vs r0/r1             core answers   +.13 [+.11,+.15] | -.01 [-.03,+.01] 00+++++++++++00----- (> 0 from
          never, < 0 from 8,000)
      joint5 vs r0/r1             corner answers +.35 [+.28,+.41] | +.69 [+.62,+.77] 00++++++++++++++++++ (> 0 from 320,
          < 0 from never)
      joint5 vs r0/r1             holding        +.03 [+.02,+.04] | +.29 [+.27,+.32] -00+++++++++++++++++ (> 0 from 480,
          < 0 from never)
      joint5 vs r0/r1             coupling       +.04 [+.02,+.05] | +.12 [+.10,+.13] +000+00+++++++++++++ (> 0 from
          1,120, < 0 from never)
      joint5 vs r0+r1 5 dims      core answers   -.03 [-.05,-.00] | -.17 [-.21,-.14] .....+000-.--------- (> 0 from
          never, < 0 from 2,400)
      joint5 vs r0+r1 5 dims      corner answers +.48 [+.41,+.55] | +.74 [+.66,+.82] .....+++++.+++++++++ (> 0 from
          2,400, < 0 from never)
      joint5 vs r0+r1 5 dims      holding        +.03 [+.02,+.04] | +.22 [+.18,+.26] .....+++++.+++++++++ (> 0 from
          2,400, < 0 from never)
      joint5 vs r0+r1 5 dims      coupling       +.01 [-.01,+.04] | +.03 [+.01,+.06] .....0000+.0000+++++ (> 0 from
          8,000, < 0 from never)
      joint5 vs random in top PCs core answers   +.16 [+.13,+.18] | +.16 [+.15,+.18] .....+++++.+++++++++ (> 0 from
          2,400, < 0 from never)
      joint5 vs random in top PCs corner answers +.08 [+.03,+.14] | +.45 [+.37,+.53] .....0++0+.+++++++++ (> 0 from
          2,400, < 0 from never)
      joint5 vs random in top PCs holding        +.02 [+.01,+.03] | +.19 [+.16,+.22] .....+++++.+++++++++ (> 0 from
          2,400, < 0 from never)
      joint5 vs random in top PCs coupling       +.05 [+.03,+.06] | +.13 [+.11,+.15] .....00+++.+++++++++ (> 0 from
          2,400, < 0 from never)
      pure5 vs its random dims    core answers   +.08 [+.07,+.10] | +.06 [+.04,+.07] .....+++++.+++++++++ (> 0 from
          2,400, < 0 from never)
      pure5 vs its random dims    corner answers +.27 [+.22,+.33] | +.74 [+.68,+.80] .....+++++.+++++++++ (> 0 from
          2,400, < 0 from never)
      pure5 vs its random dims    holding        +.04 [+.03,+.05] | +.30 [+.27,+.33] .....+++++.+++++++++ (> 0 from
          2,400, < 0 from never)
      pure5 vs its random dims    coupling       +.01 [-.00,+.03] | +.18 [+.16,+.20] .....0000+.+++++++++ (> 0 from
          2,400, < 0 from never)
      joint5_xfit vs r0/r1        core answers   +.10 [+.08,+.11] | -.04 [-.05,-.02] 0+++++++++++000----- (> 0 from
          never, < 0 from 8,000)
      joint5_xfit vs r0/r1        corner answers +.34 [+.27,+.40] | +.68 [+.61,+.76] 0-0+++++++++++++++++ (> 0 from 480,
          < 0 from never)
      joint5_xfit vs r0/r1        holding        +.03 [+.02,+.04] | +.29 [+.26,+.32] 00+0++++++++++++++++ (> 0 from 640,
          < 0 from never)
      joint5_xfit vs r0/r1        coupling       +.03 [+.01,+.04] | +.12 [+.10,+.13] +000+00+++++++++++++ (> 0 from
          1,120, < 0 from never)
      joint5 core answers per checkpoint vs random dims: 80 +.15 [+.08,+.24]; 160 +.06 [+.01,+.14]; 320 +.13
          [+.05,+.23]; 480 +.12 [+.08,+.18]; 640 +.13 [+.08,+.18]; 800 +.11 [+.07,+.15]; 960 +.13 [+.09,+.16]; 1,120
          +.17 [+.14,+.20]; 1,280 +.19 [+.16,+.22]; 1,600 +.24 [+.21,+.27]; 1,920 +.21 [+.18,+.24]; 2,400 +.20
          [+.17,+.22]; 3,200 +.17 [+.14,+.19]; 4,000 +.14 [+.12,+.16]; 5,600 +.13 [+.11,+.15]; 8,000 +.11 [+.09,+.12];
          12,000 +.10 [+.08,+.11]; 16,000 +.07 [+.06,+.09]; 24,000 +.08 [+.07,+.10]; 39,996 +.07 [+.05,+.08]
      joint5 corner answers per checkpoint vs random dims: 80 +.00 [+.00,+.00]; 160 -.02 [-.03,-.01]; 320 +.02
          [-.00,+.05]; 480 -.01 [-.03,+.02]; 640 -.02 [-.05,+.01]; 800 +.03 [-.03,+.09]; 960 +.01 [-.06,+.08]; 1,120
          +.13 [+.05,+.21]; 1,280 -.01 [-.08,+.06]; 1,600 +.13 [+.04,+.23]; 1,920 +.12 [+.02,+.20]; 2,400 +.14
          [+.05,+.23]; 3,200 +.29 [+.20,+.37]; 4,000 +.32 [+.23,+.42]; 5,600 +.39 [+.30,+.48]; 8,000 +.46 [+.37,+.54];
          12,000 +.48 [+.39,+.56]; 16,000 +.48 [+.42,+.55]; 24,000 +.52 [+.44,+.59]; 39,996 +.52 [+.44,+.60]
      joint5 core answers per checkpoint vs r0/r1: 80 -.00 [-.05,+.05]; 160 +.04 [-.01,+.10]; 320 +.06 [+.01,+.11]; 480
          +.05 [+.01,+.10]; 640 +.13 [+.09,+.17]; 800 +.11 [+.07,+.14]; 960 +.10 [+.08,+.13]; 1,120 +.14 [+.11,+.16];
          1,280 +.16 [+.14,+.19]; 1,600 +.14 [+.11,+.16]; 1,920 +.14 [+.11,+.16]; 2,400 +.09 [+.07,+.11]; 3,200 +.04
          [+.01,+.06]; 4,000 +.00 [-.02,+.03]; 5,600 +.01 [-.01,+.03]; 8,000 -.03 [-.05,-.01]; 12,000 -.04 [-.06,-.02];
          16,000 -.06 [-.08,-.04]; 24,000 -.05 [-.07,-.03]; 39,996 -.04 [-.06,-.02]
      joint5 corner answers per checkpoint vs r0/r1: 80 +.00 [+.00,+.00]; 160 -.01 [-.02,+.00]; 320 +.02 [+.01,+.04];
          480 +.05 [+.03,+.08]; 640 +.09 [+.04,+.13]; 800 +.20 [+.14,+.26]; 960 +.26 [+.19,+.33]; 1,120 +.40
          [+.32,+.48]; 1,280 +.38 [+.31,+.46]; 1,600 +.50 [+.40,+.59]; 1,920 +.50 [+.42,+.58]; 2,400 +.48 [+.39,+.57];
          3,200 +.56 [+.47,+.64]; 4,000 +.63 [+.55,+.72]; 5,600 +.70 [+.61,+.78]; 8,000 +.74 [+.65,+.83]; 12,000 +.77
          [+.68,+.85]; 16,000 +.77 [+.70,+.83]; 24,000 +.78 [+.71,+.84]; 39,996 +.83 [+.76,+.90]
      joint5 core answers per checkpoint vs r0+r1 5 dims: 80 .; 160 .; 320 .; 480 .; 640 .; 800 +.04 [+.00,+.08]; 960
          -.03 [-.06,+.01]; 1,120 -.01 [-.05,+.03]; 1,280 -.02 [-.05,+.01]; 1,600 -.12 [-.16,-.08]; 1,920 .; 2,400 -.16
          [-.21,-.12]; 3,200 -.20 [-.26,-.14]; 4,000 -.19 [-.24,-.15]; 5,600 -.18 [-.23,-.14]; 8,000 -.17 [-.20,-.13];
          12,000 -.17 [-.20,-.13]; 16,000 -.17 [-.21,-.13]; 24,000 -.16 [-.19,-.13]; 39,996 -.16 [-.19,-.12]
      joint5 corner answers per checkpoint vs r0+r1 5 dims: 80 .; 160 .; 320 .; 480 .; 640 .; 800 +.34 [+.27,+.42]; 960
          +.42 [+.35,+.49]; 1,120 +.56 [+.47,+.64]; 1,280 +.51 [+.43,+.59]; 1,600 +.59 [+.50,+.68]; 1,920 .; 2,400 +.56
          [+.48,+.65]; 3,200 +.60 [+.51,+.69]; 4,000 +.65 [+.56,+.74]; 5,600 +.74 [+.65,+.84]; 8,000 +.78 [+.68,+.87];
          12,000 +.83 [+.74,+.91]; 16,000 +.83 [+.75,+.91]; 24,000 +.81 [+.73,+.89]; 39,996 +.87 [+.79,+.95]
      late gap fraction per panel (intact / joint5 / r0+r1 5 dims / r0, r1 averaged / random dims): core answers 0.93 /
          0.67 / 0.50 / 0.66 / 0.79; corner answers 0.87 / 0.22 / 0.96 / 0.92 / 0.62; holding 0.62 / 0.11 / 0.33 / 0.40
          / 0.36; coupling 0.58 / 0.21 / 0.24 / 0.32 / 0.45
    GATES per panel vs count-matched random rows without g: specific damage [5-95%], early | late, per checkpoint signs
      res_g    core answers   +.66 [+.61,+.71] | +.89 [+.83,+.95] ++++++++++++++++++++ (> 0 from 80, < 0 from never)
      res_g    corner answers +.24 [+.20,+.28] | +.26 [+.23,+.30] 0+++++++++++++++++++ (> 0 from 160, < 0 from never)
      res_g    holding        +.04 [+.02,+.05] | +.22 [+.20,+.24] -0+0++++++++++++++++ (> 0 from 640, < 0 from never)
      res_g    coupling       +.10 [+.08,+.12] | +.34 [+.32,+.36] 000+++++++++++++++++ (> 0 from 480, < 0 from never)
      res_o    core answers   +.20 [+.18,+.23] | +.33 [+.31,+.35] ++0+++++++++++++++++ (> 0 from 480, < 0 from never)
      res_o    corner answers -.00 [-.03,+.02] | -.05 [-.08,-.02] 000-0-000+00+000---- (> 0 from never, < 0 from 12,000)
      res_o    holding        +.01 [+.01,+.02] | +.12 [+.10,+.13] 00000+++++++++++++++ (> 0 from 800, < 0 from never)
      res_o    coupling       +.05 [+.04,+.06] | +.17 [+.15,+.18] -00+++++++++++++++++ (> 0 from 480, < 0 from never)
      res_i    core answers   +.04 [+.03,+.05] | +.01 [-.00,+.01] -00+++++++++0000000+ (> 0 from 39,996, < 0 from never)
      res_i    corner answers -.03 [-.05,-.01] | +.25 [+.22,+.27] 00000---00++++++++++ (> 0 from 1,920, < 0 from never)
      res_i    holding        +.01 [+.00,+.01] | +.02 [+.01,+.03] 00000+0+++0+++++++++ (> 0 from 2,400, < 0 from never)
      res_i    coupling       +.00 [-.00,+.01] | +.02 [+.01,+.03] +00000000+000++++++0 (> 0 from never, < 0 from never)
      res_f    core answers   -.09 [-.10,-.07] | -.19 [-.19,-.18] --00---------------- (> 0 from never, < 0 from 640)
      res_f    corner answers +.08 [+.05,+.10] | -.19 [-.23,-.16] 00+++++++++0-------- (> 0 from never, < 0 from 3,200)
      res_f    holding        +.00 [-.00,+.00] | -.17 [-.19,-.16] 00+00+0000---------- (> 0 from never, < 0 from 1,920)
      res_f    coupling       -.03 [-.04,-.03] | -.13 [-.14,-.12] +00----------------- (> 0 from never, < 0 from 480)
      res_fi   core answers   -.13 [-.14,-.11] | -.21 [-.23,-.20] --0----------------- (> 0 from never, < 0 from 480)
      res_fi   corner answers +.08 [+.05,+.11] | +.19 [+.16,+.22] 00++++++++++++++++++ (> 0 from 320, < 0 from never)
      res_fi   holding        -.00 [-.01,+.00] | -.07 [-.07,-.06] 0-00-0000----------- (> 0 from never, < 0 from 1,600)
      res_fi   coupling       -.04 [-.06,-.03] | -.09 [-.10,-.08] +0------------------ (> 0 from never, < 0 from 320)
      mean_g   core answers   +.92 [+.85,+1.00] | +1.03 [+.96,+1.10] ++++++++++++++++++++ (> 0 from 80, < 0 from never)
      mean_g   corner answers +.28 [+.20,+.35] | +.30 [+.24,+.36] 0+++++++++++++++++++ (> 0 from 160, < 0 from never)
      mean_g   holding        +.04 [+.02,+.05] | +.25 [+.22,+.28] -0+000++++++++++++++ (> 0 from 960, < 0 from never)
      mean_g   coupling       +.09 [+.06,+.12] | +.33 [+.31,+.35] 00++++++++++++++++++ (> 0 from 320, < 0 from never)
      mean_o   core answers   +.13 [+.11,+.15] | +.23 [+.21,+.25] ++++++++++++++++++++ (> 0 from 80, < 0 from never)
      mean_o   corner answers -.09 [-.14,-.05] | -.22 [-.28,-.16] 00------------------ (> 0 from never, < 0 from 320)
      mean_o   holding        -.01 [-.02,+.00] | +.08 [+.07,+.11] 000000000000++++++++ (> 0 from 3,200, < 0 from never)
      mean_o   coupling       +.04 [+.03,+.05] | +.16 [+.14,+.18] -000++++++++++++++++ (> 0 from 640, < 0 from never)
      mean_i   core answers   +.07 [+.06,+.09] | +.02 [+.01,+.03] ---0++++++++0++00+++ (> 0 from 16,000, < 0 from never)
      mean_i   corner answers -.06 [-.09,-.03] | +.19 [+.15,+.23] 000------00+++++++++ (> 0 from 2,400, < 0 from never)
      mean_i   holding        +.01 [+.00,+.03] | +.04 [+.02,+.07] 000+0++0+0++++++++++ (> 0 from 1,920, < 0 from never)
      mean_i   coupling       +.00 [-.01,+.01] | +.01 [-.00,+.01] 0-00000000000+000000 (> 0 from never, < 0 from never)
      mean_f   core answers   -.05 [-.07,-.04] | -.15 [-.16,-.14] -000---------------- (> 0 from never, < 0 from 640)
      mean_f   corner answers +.13 [+.11,+.16] | -.12 [-.17,-.08] 00++++++++++0------- (> 0 from never, < 0 from 4,000)
      mean_f   holding        +.00 [-.00,+.01] | -.19 [-.21,-.17] 00+00++000---------- (> 0 from never, < 0 from 1,920)
      mean_f   coupling       -.02 [-.03,-.02] | -.12 [-.14,-.11] +000---------------- (> 0 from never, < 0 from 640)
      mean_fi  core answers   -.04 [-.06,-.02] | -.10 [-.12,-.08] -----0-------------- (> 0 from never, < 0 from 960)
      mean_fi  corner answers +.08 [+.05,+.11] | +.13 [+.10,+.17] 00++++++++++++++++++ (> 0 from 320, < 0 from never)
      mean_fi  holding        +.01 [-.00,+.01] | -.03 [-.05,-.02] 0000000000000-----00 (> 0 from never, < 0 from never)
      mean_fi  coupling       -.04 [-.05,-.03] | -.12 [-.14,-.11] +00----------------- (> 0 from never, < 0 from 480)
      late gap fraction per panel (intact; g / o / i / f resampled; random non-g rows, 3 seeds): core answers 0.93;
          -0.25 / 0.31 / 0.63 / 0.82; 0.64; corner answers 0.87; 0.13 / 0.44 / 0.14 / 0.58; 0.39; holding 0.62; 0.11 /
          0.22 / 0.31 / 0.51; 0.33; coupling 0.58; 0.04 / 0.21 / 0.36 / 0.51; 0.38
    PATH TEST AT MATCHED TRAINING STAGE: a condition's late path residual minus joint5's at the same late checkpoint,
        over the late checkpoints where their answers f differ by <= tol (n of 9) [5-95%]
      tol 0.05, n >= 3: 25 conditions: joint5_xfit (9) -.01 [-.01,+.00]; add5 (9) +.11 [+.09,+.14]; pure5 (4) +.12
          [+.11,+.13]; pc5_vm_s0 (6) +.13 [+.11,+.15]; mean_i (3) +.14 [+.11,+.16]; res_i (4) +.16 [+.14,+.18];
          mean_nog_s2 (7) +.19 [+.17,+.21]; res_nog_s2 (4) +.19 [+.17,+.21]; res_nog_s0 (4) +.19 [+.17,+.22];
          mean_nog_s0 (7) +.20 [+.18,+.23]; add5_vm_s1 (8) +.22 [+.20,+.24]; cclamp_rand10_s2 (3) +.23 [+.21,+.26];
          cclamp_rand10_s1 (7) +.23 [+.21,+.25]; keep100_vm_s1 (7) +.23 [+.22,+.25]; mean_nog_s1 (5) +.24 [+.22,+.26];
          res_nog_s1 (5) +.24 [+.22,+.26]; add5_vm_s0 (7) +.25 [+.23,+.28]; cclamp_fin5_vm (6) +.26 [+.25,+.28];
          cclamp_rand10_s0 (5) +.27 [+.25,+.29]; keep100_vm (7) +.27 [+.25,+.29]; fshift_own10_s2 (4) +.28 [+.26,+.30];
          cclamp_own5_vm (4) +.30 [+.28,+.33]; keep100 (5) +.30 [+.29,+.33]; fshift_own10_s0 (4) +.36 [+.33,+.37];
          fshift_fin10_s0 (4) +.36 [+.34,+.38]
      tol 0.10, n >= 1: 51 conditions; those within +0.10: joint3 (9) -.02 [-.03,-.01]; joint3_xfit (9) -.02
          [-.03,-.01]; joint5_xfit (9) -.01 [-.01,+.00]; mean_o (3) +.07 [+.05,+.09]; mean_f_vm (3) +.08 [+.05,+.10];
          res_f_vm (1) +.08 [+.06,+.10]; pure5 (8) +.09 [+.08,+.10]; keep50_vm (1) +.09 [+.07,+.11]
      joint5's path residual per checkpoint from 640: 640:-.05 800:-.03 960:-.05 1,120:-.05 1,280:-.05 1,600:-.05
          1,920:-.03 2,400:-.01 3,200:-.01 4,000:+.01 5,600:+.05 8,000:+.06 12,000:+.08 16,000:+.09 24,000:+.11
          39,996:+.12
      the fix round's pooled bin (answers f in [.45,.60), checkpoints >= 640) held: joint5 3,200, 4,000, 5,600;
          joint5_vm_s0 1,600, 1,920; joint5_vm_s1 1,600, 1,920, 2,400; r0_5 1,600, 1,920, 2,400; r1_5 1,600, 1,920,
          2,400; add5 3,200, 4,000, 5,600; pure5 2,400, 3,200, 4,000
      robustness, matched and drawn on the core answers' f instead (tol .05, n >= 3): joint5's late residual +.03;
          joint3 (6) -.01 [-.02,-.00]; joint5_xfit (7) -.01 [-.02,-.00]; mean_i (3) +.13 [+.11,+.16]; r0_5 (3) +.15
          [+.12,+.18]; mean_nog_s0 (4) +.17 [+.14,+.20]; res_nog_s2 (3) +.17 [+.15,+.19]; ... (29 conditions, the
          largest +.43)
      robustness, corner commitment at the core-answers rewind point, late: joint5 -.27 [-.34,-.20]
      robustness, corner commitment at the core-answers rewind point, late: pure5 -.30 [-.35,-.25]
      robustness, corner commitment at the core-answers rewind point, late: add5 +.16 [+.11,+.21]
  ESN (800, 1,600, 4,000, 12,000, 39,996): intact f answers -1.22 / -.65 / -.27 / -.10 / -.02; holding -.06 / -.04 /
      -.02 / -.03 / -.03; coupling +.01 / -.02 / -.02 / +.00 / +.00
    joint5       answers dF -.00 / -.01 / -.07 / -.09 / -.09; vs matched -.03 [-.07,+.01] / -.00 [-.03,+.02] / +.04
        [+.01,+.07] / +.01 [-.02,+.04] / -.03 [-.06,+.00]
    joint5       answers dF -.00 / -.01 / -.07 / -.09 / -.09; vs r0/r1 -.01 [-.03,-.00] / -.01 [-.03,-.00] / +.01
        [+.00,+.02] / -.01 [-.02,+.00] / -.01 [-.02,+.00]
    joint5_xfit  answers dF +.00 / -.01 / -.06 / -.08 / -.07; vs r0/r1 -.01 [-.02,+.01] / -.01 [-.02,-.00] / -.00
        [-.02,+.01] / -.02 [-.03,-.01] / -.03 [-.04,-.02]
    res_g        answers dF -.14 / -.02 / -.05 / -.07 / -.09; vs random +.10 [+.07,+.14] / +.01 [-.01,+.04] / +.04
        [+.03,+.06] / +.03 [+.02,+.05] / +.03 [+.02,+.04]
    res_g        answers dF -.14 / -.02 / -.05 / -.07 / -.09; vs random non-g rows +.13 [+.09,+.19] / +.03 [-.01,+.06] /
        +.05 [+.02,+.07] / +.08 [+.05,+.10] / +.09 [+.07,+.11]
    res_i        answers dF -.00 / +.00 / -.00 / -.00 / +.00; vs random non-g rows +.00 [-.01,+.01] / -.00 [-.00,+.00] /
        +.00 [-.00,+.01] / +.00 [+.00,+.01] / -.00 [-.01,+.01]
    res_f        answers dF -.00 / +.00 / -.00 / -.00 / +.00; vs random non-g rows -.00 [-.01,+.00] / +.00 [-.00,+.00] /
        -.00 [-.00,+.00] / +.00 [-.00,+.01] / +.00 [-.00,+.01]
    res_o        answers dF +.00 / -.00 / +.00 / +.00 / -.01; vs random non-g rows -.00 [-.01,+.00] / +.00 [-.00,+.01] /
        -.01 [-.01,-.00] / -.00 [-.01,+.00] / +.01 [+.00,+.01]
    res_fi       answers dF +.00 / +.00 / -.00 / -.00 / +.00; vs random -.08 [-.12,-.05] / -.01 [-.03,+.01] / -.02
        [-.04,-.00] / -.05 [-.07,-.03] / -.07 [-.08,-.05]
    res_fi       answers dF +.00 / +.00 / -.00 / -.00 / +.00; vs random non-g rows -.00 [-.00,+.00] / -.00 [-.01,-.00] /
        +.00 [+.00,+.01] / +.01 [+.00,+.01] / -.00 [-.01,+.00]
    fclamp_own10 answers dF +.01 / -.01 / -.01 / -.08 / -.07; vs random -.04 [-.07,-.01] / +.00 [-.01,+.02] / +.01
        [-.00,+.03] / +.05 [+.03,+.07] / +.05 [+.03,+.07]
    fclamp_own10 answers dF +.01 / -.01 / -.01 / -.08 / -.07; vs random, same |df| -.03 [-.05,-.01] / -.00 [-.02,+.01] /
        -.00 [-.02,+.01] / +.04 [+.02,+.06] / +.03 [+.01,+.05]
    fself_own10  answers dF +.00 / -.00 / -.00 / -.01 / -.00; vs random -.01 [-.02,+.00] / +.00 [-.00,+.01] / -.00
        [-.01,+.01] / +.01 [-.00,+.01] / +.00 [-.01,+.01]
    cclamp_own10 answers dF -.21 / -.23 / -.32 / -.54 / -.70; vs random -.02 [-.11,+.07] / +.14 [+.06,+.26] / +.15
        [+.09,+.22] / +.17 [+.10,+.25] / +.14 [+.06,+.24]
    keep100      answers dF -.01 / +.00 / -.00 / -.01 / +.00; vs matched +.01 [-.00,+.01] / -.00 [-.01,+.01] / -.00
        [-.01,+.00] / -.00 [-.01,+.00] / -.01 [-.02,+.00]
    ESN forget-gate perturbation |f - assigned| per unit and step (top 10% to median / random units shifted by the same
        mean / random units to median): 0.058 / 0.055 / 0.031; 0.063 / 0.061 / 0.034; 0.072 / 0.069 / 0.039; 0.082 /
        0.080 / 0.044; 0.088 / 0.086 / 0.048
    ESN u variance per step 1.6 / 2.1 / 2.9 / 4.1 / 5.1; r0 x r1 factor share 0.0072 / 0.0063 / 0.0059 / 0.0059 /
        0.0058; max tau 2.51 / 2.70 / 2.83 / 3.07 / 3.08"""
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

OUT = os.path.join(ti.TD, "through_training", "lesions")
TMP = ("/tmp/claude-1002/-home-johns-Desktop-CognitiveGridworld/7af6b091-54ff-45a0-b4dd-ef0a3bc9c778/scratchpad/"
       "tt_tmp/lesions")
T, R, NV, NO, H = ti.T, ti.R, ti.NV, ti.NO, 1000
GATES = ti.GATES
# raw checkpoint episodes (training episodes = E - E // 5): 80, 160, 320, 480, 640, 800, 960, 1,120, 1,280, 1,600, 1,920,
# 2,400, 3,200, 4,000, 5,600, 8,000, 12,000, 16,000, 24,000, 39,996
CKS = (100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000, 2400, 3000, 4000, 5000, 7000, 10000, 15000, 20000, 30000,
       49995)
CKS_ESN = (1000, 2000, 5000, 15000, 49995)                # training 800, 1,600, 4,000, 12,000, 39,996
FINAL = 49995
PCTS = (1, 2, 5, 10)                                      # long-tau sets: top 1 / 2 / 5 / 10% of the 1,000 units
KS_SUB = (3, 5)                                           # latent-interaction subspace dims
KS_DIM = (5, 10, 20, 50, 100, 200)                        # dimension lesion: kept PCs (plus the 20 read-out rows)
SEED_UNITS, SEED_ROWS, SEED_ROT = 20261211, 20261212, 20261213
N_RAND = 3                                                # seeds of every count-matched random unit / row control
N_SUB = 2                                                 # seeds of the variance-matched random subspaces
BUDGET = 8000                                             # GPU rows per chunk x (1 + donor run)
ROWS_CHEAP = ti.CHEAP
ROWS_FWD = ti.FWD
ROWS = ROWS_CHEAP + ROWS_FWD + ("kl_int",)
# families: the timeline's rise-shaped rows that the cheap + forward-set scores contain (td_tt_timeline; splitting,
# fold index and opposite corners need the 'full' score and are left out)
FAMILY_ROWS = {"answers": ("acc_goal_T", "acc_both_T", "far_T", "merged_T", "corner_truth_fwd"),
               "holding": ("held_joint_fwd", "held_two_fwd", "held_one_fwd", "dist_out_fwd", "emass_slope"),
               "coupling": ("excess", "G", "band4", "angle_out_fwd", "same_diag_fwd", "dir_slope")}
FAMILIES = tuple(FAMILY_ROWS)
# figure panels (final round): the answers family split into its four core rows and corner answers (late, the
# interaction lesion's excess over the main-effect controls on the 5-row family is the corner row alone)
PANEL_ROWS = {"core answers": FAMILY_ROWS["answers"][:4], "corner answers": ("corner_truth_fwd",),
              "holding": FAMILY_ROWS["holding"], "coupling": FAMILY_ROWS["coupling"]}
PANELS = tuple(PANEL_ROWS)
_C = {}


# ═══════════════════════════════════════════════════════════════════
# Lean forward: td_tt_intervene.forward's loop (no memory cuts) with row-level gate ablations and per-step clamps.
# Copied and generalised because the engine's hooks are hard-wired to whole gate blocks and one fixed clamp basis.
# ═══════════════════════════════════════════════════════════════════

def _fwd_donors():
    if "fwd_donors" not in _C:
        _C["fwd_donors"] = ti.donors(ti.forward_set(), "random", ti.GATE_SEED)
    return _C["fwd_donors"]


def _ctx(ref, spec, dev):
    """device constants of a lesion spec: mean_rows / res_rows (drive rows of the 4H pre-activation, gate order i f g
    o), f_units (forget gate := f_med[s], or with f_value 'self' each unit's own clean per-step probe mean f_mean[s],
    or with f_value an array (T + 1, k) the given value of each unit per step), c_units (c := clean per-step probe
    mean), V (list over steps s = 1..T of
    orthonormal (H, k_s) arrays, or one (H, k) array) with mode 'clamp' (u <- u - (u - mu_s) V V^T) or 'keep'
    (u <- mu_s + (u - mu_s) V V^T)"""
    spec = dict(spec or {})
    bad = set(spec) - {"mean_rows", "res_rows", "f_units", "f_value", "c_units", "V", "mode"}
    assert not bad, bad

    def dv(a):
        return torch.as_tensor(np.asarray(a, np.float32), device=dev)

    def ix(a):
        return torch.as_tensor(np.asarray(a, np.int64), device=dev)
    ctx = dict(mean_rows=None, res_rows=None, f_units=None, f_self=False, c_units=None, V=None, mode=None)
    if spec.get("mean_rows") is not None:
        ctx["mean_rows"], ctx["mean_rec"] = ix(spec["mean_rows"]), dv(ref["mean_rec"])
    if spec.get("res_rows") is not None:
        ctx["res_rows"] = ix(spec["res_rows"])
    if ctx["mean_rows"] is not None and ctx["res_rows"] is not None:
        assert not np.intersect1d(spec["mean_rows"], spec["res_rows"]).size, "a row is mean- or resample-ablated"
    if spec.get("f_units") is not None:
        fv = spec.get("f_value", "median")
        if isinstance(fv, np.ndarray):                       # given values (T + 1, k): f[:, units] := fv[s]
            assert fv.shape == (T + 1, len(spec["f_units"])), fv.shape
            ctx["f_units"], ctx["f_self"], ctx["f_val"] = ix(spec["f_units"]), False, dv(fv)
        else:
            assert fv in ("median", "self"), fv
            ctx["f_units"], ctx["f_self"] = ix(spec["f_units"]), fv == "self"
            ctx["f_val"] = dv(ref["f_mean"] if fv == "self" else ref["f_med"])
    if spec.get("c_units") is not None:
        ctx["c_units"], ctx["mu_c"] = ix(spec["c_units"]), dv(ref["mu_c"])
    if spec.get("V") is not None:
        V = spec["V"]
        Vs = [V] * T if (isinstance(V, np.ndarray) and V.ndim == 2) else list(V)
        assert len(Vs) == T
        ctx["V"] = [None] + [dv(v) if v.shape[1] else None for v in Vs]
        ctx["mode"] = spec.get("mode", "clamp")
        assert ctx["mode"] in ("clamp", "keep")
        ctx["mu_u"] = dv(ref["mu_u"])
    return ctx


def _lstep(xin, h, c, s, ctx, Whh, hdon=None):
    """td_tt_intervene._step with row sets: drive (mean / resample rows) -> gates -> f_units -> c -> c_units ->
    u = o tanh(c) -> clamp / keep u"""
    rec = h @ Whh.T
    if ctx["mean_rows"] is not None:
        mr = ctx["mean_rows"]
        rec[:, mr] = ctx["mean_rec"][s, mr]
    if ctx["res_rows"] is not None:
        rr = ctx["res_rows"]
        rec[:, rr] = hdon @ Whh[rr].T
    pre = xin + rec
    i, f, g, o = pre.chunk(4, -1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    if ctx["f_units"] is not None:
        fu = ctx["f_units"]
        f[:, fu] = ctx["f_val"][s, fu] if ctx["f_self"] else ctx["f_val"][s]
    c = f * c + i * g
    if ctx["c_units"] is not None:
        c[:, ctx["c_units"]] = ctx["mu_c"][s, ctx["c_units"]]
    h = o * c.tanh()
    if ctx["V"] is not None and ctx["V"][s] is not None:
        V, mu = ctx["V"][s], ctx["mu_u"][s]
        if ctx["mode"] == "clamp":
            h = h - ((h - mu) @ V) @ V.T
        else:
            h = mu + ((h - mu) @ V) @ V.T
    return h, c, f


_CLEAN = dict(mean_rows=None, res_rows=None, f_units=None, f_self=False, c_units=None, V=None, mode=None)


@torch.no_grad()
def lforward(w, spec=None, ref=None, fwd=False, dev=None, record=None, budget=BUDGET):
    """marginal log beliefs (n, T, 2, R) float32 and right (n, T, 2) of the weights w under a lesion spec (see _ctx) on
    the 10,000 probes (fwd=False) or the 5,028-episode forward set; the constants come from the clean probe run
    (ref = ti.reference(w)); resample rows use one fixed donor per episode (ti.GATE_SEED: same environment, another
    cell), whose CLEAN drive is used (as the engine). record(s, rows, dict(u, c, f)) is called at every step"""
    dev = ti._device(dev)
    E = ti.forward_set() if fwd else ti.probes()
    spec = dict(spec or {})
    obs, env_all = np.asarray(E["obs"]), np.asarray(E["env"], np.int64)
    n = len(obs)
    p = {k: v.to(dev, torch.float32) for k, v in w.items()}
    Whh, Wro, bro = p["LSTM.weight_hh_l0"], p["classifier_readout.weight"], p["classifier_readout.bias"]
    h0, c0 = p["STM"].reshape(1, H), p["LTM"].reshape(1, H)
    table = ti._table(p, E["Z"], dev)
    code_np = (obs * (1 << np.arange(NO))).sum(-1).astype(np.int64)
    if ref is None and any(spec.get(k) is not None for k in ("mean_rows", "f_units", "c_units", "V")):
        ref = ti.reference(w, dev)
    ctx = _ctx(ref, spec, dev)
    gdon = None
    if spec.get("res_rows") is not None:
        gdon = _fwd_donors() if fwd else ti._donors_cached("random", ti.GATE_SEED)
        assert (gdon >= 0).all() and (env_all[gdon] == env_all).all()
    out = dict(logq=np.zeros((n, T, NV, R), np.float32), right=np.zeros((n, T, NV), bool))
    rt_all = np.asarray(E["r"], np.int64)
    factor = 2 if gdon is not None else 1
    envs, cnt = np.unique(env_all, return_counts=True)
    chunks, cur, size = [], [], 0
    for e, c_ in zip(envs, cnt):
        if cur and (size + c_) * factor > budget:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(e)
        size += c_
    if cur:
        chunks.append(cur)

    def ix(a):
        return torch.as_tensor(np.asarray(a, np.int64), device=dev)
    t0 = time.time()
    for ec in chunks:
        Mc = np.flatnonzero(np.isin(env_all, ec))
        envM, codeM = ix(env_all[Mc]), ix(code_np[Mc])
        bm = len(Mc)
        h, c = h0.expand(bm, H).contiguous(), c0.expand(bm, H).contiguous()
        if gdon is not None:
            Fc = np.unique(gdon[Mc])
            gpos = ix(np.searchsorted(Fc, gdon[Mc]))
            envF, codeF = ix(env_all[Fc]), ix(code_np[Fc])
            hf, cf = h0.expand(len(Fc), H).contiguous(), c0.expand(len(Fc), H).contiguous()
        lg = torch.zeros(bm, T, NV * R, device=dev)
        for s in range(1, T + 1):
            xin = table[envM, codeM[:, s - 1]]
            hdon = hf[gpos] if gdon is not None else None
            h, c, f = _lstep(xin, h, c, s, ctx, Whh, hdon)
            lg[:, s - 1] = h @ Wro.T + bro
            if record is not None:
                record(s, Mc, dict(u=h, c=c, f=f))
            if gdon is not None:
                hf, cf, _ = _lstep(table[envF, codeF[:, s - 1]], hf, cf, s, _CLEAN, Whh)
        lq = torch.log_softmax(lg.double().cumsum(1).reshape(bm, T, NV, R), -1)
        out["logq"][Mc] = lq.float().cpu().numpy()
        out["right"][Mc] = (lq.argmax(-1) == ix(rt_all[Mc])[:, None, :]).cpu().numpy()
    ti._sync(dev)
    out["seconds"] = time.time() - t0
    return out


# ═══════════════════════════════════════════════════════════════════
# Lesion targets and their matched controls, per checkpoint (from the clean probe run of the same weights)
# ═══════════════════════════════════════════════════════════════════

def ck_tau(run, E):
    """the stored per-unit forget-gate time constant 1 / (1 - mean f) over probes and steps 2-30 (td_analysis)"""
    with np.load(os.path.join(ti.AN, run, f"ck_{E:05d}.npz")) as z:
        return z["tau"].astype(np.float64), z["fac"], [str(v) for v in z["fac_rows"]], [str(v) for v in z["factors"]]


def readout_basis(w):
    Q, _ = np.linalg.qr(w["classifier_readout.weight"].double().numpy().T)
    return Q                                                                          # (H, 20)


@torch.no_grad()
def define(run, E, w=None, ref=None, dev=None):
    """targets, controls and diagnostics of one checkpoint (see the docstring); everything per step s = 1..T from the
    clean run of the same weights on the 10,000 probes; GPU work one step at a time (memory)"""
    dev = ti._device(dev)
    w = ti.load(run, E) if w is None else w
    ref = ti.reference(w, dev, cov=True) if ref is None else ref
    P = ti.probes()
    n = len(P["env"])
    # clean pass: per-unit second moments of c, |c|, and the cell sums of u per step (all probes; environment halves)
    S = dict(c2=torch.zeros(T + 1, H, dtype=torch.float64, device=dev),
             cabs=torch.zeros(T + 1, H, dtype=torch.float64, device=dev),
             cell=torch.zeros(T + 1, 3, R * R, H, dtype=torch.float64, device=dev))
    cell_t = torch.as_tensor(P["cell"], device=dev)
    half_t = torch.as_tensor(P["env"] < gap.N_ENV // 2, device=dev)

    def rec(s, rows, st):
        cc, u = st["c"].double(), st["u"].double()
        S["c2"][s] += (cc * cc).sum(0)
        S["cabs"][s] += cc.abs().sum(0)
        rr = torch.as_tensor(rows, device=dev)
        ce, hm = cell_t[rr], half_t[rr]
        S["cell"][s, 0].index_add_(0, ce, u)
        S["cell"][s, 1].index_add_(0, ce[hm], u[hm])
        S["cell"][s, 2].index_add_(0, ce[~hm], u[~hm])
    lforward(w, None, None, dev=dev, record=rec, budget=5000)
    mu_c = ref["mu_c"]
    var_c = S["c2"].cpu().numpy()[1:] / n - mu_c[1:] ** 2                                # (T, H)
    vc = var_c.sum(0)                                                                  # per-unit c variance, summed
    cabs = S["cabs"].cpu().numpy()[1:].mean(0) / n                                     # mean |c| over steps
    cnt = np.bincount(P["cell"], minlength=R * R)
    assert (cnt == n // (R * R)).all(), "balanced cells"
    Q = readout_basis(w)
    Qt = torch.as_tensor(Q, device=dev)
    Pc = torch.eye(H, dtype=torch.float64, device=dev) - Qt @ Qt.T
    Wt = torch.as_tensor(w["LSTM.weight_hh_l0"].double().numpy(), device=dev)
    kS, kD = max(KS_SUB), max(KS_DIM)
    rot = {}
    for sd in range(N_SUB):
        G = np.random.default_rng(SEED_ROT + sd).standard_normal((H, H - Q.shape[1]))
        G = G - Q @ (Q.T @ G)
        rot[sd] = np.linalg.qr(G)[0]                                                   # (H, H - 20) orthonormal, _|_ Q
    rot_t = {sd: torch.as_tensor(v, device=dev) for sd, v in rot.items()}

    def factors(M):
        """r0 / r1 main effects (R, H) and the r0 x r1 interaction (R R, H) of cell means M (R R, H)"""
        Mg = M.reshape(R, R, H)
        m = Mg.mean((0, 1))
        a0, a1 = Mg.mean(1) - m, Mg.mean(0) - m
        return a0, a1, (Mg - a0[:, None] - a1[None] - m).reshape(R * R, H), m

    def top_f(A, N, k):
        """top-k eigenvectors (H, k) and eigenvalues of Pc (A^T A / N) Pc (A: factor effects (N, H))"""
        _, Sv, Vh = torch.linalg.svd(A @ Pc, full_matrices=False)
        return Vh[:k].T, Sv ** 2 / N
    keys = ("joint", "r0_", "r1_")
    Vsub = {k: np.zeros((T, H, kS)) for k in keys}
    ev = {k: np.zeros((T, kS)) for k in keys}
    trf = {k: np.zeros(T) for k in keys}
    own = {f"{k}{kk}": np.zeros(T) for k in keys for kk in KS_SUB}
    rel = {kk: np.zeros(T) for kk in KS_SUB}
    tot, totQc, mu_chk = np.zeros(T), np.zeros(T), 0.
    capt = {}
    cum = {sd: np.zeros((T, H - Q.shape[1])) for sd in rot}
    evp = np.zeros((T, kD))
    VP = np.zeros((T, H, kD))
    dvar_t = np.zeros((T - 1, 4 * H))
    for t in range(T):
        Cu = torch.as_tensor(ref["cov_u"][t + 1], device=dev)
        tot[t] = float(torch.trace(Cu))
        totQc[t] = tot[t] - float(torch.trace(Qt.T @ Cu @ Qt))
        a0, a1, ej, m = factors(S["cell"][t + 1, 0] / float(cnt[0]))
        mu_chk = max(mu_chk, float((m.cpu().numpy() - ref["mu_u"][t + 1]).__abs__().max()))
        for key, A, N in (("joint", ej, R * R), ("r0_", a0, R), ("r1_", a1, R)):
            V, e = top_f(A, N, kS)
            Vsub[key][t], ev[key][t] = V.cpu().numpy(), e[:kS].cpu().numpy()
            Cf = A.T @ A / N
            trf[key][t] = float(torch.trace(Cf))
            for kk in KS_SUB:
                own[f"{key}{kk}"][t] = float(torch.trace(V[:, :kk].T @ Cf @ V[:, :kk]))
                capt.setdefault(f"{key}{kk}", np.zeros(T))[t] = float(torch.trace(V[:, :kk].T @ Cu @ V[:, :kk]))
        Vh = [top_f(factors(S["cell"][t + 1, h] / float(cnt[0] // 2))[2], R * R, kS)[0] for h in (1, 2)]
        for kk in KS_SUB:
            rel[kk][t] = float(((Vh[0][:, :kk].T @ Vh[1][:, :kk]) ** 2).sum()) / kk    # mean cos^2 of principal angles
        M = Pc @ Cu @ Pc
        e, V = torch.linalg.eigh((M + M.T) / 2)
        evp[t], VP[t] = e.flip(0)[:kD].cpu().numpy(), V.flip(1)[:, :kD].cpu().numpy()
        for sd in rot:
            Rt = rot_t[sd]
            cum[sd][t] = torch.cumsum(((Cu @ Rt) * Rt).sum(0), 0).cpu().numpy()
        if t + 1 < T:                                                                 # the drive of step t + 2
            dvar_t[t] = ((Wt @ Cu) * Wt).sum(1).cpu().numpy()
        for kk in KS_DIM:
            B = np.linalg.qr(np.concatenate([Q, VP[t, :, :kk]], 1))[0]
            Bt = torch.as_tensor(B, device=dev)
            capt.setdefault(f"keep{kk}", np.zeros(T))[t] = float(torch.trace(Bt.T @ Cu @ Bt))
    S.clear()

    def matched(v, sd):
        """per step, the number of leading columns of rotation sd whose clean variance is closest to v (T,)"""
        return np.array([int(np.argmin(np.abs(cum[sd][t] - v[t]))) + 1 for t in range(T)])
    sub = {}
    for kk in KS_SUB:
        for key in keys:
            sub[f"{key}{kk}"] = dict(V=Vsub[key][:, :, :kk], var=capt[f"{key}{kk}"], own_factor_var=own[f"{key}{kk}"])
        for sd in rot:
            msd = matched(capt[f"joint{kk}"], sd)
            sub[f"joint{kk}_vm_s{sd}"] = dict(V=[rot[sd][:, :msd[t]] for t in range(T)], m=msd,
                                               var=np.array([cum[sd][t, msd[t] - 1] for t in range(T)]))
    dims = {}
    for kk in KS_DIM:
        B = [np.linalg.qr(np.concatenate([Q, VP[t, :, :kk]], 1))[0] for t in range(T)]
        removed = tot - capt[f"keep{kk}"]
        dims[f"keep{kk}"] = dict(V=B, removed=removed, kept=capt[f"keep{kk}"])
        msd = matched(removed, 0)
        dims[f"keep{kk}_vm"] = dict(V=[rot[0][:, :msd[t]] for t in range(T)], m=msd,
                                    removed=np.array([cum[0][t, msd[t] - 1] for t in range(T)]))
    dvar = dvar_t.sum(0)
    tau, fac, fac_rows, factors_ = ck_tau(run, E)
    tau_fin = ck_tau(run, FINAL)[0]
    tau_ref = 1. / (1. - ref["f_mean"][2:].mean(0))
    return dict(run=run, E=E, x=float(ti.train_x(E)), tau=tau, tau_fin=tau_fin, tau_ref=tau_ref, vc=vc, var_c=var_c,
                cabs=cabs, f_mean=ref["f_mean"], f_med=ref["f_med"], dvar=dvar, dvar_t=dvar_t, tot=tot, totQc=totQc,
                sub=sub, dims=dims, rel=rel, evj=ev["joint"], ev0=ev["r0_"], ev1=ev["r1_"], evp=evp,
                tr_joint=trf["joint"], tr_r0=trf["r0_"], tr_r1=trf["r1_"], fac=fac, fac_rows=fac_rows, factors=factors_,
                mu_chk=mu_chk, Q=Q)


def _match_units(target, cand, vc):
    """greedy nearest-neighbour match on log c variance, without replacement, largest targets first"""
    cand = list(cand)
    lv = np.log(np.maximum(vc, 1e-300))
    out = []
    for j in sorted(target, key=lambda u: -vc[u]):
        k = int(np.argmin(np.abs(lv[cand] - lv[j])))
        out.append(cand.pop(k))
    return np.array(out, np.int64)


def conditions(D):
    """[(name, cls, spec, info)] for one checkpoint's definitions D"""
    tau, tau_fin, vc, dvar = D["tau"], D["tau_fin"], D["vc"], D["dvar"]
    order_own, order_fin = np.argsort(-tau, kind="stable"), np.argsort(-tau_fin, kind="stable")
    k10 = H * max(PCTS) // 100
    excl = np.union1d(order_own[:k10], order_fin[:k10])
    cand = np.setdiff1d(np.arange(H), excl)
    out = [("intact", "intact", {}, {})]
    for clamp, key in (("fclamp", "f_units"), ("cclamp", "c_units")):
        for pct in PCTS:
            k = H * pct // 100
            for sname, order in (("own", order_own), ("fin", order_fin)):
                tgt = np.sort(order[:k])
                mt = np.sort(_match_units(tgt, cand, vc))
                for nm, units, kind in ((f"{clamp}_{sname}{pct}", tgt, "target"),
                                        (f"{clamp}_{sname}{pct}_vm", mt, "c-variance matched")):
                    out.append((nm, "tau", {key: units}, dict(kind=kind, n=len(units), units=units,
                                                              vc=float(vc[units].sum()), tau_mean=float(tau[units].mean()),
                                                              tau_min=float(tau[units].min()))))
            for sd in range(N_RAND):
                units = np.sort(np.random.default_rng(SEED_UNITS + 100 * sd + pct).choice(cand, k, replace=False))
                out.append((f"{clamp}_rand{pct}_s{sd}", "tau", {key: units},
                            dict(kind="count-matched random", n=k, units=units, vc=float(vc[units].sum()),
                                 tau_mean=float(tau[units].mean()), tau_min=float(tau[units].min()))))
    for abl, key in (("mean", "mean_rows"), ("res", "res_rows")):
        for b, G in enumerate(GATES):
            rows = np.arange(b * H, (b + 1) * H)
            out.append((f"{abl}_{G}", "gate", {key: rows}, dict(kind="target", n=H, dvar=float(dvar[rows].sum()))))
            other = np.setdiff1d(np.arange(4 * H), rows)
            perm = np.random.default_rng(SEED_ROWS + 10 * b).permutation(other)
            cum = np.cumsum(dvar[perm])
            L = int(np.argmin(np.abs(cum - dvar[rows].sum()))) + 1
            vm = np.sort(perm[:L])
            out.append((f"{abl}_{G}_vm", "gate", {key: vm},
                        dict(kind="drive-variance matched", n=L, dvar=float(dvar[vm].sum()),
                             by_block=[int(((vm >= q * H) & (vm < (q + 1) * H)).sum()) for q in range(4)])))
        for sd in range(N_RAND):
            rr = np.sort(np.random.default_rng(SEED_ROWS + 1000 + sd).choice(4 * H, H, replace=False))
            out.append((f"{abl}_rand_s{sd}", "gate", {key: rr},
                        dict(kind="count-matched random", n=H, dvar=float(dvar[rr].sum()),
                             by_block=[int(((rr >= q * H) & (rr < (q + 1) * H)).sum()) for q in range(4)])))
        rows = np.arange(2 * H)                                                       # the i and f blocks together
        out.append((f"{abl}_fi", "gate", {key: rows}, dict(kind="target", n=2 * H, dvar=float(dvar[rows].sum()))))
        for sd in range(N_RAND):
            rr = np.sort(np.random.default_rng(SEED_ROWS + 2000 + sd).choice(4 * H, 2 * H, replace=False))
            out.append((f"{abl}_rand2k_s{sd}", "gate", {key: rr},
                        dict(kind="count-matched random", n=2 * H, dvar=float(dvar[rr].sum()),
                             by_block=[int(((rr >= q * H) & (rr < (q + 1) * H)).sum()) for q in range(4)])))
    for k in KS_SUB:
        for nm in (f"joint{k}", f"r0_{k}", f"r1_{k}") + tuple(f"joint{k}_vm_s{sd}" for sd in range(N_SUB)):
            d = D["sub"][nm]
            V = d["V"]
            Vl = [V[t] for t in range(T)] if isinstance(V, np.ndarray) else V
            kind = ("target" if nm.startswith("joint") and "_vm" not in nm else
                    "variance-matched random" if "_vm" in nm else "additive factor")
            out.append((nm, "sub", {"V": Vl, "mode": "clamp"},
                        dict(kind=kind, dims=[int(v.shape[1]) for v in Vl], var=float(d["var"].sum()),
                             share=float(d["var"].sum() / D["tot"].sum()))))
    for k in KS_DIM:
        d, dm = D["dims"][f"keep{k}"], D["dims"][f"keep{k}_vm"]
        out.append((f"keep{k}", "dim", {"V": d["V"], "mode": "keep"},
                    dict(kind="target", dims=[k + 20] * T, removed=float(d["removed"].sum()),
                         share=float(d["removed"].sum() / D["tot"].sum()))))
        out.append((f"keep{k}_vm", "dim", {"V": dm["V"], "mode": "clamp"},
                    dict(kind="variance-matched random", dims=[int(m) for m in dm["m"]], removed=float(dm["removed"].sum()),
                         share=float(dm["removed"].sum() / D["tot"].sum()))))
    return out


# ═══════════════════════════════════════════════════════════════════
# Scoring and the per-checkpoint worker
# ═══════════════════════════════════════════════════════════════════

def kl_row(lq_int, lq_les):
    """perturbation size: KL(intact || lesioned) of the marginals, mean over steps and both variables, per environment"""
    a = lq_int.astype(np.float64)
    a = a - logsumexp(a, -1, keepdims=True)
    b = lq_les.astype(np.float64)
    b = b - logsumexp(b, -1, keepdims=True)
    kl = (np.exp(a) * (a - b)).sum(-1).mean((1, 2))
    env = ti.probes()["env"]
    num = np.bincount(env, kl, gap.N_ENV)
    den = np.bincount(env, minlength=gap.N_ENV).astype(np.float64)
    return dict(kind="ratio", num=num, den=den, value=float(num.sum() / den.sum()),
                label="belief change KL(intact || lesioned), mean over steps and variables", family="perturbation",
                formula="mean_i mean_t,v KL(q_int || q_les)")


def score_all(fo, ff, lq_int):
    rows = ti.score(fo["logq"], "cheap", right=fo["right"])
    rows.update(ti.score_fwd(ff["logq"]))
    rows["kl_int"] = kl_row(lq_int, fo["logq"])
    b = ti.resample(rows)
    return (np.array([rows[k]["value"] for k in ROWS], np.float64),
            np.stack([b[k] for k in ROWS]).astype(np.float32))


def ck_file(run, E):
    return os.path.join(OUT, run, f"ck_{E:05d}.npz")


def run_ck(run, E, dev=None, only=None):
    """all lesion conditions of one checkpoint -> OUT/<run>/ck_EEEEE.npz"""
    dev = ti._device(dev)
    t0 = time.time()
    w = ti.load(run, E)
    ref = ti.reference(w, dev, cov=True)
    D = define(run, E, w, ref, dev)
    conds = conditions(D)
    if only is not None:
        conds = [c for c in conds if c[0] in only]
    t_def = time.time() - t0
    ref_nc = {k: v for k, v in ref.items() if k != "cov_u"}
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        mem_def = torch.cuda.max_memory_allocated(dev) / 2 ** 20
        torch.cuda.reset_peak_memory_stats(dev)
    names, cls, val, boot, info, secs = [], [], [], [], [], []
    lq_int = None
    for nm, cl, spec, inf in conds:
        t1 = time.time()
        fo = lforward(w, spec, ref_nc, dev=dev)
        ff = lforward(w, spec, ref_nc, fwd=True, dev=dev)
        if nm == "intact":
            lq_int = fo["logq"]
        v, b = score_all(fo, ff, lq_int)
        names.append(nm)
        cls.append(cl)
        val.append(v)
        boot.append(b)
        info.append({k: (v_.tolist() if isinstance(v_, np.ndarray) else v_) for k, v_ in inf.items()})
        secs.append(time.time() - t1)
        print(f"{run} {E} {nm:22s} acc {v[ROWS.index('acc_goal_T')]:.4f} held2 {v[ROWS.index('held_two_fwd')]:.4f} "
              f"excess {v[ROWS.index('excess')]:+.3f} kl {v[ROWS.index('kl_int')]:.4f}  ({secs[-1]:.1f} s)", flush=True)
    mem_run = torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else 0.
    print(f"{run} {E}: peak GPU memory allocated {mem_def if dev.type == 'cuda' else 0:.0f} MB (definitions), "
          f"{mem_run:.0f} MB (conditions)", flush=True)
    os.makedirs(os.path.join(OUT, run), exist_ok=True)
    fn = ck_file(run, E)
    tmp = fn + ".tmp.npz"
    dsave = {k: D[k] for k in ("tau", "tau_fin", "tau_ref", "vc", "cabs", "f_med", "dvar", "dvar_t", "tot", "totQc",
                               "evj", "ev0", "ev1", "evp", "tr_joint", "tr_r0", "tr_r1")}
    dsave.update({f"rel{k}": D["rel"][k] for k in KS_SUB})
    dsave.update({f"var_{k}": D["sub"][k]["var"] for k in D["sub"]})
    dsave.update({f"own_{k}": D["sub"][k]["own_factor_var"] for k in D["sub"] if "own_factor_var" in D["sub"][k]})
    dsave.update({f"removed_{k}": D["dims"][k]["removed"] for k in D["dims"]})
    np.savez(tmp, run=run, E=E, x=D["x"], names=np.array(names), cls=np.array(cls), rows=np.array(ROWS),
             value=np.array(val), boot=np.array(boot), seconds=np.array(secs), t_def=t_def, mu_chk=D["mu_chk"],
             info=json.dumps(info), **dsave)
    os.replace(tmp, fn)
    print(f"saved {fn} ({time.time() - t0:.0f} s; definitions {t_def:.0f} s)", flush=True)
    ti._REF.clear()
    return fn


@torch.no_grad()
def half_subspaces(w, ref, dev):
    """cross-fitting: per environment half (h 0 = environments 0-24, 1 = 25-49) and step, the top max(KS_SUB)
    eigenvectors of Pc C Pc of the r0 x r1 interaction ('joint') and of the r0 / r1 main effects of u, estimated on
    that half's 5,000 probes only; and the clean variance of u that each half's subspace holds on the OTHER half
    ({h: {key: (T, H, k)}}, {'var_<key><k>_<h>': (T,)})"""
    P = ti.probes()
    S = torch.zeros(T + 1, 2, R * R, H, dtype=torch.float64, device=dev)
    cell_t = torch.as_tensor(P["cell"], device=dev)
    half = P["env"] >= gap.N_ENV // 2
    half_t = torch.as_tensor(half, device=dev)

    C2 = torch.zeros(T + 1, H, dtype=torch.float64, device=dev)

    def rec(s, rows, st):
        rr = torch.as_tensor(rows, device=dev)
        hm, u = half_t[rr], st["u"].double()
        S[s, 0].index_add_(0, cell_t[rr][~hm], u[~hm])
        S[s, 1].index_add_(0, cell_t[rr][hm], u[hm])
        C2[s] += st["c"].double().pow(2).sum(0)
    lforward(w, None, None, dev=dev, record=rec, budget=5000)
    vc = (C2[1:].cpu().numpy() / len(P["env"]) - ref["mu_c"][1:] ** 2).sum(0)       # per-unit c variance, summed
    Q = readout_basis(w)
    Qt = torch.as_tensor(Q, device=dev)
    Pc = torch.eye(H, dtype=torch.float64, device=dev) - Qt @ Qt.T
    kS = max(KS_SUB)
    cnt = float(len(P["env"]) // (R * R) // 2)
    V = {h: {key: np.zeros((T, H, kS)) for key in ("joint", "r0_", "r1_")} for h in (0, 1)}
    for t in range(T):
        for h in (0, 1):
            Mg = (S[t + 1, h] / cnt).reshape(R, R, H)
            m = Mg.mean((0, 1))
            a0, a1 = Mg.mean(1) - m, Mg.mean(0) - m
            ej = (Mg - a0[:, None] - a1[None] - m).reshape(R * R, H)
            for key, A in (("joint", ej), ("r0_", a0), ("r1_", a1)):
                V[h][key][t] = torch.linalg.svd(A @ Pc, full_matrices=False)[2][:kS].T.cpu().numpy()
    # the clean variance each half's subspace holds on the other half's probes
    Vt = {(h, key, k): [torch.as_tensor(V[h][key][t][:, :k], device=dev) for t in range(T)]
          for h in (0, 1) for key in V[h] for k in KS_SUB}
    acc = {kk: torch.zeros(T + 1, 2, dtype=torch.float64, device=dev) for kk in Vt}       # sum p^2, (sum p)^2 later
    acc1 = {kk: torch.zeros(T + 1, max(KS_SUB), dtype=torch.float64, device=dev) for kk in Vt}
    nb = {0: float((~half).sum()), 1: float(half.sum())}

    def rec2(s, rows, st):
        rr = torch.as_tensor(rows, device=dev)
        hm = half_t[rr]
        for (h, key, k), Vs in Vt.items():
            sel = hm if h == 0 else ~hm                                                # the OTHER half's probes
            p = st["u"][sel].double() @ Vs[s - 1]
            acc[(h, key, k)][s, 0] += (p * p).sum()
            acc1[(h, key, k)][s, :k] += p.sum(0)
    lforward(w, None, None, dev=dev, record=rec2, budget=5000)
    var = {}
    for (h, key, k) in Vt:
        n_o = nb[1 - h]
        v = acc[(h, key, k)][1:, 0] / n_o - (acc1[(h, key, k)][1:] / n_o).pow(2).sum(1)
        var[f"var_{key}{k}_{h}"] = v.cpu().numpy()
    var["vc"] = vc
    return V, var


def conditions_extra(run, E, HS=None):
    """the second pass (added after the first one ran): the forget gates of the long-tau sets clamped to each unit's
    OWN clean per-step probe mean (keeps every unit's average time constant, removes the state and input dependence of
    its forgetting), count-matched random units the same way, and every unit's forget gate at its own mean; with HS
    (half_subspaces) the cross-fitted interaction and r0 / r1 subspaces (estimated on one half of the environments,
    clamped on the other half's probes)"""
    tau, tau_fin = ck_tau(run, E)[0], ck_tau(run, FINAL)[0]
    order_own, order_fin = np.argsort(-tau, kind="stable"), np.argsort(-tau_fin, kind="stable")
    k10 = H * max(PCTS) // 100
    cand = np.setdiff1d(np.arange(H), np.union1d(order_own[:k10], order_fin[:k10]))
    out = [("intact", "intact", {}, {})]
    for pct in PCTS:
        k = H * pct // 100
        for sname, order in (("own", order_own), ("fin", order_fin)):
            units = np.sort(order[:k])
            out.append((f"fself_{sname}{pct}", "tau", {"f_units": units, "f_value": "self"},
                        dict(kind="target", n=k, units=units, tau_mean=float(tau[units].mean()))))
        for sd in range(2):
            units = np.sort(np.random.default_rng(SEED_UNITS + 100 * sd + pct).choice(cand, k, replace=False))
            out.append((f"fself_rand{pct}_s{sd}", "tau", {"f_units": units, "f_value": "self"},
                        dict(kind="count-matched random", n=k, units=units, tau_mean=float(tau[units].mean()))))
    out.append(("fself_all", "tau", {"f_units": np.arange(H), "f_value": "self"}, dict(kind="target", n=H)))
    if HS is not None:
        vc = HS[1]["vc"]
        for pct in PCTS:
            tgt = np.sort(order_own[:H * pct // 100])
            perm = np.random.default_rng(SEED_UNITS + 7 + pct).permutation(cand)
            L_ = int(np.argmin(np.abs(np.cumsum(vc[perm]) - vc[tgt].sum()))) + 1
            vt = np.sort(perm[:L_])
            for clamp, key in (("fclamp", "f_units"), ("cclamp", "c_units")):
                out.append((f"{clamp}_own{pct}_vt", "tau", {key: vt},
                            dict(kind="total c-variance matched", n=L_, units=vt, vc=float(vc[vt].sum()),
                                 vc_target=float(vc[tgt].sum()), tau_mean=float(tau[vt].mean()))))
        V, var = HS
        for k in KS_SUB:
            for key in ("joint", "r0_", "r1_"):
                sp = {h: {"V": [V[h][key][t][:, :k] for t in range(T)], "mode": "clamp"} for h in (0, 1)}
                out.append((f"{key}{k}_xfit", "sub", {"xfit": sp},
                            dict(kind="target" if key == "joint" else "additive factor",
                                 var_on_other_half=float(var[f"var_{key}{k}_0"].sum() + var[f"var_{key}{k}_1"].sum()) / 2)))
    return out


def extra_file(run, E):
    return os.path.join(OUT, run, f"extra_{E:05d}.npz")


def run_extra(run, E, dev=None):
    """the second-pass conditions of one checkpoint -> OUT/<run>/extra_EEEEE.npz (same layout as ck_EEEEE.npz)"""
    dev = ti._device(dev)
    t0 = time.time()
    w = ti.load(run, E)
    ref = ti.reference(w, dev)
    HS = half_subspaces(w, ref, dev)
    names, cls, val, boot, info = [], [], [], [], []
    lq_int = None
    for nm, cl, spec, inf in conditions_extra(run, E, HS):
        if "xfit" in spec:                     # half 0's subspace on environments 25-49, half 1's on 0-24
            fo, ff = {}, {}
            for out_, fw in ((fo, False), (ff, True)):
                a = lforward(w, spec["xfit"][0], ref, fwd=fw, dev=dev)
                b = lforward(w, spec["xfit"][1], ref, fwd=fw, dev=dev)
                env = np.asarray((ti.forward_set() if fw else ti.probes())["env"])
                hi = (env >= gap.N_ENV // 2)
                out_["logq"] = np.where(hi[:, None, None, None], a["logq"], b["logq"])
                out_["right"] = np.where(hi[:, None, None], a["right"], b["right"])
        else:
            fo = lforward(w, spec, ref, dev=dev)
            ff = lforward(w, spec, ref, fwd=True, dev=dev)
        if nm == "intact":
            lq_int = fo["logq"]
        v, b = score_all(fo, ff, lq_int)
        names.append(nm)
        cls.append(cl)
        val.append(v)
        boot.append(b)
        info.append({k: (v_.tolist() if isinstance(v_, np.ndarray) else v_) for k, v_ in inf.items()})
        print(f"{run} {E} {nm:22s} acc {v[ROWS.index('acc_goal_T')]:.4f} held2 {v[ROWS.index('held_two_fwd')]:.4f} "
              f"excess {v[ROWS.index('excess')]:+.3f} kl {v[ROWS.index('kl_int')]:.4f}", flush=True)
    fn = extra_file(run, E)
    tmp = fn + ".tmp.npz"
    np.savez(tmp, run=run, E=E, x=float(ti.train_x(E)), names=np.array(names), cls=np.array(cls), rows=np.array(ROWS),
             value=np.array(val), boot=np.array(boot), info=json.dumps(info))
    os.replace(tmp, fn)
    print(f"saved {fn} ({time.time() - t0:.0f} s)", flush=True)
    ti._REF.clear()
    return fn


def worker_extra(run, eps):
    for E in eps:
        if not os.path.exists(extra_file(run, E)):
            run_extra(run, E)


def worker(run, eps):
    for E in eps:
        if os.path.exists(ck_file(run, E)):
            print(f"{run} {E}: exists", flush=True)
            continue
        run_ck(run, E)


# ═══════════════════════════════════════════════════════════════════
# Third pass (fix round, 11-02): g-free random drive rows, |df|-matched forget-gate controls, more subspace controls
# ═══════════════════════════════════════════════════════════════════

SEED_NOG, SEED_SPAN = 20261221, 20261223
SHIFT_SETS = ("own5", "own10", "fin10")                   # long-tau sets that get a |df|-matched random control
F_CLIP = .99                                              # shifted forget gates are clipped to [0, .99]
K_SPAN = tuple(range(5, 61)) + tuple(range(65, 201, 5))   # candidate top-K PC spans of the pcspan control
SUB3 = ("joint5", "r0_5", "r1_5", "add5", "pure5", "pc5", "pcspan5_s0", "pcspan5_s1")
SPLIT = ("environment", "r0", "r1", "r0 x r1", "environment x cell", "within cell")


def nog_rows(n, sd):
    """n random rows of the recurrent drive from the i, f and o blocks only (no g row; fixed seed per n and sd; the
    same sets for mean- and resample-ablation)"""
    pool = np.concatenate([np.arange(b * H, (b + 1) * H) for b, G in enumerate(GATES) if G != "g"])
    return np.sort(np.random.default_rng(SEED_NOG + 100 * sd + n // H).choice(pool, n, replace=False))


def _stored_units(run, E):
    """{condition: units} of the first and second passes' unit conditions at one checkpoint (their stored info)"""
    out = {}
    for fn in (ck_file(run, E), extra_file(run, E)):
        with np.load(fn) as z:
            for nm, inf in zip([str(v) for v in z["names"]], json.loads(str(z["info"]))):
                if "units" in inf:
                    out[nm] = np.asarray(inf["units"], np.int64)
    return out


def third_file(run, E):
    return os.path.join(OUT, run, f"third_{E:05d}.npz")


@torch.no_grad()
def define3(run, E, w, ref, dev, subspaces=True):
    """third-pass definitions of one checkpoint, from the clean run of the same weights on the 10,000 probes.
    (a) |df|-matched forget-gate controls: for a long-tau set S (own top 5 / 10%, final top 10%) with per-step shift
    delta(s) = mean_{j in S} f_mean_j(s) - f_med(s) (how far the median clamp moves S on average), the first pass's
    count-matched random units (same seeds) get f_j(s) := clip(f_mean_j(s) + delta(s), 0, .99): the same mean shift
    of the same number of units (upward: the non-long-tau units' forget gates are too low for a downward shift of that
    size) and, like the target clamp, no state or input dependence. Also the clean per-probe perturbation of every
    forget-gate set: mean over probes, steps and units of |f - assigned| (and of assigned - f).
    (b) with subspaces, per step: add5 = the top 5 eigenvectors of Pc (C_r0 + C_r1) Pc (both latents' main effects,
    no interaction; 1.08-2.04x joint5's variance), pure5 = the top 5 of Pp C_joint Pp (Pp also projects out the
    <= 18-dim span of the main-effect cell means, so pure5 holds no main-effect variance), pc5 = the top 5 of Pc C_u
    Pc; variance-matched random dims for each (the first pass's random bases, 2 seeds); pcspan5 = 5 random dims inside
    the top-K PC span (of Pc C_u Pc), K per step chosen so that their clean variance matches joint5's (2 seeds); and
    the factor split of the clean variance inside each 5-dim subspace and in all of Pc u: environment, r0, r1, r0 x r1
    (cell means pooled over environments), environment x cell and within cell (the two streams of a cell), additive
    (balanced probes)"""
    P = ti.probes()
    n = len(P["env"])
    U = _stored_units(run, E)
    f_mean, f_med = ref["f_mean"], ref["f_med"]
    fset, delta = {}, {}
    for nm, u in U.items():
        if nm.startswith("fclamp_"):
            fset[nm] = (u, np.repeat(f_med[:, None], len(u), 1))
        elif nm.startswith("fself_"):
            fset[nm] = (u, f_mean[:, u])
    for sname in SHIFT_SETS:
        pct = int(sname[3:])
        d = f_mean[:, U[f"fclamp_{sname}"]].mean(1) - f_med                               # (T + 1,), NaN at 0
        for sd in range(N_RAND):
            ru = U[f"fclamp_rand{pct}_s{sd}"]
            fset[f"fshift_{sname}_s{sd}"] = (ru, np.clip(f_mean[:, ru] + d[:, None], 0., F_CLIP))
            delta[f"fshift_{sname}_s{sd}"] = d
    ft = {k: (torch.as_tensor(u, device=dev), torch.as_tensor(np.nan_to_num(v), device=dev)) for k, (u, v) in fset.items()}
    dabs = {k: torch.zeros(T + 1, dtype=torch.float64, device=dev) for k in fset}
    dsgn = {k: torch.zeros(T + 1, dtype=torch.float64, device=dev) for k in fset}
    cell_t, env_t = torch.as_tensor(P["cell"], device=dev), torch.as_tensor(P["env"], device=dev)
    Sc = torch.zeros(T + 1, R * R, H, dtype=torch.float64, device=dev) if subspaces else None

    def rec(s, rows, st):
        f = st["f"].double()
        for k, (u, v) in ft.items():
            dd = v[s][None] - f[:, u]
            dabs[k][s] += dd.abs().sum()
            dsgn[k][s] += dd.sum()
        if Sc is not None:
            Sc[s].index_add_(0, cell_t[torch.as_tensor(rows, device=dev)], st["u"].double())
    lforward(w, None, None, dev=dev, record=rec, budget=5000)
    out = dict(fsets={k: v for k, v in fset.items()}, delta=delta,
               dfabs={k: dabs[k].cpu().numpy()[1:] / (n * len(fset[k][0])) for k in fset},
               dfsgn={k: dsgn[k].cpu().numpy()[1:] / (n * len(fset[k][0])) for k in fset})
    if not subspaces:
        return out
    Q = readout_basis(w)
    Qt = torch.as_tensor(Q, device=dev)
    Pc = torch.eye(H, dtype=torch.float64, device=dev) - Qt @ Qt.T
    rot = {}
    for sd in range(N_SUB):                                                            # as define()
        G = np.random.default_rng(SEED_ROT + sd).standard_normal((H, H - Q.shape[1]))
        G = G - Q @ (Q.T @ G)
        rot[sd] = torch.as_tensor(np.linalg.qr(G)[0], device=dev)
    cnt = n // (R * R)
    V = {k: np.zeros((T, H, 5)) for k in SUB3}
    var = {k: np.zeros(T) for k in SUB3}
    main_var = {k: np.zeros(T) for k in SUB3}
    kspan = {sd: np.zeros(T, np.int64) for sd in range(N_SUB)}
    cum = {sd: np.zeros((T, H - Q.shape[1])) for sd in rot}
    tot, rank_main = np.zeros(T), np.zeros(T, np.int64)

    def top(A, Pm, k=5):
        return torch.linalg.svd(A @ Pm, full_matrices=False)[2][:k].T
    for t in range(T):
        Cu = torch.as_tensor(ref["cov_u"][t + 1], device=dev)
        tot[t] = float(torch.trace(Cu))
        Mg = (Sc[t + 1] / float(cnt)).reshape(R, R, H)
        m = Mg.mean((0, 1))
        a0, a1 = Mg.mean(1) - m, Mg.mean(0) - m
        ej = (Mg - a0[:, None] - a1[None] - m).reshape(R * R, H)
        A01 = torch.cat([a0, a1])
        Vt = {"joint5": top(ej, Pc), "r0_5": top(a0, Pc), "r1_5": top(a1, Pc), "add5": top(A01, Pc)}
        _, Sm, Vm = torch.linalg.svd(A01 @ Pc, full_matrices=False)
        Bm = Vm[Sm > 1e-9 * Sm.max()].T                                                 # main-effect span, off Q
        rank_main[t] = Bm.shape[1]
        Vt["pure5"] = top(ej, Pc - Bm @ Bm.T)
        Mp = Pc @ Cu @ Pc
        VP = torch.linalg.eigh((Mp + Mp.T) / 2)[1].flip(1)[:, :max(K_SPAN)]
        Vt["pc5"] = VP[:, :5]
        vj = float(torch.trace(Vt["joint5"].T @ Cu @ Vt["joint5"]))
        for sd in range(N_SUB):
            rng = np.random.default_rng(SEED_SPAN + 1000 * sd + t)
            best = None
            for K in K_SPAN:
                Vk = torch.linalg.qr(VP[:, :K] @ torch.as_tensor(rng.standard_normal((K, 5)), device=dev))[0]
                v = float(torch.trace(Vk.T @ Cu @ Vk))
                if best is None or abs(v - vj) < abs(best[1] - vj):
                    best = (Vk, v, K)
            Vt[f"pcspan5_s{sd}"], kspan[sd][t] = best[0], best[2]
        Cm = (a0.T @ a0 + a1.T @ a1) / R
        for k, Vk in Vt.items():
            V[k][t] = Vk.cpu().numpy()
            var[k][t] = float(torch.trace(Vk.T @ Cu @ Vk))
            main_var[k][t] = float(torch.trace(Vk.T @ Cm @ Vk))
        for sd, Rt in rot.items():
            cum[sd][t] = torch.cumsum(((Cu @ Rt) * Rt).sum(0), 0).cpu().numpy()
    Sc = None
    vm = {}
    for k in ("add5", "pure5", "pc5"):
        for sd in rot:
            msd = np.array([int(np.argmin(np.abs(cum[sd][t] - var[k][t]))) + 1 for t in range(T)])
            vm[f"{k}_vm_s{sd}"] = dict(V=[rot[sd][:, :msd[t]].cpu().numpy() for t in range(T)], m=msd,
                                       var=np.array([cum[sd][t, msd[t] - 1] for t in range(T)]))
    # factor split of the clean variance inside each subspace and in all of Pc u (second clean pass)
    Vs = {k: [torch.as_tensor(V[k][t], device=dev) for t in range(T)] for k in SUB3}
    mu = torch.as_tensor(ref["mu_u"], device=dev)
    dims = {k: 5 for k in SUB3}
    dims["all"] = H
    acc = {k: dict(tot=torch.zeros(T + 1, dtype=torch.float64, device=dev),
                   env=torch.zeros(T + 1, gap.N_ENV, d, dtype=torch.float64, device=dev),
                   cell=torch.zeros(T + 1, R * R, d, dtype=torch.float64, device=dev),
                   pair=torch.zeros(T + 1, dtype=torch.float64, device=dev)) for k, d in dims.items()}
    pairs_ok = [True]

    def rec2(s, rows, st):
        rows = np.asarray(rows)
        pairs_ok[0] &= bool((rows[0::2] % 2 == 0).all() and (rows[1::2] == rows[0::2] + 1).all())
        rr = torch.as_tensor(rows, device=dev)
        x = st["u"].double() - mu[s]
        for k, a in acc.items():
            p = x @ Pc if k == "all" else x @ Vs[k][s - 1]
            a["tot"][s] += (p * p).sum()
            a["env"][s].index_add_(0, env_t[rr], p)
            a["cell"][s].index_add_(0, cell_t[rr], p)
            dd = p[0::2] - p[1::2]
            a["pair"][s] += (dd * dd).sum()
    lforward(w, None, None, dev=dev, record=rec2, budget=5000)
    assert pairs_ok[0], "the two streams of a cell are adjacent rows"
    split, split_tot = {}, {}
    for k, a in acc.items():
        tt_ = a["tot"][1:].cpu().numpy() / n
        em = a["env"][1:].cpu().numpy() / (n / gap.N_ENV)                                 # (T, NE, d)
        Mg = (a["cell"][1:].cpu().numpy() / cnt).reshape(T, R, R, -1)
        g0, g1 = Mg.mean(2), Mg.mean(1)                                                  # r0 / r1 effects
        jt = Mg - g0[:, :, None] - g1[:, None]
        ev, r0v, r1v = (em ** 2).sum(-1).mean(1), (g0 ** 2).sum(-1).mean(1), (g1 ** 2).sum(-1).mean(1)
        jv, res = (jt ** 2).sum(-1).mean((1, 2)), a["pair"][1:].cpu().numpy() / (2 * n)
        split[k] = np.stack([ev, r0v, r1v, jv, tt_ - ev - r0v - r1v - jv - res, res], 1)    # (T, 6)
        split_tot[k] = tt_
    acc.clear()
    out.update(V={k: V[k] for k in SUB3}, var=var, main_var=main_var, kspan=kspan, vm=vm, tot=tot,
               rank_main=rank_main, split=split, split_tot=split_tot,
               split_chk=float(max(np.abs(split_tot[k] / var[k] - 1).max() for k in SUB3)),
               split_all_chk=float(np.abs(split_tot["all"] / (tot - np.array([float(np.trace(Q.T @ ref["cov_u"][t + 1]
                                                                                             @ Q)) for t in range(T)]))
                                          - 1).max()))
    return out


def conditions_third(D3):
    """[(name, cls, spec, info)] of the third pass: intact; count-matched random drive rows without g (1,000 and 2,000
    rows from the i, f and o blocks, 3 seeds, mean- and resample-ablated); |df|-matched random forget-gate units;
    with subspaces add5 / pure5 / pc5, their variance-matched random dims and pcspan5 (see define3)"""
    out = [("intact", "intact", {}, {})]
    for abl, key in (("mean", "mean_rows"), ("res", "res_rows")):
        for nr, tag in ((H, "nog"), (2 * H, "nog2k")):
            for sd in range(N_RAND):
                rr = nog_rows(nr, sd)
                out.append((f"{abl}_{tag}_s{sd}", "gate", {key: rr},
                            dict(kind="count-matched random, no g rows", n=nr,
                                 by_block=[int(((rr >= q * H) & (rr < (q + 1) * H)).sum()) for q in range(4)])))
    for nm, d in D3["delta"].items():
        u, val = D3["fsets"][nm]
        out.append((nm, "tau", {"f_units": u, "f_value": val},
                    dict(kind="count-matched random, |df| matched", n=len(u), units=u,
                         delta_mean=float(np.nanmean(d[1:])), dfabs=float(D3["dfabs"][nm].mean()))))
    if "V" in D3:
        for k in ("add5", "pure5", "pc5"):
            out.append((k, "sub", {"V": [D3["V"][k][t] for t in range(T)], "mode": "clamp"},
                        dict(kind="target", var=float(D3["var"][k].sum()), share=float(D3["var"][k].sum() / D3["tot"].sum()))))
            for sd in range(N_SUB):
                d = D3["vm"][f"{k}_vm_s{sd}"]
                out.append((f"{k}_vm_s{sd}", "sub", {"V": d["V"], "mode": "clamp"},
                            dict(kind="variance-matched random", dims=[int(m) for m in d["m"]], var=float(d["var"].sum()),
                                 share=float(d["var"].sum() / D3["tot"].sum()))))
        for sd in range(N_SUB):
            k = f"pcspan5_s{sd}"
            out.append((k, "sub", {"V": [D3["V"][k][t] for t in range(T)], "mode": "clamp"},
                        dict(kind="variance-matched random inside the top-K PC span", K=D3["kspan"][sd].tolist(),
                             var=float(D3["var"][k].sum()), share=float(D3["var"][k].sum() / D3["tot"].sum()))))
    return out


def run_third(run, E, dev=None, subspaces=None):
    """the third-pass conditions of one checkpoint -> OUT/<run>/third_EEEEE.npz (same layout as ck_EEEEE.npz, plus
    define3's diagnostics); the Trained RNN gets the subspace controls (subspaces=None: at the window checkpoints,
    W3_SUB; their definitions cost ~4 min per checkpoint on the shared GPU), the ESN only the gate and forget-gate ones"""
    dev = ti._device(dev)
    t0 = time.time()
    w = ti.load(run, E)
    sub = (run == "trained" and E in W3_SUB) if subspaces is None else subspaces
    ref = ti.reference(w, dev, cov=sub)
    D3 = define3(run, E, w, ref, dev, subspaces=sub)
    t_def = time.time() - t0
    ref_nc = {k: v for k, v in ref.items() if k != "cov_u"}
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        mem_def = torch.cuda.max_memory_allocated(dev) / 2 ** 20
        torch.cuda.reset_peak_memory_stats(dev)
    names, cls, val, boot, info, secs = [], [], [], [], [], []
    lq_int = None
    for nm, cl, spec, inf in conditions_third(D3):
        t1 = time.time()
        fo = lforward(w, spec, ref_nc, dev=dev)
        ff = lforward(w, spec, ref_nc, fwd=True, dev=dev)
        if nm == "intact":
            lq_int = fo["logq"]
        v, b = score_all(fo, ff, lq_int)
        names.append(nm)
        cls.append(cl)
        val.append(v)
        boot.append(b)
        info.append({k: (v_.tolist() if isinstance(v_, np.ndarray) else v_) for k, v_ in inf.items()})
        secs.append(time.time() - t1)
        print(f"{run} {E} {nm:22s} acc {v[ROWS.index('acc_goal_T')]:.4f} held2 {v[ROWS.index('held_two_fwd')]:.4f} "
              f"excess {v[ROWS.index('excess')]:+.3f} kl {v[ROWS.index('kl_int')]:.4f}  ({secs[-1]:.1f} s)", flush=True)
    mem_run = torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else 0.
    print(f"{run} {E}: peak GPU memory allocated {mem_def if dev.type == 'cuda' else 0:.0f} MB (definitions), "
          f"{mem_run:.0f} MB (conditions)", flush=True)
    dsave = {f"dfabs_{k}": v for k, v in D3["dfabs"].items()}
    dsave.update({f"dfsgn_{k}": v for k, v in D3["dfsgn"].items()})
    dsave.update({f"delta_{k}": v for k, v in D3["delta"].items()})
    if sub:
        dsave.update({f"var_{k}": v for k, v in D3["var"].items()})
        dsave.update({f"mainvar_{k}": v for k, v in D3["main_var"].items()})
        dsave.update({f"split_{k}": v for k, v in D3["split"].items()})
        dsave.update({f"kspan_s{sd}": v for sd, v in D3["kspan"].items()})
        dsave.update({f"var_{k}": d["var"] for k, d in D3["vm"].items()})
        dsave.update(tot3=D3["tot"], rank_main=D3["rank_main"], split_chk=D3["split_chk"],
                     split_all_chk=D3["split_all_chk"])
    fn = third_file(run, E)
    tmp = fn + ".tmp.npz"
    np.savez(tmp, run=run, E=E, x=float(ti.train_x(E)), names=np.array(names), cls=np.array(cls), rows=np.array(ROWS),
             value=np.array(val), boot=np.array(boot), seconds=np.array(secs), t_def=t_def, info=json.dumps(info),
             **dsave)
    os.replace(tmp, fn)
    print(f"saved {fn} ({time.time() - t0:.0f} s; definitions {t_def:.0f} s)", flush=True)
    ti._REF.clear()
    return fn


def worker_third(run, eps):
    for E in eps:
        if os.path.exists(third_file(run, E)):
            print(f"{run} {E}: third pass exists", flush=True)
            continue
        run_third(run, E)


KS_SEED2 = (20, 100)                                      # keep-PC lesions that get a second random-dims seed


def fourth_file(run, E):
    return os.path.join(OUT, run, f"fourth_{E:05d}.npz")


@torch.no_grad()
def run_fourth(run, E, dev=None):
    """the fourth pass (fix round): a second seed of the dimension lesion's variance-matched random dims (keep{k}_vm_s1,
    k in KS_SEED2: the leading columns of the first pass's second random basis, SEED_ROT + 1, removing per step the clean
    variance that keeping k PCs removes; as define()) -> OUT/<run>/fourth_EEEEE.npz"""
    dev = ti._device(dev)
    t0 = time.time()
    w = ti.load(run, E)
    ref = ti.reference(w, dev, cov=True)
    Q = readout_basis(w)
    Qt = torch.as_tensor(Q, device=dev)
    Pc = torch.eye(H, dtype=torch.float64, device=dev) - Qt @ Qt.T
    G = np.random.default_rng(SEED_ROT + 1).standard_normal((H, H - Q.shape[1]))
    rot1 = torch.as_tensor(np.linalg.qr(G - Q @ (Q.T @ G))[0], device=dev)
    V = {k: [] for k in KS_SEED2}
    var = {k: np.zeros(T) for k in KS_SEED2}
    for t in range(T):
        Cu = torch.as_tensor(ref["cov_u"][t + 1], device=dev)
        tot = float(torch.trace(Cu))
        M = Pc @ Cu @ Pc
        VP = torch.linalg.eigh((M + M.T) / 2)[1].flip(1)[:, :max(KS_SEED2)]
        cum = torch.cumsum(((Cu @ rot1) * rot1).sum(0), 0).cpu().numpy()
        for k in KS_SEED2:
            B = torch.linalg.qr(torch.cat([Qt, VP[:, :k]], 1))[0]
            removed = tot - float(torch.trace(B.T @ Cu @ B))
            m = int(np.argmin(np.abs(cum - removed))) + 1
            V[k].append(rot1[:, :m].cpu().numpy())
            var[k][t] = cum[m - 1] / removed
    ref_nc = {k: v for k, v in ref.items() if k != "cov_u"}
    conds = [("intact", "intact", {}, {})] + [
        (f"keep{k}_vm_s1", "dim", {"V": V[k], "mode": "clamp"},
         dict(kind="variance-matched random, seed 1", dims=[int(v.shape[1]) for v in V[k]],
              removed_ratio=float(np.mean(var[k])))) for k in KS_SEED2]
    names, cls, val, boot, info = [], [], [], [], []
    lq_int = None
    for nm, cl, spec, inf in conds:
        fo = lforward(w, spec, ref_nc, dev=dev)
        ff = lforward(w, spec, ref_nc, fwd=True, dev=dev)
        if nm == "intact":
            lq_int = fo["logq"]
        v, b = score_all(fo, ff, lq_int)
        names.append(nm)
        cls.append(cl)
        val.append(v)
        boot.append(b)
        info.append(inf)
        print(f"{run} {E} {nm:22s} acc {v[ROWS.index('acc_goal_T')]:.4f} held2 {v[ROWS.index('held_two_fwd')]:.4f} "
              f"excess {v[ROWS.index('excess')]:+.3f}", flush=True)
    fn = fourth_file(run, E)
    tmp = fn + ".tmp.npz"
    np.savez(tmp, run=run, E=E, x=float(ti.train_x(E)), names=np.array(names), cls=np.array(cls), rows=np.array(ROWS),
             value=np.array(val), boot=np.array(boot), info=json.dumps(info))
    os.replace(tmp, fn)
    print(f"saved {fn} ({time.time() - t0:.0f} s)", flush=True)
    ti._REF.clear()
    return fn


# raw checkpoints of the early (800-1,600) and late (>= 2,400) windows: the third pass's subspace controls run only
# there (all 20 checkpoints would take ~75 min of GPU time; the claims they test are window means)
W3_SUB = (1000, 1200, 1400, 1600, 2000, 3000, 4000, 5000, 7000, 10000, 15000, 20000, 30000, 49995)


# ═══════════════════════════════════════════════════════════════════
# Collection: gap fractions per row and family, effects, matched-control differences, intervals
# ═══════════════════════════════════════════════════════════════════

def refs_rows():
    """naive's and Bayes's levels of ROWS (without kl_int) and their resamples (paired: the same counts); cached"""
    fn = os.path.join(OUT, "refs.npz")
    if os.path.exists(fn):
        with np.load(fn) as z:
            return {k: z[k] for k in z.files}
    rf = ti.refs("cheap")
    rr = [k for k in ROWS if k != "kl_int"]
    out = {}
    for nm in ("naive", "bayes"):
        b = ti.resample({k: rf[nm][k] for k in rr})
        out[nm] = np.array([rf[nm][k]["value"] for k in rr])
        out[nm + "_boot"] = np.stack([b[k] for k in rr])
    out["rows"] = np.array(rr)
    os.makedirs(OUT, exist_ok=True)
    np.savez(fn, **out)
    return out


def _add_pass(out, files, tag, need_all=True):
    """append a later pass's conditions to a loaded run; a condition missing at a checkpoint (the third pass's subspace
    controls outside W3_SUB; with need_all=False a whole missing file: the fourth pass before 640) is NaN there; its
    intact must equal the first pass's"""
    have = [os.path.exists(f) for f in files]
    if (need_all and not all(have)) or not any(have):
        return
    Z = [np.load(f) if h else None for f, h in zip(files, have)]
    ns = [[str(v) for v in z["names"]] if z is not None else [] for z in Z]
    full = max(ns, key=len)
    assert all(nn == full[:len(nn)] for nn in ns)
    keep = [i for i, v in enumerate(full) if v != "intact"]
    i0 = out["names"].index("intact")
    out[f"{tag}_intact_diff"] = float(max(np.nanmax(np.abs(z["value"][full.index("intact")] - out["value"][k, i0]))
                                          for k, z in enumerate(Z) if z is not None))
    zf = Z[[len(nn) for nn in ns].index(len(full))]
    shp = {"value": zf["value"].shape[1:], "boot": zf["boot"].shape[1:]}

    def pad(z, nn, key):
        a = z[key] if z is not None else np.zeros((0,) + shp[key], zf[key].dtype)
        p = np.full((len(full),) + shp[key], np.nan, a.dtype if a.dtype.kind == "f" else np.float64)
        p[:len(nn)] = a
        return p[keep]
    out["names"] = out["names"] + [full[i] for i in keep]
    out["cls"] = out["cls"] + [str(zf["cls"][i]) for i in keep]
    out["value"] = np.concatenate([out["value"], np.stack([pad(z, nn, "value") for z, nn in zip(Z, ns)])], 1)
    out["boot"] = np.concatenate([out["boot"], np.stack([pad(z, nn, "boot") for z, nn in zip(Z, ns)])], 1)
    for k, (z, nn) in enumerate(zip(Z, ns)):
        inf = (json.loads(str(z["info"])) if z is not None else []) + [{} for _ in range(len(full) - len(nn))]
        out["info"][k] = out["info"][k] + [inf[i] for i in keep]
    out[{"third": "defs3", "fourth": "defs4"}[tag]] = [
        {k: z[k] for k in z.files if k not in ("value", "boot", "names", "cls", "rows", "info")} if z is not None else {}
        for z in Z]


def load_run(run):
    """all stored checkpoints of a run: dict(x (K,), E, names (C,), cls, rows, value (K, C, Rw), boot (K, C, Rw, NB),
    info [K][C], defs [K] (the per-checkpoint definition arrays))"""
    key = ("run", run)
    if key in _C:
        return _C[key]
    eps = CKS if run == "trained" else CKS_ESN
    Z = [np.load(ck_file(run, E)) for E in eps if os.path.exists(ck_file(run, E))]
    assert Z, run
    names = [str(v) for v in Z[0]["names"]]
    for z in Z:
        assert [str(v) for v in z["names"]] == names and [str(v) for v in z["rows"]] == list(ROWS)
    out = dict(x=np.array([float(z["x"]) for z in Z]), E=np.array([int(z["E"]) for z in Z]), names=names,
               cls=[str(v) for v in Z[0]["cls"]], rows=list(ROWS), value=np.stack([z["value"] for z in Z]),
               boot=np.stack([z["boot"] for z in Z]), info=[json.loads(str(z["info"])) for z in Z],
               defs=[{k: z[k] for k in z.files if k not in ("value", "boot", "names", "cls", "rows", "info")} for z in Z],
               seconds=np.stack([z["seconds"] for z in Z]))
    X = [extra_file(run, int(E)) for E in out["E"]]
    if all(os.path.exists(f) for f in X):                    # the second pass: its own conditions (its intact = check)
        Zx = [np.load(f) for f in X]
        nx = [str(v) for v in Zx[0]["names"]]
        keep = [i for i, v in enumerate(nx) if v != "intact"]
        ii = nx.index("intact")
        out["extra_intact_diff"] = float(max(np.nanmax(np.abs(z["value"][ii] - out["value"][k, names.index("intact")]))
                                             for k, z in enumerate(Zx)))
        out["names"] = names + [nx[i] for i in keep]
        out["cls"] = out["cls"] + [str(Zx[0]["cls"][i]) for i in keep]
        out["value"] = np.concatenate([out["value"], np.stack([z["value"][keep] for z in Zx])], 1)
        out["boot"] = np.concatenate([out["boot"], np.stack([z["boot"][keep] for z in Zx])], 1)
        for k, z in enumerate(Zx):
            inf = json.loads(str(z["info"]))
            out["info"][k] = out["info"][k] + [inf[i] for i in keep]
    for tag, ffn, need in (("third", third_file, True), ("fourth", fourth_file, False)):   # fix round: NaN-padded
        _add_pass(out, [ffn(run, int(E)) for E in out["E"]], tag, need)
    _C[key] = out
    return out


def gapfrac(L):
    """per row gap fractions f = (y - naive) / (Bayes - naive) of every (checkpoint, condition): f (K, C, Rw') and the
    resampled fb (K, C, Rw', NB) for the rows with references (Rw' = ROWS without kl_int)"""
    rf = refs_rows()
    rr = [str(v) for v in rf["rows"]]
    ix = [L["rows"].index(k) for k in rr]
    y, yb = L["value"][:, :, ix], L["boot"][:, :, ix].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (y - rf["naive"]) / (rf["bayes"] - rf["naive"])
        fb = (yb - rf["naive_boot"]) / (rf["bayes_boot"] - rf["naive_boot"])
    return rr, f, fb


def family_effects(L, groups=None):
    """family gap fractions of the intact network and the change under every condition, in gap units (the mean over the
    family's rows of f_les - f_int, over the rows defined in both): dict(F_int (K, fam), F_int_b (K, fam, NB),
    dF (K, C, fam), dF_b (K, C, fam, NB)); groups = {name: rows} (default FAMILY_ROWS; PANEL_ROWS: the figures' panels)"""
    groups = FAMILY_ROWS if groups is None else groups
    key = ("fam", id(L), tuple(groups))
    if key in _C:
        return _C[key]
    rr, f, fb = gapfrac(L)
    c0 = L["names"].index("intact")
    K, C = f.shape[:2]
    G = len(groups)
    out = dict(F_int=np.zeros((K, G)), F_int_b=np.zeros((K, G, gap.N_BOOT)), dF=np.zeros((K, C, G)),
               dF_b=np.zeros((K, C, G, gap.N_BOOT)))
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for j, fam in enumerate(groups):
                ii = [rr.index(k) for k in groups[fam]]
                out["F_int"][:, j] = np.nanmean(f[:, c0, ii], -1)
                out["F_int_b"][:, j] = np.nanmean(fb[:, c0, ii], 1)
                out["dF"][:, :, j] = np.nanmean(f[:, :, ii] - f[:, c0:c0 + 1, ii], -1)
                out["dF_b"][:, :, j] = np.nanmean(fb[:, :, ii] - fb[:, c0:c0 + 1, ii], 2)
    _C[key] = out
    return out


def iv(b, axis=-1):
    """5-95% over the resamples"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(b, 5, axis=axis), np.nanpercentile(b, 95, axis=axis)


def controls(name, names=None):
    """the matched controls of a target condition: {kind: [names]}; kinds: 'vm' (variance- or c-variance-matched),
    'rand' (count-matched random), 'additive' (r0 / r1 subspaces), 'vt' (same total c variance), 'nog' (count-matched
    random drive rows without g rows), 'shift' (count-matched random units with the same mean |df|), 'add5' (both
    latents' main effects, 5 dims), 'pcspan' (5 random dims in the top-K PC span, same variance), 'pc5' (top-5 PCs).
    With names: only the kinds whose controls all exist there"""
    out = _controls(name)
    if names is not None:
        out = {k: v for k, v in out.items() if all(c in names for c in v)}
    return out


def _controls(name):
    if name == "fself_all":
        return {}
    if name.endswith("_xfit"):
        k = name[5:-5]
        return dict(additive=[f"r0_{k}_xfit", f"r1_{k}_xfit"], vm=[f"joint{k}_vm_s{sd}" for sd in range(N_SUB)])
    if name.startswith("fself_"):
        pct = name[len("fself_own"):] if name.startswith("fself_own") else name[len("fself_fin"):]
        return dict(rand=[f"fself_rand{pct}_s{sd}" for sd in range(2)])
    if name.startswith(("fclamp_", "cclamp_")):
        clamp, rest = name.split("_", 1)
        pct = rest[3:]
        out = dict(vm=[f"{name}_vm"], rand=[f"{clamp}_rand{pct}_s{sd}" for sd in range(N_RAND)])
        if rest.startswith("own"):
            out["vt"] = [f"{name}_vt"]
        if clamp == "fclamp" and rest in SHIFT_SETS:
            out["shift"] = [f"fshift_{rest}_s{sd}" for sd in range(N_RAND)]
        return out
    if name.startswith(("mean_", "res_")):
        abl = name.split("_")[0]
        if name.endswith("_fi"):
            return dict(rand=[f"{abl}_rand2k_s{sd}" for sd in range(N_RAND)],
                        nog=[f"{abl}_nog2k_s{sd}" for sd in range(N_RAND)])
        return dict(vm=[f"{name}_vm"], rand=[f"{abl}_rand_s{sd}" for sd in range(N_RAND)],
                    nog=[f"{abl}_nog_s{sd}" for sd in range(N_RAND)])
    if name.startswith("joint"):
        k = name[5:]
        out = dict(vm=[f"{name}_vm_s{sd}" for sd in range(N_SUB)], additive=[f"r0_{k}", f"r1_{k}"])
        if k == "5":
            out.update(add5=["add5"], pcspan=[f"pcspan5_s{sd}" for sd in range(N_SUB)], pc5=["pc5"])
        return out
    if name in ("add5", "pure5", "pc5"):
        return dict(vm=[f"{name}_vm_s{sd}" for sd in range(N_SUB)])
    if name.startswith("keep"):
        return dict(vm=[f"{name}_vm"] + ([f"{name}_vm_s1"] if int(name[4:]) in KS_SEED2 else []))
    raise ValueError(name)


def targets(names):
    return [n for n in names if n != "intact" and not ("_vm" in n or "_vt" in n or "_rand" in n or "_nog" in n or
                                                       n.startswith(("r0_", "r1_", "fshift_", "pcspan")))]


WIN = {"early": (800., 1600.), "late": (2400., 40000.)}           # training episodes (inclusive)


def window(x, win):
    lo, hi = WIN[win]
    return (x >= lo) & (x <= hi)


def effect_table(L, FE=None):
    """per target, family and window: the change in the family's gap fraction (mean over the window's checkpoints)
    of the target and of its controls, and the specific damage (control - target; > 0: the target removes more),
    each with 5-95% over the shared resamples: {target: {fam: {win: dict(...)}}}"""
    FE = family_effects(L) if FE is None else FE
    nm = L["names"]
    out = {}
    for t in targets(nm):
        out[t] = {}
        for j, fam in enumerate(FAMILIES):
            out[t][fam] = {}
            for win in WIN:
                m = window(L["x"], win)
                e = FE["dF"][m, nm.index(t), j].mean(0)
                eb = FE["dF_b"][m, nm.index(t), j].mean(0)
                d = dict(effect=e, effect_iv=iv(eb), level=FE["F_int"][m, j].mean(0))
                for kind, cs in controls(t, nm).items():
                    ci = [nm.index(c) for c in cs]
                    ce = _seedmean(FE["dF"][m][:, ci, j]).mean(0)
                    cb = _seedmean(FE["dF_b"][m][:, ci, j]).mean(0)
                    d[kind] = dict(effect=ce, effect_iv=iv(cb), specific=ce - e, specific_iv=iv(cb - eb))
                out[t][fam][win] = d
    return out


def trajectory_residual(L, FE=None, fam_y="late", xmin=640.):
    """how far a lesioned network falls off the intact network's own developmental path: residual = its family-y gap
    fraction (holding, coupling, or 'late' = their mean) minus the intact path's value at the same answers gap fraction
    (linear interpolation over the intact checkpoints >= xmin, on the running maximum of the path's answers so that the
    path is monotone; points left of the path's start take its first value); per (checkpoint, condition), with
    resamples (the path resampled with the same counts): (res (K, C), res_b (K, C, NB))"""
    FE = family_effects(L) if FE is None else FE
    if fam_y == "late":
        Y, Yb, dY, dYb = late_f(FE)
    else:
        j = FAMILIES.index(fam_y)
        Y, Yb, dY, dYb = FE["F_int"][:, j], FE["F_int_b"][:, j], FE["dF"][:, :, j], FE["dF_b"][:, :, j]
    X, Xb = FE["F_int"][:, 0], FE["F_int_b"][:, 0]
    m = L["x"] >= xmin
    lx, ly = X[:, None] + FE["dF"][:, :, 0], Y[:, None] + dY
    lxb, lyb = Xb[:, None] + FE["dF_b"][:, :, 0], Yb[:, None] + dYb
    res = ly - np.interp(lx, np.maximum.accumulate(X[m]), Y[m])
    res_b = np.full(lxb.shape, np.nan)
    for b in range(lxb.shape[-1]):
        res_b[..., b] = lyb[..., b] - np.interp(lxb[..., b], np.maximum.accumulate(Xb[m, b]), Yb[m, b])
    return res, res_b


def family_crossings():
    """the timeline's onset (c10) and half-way (c50) crossings of each family's rows (Trained RNN, td_tt_timeline):
    {fam: dict(c10 (n,), c50 (n,), c10_med, c50_med)}"""
    if "cross" in _C:
        return _C["cross"]
    import td_tt_timeline as tt
    name = {key: nm for _, nm, key in ti.EXPORT_MAP}
    rows = {r["name"]: r for r in tt.annotate(tt.load_rows()) if r["run"] == "trained"}
    out = {}
    for fam, keys in {**FAMILY_ROWS, **PANEL_ROWS}.items():                  # families and the figures' panels
        c10 = np.array([rows[name[k]]["L"][.1]["p"] for k in keys])
        c50 = np.array([rows[name[k]]["L"][.5]["p"] for k in keys])
        out[fam] = dict(c10=c10, c50=c50, c10_med=float(np.median(c10)), c50_med=float(np.median(c50)),
                        names=[name[k] for k in keys])
    c10, c50 = (np.concatenate([out["holding"][k], out["coupling"][k]]) for k in ("c10", "c50"))
    out["late"] = dict(c10=c10, c50=c50, c10_med=float(np.median(c10)), c50_med=float(np.median(c50)),
                       names=out["holding"]["names"] + out["coupling"]["names"])
    _C["cross"] = out
    return out


def row_specific(L, tgt, ctrl, win, rows):
    """per row: the specific damage of a target against a list of controls in gap units (mean over the controls'
    gap fraction minus the target's, per checkpoint), averaged over the window's checkpoints, with 5-95% over the
    shared resamples; a tuple of rows (a family subset) is first averaged over its rows per checkpoint (nanmean, as
    family_effects): {row or tuple: (value, lo, hi)}"""
    rr, f, fb = gapfrac(L)
    nm = L["names"]
    m = window(L["x"], win)
    t, ci = nm.index(tgt), [nm.index(c) for c in ctrl]
    out = {}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for row in rows:
            q = [rr.index(k) for k in (row if isinstance(row, tuple) else (row,))]
            S = np.nanmean(_seedmean(f[m][:, ci][:, :, q]) - f[m][:, t][:, q], -1)               # (k_win,)
            Sb = np.nanmean(_seedmean(fb[m][:, ci][:, :, q]) - fb[m][:, t][:, q], 1)             # (k_win, NB)
            out[row] = (float(np.nanmean(S)), *iv(np.nanmean(Sb, 0)))
    return out


def rewind_diff(L, FE, cond, row, xmin=640.):
    """'less training' outside the path plane: the lesioned network's raw value of a row minus the intact network's
    at its rewind point (the intact checkpoints >= xmin at the same answers gap fraction: running maximum, linear
    interpolation, as trajectory_residual), per checkpoint, with resamples (path and lesioned network resampled with
    the same counts): (d (K,), d_b (K, NB))"""
    nm = L["names"]
    c, ci, r = nm.index(cond), nm.index("intact"), L["rows"].index(row)
    mp = L["x"] >= xmin
    A, Ab = FE["F_int"][:, 0], FE["F_int_b"][:, 0]
    la, lab = A + FE["dF"][:, c, 0], Ab + FE["dF_b"][:, c, 0]
    y, yb = L["value"][:, c, r], L["boot"][:, c, r].astype(np.float64)
    yi, yib = L["value"][:, ci, r], L["boot"][:, ci, r].astype(np.float64)
    d = y - np.interp(la, np.maximum.accumulate(A[mp]), yi[mp])
    db = np.stack([yb[:, b] - np.interp(lab[:, b], np.maximum.accumulate(Ab[mp, b]), yib[mp, b])
                   for b in range(yb.shape[-1])], -1)
    return d, db


def late_f(FE):
    """the late families' gap fraction (mean of holding and coupling) of the intact network and its change: (F (K,),
    F_b (K, NB), dF (K, C), dF_b (K, C, NB))"""
    jh, jc = FAMILIES.index("holding"), FAMILIES.index("coupling")
    return ((FE["F_int"][:, jh] + FE["F_int"][:, jc]) / 2, (FE["F_int_b"][:, jh] + FE["F_int_b"][:, jc]) / 2,
            (FE["dF"][:, :, jh] + FE["dF"][:, :, jc]) / 2, (FE["dF_b"][:, :, jh] + FE["dF_b"][:, :, jc]) / 2)


# ═══════════════════════════════════════════════════════════════════
# Numbers (every number of the docstring) and the full effect table
# ═══════════════════════════════════════════════════════════════════

KEY = ("joint5", "joint5_xfit", "add5", "pure5", "pc5", "res_i", "res_f", "res_fi", "res_g", "res_o", "mean_i", "mean_f",
       "mean_fi", "mean_g", "mean_o", "fclamp_own10", "fclamp_fin10", "fself_own10", "fself_all", "cclamp_own10", "keep20",
       "keep100")
ONSET_KEY = (("joint5", ("vm", "additive")), ("joint5_xfit", ("additive", "vm")), ("res_g", ("vm", "rand", "nog")),
             ("res_i", ("nog",)), ("fclamp_own10", ("vm", "rand", "vt", "shift")), ("fclamp_fin10", ("vm", "rand", "shift")))
PATH_CTRL = ("joint5_vm_s0", "joint5_vm_s1", "r0_5", "r1_5", "r0_5_xfit", "pcspan5_s0", "pcspan5_s1", "res_rand_s0",
             "res_nog_s0",
             "fclamp_own10_vt", "fclamp_rand10_s0", "fshift_own10_s0", "cclamp_own10_vt", "keep100_vm")


def table(run="trained"):
    """-> OUT/<run>_effects.tsv: every condition x checkpoint x row: value, change from intact (absolute units) with
    5-95%, and the family gap-fraction changes"""
    L = load_run(run)
    FE = family_effects(L)
    nm, x = L["names"], L["x"]
    ci = nm.index("intact")
    lines = ["condition\tclass\ttraining_episodes\trow\tvalue\tchange\tchange_lo\tchange_hi"]
    for c, n in enumerate(nm):
        for k in range(len(x)):
            for r, row in enumerate(L["rows"]):
                d = L["value"][k, c, r] - L["value"][k, ci, r]
                lo, hi = iv(L["boot"][k, c, r].astype(np.float64) - L["boot"][k, ci, r].astype(np.float64))
                lines.append(f"{n}\t{L['cls'][c]}\t{x[k]:.0f}\t{row}\t{L['value'][k, c, r]:.6g}\t{d:.6g}\t{lo:.6g}"
                             f"\t{hi:.6g}")
            for j, fam in enumerate(FAMILIES):
                lo, hi = iv(FE["dF_b"][k, c, j])
                lines.append(f"{n}\t{L['cls'][c]}\t{x[k]:.0f}\tfamily_{fam}_gap\t{FE['F_int'][k, j] + FE['dF'][k, c, j]:.6g}"
                             f"\t{FE['dF'][k, c, j]:.6g}\t{lo:.6g}\t{hi:.6g}")
    fn = os.path.join(OUT, f"{run}_effects.tsv")
    open(fn, "w").write("\n".join(lines) + "\n")
    print("saved", fn, len(lines) - 1, "lines")
    return fn


def onset(x, S, Sb, level=.1):
    """the training episode after which a damage curve stays above `level` to the end (td_tt_gap.crossing,
    log-interpolated), with 5-95% over the resamples and the share of resamples that cross"""
    c = gap.crossing(x, S, level)
    cb = gap.crossing(x, Sb, level)
    lo, hi, p = gap.interval(cb)
    return c, lo, hi, p


def _g(v, lo=None, hi=None):
    """compact signed 2-decimal number [interval]"""
    def one(a):
        t = f"{a:+.2f}"
        return t.replace("+0.", "+.").replace("-0.", "-.")
    return one(v) if lo is None else f"{one(v)} [{one(lo)},{one(hi)}]"


CTRL_NAME = {"vm": "matched", "rand": "random", "additive": "r0/r1", "vt": "same total c var.", "nog": "random non-g rows",
             "shift": "random, same |df|", "add5": "r0+r1 5 dims", "pcspan": "random in top PCs", "pc5": "top-5 PCs"}


def numbers(run="trained"):
    """prints every number of the docstring (compact; the full grid: table())"""
    L = load_run(run)
    FE = family_effects(L)
    nm, x = L["names"], L["x"]
    ET = effect_table(L, FE)
    ci = nm.index("intact")
    m = window(x, "late")
    print(f"{run}: {len(x)} checkpoints ({x[0]:,.0f}-{x[-1]:,.0f} training episodes), {len(nm)} conditions; second-pass "
          f"intact = first-pass intact to {L.get('extra_intact_diff', float('nan')):.0e}")
    ck = json.load(open(os.path.join(OUT, "checks.json")))

    def n_ent(c):
        return 5028 * T * NV if c.endswith("_fwdset") else 10000 * T * NV
    for k, v in ck.items():
        cs = [c for c in v if isinstance(v[c], dict) and "max_abs" in v[c]]
        print(f"VALIDATION {k}: lean forward vs engine max |d log q| {max(v[c]['max_abs'] for c in cs):.0e}, answers "
              f"differ at {sum(v[c]['right_diff'] for c in cs)} of {sum(n_ent(c) for c in cs):,}; joint-factor trace vs ck "
              f"fac rel {v['tr_joint_rel']:.0e}, drive variance rel {v['dvar_vs_fac_rel']:.0e}, tau rel "
              f"{v['tau_ref_vs_ck_rel']:.0e}; clamp residual {v['joint5_residual']:.0e}, kept-span residual "
              f"{v['keep20_residual']:.0e}, step-1 log q {max(v['joint5_step1_logq'], v['keep20_step1_logq']):.0e}")
    fs_ = os.path.join(OUT, "checks_self.json")
    if os.path.exists(fs_):
        cs = json.load(open(fs_))
        print("VALIDATION f at own mean: " + "; ".join(f"{k} |f - own mean| {v['self_clamp_max_abs']:.0e}, no unit = intact "
                                                        f"{v['empty_vs_intact']['max_abs']:.0e}" for k, v in cs.items()))
    f3 = os.path.join(OUT, "checks_third.json")
    if os.path.exists(f3):
        c3 = json.load(open(f3))
        for k, v in c3.items():
            txt = (f"VALIDATION third pass {k}: f at given values |f - given| {v['given_max_abs']:.0e}, given = own mean "
                   f"vs 'self' max |d log q| {v['given_own_mean_vs_self']['max_abs']:.0e}; g rows in the g-free sets "
                   f"{v['nog_g_rows']}")
            if "add5_orth" in v:
                txt += ("; add5 / pure5 / pc5 / pcspan5: orthonormal to " +
                        f"{max(v[q + '_orth'] for q in ('add5', 'pure5', 'pc5', 'pcspan5_s0')):.0e}, |Q^T V| " +
                        f"{max(v[q + '_readout'] for q in ('add5', 'pure5', 'pc5', 'pcspan5_s0')):.0e}, variance vs the "
                        "engine's captured_variance (clamp basis) rel " +
                        f"{max(v[q + '_var_vs_engine_rel'] for q in ('add5', 'pure5', 'pc5', 'pcspan5_s0')):.0e}; "
                        f"clamp residual {max(v['add5_clamp_residual'], v['pure5_clamp_residual']):.0e}; pure5 "
                        f"main-effect share {v['pure5_main_var_share']:.0e} (main-effect span {v['pure5_main_rank'][0]}-"
                        f"{v['pure5_main_rank'][1]} dims); factor split vs subspace variance rel "
                        f"{v['split_vs_var_rel']:.0e}, all of Pc u vs tr(Pc C Pc) rel {v['split_all_vs_totQc_rel']:.0e}")
            print(txt)
    for tag in ("third", "fourth"):
        if f"{tag}_intact_diff" in L:
            print(f"VALIDATION {tag} pass: its intact network = the first pass's to {L[tag + '_intact_diff']:.0e}")
    print("DEFINITIONS (clean probe run of the checkpoint)")
    for xv in (800., 4000., 39996.):
        k = int(np.flatnonzero(x == xv)[0])
        D, inf = L["defs"][k], L["info"][k]

        def I(n):
            return inf[nm.index(n)]
        tau = D["tau"]
        tot = D["tot"].sum()
        dv = D["dvar"].reshape(4, H).sum(1)
        o5, f5 = set(np.argsort(-tau)[:50]), set(np.argsort(-D["tau_fin"])[:50])
        print(f"  {xv:,.0f}: tau > 5 / > 10: {(tau > 5).sum()} / {(tau > 10).sum()} units; top 5% mean tau "
              f"{I('fclamp_own5')['tau_mean']:.1f} (final set here {I('fclamp_fin5')['tau_mean']:.1f}, {len(o5 & f5)} of 50 "
              f"shared); c-matched / target c variance {I('fclamp_own5_vm')['vc'] / I('fclamp_own5')['vc']:.2f} (5%) "
              f"{I('fclamp_own10_vm')['vc'] / I('fclamp_own10')['vc']:.2f} (10%); same-total-c-variance sets "
              f"{I('fclamp_own5_vt')['n']} / {I('fclamp_own10_vt')['n']} units "
              f"({I('fclamp_own5_vt')['vc'] / I('fclamp_own5_vt')['vc_target']:.2f} / "
              f"{I('fclamp_own10_vt')['vc'] / I('fclamp_own10_vt')['vc_target']:.2f} of the variance)")
        print(f"     u variance {tot / T:.1f} per step; r0 x r1 factor {D['tr_joint'].sum() / tot:.3f} of it; 5 dims hold "
              f"{D['var_joint5'].sum() / tot:.3f} of u, {D['own_joint5'].sum() / D['tr_joint'].sum():.2f} of the factor "
              f"(3 dims {D['var_joint3'].sum() / tot:.3f}, {D['own_joint3'].sum() / D['tr_joint'].sum():.2f}); split-half "
              f"cos^2 {D['rel5'].mean():.2f}; cross-fit on held-out half {I('joint5_xfit')['var_on_other_half'] / tot:.3f}; "
              f"r0 / r1 5 dims {D['var_r0_5'].sum() / tot:.3f} / {D['var_r1_5'].sum() / tot:.3f}; random match "
              f"{np.mean(I('joint5_vm_s0')['dims']):.0f} dims")
        print("     drive variance i / f / g / o " + " / ".join(f"{v / dv.sum():.2f}" for v in dv) + "; keep 20 / 50 / "
              "100 / 200 PCs remove " + " / ".join(f"{I('keep' + str(q))['share']:.2f}" for q in (20, 50, 100, 200))
              + " of u's variance")
    print("EFFECTS, gap units, answers / holding / coupling (window means; early 800-1,600, late >= 2,400): change dF of "
          "the lesioned network, then specific damage = control f - lesioned f per control")
    for t in [t for t in KEY if t in ET]:
        for win in WIN:
            d = {fam: ET[t][fam][win] for fam in FAMILIES}
            print(f"  {t:12s} {win:5s} dF " + " / ".join(_g(d[f]['effect'], *d[f]['effect_iv']) for f in FAMILIES))
            for kind in controls(t, nm):
                print(f"  {'':12s} {'':5s} vs {CTRL_NAME[kind]:17s} " + " / ".join(
                    _g(d[f][kind]['specific'], *d[f][kind]['specific_iv']) for f in FAMILIES))
    print("  intact f, early / late: " + "; ".join(f"{fam} {ET['joint5'][fam]['early']['level']:.2f} / "
                                                   f"{ET['joint5'][fam]['late']['level']:.2f}" for fam in FAMILIES))
    print("  intact f per checkpoint (" + ", ".join(f"{v:,.0f}" for v in x) + "):")
    for j, fam in enumerate(FAMILIES):
        print(f"    {fam:8s} " + " ".join(_g(v) for v in FE["F_int"][:, j]))
    print("ABSOLUTE UNITS, late window, change from intact: goal accuracy / held per variable (corner ends) / excess angle "
          "(deg); belief change KL(intact || lesioned)")
    for t in [t for t in KEY if t in nm]:
        c = nm.index(t)
        parts = []
        for row in ("acc_goal_T", "held_two_fwd", "excess"):
            r = L["rows"].index(row)
            d = (L["value"][m, c, r] - L["value"][m, ci, r]).mean()
            b = (L["boot"][m, c, r].astype(np.float64) - L["boot"][m, ci, r].astype(np.float64)).mean(0)
            lo, hi = iv(b)
            parts.append(f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]" if row != "excess" else f"{d:+.2f} [{lo:+.2f},{hi:+.2f}]")
        print(f"  {t:12s} " + " / ".join(parts) + f"; KL {L['value'][m, c, L['rows'].index('kl_int')].mean():.3f}")
    print("SHARES, late window: share of each family's intact f removed, -mean dF / mean f_intact, answers / holding / "
          "coupling")
    for t in [t for t in KEY if t in nm]:
        c = nm.index(t)
        sh = [_g(-FE["dF"][m, c, j].mean() / FE["F_int"][m, j].mean(),
                 *iv(-FE["dF_b"][m, c, j].mean(0) / FE["F_int_b"][m, j].mean(0))) for j in range(3)]
        print(f"  {t:12s} " + " / ".join(sh))
    rr, f_, fb_ = gapfrac(L)
    for t in ("joint5", "joint5_vm_s0"):
        c = nm.index(t)
        d = (f_[m, c] - f_[m, ci]).mean(0)
        lo, hi = iv((fb_[m, c] - fb_[m, ci]).mean(0))
        print(f"PER ROW {t}, late dF: " + "; ".join(f"{rr[q]} {_g(d[q], lo[q], hi[q])}" for fam in FAMILIES
                                                     for q in [rr.index(k) for k in FAMILY_ROWS[fam]]))
    r = L["rows"].index("commit_fwd")
    print("COMMITTED to a corner at corner splits (naive .554, Bayes 0), late: intact "
          f"{L['value'][m, ci, r].mean():.3f}; " + "; ".join(f"{t} {L['value'][m, nm.index(t), r].mean():.3f}"
                                                             for t in ("joint5", "joint5_vm_s0", "r0_5", "res_fi",
                                                                       "fclamp_own10")))
    print("PATH RESIDUAL, late window: late-family f minus the intact path's at equal answers f; late / holding / "
          "coupling")
    res = {fy: trajectory_residual(L, FE, fy) for fy in ("late", "holding", "coupling")}
    for t in [t for t in KEY + PATH_CTRL if t in nm]:
        c = nm.index(t)
        print(f"  {t:16s} " + " / ".join(_g(res[fy][0][m, c].mean(), *iv(res[fy][1][m, c].mean(0))) for fy in res))
    print("REWIND: checkpoint -> the intact checkpoint with the lesioned network's answers f (log-interpolated, path >= "
          "640) (late-family f: lesioned vs path there)")
    mp = x >= 640.
    A, Fl = FE["F_int"][mp, 0], late_f(FE)[0][mp]
    Am = np.maximum.accumulate(A)
    for t in ("joint5", "joint3", "joint5_xfit", "r0_5", "joint5_vm_s0", "res_fi", "fclamp_own10", "keep100"):
        c = nm.index(t)
        parts = []
        for xv in (2400., 4000., 12000., 39996.):
            k = int(np.flatnonzero(x == xv)[0])
            a_les = FE["F_int"][k, 0] + FE["dF"][k, c, 0]
            l_les = late_f(FE)[0][k] + late_f(FE)[2][k, c]
            xe = float(np.exp(np.interp(a_les, Am, np.log(x[mp]))))
            lab = {2400.: "2.4k", 4000.: "4k", 12000.: "12k", 39996.: "40k"}[xv]
            parts.append(f"{lab} -> {xe:,.0f} ({_g(l_les)} vs {_g(np.interp(a_les, Am, Fl))})")
        print(f"  {t:12s} " + "; ".join(parts))
    cr = family_crossings()
    print("ONSET of the specific damage (stays above .1 to the end) [5-95%] (share crossing); family c10 / c50 "
          "(timeline medians [rows]): " + "; ".join(
              f"{fam} {cr[fam]['c10_med']:,.0f} [{cr[fam]['c10'].min():,.0f}-{cr[fam]['c10'].max():,.0f}] / "
              f"{cr[fam]['c50_med']:,.0f} [{cr[fam]['c50'].min():,.0f}-{cr[fam]['c50'].max():,.0f}]" for fam in FAMILIES))
    for t, kinds in ONSET_KEY:
        if t not in nm:
            continue
        for kind in [k for k in kinds if k in controls(t, nm)]:
            S, Sb = specific(L, FE, t, kind)
            parts = []
            for j in range(3):
                c, lo, hi, p = onset(x, S[:, j], Sb[:, j, :].T)
                parts.append("never" if not np.isfinite(c) else f"{c:,.0f} [{lo:,.0f}-" +
                             (f"{hi:,.0f}" if np.isfinite(hi) else "never") + f"] ({p:.0%})")
            print(f"  {t:12s} vs {CTRL_NAME[kind]:17s} " + " / ".join(parts))
    numbers_fix(L, FE)
    if run == "trained" and all(os.path.exists(ck_file("echo", E)) for E in CKS_ESN):
        numbers_esn()


def _per_ck(x, S, Sb, fmt=_g):
    """per checkpoint 'x value' and the number of checkpoints whose 5-95% lies above / below 0"""
    lo, hi = iv(Sb)
    return (" ".join(f"{v:,.0f}:{fmt(s)}" for v, s in zip(x, S)),
            int((lo > 0).sum()), int((hi < 0).sum()))


def _signs(x, Sb):
    """per checkpoint '+' / '-' / '0' (5-95% above 0 / below 0 / neither; '.' without a value) and the first checkpoint
    from which every later one is '+' (or '-')"""
    lo, hi = iv(Sb)
    sg = "".join("." if not np.isfinite(a) else "+" if a > 0 else "-" if b < 0 else "0" for a, b in zip(lo, hi))

    def since(c):
        k = len(sg)
        while k > 0 and sg[k - 1] == c:
            k -= 1
        return f"{x[k]:,.0f}" if k < len(sg) else "never"
    return sg, since("+"), since("-")


def numbers_fix(L, FE):
    """the fix round's numbers (11-02): per-row interaction effects, the answers' damage per checkpoint, the g-free
    gate control, both ablation types' shares, the |df|-matched forget-gate control, the c clamp per unit and per
    variance, the subspace controls with their factor splits, and 'less training' tested outside the path plane"""
    nm, x = L["names"], L["x"]
    has3 = "defs3" in L
    ans4 = FAMILY_ROWS["answers"][:4]
    print("FIX ROUND 11-02")
    # (1) the interaction subspace per row
    print("  PER ROW specific damage of joint5 [5-95%], early | late (answers rows; 'core 4' = their first four rows "
          "averaged per checkpoint, no corner answers)")
    ctl = [("r0/r1", ["r0_5", "r1_5"]), ("random dims", ["joint5_vm_s0", "joint5_vm_s1"])]
    if "add5" in nm:
        ctl.append(("r0+r1 5 dims", ["add5"]))
    rows = list(FAMILY_ROWS["answers"]) + [ans4]
    for lab, cs in ctl:
        e = row_specific(L, "joint5", cs, "early", rows)
        la = row_specific(L, "joint5", cs, "late", rows)
        print(f"    vs {lab:13s} " + "; ".join(f"{'core 4' if isinstance(r, tuple) else r} {_g(*e[r])} | {_g(*la[r])}"
                                                for r in rows))
    print("  PER ROW specific damage of joint5, late [5-95%] (holding and coupling rows)")
    rows = list(FAMILY_ROWS["holding"]) + list(FAMILY_ROWS["coupling"])
    for lab, cs in ctl:
        la = row_specific(L, "joint5", cs, "late", rows)
        print(f"    vs {lab:13s} " + "; ".join(f"{r} {_g(*la[r])}" for r in rows))
    for lab, cs in (("r0/r1 cross-fit", ["r0_5_xfit", "r1_5_xfit"]),):
        la = row_specific(L, "joint5_xfit", cs, "late", list(FAMILY_ROWS["answers"]) + [ans4])
        print(f"    joint5_xfit vs {lab}, late: " + "; ".join(f"{'core 4' if isinstance(r, tuple) else r} {_g(*v)}"
                                                           for r, v in la.items()))
    # (2) the answers' specific damage per checkpoint (and holding's / coupling's against random dims)
    for kind in ("vm", "additive"):
        S, Sb = specific(L, FE, "joint5", kind)
        lo, hi = iv(Sb[:, 0])
        print(f"  joint5 answers specific damage per checkpoint vs {CTRL_NAME[kind]}: " +
              "; ".join(f"{v:,.0f} {_g(s, a, b)}" for v, s, a, b in zip(x, S[:, 0], lo, hi)))
    S, Sb = specific(L, FE, "joint5", "vm")
    for j in (1, 2):
        print(f"  joint5 {FAMILIES[j]} specific damage per checkpoint vs matched: " + _per_ck(x, S[:, j], Sb[:, j])[0])
    # (3) gate drive against g-free random rows
    if "res_nog_s0" in nm:
        k5 = int(np.flatnonzero(x == 4000.)[0])
        bb = [L["info"][k5][nm.index(f"res_nog{t}_s{sd}")]["by_block"] for t in ("", "2k") for sd in range(N_RAND)]
        print("  GATES vs count-matched random rows without g (i, f, o blocks; 3 seeds; blocks i / f / g / o of the "
              "sets: " + ", ".join("/".join(str(v) for v in b) for b in bb) + ")")
        print("    per checkpoint (80 ... 39,996): + / - / 0 = the specific damage's 5-95% above 0 / below 0 / neither; "
              "'> 0 from' = the first checkpoint from which every later one is +")
        for t in ("res_i", "res_f", "res_o", "res_g", "res_fi", "mean_i", "mean_f", "mean_o", "mean_g", "mean_fi"):
            S, Sb = specific(L, FE, t, "nog")
            parts = []
            for j, fam in enumerate(FAMILIES):
                sg, sp, sn = _signs(x, Sb[:, j])
                parts.append(f"{fam} {sg} (> 0 from {sp}, < 0 from {sn})")
            print(f"    {t:8s} " + "; ".join(parts))
        for t in ("res_i", "mean_i"):
            S, Sb = specific(L, FE, t, "nog")
            for j, fam in enumerate(FAMILIES):
                print(f"    {t} vs random non-g rows, {fam}: " + _per_ck(x, S[:, j], Sb[:, j])[0])
    # (4) shares of the late gain removed, both ablation types, and their differences
    m = window(x, "late")
    print("  SHARE DIFFERENCES, late: holding - answers / coupling - answers (shares of each family's intact f "
          "removed) [5-95%]")
    for t in ("res_i", "res_f", "res_fi", "res_o", "mean_i", "mean_f", "mean_fi", "mean_o", "fclamp_own10",
              "fclamp_fin10", "joint5"):
        c = nm.index(t)
        sh = [-FE["dF"][m, c, j].mean() / FE["F_int"][m, j].mean() for j in range(3)]
        shb = [-FE["dF_b"][m, c, j].mean(0) / FE["F_int_b"][m, j].mean(0) for j in range(3)]
        print(f"    {t:12s} {_g(sh[1] - sh[0], *iv(shb[1] - shb[0]))} / {_g(sh[2] - sh[0], *iv(shb[2] - shb[0]))}")
    # (5) long-tau forget clamp: perturbation sizes and the |df|-matched control
    if has3:
        D3 = L["defs3"]
        print("  FORGET-GATE PERTURBATION, clean run: mean over probes, steps 1-30 and units of |f - assigned value|")
        for xv in (800., 1600., 4000., 39996.):
            k = int(np.flatnonzero(x == xv)[0])
            d = D3[k]

            def mf(prefix):
                ks = [q for q in d if q.startswith("dfabs_" + prefix)]
                return np.mean([d[q].mean() for q in ks]) if ks else np.nan
            print(f"    {xv:,.0f}: long-tau top 10% to median {d['dfabs_fclamp_own10'].mean():.2f}; "
                  f"random units to median {mf('fclamp_rand10_s'):.2f}; c-matched units to median "
                  f"{d['dfabs_fclamp_own10_vm'].mean():.2f}; same-total-c-variance units to median "
                  f"{d['dfabs_fclamp_own10_vt'].mean():.2f}; random units shifted by the targets' mean shift "
                  f"{mf('fshift_own10_s'):.2f} (shift {np.nanmean(d['delta_fshift_own10_s0'][1:]):.2f}); long-tau "
                  f"at own mean {d['dfabs_fself_own10'].mean():.3f}, random at own mean {mf('fself_rand10_s'):.3f}; "
                  f"final top 10% to median {d['dfabs_fclamp_fin10'].mean():.2f}, its shifted control "
                  f"{mf('fshift_fin10_s'):.2f}")
        parts = []
        for xv in (800., 1600., 4000., 39996.):
            k = int(np.flatnonzero(x == xv)[0])
            tau, inf = L["defs"][k]["tau"], L["info"][k]
            tg = np.asarray(inf[nm.index("fclamp_own10")]["units"])
            ru = [np.asarray(inf[nm.index(f"fshift_own10_s{sd}")]["units"]) for sd in range(N_RAND)]
            dl = float(np.nanmean(D3[k]["delta_fshift_own10_s0"][2:]))
            fr = [1. - 1. / tau[u] + dl for u in ru]
            parts.append(f"{xv:,.0f}: top 10% {tau[tg].mean():.1f} -> {1. / (1. - L['defs'][k]['f_med'][2:].mean()):.2f}; "
                         f"random units {np.mean([tau[u].mean() for u in ru]):.2f} -> median "
                         f"{np.median(np.concatenate([1. / (1. - np.clip(v, 0., F_CLIP)) for v in fr])):.1f} "
                         f"(clipped at f = .99: {np.mean(np.concatenate(fr) > F_CLIP):.0%})")
        print("  TIME CONSTANTS tau = 1 / (1 - mean f) (steps 2-30) under the median clamp (targets) and the shift (its "
              "|df|-matched random units, 3 seeds): " + "; ".join(parts))
        for t in ("fclamp_own10", "fclamp_own5", "fclamp_fin10"):
            S, Sb = specific(L, FE, t, "shift")
            for j, fam in enumerate(FAMILIES):
                txt, npos, nneg = _per_ck(x, S[:, j], Sb[:, j])
                sg, sp, sn = _signs(x, Sb[:, j])
                print(f"    {t} vs random, same |df|, {fam} {sg} (> 0 from {sp}, < 0 from {sn}): {txt}")
    # (6) the c clamp: per unit and per unit of c variance
    print("  C CLAMP of the top 10%: controls' units and c variance (sum over steps of var_s(c)) relative to the "
          "targets', per checkpoint (training episodes: same-total-c-variance units n / variance ratio; c-matched "
          "units variance ratio; count-matched random units variance ratio)")
    parts = []
    for k, xv in enumerate(x):
        inf = L["info"][k]

        def I(n_):
            return inf[nm.index(n_)]
        vt, tg = I("cclamp_own10_vt"), I("cclamp_own10")
        rnd = np.mean([I(f"cclamp_rand10_s{sd}")["vc"] for sd in range(N_RAND)])
        parts.append(f"{xv:,.0f}: {vt['n']} / {vt['vc'] / vt['vc_target']:.2f}; {I('cclamp_own10_vm')['vc'] / tg['vc']:.2f}; "
                     f"{rnd / tg['vc']:.2f}")
    print("    " + "; ".join(parts))
    # (7) subspaces: variance, factor split
    if has3 and "add5" in nm:
        D3 = L["defs3"]
        print("  SUBSPACES (third pass): share of u's clean variance (pooled over steps) and the factor split of the "
              "variance inside (" + " / ".join(SPLIT) + ")")
        for xv in (800., 1600., 4000., 39996.):
            k = int(np.flatnonzero(x == xv)[0])
            d = D3[k]
            tot = d["tot3"].sum()
            print(f"    {xv:,.0f}: " + "; ".join(f"{s_} {d['var_' + s_].sum() / tot:.3f}" for s_ in SUB3) +
                  f"; top-K span K {np.mean(d['kspan_s0']):.0f} / {np.mean(d['kspan_s1']):.0f}; random dims matched "
                  f"to add5 / pure5 / pc5 {np.mean(L['info'][k][nm.index('add5_vm_s0')]['dims']):.0f} / "
                  f"{np.mean(L['info'][k][nm.index('pure5_vm_s0')]['dims']):.0f} / "
                  f"{np.mean(L['info'][k][nm.index('pc5_vm_s0')]['dims']):.0f}; main-effect (r0 + r1 cell-mean) "
                  f"variance in pure5 {d['mainvar_pure5'].sum() / tot:.1e} of u")
            for s_ in ("joint5", "r0_5", "add5", "pure5", "pc5", "pcspan5_s0", "all"):
                sp = d["split_" + s_].sum(0)
                print(f"       split {s_:10s} " + " / ".join(f"{v / sp.sum():.2f}" for v in sp) +
                      f" (r0 + r1 {(sp[1] + sp[2]) / sp.sum():.2f})")
    # (8) 'less training' outside the plane: rewind signatures
    print("  REWIND SIGNATURES, late: lesioned value minus the intact network's at the rewind point (same answers f), "
          "raw units, late-window mean [5-95%]: committed to a corner (commit_fwd) / held per variable at corner ends "
          "(held_two_fwd) / corner split held jointly (held_joint_fwd) / corner answer = truth (corner_truth_fwd)")
    conds = [c for c in ("joint5", "joint5_xfit", "joint5_vm_s0", "joint5_vm_s1", "r0_5", "r1_5", "add5", "pure5", "pc5",
                         "pcspan5_s0", "pcspan5_s1", "keep100", "keep100_vm", "keep100_vm_s1", "res_i", "res_fi",
                         "fclamp_own10", "fshift_own10_s0") if c in nm]
    sig = ("commit_fwd", "held_two_fwd", "held_joint_fwd", "corner_truth_fwd")
    for c in conds:
        parts = []
        for row in sig:
            d, db = rewind_diff(L, FE, c, row)
            parts.append(_g(d[m].mean(), *iv(db[m].mean(0))))
        print(f"    {c:16s} " + " / ".join(parts))
    rc, ri = L["rows"].index("commit_fwd"), L["rows"].index("held_joint_fwd")
    cj = nm.index("joint5")
    d, db = rewind_diff(L, FE, "joint5", "commit_fwd")
    mp = x >= 640.
    A = np.maximum.accumulate(FE["F_int"][mp, 0])
    print("    joint5 per late checkpoint (lesioned commit / held jointly; intact at the rewind point commit / held "
          "jointly): " + "; ".join(
              f"{xv:,.0f}: {L['value'][k, cj, rc]:.2f} / {L['value'][k, cj, ri]:.2f}; "
              f"{np.interp(FE['F_int'][k, 0] + FE['dF'][k, cj, 0], A, L['value'][mp, nm.index('intact'), rc]):.2f} / "
              f"{np.interp(FE['F_int'][k, 0] + FE['dF'][k, cj, 0], A, L['value'][mp, nm.index('intact'), ri]):.2f}"
              for k, xv in enumerate(x) if m[k]))
    print("    intact commit_fwd per checkpoint: " + " ".join(f"{xv:,.0f}:{v:.2f}" for xv, v in
                                                            zip(x, L["value"][:, nm.index("intact"), rc])))
    # (8b) seed spread of the random controls: each seed's own window-mean specific damage
    print("  SEED SPREAD: specific damage against each seed of a random control alone, window means, answers / holding "
          "/ coupling: [min, max] over the seeds")
    for t, kind in (("joint5", "vm"), ("joint5", "pcspan"), ("pure5", "vm"), ("add5", "vm"), ("pc5", "vm"),
                    ("res_i", "nog"), ("res_f", "nog"), ("res_g", "nog"), ("res_o", "nog"), ("res_fi", "nog"),
                    ("mean_i", "nog"), ("res_i", "rand"), ("res_fi", "rand"), ("fclamp_own10", "shift"),
                    ("fclamp_own5", "shift"), ("fclamp_fin10", "shift"), ("fclamp_own10", "rand"),
                    ("cclamp_own10", "rand"), ("keep100", "vm"), ("keep20", "vm")):
        cs = controls(t, nm).get(kind)
        if not cs or t not in nm:
            continue
        parts = []
        for win in WIN:
            mw = window(x, win)
            per = np.array([[FE["dF"][mw, nm.index(c), j].mean() - FE["dF"][mw, nm.index(t), j].mean()
                             for j in range(3)] for c in cs])                                        # (seeds, fam)
            parts.append(f"{win} " + " / ".join(f"[{_g(per[:, j].min())},{_g(per[:, j].max())}]" for j in range(3)))
        print(f"    {t:12s} vs {CTRL_NAME[kind]:17s} ({len(cs)} seeds) " + "; ".join(parts))
    # (9) path residual at matched lesioned answers
    res, res_b = trajectory_residual(L, FE, "late")
    bins = ((.45, .6), (.6, .7), (.7, .8), (.8, .9))
    la = FE["F_int"][:, 0][:, None] + FE["dF"][:, :, 0]
    print("  PATH RESIDUAL BY THE LESIONED NETWORK'S ANSWERS f (checkpoints >= 640 pooled; bins " +
          ", ".join(f"[{a:.2f},{b:.2f})" for a, b in bins) + "): mean [5-95%] (n checkpoints)")
    for c in [c for c in ("joint5", "joint5_xfit", "joint5_vm_s0", "joint5_vm_s1", "r0_5", "r1_5", "add5", "pure5",
                          "pcspan5_s0", "pcspan5_s1", "keep100", "keep100_vm", "keep100_vm_s1", "res_i", "fclamp_own10",
                          "fshift_own10_s0") if c in nm]:
        k = nm.index(c)
        parts = []
        for a, b in bins:
            sel = mp & (la[:, k] >= a) & (la[:, k] < b)
            parts.append(f"{_g(res[sel, k].mean(), *iv(res_b[sel, k].mean(0)))} ({sel.sum()})" if sel.any() else "-")
        print(f"    {c:16s} " + "  ".join(parts) + f"; late-window answers f {la[m, k].mean():.2f}")
    numbers_final(L, FE)


def _win(S, Sb, win, x):
    """window mean of a per-checkpoint curve (NaN-aware) and its 5-95% over the resamples"""
    import warnings
    m = window(x, win)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(S[m])), *iv(np.nanmean(Sb[m], 0))


def stage_matched(L, FE, ref="joint5", tol=.05, nmin=3, xfit=None):
    """the path test at matched training stage: for every condition, its late-family path residual minus the reference
    lesion's at the same late checkpoint, over the late checkpoints where the two lesioned networks' answers gap
    fractions differ by <= tol (at least nmin of them); mean [5-95%] over the shared resamples; sorted. xfit = (x (K, C),
    x_b (K, C, NB)) replaces the answers gap fraction as the matching and path axis (robustness: core answers)"""
    nm, x = L["names"], L["x"]
    if xfit is None:
        res, res_b = trajectory_residual(L, FE, "late")
        la = FE["F_int"][:, 0][:, None] + FE["dF"][:, :, 0]
    else:
        la, lab = xfit
        Y, Yb, dY, dYb = late_f(FE)
        mp = x >= 640.
        ci = nm.index("intact")
        res = Y[:, None] + dY - np.interp(la, np.maximum.accumulate(la[mp, ci]), Y[mp])
        res_b = np.stack([Yb[:, None, b] + dYb[..., b] - np.interp(lab[..., b], np.maximum.accumulate(lab[mp, ci, b]),
                                                                   Yb[mp, b]) for b in range(lab.shape[-1])], -1)
    j, late = nm.index(ref), window(x, "late")
    out = []
    for c, n_ in enumerate(nm):
        if n_ in ("intact", ref):
            continue
        sel = late & (np.abs(la[:, c] - la[:, j]) <= tol) & np.isfinite(res[:, c])
        if sel.sum() >= nmin:
            out.append(((res[sel, c] - res[sel, j]).mean(), n_, int(sel.sum()),
                        *iv((res_b[sel, c] - res_b[sel, j]).mean(0))))
    return sorted(out), res


def numbers_final(L, FE):
    """the final round's numbers (11-02, after the last check): every variance range over the checkpoints its claim
    uses; the figures' panels (core answers = the four answers rows without corner answers; corner answers) for the
    interaction subspace and the gate drive; the path test at matched training stage"""
    nm, x = L["names"], L["x"]
    PE = family_effects(L, PANEL_ROWS)
    print("FINAL ROUND 11-02 (after the last check)")
    # (1) variance ranges over each claim's checkpoints
    D = L["defs"]
    sets = (("all 20", np.ones(len(x), bool)), ("80-640", x <= 640.), ("14 window checkpoints", window(x, "early") |
                                                                        window(x, "late")),
            ("early", window(x, "early")), ("late", window(x, "late")))
    q = dict(u=np.array([d["var_joint5"].sum() / d["tot"].sum() for d in D]),
             factor=np.array([d["own_joint5"].sum() / d["tr_joint"].sum() for d in D]),
             fac_u=np.array([d["tr_joint"].sum() / d["tot"].sum() for d in D]),
             cos2=np.array([d["rel5"].mean() for d in D]))
    print("  VARIANCE of the interaction subspace (5 dims) by the checkpoints a claim uses: share of u's clean variance; "
          "share of the r0 x r1 factor; the factor's share of u; split-half cos^2")
    print("    " + "; ".join(f"{lab}: " + " / ".join(f"{v[mm].min():.3f}-{v[mm].max():.3f}" for v in q.values())
                             for lab, mm in sets))
    print("    per checkpoint, share of u: " + " ".join(f"{a:,.0f}:{b:.3f}" for a, b in zip(x, q["u"])))
    if "defs3" in L:
        k3 = [k for k in range(len(x)) if "tot3" in L["defs3"][k]]
        print("    third-pass subspaces over their " + f"{len(k3)} checkpoints ({x[k3[0]]:,.0f}-{x[k3[-1]]:,.0f}), "
              "share of u: " + "; ".join(
                  f"{s_} " + "{:.3f}-{:.3f}".format(*(lambda v: (min(v), max(v)))(
                      [L["defs3"][k]["var_" + s_].sum() / L["defs3"][k]["tot3"].sum() for k in k3]))
                  for s_ in ("joint5", "add5", "pure5", "pc5", "pcspan5_s0", "pcspan5_s1")))
        rat = [L["defs3"][k]["var_add5"].sum() / L["defs3"][k]["var_joint5"].sum() for k in k3]
        print(f"    add5's variance / joint5's: {min(rat):.2f}-{max(rat):.2f} (" +
              " ".join(f"{x[k]:,.0f}:{v:.2f}" for k, v in zip(k3, rat)) + ")")
        ks = [np.mean(L["defs3"][k][f"kspan_s{sd}"]) for k in k3 for sd in range(N_SUB)]
        print(f"    top-K span of pcspan5 (mean K over steps, both seeds): {min(ks):.0f}-{max(ks):.0f}")
    for kk in (20, 100):
        sh = np.array([L["info"][k][nm.index(f"keep{kk}")]["share"] for k in range(len(x))])
        print(f"    keep {kk} PCs removes of u's variance: late {sh[window(x, 'late')].min():.2f}-"
              f"{sh[window(x, 'late')].max():.2f}, early {sh[window(x, 'early')].min():.2f}-"
              f"{sh[window(x, 'early')].max():.2f}")
    # (2) the interaction subspace per panel
    print("  PANELS (core answers = acc_goal_T, acc_both_T, far_T, merged_T; corner answers = corner_truth_fwd; holding; "
          "coupling): specific damage [5-95%], early | late, and per checkpoint (+ / - / 0: 5-95% above / below 0 / "
          "neither; '.': no control there)")
    for t, kind, lab in (("joint5", "vm", "joint5 vs random dims"), ("joint5", "additive", "joint5 vs r0/r1"),
                         ("joint5", "add5", "joint5 vs r0+r1 5 dims"), ("joint5", "pcspan", "joint5 vs random in top PCs"),
                         ("pure5", "vm", "pure5 vs its random dims"), ("joint5_xfit", "additive", "joint5_xfit vs r0/r1")):
        if t not in nm or kind not in controls(t, nm):
            continue
        S, Sb = specific(L, PE, t, kind)
        for j, p in enumerate(PANELS):
            sg, sp, sn = _signs(x, Sb[:, j])
            print(f"    {lab:27s} {p:14s} {_g(*_win(S[:, j], Sb[:, j], 'early', x))} | "
                  f"{_g(*_win(S[:, j], Sb[:, j], 'late', x))} {sg} (> 0 from {sp}, < 0 from {sn})")
    for kind, lab in (("vm", "random dims"), ("additive", "r0/r1"), ("add5", "r0+r1 5 dims")):
        S, Sb = specific(L, PE, "joint5", kind)
        for j in (0, 1):
            lo, hi = iv(Sb[:, j])
            print(f"    joint5 {PANELS[j]} per checkpoint vs {lab}: " + "; ".join(
                f"{v:,.0f} " + (_g(s, a, b) if np.isfinite(s) else ".") for v, s, a, b in zip(x, S[:, j], lo, hi)))
    m = window(x, "late")
    lv = []
    for j, p in enumerate(PANELS):
        cur = [PE["F_int"][m, j].mean(), (PE["F_int"][m, j] + PE["dF"][m, nm.index("joint5"), j]).mean()]
        for kind in ("add5", "additive", "vm"):
            cn = [nm.index(c) for c in controls("joint5", nm)[kind]]
            cur.append((PE["F_int"][m, j] + _seedmean(PE["dF"][m][:, cn, j])).mean())
        lv.append(f"{p} " + " / ".join(f"{v:.2f}" for v in cur))
    print("    late gap fraction per panel (intact / joint5 / r0+r1 5 dims / r0, r1 averaged / random dims): " +
          "; ".join(lv))
    # (3) the gate drive per panel against g-free random rows
    if "res_nog_s0" in nm:
        print("  GATES per panel vs count-matched random rows without g: specific damage [5-95%], early | late, per "
              "checkpoint signs")
        for t in ("res_g", "res_o", "res_i", "res_f", "res_fi", "mean_g", "mean_o", "mean_i", "mean_f", "mean_fi"):
            S, Sb = specific(L, PE, t, "nog")
            for j, p in enumerate(PANELS):
                sg, sp, sn = _signs(x, Sb[:, j])
                print(f"    {t:8s} {p:14s} {_g(*_win(S[:, j], Sb[:, j], 'early', x))} | "
                      f"{_g(*_win(S[:, j], Sb[:, j], 'late', x))} {sg} (> 0 from {sp}, < 0 from {sn})")
        m = window(x, "late")
        cn = [nm.index(c) for c in controls("res_i", nm)["nog"]]
        parts = []
        for j, p in enumerate(PANELS):
            lv = [PE["F_int"][m, j].mean()] + [(PE["F_int"][m, j] + PE["dF"][m, nm.index("res_" + G), j]).mean()
                                               for G in ("g", "o", "i", "f")]
            lv.append((PE["F_int"][m, j] + _seedmean(PE["dF"][m][:, cn, j])).mean())
            parts.append(f"{p} {lv[0]:.2f}; " + " / ".join(f"{v:.2f}" for v in lv[1:5]) + f"; {lv[5]:.2f}")
        print("    late gap fraction per panel (intact; g / o / i / f resampled; random non-g rows, 3 seeds): " +
              "; ".join(parts))
    # (4) the path test at matched training stage
    print("  PATH TEST AT MATCHED TRAINING STAGE: a condition's late path residual minus joint5's at the same late "
          "checkpoint, over the late checkpoints where their answers f differ by <= tol (n of 9) [5-95%]")
    for tol, nmin, lim in ((.05, 3, None), (.10, 1, .10)):
        out, res = stage_matched(L, FE, tol=tol, nmin=nmin)
        sel = [o for o in out if lim is None or o[0] <= lim]
        print(f"    tol {tol:.2f}, n >= {nmin}: {len(out)} conditions" + (f"; those within +{lim:.2f}" if lim else "") +
              ": " + "; ".join(f"{n_} ({k}) {_g(d, lo, hi)}" for d, n_, k, lo, hi in sel))
    j = nm.index("joint5")
    print("    joint5's path residual per checkpoint from 640: " + " ".join(f"{a:,.0f}:{_g(b)}" for a, b in
                                                                         zip(x, res[:, j]) if a >= 640.))
    la = FE["F_int"][:, 0][:, None] + FE["dF"][:, :, 0]
    print("    the fix round's pooled bin (answers f in [.45,.60), checkpoints >= 640) held: " + "; ".join(
        f"{c} " + ", ".join(f"{v:,.0f}" for v in x[(x >= 640.) & (la[:, nm.index(c)] >= .45) & (la[:, nm.index(c)] < .6)])
        for c in ("joint5", "joint5_vm_s0", "joint5_vm_s1", "r0_5", "r1_5", "add5", "pure5")))
    ci = nm.index("intact")
    rr, f, fb = gapfrac(L)
    qa = [rr.index(k) for k in PANEL_ROWS["core answers"]]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ca, cab = np.nanmean(f[:, :, qa], -1), np.nanmean(fb[:, :, qa], 2)
    out, resc = stage_matched(L, FE, tol=.05, nmin=3, xfit=(ca, cab))
    print("    robustness, matched and drawn on the core answers' f instead (tol .05, n >= 3): joint5's late residual "
          f"{_g(resc[window(x, 'late'), j].mean())}; " + "; ".join(f"{n_} ({k}) {_g(d, lo, hi)}"
                                                                    for d, n_, k, lo, hi in out[:6]) + f"; ... "
          f"({len(out)} conditions, the largest {_g(out[-1][0])})")
    r = L["rows"].index("commit_fwd")
    mp = x >= 640.
    for cond in ("joint5", "pure5", "add5"):
        c = nm.index(cond)
        y, yb = L["value"][:, c, r], L["boot"][:, c, r].astype(np.float64)
        yi, yib = L["value"][:, ci, r], L["boot"][:, ci, r].astype(np.float64)
        d = y - np.interp(ca[:, c], np.maximum.accumulate(ca[mp, ci]), yi[mp])
        db = np.stack([yb[:, b] - np.interp(cab[:, c, b], np.maximum.accumulate(cab[mp, ci, b]), yib[mp, b])
                       for b in range(yb.shape[-1])], -1)
        m = window(x, "late")
        print(f"    robustness, corner commitment at the core-answers rewind point, late: {cond} "
              f"{_g(np.nanmean(d[m]), *iv(np.nanmean(db[m], 0)))}")


def numbers_esn():
    """the ESN's checkpoints: answers family change (gap units) and specific damage per checkpoint"""
    L = load_run("echo")
    FE = family_effects(L)
    nm, x = L["names"], L["x"]
    print(f"ESN ({', '.join(f'{v:,.0f}' for v in x)}): intact f answers " + " / ".join(_g(v) for v in FE["F_int"][:, 0])
          + "; holding " + " / ".join(_g(v) for v in FE["F_int"][:, 1]) + "; coupling "
          + " / ".join(_g(v) for v in FE["F_int"][:, 2]))
    for t, kind in (("joint5", "vm"), ("joint5", "additive"), ("joint5_xfit", "additive"), ("res_g", "rand"),
                    ("res_g", "nog"), ("res_i", "nog"), ("res_f", "nog"), ("res_o", "nog"), ("res_fi", "rand"),
                    ("res_fi", "nog"), ("fclamp_own10", "rand"), ("fclamp_own10", "shift"), ("fself_own10", "rand"),
                    ("cclamp_own10", "rand"), ("keep100", "vm")):
        if kind not in controls(t, nm):
            continue
        c = nm.index(t)
        S, Sb = specific(L, FE, t, kind)
        lo, hi = iv(Sb[:, 0])
        print(f"  {t:12s} answers dF " + " / ".join(_g(v) for v in FE["dF"][:, c, 0]) + f"; vs {CTRL_NAME[kind]} "
              + " / ".join(_g(v, a, b) for v, a, b in zip(S[:, 0], lo, hi)))
    if "defs3" in L:
        print("  ESN forget-gate perturbation |f - assigned| per unit and step (top 10% to median / random units shifted "
              "by the same mean / random units to median): " + "; ".join(
                  f"{d['dfabs_fclamp_own10'].mean():.3f} / "
                  f"{np.mean([d[f'dfabs_fshift_own10_s{sd}'].mean() for sd in range(N_RAND)]):.3f} / "
                  f"{np.mean([d[f'dfabs_fclamp_rand10_s{sd}'].mean() for sd in range(N_RAND)]):.3f}" for d in L["defs3"]))
    D = L["defs"]
    print("  ESN u variance per step " + " / ".join(f"{d['tot'].sum() / T:.1f}" for d in D) + "; r0 x r1 factor share "
          + " / ".join(f"{d['tr_joint'].sum() / d['tot'].sum():.4f}" for d in D) + "; max tau "
          + " / ".join(f"{d['tau'].max():.2f}" for d in D))


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

C_INT, C_TGT, C_V1, C_V2 = "#1f77b4", "#d62728", "#9467bd", "#2ca02c"
C_NAIVE, C_MARK = "0.55", "#8c510a"
C_BLOCK = {"g": "#17becf", "o": "#bcbd22", "i": "#8c564b", "f": "#e377c2"}
PANEL_TITLE = {"core answers": "Core answers at T, 4 rows", "corner answers": "Corner answers at T, 1 row",
               "holding": "Holding a split, 5 rows", "coupling": "Variables moving together, 6 rows"}
F_FORMULA = r"$f = \langle\frac{y - y_{\mathrm{naive}}}{y_{\mathrm{Bayes}} - y_{\mathrm{naive}}}\rangle_{\mathrm{rows}}$"
F_CORNER = r"$f$ of $y = P(\hat{a}_T = r\;|\;r$ in a corner$)$"
MAIN_ROWS = (   # (row title, [(target, label, colour)], [(control kind, label, colour, style)])
    ("5 latent-interaction dims\nof u clamped at every step",
     [("joint5", "interaction dims", C_TGT)],
     [("add5", "both latents' main-effect\ndims (5 dims, 1.1-2.0x\nthe variance)", C_V2, "-."),
      ("vm", "random dims,\nsame variance", C_V1, "--")]),
    ("One gate block's\nrecurrent drive resampled\n(1,000 of 4,000 rows)",
     [("res_g", "cell input g", C_BLOCK["g"]), ("res_o", "output gate o", C_BLOCK["o"]),
      ("res_i", "input gate i", C_BLOCK["i"]), ("res_f", "forget gate f", C_BLOCK["f"])],
     [("nog", "1,000 random rows\nof i, f and o", C_V1, "--")]),
)


def _xaxis(ax, FL, label=True):
    ax.set_xscale("log")
    ax.set_xlim(60, 6e4)
    ax.set_xticks([1e2, 1e3, 1e4])
    ax.set_xticklabels(["100", "1k", "10k"])
    ax.minorticks_off()
    if label:
        ax.set_xlabel("training episodes", fontsize=FL)


def fig_main(run="trained", out=None):
    """-> figure_bin/td_tt_lesions.png: the two lesions whose results survived the checks, through training: the gap
    closed f of the intact network, the lesioned network(s) and the matched controls per panel (core answers, corner
    answers, holding, coupling; PANEL_ROWS). Row 1: 5 latent-interaction dims clamped, against both latents'
    main-effect dims (add5; window checkpoints) and random dims of the same variance. Row 2: one gate block's recurrent
    drive resampled, against 1,000 random drive rows without g rows"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    FL, FT, FN = fs["label"], 11., 9.5
    L = load_run(run)
    PE = family_effects(L, PANEL_ROWS)
    nm, x = L["names"], L["x"]
    mx = x >= 320.                                     # drawn from 320 training episodes (80, 160: far below naive)
    nrow, ncol = len(MAIN_ROWS), len(PANELS)
    curves = {}                                        # (row, panel) -> [(y, y_b, colour, style, lw, z)]
    for i, (lab, tg, ctrls) in enumerate(MAIN_ROWS):
        for j in range(ncol):
            F, Fb = PE["F_int"][:, j], PE["F_int_b"][:, j]
            cs = [(F, Fb, C_INT, "-", 1.9, 6)]
            for kind, clab, col, ls in ctrls:
                cix = [nm.index(c) for c in controls(tg[0][0], nm)[kind]]
                cs.append((F + _seedmean(PE["dF"][:, cix, j]), Fb + _seedmean(PE["dF_b"][:, cix, j]), col, ls, 1.4, 4))
            for t_, tlab, tcol in tg:
                c = nm.index(t_)
                cs.append((F + PE["dF"][:, c, j], Fb + PE["dF_b"][:, c, j], tcol, "-", 1.7, 7))
            curves[i, j] = cs
    ylims = []                                         # one y range per panel column, from every curve and band drawn
    for j in range(ncol):
        lo_, hi_ = [], []
        for i in range(nrow):
            for y, yb, *_ in curves[i, j]:
                a, b = iv(yb)
                lo_.append(np.nanmin(np.r_[y[mx], a[mx]]))
                hi_.append(np.nanmax(np.r_[y[mx], b[mx]]))
        ylims.append((np.floor((min(lo_) - .03) * 10) / 10, max(1.1, np.ceil((max(hi_) + .03) * 10) / 10)))
    PW, PH, LAB, GX, GY, TOP, BOT, RM = 2.9, 2.0, 2.85, .45, .95, 2.3, 1.55, .15          # inches
    W = LAB + ncol * PW + (ncol - 1) * GX + RM
    Hh = TOP + nrow * PH + (nrow - 1) * GY + BOT
    fig = plt.figure(figsize=(W, Hh))
    for i, (lab, tg, ctrls) in enumerate(MAIN_ROWS):
        y0 = Hh - TOP - (i + 1) * PH - i * GY
        for j, p in enumerate(PANELS):
            ax = fig.add_axes([(LAB + j * (PW + GX)) / W, y0 / Hh, PW / W, PH / Hh])
            ax.axhline(0, color=C_NAIVE, lw=1.1, zorder=1)
            ax.axhline(1, color="k", lw=1.1, ls=(0, (4, 2)), zorder=1)
            for y, yb, col, ls, lw, z in curves[i, j]:
                ok = mx & np.isfinite(y)
                a, b = iv(yb)
                part = not ok[mx].all()                    # a control at the window checkpoints only: markers
                ax.fill_between(x[ok], a[ok], b[ok], color=col, alpha=.2 if z >= 6 else .13, lw=0, zorder=z - 1)
                ax.plot(x[ok], y[ok], color=col, ls=ls, lw=lw, zorder=z, marker="o" if part else None, ms=3.)
            ax.set_xscale("log")
            ax.set_xlim(250, 5.5e4)
            ax.set_xticks([300, 1e3, 3e3, 1e4, 3e4])
            ax.set_xticklabels(["300", "1k", "3k", "10k", "30k"])
            ax.minorticks_off()
            ax.set_ylim(*ylims[j])
            if i == nrow - 1:
                ax.set_xlabel("training episodes", fontsize=FL)
            if i == 0:
                ax.set_title(PANEL_TITLE[p] + "\n" + (F_CORNER if p == "corner answers" else F_FORMULA), fontsize=FT,
                             linespacing=1.5)
            if j == 0:
                ax.set_ylabel("gap closed $f$", fontsize=FL)
        yc = Hh - TOP - i * (PH + GY)                      # left margin: row title and its legend
        fig.text(.15 / W, yc / Hh, lab, fontsize=FL + 1, ha="left", va="top", linespacing=1.25, fontweight="bold")
        hs = [Line2D([], [], color=tcol, lw=2., label=tlab) for t_, tlab, tcol in tg] + \
             [Line2D([], [], color=col, ls=ls, lw=1.5, label=clab, marker="o" if kind == "add5" else None, ms=3.)
              for kind, clab, col, ls in ctrls]
        fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(.1 / W, (yc - .25 - .2 * lab.count("\n")) / Hh),
                   fontsize=FN, frameon=False, handlelength=2.3, borderaxespad=0., labelspacing=.5)
    hs = [Line2D([], [], color=C_INT, lw=1.9, label="intact Trained RNN"),
          Line2D([], [], color=C_NAIVE, lw=1.1, label="naive ($f$ = 0)"),
          Line2D([], [], color="k", lw=1.1, ls=(0, (4, 2)), label="Bayes ($f$ = 1)")]
    xm = (LAB + (ncol * PW + (ncol - 1) * GX) / 2) / W
    fig.legend(handles=hs, loc="upper center", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(xm, 1 - 1.3 / Hh))
    fig.text(.5, 1 - .1 / Hh,
             "Top: 5 latent-interaction dims are needed more than random dims of the same variance in every panel, "
             "and more than\nboth latents' main-effect dims for corner answers and holding (coupling: barely). The core "
             "answers lean on them more than on\nrandom dims from the earliest checkpoints, but from 1,600 on the "
             "main-effect dims (more variance) remove more of the core answers.\nBottom: late, of the recurrent gate "
             "drive, g matters most and f least (less than random rows); o is second except for\ncorner answers, "
             "where i matters as much as g.",
             ha="center", va="top", fontsize=11.5, linespacing=1.35)
    fig.text(xm, .1 / Hh,
             f"Trained RNN, {len(x)} checkpoints ({x[0]:,.0f}-{x[-1]:,.0f} training episodes; drawn from 320). Every "
             "point: 10,000 probe episodes (50 test environments x 100 cells x 2 streams)\n+ 5,028 forward-set episodes "
             "of the same environments (corner answers: its 2,400 unselected corner-cell episodes). "
             r"$f = (y - y_{\mathrm{naive}})\,/\,(y_{\mathrm{Bayes}} - y_{\mathrm{naive}})$ per row, averaged over the "
             "panel's rows\n(core answers: goal right, both right, far errors, distant pairs merged). Random controls: "
             "mean over 2 (dims) or 3 (rows) seeds. Bands: 5-95% over 1,000 environment resamples.\nBoth latents' "
             "main-effect dims: only at 800-1,600 and " + r"$\geq$" + " 2,400 training episodes (markers).",
             ha="center", va="bottom", fontsize=FN, linespacing=1.45)
    out = fig_path("td_tt_lesions.png") if out is None else out
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)
    return out


MAP_GROUPS = (   # (group label, [(row label, target, control kind)])
    ("5 latent-interaction\ndims clamped", [("vs random dims, same variance", "joint5", "vm"),
                                             ("vs one latent's main-effect dims", "joint5", "additive"),
                                             ("vs both latents' main-effect dims", "joint5", "add5"),
                                             ("main-effect-free part vs random dims", "pure5", "vm")]),
    ("One gate block's\nrecurrent drive\nresampled, vs 1,000\nrandom rows of i, f, o",
     [("cell input g", "res_g", "nog"), ("output gate o", "res_o", "nog"), ("input gate i", "res_i", "nog"),
      ("forget gate f", "res_f", "nog")]),
)


def _edges(x):
    lx = np.log(x)
    mid = (lx[1:] + lx[:-1]) / 2
    return np.exp(np.concatenate([[lx[0] - (mid[0] - lx[0])], mid, [lx[-1] + (lx[-1] - mid[-1])]]))


def specific(L, FE, tgt, kind):
    """specific damage of a target against one kind of its controls, per checkpoint and family (or panel: FE from
    family_effects(L, PANEL_ROWS)), in gap units: (control f - target f) = mean over the controls of dF minus the
    target's dF; (K, fam) and resamples (K, fam, NB)"""
    nm = L["names"]
    cix = [nm.index(c) for c in controls(tgt, nm)[kind]]
    t = nm.index(tgt)
    return (_seedmean(FE["dF"][:, cix]) - FE["dF"][:, t], _seedmean(FE["dF_b"][:, cix]) - FE["dF_b"][:, t])


def _seedmean(a, axis=1):
    """mean over the seeds of a control (axis), over the seeds that exist at a checkpoint (NaN where none does: the
    third pass's subspace controls outside W3_SUB)"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis)


def fig_map(run="trained", out=None, vmax=.3):
    """-> figure_bin/td_tt_lesions_map.png: the specific damage (control f - lesioned f; bright = the lesion removes
    more of the panel's behaviour than its control) of the surviving comparisons per checkpoint and panel (core answers,
    corner answers, holding, coupling), with the intact network's f above each map and the panel rows' c10 / c50"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    FL, FT, FN = fs["label"], 11., 9.5
    L = load_run(run)
    PE = family_effects(L, PANEL_ROWS)
    nm, x = L["names"], L["x"]
    cr = family_crossings()
    rows, ygrp = [], []
    for g, items in MAP_GROUPS:
        items = [it for it in items if it[1] in nm and it[2] in controls(it[1], nm)]
        ygrp.append((g, len(rows), len(rows) + len(items)))
        rows += items
    GAP = .7
    ypos, y = [], 0.
    for gi, (g, a, b) in enumerate(ygrp):
        y += GAP if gi else 0.
        ypos += [y + k for k in range(b - a)]
        y += b - a
    ybot = y - .5
    RH, PW, LAB, GX, RM = .36, 2.75, 3.7, .3, .2                                        # inches
    SUM, GS, TOP, BOT = 1.15, .62, 1.7, 2.25
    PH = (ybot + .5) * RH
    W = LAB + 4 * PW + 3 * GX + RM
    Hh = TOP + SUM + GS + PH + BOT
    fig = plt.figure(figsize=(W, Hh))
    E = _edges(x)
    im = None
    for j, p in enumerate(PANELS):
        x0 = LAB + j * (PW + GX)
        axs = fig.add_axes([x0 / W, (BOT + PH + GS) / Hh, PW / W, SUM / Hh])
        ax = fig.add_axes([x0 / W, BOT / Hh, PW / W, PH / Hh])
        F, Fb = PE["F_int"][:, j], PE["F_int_b"][:, j]
        lo, hi = iv(Fb)
        axs.fill_between(x, lo, hi, color=C_INT, alpha=.2, lw=0)
        axs.plot(x, F, color=C_INT, lw=1.8)
        axs.axhline(0, color=C_NAIVE, lw=1.1)
        axs.axhline(1, color="k", lw=1.1, ls=(0, (4, 2)))
        for a in (axs, ax):
            a.axvline(cr[p]["c10_med"], color=C_MARK, lw=1.1, ls=(0, (2, 2)), zorder=5)
            a.axvline(cr[p]["c50_med"], color=C_MARK, lw=1.1, zorder=5)
            _xaxis(a, FL, label=(a is ax))
            a.set_xlim(E[0], E[-1])
        axs.set_xticklabels([])
        axs.set_ylim(np.floor((np.nanmin(lo) - .05) * 10) / 10, max(1.15, np.ceil((np.nanmax(hi) + .05) * 10) / 10))
        axs.yaxis.set_major_locator(plt.MaxNLocator(4, steps=[1, 2, 5, 10]))
        if j == 0:
            axs.set_ylabel("intact $f$", fontsize=FL)
        axs.set_title(PANEL_TITLE[p] + "\n" + (F_CORNER if p == "corner answers" else F_FORMULA), fontsize=FT,
                      linespacing=1.5)
        for r, (lab, tgt, kind) in enumerate(rows):
            S, Sb = specific(L, PE, tgt, kind)
            v, (lo, hi) = S[:, j], iv(Sb[:, j])
            im = ax.pcolormesh(E, [ypos[r] - .5, ypos[r] + .5], v[None], cmap=plt.cm.viridis, vmin=-vmax, vmax=vmax,
                               shading="flat", rasterized=True)
            sig = ((lo > 0) | (hi < 0)) & np.isfinite(v)
            ax.plot(x[sig], [ypos[r]] * int(sig.sum()), "o", ms=3.2, color="w", mec="k", mew=.5, zorder=6, ls="none")
        ax.set_ylim(ybot, -.5)
        ax.set_yticks(ypos)
        ax.set_yticklabels([r_[0] for r_ in rows] if j == 0 else [], fontsize=FN)
    for g, a, b in ygrp:
        fig.text(.12 / W, (BOT + PH * (1 - ((ypos[a] + ypos[b - 1]) / 2 + .5) / (ybot + .5))) / Hh, g,
                 fontsize=FL + .5, ha="left", va="center", linespacing=1.25, fontweight="bold")
    hs = [Line2D([], [], color=C_INT, lw=1.8, label="intact Trained RNN"),          # key of the summary panels
          Line2D([], [], color=C_NAIVE, lw=1.1, label="naive ($f$ = 0)"),
          Line2D([], [], color="k", lw=1.1, ls=(0, (4, 2)), label="Bayes ($f$ = 1)")]
    fig.legend(handles=hs, loc="center left", bbox_to_anchor=(.12 / W, (BOT + PH + GS + SUM / 2) / Hh), fontsize=FN,
               frameon=False, handlelength=2.3, borderaxespad=0., labelspacing=.5)
    xm = (LAB + 2 * PW + 1.5 * GX) / W
    fig.text(xm, (BOT + PH + .1) / Hh, "Extra damage beyond the matched control, "
             r"$f_{\mathrm{control}} - f_{\mathrm{lesioned}}$ (gap units), per checkpoint", ha="center", va="bottom",
             fontsize=FT)
    cax = fig.add_axes([(LAB + PW * .9) / W, (BOT - 1.05) / Hh, (2.2 * PW) / W, .15 / Hh])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("bright: the lesion removes more of the behaviour than its control; dark: less", fontsize=FN + .5)
    fig.text(.5, 1 - .1 / Hh,
             "The interaction dims matter more than random dims for the core answers from the first checkpoint on, and "
             "more than both\nlatents' main-effect dims for corner answers and holding as these develop (coupling: "
             "barely, from 8k); from 1.6k on the main-effect\ndims remove more of the core answers. Gate drive: g in "
             "every panel from early on, o in all but corner answers, i more than\nrandom rows mainly for corner "
             "answers (late), f late less than random rows.",
             ha="center", va="top", fontsize=11.5, linespacing=1.35)
    fig.text(xm, .1 / Hh,
             f"Trained RNN, {len(x)} checkpoints ({x[0]:,.0f}-{x[-1]:,.0f} training episodes). Every cell: 10,000 probe "
             "+ 5,028 forward-set episodes (50 test environments; corner answers: 2,400 unselected corner-cell "
             "episodes).\nDot: 5-95% over 1,000 environment resamples excludes 0. Random controls: mean over 2 (dims) or "
             "3 (rows) seeds. Both latents' main-effect dims and the main-effect-free part:\nonly at 800-1,600 and "
             r"$\geq$" + " 2,400 training episodes (blank elsewhere). Brown lines: the panel rows' onset c10 (dotted) "
             "and half-way c50 (solid), timeline medians.",
             ha="center", va="bottom", fontsize=FN, linespacing=1.45)
    out = fig_path("td_tt_lesions_map.png") if out is None else out
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)
    return out


# ═══════════════════════════════════════════════════════════════════
# Validation: the lean forward against the engine; the definitions against the stored ck statistics
# ═══════════════════════════════════════════════════════════════════

def _cmp(a, b):
    return dict(max_abs=float(np.abs(a["logq"].astype(np.float64) - b["logq"].astype(np.float64)).max()),
                right_diff=int((a["right"] != b["right"]).sum()))


def checks(dev=None):
    dev = ti._device(dev)
    res = {}
    rng = np.random.default_rng(5)
    for run, E in (("trained", 2100), ("trained", 49995), ("echo", 2100)):
        key = f"{run}_{E}"
        w = ti.load(run, E)
        ref = ti.reference(w, dev, cov=True)
        rnc = {k: v for k, v in ref.items() if k != "cov_u"}
        units = np.sort(rng.choice(H, 100, replace=False))
        V = ti.random_subspace(50, w, 3)
        cases = [("intact", {}, {}),
                 ("mean_f", {"mean_rows": np.arange(H, 2 * H)}, dict(gate={"mean": "f"})),
                 ("mean_ifgo", {"mean_rows": np.arange(4 * H)}, dict(gate={"mean": "ifgo"})),
                 ("res_i", {"res_rows": np.arange(H)}, dict(gate={"resample": "i"})),
                 ("res_o", {"res_rows": np.arange(3 * H, 4 * H)}, dict(gate={"resample": "o"})),
                 ("f_units", {"f_units": units}, dict(gate={"f_units": units})),
                 ("c_units", {"c_units": units}, dict(gate={"c_units": units})),
                 ("clamp_u50", {"V": V}, dict(clamp={"u": V}))]
        r = {}
        for nm, spec, kw in cases:
            a = lforward(w, spec, rnc, dev=dev)
            b = ti.forward(w, device=dev, ref=ref, **kw)
            r[nm] = _cmp(a, b)
            if nm in ("intact", "res_i"):
                af = lforward(w, spec, rnc, fwd=True, dev=dev)
                bf = ti.forward(w, device=dev, ref=ref, episodes=ti.forward_set(), **kw)
                r[nm + "_fwdset"] = _cmp(af, bf)
            print(f"{key} {nm:10s} lean vs engine {r[nm]}" + (f"  forward set {r[nm + '_fwdset']}"
                                                                if nm + "_fwdset" in r else ""), flush=True)
        D = define(run, E, w, ref, dev)
        fr, fa = D["fac_rows"], D["factors"]
        iu = fr.index("u")
        r["tr_joint_vs_fac"] = float(np.abs(D["tr_joint"] - D["fac"][:, iu, fa.index("joint"), 0]).max())
        r["tr_joint_rel"] = r["tr_joint_vs_fac"] / float(D["fac"][:, iu, fa.index("joint"), 0].max())
        r["tr_r0_vs_fac"] = float(np.abs(D["tr_r0"] - D["fac"][:, iu, fa.index("r0"), 0]).max())
        r["tr_r1_vs_fac"] = float(np.abs(D["tr_r1"] - D["fac"][:, iu, fa.index("r1"), 0]).max())
        r["tot_vs_fac"] = float(np.abs(D["tot"] - D["fac"][:, iu, fa.index("total"), 0]).max())
        r["tot_max"] = float(D["tot"].max())
        blk = D["dvar_t"].reshape(T - 1, 4, H).sum(-1)                                      # steps 2..T
        fb = np.stack([D["fac"][1:, fr.index(f"Whh_u_{G}"), fa.index("total"), 0] for G in GATES], 1)
        r["dvar_vs_fac_rel"] = float(np.abs(blk / fb - 1).max())
        r["tau_ref_vs_ck_rel"] = float(np.abs(D["tau_ref"] / D["tau"] - 1).max())
        r["mu_chk"] = D["mu_chk"]
        print(f"{key} definitions: joint trace vs ck fac max |d| {r['tr_joint_vs_fac']:.2e} (rel {r['tr_joint_rel']:.1e}); "
              f"r0 {r['tr_r0_vs_fac']:.2e}, r1 {r['tr_r1_vs_fac']:.2e}; total {r['tot_vs_fac']:.2e} of "
              f"{r['tot_max']:.1f}; drive variance per block vs fac rel {r['dvar_vs_fac_rel']:.1e}; tau from the "
              f"engine's f vs ck tau rel {r['tau_ref_vs_ck_rel']:.1e}; cell-mean mean vs mu_u {r['mu_chk']:.1e}",
              flush=True)
        # clamp identities through record: per-step clamp residual (joint5), keep residual (keep20), read-out at step 1
        conds = {c[0]: c for c in conditions(D)}
        acc = {"clamp": 0., "keep": 0.}
        Vj = [torch.as_tensor(v, device=dev).float() for v in conds["joint5"][2]["V"]]
        Bk = [torch.as_tensor(v, device=dev).float() for v in conds["keep20"][2]["V"]]
        mu = torch.as_tensor(ref["mu_u"], device=dev).float()

        def rc(s, rows, st):
            acc["clamp"] = max(acc["clamp"], float(((st["u"] - mu[s]) @ Vj[s - 1]).abs().max()))

        def rk(s, rows, st):
            d = st["u"] - mu[s]
            acc["keep"] = max(acc["keep"], float((d - (d @ Bk[s - 1]) @ Bk[s - 1].T).abs().max()))
        a = lforward(w, conds["joint5"][2], rnc, dev=dev, record=rc)
        k = lforward(w, conds["keep20"][2], rnc, dev=dev, record=rk)
        i0 = lforward(w, {}, rnc, dev=dev)
        r["joint5_residual"], r["keep20_residual"] = acc["clamp"], acc["keep"]
        r["joint5_step1_logq"] = float(np.abs(a["logq"][:, 0] - i0["logq"][:, 0]).max())
        r["keep20_step1_logq"] = float(np.abs(k["logq"][:, 0] - i0["logq"][:, 0]).max())
        r["joint5_readout_orth"] = float(max(np.abs(D["Q"].T @ v).max() for v in conds["joint5"][2]["V"]))
        r["self_repeat"] = _cmp(lforward(w, conds["res_g"][2], rnc, dev=dev), lforward(w, conds["res_g"][2], rnc, dev=dev))
        print(f"{key} clamps: joint5 |(u - mu) V| <= {r['joint5_residual']:.1e}, step-1 log q change "
              f"{r['joint5_step1_logq']:.1e}, |Q^T V| {r['joint5_readout_orth']:.1e}; keep20 residual outside the kept span "
              f"{r['keep20_residual']:.1e}, step-1 log q change {r['keep20_step1_logq']:.1e}; repeat {r['self_repeat']}",
              flush=True)
        res[key] = r
        ti._REF.clear()
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "checks.json"), "w"), indent=1)
    return res


def check_self(dev=None):
    """the second pass's hook: forget gates clamped to their own clean per-step means (record), and the clamp of no
    unit = intact"""
    dev = ti._device(dev)
    res = {}
    for run, E in (("trained", 5000), ("echo", 5000)):
        w = ti.load(run, E)
        ref = ti.reference(w, dev)
        fm = torch.as_tensor(ref["f_mean"], device=dev).float()
        units = np.sort(np.argsort(-ck_tau(run, E)[0])[:50])
        ut = torch.as_tensor(units, device=dev)
        acc = {"d": 0.}

        def rec(s, rows, st):
            acc["d"] = max(acc["d"], float((st["f"][:, ut] - fm[s, ut]).abs().max()))
        lforward(w, {"f_units": units, "f_value": "self"}, ref, dev=dev, record=rec)
        a = lforward(w, {"f_units": np.zeros(0, np.int64), "f_value": "self"}, ref, dev=dev)
        b = lforward(w, {}, ref, dev=dev)
        res[f"{run}_{E}"] = dict(self_clamp_max_abs=acc["d"], empty_vs_intact=_cmp(a, b))
        print(run, E, res[f"{run}_{E}"], flush=True)
    json.dump(res, open(os.path.join(OUT, "checks_self.json"), "w"), indent=1)
    return res


def check_third(dev=None):
    """the third pass's hooks and definitions (-> checks_third.json): forget gates at given values (record: |f -
    given| on the units; given = the own mean reproduces f_value 'self'); the g-free row sets hold no g row; the new
    subspaces orthonormal, off the read-out rows, held by the clamp (|(u - mu) V| after every step), pure5 without
    main-effect variance, and their per-step variance equal to the engine's captured_variance (since 11-02 measured in
    the clamp's actual basis); the factor split adds up to the subspace's variance"""
    dev = ti._device(dev)
    res = {}
    for run, E in (("trained", 5000), ("echo", 5000)):
        w = ti.load(run, E)
        sub = run == "trained"
        ref = ti.reference(w, dev, cov=sub)
        rnc = {k: v for k, v in ref.items() if k != "cov_u"}
        D3 = define3(run, E, w, ref, dev, subspaces=sub)
        r = {}
        u, val = D3["fsets"]["fshift_own10_s0"]
        ut, vt = torch.as_tensor(u, device=dev), torch.as_tensor(np.nan_to_num(val), device=dev).float()
        acc = {"d": 0.}

        def rec(s, rows, st):
            acc["d"] = max(acc["d"], float((st["f"][:, ut] - vt[s]).abs().max()))
        lforward(w, {"f_units": u, "f_value": val}, rnc, dev=dev, record=rec)
        r["given_max_abs"] = acc["d"]
        r["given_own_mean_vs_self"] = _cmp(lforward(w, {"f_units": u, "f_value": ref["f_mean"][:, u]}, rnc, dev=dev),
                                           lforward(w, {"f_units": u, "f_value": "self"}, rnc, dev=dev))
        r["nog_g_rows"] = int(sum(((rr >= 2 * H) & (rr < 3 * H)).sum() for rr in
                                  (nog_rows(n_, sd) for n_ in (H, 2 * H) for sd in range(N_RAND))))
        r["dfabs"] = {k: float(v.mean()) for k, v in D3["dfabs"].items() if k.startswith(("fshift_own10", "fclamp_own10",
                                                                                            "fclamp_rand10"))}
        if sub:
            Q = readout_basis(w)
            for k in ("add5", "pure5", "pc5", "pcspan5_s0"):
                Vl = D3["V"][k]
                r[f"{k}_orth"] = float(max(np.abs(v.T @ v - np.eye(v.shape[1])).max() for v in Vl))
                r[f"{k}_readout"] = float(max(np.abs(Q.T @ v).max() for v in Vl))
                cv = np.array([ti.captured_variance(Vl[t], ref, w)["inside"][t] for t in range(T)])
                r[f"{k}_var_vs_engine_rel"] = float(np.abs(cv / D3["var"][k] - 1).max())
            r["pure5_main_var_share"] = float(D3["main_var"]["pure5"].sum() / D3["var"]["pure5"].sum())
            r["pure5_main_rank"] = [int(D3["rank_main"].min()), int(D3["rank_main"].max())]
            for k in ("add5", "pure5"):
                Vt_ = [torch.as_tensor(v, device=dev).float() for v in D3["V"][k]]
                mu = torch.as_tensor(ref["mu_u"], device=dev).float()
                acc["d"] = 0.

                def rc(s, rows, st):
                    acc["d"] = max(acc["d"], float(((st["u"] - mu[s]) @ Vt_[s - 1]).abs().max()))
                lforward(w, {"V": D3["V"][k], "mode": "clamp"}, rnc, dev=dev, record=rc)
                r[f"{k}_clamp_residual"] = acc["d"]
            r["split_vs_var_rel"], r["split_all_vs_totQc_rel"] = D3["split_chk"], D3["split_all_chk"]
        print(run, E, json.dumps(r), flush=True)
        res[f"{run}_{E}"] = r
        ti._REF.clear()
    json.dump(res, open(os.path.join(OUT, "checks_third.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "doc"
    if what == "run":
        worker(sys.argv[2], [int(e) for e in sys.argv[3].split(",")])
    elif what == "extra":
        worker_extra(sys.argv[2], [int(e) for e in sys.argv[3].split(",")])
    elif what == "third":
        if len(sys.argv) > 4 and sys.argv[4] == "nosub":
            for E_ in [int(e) for e in sys.argv[3].split(",")]:
                if not os.path.exists(third_file(sys.argv[2], E_)):
                    run_third(sys.argv[2], E_, subspaces=False)
        else:
            worker_third(sys.argv[2], [int(e) for e in sys.argv[3].split(",")])
    elif what == "check_third":
        check_third()
    elif what == "fourth":
        for E_ in [int(e) for e in sys.argv[3].split(",")]:
            if not os.path.exists(fourth_file(sys.argv[2], E_)):
                run_fourth(sys.argv[2], E_)
    elif what == "checks":
        checks()
    elif what == "check_self":
        check_self()
    elif what == "figure":
        fig_main()
        fig_map()
    elif what == "numbers":
        numbers()
    elif what == "table":
        table()
    else:
        print(__doc__)
