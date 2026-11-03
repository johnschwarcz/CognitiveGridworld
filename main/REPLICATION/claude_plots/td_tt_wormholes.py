"""Wormholes through training: when the Trained RNN learns to hold a corner split, what it does before, and how it
leaves one (John 11-01, before heading home: "go over our existing plots ... see if you want to do further 'through
training' analyses ... wormholes, jumps, couplings, splitting, unfolding ..."; this script: wormholes / holding a
split). Built 11-01 evening, corrected the same night after an independent verification (see CORRECTIONS).
-> main/REPLICATION/figure_bin/td_tt_wormholes.png       (a-g, the forward set below)
-> main/REPLICATION/figure_bin/td_tt_wormholes_maps.png  (a, b, the probes, no thresholds on the events)
-> main/DATA/experimental/training_dynamics/through_training/wormholes/synthesis.npz (cross-topic synthesis,
   td_tt_gap conventions; timeline.npz, the earlier export, is kept unchanged)

End of training (memory round7-jumps-order "WORMHOLES KEPT APART", figure8-learning-link, crossings-findings; the
paper's networks on the 39-environment scan): at corner splits the Trained RNN holds the split (82%, median TV .06),
the ESN never does (0.7%, .54) and commits to a corner; at Bayes's coupled crossings the Trained moves diagonally; the
factorized agents' corner errors are adjacent corners. Here the same quantities through training, for the seed-20261029
Trained RNN and ESN (one world, identical initial weights).

DATA
  Forward set (td_tt_wormholes.png): the grid episodes of td_grids_fwd.py (the probes' 50 environments x 100 cells x
    24 trajectories x 30 steps, seed 20261111), Bayes and naive recomputed here in float64: every episode with a corner
    split or a one-variable end split at any step 2-30, plus the first 12 trajectories of each corner cell (2,400,
    unselected; the corner answers use only these): 5,028 episodes. Replayed here on CPU (fp32, weights only; the
    hand-written LSTM of td_analysis, read-in and input weights applied once per (observation pattern, environment))
    at 58 checkpoints per network (every second of the 114, and the last). Counts: 3,859 corner-split stream-steps
    (steps 2-30: 5,037), 7,718 corner var-steps, 4,763 one-variable end splits (2-30: 7,384), 2,400 corner episodes,
    2,204 moves out of corner splits.
  Probes (td_tt_wormholes_maps.png and the replication): analysis/probes.npz (10,000 episodes = 50 environments x 100
    cells x 2 trajectories) with the stored logq of every checkpoint: 360 corner-split stream-steps (720 var-steps),
    365 one-variable, 400 corner episodes, 580,000 variable-steps, 4,823 moves of Bayes's >= 2 cells (3,229 diagonal
    >= 30 deg, 576 one-axis <= 15 deg). Figures use the same 58 checkpoints.

DEFINITIONS
  Marginals: Bayes's pi_v (v = r0, r1; from its joint posterior), the network's q_v (its marginal read-out; the stored
    fp16 logs renormalised), naive's. Smaller-end mass of a marginal b_v: e_v = min(b_v(0), b_v(9)).
  Corner split (recovered exactly from the 10-27 scratch script geom/wormhole_marginals.py): both of Bayes's marginals
    are split between their ends, e_v > .15 and pi_v(0) + pi_v(9) > .6 for both v, at steps 2-16 (the original
    window; steps 2-30 as a check). A (0,0)/(9,9) and a (0,9)/(9,0) split have the same marginals.
  Held (the original rule): the network's marginals keep both ends, e_v > .15 for both v (a joint event, per
    stream-step); per variable: e_v > .15 for that v.  Committed: both answers (argmax q_v) on a corner value (0 or 9),
    not held.
  Control 1, one-variable end split: v split as above, the other variable not (its smaller end < .05).
  Control 1m (per-variable holding only): the one-variable splits reweighted to the corner splits' histogram of step x
    Bayes's e_v (.05 bins); it covers 99.8% of the corner var-steps on the forward set, 92.2% on the probes.
  Control 2, matched non-splits: stream-steps where neither variable is end-split, weighted to the corner splits'
    histogram of step x log10 FR of the variable (naive's error; .25 bins) x Bayes's entropy of the variable (.2
    nats); coverage 100%.
  TV = 1/2 Sum_r |q_v(r) - pi_v(r)| per variable; averaged over variables and stream-steps afterwards.
  Corner answers at T: the 4 corner cells; answer (argmax q_0, argmax q_1) at T = the truth, an adjacent corner (one
    variable at its opposite end, the other right), the partner corner (both at their opposite ends), or other.
  Moves: posterior means mu = (E[r0], E[r1]) (float32); a move = the change into a step; coupling angle phi =
    arctan(min_v |dmu_v| / max_v |dmu_v|): 0 one variable, 45 diagonal (it ignores the sign). Leaving a corner split:
    a move of Bayes's of >= 2 cells at >= 30 deg whose previous stream-step is a corner split (steps 2-29); the
    network's move at the same step: phi, distance along Bayes's move dmu . dmu_B / |dmu_B| (cells, absolute), and
    same diagonal (sign(dmu_0 dmu_1) = Bayes's).
  Maps (probes, every variable-step / every move, no event threshold): a, the network's e_v by Bayes's e_v (580,000
    variable-steps, steps 2-30); b, the network's phi at Bayes's moves of >= 2 cells by Bayes's phi (4,823 moves).
    Step drift removed (all of these quantities drift with the step, e.g. the untrained network's beliefs sharpen
    because its read-out is summed over steps): the colour is each value minus the network's mean at that step plus
    its mean over all steps (step means pooled over environments), averaged in bins of Bayes's raw value; the summary
    (top) is the least-squares slope within steps, the n-weighted pooled within-step fit
      beta = Sum_t Sum_i (y_i^net - ybar_t^net)(y_i^B - ybar_t^B) / Sum_t Sum_i (y_i^B - ybar_t^B)^2
    (i over the points at step t, ybar_t their means at that step pooled over environments; = Sum_t n_t cov_t / Sum_t
    n_t var_t, the per-step slopes averaged with weights n_t var_t; Bayes 1). For e every step has 20,000 points; for
    phi the moves per step fall from 2,032 at step 2 to 3 at step 30, and naive's direction slope (.140) comes mostly
    from step 2 (45% of the weight, slope .249 there; .116 at steps 3-5, -.048 at steps 6-30, .051 without step 2; the
    equal-weight mean of per-step slopes is -.003 over the 15 steps with >= 30 moves, +.040 over all 29), so whether
    a network passes naive's direction level depends on this weighting. Reported too: the mean over steps of per-step
    slopes (steps with >= 30 points; equal weight per step) and the slope of one fit pooled over all steps (the first
    version's summary, which counts the drift of the step means: see CORRECTIONS).
  Averaging order: every statistic is computed per stream-step (per variable where it is per variable; joint events per
    stream-step), summed per environment and pooled over environments; slopes and maps within steps (above).
  Timing: a crossing is the training episode (log-interpolated between checkpoints) after which the curve stays past
    the level to the end of training. Levels: 10 / 50 / 90% of the change from the initial network to the end (the
    mean of the last 3 checkpoints), unless named (half-way back from a peak; a reference level). Intervals: 5-95% over
    the 50 environments resampled 1,000 times, every resample crossing the same level (a reference level: the
    resample's own reference, paired); ">end" = more than 5% of the resamples never stay past it. Two events are put in
    order only when their intervals do not overlap. x axis: training episodes E - E // 5 (every 5th episode is a test
    episode that trains nothing), the initial network at 10. Passes naive: the first of the 114 probe checkpoints at
    which the Trained RNN's goal accuracy at T beats naive's (= fig_entanglement_driver.passes_naive): 880, between
    the 800 and 880 checkpoints. Grid unfolding (td_summary_data.grids(): % of the belief grid folded at T where
    naive's grid folds, 19 checkpoints): 25 / 50 / 75% of its drop at 551 / 770 / 1,262 (the grey band).

FINDINGS (Trained RNN unless named; training episodes; forward set unless "probes"; crossings with 5-95% intervals;
"at the end" = the last checkpoint unless "mean of the last 3")
1 Holding a corner split is learned late: after passing naive (800-880) and after the grid unfolds (551-1,262). Joint
  held share <= .0034 up to 360 (at 320; 0 up to 160), .004-.007 at 400-880, .031 at 1,200, .100 at 2,000, .309
  at 4,200, .465 at 7,800, .558 at 10,400, .733 at 20,000, .825 at the end (the paper's network: 82%); 10 / 50 / 90%
  at 1,916 (1,837-2,080) / 6,192 (5,417-7,736) / 25,320 (19,049-26,355): still rising at the end. ESN <= .008 at every
  checkpoint (end .0005; paper .7%); naive .004. Steps 2-30: .802 (mean of the last 3), 10 / 50 / 90% at 1,938
  (1,864-2,349) / 6,483 (5,668-8,953) / 25,681 (24,831-27,529). Probes: .092 at 2,000, .492 at 5,800, .789 at the end;
  ESN <= .008. The original median TV (TV averaged over the two variables per stream-step, then the median; the
  paper's summary): .494 at 880, .365 at 2,000, .188 at 5,800, .071 at the end (paper .06); ESN .565 (paper .54);
  naive .538.
2 Before it holds, it commits (a naive-like phase): committed share .26 at 240, .52 at 560, .62 at 880, peak .722 at
  1,680 (5-95% .68-.76; naive .554). It first reaches half its peak at 337 (304-446; a first crossing, as it falls back
  later) and stays below half-way back to its end value (.444) after 9,597 (8,416-10,656); .171 at the end. The ESN
  climbs to naive's level and stays (.45-.53 from 2,000 on). Answer = the true cell at the split: Bayes .363, naive
  .095, Trained .164 at 880 -> .332 at the end, ESN .098.
3 Not wormhole-specific (failed hypothesis: corner splits are not learned separately from one-variable splits). Per
  variable, an end split whose partner variable is split too and one whose partner is not are held on one schedule: 10 /
  50 / 90% at 1,074 (968-1,169) / 3,961 (3,729-4,672) / 17,753 (14,362-24,893) vs 985 (761-1,362) / 3,849 (3,454-4,781)
  / 13,581 (11,951-26,987): each pair overlaps (cannot be ordered); .234 vs .242 at 2,000, .569 vs .556 at 5,800, .893
  vs .860 at the end; corner minus one-variable +.012 [-.053, +.071] averaged over 3,000-15,000. ESN .023 / .046, naive
  .045 / .063 at the end. The late raw gap (+.033 [+.011, +.060] at the end) is Bayes's, not the network's: reweighted
  to the corner splits' histogram of step x Bayes's e_v, the one-variable splits are held .891 at the end (corner minus
  matched +.001 [-.018, +.026]; matching on step alone leaves .865, on e_v alone .886), because Bayes's corner splits
  are more balanced (mean e_v .279 vs .262; e_v < .2 in 20% vs 29% of the var-steps) and barely split ends are held less
  in both sets (.66 / .68 at e_v .15-.2, .95-.99 above .25). The smaller probe sample (720 vs 365 var-steps; 2
  trajectories per cell) shows a mid-training corner advantage that the forward set does not: +.144 [+.023, +.253] at
  5,800, +.126 [+.038, +.203] at 10,400, +.102 [+.015, +.185] averaged over 3,000-15,000 (forward set +.012 [-.053,
  +.071]; the intervals overlap); matched on step x e_v (92% coverage) it shrinks to +.069 [-.021, +.146], and the
  probes' own crossings overlap too (50%: 3,823 (2,946-4,217) vs 4,654 (3,413-7,474)); at the end .872 vs .847 (matched
  .880). The joint share comes later than the per-variable shares only because it needs both variables. Smaller-end mass
  (absolute): corner .028 at 880, .086 at 2,000, .184 at 5,800, .265 at the end (Bayes .279); one-variable .035 / .091 /
  .181 / .248 (Bayes .262); ESN .013 / .022.
4 End splits approach Bayes later than equally hard non-splits, but the lag is modest in TV. TV per variable, corner /
  one-variable / matched non-splits: .507 / .508 / .512 at 880 (the same), .385 / .378 / .314 at 2,000, .220 / .222 /
  .153 at 5,800, .083 / .087 / .055 at the end; half-way at 2,334 (1,950-2,625) / 2,048 (1,801-2,452) / 1,701
  (1,565-1,909); the non-splits matched to the one-variable splits 1,525 (1,391-1,641). Each split set reaches half-way
  after its own matched control (separate intervals); corner and one-variable splits cannot be ordered. Naive .572 /
  .509 / .696; ESN .588 / .531 / .665 (mean of the last 3). Probes: .080 / .084 / .042 (mean of the last 3; their
  matched non-splits come from all cells and are easier than the forward set's, drawn from the same corner and
  split-prone episodes: the control depends on the pool beyond the matched variables).
5 Corner answers at T (2,400 corner episodes): truth 10 / 50 / 90% at 293 (264-324) / 1,361 (1,020-1,463) / 9,372
  (7,509-36,039); it first reaches the half-way level (.381) at 1,120 (1,018-1,447) but dips to .380 at 1,360. .762 at
  the end (mean of the last 3; Bayes .765, naive .170, ESN .136). Wrong corners are a transient of early training:
  adjacent corner peak .072 at 560 (.055-.088), below half-way back (.039) from 1,080 (669-1,316); partner corner peak
  .076 at 360 (.058-.096), below half-way back (.040) from 973 (677-1,249); .018 / .010 at 2,000 and .005 / .004 at the
  end (Bayes .003 / .004). From 280 to 880 the Trained's partner-corner errors (.046-.076) are above naive's (.039) and
  as common as its adjacent ones (.023-.072); naive's wrong corners are mostly adjacent (.166). ESN: adjacent up to .187
  at 2,600, .12-.18 later (naive .166); partner .03-.04. Probes (400 episodes): truth .767 (mean of the last 3),
  adjacent peak .065 at 560, partner peak .0725 at 560.
6 Leaving a corner split (2,204 diagonal moves of Bayes's out of one): distance first, direction later. Distance along
  Bayes's move .35 cells at 880 (naive .55, Bayes 3.41), 1.26 at 2,000, 2.99 at 5,800, 3.34 at the end; 10 / 50 / 90% at
  857 (748-996) / 2,644 (2,428-2,861) / 6,532 (5,415-9,328) (size |dmu| 50% at 1,871 (1,723-2,001)). Coupling angle: an
  early transient (initial 15.8 deg, 25.4 at 40, 20.2-21.0 at 80-240; the ESN's too: 27.5 at 40) returns to the initial
  / naive level by 1,200 (18.0 at 560-720, 16.4 at 880, 15.9 at 1,200; naive 16.6, Bayes 43.1), then the angle rises: 10
  / 50 / 90% of the change from the initial network at 2,331 (1,743-2,570) / 6,327 (5,405-8,003) / 25,493
  (20,383-28,243); 37.5 at the end. Distance (50% 2,644) comes before direction (50% 6,327): the intervals separate.
  Same diagonal as Bayes: .63 at 880, .72 at 2,000, .90 at 5,800, .99 at the end (initial .55; naive .53). ESN at the
  end: distance .32 cells, angle 16.5 (= naive's 16.6), same diagonal .59 (naive .53), size .51.
7 Probes, every belief and every large move (maps; within steps): the slope of the network's smaller-end mass on Bayes's
  is near 0 up to 400 (.007 initially, at most .048), passes naive's (.094) and stays above it from 748 (562-893; at the
  720 / 880 checkpoints .087 / .126), then rises: .194 at 1,200, .328 at 2,000, .585 at 4,200, .769 at 10,400, .931 at
  the end; 10 / 50 / 90% at 771 (685-852) / 3,509 (2,817-3,837) / 14,100 (12,852-23,746). Reaching naive's level cannot
  be ordered against passing naive's accuracy (800-880) or the unfolding (551-1,262). The ESN's slope sits at about
  naive's level from ~2,000 (.101 at 2,000, at most .116) and ends at .088. Per-step slopes averaged with equal weight
  per step tell the same story (naive .055; Trained .010 initially, .025 at 400, .093 at 880, .276 at 2,000, .903 at the
  end; ESN <= .083 over all 58 checkpoints (at 3,000), .048 at the end); the first version's pooled slope did not (see
  CORRECTIONS). Direction: the slope of the network's angle on Bayes's (within steps) is ~0 up to 1,200 (-.014 to
  .042; naive .140), then .095 at 2,000, .308 at 4,200, .578 at 10,400, .862 at the end; it passes naive's and stays
  above from 2,454 (1,989-2,858). That crossing depends on the weighting of steps: naive's .140 is mostly its step-2
  slope (.249, 45% of the weight; DEFINITIONS), and with equal weight per step naive's level is -.003 (the 15 steps with
  >= 30 moves; +.040 over all 29). 10 / 50 / 90% at 1,933 (1,374-2,182) / 5,852 (5,379-6,928) / 23,928 (20,024-26,347);
  ESN -.022 to .078 (end .071), below naive's mostly at step 2 (at the end .106 vs .249; without step 2 .042 vs .051).
  Per-step averages: naive -.003, Trained .894 at the end, ESN -.062 to .030; pooled: naive .127, Trained .859. At
  Bayes's large diagonal moves (3,229) the Trained's angle is 21.5-22.8 from 240 to 2,000 (naive 22.5) and 36.4 at the
  end (Bayes 39.4), 50% at 7,743 (6,982-9,130); at Bayes's large one-axis moves (576) 21-27 up to 1,200 and 8.7 at the
  end (Bayes 7.5, naive 17.9), 50% at 4,633 (3,914-5,422): one-axis moves are followed before diagonal ones (the
  intervals separate). Its move size at the diagonal moves: 1.34 at 880 (naive 1.39), 50% at 1,602 (1,197-1,701), 3.06
  at the end (Bayes 3.11); ESN angles 22.2 / 20.3 and size 1.08 at the end. (The angle means at the diagonal / one-axis
  moves pool steps; each curve keeps one fixed set of moves, so only their timing is compared.)
Timeline (Trained RNN; "<" only where the 5-95% intervals do not overlap, in each of 21 bootstrap seeds (CHECKS); "~"
cannot be ordered at this resolution):
  corner answers start (truth 10% 293, 264-324) ~ committing rises (half its peak 337, 304-446)
  < end-mass slope at naive's (748, 562-893) ~ grid unfolds (551-1,262) ~ passes naive's accuracy (800-880) ~ distance
    out of splits 10% (857, 748-996) ~ per-variable holding 10% (985-1,074, 761-1,362) ~ wrong corners half gone
    (973-1,080, 669-1,316) ~ corner answers half-way (1,361, 1,020-1,463; this one, and per-variable corner holding
    10%, can be put after passing naive)
  < joint corner split held 10% (1,916, 1,837-2,080)
  < distance out of splits 50% (2,644, 2,428-2,861) ~ same diagonal as Bayes's move out of splits 50% (2,796,
    2,455-3,430)
  < per-variable holding 50% (3,809-3,961, 3,454-4,781): distance comes before all three sets, the same diagonal only
    before the corner-split ends (3,961, 3,729-4,672); the same diagonal ~ the one-variable (3,849, 3,454-4,781) and
    matched (3,809, 3,464-4,718) splits' 50%: cannot be ordered at this resolution (the bounds, 3,430 vs 3,454 /
    3,464, are closer than the resampling noise: the intervals separate in only 8 / 7 of 21 bootstrap seeds). The
    end-mass slope's 50% (3,509, 2,817-3,837) lies between and cannot be ordered against any of them
  < direction slope 50% (5,852, 5,379-6,928) ~ joint corner split held 50% (6,192, 5,417-7,736) ~ angle out of splits
    50% (6,327, 5,405-8,003)
  < committing half-way back (9,597, 8,416-10,656)
  < the 90% points of holding and direction (13,390-25,493; joint held 25,320, 19,049-26,355).
  The corner answers at T are half-way before any holding or diagonal-leaving measure is (intervals separate; the
  nearest is the size of the network's move out of a corner split, 50% at 1,871 (1,723-2,001); the move size at any
  of Bayes's large diagonal moves, 1,602 (1,197-1,701), overlaps it). Holding a split and leaving it diagonally are
  the last part of learning (joint held .733 at 20,000, .825 at the end, still rising).

FAILED HYPOTHESES AND NULLS
  - Wormhole-specific learning: NOT found. Per variable, corner-split ends and one-variable end splits are held on one
    schedule (the 10, 50 and 90% crossings overlap); matched on step x Bayes's e_v the late raw gap vanishes (.893 vs
    .891 at the end). The probes' mid-training corner advantage (+.10 [+.02, +.19] over 3,000-15,000; +.07 [-.02,
    +.15] matched) is not seen in the forward set (+.01 [-.05, +.07]), which has ten times the var-steps.
  - No early wormhole phase: the Trained RNN never holds a corner split before passing naive (<= .7% up to 880), and
    within steps its smaller-end mass hardly follows Bayes's before 400 (slope .007-.048) and is at naive's level only
    around passing naive (748, 562-893).
  - The early angle bump is not a learned direction: both networks show it (25.4 / 27.5 deg at 40), it returns to the
    initial / naive level (~16) by 1,200, and the within-step direction slope is ~0 (-.014 to .042) up to 1,200.
  - The splits' lag behind matched non-splits is modest in TV (the same at 880, .507 / .508 / .512; half-way 2,334 /
    2,048 vs 1,701 / 1,525). The large delay appears only in the thresholded held share, which needs both ends > .15.
  - ESN: null on every measure. It never holds a split (<= .8%), never moves diagonally out of one (16.5 deg = naive's
    16.6; .32 cells), its end-mass slope stays at about naive's level (<= .116), and its wrong corners settle at
    naive's adjacent-corner level. One small exception: its moves out of corner splits follow Bayes's diagonal more
    often than naive's (.587 vs .532 at the end, gap fraction +.12; above naive's level for good from 3,207
    (53-16,998), and every resample crosses), far below Bayes's 1.
  - Unfolding comes before most holding, not all: the grid unfolds (551-1,262) while per-variable holding reaches 10%
    (985 (761-1,362) / 1,074 (968-1,169); both estimates inside the band) and the joint corner split is rare (.031 at
    1,200); per-variable holding is half-way only at 3,809-3,961 (3,454-4,781).
  - Retracted from the first version: the pooled-slope "early tracking" of the untrained network (.146 at
    initialization, above naive's .119) and its "trough" (.073 at 200), a step-drift artefact.

CORRECTIONS (11-01, after an independent verification of the first version)
  - The maps' summary slopes were one least-squares fit pooled over steps 2-30, which counts the drift of the step
    means: every observer's beliefs sharpen with the step (the untrained network's too, as its read-out is summed over
    steps: its mean e_v falls from .092 at step 2 to .024 at step 30, Bayes's from .020 to .001), so e_net and e_B fall
    together. The pooled smaller-end slope read .146 for the initial network (above naive's .119), .177 at 40 and a
    "trough" of .073 at 200: an artefact. Within steps it is .007, .014 and .023 there. The heat maps had the same
    artefact (Trained at 40: pooled bin means .008 for e_B < .01 but .036-.043 for e_B .01-.2; within steps .010-.013
    in every bin). Both are now within steps; the direction slope barely changed (pooled vs within: naive .127 vs
    .140, end .859 vs .862).
  - The posterior means were stored in fp16 (spacing .002-.008 cells), which set the smaller component of the
    network's moves out of corner splits to exactly 0 in 2.9% (Trained) / 5.3% (ESN) of them on average (up to 11.5%
    at one checkpoint), forcing phi = 0 and "not the same diagonal" for those moves, while the Bayes and naive
    references were float64 (shares measured against the first version's records). Now float32 (0.04% / 0%; first
    version -> now): same diagonal at 880 .593 -> .630, angle at 880 16.20 -> 16.43,
    minimum 15.69 -> 15.86, initial 15.5 -> 15.8; ESN at the end same diagonal .550 -> .587 and angle 16.10 -> 16.54
    (naive 16.55). Distance and size were unaffected.
  - Panel f's 22.5-degree "no preferred direction" line is dropped: 22.5 is the average of an isotropic move, not an
    uncoupled reference (topic jumps builds a matched null). The early part of the angle is a transient (25.4 deg at
    40; ESN 27.5) that returns to the initial / naive level (~16) by 1,200 before the rise; the first version called
    it a drop to naive ("gets worse"), but the initial network is already at that level (15.8).
  - "Unfolding does not wait for held splits" was overstated: per variable about 10% of split variables are already
    held inside the unfolding band; only the joint corner split is rare there.
  - The late raw gap in per-variable holding (corner .893 vs one-variable .860) is Bayes's, not corner-specific
    learning (see 3). The ".08-.16" range of the first version's finding 7 was wrong even for the pooled slope
    (.07-.18); it is replaced by the within-step slope.
  - Timing now follows the 11-01 rule: a crossing is the episode after which the curve stays past the level to the end
    (the first version used the first crossing), every crossing has an environment-bootstrap interval (resamples cross
    the same level), and two events are ordered only if their intervals do not overlap. Changed: corner answers
    half-way 1,120 -> 1,361 and 90% 4,058 -> 9,372; joint held 90% 20,728 -> 25,320; TV corner half-way 2,141 ->
    2,334; adjacent wrong corners half gone 796 -> 1,080; the angle's crossings now count from the initial network
    (50% 6,327; from its 1,200 minimum 6,343). Unchanged: joint held 10 / 50% (1,916 / 6,192), per-variable 10 / 50 /
    90%, the distance, committing half-way back (9,597).

CORRECTIONS (11-02, after a re-verification of the corrected version; no number changed)
  - The maps' summary formula was written Sum_t cov_t / Sum_t var_t, which reads as per-step moments each divided by
    its own count; the code computes the n-weighted pooled within-step fit (DEFINITIONS; maps titles a and b). For e
    the two readings agree (20,000 points at every step); for phi they do not (naive .140 vs -.071 under the literal
    reading; Trained at the end .862 vs .812), as the moves per step fall from 2,032 to 3. Naive's direction slope is
    mostly its step-2 slope, now stated next to the direction slope's "passes naive" crossing (7) and in the footnote.
  - Two ranges: the ESN's per-step-averaged end-mass slope is <= .083 (at 3,000; it read <= .082 from the printed
    checkpoints only), and the joint held share is not 0 up to 240 (.0008 at 200) but <= .0034 up to 360 (at 320; 0
    up to 160). numbers prints the ranges over all 58 checkpoints.
  - "The corner answers at T are half-way first" implied an order the intervals do not give against the wrong corners
    and the matched non-splits; it now reads "before any holding or diagonal-leaving measure" (Timeline).

CORRECTIONS (11-02, after a second re-check; no earlier number changed)
  - The timeline put the same diagonal's 50% (2,796, 2,455-3,430) before all three per-variable holding 50%
    crossings. Against the one-variable and matched splits that order rested on bounds 3,430 vs 3,454 / 3,464, which
    separate in only 8 / 7 of 21 bootstrap seeds, so it now reads "~" there (cannot be ordered at this resolution).
    Its order before the corner-split ends' 50% and distance's order before all three hold in 21 of 21 seeds, as does
    every other order of the timeline and of findings 1, 4, 6 and 7 (CHECKS).
  - The maps footnote and the direction-slope synthesis notes gave naive's equal-weight per-step slope (-.003) without
    its restriction to the 15 steps with >= 30 moves; they now say so and add the mean over all 29 steps (+.040), as
    do DEFINITIONS and finding 7.
  - "Joint held share <= .003 up to 360" stood next to .0034 at 320; it now reads <= .0034 up to 360 (at 320; 0 up to
    160).
  - SYNTHESIS EXPORT now names the c0s that cannot be ordered (the same diagonal against the joint held share and the
    angle; distance against one-variable holding) and says that committing's c0 / c50 bracket its half-way back by
    construction.

CHECKS
  Original thresholds recovered from the 10-27 scratch script geom/wormhole_marginals.py; its end-of-training numbers
    reproduced on the new run (joint held .825 vs the paper's 82%, ESN .001 vs .7%, median TV .071 vs .06, ESN .565 vs
    .54). An independent verification (11-01) rebuilt the forward set (the same 5,028 episodes, counts and masks),
    replayed it with torch.nn.LSTM and reproduced the numbers of findings 1-6 to the third decimal.
  CPU replay vs td_grids_fwd's GPU replay at the 19 shared checkpoints, both networks: posterior means within .0039
    (the reference's fp16 storage), answers at T identical in 100%. The float32 re-replay (116 files) reproduced every
    other stored key of the first version byte for byte (lq_tgt, tv8, lqT, grid checks); its means differ from the old
    fp16 ones by <= .0039 and from the verifier's independent replay by <= 5e-4 cells (angle equal to 3 decimals).
  tv8 quantisation (uint8 TV x 255): max error .0021 against the exact TV at the target stream-steps.
  Within-step slopes from the per-(environment, step) sums equal a direct fit on the raw probe beliefs at every
    checkpoint (e.g. e: .007 / .126 / .931 at 10 / 880 / the end; phi: -.003 / .029 / .862), and the verifier's.
  Every statistic shared with the first version's caches is unchanged except the move statistics that use the
    posterior means (angle, same diagonal; distance and size to the fp16 rounding).
  Step x Bayes-e_v matched control: coverage 99.8% (forward) / 92.2% (probes); end value .891 = the verifier's.
  Unfolding landmark reproduced exactly with either crossing definition (551 / 770 / 1,262); passes naive 880.
  Synthesis export: the forward set's environment e is the probes' (identical Z and environment-major indices; each
    episode's observations most likely under its own environment's table for 96.8% of the grid episodes vs 97.1% of
    the probes); with all counts 1 every resampled curve equals the full-sample curve (1.4e-14); this module's
    cross_level, strictly past, reproduces every c0 / c50 of td_tt_gap on the same resamples (4.7e-11 episodes). A
    scratch recomputation (11-02; explicit loops over environments, and the slopes from the re-verifier's own
    per-(environment, step) sums of the raw probe beliefs) matched the export's curves, bands and per-resample
    crossings to floating-point precision.
  Orders re-checked over 21 bootstrap seeds (the module's 20261101 and 1-20; order_check, printed by numbers; the
    module's seed reproduces every event): every "<" of the timeline and the orders of findings 1, 4, 6 and 7 hold in
    all 21 (smallest margin, B's 5% bound minus A's 95% bound: 21 episodes, the corner splits' TV half-way after their
    matched control); the same diagonal's 50% before the one-variable / matched holding 50% only in 8 / 7 ("~").
  Every range claim taken from the numbers command over all 58 checkpoints; pyflakes clean.

CAVEATS: one seed (20261029), one world; bands and intervals are environment-bootstrap intervals (50 environments), not
seed variability. Figures use 58 of the 114 checkpoints (every second plus the last; the CPU replay costs ~10-20 s per
checkpoint; probe statistics exist for all 114), so crossings near passing naive are resolved only to the 720 / 880 /
1,040 checkpoints. The forward set is selected on Bayes's splits (every statistic conditions on Bayes's state at the
step, so the selection only sets the counts; the corner answers use only the 2,400 unselected corner trajectories).
The .15 / .6 / .05 split thresholds and the >= 2 cells / >= 30 deg move thresholds are the original rules (the maps
give threshold-free versions). The coupling angle ignores the sign (same-diagonal share in 6), and 22.5 deg is the
isotropic average, not an uncoupled reference (topic jumps builds a matched null). The matched-control result (4)
depends on the candidate pool (forward-set controls TV .055 at the end, probe controls .041). 90% crossings of slowly
saturating curves depend on the last checkpoints.

TIMELINE EXPORT (timeline.npz, for the cross-topic synthesis; the 58 checkpoints): 6 headline measures x {trained,
echo}: corner split held (joint; family movement), committed to a corner (answers), corner answer = truth (answers),
distance out of a split (movement), angle out of a split (movement), end-mass slope within steps (probes; movement).
value / lo / hi = the curve and its 5-95% band; bayes / naive = references; level = half-way from the Trained RNN's
initial network to its end (committed: half-way back from its peak, after which it stays below), used for both
networks; half / half_lo / half_hi = the "stays past" crossing and its interval (NaN = never). Trained: 6,192
(5,417-7,736) / 9,597 (8,416-10,656) / 1,361 (1,020-1,463) / 2,644 (2,428-2,861) / 6,327 (5,405-8,003) / 3,509
(2,817-3,837). The ESN never reaches any of these levels (for "committed", more than 5% of its resamples do fall
below .444 after 8,073, so half_lo is finite while half is NaN). synthesis.npz (below) supersedes it for the
cross-topic figure; its families follow td_tt_gap (committed = movement; angle, same diagonal, direction = coupling).

SYNTHESIS EXPORT (`synthesis` -> synthesis.npz via td_tt_gap.export; the same table and checks are printed by
`numbers`, section 9). 10 measures x {Trained RNN, ESN} on the 58 checkpoints (x = gap.train_x). Gap fraction f = (y -
naive) / (Bayes - naive), naive's and Bayes's levels on the same episodes; c0 / c50 = the training episode after which
f stays above 0 / .5 (passes naive's level / the naive-Bayes midpoint for good), 5-95% over gap.boot_counts() (1,000 x
50 environment counts, seed 20261103), every row paired: the probes' environment is row // 200, and the forward set's
environment e is the probes' e (td_grids_fwd drew the grid episodes from probes.npz's L[e] with its Z; checked, see
CHECKS). Each resampled curve is the statistic recomputed with the counts as environment weights: forward-set ratios
of sums weight each environment's sums; the maps' within-step slopes are refitted, step means included; naive's and
Bayes's levels are resampled with the same counts. Rows (family, shape): Trained RNN c0 / c50; ESN.
  corner answer = truth at T (answers, rise): 741 (489-813) / 1,575 (1,477-1,895); ESN never (f -.06 at the end).
  corner split held, joint (movement, rise): 574 (280-891) / 9,245 (7,693-10,112); ESN never. Naive's level is .004,
    so c0 marks leaving 0 (exact ties of the integer counts move 85 resamples under an "at or past" rule).
  held per variable, corner-split ends (movement, rise): 731 (454-891) / 4,983 (4,677-5,623); ESN never (c0 in 2% of
    the resamples). Naive's per-variable held levels are measured on the same var-steps (Bayes 1 by definition).
  held per variable, one-variable end splits (movement, rise): 810 (391-1,153) / 5,082 (3,943-7,571); ESN never (c0 in
    17%).
  distance out of a corner split (movement, rise): 1,203 (998-1,330) / 3,021 (2,815-3,511); ESN never.
  coupling angle out of a corner split (coupling, rise): 1,522 (10-2,356) / 11,103 (8,969-12,555); ESN never (c0 in
    47%: it ends at naive's level, f +.002). c0 is poorly resolved: the curve starts at naive's level (15.8 vs 16.6
    deg), and resamples past it at every checkpoint give the first checkpoint.
  same-diagonal share of moves out of corner splits (coupling, rise): 97 (51-285) / 2,747 (2,111-3,447); ESN c0 3,207
    (53-16,998; all resamples cross), f +.12 at the end, c50 never.
  end-mass slope within steps, maps (movement, rise): 748 (538-888) / 3,936 (3,635-4,854); ESN never (c0 in 32%).
  direction slope within steps, maps (coupling, rise): 2,454 (1,991-2,874) / 10,113 (8,535-11,982); ESN never (c0 in
    1%). Naive's level is mostly its step-2 slope, so c0 depends on the weighting of steps (DEFINITIONS).
  committed to a corner (movement, return; naive .554, Bayes 0): the initial network commits nothing (f = 1), f < 0
    while the network commits more than naive, so c0 / c50 time the return: below naive's level for good at 5,460
    (3,555-9,323), below the midpoint (.277) at 18,673 (14,050-26,137). The ESN never commits more than naive (max
    .529), so its f > 0 at every checkpoint and its c0 is 10 (10-19,497): no event; c50 never.
Rows are put in order only where their intervals separate: the same diagonal's c0 (51-285) cannot be ordered against
the joint held share's (280-891; it marks leaving 0) or the angle's (10-2,356; poorly resolved), nor distance's c0
(998-1,330) against one-variable holding's (391-1,153).
Agreement with this topic's own crossings: where the definitions coincide, the numbers coincide: the slopes' c0 =
"passes naive's slope (paired)" in finding 7, 748 and 2,454 (intervals 538-888 vs 562-893 and 1,991-2,874 vs
1,989-2,858: other resamples, seed 20261101). The c50s come later than the topic's 50% crossings because the
naive-Bayes midpoint lies above half-way from the initial network to the end: corner truth 1,575 vs 1,361 (level
.468 vs .381), joint held 9,245 vs 6,192 (.502 vs .415), per variable 4,983 / 5,082 vs 3,961 / 3,849, distance 3,021
vs 2,644 (1.98 vs 1.67 cells), angle 11,103 vs 6,327 (29.8 vs 26.6 deg), end-mass slope 3,936 vs 3,509 (.547 vs .473),
direction slope 10,113 vs 5,852 (.570 vs .426). The same-diagonal share, whose two levels nearly coincide (.766 vs
.771), gives 2,747 vs 2,796; committing's c0 / c50 (5,460 / 18,673) bracket its half-way back from the peak (9,597)
by construction (one curve, staying below .554 / .444 / .277 in turn).

Run from the repository root (CPU only):
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
    PY=/home/johns/anaconda3/envs/eureka/bin/python; S=main/REPLICATION/claude_plots/td_tt_wormholes.py
    $PY $S grid                               # forward set and its Bayes side (~30 s)
    $PY $S forward trained; $PY $S forward echo
                                              # the replay, 58 checkpoints per network, ~10-20 s each (~16 min per
                                              # network); reruns skip finished checkpoints; `forward run part nparts`
                                              # splits it into parts
    $PY $S stats                              # per-checkpoint statistics, probes and forward set (~3 min)
    $PY $S numbers                            # every number above
    $PY $S timeline                           # timeline.npz (the earlier export, unchanged)
    $PY $S synthesis                          # synthesis.npz (td_tt_gap.export), its table and checks (~5 s)
    $PY $S figure                             # both figures (~30 s)
Caches (~316 MB): main/DATA/experimental/training_dynamics/through_training/wormholes/: grid_bayes.npz (forward set:
indices, Bayes and naive log marginals, FR), fwd_<run>/ck_EEEEE.npz (per checkpoint: lq_tgt fp16 log marginals at the
stream-steps with any end split, m float32 posterior means, tv8 uint8 TV x 255, lqT fp16 log marginals at T, grid
checks), fwdstats_<run>.npz / fwd_refs.npz and probe_<run>.npz / probe_refs.npz (per-environment sums per checkpoint,
per (environment, step) sums for the slopes, per (step, bin) sums for the maps; Bayes's and naive's references),
timeline.npz (x, names, run, family, value / lo / hi, bayes, naive, level, half / half_lo / half_hi, level_def),
synthesis.npz (td_tt_gap.export: names, run, family, shape, sample, paired, note, x, value / lo / hi, naive, bayes,
frac / frac_lo / frac_hi, c0 / c50 with their intervals, crossing shares and per-resample crossings)."""
import os
import sys
import glob
import time
import numpy as np
from scipy.special import logsumexp

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
GRIDS = os.path.join(TD, "summary_data", "grids")
OUT = os.path.join(TD, "through_training", "wormholes")
RUNS = ("trained", "echo")
NAME = {"trained": "Trained RNN", "echo": "ESN"}
COL = {"trained": "#1f77b4", "echo": "#ff7f0e"}
T, R, NE = 30, 10, 50
V = np.arange(R, dtype=np.float64)
EDGE, ENDS, SETTLED = .15, .6, .05     # wormhole_marginals.py: both ends > .15, end mass > .6; not split: end < .05
STEPS = (1, 16)                         # 0-based [1, 16) = steps 2-16, the original window (wormhole_marginals.py)
STEPS_ALL = (1, 30)                     # steps 2-30 (robustness)
CORNERS = (0, 9, 90, 99)                # cells r0 * 10 + r1: (0, 0), (0, 9), (9, 0), (9, 9)
CORNER_STREAMS = 12                     # unselected corner episodes per (environment, corner cell) in the forward set
A_BIG, PHI_DIAG, PHI_AXIS = 2., 30., 15.  # Bayes's large moves (cells); diagonal / one-axis coupling angles (degrees)
PHI_EDGES = np.arange(0., 46., 5.)      # Bayes's coupling angle bins (degrees)
E_EDGES = np.array([0, .01, .03, .06, .1, .15, .2, .3, .4, .5 + 1e-9])   # smaller-end mass bins (heat map)
NEB, NPB = len(E_EDGES) - 1, len(PHI_EDGES) - 1
FR_STEP, H_STEP = .25, .2               # control matching: log10 FR bins, entropy bins (nats)
E_STEP = .05                            # step x Bayes-e_v matching of the one-variable splits: e_v bins
MIN_STEP_N = 30                         # per-step slopes are averaged over the steps with >= 30 points
X0 = 10                                 # the initial network on the log axis


def train_x(ep):
    ep = np.asarray(ep)
    return np.maximum(ep - ep // 5, X0).astype(float)


def checkpoints():
    """the 114 checkpoints (raw episodes) both runs share"""
    return sorted(int(os.path.basename(f)[3:8]) for f in glob.glob(os.path.join(AN, "trained", "ck_?????.npz")))


def fig_checkpoints():
    """the figures' checkpoints: every second of the 114 and the last (58; the CPU replay of the forward set costs
    ~10-15 s per checkpoint)"""
    c = checkpoints()
    if not c:
        raise FileNotFoundError("td_tt_wormholes: no stored checkpoints found (the figures replay the forward set on "
                                "them); fetch experimental/training_dynamics/<run>_s20261029/ckpt_ep*_net.pth first "
                                "(REPRODUCE.md: python main/data_tools.py fetch)")
    return sorted(set(c[::2]) | {c[-1]})


def norm(lq):
    """probabilities from log beliefs (..., R), renormalised (fp16 logs do not sum to 1)"""
    lq = np.asarray(lq, np.float64)
    return np.exp(lq - logsumexp(lq, -1, keepdims=True))


def ends(b):
    """per variable: smaller-end mass e = min(b(0), b(9)), Bayes-style split (e > EDGE and b(0) + b(9) > ENDS), settled
    (e < SETTLED); b (..., 2, R)"""
    lo, hi = b[..., 0], b[..., R - 1]
    e = np.minimum(lo, hi)
    return e, (e > EDGE) & (lo + hi > ENDS), e < SETTLED


def fr_sym(p, q, eps=1e-15):
    p, q = np.clip(p, eps, 1 - eps), np.clip(q, eps, 1 - eps)
    return .5 * ((p * np.log(p / q)).sum(-1) + (q * np.log(q / p)).sum(-1))


def entropy(b):
    return -(b * np.log(np.clip(b, 1e-300, 1))).sum(-1)


def step_mask(shape, steps):
    m = np.zeros(shape, bool)
    m[:, steps[0]:steps[1]] = True
    return m


# ═══════════════════════════════════════════════════════════════════
# Target sets (Bayes-defined) and matched controls
# ═══════════════════════════════════════════════════════════════════

def step_e_weights(e, tgt, src):
    """weights on the src var-levels (n, T, 2) that give them the tgt var-levels' histogram of step x Bayes's e_v (bins
    of E_STEP), and the tgt var-levels whose bin has src members (the covered part)"""
    tt = np.broadcast_to(np.arange(T)[None, :, None], e.shape)
    key = tt * 1000 + np.floor(e / E_STEP).astype(int)
    kt, ct = np.unique(key[tgt], return_counts=True)
    ks, inv, cs = np.unique(key[src], return_inverse=True, return_counts=True)
    pos = np.minimum(np.searchsorted(kt, ks), len(kt) - 1)
    w = np.zeros(e.shape)
    w[src] = np.where(kt[pos] == ks, ct[pos] / cs, 0.)[inv.ravel()]
    return w, tgt & np.isin(key, ks)


def target_sets(bB, fr, H, steps=STEPS):
    """Bayes-defined sets on (episode, step) and (episode, step, variable), within the steps window:
    two (n, T): corner split (both of Bayes's marginals split between their ends; wormhole_marginals.py's rule);
    two_v (n, T, 2): the var-levels of corner splits; one_v (n, T, 2): one-variable end split (this variable split,
    the other settled); cand (n, T, 2): neither variable split (candidates for the matched controls);
    w_two / w_one (n, T, 2): control weights matching the step x log10 FR x entropy histogram of two_v / one_v;
    w_one_m (n, T, 2): weights on one_v giving it two_v's step x Bayes-e_v histogram; two_cov: the two_v var-levels
    whose step x e_v bin has one-variable splits"""
    e, split, settled = ends(bB)
    sm = step_mask(split.shape[:2], steps)
    two = split.all(-1) & sm
    two_v = two[..., None] & split
    one_v = split & settled[..., ::-1] & sm[..., None]
    cand = (~split.any(-1) & sm)[..., None] & np.ones((1, 1, 2), bool)
    fb = np.floor(np.log10(np.clip(fr, 1e-6, None)) / FR_STEP).astype(int)
    hb = np.floor(H / H_STEP).astype(int)
    tt = np.broadcast_to(np.arange(T)[None, :, None], fr.shape)
    key = (tt * 1000 + (fb + 500)) * 1000 + hb
    out = dict(two=two, two_v=two_v, one_v=one_v, cand=cand)
    kc = key[cand]
    uc, inv_c, cc = np.unique(kc, return_inverse=True, return_counts=True)
    for nm, tm in (("two", two_v), ("one", one_v)):
        ut, ct = np.unique(key[tm], return_counts=True)
        pos = np.searchsorted(uc, ut)
        ok = (pos < len(uc)) & (uc[np.minimum(pos, len(uc) - 1)] == ut)
        per = np.zeros(len(uc))
        per[pos[ok]] = ct[ok] / cc[pos[ok]]
        w = np.zeros(fr.shape)
        w[cand] = per[inv_c.ravel()]
        out[f"w_{nm}"] = w
        out[f"cover_{nm}"] = ct[ok].sum() / max(tm.sum(), 1)
    out["w_one_m"], out["two_cov"] = step_e_weights(e, two_v, one_v)
    out["cover_one_m"] = out["two_cov"].sum() / max(two_v.sum(), 1)
    return out


def bayes_jumps(m):
    """posterior-mean moves into steps 2-30: d (n, T-1, 2), amplitude (cells), coupling angle (degrees from the
    nearest axis: 0 one variable, 45 diagonal)"""
    d = np.diff(m, axis=1)
    a = np.hypot(d[..., 0], d[..., 1])
    ad = np.abs(d)
    phi = np.degrees(np.arctan2(ad.min(-1), ad.max(-1)))
    return d, a, phi


def corner_category(ans, r):
    """answer at T for a corner truth: 0 truth, 1 adjacent corner (one variable at the opposite end, the other right),
    2 partner corner (both at the opposite ends), 3 anything else"""
    right = ans == r
    flip = ans == R - 1 - r
    cat = np.full(len(r), 3)
    cat[right.all(1)] = 0
    cat[(right[:, 0] & flip[:, 1]) | (flip[:, 0] & right[:, 1])] = 1
    cat[flip.all(1)] = 2
    return cat


# ═══════════════════════════════════════════════════════════════════
# Per-checkpoint statistics as per-environment sums
# ═══════════════════════════════════════════════════════════════════

def _env_sum(env, mask, val=None):
    """sum of val (or count) over mask per environment; mask / val over (n, ...) with env (n,)"""
    ix = np.nonzero(mask)
    v = np.ones(len(ix[0])) if val is None else np.asarray(val, np.float64)[ix]
    return np.bincount(env[ix[0]], v, minlength=NE)


def _env_step_sum(env, mask, val=None):
    """sum of val (or count) over mask per (environment, step): mask (n, T', ...) -> (NE, T')"""
    ix = np.nonzero(mask)
    v = np.ones(len(ix[0])) if val is None else np.asarray(val, np.float64)[ix]
    tn = mask.shape[1]
    return np.bincount(env[ix[0]] * tn + ix[1], v, minlength=NE * tn).reshape(NE, tn)


def _step_bin_sum(mask, kbin, nb, val=None):
    """sum of val (or count) over mask per (step, bin of Bayes's value kbin): mask (n, T', ...) -> (T', nb), pooled over
    environments"""
    ix = np.nonzero(mask)
    v = np.ones(len(ix[0])) if val is None else np.asarray(val, np.float64)[ix]
    tn = mask.shape[1]
    return np.bincount(ix[1] * nb + kbin[ix], v, minlength=tn * nb).reshape(tn, nb)


def split_stats(b, bB, env, r, S, tv=None):
    """per-environment sums at Bayes's split sets S (target_sets) for marginals b (n, T, 2, R; only the target
    stream-steps are read) and their total variation tv (n, T, 2) from Bayes's marginals (computed if not given)"""
    e, _, _ = ends(b)
    held_v = e > EDGE
    joint = held_v.all(-1)
    ans = np.nan_to_num(b, nan=-1.).argmax(-1)                            # (n, T, 2)
    corner = np.isin(ans, (0, R - 1)).all(-1)
    truth = (ans == r[:, None, :]).all(-1)
    if tv is None:
        tv = .5 * np.abs(b - bB).sum(-1)                                  # (n, T, 2)
    two, two_v, one_v, cand = S["two"], S["two_v"], S["one_v"], S["cand"]
    return dict(
        joint_held=_env_sum(env, two, joint), commit=_env_sum(env, two, corner & ~joint),
        truth=_env_sum(env, two, truth), held_two=_env_sum(env, two_v, held_v), held_one=_env_sum(env, one_v, held_v),
        held_one_m=_env_sum(env, one_v, S["w_one_m"] * held_v), held_two_c=_env_sum(env, S["two_cov"], held_v),
        e_two=_env_sum(env, two_v, e), e_one=_env_sum(env, one_v, e), tv_two=_env_sum(env, two_v, tv),
        tv_one=_env_sum(env, one_v, tv), tv_ctl_two=_env_sum(env, cand, S["w_two"] * tv),
        tv_ctl_one=_env_sum(env, cand, S["w_one"] * tv))


def split_counts(env, S):
    return dict(two=_env_sum(env, S["two"]), two_v=_env_sum(env, S["two_v"]), one_v=_env_sum(env, S["one_v"]),
                w_two=_env_sum(env, S["cand"], S["w_two"]), w_one=_env_sum(env, S["cand"], S["w_one"]),
                w_one_m=_env_sum(env, S["one_v"], S["w_one_m"]), two_c=_env_sum(env, S["two_cov"]))


def jump_masks(mB):
    d, a, phi = bayes_jumps(mB)
    pbin = np.minimum(np.floor(phi / 5).astype(int), NPB - 1)
    big = a >= A_BIG
    return dict(d=d, a=a, phi=phi, pbin=pbin, big=big, diag=big & (phi >= PHI_DIAG), axis=big & (phi <= PHI_AXIS))


def jump_stats(m, J, env):
    """the network's moves (posterior means m (n, T, 2)) at Bayes's moves J (jump_masks): mean angle / size at Bayes's
    large diagonal and one-axis moves (per environment); for the maps, per (environment, step) sums of the angle and of
    angle x Bayes's angle at every move >= A_BIG, and per (step, Bayes-angle bin) sums of the angle"""
    _, a, phi = bayes_jumps(m)
    big = J["big"]
    return {"phi_diag": _env_sum(env, J["diag"], phi), "phi_axis": _env_sum(env, J["axis"], phi),
            "amp_diag": _env_sum(env, J["diag"], a), "amp_axis": _env_sum(env, J["axis"], a),
            "phi_st": _env_step_sum(env, big, phi), "phi_dot_st": _env_step_sum(env, big, phi * J["phi"]),
            "phi_tbin": _step_bin_sum(big, J["pbin"], NPB, phi)}


def jump_counts(J, env):
    big = J["big"]
    return dict(diag=_env_sum(env, J["diag"]), axis=_env_sum(env, J["axis"]), big=_env_sum(env, big),
                big_st=_env_step_sum(env, big), phiB_st=_env_step_sum(env, big, J["phi"]),
                phiB2_st=_env_step_sum(env, big, J["phi"] ** 2), tbin=_step_bin_sum(big, J["pbin"], NPB))


def _emap_masks(bB, steps=STEPS_ALL):
    eB, _, _ = ends(bB)
    sm = step_mask(eB.shape[:2], steps)[..., None] & np.ones((1, 1, 2), bool)
    k = np.minimum(np.digitize(eB, E_EDGES) - 1, NEB - 1)
    return eB, sm, k


def emap_stats(b, bB, env):
    """the network's smaller-end mass e_N(v) at every var-level of steps 2-30 (no threshold on the events): per
    (environment, step) sums of e_N and e_N x e_B (the within-step slope), per (step, Bayes-e_B bin) sums of e_N"""
    eN, _, _ = ends(b)
    eB, sm, k = _emap_masks(bB)
    return dict(e_st=_env_step_sum(env, sm, eN), e_dot_st=_env_step_sum(env, sm, eN * eB),
                e_tbin=_step_bin_sum(sm, k, NEB, eN))


def emap_counts(bB, env):
    eB, sm, k = _emap_masks(bB)
    return dict(e_st=_env_step_sum(env, sm), e_B_st=_env_step_sum(env, sm, eB), e_B2_st=_env_step_sum(env, sm, eB ** 2),
                e_tbin=_step_bin_sum(sm, k, NEB))


def corner_stats(lqT, r, env, sel):
    cat = corner_category(lqT[sel].argmax(-1), r[sel])
    return {"corner_cat": np.stack([np.bincount(env[sel], cat == k, minlength=NE) for k in range(4)], 1)}


# ═══════════════════════════════════════════════════════════════════
# Probes (10,000 episodes, every checkpoint, stored log beliefs)
# ═══════════════════════════════════════════════════════════════════

def probe_side():
    P = np.load(os.path.join(AN, "probes.npz"))
    bB, bN = norm(P["log_jb"]), norm(P["log_nb"])
    env, r = P["env"], P["r"]
    H = entropy(bB)
    S = target_sets(bB, P["fr"], H)
    S_all = target_sets(bB, P["fr"], H, STEPS_ALL)
    J = jump_masks(bB @ V)
    sel = np.isin(P["cell"], CORNERS)
    return dict(P=P, bB=bB, bN=bN, env=env, r=r, S=S, S_all=S_all, J=J, sel=sel)


def _stats_all(b, D):
    """every statistic for marginals b (n, T, 2, R) on the side D (probes or forward set)"""
    out = {k: v for k, v in split_stats(b, D["bB"], D["env"], D["r"], D["S"]).items()}
    out.update({k + "_all": v for k, v in split_stats(b, D["bB"], D["env"], D["r"], D["S_all"]).items()})
    out.update(corner_stats(np.log(np.clip(b[:, -1], 1e-300, 1)), D["r"], D["env"], D["sel"]))
    if "J" in D:
        out.update(jump_stats(b @ V, D["J"], D["env"]))
        out.update(emap_stats(b, D["bB"], D["env"]))
    return out


def _counts_all(D):
    out = {"n_" + k: v for k, v in split_counts(D["env"], D["S"]).items()}
    out.update({"n_" + k + "_all": v for k, v in split_counts(D["env"], D["S_all"]).items()})
    out["n_corner"] = np.bincount(D["env"][D["sel"]], minlength=NE)
    for k in ("cover_two", "cover_one", "cover_one_m"):
        out[k] = D["S"][k]
    if "J" in D:
        out.update({"n_jump_" + k: v for k, v in jump_counts(D["J"], D["env"]).items()})
        out.update({"n_" + k: v for k, v in emap_counts(D["bB"], D["env"]).items()})
    return out


def probe_stats(run):
    """per checkpoint, every statistic on the probes (cache probe_<run>.npz; references probe_refs.npz)"""
    path = os.path.join(OUT, f"probe_{run}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    D = probe_side()
    rows = {}
    t0 = time.time()
    eps = checkpoints()
    for ep in eps:
        z = np.load(os.path.join(AN, run, f"ck_{ep:05d}.npz"))
        for k, v in _stats_all(norm(z["logq"]), D).items():
            rows.setdefault(k, []).append(v)
        rows.setdefault("acc", []).append(z["acc"][-1])
    out = {k: np.array(v) for k, v in rows.items()}
    out.update(_counts_all(D), episode=np.array(eps), x=train_x(eps))
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    print(f"probe stats {run}: {len(eps)} checkpoints, {time.time() - t0:.0f} s -> {path}", flush=True)
    return out


def probe_refs():
    path = os.path.join(OUT, "probe_refs.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    D = probe_side()
    out = {}
    for nm, b in (("bayes", D["bB"]), ("naive", D["bN"])):
        out.update({f"{nm}_{k}": v for k, v in _stats_all(b, D).items()})
    out["naive_acc"] = D["P"]["right_naive"][np.arange(len(D["env"])), D["P"]["goal"]].mean()
    out["bayes_acc"] = D["P"]["right_bayes"][np.arange(len(D["env"])), D["P"]["goal"]].mean()
    out.update(_counts_all(D))
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    return out


# ═══════════════════════════════════════════════════════════════════
# Forward set: the grid episodes (50 environments x 100 cells x 24 trajectories), selected, replayed on CPU
# ═══════════════════════════════════════════════════════════════════

def grid_side():
    """the forward set and its Bayes side (cache grid_bayes.npz): every grid episode with a corner split or a
    one-variable end split at any step 2-30 (Bayes), plus the corner cells' first CORNER_STREAMS trajectories
    (unselected; the corner answers use only these)"""
    path = os.path.join(OUT, "grid_bayes.npz")
    if not os.path.exists(path):
        P = np.load(os.path.join(AN, "probes.npz"))
        E = np.load(os.path.join(GRIDS, "fwd_episodes.npz"))
        L, env, obs, cell = P["L"], E["env"], E["obs"], E["cell"]
        S_ = len(env) // (NE * R * R)
        stream = np.arange(len(env)) % S_
        keep = np.isin(cell, CORNERS) & (stream < CORNER_STREAMS)
        parts = {}
        for e in range(NE):
            ix = np.flatnonzero(env == e)
            o = obs[ix].astype(np.float64)
            ll = np.einsum("nti,ijk->ntjk", o, np.log(L[e])) + np.einsum("nti,ijk->ntjk", 1 - o, np.log1p(-L[e]))
            lam = np.cumsum(ll, 1)
            lp = lam - logsumexp(lam.reshape(len(ix), T, -1), -1)[..., None, None]
            ljb = np.stack([logsumexp(lp, 3), logsumexp(lp, 2)], 2)
            Lb = np.stack([L[e].mean(2), L[e].mean(1)], 1)
            lpx = np.einsum("nti,ikr->ntkr", o, np.log(Lb)) + np.einsum("nti,ikr->ntkr", 1 - o, np.log1p(-Lb))
            cl = np.cumsum(lpx, 1)
            lnb = cl - logsumexp(cl, -1, keepdims=True)
            bB = np.exp(ljb)
            _, split, settled = ends(bB)
            sm = step_mask(split.shape[:2], STEPS_ALL)
            two = split.all(-1) & sm
            one = (split & settled[..., ::-1] & sm[..., None]).any(-1)
            k = keep[ix] | (two | one).any(1)
            parts.setdefault("idx", []).append(ix[k])
            parts.setdefault("log_jb", []).append(ljb[k].astype(np.float32))
            parts.setdefault("log_nb", []).append(lnb[k].astype(np.float32))
            parts.setdefault("fr", []).append(np.stack([fr_sym(bB[k][:, :, v], np.exp(lnb[k][:, :, v]))
                                                        for v in range(2)], -1).astype(np.float32))
        out = {k: np.concatenate(v) for k, v in parts.items()}
        idx = out["idx"]
        out.update(env=env[idx], cell=cell[idx], r=E["r"][idx], stream=stream[idx], corner_sample=keep[idx],
                   obs=obs[idx], Z=E["Z"])
        os.makedirs(OUT, exist_ok=True)
        np.savez(path, **out)
        print(f"forward set: {len(idx)} of {len(env)} grid episodes ({keep.sum()} unselected corner episodes)")
    G = dict(np.load(path))
    G["bB"] = norm(G["log_jb"])
    G["bN"] = norm(G["log_nb"])
    return G


def fwd_side(G=None):
    G = G or grid_side()
    H = entropy(G["bB"])
    D = dict(bB=G["bB"], bN=G["bN"], env=G["env"], r=G["r"], sel=G["corner_sample"])
    D["S"] = target_sets(G["bB"], G["fr"], H)
    D["S_all"] = target_sets(G["bB"], G["fr"], H, STEPS_ALL)
    D["X"] = crossing_masks(G["bB"])
    D["dB"], D["aB"], _ = bayes_jumps(G["bB"] @ V)
    return D


_PAT = np.array([[(k >> i) & 1 for i in range(5)] for k in range(32)], np.float32)


def replay(w, obs, env, Z):
    """the network's marginal log beliefs (n, T, 2, R) (float64) for observations obs (n, T, 5) in environments env,
    from a checkpoint's weights w (fp32, weights only; gate order i, f, g, o = td_analysis's hand LSTM; the read-in
    and input weights are applied once per distinct (observation pattern, environment))"""
    import torch
    p = {k: v.float() for k, v in w.items()}
    with torch.no_grad():
        Zt = torch.as_tensor(Z).reshape(NE, -1)
        pat = torch.as_tensor(_PAT)
        inp = torch.cat([pat[None].expand(NE, 32, 5), Zt[:, None].expand(NE, 32, Zt.shape[-1])], -1)
        x = torch.relu(inp @ p["classifier_readin.weight"].T + p["classifier_readin.bias"])
        table = x @ p["LSTM.weight_ih_l0"].T + p["LSTM.bias_ih_l0"] + p["LSTM.bias_hh_l0"]   # (NE, 32, 4H)
        code = torch.as_tensor((obs * (1 << np.arange(5))).sum(-1).astype(np.int64))
        ee = torch.as_tensor(env.astype(np.int64))
        n, H = len(obs), p["LSTM.weight_hh_l0"].shape[1]
        h = p["STM"].reshape(1, H).expand(n, H)
        c = p["LTM"].reshape(1, H).expand(n, H)
        Whh = p["LSTM.weight_hh_l0"].T.contiguous()
        Wro, bro = p["classifier_readout.weight"].T.contiguous(), p["classifier_readout.bias"]
        lg = torch.zeros(n, T, Wro.shape[1])
        for t in range(T):
            pre = torch.addmm(table[ee, code[:, t]], h, Whh)
            i, f, g, o = pre.chunk(4, -1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            lg[:, t] = torch.addmm(bro, h, Wro)
        cum = lg.double().cumsum(1).reshape(n, T, 2, R)
        return torch.log_softmax(cum, -1).numpy()


def tgt_mask(bB):
    """(n, T): stream-steps 2-30 where any of Bayes's marginals is split between its ends (every split target lies
    in it); the forward records keep the full marginals only there"""
    _, split, _ = ends(bB)
    return split.any(-1) & step_mask(split.shape[:2], STEPS_ALL)


def forward(run, part=0, nparts=1):
    """replay the forward set at every checkpoint of this part (every nparts-th, offset part) -> fwd_<run>/ck_EEEEE.npz:
    lq_tgt fp16 log marginals at tgt_mask's stream-steps, m (n, T, 2) float32 posterior means, tv8 (n, T, 2) uint8
    total variation from Bayes's marginals x 255 (rounded), lqT (n, 2, R) fp16 log marginals at T; reruns skip
    existing files; at the 19 grid checkpoints the posterior means and the answers at T are checked against
    summary_data/grids/fwd_<run>.npz (td_grids_fwd.py, GPU replay of the same episodes)"""
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    G = grid_side()
    tm = tgt_mask(G["bB"])
    d = os.path.join(OUT, f"fwd_{run}")
    os.makedirs(d, exist_ok=True)
    F = np.load(os.path.join(GRIDS, f"fwd_{run}.npz"))
    gep = list(F["ep"])
    eps = fig_checkpoints()[part::nparts]
    t0 = time.time()
    for ep in eps:
        path = os.path.join(d, f"ck_{ep:05d}.npz")
        if os.path.exists(path):
            continue
        w = torch.load(os.path.join(TD, f"{run}_s20261029", f"ckpt_ep{ep:05d}_net.pth"), map_location="cpu",
                       weights_only=False)["weights"]
        lq = replay(w, G["obs"], G["env"], G["Z"])
        b = np.exp(lq)
        m = b @ V
        tv = .5 * np.abs(b - G["bB"]).sum(-1)
        rec = dict(lq_tgt=lq[tm].astype(np.float16), m=m.astype(np.float32), lqT=lq[:, -1].astype(np.float16),
                   tv8=np.rint(np.clip(tv, 0, 1) * 255).astype(np.uint8), episode=ep)
        if ep in gep:
            k = gep.index(ep)
            rec["check_m"] = np.abs(m - F["m"][k][G["idx"]].astype(np.float64)).max()
            rec["check_a"] = (lq[:, -1].argmax(-1) == F["a"][k][G["idx"]]).all(-1).mean()
        np.savez(path + ".tmp.npz", **rec)
        os.replace(path + ".tmp.npz", path)
        extra = f"; grid check: |m| {rec['check_m']:.4f}, answers {rec['check_a']:.4f}" if "check_m" in rec else ""
        print(f"{time.strftime('%H:%M:%S')} {run} ep {ep}: {time.time() - t0:.0f} s{extra}", flush=True)


def crossing_masks(bB):
    """Bayes's moves out of a corner split (n, T - 1), indexed like bayes_jumps (move into step t + 2): the stream-step
    before the move is a corner split (steps 2-29) and the move is large (>= A_BIG cells) and diagonal (>= PHI_DIAG)"""
    _, split, _ = ends(bB)
    src = split.all(-1)[:, :-1].copy()
    src[:, 0] = False
    _, a, phi = bayes_jumps(bB @ V)
    return src & (a >= A_BIG) & (phi >= PHI_DIAG)


def _fwd_stats_all(b, tv, lqT, m, D):
    """the forward set's statistics: split sets (marginals b valid at tgt_mask, NaN elsewhere; tv everywhere), the
    corner answers at T (unselected corner sample) and the moves (posterior means m) out of corner splits; tv_med = the
    original summary (median over corner-split stream-steps of TV averaged over the two variables)"""
    out = {k: v for k, v in split_stats(b, D["bB"], D["env"], D["r"], D["S"], tv).items()}
    out.update({k + "_all": v for k, v in split_stats(b, D["bB"], D["env"], D["r"], D["S_all"], tv).items()})
    out.update(corner_stats(lqT, D["r"], D["env"], D["sel"]))
    d, a, phi = bayes_jumps(m)
    dB = D["dB"]
    proj = (d * dB).sum(-1) / np.maximum(D["aB"], 1e-12)                  # the move's component along Bayes's (cells)
    same = np.sign(d[..., 0] * d[..., 1]) == np.sign(dB[..., 0] * dB[..., 1])  # along the same diagonal as Bayes
    out.update(phi_out=_env_sum(D["env"], D["X"], phi), amp_out=_env_sum(D["env"], D["X"], a),
               proj_out=_env_sum(D["env"], D["X"], proj), same_out=_env_sum(D["env"], D["X"], same),
               tv_med=np.median(tv[D["S"]["two"]].mean(-1)))
    return out


def fwd_stats(run):
    """per checkpoint, every statistic on the forward set (cache fwdstats_<run>.npz)"""
    path = os.path.join(OUT, f"fwdstats_{run}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    G = grid_side()
    D = fwd_side(G)
    tm = tgt_mask(G["bB"])
    eps = fig_checkpoints()
    files = [os.path.join(OUT, f"fwd_{run}", f"ck_{ep:05d}.npz") for ep in eps]
    missing = [f for f in files if not os.path.exists(f)]
    assert not missing, f"{len(missing)} forward checkpoints missing, e.g. {missing[0]}"
    rows = {}
    b = np.full(G["bB"].shape, np.nan)
    for f in files:
        z = np.load(f)
        assert z["m"].dtype == np.float32, f"{f}: posterior means not float32 (rerun forward)"
        b[tm] = norm(z["lq_tgt"])
        for k, v in _fwd_stats_all(b, z["tv8"] / 255., z["lqT"].astype(np.float64), z["m"].astype(np.float64),
                                   D).items():
            rows.setdefault(k, []).append(v)
        for k in ("check_m", "check_a"):
            rows.setdefault(k, []).append(float(z[k]) if k in z else np.nan)
    out = {k: np.array(v) for k, v in rows.items()}
    out.update(_counts_all(D), episode=np.array(eps), x=train_x(eps), n_out=_env_sum(D["env"], D["X"]))
    np.savez(path, **out)
    return out


def fwd_refs():
    path = os.path.join(OUT, "fwd_refs.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    D = fwd_side()
    out = {}
    for nm, b in (("bayes", D["bB"]), ("naive", D["bN"])):
        tv = .5 * np.abs(b - D["bB"]).sum(-1)
        out.update({f"{nm}_{k}": v for k, v in
                    _fwd_stats_all(b, tv, np.log(np.clip(b[:, -1], 1e-300, 1)), b @ V, D).items()})
    out.update(_counts_all(D))
    out["n_out"] = _env_sum(D["env"], D["X"])
    np.savez(path, **out)
    return out


# ═══════════════════════════════════════════════════════════════════
# Aggregation, bootstrap over environments, slopes, timing
# ═══════════════════════════════════════════════════════════════════

NBOOT, BOOT_SEED = 1000, 20261101


def boot_weights(nb=NBOOT, seed=BOOT_SEED):
    """(nb, NE) multinomial weights: the 50 environments resampled with replacement"""
    idx = np.random.default_rng(seed).integers(0, NE, (nb, NE))
    return np.stack([np.bincount(i, minlength=NE) for i in idx]).astype(np.float64)


def agg(num, den, W=None):
    """per-environment sums -> pooled ratio: num (K, NE, ...) / den (NE, ...) -> (K, ...); with bootstrap weights W
    (B, NE) -> (B, K, ...)"""
    if W is None:
        return num.sum(1) / den.sum(0)
    return np.einsum("be,ke...->bk...", W, num) / np.einsum("be,e...->b...", W, den)[:, None]


def ref(R_, key, den):
    """a reference observer's pooled value from its per-environment sums"""
    return R_[key].sum(0) / den.sum(0)


def _ref_rows(R_, nm):
    """a reference observer's statistics as a one-checkpoint table (keys without the 'bayes_' / 'naive_' prefix)"""
    p = nm + "_"
    return {k[len(p):]: np.asarray(v)[None] for k, v in R_.items() if k.startswith(p)}


def step_slope(dot, sN, n, sB, sB2, W=None, how="within", min_n=1):
    """least-squares slope of the network's value on Bayes's from per-(environment, step) sums: dot = Sum net x Bayes,
    sN = Sum net (K, NE, T'); n, sB = Sum Bayes, sB2 = Sum Bayes^2 (NE, T'). how: 'within' = one fit after removing each
    step's mean from both (step means pooled over environments); 'per_step' = the mean over steps of per-step slopes
    (steps with >= min_n points); 'pooled' = one fit over all steps (counts the drift of the step means). With
    bootstrap weights W (B, NE) -> (B, K), else (K,)"""
    if W is None:
        N_, SB, SB2 = n.sum(0), sB.sum(0), sB2.sum(0)
        D_, SN = dot.sum(1), sN.sum(1)
    else:
        N_, SB, SB2 = (W @ n)[:, None], (W @ sB)[:, None], (W @ sB2)[:, None]
        D_, SN = np.einsum("be,ket->bkt", W, dot), np.einsum("be,ket->bkt", W, sN)
    if how == "pooled":
        N_, SB, SB2, D_, SN = (a.sum(-1, keepdims=True) for a in (N_, SB, SB2, D_, SN))
    ok = N_ >= max(min_n, 1)
    Nz = np.where(ok, N_, 1.)
    num = np.where(ok, D_ - SN * SB / Nz, 0.)
    den = np.where(ok, SB2 - SB ** 2 / Nz, 0.)
    if how == "per_step":
        good = ok & (den > 0)
        s = np.where(good, num / np.where(good, den, 1.), 0.)
        return s.sum(-1) / good.sum(-1)
    return num.sum(-1) / den.sum(-1)


def centred_map(tbin, st, n_tbin, n_st):
    """heat-map colour: per bin of Bayes's value, the mean over the bin's points of (the network's value minus its mean
    at that step) plus the network's mean over all steps, so the bins differ only within steps. tbin (K, T', nb) sums per
    (step, bin); st (K, NE, T') sums per (environment, step); n_tbin (T', nb), n_st (NE, T') counts -> (K, nb)"""
    S_t, N_t = st.sum(1), n_st.sum(0)
    m_t = S_t / np.maximum(N_t, 1)
    num = (tbin - n_tbin[None] * m_t[..., None]).sum(1)
    return num / n_tbin.sum(0) + (S_t.sum(-1) / N_t.sum())[:, None]


def maps_slope(Z, C, kind, W=None, how="within", min_n=None):
    """the maps' summary slope (network on Bayes) for per-checkpoint probe statistics Z, with the probe counts C (the
    per-step average uses the steps with >= MIN_STEP_N points unless min_n is given; the within-step fit uses every
    step)"""
    mn = (MIN_STEP_N if how == "per_step" else 1) if min_n is None else min_n
    if kind == "e":
        return step_slope(Z["e_dot_st"], Z["e_st"], C["n_e_st"], C["n_e_B_st"], C["n_e_B2_st"], W, how, mn)
    return step_slope(Z["phi_dot_st"], Z["phi_st"], C["n_jump_big_st"], C["n_jump_phiB_st"], C["n_jump_phiB2_st"], W,
                      how, mn)


def maps_heat(Z, C, kind):
    if kind == "e":
        return centred_map(Z["e_tbin"], Z["e_st"], C["n_e_tbin"], C["n_e_st"])
    return centred_map(Z["phi_tbin"], Z["phi_st"], C["n_jump_tbin"], C["n_jump_big_st"])


def _step_moments(Z, C, kind, k=0):
    """per step, pooled over environments, at checkpoint k of the probe statistics Z: the count n_t and the centred
    sums num_t = Sigma_i (y_i^net - ybar_t^net)(y_i^B - ybar_t^B), den_t = Sigma_i (y_i^B - ybar_t^B)^2, and the offset
    from array index to 1-based step (e: (NE, 30) sums indexed by step - 1, step 1 empty; phi: (NE, 29) sums indexed
    by the step moved into - 2)"""
    if kind == "e":
        dot, sN, n, sB, sB2, off = Z["e_dot_st"], Z["e_st"], C["n_e_st"], C["n_e_B_st"], C["n_e_B2_st"], 1
    else:
        dot, sN, n, sB, sB2, off = (Z["phi_dot_st"], Z["phi_st"], C["n_jump_big_st"], C["n_jump_phiB_st"],
                                    C["n_jump_phiB2_st"], 2)
    N_, SB, SB2, D_, SN = n.sum(0), sB.sum(0), sB2.sum(0), dot[k].sum(0), sN[k].sum(0)
    Nz = np.maximum(N_, 1)
    return N_, D_ - SN * SB / Nz, SB2 - SB ** 2 / Nz, off


def step_parts(Z, C, kind, k=0, groups=((2, 2), (3, 5), (6, 30))):
    """the within-step slope as a weighted mean of per-step slopes: beta = Sigma_t w_t beta_t with w_t = Sigma_i
    (y_i^B - ybar_t^B)^2 / Sigma_t Sigma_i (y_i^B - ybar_t^B)^2. For groups of steps (first, last; 1-based) at
    checkpoint k of the probe statistics Z -> [(share of the points, share of the weight w, the group's own
    within-step slope)]"""
    N_, num, den, off = _step_moments(Z, C, kind, k)
    out = []
    for a, b in groups:
        s = slice(a - off, b - off + 1)
        out.append((N_[s].sum() / N_.sum(), den[s].sum() / den.sum(), num[s].sum() / den[s].sum()))
    return out


def equal_weight(Z, C, kind, k=0):
    """the mean over steps of per-step slopes (equal weight per step) at checkpoint k of the probe statistics Z, over
    the steps with >= MIN_STEP_N points and over every step that has a slope (>= 1 point, Bayes's values not all
    equal): [(value, number of steps)] in that order"""
    N_, _, den, _ = _step_moments(Z, C, kind, k)
    return [(float(maps_slope(Z, C, kind, how="per_step", min_n=mn)[k]), int(((N_ >= mn) & (den > 0)).sum()))
            for mn in (MIN_STEP_N, 1)]


def literal_slope(Z, C, kind, k=0):
    """Sigma_t cov_t / Sigma_t var_t with each step's moments divided by its own count: the literal reading of the
    first version's title formula, which the code never computed (equal to the within-step slope when every step has
    the same count)"""
    N_, num, den, _ = _step_moments(Z, C, kind, k)
    ok = N_ > 0
    return float((num[ok] / N_[ok]).sum() / (den[ok] / N_[ok]).sum())


def cross_level(x, y, lev, sign, strict=False):
    """the training episode (log-interpolated between checkpoints) after which y stays past lev (sign +1: at or above;
    -1: at or below; strict: strictly beyond, td_tt_gap's rule) to the end of training; NaN if the last checkpoint is
    not past it (never), x[0] if past throughout"""
    y = np.asarray(y, np.float64)
    past = sign * (y - lev) > 0 if strict else sign * (y - lev) >= 0
    if not past[-1]:
        return np.nan
    k = np.flatnonzero(~past)
    if not len(k):
        return float(x[0])
    k = int(k[-1])
    f = (lev - y[k]) / (y[k + 1] - y[k])
    return float(np.exp(np.log(x[k]) + f * (np.log(x[k + 1]) - np.log(x[k]))))


def first_level(x, y, lev, sign):
    """the first training episode (log-interpolated) at which y reaches lev (for the rise of a curve that falls back
    later, where 'stays past' does not apply); NaN if never"""
    y = np.asarray(y, np.float64)
    past = sign * (y - lev) >= 0
    if not past.any():
        return np.nan
    k = int(np.argmax(past))
    if k == 0:
        return float(x[0])
    f = (lev - y[k - 1]) / (y[k] - y[k - 1])
    return float(np.exp(np.log(x[k - 1]) + f * (np.log(x[k]) - np.log(x[k - 1]))))


def end_value(y):
    """a curve's end value: the mean of its last 3 checkpoints"""
    return np.asarray(y)[..., -3:].mean(-1)


def crossing(x, y, yb, lev, sign, levb=None, first=False):
    """cross_level (or first_level) of the curve y and its 5-95% interval over the bootstrap curves yb (B, K) crossing the
    same level (or their own paired levels levb (B,)); a resample that never gets past counts as later than the end
    (inf)"""
    fn = first_level if first else cross_level
    t = fn(x, y, lev, sign)
    lb = np.full(len(yb), lev) if levb is None else levb
    tb = np.array([fn(x, v, lv, sign) for v, lv in zip(yb, lb)])
    lo, hi = np.percentile(np.where(np.isnan(tb), 1e12, tb), [5, 95])  # never = beyond the end
    return t, *(v if v <= x[-1] else np.inf for v in (lo, hi))


def frac_times(x, y, yb, fracs=(.1, .5, .9), y0=None):
    """for each fraction f: (crossing, lo, hi, level) of the level y0 + f (end - y0), y0 = the initial network unless
    given, end = end_value(y); the bootstrap curves cross the same level"""
    y0 = y[0] if y0 is None else y0
    y1 = end_value(y)
    return {f: (*crossing(x, y, yb, y0 + f * (y1 - y0), np.sign(y1 - y0)), y0 + f * (y1 - y0)) for f in fracs}


def back_time(x, y, yb, kind="max", frac=.5):
    """the curve's peak (kind max) or trough (min), and the crossing of the level frac of the way from it back to the end
    value: (extreme value, its x, (t, lo, hi, level))"""
    k = int(np.argmax(y) if kind == "max" else np.argmin(y))
    y1 = end_value(y)
    lev = y[k] + frac * (y1 - y[k])
    return y[k], x[k], (*crossing(x, y, yb, lev, np.sign(y1 - y[k])), lev)


def passes_naive():
    """training episodes at which the Trained RNN's goal accuracy at T first exceeds naive's (probes, all 114
    checkpoints; = fig_entanglement_driver.passes_naive)"""
    S, Rf = probe_stats("trained"), probe_refs()
    k = int(np.argmax(S["acc"] > Rf["naive_acc"]))
    return float(S["x"][k])


def unfolding():
    """the belief grid's unfolding (td_summary_data.grids(): % of the grid folded at T where naive's grid folds, 19
    checkpoints): training episodes at 25 / 50 / 75% of its drop from the initial network to the end"""
    G = np.load(os.path.join(TD, "summary_data", "grids.npz"))
    x, y = G["x_trained"], G["where_naive_trained"]
    y1 = y[-3:].mean()
    return {f: cross_level(x, y, y[0] + f * (y1 - y[0]), np.sign(y1 - y[0])) for f in (.25, .5, .75)}, x, y


def _subset(S, eps):
    """the rows of per-checkpoint arrays at the episodes eps (count arrays and scalars pass unchanged)"""
    k = np.searchsorted(S["episode"], eps)
    assert (S["episode"][k] == eps).all()
    K = len(S["episode"])
    return {key: (v[k] if (np.ndim(v) and len(v) == K and not key.startswith("n_")) else v) for key, v in S.items()}


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

XMIN, XSEP, XMAX = 8, 4.6e4, 9e4          # x range; the separator before the right-edge references
TFS, LFS, KFS, NFS = 12.5, 10.5, 10, 9.5  # title, axis label, legend, footnote sizes
LS = {"solid": "-", "dashed": (0, (4, 2)), "dotted": (0, (1, 1.6))}


def xaxis(ax, cross, unf, label=True, xmax=XMAX, band=True):
    ax.set_xscale("log")
    ax.set_xticks([X0, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.set_xlim(XMIN, xmax)
    ax.minorticks_off()
    if band:
        ax.axvspan(unf[.25], unf[.75], color="0.92", lw=0, zorder=0)
    ax.axvline(cross, color="0.4", lw=1.1, ls=":", zorder=1)
    ax.axvline(XSEP, color="0.8", lw=.8)
    if label:
        ax.set_xlabel("training episodes", fontsize=LFS)
    ax.spines[["top", "right"]].set_visible(False)


def refs_edge(ax, items):
    """right-edge reference segments (td_summary's convention): items = [(value, colour, line style)]"""
    for v, c, ls in items:
        ax.plot([XSEP * 1.06, XMAX], [v, v], color=c, lw=1.8, ls=ls, solid_capstyle="butt")


def band_line(ax, x, y, yb, col, ls, lw=2.):
    lo, hi = np.nanpercentile(yb, [5, 95], 0)
    ax.fill_between(x, lo, hi, color=col, alpha=.18, lw=0)
    ax.plot(x, y, color=col, lw=lw, ls=ls)


def _title(ax, name, formula):
    ax.set_title(f"{name}\n{formula}", loc="left", fontsize=TFS, linespacing=1.5)


def figure():
    """td_tt_wormholes.png: the forward set (grid episodes) through training"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    eps = fig_checkpoints()
    F = {r: fwd_stats(r) for r in RUNS}
    Rf = fwd_refs()
    x = F["trained"]["x"]
    W = boot_weights()
    cross = passes_naive()
    unf, _, _ = unfolding()
    B, N = "k", "0.55"
    fig, axs = plt.subplots(2, 4, figsize=(19.5, 9.4))
    axs = axs.ravel()
    # a: corner splits held / committed
    ax = axs[0]
    for r in RUNS:
        for key, ls in (("joint_held", "solid"), ("commit", "dashed")):
            band_line(ax, x, agg(F[r][key], F[r]["n_two"]), agg(F[r][key], F[r]["n_two"], W), COL[r], LS[ls])
    refs_edge(ax, [(ref(Rf, "bayes_joint_held", Rf["n_two"]), B, LS["solid"]),
                   (ref(Rf, "naive_joint_held", Rf["n_two"]), N, LS["solid"]),
                   (ref(Rf, "naive_commit", Rf["n_two"]), N, LS["dashed"]),
                   (ref(Rf, "bayes_commit", Rf["n_two"]), B, LS["dashed"])])
    ax.set_ylim(-.02, 1.04)
    ax.set_ylabel("share of Bayes's corner splits", fontsize=LFS)
    _title(ax, "a   Corner split: held (solid) or committed (dashed)",
           r"held: $\min(q_v(0), q_v(9)) > .15$ for both $v$")
    # b: per variable, corner vs one-variable splits
    ax = axs[1]
    for r in RUNS:
        for key, den, ls in (("held_two", "n_two_v", "solid"), ("held_one", "n_one_v", "dashed")):
            band_line(ax, x, agg(F[r][key], F[r][den]), agg(F[r][key], F[r][den], W), COL[r], LS[ls])
    refs_edge(ax, [(1., B, LS["solid"]), (ref(Rf, "naive_held_two", Rf["n_two_v"]), N, LS["solid"]),
                   (ref(Rf, "naive_held_one", Rf["n_one_v"]), N, LS["dashed"])])
    ax.set_ylim(-.02, 1.04)
    ax.set_ylabel("share of Bayes's end splits of $v$", fontsize=LFS)
    _title(ax, "b   End split of $v$ held: the other variable\nsplit too (solid) or not split (dashed)",
           r"$\min(q_v(0), q_v(9)) > .15$")
    # c: total variation from Bayes's marginals
    ax = axs[2]
    for r in RUNS:
        for key, den, ls in (("tv_two", "n_two_v", "solid"), ("tv_one", "n_one_v", "dashed"),
                             ("tv_ctl_two", "n_w_two", "dotted")):
            band_line(ax, x, agg(F[r][key], F[r][den]), agg(F[r][key], F[r][den], W), COL[r], LS[ls])
    refs_edge(ax, [(0., B, LS["solid"]), (ref(Rf, "naive_tv_two", Rf["n_two_v"]), N, LS["solid"]),
                   (ref(Rf, "naive_tv_one", Rf["n_one_v"]), N, LS["dashed"]),
                   (ref(Rf, "naive_tv_ctl_two", Rf["n_w_two"]), N, LS["dotted"])])
    ax.set_ylim(-.02, .9)
    ax.set_ylabel("total variation (per variable)", fontsize=LFS)
    _title(ax, "c   Distance from Bayes's marginals: corner\n(solid) / one-variable splits (dashed), matched\n"
           "non-splits (dotted)", r"$\frac{1}{2}\,\Sigma_r\,|q_v(r) - \pi_v(r)|$")
    # d: corner truths, answer = truth
    ax = axs[3]
    for r in RUNS:
        num = F[r]["corner_cat"][..., 0]
        band_line(ax, x, agg(num, F[r]["n_corner"]), agg(num, F[r]["n_corner"], W), COL[r], LS["solid"])
    refs_edge(ax, [(ref(Rf, "bayes_corner_cat", Rf["n_corner"])[0], B, LS["solid"]),
                   (ref(Rf, "naive_corner_cat", Rf["n_corner"])[0], N, LS["solid"])])
    ax.set_ylim(-.02, 1.)
    ax.set_ylabel("share of corner-cell episodes", fontsize=LFS)
    _title(ax, "d   Corner cells: answer at T is the truth",
           r"$\hat r_T = r,\ \ r \in \{0, 9\}^2,\ \ \hat r_{T,v} = \mathrm{argmax}\ q_{T,v}$")
    # e: corner truths, wrong corner
    ax = axs[4]
    for r in RUNS:
        for k, ls in ((1, "solid"), (2, "dotted")):
            num = F[r]["corner_cat"][..., k]
            band_line(ax, x, agg(num, F[r]["n_corner"]), agg(num, F[r]["n_corner"], W), COL[r], LS[ls])
    rb, rn = ref(Rf, "bayes_corner_cat", Rf["n_corner"]), ref(Rf, "naive_corner_cat", Rf["n_corner"])
    refs_edge(ax, [(rb[1], B, LS["solid"]), (rb[2], B, LS["dotted"]), (rn[1], N, LS["solid"]),
                   (rn[2], N, LS["dotted"])])
    ax.set_ylim(-.004, .23)
    ax.set_ylabel("share of corner-cell episodes", fontsize=LFS)
    _title(ax, "e   Corner cells: wrong corner at T, adjacent\n(solid) or the partner corner (dotted)",
           r"adjacent: $(9 - r_0, r_1)$ or $(r_0, 9 - r_1)$;  partner: $(9 - r_0, 9 - r_1)$")
    # f, g: moves out of Bayes's corner splits (wormhole traversals)
    for ax, key, lab, (lo_, hi_), ttl, form in (
            (axs[5], "phi_out", "coupling angle (degrees; 45 = diagonal)", (0, 46),
             "f   Leaving a corner split: direction",
             r"$\phi = \arctan(\mathrm{min}_v|\Delta\mu_v|\, /\, \mathrm{max}_v|\Delta\mu_v|)$"),
            (axs[6], "proj_out", "distance along Bayes's move (cells)", (-.1, 3.9),
             "g   Leaving a corner split: distance travelled",
             r"$\Delta\mu \cdot \Delta\mu^B / |\Delta\mu^B|,\ \ \Delta\mu = (\Delta E[r_0], \Delta E[r_1])$")):
        for r in RUNS:
            band_line(ax, x, agg(F[r][key], F[r]["n_out"]), agg(F[r][key], F[r]["n_out"], W), COL[r], LS["solid"])
        refs_edge(ax, [(ref(Rf, "bayes_" + key, Rf["n_out"]), B, LS["solid"]),
                       (ref(Rf, "naive_" + key, Rf["n_out"]), N, LS["solid"])])
        ax.set_ylim(lo_, hi_)
        ax.set_ylabel(lab, fontsize=LFS)
        _title(ax, ttl + (" of the network's move\n" if key == "phi_out" else "\nby the network ") +
               "where Bayes moves out of it diagonally", form)
    for i, ax in enumerate(axs[:7]):
        xaxis(ax, cross, unf, label=i >= 3)
    # legend panel
    ax = axs[7]
    ax.axis("off")
    h = [Line2D([], [], color=COL[r], lw=2.4, label=NAME[r]) for r in RUNS]
    h += [Line2D([], [], color=B, lw=1.8, label="Bayes (right edge)"),
          Line2D([], [], color=N, lw=1.8, label="naive (right edge)"),
          Line2D([], [], color="0.4", lw=1.1, ls=":", label=f"Trained RNN passes naive ({cross:,.0f})"),
          Line2D([], [], color="0.88", lw=9, label=f"belief grid unfolds ({unf[.25]:,.0f}-{unf[.75]:,.0f}:\n"
                 "25-75% of its folds gone)")]
    ax.legend(handles=h, loc="upper left", frameon=False, fontsize=KFS + 1.5, handlelength=2.6, borderaxespad=0,
              labelspacing=1.1)
    ax.text(0, .3, "line styles: see each panel's title\nbands: 5-95% over environments resampled",
            transform=ax.transAxes, fontsize=KFS + .5, va="top", ha="left", color="0.3", linespacing=1.6)
    n_two, n_one = int(F["trained"]["n_two"].sum()), int(F["trained"]["n_one_v"].sum())
    nep = len(np.load(os.path.join(OUT, "grid_bayes.npz"))["idx"])
    note = (f"Data: the grid episodes (50 environments x 100 cells x 24 trajectories x 30 steps), of which {nep:,} are "
            f"replayed through each network: every episode with a corner split or a one-variable end split (Bayes) "
            f"and 12 trajectories of each corner cell; {len(eps)} checkpoints per network (every 2nd of 114).\n"
            f"a-c: corner split (Bayes, steps 2-16): both marginals $\\pi_v$ have $\\min(\\pi_v(0), \\pi_v(9)) > .15$ "
            f"and $\\pi_v(0) + \\pi_v(9) > .6$ ({n_two:,} stream-steps); one-variable split: $v$ split, the other "
            f"variable not ($< .05$ at one of its ends; {n_one:,}).  Committed: both answers on a corner, split not "
            f"held.  "
            f"Matched non-splits: no end split, weighted to the corner splits'\nstep x naive's error (FR, .25 log10) "
            f"x Bayes's entropy (.2 nats).  d, e: {int(F['trained']['n_corner'].sum()):,} corner-cell episodes "
            f"(12 per corner cell and environment).  f, g: {int(F['trained']['n_out'].sum()):,} moves of Bayes's "
            f"posterior mean of $\\geq 2$ cells at $\\geq 30$ degrees out of a corner split (the step before the move "
            f"is a corner split, at steps 2-29).")
    fig.text(.005, -.005, note, fontsize=NFS, va="top", ha="left", color="0.25", linespacing=1.55)
    fig.tight_layout(h_pad=2.2, w_pad=1.8)
    path = fig_path("td_tt_wormholes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


def figure_maps():
    """td_tt_wormholes_maps.png: unthresholded views on the probes (10,000 episodes), the same checkpoints"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    eps = np.array(fig_checkpoints())
    S = {r: _subset(probe_stats(r), eps) for r in RUNS}
    Rp = probe_refs()
    x = S["trained"]["x"]
    W = boot_weights()
    cross = passes_naive()
    unf, _, _ = unfolding()
    xmax = 2.2e5
    xr = {"naive": (5.2e4, 9.6e4), "bayes": (1.04e5, 1.95e5)}
    lx = np.log(x)
    mid = (lx[1:] + lx[:-1]) / 2
    xe = np.exp(np.concatenate([[lx[0] - (mid[0] - lx[0])], mid, [lx[-1] + (lx[-1] - mid[-1])]]))
    fig, axs = plt.subplots(3, 2, figsize=(15.5, 12.6), gridspec_kw={"height_ratios": [.75, 1, 1]})
    cols = (
        ("e", "Mass kept at the smaller end, every belief (no threshold)",
         r"maps: $e_v = \min(q_v(0), q_v(9))$ in bins of Bayes's $e_v$" + "\n" +
         r"top: slope within steps,  $\Sigma_t\Sigma_i\,(e^{net}_i - \bar{e}^{net}_t)(e^B_i - \bar{e}^B_t)\ /\ "
         r"\Sigma_t\Sigma_i\,(e^B_i - \bar{e}^B_t)^2$",
         Normalize(0, .45), "network's mean $e_v$ (step drift removed)"),
        ("phi", r"Direction of the network's move at Bayes's moves of $\geq 2$ cells",
         r"maps: $\phi = \arctan(\mathrm{min}_v|\Delta\mu_v|\, /\, \mathrm{max}_v|\Delta\mu_v|)$ in bins of Bayes's "
         r"$\phi$" + "\n" +
         r"top: slope within steps,  $\Sigma_t\Sigma_i\,(\phi^{net}_i - \bar{\phi}^{net}_t)(\phi^B_i - \bar{\phi}^B_t)"
         r"\ /\ \Sigma_t\Sigma_i\,(\phi^B_i - \bar{\phi}^B_t)^2$",
         Normalize(0, 45), r"network's mean $\phi$ (degrees; step drift removed)"))
    refs_all = {nm: _ref_rows(Rp, nm) for nm in ("bayes", "naive")}
    for j, (kind, ttl, form, nrm, cbl) in enumerate(cols):
        ax = axs[0, j]
        for r in RUNS:
            band_line(ax, x, maps_slope(S[r], Rp, kind), maps_slope(S[r], Rp, kind, W), COL[r], "-")
        refs_edge(ax, [(1., "k", "-"), (float(maps_slope(refs_all["naive"], Rp, kind)[0]), "0.55", "-")])
        ax.axhline(0, color="0.8", lw=.8, zorder=0)
        ax.set_ylim(-.05, 1.08)
        ax.set_ylabel("slope (Bayes 1)", fontsize=LFS)
        xaxis(ax, cross, unf, label=False, xmax=xmax)
        ax.set_title(f"{'ab'[j]}   {ttl}\n{form}", loc="left", fontsize=TFS, linespacing=1.5)
        refs = {nm: maps_heat(refs_all[nm], Rp, kind)[0] for nm in xr}
        for i, r in enumerate(RUNS):
            ax = axs[1 + i, j]
            M = maps_heat(S[r], Rp, kind)
            yed = np.arange((NEB if kind == "e" else NPB) + 1)
            ax.pcolormesh(xe, yed, M.T, cmap="viridis", norm=nrm, shading="flat", rasterized=True)
            for nm, (a0, a1) in xr.items():
                ax.pcolormesh([a0, a1], yed, refs[nm][:, None], cmap="viridis", norm=nrm, shading="flat",
                              rasterized=True)
                ax.text(np.sqrt(a0 * a1), yed[-1] + .15, {"naive": "naive", "bayes": "Bayes"}[nm], ha="center",
                        va="bottom", fontsize=NFS + .5, color="0.25")
            xaxis(ax, cross, unf, label=i == 1, xmax=xmax, band=False)
            ax.axvline(cross, color="w", lw=1.3, ls=":")
            ax.axvline(XSEP, color="w", lw=1.)
            ax.set_yticks(yed)
            if kind == "e":
                ax.set_yticklabels([f"{v:g}" if v < .5 else ".5" for v in E_EDGES])
                ax.set_ylabel("Bayes's $e_v$ (bin edges)", fontsize=LFS)
            else:
                ax.set_yticklabels([f"{v:.0f}" for v in PHI_EDGES])
                ax.set_ylabel(r"Bayes's $\phi$ (degrees)", fontsize=LFS)
            if j == 0:
                ax.text(-.17, .5, NAME[r], transform=ax.transAxes, rotation=90, va="center", ha="center",
                        fontsize=15, color=COL[r])
            ax.set_ylim(yed[0], yed[-1] + .9)
        cax = axs[2, j].inset_axes([.5, -.27, .47, .045])
        cb = fig.colorbar(ScalarMappable(norm=nrm, cmap="viridis"), cax=cax, orientation="horizontal")
        cb.set_label(cbl, fontsize=LFS)
    h = [Line2D([], [], color=COL[r], lw=2.4, label=NAME[r]) for r in RUNS]
    h += [Line2D([], [], color="k", lw=1.8, label="Bayes"), Line2D([], [], color="0.55", lw=1.8, label="naive"),
          Line2D([], [], color="0.4", lw=1.1, ls=":", label=f"passes naive ({cross:,.0f})"),
          Line2D([], [], color="0.88", lw=9, label="grid unfolds")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(.02, -.045), ncol=6, frameon=False, fontsize=KFS + 1)
    ne = int(Rp["n_e_st"].sum())
    nj = int(Rp["n_jump_big"].sum())
    nt = Rp["n_jump_big_st"].sum(0)
    nv = refs_all["naive"]
    _, w2, s2 = step_parts(nv, Rp, "phi")[0]
    (eq30, n30), (eq1, n1) = equal_weight(nv, Rp, "phi")
    fig.text(.02, -.06, f"Probes: 50 environments x 100 cells x 2 trajectories x 30 steps (10,000 episodes); "
             f"{len(eps)} checkpoints per network (every 2nd of 114).  a: every variable at steps 2-30 "
             f"({ne:,} variable-steps), binned by Bayes's smaller-end mass $e_v$.\nb: Bayes's moves of the posterior "
             f"mean into steps 2-30 of $\\geq 2$ cells ({nj:,}), binned by their coupling angle; colour = the "
             f"network's own move's angle at the same step.  Step drift removed: maps show each value minus the "
             f"network's mean at its\nstep plus its mean over all steps; slopes are fitted after removing each "
             f"step's mean from both ($i$: the points at step $t$, so a step counts by its points; in b, from "
             f"{int(nt[0]):,} moves at step 2 to {int(nt[-1])} at step 30).  Naive's direction slope "
             f"({float(maps_slope(nv, Rp, 'phi')[0]):.3f})\ncomes mostly from step 2 ({w2:.0%} of the weight, slope "
             f"{s2:.3f} there; equal weight per step (the {n30} steps with $\\geq {MIN_STEP_N}$ moves): {eq30:+.3f}; "
             f"over all {n1} steps: {eq1:+.3f}), so passing naive's level in b depends on this weighting.\n"
             f"Right strips: naive's and Bayes's own values.  Bands: 5-95% over environments resampled.",
             fontsize=NFS, color="0.25", va="top", linespacing=1.5)
    fig.tight_layout(h_pad=1.6, w_pad=3.)
    path = fig_path("td_tt_wormholes_maps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


# ═══════════════════════════════════════════════════════════════════
# Numbers and the timeline export
# ═══════════════════════════════════════════════════════════════════

SHOW = (0, 50, 100, 250, 300, 500, 700, 900, 1100, 1500, 1900, 2100, 2500, 3250, 4250, 5250, 7250, 9750, 13000,
        19000, 25000, 33000, 41000, 49995)


def gap_source(ep=49995):
    """the late raw gap in per-variable holding (corner vs one-variable end splits), forward set at checkpoint ep: the
    one-variable splits matched to the corner splits' histogram of step only, Bayes's e_v only, or both; held shares by
    Bayes-e_v bin; the composition of the two sets (mean step, mean e_v, share barely split, e_v < .2)"""
    G = grid_side()
    S = fwd_side(G)["S"]
    eB = ends(G["bB"])[0]
    two_v, one_v = S["two_v"], S["one_v"]
    tt = np.broadcast_to(np.arange(T)[None, :, None], eB.shape)
    eb = np.floor(eB / E_STEP).astype(int)
    print(f"composition: mean step corner {tt[two_v].mean() + 1:.2f} one-var {tt[one_v].mean() + 1:.2f}; mean Bayes e_v "
          f"{eB[two_v].mean():.3f} / {eB[one_v].mean():.3f}; e_v < .2 in {(eB[two_v] < .2).mean():.3f} / "
          f"{(eB[one_v] < .2).mean():.3f} of the var-steps")
    b = np.full(G["bB"].shape, np.nan)
    b[tgt_mask(G["bB"])] = norm(np.load(os.path.join(OUT, "fwd_trained", f"ck_{ep:05d}.npz"))["lq_tgt"])
    h = ends(b)[0] > EDGE
    for nm, key in (("step only", tt * 1), ("Bayes e_v only", eb), ("step x e_v", tt * 1000 + eb)):
        kt, ct = np.unique(key[two_v], return_counts=True)
        ks, inv, cs = np.unique(key[one_v], return_inverse=True, return_counts=True)
        pos = np.minimum(np.searchsorted(kt, ks), len(kt) - 1)
        wt = np.where(kt[pos] == ks, ct[pos] / cs, 0.)[inv.ravel()]
        print(f"   {train_x(ep):.0f}: corner {h[two_v].mean():.3f}, one-var raw {h[one_v].mean():.3f}, matched on {nm} "
              f"{(wt * h[one_v]).sum() / wt.sum():.3f} (corner coverage {np.isin(key[two_v], ks).mean():.3f})")
    for nm, m in (("corner", two_v), ("one-var", one_v)):
        print(f"   held by Bayes-e_v bin (.15-.2, .2-.25, ..., .45-.5), {nm}: " +
              " ".join(f"{h[m & (eb == k)].mean():.3f}" for k in range(3, 10)))


def _ft(t):
    return "never" if np.isnan(t) else (">end" if np.isinf(t) else f"{t:,.0f}")


def _fc(c):
    """(t, lo, hi[, level]) -> 't (lo-hi)'"""
    return f"{_ft(c[0])} ({_ft(c[1])}-{_ft(c[2])})"


def _load_all():
    eps = np.array(fig_checkpoints())
    F = {r: fwd_stats(r) for r in RUNS}
    P = {r: _subset(probe_stats(r), eps) for r in RUNS}
    return eps, F, fwd_refs(), P, probe_refs()


def headline_curves(F, Rf, P, Rp, W):
    """the timeline's headline measures: name -> (family, {run: (curve (K,), bootstrap curves (B, K))}, Bayes, naive,
    how the half-way level is set)"""
    def cur(D, num, den, k=None):
        n = D[num] if k is None else D[num][..., k]
        return agg(n, D[den]), agg(n, D[den], W)
    rb, rn = ref(Rf, "bayes_corner_cat", Rf["n_corner"]), ref(Rf, "naive_corner_cat", Rf["n_corner"])
    nv = float(maps_slope(_ref_rows(Rp, "naive"), Rp, "e")[0])
    return {
        "corner split held": ("movement", {r: cur(F[r], "joint_held", "n_two") for r in RUNS},
                              ref(Rf, "bayes_joint_held", Rf["n_two"]), ref(Rf, "naive_joint_held", Rf["n_two"]),
                              "half"),
        "committed to a corner": ("answers", {r: cur(F[r], "commit", "n_two") for r in RUNS},
                                  ref(Rf, "bayes_commit", Rf["n_two"]), ref(Rf, "naive_commit", Rf["n_two"]), "back"),
        "corner answer = truth": ("answers", {r: cur(F[r], "corner_cat", "n_corner", 0) for r in RUNS}, rb[0], rn[0],
                                  "half"),
        "distance out of a split": ("movement", {r: cur(F[r], "proj_out", "n_out") for r in RUNS},
                                    ref(Rf, "bayes_proj_out", Rf["n_out"]), ref(Rf, "naive_proj_out", Rf["n_out"]),
                                    "half"),
        "angle out of a split": ("movement", {r: cur(F[r], "phi_out", "n_out") for r in RUNS},
                                 ref(Rf, "bayes_phi_out", Rf["n_out"]), ref(Rf, "naive_phi_out", Rf["n_out"]), "half"),
        "end-mass slope (maps)": ("movement", {r: (maps_slope(P[r], Rp, "e"), maps_slope(P[r], Rp, "e", W))
                                               for r in RUNS}, 1., nv, "half")}


def timeline():
    """timeline.npz for the cross-topic synthesis: the headline measures for both networks, their 5-95% bands, the
    Bayes / naive levels, and the half-way crossing (the Trained RNN's level, used for both networks)"""
    eps, F, Rf, P, Rp = _load_all()
    W = boot_weights()
    x = F["trained"]["x"]
    rows = {k: [] for k in ("names", "run", "family", "value", "lo", "hi", "bayes", "naive", "level", "half", "half_lo",
                            "half_hi", "level_def")}
    for nm, (fam, cv, bv, nv, how) in headline_curves(F, Rf, P, Rp, W).items():
        y, yb = cv["trained"]
        if how == "half":
            y0, y1 = y[0], end_value(y)
            lev, sg = y0 + .5 * (y1 - y0), np.sign(y1 - y0)
            ldef = "half-way from the Trained RNN's initial network to its end (mean of the last 3 checkpoints)"
        else:
            k = int(np.argmax(y))
            lev, sg = y[k] + .5 * (end_value(y) - y[k]), -1.
            ldef = "half-way from the Trained RNN's peak back to its end (after which it stays below)"
        for r in RUNS:
            v, vb = cv[r]
            t, lo, hi = crossing(x, v, vb, lev, sg)
            for key, val in (("names", nm), ("run", r), ("family", fam), ("value", v),
                             ("lo", np.percentile(vb, 5, 0)), ("hi", np.percentile(vb, 95, 0)), ("bayes", bv),
                             ("naive", nv), ("level", lev), ("half", t), ("half_lo", lo), ("half_hi", hi),
                             ("level_def", ldef)):
                rows[key].append(val)
    out = {k: np.array(v) for k, v in rows.items()}
    for k in ("half_lo", "half_hi"):                                   # never within the bootstrap -> NaN
        out[k] = np.where(np.isinf(out[k]), np.nan, out[k])
    out["half"] = np.where(np.isinf(out["half"]), np.nan, out["half"])
    out["x"] = x
    path = os.path.join(OUT, "timeline.npz")
    np.savez(path, **out)
    for i in range(len(out["names"])):
        print(f"{out['names'][i]:25s} {out['run'][i]:8s} {out['family'][i]:9s} level {out['level'][i]:.4g}: half "
              f"{_ft(out['half'][i])} ({_ft(out['half_lo'][i])}-{_ft(out['half_hi'][i])}); end "
              f"{end_value(out['value'][i]):.3f}; Bayes {out['bayes'][i]:.3f} naive {out['naive'][i]:.3f}")
    print("saved", path)
    return out


# ═══════════════════════════════════════════════════════════════════
# Synthesis export (td_tt_gap: gap fraction, shared environment resamples)
# ═══════════════════════════════════════════════════════════════════

# name, family, shape, source: ("fwd", per-environment sum key, count key, column or None) or ("maps", kind)
SYNTHESIS = (
    ("corner answer = truth at T", "answers", "rise", ("fwd", "corner_cat", "n_corner", 0)),
    ("corner split held (joint)", "movement", "rise", ("fwd", "joint_held", "n_two", None)),
    ("held per variable, corner-split ends", "movement", "rise", ("fwd", "held_two", "n_two_v", None)),
    ("held per variable, one-variable end splits", "movement", "rise", ("fwd", "held_one", "n_one_v", None)),
    ("distance out of a corner split", "movement", "rise", ("fwd", "proj_out", "n_out", None)),
    ("coupling angle out of a corner split", "coupling", "rise", ("fwd", "phi_out", "n_out", None)),
    ("same-diagonal share of moves out of corner splits", "coupling", "rise", ("fwd", "same_out", "n_out", None)),
    ("end-mass slope within steps (maps)", "movement", "rise", ("maps", "e")),
    ("direction slope within steps (maps)", "coupling", "rise", ("maps", "phi")),
    ("committed to a corner", "movement", "return", ("fwd", "commit", "n_two", None)))


def env_alignment():
    """is environment e of the forward set the probes' environment e? The grid episodes were drawn from probes.npz's
    tables L[e] with its contexts Z (td_grids_fwd.episodes); checked on the data: identical Z, environment-major
    indices in both sets, and the share of episodes whose observations (at their true cell) are most likely under their
    own environment's table among the 50 (the same in both sets if the indices agree; ~2% if they were permuted)"""
    P = np.load(os.path.join(AN, "probes.npz"))
    G = np.load(os.path.join(OUT, "grid_bayes.npz"))
    E = np.load(os.path.join(GRIDS, "fwd_episodes.npz"))
    L = P["L"].astype(np.float64)

    def own(obs, r, env):
        k = obs.astype(np.float64).sum(1).T[None]                     # (1, 5, n) observations = 1 over the T steps
        p = L[:, :, r[:, 0], r[:, 1]]                                 # (NE, 5, n)
        return float(((k * np.log(p) + (T - k) * np.log1p(-p)).sum(1).argmax(0) == env).mean())
    n_p, n_g = len(P["env"]), len(E["env"])
    return dict(z_same=bool(np.array_equal(P["Z"], E["Z"]) and np.array_equal(P["Z"], G["Z"])),
                probes_env_major=bool(np.array_equal(P["env"], np.arange(n_p) // (n_p // NE))),
                grid_env_major=bool(np.array_equal(E["env"], np.repeat(np.arange(NE), n_g // NE))),
                fwd_env_kept=bool(np.array_equal(G["env"], E["env"][G["idx"]])),
                own_probes=own(P["obs"], P["r"], P["env"]), own_grid=own(E["obs"], E["r"], E["env"]),
                own_fwd=own(G["obs"], G["r"], G["env"]))


def synthesis_rows(F=None, Rf=None, P=None, Rp=None):
    """the export's rows (td_tt_gap.export fields) on the 58 checkpoints: every measure of SYNTHESIS for the Trained RNN
    and the ESN, its curve in each of the 1,000 shared environment resamples (gap.boot_counts(); a forward-set ratio of
    sums: each environment's sums weighted by its count; a within-step slope: refitted with the weights, step means
    included), naive's and Bayes's levels on the same episodes, resampled with the same counts -> (x, rows, checks)"""
    import td_tt_gap as gap
    eps = np.array(fig_checkpoints())
    F = F or {r: fwd_stats(r) for r in RUNS}
    Rf = Rf or fwd_refs()
    P = P or {r: _subset(probe_stats(r), eps) for r in RUNS}
    Rp = Rp or probe_refs()
    C = gap.boot_counts()
    C1 = np.ones((1, NE))
    x = gap.train_x(eps)
    assert np.array_equal(x, F["trained"]["x"]) and np.array_equal(x, P["trained"]["x"])
    al = env_alignment()
    assert al["z_same"] and al["probes_env_major"] and al["grid_env_major"] and al["fwd_env_kept"], al
    assert abs(al["own_grid"] - al["own_probes"]) < .02, al
    nep = len(np.load(os.path.join(OUT, "grid_bayes.npz"))["idx"])
    n = {k: int(Rf[k].sum()) for k in ("n_corner", "n_two", "n_two_v", "n_one_v", "n_out")}
    sample = {
        "n_corner": f"forward set: {n['n_corner']:,} unselected corner-cell episodes (12 per corner cell and "
                    f"environment), 50 environments",
        "n_two": f"forward set: {n['n_two']:,} corner-split stream-steps (Bayes; steps 2-16) of {nep:,} replayed grid "
                 f"episodes, 50 environments",
        "n_two_v": f"forward set: {n['n_two_v']:,} corner-split var-steps (Bayes; steps 2-16), 50 environments",
        "n_one_v": f"forward set: {n['n_one_v']:,} one-variable end-split var-steps (Bayes; steps 2-16), "
                   f"50 environments",
        "n_out": f"forward set: {n['n_out']:,} moves of Bayes's >= 2 cells at >= 30 deg out of corner splits (steps "
                 f"2-29), 50 environments",
        "e": f"probes: 50 environments x 200 episodes; {int(Rp['n_e_st'].sum()):,} variable-steps (steps 2-30)",
        "phi": f"probes: 50 environments x 200 episodes; {int(Rp['n_jump_big'].sum()):,} moves of Bayes's >= 2 cells "
               f"(steps 2-30)"}
    pair_fwd = (f"paired: forward-set environment e = probe environment e (grid episodes drawn from probes.npz's L[e] "
                f"with its Z; checked: identical Z and indices, own table the most likely for {al['own_grid']:.1%} "
                f"of the grid episodes vs {al['own_probes']:.1%} of the probes)")
    nv_parts = step_parts(_ref_rows(Rp, "naive"), Rp, "phi")
    (nv_eq30, n30), (nv_eq1, n1) = equal_weight(_ref_rows(Rp, "naive"), Rp, "phi")
    rows, worst = [], 0.
    for run in RUNS:
        Ec = P[run]
        for name, fam, shape, src in SYNTHESIS:
            if src[0] == "fwd":
                _, key, den, col = src
                pick = (lambda a: a) if col is None else (lambda a, c=col: a[..., c])
                assert np.array_equal(F[run][den], Rf[den])
                num = pick(F[run][key])                                           # (K, NE)
                value, boot = agg(num, Rf[den]), agg(num, Rf[den], C)
                worst = max(worst, np.abs(agg(num, Rf[den], C1)[0] - value).max())
                lv = {nm: (float(agg(pick(Rf[f"{nm}_{key}"])[None], Rf[den])[0]),
                           agg(pick(Rf[f"{nm}_{key}"])[None], Rf[den], C)[:, 0]) for nm in ("naive", "bayes")}
                smp, note = sample[den], pair_fwd
            else:
                kind = src[1]
                value, boot = maps_slope(Ec, Rp, kind), maps_slope(Ec, Rp, kind, C)
                worst = max(worst, np.abs(maps_slope(Ec, Rp, kind, C1)[0] - value).max())
                lv = {nm: (float(maps_slope(_ref_rows(Rp, nm), Rp, kind)[0]),
                           maps_slope(_ref_rows(Rp, nm), Rp, kind, C)[:, 0]) for nm in ("naive", "bayes")}
                smp = sample[kind]
                note = ("probes, environment = row // 200: the shared resamples; n-weighted pooled within-step slope "
                        "(step means refitted in every resample)")
                if kind == "phi":
                    note += (f"; naive's level ({lv['naive'][0]:.3f}) is mostly its step-2 slope ({nv_parts[0][2]:.3f}"
                             f", {nv_parts[0][1]:.0%} of the weight; equal weight per step (the {n30} steps with >= "
                             f"{MIN_STEP_N} moves): {nv_eq30:+.3f}; over all {n1} steps: {nv_eq1:+.3f}), so c0 "
                             f"depends on this weighting")
            if src[0] == "fwd" and lv["bayes"][0] == 1.:
                note += "; Bayes's level is 1 by definition"
            if name == "corner split held (joint)":
                note += (f"; naive's level is {lv['naive'][0]:.4f} (a handful of stream-steps), so c0 marks leaving 0 "
                         f"and moves with exact ties")
            if name.startswith("coupling angle"):
                k40, k12 = int(np.flatnonzero(x == 40)[0]), int(np.flatnonzero(x == 1200)[0])
                note += (f"; the curve starts at naive's level ({value[0]:.1f} deg; naive {lv['naive'][0]:.1f}), has "
                         f"an early transient ({value[k40]:.1f} at 40) and is at {value[k12]:.1f} at 1,200: c0 is "
                         f"poorly resolved (resamples past naive's level throughout give the first checkpoint)")
            if shape == "return":
                note += (f"; the initial network commits nothing (f = 1); f < 0 while the network commits more than "
                         f"naive ({lv['naive'][0]:.3f}); c0 / c50 time the return")
                if value.max() < lv["naive"][0]:
                    note += (f"; this network never commits more than naive (max {value.max():.3f}), so f > 0 at "
                             f"every checkpoint and c0 = {x[0]:.0f} (no event)")
            rows.append(dict(name=name, run=run, family=fam, shape=shape, sample=smp, paired=True, note=note,
                             value=value, boot=boot, naive=lv["naive"][0], bayes=lv["bayes"][0],
                             naive_boot=lv["naive"][1], bayes_boot=lv["bayes"][1]))
    return x, rows, dict(alignment=al, unit_counts_max_diff=float(worst), pair_fwd=pair_fwd)


def own_crossings(F, Rf, P, Rp, W=None):
    """the topic's own crossings of the Trained RNN for the synthesis measures (levels of its own change, the topic's
    resamples, seed 20261101): name -> [(label, (t, lo, hi), level)]"""
    W = boot_weights() if W is None else W
    x = F["trained"]["x"]

    def cur(key, den, col=None):
        n = F["trained"][key] if col is None else F["trained"][key][..., col]
        return agg(n, F["trained"][den]), agg(n, F["trained"][den], W)
    out = {}
    for name, _, _, src in SYNTHESIS:
        if src[0] == "fwd" and src[1] != "commit":
            y, yb = cur(*src[1:])
            c = frac_times(x, y, yb, (.5,))[.5]
            out[name] = [("50% of its change", c[:3], c[3])]
        elif src[0] == "maps":
            y, yb = maps_slope(P["trained"], Rp, src[1]), maps_slope(P["trained"], Rp, src[1], W)
            c = frac_times(x, y, yb, (.5,))[.5]
            nv = float(maps_slope(_ref_rows(Rp, "naive"), Rp, src[1])[0])
            nvb = maps_slope(_ref_rows(Rp, "naive"), Rp, src[1], W)[:, 0]
            out[name] = [("50% of its change", c[:3], c[3]), ("passes naive (paired)", crossing(x, y, yb, nv, 1., nvb),
                                                                nv)]
        else:
            y, yb = cur("commit", "n_two")
            _, _, c = back_time(x, y, yb, "max")
            out[name] = [("half-way back from its peak", c[:3], c[3])]
    return out


def _nan_diff(a, b):
    """max |a - b| where both are finite; 0 where both are NaN (never crosses); inf where only one is NaN"""
    a, b = np.atleast_1d(np.asarray(a, np.float64)), np.atleast_1d(np.asarray(b, np.float64))
    na, nb = np.isnan(a), np.isnan(b)
    d = np.where(na & nb, 0., np.where(na | nb, np.inf, np.abs(a - b)))
    return float(d.max())


def synthesis_report(x, rows, checks, own=None):
    """prints each row's c0 / c50 (td_tt_gap's gap fraction and crossings) with the level each one crosses, the same
    crossings from this module's cross_level on the same resamples and paired levels (strictly past, td_tt_gap's rule:
    must agree; with this module's own 'at or past' rule, exact ties move some resamples), and, for the Trained RNN,
    the topic's own crossings of that measure (other levels or resamples)"""
    import td_tt_gap as gap
    al = checks["alignment"]
    print(f"environment alignment: Z identical {al['z_same']}; env-major indices probes {al['probes_env_major']} / "
          f"grid {al['grid_env_major']}; forward set keeps the grid's env {al['fwd_env_kept']}; own table most likely: "
          f"probes {al['own_probes']:.4f}, all grid episodes {al['own_grid']:.4f}, forward set {al['own_fwd']:.4f}")
    print(f"all counts 1: resampled curve - full-sample curve, max |diff| over every row "
          f"{checks['unit_counts_max_diff']:.1e}")
    print(f"forward-set rows' note begins: {checks['pair_fwd']}")
    worst = 0.
    for r in rows:
        f = gap.gap_fraction(r["value"], r["naive"], r["bayes"])
        fb = gap.gap_fraction(r["boot"], r["naive_boot"], r["bayes_boot"])
        sg = np.sign(r["bayes"] - r["naive"])
        sgb = np.sign(r["bayes_boot"] - r["naive_boot"])
        parts, ties = [], []
        for nm, lev in (("c0", 0.), ("c50", .5)):
            c, cb = gap.crossing(x, f, lev), gap.crossing(x, fb, lev)
            lo, hi, p = gap.interval(cb)
            yl = r["naive"] + lev * (r["bayes"] - r["naive"])
            ylb = r["naive_boot"] + lev * (r["bayes_boot"] - r["naive_boot"])
            mine = cross_level(x, r["value"], yl, sg, strict=True)
            mb = np.array([cross_level(x, v, ylb[i], sgb[i], strict=True) for i, v in enumerate(r["boot"])])
            ma = np.array([cross_level(x, v, ylb[i], sgb[i]) for i, v in enumerate(r["boot"])])
            worst = max(worst, _nan_diff(mb, cb), _nan_diff(mine, c))
            nt = int(sum(_nan_diff(a, b) > 1e-6 for a, b in zip(ma, cb)))
            if nt:
                ties.append(f"{nm} {nt}")
            parts.append(f"{nm} (level {yl:.4g}) {_ft(c)} ({_ft(lo)}-{_ft(hi)}; {p:.0%} cross)")
        fe = f[-3:].mean()
        print(f"  {r['run']:8s} {r['name']}: naive {r['naive']:.4g}, Bayes {r['bayes']:.4g}; f at the end (mean of the "
              f"last 3) {fe:+.3f}; " + "; ".join(parts) +
              (f"; resamples moved by exact ties under 'at or past': {', '.join(ties)}" if ties else ""))
        extra = r["note"].replace(checks["pair_fwd"], "").lstrip("; ")
        if extra:
            print(f"      note: {extra}")
        print(f"      sample: {r['sample']}")
        if own is not None and r["run"] == "trained":
            print("      the topic's own: " + "; ".join(f"{lab} (level {lev:.4g}) {_fc(c)}" for lab, c, lev in
                                                     own[r["name"]]))
    print(f"this module's cross_level (strictly past) on the same resamples and paired levels vs td_tt_gap.crossing: "
          f"max |diff| over every row, level, resample {worst:.1e} (training episodes)")
    return worst


def synthesis():
    """-> through_training/wormholes/synthesis.npz (td_tt_gap.export), its table, and the checks"""
    import td_tt_gap as gap
    eps, F, Rf, P, Rp = _load_all()
    x, rows, checks = synthesis_rows(F, Rf, P, Rp)
    path = gap.export("wormholes", x, rows)
    worst = synthesis_report(x, rows, checks, own_crossings(F, Rf, P, Rp))
    assert worst < 1e-6 and checks["unit_counts_max_diff"] < 1e-9
    E = gap.load("wormholes")
    assert len(E["names"]) == len(rows) and np.array_equal(E["x"], x)
    return path


# The docstring's orders (A before B: A's 95% bound lies before B's 5% bound), re-checked over more bootstrap seeds.
ORDER_SEEDS = (BOOT_SEED,) + tuple(range(1, 21))     # the module's resamples and 20 more (the 11-02 re-check's set)
_O1 = ("corner truth 10%", "committed rise to half its peak")
_O2 = ("e slope passes naive", "grid unfolds 25-75%", "passes naive (800-880)", "proj_out 10%",
       "held per var corner 10%", "held per var one-var 10%", "held per var matched 10%", "adjacent corner back 50%",
       "partner corner back 50%", "corner truth 50%")
_O50 = ("held per var corner 50%", "held per var one-var 50%", "held per var matched 50%")
_O6 = ("phi slope 50%", "joint held 50%", "phi_out 50%")
_O8 = ("held per var corner 90%", "held per var one-var 90%", "held per var matched 90%", "phi slope 90%",
       "phi_out 90%", "joint held 90%")


def _pairs(a, b):
    return tuple((i, j) for i in a for j in b)


ORDER_CLAIMS = (                                    # (label, the pairs (A, B) that the docstring puts as A < B)
    ("timeline: corner answers start ~ committing rises < the next group", _pairs(_O1, _O2)),
    ("timeline: corner answers half-way and corner holding 10% after passing naive",
     _pairs(("passes naive (800-880)",), ("held per var corner 10%", "corner truth 50%"))),
    ("timeline and finding 1: that group < joint held 10%", _pairs(_O2, ("joint held 10%",))),
    ("timeline: joint held 10% < distance / same diagonal 50%",
     _pairs(("joint held 10%",), ("proj_out 50%", "same_out 50%"))),
    ("timeline: distance 50% < per-variable holding 50% (all three sets)", _pairs(("proj_out 50%",), _O50)),
    ("timeline: same diagonal 50% < corner-split holding 50%", (("same_out 50%", "held per var corner 50%"),)),
    ("timeline: holding / end-mass / distance / same diagonal 50% < direction / joint held / angle 50%",
     _pairs(_O50 + ("e slope 50%", "proj_out 50%", "same_out 50%"), _O6)),
    ("timeline: that group < committing half-way back", _pairs(_O6, ("committed back 50%",))),
    ("timeline: committing half-way back < the 90% points", _pairs(("committed back 50%",), _O8)),
    ("timeline: corner answers half-way before every holding / diagonal-leaving 50%",
     _pairs(("corner truth 50%",), ("joint held 50%",) + _O50 + ("proj_out 50%", "phi_out 50%", "same_out 50%",
                                                                  "amp_out 50%", "e slope 50%", "phi slope 50%"))),
    ("finding 4: each split set's TV half-way after its matched control",
     (("tv_ctl_two 50%", "tv_two 50%"), ("tv_ctl_one 50%", "tv_one 50%"))),
    ("finding 6: distance 50% < angle 50%", (("proj_out 50%", "phi_out 50%"),)),
    ("finding 7: one-axis moves' angle 50% < diagonal moves' angle 50%", (("phi_axis 50%", "phi_diag 50%"),)))
ORDER_NOT = (("same_out 50%", "held per var one-var 50%"),    # "~" in the timeline although these intervals do not
             ("same_out 50%", "held per var matched 50%"))    # overlap in the module's resamples


def order_events(F, P, Rp, W):
    """the Trained RNN's events that ORDER_CLAIMS / ORDER_NOT name, with bootstrap weights W (B, NE): name -> (t, lo,
    hi), with numbers()' definitions and names (the landmarks passes naive and the unfolding band are not resampled;
    the caller adds them)"""
    x = F["trained"]["x"]
    Ft, nvr = F["trained"], _ref_rows(Rp, "naive")
    ev = {}

    def cur(D, num, den, k=None):
        n = D[num] if k is None else D[num][..., k]
        return agg(n, D[den]), agg(n, D[den], W)
    for name, key, den, k, fr in (("corner truth", "corner_cat", "n_corner", 0, (.1, .5)),
                                  ("joint held", "joint_held", "n_two", None, (.1, .5, .9)),
                                  ("held per var corner", "held_two", "n_two_v", None, (.1, .5, .9)),
                                  ("held per var one-var", "held_one", "n_one_v", None, (.1, .5, .9)),
                                  ("held per var matched", "held_one_m", "n_w_one_m", None, (.1, .5, .9)),
                                  ("proj_out", "proj_out", "n_out", None, (.1, .5)),
                                  ("amp_out", "amp_out", "n_out", None, (.5,)),
                                  ("phi_out", "phi_out", "n_out", None, (.5, .9)),
                                  ("same_out", "same_out", "n_out", None, (.5,)),
                                  ("tv_two", "tv_two", "n_two_v", None, (.5,)),
                                  ("tv_one", "tv_one", "n_one_v", None, (.5,)),
                                  ("tv_ctl_two", "tv_ctl_two", "n_w_two", None, (.5,)),
                                  ("tv_ctl_one", "tv_ctl_one", "n_w_one", None, (.5,))):
        y, yb = cur(Ft, key, den, k)
        ev.update({f"{name} {int(f * 100)}%": c[:3] for f, c in frac_times(x, y, yb, fr).items()})
    y, yb = cur(Ft, "commit", "n_two")
    pk, _, c = back_time(x, y, yb, "max")
    ev["committed back 50%"] = c[:3]
    ev["committed rise to half its peak"] = crossing(x, y, yb, .5 * pk, 1., first=True)
    for lab, k in (("adjacent", 1), ("partner", 2)):
        y, yb = cur(Ft, "corner_cat", "n_corner", k)
        ev[f"{lab} corner back 50%"] = back_time(x, y, yb, "max")[2][:3]
    for kind in ("e", "phi"):
        y, yb = maps_slope(P["trained"], Rp, kind), maps_slope(P["trained"], Rp, kind, W)
        ev.update({f"{kind} slope {int(f * 100)}%": c[:3] for f, c in frac_times(x, y, yb).items()})
        ev[f"{kind} slope passes naive"] = crossing(x, y, yb, float(maps_slope(nvr, Rp, kind)[0]), 1.,
                                                    maps_slope(nvr, Rp, kind, W)[:, 0])
    for key, den in (("phi_diag", "n_jump_diag"), ("phi_axis", "n_jump_axis")):
        y, yb = cur(P["trained"], key, den)
        ev[f"{key} 50%"] = frac_times(x, y, yb, (.5,))[.5][:3]
    return ev


def order_check(F, P, Rp, fixed, ref=None, seeds=ORDER_SEEDS):
    """the docstring's orders over several bootstrap seeds: for each group of ORDER_CLAIMS, how many of its pairs have
    A's 95% bound before B's 5% bound in every seed, and the smallest margin (B's 5% bound - A's 95% bound, training
    episodes) over its pairs and the seeds; then the ORDER_NOT pairs. fixed: the landmarks {name: (t, lo, hi)}; ref:
    numbers()' events, which the first seed (the module's) must reproduce"""
    E = []
    for s in seeds:
        ev = order_events(F, P, Rp, boot_weights(seed=s))
        ev.update(fixed)
        E.append(ev)
    if ref is not None:
        bad = [k for k, v in E[0].items() if not np.allclose(v, ref[k], rtol=0, atol=1e-6)]
        assert not bad, f"order_events differs from numbers() for {bad}"

    def margins(a, b):
        return np.array([e[b][1] - e[a][2] for e in E])
    print(f"seeds {seeds[0]} (the module's) and {seeds[1]}-{seeds[-1]}; A < B: A's 95% bound before B's 5% bound; "
          f"margin = B's 5% bound - A's 95% bound (training episodes); the module's seed reproduces the events above: "
          f"{ref is not None}")
    low = (np.inf, "", "")
    for label, pairs in ORDER_CLAIMS:
        ms = [(a, b, margins(a, b)) for a, b in pairs]
        a, b, m = min(ms, key=lambda t: t[2].min())
        low = min(low, (m.min(), a, b))
        print(f"   {label}: {sum(bool((m_ > 0).all()) for _, _, m_ in ms)} of {len(ms)} pairs in all {len(seeds)} "
              f"seeds; smallest margin {m.min():,.0f} ({a} < {b}; {m[0]:,.0f} in the module's seed)")
        for a_, b_, m_ in ms:
            if not (m_ > 0).all():
                print(f"      NOT ROBUST: {a_} < {b_} in {int((m_ > 0).sum())} of {len(seeds)} seeds (margins "
                      f"{m_.min():,.0f} to {m_.max():,.0f})")
    print(f"   smallest margin of any claimed order over the seeds: {low[0]:,.0f} ({low[1]} < {low[2]})")
    for a, b in ORDER_NOT:
        m = margins(a, b)
        print(f"   '~' in the timeline: {a} < {b} in only {int((m > 0).sum())} of {len(seeds)} seeds (the module's "
              f"seed: bounds {E[0][a][2]:,.0f} vs {E[0][b][1]:,.0f}; margins {m.min():,.0f} to {m.max():,.0f}; the "
              f"reverse order in {int((margins(b, a) > 0).sum())})")
    return E


def numbers():
    """every number quoted in the docstring"""
    eps, F, Rf, P, Rp = _load_all()
    W = boot_weights()
    cross = passes_naive()
    unf, gx, gy = unfolding()
    x = F["trained"]["x"]
    ks = [int(np.flatnonzero(eps == e)[0]) for e in SHOW]
    events = {}

    def vals(y, nd=3, sel=None):
        return " ".join(f"{x[i]:.0f}:{y[i]:.{nd}f}" for i in (ks if sel is None else sel))

    def cur(D, num, den, k=None):
        n = D[num] if k is None else D[num][..., k]
        return agg(n, D[den]), agg(n, D[den], W)

    def tl(tt, name=None):
        if name:
            for f, c in tt.items():
                events[f"{name} {int(f * 100)}%"] = c[:3]
        return "; ".join(f"{int(f * 100)}% (level {c[3]:.3g}): {_fc(c)}" for f, c in tt.items())

    def rng(y, lo_x, hi_x=np.inf):
        s = (x >= lo_x) & (x <= hi_x)
        return f"{y[s].min():.3f}-{y[s].max():.3f}"

    print(f"checkpoints ({len(eps)}), training episodes: {', '.join(f'{v:.0f}' for v in x)}")
    print(f"passes naive at {cross:,.0f} (first of 114 probe checkpoints above naive); grid unfolding 25/50/75% at "
          f"{unf[.25]:,.0f} / {unf[.5]:,.0f} / {unf[.75]:,.0f}")
    print(f"forward set: corner splits {F['trained']['n_two'].sum():.0f} stream-steps (steps 2-30: "
          f"{F['trained']['n_two_all'].sum():.0f}), corner var-steps {F['trained']['n_two_v'].sum():.0f}, one-variable "
          f"{F['trained']['n_one_v'].sum():.0f} (2-30: {F['trained']['n_one_v_all'].sum():.0f}), corner episodes "
          f"{F['trained']['n_corner'].sum():.0f}, moves out {F['trained']['n_out'].sum():.0f}; matched-control coverage "
          f"{Rf['cover_two']:.3f} / {Rf['cover_one']:.3f}; step x e_B matching covers {Rf['cover_one_m']:.3f} of the "
          f"corner var-steps; grid check max |m| {np.nanmax(F['trained']['check_m']):.4f} / "
          f"{np.nanmax(F['echo']['check_m']):.4f}, answers {np.nanmin(F['trained']['check_a']):.4f} / "
          f"{np.nanmin(F['echo']['check_a']):.4f}")
    print(f"probes: corner splits {P['trained']['n_two'].sum():.0f}, corner var-steps {P['trained']['n_two_v'].sum():.0f}"
          f", one-variable {P['trained']['n_one_v'].sum():.0f} (step x e_B coverage {Rp['cover_one_m']:.3f}), corner "
          f"episodes {P['trained']['n_corner'].sum():.0f}, Bayes's moves >= 2 cells {Rp['n_jump_big'].sum():.0f}, "
          f"diagonal {Rp['n_jump_diag'].sum():.0f}, one-axis {Rp['n_jump_axis'].sum():.0f}, variable-steps "
          f"{Rp['n_e_st'].sum():.0f}")

    print("\n== 1 corner split held (joint), forward set")
    print(f"Bayes {ref(Rf, 'bayes_joint_held', Rf['n_two']):.3f} naive {ref(Rf, 'naive_joint_held', Rf['n_two']):.4f}")
    y, yb = cur(F["trained"], "joint_held", "n_two")
    print("trained", vals(y))
    print(f"   all checkpoints up to 880: {vals(y, 4, np.flatnonzero(x <= 880))}")
    for lo_x, hi_x in ((10, 160), (10, 240), (10, 360), (400, 880), (10, 880)):
        s = (x >= lo_x) & (x <= hi_x)
        print(f"   {lo_x}-{hi_x}: {y[s].min():.4f}-{y[s].max():.4f} (max at {x[s][np.argmax(y[s])]:.0f})")
    print("   times:", tl(frac_times(x, y, yb), "joint held"))
    ye, _ = cur(F["echo"], "joint_held", "n_two")
    print(f"ESN max over checkpoints {ye.max():.4f}, end {ye[-1]:.4f}")
    y2, yb2 = cur(F["trained"], "joint_held_all", "n_two_all")
    print(f"steps 2-30: end {end_value(y2):.3f}; times: {tl(frac_times(x, y2, yb2))}")
    yp, _ = cur(P["trained"], "joint_held", "n_two")
    ype, _ = cur(P["echo"], "joint_held", "n_two")
    print(f"probes: {vals(yp)}; ESN max {ype.max():.4f}")
    print(f"original median TV (TV averaged over the two variables per corner-split stream-step, then the median): "
          f"trained {vals(F['trained']['tv_med'])}; ESN end {F['echo']['tv_med'][-1]:.3f}; naive "
          f"{float(Rf['naive_tv_med']):.3f}")

    print("\n== 2 committed (both answers on a corner, split not held)")
    print(f"Bayes {ref(Rf, 'bayes_commit', Rf['n_two']):.3f} naive {ref(Rf, 'naive_commit', Rf['n_two']):.3f}")
    y, yb = cur(F["trained"], "commit", "n_two")
    print("trained", vals(y))
    pk, xp, c = back_time(x, y, yb, "max")
    events["committed back 50%"] = c[:3]
    print(f"   peak {pk:.3f} at {xp:,.0f}; half-way back to the end (level {c[3]:.3f}, stays below): {_fc(c)}; end "
          f"{end_value(y):.3f}")
    cr = crossing(x, y, yb, .5 * pk, 1., first=True)
    events["committed rise to half its peak"] = cr
    print(f"   rise: first reaches half its peak ({.5 * pk:.3f}) at {_fc(cr)} (first crossing: it falls back later)")
    ip = int(np.argmax(y))
    print(f"   peak 5-95% {np.percentile(yb[:, ip], 5):.3f}-{np.percentile(yb[:, ip], 95):.3f}; bootstrap argmax x "
          f"5-95%: {np.percentile(x[np.argmax(yb, 1)], 5):,.0f}-{np.percentile(x[np.argmax(yb, 1)], 95):,.0f}")
    ye, _ = cur(F["echo"], "commit", "n_two")
    print(f"ESN {vals(ye)}; range from 2,000: {rng(ye, 2000)}")
    yt, _ = cur(F["trained"], "truth", "n_two")
    print(f"answer = truth at the split: Bayes {ref(Rf, 'bayes_truth', Rf['n_two']):.3f} naive "
          f"{ref(Rf, 'naive_truth', Rf['n_two']):.3f}; trained {vals(yt)}; ESN end {cur(F['echo'], 'truth', 'n_two')[0][-1]:.3f}")

    print("\n== 3 per variable held: corner vs one-variable end splits")
    for src, D_, R_ in (("forward", F, Rf), ("probes", P, Rp)):
        print(f"-- {src}: naive corner {ref(R_, 'naive_held_two', R_['n_two_v']):.3f} one-var "
              f"{ref(R_, 'naive_held_one', R_['n_one_v']):.3f}; Bayes e corner {ref(R_, 'bayes_e_two', R_['n_two_v']):.3f}"
              f" one-var {ref(R_, 'bayes_e_one', R_['n_one_v']):.3f}")
        for r in RUNS:
            c2, c2b = cur(D_[r], "held_two", "n_two_v")
            c1, c1b = cur(D_[r], "held_one", "n_one_v")
            cm, cmb = cur(D_[r], "held_one_m", "n_w_one_m")
            cc, ccb = cur(D_[r], "held_two_c", "n_two_c")
            print(f"{r} corner {vals(c2)}")
            print(f"{r} one-var {vals(c1)}")
            print(f"{r} one-var matched to the corner splits' step x e_B {vals(cm)}")
            print(f"{r} corner (covered bins) {vals(cc)}")
            print(f"   end: corner {end_value(c2):.3f}, one-var {end_value(c1):.3f}, matched one-var {end_value(cm):.3f}"
                  f", covered corner {end_value(cc):.3f}")
            if r == "trained":
                fw = src == "forward"
                print(f"   times corner: {tl(frac_times(x, c2, c2b), 'held per var corner' if fw else None)}")
                print(f"   times one-var: {tl(frac_times(x, c1, c1b), 'held per var one-var' if fw else None)}")
                print(f"   times matched one-var: {tl(frac_times(x, cm, cmb), 'held per var matched' if fw else None)}")
                for nm_, a, ab in (("corner - one-var", c2 - c1, c2b - c1b), ("corner - matched one-var", c2 - cm,
                                                                                c2b - cmb)):
                    lo, hi = np.percentile(ab, [5, 95], 0)
                    sel = (x >= 3000) & (x <= 15000)
                    mb = ab[:, sel].mean(1)
                    print(f"   {nm_}: " + " ".join(f"{x[i]:.0f}:{a[i]:+.3f}[{lo[i]:+.3f},{hi[i]:+.3f}]" for i in ks
                                                    if x[i] >= 880))
                    print(f"      mean over 3k-15k {a[sel].mean():+.3f} [{np.percentile(mb, 5):+.3f}, "
                          f"{np.percentile(mb, 95):+.3f}]; last 3 {end_value(a):+.3f}")
            e2, _ = cur(D_[r], "e_two", "n_two_v")
            e1, _ = cur(D_[r], "e_one", "n_one_v")
            print(f"   smaller-end mass corner {vals(e2)}\n   one-var {vals(e1)}")
    print("-- the late raw gap (forward set, last checkpoint)")
    gap_source()

    print("\n== 4 TV per variable: corner / one-var / matched non-splits")
    for key, den in (("tv_two", "n_two_v"), ("tv_one", "n_one_v"), ("tv_ctl_two", "n_w_two"), ("tv_ctl_one",
                                                                                              "n_w_one")):
        y, yb = cur(F["trained"], key, den)
        ye, _ = cur(F["echo"], key, den)
        yp, _ = cur(P["trained"], key, den)
        print(f"{key}: naive {ref(Rf, 'naive_' + key, Rf[den]):.3f}; trained {vals(y)}; half {tl(frac_times(x, y, yb, (.5,)), key)}"
              f"; ESN end {end_value(ye):.3f}; probes end {end_value(yp):.3f}")

    print("\n== 5 corner answers at T (unselected corner episodes)")
    for k_, lab in ((0, "truth"), (1, "adjacent"), (2, "partner")):
        print(f"{lab}: Bayes {ref(Rf, 'bayes_corner_cat', Rf['n_corner'])[k_]:.4f} naive "
              f"{ref(Rf, 'naive_corner_cat', Rf['n_corner'])[k_]:.4f}; probes Bayes "
              f"{ref(Rp, 'bayes_corner_cat', Rp['n_corner'])[k_]:.4f} naive {ref(Rp, 'naive_corner_cat', Rp['n_corner'])[k_]:.4f}")
        for r in RUNS:
            y, yb = cur(F[r], "corner_cat", "n_corner", k_)
            yp, _ = cur(P[r], "corner_cat", "n_corner", k_)
            print(f"   {r} {vals(y, 4)}; end {end_value(y):.4f}; probes end {end_value(yp):.4f}, probes max "
                  f"{yp[1:].max():.4f} at {x[1 + int(np.argmax(yp[1:]))]:.0f}")
            if r == "trained":
                if k_ == 0:
                    print("      times:", tl(frac_times(x, y, yb), "corner truth"))
                    lev = y[0] + .5 * (end_value(y) - y[0])
                    print(f"      first reaches the 50% level ({lev:.3f}) at "
                          f"{_fc(crossing(x, y, yb, lev, 1., first=True))}; values 720-2,600: "
                          f"{vals(y, 3, [i for i in range(len(x)) if 720 <= x[i] <= 2600])}")
                else:
                    pk, xp, c = back_time(x, y, yb, "max")
                    i = int(np.argmax(y))
                    events[f"{lab} corner back 50%"] = c[:3]
                    print(f"      peak {pk:.4f} at {xp:,.0f} (5-95% {np.percentile(yb[:, i], 5):.4f}-"
                          f"{np.percentile(yb[:, i], 95):.4f}); half-way back (level {c[3]:.4f}): {_fc(c)}; values "
                          f"280-880: {vals(y, 3, [i for i in range(len(x)) if 280 <= x[i] <= 880])}")
            else:
                i = int(np.argmax(y))
                after = f"{y[i + 1:].min():.3f}-{y[i + 1:].max():.3f}" if i + 1 < len(y) else "-"
                print(f"      ESN max {y[i]:.3f} at {x[i]:,.0f}; after it {after}; from 2,000 {rng(y, 2000)}")

    print("\n== 6 moves out of corner splits (float32 posterior means)")
    for key, nd in (("proj_out", 2), ("amp_out", 2), ("phi_out", 2), ("same_out", 3)):
        print(f"{key}: Bayes {ref(Rf, 'bayes_' + key, Rf['n_out']):.3f} naive {ref(Rf, 'naive_' + key, Rf['n_out']):.3f}")
        for r in RUNS:
            y, yb = cur(F[r], key, "n_out")
            print(f"   {r} {vals(y, nd)}; end {end_value(y):.3f}")
            if r == "trained":
                print("      times:", tl(frac_times(x, y, yb, (.1, .5, .9) if key != "same_out" else (.5,)), key))
            if r == "trained" and key == "phi_out":
                k0 = 1 + int(np.argmin(np.where(x[1:] >= 100, y[1:], np.inf)))
                print(f"      initial {y[0]:.2f}; max over x <= 400 {y[x <= 400].max():.2f} at "
                      f"{x[np.argmax(np.where(x <= 400, y, -np.inf))]:.0f}; 80-240 {rng(y, 80, 240)}; minimum after "
                      f"100: {y[k0]:.2f} at {x[k0]:,.0f}; values 400-1,700: "
                      f"{vals(y, 2, [i for i in range(len(x)) if 400 <= x[i] <= 1700])}")
                print("      from the minimum: ", tl(frac_times(x, y, yb, y0=y[k0])))
            if r == "echo" and key == "phi_out":
                print(f"      ESN max over x <= 400 {y[x <= 400].max():.2f} at {x[np.argmax(np.where(x <= 400, y, -np.inf))]:.0f}")

    print("\n== 7 maps (probes, no event threshold)")
    for kind in ("e", "phi"):
        for how in ("within", "per_step", "pooled"):
            nv = float(maps_slope(_ref_rows(Rp, "naive"), Rp, kind, how=how)[0])
            bv = float(maps_slope(_ref_rows(Rp, "bayes"), Rp, kind, how=how)[0])
            print(f"{kind} slope ({how}): Bayes {bv:.3f} naive {nv:.3f}")
            if how == "per_step":
                k880 = int(np.flatnonzero(x == 880)[0])
                print("   equal weight per step, over the steps with >= 30 points / over every step with a slope: " +
                      "; ".join(f"{nm_} {a:+.4f} ({na} steps) / {b:+.4f} ({nb} steps)" for nm_, ((a, na), (b, nb)) in
                                (("naive", equal_weight(_ref_rows(Rp, "naive"), Rp, kind)),
                                 ("Trained at 880", equal_weight(P["trained"], Rp, kind, k880)),
                                 ("Trained at the end", equal_weight(P["trained"], Rp, kind, -1)),
                                 ("ESN at the end", equal_weight(P["echo"], Rp, kind, -1)))))
            if how == "within":
                n_t = Rp["n_e_st"].sum(0)[1:] if kind == "e" else Rp["n_jump_big_st"].sum(0)
                print(f"   the within-step slope = Sigma_t w_t beta_t, w_t = step t's share of Sigma_t Sigma_i "
                      f"(y_i^B - ybar_t^B)^2; points per step (steps 2-30): {' '.join(str(int(v)) for v in n_t)}")
                for nm_, Z_, k_ in (("naive", _ref_rows(Rp, "naive"), 0), ("trained at the end", P["trained"], -1),
                                    ("ESN at the end", P["echo"], -1)):
                    sp = step_parts(Z_, Rp, kind, k_)
                    print(f"   {nm_}: " + "; ".join(f"steps {a}-{b}: {p_:.1%} of the points, {w_:.1%} of the weight, "
                                                    f"slope {s_:.3f}" for (a, b), (p_, w_, s_) in
                                                    zip(((2, 2), (3, 5), (6, 30)), sp)) +
                          f"; without step 2: {step_parts(Z_, Rp, kind, k_, ((3, 30),))[0][2]:.3f}; the literal "
                          f"reading Sigma_t cov_t / Sigma_t var_t (each step normalised by its own count; not "
                          f"computed anywhere else): {literal_slope(Z_, Rp, kind, k_):.3f}")
            for r in RUNS:
                y = maps_slope(P[r], Rp, kind, how=how)
                print(f"   {r} {vals(y)}")
                print(f"      all {len(x)} checkpoints: min {y.min():.4f} at {x[np.argmin(y)]:,.0f}, max "
                      f"{y.max():.4f} at {x[np.argmax(y)]:,.0f}; 10-400 {rng(y, 0, 400)}; 10-1,200 "
                      f"{rng(y, 0, 1200)}; 2,000+ {rng(y, 2000)}")
                if how == "within":
                    yb = maps_slope(P[r], Rp, kind, W, how=how)
                    if r == "trained":
                        print("      times:", tl(frac_times(x, y, yb), f"{kind} slope"))
                        nvb = maps_slope(_ref_rows(Rp, "naive"), Rp, kind, W, how=how)[:, 0]
                        c = crossing(x, y, yb, nv, 1., nvb)
                        events[f"{kind} slope passes naive"] = c
                        print(f"      passes naive's slope ({nv:.3f}; paired per resample) and stays above: {_fc(c)}")
    for key, den in (("phi_diag", "n_jump_diag"), ("phi_axis", "n_jump_axis"), ("amp_diag", "n_jump_diag")):
        print(f"{key}: Bayes {ref(Rp, 'bayes_' + key, Rp[den]):.2f} naive {ref(Rp, 'naive_' + key, Rp[den]):.2f}")
        for r in RUNS:
            y, yb = cur(P[r], key, den)
            print(f"   {r} {vals(y, 2)}; range 240-2,000 {rng(y, 240, 2000)}; 10-1,200 {rng(y, 0, 1200)}")
            if r == "trained":
                print("      times:", tl(frac_times(x, y, yb, (.5,)), key))

    print("\n== 8 support for CORRECTIONS (probes): the step drift and the pooled maps")
    Z = P["trained"]
    for nm, st in (("Trained RNN, initial network", Z["e_st"][0]), ("Bayes", Rp["bayes_e_st"])):
        m = st.sum(0) / np.maximum(Rp["n_e_st"].sum(0), 1)
        print(f"mean e_v by step, {nm}: step 2 {m[1]:.3f}, step 30 {m[29]:.3f}")
    for xx in (40., 200.):
        i = int(np.flatnonzero(x == xx)[0])
        print(f"e map at {xx:.0f}, pooled bin means: " + " ".join(
            f"{v:.3f}" for v in Z["e_tbin"][i].sum(0) / Rp["n_e_tbin"].sum(0)) + "; within steps: " +
            " ".join(f"{v:.3f}" for v in maps_heat(Z, Rp, "e")[i]))

    print("\n== 9 synthesis export (td_tt_gap; the shared resamples gap.boot_counts(), paired): gap fraction f = (y - "
          "naive) / (Bayes - naive); c0 = passes naive's level for good, c50 = past the naive-Bayes midpoint for good")
    xs, rows, checks = synthesis_rows(F, Rf, P, Rp)
    synthesis_report(xs, rows, checks, own_crossings(F, Rf, P, Rp, W))

    print("\n== events (5-95% intervals); A before B only if A's interval ends before B's begins")
    events["passes naive (800-880)"] = (cross, 800., cross)
    events["grid unfolds 25-75%"] = (unf[.5], unf[.25], unf[.75])
    ev = sorted(events.items(), key=lambda kv: kv[1][0])
    for nm, c in ev:
        print(f"   {nm:36s} {_fc(c)}")
    print("   orderings (consecutive events):")
    for (a, ca), (b, cb) in zip(ev[:-1], ev[1:]):
        print(f"      {a} -> {b}: {'before' if ca[2] < cb[1] else 'cannot be ordered at this resolution'}")

    print(f"\n== the docstring's orders re-checked over {len(ORDER_SEEDS)} bootstrap seeds")
    order_check(F, P, Rp, {k: events[k] for k in ("passes naive (800-880)", "grid unfolds 25-75%")}, events)
    return events


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "figure"
    if cmd == "grid":
        grid_side()
    elif cmd == "forward":
        forward(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0, int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    elif cmd == "stats":
        for rn in RUNS:
            probe_stats(rn)
            fwd_stats(rn)
        probe_refs()
        fwd_refs()
    elif cmd == "numbers":
        numbers()
    elif cmd == "timeline":
        timeline()
    elif cmd == "synthesis":
        synthesis()
    elif cmd == "figure":
        figure()
        figure_maps()
