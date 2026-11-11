"""Memory cuts with donor states through training (John 11-02: "How does the network get to the correct answer before
learning optimal bayes dynamics?"; approved experiment (1) of the three inference-time causal tests on the stored
checkpoints; no new trainings). Revised 11-02 after three adversarial checks (all FIXABLE; the third on wording and
figure text only). First: the wrong-cell cut is read as a ceiling, not as reliance; new controls (shuffled own
histories, belief-matched and flat-belief donors from generated episodes, averaged states, random-donor seeds and
schemes); coupling row by row; need onsets with their level bound; the uncut build-up; the network-defined revisions.
Second: the belief-matched swap is set against its matched null (Bayes swapped with donors matched on ITS OWN marginals
from the same pools: bayes_bm_null), not against Bayes under the network's donors; the network's residual is resolved
above that null at 960-3,000, so re-evaluation from raw history is 'not the main route' instead of FAILED; the restart's
small extra cost at 480, 720 and 880 (read as a lower bound on a restart penalty) and the larger cost of a consistent
wrong-cell state are reported; step types in additive units; no ordering of overlapping intervals; both figures cut to
the panels whose results survived (5 each; the rest is below).
-> main/REPLICATION/figure_bin/td_tt_memcuts.png         figure_answers(): the answers under the restart, uncut,
                                                          under another cell's state and under the controls
   main/REPLICATION/figure_bin/td_tt_memcuts_where.png   figure_where(): the families, and where the restart's excess
                                                          falls
   numbers()  every number below (~30 s);  checks() -> OUT/checks.json;  ctl_checks() -> OUT/controls/checks.json
   cache main/DATA/experimental/training_dynamics/through_training/memcuts/ (587 MB, of which controls/ 265 MB;
   compressed npz; per-step sums stored float32: rounding <= 6e-8 of each array's scale, numbers() unchanged): bayes.npz
   and bayes_where.npz (Bayes and naive under the same cuts); trained/ck_EEEEE.npz (46) and echo/ck_EEEEE.npz (10): per
   condition the 24 score rows with their 1,000 shared resamples, per-step sums per (environment, step, entanglement
   fifth / Bayes step group), per-probe answers, centred log belief and margin of the true value at T;
   trained/where_EEEEE.npz (13) and echo/where_EEEEE.npz (5): the competitor decomposition; controls/bayes_ctl.npz
   (Bayes under the checkpoint-independent controls), controls/bayes_bm_null.npz (the matched null: per pool seed
   Bayes's statistics, its donors' counts, cells and distances) and controls/trained/ck_EEEEE.npz (46): per control
   condition the 18 cheap score rows with their resamples, answers and margins at T, per-step answers per environment;
   for the generated-pool donors Bayes under the network's donors and the matching diagnostics; the network-defined
   revision probes.
   Every curve and interval below is rebuilt from these.

QUESTION. The stored start-state truncation (td_analysis an6; td_tt_memory_events finding 7) shows that through the
answers phase the network's answers lean on memory older than 5 steps far more than Bayes's (goal accuracy lost by a
5-step cut .14-.16 at 1.6-2k training episodes vs Bayes's .031). Hypothesis: non-Bayesian re-evaluation from long raw
history, relaxing to Bayes's level as holding and coupling develop. Test: rerun the cut with real mid-episode donor
states, score every behaviour family on the cut beliefs, and set each against Bayes under the identical cut; then
(controls) swap in states that carry the cell's content from another history, or no own-cell content with the clock
kept.

DATA
  Checkpoints (td_tt_intervene.load): Trained RNN 46 = the stored checkpoints nearest 46 log-spaced targets (10-39,996
  training episodes; 39 distinct) plus 7 in the answers phase (880, 960, 1,120, 1,360, 1,680, 1,920, 2,320); ESN (echo)
  10 log-spaced (10, 20, 60, 160, 400, 1,040, 2,600, 6,400, 16,000, 39,996). 'Where' pass: 13 Trained RNN checkpoints
  (400 ... 39,996), 5 ESN. Controls (w = 5, Trained RNN only): random-cell seeds and averaged states at all 46; the
  shuffled, belief-matched and flat-belief states at 16 (400, 560, 800, 880, 960, 1,120, 1,360, 1,680, 2,080, 2,600,
  3,000, 4,400, 6,400, 11,200, 19,200, 39,996), 3 seeds each; the matched null (Bayes; checkpoint-independent) for the
  3 pool seeds; not run on the ESN (it loses <= .009 under every engine source). x = training episodes
  (td_tt_gap.train_x; the initial network at 10). Probes (analysis/probes.npz): 10,000 episodes = 50 test environments
  x 100 latent pairs (cells) x 2 trajectories, 30 steps. Engine: td_tt_intervene.forward (fp32 hand-written LSTM,
  validated there), one forward per source with w = 2, 5, 10; the controls' forward (ctl_forward) uses the engine's
  weights, drive table and cell, with its own window reruns.

DEFINITIONS
  Memory cut (w, source) (td_tt_intervene hook a): at every step t > w the cell is rerun over the inputs t-w+1..t from a
    state placed at t - w, and the read-out of the rerun state at t replaces the network's increment at t; t <= w keeps
    the network's own increments; the belief at every step is the cumulative sum, as in the network (the read-out is
    cumulative, so the cut removes only what the STATE carries from before t - w + 1, never earlier increments).
    Sources of the state at t - w: start = the trained start state (h0, c0) (= the stored trunc_right, reproduced
    exactly; 'the restart'); same cell = the state at t - w of the other trajectory of the same cell and environment
    (probe row ^ 1: the same latent pair, other observations); random cell = the state at t - w of one fixed probe of
    the same environment with another cell (seed 20261201, the same donor at every step; 'one donor episode').
  Bayes under the same cut (exact, float64; bayes_logq): for t > w Bayes's step-t increment is its marginal change from
    adding observation t to the history (the donor's observations 1..t-w, own t-w+1..t-1); start = no donor
    observations = td_analysis's windowed Bayes; t <= w Bayes's own increment; the beliefs are the cumulative sum,
    normalised. This is what an exact observer loses under the identical substitution: the matched reference for every
    source (a donor carries wrong (random cell) or extra (same cell) evidence for Bayes too). Naive's increments are
    memoryless: no cut changes naive (its loss is 0: the grey dotted zero lines).
  Controls (w = 5). Random-cell seeds: the engine's scheme (one fixed other-cell donor per probe) under 3 more seeds,
    and a fresh random other-cell probe of the environment at every window start (4 seeds; 'a new donor at each
    step'). Average of the cell's two states: (h, c) at t - 5 = the mean of the two trajectories' states of the probe's
    cell (cell content kept, each trajectory's specifics averaged; Bayes analogue, as the check's: the log of the
    mixture of the two posteriors). Environment-mean state: the mean (h, c) at t - 5 over the environment's 200 probes
    (Bayes: the log mixture of the 200 posteriors). Shuffled own history: the network rerun from (h0, c0) over the
    probe's own observations 1..t-5 with each of the 5 observation bits' sequences permuted in time (3 seeds): every bit
    keeps its count, so Bayes's joint posterior at t - 5 is exactly the probe's own while the history differs.
    Belief-matched other cell: the state at t - 5 of a generated episode of the same environment, another cell and
    another observation history, whose network marginals at t - 5 are the closest to the probe's (total variation, the
    larger of the two variables'); pools of 2,000 generated episodes per environment (20 per cell, drawn as the probes:
    o_ti ~ Bernoulli(L_i(r0, r1)) with the environment's tables), 3 pool seeds: the same belief as the network reads it
    out, from another latent pair's history. Its matched null (bayes_bm_null): Bayes swapped with the episode of the
    same pool (another cell, another history) whose BAYES marginals at t - 5 are closest to the probe's own Bayes
    marginals: the same pools, candidates and distance, matched on the exact observer's own belief; checkpoint-
    independent. Bayes under the network's own donors is no matched reference: those donors are chosen by the
    network's read-out, so where the network's belief is poor they sit far from the probe in Bayes's sufficient
    statistic; it is reported only as what those histories carry for an exact observer. Flat-belief other cell: the
    state at t - 5 of one of the 10 generated episodes of another cell and another history with the most uniform
    network marginals at t - 5 (largest summed entropy), drawn per probe and window start, 3 pool seeds: the
    mid-episode clock with as little content as the network's own read-out shows (when none of the 70 flattest episodes
    qualifies, the flattest is taken: CHECKS). Bayes under the identical substitution: the donor's observations 1..t-5
    (their per-bit counts suffice) in place of the probe's; for the averaged states the mixtures above. Controls are
    scored with td_tt_intervene.score 'cheap' (18 rows).
  Memory dependence of a behaviour row y = y(intact) - y(cut) in y's own units; Bayes's = y_Bayes - y_Bayes(cut);
    excess = network's - Bayes's (> 0: the network loses more than Bayes would). In panels a-c of the 'where' figure the
    dependence is divided by the row's naive-to-Bayes gap (y_Bayes - y_naive: td_tt_gap's fixed scale per row). Rows:
    td_tt_intervene.score 'full' on the probes: answers (goal accuracy at T, both right, far errors, fold index, ...),
    splitting (split gradient, ...), holding (end-mass slope; held per variable at Bayes's corner / one-variable end
    splits: 720 / 365 var-steps; joint held), coupling (jump excess angle, G, angle of jumps >= 4 cells, direction
    slope).
  Where (per probe, variable, step t > w; e_t = the step's increment of the centred log belief, c_t = the centred log
    belief; Bayes-defined groups, fixed through training):
    competitor's gain d_t (main): j_c = the true value's strongest competitor in the CUT belief at T; d_t =
      [e_t(r) - e_t(j_c)]_intact - [e_t(r) - e_t(j_c)]_cut, so Sigma_t d_t = L_int - L_cut with L = c_T(r) - c_T(j_c)
      (L_cut = the cut's margin; L_int >= the intact margin): the log-odds the cut takes from the true value against
      the competitor that profits, exactly additive over steps; per variable-step, all variables or only those whose
      answer the cut loses;
    groups: the variable's own entanglement fifth within its step (probes.npz ent; 4,000 var-steps per fifth and step);
      Bayes's step type: revision (Bayes's answer for the variable changes at t), settled (Bayes's answer from t-1 to T
      is its answer at T), other; probe types (goal variable, td_tt_memory_events' definitions): Bayes revises (q > .6
      on a value that is not its answer at T at some step 8-24; 2,685), Bayes settled (same answer at every step 8-30;
      2,844), neither (4,471), naive far error (2,430); network-defined revision probes (the check's definition): the
      network's goal marginal puts more than .6 on a value that is not its own answer at T at some step 8-24, and that
      answer is right; compared with the other right-at-T probes on the w5 start cut, raw and matched on the intact goal
      margin at T (deciles of the right-at-T probes, weighted by the revision probes per decile).
    secondary: single-step cut (only step t's increment from the cut; answer and margin at T recomputed); update
      change |e_cut - e_int|^2; evidence for the true value e_t(r); local update error err (td_entanglement_memcut);
      memory use u = err_cut - err_intact and the q5 contrast = u in the most entangled fifth minus the mean u of the
      other four fifths of the same steps (step-matched by construction; not matched on FR, entropy or Bayes's update
      size as td_tt_memory_events' controls); echo regression e_t ~ a N_t + b c_(t-1) per fifth (N = naive's centred
      increment; naive: a = 1, b = 0 exactly).
  Uncut build-up: the intact network's goal accuracy at step t, against Bayes's and naive's; the step at which it first
    reaches Bayes's accuracy at step 5 (or 10) (linear between the two steps around the crossing; per resample against
    Bayes's resampled level; never = beyond step 30).
  Margin of the true value at T: log q(r) - max_(j != r) log q(j), per probe and variable (positive = right).
  Statistics: per-environment sums, pooled as ratios of sums; 5-95% over the 1,000 shared resamples of the 50
    environments (td_tt_gap.boot_counts, seed 20261103), paired with each other and with the timeline's rows (the
    families' c10 / c50: td_tt_timeline.cross on the exports). Transient timing (td_tt_memory_events' rule): peak = the
    maximum of the 3-checkpoint running mean; rise = first crossing of half the peak; fall = after which the curve stays
    below half the peak (log-interpolated; levels recomputed per resample; never = after the end). Need onset = after
    which the network's dependence stays above half of Bayes's under the same cut. Two times are ordered only when their
    intervals do not overlap ('with' otherwise); (p) = paired share P(a < b). Seed means are reported with their range
    over seeds.

RESULTS (Trained RNN unless named; training episodes; [5-95%]; w = 5 unless named; goal accuracy at T = 'answers')
 1 Reproduction and the start-state transient. From the start state the cut's answers equal the stored trunc_right at
   all 46 checkpoints (w = 2 and 5: 0 of 20,000 differ at each; intact 0). Answers lost: .020 [.012, .029] (400), .091
   (880), .133 (1,360), .150 [.137, .163] (1,680), .156 [.143, .171] (2,080), .155 (3,000), .122 (4,400), .085
   (7,600), .063 (11,200), .035 [.028, .042] (39,996); largest single checkpoint .160 at 1,760; Bayes .031 [.025,
   .037]. At 1,680 the cut keeps +.043 of the network's +.193 gain over naive, where Bayes's 5-step window keeps +.276
   of +.307. Excess over Bayes: peak +.123 [.114, .137] at 1,920; rise 892 [838-1,066] (with passing naive 835
   [798-876], .02; before the accuracy c50 1,394 [1,285-1,489], 1.00); fall 6,845 [5,868-7,712]: after the c50s of the
   end-mass slope (3,936), held per variable (4,983) and angle of jumps >= 4 cells (3,855), with those of G (5,656),
   excess angle (6,248), held at one-variable splits (5,082) and joint held (9,245), before the direction slope's
   (10,113); end +.004 [-.003, +.012] (td_tt_memory_events on all 114 checkpoints: rise 905, peak +.127, fall 6,698).
   w = 2: excess peak +.282 [.265, .301] at 2,320, fall 22,495 [20,279-28,104], end +.100 [.087, .114]; w = 10: peak
   +.010 [.005, .016] (3-checkpoint running mean, at 1,680; the 1,680 checkpoint alone +.011 [.003, .019]).
 2 The engine's donors.
   Same cell (the other trajectory of the same latent pair): the network loses nothing at any checkpoint: -.005 (400),
   -.005 (880), -.010 [-.016, -.005] (1,360), -.008 (1,680), -.016 [-.022, -.010] (2,080), -.002 (4,400), +.003
   [-.002, +.007] (39,996), while Bayes under the same substitution loses .013 [.009, .018]; the excess is <= 0
   throughout (min -.029 at 2,080; end -.010 [-.014, -.006]; never resolved above 0). w = 2: -.024 to -.028 at
   1,360-2,080 (Bayes +.020); w = 10: within +-.004 (Bayes .009). So the old content the network uses is not specific
   to its own trajectory: the other trajectory of the cell serves (a net gain of .005-.016 through 400-2,080, possibly
   the donor's extra evidence), and so does the average of the cell's two states (RESULTS 3). This does not show that
   the network needs no memory older than 5 steps: the same-cell state supplies the very content it would rely on.
   Random cell (the engine's donor: one donor episode per probe): .120 (880), .202 [.187, .218] (1,680), .264 (3,000),
   .315 (7,600), .348 [.336, .361] (39,996); Bayes .344 [.334, .354] (end excess +.004 [-.004, +.012]). Its levels:
   network .373 / .456 / .452 / .431 / .412, Bayes .427, naive .464. With this donor a wrong cell's state older than 5
   steps removes all of either observer's gain over naive, so old content is used and can mislead; but the cut sits at
   the ceiling for both observers and cannot measure how much of the gain rests on old content: Bayes's gain rests
   mostly on the last 5 observations (its window keeps +.276 of +.307) and still falls below naive here. The network's
   dependence tracks its gain over naive: the fixed wrong-cell state takes it to about naive's level or below (4-seed
   means .374 / .449 / .455 / .431 / .411 at 880 / 1,680 / 3,000 / 7,600 / 39,996 against intact gains over naive of
   +.029 / +.193 / +.252 / +.282 / +.296; at 880 the loss .119 is 4x the gain). So its dependence reaches half of
   Bayes's at 1,251 [1,092-1,370] (with the accuracy c50 1,394, 1.00) because it tracks that gain, not a need. Seeds
   and schemes (the magnitudes and the dates depend on them): the same scheme under 4 seeds .202-.212 at 1,680 (end
   .344-.353; Bayes .343-.346), onset of half Bayes's 1,166-1,287; a new donor at each step (4 seeds) .149-.154 at
   1,680 (end .293-.295; Bayes .337-.345), onset 2,146-2,378: consistent wrong content misleads more than content that
   changes at every step. With a new donor at each step the network keeps up to +.06 over naive (.480 / .506 / .522 /
   .494 at 1,360 / 1,680 / 3,000 / 7,600; .466 at the end) while Bayes falls to .430.
 3 Controls: what the old state must carry.
   (i) Swaps that keep the cell's content from another history (the network's read-out, Bayes's counts, or the cell)
   cost little: at most .016, against the restart's .150 at 1,680. The belief-matched swap, matched only on the
   network's read-out, is among them, so the old state acts mostly, not entirely, through the read-out belief.
   Belief-matched other cell (another latent pair's history; network marginals matched to a median total variation of
   .043 (400) to .076 (6,400), < .10 for 68-90% of the probe-windows; 36-49% of the donors share neither latent value
   with the probe): +.001 (400), +.009 (880), +.011 (1,360), +.016 (1,680; seeds .015-.018), +.012 (2,080), +.011
   (3,000), +.007 (4,400), +.008 (6,400), +.005 (11,200), +.003 (39,996). Its matched null, an exact observer swapped
   with donors matched on its own marginals from the same pools (median TV .071, vs the network's .043-.076; 36% of the
   donors share neither latent value), loses .0026 / .0049 / .0053 (pool seeds; mean .0043). Network minus null,
   pool-paired: +.008 [+.003, +.014] (960), +.009 [+.004, +.014] (1,120), +.007 [+.002, +.012] (1,360), +.012 [+.006,
   +.018] (1,680), +.007 [+.003, +.012] (2,080), +.009 [+.004, +.015] (2,600), +.007 [+.002, +.012] (3,000): resolved
   above 0 at 960-3,000; unresolved at 560-880 and 4,400-39,996; -.003 [-.007, -.000] at 400, where the network's match
   is tighter (TV .043). The residual may partly be matching sensitivity: by quartile of each observer's own median
   matching distance per probe (1,680, mean of the 3 pools; the quartiles are different probe sets for the two
   observers) the network loses .010 / .012 / .017 / .025 and the null .002 / .002 / -.001 / .013 (pool 20261241: .006 /
   .014 / .016 / .024 vs .000 / .002 / -.002 / .011): both lose most in their loosest quartile, and the residual (+.007
   [-.001, +.016] / +.010 [+.002, +.018] / +.018 [+.008, +.028] / +.012 [+.003, +.022]) is smallest, and unresolved, in
   the tightest quartile, against +.010 to +.018 in the others, so these pools cannot show whether any of it survives an
   exact match. Bayes under the network's own donors (.028 at 400, .016 at 1,680, .005 at the end) is no matched
   reference (DEFINITIONS); it only shows that those histories carry an exact observer's belief worst where the
   network's own belief is poorest. Shuffled own history (Bayes 0 exactly): -.001 to +.007 (seed means), resolved above
   0 at 1,680 / 4,400 / 6,400: +.007 [+.003, +.010] (seeds +.006 to +.008) / +.004 / +.004. Same cell, other trajectory:
   -.016 to +.003 (Bayes .013; RESULTS 2). Average of the cell's two states: -.012 to +.003 (Bayes .004). Families under
   the belief-matched swap against the null: held per variable lost .034 / .086 / .065 / .025 at 1,680 / 4,400 / 6,400 /
   39,996 vs the null's .054 (network minus null +.0315 [-.0023, +.0638] at 4,400, -.029 [-.049, -.008] at the end),
   angle of jumps >= 4 cells .46 / 1.11 / 1.21 / .71 deg vs .68 (+.53 [+.01, +1.06] at 6,400; the restart's 4.10);
   shuffled: held per variable <= .009, jumps >= 4 cells <= .41 deg (Bayes 0).
   (ii) States without the probe's cell content, mid-episode. Flat-belief other cell (its read-out entropy median
   3.0-3.5 nats vs the probe's own 2.0-2.5; uniform 4.61) / random cell, a new donor at each step / environment-mean
   state: .016 / .023 / .019 (400), .085 / .079 / .076 (880), .132 / .129 / .131 (1,360), .155 / .151 / .153 (1,680),
   .170 / .165 / .159 (2,080), .204 / .194 / .197 (3,000), .219 / .218 / .218 (4,400), .294 / .294 / .297 (39,996),
   against the restart's .020 / .091 / .133 / .150 / .156 / .155 / .122 / .035 (Bayes: flat-belief .29-.32 over
   checkpoints and seeds, random .34, environment mean .038, start .031). Restart minus these three (paired): at
   960-2,080 every difference is unresolved (-.014 to +.014): no restart cost these controls detect there. At 480, 720
   and 880 the restart costs a little more: +.014 [+.003, +.024] (480), +.012 [+.001, +.024] (720), +.012 [+.001, +.023]
   (880) than the new-donor random cells, +.015 [+.005, +.027] than the mean state at 880 (+.006 [-.006, +.019] than the
   flat-belief state, unresolved); at 560 and 800 no positive difference is resolved. So, where resolved, the restart
   costs .012-.015 more at 480, 720 and 880 (intervals up to +.027; about a third of the restart's .038 at 480 and
   13-16% of its .091 at 880). The difference is a restart penalty minus whatever these states cost of their own
   (CONFOUNDS): if those costs are not negative, a lower bound on the penalty (equal to it only if they are 0), not an
   upper bound. The flat-belief state costs more than the restart at 560 (-.015 [-.023, -.005]); before 400 the
   new-donor and mean states cost slightly more than the restart at most checkpoints (-.003 to -.020 at 20-220); from
   2,320 (new-donor, mean) and 2,600 (flat-belief) on, the restart costs less at every checkpoint (-.039 to -.049 at
   3,000, -.096 / -.097 at 4,400, -.26 at the end), as for Bayes. One consistent wrong cell's state (the engine's
   scheme) costs more than the restart at every checkpoint from 560 on (and at most before 400): restart minus it -.020
   [-.028, -.010] (560), -.029 [-.040, -.017] (880), -.058 [-.073, -.044] (1,680), -.075 [-.092, -.058] (2,080). So in
   the answers phase a mid-episode clock without the content is no better than the restart at 960-2,080 and, for some of
   these states, a little better at 480, 720 and 880, while consistent wrong content is worse; from about 2,300-2,600
   the network, like Bayes, does much better with no old content than with another cell's. Holding behaves alike in its
   own phase: held per variable lost at 4,400 / 6,400: restart .178 / .194, flat-belief .172 / .202, new-donor random
   .164 / .229, mean .140 / .189 (Bayes .069 / flat-belief .27-.29 / random .33 / mean .071); end-mass slope: restart
   .111 / .143, flat-belief .118 / .165, new-donor random .113 / .182, mean .089 / .139. Late the flattest read-outs
   mislead as much as random cells' states (6,400: .242 vs .242; Bayes .30 with these histories): a flat read-out does
   not mean a content-free state.
   (iii) Uncut build-up: in normal episodes the network's gain over naive builds up more slowly than Bayes's. Goal
   accuracy above naive at step 5 / at T: 880 -.039 [-.053, -.025] / +.029 [+.002, +.054]; 1,680 +.028 [+.018, +.037] /
   +.193 [+.170, +.216]; 4,400 +.070 / +.270; 39,996 +.096 / +.296; Bayes +.099 / +.307 (step 2 +.044, step 8 +.152,
   step 15 +.219). The uncut network reaches Bayes's step-5 accuracy (.424) at step 13.6 [11.9, 17.0] (880), 8.3 [7.7,
   9.4] (1,360), 7.5 [7.1, 8.1] (1,680), 6.0 [5.7, 6.2] (4,400), 5.1 [5.0, 5.2] (39,996), never at 400; and Bayes's
   step-10 accuracy (.547) at step 15.4 [14.5, 16.4] (1,680), 11.1 [10.8, 11.6] (4,400), 10.1 [10.0, 10.3] (39,996),
   never at 880. (As a share of Bayes's gain, supplementary: step 5 -.40 / .28 / .71 / .98 vs T .09 / .63 / .88 / .97
   at 880 / 1,680 / 4,400 / 39,996; the step-5 denominators are small.)
 4 The start-state excess is a larger belief change, not more fragile answers: at 1,680 the cut moves the true value's
   margin at T by 1.62 nats on average (median 1.00) vs Bayes's .12 (.06); applied to Bayes's own margins the
   network's changes would cost Bayes .195 of its accuracy, Bayes's changes cost the network .026. Same cell: -.063
   (gain; Bayes +.031). Random cell: 3.02 (median 1.28) vs Bayes 8.32 (2.05). End, start state: .19 vs .12.
 5 Where the start-state excess sits (competitor's gain, nats per variable-step after the window).
   By entanglement: at 1,680 .110 / .110 / .110 / .126 / .143 from the least to the most entangled fifth vs Bayes .002 /
   .006 / .007 / .010 / .015. The excess over Bayes is present (resolved above 0) in every fifth at every checkpoint: in
   the least / most entangled fifth +.044 / +.078 (400), +.091 / +.155 (960), +.108 [.092, .125] / +.128 [.107, .148]
   (1,680), +.107 / +.123 (2,080), +.077 / +.118 (3,000), +.054 / +.075 (4,400), +.019 / +.019 (11,200); end +.005 /
   +.0029 [+.0005, +.0056]. The excess is about equal across fifths only at 1,680-2,080 (its most-minus-least difference
   +.020 [-.001, +.041] and +.016 [-.004, +.037]) and from 11,200 (+.000, -.001, -.0024 [-.0045, -.0003]); larger in the
   most entangled fifth at 400-1,200 (+.034 to +.068) and 3,000-6,400 (+.041 / +.021 / +.011); Bayes's own
   most-minus-least need is +.013 [.011, .016]. At 39,996 the network's per-fifth loss .007 / .012 / .013 / .015 /
   .018 is close to Bayes's in size and grade.
   By Bayes's step type: revision .125 [.108, .141], settled .120 [.106, .135], other .118 [.102, .136] at 1,680 (Bayes
   .012 / .008 / .006); 39,996: .015 / .014 / .010. In additive units the network's loss is larger on revision than on
   other steps by about Bayes's own margin (+.0061 [+.0030, +.0091] at 1,680 vs Bayes's +.006; resolved at every
   checkpoint from 1,680 on: +.0046 to +.0081; revision minus settled +.0050 [-.0043, +.0139] at 1,680, +.0032 [+.0002,
   +.0063] at 6,400), so the excess over Bayes is flat over Bayes's step types at 1,680-6,400 (.113 / .111 / .112 at
   1,680; every difference unresolved) and smaller on revision steps at 400-1,200 (revision minus other -.007 to -.013
   at 400-800; revision minus settled -.0163 / -.0185 at 960 / 1,200) and from 11,200 (revision minus settled -.003 to
   -.005): it is not larger on revision steps at any of the 13 checkpoints.
   By episode type (answers lost): Bayes revises .146 [.126, .166], settled .153 [.135, .172], neither .151 [.136,
   .166], naive far error .143 at 1,680 (Bayes .047 / .010 / .034 / .024); at 39,996 .048 / .017 / .039 / .038. Not
   concentrated on Bayes-revises episodes from 300 (where panel e starts) through 11,200: there revises is the largest
   of the three only at 340 and 720 (not resolved above neither), and settled exceeds neither at 400-560 and 800-1,360
   (resolved, paired; e.g. .109 [.087, .132] vs .073 [.060, .086] at 880, .158 [.137, .179] vs .117 [.101, .133] at
   1,360). From 13,600, revises is the largest at 6 of the 7 checkpoints (not at 19,200) but never resolved above
   neither (end .048 vs .039). Like Bayes's (.047 vs .010), the loss is larger on revises than on settled episodes at
   every checkpoint from 4,400 on, and at 340 (resolved, paired; .117 vs .089 at 4,400).
   By the network's own revisions (net-defined revision probes vs the other right-at-T probes, matched on the intact
   goal margin at T): +.208 [.113, .291] (400), +.135 [.103, .164] (880), +.041 [.018, .063] (1,360), +.006 [-.018,
   +.032] (1,680; raw .401 vs .336 on 1,645 revision probes), +.004 (2,080), +.009 (3,000), +.000 [-.017, +.020]
   (4,400), +.011 (6,400), +.025 [.010, .040] (11,200), +.045 [.028, .063] (39,996): not concentrated on the network's
   revisions at 1,680-6,400; concentrated there at 400-1,360 and from 11,200 (Bayes's window loses .020-.082 vs
   .011-.041 on the same sets).
   Along the episode: answers lost at steps 10 / 20 / 30 .041 / .119 / .150 at 1,680 (Bayes .004 / .020 / .031); the
   loss grows over every step after the window.
   Kind of error: the lost answers become far errors (2+ values off): +.162 [.142, .183] at 1,680 (Bayes +.016); fold
   index excess peak +2.0 [1.7, 2.5] at 1,120 (Bayes .008), fall 6,047.
   Single steps: cutting one step at a time, the answers lost sum to .044 at 1,680 against .149 for all steps cut
   (per variable; Bayes .046 vs .028); the competitor's gain only flips answers when it accumulates over the steps.
 6 Families (w = 5). Start state: answers, splitting and holding lose more than Bayes while they develop. Splitting:
   split gradient lost .034 [.018, .052] at 1,680 where Bayes's needs no memory older than 5 steps (+.002 [-.010,
   +.014]); excess peak +.030 at 1,360, fall 7,293 [3,019-21,296]. Holding: held per variable (corner) .178 [.129, .228]
   at 4,400 vs Bayes .069 [.051, .089] (excess peak +.123 [.086, .161] at 5,200, rise 2,878 [1,664-3,914], fall 24,579
   [7,724-33,626]; end .058), joint held .194 at 7,600 vs .111 (peak +.122 at 6,400), end-mass slope .123 at 3,000 vs
   .065 (peak +.066 [.030, .111] at 6,400, fall 7,943); held at one-variable splits no excess. Coupling, row by row: G
   .047-.079 at 3,000-39,996 vs Bayes .088 (no excess after 120 training episodes; peak +.015 [.001, .032] at 100, the
   untrained network); excess angle .80-1.20 deg vs 1.04 (peak excess +.097 [-.061, +.312] at 6,400: not resolved);
   angle of jumps >= 4 cells 3.4-4.1 deg at 3,000-7,600 vs 1.34 (excess peak +2.65 [2.04, 3.51] at 6,400); direction
   slope .034-.054 vs .036 (excess peak +.0156 [.0058, .0280] at 6,400): two of the four coupling rows lose more than
   Bayes while coupling develops, G and the excess angle do not. Same cell: every family loses about Bayes's amount or
   less (held per variable .075 at 4,400 vs .086; end-mass .074 vs .076; G .001-.057 vs .065; split gradient within
   +-.018), except the end-mass slope late (excess peak +.026 [.012, .050] at 13,600). Random cell: the losses of
   answers, holding and coupling grow through training; at the end the answers (.348 vs .344) and held per variable
   (.304 vs .338; excess -.033 [-.074, +.004]) are at Bayes's level, coupling below it (G .224 vs .280, excess -.0564
   [-.0701, -.0445]; jumps >= 4 cells 13.6 vs 19.9 deg, excess -6.35 [-7.40, -5.21]); the split gradient loses more than
   Bayes's none from early on (.01-.06 from 340, no trend; Bayes -.001; excess resolved at 1,120-7,600 and from 16,000,
   +.039 [+.014, +.066] at the end). Need onsets (the network's dependence above half of Bayes's under the same cut, for
   good) depend on the source and are bounded by each row's own level (a row cannot lose what it has not acquired), so
   they cannot date a need that precedes the behaviour. Start: end-mass slope 1,162 [1,014-10,006] (99%; with its c10
   1,166), held per variable 1,164 [1,073-never] (86%; with its c10 1,547), excess angle 1,720 [1,286-3,776] (with its
   c10 1,032, before its c50 6,248), G 2,420 [2,026-5,038] (after its c10 1,046 [622-1,361], with its c50 5,656), angle
   of jumps >= 4 cells 627 [534-1,710] (with its c10 740), direction slope 2,685 [2,132-3,990] (with its c10 3,255).
   Same cell: 2,718 [1,913-3,122] / 2,675 [2,242-never] (65%) / 4,446 / 14,693 / 2,497 (91%) / 4,653. Random cell
   (engine seed): 4,643 / 3,580 / 9,486 / 8,050 / 6,693 / 7,531 (over the 4 + 4 seeds of RESULTS 2: 4,220-5,226 /
   2,870-5,019 / 9,086-10,275 / 7,293-9,601 / 6,693-9,106 / 5,737-9,501). With the donors the onsets fall after the c10s
   (the same-cell direction slope's with its c10) and with or after the c50s, except the same-cell end-mass slope
   (2,718, before its c50 3,936); only the start cut puts them near the c10s, and that cut also carries the restart
   (RESULTS 3).
 7 ESN (null): its answers lose at most .009 (start), .005 (random), |.003| (same cell) to a 5-step cut, and at most
   .038 / .119 at w = 2; no transient. It holds and couples nothing to lose.
 8 Memory at the most entangled steps (td_tt_memory_events deferred its date 'q5 beats its control from 709-729' to
   these cuts; that date is 388-424 since 11-10, when the FR in its control's matching key became the forward KL(Bayes
   || naive), and 901-922 under its keys without FR). What the donor cuts can say: they have no 1-step window (that
   date uses the 1-step start cut), no source removes the content while keeping the clock (RESULTS 3), and their q5
   contrast is step-matched only. With those limits: the q5 contrast turns positive for good at 723 [624-754] (w = 2)
   and 754 [676-824] (w = 5) under the start state, as memory_events' 1-step stored cut with the symmetric FR in its
   control's key (709-729; forward FR 388-424, no FR 901-922), and under a random cell at 586 [462-653] / 542 [441-723]
   (all four before the accuracy c50 1,394 [1,285-1,489]): whenever the old content is removed or replaced by another
   cell's; under the same cell's other trajectory (content kept, trajectory changed) it turns positive only at 1,620
   [1,098-1,949] (w = 2) / 1,632 [1,015-2,005] (w = 5) and stays near 0 through 2,080 (+.0001 [-.0007, +.0009] at
   1,680), growing to Bayes's level by the end (+.0069 [.0064, .0073] vs Bayes +.0073 at w = 2). So: having the cell's
   content in the state helps the most entangled steps' updates more than the others' from about 550-750; what is
   specific to the episode's own trajectory helps them more only from about 1,600, with the accuracy c50 (1,394
   [1,285-1,489]; paired P(c50 earlier) .82 / .77). The donor cuts do not give memory_events' matched date.

FAILED, NOT THE MAIN ROUTE, NOT SEPARATED AND NULL
  (a) NOT THE MAIN ROUTE (was FAILED after the first check): non-Bayesian re-evaluation from the episode's own raw
      history (the hypothesis tested). Swaps that keep the cell's content from another history (the network's read-out,
      Bayes's counts, or the cell) cost at most .016 against the restart's .150 (RESULTS 3 (i)); the belief-matched
      swap, matched only on the read-out, is among them, so the old content acts mostly through the belief the network
      reads out. Not entirely: the belief-matched swap exceeds an exact observer's matched null by +.007 to +.012 at
      960-3,000 (resolved; possibly partly matching sensitivity: the residual is smallest, and unresolved, in the
      tightest quartile), and reshuffling the probe's own old observations (Bayes's posterior unchanged) costs +.004 to
      +.007 (resolved at 1,680, 4,400 and 6,400). At 400-1,360 the start-cut loss does concentrate on the network's own
      revision probes after margin matching (+.208 / +.135 / +.041), not at 1,680-6,400 (+.006 [-.018, +.032] at 1,680),
      and again from 11,200 (+.045 at the end). It is not concentrated on Bayes's revisions: its excess over Bayes is
      not larger on Bayes's revision steps at any checkpoint, and through about 11k training episodes it is not
      concentrated on the episodes Bayes revises (RESULTS 5). (The same-cell donor's zero cost does not discriminate:
      re-evaluating another trajectory of the same cell would also cost little.)
  (b) NOT SEPARATED (was 'FAILED: the excess long-memory reliance is real with donor states'): the start-state excess as
      missing content vs the restart regime. The donors cannot show that the excess reliance is absent (the same-cell
      state supplies the very content the network would rely on). What the controls show (RESULTS 3 (ii)): three
      mid-episode states without the probe's cell content (flat-belief, a new random cell at each step, the
      environment's mean) cost as much as the restart at 960-2,080 (no restart cost they detect there); at 480, 720 and
      880 the restart costs .012-.015 more (intervals up to +.027; resolved against the new-donor random cells at all
      three and the mean state at 880, not against the flat-belief state): if these states' own costs are not negative,
      a lower bound on a restart penalty (equal to it only if they are 0), not an upper bound; one consistent wrong
      cell's state costs more than the restart from 560 on. The uncut network builds its gain over naive more slowly
      than Bayes (RESULTS 3 (iii)). But no available state is content-free for an exact observer (Bayes loses .29-.32
      with the flat-belief histories), so missing content and restart regime are bracketed, not separated.
  (c) FAILED: the excess confined to entangled steps (memory used where Bayes needs it): it is present in every
      entanglement fifth at every checkpoint, about equal across fifths only at 1,680-2,080 and from 11,200, larger in
      the most entangled fifth at 400-1,200 and 3,000-6,400 (RESULTS 5).
  (d) FAILED: echo / self-reinforcement of the belief: the step's update re-adds b c_(t-1) with b .014-.033 across
      fifths at 1,680 vs Bayes's .014-.028 (ESN .018-.044 over its checkpoints from 400 and the fifths; .018-.052 with
      10-160): Bayes-sized. (The network's update keeps a large gain on naive's increment at the most entangled steps,
      a = .43 vs Bayes .09 at 1,680, .21 at the end.)
  (e) NOT CONFIRMED (secondary): the evidence for the true value against all values (e_t(r), centred) is lost more at
      Bayes's settled than revision steps within each step (+.029 to +.039 nats per var-step at 400-3,000, ~0 from
      7,600; Bayes -.014); in the answer-relevant competitor measure the excess is flat over step types at
      1,680-6,400 and larger on settled than revision steps only at 960-1,200 and from 11,200 (RESULTS 5): where it is
      flat, the settled-step surplus is evidence against far values that never compete.
  NULL: G and the excess angle show no start-state excess; held per variable at one-variable splits shows none; the
      ESN's dependence is at most .009 (start) and .005 (random cell) at w = 5, with no transient.

HOW THE NETWORK GETS THE ANSWERS BEFORE BAYES'S DYNAMICS (what these interventions show; RESULTS 1-3)
  Through the answers phase the network's gain over naive needs the cell content of its state from more than 5 steps
  back: restarted from (h0, c0) it keeps +.043 of its +.193 gain at 1,680, where Bayes's 5-step window keeps +.276 of
  +.307. That content acts mostly through the belief the network reads out: states that carry it from another history
  (the other trajectory of the same cell, the average of the cell's two states, the probe's own observations with every
  bit's order shuffled, another latent pair's state with the same read-out) cost at most .016, against the restart's
  .150. Not entirely: the belief-matched state costs +.007 to +.012 more than an exact observer's matched swap at
  960-3,000, possibly partly from matching sensitivity. Three mid-episode states without that content (the flattest
  read-outs of other cells, a new random cell at each step, the environment's mean state) cost as much as the restart at
  960-2,080, and some cost .012-.015 less at 480, 720 and 880. So these controls detect a small restart cost at those
  three checkpoints and none at 960-2,080, but they cannot exclude a penalty that equals their own costs ((b)); one
  consistent wrong cell's state costs more. What the network lacks is a fast build-up of that content: uncut, it reaches
  Bayes's step-5 accuracy only at step 13.6 (880), 7.5 (1,680) and 5.1 (end). As the build-up speeds up the restart
  costs less (from about 2,300-2,600 less than any content-free mid-episode state, as for Bayes), the excess moves to
  holding while holding develops (peak 5-6k; there too the content-free states cost about as much as the restart) and
  fades as holding and coupling complete (answers: half its peak at 6,845). By the end the remaining memory use
  concentrates where Bayes's does by episode type. By Bayes's step type the network's loss is larger on revision than on
  other steps by about Bayes's margin (from 1,680 on), but larger on revision than on settled steps only at 6,400
  (resolved), while Bayes's is; its excess is not larger on revision steps at any checkpoint. One other trajectory per
  cell (and the average of the two) was tested; the belief match is to the network's two read-out marginals, not to its
  whole state.

CONFOUNDS AND CAVEATS
  A donor carries evidence of its own: wrong (random cell), extra (same cell) or another history's (belief-matched,
  flat-belief); Bayes under the identical substitution is the matched reference (for the belief-matched swap: the
  matched null), and levels are drawn with naive's. A wrong cell's state sits at the ceiling for both observers, so it
  shows that old content is used and can mislead, not how much of the gain rests on it; the network's dependence there
  tracks its own gain over naive. A random-cell splice is a history that never occurs in training (the latent pair
  changes mid-episode): the network extrapolates where Bayes's response is principled, so the two dependences are not
  comparable as reliance. The random-cell magnitudes, and the dates timed on them, depend on the donor scheme and seed
  (RESULTS 2). The start state removes the content and resets the state's regime (its clock and the transient from
  (h0, c0)); the mid-episode controls keep the clock, but none removes content without adding some: for Bayes the
  flat-belief donors' histories are another cell's evidence, and the environment-mean and averaged states leave the
  state manifold (the mean state acts as wrong content late). The new-donor random cells carry wrong content that
  changes from step to step, which partly cancels in the cumulative read-out, so their equality with the restart at
  960-2,080 bounds a restart penalty rather than excluding it. The belief-matched donors match the network's read-out
  (its two marginals) to a median total variation of .04-.08; the state may carry more than the read-out (late, the
  flattest read-outs still mislead), so the residual over the matched null may mix matching sensitivity (the network's
  loss grows with its matching distance; in the tightest quartile the residual, +.007 at 1,680, is not resolved) and
  content beyond the read-out. The matched null matches Bayes on its two marginals from the same pools at about the same
  precision (median TV .071; the network's matches are tighter before 1,360: .043 at 400), and marginals are all the
  network's match uses (an independent check matched Bayes on its whole joint posterior in one pool: .0026, as on its
  marginals there). The shuffled histories keep the per-bit counts (Bayes's sufficient statistic), so they test the
  order and co-occurrence of old observations only. The same-cell result cannot separate a loss of trajectory-specific
  information from a gain of the donor's extra evidence; only the net effect (at most 0, vs Bayes's +.013) is
  established, and only one other trajectory per cell (plus the average of the two) was tested. The cumulative read-out
  keeps the network's own increments up to t = w and the window's own inputs: only the state's memory is tested. The
  probe held rows rest on 720 / 365 var-steps (one-variable ends noisy); the timeline's held rows are on the forward set
  (not cut here). Nats (margins, competitor's gain) depend on each observer's calibration; the accuracy rows do not. One
  seed and one probe world; intervals resample environments, not network seeds. 39 log-spaced + 7 checkpoints (the
  timeline uses 114; the beyond-belief controls 16): crossings are limited by that spacing. The 'where' groupings are
  Bayes-defined (plus the network-defined revision probes); grouping an intervention's effect is descriptive, not a
  separate causal test. Not done: the controls on the ESN, a 1-step donor window, the beyond-belief controls at the
  other 30 checkpoints, pools larger than 20 episodes per cell, a null matched to the network's precision checkpoint by
  checkpoint.

CHECKS (checks.json; controls/checks.json)
  Windowed Bayes from the start state = probes.npz right_window1 / 2 / 5 (0 of 20,000 differ each); Bayes under the
  donor cuts brute-forced from explicit histories (9 probes x 9 cuts x 30 steps): max |d log q| 2.4e-13; own float64
  Bayes vs log_jb 7.6e-6, its rows vs td_tt_intervene.refs 8.8e-5 (fp32 storage of log_jb); single-step swap of a
  condition with itself 1.5e-13; update change at steps <= w <= 1e-11 (exactly 0 from the float32 cache); Sigma_t
  evidence = the centred log belief at T (max over checkpoints 5.9e-5, fp32); the competitor decomposition sums to
  L_int - L_cut exactly (4.5e-13 on Bayes) and L_cut = the cut's margin (0); every 'where' forward gives the same
  answers as the harvest (all 18). The cut from the start state reproduces trunc_right at all 46 + 10 checkpoints.
  Controls: the control forward's start / same-cell / random cuts give the harvest's answers at T at all 46
  checkpoints (0 of 20,000 differ, each), its intact run 1 of 920,000 (a near-tie); Bayes under the shuffled
  histories = Bayes intact (max |d log q| 3.3e-13); the joint log-likelihood from per-bit counts = the probes'
  cumulative one (2.3e-13); the generated pools follow the environments' tables (per cell and bit, 600 draws, 75,000
  cells x bits: z mean -.004, sd .999, |z| > 3 for .29% (normal .27%), max 4.1; the probes' own 60 draws: sd .995,
  .34%, max 5.3). Re-derived at 1,680 (pool seed 20261241): the cached belief-matched and flat-belief answers are
  reproduced (0 and 0 of 20,000 differ) and the matching distances exactly; no belief-matched donor is of the probe's
  cell or history (0 of 250,000); the flat-belief fallback gave 36 probes their own first observation and 1 its own
  cell at the first window start (t = 6), a no-op or near no-op there; Bayes's reference rebuilt from the explicit
  donor histories (joint posterior from scratch, 9 probes x 25 window starts x 2 sources) equals the counts-based one
  to 3.0e-13. Matched null: its donors equal those of the second check's independent script (bayes_bm.py: identical
  per-bit counts and distances in all 3 pools), its losses are the check's (.0026 / .0049 / .0053), and the network
  minus null at all 16 checkpoints and the distance quartiles reproduce the check's (bm_null_curve.py, tv_strat.py);
  the second check's own LSTM and float64 Bayes reproduced all 22 control conditions at 6 checkpoints (0 of 20,000
  answers differ each).

COST (GPU 0 shared with other sessions' jobs at ~100% utilisation): harvest 194 s per checkpoint with 3 workers
  (forward 117 s for 3 sources x 3 windows; <= 434 MB per process); 56 checkpoints in ~62 min; 'where' pass 18
  checkpoints in ~10 min; Bayes caches ~4 min. Controls: 2 workers, 70 min wall (16 full checkpoints at 162-441 s of
  forward each, 162 s with no second worker; 30 light ones at 26-131 s), 2.1 h of forward time in all, <= 1.84 GB per
  process; Bayes controls 42 s; matched null 263 s (CPU, 2 threads); ctl_checks 53 s; numbers ~30 s; figures ~1 min.

Run from the repository root (CUDA_VISIBLE_DEVICES=0; 2 threads; check the load average first):
    CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
        /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_memcuts.py harvest 3
    (then 'where 3' for the competitor pass; 'bayes_ctl', 'controls 2' and 'bm_null' for the controls; 'checks',
    'ctl_checks', 'numbers', 'figure'; 'one <run> <raw episode>' / 'ctl_one <run> <raw episode>' redo one checkpoint;
    workers log to the session scratchpad tt_tmp/memcuts and tt_tmp/fix_memory_cuts)"""
import os
import sys
import json
import time
import subprocess
import warnings
import numpy as np
from scipy.special import logsumexp

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_intervene as ti                                                                  # noqa: E402

TD = ti.TD
OUT = os.path.join(TD, "through_training", "memcuts")
TMP = ("/tmp/claude-1002/-home-johns-Desktop-CognitiveGridworld/7af6b091-54ff-45a0-b4dd-ef0a3bc9c778/scratchpad/tt_tmp/"
       "memcuts")
TMP_FIX = ("/tmp/claude-1002/-home-johns-Desktop-CognitiveGridworld/7af6b091-54ff-45a0-b4dd-ef0a3bc9c778/scratchpad/tt_tmp/"
           "fix_memory_cuts")                                 # the controls' worker logs and claims
PY = os.environ.get("CGW_PY", "/home/johns/anaconda3/envs/eureka/bin/python")
T, R, NV = 30, 10, 2
NE, PER_ENV = gap.N_ENV, gap.PER_ENV
NB = gap.N_BOOT
WS = (2, 5, 10)
SOURCES = ti.SOURCES                                      # 'start', 'same_cell', 'random'
CUTS = [(w, s) for w in WS for s in SOURCES]
CONDS = ("intact",) + tuple(f"w{w}_{s}" for w, s in CUTS)
ROWS = ti.FULL                                            # 18 cheap rows + 6 splitting / fold rows
NF, NG = 5, 3                                             # entanglement fifths; Bayes step groups (revision, settled, other)
GROUPS = ("revision step", "settled step", "other step")
GRID = {"trained": 46, "echo": 10}                        # log-spaced targets -> 39 trained / 10 ESN checkpoints
EXTRA = {"trained": (1100, 1200, 1400, 1700, 2100, 2400, 2900), "echo": ()}  # training 880 ... 2,320 (answers phase)
BUDGET = 4000                                             # GPU rows per chunk x (1 + 17 branch rows): one environment
SRC_NAME = {"start": "start state", "same_cell": "same-cell donor", "random": "random-cell donor"}
COL = {"trained": "#1f77b4", "echo": "#ff7f0e"}
_C = {}


# ═══════════════════════════════════════════════════════════════════
# Checkpoints
# ═══════════════════════════════════════════════════════════════════

def selection(run):
    """raw episodes analysed: nearest stored checkpoints (in log training episodes) to GRID[run] log-spaced targets
    from 10 to the end, plus EXTRA[run]"""
    eps = np.array(ti.checkpoints(run))
    x = gap.train_x(eps)
    pick = np.unique([np.argmin(np.abs(np.log(x) - np.log(t))) for t in np.geomspace(x[0], x[-1], GRID[run])])
    out = set(int(e) for e in eps[pick]) | set(EXTRA[run])
    assert set(EXTRA[run]) <= set(int(e) for e in eps), "EXTRA must be stored checkpoints"
    return sorted(out)


def ck_path(run, ep):
    return os.path.join(OUT, run, f"ck_{int(ep):05d}.npz")


# ═══════════════════════════════════════════════════════════════════
# Bayes under the same cuts (exact, float64) and the Bayes-side labels
# ═══════════════════════════════════════════════════════════════════

def _probes_full():
    if "P" not in _C:
        z = np.load(os.path.join(ti.AN, "probes.npz"))
        _C["P"] = {k: z[k] for k in ("obs", "env", "r", "goal", "L", "ent", "log_nb", "inc_N", "right_bayes",
                                     "right_naive", "right_window1", "right_window2", "right_window5")}
    return _C["P"]


def _lam():
    """Lam (n, T + 1, R, R): cumulative per-step joint log-likelihoods of the probes (Lam[:, t] = steps 1..t, float64;
    td_analysis.bayes_side)"""
    if "Lam" not in _C:
        P = _probes_full()
        L = P["L"][P["env"]]
        o = P["obs"].astype(np.float64)
        ll = np.einsum("nti,nijk->ntjk", o, np.log(L)) + np.einsum("nti,nijk->ntjk", 1 - o, np.log1p(-L))
        _C["Lam"] = np.concatenate([np.zeros((len(o), 1, R, R)), np.cumsum(ll, 1)], 1)
    return _C["Lam"]


def _marg(X):
    """log marginals (unnormalised) of joint log weights X (..., R, R): (..., 2, R) = [variable 0, variable 1]"""
    return np.stack([logsumexp(X, -1), logsumexp(X, -2)], -2)


def donors():
    """the engine's donors on the probes: {'start': None, 'same_cell': row ^ 1, 'random': seed 20261201}"""
    return {"start": None, "same_cell": ti._donors_cached("same_cell"), "random": ti._donors_cached("random")}


def bayes_logq(w=None, src="start"):
    """Bayes's marginal log beliefs (n, T, 2, R) float64, intact (w None) or under the network's cut: for t > w the
    step-t increment is Bayes's marginal change from adding observation t to the history (donor's observations 1..t-w,
    own t-w+1..t-1) (source 'start': no donor observations = td_analysis's windowed Bayes); t <= w: Bayes's own
    increment; beliefs = the cumulative sum, normalised"""
    Lam = _lam()
    n = len(Lam)
    if w is None:
        m = _marg(Lam[:, 1:])
        return m - logsumexp(m, -1, keepdims=True)
    d = donors()[src]
    inc = np.zeros((n, T, NV, R))
    for t in range(1, T + 1):
        if t <= w:
            inc[:, t - 1] = _marg(Lam[:, t]) - _marg(Lam[:, t - 1])
        else:
            a = t - w
            base = 0. if d is None else Lam[d, a]
            inc[:, t - 1] = _marg(base + Lam[:, t] - Lam[:, a]) - _marg(base + Lam[:, t - 1] - Lam[:, a])
    cum = np.cumsum(inc, 1)
    return cum - logsumexp(cum, -1, keepdims=True)


def side():
    """Bayes-side labels, fixed through training: lp (n, T + 1, 2, R) Bayes's log marginals (index 0 uniform); fifth
    (n, T, 2) the variable's own entanglement fifth within the step (over both variables: 4,000 var-steps per fifth and
    step; probes.npz ent); grp (n, T, 2) Bayes's step group: 0 revision (Bayes's answer for the variable changes at t), 1
    settled (Bayes's answer at t - 1 .. T is its answer at T), 2 other (incl. step 1); probe families of the goal
    variable (td_tt_memory_events' definitions): fam (n,) 0 Bayes revises (q > .6 on a value that is not its answer at T
    at some step 8-24), 1 Bayes settled (same answer at every step 8-30), 2 neither; naive_far (n,) naive's goal answer
    at T 2+ values off; counts per (environment, step, group)"""
    if "side" in _C:
        return _C["side"]
    P = _probes_full()
    n = len(P["env"])
    lq = bayes_logq()
    lp = np.concatenate([np.full((n, 1, NV, R), -np.log(R)), lq], 1)
    ent = P["ent"]
    fifth = np.zeros((n, T, NV), np.int8)
    for t in range(T):
        rk = np.argsort(np.argsort(ent[:, t].ravel(), kind="stable"), kind="stable")
        fifth[:, t] = (rk * NF // rk.size).reshape(n, NV)
    a = lq.argmax(-1)                                                                # (n, T, 2) Bayes's answers
    fin = a[:, -1:]
    grp = np.full((n, T, NV), 2, np.int8)
    grp[:, 1:][a[:, 1:] != a[:, :-1]] = 0
    same_after = np.flip(np.cumprod(np.flip(a == fin, 1), 1), 1).astype(bool)        # a_s = a_T for all s >= t
    st = np.zeros_like(grp, dtype=bool)
    st[:, 1:] = same_after[:, :-1]                                                   # from t - 1 on
    grp[st & (grp != 0)] = 1
    ar, g = np.arange(n), P["goal"]
    qg = np.exp(lq[ar, :, g])                                                        # (n, T, R) goal marginal
    aT = a[ar, -1, g]
    hi = (qg[:, 7:24] > .6) & (np.arange(R)[None, None] != aT[:, None, None])
    fam = np.full(n, 2, np.int8)
    fam[hi.any((1, 2))] = 0
    fam[(a[ar, 7:, g] == aT[:, None]).all(1) & (fam != 0)] = 1
    naive_far = np.abs(P["log_nb"][ar, -1, g].argmax(-1) - P["r"][ar, g]) >= 2
    env = P["env"]
    N = P["inc_N"].astype(np.float64)                                               # naive's centred increment
    S = dict(lp=lp, fifth=fifth, grp=grp, fam=fam, naive_far=naive_far, env=env, r=P["r"], goal=g, N=N,
             cnt_f=_agg(np.ones((n, T, NV)), env, fifth, NF), cnt_g=_agg(np.ones((n, T, NV)), env, grp, NG),
             nn_f=_agg((N * N).sum(-1), env, fifth, NF))
    _C["side"] = S
    return S


def _agg(v, env, lab, G):
    """sums of v (n, T, 2) per (environment, step, label): (NE, T, G)"""
    n = len(env)
    idx = (env[:, None, None] * T + np.arange(T)[None, :, None]) * G + lab
    return np.bincount(idx.ravel(), np.broadcast_to(v, (n, T, NV)).ravel().astype(np.float64),
                       NE * T * G).reshape(NE, T, G)


KEEP64 = ("value", "boot")                                  # stored float64 (td_tt_gap); per-step sums as float32


def _save(path, out):
    """compressed npz; float64 arrays other than the score values / resamples stored as float32 (per-environment sums:
    relative rounding ~1e-7); _load casts them back to float64"""
    o = {k: (v.astype(np.float32) if (isinstance(v, np.ndarray) and v.ndim and v.dtype == np.float64 and
                                      k not in KEEP64) else v) for k, v in out.items()}
    np.savez_compressed(path, **o)


def _load(path):
    with np.load(path) as z:
        return {k: (z[k].astype(np.float64) if z[k].dtype == np.float32 else z[k]) for k in z.files}


# ═══════════════════════════════════════════════════════════════════
# Statistics of one belief array (network or Bayes, intact or cut)
# ═══════════════════════════════════════════════════════════════════

def cond_stats(lq, right=None):
    """everything stored per condition: the FULL score rows (values and the 1,000 shared resamples), per-step
    aggregates and per-probe answers. lq (n, T, 2, R) log beliefs; right (n, T, 2) correctness (default argmax == truth)
    Per-step quantities per (probe, step, variable), summed per (environment, step, entanglement fifth / Bayes step
    group): ev = the step's evidence for the true value, the centred increment e_t(r) with e_t = c_t - c_(t-1), c_t = the
    centred log belief (Sigma_t ev = c_T(r) exactly); err = the local update error, Jeffreys/2 between Bayes's belief at
    t - 1 moved by e_t and Bayes's belief at t (td_entanglement_memcut; 0 for Bayes intact); sz = |e_t|^2 (the step's
    update size, squared); dn = e_t . N_t with N_t naive's centred increment (gain in naive's direction = Sigma dn / Sigma
    N.N, side()['nn_f']); eb = e_t . c_(t-1), bb = c_(t-1) . c_(t-1), nb = N_t . c_(t-1) (with dn and nn: the
    regression e_t ~ a N_t + b c_(t-1) per group; b = the share of the belief before t that the step re-adds beyond the
    new evidence; naive: a = 1, b = 0). acc_goal / acc_both (NE, T): per-step answers right (goal; both variables),
    summed per environment"""
    S = side()
    P = _probes_full()
    lq = np.asarray(lq, np.float64)
    lq = lq - logsumexp(lq, -1, keepdims=True)
    r, env, n = P["r"], P["env"], len(P["env"])
    if right is None:
        right = lq.argmax(-1) == r[:, None, :]
    right = np.asarray(right, bool)
    rows = ti.score(lq, "full", right=right)
    bs = ti.resample({k: rows[k] for k in ROWS})
    c = lq - lq.mean(-1, keepdims=True)
    e = np.diff(c, axis=1, prepend=np.zeros_like(c[:, :1]))
    ev = np.take_along_axis(e, r[:, None, :, None], -1)[..., 0]                    # (n, T, 2)
    cp = np.concatenate([np.zeros_like(c[:, :1]), c[:, :-1]], 1)                     # c_(t-1), the belief before t
    la = S["lp"][:, :-1] + e
    la = la - logsumexp(la, -1, keepdims=True)
    lb = S["lp"][:, 1:]
    err = .5 * ((np.exp(la) - np.exp(lb)) * (la - lb)).sum(-1)                     # (n, T, 2)
    ar = np.arange(n)
    out = dict(value=np.array([rows[k]["value"] for k in ROWS]), boot=np.stack([bs[k] for k in ROWS]),
               ev_f=_agg(ev, env, S["fifth"], NF), err_f=_agg(err, env, S["fifth"], NF),
               ev_g=_agg(ev, env, S["grp"], NG), err_g=_agg(err, env, S["grp"], NG),
               sz_f=_agg((e * e).sum(-1), env, S["fifth"], NF), dn_f=_agg((e * S["N"]).sum(-1), env, S["fifth"], NF),
               sz_g=_agg((e * e).sum(-1), env, S["grp"], NG),
               eb_f=_agg((e * cp).sum(-1), env, S["fifth"], NF), bb_f=_agg((cp * cp).sum(-1), env, S["fifth"], NF),
               nb_f=_agg((S["N"] * cp).sum(-1), env, S["fifth"], NF),
               acc_goal=np.bincount(np.repeat(env, T) * T + np.tile(np.arange(T), n),
                                    right[ar, :, S["goal"]].ravel().astype(np.float64), NE * T).reshape(NE, T),
               acc_both=np.bincount(np.repeat(env, T) * T + np.tile(np.arange(T), n),
                                    right.all(-1).ravel().astype(np.float64), NE * T).reshape(NE, T),
               right_T=right[:, -1], cT=c[ar[:, None], -1, np.arange(NV)[None], r].astype(np.float32),
               mT=_margin(c[:, -1], r).astype(np.float32))
    return out


def _margin(x, r):
    """log-odds margin of the true value over its strongest competitor: x[..., r] - max_(j != r) x[..., j];
    x (n, ..., 2, R), r (n, 2)"""
    rr = r.reshape((len(r),) + (1,) * (x.ndim - 3) + (NV, 1))
    xr = np.take_along_axis(x, rr, -1)[..., 0]
    xm = np.where(np.arange(R) == rr, -np.inf, x)
    return xr - xm.max(-1)


def pair_stats(lq_int, lq_cut, w):
    """single-step swaps: the intact beliefs with ONLY step t's increment taken from the cut (t > w), and the answer at
    T and its margin recomputed. Per (probe, step, variable): flip = right_T(intact) - right_T(swap at t) (1 = this
    step's cut alone loses the answer), dm = margin_T(intact) - margin_T(swap at t) (nats; margin = log-odds of the true
    value over its strongest competitor); summed per (environment, step, entanglement fifth / Bayes step group), zero
    for t <= w. Answers from the argmax of the float64 beliefs (both arms the same). ci = e_cut . e_intact (with sz:
    |e_cut - e_intact|^2 = sz_cut + sz_intact - 2 ci, the squared change of the step's update)"""
    S = side()
    r, env = S["r"], S["env"]
    out = {}
    cs = []
    for lq in (lq_int, lq_cut):
        lq = np.asarray(lq, np.float64)
        cs.append(lq - lq.mean(-1, keepdims=True))
    ei = np.diff(cs[0], axis=1, prepend=np.zeros_like(cs[0][:, :1]))
    ec = np.diff(cs[1], axis=1, prepend=np.zeros_like(cs[1][:, :1]))
    ci = (ec * ei).sum(-1)                                                            # e_cut . e_intact
    out["ci_f"], out["ci_g"] = _agg(ci, env, S["fifth"], NF), _agg(ci, env, S["grp"], NG)
    cT = cs[0][:, -1]
    sw = cT[:, None] - ei[:, w:] + ec[:, w:]                                          # (n, T - w, 2, R)
    right0 = (cT.argmax(-1) == r)[:, None]
    flip = np.zeros((len(r), T, NV))
    dm = np.zeros((len(r), T, NV))
    flip[:, w:] = right0.astype(np.float64) - (sw.argmax(-1) == r[:, None])
    dm[:, w:] = _margin(cT, r)[:, None] - _margin(sw, r)
    for k, v in (("flip", flip), ("dm", dm)):
        out[f"{k}_f"] = _agg(v, env, S["fifth"], NF)
        out[f"{k}_g"] = _agg(v, env, S["grp"], NG)
    return out


PAIR = ("flip_f", "flip_g", "dm_f", "dm_g", "ci_f", "ci_g")
STACK = ("value", "boot", "ev_f", "err_f", "ev_g", "err_g", "sz_f", "sz_g", "dn_f", "eb_f", "bb_f", "nb_f", "acc_goal",
         "acc_both", "right_T", "cT", "mT") + PAIR


def _stack(res):
    for c in CONDS:
        if "flip_f" not in res[c]:                                                     # intact: no swap
            res[c].update({k: np.zeros_like(res[CONDS[1]][k]) for k in PAIR})
    return {k: np.stack([res[c][k] for c in CONDS]) for k in STACK}


def bayes_cache(force=False):
    """-> OUT/bayes.npz: Bayes intact and under the 9 cuts (CONDS order), cond_stats of each, plus the side labels'
    counts and families"""
    path = os.path.join(OUT, "bayes.npz")
    if os.path.exists(path) and not force:
        return _load(path)
    t0 = time.time()
    S = side()
    lq0 = bayes_logq()
    res = {"intact": cond_stats(lq0)}
    for w, s in CUTS:
        lqc = bayes_logq(w, s)
        res[f"w{w}_{s}"] = cond_stats(lqc)
        res[f"w{w}_{s}"].update(pair_stats(lq0, lqc, w))
        print(f"  Bayes w{w} {s}: goal accuracy at T {res[f'w{w}_{s}']['value'][0]:.4f} ({time.time() - t0:.0f} s)",
              flush=True)
    out = _stack(res)
    out.update(conds=np.array(CONDS), rows=np.array(ROWS), cnt_f=S["cnt_f"], cnt_g=S["cnt_g"], nn_f=S["nn_f"],
               fam=S["fam"], naive_far=S["naive_far"])
    nres = cond_stats(_probes_full()["log_nb"])                                      # naive (memoryless reference)
    out.update({f"naive_{k}": nres[k] for k in STACK if k not in PAIR})
    os.makedirs(OUT, exist_ok=True)
    _save(path, out)
    print(f"saved {path} ({time.time() - t0:.0f} s)", flush=True)
    return out


# ═══════════════════════════════════════════════════════════════════
# The network: one checkpoint (GPU forward with the 9 cuts, then CPU statistics)
# ═══════════════════════════════════════════════════════════════════

def run_ck(run, ep, force=False):
    path = ck_path(run, ep)
    if os.path.exists(path) and not force:
        return
    import torch
    t0 = time.time()
    w = ti.load(run, ep)
    fo, peak = {"logq_cut": {}, "right_cut": {}}, 0.
    for s in SOURCES:                                  # one forward per source (3 windows each): <= ~1 GB of GPU
        f1 = ti.forward(w, cut=[(w_, s) for w_ in WS], budget=BUDGET)
        if "logq" not in fo:
            fo.update(logq=f1["logq"], right=f1["right"])
        else:                                          # the same chunking: the intact run is bit-identical
            assert np.array_equal(fo["logq"], f1["logq"])
        fo["logq_cut"].update(f1["logq_cut"])
        fo["right_cut"].update(f1["right_cut"])
        peak = max(peak, f1["info"]["peak_gpu_mb"])
        del f1
        torch.cuda.empty_cache()
    fo["info"] = dict(peak_gpu_mb=peak)
    tf = time.time() - t0
    res = {"intact": cond_stats(fo["logq"], fo["right"])}
    for w_, s in CUTS:
        nm = f"w{w_}_{s}"
        res[nm] = cond_stats(fo["logq_cut"][nm], fo["right_cut"][nm])
        res[nm].update(pair_stats(fo["logq"], fo["logq_cut"][nm], w_))
    out = _stack(res)
    zk = np.load(os.path.join(ti.AN, run, f"ck_{int(ep):05d}.npz"))
    tr = zk["trunc_right"]                                                            # [w1, w2, w5, intact]
    chk = dict(trunc_w2=int((fo["right_cut"]["w2_start"][:, -1] != tr[:, 1]).sum()),
               trunc_w5=int((fo["right_cut"]["w5_start"][:, -1] != tr[:, 2]).sum()),
               trunc_intact=int((fo["right"][:, -1] != tr[:, 3]).sum()),
               ev_sum_identity=float(max(np.abs(res[c]["ev_f"].sum((1, 2)) - np.bincount(
                   side()["env"], res[c]["cT"].astype(np.float64).sum(-1), NE)).max() for c in CONDS)))
    out.update(conds=np.array(CONDS), rows=np.array(ROWS), run=run, ep=int(ep), x=float(gap.train_x(ep)),
               fingerprint=ti.fingerprint(w), forward_s=tf, peak_gpu_mb=fo["info"]["peak_gpu_mb"],
               seconds=time.time() - t0, checks=json.dumps(chk))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    _save(tmp, out)
    os.replace(tmp, path)
    print(f"{time.strftime('%H:%M:%S')} {run} raw {ep} (training {gap.train_x(ep):,.0f}): forward {tf:.0f} s, total "
          f"{time.time() - t0:.0f} s, peak GPU {fo['info']['peak_gpu_mb']:.0f} MB; goal accuracy intact "
          f"{out['value'][0, 0]:.4f}, w5 start / same cell / random {out['value'][CONDS.index('w5_start'), 0]:.4f} / "
          f"{out['value'][CONDS.index('w5_same_cell'), 0]:.4f} / {out['value'][CONDS.index('w5_random'), 0]:.4f}; "
          f"checks {chk}", flush=True)


def jobs():
    return [(run, ep) for run in ("trained", "echo") for ep in selection(run)]


def worker(k, nw):
    """takes the next unclaimed job (claim = an exclusive lock file in TMP; a crashed worker's claim is cleared by the
    next harvest); k, nw only label the log"""
    os.makedirs(os.path.join(TMP, "claims"), exist_ok=True)
    for run, ep in jobs():
        if os.path.exists(ck_path(run, ep)):
            continue
        try:
            fd = os.open(os.path.join(TMP, "claims", f"{run}_{ep:05d}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, f"worker {k}/{nw} pid {os.getpid()}".encode())
        os.close(fd)
        run_ck(run, ep)


def harvest(nw=3):
    """launches nw workers (GPU 0, 2 threads each; logs in TMP) after the Bayes cache; returns at once"""
    bayes_cache()
    os.makedirs(os.path.join(TMP, "claims"), exist_ok=True)
    for f in os.listdir(os.path.join(TMP, "claims")):                     # no worker runs: stale claims
        os.remove(os.path.join(TMP, "claims", f))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MPLBACKEND="Agg",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    pids = []
    for k in range(nw):
        log = open(os.path.join(TMP, f"worker_{k}.log"), "a")
        p = subprocess.Popen(["timeout", "14400", PY, os.path.abspath(__file__), "worker", str(k), str(nw)],
                             stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT, start_new_session=True)
        pids.append(p.pid)
    print("workers", pids, "logs", TMP)
    with open(os.path.join(TMP, "pids.txt"), "w") as f:
        f.write(" ".join(map(str, pids)))


# ═══════════════════════════════════════════════════════════════════
# 'Where' pass: the cut's winning competitor, step by step (additive)
# ═══════════════════════════════════════════════════════════════════

WHERE_X = (400, 560, 800, 960, 1200, 1680, 2080, 3000, 4400, 6400, 11200, 19200, 39996)   # training episodes


def where_path(run, ep):
    return os.path.join(OUT, run, f"where_{int(ep):05d}.npz")


def competitor_stats(lq_int, lq_cut):
    """where the cut lets a competitor gain on the true value, step by step (exactly additive). Per probe and
    variable, j_c = the strongest competitor of the true value r in the CUT belief at T; L = x(r) - x(j_c), x = the
    centred cumulative log belief (L_cut = the cut's margin; L_int >= the intact margin). L_int - L_cut = Sigma_t d_t with
    d_t = [e_t(r) - e_t(j_c)]_intact - [e_t(r) - e_t(j_c)]_cut (the step's increment of the true value's log-odds against
    that competitor, lost by the cut; t <= w: 0). Sums of d_t per (environment, step, entanglement fifth / Bayes step
    group) over all variable-steps (dl_*) and over the variables whose answer at T the cut loses (right intact, wrong
    cut: dlx_*, with their counts nx_*)"""
    S = side()
    r, env = S["r"], S["env"]
    cs = []
    for lq in (lq_int, lq_cut):
        lq = np.asarray(lq, np.float64)
        cs.append(lq - lq.mean(-1, keepdims=True))
    ei = np.diff(cs[0], axis=1, prepend=np.zeros_like(cs[0][:, :1]))
    ec = np.diff(cs[1], axis=1, prepend=np.zeros_like(cs[1][:, :1]))
    xc = np.where(np.arange(R) == r[..., None], -np.inf, cs[1][:, -1])               # (n, 2, R) cut, truth masked
    jc = xc.argmax(-1)                                                               # (n, 2)

    def pair(e):
        return (np.take_along_axis(e, r[:, None, :, None], -1)[..., 0]
                - np.take_along_axis(e, jc[:, None, :, None], -1)[..., 0])          # (n, T, 2)
    d = pair(ei) - pair(ec)
    lost = ((cs[0][:, -1].argmax(-1) == r) & (cs[1][:, -1].argmax(-1) != r)).astype(np.float64)    # (n, 2)
    out = {}
    for k, lab, G in (("f", S["fifth"], NF), ("g", S["grp"], NG)):
        out[f"dl_{k}"] = _agg(d, env, lab, G)
        out[f"dlx_{k}"] = _agg(d * lost[:, None, :], env, lab, G)
        out[f"nx_{k}"] = _agg(np.broadcast_to(lost[:, None, :], d.shape), env, lab, G)
    out["lost"] = lost.sum(0)
    return out


WSTACK = ("dl_f", "dlx_f", "nx_f", "dl_g", "dlx_g", "nx_g", "lost")


def where_ck(run, ep, force=False):
    """the 'where' pass at one checkpoint: the three forwards again (bit-identical to run_ck's; checked against the
    stored per-probe answers) and competitor_stats for the 9 cuts -> OUT/<run>/where_EEEEE.npz"""
    path = where_path(run, ep)
    if os.path.exists(path) and not force:
        return
    import torch
    t0 = time.time()
    w = ti.load(run, ep)
    res, lq0, rt0 = {}, None, None
    for s_ in SOURCES:
        f1 = ti.forward(w, cut=[(w_, s_) for w_ in WS], budget=BUDGET)
        if lq0 is None:
            lq0, rt0 = f1["logq"], f1["right"]
        for w_ in WS:
            nm = f"w{w_}_{s_}"
            res[nm] = competitor_stats(lq0, f1["logq_cut"][nm])
            res[nm]["right_T"] = f1["right_cut"][nm][:, -1]
        del f1
        torch.cuda.empty_cache()
    z = _load(ck_path(run, ep))
    same = bool(np.array_equal(z["right_T"][0], rt0[:, -1]) and
                all(np.array_equal(z["right_T"][cix(c)], res[c]["right_T"]) for c in CONDS[1:]))
    out = {k: np.stack([res[c][k] for c in CONDS[1:]]) for k in WSTACK}
    out.update(conds=np.array(CONDS[1:]), ep=int(ep), x=float(gap.train_x(ep)), same_as_harvest=same,
               seconds=time.time() - t0)
    _save(path, out)
    print(f"{time.strftime('%H:%M:%S')} where {run} raw {ep} (training {gap.train_x(ep):,.0f}): {time.time() - t0:.0f} s; "
          f"answers identical to the harvest: {same}", flush=True)


def bayes_where(force=False):
    path = os.path.join(OUT, "bayes_where.npz")
    if os.path.exists(path) and not force:
        return _load(path)
    lq0 = bayes_logq()
    res = {f"w{w}_{s_}": competitor_stats(lq0, bayes_logq(w, s_)) for w, s_ in CUTS}
    out = {k: np.stack([res[c][k] for c in CONDS[1:]]) for k in WSTACK}
    out.update(conds=np.array(CONDS[1:]))
    _save(path, out)
    return out


def where_jobs():
    eps = np.array(ti.checkpoints("trained"))
    x = gap.train_x(eps)
    J = [("trained", int(eps[np.argmin(np.abs(x - xv))])) for xv in WHERE_X]
    ee = np.array(selection("echo"))
    J += [("echo", int(e)) for e in ee[[0, 4, 6, 8, 9]]]
    return J


def where_worker(k, nw):
    os.makedirs(os.path.join(TMP, "claims_where"), exist_ok=True)
    for run, ep in where_jobs():
        if os.path.exists(where_path(run, ep)):
            continue
        try:
            fd = os.open(os.path.join(TMP, "claims_where", f"{run}_{ep:05d}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, f"worker {k}/{nw} pid {os.getpid()}".encode())
        os.close(fd)
        where_ck(run, ep)


def where_harvest(nw=3):
    bayes_where()
    os.makedirs(os.path.join(TMP, "claims_where"), exist_ok=True)
    for f in os.listdir(os.path.join(TMP, "claims_where")):
        os.remove(os.path.join(TMP, "claims_where", f))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MPLBACKEND="Agg",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    pids = []
    for k in range(nw):
        log = open(os.path.join(TMP, f"where_{k}.log"), "a")
        p = subprocess.Popen(["timeout", "7200", PY, os.path.abspath(__file__), "where_worker", str(k), str(nw)],
                             stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT, start_new_session=True)
        pids.append(p.pid)
    print("where workers", pids)
    with open(os.path.join(TMP, "pids_where.txt"), "w") as f:
        f.write(" ".join(map(str, pids)))


# ═══════════════════════════════════════════════════════════════════
# Controls (added 11-02 after the adversarial check): donor-scheme seeds, averaged states, and the beyond-belief tests
# (shuffled own history, belief-matched and flat-belief donors from generated same-environment episodes); w = 5
# ═══════════════════════════════════════════════════════════════════

CTL = os.path.join(OUT, "controls")
CW = 5                                                    # every control cuts to the last 5 steps
CA = T - CW                                               # window starts a = 1..25 (t = a + 5)
SEEDS_FIX = (20261211, 20261212, 20261213)                # fixed random-cell donor per probe (+ the engine's 20261201)
SEEDS_STEP = (20261221, 20261222, 20261223, 20261224)     # a fresh random-cell donor at every window start
SEEDS_SHUF = (20261231, 20261232, 20261233)               # per-bit shuffles of the probe's own observations 1..a
SEEDS_POOL = (20261241, 20261242, 20261243)               # generated same-environment pools (belief-matched / flat)
PPC = 20                                                  # pool trajectories per cell and environment (2,000)
K_FLAT = 10                                               # flat donor: one of the 10 most uniform network beliefs
CTL_FULL_X = (400, 560, 800, 880, 960, 1120, 1360, 1680, 2080, 2600, 3000, 4400, 6400, 11200, 19200, 39996)
NO = 5


def ctl_path(run, ep):
    return os.path.join(CTL, run, f"ck_{int(ep):05d}.npz")


def ctl_full_eps():
    """raw episodes of the Trained RNN checkpoints that get the beyond-belief controls (nearest to CTL_FULL_X)"""
    eps = np.array(selection("trained"))
    x = gap.train_x(eps)
    return sorted({int(eps[np.argmin(np.abs(np.log(x) - np.log(v)))]) for v in CTL_FULL_X})


def ctl_conds(ep):
    """the control conditions at a raw checkpoint: the engine's three sources (validation against the harvest), the
    random-cell donor under 3 more seeds (fixed per probe) and 4 seeds of a fresh donor per window start, the averaged
    states; at ctl_full_eps() also the shuffled own histories and the belief-matched / flat-belief donors"""
    c = ["start", "same_cell", "random"] + [f"rand_fix_{s}" for s in SEEDS_FIX] + \
        [f"rand_step_{s}" for s in SEEDS_STEP] + ["mean_pair", "mean_env"]
    if int(ep) in ctl_full_eps():
        c += [f"shuffle_{s}" for s in SEEDS_SHUF] + [f"bm_net_{s}" for s in SEEDS_POOL] + \
             [f"flat_{s}" for s in SEEDS_POOL]
    return c


def step_donors(seed):
    """(n, T + 1) donor probes: for every probe and window start a an independent random probe of the same
    environment with another cell (column a; column 0 unused)"""
    key = ("step_donors", seed)
    if key not in _C:
        P = _probes_full()
        env, cell = P["env"], ti.probes()["cell"]
        rng = np.random.default_rng(seed)
        out = np.zeros((len(env), T + 1), np.int64)
        for e in range(NE):
            ix = np.flatnonzero(env == e)
            for i in ix:
                cand = ix[cell[ix] != cell[i]]
                out[i] = cand[rng.integers(len(cand), size=T + 1)]
        _C[key] = out
    return _C[key]


def fix_donors(seed):
    """(n,) the engine's random-cell scheme (one fixed donor per probe: same environment, another cell) under seed"""
    key = ("fix_donors", seed)
    if key not in _C:
        _C[key] = ti.donors(ti.probes(), "random", seed)
    return _C[key]


def gen_pool(seed, e):
    """PPC generated trajectories of every cell of environment e, drawn as the probes (o_ti ~ Bernoulli(L_i(r0, r1))
    with the environment's own tables, i.i.d. over steps): cells (2,000,), obs (2,000, T, 5) bool"""
    L = _probes_full()["L"]
    rng = np.random.default_rng([seed, e])
    pc = np.repeat(np.arange(R * R), PPC)
    obs = rng.random((len(pc), T, NO)) < L[e][:, pc // R, pc % R].T[:, None, :]
    return pc, obs


def shuffled_obs(seed, a):
    """(n, a, 5) the probes' own observations 1..a with every observation bit's sequence permuted independently in time
    (each bit keeps its count, so Bayes's joint posterior after step a is exactly the probe's own)"""
    obs = _probes_full()["obs"][:, :a]
    perm = np.argsort(np.random.default_rng([seed, a]).random(obs.shape), axis=1)
    return np.take_along_axis(obs, perm, axis=1)


def _codes(obs):
    return (np.asarray(obs) * (1 << np.arange(NO))).sum(-1).astype(np.int64)


def ctl_forward(run, ep, conds):
    """the per-step read-out increments (n, T, 20) float32 of the intact network and under every control cut (w = 5):
    the window t-4..t is rerun from a state at a = t - 5 and its read-out replaces the increment at t (t <= 5: the
    network's own), as td_tt_intervene hook (a). States at a: 'start' (h0, c0); 'same_cell' row ^ 1; 'random' the
    engine's donor; 'rand_fix_<seed>' the engine's scheme under another seed; 'rand_step_<seed>' a fresh random-cell
    donor at every a; 'mean_pair' the average (h, c) of the two trajectories of the probe's cell; 'mean_env' the
    environment's mean (h, c) over its 200 probes at a; 'shuffle_<seed>' the network rerun from (h0, c0) over the
    probe's own observations 1..a with every bit's sequence permuted (shuffled_obs); 'bm_net_<seed>' the state at a of
    the generated episode (gen_pool: same environment, another cell, not the same observation history) whose network
    marginals at a are closest (max over the two variables of the total variation); 'flat_<seed>' the state at a of one
    of the K_FLAT generated episodes of another cell with the most uniform network beliefs at a (largest summed
    entropy of the two marginals; drawn per probe). Returns (lg {cond: (n, T, 20)}, info) with the matching
    diagnostics: tv_<seed> (n, CA) achieved distance, ent_<seed> (n, CA) the flat donor's entropy, own_ent (n, CA),
    bm_cnt_<seed> / fl_cnt_<seed> (n, CA, 5) the donor's per-bit observation counts over steps 1..a (for Bayes under
    the identical substitution), bm_cell / fl_cell and bm_idx / fl_idx (n, CA) the donors' cells and pool indices"""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    F = torch.nn.functional
    dev = ti._device()
    t0 = time.time()
    w = ti.load(run, ep)
    p = {k: v.to(dev) for k, v in w.items()}
    H = p["LSTM.weight_hh_l0"].shape[1]
    Whh, Wro, bro = p["LSTM.weight_hh_l0"], p["classifier_readout.weight"], p["classifier_readout.bias"]
    h0, c0 = p["STM"].reshape(1, H), p["LTM"].reshape(1, H)
    PP = ti.probes()
    table = ti._table(p, PP["Z"], dev)
    obs, env, cell = PP["obs"], PP["env"], PP["cell"]
    n = len(env)
    code = _codes(obs)
    same = donors()["same_cell"]
    dfix = {"random": donors()["random"], **{f"rand_fix_{s}": fix_donors(s) for s in SEEDS_FIX}}
    dstep = {f"rand_step_{s}": step_donors(s) for s in SEEDS_STEP}
    lg = {c: np.zeros((n, T, NV * R), np.float32) for c in ("intact",) + tuple(conds)}
    info = {"own_ent": np.zeros((n, CA), np.float32)}
    pools = [s for s in SEEDS_POOL if f"bm_net_{s}" in conds or f"flat_{s}" in conds]
    for s in pools:
        info.update({f"tv_{s}": np.zeros((n, CA), np.float32), f"ent_{s}": np.zeros((n, CA), np.float32),
                     f"bm_cnt_{s}": np.zeros((n, CA, NO), np.int8), f"fl_cnt_{s}": np.zeros((n, CA, NO), np.int8),
                     f"bm_cell_{s}": np.zeros((n, CA), np.int16), f"fl_cell_{s}": np.zeros((n, CA), np.int16),
                     f"bm_idx_{s}": np.zeros((n, CA), np.int16), f"fl_idx_{s}": np.zeros((n, CA), np.int16)})
    shuf = [s for s in SEEDS_SHUF if f"shuffle_{s}" in conds]
    shuf_codes = {s: [_codes(shuffled_obs(s, a)) for a in range(1, CA + 1)] for s in shuf}
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    def step(xin, h, c):
        pre = xin + h @ Whh.T
        i, f, g, o = pre.chunk(4, -1)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        return o.sigmoid() * c.tanh(), c

    def run_intact(env_t, code_t):
        b = len(code_t)
        h, c = h0.expand(b, H).contiguous(), c0.expand(b, H).contiguous()
        Hs, Cs = torch.empty(b, T + 1, H, device=dev), torch.empty(b, T + 1, H, device=dev)
        Hs[:, 0], Cs[:, 0] = h, c
        out = torch.empty(b, T, NV * R, device=dev)
        for s_ in range(T):
            h, c = step(table[env_t, code_t[:, s_]], h, c)
            Hs[:, s_ + 1], Cs[:, s_ + 1] = h, c
            out[:, s_] = F.linear(h, Wro, bro)
        return out, Hs, Cs

    def windows(env_t, code_t, DH, DC):
        """donor states (b, CA, H) at a = 1..CA -> read-out increments (b, CA, 20) at t = a + 5 (own inputs a+1..t)"""
        b = len(code_t)
        hs, cs = DH.transpose(0, 1).reshape(CA * b, H), DC.transpose(0, 1).reshape(CA * b, H)
        er = env_t.repeat(CA)
        for k in range(1, CW + 1):
            hs, cs = step(table[er, code_t[:, k:k + CA].transpose(0, 1).reshape(CA * b)], hs, cs)
        return F.linear(hs, Wro, bro).reshape(CA, b, NV * R).transpose(0, 1)

    def splice(own, cut):
        L_ = own.clone()
        L_[:, CW:] = cut
        return L_.cpu().numpy()

    def ent_of(q):
        return -(q * torch.log(q.clamp_min(1e-300))).sum(-1).sum(-1)

    def ix(a):
        return torch.as_tensor(np.asarray(a, np.int64), device=dev)
    for ch in np.array_split(np.arange(NE), 10):                             # 5 environments (1,000 probes) per chunk
        rows = np.flatnonzero(np.isin(env, ch))
        loc = np.full(n, -1, np.int64)
        loc[rows] = np.arange(len(rows))
        b = len(rows)
        env_t, code_t = ix(env[rows]), ix(code[rows])
        own, Hs, Cs = run_intact(env_t, code_t)
        lg["intact"][rows] = own.cpu().numpy()
        q = torch.softmax(own.double().cumsum(1)[:, :CA].reshape(b, CA, NV, R), -1)
        info["own_ent"][rows] = ent_of(q).cpu().numpy()
        el = ix(np.searchsorted(ch, env[rows]))
        S1, C1 = Hs[:, 1:CA + 1], Cs[:, 1:CA + 1]
        for cd in conds:
            if cd == "start":
                DH, DC = h0.expand(b, CA, H), c0.expand(b, CA, H)
            elif cd == "same_cell":
                j = ix(loc[same[rows]])
                DH, DC = S1[j], C1[j]
            elif cd in dfix:
                j = ix(loc[dfix[cd][rows]])
                DH, DC = S1[j], C1[j]
            elif cd in dstep:
                j = ix(loc[dstep[cd][rows][:, 1:CA + 1]])                    # (b, CA)
                ai = ix(np.arange(1, CA + 1))[None].expand(b, CA)
                DH, DC = Hs[j, ai], Cs[j, ai]
            elif cd == "mean_pair":
                j = ix(loc[same[rows]])
                DH, DC = .5 * (S1 + S1[j]), .5 * (C1 + C1[j])
            elif cd == "mean_env":
                mh = torch.stack([S1[el == k].mean(0) for k in range(len(ch))])
                mc_ = torch.stack([C1[el == k].mean(0) for k in range(len(ch))])
                DH, DC = mh[el], mc_[el]
            elif cd.startswith("shuffle_"):
                s = int(cd.split("_")[1])
                DH, DC = torch.empty(b, CA, H, device=dev), torch.empty(b, CA, H, device=dev)
                for a in range(1, CA + 1):
                    cda = ix(shuf_codes[s][a - 1][rows])
                    h, c = h0.expand(b, H).contiguous(), c0.expand(b, H).contiguous()
                    for s_ in range(a):
                        h, c = step(table[env_t, cda[:, s_]], h, c)
                    DH[:, a - 1], DC[:, a - 1] = h, c
            else:
                continue                                                     # pool conditions: per environment below
            lg[cd][rows] = splice(own, windows(env_t, code_t, DH.contiguous(), DC.contiguous()))
            del DH, DC
        del Hs, Cs, S1, C1
    for s in pools:                                                          # generated pools, one environment at a time
        for e in range(NE):
            rows = np.flatnonzero(env == e)
            b = len(rows)
            env_t, code_t = ix(env[rows]), ix(code[rows])
            own, _, _ = run_intact(env_t, code_t)
            q = torch.softmax(own.double().cumsum(1)[:, :CA].reshape(b, CA, NV, R), -1)
            pc, pobs = gen_pool(s, e)
            pcode = ix(_codes(pobs))
            plg, PH, PC_ = run_intact(ix(np.full(len(pc), e)), pcode)
            pq = torch.softmax(plg.double().cumsum(1)[:, :CA].reshape(len(pc), CA, NV, R), -1)
            pent = ent_of(pq)                                                # (Np, CA)
            other = ix(cell[rows])[:, None] != ix(pc)[None]                  # (b, Np) another cell
            frng = np.random.default_rng([s, e, 1])
            bj = torch.empty(b, CA, dtype=torch.long, device=dev)
            fj = torch.empty(b, CA, dtype=torch.long, device=dev)
            tv_ = torch.empty(b, CA, dtype=torch.float64, device=dev)
            for a in range(1, CA + 1):
                d = .5 * (q[:, a - 1, None] - pq[None, :, a - 1]).abs().sum(-1).amax(-1)
                same_hist = (code_t[:, None, :a] == pcode[None, :, :a]).all(-1)
                d = d.masked_fill(~other | same_hist, np.inf)
                tv_[:, a - 1], bj[:, a - 1] = d.min(1)
                top = torch.argsort(pent[:, a - 1], descending=True)[:K_FLAT + 3 * PPC]
                ok = (other & ~same_hist)[:, top]                            # (b, 70) allowed candidates, in order
                kth = torch.minimum(ix(frng.integers(K_FLAT, size=b) + 1), ok.long().sum(1))
                sel = (ok.long().cumsum(1) == kth[:, None]) & ok
                fj[:, a - 1] = top[sel.float().argmax(1)]
            ai = ix(np.arange(1, CA + 1))[None].expand(b, CA)
            for nm, jj in (("bm_net", bj), ("flat", fj)):
                if f"{nm}_{s}" in conds:
                    lg[f"{nm}_{s}"][rows] = splice(own, windows(env_t, code_t, PH[jj, ai].contiguous(),
                                                                PC_[jj, ai].contiguous()))
            info[f"tv_{s}"][rows] = tv_.cpu().numpy()
            info[f"ent_{s}"][rows] = pent[fj, ai - 1].cpu().numpy()
            cum = np.cumsum(pobs, 1).astype(np.int8)                         # (Np, T, 5) per-bit counts over 1..a
            bjn, fjn = bj.cpu().numpy(), fj.cpu().numpy()
            an = np.arange(CA)[None]
            info[f"bm_cnt_{s}"][rows] = cum[bjn, an]
            info[f"fl_cnt_{s}"][rows] = cum[fjn, an]
            info[f"bm_cell_{s}"][rows] = pc[bjn]
            info[f"fl_cell_{s}"][rows] = pc[fjn]
            info[f"bm_idx_{s}"][rows], info[f"fl_idx_{s}"][rows] = bjn, fjn
            del PH, PC_, plg
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        info["peak_gpu_mb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 20
    info["forward_s"] = time.time() - t0
    return lg, info


def bayes_cut_base(base):
    """Bayes's marginal log beliefs (n, T, 2, R) when the history before every window start a = t - 5 is replaced by
    a donor whose joint log weight at a is base (n, CA, R, R) (any normalisation): for t > 5 the increment is
    marg(base_a + own observations a+1..t) - marg(base_a + own a+1..t-1); t <= 5: Bayes's own; the cumulative sum,
    normalised (bayes_logq with a per-(probe, a) donor)"""
    Lam = _lam()
    n = len(Lam)
    inc = np.zeros((n, T, NV, R))
    for t in range(1, T + 1):
        if t <= CW:
            inc[:, t - 1] = _marg(Lam[:, t]) - _marg(Lam[:, t - 1])
        else:
            a = t - CW
            bs = base[:, a - 1]
            inc[:, t - 1] = _marg(bs + Lam[:, t] - Lam[:, a]) - _marg(bs + Lam[:, t - 1] - Lam[:, a])
    cum = np.cumsum(inc, 1)
    return cum - logsumexp(cum, -1, keepdims=True)


def _lam_counts(cnt):
    """the joint log-likelihood (n, CA, R, R) of histories given by their per-bit counts cnt (n, CA, 5) over steps
    1..a (a = 1..CA), in each probe's own environment (sufficient: the observations are i.i.d. Bernoulli given the
    cell)"""
    P = _probes_full()
    L = P["L"][P["env"]]                                                             # (n, 5, R, R)
    k = cnt.astype(np.float64)
    a = np.arange(1, CA + 1, dtype=np.float64)[None, :, None]
    return np.einsum("nai,nijk->najk", k, np.log(L)) + np.einsum("nai,nijk->najk", a - k, np.log1p(-L))


def bayes_ctl_logq(cond):
    """Bayes under the identical substitution for the checkpoint-independent controls: rand_fix / rand_step (the
    donors' histories), mean_pair (the log of the mixture of the two same-cell posteriors at a: the analogue of
    averaging the two states, as the check's), mean_env (the log of the environment's mixture of the 200 posteriors at
    a), shuffle (the shuffled history; equal to Bayes intact up to rounding: the counts are kept)"""
    Lam = _lam()
    n = len(Lam)
    P = _probes_full()
    an = np.arange(1, CA + 1)
    if cond.startswith("rand_fix_"):
        d = fix_donors(int(cond.split("_")[2]))
        return bayes_cut_base(Lam[d][:, 1:CA + 1])
    if cond.startswith("rand_step_"):
        d = step_donors(int(cond.split("_")[2]))[:, 1:CA + 1]
        return bayes_cut_base(Lam[d, an[None]])
    if cond in ("mean_pair", "mean_env"):
        post = Lam[:, 1:CA + 1] - logsumexp(Lam[:, 1:CA + 1], (-2, -1), keepdims=True)
        post = np.exp(post)
        if cond == "mean_pair":
            mix = .5 * (post + post[np.arange(n) ^ 1])
        else:
            mix = np.stack([post[P["env"] == e].mean(0) for e in range(NE)])[P["env"]]
        return bayes_cut_base(np.log(mix))
    if cond.startswith("shuffle_"):
        s = int(cond.split("_")[1])
        cnt = np.stack([shuffled_obs(s, a).sum(1) for a in an], 1)                  # (n, CA, 5)
        return bayes_cut_base(_lam_counts(cnt))
    raise ValueError(cond)


CTL_FIXED = tuple(f"rand_fix_{s}" for s in SEEDS_FIX) + tuple(f"rand_step_{s}" for s in SEEDS_STEP) + \
    ("mean_pair", "mean_env") + tuple(f"shuffle_{s}" for s in SEEDS_SHUF)


def ctl_stats(lq, right=None):
    """the control's statistics: the 18 cheap rows (td_tt_intervene.score 'cheap') with their 1,000 shared resamples,
    the answers at T (n, 2), the true value's margin at T (n, 2), per-step answers (goal / both) summed per
    environment"""
    P = _probes_full()
    lq = np.asarray(lq, np.float64)
    lq = lq - logsumexp(lq, -1, keepdims=True)
    r, env, n = P["r"], P["env"], len(P["env"])
    if right is None:
        right = lq.argmax(-1) == r[:, None, :]
    rows = ti.score(lq, "cheap", right=right)
    bs = ti.resample(rows)
    ar = np.arange(n)
    c = lq - lq.mean(-1, keepdims=True)
    idx = np.repeat(env, T) * T + np.tile(np.arange(T), n)
    return dict(value=np.array([rows[k]["value"] for k in ti.CHEAP]), boot=np.stack([bs[k] for k in ti.CHEAP]),
                right_T=right[:, -1], mT=_margin(c[:, -1], r).astype(np.float32),
                acc_goal=np.bincount(idx, right[ar, :, P["goal"]].ravel().astype(np.float64), NE * T).reshape(NE, T),
                acc_both=np.bincount(idx, right.all(-1).ravel().astype(np.float64), NE * T).reshape(NE, T))


CSTACK = ("value", "boot", "right_T", "mT", "acc_goal", "acc_both")


def bayes_ctl(force=False):
    """-> OUT/controls/bayes_ctl.npz: Bayes under the checkpoint-independent controls (CTL_FIXED), ctl_stats of each,
    and the check that the shuffled histories leave Bayes exactly as intact"""
    path = os.path.join(CTL, "bayes_ctl.npz")
    if os.path.exists(path) and not force:
        return _load(path)
    t0 = time.time()
    lq0 = bayes_logq()
    res, chk = {}, {}
    for c in CTL_FIXED:
        lq = bayes_ctl_logq(c)
        res[c] = ctl_stats(lq)
        if c.startswith("shuffle_"):
            chk[c] = float(np.abs(lq - lq0).max())
        print(f"  Bayes {c}: goal accuracy at T {res[c]['value'][0]:.4f} ({time.time() - t0:.0f} s)", flush=True)
    out = {k: np.stack([res[c][k] for c in CTL_FIXED]) for k in CSTACK}
    i0 = ctl_stats(lq0)
    out.update({f"intact_{k}": i0[k] for k in CSTACK})
    out.update(conds=np.array(CTL_FIXED), rows=np.array(ti.CHEAP), shuffle_vs_intact=json.dumps(chk))
    os.makedirs(CTL, exist_ok=True)
    _save(path, out)
    print(f"saved {path} ({time.time() - t0:.0f} s); shuffled histories vs Bayes intact, max |d log q|: {chk}")
    return out


def bm_null_match(seed):
    """the matched null of the belief-matched swap (added 11-02 after the second check): for every probe and window
    start a = 1..CA, the generated episode of the same pool (gen_pool(seed, e): same environment, another cell, another
    observation history over 1..a) whose EXACT BAYES marginals at a are closest to the probe's own Bayes marginals (total
    variation, the larger of the two variables'): the same pools, candidates and distance as the network's
    belief-matched donors (ctl_forward 'bm_net'), matched on Bayes's own belief instead of the network's read-out.
    Returns cnt (n, CA, 5) int8 (the donors' per-bit counts over 1..a), tv (n, CA), cell (n, CA)"""
    P = _probes_full()
    Lam = _lam()
    env, cell, L = P["env"], ti.probes()["cell"], P["L"]
    code = _codes(P["obs"])
    n = len(env)
    cnt = np.zeros((n, CA, NO), np.int8)
    tv = np.zeros((n, CA))
    dcell = np.zeros((n, CA), np.int16)
    for e in range(NE):
        rows = np.flatnonzero(env == e)
        pc, pobs = gen_pool(seed, e)
        pcode = _codes(pobs)
        cum = np.cumsum(pobs, 1).astype(np.float64)                                   # (Np, T, 5)
        lg, l1 = np.log(L[e]), np.log1p(-L[e])
        for a in range(1, CA + 1):
            k = cum[:, a - 1]
            lj = np.einsum("pi,ijk->pjk", k, lg) + np.einsum("pi,ijk->pjk", a - k, l1)
            pm = np.exp(_marg(lj - logsumexp(lj, (-2, -1), keepdims=True)))            # (Np, 2, R) pool, Bayes
            oj = Lam[rows, a]
            om = np.exp(_marg(oj - logsumexp(oj, (-2, -1), keepdims=True)))            # (b, 2, R) probes, Bayes
            d = .5 * np.abs(om[:, None] - pm[None]).sum(-1).max(-1)                   # (b, Np)
            ok = (cell[rows][:, None] != pc[None]) & ~(code[rows][:, None, :a] == pcode[None, :, :a]).all(-1)
            d = np.where(ok, d, np.inf)
            j = d.argmin(1)
            tv[rows, a - 1] = d[np.arange(len(rows)), j]
            cnt[rows, a - 1] = cum[j, a - 1]
            dcell[rows, a - 1] = pc[j]
    return cnt, tv, dcell


def bayes_bm_null(force=False):
    """-> OUT/controls/bayes_bm_null.npz: Bayes under the matched null of the belief-matched swap (bm_null_match) for
    every pool seed: ctl_stats (the 18 cheap rows with resamples, answers and margins at T, per-step answers), the
    achieved distances tv (n, CA), the donors' counts and cells. Checkpoint-independent (Bayes's own beliefs)"""
    path = os.path.join(CTL, "bayes_bm_null.npz")
    if os.path.exists(path) and not force:
        return _load(path)
    t0 = time.time()
    res, out = [], {}
    for s in SEEDS_POOL:
        cnt, tv, dcell = bm_null_match(s)
        res.append(ctl_stats(bayes_cut_base(_lam_counts(cnt))))
        out[f"tv_{s}"], out[f"cnt_{s}"], out[f"cell_{s}"] = tv.astype(np.float32), cnt, dcell
        print(f"  matched null, pool {s}: median TV {np.median(tv):.4f}; goal accuracy at T {res[-1]['value'][0]:.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    out.update({k: np.stack([r[k] for r in res]) for k in CSTACK})
    out.update(seeds=np.array(SEEDS_POOL), rows=np.array(ti.CHEAP), seconds=time.time() - t0)
    os.makedirs(CTL, exist_ok=True)
    _save(path, out)
    print(f"saved {path} ({time.time() - t0:.0f} s)")
    return _load(path)


def net_revisions(lq):
    """network-defined revision probes (the check's definition; goal variable): the network's belief puts more than .6
    on a value that is not its own answer at T at some step 8-24 (n,) bool"""
    P = _probes_full()
    ar, g = np.arange(len(P["goal"])), P["goal"]
    qg = np.exp(lq[ar, :, g])                                                        # (n, T, R)
    aT = qg[:, -1].argmax(-1)
    return ((qg[:, 7:24] > .6) & (np.arange(R)[None, None] != aT[:, None, None])).any((1, 2))


def ctl_ck(run, ep, force=False):
    """the controls at one checkpoint -> OUT/controls/<run>/ck_EEEEE.npz: per condition ctl_stats; for the pool
    conditions also Bayes under the identical substitution (its donors depend on the network's beliefs) and the
    matching diagnostics; the network-defined revision probes; the validation of start / same_cell / random against
    the harvest (answers at T)"""
    path = ctl_path(run, ep)
    if os.path.exists(path) and not force:
        return
    import torch
    t0 = time.time()
    conds = ctl_conds(ep)
    lg, info = ctl_forward(run, ep, conds)
    torch.cuda.empty_cache()
    tf = time.time() - t0
    P = _probes_full()

    def lqr(x):
        cum = np.cumsum(x.astype(np.float64), 1).reshape(len(x), T, NV, R)
        lq = cum - logsumexp(cum, -1, keepdims=True)
        return lq, lq.argmax(-1) == P["r"][:, None, :]
    lq0, rt0 = lqr(lg["intact"])
    res = {"intact": ctl_stats(lq0, rt0)}
    netrev = net_revisions(lq0)
    for c in conds:
        lq, rt = lqr(lg[c])
        res[c] = ctl_stats(lq, rt)
    z = _load(ck_path(run, ep))
    val = {c: int((res[c]["right_T"] != z["right_T"][cix("w5_" + c)]).sum()) for c in ("start", "same_cell", "random")}
    val["intact"] = int((res["intact"]["right_T"] != z["right_T"][0]).sum())
    cl = ("intact",) + tuple(conds)
    out = {k: np.stack([res[c][k] for c in cl]) for k in CSTACK}
    out.update(conds=np.array(cl), rows=np.array(ti.CHEAP), netrev=netrev, run=run, ep=int(ep),
               x=float(gap.train_x(ep)), forward_s=tf, validation=json.dumps(val),
               peak_gpu_mb=float(info.get("peak_gpu_mb", 0.)), own_ent=info["own_ent"])
    bres = {}
    for s in SEEDS_POOL:
        for nm in ("bm_net", "flat"):
            c = f"{nm}_{s}"
            if c in conds:
                key = "bm_cnt" if nm == "bm_net" else "fl_cnt"
                bres[c] = ctl_stats(bayes_cut_base(_lam_counts(info[f"{key}_{s}"])))
                out[f"tv_{s}"], out[f"ent_{s}"] = info[f"tv_{s}"], info[f"ent_{s}"]
                out[f"bm_cell_{s}"], out[f"fl_cell_{s}"] = info[f"bm_cell_{s}"], info[f"fl_cell_{s}"]
    if bres:
        bl = tuple(bres)
        out.update({f"bayes_{k}": np.stack([bres[c][k] for c in bl]) for k in CSTACK})
        out["bayes_conds"] = np.array(bl)
    out["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    _save(tmp, out)
    os.replace(tmp, path)
    g = {c: out["value"][i, 0] for i, c in enumerate(cl)}
    print(f"{time.strftime('%H:%M:%S')} controls {run} raw {ep} (training {gap.train_x(ep):,.0f}): forward {tf:.0f} s, "
          f"total {time.time() - t0:.0f} s, peak GPU {out['peak_gpu_mb']:.0f} MB; validation {val}; goal accuracy "
          + ", ".join(f"{c} {v:.4f}" for c, v in g.items()), flush=True)


def ctl_jobs():
    return [("trained", ep) for ep in selection("trained")]


def ctl_worker(k, nw):
    os.makedirs(os.path.join(TMP_FIX, "claims_ctl"), exist_ok=True)
    jb = ctl_jobs()
    full = set(ctl_full_eps())
    jb = [j for j in jb if j[1] in full] + [j for j in jb if j[1] not in full]       # the long jobs first
    for run, ep in jb:
        if os.path.exists(ctl_path(run, ep)):
            continue
        try:
            fd = os.open(os.path.join(TMP_FIX, "claims_ctl", f"{run}_{ep:05d}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, f"worker {k}/{nw} pid {os.getpid()}".encode())
        os.close(fd)
        ctl_ck(run, ep)


def ctl_harvest(nw=2):
    """launches nw control workers (GPU 0, 2 threads each; logs in TMP) after the Bayes control cache; returns at once"""
    bayes_ctl()
    os.makedirs(os.path.join(TMP_FIX, "claims_ctl"), exist_ok=True)
    for f in os.listdir(os.path.join(TMP_FIX, "claims_ctl")):
        os.remove(os.path.join(TMP_FIX, "claims_ctl", f))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MPLBACKEND="Agg",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    pids = []
    for k in range(nw):
        log = open(os.path.join(TMP_FIX, f"ctl_{k}.log"), "a")
        p = subprocess.Popen(["timeout", "10800", PY, os.path.abspath(__file__), "ctl_worker", str(k), str(nw)],
                             stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT, start_new_session=True)
        pids.append(p.pid)
    print("control workers", pids, "logs", TMP_FIX)
    with open(os.path.join(TMP_FIX, "pids_ctl.txt"), "w") as f:
        f.write(" ".join(map(str, pids)))


# ═══════════════════════════════════════════════════════════════════
# Analysis: dependence curves with the shared resamples, per-step and per-probe splits, crossings
# ═══════════════════════════════════════════════════════════════════

ROWNAME = {   # this module's rows -> the td_tt_timeline row of the same measure (its family's c10 / c50)
    "acc_goal_T": "goal accuracy at T", "acc_both_T": "goal and context both right at T", "far_T": "far errors at T",
    "merged_T": "distant pairs merged at T", "excess": "jump excess angle over independent",
    "G": "long-path coupling G", "band4": "angle of jumps >= 4 cells", "emass_slope": "end-mass slope within steps (maps)",
    "dir_slope": "direction slope within steps (maps)", "held_two": "held per variable, corner-split ends",
    "held_one": "held per variable, one-variable end splits", "held_joint": "corner split held (joint)",
    "split_grad": "split gradient (most minus least entangled tenth)",
    "split_grad_matched": "split gradient, distance-matched", "toward6": "six-step move toward own cells",
    "fold_idx": "fold index (corners - edges)", "corners": "opposite corners"}


def load_run(run):
    """the stored checkpoints of a run, stacked: STACK arrays (K, C, ...), x (K,) training episodes, eps, meta"""
    key = ("run", run)
    if key not in _C:
        eps = [e for e in selection(run) if os.path.exists(ck_path(run, e))]
        if not eps:
            return None
        D, meta = {k: [] for k in STACK}, []
        for e in eps:
            z = _load(ck_path(run, e))
            for k in STACK:
                D[k].append(z[k])
            meta.append(dict(checks=json.loads(str(z["checks"])), seconds=float(z["seconds"]),
                             forward_s=float(z["forward_s"]), peak=float(z["peak_gpu_mb"])))
        D = {k: np.stack(v) for k, v in D.items()}
        D.update(eps=np.array(eps), x=gap.train_x(np.array(eps)), meta=meta)
        _C[key] = D
    return _C[key]


def load_bayes():
    if "B" not in _C:
        _C["B"] = _load(os.path.join(OUT, "bayes.npz"))
    return _C["B"]


def cix(cond):
    return CONDS.index(cond)


def dep(run, cond, row):
    """memory dependence of a score row = intact - cut (the row's own units): the network's (K,) with its resamples
    (NB, K), Bayes's under the same cut (scalar, (NB,)), and the excess network - Bayes ((K,), (NB, K)); levels: the
    network's intact and cut values (K,), Bayes's intact / cut and naive's (scalars)"""
    D, B = load_run(run), load_bayes()
    c, k = cix(cond), ROWS.index(row)
    v = D["value"][:, 0, k] - D["value"][:, c, k]
    vb = (D["boot"][:, 0, k] - D["boot"][:, c, k]).T
    bv = B["value"][0, k] - B["value"][c, k]
    bb = B["boot"][0, k] - B["boot"][c, k]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = dict(x=D["x"], net=v, net_boot=vb, net_lo=np.nanpercentile(vb, 5, 0), net_hi=np.nanpercentile(vb, 95, 0),
                   bayes=bv, bayes_boot=bb, bayes_lo=np.nanpercentile(bb, 5), bayes_hi=np.nanpercentile(bb, 95),
                   ex=v - bv, ex_boot=vb - bb[:, None], intact=D["value"][:, 0, k], cut=D["value"][:, c, k],
                   bayes_intact=B["value"][0, k], bayes_cut=B["value"][c, k], naive=float(ti.refs()["naive"][row]["value"]))
        out["ex_lo"], out["ex_hi"] = np.nanpercentile(out["ex_boot"], [5, 95], 0)
    return out


def _wsteps(cond):
    return 0 if cond == "intact" else int(cond.split("_")[0][1:])


def step_measure(run, cond, kind, by="f", steps=None):
    """a per-var-step measure, averaged over the steps after the cut window (t > w; or `steps`, 0-based indices) per
    group (by 'f': the variable's own entanglement fifth within its step; 'g': Bayes's step group), as a ratio of
    sums with the shared environment resamples. kind: 'ev' evidence for the true value lost (intact - cut, nats per
    var-step); 'err' local update error added (cut - intact); 'dd' squared change of the step's update |e_cut -
    e_int|^2; 'flip' answers at T lost when only this step's increment comes from the cut (per var-step); 'dm' margin
    at T lost by that single-step swap (nats); 'sz' update size |e_cut|^2; 'gain' gain of the cut update in naive's
    direction (Sigma e.N / Sigma N.N; 'f' only). run 'bayes' = Bayes under the same cut. Returns value (K, G) / (G,),
    boot (NB, K, G) / (NB, G), and the counts"""
    B = load_bayes()
    D = B if run == "bayes" else load_run(run)
    c = cix(cond)
    w = _wsteps(cond)
    sl = np.arange(w, T) if steps is None else np.asarray(steps)
    sfx = "_" + by
    cnt = B["cnt" + sfx][:, sl].sum(1)                                                 # (NE, G)
    one = run == "bayes"

    def g(k):
        a = D[k + sfx]
        return (a[c] if one else a[:, c])[..., sl, :].sum(-2)                          # (NE, G) or (K, NE, G)

    def g0(k):
        a = D[k + sfx]
        return (a[0] if one else a[:, 0])[..., sl, :].sum(-2)
    if kind == "ev":
        num = g0("ev") - g("ev")
    elif kind == "err":
        num = g("err") - g0("err")
    elif kind == "dd":
        num = g("sz") + g0("sz") - 2 * g("ci")
    elif kind in ("flip", "dm", "sz"):
        num = g(kind)
    elif kind == "gain":
        num, cnt = g("dn"), B["nn" + sfx][:, sl].sum(1)
    else:
        raise ValueError(kind)
    C = gap.boot_counts()
    if one:
        return dict(value=num.sum(0) / cnt.sum(0), boot=(C @ num) / (C @ cnt), cnt=cnt.sum(0))
    return dict(value=num.sum(1) / cnt.sum(0), boot=np.einsum("be,keg->bkg", C, num) / (C @ cnt)[:, None, :],
                cnt=cnt.sum(0), x=D["x"])


def echo_coef(run, cond="intact", steps=None):
    """the per-step update regressed on the new evidence and the belief before the step, e_t ~ a N_t + b c_(t-1)
    (least squares over the var-steps of each entanglement fifth, steps 2-30 by default; N = naive's centred increment,
    c = the observer's own centred log belief): a (gain on the new evidence) and b (share of the belief before t
    re-added beyond the new evidence; naive: a = 1, b = 0 exactly), with resamples. run 'bayes' / 'naive' or a network
    run. Returns dict(a, b (K, G) or (G,), a_boot, b_boot (NB, ...))"""
    B = load_bayes()
    sl = np.arange(1, T) if steps is None else np.asarray(steps)
    if run in ("bayes", "naive"):
        D = {k: (B[k][cix(cond)] if run == "bayes" else B["naive_" + k])[None] for k in ("dn_f", "eb_f", "bb_f", "nb_f")}
    else:
        D = {k: load_run(run)[k][:, cix(cond)] for k in ("dn_f", "eb_f", "bb_f", "nb_f")}
    nn = B["nn_f"][:, sl].sum(1)[None]                                                 # (1, NE, G)
    S = {k: v[:, :, sl].sum(2) for k, v in D.items()}                                   # (K, NE, G)
    C = gap.boot_counts()

    def solve(nn_, nb_, bb_, dn_, eb_):
        det = nn_ * bb_ - nb_ * nb_
        return (bb_ * dn_ - nb_ * eb_) / det, (nn_ * eb_ - nb_ * dn_) / det
    a, b = solve(nn.sum(1), S["nb_f"].sum(1), S["bb_f"].sum(1), S["dn_f"].sum(1), S["eb_f"].sum(1))
    rs = {k: np.einsum("be,keg->bkg", C, v) for k, v in dict(nn=np.broadcast_to(nn, S["nb_f"].shape), **S).items()}
    ab, bbt = solve(rs["nn"], rs["nb_f"], rs["bb_f"], rs["dn_f"], rs["eb_f"])
    one = run in ("bayes", "naive")
    return dict(a=a[0] if one else a, b=b[0] if one else b, a_boot=ab[:, 0] if one else ab, b_boot=bbt[:, 0] if one else bbt)


def per_step_acc(run, cond, which="goal"):
    """answers right at each step t (goal or both variables) under the condition: (K, T) / (T,)"""
    B = load_bayes()
    D = B if run == "bayes" else load_run(run)
    a = D["acc_" + which]
    a = a[cix(cond)] if run == "bayes" else a[:, cix(cond)]
    return a.sum(-2) / (NE * PER_ENV)


def family_dep(run, cond, mask, which="goal"):
    """goal-variable accuracy at T lost under the cut (intact - cut) on the probes of a mask (n,): network (K,) and
    resamples (NB, K); Bayes (scalar, (NB,)); counts"""
    B, P = load_bayes(), _probes_full()
    ar, g, env = np.arange(len(P["goal"])), P["goal"], P["env"]
    C = gap.boot_counts()
    den = np.bincount(env, mask.astype(np.float64), NE)

    def sums(rt):                                                                      # rt (..., n, 2) -> (..., NE)
        v = rt[..., ar, g].astype(np.float64) * mask
        return np.stack([np.bincount(env, vv, NE) for vv in v.reshape(-1, len(env))]).reshape(v.shape[:-1] + (NE,))
    out = {}
    if run != "bayes":
        D = load_run(run)
        num = sums(D["right_T"][:, 0]) - sums(D["right_T"][:, cix(cond)])               # (K, NE)
        out.update(net=num.sum(1) / den.sum(), net_boot=(C @ num.T) / (C @ den)[:, None])
    nb = sums(B["right_T"][0]) - sums(B["right_T"][cix(cond)])
    out.update(bayes=nb.sum() / den.sum(), bayes_boot=(C @ nb) / (C @ den), n=int(mask.sum()))
    return out


def first_cross(x, y, level):
    """the first training episode at which y reaches `level` (log-interpolated from the previous checkpoint); NaN if
    never; y (..., K), level scalar or (...)"""
    y = np.asarray(y, np.float64)
    shp = y.shape[:-1]
    y2 = y.reshape(-1, y.shape[-1])
    lev = np.broadcast_to(np.asarray(level, np.float64), shp).reshape(-1)
    with np.errstate(invalid="ignore"):
        past = y2 >= lev[:, None]
    out = np.full(len(y2), np.nan)
    has = past.any(1)
    k = np.argmax(past, 1)
    lx = np.log(np.asarray(x, np.float64))
    for i in np.flatnonzero(has):
        j = k[i]
        if j == 0:
            out[i] = x[0]
            continue
        y0, y1 = y2[i, j - 1], y2[i, j]
        f = np.clip((lev[i] - y0) / (y1 - y0), 0, 1) if np.isfinite(y0) and y1 != y0 else 1.
        out[i] = np.exp(lx[j - 1] + f * (lx[j] - lx[j - 1]))
    return out.reshape(shp) if shp else float(out[0])


def _runmean3(y):
    yp = np.concatenate([y[..., :1], y, y[..., -1:]], -1)
    return (yp[..., :-2] + yp[..., 1:-1] + yp[..., 2:]) / 3


def rise_fall(x, y, yb):
    """a transient's timing (td_tt_memory_events' rule): peak = the maximum of the 3-checkpoint running mean; rise =
    the first crossing of half the peak, fall = after which the curve stays below half the peak to the end (both
    log-interpolated), levels recomputed in every resample. Returns dict(peak, peak_x, rise, fall, and their resamples
    and 5-95% intervals (gap.interval: a resample that never crosses counts as after the end))"""
    pk = _runmean3(y)
    pkb = _runmean3(yb)
    out = dict(peak=float(np.nanmax(pk)), peak_x=float(x[np.nanargmax(pk)]), peak_boot=np.nanmax(pkb, 1))
    out["rise"] = first_cross(x, y, out["peak"] / 2)
    out["rise_boot"] = first_cross(x, yb, out["peak_boot"] / 2)
    out["fall"] = gap.crossing(x, y, out["peak"] / 2, sign=-1.)
    out["fall_boot"] = gap.crossing(x, yb, out["peak_boot"] / 2, sign=-1.)
    for k in ("rise", "fall", "peak"):
        b = out[k + "_boot"]
        out[k + "_lo"], out[k + "_hi"], out[k + "_share"] = (gap.interval(b) if k != "peak" else
                                                              (*np.nanpercentile(b, [5, 95]), 1.))
    return out


def family_marks(names=("acc_goal_T", "split_grad", "emass_slope", "held_two", "excess", "G", "dir_slope", "band4",
                        "far_T", "held_one", "held_joint")):
    """{row: dict(c10=cross dict, c50=cross dict, row=timeline row)} of the Trained RNN (td_tt_timeline.cross on the
    exported resamples; paired with every probe-based row here)"""
    if "marks" not in _C:
        import td_tt_timeline as tt
        rows = tt.load_rows()
        out = {}
        for k in names:
            r = next(q for q in rows if q["name"] == ROWNAME[k] and q["run"] == "trained")
            out[k] = dict(c10=tt.cross(r, .1), c50=tt.cross(r, .5), c0=tt.cross(r, 0.), row=r)
        _C["marks"] = out
    return _C["marks"]


KEY_X = (400, 880, 1360, 1680, 2080, 3000, 4400, 7600, 11200, 39996)   # training episodes reported in numbers()


def _k(x, xv):
    return int(np.argmin(np.abs(np.log(x) - np.log(xv))))


def _iv(v, b):
    lo, hi = np.nanpercentile(b, [5, 95])
    return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _cx(c):
    """a crossing dict (td_tt_timeline.cross or rise_fall parts) as text"""
    if not np.isfinite(c["p"]):
        return "never"
    hi = f"{c['hi']:,.0f}" if np.isfinite(c["hi"]) else "never"
    return f"{c['p']:,.0f} [{c['lo']:,.0f}-{hi}]" + ("" if c.get("share", 1) >= .995 else f" ({c['share']:.0%})")


def _order(a, b):
    """paired share P(a < b) over the resamples (NaN = after the end) and the interval rule"""
    a_, b_ = np.where(np.isnan(a["boot"]), np.inf, a["boot"]), np.where(np.isnan(b["boot"]), np.inf, b["boot"])
    p = float((a_ < b_).mean())
    rule = ("before" if np.isfinite(a["hi"]) and a["hi"] < b["lo"] else "after" if np.isfinite(b["hi"]) and
            a["lo"] > b["hi"] else "with")
    return f"{rule} ({p:.2f})"


def timing(run="trained", cond="w5_start", row="acc_goal_T"):
    """rise / peak / fall of the excess dependence (network - Bayes) of a row under a cut, as crossing dicts"""
    d = dep(run, cond, row)
    rf = rise_fall(d["x"], d["ex"], d["ex_boot"])
    out = {"peak": rf["peak"], "peak_x": rf["peak_x"], "peak_lo": rf["peak_lo"], "peak_hi": rf["peak_hi"]}
    for k in ("rise", "fall"):
        out[k] = dict(p=rf[k], lo=rf[k + "_lo"], hi=rf[k + "_hi"], share=rf[k + "_share"], boot=rf[k + "_boot"])
    return out


def load_where(run):
    """the 'where' pass of a run, stacked: WSTACK arrays (K, 9, ...) over the cut conditions CONDS[1:], x (K,)"""
    key = ("where", run)
    if key not in _C:
        fs_ = sorted(f for f in os.listdir(os.path.join(OUT, run)) if f.startswith("where_") and f.endswith(".npz")
                     and ".tmp" not in f)
        if not fs_:
            return None
        Z = [_load(os.path.join(OUT, run, f)) for f in fs_]
        D = {k: np.stack([z[k] for z in Z]) for k in WSTACK}
        D.update(x=np.array([float(z["x"]) for z in Z]), eps=np.array([int(z["ep"]) for z in Z]),
                 same=np.array([bool(z["same_as_harvest"]) for z in Z]))
        _C[key] = D
    return _C[key]


def where_measure(run, cond, by="f", lost=False, steps=None):
    """the true value's log-odds against the cut's winning competitor, lost by the cut, per variable-step after the
    window (t > w; or `steps`), per group ('f' entanglement fifth, 'g' Bayes's step group); all variable-steps, or
    (lost=True) only those of variables whose answer at T the cut loses. Ratio of sums with the shared resamples;
    run 'bayes': Bayes under the same cut. Returns value (K, G) or (G,), boot, x, and the summed totals per group (the
    share of the whole loss)"""
    B = load_bayes()
    one = run == "bayes"
    D = load_where_bayes() if one else load_where(run)
    c = CONDS[1:].index(cond)
    w = _wsteps(cond)
    sl = np.arange(w, T) if steps is None else np.asarray(steps)
    a = D[("dlx_" if lost else "dl_") + by]
    num = (a[c] if one else a[:, c])[..., sl, :].sum(-2)
    if lost:
        n_ = D["nx_" + by]
        cnt = (n_[c] if one else n_[:, c])[..., sl, :].sum(-2)
    else:
        cnt = B["cnt_" + by][:, sl].sum(1)
    C = gap.boot_counts()
    if one:
        return dict(value=num.sum(0) / cnt.sum(0), boot=(C @ num) / (C @ cnt), total=num.sum(0), cnt=cnt.sum(0))
    if cnt.ndim == 2:
        cnt = np.broadcast_to(cnt, num.shape)
    return dict(value=num.sum(1) / cnt.sum(1), boot=np.einsum("be,keg->bkg", C, num) / np.einsum("be,keg->bkg", C, cnt),
                total=num.sum(1), cnt=cnt.sum(1), x=D["x"])


def load_where_bayes():
    if "BW" not in _C:
        _C["BW"] = bayes_where()
    return _C["BW"]


def margin_change(run, cond, k):
    """at checkpoint index k: the margin of the true value at T (log q(r) - max_(j != r) log q(j), per probe and
    variable) lost by the cut, network vs Bayes: mean, median, the network's and Bayes's answers lost, and each
    observer's margin changes applied to the other's intact margins (is the network's larger loss a larger belief
    change, or the same change on more fragile answers?)"""
    B, D = load_bayes(), load_run(run)
    c = cix(cond)
    m0, m1 = D["mT"][k, 0].astype(np.float64).ravel(), D["mT"][k, c].astype(np.float64).ravel()
    b0, b1 = B["mT"][0].astype(np.float64).ravel(), B["mT"][c].astype(np.float64).ravel()
    dn, db = m0 - m1, b0 - b1
    return dict(net_mean=dn.mean(), net_median=np.median(dn), bayes_mean=db.mean(), bayes_median=np.median(db),
                net_lost=(m0 > 0).mean() - (m1 > 0).mean(), bayes_lost=(b0 > 0).mean() - (b1 > 0).mean(),
                net_on_bayes=(b0 > 0).mean() - ((b0 - dn) > 0).mean(), bayes_on_net=(m0 > 0).mean() - ((m0 - db) > 0).mean())


def need_onset(run, cond, row, share=.5):
    """when a row starts to need memory older than w like Bayes: the training episode after which the network's
    dependence stays above `share` x Bayes's dependence under the same cut (paired resamples; gap.crossing,
    log-interpolated; never = after the end). dict(p, lo, hi, share, boot) as td_tt_timeline.cross"""
    d = dep(run, cond, row)
    sg = np.sign(d["bayes"]) if d["bayes"] != 0 else 1.
    y, yb = sg * (d["net"] - share * d["bayes"]), sg * (d["net_boot"] - share * d["bayes_boot"][:, None])
    cb = gap.crossing(d["x"], yb, 0.)
    lo, hi, sh = gap.interval(cb)
    return dict(p=gap.crossing(d["x"], y, 0.), lo=lo, hi=hi, share=sh, boot=cb)


# ═══════════════════════════════════════════════════════════════════
# Analysis of the controls
# ═══════════════════════════════════════════════════════════════════

CTL_GROUPS = (("random cell, fixed per probe", ("random",) + tuple(f"rand_fix_{s}" for s in SEEDS_FIX)),
              ("random cell, fresh at every window start", tuple(f"rand_step_{s}" for s in SEEDS_STEP)),
              ("flat-belief other cell", tuple(f"flat_{s}" for s in SEEDS_POOL)),
              ("environment-mean state", ("mean_env",)),
              ("same cell, other trajectory", ("same_cell",)),
              ("average of the two same-cell states", ("mean_pair",)),
              ("shuffled own history", tuple(f"shuffle_{s}" for s in SEEDS_SHUF)),
              ("belief-matched other cell", tuple(f"bm_net_{s}" for s in SEEDS_POOL)))


def load_bayes_ctl():
    if "BC" not in _C:
        _C["BC"] = _load(os.path.join(CTL, "bayes_ctl.npz"))
    return _C["BC"]


def load_ctl(run="trained"):
    """the control caches of a run stacked over its checkpoints: conds (union, in order; 'intact' first), value (K, C,
    18) and boot (K, C, 18, NB) of the cheap rows (NaN where a condition was not run), has (K, C); bconds (the pool
    conditions) with Bayes under the identical substitution bvalue (K, Cb, 18), bboot; right_T (K, C, n, 2); acc_goal
    (K, C, NE, T); netrev (K, n); val (validation dicts); matching diagnostics per pool seed (NaN where not run): tv_med
    / tv_q90 (median / 90% of the achieved total variation over probes and window starts), tv_lt05 / tv_lt10 (shares
    below .05 / .10), share_far (belief-matched donors sharing neither latent value with the probe's cell), ent_flat /
    ent_own (median summed entropy of the flat donors' / the probes' own network marginals at a; uniform 4.61)"""
    key = ("ctl", run)
    if key in _C:
        return _C[key]
    eps = [e for e in selection(run) if os.path.exists(ctl_path(run, e))]
    if not eps:
        return None
    P = _probes_full()
    n = len(P["env"])
    Z = [_load(ctl_path(run, e)) for e in eps]
    conds = []
    for z in Z:
        conds += [str(c) for c in z["conds"] if str(c) not in conds]
    bconds = [c for c in conds if c.startswith(("bm_net_", "flat_"))]
    K, C_, Cb, nr = len(eps), len(conds), len(bconds), len(ti.CHEAP)
    D = dict(eps=np.array(eps), x=gap.train_x(np.array(eps)), conds=conds, bconds=bconds,
             value=np.full((K, C_, nr), np.nan), boot=np.full((K, C_, nr, NB), np.nan), has=np.zeros((K, C_), bool),
             bvalue=np.full((K, Cb, nr), np.nan), bboot=np.full((K, Cb, nr, NB), np.nan),
             right_T=np.zeros((K, C_, n, NV), bool), acc_goal=np.full((K, C_, NE, T), np.nan),
             netrev=np.zeros((K, n), bool), val=[], forward_s=[], peak=[])
    for s in SEEDS_POOL:
        for k_ in ("tv_med", "tv_q90", "tv_lt05", "tv_lt10", "share_far", "ent_flat"):
            D[f"{k_}_{s}"] = np.full(K, np.nan)
    D["ent_own"] = np.full(K, np.nan)
    rc = P["r"][:, 0] * R + P["r"][:, 1]
    for i, z in enumerate(Z):
        for j, c in enumerate(z["conds"]):
            a = conds.index(str(c))
            D["value"][i, a], D["boot"][i, a], D["has"][i, a] = z["value"][j], z["boot"][j], True
            D["right_T"][i, a], D["acc_goal"][i, a] = z["right_T"][j], z["acc_goal"][j]
        if "bayes_conds" in z:
            for j, c in enumerate(z["bayes_conds"]):
                a = bconds.index(str(c))
                D["bvalue"][i, a], D["bboot"][i, a] = z["bayes_value"][j], z["bayes_boot"][j]
        D["netrev"][i] = z["netrev"]
        D["val"].append(json.loads(str(z["validation"])))
        D["forward_s"].append(float(z["forward_s"]))
        D["peak"].append(float(z["peak_gpu_mb"]))
        D["ent_own"][i] = np.median(z["own_ent"])
        for s in SEEDS_POOL:
            if f"tv_{s}" in z:
                tv = z[f"tv_{s}"]
                D[f"tv_med_{s}"][i], D[f"tv_q90_{s}"][i] = np.median(tv), np.quantile(tv, .9)
                D[f"tv_lt05_{s}"][i], D[f"tv_lt10_{s}"][i] = (tv < .05).mean(), (tv < .1).mean()
                dc = z[f"bm_cell_{s}"].astype(np.int64)
                D[f"share_far_{s}"][i] = ((dc // R != (rc // R)[:, None]) & (dc % R != (rc % R)[:, None])).mean()
                D[f"ent_flat_{s}"][i] = np.median(z[f"ent_{s}"])
    _C[key] = D
    return D


def load_bm_null():
    if "BN" not in _C:
        _C["BN"] = _load(os.path.join(CTL, "bayes_bm_null.npz"))
    return _C["BN"]


def ctl_dep(cond, row="acc_goal_T", run="trained", ref="null"):
    """a row's dependence (intact - cut, its own units; w = 5) under a control condition, on the control pass's own
    intact run: net (K,) (NaN where not run) and net_boot (NB, K); Bayes's reference: bayes (K,), bayes_boot (NB, K):
    Bayes under the identical substitution (constant over checkpoints), except for the belief-matched swap, whose
    reference is (ref 'null', the default) the matched null: Bayes swapped with donors matched on ITS OWN marginals from
    the same pool (bayes_bm_null; checkpoint-independent), or (ref 'donors') Bayes under the network's own donors (what
    those histories carry for an exact observer; not a matched reference: the donors are chosen by the network's
    read-out); excess = net - bayes with ex_boot; the levels (intact, cut; Bayes intact, cut); x"""
    D, BC = load_ctl(run), load_bayes_ctl()
    k = list(ti.CHEAP).index(row)
    c, i0 = D["conds"].index(cond), D["conds"].index("intact")
    K = len(D["x"])
    net = D["value"][:, i0, k] - D["value"][:, c, k]
    nb = (D["boot"][:, i0, k] - D["boot"][:, c, k]).T
    bi, bib = BC["intact_value"][k], BC["intact_boot"][k]
    if cond.startswith("bm_net_") and ref == "null":
        N = load_bm_null()
        j = [int(s) for s in N["seeds"]].index(int(cond.split("_")[2]))
        bcut, bcb = np.full(K, N["value"][j, k]), np.repeat(N["boot"][j, k][:, None], K, 1)
    elif cond in D["bconds"]:
        j = D["bconds"].index(cond)
        bcut, bcb = D["bvalue"][:, j, k], D["bboot"][:, j, k].T
    elif cond in CTL_FIXED:
        j = list(BC["conds"]).index(cond)
        bcut, bcb = np.full(K, BC["value"][j, k]), np.repeat(BC["boot"][j, k][:, None], K, 1)
    else:                                                                              # the engine's three sources
        B = load_bayes()
        kk, cc = ROWS.index(row), cix("w5_" + cond)
        bcut, bcb = np.full(K, B["value"][cc, kk]), np.repeat(B["boot"][cc, kk][:, None], K, 1)
    bv, bb = bi - bcut, bib[:, None] - bcb
    return dict(x=D["x"], net=net, net_boot=nb, bayes=bv, bayes_boot=bb, ex=net - bv, ex_boot=nb - bb,
                intact=D["value"][:, i0, k], cut=D["value"][:, c, k], cut_boot=D["boot"][:, c, k].T,
                intact_boot=D["boot"][:, i0, k].T, bayes_intact=bi, bayes_cut=bcut, bayes_cut_boot=bcb,
                naive=float(ti.refs()["naive"][row]["value"]))


def _ivs(v, b, f="+.3f"):
    """'v [lo, hi]' with the 5-95% of the resamples b"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(b, [5, 95])
    return f"{v:{f}} [{lo:{f}}, {hi:{f}}]"


def ctl_group(name, k, row="acc_goal_T"):
    """the conditions of a CTL_GROUPS entry at control checkpoint index k: per-seed network and Bayes dependence, their
    mean and range over seeds"""
    conds = dict(CTL_GROUPS)[name]
    d = [ctl_dep(c, row) for c in conds]
    nv = np.array([x["net"][k] for x in d])
    bv = np.array([x["bayes"][k] for x in d])
    return dict(net=nv, bayes=bv, net_mean=np.nanmean(nv), bayes_mean=np.nanmean(bv), d=d, conds=conds)


def ctl_onset(cond, row="acc_goal_T", share=.5):
    """the control's 'need onset': the training episode after which the network's dependence stays above share x
    Bayes's dependence under the identical substitution (gap.crossing on the checkpoints where the control ran)"""
    d = ctl_dep(cond, row)
    ok = np.isfinite(d["net"])
    x = d["x"][ok]
    sg = np.sign(np.nanmean(d["bayes"])) or 1.
    y = sg * (d["net"][ok] - share * d["bayes"][ok])
    yb = sg * (d["net_boot"][:, ok] - share * d["bayes_boot"][:, ok])
    cb = gap.crossing(x, yb, 0.)
    lo, hi, sh = gap.interval(cb)
    return dict(p=gap.crossing(x, y, 0.), lo=lo, hi=hi, share=sh, boot=cb)


def uncut_steps(run="trained"):
    """the intact network's goal accuracy at every step (K, T) with resamples (NB, K, T), and Bayes's and naive's (T,)
    with resamples (NB, T), from the harvest's per-(environment, step) sums (200 probes per environment and step)"""
    D, B = load_run(run), load_bayes()
    C = gap.boot_counts()
    den = NE * PER_ENV
    S = D["acc_goal"][:, 0]                                                            # (K, NE, T)
    out = dict(x=D["x"], net=S.sum(1) / den, net_boot=np.einsum("be,ket->bkt", C, S) / (C.sum(1) * PER_ENV)[:, None, None])
    for nm, A_ in (("bayes", B["acc_goal"][0]), ("naive", B["naive_acc_goal"])):
        out[nm] = A_.sum(0) / den
        out[nm + "_boot"] = (C @ A_) / (C.sum(1) * PER_ENV)[:, None]
    return out


def uncut_reach(s0=5, run="trained"):
    """the step of the episode at which the uncut network first reaches Bayes's goal accuracy at step s0 (linear in
    steps between the two steps around the crossing; numbers_fixes' rule), per checkpoint (K,), with its resamples
    (NB, K) (Bayes's level recomputed in every resample, paired); NaN = never within the 30 steps"""
    u = uncut_steps(run)

    def reach(y, lev):                                                                 # y (M, T), lev (M,)
        past = y >= lev[:, None]
        has = past.any(1)
        j = np.argmax(past, 1)
        y0 = y[np.arange(len(y)), np.maximum(j - 1, 0)]
        y1 = y[np.arange(len(y)), j]
        with np.errstate(invalid="ignore", divide="ignore"):
            st = np.where(j == 0, 1., j + (lev - y0) / (y1 - y0))
        return np.where(has, st, np.nan)
    K = len(u["x"])
    v = reach(u["net"], np.full(K, u["bayes"][s0 - 1]))
    nb = u["net_boot"].shape[0]
    b = reach(u["net_boot"].reshape(nb * K, T), np.repeat(u["bayes_boot"][:, s0 - 1], K)).reshape(nb, K)
    return dict(x=u["x"], step=v, boot=b, level=u["bayes"][s0 - 1])


def netrev_split(k):
    """at control checkpoint index k: the w5 start cut's goal answers lost on the network-defined revision probes
    (net_revisions, right at T) vs the other probes right at T: raw, and matched on the intact goal margin at T (deciles
    of the right-at-T probes, weighted by the revision probes per decile: the check's rule), with resamples; Bayes's loss
    under its own w5 start cut on the same probe sets"""
    Dc, D, B, P = load_ctl(), load_run("trained"), load_bayes(), _probes_full()
    kk = list(D["eps"]).index(int(Dc["eps"][k]))
    ar, g, env = np.arange(len(P["goal"])), P["goal"], P["env"]
    C = gap.boot_counts()
    rt0 = D["right_T"][kk, 0][ar, g]
    lost = rt0.astype(np.float64) - D["right_T"][kk, cix("w5_start")][ar, g]
    blost = B["right_T"][0][ar, g].astype(np.float64) - B["right_T"][cix("w5_start")][ar, g]
    rev, oth = Dc["netrev"][k] & rt0, ~Dc["netrev"][k] & rt0
    mg = D["mT"][kk, 0][ar, g]
    edges = np.quantile(mg[rt0], np.linspace(0, 1, 11))
    bi = np.clip(np.searchsorted(edges, mg, side="right") - 1, 0, 9)

    def sums(v, m):
        idx = env * 10 + bi
        return (np.bincount(idx, v * m, NE * 10).reshape(NE, 10), np.bincount(idx, m.astype(np.float64),
                                                                              NE * 10).reshape(NE, 10))
    Sr, Nr = sums(lost, rev)
    So, No = sums(lost, oth)

    def matched(Sr_, Nr_, So_, No_):
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = (Nr_ > 0) & (No_ > 0)
            d_ = np.where(ok, Sr_ / np.where(ok, Nr_, 1) - So_ / np.where(ok, No_, 1), 0.)
            w_ = np.where(ok, Nr_, 0.)
            return (w_ * d_).sum(-1) / w_.sum(-1)
    full = matched(Sr.sum(0), Nr.sum(0), So.sum(0), No.sum(0))
    boot = matched(C @ Sr, C @ Nr, C @ So, C @ No)
    out = dict(n_rev=int(rev.sum()), n_oth=int(oth.sum()), share=float(rev.sum() / rt0.sum()), matched=float(full),
               matched_boot=boot)
    for nm, m in (("rev", rev), ("oth", oth)):
        s_, n_ = np.bincount(env, lost * m, NE), np.bincount(env, m.astype(np.float64), NE)
        sb_ = np.bincount(env, blost * m, NE)
        out[nm] = s_.sum() / n_.sum()
        out[nm + "_boot"] = (C @ s_) / (C @ n_)
        out["bayes_" + nm] = sb_.sum() / n_.sum()
    return out


def fifth_excess(cond="w5_start"):
    """the competitor's gain per variable-step by entanglement fifth (where pass): the network's minus Bayes's under
    the same cut (K, 5) with resamples, and each observer's most-minus-least entangled difference (additive)"""
    m, b = where_measure("trained", cond, "f"), where_measure("bayes", cond, "f")
    return dict(x=m["x"], net=m["value"], bayes=b["value"], ex=m["value"] - b["value"][None],
                ex_boot=m["boot"] - b["boot"][:, None, :], d51=m["value"][:, 4] - m["value"][:, 0],
                d51_boot=m["boot"][:, :, 4] - m["boot"][:, :, 0], bd51=b["value"][4] - b["value"][0],
                bd51_boot=b["boot"][:, 4] - b["boot"][:, 0])


def q5_contrast(cond, run="trained"):
    """memory use at the most entangled fifth relative to the other four, per step-matched variable-step (every fifth
    holds 4,000 variable-steps per step): u = err_cut - err_intact (td_entanglement_memcut's local update error; > 0:
    the state older than w helps the step's update), contrast = u(q5) - mean u(q1..q4), the network's (K,) with
    resamples and its stays-above-0 crossing; Bayes's under the same cut (its intact error is 0)"""
    m, b = step_measure(run, cond, "err"), step_measure("bayes", cond, "err")
    y = m["value"][:, 4] - m["value"][:, :4].mean(-1)
    yb = m["boot"][:, :, 4] - m["boot"][:, :, :4].mean(-1)
    cb = gap.crossing(m["x"], yb, 0.)
    lo, hi, sh = gap.interval(cb)
    return dict(x=m["x"], y=y, yb=yb, u5=m["value"][:, 4], cross=dict(p=gap.crossing(m["x"], y, 0.), lo=lo, hi=hi,
                                                                     share=sh, boot=cb),
                bayes=b["value"][4] - b["value"][:4].mean(), bayes_u5=b["value"][4])


def numbers():
    """every number of the docstring (prints)"""
    B = load_bayes()
    M = family_marks()
    print("== CHECKS (OUT/checks.json)")
    with open(os.path.join(OUT, "checks.json")) as f:
        print(json.dumps(json.load(f), default=float))
    Dt, De = load_run("trained"), load_run("echo")
    print(f"\n== DATA: Trained RNN {len(Dt['x'])} checkpoints ({Dt['x'][0]:,.0f}-{Dt['x'][-1]:,.0f}); ESN {len(De['x'])}")
    print("\n== BAYES under the same cut (goal accuracy at T; both right; loss vs intact)")
    for c in CONDS:
        print(f"  {c:13s} goal {B['value'][cix(c), 0]:.4f} (loss {B['value'][0, 0] - B['value'][cix(c), 0]:+.4f}), "
              f"both {B['value'][cix(c), 1]:.4f}")
    print(f"  naive goal {ti.refs()['naive']['acc_goal_T']['value']:.4f}")
    print("\n== ANSWERS: goal accuracy at T, dependence intact - cut [5-95%], network vs Bayes's (same cut)")
    for run, D in (("trained", Dt), ("echo", De)):
        for w in WS:
            for s_ in SOURCES:
                d = dep(run, f"w{w}_{s_}", "acc_goal_T")
                ks = sorted(set(_k(D["x"], xv) for xv in KEY_X))
                print(f"  {run:7s} w{w:2d} {s_:9s} Bayes {d['bayes']:+.3f} [{d['bayes_lo']:+.3f}, {d['bayes_hi']:+.3f}] | "
                      + "; ".join(f"{D['x'][k]:,.0f}: {_iv(d['net'][k], d['net_boot'][:, k])}" for k in ks))
    print("\n== LEVELS (goal accuracy at T): intact / cut, network; Bayes intact / cut")
    for w in WS:
        for s_ in SOURCES:
            d = dep("trained", f"w{w}_{s_}", "acc_goal_T")
            print(f"  w{w:2d} {s_:9s} Bayes {d['bayes_intact']:.3f} / {d['bayes_cut']:.3f} | " + "; ".join(
                f"{Dt['x'][k]:,.0f}: {d['intact'][k]:.3f} / {d['cut'][k]:.3f}" for k in [_k(Dt["x"], xv) for xv in KEY_X]))
    print("\n== TIMING of the excess (network - Bayes dependence, goal accuracy at T)")
    acc = M["acc_goal_T"]
    late = {k: M[k]["c50"] for k in ("emass_slope", "held_two", "held_one", "held_joint", "excess", "G", "dir_slope",
                                     "band4")}
    on = {k: M[k]["c10"] for k in late}
    for w in WS:
        for s_ in SOURCES:
            tm = timing("trained", f"w{w}_{s_}")
            line = (f"  w{w:2d} {s_:9s} peak {tm['peak']:+.3f} [{tm['peak_lo']:+.3f}, {tm['peak_hi']:+.3f}] at "
                    f"{tm['peak_x']:,.0f}; rise {_cx(tm['rise'])}; fall {_cx(tm['fall'])}")
            if tm["peak"] > .02:
                line += (f"\n      rise vs passing naive {_cx(acc['c0'])}: {_order(tm['rise'], acc['c0'])}; vs accuracy "
                         f"c50 {_cx(acc['c50'])}: {_order(tm['rise'], acc['c50'])}"
                         f"\n      fall vs accuracy c50: {_order(tm['fall'], acc['c50'])}; vs every holding / coupling "
                         "c50: " + ", ".join(f"{k} {_cx(v)} {_order(tm['fall'], v)}" for k, v in late.items())
                         + "\n      fall vs holding / coupling c10: " + ", ".join(f"{k} {_order(tm['fall'], v)}"
                                                                                for k, v in on.items()))
            print(line)
    print("\n== FAMILY ROWS (w = 5): dependence intact - cut, network [5-95%] vs Bayes's (same cut)")
    for row in ("acc_goal_T", "acc_both_T", "far_T", "kl_goal_mean", "split_grad", "split_grad_matched", "fold_idx",
                "emass_slope", "held_two", "held_one", "held_joint", "excess", "G", "band4", "dir_slope"):
        mk = M.get(row)
        mtxt = f" (c10 {_cx(mk['c10'])}, c50 {_cx(mk['c50'])})" if mk else ""
        print(f"  {row}{mtxt}")
        for s_ in SOURCES:
            d = dep("trained", f"w5_{s_}", row)
            ks = [_k(Dt["x"], xv) for xv in KEY_X]
            print(f"    {s_:9s} Bayes {d['bayes']:+.4f} [{d['bayes_lo']:+.4f}, {d['bayes_hi']:+.4f}] (Bayes {d['bayes_intact']:.4f}"
                  f" -> {d['bayes_cut']:.4f}; naive {d['naive']:.4f}) | " + "; ".join(
                      f"{Dt['x'][k]:,.0f}: {d['intact'][k]:.4f}->{d['cut'][k]:.4f} {_iv(d['net'][k], d['net_boot'][:, k])}"
                      for k in ks))
    print("\n== FAMILY TRANSIENTS: the excess (network - Bayes dependence, w = 5) per row: peak [5-95%] at; rise; fall "
          "(far errors: sign flipped so that more far errors = positive)")
    for row in ("acc_goal_T", "acc_both_T", "far_T", "fold_idx", "split_grad", "emass_slope", "held_two", "held_joint",
                "held_one", "excess", "G", "band4", "dir_slope"):
        for s_ in ("start", "same_cell"):
            d = dep("trained", f"w5_{s_}", row)
            sg = -1. if row == "far_T" else 1.
            rf = rise_fall(d["x"], sg * d["ex"], sg * d["ex_boot"])
            print(f"  {row:12s} {s_:9s} peak {rf['peak']:+.4f} [{rf['peak_lo']:+.4f}, {rf['peak_hi']:+.4f}] at "
                  f"{rf['peak_x']:,.0f}; rise {rf['rise']:,.0f} [{rf['rise_lo']:,.0f}-{rf['rise_hi']:,.0f}] "
                  f"({rf['rise_share']:.2f}); fall {rf['fall']:,.0f} [{rf['fall_lo']:,.0f}-{rf['fall_hi']:,.0f}] "
                  f"({rf['fall_share']:.2f})")
    gain5 = (B["value"][cix("w5_start"), 0] - ti.refs()["naive"]["acc_goal_T"]["value"]) / (
        B["value"][0, 0] - ti.refs()["naive"]["acc_goal_T"]["value"])
    print(f"  Bayes's 5-step window from the start keeps {gain5:.2f} of its gain over naive (goal accuracy at T)")
    print("\n== NEED: after which the network's dependence stays above half of Bayes's (same cut, w = 5), vs the row's own"
          " c10 / c50 (paired)")
    for row in ("acc_goal_T", "far_T", "split_grad", "emass_slope", "held_two", "held_one", "excess", "G", "band4",
                "dir_slope"):
        mk = M.get(row)
        for s_ in SOURCES:
            d = dep("trained", f"w5_{s_}", row)
            if abs(d["bayes"]) < 1e-9 or d["bayes_lo"] * d["bayes_hi"] <= 0:
                print(f"  {row:12s} {s_:9s} Bayes's dependence not resolved ({d['bayes']:+.4f} [{d['bayes_lo']:+.4f}, "
                      f"{d['bayes_hi']:+.4f}])")
                continue
            nd = need_onset("trained", f"w5_{s_}", row)
            txt = f"  {row:12s} {s_:9s} {_cx(nd)}"
            if mk:
                txt += f"; vs c10 {_cx(mk['c10'])}: {_order(nd, mk['c10'])}; vs c50 {_cx(mk['c50'])}: {_order(nd, mk['c50'])}"
            print(txt)
    print("\n== WHERE: accumulation (goal and context answers at T, per variable): full cut vs the sum of single-step swaps")
    for s_ in SOURCES:
        for w in WS:
            c = cix(f"w{w}_{s_}")
            full = Dt["right_T"][:, 0].mean((1, 2)) - Dt["right_T"][:, c].mean((1, 2))
            ss = Dt["flip_f"][:, c].sum((1, 2, 3)) / (NV * NE * PER_ENV)
            bf = B["right_T"][0].mean() - B["right_T"][c].mean()
            bs = B["flip_f"][c].sum() / (NV * NE * PER_ENV)
            print(f"  w{w:2d} {s_:9s} Bayes full {bf:+.4f}, sum of single steps {bs:+.4f} | " + "; ".join(
                f"{Dt['x'][k]:,.0f}: {full[k]:+.4f} vs {ss[k]:+.4f}" for k in [_k(Dt["x"], xv) for xv in KEY_X]))
    print("\n== WHERE (main): the cut's winning competitor's gain on the true value, nats per variable-step after the window "
          "[5-95%], by entanglement fifth (q1 least ... q5 most) and by Bayes's step group (revision / settled / other); "
          "all variable-steps, then only the variables whose answer the cut loses")
    W = load_where("trained")
    WK = [int(np.argmin(np.abs(W["x"] - xv))) for xv in (400, 960, 1680, 2080, 4400, 11200, 39996)]
    for cond in ("w5_start", "w5_same_cell", "w5_random", "w2_start", "w10_start"):
        for by in ("f", "g"):
            for lost in (False, True):
                m, b = where_measure("trained", cond, by, lost), where_measure("bayes", cond, by, lost)
                shb = b["total"] / b["total"].sum()
                print(f"  {cond:13s} {by} {'lost' if lost else 'all ':4s} Bayes " + " ".join(f"{v:+.4f}" for v in b["value"])
                      + " (shares " + " ".join(f"{v:.2f}" for v in shb) + ")")
                for k in WK:
                    lo, hi = np.nanpercentile(m["boot"][:, k], [5, 95], 0)
                    sh = m["total"][k] / m["total"][k].sum()
                    print(f"      {W['x'][k]:6,.0f}: " + " ".join(f"{v:+.4f} [{a:+.4f}, {c:+.4f}]" for v, a, c in
                                                          zip(m["value"][k], lo, hi))
                          + "  shares " + " ".join(f"{v:.2f}" for v in sh))
    print("  answers at T identical between the 'where' pass and the harvest at every checkpoint:", bool(W["same"].all()),
          f"({len(W['x'])} Trained RNN checkpoints)")
    print("\n== MARGINS: the true value's margin at T lost by the cut (mean / median, nats), network vs Bayes; each "
          "observer's changes applied to the other's intact margins (answers lost)")
    for cond in ("w5_start", "w5_same_cell", "w5_random", "w2_start", "w10_start"):
        for xv in (960, 1680, 4400, 39996):
            k = _k(Dt["x"], xv)
            g = margin_change("trained", cond, k)
            print(f"  {cond:13s} {Dt['x'][k]:6,.0f}: network {g['net_mean']:+.3f} / {g['net_median']:+.3f} (answers lost "
                  f"{g['net_lost']:+.3f}), Bayes {g['bayes_mean']:+.3f} / {g['bayes_median']:+.3f} ({g['bayes_lost']:+.3f}); "
                  f"network's changes on Bayes's margins {g['net_on_bayes']:+.3f}, Bayes's on the network's "
                  f"{g['bayes_on_net']:+.3f}")
    print("\n== WHERE (secondary): per var-step after the window, by entanglement fifth (q1 least ... q5 most)")
    for kind in ("dd", "flip", "dm", "ev", "err", "sz", "gain"):
        for s_ in SOURCES:
            m, b = step_measure("trained", f"w5_{s_}", kind), step_measure("bayes", f"w5_{s_}", kind)
            print(f"  {kind:4s} w5 {s_:9s} Bayes " + " ".join(f"{v:+.4f}" for v in b["value"]) + " | " + "; ".join(
                f"{Dt['x'][k]:,.0f}: " + " ".join(f"{v:+.4f}" for v in m["value"][k])
                for k in [_k(Dt["x"], xv) for xv in KEY_X]))
    print("\n== WHERE (secondary): evidence for the true value lost by the w5 start cut, settled minus revision steps "
          "within each step (steps 6-30, weighted by the smaller count), network vs Bayes")
    cgs = B["cnt_g"].sum(0)
    ts_ = np.arange(5, T)
    wgt = np.minimum(cgs[ts_, 0], cgs[ts_, 1])
    nb_ = (B["ev_g"][0] - B["ev_g"][cix("w5_start")]).sum(0)[ts_] / cgs[ts_]
    db_ = ((nb_[:, 1] - nb_[:, 0]) * wgt).sum() / wgt.sum()
    for k in [_k(Dt["x"], xv) for xv in KEY_X]:
        nn_ = (Dt["ev_g"][k, 0] - Dt["ev_g"][k, cix("w5_start")]).sum(0)[ts_] / cgs[ts_]
        print(f"  {Dt['x'][k]:7,.0f}: network {((nn_[:, 1] - nn_[:, 0]) * wgt).sum() / wgt.sum():+.4f}  (Bayes {db_:+.4f})")
    print("\n== WHERE: per var-step by Bayes's step group (revision / settled / other)")
    for kind in ("dd", "flip", "dm", "ev"):
        for s_ in SOURCES:
            m, b = step_measure("trained", f"w5_{s_}", kind, by="g"), step_measure("bayes", f"w5_{s_}", kind, by="g")
            print(f"  {kind:4s} w5 {s_:9s} Bayes " + " ".join(f"{v:+.4f}" for v in b["value"]) + " | " + "; ".join(
                f"{Dt['x'][k]:,.0f}: " + " ".join(f"{v:+.4f}" for v in m["value"][k])
                for k in [_k(Dt["x"], xv) for xv in KEY_X]))
    print("\n== WHERE: probe families (goal variable): accuracy at T lost, w5, network [5-95%] vs Bayes")
    fams = {"Bayes revises": B["fam"] == 0, "Bayes settled": B["fam"] == 1, "neither": B["fam"] == 2,
            "naive far error": B["naive_far"]}
    for s_ in SOURCES:
        for nm, m in fams.items():
            f = family_dep("trained", f"w5_{s_}", m)
            print(f"  {s_:9s} {nm:15s} n {f['n']:5d} Bayes {f['bayes']:+.3f} | " + "; ".join(
                f"{Dt['x'][k]:,.0f}: {_iv(f['net'][k], f['net_boot'][:, k])}" for k in [_k(Dt["x"], xv) for xv in KEY_X]))
    print("\n== ECHO: e_t ~ a N_t + b c_(t-1) per fifth (steps 2-30), intact")
    for run in ("naive", "bayes"):
        e = echo_coef(run)
        print(f"  {run:7s} a " + " ".join(f"{v:.3f}" for v in e["a"]) + "  b " + " ".join(f"{v:+.4f}" for v in e["b"]))
    for run, D in (("trained", Dt), ("echo", De)):
        e = echo_coef(run)
        for k in sorted(set(_k(D["x"], xv) for xv in KEY_X)):
            print(f"  {run:7s} {D['x'][k]:7,.0f} a " + " ".join(f"{v:.3f}" for v in e["a"][k]) + "  b "
                  + " ".join(f"{v:+.4f}" for v in e["b"][k]) + f"  (b, all fifths [5-95%]: q5 "
                  f"{_iv(e['b'][k, 4], e['b_boot'][:, k, 4])})")
    print("\n== PER-STEP accuracy (goal, cumulative belief) lost by the w5 cut at steps 6 / 10 / 15 / 20 / 30")
    for s_ in SOURCES:
        a0, a1 = per_step_acc("trained", "intact"), per_step_acc("trained", f"w5_{s_}")
        b0, b1 = per_step_acc("bayes", "intact"), per_step_acc("bayes", f"w5_{s_}")
        st = (5, 9, 14, 19, 29)
        print(f"  {s_:9s} Bayes " + " ".join(f"{b0[t] - b1[t]:+.3f}" for t in st) + " | " + "; ".join(
            f"{Dt['x'][k]:,.0f}: " + " ".join(f"{a0[k, t] - a1[k, t]:+.3f}" for t in st)
            for k in [_k(Dt["x"], xv) for xv in KEY_X]))
    numbers_fixes()


CTL_KEY_X = (400, 880, 1360, 1680, 2080, 3000, 4400, 6400, 11200, 39996)


def numbers_fixes():
    """the numbers added by the 11-02 fixes (controls, levels, onsets, uncut build-up, network-defined revisions, the
    per-fifth excess, the q5 contrast, the corrected ranges)"""
    Dt = load_run("trained")
    naive = ti.refs()["naive"]["acc_goal_T"]["value"]
    print("\n== START CUT: what the network keeps of its gain over naive (goal accuracy at T, w = 5): cut - naive of intact"
          " - naive; Bayes's 5-step window: the same")
    d = dep("trained", "w5_start", "acc_goal_T")
    print(f"  Bayes: cut {d['bayes_cut'] - naive:+.3f} of intact {d['bayes_intact'] - naive:+.3f}")
    for xv in KEY_X:
        k = _k(Dt["x"], xv)
        print(f"  {Dt['x'][k]:7,.0f}: network cut {d['cut'][k] - naive:+.3f} of intact {d['intact'][k] - naive:+.3f}")
    d5 = dep("trained", "w5_start", "acc_goal_T")
    k = int(np.nanargmax(d5["net"]))
    print(f"  largest single-checkpoint loss (w5 start): {d5['net'][k]:.3f} at {Dt['x'][k]:,.0f}")
    print("\n== w = 10 excess (start): 3-checkpoint running-mean peak and the largest single checkpoint")
    tm = timing("trained", "w10_start")
    d10 = dep("trained", "w10_start", "acc_goal_T")
    k = int(np.nanargmax(d10["ex"]))
    print(f"  running-mean peak {tm['peak']:+.4f} [{tm['peak_lo']:+.4f}, {tm['peak_hi']:+.4f}] at {tm['peak_x']:,.0f}; "
          f"largest checkpoint {Dt['x'][k]:,.0f}: {_iv(d10['ex'][k], d10['ex_boot'][:, k])}")
    print("\n== ESN echo coefficient b (all ESN checkpoints, all fifths) and the Trained RNN's at 1,680")
    e = echo_coef("echo")
    De = load_run("echo")
    print(f"  ESN b range {np.nanmin(e['b']):+.4f} .. {np.nanmax(e['b']):+.4f} over {len(De['x'])} checkpoints "
          f"({De['x'][0]:,.0f}-{De['x'][-1]:,.0f}) x 5 fifths; most entangled fifth {np.nanmin(e['b'][:, 4]):+.4f} .. "
          f"{np.nanmax(e['b'][:, 4]):+.4f}; from 400 on: {np.nanmin(e['b'][De['x'] >= 400]):+.4f} .. "
          f"{np.nanmax(e['b'][De['x'] >= 400]):+.4f}")
    et = echo_coef("trained")
    k = _k(Dt["x"], 1680)
    print(f"  Trained RNN at {Dt['x'][k]:,.0f}: b {np.min(et['b'][k]):+.4f} .. {np.max(et['b'][k]):+.4f}; Bayes "
          f"{np.min(echo_coef('bayes')['b']):+.4f} .. {np.max(echo_coef('bayes')['b']):+.4f}")
    print("\n== COUPLING row by row under the start cut (w = 5): dependence network vs Bayes, excess peak (see FAMILY "
          "TRANSIENTS), need onsets (see NEED)")
    for row in ("G", "excess", "band4", "dir_slope"):
        d = dep("trained", "w5_start", row)
        rf = rise_fall(d["x"], d["ex"], d["ex_boot"])
        ks = [_k(Dt["x"], xv) for xv in (3000, 4400, 6400, 7600, 39996)]
        print(f"  {row:10s} Bayes {d['bayes']:+.4f} | " + "; ".join(f"{Dt['x'][k]:,.0f}: {d['net'][k]:+.4f}" for k in ks)
              + f" | excess peak {rf['peak']:+.4f} [{rf['peak_lo']:+.4f}, {rf['peak_hi']:+.4f}] at {rf['peak_x']:,.0f}")
    print("\n== UNCUT build-up (intact network): goal accuracy at step t minus naive's at step t [5-95%], vs Bayes's")
    u = uncut_steps()
    st = (2, 3, 5, 8, 10, 15, 30)
    print("  Bayes " + " ".join(f"t{t}: {u['bayes'][t - 1] - u['naive'][t - 1]:+.3f}" for t in st))
    for xv in (400, 880, 1360, 1680, 2080, 3000, 4400, 11200, 39996):
        k = _k(u["x"], xv)
        print(f"  {u['x'][k]:7,.0f}: " + " ".join(
            f"t{t}: {_ivs(u['net'][k, t - 1] - u['naive'][t - 1], u['net_boot'][:, k, t - 1] - u['naive_boot'][:, t - 1])}"
            for t in (5, 30)) + "  | " + " ".join(f"t{t}: {u['net'][k, t - 1] - u['naive'][t - 1]:+.3f}" for t in st)
            + "  | share of Bayes's gain (supplementary; small denominators early): " + " ".join(
                f"t{t}: {(u['net'][k, t - 1] - u['naive'][t - 1]) / (u['bayes'][t - 1] - u['naive'][t - 1]):+.2f}"
                for t in (5, 30)))
    print("  step at which the uncut network first reaches Bayes's accuracy at step 5 / 8 / 10 (linear in steps):")
    for xv in (880, 1360, 1680, 2080, 3000, 4400, 11200, 39996):
        k = _k(u["x"], xv)
        txt = []
        for s0 in (5, 8, 10):
            lev = u["bayes"][s0 - 1]
            yy = u["net"][k]
            j = np.flatnonzero(yy >= lev)
            if not len(j):
                txt.append(f"Bayes t{s0} never")
                continue
            j = j[0]
            stp = 1. if j == 0 else j + (lev - yy[j - 1]) / (yy[j] - yy[j - 1])    # step j + 1 is index j
            txt.append(f"Bayes t{s0} at step {stp:.1f}")
        print(f"    {u['x'][k]:7,.0f}: " + "; ".join(txt))
    print("\n== WHERE: excess per entanglement fifth (network - Bayes competitor's gain, nats per variable-step after the "
          "window, w5 start) and each observer's most-minus-least entangled difference")
    fe = fifth_excess()
    print("  Bayes per fifth " + " ".join(f"{v:+.4f}" for v in fe["bayes"]) + "; Bayes q5 - q1 "
          f"{_ivs(fe['bd51'], fe['bd51_boot'], '+.4f')}")
    lo5 = np.nanpercentile(fe["ex_boot"], 5, 0)                                       # (K, 5)
    print(f"  every fifth's excess resolved above 0 at every checkpoint: {bool((lo5 > 0).all())} (smallest 5% bound "
          f"{lo5.min():+.4f}, fifth {np.unravel_index(lo5.argmin(), lo5.shape)[1] + 1} at {fe['x'][np.unravel_index(lo5.argmin(), lo5.shape)[0]]:,.0f})")
    for k in range(len(fe["x"])):
        print(f"  {fe['x'][k]:7,.0f}: excess " + " ".join(_ivs(fe["ex"][k, q], fe["ex_boot"][:, k, q], "+.4f")
                                                     for q in (0, 4))
              + " (least, most) | all " + " ".join(f"{v:+.4f}" for v in fe["ex"][k])
              + f" | network q5 - q1 {_ivs(fe['d51'][k], fe['d51_boot'][:, k], '+.4f')}; excess q5 - q1 "
              f"{_ivs(fe['d51'][k] - fe['bd51'], fe['d51_boot'][:, k] - fe['bd51_boot'], '+.4f')}")
    print("\n== Q5 CONTRAST (memory use at the most entangled fifth minus the mean of the other four, local update error,"
          " step-matched): network [5-95%], stays-above-0 crossing; Bayes's under the same cut")
    for cond in ("w2_start", "w2_same_cell", "w2_random", "w5_start", "w5_same_cell", "w5_random"):
        q = q5_contrast(cond)
        ks = [_k(q["x"], xv) for xv in (400, 720, 880, 1360, 1680, 2080, 4400, 39996)]
        print(f"  {cond:13s} Bayes {q['bayes']:+.4f} (u q5 {q['bayes_u5']:+.4f}); crossing {_cx(q['cross'])} | "
              + "; ".join(f"{q['x'][k]:,.0f}: {_ivs(q['y'][k], q['yb'][:, k], '+.4f')}" for k in ks))
    Dc = load_ctl()
    if Dc is None:
        print("\n(no control caches)")
        return
    print(f"\n== CONTROLS (w = 5): {len(Dc['x'])} Trained RNN checkpoints; the beyond-belief conditions at "
          f"{int(Dc['has'][:, Dc['conds'].index('bm_net_' + str(SEEDS_POOL[0]))].sum())}")
    tot = {k: sum(v[k] for v in Dc["val"]) for k in Dc["val"][0]}
    n_ = len(_probes_full()["env"])
    print(f"  validation: answers at T differing from the harvest, summed over checkpoints: {tot} (of "
          f"{len(Dc['x']) * n_ * NV:,} each); probe-windows per pool seed and checkpoint {n_ * CA:,}")
    print(f"  forward median {np.median(Dc['forward_s']):.0f} s (max {np.max(Dc['forward_s']):.0f} s); peak GPU "
          f"{np.max(Dc['peak']):.0f} MB; Bayes's shuffled histories vs intact: {str(load_bayes_ctl()['shuffle_vs_intact'])}")
    print("\n== CONTROLS: goal accuracy at T lost (intact - cut), network [5-95%] | Bayes under the identical substitution "
          "(belief-matched: its matched null; Bayes under the network's donors in MATCHED NULL below);"
          " per seed, then mean (range over seeds)")
    for name, conds in CTL_GROUPS:
        print(f"  {name}")
        for xv in CTL_KEY_X:
            k = _k(Dc["x"], xv)
            g = ctl_group(name, k)
            if not np.isfinite(g["net"]).any():
                continue
            per = "; ".join(f"{_ivs(dd['net'][k], dd['net_boot'][:, k])} | {dd['bayes'][k]:+.3f}" for dd in g["d"])
            rng_ = f" mean {g['net_mean']:+.3f} ({np.nanmin(g['net']):+.3f} .. {np.nanmax(g['net']):+.3f}) | Bayes " \
                   f"{g['bayes_mean']:+.3f} ({np.nanmin(g['bayes']):+.3f} .. {np.nanmax(g['bayes']):+.3f})" \
                if len(conds) > 1 else ""
            print(f"    {Dc['x'][k]:7,.0f}: {per}{rng_}")
    print("\n== CONTROLS: levels (goal accuracy at T) at 880 / 1,680 / 4,400 / 39,996: network cut | Bayes cut "
          f"(naive {naive:.3f}; belief-matched: the matched null)")
    for name, conds in CTL_GROUPS:
        dd = [ctl_dep(c) for c in conds]
        out = []
        for xv in (880, 1680, 4400, 39996):
            k = _k(Dc["x"], xv)
            out.append(f"{Dc['x'][k]:,.0f}: {np.nanmean([x['cut'][k] for x in dd]):.3f} | "
                       f"{np.nanmean([x['bayes_cut'][k] for x in dd]):.3f}")
        print(f"  {name:42s} " + "; ".join(out) + f"  (intact {dd[0]['intact'][_k(Dc['x'], 1680)]:.3f} at 1,680)")
    print("\n== CONTROLS: the start state minus the content-free / wrong mid-episode states (network's loss, paired "
          "resamples; > 0: the restart costs more than the mid-episode state)")
    ds = ctl_dep("start")
    for name in ("flat-belief other cell", "random cell, fresh at every window start", "environment-mean state",
                 "random cell, fixed per probe"):
        conds = dict(CTL_GROUPS)[name]
        out = []
        for xv in CTL_KEY_X:
            k = _k(Dc["x"], xv)
            vals = [(ds["net"][k] - ctl_dep(c)["net"][k], ds["net_boot"][:, k] - ctl_dep(c)["net_boot"][:, k])
                    for c in conds]
            vals = [v for v in vals if np.isfinite(v[0])]
            if not vals:
                continue
            v = np.mean([a for a, _ in vals])
            b = np.mean([bb for _, bb in vals], 0)
            out.append(f"{Dc['x'][k]:,.0f}: {_ivs(v, b)}")
        print(f"  {name:42s} " + "; ".join(out))
    print("\n== CONTROLS: excess over Bayes (network - Bayes loss) of the content-preserving swaps [5-95%] (belief-matched: "
          "network - matched null)")
    for name in ("same cell, other trajectory", "average of the two same-cell states", "shuffled own history",
                 "belief-matched other cell"):
        conds = dict(CTL_GROUPS)[name]
        out = []
        for xv in CTL_KEY_X:
            k = _k(Dc["x"], xv)
            vals = [(ctl_dep(c)["ex"][k], ctl_dep(c)["ex_boot"][:, k]) for c in conds]
            vals = [v for v in vals if np.isfinite(v[0])]
            if vals:
                out.append(f"{Dc['x'][k]:,.0f}: {_ivs(np.mean([a for a, _ in vals]), np.mean([b for _, b in vals], 0))}")
        print(f"  {name:38s} " + "; ".join(out))
    print("\n== CONTROLS: matching diagnostics of the pool donors (per pool seed; total variation = max over the two "
          "variables)")
    for s in SEEDS_POOL:
        ok = np.isfinite(Dc[f"tv_med_{s}"])
        for k in np.flatnonzero(ok):
            if Dc["x"][k] not in [Dc["x"][_k(Dc["x"], xv)] for xv in CTL_KEY_X]:
                continue
            print(f"  seed {s} {Dc['x'][k]:7,.0f}: TV median {Dc[f'tv_med_{s}'][k]:.3f}, 90% {Dc[f'tv_q90_{s}'][k]:.3f}, "
                  f"< .05 {Dc[f'tv_lt05_{s}'][k]:.2f}, < .10 {Dc[f'tv_lt10_{s}'][k]:.2f}; belief-matched donors sharing "
                  f"neither latent value {Dc[f'share_far_{s}'][k]:.2f}; flat donors' entropy median "
                  f"{Dc[f'ent_flat_{s}'][k]:.2f} vs own {Dc['ent_own'][k]:.2f} (uniform {2 * np.log(R):.2f})")
    print("\n== CONTROLS: family rows (cheap scores) lost under the controls at 1,680 / 4,400 / 6,400 / 39,996: network "
          "| Bayes (seed means; belief-matched: the matched null)")
    for row in ("held_two", "held_joint", "emass_slope", "excess", "G", "band4", "dir_slope", "far_T"):
        print(f"  {row}")
        for name in ("start",) + tuple(nm for nm, _ in CTL_GROUPS):
            conds = ("start",) if name == "start" else dict(CTL_GROUPS)[name]
            dd = [ctl_dep(c, row) for c in conds]
            out = []
            for xv in (1680, 4400, 6400, 39996):
                k = _k(Dc["x"], xv)
                nv = np.array([x["net"][k] for x in dd])
                if not np.isfinite(nv).any():
                    continue
                out.append(f"{Dc['x'][k]:,.0f}: {np.nanmean(nv):+.4f} | {np.nanmean([x['bayes'][k] for x in dd]):+.4f}")
            print(f"    {name:42s} " + "; ".join(out))
    print("\n== CONTROLS: 'need onsets' under the random-cell donors per seed and scheme (dependence above half of "
          "Bayes's under the identical substitution, for good; all 46 checkpoints)")
    M = family_marks()
    for row in ("acc_goal_T", "far_T", "emass_slope", "held_two", "excess", "G", "band4", "dir_slope"):
        mk = M.get(row)
        for name in ("random cell, fixed per probe", "random cell, fresh at every window start"):
            ons = [ctl_onset(c, row) for c in dict(CTL_GROUPS)[name]]
            ps = np.array([o["p"] for o in ons])
            txt = "; ".join(_cx(o) for o in ons)
            print(f"  {row:12s} {name:42s} {txt}  (range {np.nanmin(ps):,.0f}-{np.nanmax(ps):,.0f}; never in "
                  f"{int(np.isnan(ps).sum())})" + (f"; c10 {_cx(mk['c10'])}, c50 {_cx(mk['c50'])}" if mk else ""))
    print("\n== NETWORK-DEFINED REVISIONS (goal variable; q > .6 on a value that is not the network's answer at T at some "
          "step 8-24, right at T): answers lost by the w5 start cut, revision probes vs the other right-at-T probes, raw "
          "and matched on the intact goal margin at T [5-95%]; Bayes's loss under its w5 window on the same probes")
    for xv in CTL_KEY_X:
        k = _k(Dc["x"], xv)
        r = netrev_split(k)
        print(f"  {Dc['x'][k]:7,.0f}: {r['n_rev']:5d} revision probes ({r['share']:.3f} of right at T); lost "
              f"{_ivs(r['rev'], r['rev_boot'])} vs {_ivs(r['oth'], r['oth_boot'])}; margin-matched difference "
              f"{_ivs(r['matched'], r['matched_boot'])}; Bayes {r['bayes_rev']:+.3f} vs {r['bayes_oth']:+.3f}")
    numbers_check2()


def bm_null_compare(row="acc_goal_T"):
    """the belief-matched swap against its matched null at every control checkpoint where it ran: the network's loss
    (seed mean over the 3 pools, range), the null's (Bayes swapped with donors matched on its own marginals from the same
    pools; seed mean, per seed), the network minus the null (pool-paired seed means, shared resamples), and Bayes under the
    network's donors (seed mean): dict of (K,) arrays and resamples"""
    Dc = load_ctl()
    G = dict(CTL_GROUPS)["belief-matched other cell"]
    dn = [ctl_dep(c, row) for c in G]
    dd = [ctl_dep(c, row, ref="donors") for c in G]
    ok = np.isfinite(dn[0]["net"])
    return dict(x=Dc["x"], ok=ok, net=np.mean([d["net"] for d in dn], 0), net_seeds=np.array([d["net"] for d in dn]),
                net_boot=np.mean([d["net_boot"] for d in dn], 0), null=np.mean([d["bayes"] for d in dn], 0),
                null_seeds=np.array([d["bayes"][0] for d in dn]), null_boot=np.mean([d["bayes_boot"] for d in dn], 0),
                ex=np.mean([d["ex"] for d in dn], 0), ex_boot=np.mean([d["ex_boot"] for d in dn], 0),
                donors=np.mean([d["bayes"] for d in dd], 0), donors_boot=np.mean([d["bayes_boot"] for d in dd], 0))


def bm_tv_quartiles(k, seed):
    """at control checkpoint index k and pool seed: goal answers at T lost by the belief-matched swap (network) and by
    the matched null (Bayes), by quartile of each probe's median matching distance over its 25 window starts (each
    observer's own distance; quartile edges inclusive on both sides, as the check's tv_strat.py)"""
    Dc, N, P = load_ctl(), load_bm_null(), _probes_full()
    ar, g = np.arange(len(P["goal"])), P["goal"]
    z = _load(ctl_path("trained", Dc["eps"][k]))
    zc = [str(c) for c in z["conds"]]
    tvn = np.median(z[f"tv_{seed}"], 1)
    lostn = z["right_T"][zc.index("intact")][ar, g].astype(np.float64) - z["right_T"][zc.index(f"bm_net_{seed}")][ar, g]
    BC = load_bayes_ctl()
    j = [int(s) for s in N["seeds"]].index(int(seed))
    tvb = np.median(N[f"tv_{seed}"], 1)
    lostb = BC["intact_right_T"][ar, g].astype(np.float64) - N["right_T"][j][ar, g]
    out = {}
    for nm, tv_, lost in (("net", tvn, lostn), ("null", tvb, lostb)):
        e = np.quantile(tv_, [0, .25, .5, .75, 1.])
        out[nm] = [(e[q], e[q + 1], lost[(tv_ >= e[q]) & (tv_ <= e[q + 1])].mean()) for q in range(4)]
    return out


def bm_quartile_residual(k):
    """at control checkpoint index k, per quartile of each observer's own median matching distance (as
    bm_tv_quartiles): the network's goal answers lost under the belief-matched swap and the null's, means over the 3
    pools, and the residual network - null with its 5-95% over the shared environment resamples (each observer's
    quartile probes weighted by their environment counts)"""
    Dc, N, P, BC = load_ctl(), load_bm_null(), _probes_full(), load_bayes_ctl()
    ar, g, env = np.arange(len(P["goal"])), P["goal"], P["env"]
    C = gap.boot_counts()
    z = _load(ctl_path("trained", Dc["eps"][k]))
    zc = [str(c) for c in z["conds"]]
    out = np.zeros((4, 2)), np.zeros((4, 2, NB))
    for j, s in enumerate(N["seeds"]):
        ln = z["right_T"][zc.index("intact")][ar, g].astype(np.float64) - z["right_T"][zc.index(f"bm_net_{s}")][ar, g]
        lb = BC["intact_right_T"][ar, g].astype(np.float64) - N["right_T"][j][ar, g]
        for o, (tv_, lost) in enumerate(((np.median(z[f"tv_{s}"], 1), ln), (np.median(N[f"tv_{s}"], 1), lb))):
            e = np.quantile(tv_, [0, .25, .5, .75, 1.])
            for q in range(4):
                msk = (tv_ >= e[q]) & (tv_ <= e[q + 1])
                sn, cn = np.bincount(env, lost * msk, NE), np.bincount(env, msk.astype(np.float64), NE)
                out[0][q, o] += sn.sum() / cn.sum() / len(N["seeds"])
                out[1][q, o] += (C @ sn) / (C @ cn) / len(N["seeds"])
    return dict(net=out[0][:, 0], null=out[0][:, 1], res=out[0][:, 0] - out[0][:, 1],
                res_boot=out[1][:, 0] - out[1][:, 1])


def numbers_check2():
    """the numbers added after the second check (11-02): the belief-matched swap against its matched null, by matching
    distance, and the families; the restart minus the content-free states at every control checkpoint; the step types
    in additive units; the same-cell q5 crossings against the accuracy c50; the random cell's levels by scheme"""
    Dc = load_ctl()
    N = load_bm_null()
    naive = ti.refs()["naive"]["acc_goal_T"]["value"]
    print("\n== MATCHED NULL of the belief-matched swap: Bayes swapped with donors matched on its own marginals (same "
          "pools, candidates and distance)")
    for i, s in enumerate(N["seeds"]):
        tv = N[f"tv_{s}"]
        rc = _probes_full()["r"][:, 0] * R + _probes_full()["r"][:, 1]
        dc = N[f"cell_{s}"].astype(np.int64)
        far = ((dc // R != (rc // R)[:, None]) & (dc % R != (rc % R)[:, None])).mean()
        bi = load_bayes_ctl()["intact_value"][0]
        print(f"  pool {s}: TV median {np.median(tv):.4f}, < .10 {(tv < .1).mean():.3f}; donors sharing neither latent "
              f"value {far:.2f}; goal accuracy at T lost {_ivs(bi - N['value'][i, 0], load_bayes_ctl()['intact_boot'][0] - N['boot'][i, 0], '+.4f')}")
    c = bm_null_compare()
    print("  per checkpoint: network belief-matched loss, seed mean (range) | null, seed mean | network - null [5-95%] | "
          "Bayes under the network's donors (seed mean) | median TV network (pool means) vs null")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tvn = np.nanmean([Dc[f"tv_med_{s}"] for s in SEEDS_POOL], 0)
    tvb = np.mean([np.median(N[f"tv_{s}"]) for s in N["seeds"]])
    for k in np.flatnonzero(c["ok"]):
        print(f"    {c['x'][k]:7,.0f}: {c['net'][k]:+.4f} ({np.min(c['net_seeds'][:, k]):+.4f} .. "
              f"{np.max(c['net_seeds'][:, k]):+.4f}) | {c['null'][k]:+.4f} | {_ivs(c['ex'][k], c['ex_boot'][:, k], '+.4f')} "
              f"| {c['donors'][k]:+.4f} | TV {tvn[k]:.3f} vs {tvb:.3f}")
    print("  goal answers lost by quartile of each probe's median matching distance (network: its read-out; null: Bayes's"
          " marginals); per pool seed, then the mean over the 3 pools")
    for xv in (400, 880, 1680):
        k = _k(Dc["x"], xv)
        qs = {s: bm_tv_quartiles(k, s) for s in SEEDS_POOL}
        for s in SEEDS_POOL:
            print(f"    {Dc['x'][k]:6,.0f} pool {s}: network " + "; ".join(f"TV {a:.3f}-{b:.3f}: {v:+.4f}" for a, b, v in qs[s]["net"])
                  + " | null " + "; ".join(f"TV {a:.3f}-{b:.3f}: {v:+.4f}" for a, b, v in qs[s]["null"]))
        print(f"    {Dc['x'][k]:6,.0f} mean of the pools: network " + " / ".join(
            f"{np.mean([qs[s]['net'][q][2] for s in SEEDS_POOL]):+.4f}" for q in range(4)) + " | null " + " / ".join(
            f"{np.mean([qs[s]['null'][q][2] for s in SEEDS_POOL]):+.4f}" for q in range(4)))
        rq = bm_quartile_residual(k)
        print(f"    {Dc['x'][k]:6,.0f} residual network - null per quartile (pool means, environment resamples): "
              + " / ".join(_ivs(rq["res"][q], rq["res_boot"][q], "+.4f") for q in range(4)))
    print("  families: lost under the belief-matched swap (network, seed mean) | matched null (seed mean) | network - null"
          " [5-95%] | Bayes under the network's donors")
    for row in ("held_two", "held_joint", "emass_slope", "band4", "G", "excess", "dir_slope", "far_T"):
        c = bm_null_compare(row)
        out = []
        for xv in (1680, 4400, 6400, 39996):
            k = _k(Dc["x"], xv)
            out.append(f"{Dc['x'][k]:,.0f}: {c['net'][k]:+.4f} | {c['null'][k]:+.4f} | "
                       f"{_ivs(c['ex'][k], c['ex_boot'][:, k], '+.4f')} | {c['donors'][k]:+.4f}")
        print(f"    {row:11s} " + "; ".join(out))
    print("\n== CONTROLS: the start state minus the content-free / wrong mid-episode states at every control checkpoint "
          "(network's goal answers lost, paired resamples, seed means; > 0: the restart costs more; * resolved)")
    ds = ctl_dep("start")
    for name in ("flat-belief other cell", "random cell, fresh at every window start", "environment-mean state",
                 "random cell, fixed per probe"):
        out = []
        for k in range(len(Dc["x"])):
            vals = [(ds["net"][k] - ctl_dep(cc)["net"][k], ds["net_boot"][:, k] - ctl_dep(cc)["net_boot"][:, k])
                    for cc in dict(CTL_GROUPS)[name]]
            vals = [v for v in vals if np.isfinite(v[0])]
            if not vals:
                continue
            v, b = np.mean([a for a, _ in vals]), np.mean([bb for _, bb in vals], 0)
            lo, hi = np.nanpercentile(b, [5, 95])
            out.append(f"{Dc['x'][k]:,.0f}: {_ivs(v, b)}{'*' if lo > 0 or hi < 0 else ''}")
        print(f"  {name}\n    " + "; ".join(out))
    print("  restart's own loss for scale: " + "; ".join(f"{Dc['x'][k]:,.0f}: {ds['net'][k]:+.3f}"
                                                       for k in [_k(Dc["x"], xv) for xv in (480, 720, 880, 1680)]))
    print("\n== CONTROLS: the random cell's levels (goal accuracy at T) by scheme, seed means: network cut | Bayes cut; "
          f"naive {naive:.3f}; the network's gain over naive intact")
    for name in ("random cell, fixed per probe", "random cell, fresh at every window start"):
        dd = [ctl_dep(cc) for cc in dict(CTL_GROUPS)[name]]
        out = []
        for xv in (880, 1360, 1680, 3000, 7600, 39996):
            k = _k(Dc["x"], xv)
            out.append(f"{Dc['x'][k]:,.0f}: {np.mean([x['cut'][k] for x in dd]):.3f} | {np.mean([x['bayes_cut'][k] for x in dd]):.3f}"
                       f" (loss {np.mean([x['net'][k] for x in dd]):.3f}; gain {dd[0]['intact'][k] - naive:+.3f})")
        print(f"  {name:42s} " + "; ".join(out))
    print("\n== WHERE: Bayes's step types in additive units (competitor's gain, nats per variable-step after the window, "
          "w5 start): network revision - other and revision - settled [5-95%], Bayes's; the excess per group")
    m, b = where_measure("trained", "w5_start", "g"), where_measure("bayes", "w5_start", "g")
    print(f"  Bayes: revision - other {b['value'][0] - b['value'][2]:+.4f}, revision - settled "
          f"{b['value'][0] - b['value'][1]:+.4f} (per group {' / '.join(f'{v:.4f}' for v in b['value'])})")
    for k in range(len(m["x"])):
        ex = m["value"][k] - b["value"]
        eb = m["boot"][:, k] - b["boot"]                                                # (NB, 3) excess per group
        print(f"  {m['x'][k]:7,.0f}: revision - other {_ivs(m['value'][k, 0] - m['value'][k, 2], m['boot'][:, k, 0] - m['boot'][:, k, 2], '+.4f')}"
              f"; revision - settled {_ivs(m['value'][k, 0] - m['value'][k, 1], m['boot'][:, k, 0] - m['boot'][:, k, 1], '+.4f')}"
              f"; excess per group {' / '.join(f'{v:.3f}' for v in ex)}; excess revision - other "
              f"{_ivs(ex[0] - ex[2], eb[:, 0] - eb[:, 2], '+.4f')}, revision - settled {_ivs(ex[0] - ex[1], eb[:, 0] - eb[:, 1], '+.4f')}")
    print("\n== RANDOM CELL (the engine's donor), families drawn in the where figure: the network's dependence and its excess "
          "over Bayes's under the same swap [5-95%]")
    for row in ("acc_goal_T", "split_grad", "held_two", "G", "band4"):
        d = dep("trained", "w5_random", row)
        print(f"  {row:10s} Bayes {d['bayes']:+.4f} | " + "; ".join(
            f"{d['x'][k]:,.0f}: {d['net'][k]:+.4f}, excess {_ivs(d['ex'][k], d['ex_boot'][:, k], '+.4f')}"
            for k in [_k(d["x"], xv) for xv in (1360, 1680, 4400, 7600, 11200, 19200, 39996)]))
    print("\n== Q5 CONTRAST crossings against the accuracy c50 (interval rule; paired share P(c50 < crossing))")
    acc = family_marks()["acc_goal_T"]
    for cond in ("w2_same_cell", "w5_same_cell", "w2_start", "w5_start", "w2_random", "w5_random"):
        q = q5_contrast(cond)
        print(f"  {cond:13s} {_cx(q['cross'])} vs c50 {_cx(acc['c50'])}: crossing {_order(q['cross'], acc['c50'])}; "
              f"P(c50 < crossing) {_order(acc['c50'], q['cross'])}")
    print("\n== UNCUT: Bayes's goal accuracy at steps 1-10 and naive's (for the build-up panel)")
    u = uncut_steps()
    print("  Bayes " + " ".join(f"{v:.3f}" for v in u["bayes"][:10]) + " | naive " + " ".join(f"{v:.3f}" for v in u["naive"][:10]))
    for s0 in (5, 10):
        rc = uncut_reach(s0)
        out = []
        for xv in (400, 880, 1360, 1680, 2080, 3000, 4400, 11200, 39996):
            k = _k(rc["x"], xv)
            b = np.where(np.isnan(rc["boot"][:, k]), np.inf, rc["boot"][:, k])
            lo, hi = np.percentile(b, 5, method="lower"), np.percentile(b, 95, method="higher")
            txt = "never" if not np.isfinite(rc["step"][k]) else f"{rc['step'][k]:.1f}"
            out.append(f"{rc['x'][k]:,.0f}: {txt} [{lo:.1f}, {'never' if not np.isfinite(hi) else f'{hi:.1f}'}]")
        print(f"  uncut network reaches Bayes's step-{s0} accuracy ({rc['level']:.3f}) at step: " + "; ".join(out))


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

GUIDE = "#6a3d9a"                                                 # guides: the answers' family colour (purple)
SPAN = "#e3efd9"                                                  # holding / coupling half-way span (light green)
ZERO = dict(color="0.55", lw=1., ls=(0, (1, 1.2)))               # loss panels: 0 = naive's loss under every swap
NAIVE = dict(color="0.55", lw=1.3, ls=(0, (1, 1.2)))             # naive's level
DASH = (0, (4, 1.6))
DOT = (0, (1, 1.3))
DASHDOT = (0, (6, 1.6, 1.2, 1.6))
SRC_TITLE = {"start": "restarted from $(h_0, c_0)$", "same_cell": "same cell, other trajectory",
             "random": "another cell's state"}


def _xaxis(ax, fs, label=True, lo=8):
    ax.set_xscale("log")
    ax.set_xlim(lo, 52000)
    ticks = [t for t in (10, 1e2, 300, 1e3, 1e4) if t >= lo and (t != 300 or lo > 100)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([{10: "0", 1e2: "100", 300: "300", 1e3: "1k", 1e4: "10k"}[t] for t in ticks])
    ax.minorticks_off()
    if label:
        ax.set_xlabel("training episodes", fontsize=fs["label"])


def _marks(ax, M):
    """vertical guides (purple): the Trained RNN's accuracy passes naive (dotted) and is half-way to Bayes (solid,
    thin); the holding / coupling half-way span (light green)"""
    acc = M["acc_goal_T"]
    ax.axvline(acc["c0"]["p"], color=GUIDE, lw=.9, ls=(0, (1, 1.5)), zorder=1, alpha=.8)
    ax.axvline(acc["c50"]["p"], color=GUIDE, lw=.7, ls="-", zorder=1, alpha=.8)
    late = [M[k]["c50"]["p"] for k in ("emass_slope", "held_two", "excess", "G", "dir_slope", "band4", "held_one",
                                       "held_joint")]
    ax.axvspan(min(late), max(late), color=SPAN, lw=0, zorder=0)


def _seed_mean(conds, key="net", row="acc_goal_T"):
    """mean over the seeds of a control group (K,), its pointwise range over seeds, the 5-95% band of the seed mean's
    resamples, and Bayes's dependence and level under the group's reference (means over seeds and the checkpoints where
    the reference exists; bmin / bmax its range)"""
    dd = [ctl_dep(c, row) for c in conds]
    v = np.array([d[key] for d in dd])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        b = np.mean([d[key + "_boot"] for d in dd], 0)
        bv = np.array([d["bayes"] for d in dd])
        bc = np.array([d["bayes_cut"] for d in dd])
        return dict(x=dd[0]["x"], v=np.nanmean(v, 0), lo=np.nanpercentile(b, 5, 0), hi=np.nanpercentile(b, 95, 0),
                    smin=np.nanmin(v, 0), smax=np.nanmax(v, 0), bayes=float(np.nanmean(bv)), bmin=float(np.nanmin(bv)),
                    bmax=float(np.nanmax(bv)), bayes_cut=float(np.nanmean(bc)), bayes_intact=dd[0]["bayes_intact"])


def _interval(b):
    """5-95% of crossing-type resamples (NaN = never: after everything)"""
    b = np.where(np.isnan(b), np.inf, b)
    return np.percentile(b, 5, method="lower"), np.percentile(b, 95, method="higher")


def _pair(net, bay=None, **kw):
    """a legend handle pairing the Trained RNN's line (blue) with Bayes's (black) of the same style"""
    from matplotlib.lines import Line2D
    h = Line2D([], [], color=COL["trained"], **net)
    return h if bay is None else (h, Line2D([], [], color="k", **bay))


def figure_answers():
    """-> figure_bin/td_tt_memcuts.png (lean, after the second check): (a) the restart from (h0, c0) 5 steps back, the
    network against Bayes's 5-step window (and the ESN); (b) uncut, the step at which the network reaches Bayes's
    step-5 accuracy; (c) another cell's state (the engine's random-cell swap and a new donor at every step), levels
    against Bayes under the same swap; (d) the cell's content from another history: the same read-out belief against
    Bayes's matched null, the same cell's other trajectory and the shuffled own history; (e) mid-episode states without the probe's cell
    content against the restart. In the docstring only: w = 2 / 10, the averaged states, the uncut gain at step 5 / T
    through training"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec
    from matplotlib.legend_handler import HandlerTuple
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    M = family_marks()
    De, Dc, Dt = load_run("echo"), load_ctl(), load_run("trained")
    G = dict(CTL_GROUPS)
    fig = plt.figure(figsize=(14.6, 12.0))
    gs = GridSpec(2, 6, figure=fig, left=.065, right=.985, top=.825, bottom=.205, hspace=.98, wspace=.8)
    A = {"a": fig.add_subplot(gs[0, 0:2]), "b": fig.add_subplot(gs[0, 2:4]), "c": fig.add_subplot(gs[0, 4:6]),
         "d": fig.add_subplot(gs[1, 1:3]), "e": fig.add_subplot(gs[1, 3:5])}
    TF = fs["label"] + 1.5
    LF = fs["legend"] - 1.

    def below(ax, handles, labels=None, ncol=1, dy=-.19):
        labels = [h.get_label() for h in handles] if labels is None else labels
        ax.legend(handles=handles, labels=labels, fontsize=LF, frameon=False, loc="upper center",
                  bbox_to_anchor=(.5, dy), ncol=ncol, columnspacing=1.2, handlelength=3.2,
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=.4)})

    def title(ax, s):
        ax.set_title(s, loc="left", fontsize=TF, linespacing=1.45)
    # a: the restart from (h0, c0) 5 steps back
    ax = A["a"]
    _marks(ax, M)
    ax.axhline(0, **ZERO)
    rs = dep("trained", "w5_start", "acc_goal_T")
    ax.fill_between(rs["x"], rs["net_lo"], rs["net_hi"], color=COL["trained"], alpha=.25, lw=0)
    ax.plot(rs["x"], rs["net"], color=COL["trained"], lw=2.)
    ax.axhline(rs["bayes"], color="k", lw=1.3, ls=DASH)
    de = dep("echo", "w5_start", "acc_goal_T")
    ax.plot(De["x"], de["net"], color=COL["echo"], lw=1.6)
    _xaxis(ax, fs)
    ax.set_ylim(-.02, .2)
    ax.set_ylabel("goal accuracy at $T$ lost", fontsize=fs["label"])
    title(ax, "a   restarted 5 steps back: answers lost\n"
          r"$\mathrm{acc}_T(\mathrm{intact}) - \mathrm{acc}_T(\mathrm{cut})$,  $(h, c)_{t-5} \leftarrow (h_0, c_0)$")
    below(ax, [Line2D([], [], color=COL["trained"], lw=2., label="Trained RNN"),
               Line2D([], [], color="k", lw=1.3, ls=DASH, label=f"Bayes, 5-step window ({rs['bayes']:.3f})"),
               Line2D([], [], color=COL["echo"], lw=1.6, label="ESN")])
    # b: uncut, the step at which the network reaches Bayes's step-5 accuracy
    ax = A["b"]
    u, rc = uncut_steps(), uncut_reach(5)
    t = np.arange(1, T + 1)
    blues = plt.get_cmap("Blues")
    ax.plot(t, u["naive"], **NAIVE)
    ax.plot(t, u["bayes"], color="k", lw=1.6)
    ax.axhline(rc["level"], color="k", lw=.8, ls=DASH)
    ax.plot([5], [rc["level"]], marker="s", ms=6, color="k", zorder=6)
    hb = []
    k5 = {}
    for i_, xv in enumerate((880, 1680, 39996)):
        k = _k(u["x"], xv)
        col = blues(.45 + .25 * i_)
        ax.plot(t, u["net"][k], color=col, lw=2.)
        s = rc["step"][k]
        lo, hi = _interval(rc["boot"][:, k])
        k5[xv] = s
        ax.plot([lo, hi], [rc["level"]] * 2, color=col, lw=3.4, solid_capstyle="butt", zorder=5)
        ax.plot([s], [rc["level"]], marker="o", ms=7, mfc=col, mec="k", mew=.8, zorder=7)
        hb.append(Line2D([], [], color=col, lw=2., marker="o", ms=6, mfc=col, mec="k", mew=.8,
                         label=f"Trained RNN at {u['x'][k]:,.0f}: step {s:.1f} [{lo:.1f}, {hi:.1f}]"))
    ax.set_xlim(1, T)
    ax.set_ylim(.05, .82)
    ax.set_xlabel("step $t$ of the episode (no cut)", fontsize=fs["label"])
    ax.set_ylabel("goal accuracy at step $t$", fontsize=fs["label"])
    title(ax, "b   uncut: the step at which the network knows\nwhat Bayes knows at step 5:  "
          r"$\mathrm{acc}_t = \mathrm{acc}^{\mathrm{Bayes}}_{5}$")
    below(ax, hb + [Line2D([], [], color="k", lw=1.6, marker="s", ms=5,
                           label=f"Bayes (at step 5: {rc['level']:.3f}, dashed)"),
                    Line2D([], [], label="naive", **NAIVE)])
    # c: another cell's state 5 steps back (the random-cell swap): levels against Bayes under the same swap
    ax = A["c"]
    _marks(ax, M)
    sf = _seed_mean(G["random cell, fixed per probe"], key="cut")
    ss = _seed_mean(G["random cell, fresh at every window start"], key="cut")
    si = _seed_mean(("start",), key="intact")
    naive = ti.refs()["naive"]["acc_goal_T"]["value"]
    ax.axhline(sf["bayes_intact"], color="k", lw=2.4)
    ax.axhline(sf["bayes_cut"], color="k", lw=1.2)
    ax.axhline(ss["bayes_cut"], color="k", lw=1.2, ls=DASH)
    ax.axhline(naive, **NAIVE)
    ax.plot(si["x"], si["v"], color=COL["trained"], lw=2.6)
    ax.fill_between(sf["x"], sf["lo"], sf["hi"], color=COL["trained"], alpha=.22, lw=0)
    ax.plot(sf["x"], sf["v"], color=COL["trained"], lw=1.6)
    ax.plot(ss["x"], ss["v"], color=COL["trained"], lw=1.6, ls=DASH)
    _xaxis(ax, fs)
    ax.set_ylim(.08, .82)
    ax.set_ylabel("goal accuracy at $T$", fontsize=fs["label"])
    title(ax, "c   another cell's state 5 steps back: answers\n"
          r"$\mathrm{acc}_T$,  $(h, c)_{t-5} \leftarrow$ another cell's $(h, c)_{t-5}$")
    below(ax, [_pair(dict(lw=2.6), dict(lw=2.4)), _pair(dict(lw=1.6), dict(lw=1.2)),
               _pair(dict(lw=1.6, ls=DASH), dict(lw=1.2, ls=DASH)), Line2D([], [], **NAIVE)],
          [f"intact: Trained RNN (blue), Bayes (black, {sf['bayes_intact']:.3f})",
           f"another cell, one donor episode (Bayes {sf['bayes_cut']:.3f})",
           f"another cell, new donor each step (Bayes {ss['bayes_cut']:.3f})", f"naive ({naive:.3f})"])
    # d: the same read-out belief from another history, against Bayes's matched null
    ax = A["d"]
    _marks(ax, M)
    ax.axhline(0, **ZERO)
    bm = _seed_mean(G["belief-matched other cell"])
    sh = _seed_mean(G["shuffled own history"])
    sc = _seed_mean(("same_cell",))
    ok = np.isfinite(bm["v"])
    ax.fill_between(bm["x"][ok], bm["lo"][ok], bm["hi"][ok], color=COL["trained"], alpha=.22, lw=0)
    ax.plot(bm["x"][ok], bm["v"][ok], color=COL["trained"], lw=1.9, marker="o", ms=4)
    ax.axhline(bm["bayes"], color="k", lw=1.3)
    ax.plot(sc["x"], sc["v"], color=COL["trained"], lw=1.3, ls=DASH)
    ax.axhline(sc["bayes"], color="k", lw=1.1, ls=DASH)
    ax.plot(sh["x"][ok], sh["v"][ok], color=COL["trained"], lw=1.5, ls=DOT, marker="^", ms=4.5)
    _xaxis(ax, fs)
    ax.set_ylim(-.03, .036)
    ax.set_ylabel("goal accuracy at $T$ lost", fontsize=fs["label"])
    title(ax, "d   the cell's content from another history: answers lost\n"
          r"$\mathrm{acc}_T(\mathrm{intact}) - \mathrm{acc}_T(\mathrm{swap})$,  $w = 5$  "
          f"(restart: up to {np.nanmax(rs['net']):.2f})")
    below(ax, [_pair(dict(lw=1.9, marker="o", ms=4), dict(lw=1.3)), _pair(dict(lw=1.3, ls=DASH), dict(lw=1.1, ls=DASH)),
               _pair(dict(lw=1.5, ls=DOT, marker="^", ms=4.5))],
          [f"another cell, same read-out belief (Bayes: same marginals, {bm['bayes']:.3f})",
           f"same cell, other trajectory (Bayes {sc['bayes']:.3f})",
           "own observations, each bit's order shuffled (Bayes 0)"])
    # e: mid-episode states without the probe's cell content, against the restart
    ax = A["e"]
    _marks(ax, M)
    ax.axhline(0, **ZERO)
    he, le = [], []
    for key, conds, ls, mk, lab in (
            ("start", ("start",), "-", {}, "restart from $(h_0, c_0)$"),
            ("flat", G["flat-belief other cell"], "-", dict(marker="o", ms=3.5), "flattest read-out, another cell"),
            ("rand_step", G["random cell, fresh at every window start"], DASH, {}, "another cell, new donor each step"),
            ("mean_env", ("mean_env",), DASHDOT, {}, "the environment's mean state"),
            ("rand_fix", G["random cell, fixed per probe"], DOT, {}, "another cell, one donor episode")):
        s = _seed_mean(conds)
        ok = np.isfinite(s["v"])
        lw = 2.1 if key == "start" else 1.4
        if key == "start":
            ax.fill_between(s["x"][ok], s["lo"][ok], s["hi"][ok], color=COL["trained"], alpha=.22, lw=0)
        ax.plot(s["x"][ok], s["v"][ok], color=COL["trained"], lw=lw, ls=ls, **mk)
        he.append(Line2D([], [], color=COL["trained"], lw=lw, ls=ls, **mk))
        bt = f"{s['bayes']:.3f}" if s["bmax"] - s["bmin"] < .005 else f"{s['bmin']:.3f}-{s['bmax']:.3f}"
        le.append(f"{lab} (Bayes {bt})")
    _xaxis(ax, fs)
    ax.set_ylim(-.02, .38)
    ax.set_ylabel("goal accuracy at $T$ lost", fontsize=fs["label"])
    title(ax, "e   no own-cell content, mid-episode: answers lost\n"
          r"$\mathrm{acc}_T(\mathrm{intact}) - \mathrm{acc}_T(\mathrm{swap})$,  $w = 5$,  Trained RNN")
    below(ax, he, le)
    acc = M["acc_goal_T"]
    late = [M[k]["c50"]["p"] for k in ("emass_slope", "held_two", "excess", "G", "dir_slope", "band4", "held_one",
                                       "held_joint")]
    nfull = int(Dc["has"][:, Dc["conds"].index(f"bm_net_{SEEDS_POOL[0]}")].sum())
    fig.text(.5, .036, "Trained RNN seed 20261029; probes: 50 test environments x 200 trajectories (100 latent pairs x 2), "
             f"30 steps. Trained RNN: {len(Dt['x'])} checkpoints ({nfull} for the belief-matched, shuffled and flattest "
             "states, 3 seeds each; another cell: 4 seeds per scheme;\ncurves = seed means); ESN: "
             f"{len(De['x'])} checkpoints. Belief-matched and flattest donors: 2,000 generated episodes per environment "
             "and seed. Blue: Trained RNN; black: Bayes, exact, under the same swap (d: matched on its own marginals\n"
             "from the same pools); orange: ESN; grey dotted: naive (no memory: it loses nothing under any swap). Bands / "
             "bars: 5-95% over 1,000 resamples of the environments. Guides (purple): Trained RNN accuracy passes naive\n"
             f"({acc['c0']['p']:,.0f}, dotted) and is half-way to Bayes ({acc['c50']['p']:,.0f}, solid); green: holding / "
             f"coupling half-way ({min(late):,.0f}-{max(late):,.0f}).", ha="center", va="center",
             fontsize=fs["legend"] - 1.5, linespacing=1.55)
    fig.suptitle("Restarted 5 steps back, the Trained RNN's answers lose far more than Bayes's while they develop (a: up to "
                 f"{np.nanmax(rs['net']):.2f} vs {rs['bayes']:.2f});\nuncut, it reaches Bayes's step-5 accuracy only at "
                 f"step {k5[880]:.1f} / {k5[1680]:.1f} / {k5[39996]:.1f} (880 / 1,680 / 40k training episodes; b).\nOne "
                 "other cell's state takes it to about naive's level or below, as it takes Bayes below naive; a new donor at "
                 "each step costs it less (c).\nThe same read-out belief from another history costs at most "
                 f"{np.nanmax(bm['v']):.3f}, a little more than Bayes's matched swap at 960-3,000 (d); states without\nthe "
                 "cell's content cost about as much as the restart at 960-2,080 (some a little less at 480, 720 and 880), one "
                 "fixed wrong cell's state more (e).", y=.988, fontsize=fs["title"] - 3, linespacing=1.38)
    path = fig_path("td_tt_memcuts.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


FAMS = (("acc_goal_T", "answers (goal accuracy)", "#6a3d9a"), ("split_grad", "splitting (split gradient)", "#01665e"),
        ("held_two", "holding (split held, per variable)", "#3a7a19"), ("G", "coupling (long-path G)", "#b2182b"),
        ("band4", "coupling (angle of jumps $\\geq$ 4 cells)", "#e7298a"))
X_FAM = 300                                                       # family curves drawn from here (earlier: untrained jumps)


def gap_dep(run, cond, row):
    """dependence in units of the row's naive-to-Bayes gap: (y_int - y_cut) / (y_Bayes - y_naive) (td_tt_gap's fixed
    affine scale per row), the network's (K,) and resamples, Bayes's (scalar)"""
    d = dep(run, cond, row)
    g = d["bayes_intact"] - d["naive"]
    return dict(x=d["x"], net=d["net"] / g, net_lo=d["net_lo"] / g, net_hi=d["net_hi"] / g, bayes=d["bayes"] / g)


def figure_where():
    """-> figure_bin/td_tt_memcuts_where.png (lean, after the second check): (a-c) each behaviour family's memory
    dependence (w = 5) under the restart, another cell's state and the same cell's other trajectory, in units of its
    naive-to-Bayes gap, against Bayes under the same swap; (d) the restart's excess over Bayes per entanglement fifth
    through training (additive, nats per variable-step); (e) the restart's answers lost by Bayes's episode type. In the
    docstring only: far errors, the loss along the episode, Bayes's step types, the direction slope, the excess angle"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec
    from matplotlib.legend_handler import HandlerTuple
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    M = family_marks()
    Dt, B = load_run("trained"), load_bayes()
    fig = plt.figure(figsize=(14.6, 11.2))
    gs = GridSpec(2, 6, figure=fig, left=.065, right=.985, top=.785, bottom=.2, hspace=.62, wspace=.8)
    A = {"a": fig.add_subplot(gs[0, 0:2]), "b": fig.add_subplot(gs[0, 2:4]), "c": fig.add_subplot(gs[0, 4:6]),
         "d": fig.add_subplot(gs[1, 1:3]), "e": fig.add_subplot(gs[1, 3:5])}
    TF = fs["label"] + 1.5
    LF = fs["legend"] - 1.
    lo_, hi_ = 0., 0.
    for key, s_ in (("a", "start"), ("b", "random"), ("c", "same_cell")):
        ax = A[key]
        _marks(ax, M)
        ax.axhline(0, **ZERO)
        for row, name, col in FAMS:
            d = gap_dep("trained", f"w5_{s_}", row)
            ok = (d["x"] >= X_FAM) & np.isfinite(d["net"])
            ax.axhline(d["bayes"], color=col, lw=1.2, ls=DASH)
            ax.fill_between(d["x"][ok], d["net_lo"][ok], d["net_hi"][ok], color=col, alpha=.13, lw=0)
            ax.plot(d["x"][ok], d["net"][ok], color=col, lw=1.8)
            c = M[row]["c50"]
            ax.plot([c["p"]], [0.], transform=ax.get_xaxis_transform(), marker="^", ms=8, mfc=col, mec="k", mew=.6,
                    clip_on=False, zorder=9)
            lo_, hi_ = min(lo_, np.nanmin(d["net_lo"][ok]), d["bayes"]), max(hi_, np.nanmax(d["net_hi"][ok]), d["bayes"])
        _xaxis(ax, fs, lo=240)
        ax.set_title(f"{key}   {SRC_TITLE[s_]}: lost, per family\n"
                     r"$(y_{\mathrm{intact}} - y_{\mathrm{swap}})\,/\,(y_{\mathrm{Bayes}} - y_{\mathrm{naive}})$,  $w = 5$",
                     loc="left", fontsize=TF, linespacing=1.45)
        if key == "a":
            ax.set_ylabel("share of the naive-to-Bayes gap lost", fontsize=fs["label"])
    for key in "abc":
        A[key].set_ylim(lo_ - .05, hi_ + .05)
    h1 = [Line2D([], [], color=col, lw=1.8, label=name) for _, name, col in FAMS] + \
         [Line2D([], [], color="k", lw=1.2, ls=DASH, label="Bayes, same swap (dashed, family colour)"),
          Line2D([], [], color="0.3", lw=0, marker="^", ms=8, mfc="w", mec="k", mew=.6, label="family half-way (c50)")]
    fig.legend(handles=h1, loc="center", bbox_to_anchor=(.5, .853), ncol=4, frameon=False, fontsize=LF,
               columnspacing=1.6)
    # d: the restart's excess per entanglement fifth through training (additive)
    ax = A["d"]
    _marks(ax, M)
    ax.axhline(0, **ZERO)
    fe = fifth_excess()
    vir = plt.get_cmap("viridis")
    hd = []
    for q in range(NF):
        col = vir(.08 + .84 * q / (NF - 1))
        if q in (0, NF - 1):
            lo, hi = np.nanpercentile(fe["ex_boot"][:, :, q], [5, 95], 0)
            ax.fill_between(fe["x"], lo, hi, color=col, alpha=.2, lw=0)
        ax.plot(fe["x"], fe["ex"][:, q], color=col, lw=1.8, marker="o", ms=3.5)
        hd.append(Line2D([], [], color=col, lw=1.8, marker="o", ms=3.5,
                         label={0: "1 (least entangled)", NF - 1: "5 (most)"}.get(q, str(q + 1))))
    _xaxis(ax, fs, lo=240)
    ax.set_ylabel("nats per variable-step", fontsize=fs["label"])
    ax.set_title("d   restart: extra log-odds lost per step, by entanglement\n"
                 r"$\langle d_t \rangle_{\mathrm{RNN}} - \langle d_t \rangle_{\mathrm{Bayes}}$,  "
                 r"$d_t$: true value vs its strongest competitor", loc="left", fontsize=TF, linespacing=1.45)
    ax.legend(handles=hd, fontsize=LF, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.19), ncol=5,
              columnspacing=1., handlelength=2.2, title="entanglement fifth of the step (own variable)",
              title_fontsize=LF)
    # e: answers lost by Bayes's episode type
    ax = A["e"]
    _marks(ax, M)
    ax.axhline(0, **ZERO)
    he, le = [], []
    for nm, mask, ls in (("Bayes revises", B["fam"] == 0, "-"), ("Bayes settled early", B["fam"] == 1, DASH),
                         ("neither", B["fam"] == 2, DOT)):
        f = family_dep("trained", "w5_start", mask)
        ok = Dt["x"] >= X_FAM
        ax.plot(Dt["x"][ok], f["net"][ok], color=COL["trained"], lw=1.9, ls=ls)
        ax.axhline(f["bayes"], color="k", lw=1.2, ls=ls)
        he.append(_pair(dict(lw=1.9, ls=ls), dict(lw=1.2, ls=ls)))
        le.append(f"{nm}: {f['n']:,} episodes (Bayes {f['bayes']:.3f})")
    _xaxis(ax, fs, lo=240)
    ax.set_ylabel("goal accuracy at $T$ lost", fontsize=fs["label"])
    ax.set_title("e   restart: answers lost, by episode type\n"
                 r"$\mathrm{acc}_T(\mathrm{intact}) - \mathrm{acc}_T(\mathrm{cut})$,  blue: Trained RNN, black: Bayes",
                 loc="left", fontsize=TF, linespacing=1.45)
    ax.legend(handles=he, labels=le, fontsize=LF, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.19), ncol=1,
              handlelength=3.2, handler_map={tuple: HandlerTuple(ndivide=None, pad=.4)})
    W = load_where("trained")
    fig.text(.5, .035, "Trained RNN seed 20261029; probes: 50 test environments x 200 trajectories (100 latent pairs x 2), "
             f"30 steps; {len(Dt['x'])} checkpoints (a-c, e: curves from {X_FAM} training episodes), {len(W['x'])} (d). "
             "Bands: 5-95% over 1,000 resamples of the environments.\n(d): 4,000 variable-steps per fifth and step after "
             r"the window; $d_t$ = the step's log-odds of the true value against its strongest competitor in the cut "
             "belief at $T$, intact minus cut (Bayes's per fifth subtracted).\n(e): goal variable; Bayes revises = Bayes "
             "puts more than .6 on another value at some step 8-24; settled early = Bayes's answer is the same at steps "
             "8-30. Held per variable: 720 corner-split variable-steps.\nGuides as in the answers figure; grey dotted: "
             "naive (no memory: it loses nothing under any swap).", ha="center", va="center",
             fontsize=fs["legend"] - 1.5, linespacing=1.55)
    fig.suptitle("Restarted 5 steps back (a), answers and splitting lose more than Bayes while they develop (about 1-2k "
                 "training episodes), then holding and coupling's jumps $\\geq$ 4 cells (5-6k);\ncoupling's G does not. "
                 "With another cell's state (b) the losses of answers, holding and coupling grow: by the end answers and "
                 "holding lose about Bayes's amount,\ncoupling less; splitting loses more than Bayes, which loses none. With "
                 "the same cell's other "
                 "trajectory (c) the drawn families lose about Bayes's amount or less. The restart's\nextra loss falls on "
                 "every entanglement fifth (d) and, through about 11k training episodes, is not concentrated on the episodes "
                 "Bayes revises (e).", y=.985, fontsize=fs["title"] - 3, linespacing=1.4)
    path = fig_path("td_tt_memcuts_where.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


# ═══════════════════════════════════════════════════════════════════
# Checks
# ═══════════════════════════════════════════════════════════════════

def checks():
    """validation of the Bayes references, the per-checkpoint identities and the stored reproduction of trunc_right
    -> OUT/checks.json"""
    t0 = time.time()
    P, B = _probes_full(), load_bayes()
    n = len(P["env"])
    res = {}
    for w in (1, 2, 5):                                    # windowed Bayes = td_analysis an6 (probes right_windowW)
        lq = bayes_logq(w, "start")
        res[f"window{w}_mismatch"] = int(((lq[:, -1].argmax(-1) == P["r"]) != P[f"right_window{w}"]).sum())
    lam = _lam()
    rng = np.random.default_rng(0)
    worst = 0.
    dn = donors()
    for s in SOURCES:                                      # brute force: explicit histories, joint posterior from scratch
        for w in WS:
            lq = bayes_logq(w, s)
            for i in rng.choice(n, 3, replace=False):
                cum = np.zeros((NV, R))
                for t in range(1, T + 1):
                    pre = [] if (t <= w or dn[s] is None) else [(dn[s][i], u) for u in range(1, t - w + 1)]
                    own0 = 1 if t <= w else t - w + 1

                    def post(hist):
                        lj = np.zeros((R, R))
                        for (k, u) in hist:
                            lj += lam[k, u] - lam[k, u - 1]
                        return np.stack([logsumexp(lj, 1), logsumexp(lj, 0)])
                    own = [(i, u) for u in range(own0, t + 1)]
                    cum += post(pre + own) - post(pre + own[:-1])
                    worst = max(worst, float(np.abs(cum - logsumexp(cum, -1, keepdims=True) - lq[i, t - 1]).max()))
    res["bayes_donor_bruteforce_max_abs"] = worst
    full = bayes_logq()
    res["bayes_full_vs_log_jb_max_abs"] = float(np.abs(full - np.load(os.path.join(ti.AN, "probes.npz"))["log_jb"]).max())
    rb = ti.refs("full")["bayes"]
    res["bayes_rows_vs_refs_max_abs"] = float(max(abs(B["value"][0, k] - rb[r]["value"]) for k, r in enumerate(ROWS)
                                                  if np.isfinite(rb[r]["value"])))
    z = pair_stats(full, full, 5)
    res["swap_identity_zero"] = float(max(np.abs(z[k]).max() for k in ("flip_f", "flip_g", "dm_f", "dm_g")))
    res["bayes_dd_steps_le_w"] = float(max(np.abs(B["sz_f"][cix(c)][:, :_wsteps(c)] + B["sz_f"][0][:, :_wsteps(c)]
                                                  - 2 * B["ci_f"][cix(c)][:, :_wsteps(c)]).max() for c in CONDS[1:]))
    for run in ("trained", "echo"):
        D = load_run(run)
        ch = [m["checks"] for m in D["meta"]]
        res[run] = dict(n_checkpoints=len(ch), trunc_w2=int(sum(c["trunc_w2"] for c in ch)),
                        trunc_w5=int(sum(c["trunc_w5"] for c in ch)), trunc_intact=int(sum(c["trunc_intact"] for c in ch)),
                        ev_identity=float(max(c["ev_sum_identity"] for c in ch)),
                        seconds=float(np.median([m["seconds"] for m in D["meta"]])),
                        forward_s=float(np.median([m["forward_s"] for m in D["meta"]])),
                        peak_gpu_mb=float(max(m["peak"] for m in D["meta"])))
        sw = [float(np.abs(D["sz_f"][:, c][:, :, :_wsteps(cc)] + D["sz_f"][:, 0][:, :, :_wsteps(cc)]
                           - 2 * D["ci_f"][:, c][:, :, :_wsteps(cc)]).max()) for c, cc in enumerate(CONDS) if c]
        res[run]["dd_steps_le_w"] = max(sw)
    res["seconds"] = time.time() - t0
    with open(os.path.join(OUT, "checks.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    return res


def ctl_checks(ep=2100, seed=SEEDS_POOL[0], n_brute=9):
    """checks of the controls -> OUT/controls/checks.json: (1) the start / same_cell / random cuts and the intact run
    of the controls' own forward give the harvest's answers at T at every checkpoint; (2) Bayes under the shuffled
    histories equals Bayes intact; (3) the joint log-likelihood from per-bit counts equals the probes' cumulative one;
    (4) the generated pools follow the environments' tables (per-cell bit frequencies against L, z-scores; the probes'
    own frequencies for comparison); (5) GPU re-derivation of the pool donors at one checkpoint and seed: the cached
    answers are reproduced exactly, every donor is of another cell and another observation history, and Bayes's
    reference rebuilt from the explicit donor histories (joint posterior from scratch for n_brute probes x 25 window
    starts) equals the counts-based one"""
    t0 = time.time()
    res = {}
    Dc = load_ctl()
    res["validation_mismatches"] = {k: int(sum(v[k] for v in Dc["val"])) for k in Dc["val"][0]}
    res["n_checkpoints"] = len(Dc["x"])
    res["bayes_shuffle_vs_intact"] = json.loads(str(load_bayes_ctl()["shuffle_vs_intact"]))
    P = _probes_full()
    cnt = np.stack([P["obs"][:, :a].sum(1) for a in range(1, CA + 1)], 1)
    res["lam_counts_vs_lam"] = float(np.abs(_lam_counts(cnt) - _lam()[:, 1:CA + 1]).max())
    L = P["L"]
    zs = []
    for s in SEEDS_POOL:
        for e in range(NE):
            pc, pobs = gen_pool(s, e)
            fr = pobs.reshape(R * R, PPC * T, NO).mean(1)                                 # (100, 5)
            p = L[e][:, np.arange(R * R) // R, np.arange(R * R) % R].T
            zs.append((fr - p) / np.sqrt(p * (1 - p) / (PPC * T)))
    zs = np.concatenate(zs).ravel()
    res["pool_z"] = dict(n=int(zs.size), max_abs=float(np.abs(zs).max()), share_gt3=float((np.abs(zs) > 3).mean()),
                         mean=float(zs.mean()), sd=float(zs.std()))
    fr = P["obs"].reshape(NE, R * R, 2 * T, NO).mean(2)
    p = np.stack([L[e][:, np.arange(R * R) // R, np.arange(R * R) % R].T for e in range(NE)])
    zp = ((fr - p) / np.sqrt(p * (1 - p) / (2 * T))).ravel()
    res["probe_z"] = dict(n=int(zp.size), max_abs=float(np.abs(zp).max()), share_gt3=float((np.abs(zp) > 3).mean()),
                          mean=float(zp.mean()), sd=float(zp.std()))
    conds = [f"bm_net_{seed}", f"flat_{seed}"]
    lg, info = ctl_forward("trained", ep, conds)
    z = _load(ctl_path("trained", ep))
    zc = [str(c) for c in z["conds"]]
    for c in conds:
        cum = np.cumsum(lg[c].astype(np.float64), 1).reshape(len(lg[c]), T, NV, R)
        rt = cum.argmax(-1)[:, -1] == P["r"]
        res[f"rederived_{c}_answers_differ"] = int((rt != z["right_T"][zc.index(c)]).sum())
    res["rederived_tv_max_abs_diff"] = float(np.abs(info[f"tv_{seed}"] - z[f"tv_{seed}"]).max())
    cell = ti.probes()["cell"]
    env = P["env"]
    code = _codes(P["obs"])
    bad_cell, same_hist, wrong_cnt = {}, {}, 0
    rng = np.random.default_rng(5)
    brute = []
    Lam = _lam()
    for nm in ("bm", "fl"):
        idx = info[f"{nm}_idx_{seed}"].astype(np.int64)
        bc, sh = np.zeros(CA, np.int64), np.zeros(CA, np.int64)
        for e in range(NE):
            rows = np.flatnonzero(env == e)
            pc, pobs = gen_pool(seed, e)
            pcode = _codes(pobs)
            for a in range(1, CA + 1):
                j = idx[rows, a - 1]
                bc[a - 1] += int((pc[j] == cell[rows]).sum())
                sh[a - 1] += int((pcode[j, :a] == code[rows, :a]).all(1).sum())
                wrong_cnt += int((pobs[j, :a].sum(1) != info[f"{nm}_cnt_{seed}"][rows, a - 1]).any(1).sum())
        bad_cell[nm] = {f"a{a + 1}": int(v) for a, v in enumerate(bc) if v}
        same_hist[nm] = {f"a{a + 1}": int(v) for a, v in enumerate(sh) if v}
        lqb = bayes_cut_base(_lam_counts(info[f"{nm}_cnt_{seed}"]))
        for i in rng.choice(len(env), n_brute, replace=False):
            pc, pobs = gen_pool(seed, env[i])
            Le = L[env[i]]

            def lik(o):                                                                # explicit history -> joint
                return (np.einsum("ti,ijk->jk", o.astype(np.float64), np.log(Le))
                        + np.einsum("ti,ijk->jk", 1 - o.astype(np.float64), np.log1p(-Le)))
            cum = np.zeros((NV, R))
            for t in range(1, T + 1):
                if t <= CW:
                    inc = _marg(Lam[i, t]) - _marg(Lam[i, t - 1])
                else:
                    a = t - CW
                    don = pobs[idx[i, a - 1], :a]
                    h1 = np.concatenate([don, P["obs"][i, a:t]])
                    h0_ = np.concatenate([don, P["obs"][i, a:t - 1]])
                    inc = _marg(lik(h1)) - _marg(lik(h0_))
                cum = cum + inc
                brute.append(float(np.abs(cum - logsumexp(cum, -1, keepdims=True) - lqb[i, t - 1]).max()))
    res["donor_same_cell"] = bad_cell
    res["donor_same_history"] = same_hist
    res["donor_counts_mismatch"] = wrong_cnt
    res["bayes_pool_bruteforce_max_abs"] = max(brute)
    res["seconds"] = time.time() - t0
    with open(os.path.join(CTL, "checks.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "doc"
    if what == "bayes":
        bayes_cache(force=True)
    elif what == "harvest":
        harvest(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif what == "worker":
        worker(int(sys.argv[2]), int(sys.argv[3]))
    elif what == "one":
        run_ck(sys.argv[2], int(sys.argv[3]), force=True)
    elif what == "checks":
        checks()
    elif what == "where":
        where_harvest(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif what == "where_worker":
        where_worker(int(sys.argv[2]), int(sys.argv[3]))
    elif what == "bayes_ctl":
        bayes_ctl(force=True)
    elif what == "bm_null":
        bayes_bm_null(force=True)
    elif what == "controls":
        ctl_harvest(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    elif what == "ctl_worker":
        ctl_worker(int(sys.argv[2]), int(sys.argv[3]))
    elif what == "ctl_one":
        ctl_ck(sys.argv[2], int(sys.argv[3]), force=True)
    elif what == "ctl_checks":
        ctl_checks()
    elif what == "numbers":
        numbers()
    elif what == "figure":
        figure_answers()
        figure_where()
    else:
        print(__doc__)
