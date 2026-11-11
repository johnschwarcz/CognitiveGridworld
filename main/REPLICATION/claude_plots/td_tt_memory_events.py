"""Memory use at the behavioural events through training, from the STORED memory cuts (John 11-02: "Can we identify the
neural changes that precede [the behavioural developments]? How does the network get to the correct answer before
learning optimal bayes dynamics? ... Ideally ... a causal account"). The memory cut is an intervention on the network's
state at a fixed checkpoint; here it is grouped by Bayes's behavioural events (holding a split, coupled moves,
entangled steps) and timed against the behaviour rows of td_tt_timeline on the same paired resamples. No new forward
passes. Revised 11-02 after an adversarial check (FIXABLE): controls matched on Bayes's update size too, the entangled-
steps date shown to depend on the memoryless reference, the family comparison made ceiling-free.
-> main/REPLICATION/figure_bin/td_tt_memory_events.png   figure()
   numbers()   every number below
   caches main/DATA/experimental/training_dynamics/through_training/memory_events/: memuse_<run>.npz (trained, echo:
   per checkpoint and environment, sums of memory use, intact and cut errors over every event set, its covered events
   and its weighted matched control under every matching key (memory use only for the robustness keys); for the
   Trained RNN the errors of two memoryless linear rules and the per-step regression coefficients; counts; memory-free
   Bayes's error and Bayes's update size sums), trunc_<run>.npz (six runs: per checkpoint and environment, sums of the
   excess at T per family and window, the losses, accuracy, the counts right at T / right and wrong under the w5 cut /
   Bayes right / Bayes right and windowed Bayes 5 wrong per family, the network's own revisions, the clock surrogate;
   per-step gains). Every curve and resample is rebuilt from these sums in seconds (ratio()).

REVISED 11-10 (factorization regret is the paper's forward KL everywhere). The matched controls' key contains the
  variable's FR. Until 11-10 that was the probes' symmetric KL, [KL(Bayes || naive) + KL(naive || Bayes)] / 2 with
  probabilities clipped at 1e-15; it is now KL(Bayes || naive), exact (main.utils.factorization_regret = probes.npz
  'fr'). memuse_<run>.npz and rematch_trained.npz were rebuilt (the old ones are kept as <name>_sym_superseded.npz;
  trunc_<run>.npz do not involve FR), the figure redrawn and every number that involves a matched control re-measured
  (numbers()); numbers of the events themselves, of the keys without FR (step x update size, step only) and of the
  excess at T are unchanged. What moved (symmetric -> forward):
  - Holding: beats matched steps from 2,377 [2,026-3,258] -> 2,397 [2,042-3,310]; events - control at the end +.076
    -> +.067; coverage .976 -> .956. Coupling: still never beats update-matched steps (-.032 -> -.036 [-.048, -.024]);
    the step before a large move beats its control from 4,473 [3,223-6,518] -> 6,709 [5,838-16,986].
  - CHANGED, finding 4: the most entangled fifth's stored-cut contrast turns positive for good at 388 [262-625]
    instead of 709 [496-776], i.e. before passing naive (835) in 100% of the paired resamples under every key that
    contains FR (387-549), where the symmetric FR gave 'just before or with' (706-744; paired share .949-.985). The
    keys without FR still give 901 / 922 (with passing naive). The contrast is within .0035 nats of 0 at 280, 300,
    400, 560, 640 and 720 under either FR (-.0031 to +.0002 symmetric, -.0011 to +.0033 forward), so its crossing is
    that sensitive to the control; the finding's conclusion (the stored data do not date when memory becomes
    relatively useful at entangled steps) stands, with a wider range of dates (60-3,705 over the memoryless
    references, 387-922 over the matching keys).
  - ESN, most entangled fifth: hurts slightly less than at matched steps, +.004 [+.002, +.006] -> +.008 [+.006,
    +.010], crossing 987 [376-3,759] -> 494 [298-1,364].
  The 11-02 independent check (CHECKS) covered the symmetric-FR numbers; it was not repeated.

DATA
  analysis/ent_memcut_var_<run>.npz (trained, echo; td_entanglement_memcut.py): err_int, err_cut1 (114 checkpoints,
  10,000 probes, 30 steps, 2 variables) float16. analysis/<run>/ck_EEEEE.npz trunc_right (10,000, [w1, w2, w5, intact],
  2) (td_analysis an6) for trained, trained_p1, trained_N200, trained_N100, trained_N50 and echo (the null), with logq,
  right and reg_* (the step-t logit increment regressed on N, W = SW - N, M = D - SW). analysis/probes.npz (50 test
  environments x 100 latent pairs x 2 trajectories, 30 steps; Bayes, naive and windowed-Bayes references; inc_N,
  inc_SW). The behaviour rows and their resampled crossings: td_tt_timeline.load_rows() (wormholes, jumps; the same
  1,000 environment resamples).

DEFINITIONS
  Update error (td_entanglement_memcut): err = Jeffreys/2 between Bayes's belief at t-1 moved by the network's step-t
    logit increment and Bayes's belief at t, per probe, step and variable. Intact: the network's own increment. Cut:
    the increment of the cell rerun from its trained start state over input t alone (memory cut to the last step).
    Memory use u = err_cut - err_intact (nats; > 0: the state helps the step-t update). Step 1: cut = intact.
  Memory-free Bayes's error u_Bayes (Bayes's own memory use, as Bayes's intact error is 0): the step-wise increment SW
    (the single-step likelihood marginalised over the other variable) applied to Bayes's belief at t-1, Jeffreys/2
    against Bayes's belief at t. Bayes's update size: Jeffreys/2 between Bayes's marginals of the variable at t-1 and t.
  Memoryless linear rules (clock_errors(); Trained RNN): the network's per-step regression coefficients on naive's
    increment N and the step-wise correction W = SW - N, applied to Bayes's belief at t-1 and scored like err: the
    step-1 rule uses the step-1 coefficients at every step (what the start-state cut runs, as a linear rule), the
    clock-matched rule the step-t coefficients (the network's own gain at that step, without its memory part M).
  Event sets (variable-steps, steps 2-30, all defined on Bayes's beliefs, fixed through training):
    (a) holding events: Bayes's marginal of the variable is end-split (td_tt_wormholes: e_v = min(pi(0), pi(9)) > .15
        and pi(0) + pi(9) > .6). Control candidates: variable-steps of stream-steps where neither variable is split.
    (b) coupling events: the update into each of Bayes's large diagonal moves (td_tt_wormholes / td_tt_jumps: the
        posterior-mean move mu_t - mu_(t-1) of 2+ cells at an angle >= 30 deg from an axis), both variables; the
        memory removed is that of the steps before the move. Control candidates: variable-steps with no Bayes move of
        2+ cells into the step. Checks: the step before the move (t - 1); contrast: Bayes's large one-axis moves (2+
        cells, <= 15 deg).
    (c) entanglement fifths: the variable's own entanglement (probes.npz ent), fifths within each step over both
        variables (4,000 per fifth and step): most (q5) and least (q1) entangled. Control candidates: the other fifths.
    Matched controls: candidates weighted to the event set's histogram of the key step x log10 FR of the variable
    (naive's error, .25 bins) x Bayes's entropy of the variable (.2 nats) x log10 Bayes's update size of the variable
    (.25 bins; match_keys()). FR = KL(Bayes || naive) of the variable's marginal belief, the paper's forward form,
    exact (main.utils.factorization_regret = probes.npz 'fr'); until 11-10 it was the symmetric KL clipped at 1e-15
    (REVISED 11-10 above: every number that involves a matched control was re-measured). The update size was added
    after the check found Bayes's update 3.7x larger at the large
    moves than at their FR x entropy-matched steps; the first build's key (without it; td_tt_wormholes' control 2) and
    variants (update bins .5 / .125, step x update size only, FR and entropy bins halved / doubled, step only) are kept
    as checks. 'Beats the control' uses the covered events (events whose bin has candidates).
  Long-memory excess at T (per probe, goal variable): x = (R_intact - R_cut5) - (R_Bayes - R_Bayes5): R = right at T;
    cut5 = every step's increment from the cell rerun from the start state over the last 5 inputs, summed
    (td_analysis an6); Bayes5 = windowed Bayes over the last 5 observations (the same construction for Bayes). Also w1,
    w2. Ceiling-free version per family: x_c = P(cut5 wrong | the network right at T) - P(Bayes5 wrong | Bayes right
    at T) (the w5 cut can only break answers that are right, and the network is right less often on some families).
    Episode families (goal variable; defined on Bayes's and naive's beliefs, so the network and Bayes are compared on
    the same probes): Bayes revises = more than .6 on a value at some step 8-24 that is not Bayes's answer at T; Bayes
    settled = Bayes's answer the same at every step 8-30; neither; naive far error = naive's answer at T 2+ cells off
    (the folds); settled matched = settled probes weighted to the revisions' histogram of Bayes's confidence at T (.05
    bins). The preliminary's family (each observer's own trajectory): more than .6 on a wrong value at some step 8-24
    and right at T; enrichment = its share of the w5 loss / its share of the probes.
  Statistics: sums per environment (env = probe // 200), pooled as ratios of sums; 1,000 resamples of the 50
    environments (td_tt_gap.boot_counts, seed 20261103: paired with every probe-based behaviour row and with the
    wormholes forward set). Crossings: td_tt_gap.crossing, the training episode after which a curve stays past a level
    to the end, log-interpolated; levels (natural references, no naive/Bayes gap): 0 (memory helps, the memoryless
    level / beats its control, for good); u / u_Bayes = .1 and .5 (the reference Bayes's own memory use, for good);
    half its own end value (for good; after the Bayes-referenced ones); the excess rises and falls: timed on its rise
    (half its peak = the 3-checkpoint running maximum, first reached) and its fall (below half its peak for good),
    levels recomputed in every resample. [5-95%]; two events ordered only when their intervals do not overlap ("with"
    otherwise); paired share = P(a < b) over the same resamples. 'Per resample' family bounds (the earliest / latest
    row of a family in each resample) are labelled as such; a row's own interval is quoted with the row. x: training
    episodes (raw - raw // 5), the initial network at 10.

RESULTS (Trained RNN unless named; training episodes; [5-95%]; paired share in parentheses; controls update-matched)
 Behaviour anchors (td_tt_timeline): accuracy passes naive 835 [798-876], c10 885 [847-952], c50 1,394 [1,285-1,489];
   holding c10 1,166-2,127, c50 3,021-9,245; coupling c10 213-3,255, c50 2,747-11,103.
 1 Holding events (2,489 variable-steps, median step 9; Bayes's own memory use .109, update size .145). Memory HURTS
   the network's update at Bayes's splits through the answers phase: about -.24 over 320-640 (mean -.243, range -.274
   to -.231; the trough -.325 [-.381, -.261] at 300 is one checkpoint: 280 -.198, 320 -.232), -.131 (880), -.089
   (1,120), -.032 (1,600), while at matched steps it is -.043 (400) / -.037 (640) / -.015 / -.002 / +.017. It helps
   for good from 1,779 [1,638-2,052] (the control from 1,146 [937-1,336]), reaches half of memory-free Bayes's use
   (u / u_Bayes > .5) at 2,705 [2,362-3,225] (> .1: 1,993 [1,720-2,297]; half its own end value 3,234 [2,716-3,325])
   and ends +.143 [+.116, +.166] (1.31 x Bayes's). It beats matched steps for good from 2,397 [2,042-3,310] (coverage
   .956, update size .135 vs .133; end +.067 [+.043, +.089] over the control's +.072 [+.065, +.080]; memory-free Bayes
   .106 vs .056, +.049); update bins .5 / .125: 2,396 [2,049-3,300] / 2,392 [2,031-3,294]; step x update size only
   1,991 [1,675-2,339]; the first build's key 2,337 [1,969-2,842] (end +.087 [+.064, +.110]). Orders: helps after the
   answers (accuracy c50; 0.000) and after the earliest holding onset (per resample [1,010-1,544]: 0.001; the
   end-mass slope row 1,166 [1,020-1,556]), with the other four onsets (held per variable .117 / .018, distance out
   .027, joint held .955), before every holding half-way (the earliest, distance out 3,021 [2,815-3,511]: 1.000) and
   the earliest coupling half-way (same-diagonal share 2,747 [2,111-3,447]: .996). Beats: after every holding onset
   but the joint-held one (2,127 [1,930-2,509]: with, .141), with the earliest holding half-way (.743) and before the
   other four (.982-1.000), with the earliest coupling half-way (.545). Half of Bayes's use (2,705): with the earliest
   holding half-way (.883), before the other four (.999-1.000), with the earliest coupling half-way (.514). What flips:
   the cut error at splits stays .16-.25 from 80 to 4,000 (.19-.25 over 400-4,000: .193 at 400, .233 at 2,000, .231 at
   4,000; .152 at the end) while the intact error falls .436 (400), .457 (640), .272 (1,600), .222 (2,000), .129
   (4,000), .012 (end): up to 4,000 every gain at Bayes's splits is carried by the state; the memoryless rerun improves
   only late. At matched steps the intact error falls .184 (400) -> .006 and the cut error .141 -> .077. Steps 2-16 only
   (td_tt_wormholes' window; 1,923): helps 1,760 [1,519-2,032], beats 3,251 [2,089-4,450] (first key 2,353
   [1,966-3,279]).
 2 Coupling events (6,458 variable-steps = 3,229 moves; Bayes's own memory use .145, update size .526). Same shape,
   smaller dip: the trough -.095 [-.108, -.081] at 300 is one checkpoint (280 / 320: -.061), -.081 over 320-640 (range
   -.090 to -.061; -.080 at 400, -.083 at 640, -.048 at 1,120 vs matched steps -.027 / -.023 / +.006). Helps for good
   from 1,974 [1,623-2,122] (the control from 1,206 [809-1,256]); half of memory-free Bayes's use at 3,943 [3,814-4,400]
   (> .1: 2,175 [1,997-2,422]; half its own end value 3,986 [3,853-4,467]); end +.149 [+.144, +.156]. It NEVER beats
   matched steps once Bayes's update size is matched: with the first build's key it beat them from 2,554 [2,149-3,264]
   (end +.081 [+.074, +.088]), but there Bayes's update at the events is 3.7x the controls' (.526 vs .144) and
   memory-free Bayes's need is 2.6x (.145 vs .055). Matched on update size (coverage .853; update .466 vs .435) the
   events minus control end at -.036 [-.048, -.024] and never cross (0% of resamples), as memory-free Bayes's need is
   lower there too (.123 vs .155, -.032; the control ends +.165 [+.155, +.175]); update bins .5: -.017 [-.027, -.007],
   never (0%); .125: -.038 [-.050, -.026], never (0%); step x update size only: -.008 [-.026, +.010], never (26%).
   Memory use at large moves exceeds FR x entropy-matched steps only because Bayes's update is larger there. Orders
   (helps): after accuracy c50 (0.000) and the jump-coupling onsets (same-diagonal 213, >= 4 cells 740, G 1,046: 0.000;
   the excess angle's 1,032 [735-1,684] overlaps: with, .009), before the angle-out and direction-slope onsets (2,609,
   3,255: .994 / 1.000), with the earliest coupling half-way (same-diagonal share 2,747 [2,111-3,447]: .992) and before
   the other five (1.000). Half of Bayes's use (3,943): after the earliest holding and coupling half-ways (0.000).
   Intact .539 (400) -> .327 (2,000) -> .020 (end); cut .458 -> .329 -> .170 (here the memoryless rerun improves too,
   more slowly). The step before the move (3,972): helps 1,957 [1,641-2,170], beats matched steps from 6,709
   [5,838-16,986] (first key 3,267 [3,201-4,465]); end +.014 [+.005, +.023] (memory-free Bayes .099 vs .086).
 3 FAILED (coupling-specific memory): at Bayes's large ONE-AXIS moves (1,152 variable-steps; Bayes's own .168, update
   size .435) memory turns helpful on the same schedule (2,299 [1,784-2,963] vs 1,974) and ends as high (+.157 [+.139,
   +.178] vs +.149). With the first build's key it also beat its control (3,214 [2,040-4,050]; end +.104 vs diagonal
   +.081); matched on update size (coverage .905; .400 vs .367) the contrast vanishes as at diagonal moves: +.003
   [-.018, +.024] at the end, crossing in only 63% of resamples (update bins .5: +.027 [+.006, +.049], 15,511
   [7,436-29,711] (98% cross); .125: -.004 [-.024, +.017], never (40%); step x update size: +.020 [-.001, +.042],
   19,388 [11,842-39,767] (95% cross)). Memory becomes useful at Bayes's large moves of any direction, and not beyond
   matched steps of the same Bayes update size.
 4 Entangled steps (q5 115,998, q1 116,000 variable-steps; Bayes's own memory use .054 / .007, controls .019 / .020;
   Bayes's update size .077 / .031). The stored-cut contrast at the most entangled fifth (events minus matched steps)
   turns positive for good at 388 [262-625] (the first build's key: 424 [281-639]), before passing naive (835
   [798-876]) by the interval rule under every key that contains FR: the primary key (paired share 1.000), the first
   key (1.000), update bins .5 (387 [261-580], 1.000) and .125 (390 [273-634], 1.000), FR / entropy bins halved (423
   [325-655], 1.000) and doubled (549 [388-716], 1.000) and the control re-matched in every resample (388 [270-663],
   1.000; build_rematch()); with it under the keys without FR: step x update size (901 [792-974], .114) and step only
   (922 [836-1,008], .016). (With the symmetric FR, until 11-10, the FR keys gave 706-744, 'just before or with passing
   naive': REVISED 11-10.) The contrast is small and flat where it crosses (+.0005 [-.0012, +.0022] at 400, +.0022
   [-.0002, +.0045] at 560, +.0033 [+.0010, +.0057] at 720), so the date moves with the matching (387-922 over the
   keys). It IS ALSO SET BY THE START-STATE CUT'S STEP-1 GAIN, not by an improvement of the intact network:
   from 400 to 880 the contrast rises from +.0005 [-.0012, +.0022] to +.0047 [+.0026, +.0067] because the cut's error
   gap (q5 minus matched steps) grows by +.0098 (.0144 -> .0242) while the intact gap grows too, by +.0057 (the intact
   network gets relatively worse there; .0138 -> .0195; first key: cut gap .0351 -> .0422, +.0070; intact gap .0357 ->
   .0367 (.0373 at 720), +.0010; contrast -.0005 [-.0026, +.0016] -> +.0055 [+.0036, +.0073]). The cut's gap follows the
   step-1 linear rule's within .0028 nats over 280-2,080 (first key .0030), and the network's step-1 gain on naive's
   increment (single-regressor slope, ck reg_*) rises with the later steps' gains (.215 -> .405 vs .514 -> .704 for
   steps 2-30 over 400-880; the step-1 deficit stays about -.30). Against the other memoryless references the date
   moves: clock-matched rule 60 [40-66], step-1 rule 408 [281-664], stored cut 388, the cut gap frozen at its 400 level
   3,705 [2,694-4,873] (first key: 91 [76-114], 643 [438-737], 424, 1,171 [725-1,625]). So the stored data do not
   date when memory becomes relatively useful at entangled steps; the date is deferred to the donor-state memory cuts of
   td_tt_memcuts.py (in progress 11-02). The earlier flip scan of memory use by entanglement used the same stored cut
   and carries the same caveat. Memory at q5 helps for good from 1,163 [1,061-1,306] (with accuracy c50; earlier in 100%
   of the paired resamples), reaches half of memory-free Bayes's use at 2,238 [2,079-2,418] (> .1: 1,350 [1,245-1,435];
   half its own end value 2,471 [2,323-2,698]) and ends +.063 [+.058, +.068] vs control +.024 [+.023, +.026] (+.034
   [+.031, +.037]); q5 intact .108 (400) -> .005, cut .085-.099 from 400 to 6,400 (.067 at the end). The least entangled
   fifth never beats its control (-.014 [-.015, -.013] at the end; 0% of resamples; first key -.021 [-.022, -.020]) and
   helps only from 5,213 [4,419-5,783] (end +.009; control +.022).
 5 ESN: memory hurts for good from 174-205 at every event set (splits 174 [161-188], diagonal moves 192 [177-206],
   one-axis 205 [170-417], q5 184 [174-198]; q1 1,844 [190-2,313]) and more and more (end: splits -.128 [-.157,
   -.102], diagonal moves -.104, one-axis -.054, q5 -.015, q1 -.004); it never helps for good (0% of resamples). Its
   intact error at splits RISES .197 (400) -> .349 (end) (cut .166 -> .221). Matched on update size, its memory hurts
   slightly less at the most entangled fifth than at matched steps (+.008 [+.006, +.010] at the end; crossing 494
   [298-1,364]; first key +.006 [+.004, +.008], 940 [285-3,677]); at q1 -.002 [-.002, -.001], never (0%).
 6 At the end the Trained RNN's memory use follows memory-free Bayes's need for memory (u / u_Bayes at the end: splits
   1.31, diagonal moves 1.03, q5 1.17, q1 1.23), and so do the matched contrasts on the covered events: splits +.138 vs
   control +.072 (memory-free Bayes .106 vs .056), diagonal moves +.128 vs +.165 (-.036; Bayes .123 vs .155), q5 +.058
   vs +.024 (.051 vs .019), q1 +.008 vs +.022 (.007 vs .020). The network's memoryless reads err more than memory-free
   Bayes's (at splits .152 vs .109), as its w1 excess at T ends at +.134.
 7 The long-memory excess at T (w5). Trained RNN: rises just after passing naive (half its peak first reached at 905
   [845-1,071]: with passing naive by the interval rule, after it in 99% of the paired resamples (P(rise earlier)
   .013); with accuracy c10 (.338), before accuracy c50 (1.000), with the earliest holding onset per resample (.964),
   after the same-diagonal onset (.005)), peaks +.127 [+.117, +.141] at 2,080 (3-checkpoint running mean; largest
   single checkpoint +.130 at 1,760; the network's w5 loss .156 vs Bayes's .031), falls below half its peak for good
   at 6,698 [5,900-7,831] (after the earliest holding and coupling half-ways (0.000); with the latest holding half-way
   per resample ([7,701-10,112]: .983; the joint-held row 9,245 [7,693-10,112]: .983) and with the excess-angle and G
   half-ways (.219 / .046); before the latest coupling half-way per resample ([9,566-12,625]: 1.000; the angle-out row
   11,103 [8,969-12,555]: .998)), and ends at Bayes's level, +.0071 [+.0012, +.0135]. Values: -.011 (400), +.038 (800),
   +.089 (1,120), +.110 (1,600), +.127 (2,000), +.113 (3,200), +.098 (4,000), +.066 (6,400), +.033 (12,000), +.004
   (end). w2: peak +.288 at 2,080, rise 860 [827-904], fall 25,732 [19,587-29,818], end +.104; w1 excess +.249
   (1,600), +.290 (2,400), +.134 (end): the network's short-window reads stay worse than windowed Bayes's. Other runs
   (same world and probes): power-1 loss: passes naive 1,256 [1,207-1,356], rise 1,395 [1,200-1,462] (with by the
   interval rule; after passing naive in 90% of the paired resamples, P(rise earlier) .102), peak +.115 at 2,160, fall
   9,817 [8,947-11,367], end +.007; N = 200: passes naive 2,070 [1,966-2,424], rise 1,513 [1,302-1,861] (before,
   .999), peak +.089 at 5,200 (+.083 / +.084 / +.084 at 3,200 / 4,000 / 6,400; its accuracy c50 6,311
   [5,485-7,815]), fall 27,024 [24,923-33,636], end +.029; N = 100: passes naive 6,271 [5,444-7,662], rise 5,561
   [3,184-6,450] (with, .847), peak +.042 at 13,600, no fall (5% of resamples; end +.034); N = 50: passes naive 21,072
   [14,871-30,271], peak +.021 [+.011, +.038] at 4,800 (rise 3,034 [2,757-3,399], before), below half its peak for good
   at 37,055 [5,471-39,795] (98% cross) although its accuracy never gets half-way to Bayes (0% of resamples), end +.006.
   Peak size follows width (.127 / .115 / .089 / .042 / .021); the excess rises with or before passing naive in every
   trained run (just after it in most resamples for the Trained RNN and power-1) and relaxes only in the runs that
   approach Bayes, except N = 50 (its small peak also falls below half, late). ESN: negative at every checkpoint (at
   most -.013; end -.029 [-.037, -.022]): its w5 cut breaks about as many answers as it fixes (.046 / .045 of the probes
   at 800, .045 / .045 at the end; Bayes's window .064 / .034).
 8 FAILED (answers via long-memory RE-EVALUATION): the excess is not concentrated on revisions. Bayes-defined families
   (n 2,685 / 2,844 / 4,471 / naive far 2,430; Bayes's own w5 loss .047 / .010 / .034 / .024). Unconditionally,
   revisions carry less excess than settled probes: revises - settled -.046 [-.072, -.019] (880), -.083 [-.108, -.058]
   (1,120), -.037 [-.064, -.012] (1,600), -.038 [-.060, -.016] (2,000), -.046 [-.069, -.021] (2,400), -.011 [-.034,
   +.010] (4,000); against settled probes matched to the revisions' Bayes confidence at T (coverage .999): +.003 [-.028,
   +.034], -.042 [-.075, -.011], -.001 [-.031, +.027], -.000 [-.032, +.032], -.021 [-.055, +.014], -.002 [-.035, +.031].
   That 'less' is a ceiling effect: the w5 cut can only break right answers, and the network is right less often on
   revision probes (.444 vs .544 at 800, .558 vs .712 at 1,120, .645 vs .796 at 1,600, .681 vs .839 at 2,000).
   Ceiling-free (x_c; Bayes's term .113 on revisions, .014 settled, .044 settled matched, .083 all), revises, settled
   and all probes sit at .25-.31 at 800-1,120 (revises .299 / .303 / .248, settled .304 / .294 / .288, all .296 / .305 /
   .296 at 800 / 880 / 1,120) and revises - settled is -.005 [-.039, +.030] (800), +.009 [-.023, +.041] (880), -.039
   [-.065, -.015] (1,120), +.017 [-.014, +.046] (1,600), +.025 [+.001, +.050] (2,000), +.013 [-.012, +.040] (2,400),
   +.017 [-.006, +.039] (4,000); against confidence-matched settled probes +.009 [-.035, +.050] (800), +.002 [-.033,
   +.036] (880), -.079 [-.110, -.048] (1,120), -.029 [-.063, +.002] (1,600), -.015 [-.051, +.020] (2,000), -.041 [-.079,
   -.002] (2,400), -.055 [-.084, -.026] (4,000). Ceiling-free the difference is small and its sign depends on the
   comparison: revisions carry slightly more than all settled probes at about half the checkpoints from 1,520 on (5%
   bound above 0 at 44 of the 89 checkpoints from 800, at most +.041 [+.018, +.063] at 4,600) and less than settled
   probes matched to the revisions' Bayes confidence (95% bound below 0 at 38 of the 89, 5% bound never above 0), so the
   excess is not concentrated on revisions. Naive's far-error probes carry about the average unconditional excess in the
   answers phase (+.095 vs +.089 at 1,120; +.117 vs +.127 at 2,000) and more later (+.140 vs +.113 at 3,200; +.064 vs
   +.033 at 12,000; end +.017 [+.005, +.029]; fall 10,732 [7,860-16,838]); ceiling-free they carry more at every
   checkpoint from 320 on (5% bound above 0 at 105 of the 114 checkpoints; naive far - all +.215 [+.167, +.262] at 400,
   +.165 [+.129, +.205] at 800, +.170 [+.134, +.209] at 1,120, +.084 [+.058, +.113] at 2,000, +.018 [+.004, +.033] at
   the end), but this is not interpreted: the network is right on few of them (.273 vs .441 at 800), and x_c counts only
   broken answers, so it is not a clean null where accuracy is low (the ESN's x_c is +.136 / +.088 / +.022 at 800 /
   2,000 / the end, at accuracy .212 / .287 / .431, while its total excess is negative). The preliminary's revision
   concentration (at 1,600 the w5 loss is .391 on the network's own revisions vs Bayes's .038 on the same probes; .141
   vs .031 on all probes) selected probes on the network's own trajectory: under the same rule Bayes's own revisions
   (.210 of the probes, .78 of its w5 loss) cost Bayes .113 per probe, not .038 (enrichment 3.7x). The network's own
   revisions (share of probes .074 / .127 / .165 / .179 / .197 / .211 / .205 at 800 / 1,120 / 1,600 / 2,000 / 4,000 /
   12,000 / the end) carry .57 / .51 / .46 / .44 / .46 / .64 / .87 of its w5 loss: enrichment 7.7x / 4.0x / 2.8x / 2.4x
   / 2.3x / 3.0x / 4.2x. Mixed: more concentrated than Bayes's at 800, about as concentrated at 1,120, less from 1,600
   to 4,000, more again at the end. The 'spread over episodes' conclusion rests on the Bayes-defined families,
   unconditional and ceiling-free.
 How the network gets the answers before Bayes's dynamics (from these interventions): through the answers phase
 (accuracy passes naive at 835 and is half-way at 1,394) its state makes its local updates at Bayes's split and
 large-move steps WORSE than the memoryless rerun's (about -.24 at splits over 320-640), while its answers at T
 depend on memory older than 5 steps 2-5x more than Bayes's do (w5 loss .069 at 800, .119 at 1,120, .141 at 1,600 vs
 Bayes's .031), on revisions and settled episodes alike. The intact update at Bayes's splits and large moves beats
 the memoryless rerun from 1,779 / 1,974, with the holding onsets and before holding is half-way; that is a
 behaviour-referenced crossing (see CAVEATS), and against the network's own linear memoryless rules it comes later
 (splits 2,790 / 3,567, large moves 3,383 / 4,704). Matched on Bayes's update size, memory at splits beats matched
 steps from 2,397 (with the earliest holding half-way) and reaches half of memory-free Bayes's use at 2,705; at large
 diagonal moves it never beats update-matched steps. The excess long-memory dependence relaxes while holding and
 coupling complete (6,698). At entangled steps the stored-cut contrast turns positive before passing naive under the
 FR-matched controls (388; 387-549 over the FR keys) and with it under the controls without FR (901-922), and its
 date is set by the cut's step-1 gain (60-3,705 against other memoryless references); dating it is left to the
 donor-state cuts.

START-STATE CONFOUND (and what a donor-state cut adds)
  Every stored cut restarts the cell from the trained start state: the cut state at step t is the state of steps 1..w
  of an episode whose observations are o_(t-w+1..t). Observations are i.i.d. given (environment, cell), so the cut
  increment has exactly the distribution of the intact step-w increment: the cut substitutes the network's own step-w
  rule (a regime every episode passes through, not an off-manifold state) and removes the state's clock (time since
  the start: norms, gate saturation) together with its content (older evidence). Windowed Bayes is the same
  construction for Bayes. Stored-data bounds on the clock: (i) w5 cut: a content-free clock (the step-mean increment
  over all probes, per variable and value, shifted from step 5 to steps 6-30; clock_right()) costs .0038 [.0003,
  .0076] accuracy at the excess peak against the w5 loss .156 (largest over checkpoints .023 at 14,400; Bayes's own
  clock-only loss .0022); the gain on naive's increment (ck reg_*: Sigma d N / Sigma N N) is flat over steps 5-30 (at
  2,080: step 5 .76, steps 6-30 .78; end .67 / .63). So the excess is not an additive clock or a gain clock in naive's
  direction. (ii) per-step (w1) cut: the step-1 gain differs from later steps: step 1 minus the mean of steps 2-30
  -.30 (400), -.30 (880), -.12 (2,080), +.33 (end), above 0 from 3,308 (checkpoint resolution). Through the answers
  phase the cut's updates are therefore MORE cautious in naive's direction than the intact ones; late the step-1 gain
  (.98) exceeds the later gain (.63), mirroring Bayes's own (D on N: .96 at step 1, .73 at steps 6-30; memory-free SW
  .97 at every step), so late memory-use magnitudes include a Bayes-like clock. (iii) linear memoryless rules from the
  network's own regression coefficients (clock_errors()): at Bayes's splits both the step-1 rule and the clock-matched
  rule err far less than the intact network early (.127-.159 and .134-.148 over 400-2,000 vs .222-.457; at large
  moves .261-.404 and .204-.302 vs .327-.539), so the early harm at splits is not a gain-clock effect; the intact error
  goes below the rules' for good only at 2,790 [2,461-3,212] / 3,567 [3,184-4,182] (large moves 3,383 [2,789-3,876] /
  4,704 [4,008-5,470]; q5 1,707 [1,542-1,838] / 2,259 [1,931-2,584]). At the most entangled fifth the clock's
  interaction with the event type sets the stored-cut contrast's date (finding 4). The matched controls share the step
  (same clock), so 'beats its control' removes the clock's main effect but not its interaction with the event type.
  A donor-state cut (rerun the last w inputs from a real state at t - w of (a) the same cell's other trajectory, (b) a
  random cell of the same environment at the same step, (c) the probe-mean state at that step) keeps the clock and
  replaces the content: (a) vs intact tests history beyond the cell-level belief, (b) and (c) bracket wrong vs absent
  evidence; they would tell whether the early harm at splits, the entangled-steps contrast and the T-excess come from
  the state's content or its regime. Kept per-step logits under the cut would also let every behaviour row (holding,
  coupling) be scored under the cut. Not stored here (needs new forwards; td_tt_memcuts.py, in progress 11-02).

CHECKS
  The event sets reproduce the preliminary read (2,489 split variable-steps, median step 9; -.241 at 640, +.139 at the
  end). With the symmetric FR (until 11-10) the first build's key reproduced its control weights exactly (identical
  per-environment counts) and its crossings (2,273 / 2,447 / 729); with the forward FR its crossings are 2,337 /
  2,554 / 424. At step 1 cut = intact (max |diff| 9.8e-04, fp16 storage). trunc 'intact' = right
  at T at every checkpoint (asserted). Accuracy c50 from the stored trunc answers equals td_tt_timeline's accuracy row
  in every resample (max |diff| 1.2e-11). The check's independent code (session scratch verify_memory_events, 11-02)
  reproduces the update-matched holding 2,377 [2,026-3,258], coupling -.032 [-.043, -.019] (never), q5 709 [496-776],
  the first key's decomposition (.0375 / .0421 at 400, .0446 / .0420 at 880) and its step-1 and clock-matched rule
  references to within one training episode (its rank-based fifths differ from the quantile ones by 2 variable-steps),
  the ceiling-free revision contrasts and the enrichment. Re-matching the update-matched controls inside every
  resample (build_rematch()) gives holding 2,377 [2,030-3,245], coupling never (-.032 [-.040, -.018]), q5 709
  [478-812]. (These matched numbers are the symmetric-FR ones the 11-02 check saw; it was not repeated for the
  forward FR, whose re-matched values are holding 2,397 [2,075-3,296], coupling never (-.036 [-.044, -.023]), q5 388
  [270-663].)

CAVEATS
  One seed and one probe world; intervals resample environments, not seeds. The per-step memory cut is stored only for
  the Trained RNN and the ESN; the other runs have the cut at T only. The update error applies the network's increment
  to Bayes's belief (a local error), not to the network's own belief. Only the answer at T is stored under the w-cuts,
  so the behaviour rows cannot be recomputed under the cut. Event sets are Bayes-defined; the update-matched controls
  cover 85-97% of the events (diagonal moves .853) with effective sizes 1,982 (one-axis moves) to 124,155. Circularity:
  memory use at Bayes's splits and the holding rows both measure the network's quality at Bayes's splits. With the cut's
  error nearly flat at splits (and falling more slowly than the intact error at large moves), 'memory helps for good' at
  splits (1,779) and at large moves (1,974) marks when the intact network's local update at those steps beats the
  memoryless one: a behaviour-referenced crossing at the same events the holding and coupling rows score, so its order
  against those rows reflects how the measures are scaled, not a neural precursor (and its date moves with the
  memoryless reference: iii above). The interventional content is that the memoryless rerun does not improve at Bayes's
  splits through 4,000 while the intact update does: the state carries the gain there. The ceiling-free family measure
  counts only broken answers (finding 8).

Run from the repository root (CPU, 2 threads; caches ~8 min with 3 processes (the re-matched contrasts ~1.5 min
of it), numbers ~2 min, figure ~2 min):
    MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
        /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_memory_events.py \\
        [build [runs]] [numbers] [figure]"""
import os
import sys
import time
import warnings
import numpy as np
from scipy.special import logsumexp

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_wormholes as wh                                                                  # noqa: E402

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
OUT = os.path.join(TD, "through_training", "memory_events")
MEM_RUNS = ("trained", "echo")                       # the per-step memory cut is stored for these two
TRUNC_RUNS = ("trained", "trained_p1", "trained_N200", "trained_N100", "trained_N50", "echo")
NAME = {"trained": "Trained RNN", "echo": "ESN", "trained_p1": "power-1 loss", "trained_N200": "N = 200",
        "trained_N100": "N = 100", "trained_N50": "N = 50"}
COL = {"trained": "#1f77b4", "echo": "#ff7f0e", "trained_p1": "#9467bd", "trained_N200": "#2ca02c",
       "trained_N100": "#8c564b", "trained_N50": "#e377c2"}
T, R, NE, PER_ENV = 30, 10, gap.N_ENV, gap.PER_ENV
N_FIFTH = 5
REV_WIN = slice(7, 24)                               # steps 8-24 (0-based 7..23): the preliminary's revision window
COMMIT = .6                                          # a commitment: more than .6 on one value
SETTLE_FROM = 7                                      # settled: Bayes's answer the same at every step 8-30
FAR = 2                                              # a far error: 2+ cells from the truth
CONF_STEP = .05                                      # Bayes's confidence at T, bins for the matched comparison
WINDOWS = (1, 2, 5)
# event sets on (probe, step, variable): the three asked for, then checks
SETS = ("hold", "couple", "q5", "q1", "hold16", "pre", "axis")
SET_NAME = {"hold": "Bayes's end splits (holding events)",
            "couple": "updates into Bayes's large diagonal moves (coupling events)",
            "q5": "most entangled fifth (own variable)", "q1": "least entangled fifth (own variable)",
            "hold16": "Bayes's end splits, steps 2-16 (check)",
            "pre": "the step before Bayes's large diagonal moves (check)",
            "axis": "updates into Bayes's large one-axis moves (contrast)"}
QTY = ("u", "int", "cut")                            # memory use = cut - intact; the intact and cut errors
CLK = ("s1", "st")                                   # two memoryless linear rules' errors (Trained RNN): clock_errors()
# matching keys of the controls (match_keys()); MATCH[0] is the primary one, 'fe' the first build's key
MATCH = ("upd", "fe", "upd5", "upd125", "step_upd", "fine", "coarse", "step")
MATCH_NAME = {"upd": "step x FR x entropy x update size (primary)", "fe": "step x FR x entropy (first build)",
              "upd5": "update size in .5 bins", "upd125": "update size in .125 bins",
              "step_upd": "step x update size only", "fine": "FR and entropy bins halved (no update size)",
              "coarse": "FR and entropy bins doubled (no update size)", "step": "step only"}
FULL_MATCH = ("upd", "fe")                           # keys stored with every quantity (the others: memory use only)
ROBUST_SETS = ("hold", "couple", "axis", "q5")       # the sets every matching variant is computed for
UPD_STEP = .25                                       # log10 bins of Bayes's update size (primary key)
X_FREEZE = 400.                                      # the 'cut gap frozen' reference: the cut gap at this checkpoint
PLATEAU = (320., 640.)                               # the harm plateau at Bayes's splits (training episodes)
FAMS =("all", "rev", "settled", "other", "nfar", "settled_m")
FAM_NAME = {"all": "all probes", "rev": "Bayes revises", "settled": "Bayes settled", "other": "neither",
            "nfar": "naive far error", "settled_m": "Bayes settled, matched to the revisions' confidence"}
HOLD_ROWS = ("end-mass slope within steps (maps)", "held per variable, one-variable end splits",
             "held per variable, corner-split ends", "distance out of a corner split", "corner split held (joint)")
COUPLE_ROWS = ("same-diagonal share of moves out of corner splits", "angle of jumps >= 4 cells",
               "jump excess angle over independent", "long-path coupling G", "coupling angle out of a corner split",
               "direction slope within steps (maps)")
ACC_ROW = "goal accuracy at T"


def _env():
    return np.load(os.path.join(AN, "probes.npz"))["env"]


# ═══════════════════════════════════════════════════════════════════
# Event sets (Bayes-defined, fixed over checkpoints) and matched controls
# ═══════════════════════════════════════════════════════════════════

def match(ev, cand, key):
    """weights on the cand var-steps that give them the ev var-steps' histogram of key, and the ev var-steps whose key
    has candidates (covered)"""
    ut, ct = np.unique(key[ev], return_counts=True)
    uc, inv, cc = np.unique(key[cand], return_inverse=True, return_counts=True)
    pos = np.minimum(np.searchsorted(uc, ut), len(uc) - 1)
    ok = uc[pos] == ut
    per = np.zeros(len(uc))
    per[pos[ok]] = ct[ok] / cc[pos[ok]]
    w = np.zeros(key.shape)
    w[cand] = per[inv.ravel()]
    return w, ev & np.isin(key, ut[ok])


def _jeff_np(a, b):
    """Jeffreys/2 between the softmaxes of a and b over the last axis (td_entanglement_driver._jeff)"""
    la = a - logsumexp(a, -1, keepdims=True)
    lb = b - logsumexp(b, -1, keepdims=True)
    return .5 * ((np.exp(la) - np.exp(lb)) * (la - lb)).sum(-1)


def match_keys(fr, H, upd):
    """the control-matching keys on (n, T, 2), name -> int64. Primary 'upd': step x log10 FR of the variable (naive's
    error, .25 bins) x Bayes's entropy of the variable (.2 nats) x log10 Bayes's update size of the variable (Jeffreys/2
    between Bayes's marginals at t-1 and t, .25 bins). 'fe': the same without the update size (td_tt_wormholes' control
    2; the first build's key). Variants: update-size bins .5 / .125, step x update size only, FR and entropy bins halved
    (fine) or doubled (coarse), step only"""
    tt = np.broadcast_to(np.arange(T)[None, :, None], fr.shape).astype(np.int64)

    def lg(v, step, floor):
        return np.floor(np.log10(np.clip(v, floor, None)) / step).astype(np.int64) + 500

    def fh(fs, hs):
        return (tt * 1000 + lg(fr, fs, 1e-6)) * 100 + np.floor(H / hs).astype(np.int64)

    fe = fh(wh.FR_STEP, wh.H_STEP)
    return {"upd": fe * 1000 + lg(upd, UPD_STEP, 1e-8), "fe": fe,
            "upd5": fe * 1000 + lg(upd, 2 * UPD_STEP, 1e-8), "upd125": fe * 1000 + lg(upd, UPD_STEP / 2, 1e-8),
            "step_upd": tt * 1000 + lg(upd, UPD_STEP, 1e-8), "fine": fh(wh.FR_STEP / 2, wh.H_STEP / 2),
            "coarse": fh(2 * wh.FR_STEP, 2 * wh.H_STEP), "step": tt}


def bayes_refs():
    """Bayes's log beliefs at t ('lj') and at t-1 ('prior', uniform at step 1), the memoryless regressors N (naive's
    increment) and W = SW - N (the step-wise correction), all (n, T, 2, R) float64"""
    P = np.load(os.path.join(AN, "probes.npz"))
    lj = P["log_jb"].astype(np.float64)
    prior = np.concatenate([np.full_like(lj[:, :1], -np.log(R)), lj[:, :-1]], 1)
    N = P["inc_N"].astype(np.float64)
    return dict(lj=lj, prior=prior, N=N, W=P["inc_SW"].astype(np.float64) - N)


def clock_coefs(z):
    """the network's step-t logit increment regressed on N, W = SW - N, M = D - SW (td_analysis reg_*, both variables,
    all values; least squares per step; M = 0 at step 1): (T, 3)"""
    return np.array([np.linalg.lstsq(z["reg_xtx"][t], z["reg_xtd"][t], rcond=None)[0] for t in range(T)])


def clock_errors(co, B):
    """update errors (Jeffreys/2 against Bayes's belief at t, as err) of two memoryless linear rules applied to Bayes's
    belief at t-1: s1 = the network's step-1 coefficients on N and W at every step (what the start-state cut runs, as a
    linear rule), st = its step-t coefficients on N and W (a clock-matched memoryless rule). (n, T, 2) each"""
    s1 = _jeff_np(B["prior"] + co[0, 0] * B["N"] + co[0, 1] * B["W"], B["lj"])
    st = _jeff_np(B["prior"] + co[:, 0, None, None] * B["N"] + co[:, 1, None, None] * B["W"], B["lj"])
    return s1, st


def event_sets():
    """dict set -> {'ev': mask, key: (w, cov)} on (n, T, 2) for every matching key computed for the set; 'bayes_err'
    (n, T, 2): memory-free Bayes's update error (the step-wise increment SW applied to Bayes's belief at t-1,
    Jeffreys/2 against Bayes's belief at t; Bayes's own memory use, as Bayes's intact error is 0); 'upd' (n, T, 2):
    Bayes's update size (Jeffreys/2 between its marginals at t-1 and t)"""
    P = np.load(os.path.join(AN, "probes.npz"))
    bB = wh.norm(P["log_jb"])
    H = wh.entropy(bB)
    fr = P["fr"]
    ent = P["ent"]
    n = len(bB)
    B = bayes_refs()
    upd = _jeff_np(B["prior"], B["lj"])
    _, split, _ = wh.ends(bB)
    st = np.zeros((n, T), bool)
    st[:, 1:] = True
    both = np.ones((1, 1, 2), bool)
    _, amp, phi = wh.bayes_jumps(bB @ wh.V)                      # (n, T - 1): the move into step t + 2 (1-based)

    def into(m):
        out = np.zeros((n, T), bool)
        out[:, 1:] = m
        return out

    big = into(amp >= wh.A_BIG)
    diag = into((amp >= wh.A_BIG) & (phi >= wh.PHI_DIAG))
    axis = into((amp >= wh.A_BIG) & (phi <= wh.PHI_AXIS))
    calm = st & ~big                                               # no move of Bayes's of 2+ cells into the step
    pre = np.zeros((n, T), bool)
    pre[:, 1:-1] = diag[:, 2:]                                     # steps 2-29 followed by a large diagonal move
    calm_pre = np.zeros((n, T), bool)
    calm_pre[:, 1:-1] = calm[:, 1:-1] & ~big[:, 2:]
    fifth = np.full((n, T, 2), -1)
    for t in range(1, T):                                          # within step, both variables pooled (20,000)
        edges = np.quantile(ent[:, t], np.arange(1, N_FIFTH) / N_FIFTH)
        fifth[:, t] = np.searchsorted(edges, ent[:, t])
    keys = match_keys(fr, H, upd)
    hold = split & st[..., None]
    nosplit = (~split.any(-1) & st)[..., None] & both
    st16 = np.zeros((n, T), bool)
    st16[:, 1:16] = True
    defs = {"hold": (hold, nosplit), "couple": (diag[..., None] & both, calm[..., None] & both),
            "q5": (fifth == N_FIFTH - 1, (fifth >= 0) & (fifth < N_FIFTH - 1)), "q1": (fifth == 0, fifth > 0),
            "hold16": (hold & st16[..., None], nosplit & st16[..., None]),
            "pre": (pre[..., None] & both, calm_pre[..., None] & both),
            "axis": (axis[..., None] & both, calm[..., None] & both)}
    S = {}
    for nm in SETS:
        ev, cand = defs[nm]
        S[nm] = {"ev": ev}
        for mk in MATCH:
            if mk in FULL_MATCH or nm in ROBUST_SETS:
                S[nm][mk] = match(ev, cand, keys[mk])
    S["bayes_err"] = _jeff_np(B["prior"] + B["N"] + B["W"], B["lj"])            # SW = N + W
    S["upd"] = upd
    S["counts"] = dict(n_diag=int(diag.sum()), n_axis=int(axis.sum()), n_big=int(big.sum()),
                       median_step_hold=float(np.median(np.nonzero(S["hold"]["ev"])[1] + 1)))
    return S


def _rows(S, env):
    """the summed rows: (name, matching key or None, flat var-step indices, environment, weights or None); names
    '<set>_ev' (all events), '<set>_<key>_cov' (events whose bin has candidates), '<set>_<key>_ctl' (matched control)"""
    out = []
    for nm in SETS:
        idx = np.flatnonzero(S[nm]["ev"].ravel())
        out.append((f"{nm}_ev", None, idx, env[idx // (T * 2)], None))
        for mk in MATCH:
            if mk not in S[nm]:
                continue
            w, cov = S[nm][mk]
            for part, m, ww in (("cov", cov, None), ("ctl", w > 0, w)):
                idx = np.flatnonzero(m.ravel())
                out.append((f"{nm}_{mk}_{part}", mk, idx, env[idx // (T * 2)], None if ww is None else ww.ravel()[idx]))
    return out


def build_mem(run):
    """per checkpoint and environment: sums of memory use, intact and cut errors over every set's events, covered
    events and (weighted) matched controls under every matching key (memory use only for the robustness keys), and
    for the Trained RNN the errors of the two memoryless linear rules (clock_errors()); counts; Bayes's own memory use
    and update size; the network's per-step regression coefficients -> memuse_<run>.npz"""
    path = os.path.join(OUT, f"memuse_{run}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    t0 = time.time()
    env = _env()
    S = event_sets()
    rows = _rows(S, env)
    clock = run == "trained"
    B = bayes_refs() if clock else None
    z = np.load(os.path.join(AN, f"ent_memcut_var_{run}.npz"))
    eps = z["episode"]
    K = len(eps)
    qall = QTY + (CLK if clock else ())
    be, ub = S["bayes_err"].ravel(), S["upd"].ravel()
    out = {}
    for name, mk, idx, e, w in rows:
        out[f"n_{name}"] = np.bincount(e, w, minlength=NE)
        out[f"bayes_{name}"] = np.bincount(e, be[idx] * (1. if w is None else w), minlength=NE)
        out[f"upd_{name}"] = np.bincount(e, ub[idx] * (1. if w is None else w), minlength=NE)
        for q in (qall if mk is None or mk in FULL_MATCH else ("u",)):
            out[f"{name}_{q}"] = np.zeros((K, NE))
    coefs = np.zeros((K, T, 3))
    ei_all, ec_all = z["err_int"], z["err_cut1"]
    for k in range(K):
        ei = ei_all[k].astype(np.float64).ravel()
        ec = ec_all[k].astype(np.float64).ravel()
        vals = {"u": ec - ei, "int": ei, "cut": ec}
        if clock:
            coefs[k] = clock_coefs(np.load(os.path.join(AN, run, f"ck_{int(eps[k]):05d}.npz")))
            vals["s1"], vals["st"] = (v.ravel() for v in clock_errors(coefs[k], B))
        for name, mk, idx, e, w in rows:
            for q in qall:
                if f"{name}_{q}" in out:
                    v = vals[q][idx]
                    out[f"{name}_{q}"][k] = np.bincount(e, v if w is None else w * v, minlength=NE)
    out.update(episode=eps, x=gap.train_x(eps), **S["counts"],
               check_step1=max(float(np.abs(ec_all[k, :, 0].astype(np.float64) - ei_all[k, :, 0]).max())
                               for k in range(K)))
    if clock:
        out["coefs"] = coefs
    for nm in SETS:
        for mk in MATCH:
            if mk in S[nm]:
                w = S[nm][mk][0]
                out[f"neff_{nm}_{mk}"] = w.sum() ** 2 / (w ** 2).sum()
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    print(f"{run}: memory-use sums, {K} checkpoints, {time.time() - t0:.0f} s -> {path}", flush=True)
    return out


# ═══════════════════════════════════════════════════════════════════
# Answer families (episode-defined, fixed) and the long-memory excess at T
# ═══════════════════════════════════════════════════════════════════

def answer_families():
    """goal-variable families of the 10,000 probes (bool (n,)) and the references: rev (Bayes revises: more than .6 on
    a value at some step 8-24 that is not its answer at T), settled (Bayes's answer the same at every step 8-30), other
    (neither), nfar (naive's answer at T 2+ cells off the truth), settled_m (weights on the settled probes that give
    them the revision probes' histogram of Bayes's confidence at T); Bayes's own revisions under the preliminary's rule
    (more than .6 on a wrong value at some step 8-24, right at T); Bayes's and windowed Bayes's correctness at T"""
    P = np.load(os.path.join(AN, "probes.npz"))
    n = len(P["goal"])
    ar, g = np.arange(n), P["goal"]
    rg = P["r"][ar, g]
    bB = wh.norm(P["log_jb"])[ar, :, g]
    aB = bB.argmax(-1)
    ansT = aB[:, -1]
    commit = bB[:, REV_WIN].max(-1) > COMMIT                                   # (n, 17)
    rev = (commit & (aB[:, REV_WIN] != ansT[:, None])).any(1)
    settled = (aB[:, SETTLE_FROM:] == ansT[:, None]).all(1)
    nfar = np.abs(wh.norm(P["log_nb"])[ar, -1, g].argmax(-1) - rg) >= FAR
    conf = bB[:, -1].max(-1)
    cbin = np.minimum((conf / CONF_STEP).astype(int), int(round(1 / CONF_STEP)) - 1)
    w, cov = match(rev, settled, cbin)
    B = {"bayes": P["right_bayes"][ar, g]}
    for wd in WINDOWS:
        B[wd] = P[f"right_window{wd}"][ar, g]
    own_rev = (commit & (aB[:, REV_WIN] != rg[:, None])).any(1) & B["bayes"]
    return dict(all=np.ones(n, bool), rev=rev, settled=settled, other=~rev & ~settled, nfar=nfar, w_settled_m=w,
                rev_cov=cov, conf=conf, own_rev_B=own_rev, B=B, g=g, rg=rg, env=P["env"],
                naive=P["right_naive"][ar, g])


def clock_right(cum, d, g, rg, w=5):
    """the 'clock-only' surrogate of the w-step cut: the answer at T (goal variable) when every increment of steps
    w+1..T is shifted by the step-w mean increment minus that step's mean increment (means over all probes, per variable
    position and value; no environment or cell content): argmax(cum_T + Sum_(t>w) (dbar_w - dbar_t)) == truth.
    cum (n, T, 2, R) centred cumulative logits, d (n, T, 2, R) their increments"""
    dbar = d.mean(0)                                                            # (T, 2, R)
    S = (T - w) * dbar[w - 1] - dbar[w:].sum(0)                                 # (2, R)
    ar = np.arange(len(g))
    return ((cum[:, -1] + S[None])[ar, g].argmax(-1) == rg).astype(np.float64)


def build_trunc(run):
    """per checkpoint and environment: sums of the excess (intact - w cut) - (Bayes - windowed Bayes) per family and w,
    of the network's and Bayes's w-cut losses, accuracy at T, the network's own revisions (the preliminary's rule
    on its own goal marginal: more than .6 on a wrong value at some step 8-24, right at T), and per family the
    conditional counts (right at T; right and wrong under the w5 cut; Bayes right; Bayes right and windowed Bayes 5
    wrong) -> trunc_<run>.npz"""
    path = os.path.join(OUT, f"trunc_{run}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    t0 = time.time()
    F = answer_families()
    env, g, rg = F["env"], F["g"], F["rg"]
    n = len(g)
    ar = np.arange(n)
    eps = sorted(int(os.path.basename(f)[3:8]) for f in os.listdir(os.path.join(AN, run)) if f.startswith("ck_"))
    K = len(eps)
    Bb = F["B"]["bayes"].astype(np.float64)
    bl = {wd: Bb - F["B"][wd] for wd in WINDOWS}                             # Bayes's own w-cut loss per probe
    fw = {f: (F[f] if f != "settled_m" else F["settled"]).astype(np.float64) * (F["w_settled_m"] if f == "settled_m"
                                                                                 else 1.) for f in FAMS}
    out = {f"x{wd}_{f}": np.zeros((K, NE)) for wd in WINDOWS for f in FAMS}
    out.update({f"l{wd}_{f}": np.zeros((K, NE)) for wd in WINDOWS for f in FAMS})
    out.update(acc=np.zeros((K, NE)), own_n=np.zeros((K, NE)), own_l5=np.zeros((K, NE)), own_bl5=np.zeros((K, NE)),
               clk_l5=np.zeros((K, NE)), beta_N=np.zeros((K, T)))
    out.update({f"{q}_{f}": np.zeros((K, NE)) for q in ("acc", "cw5") for f in FAMS})
    B5 = F["B"][5].astype(np.float64)
    for f in FAMS:
        out[f"n_{f}"] = np.bincount(env, fw[f], minlength=NE)
        out[f"Br_{f}"] = np.bincount(env, fw[f] * Bb, minlength=NE)                  # Bayes right
        out[f"Bcw5_{f}"] = np.bincount(env, fw[f] * Bb * (1. - B5), minlength=NE)    # ... and windowed Bayes 5 wrong
        for wd in WINDOWS:
            out[f"bl{wd}_{f}"] = np.bincount(env, fw[f] * bl[wd], minlength=NE)
    for k, ep in enumerate(eps):
        z = np.load(os.path.join(AN, run, f"ck_{ep:05d}.npz"))
        tr = z["trunc_right"][ar, :, g].astype(np.float64)                     # (n, 4): w1, w2, w5, intact
        Iv = tr[:, 3]
        assert (z["right"][ar, -1, g] == (Iv > 0)).all(), f"{run} {ep}: trunc intact != right at T"
        out["acc"][k] = np.bincount(env, Iv, minlength=NE)
        for f in FAMS:
            out[f"acc_{f}"][k] = np.bincount(env, fw[f] * Iv, minlength=NE)
            out[f"cw5_{f}"][k] = np.bincount(env, fw[f] * Iv * (1. - tr[:, 2]), minlength=NE)   # right, cut 5 wrong
        for j, wd in enumerate(WINDOWS):
            loss = Iv - tr[:, j]
            for f in FAMS:
                out[f"l{wd}_{f}"][k] = np.bincount(env, fw[f] * loss, minlength=NE)
                out[f"x{wd}_{f}"][k] = np.bincount(env, fw[f] * (loss - bl[wd]), minlength=NE)
        lq = z["logq"][ar, :, g][:, REV_WIN].astype(np.float64)
        q = np.exp(lq - lq.max(-1, keepdims=True))
        q /= q.sum(-1, keepdims=True)
        own = ((q.max(-1) > COMMIT) & (q.argmax(-1) != rg[:, None])).any(1) & (Iv > 0)
        out["own_n"][k] = np.bincount(env, own, minlength=NE)
        out["own_l5"][k] = np.bincount(env, own * (Iv - tr[:, 2]), minlength=NE)
        out["own_bl5"][k] = np.bincount(env, own * bl[5], minlength=NE)
        lqa = z["logq"].astype(np.float64)                                      # (n, T, 2, R)
        cum = lqa - lqa.mean(-1, keepdims=True)                                 # centred cumulative logits
        sur = clock_right(cum, np.diff(cum, axis=1, prepend=0.), g, rg)
        out["clk_l5"][k] = np.bincount(env, Iv - sur, minlength=NE)
        xtx, xtd = z["reg_xtx"], z["reg_xtd"]
        out["beta_N"][k] = xtd[:, 0] / np.maximum(xtx[:, 0, 0], 1e-300)       # per step, single regressor N
    ob = F["own_rev_B"]
    P = np.load(os.path.join(AN, "probes.npz"))
    lj = P["log_jb"].astype(np.float64)
    surB = clock_right(lj - lj.mean(-1, keepdims=True), P["inc_D"].astype(np.float64), g, rg)
    Nn = P["inc_N"].astype(np.float64)
    out.update(episode=np.array(eps), x=gap.train_x(eps), naive=np.bincount(env, F["naive"], minlength=NE),
               bayes=np.bincount(env, Bb, minlength=NE), own_n_B=np.bincount(env, ob, minlength=NE),
               own_l5_B=np.bincount(env, ob * bl[5], minlength=NE), rev_cov=float(F["rev_cov"].mean() /
                                                                                 F["rev"].mean()),
               clk_bl5=np.bincount(env, Bb - surB, minlength=NE),
               beta_N_bayes=np.stack([(P[f"inc_{k_}"].astype(np.float64) * Nn).sum((0, 2, 3)) / (Nn * Nn).sum((0, 2, 3))
                                      for k_ in ("D", "SW")]))
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    print(f"{run}: excess sums, {K} checkpoints, {time.time() - t0:.0f} s -> {path}", flush=True)
    return out


REMATCH_SETS = ("hold", "couple", "q5")


def build_rematch(run="trained"):
    """the events-minus-control contrast (covered events, primary key) with the control re-matched inside every
    resample (weights from the resample's own event and candidate histograms) instead of the full-sample weights: per
    set the full-sample curve (K,) and the resampled curves (N_BOOT, K) -> rematch_<run>.npz"""
    path = os.path.join(OUT, f"rematch_{run}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    t0 = time.time()
    P = np.load(os.path.join(AN, "probes.npz"))
    S = event_sets()
    B = bayes_refs()
    key = match_keys(P["fr"], wh.entropy(wh.norm(P["log_jb"])), _jeff_np(B["prior"], B["lj"]))[MATCH[0]]
    C = gap.boot_counts()
    z = np.load(os.path.join(AN, f"ent_memcut_var_{run}.npz"))
    K = len(z["episode"])
    EI, EC = z["err_int"], z["err_cut1"]
    envv = np.broadcast_to(P["env"][:, None, None], key.shape).ravel()
    out = {"episode": z["episode"]}
    for nm in REMATCH_SETS:
        ev = S[nm]["ev"]
        cand = S[nm][MATCH[0]][0] > 0                      # candidates in bins that hold events (only these weighted)
        uk, inv = np.unique(key[ev | cand], return_inverse=True)
        kid = np.full(key.shape, -1)
        kid[ev | cand] = inv.ravel()
        nk = len(uk)
        fe_, fc_ = np.flatnonzero(ev.ravel()), np.flatnonzero(cand.ravel())
        ie, ic = envv[fe_] * nk + kid.ravel()[fe_], envv[fc_] * nk + kid.ravel()[fc_]
        ev_cnt = np.bincount(ie, minlength=NE * nk).reshape(NE, nk).astype(np.float64)
        ca_cnt = np.bincount(ic, minlength=NE * nk).reshape(NE, nk).astype(np.float64)
        EVc, CAc = C @ ev_cnt, C @ ca_cnt
        EVf, CAf = ev_cnt.sum(0), ca_cnt.sum(0)
        ok, okb = CAf > 0, CAc > 0
        Wb = np.where(okb, EVc / np.where(okb, CAc, 1.), 0.)                       # per-resample weights per bin
        Nb = np.where(okb, EVc, 0.).sum(1)
        full, boot = np.zeros(K), np.zeros((len(C), K))
        for k in range(K):
            u = EC[k].astype(np.float64).ravel() - EI[k].astype(np.float64).ravel()
            ev_s = np.bincount(ie, u[fe_], minlength=NE * nk).reshape(NE, nk)
            ca_s = np.bincount(ic, u[fc_], minlength=NE * nk).reshape(NE, nk)
            full[k] = (ev_s.sum(0)[ok].sum() - (EVf[ok] / CAf[ok] * ca_s.sum(0)[ok]).sum()) / EVf[ok].sum()
            EVs, CAs = C @ ev_s, C @ ca_s
            boot[:, k] = (np.where(okb, EVs, 0.).sum(1) - (Wb * CAs).sum(1)) / Nb
        out[f"{nm}_full"], out[f"{nm}_boot"] = full, boot
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    print(f"{run}: re-matched contrasts, {K} checkpoints, {time.time() - t0:.0f} s -> {path}", flush=True)
    return out


def build(runs=None):
    for run in runs or TRUNC_RUNS:
        if run in MEM_RUNS:
            build_mem(run)
        build_trunc(run)
        if run == "trained":
            build_rematch(run)


# ═══════════════════════════════════════════════════════════════════
# Curves, crossings and the paired comparison with the behaviour rows
# ═══════════════════════════════════════════════════════════════════

def ratio(s, c, C=None):
    """per-environment sums s (K, NE) over counts c (NE,) -> pooled (K,) and resampled (N_BOOT, K)"""
    C = gap.boot_counts() if C is None else C
    return s.sum(1) / c.sum(), (C @ s.T) / (C @ c)[:, None]


def cross(x, y, yb, lev=0., levb=None, sign=1.):
    """stays past lev to the end (td_tt_gap.crossing) for the curve and every resample (their own level levb if given):
    dict(p, lo, hi, share, boot, level)"""
    p = gap.crossing(x, y, lev, sign)
    cb = gap.crossing(x, yb, lev if levb is None else levb, sign)
    lo, hi, share = gap.interval(cb)
    return dict(p=p, lo=lo, hi=hi, share=share, boot=cb, level=lev)


def _peak(y):
    """the maximum of the 3-checkpoint running mean (per row)"""
    y = np.atleast_2d(y)
    k = np.ones(3) / 3
    return np.array([np.convolve(v, k, "valid").max() for v in y])


def first_cross(x, y, lev):
    """the first training episode at which y (..., K) reaches lev (...,), log-interpolated (for the rise of a curve that
    falls back later); NaN if never"""
    y = np.atleast_2d(np.asarray(y, np.float64))
    lev = np.broadcast_to(np.asarray(lev, np.float64), (len(y),))
    past = y >= lev[:, None]
    out = np.full(len(y), np.nan)
    lx = np.log(x)
    for i in np.flatnonzero(past.any(1)):
        k = int(np.argmax(past[i]))
        if k == 0:
            out[i] = x[0]
            continue
        f = (lev[i] - y[i, k - 1]) / (y[i, k] - y[i, k - 1])
        out[i] = np.exp(lx[k - 1] + np.clip(f, 0, 1) * (lx[k] - lx[k - 1]))
    return out


def rise_fall(x, y, yb):
    """a curve that rises and falls: peak (3-checkpoint running mean) and its location; rise = half its peak, first
    reached; fall = below half its peak for good (both with the level recomputed in every resample)"""
    pk, pkb = float(_peak(y)[0]), _peak(yb)
    rb = first_cross(x, yb, pkb / 2)
    lo, hi, share = gap.interval(rb)
    rise = dict(p=float(first_cross(x, y, pk / 2)[0]), lo=lo, hi=hi, share=share, boot=rb, level=pk / 2)
    fall = cross(x, y, yb, pk / 2, pkb / 2, sign=-1.)
    kp = int(np.argmax(np.convolve(y, np.ones(3) / 3, "same")))
    return dict(peak=pk, peak_band=np.percentile(pkb, [5, 95]), x_peak=float(x[kp]), rise=rise, fall=fall,
                end=float(y[-3:].mean()), end_band=np.percentile(yb[:, -3:].mean(1), [5, 95]))


def mem_curves(D, nm, mk=MATCH[0]):
    """dict of (full (K,), resampled (N_BOOT, K)) for set nm under matching key mk: ev_<q> (all events), cov_<q>
    (covered events), ctl_<q> (the matched control) for every stored quantity q; diff = cov_u - ctl_u; full-sample
    means of Bayes's own memory use and of Bayes's update size at the events, covered events and controls; Bayes's own
    memory use at the events per resample (bayes_ev_b)"""
    C = gap.boot_counts()
    out = {}
    for q in QTY + CLK:
        if f"{nm}_ev_{q}" in D:
            out[f"ev_{q}"] = ratio(D[f"{nm}_ev_{q}"], D[f"n_{nm}_ev"], C)
        for p in ("cov", "ctl"):
            if f"{nm}_{mk}_{p}_{q}" in D:
                out[f"{p}_{q}"] = ratio(D[f"{nm}_{mk}_{p}_{q}"], D[f"n_{nm}_{mk}_{p}"], C)
    out["diff"] = (out["cov_u"][0] - out["ctl_u"][0], out["cov_u"][1] - out["ctl_u"][1])
    for p, name in (("ev", f"{nm}_ev"), ("cov", f"{nm}_{mk}_cov"), ("ctl", f"{nm}_{mk}_ctl")):
        out[f"bayes_{p}"] = float(D[f"bayes_{name}"].sum() / D[f"n_{name}"].sum())
        out[f"upd_{p}"] = float(D[f"upd_{name}"].sum() / D[f"n_{name}"].sum())
    out["bayes_ev_b"] = (C @ D[f"bayes_{nm}_ev"]) / (C @ D[f"n_{nm}_ev"])
    out["n_ev"], out["n_cov"] = float(D[f"n_{nm}_ev"].sum()), float(D[f"n_{nm}_{mk}_cov"].sum())
    out["neff"] = float(D[f"neff_{nm}_{mk}"])
    return out


def _end(y, yb):
    """the mean of the last 3 checkpoints and its 5-95% band"""
    return float(y[-3:].mean()), np.percentile(yb[:, -3:].mean(1), [5, 95])


def clock_refs(c, x):
    """the events-minus-control contrast (covered events) against four memoryless references: the stored cut ('cut',
    = beats), the step-1 linear rule ('s1'), the clock-matched linear rule ('st'), and the stored cut's gap frozen at
    its X_FREEZE value ('frozen', each resample its own); 'gap': the events-minus-control gaps of the intact error and
    of each reference's error; 'helps_<q>': the intact error at the events below reference q's for good"""
    g = {q: (c[f"cov_{q}"][0] - c[f"ctl_{q}"][0], c[f"cov_{q}"][1] - c[f"ctl_{q}"][1]) for q in ("int", "cut") + CLK}
    out = {"gap": g}
    for q in ("cut",) + CLK:
        out[q] = cross(x, g[q][0] - g["int"][0], g[q][1] - g["int"][1])
        out[f"helps_{q}"] = cross(x, c[f"ev_{q}"][0] - c["ev_int"][0], c[f"ev_{q}"][1] - c["ev_int"][1])
    kf = int(np.argmin(np.abs(x - X_FREEZE)))
    out["frozen"] = cross(x, g["cut"][0][kf] - g["int"][0], g["cut"][1][:, kf:kf + 1] - g["int"][1])
    out["x_freeze"] = float(x[kf])
    return out


def mem_summary(run):
    """per set (primary matching key): curves and crossings: helps (event memory use > 0 for good), beats (covered
    events minus the matched control > 0 for good), ctl (the control > 0 for good), b10 / b50 (u / u_Bayes past .1 /
    .5 for good; u_Bayes = memory-free Bayes's error at the events, the natural reference), half (above half its own
    end value for good), trough, plateau (mean, min, max over PLATEAU); 'keys': per matching key the beats crossing,
    the end difference and the control's coverage and Bayes references; 'clock' (Trained RNN, per full key):
    clock_refs()"""
    D = build_mem(run)
    x = D["x"]
    S = {"x": x, "D": D}
    pl = (x >= PLATEAU[0]) & (x <= PLATEAU[1])
    for nm in SETS:
        c = mem_curves(D, nm)
        y, yb = c["ev_u"]
        d, db = c["diff"]
        end, endb = y[-3:].mean(), yb[:, -3:].mean(1)
        kt = int(np.argmin(y))
        fB, fBb = y / c["bayes_ev"], yb / c["bayes_ev_b"][:, None]
        c.update(helps=cross(x, y, yb), beats=cross(x, d, db), ctl=cross(x, *c["ctl_u"]),
                 fB=(fB, fBb), b10=cross(x, fB, fBb, .1), b50=cross(x, fB, fBb, .5),
                 half=cross(x, y, yb, end / 2, endb / 2) if end > 0 else None,
                 trough=(float(y[kt]), float(x[kt]), np.percentile(yb[:, kt], [5, 95])),
                 plateau=(float(y[pl].mean()), float(y[pl].min()), float(y[pl].max())),
                 end=(float(end), np.percentile(endb, [5, 95])), end_ctl=_end(*c["ctl_u"]),
                 hurts=cross(x, y, yb, sign=-1.), end_diff=_end(d, db))
        c["keys"], c["clock"] = {}, {}
        for mk in MATCH:
            if f"{nm}_{mk}_cov_u" not in D:
                continue
            ck = c if mk == MATCH[0] else mem_curves(D, nm, mk)
            dk, dkb = ck["diff"]
            c["keys"][mk] = dict(beats=cross(x, dk, dkb), end_diff=_end(dk, dkb), end_ctl=_end(*ck["ctl_u"]),
                                 cov=ck["n_cov"] / ck["n_ev"], neff=ck["neff"], **{f"{a}_{p}": ck[f"{a}_{p}"] for a in
                                                                                  ("bayes", "upd") for p in
                                                                                  ("ev", "cov", "ctl")})
            if f"{nm}_{mk}_cov_s1" in D:
                c["clock"][mk] = clock_refs(ck, x)
        S[nm] = c
    return S


def trunc_summary(run):
    """per family: the excess curves at w = 1, 2, 5 and the w5 rise / fall; accuracy gap fraction crossings (c0, c10,
    c50; naive and Bayes resampled on the same environments); the network's own revisions (share, w5 loss) and Bayes's
    under the same rule"""
    D = build_trunc(run)
    x = D["x"]
    C = gap.boot_counts()
    S = {"x": x, "D": D}
    for wd in WINDOWS:
        for f in FAMS:
            S[f"x{wd}_{f}"] = ratio(D[f"x{wd}_{f}"], D[f"n_{f}"], C)
            S[f"l{wd}_{f}"] = ratio(D[f"l{wd}_{f}"], D[f"n_{f}"], C)
            S[f"bl{wd}_{f}"] = (float(D[f"bl{wd}_{f}"].sum() / D[f"n_{f}"].sum()),
                                (C @ D[f"bl{wd}_{f}"]) / (C @ D[f"n_{f}"]))
    for f in FAMS:
        S[f"rf_{f}"] = rise_fall(x, *S[f"x5_{f}"])
        # the ceiling-free version: P(w5 cut wrong | the network right) - P(windowed Bayes 5 wrong | Bayes right)
        num, den, bn, bd = D[f"cw5_{f}"], D[f"acc_{f}"], D[f"Bcw5_{f}"], D[f"Br_{f}"]
        S[f"cr5_{f}"] = (num.sum(1) / den.sum(1), (C @ num.T) / (C @ den.T))
        S[f"cb5_{f}"] = (float(bn.sum() / bd.sum()), (C @ bn) / (C @ bd))
        S[f"cx5_{f}"] = (S[f"cr5_{f}"][0] - S[f"cb5_{f}"][0], S[f"cr5_{f}"][1] - S[f"cb5_{f}"][1][:, None])
        S[f"acc_{f}"] = ratio(D[f"acc_{f}"], D[f"n_{f}"], C)
    S["rf_x2"] = rise_fall(x, *S["x2_all"])
    acc, accb = ratio(D["acc"], np.full(NE, float(PER_ENV)), C)
    nv, nvb = D["naive"].sum() / (NE * PER_ENV), (C @ D["naive"]) / (C @ np.full(NE, float(PER_ENV)))
    bv, bvb = D["bayes"].sum() / (NE * PER_ENV), (C @ D["bayes"]) / (C @ np.full(NE, float(PER_ENV)))
    f_, fb = gap.gap_fraction(acc, nv, bv), gap.gap_fraction(accb, nvb, bvb)
    S["acc"] = (acc, accb)
    S["acc_frac"] = (f_, fb)
    for lev in (0., .1, .5):
        S[f"acc_c{int(lev * 100)}"] = cross(x, f_, fb, lev)
    S["own_share"] = ratio(D["own_n"], np.full(NE, float(PER_ENV)), C)
    with np.errstate(invalid="ignore", divide="ignore"):
        S["own_l5"] = (D["own_l5"].sum(1) / D["own_n"].sum(1), (C @ D["own_l5"].T) / (C @ D["own_n"].T))
        S["own_bl5"] = D["own_bl5"].sum(1) / D["own_n"].sum(1)
        S["own_part"] = (D["own_l5"].sum(1) / D["l5_all"].sum(1), (C @ D["own_l5"].T) / (C @ D["l5_all"].T))
    S["own_B"] = dict(share=float(D["own_n_B"].sum() / (NE * PER_ENV)), l5=float(D["own_l5_B"].sum() /
                                                                                  D["own_n_B"].sum()),
                      part=float(D["own_l5_B"].sum() / D["bl5_all"].sum()))
    S["clk"] = ratio(D["clk_l5"], np.full(NE, float(PER_ENV)), C)
    S["clk_B"] = float(D["clk_bl5"].sum() / (NE * PER_ENV))
    return S


def behaviour_rows():
    """the Trained RNN's behaviour rows of td_tt_timeline with their crossings at f = .1 and .5 (each with the paired
    resamples 'boot'): dict name -> row"""
    import td_tt_timeline as tl
    rows = tl.annotate([r for r in tl.load_rows() if r["run"] == "trained"])
    return {r["name"]: r for r in rows}


def _inf(b):
    return np.where(np.isnan(b), np.inf, b)


def relation(a, b):
    """interval rule: 'before' / 'after' / 'with' (overlap) / '-' (a never); plus the paired share P(a < b)"""
    if not np.isfinite(a["p"]):
        return "-", float((_inf(a["boot"]) < _inf(b["boot"])).mean())
    alo, ahi = a["lo"], a["hi"] if np.isfinite(a["hi"]) else np.inf
    blo, bhi = b["lo"], b["hi"] if np.isfinite(b["hi"]) else np.inf
    rel = "after" if alo > bhi else "before" if ahi < blo else "with"
    return rel, float((_inf(a["boot"]) < _inf(b["boot"])).mean())


def family_bounds(rows, names, lev):
    """per resample the earliest / latest crossing (f = lev) over a family's rows, as crossing dicts; 'row': the row
    whose full-sample crossing is the earliest / latest (its own interval is rows[row]['L'][lev])"""
    B = np.stack([_inf(rows[n]["L"][lev]["boot"]) for n in names])
    out = {}
    for nm, v in (("earliest", B.min(0)), ("latest", B.max(0))):
        vb = np.where(np.isinf(v), np.nan, v)
        lo, hi, share = gap.interval(vb)
        pts = np.array([rows[n]["L"][lev]["p"] for n in names])
        j = int(np.nanargmin(pts) if nm == "earliest" else np.nanargmax(pts))
        out[nm] = dict(p=float(pts[j]), lo=lo, hi=hi, share=share, boot=vb, row=names[j])
    return out


def _relations(a, S, rows):
    """a crossing against accuracy (f > 0 / .1 / .5), each family's earliest / latest row per resample and every
    behaviour row (its own interval): dict -> (interval relation, paired share P(a < b))"""
    r = {("acc", lev): relation(a, S["acc"][lev]) for lev in (0., .1, .5)}
    for fam, names in (("holding", HOLD_ROWS), ("coupling", COUPLE_ROWS)):
        for lev in (.1, .5):
            for side in ("earliest", "latest"):
                r[fam, lev, side] = relation(a, S["fam"][fam][lev][side])
            for n in names:
                r[n, lev] = relation(a, rows[n]["L"][lev])
    return r


def summary():
    """everything numbers() prints and figure() draws"""
    S = {"mem": {r: mem_summary(r) for r in MEM_RUNS}, "trunc": {r: trunc_summary(r) for r in TRUNC_RUNS}}
    rows = behaviour_rows()
    S["rows"] = rows
    S["fam"] = {"holding": {lev: family_bounds(rows, HOLD_ROWS, lev) for lev in (.1, .5)},
                "coupling": {lev: family_bounds(rows, COUPLE_ROWS, lev) for lev in (.1, .5)}}
    acc = rows[ACC_ROW]
    S["acc"] = {lev: acc["L"][lev] for lev in (0., .1, .5)}
    m = S["mem"]["trained"]
    S["rel"] = {(nm, ev): _relations(m[nm][ev], S, rows) for nm in SETS for ev in ("helps", "beats", "b10", "b50")}
    # the entangled-steps contrast: every matching key and every memoryless reference against passing naive
    S["rel_q5"] = {("key", mk): relation(v["beats"], S["acc"][0.]) for mk, v in m["q5"]["keys"].items()}
    S["rel_q5"].update({(mk, q): relation(v[q], S["acc"][0.]) for mk, v in m["q5"]["clock"].items()
                        for q in ("cut", "s1", "st", "frozen")})
    t = S["trunc"]["trained"]
    S["rel_trunc"] = {part: _relations(t["rf_all"][part], S, rows) for part in ("rise", "fall")}
    S["acc_check"] = float(np.nanmax(np.abs(_inf(t["acc_c50"]["boot"]) - _inf(acc["L"][.5]["boot"]))))
    S["run_rel"] = {run: {lev: relation(S["trunc"][run]["rf_all"]["rise"], S["trunc"][run][f"acc_c{int(lev * 100)}"])
                          for lev in (0., .1, .5)} for run in TRUNC_RUNS}
    R_ = build_rematch("trained")
    S["rematch"] = {nm: dict(beats=cross(m["x"], R_[f"{nm}_full"], R_[f"{nm}_boot"]),
                             end=_end(R_[f"{nm}_full"], R_[f"{nm}_boot"])) for nm in REMATCH_SETS}
    S["rematch"]["q5"]["rel"] = relation(S["rematch"]["q5"]["beats"], S["acc"][0.])
    return S


# ═══════════════════════════════════════════════════════════════════
# Numbers
# ═══════════════════════════════════════════════════════════════════

def _f(c, nd=0):
    if c is None:
        return "n/a"
    if not np.isfinite(c["p"]):
        return f"never ({c['share']:.0%} of resamples cross)"
    hi = f"{c['hi']:,.{nd}f}" if np.isfinite(c["hi"]) else "never"
    lo = f"{c['lo']:,.{nd}f}" if np.isfinite(c["lo"]) else "?"
    return f"{c['p']:,.{nd}f} [{lo}-{hi}]" + ("" if c["share"] >= .995 else f" ({c['share']:.0%} cross)")


def _at(x, y, xv):
    return float(y[int(np.argmin(np.abs(x - xv)))])


def _b(v, vb, nd=3):
    """'v [5%, 95%]' with signs"""
    lo, hi = np.percentile(vb, [5, 95])
    return f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def _k(x, xv):
    return int(np.argmin(np.abs(x - xv)))


def _rel_line(r, names_fam=True):
    """the relations of one crossing: accuracy, family earliest / latest per resample, per row"""
    out = ["vs accuracy f > 0 / .1 / .5: " + ", ".join(f"{r['acc', lev][0]} ({r['acc', lev][1]:.3f})"
                                                       for lev in (0., .1, .5))]
    for fam, names in (("holding", HOLD_ROWS), ("coupling", COUPLE_ROWS)):
        out.append(f"vs {fam} per resample: " + ", ".join(
            f"{side} c{int(lev * 100)} {r[fam, lev, side][0]} ({r[fam, lev, side][1]:.3f})"
            for lev in (.1, .5) for side in ("earliest", "latest")))
        for lev in (.1, .5):
            out.append(f"   rows c{int(lev * 100)}: " + "; ".join(f"{n[:24]} {r[n, lev][0]} {r[n, lev][1]:.3f}"
                                                                  for n in names))
    return "\n       ".join(out)


def numbers(S=None):
    S = summary() if S is None else S
    rows = S["rows"]
    print(f"check: accuracy c50 from the stored trunc 'intact' answers vs td_tt_timeline's accuracy row, per resample: "
          f"max |diff| {S['acc_check']:.3g} training episodes")
    print("== behaviour anchors (td_tt_timeline rows; Trained RNN; [5-95%])")
    for lev in (0., .1, .5):
        print(f"  accuracy f > {lev}: {_f(S['acc'][lev])}")
    for fam, names in (("holding", HOLD_ROWS), ("coupling", COUPLE_ROWS)):
        for n in names:
            print(f"  {fam:8s} {n[:52]:52s} c10 {_f(rows[n]['L'][.1]):28s} c50 {_f(rows[n]['L'][.5])}")
        for lev in (.1, .5):
            b = S["fam"][fam][lev]
            print(f"  {fam} c{int(lev * 100)}: earliest row {b['earliest']['row']} "
                  f"{_f(rows[b['earliest']['row']]['L'][lev])}, the family's earliest per resample "
                  f"[{b['earliest']['lo']:,.0f}-{b['earliest']['hi']:,.0f}]; latest row {b['latest']['row']} "
                  f"{_f(rows[b['latest']['row']]['L'][lev])}, the family's latest per resample "
                  f"[{b['latest']['lo']:,.0f}-{b['latest']['hi']:,.0f}]")
    for run in MEM_RUNS:
        m = S["mem"][run]
        x = m["x"]
        print(f"== memory use (cut - intact, Jeffreys/2 nats) at the events, {NAME[run]} (check: at step 1 the cut "
              f"equals the intact run, max |diff| {float(m['D']['check_step1']):.1e}, fp16 storage); controls matched "
              f"on {MATCH_NAME[MATCH[0]]} unless named")
        print(f"  (Bayes's end splits: median step {float(m['D']['median_step_hold']):.0f}; Bayes's large moves of 2+ "
              f"cells {int(m['D']['n_big']):,}, diagonal {int(m['D']['n_diag']):,}, one-axis "
              f"{int(m['D']['n_axis']):,})")
        for nm in SETS:
            c = m[nm]
            print(f"  -- {SET_NAME[nm]}: {c['n_ev']:,.0f} var-steps, {c['n_cov']:,.0f} with a matched control "
                  f"({c['n_cov'] / c['n_ev']:.1%}); matched control: effective n {c['neff']:,.0f}; Bayes's own memory "
                  f"use (memory-free Bayes's error) {c['bayes_ev']:.3f} at the events, {c['bayes_cov']:.3f} at the "
                  f"covered events, {c['bayes_ctl']:.3f} at the control; Bayes's update size {c['upd_ev']:.3f} / "
                  f"{c['upd_cov']:.3f} / {c['upd_ctl']:.3f}")
            tr, pl = c["trough"], c["plateau"]
            kt = _k(x, tr[1])
            print(f"     trough {tr[0]:+.3f} [{tr[2][0]:+.3f}, {tr[2][1]:+.3f}] at {tr[1]:,.0f} (neighbours "
                  f"{x[kt - 1]:,.0f}: {c['ev_u'][0][kt - 1]:+.3f}, {x[kt + 1]:,.0f}: {c['ev_u'][0][kt + 1]:+.3f}; "
                  f"intact error at the trough {c['ev_int'][0][kt]:.3f}); over {PLATEAU[0]:,.0f}-{PLATEAU[1]:,.0f}: "
                  f"mean {pl[0]:+.3f}, range {pl[1]:+.3f} to {pl[2]:+.3f}")
            print(f"     end (last 3) {c['end'][0]:+.3f} [{c['end'][1][0]:+.3f}, {c['end'][1][1]:+.3f}]; matched "
                  f"control "
                  f"{c['end_ctl'][0]:+.3f} [{c['end_ctl'][1][0]:+.3f}, {c['end_ctl'][1][1]:+.3f}]; events - control "
                  f"{c['end_diff'][0]:+.3f} [{c['end_diff'][1][0]:+.3f}, {c['end_diff'][1][1]:+.3f}]; below 0 for good "
                  f"(hurts): {_f(c['hurts'])}")
            ce_ = _end(*c["cov_u"])
            print(f"     covered events (with a matched control) at the end (last 3): {ce_[0]:+.3f} [{ce_[1][0]:+.3f}, "
                  f"{ce_[1][1]:+.3f}] (events - control = this minus the control's end)")
            print(f"     helps for good (> 0): {_f(c['helps'])};  beats the matched control for good: {_f(c['beats'])};"
                  f"  control helps for good: {_f(c['ctl'])}")
            print(f"     relative to Bayes's own memory use (f = u / u_Bayes; end {c['fB'][0][-3:].mean():.2f}): "
                  f"f > .1 "
                  f"for good {_f(c['b10'])}, f > .5 for good {_f(c['b50'])};  above half its own end value for good: "
                  f"{_f(c['half'])}")
            line = ("     at (events / control memory use; events' intact and cut errors; control's intact and cut "
                    "errors)")
            for xv in (80, 280, 300, 320, 400, 640, 880, 1120, 1600, 2000, 2400, 3200, 4000, 6400, 12000, x[-1]):
                k = _k(x, xv)
                line += (f"\n        {x[k]:7,.0f}: {c['ev_u'][0][k]:+.3f} / {c['ctl_u'][0][k]:+.3f}; events int "
                         f"{c['ev_int'][0][k]:.3f} cut {c['ev_cut'][0][k]:.3f}; control int {c['ctl_int'][0][k]:.3f} "
                         f"cut {c['ctl_cut'][0][k]:.3f}")
            print(line)
            print("     the events' cut error, range over 80-4,000 / 400-4,000: " + " / ".join(
                f"{c['ev_cut'][0][w_].min():.3f}-{c['ev_cut'][0][w_].max():.3f}"
                for w_ in ((x >= a_) & (x <= 4000) for a_ in (80, 400))))
            for mk, v in c["keys"].items():
                print(f"     key {MATCH_NAME[mk]:44s}: coverage {v['cov']:.3f}, effective n {v['neff']:,.0f}; Bayes's "
                      f"update size {v['upd_cov']:.3f} (covered events) vs {v['upd_ctl']:.3f} (control); memory-free "
                      f"Bayes's error {v['bayes_cov']:.3f} vs {v['bayes_ctl']:.3f} "
                      f"({v['bayes_cov'] - v['bayes_ctl']:+.3f}"
                      f"); network: control at the end {v['end_ctl'][0]:+.3f}, events - control at the end "
                      f"{v['end_diff'][0]:+.3f} [{v['end_diff'][1][0]:+.3f}, {v['end_diff'][1][1]:+.3f}], beats "
                      f"{_f(v['beats'])}" + (f"; vs passing naive {S['rel_q5']['key', mk][0]} "
                                             f"({S['rel_q5']['key', mk][1]:.3f})" if run == "trained" and nm == "q5"
                                             else ""))
            print("     memory-free Bayes's error, covered events / control, per key: " + "; ".join(
                f"{mk} {v['bayes_cov'] / v['bayes_ctl']:.1f}x" for mk, v in c["keys"].items()))
            if run != "trained":
                continue
            for ev in ("helps", "beats", "b50"):
                print(f"     {ev}: " + _rel_line(S["rel"][nm, ev]))
            for mk, ck in c["clock"].items():
                g = ck["gap"]
                print(f"     memoryless references, key {MATCH_NAME[mk]} (events - control gap of each error; the "
                      f"contrast = reference gap - intact gap):")
                for q, lab in (("cut", "stored cut (beats)"), ("s1", "step-1 linear rule"),
                               ("st", "clock-matched linear rule"),
                               ("frozen", f"cut gap frozen at {ck['x_freeze']:,.0f}")):
                    print(f"        {lab:28s} crosses for good {_f(ck[q])}" + (
                        f"; vs passing naive {S['rel_q5'][mk, q][0]} ({S['rel_q5'][mk, q][1]:.3f})" if nm == "q5"
                        else ""))
                ka, kb = _k(x, 400), _k(x, 880)
                d = (g["cut"][0] - g["int"][0], g["cut"][1] - g["int"][1])
                line = (f"        gaps (cut / intact / step-1 rule / clock-matched rule; contrast [5-95%]); change "
                        f"{x[ka]:,.0f} -> {x[kb]:,.0f}: cut {g['cut'][0][kb] - g['cut'][0][ka]:+.4f}, intact "
                        f"{g['int'][0][kb] - g['int'][0][ka]:+.4f}")
                for xv in (280, 300, 400, 560, 640, 720, 800, 880, 960, 1120, 1360, 1600, 2000, 2080, 4000):
                    k = _k(x, xv)
                    line += (f"\n          {x[k]:6,.0f}: {g['cut'][0][k]:+.4f} / {g['int'][0][k]:+.4f} / "
                             f"{g['s1'][0][k]:+.4f} / {g['st'][0][k]:+.4f}; {_b(d[0][k], d[1][:, k], 4)}")
                print(line)
                win = (x >= 280) & (x <= 2080)
                dev = np.abs(g["s1"][0] - g["cut"][0])[win].max()
                print(f"        |step-1 rule gap - cut gap| over 280-2,080: max {dev:.4f}"
                      f"; the intact error at the events below each reference's for good: cut {_f(ck['helps_cut'])}, "
                      f"step-1 rule {_f(ck['helps_s1'])}, clock-matched rule {_f(ck['helps_st'])}")
            if nm in ("hold", "couple"):
                w4 = (x >= 400) & (x <= 2000)
                print(f"     the memoryless linear rules' errors at the events over 400-2,000: step-1 rule "
                      f"{c['ev_s1'][0][w4].min():.3f}-{c['ev_s1'][0][w4].max():.3f}, clock-matched rule "
                      f"{c['ev_st'][0][w4].min():.3f}-{c['ev_st'][0][w4].max():.3f}; intact "
                      f"{c['ev_int'][0][w4].min():.3f}-{c['ev_int'][0][w4].max():.3f}; at 400 / 2,000: step-1 "
                      + " / ".join(f"{c['ev_s1'][0][_k(x, v)]:.3f}" for v in (400, 2000)) + ", clock-matched "
                      + " / ".join(f"{c['ev_st'][0][_k(x, v)]:.3f}" for v in (400, 2000)) + ", intact "
                      + " / ".join(f"{c['ev_int'][0][_k(x, v)]:.3f}" for v in (400, 2000)))
    print("== the matched contrast with the control re-matched inside every resample (Trained RNN, primary key; "
          "build_rematch()): " + "; ".join(
              f"{nm} beats {_f(v['beats'])}, events - control at the end {v['end'][0]:+.3f} [{v['end'][1][0]:+.3f}, "
              f"{v['end'][1][1]:+.3f}]" for nm, v in S["rematch"].items())
          + f"; q5 vs passing naive {S['rematch']['q5']['rel'][0]} ({S['rematch']['q5']['rel'][1]:.3f})")
    print("== the long-memory excess at T: (intact - w-cut) - (Bayes - windowed Bayes), goal variable, per probe")
    for run in TRUNC_RUNS:
        t = S["trunc"][run]
        x = t["x"]
        print(f"  -- {NAME[run]}: accuracy passes naive {_f(t['acc_c0'])}, c10 {_f(t['acc_c10'])}, c50 "
              f"{_f(t['acc_c50'])}")
        for f in FAMS:
            rf = t[f"rf_{f}"]
            print(f"     w5, {FAM_NAME[f][:48]:48s} (n {t['D'][f'n_{f}'].sum():,.0f}, Bayes's own w5 loss "
                  f"{t[f'bl5_{f}'][0]:.4f}): peak {rf['peak']:+.3f} [{rf['peak_band'][0]:+.3f}, "
                  f"{rf['peak_band'][1]:+.3f}]"
                  f" at {rf['x_peak']:,.0f}; half its peak first reached {_f(rf['rise'])}; below half its peak for "
                  f"good {_f(rf['fall'])}; end {rf['end']:+.4f} [{rf['end_band'][0]:+.4f}, {rf['end_band'][1]:+.4f}]")
        rf = t["rf_x2"]
        print(f"     w2, all probes: peak {rf['peak']:+.3f} at {rf['x_peak']:,.0f}; rise {_f(rf['rise'])}; fall "
              f"{_f(rf['fall'])}; end {rf['end']:+.4f}")
        line = "     at (all / rev / settled / other / naive far / settled matched; w5 loss; w1 excess):"
        for xv in (400, 800, 1120, 1360, 1600, 2000, 2400, 3200, 4000, 6400, 12000, x[-1]):
            k = _k(x, xv)
            line += (f"\n        {x[k]:7,.0f}: " + " / ".join(f"{t[f'x5_{f}'][0][k]:+.3f}" for f in FAMS)
                     + f"; loss {t['l5_all'][0][k]:+.3f}; w1 {t['x1_all'][0][k]:+.3f}; acc {t['acc'][0][k]:.3f}")
        print(line)
        r = S["run_rel"][run]
        print("     rise vs accuracy f > 0 / .1 / .5: " + ", ".join(
            f"{r[lev][0]} (P(rise earlier) {r[lev][1]:.3f}; after it in {1 - r[lev][1]:.0%})" for lev in (0., .1, .5)))
        if run == "trained":
            for part in ("rise", "fall"):
                print(f"     {part}: " + _rel_line(S["rel_trunc"][part]))
            d = t["x5_rev"][1] - t["x5_settled"][1]
            dm = t["x5_rev"][1] - t["x5_settled_m"][1]
            for xv in (880, 1120, 1600, 2000, 2400, 4000):
                k = _k(x, xv)
                print(f"     revises - settled at {x[k]:,.0f}: {t['x5_rev'][0][k] - t['x5_settled'][0][k]:+.3f} "
                      f"[{np.percentile(d[:, k], 5):+.3f}, {np.percentile(d[:, k], 95):+.3f}]; vs settled matched on "
                      f"confidence {t['x5_rev'][0][k] - t['x5_settled_m'][0][k]:+.3f} "
                      f"[{np.percentile(dm[:, k], 5):+.3f}"
                      f", {np.percentile(dm[:, k], 95):+.3f}]")
            print(f"     revision probes covered by the confidence match: {t['D']['rev_cov']:.3f}")
            print("     ceiling-free version: P(w5 cut wrong | network right) - P(windowed Bayes 5 wrong | Bayes "
                  "right); "
                  "Bayes's term per family: " + ", ".join(f"{f} {t[f'cb5_{f}'][0]:.3f}" for f in FAMS))
            line = ("     at (network right at T per family: rev / settled / settled matched / naive far / all; "
                    "conditional excess: same order; rev - settled; rev - settled matched; naive far - all)")
            fs = ("rev", "settled", "settled_m", "nfar", "all")
            for xv in (400, 640, 800, 880, 1120, 1600, 2000, 2400, 4000, 12000, x[-1]):
                k = _k(x, xv)
                line += (f"\n        {x[k]:7,.0f}: " + " / ".join(f"{t[f'acc_{f}'][0][k]:.3f}" for f in fs) + "; "
                         + " / ".join(f"{t[f'cx5_{f}'][0][k]:+.3f}" for f in fs) + "; " + "; ".join(
                             _b(t[f"cx5_{a}"][0][k] - t[f"cx5_{b}"][0][k],
                                t[f"cx5_{a}"][1][:, k] - t[f"cx5_{b}"][1][:, k])
                             for a, b in (("rev", "settled"), ("rev", "settled_m"), ("nfar", "all"))))
            print(line)
            w8 = x >= 800
            print("     ceiling-free excess, range over 800-1,120: " + ", ".join(
                f"{f} {t[f'cx5_{f}'][0][w8 & (x <= 1120)].min():.3f}-{t[f'cx5_{f}'][0][w8 & (x <= 1120)].max():.3f}"
                for f in FAMS))
            for b_ in ("settled", "settled_m"):
                d0_, db_ = t["cx5_rev"][0] - t[f"cx5_{b_}"][0], t["cx5_rev"][1] - t[f"cx5_{b_}"][1]
                p5, p95 = np.percentile(db_, [5, 95], 0)
                up, dn = np.flatnonzero(w8 & (p5 > 0)), np.flatnonzero(w8 & (p95 < 0))
                kx = int(np.argmax(np.where(w8, d0_, -np.inf)))
                print(f"     ceiling-free rev - {b_}, the {w8.sum()} checkpoints from 800: 5% bound above 0 at "
                      f"{len(up)}"
                      + (f" (first {x[up[0]]:,.0f}, last {x[up[-1]]:,.0f}; of the {(x >= x[up[0]]).sum()} checkpoints "
                         f"from {x[up[0]]:,.0f})" if len(up) else "")
                      + f", 95% bound below 0 at {len(dn)}" + (f" (first {x[dn[0]]:,.0f}, last {x[dn[-1]]:,.0f})"
                                                               if len(dn) else "")
                      + f"; largest {_b(d0_[kx], db_[:, kx])} at {x[kx]:,.0f}")
            p5 = np.percentile(t["cx5_nfar"][1] - t["cx5_all"][1], 5, 0)
            bad = np.flatnonzero(p5 <= 0)
            print(f"     ceiling-free naive far - all: 5% bound above 0 at {(p5 > 0).sum()} of {len(x)} checkpoints"
                  + (f", at every one from {x[bad[-1] + 1]:,.0f}; at or below 0 at "
                     + ", ".join(f"{x[k]:,.0f}" for k in bad) if len(bad) and bad[-1] + 1 < len(x) else ""))
            e = S["trunc"]["echo"]
            print("     ESN, conditional excess (all probes) at 800 / 2,000 / the end: " + " / ".join(
                f"{e['cx5_all'][0][_k(e['x'], xv)]:+.3f}" for xv in (800, 2000, e["x"][-1])))
        cl, clb = t["clk"]
        k_pk = _k(x, t["rf_all"]["x_peak"])
        lo_, hi_ = np.percentile(clb[:, k_pk], [5, 95])
        bN = t["D"]["beta_N"]
        print(f"     clock-only surrogate of the w5 cut (step-mean increments, pooled over probes): its loss at the "
              f"excess peak ({x[k_pk]:,.0f}) {cl[k_pk]:+.4f} [{lo_:+.4f}, {hi_:+.4f}] vs the w5 loss "
              f"{t['l5_all'][0][k_pk]:+.4f}; largest over checkpoints {cl.max():+.4f} at {x[int(np.argmax(cl))]:,.0f}; "
              f"Bayes's own clock-only loss {t['clk_B']:+.4f}")
        g1 = bN[:, 0] - bN[:, 1:].mean(1)
        print(f"     the step-1 gain minus the mean gain of steps 2-30 (the per-step cut's clock): stays above 0 from "
              f"{gap.crossing(x, g1, 0.):,.0f} (checkpoint resolution, no interval); at 400 / 880 / 2,080 / the end "
              + " / ".join(f"{g1[_k(x, xv)]:+.2f}" for xv in (400, 880, 2080, x[-1])))
        line = ("     gain on naive's increment per step (ck reg_*: Sigma d N / Sigma N N), steps 1 / 2 / 5 / 2-30 "
                "mean / "
                "6-30 mean / 30:")
        for xv in (400, 720, 880, 1120, 1600, 2080, 4000, 12000, x[-1]):
            k = _k(x, xv)
            line += (f" {x[k]:,.0f}: {bN[k, 0]:.3f} / {bN[k, 1]:.2f} / {bN[k, 4]:.2f} / {bN[k, 1:].mean():.3f} / "
                     f"{bN[k, 5:].mean():.2f} / {bN[k, -1]:.2f};")
        print(line)
        if run == "trained":
            bb = t["D"]["beta_N_bayes"]
            print(f"     Bayes's per-step slope on naive's increment, D: steps 1 / 2 / 5 / 6-30 mean / 30 "
                  f"{bb[0, 0]:.2f} / "
                  f"{bb[0, 1]:.2f} / {bb[0, 4]:.2f} / {bb[0, 5:].mean():.2f} / {bb[0, -1]:.2f}; memory-free SW: "
                  f"{bb[1, 0]:.2f} / {bb[1, 1]:.2f} / {bb[1, 4]:.2f} / {bb[1, 5:].mean():.2f} / {bb[1, -1]:.2f}")
        if run in ("trained", "echo"):
            D_ = t["D"]
            fix = D_["cw5_all"] - D_["l5_all"]                                   # wrong at T, right under the cut
            bfix = D_["Bcw5_all"] - D_["bl5_all"]
            print(f"     the w5 cut breaks / fixes (share of all probes; Bayes's window: "
                  f"{D_['Bcw5_all'].sum() / 1e4:.3f}"
                  f" / {bfix.sum() / 1e4:.3f}) at 800 / 2,000 / the end: " + "; ".join(
                      f"{D_['cw5_all'][_k(x, xv)].sum() / 1e4:.3f} / {fix[_k(x, xv)].sum() / 1e4:.3f}"
                      for xv in (800, 2000, x[-1])))
        if run != "trained":
            continue
        ob = t["own_B"]
        k16 = _k(x, 1600)
        print(f"     the preliminary's comparison at {x[k16]:,.0f}: w5 loss on the network's own revisions "
              f"{t['own_l5'][0][k16]:.3f} vs Bayes's w5 loss on the same probes {t['own_bl5'][k16]:.3f}; on all probes "
              f"{t['l5_all'][0][k16]:.3f} vs {t['bl5_all'][0]:.3f}; full-sample excess, all probes: largest over "
              f"checkpoints {t['x5_all'][0].max():+.4f} at {x[int(np.argmax(t['x5_all'][0]))]:,.0f}")
        line = (f"     the preliminary's rule on each observer's own trajectory (wrong commitment > .6 at a step 8-24, "
                f"right at T): Bayes: share of probes {ob['share']:.3f}, its w5 loss there {ob['l5']:.3f}, share of "
                f"its "
                f"w5 loss {ob['part']:.2f} (enrichment {ob['part'] / ob['share']:.1f}x); network (share of probes, w5 "
                f"loss there, share of its w5 loss, enrichment = share of loss / share of probes):")
        for xv in (800, 1120, 1600, 2000, 4000, 12000, x[-1]):
            k = _k(x, xv)
            line += (f" {x[k]:,.0f}: {t['own_share'][0][k]:.3f}, {t['own_l5'][0][k]:.3f}, {t['own_part'][0][k]:.2f}, "
                     f"{t['own_part'][0][k] / t['own_share'][0][k]:.1f}x;")
        print(line)
    return S


# ═══════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════

FAM_COL = {"answers": "#6a3d9a", "holding": "#3a7a19", "coupling": "#b2182b"}     # td_tt_timeline's family colours
LATE_SPAN = "#f3ead7"                                                            # the late family's half-way span
U_TITLE = r"memory use $u=\mathrm{err}_{\mathrm{cut}}-\mathrm{err}_{\mathrm{intact}}$"
XMIN = 8.


def _xaxis(ax, xmax, label=True):
    ax.set_xscale("log")
    ax.set_xlim(XMIN, xmax)
    ax.set_xticks([10, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.minorticks_off()
    if label:
        ax.set_xlabel("training episodes")


def _band(ax, x, yb, col, alpha):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(yb, [5, 95], 0)
    ax.fill_between(x, lo, hi, color=col, alpha=alpha, lw=0, zorder=2)


def _ylog_interp(x, y, xv):
    return float(np.interp(np.log(xv), np.log(x), y))


def _sg(v, nd=3):
    """a signed number for figure text (mathtext minus: cmr10 has no minus glyph)"""
    return f"{v:+.{nd}f}".replace("-", "$-$")


def _mark(ax, c, y, col, filled, xmax, ms=7., zorder=8, marker="o"):
    """a crossing c at height y: marker at the point estimate, horizontal 5-95% bar"""
    if c is None or not np.isfinite(c["p"]) or c["share"] < .95:
        return
    lo = c["lo"] if np.isfinite(c["lo"]) else c["p"]
    hi = c["hi"] if np.isfinite(c["hi"]) else xmax
    ax.plot([lo, hi], [y, y], color=col, lw=2.0 if filled else 1.6, solid_capstyle="butt", zorder=zorder)
    ax.plot([c["p"]], [y], marker=marker, ms=ms, mfc=col if filled else "w", mec="k" if filled else col, mew=1.3,
            zorder=zorder + 1, ls="none")


def figure(S=None):
    """-> figure_bin/td_tt_memory_events.png"""
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtr
    from matplotlib.lines import Line2D
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    FL, TS = fs["label"], 12.
    S = summary() if S is None else S
    m, me = S["mem"]["trained"], S["mem"]["echo"]
    t = S["trunc"]["trained"]
    rows = S["rows"]
    x = m["x"]
    xmax = float(np.exp(np.log(x[-1]) + .12))
    W = 16.2
    L, PW, GX = 1.0, 4.25, 1.05
    PH1, PH2, PH3 = 2.75, 3.75, 2.75
    TOP, G12, G23, BOT = 1.65, 1.7, 1.75, 2.5
    H = TOP + PH1 + G12 + PH2 + G23 + PH3 + BOT
    fig = plt.figure(figsize=(W, H))
    y1 = H - TOP - PH1
    y2 = y1 - G12 - PH2
    y3 = y2 - G23 - PH3
    ax_a, ax_b, ax_c = (fig.add_axes([(L + i * (PW + GX)) / W, y1 / H, PW / W, PH1 / H]) for i in range(3))
    LABW = 3.55
    ax_d = fig.add_axes([(L + LABW) / W, y2 / H, (3 * PW + 2 * GX - LABW) / W, PH2 / H])
    PWE, PWF = 3.95, 4.85
    PWG = 3 * PW + 2 * GX - PWE - PWF - 2 * GX
    ax_e = fig.add_axes([L / W, y3 / H, PWE / W, PH3 / H])
    ax_f = fig.add_axes([(L + PWE + GX) / W, y3 / H, PWF / W, PH3 / H])
    ax_g = fig.add_axes([(L + PWE + PWF + 2 * GX) / W, y3 / H, PWG / W, PH3 / H])
    span = {}
    for fam, names in (("holding", HOLD_ROWS), ("coupling", COUPLE_ROWS)):
        for lev in (.1, .5):
            pts = [rows[n]["L"][lev]["p"] for n in names]
            span[fam, lev] = (min(pts), max(pts))
    acc0, acc50 = S["acc"][0.]["p"], S["acc"][.5]["p"]
    late50 = (min(span["holding", .5][0], span["coupling", .5][0]), max(span["holding", .5][1],
                                                                          span["coupling", .5][1]))
    LIGHT = {"trained": "#8fbde3", "echo": "#ffbe86"}                                  # least entangled fifth (c)
    RULE = "#138f8f"                                                                   # the memoryless linear rules (g)

    def acc_lines(ax):
        for v, ls in ((acc0, ":"), (acc50, (0, (5, 2, 1, 2)))):
            ax.axvline(v, color="k", lw=1.0, ls=ls, zorder=1)

    def spans(ax, fam):
        """the family's onset (c10) and half-way (c50) spans over its rows' crossings, as two strips at the bottom"""
        tr_ = mtr.blended_transform_factory(ax.transData, ax.transAxes)
        for lev, y0, a_, lab in ((.1, .03, .35, "onsets"), (.5, .105, .75, "half-way")):
            s0, s1 = span[fam, lev]
            ax.fill_between([s0, s1], [y0, y0], [y0 + .055, y0 + .055], transform=tr_, color=FAM_COL[fam], alpha=a_,
                            lw=0, zorder=2)
            ax.text(s0 / 1.06, y0 + .0275, f"{fam} {lab}", transform=tr_, ha="right", va="center", fontsize=FL - 1.5,
                    color=FAM_COL[fam], zorder=6, bbox=dict(boxstyle="square,pad=.08", fc="w", ec="none", alpha=.9))

    def mem_panel(ax, nm, fam, letter, title, extra=None):
        c, ce = m[nm], me[nm]
        spans(ax, fam)
        acc_lines(ax)
        ax.axhline(0, color="k", lw=.8, zorder=1)
        _band(ax, x, c["ev_u"][1], COL["trained"], .22)
        _band(ax, x, c["ctl_u"][1], COL["trained"], .12)
        ax.plot(x, c["ev_u"][0], color=COL["trained"], lw=2.3, zorder=4)
        ax.plot(x, c["ctl_u"][0], color=COL["trained"], lw=1.5, ls=(0, (4, 2)), zorder=4)
        ax.plot(x, ce["ev_u"][0], color=COL["echo"], lw=1.4, zorder=3)
        ax.plot(x, ce["ctl_u"][0], color=COL["echo"], lw=1.0, ls=(0, (4, 2)), zorder=3)
        if extra is not None:
            extra(ax)
        _mark(ax, c["helps"], 0., COL["trained"], False, xmax)
        b = c["beats"]
        if np.isfinite(b["p"]):
            _mark(ax, b, _ylog_interp(x, c["ev_u"][0], b["p"]), COL["trained"], True, xmax)
        _xaxis(ax, xmax)
        ax.set_ylabel("memory use (nats)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{letter}   {title}\n" + U_TITLE, loc="left", fontsize=TS, linespacing=1.45)

    mem_panel(ax_a, "hold", "holding", "a", "Holding events: Bayes's end splits")

    def axis_line(ax):
        ax.plot(x, m["axis"]["ev_u"][0], color=COL["trained"], lw=1.4, ls=(0, (1, 1.4)), zorder=4)

    mem_panel(ax_b, "couple", "coupling", "b", "Coupling events: Bayes's large diagonal moves", axis_line)
    ed = m["couple"]["end_diff"]
    ax_b.legend(handles=[Line2D([], [], color=COL["trained"], lw=1.4, ls=(0, (1, 1.4)),
                                label="contrast: Bayes's large\none-axis moves (no coupling)")],
                loc="upper left", frameon=False, fontsize=FL - .5, handlelength=2.2)
    ax_b.text(.99, .99, f"never beats its matched control\n(events $-$ control at the end {_sg(ed[0])})",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=FL - 1, color=COL["trained"], linespacing=1.3,
              zorder=7, bbox=dict(boxstyle="square,pad=.15", fc="w", ec="none", alpha=.9))
    # c: entanglement fifths
    ax = ax_c
    acc_lines(ax)
    ax.axhline(0, color="k", lw=.8, zorder=1)
    for nm, dark in (("q5", True), ("q1", False)):
        c, ce = m[nm], me[nm]
        ct, cE = (COL["trained"], COL["echo"]) if dark else (LIGHT["trained"], LIGHT["echo"])
        _band(ax, x, c["ev_u"][1], ct, .22 if dark else .3)
        ax.plot(x, c["ev_u"][0], color=ct, lw=2.3 if dark else 2.0, zorder=5 if dark else 4)
        ax.plot(x, c["ctl_u"][0], color=ct, lw=1.4, ls=(0, (4, 2)), zorder=5 if dark else 4)
        ax.plot(x, ce["ev_u"][0], color=cE, lw=1.3, zorder=3)
    c = m["q5"]
    _mark(ax, c["helps"], 0., COL["trained"], False, xmax)
    _mark(ax, c["beats"], _ylog_interp(x, c["ev_u"][0], c["beats"]["p"]), COL["trained"], True, xmax)
    _xaxis(ax, xmax)
    ax.set_ylabel("memory use (nats)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("c   Entangled steps (fifths of own entanglement)\n" + U_TITLE, loc="left", fontsize=TS,
                 linespacing=1.45)
    ax.legend(handles=[Line2D([], [], color=COL["trained"], lw=2.3, label="most entangled fifth"),
                       Line2D([], [], color=COL["trained"], lw=1.4, ls=(0, (4, 2)), label="... its matched control"),
                       Line2D([], [], color=LIGHT["trained"], lw=2.0, label="least entangled fifth"),
                       Line2D([], [], color=LIGHT["trained"], lw=1.4, ls=(0, (4, 2)), label="... its matched control"),
                       Line2D([], [], color=COL["echo"], lw=1.3, label="ESN, most entangled fifth"),
                       Line2D([], [], color=LIGHT["echo"], lw=1.3, label="ESN, least entangled fifth")],
              loc="upper left", frameon=False, fontsize=FL - 1., handlelength=2.2, labelspacing=.3)
    lo = min(a_.get_ylim()[0] for a_ in (ax_a, ax_b, ax_c))
    hi = max(a_.get_ylim()[1] for a_ in (ax_a, ax_b, ax_c))
    for a_ in (ax_a, ax_b):
        a_.set_ylim(lo - .09, hi * 1.25)
    ax_c.set_ylim(-.09, .135)
    # shared legend for a-c
    h = [Line2D([], [], color=COL["trained"], lw=2.3, label="Trained RNN, at the events (band 5-95%)"),
         Line2D([], [], color=COL["trained"], lw=1.5, ls=(0, (4, 2)), label="Trained RNN, matched control steps"),
         Line2D([], [], color=COL["echo"], lw=1.4, label="ESN, at the events (a, b dashed: its controls)"),
         Line2D([], [], color=COL["trained"], marker="o", ms=7, mfc="w", mec=COL["trained"], lw=1.6,
                label="memory helps for good ($u>0$ to the end)"),
         Line2D([], [], color=COL["trained"], marker="o", ms=7, mfc=COL["trained"], mec="k", lw=2.,
                label="beats its matched control for good"),
         Line2D([], [], color="k", lw=1., ls=":", label=f"accuracy passes naive ({acc0:,.0f})"),
         Line2D([], [], color="k", lw=1., ls=(0, (5, 2, 1, 2)), label=f"accuracy half-way to Bayes ({acc50:,.0f})")]
    fig.legend(handles=h, loc="center", bbox_to_anchor=(.5, (y1 - .93) / H), ncol=4, frameon=False, fontsize=FL - .3,
               columnspacing=1.6, handlelength=2.4)
    # d: crossings
    ax = ax_d
    beh = [("accuracy passes naive", "answers", [S["acc"][0.]], False),
           ("accuracy half-way", "answers", [S["acc"][.5]], True),
           ("holding onsets (5 measures, c10)", "holding", [rows[n]["L"][.1] for n in HOLD_ROWS], False),
           ("holding half-way (5 measures, c50)", "holding", [rows[n]["L"][.5] for n in HOLD_ROWS], True),
           ("coupling onsets (6 measures, c10)", "coupling", [rows[n]["L"][.1] for n in COUPLE_ROWS], False),
           ("coupling half-way (6 measures, c50)", "coupling", [rows[n]["L"][.5] for n in COUPLE_ROWS], True)]
    rf = t["rf_all"]
    q5c = m["q5"]["clock"][MATCH[0]]
    refs = [("stored cut", q5c["cut"]), ("step-1 linear rule", q5c["s1"]), ("clock-matched rule", q5c["st"]),
            (f"cut gap frozen at {q5c['x_freeze']:,.0f}", q5c["frozen"])]
    mem = [("most entangled fifth: beats matched steps,\ndated by each memoryless reference (g)", "refs", refs,
            m["q5"]["beats"]["p"], 2.5),
           ("long-memory excess at T: first reaches half its peak", "rise", rf["rise"], rf["rise"]["p"], 1.),
           ("most entangled fifth: memory helps", "helps", m["q5"]["helps"], m["q5"]["helps"]["p"], 1.),
           ("Bayes's end splits: memory helps", "helps", m["hold"]["helps"], m["hold"]["helps"]["p"], 1.),
           ("large diagonal moves: memory helps", "helps", m["couple"]["helps"], m["couple"]["helps"]["p"], 1.),
           ("Bayes's end splits: beats its matched control", "beats", m["hold"]["beats"], m["hold"]["beats"]["p"], 1.),
           ("large diagonal moves: beats its matched control", "beats", m["couple"]["beats"],
            m["couple"]["beats"]["p"], 1.),
           ("long-memory excess at T: below half its peak for good", "fall", rf["fall"], rf["fall"]["p"], 1.)]
    mem = sorted(mem, key=lambda r: (not np.isfinite(r[3]), r[3] if np.isfinite(r[3]) else 0.))
    GAPD = .7
    ypos, yy = [], 0.
    for i in range(len(beh)):
        ypos.append(yy)
        yy += 1.
    yy += GAPD
    for r in mem:
        ypos.append(yy + (r[4] - 1.) / 2)
        yy += r[4]
    trd = mtr.blended_transform_factory(ax.transAxes, ax.transData)
    for i, (lab, fam, cs, filled) in enumerate(beh):
        offs = np.linspace(-.3, .3, len(cs)) if len(cs) > 1 else [0.]
        for c, o in zip(cs, offs):
            _mark(ax, c, ypos[i] + o, FAM_COL[fam], filled, xmax, ms=5.5 if len(cs) > 1 else 7.)
        ax.text(-.01, ypos[i], lab, transform=trd, ha="right", va="center", fontsize=FL, color=FAM_COL[fam])
    shape = {"rise": ("^", False), "fall": ("v", True), "helps": ("o", False), "beats": ("o", True)}
    for j, (lab, kind, c, _, hgt) in enumerate(mem):
        i = len(beh) + j
        if j % 2 == 0:
            ax.axhspan(ypos[i] - hgt / 2, ypos[i] + hgt / 2, color="#eef3f9", lw=0, zorder=0)
        ax.text(-.01, ypos[i], lab, transform=trd, ha="right", va="center", fontsize=FL, color=COL["trained"],
                linespacing=1.2)
        if kind == "refs":
            for (rl, rc), o in zip(c, np.linspace(-.84, .84, len(c))):
                _mark(ax, rc, ypos[i] + o, COL["trained"], True, xmax, ms=6.)
                if np.isfinite(rc["p"]):
                    xr = rc["hi"] if np.isfinite(rc["hi"]) else rc["p"]
                    ax.text(xr * 1.1, ypos[i] + o, f"{rl}: {rc['p']:,.0f}", ha="left", va="center",
                            fontsize=FL - 1.5, color=COL["trained"], zorder=6,
                            bbox=dict(boxstyle="square,pad=.08", fc="w", ec="none", alpha=.9))
            continue
        mk_, filled = shape[kind]
        if np.isfinite(c["p"]):
            _mark(ax, c, ypos[i], COL["trained"], filled, xmax, ms=7.5 if mk_ != "o" else 7., marker=mk_)
        else:
            e_ = m["couple"]["end_diff"]
            ax.text(.995, ypos[i], f"never: ends below its matched control ({_sg(e_[0])} [{_sg(e_[1][0])}, "
                                   f"{_sg(e_[1][1])}])", transform=trd, ha="right", va="center", fontsize=FL - .5,
                    color=COL["trained"])
    ax.axhline(ypos[len(beh) - 1] + .5 + GAPD / 2, color="k", lw=.8)
    acc_lines(ax)
    ax.set_ylim(yy - .3, -.7)
    ax.set_yticks([])
    _xaxis(ax, xmax)
    ax.set_xlim(40, xmax)
    ax.set_xticks([50, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4])
    ax.set_xticklabels(["50", "100", "300", "1k", "3k", "10k", "30k"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="0.9", lw=.6, zorder=0)
    ax.set_title("d   When memory turns helpful, against the behaviour (Trained RNN; circles: crossings kept to the "
                 "end, "
                 "open = memory helps, filled = beats its matched\ncontrol; triangles: the excess at T, open = first "
                 "reaches half its peak, filled = below half its peak for good; bars 5-95% over the same 1,000 "
                 "environment resamples)", loc="left", fontsize=TS - .5, linespacing=1.35)
    # e: the ceiling-free excess by family
    ax = ax_e
    ax.axvspan(*late50, color=LATE_SPAN, lw=0, zorder=0)
    ax.text(np.sqrt(late50[0] * late50[1]), .985, "holding and\ncoupling\nhalf-way",
            transform=mtr.blended_transform_factory(ax.transData, ax.transAxes), ha="center", va="top", fontsize=FL - 1,
            color="#7a5c1e", linespacing=1.1)
    acc_lines(ax)
    ax.axhline(0, color="k", lw=.8, zorder=1)
    fam_style = (("all", "-", 2.4), ("settled", (0, (1, 1.3)), 1.9), ("rev", (0, (5, 2)), 1.8),
                 ("nfar", (0, (5, 2, 1, 2)), 1.6))
    fam_lab = {"all": "all probes", "settled": "Bayes settled from step 8", "rev": "Bayes revises (a commitment\n"
               r"$>$ .6 later abandoned)", "nfar": "naive's answer 2+ cells off"}
    _band(ax, x, t["cx5_all"][1], COL["trained"], .2)
    for f, ls, lw in fam_style:
        ax.plot(x, t[f"cx5_{f}"][0], color=COL["trained"], ls=ls, lw=lw, zorder=4)
    ax.plot(x, S["trunc"]["echo"]["cx5_all"][0], color=COL["echo"], lw=1.5, zorder=3)
    _xaxis(ax, xmax)
    ax.set_ylabel("excess w5 loss of right answers")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([], [], color=COL["trained"], ls=ls, lw=lw, label=fam_lab[f]) for f, ls, lw in fam_style]
              + [Line2D([], [], color=COL["echo"], lw=1.5, label="ESN, all probes")],
              loc="upper left", frameon=False, fontsize=FL - 1., handlelength=2.4, labelspacing=.3)
    ax.set_ylim(-.12, .92)
    ax.set_title("e   Long-memory excess at T of right answers, by\nepisode family (Trained RNN): "
                 r"$P(\mathrm{cut\,5\ wrong}\,|\,\mathrm{right})$" "\n"
                 r"$-\,P_{\mathrm{Bayes}}(\mathrm{window\,5\ wrong}\,|\,\mathrm{right})$, no ceiling",
                 loc="left", fontsize=TS - .5, linespacing=1.4)
    # f: every run
    ax = ax_f
    ax.axhline(0, color="k", lw=.8, zorder=1)
    for run in TRUNC_RUNS:
        tt = S["trunc"][run]
        y = tt["x5_all"][0]
        ax.plot(tt["x"], y, color=COL[run], lw=2.3 if run == "trained" else 1.6, zorder=4 if run == "trained" else 3)
        for key, mk_, filled in (("acc_c0", "o", False), ("acc_c50", "o", True)):
            c = tt[key]
            if np.isfinite(c["p"]) and c["share"] >= .95:
                ax.plot([c["p"]], [_ylog_interp(tt["x"], y, c["p"])], marker=mk_, ms=7, mfc=COL[run] if filled else "w",
                        mec="k" if filled else COL[run], mew=1.3, zorder=6, ls="none")
        rr = tt["rf_all"]
        for part, mk_, filled in (("rise", "^", False), ("fall", "v", True)):
            c = rr[part]
            if np.isfinite(c["p"]) and c["share"] >= .95 and rr["peak"] > 0:
                ax.plot([c["p"]], [rr["peak"] / 2], marker=mk_, ms=8, mfc=COL[run] if filled else "w",
                        mec="k" if filled else COL[run], mew=1.3, zorder=7, ls="none")
    _xaxis(ax, xmax)
    ax.set_ylim(-.06, .245)
    ax.set_ylabel("excess w5 loss (accuracy at T)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([], [], color=COL[r], lw=2., label=NAME[r] + ("" if r in ("trained", "echo") else
                                                                            " (LSTM)")) for r in TRUNC_RUNS],
              loc="upper left", frameon=False, fontsize=FL - 1., handlelength=1.8, labelspacing=.3)
    ax.add_artist(ax.get_legend())
    ax.legend(handles=[Line2D([], [], color="k", marker="o", ms=7, mfc="w", mec="k", ls="none",
                              label="the run's accuracy passes naive"),
                       Line2D([], [], color="k", marker="o", ms=7, mfc="k", mec="k", ls="none",
                              label="... is half-way to Bayes"),
                       Line2D([], [], color="k", marker="^", ms=8, mfc="w", mec="k", ls="none",
                              label="excess first reaches half its peak"),
                       Line2D([], [], color="k", marker="v", ms=8, mfc="k", mec="k", ls="none",
                              label="... below half its peak for good")],
              loc="upper right", frameon=False, fontsize=FL - 1., handlelength=1.2, labelspacing=.3)
    r1, rp = S["run_rel"]["trained"][0.][1], S["run_rel"]["trained_p1"][0.][1]
    ax.set_title("f   Long-memory excess at T in every trained run (all probes)\n"
                 r"$x=(R_{\mathrm{intact}}-R_{\mathrm{cut}\,5})-(R_{\mathrm{Bayes}}-R_{\mathrm{Bayes}\,5})$"
                 " per probe: it "
                 "rises with or before\npassing naive (just after it in "
                 f"{1 - r1:.0%} / {1 - rp:.0%} of paired resamples for the\nTrained RNN / power-1) and relaxes only in "
                 "the runs that approach Bayes,\nexcept N = 50, whose excess also ends below half its small peak",
                 loc="left", fontsize=TS - .5, linespacing=1.4)
    # g: what dates the entangled-steps contrast
    ax = ax_g
    g = q5c["gap"]
    acc_lines(ax)
    ax.axhline(0, color="k", lw=.8, zorder=1)
    _band(ax, x, g["int"][1], COL["trained"], .2)
    ax.plot(x, g["int"][0], color=COL["trained"], lw=2.4, zorder=5)
    ax.plot(x, g["cut"][0], color=COL["trained"], lw=1.6, ls=(0, (4, 2)), zorder=5)
    ax.plot(x, g["s1"][0], color=RULE, lw=1.5, zorder=4)
    ax.plot(x, g["st"][0], color=RULE, lw=1.5, ls=(0, (1, 1.2)), zorder=4)
    kf = int(np.argmin(np.abs(x - X_FREEZE)))
    ax.plot([x[kf], xmax], [g["cut"][0][kf]] * 2, color=COL["trained"], lw=1.2, ls=(0, (2.5, 2.5)), zorder=4)
    for q in ("cut", "s1", "st", "frozen"):
        c = q5c[q]
        if np.isfinite(c["p"]) and c["share"] >= .95:
            ax.plot([c["p"]], [_ylog_interp(x, g["int"][0], c["p"])], marker="o", ms=6, mfc=COL["trained"], mec="k",
                    mew=1.1, zorder=7, ls="none")
    _xaxis(ax, xmax)
    ax.set_ylim(-.005, .075)
    ax.set_ylabel("error gap (nats)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([], [], color=COL["trained"], lw=2.4, label="intact network (band 5-95%)"),
                       Line2D([], [], color=COL["trained"], lw=1.6, ls=(0, (4, 2)),
                              label=f"stored cut: passes it at {q5c['cut']['p']:,.0f}"),
                       Line2D([], [], color=RULE, lw=1.5, label=f"step-1 linear rule: {q5c['s1']['p']:,.0f}"),
                       Line2D([], [], color=RULE, lw=1.5, ls=(0, (1, 1.2)),
                              label=f"clock-matched rule: {q5c['st']['p']:,.0f}"),
                       Line2D([], [], color=COL["trained"], lw=1.2, ls=(0, (2.5, 2.5)),
                              label=f"cut gap frozen at {x[kf]:,.0f}: {q5c['frozen']['p']:,.0f}")],
              loc="upper left", frameon=False, fontsize=FL - 1., handlelength=2.2, labelspacing=.3)
    ax.set_title("g   Entangled steps: the memoryless reference\nsets the contrast's date (error gap, most\n"
                 r"entangled fifth $-$ matched steps; dots: crossings)", loc="left", fontsize=TS - .5, linespacing=1.4)
    # footnote and title
    nh, nc, nq = m["hold"]["n_ev"], m["couple"]["n_ev"], m["q5"]["n_ev"]
    covs = [m[nm]["n_cov"] / m[nm]["n_ev"] for nm in ("hold", "couple", "q5", "q1")]
    note = (
        "Data (seed 20261029): 10,000 probe episodes (50 test environments x 100 latent pairs x 2 trajectories, 30 "
        "steps), 114 checkpoints per run; memory cuts stored per probe, step and variable (Trained RNN, ESN) and at T "
        "(all runs).\n"
        f"Events (variable-steps, steps 2-30, defined on Bayes's beliefs): end splits {nh:,.0f}; large diagonal moves "
        f"{nc:,.0f} ({nc / 2:,.0f} moves; one-axis {m['axis']['n_ev']:,.0f}); entanglement fifths about "
        f"{round(nq, -3):,.0f} each.\n"
        "Matched controls: other variable-steps weighted to the events' histogram of step x log10 FR (KL(Bayes "
        r"$\Vert$ naive)) x Bayes's entropy "
        f"x log10 Bayes's update size (coverage {min(covs):.0%}-{max(covs):.0%}). Bands and bars: 5-95% over 1,000 "
        "resamples of the 50 environments, paired with the timeline's behaviour rows.\n"
        r"err = Jeffreys/2 between Bayes's belief at $t-1$ moved by the network's step-$t$ increment and Bayes's "
        r"belief at $t$; cut: the cell rerun from its trained start state over input $t$ only; cut 5: over the last 5 "
        "inputs (windowed Bayes: the same for Bayes).\n"
        "Linear rules (g): the network's step-1 or step-$t$ regression coefficients on naive's increment and the "
        "step-wise correction, applied to Bayes's belief at $t-1$. Every stored cut restarts from the trained start "
        "state, so it removes\nthe state's clock (the step-1 / step-5 regime) as well as its content; a donor-state "
        "cut separates the two. Families (goal variable): revises = Bayes puts more than .6 on a value at a step 8-24 "
        "that is not its answer at T;\nsettled = Bayes's answer fixed from step 8. e, f: R = right at T; cut 5 = the "
        "network's answers with the memory cut to the last 5 steps; Bayes 5 / window 5 = windowed Bayes over the last "
        "5 observations.\ne counts only the right answers the cut breaks, so it is positive for the ESN too, whose "
        "w5 cut fixes about as many answers as it breaks (f: its total excess is negative).")
    fig.text(.5, .98 / H, note, ha="center", va="center", fontsize=FL - 1.2, linespacing=1.55)
    hs, cs_, hb = m["hold"]["helps"], m["couple"]["helps"], m["hold"]["beats"]
    fig.suptitle(f"Memory-cut intervention: through the answers phase the network's state makes its updates at Bayes's "
                 f"end splits and large moves worse than a memoryless rerun;\nits update beats the memoryless one "
                 f"there only from {hs['p']:,.0f}-{cs_['p']:,.0f} training episodes (after the answers are half-way, "
                 f"{acc50:,.0f}); at splits the memoryless rerun itself does not improve through 4,000.\nMatched on "
                 f"Bayes's update size, memory beats matched steps at splits from {hb['p']:,.0f} but never at large "
                 f"diagonal moves; the long-memory excess at T is not concentrated on revisions",
                 y=1 - .25 / H, fontsize=TS + .5, linespacing=1.4)
    path = fig_path("td_tt_memory_events.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    what = sys.argv[1:] or ["numbers", "figure"]
    if "build" in what:
        build([r for r in what if r in TRUNC_RUNS] or None)
    S = None
    if "numbers" in what:
        S = numbers()
    if "figure" in what:
        figure(S)
