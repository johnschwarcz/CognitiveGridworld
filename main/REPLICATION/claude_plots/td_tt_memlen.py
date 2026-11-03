"""Memory length through training: memory cuts at every window w = 1..15 (John 11-02: "It seems like the relative
benefit from different memory lengths changes throughout training. Maybe looking more closely at all w = 1 to 15 would
reveal more"; "regarding the memory cut, I liked the swapping the hidden state with random (not start) the most").
Inference only, on the stored checkpoints. Revised 11-02 (round 3) after an adversarial check (FIXABLE): the dates are
labelled as the donor scheme's and two other schemes are run beside it (the order by w is the headline); an
accuracy-matched perfect-memory observer is added (the order is not an accuracy effect; the w = 1 date follows the
intact accuracy); the benefit per memory step is reported in bands against Bayes's, paired (the earlier average over
steps 3-15 was an artifact); the restart's extra cost is dated from where it starts (before passing naive) and its falls
are timed at absolute levels; the ESN's span grows slightly; the width / loss runs replicate it; all 114 Trained RNN
checkpoints; the figure is cut from 12 panels to 6 and leads with the random cell. Round 4 (11-02, after a second check;
wording and figure only, no number changed): the narrower runs do not stop where their accuracy stops (result 10); only
the onsets move one way, the recent steps' worth rises and falls and ends slightly reversed, not at Bayes's (results 2,
5); ranges are read at every checkpoint (numbers() prints the full traces); panel c draws Bayes's per-step levels.
-> main/REPLICATION/figure_bin/td_tt_memlen.png   figure()
   numbers()  every number below (~3 min);  checks() -> OUT/checks.json (~5-10 min on GPU 1)
   cache main/DATA/experimental/training_dynamics/through_training/memlen/ (OUT, 165 MB): bayes.npz (Bayes intact and
   under the 45 cuts, naive), bayes_schemes.npz (Bayes under the other donor schemes), thinned.npz (the thinned
   observer's per-environment counts), <run>/ck_EEEEE.npz for trained (114), echo (30), trained_p1 / N200 / N100 /
   N50 (114 each): per checkpoint the answers at T of every probe under the 46 conditions (intact + 3 sources x 15
   windows), the goal answers right per environment and step, the environments' sums of log q of the true goal value
   at T; schemes/ck_EEEEE.npz (114): the Trained RNN's answers at T intact and under the two other donor schemes
   (random cell, w = 1..15). Every number below is rebuilt from these.

READING. "The answers depend on the content older than w steps" means: replacing the state at t - w by another cell's
  state changes the answers at T (the cut costs accuracy). It is not "uses" or "needs memory": the random-cell cut is a
  ceiling, not a measure of reliance (td_tt_memcuts): a wrong cell's history misleads every observer, and the loss is
  bounded by each observer's own gain. So the loss is always read against an observer under the identical substitution:
  Bayes (the onset level), and a perfect-memory observer of the network's own accuracy (the order's control).

DATA
  Trained RNN (LSTM N = 1000, seed 20261029): all 114 stored checkpoints. Width / loss replication runs (same seed,
  environment and episode sequence; td_tt_replication): trained_p1 (N = 1000, loss power 1), trained_N200 / N100 / N50
  (power .5): 114 each. ESN: 30 (every 4th from the initial network, and the last). x = training episodes
  (td_tt_gap.train_x; the initial network at 10). Probes (analysis/probes.npz): 10,000 episodes = 50 test environments
  x 100 latent pairs (cells) x 2 trajectories, 30 steps. Weights: td_tt_intervene.load; the drive table, the cell and
  the engine's donors are td_tt_intervene's. Passing naive / half-way of every run: td_tt_replication's synthesis (goal
  accuracy at T past naive's level for good, c0, and past the naive-Bayes midpoint, c50; the same 1,000 resamples).
  The replication runs' cut files were computed with this module's run_ck by the session's GPU 1 helper (scratchpad
  memflip/helper/widths_memlen.py, 11-02), and pass the same stored-truncation check.

DEFINITIONS (td_tt_memcuts' convention)
  Memory cut to window w from a source: at every step t > w the cell is rerun over the probe's own inputs t-w+1..t from
    a state placed at t - w, and the read-out of the rerun state at t replaces the network's increment at t; t <= w
    keeps the network's own increments; the belief at every step is the cumulative sum (only what the STATE carries
    from before t - w + 1 is removed, never earlier increments). Sources of the state at t - w: start = the trained
    start state (h0, c0); same cell = the state at t - w of the other trajectory of the same cell and environment
    (probe row ^ 1); random cell = the state at t - w of another probe of the same environment with another cell.
  Donor schemes of the random cell: 'engine' (the main one) = one fixed donor per probe (seed 20261201, the same donor
    at every window start); 'another fixed seed' = the same scheme under seed 20261299; 'fresh' = a new random
    other-cell probe of the environment at every window start a (seed 777). The two others are the adversarial check's
    (check_memlen/c5_fresh.py), rebuilt exactly.
  Route (one rerun per start offset): for every a = 1..29 the cell is rerun from the source's state at a (the start
    state, or the donor's intact state after a steps; 'fresh': the donor of window start a) over the probe's own inputs
    a+1..min(a + 15, 30); the read-out after k rerun steps is the cut increment at t = a + k for w = k. The cut belief
    with window w is the cumulative sum of the intact increments for t <= w and of the offset-(t - w) reruns'
    increments for t > w; log-softmax per variable in float64; answers = argmax (as the engine).
  Bayes under the same cut (exact, float64; td_tt_memcuts.bayes_logq's definition): for t > w Bayes's increment at t
    is its marginal change from adding observation t to the history (the donor's observations 1..t-w, own t-w+1..t-1;
    start: no donor observations = windowed Bayes); t <= w Bayes's own increment; the cumulative sum, normalised.
    Computed from the per-bit counts (k = K_donor(t - w) + K(t) - K(t - w)). Naive needs no memory.
  Perfect-memory observer of the network's accuracy ('the observer'): Bayes that sees each observation with
    probability p (mask u < p, u uniform per probe and step from default_rng(seed); nested in p; a probe's mask is the
    same when it is the receiver and when it is a donor), exact float64 from the masked counts, intact and under the
    random-cell (engine donors) and start-state cuts with the definition above; p in P_GRID (17 values, 0 to 1), mask
    seeds 7 and 8 (per-environment counts averaged). At each checkpoint its loss L_obs(w) is read at the p whose intact
    goal accuracy equals the network's (linear in accuracy between grid points; in every resample with that
    resample's accuracies on both sides). It loses evidence evenly over time, so it is what an observer with perfect
    memory loses under the same cut at the network's own accuracy.
  Loss L(w) = goal accuracy at T intact - under the cut (each observer's own intact level); excess = L(w) - L_Bayes(w)
    (> 0: the network loses more than Bayes would). Far errors added = far(cut) - far(intact), far = one or both
    answers at T 2 or more values off.
  Benefit of the k-th memory step = acc(cut to k) - acc(cut to k - 1) = L(k - 1) - L(k), in accuracy; reported per
    band of steps {2}, {3-6}, {7-9}, {10-15} (every step 2..15, split explicitly) as the band's mean per step,
    (L(k0 - 1) - L(k1)) / (k1 - k0 + 1), against Bayes's and the observer's, paired.
  Onset(w) = the training episode after which the network's L(w) stays above 1/2 x Bayes's L(w) under the same cut
    (td_tt_gap.crossing: stays past, log-interpolated; the level recomputed in every resample, paired; never = after
    the end). Trend = the least-squares slope of log10 onset on w, per resample (x per step = 10^slope). Own clock =
    onset / the run's passing naive, per resample. Accuracy at an onset = the run's intact goal accuracy at T at the
    onset (log-interpolated between checkpoints, per resample).
  Restart's extra cost = the start-state excess. Transient timing (td_tt_memory_events' rule): peak = the maximum of the
    3-checkpoint running mean; rise = the first crossing of half the peak; fall = after which it stays below half the
    peak; 'resolved' = the peak's 5% bound >= .005. Absolute levels: the time after its peak after which it stays below
    .05 / .02. Span = the largest w whose excess (or L - L_obs) exceeds .02 at a checkpoint. ESN span = the largest w
    with L(w) > .02.
  Statistics: per-environment sums, ratios of sums; 5-95% over the 1,000 shared environment resamples
    (td_tt_gap.boot_counts, seed 20261103), paired across conditions, checkpoints, observers, runs and the accuracy's
    crossings. Two times are ordered only when their intervals do not overlap ('with' otherwise); (p) = paired share
    P(first < second).

RESULTS (Trained RNN N = 1000 unless named; training episodes; [5-95%]; goal accuracy at T; random cell = the engine's
donors unless named)
 1 Bayes's loss under each cut (fixed): start .163 [.152, .176] (w = 1), .074 (2), .044 (3), .031 (5), .013 (10), .006
   [.003, .009] (15); same cell .021 [.014, .026] (1) to .008 (15); random cell .456 [.444, .468] (1), .426 (2), .400
   (3), .344 (5), .192 (10), .085 [.078, .091] (15): with a wrong cell's history each memory step is worth about the
   same to Bayes (.018-.034; per band: 2nd step .0293 [.0253, .0337], steps 3-6 .0290, 7-9 .0306, 10-15 .0224 [.0210,
   .0237]). Levels: Bayes intact .7711, naive .4641; the network .255 (320), .331 (560), .493 (880), .585 (1,200), .658
   (1,680), .701 (2,400), .733 (4,000), .760 (39,996).
 2 Random cell: the answers come to depend on the content from ever further back, shorter windows first. L(w) at 320
   (accuracy .255): .135 (w = 1), .048 (3), .020 (5), <= .013 from 6; 880: .339 / .120 (5) / .028 (10) / .007 (15);
   1,680: .467 / .202 / .065 / .016; 4,000: .462 / .287 / .127 / .041; end .432 [.417, .448] / .348 / .200 / .077
   [.071, .083] vs Bayes .456 / .344 / .192 / .085: by the end the loss is within .03 of Bayes's at every w, but its
   per-step profile is slightly reversed (result 5). Onsets (L stays above half of Bayes's): 631 [598-676] (w = 1),
   846 [814-885] (2), 928 (3), 1,053 (4), 1,233 [1,092-1,370] (5), 1,605 (6), 1,715 (7), 1,968 (8), 2,369 (9), 2,639
   [2,392-2,910] (10), 2,905 (11), 3,482 (12), 3,886 (13), 3,954 (14), 4,408 [3,884-5,539] (15). Shorter before
   longer in 88 of 105 pairs (intervals apart; every pair 4 or more steps apart), never the other way; adjacent P =
   1.00 for w = 1-11, .98 / .93 / .91 for 12-14; each extra step x1.148 [x1.142, x1.166] later. At 1/4 of Bayes's loss
   304 (w = 1) to 2,221 (15), x1.131 [1.124, 1.152], 87 of 105 pairs; at 3/4 887 to 10,738 [8,115-23,227], x1.178
   [1.157, 1.219], 80 of 105; none reversed at either level.
   The w = 1 date follows the intact accuracy: the w = 1 cut leaves the answers near chance (.105 at 120, .119 at 320,
   .140 at 560, .146 at 800, .153 at 880; chance .10) while the intact accuracy rises (.180 to .493), so L(1) is about
   the intact accuracy minus .13 and the onset is where that accuracy reaches .368 [.360, .378]. In this scheme w = 1
   comes before passing naive (P = 1.00), w = 2 with it (.30), w >= 3 after; w <= 4 before the half-way point, 5-6
   with it, w >= 7 after; these alignments depend on the scheme (result 3).
 3 Donor schemes: the order holds in every scheme, the dates do not. Another fixed seed (Bayes's L .466 at w = 1 to .081
   at 15): onsets 675 [628-728] (1), 873 (2), 1,245 [1,112-1,451] (5), 2,792 [2,547-3,481] (10), 3,865 [3,624-5,432]
   (15); 84 of 105 pairs, none reversed, x1.139 [1.130, 1.157] per step; with the engine's at every w. A fresh donor
   at every window start (Bayes's L .514 at w = 1, .332 at 5, .168 at 10, .070 at 15; the network's .280 / .076 / .020
   / .003 at 880 and .427 / .295 / .136 / .052 at the end): onsets 806 [759-858] (1), 1,141 [1,088-1,228] (2), 1,240
   (3), 1,587 (4), 2,273 [1,982-2,433] (5), 2,744 (6), 3,463 (7), 3,895 (8), 4,401 (9), 5,475 [4,928-6,149] (10),
   5,609 (11), 5,627 (12), 5,805 (13), 6,462 (14), 6,924 [6,126-11,347] (15); 87 of 105 pairs, none reversed, x1.164
   [1.161, 1.192]; after the engine's at every w (P = .00). Against the accuracy: another fixed seed as the engine
   (w = 1 before passing naive, w = 2 with it (P = .01), w <= 4 before half-way, 5-6 with it); fresh: w = 1 with passing
   naive (.84), w >= 2 after; w <= 2 before half-way, 3-4 with it, w >= 5 after. So only the order by w is robust:
   about 14-16% later per extra step in every scheme (td_tt_memcuts found the same shift at w = 5).
 4 Not an accuracy effect: a perfect-memory observer of the same accuracy shows no order. Its intact accuracy over
   P_GRID: .100 (p = 0), .120, .136, .152, .167, .196, .221, .244, .284, .334, .375, .429, .487, .554, .625, .700, .771
   (p = 1; monotone in p in every resample; the two mask seeds differ by <= .0093). Its L(w) reaches half of Bayes's at
   the same accuracy for every w: .501 [.492, .510] (w = 1), .501 (2), .505 (3), .514 (5), .521 (10), .530 [.506,
   .553] (11), .504 [.460, .564] (15); slope +.0015 [-.0016, +.0046] per step, against the network's +.0231 [+.0221,
   +.0244] (its accuracy at its onsets .368 (1), .472 (2), .511 (3), .542 (4), .591 (5), .643 (6), .664 (7), .680 (8),
   .698 (9), .707 (10), .713 (11), .723 (12), .731 (13), .732 (14), .734 [.721, .748] (15); minus the observer's,
   paired: -.132 [-.143, -.120] (1), -.030 (2), +.007 [-.009, +.021] (3), +.037 (4), +.077 (5), +.187 (10), +.230
   [+.174, +.276] (15)). On the network's own accuracy trajectory the observer would reach the criterion at 901
   [858-966] (w = 1) ... 997 (14), 908 (15): 0 of 105 pairs ordered, x1.005 [0.997, 1.016] per step. The network's onset
   comes before the observer's at w = 1, with it at 2-4 and after it from w = 5.
   What changes is the profile's shape: at the same accuracy the network loses more than the observer at short windows
   and less at long ones. L(w) - L_obs(w): 120 (accuracy .180) +.066 (w = 1) to +.002 (15), >= 0 at every w; 320 +.097
   (1), +.054 (2), +.015 (3), then -.001 to -.014; 880 +.118 (1), +.023 (2), -.006 (3), -.040 (5), -.060 (7-10), -.035
   (15); 1,680 +.108 (1), .000 (2), -.063 (5), -.080 (10), -.046 (15); 4,000 +.039 (1), -.031 (5), -.049 (10); 8,000
   -.007 (1), -.010 (5), -.015 (10); end -.014 [-.022, -.006] (1), -.004 (2), +.012 (5), +.013 [+.005, +.020] (10),
   -.005 (15). w = 1 is resolved above the observer at 58 checkpoints (20-5,400) and below at every checkpoint from
   10,400; each w >= 5 is resolved below at 55-98 checkpoints, the first at 10-340 and the last at 7,600-37,600 (lowest
   -.052 (w = 15) to -.086 (w = 9), at 1,600-2,240). At the end the shape is slightly reversed: w = 1 below the
   observer, w = 4-10 above it (+.011 to +.016), all resolved.
 5 What each step of own history is worth (random cell, per band): the recent steps' worth rises and falls; from about
   300 to 6k the recent past is worth more and the older past less than to Bayes, and by the end it is slightly the
   reverse. 2nd step / steps 3-6 / 7-9 / 10-15: 120 .0099 / .0097 / .0045 / .0015 (all below Bayes's, all above the
   observer's); 320 .0469 [.0389, .0554] / .0189 / .0019 / .0008; 560 .0819 / .0191 / .0094 / .0006; 880 .1100 [.0992,
   .1196] / .0332 / .0187 / .0057; 1,200 .1156 / .0433 / .0220 / .0085; 1,680 .1315 [.1221, .1412] / .0428 [.0396,
   .0459] / .0267 / .0114 [.0101, .0128]; 2,400 .1177 / .0394 / .0311 / .0142; 4,000 .0748 / .0344 / .0320 / .0187;
   6,000 .0435 / .0319 / .0309 / .0203; 8,000 .0274 / .0288 / .0311 / .0226; 11,200 .0188 / .0283 / .0305 / .0229; end
   .0187 [.0115, .0257] / .0241 [.0225, .0257] / .0307 / .0247 [.0232, .0261].
   Against Bayes's, paired: the 2nd step above at 51 checkpoints (280-6,400; largest .1317 at 1,520), below for good
   from 7,216 [6,728-20,296] (resolved below at every checkpoint from 10,400); steps 3-6 above at 33 checkpoints
   (880-5,800), below for good from 10,778 [6,025-19,563] (resolved below at 16,000-16,800 and from 20,000); steps 7-9
   resolved below at 35 checkpoints up to 1,920 and above only at 2,080; steps 10-15 resolved below at 66 checkpoints
   up to 6,600 (all but 6,200), at or above for good from 7,765 [6,712-20,161] (99% of resamples), resolved above at
   12,000, 13,600 and every checkpoint from 16,800. So the surplus of the recent steps and the deficit of the old ones
   fade over the same period (about 6-11k), not one after the other. By the end the profile is slightly reversed: the
   2nd step -.0106 [-.0171, -.0044] and steps 3-6 -.0049 [-.0072, -.0029] below Bayes's per step, steps 10-15 +.0023
   [+.0008, +.0038] above (7-9 +.0002 [-.0024, +.0026]). Against the observer at the same accuracy (paired): the 2nd
   step +.0436 (320), +.0953 (880), +.1079 (1,680), +.0485 (4,000), -.0097 [-.0160, -.0035] (end), resolved above at
   20-6,400 (and 7,000) and below from 11,200; steps 3-6 resolved below from 22,400 (and at 16,800, 20,800); steps
   10-15 -.0034 (560), -.0041 (880), -.0063 (1,680), -.0054 (2,400), -.0022 (4,000), +.0027 [+.0013, +.0041] (end),
   resolved below at about 0.3-4k (340-4,200; also 200, 260, 4,600, 5,200) and above from 12,000 (also 80-120,
   9,600-10,400). So the recency weighting from about 0.3 to 4k is not a lower accuracy, and the end reversal holds
   against the observer too.
 6 Start state: a restart costs more than it costs Bayes from before passing naive, over up to 9 windows at about 2k,
   receding to 4. The loss at w = 1 / 2 / 5 / 10: 320 .107 / .059 / .003 / -.001; 880 .289 / .229 / .091 / .008; 1,680
   .429 / .344 / .150 / .024; 4,000 .450 / .339 / .129 / .016; end .298 / .174 / .035 / .009 (Bayes .163 / .074 / .031
   / .013). The excess over Bayes stays above 0 from 552 [484-603] (w = 1), 417 [381-493] (2), 436 [346-493] (3), 428
   [385-567] (4), all before passing naive (835 [798-876]); w = 5 from 570 (82% of resamples); w >= 6 never for good.
   Its peak: +.292 [.273, .311] (w = 1, at 3,200), +.288 (2), +.224 (3), +.179 (4), +.127 [.117, .141] (5), +.092 (6),
   +.063 (7), +.037 (8), +.025 (9), +.011 [.007, .017] (10), at 1,760-3,200; not resolved for w >= 11. It reaches half
   its peak at 826-1,218 for every w <= 10 (w = 2-6 and 9 with passing naive; 1, 7, 8, 10 after it: 920, 1,059, 1,155,
   1,218). Falls at an absolute level (stays below .02 after the peak): 3,860 [2,702-4,505] (w = 8), 6,383 (7), 8,620
   (6), 19,343 [13,888-25,938] (5), never for w <= 4 (w = 4: 10% of resamples); below .05: 2,649 (7), 4,906 (6), 7,989
   (5), 18,551 (4), 38,376 (3, 80%), never for w <= 2; longer windows before shorter in 8 of 28 pairs at .02, never the
   other way. The span (largest w with excess > .02; every checkpoint): 0 up to 480, 3-5 at 560-720, 6 at 800-880, 7
   at 960-1,200, 8-9 at 1,280-3,200 (9 at 1,440-1,520, 1,680-1,840 and 2,000; 9 [8-9] at 1,680 and 2,000), 7-8 at
   3,400-5,800, 6-7 at 6,000-8,000, 5 at 8,800-19,200 (4 at 16,800), 4 from 20,000 (4 [3-4] at the end). (Half its
   own peak, a level that moves with the peak, gives the same order: 2,819 (w = 10) ... 6,698 [5,899-7,831] (5), 10,626
   (4), 19,613 (3), 25,732 (2); 31 of 45 pairs. At w = 1 the excess sits at about half its peak over the last 10
   checkpoints (32,800-39,996: .134-.158, .134 [.121, .147] at the end; half-peak .146): not resolved.) End excess
   +.134 (1), +.100 (2), +.045 (3), +.025 [.019, .032] (4), +.004 [-.003, +.012] (5), <= 0 beyond.
   Against the observer of the same accuracy the restart costs more at every checkpoint from 20 at w = 1-2 (113 of 114
   resolved): +.056 (w = 1, 120), +.092 (320), +.164 (640), +.218 (880), +.307 (1,680), +.300 (4,000), largest +.318 at
   3,400, +.138 [.125, .151] at the end (span, every checkpoint: 1-3 at 20-400, 4-5 at 480-720, 6 at 800-880, 7 at
   960-1,200, 8-9 at 1,280-2,800 (9 at 1,440-1,520 and 1,760-1,840), 7 at 3,000-5,800 (6 at 5,400), 6 at 6,000-8,000,
   5 at 8,800-15,200, 4 from 16,000 (5 at 18,400)). So the early 'less than Bayes' (320: .107 vs .163) is the
   network's lower accuracy, not a weaker dependence.
 7 Same cell's other trajectory (null): the network never loses half of what Bayes loses, at any w (at most 38% of
   resamples cross); it gains up to .044 at w = 1 (1,520) and loses at most .013, where Bayes loses .021 (w = 1) to
   .005.
 8 Far errors added follow the accuracy lost (r = .974 random cell, .972 start state, over checkpoints x windows):
   random cell .368 (w = 1, 560), .645 (1,680), .592 (end) vs Bayes .585; at w = 15 .004 / .014 / .058 vs .062; start
   .296 / .569 / .271 vs Bayes .174; at w = 5 .087 / .162 / .025 vs .016.
 9 ESN: its dependence stays within its last 3-4 steps. Intact accuracy .100 -> .431. Random cell: L at most .248 (w =
   1), .124 (2), .047 (3), .018 (4); end .247 / .119 / .042 / .014; L(1) grows through training (.044 at 320, .107 at
   1,360, .200 at 6,200, .247 at the end); span (L > .02; all 30 checkpoints) 2 from 240 (3 at 400 and 1,360), 3 from
   2,000. Start state: span 1 at 240-1,680 (2 at 1,040), 2 at 2,000-3,800, 3 at 4,600-13,600, 4 at 16,800, then 2 or 3
   (2 at 23,200, 29,600, 36,000 and the end). Same cell: L <= .002 (gains at the end, -.018 at w = 1). It reaches half
   of Bayes's random-cell loss only at w = 1, at 11,405 [7,814-38,692] (97% of resamples), at intact accuracy .404
   [.393, .419] (the trained runs' w = 1 onsets: .350-.425).
10 Replication on the width / loss runs (random cell, engine donors; Bayes's loss is run-independent).
   The order by w holds in every run with more than one onset: power-1 loss (end accuracy .760; passes naive 1,256
   [1,207-1,356]): 1,157 [1,037-1,214] (w = 1), 1,378 (2), 1,722 (5), 4,432 [4,166-5,030] (10), 6,904 [6,318-7,413]
   (15), 92 of 105 pairs, none reversed, x1.152 [1.144, 1.160] per step; N = 200 (end .716; passes naive 2,070
   [1,966-2,424]): 1,142 [976-1,226] (1), 2,043 (2), 2,773 (3), 3,395 (4), 4,281 [3,877-7,437] (5), 7,499 (6), 14,691
   [10,082-23,563] (7), 35,200 [23,633-never] (8, 94%), never for w >= 9 (47% of resamples at 9); 31 of 105 pairs, none
   reversed, x1.555 [1.495, 1.635]; N = 100 (end .634; passes naive 6,271 [5,444-7,662]): 2,433 [2,163-3,025] (1),
   10,287 (2), 12,471 (3), 25,817 [17,001-36,241] (4), never for w >= 5 (17% at 5); 8 of 105, none reversed, x2.071
   [1.835, 2.310]; N = 50 (end .507; passes naive 21,072 [14,871-30,271]): only w = 1, 8,294 [6,060-13,882].
   The onsets do not scale with each run's own passing-naive time. Onset / passing naive: N = 1000 .76 (w = 1), 1.01
   (2), 1.48 (5), 3.16 (10), 5.28 [4.65, 6.72] (15); power 1 .92, 1.10, 1.37, 3.53, 5.49 (paired ratio to N = 1000's
   1.22 [1.14, 1.26] at w = 1, 1.08 [1.03, 1.12] at 2, .84-1.17 at 3-15, intervals including 1 except w = 9-10: 1.10
   [1.01, 1.19], 1.12 [1.01, 1.24]); N = 200 .55 (1), .99 (2), 1.34 (3), 1.64 (4), 2.07 (5), 3.62 (6), 7.10 (7), 17.0
   (8) (ratio .73 [.63, .76] at 1, 1.21-1.40 at 3-5, 3.45 [2.38, 5.40] at 7, 7.21 [4.80, -] at 8); N = 100 .39, 1.64,
   1.99, 4.12 (ratio .51 [.44, .63], 1.62, 1.79, 3.26 [2.44, 4.32]); N = 50 .39 [.28, .61].
   The windows each run reaches come at about the same accuracy in every run instead. Intact accuracy at the onsets,
   minus N = 1000's (paired): power 1 .425 (w = 1, +.057 [+.047, +.064]), .494 (2, +.023), .525 (3, +.014), then
   within -.024 and +.011 for w = 4-15 (every interval includes 0); N = 200 .350 (-.019 [-.028, -.013]), .457, .512,
   .537, .570, .637, .681 (w = 2-7, -.021 to +.017, every interval includes 0), .711 (8, +.031 [+.012, +.038]); N = 100
   .352 (-.017 [-.023, -.009]), .519 (+.047 [+.025, +.061]), .538 (+.027), .595 (+.054 [+.034, +.074]); N = 50 .386
   (+.017 [+.001, +.043]). But the narrower runs do not stop where their accuracy stops. At their longest windows they
   need more accuracy than N = 1000 (above: N = 200 w = 8, N = 100 w = 2-4, N = 50 w = 1, all resolved), and each ends
   at an accuracy at which N = 1000 had already passed one or two windows more. N = 50 ends at .507 [.486, .526],
   +.035 [+.018, +.053] above N = 1000's accuracy at its w = 2 onset, and never reaches w = 2; N = 100 ends at .634
   [.612, .655], +.043 [+.022, +.082] above N = 1000's at w = 5, which it reaches in 17% of resamples; N = 200 ends at
   .716 [.700, .731], +.019 [+.007, +.031] and +.009 [+.0001, +.018] above N = 1000's at w = 9 and 10, which it
   reaches in 47% and 1% of resamples (one window further up the end accuracy is with N = 1000's onset accuracy in
   every run: w = 3 / 6 / 11). So at the same accuracy the narrower runs lose less to the cut at those windows (their
   L(w) ends below half of Bayes's, where N = 1000's had stayed above it from a lower accuracy on).

ANSWER. Yes, the relative benefit of different memory lengths changes through training, but only the onsets move one
  way. With another cell's state in place of the content older than w steps, the Trained RNN's answers come to depend
  on that content for ever longer windows: the loss reaches half of Bayes's first at w = 1 and later for longer
  windows, about 15% later per extra step (x1.148 [x1.142, x1.166]; x1.139 and x1.164 under the two other donor
  schemes); with the engine's donors every pair of windows 4 or more steps apart is ordered (88 of 105 pairs), and no
  pair is reversed in any scheme. The dates themselves depend on the donor scheme (w = 15: 3,865 / 4,408 / 6,924), and
  the w = 1 date follows the intact accuracy, so which window coincides with passing naive is not a result. The order
  is not an accuracy effect: a perfect-memory observer of the network's accuracy reaches the same criterion at the same
  accuracy (.50-.53) for every w, i.e. at about 900-1,000 training episodes for all 15 windows. The order replicates in
  every width / loss run that reaches more than one window (x1.152 with the power-1 loss, x1.56 at N = 200, x2.07 at
  N = 100). The windows a run reaches come at about the same intact accuracy (from about .35-.43 at w = 1 to .73 at
  w = 15; within -.024 and +.057 of N = 1000's), not at a fixed multiple of the run's passing-naive time (N = 200's
  w = 7 at 7.1x its passing naive, N = 1000's at 2.05x); but the narrower runs need more accuracy at their longest
  windows and end one or two windows short of what N = 1000 had passed at their final accuracy, so at the same
  accuracy they lose less to the cut there. What each step of own history is worth rises and falls for the recent
  steps: from about 300 to 6k the 2nd step back is worth up to .13 of accuracy (Bayes .03; largest at about 1.5k) and
  steps 3-6 up to .045, while steps 10-15 are worth less than to Bayes until about 8k (against the observer: below at
  about 0.3-4k); the recent-step surplus and the old-step deficit fade over the same period (about 6-11k), and by the
  end the profile is slightly reversed (2nd step .011 and steps 3-6 .005 below Bayes's per step, steps 10-15 .002
  above). A restart from the start state costs more than it costs Bayes from 417-552 at w <= 4 (before passing naive;
  against the observer from 20), over up to 9 windows at about 1.4-2k, receding to 4. The ESN's dependence stays within
  its last 3-4 steps (span 2 -> 3).

NULLS AND FAILED HYPOTHESES
  (a) One memory span that switches on at passing naive: no; the onsets spread over 631-4,408 (675-3,865; 806-6,924
      in the other schemes) and are ordered by w.
  (b) Which window coincides with passing naive or the half-way point: not robust (engine and the other fixed seed:
      w = 1 before passing naive, w = 2 with it; fresh donors: w = 1 with it, w >= 2 after); retracted from the
      headline.
  (c) The order as an accuracy effect: rejected (result 4: the observer's onsets are not ordered, x1.005 [0.997,
      1.016] per step).
  (d) The onsets scaling with each run's own passing-naive time: holds only between the two N = 1000 runs (ratios
      .84-1.22); the narrower runs stretch (N = 200 x1.56 per step, N = 100 x2.07).
  (e) Old steps reaching Bayes's per-step weight at about 1,680 (the previous version): an averaging artifact,
      retracted. At 1,680 steps 3-6 are above Bayes's (.0428 vs .0290) and 10-15 at about half of it (.0114 vs .0224);
      10-15 reach Bayes's only from 7,765 [6,712-20,161].
  (f) 'From about passing naive a restart costs more than Bayes' and 'early it costs less' (previous version):
      corrected; the excess starts at 417-552 (w <= 4), before passing naive; against the observer the restart costs
      more from 20 on.
  (g) Same cell: no loss at any w.
  (h) Reaching Bayes's full loss (level 1) with a random cell: never for w <= 3 (end .432 / .414 / .396 vs .456 /
      .426 / .400); 19,696-32,857 with 80-96% of resamples for w = 4-10; never for w >= 11 (4-39%); the end profile is
      within .03 of Bayes's at every w.
  (i) Start-state onsets at half of Bayes's loss are not used: Bayes's restart loss is small (.006-.013 for w >= 9)
      and the restart confounds memory content with the start regime (td_tt_memcuts).
  (j) The w = 1 restart excess's fall: not resolved (at about half its peak over the last 10 checkpoints, 32,800 to
      the end).
  (k) ESN: no long dependence (span 2 -> 3 steps; half of Bayes's loss only at w = 1).
  (l) 'The narrower runs stop where their accuracy stops' (round 3): wrong; each ends at an accuracy at which N = 1000
      had passed one or two more windows, which it does not reach (result 10).
  (m) 'In one direction' and 'the end profile is Bayes's flat one' (round 3): only the onsets move one way; the recent
      steps' worth rises and falls, and the end profile is slightly reversed against Bayes's (result 5).

CONFOUNDS AND CAVEATS
  The cut keeps the network's own increments up to t = w and the window's own inputs: only the state's content older
  than w is tested. A random cell's state carries wrong evidence, so L(w) mixes losing the right content with acting on
  wrong content; the loss is a ceiling, read against Bayes under the identical substitution and against the observer.
  The observer matches the network's intact accuracy only; it is less accurate by seeing fewer observations, evenly
  over time, which is one of many ways to be less accurate. The start state confounds missing content with the start
  regime. Each run is one seed of its configuration; intervals resample environments, not seeds; the replication runs
  share the seed, the environment and the episode sequence. fp32 replays of networks trained under fp16 autocast (as
  the engine); answers from float64 cumulative sums. The ESN is read at 30 checkpoints.

CHECKS (checks.json; checks() 447 s on GPU 1 beside another job, then tie_checks())
  The route vs td_tt_intervene.forward (all 10,000 probes; w = 1, 2, 5, 10, 15 x three sources) at training 880 /
    1,680 / 39,996: max |d log q| 3.7e-4 / 1.8e-4 / 8.1e-5 (the engine's intact run differs from the route's by 2.6e-4 /
    1.8e-4 / 2.3e-5: chunk shapes), answers at T differ 0 / 0 / 1 of 300,000, goal answers at every step 0 / 3 / 2 of
    4,500,000, against cut effects of up to 168 / 138 / 204 in log q. The other fixed seed's donors through the route vs
    the engine with the same explicit donors (1,680; w = 1, 5, 15): max |d log q| 3.8e-5, 0 of 60,000 answers differ.
  A probe as its own donor equals the intact network (2,000 probes, w = 1..15, both donor sources): max |d log q|
    2.3e-5, no answer differs. Float64 brute force of the cut increments (4 probes x steps 16, 19, 30 x w = 1, 7, 15 x
    the three sources and the fresh scheme's per-window donors; 144 increments): max |d| 2.9e-6 (fresh 2.8e-6; intact
    1.5e-6), against cut effects on the increment of up to 4.1.
  The stored start-state truncation (td_analysis trunc_right: w = 1, 2, 5 and intact, both variables at T) at every
    harvested checkpoint: Trained RNN 5 of 9,120,000 answers differ (114 checkpoints), ESN 2 of 2,400,000, power 1 2,
    N = 200 2, N = 100 0, N = 50 0 of 9,120,000 each; all 11 are float near-ties (the route's top-two margin <= 5.5e-6
    nats; tie_checks). td_tt_memcuts' stored answers (intact; w = 2, 5, 10 x three
    sources; 46 Trained RNN and 5 ESN shared checkpoints): 2 of 9,200,000 and 1 of 1,000,000 differ (accuracy within
    1e-4). The adversarial check's independent route for the two other schemes (its 24 checkpoints): 8 of 14,400,000
    answers differ, every one a float near-tie (the route's top-two margin <= 9.2e-6 nats); its Bayes under them: 0
    differ.
  Bayes: intact vs probes.npz log_jb 7.6e-6 (its float32 storage); the count route vs the cumulative-likelihood route (a
    copy of td_tt_memcuts.bayes_logq, w = 1, 2, 5, 10, 15 x three sources) 8.1e-13 and vs that module itself (w = 2, 5,
    10) 7.4e-13; under the other schemes (incl. per-window donors) vs the cumulative-likelihood route 7.8e-13; windowed
    Bayes = probes.npz right_window1 / 2 / 5 (0 of 20,000 differ each); td_tt_memcuts' stored Bayes answers: 0 differ;
    the caches = recomputed (0). The thinned observer at p = 1 = Bayes (0 of 310,000 answers and 0 per-environment
    counts differ), and at seed 7 it equals the adversarial check's own thinned observer at its 11 values of p (max
    |d accuracy| 0).
  Every other-scheme file's intact answers equal the main cache's (0 differ at all 114 checkpoints). The thinned
    observer's accuracy is monotone in p in every resample (so the accuracy matching is one-to-one).
  pyflakes clean. numbers() reruns from the caches. The figure was inspected whole and in native-resolution crops.

COST: the route 6-18 s per N = 1000 checkpoint on GPU 1 (0.4-3.4 s at N = 50-200), 70-112 s on the shared GPU 0 (the
  first 19); 75 GPU-minutes for the 600 cut files (Trained RNN 47, ESN 5, power 1 16, N = 200 / 100 / 50 7), 19 for the
  other donor schemes (114 checkpoints, 9-11 s each); checks 447 s. CPU: Bayes caches 190 s + 98 s (schemes), the
  thinned observer 1,141 s (17 values of p x 2 seeds); numbers ~3 min; figure ~1 min. GPU 1: John freed it 11-02
  (16:45, then "it's all yours, both GPUs").

Run from the repository root (one GPU process, <= 2 GB; 2 CPU threads):
    CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
        /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_memlen.py harvest
    (launches the worker on GPU 1: the missing cut files of every run, then the other donor schemes); then 'bayes'
    (both Bayes caches), 'thinned' (the observer, CPU), 'checks' (with the near-tie check; 'ties' alone), 'numbers',
    'figure'; 'one <run> <raw episode>' redoes one checkpoint; the worker logs to the session scratchpad
    round3/memlen"""
import os
import sys
import json
import time
import subprocess
import warnings
import numpy as np
import torch
import torch.nn.functional as Fn
from scipy.special import logsumexp

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_intervene as ti                                                                  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False             # fp32 matmuls, as the engine
torch.backends.cudnn.allow_tf32 = False

TD = ti.TD
OUT = os.path.join(TD, "through_training", "memlen")
TMP = ("/tmp/claude-1002/-home-johns-Desktop-CognitiveGridworld/7af6b091-54ff-45a0-b4dd-ef0a3bc9c778/scratchpad/round3/"
       "memlen")
PY = os.environ.get("CGW_PY", "/home/johns/anaconda3/envs/eureka/bin/python")
T, R, NV, NO = 30, 10, 2, 5
NE, PER_ENV = gap.N_ENV, gap.PER_ENV
NB = gap.N_BOOT
WMAX = 15
WS = tuple(range(1, WMAX + 1))
SOURCES = ti.SOURCES                                      # 'start', 'same_cell', 'random'
CONDS = ("intact",) + tuple(f"w{w}_{s}" for s in SOURCES for w in WS)          # source-major, then w
NC = len(CONDS)
CHUNK = 5                                                 # environments (1,000 probes) per GPU chunk
NEED_MB = 4000                                            # free memory on the visible GPU before a checkpoint starts
REP = ("trained_p1", "trained_N200", "trained_N100", "trained_N50")     # the width / loss-power replication runs
SCHEMES = ("fixed2", "fresh")                             # the other random-cell donor schemes (the verifier's)
COL = {"trained": "#1f77b4", "echo": "#ff7f0e"}
NAME = {"trained": "Trained RNN", "echo": "ESN", "trained_p1": "power-1 loss", "trained_N200": "N = 200",
        "trained_N100": "N = 100", "trained_N50": "N = 50"}
_C = {}


def ci(src, w):
    """index of the condition (source, window) in CONDS"""
    return 1 + SOURCES.index(src) * WMAX + int(w) - 1


# ═══════════════════════════════════════════════════════════════════
# Checkpoints
# ═══════════════════════════════════════════════════════════════════

def selection(run):
    """raw episodes analysed: every stored checkpoint (114) of the Trained RNN and of the replication runs; the ESN's
    every 4th and its last (30)"""
    eps = [int(e) for e in ti.checkpoints(run)]
    if run == "echo":
        keep = set(range(0, len(eps), 4)) | {len(eps) - 1}
        return [eps[i] for i in sorted(keep)]
    return eps


def ck_path(run, ep):
    return os.path.join(OUT, run, f"ck_{int(ep):05d}.npz")


def scheme_path(ep):
    return os.path.join(OUT, "schemes", f"ck_{int(ep):05d}.npz")


def _spread(lst):
    """lst (ordered by x) in coarse-to-fine order: the two ends, then repeated midpoints, so that any prefix covers the
    range about evenly"""
    lst = list(lst)
    if len(lst) <= 2:
        return lst
    out, segs = [lst[0], lst[-1]], [(0, len(lst) - 1)]
    while segs:
        nxt = []
        for a, b in segs:
            if b - a > 1:
                m = (a + b) // 2
                out.append(lst[m])
                nxt += [(a, m), (m, b)]
        segs = nxt
    return out


def jobs(kind="cuts"):
    """the missing work, coarse to fine per run: kind 'cuts' = [(run, raw episode)] of every run's selection without a
    cache file (Trained RNN, ESN, the replication runs); 'schemes' = the Trained RNN's checkpoints without the other
    donor schemes' file"""
    if kind == "schemes":
        return [("trained", e) for e in _spread(selection("trained")) if not os.path.exists(scheme_path(e))]
    return [(run, e) for run in ("trained", "echo") + REP for e in _spread(selection(run))
            if not os.path.exists(ck_path(run, e))]


# ═══════════════════════════════════════════════════════════════════
# Probes
# ═══════════════════════════════════════════════════════════════════

def probes():
    """ti.probes() plus the likelihood tables L (50, 5, R, R) = p(x_o = 1 | r0, r1) and the per-bit cumulative counts
    K (n, T + 1, 5) of the probes' own observations (K[:, t] = counts over steps 1..t)"""
    if "P" not in _C:
        P = dict(ti.probes())
        P["L"] = np.load(os.path.join(ti.AN, "probes.npz"))["L"].astype(np.float64)
        P["K"] = np.concatenate([np.zeros((len(P["obs"]), 1, NO), np.int64),
                                 np.cumsum(P["obs"].astype(np.int64), 1)], 1)
        n = len(P["env"])
        assert np.array_equal(P["env"], np.arange(n) // PER_ENV), "probes are environment-major"
        _C["P"] = P
    return _C["P"]


def donors():
    """the engine's donors on the probes (td_tt_intervene hook a): same_cell = row ^ 1, random = seed 20261201"""
    return {"same_cell": ti._donors_cached("same_cell"), "random": ti._donors_cached("random")}


def scheme_donors():
    """the other random-cell donor schemes (the verifier's, check_memlen/c5_fresh.py, rebuilt exactly): 'fixed2' = the
    engine's scheme (one fixed probe of the same environment with another cell per probe) under seed 20261299, (n,);
    'fresh' = a fresh random probe of the environment with another cell at every window start a (seed 777), (n, T) with
    column a the donor for the window starting at a (column 0 unused)"""
    if "schemes" not in _C:
        P = probes()
        env, cell = P["env"], P["cell"]
        n = len(env)
        rng = np.random.default_rng(777)
        fr = np.full((n, T), -1, np.int64)
        for e in np.unique(env):
            ix = np.flatnonzero(env == e)
            for i in ix:
                cand = ix[cell[ix] != cell[i]]
                fr[i] = cand[rng.integers(len(cand), size=T)]
        d2 = ti.donors(P, "random", 20261299)
        for d in (d2[:, None], fr):
            assert (d >= 0).all() and (env[d] == env[:, None]).all() and (cell[d] != cell[:, None]).all()
        _C["schemes"] = {"fixed2": d2, "fresh": fr}
    return _C["schemes"]


# ═══════════════════════════════════════════════════════════════════
# Bayes under the same cuts (exact, float64, from the observation counts)
# ═══════════════════════════════════════════════════════════════════

def _bside():
    """per probe: Aenv (n, 5, R, R) = log L - log(1 - L), Benv (n, R, R) = Sigma_i log(1 - L_i), in the probe's own
    environment: the joint log-likelihood of any history with per-bit counts k over m observations is k . A + m B
    (observations are i.i.d. Bernoulli given the cell)"""
    if "bside" not in _C:
        P = probes()
        L = P["L"][P["env"]]
        _C["bside"] = (np.log(L) - np.log1p(-L), np.log1p(-L).sum(1))
    return _C["bside"]


def _joint(k, m):
    """joint log weights of histories with per-bit counts k and lengths m in each probe's environment: k (n, 5) or
    (n, S, 5), m broadcastable to k.shape[:-1] -> (n, R, R) or (n, S, R, R)"""
    A, B = _bside()
    k = np.asarray(k, np.float64)
    m = np.broadcast_to(np.asarray(m, np.float64), k.shape[:-1])
    if k.ndim == 2:
        return np.einsum("ni,nijk->njk", k, A) + m[:, None, None] * B
    return np.einsum("nsi,nijk->nsjk", k, A) + m[..., None, None] * B[:, None]


def _marg(X):
    """log marginals (unnormalised) of joint log weights X (..., R, R): (..., 2, R) = [variable 0, variable 1]"""
    return np.stack([logsumexp(X, -1), logsumexp(X, -2)], -2)


def bayes_logq(w=None, src="start", donor=None):
    """Bayes's marginal log beliefs (n, T, 2, R) float64, intact (w None) or under the network's cut (w, source)
    (td_tt_memcuts' definition): for t > w the step-t increment is Bayes's marginal change from adding observation t to
    the history (the donor's observations 1..t-w, own t-w+1..t-1; source 'start': no donor observations = windowed
    Bayes); t <= w Bayes's own increment; the beliefs are the cumulative sum, normalised. Computed from the per-bit
    counts (k = K_donor(a) + K(t) - K(a), m = a + t - a, a = t - w). donor: an explicit donor array (overrides src),
    (n,) one donor per probe or (n, T) one per probe and window start a (column a)"""
    P = probes()
    K = P["K"]
    n = len(K)
    tt = np.arange(T + 1)
    Mi = _marg(_joint(K, np.broadcast_to(tt, (n, T + 1))))                         # (n, T + 1, 2, R) intact
    if w is None:
        m = Mi[:, 1:]
        return m - logsumexp(m, -1, keepdims=True)
    d = np.asarray(donor) if donor is not None else (None if src == "start" else donors()[src])
    inc = np.diff(Mi, axis=1)                                                      # (n, T, 2, R) Bayes's own
    t = np.arange(w + 1, T + 1)                                                    # steps with a cut increment
    a = t - w
    if d is None:
        kd, md = np.zeros((n, len(t), NO)), np.zeros((n, len(t)))
    else:
        kd = (K[d][:, a] if d.ndim == 1 else K[d[:, a], a[None]]).astype(np.float64)
        md = np.broadcast_to(a.astype(np.float64), (n, len(t)))
    k1 = kd + K[:, t] - K[:, a]
    k0 = kd + K[:, t - 1] - K[:, a]
    inc[:, t - 1] = _marg(_joint(k1, md + (t - a))) - _marg(_joint(k0, md + (t - 1 - a)))
    cum = np.cumsum(inc, 1)
    return cum - logsumexp(cum, -1, keepdims=True)


def _prims(lq, right=None):
    """the stored primitives of one belief array lq (n, T, 2, R) (any normalisation): ans (n, 2) int8 answers at T;
    accg (NE, T) goal answers right per environment and step; lqg (NE,) the environments' sums of log q of the true
    goal value at T (normalised). right (n, T, 2) overrides argmax == truth"""
    P = probes()
    lq = np.asarray(lq, np.float64)
    lq = lq - logsumexp(lq, -1, keepdims=True)
    ar, g, r = np.arange(len(lq)), P["goal"], P["r"]
    a = lq.argmax(-1)
    rgt = (a == r[:, None, :]) if right is None else np.asarray(right, bool)
    return dict(ans=a[:, -1].astype(np.int8),
                accg=rgt[ar, :, g].reshape(NE, PER_ENV, T).sum(1).astype(np.int16),
                lqg=lq[ar, -1, g, r[ar, g]].reshape(NE, PER_ENV).sum(1))


def bayes_cache(force=False):
    """-> OUT/bayes.npz: Bayes intact and under the 45 cuts (CONDS order) as primitives (ans (NC, n, 2), accg (NC, NE,
    T), lqg (NC, NE)), naive's (naive_*: memoryless, no cut changes it)"""
    path = os.path.join(OUT, "bayes.npz")
    if os.path.exists(path) and not force:
        return dict(np.load(path))
    t0 = time.time()
    res = [_prims(bayes_logq())]
    for s in SOURCES:
        for w in WS:
            res.append(_prims(bayes_logq(w, s)))
        print(f"  Bayes {s}: w = 1..{WMAX} done ({time.time() - t0:.0f} s)", flush=True)
    out = {k: np.stack([r_[k] for r_ in res]) for k in ("ans", "accg", "lqg")}
    nv = _prims(probes()["log_nb"])
    out.update({f"naive_{k}": v for k, v in nv.items()}, conds=np.array(CONDS))
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(path, **out)
    print(f"saved {path} ({time.time() - t0:.0f} s)", flush=True)
    return out


def bayes_schemes(force=False):
    """-> OUT/bayes_schemes.npz: Bayes under the other donor schemes' random-cell cuts (identical substitution), ans
    (len(SCHEMES) * WMAX, n, 2) int8 in scheme-major order, plus scheme names"""
    path = os.path.join(OUT, "bayes_schemes.npz")
    if os.path.exists(path) and not force:
        return dict(np.load(path))
    t0 = time.time()
    D = scheme_donors()
    ans = np.stack([bayes_logq(w, donor=D[s])[:, -1].argmax(-1).astype(np.int8) for s in SCHEMES for w in WS])
    np.savez_compressed(path, ans=ans, schemes=np.array(SCHEMES))
    print(f"saved {path} ({time.time() - t0:.0f} s)", flush=True)
    return dict(np.load(path))


# ═══════════════════════════════════════════════════════════════════
# The accuracy-matched perfect-memory observer (thinned Bayes)
# ═══════════════════════════════════════════════════════════════════

P_GRID = (0., .005, .01, .015, .02, .03, .04, .05, .07, .1, .13, .18, .25, .35, .5, .7, 1.)
THIN_SEEDS = (7, 8)
THIN_CONDS = ("intact",) + tuple(f"w{w}_{s}" for s in ("random", "start") for w in WS)


def thinned_right(p, seed):
    """goal answers right at T (n,) per condition of THIN_CONDS of Bayes that sees each observation with probability
    p: mask = u < p with u (n, T) uniform from default_rng(seed) (nested in p; a probe's mask is the same when it is the
    receiver and when it is a donor), exact float64 from the masked counts: intact, and under the network's random-cell
    (the engine's donors) and start-state cuts at w = 1..15 with memlen's definition. Route: with
    J(t) = K(t) . A + m(t) B (the joint log weight of the probe's masked observations 1..t) and
    M_w(t) = marginals of J_donor(t - w) + J(t) - J(t - w), the cut increment at t > w is M_w(t) - M_(w-1)(t - 1), so the
    belief at T under the cut to w is M_0(w) - M_0(0) (own increments, start: uniform) + Sigma_(t = w+1..T) M_w(t) -
    Sigma_(t = w..T-1) M_(w-1)(t); the start state has J_donor = 0"""
    P = probes()
    A, B = _bside()
    obs = P["obs"].astype(np.float64)
    n = len(obs)
    u = np.random.default_rng(seed).random((n, T))
    m = (u < p).astype(np.float64) if p < 1 else np.ones((n, T))
    K = np.concatenate([np.zeros((n, 1, NO)), np.cumsum(obs * m[..., None], 1)], 1)       # (n, T + 1, 5)
    Mc = np.concatenate([np.zeros((n, 1)), np.cumsum(m, 1)], 1)                           # (n, T + 1)
    J = np.einsum("nti,nijk->ntjk", K, A) + Mc[..., None, None] * B[:, None]             # (n, T + 1, R, R)
    g, r = P["goal"], P["r"]
    ar = np.arange(n)
    out = {"intact": _marg(J[:, T]).argmax(-1)[ar, g] == r[ar, g]}
    for src in ("random", "start"):
        Jd = J[donors()["random"]] if src == "random" else None                         # the donor's J
        S_full, S_trunc = {}, {}
        for w_ in range(0, WMAX + 1):
            a = np.arange(1, T - w_ + 1)                                                  # window starts a = t - w_
            X = J[:, a + w_] - J[:, a] + (Jd[:, a] if Jd is not None else 0.)              # (n, A, R, R)
            Mw = _marg(X)                                                                 # (n, A, 2, R)
            S_full[w_] = Mw.sum(1)                                                        # t = w_+1..T
            S_trunc[w_] = Mw[:, :-1].sum(1)                                               # t = w_+1..T-1
        M00 = _marg(J[:, 0])
        for w in WS:
            bel = _marg(J[:, w]) - M00 + S_full[w] - S_trunc[w - 1]
            out[f"w{w}_{src}"] = bel.argmax(-1)[ar, g] == r[ar, g]
    return np.stack([out[c] for c in THIN_CONDS])


def thinned_cache(force=False):
    """-> OUT/thinned.npz: right (len(P_GRID), len(THIN_SEEDS), len(THIN_CONDS), NE) int16 = per-environment counts of
    goal answers right at T of the thinned observer (thinned_right), p, seeds, conds"""
    path = os.path.join(OUT, "thinned.npz")
    if os.path.exists(path) and not force:
        return dict(np.load(path))
    t0 = time.time()
    out = np.zeros((len(P_GRID), len(THIN_SEEDS), len(THIN_CONDS), NE), np.int16)
    for i, p in enumerate(P_GRID):
        for j, sd in enumerate(THIN_SEEDS):
            out[i, j] = thinned_right(p, sd).reshape(len(THIN_CONDS), NE, PER_ENV).sum(-1)
        print(f"  thinned p = {p:g}: intact accuracy {out[i, :, 0].sum() / (len(THIN_SEEDS) * NE * PER_ENV):.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(path, right=out, p=np.array(P_GRID), seeds=np.array(THIN_SEEDS), conds=np.array(THIN_CONDS))
    print(f"saved {path} ({time.time() - t0:.0f} s)", flush=True)
    return dict(np.load(path))


# ═══════════════════════════════════════════════════════════════════
# The network: every cut from one rerun per start offset
# ═══════════════════════════════════════════════════════════════════

def _step(xin, h, c, Whh):
    """one LSTM step (td_analysis / td_tt_intervene._step without hooks): pre = xin + h W_hh^T (added in place:
    IEEE addition commutes, so bitwise the engine's), c = sig(f) c + sig(i) tanh(g), h = sig(o) tanh(c)"""
    pre = h @ Whh.T
    pre += xin
    i, f, g, o = pre.chunk(4, -1)
    c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
    return torch.sigmoid(o) * torch.tanh(c), c


@torch.no_grad()
def net_cuts(w, dev=None, envs=None, dons=None, chunk=CHUNK, keep=(), keep_inc=(), sources=None):
    """the network's beliefs intact and under every cut (source x w = 1..WMAX) on the probes (or the probes of the
    environments `envs`; donors live in the probe's environment, so whole environments are self-contained).
    Route (td_tt_intervene hook a, exactly): for every start offset a = 1..T-1 the cell is rerun from the source's state
    at a (start: (h0, c0); a donor: the donor's intact state after a steps) over the probe's own inputs
    a+1..min(a + WMAX, T); the read-out of that rerun after k steps is the cut increment at t = a + k for the window
    w = k. The cut belief with window w = the cumulative sum of the intact increments for t <= w and of the offset-(t - w)
    reruns' increments (after w steps) for t > w; log-softmax per variable in float64, answers = argmax (as the engine).
    Every rerun step feeds exactly one (w, t). dons: {'same_cell': (n,), 'random': (n,)} (default the engine's).
    sources: [(name, donor)] in place of the engine's three, donor None (the start state), (n,) or (n, T) (a donor per
    probe and window start a, column a). Returns dict(conds, rows, ans (NC, n, 2) int8, rg (NC, n, T) goal answers
    right, lqg (NC, n) log q of the true goal value at T, lq {cond: (n, T, 2, R) float32} for `keep`, inc {cond: (n, T,
    20) float32} for `keep_inc`, info)"""
    t0 = time.time()
    dev = ti._device(dev)
    P = probes()
    env_all, r_all, g_all = P["env"], P["r"], P["goal"]
    p = {k: v.to(dev, torch.float32) for k, v in w.items()}
    H = p["LSTM.weight_hh_l0"].shape[1]
    Whh, Wro, bro = p["LSTM.weight_hh_l0"], p["classifier_readout.weight"], p["classifier_readout.bias"]
    h0, c0 = p["STM"].reshape(1, H), p["LTM"].reshape(1, H)
    table = ti._table(p, P["Z"], dev)
    code = (P["obs"] * (1 << np.arange(NO))).sum(-1).astype(np.int64)
    dons = donors() if dons is None else dons
    if sources is None:
        sources = [(s, None if s == "start" else np.asarray(dons[s])) for s in SOURCES]
    sources = [(nm, None if d is None else np.asarray(d, np.int64)) for nm, d in sources]
    conds = ("intact",) + tuple(f"w{w_}_{nm}" for nm, _ in sources for w_ in WS)
    nc = len(conds)
    envs = np.arange(NE) if envs is None else np.unique(np.asarray(envs))
    rows_all = np.flatnonzero(np.isin(env_all, envs))
    n = len(rows_all)
    out = dict(conds=conds, rows=rows_all, ans=np.zeros((nc, n, NV), np.int8), rg=np.zeros((nc, n, T), bool),
               lqg=np.zeros((nc, n)), lq={k: np.zeros((n, T, NV, R), np.float32) for k in keep},
               inc={k: np.zeros((n, T, NV * R), np.float32) for k in keep_inc})
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    def ix(a_):
        return torch.as_tensor(np.asarray(a_, np.int64), device=dev)
    for c_ in range(0, len(envs), chunk):
        rows = np.flatnonzero(np.isin(env_all, envs[c_:c_ + chunk]))
        o = np.searchsorted(rows_all, rows)
        b = len(rows)
        loc = np.full(len(env_all), -1, np.int64)
        loc[rows] = np.arange(b)
        env_t, code_t = ix(env_all[rows]), ix(code[rows])
        ar = torch.arange(b, device=dev)
        gl = ix(g_all[rows])
        rg_true = ix(r_all[rows])[ar, gl]                                         # (b,) the true goal value
        h, c = h0.expand(b, H).contiguous(), c0.expand(b, H).contiguous()
        Hs, Cs = torch.empty(b, T + 1, H, device=dev), torch.empty(b, T + 1, H, device=dev)
        Hs[:, 0], Cs[:, 0] = h, c
        lg = torch.empty(b, T, NV * R, device=dev)
        for s in range(T):
            h, c = _step(table[env_t, code_t[:, s]], h, c, Whh)
            Hs[:, s + 1], Cs[:, s + 1] = h, c
            lg[:, s] = Fn.linear(h, Wro, bro)

        def finish(k_, L):
            lq = torch.log_softmax(L.double().cumsum(1).reshape(b, T, NV, R), -1)
            a_ = lq.argmax(-1)                                                    # (b, T, 2)
            out["ans"][k_, o] = a_[:, -1].cpu().numpy().astype(np.int8)
            out["rg"][k_, o] = (a_[ar, :, gl] == rg_true[:, None]).cpu().numpy()
            out["lqg"][k_, o] = lq[ar, -1, gl, rg_true].cpu().numpy()
            nm = conds[k_]
            if nm in out["lq"]:
                out["lq"][nm][o] = lq.float().cpu().numpy()
            if nm in out["inc"]:
                out["inc"][nm][o] = L.cpu().numpy()
        finish(0, lg)
        A = T - 1                                                                 # start offsets a = 1..T-1
        er = env_t.repeat(A)                                                      # (A b,) offset-major
        for si, (nm, d) in enumerate(sources):
            if d is None:
                hs, cs = h0.expand(A * b, H), c0.expand(A * b, H)
            elif d.ndim == 1:
                dl = loc[d[rows]]
                assert (dl >= 0).all(), f"{nm}: a donor outside the chunk's environments"
                j = ix(dl)
                hs = Hs[j, 1:T].transpose(0, 1).reshape(A * b, H)                 # the donor's state after a steps
                cs = Cs[j, 1:T].transpose(0, 1).reshape(A * b, H)
            else:
                dl = loc[d[rows][:, 1:T]]                                         # (b, A): the donor of window start a
                assert (dl >= 0).all(), f"{nm}: a donor outside the chunk's environments"
                j = ix(dl.T)                                                      # (A, b) offset-major
                aa = torch.arange(1, T, device=dev)[:, None].expand(A, b)
                hs = Hs[j, aa].reshape(A * b, H)
                cs = Cs[j, aa].reshape(A * b, H)
            for k in range(1, WMAX + 1):
                na = T - k                                                        # offsets a = 1..T-k still inside T
                m = na * b
                xin = table[er[:m], code_t[:, k:k + na].transpose(0, 1).reshape(m)]   # input a + k (1-based)
                hs, cs = _step(xin, hs[:m], cs[:m], Whh)
                del xin
                inc = Fn.linear(hs, Wro, bro).view(na, b, NV * R).transpose(0, 1)    # (b, na, 20) at t = k+1..T
                finish(1 + si * WMAX + k - 1, torch.cat([lg[:, :k], inc], 1))
            del hs, cs
        del Hs, Cs, lg
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    out["info"] = dict(seconds=time.time() - t0, n=n,
                       peak_gpu_mb=(torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else 0.))
    return out


def run_ck(run, ep, force=False):
    """one checkpoint -> OUT/<run>/ck_EEEEE.npz: ans (NC, n, 2) int8, accg (NC, NE, T) int16, lqg (NC, NE) float64,
    and the per-checkpoint check against the stored start-state truncation (td_analysis trunc_right: w = 1, 2, 5 and
    intact, both variables at T)"""
    path = ck_path(run, ep)
    if os.path.exists(path) and not force:
        return
    t0 = time.time()
    wts = ti.load(run, ep)
    fo = net_cuts(wts)
    P = probes()
    accg = fo["rg"].reshape(NC, NE, PER_ENV, T).sum(2).astype(np.int16)
    lqg = fo["lqg"].reshape(NC, NE, PER_ENV).sum(2)
    right_T = fo["ans"] == P["r"][None]                                            # (NC, n, 2)
    tr = np.load(os.path.join(ti.AN, run, f"ck_{int(ep):05d}.npz"))["trunc_right"]  # (n, 4, 2): w1, w2, w5, intact
    chk = dict(trunc_w1=int((right_T[ci("start", 1)] != tr[:, 0]).sum()),
               trunc_w2=int((right_T[ci("start", 2)] != tr[:, 1]).sum()),
               trunc_w5=int((right_T[ci("start", 5)] != tr[:, 2]).sum()),
               trunc_intact=int((right_T[0] != tr[:, 3]).sum()))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, ans=fo["ans"], accg=accg, lqg=lqg, conds=np.array(CONDS), run=run, ep=int(ep),
                        x=float(gap.train_x(ep)), fingerprint=ti.fingerprint(wts), forward_s=fo["info"]["seconds"],
                        peak_gpu_mb=fo["info"]["peak_gpu_mb"], seconds=time.time() - t0, checks=json.dumps(chk))
    os.replace(tmp, path)
    acc = accg[:, :, -1].sum(1) / (NE * PER_ENV)
    print(f"{time.strftime('%H:%M:%S')} {run} raw {ep} (training {gap.train_x(ep):,.0f}): forward "
          f"{fo['info']['seconds']:.0f} s, total {time.time() - t0:.0f} s, peak GPU {fo['info']['peak_gpu_mb']:.0f} MB; "
          f"goal accuracy intact {acc[0]:.4f}, w 1/5/15 start {acc[ci('start', 1)]:.4f}/{acc[ci('start', 5)]:.4f}/"
          f"{acc[ci('start', 15)]:.4f}, random {acc[ci('random', 1)]:.4f}/{acc[ci('random', 5)]:.4f}/"
          f"{acc[ci('random', 15)]:.4f}; checks {chk}", flush=True)


def run_scheme(ep, force=False):
    """one Trained RNN checkpoint under the other donor schemes (random cell, w = 1..15) -> OUT/schemes/ck_EEEEE.npz:
    ans (1 + len(SCHEMES) * WMAX, n, 2) int8 (intact, then scheme-major), conds; the intact answers equal the main
    cache's (checked)"""
    path = scheme_path(ep)
    if os.path.exists(path) and not force:
        return
    t0 = time.time()
    wts = ti.load("trained", ep)
    D = scheme_donors()
    fo = net_cuts(wts, sources=[(s, D[s]) for s in SCHEMES])
    main = ck_path("trained", ep)
    same = int((np.load(main)["ans"][0] != fo["ans"][0]).sum()) if os.path.exists(main) else -1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, ans=fo["ans"], conds=np.array(fo["conds"]), ep=int(ep), x=float(gap.train_x(ep)),
                        fingerprint=ti.fingerprint(wts), seconds=time.time() - t0, intact_vs_main=same)
    os.replace(tmp, path)
    print(f"{time.strftime('%H:%M:%S')} schemes raw {ep} (training {gap.train_x(ep):,.0f}): {time.time() - t0:.0f} s; "
          f"intact answers differing from the main cache {same}", flush=True)


def _gpu_free_mb():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-i", vis],
                       capture_output=True, text=True).stdout
    used, tot = (float(v) for v in q.strip().split(","))
    return tot - used


def worker(kinds=("cuts", "schemes")):
    """runs the missing jobs (jobs(kind) for each kind, in order) in this one process on the visible GPU; before each
    checkpoint waits until it has NEED_MB free; a CUDA out-of-memory error frees the cache, waits a minute and retries (at
    most 20 times); a STOP file in TMP ends the loop"""
    for kind in kinds:
        for run, ep in jobs(kind):
            for attempt in range(20):
                if os.path.exists(os.path.join(TMP, "STOP")):
                    print("STOP file: exit", flush=True)
                    return
                torch.cuda.empty_cache()
                while _gpu_free_mb() < NEED_MB:
                    time.sleep(60)
                try:
                    run_ck(run, ep) if kind == "cuts" else run_scheme(ep)
                    break
                except torch.cuda.OutOfMemoryError:
                    print(f"{time.strftime('%H:%M:%S')} {run} {ep}: CUDA out of memory (attempt {attempt + 1}); retry in "
                          "60 s", flush=True)
                    torch.cuda.empty_cache()
                    time.sleep(60)
            torch.cuda.empty_cache()
    print(f"{time.strftime('%H:%M:%S')} all jobs done", flush=True)


def harvest(gpu="1"):
    """launches the one worker process detached (setsid; the GPU `gpu`, 1 by default: John freed it 11-02; 2 threads;
    log TMP/worker.log); returns at once"""
    os.makedirs(TMP, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MPLBACKEND="Agg",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log = open(os.path.join(TMP, "worker.log"), "a")
    p = subprocess.Popen(["timeout", "18000", PY, os.path.abspath(__file__), "worker"], stdout=log,
                         stderr=subprocess.STDOUT, env=env, cwd=ROOT, start_new_session=True)
    with open(os.path.join(TMP, "worker.pid"), "w") as f:
        f.write(str(p.pid))
    print("worker", p.pid, "log", os.path.join(TMP, "worker.log"))



# ═══════════════════════════════════════════════════════════════════
# Measures (per-environment sums; 1,000 shared environment resamples)
# ═══════════════════════════════════════════════════════════════════

STAGES = (320, 560, 880, 1200, 1680, 2400, 4000, 39996)          # training episodes quoted in the numbers
BANDS = ((2, 2), (3, 6), (7, 9), (10, 15))                       # memory steps 2..15, split explicitly


def _env_sums(ans):
    """answers at T (..., n, 2) -> per-environment sums (..., NE) of: acc (goal answer right), both (both right), far
    (one or both answers >= 2 values off; td_tt_intervene far_T)"""
    P = probes()
    r, g = P["r"], P["goal"]
    a = np.asarray(ans).astype(np.int16)
    right = a == r
    out = dict(acc=right[..., np.arange(len(g)), g], both=right.all(-1),
               far=np.abs(a - r).max(-1) >= 2)
    return {k: v.reshape(v.shape[:-1] + (NE, PER_ENV)).sum(-1).astype(np.float64) for k, v in out.items()}


def _boot(sums):
    """(..., NE) per-environment sums of a per-probe quantity -> (value (...), resamples (..., NB)): the share over
    all probes, and in every shared resample (td_tt_gap.boot_counts(); every environment holds 200 probes)"""
    C = gap.boot_counts()
    return sums.sum(-1) / (NE * PER_ENV), (sums @ C.T) / (C.sum(1) * PER_ENV)


def load_run(run):
    """the harvested checkpoints of a run, stacked: x (K,), eps, ans (K, NC, n, 2), accg (K, NC, NE, T), lqg (K, NC,
    NE), meta"""
    key = ("run", run)
    if key not in _C:
        eps = [e for e in selection(run) if os.path.exists(ck_path(run, e))]
        if not eps:
            return None
        Z = [np.load(ck_path(run, e)) for e in eps]
        D = dict(eps=np.array(eps), x=gap.train_x(np.array(eps)), ans=np.stack([z["ans"] for z in Z]),
                 accg=np.stack([z["accg"] for z in Z]).astype(np.float64), lqg=np.stack([z["lqg"] for z in Z]),
                 meta=[dict(checks=json.loads(str(z["checks"])), seconds=float(z["seconds"]),
                            forward_s=float(z["forward_s"]), peak=float(z["peak_gpu_mb"])) for z in Z])
        _C[key] = D
    return _C[key]


def load_bayes():
    if "B" not in _C:
        _C["B"] = bayes_cache()
    return _C["B"]


def measures(run="trained"):
    """the network's rows per checkpoint and condition with resamples, Bayes's and naive's: acc / far / both (K, NC)
    and *_b (K, NC, NB); bayes_acc ... (NC,), (NC, NB); naive_acc ... (scalar, (NB,)). Loss of a condition = acc(intact)
    - acc(cut) (accuracy lost by the cut), paired in every resample"""
    key = ("meas", run)
    if key not in _C:
        D, B = load_run(run), load_bayes()
        out = dict(x=D["x"], eps=D["eps"])
        for k, v in _env_sums(D["ans"]).items():
            out[k], out[k + "_b"] = _boot(v)
        for k, v in _env_sums(B["ans"]).items():
            out["bayes_" + k], out["bayes_" + k + "_b"] = _boot(v)
        for k, v in _env_sums(B["naive_ans"]).items():
            out["naive_" + k], out["naive_" + k + "_b"] = _boot(v)
        _C[key] = out
    return _C[key]


def loss(run="trained", row="acc"):
    """accuracy lost by every cut: net (K, S, W) = row(intact) - row(cut) and resamples net_b (K, S, W, NB); Bayes
    (S, W), (S, W, NB); excess = net - Bayes (paired). row 'far': far errors ADDED by the cut (cut - intact)"""
    M = measures(run)
    sg = -1. if row == "far" else 1.
    v, vb, bv, bb = M[row], M[row + "_b"], M["bayes_" + row], M["bayes_" + row + "_b"]
    K = len(M["x"])
    net = sg * (v[:, :1] - v[:, 1:]).reshape(K, len(SOURCES), WMAX)
    net_b = sg * (vb[:, :1] - vb[:, 1:]).reshape(K, len(SOURCES), WMAX, NB)
    bay = sg * (bv[:1] - bv[1:]).reshape(len(SOURCES), WMAX)
    bay_b = sg * (bb[:1] - bb[1:]).reshape(len(SOURCES), WMAX, NB)
    return dict(x=M["x"], net=net, net_b=net_b, bayes=bay, bayes_b=bay_b, ex=net - bay, ex_b=net_b - bay_b[None])


def _iv(b, axis=-1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(b, 5, axis=axis), np.nanpercentile(b, 95, axis=axis)


def _k(x, xv):
    return int(np.argmin(np.abs(np.log(x) - np.log(xv))))


def onset_core(x, net, net_b, bay, bay_b, share=.5):
    """per window (column): the training episode after which net (K, W) stays above share x bay (W,) (td_tt_gap.crossing:
    stays past, log-interpolated; the level recomputed in every resample from bay_b (W, NB), paired with net_b (K, W,
    NB); never = after the end) -> dict(p, lo, hi, frac (share of resamples that cross) (W,), boot (W, NB))"""
    W_ = net.shape[1]
    out = dict(p=np.full(W_, np.nan), lo=np.full(W_, np.nan), hi=np.full(W_, np.nan), frac=np.zeros(W_),
               boot=np.full((W_, NB), np.nan))
    for w in range(W_):
        sg = 1. if bay[w] >= 0 else -1.
        out["p"][w] = gap.crossing(x, sg * (net[:, w] - share * bay[w]), 0.)
        cb = gap.crossing(x, sg * (net_b[:, w].T - share * bay_b[w][:, None]), 0.)
        out["boot"][w] = cb
        out["lo"][w], out["hi"][w], out["frac"][w] = gap.interval(cb)
    return out


def random_loss(scheme="engine", run="trained"):
    """the random-cell loss under a donor scheme: 'engine' (the main cache: one fixed other-cell probe per probe, seed
    20261201) or one of SCHEMES (Trained RNN only), with Bayes's under the identical substitution: dict(x, net (K, W),
    net_b (K, W, NB), bayes (W,), bayes_b (W, NB))"""
    if scheme == "engine":
        d = loss(run)
        s = SOURCES.index("random")
        return dict(x=d["x"], net=d["net"][:, s], net_b=d["net_b"][:, s], bayes=d["bayes"][s], bayes_b=d["bayes_b"][s])
    key = ("scheme", scheme)
    if key not in _C:
        eps = [e for e in selection("trained") if os.path.exists(scheme_path(e))]
        k = SCHEMES.index(scheme)
        sel = [0] + list(range(1 + k * WMAX, 1 + (k + 1) * WMAX))
        A = np.stack([np.load(scheme_path(e))["ans"][sel] for e in eps])           # (K, 16, n, 2)
        v, vb = _boot(_env_sums(A)["acc"])
        Bm, Bs = load_bayes(), bayes_schemes()
        bv, bb = _boot(_env_sums(np.concatenate([Bm["ans"][:1], Bs["ans"][k * WMAX:(k + 1) * WMAX]]))["acc"])
        _C[key] = dict(x=gap.train_x(np.array(eps)), net=v[:, :1] - v[:, 1:], net_b=vb[:, :1] - vb[:, 1:],
                       bayes=bv[0] - bv[1:], bayes_b=bb[:1] - bb[1:])
    return _C[key]


def accuracy_marks():
    """the Trained RNN's goal accuracy at T passing naive (c0) and half-way to Bayes (c50) from the td_tt_jumps export
    (td_tt_gap: all 114 checkpoints, the same 1,000 environment resamples, so pairs with the onsets here are paired):
    {'c0': dict(p, lo, hi, boot), 'c50': ...}"""
    z = gap.load("jumps")
    i = next(i for i, (n, r) in enumerate(zip(z["names"], z["run"])) if str(n) == "goal accuracy at T" and
             str(r) == "trained")
    return {k: dict(p=float(z[k][i]), lo=float(z[k + "_lo"][i]), hi=float(z[k + "_hi"][i]), boot=z[k + "_boot"][i])
            for k in ("c0", "c50")}


def run_marks():
    """every run's goal accuracy at T passing naive for good (c0) and half-way to Bayes (c50), from td_tt_replication's
    synthesis (all 114 checkpoints of each run, the same 1,000 shared resamples): {run: {'c0': dict(p, lo, hi, share,
    boot), 'c50': ...}}"""
    if "marks" not in _C:
        import td_tt_replication as rp
        S = rp.load()
        _C["marks"] = {run: {k: S[run]["rows"]["acc"]["events"][lev] for k, lev in (("c0", 0.), ("c50", .5))}
                       for run in ("trained",) + REP}
    return _C["marks"]


def first_cross(x, y, level):
    """the first training episode at which y reaches `level` (log-interpolated from the previous checkpoint); NaN if
    never (td_tt_memcuts.first_cross, copied)"""
    y = np.asarray(y, np.float64)
    shp = y.shape[:-1]
    y2 = y.reshape(-1, y.shape[-1])
    lev = np.broadcast_to(np.asarray(level, np.float64), shp).reshape(-1)
    with np.errstate(invalid="ignore"):
        past = y2 >= lev[:, None]
    out = np.full(len(y2), np.nan)
    k = np.argmax(past, 1)
    lx = np.log(np.asarray(x, np.float64))
    for i in np.flatnonzero(past.any(1)):
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
    """a transient's timing (td_tt_memory_events' rule, as td_tt_memcuts.rise_fall): peak = the maximum of the
    3-checkpoint running mean; rise = the first crossing of half the peak; fall = after which the curve stays below half
    the peak to the end; levels recomputed in every resample (yb (NB, K)); intervals by td_tt_gap.interval"""
    pk, pkb = _runmean3(y), _runmean3(yb)
    out = dict(peak=float(np.nanmax(pk)), peak_x=float(x[np.nanargmax(pk)]), peak_boot=np.nanmax(pkb, 1))
    out["rise_boot"] = first_cross(x, yb, out["peak_boot"] / 2)
    out["fall_boot"] = gap.crossing(x, yb, out["peak_boot"] / 2, sign=-1.)
    out["rise"] = first_cross(x, y, out["peak"] / 2)
    out["fall"] = gap.crossing(x, y, out["peak"] / 2, sign=-1.)
    for k in ("rise", "fall"):
        out[k + "_lo"], out[k + "_hi"], out[k + "_share"] = gap.interval(out[k + "_boot"])
    out["peak_lo"], out["peak_hi"] = np.nanpercentile(out["peak_boot"], [5, 95])
    return out


def _order(a_b, b_b, a_lo, a_hi, b_lo, b_hi):
    """the interval rule for two timings (never order overlapping intervals; NaN bounds = after the end) and the
    paired share P(a < b) over the resamples (NaN = after the end)"""
    a_, b_ = np.where(np.isnan(a_b), np.inf, a_b), np.where(np.isnan(b_b), np.inf, b_b)
    p = float((a_ < b_).mean())
    a_hi_, b_hi_ = (np.inf if not np.isfinite(v) else v for v in (a_hi, b_hi))
    rule = "before" if a_hi_ < b_lo else "after" if a_lo > b_hi_ else "with"
    return rule, p


def _cx(p, lo, hi, frac=1.):
    if not np.isfinite(p):
        return "never" + ("" if frac <= .005 else f" ({frac:.0%} of resamples cross)")
    h = f"{hi:,.0f}" if np.isfinite(hi) else "never"
    return f"{p:,.0f} [{lo:,.0f}-{h}]" + ("" if frac >= .995 else f" ({frac:.0%})")


def pair_order(o):
    """the onsets' order over the window pairs w < w' (o: onset_core's dict): (number ordered shorter-before-longer by
    the interval rule, number the other way, number of pairs, the list [(w, w', rule, P)])"""
    W_ = len(o["p"])
    res = [(a + 1, b + 1) + _order(o["boot"][a], o["boot"][b], o["lo"][a], o["hi"][a], o["lo"][b], o["hi"][b])
           for a in range(W_) for b in range(a + 1, W_)]
    return sum(r_[2] == "before" for r_ in res), sum(r_[2] == "after" for r_ in res), len(res), res


def onset_trend(o, mask=None):
    """per resample the least-squares slope of log10(onset) on w over the windows with an onset in that resample (and
    in mask; at least 3): decades of training per extra step of memory; value from the full-sample onsets. o:
    onset_core's dict -> (value, resamples (NB,))"""
    W_ = len(o["p"])
    ws = np.arange(1, W_ + 1, dtype=np.float64)
    mask = np.ones(W_, bool) if mask is None else np.asarray(mask, bool)

    def slope(y):
        ok = np.isfinite(y) & mask
        if ok.sum() < 3:
            return np.nan
        return float(np.polyfit(ws[ok], np.log10(y[ok]), 1)[0])
    return slope(o["p"]), np.array([slope(o["boot"][:, i]) for i in range(NB)])


def _xstep(v, b):
    """a log10 slope per step as the factor per step 'x1.148 [x1.142, x1.166]'"""
    lo, hi = np.nanpercentile(b, [5, 95])
    return f"x{10 ** v:.3f} [x{10 ** lo:.3f}, x{10 ** hi:.3f}]"


def thinned_levels(src="random"):
    """the thinned observer (thinned_cache) per p: intact goal accuracy acc (P,), acc_b (P, NB), and its loss under the
    cut L (P, W), L_b (P, W, NB); the per-environment counts are averaged over the mask seeds first"""
    Z = thinned_cache()
    a, ab = _boot(Z["right"].astype(np.float64).mean(1))                          # (P, C), (P, C, NB)
    j = [list(Z["conds"]).index(f"w{w}_{src}") for w in WS]
    return dict(p=Z["p"], acc=a[:, 0], acc_b=ab[:, 0], L=a[:, :1] - a[:, j], L_b=ab[:, :1] - ab[:, j])


def matched(run="trained", src="random"):
    """the accuracy-matched perfect-memory observer on the run's own accuracy trajectory: at each checkpoint, the
    thinned observer's L(w) at the intact goal accuracy of the run (linear interpolation in accuracy over P_GRID; in
    every resample with that resample's accuracies on both sides): Lm (K, W), Lm_b (K, W, NB); and the run's L(w) minus
    it: D (K, W), D_b (K, W, NB)"""
    key = ("matched", run, src)
    if key not in _C:
        th = thinned_levels(src)
        M = measures(run)
        A, Ab = M["acc"][:, 0], M["acc_b"][:, 0]
        o = np.argsort(th["acc"], kind="stable")
        Lm = np.stack([np.interp(A, th["acc"][o], th["L"][o, w]) for w in range(WMAX)], 1)
        Lm_b = np.empty((len(A), WMAX, NB))
        for b in range(NB):
            o = np.argsort(th["acc_b"][:, b], kind="stable")
            for w in range(WMAX):
                Lm_b[:, w, b] = np.interp(Ab[:, b], th["acc_b"][o, b], th["L_b"][o, w, b])
        d = loss(run)
        s = SOURCES.index(src)
        _C[key] = dict(x=M["x"], Lm=Lm, Lm_b=Lm_b, D=d["net"][:, s] - Lm, D_b=d["net_b"][:, s] - Lm_b)
    return _C[key]


def _acc_at_zero(acc, y):
    """the accuracy at which y (increasing with accuracy) first reaches 0, linear between grid points; NaN if never"""
    o = np.argsort(acc, kind="stable")
    a_, y_ = acc[o], y[o]
    k = np.flatnonzero((y_[:-1] < 0) & (y_[1:] >= 0))
    if not len(k):
        return np.nan
    k = k[0]
    return float(a_[k] + (0 - y_[k]) / (y_[k + 1] - y_[k]) * (a_[k + 1] - a_[k]))


def matched_threshold(src="random", share=.5):
    """per w: the intact goal accuracy at which the thinned observer's L(w) reaches share x Bayes's L(w) under the same
    cut: value (W,), resamples (W, NB)"""
    th = thinned_levels(src)
    d = loss("trained")
    s = SOURCES.index(src)
    v = np.array([_acc_at_zero(th["acc"], th["L"][:, w] - share * d["bayes"][s, w]) for w in range(WMAX)])
    b = np.array([[_acc_at_zero(th["acc_b"][:, i], th["L_b"][:, w, i] - share * d["bayes_b"][s, w, i])
                   for i in range(NB)] for w in range(WMAX)])
    return v, b


def _band(L, ax):
    """the mean benefit per memory step over each band of BANDS from losses L(w) along axis ax: (L(k0 - 1) - L(k1)) /
    (k1 - k0 + 1)"""
    L = np.moveaxis(np.asarray(L), ax, -1)
    out = np.stack([(L[..., k0 - 2] - L[..., k1 - 1]) / (k1 - k0 + 1) for k0, k1 in BANDS], -1)
    return np.moveaxis(out, -1, ax)


def bands(run="trained", src="random"):
    """the mean benefit per memory step in each band (benefit of the k-th step = acc(cut to k) - acc(cut to k - 1) =
    L(k - 1) - L(k)): net (K, nb), net_b (K, nb, NB), Bayes's (nb,), (nb, NB), the matched observer's (K, nb), (K, nb,
    NB)"""
    d = loss(run)
    s = SOURCES.index(src)
    m = matched(run, src)
    return dict(x=d["x"], net=_band(d["net"][:, s], 1), net_b=_band(d["net_b"][:, s], 1), bayes=_band(d["bayes"][s], 0),
                bayes_b=_band(d["bayes_b"][s], 0), mat=_band(m["Lm"], 1), mat_b=_band(m["Lm_b"], 1))


def span(E, level=.02):
    """the largest window w whose value E (..., W) exceeds `level` (0 = none)"""
    return np.where(np.asarray(E) > level, np.arange(1, WMAX + 1), 0).max(-1)


def start_transient(run="trained"):
    """the start-state excess (network's loss - Bayes's) per w as a transient (rise_fall): dict of arrays (W,) peak,
    peak_lo, peak_hi, peak_x, rise, rise_lo, rise_hi, rise_share, fall, fall_lo, fall_hi, fall_share; and the absolute
    timings: pos (stays above 0 from), pos_lo, pos_hi, pos_share, first_pos (the first checkpoint above 0), and for each
    level in (.05, .02) the time after which it stays below that level (below_L, _lo, _hi, _share; only where the peak's
    5% bound exceeds the level)"""
    d = loss(run)
    s = SOURCES.index("start")
    x = d["x"]
    keys = ("peak", "peak_lo", "peak_hi", "peak_x", "rise", "rise_lo", "rise_hi", "rise_share", "fall", "fall_lo",
            "fall_hi", "fall_share", "pos", "pos_lo", "pos_hi", "pos_share", "first_pos") + tuple(
        f"below{lv}{k}" for lv in ("05", "02") for k in ("", "_lo", "_hi", "_share"))
    out = {k: np.full(WMAX, np.nan) for k in keys}
    for w in range(WMAX):
        y, yb = d["ex"][:, s, w], d["ex_b"][:, s, w].T
        rf = rise_fall(x, y, yb)
        for k in keys[:12]:
            out[k][w] = rf[k]
        out["pos"][w] = gap.crossing(x, y, 0.)
        out["pos_lo"][w], out["pos_hi"][w], out["pos_share"][w] = gap.interval(gap.crossing(x, yb, 0.))
        out["first_pos"][w] = x[np.argmax(y > 0)] if (y > 0).any() else np.nan
        k0 = int(np.nanargmax(_runmean3(y)))
        for lv, nm in ((.05, "05"), (.02, "02")):
            if rf["peak_lo"] > lv:
                out[f"below{nm}"][w] = gap.crossing(x[k0:], y[k0:], lv, sign=-1.)
                bt = gap.crossing(x[k0:], yb[:, k0:], lv, sign=-1.)
                out[f"below{nm}_lo"][w], out[f"below{nm}_hi"][w], out[f"below{nm}_share"][w] = gap.interval(bt)
    return out


def own_clock(o, c0):
    """an onset's own-clock place: onset / the run's passing naive, per resample (NaN onsets = after the end, inf):
    value, 5-95% over the resamples in which the run passes naive, and the resampled ratios"""
    ok = np.isfinite(c0["boot"])
    rb = np.where(np.isnan(o["boot"]), np.inf, o["boot"])[..., ok] / c0["boot"][ok]
    lo, hi = np.percentile(rb, 5, axis=-1, method="lower"), np.percentile(rb, 95, axis=-1, method="higher")
    return o["p"] / c0["p"], lo, hi, rb


def acc_at(run, t, tb):
    """the run's intact goal accuracy at T at the training times t (W,) and, per resample, tb (W, NB) (each
    resample's accuracy curve, log-interpolated between checkpoints; NaN where the time is NaN): value (W,), resamples
    (W, NB)"""
    M = measures(run)
    lx, a, ab = np.log(M["x"]), M["acc"][:, 0], M["acc_b"][:, 0]
    v = np.array([np.interp(np.log(tt), lx, a) if np.isfinite(tt) else np.nan for tt in t])
    b = np.full(np.shape(tb), np.nan)
    for i in range(NB):
        ok = np.isfinite(tb[:, i])
        b[ok, i] = np.interp(np.log(tb[ok, i]), lx, ab[:, i])
    return v, b



# ═══════════════════════════════════════════════════════════════════
# Checks (-> OUT/checks.json)
# ═══════════════════════════════════════════════════════════════════

CHECK_EPS = (1100, 2100, 49995)                            # raw: training 880, 1,680, 39,996
CHECK_WS = (1, 2, 5, 10, 15)
MEMCUTS = os.path.join(TD, "through_training", "memcuts")
VERIFIER = ("/tmp/claude-1002/-home-johns-Desktop-CognitiveGridworld/7af6b091-54ff-45a0-b4dd-ef0a3bc9c778/scratchpad/"
            "memflip/check_memlen")                        # the adversarial check's independent routes (optional)


def _brute(wts, rows, steps, ws, dons, srcs=SOURCES):
    """float64 numpy brute force of the cut increments: for probe i, step t and window w, the read-out increment at t
    of the cell rerun over i's inputs t-w+1..t from the source's state at t - w (start: (h0, c0); a donor: its own
    intact float64 state after t - w steps; dons[src] (n,) or (n, T): the donor of window start t - w). Returns
    {(i, t, w, src): (20,)} and the intact increments"""
    import td_tt_wormholes as tw
    P = probes()
    W = {k: v.double().numpy() for k, v in wts.items()}
    H = W["LSTM.weight_hh_l0"].shape[1]

    def drive(i, t):                                         # input drive at 1-based step t of probe i
        o = P["obs"][i, t - 1].astype(np.float64)
        inp = np.concatenate([o, P["Z"][P["env"][i]].reshape(-1).astype(np.float64)])
        x = np.maximum(W["classifier_readin.weight"] @ inp + W["classifier_readin.bias"], 0.)
        return W["LSTM.weight_ih_l0"] @ x + W["LSTM.bias_ih_l0"] + W["LSTM.bias_hh_l0"]

    def step(xin, h, c):
        pre = xin + W["LSTM.weight_hh_l0"] @ h
        i_, f, g, o = np.split(pre, 4)
        sig = lambda v: 1. / (1. + np.exp(-v))              # noqa: E731
        c = sig(f) * c + sig(i_) * np.tanh(g)
        return sig(o) * np.tanh(c), c

    def ro(h):
        return W["classifier_readout.weight"] @ h + W["classifier_readout.bias"]

    h0, c0 = W["STM"].reshape(H), W["LTM"].reshape(H)
    cache = {}

    def states(i):
        if i not in cache:
            h, c, S = h0, c0, [(h0, c0)]
            for t in range(1, T + 1):
                h, c = step(drive(i, t), h, c)
                S.append((h, c))
            cache[i] = S
        return cache[i]
    assert tw._PAT.shape == (32, NO)
    out, intact = {}, {}
    for i in rows:
        Si = states(i)
        for t in steps:
            intact[(i, t)] = ro(Si[t][0])
            for w in ws:
                for src in srcs:
                    if src == "start":
                        h, c = h0, c0
                    else:
                        dd = np.asarray(dons[src])
                        h, c = states(int(dd[i] if dd.ndim == 1 else dd[i, t - w]))[t - w]
                    for u in range(t - w + 1, t + 1):
                        h, c = step(drive(i, u), h, c)
                    out[(i, t, w, src)] = ro(h)
    return out, intact


def checks():
    """the route against the engine (td_tt_intervene.forward, full probes, 3 checkpoints, w = 1, 2, 5, 10, 15, three
    sources); a probe as its own donor (= intact); a float64 brute force; every harvested checkpoint against the stored
    start-state truncation; td_tt_memcuts' stored answers (w = 2, 5, 10) at the shared checkpoints; Bayes against the
    cumulative-likelihood route (a copy of td_tt_memcuts.bayes_logq), the module itself, windowed Bayes and memcuts'
    stored Bayes answers; the other donor schemes (the fixed-seed route against the engine with the same donors, the
    per-window-start route against a float64 brute force, both against the adversarial check's independent route, Bayes
    against the cumulative-likelihood route); the thinned observer at p = 1 against Bayes and against the check's own
    thinned observer -> OUT/checks.json"""
    P = probes()
    n = len(P["env"])
    dev = ti._device()
    res = dict(device=str(dev), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
    t00 = time.time()
    # 1. the engine
    keep = tuple(f"w{w}_{s}" for s in SOURCES for w in CHECK_WS)
    eng = {}
    for ep in CHECK_EPS:
        wts = ti.load("trained", ep)
        t0 = time.time()
        mine = net_cuts(wts, dev=dev, keep=("intact",) + keep)
        e = dict(route_s=time.time() - t0, peak_gpu_mb=mine["info"]["peak_gpu_mb"])
        for s in SOURCES:
            t0 = time.time()
            fo = ti.forward(wts, device=dev, cut=[(w, s) for w in CHECK_WS])
            e[f"engine_s_{s}"] = time.time() - t0
            e["intact_max_abs"] = float(np.abs(fo["logq"] - mine["lq"]["intact"]).max())
            e["intact_ans_T_diff"] = int((fo["right"][:, -1] != (mine["ans"][0] == P["r"])).sum())
            for w in CHECK_WS:
                nm = f"w{w}_{s}"
                rt = fo["right_cut"][nm]
                e[nm] = dict(max_abs=float(np.abs(fo["logq_cut"][nm] - mine["lq"][nm]).max()),
                             ans_T_diff=int((rt[:, -1] != (mine["ans"][ci(s, w)] == P["r"])).sum()),
                             goal_steps_diff=int((rt[np.arange(n), :, P["goal"]] != mine["rg"][ci(s, w)]).sum()),
                             cut_vs_intact=float(np.abs(mine["lq"][nm] - mine["lq"]["intact"]).max()))
            del fo
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        eng[str(ep)] = e
        print(f"engine check raw {ep}: route {e['route_s']:.0f} s, max |d log q| "
              f"{max(e[k]['max_abs'] for k in keep):.2e}, answers at T differ "
              f"{sum(e[k]['ans_T_diff'] for k in keep)} of {len(keep) * 2 * n}, goal answers per step differ "
              f"{sum(e[k]['goal_steps_diff'] for k in keep)}; cut effects up to "
              f"{max(e[k]['cut_vs_intact'] for k in keep):.1f} ({time.time() - t00:.0f} s)", flush=True)
        del mine
    res["engine"] = eng
    # 2. a probe as its own donor (the rerun from its own state over its own inputs = the intact run)
    wts = ti.load("trained", 2100)
    ident = np.arange(n)
    kself = tuple(f"w{w}_{s}" for s in ("same_cell", "random") for w in (1, 7, 15))
    senv = np.arange(0, NE, 5)                                              # 10 environments (2,000 probes)
    mine = net_cuts(wts, dev=dev, envs=senv, dons={"same_cell": ident, "random": ident}, keep=("intact",) + kself)
    rows_ = [ci(s, w) for s in ("same_cell", "random") for w in WS]
    res["self_donor"] = dict(max_abs=float(max(np.abs(mine["lq"][k] - mine["lq"]["intact"]).max() for k in kself)),
                             ans_T_diff=int((mine["ans"][rows_] != mine["ans"][0][None]).sum()),
                             goal_steps_diff=int((mine["rg"][rows_] != mine["rg"][0][None]).sum()),
                             of=int(len(rows_) * mine["info"]["n"] * NV), probes=int(mine["info"]["n"]))
    print("self donor:", res["self_donor"], flush=True)
    # 3. float64 brute force (4 probes, steps 16 / 19 / 30, w = 1 / 7 / 15, three sources and the fresh scheme)
    brows, bsteps, bws = (3, 2718, 5555, 9001), (16, 19, 30), (1, 7, 15)
    Dsc = scheme_donors()
    kinc = ("intact",) + tuple(f"w{w}_{s}" for s in SOURCES + ("fresh",) for w in bws)
    srcs = [(s, None if s == "start" else donors()[s]) for s in SOURCES] + [("fresh", Dsc["fresh"])]
    mine = net_cuts(wts, dev=dev, envs=P["env"][list(brows)], keep_inc=kinc, sources=srcs)
    pos = {i: int(np.searchsorted(mine["rows"], i)) for i in brows}
    dd = dict(donors(), fresh=Dsc["fresh"])
    bf, bi = _brute(wts, brows, bsteps, bws, dd, srcs=SOURCES + ("fresh",))
    d_cut = {s: max(float(np.abs(mine["inc"][f"w{w}_{s_}"][pos[i], t - 1] - bf[(i, t, w, s_)]).max())
                    for (i, t, w, s_) in bf if s_ == s) for s in SOURCES + ("fresh",)}
    d_int = max(float(np.abs(mine["inc"]["intact"][pos[i], t - 1] - bi[(i, t)]).max()) for (i, t) in bi)
    moved = max(float(np.abs(bf[(i, t, w, s)] - bi[(i, t)]).max()) for (i, t, w, s) in bf)
    res["brute"] = dict(max_abs_cut=max(d_cut.values()), max_abs_cut_fresh=d_cut["fresh"], max_abs_intact=d_int,
                        cut_moves_increment_by=moved, n=len(bf))
    print("brute force:", res["brute"], flush=True)
    del mine
    # 4. the stored start-state truncation at every harvested checkpoint
    tr = {}
    for run in ("trained", "echo") + REP:
        D = load_run(run)
        if D is not None:
            c = [m["checks"] for m in D["meta"]]
            tr[run] = dict(checkpoints=len(c), **{k: int(sum(m[k] for m in c)) for k in c[0]})
    res["trunc"] = tr
    print("stored truncation:", tr, flush=True)
    # 5. td_tt_memcuts' stored answers at the shared checkpoints (w = 2, 5, 10; three sources)
    mc = {}
    for run in ("trained", "echo"):
        D = load_run(run)
        if D is None:
            continue
        nd, nck, dacc = 0, 0, 0.
        for k_, ep in enumerate(D["eps"]):
            f = os.path.join(MEMCUTS, run, f"ck_{int(ep):05d}.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f)
            conds = [str(c) for c in z["conds"]]
            nck += 1
            for nm in ["intact"] + [f"w{w}_{s}" for w in (2, 5, 10) for s in SOURCES]:
                mine_rt = D["ans"][k_, CONDS.index(nm)] == P["r"]
                nd += int((mine_rt != z["right_T"][conds.index(nm)]).sum())
                acc_m = mine_rt[np.arange(n), P["goal"]].mean()
                dacc = max(dacc, abs(acc_m - float(z["value"][conds.index(nm), 0])))
        mc[run] = dict(checkpoints=nck, answers_differ=nd, of=nck * 10 * n * NV, max_abs_acc=dacc)
    res["memcuts"] = mc
    print("memcuts stored answers:", mc, flush=True)
    # 6. Bayes
    B = load_bayes()
    lj = P["log_jb"].astype(np.float64)
    bres = dict(intact_vs_log_jb=float(np.abs(bayes_logq() - (lj - logsumexp(lj, -1, keepdims=True))).max()))
    L = P["L"][P["env"]]
    o = P["obs"].astype(np.float64)
    ll = np.einsum("nti,nijk->ntjk", o, np.log(L)) + np.einsum("nti,nijk->ntjk", 1 - o, np.log1p(-L))
    Lam = np.concatenate([np.zeros((n, 1, R, R)), np.cumsum(ll, 1)], 1)

    def lam_logq(w, src=None, d=None):                     # td_tt_memcuts.bayes_logq (11-02), copied; + (n, T) donors
        d = d if d is not None else (None if src == "start" else donors()[src])
        inc = np.zeros((n, T, NV, R))
        for t in range(1, T + 1):
            if t <= w:
                inc[:, t - 1] = _marg(Lam[:, t]) - _marg(Lam[:, t - 1])
            else:
                a = t - w
                base = 0. if d is None else (Lam[d, a] if d.ndim == 1 else Lam[d[:, a], a])
                inc[:, t - 1] = _marg(base + Lam[:, t] - Lam[:, a]) - _marg(base + Lam[:, t - 1] - Lam[:, a])
        cum = np.cumsum(inc, 1)
        return cum - logsumexp(cum, -1, keepdims=True)
    try:
        import td_tt_memcuts as tm                           # verification only (another workflow edits it)
        mod = tm.bayes_logq
    except Exception as err:                                 # noqa: BLE001
        mod, bres["module_import"] = None, repr(err)
    dl, dmod, dans = 0., 0., 0
    for s in SOURCES:
        for w in CHECK_WS:
            a = bayes_logq(w, s)
            dl = max(dl, float(np.abs(a - lam_logq(w, s)).max()))
            dans += int((a[:, -1].argmax(-1) != B["ans"][ci(s, w)]).sum())
            if mod is not None and w in (2, 5, 10):
                dmod = max(dmod, float(np.abs(a - mod(w, s)).max()))
    bres.update(vs_lam_copy=dl, vs_module=(dmod if mod is not None else None), cache_vs_recomputed_answers=dans)
    bres["window_vs_right_window"] = {w: int(((B["ans"][ci("start", w)] == P["r"]) != P[f"right_window{w}"]).sum())
                                      for w in (1, 2, 5)}
    zb = np.load(os.path.join(MEMCUTS, "bayes.npz"))
    conds = [str(c) for c in zb["conds"]]
    nd, dacc = 0, 0.
    for nm in ["intact"] + [f"w{w}_{s}" for w in (2, 5, 10) for s in SOURCES]:
        rt = B["ans"][CONDS.index(nm)] == P["r"]
        nd += int((rt != zb["right_T"][conds.index(nm)]).sum())
        dacc = max(dacc, abs(rt[np.arange(n), P["goal"]].mean() - float(zb["value"][conds.index(nm), 0])))
    bres.update(memcuts_answers_differ=nd, memcuts_max_abs_acc=dacc)
    res["bayes"] = bres
    print("Bayes:", bres, flush=True)
    # 7. the other donor schemes
    sch = {}
    wts = ti.load("trained", 2100)
    fo = ti.forward(wts, device=dev, cut=[(w, Dsc["fixed2"]) for w in (1, 5, 15)])
    mine = net_cuts(wts, dev=dev, sources=[("fixed2", Dsc["fixed2"])], keep=tuple(f"w{w}_fixed2" for w in (1, 5, 15)))
    sch["engine_fixed2_raw2100"] = dict(
        max_abs=float(max(np.abs(fo["logq_cut"][f"w{w}_donor{j}"] - mine["lq"][f"w{w}_fixed2"]).max()
                          for j, w in enumerate((1, 5, 15)))),
        ans_T_diff=int(sum((fo["right_cut"][f"w{w}_donor{j}"][:, -1] != (mine["ans"][1 + w - 1] == P["r"])).sum()
                           for j, w in enumerate((1, 5, 15)))), of=3 * n * NV)
    del fo, mine
    Bs = bayes_schemes()
    dlam, dans = 0., 0
    for k, sc in enumerate(SCHEMES):
        for w in (1, 5, 15):
            a = bayes_logq(w, donor=Dsc[sc])
            dlam = max(dlam, float(np.abs(a - lam_logq(w, d=Dsc[sc])).max()))
            dans += int((a[:, -1].argmax(-1) != Bs["ans"][k * WMAX + w - 1]).sum())
    sch["bayes_vs_lam"] = dlam
    sch["bayes_cache_vs_recomputed"] = dans
    vf = os.path.join(VERIFIER, "fresh")
    if os.path.isdir(vf):
        nd, nck, ntot, diff = 0, 0, 0, []
        for f in sorted(os.listdir(vf)):
            if not f.startswith("ck_"):
                continue
            ep = int(f[3:8])
            if not os.path.exists(scheme_path(ep)):
                continue
            z, m_ = np.load(os.path.join(vf, f)), np.load(scheme_path(ep))["ans"]
            names = [str(s) for s in z["schemes"]]
            for k, sc in enumerate(SCHEMES):
                if sc in names:
                    a_, b_ = z["ans"][names.index(sc)], m_[1 + k * WMAX:1 + (k + 1) * WMAX]
                    nd += int((a_ != b_).sum())
                    ntot += a_.size
                    diff += [(ep, sc, int(w_) + 1, int(i)) for w_, i in zip(*np.nonzero((a_ != b_).any(-1)))]
            nck += 1
        margins = []                                       # the route's top-two margin of each differing answer
        for ep in sorted(set(d_[0] for d_ in diff)):
            rows = [d_ for d_ in diff if d_[0] == ep]
            fo = net_cuts(ti.load("trained", ep), dev=dev, envs=np.unique(P["env"][[d_[3] for d_ in rows]]),
                          sources=[(s, Dsc[s]) for s in SCHEMES], keep=tuple(sorted({f"w{d_[2]}_{d_[1]}" for d_ in rows})))
            for _, sc, w, i in rows:
                lq = np.sort(fo["lq"][f"w{w}_{sc}"][int(np.searchsorted(fo["rows"], i)), -1].astype(np.float64), -1)
                margins.append(float((lq[:, -1] - lq[:, -2]).min()))
        zb2 = np.load(os.path.join(vf, "bayes.npz"))
        nb2 = sum(int((zb2["ans"][1 + k] != Bs["ans"][k * WMAX:(k + 1) * WMAX]).sum()) for k in range(len(SCHEMES)))
        sch["vs_verifier_route"] = dict(checkpoints=nck, answers_differ=nd, of=ntot, probes_differ=len(diff),
                                        max_top_two_margin=max(margins, default=0.), bayes_answers_differ=nb2)
    res["schemes"] = sch
    print("schemes:", sch, flush=True)
    # 8. the thinned observer
    Z = thinned_cache()
    Bsel = B["ans"][[CONDS.index(c) for c in THIN_CONDS]]                       # Bayes's answers, the same conditions
    ip1 = list(Z["p"]).index(1.)
    benv = _env_sums(Bsel)["acc"]                                              # (C, NE)
    ar = np.arange(n)
    rt1 = thinned_right(1., THIN_SEEDS[0])
    th = dict(p1_env_counts_differ=int(sum((Z["right"][ip1, j] != benv).sum() for j in range(len(THIN_SEEDS)))),
              p1_answers_right_differ=int(((Bsel[:, ar, P["goal"]] == P["r"][ar, P["goal"]]) != rt1).sum()),
              of=int(rt1.size))
    vt = os.path.join(VERIFIER, "thinned.npy")
    if os.path.exists(vt):
        ver = np.load(vt, allow_pickle=True).item()
        j7 = list(Z["seeds"]).index(7)
        dmax, nps = 0., 0
        for p, vv in ver.items():
            if p in list(Z["p"]):
                acc = Z["right"][list(Z["p"]).index(p), j7].sum(-1) / n
                mine_ = dict(zip(Z["conds"], acc))
                dmax = max(dmax, abs(mine_["intact"] - vv["intact"]),
                           max(abs(mine_[f"w{w}_{s}"] - vv[f"{s}_w{w}"]) for w in WS for s in ("random", "start")))
                nps += 1
        th["vs_verifier_seed7"] = dict(p_values=nps, max_abs_acc=dmax)
    res["thinned"] = th
    print("thinned:", th, flush=True)
    res["seconds"] = time.time() - t00
    with open(os.path.join(OUT, "checks.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"saved {os.path.join(OUT, 'checks.json')} ({res['seconds']:.0f} s)")
    return res



def tie_checks():
    """the answers that differ from the stored start-state truncation (checks()'s part 4): for every checkpoint of every
    run with a difference, the route is rerun and the top-two margin (nats) of each differing answer's variable is
    taken -> adds 'trunc_ties' to OUT/checks.json"""
    P = probes()
    dev = ti._device()
    cols = (("w1_start", 0), ("w2_start", 1), ("w5_start", 2), ("intact", 3))
    res, margins = [], []
    for run in ("trained", "echo") + REP:
        D = load_run(run)
        for ep, mt in zip(D["eps"], D["meta"]):
            if not sum(mt["checks"].values()):
                continue
            tr = np.load(os.path.join(ti.AN, run, f"ck_{int(ep):05d}.npz"))["trunc_right"]
            fo = net_cuts(ti.load(run, ep), dev=dev, keep=tuple(c for c, _ in cols))
            for c, j in cols:
                rt = fo["ans"][CONDS.index(c)] == P["r"]
                for i, v in zip(*np.nonzero(rt != tr[:, j])):
                    lq = np.sort(fo["lq"][c][i, -1, v].astype(np.float64))
                    margins.append(float(lq[-1] - lq[-2]))
                    res.append(dict(run=run, ep=int(ep), cond=c, probe=int(i), var=int(v), margin=margins[-1]))
    cf = os.path.join(OUT, "checks.json")
    c = json.load(open(cf))
    c["trunc_ties"] = dict(answers=len(res), max_top_two_margin=max(margins, default=0.), cases=res)
    with open(cf, "w") as f:
        json.dump(c, f, indent=1)
    print("truncation differences:", len(res), "answers; largest top-two margin", max(margins, default=0.))
    return res


# ═══════════════════════════════════════════════════════════════════
# Numbers (every number of the docstring)
# ═══════════════════════════════════════════════════════════════════

SRC_NAME = {"start": "start state", "same_cell": "same cell, other trajectory", "random": "random other cell"}
SCHEME_NAME = {"engine": "engine donors (one fixed other-cell probe, seed 20261201)",
               "fixed2": "another fixed seed (20261299)", "fresh": "a fresh donor at every window start (seed 777)"}


def _fmt_row(v, f="{:+.3f}"):
    return " ".join(f.format(x) for x in v)


def _runs(x, mask):
    """the checkpoints where mask holds, as runs of consecutive checkpoints ('280, 320-6,400'; '-' if none)"""
    k = np.flatnonzero(mask)
    if not len(k):
        return "-"
    g = np.flatnonzero(np.diff(k) > 1)
    return ", ".join(f"{x[a]:,.0f}" + ("" if a == b else f"-{x[b]:,.0f}") for a, b in zip(np.r_[k[0], k[g + 1]],
                                                                                   np.r_[k[g], k[-1]]))


def _trace(x, v):
    """a trace at every checkpoint as runs of equal value ('4 at 640, 6 at 800-880, ...')"""
    out, a = [], 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[a]:
            out.append(f"{v[a]} at {x[a]:,.0f}" + ("" if i - 1 == a else f"-{x[i - 1]:,.0f}"))
            a = i
    return ", ".join(out)


def _onset_block(name, o, marks, x_acc=None):
    """prints one set of onsets (w = 1..15), their order, trend and the relation to the accuracy marks"""
    print(f"  {name}: " + "; ".join(f"w{w + 1} " + _cx(o["p"][w], o["lo"][w], o["hi"][w], o["frac"][w])
                                   for w in range(len(o["p"]))))
    nb_, na_, npairs, res = pair_order(o)
    mind = min([b - a for a, b, rule, _ in res if rule == "before"], default=None)
    full = [dw for dw in range(1, len(o["p"])) if all(r_[2] == "before" for r_ in res if r_[1] - r_[0] == dw)]
    v_, b_ = onset_trend(o)
    print(f"    shorter before longer (intervals apart) {nb_} of {npairs} pairs, the other way {na_}; smallest resolved "
          f"gap {mind}; every pair resolved from a gap of {min(full) if full else 'never'}; per extra step "
          f"{_xstep(v_, b_)} later ({np.isfinite(b_).mean():.0%} of resamples)")
    print("    adjacent P(onset_w < onset_w+1): " + " ".join(f"{r_[3]:.2f}" for r_ in res if r_[1] - r_[0] == 1))
    if marks is not None:
        for k, lab in (("c0", "passing naive"), ("c50", "half-way")):
            m = marks[k]
            rr = [_order(o["boot"][w], m["boot"], o["lo"][w], o["hi"][w], m["lo"], m["hi"]) for w in range(len(o["p"]))]
            print(f"    vs {lab} {_cx(m['p'], m['lo'], m['hi'])}: " + "; ".join(
                f"w{w + 1} {r_[0]} ({r_[1]:.2f})" for w, r_ in enumerate(rr)))
    if x_acc is not None:
        xx, aa = x_acc
        print("    intact goal accuracy at the onsets: " + " ".join(
            f"{np.interp(np.log(v), np.log(xx), aa):.3f}" if np.isfinite(v) else "-" for v in o["p"]))


def numbers():
    """prints every number of the docstring (from the caches; ~3 min)"""
    D, E = load_run("trained"), load_run("echo")
    M = measures("trained")
    d, fa = loss("trained"), loss("trained", "far")
    x = d["x"]
    s_r, s_s, s_c = (SOURCES.index(k) for k in ("random", "start", "same_cell"))
    acc_m = accuracy_marks()
    print(f"Trained RNN: {len(x)} checkpoints ({x[0]:,.0f} ... {x[-1]:,.0f}); ESN: {len(E['x'])} ({', '.join(f'{v:,.0f}' for v in E['x'])})")
    for run in REP:
        Dr = load_run(run)
        print(f"{run}: {0 if Dr is None else len(Dr['x'])} checkpoints")
    sec = {run: np.array([m_["seconds"] for m_ in load_run(run)["meta"]]) for run in ("trained", "echo") + REP
           if load_run(run) is not None}
    print("GPU seconds per checkpoint (median, range, total): " + "; ".join(
        f"{run} {np.median(v):.1f} ({v.min():.1f}-{v.max():.1f}) {v.sum() / 60:.1f} min" for run, v in sec.items())
          + f"; all {sum(v.sum() for v in sec.values()) / 60:.0f} min over {sum(len(v) for v in sec.values())} files; "
          f"peak GPU memory {max(m_['peak'] for m_ in D['meta']):.0f} MB")
    ssec = [float(np.load(scheme_path(e))["seconds"]) for e in selection("trained") if os.path.exists(scheme_path(e))]
    if ssec:
        print(f"other donor schemes: {len(ssec)} checkpoints, {sum(ssec) / 60:.0f} GPU-min; intact answers differing from "
              "the main cache: " + str(sum(int(np.load(scheme_path(e))["intact_vs_main"])
                                           for e in selection("trained") if os.path.exists(scheme_path(e)))))
    cf = os.path.join(OUT, "checks.json")
    if os.path.exists(cf):
        c = json.load(open(cf))
        print("\nCHECKS (checks.json)")
        for ep, e in c["engine"].items():
            ks = [k for k in e if k.startswith("w")]
            print(f"  engine, raw {ep} (training {gap.train_x(int(ep)):,.0f}): max |d log q| "
                  f"{max(e[k]['max_abs'] for k in ks):.2g} over {len(ks)} cuts (intact {e['intact_max_abs']:.2g}); answers "
                  f"at T differ {sum(e[k]['ans_T_diff'] for k in ks)} of {len(ks) * 20000:,}, goal answers at every step "
                  f"{sum(e[k]['goal_steps_diff'] for k in ks)} of {len(ks) * 300000:,}; the cuts move log q by up to "
                  f"{max(e[k]['cut_vs_intact'] for k in ks):.1f}")
        for k in ("self_donor", "brute", "trunc", "trunc_ties", "memcuts", "bayes", "schemes", "thinned"):
            if k in c:
                print(f"  {k}: {c[k]}")
    # ── Bayes and levels
    print(f"\nLEVELS: Bayes intact {M['bayes_acc'][0]:.4f}, naive {M['naive_acc']:.4f}; Trained RNN intact at the stages "
          + ", ".join(f"{x[_k(x, v)]:,.0f} {M['acc'][_k(x, v), 0]:.3f}" for v in STAGES))
    lo, hi = _iv(d["bayes_b"])
    print("\nBAYES'S LOSS under the same cut, w = 1..15 [5-95%]")
    for s, src in enumerate(SOURCES):
        print(f"  {SRC_NAME[src]}: " + ", ".join(f"{d['bayes'][s, w]:.3f} [{lo[s, w]:.3f}, {hi[s, w]:.3f}]"
                                                  for w in range(WMAX)))
    print("  per-step benefit (random cell) L(k-1) - L(k), k = 2..15: " + _fmt_row(-np.diff(d["bayes"][s_r]), "{:.3f}"))
    # ── 1 random cell, engine donors
    print("\n1 RANDOM OTHER CELL (engine donors): L(w), w = 1..15, at the stages")
    for v in STAGES:
        k = _k(x, v)
        print(f"    {x[k]:>6,.0f} (acc {M['acc'][k, 0]:.3f}): " + _fmt_row(d["net"][k, s_r]))
    print("     Bayes: " + _fmt_row(d["bayes"][s_r]))
    nlo, nhi = _iv(d["net_b"][:, s_r])
    print("    intervals: " + " | ".join(
        f"{x[_k(x, v)]:,.0f} w1 [{nlo[_k(x, v), 0]:+.3f}, {nhi[_k(x, v), 0]:+.3f}] w15 [{nlo[_k(x, v), 14]:+.3f}, "
        f"{nhi[_k(x, v), 14]:+.3f}]" for v in (880, 1680, 39996)))
    print("    cut accuracy at w = 1 (chance .10) and intact through 120-880: " + ", ".join(
        f"{x[_k(x, v)]:,.0f} {M['acc'][_k(x, v), ci('random', 1)]:.3f}/{M['acc'][_k(x, v), 0]:.3f}"
        for v in (120, 320, 560, 800, 880, 1680, 39996)))
    print("  ONSET: after which L(w) stays above 1/2 of Bayes's L(w) under the same cut")
    o = onset_core(x, d["net"][:, s_r], d["net_b"][:, s_r], d["bayes"][s_r], d["bayes_b"][s_r], .5)
    _onset_block("engine donors", o, acc_m, (x, M["acc"][:, 0]))
    for share in (.25, .75, 1.):
        oo = onset_core(x, d["net"][:, s_r], d["net_b"][:, s_r], d["bayes"][s_r], d["bayes_b"][s_r], share)
        _onset_block(f"level {share:g} x Bayes's", oo, None)
    # ── 2 donor schemes
    print("\n2 DONOR SCHEMES (random cell; Bayes under the identical substitution)")
    for sc in SCHEMES:
        rl = random_loss(sc)
        if len(rl["x"]) < 10:
            print(f"  {sc}: only {len(rl['x'])} checkpoints")
            continue
        print(f"  {SCHEME_NAME[sc]} ({len(rl['x'])} checkpoints): Bayes L(w) " + _fmt_row(rl["bayes"], "{:.3f}"))
        for v in (880, 1680, 4000, 39996):
            k = _k(rl["x"], v)
            print(f"    {rl['x'][k]:>6,.0f}: " + _fmt_row(rl["net"][k]))
        os_ = onset_core(rl["x"], rl["net"], rl["net_b"], rl["bayes"], rl["bayes_b"], .5)
        _onset_block(sc, os_, acc_m, (x, M["acc"][:, 0]))
        rr = [_order(os_["boot"][w], o["boot"][w], os_["lo"][w], os_["hi"][w], o["lo"][w], o["hi"][w])
              for w in range(WMAX)]
        print("    vs the engine donors' onset: " + "; ".join(f"w{w + 1} {r_[0]} ({r_[1]:.2f})" for w, r_ in enumerate(rr)))
    # ── 3 accuracy-matched perfect-memory observer
    print("\n3 ACCURACY-MATCHED PERFECT-MEMORY OBSERVER (Bayes seeing each observation with probability p; seeds "
          f"{THIN_SEEDS} averaged)")
    if os.path.exists(os.path.join(OUT, "thinned.npz")):
        for src in ("random", "start"):
            th = thinned_levels(src)
            print(f"  {src}: p " + " ".join(f"{p:g}" for p in th["p"]) + "\n    intact accuracy " +
                  _fmt_row(th["acc"], "{:.3f}") + "\n    L(1) " + _fmt_row(th["L"][:, 0], "{:.3f}") + "\n    L(5) " +
                  _fmt_row(th["L"][:, 4], "{:.3f}") + "\n    L(15) " + _fmt_row(th["L"][:, 14], "{:.3f}"))
        Z = thinned_cache()
        sd = Z["right"][:, :, 0].sum(-1) / (NE * PER_ENV)
        print("  mask seeds' intact accuracies differ by at most " + f"{np.abs(sd[:, 0] - sd[:, 1]).max():.4f}")
        v, b = matched_threshold("random")
        lo_, hi_ = _iv(b)
        print("  the accuracy at which its L(w) reaches 1/2 of Bayes's (random cell): " + "; ".join(
            f"w{w + 1} {v[w]:.3f} [{lo_[w]:.3f}, {hi_[w]:.3f}]" for w in range(WMAX)))
        av, ab = acc_at("trained", o["p"], o["boot"])
        alo, ahi = _iv(ab)
        dlo, dhi = _iv(ab - b)
        print("  the network's accuracy at its onsets [5-95%], and minus the observer's (paired): " + "; ".join(
            f"w{w + 1} {av[w]:.3f} [{alo[w]:.3f}, {ahi[w]:.3f}] {av[w] - v[w]:+.3f} [{dlo[w]:+.3f}, {dhi[w]:+.3f}]"
            for w in range(WMAX)))
        sl = np.polyfit(np.arange(1, WMAX + 1), v, 1)[0]
        slb = np.array([np.polyfit(np.arange(1, WMAX + 1), b[:, i], 1)[0] if np.isfinite(b[:, i]).all() else np.nan
                        for i in range(NB)])
        sln = np.array([np.polyfit(np.arange(1, WMAX + 1), ab[:, i], 1)[0] if np.isfinite(ab[:, i]).all() else np.nan
                        for i in range(NB)])
        print(f"  slope of that accuracy on w: observer {sl:+.4f} [{np.nanpercentile(slb, 5):+.4f}, "
              f"{np.nanpercentile(slb, 95):+.4f}] per step; network "
              f"{np.polyfit(np.arange(1, WMAX + 1), av, 1)[0]:+.4f} [{np.nanpercentile(sln, 5):+.4f}, "
              f"{np.nanpercentile(sln, 95):+.4f}]")
        m = matched("trained", "random")
        om = onset_core(x, m["Lm"], m["Lm_b"], d["bayes"][s_r], d["bayes_b"][s_r], .5)
        _onset_block("matched observer on the network's accuracy trajectory", om, acc_m, (x, M["acc"][:, 0]))
        rr = [_order(o["boot"][w], om["boot"][w], o["lo"][w], o["hi"][w], om["lo"][w], om["hi"][w]) for w in range(WMAX)]
        print("    network's onset vs the matched observer's: " + "; ".join(
            f"w{w + 1} {r_[0]} ({r_[1]:.2f})" for w, r_ in enumerate(rr)))
        for src in ("random", "start"):
            m = matched("trained", src)
            lo_, hi_ = _iv(m["D_b"])
            print(f"  network's L(w) minus the matched observer's, {SRC_NAME[src]} (w = 1..15) [5-95% at w = 1, 2, 5, 10]")
            for v in (120, 320, 640, 880, 1200, 1680, 2400, 4000, 8000, 16000, 39996):
                k = _k(x, v)
                print(f"    {x[k]:>6,.0f} (acc {M['acc'][k, 0]:.3f}): " + _fmt_row(m["D"][k]) + "   " + " ".join(
                    f"[{lo_[k, w]:+.3f}, {hi_[k, w]:+.3f}]" for w in (0, 1, 4, 9)))
            pos = [(w, x[k]) for w in range(WMAX) for k in range(len(x)) if lo_[k, w] > 0]
            neg = [(w, x[k]) for w in range(WMAX) for k in range(len(x)) if hi_[k, w] < 0]
            for w in range(WMAX):
                kp = [xx for ww, xx in pos if ww == w]
                kn = [xx for ww, xx in neg if ww == w]
                print(f"    w{w + 1}: resolved above at {len(kp)} checkpoints" + (f" ({kp[0]:,.0f}-{kp[-1]:,.0f})" if kp else "")
                      + f", below at {len(kn)}" + (f" ({kn[0]:,.0f}-{kn[-1]:,.0f})" if kn else "")
                      + f"; extreme {m['D'][:, w].max():+.3f} at {x[np.argmax(m['D'][:, w])]:,.0f} / "
                      f"{m['D'][:, w].min():+.3f} at {x[np.argmin(m['D'][:, w])]:,.0f}; end {m['D'][-1, w]:+.3f} "
                      f"[{lo_[-1, w]:+.3f}, {hi_[-1, w]:+.3f}]")
    # ── 4 per-step benefit
    print("\n4 PER-STEP BENEFIT (random cell): mean over each band of L(k - 1) - L(k); paired difference from Bayes's")
    bd = bands("trained", "random") if os.path.exists(os.path.join(OUT, "thinned.npz")) else None
    if bd is not None:
        blo, bhi = _iv(bd["bayes_b"])
        print("  Bayes: " + "; ".join(f"steps {k0}-{k1} {bd['bayes'][j]:.4f} [{blo[j]:.4f}, {bhi[j]:.4f}]"
                                     for j, (k0, k1) in enumerate(BANDS)))
        dif, dif_b = bd["net"] - bd["bayes"], bd["net_b"] - bd["bayes_b"][None]
        nlo, nhi = _iv(bd["net_b"])
        dlo, dhi = _iv(dif_b)
        mlo, mhi = _iv(bd["net_b"] - bd["mat_b"])
        for v in (120, 320, 560, 880, 1200, 1680, 2400, 4000, 6000, 8000, 11200, 16000, 23200, 39996):
            k = _k(x, v)
            print(f"  {x[k]:>6,.0f}: " + "; ".join(
                f"{k0}-{k1} {bd['net'][k, j]:.4f} [{nlo[k, j]:.4f}, {nhi[k, j]:.4f}] d {dif[k, j]:+.4f} "
                f"[{dlo[k, j]:+.4f}, {dhi[k, j]:+.4f}] m {bd['net'][k, j] - bd['mat'][k, j]:+.4f} [{mlo[k, j]:+.4f}, "
                f"{mhi[k, j]:+.4f}]" for j, (k0, k1) in enumerate(BANDS)))
        for j, (k0, k1) in enumerate(BANDS):
            above = [x[k] for k in range(len(x)) if dlo[k, j] > 0]
            below = [x[k] for k in range(len(x)) if dhi[k, j] < 0]
            up = gap.crossing(x, dif[:, j], 0.)
            ub = gap.crossing(x, dif_b[:, j].T, 0.)
            dn = gap.crossing(x, dif[:, j], 0., sign=-1.)
            db_ = gap.crossing(x, dif_b[:, j].T, 0., sign=-1.)
            print(f"  steps {k0}-{k1}: resolved above Bayes's at {len(above)} checkpoints"
                  + (f" ({above[0]:,.0f}-{above[-1]:,.0f})" if above else "") + f", below at {len(below)}"
                  + (f" ({below[0]:,.0f}-{below[-1]:,.0f})" if below else "")
                  + f"; stays above from {_cx(up, *gap.interval(ub))}; stays below from {_cx(dn, *gap.interval(db_))}; "
                  f"largest {bd['net'][:, j].max():.4f} at {x[np.argmax(bd['net'][:, j])]:,.0f}")
            print(f"    at every checkpoint: resolved above Bayes's at {_runs(x, dlo[:, j] > 0)}; below at "
                  f"{_runs(x, dhi[:, j] < 0)}; above the observer at {_runs(x, mlo[:, j] > 0)}; below it at "
                  f"{_runs(x, mhi[:, j] < 0)}")
    # ── 5 start state
    print("\n5 START STATE: excess = the network's loss - Bayes's (the restart's extra cost) per w")
    print("  L(w) at the stages: " + " | ".join(f"{x[_k(x, v)]:,.0f} " + _fmt_row(d["net"][_k(x, v), s_s][[0, 1, 4, 9]])
                                             for v in (320, 880, 1680, 4000, 39996)) + "; Bayes "
          + _fmt_row(d["bayes"][s_s][[0, 1, 4, 9]]) + " (w = 1, 2, 5, 10)")
    tr = start_transient()
    for w in range(WMAX):
        print(f"  w{w + 1}: first > 0 at {tr['first_pos'][w]:,.0f}; stays > 0 from "
              f"{_cx(tr['pos'][w], tr['pos_lo'][w], tr['pos_hi'][w], tr['pos_share'][w])}; peak {tr['peak'][w]:+.3f} "
              f"[{tr['peak_lo'][w]:+.3f}, {tr['peak_hi'][w]:+.3f}] at {tr['peak_x'][w]:,.0f}; rise (half its peak) "
              f"{_cx(tr['rise'][w], tr['rise_lo'][w], tr['rise_hi'][w], tr['rise_share'][w])}; falls below half its peak "
              f"{_cx(tr['fall'][w], tr['fall_lo'][w], tr['fall_hi'][w], tr['fall_share'][w])}; below .05 "
              + (_cx(tr['below05'][w], tr['below05_lo'][w], tr['below05_hi'][w], tr['below05_share'][w])
                 if np.isfinite(tr['below05_share'][w]) else "-") + "; below .02 "
              + (_cx(tr['below02'][w], tr['below02_lo'][w], tr['below02_hi'][w], tr['below02_share'][w])
                 if np.isfinite(tr['below02_share'][w]) else "-")
              + f"; end {d['ex'][-1, s_s, w]:+.3f} [{np.percentile(d['ex_b'][-1, s_s, w], 5):+.3f}, "
              f"{np.percentile(d['ex_b'][-1, s_s, w], 95):+.3f}]")
    print("  w = 1, late checkpoints: " + ", ".join(f"{x[k]:,.0f} {d['ex'][k, s_s, 0]:+.3f}" for k in range(len(x) - 10, len(x))))
    ok = np.flatnonzero(tr["peak_lo"] >= .005)
    ob = [w for w in ok if np.isfinite(tr["below02_share"][w])]
    for nm, keyb in (("half its own peak", "fall"), (".02", "below02")):
        ws_ = ok if keyb == "fall" else ob
        bt = {}
        for w in ws_:
            y, yb = d["ex"][:, s_s, w], d["ex_b"][:, s_s, w].T
            if keyb == "fall":
                bt[w] = rise_fall(x, y, yb)["fall_boot"]
            else:
                k0 = int(np.nanargmax(_runmean3(y)))
                bt[w] = gap.crossing(x[k0:], yb[:, k0:], .02, sign=-1.)
        res = []
        for a_ in ws_:
            for b_ in ws_:
                if a_ < b_:
                    rule, p = _order(bt[b_], bt[a_], tr[keyb + "_lo"][b_], tr[keyb + "_hi"][b_], tr[keyb + "_lo"][a_],
                                     tr[keyb + "_hi"][a_])
                    res.append((a_ + 1, b_ + 1, rule, p))
        print(f"  falls below {nm} (w = {', '.join(str(w + 1) for w in ws_)}): longer window before shorter "
              f"{sum(r_[2] == 'before' for r_ in res)} of {len(res)} pairs, the other way "
              f"{sum(r_[2] == 'after' for r_ in res)}")
    c0 = acc_m["c0"]
    print("  stays > 0 vs passing naive: " + "; ".join(
        f"w{w + 1} {_order(gap.crossing(x, d['ex_b'][:, s_s, w].T, 0.), c0['boot'], tr['pos_lo'][w], tr['pos_hi'][w], c0['lo'], c0['hi'])[0]}"
        for w in range(5)))
    rb = {w: rise_fall(x, d["ex"][:, s_s, w], d["ex_b"][:, s_s, w].T)["rise_boot"] for w in ok}
    print("  rises (half peak) vs passing naive: " + "; ".join(
        f"w{w + 1} {_order(rb[w], c0['boot'], tr['rise_lo'][w], tr['rise_hi'][w], c0['lo'], c0['hi'])[0]}" for w in ok))
    sp, spb = span(d["ex"][:, s_s]), span(np.moveaxis(d["ex_b"][:, s_s], -1, 1))
    slo, shi = np.percentile(spb, [5, 95], axis=1)
    print("  span (largest w with excess > .02) [5-95%]: " + ", ".join(
        f"{x[_k(x, v)]:,.0f} {sp[_k(x, v)]} [{slo[_k(x, v)]:.0f}-{shi[_k(x, v)]:.0f}]"
        for v in (320, 480, 640, 800, 880, 1200, 1680, 2000, 2400, 3000, 4000, 6000, 8000, 11200, 16000, 23200, 39996)))
    print(f"  span: largest {sp.max()} at {x[np.argmax(sp)]:,.0f}; first >= 1 at {x[np.argmax(sp >= 1)]:,.0f}")
    print(f"  span at every checkpoint: {_trace(x, sp)}")
    if os.path.exists(os.path.join(OUT, "thinned.npz")):
        ms = matched("trained", "start")
        so, sob = span(ms["D"]), span(np.moveaxis(ms["D_b"], -1, 1))
        olo, ohi = np.percentile(sob, [5, 95], axis=1)
        print("  span against the observer (largest w with L - L_observer > .02) [5-95%]: " + ", ".join(
            f"{x[_k(x, v)]:,.0f} {so[_k(x, v)]} [{olo[_k(x, v)]:.0f}-{ohi[_k(x, v)]:.0f}]"
            for v in (20, 60, 120, 320, 480, 640, 800, 880, 1200, 1680, 2000, 2400, 3000, 4000, 6000, 8000, 11200, 16000,
                      23200, 39996)) + f"; largest {so.max()} at {x[np.argmax(so)]:,.0f}")
        print(f"  span against the observer at every checkpoint: {_trace(x, so)}")
    # ── 6 same cell
    print("\n6 SAME CELL, OTHER TRAJECTORY (null)")
    os_ = onset_core(x, d["net"][:, s_c], d["net_b"][:, s_c], d["bayes"][s_c], d["bayes_b"][s_c], .5)
    print(f"  never stays above half of Bayes's at any w: onsets {_fmt_row(os_['p'], '{:,.0f}')}; share of resamples "
          f"crossing at most {os_['frac'].max():.0%}; largest gain {-d['net'][:, s_c].min():.3f} at "
          f"{x[np.unravel_index(np.argmin(d['net'][:, s_c]), d['net'][:, s_c].shape)[0]]:,.0f} (w = "
          f"{np.unravel_index(np.argmin(d['net'][:, s_c]), d['net'][:, s_c].shape)[1] + 1}); largest loss "
          f"{d['net'][:, s_c].max():.3f}; Bayes {_fmt_row(d['bayes'][s_c], '{:.3f}')}")
    # ── 7 far errors
    print("\n7 FAR ERRORS ADDED by the cut (cut - intact) at w = 1 / 5 / 15")
    for s, src in ((s_r, "random"), (s_s, "start")):
        print(f"  {SRC_NAME[src]}: " + "; ".join(f"{x[_k(x, v)]:,.0f} {_fmt_row(fa['net'][_k(x, v), s][[0, 4, 14]], '{:.3f}')}"
                                                for v in (560, 1680, 39996)) + "; Bayes "
              + _fmt_row(fa["bayes"][s][[0, 4, 14]], "{:.3f}"))
        a1, a2 = d["net"][:, s].ravel(), fa["net"][:, s].ravel()
        print(f"    r(far added, accuracy lost) over checkpoints x windows {np.corrcoef(a1, a2)[0, 1]:.3f}")
    # ── 8 ESN
    de = loss("echo")
    Me = measures("echo")
    xe = de["x"]
    print(f"\n8 ESN ({len(xe)} checkpoints; intact accuracy {Me['acc'][0, 0]:.3f} -> {Me['acc'][-1, 0]:.3f})")
    for s, src in enumerate(SOURCES):
        sp_e = span(de["net"][:, s])
        print(f"  {SRC_NAME[src]}: max L over checkpoints " + _fmt_row(de["net"][:, s].max(0)[:6], "{:.3f}") +
              "; end " + _fmt_row(de["net"][-1, s][:6]) + " (w = 1..6); span (largest w with L > .02) "
              + ", ".join(f"{xe[k]:,.0f} {sp_e[k]}" for k in range(0, len(xe), 3)) + f", end {sp_e[-1]}")
        print(f"    span at every checkpoint: {_trace(xe, sp_e)}")
        for w in (2, 3, 4):
            first = xe[np.argmax(de["net"][:, s, w - 1] > .02)] if (de["net"][:, s, w - 1] > .02).any() else np.nan
            print(f"    w{w}: first > .02 at {first:,.0f}")
    oe = onset_core(xe, de["net"][:, s_r], de["net_b"][:, s_r], de["bayes"][s_r], de["bayes_b"][s_r], .5)
    print("  ESN random-cell onsets (half of Bayes's): " + "; ".join(
        f"w{w + 1} " + _cx(oe["p"][w], oe["lo"][w], oe["hi"][w], oe["frac"][w]) for w in range(4)))
    ae, aeb = acc_at("echo", oe["p"][:1], oe["boot"][:1])
    print(f"  ESN intact accuracy at its w = 1 onset {ae[0]:.3f} [{np.nanpercentile(aeb[0], 5):.3f}, "
          f"{np.nanpercentile(aeb[0], 95):.3f}]")
    print("  ESN random w = 1 through training: " + ", ".join(f"{v:,.0f} {L:.3f}" for v, L in zip(xe, de["net"][:, s_r, 0])))
    # ── 9 replication
    print("\n9 REPLICATION (random cell, engine donors; each run's own passing naive from td_tt_replication)")
    mk = run_marks()
    base, base_acc = None, None
    for run in ("trained",) + REP:
        if load_run(run) is None:
            continue
        dr = loss(run)
        Mr = measures(run)
        xr = dr["x"]
        chk = [m["checks"] for m in load_run(run)["meta"]]
        print(f"  {NAME.get(run, run)} ({run}; {len(xr)} checkpoints; end accuracy {Mr['acc'][-1, 0]:.3f}; stored "
              f"truncation differs {sum(sum(c.values()) for c in chk)} of {len(chk) * 4 * 20000:,} answers)")
        orr = onset_core(xr, dr["net"][:, s_r], dr["net_b"][:, s_r], dr["bayes"][s_r], dr["bayes_b"][s_r], .5)
        _onset_block(run, orr, mk[run], (xr, Mr["acc"][:, 0]))
        av, ab = acc_at(run, orr["p"], orr["boot"])
        alo, ahi = _iv(ab)
        if run == "trained":
            base_acc = (av, ab)
        print("    accuracy at the onsets [5-95%]" + ("" if run == "trained" else ", minus N1000's (paired)") + ": "
              + "; ".join(f"w{w + 1} {av[w]:.3f} [{alo[w]:.3f}, {ahi[w]:.3f}]" + (
                  "" if run == "trained" else f" {av[w] - base_acc[0][w]:+.3f} [{np.nanpercentile(ab[w] - base_acc[1][w], 5):+.3f}, "
                  f"{np.nanpercentile(ab[w] - base_acc[1][w], 95):+.3f}]") for w in range(WMAX) if np.isfinite(av[w])))
        if run != "trained":
            ea, eab = Mr["acc"][-1, 0], Mr["acc_b"][-1, 0]
            miss = [w for w in range(WMAX) if not np.isfinite(orr["p"][w]) and np.isfinite(base_acc[0][w])]
            dif_ = {w: eab - base_acc[1][w] for w in miss}
            print(f"    end accuracy (last checkpoint) {ea:.3f} [{np.percentile(eab, 5):.3f}, "
                  f"{np.percentile(eab, 95):.3f}]" + (
                      "; minus N1000's accuracy at its onset (paired), at the windows this run does not reach: "
                      + "; ".join(f"w{w + 1} {ea - base_acc[0][w]:+.4f} [{np.nanpercentile(dif_[w], 5):+.4f}, "
                                  f"{np.nanpercentile(dif_[w], 95):+.4f}]" for w in miss) if miss else ""))
        v_, lo_, hi_, rb_ = own_clock(orr, mk[run]["c0"])
        print("    own clock (onset / passing naive): " + "; ".join(
            f"w{w + 1} {v_[w]:.2f} [{lo_[w]:.2f}, {hi_[w] if np.isfinite(hi_[w]) else np.inf:.2f}]" for w in range(WMAX)))
        if run == "trained":
            base = (orr, rb_)
        else:
            ok_ = np.isfinite(mk[run]["c0"]["boot"]) & np.isfinite(mk["trained"]["c0"]["boot"])
            ra = np.where(np.isnan(orr["boot"]), np.inf, orr["boot"])[:, ok_] / mk[run]["c0"]["boot"][ok_]
            rb0 = np.where(np.isnan(base[0]["boot"]), np.inf, base[0]["boot"])[:, ok_] / mk["trained"]["c0"]["boot"][ok_]
            with np.errstate(invalid="ignore"):
                rr = ra / rb0
            print("    own-clock ratio over N1000's, paired: " + "; ".join(
                f"w{w + 1} {v_[w] / (base[0]['p'][w] / mk['trained']['c0']['p']):.2f} "
                f"[{np.nanpercentile(rr[w], 5):.2f}, {np.nanpercentile(rr[w], 95):.2f}]" for w in range(WMAX)
                if np.isfinite(orr["p"][w])))



# ═══════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════

XMIN, XMAX = 8, 7e4
NAIVE_X = 835                                             # the Trained RNN's accuracy passes naive's (td_tt_gap c0)
RUN_STYLE = {"trained": dict(color="#1f77b4", marker="s", ls="-", mfc="#1f77b4", label="N = 1000"),
             "trained_p1": dict(color="#1f77b4", marker="s", ls="--", mfc="w", label="N = 1000, power 1"),
             "trained_N200": dict(color="#4f97d4", marker="o", ls="-", mfc="#4f97d4", label="N = 200"),
             "trained_N100": dict(color="#86b6e0", marker="^", ls="-", mfc="#86b6e0", label="N = 100"),
             "trained_N50": dict(color="#b3d0ec", marker="D", ls="-", mfc="#b3d0ec", label="N = 50")}


def _edges(x):
    lx = np.log(x)
    mid = (lx[1:] + lx[:-1]) / 2
    return np.exp(np.concatenate([[lx[0] - (mid[0] - lx[0])], mid, [lx[-1] + (lx[-1] - mid[-1])]]))


def _xaxis(ax, label=True, lim=(XMIN, XMAX)):
    ax.set_xscale("log")
    ax.set_xlim(*lim)
    ticks = [v for v in (gap.X0, 1e2, 1e3, 1e4) if lim[0] <= v <= lim[1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0" if v == gap.X0 else f"{v / 1e3:g}k" if v >= 1e3 else f"{v:g}" for v in ticks])
    ax.minorticks_off()
    if label:
        ax.set_xlabel("training episodes")


def _hbar(ax, lo, hi, y, color, lw=1.2, xmax=None, zorder=3):
    """a horizontal 5-95% bar; an open upper bound runs to xmax"""
    if not np.isfinite(lo):
        return
    hi_ = hi if np.isfinite(hi) else xmax
    ax.plot([lo, hi_], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=zorder)


def figure():
    """-> figure_bin/td_tt_memlen.png. Random other cell: (a) the accuracy lost L(w) over training x window, with its
    onsets; (b) the onsets under three donor schemes, with the accuracy-matched perfect-memory observer and the ESN;
    (c) the benefit per memory step in bands, against the observer's (dotted: its 2nd step and steps 10-15) and Bayes's
    (the strip at the right, per band [5-95%]). Replication: (d) the width / loss runs' onsets on
    each run's own clock; (e) the same onsets against each run's own accuracy. (f) A restart from the start state: the
    network's loss minus the observer's, with the span of windows where it exceeds .02"""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.legend_handler import HandlerTuple
    from main.utils import fig_path, apply_fig_style
    warnings.simplefilter("ignore", RuntimeWarning)
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    TF, FL, FK, FG = fs["label"] + 1.5, fs["label"], fs["tick"], fs["legend"] - 1
    blue, orange = COL["trained"], COL["echo"]
    d = loss("trained")
    x = d["x"]
    s_r = SOURCES.index("random")
    o = onset_core(x, d["net"][:, s_r], d["net_b"][:, s_r], d["bayes"][s_r], d["bayes_b"][s_r], .5)
    mr, ms = matched("trained", "random"), matched("trained", "start")
    om = onset_core(x, mr["Lm"], mr["Lm_b"], d["bayes"][s_r], d["bayes_b"][s_r], .5)
    de = loss("echo")
    oe = onset_core(de["x"], de["net"][:, s_r], de["net_b"][:, s_r], de["bayes"][s_r], de["bayes_b"][s_r], .5)
    mk = run_marks()
    M = measures("trained")
    wv = np.arange(1, WMAX + 1)
    xe = _edges(x)
    ye = np.arange(.5, WMAX + 1)
    PW, PH, GY = 3.2, 2.3, 1.2
    L0, TOP, BOT = .72, 2.45, 1.65
    CX = (L0, L0 + PW + 1.3, L0 + 2 * (PW + 1.3))
    W = CX[2] + PW + .95
    H = TOP + 2 * PH + GY + BOT
    fig = plt.figure(figsize=(W, H))

    def place(r, j, w=PW, dx=0.):
        y = H - TOP - r * (PH + GY) - PH
        return fig.add_axes([(CX[j] + dx) / W, y / H, w / W, PH / H])

    def title(ax, letter, name, math):
        ax.set_title(f"{letter}   {name}\n{math}", loc="left", fontsize=TF, linespacing=1.4)

    def wax(ax, top=WMAX + .7):
        ax.set_ylim(.3, top)
        ax.set_yticks([1, 5, 10, 15])
        ax.set_ylabel("window $w$ (steps kept)")

    def naive_line(ax, color="k", label=True, xv=NAIVE_X, y=WMAX + .45, text="passes\nnaive", top=None, side="right"):
        """the dotted passing-naive line (up to `top` in data units), labelled on one side"""
        ylo, yhi = ax.get_ylim()
        ax.plot([xv, xv], [ylo, yhi if top is None else top], color=color, lw=1., ls=(0, (1, 1.4)), zorder=5)
        ax.set_ylim(ylo, yhi)
        if label:
            lg = ax.get_xscale() == "log"
            xt = (xv * 1.07 if lg else xv + .006) if side == "right" else (xv / 1.07 if lg else xv - .006)
            ax.text(xt, y, text, ha="left" if side == "right" else "right", va="top", fontsize=FG - .5, color=color,
                    linespacing=1.05)

    def heat(ax, Z, vmin, vmax, cblabel, cbx):
        im = ax.pcolormesh(xe, ye, Z, cmap="viridis", vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
        ax.set_ylim(.5, WMAX + .5)
        ax.set_yticks([1, 5, 10, 15])
        ax.set_ylabel("window $w$ (steps kept)")
        _xaxis(ax, lim=(XMIN, float(xe[-1])))
        pos = ax.get_position()
        cax = fig.add_axes([cbx / W, pos.y0, .1 / W, pos.height])
        cb = fig.colorbar(im, cax=cax, extend="both" if vmin < 0 else "max")
        cb.ax.tick_params(labelsize=FK)
        cb.set_label(cblabel, fontsize=FL)
        return im
    # a: the random-cell loss, with its onsets
    ax = place(0, 0, w=PW - .32)
    heat(ax, d["net"][:, s_r].T, 0., .5, "goal accuracy at $T$ lost", CX[0] + PW - .32 + .28)
    axs = place(0, 0, w=.17, dx=PW - .32 + .05)
    axs.pcolormesh([0, 1], ye, d["bayes"][s_r][:, None], cmap="viridis", vmin=0, vmax=.5, shading="flat", rasterized=True)
    axs.set_ylim(.5, WMAX + .5)
    axs.set_yticks([])
    axs.set_xticks([.5])
    axs.set_xticklabels(["Bayes"], rotation=90, fontsize=FK)
    ok = np.isfinite(o["p"])
    ax.plot(o["p"][ok], wv[ok], color="w", lw=1.3, zorder=6, path_effects=[pe.withStroke(linewidth=2.8, foreground="k")])
    ax.plot(o["p"][ok], wv[ok], ls="none", marker="s", ms=4.2, mfc="w", mec="k", mew=.6, zorder=7)
    naive_line(ax, "w", label=False)
    title(ax, "a", "another cell's state replaces the memory older than $w$",
          r"$L(w) = \mathrm{acc}_T(\mathrm{intact}) - \mathrm{acc}_T(\mathrm{cut\ to}\ w)$;  squares: onsets (b)")
    # b: the onsets under three donor schemes, the matched observer and the ESN
    ax = place(0, 1)
    XB = (250, 6e4)
    sch = [("engine", "one other cell (as in a)", dict(marker="s", mfc=blue, ls="-"), 0.),
           ("fixed2", "another draw of cells", dict(marker="o", mfc="w", ls="--"), .25),
           ("fresh", "a new cell at every window", dict(marker="^", mfc="w", ls=":"), -.25)]
    for nm, lab, st, off in sch:
        rl = random_loss(nm)
        oo = onset_core(rl["x"], rl["net"], rl["net_b"], rl["bayes"], rl["bayes_b"], .5)
        for w in range(WMAX):
            _hbar(ax, oo["lo"][w], oo["hi"][w], w + 1 + off, blue, lw=1., xmax=XB[1])
        okk = np.isfinite(oo["p"])
        ax.plot(oo["p"][okk], wv[okk] + off, color=blue, lw=1.1, ls=st["ls"], marker=st["marker"], ms=4.3, mfc=st["mfc"],
                mec=blue, mew=1., label=lab, zorder=4)
    for w in range(WMAX):
        _hbar(ax, om["lo"][w], om["hi"][w], w + 1, "k", lw=1., xmax=XB[1])
    okk = np.isfinite(om["p"])
    ax.plot(om["p"][okk], wv[okk], color="k", lw=1., ls="--", marker="D", ms=4, mfc="w", mec="k", mew=1.,
            label="perfect-memory\nobserver of the\nsame accuracy", zorder=4)
    if np.isfinite(oe["p"][0]):
        _hbar(ax, oe["lo"][0], oe["hi"][0], 1, orange, lw=1., xmax=XB[1])
        ax.plot([oe["p"][0]], [1], ls="none", marker="o", ms=5.5, mfc=orange, mec="k", mew=.5,
                label="ESN (only $w = 1$)", zorder=5)
    _xaxis(ax, lim=XB)
    wax(ax, top=22.1)
    naive_line(ax, "k", y=9.2, top=15.5, side="left")
    ax.legend(fontsize=FG - .5, frameon=False, loc="upper left", ncol=2, handlelength=2., borderaxespad=.25,
              labelspacing=.3, columnspacing=.8)
    title(ax, "b", "onsets: shorter windows first, under every donor",
          r"$L(w)$ stays $> \frac{1}{2}\,L_{\mathrm{Bayes}}(w)$ from then on [5-95%]")
    # c: the benefit per memory step, in bands, against the observer's (dotted) and Bayes's (the strip at the right)
    ax = place(0, 2)
    bd = bands("trained", "random")
    bcol = ("#6baed6", "#3182bd", "#1f5f9e", "#08306b")                 # light = recent, dark = old
    for j, (k0, k1) in enumerate(BANDS):
        lo_, hi_ = _iv(bd["net_b"][:, j])
        ax.fill_between(x, lo_, hi_, color=bcol[j], alpha=.25, lw=0, zorder=3)
        ax.plot(x, bd["net"][:, j], color=bcol[j], lw=1.8, zorder=4,
                label="the 2nd step back" if k0 == k1 else f"steps {k0}-{k1} back")
    obs = []
    for j in (0, len(BANDS) - 1):            # the observer's 2nd step and steps 10-15 (its 3-6, 7-9 run with its 2nd)
        obs += ax.plot(x, bd["mat"][:, j], color=bcol[j], lw=1.4, ls=(0, (1., 1.2)), zorder=5)
    _xaxis(ax)
    ax.set_ylim(-.004, .15)
    naive_line(ax, "k", label=False)
    ax.set_ylabel("accuracy gained per step kept")
    hd, lb = ax.get_legend_handles_labels()
    ax.legend(hd + [tuple(obs)], lb + ["perfect-memory\nobserver (dotted)"], fontsize=FG - .5,
              frameon=False, loc="upper left", handlelength=1.6, borderaxespad=.2, labelspacing=.35,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=.3)})
    title(ax, "c", "what each step of own history is worth (as in a)",
          r"$L(k-1) - L(k)$, mean over the steps $k$ of each band")
    axs = place(0, 2, w=.42, dx=PW + .06)                              # Bayes's per-step benefit per band [5-95%]
    blo, bhi = _iv(bd["bayes_b"])
    for j in range(len(BANDS)):
        axs.plot([j + .5, j + .5], [blo[j], bhi[j]], color=bcol[j], lw=1.3, solid_capstyle="butt")
        axs.plot([j + .5], [bd["bayes"][j]], ls="none", marker="o", ms=4.2, mew=0, color=bcol[j])
    axs.set_xlim(-.1, len(BANDS) + .1)
    axs.set_ylim(ax.get_ylim())
    axs.set_yticks([])
    axs.set_xticks([len(BANDS) / 2])
    axs.set_xticklabels(["Bayes"], rotation=90, fontsize=FK)
    # d / e: the replication runs on their own clock and against their own accuracy
    axd, axe = place(1, 0), place(1, 1)
    th_v, th_b = matched_threshold("random")
    for run in ("trained",) + REP:
        dr = loss(run)
        orr = onset_core(dr["x"], dr["net"][:, s_r], dr["net_b"][:, s_r], dr["bayes"][s_r], dr["bayes_b"][s_r], .5)
        st = RUN_STYLE[run]
        v_, lo_, hi_, _ = own_clock(orr, mk[run]["c0"])
        av, ab = acc_at(run, orr["p"], orr["boot"])
        alo, ahi = _iv(ab)
        okk = np.isfinite(orr["p"])
        off = {"trained": 0., "trained_p1": .22, "trained_N200": -.22, "trained_N100": .11, "trained_N50": -.11}[run]
        for w in np.flatnonzero(okk):
            _hbar(axd, lo_[w], hi_[w], w + 1 + off, st["color"], lw=1., xmax=100.)
            _hbar(axe, alo[w], ahi[w], w + 1 + off, st["color"], lw=1., xmax=.79)
        for ax_, xv in ((axd, v_), (axe, av)):
            ax_.plot(xv[okk], wv[okk] + off, color=st["color"], lw=1.2, ls=st["ls"], marker=st["marker"], ms=4.5,
                     mfc=st["mfc"], mec="k" if st["mfc"] != "w" else st["color"], mew=.6 if st["mfc"] != "w" else 1.,
                     label=st["label"], zorder=4)
    axd.set_xscale("log")
    axd.set_xlim(.2, 100)
    axd.set_xticks([.3, 1, 3, 10, 30, 100])
    axd.set_xticklabels(["0.3", "1", "3", "10", "30", "100"])
    axd.minorticks_off()
    wax(axd, top=17.)
    naive_line(axd, "k", xv=1., y=16.85)
    axd.set_xlabel("onset / the run's passing naive")
    axd.legend(fontsize=FG - .5, frameon=False, loc="upper left", handlelength=1.5, borderaxespad=.2, labelspacing=.3,
               bbox_to_anchor=(.57, .995))
    title(axd, "d", "width / loss runs: onsets on each run's own clock",
          r"$t_{\mathrm{onset}}(w)\, /\, t_{\mathrm{passes\ naive}}$ [5-95%]")
    tlo, thi = _iv(th_b)
    okk = np.isfinite(th_v)
    for w in np.flatnonzero(okk):
        _hbar(axe, tlo[w], thi[w], w + 1, "k", lw=1.)
    axe.plot(th_v[okk], wv[okk], color="k", lw=1., ls="--", marker="D", ms=4, mfc="w", mec="k", mew=1.,
             label="perfect-memory\nobserver", zorder=4)
    if np.isfinite(oe["p"][0]):
        ae, aeb = acc_at("echo", oe["p"][:1], oe["boot"][:1])
        elo, ehi = _iv(aeb)
        _hbar(axe, elo[0], ehi[0], .62, orange, lw=1.)
        axe.plot(ae, [.62], ls="none", marker="o", ms=5.5, mfc=orange, mec="k", mew=.5, label="ESN", zorder=5)
    axe.set_xlim(.3, .79)
    wax(axe, top=17.)
    axe.axvline(M["bayes_acc"][0], color="k", lw=1.1)
    axe.text(M["bayes_acc"][0] - .006, 16.85, "Bayes", ha="right", va="top", fontsize=FG - .5)
    naive_line(axe, "0.45", xv=M["naive_acc"], y=16.85, text="naive")
    axe.set_xlabel("goal accuracy at $T$ (intact) at the onset")
    hd, lb = axe.get_legend_handles_labels()
    keep_ = [i for i, l_ in enumerate(lb) if l_ in ("perfect-memory\nobserver", "ESN")]
    axe.legend([hd[i] for i in keep_], [lb[i] for i in keep_], fontsize=FG - .5, frameon=False, loc="lower right",
               handlelength=1.5, borderaxespad=.2, labelspacing=.25, borderpad=.25,
               bbox_to_anchor=((M["bayes_acc"][0] - .3) / (.79 - .3) - .006, 0.))      # left of the Bayes line
    title(axe, "e", "the same onsets against each run's accuracy",
          r"$\mathrm{acc}_T(\mathrm{intact})$ at $t_{\mathrm{onset}}(w)$ [5-95%]")
    # f: a restart from the start state, beyond the observer
    ax = place(1, 2, w=PW - .32)
    heat(ax, ms["D"].T, -.05, .3, "lost beyond the observer", CX[2] + PW - .32 + .1)
    sp = span(ms["D"])
    ax.step(np.r_[xe[:-1], xe[-1]], np.r_[sp, sp[-1]] + .5, where="post", color="w", lw=1.3, zorder=6,
            path_effects=[pe.withStroke(linewidth=2.8, foreground="k")])
    naive_line(ax, "w", label=False)
    title(ax, "f", "a restart from the start state, beyond the observer",
          r"$L(w) - L_{\mathrm{observer}}(w)$;  line: largest $w$ with $> .02$")
    nE = len(de["x"])
    fig.text(.5, .68 / H,
             f"Seed 20261029; probes: 50 test environments x 200 trajectories (100 latent pairs x 2), 30 steps; {len(x)} "
             f"checkpoints of each Trained RNN run (N = 1000; N = 1000 with loss power 1; N = 200, 100, 50), {nE} of the "
             "ESN.\nCut to $w$: at every step $t > w$ the cell is rerun "
             "over the probe's own last $w$ inputs from a state placed at $t - w$; the rerun's read-out replaces the "
             "network's increment at $t$. Bayes: exact, under the identical substitution.\nPerfect-memory observer: Bayes "
             "that sees each observation with probability $p$, with $p$ set so that its intact accuracy equals the "
             "network's at each checkpoint (e: the accuracy at its own onset).\nDotted vertical line: passing naive (a-c, f: "
             "the N = 1000 network, 835 training episodes). c: coloured dotted = the observer's 2nd step and steps 10-15; "
             "strip = Bayes's per band.\nIntervals and bands: 5-95% over 1,000 resamples of the environments.",
             ha="center", va="center", fontsize=FG, linespacing=1.55)
    fig.suptitle(SUPTITLE, y=1 - .5 / H, fontsize=fs["title"] - 3, linespacing=1.45)
    path = fig_path("td_tt_memlen.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


SUPTITLE = ("Through training the Trained RNN's answers come to depend on its state from ever further back: another cell's state "
            "older than $w$ steps costs it half of Bayes's loss\nfirst for $w$ = 1, then later for longer windows (about 15% "
            "per step) under every donor scheme (a, b); a perfect-memory observer of its accuracy shows no such order (b).\n"
            "From about 300 to 6k its recent past is worth more, and its older past less, than to Bayes, by the end slightly "
            "the reverse (c); a restart costs it more than the observer (f).\nWidth / loss runs: the same order (N = 50 "
            "reaches only $w$ = 1); the windows a run reaches come at about N = 1000's accuracy, not at a fixed multiple of "
            "passing naive,\nbut the narrower runs end one or two windows short of what N = 1000 had passed at the same "
            "accuracy (d, e).")


# ═══════════════════════════════════════════════════════════════════
# Command line
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "numbers"
    if cmd == "bayes":
        bayes_cache(force="force" in sys.argv)
        bayes_schemes(force="force" in sys.argv)
    elif cmd == "thinned":
        thinned_cache(force="force" in sys.argv)
    elif cmd == "harvest":
        bayes_cache()
        harvest(sys.argv[2] if len(sys.argv) > 2 else "1")
    elif cmd == "worker":
        worker()
    elif cmd == "one":
        run_ck(sys.argv[2], int(sys.argv[3]), force=True)
    elif cmd == "checks":
        checks()
        tie_checks()
    elif cmd == "ties":
        tie_checks()
    elif cmd == "numbers":
        warnings.simplefilter("ignore", RuntimeWarning)
        numbers()
    elif cmd == "figure":
        figure()
    else:
        raise SystemExit(f"unknown command {cmd}")
