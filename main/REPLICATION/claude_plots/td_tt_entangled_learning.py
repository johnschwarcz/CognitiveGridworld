"""Does the network LEARN on entangled steps? Does the realized weight change between checkpoints draw on the steps of
each entanglement fifth more than their share of the loss, beyond what naive's difficulty there predicts, and when?
(workflow topic "entangled_learning"; CPU only, no training; revised 11-01 after an independent verification: a
difficulty-matched control, all 114 checkpoints, SEs of the plotted difference, timing intervals, timeline export;
revised 11-02 after a second verification: results under every matching, the early oscillation, absolute sizes, figure
labels, synthesis export; corrected 11-02 after a third: the oscillation's example windows, the 9,600-15,200 decrease, a
synthesis note per row, centred-window ties)

ANSWER (seed-20261029 runs, 3,000 training-distribution episodes, 0-40k training episodes; numbers below):
  The entanglement fifths are confounded with naive's difficulty: naive's loss per item and value rises 2.1x from the
  least to the most entangled fifth (0.117 -> 0.246; within-step Spearman .30). So every window is shown raw (b) and
  for entanglement fifths matched on naive's per-item loss (c). The excess (realized share - loss share) is a fraction
  of the window's learning: excess = B / (A - B) x (the fifth's loss share at the start - at the end), with B / (A - B)
  the loss left at the window's end over its decrease (per window 2.1-41 for the trained RNN, 10-70 for the ESN). Its
  size therefore compares across windows and networks only as a fraction of each one's learning; every window's total
  decrease is given with it (strips under b and c; FINDINGS), and loss units wherever the networks are compared.
  ESN: learns less where naive's marginal is wrong. Raw, the most entangled fifth gets less of the loss decrease than
  its share of the loss in every window from 40 to 15,200 training episodes (-.04 to -.15 of the ESN's learning;
  pooled 80-40k -.048 +- .003, i.e. that fifth's loss fell .046 per item and value less than proportional learning
  would give) and its loss fans out like naive's (q5 / q1 1.74 at the end, naive 2.11). At matched naive difficulty
  there is no entanglement-specific deficit from ~320 on (320-1,600 +.007 +- .005; pooled 80-40k +.002 +- .002), only
  early: a steady deficit in every window that starts before 180 (40-80 -.078 +- .023, 80-160 -.066 +- .011, 160-260
  -.031 +- .009, 160-320 -.018 +- .008), gone at 180-320 (-.003). A small surplus at 1,600-4,000 (+.026 +- .007;
  probes +.033 +- .009) holds only when matched on naive's loss (5 / 20 strata +.022 / +.027, FR +.030): it is absent
  at matched naive's loss x Bayes's entropy (-.007 +- .007). Matching on Bayes's entropy leaves the whole deficit
  (80-40k -.057 +- .003): the ESN's shortfall follows naive's error, not Bayes's uncertainty. Matching on naive's loss
  x Bayes's entropy leaves a small late deficit (4,000-15,200 -.020 +- .007), so "no specific deficit" depends on the
  matching variable.
  Trained RNN, early: the raw 40-80 deficit (-.115 +- .010) is partly entanglement-specific (matched -.076 +- .011). The
  ESN's matched dip there is a similar fraction of its learning (-.078 +- .023; less stable: M -.105, G -.033, probes
  -.040 +- .021) but 3.2x smaller in loss units (q5's decrease short of proportional by .0039 vs .0126 per item and
  value; the ESN's loss falls 3.3x less in that window). After the dip the trained RNN's matched excess OSCILLATES at
  the scale of the checkpoints (20-40 training episodes): a surplus in windows ending at 160-180 (80-160 +.066 +- .011,
  80-180 +.102 +- .014) and a second deficit in windows from 160-180 to 260-320 (160-260 -.031 +- .008, 180-320 -.023 +-
  .007); its size changes several-fold when one edge moves by one checkpoint (80-140 +.001, 80-160 +.066, 80-180 +.102,
  80-200 +.022; probes +.027, +.068, +.101, +.032), and its sign can flip (140-280 +.020 +- .007 vs 160-280 -.025 +-
  .009; probes +.012 / -.028). The probes replicate it (80-180 +.101 +- .017, 160-260 -.047 +- .014, 180-320 -.033 +-
  .010), so it belongs to the training trajectory, not to the sample, but no single window is a timed event (the first
  version's "repaid at 80-160" is withdrawn). The robust early contrast is pooled: at matched difficulty the trained
  RNN's learning is proportional (0-320 .000 +- .003; 40-160 -.005 +- .006) while the ESN's falls short (-.015 +- .004;
  -.069 +- .013), as a fraction of each network's learning and in loss units (40-160: -.0015 vs -.0097 per q5 item and
  value).
  Trained RNN, later: a small extra draw on entangled steps at 320-1,040 when matched on naive's loss (+.011 +- .002;
  5 / 20 strata +.011 / +.010, FR +.014; probes +.006 +- .003) that is absent at matched naive's loss x Bayes's
  entropy (+.001 +- .001) and reverses at matched Bayes's entropy (-.006 +- .002); and one at 4,000-15,200, the only
  stretch robust to every matching (+.011 to +.022; naive's loss +.013 +- .002, probes +.015 +- .002). None at
  1,040-4,000 (1,600-4,000 -.003 +- .002) or after 15,200 (-.000 +- .002). The late raw surplus (1,600-9,600 +.008 +-
  .001) is mostly a naive-difficulty effect (matched +.002 +- .001): 1,600-4,000 vanishes under matching (+.006 ->
  -.003), 4,000-15,200 keeps a bit more than half (+.023 -> +.013). Passing naive (800-880) and the memory-use flip
  (800-1,040) fall inside the 320-1,040 stretch, which is present already at 320-640 (+.012 +- .005; M +.012, G +.011;
  the probes do not confirm this window, +.005 +- .006) and gone at 1,040-1,600, so it is not specific to those
  events; its end cannot be ordered against them at this resolution.
  Loss against naive's level per fifth: the trained RNN's loss drops below naive's on the most entangled fifth at 493
  [451, 513] training episodes, before it passes naive's accuracy (800-880), and on the least entangled fifth after
  it, at 957 [924, 1,010]; below the naive-Bayes midpoint at 1,220 [1,167, 1,285] and 2,202 [2,085, 2,368]. The most
  entangled fifth crosses first because naive's loss is 2.1x higher there, not because the network's own loss is lower
  on it (q5 / q1 1.07, 1.22, 1.30, 1.27, 1.26 at 0, 80, 320, 880, 2,400). The ESN: 8,984 [7,213, 10,811] and never;
  never below the midpoint.
  Against the GRADIENT the picture is different and not specific: in both networks the gradient comes
  disproportionately from the LEAST entangled steps, yet the realized learning lands on the most entangled fifth
  2.5-11 points above its gradient share (trained; ESN 4-8 where measurable). The first-order check fails because the
  loss is curved along the trained RNN's checkpoint steps, which end past the minimum along their own direction (end
  slope > 0 in 7 of 7 profiled pairs, a visible valley at 80 and from 1,600 on; median actual / first-order .23).

WHY A NEW TEST. td_entanglement_driver.py ("gdn") compared the first-order prediction grad L(W_k) . dW with the actual
loss change between checkpoints and was ~5x off. Reasons, measured here: 20-800 Adam updates separate consecutive
checkpoints, each moving every weight by up to lr = 1e-3, and the training loss is sqrt(KL) per value (~|q - p| near
the target: L1-like), so the loss is curved along a realized step (checks a, b). This module therefore measures the
realized learning EXACTLY: the change of each fifth's loss between two checkpoints, from forward passes on fixed
episodes (no Taylor expansion), and keeps the first-order term only as the diagnostic it is.

LOSS. The run's own training loss (td_analysis.run_loss "std" = model_backward.SANITY_loss): for every episode i and
step t the goal variable's belief q (softmax of the cumulative goal logits) vs Bayes's goal marginal p, per value r
D_r = (x log(x/y) + y log(y/x)) / 2 with x, y = soft_clip(q), soft_clip(p) (eps 1e-3); the item's loss
l_it = sum_r (D_r - D_min + c)^0.5 with c = 1e-8; L = sum_it l_it / (n T R) (the mean over items and values). D_min
(training: the batch minimum, ~-3e-8 in float32, a rounding artifact) is fixed at 0 (float64 minima are ~1e-11), so
every checkpoint is scored with the same function. The loss is measured against Bayes, so Bayes's own belief scores
D = 0, i.e. the floor sqrt(c) = 1e-4 per item and value (numbers: 1.000000e-04 on M; "0 by construction" until 11-02).
Check: the loss of checkpoint 100 on the main set, 0.31710, vs the run's own logged training-batch loss at the next
update, 0.31677.

EPISODES. Training distribution (env_generators.default_context with test_set False: states 0-499, at most one test
state 0-49 per pair, the goal never a test state; latent values uniform; observations ~ Bernoulli(L) of the runs' world,
td_analysis.world_tables; Bayes side td_analysis.bayes_side, float64). Main set M: 2,000 episodes (seed 20261111;
19.8% contain a test state, goal in position 0 in 39.6%, Bayes / naive goal accuracy at T .776 / .443); gradient set G:
2 x 500 episodes (seed 20261112; 18.4%, 39.7%, .758 / .430). Every episode draws its own pair of variables (its own
environment; 2,957 distinct pairs in the 3,000, none of them one of the 50 probe environments, which pair two test
variables, as no training episode does), so resampling episodes is resampling environments. The same episodes at
every checkpoint and for both runs; per-item losses at all 114 checkpoints (0 to 39,996 training episodes).
Cross-check on the 10,000 probe episodes (50 test environments x 100 cells x 2 streams, never trained) from the
beliefs stored at every checkpoint.

GROUPS (paired per variable: only the goal variable has a loss term, so each item (i, t) is binned by the GOAL
variable's own entanglement ent[i, t, goal_i], step_dkl as everywhere): group 0 = step 1 (3.3% of items); groups 1-5 =
fifths of steps 2-30, ranked within each step (600 items per fifth and step in M + G; item share of a fifth 19.33%).
Every fifth has the same number of items at every step, so the drift of the loss over steps cannot make a
between-fifth difference.
Naive's loss per item and value by fifth (main + gradient set, as plotted; its goal marginal in place of the network's
belief): 0.1165 / 0.1408 / 0.1674 / 0.2094 / 0.2458 (step 1 0.0734; q5 / q1 2.11; naive's q5 share of its loss .275).
Within-step Spearman(goal entanglement, naive's per-item loss) .296 (mean over steps 2-30).

MATCHED CONTROL. The same fifths of the goal variable's entanglement, but ranked INSIDE 10 strata of naive's per-item
loss within each step (union of M and G: 300 items per stratum and step, 60 per matched fifth). The matched fifths have
naive losses 0.1742-0.1776 per item and value (raw 0.1165-0.2458) while their mean entanglement still runs .039 ->
.689 (raw .029 -> .723). Variants (numbers): 5 and 20 strata; FR (factorization regret of the goal variable, KL(Bayes
|| naive) of its marginal belief, the paper's forward form, exact: main.utils.factorization_regret through
td_analysis.bayes_side; it compares the same two beliefs as naive's loss, so it is not an independent control. Until
11-10 it was the symmetric KL clipped at 1e-15, i.e. naive's loss without the per-value square root; the FR numbers
below were re-measured on 11-10: only the "fr" variant changed, episodes_<set>.npz 'fr' and match_<run>.npz
'blk_fr', the old files kept as <name>_sym_superseded.npz); Bayes's goal entropy, a difficulty that does not involve naive (its matched fifths keep naive's
gradient, 0.114-0.255); naive's loss x Bayes's entropy (5 x 5 strata). Naive's loss at t is the error that the
entanglement of steps <= t has left in the factorized marginal, so matching on it asks whether the CURRENT step's
entanglement draws learning beyond the error naive already makes there. It can remove part of a genuine entanglement
effect (over-control): a matched null means "nothing beyond naive's error", not "no effect of entanglement". The
naive-matched q5 has sharper Bayes targets than q1 (mean entropy .796 vs 1.063), which the joint matching removes
(.899 vs .926): a candidate reason why some naive-matched surpluses vanish under it.

QUANTITIES (training-loss units; additive over items, so every share is a ratio of sums, never a mean of ratios):
  realized share   -dL_q / -dL over a window = [L_q(W_a) - L_q(W_b)] / [L(W_a) - L(W_b)], exact (forward passes at
                   the window's two checkpoints); 15 log-even windows with edges 0, 20, 40, 80, 160, 320, 640, 1,040,
                   1,600, 2,400, 4,000, 6,400, 9,600, 15,200, 24,800, 39,996 training episodes; pooled windows as
                   ratios of sums between their edges
  loss share       L_q / L at the window's start (= the realized share if learning removed the same fraction of every
                   fifth's loss: "proportional learning")
  excess           realized share - loss share (b, c); its SE: block jackknife of the DIFFERENCE (30 blocks of 100
                   episodes, M 20 and G 10, one block left out of both terms; the first version drew the SE of the
                   realized share alone). Identity (exact; numbers checks it to 4.5e-15): excess = B / (A - B) x
                   (L_q/L at the start - at the end), B / (A - B) = the loss at the window's end over its decrease, the
                   same for every grouping. So the excess is a fraction of the window's learning, scaled up where
                   little is learned (late windows, the ESN)
  window decrease  -dL = L(W_a) - L(W_b) per item and value of all items (strips under b and c)
  loss units       the fifth's decrease minus its proportional decrease, per item and value of the fifth
                   = excess x (-dL) / .1933 (the fifth's item share)
  GD share         <g_q, g> / |g|^2, the fifth's additive part of the squared gradient = its share of the first-order
                   decrease under plain gradient descent; estimated without the noise bias of a squared sample
                   gradient by crossing independent halves: per item <grad l_it, g^(other half)> on G (forward mode),
                   pooled as a ratio of sums over the window's sampled pairs; windows whose two half estimates differ
                   by an SE above 0.05 are not drawn (trained: 20-40 and 40-80; ESN: all but 0-20 and 4,000-9,600;
                   no pairs after 15,200)
  first-order share  grad L_q(W_k) . dW / grad L(W_k) . dW (the realized step's direction), per item by forward mode
                   on G, pooled over the window's pairs
  first-order check  [L(W_{k+1}) - L(W_k)] / grad L(W_k) . dW per pair on G
37 consecutive-checkpoint pairs for the first-order and GD terms (dW = W_{k+1} - W_k of every trainable tensor; the
ESN's LSTM is frozen and checked unchanged), greedy log-even (each start >= 1.12 x the previous): starts 0, 20, 40, ...,
180, 220, 260, 300, 340, 400, 480, 560, 640, 720, 880, 1,040, 1,200, 1,360, 1,600, 1,840, 2,080, 2,400, 2,800, 3,200,
3,600, 4,200, 4,800, 5,400, 6,200, 7,000, 8,000, 9,600 training episodes (20 updates apart to 400, 80 to 2,400, 200 to
8,000, then 800); the GD and first-order shares therefore stop at 9,600 (the window 9,600-15,200 holds one pair).

TIMING. A crossing is the training episode (log-interpolated between checkpoints; the initial network counts as 10)
after which the curve stays past the level to the end of training (39,996). 5-95% intervals: 1,000 multinomial
resamples of the 3,000 episodes (= of their environments; td_tt_gap's seed 20261103 since 11-02, before 20261101),
the level resampled with them where it is a data quantity (naive's loss); a crossing is reported only when at least
95% of the resamples cross (else "not established"). For the timeline the excess is evaluated at every checkpoint over
a window [x / sqrt 2, x sqrt 2] of cached checkpoints around it (centred 2x windows; the pink curve in b, c). Where no
crossing applies, timing is read from the 15 non-overlapping windows (never from overlapping sliding windows, whose
counts are not independent); overlapping early windows are used only to show that the early excess changes sign with
its edges.

CHECKS ("check trained 100", every pair, numbers):
  * hand LSTM vs nn.LSTM max |du| 4.8e-6; forward-mode directional derivative along dW vs autograd: relative 5.2e-9 (the
    W_hh and read-out parts equal to all printed digits), along g itself vs |g|^2: 2.1e-8; central differences along
    dW: -461.17 (eps 1e-2), -463.95 (eps 1e-3) vs -464.39; at every pair the forward-mode first-order term on G equals
    autograd's <g, dW> to 6 digits. The verifier reproduced the per-item losses with nn.LSTM and the training code's
    DKL_sym (max |diff| 2e-5) and every window share from the caches.
  * Sampling (checks e, the plotted excesses of every fifth and window): the main set alone vs the gradient set alone
    differ by at most .049 raw / .040 matched (trained) and .069 / .105 (ESN, matched at 15,200-24,800, a window with
    little learning; G alone has ~1.7x the SE); the training episodes vs the probes by at most .038 / .047 (trained)
    and .050 / .053 (ESN). The realized shares (not plotted) differ by at most .051 / .050 (|M - G| to 9,600; .073 in
    the ESN's last two windows) and .038 / .042 (training vs probes). The trained 6,400-9,600 raw surplus, all
    estimates: M + G +.019 (SE .007), M +.031, G -.005, probes +.019 (.005), and the verifier's 3,000 fresh episodes
    (seed 777, independent code) +.027 (.005): 4 of 5 positive.
  * Half-set gradients: cos(g^(1), g^(2)) median .45 (trained; -.26 to .92, 3 of 37 pairs below .2) and .35 (ESN; -.13
    to .93, 11 of 37 below .2): the population gradient is weak against the 500-episode sampling noise at some
    checkpoints, which is why the GD share is pooled and masked.
  * 11-02 (numbers): the identity excess = B / (A - B) x share change and match_windows agree to 4.5e-15; Bayes's own
    belief scores 1.000000e-04 per item and value; the synthesis curves and naive's resampled levels equal a rebuild by
    explicit episode repetition from the raw per-item loss caches (relative 3e-15); c0 equals the timeline crossing and
    c50 this module's crossing code on the same resamples. Fix-2 scratch check (not in numbers): the probe cache rebuilt
    at all 114 checkpoints equals the previous cache (window edges only) exactly at the edges.

FINDINGS. Excess of the most entangled fifth q5 (realized share - loss share at the start; SE of the difference) per
window, raw | matched on naive's loss (M / G | probes, environment-jackknife SE); [the window's total decrease -dL per
item and value; B / (A - B)].
  Trained RNN:
     0-20    -.011 (.009) | +.019 (.009)                                                       [.0411; 8.7]
    20-40    -.031 (.017) | +.017 (.017)                                                       [.0085; 41]
    40-80    -.115 (.010) | -.076 (.011)  M -.080 / G -.068 | probes -.070 (.015); raw probes -.108 (.015)  [.0321; 9.9]
    80-160   -.008 (.011) | +.066 (.011)  M +.061 / G +.075 | probes +.068 (.015)              [.0292; 9.9]
   160-320   -.022 (.008) | -.008 (.007)                     | probes -.021 (.010)              [.0468; 5.2]
   320-640   +.000 (.005) | +.012 (.005)  M +.012 / G +.011 | probes +.005 (.006)              [.0608; 3.0]
   640-1040  +.009 (.004) | +.014 (.003)  M +.016 / G +.009 | probes +.009 (.004); raw M +.014 / G -.001  [.0588; 2.1]
  1040-1600  -.008 (.005) | -.000 (.004)                                                       [.0379; 2.2]
  1600-2400  +.007 (.005) | -.003 (.004)                                                       [.0223; 2.8]
  2400-4000  +.006 (.005) | -.005 (.004)                                                       [.0172; 2.6]
  4000-6400  +.022 (.006) | +.014 (.004)  M +.008 / G +.025 | probes +.019 (.005); raw probes +.031 (.006)  [.0113; 3.0]
  6400-9600  +.019 (.007) | +.012 (.006)  M +.025 / G -.015 | probes +.009 (.004)              [.0067; 4.1]
  9600-15200 +.052 (.009) | +.029 (.007)  M +.029 / G +.028 | probes +.031 (.007); raw probes +.048 (.008)  [.0045; 5.0]
 15200-24800 -.000 (.006) | -.002 (.006)                                                       [.0048; 3.8]
 24800-39996 +.001 (.005) | +.002 (.006)                                                       [.0038; 3.7]
  Pooled, raw | matched (probes raw / matched) [B / (A - B)]: 40-160 -.059 (.006) | -.005 (.006) (-.046 / +.002) [4.7];
  0-320 -.029 (.002) | .000 (.003) (-.031 / -.000) [1.5]; 80-320 -.016 (.006) | +.016 (.005) (-.022 / +.011) [3.2];
  320-1,040 +.004 (.002) | +.011 (.002) (M +.012, G +.008; probes -.002 / +.006 (.003)) [1.03]; 320-1,600 .000 (.002)
  | +.006 (.001) [.54]; 1,600-9,600 +.008 (.001) | +.002 (.001) (probes +.009 / +.003) [.48]; 1,600-4,000 +.006
  (.003) | -.003 (.002) [1.15]; 4,000-15,200 +.023 (.002) | +.013 (.002) (M +.015, G +.010; probes +.025 / +.015)
  [1.02]; 15,200-40k .000 (.003) | -.000 (.002) [1.66]; 80-40k +.001 (.000) | +.001 (.000) [.047: the loss falls from
  .318 to .014 over it, so a pooled excess over it is ~0 by construction]. In loss units (matched, per q5 item and
  value): 40-80 -.0126, 80-160 +.0100, 40-160 -.0015, 0-320 +.0001, 320-1,040 +.0068, 4,000-15,200 +.0016.
  Non-overlapping windows from 1,600 on: raw positive in 6 of 7 (3 above 2 SE), matched 4 of 7 (3 above 2 SE), none
  below -2 SE.
  Early windows (1.6-2.5x wide; numbers prints 20), matched | probes matched: a dip in windows starting at 40 (40-80
  -.076 (.011) | -.070 (.015); 40-100 -.045 (.010) | -.039 (.014)), not at 60-100 (-.005 (.014)), smaller and not
  replicated at 60-140 (-.020 (.008) | -.006 (.017)); a surplus in windows ending at 160-180 (80-160 +.066 (.011) |
  +.068 (.015); 80-180 +.102 (.014) | +.101 (.017); 100-160 +.053 (.012); 100-180 +.096 (.016) | +.084 (.015)) and at
  240 (120-240 +.036 (.011); 140-240 +.061 (.012) | +.035 (.014)), smaller in 80-200 (+.022 (.008) | +.032 (.011)), but
  not in 80-140 (+.001 (.011)) or 100-200 (+.007 (.008)); a second deficit in windows from 160-180 to 260-320 (160-260
  -.031 (.008) | -.047 (.014); 160-280 -.025 (.009) | -.028 (.012); 180-320 -.023 (.007) | -.033 (.010)), but 160-300
  +.017 (.012) | -.010 (.024) and 200-320 +.013 (.009) | -.005 (.009). Of the 94 pairs of windows 1.5-2.6x wide within
  40-400 that differ in one edge by one checkpoint (numbers), 4 flip sign with both values beyond 2 SE in the training
  episodes (e.g. 140-280 +.020 (.007) | +.012 (.008) vs 160-280 -.025 (.009) | -.028 (.012)); the probes keep the signs
  in all 4, but beyond 2 SE on both sides in none. Centred 2x windows (pink in c): +.066 [+.045, +.087] at 120, -.031
  [-.042, -.020] at 240, -.016 [-.026, -.005] at 260. At 120, 240, 480 and 960 the window choice is an exact log tie
  broken by argmin (80-160 at 120; 180-340 at 240); the other choice gives +.102 / -.018 with the same signs (at these
  ties only values within .015 of 0 change sign: the trained RNN's raw -.008 -> +.015 at 120 and matched -.001 -> +.002
  at 960). In loss units per q5 item and value: 80-180 +.0147, 160-260 -.0049, 180-320 -.0058. Across matchings 80-160
  is +.066 (naive's loss), -.041 (Bayes's entropy), +.028 (naive's loss x entropy).
  Matching variants (q5 excess): 5 / 20 strata within .008 of 10 strata in every window; FR within .018 (20-40
  -.001 against +.017; 80-160 +.076). 320-1,040: naive's loss +.011 (.002), 5 / 20 strata +.011 / +.010, FR +.014,
  Bayes's entropy -.006 (.002), naive's loss x entropy +.001 (.001) (320-1,600: -.008 / -.003). 4,000-15,200: +.011 to
  +.022 under all six (naive's loss +.013, 5 / 20 strata +.014 / +.014, FR +.013, entropy +.022, naive's loss x
  entropy +.011, each SE .002). (With the symmetric FR until 11-10: within .015 (80-160 +.081), +.009, +.012.)
  Bayes's entropy keeps the early raw deficits (0-20 -.052, 40-80 -.097, 80-160 -.041, 1,040-1,600 -.021); naive's
  loss x entropy: 40-80 -.043, 1,040-4,000 -.015 to -.007 (pooled 1,600-9,600 -.000).
  Edges shifted (numbers): the early matched values swing with the checkpoint-scale oscillation (40-60 -.107, 60-80
  -.051, 80-100 +.135, 100-140 -.041, 120-160 +.142); the late matched surplus holds for shifted edges (3,200-5,600
  +.008, 7,000-10,400 +.018, 9,600-12,000 +.023, 12,000-15,200 +.041; raw +.016, +.027, +.037, +.080).
  Loss per item and value (q1 ... q5): start .392-.418 (step 1 .165); 880 .126-.160; 10.4k .0245-.0280 (q5 / q1
  1.14); 40k .0140 / .0134 / .0136 / .0136 / .0142 (1.02). Matched fifths: start .388 -> .422 (q5 above q1: Bayes is
  sharper on entangled items, mean entropy .80 vs 1.06), 40k .0150 / .0137 / .0134 / .0131 / .0135 (q5 below q1).
  The q5 loss share (the state the excess integrates): raw .202 -> .220 (320) -> .219 (880) -> .201 (10.4k) -> .193
  (40k); matched .204 -> .209 (80) -> .196 (880) -> .187 (10.4k) -> .183 (40k). Step 1: .165 -> .062 (10.4k).
  Gradient vs realized (windows with a reliable GD share; raw fifths): GD share q1 .192-.280 (item share .193), q5
  .107-.204: the gradient comes disproportionately from the least entangled steps; realized - GD q5 +.025 to +.112 from
  80 on. The realized step's direction already moves part of the way (first-order share between GD and realized, e.g.
  1,600-2,400 GD .138, FO .208, realized .227; 4,000-6,400 .125 / .195 / .237), and the curvature removes more of the
  least entangled steps' first-order gain (q1 FO .19-.23 vs realized .16-.18, 320-9,600).
  ESN:
     0-20    +.021 (.008) | +.021 (.006)                                                       [.0347; 10.5]
    20-40    -.024 (.020) | +.044 (.022)                                                       [.0052; 70]
    40-80    -.149 (.030) | -.078 (.023)  M -.105 / G -.033 | probes -.040 (.021); raw probes -.108 (.021)  [.0097; 36]
    80-160   -.138 (.014) | -.066 (.011)  M -.071 / G -.055 | probes -.048 (.017)              [.0176; 19]
   160-320   -.076 (.009) | -.018 (.008)  M -.027 / G -.001 | probes -.031 (.014)              [.0291; 10.5]
   320-640   -.068 (.010) | +.009 (.008)                                                       [.0213; 13]
   640-1040  -.073 (.011) | -.000 (.009)                                                       [.0208; 12.6]
  1040-1600  -.066 (.011) | +.014 (.010)                                                       [.0207; 11.7]
  1600-2400  -.061 (.011) | +.034 (.011)  M +.030 / G +.043 | probes +.038 (.011)              [.0161; 14]
  2400-4000  -.079 (.010) | +.020 (.010)                     | probes +.031 (.011)              [.0171; 12.2]
  4000-6400  -.082 (.010) | +.002 (.008)                                                       [.0150; 12.9]
  6400-9600  -.051 (.017) | +.019 (.013)                                                       [.0077; 24]
  9600-15200 -.037 (.017) | +.009 (.014)                                                       [.0070; 25]
 15200-24800 -.005 (.016) | +.006 (.015)                     | raw probes -.038 (.017)          [.0063; 27]
 24800-39996 -.022 (.019) | +.001 (.018)                     | raw probes -.023 (.022)          [.0053; 31]
  Raw: every window from 40 to 15,200 negative at >= 2.2 SE (probes -.04 to -.11); the last two are within 1.2 SE of 0.
  Pooled, raw | matched (probes) [B / (A - B)]: 40-160 -.139 (.017) | -.069 (.013) (-.104 / -.044) [12.2]; 0-320 -.054
  (.004) | -.015 (.004) (-.052 / -.012) [3.2]; 80-320 -.095 (.009) | -.034 (.008) (-.090 / -.036) [6.5]; 320-1,600
  -.064 (.007) | +.007 (.005) (-.070 / +.001) [3.8]; 1,600-4,000 -.068 (.008) | +.026 (.007) (probes +.033 (.009))
  [6.3]; 1,600-9,600 -.064 (.006) | +.017 (.005) [3.3]; 4,000-15,200 -.060 (.009) | +.008 (.008) [6.0]; 15,200-40k
  -.013 (.013) | +.004 (.012) [14.4]; 80-40k -.048 (.003) | +.002 (.002) (-.052 / +.001) [.91]. In loss units per q5
  item and value: raw 80-40k -.0456; matched 40-80 -.0039, 80-160 -.0060, 40-160 -.0097, 0-320 -.0074, 1,600-4,000
  +.0044.
  Early windows, matched (numbers): negative in every window that starts before 180 (-.016 to -.078; 160-260 -.031
  (.009), 160-280 -.018, 160-300 -.029), 180-320 -.003, 200-320 +.002; probes -.007 to -.066 except 60-100 (+.007).
  Variants, 80-40k: 5 / 20 strata -.000 / +.004, FR +.006, Bayes's entropy -.057 (every window from 20 on -.036 to
  -.128), naive's loss x entropy -.008 (.002) (40-160 -.046, 4,000-15,200 -.020 (.007), 15,200-40k -.038 (.011)).
  1,600-4,000: naive's loss +.026 (.007), 5 / 20 strata +.022 / +.027, FR +.030, Bayes's entropy -.100 (.009), naive's
  loss x entropy -.007 (.007). (FR = KL(Bayes || naive) since 11-10; with the symmetric FR +.007 and +.034; the ESN's
  FR variant is within .022 of naive's loss in every window, largest at 15,200-24,800: +.028 against +.006.)
  Ceiling reference (the verifier's): the q5 share of the loss ABOVE naive's level at the window's start vs the realized
  q5 share: 80-9,600 .135 vs .148, 160-320 .143 vs .136, 1,600-9,600 .135 vs .171, 4,000-6,400 .104 vs .164 (for the
  trained RNN the loss is below naive's on every fifth from ~1k, so this reference is empty).
  Loss per item and value: 10.4k .136 (q1) - .243 (q5) (1.79), 40k .128 - .222 (1.74), naive .1165 - .2458 (2.11);
  matched fifths at 40k .171 / .169 / .167 / .169 / .173. q5 loss share raw .202 -> .227 (880) -> .256 (10.4k) -> .257
  (40k) toward naive's .275; matched .204 -> .208 (880) -> .201 (40k).
  GD share where measurable (0-20, 4,000-9,600): q1 .191-.372, q5 .119-.189; realized - GD q5 +.042 / +.082.
First-order check (actual / first order, per pair on G): trained median .231, 27 of 37 pairs below .5, 5 negative (loss
  up while first order predicted down: 120, 160, 3,600, 4,800, 7,000); the 20 pairs to 880 range from -.54 to 1.56
  (quartiles .33-.58; the first pair .035: first order -1.047 vs actual -.037), from 2,080 on -.07 to .23. Under a
  quadratic profile a ratio below .5 means the step ends past the minimum along its chord. ESN median .692 (9 of 37
  below .5; to 880 .51-.99; from 2,080 .25-.60; none negative).
  Line profiles (500 M episodes; s = 0, 1/32, 1/16, 1/8, 1/4, 1/2, 3/4, 1; |first order| per item and value in checks
  a): trained, the end slope is positive in 7 of 7 profiled pairs (+.003 to +.021), but the loss visibly rises again
  on the grid only at 80 (mild: minimum at s = 1/2, L(1) - L(3/4) = +.0006) and from 1,600 on (a valley: 1,600 ->
  1,680 minimum at 3/4; 4,800 -> 5,000 .04169 -> .03938 (s = 1/2) -> .04138; 9,600 -> 10,400 .02873 -> .02613 ->
  .02822). At 480 and 880 the step ends just past the minimum (L(1) < L(3/4) by .0003 / .0004, end slope > 0;
  actual / first order .54 / .56). The first pair is a saturating drop (.398 -> .361 by s = 1/8, then .360-.363), not
  an overshoot. 7 of 37 pairs are profiled. ESN: nearly linear with mild convexity, minimum at s = 1 in 7 of 7, end
  slopes -.004 to +.001.
  Read-out part of the first-order change: trained median 13% (q1) / 11% (q5), -2-21% / 1-18% from 320 on (no
  difference by entanglement); ESN median 54% / 36%.

FAILED / NULL:
  * The ESN's raw deficit on entangled steps is not entanglement-specific: matched on naive's loss it disappears from
    ~320 on (the first version's "its learning keeps drawing on the least entangled steps" was a naive-difficulty
    effect). Its early matched deficit (windows starting before 180) and the joint-matching late deficit are the
    remainders.
  * The trained RNN's matched "rebound" at 80-160 (+.066) is not an event: it is one phase of a checkpoint-scale
    oscillation whose size changes several-fold when one edge moves by one checkpoint (80-140 +.001, 80-160 +.066,
    80-180 +.102, 80-200 +.022; probes +.027, +.068, +.101, +.032) and whose sign can flip (140-280 +.020 +- .007 vs
    160-280 -.025 +- .009; probes +.012 / -.028). The first version's "repaid at 80-160" is withdrawn; only pooled early
    values are compared between the networks.
  * The trained RNN's 320-1,040 extra draw is not robust to the matching variable: present matched on naive's loss
    (+.011), absent at naive's loss x Bayes's entropy (+.001), reversed at Bayes's entropy (-.006). The ESN's
    1,600-4,000 surplus likewise (+.026; -.007; -.100). Only the trained RNN's 4,000-15,200 surplus survives every
    matching.
  * No extra draw on entangled steps tied to passing naive (880) or the memory-use flip (800-1,040): the matched surplus
    of 320-1,040 starts before them (320-640 +.012) and is absent at 1,040-1,600 (-.000). Raw, the 640-1,040 window's
    +.009 is not seen in G (-.001).
  * The late raw surplus is not a robust entanglement effect at 1,600-4,000 (matched -.003 +- .002); it is at 4,000-
    15,200, about half its raw size. After 15,200 nothing (raw and matched).
  * No excess crossing is established: the four excess curves never stay past 0 to the end (trained: the surplus ends
    ~15k; ESN raw: its last windows are within noise of 0), so their timeline crossings are NaN.
  * The gradient-share comparison does not separate the networks: both put more of the realized learning on the most
    entangled fifth than its gradient share (it reflects the loss geometry: sqrt-KL overweights near-target items).
  * The first-order prediction is unusable as an attribution for the trained RNN (median actual / first-order .23;
    5 pairs change sign).
  * Abandoned design: an exact path-integrated split by weight group along the segment with Simpson's rule (first pair:
    -.173 vs the exact -.043, the change happens in the first 1/16 of the step); the weight-group split is first order
    only.
  * Per-pair (20-80 update) realized shares swing with the oscillation of the dynamics (e.g. pair 150 -> 175, M -.0031
    vs G +.0029); only window-pooled shares are read.

CAVEATS. "Learning on a step" here = the realized decrease of the loss term AT that step (the belief at t, which
integrates every earlier observation): the split by loss step; a split by the step at which the weights are USED
(td_entanglement_driver's attribution) is a different question (its exact path-integrated version was not run). The
matched result depends on the matching variable (see MATCHED CONTROL; Bayes's entropy and naive's loss x entropy give
different answers for the ESN, for the trained RNN's 320-1,040 and for the ESN's 1,600-4,000). The excess is a
fraction of each window's learning (QUANTITIES): the same share change reads up to 20x larger in one window than in
another (B / (A - B) 2.1-41 trained, 10-70 ESN), so magnitudes compare across windows or networks only as fractions
of each one's learning (FINDINGS gives the totals and loss units); pooled values over long windows are scaled down by
the same factor (trained 80-40k .047, so its +.001 says nothing; ESN .91). The GD and first-order terms use the raw
fifths and G only, and refer to the straight segment between checkpoints, not the training path (the realized shares
do not depend on it). Shares are of small decreases late (.004-.007 per item and value per window from 9,600 on), so
the late windows carry larger SEs. One seed.

TIMELINE EXPORT (timeline.npz in the cache folder; numbers / timeline writes it): x = training episodes of the 114
checkpoints (E - E // 5; 0 for the initial network). Rows (run x measure): "loss, most entangled fifth" and "loss,
least entangled fifth" (family answers; per item and value; bayes = the floor 1e-4, 0 until 11-02; naive = level =
naive's loss on that fifth; crossing = the training episode after which the loss stays below naive's on that fifth):
trained 493 [451, 513] and 957 [924, 1,010]; ESN 8,984 [7,213, 10,811] and never (.128 vs .1165 at 40k). "learning
excess, most entangled fifth" and "learning excess, matched on naive" (family learning; the q5 excess over centred 2x
windows; level 0; side + for the trained RNN (surplus), - for the ESN (deficit); bayes / naive NaN): no crossing
established (the fraction of resamples that never stay past 0: trained .91 / .40, ESN .36 / .84). value / lo / hi =
the full-sample value and the 5-95% band of the episode bootstrap (NaN where no window: the first and last
checkpoint). Since 11-02 the resamples are the synthesis ones (seed 20261103); with the earlier seed 20261101 the
intervals were [452, 513], [923, 1,005] and [7,168, 11,044], the never-fractions .92 / .39 / .37 / .84.

SYNTHESIS EXPORT (synthesis.npz in the cache folder; "synthesis" writes it, "numbers" too, which prints the table and
its checks; conventions of td_tt_gap): rows "loss, most entangled fifth" and "loss, least entangled fifth" for the
Trained RNN and the ESN (family learning, shape rise): the fifth's training loss per item and value at the 114
checkpoints (x = training episodes, the initial network at 10), naive = naive's loss on the same fifth, Bayes = the
loss floor 1e-4; gap fraction f = (naive - loss) / (naive - 1e-4) (~1 - loss / naive), so c0 = below naive's level for
good and c50 = below the naive-Bayes midpoint (~naive / 2) for good. Sample: the 3,000 training-distribution episodes,
which are not the probe environments (EPISODES), resampled by episode 1,000 times with td_tt_gap's seed 20261103
(paired False: not pairable with the probe-based topics), naive's level with them; every resampled curve is the
count-weighted ratio of sums (loss_curves; all counts 1 reproduces the full sample, asserted). The learning-excess rows
are left out (no Bayes level). c0 / c50 [5-95%]: trained q5 493 [451, 513] / 1,220 [1,167, 1,285]; q1 957 [924,
1,010] / 2,202 [2,085, 2,368]; ESN q5 8,984 [7,213, 10,811] / never (no resample crosses); q1 never / never. Gap
fraction at 880 / 10.4k / 40k: trained q5 .35 / .89 / .94, q1 -.08 / .79 / .88; ESN q5 -.29 / .01 / .10, q1 -.96 /
-.17 / -.10. The topic has no half-way crossing of its own: c0 equals its crossing at naive's level (same rule, same
resamples: identical points and intervals), and c50 was recomputed with this module's crossing code on the same
resamples (identical). One note per row (syn_note; numbers prints them): what is particular to the row, then the common
definitions. The q5 rows cross before the least entangled fifth because naive's loss is 2.1x higher on this fifth, while
the network's own loss is higher there (q5 / q1 > 1 at 113 of 114 checkpoints, trained, .993 at 39,200; 114 of 114,
ESN); the ESN's q1 stays above naive's level (.128 vs .1165 at 40k; above it at all 114 checkpoints).

Outputs (caches main/DATA/experimental/training_dynamics/through_training/entangled_learning/): episodes_main.npz,
episodes_grad.npz, naive_loss.npz; <run>/loss_{main,grad}_<E>.npz (per-item loss of all 114 checkpoints);
<run>/pair_<E0>_<E1>.npz (per item on M: losses at both ends; on G: losses, first-order derivative along dW and its
read-out part, projection on the other half's gradient; scalars); <run>/profile_<E0>_<E1>.npz; series_<run>.npz;
match_<run>.npz (loss sums by block / episode and group for every grouping); probe_shares_<run>.npz (probe loss sums
per environment and group at all 114 checkpoints); timeline.npz; synthesis.npz. Figures
main/REPLICATION/figure_bin/td_tt_entangled_learning.png (answer: a loss per fifth with naive's level (step 1 not
drawn), b raw excess, c matched excess, both with the q5 excess over 2x windows centred on every checkpoint (pink) and
strips of each window's total loss decrease) and td_tt_entangled_learning_checks.png (a line profiles, b first-order
check, c four shares of the most entangled fifth, d every matching, e sampling checks). Run from the repository root
(CPU; pairs 32-94 s each, losses ~20 s per checkpoint, profiles ~2 min, probe cache ~25 s per run; jobs stop after
--minutes and resume):
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    PY=/home/johns/anaconda3/envs/eureka/bin/python; S=main/REPLICATION/claude_plots/td_tt_entangled_learning.py
    $PY $S episodes; $PY $S check trained 100
    $PY $S data trained --minutes 7     (repeat until done; then echo)
    $PY $S losses trained echo --minutes 8; $PY $S profile trained 0 4 15 19 23 31 36 (and echo)
    $PY $S summary trained echo; $PY $S numbers (also writes timeline.npz and synthesis.npz); $PY $S synthesis
    MPLBACKEND=Agg $PY $S figure
"""
import os
import sys
import time
import glob
import warnings
import argparse
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_analysis as td                                                          # noqa: E402
import td_tt_gap as gap                                                           # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
OUT = os.path.join(td.TD, "through_training", "entangled_learning")
T, R, C = td.T, td.R, td.C
N_MAIN, N_GRAD_HALF = 2000, 500
SEED_MAIN, SEED_GRAD = 20261111, 20261112
CHUNK = 500                                                     # episodes per forward-mode chunk
GCHUNK = 250                                                    # episodes per autograd chunk
RATIO, MAX_TRAIN = 1.12, 10400
PROFILE_S = (0., 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 1.)          # line profile (a few pairs)
DMIN = 0.          # the loss's batch-minimum offset, fixed (float64 minima are ~1e-11; see LOSS)
BAYES_LOSS = 1e-8 ** .5            # Bayes's own belief (D = 0) scores sqrt(1e-8) per value: the loss's floor
N_GRP = 6                                                       # step 1 + 5 fifths
RUNS = ("trained", "echo")
PARTS = {"ro": ("classifier_readout.weight", "classifier_readout.bias"),
         "hh": ("LSTM.weight_hh_l0",),
         "in": ("classifier_readin.weight", "classifier_readin.bias", "LSTM.weight_ih_l0", "LSTM.bias_ih_l0",
                "LSTM.bias_hh_l0", "STM", "LTM")}
NAMES = [x for v in PARTS.values() for x in v]


def train_ep(E):
    E = np.asarray(E)
    return E - E // 5


# ═══════════════════════════════════════════════════════════════════
# Episodes (training distribution) and groups
# ═══════════════════════════════════════════════════════════════════

def make_episodes(n, seed):
    """n training-distribution episodes (env_generators.default_context, test_set False) of the runs' world"""
    rng = np.random.default_rng(seed)
    S, n_test = 500, 50
    ctx = rng.choice(S, size=(n, 2))
    tmask = ctx < n_test
    keep = np.zeros_like(tmask)
    keep[np.arange(n), tmask.argmax(1)] = tmask.any(1)                     # at most one test state per pair
    rep = tmask & ~keep
    ctx[rep] = rng.choice(np.arange(n_test, S), size=int(rep.sum()))
    pairs = np.sort(ctx, 1)
    u = rng.random((n, 2))
    u[pairs < n_test] = 0.                                                 # the goal is never a test state
    goal = u.argmax(1)
    r = rng.integers(R, size=(n, 2))
    ck = torch.load(td.ckpt_path(td.WORLD, 0), map_location="cpu", weights_only=False)
    L, Z = td.world_tables(ck, pairs)
    obs = rng.random((n, T, td.NO)) < L[np.arange(n), :, r[:, 0], r[:, 1]][:, None, :]
    B = td.bayes_side(obs, L, r)
    ar = np.arange(n)
    log_pg = B["log_jb"].astype(np.float64)[ar, :, goal]                    # (n, T, R) Bayes's goal marginal (log)
    ent_g = B["ent"][ar, :, goal]                                           # (n, T) the goal variable's entanglement
    return dict(pairs=pairs, goal=goal, r=r, obs=obs, Z=Z.reshape(n, -1).astype(np.float32), log_pg=log_pg,
                ent=B["ent"], ent_goal=ent_g, fr=B["fr"], right_bayes=B["right_bayes"], right_naive=B["right_naive"])


def groups(ent_goal):
    """(n, T) group of every item: 0 = step 1, 1-5 = fifths of the goal variable's entanglement within each step 2-30
    (rank-based: exactly n / 5 episodes per fifth and step)"""
    n = len(ent_goal)
    g = np.zeros(ent_goal.shape, int)
    for t in range(1, T):
        rank = np.argsort(np.argsort(ent_goal[:, t], kind="stable"), kind="stable")
        g[:, t] = 1 + rank * 5 // n
    return g


def episodes(which):
    path = os.path.join(OUT, f"episodes_{which}.npz")
    if not os.path.exists(path):
        os.makedirs(OUT, exist_ok=True)
        st = np.random.get_state()
        E = make_episodes(N_MAIN if which == "main" else 2 * N_GRAD_HALF, SEED_MAIN if which == "main" else SEED_GRAD)
        np.random.set_state(st)
        E["grp"] = groups(E["ent_goal"])
        np.savez(path, **E)
    return dict(np.load(path))


# ═══════════════════════════════════════════════════════════════════
# Network: loss head, forward with tangents (forward mode), autograd gradient
# ═══════════════════════════════════════════════════════════════════

def load_w(run, ep):
    w = torch.load(td.ckpt_path(run, ep), map_location="cpu", weights_only=False)["weights"]
    return {k: w[k].float() for k in NAMES}


def soft_clip(x, eps=1e-3):
    ciel = 1. - eps
    x = ciel - torch.sigmoid((ciel - x) / eps) * (ciel - x)
    return eps + torch.sigmoid((x - eps) / eps) * (x - eps)


def head_D(cum, goal, log_pg):
    """per-value symmetric KL D (n, T, R) of the goal belief (model_backward.DKL_sym, PM False), float64"""
    n = len(cum)
    cg = cum.reshape(n, T, C, R)[torch.arange(n), :, goal]
    x, y = soft_clip(torch.softmax(cg, -1)), soft_clip(log_pg.exp())
    return (x * torch.log(x / y) + y * torch.log(y / x)) / 2


def item_loss(D, dmin):
    """unnormalized per-item loss l_it = sum_r (D - D_min + 1e-8)^0.5 (the training loss is the mean over i, t, r)"""
    return ((D - dmin + 1e-8) ** .5).sum(-1)


def head_grad(cum, goal, log_pg, dmin):
    """per item: loss l (n, T) and d l / d cum (n, T, 20) (each item depends only on its own cumulative logits)"""
    cum = cum.double().detach().requires_grad_(True)
    with torch.enable_grad():
        l = item_loss(head_D(cum, goal, log_pg), dmin)
        G, = torch.autograd.grad(l.sum(), cum)
    return l.detach(), G


def _inp(E, sl):
    obs = torch.as_tensor(E["obs"][sl], dtype=torch.float32)
    z = torch.as_tensor(E["Z"][sl])
    return torch.cat([obs, z[:, None].expand(-1, T, -1)], -1)                 # (n, T, 15)


def forward(p, inp, tans=None):
    """the hand-written LSTM (gate order i, f, g, o) on inputs inp (n, T, 15) with weights p; tans: name -> (K, *shape)
    tangents (absent = 0) propagated exactly (forward-mode, dual numbers). Returns per-step logits (n, T, 20), their
    tangents (K, n, T, 20) or None, and the states u (n, T, H)"""
    n, H = len(inp), p["LSTM.weight_hh_l0"].shape[1]
    Wi, Wh, Wro = p["LSTM.weight_ih_l0"], p["LSTM.weight_hh_l0"], p["classifier_readout.weight"]
    a = inp @ p["classifier_readin.weight"].T + p["classifier_readin.bias"]
    x = torch.relu(a)
    xp = x @ Wi.T + p["LSTM.bias_ih_l0"] + p["LSTM.bias_hh_l0"]
    h = p["STM"].reshape(1, H).expand(n, H)
    c = p["LTM"].reshape(1, H).expand(n, H)
    tans = tans or {}
    tg = {k: tans.get(k) for k in NAMES}
    K = next(iter(tans.values())).shape[0] if tans else 0
    tb = None
    for k in ("LSTM.bias_ih_l0", "LSTM.bias_hh_l0"):
        if tg[k] is not None:
            tb = tg[k] if tb is None else tb + tg[k]
    hd = tg["STM"].reshape(K, 1, H).expand(K, n, H) if tg["STM"] is not None else None
    cd = tg["LTM"].reshape(K, 1, H).expand(K, n, H) if tg["LTM"] is not None else None
    ys, yds, us = [], [], []
    for t in range(T):
        pre = xp[:, t] + h @ Wh.T
        if K:
            pd = torch.zeros(K, n, 4 * H)
            if tg["classifier_readin.weight"] is not None or tg["classifier_readin.bias"] is not None:
                ad = torch.zeros(K, n, H)
                if tg["classifier_readin.weight"] is not None:
                    ad = ad + torch.einsum("nd,khd->knh", inp[:, t], tg["classifier_readin.weight"])
                if tg["classifier_readin.bias"] is not None:
                    ad = ad + tg["classifier_readin.bias"][:, None, :]
                pd = pd + (ad * (a[:, t] > 0)) @ Wi.T
            if tg["LSTM.weight_ih_l0"] is not None:
                pd = pd + torch.einsum("nd,khd->knh", x[:, t], tg["LSTM.weight_ih_l0"])
            if tb is not None:
                pd = pd + tb[:, None, :]
            if tg["LSTM.weight_hh_l0"] is not None:
                pd = pd + torch.einsum("nd,khd->knh", h, tg["LSTM.weight_hh_l0"])
            if hd is not None:
                pd = pd + hd @ Wh.T
        gi, gf, gg, go = pre.chunk(4, -1)
        i, f, g, o = gi.sigmoid(), gf.sigmoid(), gg.tanh(), go.sigmoid()
        c_new = f * c + i * g
        tc = c_new.tanh()
        h_new = o * tc
        if K:
            pi, pf, pg, po = pd.chunk(4, -1)
            cd_new = (f * (1 - f) * pf) * c + (i * (1 - i) * pi) * g + i * ((1 - g * g) * pg)
            if cd is not None:
                cd_new = cd_new + f * cd
            hd = (o * (1 - o) * po) * tc + o * (1 - tc * tc) * cd_new
            cd = cd_new
            yd = hd @ Wro.T
            if tg["classifier_readout.weight"] is not None:
                yd = yd + torch.einsum("nd,kod->kno", h_new, tg["classifier_readout.weight"])
            if tg["classifier_readout.bias"] is not None:
                yd = yd + tg["classifier_readout.bias"][:, None, :]
            yds.append(yd)
        ys.append(h_new @ Wro.T + p["classifier_readout.bias"])
        us.append(h_new)
        h, c = h_new, c_new
    return torch.stack(ys, 1), (torch.stack(yds, 2) if K else None), torch.stack(us, 1)


def autograd_grad(p, E, train, dmin, sl_all):
    """d L / d w of the mean training loss over the episodes sl_all (autograd; trainable tensors only)"""
    q = {k: (v.clone().requires_grad_(True) if k in train else v) for k, v in p.items()}
    grads = {k: torch.zeros_like(p[k], dtype=torch.float64) for k in train}
    idx = np.arange(sl_all.start, sl_all.stop)
    n_all = len(idx)
    for c0 in range(0, n_all, GCHUNK):
        sl = slice(idx[c0], idx[min(c0 + GCHUNK, n_all) - 1] + 1)
        inp = _inp(E, sl)
        goal = torch.as_tensor(E["goal"][sl])
        lpg = torch.as_tensor(E["log_pg"][sl])
        with torch.enable_grad():
            y, _, _ = forward(q, inp)
            l = item_loss(head_D(y.double().cumsum(1), goal, lpg), dmin)
            L = l.sum() / (n_all * T * R)
            gs = torch.autograd.grad(L, [q[k] for k in train])
        for k, gk in zip(train, gs):
            grads[k] += gk.double()
    return grads


def trainable(run):
    return [k for k in NAMES if not (td._cfg(run).get("reservoir", False) and k.startswith("LSTM."))]


# ═══════════════════════════════════════════════════════════════════
# One checkpoint pair
# ═══════════════════════════════════════════════════════════════════

def pair_list(run):
    """consecutive-checkpoint pairs, log-even in training episodes: the first pair (0 -> 25), then greedily the next
    pair whose start is at least RATIO x the previous start, up to MAX_TRAIN training episodes"""
    eps = td.checkpoints(run)
    cand = [(a, b) for a, b in zip(eps[:-1], eps[1:]) if train_ep(a) < MAX_TRAIN]
    chosen, last = [cand[0]], 0
    for a, b in cand[1:]:
        if train_ep(a) >= max(RATIO * last, 1):
            chosen.append((a, b))
            last = train_ep(a)
    return chosen


def _tangent_stack(dW, parts_list):
    """tangents (K, *shape) for NAMES: one tangent per entry of parts_list (a tuple of PARTS keys whose dW enters)"""
    out = {}
    for k in dW:
        part = next(pk for pk, v in PARTS.items() if k in v)
        rows = [dW[k] if part in parts else torch.zeros_like(dW[k]) for parts in parts_list]
        if any(part in parts for parts in parts_list):
            out[k] = torch.stack(rows)
    return out


def node_pass(w0, dW, s, E, tangents, dmin, extra=None):
    """per item at W(s) = W_k + s dW: loss l (n, T) float64 and directional derivatives (len(tangents) [+ extra], n, T)
    along the tangents (tuples of PARTS keys; a tangent of ("ro",) alone is done from the states, no propagation) and
    along extra full-parameter directions (dict name -> list of tensors)"""
    p = {k: (w0[k] + s * dW[k] if k in dW else w0[k]) for k in NAMES}
    prop = [tp for tp in tangents if tp != ("ro",)]
    tans = _tangent_stack(dW, prop)
    if extra:
        n_x = len(next(iter(extra.values())))
        for k in NAMES:
            if k in extra:
                ex = torch.stack([e.float() for e in extra[k]])
                base = tans.get(k, torch.zeros((len(prop),) + tuple(p[k].shape)))
                tans[k] = torch.cat([base, ex])
            elif k in tans:
                tans[k] = torch.cat([tans[k], torch.zeros((n_x,) + tuple(p[k].shape))])
    n = len(E["goal"])
    Kp = len(prop) + (len(next(iter(extra.values()))) if extra else 0)
    l_all = np.zeros((n, T))
    d_all = np.zeros((len(tangents) + (Kp - len(prop)), n, T))
    for c0 in range(0, n, CHUNK):
        sl = slice(c0, min(n, c0 + CHUNK))
        inp = _inp(E, sl)
        with torch.no_grad():
            y, yd, u = forward(p, inp, tans if Kp else None)
            l, G = head_grad(y.double().cumsum(1), torch.as_tensor(E["goal"][sl]), torch.as_tensor(E["log_pg"][sl]),
                             dmin)
            l_all[sl] = l.numpy()
            j = 0
            for a, tp in enumerate(tangents):
                if tp == ("ro",):
                    ydr = u @ dW["classifier_readout.weight"].T + dW["classifier_readout.bias"]
                    d_all[a, sl] = (G * ydr.double().cumsum(1)).sum(-1).numpy()
                else:
                    d_all[a, sl] = (G * yd[j].double().cumsum(1)).sum(-1).numpy()
                    j += 1
            for b in range(Kp - len(prop)):
                d_all[len(tangents) + b, sl] = (G * yd[len(prop) + b].double().cumsum(1)).sum(-1).numpy()
    return l_all, d_all


def naive_loss():
    """naive's per-item loss (its goal marginal in place of the network's belief; the same loss) on M and G (cached)"""
    path = os.path.join(OUT, "naive_loss.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["main"], z["grad"]
    out = {}
    for which, n, seed in (("main", N_MAIN, SEED_MAIN), ("grad", 2 * N_GRAD_HALF, SEED_GRAD)):
        E = episodes(which)
        ck = torch.load(td.ckpt_path(td.WORLD, 0), map_location="cpu", weights_only=False)
        L, _ = td.world_tables(ck, E["pairs"])
        st = np.random.get_state()
        B = td.bayes_side(E["obs"], L, E["r"])
        np.random.set_state(st)
        ar = np.arange(n)
        assert np.allclose(B["log_jb"].astype(np.float64)[ar, :, E["goal"]], E["log_pg"])
        lnb = torch.as_tensor(B["log_nb"].astype(np.float64)[ar, :, E["goal"]])
        x, y = soft_clip(lnb.exp()), soft_clip(torch.as_tensor(E["log_pg"]).exp())
        out[which] = item_loss((x * torch.log(x / y) + y * torch.log(y / x)) / 2, DMIN).numpy()
    np.savez(path, **out)
    return out["main"], out["grad"]


def subset(E, sl):
    n = len(E["goal"])
    return {k: v[sl] for k, v in E.items() if hasattr(v, "shape") and v.ndim and len(v) == n}


def primal_loss(w, E):
    """per-item loss l (n, T) (unnormalized; float64 head) at weights w"""
    n = len(E["goal"])
    out = np.zeros((n, T))
    with torch.no_grad():
        for c0 in range(0, n, CHUNK):
            sl = slice(c0, min(n, c0 + CHUNK))
            y, _, _ = forward(w, _inp(E, sl))
            D = head_D(y.double().cumsum(1), torch.as_tensor(E["goal"][sl]), torch.as_tensor(E["log_pg"][sl]))
            out[sl] = item_loss(D, DMIN).numpy()
    return out


def ck_loss(run, E, which, S, w=None):
    """the per-item loss of checkpoint E on episode set `which` (cached per checkpoint: pairs share checkpoints)"""
    path = os.path.join(OUT, run, f"loss_{which}_{E:05d}.npz")
    if os.path.exists(path):
        return np.load(path)["l"]
    l = primal_loss(w if w is not None else load_w(run, E), S)
    np.savez(path + ".tmp.npz", l=l)
    os.replace(path + ".tmp.npz", path)
    return l


def do_pair(run, E0, E1, M, G):
    """everything for one checkpoint pair (see the docstring): M per-item losses at both ends (exact change); on G at
    W_k the two half-set gradients (autograd) and per item the first-order change along dW (and its read-out part) and
    the projection on the OTHER half's gradient; G's losses at both ends"""
    t0 = time.time()
    w0, w1 = load_w(run, E0), load_w(run, E1)
    train = trainable(run)
    dW = {k: (w1[k] - w0[k]) for k in train}
    frozen_moved = {k: float((w1[k] - w0[k]).abs().max()) for k in NAMES if k not in train}
    assert all(v == 0 for v in frozen_moved.values()), frozen_moved
    lM0, lM1 = ck_loss(run, E0, "main", M, w0), ck_loss(run, E1, "main", M, w1)
    halves = [slice(0, N_GRAD_HALF), slice(N_GRAD_HALF, 2 * N_GRAD_HALF)]
    gG = [autograd_grad(w0, G, train, DMIN, sl) for sl in halves]
    t_grad = time.time() - t0
    nG = 2 * N_GRAD_HALF
    lG0, dall, dro, dGo = np.zeros((nG, T)), np.zeros((nG, T)), np.zeros((nG, T)), np.zeros((nG, T))
    for h, sl in enumerate(halves):
        l, d = node_pass(w0, dW, 0., subset(G, sl), [("ro", "hh", "in"), ("ro",)], DMIN,
                         extra={k: [gG[1 - h][k]] for k in train})
        lG0[sl], dall[sl], dro[sl], dGo[sl] = l, d[0], d[1], d[2]
    lG1 = ck_loss(run, E1, "grad", G, w1)
    sq = lambda a, b: float(sum((a[k].double() * b[k].double()).sum() for k in train))   # noqa: E731
    dW64 = {k: v.double() for k, v in dW.items()}
    return dict(lM0=lM0, lM1=lM1, lG0=lG0, lG1=lG1, d0_all=dall.astype(np.float32), d0_ro=dro.astype(np.float32),
                dGo=dGo.astype(np.float32), run=run, E0=E0, E1=E1, x0=float(train_ep(E0)), x1=float(train_ep(E1)),
                updates=int(train_ep(E1) - train_ep(E0)), dmin=DMIN, n_main=N_MAIN, n_grad_half=N_GRAD_HALF,
                g1g2=sq(gG[0], gG[1]), g1g1=sq(gG[0], gG[0]), g2g2=sq(gG[1], gG[1]),
                g1dW=sq(gG[0], dW64), g2dW=sq(gG[1], dW64),
                dW2_parts=np.array([sum(float((dW64[k] ** 2).sum()) for k in PARTS[pk] if k in dW64) for pk in PARTS]),
                w2_parts=np.array([sum(float((w0[k].double() ** 2).sum()) for k in PARTS[pk]) for pk in PARTS]),
                parts=np.array(list(PARTS)), t_grad=t_grad, t_total=time.time() - t0)


def data(run, minutes):
    M, G = episodes("main"), episodes("grad")
    d = os.path.join(OUT, run)
    os.makedirs(d, exist_ok=True)
    t_start = time.time()
    todo = [(a, b) for a, b in pair_list(run) if not os.path.exists(os.path.join(d, f"pair_{a:05d}_{b:05d}.npz"))]
    print(f"{run}: {len(todo)} pairs to do", flush=True)
    for a, b in todo:
        if time.time() - t_start > 60 * minutes:
            print("time budget reached; rerun to continue", flush=True)
            return False
        out = do_pair(run, a, b, M, G)
        path = os.path.join(d, f"pair_{a:05d}_{b:05d}.npz")
        np.savez(path + ".tmp.npz", **out)
        os.replace(path + ".tmp.npz", path)
        nM, nG = N_MAIN * T * R, 2 * N_GRAD_HALF * T * R
        fo = out["d0_all"].sum() / nG
        fo_ag = (out["g1dW"] + out["g2dW"]) / 2
        print(f"{time.strftime('%H:%M:%S')} {run} {a}->{b}: L {out['lM0'].sum() / nM:.5f}, exact dL main "
              f"{(out['lM1'] - out['lM0']).sum() / nM:+.6f}, grad set {(out['lG1'] - out['lG0']).sum() / nG:+.6f}, "
              f"first-order {fo:+.6f} (autograd {fo_ag:+.6f}), GD |g|^2 {out['dGo'].sum() / nG:.4g} "
              f"({out['t_total']:.0f} s)", flush=True)
    return True


def profile(run, pairs, n=500):
    """the loss along the straight segment W_k + s dW (s on PROFILE_S) on the first n episodes of M, with the slopes
    (first-order, along dW) at s = 0 and 1: why the first-order prediction fails"""
    M = subset(episodes("main"), slice(0, n))
    for a, b in pairs:
        path = os.path.join(OUT, run, f"profile_{a:05d}_{b:05d}.npz")
        if os.path.exists(path):
            continue
        w0, w1 = load_w(run, a), load_w(run, b)
        dW = {k: (w1[k] - w0[k]) for k in trainable(run)}
        L = []
        for s in PROFILE_S:
            if s in (0., 1.):
                l, d = node_pass(w0, dW, s, M, [("ro", "hh", "in")], DMIN)
                if s == 0.:
                    d0 = d[0]
                else:
                    d1 = d[0]
            else:
                l = primal_loss({k: (w0[k] + s * dW[k] if k in dW else w0[k]) for k in NAMES}, M)
            L.append(l)
        np.savez(path, s=np.array(PROFILE_S), l=np.array(L), d0=d0, d1=d1, E0=a, E1=b, n=n)
        nr = n * T * R
        print(f"{run} {a}->{b}: L(s) {np.round(np.array(L).sum((1, 2)) / nr, 5)}; slope 0 {d0.sum() / nr:+.5f}, "
              f"slope 1 {d1.sum() / nr:+.5f}", flush=True)


# ═══════════════════════════════════════════════════════════════════
# Checks
# ═══════════════════════════════════════════════════════════════════

def check(run, E0):
    """(1) forward == td_analysis's replay; (2) forward-mode tangents vs autograd directional derivatives and central
    finite differences; (3) the training loss on M vs the run's logged training-batch loss at the next update"""
    M = episodes("main")
    eps = td.checkpoints(run)
    E1 = eps[eps.index(E0) + 1]
    w0, w1 = load_w(run, E0), load_w(run, E1)
    train = trainable(run)
    dW = {k: w1[k] - w0[k] for k in train}
    sub = {k: v[:64] for k, v in M.items() if hasattr(v, "shape") and len(v) == N_MAIN}
    inp = _inp(sub, slice(0, 64))
    # (1) replay identity: nn.LSTM on the same inputs
    lstm = torch.nn.LSTM(1000, 1000, batch_first=True)
    lstm.load_state_dict({k[5:]: w0[k] for k in NAMES if k.startswith("LSTM.")})
    with torch.no_grad():
        x = torch.relu(inp @ w0["classifier_readin.weight"].T + w0["classifier_readin.bias"])
        u_ref, _ = lstm(x, (w0["STM"].reshape(1, 1, -1).expand(1, 64, -1).contiguous(),
                            w0["LTM"].reshape(1, 1, -1).expand(1, 64, -1).contiguous()))
        _, _, u = forward(w0, inp)
    print(f"(1) hand LSTM vs nn.LSTM: max |du| {(u - u_ref).abs().max():.1e}")
    # (2) tangents vs autograd and finite differences, on the total loss of 64 episodes
    dmin = DMIN
    goal, lpg = torch.as_tensor(sub["goal"]), torch.as_tensor(sub["log_pg"])

    def Ltot(p):
        y, _, _ = forward(p, inp)
        return item_loss(head_D(y.double().cumsum(1), goal, lpg), dmin).sum()
    q = {k: (v.clone().requires_grad_(True) if k in train else v) for k, v in w0.items()}
    with torch.enable_grad():
        gs = torch.autograd.grad(Ltot(q), [q[k] for k in train])
    g = dict(zip(train, gs))
    ref = sum(float((g[k].double() * dW[k].double()).sum()) for k in train)
    ref_hh = float((g["LSTM.weight_hh_l0"].double() * dW["LSTM.weight_hh_l0"].double()).sum()) if \
        "LSTM.weight_hh_l0" in train else 0.
    ref_ro = sum(float((g[k].double() * dW[k].double()).sum()) for k in PARTS["ro"])
    l, d = node_pass(w0, dW, 0., sub, [("ro", "hh", "in"), ("hh",), ("ro",)] if ref_hh else
                     [("ro", "hh", "in"), ("ro",)], dmin, extra={k: [g[k]] for k in train})
    print(f"(2) along dW: forward-mode {d[0].sum():+.6e} vs autograd {ref:+.6e} (rel {abs(d[0].sum() / ref - 1):.1e})")
    if ref_hh:
        print(f"    along dW_hh: {d[1].sum():+.6e} vs {ref_hh:+.6e}; along dW_ro: {d[2].sum():+.6e} vs {ref_ro:+.6e}")
    gg = sum(float((g[k].double() ** 2).sum()) for k in train)
    print(f"    along g itself: forward-mode {d[-1].sum():+.6e} vs |g|^2 {gg:+.6e} (rel {abs(d[-1].sum() / gg - 1):.1e})")
    with torch.no_grad():
        for eps_ in (1e-2, 1e-3):
            pp = {k: (w0[k] + eps_ * dW[k] if k in dW else w0[k]) for k in NAMES}
            pm = {k: (w0[k] - eps_ * dW[k] if k in dW else w0[k]) for k in NAMES}
            fd = (float(Ltot(pp)) - float(Ltot(pm))) / (2 * eps_)
            print(f"    central difference eps {eps_:g}: {fd:+.6e}")
    # (3) loss level vs the logged training loss (batch of 8000 training episodes, AMP, at the first update after E0)
    on = sorted(glob.glob(os.path.join(td.TD, f"{run}_s{td.SEED}", "online_*.npz")))
    ep_on = np.concatenate([np.load(f)["episode"] for f in on])
    loss_on = np.concatenate([np.load(f)["loss"] for f in on])
    i = np.searchsorted(ep_on, E0 + 1)
    l = primal_loss(w0, M)
    print(f"(3) training loss on M at ep {E0}: {l.sum() / (N_MAIN * T * R):.5f}; logged batch loss at the next 5 updates "
          f"{np.round(loss_on[i:i + 5], 5)} (episodes {ep_on[i:i + 5]})")


# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════

WIN_EDGES = (0, 20, 40, 80, 160, 320, 640, 1040, 1600, 2400, 4000, 6400, 9600, 15200, 24800, 39996)  # log-even
POOLED = ((40, 160), (0, 320), (80, 320), (320, 1040), (320, 1600), (1600, 9600), (1600, 4000), (4000, 15200),
          (15200, 39996), (80, 39996))                          # pooled windows quoted in the docstring
EARLY = ((40, 80), (40, 100), (60, 100), (60, 140), (80, 140), (80, 160), (80, 180), (80, 200), (100, 160), (100, 180),
         (100, 200), (120, 240), (140, 240), (160, 260), (160, 280), (160, 300), (160, 320), (180, 300), (180, 320),
         (200, 320))                                            # early windows, 1.6-2.5x wide (the oscillation)
N_BLK_M, N_BLK_G = 20, 10                                       # jackknife blocks of 100 episodes
MAX_RAW = 50000                                                 # per-item losses of every checkpoint (to the end)


def raw_ep(x):
    """the raw episode of training episode x (x divisible by 4)"""
    return int(x) * 5 // 4


def losses(run, minutes, max_raw=MAX_RAW):
    """per-item losses of every checkpoint up to max_raw on both episode sets (cached; for the window sums; all 114
    checkpoints since the 11-01 fix, previously up to raw 13,000)"""
    M, G = episodes("main"), episodes("grad")
    os.makedirs(os.path.join(OUT, run), exist_ok=True)
    t0 = time.time()
    for E in td.checkpoints(run):
        if time.time() - t0 > 60 * minutes:
            print("time budget reached; rerun to continue", flush=True)
            return False
        if E <= max_raw:
            w = None
            for which, S in (("main", M), ("grad", G)):
                if not os.path.exists(os.path.join(OUT, run, f"loss_{which}_{E:05d}.npz")):
                    w = w if w is not None else load_w(run, E)
                    ck_loss(run, E, which, S, w)
    print(f"{run}: losses done ({time.time() - t0:.0f} s)", flush=True)
    return True


def _blocks():
    """block id of every episode of M (0-19) and G (20-29) and the groups"""
    M, G = episodes("main"), episodes("grad")
    bM = np.arange(N_MAIN) * N_BLK_M // N_MAIN
    bG = N_BLK_M + np.arange(2 * N_GRAD_HALF) * N_BLK_G // (2 * N_GRAD_HALF)
    return M["grp"], G["grp"], bM, bG


def _bsum(v, grp, blk, nb):
    """(nb, N_GRP) sums of the per-item values v (n, T) by (block, group)"""
    out = np.zeros((nb, N_GRP))
    np.add.at(out, (np.broadcast_to(blk[:, None], grp.shape).ravel(), grp.ravel()), v.ravel())
    return out


def summarize(run):
    """per checkpoint, pair and window (see the docstring); everything as (block, group) sums so shares and their
    jackknife SEs are ratios of sums; saved to series_<run>.npz"""
    gM, gG, bM, bG = _blocks()
    nb = N_BLK_M + N_BLK_G
    union = lambda lm, lg: _bsum(lm, gM, bM, nb) + _bsum(lg, gG, bG, nb)          # noqa: E731
    S = {"items": union(np.ones((N_MAIN, T)), np.ones((2 * N_GRAD_HALF, T)))}
    cks = [E for E in td.checkpoints(run) if all(os.path.exists(os.path.join(OUT, run, f"loss_{w}_{E:05d}.npz"))
                                                for w in ("main", "grad"))]
    lk = lambda E, w: np.load(os.path.join(OUT, run, f"loss_{w}_{E:05d}.npz"))["l"]   # noqa: E731
    S["ck_E"] = np.array(cks)
    S["ck_L"] = np.array([union(lk(E, "main"), lk(E, "grad")) for E in cks])        # (n_ck, nb, N_GRP) raw sums
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, run, "pair_*.npz"))):
        z = np.load(f)
        nh = N_GRAD_HALF
        h1, h2 = slice(0, nh), slice(nh, 2 * nh)
        r = {k: z[k] for k in ("E0", "E1", "x0", "x1", "updates", "g1g2", "g1g1", "g2g2", "g1dW", "g2dW", "dW2_parts",
                               "w2_parts")}
        r["L0"] = union(z["lM0"], z["lG0"])
        r["dL"] = union(z["lM1"] - z["lM0"], z["lG1"] - z["lG0"])
        r["dL_M"] = _bsum(z["lM1"] - z["lM0"], gM, bM, nb)
        r["dL_G"] = _bsum(z["lG1"] - z["lG0"], gG, bG, nb)
        r["F"] = _bsum(z["d0_all"].astype(np.float64), gG, bG, nb)
        r["F_ro"] = _bsum(z["d0_ro"].astype(np.float64), gG, bG, nb)
        r["EG"] = np.stack([np.bincount(gG[s_].ravel(), z["dGo"][s_].astype(np.float64).ravel(), N_GRP)
                            for s_ in (h1, h2)])                                      # (2 halves, N_GRP) raw sums
        rows.append(r)
    for k in rows[0]:
        S["pr_" + k] = np.array([r[k] for r in rows])
    np.savez(os.path.join(OUT, f"series_{run}.npz"), **S)
    return S


def share_jk(num):
    """shares num[..., q].sum(blocks) / num.sum(blocks, groups) over all groups and their block-jackknife SE;
    num (nb, N_GRP) -> (share (N_GRP,), se (N_GRP,))"""
    nb = len(num)
    tot = num.sum(0)
    sh = tot / tot.sum()
    loo = (tot[None] - num) / (tot.sum() - num.sum(1))[:, None]
    return sh, np.sqrt((nb - 1) / nb * ((loo - loo.mean(0)) ** 2).sum(0))


def probe_shares(run):
    """the same realized window shares on the 10,000 probe episodes (test pairs, never trained; analysis/probes.npz),
    from the beliefs stored at every checkpoint (analysis/<run>/ck_*.npz "logq", float16, renormalized): the training
    loss of the goal variable (= the stream) per item; fifths of the goal variable's own entanglement within each step
    (2,000 per fifth and step), raw ("none") and within 10 strata of naive's per-item loss ("naive", 200 per matched
    fifth, stratum and step); sums per environment (50) for an environment jackknife ("env_<grouping>" (114
    checkpoints = "edges", 50, N_GRP), so any window between checkpoints can be replicated); "L", "dL", "L0": the
    realized and loss shares of the 15 windows of WIN_EDGES; cached in probe_shares_<run>.npz (until 11-02 only the
    window edges were cached)"""
    path = os.path.join(OUT, f"probe_shares_{run}.npz")
    xs = [int(x) for x in train_ep(td.checkpoints(run))]
    if os.path.exists(path):
        z = dict(np.load(path))
        if "env_naive" in z and list(z["edges"]) == xs and tuple(z.get("win_edges", ())) == WIN_EDGES:
            return z
    P = np.load(os.path.join(td.TD, "analysis", "probes.npz"))
    goal, env = P["goal"], P["env"]
    n = len(goal)
    ar = np.arange(n)
    ent = P["ent"][ar, :, goal]
    y = soft_clip(torch.as_tensor(P["log_jb"].astype(np.float64)[ar, :, goal]).exp())

    def loss(lq):
        lq = lq - lq.logsumexp(-1, keepdim=True)
        x = soft_clip(lq.exp())
        return item_loss((x * torch.log(x / y) + y * torch.log(y / x)) / 2, DMIN).numpy()
    naive = loss(torch.as_tensor(P["log_nb"].astype(np.float64)[ar, :, goal]))
    grps = {"none": groups(ent), "naive": strat_groups(ent, [naive], [N_STRATA])}
    n_env = int(env.max()) + 1
    out = dict(edges=np.array(xs), win_edges=np.array(WIN_EDGES), n_env=n_env)
    idx = {k: (env[:, None] * N_GRP + g).ravel() for k, g in grps.items()}
    for k in grps:
        out["env_" + k] = np.zeros((len(xs), n_env, N_GRP))
    for c, x in enumerate(xs):
        l = loss(torch.as_tensor(np.load(os.path.join(td.TD, "analysis", run, f"ck_{raw_ep(x):05d}.npz"))["logq"]
                                 .astype(np.float64)[ar, :, goal]))
        for k in grps:
            out["env_" + k][c] = np.bincount(idx[k], l.ravel(), n_env * N_GRP).reshape(n_env, N_GRP)
    for k, g in grps.items():
        out["naive_" + k] = np.bincount(g.ravel(), naive.ravel(), N_GRP) / (np.bincount(g.ravel(), None, N_GRP) * R)
    Lq = out["env_none"][[xs.index(x) for x in WIN_EDGES]].sum(1)
    out.update(L=Lq, dL=-(Lq[1:] - Lq[:-1]) / -(Lq[1:] - Lq[:-1]).sum(1, keepdims=True),
               L0=Lq[:-1] / Lq[:-1].sum(1, keepdims=True))
    np.savez(path, **out)
    return out


def windows(S):
    """per window [a, b) of WIN_EDGES (training episodes): the exact realized change of every group over the whole
    window from the checkpoint losses, the loss at its start, and the first-order / GD terms pooled (ratio of sums)
    over the sampled pairs whose interval starts in it"""
    ckx = train_ep(S["ck_E"])
    out = {k: [] for k in ("a", "b", "xc", "n_pairs", "dL", "dL_se", "L0", "L0_se", "ex", "ex_se", "F", "F_se", "EG",
                           "EG_se", "EG_h", "dLtot", "Ltot0", "Ftot", "dLtot_pairs", "updates")}
    for a, b in zip(WIN_EDGES[:-1], WIN_EDGES[1:]):
        ia, ib = int(np.where(ckx == a)[0][0]), int(np.where(ckx == b)[0][0])
        dLb = S["ck_L"][ib] - S["ck_L"][ia]                                           # (nb, N_GRP)
        sel = (S["pr_x0"] >= a) & (S["pr_x0"] < b)
        Fb = S["pr_F"][sel].sum(0)
        EGh = S["pr_EG"][sel].sum(0)                                                  # (2, N_GRP)
        sh, se = share_jk(-dLb)
        l0, l0se = share_jk(S["ck_L"][ia])
        _, _, ex, exse = excess_jk(S["ck_L"][ia], S["ck_L"][ib])
        with np.errstate(invalid="ignore", divide="ignore"):                         # windows past the pairs: NaN
            f, fse = share_jk(-Fb)
            eg = EGh.sum(0) / EGh.sum()
            egh = EGh / EGh.sum(1, keepdims=True)
        for k, v in (("a", a), ("b", b), ("xc", np.sqrt(max(a, 10) * b)), ("n_pairs", int(sel.sum())), ("dL", sh),
                     ("dL_se", se), ("L0", l0), ("L0_se", l0se), ("ex", ex), ("ex_se", exse), ("F", f), ("F_se", fse),
                     ("EG", eg), ("EG_se", np.abs(egh[0] - egh[1]) / 2), ("EG_h", egh), ("dLtot", dLb.sum()),
                     ("Ltot0", S["ck_L"][ia].sum()), ("Ftot", Fb.sum()), ("dLtot_pairs", S["pr_dL"][sel].sum()),
                     ("updates", b - a)):
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


def excess_jk(A, B):
    """a window from the block sums of the loss at its start A and end B (nb, N_GRP): realized share of the decrease,
    loss share at the start, excess = realized share - loss share, and the block-jackknife SE of the excess (one block
    left out of BOTH terms: the SE of the difference)"""
    d = A - B
    rs, ls = d.sum(0) / d.sum(), A.sum(0) / A.sum()
    loo = (d.sum(0)[None] - d) / (d.sum() - d.sum(1))[:, None] - (A.sum(0)[None] - A) / (A.sum() - A.sum(1))[:, None]
    nb = len(A)
    return rs, ls, rs - ls, np.sqrt((nb - 1) / nb * ((loo - loo.mean(0)) ** 2).sum(0))


# ═══════════════════════════════════════════════════════════════════
# Difficulty-matched control: entanglement fifths inside strata of a difficulty variable
# ═══════════════════════════════════════════════════════════════════

N_EP = N_MAIN + 2 * N_GRAD_HALF                                 # the union: M (2,000) then G (1,000)
N_STRATA = 10
MATCHINGS = {"naive": (("naive",), (N_STRATA,)),              # name: (matching variables, strata per variable)
             "naive5": (("naive",), (5,)),
             "naive20": (("naive",), (20,)),
             "fr": (("fr",), (N_STRATA,)),
             "H": (("H",), (N_STRATA,)),
             "naiveH": (("naive", "H"), (5, 5))}
GROUPINGS = ("none",) + tuple(MATCHINGS)
GROUPING_LABEL = {"none": "raw entanglement fifths", "naive": "matched on naive's loss (10 strata)",
                  "naive5": "naive's loss, 5 strata", "naive20": "naive's loss, 20 strata",
                  "fr": "matched on FR (10 strata)", "H": "matched on Bayes's entropy (10 strata)",
                  "naiveH": "naive's loss x Bayes's entropy (5 x 5)"}


def match_vars():
    """per item (i, t) of the union (N_EP, T): the goal variable's own entanglement "ent" and the matching variables:
    naive's per-item loss "naive" (naive's goal marginal in place of the network's belief, the same loss), the goal
    variable's FR "fr" (KL(Bayes || naive) of its marginal, forward, exact: main.utils.factorization_regret through
    td_analysis.bayes_side; the same two beliefs as naive's loss compares) and Bayes's goal entropy "H" (nats)"""
    M, G = episodes("main"), episodes("grad")
    nl_m, nl_g = naive_loss()
    goal = np.concatenate([M["goal"], G["goal"]])
    lpg = np.concatenate([M["log_pg"], G["log_pg"]])
    return dict(ent=np.concatenate([M["ent_goal"], G["ent_goal"]]), naive=np.concatenate([nl_m, nl_g]),
                fr=np.concatenate([M["fr"], G["fr"]])[np.arange(N_EP), :, goal], H=-(np.exp(lpg) * lpg).sum(-1))


def strat_groups(v, by, n_strata):
    """(n, T) groups: 0 = step 1; 1-5 = fifths of v (rank-based) inside every cell of the nested strata of the
    variables `by` (n_strata each, rank-based), within each step 2-30; no `by`: plain fifths of v"""
    n = len(v)
    g = np.zeros(v.shape, int)
    for t in range(1, T):
        cells = [np.arange(n)]
        for b, ns in zip(by, n_strata):
            new = []
            for c in cells:
                o = c[np.argsort(b[c, t], kind="stable")]
                new += [o[s * len(o) // ns:(s + 1) * len(o) // ns] for s in range(ns)]
            cells = new
        for c in cells:
            o = c[np.argsort(v[c, t], kind="stable")]
            for k in range(5):
                g[o[k * len(o) // 5:(k + 1) * len(o) // 5], t] = k + 1
    return g


def grouping(name, V=None):
    """(N_EP, T) groups of the union: "none" = the entanglement fifths of the main analysis (ranked within each step
    and episode set); otherwise entanglement fifths matched on MATCHINGS[name] (ranked within each step of the union)"""
    if name == "none":
        return np.concatenate([episodes("main")["grp"], episodes("grad")["grp"]])
    V = V if V is not None else match_vars()
    by, ns = MATCHINGS[name]
    return strat_groups(V["ent"], [V[b] for b in by], ns)


def ep_blocks():
    """block id (0-29) of every episode of the union: 20 blocks of 100 in M, then 10 in G"""
    return np.concatenate([np.arange(N_MAIN) * N_BLK_M // N_MAIN,
                           N_BLK_M + np.arange(2 * N_GRAD_HALF) * N_BLK_G // (2 * N_GRAD_HALF)])


def match_data(run):
    """per checkpoint and grouping, sums of the per-item loss by (block, group) "blk_<g>" (n_ck, 30, N_GRP); for
    "none" and "naive" also by (episode, group) "ep_<g>" (n_ck, N_EP, N_GRP) for the bootstrap, with naive's loss
    "nv_<g>" and the item counts "cnt_<g>" (N_EP, N_GRP); cached in match_<run>.npz (rebuilt when checkpoints were
    added)"""
    path = os.path.join(OUT, f"match_{run}.npz")
    cks = [E for E in td.checkpoints(run) if all(os.path.exists(os.path.join(OUT, run, f"loss_{w}_{E:05d}.npz"))
                                                for w in ("main", "grad"))]
    if os.path.exists(path):
        z = dict(np.load(path))
        if list(z["ck_E"]) == cks:
            return z
    V = match_vars()
    l = np.array([np.concatenate([np.load(os.path.join(OUT, run, f"loss_{w}_{E:05d}.npz"))["l"]
                                  for w in ("main", "grad")]) for E in cks])        # (n_ck, N_EP, T)
    blk = ep_blocks()
    out = dict(ck_E=np.array(cks))
    for name in GROUPINGS:
        oh = (grouping(name, V)[..., None] == np.arange(N_GRP)).astype(np.float64)   # (N_EP, T, N_GRP)
        ep = np.einsum("cet,etq->ceq", l, oh)
        out["blk_" + name] = np.stack([ep[:, blk == b].sum(1) for b in range(N_BLK_M + N_BLK_G)], 1)
        if name in ("none", "naive"):
            out["ep_" + name] = ep
            out["nv_" + name] = np.einsum("et,etq->eq", V["naive"], oh)
            out["cnt_" + name] = oh.sum(1)
    np.savez(path, **out)
    return out


def match_windows(Z, wins, name):
    """per window (a, b) in training episodes: realized share, loss share, excess and its jackknife SE (blocks; the
    SE of the difference), each (n_win, N_GRP); the excess from M alone and from G alone; the total decrease (loss
    units, per item and value)"""
    ckx = train_ep(Z["ck_E"])
    Bk = Z["blk_" + name]
    out = {k: [] for k in ("rs", "ls", "ex", "se", "ex_M", "ex_G", "dL")}
    for a, b in wins:
        ia, ib = int(np.where(ckx == a)[0][0]), int(np.where(ckx == b)[0][0])
        rs, ls, ex, se = excess_jk(Bk[ia], Bk[ib])
        for k, v in (("rs", rs), ("ls", ls), ("ex", ex), ("se", se)):
            out[k].append(v)
        for k, s in (("ex_M", slice(0, N_BLK_M)), ("ex_G", slice(N_BLK_M, None))):
            A, B = Bk[ia, s].sum(0), Bk[ib, s].sum(0)
            out[k].append((A - B) / (A - B).sum() - A / A.sum())
        out["dL"].append((Bk[ia] - Bk[ib]).sum() / (N_EP * T * R))
    return {k: np.array(v) for k, v in out.items()}


def probe_windows(pr, name, wins=None):
    """the probes' excess per window of WIN_EDGES (or the pooled windows `wins` with edges in WIN_EDGES) with the
    environment-jackknife SE (50 environments): (ex (n_win, N_GRP), se)"""
    E_ = pr["env_" + name]
    edges = list(pr["edges"])
    wins = wins or list(zip(WIN_EDGES[:-1], WIN_EDGES[1:]))
    res = [excess_jk(E_[edges.index(a)], E_[edges.index(b)]) for a, b in wins]
    return np.array([r[2] for r in res]), np.array([r[3] for r in res])


# ═══════════════════════════════════════════════════════════════════
# Timing: episode (= environment) bootstrap, crossings, timeline export
# ═══════════════════════════════════════════════════════════════════

N_BOOT, BOOT_SEED = gap.N_BOOT, gap.BOOT_SEED   # 1,000, 20261103: td_tt_gap's seed since 11-02 (was 20261101)
SPAN = 2.                       # timeline windows: [x / sqrt(SPAN), x * sqrt(SPAN)] around every checkpoint


def boot_counts(n=N_EP, n_boot=N_BOOT, seed=BOOT_SEED):
    """multinomial resampling counts (n_boot, n) of the 3,000 episodes; every episode has its own pair of variables
    (environment), so this is the environment bootstrap"""
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1. / n), size=n_boot).astype(np.float64)


def centred_windows(ckx, span=SPAN):
    """for every checkpoint k a window [a_k, b_k] of cached checkpoints around it: a_k the checkpoint below x_k nearest
    (in log x; the initial network counts as 10) to x_k / sqrt(span), b_k the one above nearest to x_k * sqrt(span);
    -1 at the first and last checkpoint"""
    lx = np.log(np.maximum(ckx, X0).astype(float))
    a, b = np.full(len(ckx), -1), np.full(len(ckx), -1)
    for k in range(len(ckx)):
        lo, hi = np.where(ckx < ckx[k])[0], np.where(ckx > ckx[k])[0]
        if len(lo) and len(hi):
            a[k] = lo[np.argmin(np.abs(lx[lo] - lx[k] + np.log(span) / 2))]
            b[k] = hi[np.argmin(np.abs(lx[hi] - lx[k] - np.log(span) / 2))]
    return a, b


def centred_ties(ckx, span=SPAN):
    """checkpoints whose centred window (centred_windows) has an exact log tie at an edge: another checkpoint as far
    from x / sqrt(span) (below) or x sqrt(span) (above) in log x as the one argmin kept (in integers: a a' span = x^2,
    b b' = span x^2), so floating-point rounding decided -> [(x, kept window, the other window)] in training episodes"""
    a, b = centred_windows(ckx, span)
    xv = np.maximum(ckx, X0).astype(np.int64)
    out = []
    for k in np.where(a >= 0)[0]:
        for c in range(len(ckx)):
            if c < k and c != a[k] and xv[c] * xv[a[k]] * span == xv[k] ** 2:
                out.append((int(ckx[k]), (int(ckx[a[k]]), int(ckx[b[k]])), (int(ckx[c]), int(ckx[b[k]]))))
            if c > k and c != b[k] and xv[c] * xv[b[k]] == span * xv[k] ** 2:
                out.append((int(ckx[k]), (int(ckx[a[k]]), int(ckx[b[k]])), (int(ckx[a[k]]), int(ckx[c]))))
    return out


def crossing(x, v, level, side):
    """the training episode after which v stays past `level` on `side` (+1 above, -1 below) up to its last valid
    point (the end of training), log-interpolated between the last point not past it and the next; NaN if the last
    valid point is not past it (never); the first valid x if every valid point is past it"""
    ok = np.isfinite(v)
    xs, vs = np.maximum(np.asarray(x, float)[ok], X0), np.asarray(v, float)[ok]
    past = side * (vs - level) > 0
    if not len(vs) or not past[-1]:
        return np.nan
    if past.all():
        return float(xs[0])
    j = int(np.where(~past)[0][-1])
    f = (level - vs[j]) / (vs[j + 1] - vs[j])
    return float(np.exp(np.log(xs[j]) + f * np.log(xs[j + 1] / xs[j])))


def crossing_ci(x, V, level, side):
    """5-95% interval of the crossing over bootstrap curves V (n_boot, n) (level scalar or (n_boot,)); a resample that
    never stays past counts as later than any crossing (hi = NaN when more than 5% never do); also that fraction"""
    lev = np.broadcast_to(np.asarray(level, float), (len(V),))
    c = np.array([crossing(x, v, lv, side) for v, lv in zip(V, lev)])
    never = float(np.isnan(c).mean())
    c = np.sort(np.where(np.isnan(c), np.inf, c))                    # order statistics (no inf arithmetic)
    lo, hi = c[int(.05 * (len(c) - 1))], c[int(np.ceil(.95 * (len(c) - 1)))]
    return float(lo) if np.isfinite(lo) else np.nan, float(hi) if np.isfinite(hi) else np.nan, never


def loss_curves(Z, C, q, name="none"):
    """the loss per item and value of group q at every checkpoint (n_ck,), its bootstrap curves (n_boot, n_ck), and
    naive's level for that group (full, per resample)"""
    ep, nv, cnt = Z["ep_" + name][:, :, q], Z["nv_" + name][:, q], Z["cnt_" + name][:, q]
    full = ep.sum(1) / (cnt.sum() * R)
    boot = (C @ ep.T) / ((C @ cnt) * R)[:, None]
    return full, boot, nv.sum() / (cnt.sum() * R), (C @ nv) / ((C @ cnt) * R)


def excess_curves(Z, C, q, name, span=SPAN):
    """the excess of group q over the window around every checkpoint (centred_windows), full (n_ck,) and bootstrap
    (n_boot, n_ck); NaN where no window"""
    ckx = train_ep(Z["ck_E"])
    a, b = centred_windows(ckx, span)
    ep = Z["ep_" + name]
    full, boot = np.full(len(ckx), np.nan), np.full((len(C), len(ckx)), np.nan)
    for k in np.where(a >= 0)[0]:
        A, D = ep[a[k]], ep[a[k]] - ep[b[k]]
        full[k] = D[:, q].sum() / D.sum() - A[:, q].sum() / A.sum()
        Ab, Db = C @ A, C @ D
        boot[:, k] = Db[:, q] / Db.sum(1) - Ab[:, q] / Ab.sum(1)
    return full, boot


TIMELINE = (  # (name, family, kind, grouping, group)
    ("loss, most entangled fifth", "answers", "loss", "none", 5),
    ("loss, least entangled fifth", "answers", "loss", "none", 1),
    ("learning excess, most entangled fifth", "learning", "excess", "none", 5),
    ("learning excess, matched on naive", "learning", "excess", "naive", 5))
EXCESS_SIDE = {"trained": 1, "echo": -1}      # the side of each run's headline effect (surplus / deficit)


def timeline():
    """the topic's headline measures at every checkpoint for both runs with 5-95% bootstrap bands and crossings;
    writes timeline.npz (see TIMELINE EXPORT in the docstring) and returns its dict"""
    C = boot_counts()
    rows = []
    for run in RUNS:
        Z = match_data(run)
        x = train_ep(Z["ck_E"]).astype(float)
        for name, fam, kind, gname, q in TIMELINE:
            if kind == "loss":
                full, boot, nv, nv_b = loss_curves(Z, C, q, gname)
                lev, lev_b, side, bayes, naive = nv, nv_b, -1, BAYES_LOSS, nv
            else:
                full, boot = excess_curves(Z, C, q, gname)
                lev, lev_b, side, bayes, naive = 0., 0., EXCESS_SIDE[run], np.nan, np.nan
            half = crossing(x, full, lev, side)
            h_lo, h_hi, never = crossing_ci(x, boot, lev_b, side)
            if np.isnan(half) or never > .05:          # never, or not established (> 5% of resamples never cross)
                half = h_lo = h_hi = np.nan
            with warnings.catch_warnings():                     # all-NaN columns: no window at the first / last
                warnings.simplefilter("ignore", RuntimeWarning)
                lo, hi = np.nanpercentile(boot, [5, 95], axis=0)
            rows.append(dict(name=name, run=run, family=fam, value=full, lo=lo, hi=hi, bayes=bayes, naive=naive,
                             level=lev, half=half, half_lo=h_lo, half_hi=h_hi, never=never, side=side))
    xs = {len(r["value"]) for r in rows}
    assert len(xs) == 1, "both runs need the same checkpoints"
    out = dict(x=x, names=np.array([r["name"] for r in rows]), run=np.array([r["run"] for r in rows]),
               family=np.array([r["family"] for r in rows]))
    for k in ("value", "lo", "hi"):
        out[k] = np.array([r[k] for r in rows])
    for k in ("bayes", "naive", "level", "half", "half_lo", "half_hi"):
        out[k] = np.array([r[k] for r in rows], float)
    np.savez(os.path.join(OUT, "timeline.npz"), **out)
    out.update(never=np.array([r["never"] for r in rows]), side=np.array([r["side"] for r in rows]))
    return out


SYNTHESIS = (("loss, most entangled fifth", 5), ("loss, least entangled fifth", 1))
SYN_SAMPLE = ("training-distribution episodes: 3,000 (2,000 main + 1,000 gradient set; each draws its own pair of "
              "variables: 2,957 distinct, none of the 50 probe environments), resampled by episode")
SYN_NOTE = ("training loss per item and value on the fifth (goal variable's own entanglement, ranked within steps "
            "2-30); naive = naive's loss on that fifth, resampled with the episodes; Bayes = the loss floor "
            "sqrt(1e-8) = 1e-4 (its own belief scores D = 0), the same in every resample; gap fraction = "
            "(naive - loss) / (naive - 1e-4); unpaired: 1,000 multinomial resamples of the 3,000 episodes, "
            "seed 20261103")


def syn_note(Z, q):
    """the row-specific part of the synthesis note of fifth q (5 or 1) of one run, from the full-sample curves: the
    most entangled fifth crosses naive's level first only because naive's loss is higher on it (the network's own
    loss is higher there too); a fifth whose loss stays above naive's level at every checkpoint says so"""
    x, C1 = gap.train_x(Z["ck_E"]), np.ones((1, N_EP))
    (l5, _, n5, _), (l1, _, n1, _) = loss_curves(Z, C1, 5), loss_curves(Z, C1, 1)
    lq, nq = (l5, n5) if q == 5 else (l1, n1)
    parts = []
    if q == 5:
        c5, c1 = crossing(x, l5, n5, -1), crossing(x, l1, n1, -1)
        assert np.isfinite(c5) and (np.isnan(c1) or c5 < c1), "q5 must cross naive's level before q1"
        parts.append(f"crosses before the least entangled fifth because naive's loss is {n5 / n1:.1f}x higher on this "
                     f"fifth; the network's own loss is higher here (q5/q1 > 1 at {int((l5 > l1).sum())} of {len(x)} "
                     "checkpoints)")
    if (lq > nq).all():
        parts.append(f"stays above naive's level ({f'{lq[-1]:.3f}'.lstrip('0')} vs {f'{nq:.4f}'.lstrip('0')} at 40k)")
    return "; ".join(parts)


def synthesis():
    """-> through_training/entangled_learning/synthesis.npz (td_tt_gap.export: gap fraction toward Bayes, passes naive
    for good (c0) and the naive-Bayes midpoint for good (c50)): the loss on the most and on the least entangled fifth,
    Trained RNN and ESN, at the 114 checkpoints (x = training episodes, the initial network at 10). The episodes are not
    the probe environments, so they are resampled by episode (boot_counts(): 1,000 x 3,000, seed 20261103; paired
    False); every resampled curve and naive's level are ratios of sums weighted by the episode counts (loss_curves).
    The learning-excess rows are left out (no Bayes level). One note per row: syn_note (what is particular to the
    row; none for the Trained RNN's least entangled fifth), then SYN_NOTE (common)"""
    C = boot_counts()
    assert C.shape == (gap.N_BOOT, N_EP) and BOOT_SEED == gap.BOOT_SEED
    rows, xs = [], []
    for run in RUNS:
        Z = match_data(run)
        xs.append(gap.train_x(Z["ck_E"]))
        for name, q in SYNTHESIS:
            full, boot, nv, nv_b = loss_curves(Z, C, q, "none")
            f1, b1, n1, nb1 = loss_curves(Z, np.ones((1, N_EP)), q, "none")      # all counts 1 = the full sample
            assert np.allclose(b1[0], full, rtol=1e-12, atol=0) and np.isclose(nb1[0], nv, rtol=1e-12, atol=0)
            own = syn_note(Z, q)
            rows.append(dict(name=name, run=run, family="learning", shape="rise", sample=SYN_SAMPLE, paired=False,
                             note=(own + "; " if own else "") + SYN_NOTE, value=full, boot=boot, naive=nv,
                             bayes=BAYES_LOSS, naive_boot=nv_b, bayes_boot=np.full(len(C), BAYES_LOSS)))
    assert np.array_equal(xs[0], xs[1]), "both runs need the same checkpoints"
    return gap.export("entangled_learning", xs[0], rows)


def window_scale(Z, wins, name):
    """per window (a, b): the factor B / (A - B) (loss at the window's end over its decrease, totals over all items)
    in the identity excess_q = B / (A - B) x (loss share of q at the start - at the end); the share change; the
    excess in loss units, (decrease of q - its proportional decrease) per item and value of q; the total decrease
    per item and value of all items"""
    ckx = train_ep(Z["ck_E"])
    cnt = Z["cnt_" + name].sum(0) if "cnt_" + name in Z else Z["cnt_none"].sum(0)
    out = {k: [] for k in ("factor", "s0", "s1", "ex", "abs_ex", "dL")}
    for a, b in wins:
        A = Z["blk_" + name][int(np.where(ckx == a)[0][0])].sum(0)
        B = Z["blk_" + name][int(np.where(ckx == b)[0][0])].sum(0)
        D = (A - B).sum()
        for k, v in (("factor", B.sum() / D), ("s0", A / A.sum()), ("s1", B / B.sum()),
                     ("ex", (A - B) / D - A / A.sum()), ("abs_ex", ((A - B) - D * A / A.sum()) / (cnt * R)),
                     ("dL", D / (N_EP * T * R))):
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


def panel_e(run, Z=None, pr=None):
    """the two-estimate pairs of checks panel e (every fifth 1-5 and window of WIN_EDGES), raw ("none") and matched
    ("naive"): main set alone vs gradient set alone, training episodes vs probes; and their largest disagreement with
    its window -> {grouping: {"MG": (x, y), "TP": (x, y), "MG_max": (value, window), "TP_max": ...}}"""
    Z = Z if Z is not None else match_data(run)
    pr = pr if pr is not None else probe_shares(run)
    out = {}
    for gname in ("none", "naive"):
        Wm = match_windows(Z, WINS, gname)
        pe, _ = probe_windows(pr, gname)
        d = {"MG": (Wm["ex_M"][:, 1:], Wm["ex_G"][:, 1:]), "TP": (Wm["ex"][:, 1:], pe[:, 1:])}
        for k in ("MG", "TP"):
            diff = np.abs(d[k][0] - d[k][1])
            w = int(np.unravel_index(np.argmax(diff), diff.shape)[0])
            d[k + "_max"] = (float(diff.max()), WINS[w])
        out[gname] = d
    return out


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

COL = {"trained": "#1f77b4", "echo": "#ff7f0e"}
CENTRED_COL = "#e7298a"                         # the most entangled fifth over 2x windows centred on every checkpoint
NAME = {"trained": "Trained RNN", "echo": "ESN"}
X0 = 10                                                        # the initial network is drawn at 10, labelled "0"
FLIP = (800, 1040)                                             # memory-use flip by entanglement (td_flip_scan)
XMAX = 6e4
X_NAIVE = 9.5e4                                                # where naive's levels are drawn in panel a


def passes_naive():
    """training episodes at which the trained RNN's goal accuracy at T on the probes first exceeds naive's"""
    P = np.load(os.path.join(td.TD, "analysis", "probes.npz"))
    goal = P["goal"]
    naive = P["right_naive"][np.arange(len(goal)), goal].mean()
    for f in sorted(glob.glob(os.path.join(td.TD, "analysis", "trained", "ck_*.npz"))):
        z = np.load(f)
        if z["acc"][-1] > naive:
            return float(max(train_ep(int(z["episode"])), X0))


def _time_axis(ax, cross, xlabel=True):
    ax.set_xscale("log")
    ax.set_xlim(X0 * .8, XMAX)
    ax.set_xticks([X0, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.minorticks_off()
    ax.axvspan(*FLIP, color="0.88", lw=0, zorder=0)
    ax.axvline(cross, color="0.4", lw=1, ls=":", zorder=1)
    if xlabel:
        ax.set_xlabel("training episodes")
    ax.spines[["top", "right"]].set_visible(False)


GD_SE_MAX = .05                       # windows whose two half-set GD shares differ by more (SE) are not drawn
PROFILE_PAIRS = (0, 4, 15, 19, 23, 31, 36)                    # indices into pair_list: 0, 80, 480, 880, 1600, 4800, 9600


def windows_split(S):
    """the realized window shares from the main set alone and from the gradient set alone (sampling check)"""
    ckx = train_ep(S["ck_E"])
    out = []
    for a, b in zip(WIN_EDGES[:-1], WIN_EDGES[1:]):
        ia, ib = int(np.where(ckx == a)[0][0]), int(np.where(ckx == b)[0][0])
        d = S["ck_L"][ib] - S["ck_L"][ia]
        out.append([d[:N_BLK_M].sum(0) / d[:N_BLK_M].sum(), d[N_BLK_M:].sum(0) / d[N_BLK_M:].sum()])
    return np.array(out)                                                             # (windows, [main, grad], N_GRP)


def _style():
    import matplotlib.pyplot as plt
    from main.utils import apply_fig_style
    apply_fig_style()
    plt.rcParams.update({"mathtext.fontset": "cm", "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "axes.labelsize": 12, "legend.fontsize": 11})
    return plt


def _fifth_colors(plt):
    return plt.cm.Purples(np.linspace(.35, 1., 5))


def _footer(fig, text, y=-.06, width=215):
    import textwrap
    lines = [ln for para in text.split("\n") for ln in textwrap.wrap(para, width)]
    fig.text(.5, y, "\n".join(lines), ha="center", va="top", fontsize=11, color="0.3", linespacing=1.55)


DATA_LINE = ("Data: 3,000 training-distribution episodes x 30 steps (2,000 main + 1,000 gradient set; each draws its own "
             "pair of variables, as in training: 2,957 distinct pairs), the same episodes at all 114 checkpoints (0-40k "
             "training episodes); 15 log-even windows; 37 consecutive-checkpoint pairs between 0 and 9.6k for the "
             "first-order terms.")
WINS = list(zip(WIN_EDGES[:-1], WIN_EDGES[1:]))
XC = np.array([np.sqrt(max(a, 10) * b) for a, b in WINS])                     # window centres (log)


def figure():
    plt = _style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from main.utils import fig_path
    cross = passes_naive()
    kcol = _fifth_colors(plt)
    fig, axs = plt.subplots(2, 3, figsize=(17.5, 9.8), squeeze=False)
    for i, run in enumerate(RUNS):
        Z = match_data(run)
        x_ck = np.maximum(train_ep(Z["ck_E"]), X0)
        # a: loss per item and value of every group; naive's level of each fifth (grey, edged in the fifth's colour)
        ax = axs[i, 0]
        cnt, nv = Z["cnt_none"].sum(0), Z["nv_none"].sum(0)
        per_item = Z["blk_none"].sum(1) / (cnt * R)
        for q in range(1, 6):
            ax.plot(x_ck, per_item[:, q], color=kcol[q - 1], lw=2.2)
            ax.plot(X_NAIVE, nv[q] / (cnt[q] * R), "s", mfc="0.6", mec=kcol[q - 1], mew=2.2, ms=8.5, clip_on=False,
                    zorder=5)
        ax.text(X_NAIVE, nv[5] / (cnt[5] * R) * 1.2, "naive", ha="center", va="bottom", fontsize=11.5, color="0.35")
        ax.set_yscale("log")
        ax.set_ylim(.012, .5)
        ax.set_yticks([.02, .05, .1, .2, .5])
        ax.set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
        ax.set_ylabel("training loss per item and value")
        # b, c: realized share minus loss share per window, raw fifths and fifths matched on naive's loss
        for j, gname in ((1, "none"), (2, "naive")):
            W = match_windows(Z, WINS, gname)
            ax = axs[i, j]
            for q in range(1, 6):
                ax.errorbar(XC, W["ex"][:, q], W["se"][:, q], color=kcol[q - 1], lw=2.2, marker="o", ms=5,
                            capsize=0, zorder=3 + q)
            xc_full, _ = excess_curves(Z, np.ones((1, N_EP)), 5, gname)      # q5, 2x window around every checkpoint
            ax.plot(x_ck, xc_full, color=CENTRED_COL, lw=1.3, zorder=9)
            ax.axhline(0, color="0.3", lw=1)
            ax.set_ylim(-.21, .21)
            ax.set_ylabel(r"realized share $-$ loss share")
        for j in range(3):
            _time_axis(axs[i, j], cross, xlabel=i == 1)
        axs[i, 0].set_xlim(X0 * .8, 1.3e5)
        axs[i, 0].text(-.3, .5, NAME[run], transform=axs[i, 0].transAxes, rotation=90, va="center", ha="center",
                       fontsize=16, color=COL[run])
    titles = (("a   Training loss on each entanglement fifth",
               r"$\bar\ell_q = \Sigma_{(i,t)\in q}\Sigma_r\sqrt{D_{itr}+c}\ /\ (|q|R)$;   Bayes $=\sqrt{c}$"),
              ("b   Learning beyond each fifth's loss share",
               r"$-\Delta L_q/-\Delta L\ -\ L_q/L$ per window   (0 = proportional)"),
              ("c   The same at matched naive difficulty",
               r"$-\Delta L_q/-\Delta L\ -\ L_q/L$;  fifths in 10 strata of naive's $\ell_{it}$"))
    for j, (t1, t2) in enumerate(titles):
        axs[0, j].set_title(f"{t1}\n{t2}", loc="left", fontsize=13.5, linespacing=1.6)
    h = [Line2D([], [], color=kcol[q], lw=2.5, label=lab) for q, lab in
         ((0, "least entangled fifth"), (2, "middle fifth"), (4, "most entangled fifth"))]
    h += [Line2D([], [], color=CENTRED_COL, lw=1.5,
                 label="most entangled, 2x window centred on every checkpoint (b, c)"),
          Line2D([], [], color="0.6", marker="s", ls="", ms=8, mec="0.3", label="naive, per fifth (a)"),
          Line2D([], [], color="0.4", lw=1, ls=":", label=f"Trained RNN passes naive ({cross:.0f})"),
          Patch(color="0.88", label="memory-use flip (800-1,040)")]
    fig.tight_layout(h_pad=2.4, w_pad=2.2)
    # strips under b and c: each window's total loss decrease, the absolute size of the shares above it
    for i, run in enumerate(RUNS):
        dL = match_windows(match_data(run), WINS, "none")["dL"]
        for j in (1, 2):
            ax = axs[i, j]
            pos = ax.get_position()
            hs, gp = pos.height * .24, pos.height * .035
            ax.set_position([pos.x0, pos.y0 + hs + gp, pos.width, pos.height - hs - gp])
            sx = fig.add_axes([pos.x0, pos.y0, pos.width, hs])
            sx.plot(XC, dL, color=COL[run], lw=1.8, marker="o", ms=4.5)
            sx.set_yscale("log")
            sx.set_ylim(.0025, .12)
            sx.set_yticks([.003, .01, .03, .1])
            sx.set_yticklabels(["0.003", "0.01", "0.03", "0.1"])
            sx.minorticks_off()
            sx.set_ylabel(r"$-\Delta L$")
            sx.text(.99, .95, "window's total loss decrease", transform=sx.transAxes, ha="right", va="top",
                    fontsize=11.5, color=COL[run])
            _time_axis(sx, cross, xlabel=i == 1)
            sx.set_xticks([X0, 1e2, 1e3, 1e4])
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")
            fig.align_ylabels([ax, sx])
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(.5, .015), ncol=3, frameon=False, fontsize=12)
    _footer(fig, "Fifths: each step's items ranked by the goal variable's own entanglement (steps 2-30; 600 items per "
            "fifth and step; step 1, 3.3% of the items, counts in L but is not drawn); c: ranked inside 10 strata of "
            "naive's per-item loss within each step (60 items per stratum, fifth and step), so the matched fifths have "
            "the same naive loss (0.174-0.178 per item and value, against 0.117-0.246 for the raw fifths) and still "
            "span the entanglement (mean 0.04 vs 0.69).  The loss is the run's own training loss (goal variable, "
            "square root of the symmetric KL per value, c = 1e-8) measured against Bayes, so Bayes's own belief scores "
            "the floor 1e-4.  Per window: the share of the exact loss decrease between the window's two checkpoints "
            "that falls on the fifth, minus the fifth's share of the loss at the window's start; bars: block-jackknife "
            "SE of that difference (30 blocks of 100 episodes).  It is a fraction of the window's learning: strips = "
            "the window's total decrease per item and value (excess in loss units = excess x strip / 0.193).\n"
            + DATA_LINE, y=-.1, width=200)
    path = fig_path("td_tt_entangled_learning.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


def _dis_text(dis, k):
    """'.049 / .040 (Trained RNN), .069 / .105 (ESN; matched at 15,200-24,800)': the largest raw / matched
    disagreement of checks panel e per run, with the window of the largest one"""
    parts = []
    for run in RUNS:
        (r0, w0), (r1, w1) = dis[run]["none"][k + "_max"], dis[run]["naive"][k + "_max"]
        w, lab = (w1, "matched") if r1 >= r0 else (w0, "raw")
        parts.append(f"{r0:.3f} / {r1:.3f} ({NAME[run]}; {lab} at {w[0]:,}-{w[1]:,})")
    return " and ".join(parts)


MATCH_STYLE = {"none": dict(color=None, lw=2.6, ls="-", marker="o", label="raw fifths"),
               "naive": dict(color="#17becf", lw=2.6, ls="-", marker="o", label="naive's loss, 10 strata"),
               "naive5": dict(color="#17becf", lw=1.1, ls="-", marker="", label="naive's loss, 5 or 20 strata"),
               "naive20": dict(color="#17becf", lw=1.1, ls="-", marker="", label=None),
               "fr": dict(color="#bcbd22", lw=1.8, ls="--", marker="", label="FR, 10 strata"),
               "H": dict(color="#d62728", lw=1.8, ls="-.", marker="", label="Bayes's entropy, 10 strata"),
               "naiveH": dict(color="#8c564b", lw=1.8, ls=":", marker="", label="naive's loss x entropy, 5 x 5")}


def figure_checks():
    plt = _style()
    from matplotlib.lines import Line2D
    from main.utils import fig_path
    cross = passes_naive()
    kcol = _fifth_colors(plt)
    fig, axs = plt.subplots(2, 5, figsize=(26, 9.8), squeeze=False)
    dis = {}                                     # largest disagreements of the plotted excesses (e), for the footnote
    for i, run in enumerate(RUNS):
        S = dict(np.load(os.path.join(OUT, f"series_{run}.npz")))
        W = windows(S)
        Z = match_data(run)
        pl = pair_list(run)
        dis[run] = panel_e(run, Z)
        ok = (W["EG_se"][:, 1:] <= GD_SE_MAX).all(1) & (W["n_pairs"] > 0)
        # a: loss along the realized step, in units of the first-order prediction (absolute size in the legend)
        ax = axs[i, 0]
        pcol = plt.cm.viridis(np.linspace(0, .9, len(PROFILE_PAIRS)))
        for c, j in zip(pcol, PROFILE_PAIRS):
            f = os.path.join(OUT, run, f"profile_{pl[j][0]:05d}_{pl[j][1]:05d}.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f)
            nr = int(z["n"]) * T * R
            Ls = z["l"].sum((1, 2)) / nr
            fo = abs(z["d0"].sum() / nr)
            ax.plot(z["s"], (Ls - Ls[0]) / fo, color=c, lw=2.2, marker="o", ms=3.5,
                    label=f"{int(train_ep(pl[j][0]))} ({int(train_ep(pl[j][1]) - train_ep(pl[j][0]))}): {fo:.3f}")
        ax.plot([0, 1], [0, -1], color="k", lw=1.2, ls="--")
        ax.set_ylim(-1.05, .15)
        ax.set_xlabel(r"position along the realized step $s$")
        ax.set_ylabel("loss change, first-order units")
        ax.legend(fontsize=10, frameon=False, title=r"start (updates): $|\nabla L\cdot\Delta W|$", title_fontsize=10,
                  loc="lower left", bbox_to_anchor=(0, 0), ncol=2, columnspacing=1.)
        # b: realized fraction of the first-order prediction, per pair and pooled per window
        ax = axs[i, 1]
        xp = np.maximum(S["pr_x0"], X0)
        rat = S["pr_dL_G"].sum((1, 2)) / S["pr_F"].sum((1, 2))
        ax.plot(xp, rat, "o", color=COL[run], ms=5.5, alpha=.8)
        wr = []
        for a, b in WINS:
            sel = (S["pr_x0"] >= a) & (S["pr_x0"] < b)
            wr.append(S["pr_dL_G"][sel].sum() / S["pr_F"][sel].sum() if sel.any() else np.nan)
        ax.plot(XC, wr, color=COL[run], lw=2)
        ax.axhline(1, color="0.3", lw=1)
        ax.axhline(0, color="0.6", lw=.8)
        ax.set_ylim(-.7, 1.7)
        ax.set_ylabel(r"actual / first-order change")
        _time_axis(ax, cross, xlabel=i == 1)
        # c: most entangled fifth: loss share, GD share, first-order (direction of the realized step), exact
        ax = axs[i, 2]
        q = 5
        ax.plot(XC, W["L0"][:, q], color=kcol[4], lw=1.2, alpha=.8)
        ax.errorbar(XC, np.where(ok, W["EG"][:, q], np.nan), np.where(ok, W["EG_se"][:, q], np.nan), color="#2ca02c",
                    lw=1.8, marker="s", ms=5, capsize=0)
        ax.errorbar(XC, np.where(W["n_pairs"] > 0, W["F"][:, q], np.nan), W["F_se"][:, q], color=kcol[4], lw=1.8,
                    ls="--", marker="o", ms=5.5, mfc="white", capsize=0)
        ax.errorbar(XC, W["dL"][:, q], W["dL_se"][:, q], color=kcol[4], lw=2.4, marker="o", ms=5.5, capsize=0)
        ax.set_ylim(.0, .35)
        ax.set_ylabel("most entangled fifth: share")
        _time_axis(ax, cross, xlabel=i == 1)
        # d: the most entangled fifth's excess under every matching
        ax = axs[i, 3]
        for gname in GROUPINGS:
            st = dict(MATCH_STYLE[gname])
            st["color"] = st["color"] or kcol[4]
            st.pop("label")
            Wm = match_windows(Z, WINS, gname)
            ax.plot(XC, Wm["ex"][:, 5], ms=4.5, zorder=4 if gname in ("none", "naive") else 3, **st)
        ax.axhline(0, color="0.3", lw=1)
        ax.set_ylim(-.17, .1)
        ax.set_ylabel(r"most entangled fifth: realized $-$ loss share")
        _time_axis(ax, cross, xlabel=i == 1)
        # e: two estimates of every fifth's excess per window: main vs gradient set, training episodes vs probes
        ax = axs[i, 4]
        pr = probe_shares(run)
        for gname, filled in (("none", True), ("naive", False)):
            Wm = match_windows(Z, WINS, gname)
            pe, _ = probe_windows(pr, gname)
            for q in range(1, 6):
                mf = kcol[q - 1] if filled else "white"
                ax.plot(Wm["ex_M"][:, q], Wm["ex_G"][:, q], "o", color=kcol[q - 1], mfc=mf, mew=1.3, ms=5.5)
                ax.plot(Wm["ex"][:, q], pe[:, q], "^", color=kcol[q - 1], mfc=mf, mew=1.3, ms=6)
        ax.plot([-.25, .25], [-.25, .25], color="0.5", lw=.8)
        ax.set_xlim(-.25, .25)
        ax.set_ylim(-.25, .25)
        ax.set_xlabel("excess, first estimate")
        ax.set_ylabel("excess, second estimate")
        if i == 0:
            ax.legend(handles=[Line2D([], [], color="0.35", marker="o", ls="", ms=6, label="main vs gradient set"),
                               Line2D([], [], color="0.35", marker="^", ls="", ms=6.5,
                                      label="training episodes vs probes"),
                               Line2D([], [], color="0.35", marker="o", ls="", ms=6, mfc="white",
                                      label="open: matched on naive's loss")],
                      loc="upper left", fontsize=10.5, frameon=False, handletextpad=.2)
        for j in range(5):
            axs[i, j].spines[["top", "right"]].set_visible(False)
        axs[i, 0].text(-.3, .5, NAME[run], transform=axs[i, 0].transAxes, rotation=90, va="center", ha="center",
                       fontsize=16, color=COL[run])
    titles = (("a   Loss along the realized step",
               r"$[L(W_k+s\Delta W)-L(W_k)]\ /\ |\nabla L\cdot\Delta W|$;  dashed: first order"),
              ("b   First-order check", r"$[L(W_{k+1})-L(W_k)]\ /\ \nabla L(W_k)\cdot\Delta W$;  line: per window"),
              ("c   Most entangled fifth: four shares",
               r"$L_5/L$, $\langle g_5,g\rangle/|g|^2$, $g_5\!\cdot\!\Delta W/g\!\cdot\!\Delta W$, exact"),
              ("d   Which difficulty explains it?", r"$-\Delta L_5/-\Delta L - L_5/L$ under each matching"),
              ("e   Sampling checks", r"$-\Delta L_q/-\Delta L - L_q/L$, every fifth and window"))
    for j, (t1, t2) in enumerate(titles):
        axs[0, j].set_title(f"{t1}\n{t2}", loc="left", fontsize=13.5, linespacing=1.6)
    h = [Line2D([], [], color=kcol[4], lw=1.2, label="loss share (c)"),
         Line2D([], [], color="#2ca02c", lw=1.8, marker="s", ms=5, label="gradient (GD) share (c)"),
         Line2D([], [], color=kcol[4], lw=1.8, ls="--", marker="o", mfc="white", ms=5.5,
                label="first order (direction) (c)"),
         Line2D([], [], color=kcol[4], lw=2.4, marker="o", ms=5.5, label="realized, exact (c)")]
    for gname in GROUPINGS:
        st = MATCH_STYLE[gname]
        if st["label"]:
            h.append(Line2D([], [], color=st["color"] or kcol[4], lw=st["lw"], ls=st["ls"], label=st["label"] + " (d)"))
    fig.tight_layout(h_pad=2.4, w_pad=2.)
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(.5, .015), ncol=5, frameon=False, fontsize=12)
    _footer(fig, "a: the loss on 500 main-set episodes along the straight segment from checkpoint k to k+1, in units of "
            "the first-order prediction at checkpoint k (dashed); the legend gives each pair's absolute first-order "
            "change per item and value; the end point is that pair's actual / first-order ratio.  b, c: first-order "
            "and gradient terms on the 1,000-episode gradient set (pairs to 9.6k only); GD windows whose two "
            "500-episode half estimates differ by an SE above 0.05 are not drawn.  d: entanglement fifths ranked "
            "within strata of each difficulty variable, within each step (FR = KL(Bayes || naive) of the goal "
            "marginal).  e: the plotted excesses (raw / matched) differ between the main and the "
            f"gradient set by at most {_dis_text(dis, 'MG')}, and between the training episodes and the probes by at "
            f"most {_dis_text(dis, 'TP')}; probes = 10,000 probe episodes (50 test environments x 100 cells x 2 streams, "
            "never trained), from the beliefs stored at every checkpoint.\n" + DATA_LINE, y=-.085, width=250)
    path = fig_path("td_tt_entangled_learning_checks.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


# ═══════════════════════════════════════════════════════════════════
# Numbers (every number of the docstring)
# ═══════════════════════════════════════════════════════════════════

def _fmt_ex(ex, se):
    return f"{ex:+.3f} ({se:.3f})"


def numbers():
    M, G = episodes("main"), episodes("grad")
    V = match_vars()
    print("== episodes")
    for nm, E in (("M", M), ("G", G)):
        ar = np.arange(len(E["goal"]))
        print(f"  {nm}: test state in {np.mean((E['pairs'] < 50).any(1)):.3f}; goal in position 0 "
              f"{np.mean(E['goal'] == 0):.3f}; Bayes / naive goal accuracy at T {E['right_bayes'][ar, E['goal']].mean():.3f}"
              f" / {E['right_naive'][ar, E['goal']].mean():.3f}")
    prs = np.sort(np.concatenate([M["pairs"], G["pairs"]]), 1)
    P = np.load(os.path.join(td.TD, "analysis", "probes.npz"))
    pp = {tuple(p) for p in np.sort(P["pairs"], 1)}
    print(f"  M + G: {len({tuple(p) for p in prs})} distinct pairs of variables in {len(prs)} episodes; episodes whose "
          f"pair is one of the 50 probe environments: {sum(tuple(p) in pp for p in prs)} (probe environments "
          f"pairing two test variables: {np.mean((P['pairs'] < 50).all(1)):.0%}; training episodes with two: "
          f"{int((prs < 50).all(1).sum())})")
    rho = np.mean([np.corrcoef(np.argsort(np.argsort(V["ent"][:, t])), np.argsort(np.argsort(V["naive"][:, t])))[0, 1]
                   for t in range(1, T)])
    print(f"  within-step Spearman(goal entanglement, naive's per-item loss), mean over steps 2-30: {rho:.3f}")
    for gname in GROUPINGS:
        g = grouping(gname, V)
        it = np.bincount(g.ravel(), minlength=N_GRP)
        print(f"  [{gname:7s}] naive loss per item-value by group (step 1, q1-q5) "
              f"{np.round(np.bincount(g.ravel(), V['naive'].ravel(), N_GRP) / (it * R), 4)}; mean ent "
              f"{np.round(np.bincount(g.ravel(), V['ent'].ravel(), N_GRP) / it, 3)}; mean Bayes entropy "
              f"{np.round(np.bincount(g.ravel(), V['H'].ravel(), N_GRP) / it, 3)}")
    tl = timeline()
    for run in RUNS:
        S = dict(np.load(os.path.join(OUT, f"series_{run}.npz")))
        W = windows(S)
        Z = match_data(run)
        pr = probe_shares(run)
        ckx = train_ep(Z["ck_E"])
        R0, R1 = match_windows(Z, WINS, "none"), match_windows(Z, WINS, "naive")
        pe0, ps0 = probe_windows(pr, "none")
        pe1, ps1 = probe_windows(pr, "naive")
        print(f"\n== {run}: {len(S['pr_x0'])} pairs, {len(Z['ck_E'])} checkpoints with losses")
        print("window | total -dL per item-value | RAW q5: realized, loss share, excess (SE of the difference), "
              "M / G, probes (env SE) | q1 excess | GD q5 / q1 (half SE) | first-order q5 / q1 || MATCHED q5 excess "
              "(SE), M / G, probes (env SE) | q1 excess")
        for k, (a, b) in enumerate(WINS):
            print(f"  {a:5d}-{b:5d} | {-R0['dL'][k]:.5f} | {R0['rs'][k, 5]:.3f} {R0['ls'][k, 5]:.3f} "
                  f"{_fmt_ex(R0['ex'][k, 5], R0['se'][k, 5])} {R0['ex_M'][k, 5]:+.3f}/{R0['ex_G'][k, 5]:+.3f} "
                  f"{_fmt_ex(pe0[k, 5], ps0[k, 5])} | {_fmt_ex(R0['ex'][k, 1], R0['se'][k, 1])} | "
                  f"{W['EG'][k, 5]:.3f}/{W['EG'][k, 1]:.3f} ({W['EG_se'][k, 5]:.3f}) | {W['F'][k, 5]:.3f}/{W['F'][k, 1]:.3f}"
                  f" || {_fmt_ex(R1['ex'][k, 5], R1['se'][k, 5])} {R1['ex_M'][k, 5]:+.3f}/{R1['ex_G'][k, 5]:+.3f} "
                  f"{_fmt_ex(pe1[k, 5], ps1[k, 5])} | {_fmt_ex(R1['ex'][k, 1], R1['se'][k, 1])}")
        late = [k for k, (a, b) in enumerate(WINS) if a >= 1600]
        for nm, Rg in (("raw", R0), ("matched", R1)):
            print(f"  non-overlapping windows from 1,600 on with a positive q5 excess ({nm}): "
                  f"{int((Rg['ex'][late, 5] > 0).sum())} of {len(late)}; above 2 SE: "
                  f"{int((Rg['ex'][late, 5] > 2 * Rg['se'][late, 5]).sum())}; below -2 SE: "
                  f"{int((Rg['ex'][late, 5] < -2 * Rg['se'][late, 5]).sum())}")
        print("  pooled windows, q5 excess (SE) by grouping [q1 excess for raw and naive]; probes raw / matched:")
        PW = [w for w in POOLED if w[0] in ckx and w[1] in ckx]
        pp0, pps0 = probe_windows(pr, "none", PW)
        pp1, pps1 = probe_windows(pr, "naive", PW)
        for ip, (a, b) in enumerate(PW):
            parts = []
            for gname in GROUPINGS:
                Wg = match_windows(Z, [(a, b)], gname)
                s = f"{gname} {_fmt_ex(Wg['ex'][0, 5], Wg['se'][0, 5])}"
                if gname in ("none", "naive"):
                    s += f" [q1 {Wg['ex'][0, 1]:+.3f}; M {Wg['ex_M'][0, 5]:+.3f} G {Wg['ex_G'][0, 5]:+.3f}]"
                parts.append(s)
            parts.append(f"probes {_fmt_ex(pp0[ip, 5], pps0[ip, 5])} / {_fmt_ex(pp1[ip, 5], pps1[ip, 5])}")
            print(f"    {a}-{b}: " + " | ".join(parts))
        # absolute size: excess_q = B / (A - B) x (loss-share change); the excess in loss units
        print("  absolute size per window: total -dL per item-value | factor B/(A-B) | q5 loss share start -> end "
              "(raw; matched) | q5 excess in loss units = (q5 decrease - proportional decrease) per q5 item-value, raw "
              "/ matched:")
        idm = 0.
        for wl, lab in ((WINS, "window"), (PW, "pooled")):
            S0, S1 = window_scale(Z, wl, "none"), window_scale(Z, wl, "naive")
            for Sx, Wx in ((S0, match_windows(Z, wl, "none")), (S1, match_windows(Z, wl, "naive"))):
                idm = max(idm, np.abs(Sx["ex"] - Sx["factor"][:, None] * (Sx["s0"] - Sx["s1"])).max(),
                          np.abs(Sx["ex"] - Wx["ex"]).max())
            for k, (a, b) in enumerate(wl):
                print(f"    {lab} {a:5d}-{b:5d}: {S0['dL'][k]:.4f} (L {S0['dL'][k] * (1 + S0['factor'][k]):.4f} -> "
                      f"{S0['dL'][k] * S0['factor'][k]:.4f}) | {S0['factor'][k]:6.3f} | {S0['s0'][k, 5]:.4f} -> "
                      f"{S0['s1'][k, 5]:.4f}; {S1['s0'][k, 5]:.4f} -> {S1['s1'][k, 5]:.4f} | "
                      f"{S0['abs_ex'][k, 5]:+.5f} / {S1['abs_ex'][k, 5]:+.5f}")
        print(f"  identity check: max |excess - factor x share change| and |excess - match_windows| {idm:.1e}")
        # early windows (the checkpoint-scale oscillation): training episodes and probes
        EW = [w for w in EARLY if w[0] in ckx and w[1] in ckx]
        e0, e1 = match_windows(Z, EW, "none"), match_windows(Z, EW, "naive")
        p0, ps0_ = probe_windows(pr, "none", EW)
        p1, ps1_ = probe_windows(pr, "naive", EW)
        se1 = window_scale(Z, EW, "naive")
        print("  early windows, q5 excess raw | matched (SE of the difference) | probes raw | matched (env SE) | "
              "matched in loss units per q5 item-value:")
        for k, (a, b) in enumerate(EW):
            print(f"    {a:3d}-{b:3d}: {_fmt_ex(e0['ex'][k, 5], e0['se'][k, 5])} | "
                  f"{_fmt_ex(e1['ex'][k, 5], e1['se'][k, 5])} | {_fmt_ex(p0[k, 5], ps0_[k, 5])} | "
                  f"{_fmt_ex(p1[k, 5], ps1_[k, 5])} | {se1['abs_ex'][k, 5]:+.5f}")
        # one edge moved by one checkpoint: every window 1.5-2.6x wide with both edges among the checkpoints in 40-400
        c4 = [int(x) for x in ckx if 40 <= x <= 400]
        OW = [(a, b) for a in c4 for b in c4 if b > a and 1.5 <= b / a <= 2.6]
        o1 = match_windows(Z, OW, "naive")
        op1, ops1 = probe_windows(pr, "naive", OW)
        ix, nxt = {w: k for k, w in enumerate(OW)}, dict(zip(c4[:-1], c4[1:]))
        oe = [(ix[(a, b)], ix[w2]) for a, b in OW for w2 in ((a, nxt.get(b)), (nxt[a], b)) if w2 in ix]
        ot, ots, opp, ops5 = o1["ex"][:, 5], o1["se"][:, 5], op1[:, 5], ops1[:, 5]
        fl = [(i, j) for i, j in oe if np.sign(ot[i]) != np.sign(ot[j]) and abs(ot[i]) > 2 * ots[i]
              and abs(ot[j]) > 2 * ots[j]]
        same = [(i, j) for i, j in fl if np.sign(opp[i]) == np.sign(ot[i]) and np.sign(opp[j]) == np.sign(ot[j])]
        both = [(i, j) for i, j in same if abs(opp[i]) > 2 * ops5[i] and abs(opp[j]) > 2 * ops5[j]]
        print(f"  one edge moved by one checkpoint (matched q5 excess; windows 1.5-2.6x wide, edges at every "
              f"checkpoint in 40-{c4[-1]}): {len(OW)} windows, {len(oe)} pairs; sign flips with both values beyond "
              f"2 SE in the training episodes: {len(fl)}; probes with the same signs: {len(same)}, of which "
              f"beyond 2 SE on both sides: {len(both)}")
        for i, j in fl:
            print(f"    {OW[i][0]}-{OW[i][1]} {_fmt_ex(ot[i], ots[i])} | probes {_fmt_ex(opp[i], ops5[i])}  ->  "
                  f"{OW[j][0]}-{OW[j][1]} {_fmt_ex(ot[j], ots[j])} | probes {_fmt_ex(opp[j], ops5[j])}")
        # checks panel e: the plotted excesses (raw and matched), largest disagreements
        pe_ = panel_e(run, Z, pr)
        print("  checks panel e (plotted excesses, fifths 1-5, 15 windows): largest |main - gradient set| raw "
              f"{pe_['none']['MG_max'][0]:.3f} {pe_['none']['MG_max'][1]}, matched {pe_['naive']['MG_max'][0]:.3f} "
              f"{pe_['naive']['MG_max'][1]}; |training - probes| raw {pe_['none']['TP_max'][0]:.3f} "
              f"{pe_['none']['TP_max'][1]}, matched {pe_['naive']['TP_max'][0]:.3f} {pe_['naive']['TP_max'][1]}")
        print("  per window, q5 excess under every matching:")
        for gname in GROUPINGS:
            Wg = match_windows(Z, WINS, gname)
            print(f"    {gname:7s} " + " ".join(f"{e:+.3f}" for e in Wg["ex"][:, 5]))
        sh = [(20, 60), (40, 60), (60, 80), (60, 100), (80, 100), (80, 120), (100, 140), (120, 160), (100, 200),
              (120, 240), (3200, 5600), (4000, 5000), (5000, 6400), (6400, 8000), (8000, 9600), (7000, 10400),
              (9600, 12000), (12000, 15200)]
        sh = [w for w in sh if w[0] in ckx and w[1] in ckx]
        for gname in ("none", "naive"):
            Wg = match_windows(Z, sh, gname)
            print(f"  shifted windows, q5 excess ({gname}): " + " ".join(
                f"{a}-{b}:{e:+.3f}({s:.3f})" for (a, b), e, s in zip(sh, Wg["ex"][:, 5], Wg["se"][:, 5])))
        cnt, nv = Z["cnt_none"].sum(0), Z["nv_none"].sum(0)
        per = Z["blk_none"].sum(1) / (cnt * R)
        per_m = Z["blk_naive"].sum(1) / (Z["cnt_naive"].sum(0) * R)
        for x in (0, 80, 320, 880, 2400, 10400, 39996):
            k = int(np.where(ckx == x)[0][0])
            print(f"  loss per item-value at {x}: step 1 {per[k, 0]:.4f}, q1-q5 {np.round(per[k, 1:], 4)} "
                  f"(q5/q1 {per[k, 5] / per[k, 1]:.2f}); matched q1-q5 {np.round(per_m[k, 1:], 4)}; q5 loss share raw "
                  f"{Z['blk_none'][k].sum(0)[5] / Z['blk_none'][k].sum():.3f}, matched "
                  f"{Z['blk_naive'][k].sum(0)[5] / Z['blk_naive'][k].sum():.3f}")
        print(f"  naive per item-value (M + G): step 1 {nv[0] / (cnt[0] * R):.4f}, q1-q5 {np.round(nv[1:] / (cnt[1:] * R), 4)}"
              f" (q5/q1 {nv[5] / cnt[5] / (nv[1] / cnt[1]):.2f}); naive's q5 loss share {nv[5] / nv.sum():.3f}")
        nvm = Z["nv_naive"].sum(0) / (Z["cnt_naive"].sum(0) * R)
        print(f"  naive per item-value, matched fifths: {np.round(nvm[1:], 4)}")
        # first-order check, profiles, read-out part, GD half sets
        rat = S["pr_dL_G"].sum((1, 2)) / S["pr_F"].sum((1, 2))
        print("  first-order check per pair (actual / first order on G):",
              " ".join(f"{int(a)}:{b:.2f}" for a, b in zip(S["pr_x0"], rat)))
        early = S["pr_x0"] <= 880
        nG = 2 * N_GRAD_HALF * T * R
        print(f"  first pair on G: first order {S['pr_F'][0].sum() / nG:+.4f}, actual {S['pr_dL_G'][0].sum() / nG:+.4f}; "
              f"realized q1 share per window {np.round(R0['rs'][:, 1], 3)}")
        print(f"  median {np.median(rat):.3f}; < 0.5: {(rat < .5).sum()} of {len(rat)}; negative: {(rat < 0).sum()} at "
              f"{[int(a) for a, b in zip(S['pr_x0'], rat) if b < 0]}; pairs to 880: {early.sum()}, range "
              f"{rat[early].min():.2f} to {rat[early].max():.2f}, quartiles {np.round(np.percentile(rat[early], [25, 75]), 2)};"
              f" from 2,080: {rat[S['pr_x0'] >= 2080].min():.2f} to {rat[S['pr_x0'] >= 2080].max():.2f}")
        pl = pair_list(run)
        for j in PROFILE_PAIRS:
            f = os.path.join(OUT, run, f"profile_{pl[j][0]:05d}_{pl[j][1]:05d}.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f)
            nr = int(z["n"]) * T * R
            Ls = z["l"].sum((1, 2)) / nr
            print(f"  profile {int(train_ep(pl[j][0]))} -> {int(train_ep(pl[j][1]))}: L(s) {np.round(Ls, 5)}; argmin s "
                  f"{z['s'][np.argmin(Ls)]:.3f}; L(1) - L(3/4) {Ls[-1] - Ls[-2]:+.5f}; |first order| "
                  f"{abs(z['d0'].sum() / nr):.4f}; end slope {z['d1'].sum() / nr:+.5f}; actual / first order "
                  f"{(Ls[-1] - Ls[0]) / (z['d0'].sum() / nr):.3f}")
        ro = S["pr_F_ro"].sum(1) / S["pr_F"].sum(1)
        late_p = S["pr_x0"] >= 320
        print(f"  read-out part of the first-order change, median q1 / q5 {np.median(ro[:, 1]):.2f} / "
              f"{np.median(ro[:, 5]):.2f}; from 320: {ro[late_p, 1].min():.2f}-{ro[late_p, 1].max():.2f} / "
              f"{ro[late_p, 5].min():.2f}-{ro[late_p, 5].max():.2f}")
        cos = S["pr_g1g2"] / np.sqrt(S["pr_g1g1"] * S["pr_g2g2"])
        okw = (W["EG_se"][:, 1:] <= GD_SE_MAX).all(1) & (W["n_pairs"] > 0)
        print(f"  cos(g1, g2): median {np.median(cos):.2f}, {cos.min():.2f} to {cos.max():.2f}, below .2: "
              f"{(cos < .2).sum()} of {len(cos)}; GD windows drawn: "
              f"{[f'{a}-{b}' for (a, b), o in zip(WINS, okw) if o]}; GD share q1 {np.nanmin(W['EG'][okw, 1]):.3f}-"
              f"{np.nanmax(W['EG'][okw, 1]):.3f}, q5 {np.nanmin(W['EG'][okw, 5]):.3f}-{np.nanmax(W['EG'][okw, 5]):.3f}")
        rg = W["dL"][:, 5] - W["EG"][:, 5]
        print(f"  realized - GD share, q5, drawn windows from 80: {np.round(rg[okw & (W['a'] >= 80)], 3)}")
        # sampling maxima
        sp = windows_split(S)
        dmg = np.abs(sp[:, 0, 1:] - sp[:, 1, 1:]).max(1)
        dtp = np.abs(W["dL"][:, 1:] - pr["dL"][:, 1:]).max(1)
        dmg_x = np.abs(R1["ex_M"][:, 1:] - R1["ex_G"][:, 1:]).max(1)
        dtp_x = np.abs(R1["ex"][:, 1:] - pe1[:, 1:]).max(1)
        to96 = np.array([b <= 9600 for a, b in WINS])
        print(f"  sampling, largest |main - gradient set| realized share per window: {np.round(dmg, 3)} (max "
              f"{dmg.max():.3f}; to 9,600 {dmg[to96].max():.3f}); |training - probes|: {np.round(dtp, 3)} (max "
              f"{dtp.max():.3f}; to 9,600 {dtp[to96].max():.3f})")
        print(f"  matched excess, largest |M - G| per window {np.round(dmg_x, 3)}; |training - probes| "
              f"{np.round(dtp_x, 3)}; median SE of the G-only excess ~ jackknife SE x sqrt(3): "
              f"{np.median(R1['se'][:, 1:]) * 3 ** .5:.3f}")
        # ceiling reference: the share of the loss above naive's level at the window start
        Nq = nv
        rows = []
        for a, b in ((80, 9600), (160, 320), (1600, 9600), (4000, 6400)):
            ia, ib = int(np.where(ckx == a)[0][0]), int(np.where(ckx == b)[0][0])
            A, B = Z["blk_none"][ia].sum(0), Z["blk_none"][ib].sum(0)
            ex = np.maximum(A - Nq, 0)
            rows.append(f"{a}-{b}: realized {(A - B)[5] / (A - B).sum():.3f}, above-naive share {ex[5] / ex.sum():.3f}")
        print("  q5 realized share vs its share of the loss above naive's level (start): " + "; ".join(rows))
    print("\n== timeline (timeline.npz): crossing [5-95%] (fraction of resamples that never stay past), level")
    for k in range(len(tl["names"])):
        print(f"  {tl['run'][k]:7s} {tl['names'][k]:45s} {tl['half'][k]:8.0f} [{tl['half_lo'][k]:.0f}, "
              f"{tl['half_hi'][k]:.0f}] ({tl['never'][k]:.3f}) level {tl['level'][k]:+.4f} side {tl['side'][k]:+d}; "
              f"value at 10.4k {tl['value'][k][np.argmin(np.abs(tl['x'] - 10400))]:+.4f}, last valid "
              f"{tl['value'][k][np.isfinite(tl['value'][k])][-1]:+.4f}")
    sel = (tl["x"] >= 20) & (tl["x"] <= 400)
    print("  q5 excess over centred 2x windows [5-95% band], 20-400 training episodes:")
    for k in range(len(tl["names"])):
        if "excess" in tl["names"][k]:
            print(f"    {tl['run'][k]} {tl['names'][k]}: " + " ".join(
                f"{int(x)}:{v:+.3f}[{lo:+.3f},{hi:+.3f}]" for x, v, lo, hi in
                zip(tl["x"][sel], tl["value"][k][sel], tl["lo"][k][sel], tl["hi"][k][sel])))
    print("  exact log ties in the centred 2x window choice (centred_ties; argmin's pick is kept), q5 excess with the "
          "kept | the other window, raw and matched:")
    for run in RUNS:
        Z = match_data(run)
        for x, kept, other in centred_ties(train_ep(Z["ck_E"])):
            ex = {g: match_windows(Z, [kept, other], g)["ex"][:, 5] for g in ("none", "naive")}
            print(f"    {run:7s} at {x}: {kept[0]}-{kept[1]} | {other[0]}-{other[1]}: raw {ex['none'][0]:+.4f} | "
                  f"{ex['none'][1]:+.4f}; matched {ex['naive'][0]:+.4f} | {ex['naive'][1]:+.4f}")
    # Bayes's own belief on the main set: the loss's floor
    cum = np.zeros((N_MAIN, T, C, R))
    cum[np.arange(N_MAIN), :, M["goal"]] = M["log_pg"]
    lb = item_loss(head_D(torch.as_tensor(cum.reshape(N_MAIN, T, C * R)), torch.as_tensor(M["goal"]),
                          torch.as_tensor(M["log_pg"])), DMIN).numpy()
    print(f"\n== Bayes's own belief (its goal marginal as the network's belief), main set: loss per item and value "
          f"{lb.sum() / (N_MAIN * T * R):.6e} (floor sqrt(1e-8) = {BAYES_LOSS:.6e})")
    # synthesis export, and its crossings against this module's own code on the same resamples
    print("\n== synthesis export (td_tt_gap; gap fraction f = (naive - loss) / (naive - Bayes))")
    synthesis()
    syn = gap.load("entangled_learning")
    tl_loss = {(tl["run"][k], tl["names"][k]): k for k in range(len(tl["names"]))}
    Cb = boot_counts()
    # independent rebuild of resampled curves: explicit episode lists, raw per-item loss caches, groups from episodes
    grp = np.concatenate([M["grp"], G["grp"]])
    rng, worst = np.random.default_rng(5), 0.
    for run in RUNS:
        Z = match_data(run)
        for E in (1100, 15000):
            l = np.concatenate([np.load(os.path.join(OUT, run, f"loss_{w}_{E:05d}.npz"))["l"]
                                for w in ("main", "grad")])
            for q in (5, 1):
                _, boot, _, nv_b = loss_curves(Z, Cb, q, "none")
                for b in rng.choice(N_BOOT, 3, replace=False):
                    idx = np.repeat(np.arange(N_EP), Cb[b].astype(int))
                    m = grp[idx] == q
                    worst = max(worst, abs(l[idx][m].mean() / R / boot[b, list(Z["ck_E"]).index(E)] - 1),
                                abs(V["naive"][idx][m].mean() / R / nv_b[b] - 1))
    print(f"  check: resampled curves and naive levels rebuilt by explicit episode repetition (2 runs x 2 checkpoints "
          f"x 2 fifths x 3 resamples): largest relative difference {worst:.1e}")
    print("  check: c0 vs this module's timeline crossing at naive's level; c50 recomputed with this module's crossing "
          "code (loss below (naive + Bayes) / 2), same resamples:")
    for i in range(len(syn["names"])):
        run, name = str(syn["run"][i]), str(syn["names"][i])
        Z = match_data(run)
        x = gap.train_x(Z["ck_E"])
        full, boot, nv, nv_b = loss_curves(Z, Cb, dict(SYNTHESIS)[name], "none")
        own50 = crossing(x, full, (nv + BAYES_LOSS) / 2, -1)
        lo50, hi50, nev50 = crossing_ci(x, boot, (nv_b + BAYES_LOSS) / 2, -1)
        k = tl_loss[(run, name)]
        print(f"    {run:7s} {name:28s} c0 {syn['c0'][i]:8.1f} [{syn['c0_lo'][i]:.1f}, {syn['c0_hi'][i]:.1f}] "
              f"({syn['c0_p'][i]:.3f} cross) vs timeline {tl['half'][k]:8.1f} [{tl['half_lo'][k]:.1f}, "
              f"{tl['half_hi'][k]:.1f}] ({1 - tl['never'][k]:.3f}) | c50 {syn['c50'][i]:8.1f} [{syn['c50_lo'][i]:.1f}, "
              f"{syn['c50_hi'][i]:.1f}] ({syn['c50_p'][i]:.3f}) vs own {own50:8.1f} [{lo50:.1f}, {hi50:.1f}] "
              f"({1 - nev50:.3f}) | gap fraction at 880 / 10.4k / 40k "
              f"{syn['frac'][i][np.searchsorted(x, 880)]:+.3f} / {syn['frac'][i][np.searchsorted(x, 10400)]:+.3f} / "
              f"{syn['frac'][i][-1]:+.3f}; naive {syn['naive'][i]:.4f}")
    print("  facts behind the notes (full sample, every checkpoint): the network's own loss q5 / q1; each fifth's loss "
          "against naive's level on it")
    C1 = np.ones((1, N_EP))
    for run in RUNS:
        Z = match_data(run)
        x = train_ep(Z["ck_E"])
        (l5, _, n5, _), (l1, _, n1, _) = loss_curves(Z, C1, 5), loss_curves(Z, C1, 1)
        r = l5 / l1
        print(f"    {run:7s} q5 / q1 > 1 at {int((r > 1).sum())} of {len(r)} checkpoints ({r.min():.3f}-{r.max():.3f}; "
              f"<= 1 at {[int(v) for v in x[r <= 1]]}: {np.round(r[r <= 1], 3)}); naive's q5 / q1 {n5 / n1:.2f}; above "
              f"naive's level at {int((l5 > n5).sum())} (q5) / {int((l1 > n1).sum())} (q1) checkpoints; at {int(x[-1])}"
              f" q5 {l5[-1]:.4f} vs {n5:.4f}, q1 {l1[-1]:.4f} vs {n1:.4f}")
    print("  notes (one per row):")
    for i in range(len(syn["names"])):
        print(f"    {syn['run'][i]:7s} {syn['names'][i]}: {syn['note'][i]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--minutes", type=float, default=22.)
    ap.add_argument("--max-raw", type=int, default=MAX_RAW)
    a = ap.parse_args()
    if a.what == "episodes":
        for which in ("main", "grad"):
            E = episodes(which)
            print(which, {k: v.shape for k, v in E.items()})
    elif a.what == "check":
        check(a.args[0], int(a.args[1]))
    elif a.what == "data":
        for run_ in a.args:
            if not data(run_, a.minutes):
                break
    elif a.what == "losses":
        for run_ in a.args or RUNS:
            if not losses(run_, a.minutes, a.max_raw):
                break
    elif a.what == "summary":
        for run_ in a.args or RUNS:
            summarize(run_)
            match_data(run_)
            probe_shares(run_)
    elif a.what == "profile":
        run_ = a.args[0]
        pl = pair_list(run_)
        profile(run_, [pl[int(j)] for j in a.args[1:]])
    elif a.what == "timeline":
        timeline()
    elif a.what == "synthesis":
        synthesis()
    elif a.what == "figure":
        figure()
        figure_checks()
    elif a.what == "numbers":
        numbers()
    elif a.what == "pairs":
        for run_ in a.args or RUNS:
            print(run_, [(x, y, int(train_ep(x))) for x, y in pair_list(run_)])
