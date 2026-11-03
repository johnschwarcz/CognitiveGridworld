"""Which parameter group's change produced each behavioural family's improvement, and in which training window? Exact
weight swaps between the stored checkpoints of the Trained RNN (John 11-02: link the neural changes to the behavioural
developments through training, ideally causally - "How does the network get to the correct answer before learning
optimal bayes dynamics?"; approved experiment (2) of three, weight swaps along training; inference only, no training).
Fix round (11-02, after the adversarial check): size-matched random controls, read-out split, gate gains, paired
orderings; the class-level reading of the first version is corrected (FAILED / NULL). Final round (11-02, after the last
check): four labels restated (the read-out on the answers rows; the read-in / read-out 'concentration'; the read-out's
held share; W_out's 'direction', now measured by a fixed-norm direction swap, 48 networks), W_ih also set against its
proportional share, lean figures (John: "Swaps and lesions are also very overwhelming, let me know if there is anything
significant there"). Labels round (11-02, after the final check; wording and figure text only, no computation or
stored value changed): the direction swap is fixed-norm, not matched on calibration (finding 11); scopes and exceptions
added to the SIGNIFICANT / ANSWER wording; black panel titles in td_tt_swaps.png, formula titles in
td_tt_swaps_alpha.png.

SIGNIFICANT (what survived every check). In 800-11,200 training episodes, where the random sets ran, the g (candidate)
rows' change carries more of the learning than random weights of the same size (which family in which window: the
ANSWER), and of the forget, input and output rows and the input weights, as classes, none is resolved above the random
share (most are below it) wherever a family changes by >= .1 of the naive-Bayes gap (random sets of W_hh for the W_hh
rows, of W for W_ih; the exceptions, one of them against random sets of W, in the ANSWER; ->
figure_bin/td_tt_swaps.png). Over longer stretches (800 -> 1,600 -> 4,000 -> 11,200) the sets' changes work only
together: taking one set back costs up to more than the whole improvement, taking one forward alone gains little or
loses (-> figure_bin/td_tt_swaps_alpha.png).

DESIGN
  Network: the Trained RNN (LSTM N = 1,000, seed 20261029), 114 checkpoints (training episodes 10 = the initial network
  ... 39,996). Weight sets: the 15 fine groups of td_tt_intervene.groups(), a partition of all 8,046,020 weights:
  read-in (W, b; 16,000); W_ih and W_hh per gate block i / f / g / o (rows k H .. (k + 1) H, 1,000,000 weights each);
  the LSTM biases per gate (the gate's b_ih and b_hh rows, 2,000); read-out (W, b; 20,020); start state (h0, c0; 2,000).
  (1) Along the training path. For each of the 113 consecutive checkpoint pairs (k, k + 1) and set g:
      dA_g = m(W_k with g <- W_k+1) - m(W_k)          (adding g's change to the earlier network)
      dB_g = m(W_k+1) - m(W_k+1 with g <- W_k)        (removing g's change from the later network)
      phi_g = (dA_g + dB_g) / 2                        (two-endpoint Shapley value)
      interaction = [m(W_k+1) - m(W_k)] - Sigma_{15 groups} phi_g.
    For an m quadratic in the weights phi_g = g's linear term + its own curvature + half of each pairwise cross term,
    phi is additive over disjoint sets (a class's phi = the sum of its groups') and the interaction is exactly 0: the
    interaction measures third- and higher-order dependence only. A small interaction means near-quadratic dependence,
    NOT additivity: the pairwise cross terms can be (and are, finding 2) as large as the change, and phi splits each
    evenly between its two sets (a Shapley split of shared effects). Pooled = sums over the consecutive pairs of a
    window; the change telescopes to the window's end points. Windows (training episodes; the checkpoints nearest the
    requested 835 / 1,575 / 4,000 / 11,000): 10-800 (25 pairs), 800-1,600 (10), 1,600-4,000 (18), 4,000-11,200 (24),
    11,200-39,996 (36). Single pairs are never read alone. Classes (CLASSES) are sums of fine groups' phi: W_ih (4
    blocks, 4M weights), W_hh f+i (2M), W_hh g+o (2M), W_hh g and W_hh o (1M each), read-out, read-in, biases (4),
    start; per gate (W_ih + W_hh + bias rows of the gate, 2,002,000; 'the g gate' = all g rows and the g biases).
  (2) Size-matched random sets (fix round; not part of the partition). A class's share of a change says nothing about
    concentration until it is set against a random set of the same size from the same place. Random families
    (CTRL_FAM: k entries drawn without replacement from a pool, 3 seeds each, swapped and pooled exactly like the
    groups): rW1M = 1M of W = W_ih + W_hh (8M; seed 1 = the path experiment's rand_W), rHH1M = 1M of W_hh and rIH1M =
    1M of W_ih (tensor-matched controls of single blocks), rG1M = 1M of the g rows of W_ih + W_hh (2M; gate-matched:
    the same tanh gain), rHH2M = 2M of W_hh (for W_hh f+i, W_hh g+o), rW2M = 2M of W (for a gate's W_ih + W_hh rows;
    its 2,000 biases unmatched), rW4M = 4M of W (for W_ih), rb2k = 2,000 of the 8,000 biases (seed 1 = rand_b). Class
    minus random = the class's value minus the 3 seeds' mean, with the 5-95% of the paired difference (same resamples)
    and the range over the seeds. Run for 800-1,600 and 4,000-11,200 (every family), 1,600-4,000 (rHH1M, rHH2M, rW2M,
    rW4M); CTRL_NOT_RUN lists the rest. Where a family was not run, a class is set against its proportional share of
    the pool (k / pool x the pool's summed values), the value the random families matched wherever they ran
    (finding 8). A class's value is a sum of single-group swaps, a random set is swapped as one unit: the two agree to
    second order only, and the random half of W (rW4M) deviates systematically (finding 8), so W_ih (half of W) is
    also set against its proportional share, which is built like the class (final round).
  (3) Alpha paths between checkpoints further apart, E -> L = 800 -> 1,600, 1,600 -> 4,000, 4,000 -> 11,200 (the
    middle windows' edges): rewind = L with g <- (1 - a) L + a E; graft = E with g <- (1 - a) E + a L; a = .25, .5,
    .75, 1 (a = 0: the parent). Effects at a = 1: graft = f(E with g <- L) - f(E); rewind = f(L) - f(L with g <- E);
    their mean is the long pair's two-endpoint value, compared with the same set's adjacent-pair phi summed over the
    same stretch (path check). Interpretable: the exact training loss stays between the parents' at every alpha
    (compatible) AND the row's alpha curve is monotone (full sample). Over these long stretches a rewind measures how
    much the rest of the network has adapted to the set (co-adaptation), not the set's own contribution: the rewinds
    are descriptive, never a control. The random 1M of W has 3 seeds here too (ALPHA_CTRL).
  (4) Read-out split (4,000-11,200): the read-out's change swapped whole (ro_all = the group 'readout', rerun), W_out
    only, b_out only, as a pure change of scale (ro_scale: W_out <- s W_out, s = the other checkpoint's least-squares
    scale on this one's) and, final round, as a fixed-norm change of direction (ro_dir: the other checkpoint's W_out at
    this one's Frobenius norm; stored apart in ro_dir/; not matched on calibration: the belief's entropy changes too,
    finding 11), with the goal marginal's entropy at T as an extra row: is the read-out's holding part a change of what
    is read from the states or only a sharper / flatter belief (calibration)?
  (5) Gate gains (800-1,600, 4,000-11,200; every pair): W_k on the probes (a direct forward, = the engine's); per gate
    block the change of its drive, (W_k+1 - W_k)_G x_t or (W_k+1 - W_k)_G u_(t-1), and its effect on the gate's output
    (slope x drive change; sigma' for i, f, o, tanh' for g), RMS over probes, steps and units.
  (6) Figures (final round: lean, only what survived the checks). td_tt_swaps.png: per family (answers, holding a split,
    coupling) and the training loss, in the three windows where every random family it uses ran (800-1,600, 1,600-4,000,
    4,000-11,200), the excess share (class - the mean of its 3 random sets) / the change of W_hh g, the g gate, W_hh
    f+i, W_hh o and W_ih (FIG_SETS); drawn where the change's 5-95% excludes 0, |change| >= .1 (FIG_MIN; stricter than
    the text's .05: the two changes below .1, answers 4,000-11,200 and coupling 800-1,600, give excess shares up to
    +1.20 and down to -.65; the figure's footnote names them) and |interaction| <= |change|. td_tt_swaps_alpha.png: per
    long pair the training loss's rewind and graft (alpha = 1) of the 8 W blocks, the read-in, the read-out and the
    random 1M of W (mean of 3 draws), against the pair's whole change. Everything else (fine groups per row, shares per
    class, read-in / read-out, 10-800 and 11,200-39,996, the alpha paths' intermediate points and path check) is in
    `numbers` and below.

MEASURES (every network scored by td_tt_intervene: its fp32 forward on the 10,000 probes = 50 test environments x 100
  latent cells x 2 streams of 30 steps, and on the 5,028-episode forward set of td_tt_wormholes)
  ROWS = the engine's 18 cheap probe rows + its 8 forward-set rows: answers (goal accuracy at T, both right, far
  errors, distant pairs merged, corner answer = truth; log q of the true goal value at T and log q0 + log q1;
  KL(Bayes || net) of the goal marginal at T and averaged over the 30 steps, and of both variables), holding (held per
  variable at corner-split ends and at one-variable end splits, joint corner split held: probes and forward set;
  end-mass slope; distance out of a corner split; committed to a corner), coupling (jump excess angle over the
  independence null, long-path coupling G, angle of jumps >= 4 cells, direction slope, coupling angle out of a corner
  split, same-diagonal share). The timeline's held rows are the forward set's (the probe held rows rest on 720 / 365
  var-steps). Training loss: the run's own loss (td_tt_entangled_learning: power-.5 symmetric KL of the goal marginal vs
  Bayes per value) on its main set M of 2,000 training-distribution episodes, through a GPU port of its primal_loss
  (its forward is hard-wired to the CPU: copied without the tangents as _loss_forward).
  Units: gap fraction f = (y - y_naive) / (y_Bayes - y_naive) with naive's and Bayes's levels on the same episodes
  (td_tt_gap), so a change dy is df = dy / (y_Bayes - y_naive): one fixed affine map per row, additive (naive -> Bayes
  = +1). Training loss: naive's loss on M .1726 per item and value, Bayes's = the floor 1e-4. Intervals: 5-95% over
  td_tt_gap.boot_counts() (1,000 resamples of the 50 test environments, seed 20261103; the same matrix as the timeline,
  so the rows are paired with each other and with td_tt_timeline); the training loss over 1,000 resamples of its 2,000
  episodes (seed 20261206; every episode its own pair of variables; not paired with the rest). Shares (a set's phi / the
  window's change) are given only with the change's absolute size, and only where the change's 5-95% excludes 0,
  |change| >= .05 (MIN_CHANGE) and |interaction| <= |change| (else 'non-additive'). Family means: the equal-weight mean
  over FAMILY_ROWS of each row's f-unit change or contribution (answers: goal accuracy, both right, far errors, corner
  answer; holding: held per variable, one-variable, joint, end-mass slope, distance out; coupling: excess angle, G,
  angle of jumps >= 4 cells (missing in 10-800), direction slope, angle out, same-diagonal share). Orderings between
  sets are stated only with their paired difference (ORDERINGS), never from overlapping intervals.

ANSWER (Trained RNN, seed 20261029; f units = gap fraction, naive -> Bayes = +1; [5-95%] over the shared resamples;
'random' = the mean of 3 random sets of the same size from the same place, differences paired; shares are of the
window's change; every number printed by `numbers` unless marked otherwise)
  The g (candidate) rows carry more of the learning than random weights of their size (td_tt_swaps.png; each of the 39
  drawn points whose 5-95% excludes 0 keeps its sign for each of the 3 random draws). The recurrent g rows (W_hh g, 1M)
  against random 1M sets of W_hh: answers +.21 [.16, .27] / +.26 [.13, .40] of the change (800-1,600 / 1,600-4,000),
  holding +.23 [.13, .32] / +.23 [.15, .32], coupling +.20 [.05, .33] (1,600-4,000), the training loss +.21 [.18, .24]
  / +.21 [.17, .25] / +.17 [.10, .23] (800-1,600 / 1,600-4,000 / 4,000-11,200); not resolved in 4,000-11,200 on
  holding (+.10 [-.04, .25]) or coupling (+.05 [-.02, .13]). Accuracy, 800-1,600: +.229 [.176, .285] of +.653 against
  +.105 for a random 1M of W_hh (seeds .090-.115), a difference of +.124 [.076, .170]. All g rows with the g biases (the
  g gate, 2M) against random 2M sets of W: answers +.36 [.30, .42] / +.49 [.34, .65], holding +.38 [.21, .53] / +.18
  [.07, .30] / +.26 [.06, .47], coupling +.16 [.02, .29] (1,600-4,000), the loss +.31 [.27, .36] / +.24 [.19, .29] /
  +.22 [.14, .30]. Why the g rows: tanh's slope (.7680 on the probes against .1559-.1622 for the sigmoid gates) makes
  the same weight change move the g gate's output most; W_hh g's drive change is the smallest of the W_hh blocks, its
  gate-output change the largest of all 8 blocks in 10 of 10 and 24 of 24 pairs (finding 12). Part of W_hh g's lead is
  therefore any g row's: against random 1M sets of the g rows (W_ih + W_hh; the same gain) it still leads on log q
  (+.117 [.080, .156]), held per variable (+.030 [.011, .049]) and the loss (+.042 [.030, .055]) in 800-1,600 and on the
  loss in 4,000-11,200 (+.11 [.05, .18] of the change), not on accuracy (+.043 [-.004, .088]) or the holding family
  (800-1,600: +.12 [-.03, .26] of the change; 4,000-11,200: -.01 [-.14, .14]), and below them on the small answers
  change of 4,000-11,200 (-.38 [-.69, -.08] of +.095).
  The other rows, as classes, are never resolved above the random share in 800-11,200 wherever the family changes by
  >= .1 of the gap, and most fall below it: W_hh f+i, W_hh o and W_ih never exceed the random sets of td_tt_swaps.png
  there (random sets of W_hh for the W_hh rows, a random half of W for W_ih). Exceptions: against a random 2M of W, the
  cross-tensor baseline, W_hh f+i exceed on answers in 1,600-4,000 (+.11 [.03, .18]), through their i rows (one block,
  W_hh i, exceeds a random 1M of W_hh there, +.08 [.01, .16]; finding 4); on the small answers change of 4,000-11,200
  (+.095, not drawn) W_hh f+i exceed a random 2M of W_hh by +.61 [.37, .86] of it; after 11,200, where no random set
  ran, W_ih exceeds its proportional share on holding (below). The task's hypothesis fails at matched size, and
  the first version's 'mirror image' is retracted. W_hh f+i carry .01 [-.06, .08] of the 4,000-11,200 holding change,
  what a random half of W_hh carries (.06; difference -.04 [-.15, .06]). W_ih carries .29 [.23, .35] of the 800-1,600
  answers change, less than a random half of W (.51; -.22 [-.28, -.16]; against its proportional share -.19 [-.25,
  -.13]). The read-out's part of H is mixed by row and has no random counterpart: in 800-1,600 it lowers the answers
  family mean (-.042 [-.057, -.026] f) through one row, corner answer = truth (-.274 [-.316, -.232] of that row's +.458
  [.377, .538]; the same row -.180 [-.246, -.121] in 1,600-4,000 and -.121 [-.209, -.043] in 4,000-11,200), while it
  adds goal accuracy (+.052 [.024, .080]) and both right (+.047 [.025, .068]). Early, W_hh f+i carry less than a random
  half of W_hh (answers -.12 [-.16, -.08]); late, W_ih carries what a random half of W carries (holding +.05 [-.12,
  .21]; against its proportional share +.10 [-.07, .25]): its standing rises (paired change +.27 [.09, .44]), and it
  exceeds its proportional share in 11,200-39,996 (+.104 [.031, .184] f; no random set run there).
  4,000-11,200 (the holding window): no tensor-level class exceeds its random baseline on holding; only g rows do: the g
  gate (.39 of the change against .13 for a random 2M of W; +.26 [.06, .47]) and W_ih's g rows (+.18 [.04, .32] of the
  change over a random 1M of W_ih; on the end-mass slope +.137 [.057, .218] f); W_hh g alone +.10 [-.04, .25] (not
  resolved). The read-in (16,000 weights) and the read-out (20,020) are whole tensors with no random counterpart: they
  carry +.095 [.019, .162] (.29) and +.087 [.030, .138] (.26) of the holding family, not separable from W_hh g+o (+.044
  [-.021, .114]; differences +.051 [-.074, .167], +.043 [-.079, .157]) or W_ih (+.111 [.032, .201]; -.016 [-.177,
  .125], -.024 [-.141, .082]). W_out's change splits into a fixed-norm direction swap and a scale swap that partly
  cancel (finding 11). The direction swap (the other checkpoint's W_out at this one's norm) adds end-mass slope (+.252
  [.190, .320]) and held per variable (+.126 [.086, .166]), but it also flattens the goal belief (entropy +.065 [.060,
  .070] nats), and calibration alone moves the held rows (the scale swap), so its values mix the change of direction
  with a flatter belief (its joint-held and distance-out values are not read as direction). The scale swap, a sharper
  goal belief (entropy -.135 [-.137, -.133] nats), removes held splits (held per variable -.109 [-.117, -.099], joint
  -.127 [-.140, -.112]) and end-mass slope (-.110 [-.116, -.104]) and adds distance out (+.169 [.155, .182]). The
  end-mass slope is clear of calibration: W_out's whole change sharpens the belief (-.071 [-.075, -.065] nats), and
  sharpening alone lowers the slope (the scale swap), yet the change raises it (+.142 [.081, .208]). Net: no resolved
  held share (held per variable +.009 [-.027, .048], joint -.031 [-.079, .018], one-variable ends +.151 [-.024, .307]).
  Attribution: two-endpoint Shapley values of single-set swaps between consecutive checkpoints. After 800 the
  third-order residual is |.| <= .039 f on every row except goal accuracy (1,600-4,000: -.108 [-.178, -.032], -.36 of
  its change; later +.032 / +.031, .52 / 1.13 of accuracy's small changes): near-quadratic. But the pairwise
  interactions are as large as the changes (summed one-sided terms +1.234 / +.013 against +.653 for accuracy in
  800-1,600), so each value contains an even split of shared effects; random sets split the same way and carry their
  proportional share of their pool (seed-mean deviation -.039 .. +.049 f, mean |.| .006; the random half of W deviates
  systematically, finding 8), so 'more than a random set of its size' means more than its share of the pool. Over the
  alpha paths' longer stretches (10-24 checkpoint pairs) no single set's change is sufficient: rewinding W_hh g alone
  costs +.436 / +.470 / +.417 of the training loss's gap against whole changes of +.408 / +.230 / +.114, and the 15
  groups' grafts gain at most +.015 (co-adaptation; td_tt_swaps_alpha.png, finding 10).

FINDINGS
  1. What changes per window (10-800 / 800-1,600 / 1,600-4,000 / 4,000-11,200 / 11,200-39,996): goal accuracy
     +1.110 [1.017, 1.204] / +.653 [.604, .705] / +.298 [.256, .342] / +.061 [.034, .086] / +.028 [.013, .042] (from
     chance; naive's level is 0); log q of the truth +.001 [-.130, .138] / +.602 / +.160 / +.030 / +.008; KL to Bayes
     over all steps +.179 / +.570 / +.185 / +.044 / +.014; held per variable (forward set) +.051 / +.118 / +.284 / +.331
     [.310, .352] / +.151; joint corner split held +.004 / +.054 / +.202 / +.365 [.344, .387] / +.203; end-mass slope
     +.109 / +.175 / +.292 / +.332 / +.111; excess angle +.179 / +.086 / +.191 / +.283 / +.218; direction slope +.042
     [-.024, .106] / +.034 [-.023, .091] / +.246 / +.374 / +.309; training loss +1.417 / +.408 / +.230 / +.114 / +.069.
     Family means (FAMILY_ROWS): answers +.863 / +.520 / +.252 / +.095 / +.012; holding +.063 / +.132 / +.319 / +.330 /
     +.130; coupling +.122 / +.080 / +.231 / +.286 / +.209.
  2. Near-quadratic, not additive. From 800 on the third-order residual (the interaction) is |.| <= .039 f on every
     row of NUM_ROWS (accuracy, far errors, log q, KL, the held rows, end-mass slope, distance out, excess, G, direction
     slope, the loss) except goal accuracy at 1,600-4,000 (-.108 [-.178, -.032], a threshold row; -.36 of its change;
     at 4,000-11,200 and 11,200-39,996 its +.032 and +.031 are .52 and 1.13 of accuracy's small changes). In 10-800 the
     continuous rows are far from quadratic: interaction +2.841 [2.562, 3.184] on log q (change +.001), +2.587 on KL
     (change +.179), -2.634 on G, -.299 on the excess angle (the early collapse to saturated, over-confident states);
     accuracy +.050 [-.061, .166] of +1.110. The small residual does not make the sets additive: summed over the
     groups, adding one group's change to W_k (dA) gives about twice the change and removing it from W_k+1 (dB) about
     none (accuracy 800-1,600: +1.234 / +.013 vs +.653; held per variable 4,000-11,200: +.728 / -.061 vs +.331), so the
     pairwise cross terms are negative and as large as the change (the simultaneous changes substitute for each other:
     concave along the realized step, as the end slope > 0 of td_tt_entangled_learning). phi splits every cross term
     evenly between its two sets: a Shapley split of shared effects, not an additive account. Single sets can change
     sign between the two sides (W_ih, accuracy 800-1,600: dA +.407, dB -.084; W_hh f+i, held per variable
     4,000-11,200: +.129 / -.088), and random sets split the same way (random 1M of W_hh, accuracy 800-1,600: +.291 /
     -.080; random 2M of W_hh, held per variable 4,000-11,200: +.235 / -.181): the one-sided terms mostly measure a
     generic overshoot of single-set swaps, and a set's phi is read only against the phi of random sets of its size.
  3. 800-1,600: the g rows of W_hh, against random sets. W_hh g is the largest single group on accuracy, far errors, log
     q, KL, held per variable, joint held, distance out and the loss (paired difference to the runner-up: accuracy +.100
     [.019, .181] over W_ih g; log q +.192 [.137, .243] over W_hh f; held per variable +.043 [.018, .069] over the
     read-in; loss +.077 [.054, .102] over W_ih g); on the end-mass slope it is not separable from W_ih g (+.028 [-.039,
     .090]). Its values, phi [5-95%] (dA / dB): accuracy +.229 [.176, .285] (+.405 / +.053) of +.653; log q +.295 [.241,
     .353] (+.445 / +.146) of +.602; KL +.266 [.223, .312] of +.570; held per variable +.063 [.037, .090] (+.082 /
     +.043) of +.118; end-mass slope +.072 [.041, .103] of +.175; training loss +.144 [.127, .162] (+.238 / +.049) of
     +.408. Against a random 1M of W_hh (3 seeds): accuracy +.124 [.076, .170] (over the seeds +.114 to +.139), log q
     +.163 [.117, .208], KL +.149 [.113, .184], held per variable +.039 [.018, .062], joint held +.023 [.008, .038],
     end-mass slope +.040 [.017, .064]. Against a random 1M of the g rows of W_ih + W_hh (gate-matched): log q +.117
     [.080, .156], KL +.102 [.074, .131], held per variable +.030 [.011, .049], joint held +.018 [.005, .030], loss
     +.042 [.030, .055], answers family +.043 [.014, .072]; not accuracy (+.043 [-.004, .088]), the end-mass slope
     (+.018 [-.017, .050]) or the holding family (+.016 [-.005, .033]); later, in 4,000-11,200, it falls below this
     control on the small answers change (-.38 [-.69, -.08] of +.095; the ANSWER). Gate gain (finding 12): W_hh g's
     drive change is the smallest of the four W_hh blocks (RMS .0816 against .1314-.1745; squared displacement 182
     against 319-384), its gate-output change the largest of all 8 blocks (RMS .0591 against .0191-.0358). W_ih g:
     +.129 [.092, .170] of accuracy, above a random 1M of W_ih (+.081 [.048, .118]) but below a random 1M of the g rows
     (answers share -.10 [-.15, -.04]); second on accuracy only (over W_hh f +.058 [.015, .104]; on log q W_hh f exceeds
     it, -.042 [-.079, -.006]). The read-out's change lowers log q (-.100 [-.121, -.081]) while adding accuracy (+.052).
     Answers and holding: gate g carries similar shares of both (.60 [.53, .67] and .59 [.42, .77]; difference -.01
     [-.19, .18] - an interval this wide does not show the same use), both far above a random 2M of W (+.36 [.30, .42]
     and +.38 [.21, .53]); W_ih (-.18 [-.34, -.02]), the read-out (+.12 [.07, .18]) and gate i (-.10 [-.19, -.02])
     already differ between the two families.
  4. 1,600-4,000: both families still lean on the recurrent g rows; the forget / input rows fall below random on holding
     (on answers they are at random). Holding (+.319): W_hh g+o .51 [.40, .64] against .32 for a random 2M of W_hh (+.19
     [.14, .25]); W_hh g .38 [.27, .50] against .15 for a random 1M of W_hh (+.23 [.15, .32]); W_hh f+i .09 [.05, .12]
     against .32 (-.23 [-.31, -.16]); W_ih .01 [-.10, .13] against .39 for a random 4M of W (-.37 [-.49, -.25]; against
     its proportional share -.29 [-.39, -.19]); gate g .33 [.20, .47] against .15 for a random 2M of W (+.18 [.07,
     .30]); the read-out and the read-in .16 each. The forget rows work against holding (W_hh f: held per variable -.017
     [-.033, -.002], joint -.025 [-.036, -.014], end-mass slope -.019 [-.036, -.003]). Answers (+.252): gate g .75 [.58,
     .93] against .26 (+.49 [.34, .65]); W_hh g .46 [.30, .62] against .20 (+.26 [.13, .40]); W_hh f+i .36 [.29, .44]
     against .42 (-.06 [-.16, .04]; its i rows alone above a random 1M of W_hh, +.08 [.01, .16], its f rows below, -.12
     [-.18, -.06]; against a random 2M of W, the cross-tensor baseline, +.11 [.03, .18]); W_ih .33 [.19, .47] against
     .60 (-.27 [-.41, -.13]); the read-out -.19. Holding share minus answers share, against the same difference of the
     matched random sets: W_hh f+i -.28 [-.36, -.20] against -.10 (-.17 [-.28, -.07]); gate g -.42 [-.59, -.24] against
     -.11 (-.31 [-.45, -.17]); W_ih -.32 [-.49, -.13] against -.21 (-.11 [-.27, .06]); W_hh g+o -.03 [-.20, .14] against
     -.10 (+.07 [-.02, .17]); the read-out +.34 [.25, .44] (no random counterpart). Distance out of a split: W_hh g
     +.186, read-out +.149 of +.539.
  5. 4,000-11,200: holding and coupling, against random sets. Holding family (+.330): W_ih +.111 [.032, .201] (.34)
     against .29 for a random 4M of W (+.05 [-.12, .21]; against its proportional share +.10 [-.07, .25]); W_hh f+i
     +.004 [-.019, .026] (.01) against .06 for a random 2M of W_hh (-.04 [-.15, .06]; against a random 2M of W -.12
     [-.23, -.01]); W_hh g+o +.044 [-.021, .114] (.13); W_hh g +.049 [-.016, .119] (.15 against .05 for a random 1M of
     W_hh: +.10 [-.04, .25]); gate g +.130 [.033, .237] (.39 against .13: +.26 [.06, .47]); W_ih g .26 [.07, .46]
     against .08 for a random 1M of W_ih (+.18 [.04, .32]); the read-in +.095 [.019, .162] (.29) and the read-out +.087
     [.030, .138] (.26), whole small tensors with no random counterpart (finding 11 for the read-out). Paired: W_ih, the
     read-in and the read-out each exceed W_hh f+i (+.107 [.021, .201], +.091 [.004, .169], +.083 [.012, .146]); none is
     separable from W_hh g+o (+.067 [-.037, .161], +.051 [-.074, .167], +.043 [-.079, .157]), and the read-in and the
     read-out are not separable from W_ih (-.016 [-.177, .125], -.024 [-.141, .082]). W_ih as a whole, against a random
     4M of W / against its proportional share: above both on the end-mass slope (+.244 [.098, .390]; +.129 [.034, .215]
     / +.163 [.064, .255]) and the excess angle (+.174; +.068 [.022, .113] / +.084 [.040, .124]), at both on held per
     variable (+.148 [.074, .222]; +.029 [-.032, .083] / +.037 [-.014, .088]) and joint held (+.038 [-.031, .104] /
     +.048 [-.016, .114]), below both on the direction slope (-.095 [-.152, -.030] / -.079 [-.134, -.019]). Its g rows
     lead: above a random 1M of W_ih on the end-mass slope (+.196 [.085, .312]; +.137 [.057, .218]), excess angle (+.106
     [.066, .147]; +.063 [.031, .096]), joint held (+.102 [.028, .174]; +.064 [.004, .122]), distance out (+.077; +.070
     [.016, .121]), not held per variable (+.074 [.004, .142]; +.040 [-.015, .095]); above the other W_ih blocks on the
     end-mass slope (+.177 to +.183, each interval > 0) and the excess angle (+.068 to +.093), not on held per variable
     (+.043 to +.055, each including 0). Coupling (+.286): W_ih .53, what a random 4M of W carries (.53; .00 [-.09,
     .10]). Direction slope (+.374): W_hh g +.120 [.071, .168], above a random 1M of W_hh (+.063 [.022, .103]) and above
     W_hh f (+.057 [.003, .104]); W_hh f +.063 [.040, .088], at a random 1M of W_hh (+.006 [-.017, .032]). The small
     answers change (+.095) is carried .96 by W_hh f+i (above a random 2M of W_hh by +.61 [.37, .86] of it), against
     the read-out (-.63).
  6. 11,200-39,996 (no random run; against the proportional share of the pool, finding 8): holding (+.130): W_ih +.115
     [.021, .233] against .011 for its share of W (+.104 [.031, .184]; .80 [.26, 1.47] of the change); W_hh g+o -.099
     [-.167, -.029] against -.046 (-.053 [-.086, -.018]); W_hh f+i +.006 [-.023, .033], above their proportional share
     (-.046; +.053 [.018, .086]) only because W_hh g+o take holding away (the share is half of W_hh's summed value, which
     W_hh g+o make negative): the f+i rows themselves add none; the read-out +.101 [.027, .164]. End-mass slope: W_ih
     +.292 against .021 (+.270 [.085, .489]); its g rows +.238 [.026, .488], above its o rows (paired difference +.257
     [.031, .500]; the o rows themselves -.019 [-.086, .059]); W_hh g -.127, W_hh o -.098 [-.158, -.040]. Coupling
     (+.209): W_ih +.079 against .063 (+.016 [-.014, .051]); the read-in +.070 [.035, .105]. The answers change (+.012)
     is below the share threshold.
  7. The task's hypothesis, against random sets of the same size (family shares, paired). W_hh f+i carry .01 [-.06, .08]
     of the 4,000-11,200 holding change against .06 (seeds -.02 to .10) for a random 2M of W_hh (-.04 [-.15, .06]) and
     .13 for a random 2M of W (-.12 [-.23, -.01]); W_ih carries .29 [.23, .35] of the 800-1,600 answers change against
     .51 (seeds .49-.52) for a random 4M of W (-.22 [-.28, -.16]; against its proportional share -.19 [-.25, -.13]), and
     the read-out -.08 of it - through the corner-answer row alone (-.274 [-.316, -.232] f); the read-out adds goal
     accuracy (+.052 [.024, .080]) and both right (+.047 [.025, .068]), so its half of H is mixed by row. Retracted: the
     first version read W_hh f+i's .23 [.19, .26] of the early answers change as substantial and W_ih + read-out's .60
     [.33, .84] of the late holding change as concentration ('mirror image'). At matched size W_hh f+i carry less of the
     early answers change than a random half of W_hh (.35; -.12 [-.16, -.08]; a random 2M of W .24: -.02 [-.05, .03])
     and W_ih what a random half of W carries of the late holding change. What does move is W_ih's standing: its excess
     over a random 4M of W rises from the early answers (-.22) to the late holding (+.05), paired change +.27 [.09,
     .44]; W_hh f+i's from -.12 to -.04 (+.08 [-.04, .19]).
  8. Controls. The random families carry their proportional share of their pool: seed mean minus (k / pool) x the pool's
     summed values is -.039 .. +.049 f over the 216 family x row x window cells where they ran (mean |.| .006; 5-95%
     excludes 0 in 34, the largest for the random 4M of W in 1,600-4,000: end-mass slope +.049, accuracy -.039); e.g. a
     random 4M of W on held per variable 4,000-11,200: +.119 against .111, a random 1M of W +.025 against .028. A
     class's value is a sum of single-group swaps and a random set is swapped as one unit; the two agree to second order
     only. The random half of W deviates systematically: the same sign in all 3 draws in 16 of its 36 cells (1 in 4
     expected by chance; |seed mean| / (draw sd / sqrt 3) > 4.30 in 8), per draw -.062 .. +.060 (end-mass slope
     1,600-4,000 +.060 / +.051 / +.037; 4,000-11,200 +.029 / +.058 / +.014; accuracy 1,600-4,000 -.062 / -.025 / -.029;
     holding family 1,600-4,000 +.030 / +.029 / +.021); the 1M and 2M families deviate less (|seed mean| at most .028,
     on average .0031-.0067 against .0106). W_ih is therefore also given against its proportional share, which is built
     like the class (W_IH AGAINST BOTH BASELINES; e.g. holding 1,600-4,000 -.093 [-.124, -.059] f against -.120 [-.155,
     -.080] for the random 4M; end-mass slope 4,000-11,200 +.163 [.064, .255] against +.129 [.034, .215]); no
     conclusion changes. The adversarial check's 'a random 4M of W (.088) is not 4x a random 1M (.030)' (one draw each,
     held per variable 4,000-11,200; the first version had no 4M set) was draw spread at that cell: the 3 draws give
     .125 / .129 / .104 (proportional share .111). Seed spread, random 1M of W: accuracy 800-1,600 .074 / .067 / .063,
     held per variable 4,000-11,200 .030 / .029 / .017; random 2k of biases on held per variable 4,000-11,200: -.003 /
     -.010 / -.005. The first version's control still tracks the mean W block (accuracy 800-1,600 +.074 vs +.068; held
     per variable 4,000-11,200 +.030 vs +.028; loss +.039 vs +.044 and +.012 vs +.012). Biases and the start state, on
     the rows of NUM_ROWS and the three family means (BIASES AND START STATE), carry |value| <= .02 f in .93 of the 128
     row x window x set cells after 800 (largest: biases on one-variable holding +.067 [.015, .101] and the start state
     on G +.057 [.005, .110], both 11,200-39,996) and in .78 of the 32 in 10-800 (largest: biases on G +.267 [.162,
     .358], on log q -.108 [-.124, -.094]); the g gate's own biases (unmatched in rW2M) carry |value| <= .007 f on the
     families and the loss in 800-11,200.
  9. Training loss (the run's own objective): W_hh g +.329 [.281, .377] of +1.417 (10-800), +.144 of +.408, +.086 of
     +.230, +.037 of +.114, +.012 of +.069; W_ih g +.330 (10-800), +.067, +.019, +.012, +.006; its interaction -.303
     [-.340, -.263] in 10-800 and <= .026 after. Gate g's share of the loss decrease is .48 / .52 / .46 / .43 / .24;
     against a random 2M of W .21 / .22 / .21 in 800-1,600 / 1,600-4,000 / 4,000-11,200 (+.31 [.27, .36], +.24 [.19,
     .29], +.22 [.14, .30]).
 10. Alpha paths (E -> L). Training loss inside the parents' range at every alpha: grafts 4 / 3 / 0 of 17 (800 -> 1,600:
     W_hh i, W_hh f, W_hh o, random W; 1,600 -> 4,000: two bias groups and the random bias set), rewinds 14 / 11 / 12 of
     17. Grafting one set of the later network into the earlier one moves the earlier network AWAY from the later
     behaviour (sum over the 15 groups: accuracy -.787 / -1.821 / -2.718; held per variable +.008 / -.166 / -1.086), and
     rewinding one set undoes much more than the change in sum (accuracy +2.989 / +2.909 / +2.487 vs +.653 / +.298 /
     +.061; held per variable +.364 / +1.282 / +2.458 vs +.118 / +.284 / +.331): necessity without sufficiency, i.e.
     co-adaptation. td_tt_swaps_alpha.png draws the training loss (f units, + = lower loss): rewinding W_hh g alone
     costs +.436 [.416, .454] / +.470 [.452, .489] / +.417 [.401, .433] against whole changes of +.408 / +.230 / +.114;
     also above the change: the read-in +.405 and W_ih g +.265 at 1,600 -> 4,000, the read-in +.350, W_ih g +.332, the
     read-out +.200 and W_ih o +.158 at 4,000 -> 11,200; the read-in +.396 [.376, .416] at 800 -> 1,600. The 15 groups'
     grafts gain at most +.015 (W_hh i, 800 -> 1,600; W_hh o +.006, W_hh f +.005) and lose for every other W block, the
     read-in and the read-out (down to -.370, W_hh g at 4,000 -> 11,200); the random 1M of W (3 draws): rewind +.055 /
     +.049 / +.052, graft +.028 / -.008 / -.030. The rewinds therefore describe dependence, not contribution, and serve
     as no control. Descriptively, at 4,000 -> 11,200 (held per variable, interpretable rewinds): W_ih i +.153 [.125,
     .184], W_hh o +.146, W_hh i +.123, W_ih f +.119, a random 1M of W +.102 (seeds .091-.115), W_hh f +.052 [.033,
     .071]; against the random seeds' mean (paired) the forget rows' rewind costs less (-.049 [-.069, -.029]; joint held
     -.076 [-.097, -.052]), the input rows' does not (+.021 [-.006, .046]). At 800 -> 1,600 the read-in alone, rewound,
     loses the whole accuracy change (+.654 [.600, .715], loss in range) although its adjacent-pair value is +.024. Path
     check (the long pair's (graft + rewind) / 2 vs the same set's adjacent-pair phi summed over the stretch, 15 groups
     x 8 rows): r = .89 / .71 / .60, slope .87-1.23, mean |difference| .013-.023 vs mean |value| .020-.024; long-pair
     values agree with the path only roughly, the less the longer the stretch.
 11. Read-out split, 4,000-11,200 (check against calibration leakage; the fixed-norm direction swap added in the final
     round). The whole read-out's change sharpens the goal belief (entropy at T -.069 [-.074, -.065] nats against a
     window change of +.002 [-.002, .005]). Its W_out part (entropy -.071 [-.075, -.065] nats) splits into a fixed-norm
     change of direction and a change of scale that partly cancel; direction + scale reproduce the W_out-only swap
     (end-mass slope +.142 = +.252 - .110) within .006 f on the 9 printed rows (.015 over all 26 rows; CHECKS). Direction
     (ro_dir, the other checkpoint's W_out at this one's norm): end-mass slope +.252 [.190, .320], held per variable
     +.126 [.086, .166], one-variable ends +.273 [.100, .428], joint +.107 [.059, .157], distance out +.006 [-.038,
     .048], goal accuracy -.052 [-.093, -.012], entropy +.065 [.060, .070] nats (flatter). Keeping the norm does not keep
     the calibration: this swap also flattens the belief, and calibration alone moves the held rows and the distance out
     (the scale swap), so these values mix the change of direction with a flatter belief; its joint-held and
     distance-out values are not read as direction. Scale (ro_scale, the direction unchanged): end-mass slope -.110
     [-.116, -.104], held per variable -.109 [-.117, -.099], one-variable -.119 [-.132, -.106], joint -.127 [-.140,
     -.112], distance out +.169 [.155, .182], entropy -.135 [-.137, -.133] nats (sharper). Whole read-out: end-mass
     slope +.132 [.071, .198] (W_out only +.142 [.081, .208], b_out only -.010), distance out +.173 [.132, .216], held
     per variable +.009 [-.027, .048], joint -.031 [-.079, .018], one-variable +.151 [-.024, .307]. So the direction
     swap's held splits and the sharpening's loss of them about cancel (held per variable +.126 against -.109).
     Leakage is ruled out for the end-mass slope: W_out's whole change sharpens the belief, and sharpening alone lowers
     the slope, yet the change raises it. It is not ruled out for the distance out, which this split does not divide
     between direction and calibration (the direction swap's +.006 includes its flatter belief).
 12. Gate gains (every pair of the two windows; mean [min-max over pairs]). 800-1,600: gate slope i / f / g / o .1589 /
     .1622 / .7680 [.7383-.8061] / .1559; RMS drive change of the W_hh blocks .1684 / .1314 / .0816 / .1745; RMS
     gate-output change from the W_hh blocks .0252 / .0203 / .0591 [.0460-.0757] / .0235, from the W_ih blocks .0194 /
     .0222 / .0358 / .0191. 4,000-11,200: slope .1509 / .1348 / .6709 / .1446; output change, W_hh .0134 / .0122 /
     .0276 / .0121, W_ih .0124 / .0130 / .0190 / .0115. W_hh g's output change is the largest of the 8 blocks in 10 of
     10 and 24 of 24 pairs.

FAILED / NULL
  - H (task): W_hh f/i rows carry the 4,000-11,000 holding gain; W_ih / read-out carry the 835-1,575 answers gain.
    Fails in both halves at matched size (finding 7; the read-out's part is mixed by row, it has no random counterpart).
  - Retracted from the first version: the 'mirror image' of H (W_hh f+i carry the early answers change, W_ih +
    read-out the late holding change): at matched size W_hh f+i carry less than a random half of W_hh early and W_ih
    what a random half of W carries late (finding 7). '3.1-4.5x a random block of the same size': that control mixed
    W_ih and W_hh entries and was one draw; against a random 1M of W_hh, W_hh g leads accuracy by +.124 [.076, .170],
    and against the gate-matched control its accuracy lead is not resolved (finding 3). 'Moved the least' as a mark of
    specificity: its weights moved least, its gate's output most (finding 12). 'Late holding from the input side and
    the read-out rather than the recurrent rows': not separable from W_hh g+o (finding 5). 'In 800-1,600 answers and
    holding use the same weights': not shown (finding 3).
  - Restated in the final round (labels that over-read the data): 'the read-out works against those answers' - only
    through the corner-answer row; it adds goal accuracy (finding 7). 'The late holding gain is concentrated in the g
    gate and the small read-in and read-out' - the read-in and read-out are whole tensors without a random counterpart
    and not separable from W_hh g+o or W_ih; only g rows exceed a random set (finding 5). 'The read-out holds no split'
    - a cancellation: the fixed-norm direction swap adds held per variable +.126 [.086, .166] (with a flatter belief),
    the sharpening removes -.109 [-.117, -.099] (finding 11). 'The end-mass-slope gain is carried by W_out's direction'
    - the fixed-norm direction swap gives +.252 [.190, .320] (with a flatter belief), and calibration does not explain
    W_out's +.142 (finding 11).
  - Restated in the labels round: the direction swap is fixed-norm, not a pure change of direction - it also flattens
    the goal belief (+.065 [.060, .070] nats), so its held, joint and distance-out values are not read as direction
    alone; 'the distance out is calibration of the same states' is withdrawn (finding 11).
  - Read-out leakage: ruled out for the end-mass slope, not for the distance out (finding 11).
  - Sufficiency of any single set's change over 1,600-11,200: none (finding 10).
  - A partition-level account of 10-800: the continuous rows are non-additive there (finding 2).

CHECKS (`checks` -> swaps/checks.json)
  The 15 groups cover all 8,046,020 weights once; every swap replaces exactly its set (bitwise) and swapping all 15
  groups in turn gives W_k+1 exactly; controls rand_W 1,000,000 (500,058 in W_ih, 499,942 in W_hh), rand_b 2,000. Random
  families: seed 1 of rW1M / rb2k rebuilt bit-identically as rand_W / rand_b; every set has exactly its k entries inside
  its pool (rG1M: rows 2,000-2,999 of W_ih and W_hh only); family swaps exact (bitwise; 3 sets tested); seeds 1 and 2 of
  rHH2M share 1,000,162 entries (k^2 / pool = 1,000,000 expected). Read-out split: ro_all reproduces the stored
  'readout' swaps exactly (0 in values and losses, 24 pairs); the fixed-norm direction swap keeps this checkpoint's
  W_out norm (relative difference 1.6e-10), takes the other's direction (cosine 1 - 6e-15) and leaves every other weight
  unchanged; direction + scale reproduce the W_out-only swap within .006 f on the 9 rows `numbers` prints (.015 over all
  26 rows, on the probes' one-variable held row; recomputed from ro/ and ro_dir/ in the labels round, not printed by
  `numbers`). Gate gains: the direct probe forward vs the engine's, max |d log q| 3.8e-04 (fp32, other matmul shapes).
  Intact goal accuracy = the stored ck trunc_acc (td_analysis) at 113 of 114 checkpoints, one probe of 10,000 apart at
  training 200 (an fp16 tie). Training-loss port vs the stored td_tt_entangled_learning losses (114 checkpoints): max
  per-episode relative difference 2.6e-5 (fp32 GPU vs CPU matmuls). Three intact checkpoints re-measured in another
  process: values, resamples and losses identical (0; rerun 11-02 18:50, after the engine's last edit at 18:27, the
  ro_dir runs' engine). Final round: `numbers` reproduces the fix round's output (2,062 of its 2,076 lines verbatim, the
  other 14 extended in place); every resolved figure point keeps its sign for each random draw (FIGURE block); the
  adversarial check's 4th random 4M draw (below) reproduced with its own code. Labels round: `numbers` differs from the
  final round's output in 2 labels only (every numeric token identical); the figures' plotted values are unchanged
  (td_tt_swaps.png: its panels pixel-identical). Bookkeeping, true by construction (not checks): change = Sigma phi +
  interaction (0 in values and resamples); telescoping 8.9e-16. Pair (39,000, 40,000) was computed twice (two workers
  met there; identical content).

CAVEATS
  One seed, one run. Two-endpoint Shapley uses 2 of the 15! orderings (exact for quadratic dependence; the interaction
  measures the rest). The attribution is relative to the checkpoint spacing (20-800 updates per pair); Adam steps
  between checkpoints are not resolved. Threshold rows (accuracy, held shares) move by near-tie flips. Shares are given
  only where the change's 5-95% excludes 0, |change| >= .05 and |interaction| <= |change| (the figure: >= .1); the
  family means are equal-weight means of rows on their own naive-Bayes scales. Random families: 3 draws each; the seed
  range is the range of 3 independent draws and understates the draw spread (the adversarial check's own 4th draw of a
  random 4M of W, seed 7, gives +.088 [.035, .139] on held per variable 4,000-11,200 against the module's .104-.129;
  reproduced from verify_weight_swaps/res_ctl3.npz with its code, not printed by `numbers`); the intervals cover the
  environments only. The read-in (16,000) and read-out (20,020) are whole tensors with no random counterpart: their
  values say what those tensors' changes did, not that they are concentrated per weight. A gate's 2,000 biases are
  unmatched in rW2M (the g gate's carry <= .007 f in 800-11,200). Not run (CTRL_NOT_RUN): every new family in 10-800;
  rIH1M, rG1M and seeds 2-3 of rW1M / rb2k in 1,600-4,000 and 11,200-39,996; rHH1M, rHH2M, rW2M and rW4M in
  11,200-39,996 (there: the proportional share, finding 6). Far errors and merged pairs start far outside the
  naive-Bayes range (10-800: +1.615 and +38.754 f). The probes are test environments; the loss is on training-
  distribution episodes (its own resamples). Long-pair grafts and rewinds mix function with co-adaptation (finding 10).

Run from the repository root (GPU 0 beside other jobs; 2 CPU threads per process):
    export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
    PY=/home/johns/anaconda3/envs/eureka/bin/python; S=main/REPLICATION/claude_plots/td_tt_swaps.py
    for j in 0 1 2; do $PY $S run --worker $j --nworkers 3 & done; wait     (resumable; a STOP file in the cache stops)
    ($PY $S run --kinds pair,alpha --reverse: a late helper that takes the missing units from the end)
    fix round: for j in 0 1; do $PY $S run --worker $j --nworkers 2 --kinds ro,alpha_ctrl,ctrl1,ctrl2 & done; wait
    final round: for j in 0 1; do CUDA_VISIBLE_DEVICES=$j $PY $S run --worker $j --nworkers 2 --kinds ro_dir & done; wait
    $PY $S checks; $PY $S numbers; $PY $S summary; MPLBACKEND=Agg $PY $S figure; MPLBACKEND=Agg $PY $S figure_alpha
  Cost (11-02, GPU 0 shared with other jobs): path 3,956 networks + alpha 408, median 5.4 s per network, 12:03-13:55
  (3 workers + 1 helper, ~1.1 GB GPU each). Fix round 15:48-17:13 (2 workers, then 1 + 1 helper; load 5-12 from other
  jobs): 1,928 random-family networks (median 4.3 s per network), 192 read-out split networks, 48 alpha networks,
  gate gains on 34 pairs. Final round 18:29-18:31: 48 fixed-norm direction read-out networks (4-5 s each; one worker
  per GPU, both shared). checks + numbers + summary + both figures ~2 min (1 min 42 s on 11-02 18:50; 7 min when
  loaded).
Cache: main/DATA/experimental/training_dynamics/through_training/swaps/ (intact/ck_EEEEE.npz, pairs/pair_E0_E1.npz,
alpha/alpha_E0_E1_<set>.npz (incl. rW1M_2 / rW1M_3), ctrl/<family>_<seed>/pair_E0_E1.npz, ro/pair_E0_E1.npz,
ro_dir/pair_E0_E1.npz: per network the 26 rows' values, their 1,000 resamples (float32) and the 2,000 per-episode
training losses (ro, ro_dir: + the goal entropy per environment); gains.npz; checks.json; summary.npz)."""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_intervene as ti                                                                  # noqa: E402
import td_tt_entangled_learning as tel                                                        # noqa: E402

OUT = os.path.join(gap.TT, "swaps")
TMP = "/tmp/claude-1002/-home-johns-Desktop-CognitiveGridworld/7af6b091-54ff-45a0-b4dd-ef0a3bc9c778/scratchpad/tt_tmp/swaps"
RUN = "trained"
T, R, NV = ti.T, ti.R, ti.NV
FINE = tuple(ti.groups())                         # the engine's 15 fine groups: a partition of every weight
CONTROLS = ("rand_W", "rand_b")                   # count-matched random entry sets (not part of the partition)
ALL_G = FINE + CONTROLS
NF, NG = len(FINE), len(ALL_G)
ROWS = ti.CHEAP + ti.FWD                          # 18 probe rows + 8 forward-set rows
NR = len(ROWS)
NB = gap.N_BOOT
CTRL = {"rand_W": (("LSTM.weight_ih_l0", "LSTM.weight_hh_l0"), 1_000_000, 20261204),     # = one gate block of W
        "rand_b": (("LSTM.bias_ih_l0", "LSTM.bias_hh_l0"), 2_000, 20261205)}             # = one gate's biases
WIN_X = (10., 800., 1600., 4000., 11200., 39996.)  # window edges, training episodes (checkpoints nearest the requested
                                                   # 835 / 1,575 / 4,000 / 11,000)
ALPHA_PAIRS = ((1000, 2000), (2000, 5000), (5000, 14000))   # raw: training 800 -> 1,600, 1,600 -> 4,000, 4,000 -> 11,200
ALPHAS = (.25, .5, .75, 1.)
BUDGET = 5000                                     # GPU rows per chunk (<= ~0.7 GB per forward)
LOSS_CHUNK = 500
LOSS_BOOT_SEED = 20261206                         # episode resamples of the training loss (its own units)
_C = {}


# ═══════════════════════════════════════════════════════════════════
# Weight sets: fine groups (td_tt_intervene) and count-matched random controls
# ═══════════════════════════════════════════════════════════════════

def control_masks():
    """{name: [(key, bool mask), ...]}: rand_W = 1,000,000 entries drawn without replacement from the 8,000,000 entries of
    W_ih and W_hh (the size of one gate block of either), rand_b = 2,000 of the 8,000 entries of b_ih and b_hh (the size
    of one gate's biases); fixed seeds"""
    if "masks" not in _C:
        w = ti.load(RUN, 0)
        out = {}
        for name, (keys, k, seed) in CTRL.items():
            sizes = [w[key].numel() for key in keys]
            m = np.zeros(sum(sizes), bool)
            m[np.random.default_rng(seed).choice(len(m), k, replace=False)] = True
            parts, o = [], 0
            for key, s in zip(keys, sizes):
                parts.append((key, torch.as_tensor(m[o:o + s].reshape(w[key].shape))))
                o += s
            out[name] = parts
        _C["masks"] = out
    return _C["masks"]


def entries(name):
    """[(key, index)] of a weight set: a row slice (fine groups) or a boolean mask (controls, random families)"""
    if name in FINE:
        return ti.groups()[name]
    if name in CONTROLS:
        return control_masks()[name]
    return family_mask(name)


# Size-matched random families (fix round, 11-02): k random entries of a pool, 3 seeds each. Seed 1 of rW1M / rb2k is
# the path experiment's rand_W / rand_b (rebuilt bit-identically by family_mask; `checks`).
G_ROWS = (2000, 3000)                             # gate g's rows of W_ih and W_hh (block 2 of i, f, g, o; H = 1,000)
CTRL_POOLS = {"W": (("LSTM.weight_ih_l0", None), ("LSTM.weight_hh_l0", None)),          # 8,000,000 entries
              "W_hh": (("LSTM.weight_hh_l0", None),), "W_ih": (("LSTM.weight_ih_l0", None),),   # 4,000,000 each
              "g rows": (("LSTM.weight_ih_l0", G_ROWS), ("LSTM.weight_hh_l0", G_ROWS)),    # 2,000,000
              "b": (("LSTM.bias_ih_l0", None), ("LSTM.bias_hh_l0", None))}               # 8,000
CTRL_FAM = {"rW1M": ("W", 1_000_000, (20261204, 20261311, 20261312)),
            "rb2k": ("b", 2_000, (20261205, 20261321, 20261322)),
            "rHH1M": ("W_hh", 1_000_000, (20261331, 20261332, 20261333)),
            "rIH1M": ("W_ih", 1_000_000, (20261341, 20261342, 20261343)),
            "rG1M": ("g rows", 1_000_000, (20261351, 20261352, 20261353)),
            "rHH2M": ("W_hh", 2_000_000, (20261361, 20261362, 20261363)),
            "rW2M": ("W", 2_000_000, (20261371, 20261372, 20261373)),
            "rW4M": ("W", 4_000_000, (20261381, 20261382, 20261383))}
CTRL_ALIAS = {"rW1M_1": "rand_W", "rb2k_1": "rand_b"}      # stored in the path experiment's pair files
FAM_LABEL = {"rW1M": "random 1M of W", "rb2k": "random 2k of biases", "rHH1M": "random 1M of W_hh",
             "rIH1M": "random 1M of W_ih", "rG1M": "random 1M of the g rows (W_ih + W_hh)",
             "rHH2M": "random 2M of W_hh", "rW2M": "random 2M of W", "rW4M": "random 4M of W"}


def fam_sets(fam):
    """the 3 set names of a random family (seed order)"""
    return tuple(f"{fam}_{j + 1}" for j in range(len(CTRL_FAM[fam][2])))


def family_mask(name):
    """[(key, bool mask)] of a family set 'fam_j': CTRL_FAM[fam][1] entries drawn without replacement from the
    concatenated (flattened, in pool order) entries of the pool, seed CTRL_FAM[fam][2][j - 1]"""
    key_ = ("fmask", name)
    if key_ not in _C:
        fam, j = name.rsplit("_", 1)
        pool, k, seeds = CTRL_FAM[fam]
        w = ti.load(RUN, 0)
        subs = []
        for key, rows in CTRL_POOLS[pool]:
            shp = tuple(w[key].shape)
            r0, r1 = (0, shp[0]) if rows is None else rows
            subs.append((key, shp, r0, r1, (r1 - r0) * int(np.prod(shp[1:]))))
        m = np.zeros(sum(s[4] for s in subs), bool)
        m[np.random.default_rng(seeds[int(j) - 1]).choice(len(m), k, replace=False)] = True
        out, o = [], 0
        for key, shp, r0, r1, n in subs:
            full = np.zeros(shp, bool)
            full[r0:r1] = m[o:o + n].reshape((r1 - r0,) + shp[1:])
            out.append((key, torch.as_tensor(full)))
            o += n
        _C[key_] = out
    return _C[key_]


def swap(wa, wb, name, alpha=1.):
    """wa, except the weight set `name` = (1 - alpha) wa + alpha wb (alpha 1: wb's values exactly); name: a fine group,
    a control, a random family set ('rHH1M_2', ...) or a read-out split variant (RO_VARIANTS)"""
    if name in FINE:
        return ti.hybrid(wa, wb, name, alpha)
    if name in RO_ALL:
        assert alpha == 1.
        return ro_swap(wa, wb, name)
    out = {k: v.clone() for k, v in wa.items()}
    a = float(alpha)
    for key, ix in entries(name):
        out[key][ix] = (1. - a) * wa[key][ix] + a * wb[key][ix]
    return out


def n_params(name):
    w = ti.load(RUN, 0) if "w0" not in _C else _C["w0"]
    _C["w0"] = w
    return int(sum(w[key][ix].numel() for key, ix in entries(name)))


def sq_disp(wa, wb, name):
    """|W_b - W_a|^2 over the set's entries (float64)"""
    if name in RO_ALL:
        return float("nan")
    return float(sum(((wb[key][ix].double() - wa[key][ix].double()) ** 2).sum() for key, ix in entries(name)))


# Read-out split (fix round; the read-out leakage check, 4,000-11,200): which part of the read-out's change carries its
# contribution. ro_all = W_out and b_out (the fine group 'readout', rerun with the goal entropy), ro_W = W_out only,
# ro_b = b_out only, ro_scale = W_out <- s W_out of the same network, s = <W_out,other, W_out> / |W_out|^2 (the
# least-squares scale of the other checkpoint's W_out on this one's: a pure change of the belief scale), b_out untouched
RO_VARIANTS = ("ro_all", "ro_W", "ro_b", "ro_scale")
RO_DIR = ("ro_dir",)                              # final round (stored apart, ro_dir/): W_out <- (|W_out| / |W_out,other|)
                                                  # W_out,other, the other checkpoint's W_out at this one's Frobenius
                                                  # norm (its direction only), b_out untouched
RO_ALL = RO_VARIANTS + RO_DIR
RO_WINDOW = 3                                     # 4,000-11,200


def ro_swap(wa, wb, v):
    out = dict(wa)
    kW, kb = "classifier_readout.weight", "classifier_readout.bias"
    if v in ("ro_all", "ro_W"):
        out[kW] = wb[kW].clone()
    if v in ("ro_all", "ro_b"):
        out[kb] = wb[kb].clone()
    if v == "ro_scale":
        Wa, Wb = wa[kW].double(), wb[kW].double()
        out[kW] = (float((Wa * Wb).sum() / (Wa * Wa).sum()) * Wa).float()
    if v == "ro_dir":
        Wa, Wb = wa[kW].double(), wb[kW].double()
        out[kW] = (float(Wa.norm() / Wb.norm()) * Wb).float()
    return out


# ═══════════════════════════════════════════════════════════════════
# Training loss on the training distribution (td_tt_entangled_learning.primal_loss, ported to the GPU)
# ═══════════════════════════════════════════════════════════════════

def _main_set():
    if "M" not in _C:
        _C["M"] = tel.episodes("main")
    return _C["M"]


def _loss_forward(p, inp):
    """td_tt_entangled_learning.forward without the tangents, on the device of p (copied: the original builds its
    tensors on the CPU). Per-step read-out logits (n, T, 20)"""
    n, H = len(inp), p["LSTM.weight_hh_l0"].shape[1]
    Wh, Wro = p["LSTM.weight_hh_l0"], p["classifier_readout.weight"]
    x = torch.relu(inp @ p["classifier_readin.weight"].T + p["classifier_readin.bias"])
    xp = x @ p["LSTM.weight_ih_l0"].T + p["LSTM.bias_ih_l0"] + p["LSTM.bias_hh_l0"]
    h = p["STM"].reshape(1, H).expand(n, H)
    c = p["LTM"].reshape(1, H).expand(n, H)
    ys = []
    for t in range(T):
        gi, gf, gg, go = (xp[:, t] + h @ Wh.T).chunk(4, -1)
        i, f, g, o = gi.sigmoid(), gf.sigmoid(), gg.tanh(), go.sigmoid()
        c = f * c + i * g
        h = o * c.tanh()
        ys.append(h @ Wro.T + p["classifier_readout.bias"])
    return torch.stack(ys, 1)


@torch.no_grad()
def train_loss(w, dev):
    """per-episode training loss on the main set M (2,000 training-distribution episodes, seed 20261111): (2,000,) sums
    over steps and values of the run's own loss (td_tt_entangled_learning: head_D, item_loss, D_min 0, float64 head);
    the mean loss L = sum / (2,000 x 30 x 10)"""
    E = _main_set()
    n = len(E["goal"])
    p = {k: v.to(dev, torch.float32) for k, v in w.items()}
    out = np.zeros(n)
    for c0 in range(0, n, LOSS_CHUNK):
        sl = slice(c0, min(n, c0 + LOSS_CHUNK))
        y = _loss_forward(p, tel._inp(E, sl).to(dev))
        D = tel.head_D(y.double().cumsum(1).cpu(), torch.as_tensor(E["goal"][sl]), torch.as_tensor(E["log_pg"][sl]))
        out[sl] = tel.item_loss(D, tel.DMIN).sum(1).numpy()
    return out


LOSS_NORM = 2000 * T * R


# ═══════════════════════════════════════════════════════════════════
# Gate gains (fix round): how far each block's change moves its gate's OUTPUT
# ═══════════════════════════════════════════════════════════════════

GAIN_CHUNK = 5000


@torch.no_grad()
def gate_gains(wa, wb, dev):
    """W_a run on the 10,000 probes (a direct per-step LSTM, the engine's cell); per gate block G (i, f, g, o) of W_ih
    and W_hh the change of the block's drive, d_t = (W_b - W_a)_G x_t (W_ih; x_t = relu(read-in [o_t, Z])) or
    (W_b - W_a)_G u_(t-1) (W_hh), and its effect on the gate's output, slope_G(pre_t) * d_t (sigma' for i, f, o; tanh'
    for g); RMS over probes, steps and units. Returns dict(slope (4,) mean gate slope, pre_ih / pre_hh / out_ih /
    out_hh (4,) RMS, logq_maxdiff vs the engine's forward of W_a)"""
    P = ti.probes()
    n = len(P["env"])
    p = {k: v.to(dev, torch.float32) for k, v in wa.items()}
    dWi = (wb["LSTM.weight_ih_l0"] - wa["LSTM.weight_ih_l0"]).to(dev)
    dWh = (wb["LSTM.weight_hh_l0"] - wa["LSTM.weight_hh_l0"]).to(dev)
    H = p["LSTM.weight_hh_l0"].shape[1]
    Zall = torch.as_tensor(np.asarray(P["Z"], np.float32), device=dev).reshape(len(P["Z"]), -1)
    acc = {k: torch.zeros(4, dtype=torch.float64, device=dev) for k in ("slope", "pre_ih", "pre_hh", "out_ih", "out_hh")}
    lq_all = np.zeros((n, T, NV, R), np.float32)
    b = p["LSTM.bias_ih_l0"] + p["LSTM.bias_hh_l0"]
    for c0 in range(0, n, GAIN_CHUNK):
        sl_ = slice(c0, min(n, c0 + GAIN_CHUNK))
        m = sl_.stop - sl_.start
        obs = torch.as_tensor(np.asarray(P["obs"][sl_], np.float32), device=dev)
        zin = Zall[torch.as_tensor(np.asarray(P["env"][sl_], np.int64), device=dev)]
        h = p["STM"].reshape(1, H).expand(m, H)
        c = p["LTM"].reshape(1, H).expand(m, H)
        lg = torch.zeros(m, T, NV * R, device=dev)
        for t in range(T):
            x = torch.relu(torch.cat([obs[:, t], zin], -1) @ p["classifier_readin.weight"].T
                           + p["classifier_readin.bias"])
            pre = x @ p["LSTM.weight_ih_l0"].T + b + h @ p["LSTM.weight_hh_l0"].T
            dpi, dph = x @ dWi.T, h @ dWh.T
            gi, gf, gg, go = pre.chunk(4, -1)
            i, f, g, o = gi.sigmoid(), gf.sigmoid(), gg.tanh(), go.sigmoid()
            slopes = (i * (1 - i), f * (1 - f), 1 - g * g, o * (1 - o))
            for k in range(4):
                a_, h_ = dpi[:, k * H:(k + 1) * H], dph[:, k * H:(k + 1) * H]
                acc["slope"][k] += slopes[k].double().sum()
                acc["pre_ih"][k] += (a_.double() ** 2).sum()
                acc["pre_hh"][k] += (h_.double() ** 2).sum()
                acc["out_ih"][k] += ((slopes[k] * a_).double() ** 2).sum()
                acc["out_hh"][k] += ((slopes[k] * h_).double() ** 2).sum()
            c = f * c + i * g
            h = o * c.tanh()
            lg[:, t] = h @ p["classifier_readout.weight"].T + p["classifier_readout.bias"]
        lq_all[sl_] = torch.log_softmax(lg.double().cumsum(1).reshape(m, T, NV, R), -1).float().cpu().numpy()
    N = float(n * T * H)
    out = {k: (v / N).cpu().numpy() for k, v in acc.items()}
    for k in ("pre_ih", "pre_hh", "out_ih", "out_hh"):
        out[k] = np.sqrt(out[k])
    out["logq_maxdiff"] = float(np.abs(lq_all - ti.forward(wa, device=dev, budget=BUDGET)["logq"]).max())
    return out


def gains_cache(dev=None):
    """gate gains of every consecutive pair of the windows 800-1,600 and 4,000-11,200 (-> OUT/gains.npz)"""
    path = os.path.join(OUT, "gains.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    dev = ti._device(dev)
    WP = _window_pair_eps()
    rows = []
    for wi in (1, 3):
        for a, b in WP[wi]:
            g_ = gate_gains(ti.load(RUN, a), ti.load(RUN, b), dev)
            rows.append((wi, a, b, g_))
            print(f"  gains {a} -> {b}: slope {np.round(g_['slope'], 3)}, out_hh {np.round(g_['out_hh'], 4)}, "
                  f"out_ih {np.round(g_['out_ih'], 4)}, logq vs engine {g_['logq_maxdiff']:.1e}", flush=True)
    out = dict(win=np.array([r[0] for r in rows]), ep=np.array([(r[1], r[2]) for r in rows]))
    for k in ("slope", "pre_ih", "pre_hh", "out_ih", "out_hh", "logq_maxdiff"):
        out[k] = np.array([r[3][k] for r in rows])
    np.savez(path, **out)
    return out


# ═══════════════════════════════════════════════════════════════════
# One network: behaviour rows (probes + forward set), their resamples, the training loss
# ═══════════════════════════════════════════════════════════════════

def measure(w, dev, extra=False):
    """dict(val (NR,) float64, boot (NR, 1000) float32 on td_tt_gap.boot_counts(), loss (2,000,) per-episode training
    loss, sec): ROWS = td_tt_intervene's cheap probe rows + its forward-set rows, of the network w; extra: also ent_env
    (50,) the per-environment sums of the goal marginal's entropy at T on the probes (nats; counts: 200 per environment)"""
    t0 = time.time()
    fo = ti.forward(w, device=dev, budget=BUDGET)
    ff = ti.forward(w, device=dev, episodes=ti.forward_set(), budget=BUDGET)
    loss = train_loss(w, dev)
    rows = ti.score(fo["logq"], "cheap", right=fo["right"])
    rows.update(ti.score_fwd(ff["logq"]))
    bt = ti.resample(rows)
    out = dict(val=np.array([rows[k]["value"] for k in ROWS]), boot=np.stack([bt[k] for k in ROWS]).astype(np.float32),
               loss=loss, sec=time.time() - t0)
    if extra:
        out["ent_env"] = goal_entropy_env(fo["logq"])
    return out


def goal_entropy_env(logq):
    """(50,) per-environment sums over the probes of the goal marginal's entropy at T (nats, float64)"""
    P = ti.probes()
    lq = logq[np.arange(len(logq)), -1, np.asarray(P["goal"], np.int64)].astype(np.float64)       # (n, R)
    return np.bincount(np.asarray(P["env"], np.int64), -(np.exp(lq) * lq).sum(-1), minlength=gap.N_ENV)


# ═══════════════════════════════════════════════════════════════════
# Work units (resumable; one npz per unit; workers split the list)
# ═══════════════════════════════════════════════════════════════════

def checkpoints():
    return ti.checkpoints(RUN)


CTRL_TIERS = {    # random families run per tier: (family, seeds (1-based), windows (indices into the 5 windows)), in order
    "ctrl1": (("rHH1M", (1, 2, 3), (1, 3)), ("rHH2M", (1, 2, 3), (1, 3)), ("rW2M", (1, 2, 3), (1, 3)),
              ("rW4M", (1, 2, 3), (1, 3)), ("rIH1M", (1, 2, 3), (1, 3)), ("rG1M", (1, 2, 3), (1, 3)),
              ("rW1M", (2, 3), (1, 3)), ("rb2k", (2, 3), (1, 3))),
    "ctrl2": (("rHH2M", (1, 2, 3), (2,)), ("rW4M", (1, 2, 3), (2,)), ("rW2M", (1, 2, 3), (2,)),
              ("rHH1M", (1, 2, 3), (2,))),
    "ctrl2b": (("rHH2M", (1, 2, 3), (4,)), ("rW4M", (1, 2, 3), (4,)))}
CTRL_NOT_RUN = ("left out for time (GPU 0 shared): every new family in 10-800 (non-additive for the continuous rows; "
                "only seed 1 of rW1M / rb2k, the path runs, exists there); rIH1M, rG1M and seeds 2-3 of rW1M / rb2k in "
                "1,600-4,000 and 11,200-39,996; rHH1M, rHH2M, rW2M and rW4M in 11,200-39,996 (tier ctrl2b, not run: "
                "there the proportional share stands in)")
ALPHA_CTRL = ("rW1M_2", "rW1M_3")                 # extra seeds of the random 1M of W on the alpha paths


def _window_pair_eps():
    """per window: [(E0, E1)] raw episodes of its consecutive checkpoint pairs"""
    eps = checkpoints()
    return [[(eps[k], eps[k + 1]) for k in P_] for P_ in window_pairs(gap.train_x(np.array(eps)))]


def units(kinds=("intact", "pair", "alpha")):
    eps = checkpoints()
    U = []
    for kind in kinds:
        if kind == "intact":
            U += [("intact", (e,)) for e in eps]
        elif kind == "pair":
            U += [("pair", (a, b)) for a, b in zip(eps[:-1], eps[1:])]
        elif kind == "alpha":
            U += [("alpha", (a, b, g)) for a, b in ALPHA_PAIRS for g in ALL_G]
        elif kind == "alpha_ctrl":
            U += [("alpha", (a, b, g)) for a, b in ALPHA_PAIRS for g in ALPHA_CTRL]
        elif kind in CTRL_TIERS:
            WP = _window_pair_eps()
            for fam, seeds, wins in CTRL_TIERS[kind]:
                for wi in wins:
                    for j in seeds:
                        U += [("ctrl", (a, b, f"{fam}_{j}")) for a, b in WP[wi]]
        elif kind in ("ro", "ro_dir"):
            U += [(kind, (a, b)) for a, b in _window_pair_eps()[RO_WINDOW]]
        else:
            raise KeyError(kind)
    return U


def unit_path(u):
    kind, a = u
    if kind == "intact":
        return os.path.join(OUT, "intact", f"ck_{a[0]:05d}.npz")
    if kind == "pair":
        return os.path.join(OUT, "pairs", f"pair_{a[0]:05d}_{a[1]:05d}.npz")
    if kind == "ctrl":
        return os.path.join(OUT, "ctrl", a[2], f"pair_{a[0]:05d}_{a[1]:05d}.npz")
    if kind in ("ro", "ro_dir"):
        return os.path.join(OUT, kind, f"pair_{a[0]:05d}_{a[1]:05d}.npz")
    return os.path.join(OUT, "alpha", f"alpha_{a[0]:05d}_{a[1]:05d}_{a[2]}.npz")


def _save(path, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **kw)
    os.replace(tmp, path)


def _stack(ms):
    return dict(val=np.stack([m["val"] for m in ms]), boot=np.stack([m["boot"] for m in ms]),
                loss=np.stack([m["loss"] for m in ms]), sec=np.array([m["sec"] for m in ms]))


def _stack_x(ms):
    out = _stack(ms)
    out["ent_env"] = np.stack([m["ent_env"] for m in ms])
    return out


def do_unit(u, dev):
    kind, a = u
    t0 = time.time()
    if kind == "intact":
        m = measure(ti.load(RUN, a[0]), dev)
        _save(unit_path(u), ep=a[0], x=float(gap.train_x(a[0])), rows=np.array(ROWS), **m)
    elif kind == "pair":
        wa, wb = ti.load(RUN, a[0]), ti.load(RUN, a[1])
        A = _stack([measure(swap(wa, wb, g), dev) for g in ALL_G])          # W_k with g <- W_k+1
        B = _stack([measure(swap(wb, wa, g), dev) for g in ALL_G])          # W_k+1 with g <- W_k
        _save(unit_path(u), ep=np.array(a), x=gap.train_x(np.array(a)), groups=np.array(ALL_G), rows=np.array(ROWS),
              sq=np.array([sq_disp(wa, wb, g) for g in ALL_G]), **{f"A_{k}": v for k, v in A.items()},
              **{f"B_{k}": v for k, v in B.items()})
    elif kind == "ctrl":                                                 # one random family set on one pair
        wa, wb = ti.load(RUN, a[0]), ti.load(RUN, a[1])
        A, B = measure(swap(wa, wb, a[2]), dev), measure(swap(wb, wa, a[2]), dev)
        _save(unit_path(u), ep=np.array(a[:2]), x=gap.train_x(np.array(a[:2])), group=a[2], rows=np.array(ROWS),
              sq=sq_disp(wa, wb, a[2]), **{f"A_{k}": v for k, v in A.items()}, **{f"B_{k}": v for k, v in B.items()})
    elif kind == "ro":                                                   # the read-out split on one pair
        wa, wb = ti.load(RUN, a[0]), ti.load(RUN, a[1])
        A = _stack_x([measure(swap(wa, wb, v), dev, extra=True) for v in RO_VARIANTS])
        B = _stack_x([measure(swap(wb, wa, v), dev, extra=True) for v in RO_VARIANTS])
        ent_I = np.stack([goal_entropy_env(ti.forward(w_, device=dev, budget=BUDGET)["logq"]) for w_ in (wa, wb)])
        _save(unit_path(u), ep=np.array(a), x=gap.train_x(np.array(a)), variants=np.array(RO_VARIANTS),
              rows=np.array(ROWS), ent_I=ent_I, **{f"A_{k}": v for k, v in A.items()},
              **{f"B_{k}": v for k, v in B.items()})
    elif kind == "ro_dir":                                               # W_out's direction only (final round)
        wa, wb = ti.load(RUN, a[0]), ti.load(RUN, a[1])
        A = _stack_x([measure(swap(wa, wb, v), dev, extra=True) for v in RO_DIR])
        B = _stack_x([measure(swap(wb, wa, v), dev, extra=True) for v in RO_DIR])
        _save(unit_path(u), ep=np.array(a), x=gap.train_x(np.array(a)), variants=np.array(RO_DIR),
              rows=np.array(ROWS), **{f"A_{k}": v for k, v in A.items()}, **{f"B_{k}": v for k, v in B.items()})
    elif kind == "alpha":
        wE, wL = ti.load(RUN, a[0]), ti.load(RUN, a[1])
        g = a[2]
        G_ = _stack([measure(swap(wE, wL, g, al), dev) for al in ALPHAS])   # graft: E with g -> (1 - al) E + al L
        R_ = _stack([measure(swap(wL, wE, g, al), dev) for al in ALPHAS])   # rewind: L with g -> (1 - al) L + al E
        _save(unit_path(u), ep=np.array(a[:2]), x=gap.train_x(np.array(a[:2])), group=g, alphas=np.array(ALPHAS),
              rows=np.array(ROWS), sq=sq_disp(wE, wL, g), **{f"graft_{k}": v for k, v in G_.items()},
              **{f"rewind_{k}": v for k, v in R_.items()})
    return time.time() - t0


def worker(j, nw, kinds, reverse=False):
    """units U[j::nw] in order; reverse: every unit not on disk yet, from the end of the list (a late helper; a unit can
    then be computed twice at the meeting point: identical content, atomic writes)"""
    dev = ti._device()
    U = units(kinds)
    mine = U[::-1] if reverse else U[j::nw]
    todo = [u for u in mine if not os.path.exists(unit_path(u))]
    print(f"worker {j}/{nw} on {dev}: {len(todo)} of {len(mine)} units to do ({len(U)} in all)", flush=True)
    for i, u in enumerate(todo):
        if os.path.exists(os.path.join(OUT, "STOP")):
            print("STOP file: exiting", flush=True)
            return
        if os.path.exists(unit_path(u)):
            continue
        s = do_unit(u, dev)
        print(f"  [{i + 1}/{len(todo)}] {u[0]} {u[1]}: {s:.1f} s  ({time.strftime('%H:%M:%S')})", flush=True)
    print("worker done", flush=True)


# ═══════════════════════════════════════════════════════════════════
# References (naive, Bayes) and the gap-fraction scale
# ═══════════════════════════════════════════════════════════════════

def references():
    """naive / Bayes levels of ROWS on the same probes and forward set (td_tt_intervene.refs), full sample (NR,) and in
    the shared resamples (NR, NB); gap = Bayes - naive (the gap-fraction scale: f = (y - naive) / gap, so a change dy
    is df = dy / gap, additive); the training loss's references (naive's goal marginal on M; Bayes = the loss floor)"""
    if "refs" not in _C:
        rf = ti.refs("full")
        out = {}
        for nm in ("naive", "bayes"):
            rows = {k: rf[nm][k] for k in ROWS}
            bt = ti.resample(rows)
            out[nm] = np.array([rows[k]["value"] for k in ROWS])
            out[nm + "_boot"] = np.stack([bt[k] for k in ROWS])
        out["gap"] = out["bayes"] - out["naive"]
        out["gap_boot"] = out["bayes_boot"] - out["naive_boot"]
        nl = tel.naive_loss()[0].sum(1)                                     # (2,000,) naive's per-episode loss sums
        out["loss_naive_ep"] = nl
        out["loss_naive"] = nl.sum() / LOSS_NORM
        out["loss_bayes"] = tel.BAYES_LOSS
        _C["refs"] = out
    return _C["refs"]


def loss_counts():
    """(NB, 2,000) episode resample counts of the training-loss set M (its own units: every episode is its own pair of
    variables; seed 20261206); not paired with the probe resamples"""
    rng = np.random.default_rng(LOSS_BOOT_SEED)
    return rng.multinomial(2000, np.full(2000, 1 / 2000), size=NB).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════
# Collect the runs; two-endpoint Shapley decomposition; windows
# ═══════════════════════════════════════════════════════════════════

def collect(partial=False):
    """every stored unit of the path experiment: eps, x (K,); I_val (K, NR), I_boot (K, NR, NB) float32, I_loss (K,
    2,000); pairs: A_* (W_k with g <- W_k+1) and B_* (W_k+1 with g <- W_k): _val (P, NG, NR), _boot (P, NG, NR, NB)
    float32, _loss (P, NG, 2,000); sq (P, NG) squared displacement of each set; done (P,) bool"""
    eps = checkpoints()
    K, P = len(eps), len(eps) - 1
    D = dict(eps=np.array(eps), x=gap.train_x(np.array(eps)), I_val=np.full((K, NR), np.nan),
             I_boot=np.full((K, NR, NB), np.nan, np.float32), I_loss=np.full((K, 2000), np.nan),
             I_done=np.zeros(K, bool), done=np.zeros(P, bool), sq=np.full((P, NG), np.nan))
    for s in ("A", "B"):
        D.update({f"{s}_val": np.full((P, NG, NR), np.nan), f"{s}_boot": np.full((P, NG, NR, NB), np.nan, np.float32),
                  f"{s}_loss": np.full((P, NG, 2000), np.nan)})
    for k, e in enumerate(eps):
        p = unit_path(("intact", (e,)))
        if os.path.exists(p):
            z = np.load(p)
            assert list(z["rows"]) == list(ROWS)
            D["I_val"][k], D["I_boot"][k], D["I_loss"][k], D["I_done"][k] = z["val"], z["boot"], z["loss"], True
    for k in range(P):
        p = unit_path(("pair", (eps[k], eps[k + 1])))
        if os.path.exists(p):
            z = np.load(p)
            assert list(z["groups"]) == list(ALL_G) and list(z["rows"]) == list(ROWS)
            for s in ("A", "B"):
                for f in ("val", "boot", "loss"):
                    D[f"{s}_{f}"][k] = z[f"{s}_{f}"]
            D["sq"][k], D["done"][k] = z["sq"], True
    if not partial:
        assert D["I_done"].all() and D["done"].all(), (f"missing units: {int((~D['I_done']).sum())} intact, "
                                                       f"{int((~D['done']).sum())} pairs")
    return D


def window_pairs(x):
    """pair indices of each window: pairs (k, k + 1) between the window's edge checkpoints (WIN_X)"""
    out = []
    for lo, hi in zip(WIN_X[:-1], WIN_X[1:]):
        k0, k1 = int(np.flatnonzero(x == lo)[0]), int(np.flatnonzero(x == hi)[0])
        out.append(np.arange(k0, k1))
    return out


def win_label(w):
    a, b = WIN_X[w], WIN_X[w + 1]
    f = (lambda v: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")        # noqa: E731
    return f"{f(a) if a > 10 else '0'}-{f(b) if b < 39996 else '40k'}"


def decompose(D=None, pairs=None):
    """two-endpoint Shapley values pooled over a set of pairs (default: each window). Per pair k and set g:
    dA = m(W_k with g <- W_k+1) - m(W_k), dB = m(W_k+1) - m(W_k+1 with g <- W_k), phi = (dA + dB) / 2; the change
    m(W_k+1) - m(W_k) = sum of phi over the 15 fine groups + the interaction. Pooled = sums over the pairs (the change
    telescopes to the window's endpoints). Returns a list (one per pair set) of dicts with, in the rows' units and in
    gap-fraction units (suffix _f = / (Bayes - naive), per resample with that resample's references): tot (NR,),
    phi / dA / dB (NG, NR), inter (NR,), each with _boot (..., NB); and the training loss (L units, mean per item and
    value; _f with naive's and Bayes's loss; _boot over the episode resamples)"""
    D = collect() if D is None else D
    rf = references()
    C = loss_counts()
    sets = window_pairs(D["x"]) if pairs is None else pairs
    out = []
    for P_ in sets:
        k0, k1 = int(P_[0]), int(P_[-1]) + 1
        assert np.array_equal(P_, np.arange(k0, k1)), "pairs must be consecutive"
        iv, ib = D["I_val"], D["I_boot"]
        r = dict(pairs=P_, x0=float(D["x"][k0]), x1=float(D["x"][k1]), n_pairs=len(P_))
        r["tot"] = iv[k1] - iv[k0]
        r["tot_boot"] = ib[k1].astype(np.float64) - ib[k0]
        dA = (D["A_val"][P_] - iv[P_][:, None]).sum(0)
        dB = (iv[P_ + 1][:, None] - D["B_val"][P_]).sum(0)
        dAb = np.zeros((NG, NR, NB))
        dBb = np.zeros((NG, NR, NB))
        for k in P_:
            dAb += D["A_boot"][k].astype(np.float64) - ib[k][None].astype(np.float64)
            dBb += ib[k + 1][None].astype(np.float64) - D["B_boot"][k]
        r.update(dA=dA, dB=dB, phi=(dA + dB) / 2, dA_boot=dAb, dB_boot=dBb, phi_boot=(dAb + dBb) / 2)
        r["inter"] = r["tot"] - r["phi"][:NF].sum(0)
        r["inter_boot"] = r["tot_boot"] - r["phi_boot"][:NF].sum(0)
        g, gb = rf["gap"], rf["gap_boot"]
        for k in ("tot", "inter"):
            r[k + "_f"] = r[k] / g
            r[k + "_f_boot"] = r[k + "_boot"] / gb
        for k in ("phi", "dA", "dB"):
            r[k + "_f"] = r[k] / g
            r[k + "_f_boot"] = r[k + "_boot"] / gb[None]
        # training loss: per episode, then summed over pairs; resampled over episodes
        lI = D["I_loss"]
        eA = (D["A_loss"][P_] - lI[P_][:, None]).sum(0)                  # (NG, 2,000)
        eB = (lI[P_ + 1][:, None] - D["B_loss"][P_]).sum(0)
        e_phi = (eA + eB) / 2
        e_tot = lI[k1] - lI[k0]
        lg = rf["loss_bayes"] - rf["loss_naive"]
        lgb = (rf["loss_bayes"] * 2000 - C @ rf["loss_naive_ep"] / (T * R)) / 2000     # resampled naive level, per item
        r.update(L_tot=e_tot.sum() / LOSS_NORM, L_tot_boot=C @ e_tot / LOSS_NORM, L_phi=e_phi.sum(1) / LOSS_NORM,
                 L_phi_boot=(C @ e_phi.T).T / LOSS_NORM, L_dA_f=eA.sum(1) / LOSS_NORM / lg,
                 L_dB_f=eB.sum(1) / LOSS_NORM / lg)
        r["L_inter"] = r["L_tot"] - r["L_phi"][:NF].sum()
        r["L_inter_boot"] = r["L_tot_boot"] - r["L_phi_boot"][:NF].sum(0)
        for k in ("L_tot", "L_phi", "L_inter"):
            r[k + "_f"] = r[k] / lg
            r[k + "_f_boot"] = r[k + "_boot"] / lgb
        r["sq"] = D["sq"][P_].sum(0)                                      # summed squared displacement per set
        out.append(r)
    return out


def ci(b, axis=-1):
    """5-95% of resamples along axis (NaN where every resample is NaN, e.g. band4 in 10-800)"""
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(b, 5, axis=axis), np.nanpercentile(b, 95, axis=axis)


# ═══════════════════════════════════════════════════════════════════
# Alpha paths between checkpoints further apart: graft (sufficiency) and rewind (necessity)
# ═══════════════════════════════════════════════════════════════════

def alpha_collect(D=None, sets=ALL_G):
    """{(E0, E1): dict} per alpha pair: alphas (5,) = 0, .25, .5, .75, 1; graft (E with g -> (1 - a) E + a L) and rewind
    (L with g -> (1 - a) L + a E): _val (NG, 5, NR), _boot (NG, 5, NR, NB), _loss (NG, 5) mean training loss; alpha 0 =
    the parent (E for graft, L for rewind); parents' rows; sq (NG,) squared displacement E -> L per set (NG = len(sets);
    sets: ALL_G, or the extra random seeds ALPHA_CTRL)"""
    D = collect(partial=True) if D is None else D
    out = {}
    nS = len(sets)
    for a, b in ALPHA_PAIRS:
        kE, kL = int(np.flatnonzero(D["eps"] == a)[0]), int(np.flatnonzero(D["eps"] == b)[0])
        r = dict(eps=(a, b), x=(float(D["x"][kE]), float(D["x"][kL])), kE=kE, kL=kL, sets=tuple(sets),
                 alphas=np.array((0.,) + ALPHAS), sq=np.full(nS, np.nan), done=np.zeros(nS, bool))
        for d, k in (("graft", kE), ("rewind", kL)):
            r[f"{d}_val"] = np.full((nS, 5, NR), np.nan)
            r[f"{d}_boot"] = np.full((nS, 5, NR, NB), np.nan)
            r[f"{d}_loss"] = np.full((nS, 5), np.nan)
            r[f"{d}_loss_ep"] = np.full((nS, 5, 2000), np.nan)
            r[f"{d}_val"][:, 0] = D["I_val"][k]
            r[f"{d}_boot"][:, 0] = D["I_boot"][k]
            r[f"{d}_loss"][:, 0] = D["I_loss"][k].sum() / LOSS_NORM
            r[f"{d}_loss_ep"][:, 0] = D["I_loss"][k]
        for gi, g in enumerate(sets):
            p = unit_path(("alpha", (a, b, g)))
            if not os.path.exists(p):
                continue
            z = np.load(p)
            assert str(z["group"]) == g and list(z["rows"]) == list(ROWS)
            for d in ("graft", "rewind"):
                r[f"{d}_val"][gi, 1:] = z[f"{d}_val"]
                r[f"{d}_boot"][gi, 1:] = z[f"{d}_boot"]
                r[f"{d}_loss"][gi, 1:] = z[f"{d}_loss"].sum(1) / LOSS_NORM
                r[f"{d}_loss_ep"][gi, 1:] = z[f"{d}_loss"]
            r["sq"][gi], r["done"][gi] = float(z["sq"]), True
        r["LE"], r["LL"] = D["I_loss"][kE].sum() / LOSS_NORM, D["I_loss"][kL].sum() / LOSS_NORM
        out[(a, b)] = r
    return out


def alpha_summary(A=None, D=None, sets=ALL_G):
    """per alpha pair, set and row (gap-fraction units, f = (y - naive) / gap):
    graft = f(E with g <- L) - f(E) (sufficiency), rewind = f(L) - f(L with g <- E) (necessity), both at alpha 1, with
    5-95%; tot = f(L) - f(E); long = (graft + rewind) / 2 (the long pair's two-endpoint Shapley value); compatible[d] =
    the training loss at every alpha between the parents' losses (inclusive); monotone[d][row] = the row's full-sample
    alpha curve monotone (non-strict); path = the same set's adjacent-pair Shapley values summed over the window E -> L"""
    D = collect() if D is None else D
    A = alpha_collect(D, sets) if A is None else A
    rf = references()
    g, gb = rf["gap"], rf["gap_boot"]
    out = {}
    for key, r in A.items():
        s = dict(eps=key, x=r["x"], done=r["done"], LE=r["LE"], LL=r["LL"], sq=r["sq"])
        fv = {d: (r[f"{d}_val"] - rf["naive"]) / g for d in ("graft", "rewind")}            # (NG, 5, NR)
        fb = {d: (r[f"{d}_boot"] - rf["naive_boot"][None, None]) / gb[None, None] for d in ("graft", "rewind")}
        s["f_val"], s["f_boot"] = fv, fb
        s["tot"] = fv["rewind"][0, 0] - fv["graft"][0, 0]
        s["tot_boot"] = fb["rewind"][0, 0] - fb["graft"][0, 0]
        s["graft"] = fv["graft"][:, -1] - fv["graft"][:, 0]                                   # (NG, NR)
        s["graft_boot"] = fb["graft"][:, -1] - fb["graft"][:, 0]
        s["rewind"] = fv["rewind"][:, 0] - fv["rewind"][:, -1]
        s["rewind_boot"] = fb["rewind"][:, 0] - fb["rewind"][:, -1]
        s["long"] = (s["graft"] + s["rewind"]) / 2
        s["long_boot"] = (s["graft_boot"] + s["rewind_boot"]) / 2
        C = loss_counts()
        lg = rf["loss_bayes"] - rf["loss_naive"]
        lgb = rf["loss_bayes"] - C @ rf["loss_naive_ep"] / LOSS_NORM
        eg, er = r["graft_loss_ep"], r["rewind_loss_ep"]                                   # (NG, 5, 2,000)
        s["L_tot"] = (er[0, 0] - eg[0, 0]).sum() / LOSS_NORM / lg
        s["L_tot_boot"] = C @ (er[0, 0] - eg[0, 0]) / LOSS_NORM / lgb
        s["L_graft"] = (eg[:, -1] - eg[:, 0]).sum(-1) / LOSS_NORM / lg
        s["L_graft_boot"] = (C @ (eg[:, -1] - eg[:, 0]).T).T / LOSS_NORM / lgb[None]
        s["L_rewind"] = (er[:, 0] - er[:, -1]).sum(-1) / LOSS_NORM / lg
        s["L_rewind_boot"] = (C @ (er[:, 0] - er[:, -1]).T).T / LOSS_NORM / lgb[None]
        s["L_long"] = (s["L_graft"] + s["L_rewind"]) / 2
        lo, hi = min(r["LE"], r["LL"]), max(r["LE"], r["LL"])
        s["loss"] = {d: r[f"{d}_loss"] for d in ("graft", "rewind")}
        s["compatible"] = {d: np.all((r[f"{d}_loss"] >= lo - 1e-12) & (r[f"{d}_loss"] <= hi + 1e-12), 1)
                           for d in ("graft", "rewind")}
        dif = {d: np.diff(fv[d], axis=1) for d in ("graft", "rewind")}                       # (NG, 4, NR)
        s["monotone"] = {d: np.all(dif[d] >= 0, 1) | np.all(dif[d] <= 0, 1) for d in ("graft", "rewind")}
        dl = {d: np.diff(r[f"{d}_loss"], axis=1) for d in ("graft", "rewind")}               # (NG, 4)
        s["L_monotone"] = {d: np.all(dl[d] >= 0, 1) | np.all(dl[d] <= 0, 1) for d in ("graft", "rewind")}
        P_ = np.arange(r["kE"], r["kL"])
        if D["done"][P_].all() and tuple(sets) == ALL_G:
            w = decompose(D, [P_])[0]
            s["path"], s["path_boot"], s["path_tot"] = w["phi_f"], w["phi_f_boot"], w["tot_f"]
        out[key] = s
    return out


# ═══════════════════════════════════════════════════════════════════
# Classes of the fine groups (sums of their Shapley values: additive within the partition)
# ═══════════════════════════════════════════════════════════════════

CLASSES = {"W_ih": ("W_ih_i", "W_ih_f", "W_ih_g", "W_ih_o"), "W_hh f+i": ("W_hh_f", "W_hh_i"),
           "W_hh g+o": ("W_hh_g", "W_hh_o"), "read-out": ("readout",), "read-in": ("readin",),
           "biases": ("b_i", "b_f", "b_g", "b_o"), "start": ("start",)}
GATE_CLASSES = {f"gate {G}": (f"W_ih_{G}", f"W_hh_{G}", f"b_{G}") for G in ti.GATES}
CLASSES.update(GATE_CLASSES)                      # (overlapping views; each one is a sum over the partition)
CLASSES.update({"W_hh g": ("W_hh_g",), "W_hh o": ("W_hh_o",), "biases+start": ("b_i", "b_f", "b_g", "b_o", "start")})
MAIN_CLASSES = ("W_ih", "W_hh f+i", "W_hh g+o", "read-out", "read-in", "biases", "start")   # a partition
FAMILY_ROWS = {"answers": ("acc_goal_T", "acc_both_T", "far_T", "corner_truth_fwd"),
               "holding": ("held_two_fwd", "held_one_fwd", "held_joint_fwd", "emass_slope", "dist_out_fwd"),
               "coupling": ("excess", "G", "band4", "dir_slope", "angle_out_fwd", "same_diag_fwd")}


def class_values(w, k, cls):
    """a class's contribution (sum of its fine groups' phi) to row k in window dict w: (value, resamples), f units; cls a
    CLASSES key or a tuple of fine group names"""
    ix = [ALL_G.index(g) for g in (CLASSES[cls] if isinstance(cls, str) else cls)]
    if k == "loss":
        return w["L_phi_f"][ix].sum(), w["L_phi_f_boot"][ix].sum(0)
    i = ROWS.index(k)
    return w["phi_f"][ix, i].sum(), w["phi_f_boot"][ix, i].sum(0)


def family_values(w, fam, what="tot", cls=None):
    """the mean over a family's rows (FAMILY_ROWS) of the window change (what='tot') or of a class's contribution
    (what='cls'; cls a CLASSES key or 'inter'), in f units (each row on its own naive-Bayes scale; an equal-weight mean
    of additive quantities), with resamples; rows without a value in the window (band4 before 50 jumps >= 4 cells
    exist: the first window) are left out"""
    v, b = [], []
    for k in FAMILY_ROWS[fam]:
        i = ROWS.index(k)
        if not np.isfinite(w["tot_f"][i]):
            continue
        if what == "tot":
            v.append(w["tot_f"][i])
            b.append(w["tot_f_boot"][i])
        elif cls == "inter":
            v.append(w["inter_f"][i])
            b.append(w["inter_f_boot"][i])
        else:
            a, ab = class_values(w, k, cls)
            v.append(a)
            b.append(ab)
    with np.errstate(invalid="ignore"):
        return float(np.mean(v)), np.nanmean(np.stack(b), 0)


# ═══════════════════════════════════════════════════════════════════
# Size-matched random controls: their values per window; class minus matched random
# ═══════════════════════════════════════════════════════════════════

# class (label, fine groups) -> its size-matched random families (first = the primary match): same tensor and size
# for a tensor class, same gate rows for the gate-matched one, same size from W for a class that spans W_ih and W_hh
MATCH = (("W_hh g", ("W_hh_g",), ("rHH1M", "rG1M", "rW1M")), ("W_hh f", ("W_hh_f",), ("rHH1M",)),
         ("W_hh i", ("W_hh_i",), ("rHH1M",)), ("W_hh o", ("W_hh_o",), ("rHH1M",)),
         ("W_ih g", ("W_ih_g",), ("rIH1M", "rG1M")), ("W_ih i", ("W_ih_i",), ("rIH1M",)),
         ("W_ih f", ("W_ih_f",), ("rIH1M",)), ("W_ih o", ("W_ih_o",), ("rIH1M",)),
         ("W_hh f+i", CLASSES["W_hh f+i"], ("rHH2M", "rW2M")), ("W_hh g+o", CLASSES["W_hh g+o"], ("rHH2M", "rW2M")),
         ("gate g", CLASSES["gate g"], ("rW2M",)), ("gate i", CLASSES["gate i"], ("rW2M",)),
         ("gate f", CLASSES["gate f"], ("rW2M",)), ("gate o", CLASSES["gate o"], ("rW2M",)),
         ("W_ih", CLASSES["W_ih"], ("rW4M",)))
MATCH_D = {m[0]: m for m in MATCH}
FAMILIES = ("answers", "holding", "coupling")


def _loss_gap():
    """the training loss's naive -> Bayes gap per item and value: full sample and per episode resample (NB,)"""
    rf = references()
    return rf["loss_bayes"] - rf["loss_naive"], rf["loss_bayes"] - loss_counts() @ rf["loss_naive_ep"] / LOSS_NORM


def ctrl_window(name, wi, D, W):
    """a random family set's two-endpoint value pooled over window wi, in decompose's units: phi_f / dA_f / dB_f (NR,)
    with _boot (NR, NB), L_phi_f with L_phi_f_boot (NB,), sq; None if any of the window's pairs is missing"""
    key = ("cw", name, wi)
    if key in _C:
        return _C[key]
    w = W[wi]
    if name in CTRL_ALIAS:
        gi = ALL_G.index(CTRL_ALIAS[name])
        r = {k: w[k][gi] for k in ("phi_f", "phi_f_boot", "dA_f", "dA_f_boot", "dB_f", "dB_f_boot", "L_phi_f",
                                   "L_phi_f_boot", "sq")}
    else:
        eps = D["eps"]
        paths = [unit_path(("ctrl", (int(eps[k]), int(eps[k + 1]), name))) for k in w["pairs"]]
        if not all(os.path.exists(p) for p in paths):
            _C[key] = None
            return None
        iv, ib, lI = D["I_val"], D["I_boot"], D["I_loss"]
        dA, dB, dAb, dBb = np.zeros(NR), np.zeros(NR), np.zeros((NR, NB)), np.zeros((NR, NB))
        eA, eB, sq = np.zeros(2000), np.zeros(2000), 0.
        for k, p in zip(w["pairs"], paths):
            z = np.load(p)
            assert str(z["group"]) == name and list(z["rows"]) == list(ROWS)
            dA += z["A_val"] - iv[k]
            dB += iv[k + 1] - z["B_val"]
            dAb += z["A_boot"].astype(np.float64) - ib[k]
            dBb += ib[k + 1].astype(np.float64) - z["B_boot"]
            eA += z["A_loss"] - lI[k]
            eB += lI[k + 1] - z["B_loss"]
            sq += float(z["sq"])
        rf = references()
        g, gb = rf["gap"], rf["gap_boot"]
        lg, lgb = _loss_gap()
        e_phi = (eA + eB) / 2
        r = dict(phi_f=(dA + dB) / 2 / g, phi_f_boot=(dAb + dBb) / 2 / gb, dA_f=dA / g, dA_f_boot=dAb / gb,
                 dB_f=dB / g, dB_f_boot=dBb / gb, L_phi_f=e_phi.sum() / LOSS_NORM / lg,
                 L_phi_f_boot=loss_counts() @ e_phi / LOSS_NORM / lgb, sq=sq)
    _C[key] = r
    return r


def ctrl_done(fam, wi, D, W):
    """True when every seed of a random family has the whole window"""
    return all(ctrl_window(s, wi, D, W) is not None for s in fam_sets(fam))


def item_value(w, item, src, r=None, part="phi"):
    """(value, resamples) of an item (a ROWS name, 'loss', or a family of FAMILY_ROWS: the equal-weight mean of its
    rows with a value in the window) for src = 'tot' (the window change), 'inter', a tuple of fine groups (their summed
    values), or 'ctrl' (the random set r = ctrl_window(...)); part = 'phi', 'dA' or 'dB'"""
    if item in FAMILY_ROWS:
        v, b = [], []
        for k in FAMILY_ROWS[item]:
            if np.isfinite(w["tot_f"][ROWS.index(k)]):
                a, ab = item_value(w, k, src, r, part)
                v.append(a)
                b.append(ab)
        with np.errstate(invalid="ignore"):
            return float(np.mean(v)), np.nanmean(np.stack(b), 0)
    if item == "loss":
        if src == "tot":
            return w["L_tot_f"], w["L_tot_f_boot"]
        if src == "inter":
            return w["L_inter_f"], w["L_inter_f_boot"]
        assert part == "phi", "one-sided loss terms are not kept"
        if src == "ctrl":
            return r["L_phi_f"], r["L_phi_f_boot"]
        ix = [ALL_G.index(g) for g in src]
        return w["L_phi_f"][ix].sum(), w["L_phi_f_boot"][ix].sum(0)
    i = ROWS.index(item)
    if src in ("tot", "inter"):
        return w[f"{src}_f"][i], w[f"{src}_f_boot"][i]
    if src == "ctrl":
        return r[f"{part}_f"][i], r[f"{part}_f_boot"][i]
    ix = [ALL_G.index(g) for g in src]
    return w[f"{part}_f"][ix, i].sum(), w[f"{part}_f_boot"][ix, i].sum(0)


def matched(W, wi, item, groups_, fam, D):
    """a class (tuple of fine groups) against a size-matched random family in window wi for an item: dict(v, b (class),
    vs (3,) / bs (3, NB) the seeds, vm / bm their mean, d = v - vm with d_b, seed range of v - vs, tot / tot_b the
    window change); None if the family is incomplete there"""
    if not ctrl_done(fam, wi, D, W):
        return None
    w = W[wi]
    v, b = item_value(w, item, groups_)
    rs = [ctrl_window(s, wi, D, W) for s in fam_sets(fam)]
    vb = [item_value(w, item, "ctrl", r) for r in rs]
    vs, bs = np.array([x[0] for x in vb]), np.stack([x[1] for x in vb])
    tot, tb = item_value(w, item, "tot")
    return dict(v=v, b=b, vs=vs, bs=bs, vm=vs.mean(), bm=bs.mean(0), d=v - vs.mean(), d_b=b - bs.mean(0),
                rng=(float((v - vs).min()), float((v - vs).max())), tot=tot, tot_b=tb)


def _shown(w, item):
    """shares of an item's change are read only where its 5-95% excludes 0, |change| >= MIN_CHANGE and |interaction| <=
    |change|"""
    tot, tb = item_value(w, item, "tot")
    lo, hi = ci(tb)
    return bool((lo > 0 or hi < 0) and abs(tot) >= MIN_CHANGE and abs(item_value(w, item, "inter")[0]) <= abs(tot))


def _sh(x, tot, xb, tb):
    """a share x / tot with its paired resamples (percentiles of the ratio)"""
    with np.errstate(invalid="ignore", divide="ignore"):
        return x / tot, xb / tb


def matched_txt(m, share=False):
    """class [5-95%] vs the random family's seed mean (seed range) -> difference [5-95%] (seed range); shares of the
    window change when share=True"""
    if m is None:
        return "n/c"
    if share:
        s_c, s_cb = _sh(m["v"], m["tot"], m["b"], m["tot_b"])
        s_m, s_mb = _sh(m["vm"], m["tot"], m["bm"], m["tot_b"])
        s_d, s_db = _sh(m["d"], m["tot"], m["d_b"], m["tot_b"])
        rs = m["vs"] / m["tot"]
        return (f"{_ci_txt(s_c, s_cb, '+.2f')} vs {s_m:+.2f} ({rs.min():+.2f}..{rs.max():+.2f}) -> "
                f"{_ci_txt(s_d, s_db, '+.2f')} ({m['rng'][0] / m['tot']:+.2f}..{m['rng'][1] / m['tot']:+.2f})")
    return (f"{_ci_txt(m['v'], m['b'])} vs {m['vm']:+.3f} ({m['vs'].min():+.3f}..{m['vs'].max():+.3f}) -> "
            f"{_ci_txt(m['d'], m['d_b'])} ({m['rng'][0]:+.3f}..{m['rng'][1]:+.3f})")


# ═══════════════════════════════════════════════════════════════════
# Numbers (every number of the docstring)
# ═══════════════════════════════════════════════════════════════════

def _ci_txt(v, b, fmt="+.3f"):
    lo, hi = ci(b)
    return f"{v:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def numbers():
    """prints every number of the docstring (needs all units; reads OUT/checks.json if present)"""
    D = collect()
    W = decompose(D)
    rf = references()
    sec = []
    for k in range(len(D["eps"]) - 1):
        z = np.load(unit_path(("pair", (D["eps"][k], D["eps"][k + 1]))))
        sec += list(z["A_sec"]) + list(z["B_sec"])
    print(f"DATA: {len(D['eps'])} checkpoints, {int(D['done'].sum())} consecutive pairs x {NG} sets x 2 swaps = "
          f"{2 * NG * int(D['done'].sum()):,} swapped networks + {len(D['eps'])} intact; median {np.median(sec):.1f} s per "
          f"network (3 workers sharing GPU 0)")
    print("windows (training episodes): " + "; ".join(f"{win_label(i)}: {w['n_pairs']} pairs" for i, w in enumerate(W)))
    csec, cn = [], 0
    for f_ in CTRL_FAM:
        for n_ in fam_sets(f_):
            dd = os.path.join(OUT, "ctrl", n_)
            for fn in (sorted(os.listdir(dd)) if os.path.isdir(dd) else []):
                z = np.load(os.path.join(dd, fn))
                csec += [float(z["A_sec"]), float(z["B_sec"])]
                cn += 1
    rsec = []
    for fn in sorted(os.listdir(os.path.join(OUT, "ro"))) if os.path.isdir(os.path.join(OUT, "ro")) else []:
        z = np.load(os.path.join(OUT, "ro", fn))
        rsec += list(z["A_sec"]) + list(z["B_sec"])
    print(f"CONTROL DATA (fix round, 2 workers sharing GPU 0 with other jobs): {cn:,} random-family pair swaps x 2 = "
          f"{2 * cn:,} networks (median {np.median(csec) if csec else float('nan'):.1f} s per network); read-out split "
          f"{len(rsec)} networks; alpha paths: 2 more seeds of rW1M = {len(ALPHA_PAIRS) * len(ALPHA_CTRL) * 8} networks")
    print("references (naive -> Bayes): " + "; ".join(f"{k} {rf['naive'][i]:+.4g} -> {rf['bayes'][i]:+.4g}"
                                                       for i, k in enumerate(ROWS))
          + f"; loss {rf['loss_naive']:.5f} -> {rf['loss_bayes']:.0e}")
    p = os.path.join(OUT, "checks.json")
    if os.path.exists(p):
        with open(p) as f:
            c = json.load(f)
        print(f"CHECKS: {c['partition_total']:,} of {c['all_weights']:,} weights in the 15 groups; swaps exact "
              f"{all(c['swap_exact'].values())}; all groups swapped in turn = W_k+1 {c['all_fine_swapped_is_wb']}; "
              f"intact accuracy vs ck: {c['intact_acc_vs_ck']}; loss vs stored: {c['intact_loss_vs_stored']}; "
              f"reproduce: {c['reproduce']}; bookkeeping (true by construction): window identity "
              f"{c.get('window_identity')}, telescoping {c.get('telescoping')}")
        if "families" in c:
            print("CHECKS (random families): seed 1 = the path controls " + str(c["family_seed1_is_control"])
                  + f"; swaps exact {c['family_swap_exact']}; " + "; ".join(
                      f"{f_} {v['size'][0]:,}/{v['size'][1]:,}/{v['size'][2]:,} of {v['pool_size']:,} "
                      f"({'+'.join(k.split('.')[-1] for k in v['keys'])}; inside the pool rows {v['inside_pool_rows']}; "
                      f"seeds 1-2 share {v['overlap_s1_s2']:,})" for f_, v in c["families"].items())
                  + f"; read-out split ro_all = the stored read-out swaps: max |d| {c.get('ro_all_vs_stored_readout')}"
                  + f"; read-out fixed-norm direction swap: {c.get('ro_dir_construction')}"
                  + f"; gate-gain forward vs the engine's: max |d log q| {c.get('gains_forward_vs_engine')}")
    print("\nWINDOW CHANGE Delta f [5-95%] (gap fraction; naive -> Bayes = +1):")
    print(f"  {'row':18s}" + "".join(f"{win_label(i):>26s}" for i in range(len(W))))
    for k in ROWS:
        i = ROWS.index(k)
        print(f"  {k:18s}" + "".join(f"{_ci_txt(w['tot_f'][i], w['tot_f_boot'][i]):>26s}" for w in W))
    print(f"  {'loss':18s}" + "".join(f"{_ci_txt(w['L_tot_f'], w['L_tot_f_boot']):>26s}" for w in W))
    print("\nINTERACTION (change - sum of the 15 groups' phi), f units [5-95%], and as a share of the change:")
    for k in NUM_ROWS:
        tb = _window_table(W, k)
        print(f"  {k:18s}" + "".join(f"{t['inter']:+.3f} [{t['inter_lo']:+.3f},{t['inter_hi']:+.3f}] "
                                     f"({t['share'][NF]:+.2f})  " for t in tb))
    print("\nCLASS CONTRIBUTIONS per window, f units [5-95%] (share of the window's change); classes = sums of fine "
          "groups:")
    for k in NUM_ROWS:
        print(f"  {k} (change: " + ", ".join(f"{t['tot']:+.3f}" for t in _window_table(W, k)) + ")")
        for cls in CLASSES:
            vals = []
            for w, t in zip(W, _window_table(W, k)):
                v, b = class_values(w, k, cls)
                vals.append(f"{_ci_txt(v, b)} ({v / t['tot']:+.2f})")
            print(f"    {cls:9s} " + "  ".join(vals))
    print("\nFINE GROUPS per window (f units [5-95%]) for the figure's rows; controls last:")
    for k in NUM_ROWS:
        print(f"  {k}")
        tab = _window_table(W, k)
        for gi, g in enumerate(ALL_G):
            print(f"    {g:8s} " + "  ".join(f"{t['phi'][gi]:+.3f} [{t['phi_lo'][gi]:+.3f},{t['phi_hi'][gi]:+.3f}]"
                                            for t in tab))
    print("\nCONTROLS vs the groups they match (f units): random 1M of W vs the mean of the 8 W blocks; random 2k of "
          "biases vs the mean of the 4 bias groups")
    blocks = [ALL_G.index(g) for g in FINE if g.startswith("W_")]
    bias = [ALL_G.index(g) for g in FINE if g.startswith("b_")]
    for k in NUM_ROWS:
        tab = _window_table(W, k)
        print(f"  {k:16s} " + "  ".join(f"{t['phi'][ALL_G.index('rand_W')]:+.3f} vs {t['phi'][blocks].mean():+.3f} | "
                                         f"{t['phi'][ALL_G.index('rand_b')]:+.4f} vs {t['phi'][bias].mean():+.4f}"
                                         for t in tab))
    print("\nONE-SIDED TERMS (pooled; f units): sum over groups of dA (adding the group's change to W_k) / of dB "
          "(removing it from W_k+1) / the change")
    for k in NUM_ROWS:
        if k == "loss":
            continue
        i = ROWS.index(k)
        print(f"  {k:16s} " + "  ".join(f"{w['dA_f'][:NF, i].sum():+.3f} / {w['dB_f'][:NF, i].sum():+.3f} / "
                                         f"{w['tot_f'][i]:+.3f}" for w in W))
    print("\nFAMILY MEANS (equal-weight mean over the family's rows of f-unit quantities) per window: change and class "
          "contributions [5-95%]")
    for fam in FAMILY_ROWS:
        print(f"  {fam} ({', '.join(FAMILY_ROWS[fam])})")
        print("    change    " + "  ".join(_ci_txt(*family_values(w, fam)) for w in W))
        for cls in CLASSES:
            print(f"    {cls:9s} " + "  ".join(_ci_txt(*family_values(w, fam, 'cls', cls)) for w in W))
    dissociation(W)
    headline(W, D)
    controls_numbers(W, D)
    size_scaling(W, D)
    prop_numbers(W, D)
    wih_baselines(W, D)
    orderings(W)
    readout_rows(W)
    small_sets(W)
    gains_numbers()
    ro_numbers(W, D)
    figure_numbers(W, D)
    alpha_numbers(D)


CTRL_ITEMS = ("acc_goal_T", "far_T", "logq_goal_T", "kl_goal_mean", "held_two_fwd", "held_one_fwd",
              "held_joint_fwd", "emass_slope", "dist_out_fwd", "excess", "G", "dir_slope", "loss", "answers", "holding",
              "coupling")


def controls_numbers(W, D):
    """the random families (3 seeds) and every class against its size-matched random families"""
    nW = len(W)
    print("\nRANDOM FAMILIES (k random entries of a pool, 3 seeds; two-endpoint values pooled over the window like the "
          "groups'): per window the 3 seeds' values s1 / s2 / s3 -> their mean [5-95% over the shared resamples of the "
          "mean]; n/c = not computed (" + CTRL_NOT_RUN + ")")
    for fam, (pool, k, seeds) in CTRL_FAM.items():
        print(f"  {fam} = {FAM_LABEL[fam]} ({k:,} of the pool '{pool}'; seeds {', '.join(map(str, seeds))})")
        print(f"    {'':16s}" + "".join(f"{win_label(i):>44s}" for i in range(nW)))
        for item in CTRL_ITEMS:
            cells = []
            for wi in range(nW):
                if not ctrl_done(fam, wi, D, W):
                    cells.append("n/c")
                    continue
                vb = [item_value(W[wi], item, "ctrl", ctrl_window(s, wi, D, W)) for s in fam_sets(fam)]
                vs = [x[0] for x in vb]
                cells.append(" / ".join(f"{v:+.3f}" for v in vs) + " -> "
                             + _ci_txt(np.mean(vs), np.mean([x[1] for x in vb], 0)))
            print(f"    {item:16s}" + "".join(f"{c:>44s}" for c in cells))
    print("\nCLASS MINUS SIZE-MATCHED RANDOM (f units): class [5-95%] vs the family's seed mean (seed range) -> "
          "difference [5-95% over the paired resamples] (range of class - each seed); per window; n/c = not computed")
    for lab, grp, fams in MATCH:
        for fam in fams:
            print(f"  {lab} vs {FAM_LABEL[fam]}")
            for item in CTRL_ITEMS:
                cells = [matched_txt(matched(W, wi, item, grp, fam, D)) for wi in range(nW)]
                if all(c == "n/c" for c in cells):
                    continue
                print(f"    {item:16s} " + " | ".join(f"{win_label(wi)}: {c}" for wi, c in enumerate(cells)
                                                     if c != "n/c"))
    print("\nFAMILY SHARES, CLASS vs SIZE-MATCHED RANDOM: share of the family's window change [5-95%] vs the random "
          "family's seed-mean share (seed range) -> difference [5-95%] (seed range); windows where the family's change "
          "fails the share rule (5-95% excludes 0, >= .05, |interaction| <= change) are left out")
    for lab, grp, fams in MATCH:
        for fam in fams:
            print(f"  {lab} vs {FAM_LABEL[fam]}")
            for item in FAMILIES + ("loss",):
                cells = [matched_txt(matched(W, wi, item, grp, fam, D), share=True) if _shown(W[wi], item) else "n/c"
                         for wi in range(nW)]
                print(f"    {item:9s} " + " | ".join(f"{win_label(wi)}: {c}" for wi, c in enumerate(cells)
                                                   if c != "n/c"))
    print("\nTASK HYPOTHESIS AGAINST SIZE-MATCHED RANDOM SETS (family shares; class - random, paired):")
    for lab, fam, wi, item in (("W_hh f+i", "rHH2M", 1, "answers"), ("W_hh f+i", "rHH2M", 3, "holding"),
                               ("W_hh f+i", "rW2M", 1, "answers"), ("W_hh f+i", "rW2M", 3, "holding"),
                               ("W_ih", "rW4M", 1, "answers"), ("W_ih", "rW4M", 3, "holding"),
                               ("W_ih", "rW4M", 1, "holding"), ("W_ih", "rW4M", 3, "answers")):
        m = matched(W, wi, item, MATCH_D[lab][1], fam, D)
        print(f"  {lab} vs {FAM_LABEL[fam]}, {item} {win_label(wi)}: {matched_txt(m, share=True)}")
    print("\nDISSOCIATION AGAINST RANDOM (holding share - answers share of the same window): class [5-95%] vs the random "
          "family's seed mean (seed range) -> class - random [5-95%]")
    for lab, fam in (("W_hh f+i", "rHH2M"), ("W_hh g+o", "rHH2M"), ("W_hh g", "rHH1M"), ("gate g", "rW2M"),
                     ("W_ih", "rW4M")):
        cells = []
        for wi in range(1, nW):
            mh, ma = (matched(W, wi, it_, MATCH_D[lab][1], fam, D) for it_ in ("holding", "answers"))
            if mh is None or ma is None or not (_shown(W[wi], "holding") and _shown(W[wi], "answers")):
                continue
            c_v = mh["v"] / mh["tot"] - ma["v"] / ma["tot"]
            c_b = mh["b"] / mh["tot_b"] - ma["b"] / ma["tot_b"]
            r_s = mh["vs"] / mh["tot"] - ma["vs"] / ma["tot"]
            d_v = mh["d"] / mh["tot"] - ma["d"] / ma["tot"]
            d_b = mh["d_b"] / mh["tot_b"] - ma["d_b"] / ma["tot_b"]
            cells.append(f"{win_label(wi)}: {_ci_txt(c_v, c_b, '+.2f')} vs {r_s.mean():+.2f} ({r_s.min():+.2f}.."
                         f"{r_s.max():+.2f}) -> {_ci_txt(d_v, d_b, '+.2f')}")
        print(f"  {lab} vs {FAM_LABEL[fam]}: " + " | ".join(cells))
    for lab, fam, item in (("W_hh f+i", "rHH2M", "holding"), ("W_ih", "rW4M", "holding")):
        m3, m1 = matched(W, 3, item, MATCH_D[lab][1], fam, D), matched(W, 1, "answers", MATCH_D[lab][1], fam, D)
        if m3 is None or m1 is None:
            continue
        s3, s3b = _sh(m3["d"], m3["tot"], m3["d_b"], m3["tot_b"])
        s1, s1b = _sh(m1["d"], m1["tot"], m1["d_b"], m1["tot_b"])
        print(f"  {lab}: (late holding excess share) - (early answers excess share), excess = class - {fam} seed mean: "
              f"{_ci_txt(s3 - s1, s3b - s1b, '+.2f')}")


RO_ROWS = ("emass_slope", "dist_out_fwd", "held_two_fwd", "held_one_fwd", "held_joint_fwd", "acc_goal_T",
           "logq_goal_T", "kl_goal_mean", "loss")


def ro_summary(W, D):
    """the read-out split over RO_WINDOW's pairs: per variant (RO_VARIANTS, + RO_DIR when its units are all stored;
    'variants') phi / dA / dB in f units (NR,) with resamples, the training loss's phi (f units), and the goal entropy
    at T (nats; phi, dA, dB with resamples); the window's entropy change; ro_all vs the stored 'readout' swaps (max
    |d|); None if a pair of the ro units is missing"""
    w = W[RO_WINDOW]
    eps = D["eps"]
    prs = [(int(eps[k]), int(eps[k + 1])) for k in w["pairs"]]
    paths = [unit_path(("ro", p_)) for p_ in prs]
    if not all(os.path.exists(p) for p in paths):
        return None
    dpaths = [unit_path(("ro_dir", p_)) for p_ in prs]
    variants = RO_ALL if all(os.path.exists(p) for p in dpaths) else RO_VARIANTS
    rf = references()
    g, gb = rf["gap"], rf["gap_boot"]
    lg, lgb = _loss_gap()
    C, CL = gap.boot_counts(), loss_counts()
    nV = len(variants)
    iv, ib, lI = D["I_val"], D["I_boot"], D["I_loss"]
    acc = {k: np.zeros((nV, NR)) for k in ("dA", "dB")}
    accb = {k: np.zeros((nV, NR, NB)) for k in ("dA", "dB")}
    eL = np.zeros((nV, 2000))
    ent = {k: np.zeros((nV, gap.N_ENV)) for k in ("dA", "dB")}
    ent0 = ent1 = None
    dmax = 0.
    gi = ALL_G.index("readout")
    for j, (k, p) in enumerate(zip(w["pairs"], paths)):
        z = np.load(p)
        assert list(z["variants"]) == list(RO_VARIANTS) and list(z["rows"]) == list(ROWS)
        zs = [z]
        if variants == RO_ALL:
            zd = np.load(dpaths[j])
            assert list(zd["variants"]) == list(RO_DIR) and list(zd["rows"]) == list(ROWS)
            zs.append(zd)

        def cat(key):
            return np.concatenate([z_[key] for z_ in zs])
        acc["dA"] += cat("A_val") - iv[k]
        acc["dB"] += iv[k + 1] - cat("B_val")
        accb["dA"] += cat("A_boot").astype(np.float64) - ib[k]
        accb["dB"] += ib[k + 1].astype(np.float64) - cat("B_boot")
        eL += ((cat("A_loss") - lI[k]) + (lI[k + 1] - cat("B_loss"))) / 2
        ent["dA"] += cat("A_ent_env") - z["ent_I"][0]
        ent["dB"] += z["ent_I"][1] - cat("B_ent_env")
        if j == 0:
            ent0 = z["ent_I"][0]
        ent1 = z["ent_I"][1]
        for s in "AB":
            dmax = max(dmax, float(np.nanmax(np.abs(z[f"{s}_val"][0] - D[f"{s}_val"][k, gi]))),
                       float(np.abs(z[f"{s}_loss"][0] - D[f"{s}_loss"][k, gi]).max()))
    nP = gap.N_ENV * gap.PER_ENV
    out = dict(variants=variants, phi_f=(acc["dA"] + acc["dB"]) / 2 / g,
               phi_f_boot=(accb["dA"] + accb["dB"]) / 2 / gb, dA_f=acc["dA"] / g, dB_f=acc["dB"] / g,
               L_phi_f=eL.sum(1) / LOSS_NORM / lg, L_phi_f_boot=(CL @ eL.T).T / LOSS_NORM / lgb,
               ro_all_vs_stored=dmax, ent_tot=(ent1 - ent0).sum() / nP, ent_tot_boot=C @ (ent1 - ent0) / nP,
               ent_level=(ent0.sum() / nP, ent1.sum() / nP))
    for k in ("dA", "dB"):
        out[f"ent_{k}"] = ent[k].sum(1) / nP
        out[f"ent_{k}_boot"] = (C @ ent[k].T).T / nP
    out["ent_phi"] = (out["ent_dA"] + out["ent_dB"]) / 2
    out["ent_phi_boot"] = (out["ent_dA_boot"] + out["ent_dB_boot"]) / 2
    return out


def ro_numbers(W, D):
    S_ = ro_summary(W, D)
    if S_ is None:
        print("\nREAD-OUT SPLIT: not complete")
        return
    w = W[RO_WINDOW]
    V = S_["variants"]
    print(f"\nREAD-OUT SPLIT, {win_label(RO_WINDOW)} ({len(w['pairs'])} pairs): ro_all = W_out and b_out (= the group "
          f"'readout'; vs the stored swaps max |d| {S_['ro_all_vs_stored']:.1e}), ro_W = W_out only, ro_b = b_out only, "
          "ro_scale = W_out <- s W_out (the other checkpoint's least-squares scale: belief scale only), ro_dir = the "
          "other checkpoint's W_out at this one's norm (a fixed-norm direction swap, not matched on calibration: see its "
          "goal entropy; final round). phi [5-95%] (dA / dB), f units; window change in brackets; last: ro_dir + "
          "ro_scale vs ro_W (how far direction and scale add up)")
    for k in RO_ROWS:
        if k == "loss":
            tot, tb = w["L_tot_f"], w["L_tot_f_boot"]
            vals = S_["L_phi_f"]
            cells = [_ci_txt(S_["L_phi_f"][j], S_["L_phi_f_boot"][j]) for j in range(len(V))]
        else:
            i = ROWS.index(k)
            tot, tb = w["tot_f"][i], w["tot_f_boot"][i]
            vals = S_["phi_f"][:, i]
            cells = [f"{_ci_txt(S_['phi_f'][j, i], S_['phi_f_boot'][j, i])} ({S_['dA_f'][j, i]:+.3f} / "
                     f"{S_['dB_f'][j, i]:+.3f})" for j in range(len(V))]
        add = (f"  | dir + scale {vals[V.index('ro_dir')] + vals[V.index('ro_scale')]:+.3f} vs W {vals[V.index('ro_W')]:+.3f}"
               if "ro_dir" in V else "")
        print(f"  {k:15s} [{_ci_txt(tot, tb)}]  " + "  ".join(f"{v}: {c}" for v, c in zip(V, cells)) + add)
    print(f"  goal entropy at T, nats [window change {_ci_txt(S_['ent_tot'], S_['ent_tot_boot'])}; level "
          f"{S_['ent_level'][0]:.4f} -> {S_['ent_level'][1]:.4f}]  " + "  ".join(
              f"{v}: {_ci_txt(S_['ent_phi'][j], S_['ent_phi_boot'][j])} ({S_['ent_dA'][j]:+.3f} / "
              f"{S_['ent_dB'][j]:+.3f})" for j, v in enumerate(V)))


def gains_numbers():
    G_ = gains_cache()
    print("\nGATE GAINS (W_k on the 10,000 probes; per pair, then the mean over the window's pairs [min-max over pairs]): "
          "mean gate slope (sigma' for i, f, o; tanh' for g); RMS of each block's drive change (pre) and of its effect "
          "on the gate's output (slope x drive change), gates i / f / g / o; direct forward vs the engine's: max |d log q| "
          f"{G_['logq_maxdiff'].max():.1e}")
    for wi in (1, 3):
        s = G_["win"] == wi
        print(f"  {win_label(wi)} ({int(s.sum())} pairs):")
        for k in ("slope", "pre_ih", "pre_hh", "out_ih", "out_hh"):
            v = G_[k][s]
            print(f"    {k:7s} " + "  ".join(f"{G}: {v[:, j].mean():.4f} [{v[:, j].min():.4f}-{v[:, j].max():.4f}]"
                                            for j, G in enumerate(ti.GATES)))
        oh = G_["out_hh"][s]
        rank = (oh[:, 2][:, None] >= np.concatenate([oh, G_["out_ih"][s]], 1)).all(1)
        print(f"    W_hh g's output change is the largest of the 8 blocks in {int(rank.sum())} of {int(s.sum())} pairs")


def paired(w, item, a, b):
    """(value, [5-95%]) of contribution(a) - contribution(b) in window dict w; a, b tuples of fine groups"""
    va, ba = item_value(w, item, a)
    vb_, bb = item_value(w, item, b)
    return va - vb_, ba - bb


def orderings(W):
    """paired differences behind every ordering in the text"""
    print("\nORDERINGS (paired differences, f units [5-95%])")
    w = W[1]
    for k in ("acc_goal_T", "far_T", "logq_goal_T", "kl_goal_mean", "held_two_fwd", "held_joint_fwd", "emass_slope",
              "dist_out_fwd", "loss"):
        ph = w["L_phi_f"][:NF] if k == "loss" else w["phi_f"][:NF, ROWS.index(k)]
        o = np.argsort(-ph)
        top, sec_ = FINE[o[0]], FINE[o[1]]
        v, b = paired(w, k, (top,), (sec_,))
        print(f"  {win_label(1)} {k:14s} largest {top} {ph[o[0]]:+.3f}, next {sec_} {ph[o[1]]:+.3f}: difference "
              f"{_ci_txt(v, b)}")
    for k in ("acc_goal_T", "logq_goal_T", "kl_goal_mean", "held_two_fwd"):
        print(f"  {win_label(1)} {k:14s} W_ih g - W_hh f {_ci_txt(*paired(w, k, ('W_ih_g',), ('W_hh_f',)))}; "
              f"W_ih g - W_hh o {_ci_txt(*paired(w, k, ('W_ih_g',), ('W_hh_o',)))}")
    w = W[3]
    for k in ("held_two_fwd", "held_joint_fwd", "emass_slope", "excess"):
        print(f"  {win_label(3)} {k:14s} " + "; ".join(
            f"W_ih g - W_ih {G} {_ci_txt(*paired(w, k, ('W_ih_g',), (f'W_ih_{G}',)))}" for G in "ifo"))
    k = "dir_slope"
    print(f"  {win_label(3)} {k:14s} W_hh g - W_hh f {_ci_txt(*paired(w, k, ('W_hh_g',), ('W_hh_f',)))}; "
          f"W_hh g - W_ih g {_ci_txt(*paired(w, k, ('W_hh_g',), ('W_ih_g',)))}")
    for a in ("W_ih", "read-in", "read-out"):
        print(f"  {win_label(3)} holding family: {a} - W_hh g+o "
              f"{_ci_txt(*paired(w, 'holding', CLASSES[a], CLASSES['W_hh g+o']))}; {a} - W_hh f+i "
              f"{_ci_txt(*paired(w, 'holding', CLASSES[a], CLASSES['W_hh f+i']))}"
              + (f"; {a} - W_ih {_ci_txt(*paired(w, 'holding', CLASSES[a], CLASSES['W_ih']))}" if a != "W_ih" else ""))
    w = W[4]
    for k in ("held_two_fwd", "emass_slope"):
        print(f"  {win_label(4)} {k:14s} W_ih o - W_ih g {_ci_txt(*paired(w, k, ('W_ih_o',), ('W_ih_g',)))}")


HEAD_SETS = (("W_hh g", ("W_hh_g",)), ("W_ih g", ("W_ih_g",)), ("W_hh f", ("W_hh_f",)),
             ("W_hh f+i", CLASSES["W_hh f+i"]), ("W_hh g+o", CLASSES["W_hh g+o"]), ("W_ih", CLASSES["W_ih"]),
             ("gate g", CLASSES["gate g"]), ("read-in", CLASSES["read-in"]), ("read-out", CLASSES["read-out"]))
HEAD_ROWS = {1: ("acc_goal_T", "logq_goal_T", "kl_goal_mean", "held_two_fwd", "held_joint_fwd", "emass_slope", "loss"),
             3: ("held_two_fwd", "held_joint_fwd", "emass_slope", "dist_out_fwd", "excess", "dir_slope", "acc_goal_T",
                 "loss")}
HEAD_FAMS = ("rHH1M", "rIH1M", "rG1M", "rHH2M", "rW2M", "rW4M", "rW1M")


POOL_GROUPS = {"W": tuple(g for g in FINE if g.startswith("W_")), "W_hh": CLASSES["W_hh f+i"] + CLASSES["W_hh g+o"],
               "W_ih": CLASSES["W_ih"], "g rows": ("W_ih_g", "W_hh_g")}
SCALE_ITEMS = ("acc_goal_T", "logq_goal_T", "kl_goal_mean", "held_two_fwd", "held_joint_fwd", "emass_slope", "excess",
               "dir_slope", "loss", "answers", "holding", "coupling")


SCALE_FAMS = ("rW1M", "rW2M", "rW4M", "rHH1M", "rHH2M", "rIH1M", "rG1M")
POOL_SIZE = {"W": 8_000_000, "W_hh": 4_000_000, "W_ih": 4_000_000, "g rows": 2_000_000}


def scaling_cells(W, D):
    """[(fam, item, window, seed-mean value, proportional share, deviation, deviation's resamples, per-draw
    deviations (3,))] for every random family (SCALE_FAMS), item (SCALE_ITEMS) and window where the family ran;
    proportional share = (k / pool) x the summed values of the pool's blocks"""
    if "scaling" not in _C:
        out = []
        for fam in SCALE_FAMS:
            pool, k_, _ = CTRL_FAM[fam]
            fr = k_ / POOL_SIZE[pool]
            for item in SCALE_ITEMS:
                for wi in range(len(W)):
                    if not ctrl_done(fam, wi, D, W):
                        continue
                    w = W[wi]
                    vb = [item_value(w, item, "ctrl", ctrl_window(s_, wi, D, W)) for s_ in fam_sets(fam)]
                    pv, pb = item_value(w, item, POOL_GROUPS[pool])
                    mv, mb = np.mean([x[0] for x in vb]), np.mean([x[1] for x in vb], 0)
                    out.append((fam, item, wi, mv, fr * pv, mv - fr * pv, mb - fr * pb,
                                np.array([x[0] for x in vb]) - fr * pv))
        _C["scaling"] = out
    return _C["scaling"]


def size_scaling(W, D):
    """each random family's seed mean against its pool's proportional share, (k / pool size) x the summed values of
    the pool's blocks (what a random subset carries in expectation if the value were additive over single weights)"""
    print("\nSIZE SCALING: random family seed mean - (k / pool) x sum of the pool's blocks' values, f units [5-95%] "
          "(0 = the random set carries its proportional share; the verifier's 'not linear in size' test)")
    cells_ = scaling_cells(W, D)
    for fam in SCALE_FAMS:
        pool, k_, _ = CTRL_FAM[fam]
        lines = []
        for item in SCALE_ITEMS:
            cells = [f"{win_label(wi)} {mv:+.3f} vs {pv:+.3f}: {_ci_txt(d, db)}"
                     for f_, it_, wi, mv, pv, d, db, _ in cells_ if f_ == fam and it_ == item]
            if cells:
                lines.append(f"    {item:15s} " + " | ".join(cells))
        if lines:
            print(f"  {fam} ({FAM_LABEL[fam]}; k / pool = {k_ / POOL_SIZE[pool]:.3f})")
            print("\n".join(lines))
    if cells_:
        d_ = np.array([c[5] for c in cells_])
        sig = [c for c in cells_ if ci(c[6])[0] > 0 or ci(c[6])[1] < 0]
        print(f"  over all {len(cells_)} cells: deviation {d_.min():+.3f} .. {d_.max():+.3f} f (max |.| "
              f"{np.abs(d_).max():.3f}, mean |.| {np.abs(d_).mean():.3f}); 5-95% excludes 0 in {len(sig)}: "
              + ", ".join(f"{c[0]} {c[1]} {win_label(c[2])} {c[5]:+.3f}" for c in sig))
        print("  per draw (each seed's deviation from the proportional share; a class's value is a sum of single-group "
              "swaps, a random set is swapped as one unit, so they agree only to second order): per family the cells, "
              "the per-draw range, |seed mean| (mean, max), cells with the same sign in all 3 draws (1 in 4 expected by "
              "chance), cells with |seed mean| / (draw sd / sqrt 3) > 4.30 (t, 2 df, two-sided 5%)")
        for fam in SCALE_FAMS:
            cf = [c for c in cells_ if c[0] == fam]
            if not cf:
                continue
            dv = np.array([c[7] for c in cf])
            same = ((dv > 0).all(1) | (dv < 0).all(1))
            with np.errstate(invalid="ignore", divide="ignore"):
                t_ = dv.mean(1) / (dv.std(1, ddof=1) / np.sqrt(3))
            print(f"    {fam:6s} {len(cf)} cells; per draw {dv.min():+.3f} .. {dv.max():+.3f}; |seed mean| mean "
                  f"{np.abs(dv.mean(1)).mean():.4f}, max {np.abs(dv.mean(1)).max():.3f}; same sign {int(same.sum())} "
                  f"of {len(cf)}; |t| > 4.30 in {int((np.abs(t_) > 4.303).sum())}")
        print("    random 4M of W, cells with the same sign in all 3 draws and |seed mean| >= .02: " + "; ".join(
            f"{c[1]} {win_label(c[2])} {c[7][0]:+.3f} / {c[7][1]:+.3f} / {c[7][2]:+.3f}" for c in cells_
            if c[0] == "rW4M" and ((c[7] > 0).all() or (c[7] < 0).all()) and abs(c[5]) >= .02))


PROP_MATCH = (("W_ih", CLASSES["W_ih"], "W", .5), ("W_hh f+i", CLASSES["W_hh f+i"], "W_hh", .5),
              ("W_hh g+o", CLASSES["W_hh g+o"], "W_hh", .5), ("W_hh g", ("W_hh_g",), "W_hh", .25),
              ("W_ih g", ("W_ih_g",), "W_ih", .25), ("gate g (W rows)", ("W_ih_g", "W_hh_g"), "W", .25))


def prop_numbers(W, D):
    """where the random families were not run: each class against its proportional share of the pool (k / pool x the
    pool's summed values; the expectation the random families matched where they ran, SIZE SCALING), paired"""
    print("\nPROPORTIONAL-SHARE BASELINE (windows without the matching random family): class - (k / pool) x the pool's "
          "summed values, f units [5-95%], and as a share of the window's change")
    fam_of = {"W_ih": "rW4M", "W_hh f+i": "rHH2M", "W_hh g+o": "rHH2M", "W_hh g": "rHH1M", "W_ih g": "rIH1M",
              "gate g (W rows)": "rW2M"}
    for lab, grp, pool, fr in PROP_MATCH:
        for item in ("held_two_fwd", "held_joint_fwd", "emass_slope", "excess", "dir_slope", "acc_goal_T", "loss",
                     "answers", "holding", "coupling"):
            cells = []
            for wi, w in enumerate(W):
                if ctrl_done(fam_of[lab], wi, D, W):
                    continue
                v, b = item_value(w, item, grp)
                pv, pb = item_value(w, item, POOL_GROUPS[pool])
                d, db = v - fr * pv, b - fr * pb
                txt = f"{win_label(wi)}: {v:+.3f} vs {fr * pv:+.3f} -> {_ci_txt(d, db)}"
                if _shown(w, item):
                    tot, tb = item_value(w, item, "tot")
                    txt += f" (share {_ci_txt(d / tot, db / tb, '+.2f')})"
                cells.append(txt)
            if cells:
                print(f"  {lab:16s} {item:15s} " + " | ".join(cells))


def wih_baselines(W, D):
    """W_ih against both of its baselines in every window: the random 4M of W (swapped as one unit) and its
    proportional share, 1/2 of W's summed values (built like the class: a sum of single-group swaps)"""
    print("\nW_IH AGAINST BOTH BASELINES: class - proportional share (1/2 of the 8 W blocks' summed values) and class - "
          "random 4M of W (3-draw mean; where run), f units [5-95%]; shares of the change where shown")
    for item in ("acc_goal_T", "held_two_fwd", "held_joint_fwd", "emass_slope", "excess", "dir_slope", "loss", "answers",
                 "holding", "coupling"):
        cells = []
        for wi, w in enumerate(W):
            v, b = item_value(w, item, CLASSES["W_ih"])
            pv, pb = item_value(w, item, POOL_GROUPS["W"])
            d, db = v - .5 * pv, b - .5 * pb
            m = matched(W, wi, item, CLASSES["W_ih"], "rW4M", D)
            txt = f"{win_label(wi)}: prop {_ci_txt(d, db)}" + (f", rW4M {_ci_txt(m['d'], m['d_b'])}" if m else "")
            if _shown(w, item):
                tot, tb = item_value(w, item, "tot")
                txt += (f" (shares: prop {_ci_txt(d / tot, db / tb, '+.2f')}"
                        + (f", rW4M {_ci_txt(m['d'] / tot, m['d_b'] / tb, '+.2f')}" if m else "") + ")")
            cells.append(txt)
        print(f"  {item:14s} " + " | ".join(cells))


def readout_rows(W):
    """the read-out's contribution to each answers row per window: its family mean hides rows of opposite sign"""
    print("\nREAD-OUT ON THE ANSWERS ROWS: the family mean, then each row's phi [5-95%] of the row's change [5-95%], f "
          "units")
    for wi, w in enumerate(W):
        fv, fb = item_value(w, "answers", CLASSES["read-out"])
        print(f"  {win_label(wi)}: family {_ci_txt(fv, fb)}; " + "; ".join(
            f"{k} {_ci_txt(*item_value(w, k, CLASSES['read-out']))} of {_ci_txt(*item_value(w, k, 'tot'))}"
            for k in FAMILY_ROWS["answers"]))


def small_sets(W):
    """the biases' and the start state's values (NUM_ROWS and the family means): share of cells with |phi| <= .02 f and
    the largest |phi|, in 10-800 and after 800; the g gate's own biases (b_g, unmatched in rW2M) in 800-11,200"""
    print("\nBIASES AND START STATE (NUM_ROWS + family means; f units [5-95%])")
    for wis, lab in (((0,), win_label(0)), ((1, 2, 3, 4), "after 800")):
        big = []
        for wi in wis:
            for item in NUM_ROWS + FAMILIES:
                for cls in ("biases", "start"):
                    v, b = item_value(W[wi], item, CLASSES[cls])
                    if np.isfinite(v):
                        big.append((abs(v), cls, item, wi, v, b))
        big.sort(key=lambda t: -t[0])
        print(f"  {lab}: {len(big)} cells, |phi| <= .02 in {np.mean([t[0] <= .02 for t in big]):.2f}; largest: "
              + "; ".join(f"{t[1]} {t[2]} {win_label(t[3])} {_ci_txt(t[4], t[5])}" for t in big[:5]))
    bg = [(abs(item_value(W[wi], it, ("b_g",))[0]), it, wi) for wi in (1, 2, 3) for it in FAMILIES + ("loss",)]
    m_ = max(bg)
    print(f"  b_g (the g gate's 2,000 biases), families and loss, 800-11,200: max |phi| {m_[0]:.3f} ({m_[1]}, "
          f"{win_label(m_[2])})")


def headline(W, D):
    """the headline sets' two-endpoint values phi [5-95%] with their one-sided terms (dA = adding the set's change to
    W_k, dB = removing it from W_k+1) in 800-1,600 and 4,000-11,200, the same for the random families (seed means), and
    the paired W_hh g - W_ih g differences; squared displacement of the W blocks per window"""
    for wi in (1, 3):
        w = W[wi]
        print(f"\nHEADLINE SETS, {win_label(wi)}: phi [5-95%] (dA / dB), f units; share of the change")
        for k in HEAD_ROWS[wi]:
            tot, tb = item_value(w, k, "tot")
            cells = []
            for lab, grp in HEAD_SETS:
                v, b = item_value(w, k, grp)
                if k == "loss":
                    ix = [ALL_G.index(g) for g in grp]
                    a_, b_ = w["L_dA_f"][ix].sum(), w["L_dB_f"][ix].sum()
                else:
                    a_, b_ = item_value(w, k, grp, part="dA")[0], item_value(w, k, grp, part="dB")[0]
                cells.append(f"{lab} {_ci_txt(v, b)} ({a_:+.3f} / {b_:+.3f}; {v / tot:+.2f})")
            print(f"  {k} (change {_ci_txt(tot, tb)}): " + "; ".join(cells))
            rc = []
            for fam in HEAD_FAMS:
                if not ctrl_done(fam, wi, D, W):
                    continue
                rs = [ctrl_window(s_, wi, D, W) for s_ in fam_sets(fam)]
                ph = np.mean([item_value(w, k, "ctrl", r)[0] for r in rs])
                if k == "loss":
                    rc.append(f"{fam} {ph:+.3f}")
                    continue
                a_ = np.mean([item_value(w, k, "ctrl", r, "dA")[0] for r in rs])
                b_ = np.mean([item_value(w, k, "ctrl", r, "dB")[0] for r in rs])
                rc.append(f"{fam} {ph:+.3f} ({a_:+.3f} / {b_:+.3f})")
            print("      random families (seed means): " + "; ".join(rc))
    a, b = ALL_G.index("W_hh_g"), ALL_G.index("W_ih_g")
    for wi in (1, 3):
        ww = W[wi]
        print(f"W_hh g - W_ih g, {win_label(wi)} (paired, f units): " + "; ".join(
            f"{k} {ww['phi_f'][a, ROWS.index(k)] - ww['phi_f'][b, ROWS.index(k)]:+.3f} "
            f"[{ci(ww['phi_f_boot'][a, ROWS.index(k)] - ww['phi_f_boot'][b, ROWS.index(k)])[0]:+.3f}, "
            f"{ci(ww['phi_f_boot'][a, ROWS.index(k)] - ww['phi_f_boot'][b, ROWS.index(k)])[1]:+.3f}]"
            for k in NUM_ROWS if k != "loss"))
    print("squared displacement summed over each window's pairs (W blocks): "
          + "; ".join(f"{win_label(i)}: " + ", ".join(f"{g} {ww['sq'][ALL_G.index(g)]:.0f}" for g in FINE
                                                      if g.startswith("W_")) for i, ww in enumerate(W)))


def export_summary(D=None):
    """-> OUT/summary.npz: per window (5) the change and every set's contribution in gap-fraction units with 5-95%, the
    interaction, the one-sided terms, the training loss's decomposition, and the alpha-path effects (for other modules)"""
    D = collect() if D is None else D
    W = decompose(D)
    out = dict(rows=np.array(ROWS), sets=np.array(ALL_G), n_fine=NF, win_x=np.array(WIN_X),
               n_pairs=np.array([w["n_pairs"] for w in W]))
    for k in ("tot_f", "inter_f", "L_tot_f", "L_inter_f"):
        out[k] = np.array([w[k] for w in W])
        lo, hi = zip(*[ci(w[k + "_boot"]) for w in W])
        out[k + "_lo"], out[k + "_hi"] = np.array(lo), np.array(hi)
    for k in ("phi_f", "dA_f", "dB_f", "L_phi_f"):
        out[k] = np.array([w[k] for w in W])
        lo, hi = zip(*[ci(w[k + "_boot"]) for w in W])
        out[k + "_lo"], out[k + "_hi"] = np.array(lo), np.array(hi)
    out["sq"] = np.array([w["sq"] for w in W])
    for f_ in CTRL_FAM:                                                    # the random families (NaN: not computed)
        for n_ in fam_sets(f_):
            v = np.full((len(W), NR), np.nan)
            lv = np.full(len(W), np.nan)
            for wi in range(len(W)):
                r = ctrl_window(n_, wi, D, W)
                if r is not None:
                    v[wi], lv[wi] = r["phi_f"], r["L_phi_f"]
            out[f"ctrl_{n_}_phi_f"], out[f"ctrl_{n_}_L_phi_f"] = v, lv
    S = alpha_summary(D=D)
    for j, (key, s) in enumerate(S.items()):
        out[f"alpha{j}_eps"] = np.array(key)
        for k in ("tot", "graft", "rewind", "long", "L_tot", "L_graft", "L_rewind"):
            out[f"alpha{j}_{k}"] = s[k]
        for d in ("graft", "rewind"):
            out[f"alpha{j}_compatible_{d}"] = s["compatible"][d]
            out[f"alpha{j}_monotone_{d}"] = s["monotone"][d]
            out[f"alpha{j}_loss_{d}"] = s["loss"][d]
    path = os.path.join(OUT, "summary.npz")
    np.savez(path, **out)
    print("saved", path)
    return path


def _share(w, fam, sets):
    """share of a family's window change (family mean) carried by a union of classes / fine groups: (value, resamples)"""
    ft = family_values(w, fam)
    v, b = 0., 0.
    for c in sets:
        cv, cb = family_values(w, fam, "cls", c if (c in CLASSES or c == "inter") else (c,))
        v, b = v + cv, b + cb
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / ft[0], b / ft[1]


def dissociation(W):
    """the task's hypothesis (W_hh f + i rows carry the 4,000-11,200 holding gain while W_ih / read-out carry the
    800-1,600 answers gain) and every class's share of each family's window change, with paired family differences"""
    print("\nTASK HYPOTHESIS: shares of a family's window change (family means; 5-95% over the paired resamples)")
    wa, wl = W[1], W[3]
    for lab, sets in (("W_hh f + i", ("W_hh_f", "W_hh_i")), ("W_ih + read-out", ("W_ih", "read-out")),
                      ("W_hh g", ("W_hh_g",)), ("gate g", ("gate g",))):
        out = []
        for fam, w, wn in (("answers", wa, win_label(1)), ("holding", wa, win_label(1)), ("answers", wl, win_label(3)),
                           ("holding", wl, win_label(3)), ("coupling", wl, win_label(3))):
            v, b = _share(w, fam, sets)
            lo, hi = ci(b)
            out.append(f"{fam} {wn} {v:+.2f} [{lo:+.2f},{hi:+.2f}]")
        print(f"  {lab:16s} " + "; ".join(out))
    for lab, sets, (f1, w1), (f2, w2) in (
            ("W_hh f + i: holding late - answers early", ("W_hh_f", "W_hh_i"), ("holding", wl), ("answers", wa)),
            ("W_ih + read-out: answers early - holding late", ("W_ih", "read-out"), ("answers", wa), ("holding", wl)),
            ("W_hh f + i: holding late - answers late", ("W_hh_f", "W_hh_i"), ("holding", wl), ("answers", wl)),
            ("W_ih + read-out: answers early - holding early", ("W_ih", "read-out"), ("answers", wa), ("holding", wa))):
        v1, b1 = _share(w1, f1, sets)
        v2, b2 = _share(w2, f2, sets)
        lo, hi = ci(b1 - b2)
        print(f"  {lab:48s} {v1 - v2:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("\nDISSOCIATION TABLE (share of each family's window change carried by a class; differences between families "
          "in the same window, 5-95% over the paired resamples):")
    for wi, w in enumerate(W):
        ft = {fam: family_values(w, fam) for fam in FAMILY_ROWS}
        print(f"  {win_label(wi)}: family change " + ", ".join(f"{fam} {_ci_txt(*ft[fam])}" for fam in ft))
        for cls in list(CLASSES) + ["inter"]:
            sh = {fam: _share(w, fam, (cls,)) for fam in FAMILY_ROWS}
            txt = []
            for a_, b_ in (("holding", "answers"), ("coupling", "answers"), ("holding", "coupling")):
                d, db = sh[a_][0] - sh[b_][0], sh[a_][1] - sh[b_][1]
                lo, hi = ci(db)
                txt.append(f"{a_[:4]}-{b_[:4]} {d:+.2f} [{lo:+.2f},{hi:+.2f}]")
            print(f"    {cls:9s} shares " + ", ".join(f"{fam[:4]} {sh[fam][0]:+.2f}" for fam in sh) + " | "
                  + "; ".join(txt))


def alpha_ctrl_numbers(D, S):
    """the random 1M of W on the alpha paths with 3 seeds (seed 1 = rand_W of the path experiment, ALPHA_CTRL)"""
    S2 = alpha_summary(D=D, sets=ALPHA_CTRL)
    print("\nALPHA PATHS, random 1M of W, 3 seeds (seed 1 = rand_W): rewind / graft at alpha 1, f units [5-95%], "
          "C = loss in the parents' band at every alpha, M = monotone (graft / rewind); seed mean")
    for key, s in S.items():
        s2 = S2[key]
        if not s2["done"].all():
            print(f"  pair {s['x'][0]:,.0f} -> {s['x'][1]:,.0f}: extra seeds not complete")
            continue
        print(f"  pair {s['x'][0]:,.0f} -> {s['x'][1]:,.0f}")
        for k in ALPHA_ROWS:
            tot, tb, eff, ok = _alpha_cols(s, k)
            _, _, eff2, ok2 = _alpha_cols(s2, k)
            gi = ALL_G.index("rand_W")
            cells, rw, gr = [], [], []
            for j, (e_, o_, ix) in enumerate(((eff, ok, gi), (eff2, ok2, 0), (eff2, ok2, 1))):
                rw.append(e_["rewind"][0][ix])
                gr.append(e_["graft"][0][ix])
                cells.append(f"s{j + 1} rewind {_ci_txt(e_['rewind'][0][ix], e_['rewind'][1][ix])} graft "
                             f"{e_['graft'][0][ix]:+.3f} C/M ok {int(o_['graft'][ix])}/{int(o_['rewind'][ix])}")
            print(f"    {k:15s} (change {tot:+.3f}) " + "; ".join(cells) + f" | mean rewind {np.mean(rw):+.3f} "
                  f"(range {min(rw):+.3f}..{max(rw):+.3f}), graft {np.mean(gr):+.3f}")
            if k in ("held_two_fwd", "held_joint_fwd"):
                mb = (eff["rewind"][1][gi] + eff2["rewind"][1].sum(0)) / 3
                print("      rewind minus the random seed mean (paired): " + "; ".join(
                    f"{g} {_ci_txt(eff['rewind'][0][ALL_G.index(g)] - np.mean(rw), eff['rewind'][1][ALL_G.index(g)] - mb)}"
                    f" (interpretable {int(ok['rewind'][ALL_G.index(g)])})" for g in ("W_hh_f", "W_hh_i", "W_ih_i",
                                                                                     "W_hh_o", "W_ih_f")))


def alpha_numbers(D):
    S = alpha_summary(D=D)
    print("\nALPHA PATHS (graft = f(E with g <- L) - f(E); rewind = f(L) - f(L with g <- E); long = their mean; path = "
          "the same set's adjacent-pair phi summed over E -> L; f units; C = training loss inside [L_L, L_E] at every "
          "alpha (graft/rewind); M = the row's alpha curve monotone (graft/rewind))")
    for key, s in S.items():
        cg, cr = s["compatible"]["graft"], s["compatible"]["rewind"]
        print(f"  pair {s['x'][0]:,.0f} -> {s['x'][1]:,.0f}: training loss {s['LE']:.5f} -> {s['LL']:.5f}; loss inside the "
              f"parents' range at every alpha: graft {int(cg.sum())}/{NG} ({', '.join(g for g, c in zip(ALL_G, cg) if c)}), "
              f"rewind {int(cr.sum())}/{NG} ({', '.join(g for g, c in zip(ALL_G, cr) if c)})")
        mx = {d: s["loss"][d].max(1) for d in ("graft", "rewind")}
        print("    max loss along the path (graft / rewind): " + ", ".join(
            f"{g} {mx['graft'][gi]:.4f}/{mx['rewind'][gi]:.4f}" for gi, g in enumerate(ALL_G)))
        print("    sums over the 15 groups vs the change (f units): row: change | sum rewind | sum graft | sum long | sum "
              "path; interpretable (C and M) rewinds / grafts")
        for k in ALPHA_ROWS:
            tot, tb, eff, ok = _alpha_cols(s, k)
            sr, sg = eff["rewind"][0][:NF].sum(), eff["graft"][0][:NF].sum()
            pth = (f"{s['path'][:NF, ROWS.index(k)].sum():+.3f}" if ("path" in s and k != "loss") else "n/a")
            print(f"      {k:16s} {_ci_txt(tot, tb)} | {sr:+.3f} | {sg:+.3f} | {(sr + sg) / 2:+.3f} | {pth}; "
                  f"{int(ok['rewind'][:NF].sum())} / {int(ok['graft'][:NF].sum())} of 15; sum rewind / change "
                  f"{sr / tot:.1f}")
        for k in ALPHA_ROWS:
            tot, tb, eff, ok = _alpha_cols(s, k)
            print(f"    {k} (change {_ci_txt(tot, tb)})")
            for gi, g in enumerate(ALL_G):
                if not s["done"][gi]:
                    continue
                pth = f"{s['path'][gi, ROWS.index(k)]:+.3f}" if ("path" in s and k != "loss") else "  n/a"
                mono = (s["L_monotone"] if k == "loss" else
                        {d: s["monotone"][d][:, ROWS.index(k)] for d in ("graft", "rewind")})
                print(f"      {g:8s} rewind {_ci_txt(eff['rewind'][0][gi], eff['rewind'][1][gi])} graft "
                      f"{_ci_txt(eff['graft'][0][gi], eff['graft'][1][gi])} long "
                      f"{(eff['rewind'][0][gi] + eff['graft'][0][gi]) / 2:+.3f} path {pth} "
                      f"C {int(cg[gi])}/{int(cr[gi])} M {int(mono['graft'][gi])}/{int(mono['rewind'][gi])}")
        if "path" in s:
            sel = [ROWS.index(k) for k in ALPHA_ROWS if k != "loss"]
            a_, b_ = s["path"][:NF][:, sel].ravel(), s["long"][:NF][:, sel].ravel()
            print(f"    path check over 15 groups x {len(sel)} rows: r(long, path) = {np.corrcoef(a_, b_)[0, 1]:.3f}; "
                  f"slope (long on path, through 0) {float((a_ * b_).sum() / (a_ * a_).sum()):.3f}; mean |long - path| "
                  f"{np.abs(b_ - a_).mean():.3f} vs mean |path| {np.abs(a_).mean():.3f}")
    alpha_ctrl_numbers(D, S)


# ═══════════════════════════════════════════════════════════════════
# Figures (final round: lean - only what survived the checks; everything else is in the docstring and `numbers`)
# ═══════════════════════════════════════════════════════════════════

NUM_ROWS = ("acc_goal_T", "far_T", "logq_goal_T", "kl_goal_mean", "held_two_fwd", "held_one_fwd", "held_joint_fwd",
            "emass_slope", "dist_out_fwd", "excess", "G", "dir_slope", "loss")
ROW_ORDER = FINE + ("inter",) + CONTROLS
MIN_CHANGE = .05                                  # text shares only where |change| >= 5% of the naive-Bayes gap


def _window_table(W, k):
    """for a row k (or 'loss'): per window tot_f, its 5-95%, phi_f (NG,), inter_f, and shares phi / tot with resamples"""
    out = []
    for w in W:
        if k == "loss":
            tot, tb, ph, pb, it, ib = (w["L_tot_f"], w["L_tot_f_boot"], w["L_phi_f"], w["L_phi_f_boot"], w["L_inter_f"],
                                       w["L_inter_f_boot"])
        else:
            i = ROWS.index(k)
            tot, tb, ph, pb, it, ib = (w["tot_f"][i], w["tot_f_boot"][i], w["phi_f"][:, i], w["phi_f_boot"][:, i],
                                       w["inter_f"][i], w["inter_f_boot"][i])
        lo, hi = ci(tb)
        plo, phi_hi = ci(pb)
        ilo, ihi = ci(ib)
        with np.errstate(invalid="ignore", divide="ignore"):
            sh = np.concatenate([ph[:NF], [it], ph[NF:]]) / tot
            shb = np.concatenate([pb[:NF], ib[None], pb[NF:]]) / tb[None]
        out.append(dict(tot=tot, lo=lo, hi=hi, phi=ph, phi_lo=plo, phi_hi=phi_hi, inter=it, inter_lo=ilo, inter_hi=ihi,
                        share=sh, share_lo=ci(shb)[0], share_hi=ci(shb)[1],
                        changed=bool((lo > 0 or hi < 0) and abs(tot) >= MIN_CHANGE and abs(it) <= abs(tot)),
                        nonadd=bool((lo > 0 or hi < 0) and abs(tot) >= MIN_CHANGE and abs(it) > abs(tot)),
                        sig=np.concatenate([((plo > 0) | (phi_hi < 0))[:NF], [(ilo > 0) | (ihi < 0)],
                                            ((plo > 0) | (phi_hi < 0))[NF:]])))       # in ROW_ORDER, like share
    return out


FAM_COL = {"answers": "#6a3d9a", "holding": "#3a7a19", "coupling": "#b2182b", "loss": "#8c510a"}
FIG_FAMS = (("answers", "answers"), ("holding", "holding a split"), ("coupling", "coupling"), ("loss", "training loss"))
# the figure's weight classes: (label, fine groups, size-matched random family, colour, line style, legend text)
FIG_SETS = (("W_hh g", ("W_hh_g",), "rHH1M", "#a8a800", "-", r"recurrent g rows, $W_{hh}$ g (1M)  $-$  random 1M of $W_{hh}$"),
            ("gate g", CLASSES["gate g"], "rW2M", "#4d4d00", (0, (4, 2)),
             r"all g rows, $W_{ih}$ g + $W_{hh}$ g, + g biases (2M)  $-$  random 2M of $W$"),
            ("W_hh f+i", CLASSES["W_hh f+i"], "rHH2M", "#d62728", "-",
             r"recurrent f + i rows, $W_{hh}$ f + i (2M)  $-$  random 2M of $W_{hh}$"),
            ("W_hh o", ("W_hh_o",), "rHH1M", "#9467bd", "-", r"recurrent o rows, $W_{hh}$ o (1M)  $-$  random 1M of $W_{hh}$"),
            ("W_ih", CLASSES["W_ih"], "rW4M", "#17becf", "-", r"input weights, $W_{ih}$, all 4 gates (4M)  $-$  random 4M of $W$"))
FIG_WIN = (1, 2, 3)                               # the windows where every one of these random families ran
FIG_MIN = .1                                      # figure: excess shares drawn only where the family's change >= .1


def excess_cells(W, D):
    """the figure's data. Per item (the 3 family means and the training loss) and window of FIG_WIN: the change (tot,
    5-95%), the interaction, drawn = 5-95% excludes 0, |change| >= FIG_MIN and |interaction| <= |change|; per class of
    FIG_SETS its excess share over the size-matched random family, (class - the 3 draws' mean) / change, with the 5-95%
    of the paired resamples' ratios and the 3 per-draw values"""
    if "excess" not in _C:
        out = {}
        for item, _ in FIG_FAMS:
            for wi in FIG_WIN:
                w = W[wi]
                tot, tb = item_value(w, item, "tot")
                lo, hi = ci(tb)
                it = item_value(w, item, "inter")[0]
                c = dict(tot=tot, lo=lo, hi=hi, inter=it,
                         drawn=bool((lo > 0 or hi < 0) and abs(tot) >= FIG_MIN and abs(it) <= abs(tot)))
                for lab, grp, fam, *_ in FIG_SETS:
                    m = matched(W, wi, item, grp, fam, D)
                    s, sb = _sh(m["d"], m["tot"], m["d_b"], m["tot_b"])
                    l_, h_ = ci(sb)
                    c[lab] = dict(v=float(s), lo=float(l_), hi=float(h_), draws=(m["v"] - m["vs"]) / m["tot"])
                out[(item, wi)] = c
        _C["excess"] = out
    return _C["excess"]


def figure_numbers(W, D):
    """every value drawn in td_tt_swaps.png, with the per-draw values; how many drawn resolved points keep their sign
    for each draw"""
    E = excess_cells(W, D)
    print(f"\nFIGURE td_tt_swaps.png: excess share over the size-matched random family, (class - mean of 3 draws) / the "
          f"change [5-95%] (per-draw range); drawn where the change's 5-95% excludes 0, |change| >= {FIG_MIN} and "
          "|interaction| <= |change|")
    n_res = n_keep = 0
    for item, _ in FIG_FAMS:
        for wi in FIG_WIN:
            c = E[(item, wi)]
            cells = []
            for lab, *_ in FIG_SETS:
                x = c[lab]
                cells.append(f"{lab} {x['v']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}] ({x['draws'].min():+.2f}.."
                             f"{x['draws'].max():+.2f})")
                if c["drawn"] and (x["lo"] > 0 or x["hi"] < 0):
                    n_res += 1
                    n_keep += int((np.sign(x["draws"]) == np.sign(x["v"])).all())
            print(f"  {item:8s} {win_label(wi):8s} change {c['tot']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] "
                  f"{'drawn' if c['drawn'] else 'NOT DRAWN'}: " + "; ".join(cells))
    print(f"  drawn points whose 5-95% excludes 0: {n_res}; of them with the same sign for each of the 3 draws: {n_keep}")


def _f2(v):
    """+.52 style (2 decimals, no leading zero)"""
    return f"{v:+.2f}".replace("0.", ".", 1)


def figure(W=None, D=None, name="td_tt_swaps.png"):
    """-> figure_bin/td_tt_swaps.png: per behaviour family (answers, holding a split, coupling) and the training loss,
    how much more of each window's change a class of weights carried than random weights of the same size: (the
    class's two-endpoint Shapley value - the mean of 3 random sets') / the change, 5-95% bars, in the three windows
    where the random sets ran (800-1,600, 1,600-4,000, 4,000-11,200)"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    D = collect() if D is None else D
    W = decompose(D) if W is None else W
    E = excess_cells(W, D)
    nW = len(FIG_WIN)
    PW, PH, GX, LAB = 2.9, 2.3, .32, 1.05
    Y0, TOP = 2.75, 1.15                                                 # panel bottom; room above the panels
    Wd = LAB + 4 * PW + 3 * GX + .3
    Ht = Y0 + PH + TOP
    YL = (-.55, .75)
    fig = plt.figure(figsize=(Wd, Ht))
    xs = np.arange(nW)
    dx = np.linspace(-.22, .22, len(FIG_SETS))
    for j, (item, nm) in enumerate(FIG_FAMS):
        ax = fig.add_axes([(LAB + j * (PW + GX)) / Wd, Y0 / Ht, PW / Wd, PH / Ht])
        ax.axhline(0, color="k", lw=1.1, ls="--", zorder=1)
        for k, wi in enumerate(FIG_WIN):
            if not E[(item, wi)]["drawn"]:
                ax.axvspan(k - .5, k + .5, color="#fbf3d5", lw=0, zorder=0)
                ax.text(k, YL[1] - .04, "change\n$<$ .1:\nnot\ndrawn", ha="center", va="top", fontsize=fs["tick"],
                        linespacing=1.15)
        for si, (lab, grp, fam, col, ls, _) in enumerate(FIG_SETS):
            v, lo, hi = (np.array([E[(item, wi)][lab][q] if E[(item, wi)]["drawn"] else np.nan for wi in FIG_WIN])
                         for q in ("v", "lo", "hi"))
            x = xs + dx[si]
            ax.vlines(x, lo, hi, color=col, lw=1.2, zorder=2)
            ax.plot(x, v, color=col, ls=ls, lw=1.7 if lab.endswith(" g") else 1.2, marker="o", ms=4.6,
                    mfc="w" if lab == "gate g" else col, mec=col, mew=1.2, zorder=3)
        ax.set_ylim(*YL)
        ax.set_xlim(-.5, nW - .5)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{win_label(wi)}\n$\\Delta f$ {_f2(E[(item, wi)]['tot'])}" for wi in FIG_WIN],
                           fontsize=fs["tick"])
        ax.tick_params(axis="x", length=0, pad=3)
        ax.set_yticks([-.4, -.2, 0, .2, .4, .6])
        if j == 0:
            ax.set_yticklabels(["-.4", "-.2", "0", ".2", ".4", ".6"])
            ax.set_ylabel("share of the change carried\nbeyond random weights", fontsize=fs["label"])
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{'abcd'[j]}   {nm}\n"
                     r"$(\Sigma_{\mathrm{class}}\,\phi - \overline{\phi}_{\mathrm{random}})\;/\;\Delta f$",
                     loc="left", fontsize=fs["label"] + 1, color="k", linespacing=1.4, pad=4)
    hand = [Line2D([], [], color=c_, ls=ls, lw=1.6, marker="o", ms=4.6, mfc="w" if lab == "gate g" else c_, mec=c_)
            for lab, _, _, c_, ls, _ in FIG_SETS]
    labs = [t_ for *_, t_ in FIG_SETS]
    hand.append(Line2D([], [], color="k", ls="--", lw=1.1))
    labs.append("0 = what random weights of the same size carry (mean of 3 draws)")
    fig.legend(hand, labs, loc="upper left", bbox_to_anchor=(LAB / Wd, (Y0 - .7) / Ht), ncol=2,
               fontsize=fs["tick"] + .5, frameon=False, handlelength=2.6, columnspacing=2.2, labelspacing=.45)
    fig.suptitle("Swapping weights between consecutive checkpoints (800-11,200 training episodes): the g (candidate) "
                 "rows carry more of\nthe learning than random weights of the same size; of the other rows, as classes, "
                 "none is resolved above the random share (most are below it)", fontsize=fs["title"] - 2,
                 y=1 - .12 / Ht, linespacing=1.3)
    res_ = [c[lab] for c in E.values() if c["drawn"] for lab, *_ in FIG_SETS if c[lab]["lo"] > 0 or c[lab]["hi"] < 0]
    n_res = len(res_)
    n_keep = sum(int((np.sign(x["draws"]) == np.sign(x["v"])).all()) for x in res_)
    short = {"W_hh f+i": r"$W_{hh}$ f + i", "W_hh o": r"$W_{hh}$ o", "W_ih": r"$W_{ih}$"}
    small = []                                    # the undrawn small changes, and any non-g class above random there
    for (item, wi), c in E.items():
        if c["drawn"]:
            continue
        up = [f"{short[lab]} {_f2(c[lab]['v'])} [{_f2(c[lab]['lo'])}, {_f2(c[lab]['hi'])}]" for lab in short
              if c[lab]["lo"] > 0]
        tot_ = f"{c['tot']:+.3f}".replace("0.", ".", 1)
        small.append(f"{dict(FIG_FAMS)[item]} {win_label(wi)}, $\\Delta f$ {tot_}"
                     + (", where " + ", ".join(up) + " also exceed random" if up else ""))
    npairs = [W[wi]["n_pairs"] for wi in FIG_WIN]
    n_grp = 2 * NF * sum(npairs)
    n_rnd = 2 * 3 * sum(npairs) * len({f_ for _, _, f_, *_ in FIG_SETS})
    lines = [
        f"Data (seed 20261029, Trained RNN): {sum(npairs)} consecutive checkpoint pairs ({' / '.join(map(str, npairs))} "
        f"per window); per pair every weight set swapped both ways: {n_grp:,} networks for the 15 weight groups + "
        f"{n_rnd:,} for 4 random families x 3 draws;",
        "each network scored on 10,000 probe episodes (50 test environments x 100 latent pairs x 2 trajectories), "
        "5,028 forward-set episodes (held rows) and 2,000 training episodes (training loss).",
        r"$\phi$: two-endpoint Shapley value of a swap, summed over the window's pairs; $f$: gap fraction (naive $\to$ "
        r"Bayes = 1); a family = the mean of its rows; $\Delta f$: the window's change.",
        f"Bars: 5-95% over 1,000 resamples of the 50 environments (loss: of its 2,000 episodes), paired with the random "
        f"sets; {n_keep} of the {n_res} points whose bar excludes 0 keep their sign for each of the 3 draws.",
        "Not drawn (all in the docstring): changes $<$ .1 of the gap (" + "; ".join(small) + ");",
        "the read-in and read-out (whole tensors, no random counterpart); 0-800 and 11.2k-40k (none of these four "
        "random families ran there)."]
    fig.text(LAB / Wd, .12 / Ht, "\n".join(lines), ha="left", va="bottom", fontsize=fs["tick"], linespacing=1.5)
    path = fig_path(name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


ALPHA_ROWS = ("acc_goal_T", "logq_goal_T", "held_two_fwd", "held_joint_fwd", "emass_slope", "excess", "G",
              "dir_slope", "loss")


def _alpha_cols(s, k):
    """per alpha pair summary s and row k: total change (value, resamples) and per set graft / rewind effects (NG,) with
    resamples (NG, NB), interpretable flags (NG,) per direction"""
    if k == "loss":
        tot, tb = s["L_tot"], s["L_tot_boot"]
        eff = {"graft": (s["L_graft"], s["L_graft_boot"]), "rewind": (s["L_rewind"], s["L_rewind_boot"])}
        mono = {d: s["L_monotone"][d] for d in eff}
    else:
        i = ROWS.index(k)
        tot, tb = s["tot"][i], s["tot_boot"][i]
        eff = {d: (s[d][:, i], s[d + "_boot"][:, i]) for d in ("graft", "rewind")}
        mono = {d: s["monotone"][d][:, i] for d in eff}
    ok = {d: s["compatible"][d] & mono[d] for d in eff}
    return tot, tb, eff, ok


ALPHA_FIG_SETS = ("readin", "W_ih_i", "W_ih_f", "W_ih_g", "W_ih_o", "W_hh_i", "W_hh_f", "W_hh_g", "W_hh_o", "readout")
ALPHA_XLAB = {"readin": "read-in", "W_ih_i": r"$W_{ih}$ i", "W_ih_f": r"$W_{ih}$ f", "W_ih_g": r"$W_{ih}$ g",
              "W_ih_o": r"$W_{ih}$ o", "W_hh_i": r"$W_{hh}$ i", "W_hh_f": r"$W_{hh}$ f", "W_hh_g": r"$W_{hh}$ g",
              "W_hh_o": r"$W_{hh}$ o", "readout": "read-out", "rand": "random 1M\nof $W$ (3)"}
DIR_STYLE = {"rewind": ("#8e0152", "o", None), "graft": ("#276419", "s", "w")}   # colour, marker, face


def alpha_loss_effects(S, S2):
    """per alpha pair: the training loss's whole change (value, 5-95%) and, per set of ALPHA_FIG_SETS + 'rand' (the
    random 1M of W: mean of its 3 draws), the rewind and graft effects at alpha 1 (value, 5-95%; rand: + per draw), f
    units; and the largest |effect| of the biases and the start state"""
    out = {}
    for key, s in S.items():
        s2 = S2[key]
        r = dict(tot=s["L_tot"], tot_ci=ci(s["L_tot_boot"]), x=s["x"])
        for d in ("rewind", "graft"):
            v, vb = s[f"L_{d}"], s[f"L_{d}_boot"]
            ix = [ALL_G.index(g) for g in ALPHA_FIG_SETS]
            lo, hi = ci(vb[ix])
            irw = ALL_G.index("rand_W")
            draws = np.concatenate([[v[irw]], s2[f"L_{d}"]])
            rb = (vb[irw] + s2[f"L_{d}_boot"].sum(0)) / 3
            r[d] = dict(v=np.append(v[ix], draws.mean()), lo=np.append(lo, ci(rb)[0]), hi=np.append(hi, ci(rb)[1]),
                        rand_draws=draws)
        small = [ALL_G.index(g) for g in ("b_i", "b_f", "b_g", "b_o", "start")]
        r["small_max"] = float(max(np.abs(s["L_rewind"][small]).max(), np.abs(s["L_graft"][small]).max()))
        out[key] = r
    return out


def figure_alpha(S=None, D=None, name="td_tt_swaps_alpha.png"):
    """-> figure_bin/td_tt_swaps_alpha.png, one panel per checkpoint pair E -> L (800 -> 1,600, 1,600 -> 4,000, 4,000
    -> 11,200): the training loss's change (gap fraction) when one weight set alone is taken back in L to its E values
    (rewind) or taken forward from L into E (graft), at alpha 1 of the stored alpha paths, against the pair's whole
    change"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    D = collect() if D is None else D
    S = alpha_summary(D=D) if S is None else S
    S2 = alpha_summary(D=D, sets=ALPHA_CTRL)
    A = alpha_loss_effects(S, S2)
    keys = list(A)
    nx = len(ALPHA_FIG_SETS) + 1
    PW, PH, GX, LAB = 3.5, 2.5, .32, 1.0
    Y0, TOP = 2.6, 1.6
    Wd = LAB + len(keys) * PW + (len(keys) - 1) * GX + .3
    Ht = Y0 + PH + TOP
    YL = (-.42, .55)
    fig = plt.figure(figsize=(Wd, Ht))
    xs = np.arange(nx)
    for pi, key in enumerate(keys):
        a = A[key]
        ax = fig.add_axes([(LAB + pi * (PW + GX)) / Wd, Y0 / Ht, PW / Wd, PH / Ht])
        ax.axhline(0, color="k", lw=.8, zorder=1)
        ax.axhline(a["tot"], color=FAM_COL["loss"], lw=1.4, ls="--", zorder=1)
        ax.axvspan(nx - 1.5, nx - .5, color="#fbf3d5", lw=0, zorder=0)
        for d, off in (("rewind", -.14), ("graft", .14)):
            col, mk, face = DIR_STYLE[d]
            e = a[d]
            ax.vlines(xs + off, e["lo"], e["hi"], color=col, lw=1.2, zorder=2)
            ax.plot(xs + off, e["v"], ls="none", marker=mk, ms=5.2, color=col, mfc=face or col, mew=1.2, zorder=3)
        ax.set_xlim(-.6, nx - .4)
        ax.set_ylim(*YL)
        ax.set_xticks(xs)
        ax.set_xticklabels([ALPHA_XLAB[g] for g in ALPHA_FIG_SETS] + [ALPHA_XLAB["rand"]], rotation=60, ha="right",
                           rotation_mode="anchor", fontsize=fs["tick"])
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_yticks([-.4, -.2, 0, .2, .4])
        if pi == 0:
            ax.set_yticklabels(["-.4", "-.2", "0", ".2", ".4"])
            ax.set_ylabel(r"training loss, $\Delta f$" + "\n(+ = closer to Bayes)", fontsize=fs["label"])
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{'abc'[pi]}   E = {a['x'][0]:,.0f} $\\to$ L = {a['x'][1]:,.0f} training episodes\n"
                     r"rewind of one set $s$ ($\alpha$ = 1): $f(L) - f(L_{s\leftarrow E})$" "\n"
                     r"graft of one set $s$ ($\alpha$ = 1): $f(E_{s\leftarrow L}) - f(E)$" "\n"
                     f"whole change $f(L) - f(E)$ = {_f2(a['tot'])} (dashed)", loc="left", fontsize=fs["label"] + 1,
                     linespacing=1.4, pad=4)
    hand = [Line2D([], [], color=DIR_STYLE["rewind"][0], marker="o", ls="none", ms=5.2),
            Line2D([], [], color=DIR_STYLE["graft"][0], marker="s", mfc="w", ls="none", ms=5.2, mew=1.2),
            Line2D([], [], color=FAM_COL["loss"], ls="--", lw=1.4)]
    labs = [r"rewind, what is lost: $f(L) - f(L_{s\leftarrow E})$, $L$ with set $s$'s values from $E$",
            r"graft, what is gained: $f(E_{s\leftarrow L}) - f(E)$, $E$ with set $s$'s values from $L$",
            r"the pair's whole change $f(L) - f(E)$"]
    fig.legend(hand, labs, loc="upper left", bbox_to_anchor=(LAB / Wd, (Y0 - 1.0) / Ht), ncol=2,
               fontsize=fs["tick"] + .5, frameon=False, handlelength=2.0, columnspacing=2.2, labelspacing=.45)
    fig.suptitle("Checkpoints further apart: taking one weight set back costs up to more than the whole improvement; "
                 "taking one set\nforward alone gains little or loses - over these stretches the sets' changes work "
                 "only together (co-adaptation)", fontsize=fs["title"] - 2, y=1 - .12 / Ht, linespacing=1.3)
    small = max(a["small_max"] for a in A.values())
    spread = max(float(np.abs(a[d]["rand_draws"] - a[d]["rand_draws"].mean()).max()) for a in A.values()
                 for d in ("rewind", "graft"))
    lines = [
        f"Data (seed 20261029, Trained RNN): {len(keys)} checkpoint pairs x {NG} weight sets x 2 directions = "
        f"{len(keys) * NG * 2} networks (alpha = 1 of the stored alpha paths) + {len(keys) * len(ALPHA_CTRL) * 2} for 2 "
        f"more draws of a random 1M of $W$ (drawn: the mean of 3 draws; each draw within {spread:.3f} of it);",
        r"each scored by its training loss on 2,000 training-distribution episodes; $f$: gap fraction of the loss "
        r"(naive $\to$ Bayes = 1). Bars: 5-95% over 1,000 resamples of the episodes.",
        "If the sets' changes added up, each set's rewind and graft would both equal its own part of the dashed whole "
        f"change. Biases and the start state ($|$effect$|$ $\\leq$ {small:.3f}) are not drawn;",
        "the behaviour rows show the same in sum (the module's docstring, finding 10)."]
    fig.text(LAB / Wd, .12 / Ht, "\n".join(lines).replace(" 0.", " ."), ha="left", va="bottom",
             fontsize=fs["tick"], linespacing=1.5)
    path = fig_path(name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


# ═══════════════════════════════════════════════════════════════════
# Checks (-> OUT/checks.json)
# ═══════════════════════════════════════════════════════════════════

CHECK_PAIR = (2000, 2100)                         # training 1,600 -> 1,680
REPRO_EPS = (25, 2100, 49995)                     # intact units re-measured in this process


def checks(dev=None):
    """validation of the swap machinery and of the stored runs (prints; -> OUT/checks.json)"""
    dev = ti._device(dev)
    res = {}
    t0 = time.time()
    # 1. weight sets: sizes, exact swaps, the partition
    wa, wb = ti.load(RUN, CHECK_PAIR[0]), ti.load(RUN, CHECK_PAIR[1])
    res["n_params"] = {g: n_params(g) for g in ALL_G}
    res["partition_total"] = int(sum(res["n_params"][g] for g in FINE))
    res["all_weights"] = int(sum(v.numel() for v in wa.values()))
    exact = {}
    for g in ALL_G:
        h = swap(wa, wb, g)
        ok = True
        for key in ti.KEYS:
            sel = torch.zeros_like(wa[key], dtype=torch.bool)
            for kk, ix in entries(g):
                if kk == key:
                    sel[ix] = True
            ok &= bool(torch.equal(h[key][sel], wb[key][sel])) and bool(torch.equal(h[key][~sel], wa[key][~sel]))
        exact[g] = ok
    res["swap_exact"] = exact
    h = wa
    for g in FINE:
        h = swap(h, wb, g)
    res["all_fine_swapped_is_wb"] = bool(all(torch.equal(h[k], wb[k]) for k in ti.KEYS))
    res["controls_inside"] = {g: [k for k, _ in entries(g)] for g in CONTROLS}
    # 1b. the random families: sizes, pools, seed 1 of rW1M / rb2k = rand_W / rand_b, overlap between seeds
    cm = control_masks()
    res["family_seed1_is_control"] = {f_: bool(all(k1 == k2 and torch.equal(m1, m2) for (k1, m1), (k2, m2)
                                                   in zip(family_mask(f"{f_}_1"), cm[CTRL_ALIAS[f"{f_}_1"]])))
                                      for f_ in ("rW1M", "rb2k")}
    fam_chk = {}
    for f_, (pool, k_, _) in CTRL_FAM.items():
        ms = [family_mask(n_) for n_ in fam_sets(f_)]
        inside = True
        for m_ in ms:
            for key, mk in m_:
                rows_ = dict(CTRL_POOLS[pool])[key]
                if rows_ is not None:
                    inside &= int(mk[:rows_[0]].sum() + mk[rows_[1]:].sum()) == 0
        npool = 0
        for key, rows_ in CTRL_POOLS[pool]:
            shp = tuple(wa[key].shape)
            r0, r1 = (0, shp[0]) if rows_ is None else rows_
            npool += (r1 - r0) * int(np.prod(shp[1:]))
        fam_chk[f_] = dict(size=[int(sum(int(mk.sum()) for _, mk in m_)) for m_ in ms], target=k_, pool_size=npool,
                           keys=sorted({key for m_ in ms for key, _ in m_}), inside_pool_rows=bool(inside),
                           overlap_s1_s2=int(sum(int((a_[1] & b_[1]).sum()) for a_, b_ in zip(ms[0], ms[1]))))
    res["families"] = fam_chk
    exact_f = {}
    for n_ in ("rHH1M_2", "rG1M_1", "rW4M_3"):
        h = swap(wa, wb, n_)
        ok = True
        for key in ti.KEYS:
            sel = torch.zeros_like(wa[key], dtype=torch.bool)
            for kk, ix in entries(n_):
                if kk == key:
                    sel[ix] = True
            ok &= bool(torch.equal(h[key][sel], wb[key][sel])) and bool(torch.equal(h[key][~sel], wa[key][~sel]))
        exact_f[n_] = ok
    res["family_swap_exact"] = exact_f
    # 1c. the direction-only read-out swap (final round): this checkpoint's norm, the other's direction, nothing else
    kW = "classifier_readout.weight"
    h = swap(wa, wb, "ro_dir")
    Hd, Wad, Wbd = h[kW].double(), wa[kW].double(), wb[kW].double()
    res["ro_dir_construction"] = dict(
        norm_rel_diff=float(abs(Hd.norm() / Wad.norm() - 1)),
        cos_to_other=float((Hd * Wbd).sum() / (Hd.norm() * Wbd.norm())),
        others_unchanged=bool(all(torch.equal(h[k], wa[k]) for k in ti.KEYS if k != kW)))
    print(f"read-out direction-only swap: {res['ro_dir_construction']}", flush=True)
    print(f"random families: seed 1 = path controls {res['family_seed1_is_control']}; swaps exact {exact_f}; "
          + "; ".join(f"{f_} sizes {v['size']} (target {v['target']:,}) in pool rows {v['inside_pool_rows']}, seeds 1/2 "
                      f"overlap {v['overlap_s1_s2']:,}" for f_, v in fam_chk.items()), flush=True)
    print(f"weight sets: {res['partition_total']:,} of {res['all_weights']:,} weights in the 15 fine groups; swaps exact "
          f"{all(exact.values())}; all fine groups swapped in turn = W_k+1: {res['all_fine_swapped_is_wb']}; controls "
          + ", ".join(f"{g} {res['n_params'][g]:,}" for g in CONTROLS), flush=True)
    # 2. the stored intact runs vs the stored ck accuracy (td_analysis) and the stored training losses
    D = collect(partial=True)
    da, dl, flips = [], [], []
    for k, e in enumerate(D["eps"]):
        if not D["I_done"][k]:
            continue
        z = np.load(os.path.join(ti.AN, RUN, f"ck_{e:05d}.npz"))
        da.append(abs(D["I_val"][k, ROWS.index("acc_goal_T")] - float(z["trunc_acc"][3, 0])))
        flips.append(int(round(da[-1] * 10000)))
        p = os.path.join(tel.OUT, RUN, f"loss_main_{e:05d}.npz")
        if os.path.exists(p):
            dl.append(float(np.abs(D["I_loss"][k] / np.load(p)["l"].sum(1) - 1).max()))
    res["intact_acc_vs_ck"] = dict(n=len(da), max_abs=float(max(da)), n_differ=int(sum(f > 0 for f in flips)),
                                   probes_differ=int(sum(flips)))
    res["intact_loss_vs_stored"] = dict(n=len(dl), max_rel_per_episode=float(max(dl)))
    print(f"intact goal accuracy vs stored ck trunc_acc[intact]: {res['intact_acc_vs_ck']}; training loss vs stored "
          f"loss_main (per episode, relative): {res['intact_loss_vs_stored']}", flush=True)
    # 3. reproducibility across processes: re-measure three intact checkpoints here
    rep = {}
    for e in REPRO_EPS:
        k = int(np.flatnonzero(D["eps"] == e)[0])
        m = measure(ti.load(RUN, e), dev)
        rep[str(e)] = dict(val=float(np.nanmax(np.abs(m["val"] - D["I_val"][k]))),
                           boot=float(np.nanmax(np.abs(m["boot"].astype(np.float64) - D["I_boot"][k]))),
                           loss=float(np.abs(m["loss"] - D["I_loss"][k]).max()))
    res["reproduce"] = rep
    print(f"re-measured in this process vs the workers' files (max |d| value / resamples / loss): {rep}", flush=True)
    # 4. decomposition identities and one-sided terms (all pairs done)
    if D["done"].all() and D["I_done"].all():
        W = decompose(D)
        res["window_identity"] = float(max(np.nanmax(np.abs(w["tot"] - w["phi"][:NF].sum(0) - w["inter"])) for w in W))
        res["window_identity_boot"] = float(max(np.nanmax(np.abs(w["tot_boot"] - w["phi_boot"][:NF].sum(0)
                                                                 - w["inter_boot"])) for w in W))
        res["telescoping"] = float(max(np.nanmax(np.abs(w["tot"] - (D["I_val"][w["pairs"] + 1]
                                                                    - D["I_val"][w["pairs"]]).sum(0))) for w in W))
        res["nan_rows"] = {win_label(i): [ROWS[j] for j in np.flatnonzero(~np.isfinite(w["tot"]))]
                           for i, w in enumerate(W)}
        print(f"window identities: sum of phi + interaction = change {res['window_identity']:.1e} (resamples "
              f"{res['window_identity_boot']:.1e}); telescoping {res['telescoping']:.1e}; rows without a value: "
              f"{res['nan_rows']}", flush=True)
        S_ = ro_summary(W, D)
        if S_ is not None:
            res["ro_all_vs_stored_readout"] = S_["ro_all_vs_stored"]
            print(f"read-out split: ro_all vs the stored 'readout' swaps, max |d| (values, losses) "
                  f"{S_['ro_all_vs_stored']:.1e}", flush=True)
    gp = os.path.join(OUT, "gains.npz")
    if os.path.exists(gp):
        res["gains_forward_vs_engine"] = float(np.load(gp)["logq_maxdiff"].max())
    res["seconds"] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "checks.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"saved {os.path.join(OUT, 'checks.json')} ({res['seconds']:.0f} s)")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="doc")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--kinds", default="intact,pair,alpha")
    ap.add_argument("--reverse", action="store_true")
    A_ = ap.parse_args()
    if A_.what == "run":
        worker(A_.worker, A_.nworkers, tuple(A_.kinds.split(",")), A_.reverse)
    elif A_.what == "checks":
        checks()
    elif A_.what == "numbers":
        numbers()
    elif A_.what == "summary":
        export_summary()
    elif A_.what == "figure":
        figure()
    elif A_.what == "figure_alpha":
        figure_alpha()
    else:
        print(__doc__)
