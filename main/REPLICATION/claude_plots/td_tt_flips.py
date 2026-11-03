"""Multi-bit observation flips through training: is there a phase transition in (lag x number of bits flipped)?
(John 11-02, on td_tt_chaos_sweep.png panel d, where the Trained RNN's growth of one flipped observation's impact rose
to ~+2 nats at lag 20 around 300-1,000 episodes, far above Bayes's +.21, and vanished by the end: "I wonder if we would
see a phase transition with variables 'lag' and '# bits flipped'. If we need more data at higher lags we can pool lags
over bins of 3 lags: 1-3, 6-9, 12-15, 18-21.")
-> main/REPLICATION/figure_bin/td_tt_flips.png  figure()
   main/DATA/experimental/training_dynamics/through_training/flips/combined.npz  export(): the arrays for the combined
   chaos figure (layout: COMBINED_LAYOUT, also stored in the file as 'layout')
   numbers(): every number below (~1 min, CPU); checks() -> flips/checks.json (~6 min, CPU)
   caches flips/: <run>/ck_EEEEE.npz per checkpoint (dh, skl per probe, flip step, k, lag 0-21 and variable, float16,
   means over the C(5, k) flip sets; per-environment sums of dh, skl and of the linear response, float64; the k = 1
   check against the chaos sweep), <run>/cls_EEEEE.npz per checkpoint (count classes: per-environment sums by (k, j)
   and of the paired contrasts, float64), refs.npz and refs_classes.npz (the exact observers), harvest.log, classes.log.
   (td_tt_flips_panels.npz is the previous round's export, superseded by combined.npz.)

ANSWER: no boundary in (lag, k) at any stage and no critical k; the one abrupt change is an early COUNT CODE. At 20-40
training episodes the Trained RNN's response to a flip depends on how many of the 5 observations are 1, not on which: a
2-bit flip that swaps a 0 and a 1 barely moves it (growth +.12 vs +1.77 nats for a 2-bit flip that changes the count, at
20 episodes), and 3- or 5-bit flips that change the count by one act like single flips. Bayes, naive and factorized-
state Bayes are not count-coded, nor is the initial network; the ESN starts count-coded too, but its code fades
gradually (from +.596 at 140 episodes to +.015 [-.061, +.093] at 9,600; at the end +.118 [+.023, +.221], resolved
above 0 again).
In the Trained RNN the count code switches off between 40 and 100 episodes (3 checkpoint intervals): count-preserving
2-bit growth rises from +.115 to +1.894, and the zig-zag the code leaves in k falls below half its peak between 40 and
60 episodes. After that the response is smooth in k (no increment stands out) and nearly concave in lag (no onset). The
long-lag amplification rises within the first 20-75 episodes; from 200-360 to 640-1,040 episodes single-flip growth
does not change detectably while 5-bit growth keeps rising (paired difference +.796 [+.596, +.992]); every k falls below
half its peak at 4.2-5.2k episodes (0.7-0.8 decades after passing naive) and ends at or below Bayes's level. Several
flips act more than linearly for good from 483-633 episodes, as Bayes's marginals do. The bulk (SKL) growth turns
contractive, long lags first, after the answers are half-way to Bayes (k = 1-3; k = 4 just after, the intervals nearly
touch; k = 5 overlapping). All five Trained RNN runs show the count code early. It is large in the loss-power-.5
widths, where their long-lag growth first rises (index peak +1.2 to +1.8 nats at 20-120 episodes; about half of the
index's collapse falls in one checkpoint interval; the zig-zag falls from 81-100% to 4-16% of its peak in one
interval). It is small in nats at loss power 1 (index +.087 at 20 episodes, count-preserving growth +.009), whose
long-lag growth rises only at ~1k episodes, without a count code. Single-flip growth falls below half its peak
0.59-0.86 decades after each run passes naive (N = 50: not by the end), only roughly in step (N = 100's interval
separates from N = 1000's); the 5-bit fall does not keep pace (N = 200 at 1.19 [1.06, open] decades against N =
1000's 0.71 [0.65, 0.74]; N = 100 not by +0.80).

DATA
  Networks: the seed-20261029 Trained RNN (trained, N = 1000) and ESN (echo), all 114 stored checkpoints each (training
  episodes E - E // 5; the initial network drawn at 10); replication: trained_N200, trained_N100, trained_N50 (loss
  power .5) and trained_p1 (N = 1000, loss power 1), the same world, probes and episode stream, 114 checkpoints each.
  Probes: td_tt_chaos.subset() (copied: 20 per test environment, seed 20261104; 1,000 of analysis/probes.npz's 10,000
  episodes = 50 environments x 100 cells x 2 streams of 30 steps; equal to the stored chaos sweep's rows). Flips at
  steps s = 1, 3, 5, 7, 9 (every s allows lag 21: s + 21 <= 30), every non-empty subset of the 5 observations at s (31
  flip sets: C(5, k) for k = 1..5), lags 0-21: per network and checkpoint 1,000 x 5 x 31 = 155,000 reruns of 22 steps.
DEFINITIONS
  Flip (td_tt_chaos.network's loop, copied): the network is rerun from its intact state at s - 1 with the observation
    pattern code ^ mask at s and the intact observations after; D_t = the cumulative sum over steps s..t of (flipped -
    intact) read-out increments (float64) = the change of the log belief up to a constant. Per variable, between the
    intact marginal p and the flipped q at t: d_H = max_r (log p_r - log q_r) - min_r (...) (Hilbert's projective
    metric: the range of the log-ratio; set by the values both beliefs treat as near-impossible) and SKL = Sigma_r
    (p_r - q_r)(log p_r - log q_r) / 2 (the bulk; SKL <= d_H / 2). chi(s + lag, s) = the measure at step s + lag;
    immediate impact = chi(s, s); growth = chi(s + lag, s) - chi(s, s), paired on the same flip set and probe (0 at lag
    0); a growth of 0 means the effect is back at its immediate size.
  Averaging (statistic per finest unit, then means): per probe and variable over the C(5, k) flip sets of each k, then
    over the 5 flip steps, the 2 variables (also kept apart) and the probes, as per-environment sums; pooled as ratios
    of sums; [5-95%] over the 1,000 shared environment resamples (td_tt_gap.boot_counts(), paired with the timeline's
    and td_tt_replication's rows). Lag bins (John's): 1-3, 6-9, 12-15, 18-21, each the mean over its lags; every lag
    0-21 rests on the same flips, so no pooling is needed for data.
  Count class (k, j): of the k flipped bits, j were 1, so the number of ones changes by k - 2j (net). Count-preserving
    = net 0 (2 bits: one 0 -> 1 and one 1 -> 0). Class mean = the mean over its (probe, flip step, flip set) units
    (variable means), a ratio of environment sums. COUNT-CODE INDEX (k2_cc): per probe and flip step, the mean d_H
    growth of the count-changing 2-bit flip sets (net +-2) minus that of the count-preserving ones, averaged over the
    (probe, flip step) units where both exist (paired): the extra growth from changing the count. It marks a count code
    (a code of how many observations are on) only where the count-preserving growth itself is ~0, as at 20-40
    episodes; it is also large when both kinds of flip grow strongly and count-changing ones grow more (a partial count
    dependence, not a code, as at 100-880 episodes); ~0 when swapping a 0 and a 1 matters as much as changing the
    count. The switch is therefore read from the count-preserving growth and the alternation, not from the index.
    k3_vs_k1: 3-bit flips with net +-1 minus single flips (paired; ~0 under a count code).
  Alternation (the count code's footprint in the C(5, k) means; even k include count-preserving sets): with inc_k =
    G(k) - G(k - 1) (G(0) = 0 exactly), ((inc3 - (inc2 + inc4) / 2) - (inc4 - (inc3 + inc5) / 2)) / 2 = (G(5) - 4 G(4)
    + 6 G(3) - 4 G(2) + G(1)) / 4, a quarter of the 4th difference of G: exactly 0 when inc2..inc5 lie on a quadratic
    in k (e.g. change linearly), small for smooth profiles, > 0 when the odd-numbered bits 3 and 5 add more than their
    neighbours. Switch time: the checkpoint interval after which it stays below half its early peak (raw curve; every
    resample its own peak). numbers() also prints the log-interpolated point inside that interval with its resampled
    range; the 20-episode checkpoint grid does not resolve it.
  Linear response: the same measures of the sum of the k single-flip changes D^(o) over the flipped observations (k = 1:
    the flip itself). Actual - linear > 0: several flips act more than the sum of single flips (d_H).
  Exact references (float64, the same flips and probes): Bayes = the joint posterior's marginals (a k-bit flip adds
    Sigma_o sign_o logit L_o(r0, r1) to the joint log posterior), naive (a translation of each log marginal: d_H growth
    exactly 0), factorized-state Bayes (td_tt_chaos.adf_run, rerun from its intact marginals at s - 1).
  Trend over training: the running median over 5 checkpoints (window truncated at the ends; the initial network keeps
    its own value): one checkpoint cannot move it, a sustained change does. On the trend: peak = the maximum of its
    3-checkpoint running mean; rise f = first reaching f x peak (log-interpolated); fall f = below f x peak for good
    (td_tt_gap.crossing); SKL zero = below 0 for good. Two crossings are ordered only when their 5-95% intervals do not
    overlap. Excursion = curve - trend. Sharpness of a decline: the largest single checkpoint-interval step of the trend
    while it falls from 75% to 25% of its peak, as a share of that drop, and the number of intervals it spans.
  Plateau comparison: the mean growth over the checkpoints at 200-360 (9) and 640-1,040 (6) training episodes.
  Own clock (width runs): training episodes / the run's own passing-naive time c0 (td_tt_replication; same resamples).
  Per-update periodicity: the online logs (one update per training episode) at 160-400 training episodes: the loss and
    the update norms of W_hh, W_ih and the read-out, minus a cubic trend; amplitude of the best sinusoid at each period
    of 10-120 updates; test: the largest amplitude at 55-70 updates against 1,000 AR(1) surrogates with the series'
    lag-1 correlation and variance.
RESULTS (Trained RNN unless named; d_H; lags 18-21 unless named; k = 1 / 5 where two values; [5-95%])
 1 The early count code. The initial network has none (count-preserving vs count-changing 2-bit growth +.036 vs +.035).
   At 20 episodes count-preserving 2-bit flips grow +.117 [+.112, +.123] and count-changing ones +1.770 [+1.703, +1.831]
   (immediate impacts .055 vs .542); the class means follow the count change, not k: single flips +.90, 3 bits net +-1
   +.95, 5 bits net +-1 +1.00; 2 bits net +-2 +1.77, 4 bits net +-2 +1.87; 3 bits net +-3 +2.53, 5 bits net +-3 +2.69;
   4 bits net 0 +.16 (k3_vs_k1 +.007 [+.003, +.011]). The switch: count-preserving growth is still +.115 [+.108,
   +.122] at 40, then +.636 (60), +1.442 (80) and +1.894 [+1.758, +2.029] at 100, where count-changing growth is
   +1.998. The count-code index (the extra growth from changing the count): +1.778 [+1.727, +1.821] (20), +1.513 (40),
   +1.102 (60), +.377 (80), +.153 [+.053, +.253] (100); of the 1.625 [1.503, 1.745]-nat drop from 20 to 100, .45
   [.41-.48] falls in 60 -> 80 (.16 / .25 / .45 / .14 per interval). By 100 the class means mostly follow k
   (single flips +1.37; 3 bits net +-1 +2.32 vs net +-3 +2.51; 4 bits net 0 +2.66 vs net +-4 +2.96; k3_vs_k1 +.928
   [+.849, +1.008]). In the C(5, k) means it is a zig-zag in k: increments +.90 / +.08 / +.42 / +.09 / +.32 at 20;
   alternation +.311 [+.293, +.327] (20), +.252 (40), +.051 [+.040, +.061] (60), -.007 (80); below half its peak for
   good between 40 and 60 episodes (one checkpoint interval), within +-.023 from 80 on (lags 1-3: +.095 at 20, below
   half between 40 and 60). Bayes, naive and factorized-state Bayes are not count-coded: index +.019 [-.010, +.047], 0,
   +.251 [+.010, +.495] against count-changing growths of +.40, 0, +4.49 (factorized-state Bayes's count-preserving
   flips grow +4.29: a small count dependence, not a code); their k3_vs_k1 +.374, 0, +2.88. After the switch the index
   stays above 0 at every printed checkpoint up to 880 episodes, as a partial count dependence: count-preserving flips
   grow +1.9 to +2.7 nats at 100-400 (not a count code) while the index peaks at +.917 [+.696, +1.150] at 240 (half its
   early peak) and is +.185 [+.095, +.283] at 880. It is ~0 from ~1,200 (+.094 [-.002, +.188]) to the end (-.040
   [-.095, +.014]). The count-coded response is linear in the bits: actual - linear within +-.03 at 20-40 for every k.
 2 No critical k and no onset in lag, after the switch. The largest change of the per-bit increment from one k to the
   next (over k = 2-5) is .345 [.330, .358] at 20 and .283 at 40 (the count code), .087 at 60 and at most .179 [.156,
   .203] (at 180) from 80 on, against the increments themselves (+1.67 / +.65 / +.58 / +.56 / +.55 at 320, +1.51 /
   +.77 / +.75 / +.74 / +.78 at 880: after the first bit nearly flat in k). The previous record's "at most .073 at
   any stage" was measured at four stages (320, 880, 1,680, 39,996); 40 of the 113 trained checkpoints exceed .073.
   The first bit's larger step is the concavity of d_H (the range of a sum of changes in different directions):
   present in the ESN (+.70 then +.33 / +.27 / +.25 / +.23 at 1,680), factorized-state Bayes (+2.82 then +1.58 / +1.34
   / +1.11 / +.74) and the linear response (+1.67 then +.77 / +.67 / +.62 / +.60 at 320), not in Bayes (+.21 / +.17 /
   +.21 / +.24 / +.28). Lag: the largest second difference over lags 1-20 at any checkpoint is +.055 (640 episodes, k =
   4, lag 3; above +.02 at 8 checkpoints between 400 and 1,440; <= +.020 at the previous record's four stages),
   against growths of 1-5 nats: nearly concave, no onset.
 3 Through training. Half the peak is reached inside the first checkpoint interval (k = 1, before 20 episodes) to 75
   [70-79] (k = 5). Single flips then level off while multi-bit flips keep rising: from the 200-360 to the 640-1,040
   checkpoints the growth changes by -.064 [-.200, +.064] (k = 1), +.249 [+.035, +.455] (k = 3), +.732 [+.421, +1.034]
   (k = 5); paired (k = 5) - (k = 1) +.796 [+.596, +.992]. Peak values 1.69 [1.62-1.89] (k = 1) to 4.78 [4.42-5.05]
   (k = 5; the peak times are not compared: their resampled locations were not computed), against Bayes +.21 / +1.11
   and factorized-state Bayes +2.82 / +7.58. Below half the peak for good at 5,226 [4,134-5,581] (k = 1), 5,220 /
   4,595 / 4,356 (k = 2-4), 4,237 [3,754-4,586] (k = 5): overlapping intervals, no order in k. The decline spans
   43 / 39 checkpoint intervals with no single step above .090 [.063-.142] / .115 [.081-.141] of the 75 -> 25%-of-peak
   drop. End +.021 [-.025, +.070] / +.113 [-.046, +.298], at or below Bayes's level. Lags 1-3: peak .85 / 2.64 at
   1,360, below half at 4,503 / 4,419.
 4 Linear response. Actual - linear (k = 5) stays within +-.08 up to 100 episodes, then changes sign repeatedly: +.24
   [+.12, +.36] at 160 (more than linear), -.82 [-1.31, -.35] at 300 (less), 5 sign changes (between 10-20, 40-60,
   100-120, 200-220, 480-560); above 0 for good from 483 [376-647] (k = 4: 530 [393-670]; k = 3: 579 [439-722]; k = 2:
   633 [545-803]), in the period when multi-bit growth keeps rising while single-flip growth does not (3). Afterwards
   non-monotone: +.75 [+.57, +.93] at 1,040, +.29 [+.21, +.37] at 4,200, +.97 [+.89, +1.04] at the end; Bayes's
   marginals +1.02 (more than linear as well), factorized-state Bayes -2.28 and the ESN -.19 (less).
 5 The bulk (SKL). At 320 episodes the flip's SKL grows over the lags (+.066 [+.046, +.089] / +.268 [+.200, +.342] at
   lags 18-21, from impacts of .015 / .083), where Bayes's shrinks (-.140 / -.769), naive's too (-.076 / -.527). It
   turns negative for good at lags 18-21 at 1,642-1,661 for every k (k = 1: 1,656 [1,536-1,816]; k = 5: 1,642
   [1,461-1,695]), at 12-15 at 1,845-1,919, at 6-9 at 2,743-2,793 and at 1-3 at 6,357 [5,764-7,059] (k = 1) to 7,865
   [6,613-8,651] (k = 5). Orders with separated intervals: 18-21 before 12-15 (k = 2-5; k = 1 overlapping), 12-15 before
   6-9 and 6-9 before 1-3 (every k); lags 18-21 after the goal accuracy is half-way to Bayes (1,394 [1,285-1,489]) for k
   = 1-3, just after it for k = 4 (1,654 [1,490-1,705]: the intervals nearly touch), overlapping for k = 5. End -.125 /
   -.675 (Bayes -.140 / -.769); lags 1-3 -.040 / -.117 (Bayes -.053 / -.174).
 6 Excursions at 160-400 episodes. Single checkpoints depart from the trend at long lags: RMS .222 [.182-.282] / .523
   [.450-.634], largest +.527 / +1.246 at 300, against .096 / .192 at 400-2,400 and .006 / .018 after 2,400; small at
   lags 1-3 (RMS .017 / .041 [.034-.047]). They follow the phase of the checkpoint grid: checkpoints at raw episodes
   divisible by 75 sit above the trend (k = 5: mean +.709 [+.581, +.834], n = 4), the next ones at it (-.022 [-.093,
   +.056]), the ones after below (-.353 [-.479, -.266]). The per-update logs oscillate with 63-68 updates in this window
   (best period: loss 63, update norms of W_hh 66, W_ih 68, read-out 65; AR(1) surrogates reach the data's 55-70-update
   amplitude in a share .006 / .003 / .013 / < .001 of 1,000). Sampled ~3.25 times per cycle by the 20-update
   checkpoint grid, such an oscillation would give a slowly drifting period-3 pattern. The link is not established:
   numbers() runs the test only in this window; a round-3 re-implementation found the same test also passes for 3 of
   the 4 series at 640-880 and for all 4 at 880-1,120 episodes, where the excursions are smaller; and the AR(1) null
   may be anti-conservative for these series. Its cause is not identified. ESN: RMS <= .023.
 7 ESN (no count switch, no fall). Growth rises monotonically (largest decrease between checkpoints -.0055) and
   saturates within 6-9 lags: end +1.391 [+1.364, +1.418] / +3.616 [+3.498, +3.726], half of it at 1,507 / 1,842.
   Its early response is count-coded too (count-preserving 2-bit growth +.034 to +.058 against count-changing
   +.142 to +.560 at 20-100 episodes; k3_vs_k1 +.007 at 20; index up to +.596 [+.588, +.603] at 140), but the code
   fades gradually: the largest one-interval drop within 5 intervals of the peak is .044 nats (Trained RNN .725), the
   index is +.278 at 400 and +.160 [+.112, +.211] at 1,200, first below a tenth of its peak at 3,200 and +.015
   [-.061, +.093] at 9,600; +.066 [-.020, +.160] at 20,000; at the end +.118 [+.023, +.221], resolved above 0 again.
   SKL at lags 18-21 negative for good from 681 [507-792] (k = 1) / 345 [261-438] (k = 5); at lags 1-3 it stays positive
   to the end (+.032 [+.030, +.034] / +.192 [+.179, +.204]; never below 0), where the Trained RNN's (-.040 / -.117) and
   Bayes's (-.053 / -.174) are negative.
 8 Variables and flip steps. At the end the two variables' growth has opposite signs for k = 1 (r0 +.070 [+.016,
   +.129], r1 -.056 [-.104, -.006]); for k = 5 neither differs from 0 (+.172 [-.004, +.366], -.023 [-.201, +.178]).
   Paired r0 - r1: +.126 [+.085, +.169] (k = 1), +.195 [+.051, +.355] (k = 5); +.154 [+.063, +.242] at 880 (k = 1).
   The split is by latent index, not role: goal - contextual variable +.001 [-.028, +.030] (k = 1). Flip step: from the
   trained start state (s = 1) the growth is smaller at 320 (+1.15 [+1.02, +1.30] vs +1.56 [+1.42, +1.70] to +2.01 for
   s = 3-9, k = 1) and larger at the end (+.14 [+.07, +.19] vs -.07 to +.01; k = 5 +1.33 [+1.13, +1.55] vs -.42 to
   +.02).
 9 Replication (width runs; one seed each; passing naive c0 at 835 / 1,256 / 2,070 / 6,271 / 21,072 episodes for N =
   1000 / N = 1000 loss power 1 / 200 / 100 / 50). Every run has the count code early, the power-.5 runs where their
   long-lag growth first rises. At each power-.5 run's index peak, count-preserving vs count-changing 2-bit growth is
   +.117 vs +1.770 (N = 1000, 20 episodes) / +.146 vs +1.606 (N = 200, 60) / +.185 vs +1.382 (N = 100, 80) / +.373 vs
   +1.991 (N = 50, 120), and 3-bit flips that change the count by one act like single flips (k3_vs_k1 +.007 / +.005 /
   +.004 / +.009); index peak +1.778 / +1.501 / +1.220 / +1.686. The loss-power-1 run has the same code at a small
   amplitude in nats: at 20 episodes count-preserving +.009 vs count-changing +.094 (index +.087 [+.086, +.088];
   k3_vs_k1 -.001; immediate impacts .026 vs .154; count-preserving growth <= +.009 at 20-120 episodes), while its
   long-lag growth is still small overall (single flips reach half their peak only at 969). At its own rise (~1k
   episodes) count-preserving flips grow almost as much as count-changing ones (+.388 vs +.447 at 880): no count code
   there, only a small count dependence (index +.051 [+.030, +.077] at 880, +.069 [+.035, +.107] at 1,200).
   The index's collapse is not one step. Its largest one-interval drop, .725 [.651, .795] (60 -> 80) / .710 [.659,
   .760] (60 -> 80) / .634 [.548, .719] (100 -> 120) / .740 [.677, .801] (120 -> 140) nats for N = 1000 / 200 / 100 /
   50, carries .45 [.41-.48] / .51 [.49-.53] / .56 [.47-.68] / .48 [.45-.52] of the drop from the peak to the first
   checkpoint below a tenth of it. The rest falls in the neighbouring intervals for N = 1000 (.16 / .25 before, .14
   after; below a tenth at 100) and N = 200 (.49 in 80 -> 100); for N = 100 the index then hovers at +.17 to +.41
   (120-380) and first falls below a tenth at 4,800; for N = 50 the rest spreads over the next 6 intervals (.08 / .21 /
   .05 / .10 / -.02 / .10; below a tenth at 260). The zig-zag falls in one checkpoint interval, from 81-100% of its
   peak (+.311 / +.233 / +.149 / +.161) to 4-16% of it: 40 -> 60 (+.252 -> +.051) / 60 -> 80 (+.233 -> +.009) / 80
   -> 100 (+.149 -> +.018) / 120 -> 140 (+.136 -> +.015), and stays below half of it from then on (for N = 50 the
   resampled crossings reach 8,878: in at least 5% of resamples it is above half again that late). The switch does not
   keep pace with passing naive (the zig-zag falls at 40-60 episodes for N = 1000 and at 120-140 for N = 50 while c0
   goes 835 -> 21,072). A partial count dependence peaks later in every power-.5 run (index +.917 at 240 / +.590 at 240
   / +.843 at 560 / +.677 at 720) while count-preserving flips grow 2.3-3.5 nats there.
   The fall: single-flip growth falls below half its peak 0.80 [0.70, 0.82] (N = 1000), 0.69 [0.66, 0.73] (power 1),
   0.86 [0.68, 0.98] (N = 200) and 0.59 [0.47, 0.69] (N = 100) decades after the run passes naive: roughly in step,
   though N = 100's interval separates from N = 1000's; N = 50 is still above half at the end (0.28 decades after its
   c0). The 5-bit fall does not keep pace: 0.71 [0.65, 0.74] (N = 1000) / 0.59 [0.56, 0.61] (power 1) / 1.19 [1.06,
   open] (N = 200; 82% of resamples cross) decades, N = 200's interval separated from N = 1000's; N = 100 has not
   fallen by the end of its training, +0.80 decades after its c0 (end +3.56 against a peak of 6.06); N = 50 (training
   ends +0.28 decades after its c0) is untested. Rise (half the peak, k = 1): before 20 / 969 / 89 / 109 / 152
   episodes.
NULLS AND FAILED HYPOTHESES
  (a) A phase transition (boundary) in (lag, k): none at any checkpoint; the 20-episode map's structure is the count
      code's pairing of k (1 with 2, 3 with 4), not a boundary (figure f-h).
  (b) A critical k: none; the count code depends on the count change, not on a threshold k (1).
  (c) RETRACTED from the previous record: "the increment changes by at most .073 at any stage", "the per-k heat maps are
      scaled copies", "the only abrupt feature is single-checkpoint excursions" and "no abrupt sustained change through
      training": all fail at 20-60 episodes (1, 2).
  (d) RETRACTED: "every k rises, peaks and fades together" (single flips level off while multi-bit flips rise, 3) and
      "the k-flip growth moves smoothly from below to above the sum of single flips" (5 sign changes before ~480, 4).
  (e) RETRACTED: "the fastest change measured is the SKL crossover": its 75 -> 25%-of-peak window covered only the top
      of a decline that continues far below 0 (a window artifact); the SKL event is now reported as zero crossings.
  (f) RETRACTED orders of overlapping intervals: SKL zero (lags 18-21) after the accuracy half-way point for k = 5, and
      before lags 12-15 for k = 1 (5). The old "growth 0 = the bulk effect stops growing" label: growth 0 means back at
      the immediate size.
  (g) The count code inherited from initialization: no (1). The count switch keeping pace with passing naive: no (9).
  (h) The chaos sweep's panel-d growth as a pure tail (d_H-only) effect: refuted early (the SKL grows too, 5), true
      late.
  (i) Failed methods of the previous round (kept for the record): logistic flank fits broke on the spiky curves; step
      shares before the trend counted reverting spikes; a quadratic fit in k cannot follow d_H's first bit.
  (j) The 5-bit fall keeping pace with passing naive: no. On each run's own clock N = 200 falls 1.19 [1.06, open]
      decades after its c0 against N = 1000's 0.71 [0.65, 0.74]; N = 100 has not fallen by the end of its training
      (+0.80 decades after its c0); N = 50 is untested (training ends +0.28 decades after its c0). Only single flips
      fall roughly in step (9).
  (k) RETRACTED from the previous round: "the count code collapses in one large step right after its peak" (about half
      of the index's drop falls in one interval, 9); "the loss-power-1 run shows only a trace" (the same code at a small
      amplitude in nats, 9); "the fall keeps pace with passing naive" stated without restriction (single flips only, 9);
      switch times to the episode ("49 [48-49]" etc.: points interpolated inside one checkpoint interval, 1 and 9); the
      order of the k = 1 and k = 5 peak times (their locations were not resampled, 3); "the count-code index switches
      off" (it also measures a partial count dependence, DEFINITIONS).
CAVEATS
  One seed per width and one probe world; intervals resample environments, not checkpoints or seeds. Flip steps 1-9
  only; s = 1 starts from the trained start state. The count code is seen at the first two trained checkpoints (20,
  40): its onset (0 -> 20 episodes) is unresolved. d_H is set by near-impossible values, SKL by the bulk; both shown.
  fp32 replays of networks trained under fp16 autocast (td_tt_intervene). The 63-68-update oscillation was not traced
  to the episode stream and is not specific to the excursion window (6); whether the excursions sit at near-marginal
  state dynamics is untested (needs the corrected Lyapunov exponents of td_tt_chaos). The plateau windows (200-360,
  640-1,040) are a choice around the single-flip plateau and the multi-bit peak. The count-code index reads as a count
  code only where the count-preserving growth is ~0 (DEFINITIONS).
CHECKS (checks(); flips/checks.json)
  The probe subset = the stored chaos sweep's rows (24 of 24 files checked). Flips vs explicitly flipped observation
  arrays through td_tt_intervene.forward (10 probes x 5 flip steps x one flip set per k, CPU; Trained RNN at 20, 880,
  1,680, 39,996, ESN at 39,996): max |d log q| <= 1.8e-5, |d d_H| <= 1.9e-5, |d SKL| <= 1.1e-5; intact log beliefs vs
  the engine <= 1.5e-5. The k = 1 flips vs the stored chaos sweep's chi (s = 1, 3, 5, lags 0-21, both variables): all
  228 checkpoints of the Trained RNN and the ESN within one fp16 ulp (max |d| .031). Exact observers vs brute force from
  explicitly flipped observation arrays (20 probes, all 31 flip sets, every s): <= 9.2e-14; Bayes's and naive's intact
  marginals vs probes.npz <= 6.6e-6. Naive's d_H constant in lag (0) and its linear response exact (0). In every
  harvested file (684: six runs x 114 checkpoints): SKL <= d_H / 2 and >= 0, the stored float16 per-probe values vs the
  float64 environment sums relative <= 4.2e-4, the k = 1 linear response = the flip itself (0). Count classes: the class
  of every flip set (k - 2j) equals the net change of ones of the explicitly flipped observations (0 mismatches in
  155,000); the classes' C(5, k) means equal the stored harvest in all 684 files (max |d| 1.5e-5 / 6.6e-6 for d_H /
  SKL), with 3,100 units per environment; the exact observers' classes reproduce refs.npz (<= 9.7e-6, its float32
  storage). The Trained RNN's accuracy crossings from td_tt_gap 'jumps' equal td_tt_replication's (resampled crossings
  within 2.5e-11). pyflakes clean; the figure viewed whole and in native-resolution crops.
COST: count classes on GPU 1 (John 11-02: both GPUs are available) 0.9-5 s per checkpoint (684 checkpoints, ~26 min,
  peak 833 MB); the per-checkpoint harvest of the previous round: GPU 0 12-30 s, GPU 1 ~1-5 s per checkpoint (the width
  runs by a helper running this module's harvest_one); references 48 s, their classes 31 s (CPU); analysis + numbers
  ~1 min; figure ~40 s; checks ~6 min.

Run from the repository root (one GPU, 2 CPU threads):
    CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
        /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_flips.py harvest classes
    (then on the CPU: refs refs_classes checks numbers figure export)"""
import os
import sys
import json
import time
from math import comb
import numpy as np
import torch
from scipy.special import logsumexp

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_intervene as ti                                                                  # noqa: E402

T, R, NV, NO = ti.T, ti.R, ti.NV, ti.NO
OUT = os.path.join(gap.TT, "flips")
CHAOS = os.path.join(gap.TT, "chaos")
SUB_SEED, PER_ENV_SUB = 20261104, 20            # td_tt_chaos.subset(): 20 probes per environment
FLIP_S = (1, 3, 5, 7, 9)                        # flip steps; every one allows lag 21 (s + 21 <= 30)
NLAG = 22                                       # lags 0..21
KS = (1, 2, 3, 4, 5)
MASKS = np.arange(1, 1 << NO)                   # the 31 non-empty subsets of the 5 observations (bit o = observation o)
MBITS = ((MASKS[:, None] >> np.arange(NO)) & 1).astype(np.float64)                  # (31, 5)
KOF = MBITS.sum(1).astype(np.int64)                                                 # bits flipped per subset
SINGLE = np.array([int(np.flatnonzero(MASKS == (1 << o))[0]) for o in range(NO)])  # the k = 1 subsets
KMAT = np.stack([(KOF == k) / comb(NO, k) for k in KS])                             # (5, 31): mean over C(5, k)
LAG_BINS = ((1, 3), (6, 9), (12, 15), (18, 21))                                     # John's bins (inclusive)
NB = 1 + len(LAG_BINS)                                                              # lag 0 and the 4 lag bins
NJ = NO + 1                                                    # j = 0..5: flipped bits that were 1 (net change k - 2j)
NCLS = len(KS) * NJ                                                                 # count classes (k, j)
PAIRS = ("k2_cc", "k4_cc", "k3_cc", "k3_vs_k1")                                     # paired count-code contrasts
RUNS = ("trained", "echo")
WIDTH_RUNS = ("trained_p1", "trained_N200", "trained_N100", "trained_N50")         # replication (same seed and world)
SWEEP_S = (1, 3, 5, 10, 15, 20, 25)                                                 # td_tt_chaos.FLIP_S (stored sweep)
PY = "/home/johns/anaconda3/envs/eureka/bin/python"
MEAS = ("dh", "skl")


# ═══════════════════════════════════════════════════════════════════
# Probe subset and the exact observers (CPU, float64); copied from td_tt_chaos (being fixed elsewhere)
# ═══════════════════════════════════════════════════════════════════

def subset():
    """the fixed probe subset of td_tt_chaos.subset(): PER_ENV_SUB rows per environment (seed 20261104, sorted)"""
    P = ti.probes()
    rng = np.random.default_rng(SUB_SEED)
    rows = [rng.choice(np.flatnonzero(P["env"] == e), PER_ENV_SUB, replace=False) for e in range(gap.N_ENV)]
    return np.sort(np.concatenate(rows))


def tables():
    """L (50, 5, R, R) = p(x_o = 1 | r0, r1) per environment"""
    with np.load(os.path.join(ti.AN, "probes.npz")) as Z:
        return Z["L"].astype(np.float64)


def osc(a, axis=-1):
    return a.max(axis) - a.min(axis)


def step_loglik(obs, L):
    """(n, T, R, R) log p(x_t | r0, r1) for obs (n, T, 5) and per-episode tables L (n, 5, R, R)"""
    o = obs.astype(np.float64)
    return np.einsum("nti,nijk->ntjk", o, np.log(L)) + np.einsum("nti,nijk->ntjk", 1 - o, np.log1p(-L))


def marg_log(u):
    """unnormalised log marginals (..., 2, R) of joint log beliefs u (..., R, R)"""
    return np.stack([logsumexp(u, -1), logsumexp(u, -2)], -2)


def _norm(lm):
    return lm - logsumexp(lm, -1, keepdims=True)


def adf_run(ll, la0=None):
    """factorized-state Bayes (td_analysis's ADF, td_tt_chaos.adf_run) on per-step log-likelihoods ll (n, S, R, R) from
    log marginals la0 (n, 2, R) (uniform if None): normalised log marginals (n, S, 2, R) after each step"""
    n, S = ll.shape[:2]
    la = np.full((n, 2, R), -np.log(R)) if la0 is None else la0
    out = np.zeros((n, S, 2, R))
    for t in range(S):
        j = la[:, 0, :, None] + la[:, 1, None, :] + ll[:, t]
        j = j - logsumexp(j.reshape(n, -1), -1)[:, None, None]
        la = marg_log(j)
        out[:, t] = la
    return out


def measures_np(lp, D):
    """lp (..., R) the intact normalised log belief, D (..., R) the flip's change of the log belief (any constant):
    d_H = osc(D) (Hilbert's projective metric = the range of the log-ratio) and the symmetric KL
    SKL = Sigma_r (p_r - q_r) (log p_r - log q_r) / 2 with q = the flipped belief"""
    lq = _norm(lp + D)
    return osc(D), ((np.exp(lp) - np.exp(lq)) * (lp - lq)).sum(-1) / 2


def kmean_np(a, axis):
    """mean over the C(5, k) subsets of each k: the subset axis (31) -> k (5)"""
    return np.moveaxis(np.tensordot(KMAT, np.moveaxis(a, axis, 0), (1, 0)), 0, axis)


def exact_flips(rows, chunk=25, sup=True, per_set=False):
    """the exact observers' responses to every flip set (float64): {'<obs>_<meas>': (n, len(FLIP_S), 5 k, NLAG, NV)}
    for obs in bayes (the joint posterior's marginals; a k-bit flip adds the sum of the k bits' log-likelihood-ratio
    changes to the joint log posterior), naive (a translation of each log marginal) and adf (factorized-state Bayes
    rerun from its intact marginals at s - 1 with the flipped step), meas in dh / skl (means over the C(5, k) subsets);
    with sup: '<obs>_<meas>_sup' = the same measures of the superposition of the 5 single flips' log-belief changes.
    per_set: returns (that dict, {'<obs>_<meas>': (n, len(FLIP_S), 31, NB, NV)} chi at lag 0 and its lag-bin means per
    flip set)"""
    P = ti.probes()
    L = tables()
    obs, env = P["obs"][rows], P["env"][rows]
    n, nS = len(rows), len(FLIP_S)
    keys = [f"{o}_{m}" for o in ("bayes", "naive", "adf") for m in MEAS]
    keys += [k + "_sup" for k in keys] if sup else []
    out = {k: np.zeros((n, nS, len(KS), NLAG, NV)) for k in keys}
    sets = {f"{o}_{m}": np.zeros((n, nS, len(MASKS), NB, NV)) for o in ("bayes", "naive", "adf") for m in MEAS} \
        if per_set else None
    for a0 in range(0, n, chunk):
        sl = slice(a0, min(n, a0 + chunk))
        b = sl.stop - sl.start
        Le = L[env[sl]]
        ll = step_loglik(obs[sl], Le)
        u = np.cumsum(ll, 1)
        lmB = _norm(marg_log(u))                                                   # (b, T, 2, R)
        logit = np.log(Le) - np.log1p(-Le)                                         # (b, 5, R, R)
        Lb = np.stack([Le.mean(3), Le.mean(2)], 2)                                 # naive tables (b, 5, 2, R)
        o_ = obs[sl].astype(np.float64)
        cn = _norm(np.cumsum(np.einsum("nti,nikr->ntkr", o_, np.log(Lb)) +
                             np.einsum("nti,nikr->ntkr", 1 - o_, np.log1p(-Lb)), 1))
        logit_n = np.log(Lb) - np.log1p(-Lb)
        la = adf_run(ll)
        for a, s in enumerate(FLIP_S):
            win = slice(s - 1, s - 1 + NLAG)
            sign = 1. - 2. * obs[sl, s - 1].astype(np.float64)                     # (b, 5)
            dJ = np.einsum("mo,bo,borc->bmrc", MBITS, sign, logit)                 # (b, 31, R, R)
            res = {}
            lm2 = _norm(marg_log(u[:, None, win] + dJ[:, :, None]))               # (b, 31, NLAG, 2, R)
            res["bayes"] = (lmB[:, None, win], lm2 - lmB[:, None, win])
            dN = np.einsum("mo,bo,bovr->bmvr", MBITS, sign, logit_n)               # (b, 31, 2, R)
            res["naive"] = (cn[:, None, win], np.broadcast_to(dN[:, :, None], (b, len(MASKS), NLAG, NV, R)))
            ll2 = np.repeat(ll[:, None, win], len(MASKS), 1)                       # (b, 31, NLAG, R, R)
            ll2[:, :, 0] += dJ
            la0 = None if s == 1 else np.repeat(la[:, s - 2][:, None], len(MASKS), 1).reshape(-1, 2, R)
            la2 = adf_run(ll2.reshape(-1, NLAG, R, R), la0).reshape(b, len(MASKS), NLAG, NV, R)
            res["adf"] = (la[:, None, win], la2 - la[:, None, win])
            for nm, (lp, D) in res.items():
                dh, sk = measures_np(lp, D)
                out[f"{nm}_dh"][sl, a], out[f"{nm}_skl"][sl, a] = kmean_np(dh, 1), kmean_np(sk, 1)
                if per_set:
                    sets[f"{nm}_dh"][sl, a], sets[f"{nm}_skl"][sl, a] = lag_bin_t(dh, 2), lag_bin_t(sk, 2)
                if sup:
                    Ds = np.einsum("mo,bojvr->bmjvr", MBITS, D[:, SINGLE])
                    dh, sk = measures_np(lp, Ds)
                    out[f"{nm}_dh_sup"][sl, a], out[f"{nm}_skl_sup"][sl, a] = kmean_np(dh, 1), kmean_np(sk, 1)
    return (out, sets) if per_set else out


def refs_build():
    """the exact observers on the probe subset -> OUT/refs.npz (float32)"""
    t0 = time.time()
    rows = subset()
    E = exact_flips(rows)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "refs.npz"), rows=rows, **{k: v.astype(np.float32) for k, v in E.items()})
    print(f"refs: {len(rows)} probes, {time.time() - t0:.0f} s", flush=True)


# ═══════════════════════════════════════════════════════════════════
# Networks (GPU 0): the intact run, then every flip set rerun from the intact state at s - 1
# ═══════════════════════════════════════════════════════════════════

def lstm_step(xin, h, c, Whh):
    """td_tt_chaos.lstm_step (td_analysis's cell)"""
    pre = xin + h @ Whh.T
    i, f, g, o = pre.chunk(4, -1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    c2 = f * c + i * g
    tc = c2.tanh()
    return o * tc, c2, (i, f, g, o, tc)


def measures_t(C, D):
    """torch version of measures_np from cumulative logits: C (..., R) the intact cumulative read-out (float64), D the
    flip's cumulative change"""
    lp = torch.log_softmax(C, -1)
    lq = torch.log_softmax(C + D, -1)
    return D.amax(-1) - D.amin(-1), ((lp.exp() - lq.exp()) * (lp - lq)).sum(-1) / 2


def lag_bin_t(v, axis=2):
    """chi at lag 0 and the mean of chi over each of John's lag bins: the lag axis (NLAG) -> (NB) (torch or numpy)"""
    pick = (lambda a, b: v.narrow(axis, a, b - a)) if torch.is_tensor(v) else \
        (lambda a, b: np.take(v, np.arange(a, b), axis))
    stack = torch.stack if torch.is_tensor(v) else np.stack
    return stack([pick(0, 1).mean(axis)] + [pick(a, b + 1).mean(axis) for a, b in LAG_BINS], axis)


@torch.no_grad()
def net_flips(run, raw, rows, dev, chunk=250, keep=False, per_set=False):
    """the network's responses to every flip set: {'dh', 'skl', 'dh_sup', 'skl_sup'} (n, len(FLIP_S), 5 k, NLAG, NV)
    float32 (means over the C(5, k) subsets; sup = the superposition of the 5 single flips' log-belief changes). The
    flip loop of td_tt_chaos.network: from the intact state at s - 1, observation pattern code ^ mask at s, intact
    observations after; increments d_t = read-out(flipped) - read-out(intact) in float64, D = their cumulative sum (the
    change of the cumulative read-out = of the log belief up to a constant). keep: also the intact log beliefs (n, T,
    2, R) and the subset-resolved D for checks. per_set: also 'set_dh', 'set_skl' (n, len(FLIP_S), 31 flip sets, NB, NV)
    float32 = chi at lag 0 and its mean over each lag bin, per flip set (before the C(5, k) mean), and 'code' (n, T) the
    observation patterns"""
    w = ti.load(run, raw)
    p = {k: v.to(dev, torch.float32) for k, v in w.items()}
    H = p["LSTM.weight_hh_l0"].shape[1]
    Whh, Wro, bro = p["LSTM.weight_hh_l0"], p["classifier_readout.weight"], p["classifier_readout.bias"]
    P = ti.probes()
    table = ti._table(p, P["Z"], dev)
    obs, env = P["obs"][rows], P["env"][rows]
    code = torch.as_tensor((obs * (1 << np.arange(NO))).sum(-1).astype(np.int64), device=dev)
    envt = torch.as_tensor(env.astype(np.int64), device=dev)
    n, nS, nm = len(rows), len(FLIP_S), len(MASKS)
    KM = torch.as_tensor(KMAT, device=dev)
    MB = torch.as_tensor(MBITS, device=dev)
    sgl = torch.as_tensor(SINGLE, device=dev)
    flip = torch.as_tensor(MASKS, device=dev)
    out = {k: np.zeros((n, nS, len(KS), NLAG, NV), np.float32) for k in ("dh", "skl", "dh_sup", "skl_sup")}
    if keep:
        out["logq"] = np.zeros((n, T, NV, R), np.float32)
        out["D"] = np.zeros((n, nS, nm, NLAG, NV, R), np.float32)
    if per_set:
        out.update(set_dh=np.zeros((n, nS, nm, NB, NV), np.float32), set_skl=np.zeros((n, nS, nm, NB, NV), np.float32),
                   code=code.cpu().numpy())
    smax = max(FLIP_S)
    for b0 in range(0, n, chunk):
        sl = slice(b0, min(n, b0 + chunk))
        b = sl.stop - sl.start
        h = p["STM"].reshape(1, H).expand(b, H).contiguous()
        c = p["LTM"].reshape(1, H).expand(b, H).contiguous()
        hs, cs, lg = [h], [c], []
        for t in range(T):
            h, c, _ = lstm_step(table[envt[sl], code[sl, t]], h, c, Whh)
            if t < smax - 1:
                hs.append(h)
                cs.append(c)
            lg.append(torch.nn.functional.linear(h, Wro, bro))
        lg = torch.stack(lg, 1).double()                                          # (b, T, 20)
        C = lg.cumsum(1).reshape(b, T, NV, R)
        if keep:
            out["logq"][sl] = torch.log_softmax(C, -1).float().cpu().numpy()
        for a, s in enumerate(FLIP_S):
            hb = hs[s - 1].repeat_interleave(nm, 0)
            cb = cs[s - 1].repeat_interleave(nm, 0)
            eb = envt[sl].repeat_interleave(nm)
            d = torch.zeros(b * nm, NLAG, NV * R, device=dev, dtype=torch.float64)
            for j in range(NLAG):
                t = s - 1 + j
                cd = code[sl, t].repeat_interleave(nm)
                if j == 0:
                    cd = cd ^ flip.repeat(b)
                hb, cb, _ = lstm_step(table[eb, cd], hb, cb, Whh)
                d[:, j] = torch.nn.functional.linear(hb, Wro, bro).double() - lg[:, t].repeat_interleave(nm, 0)
            D = d.cumsum(1).reshape(b, nm, NLAG, NV, R)
            Ci = C[:, None, s - 1:s - 1 + NLAG]                                    # (b, 1, NLAG, NV, R)
            dh, sk = measures_t(Ci, D)
            Ds = torch.einsum("mo,bojvr->bmjvr", MB, D[:, sgl])
            dhs, sks = measures_t(Ci, Ds)
            for key, v in (("dh", dh), ("skl", sk), ("dh_sup", dhs), ("skl_sup", sks)):
                out[key][sl, a] = torch.einsum("km,bmjv->bkjv", KM, v).float().cpu().numpy()
            if keep:
                out["D"][sl, a] = D.float().cpu().numpy()
            if per_set:
                out["set_dh"][sl, a] = lag_bin_t(dh).float().cpu().numpy()
                out["set_skl"][sl, a] = lag_bin_t(sk).float().cpu().numpy()
    return out


def ck_file(run, raw):
    return os.path.join(OUT, run, f"ck_{raw:05d}.npz")


def harvest_list(thin_echo=False):
    """(run, raw) jobs: every checkpoint of both runs, every third first (a coarse pass over training), then the rest;
    thin_echo keeps every second ESN checkpoint (and its last)"""
    jobs = []
    for run in RUNS:
        eps = ti.checkpoints(run)
        if run == "echo" and thin_echo:
            eps = sorted(set(eps[::2]) | {eps[-1]})
        for i, raw in enumerate(eps):
            jobs.append((0 if i % 3 == 0 else 1, 0 if run == "trained" else 1, run, raw))
    return [(r, e) for _, _, r, e in sorted(jobs)]


def harvest_one(run, raw, rows, dev):
    """one checkpoint -> OUT/<run>/ck_<raw>.npz: dh, skl (n, 5 s, 5 k, NLAG, NV) float16 per probe and variable (means
    over the C(5, k) subsets); dh_sup, skl_sup per-environment sums (50, 5, 5, NLAG, NV) float64; the k = 1 check
    against the stored chaos sweep (s = 1, 3, 5): max |d| and the share within one fp16 ulp"""
    t0 = time.time()
    N = net_flips(run, raw, rows, dev)
    env = ti.probes()["env"][rows]
    o = dict(rows=rows, raw=raw, x=float(gap.train_x(raw)), dh=N["dh"].astype(np.float16),
             skl=N["skl"].astype(np.float16))
    for k in ("dh_sup", "skl_sup", "dh", "skl"):
        o[f"{k}_envsum"] = np.stack([N[k][env == e].astype(np.float64).sum(0) for e in range(gap.N_ENV)])
    f = os.path.join(CHAOS, "sweep", f"{run}_{raw:05d}.npz")
    if os.path.exists(f):
        with np.load(f) as Z:
            assert np.array_equal(Z["rows"], rows)
            ref = Z["chi"].astype(np.float64)                                      # (n, 7, T, 2) fp16
        mine, theirs = [], []
        for a, s in enumerate(FLIP_S):
            if s in SWEEP_S[:3]:
                mine.append(N["dh"][:, a, 0])                                      # (n, NLAG, 2)
                theirs.append(ref[:, SWEEP_S.index(s), s - 1:s - 1 + NLAG])
        mine, theirs = np.stack(mine), np.stack(theirs)
        ulp = np.spacing(np.abs(theirs).astype(np.float16)).astype(np.float64)
        o.update(sweep_max_abs=float(np.abs(mine - theirs).max()),
                 sweep_within_ulp=float((np.abs(mine.astype(np.float16).astype(np.float64) - theirs) <= ulp).mean()))
    o["seconds"] = time.time() - t0
    path = ck_file(run, raw)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **o)
    os.replace(tmp, path)
    return o


def harvest(thin_echo=False):
    """every checkpoint of both runs (resumable: existing files are skipped; stops before the next checkpoint when
    OUT/STOP exists)"""
    dev = ti._device()
    assert dev.type == "cuda", "run on one GPU (CUDA_VISIBLE_DEVICES=1 or 0)"
    rows = subset()
    jobs = harvest_list(thin_echo)
    todo = [j for j in jobs if not os.path.exists(ck_file(*j))]
    print(f"{len(jobs)} checkpoints, {len(todo)} to do", flush=True)
    for run, raw in todo:
        if os.path.exists(os.path.join(OUT, "STOP")):
            print("STOP file: stopping", flush=True)
            return
        if os.path.exists(ck_file(run, raw)):                                     # done meanwhile (another worker)
            continue
        for attempt in range(30):
            try:
                o = harvest_one(run, raw, rows, dev)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM at {run} {raw}; waiting 60 s", flush=True)
                time.sleep(60)
        else:
            raise RuntimeError("GPU memory unavailable for 30 min")
        print(f"{run} {raw} ({o['x']:,.0f}): {o['seconds']:.1f} s; sweep k=1 max |d| {o.get('sweep_max_abs', np.nan):.4f}"
              f", within 1 fp16 ulp {o.get('sweep_within_ulp', np.nan):.5f}; peak GPU "
              f"{torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB", flush=True)


# ═══════════════════════════════════════════════════════════════════
# Count classes: every flip set kept apart (before the C(5, k) mean) -> the early count code
# ═══════════════════════════════════════════════════════════════════

CLS_RUNS = RUNS + WIDTH_RUNS
PAIR_DEF = {"k2_cc": ((2, (0, 2)), (2, (1,))),            # (k, j's) of the first class group minus the second's
            "k4_cc": ((4, (0, 1, 3, 4)), (4, (2,))),
            "k3_cc": ((3, (0, 3)), (3, (1, 2))),
            "k3_vs_k1": ((3, (1, 2)), (1, (0, 1)))}


def popcount(a):
    a = np.asarray(a, np.int64)
    return sum((a >> o) & 1 for o in range(NO))


def count_classes(V, code_s, env):
    """per-flip-set values -> per-environment sums by count class, and the paired count-code contrasts.
    V (n, nS, 31, NB): per probe, flip step and flip set, the mean over the 2 variables of [the immediate impact
    chi(s, s), the growth chi(s + lag, s) - chi(s, s) averaged over each lag bin]; code_s (n, nS) the observation
    pattern at each flip step (bit o = observation o); env (n,).
    Count class (k, j): k bits flipped, j of them were 1, so the number of ones in the observation changes by k - 2j.
    -> cls_sum (50, NCLS, NB), cls_n (50, NCLS) over (probe, flip step, flip set) units; pair_sum (50, 4, NB), pair_n
    (50, 4) over (probe, flip step) units of PAIRS, each the difference of two class-group means of the SAME probe and
    flip step (a unit counts only where both groups exist):
      k2_cc     2 bits: count-changing (net +-2) minus count-preserving (net 0; one 0 -> 1 and one 1 -> 0)
      k4_cc     4 bits: count-changing (net +-2, +-4) minus count-preserving (net 0)
      k3_cc     3 bits: net +-3 minus net +-1
      k3_vs_k1  3 bits with net +-1 minus 1 bit (net +-1): more bits at the same count change"""
    j = popcount(np.asarray(code_s)[:, :, None] & MASKS[None, None, :])           # (n, nS, 31)
    cls = (KOF - 1)[None, None, :] * NJ + j
    oh = (cls[..., None] == np.arange(NCLS)).astype(np.float64)                  # (n, nS, 31, NCLS)
    CS = np.einsum("psmc,psmb->pscb", oh, V)                                     # per (probe, s) class sums
    CN = oh.sum(2)                                                               # (n, nS, NCLS)
    E = np.eye(gap.N_ENV)[np.asarray(env)]                                       # (n, 50)
    out = dict(cls_sum=np.einsum("pe,pscb->ecb", E, CS), cls_n=np.einsum("pe,psc->ec", E, CN))
    ps, pn = np.zeros((gap.N_ENV, len(PAIRS), NB)), np.zeros((gap.N_ENV, len(PAIRS)))
    for q, nm in enumerate(PAIRS):
        grp = []
        for k, js in PAIR_DEF[nm]:
            c = [(k - 1) * NJ + jj for jj in js]
            grp.append((CS[:, :, c].sum(2), CN[:, :, c].sum(2)))
        (sa, na), (sb, nb) = grp
        ok = (na > 0) & (nb > 0)                                                 # (n, nS)
        d = np.where(ok[..., None], sa / np.maximum(na, 1)[..., None] - sb / np.maximum(nb, 1)[..., None], 0.)
        ps[:, q] = np.einsum("pe,psb->eb", E, d)
        pn[:, q] = np.einsum("pe,ps->e", E, ok.astype(np.float64))
    out.update(pair_sum=ps, pair_n=pn)
    return out


def set_values(S):
    """per-flip-set chi (n, nS, 31, NB, NV) (lag 0, lag-bin means) -> the variable mean with index 0 the immediate
    impact and 1..4 the growth (lag-bin mean minus lag 0)"""
    V = np.asarray(S, np.float64).mean(-1)
    V[..., 1:] -= V[..., :1]
    return V


def cls_file(run, raw):
    return os.path.join(OUT, run, f"cls_{raw:05d}.npz")


def classes_one(run, raw, rows, dev):
    """one checkpoint -> OUT/<run>/cls_<raw>.npz: for m in dh / skl, cls_sum_<m>, pair_sum_<m> (count_classes), cls_n,
    pair_n; check_<m> = max |difference| between the C(5, k) means of these per-set values and the stored harvest
    (ck_<raw>.npz, lag bins, per-environment sums)"""
    t0 = time.time()
    N = net_flips(run, raw, rows, dev, per_set=True)
    env = ti.probes()["env"][rows]
    code_s = N["code"][:, [s - 1 for s in FLIP_S]]
    o = dict(raw=raw, x=float(gap.train_x(raw)))
    E = np.eye(gap.N_ENV)[env]
    for m in MEAS:
        V = set_values(N[f"set_{m}"])
        c = count_classes(V, code_s, env)
        o.update({f"cls_sum_{m}": c["cls_sum"], f"pair_sum_{m}": c["pair_sum"], "cls_n": c["cls_n"],
                  "pair_n": c["pair_n"]})
        if os.path.exists(ck_file(run, raw)):
            with np.load(ck_file(run, raw)) as Z:
                S = Z[f"{m}_envsum"]                                               # (50, nS, 5, NLAG, NV)
            ref = np.concatenate([impact_env(S)[..., None], lag_bins(growth_env(S))], -1) / NV     # (50, 5, NB)
            mine = np.einsum("pe,pkb->ekb", E, kmean_np(V, 2).sum(1))
            o[f"check_{m}"] = float(np.abs(mine - ref).max())
    o["seconds"] = time.time() - t0
    path = cls_file(run, raw)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **o)
    os.replace(tmp, path)
    return o


def harvest_classes(runs=CLS_RUNS):
    """count classes at every checkpoint of the runs (resumable; stops before the next checkpoint when OUT/STOP_CLS
    exists): the Trained RNN up to 2,400 training episodes first, then every run up to 800, then the rest"""
    dev = ti._device()
    assert dev.type == "cuda", "run on one GPU (CUDA_VISIBLE_DEVICES=1 or 0)"
    rows = subset()
    jobs = []
    for r_, run in enumerate(runs):
        for raw in ti.checkpoints(run):
            pri = 0 if (run == "trained" and raw <= 3000) else (1 if raw <= 1000 else (2 if run == "trained" else 3))
            jobs.append((pri, raw, r_, run))
    jobs = [(run, raw) for _, raw, _, run in sorted(jobs)]
    todo = [j for j in jobs if not os.path.exists(cls_file(*j))]
    print(f"{len(jobs)} checkpoints, {len(todo)} to do", flush=True)
    for run, raw in todo:
        if os.path.exists(os.path.join(OUT, "STOP_CLS")):
            print("STOP_CLS file: stopping", flush=True)
            return
        if os.path.exists(cls_file(run, raw)):
            continue
        o = classes_one(run, raw, rows, dev)
        print(f"{time.strftime('%H:%M:%S')} {run} {raw} ({o['x']:,.0f}): {o['seconds']:.1f} s; check vs harvest "
              f"dh {o.get('check_dh', np.nan):.2e} skl {o.get('check_skl', np.nan):.2e}; peak GPU "
              f"{torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MB", flush=True)
    print("done", flush=True)


def refs_classes():
    """the exact observers' count classes on the probe subset -> OUT/refs_classes.npz: '<obs>_cls_sum_<m>',
    '<obs>_pair_sum_<m>' (count_classes), cls_n, pair_n"""
    t0 = time.time()
    rows = subset()
    P = ti.probes()
    _, Sset = exact_flips(rows, sup=False, per_set=True)
    code_s = (P["obs"][rows][:, [s - 1 for s in FLIP_S]] * (1 << np.arange(NO))).sum(-1)
    env = P["env"][rows]
    out = {}
    for k, S in Sset.items():                                                    # k = '<obs>_<meas>'
        nm, m = k.split("_")
        c = count_classes(set_values(S), code_s, env)
        out.update({f"{nm}_cls_sum_{m}": c["cls_sum"], f"{nm}_pair_sum_{m}": c["pair_sum"], "cls_n": c["cls_n"],
                    "pair_n": c["pair_n"]})
    np.savez_compressed(os.path.join(OUT, "refs_classes.npz"), **out)
    print(f"refs_classes: {time.time() - t0:.0f} s", flush=True)




# ═══════════════════════════════════════════════════════════════════
# Statistics: per-environment sums -> pooled means (ratios of count-weighted sums), 5-95% over the shared resamples
# ═══════════════════════════════════════════════════════════════════

ALL = ("dh", "skl", "dh_sup", "skl_sup")
UNITS = PER_ENV_SUB * len(FLIP_S) * NV          # per environment, (k, lag): 20 probes x 5 flip steps x 2 variables


def load_run(run):
    """per-environment sums of the run's harvested checkpoints in checkpoint order: x (K,) training episodes, raw (K,),
    and for m in ALL: (K, 50, len(FLIP_S), 5 k, NLAG, NV) (sums over the environment's 20 probes; each value already
    the mean over the C(5, k) subsets)"""
    eps = [e for e in ti.checkpoints(run) if os.path.exists(ck_file(run, e))]
    out = {m: [] for m in ALL}
    for e in eps:
        with np.load(ck_file(run, e)) as Z:
            for m in ALL:
                out[m].append(Z[f"{m}_envsum"])
    out = {m: np.stack(v) for m, v in out.items()} if eps else {}
    out.update(x=gap.train_x(np.array(eps)), raw=np.array(eps), n_total=len(ti.checkpoints(run)))
    return out


def refs_env():
    """the exact observers as per-environment sums: {'<obs>_<meas>[_sup]': (50, len(FLIP_S), 5 k, NLAG, NV)}"""
    with np.load(os.path.join(OUT, "refs.npz")) as Z:
        rows = Z["rows"]
        env = ti.probes()["env"][rows]
        return {k: np.stack([Z[k][env == e].astype(np.float64).sum(0) for e in range(gap.N_ENV)])
                for k in Z.files if k != "rows"}


def pooled(Se, units=UNITS, C=None):
    """per-environment sums Se (..., 50, *rest) (env axis -1 - len(rest) is given by putting env FIRST: Se (50, ...))
    -> pooled mean (...) and resampled (N_BOOT, ...): ratios of count-weighted sums (every environment holds the same
    number of units)"""
    C = gap.boot_counts() if C is None else C
    Se = np.asarray(Se, np.float64)
    val = Se.sum(0) / (units * gap.N_ENV)
    bt = np.tensordot(C, Se, (1, 0)) / (units * C.sum(1)).reshape((-1,) + (1,) * (Se.ndim - 1))
    return val, bt


def growth_env(S, var=None):
    """per-environment sums of the paired growth chi(s + lag, s) - chi(s, s) (its own flip set and probe) over the flip
    steps and the variables (var = 0 / 1: that variable only): S (..., 50, nS, 5, NLAG, NV) -> (..., 50, 5, NLAG)"""
    G = S - S[..., :1, :]
    return G.sum(-4).sum(-1) if var is None else G[..., var].sum(-3)


def impact_env(S, var=None):
    """per-environment sums of the immediate impact chi(s, s): (..., 50, 5)"""
    I0 = S[..., 0, :]
    return I0.sum(-3).sum(-1) if var is None else I0[..., var].sum(-2)


def lag_bins(G, axis=-1):
    """the mean over the lags of each of John's bins (equal weight per lag): lag axis (NLAG) -> (4)"""
    G = np.moveaxis(G, axis, -1)
    return np.moveaxis(np.stack([G[..., a:b + 1].mean(-1) for a, b in LAG_BINS], -1), -1, axis)


def run_stats(run, D=None):
    """value curves and resamples of one network: for m in ALL: G_<m> (K, 5 k, NLAG) growth (value), Gb_<m> (K, 5, 4)
    lag-bin growth and its resamples Gb_<m>_boot (N_BOOT, K, 5, 4), I_<m> (K, 5) immediate impact and I_<m>_boot;
    G_<m>_var (K, 5, NLAG, 2) per variable (value only)"""
    D = load_run(run) if D is None else D
    out = dict(x=D["x"], raw=D["raw"], n_total=D["n_total"])
    for m in ALL:
        Ge = np.moveaxis(growth_env(D[m]), 1, 0)                                   # (50, K, 5, NLAG)
        out[f"G_{m}"] = Ge.sum(0) / (UNITS * gap.N_ENV)
        out[f"Gb_{m}"], out[f"Gb_{m}_boot"] = pooled(lag_bins(Ge))
        out[f"I_{m}"], out[f"I_{m}_boot"] = pooled(np.moveaxis(impact_env(D[m]), 1, 0))
        out[f"G_{m}_var"] = np.stack([growth_env(D[m], v_).sum(1) for v_ in range(NV)], -1) / (
            PER_ENV_SUB * len(FLIP_S) * gap.N_ENV)
    return out


def ref_stats(E=None):
    """the exact observers: for obs in bayes / naive / adf and m in ALL: G (5, NLAG), Gb (5, 4) and Gb_boot (N_BOOT, 5,
    4), I (5,) and I_boot"""
    E = refs_env() if E is None else E
    out = {}
    for nm in ("bayes", "naive", "adf"):
        for m in ALL:
            S = E[f"{nm}_{m}"]
            Ge = growth_env(S)                                                     # (50, 5, NLAG)
            out[f"{nm}_G_{m}"] = Ge.sum(0) / (UNITS * gap.N_ENV)
            out[f"{nm}_Gb_{m}"], out[f"{nm}_Gb_{m}_boot"] = pooled(lag_bins(Ge))
            out[f"{nm}_I_{m}"], out[f"{nm}_I_{m}_boot"] = pooled(impact_env(S))
            out[f"{nm}_G_{m}_full"] = pooled(Ge)
    return out


# ── timing of a transient through training (td_tt_memory_events' rules, copied) ──

def _peak(y):
    """the maximum of the 3-checkpoint running mean (per row)"""
    y = np.atleast_2d(y)
    k = np.ones(3) / 3
    return np.array([np.convolve(v, k, "valid").max() for v in y])


def first_cross(x, y, lev):
    """the first training episode at which y (..., K) reaches lev (...,), log-interpolated; NaN if never"""
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


TREND_W = 5                                                                      # checkpoints in the running median


def trend(v, w=TREND_W):
    """the curve's trend over training: a running median over w checkpoints along the last axis (window truncated at
    the ends); the initial network (index 0, drawn at 10) keeps its own value and enters no window, so a single
    checkpoint's excursion cannot move the trend while a sustained change does"""
    v = np.asarray(v, np.float64)
    out = v.copy()
    K, h = v.shape[-1], w // 2
    for i in range(1, K):
        a, b = max(1, i - h), min(K, i + h + 1)
        out[..., i] = np.median(v[..., a:b], axis=-1)
    return out


def transient(x, y, yb, fracs=(.25, .5, .75)):
    """a curve that rises and falls: peak (3-checkpoint running mean, its location), and for every fraction f of the
    peak: rise_f = first reached, fall_f = below it for good (gap.crossing, sign -1); levels recomputed in every
    resample; widths in decades: rise 25 -> 75%, fall 75 -> 25%. Intervals: 5-95% (gap.interval: a resample that never
    crosses counts as after the end)"""
    pk, pkb = float(_peak(y)[0]), _peak(yb)
    kp = int(np.argmax(np.convolve(y, np.ones(3) / 3, "same")))
    out = dict(peak=pk, peak_lo=float(np.percentile(pkb, 5)), peak_hi=float(np.percentile(pkb, 95)),
               x_peak=float(x[kp]), end=float(y[-3:].mean()),
               end_lo=float(np.percentile(yb[:, -3:].mean(1), 5)), end_hi=float(np.percentile(yb[:, -3:].mean(1), 95)))
    for f in fracs:
        rb = first_cross(x, yb, f * pkb)
        fb = gap.crossing(x, yb, f * pkb, -1.)
        for nm, p, b in (("rise", float(first_cross(x, y, f * pk)[0]), rb), ("fall", gap.crossing(x, y, f * pk, -1.), fb)):
            lo, hi, sh = gap.interval(b)
            out[f"{nm}{int(100 * f)}"] = dict(p=p, lo=lo, hi=hi, share=sh, boot=b)
    for nm, a, b in (("rise", 25, 75), ("fall", 75, 25)):
        wb = np.log10(out[f"{nm}{b}"]["boot"]) - np.log10(out[f"{nm}{a}"]["boot"])
        out[f"w_{nm}"] = dict(p=float(np.log10(out[f"{nm}{b}"]["p"]) - np.log10(out[f"{nm}{a}"]["p"])),
                              lo=float(np.nanpercentile(wb, 5)), hi=float(np.nanpercentile(wb, 95)),
                              share=float(np.isfinite(wb).mean()))
    return out


def iv(b, axis=0):
    """5-95% interval"""
    return np.nanpercentile(b, 5, axis), np.nanpercentile(b, 95, axis)


# ═══════════════════════════════════════════════════════════════════
# Checks (CPU; -> OUT/checks.json)
# ═══════════════════════════════════════════════════════════════════

CHECK_CK = (("trained", 25), ("trained", 1100), ("trained", 2100), ("trained", 49995), ("echo", 49995))


def _flip_obs(obs, s, m):
    o2 = obs.copy()
    bits = ((m >> np.arange(NO)) & 1).astype(bool)
    o2[:, s - 1, bits] = ~o2[:, s - 1, bits]
    return o2


def checks():
    """numerical checks of the flip engine and the exact observers (CPU, small subsets) and of the harvested files"""
    out = {}
    P = ti.probes()
    rows_all = subset()
    # 1. the probe subset = the stored chaos sweep's (td_tt_chaos.subset())
    ok = []
    for run in RUNS:
        for e in ti.checkpoints(run)[::10]:
            f = os.path.join(CHAOS, "sweep", f"{run}_{e:05d}.npz")
            if os.path.exists(f):
                with np.load(f) as Z:
                    ok.append(bool(np.array_equal(Z["rows"], rows_all)))
    out["subset_equals_sweep_rows"] = [int(sum(ok)), len(ok)]
    # 2. network flips vs explicitly flipped observation arrays through td_tt_intervene.forward (the validated engine)
    rows = rows_all[::100]                                                          # 10 probes, one per 5 environments
    dev = torch.device("cpu")
    sel_m = [int(np.flatnonzero(MASKS == (1 << k) - 1)[0]) for k in KS]             # masks 1, 3, 7, 15, 31
    for run, raw in CHECK_CK:
        N = net_flips(run, raw, rows, dev, chunk=5, keep=True)
        w = ti.load(run, raw)
        eng = ti.forward(w, rows=rows, device=dev)
        lp_all = eng["logq"].astype(np.float64)
        e_dh, e_skl, e_lq = 0., 0., 0.
        for a, s in enumerate(FLIP_S):
            for mi in sel_m:
                o2 = _flip_obs(P["obs"][rows], s, MASKS[mi])
                E = dict(obs=o2, env=P["env"][rows], Z=P["Z"], r=P["r"][rows], cell=P["cell"][rows])
                f2 = ti.forward(w, episodes=E, device=dev)["logq"].astype(np.float64)
                win = slice(s - 1, s - 1 + NLAG)
                lp, lq = lp_all[:, win], f2[:, win]
                dh_b, skl_b = measures_np(lp, lq - lp)
                Dm = N["D"][:, a, mi].astype(np.float64)
                lq_m = _norm(N["logq"][:, win].astype(np.float64) + Dm)
                e_dh = max(e_dh, float(np.abs(dh_b - osc(Dm)).max()))
                e_skl = max(e_skl, float(np.abs(skl_b - ((np.exp(lp) - np.exp(lq_m)) * (lp - lq_m)).sum(-1) / 2).max()))
                e_lq = max(e_lq, float(np.abs(lq - lq_m).max()))
        out[f"engine_{run}_{raw}"] = dict(intact_logq_max=float(np.abs(N["logq"] - eng["logq"]).max()),
                                          flipped_logq_max=e_lq, dh_max=e_dh, skl_max=e_skl,
                                          dh_range=[float(N["dh"].min()), float(N["dh"].max())],
                                          sup_k1_equals_actual=float(np.abs(N["dh_sup"][:, :, 0] - N["dh"][:, :, 0]).max()))
    # 3. the k = 1 flips vs the stored chaos sweep (every harvested checkpoint; s = 1, 3, 5; fp16 storage)
    mx, ulp, n = 0., 1., 0
    for run in RUNS:
        for e in ti.checkpoints(run):
            if os.path.exists(ck_file(run, e)):
                with np.load(ck_file(run, e)) as Z:
                    if "sweep_max_abs" in Z.files:
                        mx, ulp, n = max(mx, float(Z["sweep_max_abs"])), min(ulp, float(Z["sweep_within_ulp"])), n + 1
    out["sweep_k1"] = dict(checkpoints=n, max_abs=mx, min_share_within_1_fp16_ulp=ulp)
    # 4. exact observers vs brute force (explicitly flipped observations, all 31 subsets, every s; 20 probes)
    rows = rows_all[::50]
    E = exact_flips(rows, chunk=10)
    L = tables()
    obs, env = P["obs"][rows], P["env"][rows]
    Le = L[env]
    lmB = _norm(marg_log(np.cumsum(step_loglik(obs, Le), 1)))
    Lb = np.stack([Le.mean(3), Le.mean(2)], 2)

    def naive_log(o):
        o = o.astype(np.float64)
        return _norm(np.cumsum(np.einsum("nti,nikr->ntkr", o, np.log(Lb)) +
                               np.einsum("nti,nikr->ntkr", 1 - o, np.log1p(-Lb)), 1))
    cn = naive_log(obs)
    la = adf_run(step_loglik(obs, Le))
    out["bayes_vs_log_jb"] = float(np.abs(lmB - P["log_jb"][rows]).max())
    out["naive_vs_log_nb"] = float(np.abs(cn - P["log_nb"][rows]).max())
    err = {k: 0. for k in ("bayes", "naive", "adf")}
    for a, s in enumerate(FLIP_S):
        acc = {k: {m: np.zeros((len(rows), len(MASKS), NLAG, NV)) for m in MEAS} for k in err}
        win = slice(s - 1, s - 1 + NLAG)
        for mi, m_ in enumerate(MASKS):
            o2 = _flip_obs(obs, s, m_)
            ll2 = step_loglik(o2, Le)
            for k, lp, lq in (("bayes", lmB[:, win], _norm(marg_log(np.cumsum(ll2, 1)))[:, win]),
                              ("naive", cn[:, win], naive_log(o2)[:, win]), ("adf", la[:, win], adf_run(ll2)[:, win])):
                acc[k]["dh"][:, mi], acc[k]["skl"][:, mi] = measures_np(lp, lq - lp)
        for k in err:
            for m in MEAS:
                err[k] = max(err[k], float(np.abs(kmean_np(acc[k][m], 1) - E[f"{k}_{m}"][:, a]).max()))
    out["exact_vs_brute_force_max"] = err
    # 5. stored references: naive's d_H constant in lag; naive's superposition exact; SKL bounds
    with np.load(os.path.join(OUT, "refs.npz")) as Z:
        Rf = {k: Z[k].astype(np.float64) for k in Z.files if k != "rows"}
    out["naive_dh_lag_dev_max"] = float(np.abs(Rf["naive_dh"] - Rf["naive_dh"][..., :1, :]).max())
    out["naive_sup_minus_actual_max"] = max(float(np.abs(Rf["naive_dh_sup"] - Rf["naive_dh"]).max()),
                                            float(np.abs(Rf["naive_skl_sup"] - Rf["naive_skl"]).max()))
    out["refs_skl_over_half_dh_max"] = max(float((Rf[f"{k}_skl"] - Rf[f"{k}_dh"] / 2).max())
                                           for k in ("bayes", "naive", "adf"))
    # 6. harvested files (every checkpoint of every run): float16 per-probe arrays vs the float64 environment sums;
    #    SKL bounds; k = 1 superposition
    rel, b_hi, b_lo, sup1, nf = 0., -np.inf, np.inf, 0., 0
    envs = P["env"][rows_all]
    for run, raw in [(r, e) for r in CLS_RUNS for e in ti.checkpoints(r) if os.path.exists(ck_file(r, e))]:
        nf += 1
        with np.load(ck_file(run, raw)) as Z:
            for m in MEAS:
                v = Z[m].astype(np.float64)
                se = np.stack([v[envs == e].sum(0) for e in range(gap.N_ENV)])
                rel = max(rel, float(np.abs(se - Z[f"{m}_envsum"]).max() / np.abs(Z[f"{m}_envsum"]).max()))
            dh, sk = Z["dh"].astype(np.float64), Z["skl"].astype(np.float64)
            b_hi = max(b_hi, float((sk - dh / 2).max()))
            b_lo = min(b_lo, float(sk.min()))
            sup1 = max(sup1, float(np.abs(Z["dh_sup_envsum"][:, :, 0] - Z["dh_envsum"][:, :, 0]).max()))
    out["harvest_files_checked"] = nf
    out["fp16_envsum_rel_max"] = rel
    out["net_skl_minus_half_dh_max"] = b_hi
    out["net_skl_min"] = b_lo
    out["net_sup_k1_minus_actual_envsum_max"] = sup1
    # 7. count classes: every flip set's class k - 2 j = the net change of ones of the explicitly flipped observations
    obs = P["obs"][rows_all]
    code = (obs * (1 << np.arange(NO))).sum(-1)
    bad = 0
    for s_ in FLIP_S:
        for mi, m_ in enumerate(MASKS):
            net = _flip_obs(obs, s_, m_)[:, s_ - 1].sum(-1).astype(int) - obs[:, s_ - 1].sum(-1).astype(int)
            bad += int(np.sum(net != KOF[mi] - 2 * popcount(code[:, s_ - 1] & m_)))
    out["class_net_change_mismatches"] = bad
    # 8. harvested count classes (every file): their C(5, k) means = the stored harvest; units per environment
    mx, nfc, units_ok = {m: 0. for m in MEAS}, 0, True
    for run in CLS_RUNS:
        for e in ti.checkpoints(run):
            if os.path.exists(cls_file(run, e)):
                nfc += 1
                with np.load(cls_file(run, e)) as Zf:
                    for m in MEAS:
                        mx[m] = max(mx[m], float(Zf[f"check_{m}"]))
                    units_ok &= bool(np.all(Zf["cls_n"].sum(1) == PER_ENV_SUB * len(FLIP_S) * len(MASKS)))
    out["classes_vs_harvest"] = dict(files=nfc, max_abs_dh=mx["dh"], max_abs_skl=mx["skl"], units_per_env_ok=units_ok)
    # 9. the exact observers' count classes: their C(5, k) means = refs.npz (lag bins, per-environment sums)
    E_ = refs_env()
    with np.load(os.path.join(OUT, "refs_classes.npz")) as Zf:
        RC = {k: Zf[k] for k in Zf.files}
    kn = np.array([comb(NO, k) for k in KS], np.float64)
    err = 0.
    for nm in ("bayes", "naive", "adf"):
        for m in MEAS:
            mine = RC[f"{nm}_cls_sum_{m}"].reshape(gap.N_ENV, len(KS), NJ, NB).sum(2) / kn[None, :, None]
            S = E_[f"{nm}_{m}"]
            ref = np.concatenate([impact_env(S)[..., None], lag_bins(growth_env(S))], -1) / NV
            err = max(err, float(np.abs(mine - ref).max()))
    out["refs_classes_vs_refs_max"] = err
    # 10. the Trained RNN's accuracy crossings: td_tt_gap 'jumps' = td_tt_replication's synthesis (same resamples)
    a = accuracy_crossings(("trained",))["trained"]
    with np.load(REPL, allow_pickle=False) as Zr:
        ev = list(Zr["trained|acc|events"])
        out["acc_jumps_vs_replication_boot_max"] = max(
            float(np.nanmax(np.abs(a[c]["boot"] - Zr[f"trained|acc|ev{ev.index(lev)}|boot"])))
            for c, lev in (("c0", "0.0"), ("c50", "0.5")))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "checks.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out.items():
        print(f"  {k}: {v}")
    return out


# ═══════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════

C_NET = {"trained": "#1f77b4", "echo": "#ff7f0e"}
NAME = {"trained": "Trained RNN", "echo": "ESN", "trained_p1": "N = 1000, loss power 1", "trained_N200": "N = 200",
        "trained_N100": "N = 100", "trained_N50": "N = 50"}
C_RUN = {"trained": "#1f77b4", "trained_N200": "#17becf", "trained_N100": "#2ca02c", "trained_N50": "#8c564b",
         "trained_p1": "#9467bd"}
SHORT = {"trained": "N = 1000", "trained_N200": "N = 200", "trained_N100": "N = 100", "trained_N50": "N = 50",
         "trained_p1": "N = 1000, loss power 1"}
C_ACC = "#d9a400"                                       # the Trained RNN's accuracy crossings (bands)
MAP_RAW = (25, 1100)                                    # phase diagrams: training 20, 880


def shades(hexcol, n=5):
    """n shades of one hue from light (k = 1) to dark (k = n)"""
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(hexcol))
    out = []
    for t in np.linspace(0, 1, n):
        out.append(tuple(c + (1 - c) * .62 * (1 - 2 * t)) if t < .5 else tuple(c * (1 - .55 * (2 * t - 1))))
    return out


def _contours(ax, X, Y, Z, levels, fs, fmt="%g"):
    """white labelled iso-lines with a dark outline (the maps' summary lines)"""
    import matplotlib.patheffects as pe
    lev = [v for v in levels if Z.min() < v < Z.max()]
    if not lev:
        return
    cs = ax.contour(X, Y, Z, levels=lev, colors="w", linewidths=1.3)
    for c in cs.collections if hasattr(cs, "collections") else [cs]:
        c.set_path_effects([pe.Stroke(linewidth=2.8, foreground="k", alpha=.55), pe.Normal()])
    for t in ax.clabel(cs, fmt=fmt, fontsize=fs, inline=True, inline_spacing=3):
        t.set_path_effects([pe.Stroke(linewidth=2.2, foreground="k", alpha=.6), pe.Normal()])


def _ticks_right(ax, ys, color, x0=4.66, x1=4.78, lw=2.2):
    """short horizontal ticks at the right edge (log10 training episodes x0..x1): a reference's levels"""
    for y in np.atleast_1d(ys):
        ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=5, clip_on=False)


def _logx(ax, FL, xmax=4.8, label="training episodes"):
    ax.set_xlim(np.log10(8), xmax)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.set_xlabel(label, fontsize=FL)


def _acc_bands(ax, acc):
    """the Trained RNN's goal accuracy passes naive (c0) and half-way to Bayes (c50): 5-95% bands"""
    for c in ("c0", "c50"):
        ax.axvspan(np.log10(acc[c]["lo"]), np.log10(acc[c]["hi"]), color=C_ACC, alpha=.28, lw=0, zorder=0)


def _title(ax, letter, name, formula, TS, FT):
    """'letter  name' on top, the formula (one or two lines) below it"""
    n = formula.count("\n") + 1
    ax.set_title(f"{letter}   {name}", loc="left", fontsize=TS, pad=6 + 14.5 * n)
    ax.text(0, 1.018, formula, transform=ax.transAxes, fontsize=FT, ha="left", va="bottom", linespacing=1.25)


def figure(path=None, A=None):
    """td_tt_flips.png. Row 1 through training (log axis; bands 5-95% over environment resamples): a the d_H growth of
    a k-bit flip at lags 18-21 (Trained RNN, ESN, Bayes); b the early count code (2-bit flips that keep vs change the
    number of ones); c the response beyond the sum of single flips; d the bulk (SKL) growth by lag bin; e the fall of
    the single-flip growth on each width run's own clock. Row 2: phase diagrams (lag x k) of the d_H growth at 20 and
    880 training episodes and Bayes's (f-h); i the extra growth from changing the count (count-code index) in every
    run, two columns wide"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    FL, TS, FT, FC = fs["label"] + 1, 13.5, 10.5, 10
    A = sharpness(analysis()) if A is None else A
    Z, RS, acc = A["Z"], A["R"], A["acc"]["trained"]
    z = Z["trained"]
    fig = plt.figure(figsize=(21.5, 10.6))
    gs = fig.add_gridspec(2, 5, hspace=.66, wspace=.36, left=.045, right=.985, top=.80, bottom=.155)
    kcol = {r: shades(C_NET[r]) for r in RUNS}
    B = len(LAG_BINS) - 1
    # a: growth per k through training
    ax = fig.add_subplot(gs[0, 0])
    for run in ("echo", "trained"):
        zz = Z[run]
        lx = np.log10(zz["x"])
        for k in range(len(KS)):
            lo, hi = iv(zz["Gb_dh_boot"][:, :, k, B])
            ax.fill_between(lx, lo, hi, color=kcol[run][k], alpha=.25, lw=0)
            ax.plot(lx, zz["Gb_dh"][:, k, B], color=kcol[run][k], lw=1.9 if run == "trained" else 1.4)
    _ticks_right(ax, RS["bayes_Gb_dh"][:, B], "k")
    _ticks_right(ax, [0.], "0.55")
    _acc_bands(ax, acc)
    ax.axhline(0, color="k", lw=.6)
    ax.text(np.log10(300), 5.45, "Trained RNN", color=C_NET["trained"], fontsize=FL + .5, ha="center")
    ax.text(np.log10(9000), 4.05, "ESN", color=C_NET["echo"], fontsize=FL + .5, ha="center")
    ax.set_ylim(-.35, 6.1)
    _logx(ax, FL)
    ax.set_ylabel(r"growth of $d_H$ (nats)", fontsize=FL)
    _title(ax, "a", "Growth of a k-bit flip's effect",
           r"$\chi(s{+}\mathrm{lag},s)-\chi(s,s)$, $\chi=d_H$; lags 18-21" + "\nk = 1 (light) ... 5 (dark)", TS, FT)
    # b: the count code
    ax = fig.add_subplot(gs[0, 1])
    for run in ("echo", "trained"):
        c = A["cls"][run]
        lx = np.log10(c["x"])
        for g, ls in (("k2_n2", "-"), ("k2_n0", (0, (3, 1.6)))):
            lo, hi = iv(c[f"{g}_dh_boot"][:, :, 4])
            ax.fill_between(lx, lo, hi, color=C_NET[run], alpha=.2, lw=0)
            ax.plot(lx, c[f"{g}_dh"][:, 4], color=C_NET[run], ls=ls, lw=2. if run == "trained" else 1.4)
    rc = A["cls_ref"]["bayes"]
    _ticks_right(ax, [rc["k2_n2_dh"][0, 4], rc["k2_n0_dh"][0, 4]], "k")
    _acc_bands(ax, acc)
    ax.axhline(0, color="k", lw=.6)
    ax.text(np.log10(8.6), 4.8, "count code at\n20-40 episodes:\nblue dashed\nstays near 0", fontsize=FL, ha="left",
            va="top")
    ax.legend(handles=[Line2D([], [], color="k", lw=1.8, label=r"change (net $\pm2$)"),
                       Line2D([], [], color="k", lw=1.8, ls=(0, (3, 1.6)), label="keep (net 0)"),
                       Line2D([], [], color=C_NET["trained"], lw=2., label="Trained RNN"),
                       Line2D([], [], color=C_NET["echo"], lw=1.6, label="ESN")],
              loc="upper right", frameon=False, fontsize=FL - .5, handlelength=2.2, borderaxespad=.1, labelspacing=.3)
    ax.set_ylim(-.2, 4.9)
    _logx(ax, FL)
    ax.set_ylabel(r"growth of $d_H$, 2-bit flips (nats)", fontsize=FL)
    _title(ax, "b", "Early count code", r"$\chi(s{+}\mathrm{lag},s)-\chi(s,s)$, $\chi=d_H$; 2-bit flips, lags 18-21"
           + "\nkeep (dashed) or change (solid) the number of ones", TS, FT)
    # c: beyond the sum of single flips
    ax = fig.add_subplot(gs[0, 2])
    sup, supb = A[("trained", "sup")]
    lx = np.log10(z["x"])
    for k in range(1, len(KS)):
        lo, hi = iv(-supb[:, :, k, B])
        ax.fill_between(lx, lo, hi, color=kcol["trained"][k], alpha=.25, lw=0)
        ax.plot(lx, -sup[:, k, B], color=kcol["trained"][k], lw=1.9)
    _ticks_right(ax, RS["bayes_Gb_dh"][1:, B] - RS["bayes_Gb_dh_sup"][1:, B], "k")
    _acc_bands(ax, acc)
    ax.axhline(0, color="k", lw=.8)
    ax.text(np.log10(12), .93, "more than linear", fontsize=FL, va="center")
    ax.text(np.log10(12), -.9, "less than linear", fontsize=FL, va="center")
    ax.set_ylim(-1.15, 1.25)
    _logx(ax, FL)
    ax.set_ylabel(r"growth of $d_H$: actual $-$ linear (nats)", fontsize=FL)
    _title(ax, "c", "Beyond the sum of single flips",
           r"growth(k bits) $-$ growth($\Sigma$ single-flip changes)" + "\nk = 2 (light) ... 5 (dark); lags 18-21", TS, FT)
    # d: bulk (SKL) growth by lag bin, k = 5
    ax = fig.add_subplot(gs[0, 3])
    bcol = shades(C_NET["trained"], 4)
    k = len(KS) - 1
    for b in range(len(LAG_BINS)):
        lo, hi = iv(z["Gb_skl_boot"][:, :, k, b])
        ax.fill_between(lx, lo, hi, color=bcol[b], alpha=.25, lw=0)
        ax.plot(lx, z["Gb_skl"][:, k, b], color=bcol[b], lw=1.9)
        zc = A[("trained", "skl", k, b, "zero")]
        if np.isfinite(zc["p"]):
            ax.plot(np.log10(zc["p"]), 0, "o", color=bcol[b], ms=6.5, mec="k", mew=.7, zorder=6)
    _ticks_right(ax, RS["bayes_Gb_skl"][k], "k")
    _acc_bands(ax, acc)
    ax.axhline(0, color="k", lw=.8)
    ax.legend(handles=[Line2D([], [], color=bcol[b], lw=2.2, label=f"lags {a_}-{b_}")
                       for b, (a_, b_) in enumerate(LAG_BINS)] +
              [Line2D([], [], color="k", marker="o", ls="", ms=6, mfc="w", label=r"$<0$ for good")],
              loc="lower left", frameon=False, fontsize=FL - .5, handlelength=1.6, borderaxespad=.1)
    _logx(ax, FL)
    ax.set_ylabel("growth of SKL (nats)", fontsize=FL)
    _title(ax, "d", "Bulk divergence turns contractive",
           r"SKL$(s{+}\mathrm{lag})-$SKL$(s)$; k = 5" + "\n" + r"SKL $=\Sigma_r(p_r-q_r)\log(p_r/q_r)/2$", TS, FT)
    # e: the fall on each run's own clock (k = 1)
    ax = fig.add_subplot(gs[0, 4])
    allacc = A["acc"]
    for run in ("trained_p1", "trained_N50", "trained_N100", "trained_N200", "trained"):
        zz = Z[run]
        c0 = allacc[run]["c0"]
        lxo = np.log10(zz["x"] / c0["p"])
        ax.plot(lxo, zz["Gb_dh"][:, 0, B], color=C_RUN[run], lw=2. if run == "trained" else 1.5)
        t = A[(run, "dh", 0, B, "tr")]["fall50"]
        if np.isfinite(t["p"]):
            rb = np.log10(np.where(np.isfinite(t["boot"]), t["boot"], np.inf) / c0["boot"])
            lo, hi = np.percentile(rb, 5), np.percentile(rb, 95)
            yv = .5 * A[(run, "dh", 0, B, "tr")]["peak"]
            ax.plot([lo, min(hi, 2.2)], [yv, yv], color=C_RUN[run], lw=1.4)
            ax.plot(np.log10(t["p"] / c0["p"]), yv, "o", color=C_RUN[run], ms=6.5, mec="k", mew=.7, zorder=6)
    ax.axvline(0, ymax=.8, color="k", lw=1, ls=":")
    ax.text(.06, -.42, "passes naive", fontsize=FL, va="bottom", ha="left")
    ax.axhline(0, color="k", lw=.6)
    ax.legend(handles=[Line2D([], [], color=C_RUN[r], lw=2, label=SHORT[r]) for r in
                       ("trained", "trained_N200", "trained_N100", "trained_N50", "trained_p1")],
              loc="upper left", ncol=2, frameon=False, fontsize=FL - 1.5, handlelength=1.2, borderaxespad=.1,
              columnspacing=1., labelspacing=.25)
    ax.set_xlim(-2.3, 2.45)
    ax.set_ylim(-.5, 3.55)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels(["1/100", "1/10", "1", "10", "100"])
    ax.set_xlabel("training episodes / own passing-naive time", fontsize=FL)
    ax.set_ylabel(r"growth of $d_H$, k = 1 (nats)", fontsize=FL)
    _title(ax, "e", "Single-flip fall follows passing naive",
           r"$\chi(s{+}\mathrm{lag},s)-\chi(s,s)$, $\chi=d_H$; k = 1, lags 18-21" + "\n"
           "circles: below half the peak for good (5-95%)", TS, FT)
    # f-h: phase diagrams (f on its own colour scale, g-h on a shared one)
    L1 = np.arange(1, NLAG)
    maps = []
    for raw in MAP_RAW:
        i = int(np.flatnonzero(z["raw"] == raw)[0])
        maps.append((f"Trained RNN, {z['x'][i]:,.0f} episodes", z["G_dh"][i][:, 1:]))
    maps[0] = (maps[0][0] + " (own scale)", maps[0][1])
    maps.append(("Bayes (exact)", RS["bayes_G_dh"][:, 1:]))
    vmax = max(m.max() for _, m in maps[1:])
    form = r"$\chi(s{+}\mathrm{lag},s)-\chi(s,s)$, $\chi=d_H$, over (lag, k)"
    notes = {0: "\ncount code: rows pair up (k = 1 with 2, 3 with 4)",
             1: "\nnear the amplification's peak: smooth; scale of h", 2: "\nthe joint posterior's marginals"}
    levels = {0: (.25, .5, .75, 1, 1.25, 1.5, 1.75)}
    for j, (lab, G) in enumerate(maps):
        ax = fig.add_subplot(gs[1, j])
        vm = G.max() if j == 0 else vmax
        im = ax.pcolormesh(L1, np.array(KS), G, cmap="viridis", vmin=0, vmax=vm, shading="nearest", rasterized=True)
        _contours(ax, L1, np.array(KS), G, levels.get(j, (.5, 1, 2, 3, 4)), FC)
        ax.set_yticks(KS)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel("lag after the flip (steps)", fontsize=FL)
        if j == 0:
            ax.set_ylabel("bits flipped, k", fontsize=FL)
        _title(ax, "fgh"[j], lab, form + notes[j], TS, FT)
        if j in (0, 2):
            cb = fig.colorbar(im, cax=ax.inset_axes([1.03, 0., .045, 1.]))
            cb.set_label(r"growth of $d_H$ (nats); bright = more", fontsize=FL - .5)
    # i: the extra growth from changing the count, every run (two columns wide)
    ax = fig.add_subplot(gs[1, 3:5])
    pos = ax.get_position()
    ax.set_position([pos.x0 + .018, pos.y0, pos.width - .018 - .08, pos.height])     # room for the legend at the right
    for run in ("echo", "trained_p1", "trained_N50", "trained_N100", "trained_N200", "trained"):
        c = A["cls"][run]
        col = C_NET["echo"] if run == "echo" else C_RUN[run]
        lo, hi = iv(c["k2_cc_dh_boot"][:, :, 4])
        lxc = np.log10(c["x"])
        ax.fill_between(lxc, lo, hi, color=col, alpha=.2, lw=0)
        ax.plot(lxc, c["k2_cc_dh"][:, 4], color=col, lw=2. if run == "trained" else 1.5)
        if run != "echo":
            ax.plot(np.log10(allacc[run]["c0"]["p"]), -.33, "^", color=col, ms=8, mec="k", mew=.6, clip_on=False)
    ax.axhline(0, color="k", lw=.6)
    ax.text(np.log10(9), -.33, "passes naive:", fontsize=FL - .5, va="center")
    ax.text(np.log10(9.5), 2.18, "count code (early peaks, N = 1000 to 50), growth of 2-bit flips:\nkeep the count "
            "+.12 to +.37, change it +1.4 to +2.0 nats", fontsize=FL, va="center")
    ax.text(np.log10(200), 1.3, "partial count dependence:\ncount-preserving flips grow 2.3-3.5 nats",
            fontsize=FL, ha="left", va="bottom")
    ax.text(np.log10(900), .85, "loss power 1 (purple): the same code,\nsmall in nats (+.09 at 20 episodes)",
            fontsize=FL - .5, va="bottom")
    ax.legend(handles=[Line2D([], [], color=C_RUN[r], lw=2, label=SHORT[r]) for r in
                       ("trained", "trained_N200", "trained_N100", "trained_N50", "trained_p1")] +
              [Line2D([], [], color=C_NET["echo"], lw=2, label="ESN")],
              loc="upper left", bbox_to_anchor=(1.01, 1.), frameon=False, fontsize=FL - .5, handlelength=1.3,
              borderaxespad=0., labelspacing=.45)
    ax.set_ylim(-.45, 2.45)
    _logx(ax, FL)
    ax.set_ylabel("count-changing $-$ count-preserving (nats)", fontsize=FL)
    _title(ax, "i", "Extra growth from changing the count (count-code index)",
           r"growth(net $\pm2$) $-$ growth(net 0), 2-bit flips, paired; growth $=\chi(s{+}\mathrm{lag},s)-\chi(s,s)$, "
           r"$\chi=d_H$; lags 18-21" + "\na count code only where count-preserving growth is near 0 (panel b); "
           "every run", TS, FT)
    # shared legend for the accuracy bands and reference ticks
    fig.legend(handles=[Patch(color=C_ACC, alpha=.5, label="Trained RNN (N = 1000) accuracy passes naive / half-way "
                              "to Bayes (5-95%)"),
                        Line2D([], [], color="k", lw=2.4, label="Bayes, exact (ticks at the right edge)"),
                        Line2D([], [], color="0.55", lw=2.4, label="naive, exact (growth 0)")],
               loc="upper center", bbox_to_anchor=(.5, .918), ncol=3, frameon=False, fontsize=FL + .5)
    fig.text(.5, .985, "Flipping k of the 5 observations at once: no boundary in (lag, k) at any stage; the one abrupt "
             "change is an early count code that switches off between 40 and 100 episodes (N = 1000; peaks later at "
             "smaller N, panel i)", ha="center", va="top", fontsize=TS + 1.5)
    fig.text(.5, .95, "The amplification falls gradually: single flips fall below half their peak 0.6-0.9 decades "
             "after each run passes naive (N = 50: not by the end); late in training several flips act more than "
             "linearly, as in Bayes", ha="center", va="top", fontsize=TS + .5)
    nck = {r: len(Z[r]["x"]) for r in Z}
    fig.text(.5, .045, "Data: 1,000 probes (20 per test environment x 50 environments) x 30 steps; every subset of the 5 "
             "observations flipped (31 flip sets) at steps s = 1, 3, 5, 7, 9; the network rerun from its intact state; "
             "means over flip sets, flip steps, both variables and probes.\n"
             f"Checkpoints: Trained RNN {nck['trained']}, ESN {nck['echo']}, width runs {nck['trained_N200']} each "
             "(seed 20261029, one seed per width). Bands: 5-95% over 1,000 resamples of the 50 environments. "
             "Bayes, naive: exact (float64), the same flips.", ha="center", va="center", fontsize=FL, linespacing=1.6)
    path = fig_path("td_tt_flips.png") if path is None else path
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


# ═══════════════════════════════════════════════════════════════════
# Analysis: every number of the record (numbers()), the figure's arrays, the combined figure's export (export())
# ═══════════════════════════════════════════════════════════════════

REPL = os.path.join(gap.TT, "replication", "synthesis.npz")
STAGES = (25, 50, 75, 125, 400, 1100, 2100, 49995)          # raw episodes: training 20, 40, 60, 100, 320, 880, 1,680, 39,996
EARLY = (10, 200)                                           # the early window of the count code (training episodes)
PLATEAU = ((200, 360), (640, 1040))                         # the two windows of the plateau comparison (training episodes)
CLASS_GROUPS = {"k1": (1, (0, 1)), "k2_n0": (2, (1,)), "k2_n2": (2, (0, 2)), "k3_n1": (3, (1, 2)),
                "k3_n3": (3, (0, 3)), "k4_n0": (4, (2,)), "k4_n2": (4, (1, 3)), "k4_n4": (4, (0, 4)),
                "k5_n1": (5, (2, 3)), "k5_n3": (5, (1, 4)), "k5_n5": (5, (0, 5))}       # (k, j's): |net| = |k - 2j|


def accuracy_crossings(runs=("trained",)):
    """each run's goal accuracy at T: passes naive (c0) and half-way to Bayes (c50), with the 1,000 resampled crossings on
    the shared environment resamples. The Trained RNN: td_tt_gap export 'jumps'; the width runs: td_tt_replication's
    synthesis (every run measured on the same 50 environments with the same resample matrix)"""
    out = {}
    for run in runs:
        if run == "trained":
            a = gap.load("jumps")
            i = [j for j in range(len(a["names"])) if a["names"][j] == "goal accuracy at T" and a["run"][j] == run][0]
            out[run] = {c: dict(p=float(a[c][i]), lo=float(a[f"{c}_lo"][i]), hi=float(a[f"{c}_hi"][i]),
                                boot=a[f"{c}_boot"][i]) for c in ("c0", "c50")}
            continue
        with np.load(REPL, allow_pickle=False) as Zr:
            ev = list(Zr[f"{run}|acc|events"])
            d = {}
            for c, lev in (("c0", "0.0"), ("c50", "0.5")):
                p = f"{run}|acc|ev{ev.index(lev)}|"
                d[c] = dict(p=float(Zr[p + "p"]), lo=float(Zr[p + "lo"]), hi=float(Zr[p + "hi"]), boot=Zr[p + "boot"])
            out[run] = d
    return out


def lag_accel(G):
    """A(lag) = G(lag + 1) - 2 G(lag) + G(lag - 1) for lags 1..20 (G (..., NLAG)): a sudden onset of growth at some lag
    shows as a large positive A there; a concave (saturating) growth has A <= 0"""
    return G[..., 2:] - 2 * G[..., 1:-1] + G[..., :-2]


def load_stage(run, raw):
    """the per-environment sums of one checkpoint (as load_run, without the stacking)"""
    with np.load(ck_file(run, raw)) as Zf:
        return {m: Zf[f"{m}_envsum"] for m in ALL}


def load_classes(run):
    """the count classes of a run's harvested checkpoints: cls_sum_<m> (50, K, NCLS, NB), cls_n (50, K, NCLS),
    pair_sum_<m> (50, K, 4, NB), pair_n (50, K, 4) (environment first), x, raw"""
    eps = [e for e in ti.checkpoints(run) if os.path.exists(cls_file(run, e))]
    keys = ("cls_sum_dh", "cls_sum_skl", "cls_n", "pair_sum_dh", "pair_sum_skl", "pair_n")
    out = {k: [] for k in keys}
    for e in eps:
        with np.load(cls_file(run, e)) as Zf:
            for k in keys:
                out[k].append(Zf[k])
    out = {k: np.stack(v, 1) for k, v in out.items()}
    out.update(x=gap.train_x(np.array(eps)), raw=np.array(eps))
    return out


def class_stats(Cd, C=None):
    """count-class curves: for m in dh / skl, '<group>_<m>' (K, NB) and '<group>_<m>_boot' (N_BOOT, K, NB) for the
    CLASS_GROUPS (the group's mean over its (probe, flip step, flip set) units: a ratio of environment sums), and
    '<pair>_<m>' / '_boot' for PAIRS (the mean over (probe, flip step) units of the paired difference); NB index 0 = the
    immediate impact chi(s, s), 1..4 = the growth in the lag bins. Cd: load_classes() or the references' file with a
    checkpoint axis of length 1"""
    C = gap.boot_counts() if C is None else C
    out = dict(x=Cd.get("x"), raw=Cd.get("raw"))
    for m in MEAS:
        for g, (k, js) in CLASS_GROUPS.items():
            c = [(k - 1) * NJ + j for j in js]
            num = Cd[f"cls_sum_{m}"][:, :, c].sum(2)                              # (50, K, NB)
            den = np.broadcast_to(Cd["cls_n"][:, :, c].sum(2)[..., None], num.shape)
            out[f"{g}_{m}"], out[f"{g}_{m}_boot"] = gap.env_ratio(num, den, C)
        for q, pn in enumerate(PAIRS):
            num = Cd[f"pair_sum_{m}"][:, :, q]
            den = np.broadcast_to(Cd["pair_n"][:, :, q][..., None], num.shape)
            out[f"{pn}_{m}"], out[f"{pn}_{m}_boot"] = gap.env_ratio(num, den, C)
    return out


def ref_classes():
    """the exact observers' count classes: {obs: class_stats} (a checkpoint axis of length 1)"""
    with np.load(os.path.join(OUT, "refs_classes.npz")) as Zf:
        R_ = {k: Zf[k] for k in Zf.files}
    out = {}
    for nm in ("bayes", "naive", "adf"):
        Cd = {"cls_n": R_["cls_n"][:, None], "pair_n": R_["pair_n"][:, None]}
        for m in MEAS:
            Cd[f"cls_sum_{m}"] = R_[f"{nm}_cls_sum_{m}"][:, None]
            Cd[f"pair_sum_{m}"] = R_[f"{nm}_pair_sum_{m}"][:, None]
        out[nm] = class_stats(Cd)
    return out


def k_structure(z):
    """from the C(5, k) means of the d_H growth (every checkpoint, every lag bin): inc = the per-bit increments
    G(k) - G(k - 1) (G(0) = 0 exactly), (K, 4 bins, 5 k); kmax = their largest change from one k to the next over
    k = 2..5, (K, 4); zigzag = the odd-minus-even increment (G(3) - G(2) + G(5) - G(4)) / 2 - (G(2) - G(1) + G(4) -
    G(3)) / 2, (K, 4) (> 0: adding an even-numbered bit adds less, as a count code predicts); with '_boot' resamples"""
    out = {}
    for suf, Gb in (("", z["Gb_dh"]), ("_boot", z["Gb_dh_boot"])):
        g = np.swapaxes(Gb, -1, -2)                                               # (..., K, 4 bins, 5 k)
        inc = np.diff(np.concatenate([np.zeros(g.shape[:-1] + (1,)), g], -1), axis=-1)
        out["inc" + suf] = inc
        out["kmax" + suf] = np.abs(np.diff(inc[..., 1:], axis=-1)).max(-1)
        out["zigzag" + suf] = (inc[..., 2] + inc[..., 4]) / 2 - (inc[..., 1] + inc[..., 3]) / 2
    return out


def crossing_dict(x, y, yb, level=0., sign=-1.):
    """a stays-past crossing (gap.crossing) of the curve's trend and of every resample's trend: dict p, lo, hi, share,
    boot"""
    b = gap.crossing(x, trend(yb), level, sign)
    lo, hi, sh = gap.interval(b)
    return dict(p=gap.crossing(x, trend(y), level, sign), lo=lo, hi=hi, share=sh, boot=b)


def order(a, b):
    """'before' / 'after' only when the 5-95% intervals of two crossings (dicts) do not overlap; else 'overlapping'
    (a resample that never crosses counts as after the end: an open bound is +inf)"""
    alo = a["lo"] if np.isfinite(a["lo"]) else np.inf
    blo = b["lo"] if np.isfinite(b["lo"]) else np.inf
    ahi = a["hi"] if np.isfinite(a["hi"]) else np.inf
    bhi = b["hi"] if np.isfinite(b["hi"]) else np.inf
    return "before" if ahi < blo else ("after" if bhi < alo else "overlapping")


def window_mean(x, y, w):
    """the mean of y (..., K) over the checkpoints with w[0] <= x <= w[1]"""
    m = (x >= w[0]) & (x <= w[1])
    return y[..., m].mean(-1), int(m.sum())


def per_variable_end(run, raw):
    """at one checkpoint, lags 18-21: the d_H growth per variable (r0, r1) and per role (goal, contextual variable) with
    resamples and the paired differences r0 - r1 and goal - context: {name: (value (5 k,), boot (N_BOOT, 5))}"""
    S = load_stage(run, raw)["dh"]                                                # (50, nS, 5, NLAG, NV)
    units = PER_ENV_SUB * len(FLIP_S)
    out = {}
    Gv = [lag_bins(growth_env(S, v))[..., -1] for v in range(NV)]                # (50, 5) each
    for nm, Se in (("r0", Gv[0]), ("r1", Gv[1]), ("r0 - r1", Gv[0] - Gv[1])):
        out[nm] = pooled(Se, units)
    P = ti.probes()
    with np.load(ck_file(run, raw)) as Zf:
        dh, rows = Zf["dh"].astype(np.float64), Zf["rows"]
    g = lag_bins(dh - dh[..., :1, :], axis=-2)[..., -1, :].sum(1)                 # (n, 5 k, NV): summed over flip steps
    goal, env = P["goal"][rows].astype(np.int64), P["env"][rows]
    gg, gc = g[np.arange(len(rows)), :, goal], g[np.arange(len(rows)), :, 1 - goal]
    for nm, a in (("goal", gg), ("context", gc), ("goal - context", gg - gc)):
        out[nm] = pooled(np.stack([a[env == e].sum(0) for e in range(gap.N_ENV)]), units)
    out["goal_share_r0"] = float(np.mean(goal == 0))
    return out


def per_step(run, raw):
    """at one checkpoint, lags 18-21: the d_H growth per flip step s (k = 1..5) with resamples: (value (nS, 5), boot)"""
    S = load_stage(run, raw)["dh"]
    G = lag_bins((S - S[..., :1, :]).sum(-1))[..., -1]                            # (50, nS, 5)
    return pooled(G, PER_ENV_SUB * NV)


def online_period(run="trained", window=(160, 400), periods=np.arange(10, 121), n_sur=1000, seed=20261105):
    """the per-update training logs in a window of training episodes (one update per training episode): for the loss
    and the update norms of W_hh, W_ih and the read-out, after removing a cubic trend, the amplitude of the best-fitting
    sinusoid at every period (updates); the period with the largest amplitude and its amplitude's rank against 1,000 AR(1)
    surrogates with the series' own lag-1 autocorrelation and variance (the share of surrogates whose largest amplitude
    over 55-70 updates is at least the data's)"""
    import glob
    d = os.path.join(ti.TD, f"{run}_s{ti.SEED}")
    ep, cols = [], {k: [] for k in ("loss", "W_hh", "W_ih", "readout")}
    key = {"W_hh": "LSTM.weight_hh_l0", "W_ih": "LSTM.weight_ih_l0", "readout": "classifier_readout.weight"}
    for f in sorted(glob.glob(os.path.join(d, "online_*.npz"))):
        with np.load(f) as Zf:
            x_ = Zf["episode"] - Zf["episode"] // 5
            if x_[0] > window[1]:
                break
            names = list(Zf["names"])
            ep.append(x_)
            cols["loss"].append(Zf["loss"])
            for k_, nm in key.items():
                cols[k_].append(Zf["update_norm"][:, names.index(nm)])
    x = np.concatenate(ep).astype(np.float64)
    w = (x > window[0]) & (x <= window[1])
    t = (x[w] - x[w].mean()) / x[w].std()
    rng = np.random.default_rng(seed)
    band = (periods >= 55) & (periods <= 70)
    basis = [np.stack([np.cos(2 * np.pi * x[w] / P_), np.sin(2 * np.pi * x[w] / P_), np.ones(w.sum())], 1)
             for P_ in periods]
    pinv = [np.linalg.pinv(B) for B in basis]

    def amp(y):
        return np.array([np.hypot(*(pv @ y)[:2]) for pv in pinv])
    out = {}
    for k_, v in cols.items():
        yy = np.concatenate(v)[w]
        y = yy - np.polyval(np.polyfit(t, yy, 3), t)
        a = amp(y)
        r1 = np.corrcoef(y[:-1], y[1:])[0, 1]
        sd = y.std() * np.sqrt(1 - r1 ** 2)
        sur = np.zeros(n_sur)
        for i in range(n_sur):
            e = rng.normal(0, sd, len(y))
            s_ = np.zeros(len(y))
            s_[0] = rng.normal(0, y.std())
            for j in range(1, len(y)):
                s_[j] = r1 * s_[j - 1] + e[j]
            sur[i] = amp(s_)[band].max()
        out[k_] = dict(best=int(periods[np.argmax(a)]), best_in_band=int(periods[band][np.argmax(a[band])]),
                       p_band=float(np.mean(sur >= a[band].max())), r1=float(r1))
    out["n_updates"] = int(w.sum())
    return out


def excursion_phase(z, k=4, b=3, window=(160, 400)):
    """the d_H growth's single-checkpoint excursions (curve - trend) in a window, grouped by the checkpoint's raw episode
    mod 75 (the period-3 pattern on the 25-raw-episode grid): mean excursion per phase class with resamples"""
    y, yb = z["Gb_dh"][:, k, b], z["Gb_dh_boot"][:, :, k, b]
    D, Db = y - trend(y), yb - trend(yb)
    x, raw = z["x"], z["raw"]
    m = (x > window[0]) & (x <= window[1])
    out = {}
    for ph in (0, 25, 50):
        s = m & (raw % 75 == ph)
        out[ph] = dict(n=int(s.sum()), mean=float(D[s].mean()), iv=iv(Db[:, s].mean(1)), vals=D[s], x=x[s])
    return out


def analysis(runs=RUNS + WIDTH_RUNS):
    """all statistics (dict A): per run stats (run_stats), the references (ref_stats), the accuracy crossings, the count
    classes (class_stats; network runs and references), the k-structure and superposition at every checkpoint,
    transients (trend) per (run, measure, k, lag bin), SKL zero crossings, plateau windows, lag concavity at every
    checkpoint"""
    Z = {r: run_stats(r) for r in runs}
    A = dict(Z=Z, R=ref_stats(), acc=accuracy_crossings(runs=[r for r in runs if r != "echo"]))
    A["cls"] = {r: class_stats(load_classes(r)) for r in runs if os.path.exists(cls_file(r, ti.checkpoints(r)[0]))}
    A["cls_ref"] = ref_classes()
    for run, z in Z.items():
        x = z["x"]
        A[(run, "kstr")] = k_structure(z)
        sup, supb = z["Gb_dh_sup"] - z["Gb_dh"], z["Gb_dh_sup_boot"] - z["Gb_dh_boot"]
        A[(run, "sup")] = (sup, supb)
        for k in range(len(KS)):
            A[(run, "sup_neg", k)] = crossing_dict(x, sup[:, k, -1], supb[:, :, k, -1], 0., -1.)
            for b in range(len(LAG_BINS)):
                for m in ("dh", "skl"):
                    y, yb = trend(z[f"Gb_{m}"][:, k, b]), trend(z[f"Gb_{m}_boot"][:, :, k, b])
                    A[(run, m, k, b, "tr")] = transient(x, y, yb)
                A[(run, "skl", k, b, "zero")] = crossing_dict(x, z["Gb_skl"][:, k, b], z["Gb_skl_boot"][:, :, k, b])
        a2 = lag_accel(z["G_dh"])                                                 # (K, 5, 20)
        A[(run, "lagacc")] = a2
        if run in ("trained", "echo"):
            A[(run, "plateau")] = {k: [window_mean(x, z["Gb_dh"][:, k, -1], w) for w in PLATEAU] +
                                   [window_mean(x, z["Gb_dh_boot"][:, :, k, -1], w) for w in PLATEAU]
                                   for k in range(len(KS))}
    return A


# ── sharpness through training, model-free: single-interval steps and single-checkpoint excursions ──

SPIKE_WINDOWS = ((20, 160), (160, 400), (400, 2400), (2400, 40000))     # training episodes (spacing 20 / 20 / 80 / 200-800)


def fall_step(x, y, yb):
    """the abruptness of the main decline: between the checkpoints where the curve falls below 75% and below 25% of its
    peak for good (peak = the maximum of the 3-checkpoint running mean), the largest change between two adjacent
    checkpoints as a share of that 50%-of-peak drop (each resample its own levels); a jump would put ~100% in one
    interval. Also the number of checkpoint intervals the drop spans"""
    def one(v):
        pk = np.convolve(v, np.ones(3) / 3, "valid").max()
        i75 = int(np.flatnonzero(v >= .75 * pk)[-1])                              # last checkpoint at or above
        hi25 = np.flatnonzero(v >= .25 * pk)
        if v[-1] >= .25 * pk or not len(hi25):
            return np.nan, -1, 0
        i25 = int(hi25[-1]) + 1                                                    # first checkpoint below for good
        d = np.abs(np.diff(v[i75:i25 + 1]))
        return d.max() / (.5 * pk), i75 + int(np.argmax(d)), i25 - i75
    s_, i_, n_ = one(y)
    sb = np.array([one(v)[0] for v in yb])
    return dict(share=float(s_), share_iv=iv(sb), at=(float(x[i_]), float(x[i_ + 1])) if i_ >= 0 else (np.nan, np.nan),
                n_int=n_)


def spikes(x, y, yb):
    """single-checkpoint excursions: the curve minus its trend (trend()), in the curve's units; per window of
    SPIKE_WINDOWS their RMS and the largest |excursion| with its checkpoint and the 5-95% interval of that value"""
    D, Db = y - trend(y), yb - trend(yb)
    out = []
    for lo, hi in SPIKE_WINDOWS:
        m = (x > lo) & (x <= hi)
        j = int(np.flatnonzero(m)[np.argmax(np.abs(D[m]))])
        out.append(dict(window=(lo, hi), n=int(m.sum()), rms=float(np.sqrt(np.mean(D[m] ** 2))),
                        rms_iv=iv(np.sqrt(np.mean(Db[:, m] ** 2, 1))), max=float(D[j]), at=float(x[j]),
                        max_iv=iv(Db[:, j])))
    return out


def sharpness(A):
    """the Trained RNN's and the ESN's curves (lag bins 1-3 and 18-21, every k): fall_step on the trend (the Trained
    RNN's d_H and SKL growth) and the excursions from the trend -> A[(run, m, k, b, 'step' | 'spikes')]"""
    for run, z in A["Z"].items():
        for m in ("dh", "skl"):
            for k in range(len(KS)):
                for b in (0, len(LAG_BINS) - 1):
                    y, yb = z[f"Gb_{m}"][:, k, b], z[f"Gb_{m}_boot"][:, :, k, b]
                    if run == "trained":
                        A[(run, m, k, b, "step")] = fall_step(z["x"], trend(y), trend(yb))
                    A[(run, m, k, b, "spikes")] = spikes(z["x"], y, yb)
    return A




def alternation(inc):
    """the k-alternation of the per-bit increments (..., 5 k): ((inc3 - (inc2 + inc4) / 2) - (inc4 - (inc3 + inc5) / 2))
    / 2 = (G(5) - 4 G(4) + 6 G(3) - 4 G(2) + G(1)) / 4, a quarter of the 4th difference of G; exactly 0 when
    inc2..inc5 lie on a quadratic in k (e.g. change linearly), small for smooth profiles, > 0 when the odd-numbered bits
    3 and 5 add more than their neighbours (the C(5, k) means of a count code: even k include count-preserving flip
    sets)"""
    return ((inc[..., 2] - (inc[..., 1] + inc[..., 3]) / 2) - (inc[..., 3] - (inc[..., 2] + inc[..., 4]) / 2)) / 2


def alt_crossing(x, alt, altb):
    """the alternation's early peak (largest value at 10-200 training episodes) and the training episode after which
    the curve stays below half of it (gap.crossing on the raw curve, not the trend: the switch is faster than the trend's
    5-checkpoint window; every resample uses its own early peak): dict peak, x_peak, p, lo, hi, share, boot"""
    e = (x > EARLY[0]) & (x <= EARLY[1])
    ip = np.flatnonzero(e)[np.argmax(alt[e])]
    pkb = altb[:, e].max(1)
    cb = gap.crossing(x, altb, .5 * pkb, -1.)
    lo, hi, sh = gap.interval(cb)
    return dict(peak=float(alt[ip]), peak_iv=iv(pkb), x_peak=float(x[ip]), p=gap.crossing(x, alt, .5 * alt[ip], -1.),
                lo=lo, hi=hi, share=sh, boot=cb)


def _f(v, lo=None, hi=None, d=3, sign=True):
    v, lo, hi = (None if a is None else float(a) + 0. for a in (v, lo, hi))                  # no '-0.000'
    if abs(v) < .5 * 10 ** -d:
        v = 0.
    s_ = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s_ if lo is None else s_ + (f" [{lo:+.{d}f}, {hi:+.{d}f}]" if sign else f" [{lo:.{d}f}-{hi:.{d}f}]")


def _fb(v, b, d=3, sign=True):
    """value with the 5-95% interval of its resamples"""
    lo, hi = iv(b)
    return _f(v, lo, hi, d, sign)


def _t(c, x1=None):
    """a crossing dict -> text (training episodes; open bounds 'never'); a crossing inside the first checkpoint interval
    (at or below x1) is written '< x1 (first interval)'"""
    def one(v):
        return "never" if not np.isfinite(v) else f"{v:,.0f}"
    if x1 is not None and np.isfinite(c["p"]) and c["p"] <= x1:
        return f"< {x1:,.0f} (first interval)"
    return f"{one(c['p'])} [{one(c['lo'])}-{one(c['hi'])}]" + (f" ({c['share']:.0%})" if c.get("share", 1) < 1 else "")


def _bracket(x, y, lev):
    """the checkpoint interval (x[i], x[i + 1]) that holds a stays-below crossing of y (gap.crossing, sign -1): i = the
    last checkpoint not below lev (NaN as the right end if y never stays below)"""
    i = int(np.flatnonzero(~(np.asarray(y) < lev))[-1])
    return x[i], (x[i + 1] if i + 1 < len(x) else np.nan)


def _decades(a, b):
    """log10(a) - log10(b) of two crossings with paired resamples (a never crossing: +inf): value and 5-95%"""
    if not np.isfinite(a["p"]):
        return "never"
    ab = np.where(np.isfinite(a["boot"]), a["boot"], np.inf)
    d = np.log10(ab) - np.log10(b["boot"])
    v = np.log10(a["p"]) - np.log10(b["p"])
    lo, hi = np.percentile(d, 5), np.percentile(d, 95)
    return f"{v:+.2f} [{lo:+.2f}, " + (f"{hi:+.2f}]" if np.isfinite(hi) else "open]")


def numbers(A=None):
    """every number of the docstring record"""
    A = sharpness(analysis()) if A is None else A
    Z, RS = A["Z"], A["R"]
    np.set_printoptions(linewidth=200)
    if os.path.exists(os.path.join(OUT, "checks.json")):
        with open(os.path.join(OUT, "checks.json")) as fh:
            print("== checks (checks.json)")
            for k, v in json.load(fh).items():
                print(f"  {k}: {v}")
    acc = A["acc"]
    print("== goal accuracy at T: passes naive (c0), half-way to Bayes (c50) [5-95%]")
    for run, a in acc.items():
        print(f"  {run}: c0 {_t(a['c0'])}, c50 {_t(a['c50'])}")
    print("== references (exact, float64; 1,000 probes): immediate impact chi(s, s) by k = 1..5; growth by k in the lag "
          f"bins {LAG_BINS} [5-95%]")
    for nm in ("bayes", "naive", "adf"):
        for m in ("dh", "skl"):
            I, Ib = RS[f"{nm}_I_{m}"], RS[f"{nm}_I_{m}_boot"]
            print(f"  {nm:5s} {m:3s} impact " + ", ".join(_fb(I[k], Ib[:, k], sign=False) for k in range(5)))
            for b in range(len(LAG_BINS)):
                G, Gb = RS[f"{nm}_Gb_{m}"][:, b], RS[f"{nm}_Gb_{m}_boot"][:, :, b]
                print(f"        growth lags {LAG_BINS[b][0]}-{LAG_BINS[b][1]}: " +
                      ", ".join(_fb(G[k], Gb[:, k]) for k in range(5)))
            g = RS[f"{nm}_Gb_{m}"][:, -1]
            print("        increments G(k) - G(k-1), lags 18-21: " + ", ".join(f"{v:+.3f}" for v in np.diff(np.r_[0., g])))
            if nm != "naive":
                print("        actual - linear (growth of the k-bit flip minus that of the summed single flips), lags "
                      "18-21: " + ", ".join(f"{v:+.3f}" for v in g - RS[f"{nm}_Gb_{m}_sup"][:, -1]))
    print("  count classes (d_H; growth lags 18-21 / immediate impact): 2 bits count-preserving (net 0) vs "
          "count-changing (net +-2); k2_cc = their paired difference; k3_vs_k1 = 3 bits with net +-1 minus 1 bit")
    for nm, c in A["cls_ref"].items():
        print(f"    {nm}: " + "; ".join(f"{g} {c[g + '_dh'][0, 4]:+.3f} / {c[g + '_dh'][0, 0]:.3f}"
                                     for g in ("k2_n0", "k2_n2", "k1", "k3_n1", "k3_n3")) +
              f"; k2_cc {_fb(c['k2_cc_dh'][0, 4], c['k2_cc_dh_boot'][:, 0, 4])} (impact "
              f"{_fb(c['k2_cc_dh'][0, 0], c['k2_cc_dh_boot'][:, 0, 0])}); k3_vs_k1 "
              f"{_fb(c['k3_vs_k1_dh'][0, 4], c['k3_vs_k1_dh_boot'][:, 0, 4])}")
    for run in RUNS:
        z = Z[run]
        x = z["x"]
        print(f"== {NAME[run]} ({len(x)}/{z['n_total']} checkpoints): stages")
        for raw in STAGES:
            if raw not in z["raw"]:
                continue
            i = int(np.flatnonzero(z["raw"] == raw)[0])
            print(f"  -- {x[i]:,.0f} training episodes")
            for m in ("dh", "skl"):
                I, Ib = z[f"I_{m}"][i], z[f"I_{m}_boot"][:, i]
                print(f"    {m:3s} impact " + ", ".join(_fb(I[k], Ib[:, k], sign=False) for k in range(5)))
                for b in range(len(LAG_BINS)):
                    print(f"        growth lags {LAG_BINS[b][0]}-{LAG_BINS[b][1]}: " +
                          ", ".join(_fb(z[f'Gb_{m}'][i, k, b], z[f'Gb_{m}_boot'][:, i, k, b]) for k in range(5)))
            ks = A[(run, "kstr")]
            print("        d_H increments G(k) - G(k-1), lags 18-21: " +
                  ", ".join(_fb(ks["inc"][i, 3, k], ks["inc_boot"][:, i, 3, k]) for k in range(5)) +
                  f"; largest change from one k to the next (k = 2..5): {_fb(ks['kmax'][i, 3], ks['kmax_boot'][:, i, 3])}"
                  f"; alternation {_fb(alternation(ks['inc'][i, 3]), alternation(ks['inc_boot'][:, i, 3]))}")
            sup, supb = A[(run, "sup")]
            print("        d_H actual - linear, lags 18-21, k = 1..5: " +
                  ", ".join(_fb(-sup[i, k, 3], -supb[:, i, k, 3]) for k in range(5)))
            print("        d_H growth of the linear response (summed single-flip changes), lags 18-21, increments: " +
                  ", ".join(f"{v:+.3f}" for v in np.diff(np.r_[0., z['Gb_dh_sup'][i, :, 3]])))
        i_end = len(x) - 1
        pv = per_variable_end(run, z["raw"][i_end])
        print(f"  -- per variable and per role at the end ({x[i_end]:,.0f}), d_H growth lags 18-21, k = 1 / 5 (goal = "
              f"variable 0 in {pv['goal_share_r0']:.0%} of the probes):")
        for nm in ("r0", "r1", "r0 - r1", "goal", "context", "goal - context"):
            v, bt = pv[nm]
            print(f"     {nm}: k=1 {_fb(v[0], bt[:, 0])}, k=5 {_fb(v[4], bt[:, 4])}")
        for raw in (400, z["raw"][-1]):
            i = int(np.flatnonzero(z["raw"] == raw)[0])
            v, bt = per_step(run, raw)
            print(f"  -- per flip step at {x[i]:,.0f}, d_H growth lags 18-21, k = 1 / 5: " +
                  "; ".join(f"s={s}: {_fb(v[a, 0], bt[:, a, 0], 2)} / {_fb(v[a, 4], bt[:, a, 4], 2)}"
                            for a, s in enumerate(FLIP_S)))
        if run == "trained":
            for raw in (1100, 2100):
                i = int(np.flatnonzero(z["raw"] == raw)[0])
                pv = per_variable_end(run, raw)
                v, bt = pv["r0 - r1"]
                print(f"  -- r0 - r1 at {x[i]:,.0f}: k=1 {_fb(v[0], bt[:, 0])}, k=5 {_fb(v[4], bt[:, 4])}")
    # k-structure at every checkpoint
    z = Z["trained"]
    x = z["x"]
    ks = A[("trained", "kstr")]
    print("== Trained RNN k-structure at every checkpoint (C(5, k) means, d_H growth)")
    for b in (3, 0):
        alt, altb = alternation(ks["inc"][:, b]), alternation(ks["inc_boot"][:, :, b])
        print(f"  lags {LAG_BINS[b][0]}-{LAG_BINS[b][1]}: alternation at 10-100: " +
              "; ".join(f"{x[i]:,.0f} {_fb(alt[i], altb[:, i])}" for i in range(6)))
        ac = alt_crossing(x, alt, altb)
        xa, xb = _bracket(x, alt, .5 * ac["peak"])
        print(f"     early peak {_f(ac['peak'], *ac['peak_iv'])} at {x[np.flatnonzero(x == ac['x_peak'])[0]]:,.0f}; below "
              f"half of it for good between {xa:,.0f} and {xb:,.0f} (one checkpoint interval; the point "
              f"log-interpolated inside it, {_t(ac)}, is finer than the grid resolves)")
        late = np.flatnonzero(x >= 80)
        j = late[np.argmax(np.abs(alt[late]))]
        print(f"     from 80 on: max |alternation| {_fb(alt[j], altb[:, j])} at {x[j]:,.0f}")
        print("     largest change of the increment from one k to the next: " +
              "; ".join(f"{x[i]:,.0f} {_fb(ks['kmax'][i, b], ks['kmax_boot'][:, i, b])}" for i in range(1, 4)))
        j = late[np.argmax(ks["kmax"][late, b])]
        n073 = int(np.sum(ks["kmax"][1:, b] > .073))
        print(f"     from 80 on: largest {_fb(ks['kmax'][j, b], ks['kmax_boot'][:, j, b])} at {x[j]:,.0f}; "
              f"checkpoints (of {len(x) - 1} trained) above the old claim's .073: {n073}")
    # count code
    print("== count code (per flip set before the C(5, k) mean): d_H at lags 18-21 [5-95%]; impact = chi(s, s)")
    for run, c in A["cls"].items():
        xc = c["x"]
        print(f"  {run} ({len(xc)} checkpoints): x: count-preserving 2-bit growth / count-changing 2-bit growth | "
              "k2_cc (paired: the extra growth from changing the count) | k3_vs_k1 (paired) | impact preserving / "
              "changing")
        for i in range(len(xc)):
            if xc[i] > 400 and xc[i] not in (560, 880, 1200, 1680, 2400, 4200, 9600, 20000, 39996):
                continue
            print(f"     {xc[i]:6,.0f}: {_fb(c['k2_n0_dh'][i, 4], c['k2_n0_dh_boot'][:, i, 4])} / "
                  f"{_fb(c['k2_n2_dh'][i, 4], c['k2_n2_dh_boot'][:, i, 4])} | "
                  f"{_fb(c['k2_cc_dh'][i, 4], c['k2_cc_dh_boot'][:, i, 4])} | "
                  f"{_fb(c['k3_vs_k1_dh'][i, 4], c['k3_vs_k1_dh_boot'][:, i, 4])} | "
                  f"{c['k2_n0_dh'][i, 0]:.3f} / {c['k2_n2_dh'][i, 0]:.3f}")
        y, yb = c["k2_cc_dh"][:, 4], c["k2_cc_dh_boot"][:, :, 4]
        e = (xc > EARLY[0]) & (xc <= EARLY[1])
        ip = np.flatnonzero(e)[np.argmax(y[e])]
        print(f"     early peak of k2_cc (10-200): {_fb(y[ip], yb[:, ip])} at {xc[ip]:,.0f}; the 4 checkpoints from it on: " +
              ", ".join(f"{xc[j]:,.0f} {y[j]:+.3f}" for j in range(ip, min(ip + 4, len(xc)))))
        j1 = ip + int(np.argmax(np.diff(-y[ip:ip + 6])))                          # the largest one-interval drop
        print(f"     largest one-interval drop within 5 intervals of the early peak: {xc[j1]:,.0f}->{xc[j1 + 1]:,.0f} "
              f"{_fb(y[j1] - y[j1 + 1], yb[:, j1] - yb[:, j1 + 1])} nats")
        below = np.flatnonzero(y[ip:] < .1 * y[ip])
        i10 = ip + int(below[0]) if len(below) else None                          # first reached (not for good)
        if i10 is None:
            print("     never below 10% of the early peak")
            later = np.flatnonzero(xc > xc[ip + 1])
        else:
            drops = [(xc[j], xc[j + 1], y[j] - y[j + 1], yb[:, j] - yb[:, j + 1]) for j in range(ip, i10)]
            tot, totb = y[ip] - y[i10], yb[:, ip] - yb[:, i10]
            print(f"     collapse from the early peak to the first checkpoint below 10% of it ({xc[i10]:,.0f}): drops "
                  "per interval (nats, share of the total): " + "; ".join(
                      f"{a_:,.0f}->{b_:,.0f} {_fb(d_, db_)} ({_fb(d_ / tot, db_ / totb, 2, False)})"
                      for a_, b_, d_, db_ in drops) + f"; total {_fb(tot, totb)}")
            later = np.flatnonzero(xc > xc[i10])
        if len(later):
            j = later[np.argmax(y[later])]
            print(f"     after that: largest k2_cc {_fb(y[j], yb[:, j])} at {xc[j]:,.0f} (count-preserving growth there "
                  f"{c['k2_n0_dh'][j, 4]:+.3f}); end {_fb(y[-1], yb[:, -1])}")
        if run == "trained":
            for g in ("k1", "k3_n1", "k3_n3", "k4_n0", "k4_n2", "k4_n4", "k5_n1", "k5_n3", "k5_n5"):
                print(f"     {g} at 20 / 40 / 100: " + " / ".join(f"{c[g + '_dh'][j, 4]:+.3f}" for j in (1, 2, 5)))
    # superposition through training
    print("== actual - linear (d_H growth of the k-bit flip minus that of the summed single-flip changes, lags 18-21) "
          "through training (Trained RNN): sign changes of the curve; above 0 (more than linear) for good (trend)")
    sup, supb = A[("trained", "sup")]
    for k in range(1, 5):
        d = -sup[:, k, 3]
        ch = np.flatnonzero(np.diff(np.sign(d)) != 0)
        print(f"  k={k + 1}: sign changes between " + ", ".join(f"{x[j]:,.0f}-{x[j + 1]:,.0f}" for j in ch) +
              f"; above 0 for good {_t(A[('trained', 'sup_neg', k)])}; |value| <= {np.abs(d[1:6]).max():.3f} at 20-100"
              f", <= {np.abs(d[1:3]).max():.3f} at 20-40")
        sel = [j for j in range(len(x)) if x[j] in (20, 40, 100, 160, 300, 640, 880, 1040, 1680, 4200, 39996)]
        print("      " + "; ".join(f"{x[j]:,.0f} {_fb(d[j], -supb[:, j, k, 3], 2)}" for j in sel))
    # plateau
    print(f"== plateau: d_H growth, lags 18-21, mean over the checkpoints in {PLATEAU[0]} and {PLATEAU[1]}")
    pl = A[("trained", "plateau")]
    for k in range(5):
        (a_, na), (b_, nb), (ab, _), (bb, _) = pl[k]
        print(f"  k={k + 1}: {a_:.3f} ({na} checkpoints) -> {b_:.3f} ({nb}): change {_fb(b_ - a_, bb - ab)}")
    for k in (2, 4):
        dv = (pl[k][1][0] - pl[k][0][0]) - (pl[0][1][0] - pl[0][0][0])
        db = (pl[k][3][0] - pl[k][2][0]) - (pl[0][3][0] - pl[0][2][0])
        print(f"  paired (k={k + 1} change) - (k=1 change): {_fb(dv, db)}")
    # transients
    for run in RUNS:
        z = Z[run]
        x = z["x"]
        print(f"== {NAME[run]} through training (per k; transient() on the trend: peak = max of its 3-checkpoint running "
              "mean)")
        for m in ("dh", "skl"):
            for b in range(len(LAG_BINS)):
                print(f"    {m} lags {LAG_BINS[b][0]}-{LAG_BINS[b][1]}:")
                for k in range(5):
                    t = A[(run, m, k, b, "tr")]
                    line = (f"      k={k + 1}: peak {_f(t['peak'], t['peak_lo'], t['peak_hi'], sign=False)} at "
                            f"{t['x_peak']:,.0f}; rise 25/50/75% {_t(t['rise25'], x[1])} / {_t(t['rise50'], x[1])} / "
                            f"{_t(t['rise75'], x[1])}; fall 75/50/25% {_t(t['fall75'])} / {_t(t['fall50'])} / "
                            f"{_t(t['fall25'])}; end {_f(t['end'], t['end_lo'], t['end_hi'])}")
                    if m == "skl":
                        zc = A[(run, "skl", k, b, "zero")]
                        line += f"; below 0 for good {_t(zc)}"
                        if run == "trained":
                            line += (f" ({order(zc, acc['trained']['c50'])} the accuracy c50; vs c0: "
                                     f"{order(zc, acc['trained']['c0'])})")
                    print(line)
        if run == "trained":
            for k in range(5):
                print(f"    SKL zero, k={k + 1}: lags 18-21 vs 12-15: "
                      f"{order(A[(run, 'skl', k, 3, 'zero')], A[(run, 'skl', k, 2, 'zero')])}; 12-15 vs 6-9: "
                      f"{order(A[(run, 'skl', k, 2, 'zero')], A[(run, 'skl', k, 1, 'zero')])}; 6-9 vs 1-3: "
                      f"{order(A[(run, 'skl', k, 1, 'zero')], A[(run, 'skl', k, 0, 'zero')])}")
        a2 = A[(run, "lagacc")]
        mx = a2.max(-1)
        j, k = np.unravel_index(np.argmax(mx), mx.shape)
        print(f"  lag concavity, every checkpoint: largest second difference over lags 1-20 {mx.max():+.3f} at "
              f"{x[j]:,.0f}, k = {k + 1}, lag {np.argmax(a2[j, k]) + 1}; checkpoints above +.02: "
              f"{[f'{v:,.0f}' for v in x[mx.max(1) > .02]]}; at the 4 old stages <= "
              f"{max(mx[np.flatnonzero(z['raw'] == r)[0]].max() for r in (400, 1100, 2100, 49995)):+.3f}")
        G = z["G_dh"]
        print(f"  d_H growth range over all checkpoints and k, lags 1-21: {G[:, :, 1:].min():+.3f} to {G[:, :, 1:].max():+.3f}")
    # sharpness
    print("== sharpness through training (trend = running median over 5 checkpoints): (i) the largest step of the trend "
          "between adjacent checkpoints while it falls from 75% to 25% of its peak (for good), as a share of that drop "
          "[5-95%]; (ii) single-checkpoint excursions = curve - trend (nats): RMS per training window and the largest")
    for run in RUNS:
        for b in (0, len(LAG_BINS) - 1):
            for k in (0, len(KS) - 1):
                lab = f"  {NAME[run]} d_H lags {LAG_BINS[b][0]}-{LAG_BINS[b][1]} k={k + 1}:"
                if run == "trained":
                    f = A[(run, "dh", k, b, "step")]
                    lab += (f" largest step {_f(f['share'], *f['share_iv'], 3, False)} of the 75 -> 25%-of-peak "
                            f"decline (between {f['at'][0]:,.0f} and {f['at'][1]:,.0f}; {f['n_int']} checkpoint "
                            "intervals);")
                sp = A[(run, "dh", k, b, "spikes")]
                lab += " excursions " + "; ".join(
                    f"{w['window'][0]:,}-{w['window'][1]:,} ({w['n']}): RMS {w['rms']:.3f} [{w['rms_iv'][0]:.3f}-"
                    f"{w['rms_iv'][1]:.3f}], largest {w['max']:+.3f} at {w['at']:,.0f}" for w in sp)
                print(lab)
    ph = excursion_phase(Z["trained"])
    print("  excursions at 160-400 (k = 5, lags 18-21) by raw episode mod 75: " + "; ".join(
        f"{p_}: n {v['n']}, mean {_f(v['mean'], *v['iv'])} (values " + ", ".join(f"{q:+.2f}" for q in v["vals"]) + ")"
        for p_, v in ph.items()))
    op = online_period()
    print(f"  per-update logs, training episodes 160-400 ({op['n_updates']} updates), cubic trend removed: " + "; ".join(
        f"{k_}: best period {v['best']} (within 55-70: {v['best_in_band']}), AR(1) surrogates with at least its 55-70 "
        f"amplitude {v['p_band']:.3f} (lag-1 r {v['r1']:+.2f})" for k_, v in op.items() if k_ != "n_updates"))
    # ESN extras
    ze = Z["echo"]
    ie = len(ze["x"]) - 1
    print("== ESN at the end: SKL growth lags 1-3, k = 1 / 5: "
          f"{_fb(ze['Gb_skl'][ie, 0, 0], ze['Gb_skl_boot'][:, ie, 0, 0])} / "
          f"{_fb(ze['Gb_skl'][ie, 4, 0], ze['Gb_skl_boot'][:, ie, 4, 0])}; min over checkpoints of lags 1-3 SKL growth "
          f"(k = 1..5): {ze['Gb_skl'][:, :, 0].min():+.4f}; Trained RNN end "
          f"{_fb(Z['trained']['Gb_skl'][-1, 0, 0], Z['trained']['Gb_skl_boot'][:, -1, 0, 0])} / "
          f"{_fb(Z['trained']['Gb_skl'][-1, 4, 0], Z['trained']['Gb_skl_boot'][:, -1, 4, 0])}")
    dd = np.diff(ze["Gb_dh"][:, :, -1], axis=0)
    print(f"  ESN d_H growth lags 18-21: largest decrease between adjacent checkpoints {dd.min():+.4f}")
    # width runs
    print("== replication in the width runs (seed 20261029, same world): d_H growth lags 18-21 on the trend; fall50 - c0 "
          "in decades (paired resamples)")
    for run in ("trained",) + WIDTH_RUNS:
        z = Z[run]
        a = acc[run]
        print(f"  {run}: training ends {np.log10(z['x'][-1] / a['c0']['p']):+.2f} decades after its c0")
        for k in (0, 4):
            t = A[(run, "dh", k, 3, "tr")]
            print(f"  {run} k={k + 1}: rise 50% {_t(t['rise50'], z['x'][1])}; peak "
                  f"{_f(t['peak'], t['peak_lo'], t['peak_hi'], 2, False)} at {t['x_peak']:,.0f}; fall 50% "
                  f"{_t(t['fall50'])}; end {_f(t['end'], t['end_lo'], t['end_hi'], 2)}; fall50 - c0 "
                  f"{_decades(t['fall50'], a['c0'])} decades ({order(t['fall50'], a['c0'])} c0, "
                  f"{order(t['fall50'], a['c50'])} c50)")
        ks = A[(run, "kstr")]
        alt, altb = alternation(ks["inc"][:, 3]), alternation(ks["inc_boot"][:, :, 3])
        ac = alt_crossing(z["x"], alt, altb)
        late = np.flatnonzero(z["x"] > ac["boot"][np.isfinite(ac["boot"])].max()) if np.isfinite(ac["p"]) else []
        xa, xb = _bracket(z["x"], alt, .5 * ac["peak"])
        print(f"     alternation (lags 18-21): early peak {_f(ac['peak'], *ac['peak_iv'])} at {ac['x_peak']:,.0f}; "
              f"below half of it for good between {xa:,.0f} and {xb:,.0f} (one checkpoint interval; log-interpolated "
              f"inside it {_t(ac)}, finer than the grid resolves); largest |value| after that "
              f"{np.abs(alt[late]).max() if len(late) else np.nan:.3f}; first checkpoints: " +
              ", ".join(f"{z['x'][i]:,.0f} {alt[i]:+.3f}" for i in range(1, 9)))
    return A


COMBINED_LAYOUT = """combined.npz (td_tt_flips.export; float32 unless noted; [lo, hi] = 5-95% over the 1,000 shared environment
resamples, td_tt_gap.boot_counts()):
  ks (5,) bits flipped k = 1..5; lag_bins (4, 2) John's lag bins 1-3, 6-9, 12-15, 18-21 (inclusive; each the mean over
    its lags); flip_steps (5,) s = 1, 3, 5, 7, 9; nb_index (5,) str: index 0 of the NB axis = the immediate impact
    chi(s, s) (lag 0), 1..4 = the growth in the lag bins
  for <run> in trained (Trained RNN, N = 1000), echo (ESN), trained_p1, trained_N200, trained_N100, trained_N50:
    <run>_x (K,) float64 training episodes (initial network at 10); <run>_raw (K,) raw checkpoint episodes
    <run>_growth_<m>, _lo, _hi (K, 5 k, 4 bins), m = dh (Hilbert metric d_H) / skl (symmetric KL): growth
      chi(s + lag, s) - chi(s, s) of a k-bit flip (mean over the C(5, k) flip sets, flip steps, variables, probes)
    <run>_impact_<m>, _lo, _hi (K, 5 k): chi(s, s)
    <run>_actual_minus_linear_dh, _lo, _hi (K, 5 k, 4 bins): the growth minus the growth of the summed single-flip
      changes (> 0: several flips act more than linearly; < 0: less)
    <run>_alternation, _lo, _hi (K, 4 bins): the k-alternation of the per-bit increments (alternation(): a quarter of
      the 4th difference of G over k = 1..5; 0 when the increments over k = 2..5 lie on a quadratic in k, small for
      smooth profiles, > 0 under a count code)
    <run>_cc_x (Kc,) training episodes of the count-class files (= <run>_x once every checkpoint is harvested)
    <run>_count_index, _lo, _hi (Kc, NB): the COUNT-CODE INDEX = k2_cc in d_H, the extra growth from changing the
      count: per probe and flip step, the mean over the 2-bit flip sets that change the number of ones (net +-2) minus
      the mean over those that keep it (net 0: one 0 -> 1, one 1 -> 0), averaged over (probe, flip step) units; NB axis
      as above (index 4 = lags 18-21). It marks a count code only where <run>_count_preserving is itself ~0 (20-40
      episodes in the Trained RNN); where both classes grow strongly (100-880 episodes) it is a partial count
      dependence, not a code
    <run>_count_preserving, _lo, _hi and <run>_count_changing, _lo, _hi (Kc, NB): the two class means themselves
      (k = 2, net 0 / net +-2; d_H)
    <run>_k3_vs_k1, _lo, _hi (Kc, NB): 3-bit flips with net +-1 minus 1-bit flips (paired; d_H): ~0 under a count code
    <run>_c0, _c0_lo, _c0_hi, <run>_c50, ... (scalars, NaN = never; not for echo): the goal accuracy at T passes naive /
      is half-way to Bayes for good
  for <obs> in bayes, naive, adf (factorized-state Bayes): exact, float64 sources, the same flips and probes:
    <obs>_growth_<m>, _lo, _hi (5 k, 4 bins); <obs>_impact_<m> (5 k); <obs>_actual_minus_linear_dh (5 k, 4 bins);
    <obs>_count_index, <obs>_count_preserving, <obs>_count_changing, <obs>_k3_vs_k1 (NB,)"""


def export(A=None):
    """the combined figure's arrays -> OUT/combined.npz (layout: COMBINED_LAYOUT, also stored as 'layout')"""
    A = sharpness(analysis()) if A is None else A
    Z, RS = A["Z"], A["R"]
    f32 = np.float32
    out = dict(layout=np.array(COMBINED_LAYOUT), ks=np.array(KS), lag_bins=np.array(LAG_BINS),
               flip_steps=np.array(FLIP_S), nb_index=np.array(["impact (lag 0)"] + [f"growth lags {a}-{b}"
                                                                                     for a, b in LAG_BINS]))

    def put(key, v, b=None):
        out[key] = np.asarray(v, f32)
        if b is not None:
            lo, hi = iv(b)
            out[key + "_lo"], out[key + "_hi"] = np.asarray(lo, f32), np.asarray(hi, f32)
    for run, z in Z.items():
        out[f"{run}_x"], out[f"{run}_raw"] = z["x"], z["raw"]
        for m in ("dh", "skl"):
            put(f"{run}_growth_{m}", z[f"Gb_{m}"], z[f"Gb_{m}_boot"])
            put(f"{run}_impact_{m}", z[f"I_{m}"], z[f"I_{m}_boot"])
        sup, supb = A[(run, "sup")]
        put(f"{run}_actual_minus_linear_dh", -sup, -supb)
        ks = A[(run, "kstr")]
        put(f"{run}_alternation", alternation(ks["inc"]), alternation(ks["inc_boot"]))
        c = A["cls"].get(run)
        if c is not None:
            out[f"{run}_cc_x"] = c["x"]
            for key, g in (("count_index", "k2_cc"), ("count_preserving", "k2_n0"), ("count_changing", "k2_n2"),
                           ("k3_vs_k1", "k3_vs_k1")):
                put(f"{run}_{key}", c[f"{g}_dh"], c[f"{g}_dh_boot"])
        if run in A["acc"]:
            for cc in ("c0", "c50"):
                for f_ in ("p", "lo", "hi"):
                    out[f"{run}_{cc}" + ("" if f_ == "p" else f"_{f_}")] = float(A["acc"][run][cc][f_])
    for nm in ("bayes", "naive", "adf"):
        for m in ("dh", "skl"):
            put(f"{nm}_growth_{m}", RS[f"{nm}_Gb_{m}"], RS[f"{nm}_Gb_{m}_boot"])
            put(f"{nm}_impact_{m}", RS[f"{nm}_I_{m}"])
        put(f"{nm}_actual_minus_linear_dh", RS[f"{nm}_Gb_dh"] - RS[f"{nm}_Gb_dh_sup"])
        c = A["cls_ref"][nm]
        for key, g in (("count_index", "k2_cc"), ("count_preserving", "k2_n0"), ("count_changing", "k2_n2"),
                       ("k3_vs_k1", "k3_vs_k1")):
            put(f"{nm}_{key}", c[f"{g}_dh"][0])
    path = os.path.join(OUT, "combined.npz")
    np.savez_compressed(path, **out)
    print("saved", path, f"({os.path.getsize(path) / 2 ** 20:.2f} MB, {len(out)} arrays)")
    return path


if __name__ == "__main__":
    what = sys.argv[1:] or ["numbers"]
    if "harvest" in what:
        harvest(thin_echo="thin_echo" in what)
    if "classes" in what:
        harvest_classes()
    if "refs" in what:
        refs_build()
    if "refs_classes" in what:
        refs_classes()
    if "checks" in what:
        checks()
    if {"numbers", "figure", "export"} & set(what):
        A_ = sharpness(analysis())
        if "numbers" in what:
            numbers(A_)
        if "figure" in what:
            figure(A=A_)
        if "export" in what:
            export(A_)
