"""The combined chaos figure (John 11-02; ratios John 11-03): pairs of marginal beliefs given the same observation (40
bins; three definitions of the distance; the split by the observation's entanglement) and a flipped observation's
effect through training (single flips; k-bit flips by lag bin; count classes), in one figure. Every change of a
distance is drawn as the ratio after / before.

John 11-02 (verbatim): "the chaos pairs is nice, can you please split it from 20 to 40 bins. Also do you do this for
both variable marginals seperately or use the average distance? It would be nice to try different ways of doing it.
It would be interesting to see if this changes when both variable marganals are close versus only one. Also instead of
splitting by fr, can you please split by the entanglement of the observation. And let's combine the chaos pairs and
chaos pairs fr (now entanglement) into 1 figure (for the entanglement split we can just show different averages as
curves on a single panel per agent). Chaos sweep d look extremely interesting, let's put that into this figure as well.
I wonder if we would see a phase transition with variables 'lag' and '# bits flipped'. If we need more data at higher
lags we can pool lags over bins of 3 lags: 1-3, 6-9, 12-15, 18-21."
John 11-03 (verbatim): "Can you please replace the 2nd row of this figure to be the ratio of the distances, instead of
the gaps, as a function of the pre-observation distance", then "Can we repeat this change for all the belief distance
gaps, switching from distance to ratio?"

-> main/REPLICATION/figure_bin/td_tt_chaos_pairs.png (figure()). It takes the name of the 20-bin pairs figure and
   replaces the FR figure td_tt_chaos_pairs_fr.png, which figure() deletes. td_tt_chaos_pairs.py's `figure` now writes
   its 20-bin figure as td_tt_chaos_pairs_detail.png, and its `fr` only prints a note, so neither overwrites this
   figure or re-creates the FR figure. The difference version of 11-02 (Delta d_H, growth in nats) is in git history;
   its numbers stay in td_tt_chaos_pairs.py, td_tt_flips.py and td_tt_chaos.py.
SOURCES. Caches of three modules and three of this module's own:
  - td_tt_chaos_pairs.py `build` / `combined` -> through_training/chaos/pairs_v2.npz (per case: d_H and SKL before /
    after), pairs_combined.npz (40 bins; row 1's violins; the difference contrasts, used by checks), pairs_shuf40.npz
    (the split groups: within-cell thirds, entanglement fifths) and pairs_lq.npz (the marginal log beliefs).
  - ratio_build() (`ratios`; CPU, about 25 s) -> chaos/pairs_ratio.npz (rows 2-4): per agent and case ln r = ln(d_H
    after / d_H before) and ln(SKL after / SKL before); the shuffled ln r, re-drawn with td_tt_chaos_pairs.combined's
    groups and seeds (the log of each of the 5 draws, then their mean; pairs_shuf40 stores the draws' mean d_H after,
    whose log is larger by Jensen); per bin and split the means of ln r with the 1,000 shared resamples, the matched
    contrasts and the closest-pair summaries (td_tt_chaos_pairs.bin_stats / step_lines / matched / common_hilo / _ratio
    applied to ln r); KDE violins of log10 r (td_tt_chaos_pairs.violin_stats).
  - fratio_harvest() (`fratio-harvest part nparts`; one GPU per process, about 4 s per checkpoint, 114 checkpoints)
    -> through_training/flips/ratio/trained_EEEEE.npz (row 5): td_tt_flips.net_flips's subset-resolved flips (31 flip
    sets x s = 1, 3, 5, 7, 9 x lags 0-21 x 2 variables, 1,000 probes), chi = d_H per flip set, lag and variable, reduced
    to environment sums of log ratios (_fratio_reduce). fratio_refs() (`fratio-refs`; CPU, about 25 s) -> flips/ratio/
    refs.npz: Bayes, naive and factorized-state Bayes on the same flips (exact_set_chi = td_tt_flips.exact_flips's
    computation kept per flip set).
  - td_tt_flips.py `export` -> flips/combined.npz: the checkpoints, lag bins and accuracy crossings (and checks).
  - td_tt_chaos.py `sweep` / `pilot` -> chaos/sweep, chaos/pilot/bayes_side.npz: checks only (the difference
    version's q).

RATIOS (John 11-03). Each case's (or flip set's) ratio is taken first; the log ratios are then averaged exactly as the
  differences were (the same units, cells, weights and resamples), and a drawn value is exp(mean ln ratio), the
  geometric mean. Log axes, labelled with the ratios. 1 = no change, > 1 = the distance grows. Naive's ratio is 1 in
  every pair and for every flip (T1 of td_tt_chaos.py: a shared observation, or a flip, shifts the log beliefs by a
  fixed vector).

WHAT IS DRAWN (pairs: end of training, raw checkpoint 49995 = 39,996 training episodes; one seed per network)
  Pairs: two probes of the same test environment and step t (2-30) with the identical 5-observation vector at t and
    different observation counts before t. A case = one variable of a pair-step: 1,710,703 pair-steps x 2 variables =
    3,421,406 cases per agent. d_H(p, q) = max_r log(p_r / q_r) - min_r log(p_r / q_r) per variable between the pair's
    marginal beliefs (Hilbert's projective metric; the range of the log-ratio; it can be set by values both beliefs
    treat as near-impossible); before = after step t - 1, after = after step t, r = after / before. SKL(p, q) =
    Sigma_r (p_r - q_r) log(p_r / q_r) / 2 (the bulk, values weighted by their probability).
  40 bins of d_H before: log-spaced over the pooled 0.1st-99th percentiles of the four agents' cases (0.250-118.9
    nats, each bin x1.167). Cases inside the bins: Bayes 3,390,188 (below 102, above 31,116), naive 3,388,427 (10,811
    / 22,168), Trained RNN 3,420,686 (289 / 431), ESN 3,335,784 (2,480 / 83,142); fewest per bin 130 / 2,105 / 189 /
    930. Rows 2-4 use the cases inside the bins (d_H before >= 0.25 nats), texts included.
  a-d  violins of log10 d_H after per bin (pairs_combined vio_*: Gaussian KDE, Scott's rule, on at most 20,000 cases
    per bin, seeded subsample, evaluated from the subsample's lowest to its highest case, each violin scaled to its own
    maximum); white dots: medians of all cases in the bin; dashed: after = before. Naive's lie on it exactly (T1).
    y range: from .07 below the lowest point of any violin (no violin is cut) to the highest of the four agents' 99.9th
    percentiles of log10 d_H after (+2.080) + .12.
  e-g  violins of log10 r per bin (pairs_ratio vio_*: the same KDE, subsample seed RATIO_SEED); white dots: medians of
    all cases in the bin; line: the geometric mean per bin (agent colour, outlined); dark red dashed: the shuffled
    step's (the step's difference log p_t - log p_{t-1} - (log q_t - log q_{t-1}) of a donor case of the same variable,
    entanglement fifth, within-cell third of the other variable's distance and bin, with a random sign); grey: naive, 1
    in every pair. y range: .08 below the three agents' lowest per-bin .5th percentile of log10 r to .1 above the
    highest 99.5th (x0.32 to x18.6); the violins' extreme tails are cut. Texts: geometric mean [5-95%], shuffled, the
    share with r > 1, and where actual / shuffled (paired per case) is above 1 (5% > 1).
  h  the geometric mean of SKL after / SKL before per bin of d_H before, all four agents, 5-95% bands.
  i-k  geometric means per bin with 5-95% bands, four definitions of "close": one variable, all pairs (thin; e-g's
    line); the pair's average (dashed; a unit = a pair-step, x = the mean of its two variables' d_H before, r = the mean
    after / the mean before = the ratio of the d_H between the pair's products of marginals over the 100 cells); the
    other variable close / far (dotted / dash-dot): the lowest / highest third of the other variable's d_H before within
    each cell (step t, bin of the own d_H before), so matched on step and own distance by construction. Drawn at bins
    where the group holds >= 100 cases. Texts: far / close third matched on step and bin; the closest pairs inside the
    bins: pair-steps whose average d_H before is 0.25-1 nat, single variables at 0.25-1 nat.
  l  the thirds' geometric means matched on step and own bin (td_tt_chaos_pairs.matched: per cell (t, bin) with >= 20
    cases in every third, the third's mean of ln r; cells averaged with their case counts as weights).
  m-o  the step-standardised fifth lines (td_tt_chaos_pairs.step_lines on ln r: per bin, the cells (t, bin) with >= 20
    cases in every fifth, each cell's fifth means weighted by the cell's count); drawn where those cells hold >= 500
    cases (from 0.68 / 0.58 / 0.32 nats). Entanglement of the observation: probes.npz 'ent' at step t (per variable the
    symmetric KL between naive's step likelihood and Bayes's effective step update, each normalised over the 10 values;
    the locked definition), the mean of the two variables, then of the two probes: one value per pair-step. Fifths over
    the pair-steps (edges .006 / .051 / .103 / .177 / .313 / 2.512), the same for every agent. Text: most / least
    fifth, matched on step and bin.
  p  the fifths' matched geometric means, as l.
  q  single flips (each of the 5 observations alone; td_tt_flips's flips at s = 1, 3, 5, 7, 9, each leaving lag 21
    within the episode; the chaos sweep stored only the mean over the 5 observations, so its flips at s = 1, 3, 5, 10
    could not give per-flip ratios): growth factor chi(s + lag, s) / chi(s, s) per flip set, step and variable, chi =
    d_H between the intact and the flipped marginals; log ratios averaged over the 5 sets, the 5 flip steps and the 2
    variables per probe, then environment means over 1,000 probes (20 per test environment) with the shared resamples.
    All 114 checkpoints; lags 1, 2, 5, 10, 20. Bayes and factorized-state Bayes at lag 20 (black, solid / dash-dot;
    exact float64 on the same probes and flips); naive 1. Yellow bands: the Trained RNN's goal accuracy at T passes
    naive / is half-way to Bayes for good (td_tt_gap 'jumps'; 5-95%).
  r  every subset of the 5 observations flipped at once (31 flip sets) at the same steps; per flip set the mean over
    the lag bin's lags of ln chi(s + lag, s) / chi(s, s), then the mean over the C(5, k) sets of each k, the flip steps,
    both variables and the probes; John's lag bins. The Trained RNN at 4 of the 114 checkpoints (20, 100, 880, 39,996
    training episodes), then Bayes and factorized-state Bayes (exact); naive 1. Shared y axis.
  s  count classes: 2-bit flip sets that change the number of 1s by 2 (both 0->1 or both 1->0) and that keep it (one
    0->1, one 1->0); class geometric means of the growth factor at lags 18-21 with 5-95% bands; the exact observers in
    the text. The note: the immediate impact chi(s, s), change / keep, paired per (probe, flip step) (td_tt_flips's
    k2_cc index on ln chi(s, s)).
  Intervals everywhere: 5-95% over the 1,000 shared resamples of the 50 test environments (td_tt_gap.boot_counts();
    ratios of count-weighted environment sums). They do not include variation across training seeds.

RESULTS (numbers(); ratios = geometric means; [5-95%]; Bayes / Trained RNN / ESN unless named)
  1 The distance ratio (e-h). d_H after / before x1.030 [x1.029, x1.031] / x1.023 [x1.022, x1.025] / x1.064 [x1.063,
    x1.065]; shuffled x1.011 / x1.010 / x1.001; naive 1 in every pair. It grows in .706 / .606 / .835 of the cases.
    The factor falls with the distance before: at 0.27 / 0.50 / 0.93 / 2.0 / 3.2 / 9.4 / 32 / 94 nats Bayes x1.59 /
    1.83 / 1.46 / 1.15 / 1.07 / 1.03 / 1.02 / 1.015, Trained RNN x2.22 / 1.80 / 1.44 / 1.14 / 1.05 / 1.02 / 1.01 /
    1.005, ESN x1.55 / 1.40 / 1.29 / 1.19 / 1.14 / 1.06 / 1.03 / 1.02. Against the shuffled step (paired per case):
    Bayes's and the Trained RNN's factors are not resolved from it below 1.17 nats (at 0.50 nats x1.02 [x0.92, x1.13] /
    x0.99 [x0.90, x1.09]) and exceed it from 1.17 nats on (x1.01-1.04; the Trained RNN's last bin not resolved); the
    ESN's exceeds it in every bin (x1.12 [x1.05, x1.18] at 0.27 nats, x1.21 at 0.50, x1.06 at 9.35, x1.02 at 94). So
    the close pairs' large factors (x1.4-2.2 below 1 nat) are what a step difference of that size does to so small a
    distance. No agent's factor is constant across distances, as exponential separation would make it (within 0.25-119
    nats; smaller distances are not sampled); in all three it approaches 1 as the distance grows, as for an additive
    separation. The bulk: SKL after / before x.949 [x.947, x.951] / x.957 [x.954, x.959] / x.997 [x.995, x.999], naive
    x.868 [x.866, x.870]; above 1 for close pairs (5% > 1 up to 1.59 / 1.59 / 4.01 nats; x2.22 / x2.06 / x1.35 at 0.50
    nats), below 1 from 1.85 / 1.59 / 5.45 nats on (Bayes x.84, Trained RNN x.87 at 3.2 nats), about 1 at the far end;
    naive below 1 at every distance (x.77-.95): its shared update brings the bulk together.
  2 Both close or only one (i-l). Far / close third, matched on step and own bin: x1.028 [x1.027, x1.030] / x1.041
    [x1.040, x1.043] / x1.001 [x1.000, x1.002] (cells cover .999); thirds Bayes x1.018 / x1.024 / x1.047, Trained RNN
    x1.004 [x1.002, x1.006] / x1.020 / x1.046, ESN x1.064 / x1.063 / x1.065. Trained RNN / Bayes on one population
    x1.012 [x1.010, x1.013] (726 common cells). The all-bin values are small because most cases are far apart, where
    every factor is near 1; per bin, close / far third at 0.50 nats x1.30 / x2.48 (Bayes), x1.43 / x2.36 (Trained
    RNN), x1.39 / x1.37 (ESN); at 2.0 nats x1.10 / 1.25, x1.08 / 1.22, x1.19 / 1.19. The closest pairs: pair-steps
    whose average d_H before is 0.25-1 nat x1.16 [x1.11, x1.23] / x1.30 [x1.24, x1.39] / x1.42 [x1.38, x1.45] (1,074 /
    1,272 / 6,988 pair-steps) against single variables at 0.25-1 nat x1.56 [x1.48, x1.67] / x1.60 [x1.52, x1.70] /
    x1.35 [x1.33, x1.37] (8,301 / 12,696 / 47,402 cases): with both variables close, Bayes's and the Trained RNN's pairs
    separate less; the ESN's do not.
  3 Entanglement (m-p). Most / least entangled fifth, matched on step and bin: x1.020 [x1.018, x1.023] / x1.036
    [x1.033, x1.040] / x1.003 [x1.001, x1.004] (cells cover .984 / .987 / .992); fifths Bayes x1.022 / 1.026 / 1.027 /
    1.032 / 1.043, Trained RNN x1.011 [x1.007, x1.014] / 1.014 / 1.019 / 1.027 / 1.047 [x1.045, x1.049], ESN x1.062 /
    1.064 / 1.064 / 1.064 / 1.065. Per bin, least / most fifth at 0.93 nats x1.34 / 1.89, x1.17 / 1.84, x1.27 / 1.31;
    at 2.0 nats x1.11 / 1.26, x1.11 / 1.24, x1.19 / 1.20. Trained RNN / Bayes on one population x1.016 [x1.012,
    x1.019] (594 common cells). For Bayes the dependence is partly built in: the entanglement is computed from Bayes's
    beliefs, and with zero entanglement its update would equal naive's (td_tt_chaos_pairs.py READING 6).
  4 One flipped bit through training (q). Growth factor at lag 20: x2.11 [x2.10, x2.12] in the initial network (10
    episodes; its impacts are tiny, chi(s, s) .03 nats), x4.38 [x4.26, x4.48] at 20 (the largest), x3.77 at 40, x4.07
    at 100 and 180, x3.39 at 240, x2.78 at 320, x2.09 [x2.04, x2.16] at 880, x1.87 at 1,680, x1.50 at 4,200, x.99
    [x.98, x1.01] at the end; last above Bayes's level at 24,000 [21,600-28,800] episodes (5-95% over the resamples of
    the curve, Bayes's level fixed). Exact observers at lags 1 / 2 / 5 / 10 / 20: Bayes x1.020 / 1.029 / 1.044 / 1.054
    / 1.067 [x1.059, x1.075], factorized-state Bayes x1.08 / 1.14 / 1.28 / 1.42 / 1.60, naive 1. As a ratio, the
    amplification is largest at 20-180 episodes and falls from ~200 on, well before the network passes naive (835). As
    a difference (11-02: +.87 at 20, +1.65 at 320, +1.52 at 880 nats, from the chaos sweep's flips) it peaked later,
    because the immediate impact grows through training (k = 1: chi(s, s) .23 at 20, 1.00 at 880, 2.19 nats at the
    end, geometric means).
  5 k-bit flips (r). Growth factor at lags 1-3 / 6-9 / 12-15 / 18-21: at 20 episodes k = 1 x2.10 / 3.56 / 4.13 / 4.36,
    k = 2 x1.91 / 3.03 / 3.43 / 3.60, k = 3 x2.11 / 3.60 / 4.17 / 4.41, k = 4 x1.96 / 3.17 / 3.61 / 3.79, k = 5 x2.12
    / 3.63 / 4.21 / 4.45 (even k lower: the count code); at 100 k = 2 is highest (x5.00 at lags 18-21, k = 1 x4.05); at
    880 x1.66-2.47, rising with k and lag; at the end x.98-1.08. Bayes x1.03-1.19, factorized-state Bayes x1.13-1.69,
    naive 1. The parity zig-zag (alternation of the log ratio in k, td_tt_flips's definition, lags 18-21): +.355
    [+.336, +.372] at 20, +.204 at 40, then reversed: -.293 [-.314, -.273] at 60, -.256 at 80, -.127 at 100, -.062 at
    120; within +-.02 from 200 on (largest |.| .017, at 480). Concave in lag at every checkpoint and k (largest second
    difference over the lag bins +.04, in log units).
  6 Count classes (s). 2-bit flips, change / keep the count, growth factor at lags 18-21: x4.30 [x4.18, x4.42] /
    x2.96 [x2.90, x3.02] at 20 episodes, x3.73 / x2.93 at 40; x3.68 / x6.35 at 60, x3.43 / x8.12 [x7.78, x8.46] at 80,
    x3.43 / x7.53 at 100, x3.38 / x5.48 at 140, x3.15 / x3.79 at 240, x2.35 / x2.47 at 400, x2.11 / x2.17 at 880,
    x.99 / x1.01 at the end. Paired change / keep: x1.53 [x1.51, x1.55] at 20, x1.34 at 40, below 1 at every
    checkpoint from 60 to 380 (x.42 [x.41, x.44] at 80), at 560 and at most checkpoints from 1,200 to 12,800 (x.94-.96),
    not resolved elsewhere. Immediate impacts chi(s, s), change / keep, paired: x10.3 [x10.1, x10.4] at 20, x11.2 at
    40, x7.5 at 60, x4.7 at 80, x3.4 at 100, x2.3 at 140, x1.9 at 240, x1.2 at 880, x1.08 [x1.04, x1.12] at the end
    (exact observers x1.08 / x1.09 / x1.08: count-changing flips move the evidence slightly more). So the count code
    lives in the immediate impact (count-keeping flips start ~10 times smaller at 20-40 episodes), and while it fades
    (60-380 episodes) the count-keeping flips' small effects are amplified the most (up to x8.1 over 20 steps). The
    initial network: x2.07 / x2.12 (paired x.98 [x.97, x.98]) with equal immediate impacts (.031 / .031). Exact
    observers: Bayes x1.10 / x1.10 (paired x1.002 [x.996, x1.009]), factorized-state Bayes x1.58 / x1.58 (x1.006
    [x.995, x1.018]), naive 1 / 1.
  Accuracy crossings: passes naive 835 [798-876], half-way to Bayes 1,394 [1,285-1,489] training episodes.
NULLS AND LIMITS
  - The ESN's factor barely depends on the observation's entanglement (x1.003 [x1.001, x1.004]) or on the other
    variable's distance (x1.001 [x1.000, x1.002]), against its mean x1.064.
  - Bayes's and the Trained RNN's factors are not resolved from the shuffled step's below 1.17 nats.
  - The matched all-bin contrasts (l, p) average over every distance and are dominated by far pairs, whose factors are
    near 1; the per-bin curves (i-k, m-o) show where the dependence is.
  - No phase transition in (lag, bits flipped): concave in lag everywhere; in k only the parity zig-zag at 20-40
    episodes and its reversal at 60-120 (the count code fading), within +-.02 from 200 on. The initial network already
    amplifies a flip x2.1 over 20 steps.
  - Ratios of small distances: inside the pairs' bins d_H before >= 0.25 nats; the smallest single-set chi(s, s) is
    .0033 nats (early checkpoints; the float32 storage of the flips' log-belief changes gives relative errors of at most
    ~3e-4 there).
  - The flips (q-s) are the Trained RNN's; the ESN's and the width runs' differences are in td_tt_chaos_sweep.png and
    td_tt_flips.png.
CHECKS (checks(); all pass)
  - Ratios: bin counts, per-bin and all-case mean ln r recomputed from pairs_v2.npz equal pairs_ratio (<= 3.8e-15); the
    violin medians (0); the split groups (thirds, fifths, cells) give the pairs module's stored difference contrasts
    cond3|dh|step+bin and ent5|dh|step+bin in every resample (0). ratio_build's own: d_H before from pairs_lq =
    pairs_v2 (0); the re-drawn shuffle's seed means = pairs_shuf40 (0); pair-average bins + outside = pair-steps (0);
    naive's |ln r| <= 7.9e-14.
  - Flips: the C(5, k) means of the per-set chi = net_flips's float64 means (<= 1.5e-5; float32 storage) and the exact
    observers' = refs.npz (<= 1.5e-5); naive's log ratios 0 exactly (q, r, classes); 155,000 class units per checkpoint
    (0); the 114 checkpoints of the flips (0).
  - Differences (row 1's data): bin counts and per-bin means of Delta d_H and Delta SKL from pairs_v2.npz equal
    pairs_combined (<= 1.5e-14); the all-case mean Delta d_H in every resample (<= 3.1e-15); the sweep and the flips
    consistent with td_tt_chaos.py and td_tt_gap (0).
  - pyflakes clean; the figure viewed whole and in native-resolution crops of every row, the header and the footnote.

Run from the repository root (CPU unless noted; each about 10-30 s):
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg \\
        /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_chaos_combined.py \\
        ratios | fratio-refs | figure | numbers | checks
    The flips harvest (GPU; resumable; one process per GPU, about 4 min for 114 checkpoints on two GPUs):
    CUDA_VISIBLE_DEVICES=0 ... td_tt_chaos_combined.py fratio-harvest 0 2   and   CUDA_VISIBLE_DEVICES=1 ... 1 2"""

import os
import re
import sys
import glob
import time
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
CHAOS = os.path.join(gap.TT, "chaos")
PAIRS = os.path.join(CHAOS, "pairs_combined.npz")          # td_tt_chaos_pairs.py combined (40 bins; splits; contrasts)
UNITS = os.path.join(CHAOS, "pairs_v2.npz")                # td_tt_chaos_pairs.py build (per-unit measures)
RATIO = os.path.join(CHAOS, "pairs_ratio.npz")             # ratio_build (row 2: ratios after / before; John 11-03)
RATIO_SEED = 20261107 + 3000                               # the ratio violins' KDE subsamples (pairs' PAIR_SEED + 3000)
FRATIO = os.path.join(gap.TT, "flips", "ratio")            # fratio_harvest / fratio_refs (row 5 as ratios; John 11-03)
Q_LAGS = (1, 2, 5, 10, 20)                                 # q's lags (as the chaos sweep's)
CC_CHANGE, CC_KEEP = [6, 8], [7]                           # td_tt_flips classes (k - 1) * 6 + j: 2 bits, net +-2 / 0
FLIPS = os.path.join(gap.TT, "flips", "combined.npz")      # td_tt_flips.py export (k-bit flips through training)
SWEEP = os.path.join(CHAOS, "sweep")                       # td_tt_chaos.py sweep (single flips, every checkpoint)
BSIDE = os.path.join(CHAOS, "pilot", "bayes_side.npz")     # td_tt_chaos.py pilot (exact observers, the same flips)
PROBES = os.path.join(TD, "analysis", "probes.npz")
FIG, OLD_FIG = "td_tt_chaos_pairs.png", "td_tt_chaos_pairs_fr.png"
AG = ("bayes", "trained", "echo")                          # the agents whose d_H changes (naive's never does, T1)
NAME = {"bayes": "Bayes", "naive": "naive", "trained": "Trained RNN", "echo": "ESN"}
COL = {"bayes": "k", "naive": "0.55", "trained": "#1f77b4", "echo": "#ff7f0e"}
SHUF_C, C_ACC = "#a50f15", "#d9a400"                       # shuffled null (dark red); accuracy crossings (bands)
ADF_LS = (0, (6, 2, 1.5, 2))                               # factorized-state Bayes (dash-dot), as in td_tt_chaos.py
MIN_UNITS = 100                                            # a split curve is drawn at bins with >= 100 units per group
# the chaos sweep's single flips (td_tt_chaos.py M2; constants copied so that this module does not import it)
SW_FLIP_S = (1, 3, 5, 10, 15, 20, 25)
SW_FIXED = tuple(a for a, s in enumerate(SW_FLIP_S) if s <= 10)    # flips at steps 1, 3, 5, 10: every lag up to 20
SW_LAGS = (1, 2, 5, 10, 20)
SW_STAGES = (320, 880, 1680, 4200)                         # training episodes quoted for the sweep (and the end)
MAP_RAW = (25, 125, 1100, 49995)                           # phase diagrams: training 20, 100, 880, 39,996 episodes
CC_X = (20, 40, 60, 80, 100, 240, 1200)                    # count code: training episodes quoted
_CACHE = {}


# ═══════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════

def _npz(path):
    if path not in _CACHE:
        with np.load(path) as Z:
            _CACHE[path] = {k: Z[k] for k in Z.files}
    return _CACHE[path]


def pairs():
    """pairs_combined.npz as a dict (layout: td_tt_chaos_pairs.py, COMBINED CACHE)"""
    return _npz(PAIRS)


def flips():
    """flips/combined.npz as a dict (layout: td_tt_flips.COMBINED_LAYOUT)"""
    return _npz(FLIPS)


def _W():
    """(1 + N_BOOT, 50): the full sample (row 0), then td_tt_gap's shared environment resamples"""
    return np.vstack([np.ones((1, gap.N_ENV)), gap.boot_counts()])


def _iv(b):
    """value, 5% and 95% of a (1 + N_BOOT, ...) array (row 0 = the full sample)"""
    return b[0], np.nanpercentile(b[1:], 5, 0), np.nanpercentile(b[1:], 95, 0)


def pstat(name):
    """a stored contrast or summary of pairs_combined.npz: (value, 5%, 95%)"""
    P = pairs()
    i = list(P["stat_names"]).index(name)
    return _iv(P["stat_boot"][i])


def pstat_aux(name):
    """the stored auxiliary number of a contrast (matched: the share of cases in the used cells; 'close': the number
    of cases or pair-steps; Trained - Bayes: the number of common cells)"""
    P = pairs()
    return float(P["stat_aux"][list(P["stat_names"]).index(name)])


def unit_summary():
    """from the per-unit cache pairs_v2.npz (all units = pair-step x variable): per agent the mean change of SKL and of
    d_H, each a ratio of environment sums with its resamples (the pairs module's estimator), and the share of units
    whose d_H grows; the 1st / 99th percentiles of Delta d_H and of log10 d_H after over the units inside the 40 bins
    (the axis ranges)"""
    if "units" in _CACHE:
        return _CACHE["units"]
    P = pairs()
    e40 = P["edges40"]
    W = _W()
    out = {}
    with np.load(UNITS) as Z:
        env = Z["env"]
        c = np.bincount(env, None, gap.N_ENV) * 2.
        for k in P["agents"]:
            pre, post = Z[f"pre_{k}"], Z[f"post_{k}"]
            dskl = Z[f"skl_post_{k}"] - Z[f"skl_pre_{k}"]
            inb = (pre >= e40[0]) & (pre < e40[-1])
            r = {}
            for nm, v in (("skl", dskl), ("dh", post - pre), ("grow", (post > pre + 1e-9).astype(np.float64))):
                s = np.bincount(env, v.sum(1), gap.N_ENV)
                r[nm] = _iv((W @ s) / (W @ c))
            r["dh_pct"] = np.percentile((post - pre)[inb], [1, 99])
            r["after_pct"] = np.percentile(np.log10(np.maximum(post[inb], 1e-300)), [.1, 99.9])
            r["n_in"] = int(inb.sum())
            out[k] = r
    _CACHE["units"] = out
    return out


def ratio_build():
    """John 11-03: row 2 as the ratio of the distances after / before instead of their difference -> pairs_ratio.npz.
    Per agent and case (pairs_v2.npz): ln r = ln(d_H after / d_H before) and ln(SKL after / SKL before); the combined
    figure's main shuffle (td_tt_chaos_pairs.combined: the same groups and seeds, re-drawn here) as the mean over its 5
    seeds of ln(shuffled d_H after / d_H before) per case (each draw's log, then the mean, as for the actual cases;
    pairs_shuf40.npz stores the seeds' mean d_H after, whose log is larger by Jensen). Per bin of d_H before (the 40
    bins): the mean of each over the cases (ratios of environment sums; the full sample, then the 1,000 shared
    resamples: (1 + N_BOOT, 40)), the median of ln r and the KDE violins of log10 r (td_tt_chaos_pairs.violin_stats);
    over all cases inside the bins: the same means and the share of cases with r > 1. CPU, about 3 min"""
    import td_tt_chaos_pairs as tp
    t0 = time.time()
    P = pairs()
    e40 = P["edges40"]
    nb = len(e40) - 1
    W = _W()
    rng = np.random.default_rng(RATIO_SEED)
    seeds = tp.SHUF40_SEED + np.arange(tp.N_SHUF)
    out = dict(agents=P["agents"], edges40=e40, seeds=seeds)
    A = len(P["agents"])
    out.update({f"vio_{s}": np.full((A, nb, 100), np.nan) for s in ("y", "d")})
    out["vio_med"], out["med"] = np.full((A, nb), np.nan), np.full((A, nb), np.nan)
    out["pct"] = np.full((A, nb, 4), np.nan)                              # log10 r: .5 / 1 / 99 / 99.5 percentiles
    chk, mt_keep = {}, {}
    with np.load(UNITS) as Z, np.load(tp.SHUF40_CACHE) as SH, np.load(tp.LQ_CACHE) as LQ:
        I, J, S = Z["I"], Z["J"], Z["t"]
        m = len(I)
        env = np.repeat(Z["env"], tp.NV)
        g5 = np.repeat(SH["g5"].astype(np.int64), tp.NV)
        for ia, k in enumerate(P["agents"]):
            pre, post = Z[f"pre_{k}"].ravel(), Z[f"post_{k}"].ravel()
            b = tp.unit_bins(pre, e40)
            inb = (b >= 0) & (b < nb)
            assert pre[inb].min() > 0 and post[inb].min() > 0 and Z[f"skl_pre_{k}"].ravel()[inb].min() > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = np.log(post / pre)
                series = {"dh": lr, "skl": np.log(Z[f"skl_post_{k}"].ravel() / Z[f"skl_pre_{k}"].ravel()),
                          "grow": (post > pre + 1e-9).astype(np.float64)}
            if k != "naive":                                              # naive: r = 1 in every case (T1)
                lq = LQ[k]
                D = lq[I, S - 2] - lq[J, S - 2]
                dl = (lq[I, S - 1] - lq[I, S - 2]) - (lq[J, S - 1] - lq[J, S - 2])
                pre2 = tp.osc(D)
                chk[f"d_H before from pairs_lq = pairs_v2, {k}"] = float(np.abs(pre2.ravel() - pre).max())
                gid = ((g5 * tp.OTH_K + SH[f"cond_{k}"].ravel().astype(np.int64)) * (nb + 2) + b + 1).reshape(m, tp.NV)
                acc_l, acc_s = np.zeros((m, tp.NV)), np.zeros((m, tp.NV))
                for sd in seeds:                                          # tp.pair_shuffle's draws, in its order
                    r_ = np.random.default_rng(int(sd))
                    for v in range(tp.NV):
                        perm = tp._within_perm(gid[:, v], r_)
                        sgn = r_.choice([-1., 1.], m)
                        sh = tp.osc(D[:, v] + sgn[:, None] * dl[perm, v])
                        acc_s[:, v] += sh
                        acc_l[:, v] += np.log(np.maximum(sh, 1e-300) / pre2[:, v])
                chk[f"re-drawn shuffle: the seeds' mean d_H after = pairs_shuf40, {k}"] = float(
                    np.abs(acc_s / len(seeds) - SH[f"shuf_{k}"]).max())
                series["shuf"] = (acc_l / len(seeds)).ravel()
                del D, dl, acc_s, acc_l
            key = env[inb] * nb + b[inb]
            C = W @ np.bincount(key, None, gap.N_ENV * nb).reshape(gap.N_ENV, nb)
            for nm, v in series.items():
                Sm = W @ np.bincount(key, v[inb], gap.N_ENV * nb).reshape(gap.N_ENV, nb)
                out[f"bin_{nm}_{k}"] = Sm / C                             # (1 + N_BOOT, 40)
                out[f"all_{nm}_{k}"] = Sm.sum(1) / C.sum(1)               # (1 + N_BOOT,)
            out[f"n_{k}"] = C[0]
            if k != "naive":
                # rows 3-4 as ratios: the pairs module's splits and estimators applied to ln r
                t_u = np.repeat(S, tp.NV)
                tb = t_u * nb + b
                cg = SH[f"cond_{k}"].ravel().astype(np.int64)
                out[f"cond_{k}"], out[f"cond_n_{k}"] = tp.bin_stats(lr, b, inb, env, W, nb, cg, tp.OTH_K)
                pa, qa = Z[f"pre_{k}"].mean(1), Z[f"post_{k}"].mean(1)    # the pair's average d_H before / after
                ba = tp.unit_bins(pa, e40)
                inba = (ba >= 0) & (ba < nb)
                lra = np.log(qa / pa)
                r_, n_ = tp.bin_stats(lra, ba, inba, Z["env"], W, nb)
                out[f"avg_{k}"], out[f"avg_n_{k}"] = r_[:, 0], n_[0]
                r_, used = tp.step_lines(lr, b, g5, inb, env, t_u, W, tp.ENT_K, nb)
                r_[:, :, used < tp.LINE_MIN] = np.nan
                out[f"ent_{k}"], out[f"ent_used_{k}"] = r_, used
                for nm, grp, K in (("cond3", cg, tp.OTH_K), ("ent5", g5, tp.ENT_K)):
                    mt = tp.matched(lr, grp, tb, inb, env, W, K=K)
                    out[f"{nm}_std_{k}"], out[f"{nm}_hilo_{k}"] = mt["std"], mt["hilo"]
                    out[f"{nm}_share_{k}"] = mt["share"]
                    if k in ("bayes", "trained"):
                        mt_keep[(nm, k)] = mt
                sv, sa = inb & (pre < 1.), inba & (pa < 1.)                # the closest pairs inside the bins
                out[f"close_var_{k}"], out[f"close_var_n_{k}"] = tp._ratio(lr, sv, env, W), int(sv.sum())
                out[f"close_avg_{k}"], out[f"close_avg_n_{k}"] = tp._ratio(lra, sa, Z["env"], W), int(sa.sum())
                chk[f"pair-average bins + outside = pair-steps, {k}"] = float(n_.sum() + (~inba).sum() - m)
            l10 = lr / np.log(10.)
            for i in range(nb):
                x = l10[inb & (b == i)]
                out["med"][ia, i] = np.median(x) * np.log(10.)
                out["pct"][ia, i] = np.percentile(x, [.5, 1, 99, 99.5])
                if k != "naive":
                    out["vio_y"][ia, i], out["vio_d"][ia, i], out["vio_med"][ia, i] = tp.violin_stats(x, rng)
            if k == "naive":
                chk["naive: max |ln r| inside the bins (T1: 0)"] = float(np.abs(lr[inb]).max())
            print(f"  {k}: {time.time() - t0:.0f} s", flush=True)
    for nm in ("cond3", "ent5"):                                          # Trained RNN - Bayes on one population
        out[f"{nm}_hilo_trained-bayes"], out[f"{nm}_ncommon"] = tp.common_hilo(mt_keep[(nm, "trained")],
                                                                             mt_keep[(nm, "bayes")])
    out["chk_names"], out["chk_values"] = np.array(list(chk)), np.array(list(chk.values()))
    np.savez(RATIO, **out)
    print(f"saved {RATIO} ({time.time() - t0:.0f} s); checks:")
    for k_, v in chk.items():
        print(f"  {k_}: {v:.3g}")


def ratios():
    """pairs_ratio.npz as a dict (ratio_build)"""
    return _npz(RATIO)


# ── row 5 as ratios (John 11-03): every flip set's own chi(s + lag, s) / chi(s, s), harvested per set and lag

def _fratio_file(run, raw):
    return os.path.join(FRATIO, f"{run}_{raw:05d}.npz")


def _fratio_reduce(chi, code_s, env):
    """per-set chi (n, 5 flip steps, 31 flip sets, 22 lags, 2 variables) -> row 5's per-environment sums of log
    ratios ln chi(s + lag, s) - ln chi(s, s) (the statistic per flip set and variable, then means):
      q_sum (50, 5): single flips (the 5 k = 1 sets) at the lags Q_LAGS; per probe the mean over the 5 sets, the 5 flip
        steps and the 2 variables; summed over the environment's probes (q_n (50,) probes)
      r_sum (50, 5 k, 4 lag bins): per set the mean over the bin's lags, then per probe the mean over the C(5, k) sets
        of each k, the flip steps and the variables; r0_sum (50, 5 k): the same for ln chi(s, s) (the immediate impact)
      count classes (td_tt_flips.count_classes on V (n, 5, 31, 5) = the variable means of [ln chi(s, s), the 4 lag
        bins' log ratios]): cls_sum (50, 30, 5), cls_n (50, 30), pair_sum (50, 4, 5), pair_n (50, 4)"""
    import td_tt_flips as tf
    lc = np.log(chi.astype(np.float64))                                          # (n, nS, 31, 22, 2)
    lr = lc - lc[:, :, :, :1]
    E = np.eye(gap.N_ENV)[np.asarray(env)]                                       # (n, 50)
    q = lr[:, :, tf.SINGLE][:, :, :, list(Q_LAGS)].mean((1, 2, 4))               # (n, 5 lags)
    bins = np.stack([lr[:, :, :, a:b + 1].mean(3) for a, b in tf.LAG_BINS], 3)   # (n, nS, 31, 4, 2)
    rk = tf.kmean_np(bins.mean((1, 4)), 1)                                       # (n, 5 k, 4)
    r0 = tf.kmean_np(lc[:, :, :, 0].mean((1, 3)), 1)                             # (n, 5 k)
    V = np.concatenate([lc[:, :, :, :1].mean(-1), bins.mean(-1)], 3)             # (n, nS, 31, 5)
    c = tf.count_classes(V, code_s, env)
    return dict(q_sum=E.T @ q, q_n=E.sum(0), r_sum=np.einsum("pe,pkb->ekb", E, rk), r0_sum=E.T @ r0,
                cls_sum=c["cls_sum"], cls_n=c["cls_n"], pair_sum=c["pair_sum"], pair_n=c["pair_n"])


def fratio_one(run, raw, rows, dev):
    """one checkpoint -> flips/ratio/<run>_<raw>.npz (_fratio_reduce of td_tt_flips.net_flips's subset-resolved
    changes D: chi = d_H = osc(D) per flip set, lag and variable); check = max |C(5, k) mean of these chi - net_flips's
    own float64 d_H means| (float32 storage of D)"""
    import td_tt_flips as tf
    t0 = time.time()
    N = tf.net_flips(run, raw, rows, dev, keep=True, per_set=True)
    chi = N["D"].max(-1) - N["D"].min(-1)                                        # (n, nS, 31, 22, 2) float32
    env = tf.ti.probes()["env"][rows]
    o = _fratio_reduce(chi, N["code"][:, [s - 1 for s in tf.FLIP_S]], env)
    o.update(raw=raw, x=float(gap.train_x(raw)), chi0_min=float(chi[:, :, :, 0].min()),
             check=float(np.abs(tf.kmean_np(chi.astype(np.float64), 2) - N["dh"]).max()), seconds=time.time() - t0)
    path = _fratio_file(run, raw)
    os.makedirs(FRATIO, exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **o)
    os.replace(tmp, path)
    return o


def fratio_harvest(part=0, nparts=1, run="trained"):
    """every checkpoint of the run (this process: every nparts-th, from part; resumable, skips finished files). One
    GPU per process (CUDA_VISIBLE_DEVICES=0 or 1); about 5 s per checkpoint"""
    import td_tt_flips as tf
    dev = tf.ti._device()
    rows = tf.subset()
    todo = [r for r in tf.ti.checkpoints(run)[part::nparts] if not os.path.exists(_fratio_file(run, r))]
    print(f"{run}: {len(todo)} checkpoints to do (part {part} of {nparts})", flush=True)
    for raw in todo:
        if os.path.exists(_fratio_file(run, raw)):
            continue
        o = fratio_one(run, raw, rows, dev)
        print(f"{time.strftime('%H:%M:%S')} {run} {raw} ({o['x']:,.0f}): {o['seconds']:.1f} s; check {o['check']:.1e};"
              f" min chi(s, s) {o['chi0_min']:.4f}", flush=True)
    print("done", flush=True)


def _envr(num, den):
    """ratios of environment sums with the shared resamples: num (50, ...), den (50,) or (50, ...) ->
    (1 + N_BOOT, ...)"""
    W = _W()
    num = np.asarray(num, np.float64)
    den = np.broadcast_to(np.asarray(den, np.float64).reshape((gap.N_ENV,) + (1,) * (num.ndim - 1)), num.shape) \
        if np.ndim(den) == 1 else np.asarray(den, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((W @ num.reshape(gap.N_ENV, -1)) / (W @ den.reshape(gap.N_ENV, -1))).reshape((len(W),) + num.shape[1:])


def fratio_stats(run="trained"):
    """row 5 as ratios (flips/ratio): mean log ratios per checkpoint as ratios of environment sums, (1 + N_BOOT) rows
    (row 0 the full sample): q (K, 1 + N_BOOT, 5 lags), r (K, 1 + N_BOOT, 5 k, 4 lag bins), r0 (K, 1 + N_BOOT, 5 k)
    = ln chi(s, s); change / keep (K, 1 + N_BOOT, 5) the 2-bit class means (index 0 = ln chi(s, s), 1..4 the lag bins'
    log ratios) and index (K, 1 + N_BOOT, 5) their paired difference per (probe, flip step) (td_tt_flips's k2_cc);
    refs {bayes, naive, adf}: the same without the checkpoint axis"""
    key = ("fratio", run)
    if key in _CACHE:
        return _CACHE[key]
    pat = re.compile(rf"^{re.escape(run)}_(\d{{5}})\.npz$")
    raws = sorted(int(m.group(1)) for m in (pat.match(os.path.basename(f))
                                            for f in glob.glob(os.path.join(FRATIO, "*"))) if m)
    keys = ("q_sum", "q_n", "r_sum", "r0_sum", "cls_sum", "cls_n", "pair_sum", "pair_n")
    Z = {k: [] for k in keys}
    for raw in raws:
        with np.load(_fratio_file(run, raw)) as f:
            for k in keys:
                Z[k].append(f[k])
    Z = {k: np.stack(v) for k, v in Z.items()}                                   # checkpoint axis first

    def summarise(z):
        """z: the arrays of one source with a checkpoint axis first"""
        o = {}
        for nm, num, den in (("q", z["q_sum"], z["q_n"]), ("r", z["r_sum"], z["q_n"]), ("r0", z["r0_sum"], z["q_n"]),
                             ("change", z["cls_sum"][:, :, CC_CHANGE].sum(2), z["cls_n"][:, :, CC_CHANGE].sum(2)),
                             ("keep", z["cls_sum"][:, :, CC_KEEP].sum(2), z["cls_n"][:, :, CC_KEEP].sum(2)),
                             ("index", z["pair_sum"][:, :, 0], z["pair_n"][:, :, 0])):
            o[nm] = np.stack([_envr(num[i], den[i]) for i in range(len(num))])
        return o
    out = summarise(Z)
    out.update(x=gap.train_x(np.array(raws)), raw=np.array(raws))
    out["refs"] = {}
    with np.load(os.path.join(FRATIO, "refs.npz")) as Rf:
        for o in ("bayes", "naive", "adf"):
            z = {k: Rf[f"{o}_{k}"][None] for k in keys}
            out["refs"][o] = {nm: v[0] for nm, v in summarise(z).items()}
    _CACHE[key] = out
    return out


def exact_set_chi(rows, chunk=25):
    """the exact observers' chi per flip set, lag and variable (float64; td_tt_flips.exact_flips's flips, before its
    C(5, k) means): {'bayes', 'naive', 'adf'}: (n, 5 flip steps, 31, 22, 2)"""
    import td_tt_flips as tf
    P = tf.ti.probes()
    L = tf.tables()
    obs, env = P["obs"][rows], P["env"][rows]
    n, nS, nm = len(rows), len(tf.FLIP_S), len(tf.MASKS)
    out = {o: np.zeros((n, nS, nm, tf.NLAG, tf.NV)) for o in ("bayes", "naive", "adf")}
    for a0 in range(0, n, chunk):
        sl = slice(a0, min(n, a0 + chunk))
        b = sl.stop - sl.start
        Le = L[env[sl]]
        ll = tf.step_loglik(obs[sl], Le)
        u = np.cumsum(ll, 1)
        lmB = tf._norm(tf.marg_log(u))
        logit = np.log(Le) - np.log1p(-Le)
        Lb = np.stack([Le.mean(3), Le.mean(2)], 2)
        logit_n = np.log(Lb) - np.log1p(-Lb)
        la = tf.adf_run(ll)
        for a, s in enumerate(tf.FLIP_S):
            win = slice(s - 1, s - 1 + tf.NLAG)
            sign = 1. - 2. * obs[sl, s - 1].astype(np.float64)
            dJ = np.einsum("mo,bo,borc->bmrc", tf.MBITS, sign, logit)
            lm2 = tf._norm(tf.marg_log(u[:, None, win] + dJ[:, :, None]))
            out["bayes"][sl, a] = tf.osc(lm2 - lmB[:, None, win])
            dN = np.einsum("mo,bo,bovr->bmvr", tf.MBITS, sign, logit_n)
            out["naive"][sl, a] = tf.osc(dN)[:, :, None]
            ll2 = np.repeat(ll[:, None, win], nm, 1)
            ll2[:, :, 0] += dJ
            la0 = None if s == 1 else np.repeat(la[:, s - 2][:, None], nm, 1).reshape(-1, 2, tf.R)
            la2 = tf.adf_run(ll2.reshape(-1, tf.NLAG, tf.R, tf.R), la0).reshape(b, nm, tf.NLAG, tf.NV, tf.R)
            out["adf"][sl, a] = tf.osc(la2 - la[:, None, win])
    return out


def fratio_refs():
    """the exact observers on the same probes and flips -> flips/ratio/refs.npz ('<obs>_<key>' of _fratio_reduce);
    check_<obs> = max |C(5, k) mean of chi - refs.npz's <obs>_dh| (float32 storage there). CPU, a few minutes"""
    import td_tt_flips as tf
    t0 = time.time()
    rows = tf.subset()
    P = tf.ti.probes()
    code_s = (P["obs"][rows][:, [s - 1 for s in tf.FLIP_S]] * (1 << np.arange(tf.NO))).sum(-1)
    X = exact_set_chi(rows)
    out = {}
    with np.load(os.path.join(tf.OUT, "refs.npz")) as Rf:
        for o, chi in X.items():
            out.update({f"{o}_{k}": v for k, v in _fratio_reduce(chi, code_s, P["env"][rows]).items()})
            out[f"check_{o}"] = float(np.abs(tf.kmean_np(chi, 2) - Rf[f"{o}_dh"]).max())
    os.makedirs(FRATIO, exist_ok=True)
    np.savez(os.path.join(FRATIO, "refs.npz"), **out)
    print(f"refs: {time.time() - t0:.0f} s; checks " + ", ".join(f"{o} {out[f'check_{o}']:.1e}" for o in X), flush=True)


def sweep(run="trained"):
    """the chaos sweep's single flips (td_tt_chaos.py M2, as its sweep_figure draws them): per checkpoint the paired
    rise chi(s + lag, s) - chi(s, s) of d_H between the intact and flipped marginals (mean of the 2 variables), averaged
    over the 5 observations (stored) and the flips at steps 1, 3, 5, 10, per probe; then environment means with the
    shared resamples (1,000 probes = 20 per environment). Returns x (K,), raw (K,), rise (K, 5 lags, 3) = value / 5% /
    95%, the probe rows and the per-probe rises val (K, 5 lags, n)"""
    key = ("sweep", run)
    if key in _CACHE:
        return _CACHE[key]
    pat = re.compile(rf"^{re.escape(run)}_(\d{{5}})\.npz$")
    raws = sorted(int(m.group(1)) for m in (pat.match(os.path.basename(f)) for f in glob.glob(os.path.join(SWEEP, "*")))
                  if m)
    rows, val = None, []
    for raw in raws:
        with np.load(os.path.join(SWEEP, f"{run}_{raw:05d}.npz")) as Z:
            if rows is None:
                rows = Z["rows"]
            assert np.array_equal(Z["rows"], rows)
            chi = Z["chi"].astype(np.float64).mean(-1)                                    # (n, 7 flips, T)
        val.append(np.stack([chi_rise(chi, L) for L in SW_LAGS]))                          # (5 lags, n)
    V = np.stack(val)                                                                      # (K, 5, n)
    out = dict(x=gap.train_x(np.array(raws)), raw=np.array(raws), rows=rows,
               rise=np.stack(env_boot(V, rows), -1), val=V)
    _CACHE[key] = out
    return out


def sweep_last_above(level, lag=20):
    """the last checkpoint at which the Trained RNN's sweep rise at `lag` lies above a fixed `level` (Bayes's line), in
    the full sample and in each shared environment resample of the curve: (value, 5%, 95%) by td_tt_gap.interval"""
    S = sweep()
    v = S["val"][:, SW_LAGS.index(lag)]                                                   # (K, n probes)
    env = _npz(PROBES)["env"][S["rows"]] if "env" in _npz(PROBES) else S["rows"] // gap.PER_ENV
    boot = gap.boot_counts()
    S_ = np.stack([v[:, env == e].sum(-1) for e in range(gap.N_ENV)], -1)                 # (K, 50)
    C_ = np.bincount(env, minlength=gap.N_ENV).astype(np.float64)
    curves = np.column_stack([S_.sum(-1) / C_.sum(), (S_ @ boot.T) / (boot @ C_)])        # (K, 1 + N_BOOT)
    last = np.array([S["x"][np.flatnonzero(c > level)[-1]] if (c > level).any() else np.nan for c in curves.T])
    lo, hi, _ = gap.interval(last[1:])
    return float(last[0]), lo, hi


def chi_rise(chi, lag):
    """the paired rise chi(s + lag, s) - chi(s, s), averaged over the fixed flips (steps 1, 3, 5, 10) per probe: chi
    (..., 7 flips, T) -> (...) (td_tt_chaos.chi_rise)"""
    return np.mean([chi[..., a, SW_FLIP_S[a] - 1 + lag] - chi[..., a, SW_FLIP_S[a] - 1] for a in SW_FIXED], 0)


def env_boot(v, rows):
    """environment means of per-probe values v (..., n): value, 5% and 95% over the shared resamples
    (td_tt_chaos._env_boot)"""
    env = _npz(PROBES)["env"][rows] if "env" in _npz(PROBES) else rows // gap.PER_ENV
    boot = gap.boot_counts()
    S_ = np.stack([v[..., env == e].sum(-1) for e in range(gap.N_ENV)], -1)
    C_ = np.bincount(env, minlength=gap.N_ENV).astype(np.float64)
    bt = (S_ @ boot.T) / (boot @ C_)
    return S_.sum(-1) / C_.sum(), np.percentile(bt, 5, -1), np.percentile(bt, 95, -1)


def sweep_refs():
    """the exact observers on the same flips (td_tt_chaos.py pilot/bayes_side.npz): the mean rise per lag (5,) for
    Bayes, naive and factorized-state Bayes (adf); chi = mean over the 2 variables, then the 5 observations"""
    if "sweep_refs" in _CACHE:
        return _CACHE["sweep_refs"]
    B = _npz(BSIDE)
    out = {nm: np.array([chi_rise(B[f"chi_{nm}"].mean(-1).mean(-2), L).mean() for L in SW_LAGS])
           for nm in ("bayes", "naive", "adf")}
    _CACHE["sweep_refs"] = out
    return out


# ═══════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════

def _m(x, d=2):
    """a signed number in mathtext (a true minus sign)"""
    return f"${x:+.{d}f}$"


def _mi(t, d=2):
    """(value, lo, hi) -> 'value [lo, hi]' in mathtext"""
    return f"{_m(t[0], d)} [{_m(t[1], d)}, {_m(t[2], d)}]"


def _logx(ax, lo=-.68, hi=2.12):
    from matplotlib.ticker import FixedLocator
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator([0, 1, 2]))
    ax.set_xticklabels(["1", "10", "100"])
    ax.xaxis.set_minor_locator(FixedLocator([np.log10(m * 10. ** e) for e in (-1, 0, 1) for m in range(2, 10)
                                             if lo <= np.log10(m * 10. ** e) <= hi]))
    ax.spines[["top", "right"]].set_visible(False)


def _trainx(ax, FL, xmax=4.78):
    ax.set_xlim(np.log10(8), xmax)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.set_xlabel("training episodes", fontsize=FL)
    ax.spines[["top", "right"]].set_visible(False)


def _title(ax, letter, name, formula, TS, FT, pad_lines=None):
    """'letter   name' on top, the formula below it (as td_tt_flips)"""
    n = formula.count("\n") + 1 if pad_lines is None else pad_lines
    ax.set_title(f"{letter}   {name}", loc="left", fontsize=TS, pad=5 + 14.5 * n)
    ax.text(0, 1.02, formula, transform=ax.transAxes, fontsize=FT, ha="left", va="bottom", linespacing=1.25)


def _violins(ax, Y, D, med, pos, half, color, alpha):
    """cached KDE violins (grid Y and density D, (bins, 100); each scaled to its own maximum, as matplotlib's
    violinplot) and the medians of all units in the bin (white dots)"""
    ok = []
    for i in range(len(pos)):
        y, d = Y[i], D[i]
        if not (np.isfinite(d).all() and d.max() > 0):
            continue
        w = d / d.max() * half
        ax.fill_betweenx(y, pos[i] - w, pos[i] + w, color=color, alpha=alpha, lw=0, zorder=2)
        ok.append(i)
    ok = np.array(ok)
    ax.plot(pos[ok], med[ok], "o", ms=2.7, mfc="w", mec="k", mew=.5, zorder=5)


def _ratio_y(ax, lo, hi, ticks=(.5, 1, 2, 5, 10)):
    """a log10 axis labelled with the ratios themselves"""
    from matplotlib.ticker import FixedLocator
    ax.set_ylim(lo, hi)
    t = [x for x in ticks if lo <= np.log10(x) <= hi]
    ax.yaxis.set_major_locator(FixedLocator(np.log10(t)))
    ax.set_yticklabels([f"{x:g}" for x in t])
    ax.yaxis.set_minor_locator(FixedLocator([np.log10(m * 10. ** e) for e in (-1, 0, 1) for m in range(2, 10)
                                             if lo <= np.log10(m * 10. ** e) <= hi and m * 10. ** e not in t]))


def _x(v, d=3):
    """a ratio in mathtext (from its natural log)"""
    return rf"$\times${np.exp(v):.{d}f}"


def _xi(t, d=3):
    """(ln value, ln lo, ln hi) -> 'xvalue [xlo, xhi]'"""
    return f"{_x(t[0], d)} [{_x(t[1], d)}, {_x(t[2], d)}]"


def _mean_style(k, lw=2.):
    import matplotlib.patheffects as pe
    return dict(color=COL[k], lw=lw, solid_capstyle="round", zorder=6,
                path_effects=[pe.Stroke(linewidth=lw + 1.6, foreground="w" if k == "bayes" else "k"), pe.Normal()])


def _band(ax, x, mean, lo, hi, color, ok=None, alpha=.2, **kw):
    ok = np.isfinite(mean) if ok is None else ok & np.isfinite(mean)
    m, l_, h_ = (np.where(ok, a, np.nan) for a in (mean, lo, hi))
    ax.fill_between(x, l_, h_, color=color, alpha=alpha, lw=0, zorder=kw.get("zorder", 3) - 1)
    return ax.plot(x, m, color=color, **kw)


def stats():
    """every number drawn in the figure (texts, legends) and quoted in the docstring, from the caches"""
    P, F = pairs(), flips()
    U = unit_summary()
    st = {"units": U}
    for k in AG:
        st[k] = {nm: pstat(f"all|{nm}|{k}") for nm in ("dh", "shuf", "dh-shuf", "aligned", "sym")}
        # the four contrasts that involve FR (...+fr, fr3|...) are loaded but neither drawn nor printed here; in the
        # pairs_combined.npz of 11-02 their FR is the symmetric clipped KL (td_tt_chaos_pairs.py: FACTORIZATION REGRET)
        for nm in ("ent5|dh|step+bin", "cond3|dh|step+bin", "close|var<1|dh", "close|avg<1|dh", "ent5|dh|raw",
                   "ent5|dh|step+bin+fr", "ent5|dh|step+bin+cond3", "ent5|dh|band+bin+env", "cond3|dh|step+bin+ent5",
                   "cond3|dh|step+bin+fr", "fr3|dh|step+bin", "fr3|dh|step+bin+ent5"):
            st[k][nm] = pstat(f"{nm}|{k}")
        ia = list(P["agents"]).index(k)
        cm = list(P["cmeas"]).index("dh")
        st[k]["ent5_means"] = _iv(P["ent5_std"][ia, cm].T)                          # 3 x (5,)
        st[k]["cond3_means"] = _iv(P["cond3_std"][ia, cm].T)                        # 3 x (3,)
    for nm in ("ent5|dh|step+bin|trained-bayes", "cond3|dh|step+bin|trained-bayes"):
        st[nm] = pstat(nm)
    # flips: the count code, the phase diagrams, the references
    x = F["trained_cc_x"]
    st["cc"] = {xx: {g: tuple(F[f"trained_count_{g}{s}"][int(np.flatnonzero(x == xx)[0]), 4]
                              for s in ("", "_lo", "_hi")) for g in ("changing", "preserving", "index")}
                for xx in CC_X}
    st["cc_ref"] = {o: (float(F[f"{o}_count_changing"][4]), float(F[f"{o}_count_preserving"][4]),
                        float(F[f"{o}_count_index"][4])) for o in ("bayes", "naive", "adf")}
    st["maps"] = {}
    for raw in MAP_RAW:
        i = int(np.flatnonzero(F["trained_raw"] == raw)[0])
        st["maps"][raw] = (float(F["trained_x"][i]), F["trained_growth_dh"][i])          # (5 k, 4 bins)
    for o in ("bayes", "naive", "adf"):
        st["maps"][o] = (np.nan, F[f"{o}_growth_dh"])
    st["acc"] = {c: (float(F[f"trained_{c}"]), float(F[f"trained_{c}_lo"]), float(F[f"trained_{c}_hi"]))
                 for c in ("c0", "c50")}
    # the chaos sweep (single flips at steps 1, 3, 5, 10)
    S = sweep()
    st["sweep"] = S
    st["sweep_refs"] = sweep_refs()
    return st


def figure(path=None):
    """td_tt_chaos_pairs.png (John 11-02, the combined figure): rows 1-4 the pairs at the end of training (40 bins of
    the distance before), row 5 a flipped observation's effect through training (Trained RNN). Deletes the superseded
    td_tt_chaos_pairs_fr.png"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    FL, TS, FT, FX = fs["label"] + .5, 13., 10.5, 10.
    P, F = pairs(), flips()
    st = stats()
    U = st["units"]
    ag = list(P["agents"])
    le = np.log10(P["edges40"])
    pos = np.log10(P["centres40"])
    half = (le[1] - le[0]) * .85 / 2
    fig = plt.figure(figsize=(19.5, 24.))
    top = fig.add_gridspec(4, 4, left=.047, right=.985, top=.878, bottom=.318, hspace=.62, wspace=.27)
    bot = fig.add_gridspec(1, 4, left=.047, right=.985, top=.218, bottom=.108, wspace=.27)
    cols = ("bayes", "trained", "echo")
    xlab = r"distance before, $d_H(p_{t-1}, q_{t-1})$ (nats)"
    zero = dict(color=COL["naive"], lw=2.4, zorder=1)                                     # naive: Delta d_H = 0 (T1)

    def lab(ax, j, text):
        if j == 0:
            ax.set_ylabel(text, fontsize=FL)
    # ── row 1: the distance after (violins); the y range starts below the lowest violin point, so none is cut
    ylo = float(np.nanmin(P["vio_y"][:, 0])) - .07
    yhi = max(U[k]["after_pct"][1] for k in ag) + .12
    for j, k in enumerate(cols + ("naive",)):
        ax = fig.add_subplot(top[0, j])
        ia = ag.index(k)
        ax.plot([le[0], le[-1]], [le[0], le[-1]], color="k", lw=1., ls=(0, (4, 3)), zorder=1)
        _violins(ax, P["vio_y"][ia, 0], P["vio_d"][ia, 0], P["vio_med"][ia, 0], pos, half, COL[k],
                 .75 if k == "bayes" else .6)
        _logx(ax)
        ax.set_ylim(ylo, yhi)
        ax.set_yticks([-1, 0, 1, 2])
        ax.set_yticklabels(["0.1", "1", "10", "100"])
        ax.set_xlabel(xlab, fontsize=FL)
        lab(ax, j, r"distance after, $d_H(p_t, q_t)$ (nats)")
        if k == "naive":
            ax.text(.04, .96, "after = before in every pair: the same\nobservation shifts both log beliefs alike",
                    transform=ax.transAxes, ha="left", va="top", fontsize=FX)
        _title(ax, "abcd"[j], f"{NAME[k]}: the distance after the observation",
               r"$d_H(p, q) = \mathrm{max}_r\,\log(p_r/q_r) - \mathrm{min}_r\,\log(p_r/q_r)$" + "\n" +
               r"$d_H(p_t, q_t)$ against $d_H(p_{t-1}, q_{t-1})$; violins: the cases of each bin", TS, FT)
    # ── row 2 (John 11-03): the ratio of the distances after / before (violins, geometric mean, shuffle); h: the bulk
    # (SKL) as the same ratio. Log axes labelled with the ratios; naive's d_H ratio is 1 in every pair (T1)
    Rt, L10 = ratios(), np.log(10.)
    a3 = [ag.index(k) for k in cols]
    rlo = float(np.nanmin(Rt["pct"][a3, :, 0])) - .08                     # the bins' .5th / 99.5th percentiles
    rhi = float(np.nanmax(Rt["pct"][a3, :, 3])) + .1
    for j, k in enumerate(cols):
        ax = fig.add_subplot(top[1, j])
        ia = ag.index(k)
        ax.axhline(0, **zero)
        _violins(ax, Rt["vio_y"][ia], Rt["vio_d"][ia], Rt["vio_med"][ia], pos, half, COL[k],
                 .75 if k == "bayes" else .6)
        ax.plot(pos, Rt[f"bin_shuf_{k}"][0] / L10, **_null_style())
        ax.plot(pos, Rt[f"bin_dh_{k}"][0] / L10, **_mean_style(k))
        ax.text(.97, .96, f"geometric mean {_xi(_iv(Rt[f'all_dh_{k}']))}\nshuffled {_x(Rt[f'all_shuf_{k}'][0])}; "
                f"$d_H$ grows in {100 * Rt[f'all_grow_{k}'][0]:.0f}% of cases\n" + beyond_shuffle(k),
                transform=ax.transAxes, ha="right", va="top", fontsize=FX, linespacing=1.3)
        _logx(ax)
        _ratio_y(ax, rlo, rhi)
        ax.set_xlabel(xlab, fontsize=FL)
        lab(ax, j, r"ratio, $d_H$ after / $d_H$ before")
        _title(ax, "efg"[j], f"{NAME[k]}: the ratio of the distances",
               r"$d_H(\mathrm{after})\,/\,d_H(\mathrm{before})$ against $d_H(\mathrm{before})$; violins: the cases of "
               "each bin;\nline: the geometric mean per bin, " + r"$\exp\,\langle\log(\mathrm{after/before})\rangle$",
               TS, FT)
    ax = fig.add_subplot(top[1, 3])
    ax.axhline(0, color="k", lw=.7, zorder=1)
    for k in ("naive", "bayes", "trained", "echo"):
        v, lo, hi = (a / L10 for a in _iv(Rt[f"bin_skl_{k}"]))
        _band(ax, pos, v, lo, hi, COL[k], lw=2.2, zorder=5)
    sk = np.array([_iv(Rt[f"bin_skl_{k}"])[1:] for k in Rt["agents"]]) / L10
    _ratio_y(ax, float(np.nanmin(sk)) - .03, float(np.nanmax(sk)) + .12, ticks=(.7, 1, 1.5, 2, 3, 4))
    _logx(ax)
    ax.set_xlabel(xlab, fontsize=FL)
    ax.set_ylabel("ratio after / before, SKL", fontsize=FL)
    ax.text(.97, .96, "geometric mean over all pairs:\n" + "\n".join(
        f"{NAME[k]} {_xi(_iv(Rt[f'all_skl_{k}']))}" for k in ("bayes", "trained", "echo", "naive")),
        transform=ax.transAxes, ha="right", va="top", fontsize=FX, linespacing=1.3)
    _title(ax, "h", "SKL: close pairs part, the others come closer",
           r"SKL(after) / SKL(before), geometric mean per bin;" + "\n" +
           r"SKL$(p,q) = \Sigma_r(p_r - q_r)\log(p_r/q_r)/2$", TS, FT)
    # ── rows 3-4 (John 11-03: ratios, not differences): the geometric mean of d_H after / before per bin, on log axes
    # labelled with the ratios; naive's ratio is 1 in every pair (T1). Row 3: both close or only one (definitions of the
    # distance); l: the other variable's thirds (matched)
    dls = {"var": "-", "avg": (0, (5, 2.2)), "close": (0, (1, 1.3)), "far": (0, (8, 1.8, 1.4, 1.8))}
    dlw = {"var": 1.3, "avg": 2., "close": 2.5, "far": 2.2}
    dlab = {"var": "one variable, all pairs", "avg": "the pair's average", "close": "other variable close",
            "far": "other variable far"}
    curves3 = {k: (("var", Rt[f"bin_dh_{k}"], Rt[f"n_{k}"]), ("avg", Rt[f"avg_{k}"], Rt[f"avg_n_{k}"]),
                   ("close", Rt[f"cond_{k}"][:, 0], Rt[f"cond_n_{k}"][0]),
                   ("far", Rt[f"cond_{k}"][:, 2], Rt[f"cond_n_{k}"][2])) for k in cols}
    drawn = np.concatenate([np.where(n_ >= MIN_UNITS, a[0], np.nan) for k in cols for _, a, n_ in curves3[k]] +
                           [Rt[f"ent_{k}"][0].ravel() for k in cols]) / L10
    y34 = (min(0., np.nanmin(drawn)) - .025, np.nanmax(drawn) + .05)                # the drawn curves (full sample)
    std = {(nm, k): _iv(Rt[f"{nm}_std_{k}"].T) for nm in ("cond3", "ent5") for k in cols}
    ylp = (min(0., min(v[1].min() for v in std.values()) / L10) - .002,
           max(v[2].max() for v in std.values()) / L10 + .006)
    rlab = r"ratio, $d_H$ after / $d_H$ before"
    for j, k in enumerate(cols):
        ax = fig.add_subplot(top[2, j])
        ax.axhline(0, **zero)
        for nm, arr, n_ in curves3[k]:
            m_, l_, h_ = (a / L10 for a in _iv(arr))
            _band(ax, pos, m_, l_, h_, COL[k], ok=n_ >= MIN_UNITS, alpha=.12, lw=dlw[nm], ls=dls[nm], zorder=5)
        ax.text(.97, .96, f"far / close third, matched: {_xi(_iv(Rt[f'cond3_hilo_{k}']))}\n"
                f"pair average 0.25-1 nat: {_x(Rt[f'close_avg_{k}'][0], 2)}\n"
                f"one variable 0.25-1 nat: {_x(Rt[f'close_var_{k}'][0], 2)}", transform=ax.transAxes, ha="right",
                va="top", fontsize=FX, linespacing=1.3)
        if k == "echo":
            ax.legend(handles=[Line2D([], [], color=COL[k], lw=dlw[n], ls=dls[n], label=dlab[n]) for n in dls],
                      loc="center right", bbox_to_anchor=(1., .56), ncol=2, frameon=False, fontsize=FX - .5,
                      handlelength=3., columnspacing=1.2, borderaxespad=.3)
        _logx(ax)
        _ratio_y(ax, *y34, ticks=(1, 1.5, 2, 3))
        ax.set_xlabel("distance before (nats): the variable's own, or the pair's average", fontsize=FL)
        lab(ax, j, rlab)
        _title(ax, "ijk"[j], f"{NAME[k]}: both variables close, or only one?",
               r"geometric mean of $d_H$ after / before per bin;" + "\n" +
               r"'other': the other variable's $d_H$ before", TS, FT)
    ax = fig.add_subplot(top[2, 3])
    ax.axhline(0, **zero)
    xs = np.arange(1, 4)
    for k in cols:
        v, lo, hi = (a / L10 for a in std[("cond3", k)])
        ax.fill_between(xs, lo, hi, color=COL[k], alpha=.2, lw=0)
        ax.plot(xs, v, "o-", color=COL[k], lw=2.2, ms=6, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(["close\n(lowest third)", "middle", "far\n(highest third)"])
    ax.set_xlim(.75, 3.25)
    _ratio_y(ax, *ylp, ticks=(1, 1.02, 1.04, 1.06, 1.08))
    ax.set_xlabel("the other variable's distance before", fontsize=FL)
    ax.set_ylabel("ratio, matched (geometric mean)", fontsize=FL)
    ax.spines[["top", "right"]].set_visible(False)
    _direct_labels(ax, {k: std[("cond3", k)][0] / L10 for k in cols}, xs, 1, FX)
    _title(ax, "l", "The other variable's distance, all bins",
           r"geometric mean of $d_H$ after / before per third," + "\n" +
           r"matched on step $t$ and own $d_H$ before", TS, FT)
    # ── row 4: the entanglement of the observation; p: fifths (matched)
    ecol = plt.cm.viridis(np.linspace(0., .9, 5))
    for j, k in enumerate(cols):
        ax = fig.add_subplot(top[3, j])
        ax.axhline(0, **zero)
        v5, lo5, hi5 = (a / L10 for a in _iv(Rt[f"ent_{k}"]))                       # (5 fifths, 40 bins) each
        for g in range(5):
            _band(ax, pos, v5[g], lo5[g], hi5[g], ecol[g], alpha=.12, lw=2.1, zorder=5)
        ax.text(.97, .96, f"most / least entangled fifth, matched:\n{_xi(_iv(Rt[f'ent5_hilo_{k}']))}",
                transform=ax.transAxes, ha="right", va="top", fontsize=FX, linespacing=1.3)
        if k == "echo":
            ax.legend(handles=[Line2D([], [], color=ecol[g], lw=2.6, label=lb_) for g, lb_ in
                               enumerate(("1 least", "2", "3", "4", "5 most"))],
                      loc="center right", bbox_to_anchor=(1., .56), ncol=5, frameon=False, fontsize=FX - .5,
                      title="fifth of entanglement", title_fontsize=FX - .5, handlelength=1.4, columnspacing=.9,
                      handletextpad=.4, borderaxespad=.3)
        _logx(ax)
        _ratio_y(ax, *y34, ticks=(1, 1.5, 2, 3))
        ax.set_xlabel(xlab, fontsize=FL)
        lab(ax, j, rlab)
        _title(ax, "mno"[j], f"{NAME[k]}: split by the observation's entanglement",
               r"geometric mean of $d_H$ after / before per bin and fifth," + "\n" +
               r"standardised over steps $t$", TS, FT)
    ax = fig.add_subplot(top[3, 3])
    ax.axhline(0, **zero)
    xs = np.arange(1, 6)
    for k in cols:
        v, lo, hi = (a / L10 for a in std[("ent5", k)])
        ax.fill_between(xs, lo, hi, color=COL[k], alpha=.2, lw=0)
        ax.plot(xs, v, "-", color=COL[k], lw=2.2, zorder=4)
        ax.scatter(xs, v, s=46, c=ecol, edgecolors=COL[k], linewidths=1.4, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["1\nleast", "2", "3", "4", "5\nmost"])
    ax.set_xlim(.7, 5.3)
    _ratio_y(ax, *ylp, ticks=(1, 1.02, 1.04, 1.06, 1.08))
    ax.set_xlabel("entanglement of the observation (fifth)", fontsize=FL)
    ax.set_ylabel("ratio, matched (geometric mean)", fontsize=FL)
    ax.spines[["top", "right"]].set_visible(False)
    _direct_labels(ax, {k: std[("ent5", k)][0] / L10 for k in cols}, xs, 2, FX)
    _title(ax, "p", "Entanglement, all bins",
           r"geometric mean of $d_H$ after / before per fifth," + "\n" + r"matched on step $t$ and $d_H$ before",
           TS, FT)
    # ── row 5: a flipped observation through training (Trained RNN), as growth factors chi(s + lag, s) / chi(s, s) of
    # each flip set (John 11-03); flips at s = 1, 3, 5, 7, 9 (td_tt_flips's), harvested per flip set (fratio_harvest)
    acc = st["acc"]
    Fr = fratio_stats()
    Rf = Fr["refs"]
    lx = np.log10(Fr["x"])

    def acc_bands(ax):
        for c in ("c0", "c50"):
            ax.axvspan(np.log10(acc[c][1]), np.log10(acc[c][2]), color=C_ACC, alpha=.3, lw=0, zorder=0)
    ax = fig.add_subplot(bot[0, 0])
    lag_cols = plt.cm.viridis(np.linspace(.1, .9, len(Q_LAGS)))
    for i, L in enumerate(Q_LAGS):
        v, lo, hi = (a / L10 for a in _iv(Fr["q"][:, :, i].T))
        ax.fill_between(lx, lo, hi, color=lag_cols[i], alpha=.22, lw=0)
        ax.plot(lx, v, color=lag_cols[i], lw=1.8, label=f"lag {L}")
    qb, qa = Rf["bayes"]["q"][0, -1], Rf["adf"]["q"][0, -1]
    ax.axhline(qb / L10, color="k", lw=1.3)
    ax.axhline(qa / L10, color="k", lw=1.3, ls=ADF_LS)
    ax.axhline(0, **zero)
    acc_bands(ax)
    qtop = float(np.nanmax(Fr["q"][:, 1:])) / L10
    _ratio_y(ax, -.06, qtop + .2, ticks=(1, 1.5, 2, 3, 4, 5))
    _trainx(ax, FL)
    ax.set_ylabel(r"growth factor, $\chi(s + \mathrm{lag}, s)\,/\,\chi(s, s)$", fontsize=FL)
    l1 = ax.legend(loc="upper right", bbox_to_anchor=(1., 1.), ncol=2, frameon=False, fontsize=FX - .5,
                   handlelength=1.4, labelspacing=.25, columnspacing=.8)
    ax.add_artist(l1)
    ax.legend(handles=[Line2D([], [], color="k", lw=1.3, label=f"Bayes ({_x(qb, 2)})"),
                       Line2D([], [], color="k", lw=1.3, ls=ADF_LS, label=f"fact.-state Bayes ({_x(qa, 2)})")],
              loc="upper right", bbox_to_anchor=(1., .76), frameon=False, fontsize=FX - .5, labelspacing=.2,
              borderpad=.1, borderaxespad=0., handlelength=2., title="exact observers, lag 20:",
              title_fontsize=FX - .5, alignment="right")
    _title(ax, "q", "Growth factor of one flipped bit's effect",
           r"$\chi(t, s) = d_H$(intact, flipped marginals) at step $t$; flip" + "\n" +
           r"at step $s$ = 1, 3, 5, 7, 9; geometric mean over flips", TS, FT)
    # r: growth factors by bits flipped and lag bin, at four stages and for the exact observers
    sub = bot[0, 1:3].subgridspec(1, 6, wspace=.1)
    stages = [(raw, int(np.flatnonzero(Fr["raw"] == raw)[0])) for raw in MAP_RAW]
    maps = [(i, f"{Fr['x'][i]:,.0f} episodes" if raw != MAP_RAW[-1] else f"{Fr['x'][i]:,.0f} (end)")
            for raw, i in stages] + [("bayes", "Bayes"), ("adf", "fact.-state Bayes")]
    bcol = plt.cm.viridis(np.linspace(.15, .9, 4))
    lb = F["lag_bins"]
    ks = np.arange(1, 6)
    rr = [Fr["r"][i] for _, i in stages] + [Rf["bayes"]["r"], Rf["adf"]["r"]]
    r_lo = min(0., min(float(np.nanpercentile(a[1:], 5, 0).min()) for a in rr) / L10) - .03
    r_hi = max(float(np.nanpercentile(a[1:], 95, 0).max()) for a in rr) / L10 + .04
    for j, ((key, ttl), arr) in enumerate(zip(maps, rr)):
        ax = fig.add_subplot(sub[0, j])
        G, lo, hi = (a / L10 for a in _iv(arr))                                         # (5 k, 4 lag bins)
        ax.axhline(0, **zero)
        for b in range(len(lb)):
            ax.fill_between(ks, lo[:, b], hi[:, b], color=bcol[b], alpha=.25, lw=0)
            ax.plot(ks, G[:, b], "o-", color=bcol[b], lw=1.8, ms=3.2, clip_on=True)
        ax.set_xlim(.6, 5.4)
        ax.set_xticks(ks)
        ax.spines[["top", "right"]].set_visible(False)
        _ratio_y(ax, r_lo, r_hi, ticks=(1, 1.5, 2, 3, 4, 5))
        if j:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"growth factor of $d_H$", fontsize=FL)
        ax.set_title(ttl, fontsize=FX, pad=3)
        if j == 0:
            ax.annotate("even k lower:\ncount code", xy=(2, G[1, 3]), xytext=(1.05, .2), fontsize=FX - 1,
                        textcoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="->", lw=.9, color="k", shrinkB=3))
            ax.text(0, 1.21, "r   Growth factor by lag and bits flipped: no boundary; an early parity zig-zag in k "
                    "(4 checkpoints shown)",
                    transform=ax.transAxes, fontsize=TS, ha="left", va="bottom")
            ax.text(0, 1.085, r"$\chi(s{+}\mathrm{lag}, s)\,/\,\chi(s, s)$ of a k-bit flip per lag bin "
                    "(geometric mean); "
                    "Trained RNN at 4 stages, then the exact observers (naive: 1)", transform=ax.transAxes,
                    fontsize=FT, ha="left", va="bottom")
        if j == 2:
            ax.set_xlabel("bits flipped at once, k (of the 5 observations)", fontsize=FL)
            ax.xaxis.set_label_coords(1.05, -.13)
        if j == 3:
            ax.legend(handles=[Line2D([], [], color=bcol[b], lw=1.8, marker="o", ms=3.2, label=f"lags {a}-{c}")
                               for b, (a, c) in enumerate(lb)][::-1], loc="upper center", frameon=False,
                      fontsize=FX - .5, handlelength=1.5, title="lag after the flip", title_fontsize=FX - .5)
    # s: the count classes (2-bit flips that change / keep the number of 1s)
    ax = fig.add_subplot(bot[0, 3])
    cs = {g: _iv(Fr[g][:, :, 4].T) for g in ("change", "keep")}
    for g, ls in (("change", "-"), ("keep", (0, (3, 1.6)))):
        v, lo, hi = (a / L10 for a in cs[g])
        ax.fill_between(lx, lo, hi, color=COL["trained"], alpha=.2, lw=0)
        ax.plot(lx, v, color=COL["trained"], ls=ls, lw=2.)
    acc_bands(ax)
    ax.axhline(0, **zero)
    ax.text(.995, .97, "exact, change / keep:\n" + "\n".join(
        f"{nm} {_x(Rf[o]['change'][0, 4], 2)} / {_x(Rf[o]['keep'][0, 4], 2)}"
        for o, nm in (("bayes", "Bayes"), ("adf", "fact.-state Bayes"), ("naive", "naive"))),
        transform=ax.transAxes, ha="right", va="top", fontsize=FX - .5, linespacing=1.3)
    i0 = {xx: int(np.argmin(np.abs(Fr["x"] - xx))) for xx in (20, 40, 60, 100)}
    imm = np.exp(Fr["index"][:, 0, 0])                                                     # change / keep, chi(s, s)
    ipk = int(np.argmax(cs["keep"][0]))
    ax.annotate(f"keep-the-count flips start\n{_x(np.log(imm[i0[20]]), 0)}-{imm[i0[40]]:.0f} smaller at 20-40 "
                f"episodes\nand {_x(np.log(min(imm[i0[100]], imm[i0[60]])), 0)}-{max(imm[i0[60]], imm[i0[100]]):.0f}"
                " smaller at 60-100,\nthen grow the most",
                xy=(lx[ipk], cs["keep"][0][ipk] / L10), xytext=(.03, .07), textcoords="axes fraction",
                fontsize=FX - 1, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", lw=.9, color="k", shrinkA=2, shrinkB=3, relpos=(.62, 1.)))
    s_hi = max(float(np.nanmax(cs[g][2])) for g in cs) / L10
    _ratio_y(ax, -.06, s_hi + .12, ticks=(1, 2, 3, 5, 10))
    _trainx(ax, FL)
    ax.set_ylabel(r"growth factor, 2-bit flips", fontsize=FL)
    ax.legend(handles=[Line2D([], [], color=COL["trained"], lw=2., label=r"change the count of 1s"),
                       Line2D([], [], color=COL["trained"], lw=2., ls=(0, (3, 1.6)), label="keep the count")],
              loc="center right", bbox_to_anchor=(1., .52), frameon=False, fontsize=FX - .5, handlelength=2.2,
              borderaxespad=.1)
    _title(ax, "s", "Kept count: small at first, then amplified most",
           "2-bit flips, lags 18-21 (as in r); keep = one\n" +
           r"0$\to$1 and one 1$\to$0; change = net $\pm$2", TS, FT)
    # ── header: suptitle, legends, section labels; footnote
    fig.suptitle(suptitle(), fontsize=TS + 2, y=.993, linespacing=1.5)
    hd = [tuple(Line2D([], [], **_mean_style(k)) for k in cols), Line2D([], [], ls="none", marker="o", ms=5,
                                                                        mfc="w", mec="k", mew=.6),
          Line2D([], [], color="k", lw=1., ls=(0, (4, 3))), Line2D([], [], **_null_style()),
          Line2D([], [], **zero), tuple(Patch(color=COL[k], alpha=.25, lw=0) for k in cols)]
    lbl = ["geometric mean per bin (Bayes, Trained RNN, ESN)", "median", "after = before",
           "shuffled: the step's difference of another case, random orientation",
           "naive: ratio 1 (no change in any pair)", "5-95% over environment resamples"]
    fig.legend(hd, lbl, loc="upper center", bbox_to_anchor=(.5, .942), ncol=6, frameon=False, fontsize=FX,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=.3)}, handlelength=3.4, columnspacing=1.6)
    fig.text(.008, .912, "Pairs of marginal beliefs given the same observation vector (end of training)",
             fontsize=TS + 2, ha="left", va="bottom", weight="bold")
    fig.text(.008, .262, "A flipped observation through training (Trained RNN)", fontsize=TS + 2, ha="left",
             va="bottom", weight="bold")
    fig.legend(handles=[Patch(color=C_ACC, alpha=.45, label="Trained RNN accuracy passes naive / half-way to Bayes "
                              "(5-95%)")], loc="lower right", bbox_to_anchor=(.985, .258), frameon=False, fontsize=FX)
    fig.text(.5, .006, footnote(st), ha="center", va="bottom", fontsize=FX - .5, linespacing=1.55)
    path = fig_path(FIG) if path is None else path
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    old = fig_path(OLD_FIG)
    if path == fig_path(FIG) and os.path.exists(old):
        os.remove(old)
        print("deleted the superseded", old)
    return path


def _direct_labels(ax, vals, xs, at, FX):
    """agent names written next to their lines (at index `at` of xs; offsets in shares of the y range), and naive's
    line (log ratio 0: a ratio of 1)"""
    lo, hi = ax.get_ylim()
    sp = hi - lo
    order = sorted(vals, key=lambda k: vals[k][at])
    for i, k in enumerate(order):
        y = vals[k][at]
        below = i == 0 and (y - lo) / sp > .2
        ax.text(xs[at], y + (-.06 if below else .05) * sp, NAME[k], color=COL[k], fontsize=FX + .5, ha="center",
                va="top" if below else "bottom", path_effects=_halo())
    ax.text(xs[-1], .02 * sp, "naive: 1", color="0.4", fontsize=FX - .5, ha="right", va="bottom")


def _halo():
    import matplotlib.patheffects as pe
    return [pe.Stroke(linewidth=3, foreground="w"), pe.Normal()]


def _null_style():
    import matplotlib.patheffects as pe
    return dict(color=SHUF_C, lw=1.6, ls=(0, (3.5, 2)), zorder=7, path_effects=[pe.Stroke(linewidth=3., foreground="w"),
                                                                               pe.Normal()])


SUPTITLE_SPLITS = ("Bayes's and the Trained RNN's pairs separate most when their beliefs about the other variable "
                   "differ (i-l) and after entangled observations (m-p); the ESN's hardly depend on either")


def _resolved(k):
    """per bin, is the geometric mean ratio above / below the shuffled step's? (the paired difference of the mean ln r
    per bin; 5-95% over the shared resamples excludes 0)"""
    Rt = ratios()
    _, lo, hi = _iv(Rt[f"bin_dh_{k}"] - Rt[f"bin_shuf_{k}"])
    return lo > 0, hi < 0


def beyond_shuffle(k):
    """e-g's third text line: the bins where the ratio exceeds the shuffled step's"""
    e = ratios()["edges40"]
    up, down = _resolved(k)
    i = np.flatnonzero(up)
    if not len(i):
        txt = "not resolved from the shuffled in any bin"
    elif len(i) == len(up):
        txt = "above the shuffled at every distance"
    else:
        a, b = i[0], i[-1]
        where = f"from {e[a]:.1f} nats on" if b == len(up) - 1 else f"at {e[a]:.1f}-{e[b + 1]:.0f} nats"
        txt = f"above the shuffled {where}" + ("" if len(i) == b - a + 1 else " (with gaps)")
    return txt + (f"; below it in {down.sum()} bins" if down.any() else "")


def suptitle():
    """the header (four lines): the ratios of row 2 (pairs_ratio.npz), SUPTITLE_SPLITS, and row 5's (flips/ratio)"""
    Rt, c = ratios(), pairs()["centres40"]
    i05, i32 = (int(np.argmin(np.abs(np.log(c / x)))) for x in (.5, 32.))
    g = {k: np.exp(Rt[f"bin_dh_{k}"][0]) for k in AG}
    first = {k: Rt["edges40"][np.flatnonzero(_resolved(k)[0])[0]] if _resolved(k)[0].any() else np.nan for k in AG}
    two = (f"from {first['bayes']:.1f} nats on" if np.isclose(first["bayes"], first["trained"]) else
           f"from {first['bayes']:.1f} / {first['trained']:.1f} nats on")
    esn = "at every distance" if _resolved("echo")[0].all() else f"from {first['echo']:.1f} nats on"
    Fr = fratio_stats()
    q20 = np.exp(Fr["q"][:, 0, Q_LAGS.index(20)])
    i20, i880 = (int(np.argmin(np.abs(Fr["x"] - xx))) for xx in (20, 880))
    imm = np.exp(Fr["index"][:, 0, 0])                                     # change / keep: the immediate impacts
    i40 = int(np.argmin(np.abs(Fr["x"] - 40)))
    return ("Two marginal beliefs given the same observation (a-h): the range of their log-ratio, $d_H$, grows by a "
            "factor that falls with the distance before (naive: no change),\n"
            rf"$\times${g['bayes'][i05]:.1f} / {g['trained'][i05]:.1f} / {g['echo'][i05]:.1f} at {c[i05]:.1f} nats, "
            rf"$\times${g['bayes'][i32]:.2f} / {g['trained'][i32]:.2f} / {g['echo'][i32]:.2f} at {c[i32]:.0f} nats "
            f"(Bayes / Trained RNN / ESN); it exceeds a shuffled step's {two} (ESN: {esn})\n" + SUPTITLE_SPLITS +
            "\nThrough training (q-s), a flipped bit's effect is amplified "
            rf"$\times${q20[i20]:.1f} over 20 steps at {Fr['x'][i20]:.0f} episodes, $\times${q20[i880]:.1f} at "
            rf"{Fr['x'][i880]:.0f} and $\times${q20[-1]:.1f} at the end "
            rf"(Bayes $\times${np.exp(Fr['refs']['bayes']['q'][0, -1]):.2f}); "
            rf"kept-count flips start $\times${min(imm[i20], imm[i40]):.0f}-{max(imm[i20], imm[i40]):.0f} smaller, "
            "then grow the most")


def footnote(st):
    """the data counts (six lines)"""
    P = pairs()
    n_in = {k: int(P["var_n"][list(P["agents"]).index(k)].sum()) for k in P["agents"]}
    Fr = fratio_stats()
    return "\n".join((
        "Pairs (a-p): 50 test environments x 200 probes (100 cells x 2 streams of 30 steps). A pair = two probes of "
        "the same environment and step t (2-30) with the identical 5-observation vector at t and different counts "
        "before t;",
        f"a case = one variable of a pair-step: {int(P['n_pairsteps']):,} pair-steps x 2 variables = "
        f"{int(P['n_units']):,} cases per agent. 40 bins of the distance before (log-spaced, {P['edges40'][0]:.2f}-"
        f"{P['edges40'][-1]:.0f} nats); cases inside the bins: " + ", ".join(f"{NAME[k]} {n_in[k]:,}"
                                                                          for k in P["agents"]) + ".",
        "Ratios (e-s): geometric means, exp of the mean log ratio of each case; e-p: the cases inside the bins. "
        "Violins: KDE on at most 20,000 cases per bin; medians: all cases in the bin. Shuffled: donor cases of the "
        "same variable, entanglement fifth, other-variable third and bin; 5 seeds.",
        "i-l: thirds of the other variable's distance within each (step, own bin) cell; pair average: bins of the mean "
        f"of the two variables' distances; curves where a group holds $\\geq${MIN_UNITS} cases. m-p: the observation's "
        "entanglement (mean of the two variables and the two probes), fifths over pair-steps.",
        "Matched: per (step, bin) cell with $\\geq$20 cases in every group, cell means averaged with the cell's count "
        f"as weight. Flips (q-s): Trained RNN (N = 1000, seed 20261029), {len(Fr['x'])} checkpoints; 1,000 probes (20 "
        "per environment) x 30 steps; every subset of the 5 observations (31 flip sets)",
        "at s = 1, 3, 5, 7, 9 (q: the 5 single flips); each flip set's log ratio, then means over flip sets, flip "
        "steps, "
        "both variables and probes. [ ] and bands: 5-95% over 1,000 resamples of the 50 test environments. "
        "Bayes, naive, factorized-state Bayes: exact."))


# ═══════════════════════════════════════════════════════════════════
# Numbers (every number of the figure and the docstring) and checks
# ═══════════════════════════════════════════════════════════════════

def _t(t, d=3):
    """(value, lo, hi) -> 'value [lo, hi]'"""
    return f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"


def _tx(t, d=3):
    """(ln value, ln lo, ln hi) -> 'xvalue [xlo, xhi]' (plain text)"""
    return f"x{np.exp(t[0]):.{d}f} [x{np.exp(t[1]):.{d}f}, x{np.exp(t[2]):.{d}f}]"


def _ivk(a):
    """(K, 1 + N_BOOT, ...) -> value, 5%, 95% per checkpoint"""
    return a[:, 0], np.nanpercentile(a[:, 1:], 5, 1), np.nanpercentile(a[:, 1:], 95, 1)


def q_last_above(level, lag=20):
    """the last checkpoint at which the Trained RNN's growth factor at `lag` (q) lies above a fixed `level` (a log
    ratio, Bayes's): (value, 5%, 95%) over the resamples of the curve (td_tt_gap.interval)"""
    Fr = fratio_stats()
    c = Fr["q"][:, :, Q_LAGS.index(lag)]                                              # (K, 1 + N_BOOT)
    last = np.array([Fr["x"][np.flatnonzero(c[:, b] > level)[-1]] if (c[:, b] > level).any() else np.nan
                     for b in range(c.shape[1])])
    lo, hi, _ = gap.interval(last[1:])
    return float(last[0]), lo, hi


def numbers():
    """every number of the figure and the docstring (ratios: x = exp of the mean log ratio; [5-95%])"""
    P = pairs()
    st = stats()
    U = st["units"]
    Rt, Fr = ratios(), fratio_stats()
    ag = list(P["agents"])
    cen, e40 = P["centres40"], P["edges40"]
    print("== caches")
    for f in (PAIRS, UNITS, RATIO, os.path.join(FRATIO, "refs.npz"), FLIPS):
        print(f"  {os.path.relpath(f, gap.TT)}: {os.path.getsize(f) / 2 ** 20:.1f} MB, modified "
              f"{time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(f)))}")
    print(f"  flips/ratio: {len(Fr['x'])} Trained RNN checkpoints ({Fr['x'][0]:,.0f}-{Fr['x'][-1]:,.0f} episodes)")
    print("== pairs: data")
    print(f"  {int(P['n_pairsteps']):,} pair-steps x 2 variables = {int(P['n_units']):,} cases per agent; 40 bins "
          f"{e40[0]:.3f}-{e40[-1]:.1f} nats (pooled percentiles {P['edge_pct']}; width x{e40[1] / e40[0]:.3f})")
    for k in ag:
        ia = ag.index(k)
        print(f"  {NAME[k]:11s} inside the bins {int(P['var_n'][ia].sum()):,} (below {int(P['var_out'][ia, 0]):,}, "
              f"above {int(P['var_out'][ia, 1]):,}); fewest per bin {int(P['var_n'][ia].min()):,}")
    print("== a-d: log10 d_H after, 0.1st / 99.9th percentile of the cases inside the bins")
    for k in ag:
        print(f"  {NAME[k]:11s} {U[k]['after_pct'][0]:+.3f} / {U[k]['after_pct'][1]:+.3f}")
    sel = [int(np.argmin(np.abs(np.log(cen / x)))) for x in (.27, .5, 1, 2, 3, 10, 32, 94)]
    print("== e-g: d_H after / before, cases inside the bins (geometric means)")
    for k in AG:
        up, down = _resolved(k)
        d = _iv(Rt[f"bin_dh_{k}"] - Rt[f"bin_shuf_{k}"])
        print(f"  {NAME[k]:11s} {_tx(_iv(Rt[f'all_dh_{k}']))}; shuffled {_tx(_iv(Rt[f'all_shuf_{k}']))}; actual / "
              f"shuffled {_tx(_iv(Rt[f'all_dh_{k}'] - Rt[f'all_shuf_{k}']))}; grows in "
              f"{Rt[f'all_grow_{k}'][0]:.3f}; above the shuffled in {up.sum()} bins ({e40[np.argmax(up)]:.2f} nats "
              f"first), below in {down.sum()}")
        print(f"  {NAME[k]:11s} per bin (" + " / ".join(f"{cen[i]:.2f}" for i in sel) + " nats): " + " / ".join(
            f"{np.exp(Rt[f'bin_dh_{k}'][0, i]):.3f}" for i in sel) + "; shuffled " + " / ".join(
            f"{np.exp(Rt[f'bin_shuf_{k}'][0, i]):.3f}" for i in sel))
        print(f"  {NAME[k]:11s} actual / shuffled per bin: " + "; ".join(
            f"{cen[i]:.2f} x{np.exp(d[0][i]):.3f} [x{np.exp(d[1][i]):.3f}, x{np.exp(d[2][i]):.3f}]" for i in sel))
        ia = ag.index(k)
        print(f"  {NAME[k]:11s} axis: lowest .5th percentile of log10 r {np.nanmin(Rt['pct'][ia, :, 0]):+.3f}, "
              f"highest 99.5th {np.nanmax(Rt['pct'][ia, :, 3]):+.3f}")
    print("== h: SKL after / before (geometric means), cases inside the bins")
    for k in ("bayes", "trained", "echo", "naive"):
        v = _iv(Rt[f"bin_skl_{k}"])
        below, above = np.flatnonzero(v[2] < 0), np.flatnonzero(v[1] > 0)
        ab = f"above 1 (5% > 1) up to {e40[above[-1] + 1]:.2f} nats" if len(above) else "never above 1"
        bl = f"below 1 (95% < 1) in {len(below)} bins, from {e40[below[0]]:.2f} nats" if len(below) else "never below 1"
        print(f"  {NAME[k]:11s} {_tx(_iv(Rt[f'all_skl_{k}']))}; per bin " + " / ".join(
            f"{np.exp(v[0][i]):.3f}" for i in sel) + f"; {ab}; {bl}")
    print("== i-l: both close or only one (thirds of the other variable's d_H before within (step, own bin) cells)")
    for k in AG:
        s3 = _iv(Rt[f"cond3_std_{k}"].T)
        print(f"  {NAME[k]:11s} far / close matched {_tx(_iv(Rt[f'cond3_hilo_{k}']))} (cells cover "
              f"{float(Rt[f'cond3_share_{k}']):.3f}); thirds " + " / ".join(
                  _tx((s3[0][j], s3[1][j], s3[2][j])) for j in range(3)))
        print(f"  {NAME[k]:11s} 0.25-1 nat: pair average {_tx(_iv(Rt[f'close_avg_{k}']))} "
              f"({int(Rt[f'close_avg_n_{k}']):,} pair-steps), one variable {_tx(_iv(Rt[f'close_var_{k}']))} "
              f"({int(Rt[f'close_var_n_{k}']):,} cases)")
        c, n = Rt[f"cond_{k}"], Rt[f"cond_n_{k}"]
        print(f"  {NAME[k]:11s} per bin close / far: " + "; ".join(
            f"{cen[i]:.2f} {np.exp(c[0, 0, i]):.2f} / {np.exp(c[0, 2, i]):.2f}" for i in sel
            if n[0, i] >= MIN_UNITS and n[2, i] >= MIN_UNITS))
    print(f"  Trained RNN / Bayes, far / close, one population: {_tx(_iv(Rt['cond3_hilo_trained-bayes']))} "
          f"({int(Rt['cond3_ncommon'])} common cells)")
    print("== m-p: the observation's entanglement (fifths over pair-steps)")
    for k in AG:
        s5 = _iv(Rt[f"ent5_std_{k}"].T)
        print(f"  {NAME[k]:11s} most / least matched {_tx(_iv(Rt[f'ent5_hilo_{k}']))} (cells cover "
              f"{float(Rt[f'ent5_share_{k}']):.3f}); fifths " + " / ".join(
                  f"x{np.exp(s5[0][j]):.3f}" for j in range(5)) + f" (least [x{np.exp(s5[1][0]):.3f}, "
              f"x{np.exp(s5[2][0]):.3f}], most [x{np.exp(s5[1][4]):.3f}, x{np.exp(s5[2][4]):.3f}])")
        e_ = Rt[f"ent_{k}"][0]
        first = cen[np.argmax(np.isfinite(e_[0]))]
        print(f"  {NAME[k]:11s} lines from {first:.2f} nats; per bin least / most: " + "; ".join(
            f"{cen[i]:.2f} {np.exp(e_[0, i]):.2f} / {np.exp(e_[4, i]):.2f}" for i in sel if np.isfinite(e_[0, i])))
    print(f"  Trained RNN / Bayes, most / least, one population: {_tx(_iv(Rt['ent5_hilo_trained-bayes']))} "
          f"({int(Rt['ent5_ncommon'])} common cells)")
    print("== q: one flipped bit, growth factor chi(s + lag, s) / chi(s, s) (5 single flips x s = 1, 3, 5, 7, 9)")
    Rf = Fr["refs"]
    for o in ("bayes", "naive", "adf"):
        print(f"  {o:6s} " + ", ".join(f"lag {L} {_tx(_iv(Rf[o]['q'][:, j]))}" for j, L in enumerate(Q_LAGS)))
    x = Fr["x"]
    v, lo, hi = _ivk(Fr["q"])
    for xx in (10, 20, 40, 100, 180, 240, 320, 880, 1680, 4200, x[-1]):
        i = int(np.argmin(np.abs(x - xx)))
        print(f"  Trained RNN {x[i]:7,.0f}: " + ", ".join(f"lag {L} {_tx((v[i, j], lo[i, j], hi[i, j]), 2)}"
                                                        for j, L in enumerate(Q_LAGS)))
    j20 = Q_LAGS.index(20)
    ip = int(np.argmax(v[:, j20]))
    cr = q_last_above(Rf["bayes"]["q"][0, j20])
    print(f"  lag 20: largest x{np.exp(v[ip, j20]):.2f} at {x[ip]:,.0f} episodes; last above Bayes "
          f"(x{np.exp(Rf['bayes']['q'][0, j20]):.3f}) at {cr[0]:,.0f} [{cr[1]:,.0f}-{cr[2]:,.0f}]; below 1 at the end: "
          f"{_tx((v[-1, j20], lo[-1, j20], hi[-1, j20]))}")
    print("== r: growth factor by k (rows) and lag bin (columns 1-3 / 6-9 / 12-15 / 18-21)")
    for raw in MAP_RAW:
        i = int(np.flatnonzero(Fr["raw"] == raw)[0])
        G = np.exp(Fr["r"][i, 0])
        print(f"  Trained RNN {x[i]:7,.0f}: " + "; ".join(f"k={k + 1} " + " / ".join(f"{g:.2f}" for g in G[k])
                                                        for k in range(5)) +
              "; immediate chi(s, s) per k " + " / ".join(f"{np.exp(a):.3f}" for a in Fr["r0"][i, 0]))
    for o in ("bayes", "naive", "adf"):
        G = np.exp(Rf[o]["r"][0])
        print(f"  {o:18s} " + "; ".join(f"k={k + 1} " + " / ".join(f"{g:.2f}" for g in G[k]) for k in range(5)))
    L_ = Fr["r"][:, :, :, 3]                                               # (K, 1 + N_BOOT, 5 k), lags 18-21
    inc = np.diff(np.concatenate([np.zeros(L_.shape[:2] + (1,)), L_], 2), axis=2)       # per-bit increments of ln
    alt = ((inc[..., 2] - (inc[..., 1] + inc[..., 3]) / 2) - (inc[..., 3] - (inc[..., 2] + inc[..., 4]) / 2)) / 2
    kmax = np.abs(np.diff(inc[..., 1:], axis=2)).max(2)
    late = x >= 200
    for nm, a in (("alternation (zig-zag in k, log units)", alt), ("largest change of the per-bit increment, k = 2-5",
                                                                   kmax)):
        av, al, ah = _ivk(a)
        i_ = int(np.argmax(np.abs(av[late])))
        print(f"  {nm}: " + ", ".join(f"{x[i]:,.0f} {av[i]:+.3f} [{al[i]:+.3f}, {ah[i]:+.3f}]" for i in range(1, 7)) +
              f"; from 200 episodes on at most {np.abs(av[late]).max():.3f} (at {x[late][i_]:,.0f})")
    d2 = np.diff(Fr["r"][:, 0], 2, axis=2)                       # (K, 5 k, 2): second difference over lag bins
    print(f"  concavity in lag: largest second difference of the log growth factor over the lag bins "
          f"{d2.max():+.3f} (all checkpoints and k)")
    print("== s: count classes, 2-bit flips (change = net +-2, keep = net 0); growth factor at lags 18-21; immediate "
          "chi(s, s) as geometric means; index = paired change / keep per (probe, flip step)")
    cs = {g: _ivk(Fr[g]) for g in ("change", "keep", "index")}
    for xx in (10, 20, 40, 60, 80, 100, 140, 240, 400, 880, 1200, 4200, x[-1]):
        i = int(np.argmin(np.abs(x - xx)))
        f = lambda g, b: (cs[g][0][i, b], cs[g][1][i, b], cs[g][2][i, b])      # noqa: E731
        print(f"  {x[i]:7,.0f}: change {_tx(f('change', 4), 2)}, keep {_tx(f('keep', 4), 2)}, index "
              f"{_tx(f('index', 4), 2)}; immediate change {np.exp(cs['change'][0][i, 0]):.3f} / keep "
              f"{np.exp(cs['keep'][0][i, 0]):.3f}, index {_tx(f('index', 0), 2)}")
    for o in ("bayes", "naive", "adf"):
        print(f"  {o:6s} change {_tx(_iv(Rf[o]['change'][:, 4]))}, keep {_tx(_iv(Rf[o]['keep'][:, 4]))}, index "
              f"{_tx(_iv(Rf[o]['index'][:, 4]))}; immediate index {_tx(_iv(Rf[o]['index'][:, 0]))}")
    below = np.flatnonzero(cs["index"][2][:, 4] < 0)
    print("  keep grows more than change (index 95% < 1) at: " + ", ".join(f"{x[i]:,.0f}" for i in below))
    above = np.flatnonzero(cs["index"][1][:, 4] > 0)
    print("  change grows more than keep (index 5% > 1) at: " + ", ".join(f"{x[i]:,.0f}" for i in above))
    a = st["acc"]
    print(f"  accuracy crossings: passes naive {a['c0'][0]:,.0f} [{a['c0'][1]:,.0f}-{a['c0'][2]:,.0f}], half-way to "
          f"Bayes {a['c50'][0]:,.0f} [{a['c50'][1]:,.0f}-{a['c50'][2]:,.0f}]")
    print("== cache checks")
    for n_, v_ in zip(Rt["chk_names"], Rt["chk_values"]):
        print(f"  pairs_ratio: {n_}: {v_:.2g}")
    with np.load(os.path.join(FRATIO, "refs.npz")) as Rf_:
        print("  flips/ratio refs: " + ", ".join(f"{o} {float(Rf_[f'check_{o}']):.1e}" for o in ("bayes", "naive",
                                                                                                    "adf")))
    ch = [float(np.load(_fratio_file("trained", r))["check"]) for r in Fr["raw"]]
    mn = [float(np.load(_fratio_file("trained", r))["chi0_min"]) for r in Fr["raw"]]
    print(f"  flips/ratio Trained RNN: max check {max(ch):.1e}; smallest single-set chi(s, s) {min(mn):.4f}")


def checks():
    """independent checks of what the figure draws (prints max |differences|)"""
    P, F = pairs(), flips()
    assert F["trained_x"].shape == (114,)
    ag = list(P["agents"])
    e40 = P["edges40"]
    nb = len(e40) - 1
    idh, iskl = list(P["meas"]).index("dh"), list(P["meas"]).index("skl")
    out = {}
    with np.load(UNITS) as Z:
        env = Z["env"]
        for k in ag:
            ia = ag.index(k)
            pre, post = Z[f"pre_{k}"].ravel(), Z[f"post_{k}"].ravel()
            dskl = (Z[f"skl_post_{k}"] - Z[f"skl_pre_{k}"]).ravel()
            b = np.clip(np.searchsorted(e40, pre, side="right") - 1, -1, nb)
            b[pre >= e40[-1]] = nb
            inb = (b >= 0) & (b < nb)
            n = np.bincount(b[inb], minlength=nb)
            out[f"bin counts = var_n, {k}"] = float(np.abs(n - P["var_n"][ia]).max())
            for nm, v, im in (("Delta d_H", post - pre, idh), ("Delta SKL", dskl, iskl)):
                m_ = np.bincount(b[inb], v[inb], nb) / n
                out[f"per-bin mean {nm} = var_mean (full sample), {k}"] = float(np.nanmax(np.abs(
                    m_ - P["var_mean"][ia, im])))
            if k != "naive":
                med = np.array([np.median((post - pre)[inb & (b == i)]) for i in (0, 10, 20, 30, 39)])
                out[f"violin medians of Delta d_H (5 bins) = vio_med, {k}"] = float(np.abs(
                    med - P["vio_med"][ia, 1, [0, 10, 20, 30, 39]]).max())
            out[f"cond_n / ent_n summed over groups = var_n, {k}"] = float(max(
                np.abs(P["cond_n"][ia].sum(0) - P["var_n"][ia]).max(), np.abs(P["ent_n"][ia].sum(0) -
                                                                              P["var_n"][ia]).max()))
            out[f"avg_n + avg_out = pair-steps, {k}"] = float(P["avg_n"][ia].sum() + P["avg_out"][ia].sum() -
                                                              P["n_pairsteps"])
    U = unit_summary()
    for k in AG:
        i = list(P["stat_names"]).index(f"all|dh|{k}")
        b = P["stat_boot"][i]
        W = _W()
        with np.load(UNITS) as Z:
            s = np.bincount(env, (Z[f"post_{k}"] - Z[f"pre_{k}"]).sum(1), gap.N_ENV)
        c = np.bincount(env, None, gap.N_ENV) * 2.
        out[f"all-case mean Delta d_H from pairs_v2 = stat all|dh (every resample), {k}"] = float(
            np.abs((W @ s) / (W @ c) - b).max())
        out[f"unit_summary dh = stat all|dh (value), {k}"] = float(abs(U[k]["dh"][0] - b[0]))
    # the ratios (John 11-03): pairs_ratio recomputed from pairs_v2; the splits' groups reproduce the pairs module's
    # stored difference contrasts; flips/ratio: naive exactly 1, the class counts complete
    import td_tt_chaos_pairs as tp
    Rt = ratios()
    W = _W()
    with np.load(UNITS) as Z, np.load(tp.SHUF40_CACHE) as SH:
        env_u = np.repeat(Z["env"], tp.NV)
        g5 = np.repeat(SH["g5"].astype(np.int64), tp.NV)
        t_u = np.repeat(Z["t"], tp.NV)
        for k in AG:
            pre, post = Z[f"pre_{k}"].ravel(), Z[f"post_{k}"].ravel()
            b = tp.unit_bins(pre, e40)
            inb = (b >= 0) & (b < nb)
            lr = np.log(post / pre)
            n = np.bincount(b[inb], minlength=nb)
            out[f"ratio: bin counts = n_<agent>, {k}"] = float(np.abs(n - Rt[f"n_{k}"]).max())
            out[f"ratio: per-bin mean ln r = bin_dh (full sample), {k}"] = float(np.abs(
                np.bincount(b[inb], lr[inb], nb) / n - Rt[f"bin_dh_{k}"][0]).max())
            out[f"ratio: all-case mean ln r = all_dh (full sample), {k}"] = float(abs(lr[inb].mean() -
                                                                                     Rt[f"all_dh_{k}"][0]))
            ia = ag.index(k)
            med = np.array([np.median(lr[inb & (b == i)]) / np.log(10.) for i in (0, 10, 20, 30, 39)])
            out[f"ratio: violin medians (5 bins) = medians of log10 r, {k}"] = float(np.abs(
                med - Rt["vio_med"][ia, [0, 10, 20, 30, 39]]).max())
            tb = t_u * nb + b
            cg = SH[f"cond_{k}"].ravel().astype(np.int64)
            for nm, grp, K in (("cond3", cg, tp.OTH_K), ("ent5", g5, tp.ENT_K)):
                mt = tp.matched(post - pre, grp, tb, inb, env_u, W, K=K)
                out[f"ratio: the {nm} groups give the stored Delta d_H contrast {nm}|dh|step+bin (every resample), "
                    f"{k}"] = float(np.abs(mt["hilo"] - P["stat_boot"][list(P["stat_names"]).index(
                        f"{nm}|dh|step+bin|{k}")]).max())
    Fr = fratio_stats()
    out["flips/ratio: naive's log ratios (q, r, change, keep, index; lags), max |.|"] = float(max(
        np.abs(Fr["refs"]["naive"][nm][..., 1:] if nm in ("change", "keep", "index") else Fr["refs"]["naive"][nm]).max()
        for nm in ("q", "r", "change", "keep", "index")))
    n_cls = [float(np.load(_fratio_file("trained", r))["cls_n"].sum()) for r in Fr["raw"]]
    out["flips/ratio: class units per checkpoint - 1,000 probes x 5 flip steps x 31 sets"] = float(
        np.abs(np.array(n_cls) - 1000 * 5 * 31).max())
    out["flips/ratio: checkpoints = flips' 114"] = float(np.abs(Fr["x"] - F["trained_x"]).max())
    # the chaos sweep: this module's copy vs td_tt_chaos.py's own functions (the owner may be editing it)
    S = sweep()
    try:
        import td_tt_chaos as tc
        z = tc.sweep_load("trained")
        chi = z["chi"].astype(np.float64).mean(-1)
        d = 0.
        for j, L in enumerate(SW_LAGS):
            v, lo, hi = tc._env_boot(tc.chi_rise(chi, L), z["rows"])
            d = max(d, np.abs(v - S["rise"][:, j, 0]).max(), np.abs(lo - S["rise"][:, j, 1]).max(),
                    np.abs(hi - S["rise"][:, j, 2]).max())
        out["sweep rises = td_tt_chaos.sweep_load / chi_rise / _env_boot (114 x 5 lags, value and 5-95%)"] = float(d)
        out["sweep x = td_tt_chaos's"] = float(np.abs(z["x"] - S["x"]).max())
        B = dict(np.load(BSIDE))
        R = sweep_refs()
        out["sweep references = td_tt_chaos's (Bayes, naive, ADF; 5 lags)"] = float(max(
            np.abs(np.array([tc.chi_rise(B[f"chi_{nm}"].mean(-1).mean(-2), L).mean() for L in SW_LAGS]) - R[nm]).max()
            for nm in ("bayes", "naive", "adf")))
    except Exception as e:                                                                  # noqa: BLE001
        out["sweep vs td_tt_chaos"] = f"not run: {type(e).__name__}: {e}"
    # flips: checkpoints and the count code
    out["flips x = td_tt_gap.train_x(raw)"] = float(np.abs(F["trained_x"] - gap.train_x(F["trained_raw"])).max())
    out["flips count-class x = growth x"] = float(np.abs(F["trained_cc_x"] - F["trained_x"]).max())
    d = F["trained_count_changing"][:, 4] - F["trained_count_preserving"][:, 4] - F["trained_count_index"][:, 4]
    out["count index - (change - keep), lags 18-21: max |d| (the index is paired per (probe, flip step); the two class "
        "means average over all their units)"] = float(np.abs(d).max())
    out["sweep x = flips x (the same 114 checkpoints)"] = float(np.abs(S["x"] - F["trained_x"]).max())
    for k_, v in out.items():
        print(f"  {k_}: {v if isinstance(v, str) else f'{v:.3g}'}")
    return out


if __name__ == "__main__":
    what = sys.argv[1:] or ["numbers"]
    if "fratio-harvest" in what:
        i = what.index("fratio-harvest")
        fratio_harvest(int(what[i + 1]), int(what[i + 2]))
    if "fratio-refs" in what:
        fratio_refs()
    if "ratios" in what:
        ratio_build()
    if "figure" in what:
        figure(path=os.environ.get("FIG_OUT"))
    if "numbers" in what:
        numbers()
    if "checks" in what:
        checks()
