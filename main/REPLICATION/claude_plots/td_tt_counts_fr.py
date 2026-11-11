"""Counts decoded from the state, split by factorization regret, through training (John 11-02, with td_summary.png's
panel d attached: "I would also like a figure that shows this split by Factorization Regret.")

Panel d of td_summary.png is the ridge R^2 of the counts n_T decoded from u at step 30. The counts are the 5
observations summed over the 30 steps; Bayes's and naive's beliefs depend on the observations only through them.
Of the probes' 50 held-out environments (pairs of test variables, never a training goal nor seen as a pair),
td_analysis fits the decoder on 20, chooses the penalty (lambda = c tr(G) / D, c in 1e-4 .. 1) on 5 more, refits on
the 25 and tests on the other 25 (the decoder-test environments), re-choosing c at every checkpoint. Here the same
decoder (td_analysis's float32 solve) runs per checkpoint at every c, with per-episode predictions (CHECKS); the
control is one fixed penalty, c = 0.1 at every checkpoint of both networks. The 5,000 decoder-test episodes are split
into fifths of their factorization regret at T: FR_T = the mean over the two variables of KL(Bayes || naive) of the
marginal beliefs at step 30 (the paper's forward form, exact: main.utils.factorization_regret = probes.npz 'fr';
since 11-10). Until then, and in CHECKS and RESULTS below, FR_T was the mean of the symmetric KL, [KL(Bayes || naive)
+ KL(naive || Bayes)] / 2 (td_analysis.fr_sym, which clips the marginals to [1e-15, 1 - 1e-15] and so caps each
variable's FR near 34.54 nats; the largest FR_T was 28.05 clipped, 36.1 exact); FORWARD FR below gives the
re-measurement. Each fifth's part is on the total's scale (memory: split r additively): R^2_g = 1 - SSE_g / (n_g
SST / n), SST around the decoder-test mean; the five parts average (weights n_g = 1,000) to the total R^2, so a fifth
below the total is decoded worse than the average decoder-test episode, in the same units. The top fifth's contrast
D5 = R^2_5 - mean(R^2_1..4) = (mean MSE of fifths 1-4 - MSE_5) / (SST / n); the graded checks D4 = R^2_4 -
mean(R^2_1..3) and T2 = mean(R^2_4, R^2_5) - mean(R^2_1..3).
Intervals [5-95%]: the 1,000 shared environment resamples (td_tt_gap.boot_counts) restricted to the 25 decoder-test
environments (15-40 draws of them per resample); every statistic is recomputed per resample (SST around the
resample's own mean) with the decoder and its penalty fixed, so the intervals cover which decoder-test environments
were drawn, not the decoder's fit. The penalty also moves D5 (RESULTS 3): D5's range over the five c exceeds the
width of the chosen-c interval at 56 of 114 Trained-RNN checkpoints (53 with the float64 solve; 34 over c = .01-1,
the penalties ever chosen), e.g. .159 against .070 at 480, but not at 1,120 (.068 against .148), 1,920 (.048
against .089) or the end (.050 against .052); the ESN's range over c = .01-1 exceeds it at none. Significant = the
95th percentile below 0. Changes over training and differences between the networks or the penalties are paired on
the same resamples; an order is stated only with its paired difference.
-> main/REPLICATION/figure_bin/td_tt_counts_fr.png: a, b the parts through training (chosen c); c, d D5 at the chosen
c (band shaded) and at c = 0.1 (dashed, band between thin lines), the chosen c in a strip below. Caches
through_training/counts_fr/decode.npz (per-episode SSE at every c) and decode64.npz (the float64 check). The SSE do
not depend on FR: load() takes FR_T from the probes and ignores decode.npz's own fr_T field, which in the cache of
11-02 is the symmetric clipped one (a rebuild writes the forward one).

FORWARD FR (11-10; numbers(), CPU, ~25 s. The figure was not redrawn: it left figure_bin with the commit of 11-05)
  FR_T fifths 0.0032-0.30 / 0.30-0.67 / 0.67-1.4 / 1.4-4.0 / 4.0-43 nats (symmetric: 0.0032-0.42 / 0.42-1.0 / 1.0-2.4
  / 2.4-6.9 / 6.9-28). 68.7% of the decoder-test episodes are in the same fifth under both, 793 of the 1,000 top-fifth
  episodes are shared, rank correlation .953 (one variable's FR now reaches 85.1 nats, FR_T 42.5).
  The four results stand. Forward-FR values (the symmetric ones are in RESULTS):
  1. The dip is global: parts at the dip (1,120) .433 / .429 / .450 / .425 / .407 (total .429); initial .854 / .849 /
  .856 / .839 / .837; end .906 / .893 / .901 / .898 / .908. Every fifth falls by .41-.43 (paired 5% bounds .353-.373;
  +186 to +197 counts^2). Fifths cut within each environment .394-.453, on environment-centred log FR_T .400-.468. On
  environment-centred raw FR_T they are .306 / .458 / .520 / .454 / .407: with the uncapped FR an environment's mean
  is set by its largest episodes, so centring on it sorts the episodes by environment (57.8% in the same fifth as the
  global split) and this variant no longer isolates a within-environment effect; its D5 is as the others' (-.028
  [-.091, +.035] at the dip, -.067 at 480, -.091 at 1,920, +.009 at the end).
  2. Initial network: D5 -.012 [-.024, -.000], D4 -.013 [-.022, -.005], T2 -.014 [-.023, -.005]. ESN: D5 -.012 to
  -.019 up to 1,200 and -.034 to -.040 from 8,800, significant at 103 of 114 checkpoints; D4 at all 114; end minus
  initial -.024 [-.039, -.009] (c = 0.1: -.020 [-.041, -.002]).
  3. Trained RNN, D5: significant at 83 of 114 checkpoints (10-560, 1,360, 1,520, 1,680-23,200, 25,600); -.067 [-.105,
  -.030] at 480, -.027 [-.100, +.045] at the dip (unresolved), -.093 [-.140, -.047] at 1,920, +.009 [-.016, +.045] at
  the end. Paired: mean over 200-560 minus mean over 20-120 -.034 [-.056, -.014]; 1,280-8,000 minus 200-560 -.000
  [-.025, +.025]; 8,800-24,000 minus 1,280-8,000 +.038 [+.015, +.060]; end minus initial +.021 [-.009, +.057] (c = 0.1:
  +.006 [-.014, +.026]). Fixed c = 0.1: significant at 97 of 114. One difference: at the common penalty c = 0.1 the
  Trained RNN's D5 is above the ESN's at 38,400 and at the end (+.027 [+.001, +.052]); with the symmetric FR it was
  above at no checkpoint for c = 1e-3 to 1 (+.019 [-.000, +.037]). At c = .01 and c = 1 it is still unresolved at
  the end (+.047 [-.002, +.097], +.003 [-.025, +.029]).
  4. ESN: .854 / .849 / .856 / .839 / .837 at the start -> .797 / .793 / .798 / .772 / .754 at the end; the fifths
  fall by .057-.084; D5 -.036 [-.055, -.019] at the end.

CHECKS (numbers)
  The total R^2 at the chosen c equals td_analysis's stored dec_r2[step 30, u, counts] at all 114 checkpoints of both
  networks (max |d| 0); the chosen c is the first maximum of the validation R^2; the parts average to the total
  (max |d| 7e-16). The rebuild at every penalty (11-02) reproduced the first build's chosen-c per-episode SSE bit for
  bit, and its Trained-RNN SSE at every c equals the verifier's separate rerun bit for bit.
  Precision (build64: the same decoder solved in float64 on the same replayed u, every checkpoint and penalty; its
  float32 recomputation equals the cache bit for bit at every penalty, and its SSE equals the verifier's separate
  numpy float64 ridge at that check's 8 checkpoints per network to 3e-10 of the mean, 11-02): the float64 choice of c
  is the float32 one at all 114 checkpoints of both networks; at c >= .01 D5 moves by at most .0002 and the total R^2
  by at most .0006. At c <= 1e-3 the float32 solve is ill-conditioned: the Trained RNN's D5 moves by up to .0171 /
  .0024 and its total R^2 by up to .0658 / .0022 (c = 1e-4 / 1e-3), the ESN's by up to .0519 / .0057 and .2484 /
  .0187 (its R^2 there falls to -2.287 / -.154 in float64). No number quoted here moves by more than .001 (the c =
  .01 minimum .312 is .311 in float64); the five-c count above is 53 in float64 against 56; the Trained-RNN-minus-ESN
  runs at one common penalty and at the mixed penalties (RESULTS 3) are the same in float64.
  Fifths (of the symmetric FR, until 11-10): 1,000 decoder-test episodes each (5,000 distinct FR_T values, 25
  environments). Clipping: one variable's FR
  reaches 34.40 nats clipped (the cap: log((1 - 1e-15) / 1e-15) = 34.54), 58.1 exact; FR_T 28.05 clipped (2 episodes
  within .5 nats of it), 36.1 exact (top-fifth edge 7.08 vs 6.91 clipped); 98.6% of the episodes fall in the same
  fifth, 974 of the 1,000 top-fifth episodes are shared, rank correlation .9990.
  Resampling: centring SST on each resample's own mean (the first version used the full decoder-test mean) moves D5's
  interval ends by at most .001 (significant checkpoints 85 Trained RNN / 98 ESN either way); a 25-of-25 bootstrap of
  the decoder-test environments (seed 20261104) moves them by at most .005 / .002 (85 / 95 significant).
RESULTS (measured 11-02 with the symmetric clipped FR: FR_T fifths 0.0032-0.42 / 0.42-1.0 / 1.0-2.4 / 2.4-6.9 /
  6.9-28 nats; chosen c unless stated; the forward-FR re-measurement is above)
  1. Failed hypothesis: the dip is not concentrated where factorization fails; it is global. Trained RNN, total R^2
  .429 at its minimum (1,120 training episodes): fifths .440 / .423 / .445 / .441 / .395 (5-95% [.356, .507] /
  [.340, .495] / [.369, .505] / [.371, .497] / [.313, .468]); initial network .854 / .849 / .855 / .840 / .837; end
  .915 / .893 / .900 / .894 / .905. Every fifth falls by .40-.44 (paired 5% bounds .353-.382); in absolute units the
  dip adds +183 to +203 counts^2 of squared error to every fifth (initial MSE 67-74; the counts' variance SST/n = 458
  counts^2 over the 5 counts, per fifth 483 / 473 / 441 / 423 / 470, not ordered by FR). Not an environment effect:
  every fifth draws on all 25 decoder-test environments (effective number 21.1-23.8; 15-66 of each environment's 200
  episodes in the top fifth; 3.1% of FR_T's variance is between environments, 5.3% of log FR_T's, 17.1% of the
  counts'), and fifths cut within each environment / on environment-centred FR_T / on environment-centred log FR_T
  give parts at the dip of .405-.448 / .402-.454 / .407-.441, D5 at the dip -.030 / -.034 / -.027 (n.s.), at 480
  -.066 / -.063 / -.061 and at 1,920 -.087 / -.096 / -.082 (significant), at the end +.004 / +.005 / +.004.
  2. A small deficit of the top FR fifths exists without training, and it is graded. Initial network: D5 -.012
  [-.025, +.003] (n.s.), D4 -.013 [-.021, -.006], T2 -.014 [-.023, -.005] (per-fifth MSE 66.9 / 69.0 / 66.6 / 73.4 /
  74.4 counts^2). ESN: D5 -.011 to -.016 up to 1,200 training episodes and -.025 to -.035 from 8,800, significant at
  98 of 114 checkpoints; D4 significant at all 114 (-.029 to -.013), T2 at all 114 as well. The ESN's change is small
  and rests on paired tests: end minus initial -.023 [-.039, -.006] (c = 0.1: -.017 [-.034, -.001]), mean from 8,800
  minus mean up to 1,200 -.017 [-.030, -.005] (c = 0.1: -.014 [-.028, -.001]); pointwise below the initial value
  only at 20 and where the ESN's chosen c is 1 (24,800, 27,200-39,996), at c = 0.1 at 20, 120, 2,320-3,800,
  8,000-11,200 and 35,200-39,996. Trained RNN: D4 significant at 87 of 114 (10-560, 1,840-2,080, 2,240-2,320,
  2,600-28,800, 31,200-33,600), not around the dip (+.005 [-.044, +.056] at 1,120); per-fifth MSE 187.7 / 190.8 /
  184.6 / 204.0 / 239.9 counts^2 at 1,920. Fifths 4 and 5 are both decoded worse, so D5's reference (the mean of
  fifths 1-4) contains a worse fifth.
  3. Trained RNN, D5: significant at 85 of 114 checkpoints, 20-60, 100-560, 1,280-24,000 and 25,600 (not at 10, 80,
  640-1,200, 24,800 or from 26,400). At 20-120 -.024 to -.015, not different (paired) from its initial value or the
  ESN's (at c = 0.1 below both at 100 only); from 140 it is below its initial value at 140-180, 220-560 and
  1,360-8,000 (paired) and below the ESN's at 140, 180-560 and 1,360-5,000. It grows during the descent: -.036 to
  -.072 at 200-560 (-.066 [-.101, -.032] at 480); mean over 200-560 minus mean over 20-120 -.042 [-.062, -.023] (c =
  0.1: -.034 [-.053, -.016]), paired. It is not detectably larger in the recovery: -.034 to -.105 at 1,280-8,000
  (-.105 [-.149, -.060] at 1,920, -.068 [-.104, -.031] at 2,400); mean over 1,280-8,000 minus mean over 200-560 -.001
  [-.029, +.024] (c = 0.1: -.007 [-.034, +.017]) and 1,920 minus 480 -.039 [-.075, +.001] (n.s.; the chosen c is 0.1
  at both), paired. It then shrinks: -.027 to -.013 at 8,800-24,000 (mean over 8,800-24,000 minus mean over
  1,280-8,000 +.042 [+.020, +.063], paired; no longer different from its initial value or the ESN's, at c = 0.1 below
  its initial value through 9,600) and -.014 to +.005 at 24,800-39,996 (end +.004 [-.017, +.035], n.s.; change from
  initial +.016 [-.012, +.053], n.s.). It is above the ESN's (paired) only at 24,800 and 27,200-39,996, exactly where
  the ESN's chosen c is 1 and the Trained RNN's .01 (end +.039 [+.012, +.071]). At one common penalty it is above at
  no checkpoint for c = 1e-3 to 1 (c = 0.1: end +.019 [-.000, +.037]), and for c = 1e-4 only at 11,200, where the
  ESN's total R^2 at that penalty is negative. So its significance comes from the two chosen penalties: at one common
  penalty the difference is unresolved at 24,800-39,996 (end +.033 [-.003, +.074] at c = .01, -.005 [-.031, +.019] at
  c = 1), and either chosen penalty alone, with the other network at c = 0.1, puts the Trained RNN above the ESN at
  the end: Trained RNN at .01 minus ESN at 0.1 above 0 from 30,400 (end +.033 [+.005, +.067]), Trained RNN at 0.1
  minus ESN at 1 from 24,000 (end +.025 [+.007, +.042]).
  Around the dip it is unresolved, not absent: -.043 [-.116, +.033] at the bottom (1,120), an interval 2.8x / 3.5x as
  wide as the median at 20-560 (.052) / 1,840-8,000 (.042); -.004 / +.000 / -.007 / -.013 at 720-960, where the chosen
  penalty jumps to c = 1 (c = .01 at 10-260, .1 at 280-640, 1 at 720-960, .1 at 1,040-1,120, 1 at 1,200-1,760, .1 at
  1,840-8,000, .01 from 8,800). The penalty moves D5: at 480 -.180 / -.156 / -.109 / -.066 / -.020 for c = 1e-4 /
  1e-3 / .01 / .1 / 1 (the last n.s.; paired, c = .01 minus c = 1: -.089 [-.143, -.041]), at 1,120 -.106 [-.186,
  -.024] / -.102 [-.182, -.020] / -.076 / -.043 / -.038 (-.038 [-.071, -.005]), at 1,920 -.131 / -.129 / -.120 /
  -.105 / -.083 (each significant; paired -.037 [-.084, +.003], n.s.); at the end the dependence reverses: +.010 /
  +.009 / +.004 / -.010 / -.040 [-.066, -.014] (paired +.044 [+.012, +.078]). The total R^2 dips at every c (minimum
  .239 / .249 / .312 / .429 / .446 at 1,280 / 1,280 / 1,280 / 1,120 / 1,120).
  Fixed c = 0.1 (test R^2 at most .038 below the chosen c's, at 40, median .004; up to .030 above it, at 1,760): D5
  significant at 99 of 114 checkpoints, 20-560, 720 (-.051 [-.090, -.008]), 1,360-34,400 and 37,600; at 800-1,200
  -.026 to -.043 (n.s.), the size it has at 200 (-.030 [-.053, -.009]); -.018 to -.010 at 24,800-39,996, significant
  at 14 of these 20 checkpoints (through 34,400 and at 37,600; end -.010 [-.024, +.004], n.s.).
  So: the top FR fifths are decoded slightly worse at initialization (T2 -.014 [-.023, -.005]; D5 alone n.s.) and in
  the ESN throughout, where the deficit grows a little (paired, RESULTS 2); graded: fifth 4 is worse too. In the
  Trained RNN it grows during the descent (to about .04-.07; paired), is unresolved at the dip, is not detectably
  larger in the recovery (about .03-.11; paired), shrinks to .01-.03 at 8,800-24,000 and returns to its initial size
  by the end (paired change from initial +.016 [-.012, +.053] at the chosen c, +.001 [-.016, +.020] at c = 0.1, both
  n.s.; at c = 0.1 D5 is still -.010 to -.018 at 24,800-39,996, significant through 34,400).
  4. ESN: .854 / .849 / .855 / .840 / .837 at the start -> .795 / .793 / .800 / .770 / .755 at the end; total R^2 .847
  -> .783 (paired fall +.064 [+.050, +.078]), its minimum at the end (no dip). The fifths fall by .055-.082 (paired 5%
  bounds .037-.063; +25 to +38 counts^2), fifths 4 and 5 by more than fifths 1-3 on average (paired change of T2
  from the start to the end -.019 [-.034, -.006]; c = 0.1 -.016 [-.031, -.002]). D5 -.035 [-.051, -.019] at the end
  (chosen c = 1 at 24,800 and from 27,200), -.028 [-.047, -.011] at c = 0.1 (which never lowers the ESN's test R^2;
  up to .009 higher).
  Corrected 11-02 (verifier): the first version's '95% bound below 0 at 200-560 and 1,280-25,600' was wrong (20-60,
  100-560, 1,280-24,000 and 25,600, not 24,800); its 'present before and after the dip, absent at it' rested on the
  penalty jumping to c = 1 near the dip; its size '.04-.10' left out the deficit at initialization and in the ESN and
  fifth 4; its panel c (fifths at the start, dip and end without intervals: fifth 5 lowest at the dip, n.s.) is
  replaced by the contrast panels c, d.
  Corrected 11-02 (second check): 'near the initial network and the ESN at 20-180' held only at 20-120 (at 140-180
  below its initial value, and below the ESN's at 140 and 180, paired); the end's 'above the ESN's by +.039' is
  significant only through the penalties (only where the ESN's chosen c is 1; n.s. at c = 0.1); 'the penalty moves D5
  by more than the intervals do' holds at about half of the checkpoints (56 of 114), not at 1,120, 1,920 or the end;
  'FR_T saturates at 28 nats' was wrong (28.05 is the sample's largest FR_T; the clip caps one variable's FR near
  34.54); 'not significant at the end' is now 'returns to its initial size' (at c = 0.1 still significant through
  34,400); the ESN's growth and 'most in fifths 4 and 5' now rest on paired tests; the float32 solve's precision at
  c <= 1e-3 is now checked (build64); the two bands of c, d are told apart in the figure.
  Corrected 11-02 (third check): that significance was put down to the ESN's chosen c alone, but the Trained RNN's
  chosen c = .01 alone gives it too (RESULTS 3); 'grows during the descent and the recovery' (and the suptitle's
  'grows while the Trained RNN descends and recovers') read as further growth in the recovery, which no paired test
  shows.
  Corrected 11-02 (fourth check): that correction's 'not from a difference between the networks at one penalty' read
  an unresolved difference as none: at one common penalty it is unresolved at 24,800-39,996 (RESULTS 3).

Run from the repository root (build: GPU, ~9 min on a crowded GPU 0, 1.2 GB; build64: GPU, ~3.5 min, 1.2 GB;
numbers: CPU, ~20 s; figure: CPU, ~5 s), with the command build, build64, numbers or figure:
    CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
        /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_counts_fr.py numbers"""
import os
import sys
import time
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_intervene as ti                                                                  # noqa: E402
from main.utils import factorization_regret                                                   # noqa: E402

OUT = os.path.join(gap.TT, "counts_fr")
RIDGE_C = (1e-4, 1e-3, 1e-2, 1e-1, 1.)                     # td_analysis.RIDGE_C: penalty = c * tr(G) / D
C_FIX = 1e-1                                               # the fixed penalty of the control (every checkpoint)
DASH, EDGE_LW = (0, (4, 1.6)), .8                          # the fixed-c curve's dashes; its band's edge lines
RUNS = (("trained", "Trained RNN", "#1f77b4"), ("echo", "ESN", "#ff7f0e"))
DARK = {"trained": "#08468c", "echo": "#a63603"}           # the fixed-penalty curves (dark, not grey)
NG = 5
BOOT25_SEED = 20261104                                     # the 25-of-25 bootstrap of the robustness check
SHOW_X = (10, 20, 60, 100, 120, 140, 160, 180, 200, 480, 560, 640, 720, 800, 880, 960, 1040, 1120, 1200, 1920, 2400,
          8000, 24000, 24800, 25600, 34400, 37600)         # checkpoints quoted in the docstring (plus the last one)
STRETCHES = ((20, 120), (140, 180), (200, 560), (640, 1200), (1280, 8000), (8800, 24000), (24800, 39996))   # quoted
EARLY, LATE = (20, 120), (24800, 39996)                    # the stretch near initialization; the last 20 checkpoints
DESCENT, RECOVERY = (200, 560), (1280, 8000)               # the Trained RNN's descent and recovery (RESULTS 3)
PAIRS = tuple((c, c) for c in RIDGE_C) + ((1e-2, 1e-1), (1e-1, 1.))  # (Trained RNN's c, ESN's c): RESULTS 3's contrasts


# ═══════════════════════════════════════════════════════════════════
# Build: the decoder of td_summary panel d at every penalty, per test episode
# ═══════════════════════════════════════════════════════════════════

def _probe_side():
    P = ti.probes()
    Z = np.load(os.path.join(ti.AN, "probes.npz"))
    env = P["env"]
    sets = {k: np.isin(env, Z[f"dec_{k}"]) for k in ("train", "val", "test")}
    counts = P["obs"].sum(1).astype(np.float32)                                          # n_T (n, 5), inclusive
    fr_T = Z["fr"][:, -1].astype(np.float64).mean(-1)          # KL(Bayes || naive) at T, mean of the two variables
    return P, sets, counts, fr_T


def _ridge(X, Y, fit, val, test, dev, dt=torch.float32):
    """td_analysis.ridge_r2 for one target, at every penalty: (test predictions (len(RIDGE_C), n_test, k), validation
    R^2 per penalty, index of the chosen penalty (the first maximum, as td_analysis), test R^2 at the chosen penalty).
    dt = float32 is td_analysis's solve (bit for bit); float64 the precision check (build64)"""
    Y = Y.to(dt)

    def solve(rows):
        Xr = X[torch.as_tensor(rows, device=dev)].double()
        mx, sx = Xr.mean(0), Xr.std(0).clamp_min(1e-6)
        Xs = ((Xr - mx) / sx).to(dt)
        ev, V = torch.linalg.eigh(Xs.T @ Xs)
        return mx, sx, ev.clamp_min(0), V, Xs

    def predict(sol, rows, rows_out, c):
        mx, sx, ev, V, Xs = sol
        Yr = Y[torch.as_tensor(rows, device=dev)]
        my = Yr.mean(0)
        lam = c * ev.sum() / len(ev)
        Bm = V @ (((V.T @ (Xs.T @ (Yr - my))) / (ev + lam)[:, None]))
        Xo = ((X[torch.as_tensor(rows_out, device=dev)].double() - mx) / sx).to(dt)
        return Xo @ Bm + my

    def r2(Yhat, Yo):
        return 1 - ((Yo - Yhat) ** 2).sum().item() / ((Yo - Yo.mean(0)) ** 2).sum().item()

    sol_fit, sol_all = solve(fit), solve(fit | val)
    Yv, Yt = Y[torch.as_tensor(val, device=dev)], Y[torch.as_tensor(test, device=dev)]
    vr = np.array([r2(predict(sol_fit, fit, val, c), Yv) for c in RIDGE_C])
    best = int(np.argmax(vr))
    preds = [predict(sol_all, fit | val, test, c) for c in RIDGE_C]
    return preds, vr, best, r2(preds[best], Yt)


def _states(w, n, dev):
    """u at step 30 of every probe (n, H) for the weights w (td_tt_intervene's clean fp32 replay)"""
    U = torch.zeros(n, w["LSTM.weight_hh_l0"].shape[1], device=dev)

    def rec(s, rows, st):
        if s == ti.T:
            U[torch.as_tensor(rows, device=dev)] = st["u"]
    ti.forward(w, device=dev, record=rec)
    return U


@torch.no_grad()
def build(dev=None):
    """per checkpoint of both networks: the decoder of td_summary panel d at every penalty -> per-test-episode squared
    errors summed over the 5 counts (sse_c (K, 5 penalties, 5,000)), validation R^2 per penalty, the chosen penalty
    and its test R^2 (= the stored dec_r2[step 30, u, counts], CHECK)"""
    dev = ti._device(dev)
    os.makedirs(OUT, exist_ok=True)
    P, sets, counts, fr_T = _probe_side()
    test = sets["test"]
    Y = torch.as_tensor(counts, device=dev)
    Yt = counts[test].astype(np.float64)
    out = {"fr_T": fr_T[test], "env_test": P["env"][test], "Y_test": Yt}
    for run, _, _ in RUNS:
        eps = ti.checkpoints(run)
        K, n = len(eps), int(test.sum())
        sse_c = np.zeros((K, len(RIDGE_C), n))
        r2s, cs, stored, vr = np.zeros(K), np.zeros(K), np.zeros(K), np.zeros((K, len(RIDGE_C)))
        t0 = time.time()
        for k, raw in enumerate(eps):
            U = _states(ti.load(run, raw), len(P["env"]), dev)
            preds, vr[k], best, r2s[k] = _ridge(U, Y, sets["train"], sets["val"], test, dev)
            cs[k] = RIDGE_C[best]
            for i, pred in enumerate(preds):
                sse_c[k, i] = ((pred.double().cpu().numpy() - Yt) ** 2).sum(-1)
            stored[k] = np.load(os.path.join(ti.AN, run, f"ck_{raw:05d}.npz"))["dec_r2"][9, 0, 0]
        print(f"{run}: {K} checkpoints, {time.time() - t0:.0f} s; R^2 vs stored max |d| "
              f"{np.abs(r2s - stored).max():.2e}", flush=True)
        ci = np.array([RIDGE_C.index(c) for c in cs])
        out.update({f"{run}_x": np.array([gap.train_x(e) for e in eps]), f"{run}_sse": sse_c[np.arange(K), ci],
                    f"{run}_sse_c": sse_c, f"{run}_val_r2": vr, f"{run}_r2": r2s, f"{run}_c": cs,
                    f"{run}_stored": stored})
    np.savez(os.path.join(OUT, "decode.npz"), ridge_c=np.array(RIDGE_C), **out)
    if dev.type == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated(dev) / 2 ** 30:.2f} GB")


@torch.no_grad()
def build64(dev=None):
    """the precision check of the float32 solve: per checkpoint of both networks the same decoder on the same replayed
    u, solved in float64 -> decode64.npz: per-test-episode SSE at every penalty (sse_c), validation R^2 (val_r2); and
    the float32 SSE recomputed from this replay vs the cache (d32, max |d| per checkpoint over the penalties)"""
    dev = ti._device(dev)
    P, sets, counts, _ = _probe_side()
    test = sets["test"]
    Y = torch.as_tensor(counts, device=dev)
    Yt = counts[test].astype(np.float64)
    D = load()
    out = {}
    for run, _, _ in RUNS:
        eps = ti.checkpoints(run)
        K, n = len(eps), int(test.sum())
        sse_c, vr, d32 = np.zeros((K, len(RIDGE_C), n)), np.zeros((K, len(RIDGE_C))), np.zeros(K)
        t0 = time.time()
        for k, raw in enumerate(eps):
            U = _states(ti.load(run, raw), len(P["env"]), dev)
            p32 = _ridge(U, Y, sets["train"], sets["val"], test, dev)[0]
            d32[k] = max(np.abs(((p.double().cpu().numpy() - Yt) ** 2).sum(-1) - D[f"{run}_sse_c"][k, i]).max()
                         for i, p in enumerate(p32))
            preds, vr[k], _, _ = _ridge(U, Y, sets["train"], sets["val"], test, dev, torch.float64)
            for i, pred in enumerate(preds):
                sse_c[k, i] = ((pred.cpu().numpy() - Yt) ** 2).sum(-1)
        print(f"{run}: {K} checkpoints, {time.time() - t0:.0f} s; float32 recomputed vs cache max |d| {d32.max():.2e}",
              flush=True)
        out.update({f"{run}_x": np.array([gap.train_x(e) for e in eps]), f"{run}_sse_c": sse_c, f"{run}_val_r2": vr,
                    f"{run}_d32": d32})
    np.savez(os.path.join(OUT, "decode64.npz"), ridge_c=np.array(RIDGE_C), **out)
    if dev.type == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated(dev) / 2 ** 30:.2f} GB")


# ═══════════════════════════════════════════════════════════════════
# The split: parts per FR fifth on the total's scale, resampled over the decoder-test environments
# ═══════════════════════════════════════════════════════════════════

def load():
    """decode.npz, with FR_T taken from the probes (forward, exact), not from the cache: a cache built before 11-10
    stores the mean of the two variables' symmetric clipped KL in its own 'fr_T' (kept as 'fr_T_cache'); the
    per-episode SSE do not depend on FR, so the cache needs no rebuild"""
    with np.load(os.path.join(OUT, "decode.npz")) as z:
        D = {k: z[k] for k in z.files}
    with np.load(os.path.join(ti.AN, "probes.npz")) as Z:
        test = np.isin(Z["env"], Z["dec_test"])
        assert np.array_equal(Z["env"][test], D["env_test"]), "decode.npz: other decoder-test episodes than the probes'"
        D["fr_T_cache"], D["fr_T"] = D["fr_T"], Z["fr"][test, -1].astype(np.float64).mean(-1)
    return D


def fifths(v):
    """labels (n,) of the quantile fifths of v and the edges (NG + 1,)"""
    edges = np.quantile(v, np.linspace(0, 1, NG + 1))
    return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, NG - 1), edges


def boot_weights(env):
    """(N_BOOT, n): how often each decoder-test episode's environment is drawn in each of the shared resamples
    (td_tt_gap.boot_counts, 50 draws over the 50 probe environments; restricted to the 25 decoder-test environments)"""
    return gap.boot_counts()[:, env]


def split(sse, Y, g, W=None, centre="resample"):
    """R^2 per group on the total's scale, R^2_g = 1 - SSE_g / (n_g SST / n), and the total R^2 = sum_g n_g R^2_g / n.
    sse (K, n) per-episode squared errors summed over the 5 counts, Y (n, 5) the counts, g (n,) group labels 0..NG-1;
    W (B, n) episode weights of B resamples (None: the full sample). SST around each resample's own weighted mean
    (centre='test': around the full decoder-test mean, the first version's choice) -> parts (K, NG), total (K,), or
    (B, K, NG), (B, K)"""
    full = W is None
    W = np.ones((1, len(Y))) if full else np.asarray(W, np.float64)
    sw = W.sum(1)
    Yc = Y - Y.mean(0)
    mu = (W @ Yc) / sw[:, None] if centre == "resample" else np.zeros((len(W), Y.shape[1]))
    var = (W @ (Yc ** 2).sum(1)) / sw - (mu ** 2).sum(1)                     # (B,) mean SST around mu
    tot = 1 - (W @ sse.T) / sw[:, None] / var[:, None]
    pr = np.stack([1 - ((W * (g == j)) @ sse.T) / (W * (g == j)).sum(1)[:, None] / var[:, None]
                   for j in range(NG)], -1)
    return (pr[0], tot[0]) if full else (pr, tot)


def d5(pr):
    """the top fifth's contrast: R^2_5 - mean(R^2_1..4) = (mean MSE of fifths 1-4 - MSE_5) / (SST / n)"""
    return pr[..., 4] - pr[..., :4].mean(-1)


def d4(pr):
    """the graded check: R^2_4 - mean(R^2_1..3)"""
    return pr[..., 3] - pr[..., :3].mean(-1)


def t2(pr):
    """the top two fifths against the bottom three: mean(R^2_4, R^2_5) - mean(R^2_1..3)"""
    return pr[..., 3:].mean(-1) - pr[..., :3].mean(-1)


def band(b):
    """5-95% over the resamples (axis 0)"""
    return np.percentile(b, [5, 95], 0)


def stats(D=None):
    """per network, at the chosen penalty ('ch') and at c = C_FIX ('fx'): parts pr (K, NG), total tot (K,), and their
    resampled versions bpr (N_BOOT, K, NG), btot (N_BOOT, K); plus the fifths, edges and resample weights"""
    D = load() if D is None else D
    g, edges = fifths(D["fr_T"])
    W = boot_weights(D["env_test"])
    S = {"D": D, "g": g, "edges": edges, "W": W}
    for run, _, _ in RUNS:
        r = {"x": D[f"{run}_x"], "c": D[f"{run}_c"]}
        for tag, sse in (("ch", D[f"{run}_sse"]), ("fx", D[f"{run}_sse_c"][:, RIDGE_C.index(C_FIX)])):
            pr, tot = split(sse, D["Y_test"], g)
            bpr, btot = split(sse, D["Y_test"], g, W)
            r[tag] = dict(sse=sse, pr=pr, tot=tot, bpr=bpr, btot=btot)
        S[run] = r
    return S


def _ex(v):
    return f"{v:,.0f}"


def _runs(x, mask):
    """the checkpoints where mask holds, as runs of consecutive checkpoints (training episodes)"""
    out, j, K = [], 0, len(x)
    while j < K:
        if mask[j]:
            i = j
            while j + 1 < K and mask[j + 1]:
                j += 1
            out.append(_ex(x[i]) if i == j else f"{_ex(x[i])}-{_ex(x[j])}")
        j += 1
    return ", ".join(out) if out else "none"


def _seg(x, c):
    """the chosen penalty's segments: '0.01 at 10-260; 0.1 at 280-640; ...'"""
    out, i = [], 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and c[j + 1] == c[i]:
            j += 1
        out.append(f"{c[i]:g} at {_ex(x[i])}" + (f"-{_ex(x[j])}" if j > i else ""))
        i = j + 1
    return "; ".join(out)


def _ci(v, b):
    lo, hi = band(b)
    return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _k(x, v):
    return int(np.argmin(np.abs(x - v)))


def edge_label(v):
    """FR edges with two significant digits, the same way everywhere (legend, docstring)"""
    s = np.format_float_positional(v, precision=2, unique=False, fractional=False, trim="k")
    return s.rstrip(".") if s.endswith(".") else s


# ═══════════════════════════════════════════════════════════════════
# Numbers: every number of the docstring
# ═══════════════════════════════════════════════════════════════════

def _mean_in(v, x, lo, hi):
    """the mean over the checkpoints lo <= x <= hi (last axis)"""
    return v[..., (x >= lo) & (x <= hi)].mean(-1)


def _net_pairs(src, Y, g, W, x):
    """RESULTS 3's Trained RNN minus ESN, D5 paired on the shared resamples, at one common penalty and at the two mixed
    ones (PAIRS), from the per-episode SSE at every c in src (decode.npz, or decode64.npz for the float64 check)"""
    d = {}
    for run, _, _ in RUNS:
        for i, c in enumerate(RIDGE_C):
            pr, tot = split(src[f"{run}_sse_c"][:, i], Y, g)
            d[run, c] = d5(pr), d5(split(src[f"{run}_sse_c"][:, i], Y, g, W)[0]), tot
    for ct, ce in PAIRS:
        (vt, bt, tt), (ve, be, te) = d["trained", ct], d["echo", ce]
        lo, hi = band(bt - be)
        neg = "".join(f"; {nm}'s total R^2 at its c below 0 at {_runs(x, t < 0)}"
                      for nm, t in (("Trained RNN", tt), ("ESN", te)) if (t < 0).any())
        print(f"  Trained RNN at c = {ct:g} minus ESN at c = {ce:g}: 5% bound above 0 at {_runs(x, lo > 0)}; 95% "
              f"bound below 0 at {_runs(x, hi < 0)}; end {_ci(vt[-1] - ve[-1], (bt - be)[:, -1])}{neg}")


def numbers():
    S = stats()
    D, g, edges, W = S["D"], S["g"], S["edges"], S["W"]
    Y, env, fr = D["Y_test"], D["env_test"], D["fr_T"]
    n = len(Y)
    print("=== CHECKS")
    Z = np.load(os.path.join(ti.AN, "probes.npz"))
    assert np.array_equal(Z["env"], np.arange(len(Z["env"])) // gap.PER_ENV), "probes are environment-major"
    test = np.isin(Z["env"], Z["dec_test"])
    assert np.array_equal(Z["env"][test], env) and np.array_equal(D["trained_x"], D["echo_x"])
    for run, name, _ in RUNS:
        r = S[run]
        dd = np.abs(D[f"{run}_r2"] - D[f"{run}_stored"]).max()
        ci_ = np.array([RIDGE_C.index(c) for c in D[f"{run}_c"]])
        print(f"{name}: total R^2 (chosen c) vs td_analysis's stored dec_r2[step 30, u, counts] max |d| {dd:.1e} over "
              f"{len(r['x'])} checkpoints; chosen c = first argmax of the validation R^2: "
              f"{np.array_equal(ci_, D[f'{run}_val_r2'].argmax(1))}; parts weighted by n_g vs total max |d| "
              f"{np.abs((r['ch']['pr'] * np.bincount(g)).sum(1) / n - r['ch']['tot']).max():.1e}; "
              f"split total vs cached R^2 max |d| {np.abs(r['ch']['tot'] - D[f'{run}_r2']).max():.1e}")
    print(f"initial networks identical (same per-episode SSE): {np.array_equal(D['trained_sse'][0], D['echo_sse'][0])}")
    print(f"decoder-test episodes {n} in {len(np.unique(env))} environments; fifth sizes {np.bincount(g).tolist()}; "
          f"{len(np.unique(fr))} distinct FR_T values; FR_T edges (nats) " + " / ".join(f"{e:.4g}" for e in edges)
          + "; legend labels " + " / ".join(edge_label(e) for e in edges))
    fr_var = Z["fr"][test, -1].astype(np.float64)                                        # (n, 2), forward, exact
    assert np.array_equal(fr_var.mean(-1), fr)
    fr_chk = factorization_regret(Z["log_jb"][test, -1], Z["log_nb"][test, -1], logs=True)   # from the float32 logs
    fs_var = Z["fr_sym"][test, -1].astype(np.float64)                                    # the superseded symmetric one
    fs = fs_var.mean(-1)
    g_s, e_s = fifths(fs)
    rk = np.corrcoef(np.argsort(np.argsort(fs)), np.argsort(np.argsort(fr)))[0, 1]
    print(f"FR = KL(Bayes || naive), forward, exact (probes 'fr'; from the stored float32 log beliefs: max |d| "
          f"{np.abs(fr_chk - fr_var).max():.1e}); the cache's own fr_T equals it: "
          f"{bool(np.allclose(D['fr_T_cache'], fr))}, equals the symmetric clipped one: "
          f"{bool(np.allclose(D['fr_T_cache'], fs))}")
    print(f"against the superseded symmetric clipped FR (probes 'fr_sym', used until 11-10): per-variable FR max "
          f"{fr_var.max():.2f} forward, {fs_var.max():.2f} symmetric (capped near {np.log((1 - 1e-15) / 1e-15):.2f}); "
          f"FR_T max {fr.max():.2f} vs {fs.max():.2f}; mean {fr.mean():.2f} vs {fs.mean():.2f}; edges forward "
          + " / ".join(f"{e:.4g}" for e in edges) + ", symmetric " + " / ".join(f"{e:.4g}" for e in e_s)
          + f"; same fifth for {(g_s == g).mean():.1%} of the episodes, {((g_s == 4) & (g == 4)).sum()} of the 1,000 "
          f"top-fifth episodes shared; rank correlation {rk:.4f}")
    m_t = W.sum(1) / gap.PER_ENV
    print(f"resamples restricted to the 25 decoder-test environments: {m_t.min():.0f}-{m_t.max():.0f} draws (mean "
          f"{m_t.mean():.2f}, sd {m_t.std():.2f})")

    print("\n=== PER CHECKPOINT (chosen c; fixed c = 0.1): total R^2, D5 = R^2_5 - mean(R^2_1..4) [5-95%], "
          "D4 = R^2_4 - mean(R^2_1..3) [5-95%]")
    for run, name, _ in RUNS:
        r = S[run]
        x, ch, fx = r["x"], r["ch"], r["fx"]
        print(f"--- {name}")
        for k in range(len(x)):
            print(f"  {x[k]:>7,.0f}  c {r['c'][k]:<5g} R^2 {ch['tot'][k]:.3f}  "
                  f"D5 {_ci(d5(ch['pr'])[k], d5(ch['bpr'])[:, k])}"
                  f"  | c=0.1: R^2 {fx['tot'][k]:.3f}  D5 {_ci(d5(fx['pr'])[k], d5(fx['bpr'])[:, k])}"
                  f"  | D4 {_ci(d4(ch['pr'])[k], d4(ch['bpr'])[:, k])}")

    print("\n=== THE DIP IS GLOBAL (chosen c): parts per FR fifth, 1..5, [5-95%] of each")
    for run, name, _ in RUNS:
        r = S[run]
        x, ch = r["x"], r["ch"]
        km = int(np.argmin(ch["tot"]))
        lo, hi = band(ch["bpr"])
        word = "dip's bottom" if run == "trained" else "minimum"
        for lab, k in (("initial network", 0), (word, km), ("end", len(x) - 1)):
            print(f"{name} {lab} ({_ex(x[k])}): total {ch['tot'][k]:.3f}; parts " + " / ".join(
                f"{v:.3f}" for v in ch["pr"][k]) + "; bands " + " / ".join(
                f"[{a:.3f}, {b:.3f}]" for a, b in zip(lo[k], hi[k])))
        fall, bfall = ch["pr"][0] - ch["pr"][km], ch["bpr"][:, 0] - ch["bpr"][:, km]
        ftot = _ci(ch["tot"][0] - ch["tot"][km], ch["btot"][:, 0] - ch["btot"][:, km])
        print(f"  {name}: fall from the initial network to the {word} (paired [5-95%]): " + " / ".join(
            _ci(fall[j], bfall[:, j]) for j in range(NG)) + f"; total {ftot}")

    print("\n=== D5 SIGNIFICANCE (95th percentile below 0) and sizes")
    for run, name, _ in RUNS:
        r = S[run]
        x = r["x"]
        for tag, lab in (("ch", "chosen c"), ("fx", "c = 0.1")):
            v, b = d5(r[tag]["pr"]), d5(r[tag]["bpr"])
            lo, hi = band(b)
            sig = hi < 0
            print(f"{name}, {lab}: 95% bound below 0 at {sig.sum()} of {len(x)} checkpoints: {_runs(x, sig)}; "
                  f"D5 range over checkpoints {v.min():+.3f} to {v.max():+.3f}; over the significant ones "
                  f"{v[sig].min():+.3f} to {v[sig].max():+.3f}; two-sided (97.5th below 0): "
                  f"{_runs(x, np.percentile(b, 97.5, 0) < 0)}")
            print("  at " + "; ".join(f"{_ex(x[_k(x, s)])}: {_ci(v[_k(x, s)], b[:, _k(x, s)])}" for s in SHOW_X)
                  + f"; end ({_ex(x[-1])}): {_ci(v[-1], b[:, -1])}")
            inside = [(x0, x1, (x >= x0) & (x <= x1)) for x0, x1 in STRETCHES]
            print("  D5 range per stretch: " + "; ".join(
                f"{_ex(x0)}-{_ex(x1)}: {v[m].min():+.3f} to {v[m].max():+.3f} ({sig[m].sum()}/{m.sum()} significant)"
                for x0, x1, m in inside))
            wd = hi - lo
            km = int(np.argmin(r["ch"]["tot"]))
            pre, post = (x >= 20) & (x <= 560), (x >= 1840) & (x <= 8000)
            print(f"  interval width (95th - 5th) at the total's minimum ({_ex(x[km])}) {wd[km]:.3f}; max over "
                  f"640-1,760 {wd[(x >= 640) & (x <= 1760)].max():.3f}; median at 20-560 {np.median(wd[pre]):.3f}, at "
                  f"1,840-8,000 {np.median(wd[post]):.3f}; ratio of the minimum's width to these "
                  f"{wd[km] / np.median(wd[pre]):.1f}x / {wd[km] / np.median(wd[post]):.1f}x")
        if run == "trained":
            x6 = (x >= 640) & (x <= 1200)
            print("  640-1,200, chosen c / c = 0.1: " + "; ".join(
                f"{_ex(x[k])} (c {r['c'][k]:g}) {d5(r['ch']['pr'])[k]:+.3f} / {d5(r['fx']['pr'])[k]:+.3f}"
                for k in np.flatnonzero(x6)))

    print("\n=== THE PENALTY: chosen c per checkpoint; D5 and total R^2 at each fixed c (float32, as td_analysis; "
          "c <= 1e-3: see PRECISION)")
    for run, name, _ in RUNS:
        x = S[run]["x"]
        print(f"{name}: chosen c {_seg(x, S[run]['c'])}")
        res = {}
        for i, c in enumerate(RIDGE_C):
            sse = D[f"{run}_sse_c"][:, i]
            pr, tot = split(sse, Y, g)
            bpr, _ = split(sse, Y, g, W)
            res[c] = (pr, tot, bpr)
            v, b = d5(pr), d5(bpr)
            hi = band(b)[1]
            km = int(np.argmin(tot))
            print(f"  c={c:g}: total R^2 minimum {tot[km]:.3f} at {_ex(x[km])}, end {tot[-1]:.3f}; D5 95% bound "
                  f"below 0 at {_runs(x, hi < 0)}")
        loss = S[run]["ch"]["tot"] - S[run]["fx"]["tot"]
        print(f"  fixed c = 0.1 vs the chosen penalty: total R^2 (chosen - fixed) at most {loss.max():+.4f} (at "
              f"{_ex(x[np.argmax(loss)])}), median {np.median(loss):+.4f}, at least {loss.min():+.4f} (at "
              f"{_ex(x[np.argmin(loss)])})")
        for s in (200, 480, 1120, 1920, 2400, x[-1]):
            k = _k(x, s)
            print(f"  D5 at {_ex(x[k])} (chosen c {S[run]['c'][k]:g}) for c = " + " / ".join(f"{c:g}" for c in RIDGE_C)
                  + ": " + " / ".join(_ci(d5(res[c][0])[k], d5(res[c][2])[:, k]) for c in RIDGE_C))
        d5c = np.stack([d5(res[c][0]) for c in RIDGE_C])                                 # (5, K)
        b5 = {c: d5(res[c][2]) for c in (RIDGE_C[0], 1e-2, 1.)}
        for s in (200, 480, 1120, 1920, 2400, x[-1]):
            k = _k(x, s)
            print(f"  paired across penalties at {_ex(x[k])}: D5(c = 1e-4) - D5(c = 1) "
                  f"{_ci(d5c[0, k] - d5c[4, k], b5[RIDGE_C[0]][:, k] - b5[1.][:, k])}; D5(c = .01) - D5(c = 1) "
                  f"{_ci(d5c[2, k] - d5c[4, k], b5[1e-2][:, k] - b5[1.][:, k])}")
        lo, hi = band(d5(S[run]["ch"]["bpr"]))
        wd = hi - lo
        r5, r3 = np.ptp(d5c, 0), np.ptp(d5c[2:], 0)
        print(f"  D5's range over the penalties vs the width of the chosen-c 5-95% interval: over the five c larger at "
              f"{(r5 > wd).sum()} of {len(x)} checkpoints, over c = .01-1 (the penalties ever chosen) at "
              f"{(r3 > wd).sum()}; at " + "; ".join(
                  f"{_ex(x[k])}: {r5[k]:.3f} (c = .01-1 {r3[k]:.3f}) vs {wd[k]:.3f}"
                  for k in (_k(x, 480), _k(x, 1120), _k(x, 1920), len(x) - 1)))

    print("\n=== GROWTH AND RETURN: paired changes of D5 on the same resamples (chosen c; c = 0.1)")
    x = S["trained"]["x"]
    early = (x >= EARLY[0]) & (x <= EARLY[1])
    late = (x >= LATE[0]) & (x <= LATE[1])
    for tag, lab in (("ch", "chosen c"), ("fx", "c = 0.1")):
        bt, be = d5(S["trained"][tag]["bpr"]), d5(S["echo"][tag]["bpr"])
        vt, ve = d5(S["trained"][tag]["pr"]), d5(S["echo"][tag]["pr"])
        dch, dte, dce = bt - bt[:, :1], bt - be, be - be[:, :1]
        lo_c, hi_c = band(dch)
        lo_e, hi_e = band(dte)
        e1 = S["echo"]["c"] == 1
        print(f"{lab}: Trained RNN D5 minus its initial value, 95% bound below 0 at {_runs(x, hi_c < 0)}; 5% bound "
              f"above 0 at {_runs(x, lo_c > 0)}")
        print(f"{lab}: Trained RNN D5 minus the ESN's at the same checkpoint, 95% bound below 0 at "
              f"{_runs(x, hi_e < 0)}; 5% bound above 0 at {_runs(x, lo_e > 0)} (the ESN's chosen c is 1 at "
              f"{_runs(x, e1)}, where the Trained RNN's is " + " / ".join(
                  f"{c:g}" for c in np.unique(S["trained"]["c"][e1])) + ")")
        print(f"  {_ex(EARLY[0])}-{_ex(EARLY[1])}: Trained RNN D5 {vt[early].min():+.3f} to {vt[early].max():+.3f}, "
              f"ESN {ve[early].min():+.3f} to {ve[early].max():+.3f}; Trained minus initial "
              f"{(vt - vt[0])[early].min():+.3f} to {(vt - vt[0])[early].max():+.3f} (95% bound below 0 at "
              f"{_runs(x[early], (hi_c < 0)[early])}), Trained minus ESN {(vt - ve)[early].min():+.3f} to "
              f"{(vt - ve)[early].max():+.3f} (95% bound below 0 at {_runs(x[early], (hi_e < 0)[early])})")
        for s in (20, 100, 120, 140, 160, 180, 200, 480, 1920, 2400, x[-1]):
            k = _k(x, s)
            print(f"  {_ex(x[k])}: change from initial {_ci(vt[k] - vt[0], dch[:, k])}; Trained - ESN "
                  f"{_ci(vt[k] - ve[k], dte[:, k])}")
        for a, b in ((DESCENT, EARLY), (RECOVERY, DESCENT), ((8800, 24000), RECOVERY)):
            print(f"  mean D5 over {_ex(a[0])}-{_ex(a[1])} minus mean over {_ex(b[0])}-{_ex(b[1])}: "
                  f"{_ci(_mean_in(vt, x, *a) - _mean_in(vt, x, *b), _mean_in(bt, x, *a) - _mean_in(bt, x, *b))}")
        ka, kb = _k(x, 1920), _k(x, 480)
        print(f"  D5 at {_ex(x[ka])} minus D5 at {_ex(x[kb])} (chosen c {S['trained']['c'][ka]:g} / "
              f"{S['trained']['c'][kb]:g}): {_ci(vt[ka] - vt[kb], bt[:, ka] - bt[:, kb])}")
        sl = band(bt)[1] < 0
        print(f"  {_ex(LATE[0])}-{_ex(LATE[1])}: Trained RNN D5 {vt[late].min():+.3f} to {vt[late].max():+.3f}, 95% "
              f"bound below 0 at {sl[late].sum()} of {late.sum()} checkpoints ({_runs(x[late], sl[late])})")
        a, b = (8800, x[-1]), (10, 1200)
        ma, mb = (x >= a[0]) & (x <= a[1]), (x >= b[0]) & (x <= b[1])
        print(f"  ESN D5 up to {_ex(b[1])} {ve[mb].min():+.3f} to {ve[mb].max():+.3f}, from {_ex(a[0])} "
              f"{ve[ma].min():+.3f} to {ve[ma].max():+.3f}; minus its initial value, 95% bound below 0 at "
              f"{_runs(x, band(dce)[1] < 0)}; end {_ci(ve[-1] - ve[0], dce[:, -1])}; mean from {_ex(a[0])} minus "
              f"mean up to {_ex(b[1])} "
              f"{_ci(_mean_in(ve, x, *a) - _mean_in(ve, x, *b), _mean_in(be, x, *a) - _mean_in(be, x, *b))}")

    print("\n=== TRAINED RNN MINUS ESN (D5, paired) AT ONE COMMON PENALTY AND AT THE MIXED PENALTIES (float32, as "
          "td_analysis; c <= 1e-3: see PRECISION)")
    _net_pairs(D, Y, g, W, S["trained"]["x"])

    print("\n=== GRADED: D4 = R^2_4 - mean(R^2_1..3) (chosen c); T2 = mean(R^2_4, R^2_5) - mean(R^2_1..3)")
    for run, name, _ in RUNS:
        x = S[run]["x"]
        v, b = d4(S[run]["ch"]["pr"]), d4(S[run]["ch"]["bpr"])
        print(f"{name}: D4 95% bound below 0 at {(band(b)[1] < 0).sum()} of {len(x)}: {_runs(x, band(b)[1] < 0)}; "
              f"range {v.min():+.3f} to {v.max():+.3f}; at " + "; ".join(
                  f"{_ex(x[_k(x, s)])}: {_ci(v[_k(x, s)], b[:, _k(x, s)])}" for s in (10, 200, 480, 1120, 1920, 2400))
              + f"; end {_ci(v[-1], b[:, -1])}")
        for tag, lab in (("ch", "chosen c"), ("fx", "c = 0.1")):
            v, b = t2(S[run][tag]["pr"]), t2(S[run][tag]["bpr"])
            print(f"  {name} T2 ({lab}): 95% bound below 0 at {(band(b)[1] < 0).sum()} of {len(x)}; initial "
                  f"{_ci(v[0], b[:, 0])}; end {_ci(v[-1], b[:, -1])}; end minus initial (paired) "
                  f"{_ci(v[-1] - v[0], b[:, -1] - b[:, 0])}")

    print("\n=== ABSOLUTE UNITS (counts^2, summed over the 5 counts)")
    Yc = Y - Y.mean(0)
    sst = (Yc ** 2).sum(1)
    print(f"total count variance SST/n {sst.mean():.1f} (per count " + " / ".join(f"{v:.1f}" for v in (Yc ** 2).mean(0))
          + "); per fifth, around the decoder-test mean " + " / ".join(f"{sst[g == j].mean():.0f}" for j in range(NG))
          + ", around the fifth's own mean " + " / ".join(
              f"{((Y[g == j] - Y[g == j].mean(0)) ** 2).sum(1).mean():.0f}" for j in range(NG)))
    for run, name, _ in RUNS:
        x, sse = S[run]["x"], S[run]["ch"]["sse"]
        km = int(np.argmin(S[run]["ch"]["tot"]))
        mse = np.stack([sse[:, g == j].mean(1) for j in range(NG)], 1)                 # (K, NG)
        for k in sorted({_k(x, s) for s in (10, 200, 480, 1920, 2400)} | {km, len(x) - 1}):
            print(f"{name} {_ex(x[k])}: per-fifth MSE " + " / ".join(f"{v:.1f}" for v in mse[k])
                  + f"; fifth 5 minus the mean of fifths 1-4 {mse[k, 4] - mse[k, :4].mean():+.1f}")
        print(f"  {name}: added error from the initial network to the {'dip' if run == 'trained' else 'minimum'} "
              f"({_ex(x[km])}): " + " / ".join(f"{v:+.1f}" for v in mse[km] - mse[0]))

    print("\n=== ENVIRONMENTS")
    envs = np.unique(env)
    tab = np.array([[np.sum((g == j) & (env == e)) for e in envs] for j in range(NG)])
    eff = [1 / ((t / t.sum()) ** 2).sum() for t in tab]
    print("environments per fifth " + " / ".join(str(int((t > 0).sum())) for t in tab) + " of 25; effective number "
          + " / ".join(f"{v:.1f}" for v in eff) + f"; each environment's share of the top fifth {tab[4].min()}-"
          f"{tab[4].max()} of its 200 episodes")
    ix = np.searchsorted(envs, env)

    def between(v):
        mu = np.array([v[env == e].mean(0) for e in envs])[ix]
        return ((mu - v.mean(0)) ** 2).sum(-1).mean() / ((v - v.mean(0)) ** 2).sum(-1).mean() if v.ndim > 1 else \
            np.var(mu) / np.var(v)
    print(f"between-environment share of the variance: FR_T {between(fr):.1%}, log FR_T {between(np.log(fr)):.1%}, "
          f"counts {between(Y):.1%}")

    print("\n=== WITHIN-ENVIRONMENT SPLITS (chosen c): D5 [5-95%]")
    gA = np.zeros(n, int)
    for e in envs:
        i = np.flatnonzero(env == e)
        gA[i] = np.argsort(np.argsort(fr[i])) * NG // len(i)
    gB = fifths(fr - np.array([fr[env == e].mean() for e in envs])[ix])[0]
    lf = np.log(fr)
    gC = fifths(lf - np.array([lf[env == e].mean() for e in envs])[ix])[0]
    for lab, gg in (("A fifths cut within each environment", gA), ("B environment-centred FR_T", gB),
                    ("C environment-centred log FR_T", gC)):
        print(f"{lab}: fifth sizes {np.bincount(gg).tolist()}, same fifth as the global split {(gg == g).mean():.1%}")
        for run, name, _ in RUNS:
            x, sse = S[run]["x"], S[run]["ch"]["sse"]
            pr, tot = split(sse, Y, gg)
            bpr, _ = split(sse, Y, gg, W)
            v, b = d5(pr), d5(bpr)
            km = int(np.argmin(tot))
            print(f"  {name}: parts at the {'dip' if run == 'trained' else 'minimum'} " + " / ".join(
                f"{u:.3f}" for u in pr[km]) + "; D5 at " + "; ".join(
                f"{_ex(x[_k(x, s)])}: {_ci(v[_k(x, s)], b[:, _k(x, s)])}" for s in (200, 480, 1120, 1920))
                + f"; end {_ci(v[-1], b[:, -1])}; 95% bound below 0 at {_runs(x, band(b)[1] < 0)}")

    print("\n=== RESAMPLING VARIANTS (D5, chosen c)")
    rng = np.random.default_rng(BOOT25_SEED)
    tenv = np.unique(env)
    B25 = np.zeros((gap.N_BOOT, gap.N_ENV))
    B25[:, tenv] = rng.multinomial(len(tenv), np.full(len(tenv), 1 / len(tenv)), size=gap.N_BOOT)
    for run, name, _ in RUNS:
        x, sse = S[run]["x"], S[run]["ch"]["sse"]
        lo0, hi0 = band(d5(S[run]["ch"]["bpr"]))
        for lab, Wv, cen in (("SST around the full decoder-test mean (the first version)", W, "test"),
                             (f"25-of-25 bootstrap of the decoder-test environments (seed {BOOT25_SEED})", B25[:, env],
                              "resample")):
            lo, hi = band(d5(split(sse, Y, g, Wv, cen)[0]))
            mv = max(np.abs(lo - lo0).max(), np.abs(hi - hi0).max())
            print(f"{name}, {lab}: interval ends move by at most {mv:.3f}; 95% bound below 0 at {(hi < 0).sum()} "
                  f"checkpoints (as used: {(hi0 < 0).sum()})")

    print("\n=== PRECISION: the float32 solve (td_analysis's) vs the same decoder solved in float64 (build64: same "
          "replayed u, every checkpoint and penalty)")
    path64 = os.path.join(OUT, "decode64.npz")
    if not os.path.exists(path64):
        print("decode64.npz missing: run build64")
        return
    with np.load(path64) as z:
        E = {k: z[k] for k in z.files}
    for run, name, _ in RUNS:
        x = S[run]["x"]
        assert np.array_equal(E[f"{run}_x"], x)
        c64 = np.array(RIDGE_C)[E[f"{run}_val_r2"].argmax(1)]
        print(f"{name}: float32 SSE recomputed from build64's replay vs the cache max |d| {E[f'{run}_d32'].max():.1e} "
              f"(every penalty); float64 choice of c = the float32 choice at {(c64 == S[run]['c']).sum()} of "
              f"{len(x)} checkpoints")
        d5_64, ks = [], (_k(x, 480), _k(x, 1120), _k(x, 1920), len(x) - 1)
        for i, c in enumerate(RIDGE_C):
            s32, s64 = D[f"{run}_sse_c"][:, i], E[f"{run}_sse_c"][:, i]
            pr32, tot32 = split(s32, Y, g)
            pr64, tot64 = split(s64, Y, g)
            d5_64.append(d5(pr64))
            dd, dt_ = np.abs(d5(pr32) - d5(pr64)), np.abs(tot32 - tot64)
            print(f"  c={c:g}: max |d| D5 {dd.max():.4f} (at {_ex(x[dd.argmax()])}), total R^2 {dt_.max():.4f} (at "
                  f"{_ex(x[dt_.argmax()])}), per-episode SSE {np.abs(s32 - s64).max():.3g} counts^2; float64 total "
                  f"R^2 minimum {tot64.min():.3f} at {_ex(x[tot64.argmin()])}, range {tot64.min():.3f} to "
                  f"{tot64.max():.3f}")
            if c < 1e-2:
                b32, b64 = d5(split(s32[list(ks)], Y, g, W)[0]), d5(split(s64[list(ks)], Y, g, W)[0])
                print("    D5 float32 / float64 at " + "; ".join(
                    f"{_ex(x[k])}: {_ci(d5(pr32)[k], b32[:, j])} / {_ci(d5(pr64)[k], b64[:, j])}"
                    for j, k in enumerate(ks)))
        d5_64 = np.stack(d5_64)
        print("  float64 D5 at " + "; ".join(f"{_ex(x[k])}: " + " / ".join(f"{v:+.3f}" for v in d5_64[:, k])
                                             for k in ks) + " (c = 1e-4 / 1e-3 / .01 / .1 / 1)")
        lo, hi = band(d5(S[run]["ch"]["bpr"]))
        print(f"  float64: D5's range over the five c larger than the chosen-c interval's width at "
              f"{(np.ptp(d5_64, 0) > hi - lo).sum()} of {len(x)} checkpoints, over c = .01-1 at "
              f"{(np.ptp(d5_64[2:], 0) > hi - lo).sum()}")
    print("float64, Trained RNN minus ESN (D5, paired) at one common penalty and at the mixed penalties:")
    _net_pairs(E, Y, g, W, S["trained"]["x"])


# ═══════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════

def figure():
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from main.utils import fig_path, apply_fig_style

    class EdgeBand(HandlerBase):
        """legend key of the fixed-c curve: its dashed line between the two thin lines of its band's edges"""
        def create_artists(self, legend, orig, xd, yd, w, h, fontsize, trans):
            x0, y0 = -xd, -yd
            return [Line2D([x0, x0 + w], [y, y], color=orig.get_color(), lw=lw, ls=ls, transform=trans)
                    for y, lw, ls in ((y0, EDGE_LW, "-"), (y0 + h, EDGE_LW, "-"), (y0 + h / 2, 1.5, DASH))]
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    FL, TS = fs["label"], 12.
    S = stats()
    edges = S["edges"]
    acc = gap.load("jumps")
    acc0 = float(acc["c0"][list(acc["names"]).index("goal accuracy at T")])
    cols = plt.cm.viridis(np.linspace(.08, .92, NG))
    fig = plt.figure(figsize=(12.6, 10.6))
    outer = fig.add_gridspec(2, 2, height_ratios=(1, 1.05), hspace=.34, wspace=.2, left=.075, right=.985, top=.865,
                             bottom=.165)
    xt = ([10, 1e2, 1e3, 1e4], ["0", "100", "1k", "10k"])

    def xaxis(a_, label=True):
        a_.set_xscale("log")
        a_.set_xlim(8, 7e4)
        a_.set_xticks(xt[0])
        a_.set_xticklabels(xt[1] if label else [])
        a_.minorticks_off()
        a_.spines[["top", "right"]].set_visible(False)

    tops, cons = [], []
    for col, (run, name, color) in enumerate(RUNS):
        r = S[run]
        x, ch, fx = r["x"], r["ch"], r["fx"]
        # a / b: the parts through training
        a_ = fig.add_subplot(outer[0, col])
        lo, hi = band(ch["bpr"])
        for j in range(NG):
            a_.fill_between(x, lo[:, j], hi[:, j], color=cols[j], alpha=.16, lw=0)
            a_.plot(x, ch["pr"][:, j], color=cols[j], lw=1.9,
                    label=f"{j + 1}: {edge_label(edges[j])}-{edge_label(edges[j + 1])}")
        a_.plot(x, ch["tot"], color="k", lw=1.2, ls=(0, (4, 2)),
                label="all 5,000 decoder-test episodes\n(summary figure, panel d)")
        a_.axvline(acc0, color="k", lw=1, ls=":")
        xaxis(a_)
        a_.set_xlabel("training episodes", fontsize=FL)
        a_.set_ylabel(r"held-out R$^2$, part of each FR fifth", fontsize=FL)
        a_.set_title(f"{'ab'[col]}   {name}: counts $n_T$ decoded from $u$ at step 30, per FR fifth\n"
                     r"R$^2_g = 1 - $SSE$_g / (n_g\,$SST$/n)$", loc="left", fontsize=TS)
        tops.append(a_)
        # c / d: the top fifth's contrast, chosen and fixed penalty
        inner = outer[1, col].subgridspec(2, 1, height_ratios=(1, .2), hspace=.1)
        a_ = fig.add_subplot(inner[0])
        a_.axhline(0, color="k", lw=.9)
        lo, hi = band(d5(ch["bpr"]))
        a_.fill_between(x, lo, hi, color=color, alpha=.24, lw=0)
        for e_ in band(d5(fx["bpr"])):
            a_.plot(x, e_, color=DARK[run], lw=EDGE_LW)
        a_.plot(x, d5(ch["pr"]), color=color, lw=2.1)
        a_.plot(x, d5(fx["pr"]), color=DARK[run], lw=1.5, ls=DASH)
        hfx = Line2D([], [], color=DARK[run])
        a_.legend([(Patch(facecolor=color, alpha=.24, lw=0), Line2D([], [], color=color, lw=2.1)), hfx],
                  ["chosen c (as in a, b) and its 5-95%", f"fixed c = {C_FIX:g} and its 5-95%"],
                  handler_map={hfx: EdgeBand()}, loc="upper left", frameon=False, fontsize=FL - 1, handlelength=2.6,
                  handleheight=1.3)
        a_.axvline(acc0, color="k", lw=1, ls=":")
        xaxis(a_, label=False)
        a_.set_ylabel(r"$\Delta_5$ (R$^2$ units)", fontsize=FL)
        a_.set_title(f"{'cd'[col]}   {name}: top FR fifth minus the other four\n"
                     r"$\Delta_5 = $R$^2_5 - \frac{1}{4}\,\Sigma_{g=1}^{4}\,$R$^2_g$", loc="left", fontsize=TS)
        cons.append(a_)
        # the chosen penalty
        s_ = fig.add_subplot(inner[1], sharex=a_)
        s_.step(x, r["c"], where="mid", color=color, lw=1.6)
        s_.axhline(C_FIX, color=DARK[run], lw=1, ls=DASH)
        s_.set_yscale("log")
        s_.set_ylim(.004, 3)
        s_.set_yticks([.01, .1, 1])
        s_.set_yticklabels([".01", ".1", "1"])
        s_.minorticks_off()
        s_.set_ylabel("chosen c", fontsize=FL - 1)
        xaxis(s_)
        a_.tick_params(axis="x", labelbottom=False)
        s_.set_xlabel("training episodes", fontsize=FL)
    lo_ = min(a_.get_ylim()[0] for a_ in tops)
    hi_ = max(a_.get_ylim()[1] for a_ in tops)
    for a_ in tops:
        a_.set_ylim(lo_, hi_)
    lo_ = min(a_.get_ylim()[0] for a_ in cons)
    hi_ = max(a_.get_ylim()[1] for a_ in cons)
    for a_ in cons:
        a_.set_ylim(lo_, hi_)
    tops[0].legend(loc="lower left", frameon=False, fontsize=FL - 1, handlelength=1.6, title=r"FR$_T$ fifth (nats)",
                   title_fontsize=FL - 1)
    fig.suptitle("Counts decoded from the state, split by factorization regret (FR) at T: all five FR fifths dip "
                 "together (a); the top fifth's small deficit (c)\ngrows as the Trained RNN descends, is unresolved at "
                 "the dip, is not detectably larger in the recovery, and is back to its initial size by the end",
                 fontsize=TS + 1, y=.975)
    foot = ("Data: 10,000 probes (50 held-out environments x 200 episodes, 30 steps); 114 checkpoints per network "
            "(seed 20261029). The decoder is that of the summary figure's panel d:\nfitted on 25 of these "
            "environments, its penalty c chosen on 5 of them and re-chosen at every checkpoint (strips), tested on "
            "the other 25 = 5,000 episodes, 1,000 per FR fifth;\ndashed in c, d: c = 0.1 at every checkpoint. FR at "
            r"T = the mean of the two variables' KL(Bayes $\Vert$ naive) of the marginal beliefs at step 30."
            "\nBands: 5-95% over 1,000 resamples of the 25 decoder-test environments, with the "
            "decoder and its penalty fixed (c, d: shaded for the chosen c, between thin lines for c = 0.1).\n"
            "Dotted: the Trained RNN passes naive.")
    fig.text(.5, .062, foot, ha="center", va="center", fontsize=FL - 1, linespacing=1.55)
    path = fig_path("td_tt_counts_fr.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    what = sys.argv[1:] or ["figure"]
    if "build" in what:
        build()
    if "build64" in what:
        build64()
    if "numbers" in what:
        numbers()
    if "figure" in what:
        figure()
