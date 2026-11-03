"""The neural timeline of the Trained RNN, on the footing of the behavioural one (John 11-02: "Can we identify the
neural changes that precede [the behavioural developments]? How does the network get to the correct answer before
learning optimal bayes dynamics? Does it co-occur with changes in effective dimensionality, weight gradients,
correlations with entanglement, etc? ... Ideally ... a causal account"). Every neural measure is reduced per test
environment and resampled with the behavioural timeline's shared matrix (td_tt_gap.boot_counts), so each neural crossing
is paired with each behavioural crossing of td_tt_timeline's tested rows. The ESN is the null, placed on the Trained
RNN's scale; the slower trained runs (power-1 loss, widths 200 / 100 / 50) test whether a precursor moves with the
behaviour. See also td_tt_replication for the cross-run results with each run's own behaviour rows (the order on each
run's clock, matched accuracy, the N100 / N50 natural experiment) and td_tt_credit_lags for the credit at every lag
(r(coherence, entanglement) per lag and cumulated over lags).
-> main/REPLICATION/figure_bin/td_tt_neural.png         figure(): (a) progress f of every row, Trained RNN (top: the
                                                           behavioural families' mean f), (b) the ESN on the Trained
                                                           RNN's scale, (c) crossings against the families' c10 / c50
   main/REPLICATION/figure_bin/td_tt_neural_order.png   figure_order(): every event x every family (paired shares)
   numbers()      every number below;   synthesis()  the export
   caches main/DATA/experimental/training_dynamics/through_training/neural/: ck_<run>.npz (per-environment primitives
   from analysis/<run>/ck_*.npz), replay_<run>/rp_EEEEE.npz + replay_<run>.npz (CPU replay, below), memerr_<run>.npz
   (the two terms of memory use and naive's memoryless reference, memory_errors()), synthesis.npz

Definitions (a row is one measure y per checkpoint: computed per probe (or per probe-step), summed or summarised per
environment (env = probe // 200), then averaged over environments; resamples re-weight the 50 environments):
  memory (interventions on the state at inference; per probe at T, both variables, each scored against itself):
    memory benefit at T, cut to w = 1, 2 steps: accuracy intact minus accuracy with the memory cut to the last w inputs
      (td_analysis an6, ck trunc_right: the cell rerun from its start state at every step, per-step logits still
      summed). a = 0 (a memoryless observer such as naive loses nothing), b = Bayes's own benefit at the same cut on
      the same probes (right_bayes - right_window_w).
    accuracy at T with memory cut to 1 step (the network's memoryless part): a = the initial network, b = naive's
      accuracy (step-wise Bayes, the best memoryless observer, is higher still).
    benefit of memory older than 5 steps minus Bayes's (w = 5 benefit minus Bayes's own): rises and falls; a = 0
      (Bayes's level), b = its peak (3-checkpoint running mean, per resample).
    memory use in the step update (analysis/ent_memcut_var_<run>.npz): the update error (Jeffreys/2 between Bayes's
      belief at t-1 moved by the network's step-t increment and Bayes's belief at t) with the memory cut to the last
      step minus intact (> 0: memory makes the step update more like Bayes's), mean over variable-steps 2-30; also on
      each step's most entangled fifth (each variable with its own entanglement, both variables ranked within the
      step). a = 0, b = Bayes's own memory use in the same metric (its step-wise increment instead of its own), i.e.
      step-wise Bayes's memoryless update error. b depends on that choice of memoryless reference: naive's increment
      gives the other reference (memory_errors: nmu_*). The two terms, the intact and the cut update error, are
      printed separately by numbers().
  activity (per probe, mean over steps 2-30; a = the initial network, b = the end value, both per resample):
    effective dimension (ck effdim_u: projection PR of u on the checkpoint's own PCA basis), |u|, turn
    1 - cos(u_{t-1}, u_t); state dimension per step = PR tr(C)^2 / tr(C^2) of the covariance of u at a step, averaged
    over steps 2-30, within each environment (200 probes; mean over environments) and pooled over environments (the
    locked convention; resampled exactly from per-environment sums, see pr_resampled, and bias-corrected).
  locks: within-episode r with entanglement e_t (mean of the two variables' ent) over steps 17-30, both z-scored
    within the episode (td_tt_credit_lags.lag_products at lag 0), averaged over episodes: |u|, effective dimension,
    turn. a = 0, b = the end value.
  credit (td_analysis credit(): 5,000 probes x error steps 17-30, 85-109 per environment; cross-band: the first 2,000):
    coherence ||Sigma_k delta_k||^2 / Sigma_k ||delta_k||^2 (per environment the median over its probe-steps; a = the
    initial network, b = 1: below 1 the lag pieces cancel); update dimension (median; own change); r(update dimension,
    e) (decoupling: a = the initial network's novelty-driven value, b = 0; negative lock: a = 0, b = the end value);
    cross-band share on the final checkpoint's 10% bands (median; a = 0, b = the end value; below 0 = band pieces
    cancel); r(coherence, e) and r(cross-band share, e) at lag 0 (a = 0, b = end) and summed over lags 1-4 (errors
    1-4 steps after the entangled step; td_tt_credit_lags; they rise and fall: b = the peak).
  weights (no probe noise, no resamples: a crossing's bracket is the two checkpoints around it, marked *): W_hh block
    spectral radius > 1 (i, f, g, o); effective rank of W_hh - W_hh(0) (top 64 singular values); ||W - W(0)|| per
    tensor (own change). The counts of units with tau > 5 / > 10 (tau = 1 / (1 - mean forget gate) over probes and
    steps 2-30) ARE resampled: the replay stores the per-environment mean gate per unit.
  CPU replay (replay(): fp32 hand LSTM = td_analysis's; all 10,000 probes, every checkpoint): per step the
    per-environment sums of u, u u^T traces, Frobenius products and s_f^T S_e s_g that give the pooled PR of every
    resample exactly (checked against brute force), the within-environment PR, and the per-environment mean forget gate
    per unit.
  progress f = (y - a) / (b - a): one affine map per row (no own-spread normalisation). Events: f stays past .1
    (onset), .5 (half-way) and the natural boundary (f = 1: Bayes's level, coherence 1, r(update dimension) = 0,
    naive's accuracy; f = 0: the sign of memory use, of r(|u|, e), of the cross-band share) to the end of training
    (td_tt_gap.crossing, log-interpolated); rows that rise and fall: half the peak, first reached, and back below half
    the peak for good. [5-95%] over the 1,000 shared resamples; drawn if at least 95% of the resamples cross.
  Order tests (Trained RNN): every event against every tested behavioural row's c10 and c50 (td_tt_timeline: rise-
    shaped, paired; families answers, splitting, holding, coupling): interval rule (before / overlaps / after) and the
    paired share P(neural < behaviour) (weight rows: the point time against the behavioural resamples, unpaired). Per
    family: before every row / after every row / overlaps, and P(before every row), P(after every row) over the paired
    resamples. Behavioural crossings past from the start in at least 5% of the resamples cannot be preceded and are
    left out of the family verdicts (the three splitting onsets, the fold index onset).
  ESN: the same rows with the Trained RNN's a and b (absolute units, per resample): a measure that barely moves is not
    stretched to the full scale.
  Cross-run check: the same rows on trained_p1, trained_N200, trained_N100, trained_N50 (each on its own scale and
    resamples; no replay rows; no credit rows for p1) against each run's accuracy passing naive (the timeline's
    behavioural rows exist only for the Trained RNN and the ESN).

Results (Trained RNN; [5-95%]; * = weight row, checkpoint bracket; P = paired share; every number printed by numbers()):
  1. How the answers come before Bayes-like dynamics: memory carries the answers from the start, but it makes the step
     update more Bayes-like only after the answers' onset, around their half-way. With the memory cut to 1 step the
     goal accuracy at T is .157 / .166 / .192 / .221 at 300 / 560 / 800 / 1,360 training episodes while the intact
     network is .230 / .331 / .441 / .609 (naive .464, step-wise Bayes .608); this memoryless part is half-way from the
     untrained network to naive's accuracy only at 3,833 [3,490-5,468], with the late families, and reaches it at the
     end (39,701, 68% of resamples). Memory use in the step update is +.0020 nats at the untrained network, negative
     (memory makes the update less like Bayes's) at every checkpoint from 20 to 1,120 on the most entangled fifth and
     to 1,600 overall, and positive for good from 1,163 [1,060-1,305] / 1,632 [1,462-1,785]: after every answers onset
     row, overlapping the answers' half-way (P(after every row) .00 / .57), before every holding and coupling half-way
     row (P(before all) 1.00). It is half Bayes's own memory use at 2,238 [2,079-2,418] / 3,252 [3,068-3,372] and past
     it at 4,128 [3,744-4,651] / 6,197 [5,738-7,222], inside the late half-way window; these two events depend on the
     reference (below). Reliance on memory older than 5 steps beyond Bayes's (peak .132) is half-way up at 958
     [857-1,059] and back below half at 6,625 [5,871-7,465]; it never returns to Bayes's level (+.009 at the end).
     What the memory-use crossings time: memory use is the cut minus the intact update error, and overall the intact
     error does the moving (intact .0640 / .0607 / .0498 / .0284 / .0033 nats at 320 / 800 / 1,600 / 4,000 / end, cut
     .0524 / .0510 / .0494 / .0434 / .0296; from 320 to 1,680 memory use gains .0122 = the intact error's fall .0154 +
     the cut error's change -.0031), so the overall crossing (1,632) mostly times the intact update becoming more
     Bayes-like (a behavioural quantity) over a nearly flat memoryless one. On the most entangled fifth the cut error
     rises too (320 to 1,200: +.0251 = .0141 intact fall + .0111 cut rise). 'Half' and 'past Bayes's own' take
     step-wise Bayes's memoryless update error as b (.0225 / .0539); with naive's (.0418 / .1154) the network is
     half-way at 5,700 [5,184-6,116] / 4,566 [4,161-5,012] and never passes it (largest memory use .0285 / .0730); its
     own cut error at the end (.0296 / .0675) lies between the two references.
  2. Before the answers' half-way (774-1,575; before every row, P(before all) >= .99): the state starts to turn between
     steps (10% 149 [145-153], half 374 [362-383]); the update decouples from input novelty (r(update dimension, e)
     half-way to 0 at 262 [225-282]) and gains dimensions (10% 344 [318-355]); coherence leaves its initial level (10%
     of the way to 1 at 435 [416-458]); errors 1-4 steps after an entangled step get coherent credit (half the peak:
     coherence 467 [432-593], cross-band 450 [372-589]); the state dimension starts to grow (10%: effective dimension
     500 [491-508], per-step PR within environments 507 [498-517], pooled 537 [521-554]); the recurrence turns
     expansive (radius > 1: o 524*, i 594*, f 633*, brackets 480-640). Before the answers' onset (422-885) only events
     up to 303 precede every row.
  3. With the answers' half-way (overlapping its rows): coherence half-way to 1 (767 [713-826]), the within-environment
     PR half-way (788 [771-811]), r(|u|, e) turning positive (789 [678-1,049]), the decoupling complete (r(update
     dimension, e) below 0 at 861 [808-1,066]), long-memory reliance half-way up (958), the pooled PR, update dimension
     and effective dimension half-way (992 [964-1,020], 1,007 [957-1,058], 1,038 [1,018-1,076]), and memory helping the
     step update for good (1,163 on the most entangled fifth, 1,632 overall: after every answers onset row, before
     every holding and coupling half-way row; see 1).
  4. Before every holding half-way row (P(before all) >= .99) but overlapping coupling's half-way (its earliest
     row, the same-diagonal share, at 2,747 [2,111-3,447]): the second wave of long-time-constant units (tau > 10
     count, 10% of its change at 2,061 [1,885-2,298]; P(before every coupling half-way row) .97); the negative
     update-dimension lock (10% at 2,123 [1,763-2,488]; .88); cancelling credit (coherence below 1 at 2,213
     [2,145-2,426]; .89); r(cross-band share, e) starting (10% at 2,495 [2,346-2,650]; .78). Coherence below 1 and the
     cross-band lock's start come after every answers half-way row; the tau > 10 onset and the negative lock overlap
     its latest row (corner answer = truth, 1,575 [1,477-1,895]; P(after every row) .97 / .95).
  5. With the late half-way (holding 3,021-9,245, coupling 2,747-11,103; overlapping their rows): cancellation across
     the final bands (below 0 at 3,322 [3,141-3,479], half its end value at 9,602 [9,261-9,935]); memory use past
     Bayes's own (4,128 / 6,197; reference-dependent, see 1); the memoryless part half-way to naive (3,833); long-tau
     halves (tau > 5: 4,866 [4,510-5,155]; tau > 10: 7,283 [6,580-7,899]); the entanglement locks half-way (update
     dimension 5,199 [4,598-5,848], |u| 5,555 [5,223-5,968], cross-band 5,854 [5,332-6,623], coherence 6,885
     [6,347-7,168]); long-memory reliance falling back (6,625). Only the fall of the lag 1-4 credit lock (16,498
     [13,708-22,579]) comes after every late row.
  6. Onsets: holding's onset (1,166-2,127) is preceded by every event up to the read-in's half change (846*) except
     r(|u|, e) turning positive (789 [678-1,049]), which overlaps the end-mass slope's onset (1,166 [1,020-1,556];
     P(before all) .98); memory helping the most entangled steps (1,163) overlaps it too (P(before all) .54).
     Coupling's onset is preceded only by events of the first 100 episodes (its same-diagonal share starts at 213
     [97-328]); splitting's onsets cannot be ordered (past from the start in at least 5% of the resamples).
  7. Across the slower trained runs (passing naive at 1,256 (p1) / 2,070 (N200) / 6,271 (N100) / 21,072 (N50) vs 835):
     coherence half-way to 1 overlaps passing naive in N1000, N200 and N100 (time ratios .92 / 1.22 / .95), whose
     passing-naive times span a 7.5x range of speeds; in N50 (25x) it comes before it (.55: 11,608 [10,075-14,606] vs
     21,072 [14,871-30,271]). The state turn (half) precedes passing naive in 4 of 5 runs (ratios .31-.73; p1 1.04,
     overlapping), the effective-dimension onset in 5 of 5 (.06-.67). The radius crossings, the update dimension and
     the decoupling precede it in every run at nearly fixed times (ratios down to .01-.12 in N100 / N50). The
     final-band cancellation comes at ~4x the passing-naive time in N1000 / N200 / N100 (3.98 / 4.03 / 4.04; N50 1.68).
  8. The ESN (on the Trained RNN's scale) makes none of the recurrent changes for good: no state turn (end f .02); no
     growth of the within-environment dimension (3.99 -> 3.56) and 15% of the pooled one (10.7 -> 18.1); no
     entanglement locks (r(|u|, e) -.16 -> -.22); no credit reorganisation (coherence 3.14 -> 3.44, update dimension
     1.01, r(update dimension, e) .375 -> .398, cross-band ~0); its r(state turn, e) keeps the untrained network's
     input-driven level (.31 -> .32, f 1.21: not learned); no long-memory reliance; no recurrent weight change (radius
     .59, no long-tau units). On the state, credit and time-constant changes the main title names its largest end f is
     .15 (the pooled PR). Its memory benefit at T ends at about half of Bayes's (.079 / .038 = end f .49 / .52 with the
     memory cut to 1 / 2 steps), but its half-way crossing is not established (only 25% / 59% of resamples cross). Its
     memory helps the step update only up to 180 episodes: +.0020 [+.0020, +.0021] nats at the untrained network
     (shared with the Trained RNN), +.0144 [+.0141, +.0146] at 20 (.64 of Bayes's own; most entangled fifth +.0176,
     .33), +.0067 at 80, +.0024 at 160; the 5-95% band is above 0 at every checkpoint up to 180 (most entangled fifth:
     to 160), about 0 at 200 (-.0001 [-.0007, +.0004]) and below 0 at every checkpoint from 220 (most entangled fifth:
     200) to the end (-.0083): it never helps for good. The transient comes from its memoryless update getting worse
     (cut error .0559 -> .0728 from the untrained network to 20 episodes, intact .0539 -> .0584), not from a better
     intact update. It reaches only the memory benefit's 10% (117 [113-243]), the memoryless part half-way to naive
     (6,607 [4,316-8,114]; it ends at .352), |u| growth (half at 7,521 [7,011-8,061]: read-in gain), the onsets of the
     effective dimension (2,548 [1,261-5,042]) and of the pooled PR (471 [424-4,532]), and the read-in / read-out
     distances (it moves them 3-4x further: 77 / 75 vs 25 / 18).

Failed hypotheses, nulls and counterevidence:
  - Memory benefit at T passing Bayes's own (cut to 1 step: 554 [481-600]; to 2: 410 [334-497]; before passing naive,
    P 1.00, and in all 5 trained runs at .45-.66 of its time) is NOT an independent precursor: at the crossing the
    intact accuracy is .328 and the memory-cut one .164 (both variables; .33-.40 vs .16-.24 in every run), so the
    crossing is the intact accuracy passing ~.33 over a near-chance memoryless part.
  - Memory use in the step update is not a separate neural precursor either: its overall crossing (1,632) is the
    intact update error falling (.0154 of the .0122 gain from 320 to 1,680) over a nearly flat cut error, and its
    'half' and 'past Bayes's own' events move with the memoryless reference (naive's: half 5,700 / 4,566, never
    passed). On the most entangled fifth the cut error rising contributes .0111 of the .0251 gain up to its crossing.
  - Resampling the pooled per-step PR without a correction misleads: resampled PRs are .854-.978 of the full-sample
    value (an environment drawn twice adds no orientation diversity; the shortfall grows with training), and the
    uncorrected half-way interval [924-988] excludes its own point estimate (992).
  - No neural event after the first 100 episodes precedes coupling's onset or splitting's onsets: those behavioural
    onsets are at or near the start.
  - The expansive recurrence, the update dimension and the decoupling from novelty are not paced with the answers: they
    precede passing naive in every width at nearly fixed times (N100: radius 439-757* vs passing naive 6,271), and the
    decoupling never completes (r(update dimension, e) never stays below 0) in N200 / N100 / N50, which pass naive.
  - Cancelling credit (coherence < 1), cross-band < 0 and long-tau units are not needed to pass naive: absent in N100 /
    N50; in N200 coherence < 1 only at 21,397, ten times after it passes naive (2,070).
  - The dimension's half-way points (within-environment PR 788, pooled 992, effective dimension 1,038) are not
    precursors of passing naive (835 [798-876]): they overlap or follow it (in N200 / N100 / N50 the effective dimension
    is half-way before passing naive).
  - Memory helping the most entangled steps (1,163) comes with holding's onset, not before it (P(before all) .54).
  - Untimeable or unspecific: |u| saturates within the first ~20 episodes (10% at 10, half at 11) and grows in the ESN
    too (half at 7,521); the tau > 5 count has no onset (its first wave, 52 units at 20, never falls below 10% of the
    end count); r(state turn, e) is input-driven at init (.31, kept by the ESN), so only its recovery (2,804
    [2,466-3,154]) is a training event.
  Corrected after verification (11-02; the first version said):
  - "memory makes the step update Bayes-like only after the answers": only after their onset; it overlaps their
    half-way (P(after every half-way row) .00 / .57).
  - "together with cancelling credit and long time constants", before holding and coupling are half-way: they precede
    holding's half-way but overlap coupling's (P(before all) .89 / .97).
  - "every event up to 846 precedes holding's onset": r(|u|, e) turning positive (789) overlaps it (P .98).
  - "coherence half-way stays with passing naive over a 25x range of speeds": over 7.5x (N1000 / N200 / N100); in N50
    it comes before (.55).
  - "memory use stays negative until 1,163 / 1,632": it is +.0020 at the untrained network and negative from 20.
  - "the ESN's memory never helps its step update" (already retracted in the inventory): it helps transiently (band
    above 0 up to 180, the untrained network included; up to .64 of Bayes's own) and hurts from 220.
  - Figure: panel c's behaviour bars are ranges of the rows' point crossings, not 5-95% intervals. The shaded early span
    is the answers' half-way only (774-1,575); splitting keeps all three rows in its bar and in the tests (219-927), its
    six-step row marked x (td_tt_splitting flags it as no clean timing marker; without it 793-927).

Checks:
  - The replay reproduces the stored forward pass: |u| to 9.5e-6 at checkpoint 1,000 (verify_replay); over all 114
    checkpoints the per-step PR to 3.3e-7 (relative; ESN 3.6e-7), tau to 2.8e-5 (ESN 2.0e-7), tau counts identical at
    every checkpoint. The resampled-PR formula equals brute force on random data (max relative |diff| printed).
  - Coherence lag products are identical to td_tt_credit_lags' cache (|diff| 0); its crossings are reproduced (lags 1-4
    half the peak 467 [432-593]; lag 0 half the end value 6,885 [6,347-7,168]).
  - Goal accuracy at T from ck 'right' equals the timeline's export (|diff| 1e-16; c0 835, c50 1,394; resampled c50
    identical in 100%).
  - The ESN and the Trained RNN are identical at init on every row (|diff| 0) except the cross-band rows (each network's
    own final bands).
  - Memory use = the cut minus the intact update error (memory_errors) equals the stored rows exactly (|diff| 0, both
    networks, all variable-steps and the most entangled fifth).
  - Memory benefit on the goal variable only: passes Bayes's own at 552 [484-603] (both variables 554 [481-600]); w = 2:
    417 [381-493] (410 [334-497]).
  - Cross-band r on each checkpoint's own bands (5,000 probes): lag 0 10% at 2,070 [1,786-2,456], half 5,965
    [5,830-6,820] (final bands 2,495 / 5,854); lags 1-4 half the peak at 522 [462-648] (450).
  - Pooled PR with the bias correction: 10% at 537 [521-554], half 992 [964-1,020] (point estimates inside).

Caveats: one seed and one probe world; intervals resample environments, not seeds. Correlational except the memory
cuts, which are interventions at inference at a fixed checkpoint (start-state confound: the cut state is a step-1..w
state from the trained start state); nothing here manipulates learning. The memory-use rows largely time the network's
own intact update error (1). The credit rows are the gradient a probe would give (both variables' losses, error steps
17-30, 16 lags), not the realised update. The effective dimension uses each checkpoint's own basis (it rotates until
~2k); the cross-band rows use each network's own final bands (the ESN's frame differs). The late neural changes keep
changing after behaviour saturates, so co-occurrence with the late families is not specific by itself. Family verdicts
depend on which rows a family has; the per-row verdicts and shares are in synthesis.npz (ord_verdict, ord_p) and
printed by numbers().

Run from the repository root (CPU only; at most 4 processes):
    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
        /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_neural.py build \\
        trained echo trained_p1 trained_N200 trained_N100 trained_N50        # ~5 min, one process
    ... td_tt_neural.py replay <run> <part> <nparts>     # ~75 s per checkpoint per process at 2 threads (~1.5 h
                                                         # for trained and echo, 3-4 processes)
    ... td_tt_neural.py verify_replay <run> <ep>          # the replay against the stored ck file of one checkpoint
    MPLBACKEND=Agg ... td_tt_neural.py numbers synthesis figure                  # ~1 min"""
import os
import sys
import glob
import time
import warnings
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_credit_lags as cl                                                                # noqa: E402

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
OUT = os.path.join(TD, "through_training", "neural")
RUNS = ("trained", "echo")
NAME = {"trained": "Trained RNN", "echo": "ESN"}
COL = {"trained": "#1f77b4", "echo": "#ff7f0e"}
T, R, NV, H_DEF = 30, 10, 2, 1000
ACT = ("effdim_u", "norm_u", "norm_c", "turn_u", "turn_c")        # per-probe activity keys (ck)
CRED = ("cr_coh", "cr_dim", "cr_bandx", "cr_E")                     # per (credit probe, error step) keys (ck)
WIN = slice(16, 30)                                                 # steps 17-30: the locked within-episode window
STEPS = slice(1, 30)                                                # steps 2-30: activity means (step 1: start state)
TRUNC_W = (1, 2, 5)
TAU0 = int(np.where(cl.TAU == 0)[0][0])


def ck_files(run):
    return sorted(glob.glob(os.path.join(AN, run, "ck_*.npz")))


def probes():
    return np.load(os.path.join(AN, "probes.npz"))


def _env_sums(env, v, n_env=gap.N_ENV):
    """per-environment sums of v (n, ...) over its rows"""
    out = np.zeros((n_env,) + v.shape[1:])
    np.add.at(out, env, v)
    return out


def _env_median(env, v, n_env=gap.N_ENV):
    """per-environment median of v (n, k) over all its rows and columns"""
    return np.array([np.median(v[env == e]) for e in range(n_env)])


def r_lock(y, e):
    """per episode, the within-episode Pearson r of y and e over the locked window (both z-scored within the episode;
    td_tt_credit_lags.lag_products at lag 0): (n,)"""
    return cl.lag_products(y, e)[:, TAU0]


# ═══════════════════════════════════════════════════════════════════
# Per-environment primitives from the stored checkpoint files
# ═══════════════════════════════════════════════════════════════════

def bayes_refs():
    """per environment (sums over its 200 probes x both variables): Bayes's own memory benefit at T for w = 1, 2, 5
    (right_bayes - right_window_w, each variable with itself), and the goal-only version (sums over 200 probes)"""
    P = probes()
    env = P["env"]
    rb = P["right_bayes"].astype(np.float64)
    ben = np.stack([rb - P[f"right_window{w}"] for w in TRUNC_W], 1)                  # (n, 3, 2)
    ar = np.arange(len(env))
    return dict(bben=_env_sums(env, ben.sum(-1)), bben_goal=_env_sums(env, ben[ar, :, P["goal"]]),
                bnaive2=_env_sums(env, P["right_naive"].sum(-1).astype(np.float64)),
                bsw2=_env_sums(env, P["right_window1"].sum(-1).astype(np.float64)),
                bacc=_env_sums(env, rb[ar, P["goal"]]), nacc=_env_sums(env, P["right_naive"][ar, P["goal"]]
                                                                         .astype(np.float64)))


def build_ck(run, force=False):
    """per checkpoint and environment, from analysis/<run>/ck_*.npz -> through_training/neural/ck_<run>.npz: ben (K,
    50, 3) sums over probes x both variables of (intact - memory cut to w) at T, w = 1, 2, 5; ben_goal (goal
    variable only); acc (K, 50) goal right at T; acc2 (K, 50) both variables right at T (intact); act (K, 50, 5)
    sums over probes of the per-probe mean over steps 2-30 of ACT; lock (K, 50, 5) sums over probes of the per-probe
    within-episode r of ACT with entanglement (steps 17-30); cmed (K, 50, 4) per-environment medians of CRED over
    the credit probes' (probe, step) items, cmed_f (K, 50) the same for cr_bandx_fixed; clag (K, 4, 50, 17), clag_f
    (K, 50, 17) environment sums of the per-episode lag products (td_tt_credit_lags convention) of CRED /
    cr_bandx_fixed with entanglement; cnt_cr, cnt_crf (50,) credit probes per environment; weights: rho (K, 4), tau
    (K, H), sv_dhh (K, 64), w_dinit, w_norm (K, n_tensors), w_names"""
    path = os.path.join(OUT, f"ck_{run}.npz")
    if os.path.exists(path) and not force:
        return dict(np.load(path))
    os.makedirs(OUT, exist_ok=True)
    P = probes()
    env, goal = P["env"], P["goal"]
    ar = np.arange(len(env))
    rev = P["ent"].astype(np.float64).mean(-1)                                       # (n, T): mean of the 2 variables
    files = ck_files(run)
    K = len(files)
    out = {k: [] for k in ("episode", "ben", "ben_goal", "acc", "acc2", "act", "lock", "cmed", "cmed_f", "clag",
                           "clag_f", "rho", "tau", "sv_dhh", "w_dinit", "w_norm")}
    t0 = time.time()
    for i, f in enumerate(files):
        z = np.load(f)
        out["episode"].append(int(z["episode"]))
        tr = z["trunc_right"].astype(np.float64)                                     # (n, 4 [w1, w2, w5, intact], 2)
        b = tr[:, 3:4] - tr[:, :3]                                                   # (n, 3, 2)
        out["ben"].append(_env_sums(env, b.sum(-1)))
        out["ben_goal"].append(_env_sums(env, b[ar, :, goal]))
        out["acc"].append(_env_sums(env, z["right"][ar, -1, goal].astype(np.float64)))
        out["acc2"].append(_env_sums(env, tr[:, 3].sum(-1)))                         # intact, both variables
        A = np.stack([z[k][:, STEPS].astype(np.float64).mean(1) for k in ACT], 1)     # (n, 5)
        out["act"].append(_env_sums(env, A))
        L = np.stack([r_lock(z[k][:, WIN].astype(np.float64), rev[:, WIN]) for k in ACT], 1)
        out["lock"].append(_env_sums(env, L))
        if "cr_coh" in z.files:
            rows = z["cr_rows"]
            ec = rows // gap.PER_ENV
            out["cmed"].append(np.stack([_env_median(ec, z[k]) for k in CRED], 1))
            out["clag"].append(np.stack([_env_sums(ec, cl.lag_products(z[k].astype(np.float64), rev[rows][:, WIN]))
                                         for k in CRED]))
            nf = len(z["cr_bandx_fixed"])
            ef = rows[:nf] // gap.PER_ENV
            out["cmed_f"].append(_env_median(ef, z["cr_bandx_fixed"]))
            out["clag_f"].append(_env_sums(ef, cl.lag_products(z["cr_bandx_fixed"].astype(np.float64),
                                                                rev[rows[:nf]][:, WIN])))
            cnt_cr = np.bincount(ec, minlength=gap.N_ENV).astype(np.float64)
            cnt_crf = np.bincount(ef, minlength=gap.N_ENV).astype(np.float64)
        out["rho"].append(z["rho_hh"][:, 0])
        out["tau"].append(z["tau"])
        out["sv_dhh"].append(z["sv_dhh"])
        out["w_dinit"].append(z["w_dinit"])
        out["w_norm"].append(z["w_norm"])
        w_names = z["w_names"]
        if i % 20 == 0:
            print(f"  {run} {i + 1}/{K} ({time.time() - t0:.0f} s)", flush=True)
    res = {k: np.array(v) for k, v in out.items() if len(v)}
    res["w_names"] = w_names
    if "cmed" in res:
        res.update(cnt_cr=cnt_cr, cnt_crf=cnt_crf)
    mc = os.path.join(AN, f"ent_memcut_var_{run}.npz")
    if os.path.exists(mc):
        res.update(memory_use(run, P, res["episode"]))
    np.savez(path, **res)
    print("saved", path, f"({time.time() - t0:.0f} s)")
    return res


def bayes_memory_use(P, inc="inc_SW"):
    """Bayes's own memory use in the same metric (per variable-step): the update error of the memoryless (step-wise)
    Bayes increment applied to Bayes's belief at t - 1, Jeffreys/2 against Bayes's belief at t (its intact update has
    error 0): (n, T, 2), step 1 = 0. inc="inc_N": the same with naive's increment (the other memoryless reference)"""
    lj = P["log_jb"].astype(np.float64)
    prior = np.concatenate([np.full_like(lj[:, :1], -np.log(R)), lj[:, :-1]], 1)
    a = prior + P[inc].astype(np.float64)
    la = a - np.logaddexp.reduce(a, -1, keepdims=True)
    lb = lj - np.logaddexp.reduce(lj, -1, keepdims=True)
    return .5 * ((np.exp(la) - np.exp(lb)) * (la - lb)).sum(-1)


def top_fifth(P):
    """(n, T, 2) bool: the variable-steps in the most entangled fifth of their step (each variable with its own
    entanglement; both variables' values of a step ranked together, as in the flip scan)"""
    e = P["ent"].astype(np.float64)                                                   # (n, T, 2)
    n = len(e)
    out = np.zeros(e.shape, bool)
    for t in range(T):
        v = e[:, t].reshape(-1)
        out[:, t] = (v >= np.quantile(v, .8)).reshape(n, 2)
    return out


def memory_use(run, P, episodes):
    """per checkpoint and environment, memory use in the per-step update (analysis/ent_memcut_var_<run>.npz:
    err_cut1 - err_int, the update error with the memory cut to the last step minus intact; positive = memory helps):
    mu_all (K, 50) sums over the environment's variable-steps 2-30 (count 200 x 29 x 2), mu_top (K, 50) sums over its
    most entangled fifth (count mu_top_cnt (50,)); Bayes's own: bmu_all, bmu_top (50,)"""
    z = np.load(os.path.join(AN, f"ent_memcut_var_{run}.npz"))
    assert np.array_equal(z["episode"], episodes)
    env = P["env"]
    top = top_fifth(P)
    top[:, 0] = False
    stp = np.zeros((1, T, 1), bool)
    stp[:, 1:] = True
    bm = bayes_memory_use(P)
    out = dict(mu_top_cnt=_env_sums(env, top.sum((1, 2)).astype(np.float64)),
               bmu_all=_env_sums(env, (bm * stp).sum((1, 2))), bmu_top=_env_sums(env, (bm * top).sum((1, 2))))
    A, B = [], []
    for k in range(len(episodes)):
        d = z["err_cut1"][k].astype(np.float64) - z["err_int"][k].astype(np.float64)    # (n, T, 2)
        A.append(_env_sums(env, (d * stp).sum((1, 2))))
        B.append(_env_sums(env, (d * top).sum((1, 2))))
    out.update(mu_all=np.array(A), mu_top=np.array(B))
    return out


def memory_errors(run, force=False):
    """the two terms of memory use, per checkpoint and environment (analysis/ent_memcut_var_<run>.npz): sums of the
    intact update error (ei_*) and of the update error with the memory cut to the last step (ec_*) over the
    environment's variable-steps 2-30 (*_all) and over its most entangled fifth (*_top); and naive's memoryless
    reference in the same metric (naive's increment on Bayes's belief at t - 1, Jeffreys/2 against Bayes's belief at
    t): nmu_all, nmu_top (50,) -> through_training/neural/memerr_<run>.npz (memory use = ec - ei)"""
    path = os.path.join(OUT, f"memerr_{run}.npz")
    if os.path.exists(path) and not force:
        return dict(np.load(path))
    P = probes()
    env = P["env"]
    top = top_fifth(P)
    top[:, 0] = False
    stp = np.zeros((1, T, 1), bool)
    stp[:, 1:] = True
    z = np.load(os.path.join(AN, f"ent_memcut_var_{run}.npz"))
    Ei, Ec = z["err_int"], z["err_cut1"]                                            # read once (float16)
    out = {k: [] for k in ("ei_all", "ec_all", "ei_top", "ec_top")}
    for k in range(len(z["episode"])):
        for key, E in (("ei", Ei), ("ec", Ec)):
            e = E[k].astype(np.float64)
            out[key + "_all"].append(_env_sums(env, (e * stp).sum((1, 2))))
            out[key + "_top"].append(_env_sums(env, (e * top).sum((1, 2))))
    nm = bayes_memory_use(P, "inc_N")
    res = {k: np.array(v) for k, v in out.items()}
    res.update(episode=z["episode"], nmu_all=_env_sums(env, (nm * stp).sum((1, 2))),
               nmu_top=_env_sums(env, (nm * top).sum((1, 2))))
    np.savez(path, **res)
    print("saved", path)
    return res


# ═══════════════════════════════════════════════════════════════════
# CPU replay: per-step PR of u (resampled exactly) and per-environment forget gates
# ═══════════════════════════════════════════════════════════════════

_PAT = np.array([[(k >> i) & 1 for i in range(5)] for k in range(32)], np.float32)


def ckpt_path(run, ep):
    return os.path.join(TD, f"{run}_s20261029", f"ckpt_ep{ep:05d}_net.pth")


def pr_resampled(Tr, D, B, M, n_e, C):
    """participation ratio tr(C)^2 / tr(C^2) of the covariance of all probes pooled over environments, where
    environment e enters with weight C[:, e] (a resample's counts): from the per-environment sums of the (centred)
    states: Tr[e] = sum |u|^2, D[e, f] = s_e . s_f (s_e = sum u), B[e, f] = <S_e, S_f>_F (S_e = sum u u^T),
    M[e, f, g] = s_f^T S_e s_g. Exact: C_w = A - m m^T with A = sum_e w_e S_e / W, m = sum_e w_e s_e / W,
    W = sum_e w_e n_e; tr(C_w) = sum_e w_e Tr_e / W - m.m; ||C_w||^2 = ||A||^2 - 2 m^T A m + (m.m)^2"""
    C = np.atleast_2d(np.asarray(C, np.float64))
    ne = len(n_e)
    W = C @ n_e
    mm = np.einsum("be,ef,bf->b", C, D, C) / W ** 2
    tr = C @ Tr / W - mm
    AA = np.einsum("be,ef,bf->b", C, B, C) / W ** 2
    X = (C @ M.transpose(1, 0, 2).reshape(ne, ne * ne)).reshape(len(C), ne, ne)         # [b, e, g] = sum_f C_bf M_efg
    mAm = np.einsum("be,beg,bg->b", C, X, C) / W ** 3
    tr2 = AA - 2 * mAm + mm ** 2
    return tr ** 2 / tr2, tr


def replay_checkpoint(run, ep, P, C, verify=False):
    """one checkpoint, all 10,000 probes, fp32 hand LSTM (gate order i, f, g, o; td_analysis._cell): per step the
    pooled-over-environments PR of u (full sample and every resample of C), the within-environment PR of u (50,), the
    pooled variance; and the per-environment, per-unit mean forget gate over steps 2-30 (ck "tau" = 1 / (1 - mean f))"""
    import torch
    w = torch.load(ckpt_path(run, ep), map_location="cpu", weights_only=False)["weights"]
    p = {k: v.float() for k, v in w.items() if k not in ("all_K", "all_Q")}
    env = P["env"]
    n = len(env)
    assert np.array_equal(env, np.arange(n) // gap.PER_ENV), "probes must be environment-major, 200 per environment"
    n_e = np.bincount(env, minlength=gap.N_ENV).astype(np.float64)
    ee = torch.as_tensor(env.astype(np.int64))
    code = torch.as_tensor((P["obs"] * (1 << np.arange(5))).sum(-1).astype(np.int64))
    NE = len(P["Z"])
    out = dict(pr_full=np.zeros(T), pr_boot=np.zeros((T, len(C))), pr_env=np.zeros((T, gap.N_ENV)),
               var_full=np.zeros(T), episode=ep)
    with torch.no_grad():
        Zt = torch.as_tensor(P["Z"]).reshape(NE, -1)
        pat = torch.as_tensor(_PAT)
        inp = torch.cat([pat[None].expand(NE, 32, 5), Zt[:, None].expand(NE, 32, Zt.shape[-1])], -1)
        x = torch.relu(inp @ p["classifier_readin.weight"].T + p["classifier_readin.bias"])
        table = x @ p["LSTM.weight_ih_l0"].T + p["LSTM.bias_ih_l0"] + p["LSTM.bias_hh_l0"]   # (NE, 32, 4H)
        H = p["LSTM.weight_hh_l0"].shape[1]
        h = p["STM"].reshape(1, H).expand(n, H).contiguous()
        c = p["LTM"].reshape(1, H).expand(n, H).contiguous()
        Whh = p["LSTM.weight_hh_l0"].T.contiguous()
        F = torch.zeros(gap.N_ENV, H, dtype=torch.float64)
        norms = np.zeros((n, T), np.float32) if verify else None
        for t in range(T):
            pre = torch.addmm(table[ee, code[:, t]], h, Whh)
            i, f, g, o = pre.chunk(4, -1)
            f = torch.sigmoid(f)
            c = f * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            if t >= 1:
                F.index_add_(0, ee, f.double())
            if verify:
                norms[:, t] = h.norm(dim=-1).numpy()
            u = h.double()
            u = u - u.mean(0)                                                       # centring: covariances unchanged
            U = u.reshape(gap.N_ENV, -1, H)                                         # environment-major probes
            s = U.sum(1)                                                            # (50, H)
            S = torch.bmm(U.transpose(1, 2), U)                                     # (50, H, H)
            Sf = S.reshape(gap.N_ENV, -1)
            B = (Sf @ Sf.T).numpy()
            Ss = torch.bmm(S, s.T[None].expand(gap.N_ENV, H, gap.N_ENV))           # (50 e, H, 50 g) = S_e s_g
            M = torch.einsum("fh,ehg->efg", s, Ss).numpy()
            D = (s @ s.T).numpy()
            Tr = S.diagonal(dim1=1, dim2=2).sum(-1).numpy()
            pr, tr = pr_resampled(Tr, D, B, M, n_e, np.ones((1, gap.N_ENV)))
            out["pr_full"][t], out["var_full"][t] = pr[0], tr[0]
            out["pr_boot"][t] = pr_resampled(Tr, D, B, M, n_e, C)[0]
            ne = n_e[0]
            dg = np.einsum("eee->e", M)
            tr_e = Tr / ne - np.diag(D) / ne ** 2
            tr2_e = np.diag(B) / ne ** 2 - 2 * dg / ne ** 3 + np.diag(D) ** 2 / ne ** 4
            out["pr_env"][t] = tr_e ** 2 / tr2_e
            del S, Sf, Ss
    out["f_env"] = (F / (n_e[:, None] * (T - 1))).numpy().astype(np.float32)     # (50, H) mean forget gate
    if verify:
        out["norms"] = norms
    return out


def replay(run, part=0, nparts=1):
    """every nparts-th checkpoint of a run (offset part) -> through_training/neural/replay_<run>/rp_EEEEE.npz (reruns
    skip existing files)"""
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    d = os.path.join(OUT, f"replay_{run}")
    os.makedirs(d, exist_ok=True)
    P = probes()
    C = gap.boot_counts()
    eps = [int(os.path.basename(f)[3:8]) for f in ck_files(run)]
    t0 = time.time()
    for k, ep in enumerate(eps):
        if k % nparts != part:
            continue
        path = os.path.join(d, f"rp_{ep:05d}.npz")
        if os.path.exists(path):
            continue
        res = replay_checkpoint(run, ep, P, C)
        np.savez(path + ".tmp.npz", **res)
        os.replace(path + ".tmp.npz", path)
        print(f"{time.strftime('%H:%M:%S')} {run} ep {ep} ({time.time() - t0:.0f} s): PR {res['pr_full'][1:].mean():.2f}",
              flush=True)


def collect_replay(run):
    """the replay files of a run -> dict: episode (K,), pr_full (K, T), pr_boot (K, T, N_BOOT), pr_env (K, T, 50),
    f_env (K, 50, H)"""
    path = os.path.join(OUT, f"replay_{run}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    eps = [int(os.path.basename(f)[3:8]) for f in ck_files(run)]
    fs = [os.path.join(OUT, f"replay_{run}", f"rp_{ep:05d}.npz") for ep in eps]
    miss = [f for f in fs if not os.path.exists(f)]
    assert not miss, f"{len(miss)} replay files missing for {run}"
    out = {k: [] for k in ("episode", "pr_full", "pr_boot", "pr_env", "var_full", "f_env")}
    for f in fs:
        z = np.load(f)
        for k in out:
            out[k].append(z[k])
    res = {k: np.array(v) for k, v in out.items()}
    np.savez(path, **res)
    print("saved", path)
    return res


def verify_replay(run="trained", ep=1000):
    """the replay against the stored ck file of the same checkpoint: |u| per probe-step, the pooled per-step PR of u
    (ck fac row u, factor total) and tau (ck tau)"""
    P = probes()
    z = np.load(os.path.join(AN, run, f"ck_{ep:05d}.npz"))
    t0 = time.time()
    res = replay_checkpoint(run, ep, P, gap.boot_counts()[:5], verify=True)
    tau = 1 / (1 - res["f_env"].astype(np.float64).mean(0))
    pr_ck = z["fac"][:, 6, 0, 1]
    print(f"replay {run} {ep}: {time.time() - t0:.1f} s; max |d|u|| {np.abs(res['norms'] - z['norm_u']).max():.2e} "
          f"(|u| up to {z['norm_u'].max():.2f}); per-step PR max |diff| vs ck fac "
          f"{np.nanmax(np.abs(res['pr_full'] - pr_ck)):.2e} (PR {np.nanmean(pr_ck[1:]):.2f}); tau max |diff| "
          f"{np.abs(tau - z['tau']).max():.2e}")
    return res


# ═══════════════════════════════════════════════════════════════════
# Rows: a curve, its two references, and its timed events
# ═══════════════════════════════════════════════════════════════════

GROUPS = ("memory", "activity", "lock", "credit", "weights")
GROUP_NAME = {"memory": "memory (cut at inference)", "activity": "activity", "lock": "activity locked to entanglement",
              "credit": "learning signal (credit)", "weights": "weights"}
GROUP_COL = {"memory": "#1b7837", "activity": "#2166ac", "lock": "#5e3c99", "credit": "#b35806",
             "weights": "#4d4d4d"}
ROLES = ("onset", "half", "natural", "fall")


def _end(v):
    """the end value: mean of the last 3 checkpoints (last axis)"""
    return np.asarray(v)[..., -3:].mean(-1)


def _ratio(num, den):
    """a ratio of environment sums: num (K, 50), den (50,) or (K, 50) -> (full (K,), resampled (N_BOOT, K))"""
    den = np.broadcast_to(np.asarray(den, np.float64), num.shape)
    return gap.env_ratio(num.T, den.T)


def _emean(v):
    """an environment-mean statistic: v (K, 50) per-environment values -> (full (K,), resampled (N_BOOT, K))"""
    return gap.env_mean(np.asarray(v).T)


def _const(num, den):
    """a ratio of environment sums that does not change with training: (full float, resampled (N_BOOT,))"""
    f, b = gap.env_ratio(np.asarray(num, np.float64)[:, None], np.asarray(den, np.float64)[:, None])
    return float(f[0]), b[:, 0]


def _ev(lab, role, level, rule="stay"):
    """an event: f passes `level` (rule 'stay': for good, 'first': first reached, 'below': stays below to the end)"""
    return dict(lab=lab, role=role, level=float(level), rule=rule)


def _peak(y):
    return cl._peak(y)


def bracket(x, f, level, sign=1.):
    """the two checkpoints around a stays-past crossing (the resolution of a crossing without resamples)"""
    with np.errstate(invalid="ignore"):
        past = sign * (np.asarray(f) - level) > 0
    if not past[-1]:
        return np.nan, np.nan
    if past.all():
        return float(x[0]), float(x[0])
    k = len(past) - 1 - int(np.argmax(~past[::-1]))
    return float(x[k]), float(x[k + 1])


def timed(row):
    """adds row['f'] (K,), row['fb'] (N_BOOT, K) or None and row['ev'] = events with p, lo, hi, share, boot"""
    x = row["x"]
    a, b = row["a"], row["b"]
    with np.errstate(invalid="ignore", divide="ignore"):
        row["f"] = (row["value"] - a) / (b - a)
        row["fb"] = None if row["boot"] is None else \
            (row["boot"] - np.asarray(row["a_boot"])[:, None]) / (np.asarray(row["b_boot"]) - row["a_boot"])[:, None]
    for e in row["events"]:
        lev, rule = e["level"], e["rule"]
        if rule == "first":
            e["p"] = float(cl.first_crossing(x, row["f"][None], np.array([lev]))[0])
            cb = None if row["fb"] is None else cl.first_crossing(x, row["fb"], np.full(len(row["fb"]), lev))
        else:
            sg = -1. if rule == "below" else 1.
            e["p"] = float(gap.crossing(x, row["f"], lev, sg))
            cb = None if row["fb"] is None else gap.crossing(x, row["fb"], lev, sg)
        e["boot"] = cb
        if cb is None:
            if rule == "first":
                k = int(np.argmax(row["f"] > lev)) if np.any(row["f"] > lev) else None
                e["lo"], e["hi"] = (np.nan, np.nan) if k is None else (float(x[max(k - 1, 0)]), float(x[k]))
            else:
                e["lo"], e["hi"] = bracket(x, row["f"], lev, -1. if rule == "below" else 1.)
            e["share"] = float(np.isfinite(e["p"]))
        else:
            e["lo"], e["hi"], e["share"] = gap.interval(cb)
        e["start"] = bool(np.isfinite(e["p"]) and e["p"] <= x[0] * 1.0001)        # past from the first checkpoint
    row["ev"] = row["events"]
    return row


def _row(key, name, group, run, x, value, boot, a, a_boot, b, b_boot, a_lab, b_lab, events, sample, note="",
         unit=""):
    nb = gap.N_BOOT
    return timed(dict(key=key, name=name, group=group, run=run, x=np.asarray(x, np.float64),
                      value=np.asarray(value, np.float64), boot=None if boot is None else np.asarray(boot, np.float64),
                      a=float(a), b=float(b), a_boot=None if boot is None else np.broadcast_to(a_boot, (nb,)),
                      b_boot=None if boot is None else np.broadcast_to(b_boot, (nb,)), a_lab=a_lab, b_lab=b_lab,
                      events=[dict(e) for e in events], sample=sample, note=note, unit=unit,
                      paired=boot is not None))


def _refs(ref, key, a, a_boot, b, b_boot):
    """the Trained RNN's references for the ESN (its levels in absolute units, per resample)"""
    if ref is None:
        return a, a_boot, b, b_boot
    r = ref[key]
    return r["a"], r["a_boot"], r["b"], r["b_boot"]


def make_rows(run, ref=None):
    """every neural row of a run (ref: the Trained RNN's rows by key: the ESN is placed on the Trained RNN's scale,
    a and b in absolute units per resample, so a measure that barely moves is not stretched to the full scale)"""
    D = build_ck(run)
    x = gap.train_x(D["episode"])
    BR = bayes_refs()
    nb = gap.N_BOOT
    z0 = np.zeros(nb)
    rows = []
    S_PR = "probes: 50 environments x 200 episodes"
    S_CR = "credit probes: 5,000 episodes (50 environments, 85-109 each) x error steps 17-30"
    S_CF = "credit probes: the first 2,000 (50 environments, 27-53 each) x error steps 17-30"

    def add(key, name, group, value, boot, a, a_boot, b, b_boot, a_lab, b_lab, events, sample, note="", unit=""):
        a, a_boot, b, b_boot = _refs(ref, key, a, a_boot, b, b_boot)
        rows.append(_row(key, name, group, run, x, value, boot, a, a_boot, b, b_boot, a_lab, b_lab, events, sample,
                         note, unit))

    def own(key, name, group, value, boot, sample, note="", unit="", events=None):
        add(key, name, group, value, boot, value[0], None if boot is None else boot[:, 0], _end(value),
            None if boot is None else _end(boot), "the initial network", "the end value (last 3 checkpoints)",
            events or [_ev("10% of its change", "onset", .1), _ev("half its change", "half", .5)], sample, note, unit)

    # ── memory: cut at inference (per probe at T, both variables each scored against itself) ──
    for j, w in enumerate(TRUNC_W[:2]):
        y, yb = _ratio(D["ben"][:, :, j], np.full(gap.N_ENV, 2. * gap.PER_ENV))
        B, Bb = _const(BR["bben"][:, j], np.full(gap.N_ENV, 2. * gap.PER_ENV))
        add(f"ben{w}", f"memory benefit at T, cut to {w} step{'s' if w > 1 else ''}", "memory", y, yb, 0., z0, B, Bb,
            "0 (a memoryless observer: naive)", f"Bayes's own benefit at the same cut ({B:.3f})",
            [_ev("10% of Bayes's", "onset", .1), _ev("half Bayes's", "half", .5),
             _ev("passes Bayes's own benefit", "natural", 1.)], S_PR + " x 2 variables (accuracy at T)",
            unit="accuracy")
    y5, y5b = _ratio(D["ben"][:, :, 2], np.full(gap.N_ENV, 2. * gap.PER_ENV))
    B5, B5b = _const(BR["bben"][:, 2], np.full(gap.N_ENV, 2. * gap.PER_ENV))
    v, vb = y5 - B5, y5b - B5b[:, None]
    add("ben5x", "memory older than 5 steps: benefit minus Bayes's", "memory", v, vb, 0., z0, _peak(v),
        np.array([_peak(r) for r in vb]), "0 (Bayes's own benefit of memory older than 5 steps)",
        "its peak (3-checkpoint running mean)",
        [_ev("half its peak, first", "half", .5, "first"), _ev("back below half its peak for good", "fall", .5, "below"),
         _ev("back to Bayes's level for good", "fall", 0., "below")],
        S_PR + " x 2 variables (accuracy at T)",
        "rises and falls: timed on its rise (first reached) and on its fall", unit="accuracy")
    n2 = np.full(gap.N_ENV, 2. * gap.PER_ENV)
    y, yb = _ratio(D["acc2"] - D["ben"][:, :, 0], n2)
    Nv, Nb = _const(BR["bnaive2"], n2)
    add("cut1", "accuracy at T with memory cut to 1 step", "memory", y, yb, y[0], yb[:, 0], Nv, Nb,
        "the initial network", f"naive's accuracy ({Nv:.3f}; step-wise Bayes {_const(BR['bsw2'], n2)[0]:.3f})",
        [_ev("10% of the way to naive", "onset", .1), _ev("half-way to naive", "half", .5),
         _ev("reaches naive's accuracy for good", "natural", 1.)], S_PR + " x 2 variables (accuracy at T)",
        "the memoryless part of the network (its per-step increments, each from the start state)", unit="accuracy")
    if "mu_all" in D:
        n_all = np.full(gap.N_ENV, gap.PER_ENV * (T - 1) * NV)
        for key, num, cnt, bnum, lab in (("mu_all", D["mu_all"], n_all, D["bmu_all"], "all variable-steps 2-30"),
                                         ("mu_top", D["mu_top"], D["mu_top_cnt"], D["bmu_top"],
                                          "most entangled fifth")):
            y, yb = _ratio(num, cnt)
            B, Bb = _const(bnum, cnt)
            add(key, f"memory use in the step update, {lab}", "memory", y, yb, 0., z0, B, Bb,
                "0 (memory neither helps nor hurts the update)", f"Bayes's own memory use ({B:.4f} nats)",
                [_ev("memory helps (> 0) for good", "natural", 0.), _ev("half Bayes's", "half", .5),
                 _ev("passes Bayes's own", "natural", 1.)],
                S_PR + " x 29 steps x 2 variables" + ("" if key == "mu_all" else " (top fifth of each step)"),
                "update error (Jeffreys/2 to Bayes's belief) with the memory cut to the last step minus intact",
                unit="nats")

    # ── activity (per probe: mean over steps 2-30) ──
    n_pr = np.full(gap.N_ENV, float(gap.PER_ENV))
    for j, key, name, unit in ((0, "effdim", "state effective dimension (own basis)", "PCs"),
                               (1, "norm_u", "state size |u|", ""),
                               (3, "turn_u", "state turn between steps", "")):
        y, yb = _ratio(D["act"][:, :, j], n_pr)
        own(key, name, "activity", y, yb, S_PR + " x 29 steps", unit=unit)
    RP = collect_replay(run) if os.path.exists(os.path.join(OUT, f"replay_{run}.npz")) or \
        _replay_complete(run) else None
    if RP is not None:
        assert np.array_equal(RP["episode"], D["episode"])
        y, yb = _emean(RP["pr_env"][:, 1:].mean(1))
        own("pr_env", "state dimension per step within an environment (PR)", "activity", y, yb,
            S_PR + " x 29 steps (one cloud per environment and step)", unit="PR")
        y, yb = pr_pooled(RP)
        own("pr_pool", "state dimension per step (PR, environments pooled)", "activity", y, yb,
            S_PR + " x 29 steps (one cloud per step)", "resampled PRs are biased low by duplicated environments "
            "(more so later in training): each resampled curve is rescaled per checkpoint to the full-sample value "
            "(bias correction; uncorrected crossings in numbers())", unit="PR")

    # ── activity locked to entanglement (within-episode r, steps 17-30) ──
    for j, key, name, evs in (
            (1, "r_norm", "r(|u|, entanglement)", [_ev("turns positive for good", "natural", 0.),
                                                   _ev("10% of its end value", "onset", .1),
                                                   _ev("half its end value", "half", .5)]),
            (0, "r_effdim", "r(effective dimension, entanglement)", [_ev("10% of its end value", "onset", .1),
                                                                     _ev("half its end value", "half", .5)]),
            (3, "r_turn", "r(state turn, entanglement)", [_ev("recovers past half its end value", "half", .5)])):
        y, yb = _ratio(D["lock"][:, :, j], n_pr)
        add(key, name, "lock", y, yb, 0., z0, _end(y), _end(yb), "0 (no lock)", "its end value", evs,
            S_PR + " (r per episode, steps 17-30)",
            "input-driven at init (.31; the ESN keeps it): dips, then recovers" if key == "r_turn" else "")

    # ── learning signal (credit; per environment: median over its (probe, error step) items) ──
    if "cmed" in D:
        cnt, cnt_f = D["cnt_cr"], D["cnt_crf"]
        y, yb = _emean(D["cmed"][:, :, 0])
        add("coh", "credit coherence (median)", "credit", y, yb, y[0], yb[:, 0], 1., np.ones(nb),
            "the initial network", "1 (lag pieces neither add nor cancel)",
            [_ev("10% of the way to 1", "onset", .1), _ev("half-way to 1", "half", .5),
             _ev("below 1 for good (pieces cancel)", "natural", 1.)], S_CR)
        y, yb = _emean(D["cmed"][:, :, 1])
        own("dim", "update dimension (median)", "credit", y, yb, S_CR)
        y, yb = _ratio(D["clag"][:, 1, :, TAU0], cnt)
        add("r_dim", "r(update dimension, entanglement): decoupling", "credit", y, yb, y[0], yb[:, 0], 0., z0,
            "the initial network (novelty-driven)", "0 (no lock)",
            [_ev("10% of the way to 0", "onset", .1), _ev("half-way to 0", "half", .5),
             _ev("crosses 0 for good", "natural", 1.)], S_CR + " (r per episode)")
        add("r_dim_neg", "r(update dimension, entanglement): negative lock", "credit", y, yb, 0., z0, _end(y),
            _end(yb), "0 (no lock)", "its end value", [_ev("10% of its end value", "onset", .1),
                                                       _ev("half its end value", "half", .5)],
            S_CR + " (r per episode)")
        y, yb = _emean(D["cmed_f"])
        add("bandx", "cross-band share, final bands (median)", "credit", y, yb, 0., z0, _end(y), _end(yb),
            "0 (band pieces neither add nor cancel)", "its end value",
            ([_ev("below 0 for good (pieces cancel)", "natural", 0.)] if ref is None else [])
            + [_ev("10% of its end value", "onset", .1), _ev("half its end value", "half", .5)], S_CF,
            "the sign crossing is timed for the Trained RNN only (the ESN hovers at +-.003)")
        for key, name, num, cn in (("r_coh0", "r(coherence, entanglement), lag 0", D["clag"][:, 0], cnt),
                                   ("r_bx0", "r(cross-band share, entanglement), lag 0", D["clag_f"], cnt_f)):
            y, yb = _ratio(num[..., TAU0], cn)
            add(key, name, "credit", y, yb, 0., z0, _end(y), _end(yb), "0 (no lock)", "its end value",
                [_ev("10% of its end value", "onset", .1), _ev("half its end value", "half", .5)],
                (S_CR if cn is cnt else S_CF) + " (r per episode)")
        lag14 = (cl.TAU >= 1) & (cl.TAU <= 4)
        for key, name, num, cn in (("r_coh14", "r(coherence, entanglement), lags 1-4", D["clag"][:, 0], cnt),
                                   ("r_bx14", "r(cross-band share, entanglement), lags 1-4", D["clag_f"], cnt_f)):
            y, yb = _ratio(num[..., lag14].sum(-1), cn)
            add(key, name, "credit", y, yb, 0., z0, _peak(y), np.array([_peak(r) for r in yb]), "0 (no lock)",
                "its peak (3-checkpoint running mean)",
                [_ev("half its peak, first", "half", .5, "first"),
                 _ev("back below half its peak for good", "fall", .5, "below")],
                (S_CR if cn is cnt else S_CF) + " (r per episode, summed over lags 1-4)",
                "errors 1-4 steps after the entangled step; rises and falls")

    # ── weights (no probe noise: point times, the checkpoint bracket as resolution), except the tau counts ──
    for k, g in enumerate("ifgo"):
        y = D["rho"][:, k]
        add(f"rho_{g}", f"W_hh {g} block: spectral radius", "weights", y, None, y[0], None, 1., None,
            "the initial network (.59)", "1", [_ev("radius > 1 for good", "natural", 1.)], "weights")
    if RP is not None:
        fe = RP["f_env"].astype(np.float64)                                          # (K, 50, H)
        C = gap.boot_counts()
        for th in (5., 10.):
            full = (1 / (1 - fe.mean(1)) > th).sum(-1).astype(np.float64)
            boot = np.stack([(1 / (1 - (C @ fe[k]) / C.sum(1)[:, None]) > th).sum(-1) for k in range(len(fe))], 1)
            own(f"tau{th:.0f}", f"units with tau > {th:.0f}", "weights", full, boot.astype(np.float64),
                S_PR + " x 29 steps (mean forget gate per unit)", "tau = 1 / (1 - mean f); the first wave at 20-80 "
                "is a transient" + ("; its onset is not timed (the first wave stays above 10% of the end count)"
                                    if th == 5. else ""), unit="units",
                events=[_ev("half its change", "half", .5)] if th == 5. else None)
    else:
        for th in (5., 10.):
            y = (D["tau"] > th).sum(-1).astype(np.float64)
            own(f"tau{th:.0f}", f"units with tau > {th:.0f}", "weights", y, None, "probes pooled", unit="units",
                events=[_ev("half its change", "half", .5)] if th == 5. else None)
    sv = D["sv_dhh"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rk = np.nan_to_num((sv ** 2).sum(1) ** 2 / (sv ** 4).sum(1))
    own("rank_dhh", "effective rank of W_hh - W_hh(0)", "weights", rk, None, "weights",
        "top 64 singular values: a lower bound late", unit="")
    names = list(D["w_names"])
    for nm, lab in (("LSTM.weight_hh_l0", "W_hh"), ("LSTM.weight_ih_l0", "W_ih"),
                    ("classifier_readin.weight", "read-in"), ("classifier_readout.weight", "read-out")):
        y = D["w_dinit"][:, names.index(nm)]
        own(f"dw_{lab}", f"||W - W(0)||, {lab}", "weights", y, None, "weights", unit="")
    return rows


def pr_pooled(RP, correct=True):
    """the per-step PR of u pooled over environments, averaged over steps 2-30: (full (K,), resampled (N_BOOT, K)).
    A resample's PR is biased low (an environment drawn twice adds no orientation diversity); correct=True rescales
    every resampled curve per checkpoint by full / mean over resamples (a multiplicative bias correction)"""
    y = RP["pr_full"][:, 1:].mean(1)
    yb = RP["pr_boot"][:, 1:].mean(1).T                                                  # (N_BOOT, K)
    if correct:
        yb = yb * (y / yb.mean(0))[None]
    return y, yb


def _replay_complete(run):
    d = os.path.join(OUT, f"replay_{run}")
    return os.path.isdir(d) and len(glob.glob(os.path.join(d, "rp_?????.npz"))) == len(ck_files(run))


def all_rows():
    """{run: [rows]} with the ESN on the Trained RNN's scale"""
    tr = make_rows("trained")
    es = make_rows("echo", ref={r["key"]: r for r in tr})
    return {"trained": tr, "echo": es}


# ═══════════════════════════════════════════════════════════════════
# Order tests against the behavioural timeline
# ═══════════════════════════════════════════════════════════════════

FAMS = ("answers", "splitting", "holding", "coupling")
BLEV = {"c10": .1, "c50": .5}


def behaviour():
    """the behavioural timeline's tested rows (td_tt_timeline: rise-shaped, paired, answers / splitting / holding /
    coupling) with their c10 and c50 crossings (p, lo, hi, share, boot) on the shared resamples"""
    import td_tt_timeline as tl
    rows = tl.annotate(tl.load_rows())
    B = dict(tl.blocks(rows))
    out = []
    for r, e in B["tested"]:
        out.append(dict(name=r["name"], family=r["family"], topic=r["topic"], x0=float(r["x"][0]),
                        c10=r["L"][.1], c50=r["L"][.5], x=r["x"], f_trained=tl.fracs(r)[0],
                        f_echo=None if e is None else tl.fracs(e)[0]))
    return out


def _inf(v):
    v = np.asarray(v, np.float64)
    return np.where(np.isnan(v), np.inf, v)


def verdict(e, c):
    """'before' / 'after' / 'with' by the interval rule (an open upper bound is +inf); '-' if the neural event never
    happens"""
    if not np.isfinite(e["p"]):
        return "-"
    ehi = e["hi"] if np.isfinite(e["hi"]) else np.inf
    chi = c["hi"] if np.isfinite(c["hi"]) else np.inf
    return "before" if ehi < c["lo"] else "after" if e["lo"] > chi else "with"


def p_before(e, c):
    """share of resamples in which the neural event precedes the behavioural crossing: paired (both on the same
    resamples) for probe-based rows; for weight rows the point time against the behavioural resamples (unpaired)"""
    cb = _inf(c["boot"])
    if e["boot"] is None:
        return float((e["p"] < cb).mean()) if np.isfinite(e["p"]) else 0.
    return float((_inf(e["boot"]) < cb).mean())


def at_start(c, x0=gap.X0):
    """a behavioural crossing that cannot be preceded: past from the first checkpoint in the full sample or in at least
    5% of the resamples (its 5% bound is the first checkpoint: not established as a training event)"""
    return bool((np.isfinite(c["p"]) and c["p"] <= x0 * 1.0001) or (np.isfinite(c["lo"]) and c["lo"] <= x0 * 1.0001))


def orders(rows, beh):
    """per Trained RNN event: verdict and P(neural < behaviour) against every tested behaviour row at c10 and c50, and
    per family: the joint shares P(neural < every row) and P(neural > every row) over the paired resamples (rows past
    from the start excluded) and the family verdict (before every row / after every row / overlaps)"""
    out = []
    for r in rows:
        for e in r["ev"]:
            d = dict(row=r, ev=e, per={}, fam={})
            for lev in BLEV:
                for b in beh:
                    c = b[lev]
                    d["per"][(b["name"], lev)] = (verdict(e, c), p_before(e, c), at_start(c))
                for fam in FAMS:
                    cs = [b[lev] for b in beh if b["family"] == fam and not at_start(b[lev])]
                    if not cs:
                        continue
                    vs = [verdict(e, c) for c in cs]
                    cmin = np.min([_inf(c["boot"]) for c in cs], 0)
                    cmax = np.max([_inf(c["boot"]) for c in cs], 0)
                    eb = _inf(e["boot"]) if e["boot"] is not None else np.full(len(cmin), e["p"] if np.isfinite(
                        e["p"]) else np.inf)
                    fv = "-" if vs[0] == "-" else "before" if all(v == "before" for v in vs) else \
                        "after" if all(v == "after" for v in vs) else "with"
                    d["fam"][(fam, lev)] = dict(verdict=fv, n_before=vs.count("before"), n_with=vs.count("with"),
                                                n_after=vs.count("after"), n=len(vs),
                                                p_all_before=float((eb < cmin).mean()),
                                                p_all_after=float((eb > cmax).mean()))
            out.append(d)
    return out


def accuracy_row(run):
    """goal accuracy at T from the stored correctness (ck "right"), gap fraction between naive and Bayes (both per
    environment and resampled): c0 = passes naive for good, c50 = half-way (the timeline's 'goal accuracy at T')"""
    D = build_ck(run)
    BR = bayes_refs()
    x = gap.train_x(D["episode"])
    n = np.full(gap.N_ENV, float(gap.PER_ENV))
    y, yb = _ratio(D["acc"], n)
    a, ab = _const(BR["nacc"], n)
    b, bb = _const(BR["bacc"], n)
    return _row("acc", "goal accuracy at T", "behaviour", run, x, y, yb, a, ab, b, bb, "naive", "Bayes",
                [_ev("passes naive for good", "natural", 0.), _ev("10% of the gap", "onset", .1),
                 _ev("half-way to Bayes", "half", .5)], "probes: 50 environments x 200 episodes")


CROSS_RUNS = ("trained", "trained_p1", "trained_N200", "trained_N100", "trained_N50")
CROSS_EVENTS = (("ben1", "natural"), ("ben2", "natural"), ("cut1", "half"), ("turn_u", "onset"), ("turn_u", "half"),
                ("effdim", "onset"), ("effdim", "half"), ("dim", "onset"), ("dim", "half"), ("r_dim", "half"),
                ("r_dim", "natural"), ("coh", "onset"), ("coh", "half"), ("coh", "natural"), ("r_coh14", "half"),
                ("rho_o", "natural"), ("rho_i", "natural"), ("rho_f", "natural"), ("rank_dhh", "half"),
                ("tau10", "onset"), ("tau5", "half"), ("bandx", "natural"), ("r_coh0", "half"))


def cross_runs():
    """the same neural rows on the slower trained runs (loss power 1, widths 200 / 100 / 50; each on its own scale and
    its own resamples of the same environments), against each run's accuracy passing naive (c0) and half-way (c50): a
    precursor must keep its order when the behaviour moves"""
    out = {}
    for run in CROSS_RUNS:
        rows = {r["key"]: r for r in make_rows(run)}
        acc = accuracy_row(run)
        ev = {}
        for key, role in CROSS_EVENTS:
            if key not in rows:
                continue
            e = next((q for q in rows[key]["ev"] if q["role"] == role), None)
            if e is None:
                continue
            ev[(key, role)] = dict(e=e, name=rows[key]["name"], vs={lab: (verdict(e, c), p_before(e, c)) for lab, c in
                                                                     (("c0", acc["ev"][0]), ("c50", acc["ev"][2]))})
        D = build_ck(run)
        x = gap.train_x(D["episode"])
        n2 = 2. * gap.N_ENV * gap.PER_ENV
        intact, cut = D["acc2"].sum(1) / n2, (D["acc2"] - D["ben"][:, :, 0]).sum(1) / n2
        e = rows["ben1"]["ev"][2]
        k = int(np.argmin(np.abs(np.log(x) - np.log(e["p"])))) if np.isfinite(e["p"]) else None
        out[run] = dict(acc=acc, ev=ev, at_ben=None if k is None else (x[k], intact[k], cut[k]),
                        cut_end=cut[-3:].mean(), intact_end=intact[-3:].mean())
    return out


def family_ranges(beh):
    """per family: (min, max) of the c10 and c50 point crossings (crossings past from the start left out)"""
    out = {}
    for fam in FAMS:
        for lev in BLEV:
            ps = [b[lev]["p"] for b in beh if b["family"] == fam and not at_start(b[lev]) and np.isfinite(b[lev]["p"])]
            if ps:
                out[(fam, lev)] = (min(ps), max(ps))
    return out


# ═══════════════════════════════════════════════════════════════════
# Export, numbers
# ═══════════════════════════════════════════════════════════════════

def synthesis(R=None, beh=None, O=None):
    """-> through_training/neural/synthesis.npz: rows (both runs; the ESN on the Trained RNN's scale) with their
    curves, references and resampled curves, every event with its resampled crossings, and the Trained RNN's order
    tests against the behavioural timeline's tested rows (see the module docstring)"""
    R = R or all_rows()
    beh = beh or behaviour()
    rows = R["trained"] + R["echo"]
    O = O or orders(R["trained"], beh)
    nb, K = gap.N_BOOT, len(rows[0]["x"])
    out = dict(x=rows[0]["x"], n_boot=nb, seed=gap.BOOT_SEED)
    for k in ("key", "name", "group", "run", "sample", "note", "unit", "a_lab", "b_lab"):
        out[k] = np.array([r[k] for r in rows])
    out["paired"] = np.array([r["paired"] for r in rows])
    for k in ("a", "b"):
        out[k] = np.array([r[k] for r in rows])
        out[k + "_boot"] = np.array([np.full(nb, np.nan) if r[k + "_boot"] is None else r[k + "_boot"] for r in rows])
    out["value"] = np.array([r["value"] for r in rows])
    out["f"] = np.array([r["f"] for r in rows])
    out["boot"] = np.array([np.full((nb, K), np.nan) if r["boot"] is None else r["boot"] for r in rows])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["lo"], out["hi"] = np.nanpercentile(out["boot"], [5, 95], 1)
    E = [(i, e) for i, r in enumerate(rows) for e in r["ev"]]
    out["ev_row"] = np.array([i for i, _ in E])
    for k in ("lab", "role", "rule"):
        out["ev_" + k] = np.array([e[k] for _, e in E])
    for k in ("level", "p", "lo", "hi", "share"):
        out["ev_" + k] = np.array([e[k] for _, e in E], np.float64)
    out["ev_start"] = np.array([e["start"] for _, e in E])
    out["ev_boot"] = np.array([np.full(nb, np.nan) if e["boot"] is None else e["boot"] for _, e in E])
    out["beh_name"] = np.array([b["name"] for b in beh])
    out["beh_family"] = np.array([b["family"] for b in beh])
    code = {"before": -1, "with": 0, "after": 1, "-": 9}
    levs = list(BLEV)
    out["ord_levels"] = np.array(levs)
    out["ord_verdict"] = np.array([[[code[d["per"][(b["name"], lev)][0]] for lev in levs] for b in beh] for d in O])
    out["ord_p"] = np.array([[[d["per"][(b["name"], lev)][1] for lev in levs] for b in beh] for d in O])
    out["ord_unprecedable"] = np.array([[b[lev] is not None and at_start(b[lev]) for lev in levs] for b in beh])
    out["fam"] = np.array(FAMS)
    out["fam_verdict"] = np.array([[[code[d["fam"][(f, lev)]["verdict"]] if (f, lev) in d["fam"] else 9
                                     for lev in levs] for f in FAMS] for d in O])
    for k in ("p_all_before", "p_all_after"):
        out["fam_" + k] = np.array([[[d["fam"][(f, lev)][k] if (f, lev) in d["fam"] else np.nan for lev in levs]
                                     for f in FAMS] for d in O])
    out["ord_ev"] = np.array([next(j for j, (i, e) in enumerate(E) if e is d["ev"]) for d in O])
    path = os.path.join(OUT, "synthesis.npz")
    np.savez(path, **out)
    print("saved", path)
    return path


def _fmt(e):
    if not np.isfinite(e["p"]):
        return f"never ({e['share']:.0%} of resamples cross)" if e["boot"] is not None else "never"
    hi = f"{e['hi']:,.0f}" if np.isfinite(e["hi"]) else "never"
    s = f"{e['p']:,.0f} [{e['lo']:,.0f}-{hi}]"
    if e["boot"] is None:
        return s + " (checkpoints)"
    return s + ("" if e["share"] >= .995 else f" ({e['share']:.0%} cross)")


def check_pr_formula(seed=0):
    """pr_resampled against brute force (rows duplicated by the resample counts) on random data: max relative |diff|"""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(gap.N_ENV, 4, 6)) + rng.normal(size=(gap.N_ENV, 1, 6))
    Cb = rng.multinomial(gap.N_ENV, np.full(gap.N_ENV, 1 / gap.N_ENV), size=5).astype(np.float64)
    s = U.sum(1)
    S = np.einsum("eih,eij->ehj", U, U)
    pr, _ = pr_resampled(np.einsum("ehh->e", S), s @ s.T, np.einsum("ehj,fhj->ef", S, S),
                         np.einsum("fh,ehj,gj->efg", s, S, s), np.full(gap.N_ENV, 4.), Cb)
    err = 0.
    for b in range(len(Cb)):
        X = np.repeat(U.reshape(-1, 6), np.repeat(Cb[b].astype(int), 4), axis=0)
        Cv = np.cov(X.T, bias=True)
        err = max(err, abs(pr[b] - np.trace(Cv) ** 2 / np.trace(Cv @ Cv)) / pr[b])
    return err


def checks(R):
    """consistency checks printed by numbers()"""
    out = [f"pooled-PR resampling formula vs brute force (random data, 5 resamples): max relative |diff| "
           f"{check_pr_formula():.1e}"]
    # (1) coherence lag products = td_tt_credit_lags' cache
    if os.path.exists(os.path.join(cl.OUT, "credit_lags.npz")):
        L = np.load(os.path.join(cl.OUT, "credit_lags.npz"))
        for run in RUNS:
            D = build_ck(run)
            out.append(f"coherence lag products (environment sums) vs td_tt_credit_lags' cache, {NAME[run]}: max |diff| "
                       f"{np.abs(D['clag'][:, 0] - L[f'{run}_sum']).max():.1e}")
    # (2) accuracy from the stored correctness = the timeline's 'goal accuracy at T' (td_tt_jumps export)
    J = gap.load("jumps")
    i = [k for k in range(len(J["names"])) if J["names"][k] == "goal accuracy at T" and J["run"][k] == "trained"][0]
    acc = accuracy_row("trained")
    same = np.mean(np.isclose(_inf(acc["ev"][2]["boot"]), _inf(J["c50_boot"][i])))
    out.append(f"goal accuracy at T from ck 'right' vs the jumps export: max |diff| "
               f"{np.abs(acc['value'] - J['value'][i]).max():.1e}; c0 {acc['ev'][0]['p']:,.0f} vs {J['c0'][i]:,.0f}, "
               f"c50 {acc['ev'][2]['p']:,.0f} vs {J['c50'][i]:,.0f}; resampled c50 identical in {same:.1%}")
    # (3) the replay against the stored forward pass (every checkpoint)
    for run in RUNS:
        if not os.path.exists(os.path.join(OUT, f"replay_{run}.npz")):
            continue
        RP = collect_replay(run)
        pr_ck, tau_ck = [], []
        for f in ck_files(run):
            z = np.load(f)
            pr_ck.append(z["fac"][:, 6, 0, 1])
            tau_ck.append(z["tau"])
        pr_ck, tau_ck = np.array(pr_ck), np.array(tau_ck)
        tau_rp = 1 / (1 - RP["f_env"].astype(np.float64).mean(1))
        rel = np.nanmax(np.abs(RP["pr_full"][:, 1:] - pr_ck[:, 1:]) / pr_ck[:, 1:])
        out.append(f"replay vs ck, {NAME[run]} (114 checkpoints): per-step PR max relative |diff| {rel:.1e}; tau max |diff| "
                   f"{np.abs(tau_rp - tau_ck).max():.1e}; units with tau > 5 / > 10 equal at "
                   f"{np.mean((tau_rp > 5).sum(1) == (tau_ck > 5).sum(1)):.0%} / "
                   f"{np.mean((tau_rp > 10).sum(1) == (tau_ck > 10).sum(1)):.0%} of checkpoints")
        if run == "trained":
            y, yb = pr_pooled(RP, correct=False)
            x = gap.train_x(RP["episode"])
            ks = [0, 10, 30, 45, 60, 113]
            out.append("resampled pooled PR / full-sample PR (duplicated environments lower the PR): " + ", ".join(
                f"{x[k]:,.0f}: {yb[:, k].mean() / y[k]:.3f}" for k in ks))
            r = _row("p", "", "activity", run, x, y, yb, y[0], yb[:, 0], _end(y), _end(yb), "", "",
                     [_ev("", "onset", .1), _ev("", "half", .5)], "")
            rc = next(q for q in R["trained"] if q["key"] == "pr_pool")
            out.append("pooled PR crossings without the bias correction: " + ", ".join(
                f"{e['role']} {_fmt(e)}" for e in r["ev"]) + "; with it: " + ", ".join(
                f"{e['role']} {_fmt(e)}" for e in rc["ev"]))
    # (4) the ESN starts where the Trained RNN starts (identical initial networks)
    tr = {r["key"]: r for r in R["trained"]}
    d0 = max(abs(r["value"][0] - tr[r["key"]]["value"][0]) for r in R["echo"] if not r["key"].startswith("r_bx")
             and r["key"] != "bandx")
    out.append(f"ESN and Trained RNN initial values (identical initial networks): max |diff| over rows {d0:.1e} "
               "(the cross-band rows differ at init by design: each network's FINAL checkpoint's bands)")
    # (5) goal-only memory benefit
    D = build_ck("trained")
    BR = bayes_refs()
    x = gap.train_x(D["episode"])
    for j, w in enumerate(TRUNC_W[:2]):
        y, yb = _ratio(D["ben_goal"][:, :, j], np.full(gap.N_ENV, float(gap.PER_ENV)))
        B, Bb = _const(BR["bben_goal"][:, j], np.full(gap.N_ENV, float(gap.PER_ENV)))
        r = _row("g", "", "memory", "trained", x, y, yb, 0, np.zeros(gap.N_BOOT), B, Bb, "", "",
                 [_ev("", "half", .5), _ev("", "natural", 1.)], "")
        both = tr[f"ben{w}"]["ev"]
        out.append(f"memory benefit w = {w}, goal variable only (Bayes {B:.3f}): half {_fmt(r['ev'][0])}, passes Bayes's "
                   f"{_fmt(r['ev'][1])} (both variables: {_fmt(both[1])}, {_fmt(both[2])})")
    # (5b) what the memory benefit means: intact and memory-cut accuracies (goal variable) around the crossings
    acc_i = D["acc"].sum(1) / (gap.N_ENV * gap.PER_ENV)
    cut = acc_i[:, None] - D["ben_goal"].sum(1) / (gap.N_ENV * gap.PER_ENV)                  # (K, 3) w = 1, 2, 5
    P = probes()
    ar = np.arange(len(P["goal"]))
    refs = {k: P[k][ar, P["goal"]].mean() for k in ("right_naive", "right_window1", "right_window2", "right_bayes")}
    line = []
    for xv in (300, 554, 835, 1394, 4000, x[-1]):
        k = int(np.argmin(np.abs(np.log(x) - np.log(xv))))
        line.append(f"{x[k]:,.0f}: intact {acc_i[k]:.3f}, cut to 1 {cut[k, 0]:.3f}, cut to 2 {cut[k, 1]:.3f}")
    out.append("goal accuracy at T, intact vs memory cut (naive " + f"{refs['right_naive']:.3f}, step-wise Bayes "
               f"{refs['right_window1']:.3f}, Bayes cut to 2 {refs['right_window2']:.3f}, Bayes "
               f"{refs['right_bayes']:.3f}): " + "; ".join(line))
    # (5c) the long-time-constant counts at a few checkpoints (the first wave and the second)
    cnt5 = (D["tau"] > 5).sum(-1)
    cnt10 = (D["tau"] > 10).sum(-1)
    out.append("units with tau > 5 / > 10 (ck tau): " + ", ".join(
        f"{x[k]:,.0f}: {cnt5[k]} / {cnt10[k]}" for k in [int(np.argmin(np.abs(x - v))) for v in
                                                         (20, 80, 320, 960, 2000, 4000, 8000, x[-1])]))
    # (6) cross-band r on the checkpoint's own bands (5,000 credit probes) instead of the final bands
    for lag, sl in (("lag 0", cl.TAU == 0), ("lags 1-4", (cl.TAU >= 1) & (cl.TAU <= 4))):
        y, yb = _ratio(D["clag"][:, 2][..., sl].sum(-1), D["cnt_cr"])
        if lag == "lag 0":
            r = _row("o", "", "credit", "trained", x, y, yb, 0., np.zeros(gap.N_BOOT), _end(y), _end(yb), "", "",
                     [_ev("", "onset", .1), _ev("", "half", .5)], "")
        else:
            r = _row("o", "", "credit", "trained", x, y, yb, 0., np.zeros(gap.N_BOOT), _peak(y),
                     np.array([_peak(q) for q in yb]), "", "", [_ev("", "half", .5, "first"),
                                                                 _ev("", "fall", .5, "below")], "")
        out.append(f"r(cross-band share, entanglement) {lag} on own bands: " + ", ".join(
            f"{e['role']} {_fmt(e)}" for e in r["ev"]))
    return out


MU_X = (10, 20, 40, 80, 100, 160, 180, 200, 220, 240, 320, 800, 1200, 1600, 4000)


def _at(x, v):
    """index of the checkpoint nearest to v training episodes (log scale)"""
    return int(np.argmin(np.abs(np.log(x) - np.log(v))))


def _sign_runs(x, lo, hi):
    """runs of consecutive checkpoints whose 5-95% band is above 0 (+), below 0 (-) or includes 0 (0)"""
    s = np.where(lo > 0, "+", np.where(hi < 0, "-", "0"))
    runs, k0 = [], 0
    for k in range(1, len(s) + 1):
        if k == len(s) or s[k] != s[k0]:
            runs.append(f"{x[k0]:,.0f}" + ("" if k - 1 == k0 else f"-{x[k - 1]:,.0f}") + f" {s[k0]}")
            k0 = k
    return runs


def memory_parts(R):
    """memory use in the step update split into its two terms (the intact and the memory-cut update error), its two
    memoryless references (Bayes's own = step-wise Bayes's error; naive's error), the sign of its 5-95% band per
    checkpoint, and its crossings with naive's reference as b: lines for numbers()"""
    out = []
    for run in RUNS:
        D = build_ck(run)
        M = memory_errors(run)
        assert np.array_equal(M["episode"], D["episode"])
        x = gap.train_x(D["episode"])
        rows = {r["key"]: r for r in R[run]}
        for part, cnt in (("all", np.full(gap.N_ENV, gap.PER_ENV * (T - 1) * NV)), ("top", D["mu_top_cnt"])):
            mu = M["ec_" + part] - M["ei_" + part]
            yi = _ratio(M["ei_" + part], cnt)[0]
            yc = _ratio(M["ec_" + part], cnt)[0]
            y, yb = _ratio(mu, cnt)
            lo, hi = np.percentile(yb, [5, 95], 0)
            bB, bBb = _const(D["bmu_" + part], cnt)
            bN, bNb = _const(M["nmu_" + part], cnt)
            lab = "all variable-steps 2-30" if part == "all" else "most entangled fifth"
            out.append(f"{NAME[run]}, {lab}: (cut - intact) vs the stored memory use: max |diff| "
                       f"{np.abs(mu - D['mu_' + part]).max():.1e}; memoryless references: step-wise Bayes's error "
                       f"(Bayes's own memory use, b) {bB:.4f} [{np.percentile(bBb, 5):.4f}-{np.percentile(bBb, 95):.4f}],"
                       f" naive's {bN:.4f} [{np.percentile(bNb, 5):.4f}-{np.percentile(bNb, 95):.4f}] nats")
            cells = [f"{x[k]:,.0f}: {yi[k]:.4f} / {yc[k]:.4f} / {y[k]:+.4f} [{lo[k]:+.4f}, {hi[k]:+.4f}]"
                     for k in sorted({_at(x, v) for v in MU_X})]
            cells.append(f"end (last 3): {_end(yi):.4f} / {_end(yc):.4f} / {_end(y):+.4f}")
            out.append("   intact error / cut error / memory use [5-95%] (nats): " + "; ".join(cells))
            out.append("   sign of the 5-95% band per checkpoint: " + ", ".join(_sign_runs(x, lo, hi)))
            km = int(np.argmax(y))
            out.append(f"   largest memory use {y[km]:+.4f} at {x[km]:,.0f} ({y[km] / bB:.2f} of Bayes's own, "
                       f"{y[km] / bN:.2f} of naive's reference); end {_end(y):+.4f} ({_end(y) / bB:.2f} / "
                       f"{_end(y) / bN:.2f}); cut error at the end {_end(yc):.4f}: "
                       f"{'between the two references' if bB < _end(yc) < bN else 'outside the references'}")
            if run == "trained":
                ev = rows["mu_" + part]["ev"][0]                                          # > 0 for good
                kc = int(np.where(y <= 0)[0][-1])                                          # last checkpoint not past 0
                k20 = _at(x, 20)
                out.append(f"   memory use {y[0]:+.4f} at the untrained network; negative at every checkpoint from "
                           f"{x[k20]:,.0f} to {x[kc]:,.0f}: {bool(np.all(y[k20:kc + 1] < 0))}; > 0 for good from "
                           f"{_fmt(ev)}")
                for k1, k2 in ((_at(x, 320), kc + 1), (_at(x, 320), _at(x, 4000))):
                    out.append(f"   additive split, {x[k1]:,.0f} -> {x[k2]:,.0f}: memory use {y[k2] - y[k1]:+.4f} = "
                               f"intact error's fall {yi[k1] - yi[k2]:+.4f} + cut error's change "
                               f"{yc[k2] - yc[k1]:+.4f}")
                r = _row("n", "", "memory", run, x, y, yb, 0., np.zeros(gap.N_BOOT), bN, bNb, "", "",
                         [_ev("half naive's", "half", .5), _ev("passes naive's", "natural", 1.)], "")
                rb = rows["mu_" + part]["ev"]
                out.append(f"   with naive's memoryless error as b instead of Bayes's: half {_fmt(r['ev'][0])}, passes "
                           f"{_fmt(r['ev'][1])} (with Bayes's: half {_fmt(rb[1])}, passes {_fmt(rb[2])})")
            else:
                k10, k20 = _at(x, 10), _at(x, 20)
                out.append(f"   at {x[k20]:,.0f}: memory use {y[k20]:+.4f} [{lo[k20]:+.4f}, {hi[k20]:+.4f}] = "
                           f"{y[k20] / bB:.2f} of Bayes's own; intact error {yi[k10]:.4f} -> {yi[k20]:.4f}, cut error "
                           f"{yc[k10]:.4f} -> {yc[k20]:.4f} ({x[k10]:,.0f} -> {x[k20]:,.0f})")
    return out


def numbers():
    R = all_rows()
    beh = behaviour()
    O = orders(R["trained"], beh)
    print("== checks")
    for line in checks(R):
        print("  " + line)
    print("== memory use in the step update (cut minus intact update error, nats): its two terms, its references, the "
          "sign of its band")
    for line in memory_parts(R):
        print("  " + line)
    tr, es = {r["key"]: r for r in R["trained"]}, {r["key"]: r for r in R["echo"]}
    endf = {k: float(_end(es[k]["f"])) for k in TITLE_KEYS if k in es}

    def _endf(keys):
        return ", ".join(f"{SHORT.get(k, k)} {float(_end(es[k]['f'])):+.2f}" for k in keys if k in es)
    etr = [_transient(es[k]) for k in TITLE_MU if k in es]
    print("== ESN end f on the Trained RNN's scale, the changes the main title names. The state, credit and "
          "time-constant changes: " + _endf(TITLE_KEYS) + f"; largest {max(endf.values()):+.2f}. Memory: "
          + _endf(TITLE_BEN) + "; " + _endf(TITLE_MU) + " (5-95% band above 0 "
          + " / ".join(_helps(t) for t in etr) + " episodes, below 0 from " + " / ".join(f"{t[2]:,.0f}" for t in etr)
          + f" to the end). The ESN's r(state turn, entanglement): init {es['r_turn']['value'][0]:+.3f} = the Trained "
          f"RNN's {tr['r_turn']['value'][0]:+.3f}, end {_end(es['r_turn']['value']):+.3f} "
          f"(f {_end(es['r_turn']['f']):+.2f})")
    for run in RUNS:
        print(f"== {NAME[run]}: rows and events" + (" (on the Trained RNN's scale: a and b are the Trained RNN's, in "
                                                       "absolute units, per resample)" if run == "echo" else ""))
        for r in R[run]:
            print(f"  [{r['group']}] {r['name']}  ({r['a_lab']} -> {r['b_lab']}; a {r['a']:+.4g}, b {r['b']:+.4g}; "
                  f"start {r['value'][0]:+.4g}, end {_end(r['value']):+.4g}, end f {_end(r['f']):+.2f}"
                  f"{'' if r['paired'] else '; weights: no resamples'})")
            for e in r["ev"]:
                print(f"      {e['role']:7s} {e['lab']:42s} {_fmt(e)}{'  (past from the start)' if e['start'] else ''}")
    print("== order tests, Trained RNN (families: interval rule against every row; P = paired share over the 1,000 "
          "resamples that the event precedes every row / follows every row; weight rows: the point time against the "
          "behavioural resamples, unpaired)")
    for fam in FAMS:
        for lev in BLEV:
            cs = [b for b in beh if b["family"] == fam]
            ex = [b["name"] for b in cs if at_start(b[lev])]
            keep = [b for b in cs if not at_start(b[lev])]
            if not keep:
                print(f"  -- {fam} {lev}: every row is past from the start in >= 5% of resamples ({', '.join(ex)}): "
                      f"no order test")
                continue
            ps = sorted(b[lev]["p"] for b in keep)
            p6 = sorted(b[lev]["p"] for b in keep if not b["name"].startswith("six-step"))
            print(f"  -- {fam} {lev}: {len(keep)} rows, {ps[0]:,.0f}-{ps[-1]:,.0f}" + (f" (left out, past from the "
                                                                                         f"start: {', '.join(ex)})" if ex else "")
                  + (f" (without the six-step move toward own cells, which td_tt_splitting flags as no clean timing "
                     f"marker: {p6[0]:,.0f}-{p6[-1]:,.0f})" if len(p6) < len(ps) else ""))
            for v in ("before", "with", "after"):
                items = [d for d in O if d["fam"].get((fam, lev), {}).get("verdict") == v]
                items.sort(key=lambda d: d["ev"]["p"])
                print(f"     {v.upper()} every row ({len(items)}):" if v != "with" else
                      f"     OVERLAPS at least one row ({len(items)}):")
                for d in items:
                    f_ = d["fam"][(fam, lev)]
                    print(f"        {d['row']['name'][:52]:52s} {d['ev']['lab'][:34]:34s} {_fmt(d['ev']):30s} "
                          f"P(before all) {f_['p_all_before']:.3f}  P(after all) {f_['p_all_after']:.3f}  "
                          f"[{f_['n_before']}/{f_['n_with']}/{f_['n_after']} of {f_['n']}]")
            nev = [d for d in O if not np.isfinite(d["ev"]["p"])]
            if nev and fam == FAMS[0] and lev == "c10":
                print("     (never: " + "; ".join(f"{d['row']['name']}: {d['ev']['lab']}" for d in nev) + ")")
    print("== SUMMARY per family (Trained RNN events drawn in the figures: >= 95% of resamples cross; 'phase 0' = "
          "before 100 training episodes)")
    for fam, lev in TESTS:
        keep = [b for b in beh if b["family"] == fam and not at_start(b[lev])]
        ps = sorted(b[lev]["p"] for b in keep)
        sel = [d for d in O if shown(d["ev"]) and (fam, lev) in d["fam"]]
        sel.sort(key=lambda d: d["ev"]["p"])

        def lst(v, early=None):
            out = []
            for d in sel:
                f_ = d["fam"][(fam, lev)]
                if f_["verdict"] != v or (early is not None and (d["ev"]["p"] < 100) != early):
                    continue
                out.append(f"{SHORT.get(d['row']['key'], d['row']['name'])}: {d['ev']['lab']} {d['ev']['p']:,.0f}"
                           f"{'*' if d['ev']['boot'] is None else ''} (P {f_['p_all_before']:.2f})")
            return out
        b0, b1 = lst("before", True), lst("before", False)
        print(f"  -- {fam} {lev} ({len(keep)} rows, {ps[0]:,.0f}-{ps[-1]:,.0f}):")
        print(f"     before every row, phase 0 ({len(b0)}): " + "; ".join(b0))
        print(f"     before every row, later ({len(b1)}): " + "; ".join(b1))
        w_ = lst("with")
        print(f"     overlaps ({len(w_)}): " + "; ".join(w_))
        a_ = lst("after")
        print(f"     after every row: {len(a_)} events, the first: " + "; ".join(a_[:3]))
        bef = [d for d in sel if d["fam"][(fam, lev)]["verdict"] == "before"]
        if bef:
            t_last = max(d["ev"]["p"] for d in bef)
            exc = [d for d in sel if d["ev"]["p"] <= t_last and d["fam"][(fam, lev)]["verdict"] != "before"]
            print(f"     every drawn event up to {t_last:,.0f} precedes every row" + ("" if not exc else ", except: " +
                  "; ".join(f"{SHORT.get(d['row']['key'], d['row']['name'])}: {d['ev']['lab']} {_fmt(d['ev'])} "
                            f"({d['fam'][(fam, lev)]['verdict']}, P(before all) {d['fam'][(fam, lev)]['p_all_before']:.3f}"
                            f", {d['fam'][(fam, lev)]['n_before']}/{d['fam'][(fam, lev)]['n_with']}/"
                            f"{d['fam'][(fam, lev)]['n_after']} of {d['fam'][(fam, lev)]['n']})" for d in exc)))
    print("== every Trained RNN event x every behavioural row: interval rule ('<' before, '~' overlaps, '>' after, "
          "'.' the behavioural crossing is past from the start, '-' the event never happens), then the paired share "
          "P(neural < behaviour) in tenths (0-9, '*' = 1.0); columns: behavioural rows by family (c10 | c50)")
    cols = [b for fam in FAMS for b in beh if b["family"] == fam]
    for k, b in enumerate(cols):
        print(f"     col {k + 1:2d}: {b['family']:9s} {b['name']}")
    sym = {"before": "<", "with": "~", "after": ">", "-": "-"}
    for d in sorted([d for d in O if np.isfinite(d["ev"]["p"])], key=lambda d: d["ev"]["p"]):
        line = []
        for lev in BLEV:
            v = "".join("." if d["per"][(b["name"], lev)][2] else sym[d["per"][(b["name"], lev)][0]] for b in cols)
            pp = "".join("*" if d["per"][(b["name"], lev)][1] >= .995 else str(int(d["per"][(b["name"], lev)][1] * 10))
                         for b in cols)
            line.append(f"{v} {pp}")
        print(f"  {d['ev']['p']:7,.0f} {(SHORT.get(d['row']['key'], d['row']['name']) + ': ' + d['ev']['lab'])[:62]:62s}"
              f"{'*' if d['ev']['boot'] is None else ' '} c10 {line[0]} | c50 {line[1]}")
    print("== per behavioural row (Trained RNN events in time order; verdict / paired P(neural first))")
    for b in beh:
        for lev in BLEV:
            c = b[lev]
            if at_start(c):
                continue
            before = [d for d in O if d["per"][(b["name"], lev)][0] == "before"]
            print(f"  {b['family']:9s} {b['name'][:46]:46s} {lev} {_fmt(c):28s} preceded by {len(before)} of "
                  f"{sum(np.isfinite(d['ev']['p']) for d in O)} events; last one before: " +
                  (max(before, key=lambda d: d['ev']['p'])['row']['name'] + ', ' +
                   max(before, key=lambda d: d['ev']['p'])['ev']['lab'] if before else "-"))
    print("== cross-run check (each run on its own scale and resamples; verdict / P(neural first) against the run's "
          "accuracy passing naive (c0) and half-way (c50))")
    X = cross_runs()
    c0s = {run: d["acc"]["ev"][0]["p"] for run, d in X.items()}
    print("  -- passing naive per run, and its ratio to N1000's (the range of speeds): " + ", ".join(
        f"{run.replace('trained_', '').replace('trained', 'N1000')} {c:,.0f} (x{c / c0s['trained']:.2f})"
        for run, c in c0s.items()))
    print("  -- each event's time / the run's passing-naive time (point estimates), and the interval verdict against "
          "passing naive per run (b before, w overlaps, a after, - never)")
    for key, role in CROSS_EVENTS:
        cells, nb, nr = [], 0, 0
        for run, d in X.items():
            q = d["ev"].get((key, role))
            if q is None:
                cells.append(f"{run.replace('trained_', '').replace('trained', 'N1000'):>6s}   n/a")
                continue
            c0 = d["acc"]["ev"][0]["p"]
            v = q["vs"]["c0"][0]
            nr += 1
            nb += v == "before"
            r_ = q["e"]["p"] / c0 if np.isfinite(q["e"]["p"]) else np.nan
            cells.append(f"{run.replace('trained_', '').replace('trained', 'N1000'):>6s} {r_:5.2f}{v[0]}")
        name = (SHORT.get(key, key) + ": " + next(q for d in X.values() for k, q in d["ev"].items()
                                                    if k == (key, role))["e"]["lab"])[:58]
        print(f"     {name:58s} " + " ".join(cells) + f"   before in {nb} of {nr}")
    for run, d in X.items():
        a = d["acc"]["ev"]
        ab = d["at_ben"]
        print(f"  {run}: accuracy passes naive {_fmt(a[0])}, c50 {_fmt(a[2])}, end {d['acc']['value'][-1]:.3f}; at the "
              f"memory-benefit crossing ({ab[0]:,.0f}) intact {ab[1]:.3f} vs cut to 1 step {ab[2]:.3f} (both variables);"
              f" at the end intact {d['intact_end']:.3f}, cut {d['cut_end']:.3f}" if ab else
              f"  {run}: accuracy passes naive {_fmt(a[0])}, c50 {_fmt(a[2])}")
        for (k, role), q in d["ev"].items():
            print(f"      {q['name'][:50]:50s} {q['e']['lab'][:30]:30s} {_fmt(q['e']):30s} c0: {q['vs']['c0'][0]:6s} "
                  f"({q['vs']['c0'][1]:.2f})  c50: {q['vs']['c50'][0]:6s} ({q['vs']['c50'][1]:.2f})")
    return R, beh, O


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

FAM_COL = {"answers": "#6a3d9a", "splitting": "#01665e", "holding": "#3a7a19", "coupling": "#b2182b"}  # td_tt_timeline
NAIVE_GREY = "0.72"
XMIN = 8.
MARK = {"onset": dict(marker="o", filled=False), "half": dict(marker="o", filled=True),
        "natural": dict(marker="D", filled=True), "fall": dict(marker="v", filled=True)}
TAG = {"ben1": "0 to Bayes", "ben2": "0 to Bayes", "cut1": "start to naive", "ben5x": "0 to peak",
       "mu_all": "0 to Bayes", "mu_top": "0 to Bayes",
       "coh": "start to 1", "r_dim": "start to 0", "rho_i": "start to 1", "rho_f": "start to 1", "rho_g": "start to 1",
       "rho_o": "start to 1", "r_coh14": "0 to peak", "r_bx14": "0 to peak"}
SHORT = {"ben1": "memory benefit at T, cut to 1 step", "ben2": "memory benefit at T, cut to 2 steps",
         "ben5x": "benefit of memory > 5 steps, minus Bayes's", "mu_all": "memory use in the step update",
         "cut1": "accuracy at T with memory cut to 1 step",
         "mu_top": "memory use in the step update, most entangled fifth",
         "effdim": "effective dimension of the state", "norm_u": "state size |u|", "turn_u": "state turn between steps",
         "pr_pool": "state dimension per step (PR, pooled)", "pr_env": "state dimension per step, within environments",
         "r_norm": "r(|u|, entanglement)", "r_effdim": "r(effective dimension, entanglement)",
         "r_turn": "r(state turn, entanglement)", "coh": "credit coherence", "dim": "update dimension",
         "r_dim": "r(update dimension, entanglement): decoupling", "r_dim_neg": "r(update dimension, entanglement): negative",
         "bandx": "cross-band share (final bands)", "r_coh0": "r(coherence, entanglement), lag 0",
         "r_coh14": "r(coherence, entanglement), lags 1-4", "r_bx0": "r(cross-band share, entanglement), lag 0",
         "r_bx14": "r(cross-band share, entanglement), lags 1-4", "rho_i": "W_hh radius, i block",
         "rho_f": "W_hh radius, f block", "rho_g": "W_hh radius, g block", "rho_o": "W_hh radius, o block",
         "tau5": "units with tau > 5", "tau10": "units with tau > 10", "rank_dhh": "effective rank of W_hh - W_hh(0)",
         "dw_W_hh": "||W - W(0)||, W_hh", "dw_W_ih": "||W - W(0)||, W_ih", "dw_read-in": "||W - W(0)||, read-in",
         "dw_read-out": "||W - W(0)||, read-out"}


def _tag(r):
    return TAG.get(r["key"], "0 to end" if r["a_lab"].startswith("0") else "start to end")


def _tex(t):
    """mathtext for the glyphs cmr10 lacks (|, <, >) and for weight names (only outside existing $...$)"""
    t = t.replace("||W - W(0)||", r"$\Vert W - W(0)\Vert$").replace("|u|", r"$|u|$")
    t = t.replace("tau", r"$\tau$").replace("W_hh(0)", r"$W_{hh}(0)$").replace("W_hh", r"$W_{hh}$") \
        .replace("W_ih", r"$W_{ih}$")
    parts = t.split("$")
    for k in range(0, len(parts), 2):
        parts[k] = parts[k].replace(">=", r"$\geq$").replace("<=", r"$\leq$").replace(">", "$>$").replace("<", "$<$")
    return "$".join(parts)


def _tagtex(r):
    a, b = _tag(r).split(" to ")
    return f"[{a}" + r" $\to$ " + f"{b}]"


def _edges(x):
    return cl._edges(x)


def _xaxis(ax, xmax, label=True):
    ax.set_xscale("log")
    ax.set_xlim(XMIN, xmax)
    ax.set_xticks([10, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.minorticks_off()
    if label:
        ax.set_xlabel("training episodes")


def _row_time(r):
    """a row's time for ordering: its half-way event (else its first event)"""
    e = next((e for e in r["ev"] if e["role"] == "half"), r["ev"][0])
    return e["p"] if np.isfinite(e["p"]) else np.inf


def _layout(rows):
    """row positions: groups in GROUPS order with a gap between groups; rows in time order within a group"""
    ypos, y, heads = {}, 0., []
    for g in GROUPS:
        rs = sorted([r for r in rows if r["group"] == g], key=_row_time)
        if not rs:
            continue
        heads.append((g, y))
        y += .9
        for r in rs:
            ypos[r["key"]] = y
            y += 1.
        y += .35
    return ypos, heads, y


SHOWN = .95                                    # an event is drawn if at least 95% of the resamples cross (timeline)
GREY_TOL = .05                                 # heat maps: f in [-0.05, 0) is drawn as 0 (no change), below is grey


def shown(e):
    return bool(np.isfinite(e["p"]) and not e["start"] and e["share"] >= SHOWN)


def _mark(ax, yy, e, color, size=6., halo=None, bar_color=None, lw=2.):
    """one event: its marker at the point estimate and its 5-95% bar (weight rows: the checkpoint bracket, dotted)"""
    if not shown(e):
        return False
    m = MARK[e["role"]]
    lo = e["lo"] if np.isfinite(e["lo"]) else e["p"]
    hi = e["hi"] if np.isfinite(e["hi"]) else 6e4
    ax.plot([max(lo, XMIN), hi], [yy] * 2, color=bar_color or color, lw=lw if e["boot"] is not None else 1.2,
            ls="-" if e["boot"] is not None else (0, (1, 1)), solid_capstyle="butt", zorder=6, path_effects=halo)
    ax.plot([max(e["p"], XMIN)], [yy], marker=m["marker"], ms=size * (.85 if m["marker"] == "D" else 1.),
            mfc=color if m["filled"] else "w", mec="k" if color != COL["echo"] else color, mew=.9, zorder=7, ls="none")
    return True


def _transient(q):
    """an ESN memory-use row: the run of checkpoints whose 5-95% band is above 0 that holds the first training
    checkpoint (first, last; first = the initial network when its band is above 0 too) and the checkpoint from which
    the band stays below 0 to the end"""
    lo, hi = np.percentile(q["boot"], [5, 95], 0)
    x = q["x"]
    k = int(np.argmax(x > gap.X0))
    pos = lo > 0
    first, last = k, k
    while first > 0 and pos[first - 1]:
        first -= 1
    while last + 1 < len(x) and pos[last + 1]:
        last += 1
    neg = hi < 0
    kn = (int(np.where(~neg)[0][-1]) + 1 if (~neg).any() else 0) if neg[-1] else None
    return (x[first] if pos[k] else np.nan), x[last], (x[kn] if kn is not None else np.nan)


def _helps(t):
    """the span of a _transient (first, last, _): 'up to <last>' when it starts at the initial network"""
    return f"up to {t[1]:,.0f}" if t[0] <= gap.X0 else f"at {t[0]:,.0f}-{t[1]:,.0f}"


def _data_counts():
    """the data counts the footnotes state: probes, checkpoints, credit probes (all / cross-band), error steps"""
    D = build_ck("trained")
    return dict(n=gap.N_ENV * gap.PER_ENV, env=gap.N_ENV, per=gap.PER_ENV, K=len(D["episode"]), pairs=R * R,
                traj=gap.PER_ENV // (R * R), cr=int(D["cnt_cr"].sum()), crf=int(D["cnt_crf"].sum()),
                steps=WIN.stop - WIN.start, boot=gap.N_BOOT)


def _mu_note():
    """the memory-use footnote: its two terms at a few checkpoints and its two memoryless references (Trained RNN, all
    variable-steps 2-30)"""
    D, M = build_ck("trained"), memory_errors("trained")
    x = gap.train_x(D["episode"])
    cnt = np.full(gap.N_ENV, gap.PER_ENV * (T - 1) * NV)
    yi, yc = _ratio(M["ei_all"], cnt)[0], _ratio(M["ec_all"], cnt)[0]
    ks = [_at(x, 320), _at(x, 1600)]
    bB, bN = _const(D["bmu_all"], cnt)[0], _const(M["nmu_all"], cnt)[0]
    reached = bool(np.any(yc - yi >= bN))
    return ("Memory use = the update error with the memory cut to the last step minus the intact one; it mostly tracks "
            "the intact error (all variable-steps at " + " / ".join(f"{x[k]:,.0f}" for k in ks) + " / end: intact "
            + " / ".join(f"{v:.3f}".lstrip("0") for v in [yi[k] for k in ks] + [_end(yi)]) + ", cut "
            + " / ".join(f"{v:.3f}".lstrip("0") for v in [yc[k] for k in ks] + [_end(yc)])
            + " nats).\nIts $b$ (Bayes's own memory use) is step-wise Bayes's memoryless update error, "
            + f"{bB:.3f}".lstrip("0") + "; with naive's, " + f"{bN:.3f}".lstrip("0") + ", as $b$ it would "
            + ("be passed" if reached else "never be passed") + " (so `passes Bayes's own' depends on this reference).")


def family_f(beh, x, run):
    """per family: the mean over its tested rows of the gap fraction f (each row's f log-interpolated onto x)"""
    out = {}
    for fam in FAMS:
        fs = [np.interp(np.log(x), np.log(b["x"]), b[f"f_{run}"]) for b in beh if b["family"] == fam
              and b[f"f_{run}"] is not None]
        out[fam] = np.mean(fs, 0)
    return out


def figure(R=None, beh=None, O=None):
    """-> figure_bin/td_tt_neural.png: (a) each Trained RNN row's progress f = (y - a) / (b - a) through training
    (top: the behavioural families' mean gap fraction), (b) the ESN on the Trained RNN's scale (annotated: its
    transient memory use, its input-driven r(state turn, e)), (c) the events with their 5-95% intervals against the
    behavioural families' onset (c10) and half-way (c50) crossings (behaviour: each row's point crossing and the range
    of the family's points; splitting's six-step row marked x). The title (TITLE_FIG1) is filled from the data"""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    FL, TS = fs["label"], 12.
    R = R or all_rows()
    beh = beh or behaviour()
    rows = R["trained"]
    es = {r["key"]: r for r in R["echo"]}
    rng = family_ranges(beh)
    x = rows[0]["x"]
    ypos, heads, ybot = _layout(rows)
    yfam = {f: i - len(FAMS) - .45 for i, f in enumerate(FAMS)}                     # family rows above the neural ones
    ytop = min(yfam.values()) - .5 - 1.0
    xmax = float(_edges(x)[-1]) * 1.1
    RH = .255
    PW, PWC, LAB, GX, CBW, RM = 3.2, 4.0, 4.1, .35, .7, .3
    TOP, BOT = 2.4, 2.9
    PH = (ybot - ytop) * RH
    W = LAB + 2 * PW + PWC + 2 * GX + CBW + RM
    H = TOP + PH + BOT
    fig = plt.figure(figsize=(W, H))
    xs = [LAB, LAB + PW + GX, LAB + 2 * PW + 2 * GX + CBW]
    axes = [fig.add_axes([x0 / W, BOT / H, (PWC if k == 2 else PW) / W, PH / H]) for k, x0 in enumerate(xs)]
    cmap = plt.cm.viridis.copy()
    cmap.set_under(NAIVE_GREY)
    cmap.set_bad("w")
    halo = [pe.withStroke(linewidth=2.2, foreground="k")]
    A = accuracy_row("trained")
    pass_naive = A["ev"][0]["p"]
    early = [b["c50"]["p"] for b in beh if b["family"] == "answers" and not at_start(b["c50"])]
    late = [b["c50"]["p"] for b in beh if b["family"] in ("holding", "coupling") and not at_start(b["c50"])]
    spans = ((min(early), max(early), FAM_COL["answers"]), (min(late), max(late), FAM_COL["coupling"]))
    esn_tr = {k: _transient(es[k]) for k in ("mu_all", "mu_top") if k in es}
    tx = 1.15 * np.nanmax([v[2] for v in esn_tr.values()]) if esn_tr else 300.
    im = None
    for j, run in enumerate(RUNS):
        ax = axes[j]
        ff = family_f(beh, x, run)
        for fam in FAMS:
            yy = yfam[fam]
            ax.pcolormesh(_edges(x), [yy - .5, yy + .5], ff[fam][None], cmap=cmap, vmin=0., vmax=1., shading="flat",
                          rasterized=True)
        for r in rows:
            q = r if run == "trained" else es.get(r["key"])
            if q is None:
                continue
            yy = ypos[r["key"]]
            fv = np.where((q["f"] < 0) & (q["f"] >= -GREY_TOL), 0., np.clip(q["f"], -1e3, 1e3))
            im = ax.pcolormesh(_edges(q["x"]), [yy - .5, yy + .5], fv[None], cmap=cmap, vmin=0., vmax=1.,
                               shading="flat", rasterized=True)
            for e in q["ev"]:
                if e["role"] in ("half", "natural", "fall"):
                    _mark(ax, yy, e, COL[run], 4.6, halo=halo, bar_color="w", lw=1.5)
            if run == "echo" and r["key"] in esn_tr:                             # the ESN's transient memory use
                if np.isfinite(esn_tr[r["key"]][0]):
                    ax.text(tx, yy, f"helps only {_helps(esn_tr[r['key']])}, then hurts", ha="left", va="center",
                            fontsize=FL - 1., color="k", zorder=8)
            if run == "echo" and r["key"] == "r_turn":                           # input-driven, not learned
                ax.text(XMIN * 1.35, yy, "input-driven at init (r = " + f"{q['value'][0]:.2f}".lstrip("0")
                        + "), kept: not learned", ha="left", va="center", fontsize=FL - 1., color="k", zorder=8)
        _xaxis(ax, xmax)
        ax.set_ylim(ybot, ytop)
        ax.set_yticks([])
        ax.axhline(max(yfam.values()) + .5 + .225, color="k", lw=1.)
        for g, yh in heads[1:]:
            ax.axhline(yh - .2, color="k", lw=.6)
    tr0 = axes[0].get_yaxis_transform()
    axes[0].text(-.02, min(yfam.values()) - .95, "behaviour (family mean of $f$)", transform=tr0, ha="right",
                 va="center", fontsize=FL + .5, fontweight="bold")
    for fam in FAMS:
        axes[0].text(-.02, yfam[fam], fam, transform=tr0, ha="right", va="center", fontsize=FL, color=FAM_COL[fam],
                     fontweight="bold")
    for g, yh in heads:
        axes[0].text(-.02, yh + .2, GROUP_NAME[g], transform=tr0, ha="right", va="center", fontsize=FL + .5,
                     fontweight="bold")
    for r in rows:
        axes[0].text(-.02, ypos[r["key"]], _tex(SHORT.get(r["key"], r["name"])) + "   " + _tagtex(r),
                     transform=tr0, ha="right", va="center", fontsize=FL - .3)
    axes[0].set_title("a   Trained RNN: progress $f$\n"
                      r"$f = (y - a)\,/\,(b - a)$,  $[a \to b]$ per row", loc="left", fontsize=TS, linespacing=1.5)
    axes[1].set_title("b   ESN on the Trained RNN's scale\n"
                      r"same $a$ and $b$ as in a", loc="left", fontsize=TS, linespacing=1.5)
    cax = fig.add_axes([(xs[1] + PW + .13) / W, BOT / H + .5 * PH / H, .13 / W, .28 * PH / H])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_ticks([0, .5, 1])
    cb.set_ticklabels([r"0 = $a$", ".5", r"1 = $b$"])
    cb.ax.tick_params(labelsize=FL - .5)
    cb.ax.set_title("$f$", fontsize=FL + .5)
    cax.text(-.6, -.06, "grey:\n" + r"$f < -0.05$", transform=cax.transAxes, ha="left", va="top", fontsize=FL - 1)
    # (c) crossings
    ax = axes[2]
    for lo_, hi_, col in spans:
        ax.axvspan(lo_, hi_, color=col, alpha=.11, lw=0, zorder=0)
    ax.axvline(pass_naive, color="k", lw=1.1, ls=":", zorder=1)
    for i, r in enumerate(sorted(rows, key=lambda q: ypos[q["key"]])):
        yy = ypos[r["key"]]
        if i % 2 == 0:
            ax.axhspan(yy - .5, yy + .5, color="#efece2", lw=0, zorder=0, alpha=.7)
        for e in r["ev"]:
            _mark(ax, yy - .1, e, COL["trained"], 6.)
        q = es.get(r["key"])
        if q is not None:
            for e in q["ev"]:
                if e["role"] in ("half", "natural") and shown(e):
                    ax.plot([max(e["p"], XMIN)], [yy + .28], marker="s", ms=4.5, mfc=COL["echo"], mec=COL["echo"],
                            zorder=7, ls="none")
    for fam in FAMS:
        yy = yfam[fam]
        if (fam, "c10") in rng:
            a10, b10 = rng[(fam, "c10")]
            ax.plot([a10, b10], [yy + .18] * 2, color=FAM_COL[fam], lw=2.2, alpha=.55, solid_capstyle="butt", zorder=5)
        a50, b50 = rng[(fam, "c50")]
        ax.plot([a50, b50], [yy - .12] * 2, color=FAM_COL[fam], lw=5.5, solid_capstyle="butt", zorder=5)
        for b in beh:
            if b["family"] != fam:
                continue
            for lev, dy, filled in (("c10", .18, False), ("c50", -.12, True)):
                c = b[lev]
                if not np.isfinite(c["p"]) or at_start(c):
                    continue
                if b["name"].startswith("six-step"):                             # no clean timing marker (its topic)
                    ax.plot([c["p"]], [yy + dy], marker="X", ms=6.5, mfc="w", mec=FAM_COL[fam], mew=1.1, zorder=6,
                            ls="none")
                    continue
                ax.plot([c["p"]], [yy + dy], marker="o", ms=3.6, mfc=FAM_COL[fam] if filled else "w",
                        mec="w" if filled else FAM_COL[fam], mew=.8, zorder=6, ls="none")
    ax.axhline(max(yfam.values()) + .5 + .225, color="k", lw=1.)
    _xaxis(ax, xmax)
    ax.set_ylim(ybot, ytop)
    ax.set_yticks([])
    for g, yh in heads[1:]:
        ax.axhline(yh - .2, color="k", lw=.6)
    ax.set_title("c   When each change happens\n"
                 "neural: 5-95%; behaviour: range of the rows' points", loc="left", fontsize=TS, linespacing=1.5)
    fc = FAM_COL["answers"]
    h1 = [Line2D([], [], color=COL["trained"], marker="o", ms=6, mfc="w", mec="k", lw=1.8, label="onset: 10% of the way"),
          Line2D([], [], color=COL["trained"], marker="o", ms=6, mfc=COL["trained"], mec="k", lw=1.8,
                 label="half-way (rises and falls: half its peak, first)"),
          Line2D([], [], color=COL["trained"], marker="D", ms=5, mfc=COL["trained"], mec="k", lw=1.8,
                 label="natural boundary: Bayes's level, coherence 1, sign"),
          Line2D([], [], color=COL["trained"], marker="v", ms=6, mfc=COL["trained"], mec="k", lw=1.8,
                 label="falls back below half its peak"),
          Line2D([], [], color=COL["trained"], lw=1.2, ls=(0, (1, 1)), label="weights: checkpoint bracket"),
          Line2D([], [], color=COL["echo"], marker="s", ms=5, mfc=COL["echo"], mec=COL["echo"], lw=0,
                 label="ESN reaches the Trained RNN's level"),
          Line2D([], [], color="k", lw=1.1, ls=":", label=f"Trained RNN passes naive ({pass_naive:,.0f})"),
          Line2D([], [], color=FAM_COL["splitting"], marker="X", ms=7, mfc="w", mec=FAM_COL["splitting"], mew=1.1,
                 lw=0, label="splitting's six-step row: no clean timing marker (kept in the tests)"),
          Patch(color=spans[0][2], alpha=.25, label=f"answers half-way ({spans[0][0]:,.0f}-{spans[0][1]:,.0f})"),
          Patch(color=spans[1][2], alpha=.22, label=f"holding, coupling half-way ({spans[1][0]:,.0f}-{spans[1][1]:,.0f})"),
          Line2D([], [], color=fc, alpha=.55, lw=2.2, marker="o", ms=4.5, mfc="w", mec=fc,
                 label="family onset $c_{10}$: its rows' points, their range"),
          Line2D([], [], color=fc, lw=5.5, marker="o", ms=4.5, mfc=fc, mec="w",
                 label="family half-way $c_{50}$: its rows' points, their range")]
    fig.legend(handles=h1, loc="center", bbox_to_anchor=(.5, 1.97 / H), ncol=3, frameon=False, fontsize=FL,
               columnspacing=1.8, handlelength=2.2)
    dc = _data_counts()
    note = (f"Data (seed 20261029): probes = {dc['n']:,} episodes ({dc['env']} test environments x {dc['pairs']} latent pairs x "
            f"{dc['traj']} trajectories, {T} steps) at {dc['K']} checkpoints per network; credit: {dc['cr']:,} probes x {dc['steps']} "
            f"error steps (cross-band: the first {dc['crf']:,}).\nBehaviour rows: the mean gap fraction of each family's "
            "rows (behavioural timeline; white = a row without a value).\n"
            f"Intervals: 5-95% over {dc['boot']:,} resamples of the {dc['env']} environments, the same resamples as the "
            "behavioural timeline (paired); weight rows have no resamples (bracket = the two checkpoints around the "
            "crossing).\n" + _mu_note())
    fig.text(.5, .72 / H, note, ha="center", va="center", fontsize=FL - 1, linespacing=1.5)
    trk = {r["key"]: r for r in rows}
    emu = es["mu_all"]["value"]
    kpk = int(np.argmax(emu))
    title = TITLE_FIG1.format(mu_top=trk["mu_top"]["ev"][0]["p"], mu_all=trk["mu_all"]["ev"][0]["p"],
                              esn_f=f"{max(float(_end(es[k]['f'])) for k in TITLE_KEYS if k in es):.2f}".lstrip("0"),
                              esn_ben=" / ".join(f"{float(_end(es[k]['f'])):.2f}".lstrip("0") for k in TITLE_BEN),
                              esn_helps=_helps(esn_tr["mu_all"]), esn_mu0=f"{emu[0]:+.4f}".replace("0.", ".", 1),
                              esn_mupk=f"{emu[kpk]:+.4f}".replace("0.", ".", 1), esn_xpk=es["mu_all"]["x"][kpk])
    fig.suptitle(title, y=1 - .3 / H, fontsize=TS + 1.5, linespacing=1.4, va="top")
    path = fig_path("td_tt_neural.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


TITLE_FIG1 = ("Before the answers are half-way, the Trained RNN's state starts to turn and widen and its learning signal "
              "decouples from input novelty;\nmemory carries the answers but makes the step update more Bayes-like only "
              "after the answers' onset, around their half-way ({mu_top:,.0f}-{mu_all:,.0f}), before holding and coupling "
              "are half-way;\ncancelling credit and the second wave of long time constants follow, before holding's "
              "half-way and overlapping coupling's.\nThe ESN makes none of these changes for good: it ends at most {esn_f} "
              "of the way on the state, credit and time-constant changes,\nits memory benefit at T ends at about half of "
              "Bayes's ({esn_ben}), and its memory helps the step update only {esn_helps} episodes ({esn_mu0} nats at "
              "init, {esn_mupk} at {esn_xpk:,.0f}).")
TITLE_KEYS = ("turn_u", "pr_env", "pr_pool", "effdim", "r_dim", "coh", "bandx", "tau5",
              "tau10")                  # the state, credit and time-constant changes the main title names (ESN: end f)
TITLE_BEN = ("ben1", "ben2")            # the memory benefit at T ('memory carries the answers'; ESN: end f)
TITLE_MU = ("mu_all", "mu_top")         # memory use in the step update (ESN: end f and its transient)
TITLE_FIG2 = ("Which neural change comes first? Each Trained RNN event against each behavioural family's onset ($c_{10}$) "
              "and half-way ($c_{50}$)\ncolour: share of the paired resamples in which the change comes before every row "
              "of the family;  symbols: before / after every row (5-95% intervals)")
TESTS = (("answers", "c10"), ("answers", "c50"), ("splitting", "c50"), ("holding", "c10"), ("holding", "c50"),
         ("coupling", "c10"), ("coupling", "c50"))


def figure_order(R=None, beh=None, O=None):
    """-> figure_bin/td_tt_neural_order.png: every Trained RNN event (rows, in time order) against each behavioural
    family's onset (c10) and half-way (c50): colour = paired share of resamples in which the event precedes every row
    of the family; symbol = the interval rule against every row (before / after; none = overlaps at least one)"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    FL, TS = fs["label"], 12.
    R = R or all_rows()
    beh = beh or behaviour()
    O = O or orders(R["trained"], beh)
    es = {r["key"]: r for r in R["echo"]}
    items = sorted([d for d in O if shown(d["ev"])], key=lambda d: d["ev"]["p"])
    n = len(items)
    RH, CW = .235, .78
    LAB, TW, EW, RM = 5.6, 1.25, .95, .25
    TOP, BOT = 1.75, 2.1
    W = LAB + TW + len(TESTS) * CW + EW + RM
    H = TOP + n * RH + BOT
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([(LAB + TW) / W, BOT / H, len(TESTS) * CW / W, n * RH / H])
    M = np.full((n, len(TESTS)), np.nan)
    for i, d in enumerate(items):
        for j, (fam, lev) in enumerate(TESTS):
            f_ = d["fam"].get((fam, lev))
            if f_ is not None:
                M[i, j] = f_["p_all_before"]
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("w")
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                   extent=(-.5, len(TESTS) - .5, n - .5, -.5))
    for i, d in enumerate(items):
        for j, (fam, lev) in enumerate(TESTS):
            f_ = d["fam"].get((fam, lev))
            if f_ is None:
                continue
            if f_["verdict"] == "before":
                ax.plot([j], [i], marker="<", ms=6.5, mfc="w", mec="k", mew=.6, ls="none")
            elif f_["verdict"] == "after":
                ax.plot([j], [i], marker=">", ms=6.5, mfc="k", mec="w", mew=.6, ls="none")
    ax.set_xticks(range(len(TESTS)))
    rng = family_ranges(beh)
    ax.set_xticklabels([f"{fam}\n{lev[0]}$_{{{lev[1:]}}}$\n{rng[(fam, lev)][0]:,.0f}-\n{rng[(fam, lev)][1]:,.0f}"
                        for fam, lev in TESTS], fontsize=FL - .5, linespacing=1.15)
    for t, (fam, _) in zip(ax.get_xticklabels(), TESTS):
        t.set_color(FAM_COL[fam])
    ax.xaxis.tick_top()
    ax.set_yticks([])
    for j in (1.5, 2.5, 4.5):
        ax.axvline(j, color="w", lw=2.5)
    ax.tick_params(length=0)
    tr = ax.get_yaxis_transform()
    for i, d in enumerate(items):
        r, e = d["row"], d["ev"]
        lab = _tex(SHORT.get(r["key"], r["name"])) + ": " + _tex(e["lab"]) + ("*" if e["boot"] is None else "")
        ax.text(-(TW + .1) / (len(TESTS) * CW), i, lab, transform=tr, ha="right", va="center", fontsize=FL - .6)
        hi = f"{e['hi']:,.0f}" if np.isfinite(e["hi"]) else "never"
        ax.text(-.12 / (len(TESTS) * CW), i, f"{e['p']:,.0f} [{e['lo']:,.0f}-{hi}]", transform=tr, ha="right",
                va="center", fontsize=FL - .8)
        q = es.get(r["key"])
        qe = None if q is None else next((x_ for x_ in q["ev"] if x_["lab"] == e["lab"]), None)
        if qe is not None and qe["role"] == "fall":
            rise = next((x_ for x_ in q["ev"] if x_["rule"] == "first"), None)
            if rise is None or not shown(rise):
                qe = dict(qe, p=np.nan, share=0.)                     # never rose, so it cannot fall back
        if qe is None:
            ax.text(1 + .1 / (len(TESTS) * CW), i, "-", transform=tr, ha="left", va="center", fontsize=FL - 1.3,
                    color="k", alpha=.5)
        elif shown(qe):
            ax.text(1 + .1 / (len(TESTS) * CW), i, f"{qe['p']:,.0f}", transform=tr, ha="left", va="center",
                    fontsize=FL - .8, color=COL["echo"])
        elif qe is not None:
            ax.text(1 + .1 / (len(TESTS) * CW), i, "never" if not np.isfinite(qe["p"]) or qe["share"] < SHOWN
                    else "at start", transform=tr, ha="left", va="center", fontsize=FL - 1.3, color="k", alpha=.5)
    ax.text(1 + .1 / (len(TESTS) * CW), -1.1, "ESN", transform=tr, ha="left", va="center", fontsize=FL,
            color=COL["echo"], fontweight="bold")
    ax.text(-.12 / (len(TESTS) * CW), -1.1, "Trained RNN", transform=tr, ha="right", va="center", fontsize=FL,
            color=COL["trained"], fontweight="bold")
    for i in range(1, n):
        if i % 5 == 0:
            ax.axhline(i - .5, color="w", lw=.6, alpha=.6)
    cax = fig.add_axes([(LAB + TW) / W, 1.34 / H, len(TESTS) * CW * .5 / W, .13 / H])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, .5, 1])
    cb.ax.tick_params(labelsize=FL - 1)
    cb.set_label("P(the neural change comes before every row of the family), paired resamples", fontsize=FL - .5)
    h = [Line2D([], [], marker="<", ms=7, mfc="w", mec="k", ls="none", label="before every row (5-95% intervals)"),
         Line2D([], [], marker=">", ms=8, mfc="k", mec="k", ls="none", label="after every row"),
         Line2D([], [], marker="s", ms=7, mfc="#3b528b", mec="#3b528b", ls="none",
                label="no symbol: overlaps at least one row")]
    fig.legend(handles=h, loc="center left", bbox_to_anchor=((LAB + TW + len(TESTS) * CW * .53) / W, 1.41 / H),
               frameon=False, fontsize=FL - .5, ncol=1, handletextpad=.4)
    dc = _data_counts()
    six = [b["c50"]["p"] for b in beh if b["family"] == "splitting" and b["name"].startswith("six-step")]
    rest = [b["c50"]["p"] for b in beh if b["family"] == "splitting" and not at_start(b["c50"])
            and not b["name"].startswith("six-step")]
    fig.text(.5, .42 / H, f"Data (seed 20261029): probes = {dc['n']:,} episodes ({dc['env']} test environments x "
             f"{dc['per']}) at {dc['K']} checkpoints; credit: {dc['cr']:,} probes x {dc['steps']} error steps (cross-band: "
             f"the first {dc['crf']:,}); intervals and P over {dc['boot']:,} paired resamples of the {dc['env']} "
             "environments.\n* weights: no resamples; the point time against the behavioural resamples (not paired). "
             "Behavioural crossings past from the start in at least 5% of the resamples are left out (splitting's "
             "onsets, the fold index onset).\n" + (f"Splitting $c_{{50}}$ includes the six-step move toward own cells "
                                                    f"({six[0]:,.0f}), no clean timing marker (flagged by the splitting topic); "
                                                    f"without it {min(rest):,.0f}-{max(rest):,.0f}.\n" if six and rest
                                                    else "")
             + "ESN column: when the ESN reaches the same level on the Trained RNN's scale (at least 95% of resamples); "
             "never = it does not.", ha="center", va="center", fontsize=FL - 1, linespacing=1.4)
    fig.suptitle(TITLE_FIG2 if TITLE_FIG2 else "Order tests", x=.5, y=1 - .3 / H, fontsize=TS + 1, linespacing=1.4)
    path = fig_path("td_tt_neural_order.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    what = sys.argv[1:]
    if what and what[0] == "build":
        for r in what[1:] or RUNS:
            build_ck(r, force=True)
    if what and what[0] == "replay":
        replay(what[1], int(what[2]) if len(what) > 2 else 0, int(what[3]) if len(what) > 3 else 1)
    if what and what[0] == "verify_replay":
        verify_replay(*(what[1:2] or ["trained"]), *([int(what[2])] if len(what) > 2 else []))
    if what and what[0] in ("numbers", "synthesis", "figure"):
        RR, BB, OO = numbers() if "numbers" in what else (None, None, None)
        if "synthesis" in what:
            synthesis(RR, BB, OO)
        if "figure" in what:
            figure(RR, BB, OO)
            figure_order(RR, BB, OO)
