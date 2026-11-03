"""Two lean figures of the neural / mechanistic results through training (John 11-02, on td_tt_mechanism.png,
td_tt_neural*.png, td_tt_replication.png, td_tt_memory_events.png: "there is just a lot going on ... Let's try to
compress the neural and mechanistic results into clear visualizations that show the processes happening, without all
the noise"; "Swaps and lesions are also very overwhelming").
-> main/REPLICATION/figure_bin/td_tt_processes.png    figure_processes(): what changes inside the Trained RNN as it
                                                       learns (6 panels, one shared training-episode axis)
   main/REPLICATION/figure_bin/td_tt_mechanisms.png   figure_mechanisms(): what each mechanism test shows (4 panels)
   numbers()   every drawn value with its source (and the built-in reproduction checks)
Supersedes the synthesis figure td_tt_mechanism.png (deleted 11-02, John agreed it can be ignored; td_tt_mechanism.py
itself is left unchanged and would re-create it if run).
Labels round (11-02, after the independent check: FIXABLE, every drawn number correct): titles, suptitles, legends,
footnotes and this docstring restated within the source modules' scopes ('depend on', not 'use', memory; no tie of the
memory horizon to passing naive; 'shrinks on average', not 'never expands'; transient amplification, not 'the state
grows'; tau_1 94-111 at the end; the lesion's overlapping early intervals no longer ordered; the swaps claim scoped to
the drawn classes; 'separate more after', not 'split by', entangled observations); panel d of the first figure draws
every stored lag (1, 2, 5, 10, 20) instead of lag 20 alone (John: never handpick a lag); the count-code band's end is
read from the count-preserving growth (the same 100 episodes as before); no other drawn value changed.
Panel f round (11-03): panel f of the first figure draws td_tt_chaos_ent's verified headline, the one-step expansion
log rho(J_t) by entanglement fifth (it replaces the flip's growth g_t by fifth, which showed only 'shrinks on average');
its title, the suptitle's (f) clause (the suptitle's line break moved so that it fits; the other clauses verbatim) and
the footnote's f lines (one line more; the footnote sits .06 in lower to keep clear of the x labels) follow from it;
nothing else in either figure changed.

Nothing new is measured here. Every drawn series is recomputed at run time from the caches of the verified modules,
with their own functions (their docstrings are the record: definitions, checks, caveats, failed hypotheses):
  td_tt_gap / td_tt_jumps   goal accuracy at T (export 'jumps'), passing naive c0 = 835 [798-876]
  td_tt_memlen              memory horizon: random-cell memory cuts, onsets per window w = 1..15 (engine donors)
  td_tt_memcuts             within-episode build-up rule (uncut_reach), Bayes's per-step accuracy, restart and
                            belief-matched swaps (dep, _seed_mean)
  td_tt_chaos               tau_1 = -1/lambda_1 (lam_curve), within-episode amplification (lyap amp)
  td_tt_chaos_combined      the chaos sweep's single-flip rise at its 5 stored lags (sweep, sweep_refs); the pairs'
                            entanglement fifths (pairs_combined.npz ent5_std); td_tt_flips' count classes
                            (flips/combined.npz)
  td_tt_chaos_ent           the one-step expansion log rho(J_t) per entanglement fifth and over all steps; the flip's
                            state growth g_t on the step after the flip and over later lags, for the footnote
                            (series_<run>.npz, run_curves)
  td_tt_lesions             specific damage of the 5 latent-interaction dims against each latent's own 5 dims
  td_tt_swaps               excess share of weight classes over size-matched random sets (excess_cells)
The only recomputation not done by a source module: panel c of the first figure applies td_tt_memcuts.uncut_reach's
rule to td_tt_memlen's intact per-step accuracies (all 114 checkpoints instead of memcuts' 46); numbers() checks that
it equals memcuts' own values at the 46 shared checkpoints (max |d| 0; the two caches differ in 1 of 69,000
per-environment step sums, an fp near-tie).

WHAT IS DRAWN (Trained RNN = LSTM N = 1,000, seed 20261029; ESN = the same network with its LSTM weights frozen at
initialization, only the read-in, read-out and start state trained (train_training_dynamics.py 'echo': reservoir);
x = training episodes, the initial network at 10 drawn as 0, axis 8-70,000 in the first figure; dotted vertical: the
Trained RNN's goal accuracy passes naive's for good, 835; light bands: the half-way spans of answers / splitting
(774-1,575) and holding / coupling (2,747-11,103) of td_tt_timeline RESULTS (additive scale; the same constants as
td_tt_mechanism.EARLY_SPAN / LATE_SPAN); bands and bars 5-95% over the 1,000 shared resamples of the 50 test
environments, td_tt_gap.boot_counts(), unless named; one legend row per figure: Trained RNN blue (by lag, viridis, in
panel d of the first figure), ESN orange, Bayes black, naive grey; grey marks naive only)
td_tt_processes.png
  a  goal accuracy at T, Trained RNN and ESN (114 checkpoints each), naive .464 and Bayes .771 on the same probes.
  b  memory horizon: the state at t - w replaced by the state of another cell's probe of the same environment (one
     fixed donor per probe, seed 20261201), the cell rerun over the last w inputs; L(w) = goal accuracy at T lost;
     onset(w) = the training episode after which L(w) stays above half of Bayes's L(w) under the same substitution;
     w = 1..15 with 5-95% bars; the ESN reaches it only at w = 1. The dates depend on the donor scheme (td_tt_memlen
     result 3: w = 15 at 3,865 / 4,408 / 6,924 under three schemes); the order by w does not (x1.148 per step here,
     x1.139 / x1.164 under the other two; no pair reversed in any scheme). Worded 'depend on', not 'use': the
     random-cell cut is a ceiling, not a measure of reliance (td_tt_memlen READING), and the onsets (631-4,408 here;
     675-3,865 and 806-6,924 under the other schemes) are not tied to passing naive (td_tt_memlen NULLS a, b).
  c  within-episode speed: the first step t (linear between steps) at which the uncut network's goal accuracy at
     step t reaches Bayes's at step 5 (.424); Bayes = 5 by definition; never within 30 steps before 800 (the bar at
     the top); Trained RNN only (the ESN's is not in td_tt_memcuts' record).
  d  a flipped observation's growing effect: chi(s + lag, s) - chi(s, s) at every stored lag (1, 2, 5, 10, 20; John:
     never handpick a lag), chi = Hilbert distance d_H between the intact and flipped marginal beliefs (mean of 2
     variables), one of 5 observations flipped at s = 1, 3, 5, 10. Trained RNN: one viridis line per lag with its
     5-95% band (as td_tt_chaos_sweep e); ESN: thin orange lines at the same lags, no bands (its rise grows with the
     lag at all 114 checkpoints, so lag 20 is its top line). Bayes's rise +.064 / +.100 / +.153 / +.191 / +.236 at
     lags 1 / 2 / 5 / 10 / 20 (drawn: lag 20, its largest; the others in the footnote), naive 0 (exact, same flips).
     Shaded: td_tt_flips' early count code (the response follows how many of the 5 observations are 1): darker = on
     (count-preserving 2-bit flips grow < .1 of count-changing ones: 20, 40), lighter = switching off until the
     count-preserving growth is within a tenth of the count-changing growth (100: +1.894 vs +1.998); the same window
     as td_tt_flips' 'switches off between 40 and 100 episodes', which is read from the count-preserving growth, not
     from the count-code index (that also measures a partial count dependence).
  e  tau_1 = -1/lambda_1, the networks' (u, c) state's asymptotic decay time when driven beyond the 30 trained steps
     (lambda_1: Trained RNN 1,500-step QR estimate, mean of 4 frames, band = union of the frames' bands; ESN 300-step
     estimate); every lambda_1 < 0: never chaotic; Bayes and naive never decay (all exponents exactly 0 in log
     coordinates, td_tt_chaos T1: each side's chart is stated on the panel). At the end the QR estimate (94, drawn) is
     biased low; the frame-free bound gives 111, so tau_1 = 94-111 there (td_tt_chaos RESULTS; printed by numbers()).
     Bottom strip (key at the upper right): checkpoints where the frame-free within-episode growth (1/30) log
     sigma_1(J_30 ... J_1) is above 0 with its 5-95% interval: some perturbation direction of the state is transiently
     amplified over an episode (non-normal amplification; td_tt_chaos M1a: not a chaos criterion): 60-80 and every
     checkpoint from 3,000.
  f  the one-step expansion log rho(J_t): the log of the largest |eigenvalue| of the step Jacobian J_t = d(u_t, c_t) /
     d(u_{t-1}, c_{t-1}) (2,000 dims) at the intact state and observation t, at every step of the 1,000 probes
     (td_tt_chaos_ent (1): Arnoldi, checked against exact eigenvalues; coordinate-free: S^-1 J S has J's eigenvalues).
     > 0: the step map expands an eigen-direction; potential expansion, not realised growth (consecutive steps are
     different maps). Trained RNN: the most / least entangled fifth of the step's observations (fifths ranked within
     each step, 200 probes per fifth and step) and all steps (the mean; pooled ratios of sums over the environments,
     td_tt_chaos_ent's averaging); ESN: its all-step mean. Broken y axis: upper -.055 to +.022, lower -.43 to -.32 (the
     ESN, and the initial network at -.416). Bayes and naive: log rho = 0 exactly (translations in log coordinates,
     td_tt_chaos T1). Title scoped to passing naive: the most entangled fifth is > 0 while the mean is < 0 (both
     bounds) at 560-640, 880-1,200 and 1,360 (at 720-800 the mean's interval reaches 0); at 60-80 all steps expand on
     average (the least entangled most), a resolved small effect stated in the footnote. Footnote, not drawn: actual
     perturbations grow only transiently: g_t on the step after a flip is > 0 on average at 800-5,000 and at 720 only
     in the two most entangled fifths (most - least resolved there); over lags 1-4, 5-20 and 21-29 every fifth is < 0
     at all 114 checkpoints (td_tt_chaos_ent (2)). The footnote's ranges are read from the series at draw time and its
     worded claims asserted there.
td_tt_mechanisms.png
  a  5 latent-interaction dims of u clamped at every step (td_tt_lesions joint5) against each latent's own 5
     main-effect dims (r0 / r1, the two lesions averaged): specific damage ('extra damage' on the axis) = f(r0/r1
     clamped) - f(interaction clamped), f = gap fraction (naive 0, Bayes 1) averaged over each panel's rows (core
     answers: goal right, both right, far errors, distant pairs merged; corner answers: answer = truth at corner cells,
     forward set; holding 5 rows; coupling 6 rows); 20 checkpoints, 18 drawn, from 320 (as td_tt_lesions.fig_main).
     Resolved above 0: core answers at 320-3,200 only (below 0 from 8,000), corner answers from 320 on and growing,
     holding from 480 (above +.05 from 1,600), coupling at 640 and from 1,120. Core and corner answers both start at
     320 and overlap at 320-480, so neither comes 'first'; corner answers are hit more from 640 on. The footnote gives
     the late window means against the module's two other controls: random dims of the same variance ('vm') and a
     5-dim subspace of both latents' main effects ('add5', more variance than the interaction dims), against which the
     core answers are not specially hit (td_tt_lesions RESULTS 1).
  b  weight swaps between consecutive checkpoints (td_tt_swaps): (class's two-endpoint Shapley value - the mean of 3
     random sets of its size) / the window's change; filled = all candidate-gate (g) weights (W_ih g + W_hh g + g
     biases, 2M) against random 2M of W; open = the other classes (recurrent f + i rows vs random 2M of W_hh, recurrent
     o rows vs random 1M of W_hh, all input weights W_ih, including W_ih's g rows that the filled class shares, vs
     random 4M of W); drawn where td_tt_swaps draws them (the change's 5-95% excludes 0, |change| >= .1, |interaction|
     <= |change|); training loss resampled over its own 2,000 episodes. Families as td_tt_swaps (purple = answers, as
     in td_tt_timeline): answers = goal right, both right, far errors and corner answer = truth (not panel a's core
     answers, which have distant pairs merged instead of the corner row). The title holds for these classes against
     these baselines only; td_tt_swaps ANSWER lists the exceptions, two quoted in the footnote (not recomputed here).
     Window separators pale blue (grey is naive's), stopping below the legend.
  c  goal accuracy at T lost when the state 5 steps back is replaced: restart = the trained start state (46
     checkpoints, 35 drawn, from 280); same belief = the state of a generated episode of another cell and history whose
     read-out marginals match the probe's (16 checkpoints, mean of 3 pools); Bayes under the identical substitutions
     (5-step window .031; its matched null .004 = Bayes swapped with donors matched on its own marginals).
  d  end of training: two probes of the same environment and step with the identical observation; mean change of the
     d_H between their marginal beliefs after it, matched on step and the d_H before (40 bins), per entanglement fifth
     of the observation (td_tt_chaos_combined p; 1,710,703 pair-steps); naive exactly 0. Observational: the pairs
     separate more after the more entangled observations (not 'are split by' them); for Bayes the dependence is
     partly built in (the entanglement is computed from Bayes's beliefs); the ESN's pairs separate as much (.91-.95)
     after every fifth.

RESULTS (printed by numbers(); [5-95%])
  Accuracy: the Trained RNN passes naive at 835 [798-876] and is half-way to Bayes at 1,394 [1,285-1,489]; .760 at the
    end (Bayes .771); the ESN ends at .431 (naive .464).
  Memory horizon: onsets 631 [598-676] (w = 1), 846 (2), 1,233 [1,092-1,370] (5), 2,639 [2,392-2,910] (10), 4,408
    [3,884-5,539] (15); x1.148 [x1.142, x1.166] per extra step; 88 of 105 pairs ordered (every pair 4 or more steps
    apart), none reversed; ESN w = 1 only: 11,405 [7,814-38,692] (97% of resamples). Other donor schemes: w = 1 / 15
    at 675 / 3,865 and 806 / 6,924, x1.139 / x1.164 per step, none reversed (only the order holds under every scheme).
  Within-episode speed: step 20.9 [17.0, never] at 800 (89% reach), 13.6 [11.9, 17.0] at 880, 7.5 [7.1, 8.1] at
    1,680, 6.1 at 4,000, 5.1 [5.0, 5.2] at the end; never within 30 steps up to 720.
  Flip growth, lags 1 / 2 / 5 / 10 / 20 (Bayes +.064 / +.100 / +.153 / +.191 / +.236): the Trained RNN peaks at +.68
    (1,200) / +.89 (1,360) / +1.28 (180) / +1.60 (180) / +2.22 (300, one of the 160-400-episode excursions that follow
    the checkpoint grid); lag 20 +.87 at 20, +1.65 [+1.50, +1.81] at 320, +1.52 (880), +1.30 (1,680), +1.01 (4,200);
    at the end +.053 [+.038, +.067] / +.040 [+.022, +.057] / -.032 [-.057, -.006] / -.047 [-.083, -.010] / +.019
    [-.028, +.067], each below Bayes's at its lag; last above Bayes's level at 35,200 [30,400-39,996] / 28,000 /
    17,600 / 19,200 / 21,600 [19,200-27,200]. Lags 1-2 peak after passing naive (1,200-1,360), lags 5-20 before it
    (180-300); all fade after it, to within .06 of 0 at the end. ESN: grows at every lag, to +.68 / +1.02 / +1.34 /
    +1.40 / +1.40 at the end (lag 20 +.50 at 320); its rise grows with the lag at all 114 checkpoints.
  tau_1: Trained RNN 2.3 (initial) -> 57 (80) -> 18.2 (880, shortest after 100) -> 22 (1,680) -> 55 (4,200) -> 94-111
    at the end (QR estimate 94 [91.5, 97.4], drawn; frame-free bound -.00898 [-.00927, -.00867]: tau_1 <= 111.4); ESN
    2.3 -> 2.8. Within-episode growth resolved above 0 at 60-80 and at every checkpoint from 3,000 (point estimate
    positive from 2,800); +.016 at the end; the ESN's never above 0 (at most -.324).
  One-step expansion log rho(J_t) (114 checkpoints each): Trained RNN most entangled fifth > 0 at 80, 560-1,200 and
    1,360, max +.0156 [+.0113, +.0198] at 800; > 0 while the mean < 0 (both bounds) at 560-640, 880-1,200, 1,360; the
    mean > 0 at 60-80 (+.0060 [+.0047, +.0073] at 80; least entangled fifth +.0104), < 0 at 10-40, 100-640 and
    880-39,996; at the end mean -.0101 [-.0106, -.0095], most / least entangled fifth -.0062 / -.0136. Share of steps
    with log rho > 0 at 880: .431 in the most, .210 in the least entangled fifth. ESN mean -.4176 (40) to -.3399
    (end), < 0 at all 114. Footnote (not drawn): g_t on the step after a flip > 0 on average at 800-5,000, fifths 4-5
    at 720-5,000 (at 720: +.0131 / +.0131, fifth 1 -.0102; most - least +.0233 [+.0113, +.0332]); over lags 1-4 /
    5-20 / 21-29 every fifth < 0 at all 114 (largest -.0571 / -.0269 / -.0159).
  Lesion, beyond each latent's own dims: core answers +.13 [+.11, +.15] early (800-1,600), -.01 [-.03, +.01] late
    (>= 2,400), resolved above 0 at 320-3,200 only; corner answers +.35 / +.69 [+.62, +.77], above 0 from 320 on
    (+.020 [+.007, +.036] at 320 vs core +.058 [+.008, +.110]; +.052 vs +.050 at 480; +.197 vs +.110 at 800); holding
    +.03 / +.29 [+.27, +.32]; coupling +.04 / +.12 [+.10, +.13]. Late, against random dims of the same variance +.12 /
    +.40 / +.25 / +.24 (core / corner / holding / coupling); against both latents' main-effect dims -.17 / +.74 / +.22
    / +.03 (the footnote).
  Swaps: all g weights above random in every drawn cell except coupling 4,000-11,200 (+.10 [-.00, +.20]); of the 30
    drawn points of the other classes none is resolved above 0 and 22 are resolved below. Exceptions outside the drawn
    classes / baselines (td_tt_swaps ANSWER, quoted): against a random 2M of W, W_hh f+i exceed on answers in
    1,600-4,000 (+.11 [.03, .18]), through W_hh i (+.08 [.01, .16] over a random 1M of W_hh).
  Memory content: restart up to .160 (1,760; .150 [.137, .163] at 1,680) vs Bayes .031; same belief at most .016
    (1,680) vs Bayes's matched null .004.
  Pairs, least -> most entangled fifth: Trained RNN +.081 -> +.732, Bayes +.378 -> +.838, ESN +.914 -> +.949.

CAVEATS (the source modules' own, kept in mind for the wording): one seed and one probe world, intervals resample
  environments, not seeds. b: L(w) is a ceiling (another cell's history misleads Bayes too), read only against Bayes
  under the identical substitution; adjacent windows' intervals overlap (only pairs 4 or more steps apart are ordered).
  b: the onsets' dates depend on the donor scheme and are not tied to passing naive (td_tt_memlen NULLS a, b).
  d: the 160-400-episode spikes follow the checkpoint grid's phase (td_tt_flips RESULTS 6); the sweep stores lags 1, 2,
  5, 10, 20 only (all drawn). e: tau_1 describes the state driven beyond the 30 trained steps (within episodes
  td_tt_chaos uses tau_flip); the networks' exponents are in the Euclidean (u, c) chart, Bayes's and naive's in log
  coordinates (td_tt_chaos T1); the strip is transient amplification along some direction, not chaos (M1a). f: log
  rho > 0 is potential expansion (a direction the step map would expand were the same step repeated), not realised
  growth: on average, actual perturbations grow only on the step after a flip (the footnote; single flips can grow
  later: P(g_t > 0) up to .175 in the most entangled fifth at lags 5-20, td_tt_chaos_ent); 'entangled steps expand' is the
  fifth's mean, not every step (.431 of the most entangled steps expand at 880); the fifths' contrast is partly the
  step's own observation and partly the states at which observations are entangled (td_tt_chaos_ent counterfactual:
  own part +.007 to +.013 of +.022 to +.032 at 360-1,200 episodes; quoted, not recomputed), and the footnote's lag-1
  lead of the more entangled steps at 720 goes with the state, not the step's own observation (td_tt_chaos_ent (3):
  'state' contrast > 0 at 120-800, 'obs' not at 720; quoted); the Euclidean stretch log sigma_max(J_t) is
  chart-dependent (td_tt_chaos_ent CORRECTIONS) and no longer used here.
  Mechanisms a: against both latents' main-effect dims the core answers are not specially hit; b: the read-in /
  read-out have no random counterpart, and over longer stretches one set's change works only together with the rest
  (td_tt_swaps finding 10); the claim is scoped to the drawn classes and baselines (exceptions in the footnote);
  c: 'mostly', not entirely: the same-belief swap exceeds Bayes's matched null by +.007-.012 at 960-3,000
  (td_tt_memcuts RESULTS 3); d: observational (matched), and for Bayes the entanglement dependence is partly built in
  (the entanglement is computed from Bayes's beliefs).
SCOPES (light recheck, 11-02): the flip's growth peaks and fades at every lag, but lags 5-20 peak before passing naive and
  lags 1-2 at 1,200-1,360 (after it); at the end it is about 0 at every lag (lags 5 and 10 resolved slightly below 0) and
  below Bayes's at lags 2-20 (lag 1: +.053 [+.038, +.067] vs +.064, not resolved); panel c has no ESN (its per-step
  accuracy is not recorded), so 'the ESN shows none of these developments' covers b, d and e only.
CHECKS: numbers() reproduces every quoted number from the caches, except the td_tt_swaps exceptions and
  td_tt_chaos_ent's counterfactual split and state-level reading (CAVEATS f), quoted from their verified modules;
  panel c's reach equals td_tt_memcuts.uncut_reach(5) at the 46 shared checkpoints (max |d| 0); the cached chaos_ent
  series are asserted complete and current (td_tt_chaos_ent.summarize's own test: checkpoint and log rho files,
  layout version, measures, groupings; log rho finite at every checkpoint), and f's footnote claims are asserted at
  draw time (the first fifths whose lag-1 growth is > 0 are the most entangled ones; every later lag group < 0 in
  every fifth); f's values equal td_tt_chaos_ent's RESULTS (1) and (2); the count-code window from the
  count-preserving growth equals the index rule's (20-40 on, off at 100); pyflakes clean; both figures viewed whole
  and in native-resolution crops, and checked at savefig time for text or legends on data, text on text and text off
  the canvas (none; td_tt_processes.png again after the 11-03 panel f update).

Run from the repository root (CPU only, ~1 min; nothing is written except the two PNGs):
    CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
        OPENBLAS_NUM_THREADS=2 /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/td_tt_processes.py [numbers] [figure]"""
import os
import sys
import warnings
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")         # CPU only: every series is read from caches
import td_tt_gap as gap                                                                       # noqa: E402

BLUE, ORANGE, NAIVE_C = "#1f77b4", "#ff7f0e", "0.55"
EARLY_SPAN, LATE_SPAN = (774., 1575.), (2747., 11103.)    # td_tt_mechanism / td_tt_timeline half-way ranges
EARLY_C, LATE_C = "#ece4f5", "#e3f0dc"
CC_ON_C, CC_OFF_C = "#f6e7a6", "#fbf4d6"                  # count code on / switching off
FAM_C = {"core answers": "#6a3d9a", "corner answers": "#c51b7d", "holding": "#3a7a19", "coupling": "#b2182b",
         "answers": "#6a3d9a", "loss": "#8c510a"}         # purple = the answers family (td_tt_timeline / td_tt_swaps)
SEP_C = "#c6dbef"                                         # window separators (pale blue: grey is reserved for naive)
XMIN, XMAX = 8., 7e4
NB = gap.N_BOOT
_M = {}


def _iv(b, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(b, 5, axis=axis), np.nanpercentile(b, 95, axis=axis)


def _cross_iv(b):
    """5-95% of crossing-type resamples, NaN = never (after everything): td_tt_memcuts._interval's rule"""
    b = np.where(np.isnan(b), np.inf, b)
    return np.percentile(b, 5, method="lower"), np.percentile(b, 95, method="higher")


def memo(f):
    def g():
        if f.__name__ not in _M:
            _M[f.__name__] = f()
        return _M[f.__name__]
    g.__name__, g.__doc__ = f.__name__, f.__doc__
    return g


# ═══════════════════════════════════════════════════════════════════
# The drawn series (each recomputed from a verified module's caches / functions)
# ═══════════════════════════════════════════════════════════════════

@memo
def acc():
    """a: goal accuracy at T (td_tt_gap.load('jumps'), rows 'goal accuracy at T' of the Trained RNN and the ESN)"""
    z = gap.load("jumps")
    nm, rn = [str(v) for v in z["names"]], [str(v) for v in z["run"]]
    out = dict(x=z["x"], src="td_tt_gap.load('jumps') ['goal accuracy at T']")
    for run in ("trained", "echo"):
        i = [k for k in range(len(nm)) if nm[k] == "goal accuracy at T" and rn[k] == run][0]
        out[run] = (z["value"][i], z["lo"][i], z["hi"][i])
        if run == "trained":
            out.update(naive=float(z["naive"][i]), bayes=float(z["bayes"][i]),
                       c0=(float(z["c0"][i]), float(z["c0_lo"][i]), float(z["c0_hi"][i])),
                       c50=(float(z["c50"][i]), float(z["c50_lo"][i]), float(z["c50_hi"][i])))
    return out


@memo
def horizon():
    """b: random-cell memory-cut onsets per window (td_tt_memlen.loss / onset_core, engine donors), the ESN's, the
    trend per step and the pairwise order; the other donor schemes' trends (the note on the panel)"""
    import td_tt_memlen as tm
    s = tm.SOURCES.index("random")
    d, de = tm.loss("trained"), tm.loss("echo")
    o = tm.onset_core(d["x"], d["net"][:, s], d["net_b"][:, s], d["bayes"][s], d["bayes_b"][s], .5)
    oe = tm.onset_core(de["x"], de["net"][:, s], de["net_b"][:, s], de["bayes"][s], de["bayes_b"][s], .5)
    out = dict(w=np.arange(1, tm.WMAX + 1), o=o, oe=oe, n_ck=len(d["x"]), n_ck_e=len(de["x"]),
               src="td_tt_memlen.loss + onset_core (random cell, engine donors; level 1/2 of Bayes's L(w))")
    v, b = tm.onset_trend(o)
    out["trend"] = (10 ** v, *(10 ** np.nanpercentile(b, [5, 95])))
    nb_, na_, n_, _ = tm.pair_order(o)
    out["pairs"] = (nb_, na_, n_)
    out["schemes"] = {}
    for sc in tm.SCHEMES:
        rl = tm.random_loss(sc)
        os_ = tm.onset_core(rl["x"], rl["net"], rl["net_b"], rl["bayes"], rl["bayes_b"], .5)
        v, b = tm.onset_trend(os_)
        nb2, na2, n2, _ = tm.pair_order(os_)
        out["schemes"][sc] = dict(p=os_["p"], trend=(10 ** v, *(10 ** np.nanpercentile(b, [5, 95]))),
                                  pairs=(nb2, na2, n2))
    return out


@memo
def speed():
    """c: the step at which the uncut network first reaches Bayes's goal accuracy at step 5 (td_tt_memcuts.uncut_reach's
    rule and Bayes's level from its cache) on td_tt_memlen's intact per-step accuracies (all 114 checkpoints);
    checked against td_tt_memcuts.uncut_reach(5) at the shared checkpoints"""
    import td_tt_memcuts as mc
    import td_tt_memlen as tm
    u, rc = mc.uncut_steps(), mc.uncut_reach(5)
    D = tm.load_run("trained")
    S = D["accg"][:, 0]                                                     # (K, NE, T) goal right, intact
    C = gap.boot_counts()
    T = S.shape[-1]
    net = S.sum(1) / (gap.N_ENV * gap.PER_ENV)
    net_b = np.einsum("be,ket->bkt", C, S) / (C.sum(1) * gap.PER_ENV)[:, None, None]

    def reach(y, lev):                                                      # = td_tt_memcuts.uncut_reach's reach()
        past = y >= lev[:, None]
        has = past.any(1)
        j = np.argmax(past, 1)
        y0 = y[np.arange(len(y)), np.maximum(j - 1, 0)]
        y1 = y[np.arange(len(y)), j]
        with np.errstate(invalid="ignore", divide="ignore"):
            st = np.where(j == 0, 1., j + (lev - y0) / (y1 - y0))
        return np.where(has, st, np.nan)
    K = len(D["x"])
    v = reach(net, np.full(K, u["bayes"][4]))
    b = reach(net_b.reshape(NB * K, T), np.repeat(u["bayes_boot"][:, 4], K)).reshape(NB, K)
    lo, hi = np.array([_cross_iv(b[:, k]) for k in range(K)]).T
    dmax, n_sh = 0., 0
    for i, xx in enumerate(rc["x"]):                                        # reproduction check (46 checkpoints)
        k = int(np.flatnonzero(D["x"] == xx)[0])
        n_sh += 1
        a_, c_ = v[k], rc["step"][i]
        dmax = max(dmax, 0. if (np.isnan(a_) and np.isnan(c_)) else abs(a_ - c_))
    return dict(x=D["x"], step=v, lo=lo, hi=hi, share=np.isfinite(b).mean(0), level=float(u["bayes"][4]),
                bayes_steps=u["bayes"], check=(n_sh, dmax),
                src="td_tt_memcuts.uncut_reach rule on td_tt_memlen.load_run('trained')['accg'] (intact); Bayes's level "
                    "td_tt_memcuts.uncut_steps()['bayes'][4]")


@memo
def flips():
    """d: the chaos sweep's paired rise at every stored lag (1, 2, 5, 10, 20; td_tt_chaos_combined.sweep, sweep_refs,
    sweep_last_above) and td_tt_flips' count classes (flips/combined.npz) for the count-code window: on where the
    count-preserving 2-bit growth is below a tenth of the count-changing growth; switching off until the
    count-preserving growth is within a tenth of it (td_tt_flips reads the switch from this growth, not from the
    count-code index)"""
    import td_tt_chaos_combined as cc
    lags = tuple(cc.SW_LAGS)
    S, E, R = cc.sweep("trained"), cc.sweep("echo"), cc.sweep_refs()
    F = cc.flips()
    xc = F["trained_cc_x"]
    keep, chg, idx = (F[f"trained_count_{g}"][:, 4] for g in ("preserving", "changing", "index"))
    on = xc[(keep < .1 * chg) & (xc > gap.X0)]
    off = float(xc[np.flatnonzero((xc > on.max()) & (keep >= .9 * chg))[0]])
    return dict(lags=lags, x=S["x"], x_e=E["x"],                                       # rises: (lag, v / lo / hi, K)
                trained=np.transpose(S["rise"], (1, 2, 0)), echo=np.transpose(E["rise"], (1, 2, 0)),
                bayes=np.asarray(R["bayes"], float), naive=np.asarray(R["naive"], float),
                last_above=[cc.sweep_last_above(float(R["bayes"][i]), lag=L) for i, L in enumerate(lags)],
                cc_x=xc, cc_keep=keep, cc_chg=chg, cc_idx=idx, cc_on=(float(on.min()), float(on.max())), cc_off=off,
                src="td_tt_chaos_combined.sweep / sweep_refs (lags 1, 2, 5, 10, 20); count code: flips/combined.npz "
                    "trained_count_preserving / _changing (lags 18-21)")


@memo
def decay():
    """e: tau_1 = -1/lambda_1 (td_tt_chaos.lam_curve) and the within-episode amplification (td_tt_chaos.lyap_load amp
    over steps 1-30, environment means by td_tt_chaos._env_boot)"""
    import td_tt_chaos as tc
    out = dict(src="td_tt_chaos.lam_curve (tau_1 = -1/lambda_1); amp: lyap_load(run)['amp'][..., 0] via _env_boot")
    for run in ("trained", "echo"):
        z = tc.lyap_load(run)
        x, v, lo, hi, is_long = tc.lam_curve(run, z)
        a, alo, ahi = tc._env_boot(z["amp"][..., 0], z["rows"])
        out[run] = dict(x=x, lam=v, lam_lo=lo, lam_hi=hi, is_long=is_long, tau=-1. / v, tau_lo=-1. / lo,
                        tau_hi=-1. / hi, amp=a, amp_lo=alo, amp_hi=ahi, x_amp=z["x"])
    zl = tc.long_load("trained")                       # not drawn: the frame-free bound at the end (the 94-111 bracket)
    out["bound_end"] = (float(zl["x"][-1]), *tc._env_boot(zl["upper"][-1], zl["rows"]))
    return out


@memo
def entangled():
    """f: one-step expansion log rho(J_t) per entanglement fifth and over all steps (td_tt_chaos_ent series_<run>.npz,
    run_curves(Z, 'lrho'): fifths v / lo / hi, all-unit mean o / olo / ohi); not drawn (the footnote): the flip's
    growth on the first step after the flip (run_curves 'g:1') and over the later lag groups ('g:1-4', 'g:5-20',
    'g:21-29'). The cached series must be complete and current: td_tt_chaos_ent.summarize's own test (checkpoint and
    log rho files, layout version, measures, groupings), asserted here instead of rebuilding (nothing is written)"""
    import td_tt_chaos_ent as ce
    out = dict(src="td_tt_chaos_ent series_<run>.npz + run_curves(Z, 'lrho') (main grouping: fifths 1 and 5, the "
                   "all-unit mean); footnote: run_curves(Z, 'g:1' / 'g:1-4' / 'g:5-20' / 'g:21-29')")
    for run in ("trained", "echo"):
        Z = dict(np.load(os.path.join(ce.OUT, f"series_{run}.npz")))
        eps = [e for e in ce.ti.checkpoints(run) if os.path.exists(ce.ck_file(run, e))]
        ext = [e for e in eps if os.path.exists(ce.ext_file(run, e))]
        assert (list(Z["raw"]) == eps and list(Z["ext_raw"]) == ext and int(Z["version"]) == ce.SERIES_VERSION
                and list(Z["groups"]) == list(ce.GROUPS) and list(Z["meas"]) == list(ce.MEAS)), \
            f"series_{run}.npz out of date (td_tt_chaos_ent.py numbers rebuilds it)"
        c = ce.run_curves(Z, "lrho")
        assert np.isfinite(c["o"]).all(), f"log rho missing at some {run} checkpoints"
        p = ce.run_curves(Z, "lrho_pos")                                    # not drawn: the share of expanding steps
        out[run] = dict(x=Z["x"], v=c["v"], lo=c["lo"], hi=c["hi"], o=c["o"], olo=c["olo"], ohi=c["ohi"],
                        pos=p["v"], pos_o=p["o"], n_all=len(ce.ti.checkpoints(run)))
    Z = dict(np.load(os.path.join(ce.OUT, "series_trained.npz")))
    out["lag1"] = ce.run_curves(Z, "g:1")
    out["later"] = {lg: ce.run_curves(Z, f"g:{lg}") for lg in ("1-4", "5-20", "21-29")}
    out["later_pos"] = {lg: ce.run_curves(Z, f"g:{lg}_pos")["v"] for lg in out["later"]}     # P(g_t > 0) (CAVEATS f)
    return out


@memo
def lesion():
    """a (mechanisms): specific damage of joint5 against r0 / r1 (td_tt_lesions.specific(L, family_effects(L,
    PANEL_ROWS), 'joint5', 'additive')), per checkpoint and panel, and the early / late window means"""
    import td_tt_lesions as tl
    L = tl.load_run("trained")
    PE = tl.family_effects(L, tl.PANEL_ROWS)
    s, sb = tl.specific(L, PE, "joint5", "additive")
    lo, hi = tl.iv(sb)
    win, other = {}, {}
    for w in ("early", "late"):
        m = tl.window(L["x"], w)
        win[w] = (np.nanmean(s[m], 0), *tl.iv(np.nanmean(sb[m], 0)))
    for kind in ("vm", "add5"):                        # the footnote's other controls (window means, seeds averaged)
        s2, sb2 = tl.specific(L, PE, "joint5", kind)
        other[kind] = {}
        for w in ("early", "late"):
            m = tl.window(L["x"], w) & np.isfinite(s2[:, 0])
            other[kind][w] = (np.nanmean(s2[m], 0), *tl.iv(np.nanmean(sb2[m], 0)))
    return dict(x=L["x"], panels=tl.PANELS, s=s, lo=lo, hi=hi, win=win, other=other, intact=PE["F_int"],
                src="td_tt_lesions.specific(L, family_effects(L, PANEL_ROWS), 'joint5', 'additive'); footnote: "
                    "'vm' (random dims, same variance) and 'add5' (both latents' main-effect dims) window means")


@memo
def swaps():
    """b (mechanisms): td_tt_swaps.excess_cells (the cells td_tt_swaps.png draws)"""
    import td_tt_swaps as sw
    D = sw.collect()
    W = sw.decompose(D)
    E = sw.excess_cells(W, D)
    return dict(E=E, fams=sw.FIG_FAMS, wins=sw.FIG_WIN, labels={wi: sw.win_label(wi) for wi in sw.FIG_WIN},
                x0={wi: float(W[wi]["x0"]) for wi in sw.FIG_WIN}, x1={wi: float(W[wi]["x1"]) for wi in sw.FIG_WIN},
                n_pairs=[W[wi]["n_pairs"] for wi in sw.FIG_WIN], src="td_tt_swaps.excess_cells(decompose(collect()))")


@memo
def content():
    """c (mechanisms): the restart (td_tt_memcuts.dep 'w5_start') and the belief-matched swap (td_tt_memcuts._seed_mean
    of the 3 pools, against Bayes's matched null)"""
    import td_tt_memcuts as mc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rs = mc.dep("trained", "w5_start", "acc_goal_T")
        bm = mc._seed_mean(dict(mc.CTL_GROUPS)["belief-matched other cell"])
    ok = np.isfinite(bm["v"])
    return dict(rs=rs, bm=dict(x=bm["x"][ok], v=bm["v"][ok], lo=bm["lo"][ok], hi=bm["hi"][ok], bayes=bm["bayes"],
                               bmin=bm["bmin"], bmax=bm["bmax"]),
                src="td_tt_memcuts.dep('trained', 'w5_start', 'acc_goal_T'); _seed_mean(belief-matched other cell)")


@memo
def pairs():
    """d (mechanisms): the step- and bin-matched fifth means of the pairs' d_H change (pairs_combined.npz ent5_std via
    td_tt_chaos_combined._iv) and the most - least contrasts (pstat 'ent5|dh|step+bin|<agent>')"""
    import td_tt_chaos_combined as cc
    P = cc.pairs()
    cm = list(P["cmeas"]).index("dh")
    out = dict(edges=P["ent5_edges"], n_units=P["n_units"], n_pairsteps=P["n_pairsteps"],
               src="td_tt_chaos_combined.pairs()['ent5_std'] (dh) + pstat('ent5|dh|step+bin|<agent>')")
    for k in ("bayes", "trained", "echo"):
        ia = list(P["agents"]).index(k)
        out[k] = cc._iv(P["ent5_std"][ia, cm].T)
        out[k + "_contrast"] = cc.pstat(f"ent5|dh|step+bin|{k}")
    return out


# ═══════════════════════════════════════════════════════════════════
# numbers
# ═══════════════════════════════════════════════════════════════════

def _row(x, v, lo=None, hi=None, f="{:+.3f}", per=6):
    """every point of a drawn curve, `per` per line"""
    cells = []
    for i in range(len(x)):
        c = f"{x[i]:,.0f}: " + (f.format(v[i]) if np.isfinite(v[i]) else "never")
        if lo is not None:
            l_ = f.format(lo[i]) if np.isfinite(lo[i]) else "-"
            h_ = f.format(hi[i]) if np.isfinite(hi[i]) else "never"
            c += f" [{l_}, {h_}]"
        cells.append(c)
    for i in range(0, len(cells), per):
        print("     " + "; ".join(cells[i:i + per]))


def _fx(v):
    return "never" if not np.isfinite(v) else f"{v:,.0f}"


def numbers():
    """every drawn value of both figures, with its source"""
    print("=" * 100 + "\ntd_tt_processes.png\n" + "=" * 100)
    A = acc()
    print(f"\na  {A['src']}; {len(A['x'])} checkpoints")
    print(f"   naive {A['naive']:.4f}, Bayes {A['bayes']:.4f}; Trained RNN passes naive (c0) {A['c0'][0]:,.0f} "
          f"[{A['c0'][1]:,.0f}-{A['c0'][2]:,.0f}] (the dotted line, every panel), half-way (c50) {A['c50'][0]:,.0f} "
          f"[{A['c50'][1]:,.0f}-{A['c50'][2]:,.0f}]")
    for run in ("trained", "echo"):
        print(f"   {run}:")
        _row(A["x"], *A[run])
    print(f"   spans (td_tt_timeline half-way ranges): answers / splitting {EARLY_SPAN[0]:,.0f}-{EARLY_SPAN[1]:,.0f}, "
          f"holding / coupling {LATE_SPAN[0]:,.0f}-{LATE_SPAN[1]:,.0f}")
    H = horizon()
    o, oe = H["o"], H["oe"]
    print(f"\nb  {H['src']}; Trained RNN {H['n_ck']} checkpoints, ESN {H['n_ck_e']}")
    for w in range(len(H["w"])):
        print(f"   w = {w + 1:2d}: onset {_fx(o['p'][w])} [{_fx(o['lo'][w])}-{_fx(o['hi'][w])}] "
              f"({o['frac'][w]:.0%} of resamples cross)")
    print("   ESN: " + "; ".join(f"w = {w + 1} {_fx(oe['p'][w])} [{_fx(oe['lo'][w])}-{_fx(oe['hi'][w])}] "
                                 f"({oe['frac'][w]:.0%})" for w in range(4)) + "; drawn: w = 1 only")
    t = H["trend"]
    print(f"   per extra step x{t[0]:.3f} [x{t[1]:.3f}, x{t[2]:.3f}]; pairs ordered shorter-first {H['pairs'][0]} of "
          f"{H['pairs'][2]}, reversed {H['pairs'][1]}")
    for sc, r in H["schemes"].items():
        print(f"   other donor scheme '{sc}' (note only, not drawn): x{r['trend'][0]:.3f} [x{r['trend'][1]:.3f}, "
              f"x{r['trend'][2]:.3f}] per step, ordered {r['pairs'][0]} / reversed {r['pairs'][1]} of {r['pairs'][2]}; "
              f"w = 1 / 15 at {_fx(r['p'][0])} / {_fx(r['p'][-1])}")
    Sp = speed()
    print(f"\nc  {Sp['src']}")
    print("   Bayes's goal accuracy at steps 1-10: " + " ".join(f"{v:.4f}" for v in Sp["bayes_steps"][:10]) +
          f"; level (step 5) {Sp['level']:.4f}")
    print(f"   reproduction: equals td_tt_memcuts.uncut_reach(5) at {Sp['check'][0]} shared checkpoints, max |d| "
          f"{Sp['check'][1]:.2g}")
    nev = Sp["x"][~np.isfinite(Sp["step"])]
    print(f"   never within 30 steps (the top bar): {len(nev)} checkpoints, {nev.min():,.0f}-{nev.max():,.0f}")
    ok = np.isfinite(Sp["step"])
    print("   step [5-95%] (resamples: share reaching):")
    _row(Sp["x"][ok], Sp["step"][ok], Sp["lo"][ok], Sp["hi"][ok], f="{:.2f}")
    print("   share of resamples reaching at the first drawn checkpoints: " + ", ".join(
        f"{x:,.0f} {s:.3f}" for x, s in zip(Sp["x"][ok][:4], Sp["share"][ok][:4])))
    Fl = flips()
    lg = Fl["lags"]
    print(f"\nd  {Fl['src']}")
    print("   exact observers' rise at lags " + " / ".join(map(str, lg)) + ": Bayes " +
          " / ".join(f"{v:+.4f}" for v in Fl["bayes"]) + " (drawn: lag 20), naive " +
          " / ".join(f"{v:+.4f}" for v in Fl["naive"]))
    for i, L in enumerate(lg):
        for run, x in (("trained", Fl["x"]), ("echo", Fl["x_e"])):
            v, lo, hi = Fl[run][i]
            k = int(np.argmax(v))
            la = Fl["last_above"][i]
            print(f"   {run}, lag {L}: peak {v[k]:+.3f} at {x[k]:,.0f}; end {v[-1]:+.3f} [{lo[-1]:+.3f}, {hi[-1]:+.3f}]"
                  + (f" (Bayes {Fl['bayes'][i]:+.3f}); last above Bayes's level at {la[0]:,.0f} "
                     f"[{la[1]:,.0f}-{la[2]:,.0f}]" if run == "trained" else ""))
            _row(x, v, lo, hi)
    tr_end, es = Fl["trained"][:, 0, -1], Fl["echo"][:, 0, :]
    print(f"   Trained RNN at the end below Bayes's level at every lag: {bool((tr_end < Fl['bayes']).all())}; ESN rise "
          f"non-decreasing in the lag at {int((np.diff(es, axis=0) >= 0).all(0).sum())} of {es.shape[1]} checkpoints")
    print(f"   count code (lags 18-21, 2-bit flips): on where count-preserving growth < .1 x count-changing: "
          f"{Fl['cc_on'][0]:,.0f}-{Fl['cc_on'][1]:,.0f}; switching off until count-preserving growth >= .9 x "
          f"count-changing: {Fl['cc_off']:,.0f} (td_tt_flips: the count code switches off between 40 and 100 episodes)")
    for x, a, b, c in zip(Fl["cc_x"][:7], Fl["cc_keep"][:7], Fl["cc_chg"][:7], Fl["cc_idx"][:7]):
        print(f"     {x:,.0f}: preserving {a:+.3f}, changing {b:+.3f} (preserving / changing {a / b:.3f}); "
              f"count-code index {c:+.3f}")
    Dc = decay()
    print(f"\ne  {Dc['src']}")
    for run in ("trained", "echo"):
        r = Dc[run]
        print(f"   {run} tau_1 [5-95%] (lambda_1 from the {'1,500-step' if r['is_long'].all() else '300-step'} "
              f"estimate at {r['is_long'].sum()} / {len(r['x'])} checkpoints):")
        _row(r["x"], r["tau"], r["tau_lo"], r["tau_hi"], f="{:.1f}")
        print(f"   {run} lambda_1 max {np.max(r['lam']):+.4f} (every checkpoint < 0: {bool((r['lam'] < 0).all())})")
    r = Dc["trained"]
    k = np.argmin(np.where(r["x"] > 100, r["tau"], np.inf))
    print(f"   Trained RNN shortest tau_1 after 100 episodes: {r['tau'][k]:.1f} at {r['x'][k]:,.0f}")
    xb, bv, blo, bhi = Dc["bound_end"]
    print(f"   not drawn: frame-free bound at the end ({xb:,.0f}) {bv:+.5f} [{blo:+.5f}, {bhi:+.5f}] -> tau_1 <= "
          f"{-1 / bv:.1f}: tau_1 at the end {r['tau'][-1]:.0f}-{-1 / bv:.0f} "
          "(td_tt_chaos.long_load('trained')['upper'])")
    res = r["amp_lo"] > 0
    print("   within-episode growth (1/30) log sigma_1, resolved above 0 (the strip) at: " + _ranges(r["x_amp"], res) +
          f"; point estimate > 0 for good from {r['x_amp'][np.flatnonzero(r['amp'] <= 0)[-1] + 1]:,.0f}")
    _row(r["x_amp"], r["amp"], r["amp_lo"], r["amp_hi"], f="{:+.4f}")
    print(f"   ESN within-episode growth max {np.max(Dc['echo']['amp']):+.4f} (never > 0)")
    En = entangled()
    print(f"\nf  {En['src']}")
    for run in ("trained", "echo"):
        r = En[run]
        print(f"   {run}: log rho at {len(r['x'])} of {r['n_all']} checkpoints; all steps (the mean):")
        _row(r["x"], r["o"], r["olo"], r["ohi"], f="{:+.4f}")
        if run == "trained":                                                # the ESN: its mean only (drawn)
            for q, nm in ((4, "most entangled fifth"), (0, "least entangled fifth")):
                print(f"   {run}, {nm}:")
                _row(r["x"], r["v"][:, q], r["lo"][:, q], r["hi"][:, q], f="{:+.4f}")
    r = En["trained"]
    xf, f5, f1 = r["x"], r["v"][:, 4], r["v"][:, 0]
    k5 = int(np.argmax(f5))
    print(f"   Trained RNN: most entangled fifth > 0 (5% bound) at {_ranges(xf, r['lo'][:, 4] > 0)}; max {f5[k5]:+.4f} "
          f"[{r['lo'][k5, 4]:+.4f}, {r['hi'][k5, 4]:+.4f}] at {xf[k5]:,.0f}; > 0 while the mean < 0 (both bounds) at "
          f"{_ranges(xf, (r['lo'][:, 4] > 0) & (r['ohi'] < 0))}")
    print(f"   Trained RNN: mean > 0 (5% bound) at {_ranges(xf, r['olo'] > 0)}; < 0 (95% bound) at "
          f"{_ranges(xf, r['ohi'] < 0)}; least entangled fifth > 0 at {_ranges(xf, r['lo'][:, 0] > 0)}; at the end mean "
          f"{r['o'][-1]:+.4f}, most {f5[-1]:+.4f}, least {f1[-1]:+.4f} entangled fifth")
    k88 = int(np.flatnonzero(xf == 880.)[0])
    print("   Trained RNN, not drawn (CAVEATS f): share of steps with log rho > 0 at 880, fifths 1-5 " + " / ".join(
        f"{v:.3f}" for v in r["pos"][k88]) + f", all steps {r['pos_o'][k88]:.3f}; at the end fifths 1 / 5 "
          f"{r['pos'][-1, 0]:.3f} / {r['pos'][-1, 4]:.3f}")
    es_ = En["echo"]
    print(f"   ESN: mean {es_['o'].min():+.4f} to {es_['o'].max():+.4f} ({es_['o'][-1]:+.4f} at the end); < 0 (95% "
          f"bound) at {int((es_['ohi'] < 0).sum())} of {len(es_['x'])} checkpoints; its most entangled fifth (not "
          f"drawn): largest 95% bound {np.max(es_['hi'][:, 4]):+.4f}")
    g1 = En["lag1"]
    print("   footnote (not drawn): the flip's state-difference growth on the first step after the flip (g_t at lag 1) "
          "> 0 (5% bound) at: " + "; ".join(f"fifth {q + 1} {_ranges(xf, g1['lo'][:, q] > 0)}" for q in range(5))
          + f"; all steps {_ranges(xf, g1['olo'] > 0)}")
    k = int(np.flatnonzero((g1["lo"] > 0).any(1))[0])
    print(f"     first checkpoint with a fifth > 0: {xf[k]:,.0f}: fifths 1-5 " + " / ".join(
        f"{v:+.4f} [{lo:+.4f}]" for v, lo in zip(g1["v"][k], g1["lo"][k])) + " (value [5% bound]); most - least "
          f"{g1['c'][k]:+.4f} [{g1['clo'][k]:+.4f}, {g1['chi'][k]:+.4f}]")
    for lg, rl in En["later"].items():
        pp = En["later_pos"][lg][:, 4]
        print(f"     lags {lg}: every fifth < 0 (each 95% bound) at {int(np.all(rl['hi'] < 0, 1).sum())} of {len(xf)} "
              f"checkpoints; any fifth > 0 (5% bound) at {int((rl['lo'] > 0).any(1).sum())}; largest fifth value "
              f"{np.max(rl['v']):+.4f}; single flips (CAVEATS f): P(g_t > 0) in the most entangled fifth max "
              f"{np.max(pp):.3f} at {xf[int(np.argmax(pp))]:,.0f}")
    print("\n" + "=" * 100 + "\ntd_tt_mechanisms.png\n" + "=" * 100)
    Le = lesion()
    print(f"\na  {Le['src']}; drawn from 320")
    for j, p in enumerate(Le["panels"]):
        print(f"   {p}:")
        _row(Le["x"], Le["s"][:, j], Le["lo"][:, j], Le["hi"][:, j], f="{:+.3f}")
        print("     window means: " + "; ".join(f"{w} {Le['win'][w][0][j]:+.3f} [{Le['win'][w][1][j]:+.3f}, "
                                                f"{Le['win'][w][2][j]:+.3f}]" for w in ("early", "late")))
        for kind, nm in (("vm", "random dims, same variance"), ("add5", "both latents' main-effect dims")):
            print(f"     footnote, against {nm}: " + "; ".join(
                f"{w} {Le['other'][kind][w][0][j]:+.3f} [{Le['other'][kind][w][1][j]:+.3f}, "
                f"{Le['other'][kind][w][2][j]:+.3f}]" for w in ("early", "late")))
    Sw = swaps()
    print(f"\nb  {Sw['src']}; consecutive checkpoint pairs per window {Sw['n_pairs']}")
    n_other = n_below = n_above = 0
    for item, nm in Sw["fams"]:
        for wi in Sw["wins"]:
            c = Sw["E"][(item, wi)]
            print(f"   {nm:16s} {Sw['labels'][wi]:9s} change {c['tot']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] "
                  f"{'drawn' if c['drawn'] else 'NOT drawn'}: " + "; ".join(
                      f"{lab} {c[lab]['v']:+.2f} [{c[lab]['lo']:+.2f}, {c[lab]['hi']:+.2f}]" for lab in SW_SETS))
            if c["drawn"]:
                for lab in SW_SETS[1:]:
                    n_other += 1
                    n_below += c[lab]["hi"] < 0
                    n_above += c[lab]["lo"] > 0
    print(f"   other classes, drawn points: {n_other}; resolved below 0: {n_below}; above 0: {n_above}")
    Ct = content()
    rs, bm = Ct["rs"], Ct["bm"]
    print(f"\nc  {Ct['src']}")
    print(f"   restart, Bayes's 5-step window {rs['bayes']:.4f} [{rs['bayes_lo']:.4f}, {rs['bayes_hi']:.4f}]:")
    _row(rs["x"], rs["net"], rs["net_lo"], rs["net_hi"], f="{:.3f}")
    print(f"   same belief (3 pools' mean), Bayes's matched null {bm['bayes']:.4f} (pools {bm['bmin']:.4f}-"
          f"{bm['bmax']:.4f}):")
    _row(bm["x"], bm["v"], bm["lo"], bm["hi"], f="{:.4f}")
    Pa = pairs()
    print(f"\nd  {Pa['src']}; entanglement fifth edges " + " / ".join(f"{e:.3f}" for e in Pa["edges"]))
    for k in ("trained", "bayes", "echo"):
        v, lo, hi = Pa[k]
        c = Pa[k + "_contrast"]
        print(f"   {k}: " + "; ".join(f"fifth {q + 1} {v[q]:+.3f} [{lo[q]:+.3f}, {hi[q]:+.3f}]" for q in range(5)) +
              f"; most - least {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]")
    print("   naive: 0 exactly (td_tt_chaos T1)")


def _ranges(x, mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append(f"{x[i]:,.0f}" + ("" if j == i else f"-{x[j]:,.0f}"))
            i = j + 1
        else:
            i += 1
    return ", ".join(out) if out else "none"


SW_SETS = ("gate g", "W_hh f+i", "W_hh o", "W_ih")        # td_tt_swaps.FIG_SETS labels: g first, then the others


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

def _style():
    import matplotlib.pyplot as plt
    from main.utils import apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    return fs


def _xaxis(ax, lim=(XMIN, XMAX), label=True, fs=None):
    ax.set_xscale("log")
    ax.set_xlim(*lim)
    ticks = [t for t in (gap.X0, 1e2, 1e3, 1e4) if lim[0] <= t <= lim[1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0" if t == gap.X0 else f"{t / 1e3:g}k" if t >= 1e3 else f"{t:g}" for t in ticks])
    ax.minorticks_off()
    if label:
        ax.set_xlabel("training episodes", fontsize=fs["label"])


def _guides(ax, c0):
    ax.axvspan(*EARLY_SPAN, color=EARLY_C, lw=0, zorder=0)
    ax.axvspan(*LATE_SPAN, color=LATE_C, lw=0, zorder=0)
    ax.axvline(c0, color="k", lw=1., ls=(0, (1, 1.6)), zorder=1)


def _title(ax, letter, finding, formula, fs):
    ax.set_title(f"{letter}   {finding}\n{formula}", loc="left", fontsize=fs["label"] + 1.5, linespacing=1.45, pad=6)


def _band(ax, x, v, lo, hi, color, lw=2., ls="-", alpha=.22, z=3, **kw):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0, zorder=z - 1)
    return ax.plot(x, v, color=color, lw=lw, ls=ls, zorder=z, **kw)


def _halo(lw=3.):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=lw, foreground="w")]




def _strip_runs(x, mask):
    """[(x_first, x_last)] of the runs of consecutive checkpoints where mask holds"""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((x[i], x[j]))
            i = j + 1
        else:
            i += 1
    return out


def figure_processes():
    """-> figure_bin/td_tt_processes.png"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, NullLocator
    from main.utils import fig_path
    fs = _style()
    FL, FT, FN = fs["label"], fs["tick"], fs["label"]
    A, H, Sp, Fl, Dc, En = acc(), horizon(), speed(), flips(), decay(), entangled()
    c0 = A["c0"][0]
    PW, PH, LM, GX, RM = 3.7, 2.25, .78, .92, .42
    TOP, TT, GY, BOT = 1.2, .62, .5, 1.95                    # suptitle + legend; titles; row gap; x label + footnote
    W = LM + 3 * PW + 2 * GX + RM
    Hh = TOP + 2 * (TT + PH) + GY + BOT
    fig = plt.figure(figsize=(W, Hh))
    axs = {}
    for r in range(2):
        for c in range(3):
            y0 = Hh - TOP - TT - PH - r * (PH + GY + TT)
            axs["abcdef"[3 * r + c]] = fig.add_axes([(LM + c * (PW + GX)) / W, y0 / Hh, PW / W, PH / Hh])
    for k, ax in axs.items():
        _guides(ax, c0)
        _xaxis(ax, label=k in "def", fs=fs)
        ax.spines[["top", "right"]].set_visible(False)
    # a: accuracy
    ax = axs["a"]
    for run, col in (("echo", ORANGE), ("trained", BLUE)):
        _band(ax, A["x"], *A[run], col)
    ax.axhline(A["bayes"], color="k", lw=1.3, zorder=2)
    ax.axhline(A["naive"], color=NAIVE_C, lw=1.3, zorder=2)
    ax.text(11, A["bayes"] + .015, f"Bayes {A['bayes']:.3f}".replace("0.", "."), fontsize=FT, va="bottom")
    ax.text(11, A["naive"] + .015, f"naive {A['naive']:.3f}".replace("0.", "."), fontsize=FT, va="bottom",
            color="0.35")
    ax.set_ylim(.05, .86)
    ax.set_yticks([.1, .3, .5, .7])
    ax.set_yticklabels([".1", ".3", ".5", ".7"])
    ax.set_ylabel("goal accuracy at $T$", fontsize=FL)
    _title(ax, "a", "The answers pass naive at 835; the ESN's never do",
           r"$P(\hat{a}_T = r_{\mathrm{goal}})$, 10,000 probe episodes", fs)
    # b: memory horizon staircase
    ax = axs["b"]
    o, oe, w = H["o"], H["oe"], H["w"]
    for i in range(len(w)):
        hi_ = o["hi"][i] if np.isfinite(o["hi"][i]) else XMAX
        ax.plot([o["lo"][i], hi_], [w[i], w[i]], color=BLUE, lw=1.6, solid_capstyle="butt", zorder=3)
    ax.plot(o["p"], w, color=BLUE, lw=1., zorder=4, alpha=.6)
    ax.plot(o["p"], w, ls="none", marker="s", ms=4.6, mfc=BLUE, mec="k", mew=.5, zorder=5)
    if np.isfinite(oe["p"][0]):
        hi_ = oe["hi"][0] if np.isfinite(oe["hi"][0]) else XMAX
        ax.plot([oe["lo"][0], hi_], [1, 1], color=ORANGE, lw=1.6, solid_capstyle="butt", zorder=3)
        ax.plot([oe["p"][0]], [1], ls="none", marker="o", ms=6, mfc=ORANGE, mec="k", mew=.5, zorder=5)
        ax.text(oe["p"][0], 1.9, "ESN: only $w$ = 1", fontsize=FT, ha="center", va="bottom", color="#b35900")
    t = H["trend"]
    ax.text(12, 15.2, f"x{t[0]:.2f} later per step back;\nthe dates depend on the\ndonor, the order does not",
            fontsize=FT, va="top", ha="left", linespacing=1.3)
    ax.set_ylim(.3, 15.8)
    ax.set_yticks([1, 5, 10, 15])
    ax.set_ylabel("memory window $w$ (steps)", fontsize=FL)
    _title(ax, "b", "The answers come to depend on ever older memory",
           r"onset($w$): $L(w) > \frac{1}{2}L_{\mathrm{Bayes}}(w)$ from then on", fs)
    # c: within-episode speed
    ax = axs["c"]
    ok = np.isfinite(Sp["step"])
    top = 31.
    hi_c = np.where(np.isfinite(Sp["hi"]), Sp["hi"], top)
    _band(ax, Sp["x"][ok], Sp["step"][ok], Sp["lo"][ok], hi_c[ok], BLUE)
    nev = Sp["x"][~ok]
    ax.plot([nev.min(), nev.max()], [30, 30], color=BLUE, lw=4, solid_capstyle="butt", zorder=3, alpha=.8)
    ax.text(nev.min(), 28.2, "not within 30 steps", fontsize=FT, va="top", ha="left", color=BLUE)
    ax.axhline(5, color="k", lw=1.3, zorder=2)
    ax.text(11, 5.6, "Bayes: step 5", fontsize=FT, va="bottom")
    ax.set_ylim(0, top)
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_ylabel("steps into the episode", fontsize=FL)
    _title(ax, "c", "Within an episode it catches up with Bayes sooner",
           r"first $t$ with $\mathrm{acc}_t \geq \mathrm{acc}^{\mathrm{Bayes}}_{5} = $" + f"{Sp['level']:.3f}"[1:], fs)
    # d: flip impact growth at every stored lag (Trained RNN by lag, viridis as td_tt_chaos_sweep e; ESN thin orange)
    ax = axs["d"]
    lags = Fl["lags"]
    ax.axvspan(Fl["cc_on"][0], Fl["cc_on"][1], color=CC_ON_C, lw=0, zorder=0)
    ax.axvspan(Fl["cc_on"][1], Fl["cc_off"], color=CC_OFF_C, lw=0, zorder=0)
    ax.text(np.sqrt(Fl["cc_on"][0] * Fl["cc_off"]), 2.84, "count\ncode", fontsize=FT, ha="center", va="top",
            linespacing=1.1, color="#7a5c00")
    lag_c = plt.cm.viridis(np.linspace(.1, .9, len(lags)))
    for i in range(len(lags)):
        ax.plot(Fl["x_e"], Fl["echo"][i][0], color=ORANGE, lw=1., zorder=3)
    for i in range(len(lags)):
        _band(ax, Fl["x"], *Fl["trained"][i], lag_c[i], lw=1.7, alpha=.2, z=4)
    ax.axhline(Fl["bayes"][-1], color="k", lw=1.3, zorder=2)
    ax.axhline(0, color=NAIVE_C, lw=1.3, zorder=2)
    ax.text(1000, Fl["bayes"][-1] - .035, "Bayes, lag 20", fontsize=FT, ha="left", va="top")   # free: 900-5,600
    ax.text(Fl["x_e"][-1], 1.55, "ESN, the same lags\n(lag 1 lowest)",
            fontsize=FT, ha="right", va="bottom", linespacing=1.2,
            color="#b35900")
    ax.legend(handles=[Line2D([], [], color=c_, lw=2.2, label=f"{L}") for c_, L in zip(lag_c, lags)], ncol=len(lags),
              title="Trained RNN, lag", title_fontsize=FT, loc="upper right", bbox_to_anchor=(1., 1.02), frameon=False,
              fontsize=FT, handlelength=1.1, handletextpad=.35, columnspacing=.8, borderaxespad=.2)
    ax.set_ylim(-.25, 2.9)
    ax.set_yticks([0, .5, 1, 1.5, 2, 2.5])
    ax.set_yticklabels(["0", ".5", "1", "1.5", "2", "2.5"])
    ax.set_ylabel("growth of its effect (nats)", fontsize=FL)
    _title(ax, "d", "While it learns, a flip's effect grows; at the end, about 0",
           r"$\chi(s+\mathrm{lag}, s) - \chi(s, s)$,  $\chi = d_H$(intact, flipped beliefs)", fs)
    # e: decay time
    ax = axs["e"]
    for run, col in (("echo", ORANGE), ("trained", BLUE)):
        r = Dc[run]
        _band(ax, r["x"], r["tau"], r["tau_lo"], r["tau_hi"], col)
    r = Dc["trained"]
    ax.set_yscale("log")
    ax.set_ylim(.95, 600)
    runs = _strip_runs(r["x_amp"], r["amp_lo"] > 0)          # within-episode growth resolved above 0
    for a_, b_ in runs:
        ax.plot([a_, b_], [1.2, 1.2], color=BLUE, lw=4.5, solid_capstyle="butt", alpha=.85, zorder=3)
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=4.5, alpha=.85, solid_capstyle="butt",
                              label="bottom: some perturbation\nof the state is amplified\nwithin the episode")],
              loc="upper right", bbox_to_anchor=(1., 1.01), frameon=False, fontsize=FT, handlelength=1.6,
              handletextpad=.6, borderaxespad=.2)
    ax.text(11, 520, "Bayes, naive (log coordinates):\nnever decay ($\\tau = \\infty$)", fontsize=FT, va="top",
            linespacing=1.15)
    yt = [2, 5, 10, 20, 50, 100]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.set_yticklabels([str(v) for v in yt])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("decay time (steps)", fontsize=FL)
    _title(ax, "e", "After passing naive its state forgets ever more slowly",
           r"$\tau_1 = -1/\lambda_1$ beyond 30 steps; $\lambda_1 < 0$: never chaotic", fs)
    # f: one-step expansion log rho(J_t) by entanglement fifth; broken y axis (upper: the Trained RNN around 0; lower:
    # the ESN and the initial network, -.42 to -.34)
    ax = axs["f"]
    fx0, fy0, fw, fh = ax.get_position().bounds
    H_LO, H_GAP = .26, .05                                    # lower segment and gap, shares of the panel's height
    ax.set_position([fx0, fy0 + (H_LO + H_GAP) * fh, fw, (1 - H_LO - H_GAP) * fh])
    axl = fig.add_axes([fx0, fy0, fw, H_LO * fh])
    _guides(axl, c0)
    _xaxis(axl, fs=fs)
    axl.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("")
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.spines["bottom"].set_visible(False)
    YU, YL = (-.055, .022), (-.43, -.32)
    ax.axhspan(0, YU[1], color="#f3dede", lw=0, zorder=0)
    rt, re_ = En["trained"], En["echo"]
    LS_MEAN, LS_LEAST = (0, (4, 1.6)), (0, (1, 1.3))
    for a_ in (ax, axl):                                      # every series in both segments, clipped to each
        _band(a_, rt["x"], rt["v"][:, 0], rt["lo"][:, 0], rt["hi"][:, 0], BLUE, lw=1.6, ls=LS_LEAST, alpha=.16)
        _band(a_, rt["x"], rt["o"], rt["olo"], rt["ohi"], BLUE, lw=1.8, ls=LS_MEAN, alpha=.2)
        _band(a_, rt["x"], rt["v"][:, 4], rt["lo"][:, 4], rt["hi"][:, 4], BLUE, lw=2.1, alpha=.22)
        _band(a_, re_["x"], re_["o"], re_["olo"], re_["ohi"], ORANGE, lw=1.8, ls=LS_MEAN)
    ax.axhline(0, color="k", lw=1.3, zorder=2)
    ax.text(XMAX / 1.05, .0045, "Bayes, naive: 0", fontsize=FT, va="center", ha="right")
    ax.text(XMAX / 1.05, .0158, "the step map expands", fontsize=FT, va="center", ha="right", color="#8a2a2a")
    axl.text(28, -.36, "ESN, all steps (mean)", fontsize=FT, va="center", ha="left", color="#b35900")
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=2.1, label="most entangled fifth"),
                       Line2D([], [], color=BLUE, lw=1.8, ls=LS_MEAN, label="all steps (mean)"),
                       Line2D([], [], color=BLUE, lw=1.6, ls=LS_LEAST, label="least entangled fifth")],
              loc="lower right", bbox_to_anchor=(1., 0.), frameon=False, fontsize=FT, handlelength=2.4,
              handletextpad=.6, labelspacing=.3, borderaxespad=.2)
    ax.set_ylim(*YU)
    ax.set_yticks([-.04, -.02, 0, .02])
    ax.set_yticklabels(["-.04", "-.02", "0", ".02"])
    axl.set_ylim(*YL)
    axl.set_yticks([-.4, -.35])
    axl.set_yticklabels(["-.4", "-.35"])
    brk = dict(marker=[(-1, -.6), (1, .6)], markersize=7, linestyle="none", color="k", mec="k", mew=.9,
               clip_on=False, zorder=6)
    ax.plot([0], [0], transform=ax.transAxes, **brk)                  # the break, on the left spine
    axl.plot([0], [1], transform=axl.transAxes, **brk)
    ax.set_ylabel("one-step expansion (nats)", fontsize=FL)
    fig.canvas.draw()                                         # the label: left of both segments' tick labels, centred
    rnd = fig.canvas.get_renderer()                           # on the whole panel (as matplotlib places single ones)
    tx0 = min(t.get_window_extent(rnd).x0 for a_ in (ax, axl) for t in a_.get_yticklabels() if t.get_text())
    ax.yaxis.set_label_coords(ax.transAxes.inverted().transform((tx0 - 4. / 72 * fig.dpi, 0))[0],
                              (.5 - H_LO - H_GAP) / (1 - H_LO - H_GAP))
    _title(ax, "f", "Passing naive, entangled steps expand; mean contracts",
           r"$\log\rho(J_t)$, $\rho$ = largest $|$eigenvalue$|$ of the step Jacobian $J_t$", fs)
    # legend, suptitle, footnote
    hs = [Line2D([], [], color=BLUE, lw=2.2, label="Trained RNN"), Line2D([], [], color=ORANGE, lw=2.2, label="ESN"),
          Line2D([], [], color="k", lw=1.4, label="Bayes"), Line2D([], [], color=NAIVE_C, lw=1.4, label="naive"),
          Line2D([], [], color="k", lw=1.1, ls=(0, (1, 1.6)), label=f"Trained RNN passes naive ({c0:,.0f})"),
          Patch(color=EARLY_C, label="answers, splitting half-way"),
          Patch(color=LATE_C, label="holding, coupling half-way")]
    fig.legend(handles=hs, loc="center", ncol=7, bbox_to_anchor=(.5, 1 - .93 / Hh), frameon=False,
               fontsize=fs["legend"] - .5, handlelength=2., columnspacing=1.6)
    fig.suptitle("What changes inside the Trained RNN as it learns: its answers come to depend on ever older memory, "
                 "which builds up sooner within each episode (b, c); a flipped\nobservation's growing effect peaks and "
                 "then fades as it learns (d), and its state forgets ever more slowly, without ever turning chaotic (e), "
                 "though entangled steps briefly expand (f)",
                 y=1 - .1 / Hh, fontsize=fs["title"] - 2.5, linespacing=1.4, va="top")
    f2 = lambda v: f"{v:+.2f}".replace("0.", ".", 1)                                   # noqa: E731
    bv, r_tau_end = Dc["bound_end"][1], Dc["trained"]["tau"][-1]
    xt, g1 = rt["x"], En["lag1"]                              # f's footnote, read from the series (numbers() prints them)
    g1pos = g1["lo"] > 0
    k1 = int(np.flatnonzero(g1pos.any(1))[0])
    n1, nq = int(g1pos[k1].sum()), g1pos.shape[1]
    assert g1pos[k1, nq - n1:].all() and n1 < nq, "the first fifths whose lag-1 growth is > 0 are not the most entangled"
    assert all(np.all(rl["hi"] < 0) for rl in En["later"].values()), "a later lag group is not < 0 in every fifth"
    top_n = {1: "the most entangled fifth"}.get(n1, f"the {('two', 'three', 'four')[n1 - 2]} most entangled fifths")
    foot = (f"Seed 20261029; Trained RNN = LSTM, N = 1,000; ESN = the same network with its LSTM weights frozen "
            f"(read-in, read-out and start state trained); {len(A['x'])} checkpoints each (b: ESN {H['n_ck_e']}).\n"
            "a-c: 10,000 probe episodes (50 test environments x 100 latent pairs x 2), 30 steps. b: the state $w$ steps "
            "back replaced by another cell's, $L$ = goal accuracy at $T$ lost, Bayes under the same swap; c: Trained RNN "
            "only.\n"
            "d, f: 1,000 probes (20 per environment), each of 5 observations flipped at steps 1, 3, 5, 10. d: the 5 "
            "stored lags (ESN: lines only); Bayes's rise " + " / ".join(f2(v) for v in Fl["bayes"]) + " at lags " +
            " / ".join(map(str, lags)) + " (drawn: lag 20);\n"
            "count code: the response follows how many observations are 1. e: 500 probes, $\\lambda_1$ along "
            "1,500-step sampled sequences (ESN 300); at the end, with the frame-free bound, $\\tau_1$ = "
            f"{r_tau_end:.0f}-{-1 / bv:.0f} steps;\n"
            "strip: 5-95% of $\\frac{1}{30}\\log\\sigma_1(J_{30}\\cdots J_1)$ above 0 (the largest direction: transient "
            "amplification, not chaos). e, f: the networks' $(u, c)$ state; Bayes and naive in log coordinates, where "
            "their exponents and $\\log\\rho$ are exactly 0.\n"
            "f: entanglement fifths, ranked within each step; the most entangled fifth is $> 0$ while the mean is $< 0$ "
            f"(both 5-95%) at {_ranges(xt, (rt['lo'][:, 4] > 0) & (rt['ohi'] < 0))}; the mean is $> 0$ at "
            f"{_ranges(xt, rt['olo'] > 0)}. Actual perturbations grow only transiently:\n"
            f"a flip's state difference grows on the next step on average at {_ranges(xt, g1['olo'] > 0)} (at "
            f"{xt[k1]:,.0f} only in {top_n}) and shrinks over lags 1-4, 5-20, 21-29.  Bands and bars: 5-95% over 1,000 "
            "resamples of the 50 test environments.")
    fig.text(.5, .04 / Hh, foot, ha="center", va="bottom", fontsize=FN - .5, linespacing=1.5)   # 7 lines (f: 2)
    path = fig_path("td_tt_processes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


def figure_mechanisms():
    """-> figure_bin/td_tt_mechanisms.png"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.legend_handler import HandlerTuple
    from main.utils import fig_path
    fs = _style()
    FL, FT, FN = fs["label"], fs["tick"], fs["label"]
    Le, Sw, Ct, Pa, A = lesion(), swaps(), content(), pairs(), acc()
    c0 = A["c0"][0]
    PW, PH, LM, GX, RM = 4.75, 2.4, .85, 1.85, .35
    TOP, TT, GY, BOT = 1.32, .62, .62, 2.55
    W = LM + 2 * PW + GX + RM
    Hh = TOP + 2 * (TT + PH) + GY + BOT
    fig = plt.figure(figsize=(W, Hh))
    axs = {}
    for r in range(2):
        for c in range(2):
            y0 = Hh - TOP - TT - PH - r * (PH + GY + TT)
            axs["abcd"[2 * r + c]] = fig.add_axes([(LM + c * (PW + GX)) / W, y0 / Hh, PW / W, PH / Hh])
    for ax in axs.values():
        ax.spines[["top", "right"]].set_visible(False)
    XL = (250., 5.2e4)
    # a: lesion
    ax = axs["a"]
    _guides(ax, c0)
    m = Le["x"] >= 320.
    ax.axhline(0, color="k", lw=1., zorder=2)
    for j, p in enumerate(Le["panels"]):
        col = FAM_C[p]
        _band(ax, Le["x"][m], Le["s"][m, j], Le["lo"][m, j], Le["hi"][m, j], col, lw=2.)
        ax.text(XL[1] * 1.06, Le["s"][-1, j], p, fontsize=FT + .5, va="center", ha="left", color=col, clip_on=False)
    _xaxis(ax, lim=XL, fs=fs)
    ax.set_ylim(-.12, .95)
    ax.set_yticks([0, .2, .4, .6, .8])
    ax.set_yticklabels(["0", ".2", ".4", ".6", ".8"])
    ax.set_ylabel("extra damage (fraction of\nthe naive-Bayes gap)", fontsize=FL)
    _title(ax, "a", "Interaction dims: core answers only while learning, corners ever more, holding late",
           r"$f$(each variable's own 5 dims clamped) $-$ $f$(5 interaction dims clamped)", fs)
    # b: swaps
    ax = axs["b"]
    E, wins = Sw["E"], Sw["wins"]
    fams = [f for f, _ in Sw["fams"]]
    offs = np.linspace(-.3, .3, len(fams))
    ax.axhline(0, color="k", lw=1., zorder=1)
    for k, wi in enumerate(wins):
        for f_, item in enumerate(fams):
            c = E[(item, wi)]
            if not c["drawn"]:
                continue
            col = FAM_C[item]
            x0 = k + offs[f_]
            g = c["gate g"]
            ax.plot([x0, x0], [g["lo"], g["hi"]], color=col, lw=1.6, zorder=3)
            ax.plot([x0], [g["v"]], marker="o", ms=7, mfc=col, mec="k", mew=.6, ls="none", zorder=4)
            for lab in SW_SETS[1:]:
                ax.plot([x0 + .06], [c[lab]["v"]], marker="o", ms=4.2, mfc="w", mec=col, mew=1.1, ls="none",
                        zorder=4)
    ax.set_xlim(-.55, len(wins) - .45)
    ax.set_xticks(range(len(wins)))
    ax.set_xticklabels([f"{Sw['x0'][wi]:,.0f}-{Sw['x1'][wi]:,.0f}" for wi in wins], fontsize=FT)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("training window (episodes)", fontsize=FL)
    ax.set_ylim(-.6, 1.12)
    ax.set_yticks([-.4, -.2, 0, .2, .4, .6])
    ax.set_yticklabels(["-.4", "-.2", "0", ".2", ".4", ".6"])
    ax.set_ylabel("share of the window's change\nbeyond random weights", fontsize=FL)
    hs = [Line2D([], [], marker="o", ms=7, mfc="k", mec="k", mew=.6, ls="none", label="all g-gate weights"),
          Line2D([], [], marker="o", ms=4.5, mfc="w", mec="k", mew=1.1, ls="none",
                 label="$W_{hh}$ f + i rows, $W_{hh}$ o rows,\nall $W_{ih}$ (incl. its g rows)")]
    fam_lab = {"answers": "answers (incl. corner)"}
    hs += [Patch(color=FAM_C[f], label=fam_lab.get(f, nm)) for f, nm in Sw["fams"]]
    leg = ax.legend(handles=hs, loc="upper center", ncol=3, frameon=False, fontsize=FT, bbox_to_anchor=(.5, 1.03),
                    columnspacing=1.2, handletextpad=.4, handlelength=1.4)
    fig.canvas.draw()                                       # window separators stop below the legend
    top = leg.get_window_extent().transformed(ax.transAxes.inverted()).y0 - .03
    for k in range(1, len(wins)):
        ax.axvline(k - .5, ymax=top, color=SEP_C, lw=.9, zorder=0)
    _title(ax, "b", "Of these classes, only the g weights beat random weights of their size",
           r"($\Sigma\,\phi_{\mathrm{class}} - \bar{\phi}_{\mathrm{random\ of\ its\ size}}$) / the window's change", fs)
    # c: memory content
    ax = axs["c"]
    _guides(ax, c0)
    rs, bm = Ct["rs"], Ct["bm"]
    mr = rs["x"] >= XL[0]
    _band(ax, rs["x"][mr], rs["net"][mr], rs["net_lo"][mr], rs["net_hi"][mr], BLUE, lw=2.)
    DASH = (0, (4, 1.6))
    _band(ax, bm["x"], bm["v"], bm["lo"], bm["hi"], BLUE, lw=1.6, ls=DASH, marker="o", ms=3.5)
    ax.axhline(rs["bayes"], color="k", lw=1.3, zorder=2)
    ax.axhline(bm["bayes"], color="k", lw=1.3, ls=DASH, zorder=2)
    ax.axhline(0, color=NAIVE_C, lw=1., zorder=1)
    hs = [(Line2D([], [], color=BLUE, lw=2.), Line2D([], [], color="k", lw=1.3)),
          (Line2D([], [], color=BLUE, lw=1.6, ls=DASH, marker="o", ms=3.5), Line2D([], [], color="k", lw=1.3, ls=DASH))]
    ax.legend(hs, ["restart from the start state", "the same belief from\nanother cell's history"], loc="upper right",
              bbox_to_anchor=(1., 1.02), frameon=False, fontsize=FT + .5, handlelength=3.6, labelspacing=.6,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=.4)})
    _xaxis(ax, lim=XL, fs=fs)
    ax.set_ylim(-.012, .19)
    ax.set_yticks([0, .05, .1, .15])
    ax.set_yticklabels(["0", ".05", ".1", ".15"])
    ax.set_ylabel("goal accuracy at $T$ lost", fontsize=FL)
    _title(ax, "c", "Old memory acts mostly through the belief it holds",
           r"$\mathrm{acc}_T(\mathrm{intact}) - \mathrm{acc}_T$(state 5 steps back replaced)", fs)
    # d: pairs
    ax = axs["d"]
    q = np.arange(1, 6)
    ax.plot([.7, 5.1], [0, 0], color=NAIVE_C, lw=1.4, zorder=1)
    for k, col, name in (("bayes", "k", "Bayes"), ("trained", BLUE, "Trained RNN"), ("echo", ORANGE, "ESN")):
        v, lo, hi = Pa[k]
        ax.vlines(q, lo, hi, color=col, lw=1.6, zorder=3)
        ax.plot(q, v, color=col, lw=1.8, marker="o", ms=5.5, mec="k" if col != "k" else "w", mew=.5, zorder=4)
        ax.text(5.22, v[-1], name, fontsize=FT + .5, va="center", ha="left", color=col)
    ax.text(5.22, 0, "naive", fontsize=FT + .5, va="center", ha="left", color="0.4")
    ax.set_xlim(.7, 6.15)
    ax.set_xticks(q)
    ax.set_xticklabels(["1\nleast", "2", "3", "4", "5\nmost"], fontsize=FT)
    ax.tick_params(axis="x", length=3)
    ax.set_xlabel("entanglement of the shared observation (fifth)", fontsize=FL)
    ax.set_ylim(-.06, 1.05)
    ax.set_yticks([0, .2, .4, .6, .8, 1])
    ax.set_yticklabels(["0", ".2", ".4", ".6", ".8", "1"])
    ax.set_ylabel("change of their distance (nats)", fontsize=FL)
    _title(ax, "d", "End: beliefs separate more after entangled observations, as Bayes's do",
           r"$\Delta d_H(p, q)$ after the same observation,  $d_H = \mathrm{max}_r \log\frac{p_r}{q_r}"
           r" - \mathrm{min}_r \log\frac{p_r}{q_r}$", fs)
    hs = [Line2D([], [], color=BLUE, lw=2.2, label="Trained RNN"), Line2D([], [], color=ORANGE, lw=2.2, label="ESN"),
          Line2D([], [], color="k", lw=1.4, label="Bayes"), Line2D([], [], color=NAIVE_C, lw=1.4, label="naive"),
          Line2D([], [], color="k", lw=1.1, ls=(0, (1, 1.6)), label=f"Trained RNN passes naive ({c0:,.0f})"),
          Patch(color=EARLY_C, label="answers, splitting half-way"),
          Patch(color=LATE_C, label="holding, coupling half-way")]
    fig.legend(handles=hs, loc="center", ncol=7, bbox_to_anchor=(.5, 1 - .93 / Hh), frameon=False,
               fontsize=fs["legend"] - 1., handlelength=1.8, columnspacing=1.2)
    fig.suptitle("What each mechanism test shows: beyond each variable's own dims, a few interaction directions matter "
                 "for the core answers only while they are learned (320-3,200),\nfor corner answers increasingly from 320 "
                 "and for holding late (a); of these weight classes, only the candidate-gate weights beat random weights "
                 "of their size (b);\nold memory acts mostly through the belief (c); and beliefs separate more after "
                 "entangled observations (d)",
                 y=1 - .1 / Hh, fontsize=fs["title"] - 3.5, linespacing=1.4, va="top")
    import textwrap
    npairs = Sw["n_pairs"]
    vm, a5 = (Le["other"][k]["late"][0] for k in ("vm", "add5"))
    f2 = lambda v: f"{v:+.2f}".replace("0.", ".", 1)                                   # noqa: E731
    wl = Sw["wins"][-1]
    gc = Sw["E"][("coupling", wl)]["gate g"]                                            # the one unresolved g point
    nd = int((Le["x"] >= 320.).sum())
    rs_x = Ct["rs"]["x"]
    paras = [
        f"Trained RNN, seed 20261029. a: {len(Le['x'])} checkpoints ({nd} drawn, from 320), each lesion scored on 10,000 "
        "probe + 5,028 forward-set episodes; $f$ = (y $-$ naive) / (Bayes $-$ naive), averaged over each behaviour's "
        "rows (core answers 4, corner answers 1, holding 5, coupling 6). Late (from 2,400) the interaction dims cost "
        "every behaviour more than random dims of the same variance (core / corner / holding / coupling: " +
        " / ".join(f2(v) for v in vm) + "), but more than both variables' main-effect dims together (5 dims, more "
        f"variance) mainly for corners and holding ({f2(a5[1])} / {f2(a5[2])}; coupling {f2(a5[3])}, core answers "
        f"{f2(a5[0])}).",
        f"b: {sum(npairs)} consecutive checkpoint pairs ({' / '.join(map(str, npairs))} per window), every weight class "
        "swapped both ways, against 3 random sets of its size; drawn where the window's change is at least .1 of the "
        "gap; answers = goal right, both right, far errors and corner answer = truth. One drawn g point is not "
        f"resolved (coupling, {Sw['x0'][wl]:,.0f}-{Sw['x1'][wl]:,.0f}: {f2(gc['v'])} [{f2(gc['lo'])}, "
        f"{f2(gc['hi'])}]). Not drawn (two of the swaps module's exceptions): against a random 2M of W, $W_{{hh}}$ f + i "
        "beat it on answers in 1,600-4,000 (+.11 [.03, .18]), through $W_{hh}$ i (+.08 [.01, .16] over a random 1M of "
        "$W_{hh}$).",
        f"c: restart {len(rs_x)} checkpoints ({int((rs_x >= XL[0]).sum())} drawn, from {rs_x[rs_x >= XL[0]][0]:,.0f}), "
        f"same belief {len(Ct['bm']['x'])} (3 pools of 2,000 generated episodes per environment).  d: end of training, "
        f"{int(Pa['n_pairsteps']):,} pair-steps x 2 variables per agent, matched on step and distance before; for Bayes "
        "the dependence on entanglement is partly built in (the entanglement is computed from Bayes's beliefs).  Bands "
        "and bars: 5-95% over 1,000 resamples of the 50 test environments (b, training loss: over its 2,000 episodes)."]
    foot = "\n".join("\n".join(textwrap.wrap(p_, 205)) for p_ in paras)
    fig.text(.5, .1 / Hh, foot, ha="center", va="bottom", fontsize=FN - .5, linespacing=1.5)
    path = fig_path("td_tt_mechanisms.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    what = sys.argv[1:] or ["numbers", "figure"]
    if "numbers" in what:
        numbers()
    if "figure" in what:
        figure_processes()
        figure_mechanisms()
