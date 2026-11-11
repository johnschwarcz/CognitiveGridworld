"""Fig. 2 (main), how learning goes beyond naive, Trained RNN against ESN (the same LSTM, recurrence frozen at
initialization, read-in and read-out trained). Figure 1: fig_learn_fr.py.
  a, b  With Bayes's surprise held out, the Trained RNN's activity starts to respond to entangled observations as it
        passes naive (tau +1..+2, r ~.05 at 880). Its activity and its learning signal end locked onto the entangled step
        itself (r .24 / .16, from ~1.5k / ~5k). Before that, both follow surprise: about 75% of the raw lock at 1,040
        is surprise. The ESN never responds to entanglement beyond surprise (|r| <= .03).
  c     Passing naive needs no memory: at 880 memory still hurts in every fifth of both parts. Memory use then turns
        positive, most where the update needs the past (contrast > 0 from 1,009, the most history-dependent fifth from
        1,280; memory-free Bayes is passed at 1,346). It ends at Bayes's profile, but any network with Bayes's step
        updates would; what is measured is when it forms. The ESN's memory hurts in every fifth from 220 on.
  d     Across observation dimensions: the Trained RNN builds a within-step interaction early (91% of Bayes's by 480,
        before passing naive) and one through time later (27% at 880, 72% at the end, mostly not Bayes's pattern). The
        ESN builds a within-step interaction (152% of Bayes's at the end; its gates are nonlinear) but almost none
        through time (<= 7%, Bayes-aligned kappa <= .002).
  A frozen recurrence read in and out linearly does not pass naive. Trained nonlinear processing of the held past does,
  whether trainable recurrence or trained layers around the frozen core (the MLP-rescued ESN, .746, n = 1).
-> main/REPLICATION/figure_bin/fig_learn_mechanism.png (5.5 in wide = the paper's text width, 300 dpi; type 10-11 pt)
   figure();  numbers() / numbers_d() / numbers_e() print every number of RESULTS below
Run from the repository root (CPU, 4 threads; ~10 s for the figure once the caches exist):
    MPLBACKEND=Agg OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/fig_learn_mechanism.py [numbers [abc | d | e]]

Networks and axes. Seed-20261029 runs, one training run per network (LSTM, N = 1,000; one world, identical initial
weights). Probes: analysis/probes.npz, 10,000 test episodes (50 test environments x 100 cells x 2 trajectories) replayed
at 114 checkpoints per network. Training episodes = E - E // 5 (every 5th episode is a test episode); the initial network
is drawn at 10 ("0"). Dotted line: the Trained RNN passes naive = its first checkpoint with goal accuracy at T above
naive's .464: 880 (log-interpolated, for good: 835 [798-876]; fig_learn_fr draws the same 880).

Definitions
  Entanglement (paper B.3; probes 'ent'), per step t and variable: 1/2 J(Delta^naive_t, Delta^Bayes_t), Jeffreys / 2
    (J(p, q) = KL(p||q) + KL(q||p)) between naive's normalised step likelihood and Bayes's step update of the variable's
    marginal. Its two parts (they do not add up exactly):
      within-step  1/2 J(Delta_N, Delta_mf)   memoryless: ~.088 nats at every step, t = 1 included
      history      1/2 J(Delta_mf, Delta_B)   what Bayes's update needs from the past: 0 at t = 1
    Delta_mf = memory-free Bayes's step-t update (the step-wise rule: the other variable marginalised within the step
    under a uniform belief; theory_v11 w = 1).
  Surprise = Bayes's -log p(o_t | o_<t) (joint posterior before t and the environment's tables).
  a, b  Lock statistic (gradflow_analysis _zs / _lagged = plot_training_dynamics 'xc_upd' / 'xc_act', recomputed here to
    1e-16): the measure and the entanglement (mean over the two variables) are z-scored within each episode over steps
    17-30; r(tau) = the mean over the episode's steps of z(measure)_{t+tau} z(entanglement)_t, then over episodes
    (tau > 0: the measure tau steps after the entangled step). Drawn: the entanglement series with surprise regressed out
    within each episode (one slope per episode, re-z-scored), i.e. r with the part of entanglement that surprise does
    not explain. Raw and surprise-only versions are in RESULTS (the raw map is John's att06 map: supplement,
    fig_td_summary j / td_xcorr_lags). No other normalisation.
    a  Credit coherence (td_analysis credit pass, 'cr_coh'): both variables' errors at step s sent back k = 0..15 steps
       through the recurrence; ||Sigma_k delta_k||^2 / Sigma_k ||delta_k||^2 (high: they add up to one consistent
       recurrent-weight change). 5,000 of the probes (100 per environment), error steps 17-30. This is the structure of
       the learning signal, not its amount.
    b  Activity norm ||u_t||_2 of the LSTM hidden state (1,000 units), all 10,000 probes, steps 17-30.
    Maps: lag tau = -8..8 by training episodes (viridis, bright = more; one scale -.05..+.25 for both); triangles mark
    the columns drawn below at 40 / 200 / 1,040 / 39,996 (= end) training episodes (blue shades), with the ESN's end
    column (orange).
  c  Memory use = err_cut - err_intact (nats) per probe, step and variable (td_entanglement_memcut,
     ent_memcut_var_<run>.npz): err = 1/2 J(B_{t-1} + Delta^net_t, B_t), the network's step-t logit increment applied to
     Bayes's log belief at t-1, against Bayes's belief at t. Cut: the network's cell is rerun from its start state over
     input t alone (memory cut to the last step). Bayes under this identical cut is exactly memory-free Bayes, so the
     right-edge diamonds are Bayes's own memory use. Under the random-cell donor cut John prefers, Bayes's loss is flat
     in FR, so the start-state cut is the one with a Bayes reference per fifth; the donor version is supplementary
     (td_tt_memcuts, total entanglement only). Lines: fifths of the history part (left) and of the within-step part
     (right), each variable's value ranked among the 10,000 probes at the same step (2-30) and variable (2,000 per
     fifth, step and variable), pooled over steps and both variables (ratio of sums); light -> dark purple = least ->
     most. One y-range for all four axes. Bands: 5-95% over 1,000 resamples of the 50 test environments. Caches:
     analysis/learn_main_mem{hist,within}_<run>.npz.
  d  Mixed second difference of the step-t read-out increment inc_t = W_ro u_t + b_ro (Bayes: the step-t change of
     its exact log marginal), each variable's 10 values centred, mean |MD| per value (nats, linear axis from 0):
       within a step   flip bits i != j of o_t: MD = inc_t(o_t^ij) - inc_t(o_t^i) - inc_t(o_t^j) + inc_t(o_t), the
                       network stepped from its intact state at t - 1 (td_tt_mixed_diff0; 10 bit pairs, t = 5, 10, ...,
                       30: summary0.npz 'md0_allt')
       through time    flip bit i at t and bit j at s = t - d, d = 1, 2, 3, 5, 8, 12, s >= 1 (td_tt_mixed_diff; 32
                       (t, s) pairs x 25 bit pairs: summary.npz 'md')
     1,000 probes (20 per test environment), both variables, the same at all 34 checkpoints per network; bands 5-95%
     over 1,000 environment resamples. Exact references: naive 0 on both axes (its increment is a sum over bits);
     memory-free Bayes 0 through time and .47 within a step; any linear reservoir read out linearly and summed: 0
     through time (theory_v11 window theorem; checked to 1.8e-14). The ESN is a gated LSTM, not a linear reservoir.
     Thin dash-dot: each network's Bayes-aligned part, computed per (t, s) pair (through time) or per step (within a
     step) and averaged after: mean_p(kappa_p x mean|MD_Bayes|_p), kappa_p = Sigma MD MD_Bayes / Sigma MD_Bayes^2 within
     the pair (cache analysis/learn_main_fig2_aligned.npz). The pooled kappa of the earlier version weighted pairs by
     Bayes's MD energy (27% at t = 5) and is kept as a secondary number.

RESULTS (11-09; every number printed by `numbers` from the caches; [5-95%] over resamples of the 50 test environments)
  Passing naive: first checkpoint above naive's .464 at 880; log-interpolated, for good, 835 [798-876].
  a, b  r(entanglement, surprise) within episodes, steps 17-30: +.445.
     Credit coherence, Trained RNN, tau 0 / +1 / +2 (lags 1-4 summed), surprise held out | raw:
       initial -.017 / .000 / -.007 | -.060 / -.009 / -.008;  200 -.005 / -.003 / +.001 | -.021 / -.014 / -.005
       480  +.008 / +.009 / +.011 (+.050) | -.007 / +.012 / +.037 (+.152)
       880  +.008 / +.021 / +.026 (+.057) | +.004 / +.059 / +.080 (+.224)
       1,040 +.002 / +.017 / +.023 (+.050) | +.006 / +.055 / +.087 (+.227)
       5,000 +.043 / +.027 / +.005 | +.117 / +.125 / +.089;  end +.158 / +.022 / +.009 | +.336 / +.115 / +.022
       Held out, the largest r over lags is <= .039 at every checkpoint up to 2,400; tau 0 stays above .05 from 5,200
       and above .1 from 8,006; map range -.035 .. +.166. Surprise itself: +.100 at tau +2 (880), +.367 at tau 0 (end).
     Activity norm, Trained RNN, same layout:
       initial -.030 / -.010 / -.003 | -.159 / -.085 / -.037;  200 +.017 / +.008 / +.002 | -.124 / -.069 / -.044
       880  +.042 / +.051 / +.047 (+.101) | +.011 / +.025 / +.101 (+.245)
       1,040 +.041 / +.066 / +.059 (+.110) | +.013 / +.052 / +.136 (+.291)
       2,000 +.056 / +.104 / +.060 | +.038 / +.124 / +.167;  end +.237 / +.084 / +.006 | +.364 / +.178 / +.064
       Held out, the largest r over lags is .029 (480), .051 (880), .066 (1,040), .109 (2,400); tau 0 stays above .05
       from 1,524 and above .1 from 4,923; map range -.040 .. +.237.
     ESN, held out: credit between -.026 and +.012 at lags 0..+4 at every checkpoint (end tau 0 -.013; raw -.042,
       surprise -.047); norm between -.034 and +.004 (end tau 0 -.028; raw -.224, surprise -.369; initial -.030, raw
       -.159). The raw negative "shrinks at entangled observations" is a surprise response.
  c  Parts, mean by step t = 1 / 2 / 5 / 10 / 20 / 30 (nats): history .000 / .079 / .121 / .137 / .146 / .147;
     within-step .088 / .090 / .088 / .089 / .087 / .089; total .088 / .197 / .244 / .261 / .269 / .270 (= stored 'ent'
     to 2e-7). Per fifth: history .010 / .030 / .062 / .129 / .448, within-step .007 / .023 / .052 / .102 / .257.
     Bayes's memory use per fifth: history .0019 / .0053 / .0107 / .0212 / .0734; within-step .0161 / .0199 / .0214 /
     .0244 / .0306. r at tau 0 of Bayes's memory use with the within-step / history part .070 / .577 (total .370).
     Trained RNN by HISTORY fifth, least -> most (most [5-95%]; contrast = most - least [5-95%]):
       20     -.0640 -.0643 -.0664 -.0658 -.0652: memory hurts everywhere (the early collapse)
       480    -.0075 -.0089 -.0109 -.0130 -.0148 [-.0181, -.0116]; contrast -.0073 [-.0104, -.0038] (lowest -.0095, 560)
       880    -.0063 -.0084 -.0093 -.0099 -.0106; contrast -.0043 [-.0073, -.0016]: every fifth < 0 as it passes naive
       1,040  contrast +.0009 [-.0020, +.0036];  1,200 +.0037 [+.0009, +.0066]
       2,000  -.0026 -.0027 -.0004 +.0025 +.0142 [+.0111, +.0176]
       end    +.0045 +.0085 +.0146 +.0264 +.0764 [+.0717, +.0812]; contrast +.0719 (Bayes .0019 -> .0734)
       contrast > 0 for good from 1,009 [928-1,162]; the most dependent fifth > 0 from 1,280 [1,178-1,413], the least
       from 2,798 [2,212-3,501].
     Trained RNN by WITHIN-STEP fifth: 480 -.0046 -.0065 -.0084 -.0116 -.0240 (contrast -.0194 [-.0249, -.0135]); 880
       -.0087 .. -.0087 (contrast +.0000); 1,200 contrast +.0096; 2,000 +.0201; end +.0175 +.0228 +.0243 +.0286 +.0373
       (contrast +.0199 [+.0170, +.0230]; Bayes .0161 -> .0306); contrast > 0 for good from 879 [786-979], the most
       within-step-entangled fifth > 0 from 1,158 [1,053-1,285]. During the flip (1,000-2,000) memory use orders by the
       within-step part as much as by the history part. At the end it orders mostly by the history part (contrast .072
       vs .020; r at tau 0 with history / within .471 / .078, against .063 / .063 at 880 and .155 / .092 at 2,000).
     ESN: memory helps every fifth until 200 (+.012 .. +.017 at 20) and hurts every fifth from 220 to the end. End by
       history fifth -.0049 -.0070 -.0093 -.0122 -.0082 (contrast -.0033 [-.0062, -.0006]); by within-step fifth
       -.0042 -.0061 -.0067 -.0088 -.0158 (contrast -.0117 [-.0168, -.0072]: memory hurts most where the within-step
       interaction is largest). Never positive for good in any fifth; r at tau 0 -.010 / +.022.
     For Bayes the profile is close to construction; any network whose step updates equal Bayes's reproduces it. The
     measurement is when and in what order the networks form it.
  d  Bayes: within a step .0754 [.0723, .0784] (by t .158 / .092 / .065 / .053 / .046 / .038 at t = 5 .. 30); through
     time .0748 [.0717, .0779] (flat in lag, .0563-.0565 at t = 15-30; by t (lags 1-3) .176 / .102 / .073 / .059 / .051
     / .043). Memory-free Bayes within a step .471; naive 0.
     Trained RNN, within a step | through time (Bayes-aligned part in brackets, nats):
       initial .0028 | .00005;  20 .0063 | .0015 (kappa 0);  200 .0423 | .0055
       480   .0686 (.0006) = 91% of Bayes's | .0128 [.0124, .0131] (.0005) = 17%
       880   .0730 (.0018) = 97%           | .0199 [.0194, .0203] (.0012) = 27%
       2,400 .0739 (.0047)                  | .0356 (.0037)
       8,000 .0853 (.0105)                  | .0540 (.0089)
       end   .0883 [.0862, .0904] (.0201) = 117% | .0542 [.0529, .0554] (.0133) = 72%
       Within a step it reaches half its end value by 212 training episodes; through time by 1,455, and plateaus from
       8,000 (.0540-.0545).
       kappa, mean over (t, s) pairs: .0057 (480), .0141 (880), .129 (end), still rising (.091 at 8,000); pooled (energy-
       weighted) .0070 / .0174 / .193. By t at the end .400 / .183 / .122 / .087 / .067 / .061 (t = 5 .. 30). Within a
       step, mean per-step kappa .0073 / .0230 / .214 (480 / 880 / end).
     ESN: within a step .0034 (20), .0198 (480), .0259 (880), .0424 (2,400), .0694 (8,000), .1147 [.1100, .1191] (end) =
       152% of Bayes's (Bayes-aligned .0079; kappa ~.10 at every t). Through time .00034 (480), .00045 [.00044, .00045]
       (880), .0049 [.0048, .0050] (end) = 7% of Bayes's, Bayes-aligned <= .00012 (per-pair kappa <= .0014), fading with
       lag (end .0133 / .0071 / .0041 / .0016 / .0004 / .0001 at lags 1 / 2 / 3 / 5 / 8 / 12; Bayes flat .056).
     Trained / ESN: through time 37.9x (480), 44.5x (880), 11.0x (end); within a step 3.5x / 2.8x / .8x.
     Other trained runs at the end, through time (per-pair kappa; pooled): N = 200 .0554 (.095; .124), 100 .0448 (.057;
       .071), 50 .0506 (.037; .045), power 1 .0500 (.117; .173): the size does not order with width, kappa does in both
       versions (N = 1,000: .129; .193).

SUPPLEMENT (moved out of the figure): entanglement-ranked gradient blocks, branches from 480 to 1,280 (td_tt_learnmask_fr
  .png and its docstring; summary.npz equals a recomputation from the 110 per-checkpoint replays exactly). The training
  loss, the ranking (the goal's total entanglement within each step) and the detach act on the goal variable only. The
  read-out ignores which variable is the goal (control at 1,280: goal .617, context .611; main run, end: .760 / .763),
  so both variables were pooled; the halves (2,000 paired environment resamples), at 1,280:
             dd50                   dd20                   50% most - random, top   50% least - random, bottom
    goal     -.044 [-.070, -.017]   -.011 [-.036, +.012]   -.016 [-.036, +.007]     -.027 [-.044, -.011]
    context  -.061 [-.088, -.032]   -.041 [-.064, -.020]   -.037 [-.060, -.015]     -.027 [-.043, -.010]
    both     -.052 [-.073, -.031]   -.026 [-.046, -.009]   -.026 [-.044, -.007]     -.027 [-.039, -.016]
  dd50 mean over the 10 post-branch checkpoints: goal -.053 [-.067, -.040], both -.057 [-.069, -.045]. On the goal
  variable only dd50 and the least-entangled block's bottom-fifth cost hold; the 20% dose step and the most-entangled
  block's top-fifth cost are not resolved. One branch run per condition. The per-step entanglement ranking blocks .705
  of a top-FR-fifth trajectory's steps against .461 of a bottom-fifth one (an FR ranking: .942 / .119; fig_learn_fr), so
  the block is a diluted FR selection and its dissociation is no larger than that dilution predicts. The FR-ranked,
  from-episode-0 version is Fig. 1d.

Corrections to the plan's claims (wf1 verdict.final_plan[1]):
  - "locks onto entangled observations as it passes naive": only the activity starts to (held out, r ~.05 at tau
    +1..+2 at 880). The credit lock up to 2,400 is surprise (held out <= .039), and the on-step locks form late.
  - "half-built at 467 before passing naive" (raw lags 1-4) is a surprise response; dropped. "The tau +2 lock at each
    width's passing-naive time" rested on each run's maximum over training, not its value at passing naive; dropped.
  - "the trained population first shrinks like the frozen one" and "the ESN shrinks at entangled observations": surprise
    responses (held out -.03); dropped.
  - "memory is used where Bayes's update needs the past" holds at the end but follows from accuracy under this cut; the
    flip comes after passing naive, between it and passing memory-free Bayes, and also orders by the within-step part.
  - d: "the ESN never reads an observation in light of earlier ones" -> through time its interaction is 7% of Bayes's,
    short-lag and not Bayes-aligned, but not 0. It does build a within-step interaction (152% of Bayes's). The floor
    label no longer names linear reservoirs next to the ESN.
  - e: the branch panel moved to the supplement (seed gate unmet, goal-only half weaker, FR dilution).
  Not done: a C = 1 through-training run (the clean specificity control for a and b), the final-log-odds MD
  (supplement; only the preview's numbers exist), the MLP-rescued ESN positive control for d (the replay code cannot
  load it), the random-donor cut per step and variable for c. These figures are the controlled Experiment-1 test; the
  RCC / |S| part of the question is answered separately.
Data: analysis/ck_*.npz (cr_coh, cr_rows, norm_u) with probes.npz (a, b; series_<run>.npz read only, as a check),
analysis/ent_memcut_var_<run>.npz with probes.npz (c), through_training/mixed_diff/{summary, summary0, bayes, obs0,
ck_*, ck0_*}.npz (d), through_training/learnmask_fr/ (supplement e)"""
import os
import sys
import glob
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.colors import Normalize, LogNorm, LinearSegmentedColormap         # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm                        # noqa: E402
from matplotlib.cm import ScalarMappable                                          # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from scipy.special import log_softmax, logsumexp                                  # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import td_tt_gap as gap                                                           # noqa: E402

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
TT = os.path.join(TD, "through_training")
MD_DIR = os.path.join(TT, "mixed_diff")
MD_CACHE = os.path.join(MD_DIR, "summary.npz")                                    # d, through time (td_tt_mixed_diff)
MD0_CACHE = os.path.join(MD_DIR, "summary0.npz")                                  # d, within a step (td_tt_mixed_diff0)
LM_CACHE = os.path.join(TT, "learnmask_fr", "summary.npz")                        # supplement (numbers_e)
RUNS = ("trained", "echo")
COL = {"trained": "#1f77b4", "echo": "#ff7f0e", "bayes": "k", "naive": "0.55"}
NAME = {"trained": "Trained RNN", "echo": "ESN"}
TAU = np.arange(-8, 9)                                                            # gradflow_analysis.TAU
WIN = slice(16, 30)                                                               # a, b: steps 17-30 (locked window)
STEPS = slice(1, 30)                                                              # c: steps 2-30
T, N_ENV, N_BIN = 30, 50, 5
X0, XMIN, XMAX = 10, 8, 7e4                                                       # the initial network at X0 ("0")
CXMAX, XSEP = 8.6e4, 4.5e4                                                        # c: right-edge column (Bayes)
XREF = np.geomspace(5.0e4, 7.6e4, N_BIN)                                          # c: Bayes's diamonds, staggered
STAGES = (40, 200, 1040, 39996)                                                   # stage lines (training episodes)
NORM = Normalize(-.05, .25)                                                       # one colour scale for a and b
YL = (-.065, .265)                                                                # stage lines: one y-range for a, b
KCOL = plt.cm.Purples(np.linspace(.3, 1., N_BIN))                                 # fifths: least (light) to most
BLUES = LinearSegmentedColormap.from_list("blues_trunc", plt.cm.Blues(np.linspace(.35, 1., 256)))
SNORM = LogNorm(STAGES[0], STAGES[-1])
PARTS = ("history", "within")
TS, FL, FT = 11, 10, 10                                                           # title, axis label, ticks / keys
FIG_W, DPI = 5.5, 300                                                             # print width (the paper's), inches


# ── data ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def jeff2(la, lb):
    """Jeffreys / 2 between the distributions of two (unnormalised) log vectors over the last axis"""
    la, lb = log_softmax(la, -1), log_softmax(lb, -1)
    return .5 * ((np.exp(la) - np.exp(lb)) * (la - lb)).sum(-1)


def probes():
    z = np.load(os.path.join(AN, "probes.npz"))
    return {k: z[k] for k in ("env", "goal", "right_naive", "inc_N", "inc_SW", "inc_D", "log_jb", "ent", "L", "obs",
                              "logpost")}


def series(run):
    """plot_training_dynamics' per-checkpoint summaries, read straight from analysis/series_<run>.npz (its series()
    rewrites that cache on every call; this figure only reads it)"""
    s = np.load(os.path.join(AN, f"series_{run}.npz"), allow_pickle=True)["s"].item()
    eps = np.array(sorted(s))
    out = {"episode": eps, "x": gap.train_x(eps)}
    for k in ("acc", "xc_upd", "xc_act"):
        out[k] = np.array([s[e][k] for e in eps])
    out["coh"], out["norm"], out["effdim"] = out["xc_upd"][:, 0], out["xc_act"][:, 0], out["xc_act"][:, 1]
    return out


def passes_naive(S, P):
    """the first checkpoint at which the Trained RNN's goal accuracy at T is above naive's (fig_entanglement_driver
    .passes_naive), and naive's accuracy"""
    ar = np.arange(len(P["goal"]))
    naive = P["right_naive"][ar, P["goal"]].mean()
    acc = S["trained"]["acc"][:, -1]
    return float(S["trained"]["x"][int(np.argmax(acc > naive))]), float(naive)


def surprise(P, chunk=2000):
    """Bayes's surprise -log p(o_t | o_<t) per probe and step (n, T): the joint log posterior before step t (uniform at
    t = 1) and the environment's tables p(o_t | r0, r1)"""
    n = len(P["env"])
    out = np.zeros((n, T))
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        L = P["L"][P["env"][sl]].astype(np.float64)                                # (b, 5, R, R)
        lp = log_softmax(P["logpost"][sl].astype(np.float64), -1)                  # (b, T, R * R) after step t
        lprior = np.concatenate([np.full_like(lp[:, :1], -np.log(lp.shape[-1])), lp[:, :-1]], 1)
        o = P["obs"][sl].astype(np.float64)
        ll = np.einsum("bti,bijk->btjk", o, np.log(L)) + np.einsum("bti,bijk->btjk", 1 - o, np.log1p(-L))
        out[sl] = -logsumexp(lprior + ll.reshape(len(o), T, -1), -1)
    return out


def _zs(a):
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-12)


def _lagged(a, b, taus):
    """mean over episodes and steps of a[t + tau] b[t], a and b (n, steps) already z-scored within episode"""
    out = []
    for lag in taus:
        if lag < 0:
            out.append((a[:, :lag] * b[:, -lag:]).mean())
        elif lag > 0:
            out.append((a[:, lag:] * b[:, :-lag]).mean())
        else:
            out.append((a * b).mean())
    return np.array(out)


def _resid(b, c):
    """b with c regressed out within each episode (both z-scored over the window; one slope per episode), re-z-scored"""
    B, C = _zs(b), _zs(c)
    k = (B * C).sum(1, keepdims=True) / ((C * C).sum(1, keepdims=True) + 1e-12)
    return _zs(B - k * C)


def lock(run, P):
    """a, b at every checkpoint: r(tau) of credit coherence ('coh') and activity norm ('norm') with the entanglement
    series, Bayes's surprise regressed out of it within each episode (the drawn maps), with the raw entanglement series
    ('_raw' = series_<run>.npz's xc_upd / xc_act) and with surprise itself ('_surp'); cached in analysis/
    learn_main_fig2_lock_<run>.npz"""
    cache = os.path.join(AN, f"learn_main_fig2_lock_{run}.npz")
    fs = sorted(glob.glob(os.path.join(AN, run, "ck_*.npz")))
    eps = np.array([int(os.path.basename(f)[3:8]) for f in fs])
    if os.path.exists(cache):
        z = dict(np.load(cache))
        if np.array_equal(z["episode"], eps):
            return z
    ent, sp = P["ent"].mean(-1)[:, WIN], surprise(P)[:, WIN]
    E = {"": _resid(ent, sp), "_raw": _zs(ent), "_surp": _zs(sp)}
    out = {f"{m}{k}": [] for m in ("coh", "norm") for k in E}
    for f in fs:
        z = np.load(f)
        rows = z["cr_rows"]
        coh = _zs(z["cr_coh"].astype(np.float64))
        nu = _zs(z["norm_u"][:, WIN].astype(np.float64))
        for k, e in E.items():
            out[f"coh{k}"].append(_lagged(coh, e[rows], TAU))
            out[f"norm{k}"].append(_lagged(nu, e, TAU))
    out = {k: np.array(v) for k, v in out.items()}
    out.update(episode=eps, x=gap.train_x(eps), r_ent_surp=float((_zs(ent) * _zs(sp)).mean()))
    np.savez(cache, **out)
    print("saved", cache)
    return out


def parts(P):
    """per probe, step and variable (n, T, 2): entanglement = 1/2 J(naive's increment, Bayes's) and its two parts,
    within-step 1/2 J(naive's, memory-free Bayes's) and history 1/2 J(memory-free Bayes's, Bayes's)"""
    iN, iS, iD = (P[k].astype(np.float64) for k in ("inc_N", "inc_SW", "inc_D"))
    return {"total": jeff2(iN, iD), "within": jeff2(iN, iS), "history": jeff2(iS, iD)}


def fifths(h):
    """(n, T, 2) -> (n, 29, 2): the fifth of each value by its rank among the n probes at the same step (2-30) and
    variable; every fifth holds exactly n / 5 values of every step and variable"""
    x = h[:, STEPS]
    r = np.argsort(np.argsort(x, 0, kind="stable"), 0, kind="stable")
    f = r * N_BIN // x.shape[0]
    assert all((np.bincount(f[:, t, v], minlength=N_BIN) == x.shape[0] // N_BIN).all()
               for t in range(f.shape[1]) for v in range(2)), "unequal fifths"
    return f


def bayes_memory(P):
    """Bayes's own memory use = the update error of memory-free Bayes (its step-t increment applied to Bayes's belief
    at t-1, 1/2 J against Bayes's belief at t; Bayes's intact error is 0), (n, T, 2)"""
    lj = P["log_jb"].astype(np.float64)
    prior = np.concatenate([np.full_like(lj[:, :1], -np.log(lj.shape[-1])), lj[:, :-1]], 1)
    return jeff2(prior + P["inc_SW"].astype(np.float64), lj)


def _index(P, F):
    """flat (environment, fifth) index of every value in F (n, 29, 2)"""
    return (P["env"][:, None, None] * N_BIN + F).ravel()


def memhist(run, P, F, part="history"):
    """memory use (err_cut1 - err_int) summed per environment and fifth of one part of entanglement, every checkpoint:
    {'episode', 'x', 'sums' (K, 50, 5), 'cnt' (50, 5)}; cached in analysis/learn_main_mem{hist | within}_<run>.npz"""
    src = os.path.join(AN, f"ent_memcut_var_{run}.npz")
    cache = os.path.join(AN, f"learn_main_mem{'hist' if part == 'history' else part}_{run}.npz")
    m = np.load(src)
    eps = m["episode"]
    idx = _index(P, F)
    cnt = np.bincount(idx, minlength=N_ENV * N_BIN).reshape(N_ENV, N_BIN).astype(np.float64)
    check = int((F.ravel().astype(np.int64) * (np.arange(F.size) % 7919)).sum())          # the fifth assignment
    if os.path.exists(cache):
        z = dict(np.load(cache))
        if np.array_equal(z["episode"], eps) and int(z.get("check", -1)) == check:
            return z
    ei, ec = m["err_int"], m["err_cut1"]
    sums = np.stack([np.bincount(idx, (ec[k][:, STEPS].astype(np.float64) - ei[k][:, STEPS].astype(np.float64)).ravel(),
                                 N_ENV * N_BIN).reshape(N_ENV, N_BIN) for k in range(len(eps))])
    out = {"episode": eps, "x": gap.train_x(eps), "sums": sums, "cnt": cnt, "check": check}
    np.savez(cache, **out)
    print("saved", cache)
    return out


def env_curves(sums, cnt, C):
    """ratio-of-sums means (K, 5) and their environment resamples (B, K, 5)"""
    full = sums.sum(1) / cnt.sum(0)
    boot = np.einsum("be,kef->bkf", C, sums) / (C @ cnt)[:, None, :]
    return full, boot


def memory_data(P):
    """c: per part (history, within-step) and run, the fifth curves with 5-95% environment bands, and Bayes's own
    memory use per fifth"""
    pt = parts(P)
    C = gap.boot_counts()
    bm = bayes_memory(P)[:, STEPS].ravel()
    D = {"parts": pt}
    for part in PARTS:
        F = fifths(pt[part])
        D[part] = {"F": F}
        for run in RUNS:
            z = memhist(run, P, F, part)
            full, boot = env_curves(z["sums"], z["cnt"], C)
            D[part][run] = dict(x=z["x"], full=full, boot=boot, lo=np.percentile(boot, 5, 0),
                                hi=np.percentile(boot, 95, 0))
        idx = _index(P, F)
        D[part]["bayes"] = np.bincount(idx, bm, N_ENV * N_BIN).reshape(N_ENV, N_BIN).sum(0) / \
            np.bincount(idx, minlength=N_ENV * N_BIN).reshape(N_ENV, N_BIN).sum(0)
    return D


def _fix_x(x):
    """training episodes for the log axis: raw checkpoint episodes (beyond 45,000) are converted, 0 -> X0"""
    x = np.asarray(x, float)
    if x.max() > 45000:
        x = gap.train_x(x.astype(int))
    return np.where(x > 0, x, X0)


def load_cache(path):
    return dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else None


def interaction():
    """d: per network and checkpoint, mean |MD| per value through time (summary.npz: 32 (t, s) pairs, s < t) and within
    a step (summary0.npz 'md0_allt': the 10 bit pairs of o_t at the same six steps), with 5-95% environment bands, and
    the Bayes-aligned part of each, computed per (t, s) pair (through time) or per step (within a step), then averaged
    over pairs / steps: mean_p(kappa_p x mean|MD_Bayes|_p), kappa_p = Sigma MD MD_Bayes / Sigma MD_Bayes^2 within the
    pair. Cached in analysis/learn_main_fig2_aligned.npz"""
    Z, Z0 = load_cache(MD_CACHE), load_cache(MD0_CACHE)
    if Z is None or Z0 is None:
        return None
    cache = os.path.join(AN, "learn_main_fig2_aligned.npz")
    raws = {r: np.asarray(Z[f"raw_{r}"]) for r in RUNS}
    A = load_cache(cache)
    if A is None or any(not np.array_equal(A.get(f"raw_{r}"), raws[r]) for r in RUNS):
        B, O = np.load(os.path.join(MD_DIR, "bayes.npz")), np.load(os.path.join(MD_DIR, "obs0.npz"))
        n, nt, nt0 = len(B["rows"]), int(B["n_terms_per_probe_pair_var"]), 100      # 25 / 10 bit pairs x 10 values
        assert np.array_equal(B["rows"], O["rows"])
        sbb_p, absb_p = B["s_bb"].sum((0, 2)), B["s_abs"].sum((0, 2)) / (n * 2 * nt)
        sbb_t, absb_t = O["bayes_s_bb"].sum((0, 2)), O["bayes_s_abs"].sum((0, 2)) / (n * 2 * nt0)
        A = {"absb_p": absb_p, "absb_t": absb_t, "pairs": B["pairs"], "ts": O["ts"]}
        for r in RUNS:
            kp, kt, m0 = [], [], []
            for raw in raws[r]:
                c = np.load(os.path.join(MD_DIR, f"ck_{r}_{int(raw):05d}.npz"))
                c0 = np.load(os.path.join(MD_DIR, f"ck0_{r}_{int(raw):05d}.npz"))
                assert np.array_equal(c["rows"], B["rows"]) and np.array_equal(c0["rows"], B["rows"])
                kp.append(c["s_nb"].sum((0, 2)) / sbb_p)
                kt.append(c0["s_nb"].sum((0, 2)) / sbb_t)
                m0.append(c0["s_abs"].sum() / (n * len(O["ts"]) * 2 * nt0))
            A.update({f"raw_{r}": raws[r], f"kappa_p_{r}": np.array(kp), f"kappa_t_{r}": np.array(kt),
                      f"md0_check_{r}": np.array(m0)})
        np.savez(cache, **A)
        print("saved", cache)
    out = {"bayes": float(Z["md_bayes"]), "bayes0": float(Z0["md0_bayes_allt"][0]),
           "memfree0": float(Z0["md0_memfree_allt"][0])}
    for r in RUNS:
        x = _fix_x(Z[f"x_{r}"])
        assert np.allclose(x, _fix_x(Z0[f"x_{r}"]))
        assert np.allclose(A[f"md0_check_{r}"], Z0[f"md0_allt_{r}"], rtol=1e-6)
        out[r] = dict(x=x, hist=np.asarray(Z[f"md_{r}"], float), hist_lo=Z[f"md_{r}_lo"], hist_hi=Z[f"md_{r}_hi"],
                      within=np.asarray(Z0[f"md0_allt_{r}"], float), within_lo=Z0[f"md0_allt_{r}_lo"],
                      within_hi=Z0[f"md0_allt_{r}_hi"],
                      hist_al=(A[f"kappa_p_{r}"] * A["absb_p"]).mean(1), within_al=(A[f"kappa_t_{r}"] * A["absb_t"]).mean(1),
                      kappa_p=A[f"kappa_p_{r}"], kappa_t=A[f"kappa_t_{r}"])
    out["A"] = A
    return out


# ── panels ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def edges(x):
    lx = np.log(x)
    mid = (lx[1:] + lx[:-1]) / 2
    return np.exp(np.concatenate([[lx[0] - (mid[0] - lx[0])], mid, [lx[-1] + (lx[-1] - mid[-1])]]))


def stage_index(x, xv):
    return int(np.argmin(np.abs(np.log(x) - np.log(xv))))


def train_axis(ax, cross, label=True, xlim=(XMIN, XMAX), line="0.4"):
    ax.set_xscale("log")
    ax.set_xticks([X0, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.set_xlim(*xlim)
    ax.minorticks_off()
    if label:
        ax.set_xlabel("training episodes", fontsize=FL, labelpad=1)
    ax.axvline(cross, color=line, lw=.9, ls=":", zorder=5)


def head(ax, letter, name, formula, up=0.):
    """two title lines on fixed baselines above the axes (raised by `up` points over a row of column headers)"""
    for text, dy, fs in ((formula, 6 + up, FL), (f"{letter}   {name}" if letter else name, 21 + up, TS)):
        if text:
            ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, dy), textcoords="offset points",
                        ha="left", va="baseline", fontsize=fs, annotation_clip=False)


def lag_map(ax, S, key, cross, ylabel):
    x = S["x"]
    xe = edges(x)
    te = np.concatenate([TAU - .5, [TAU[-1] + .5]])
    ax.pcolormesh(xe, te, S[key].T, cmap="viridis", norm=NORM, shading="flat", rasterized=True)
    for xv in STAGES:                                                     # where the stage lines below are taken
        k = stage_index(x, xv)
        ax.plot(x[k], 1.0, marker="v", ms=6, color=BLUES(SNORM(x[k])), mec="k", mew=.5, clip_on=False, zorder=7,
                transform=ax.get_xaxis_transform())
    train_axis(ax, cross, xlim=(xe[0], xe[-1]), line="w")
    ax.set_yticks([-8, -4, 0, 4, 8])
    if ylabel:
        ax.set_ylabel(r"lag $\tau$ (steps after)", fontsize=FL, labelpad=1)
    else:
        ax.tick_params(labelleft=False)


def stage_lines(ax, L, key, ylabel):
    st = L["trained"]
    hs = []
    for xv in STAGES:
        k = stage_index(st["x"], xv)
        end = k == len(st["x"]) - 1
        hs += ax.plot(TAU, st[key][k], color=BLUES(SNORM(st["x"][k])), lw=1.6, marker="o", ms=2.4, zorder=3,
                      label="end" if end else f"{st['x'][k]:,.0f}")
    hs += ax.plot(TAU, L["echo"][key][-1], color=COL["echo"], lw=1.5, marker="s", ms=2.4, zorder=2, label="ESN, end")
    ax.axhline(0, color="k", lw=.6)
    ax.axvline(0, color="k", lw=.7, ls=(0, (2, 2)))
    ax.set_xticks([-8, -4, 0, 4, 8])
    ax.set_xlim(-8.5, 8.5)
    ax.set_ylim(*YL)
    ax.set_yticks([0, .1, .2])
    ax.set_xlabel(r"lag $\tau$ (steps after)", fontsize=FL, labelpad=1)
    if ylabel:
        ax.set_ylabel(r"$r$ with entanglement", fontsize=FL, labelpad=1)
    else:
        ax.tick_params(labelleft=False)
    ax.spines[["top", "right"]].set_visible(False)
    return hs


def memory_panel(ax, d, cross, ylim, bayes=None, label=True):
    x = d["x"]
    for k in range(N_BIN):
        ax.fill_between(x, d["lo"][:, k], d["hi"][:, k], color=KCOL[k], alpha=.28, lw=0, zorder=2)
        ax.plot(x, d["full"][:, k], color=KCOL[k], lw=1.4, zorder=3)
    ax.axhline(0, color="0.45", lw=.7, zorder=1)
    if bayes is not None:
        ax.axvline(XSEP, color="0.8", lw=.7)
        for k in range(N_BIN):
            ax.plot(XREF[k], bayes[k], "D", color=KCOL[k], mec="k", mew=.5, ms=3.6, zorder=6, clip_on=False)
        ax.text(XREF[2], bayes.max() + .016, "Bayes", ha="center", va="bottom", fontsize=FT)
    ax.set_ylim(*ylim)
    train_axis(ax, cross, label=label, xlim=(XMIN, CXMAX))
    if not label:
        ax.tick_params(labelbottom=False)
    ax.spines[["top", "right"]].set_visible(False)


def interaction_panel(ax, I, cross, axis, ylim):
    """d: mean |MD| per value through training (linear y from 0) on one axis ('within' or 'hist'); thin dash-dot: each
    network's Bayes-aligned part; dashed black: Bayes"""
    for run in RUNS:
        d = I[run]
        ax.fill_between(d["x"], d[f"{axis}_lo"], d[f"{axis}_hi"], color=COL[run], alpha=.2, lw=0, zorder=2)
        ax.plot(d["x"], d[axis], color=COL[run], lw=1.6, marker="o", ms=1.8, zorder=4)
        ax.plot(d["x"], d[f"{axis}_al"], color=COL[run], lw=1., ls=(0, (4, 1.5, 1, 1.5)), zorder=3)
    b = I["bayes0" if axis == "within" else "bayes"]
    ax.axhline(b, color="k", lw=1.1, ls=(0, (5, 2.5)), zorder=3)
    ax.axhline(0, color="0.3", lw=.8, zorder=1)
    ax.set_ylim(*ylim)
    ax.set_yticks([0, .05, .1])
    train_axis(ax, cross)
    ax.spines[["top", "right"]].set_visible(False)
    return b


def fifths_key(fig, x, y, w, h, Wf, Hf):
    """the purple fifth key: a five-swatch bar from least (light) to most (dark)"""
    cax = fig.add_axes([x / Wf, y / Hf, w / Wf, h / Hf])
    cb = plt.colorbar(ScalarMappable(norm=BoundaryNorm(np.arange(N_BIN + 1), N_BIN), cmap=ListedColormap(KCOL)),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([])
    cb.ax.minorticks_off()
    cb.ax.tick_params(length=0, which="both")
    cb.outline.set_linewidth(.5)
    return cax


# ── figure ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def counts_line(I, K):
    """the figure's one data line: 50 environments x trajectories per environment x checkpoints"""
    kd = len(I["trained"]["x"]) if I is not None else 0
    return (rf"50 environments $\times$ 200 probes $\times$ {K} checkpoints (a: 100; d: 20 $\times$ {kd}); "
            "one run each")


def figure(path=None):
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({"xtick.labelsize": FT, "ytick.labelsize": FT, "axes.labelsize": FL, "legend.fontsize": FT,
                         "axes.linewidth": .7, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "xtick.major.pad": 2, "ytick.major.pad": 2})
    P = probes()
    S = {r: series(r) for r in RUNS}
    cross, _ = passes_naive(S, P)
    Lk = {r: lock(r, P) for r in RUNS}
    Dm = memory_data(P)
    I = interaction()
    xl, wp, gp = .52, 2.0, .18                                     # left edge, panel width, gap (inches)
    xr = xl + wp + gp
    t1, hm = .64, 1.05                                             # a, b maps: top, height
    t2, hs = t1 + hm + .48, .85                                    # stage lines
    t3, hc, gr = t2 + hs + 1.04, .74, .1                           # c: 2 x 2
    t4, hd = t3 + 2 * hc + gr + 1.04, .9                           # d
    yk = t4 + hd + .56                                             # key line baseline
    Wf, Hf = FIG_W, yk + .27
    fig = plt.figure(figsize=(Wf, Hf))

    def add(x, top, w, h):
        return fig.add_axes([x / Wf, (Hf - top - h) / Hf, w / Wf, h / Hf])

    def ftext(x, top, s, **k):
        k.setdefault("va", "baseline")
        return fig.text(x / Wf, (Hf - top) / Hf, s, **k)

    # ── a, b: lag maps (surprise held out) and their stage columns ──────────────────────────────────────────────────
    am, bm = add(xl, t1, wp, hm), add(xr, t1, wp, hm)
    lag_map(am, Lk["trained"], "coh", cross, ylabel=True)
    lag_map(bm, Lk["trained"], "norm", cross, ylabel=False)
    head(am, "a", "Credit coherence", r"$\Vert\Sigma_k\delta_k\Vert^2/\Sigma_k\Vert\delta_k\Vert^2$, Trained RNN", up=3)
    head(bm, "b", "Activity norm", r"$\Vert u_t\Vert_2$, Trained RNN", up=3)
    cax = add(xr + wp + .1, t1, .07, hm)
    cb = plt.colorbar(ScalarMappable(norm=NORM, cmap="viridis"), cax=cax)
    cb.set_ticks([0, .1, .2])
    cb.ax.tick_params(labelsize=FT, length=2, pad=1.5)
    cb.outline.set_linewidth(.5)
    ftext(xr + wp + .55, t1 + hm / 2, "$r$ with entanglement,\nsurprise held out", rotation=90, ha="center",
          va="center", fontsize=FT, linespacing=1.1)
    al, bl = add(xl, t2, wp, hs), add(xr, t2, wp, hs)
    hs_ = stage_lines(al, Lk, "coh", ylabel=True)
    stage_lines(bl, Lk, "norm", ylabel=False)
    kx = xr + wp + .1                                              # the stage key, right of b's stage lines
    ftext(kx, t2 + .02, "Trained", fontsize=FT, ha="left", color="0.15")
    for i, h in enumerate(hs_):
        esn = h.get_label() == "ESN, end"
        yy = t2 + .19 + i * .155 + (.05 if esn else 0)
        fig.add_artist(Line2D([kx / Wf, (kx + .16) / Wf], [(Hf - yy + .035) / Hf] * 2, color=h.get_color(), lw=1.6))
        ftext(kx + .2, yy, "ESN" if esn else h.get_label(), fontsize=FT, ha="left")
        if esn:
            ftext(kx + .2, yy + .15, "end", fontsize=FT, ha="left")

    # ── c: memory use by fifths of the history and the within-step part, Trained RNN / ESN ──────────────────────────
    lo = min(Dm[p][r]["lo"].min() for p in PARTS for r in RUNS)
    hi = max(max(Dm[p][r]["hi"].max() for r in RUNS) for p in PARTS)
    hi = max(hi, max(Dm[p]["bayes"].max() for p in PARTS) + .03)
    ylim_c = (lo - .006, hi + .004)
    cpan = {}
    for i, run in enumerate(RUNS):
        for j, part in enumerate(PARTS):
            ax = add(xl + j * (wp + gp), t3 + i * (hc + gr), wp, hc)
            memory_panel(ax, Dm[part][run], cross, ylim_c, bayes=Dm[part]["bayes"] if i == 0 else None,
                         label=i == 1)
            ax.set_yticks([-.05, 0, .05])
            if j == 0:
                ax.set_ylabel("nats", fontsize=FL, labelpad=1)
                ax.text(.03, .95, NAME[run], transform=ax.transAxes, ha="left", va="top", fontsize=FT, color=COL[run],
                        weight="bold")
            else:
                ax.tick_params(labelleft=False)
            cpan[run, part] = ax
    head(cpan["trained", "history"], "c", "Memory use, start-state cut",
         r"$\mathrm{err_{cut}}-\mathrm{err_{intact}}$, by fifths of each part of entanglement", up=15)
    for j, (part, lab) in enumerate((("history", r"history part $\frac{1}{2}J(\Delta_{\rm mf},\,\Delta_{\rm B})$"),
                                     ("within", r"within-step part $\frac{1}{2}J(\Delta_{\rm N},\,\Delta_{\rm mf})$"))):
        ftext(xl + j * (wp + gp), t3 - .06, lab, fontsize=FT, ha="left", color="0.1")
    kxx, kyb = xr + wp - .5, t3 - 36 / 72                       # the fifth key, on the title's baseline
    fifths_key(fig, kxx, Hf - kyb, .5, .09, Wf, Hf)
    ftext(kxx - .05, kyb, "fifth: least", fontsize=FT, ha="right", color="0.2")
    ftext(kxx + .55, kyb, "most", fontsize=FT, ha="left", color="0.2")

    # ── d: interaction across observation dimensions, within a step and through time ────────────────────────────────
    ylim_d = (0, .13)
    dw, dh = add(xl, t4, wp, hd), add(xr, t4, wp, hd)
    if I is not None:
        b0 = interaction_panel(dw, I, cross, "within", ylim_d)
        b1 = interaction_panel(dh, I, cross, "hist", ylim_d)
        dw.set_ylabel("nats", fontsize=FL, labelpad=1)
        dh.tick_params(labelleft=False)
        dw.text(XMAX, b0 - .004, "Bayes", fontsize=FT, ha="right", va="top")
        dh.text(XMAX, b1 + .004, "Bayes", fontsize=FT, ha="right", va="bottom")
        box = dict(fc="w", ec="none", pad=.6)
        dw.text(XMIN * 1.15, .127, "naive: 0\nmemory-free Bayes: .47", fontsize=FT, ha="left", va="top",
                color="0.35", linespacing=1.15, zorder=6, bbox=box)       # over the dotted line; no data up there
        dh.text(XMIN * 1.15, .127, "naive, memory-free Bayes: 0", fontsize=FT, ha="left", va="top", color="0.35",
                zorder=6, bbox=box)
    for ax in (dw, dh):
        ax.xaxis.label.set_visible(False)
    ftext(xl + wp + gp / 2, t4 + hd + .34, "training episodes", ha="center", fontsize=FL)
    head(dw, "d", r"Interaction across observation dimensions",
         r"mean $|\Delta\Delta\,\mathrm{inc}_t|$: mixed difference of the step-$t$ read-out increment", up=15)
    ftext(xl, t4 - .06, r"within a step: bits $i\neq j$ of $o_t$", fontsize=FT, ha="left", color="0.1")
    ftext(xr, t4 - .06, r"through time: $o_t$ and $o_s$, $s<t$", fontsize=FT, ha="left", color="0.1")

    # ── one key line and one data line ──────────────────────────────────────────────────────────────────────────────
    items = [(COL["trained"], "-", 1.6, "Trained RNN"), (COL["echo"], "-", 1.6, "ESN"),
             ("k", (0, (5, 2.5)), 1.1, "Bayes"), ("0.4", ":", 1.1, "passes naive"),
             ("0.3", (0, (4, 1.5, 1, 1.5)), 1., "Bayes-aligned part")]
    x = .12
    for c, ls, lw, lab in items:
        fig.add_artist(Line2D([x / Wf, (x + .22) / Wf], [(Hf - yk + .035) / Hf] * 2, color=c, ls=ls, lw=lw))
        t_ = ftext(x + .26, yk, lab, fontsize=FT, ha="left")
        fig.canvas.draw()
        x = t_.get_window_extent().x1 / fig.dpi + .14
    print(f"key line ends at {x - .14:.2f} in (figure {Wf} in)")
    assert x - .14 <= Wf - .05, "the key line does not fit"
    note = ftext(Wf / 2, Hf - .07, counts_line(I, len(S["trained"]["x"])), ha="center", fontsize=FT, color="0.25")
    fig.canvas.draw()
    bb = note.get_window_extent()
    assert bb.x0 >= 0 and bb.x1 <= fig.bbox.width, "the data line does not fit"
    path = path or fig_path("fig_learn_mechanism.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("saved", path)
    return path


# ── numbers (every number of the RESULTS block) ─────────────────────────────────────────────────────────────────────

def _r0_var(y, e):
    """r at tau = 0 of each variable's y with its own e (steps 17-30, z-scored within episode), then the mean"""
    return np.mean([(_zs(y[:, WIN, k]) * _zs(e[:, WIN, k])).mean() for k in range(2)])


def _for_good(x, y, sign=1.):
    """first training episode after which y stays past 0 to the end (td_tt_gap.crossing, log-interpolated)"""
    return gap.crossing(x, y, 0., sign)


def numbers():
    P = probes()
    S = {r: series(r) for r in RUNS}
    cross, naive = passes_naive(S, P)
    ar = np.arange(len(P["goal"]))
    C = gap.boot_counts()
    print("== passing naive")
    files = sorted(glob.glob(os.path.join(AN, "trained", "ck_*.npz")))
    acc_env = np.array([np.bincount(P["env"], np.load(f)["right"][ar, -1, P["goal"]], N_ENV) for f in files]) / \
        np.bincount(P["env"], minlength=N_ENV)
    nai_env = np.bincount(P["env"], P["right_naive"][ar, P["goal"]], N_ENV) / np.bincount(P["env"], minlength=N_ENV)
    st = S["trained"]
    p = gap.crossing(st["x"], acc_env.mean(1), nai_env.mean())
    lo, hi, _ = gap.interval(gap.crossing(st["x"], (C @ acc_env.T) / C.sum(1, keepdims=True), (C @ nai_env) / C.sum(1)))
    print(f"  naive's goal accuracy at T {naive:.3f}; first checkpoint above it {cross:,.0f}; log-interpolated, for "
          f"good: {p:,.0f} [{lo:,.0f}-{hi:,.0f}]")
    Lk = {r: lock(r, P) for r in RUNS}
    i0 = int(np.flatnonzero(TAU == 0)[0])
    print(f"== a, b lock statistic; r(entanglement, Bayes's surprise) within episodes, steps 17-30: "
          f"{float(Lk['trained']['r_ent_surp']):+.3f}")
    for r in RUNS:
        L = Lk[r]
        same = {m: np.abs(L[f"{m}_raw"] - S[r]["coh" if m == "coh" else "norm"]).max() for m in ("coh", "norm")}
        print(f"  {NAME[r]}: raw maps vs series_{r}.npz max |diff| {same['coh']:.1e} / {same['norm']:.1e}; drawn "
              f"(surprise held out) map range coh {L['coh'].min():+.3f} .. {L['coh'].max():+.3f}, norm "
              f"{L['norm'].min():+.3f} .. {L['norm'].max():+.3f}")
        for m in ("coh", "norm"):
            print(f"   {m}: tau 0 / +1 / +2 / lags 1-4 summed  |  held out  |  raw  |  with surprise itself")
            for xv in ((10, 40, 200, 480, 640, 880, 1040, 1600, 2000, 5000, 10000, 39996) if r == "trained" else
                       (10, 40, 200, 880, 1040, 39996)):
                k = stage_index(L["x"], xv)
                f = lambda a: (f"{a[k, i0]:+.3f} {a[k, i0 + 1]:+.3f} {a[k, i0 + 2]:+.3f} "     # noqa: E731
                               f"{a[k, i0 + 1:i0 + 5].sum():+.3f}")
                print(f"     {L['x'][k]:6,.0f}: {f(L[m])}  |  {f(L[m + '_raw'])}  |  {f(L[m + '_surp'])}")
        if r == "trained":
            for m in ("coh", "norm"):
                mx = L[m].max(1)
                print(f"   {m} held out, largest r over lags per checkpoint up to 2,400: " + " ".join(
                    f"{x_:,.0f}:{v:.3f}" for x_, v in zip(L["x"], mx) if 400 <= x_ <= 2400))
                for lev in (.05, .1):
                    j = _for_good(L["x"], L[m][:, i0] - lev)
                    print(f"     tau 0 above {lev} for good from {j:,.0f}")
        else:
            for m in ("coh", "norm"):
                pos = (TAU >= 0) & (TAU <= 4)
                print(f"   ESN {m} held out: max over lags 0..+4 and checkpoints {L[m][:, pos].max():+.3f}, min "
                      f"{L[m][:, pos].min():+.3f}; raw tau 0 at the end {L[m + '_raw'][-1, i0]:+.3f}, surprise "
                      f"{L[m + '_surp'][-1, i0]:+.3f}")
    pt = parts(P)
    print("== entanglement and its parts, mean by step t = 1, 2, 5, 10, 20, 30 (nats; per probe and variable):")
    print(f"  stored entanglement vs 1/2 J(naive, Bayes) recomputed: max |diff| "
          f"{np.abs(pt['total'] - P['ent']).max():.1e}")
    for k_, v in pt.items():
        print(f"  {k_:8s} " + " ".join(f"{v[:, t].mean():.3f}" for t in (0, 1, 4, 9, 19, 29)))
    bm = bayes_memory(P)
    print(f"  Bayes's memory use r at tau 0 with total / within-step / history: {_r0_var(bm, pt['total']):.3f} / "
          f"{_r0_var(bm, pt['within']):.3f} / {_r0_var(bm, pt['history']):.3f}")
    m_ = {r: np.load(os.path.join(AN, f"ent_memcut_var_{r}.npz")) for r in RUNS}
    for r in RUNS:
        xm = gap.train_x(m_[r]["episode"])
        out = []
        for xv in (480, 880, 1040, 2000, 39996):
            i = stage_index(xm, xv)
            mu = m_[r]["err_cut1"][i].astype(np.float64) - m_[r]["err_int"][i].astype(np.float64)
            out.append(f"{xm[i]:,.0f} {_r0_var(mu, pt['within']):+.3f} / {_r0_var(mu, pt['history']):+.3f}")
        print(f"  {NAME[r]} memory use r at tau 0 with within-step / history: " + "; ".join(out))
    Dm = memory_data(P)
    for part in PARTS:
        D = Dm[part]
        print(f"== c memory use (nats) by fifths of the {part} part, least .. most [most 5-95%]; contrast = most - least")
        print("  mean part per fifth (nats): " + " / ".join(
            f"{np.mean(pt[part][:, STEPS][D['F'] == f]):.3f}" for f in range(N_BIN)) +
            "; Bayes's own memory use per fifth: " + " / ".join(f"{v:.4f}" for v in D["bayes"]))
        for r in RUNS:
            d = D[r]
            con, conb = d["full"][:, -1] - d["full"][:, 0], d["boot"][..., -1] - d["boot"][..., 0]
            for xv in (20, 200, 480, 560, 880, 1040, 1200, 1360, 2000, 39996):
                k = stage_index(d["x"], xv)
                print(f"  {NAME[r]:11s} {d['x'][k]:6,.0f}: " + " ".join(f"{d['full'][k, f]:+.4f}" for f in range(N_BIN))
                      + f"  [most {d['lo'][k, -1]:+.4f}, {d['hi'][k, -1]:+.4f}]  contrast {con[k]:+.4f} "
                      f"[{np.percentile(conb[:, k], 5):+.4f}, {np.percentile(conb[:, k], 95):+.4f}]")
            for lab, y, yb in (("contrast > 0", con, conb),
                               ("most-dependent fifth > 0", d["full"][:, -1], d["boot"][..., -1]),
                               ("least-dependent fifth > 0", d["full"][:, 0], d["boot"][..., 0])):
                p = _for_good(d["x"], y)
                lo, hi, sh = gap.interval(gap.crossing(d["x"], yb, np.zeros(len(yb))))
                print(f"    {lab} for good from: " + (f"{p:,.0f} [{lo:,.0f}-{hi:,.0f}] ({sh:.0%} of resamples)"
                                                      if np.isfinite(p) else f"never ({sh:.0%} of resamples)"))
            early = d["x"] <= 880
            k = int(np.argmin(np.where(early, con, np.inf)))
            print(f"    most negative contrast up to 880: {con[k]:+.4f} at {d['x'][k]:,.0f}; every fifth < 0 at 880: "
                  f"{bool((d['full'][stage_index(d['x'], 880)] < 0).all())}")
            if (d["full"][-1] < 0).all():
                print(f"    every fifth < 0 from {d['x'][np.flatnonzero((d['full'] > 0).any(1))[-1] + 1]:,.0f} to the end")


def kappa_start(Z, run="trained"):
    """index of the first checkpoint from which the run's pooled kappa interval stays above 0"""
    k = np.asarray(Z[f"kappa_{run}"], float)
    lo = np.asarray(Z[f"kappa_{run}_lo"], float) if f"kappa_{run}_lo" in Z else k
    bad = np.flatnonzero(~(lo > 0))
    return int(bad[-1] + 1) if len(bad) else 0


def numbers_d():
    """every number of RESULTS d: through time from mixed_diff/summary.npz (+ bayes.npz), within a step from
    summary0.npz (+ obs0.npz), the Bayes-aligned parts per (t, s) pair / per step from the per-checkpoint files"""
    Z, Z0, I = load_cache(MD_CACHE), load_cache(MD0_CACHE), interaction()
    if I is None:
        print("== d: no cache yet")
        return
    A = I["A"]
    b, b0, w0 = I["bayes"], I["bayes0"], I["memfree0"]
    print(f"== d interaction; {int(Z['n_probes']):,} probes (20 per environment), {len(Z['pairs'])} (t, s) pairs (t = "
          f"{', '.join(map(str, Z['ts']))}; lags {', '.join(map(str, Z['lags']))}; 25 bit pairs) and 6 steps x 10 bit "
          f"pairs within o_t; both variables; [5-95% over {int(Z['n_boot']):,} environment resamples]")
    print(f"  Bayes: through time {b:.4f} [{float(Z['md_bayes_lo']):.4f}, {float(Z['md_bayes_hi']):.4f}]; within a step "
          f"{b0:.4f} [{Z0['md0_bayes_allt'][1]:.4f}, {Z0['md0_bayes_allt'][2]:.4f}] (t = 15-30: "
          f"{Z0['md0_bayes'][0]:.4f}); memory-free Bayes within a step {w0:.4f} (t = 15-30 {Z0['md0_memfree'][0]:.4f}),"
          f" through time 0; naive 0 on both")
    print("  Bayes through time, by t (lags 1-3) " + " / ".join(f"{v:.3f}" for v in Z["md_t_bayes"]) + "; by lag (t 15-30) "
          + " / ".join(f"{v:.4f}" for v in Z["md_lag_bayes"]) + "; within a step by t " +
          " / ".join(f"{v:.3f}" for v in A["absb_t"]))
    for r in RUNS:
        d = I[r]
        x = d["x"]
        kp_pool = np.asarray(Z[f"kappa_{r}"], float)
        print(f"  {NAME[r]}: within a step [5-95%] (Bayes-aligned) | through time [5-95%] (Bayes-aligned) | kappa per "
              f"(t, s) pair, mean (pooled) | kappa per step within, mean")
        for xv in (10, 20, 40, 80, 120, 200, 480, 880, 1040, 1280, 2400, 4000, 8000, 39996):
            k = stage_index(x, xv)
            print(f"    {x[k]:6,.0f}: {d['within'][k]:.4f} [{d['within_lo'][k]:.4f}, {d['within_hi'][k]:.4f}] "
                  f"({d['within_al'][k]:.4f}) | {d['hist'][k]:.5f} [{d['hist_lo'][k]:.5f}, {d['hist_hi'][k]:.5f}] "
                  f"({d['hist_al'][k]:.5f}) | {d['kappa_p'][k].mean():+.4f} ({kp_pool[k]:+.4f}) | "
                  f"{d['kappa_t'][k].mean():+.4f}")
        for nm, key, ref in (("within a step", "within", b0), ("through time", "hist", b)):
            y = d[key]
            half = gap.crossing(x, y - y[-1] / 2, 0.)
            print(f"    {nm}: end {y[-1] / ref:.0%} of Bayes's; half of its end value from {half:,.0f}; at 880 "
                  f"{y[stage_index(x, 880)] / ref:.0%}, at 480 {y[stage_index(x, 480)] / ref:.0%}")
        kt = d["kappa_p"][-1]
        tt = A["pairs"][:, 0]
        print("    end kappa by t (mean over its pairs): " + " / ".join(
            f"t {t}: {kt[tt == t].mean():.3f}" for t in np.unique(tt)) + "; within a step by t: " +
            " / ".join(f"{v:.3f}" for v in d["kappa_t"][-1]))
    xt = I["trained"]["x"]
    for xv in (480, 880, 39996):
        k = stage_index(xt, xv)
        print(f"  {xt[k]:6,.0f}: through time Trained / ESN {I['trained']['hist'][k] / I['echo']['hist'][k]:.1f}x; within "
              f"a step {I['trained']['within'][k] / I['echo']['within'][k]:.1f}x")
    late = xt >= 8000
    print(f"  Trained RNN through time from 8,000 on: {I['trained']['hist'][late].min():.4f} .. "
          f"{I['trained']['hist'][late].max():.4f}; mean per-pair kappa {I['trained']['kappa_p'][late][0].mean():.3f} -> "
          f"{I['trained']['kappa_p'][-1].mean():.3f}")
    print(f"  ESN through time at most {I['echo']['hist'].max():.4f}; its per-pair kappa at most "
          f"{I['echo']['kappa_p'].mean(1).max():+.4f}; ESN lag profile at the end " +
          " / ".join(f"{v:.4f}" for v in Z["md_lag_echo"][-1]))
    if "extra_runs" in Z:
        sbb_p = np.load(os.path.join(MD_DIR, "bayes.npz"))["s_bb"].sum((0, 2))
        kp = {r: (np.load(os.path.join(MD_DIR, f"ck_{r}_49995.npz"))["s_nb"].sum((0, 2)) / sbb_p).mean()
              for r in Z["extra_runs"]}
        print("  other trained runs at the end, through time: " + "; ".join(
            f"{r} {m:.4f} kappa per pair {kp[r]:.3f} (pooled {k_:.3f})" for r, m, k_ in
            zip(Z["extra_runs"], Z["md_extra_end"], Z["kappa_extra_end"])))


def numbers_e():
    """supplement (moved out of the figure; td_tt_learnmask_fr.png): the entanglement-ranked branches, both variables
    pooled (summary.npz, recomputed from the per-checkpoint replays) and split into the goal and the context variable"""
    Z = load_cache(LM_CACHE)
    if Z is None:
        print("== e: no cache yet")
        return
    import td_tt_learnmask_fr as lf                                               # read-only use of its loaders
    A, mf, q = lf.load(lf.BRANCHES + ("ctrl_s0",))
    worst = 0.
    for sfx, share in lf.SHARES[::-1]:
        for blk in lf.BLOCKS:
            xs, d, db = lf.paired(A, f"det_{blk}{sfx}", "ctrl")
            i = int(np.flatnonzero(xs == lf.X_END)[0])
            lo, hi = lf.band(db[:, i])
            key = f"d{share}_{blk}"
            worst = max(worst, *(np.abs(u - Z[key + s_]).max() for u, s_ in ((d[i], ""), (lo, "_lo"), (hi, "_hi"))))
    print(f"== supplement e: summary.npz vs a recomputation from the replays, max |diff| {worst:.1e}")
    # each variable's role: the training loss, the ranking and the detach act on the goal variable only
    P = np.load(os.path.join(AN, "probes.npz"))
    fr, bins, _, _ = lf.fr_fifths()
    ar, goal, env = np.arange(len(P["goal"])), P["goal"], P["env"]
    isg = np.zeros(bins.shape, bool)
    isg[ar, goal] = True
    Cb = lf.boot_weights()

    def acc(run, raw, mask):
        z = np.load(lf.path(run, raw))
        idx = (np.repeat(env, 2) * N_BIN + bins.ravel())[mask.ravel()]
        S_ = np.bincount(idx, z["right"].ravel()[mask.ravel()].astype(np.float64), N_ENV * N_BIN).reshape(N_ENV, N_BIN)
        N_ = np.bincount(idx, minlength=N_ENV * N_BIN).reshape(N_ENV, N_BIN).astype(np.float64)
        return np.concatenate([(S_.sum(0) / N_.sum(0))[None], (Cb @ S_) / (Cb @ N_)])        # (1 + B, 5)

    def fmt(v):
        return f"{v[0]:+.3f} [{np.percentile(v[1:], 5):+.3f}, {np.percentile(v[1:], 95):+.3f}]"
    end = 1600                                                                        # raw episode of 1,280
    for role, mask in (("goal", isg), ("context", ~isg), ("both", np.ones_like(isg))):
        a = {c: acc(c, end, mask) for c in ("ctrl", "det_hi_f50", "det_lo_f50", "det_rand_f50", "det_hi", "det_lo",
                                             "det_rand")}
        dd50 = (a["det_hi_f50"][:, 4] - a["det_lo_f50"][:, 4]) - (a["det_hi_f50"][:, 0] - a["det_lo_f50"][:, 0])
        dd20 = (a["det_hi"][:, 4] - a["det_lo"][:, 4]) - (a["det_hi"][:, 0] - a["det_lo"][:, 0])
        post = [(lambda h, l_: (h[:, 4] - l_[:, 4]) - (h[:, 0] - l_[:, 0]))(acc("det_hi_f50", rw, mask),
                                                                            acc("det_lo_f50", rw, mask))
                for rw in range(700, 1700, 100)]
        print(f"  {role:7s} at 1,280: dd50 {fmt(dd50)}; dd20 {fmt(dd20)}; 50% most - random top fifth "
              f"{fmt(a['det_hi_f50'][:, 4] - a['det_rand_f50'][:, 4])}; 50% least - random bottom fifth "
              f"{fmt(a['det_lo_f50'][:, 0] - a['det_rand_f50'][:, 0])}; dd50 mean over the 10 post-branch checkpoints "
              f"{fmt(np.mean(post, 0))}")
    xs, _, _ = lf.paired(A, "ctrl_s0", "ctrl")
    zc = {rw: np.load(lf.path("ctrl", rw))["right"] for rw in (1600,)}
    print(f"  read-out goal-agnostic: control at 1,280, accuracy goal {zc[1600][ar, goal].mean():.3f}, context "
          f"{zc[1600][ar, 1 - goal].mean():.3f}")


if __name__ == "__main__":
    arg = sys.argv[1:]
    if arg[:1] == ["numbers"]:
        part = arg[1] if len(arg) > 1 else "all"
        if part in ("all", "abc"):
            numbers()
        if part in ("all", "d"):
            numbers_d()
        if part in ("all", "e"):
            numbers_e()
    else:
        figure()
