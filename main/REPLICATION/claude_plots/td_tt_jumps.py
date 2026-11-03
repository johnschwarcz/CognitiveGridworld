"""Jumps and couplings of the belief through training (John 11-01, away: "go over our existing plots ... further
'through training' analyses ... jumps, couplings"). The end-of-training findings (crossings-findings, fig1c_crossings.py
figure_jumps; fig1c-round5-mechanisms "update correlation"): large jumps of the joint agents are diagonal, those of the
factorized agents move one variable; long belief paths of the joint agents are coupled moves, those of the factorized
agents are one-variable wandering. Here: when does each signature appear in training, relative to passing naive (880
training episodes) and the memory-use flip (800-1,040, fig_td_summary e)?
Revised 11-01 night after an independent verification (scratchpad tt_review/jumps.txt): 22.5 deg is the mean angle of
isotropic jumps, not of two variables that move independently, so c and the steps figure now show the excess over a
matched independence null; every crossing has a 5-95% environment-bootstrap interval; h shows absolute sizes.
-> main/REPLICATION/figure_bin/td_tt_jumps.png        figure()       (2 x 4 panels)
   main/REPLICATION/figure_bin/td_tt_jumps_steps.png  figure_steps() (the excess step by step)
   caches main/DATA/experimental/training_dynamics/through_training/jumps/: jumps_<run>.npz, jumps_references.npz
   (per checkpoint: amplitude-bin counts / angle sums, the amplitude x angle histogram, per-step and per-environment
   squared-length sums, per-episode rho (Pearson, Spearman, steps 5-30), shuffled rho and path length, update
   spectra); jumps_extra_<run>.npz, jumps_extra_references.npz (extra_stats: per environment and step sums of the
   jumps and of the matched null and its two variants, large-jump bands and accuracy per environment, the update
   covariance's top eigenvalue and total per step with their bootstrap); jumps_boot.npz, jumps_boot_variants.npz
   (every timed measure in each environment resample); jumps_checks.npz (the check command's lines); timeline.npz (the
   cross-topic export, timeline(): excess, G, >= 4 cell angle and update share of the Trained RNN and the ESN).

Data: analysis/probes.npz, 10,000 probe episodes (50 environments x 100 latent pairs x 2 trajectories, 30 steps); the
networks' marginal log beliefs logq from analysis/<run>/ck_EEEEE.npz at the 114 standard checkpoints (seed 20261029;
the joint runs' 32 extra early checkpoints are skipped so that every curve has the same sampling); Bayes's and naive's
marginals (log_jb, log_nb) on the same probes. Runs: trained (Trained RNN), echo (ESN), joint_trained (trained LSTM,
joint read-out over the 100 cells, goal-only loss; its marginals), joint_echo (frozen LSTM, joint read-out, goal-only
loss: the trapped joint ESN). x axis: training episodes E - E // 5, log scale, the initial network at 10 (labelled 0).

Definitions (each statistic at its finest unit, then averaged):
  q_t = softmax of the renormalised log marginals (per variable); posterior means mu_t = (E[r_0], E[r_1]),
    E[r] = Sigma_r r q(r); jump Delta mu_t = mu_t - mu_(t-1) for t = 2..30 (29 jumps per episode, 290,000 per
    checkpoint). Amplitude |Delta mu_t| (cells); angle phi_t = arctan(|Delta|_smaller / |Delta|_larger) in degrees,
    0 = one variable moves, 45 = both alike. Isotropic jumps (no preferred direction) average 22.5 deg; two variables
    that move independently average less, the more unequal their amplitudes are.
  phi_bar = Sigma |Delta mu|^2 phi / Sigma |Delta mu|^2 over all 290,000 jumps (squared-length weights, no threshold).
  Independent (the matched null, null_partner): variable 0's jumps of episode i paired with variable 1's jumps of
    another episode of the same environment, stream (goal) and step, the next latent pair (cell + 1 mod 100);
    phi_bar_indep = the same weighted angle over these 290,000 pairs. Excess = phi_bar - phi_bar_indep (degrees,
    additive; 0 = the two variables move as if independent). Variants (checks): the mean over all 99 shifts of the
    latent pair; the same latent pair's other stream.
  a, b  mean phi_t per amplitude bin (2^-7 .. 12.7 cells, 3 bins per octave; grey: < 50 jumps), per checkpoint; right
    strip: naive, Bayes. White line a_half: half of Sigma |Delta mu|^2 comes from jumps larger than it.
  c  the excess.   d  goal accuracy at T = 30 (ck acc[-1]).
  e, f  rho_i = Pearson over the 29 steps of episode i between Delta E[r_0] and Delta E[r_1] (the two variables of
    the same episode and step: paired), |rho_i| averaged within fifths of the episode's path length S_i = Sigma_t
    |Delta mu_it| (ranked at every checkpoint, 2,000 episodes per fifth); dotted: one variable's 29 jumps shuffled
    within the episode (fixed permutation, seed 20261102). Right strip: naive (squares), Bayes (diamonds) per fifth.
  g  G = mean |rho| of the longest fifth - of the shortest (|rho| units); naive's is negative, Bayes's positive. Band:
    its reliable sign change (td_flip_scan.flips: |G| > 2 sigma for >= 5 checkpoints on both sides, sigma from
    environments 0-24 vs 25-49).
  h  per step t, the covariance over the 10,000 episodes of Delta q_t (20 probabilities): top eigenvalue lambda_1 and
    total variance Sigma lambda, geometric means over steps 2-30 (log axis: the gap between a run's thin and thick line
    is the geometric mean over steps of the share lambda_1 / Sigma lambda, "share" below).
  Crossing (crossing()): the training episode after which the curve stays past a level to the end of training,
    log-interpolated between the last checkpoint not past it and the next one. Half-way level: the midpoint of naive's
    and Bayes's values (for the share: half-way from its peak down to Bayes's value). [5-95%]: over 1,000 resamples of
    the 50 environments (seed 20261103), naive and Bayes resampled with the same environments and the level recomputed
    in each (G's fifths cut at equal shares of the resampled weight, _G_boot). Two crossings are ordered only when
    their intervals do not overlap; paired ratio = crossing a / crossing b over the same resamples (median [5-95%]).
  td_tt_jumps_steps: the excess per step t (phi_bar_t - phi_bar_t,indep over the 10,000 episodes); white line: per step,
    the half-way crossing (midpoint of naive's and Bayes's excess at that step) with its 5-95% band. Step groups (2-4,
    5-9, 10-19, 20-30): the mean of the per-step values.

References (probes; Bayes / naive): accuracy .771 / .464. phi_bar 28.88 / 14.95, independent 16.57 / 13.98, excess
+12.30 / +0.97 (all shifts +12.88 / +0.87, same latent pair +12.00 / +0.83). Mean angle of jumps >= 4 cells 38.86 (609
jumps) / 7.67 (423), 2-4 cells 31.29 / 13.35, 1-2 cells 24.83 / 16.28, 1/8-1 cell 20.46 / 20.28; a_half 1.92 / 1.75
cells; A-weighted angle 23.61 / 17.70, top 1% of jumps 34.49 / 11.24. |rho| all episodes .479 / .417 (shuffled .140 /
.143); by path fifth Bayes .401 .405 .439 .509 .641, naive .509 .452 .434 .393 .299; G +.239 / -.209 (Spearman +.171 /
-.156; steps 5-30 +.280 / -.198; deciles +.292 / -.273). Update: share .087 / .116 (arithmetic mean over steps .089 /
.122), lambda_1 .00273 / .00263, total .03143 / .02255. Per step (2-4 / 5-9 / 10-19 / 20-30): phi_bar_t Bayes 30.0 /
28.6 / 26.2 / 21.2, naive 20.2 / 13.4 / 8.8 / 5.8; independent Bayes 18.8 / 13.5 / 11.2 / 9.9, naive 19.0 / 12.4 / 8.3
/ 5.4; excess Bayes +11.2 / +15.0 / +15.1 / +11.3, naive +1.2 / +1.0 / +0.5 / +0.5. Half-way levels: excess +6.638, G
+.015, >= 4 cells 23.267, accuracy .618.

Findings (Trained RNN unless named; training episodes; [5-95%]):
1. The two variables start to move together only long after the network passes naive. The excess rises smoothly in
   log time from a few hundred episodes on: +0.44 (240), +1.20 (400), +1.64 (880: 5.9% [0.6-11.0%] of the naive ->
   Bayes distance), +2.48 (1,200), +3.35 (2,000), +5.04 (4,000), +7.00 (6,800), +10.72 (39,996; Bayes +12.30, -1.585
   [-1.790, -1.372] short of it); half-way at 6,248 [5,366-7,529]. Long-path coupling: G -0.167 at 880 (fifths .464
   .438 .412 .378 .297 = naive's order; 9.5% [1.9-16.5%]); it changes sign reliably between 3,400 and 6,600 (-0.134
   -> +0.173, halves agree 100%), crosses 0 at 5,220 [3,618-5,934], is half-way at 5,656 [4,901-6,481] and ends at
   +.202 (fifths .389 .402 .431 .493 .592; Bayes +.239). Large jumps (>= 4 cells): 11.48 (880; 12.2% [3.4-20.8%]),
   16.77 (1,600), 24.14 (4,000), 29.81 (8,000), 36.93 (39,996; 645 jumps; Bayes 38.86), half-way at 3,855
   [3,363-4,830]; 2-4 cells half-way at 5,233 [3,892-6,493].
   Order: the Trained RNN passes naive at 835 [798-876] (the bootstrap of the log-interpolated crossing; landmark 880)
   and its memory use flips at 800-1,040; accuracy is half-way at 1,394 [1,285-1,489]; then the large-jump angle
   (3,855); then G and the excess (5,656 and 6,248), which cannot be ordered at this resolution (paired ratio .91
   [.76-1.05]). Paired ratios to passing naive: accuracy half-way 1.67 [1.60-1.74], large jumps 4.60 [4.06-5.61], G
   crossing 0 6.20 [4.41-7.10], G half-way 6.75 [5.86-7.92], excess 7.45 [6.57-8.77]. By then accuracy is .733 (4,000)
   to .747 (5,800). The belief-grid unfolding (25-75% at 551-1,262, td_summary_data.grids(); no interval here) ends
   before the first of these intervals starts (3,363).
2. At every step, but the middle steps first. Excess per step group (2-4 / 5-9 / 10-19 / 20-30) at 880: +1.6 / +1.9 /
   +2.2 / +1.2 (naive's +1.2 / +1.0 / +0.5 / +0.5); 4,000: +4.0 / +6.8 / +6.8 / +5.4; 39,996: +10.0 / +13.6 / +12.5 /
   +8.4. Half-way: steps 5-9 at 4,900 [4,162-5,853] and 10-19 at 5,003 [4,015-6,010] (not ordered between them), steps
   2-4 later at 8,266 [6,927-9,187]; steps 20-30 at 10,619 [5,240-27,275] cannot be ordered against the others. Every
   one of the 29 steps crosses (per step 3,442 to 39,232; steps 5-16 within 3,442-6,152).
3. ESN: the coupling never appears. Its excess stays at +0.63 to +1.79 from 240 on (+0.91 at 880, +1.45 at 39,996:
   naive's +0.97 plus +0.475 [+0.263, +0.672]); its raw phi_bar falls from 21.59 (880) to 17.32 (39,996) only because
   its independent level falls from 20.68 to 15.87 (its two variables' amplitudes grow more unequal). G -0.268 to
   -0.147 from 240 on (-0.183 at the end, naive's plus +0.026 [-0.014, +0.058]; fifths .503 .505 .488 .430 .319).
   Jumps >= 4 cells: 3.98 deg at the end from only 91 jumps (Trained RNN 645, Bayes 609, naive 423), consistent with
   the 20k-harvest analysis of other networks on other probes (Optimal 39.5, Trained 37.5, Naive 5.9, ESN 4.2 there).
   Accuracy .431 (naive's minus .033 [.017, .050]).
4. Joint read-outs (dashed, secondary). The initial joint network's raw phi_bar is Bayes-like (28.73) but its
   independent level is 27.25, so its excess is only +1.47 (naive +0.97): its jumps are diagonal because both
   marginals of one joint softmax move by similar small amounts, not because they move together; its G is -0.302
   (fifths .638 .568 .501 .423 .336; naive -0.209) and its step profile is inverted (phi_bar_t 18.6 / 13.9 / 27.1 /
   35.8). The trapped joint ESN (accuracy .427 at the end, at most .428, never above naive) does move its two variables
   together: excess +7.20 to +11.28 from 400 on (+11.02 at the end), G +.062 to +.185 from 400 on (+.132 at the end,
   fifths .525 .519 .540 .550 .657). joint_trained: excess half-way at 486 [350-518] and G crossing 0 at 534 [504-676],
   both before it passes naive at 839 [778-900] (its G half-way, 546 [516-803], cannot be ordered against that); it
   ends at excess +12.07, G +.218, accuracy .686. So in joint-output networks neither signature diagnoses joint
   inference; in the marginal-output Trained RNN they mark the late phase of learning.
5. Early transient (20-160 training episodes): one direction carries a small update. Trained RNN at 20 / 40 / 60 / 80:
   excess +6.33 / +12.62 / +6.16 / +2.18 (raw phi_bar 26.61 / 34.26 / 28.78 / 24.67), |rho| .635 / .829 / .604 / .563;
   share .692 / .809 / .412 / .287 (Bayes .087) with lambda_1 .00270 / .00308 / .00137 / .00104 (Bayes .00273) of a
   total .00391 / .00381 / .00334 / .00363 (Bayes .03143); the share is half-way back down at 58 [57-59]. At 40 the
   direction loads both variables (weaker-variable loading .463 of a possible .5; .224 / .414 / .133 at 20 / 60 / 80),
   while the state collapses (PR(u) 1.26 / 1.39 / 2.45 / 2.99); by 120 the excess is +0.24. ESN: at 40-80 one
   direction (share .783-.844) on one variable (loading .043-.150; excess -0.70 to +0.62); at 120 / 160 it loads both
   (.368 / .483): excess +7.08 / +6.87, |rho| .775 / .593, share .706 / .553, without a state collapse (PR(u)
   9.68-10.34); +3.59 at 200, +1.79 at 240; its share is half-way back down at 178 [173-184]. Where the ESN's shared
   direction arises (read-in or read-out) is not resolved.

Corrections (this revision):
- 22.5 deg is the isotropic level, not the independent one (independent: naive 13.98, Bayes 16.57, Trained RNN
  15.23-16.72 from 640 on). The former "phi_bar stays below 22.5 (the factorized side) ... reliably flips between 4,600
  and 6,800 (-3.593 -> +3.678)" was a level crossing, not a change of coupling (phi_bar crosses 22.5 at 5,681
  [4,662-6,784]); the excess rises smoothly from about 240 on, and the lateness stands.
- The step boundary at 22.5 (step ~3 until 4,000, 16.5 at 39,996) followed the null's step profile (independent from
  19.0 at steps 2-4 to 5.4 at steps 20-30 for naive); replaced by the per-step excess (finding 2).
- "The ESN drifts toward naive" and "joint read-outs start at Bayes's phi_bar" were isotropy effects (findings 3, 4).
- Half-way points are now log-interpolated crossings with intervals (before: the first checkpoint from which the
  curve stays past, accuracy 1,440, phi_bar 4,800, G 5,800; raw phi_bar now 4,700 [3,983-5,539]). The A-weighted and
  top-1% angles are no longer timed (no per-environment primitives).
- h shows absolute sizes; the share is now a geometric mean over steps (the arithmetic mean gives .699 / .810 / .437 /
  .304 at 20-80).
Failed / qualified:
- H "coupled jumps arrive with passing naive or the memory flip": FAILED. At 880 the coupling measures have covered
  5.9% (excess), 9.5% (G) and 12.2% (>= 4 cells) of the naive -> Bayes distance (accuracy 9.3% [0.8-16.9%]; raw
  phi_bar 21.2% [14.2-28.3%]); they are half-way 4.60-7.45 times later (finding 1).
- H "coupled movement marks joint inference": holds only for marginal-output networks after the early transient. The
  trapped joint ESN couples at accuracy .427, and every network couples transiently at 20-160 through one update
  direction.
- Dropped: the update-dimension panel (PR of the log-belief increments, check_floor). Unfloored it ranks Bayes 4.50 >
  Trained 3.77 > naive 2.55 ~ ESN 2.48, but 30 nats below the max floored the ESN exceeds the Trained RNN (6.24 vs
  5.67; Bayes 6.82, naive 4.49), 10 nats below both sit near Bayes (9.03, 9.69; Bayes 10.07, naive 6.96) and in
  probability space Bayes, naive and the networks from 880 on are 13.82-15.14: its late ordering lives in the tails.
  Only its early dip is robust (Trained 40: 1.04 / 1.52 / 3.52 / 1.50; ESN 120: 1.10 / 1.19 / 3.50 / 1.91), which h
  shows as lambda_1 ~ Sigma lambda.

Checks (numbers prints them; the check command recomputes them): stored fp16 logq vs an fp32 CPU forward
(td_entanglement_driver weights-only LSTM, 2,000 probes): phi_bar 16.995 / 17.000 (initial), 34.294 / 34.294 (E=50),
18.167 / 18.167 (E=1,000), 27.508 / 27.508 (E=49,995), ESN 28.707 / 28.706 (E=150), 17.709 / 17.709 (end), posterior
means within 0.0018 cells. Drift (the probe-average jump of a step, the same in every episode): 67% of the squared
movement of the initial network; Trained RNN 15% / 8% at 20 / 40, at most 2.7% from 120 on; ESN 92% / 71% at 20 / 40,
9.2% / 5.2% at 120 / 160, at most 2.1% from 240 on. Removing it changes phi_bar by at most 0.32 deg, the excess by at
most 0.30 and |rho| by at most .008 from 240 on; the transients survive it (Trained 40: excess +12.62 -> +13.08; ESN
120: +7.08 -> +8.70). Null variants, excess half-way: all shifts 5,270 [4,567-6,121], same latent pair 6,631
[5,658-8,210] (matched null 6,248). G variants, half-way: Spearman 2,294 [1,684-4,955], steps 5-30 3,642 [2,891-5,135],
deciles 5,242 [3,940-6,471]: all after passing naive and the memory flip. Shuffled-step |rho| .133-.154 at every
checkpoint of every run; jumps below 1/128 cell (outside the amplitude bins) 1.9% (Trained RNN), 2.8% (ESN).
Caveats: one seed and one probe set (50 test environments). Passing naive is a landmark (880; its bootstrap above),
the memory flip (800-1,040) and the unfolding (551-1,262) are taken from fig_td_summary / td_summary_data and were not
recomputed here. The per-step excess is noisy at late steps (crossings 5,226-39,232 at steps 17-30). The size of the
long-path X-shape depends on the correlation: the Spearman G is -0.052 to -0.003 over 400-1,280 (-0.072 at 240; naive
-0.156) and half-way at 2,294. The joint runs' marginals come from a joint softmax.
Run from the repository root (CPU; the caches ~25 min the first time, then figure + numbers + timeline ~1 min; check
~2 min):
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg \\
        /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_jumps.py \\
        [data check figure numbers timeline synthesis]"""
import os
import sys
import glob
import numpy as np
from scipy.special import logsumexp

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
OUT = os.path.join(TD, "through_training", "jumps")
RUNS = ("trained", "echo", "joint_trained", "joint_echo")
NAME = {"trained": "Trained RNN", "echo": "ESN", "joint_trained": "Trained RNN, joint read-out",
        "joint_echo": "ESN, joint read-out"}
COL = {"trained": "#1f77b4", "echo": "#ff7f0e", "joint_trained": "#1f77b4", "joint_echo": "#ff7f0e"}
LS = {"trained": "-", "echo": "-", "joint_trained": (0, (4, 2)), "joint_echo": (0, (4, 2))}
R, T = 10, 30
N_ENV, PER_ENV = 50, 200                              # probes: environment-major, 200 episodes per environment
RV = np.arange(R, dtype=np.float64)
A_EDGES = 2. ** np.arange(-7, 4, 1 / 3)               # jump amplitude bins (cells): 1/128 to 12.7, 3 per octave
NA = len(A_EDGES) - 1
P_EDGES = np.arange(0, 46, 5.)                         # angle bins (degrees) of the stored 2-D histogram
NP_ = len(P_EDGES) - 1
MIN_N = 50                                             # heat-map bins with fewer jumps are grey
N_FIFTH = 5
NULL_SEED = 20261102
X0 = 10                                                # the initial network is drawn at 10 training episodes
FLAT = 22.5                                            # mean angle of isotropic jumps (no preferred direction)
FLIP_MEMORY = (800, 1040)                              # the Trained RNN's memory-use flip (fig_td_summary e)
BOOT_SEED = 20261103                                   # environment resamples: one matrix for every run (paired)
N_BOOT = 1000
BANDS = ((4., np.inf), (2., 4.))                       # large-jump bands (cells) with per-environment angle sums


def train_x(ep):
    ep = np.asarray(ep)
    return np.maximum(ep - ep // 5, X0).astype(float)


def std_episodes():
    """the 114 standard checkpoints (raw episodes) every run shares (the joint runs' 32 extra early ones are skipped:
    one sampling for every curve)"""
    return sorted(int(os.path.basename(f)[3:8]) for f in glob.glob(os.path.join(AN, "trained", "ck_*.npz")))


def _null_perm(n, m):
    rng = np.random.default_rng(NULL_SEED)
    return np.argsort(rng.random((n, m)), 1)


def _pearson(x, y):
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x * y).sum(1) / np.sqrt((x ** 2).sum(1) * (y ** 2).sum(1))


def _ranks(x):
    return np.argsort(np.argsort(x, 1, kind="stable"), 1, kind="stable").astype(np.float64)


def _pr(x):
    """participation ratio of the covariance of x (n, k) over its rows; also the top PC's variance share and the
    squared loading of the top PC on the first 10 coordinates (variable 0)"""
    x = x - x.mean(0)
    ev, V = np.linalg.eigh(x.T @ x / len(x))
    ev = ev.clip(0)
    return ev.sum() ** 2 / (ev ** 2).sum(), ev[-1] / ev.sum(), (V[:R, -1] ** 2).sum()


def _angle(a0, a1):
    """angle from the nearest axis (degrees) of jumps with absolute components a0, a1"""
    return np.degrees(np.arctan2(np.minimum(a0, a1), np.maximum(a0, a1)))


def null_partner(n, shift=1):
    """the matched independence null pairs row i's variable-0 jumps with the variable-1 jumps of this row: same
    environment, same stream (goal), same step, the latent pair `shift` cells on (cell + shift mod 100)"""
    i = np.arange(n)
    return (i // PER_ENV) * PER_ENV + ((i % PER_ENV) // 2 + shift) % 100 * 2 + i % 2


def boot_counts():
    """(N_BOOT, 50): how often each environment is drawn in each resample (50 draws with replacement)"""
    rng = np.random.default_rng(BOOT_SEED)
    return rng.multinomial(N_ENV, np.full(N_ENV, 1 / N_ENV), size=N_BOOT).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════
# Data: per checkpoint, the jumps of the posterior mean (primitives)
# ═══════════════════════════════════════════════════════════════════

def stats(lq, perm):
    """one observer's beliefs lq (n, T, 2, R) log marginals (any normalisation) -> primitives of its jumps"""
    n = lq.shape[0]
    env = np.arange(n) // PER_ENV
    lq = lq.astype(np.float64)
    lq = lq - logsumexp(lq, -1, keepdims=True)
    q = np.exp(lq)
    mu = q @ RV                                                                   # (n, T, 2) E[r] per variable
    d = np.diff(mu, axis=1)                                                       # (n, 29, 2) jumps into steps 2..30
    a = np.abs(d)
    A = np.hypot(a[..., 0], a[..., 1])                                            # amplitude (cells)
    phi = np.degrees(np.arctan2(a.min(-1), a.max(-1)))                            # angle from the nearest axis
    ia = np.searchsorted(A_EDGES, A, side="right") - 1
    ok = (ia >= 0) & (ia < NA)
    ip = np.minimum((phi / 5).astype(int), NP_ - 1)
    out = {"cnt": np.bincount(ia[ok], minlength=NA), "sphi": np.bincount(ia[ok], phi[ok], minlength=NA),
           "h2": np.bincount(ia[ok] * NP_ + ip[ok], minlength=NA * NP_).reshape(NA, NP_),
           "n_below": int((ia < 0).sum()), "n_above": int((ia >= NA).sum())}
    w = A ** 2
    out["e_step"], out["ephi_step"] = w.sum(0), (w * phi).sum(0)                 # (29,) per step
    out["e_env"] = np.bincount(env, w.sum(1), minlength=N_ENV)                    # (50,) per environment
    out["ephi_env"] = np.bincount(env, (w * phi).sum(1), minlength=N_ENV)
    out["a2w"] = (w * phi).sum() / w.sum()
    out["a1w"] = (A * phi).sum() / A.sum()
    top = A >= np.quantile(A, .99)
    out["phi_top1"] = phi[top].mean()
    Af, wf = A.ravel(), w.ravel()
    o = np.argsort(Af)
    cw = np.cumsum(wf[o])
    out["a_half"] = Af[o][np.searchsorted(cw, cw[-1] / 2)]                       # half of sum A^2 above this amplitude
    out["path"] = A.sum(1).astype(np.float32)                                    # (n,) path length S, steps 2-30
    out["rho"] = _pearson(d[..., 0], d[..., 1]).astype(np.float32)               # (n,) signed, per episode
    out["rho_null"] = _pearson(d[..., 0], np.take_along_axis(d[..., 1], perm, 1)).astype(np.float32)
    out["rho_spear"] = _pearson(_ranks(d[..., 0]), _ranks(d[..., 1])).astype(np.float32)
    out["rho_s5"] = _pearson(d[:, 3:, 0], d[:, 3:, 1]).astype(np.float32)       # jumps into steps 5..30
    out["path_s5"] = A[:, 3:].sum(1).astype(np.float32)
    drift = d.mean(0)
    out["drift_share"] = n * (drift ** 2).sum() / (d ** 2).sum()
    inc = np.diff(lq, axis=1)                                                     # (n, 29, 2, R) log-belief increment
    inc = (inc - inc.mean(-1, keepdims=True)).reshape(n, T - 1, 2 * R)
    dq = np.diff(q, axis=1).reshape(n, T - 1, 2 * R)                              # probability increment
    out["inc_pr"] = np.array([_pr(inc[:, t])[0] for t in range(T - 1)])          # (29,) per step
    v = np.array([_pr(dq[:, t]) for t in range(T - 1)])
    out["dq_pr"], out["dq_top"], out["dq_load0"] = v[:, 0], v[:, 1], v[:, 2]     # (29,) each
    return out


def _stack(rows):
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def build(run):
    """per checkpoint primitives of the run's jumps -> through_training/jumps/jumps_<run>.npz"""
    path = os.path.join(OUT, f"jumps_{run}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    os.makedirs(OUT, exist_ok=True)
    eps = std_episodes()
    perm = None
    rows, acc, pr_u = [], [], []
    for ep in eps:
        z = np.load(os.path.join(AN, run, f"ck_{ep:05d}.npz"))
        lq = z["logq"]
        if perm is None:
            perm = _null_perm(lq.shape[0], T - 1)
        rows.append(stats(lq, perm))
        acc.append(z["acc"][-1])
        pr_u.append(np.nanmean(z["fac"][:, 6, 0, 1]))
    D = _stack(rows)
    D.update(episode=np.array(eps), x=train_x(eps), acc=np.array(acc), pr_u=np.array(pr_u))
    np.savez(path, **D)
    print(f"saved {path}", flush=True)
    return D


def references():
    """Bayes's and naive's jumps on the same probes, computed identically from their marginals"""
    path = os.path.join(OUT, "jumps_references.npz")
    if os.path.exists(path):
        return {k: v for k, v in np.load(path).items()}
    os.makedirs(OUT, exist_ok=True)
    P = np.load(os.path.join(AN, "probes.npz"))
    n = len(P["goal"])
    assert (P["env"] == np.arange(n) // PER_ENV).all()
    perm = _null_perm(n, T - 1)
    out = {}
    for name, key in (("bayes", "log_jb"), ("naive", "log_nb")):
        for k, v in stats(P[key], perm).items():
            out[f"{name}_{k}"] = np.asarray(v)
        out[f"{name}_acc"] = np.asarray(P[f"right_{name}"][np.arange(n), P["goal"]].mean())
    np.savez(path, **out)
    print(f"saved {path}", flush=True)
    return out


def extra_stats(lq, right_T, C):
    """the primitives added by the fix: per (environment, step) sums of the squared jump length and of its product with
    the angle, for the jumps and for the matched independence null (null_partner, shift 1) and two variants (the mean
    over all 99 shifts; the same latent pair's other stream); the >= 4 and 2-4 cell bands and the goal accuracy at T
    per environment; the top eigenvalue and the total variance of the update Delta q_t per step, with their
    environment bootstrap (C) as geometric means over steps"""
    n = lq.shape[0]
    lq = lq.astype(np.float64)
    lq = lq - logsumexp(lq, -1, keepdims=True)
    q = np.exp(lq)
    a = np.abs(np.diff(q @ RV, axis=1))                                           # (n, 29, 2) |jump| per variable
    w = (a ** 2).sum(-1)
    phi = _angle(a[..., 0], a[..., 1])

    def es(v):                                                                     # (n, 29) -> (50, 29) sums
        return v.reshape(N_ENV, PER_ENV, T - 1).sum(1)
    out = {"e_es": es(w), "ephi_es": es(w * phi)}
    for name, rows in (("n", [null_partner(n)]), ("nall", [null_partner(n, s) for s in range(1, 100)]),
                       ("ncell", [np.arange(n) ^ 1])):
        out[name + "_es"], out[name + "phi_es"] = 0., 0.
        for rr in rows:
            a1 = a[rr, :, 1]
            wn = a[..., 0] ** 2 + a1 ** 2
            out[name + "_es"] = out[name + "_es"] + es(wn)
            out[name + "phi_es"] = out[name + "phi_es"] + es(wn * _angle(a[..., 0], a1))
    A = np.sqrt(w)
    m = [(A >= lo) & (A < hi) for lo, hi in BANDS]
    out["band_cnt"] = np.stack([es(mm.astype(np.float64)).sum(1) for mm in m], 1)  # (50, 2)
    out["band_sphi"] = np.stack([es(mm * phi).sum(1) for mm in m], 1)
    out["acc_env"] = right_T.reshape(N_ENV, PER_ENV).mean(1)
    X = np.diff(q, axis=1).reshape(N_ENV, PER_ENV, T - 1, 2 * R)                  # Delta q_t (20 probabilities)
    M1, M2 = X.sum(1), np.einsum("epti,eptj->etij", X, X, optimize=True)         # per environment and step

    def spec(c):                                                                   # c (b, 50) -> lam1, total (b, 29)
        W = PER_ENV * c.sum(1)[:, None, None]
        mu = (c @ M1.reshape(N_ENV, -1)).reshape(len(c), T - 1, 2 * R) / W
        cov = (c @ M2.reshape(N_ENV, -1)).reshape(len(c), T - 1, 2 * R, 2 * R) / W[..., None]
        ev = np.linalg.eigvalsh(cov - mu[..., :, None] * mu[..., None, :]).clip(0)
        return ev[..., -1], ev.sum(-1)
    out["lam1"], out["tot"] = [v[0] for v in spec(np.ones((1, N_ENV)))]          # (29,) all probes
    b = [spec(cc) for cc in np.array_split(C, 4)]
    l1, tot = [np.maximum(np.concatenate([v[i] for v in b]), 1e-300) for i in (0, 1)]
    out["lam1_boot"], out["tot_boot"] = np.exp(np.log(l1).mean(1)), np.exp(np.log(tot).mean(1))
    out["share_boot"] = np.exp(np.log(l1 / tot).mean(1))                          # (N_BOOT,) geometric means
    return out


def build_extra(run):
    """per checkpoint extra primitives of the run (extra_stats) -> through_training/jumps/jumps_extra_<run>.npz"""
    path = os.path.join(OUT, f"jumps_extra_{run}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    P = np.load(os.path.join(AN, "probes.npz"))
    rows, goal, C = np.arange(len(P["goal"])), P["goal"], boot_counts()
    out = []
    for ep in std_episodes():
        z = np.load(os.path.join(AN, run, f"ck_{ep:05d}.npz"))
        out.append(extra_stats(z["logq"], z["right"][rows, -1, goal], C))
    D = _stack(out)
    D["episode"] = np.array(std_episodes())
    np.savez(path, **D)
    print(f"saved {path}", flush=True)
    return D


def references_extra():
    """Bayes's and naive's extra primitives on the same probes -> jumps_extra_references.npz (keys bayes_*, naive_*)"""
    path = os.path.join(OUT, "jumps_extra_references.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    P = np.load(os.path.join(AN, "probes.npz"))
    rows, goal, C = np.arange(len(P["goal"])), P["goal"], boot_counts()
    out = {}
    for name, key in (("bayes", "log_jb"), ("naive", "log_nb")):
        for k, v in extra_stats(P[key], P[f"right_{name}"][rows, goal], C).items():
            out[f"{name}_{k}"] = np.asarray(v)
    np.savez(path, **out)
    print(f"saved {path}", flush=True)
    return out


def load():
    D = {run: build(run) for run in RUNS}
    D["ref"] = references()
    D["extra"] = {run: build_extra(run) for run in RUNS}
    D["extra_ref"] = references_extra()
    D["boot"] = boot(D)
    D["boot"].update(boot_variants(D))
    return D


# ═══════════════════════════════════════════════════════════════════
# Environment bootstrap and crossings
# ═══════════════════════════════════════════════════════════════════

def _G_boot(path, rho, C, nb=N_FIFTH):
    """G in every environment resample: mean |rho| of the longest minus the shortest path fifth, each episode weighted
    by its environment's count and the fifths cut at equal shares of the total weight (an episode on a cut is split
    between the two fifths); with every count 1 this is rho_fifths exactly"""
    env = np.arange(len(path)) // PER_ENV
    o = np.argsort(path, kind="stable")
    r = np.abs(np.asarray(rho, np.float64)[o])
    ok = np.isfinite(r).astype(np.float64)
    r = np.where(ok > 0, r, 0.)
    out = []
    for c in np.array_split(C, 4):
        w = c[:, env[o]]
        hi = np.cumsum(w, 1)
        lo = hi - w
        cut = hi[:, -1:] / nb
        bot = np.clip(np.minimum(hi, cut) - lo, 0, None)
        top = np.clip(hi - np.maximum(lo, hi[:, -1:] - cut), 0, None)
        out.append((top @ r) / (top @ ok) - (bot @ r) / (bot @ ok))
    return np.concatenate(out)


def _parts(D, key):
    """(main primitives, extra primitives) of a run or of a reference (references get a leading checkpoint axis of 1)"""
    if key in RUNS:
        return D[key], D["extra"][key]
    def pick(Z):
        return {k[len(key) + 1:]: np.asarray(v)[None] for k, v in Z.items() if k.startswith(key + "_")}
    return pick(D["ref"]), pick(D["extra_ref"])


def boot(D):
    """the timing measures in every environment resample (boot_counts: the same resamples for every run and for Bayes
    and naive, so differences are paired) -> through_training/jumps/jumps_boot.npz, keys '<run or reference>|<measure>'
    of shape (N_BOOT, K checkpoints) (K = 1 for Bayes and naive); 'excess_step' (N_BOOT, K, 29) float32"""
    path = os.path.join(OUT, "jumps_boot.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    C = boot_counts()

    def rat(num, den):                                                             # (K, 50[, 29]) sums -> (B, K[, 29])
        return np.einsum("be,ke...->bk...", C, num) / np.einsum("be,ke...->bk...", C, den)
    out = {}
    for key in RUNS + ("bayes", "naive"):
        z, X = _parts(D, key)
        K = len(z["a2w"])
        phib = rat(X["ephi_es"].sum(-1), X["e_es"].sum(-1))
        out[f"{key}|phibar"] = phib
        for name, pre in (("excess", "n"), ("excess_all", "nall"), ("excess_cell", "ncell")):
            out[f"{key}|{name}"] = phib - rat(X[pre + "phi_es"].sum(-1), X[pre + "_es"].sum(-1))
        out[f"{key}|excess_step"] = (rat(X["ephi_es"], X["e_es"]) - rat(X["nphi_es"], X["n_es"])).astype(np.float32)
        for j in range(len(BANDS)):
            cnt = np.einsum("be,ke->bk", C, X["band_cnt"][..., j])
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"{key}|band{j}"] = np.where(cnt >= MIN_N, np.einsum("be,ke->bk", C, X["band_sphi"][..., j]) / cnt,
                                                 np.nan)
        out[f"{key}|acc"] = np.einsum("be,ke->bk", C, X["acc_env"]) / C.sum(1)[:, None]
        out[f"{key}|grad"] = np.stack([_G_boot(z["path"][k], z["rho"][k], C) for k in range(K)], 1)
        for m in ("share", "lam1", "tot"):
            out[f"{key}|{m}"] = X[m + "_boot"].T
        print(f"  boot {key}", flush=True)
    np.savez(path, **out)
    print(f"saved {path}", flush=True)
    return out


G_VARIANTS = (("grad_spear", "rho_spear", "path", N_FIFTH), ("grad_s5", "rho_s5", "path_s5", N_FIFTH),
              ("grad_dec", "rho", "path", 10))


def boot_variants(D):
    """G with Spearman rho, over steps 5-30 only, and by deciles, in every environment resample (Trained RNN, Bayes,
    naive) -> through_training/jumps/jumps_boot_variants.npz"""
    path = os.path.join(OUT, "jumps_boot_variants.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    C = boot_counts()
    out = {}
    for key in ("trained", "bayes", "naive"):
        z, _ = _parts(D, key)
        for name, rk, pk, nb in G_VARIANTS:
            out[f"{key}|{name}"] = np.stack([_G_boot(z[pk][k], z[rk][k], C, nb) for k in range(len(z["a2w"]))], 1)
    np.savez(path, **out)
    print(f"saved {path}", flush=True)
    return out


def crossing(x, y, level, sign=1.):
    """the training episode after which y stays past `level` (sign +1: above it, -1: below it) to the end of training,
    log-interpolated between the last checkpoint not past it and the next one; NaN if y ends not past it, x[0] if it
    is past it at every checkpoint. y (..., K) with level a scalar or of shape (...); a checkpoint without a value
    counts as not past"""
    y = np.asarray(y, np.float64)
    shp = y.shape[:-1]
    y2 = y.reshape(-1, y.shape[-1])
    lev = np.broadcast_to(np.asarray(level, np.float64), shp).reshape(-1)
    with np.errstate(invalid="ignore"):
        past = sign * (y2 - lev[:, None]) > 0
    notp = ~past
    last = y2.shape[1] - 1 - np.argmax(notp[:, ::-1], 1)                       # the last checkpoint not past
    out = np.full(len(y2), np.nan)
    out[past.all(1)] = x[0]
    sel = np.where(past[:, -1] & notp.any(1))[0]
    k = last[sel]
    y0, y1 = y2[sel, k], y2[sel, k + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(np.isfinite(y0), (lev[sel] - y0) / (y1 - y0), 1.)
    f = np.clip(np.nan_to_num(f, nan=1.), 0, 1)
    lx = np.log(np.asarray(x, np.float64))
    out[sel] = np.exp(lx[k] + f * (lx[k + 1] - lx[k]))
    return out.reshape(shp) if shp else float(out[0])


def interval(cb):
    """5-95% interval of resampled crossings (a resample that never crosses counts as after the end of training: an
    open bound is NaN) and the share of resamples that cross"""
    v = np.where(np.isnan(cb), np.inf, cb)
    lo, hi = np.percentile(v, 5, method="lower"), np.percentile(v, 95, method="higher")
    return (float(lo) if np.isfinite(lo) else np.nan), (float(hi) if np.isfinite(hi) else np.nan), \
        float(np.isfinite(v).mean())


# ═══════════════════════════════════════════════════════════════════
# Summaries
# ═══════════════════════════════════════════════════════════════════

def rho_fifths(path, rho, nb=N_FIFTH):
    """mean |rho| per path-length bin of equal counts (shortest -> longest; ranked, exactly n / nb per bin)"""
    r = np.argsort(np.argsort(path, kind="stable"), kind="stable") * nb // len(path)
    return np.array([np.nanmean(np.abs(rho[r == k])) for k in range(nb)])


def summary(D):
    """per run and reference: the curves of the figures, with environment halves A / B (environments 0-24 / 25-49)"""
    e = np.arange(N_ENV * PER_ENV) // PER_ENV
    hA, hB = e < N_ENV // 2, e >= N_ENV // 2
    S = {}
    for key in RUNS + ("bayes", "naive"):
        z, X = _parts(D, key)
        K = len(z["a2w"])
        s = {"a2w": z["a2w"].astype(float), "a1w": z["a1w"].astype(float), "phi_top1": z["phi_top1"].astype(float),
             "a2w_A": z["ephi_env"][:, :N_ENV // 2].sum(1) / z["e_env"][:, :N_ENV // 2].sum(1),
             "a2w_B": z["ephi_env"][:, N_ENV // 2:].sum(1) / z["e_env"][:, N_ENV // 2:].sum(1),
             "fifths": np.stack([rho_fifths(z["path"][k], z["rho"][k]) for k in range(K)]),
             "fifths_null": np.stack([rho_fifths(z["path"][k], z["rho_null"][k]) for k in range(K)]),
             "rho_all": np.nanmean(np.abs(z["rho"]), 1), "null_all": np.nanmean(np.abs(z["rho_null"]), 1),
             "inc_pr": z["inc_pr"].mean(1), "dq_pr": z["dq_pr"].mean(1), "acc": z["acc"].astype(float),
             "a_half": z["a_half"].astype(float), "cnt": z["cnt"],
             "angle": np.where(z["cnt"] >= MIN_N, z["sphi"] / np.maximum(z["cnt"], 1), np.nan),
             "step_angle": z["ephi_step"] / z["e_step"], "drift": z["drift_share"].astype(float),
             "dq_top": z["dq_top"].mean(1), "dq_load0": z["dq_load0"], "n_below": z["n_below"]}
        s["grad"] = s["fifths"][:, -1] - s["fifths"][:, 0]
        for name, rk, pk, nb in (("grad_spear", "rho_spear", "path", N_FIFTH), ("grad_s5", "rho_s5", "path_s5", N_FIFTH),
                                 ("grad_dec", "rho", "path", 10)):
            f = np.stack([rho_fifths(z[pk][k], z[rk][k], nb) for k in range(K)])
            s[name] = f[:, -1] - f[:, 0]
        for h, m in (("A", hA), ("B", hB)):
            f = np.stack([rho_fifths(z["path"][k][m], z["rho"][k][m]) for k in range(K)])
            s["grad_" + h] = f[:, -1] - f[:, 0]
        # the fix: excess over the matched independence null, large-jump bands, absolute update sizes
        s["phibar"] = X["ephi_es"].sum((-1, -2)) / X["e_es"].sum((-1, -2))           # = a2w
        s["indep"] = X["nphi_es"].sum((-1, -2)) / X["n_es"].sum((-1, -2))           # matched independence null
        s["excess"] = s["phibar"] - s["indep"]
        for name, pre in (("all", "nall"), ("cell", "ncell")):
            s["indep_" + name] = X[pre + "phi_es"].sum((-1, -2)) / X[pre + "_es"].sum((-1, -2))
            s["excess_" + name] = s["phibar"] - s["indep_" + name]
        s["indep_step"] = X["nphi_es"].sum(-2) / X["n_es"].sum(-2)                   # (K, 29)
        s["excess_step"] = s["step_angle"] - s["indep_step"]
        s["band_cnt"] = X["band_cnt"].sum(-2)                                        # (K, 2): >= 4, 2-4 cells
        s["band"] = np.where(s["band_cnt"] >= MIN_N, X["band_sphi"].sum(-2) / np.maximum(s["band_cnt"], 1), np.nan)
        s["band0"], s["band1"] = s["band"][:, 0], s["band"][:, 1]
        gm = lambda v: np.exp(np.log(np.maximum(v, 1e-300)).mean(-1))                # noqa: E731
        s["lam1"], s["tot"], s["share"] = gm(X["lam1"]), gm(X["tot"]), gm(X["lam1"] / np.maximum(X["tot"], 1e-300))
        s["weak_load"] = np.minimum(z["dq_load0"], 1 - z["dq_load0"]).mean(-1)        # top PC's weaker variable
        s["acc_env_mean"] = X["acc_env"].mean(-1)
        if key in RUNS:
            s["x"], s["pr_u"], s["episode"] = z["x"], z["pr_u"], z["episode"]
        S[key] = s
    return S


# timed measures: name -> (curve key in summary and bootstrap, sign, level kind); level kinds: 'half' = the midpoint of
# naive's and Bayes's values, 'zero' = 0, 'naive' = naive's value, 'iso' = 22.5 deg (isotropic), 'peak' = half-way from
# the curve's maximum down to Bayes's value
TIMED = {"excess": ("excess", 1, "half"), "grad": ("grad", 1, "half"), "grad0": ("grad", 1, "zero"),
         "band0": ("band0", 1, "half"), "band1": ("band1", 1, "half"), "acc": ("acc", 1, "half"),
         "share": ("share", -1, "peak"), "acc_naive": ("acc", 1, "naive"), "excess_all": ("excess_all", 1, "half"),
         "excess_cell": ("excess_cell", 1, "half"), "phibar": ("phibar", 1, "half"), "phibar_iso": ("phibar", 1, "iso"),
         "share_half": ("share", -1, "half"), "grad_spear": ("grad_spear", 1, "half"),
         "grad_s5": ("grad_s5", 1, "half"), "grad_dec": ("grad_dec", 1, "half")}


def _level(kind, y, nv, bv):
    """the crossing level of one curve (or of each resample: y (..., K), nv / bv (...))"""
    if kind == "half":
        return (nv + bv) / 2
    if kind == "zero":
        return np.zeros_like(nv)
    if kind == "naive":
        return nv
    if kind == "iso":
        return np.full_like(nv, FLAT)
    return (np.nanmax(y, -1) + bv) / 2                                                   # 'peak'


def timing(S, B):
    """{(run, measure): {level, point, lo, hi, frac}}: the crossing (crossing(): stays past the level to the end) of the
    full-probe curve and its 5-95% interval over the 1,000 environment resamples (levels recomputed per resample from
    the same resample's naive and Bayes)"""
    out = {}
    for run in RUNS:
        x = S[run]["x"]
        for name, (key, sign, kind) in TIMED.items():
            if f"{run}|{key}" not in B:                                                 # G variants: Trained RNN only
                continue
            y, nv, bv = S[run][key], S["naive"][key][0], S["bayes"][key][0]
            lev = float(_level(kind, y, np.float64(nv), np.float64(bv)))
            yb, nb, bb = B[f"{run}|{key}"], B[f"naive|{key}"][:, 0], B[f"bayes|{key}"][:, 0]
            cb = crossing(x, yb, _level(kind, yb, nb, bb), sign)
            lo, hi, frac = interval(cb)
            out[(run, name)] = dict(level=lev, point=crossing(x, y, lev, sign), lo=lo, hi=hi, frac=frac, boot=cb)
    return out


def step_timing(S, B, run):
    """per step t: the crossing of the excess at that step through the midpoint of naive's and Bayes's excess at that
    step, with its 5-95% interval -> point (29,), lo (29,), hi (29,), frac (29,)"""
    x = S[run]["x"]
    lev = (S["naive"]["excess_step"][0] + S["bayes"]["excess_step"][0]) / 2
    point = crossing(x, S[run]["excess_step"].T, lev)
    yb = B[f"{run}|excess_step"].astype(np.float64)                                    # (N_BOOT, K, 29)
    levb = (B["naive|excess_step"][:, 0] + B["bayes|excess_step"][:, 0]) / 2            # (N_BOOT, 29)
    cb = crossing(x, np.moveaxis(yb, 1, 2), levb)                                        # (N_BOOT, 29)
    iv = np.array([interval(cb[:, t]) for t in range(T - 1)])
    return dict(level=lev, point=point, lo=iv[:, 0], hi=iv[:, 1], frac=iv[:, 2])


STEP_GROUPS = ((0, 3), (3, 8), (8, 18), (18, 29))     # steps 2-4, 5-9, 10-19, 20-30 (indices into steps 2..30)


def group_timing(S, B, run):
    """per step group: the crossing of the group's mean per-step excess through the mean of its steps' half-way levels,
    with its 5-95% interval"""
    x = S[run]["x"]
    yb = B[f"{run}|excess_step"].astype(np.float64)                                    # (N_BOOT, K, 29)
    levb = (B["naive|excess_step"][:, 0] + B["bayes|excess_step"][:, 0]) / 2            # (N_BOOT, 29)
    lev = (S["naive"]["excess_step"][0] + S["bayes"]["excess_step"][0]) / 2
    out = []
    for a, b in STEP_GROUPS:
        cb = crossing(x, yb[:, :, a:b].mean(2), levb[:, a:b].mean(1))
        lo, hi, frac = interval(cb)
        out.append(dict(level=lev[a:b].mean(), point=crossing(x, S[run]["excess_step"][:, a:b].mean(1),
                                                                 lev[a:b].mean()), lo=lo, hi=hi, frac=frac, boot=cb))
    return out


def order(ca, cb):
    """paired comparison of two crossings over the same environment resamples: the ratio crossing a / crossing b
    (median and 5-95% interval of the per-resample log ratio; resamples where either never crosses are dropped) and
    the share of resamples used"""
    ok = np.isfinite(ca) & np.isfinite(cb)
    r = np.log10(ca[ok] / cb[ok])
    return 10 ** np.median(r), 10 ** np.percentile(r, 5), 10 ** np.percentile(r, 95), ok.mean()


def at(S, run, xv):
    """index of the checkpoint nearest to xv training episodes"""
    return int(np.argmin(np.abs(S[run]["x"] - xv)))


def gap_fraction(S, B, run, key, xv):
    """(value - naive) / (Bayes - naive) at the checkpoint nearest xv, with its 5-95% interval over the resamples"""
    import warnings
    k = at(S, run, xv)
    f = (S[run][key][k] - S["naive"][key][0]) / (S["bayes"][key][0] - S["naive"][key][0])
    fb = (B[f"{run}|{key}"][:, k] - B[f"naive|{key}"][:, 0]) / (B[f"bayes|{key}"][:, 0] - B[f"naive|{key}"][:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                                 # no jumps >= 4 cells yet
        return f, np.nanpercentile(fb, 5), np.nanpercentile(fb, 95)


def paired_diff(S, B, run, key, ref, k=-1):
    """run's value at checkpoint k minus the reference's, with its 5-95% interval over the paired resamples"""
    import warnings
    d = S[run][key][k] - S[ref][key][0]
    db = B[f"{run}|{key}"][:, k] - B[f"{ref}|{key}"][:, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return d, np.nanpercentile(db, 5), np.nanpercentile(db, 95)


def flip_list(x, G, GA, GB):
    """td_flip_scan's reliable sign changes of G (|G| > 2 sigma for >= 5 checkpoints on both sides; sigma from the
    environment halves), with the training episodes where the first run ends and the second starts"""
    import td_flip_scan as tfs
    return [dict(f, x_end=float(x[f["end"]]), x_start=float(x[f["start"]])) for f in tfs.flips(G, GA, GB)]


def passes_naive():
    import fig_entanglement_driver as fd
    return fd.passes_naive(np.load(os.path.join(AN, "probes.npz")))


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

TS, FL, FK, FT = 12.5, 11, 10, 10                    # title, axis label, tick, legend / footnote sizes
XMIN = 8
SECONDARY = dict(lw=1.1, alpha=.55)                  # the joint read-out runs are drawn thinner and lighter
STAGES = (880, 2000, 4000, 8000, 39996)              # checkpoints of the per-step profiles (td_tt_jumps_steps c)


def _edges(x):
    lx = np.log(x)
    mid = (lx[1:] + lx[:-1]) / 2
    return np.exp(np.concatenate([[lx[0] - (mid[0] - lx[0])], mid, [lx[-1] + (lx[-1] - mid[-1])]]))


def _xmax():
    """every panel ends at the right edge of the heat maps' last column (about 40,400 training episodes)"""
    return float(_edges(train_x(std_episodes()))[-1])


def _xaxis(ax, label=True):
    ax.set_xscale("log")
    ax.set_xlim(XMIN, _xmax())
    ax.set_xticks([X0, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.minorticks_off()
    ax.tick_params(labelsize=FK)
    if label:
        ax.set_xlabel("training episodes", fontsize=FL)


def _marks(ax, cross, on_heat=False):
    """the Trained RNN's passing-naive line and its memory-use flip"""
    if on_heat:
        ax.axvspan(*FLIP_MEMORY, color="w", alpha=.3, lw=0, zorder=3)
        ax.axvline(cross, color="w", lw=1.1, ls=":", zorder=4)
        return
    ax.axvspan(*FLIP_MEMORY, color="0.55", alpha=.22, lw=0, zorder=0)
    ax.axvline(cross, color="0.25", lw=1.1, ls=":", zorder=1)


def _half_mark(ax, tm, color=COL["trained"]):
    """the half-way crossing (dot at its level) with its 5-95% interval over the environment resamples (bar)"""
    p, lo, hi = tm["point"], tm["lo"], tm["hi"]
    if not np.isfinite(p):
        return
    hi = hi if np.isfinite(hi) else _xmax()
    ax.errorbar([p], [tm["level"]], xerr=[[p - lo], [hi - p]], fmt="o", ms=7.5, mfc=color, mec="k", mew=1.,
                ecolor="k", elinewidth=1.8, capsize=4.5, capthick=1.8, zorder=6)


def _title(ax, letter, name, formula):
    ax.set_title(f"{letter}   {name}\n{formula}", loc="left", fontsize=TS, linespacing=1.5)


def _ref_lines(ax, S, key, dy=(0., 0.), lw=1.4, names=("Bayes", "naive"), factor=(1., 1.)):
    """Bayes (black) and naive (grey) levels, labelled just outside the right edge (label offsets: dy added, data
    units; factor multiplied, for log axes)"""
    import matplotlib.transforms as mtr
    tr = mtr.blended_transform_factory(ax.transAxes, ax.transData)
    for (k, col), name, o, f in zip((("bayes", "k"), ("naive", "0.5")), names, dy, factor):
        v = S[k][key][0]
        ax.axhline(v, color=col, lw=lw, zorder=1)
        if name:
            ax.text(1.015, (v + o) * f, name, transform=tr, fontsize=FT, color=col, va="center", ha="left")


def _strip_heat(ax, S, key, edges_y, cmap, vmin, vmax, white=None):
    """the narrow reference axis next to a heat map: naive's and Bayes's columns"""
    import matplotlib.patheffects as pe
    ax.set_facecolor("0.82")
    ax.pcolormesh([0, 1, 2], edges_y, np.stack([S["naive"][key][0], S["bayes"][key][0]], 1), cmap=cmap, vmin=vmin,
                  vmax=vmax, shading="flat", rasterized=True)
    if white is not None:
        for j, k in enumerate(("naive", "bayes")):
            ax.plot([j + .1, j + .9], [S[k][white][0]] * 2, color="w", lw=1.8,
                    path_effects=[pe.withStroke(linewidth=3.2, foreground="k")])
    ax.set_yticks([])
    ax.set_xticks([.5, 1.5])
    ax.set_xticklabels(["naive", "Bayes"], rotation=90, fontsize=FK)


def figure(D=None, S=None):
    """-> figure_bin/td_tt_jumps.png"""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    D = D or load()
    S = S or summary(D)
    tm = timing(S, D["boot"])
    cross = passes_naive()
    st = S["trained"]
    fl_g = flip_list(st["x"], st["grad"], st["grad_A"], st["grad_B"])[-1]
    fifth_col = plt.cm.Greens(np.linspace(.3, 1., N_FIFTH))
    PW, PH, SW, GX, GXC, GY = 3.75, 2.75, .34, 1.3, 2.05, 1.6           # GXC: after the colour bar column
    L, TOP, BOT = .95, 1.0, 2.55
    W = L + 4 * PW + 2 * GX + GXC + .7
    H = TOP + 2 * PH + GY + BOT
    fig = plt.figure(figsize=(W, H))
    xcol = [L, L + PW + GX, L + 2 * PW + GX + GXC, L + 3 * PW + 2 * GX + GXC]

    def place(r, c, w=PW, dx=0.):
        y = H - TOP - PH - r * (PH + GY)
        return fig.add_axes([(xcol[c] + dx) / W, y / H, w / W, PH / H])

    def line(ax, run, key):
        kw = SECONDARY if run.startswith("joint") else dict(lw=2.)
        ax.plot(S[run]["x"], S[run][key], color=COL[run], ls=LS[run], zorder=3, **kw)

    fig.text(.5, 1 - .32 / H, r"Jumps of the belief's posterior mean through training:  $\Delta\mu_t = (\Delta E[r_0], "
             r"\Delta E[r_1])$ at every step;  amplitude $|\Delta\mu_t|$ (cells) and angle $\varphi_t$ from the nearest "
             r"axis (0$^\circ$: one variable moves, 45$^\circ$: both alike)", ha="center", va="center", fontsize=TS + 1)
    # a, b: mean angle by amplitude x training
    xe = _edges(st["x"])
    for c, run in enumerate(("trained", "echo")):
        s = S[run]
        ax = place(0, c, w=PW - SW - .1)
        ax.set_facecolor("0.82")
        im = ax.pcolormesh(xe, A_EDGES, s["angle"].T, cmap="viridis", vmin=0, vmax=45, shading="flat",
                           rasterized=True)
        ax.plot(s["x"], s["a_half"], color="w", lw=1.8, path_effects=[pe.withStroke(linewidth=3.2, foreground="k")])
        ax.set_yscale("log")
        ax.set_ylim(2. ** -5, A_EDGES[-1])
        ax.set_yticks([.1, 1, 10])
        ax.set_yticklabels(["0.1", "1", "10"])
        _xaxis(ax)
        _marks(ax, cross, on_heat=True)
        ax.set_ylabel(r"jump amplitude $|\Delta\mu_t|$ (cells)", fontsize=FL)
        _title(ax, "ab"[c], f"{NAME[run]}: coupling angle of its jumps",
               r"mean $\varphi_t = \arctan(|\Delta|_{\mathrm{smaller}} / |\Delta|_{\mathrm{larger}})$ per amplitude")
        axr = place(0, c, w=SW, dx=PW - SW)
        _strip_heat(axr, S, "angle", A_EDGES, "viridis", 0, 45, white="a_half")
        axr.set_yscale("log")
        axr.set_ylim(2. ** -5, A_EDGES[-1])
        axr.minorticks_off()
        axr.set_yticks([])
    cax = fig.add_axes([(xcol[1] + PW + .12) / W, (H - TOP - PH) / H, .12 / W, PH / H])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, 15, 30, 45])
    cb.ax.tick_params(labelsize=FK)
    cb.set_label(r"mean angle $\varphi_t$ (degrees)", fontsize=FL - .5)
    # c: excess angle over the matched independence null
    ax = place(0, 2)
    ax.axhline(0, color="0.6", lw=1., ls=(0, (1, 2)), zorder=1)
    for run in RUNS:
        line(ax, run, "excess")
    _ref_lines(ax, S, "excess")
    _half_mark(ax, tm[("trained", "excess")])
    ax.set_ylim(-2, 16)
    _xaxis(ax)
    _marks(ax, cross)
    ax.set_ylabel(r"excess over independent (degrees)", fontsize=FL)
    ax.spines[["top", "right"]].set_visible(False)
    _title(ax, "c", "Coupling of the belief's movement",
           r"$\bar\varphi - \bar\varphi_{\mathrm{indep}}$,   $\bar\varphi = \Sigma_{i,t}\, |\Delta\mu_{it}|^2\, "
           r"\varphi_{it} \,/\, \Sigma_{i,t}\, |\Delta\mu_{it}|^2$")
    # d: accuracy
    ax = place(0, 3)
    for run in RUNS:
        line(ax, run, "acc")
    _ref_lines(ax, S, "acc")
    _half_mark(ax, tm[("trained", "acc")])
    ax.set_ylim(.05, .83)
    _xaxis(ax)
    _marks(ax, cross)
    ax.set_ylabel("goal accuracy", fontsize=FL)
    ax.spines[["top", "right"]].set_visible(False)
    _title(ax, "d", "Accuracy at T", r"goal variable at step 30: $\langle\, \mathrm{argmax}\, q_{30} = r^{\ast} \rangle$")
    # e, f: |rho| by path-length fifth
    for c, run in enumerate(("trained", "echo")):
        s = S[run]
        ax = place(1, c, w=PW - SW - .1)
        for k in range(N_FIFTH):
            ax.plot(s["x"], s["fifths"][:, k], color=fifth_col[k], lw=1.9)
        ax.plot(s["x"], s["null_all"], color="0.45", lw=1.1, ls=(0, (1, 2)))
        ax.text(XMIN * 1.15, s["null_all"][-1] + .015, "one variable's steps shuffled", fontsize=FT - .5,
                color="0.35", va="bottom")
        ax.set_ylim(.1, 1.0)
        _xaxis(ax)
        _marks(ax, cross)
        ax.set_ylabel(r"mean $|\rho_i|$", fontsize=FL)
        ax.spines[["top", "right"]].set_visible(False)
        _title(ax, "ef"[c], f"{NAME[run]}: coupling of the two updates",
               r"$|\rho_i| = |\mathrm{corr}_t(\Delta E[r_0], \Delta E[r_1])|$ per episode $i$")
        axr = place(1, c, w=SW, dx=PW - SW)
        for j, (k, mk) in enumerate((("naive", "s"), ("bayes", "D"))):
            for f in range(N_FIFTH):
                axr.plot(j + .5, S[k]["fifths"][0, f], mk, ms=6.5, color=fifth_col[f],
                         mec="k" if k == "bayes" else "0.45", mew=.8)
        axr.set_xlim(0, 2)
        axr.set_ylim(.1, 1.0)
        axr.set_yticks([])
        axr.set_xticks([.5, 1.5])
        axr.set_xticklabels(["naive", "Bayes"], rotation=90, fontsize=FK)
        axr.spines[["top", "right", "left"]].set_visible(False)
        if c == 0:
            h = [Line2D([], [], color=fifth_col[k], lw=2.2, label=lab) for k, lab in
                 ((0, "shortest fifth"), (2, "middle fifth"), (4, "longest fifth"))]
            ax.legend(handles=h, loc="upper right", frameon=False, fontsize=FT - .5, handlelength=1.6,
                      title=r"path length $S_i$", title_fontsize=FT - .5)
    # g: path-length gradient of |rho|
    ax = place(1, 2)
    ax.axvspan(fl_g["x_end"], fl_g["x_start"], color=COL["trained"], alpha=.16, lw=0, zorder=0)
    ax.axhline(0, color="0.6", lw=1., ls=(0, (1, 2)), zorder=1)
    for run in RUNS:
        line(ax, run, "grad")
    _ref_lines(ax, S, "grad")
    _half_mark(ax, tm[("trained", "grad")])
    ax.set_ylim(-.66, .5)
    _xaxis(ax)
    _marks(ax, cross)
    ax.set_ylabel(r"$\Delta$ mean $|\rho_i|$", fontsize=FL)
    ax.spines[["top", "right"]].set_visible(False)
    _title(ax, "g", "Long paths vs short paths", r"mean $|\rho_i|$: longest fifth $-$ shortest fifth (e, f)")
    # h: absolute size of the belief update and of its top direction
    ax = place(1, 3)
    for run in ("trained", "echo"):
        s = S[run]
        ax.plot(s["x"], s["tot"], color=COL[run], lw=2.4, zorder=3)
        ax.plot(s["x"], s["lam1"], color=COL[run], lw=1.1, zorder=3)
    _ref_lines(ax, S, "tot", lw=1.6, names=(r"Bayes $\Sigma\lambda$", r"naive $\Sigma\lambda$"), factor=(1.2, .83))
    _ref_lines(ax, S, "lam1", lw=.9, names=(r"both $\lambda_1$", None))
    ax.set_yscale("log")
    ax.set_ylim(2e-5, .2)
    _xaxis(ax)
    _marks(ax, cross)
    ax.annotate("early: thin $\\approx$ thick,\none direction carries the update", xy=(42, 5e-3),
                xytext=(11, .085), fontsize=FT - .5, color="0.2", ha="left", va="center", linespacing=1.2,
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1., shrinkA=2, shrinkB=2))
    ax.set_ylabel(r"variance of $\Delta q_t$", fontsize=FL)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([], [], color="0.3", lw=2.2, label=r"total $\Sigma\lambda$"),
                       Line2D([], [], color="0.3", lw=1.1, label=r"top direction $\lambda_1$")],
              loc="lower right", frameon=False, fontsize=FT - .5, handlelength=1.6)
    _title(ax, "h", "Size of the belief update, top direction",
           r"$\lambda_1$ and $\Sigma\lambda$ of $\mathrm{cov}_i(\Delta q_t)$ (20 probabilities)")
    h = [Line2D([], [], color=COL[r], ls=LS[r], lw=2 if not r.startswith("joint") else SECONDARY["lw"],
                alpha=1 if not r.startswith("joint") else SECONDARY["alpha"], label=NAME[r]) for r in RUNS]
    h += [Line2D([], [], color="k", lw=1.4, label="Bayes (same probes)"),
          Line2D([], [], color="0.5", lw=1.4, label="naive (same probes)"),
          Line2D([], [], color="0.25", lw=1.1, ls=":", label=f"Trained RNN passes naive ({cross:,.0f} episodes)"),
          Patch(color="0.55", alpha=.35, lw=0, label="its memory use flips by entanglement (800-1,040)"),
          Line2D([], [], color="k", lw=1.8, marker="o", ms=7.5, mfc=COL["trained"], mec="k",
                 label="its half-way crossing, naive to Bayes (5-95%)"),
          Patch(color=COL["trained"], alpha=.3, lw=0, label="g: its G crosses 0 (reliably, both halves)")]
    fig.legend(handles=h, loc="center", bbox_to_anchor=(.5, 1.68 / H), ncol=5, frameon=False, fontsize=FT,
               columnspacing=1.8, handlelength=2.4)
    fig.text(.5, .62 / H, "Data: 10,000 probe episodes (50 environments x 100 latent pairs x 2 trajectories, 30 "
             "steps; the same for every network and for Bayes and naive), 29 jumps each (steps 2-30) = 290,000 jumps "
             "per checkpoint; 114 checkpoints per network (seed 20261029).\na, b: the mean over each bin's jumps (grey: "
             f"fewer than {MIN_N}); white line: half of " r"$\Sigma|\Delta\mu_t|^2$" " comes from larger jumps.   c: "
             "independent = variable 0's jumps paired with variable 1's jumps of another episode (same environment, goal "
             "and step; next latent pair).\ne, f: the Pearson $r$ over each episode's 29 steps, then $|r|$ averaged "
             r"within fifths of path length $S_i = \Sigma_t |\Delta\mu_{it}|$ (2,000 episodes each); dotted: one "
             "variable's steps shuffled within the episode.   h: per step, over the 10,000 episodes; geometric mean over "
             "steps 2-30\n(the gap between a run's thin and thick line is the top direction's share).   Half-way "
             "crossings: the training episode after which the curve stays past the midpoint of naive and Bayes; 5-95% "
             "over 1,000 resamples of the 50 environments.",
             ha="center", va="center", fontsize=FT - .5, color="0.3", linespacing=1.6)
    path = fig_path("td_tt_jumps.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


def figure_steps(D=None, S=None):
    """-> figure_bin/td_tt_jumps_steps.png: the excess angle step by step"""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    D = D or load()
    S = S or summary(D)
    cross = passes_naive()
    PW, PH, SW, GX, GXC = 3.9, 2.9, .34, 1.3, 2.0
    L, TOP, BOT = .95, 1.0, 1.95
    W = L + 3 * PW + GX + GXC + 1.0
    H = TOP + PH + BOT
    fig = plt.figure(figsize=(W, H))
    xcol = [L, L + PW + GX, L + 2 * PW + GX + GXC]
    y0 = H - TOP - PH
    steps = np.arange(2, T + 1)
    se = np.arange(1.5, T + 1)
    VMAX = 16
    fig.text(.5, 1 - .3 / H, r"Where in the episode the two variables move together:  excess angle per step "
             r"$\bar\varphi_t - \bar\varphi_{t,\mathrm{indep}}$,  $\bar\varphi_t = \Sigma_i\, |\Delta\mu_{it}|^2 "
             r"\varphi_{it} \,/\, \Sigma_i\, |\Delta\mu_{it}|^2$  (over the 10,000 probe episodes)", ha="center",
             va="center", fontsize=TS + 1)
    xe = _edges(S["trained"]["x"])
    for c, run in enumerate(("trained", "echo")):
        ax = fig.add_axes([xcol[c] / W, y0 / H, (PW - SW - .1) / W, PH / H])
        im = ax.pcolormesh(xe, se, S[run]["excess_step"].T, cmap="viridis", vmin=0, vmax=VMAX, shading="flat",
                           rasterized=True)
        st = step_timing(S, D["boot"], run)
        ok = np.isfinite(st["point"])
        if ok.any():
            lo = np.where(np.isfinite(st["lo"]), st["lo"], np.nan)
            hi = np.where(np.isfinite(st["hi"]), st["hi"], _xmax())
            ax.fill_betweenx(steps, lo, hi, where=ok, color="w", alpha=.35, lw=0, zorder=4)
            ax.plot(np.where(ok, st["point"], np.nan), steps, color="w", lw=1.8, zorder=5,
                    path_effects=[pe.withStroke(linewidth=3.2, foreground="k")])
        else:
            ax.text(.5, .5, "no step reaches\nhalf-way", transform=ax.transAxes, ha="center", va="center",
                    fontsize=FT + 1, color="w", zorder=5, path_effects=[pe.withStroke(linewidth=2.5, foreground="k")])
        _xaxis(ax)
        _marks(ax, cross, on_heat=True)
        ax.set_ylim(1.5, T + .5)
        ax.set_yticks([2, 5, 10, 15, 20, 25, 30])
        ax.set_ylabel("step $t$ of the episode", fontsize=FL)
        _title(ax, "ab"[c], f"{NAME[run]}: coupling by step",
               r"$\bar\varphi_t - \bar\varphi_{t,\mathrm{indep}}$ per step and checkpoint")
        axr = fig.add_axes([(xcol[c] + PW - SW) / W, y0 / H, SW / W, PH / H])
        _strip_heat(axr, S, "excess_step", se, "viridis", 0, VMAX)
        axr.set_ylim(1.5, T + .5)
    cax = fig.add_axes([(xcol[1] + PW + .12) / W, y0 / H, .12 / W, PH / H])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_ticks([0, 4, 8, 12, 16])
    cb.ax.tick_params(labelsize=FK)
    cb.set_label("excess angle (degrees)", fontsize=FL - .5)
    ax = fig.add_axes([xcol[2] / W, y0 / H, PW / W, PH / H])
    ax.axhline(0, color="0.6", lw=1., ls=(0, (1, 2)))
    for key, col in (("bayes", "k"), ("naive", "0.5")):
        ax.plot(steps, S[key]["excess_step"][0], color=col, lw=1.8)
    blues = plt.cm.Blues(np.linspace(.35, 1., len(STAGES)))
    for j, xv in enumerate(STAGES):
        ax.plot(steps, S["trained"]["excess_step"][at(S, "trained", xv)], color=blues[j], lw=2)
    ax.plot(steps, S["echo"]["excess_step"][-1], color=COL["echo"], lw=2)
    ax.set_xlim(1.5, 30.5)
    ax.set_ylim(-2, 18)
    ax.set_xticks([2, 5, 10, 15, 20, 25, 30])
    ax.tick_params(labelsize=FK)
    ax.set_xlabel("step $t$ of the episode", fontsize=FL)
    ax.set_ylabel("excess over independent (degrees)", fontsize=FL)
    ax.spines[["top", "right"]].set_visible(False)
    _title(ax, "c", "Trained RNN at five stages, by step", r"$\bar\varphi_t - \bar\varphi_{t,\mathrm{indep}}$; "
           "Bayes and naive on the same probes")
    h = [Line2D([], [], color=blues[j], lw=2, label=f"Trained RNN at {xv:,}") for j, xv in enumerate(STAGES)]
    h += [Line2D([], [], color=COL["echo"], lw=2, label="ESN at 39,996"), Line2D([], [], color="k", lw=1.8, label="Bayes"),
          Line2D([], [], color="0.5", lw=1.8, label="naive"),
          Line2D([], [], color="w", lw=1.8, path_effects=[pe.withStroke(linewidth=3.2, foreground="k")],
                 label="a: half-way crossing per step (band 5-95%)"),
          Line2D([], [], color="0.25", lw=1.1, ls=":", label=f"a, b: Trained RNN passes naive ({cross:,.0f})")]
    fig.legend(handles=h, loc="center", bbox_to_anchor=(.5, 1.12 / H), ncol=5, frameon=False, fontsize=FT,
               columnspacing=1.6, handlelength=2.4)
    fig.text(.5, .42 / H, "Data: 10,000 probe episodes (50 environments x 100 latent pairs x 2 trajectories) per checkpoint, "
             "114 checkpoints per network; the jump into step t is weighted by its squared length.\nIndependent: variable "
             "0's jumps paired with variable 1's jumps of another episode (same environment, goal and step; next latent "
             "pair).   White band: the Trained RNN's memory-use flip (800-1,040).\nHalf-way per step: after it the "
             "step's excess stays past the midpoint of naive's and Bayes's at that step; band: 5-95% over 1,000 "
             "resamples of the 50 environments.", ha="center", va="center", fontsize=FT - .5, color="0.3",
             linespacing=1.6)
    path = fig_path("td_tt_jumps_steps.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


# the headline measures exported for the cross-topic synthesis: (curve key, short name, family)
TIMELINE = (("excess", "jump excess angle (deg)", "movement"),
            ("grad", "long-path coupling G", "movement"),
            ("band0", "angle of jumps >= 4 cells (deg)", "movement"),
            ("share", "update top-direction share", "movement"))


def timeline(D=None, S=None):
    """-> through_training/jumps/timeline.npz: the headline measures of the Trained RNN and the ESN (TIMELINE) at the
    114 checkpoints with 5-95% environment-bootstrap bands, Bayes's and naive's levels, the crossing level and the
    crossing with its interval (timing(); NaN = never). Levels: half-way between naive and Bayes, except the share:
    half-way from its peak down to Bayes (the end of the early one-direction phase)"""
    import warnings
    D = D or load()
    S = S or summary(D)
    B, tm = D["boot"], timing(S, D["boot"])
    rows = []
    for run in ("trained", "echo"):
        for key, name, family in TIMELINE:
            t = tm[(run, key)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)                     # all-NaN columns (no jumps >= 4)
                lo, hi = np.nanpercentile(B[f"{run}|{key}"], [5, 95], 0)
            rows.append(dict(names=name, run=run, family=family, value=S[run][key], lo=lo, hi=hi,
                             bayes=S["bayes"][key][0], naive=S["naive"][key][0], level=t["level"], half=t["point"],
                             half_lo=t["lo"], half_hi=t["hi"]))
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["x"] = S["trained"]["x"]
    path = os.path.join(OUT, "timeline.npz")
    np.savez(path, **out)
    print("saved", path)
    for r in rows:
        print(f"  {r['run']:8s} {r['names']:34s} {r['family']:9s} level {r['level']:+.3f} (naive {r['naive']:+.3f}, "
              f"Bayes {r['bayes']:+.3f}) half {r['half']:,.0f} [{r['half_lo']:,.0f}-{r['half_hi']:,.0f}]")
    return path


SYNTHESIS = (("acc", "goal accuracy at T", "answers"),
             ("excess", "jump excess angle over independent", "coupling"),
             ("grad", "long-path coupling G", "coupling"),
             ("band0", "angle of jumps >= 4 cells", "coupling"))


def synthesis(D=None, S=None):
    """-> through_training/jumps/synthesis.npz (td_tt_gap.export: gap fraction toward Bayes, passes naive and the
    naive-Bayes midpoint for good, on the shared environment resamples): goal accuracy, the excess angle, G and the
    angle of jumps >= 4 cells, Trained RNN and ESN. The update top-direction share is left out (naive .117 vs Bayes
    .087: the gap is small next to its early peak, so its gap fraction is not a timing measure)"""
    import td_tt_gap as gap
    D = D or load()
    S = S or summary(D)
    B = D["boot"]
    assert np.array_equal(boot_counts(), gap.boot_counts())
    rows = []
    for run in ("trained", "echo"):
        for key, name, family in SYNTHESIS:
            note = "ESN: NaN at checkpoints with < 50 jumps >= 4 cells" if (key == "band0" and run == "echo") else ""
            rows.append(dict(name=name, run=run, family=family, shape="rise", sample="probes: 50 environments x 200",
                             paired=True, note=note, value=S[run][key], boot=B[f"{run}|{key}"],
                             naive=S["naive"][key][0], bayes=S["bayes"][key][0],
                             naive_boot=B[f"naive|{key}"][:, 0], bayes_boot=B[f"bayes|{key}"][:, 0]))
    return gap.export("jumps", S["trained"]["x"], rows)


# ═══════════════════════════════════════════════════════════════════
# Numbers (every number of the docstring) and the checks
# ═══════════════════════════════════════════════════════════════════

def numbers(D=None, S=None):
    """prints every number of the module docstring"""
    D = D or load()
    S = S or summary(D)
    B = D["boot"]
    tm = timing(S, B)
    cross = passes_naive()
    f2, f3 = (lambda v: f"{v:.2f}"), (lambda v: f"{v:.3f}")
    iv = lambda t: f"{t['point']:,.0f} [{t['lo']:,.0f}-{t['hi']:,.0f}] (crosses in {t['frac']:.0%})"  # noqa: E731
    sg = lambda v: "/".join(f"{np.mean(v[a:b]):+.1f}" for a, b in ((0, 3), (3, 8), (8, 18), (18, 29)))  # noqa: E731
    print(f"Trained RNN passes naive at {cross:,.0f} training episodes (landmark); environment bootstrap of the "
          f"log-interpolated crossing of naive's accuracy: {iv(tm[('trained', 'acc_naive')])}")
    print("\n== references (probes) ==")
    for k in ("bayes", "naive"):
        s, z = S[k], D["ref"]
        print(f"{k}: acc {f3(s['acc'][0])} (per-environment mean {f3(s['acc_env_mean'][0])}); phi_bar {f2(s['phibar'][0])}"
              f" (cache a2w {f2(s['a2w'][0])}), independent {f2(s['indep'][0])}, excess {s['excess'][0]:+.2f}; null "
              f"variants: all 99 shifts {f2(s['indep_all'][0])} (excess {s['excess_all'][0]:+.2f}), same latent pair "
              f"{f2(s['indep_cell'][0])} (excess {s['excess_cell'][0]:+.2f})")
        print(f"   a1w {f2(s['a1w'][0])}, top-1% angle {f2(s['phi_top1'][0])}; mean angle >= 4 cells "
              f"{f2(s['band0'][0])} ({s['band_cnt'][0, 0]:.0f} jumps; amplitude bins {f2(_band_angle(z, k, 4, 16))}), "
              f"2-4 {f2(s['band1'][0])} ({s['band_cnt'][0, 1]:.0f}), 1-2 {f2(_band_angle(z, k, 1, 2))}, 1/8-1 "
              f"{f2(_band_angle(z, k, .125, 1))}; a_half {f2(s['a_half'][0])}")
        print(f"   |rho| all {f3(s['rho_all'][0])}, shuffled {f3(s['null_all'][0])}, fifths {np.round(s['fifths'][0], 3)}"
              f", G {s['grad'][0]:+.3f} (Spearman {s['grad_spear'][0]:+.3f}, steps 5-30 {s['grad_s5'][0]:+.3f}, deciles "
              f"{s['grad_dec'][0]:+.3f})")
        print(f"   update: share (geometric mean over steps) {f3(s['share'][0])}, arithmetic mean {f3(s['dq_top'][0])}; "
              f"lambda_1 {s['lam1'][0]:.5f}, total {s['tot'][0]:.5f}; log-increment PR {f2(s['inc_pr'][0])}")
        print(f"   per step 2-4/5-9/10-19/20-30: phi_bar_t {_step_groups(s['step_angle'][0])}, independent "
              f"{_step_groups(s['indep_step'][0])}, excess {sg(s['excess_step'][0])}; 22.5 crossings (isotropic) at steps "
              f"{[round(v, 1) for v in _crossings(np.arange(2, T + 1), s['step_angle'][0], FLAT)]}")
    for key in ("excess", "grad", "band0", "acc", "phibar"):
        lv = (S["naive"][key][0] + S["bayes"][key][0]) / 2
        print(f"   half-way level {key}: {lv:+.3f}")
    print("   quoted, not recomputed: the 20k-harvest analysis (fig1c_crossings.py, other networks and probes), mean angle "
          "of jumps >= 4 cells: Optimal 39.5, Trained 37.5, Naive 5.9, ESN 4.2")
    for run in RUNS:
        s = S[run]
        x = s["x"]
        print(f"\n== {run} ==")
        for name in TIMED:
            if (run, name) in tm:
                t = tm[(run, name)]
                print(f"  crossing {name:12s} level {t['level']:+8.3f}: {iv(t)}")
        pairs = {"trained": (("acc", "band0"), ("acc", "grad"), ("acc", "excess"), ("band0", "grad"), ("band0", "excess"),
                             ("grad", "excess"), ("grad0", "excess"), ("band0", "band1"), ("band1", "excess"),
                             ("acc_naive", "band0"), ("acc_naive", "grad0"), ("acc_naive", "excess")),
                 "joint_trained": (("excess", "acc_naive"), ("grad0", "acc_naive"), ("grad", "acc_naive"),
                                   ("band0", "acc_naive"))}.get(run, ())
        if run == "trained":
            pairs = pairs + (("acc", "acc_naive"), ("band0", "acc_naive"), ("grad0", "acc_naive"), ("grad", "acc_naive"),
                             ("excess", "acc_naive"))
        for a, b in pairs:
            r = order(tm[(run, a)]["boot"], tm[(run, b)]["boot"])
            print(f"  paired order, crossing {a} / crossing {b}: {r[0]:.2f} [{r[1]:.2f}-{r[2]:.2f}] ({r[3]:.0%} of "
                  "resamples)")
        if run == "trained":
            gt = group_timing(S, B, run)
            gname = [f"{a + 2}-{b + 1}" for a, b in STEP_GROUPS]
            for i, j in ((1, 0), (2, 0), (1, 2), (3, 1), (3, 2), (3, 0)):
                r = order(gt[i]["boot"], gt[j]["boot"])
                print(f"  paired order, steps {gname[i]} / steps {gname[j]}: {r[0]:.2f} [{r[1]:.2f}-{r[2]:.2f}] "
                      f"({r[3]:.0%} of resamples)")
        for name, G, GA, GB in (("phi_bar - 22.5 (isotropic)", s["a2w"] - FLAT, s["a2w_A"] - FLAT, s["a2w_B"] - FLAT),
                                ("G", s["grad"], s["grad_A"], s["grad_B"])):
            for f in flip_list(x, G, GA, GB):
                print(f"  reliable sign change of {name}: {f['before']:+.3f} -> {f['after']:+.3f} between "
                      f"{f['x_end']:,.0f} and {f['x_start']:,.0f} (halves agree {f['rep']:.0%}, sigma {f['sigma']:.3g})")
        for key in ("acc", "phibar", "grad"):                                               # superseded definition
            mid = (S["naive"][key][0] + S["bayes"][key][0]) / 2
            past = (s[key] - mid) * np.sign(S["bayes"][key][0] - S["naive"][key][0]) > 0
            first = next((float(x[k]) for k in range(len(x)) if past[k:].all()), np.nan)
            print(f"  former half-way ({key}: the first checkpoint from which it stays past the midpoint): {first:,.0f}")
        for key in ("excess", "grad", "band0", "acc", "phibar"):
            for xv in (880, 1040):
                f, lo, hi = gap_fraction(S, B, run, key, xv)
                print(f"  share of the naive -> Bayes distance at {xv:,}, {key}: {f:.1%} [{lo:.1%}-{hi:.1%}]")
        for key, ref in (("excess", "bayes"), ("excess", "naive"), ("grad", "bayes"), ("grad", "naive"),
                         ("band0", "bayes"), ("acc", "bayes"), ("acc", "naive")):
            d, lo, hi = paired_diff(S, B, run, key, ref)
            print(f"  end of training, {key} minus {ref}: {d:+.3f} [{lo:+.3f}-{hi:+.3f}]")
        m = x >= 240
        print(f"  from 240 on: excess {s['excess'][m].min():+.2f} to {s['excess'][m].max():+.2f}, G "
              f"{s['grad'][m].min():+.3f} to {s['grad'][m].max():+.3f}, independent {s['indep'][m].min():.2f}-"
              f"{s['indep'][m].max():.2f}, phi_bar {s['phibar'][m].min():.2f}-{s['phibar'][m].max():.2f}")
        m = x >= 400
        print(f"  from 400 on: excess {s['excess'][m].min():+.2f} to {s['excess'][m].max():+.2f}, G "
              f"{s['grad'][m].min():+.3f} to {s['grad'][m].max():+.3f}, phi_bar {s['phibar'][m].min():.2f}-"
              f"{s['phibar'][m].max():.2f}; max accuracy {s['acc'].max():.3f}")
        m = x >= 640
        print(f"  from 640 on: independent {s['indep'][m].min():.2f}-{s['indep'][m].max():.2f}")
        m = (x >= 400) & (x <= 1280)
        print(f"  400-1,280: G Spearman {s['grad_spear'][m].min():+.3f} to {s['grad_spear'][m].max():+.3f}")
        if run == "trained":
            st = step_timing(S, B, run)
            ok = np.isfinite(st["point"])
            print(f"  per-step half-way crossings: {ok.sum()} of 29 steps cross; points {np.nanmin(st['point']):,.0f}-"
                  f"{np.nanmax(st['point']):,.0f} (median {np.nanmedian(st['point']):,.0f}); 5% bounds "
                  f"{np.nanmin(st['lo']):,.0f}-{np.nanmax(st['lo']):,.0f}, 95% bounds {np.nanmin(st['hi']):,.0f}-"
                  f"{np.nanmax(st['hi']):,.0f}")
            for t_ in range(T - 1):
                print(f"     step {t_ + 2:2d}: level {st['level'][t_]:+.2f}, crossing {st['point'][t_]:,.0f} "
                      f"[{st['lo'][t_]:,.0f}-{st['hi'][t_]:,.0f}] ({st['frac'][t_]:.0%})")
        else:
            st = step_timing(S, B, run)
            print(f"  per-step half-way crossings: {np.isfinite(st['point']).sum()} of 29 steps cross")
        for (a, b), g in zip(STEP_GROUPS, group_timing(S, B, run)):
            print(f"  step group {a + 2}-{b + 1}: level {g['level']:+.2f}, crossing {iv(g)}")
        print("  checkpoint: acc | phi_bar indep excess | >=4 cells angle (n) 2-4 (n) | |rho| fifths G | update share "
              "(geo / arith) lambda_1 total, weaker-variable loading | PR(u) drift | per step: excess 2-4/5-9/10-19/"
              "20-30, phi_bar_t, 22.5 crossings")
        for xv in (10, 20, 40, 60, 80, 120, 160, 200, 240, 260, 400, 640, 800, 880, 1040, 1200, 1280, 1440, 1600, 2000,
                   3200, 3400, 4000, 4600, 5600, 5800, 6600, 6800, 8000, 12000, 16000, 24000, 39996):
            k = at(S, run, xv)
            print(f"  {x[k]:7,.0f}: {f3(s['acc'][k])} | {f2(s['phibar'][k])} {f2(s['indep'][k])} {s['excess'][k]:+.2f} | "
                  f"{f2(s['band0'][k])} ({s['band_cnt'][k, 0]:.0f}) {f2(s['band1'][k])} ({s['band_cnt'][k, 1]:.0f}) | "
                  f"{f3(s['rho_all'][k])} {np.round(s['fifths'][k], 3)} {s['grad'][k]:+.3f} (Spearman "
                  f"{s['grad_spear'][k]:+.3f}) | {f3(s['share'][k])} / "
                  f"{f3(s['dq_top'][k])} {s['lam1'][k]:.5f} {s['tot'][k]:.5f}, {f3(s['weak_load'][k])} | "
                  f"{f2(s['pr_u'][k])} {s['drift'][k]:.1%} | {sg(s['excess_step'][k])}, "
                  f"{_step_groups(s['step_angle'][k])}, "
                  f"{[round(v, 1) for v in _crossings(np.arange(2, T + 1), s['step_angle'][k], FLAT)]}")
        print(f"  shuffled-step |rho| over checkpoints {s['null_all'].min():.3f}-{s['null_all'].max():.3f}; jumps below "
              f"1/128 cell (outside the amplitude bins) {s['n_below'].sum() / (len(x) * 10000 * 29):.1%}")
        for x0 in (120, 240):
            print(f"  drift share (the probe-average jump's part of the squared movement) from {x0} on: at most "
                  f"{s['drift'][x >= x0].max():.1%}")
        if run == "echo":
            m = (x >= 40) & (x <= 80)
            print(f"  40-80: share {s['share'][m].min():.3f}-{s['share'][m].max():.3f} (arithmetic "
                  f"{s['dq_top'][m].min():.3f}-{s['dq_top'][m].max():.3f}), weaker loading {s['weak_load'][m].min():.3f}-"
                  f"{s['weak_load'][m].max():.3f}, phi_bar {s['phibar'][m].min():.2f}-{s['phibar'][m].max():.2f}, excess "
                  f"{s['excess'][m].min():+.2f} to {s['excess'][m].max():+.2f}")
            m = (x >= 120) & (x <= 160)
            print(f"  120-160: PR(u) {s['pr_u'][m].min():.2f}-{s['pr_u'][m].max():.2f}")
    path = os.path.join(OUT, "jumps_checks.npz")
    print("\n== checks (stored by the check command) ==")
    print("\n".join(np.load(path)["lines"]) if os.path.exists(path) else "  (run the check command first)")


def _band_angle(z, key, lo, hi, k=None):
    """mean angle of the jumps with lo <= amplitude < hi from the stored amplitude bins (bin edges ~ lo, hi)"""
    pre = "" if key is None else key + "_"
    cnt, sphi = np.asarray(z[pre + "cnt"]), np.asarray(z[pre + "sphi"])
    if k is not None:
        cnt, sphi = cnt[k], sphi[k]
    m = (A_EDGES[:-1] >= lo * (1 - 1e-9)) & (A_EDGES[1:] <= hi * (1 + 1e-9))          # bins inside [lo, hi)
    return sphi[m].sum() / cnt[m].sum() if cnt[m].sum() else np.nan


def _crossings(t, y, level):
    """the t where y crosses level (linear interpolation between neighbours)"""
    s = np.sign(y - level)
    k = np.where(s[:-1] * s[1:] < 0)[0]
    return [t[i] + (level - y[i]) / (y[i + 1] - y[i]) * (t[i + 1] - t[i]) for i in k]


def _step_groups(sa):
    return "/".join(f"{np.mean(sa[a:b]):.1f}" for a, b in ((0, 3), (3, 8), (8, 18), (18, 29)))


def check_fp32(n=2000):
    """the stored fp16 log beliefs against an fp32 CPU forward of the checkpoint (td_entanglement_driver's weights-only
    LSTM) on n probes (perm[:n]) -> printed lines"""
    import torch
    import td_analysis as td
    import td_entanglement_driver as tdd
    torch.set_num_threads(2)
    P = np.load(os.path.join(AN, "probes.npz"))
    rows = P["perm"][:n]
    obs = torch.as_tensor(P["obs"][rows], dtype=torch.float32)
    zin = torch.as_tensor(P["Z"][P["env"][rows]]).reshape(n, -1)
    perm = _null_perm(n, T - 1)
    lines = []
    for run, ep in (("trained", 0), ("trained", 50), ("trained", 1000), ("trained", 49995), ("echo", 150),
                    ("echo", 49995)):
        w = torch.load(td.ckpt_path(run, ep), map_location="cpu", weights_only=False)["weights"]
        p = {k: v.float() for k, v in w.items()}
        with torch.no_grad():
            u = tdd._run(p, tdd._pre(p, obs, zin), *tdd._start(p, n))
            lg = tdd._readout(p, u).double().cumsum(1).reshape(n, T, 2, R).numpy()
        st = np.load(os.path.join(AN, run, f"ck_{ep:05d}.npz"))["logq"][rows]
        a, b = stats(lg, perm), stats(st, perm)
        mu = [np.exp(v - logsumexp(v, -1, keepdims=True)) @ RV for v in (lg, st.astype(np.float64))]
        lines.append(f"  fp32 {run} E={ep}: phi_bar {a['a2w']:.3f} / {b['a2w']:.3f}, |rho| {np.abs(a['rho']).mean():.4f}"
                     f" / {np.abs(b['rho']).mean():.4f}, path {a['path'].mean():.4f} / {b['path'].mean():.4f} (fp32 / "
                     f"stored); max |d mu| {np.abs(mu[0] - mu[1]).max():.4f} cells")
    return lines


def check_drift(eps=(0, 25, 50, 100, 150, 200, 300, 500, 1100, 5000, 49995)):
    """the same statistics after removing the probe-average jump of each step (the input-independent drift of the
    mean belief): phi_bar, its excess over the matched null and mean |rho|, raw / drift removed -> printed lines"""
    lines, worst = [], {}
    partner = null_partner(N_ENV * PER_ENV)
    for run in ("trained", "echo"):
        for ep in eps:
            lq = np.load(os.path.join(AN, run, f"ck_{ep:05d}.npz"))["logq"].astype(np.float64)
            lq = lq - logsumexp(lq, -1, keepdims=True)
            d = np.diff(np.exp(lq) @ RV, axis=1)
            out = []
            for dd in (d, d - d.mean(0, keepdims=True)):
                a = np.abs(dd)
                w = (a ** 2).sum(-1)
                a1 = a[partner, :, 1]
                wn = a[..., 0] ** 2 + a1 ** 2
                pb = (w * _angle(a[..., 0], a[..., 1])).sum() / w.sum()
                out.append((pb, pb - (wn * _angle(a[..., 0], a1)).sum() / wn.sum(),
                            np.abs(_pearson(dd[..., 0], dd[..., 1])).mean()))
            lines.append(f"  drift {run} {train_x(ep):6.0f}: phi_bar {out[0][0]:.2f} / {out[1][0]:.2f}, excess "
                         f"{out[0][1]:+.2f} / {out[1][1]:+.2f}, |rho| {out[0][2]:.3f} / {out[1][2]:.3f} (raw / drift "
                         "removed)")
            if train_x(ep) >= 240:
                ch = np.abs(np.subtract(out[1], out[0]))
                worst[run] = np.maximum(worst.get(run, 0.), ch)
    for run, ch in worst.items():
        lines.append(f"  drift {run}, largest change from 240 on: phi_bar {ch[0]:.2f}, excess {ch[1]:.2f}, |rho| "
                     f"{ch[2]:.3f}")
    return lines


def check_floor(eps=(("trained", 50), ("trained", 1100), ("trained", 49995), ("echo", 150), ("echo", 49995))):
    """the update-dimension panel that was dropped: PR of the per-step belief increment over the 10,000 episodes (per
    step, then averaged) in log space (centred per variable) with no floor, a floor 30 / 10 nats below each marginal's
    max, and in probability space -> printed lines"""
    P = np.load(os.path.join(AN, "probes.npz"))
    rows = [("bayes", P["log_jb"]), ("naive", P["log_nb"])]
    rows += [(f"{r} {train_x(e):.0f}", np.load(os.path.join(AN, r, f"ck_{e:05d}.npz"))["logq"]) for r, e in eps]
    lines = []
    for name, lq in rows:
        lq = lq.astype(np.float64)
        lq = lq - logsumexp(lq, -1, keepdims=True)
        out = []
        for floor in (None, 30., 10., "prob"):
            if floor == "prob":
                x = np.exp(lq)
            else:
                x = lq if floor is None else np.maximum(lq, lq.max(-1, keepdims=True) - floor)
            inc = np.diff(x, axis=1)
            if floor != "prob":
                inc = inc - inc.mean(-1, keepdims=True)
            inc = inc.reshape(len(lq), T - 1, 2 * R)
            out.append(np.mean([_pr(inc[:, t])[0] for t in range(T - 1)]))
        lines.append(f"  floor {name:14s} log {out[0]:.2f}, floor -30 {out[1]:.2f}, floor -10 {out[2]:.2f}, "
                     f"probability {out[3]:.2f}")
    return lines


def checks():
    """runs the three checks and stores their lines in through_training/jumps/jumps_checks.npz (printed by numbers)"""
    lines = check_fp32() + check_drift() + check_floor()
    np.savez(os.path.join(OUT, "jumps_checks.npz"), lines=np.array(lines))
    print("\n".join(lines))
    return lines


if __name__ == "__main__":
    what = sys.argv[1:] or ["data", "figure", "numbers", "timeline"]
    D = load()
    S = summary(D)
    if "check" in what:
        checks()
    if "figure" in what:
        figure(D, S)
        figure_steps(D, S)
    if "numbers" in what:
        numbers(D, S)
    if "timeline" in what:
        timeline(D, S)
    if "synthesis" in what:
        synthesis(D, S)
