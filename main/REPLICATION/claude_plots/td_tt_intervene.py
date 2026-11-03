"""Shared intervention engine for the inference-time causal experiments on the stored training-dynamics checkpoints
(John 11-02: link the neural changes to the behavioural developments through training, ideally causally - "How does
the network get to the correct answer before learning optimal bayes dynamics?"; approved: (1) memory cuts with donor
states, (2) weight swaps along training, (3) lesions through training; inference only, no new trainings). This module
builds and validates the engine that the three experiments share. It draws no conclusion about the network: the
smoke-test numbers below are not experiments.

DATA
  Checkpoints main/DATA/experimental/training_dynamics/<run>_s20261029/ckpt_epEEEEE_net.pth (114 per run; E = the raw
  episode of the file; training episodes E - E // 5, the initial network drawn at 10). Probes analysis/probes.npz:
  10,000 episodes = 50 test environments x 100 cells x 2 streams of 30 steps; environment = row // 200; the two streams
  of a cell are rows 2k, 2k + 1. Forward set: td_tt_wormholes.grid_side() (5,028 grid episodes of the same 50
  environments with a corner or one-variable end split, plus 12 unselected trajectories per corner cell).

API (import td_tt_intervene as ti)
  load(run, E) -> w: the 10 fp32 tensors of KEYS (STM = h0, LTM = c0, read-in W / b, W_ih, W_hh, b_ih, b_hh, read-out
    W / b). Loads trained, echo, trained_p1, trained_N50 / N100 / N200, trained_lastT(_p1) (single-Linear read-in, 20-row
    marginal read-out); refuses mlp55 and the joint read-outs.
  groups(H) -> {name: [(key, row slice), ...]}: 15 fine groups that partition every weight: readin; W_ih_i / _f / _g /
    _o and W_hh_i / ... (rows k H .. (k + 1) H of gate block k, order i, f, g, o); b_i / ... (the b_ih and b_hh rows of
    the block); readout; start. GROUP_SETS: W_ih, W_hh, bias (4 blocks each), lstm (all 12), gate_i / _f / _g / _o
    (W_ih + W_hh + b of one gate), all.
  hybrid(wa, wb, names, alpha) -> w: wa, except the named groups = (1 - alpha) wa + alpha wb (alpha 0 / 1: the parents
    exactly). Pass the hybrid to forward as its weights.
  forward(w, rows=None, device=None, cut=None, clamp=None, gate=None, keep=(), record=None, episodes=None, ref=None,
    budget=30000) -> dict(rows, logq (n, 30, 2, 10) float32, right (n, 30, 2) bool, [logq_cut, right_cut: {name: ...}],
    [u, c, gates, logits], info). rows: any subset of the probes (or of episodes, e.g. forward_set()), returned in the
    caller's order; device None = cuda:0 when CUDA_VISIBLE_DEVICES is 0 or 1 (one visible GPU, cuda:0 inside the
    process; 1 only when John has freed GPU 1), else the CPU; budget = GPU rows per chunk x (1 + live memory-cut
    windows) (30,000: <= 3 GB).
  reference(w, device, cov=False) -> the clean-run constants of the hooks (below); orth_readout(V, w) (the basis a u
    clamp acts on), random_subspace(k, w, seed), pcs_off_readout(k, w, ref), captured_variance(V, ref, w) (the clean
    variance in that basis): for clamps and their variance-matched controls.
  score(logq, 'cheap' | 'full', right=None) -> {name: row}; score_fwd(logq_fwd) -> the forward-set rows;
    resample(rows) -> {name: (1,000,)}; refs() -> {'bayes': rows, 'naive': rows}; gap_rows(rows) -> gap fractions with
    their resamples; curves([rows per checkpoint]) -> {name: (value (K,), boot (1,000, K))} for td_tt_gap.export.
  donors(E, 'same_cell' | 'random', seed) -> one donor episode per episode.

FORWARD (td_analysis.forward's hand-written LSTM, fp32, TF32 off)
  drive x_t = relu(W_in [o_t, Z] + b_in) W_ih^T + (b_ih + b_hh), tabulated once per (environment, observation pattern)
  (td_tt_wormholes.replay's trick); pre_t = x_t + W_hh u_(t-1); i, f, g, o = sig, sig, tanh, sig of the four blocks;
  c_t = f c_(t-1) + i g; u_t = o tanh(c_t); read-out increment l_t = W_out u_t + b_out; log beliefs = log softmax per
  variable of the cumulative sum of l_1..l_t (float64), returned as float32; right = argmax == truth from the float64
  values. Start state (STM, LTM). Probes run in chunks of whole environments (every donor lives in its episode's
  environment); a subset runs together with its donors only.
HOOKS (each optional, all composable; every per-step constant is indexed by the absolute step s = 1..30 and comes from
  the clean run of the same weights on all 10,000 probes: reference(w))
  (a) memory cut, cut=(w, source) or a list of them: at every step t > w the cell is rerun over the inputs t-w+1..t
    from the source's state at t - w, and the read-out of the rerun state at t (W_out u'_t + b_out) replaces the
    network's increment at t (t <= w: the network's own); the cumulative sum of these increments gives the belief under
    the cut at every step (logq_cut['w<w>_<source>']). source 'start': the trained start state (STM, LTM) = td_analysis
    an6 (the stored trunc_right); 'same_cell': the state at t - w of the other stream of the same cell and environment
    (row ^ 1); 'random': the state at t - w of one fixed random probe of the same environment with another cell (seed
    20261201; one donor per probe for every step); an index array (n,): any donors in the episode's own environment
    (forward asserts it). A donor's state comes from its own run under the same hooks; the rerun window gets the same
    hooks.
  (b) weight hybrids: hybrid(...) passed as w (its own clean run sets the hooks' constants).
  (c) clamp={'u': V}: after every step u_s <- u_s - (u_s - mu_s) V' V'^T, mu_s the clean probe mean of u_s and V' =
    orth_readout(V, w), an orthonormal basis of P span(V), the projection of span(V) off the read-out's 20 rows (P = I
    - Q Q^T, Q an orthonormal basis of the rows), so the read-out at the same step is untouched and every effect goes
    through the dynamics. span(V') = span(V) only when V is orthogonal to the rows (random_subspace, pcs_off_readout).
    Otherwise span(V') != span(V): V' contains the part of span(V) orthogonal to the rows (k = dim span(V): dimension
    >= max(0, k - 20), generically exactly that: {0} for k <= 20, k - 20 for k > 20; 10 of the 30 clamped directions
    for the top 30 principal components of u at training 1,680 and 39,996), every other direction of V' lies outside
    span(V), and any directions of span(V) inside the rows' span are dropped (then dim V' < k). For the top 10 principal
    components of u at training 1,680, 12.6% of their squared norm lies in the rows' span (|Q^T U|_F^2 / k, the mean
    cos^2 over any orthonormal basis U of span(V)) and V' is up to 47.3 deg from span(V) (VALIDATION). Clamp sizes and
    variance-matched controls must be measured in V': captured_variance(V, ref, w) does. clamp={'c': V}: the same on
    c_s with V' an orthonormal basis of span(V) itself (no projection); applied before u_s = o tanh(c_s).
  (d) gate={'mean': 'fi', 'resample': 'g', 'donor': (n,), 'f_units': idx, 'f_value': ..., 'c_units': idx}:
    mean-ablation replaces gate block G's recurrent drive W_hh,G u_(s-1) by its clean per-step probe mean
    W_hh,G mu_(s-1) (exact by linearity; step 1 is untouched, its drive comes from the shared start state);
    resample-ablation by the CLEAN drive of one fixed random probe of the same environment with another cell (seed
    20261203; the donor's clean run is computed alongside); f_units: those units' forget gates set to f_value
    ('median' (default) = f_med[s], the median over all units of the clean per-unit probe mean of f at step s; or a
    number, or 30 values); c_units: those units' c_s set to its clean per-step probe mean, before u_s is formed.
  Order within a step: recurrent drive (mean / resample) -> gates -> f_units -> c = f c + i g -> c_units -> clamp c ->
    u = o tanh(c) -> clamp u -> read-out.
  keep: 'u', 'c' (n, 30, H), 'gates' (n, 30, 4, H) after the hooks, 'logits' (n, 30, 20) per-step increments (1.2 GB per
    (n, 30, H) array of 10,000 probes); record(s, rows, dict(u, c, i, f, g, o, rec)): called at every step with device
    tensors of the requested rows (rec = the recurrent drive actually used).
  reference(w): mu_u, mu_c (31, H) per-step probe means (index 0 = the start state), f_mean (31, H), f_med (31,),
    mean_rec (31, 4H) = W_hh mu_u[s - 1]; cov=True adds cov_u (31, H, H), the per-step covariance of u over the probes
    (for captured_variance: the clean variance inside a clamp's actual basis V', per step and pooled; and
    pcs_off_readout).

SCORE (behaviour rows of beliefs on the 10,000 probes, with per-environment primitives; resample(rows) gives each row in
  the 1,000 shared environment resamples of td_tt_gap.boot_counts() (seed 20261103): ratios of count-weighted
  environment sums, within-step slopes refitted, G's fifths re-cut on the weights (td_tt_jumps._G_boot), environment
  means count-weighted (td_tt_gap.env_mean); refs() scores Bayes's and naive's marginals the same way, so gap fractions
  are paired)
  cheap (18 rows): acc_goal_T goal accuracy at T; acc_both_T both variables right at T; logq_goal_T log q of the true
    goal value at T; logq_both_T log q0(r0) + log q1(r1) at T; kl_goal_T / kl_goal_mean KL(Bayes || net) of the goal
    marginal at T / averaged over the 30 steps; kl_both_T / kl_both_mean the same averaged over the two variables
    (KL = Sigma_r pi(r) (log pi(r) - log q(r)), pi = the probes' log_jb renormalised, float64); far_T far errors at T
    (one or both answers >= 2 values off) and merged_T distant pairs merged at T (the 18,432 far-apart probe pairs per
    environment with the same answer cell) as td_tt_latent; excess (jump excess angle over the matched independence
    null), G (long-path coupling), band4 (mean angle of jumps >= 4 cells; NaN below 50 jumps) as td_tt_jumps;
    held_joint / held_two / held_one (corner split held, held per variable at corner-split ends and at one-variable end
    splits; Bayes's splits at steps 2-16 of the probes: 360 stream-steps, 720 and 365 var-steps) and emass_slope /
    dir_slope (the within-step slopes of the maps) as td_tt_wormholes on the probes.
  full adds (td_tt_splitting._one on the posterior means): split_grad, split_grad_matched, toward6, apart6, fold_idx,
    corners. score_fwd (the 5,028-episode forward set, td_tt_wormholes._fwd_stats_all): corner_truth_fwd,
    held_joint_fwd, held_two_fwd, held_one_fwd (the timeline's held rows), dist_out_fwd, angle_out_fwd, same_diag_fwd,
    commit_fwd.
  Reused unchanged: td_tt_wormholes._stats_all / step_slope / _fwd_stats_all / probe_side / fwd_side, td_tt_jumps
    _pearson / _angle / null_partner / rho_fifths / _G_boot, td_tt_latent.far_pairs, td_tt_splitting._one / positions,
    td_tt_gap.boot_counts / env_mean. Copied and trimmed for speed (checked bit-identical below): td_tt_jumps.stats
    (path length and Pearson rho per episode) and extra_stats (the matched-null and >= 4 cell band sums), and
    td_tt_latent.answer_stats at T only (far errors, merged pairs, both right).

VALIDATION (checks -> through_training/intervene/checks.json; numbers reprints it; GPU 0 shared with another job.
  First run 11-02 11:17, 244 s; rerun after the 11-02 fixes ('checks keep_timing', finished 15:59, 164 s): every number
  of the first run reproduced bit for bit except the wall-clock seconds and the random clamp's variance share (7e-16
  relative: a new summation order); TIMING is the first run's block, kept)
  Checkpoints: raw 0 / 1,100 / 2,100 / 5,250 / 49,995 = training 10 / 880 / 1,680 / 4,200 / 39,996 (in both the 114
  probe checkpoints and the wormholes' 58); memory cuts at raw 1,100 / 2,100 / 49,995.
  Weights. The 15 fine groups cover each of the 8,046,020 weights exactly once. hybrid(alpha 0) = wa and hybrid(alpha 1)
    = wb in every tensor, and their forwards equal the parents' (max |d logq| 0); hybrid('W_hh_f', 1) replaces exactly
    rows 1,000-1,999 of W_hh; hybrid('W_hh', .5) = the parents' mean (0).
  Intact forward vs the stored ck logq (td_analysis; fp16): max |d| .002 / .062 / .062 / .062 / .031 = at most 1.00
    fp16 ulp, i.e. the storage rounding; fp16(engine) = stored for 99.993-99.995% of the 6,000,000 entries per
    checkpoint; right identical at all 5 x 600,000 (probe, step, variable); the answers at T (argmax) differ 3 /
    0 / 1 / 0 / 1 times of 20,000, every one at an fp16 tie of the stored logq (11 / 4 / 3 / 2 / 3 ties).
  Forward set vs td_tt_wormholes' stored CPU replays: log beliefs at the split steps max |d| .002-.031 (fp16-equal >=
    99.944%), posterior means <= 3e-4, answers at T differ 2 / 1 / 0 / 1 / 0 of 5,028.
  Scores. The stored logq scored here vs every exported timeline row at those checkpoints (goal accuracy, both right,
    far errors, merged pairs, excess angle, G, >= 4 cell angle, end-mass and direction slopes, both split gradients,
    the six-step rows, fold index, opposite corners): max |d| 2.1e-14, in the values and in all 1,000 resampled
    values; the probe held rows vs td_tt_wormholes' probe cache: identical per-environment sums and counts; the trimmed
    copies bit-identical (path, rho, the four null / jump sums, both band sums, far errors, merged pairs, both right);
    refs() vs the exports' naive / Bayes levels: 1.4e-14 (resamples 2.8e-14). The engine's own float32 beliefs vs the
    exports (which scored fp16-rounded beliefs), max over the checkpoints, value / resamples / after rounding the
    engine's beliefs to fp16: accuracy, both right, far errors 1e-16 (identical answers); merged pairs 1.3e-4 / 7e-4 /
    4e-19; excess angle 7e-4 / .0023 / 1e-5 deg; G 7e-4 / .0024 / 6e-5; >= 4 cell angle .034 / .18 / 1e-5 deg; end-mass
    slope 2e-6 / 1e-5 / 1e-7; direction slope .0011 / .0041 / 1e-4; split gradient 4e-4 / .0019 / 3e-5, matched 7e-4 /
    .0023 / 2e-5; toward6 .0010 / .0026 / 1.3e-4; apart6 4e-4 / .0012 / 4e-5; fold index and opposite corners 3e-5 /
    1.5e-4 / 2e-7. Forward set (score_fwd): corner answers and held per variable at one-variable splits identical; held
    per variable at corner-split ends 1.3e-4 / 7e-4 / 0; joint held 2.6e-4 / .0015 / 0; committed 5e-4 / .0022 / 0;
    distance out 1e-7; angle out 2e-5 deg; same-diagonal share 9e-4 / .0037 (the exports' moves used float32 posterior
    means, so rounding to fp16 does not apply: .0048 deg / .0018). So the stored rows are reproduced; the engine's
    full-precision beliefs move threshold and argmax measures by these amounts (1-2 moves, stream-steps or var-steps
    of the forward set): compare intervened and intact networks both from the engine, never an engine row against an
    export.
    Held per variable on the probes (corner / one-variable ends), engine: 0 / 0, .067 / .055, .194 / .134, .522 / .444,
    .872 / .847 (720 / 365 var-steps; the timeline's held rows are the forward set's).
  Memory cut from the start state vs the stored trunc_right: 0 of 20,000 answers at T differ, for each of w = 1, 2, 5
    at each of the 3 checkpoints (goal accuracy .2040 / .2642 / .4020 at 880, .2290 / .3133 / .5075 at 1,680, .4627 /
    .5863 / .7254 at 39,996 = the stored trunc_acc); the intact answers = trunc_right[:, 3]. One forward with the 5
    cuts (w 1, 2, 5 start; w 5 same_cell, random): 14.5 / 14.5 / 14.4 s in the rerun (8.4-8.8 s in the first run).
  Donor cuts brute-forced in float64 (4 probes x steps 8, 19, 30 x same_cell / random, w = 5, from the donors' intact
    states at t - 5): the engine's centred increment at t agrees to 4.7e-6 (it differs from the intact increment by up
    to 2.17); a cut whose donor is the probe itself equals the intact network to 1.3e-4 in log q (1 of 600,000 answers
    flips at a near-tie: different matrix shapes).
  Hooks: clamps with V = 0 and with V = the read-out's 20 rows (rank 0 after orthogonalisation) are exact no-ops (max
    |d logq| 0); a random 50-dim clamp leaves |(u_s - mu_s) V| <= 1.5e-6 after every step and the step-1 read-out
    unchanged to 2.4e-6 (|W_out V| = 1.5e-16; 4.7% of the clean variance of u, pooled over steps, the same in the
    clamped basis and in span(V) (difference 0: random_subspace is already off the rows); goal accuracy .627 vs
    intact .658 at 1,680); mean-ablating all four gate blocks leaves step 1 unchanged (9.5e-7); resample-ablating all
    four with the probe as its own donor equals the intact network exactly (0); the recorded resampled f drive equals
    W_hh,f u(s-1) of the donor's clean run (float64 brute force) to 1.1e-5; mean_rec equals the mean recorded drive to
    1.6e-5; the f and c clamp values are exact (0). Donors: same_cell = row ^ 1 for every probe; every random / gate
    donor in the probe's environment with another cell (6,311 distinct random donors; 45 probes share their cut and
    gate donor).
  Fixes 11-02 (the engine verifier's issues; checks 'fixes'). (1) _parse_cuts called np.ndim on the first element
    before testing for a tuple, so a list of exactly two cuts whose first held a donor array raised ValueError (np.ndim
    of the ragged (w, array)). Of 12 cut specs, 4 raised before the fix ([(5, d1), (2, d2)], [(5, d), (5, 'start')],
    ((5, d1), (2, d2)), [[5, d], [2, 'random']]); all 12 parse as specified now, and the other 8 get the same single /
    list decision as before, so no earlier result changes. Through forward at training 1,680, [(5, same-cell donors),
    (2, random donors)] and [(5, random donors), (5, 'start')] equal the same cuts named by source exactly (max |d logq|
    0 in the main run and the cuts, no answer differs), against cut effects of up to 138 / 107 in log q.
    (2) The u clamp acts on V' = orth_readout(V) (hook (c)); captured_variance now reports the variance in V' (that in
    span(V) as span_*). Top 10 / 30 principal components of u (td_analysis basis_V100: the stored PCA basis of u on
    the basis set, covariance about the mean over all its steps), pooled clean variance share of u in span(V) / in V':
    .2960 / .2603 and .5171 / .4721 at training 1,680, .2269 / .2214 and .4263 / .4170 at 39,996; share of span(V)'s
    squared norm in the read-out rows' span (|Q^T U|_F^2 / k, the mean cos^2) .126 / .066 and .018 / .020; largest
    principal angle between V' and span(V) 47.3 / 54.5 and 14.6 / 25.7 deg (no dimension lost). V' keeps the part of
    span(V) orthogonal to the rows: 10 of its 30 directions for the 30 PCs (principal angle 0 at both checkpoints;
    10 = 30 - 20, as Q^T U has rank 20), none for the 10 PCs; its other directions lie outside span(V) (scipy subspace
    angles of the engine's second 11-02 check; not stored in checks.json). Clamping the raw top 10 PCs at 1,680 holds
    |(u_s - mu_s) V'| <= 3e-6 after every step but leaves a mean square deviation from the clean mean of .0336 of the
    clean total variance inside span(V) (clean .2960). pcs_off_readout (top eigenvectors of P (Sigma_t C_t) P) has no
    norm in the rows' span and loses nothing to the projection (angle 0): .3193 / .5452 at 1,680 and .2930 / .5190 at
    39,996 (k = 10 / 30). The lesion experiment builds every clamp basis off the rows (td_tt_lesions: eigenvectors of
    Pc C Pc, projected random rotations) and never called captured_variance, so the fix changes none of its numbers.
SMOKE TEST (not an experiment; goal accuracy at T, memory cut to w = 5, at training 880 / 1,680 / 39,996): from the
  start state .4020 / .5075 / .7254; from the same cell's other stream .4974 / .6651 / .7576; from a random other cell
  .3730 / .4555 / .4121; intact .4927 / .6576 / .7604. The same-cell donor can beat the intact network (it adds a
  second trajectory's evidence about the same cell); the random donor carries wrong evidence; the start state carries
  none but puts the cell in its first-steps regime. The three sources bracket the use of memory older than w; none
  isolates it.
TIMING (the first run's, 11-02 11:17, kept in the rerun; best of 2, GPU 0 shared with another job at ~99% utilisation;
  peak GPU memory of this process <= 3.0 GB)
  GPU seconds per forward of the 10,000 probes: intact .59; memory cut w 5 from the start state 2.45, w 10 3.97, w 1 +
  2 + 5 together 4.07, w 5 from all three sources together 7.09; clamp of u (50 dims) .65; f mean-ablation .60; f
  resample-ablation 1.09 (all four 1.38); f clamp of 100 units .62; cut w 5 random + f resample + clamp 5.23; forward
  set (5,028) .33; reference .45 (3.52 with cov_u). CPU (2 threads): score cheap 1.21 s, full 4.73 s, resample .36-.46
  s per set, gap_rows ~1 s (its first call in a process also builds refs(): ~15 s), score_fwd .45 s; a CPU forward of
  1,000 probes 3.93 s (~40 s per 10,000). The first score() in a process builds the probe side (~3 s).
CAVEATS
  fp32 replays of networks trained under fp16 autocast (as td_analysis). Every chunking or batch shape gives the same
  result to ~1e-4 in log q (a subset, a branch batch), not bit for bit; calls with identical weights and arguments are
  bit-identical (the hybrid checks). The
  hooks' constants always come from the clean run of the same weights on the 10,000 probes, also when other episodes
  are forwarded or when a hybrid is the network. A memory-cut window started at t - w runs its steps under the hooks
  of their absolute steps (a clamp pulls a fresh start-state window towards the mean of the absolute step). Mean
  ablation moves the state off its manifold; resample ablation keeps the drive on it but from another episode. The
  forward set's same-cell donors exist only where the selected set holds at least 2 episodes of the cell (otherwise
  forward stops with an assertion). The probe held rows rest on 720 / 365 var-steps.

Run from the repository root (GPU 0 beside another job; 2 CPU threads):
    CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \\
        /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_intervene.py checks
    (then 'numbers' prints the saved checks; 'timing' reruns the timing alone; 'checks keep_timing' reruns every check
    but keeps the saved timing block)"""
import os
import sys
import json
import time
import hashlib
import numpy as np
import torch
import torch.nn.functional as Fn
from scipy.special import logsumexp

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_jumps as tj                                                                      # noqa: E402
import td_tt_latent as tl                                                                     # noqa: E402
import td_tt_splitting as ts                                                                  # noqa: E402
import td_tt_wormholes as tw                                                                  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False             # fp32 matmuls, as gradflow_analysis / td_analysis
torch.backends.cudnn.allow_tf32 = False

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
OUT = os.path.join(TD, "through_training", "intervene")
SEED = 20261029
T, R, NV, NO = 30, 10, 2, 5
NE, PER_ENV = gap.N_ENV, gap.PER_ENV
GATES = ("i", "f", "g", "o")                              # nn.LSTM gate order
KEYS = ("STM", "LTM", "classifier_readin.weight", "classifier_readin.bias", "LSTM.weight_ih_l0", "LSTM.weight_hh_l0",
        "LSTM.bias_ih_l0", "LSTM.bias_hh_l0", "classifier_readout.weight", "classifier_readout.bias")
DONOR_SEED, PARTNER_SEED, GATE_SEED = 20261201, 20261202, 20261203
SOURCES = ("start", "same_cell", "random")
BUDGET = 30000                                            # GPU rows per chunk x (1 + live branches)
CHECK_EPS = (0, 1100, 2100, 5250, 49995)                  # raw episodes: training 10 (initial), 880, 1,680, 4,200, 39,996
CUT_EPS = (1100, 2100, 49995)
_CACHE, _REF = {}, {}


# ═══════════════════════════════════════════════════════════════════
# Weights: checkpoints, parameter groups, hybrids
# ═══════════════════════════════════════════════════════════════════

def ckpt_path(run, episode):
    return os.path.join(TD, f"{run}_s{SEED}", f"ckpt_ep{int(episode):05d}_net.pth")


def checkpoints(run="trained"):
    """raw episodes of the run's stored checkpoints (114 for trained / echo)"""
    d = os.path.join(TD, f"{run}_s{SEED}")
    return sorted(int(f[7:12]) for f in os.listdir(d) if f.startswith("ckpt_ep") and f.endswith("_net.pth"))


def train_x(ep):
    """training episodes of raw checkpoint episodes (td_tt_gap.train_x; the initial network at 10)"""
    return gap.train_x(ep)


def load(run, episode):
    """the weights of a stored checkpoint: {key: fp32 CPU tensor} for the 10 tensors of KEYS (episode = the raw episode
    of the file name; training episodes = E - E // 5). Marginal read-outs only (20 rows: 2 variables x 10 values)"""
    w = torch.load(ckpt_path(run, episode), map_location="cpu", weights_only=False)["weights"]
    miss = [k for k in KEYS if k not in w]
    assert not miss, f"{run} {episode}: missing {miss}"
    out = {k: w[k].detach().float().clone().contiguous() for k in KEYS}
    assert out["classifier_readout.weight"].shape[0] == NV * R, "marginal read-out (20 rows) only"
    return out


def groups(H=1000):
    """the fine parameter groups, a partition of every weight: name -> [(key, row slice), ...]. read-in (W, b); W_ih and
    W_hh per gate block i / f / g / o (rows k H .. (k + 1) H); the LSTM biases per gate (b_ih and b_hh rows of the
    block); read-out (W, b); start state (STM = h0, LTM = c0)"""
    a = slice(None)
    g = {"readin": [("classifier_readin.weight", a), ("classifier_readin.bias", a)]}
    for kind, key in (("W_ih", "LSTM.weight_ih_l0"), ("W_hh", "LSTM.weight_hh_l0")):
        for k, G in enumerate(GATES):
            g[f"{kind}_{G}"] = [(key, slice(k * H, (k + 1) * H))]
    for k, G in enumerate(GATES):
        g[f"b_{G}"] = [("LSTM.bias_ih_l0", slice(k * H, (k + 1) * H)), ("LSTM.bias_hh_l0", slice(k * H, (k + 1) * H))]
    g["readout"] = [("classifier_readout.weight", a), ("classifier_readout.bias", a)]
    g["start"] = [("STM", a), ("LTM", a)]
    return g


GROUP_SETS = {"W_ih": tuple(f"W_ih_{G}" for G in GATES), "W_hh": tuple(f"W_hh_{G}" for G in GATES),
              "bias": tuple(f"b_{G}" for G in GATES)}
GROUP_SETS["lstm"] = GROUP_SETS["W_ih"] + GROUP_SETS["W_hh"] + GROUP_SETS["bias"]
GROUP_SETS.update({f"gate_{G}": (f"W_ih_{G}", f"W_hh_{G}", f"b_{G}") for G in GATES})
GROUP_SETS["all"] = tuple(groups())


def expand(names, H=1000):
    """fine group names of a name or list of names (fine names or GROUP_SETS keys), in order, without repeats"""
    fine, out = groups(H), []
    for nm in ([names] if isinstance(names, str) else list(names)):
        out += list(GROUP_SETS[nm]) if nm in GROUP_SETS else [nm]
    bad = [nm for nm in out if nm not in fine]
    assert not bad, f"unknown groups {bad}"
    return list(dict.fromkeys(out))


def hybrid(wa, wb, names, alpha=1.):
    """a weights dict assembled from two checkpoints: wa everywhere, except the groups `names` (fine or GROUP_SETS),
    which are (1 - alpha) wa + alpha wb (alpha 0: wa exactly; alpha 1: wb exactly)"""
    H = wa["LSTM.weight_hh_l0"].shape[1]
    fine = groups(H)
    out = {k: v.clone() for k, v in wa.items()}
    a = float(alpha)
    for nm in expand(names, H):
        for key, sl in fine[nm]:
            out[key][sl] = (1. - a) * wa[key][sl] + a * wb[key][sl]
    return out


def fingerprint(w):
    h = hashlib.sha1()
    for k in KEYS:
        h.update(w[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════
# Episode sets and donors
# ═══════════════════════════════════════════════════════════════════

def probes():
    """the probe set (analysis/probes.npz: 10,000 episodes = 50 test environments x 100 cells x 2 streams of 30 steps,
    environment-major, the two streams of a cell adjacent; environment = row // 200)"""
    if "probes" not in _CACHE:
        P = np.load(os.path.join(AN, "probes.npz"))
        _CACHE["probes"] = {k: P[k] for k in ("obs", "env", "Z", "cell", "r", "goal", "log_jb", "log_nb", "right_bayes",
                                              "right_naive", "right_window1", "right_window2", "right_window5")}
    return _CACHE["probes"]


def forward_set():
    """td_tt_wormholes' forward set (grid_side(): 5,028 grid episodes of the probes' 50 environments with a corner or
    one-variable end split, plus 12 unselected trajectories per corner cell; obs, env, Z, cell, r, Bayes's marginals)"""
    if "fwd" not in _CACHE:
        _CACHE["fwd"] = tw.grid_side()
    return _CACHE["fwd"]


def donors(E, kind, seed=None):
    """(n,) one donor episode per episode of the set E (fixed seed): 'same_cell' = a random other episode of the same
    environment and cell (on the probes: the other stream, row ^ 1), 'random' = a random episode of the same environment
    with another cell; -1 where none exists"""
    seed = {"same_cell": PARTNER_SEED, "random": DONOR_SEED}[kind] if seed is None else seed
    env, cell = np.asarray(E["env"]), np.asarray(E["cell"])
    rng = np.random.default_rng(seed)
    out = np.full(len(env), -1, np.int64)
    for e in np.unique(env):
        ix = np.flatnonzero(env == e)
        ce = cell[ix]
        for j, i in enumerate(ix):
            cand = ix[(ce == ce[j]) & (ix != i)] if kind == "same_cell" else ix[ce != ce[j]]
            if len(cand):
                out[i] = cand[rng.integers(len(cand))]
    return out


def _donors_cached(kind, seed=None):
    key = ("donors", kind, seed)
    if key not in _CACHE:
        _CACHE[key] = donors(probes(), kind, seed)
    return _CACHE[key]


# ═══════════════════════════════════════════════════════════════════
# Forward: the hand-written LSTM of td_analysis with composable hooks
# ═══════════════════════════════════════════════════════════════════

def _device(dev=None):
    # one visible GPU: GPU 0 by default; GPU 1 only when John has freed it (11-02: "you can utilize GPU 1, there is
    # nothing on it") - run with CUDA_VISIBLE_DEVICES=1 then; it appears as cuda:0 inside the process
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if dev is None:
        dev = "cuda:0" if (vis in ("0", "1") and torch.cuda.is_available()) else "cpu"
    dev = torch.device(dev)
    if dev.type == "cuda":
        assert vis in ("0", "1"), "one GPU only: run with CUDA_VISIBLE_DEVICES=0 (or =1 when John has freed GPU 1)"
        return torch.device("cuda:0")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    return dev


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _table(p, Z, dev):
    """the LSTM input drive incl. both biases for every (environment, observation pattern): (NE, 32, 4H); pattern k has
    observation bit i = (k >> i) & 1 (td_tt_wormholes._PAT); x = relu(read-in([o, Z])), drive = x W_ih^T + (b_ih + b_hh)
    (td_analysis: (inp + b) + rec)"""
    Zt = torch.as_tensor(np.asarray(Z, np.float32), device=dev).reshape(len(Z), -1)
    pat = torch.as_tensor(tw._PAT, device=dev)
    inp = torch.cat([pat[None].expand(len(Z), 32, NO), Zt[:, None].expand(len(Z), 32, Zt.shape[-1])], -1)
    x = torch.relu(Fn.linear(inp, p["classifier_readin.weight"], p["classifier_readin.bias"]))
    return x @ p["LSTM.weight_ih_l0"].T + (p["LSTM.bias_ih_l0"] + p["LSTM.bias_hh_l0"])


def orth_readout(V, w, tol=1e-6):
    """the basis a u clamp acts on: an orthonormal basis (H, k') of P span(V), the projection of span(V) (V: (H, k), any
    columns) off the read-out's 20 rows (P = I - Q Q^T, Q an orthonormal basis of the rows); k' = its numerical rank
    (0 for V = 0 or V inside the rows' span). P span(V) = span(V) only when V is orthogonal to the rows
    (random_subspace, pcs_off_readout). In general P span(V) contains the part of span(V) orthogonal to the rows
    (span(V) meets their complement in a space of dimension >= max(0, dim span(V) - 20), generically exactly that: {0}
    for dim span(V) <= 20), every other direction of P span(V) lies outside span(V), possibly well outside (raw
    principal components of u: VALIDATION), and any directions of span(V) inside the rows' span are dropped
    (then k' < dim span(V)); measure a clamp's variance with captured_variance(V, ref, w)"""
    V = np.asarray(V, np.float64)
    H = w["LSTM.weight_hh_l0"].shape[1]
    V = V.reshape(H, -1)
    Q, _ = np.linalg.qr(w["classifier_readout.weight"].double().cpu().numpy().T)            # (H, 20)
    Vp = V - Q @ (Q.T @ V)
    if Vp.shape[1] == 0 or not np.any(Vp):
        return np.zeros((H, 0))
    U, S, _ = np.linalg.svd(Vp, full_matrices=False)
    return U[:, S > tol * max(1., float(np.linalg.norm(V, axis=0).max()))]


def _orthonormal(V, tol=1e-6):
    V = np.asarray(V, np.float64)
    if V.size == 0 or not np.any(V):
        return np.zeros((V.shape[0], 0))
    U, S, _ = np.linalg.svd(V, full_matrices=False)
    return U[:, S > tol * max(1., float(S.max()))]


def _context(w, p, H, clamp, gate, E, ref, dev):
    """the per-step intervention constants on the device; the gate-resample donors; info"""
    clamp, gate = dict(clamp or {}), dict(gate or {})
    bad = (set(clamp) - {"u", "c"}) | (set(gate) - {"mean", "resample", "donor", "f_units", "f_value", "c_units"})
    assert not bad, f"unknown hook keys {bad}"
    mean = sorted(GATES.index(g) for g in gate.get("mean", ()))
    res = sorted(GATES.index(g) for g in gate.get("resample", ()))
    assert not set(mean) & set(res), "a gate block is either mean- or resample-ablated"
    ctx = dict(mean=mean, res=res, f_units=None, f_val=None, c_units=None, Vu=None, Vc=None, H=H)
    info = {}
    need = (bool(mean) or gate.get("f_units") is not None or gate.get("c_units") is not None or bool(clamp))
    if need:
        ref = reference(w, dev) if ref is None else ref

        def dv(a):
            return torch.as_tensor(np.asarray(a, np.float32), device=dev)
        ctx.update(mu_u=dv(ref["mu_u"]), mu_c=dv(ref["mu_c"]), mean_rec=dv(ref["mean_rec"]))
        if "u" in clamp:
            V = orth_readout(clamp["u"], w)
            info["clamp_u_rank"] = (int(np.linalg.matrix_rank(np.asarray(clamp["u"], np.float64).reshape(H, -1)))
                                    if np.any(clamp["u"]) else 0, int(V.shape[1]))
            ctx["Vu"] = dv(V) if V.shape[1] else None
        if "c" in clamp:
            V = _orthonormal(np.asarray(clamp["c"], np.float64).reshape(H, -1))
            info["clamp_c_rank"] = int(V.shape[1])
            ctx["Vc"] = dv(V) if V.shape[1] else None
        if gate.get("f_units") is not None:
            fv = gate.get("f_value", "median")
            if isinstance(fv, str):
                assert fv == "median", fv
                fv = ref["f_med"]
            fv = np.asarray(fv, np.float64)
            fv = np.full(T + 1, float(fv)) if fv.ndim == 0 else (np.concatenate([[np.nan], fv]) if len(fv) == T else fv)
            assert fv.shape == (T + 1,), fv.shape
            ctx["f_units"] = torch.as_tensor(np.asarray(gate["f_units"], np.int64), device=dev)
            ctx["f_val"] = dv(fv)
        if gate.get("c_units") is not None:
            ctx["c_units"] = torch.as_tensor(np.asarray(gate["c_units"], np.int64), device=dev)
    gdon = None
    if res:
        gdon = (np.asarray(gate["donor"], np.int64) if gate.get("donor") is not None else
                (donors(E, "random", GATE_SEED) if E is not probes() else _donors_cached("random", GATE_SEED)))
    return ctx, gdon, info


def _step(xin, h, c, s, ctx, Whh, hdon=None):
    """one LSTM step at absolute step s (1-based) with the hooks of ctx (td_analysis._cell otherwise): xin (b, 4H) the
    input drive incl. biases, (h, c) the previous state, hdon (b, H) the gate-resample donors' clean previous state"""
    H = ctx["H"]
    rec = h @ Whh.T
    for k in ctx["mean"]:
        rec[:, k * H:(k + 1) * H] = ctx["mean_rec"][s, k * H:(k + 1) * H]
    if ctx["res"]:
        rd = hdon @ Whh.T
        for k in ctx["res"]:
            rec[:, k * H:(k + 1) * H] = rd[:, k * H:(k + 1) * H]
    pre = xin + rec
    i, f, g, o = pre.chunk(4, -1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    if ctx["f_units"] is not None:
        f[:, ctx["f_units"]] = ctx["f_val"][s]
    c = f * c + i * g
    if ctx["c_units"] is not None:
        c[:, ctx["c_units"]] = ctx["mu_c"][s, ctx["c_units"]]
    if ctx["Vc"] is not None:
        c = c - ((c - ctx["mu_c"][s]) @ ctx["Vc"]) @ ctx["Vc"].T
    h = o * c.tanh()
    if ctx["Vu"] is not None:
        h = h - ((h - ctx["mu_u"][s]) @ ctx["Vu"]) @ ctx["Vu"].T
    return h, c, (i, f, g, o), rec


_CLEAN = dict(mean=[], res=[], f_units=None, f_val=None, c_units=None, Vu=None, Vc=None)


def _parse_cuts(cut, E):
    """[(name, w, donor (n,) or None)] from one cut (w, source) or a list / tuple of them; source 'start' |
    'same_cell' | 'random' | an explicit donor index array (n,). One cut is told from a list of cuts by its first
    element: a window w (a number) vs a cut (a tuple or list), tested with isinstance BEFORE np.ndim (np.ndim of a
    ragged (w, array) cut raises; the 11-02 fix)"""
    if cut is None or len(cut) == 0:
        return []
    single = not isinstance(cut[0], (tuple, list)) and np.ndim(cut[0]) == 0
    cuts = [cut] if single else list(cut)
    out, k = [], 0
    is_p = E is probes()
    for c_ in cuts:
        assert len(c_) == 2, f"a cut is (w, source), got {len(c_)} elements"
        w_, src = c_
        w_ = int(w_)
        assert 1 <= w_ < T, w_
        if isinstance(src, str):
            assert src in SOURCES, src
            d = None if src == "start" else (_donors_cached(src) if is_p else donors(E, src))
            name = f"w{w_}_{src}"
        else:
            d, name = np.asarray(src, np.int64), f"w{w_}_donor{k}"
            k += 1
        out.append((name, w_, d))
    names = [c[0] for c in out]
    assert len(set(names)) == len(names), names
    return out


@torch.no_grad()
def forward(w, rows=None, device=None, cut=None, clamp=None, gate=None, keep=(), record=None, episodes=None, ref=None,
            budget=BUDGET):
    """the marginal log beliefs of the weights w (with hooks) on the probes (or episodes), see the module docstring.
    Returns dict(rows, logq (n, T, 2, R) float32, right (n, T, 2) bool [from the float64 cumulative logits];
    with cut: logq_cut / right_cut {name: ...}; keep: 'u', 'c' (n, T, H), 'gates' (n, T, 4, H) i/f/g/o after the hooks,
    'logits' (n, T, 20) per-step read-out increments; info (seconds, clamp ranks, peak GPU memory))"""
    t0 = time.time()
    dev = _device(device)
    E = probes() if episodes is None else episodes
    obs, env_all = np.asarray(E["obs"]), np.asarray(E["env"], np.int64)
    n_all = len(obs)
    rows = np.arange(n_all) if rows is None else np.asarray(rows, np.int64).ravel()
    assert len(np.unique(rows)) == len(rows) and rows.min() >= 0 and rows.max() < n_all, "rows: distinct indices"
    srt = np.argsort(rows, kind="stable")
    Rs = rows[srt]
    p = {k: v.to(dev, torch.float32) for k, v in w.items()}
    H = p["LSTM.weight_hh_l0"].shape[1]
    Whh, Wro, bro = p["LSTM.weight_hh_l0"], p["classifier_readout.weight"], p["classifier_readout.bias"]
    h0, c0 = p["STM"].reshape(1, H), p["LTM"].reshape(1, H)
    table = _table(p, E["Z"], dev)
    code_np = (obs * (1 << np.arange(NO))).sum(-1).astype(np.int64)                      # (n, T) pattern index
    cuts = _parse_cuts(cut, E)
    ctx, gdon, info = _context(w, p, H, clamp, gate, E, ref, dev)
    ctx_clean = dict(_CLEAN, H=H)
    for name, _, d in cuts:
        if d is not None:
            assert (d[Rs] >= 0).all() and (env_all[d[Rs]] == env_all[Rs]).all(), f"{name}: donors missing or in another env"
    cut_d = [d[Rs] for _, _, d in cuts if d is not None]
    M = np.unique(np.concatenate([Rs] + cut_d)) if cut_d else Rs                          # main-run rows
    if gdon is not None:
        assert (gdon[M] >= 0).all() and (env_all[gdon[M]] == env_all[M]).all(), "gate donors missing or in another env"
    F = np.unique(gdon[M]) if gdon is not None else np.zeros(0, np.int64)                 # clean-run rows (donors)
    nbr = sum(c[1] for c in cuts)
    factor = 1 + nbr + (1 if len(F) else 0)
    keep = tuple(keep)
    bad = set(keep) - {"u", "c", "gates", "logits"}
    assert not bad, bad
    nR = len(Rs)
    out = dict(logq=np.zeros((nR, T, NV, R), np.float32), right=np.zeros((nR, T, NV), bool))
    if cuts:
        out["logq_cut"] = {c[0]: np.zeros((nR, T, NV, R), np.float32) for c in cuts}
        out["right_cut"] = {c[0]: np.zeros((nR, T, NV), bool) for c in cuts}
    for k in keep:
        out[k] = np.zeros((nR, T) + {"u": (H,), "c": (H,), "gates": (4, H), "logits": (NV * R,)}[k], np.float32)
    rt_all = np.asarray(E["r"], np.int64)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    # environment chunks (donors are always in the episode's own environment)
    envs, cnt = np.unique(env_all[M], return_counts=True)
    chunks, cur, size = [], [], 0
    for e, c_ in zip(envs, cnt):
        if cur and (size + c_) * factor > budget:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(e)
        size += c_
    if cur:
        chunks.append(cur)

    def ix(a):
        return torch.as_tensor(np.asarray(a, np.int64), device=dev)
    for ec in chunks:
        Mc = M[np.isin(env_all[M], ec)]
        Rc = Rs[np.isin(env_all[Rs], ec)]
        Fc = F[np.isin(env_all[F], ec)] if len(F) else F
        oR = np.searchsorted(Rs, Rc)                                                     # positions in the output
        same = len(Mc) == len(Rc)
        posR = None if same else ix(np.searchsorted(Mc, Rc))
        dpos = {nm: ix(np.searchsorted(Mc, d[Rc])) for nm, _, d in cuts if d is not None}
        gpos = ix(np.searchsorted(Fc, gdon[Mc])) if len(Fc) else None
        envM, codeM = ix(env_all[Mc]), ix(code_np[Mc])
        bm, br = len(Mc), len(Rc)
        h, c = h0.expand(bm, H).contiguous(), c0.expand(bm, H).contiguous()
        if len(Fc):
            envF, codeF = ix(env_all[Fc]), ix(code_np[Fc])
            hf, cf = h0.expand(len(Fc), H).contiguous(), c0.expand(len(Fc), H).contiguous()
        lg = torch.zeros(bm, T, NV * R, device=dev)
        cut_lg = {nm: torch.zeros(br, T, NV * R, device=dev) for nm, _, _ in cuts}
        wins = {nm: w_ for nm, w_, _ in cuts}
        live = []                                                       # [name, s0, h (br, H), c (br, H)]
        sel = (lambda a: a) if same else (lambda a: a[posR])            # noqa: E731
        for s in range(1, T + 1):
            xin = table[envM, codeM[:, s - 1]]
            hdon = hf[gpos] if len(Fc) else None
            h, c, gts, rec = _step(xin, h, c, s, ctx, Whh, hdon)
            lg[:, s - 1] = Fn.linear(h, Wro, bro)
            if record is not None or keep:
                st = dict(u=sel(h), c=sel(c), i=sel(gts[0]), f=sel(gts[1]), g=sel(gts[2]), o=sel(gts[3]), rec=sel(rec))
                if record is not None:
                    record(s, Rc, st)
                if "u" in keep:
                    out["u"][oR, s - 1] = st["u"].cpu().numpy()
                if "c" in keep:
                    out["c"][oR, s - 1] = st["c"].cpu().numpy()
                if "gates" in keep:
                    out["gates"][oR, s - 1] = torch.stack([st[k] for k in GATES], 1).cpu().numpy()
            if len(Fc):
                hf, cf, _, _ = _step(table[envF, codeF[:, s - 1]], hf, cf, s, ctx_clean, Whh)
            if live:
                k = len(live)
                hb, cb = torch.cat([b[2] for b in live]), torch.cat([b[3] for b in live])
                xb = sel(xin).repeat(k, 1)
                hdb = sel(hdon).repeat(k, 1) if hdon is not None else None
                hb, cb, _, _ = _step(xb, hb, cb, s, ctx, Whh, hdb)
                hb, cb = hb.view(k, br, H), cb.view(k, br, H)
                nxt = []
                for j, b in enumerate(live):
                    if b[1] + wins[b[0]] == s:
                        cut_lg[b[0]][:, s - 1] = Fn.linear(hb[j], Wro, bro)
                    else:
                        nxt.append([b[0], b[1], hb[j], cb[j]])
                live = nxt
            for nm, w_, d in cuts:
                if s + w_ <= T:
                    if d is None:
                        live.append([nm, s, h0.expand(br, H), c0.expand(br, H)])
                    else:
                        live.append([nm, s, h[dpos[nm]], c[dpos[nm]]])
        lgR = sel(lg)
        if "logits" in keep:
            out["logits"][oR] = lgR.cpu().numpy()
        rt = ix(rt_all[Rc])
        for nm, w_, _ in cuts:
            cut_lg[nm][:, :w_] = lgR[:, :w_]                            # t <= w: the window reaches the start
        for key, L in [("", lgR)] + [(nm, cut_lg[nm]) for nm, _, _ in cuts]:
            lq = torch.log_softmax(L.double().cumsum(1).reshape(br, T, NV, R), -1)
            right = (lq.argmax(-1) == rt[:, None, :]).cpu().numpy()
            if key == "":
                out["logq"][oR], out["right"][oR] = lq.float().cpu().numpy(), right
            else:
                out["logq_cut"][key][oR], out["right_cut"][key][oR] = lq.float().cpu().numpy(), right
    if not np.array_equal(srt, np.arange(len(srt))):                                  # back to the caller's order
        inv = np.empty_like(srt)
        inv[srt] = np.arange(len(srt))
        for k in ("logq", "right") + keep:
            out[k] = out[k][inv]
        for k in ("logq_cut", "right_cut"):
            if k in out:
                out[k] = {nm: v[inv] for nm, v in out[k].items()}
    _sync(dev)
    info.update(seconds=time.time() - t0, chunks=len(chunks), main_rows=len(M), donor_rows=len(F),
                peak_gpu_mb=(torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else 0.))
    out.update(rows=rows, info=info)
    return out


def reference(w, device=None, cov=False):
    """clean-run statistics of the weights w (no hooks) on all 10,000 probes, the constants of the hooks: mu_u, mu_c
    (T + 1, H) per-step probe means of u_t and c_t (index 0 = the start state); f_mean (T + 1, H) the per-unit probe mean
    of the forget gate, f_med (T + 1,) its median over units (index 0 NaN); mean_rec (T + 1, 4H) = W_hh mu_u[t - 1], the
    per-step probe mean of the recurrent drive (linear); with cov: cov_u (T + 1, H, H) the per-step covariance of u_t
    over the probes (float64; index 0 zero). Cached per weights (sha1)"""
    dev = _device(device)
    key = (fingerprint(w), dev.type, bool(cov))
    if key in _REF:
        return _REF[key]
    H = w["LSTM.weight_hh_l0"].shape[1]
    z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)          # noqa: E731
    S = dict(u=z(T + 1, H), c=z(T + 1, H), f=z(T + 1, H))
    if cov:
        S["uu"] = z(T + 1, H, H)

    def rec_fn(s, rows, st):
        for k, a in (("u", st["u"]), ("c", st["c"]), ("f", st["f"])):
            S[k][s] += a.double().sum(0)
        if cov:
            ud = st["u"].double()
            S["uu"][s] += ud.T @ ud
    forward(w, device=dev, record=rec_fn)
    n = len(probes()["env"])
    mu_u, mu_c, f_mean = (S[k].cpu().numpy() / n for k in ("u", "c", "f"))
    mu_u[0] = w["STM"].double().numpy().ravel()
    mu_c[0] = w["LTM"].double().numpy().ravel()
    f_mean[0] = np.nan
    out = dict(mu_u=mu_u, mu_c=mu_c, f_mean=f_mean, f_med=np.concatenate([[np.nan], np.median(f_mean[1:], 1)]),
               mean_rec=np.concatenate([np.full((1, 4 * H), np.nan),
                                        mu_u[:-1] @ w["LSTM.weight_hh_l0"].double().numpy().T]))
    if cov:
        uu = S["uu"].cpu().numpy() / n
        out["cov_u"] = uu - mu_u[:, :, None] * mu_u[:, None, :]
        out["cov_u"][0] = 0.
    if len(_REF) > 6:
        _REF.pop(next(iter(_REF)))
    _REF[key] = out
    return out


def _readout_q(w):
    """an orthonormal basis Q (H, 20) of the read-out's rows (as orth_readout)"""
    return np.linalg.qr(w["classifier_readout.weight"].double().cpu().numpy().T)[0]


def captured_variance(V, ref, w):
    """the clean variance of u that a clamp {'u': V} acts on (ref = reference(w, cov=True); V (H, k), any columns, as
    passed to forward). Per step t = 1..T: inside = the variance of u_t in the clamp's actual basis V' = orth_readout(V,
    w) (the projection of span(V) off the read-out rows; span(V) itself only when V is orthogonal to the rows), total =
    the total variance; share = inside / total; pooled_share = the sums over the steps. Clamp sizes and variance-matched
    controls use these. Diagnostics of the projection: k / k_clamp = dim span(V) / dim V'; span_inside /
    span_pooled_share = the same variance in span(V) itself (the number this function gave before the 11-02 fix, which
    overstated raw principal-component clamps); readout_share = |Q^T U|_F^2 / k, the share of span(V)'s squared norm
    in the rows' span (the mean cos^2 over any orthonormal basis U of span(V), not a share of the norm itself;
    fig1c_lesion_test's in_Q); max_angle_deg = the largest principal angle between span(V') and span(V)"""
    H = w["LSTM.weight_hh_l0"].shape[1]
    Vr = np.asarray(V, np.float64).reshape(H, -1)
    U, Vc = _orthonormal(Vr), orth_readout(Vr, w)
    C = ref["cov_u"][1:]
    total = np.trace(C, axis1=1, axis2=2)

    def var_in(B):                                                                   # (T,) tr(B^T C_t B)
        return np.einsum("thk,hk->t", C @ B, B) if B.shape[1] else np.zeros(len(C))
    inside, span_inside = var_in(Vc), var_in(U)
    cos = np.linalg.svd(U.T @ Vc, compute_uv=False) if U.shape[1] and Vc.shape[1] else np.ones(0)
    return dict(inside=inside, total=total, share=inside / total, pooled_share=float(inside.sum() / total.sum()),
                k=int(U.shape[1]), k_clamp=int(Vc.shape[1]), span_inside=span_inside,
                span_pooled_share=float(span_inside.sum() / total.sum()),
                readout_share=float(((_readout_q(w).T @ U) ** 2).sum() / U.shape[1]) if U.shape[1] else 0.,
                max_angle_deg=float(np.degrees(np.arccos(np.clip(cos.min(), 0., 1.)))) if len(cos) else 0.)


def pcs_off_readout(k, w, ref):
    """the top k principal components of u off the read-out rows (fig1c_lesion_test.subspaces' convention): the top k
    eigenvectors (H, k) of P C P, P = I - Q Q^T, C = Sigma_t ref['cov_u'][t] (the within-step covariances over the
    probes, pooled over t = 1..T): of all k-dim subspaces orthogonal to the rows, the one with the most pooled clean
    variance. A u clamp acts on exactly this span (orth_readout keeps it), so captured_variance's clamp and span
    numbers agree for it"""
    Q = _readout_q(w)
    H = Q.shape[0]
    assert 1 <= k <= H - Q.shape[1], k
    Pm = np.eye(H) - Q @ Q.T
    M = Pm @ ref["cov_u"][1:].sum(0) @ Pm
    _, U = np.linalg.eigh((M + M.T) / 2)
    return np.ascontiguousarray(U[:, ::-1][:, :k])


def random_subspace(k, w, seed):
    """a uniformly random k-dimensional orthonormal V (H, k) orthogonal to the read-out rows (Gaussian draw projected
    off the read-out's row space, fixed seed)"""
    H = w["LSTM.weight_hh_l0"].shape[1]
    V = orth_readout(np.random.default_rng(seed).standard_normal((H, k)), w)
    assert V.shape[1] == k
    return V


# ═══════════════════════════════════════════════════════════════════
# Scoring: behaviour rows with per-environment primitives
# ═══════════════════════════════════════════════════════════════════

LABELS = {   # name: (label, family, formula)
    "acc_goal_T": ("goal accuracy at T", "answers", "mean_i 1[argmax q_g,T = r_g]"),
    "acc_both_T": ("goal and context both right at T", "answers", "mean_i 1[both argmax q_v,T = r_v]"),
    "logq_goal_T": ("log q of the true goal value at T (nats)", "answers", "mean_i log q_g,T(r_g)"),
    "logq_both_T": ("log q0(r0) + log q1(r1) at T (nats)", "answers", "mean_i Sigma_v log q_v,T(r_v)"),
    "kl_goal_T": ("KL(Bayes || net) of the goal marginal at T (nats)", "answers", "mean_i KL(pi_g,T || q_g,T)"),
    "kl_goal_mean": ("KL(Bayes || net) of the goal marginal, mean over steps", "answers",
                     "mean_i mean_t KL(pi_g,t || q_g,t)"),
    "kl_both_T": ("KL(Bayes || net) at T, mean of the two variables", "answers", "mean_i mean_v KL(pi_v,T || q_v,T)"),
    "kl_both_mean": ("KL(Bayes || net), mean over steps and variables", "answers",
                     "mean_i mean_t,v KL(pi_v,t || q_v,t)"),
    "far_T": ("far errors at T", "answers", "mean_i 1[max_v |a_v - r_v| >= 2]"),
    "merged_T": ("distant pairs merged at T", "answers", "share of far-apart probe pairs with the same answer cell"),
    "excess": ("jump excess angle over independent", "coupling", "phi_bar - phi_bar_indep (deg)"),
    "G": ("long-path coupling G", "coupling", "mean |rho| longest fifth - shortest fifth"),
    "band4": ("angle of jumps >= 4 cells", "coupling", "Sigma phi / count over jumps |dmu| >= 4 (deg)"),
    "held_joint": ("corner split held (joint), probes", "holding", "share of corner-split stream-steps, both e_v > .15"),
    "held_two": ("held per variable, corner-split ends, probes", "holding", "share of corner var-steps, e_v > .15"),
    "held_one": ("held per variable, one-variable end splits, probes", "holding", "share of one-var var-steps, e_v > .15"),
    "emass_slope": ("end-mass slope within steps (maps)", "holding", "Sigma_t cov_t(e_N, e_B) / Sigma_t var_t(e_B)"),
    "dir_slope": ("direction slope within steps (maps)", "coupling", "Sigma_t cov_t(phi_N, phi_B) / Sigma_t var_t(phi_B)"),
    "split_grad": ("split gradient (most minus least entangled tenth)", "splitting", "dD(top tenth) - dD(bottom tenth)"),
    "split_grad_matched": ("split gradient, distance-matched", "splitting", "the same within thirds of g"),
    "toward6": ("six-step move toward own cells", "splitting", "dD_6 entangled third - calm third"),
    "apart6": ("over-splitting, six steps apart", "splitting", "dd_6 entangled third - calm third"),
    "fold_idx": ("fold index (corners - edges)", "answers", "pair distance at T: (9,9) - mean of (9,0), (0,9)"),
    "corners": ("opposite corners", "answers", "pair distance at T at separation (9, 9)"),
    "corner_truth_fwd": ("corner answer = truth at T", "answers", "forward set, unselected corner episodes"),
    "held_joint_fwd": ("corner split held (joint)", "holding", "forward set, corner-split stream-steps"),
    "held_two_fwd": ("held per variable, corner-split ends", "holding", "forward set, corner var-steps"),
    "held_one_fwd": ("held per variable, one-variable end splits", "holding", "forward set, one-var var-steps"),
    "dist_out_fwd": ("distance out of a corner split", "holding", "forward set, cells along Bayes's move"),
    "angle_out_fwd": ("coupling angle out of a corner split", "coupling", "forward set, deg"),
    "same_diag_fwd": ("same-diagonal share of moves out of corner splits", "coupling", "forward set"),
    "commit_fwd": ("committed to a corner", "holding", "forward set, corner-split stream-steps"),
}
CHEAP = ("acc_goal_T", "acc_both_T", "logq_goal_T", "logq_both_T", "kl_goal_T", "kl_goal_mean", "kl_both_T",
         "kl_both_mean", "far_T", "merged_T", "excess", "G", "band4", "held_joint", "held_two", "held_one",
         "emass_slope", "dir_slope")
FULL = CHEAP + ("split_grad", "split_grad_matched", "toward6", "apart6", "fold_idx", "corners")
FWD = ("corner_truth_fwd", "held_joint_fwd", "held_two_fwd", "held_one_fwd", "dist_out_fwd", "angle_out_fwd",
       "same_diag_fwd", "commit_fwd")


def _side():
    """probe-side constants of the scores (cached): probes, td_tt_wormholes.probe_side and its counts, the far-apart
    probe pairs (td_tt_latent.far_pairs), Bayes's normalised log marginals, the jumps' matched-null partners"""
    if "side" not in _CACHE:
        P = probes()
        n = len(P["env"])
        assert np.array_equal(P["env"], np.arange(n) // PER_ENV), "probes are environment-major"
        D = tw.probe_side()
        I, J, Ep = tl.far_pairs(P["r"], P["env"])
        lB = P["log_jb"].astype(np.float64)
        lB = lB - logsumexp(lB, -1, keepdims=True)
        _CACHE["side"] = dict(D=D, Cn=tw._counts_all(D), I=I, J=J, Ep=Ep, lB=lB, pB=np.exp(lB),
                              npair=np.bincount(Ep, minlength=NE).astype(np.float64), null=tj.null_partner(n),
                              cnt=np.bincount(P["env"], minlength=NE).astype(np.float64), Pi=ts._probe_inputs())
    return _CACHE["side"]


def _jump_prims(q, null):
    """td_tt_jumps' primitives of the timed rows, copied and trimmed for speed (the checks verify bit-identity with the
    originals): from stats() the per-episode path length S and Pearson rho (float32), from extra_stats() the per
    (environment) sums over steps of |dmu|^2 and |dmu|^2 phi for the jumps and for the matched independence null
    (shift 1), and the >= 4 cell band's counts and angle sums. q: normalised marginals (n, T, 2, R) float64"""
    mu = q @ tj.RV
    d = np.diff(mu, axis=1)
    a = np.abs(d)
    path = np.hypot(a[..., 0], a[..., 1]).sum(1).astype(np.float32)
    rho = tj._pearson(d[..., 0], d[..., 1]).astype(np.float32)
    a = np.abs(np.diff(q @ tj.RV, axis=1))
    w = (a ** 2).sum(-1)
    phi = tj._angle(a[..., 0], a[..., 1])

    def es(v):
        return v.reshape(NE, PER_ENV, T - 1).sum(1)
    a1 = a[null, :, 1]
    wn = a[..., 0] ** 2 + a1 ** 2
    A = np.sqrt(w)
    m = (A >= tj.BANDS[0][0]) & (A < tj.BANDS[0][1])
    return dict(path=path, rho=rho, e=es(w).sum(-1), ephi=es(w * phi).sum(-1), n=es(wn).sum(-1),
                nphi=es(wn * tj._angle(a[..., 0], a1)).sum(-1), band_cnt=es(m.astype(np.float64)).sum(1),
                band_sphi=es(m * phi).sum(1))


def _row(name, kind, **prim):
    lab, fam, formula = LABELS[name]
    r = dict(kind=kind, label=lab, family=fam, formula=formula, **prim)
    r["value"] = _value(r)
    return r


def _value(r):
    k = r["kind"]
    with np.errstate(invalid="ignore", divide="ignore"):
        if k == "ratio":
            d = r["den"].sum()
            return float(r["num"].sum() / d) if d >= max(r.get("min_den", 0), 1e-300) else np.nan
        if k == "ratio_diff":
            return float(r["num"].sum() / r["den"].sum() - r["num2"].sum() / r["den2"].sum())
        if k == "slope":
            return float(tw.step_slope(r["dot"][None], r["sN"][None], r["n"], r["sB"], r["sB2"])[0])
        if k == "G":
            f = tj.rho_fifths(r["path"], r["rho"])
            return float(f[-1] - f[0])
        if k == "env_mean":
            v = r["v"]
            ok = np.isfinite(v)
            return float(np.where(ok, v, 0.).sum() / ok.sum())
    raise ValueError(k)


def resample(rows, C=None):
    """{name: (N_BOOT,) the row in every shared environment resample} (td_tt_gap.boot_counts(): counts of the 50
    environments; ratios of count-weighted sums, the within-step slopes refitted, G's fifths re-cut on the weights, env
    means count-weighted; a ratio with min_den is NaN where the resample's count is below it)"""
    C = gap.boot_counts() if C is None else np.asarray(C, np.float64)
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, r in rows.items():
            kind = r["kind"]
            if kind == "ratio":
                den = C @ r["den"]
                v = (C @ r["num"]) / den
                out[k] = np.where(den >= r.get("min_den", 0), v, np.nan) if r.get("min_den") else v
            elif kind == "ratio_diff":
                out[k] = (C @ r["num"]) / (C @ r["den"]) - (C @ r["num2"]) / (C @ r["den2"])
            elif kind == "slope":
                out[k] = tw.step_slope(r["dot"][None], r["sN"][None], r["n"], r["sB"], r["sB2"], W=C)[:, 0]
            elif kind == "G":
                out[k] = tj._G_boot(r["path"], r["rho"], C)
            elif kind == "env_mean":
                out[k] = gap.env_mean(r["v"], C)[1]
            else:
                raise ValueError(kind)
    return out


def score(logq, which="cheap", right=None):
    """behaviour rows of beliefs on the 10,000 probes: {name: dict(value, kind, label, family, formula, per-environment
    primitives)}; which 'cheap' (CHEAP) or 'full' (FULL). logq (n, T, 2, R) log marginals in any normalisation (fp16 ck
    logq or the engine's float32); right (n, T, 2) or (n, 2) correctness at T for the accuracies (default: argmax of logq
    == truth; the far errors / merged pairs always use argmax of logq, as td_tt_latent)"""
    assert which in ("cheap", "full"), which
    S = _side()
    P = probes()
    raw = np.asarray(logq)
    n = len(P["env"])
    assert raw.shape == (n, T, NV, R), raw.shape
    lq = raw.astype(np.float64)
    lq = lq - logsumexp(lq, -1, keepdims=True)
    q = np.exp(lq)
    ar, env, r, goal = np.arange(n), P["env"], P["r"], P["goal"]
    ans = raw.argmax(-1)
    if right is None:
        rT = ans[:, -1] == r
    else:
        right = np.asarray(right, bool)
        rT = right[:, -1] if right.ndim == 3 else right

    def es(v):
        return np.bincount(env, np.asarray(v, np.float64), NE)
    cnt = S["cnt"]
    rows = {"acc_goal_T": _row("acc_goal_T", "ratio", num=es(rT[ar, goal]), den=cnt),
            "acc_both_T": _row("acc_both_T", "ratio", num=es(rT.all(-1)), den=cnt),
            "logq_goal_T": _row("logq_goal_T", "ratio", num=es(lq[ar, -1, goal, r[ar, goal]]), den=cnt),
            "logq_both_T": _row("logq_both_T", "ratio", num=es(lq[ar, -1, 0, r[:, 0]] + lq[ar, -1, 1, r[:, 1]]),
                                den=cnt)}
    kl = (S["pB"] * (S["lB"] - lq)).sum(-1)                                          # (n, T, 2) KL(Bayes || net)
    rows.update(kl_goal_T=_row("kl_goal_T", "ratio", num=es(kl[ar, -1, goal]), den=cnt),
                kl_goal_mean=_row("kl_goal_mean", "ratio", num=es(kl[ar, :, goal].mean(1)), den=cnt),
                kl_both_T=_row("kl_both_T", "ratio", num=es(kl[:, -1].mean(-1)), den=cnt),
                kl_both_mean=_row("kl_both_mean", "ratio", num=es(kl.mean((1, 2))), den=cnt))
    aT = ans[:, -1]
    acell = aT[:, 0] * R + aT[:, 1]
    rows["far_T"] = _row("far_T", "ratio", num=es(np.abs(aT - r).max(-1) >= 2), den=cnt)
    rows["merged_T"] = _row("merged_T", "ratio", num=np.bincount(S["Ep"], acell[S["I"]] == acell[S["J"]], NE),
                            den=S["npair"])
    jp = _jump_prims(q, S["null"])
    rows["excess"] = _row("excess", "ratio_diff", num=jp["ephi"], den=jp["e"], num2=jp["nphi"], den2=jp["n"])
    rows["G"] = _row("G", "G", path=jp["path"], rho=jp["rho"])
    rows["band4"] = _row("band4", "ratio", num=jp["band_sphi"], den=jp["band_cnt"], min_den=float(tj.MIN_N))
    st = tw._stats_all(q, S["D"])
    Cn = S["Cn"]
    rows.update(held_joint=_row("held_joint", "ratio", num=st["joint_held"], den=Cn["n_two"]),
                held_two=_row("held_two", "ratio", num=st["held_two"], den=Cn["n_two_v"]),
                held_one=_row("held_one", "ratio", num=st["held_one"], den=Cn["n_one_v"]),
                emass_slope=_row("emass_slope", "slope", dot=st["e_dot_st"], sN=st["e_st"], n=Cn["n_e_st"],
                                 sB=Cn["n_e_B_st"], sB2=Cn["n_e_B2_st"]),
                dir_slope=_row("dir_slope", "slope", dot=st["phi_dot_st"], sN=st["phi_st"], n=Cn["n_jump_big_st"],
                               sB=Cn["n_jump_phiB_st"], sB2=Cn["n_jump_phiB2_st"]))
    if which == "full":
        o = ts._one(ts.positions(raw), S["Pi"], np.random.default_rng(1))         # td_tt_splitting.build_probes' rng
        corr = np.nanmean(o["corr"].astype(np.float64), -2)                          # (NE, ND) mean over t0
        cg = np.nanmean(o["corr_g"].astype(np.float64), -2)
        L6 = {k: np.nanmean(o[k][..., ts.K].astype(np.float64), -2) for k in ("corr_lag", "split_lag")}
        Ff = o["fold"][:, -1].astype(np.float64)                                      # (NE, R, R) at T
        corners = Ff[:, R - 1, R - 1]
        for nm, v in (("split_grad", corr[:, -1] - corr[:, 0]), ("split_grad_matched", cg[:, -1] - cg[:, 0]),
                      ("toward6", L6["corr_lag"][:, 0] - L6["corr_lag"][:, 1]),
                      ("apart6", L6["split_lag"][:, 0] - L6["split_lag"][:, 1]),
                      ("fold_idx", corners - (Ff[:, R - 1, 0] + Ff[:, 0, R - 1]) / 2), ("corners", corners)):
            rows[nm] = _row(nm, "env_mean", v=v)
    return rows


def _fwd_side():
    if "fwd_side" not in _CACHE:
        G = forward_set()
        D = tw.fwd_side(G)
        Cn = tw._counts_all(D)
        Cn["n_out"] = tw._env_sum(D["env"], D["X"])
        _CACHE["fwd_side"] = (D, Cn)
    return _CACHE["fwd_side"]


def score_fwd(logq):
    """the forward-set rows of td_tt_wormholes (FWD; its _fwd_stats_all on full beliefs) for beliefs logq (5,028, T, 2,
    R) on forward_set(), as ratios of per-environment sums"""
    D, Cn = _fwd_side()
    raw = np.asarray(logq)
    assert raw.shape == D["bB"].shape, raw.shape
    lq = raw.astype(np.float64)
    lq = lq - logsumexp(lq, -1, keepdims=True)
    b = np.exp(lq)
    st = tw._fwd_stats_all(b, .5 * np.abs(b - D["bB"]).sum(-1), lq[:, -1], b @ tw.V, D)
    return {"corner_truth_fwd": _row("corner_truth_fwd", "ratio", num=st["corner_cat"][:, 0], den=Cn["n_corner"]),
            "held_joint_fwd": _row("held_joint_fwd", "ratio", num=st["joint_held"], den=Cn["n_two"]),
            "held_two_fwd": _row("held_two_fwd", "ratio", num=st["held_two"], den=Cn["n_two_v"]),
            "held_one_fwd": _row("held_one_fwd", "ratio", num=st["held_one"], den=Cn["n_one_v"]),
            "dist_out_fwd": _row("dist_out_fwd", "ratio", num=st["proj_out"], den=Cn["n_out"]),
            "angle_out_fwd": _row("angle_out_fwd", "ratio", num=st["phi_out"], den=Cn["n_out"]),
            "same_diag_fwd": _row("same_diag_fwd", "ratio", num=st["same_out"], den=Cn["n_out"]),
            "commit_fwd": _row("commit_fwd", "ratio", num=st["commit"], den=Cn["n_two"])}


def refs(which="full"):
    """Bayes's and naive's rows on the same probes (and forward set): {'bayes': rows, 'naive': rows} (cached)"""
    key = ("refs", which)
    if key not in _CACHE:
        P = probes()
        out = {nm: score(P[f"log_{k}"], which, right=P[f"right_{nm}"]) for nm, k in (("bayes", "jb"), ("naive", "nb"))}
        D, _ = _fwd_side()
        for nm, b in (("bayes", D["bB"]), ("naive", D["bN"])):
            out[nm].update(score_fwd(np.log(np.clip(b, 1e-300, 1))))
        _CACHE[key] = out
    return _CACHE[key]


def gap_rows(rows, ref=None, C=None):
    """td_tt_gap conventions for rows: {name: dict(value, boot, naive, bayes, naive_boot, bayes_boot, frac, frac_boot,
    lo, hi, frac_lo, frac_hi)} with gap fraction f = (y - naive) / (Bayes - naive) on the shared resamples (paired:
    naive and Bayes resampled with the same counts)"""
    default = C is None and ref is None
    C = gap.boot_counts() if C is None else C
    ref = refs() if ref is None else ref
    if default:                                                     # the references' resamples, once per process
        if "ref_boot" not in _CACHE:
            _CACHE["ref_boot"] = (resample(ref["bayes"], C), resample(ref["naive"], C))
        bb, nb = _CACHE["ref_boot"]
    else:
        bb, nb = resample(ref["bayes"], C), resample(ref["naive"], C)
    yb = resample(rows, C)
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, r in rows.items():
            if k not in ref["bayes"]:
                continue
            nv, bv = ref["naive"][k]["value"], ref["bayes"][k]["value"]
            f, fb = (r["value"] - nv) / (bv - nv), (yb[k] - nb[k]) / (bb[k] - nb[k])
            lo, hi = np.nanpercentile(yb[k], [5, 95])
            flo, fhi = np.nanpercentile(fb, [5, 95])
            out[k] = dict(value=r["value"], boot=yb[k], naive=nv, bayes=bv, naive_boot=nb[k], bayes_boot=bb[k],
                          frac=f, frac_boot=fb, lo=lo, hi=hi, frac_lo=flo, frac_hi=fhi)
    return out


def curves(rows_list, names=None):
    """per-checkpoint stacks for td_tt_gap.export: {name: (value (K,), boot (N_BOOT, K))} from a list of score rows"""
    names = list(rows_list[0]) if names is None else names
    bs = [resample({k: r[k] for k in names}) for r in rows_list]
    return {k: (np.array([r[k]["value"] for r in rows_list]), np.stack([b[k] for b in bs], 1)) for k in names}


# ═══════════════════════════════════════════════════════════════════
# Validation and timing (checks -> OUT/checks.json; numbers prints it)
# ═══════════════════════════════════════════════════════════════════

EXPORT_MAP = (   # (topic, exported row name, this module's row); 'fwd' rows are scored on the forward set
    ("jumps", "goal accuracy at T", "acc_goal_T"),
    ("latent", "goal and context both right at T", "acc_both_T"),
    ("latent", "far errors at T", "far_T"),
    ("latent", "distant pairs merged at T", "merged_T"),
    ("jumps", "jump excess angle over independent", "excess"),
    ("jumps", "long-path coupling G", "G"),
    ("jumps", "angle of jumps >= 4 cells", "band4"),
    ("wormholes", "end-mass slope within steps (maps)", "emass_slope"),
    ("wormholes", "direction slope within steps (maps)", "dir_slope"),
    ("splitting", "split gradient (most minus least entangled tenth)", "split_grad"),
    ("splitting", "split gradient, distance-matched", "split_grad_matched"),
    ("splitting", "six-step move toward own cells", "toward6"),
    ("splitting", "over-splitting, six steps apart", "apart6"),
    ("splitting", "fold index (corners - edges)", "fold_idx"),
    ("splitting", "opposite corners", "corners"),
    ("wormholes", "corner answer = truth at T", "corner_truth_fwd"),
    ("wormholes", "corner split held (joint)", "held_joint_fwd"),
    ("wormholes", "held per variable, corner-split ends", "held_two_fwd"),
    ("wormholes", "held per variable, one-variable end splits", "held_one_fwd"),
    ("wormholes", "distance out of a corner split", "dist_out_fwd"),
    ("wormholes", "coupling angle out of a corner split", "angle_out_fwd"),
    ("wormholes", "same-diagonal share of moves out of corner splits", "same_diag_fwd"),
    ("wormholes", "committed to a corner", "commit_fwd"))
PROBE_HELD = (("held_joint", "joint_held", "n_two"), ("held_two", "held_two", "n_two_v"), ("held_one", "held_one", "n_one_v"))


def _ck(run, ep):
    return np.load(os.path.join(AN, run, f"ck_{ep:05d}.npz"))


def _exports():
    out = {}
    for t in sorted({m[0] for m in EXPORT_MAP}):
        z = gap.load(t)
        for i in range(len(z["names"])):
            out[(t, str(z["names"][i]), str(z["run"][i]))] = dict(x=z["x"], value=z["value"][i], boot=z["boot"][i],
                                                                  naive=float(z["naive"][i]), bayes=float(z["bayes"][i]),
                                                                  naive_boot=z["naive_boot"][i],
                                                                  bayes_boot=z["bayes_boot"][i])
    return out


def _nanmax_absdiff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    both = np.isnan(a) & np.isnan(b)
    d = np.where(both, 0., np.abs(a - b))
    return float(np.nan_to_num(d, nan=np.inf).max())


def _logq_vs_stored(lq, right, z):
    """the engine's logq (float32) and right against a ck file's fp16 logq and float64-derived right"""
    s = z["logq"]
    d = np.abs(lq.astype(np.float64) - s.astype(np.float64))
    ulp = np.abs(np.spacing(s)).astype(np.float64)                                   # fp16 spacing at the stored value
    a_mine, a_st = lq[:, -1].argmax(-1), s[:, -1].argmax(-1)
    ties = (s[:, -1] == s[:, -1].max(-1, keepdims=True)).sum(-1) > 1
    return dict(max_abs=float(d.max()), max_ulp=float((d / ulp).max()),
                fp16_equal=float((lq.astype(np.float16) == s).mean()), right_diff_all=int((right != z["right"]).sum()),
                right_diff_T=int((right[:, -1] != z["right"][:, -1]).sum()),
                argmax_T_diff=int((a_mine != a_st).sum()), argmax_T_diff_not_tie=int(((a_mine != a_st) & ~ties).sum()),
                fp16_ties_T=int(ties.sum()))


def _fix_checks(dev, wa, wb, ref_a):
    """the 11-02 fixes (verifier issues; wa / wb = raw 2,100 / 49,995, ref_a = reference(wa, cov=True)): _parse_cuts on
    single cuts and on lists that start with a donor array (and the pre-fix test on the same specs); two such lists
    through forward against the same cuts named by source; captured_variance in the clamp's actual basis vs span(V) for
    the top 10 / 30 principal components of u (td_analysis basis_V100: the stored PCA basis of u, pooled over the basis
    set's steps) and for pcs_off_readout; a clamp of the raw top 10 PCs: residual in V' and deviation left in span(V)"""
    P = probes()
    n = len(P["env"])
    ds_, dr_ = _donors_cached("same_cell"), _donors_cached("random")
    out = {}

    def old_single(c):                                 # the pre-fix test (np.ndim before isinstance)
        return len(c) == 2 and np.ndim(c[0]) == 0 and not isinstance(c[0], tuple)
    cases = (("(5, 'start')", (5, "start"), [("w5_start", 5, None)]),
             ("[(5, 'start')]", [(5, "start")], [("w5_start", 5, None)]),
             ("(5, d)", (5, ds_), [("w5_donor0", 5, ds_)]),
             ("[(5, d)]", [(5, ds_)], [("w5_donor0", 5, ds_)]),
             ("[(5, d1), (2, d2)]", [(5, ds_), (2, dr_)], [("w5_donor0", 5, ds_), ("w2_donor1", 2, dr_)]),
             ("[(5, d), (5, 'start')]", [(5, dr_), (5, "start")], [("w5_donor0", 5, dr_), ("w5_start", 5, None)]),
             ("((5, d1), (2, d2))", ((5, ds_), (2, dr_)), [("w5_donor0", 5, ds_), ("w2_donor1", 2, dr_)]),
             ("[[5, d], [2, 'random']]", [[5, ds_], [2, "random"]], [("w5_donor0", 5, ds_), ("w2_random", 2, dr_)]),
             ("[(5, 'start'), (5, d)]", [(5, "start"), (5, ds_)], [("w5_start", 5, None), ("w5_donor0", 5, ds_)]),
             ("[(5, d1), (2, d1), (3, 'start')]", [(5, ds_), (2, ds_), (3, "start")],
              [("w5_donor0", 5, ds_), ("w2_donor1", 2, ds_), ("w3_start", 3, None)]),
             ("(np.int64(5), 'same_cell')", (np.int64(5), "same_cell"), [("w5_same_cell", 5, ds_)]),
             ("[] (no cut)", [], []))
    pc = {}
    for lab, c, exp in cases:
        try:
            old, old_err = bool(old_single(c)), "none"
        except ValueError:
            old, old_err = None, "ValueError"
        got = _parse_cuts(c, P)
        ok = len(got) == len(exp) and all(
            g[0] == e[0] and g[1] == e[1] and ((g[2] is None) if e[2] is None else (g[2] is not None and
                                                                                    np.array_equal(g[2], e[2])))
            for g, e in zip(got, exp))
        new = bool(len(c)) and not isinstance(c[0], (tuple, list)) and np.ndim(c[0]) == 0
        pc[lab] = dict(ok=bool(ok), old_error=old_err, same_decision=(None if old is None else bool(old == new)),
                       names=[g[0] for g in got])
    out["parse"] = pc
    print("\nfix 1, _parse_cuts: " + "; ".join(f"{k} {'ok' if v['ok'] else 'WRONG'} (before: {v['old_error']})"
                                               for k, v in pc.items()), flush=True)
    pf = {}
    for lab, ca, cn in (("[(5, same-cell donors), (2, random donors)]", [(5, ds_), (2, dr_)],
                         [(5, "same_cell"), (2, "random")]),
                        ("[(5, random donors), (5, 'start')]", [(5, dr_), (5, "start")],
                         [(5, "random"), (5, "start")])):
        fa_, fn_ = forward(wa, device=dev, cut=ca), forward(wa, device=dev, cut=cn)
        na, nn = [c[0] for c in _parse_cuts(ca, P)], [c[0] for c in _parse_cuts(cn, P)]
        d, rd = float(np.abs(fa_["logq"] - fn_["logq"]).max()), int((fa_["right"] != fn_["right"]).sum())
        for a_, b_ in zip(na, nn):
            d = max(d, float(np.abs(fa_["logq_cut"][a_] - fn_["logq_cut"][b_]).max()))
            rd += int((fa_["right_cut"][a_] != fn_["right_cut"][b_]).sum())
        pf[lab] = dict(max_abs=d, right_diff=rd, names=na, named=nn,
                       cut_vs_intact=max(float(np.abs(fn_["logq_cut"][b_] - fn_["logq"]).max()) for b_ in nn))
        print(f"  forward {lab} vs {cn}: max |d logq| {d:.2g} (main and cuts), right differs {rd}; the cuts move log q "
              f"by up to {pf[lab]['cut_vs_intact']:.3g}", flush=True)
    out["parse_forward"] = pf
    refs_ = {2100: (wa, ref_a), 49995: (wb, reference(wb, dev, cov=True))}
    cb = {}
    for ep_, (w_, r_) in refs_.items():
        Vb = _ck("trained", ep_)["basis_V100"].astype(np.float64)
        for k in (10, 30):
            for kind, V_ in (("pcs", Vb[:, :k]), ("pcs_off_readout", pcs_off_readout(k, w_, r_))):
                cv = captured_variance(V_, r_, w_)
                cb[f"{ep_}_{kind}{k}"] = {kk: cv[kk] for kk in ("k", "k_clamp", "readout_share", "max_angle_deg",
                                                                "span_pooled_share", "pooled_share")}
    out["clamp_basis"] = cb
    print("fix 2, clamp basis (pooled clean variance share in span(V) / in the clamped basis V'; share of span(V)'s "
          "squared norm in the read-out rows' span; largest principal angle V' vs span(V)): " + "; ".join(
              f"{k} {v['span_pooled_share']:.4f} / {v['pooled_share']:.4f}, {v['readout_share']:.3f}, "
              f"{v['max_angle_deg']:.1f} deg" for k, v in cb.items()), flush=True)
    V10 = _ck("trained", 2100)["basis_V100"][:, :10].astype(np.float64)
    mu = torch.as_tensor(ref_a["mu_u"], device=dev).float()
    Vct, Ut = (torch.as_tensor(a, device=dev).float() for a in (orth_readout(V10, wa), _orthonormal(V10)))
    acc = dict(res=0., span=0.)

    def rec(s, rows, st):
        dd = st["u"] - mu[s]
        acc["res"] = max(acc["res"], float((dd @ Vct).abs().max()))
        acc["span"] += float(((dd @ Ut).double() ** 2).sum())
    fo = forward(wa, device=dev, clamp={"u": V10}, record=rec)
    cv = captured_variance(V10, ref_a, wa)
    tot = float(cv["total"].sum())
    out["clamp_pcs10"] = dict(ep=2100, rank=list(fo["info"]["clamp_u_rank"]), residual_in_clamp_basis=acc["res"],
                              span_left_share=acc["span"] / n / tot, span_clean_share=cv["span_pooled_share"],
                              clamp_clean_share=cv["pooled_share"])
    c_ = out["clamp_pcs10"]
    print(f"  clamp of the raw top 10 PCs at raw 2,100 (ranks {c_['rank']}): |(u - mu) V'| <= "
          f"{c_['residual_in_clamp_basis']:.2g}; mean square deviation from the clean mean left inside span(V) "
          f"{c_['span_left_share']:.4f} of the clean total (clean {c_['span_clean_share']:.4f})", flush=True)
    return out


def checks(dev=None, keep_timing=False):
    """the validation of the engine (prints; -> OUT/checks.json); keep_timing: reuse the saved run's timing block"""
    dev = _device(dev)
    t_all = time.time()
    res = {"device": str(dev), "check_eps": list(CHECK_EPS), "cut_eps": list(CUT_EPS)}
    P = probes()
    n = len(P["env"])
    # 0. groups partition every weight exactly once
    w0 = load("trained", 0)
    H = w0["LSTM.weight_hh_l0"].shape[1]
    cover = {k: torch.zeros_like(v) for k, v in w0.items()}
    for nm, parts in groups(H).items():
        for key, sl in parts:
            cover[key][sl] += 1
    res["groups"] = dict(n_fine=len(groups(H)), n_params=int(sum(v.numel() for v in w0.values())),
                         covered_once=bool(all(bool((v == 1).all()) for v in cover.values())))
    print(f"groups: {res['groups']['n_fine']} fine groups over {res['groups']['n_params']:,} weights, every weight in "
          f"exactly one: {res['groups']['covered_once']}", flush=True)
    ds, dr, dg = _donors_cached("same_cell"), _donors_cached("random"), _donors_cached("random", GATE_SEED)
    res["donors"] = dict(same_cell_is_other_stream=bool(np.array_equal(ds, np.arange(n) ^ 1)),
                         random_same_env=bool(all((P["env"][d] == P["env"]).all() for d in (dr, dg))),
                         random_other_cell=bool(all((P["cell"][d] != P["cell"]).all() for d in (dr, dg))),
                         random_distinct_donors=int(len(np.unique(dr))), gate_vs_cut_donor_same=int((dr == dg).sum()))
    print(f"donors: {res['donors']}", flush=True)
    S_, (Df, Cf) = _side(), _fwd_side()
    res["counts"] = dict(probe_two=float(S_["Cn"]["n_two"].sum()), probe_two_v=float(S_["Cn"]["n_two_v"].sum()),
                         probe_one_v=float(S_["Cn"]["n_one_v"].sum()), probe_var_steps=float(S_["Cn"]["n_e_st"].sum()),
                         probe_big_moves=float(S_["Cn"]["n_jump_big_st"].sum()), far_pairs_per_env=float(S_["npair"][0]),
                         fwd_two=float(Cf["n_two"].sum()), fwd_two_v=float(Cf["n_two_v"].sum()),
                         fwd_one_v=float(Cf["n_one_v"].sum()), fwd_corner=float(Cf["n_corner"].sum()),
                         fwd_out=float(Cf["n_out"].sum()), fwd_episodes=int(len(Df["env"])))
    print(f"counts: {res['counts']}", flush=True)
    # 1-2. intact forward vs stored ck; scores vs the exports; forward set vs the stored forward records
    ex = _exports()
    G = forward_set()
    tm = tw.tgt_mask(G["bB"])
    res["forward"], res["score"], res["fwd_records"], res["probe_held"] = {}, {}, {}, {}
    pcache = np.load(os.path.join(tw.OUT, "probe_trained.npz"))
    for ep in CHECK_EPS:
        x = float(train_x(ep))
        w = load("trained", ep)
        z = _ck("trained", ep)
        fo = forward(w, device=dev)
        cmp_ = _logq_vs_stored(fo["logq"], fo["right"], z)
        cmp_["seconds"] = fo["info"]["seconds"]
        res["forward"][str(ep)] = cmp_
        print(f"\nraw {ep} (training {x:,.0f}): intact forward vs stored ck logq: max |d| {cmp_['max_abs']:.2e} = "
              f"{cmp_['max_ulp']:.2f} fp16 ulp; fp16(engine) == stored for {cmp_['fp16_equal']:.6f} of entries; right "
              f"differs at {cmp_['right_diff_all']} of {z['right'].size:,} (n, t, v), at T {cmp_['right_diff_T']}; "
              f"argmax at T differs {cmp_['argmax_T_diff']} ({cmp_['argmax_T_diff_not_tie']} not at an fp16 tie; fp16 "
              f"ties at T {cmp_['fp16_ties_T']}); {cmp_['seconds']:.2f} s", flush=True)
        rs = score(z["logq"], "full", right=z["right"])
        re_ = score(fo["logq"], "full", right=fo["right"])
        rh = score(fo["logq"].astype(np.float16), "full", right=fo["right"])        # the engine rounded as stored
        ff = forward(w, device=dev, episodes=G)
        rf = score_fwd(ff["logq"])
        rfh = score_fwd(ff["logq"].astype(np.float16))
        rec = np.load(os.path.join(tw.OUT, "fwd_trained", f"ck_{ep:05d}.npz"))
        lqf = ff["logq"]
        mf = np.exp(lqf.astype(np.float64)) @ tw.V
        res["fwd_records"][str(ep)] = dict(
            lq_tgt_max_abs=float(np.abs(lqf[tm].astype(np.float64) - rec["lq_tgt"].astype(np.float64)).max()),
            lq_tgt_fp16_equal=float((lqf[tm].astype(np.float16) == rec["lq_tgt"]).mean()),
            m_max_abs=float(np.abs(mf - rec["m"].astype(np.float64)).max()),
            answers_T_diff=int((lqf[:, -1].argmax(-1) != rec["lqT"].argmax(-1)).sum()))
        fr = res["fwd_records"][str(ep)]
        print(f"  forward set (5,028 episodes) vs td_tt_wormholes' stored records: log beliefs at the split steps max |d| "
              f"{fr['lq_tgt_max_abs']:.2e} (fp16 equal {fr['lq_tgt_fp16_equal']:.6f}), posterior means max |d| "
              f"{fr['m_max_abs']:.2e}, answers at T differ {fr['answers_T_diff']}", flush=True)
        ys, ye = resample(rs), resample(re_)
        yf = resample(rf)
        sc = {}
        for topic, name, key in EXPORT_MAP:
            e = ex[(topic, name, "trained")]
            k = np.flatnonzero(e["x"] == x)
            if not len(k):
                continue
            k = int(k[0])
            if key.endswith("_fwd"):
                d = dict(export=float(e["value"][k]), stored=None, engine=rf[key]["value"],
                         d_stored=None, d_engine=_nanmax_absdiff(rf[key]["value"], e["value"][k]),
                         boot_stored=None, boot_engine=_nanmax_absdiff(yf[key], e["boot"][:, k]),
                         d_engine_fp16=_nanmax_absdiff(rfh[key]["value"], e["value"][k]))
            else:
                d = dict(export=float(e["value"][k]), stored=rs[key]["value"], engine=re_[key]["value"],
                         d_stored=_nanmax_absdiff(rs[key]["value"], e["value"][k]),
                         d_engine=_nanmax_absdiff(re_[key]["value"], e["value"][k]),
                         boot_stored=_nanmax_absdiff(ys[key], e["boot"][:, k]),
                         boot_engine=_nanmax_absdiff(ye[key], e["boot"][:, k]),
                         d_engine_fp16=_nanmax_absdiff(rh[key]["value"], e["value"][k]))
            sc[key] = d
            st_txt = ("" if d["stored"] is None else
                      f"stored-logq score {d['stored']:+.6f} (|d| {d['d_stored']:.1e}, resamples {d['boot_stored']:.1e}); ")
            print(f"  {name[:52]:52s} export {d['export']:+.6f} | {st_txt}engine {d['engine']:+.6f} "
                  f"(|d| {d['d_engine']:.1e}, resamples {d['boot_engine']:.1e}; engine rounded to fp16 |d| "
                  f"{d['d_engine_fp16']:.1e})", flush=True)
        kp = int(np.flatnonzero(pcache["episode"] == ep)[0])
        ph = {}
        for key, ckey, nkey in PROBE_HELD:
            ph[key] = dict(cache=float(pcache[ckey][kp].sum() / pcache[nkey].sum()), stored=rs[key]["value"],
                           engine=re_[key]["value"],
                           prim_diff_stored=float(np.abs(rs[key]["num"] - pcache[ckey][kp]).max()),
                           count_diff=float(np.abs(rs[key]["den"] - pcache[nkey]).max()))
            print(f"  {LABELS[key][0][:52]:52s} wormholes probe cache {ph[key]['cache']:.6f} | stored-logq score "
                  f"{ph[key]['stored']:.6f} (per-env sums |d| {ph[key]['prim_diff_stored']:.1e}, counts |d| "
                  f"{ph[key]['count_diff']:.1e}) | engine {ph[key]['engine']:.6f}", flush=True)
        res["score"][str(ep)] = sc
        res["probe_held"][str(ep)] = ph
        if ep == CHECK_EPS[1]:                                     # the trimmed jumps primitives vs the originals
            S = _side()
            lq = z["logq"].astype(np.float64)
            q = np.exp(lq - logsumexp(lq, -1, keepdims=True))
            jp = _jump_prims(q, S["null"])
            o1 = tj.stats(z["logq"], tj._null_perm(n, T - 1))
            o2 = tj.extra_stats(z["logq"], z["right"][np.arange(n), -1, P["goal"]], gap.boot_counts())
            same = dict(path=np.array_equal(jp["path"], o1["path"]), rho=np.array_equal(jp["rho"], o1["rho"], equal_nan=True),
                        e=np.array_equal(jp["e"], o2["e_es"].sum(-1)), ephi=np.array_equal(jp["ephi"], o2["ephi_es"].sum(-1)),
                        n=np.array_equal(jp["n"], o2["n_es"].sum(-1)), nphi=np.array_equal(jp["nphi"], o2["nphi_es"].sum(-1)),
                        band_cnt=np.array_equal(jp["band_cnt"], o2["band_cnt"][:, 0]),
                        band_sphi=np.array_equal(jp["band_sphi"], o2["band_sphi"][:, 0]))
            ast = tl.answer_stats(z["logq"], P["r"], P["env"], right=z["right"])
            same.update(far=np.array_equal(rs["far_T"]["num"] / rs["far_T"]["den"], ast["far_env"][-1]),
                        merged=bool(np.allclose(rs["merged_T"]["num"] / rs["merged_T"]["den"], ast["coll_env"][-1],
                                                rtol=0, atol=1e-15)),
                        both=bool(np.allclose(rs["acc_both_T"]["num"] / 200., ast["right_env"][-1], rtol=0, atol=1e-15)))
            res["trimmed_identical"] = {k: bool(v) for k, v in same.items()}
            print(f"  trimmed copies vs td_tt_jumps.stats / extra_stats and td_tt_latent.answer_stats (bit-identical): "
                  f"{res['trimmed_identical']}", flush=True)
    # references vs the exports' naive / Bayes levels
    rf_ = refs("full")
    res["refs"] = {}
    for topic, name, key in EXPORT_MAP:
        e = ex[(topic, name, "trained")]
        bb, nb = resample({key: rf_["bayes"][key]})[key], resample({key: rf_["naive"][key]})[key]
        res["refs"][key] = dict(naive=rf_["naive"][key]["value"], bayes=rf_["bayes"][key]["value"],
                                export_naive=e["naive"], export_bayes=e["bayes"],
                                d=max(_nanmax_absdiff(rf_["naive"][key]["value"], e["naive"]),
                                      _nanmax_absdiff(rf_["bayes"][key]["value"], e["bayes"])),
                                d_boot=max(_nanmax_absdiff(nb, e["naive_boot"]), _nanmax_absdiff(bb, e["bayes_boot"])))
    worst = max(res["refs"].values(), key=lambda v: v["d"])
    print(f"\nreferences (naive, Bayes) vs every export row's levels: max |d| "
          f"{max(v['d'] for v in res['refs'].values()):.1e}, resamples {max(v['d_boot'] for v in res['refs'].values()):.1e}"
          f" (worst row {[k for k, v in res['refs'].items() if v is worst][0]})", flush=True)
    for key in ("acc_goal_T", "logq_goal_T", "kl_goal_T", "kl_goal_mean", "kl_both_T", "kl_both_mean", "logq_both_T",
                "held_joint", "held_two", "held_one"):
        print(f"  {LABELS[key][0][:52]:52s} naive {rf_['naive'][key]['value']:+.4f}  Bayes {rf_['bayes'][key]['value']:+.4f}")
    # 3. memory cut from the start state vs the stored truncation
    res["cut"] = {}
    srcs = [(w_, "start") for w_ in (1, 2, 5)] + [(5, "same_cell"), (5, "random")]
    for ep in CUT_EPS:
        w = load("trained", ep)
        z = _ck("trained", ep)
        fo = forward(w, device=dev, cut=srcs)
        d = {}
        for j, w_ in enumerate((1, 2, 5)):
            rc = fo["right_cut"][f"w{w_}_start"][:, -1]
            d[f"w{w_}"] = dict(mismatch=int((rc != z["trunc_right"][:, j]).sum()),
                               acc_goal=float(rc[np.arange(n), P["goal"]].mean()), stored_acc_goal=float(z["trunc_acc"][j, 0]),
                               acc_both_var=float(rc.mean()), stored_acc_both_var=float(z["trunc_acc"][j, 1]))
        d["intact_mismatch"] = int((fo["right"][:, -1] != z["trunc_right"][:, 3]).sum())
        for src in ("same_cell", "random"):
            rc = fo["right_cut"][f"w5_{src}"][:, -1]
            d[f"w5_{src}_acc_goal"] = float(rc[np.arange(n), P["goal"]].mean())
        d["intact_acc_goal"] = float(fo["right"][np.arange(n), -1, P["goal"]].mean())
        d["seconds"] = fo["info"]["seconds"]
        res["cut"][str(ep)] = d
        print(f"\nmemory cut from the start state, raw {ep} (training {train_x(ep):,.0f}; {d['seconds']:.1f} s for 5 cuts): "
              + "; ".join(f"w {w_}: answers at T differ from trunc_right at {d[f'w{w_}']['mismatch']} of {2 * n:,}, goal "
                          f"acc {d[f'w{w_}']['acc_goal']:.4f} (stored {d[f'w{w_}']['stored_acc_goal']:.4f})"
                          for w_ in (1, 2, 5))
              + f"; intact vs trunc_right[:, 3] {d['intact_mismatch']}", flush=True)
        print(f"  smoke test (not an experiment): goal accuracy at T, intact {d['intact_acc_goal']:.4f}, w 5 from the start "
              f"state {d['w5']['acc_goal']:.4f}, from the same cell's other stream {d['w5_same_cell_acc_goal']:.4f}, from a "
              f"random other cell of the environment {d['w5_random_acc_goal']:.4f}", flush=True)
    # 4. identities
    wa, wb = load("trained", 2100), load("trained", 49995)
    fa, fb = forward(wa, device=dev, keep=("logits",)), forward(wb, device=dev)
    idn = {}
    h0_, h1_ = hybrid(wa, wb, "all", 0.), hybrid(wa, wb, "all", 1.)
    idn["hybrid_weights_alpha0"] = bool(all(torch.equal(h0_[k], wa[k]) for k in KEYS))
    idn["hybrid_weights_alpha1"] = bool(all(torch.equal(h1_[k], wb[k]) for k in KEYS))
    idn["hybrid_logq_alpha0"] = float(np.abs(forward(h0_, device=dev)["logq"] - fa["logq"]).max())
    idn["hybrid_logq_alpha1"] = float(np.abs(forward(h1_, device=dev)["logq"] - fb["logq"]).max())
    hp = hybrid(wa, wb, "W_hh_f", 1.)
    sl = slice(H, 2 * H)
    idn["hybrid_partial_exact"] = bool(torch.equal(hp["LSTM.weight_hh_l0"][sl], wb["LSTM.weight_hh_l0"][sl]) and
                                       torch.equal(hp["LSTM.weight_hh_l0"][:H], wa["LSTM.weight_hh_l0"][:H]) and
                                       torch.equal(hp["LSTM.weight_hh_l0"][2 * H:], wa["LSTM.weight_hh_l0"][2 * H:]) and
                                       all(torch.equal(hp[k], wa[k]) for k in KEYS if k != "LSTM.weight_hh_l0"))
    hh = hybrid(wa, wb, "W_hh", .5)
    idn["hybrid_half_W_hh"] = float((hh["LSTM.weight_hh_l0"] - .5 * (wa["LSTM.weight_hh_l0"] + wb["LSTM.weight_hh_l0"]))
                                    .abs().max())
    idn["clamp_V0"] = float(np.abs(forward(wa, device=dev, clamp={"u": np.zeros((H, 10))})["logq"] - fa["logq"]).max())
    Wro = wa["classifier_readout.weight"].double().numpy().T
    idn["clamp_readout_rows_rank"] = int(orth_readout(Wro, wa).shape[1])
    idn["clamp_readout_rows"] = float(np.abs(forward(wa, device=dev, clamp={"u": Wro})["logq"] - fa["logq"]).max())
    ref_a = reference(wa, dev, cov=True)
    V = random_subspace(50, wa, 1)
    mu = torch.as_tensor(ref_a["mu_u"], device=dev).float()
    Vt = torch.as_tensor(V, device=dev).float()
    acc = {"res": 0.}

    def chk(s, rows, st):
        acc["res"] = max(acc["res"], float(((st["u"] - mu[s]) @ Vt).abs().max()))
    fc = forward(wa, device=dev, clamp={"u": V}, record=chk, keep=("logits",))
    idn["clamp50_residual_in_V"] = acc["res"]
    idn["clamp50_step1_logit_change"] = float(np.abs(fc["logits"][:, 0] - fa["logits"][:, 0]).max())
    idn["clamp50_readout_orth"] = float(np.abs(Wro.T @ V).max())
    cv = captured_variance(V, ref_a, wa)
    idn["clamp50_variance_share"] = float(cv["pooled_share"])
    idn["clamp50_variance_share_span_minus_clamp"] = float(cv["span_pooled_share"] - cv["pooled_share"])
    idn["clamp50_acc_goal"] = float(fc["right"][np.arange(n), -1, P["goal"]].mean())
    idn["intact_acc_goal"] = float(fa["right"][np.arange(n), -1, P["goal"]].mean())
    sd = forward(wa, device=dev, cut=(5, np.arange(n)))
    idn["self_donor_cut_max_abs"] = float(np.abs(sd["logq_cut"]["w5_donor0"] - fa["logq"]).max())
    idn["self_donor_cut_right_diff"] = int((sd["right_cut"]["w5_donor0"] != fa["right"]).sum())
    sg = forward(wa, device=dev, gate=dict(resample=GATES, donor=np.arange(n)))
    idn["self_donor_resample_max_abs"] = float(np.abs(sg["logq"] - fa["logq"]).max())
    idn["self_donor_resample_right_diff"] = int((sg["right"] != fa["right"]).sum())
    gm = forward(wa, device=dev, gate=dict(mean=GATES), keep=("logits",))
    idn["mean_ablation_step1_logit_change"] = float(np.abs(gm["logits"][:, 0] - fa["logits"][:, 0]).max())
    drv = {"s": np.zeros((T + 1, 4 * H))}

    def mrec(s, rows, st):
        drv["s"][s] += st["rec"].double().sum(0).cpu().numpy()
    forward(wa, device=dev, record=mrec)
    idn["mean_rec_vs_direct"] = float(np.abs(drv["s"][1:] / n - ref_a["mean_rec"][1:]).max())
    units = np.random.default_rng(3).choice(H, 100, replace=False)
    fval = {"d": 0.}
    fmed = torch.as_tensor(ref_a["f_med"], device=dev).float()

    def fchk(s, rows, st):
        fval["d"] = max(fval["d"], float((st["f"][:, units] - fmed[s]).abs().max()))
    forward(wa, device=dev, gate=dict(f_units=units), record=fchk)
    idn["f_clamp_value"] = fval["d"]
    cval = {"d": 0.}
    muc = torch.as_tensor(ref_a["mu_c"], device=dev).float()

    def cchk(s, rows, st):
        cval["d"] = max(cval["d"], float((st["c"][:, units] - muc[s, units]).abs().max()))
    forward(wa, device=dev, gate=dict(c_units=units), record=cchk)
    idn["c_clamp_value"] = cval["d"]
    # brute force in float64, independent of the streaming branches and the chunking: donor cuts and gate drives for a
    # few probes, from the intact states of their donors
    pid, steps = np.array([0, 777, 4321, 9998]), (8, 19, 30)
    ds_, dr_, dg_ = _donors_cached("same_cell"), _donors_cached("random"), _donors_cached("random", GATE_SEED)
    kp = forward(wa, rows=np.unique(np.concatenate([pid, ds_[pid], dr_[pid], dg_[pid]])), device=dev, keep=("u", "c"))
    pos = {int(r): k for k, r in enumerate(kp["rows"])}
    p64 = {k: v.double() for k, v in wa.items()}

    def drive(i, t):
        x = torch.relu(p64["classifier_readin.weight"] @ torch.cat([torch.as_tensor(P["obs"][i, t - 1], dtype=torch.float64),
                                                                   torch.as_tensor(P["Z"][P["env"][i]], dtype=torch.float64)
                                                                   .reshape(-1)]) + p64["classifier_readin.bias"])
        return p64["LSTM.weight_ih_l0"] @ x + p64["LSTM.bias_ih_l0"] + p64["LSTM.bias_hh_l0"]
    fo = forward(wa, device=dev, cut=[(5, "same_cell"), (5, "random")])
    worst, moved = 0., 0.
    for src, dd in (("same_cell", ds_), ("random", dr_)):
        lc = fo["logq_cut"][f"w5_{src}"].astype(np.float64)
        for i in pid:
            for t in steps:
                h = torch.as_tensor(kp["u"][pos[int(dd[i])], t - 6], dtype=torch.float64)    # donor's state at t - 5
                c = torch.as_tensor(kp["c"][pos[int(dd[i])], t - 6], dtype=torch.float64)
                for s in range(t - 4, t + 1):
                    ig, fg, gg, og = (drive(i, s) + p64["LSTM.weight_hh_l0"] @ h).chunk(4)
                    c = fg.sigmoid() * c + ig.sigmoid() * gg.tanh()
                    h = og.sigmoid() * c.tanh()
                lb = (p64["classifier_readout.weight"] @ h + p64["classifier_readout.bias"]).reshape(NV, R).numpy()
                inc = lc[i, t - 1] - lc[i, t - 2]
                worst = max(worst, float(np.abs((inc - inc.mean(-1, keepdims=True)) - (lb - lb.mean(-1, keepdims=True)))
                                         .max()))
                ii = fa["logits"][i, t - 1].reshape(NV, R).astype(np.float64)
                moved = max(moved, float(np.abs((ii - ii.mean(-1, keepdims=True)) - (lb - lb.mean(-1, keepdims=True)))
                                         .max()))
    idn["brute_donor_cut_max_abs"] = worst
    idn["brute_donor_cut_vs_intact"] = moved
    gr = {}

    def grec(s, rows, st):
        sel_ = np.flatnonzero(np.isin(rows, pid))
        if len(sel_):
            gr.setdefault(s, []).extend(zip(rows[sel_], st["rec"][torch.as_tensor(sel_, device=dev)].double().cpu().numpy()))
    forward(wa, device=dev, gate=dict(resample="f"), record=grec)
    gw = 0.
    assert all(len(gr[s]) == len(pid) for s in range(1, T + 1))
    for s in range(1, T + 1):
        for rr, vec in gr[s]:
            hprev = (p64["STM"].reshape(-1) if s == 1 else
                     torch.as_tensor(kp["u"][pos[int(dg_[rr])], s - 2], dtype=torch.float64))
            exp_f = (p64["LSTM.weight_hh_l0"][H:2 * H] @ hprev).numpy()
            gw = max(gw, float(np.abs(vec[H:2 * H] - exp_f).max()))
    idn["brute_resample_drive_max_abs"] = gw
    res["identities"] = idn
    print("\nidentities: " + "; ".join(f"{k} {v:.2e}" if isinstance(v, float) else f"{k} {v}" for k, v in idn.items()),
          flush=True)
    # 5. the 11-02 fixes (verifier issues): cut parsing, the clamp's actual basis
    res["fixes"] = _fix_checks(dev, wa, wb, ref_a)
    if keep_timing:                                    # the saved run's timing, not re-measured (the GPU is shared)
        fn = os.path.join(OUT, "checks.json")
        with open(fn) as f:
            old = json.load(f)
        res["timing"] = old["timing"]
        res["timing_note"] = old.get("timing_note", "kept from the checks.json saved "
                                     + time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(fn))))
        print(f"\ntiming: {res['timing_note']} (not re-measured)", flush=True)
    else:
        res["timing"] = timing(dev)
    res["finished"] = time.strftime("%Y-%m-%d %H:%M")
    res["seconds_total"] = time.time() - t_all
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "checks.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"\nsaved {os.path.join(OUT, 'checks.json')} ({time.time() - t_all:.0f} s)\n")
    summary(json.loads(json.dumps(res, default=float)))
    return res


def timing(dev=None, reps=2):
    """GPU seconds per forward of the 10,000 probes for each hook (the faster of reps runs; while GPU 0 is shared) and
    CPU seconds per score set (this process's threads)"""
    dev = _device(dev)
    P = probes()
    n = len(P["env"])
    w = load("trained", 2100)
    H = w["LSTM.weight_hh_l0"].shape[1]
    t = {}
    _REF.clear()
    t0 = time.time()
    reference(w, dev)
    t["reference (first, incl. the clean pass)"] = time.time() - t0
    _REF.clear()
    t0 = time.time()
    reference(w, dev, cov=True)
    t["reference with cov_u (first)"] = time.time() - t0
    conds = {"intact": {}, "cut w 5 start": dict(cut=(5, "start")), "cut w 1, 2, 5 start":
             dict(cut=[(1, "start"), (2, "start"), (5, "start")]), "cut w 5, three sources":
             dict(cut=[(5, s) for s in SOURCES]), "cut w 10 start": dict(cut=(10, "start")),
             "clamp u 50 dims": dict(clamp={"u": random_subspace(50, w, 1)}), "gate f mean": dict(gate=dict(mean="f")),
             "gate f resample": dict(gate=dict(resample="f")), "gate all four resample": dict(gate=dict(resample=GATES)),
             "f clamp 100 units": dict(gate=dict(f_units=np.arange(100))),
             "cut w 5 random + gate f resample + clamp": dict(cut=(5, "random"), gate=dict(resample="f"),
                                                              clamp={"u": random_subspace(50, w, 1)})}
    peak = {}
    for nm, kw in conds.items():
        best = np.inf
        for _ in range(reps):
            fo = forward(w, device=dev, **kw)
            best = min(best, fo["info"]["seconds"])
        t[f"forward 10,000: {nm}"] = best
        peak[f"forward 10,000: {nm}"] = fo["info"]["peak_gpu_mb"]
    lq = fo["logq"]
    lq_int = forward(w, device=dev)["logq"]
    for which in ("cheap", "full"):
        score(lq_int, which)                                                  # warm caches (probe side, far pairs)
        t0 = time.time()
        rows = score(lq, which)
        t[f"score {which} (CPU)"] = time.time() - t0
        t0 = time.time()
        resample(rows)
        t[f"resample {which} rows (CPU)"] = time.time() - t0
    refs("full")
    t0 = time.time()
    gap_rows(score(lq, "cheap"))
    t["score cheap + gap_rows (CPU)"] = time.time() - t0
    G = forward_set()
    ff = forward(w, device=dev, episodes=G)
    t["forward forward-set 5,028 (GPU)"] = ff["info"]["seconds"]
    score_fwd(ff["logq"])
    t0 = time.time()
    score_fwd(ff["logq"])
    t["score_fwd (CPU)"] = time.time() - t0
    t0 = time.time()
    rows_cpu = np.flatnonzero(P["env"] < 5)
    forward(w, rows=rows_cpu, device="cpu")
    t[f"forward 1,000 probes intact on CPU ({torch.get_num_threads()} threads)"] = time.time() - t0
    for k, v in t.items():
        print(f"  {k:60s} {v:7.2f} s" + (f"   (peak GPU {peak[k]:,.0f} MB)" if k in peak else ""), flush=True)
    return dict(seconds=t, peak_gpu_mb=peak, H=H, n=n)


def summary(res):
    """the docstring's numbers from a checks result"""
    eps = [str(e) for e in res["check_eps"]]
    fw = res["forward"]
    print(f"SUMMARY ({res['device']}; checkpoints raw {', '.join(eps)} = training "
          f"{', '.join(f'{train_x(int(e)):,.0f}' for e in eps)})")
    g, d, c = res["groups"], res["donors"], res["counts"]
    print(f"groups: {g['n_fine']} fine groups over {g['n_params']:,} weights, each weight in exactly one: "
          f"{g['covered_once']}")
    print(f"donors: same_cell = row ^ 1 for every probe {d['same_cell_is_other_stream']}; random and gate donors in the "
          f"same environment {d['random_same_env']}, another cell {d['random_other_cell']}; {d['random_distinct_donors']:,} "
          f"distinct random donors; {d['gate_vs_cut_donor_same']} probes with the same cut and gate donor")
    print(f"counts: probes {c['probe_two']:.0f} corner-split stream-steps, {c['probe_two_v']:.0f} / {c['probe_one_v']:.0f} "
          f"var-steps, {c['probe_var_steps']:,.0f} var-steps of the maps, {c['probe_big_moves']:,.0f} moves >= 2 cells, "
          f"{c['far_pairs_per_env']:,.0f} far pairs per environment; forward set {c['fwd_episodes']:,} episodes, "
          f"{c['fwd_two']:,.0f} / {c['fwd_two_v']:,.0f} / {c['fwd_one_v']:,.0f}, {c['fwd_corner']:,.0f} corner episodes, "
          f"{c['fwd_out']:,.0f} moves out")
    print("intact forward vs stored ck logq: max |d| " + ", ".join(f"{fw[e]['max_abs']:.3g}" for e in eps)
          + f" (max {max(fw[e]['max_ulp'] for e in eps):.2f} fp16 ulp); fp16(engine) == stored "
          + ", ".join(f"{fw[e]['fp16_equal']:.6f}" for e in eps) + "; right differs at "
          + f"{sum(fw[e]['right_diff_all'] for e in eps)} of 5 x 600,000; argmax at T differs "
          + ", ".join(str(fw[e]['argmax_T_diff']) for e in eps) + f" ({sum(fw[e]['argmax_T_diff_not_tie'] for e in eps)} "
          + "not at an fp16 tie; ties at T " + ", ".join(str(fw[e]['fp16_ties_T']) for e in eps) + ")")
    fr = res["fwd_records"]
    print("forward set vs td_tt_wormholes' stored records: log beliefs max |d| " + ", ".join(
        f"{fr[e]['lq_tgt_max_abs']:.3g}" for e in eps) + "; fp16 equal >= "
          + f"{min(fr[e]['lq_tgt_fp16_equal'] for e in eps):.6f}; posterior means max |d| "
          + f"{max(fr[e]['m_max_abs'] for e in eps):.2g}; answers at T differ "
          + ", ".join(str(fr[e]['answers_T_diff']) for e in eps) + " of 5,028")
    sc = res["score"]
    keys = list(sc[eps[0]])
    ds = [sc[e][k][f] for e in eps for k in keys for f in ("d_stored", "boot_stored") if sc[e][k][f] is not None]
    print(f"stored-logq scores vs the exports (every row x checkpoint): max |d| {max(ds):.2g} (values and resamples)")
    print("engine scores vs the exports, max over the checkpoints of |d| value / resamples / engine rounded to fp16:")
    for k in keys:
        a = [sc[e][k]["d_engine"] for e in eps]
        b = [sc[e][k]["boot_engine"] for e in eps]
        h = [sc[e][k].get("d_engine_fp16", np.nan) for e in eps]
        print(f"  {k:20s} {max(a):.2g} / {max(b):.2g} / {np.nanmax(h) if np.isfinite(h).any() else np.nan:.2g}")
    ph = res["probe_held"]
    print("probe held rows vs td_tt_wormholes' probe cache: per-environment sums max |d| "
          f"{max(ph[e][k]['prim_diff_stored'] for e in eps for k in ph[e]):.2g}, counts "
          f"{max(ph[e][k]['count_diff'] for e in eps for k in ph[e]):.2g}; held per variable (corner / one-variable), "
          "engine: " + ", ".join(f"{ph[e]['held_two']['engine']:.3f} / {ph[e]['held_one']['engine']:.3f}" for e in eps))
    print(f"trimmed copies bit-identical: {all(res['trimmed_identical'].values())} {res['trimmed_identical']}")
    rr = res["refs"]
    print(f"references vs the exports' naive / Bayes levels: max |d| {max(v['d'] for v in rr.values()):.2g}, "
          f"resamples {max(v['d_boot'] for v in rr.values()):.2g}")
    for e, v in res["cut"].items():
        print(f"memory cut, start state, raw {e} (training {train_x(int(e)):,.0f}): answers at T differ from "
              f"trunc_right at {v['w1']['mismatch']} / {v['w2']['mismatch']} / {v['w5']['mismatch']} of 20,000 (w 1 / 2 / "
              f"5), intact {v['intact_mismatch']}; goal accuracy w 1 / 2 / 5 {v['w1']['acc_goal']:.4f} / "
              f"{v['w2']['acc_goal']:.4f} / {v['w5']['acc_goal']:.4f}; smoke test w 5: start {v['w5']['acc_goal']:.4f}, "
              f"same cell {v['w5_same_cell_acc_goal']:.4f}, random cell {v['w5_random_acc_goal']:.4f}, intact "
              f"{v['intact_acc_goal']:.4f}; {v['seconds']:.1f} s for 5 cuts")
    print("identities: " + "; ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
                                     for k, v in res["identities"].items()))
    fx = res.get("fixes")
    if fx:
        pc = fx["parse"]
        bad = [k for k, v in pc.items() if v["old_error"] != "none"]
        print(f"fix 1 (_parse_cuts): {sum(v['ok'] for v in pc.values())} of {len(pc)} cut specs parse as specified; "
              f"{len(bad)} raised ValueError before the fix ({', '.join(bad)}); the others got the same single / list "
              f"decision before: {all(v['same_decision'] for v in pc.values() if v['same_decision'] is not None)}")
        for k, v in fx["parse_forward"].items():
            print(f"  forward {k} vs the same cuts named by source: max |d logq| {v['max_abs']:.2g} (main and cuts), "
                  f"right differs {v['right_diff']}; the cuts move log q by up to {v['cut_vs_intact']:.3g}")
        print("fix 2 (clamp basis), pooled clean variance share of u in span(V) / in the clamped basis V' = "
              "orth_readout(V); share of span(V)'s squared norm (mean cos^2) in the read-out rows' span; largest "
              "principal angle V' vs span(V):")
        for k, v in fx["clamp_basis"].items():
            e_, kind = k.split("_", 1)
            print(f"  raw {e_} (training {train_x(int(e_)):,.0f}) {kind:18s} {v['span_pooled_share']:.4f} / "
                  f"{v['pooled_share']:.4f}; {v['readout_share']:.3f}; {v['max_angle_deg']:.1f} deg (dims {v['k']} -> "
                  f"{v['k_clamp']})")
        c_ = fx["clamp_pcs10"]
        print(f"  clamp of the raw top 10 PCs at raw {c_['ep']} (ranks {c_['rank']}): |(u - mu) V'| <= "
              f"{c_['residual_in_clamp_basis']:.2g} after every step, while the mean square deviation from the clean "
              f"mean left inside span(V) is {c_['span_left_share']:.4f} of the clean total (clean "
              f"{c_['span_clean_share']:.4f}; clamped basis {c_['clamp_clean_share']:.4f})")
    tm = res["timing"]
    print(f"timing (H {tm['H']}, {tm['n']:,} probes; GPU 0 shared with another job at ~99% utilisation"
          + (f"; {res['timing_note']}" if res.get("timing_note") else "") + "):")
    for k, v in tm["seconds"].items():
        print(f"  {k:60s} {v:6.2f} s" + (f" (peak GPU {tm['peak_gpu_mb'][k]:,.0f} MB)" if k in tm["peak_gpu_mb"] else ""))
    print(f"checks total {res['seconds_total']:.0f} s" + (f" (finished {res['finished']}; timing not re-measured)"
                                                          if res.get("timing_note") else ""))


def numbers():
    """prints the docstring's numbers from the saved checks (OUT/checks.json)"""
    with open(os.path.join(OUT, "checks.json")) as f:
        summary(json.load(f))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "doc"
    if what == "checks":
        checks(keep_timing=len(sys.argv) > 2 and sys.argv[2] == "keep_timing")
    elif what == "timing":
        timing()
    elif what == "numbers":
        numbers()
    else:
        print(__doc__)
