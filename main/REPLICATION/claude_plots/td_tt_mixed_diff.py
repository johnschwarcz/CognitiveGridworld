"""Observation x past interaction through training (Fig 2d of the learning figures): does the network read a new
observation in light of earlier ones? The mixed second difference of the step-t read-out increment under one flipped
observation bit at step t and one at an earlier step s, for the Trained RNN and the ESN through training, with optimal
Bayes's exact value as the reference. Raw quantities only: no decoder, no fit.
-> main/REPLICATION/figure_bin/td_tt_mixed_diff.png
   main/DATA/experimental/training_dynamics/through_training/mixed_diff/: ck_<run>_<raw>.npz (one per checkpoint),
   bayes.npz, checks.json, summary.npz

RESULTS  (`numbers`; 1,000 probes = 50 test environments x 20, 32 (t, s) pairs, 25 bit pairs, both variables; mean
  |MD| per value in nats [5-95% over 1,000 environment resamples]; training episodes; passing naive at 835)
  Theorem zeros hold in the code: naive max |MD| 2.2e-16, memory-free Bayes 4.4e-16, a random linear reservoir run
    through the networks' code path 1.8e-14 (float64; mean 9.5e-16, its increments ~1.8 nats).
  Bayes (exact): .0748 [.0717, .0779] (rms .172: a few large terms carry the energy). Flat in lag, as exchangeable
    observations require (t = 15-30: .0564 at every lag 1-12); falls with t (lags 1-3: .176 / .102 / .073 / .059 /
    .051 / .043 at t = 5 / 10 / 15 / 20 / 25 / 30). Goal .0749, context .0746.
  Trained RNN acquires an observation x past interaction of Bayes's order. Initial network .00005 (shared with the
    ESN) -> .0015 by 20 episodes (kappa 0: the gates' non-linearity, not Bayes's pattern) -> .0055 (200) -> .0128
    [.0124, .0131] (480) -> .0199 [.0194, .0203] (880) -> .0226 (1,040) -> .0284 (1,600) -> .0356 (2,400) -> .0459
    (4,000) -> a plateau at .054 from 8,000 on; end .0542 [.0529, .0554] = 72% of Bayes's.
  Only part of it is Bayes's pattern. kappa ~0 to 80 episodes; .001 (200); .007 [.006, .008] (480); .017 [.015, .020]
    (880); .022 (1,040); .035 (1,600); .053 (2,400); .083 (4,000); .127 (8,000); end .193 [.178, .209], still rising
    after mean |MD| has plateaued. energy .014 (480) -> .033 (880) -> .233 (end); residual (not Bayes's) .014 -> .032
    -> .196. At the end kappa falls with t (lags 1-3): .40 / .19 / .13 / .094 / .074 / .067 at t = 5 ... 30 (Bayes's MD
    energy: 27% at t = 5, 9% at t = 30), so the pooled kappa leans on early steps; at 880 it is flat in t (.029 ->
    .020).
  Its reach in lag grows through training (t = 15-30, lag 1 vs lag 12): .0197 vs .0033 (480), .0333 vs .0052 (880),
    .0369 vs .0062 (1,040), .0454 vs .0082 (1,600), .0665 vs .0193 (4,000), end .0525 vs .0332; Bayes .0564 at both.
    kappa by lag at the end .112 (lag 1) -> .081 (lag 12); at 880 .031 -> .001. By t (lags 1-3) at the end .135 / .075
    / .055 / .047 / .042 / .039 (Bayes's decline with t); at 880 nearly flat, .032 -> .027.
  ESN: near zero throughout. .00005 -> .00034 (480) -> .00045 [.00044, .00045] (880) -> .0005 (1,040) -> .0012 (4,000)
    -> .0020 (8,000) -> end .0049 [.0048, .0050] = 6.6% of Bayes's (the Trained RNN's is 44x larger at 880 and 11x at
    the end). kappa <= .0016 at every checkpoint (end .0016 [.0013, .0020]); energy .0027 at the end. Short-range: at
    the end .0133 / .0071 / .0041 / .0016 / .0004 / .0001 at lags 1 / 2 / 3 / 5 / 8 / 12 (about halving per step)
    against Bayes's flat .056; flat in t (.0081-.0084).
  Goal vs context: equal in every agent (Trained RNN end .0543 / .0541, kappa .194 / .192; ESN .0049 / .0049).
  Other trained runs at the end (`extras`): N = 200 / 100 / 50 and power 1 (N = 1000): mean |MD| .0554 / .0448 / .0506
    / .0500 (all of Bayes's order), kappa .124 / .071 / .045 / .173 (main run .193). The size of the interaction does
    not order with width; its Bayes-aligned share does.
  The preview (wf1 theory_designer/mixed_diff.py) is reproduced on its own subset to every printed digit (CHECKS (6)):
    Bayes .0629; Trained RNN .0149 / .0223 / .0508 at 480 / 880 / end, kappa .005 / .012 / .113, energy .019 / .041 /
    .199; ESN .0005 / .0059 (880 / end). The full set's end kappa (.193) is larger than the preview's (.113) because it
    adds t = 5 (kappa .40) and weighs by Bayes's energy, which is largest early; the preview's pairs on these 1,000
    probes give .137.
  Cost: 56 s per checkpoint at 4 CPU threads (float32, 250 probes per pass); 68 checkpoints in 64 min.

DEFINITIONS
  Probes: td_tt_chaos.subset() (= td_tt_chaos_ent.rows()): 20 episodes per test environment x 50 environments = 1,000
    probe episodes of 30 steps from analysis/probes.npz, the same for every agent and checkpoint.
  Read-out increment: the networks emit belief increments; their log marginal belief at t is the cumulative sum
    Sigma_{k<=t} (W_ro u_k + b_ro), log-softmaxed per variable. inc_t = W_ro u_t + b_ro, each variable's 10 values
    centred (minus their mean). Bayes: inc_t = log p(r_v | o_1..t) - log p(r_v | o_1..t-1), centred: the change of the
    exact marginal log posterior at step t (uniform prior; the joint posterior over the 100 cells, marginalized).
  Mixed second difference: flip observation bit i of step t (o_t -> o_t') and bit j of an earlier step s = t - d
    (o_s -> o_s'); every other observation intact (the networks are rerun from their intact state at s - 1, with the
    intact observations between s and t):
        MD = inc_t(o_t', o_s') - inc_t(o_t', o_s) - inc_t(o_t, o_s') + inc_t(o_t, o_s)
    per variable and value. MD is the part of the new observation's effect on the update that depends on the earlier
    observation: 0 iff flipping o_t moves the update by the same amount whatever o_s was. Steps t in {5, 10, 15, 20, 25,
    30}, lags d = t - s in {1, 2, 3, 5, 8, 12} with s >= 1 (32 (t, s) pairs: t = 5 has d <= 3, t = 10 has d <= 8), all
    5 x 5 bit pairs (i, j), both variables.
  Why exactly 0 for naive Bayes, memory-free Bayes and any linear reservoir read out linearly and summed (theorem;
    theory_v11 Prop. formats (a) and Cor. memory, the window theorem): if the step-t increment is a sum of functions of
    single steps' observations, inc_t = f_t(o_t) + Sigma_{k<t} g_{t,k}(o_k), flipping o_s changes it by g_{t,s}(o_s') -
    g_{t,s}(o_s) whatever o_t is, and the four terms cancel exactly. Naive Bayes: inc_t = Sigma_i log
    pbar_i(o_t,i | r_v) (each observation's likelihood averaged over the other variable), a function of o_t alone.
    Memory-free (step-wise) Bayes: inc_t = log Sigma_c prod_i p(o_t,i | r_v, c), a function of o_t alone. A reservoir
    that responds linearly, r_t = A r_{t-1} + B phi(o_t, z), read out linearly: W r_t = Sigma_{k<=t} W A^(t-k) B
    phi(o_k, z) + W A^t r_0, a sum of per-step functions for ANY memory A, so no length or width of linear memory
    produces an observation x past interaction; centring over values is linear and keeps the sum. Optimal Bayes's
    increment, log Sigma_c p(o_t | r_v, c) p(c | r_v, o_<t) (up to a constant over values), weighs o_t's likelihoods by
    the past through p(c | r_v, o_<t): zero when the observations do not depend on the other variable c, non-zero
    otherwise (theory_v11 A1). An LSTM's gates make u_t non-linear in (u_{t-1}, o_t); in a near-linear regime its MD
    is near 0. A network whose belief equalled Bayes's at every step would have Bayes's increments, so MD = MD_Bayes
    (kappa = 1, energy = 1, residual = 0).
  Statistics (pooled: every (probe, pair, bit pair, variable, value) term weighs the same):
    mean |MD| per value (nats) = Sigma |MD| / number of terms; Bayes's exact mean |MD| is the reference.
    kappa (Bayes-aligned share) = Sigma MD MD_Bayes / Sigma MD_Bayes^2, the coefficient of the projection of the
      network's MD pattern on Bayes's (1 = Bayes's observation x past interaction; 0 = none, or orthogonal to it).
    energy = Sigma MD^2 / Sigma MD_Bayes^2; residual = Sigma (MD - kappa MD_Bayes)^2 / Sigma MD_Bayes^2 = energy -
      kappa^2: the part of the network's interaction that is not Bayes's.
  Intervals: 5-95% over the 1,000 shared environment resamples (td_tt_gap.boot_counts(): each resample draws the 50
    test environments with replacement; every statistic is a ratio of per-environment sums, recomputed per resample).
  Checkpoints: RAWS (34 per run, raw episode numbers; training episodes = raw - raw // 5, the initial network drawn at
    10), log-spaced with every stored checkpoint from 480 to 1,280 training episodes (passing naive: 835).
  Profiles: by lag over the steps that have every lag (t = 15-30, T_BAL: 4 steps x 6 lags, so the lag profile is not
    confounded with t); by step over the lags every step has (1-3, LAG_BAL). Goal / context: each probe's goal variable
    and its contextual variable (probes.npz 'goal'), each paired with itself.
  Precision: the observers in float64; the networks in float32 (their stored precision), 250 probes per pass. The
    float32 MD agrees with a float64 rerun to 0.03% of mean |MD| or better at every tested checkpoint (CHECKS (7)), the
    initial network's 5e-5 nats included.
  Commands: time | checks | check_linear | harvest [runs] | extras | summary | figure | numbers | all.
  Caches (mixed_diff/): ck_<run>_<raw>.npz: s_abs, s_nb, s_nn (1,000 probes, 32 pairs, 2 variables r0 / r1) = per
    probe, pair and variable the sums over the 25 bit pairs and 10 values of |MD|, MD MD_Bayes and MD^2; ji_abs, ji_nb,
    ji_nn (32 pairs, 5 j = bit of s, 5 i = bit of t, 2 variables) = sums over probes and values; pairs (t, s), rows, x,
    secs. bayes.npz: s_abs, s_bb (1,000, 32, 2), ji_abs, ji_bb, rows, env, pairs. summary.npz: x_<run>, raw_<run>;
    md_<run>, md_<run>_lo, md_<run>_hi (mean |MD| per value, 5-95%); kappa_, energy_, resid_<run> (+ _lo / _hi);
    md_lag_<run> (+ _lo / _hi), kappa_lag_<run> (K, 6 lags; t = 15-30); md_t_<run>, kappa_t_<run> (K, 6 steps; lags
    1-3); md_role_<run>, kappa_role_<run> (K, goal / context); rms_<run>; md_bayes (+ _lo / _hi), rms_bayes,
    md_lag_bayes, md_t_bayes, md_role_bayes; extra_runs with md_extra_end (+ _lo / _hi), kappa_extra_end (+ _lo / _hi),
    energy_extra_end (end of training of the widths N = 200 / 100 / 50 and the power-1 run); pairs, lags, ts, t_bal,
    lag_bal, n_probes, n_env, n_boot, boot_seed; secs_<run> (harvest seconds per checkpoint).

CHECKS  (`checks`, `check_linear`; written to mixed_diff/checks.json; all passed)
  (1) Bayes reconstruction: the exact joint posterior's centred log marginals vs the stored float32 probes.npz
      'log_jb', all 30 steps, both variables, the 1,000 probes: max 8.0e-6, mean 3.1e-7 (float32 storage).
  (2) Bayes's MD by an independent path: the posterior recomputed from scratch for each of the four flipped observation
      sequences (no cumulative-sum shortcut; 100 probes x 4 pairs x 4 bit pairs): max |difference| 5.1e-14.
  (3) Networks' MD by an independent path: td_tt_chaos_ent.replay of the four whole flipped episodes per bit pair from
      the start state (50 probes x 4 pairs x 4 bit pairs): max |difference| 1.8e-6 / 3.8e-6 / 1.7e-6 / 9.5e-7 for the
      Trained RNN / ESN at the end / Trained RNN / ESN at 880 (float32 rounding; mean |MD| .065 / .0042 / .019 /
      .00043).
  (4) Theorem zeros: naive's and memory-free Bayes's MD through the same differencing code as Bayes's (max 2.2e-16,
      4.4e-16); a random float64 linear reservoir (H = 64, spectral radius .95) through the same code path as the
      networks (max 1.8e-14).
  (5) The step from a shared recurrent product (_act) vs td_tt_chaos.lstm_step: bitwise equal.
  (6) The preview (wf1 theory_designer/mixed_diff.py: 500 probes, t in {10, 20, 30}, d in {1, 3, 8}) reproduced on its
      own subset (RESULTS).
  (7) float32 vs float64 networks (the first 250 probes, all 32 pairs): mean |difference| / mean |MD| 3.1e-4 (initial
      network, mean |MD| 5.1e-5), 2.3e-4 (ESN 880), 1.2e-4 (ESN end), 9e-6 (Trained RNN 880), 4.5e-6 (Trained RNN
      end)."""
import os
import sys
import json
import time
import numpy as np
import torch
from scipy.special import logsumexp

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_intervene as ti                                                                  # noqa: E402
import td_tt_chaos as tc                                                                      # noqa: E402
import td_tt_chaos_ent as ce                                                                  # noqa: E402

T, R, NV, NO = ti.T, ti.R, ti.NV, ti.NO
OUT = os.path.join(gap.TT, "mixed_diff")
RUNS = ("trained", "echo")
EXTRA_RUNS = ("trained_N200", "trained_N100", "trained_N50", "trained_p1")        # end of training only (`extras`)
TS = (5, 10, 15, 20, 25, 30)
LAGS = (1, 2, 3, 5, 8, 12)
PAIRS = tuple((t, t - d) for t in TS for d in LAGS if t - d >= 1)               # (t, s): 32 pairs
T_BAL, LAG_BAL = (15, 20, 25, 30), (1, 2, 3)        # profiles: by lag over the steps with every lag; by t over lags 1-3
RAWS = (0, 25, 50, 75, 100, 150, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
        1800, 2000, 2500, 3000, 4000, 5000, 7000, 10000, 15000, 20000, 30000, 40000, 49995)
DT = torch.float64 if os.environ.get("MD_DTYPE") == "64" else torch.float32        # networks: fp32 (CHECKS (7))
THREADS = int(os.environ.get("MD_THREADS", "4"))
CHUNK = int(os.environ.get("MD_CHUNK", "250" if DT == torch.float32 else "125"))  # probes per pass: 6 x 250 x 4H x 4 B
LIN_SEED = 20261109


# ═══════════════════════════════════════════════════════════════════
# Observers (exact, float64): Bayes, naive, memory-free Bayes
# ═══════════════════════════════════════════════════════════════════

def side(rws=None):
    """observations (n, T, NO) int, environments (n,), log p(x = 1 | r0, r1) and log p(x = 0 | r0, r1) (n, NO, R, R)"""
    rws = ce.rows() if rws is None else rws
    P = np.load(os.path.join(ti.AN, "probes.npz"))
    obs, env = P["obs"][rws].astype(np.int64), P["env"][rws]
    L = P["L"][env].astype(np.float64)                                           # in [.021, .979]: no clipping
    return dict(rows=np.asarray(rws), obs=obs, env=env, l1=np.log(L), l0=np.log1p(-L),
                log_jb=P["log_jb"][rws].astype(np.float64))


def ll_step(S, o):
    """joint log-likelihood of one step's observations o (n, NO) over the 100 cells: (n, R, R)"""
    return (o[:, :, None, None] * S["l1"] + (1 - o[:, :, None, None]) * S["l0"]).sum(1)


def flip(o, b):
    o2 = o.copy()
    o2[:, b] = 1 - o2[:, b]
    return o2


def cen(a):
    return a - a.mean(-1, keepdims=True)


def marg(J):
    """(n, R, R) joint log weights -> (n, 2, R) log marginals (variable 0 = r0 = axis 1, variable 1 = r1 = axis 2)"""
    return np.stack([logsumexp(J, 2), logsumexp(J, 1)], 1)


def inc_obs(kind, S, prev, o):
    """an observer's centred step increment (n, 2, R) from its joint log weights before the step (prev, (n, R, R)) and
    the step's observations o: 'bayes' = the change of the exact marginal log posterior; 'naive' = Sigma_i log
    pbar_i(o_i | r_v), pbar averaged over the other variable (uniform); 'stepwise' (memory-free Bayes) = log Sigma_c
    prod_i p(o_i | r_v, c). The last two ignore prev."""
    if kind == "bayes":
        return cen(marg(prev + ll_step(S, o)) - marg(prev))
    if kind == "stepwise":
        return cen(marg(ll_step(S, o)))
    if kind == "naive":
        p1 = np.exp(S["l1"])                                                      # (n, NO, R, R)
        pb = np.stack([p1.mean(3), p1.mean(2)], 1)                       # (n, 2, NO, R): the other variable averaged
        oo = o[:, None, :, None]
        return cen((oo * np.log(pb) + (1 - oo) * np.log1p(-pb)).sum(2))
    raise ValueError(kind)


def observer_md(kind, S, pairs=PAIRS):
    """MD (len(pairs), 5 j, 5 i, n, 2, R) of an exact observer: bit j of step s, bit i of step t flipped"""
    obs = S["obs"]
    n = len(obs)
    LLs = np.stack([ll_step(S, obs[:, k]) for k in range(T)], 1)                  # (n, T, R, R)
    cum = np.concatenate([np.zeros((n, 1, R, R)), np.cumsum(LLs, 1)], 1)          # cum[:, k] = steps 1..k
    out = np.zeros((len(pairs), NO, NO, n, NV, R))
    for p, (t, s) in enumerate(pairs):
        base = cum[:, t - 1]
        prevs = [base] + [base - LLs[:, s - 1] + ll_step(S, flip(obs[:, s - 1], j)) for j in range(NO)]
        ots = [obs[:, t - 1]] + [flip(obs[:, t - 1], i) for i in range(NO)]
        I = np.array([[inc_obs(kind, S, pv, o) for o in ots] for pv in prevs])  # (6 states, 6 inputs, n, 2, R)
        out[p] = I[1:, 1:] - I[1:, :1] - I[:1, 1:] + I[:1, :1]
    return out


def bayes_md_direct(S, rows_sub, pairs, bits):
    """CHECK (2): Bayes's MD with the posterior recomputed from scratch for each flipped observation sequence"""
    out = []
    for (t, s) in pairs:
        for (j, i) in bits:
            vals = {}
            for fs in (0, 1):
                for ft in (0, 1):
                    o = S["obs"][rows_sub].copy()
                    if fs:
                        o[:, s - 1, j] = 1 - o[:, s - 1, j]
                    if ft:
                        o[:, t - 1, i] = 1 - o[:, t - 1, i]
                    sub = {k: S[k][rows_sub] for k in ("l1", "l0")}
                    lt = sum(ll_step(sub, o[:, k]) for k in range(t))
                    lp = sum(ll_step(sub, o[:, k]) for k in range(t - 1))
                    vals[(fs, ft)] = cen(marg(lt) - marg(lp))
            out.append(vals[(1, 1)] - vals[(1, 0)] - vals[(0, 1)] + vals[(0, 0)])
    return np.array(out)


# ═══════════════════════════════════════════════════════════════════
# Networks (CPU, float64)
# ═══════════════════════════════════════════════════════════════════

def _act(pre, c):
    """the LSTM step from its full pre-activation (td_tt_chaos.lstm_step after its recurrent product, same operations)
    """
    i, f, g, o = pre.chunk(4, -1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    c2 = f * c + i * g
    return o * c2.tanh(), c2


class LSTMCell:
    """a checkpoint's LSTM: drive table (NE, 32, 4H) incl. both biases, recurrent product, step, centred read-out"""

    def __init__(self, run, raw, dtype=DT):
        self.p, self.table = ce.setup(run, raw, torch.device("cpu"), dtype)
        self.Whh = self.p["LSTM.weight_hh_l0"]
        self.Wro, self.bro = self.p["classifier_readout.weight"], self.p["classifier_readout.bias"]

    def rec(self, h):
        return h @ self.Whh.T

    def act(self, pre, c):
        return _act(pre, c)

    def inc(self, h):
        x = torch.nn.functional.linear(h, self.Wro, self.bro).reshape(-1, NV, R)
        return x - x.mean(-1, keepdim=True)

    def replay(self, envt, code):
        _, U, C, _ = ce.replay(self.p, self.table, envt, code)
        return U, C


class LinearCell:
    """CHECK (4): a random linear reservoir r_t = A r_{t-1} + B phi(o_t, z), read out linearly; the same interface"""

    def __init__(self, H=64, rho=.95, seed=LIN_SEED, dtype=torch.float64):
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(H, H, generator=g, dtype=dtype)
        self.A = A * rho / torch.linalg.eigvals(A).abs().max().to(dtype)
        self.table = torch.randn(gap.N_ENV, 1 << NO, H, generator=g, dtype=dtype)
        self.Wro = torch.randn(NV * R, H, generator=g, dtype=dtype) / H ** .5
        self.bro = torch.randn(NV * R, generator=g, dtype=dtype)
        self.h0 = torch.randn(H, generator=g, dtype=dtype)

    def rec(self, h):
        return h @ self.A.T

    def act(self, pre, c):
        return pre, c

    def inc(self, h):
        x = torch.nn.functional.linear(h, self.Wro, self.bro).reshape(-1, NV, R)
        return x - x.mean(-1, keepdim=True)

    def replay(self, envt, code):
        n, H = len(envt), self.A.shape[0]
        U = torch.empty(n, T + 1, H, dtype=self.A.dtype)
        U[:, 0] = self.h0
        for t in range(T):
            U[:, t + 1] = self.table[envt, code[:, t]] + self.rec(U[:, t])
        return U, torch.zeros_like(U)


def codes(rws):
    envt, code = ce.codes(rws, torch.device("cpu"))
    return envt, code


def net_md(cell, envt, code, U, C, pairs=PAIRS, reduce=None, chunk=None):
    """MD of the network's centred read-out increment for every pair (t, s), bit j of step s and bit i of step t.
    Fast path: the 5 flipped trajectories of step s start from the intact state at s - 1 (one recurrent product shared
    by the 5 flips) and run with the intact observations to t - 1; at t the 6 patterns (intact o_t and its 5 flips)
    share one recurrent product per state. reduce(p, md, sl) is called with md (5 j, 5 i, len(sl), 2, R) float64 numpy
    per pair and probe chunk sl (returns the full array (len(pairs), 5, 5, n, 2, R) when reduce is None). chunk: probes
    per pass (CHUNK: every tensor stays below glibc's 32 MB mmap threshold, so freed memory is reused without page
    faults)"""
    n = len(envt)
    chunk = chunk or CHUNK
    if n > chunk:
        full = None if reduce is not None else np.zeros((len(pairs), NO, NO, n, NV, R))
        for a in range(0, n, chunk):
            sl = slice(a, min(n, a + chunk))
            if reduce is None:
                def red(p, md, s2, sl=sl):
                    full[p][:, :, sl] = md
            else:
                def red(p, md, s2, sl=sl):
                    reduce(p, md, sl)
            net_md(cell, envt[sl], code[sl], U[sl], C[sl], pairs, red, chunk)
        return full
    need = {}
    for p, (t, s) in enumerate(pairs):
        need.setdefault(s, {})[t] = p
    full = None if reduce is not None else np.zeros((len(pairs), NO, NO, n, NV, R))
    env5, env6 = envt.repeat(NO), envt.repeat(NO + 1)

    def evaluate(t, hf, cf):
        h0, c0 = torch.cat([U[:, t - 1], hf]), torch.cat([C[:, t - 1], cf])              # (6n, H): intact, flips j
        Rc = cell.rec(h0)
        I = []
        for ii in range(NO + 1):
            cd = code[:, t - 1] if ii == 0 else code[:, t - 1] ^ (1 << (ii - 1))
            h1, _ = cell.act(cell.table[env6, cd.repeat(NO + 1)] + Rc, c0)
            I.append(cell.inc(h1).reshape(NO + 1, n, NV, R))
        I = torch.stack(I, 1)                                                             # (6 states, 6 inputs, ...)
        return (I[1:, 1:] - I[1:, :1] - I[:1, 1:] + I[:1, :1]).cpu().numpy()

    with torch.no_grad():
        for s in sorted(need):
            ts = need[s]
            cds = torch.cat([code[:, s - 1] ^ (1 << j) for j in range(NO)])              # j-major (5n,)
            h, c = cell.act(cell.table[env5, cds] + cell.rec(U[:, s - 1]).repeat(NO, 1), C[:, s - 1].repeat(NO, 1))
            k = s                                                                         # h, c = state after step k
            for t in sorted(ts):
                while k < t - 1:
                    k += 1
                    h, c = cell.act(cell.table[env5, code[:, k - 1].repeat(NO)] + cell.rec(h), c)
                md = evaluate(t, h, c)
                p = ts[t]
                if reduce is None:
                    full[p] = md.reshape(NO, NO, n, NV, R)
                else:
                    reduce(p, md.reshape(NO, NO, n, NV, R), slice(0, n))
    return full


def net_md_direct(cell, envt, code, pairs, bits):
    """CHECK (3): MD from replays of the four whole flipped episodes (from the start state) per pair and bit pair"""
    out = []
    with torch.no_grad():
        for (t, s) in pairs:
            for (j, i) in bits:
                vals = {}
                for fs in (0, 1):
                    for ft in (0, 1):
                        cd = code.clone()
                        if fs:
                            cd[:, s - 1] ^= (1 << j)
                        if ft:
                            cd[:, t - 1] ^= (1 << i)
                        U, _ = cell.replay(envt, cd)
                        vals[(fs, ft)] = cell.inc(U[:, t]).cpu().numpy()
                out.append(vals[(1, 1)] - vals[(1, 0)] - vals[(0, 1)] + vals[(0, 0)])
    return np.array(out)


# ═══════════════════════════════════════════════════════════════════
# Harvest: per-probe sums per checkpoint
# ═══════════════════════════════════════════════════════════════════

def bayes_file():
    return os.path.join(OUT, "bayes.npz")


def ck_file(run, raw):
    return os.path.join(OUT, f"ck_{run}_{int(raw):05d}.npz")


def bayes_ref(S=None):
    """Bayes's exact MD (len(PAIRS), 5, 5, n, 2, R) on the probes (computed once; its per-probe sums cached)"""
    S = side() if S is None else S
    MDb = observer_md("bayes", S)
    if not os.path.exists(bayes_file()):
        os.makedirs(OUT, exist_ok=True)
        np.savez(bayes_file(), rows=S["rows"], env=S["env"], pairs=np.array(PAIRS),
                 s_abs=np.abs(MDb).sum((1, 2, 5)).transpose(1, 0, 2),                 # (n, P, 2)
                 s_bb=(MDb ** 2).sum((1, 2, 5)).transpose(1, 0, 2),
                 ji_abs=np.abs(MDb).sum((3, 5)), ji_bb=(MDb ** 2).sum((3, 5)),          # (P, 5, 5, 2)
                 n_terms_per_probe_pair_var=NO * NO * R)
    return MDb


def harvest_one(run, raw, MDb, S, envt, code):
    """per-probe sums (n, P, 2) of |MD|, MD MD_Bayes and MD^2 over the 25 bit pairs and 10 values, and per bit pair
    (P, 5, 5, 2) sums over probes and values -> ck_<run>_<raw>.npz"""
    t0 = time.time()
    cell = LSTMCell(run, raw)
    with torch.no_grad():
        U, C = cell.replay(envt, code)
    n = len(envt)
    acc = {k: np.zeros((n, len(PAIRS), NV)) for k in ("s_abs", "s_nb", "s_nn")}
    ji = {k: np.zeros((len(PAIRS), NO, NO, NV)) for k in ("ji_abs", "ji_nb", "ji_nn")}

    def red(p, md, sl):
        b = MDb[p][:, :, sl]
        acc["s_abs"][sl, p] = np.abs(md).sum((0, 1, 4))
        acc["s_nb"][sl, p] = (md * b).sum((0, 1, 4))
        acc["s_nn"][sl, p] = (md ** 2).sum((0, 1, 4))
        ji["ji_abs"][p] += np.abs(md).sum((2, 4))
        ji["ji_nb"][p] += (md * b).sum((2, 4))
        ji["ji_nn"][p] += (md ** 2).sum((2, 4))

    net_md(cell, envt, code, U, C, reduce=red)
    np.savez(ck_file(run, raw), run=run, raw=raw, x=gap.train_x(raw), pairs=np.array(PAIRS), rows=S["rows"],
             secs=time.time() - t0, **acc, **ji)
    return time.time() - t0


def harvest(runs=RUNS, raws=RAWS):
    torch.set_num_threads(THREADS)
    os.makedirs(OUT, exist_ok=True)
    S = side()
    t0 = time.time()
    MDb = bayes_ref(S)
    print(f"Bayes MD: {time.time() - t0:.0f} s; mean |MD| {np.abs(MDb).mean():.4f}", flush=True)
    envt, code = codes(S["rows"])
    jobs = [(r, x) for x in raws for r in runs]                     # interleaved: both runs advance together
    for k, (run, raw) in enumerate(jobs):
        if os.path.exists(ck_file(run, raw)):
            continue
        dt = harvest_one(run, raw, MDb, S, envt, code)
        z = np.load(ck_file(run, raw))
        print(f"[{k + 1}/{len(jobs)}] {run:7s} raw {raw:5d} (x {gap.train_x(raw):6.0f}): mean |MD| "
              f"{z['s_abs'].sum() / (z['s_abs'].size * NO * NO * R):.5f}  kappa "
              f"{z['s_nb'].sum() / np.load(bayes_file())['s_bb'].sum():+.4f}  {dt:.0f} s", flush=True)


def extras(runs=EXTRA_RUNS, raw=49995):
    """the end-of-training checkpoint of the other trained widths (N = 200, 100, 50) and of the power-1 Trained RNN"""
    harvest(runs, (raw,))


# ═══════════════════════════════════════════════════════════════════
# Summary: environment bootstrap
# ═══════════════════════════════════════════════════════════════════

def _env_sum(a, env):
    """(n, ...) per-probe values -> (50, ...) per-environment sums"""
    out = np.zeros((gap.N_ENV,) + a.shape[1:])
    np.add.at(out, env, a)
    return out


def _stats(env, s_abs, s_nb, s_nn, s_bb, sel=None, C=None):
    """pooled statistics over the probes, (pairs in sel), both variables: mean |MD| per value, kappa, energy,
    residual; each (full, resampled (N_BOOT,))"""
    C = gap.boot_counts() if C is None else C
    sel = slice(None) if sel is None else sel
    f = lambda a: _env_sum(a[:, sel].reshape(len(a), -1).sum(1), env)                      # noqa: E731
    n_cols = np.asarray(s_abs[:, sel]).reshape(len(env), -1).shape[1]                      # (pair, variable) entries
    cnt = _env_sum(np.full(len(env), float(n_cols * NO * NO * R)), env)
    out = {}
    out["md"] = gap.env_ratio(f(s_abs), cnt, C)
    if s_nb is not None:
        out["kappa"] = gap.env_ratio(f(s_nb), f(s_bb), C)
        out["energy"] = gap.env_ratio(f(s_nn), f(s_bb), C)
        out["resid"] = (out["energy"][0] - out["kappa"][0] ** 2, out["energy"][1] - out["kappa"][1] ** 2)
    return out


def _band(b):
    lo, hi = np.percentile(b, [5, 95], 0)
    return lo, hi


def summary(runs=RUNS):
    """-> mixed_diff/summary.npz (keys in the module docstring's RESULTS)"""
    B = np.load(bayes_file())
    env = B["env"]
    C = gap.boot_counts()
    pairs = np.array(PAIRS)
    lag = pairs[:, 0] - pairs[:, 1]
    out = dict(pairs=pairs, lags=np.array(LAGS), ts=np.array(TS), n_probes=len(env), n_env=gap.N_ENV,
               n_boot=gap.N_BOOT, boot_seed=gap.BOOT_SEED)
    sb = _stats(env, B["s_abs"], None, None, None, C=C)
    out["md_bayes"] = sb["md"][0]
    out["md_bayes_lo"], out["md_bayes_hi"] = _band(sb["md"][1])
    out["rms_bayes"] = np.sqrt(B["s_bb"].sum() / (B["s_bb"].size * NO * NO * R))
    sel_lag = [(lag == d) & np.isin(pairs[:, 0], T_BAL) for d in LAGS]          # balanced: t = 15-30 at every lag
    sel_t = [(pairs[:, 0] == t) & np.isin(lag, LAG_BAL) for t in TS]             # balanced: lags 1-3 at every t
    out["t_bal"], out["lag_bal"] = np.array(T_BAL), np.array(LAG_BAL)
    out["md_lag_bayes"] = np.array([_stats(env, B["s_abs"], None, None, None, sel=sl, C=C)["md"][0] for sl in sel_lag])
    out["md_t_bayes"] = np.array([_stats(env, B["s_abs"], None, None, None, sel=sl, C=C)["md"][0] for sl in sel_t])
    goal = np.load(os.path.join(ti.AN, "probes.npz"))["goal"][B["rows"]]
    role = np.stack([goal, 1 - goal], 1)                                          # (n, 2): goal, contextual variable

    def by_role(a):
        """(n, P, 2) per variable -> (n, P, 2) per role (0 = the probe's goal variable, 1 = its contextual one)"""
        return np.take_along_axis(a, role[:, None, :].repeat(a.shape[1], 1), 2)
    out["md_role_bayes"] = by_role(B["s_abs"]).sum((0, 1)) / (len(env) * len(PAIRS) * NO * NO * R)
    for run in runs:
        raws = [r for r in RAWS if os.path.exists(ck_file(run, r))]
        out[f"raw_{run}"] = np.array(raws)
        out[f"x_{run}"] = gap.train_x(np.array(raws))
        cols = {k: [] for k in ("md", "md_lo", "md_hi", "kappa", "kappa_lo", "kappa_hi", "energy", "energy_lo",
                                "energy_hi", "resid", "resid_lo", "resid_hi", "md_lag", "md_lag_lo", "md_lag_hi",
                                "kappa_lag", "md_t", "kappa_t", "rms", "md_role", "kappa_role", "secs")}
        for raw in raws:
            z = np.load(ck_file(run, raw))
            st = _stats(env, z["s_abs"], z["s_nb"], z["s_nn"], B["s_bb"], C=C)
            for k in ("md", "kappa", "energy", "resid"):
                cols[k].append(st[k][0])
                lo, hi = _band(st[k][1])
                cols[k + "_lo"].append(lo)
                cols[k + "_hi"].append(hi)
            ml, ll, hl, kl = [], [], [], []
            for sl in sel_lag:
                sd = _stats(env, z["s_abs"], z["s_nb"], z["s_nn"], B["s_bb"], sel=sl, C=C)
                ml.append(sd["md"][0])
                lo, hi = _band(sd["md"][1])
                ll.append(lo)
                hl.append(hi)
                kl.append(sd["kappa"][0])
            cols["md_lag"].append(ml)
            cols["md_lag_lo"].append(ll)
            cols["md_lag_hi"].append(hl)
            cols["kappa_lag"].append(kl)
            mt, kt = [], []
            for sl in sel_t:
                sd = _stats(env, z["s_abs"], z["s_nb"], z["s_nn"], B["s_bb"], sel=sl, C=C)
                mt.append(sd["md"][0])
                kt.append(sd["kappa"][0])
            cols["md_t"].append(mt)
            cols["kappa_t"].append(kt)
            cols["rms"].append(np.sqrt(z["s_nn"].sum() / (z["s_nn"].size * NO * NO * R)))
            cols["md_role"].append(by_role(z["s_abs"]).sum((0, 1)) / (len(env) * len(PAIRS) * NO * NO * R))
            cols["kappa_role"].append(by_role(z["s_nb"]).sum((0, 1)) / by_role(B["s_bb"]).sum((0, 1)))
            cols["secs"].append(float(z["secs"]))
        for k, v in cols.items():                                                  # md_trained, md_trained_lo, ...
            out[f"{k[:-3]}_{run}{k[-3:]}" if k[-3:] in ("_lo", "_hi") else f"{k}_{run}"] = np.array(v)
    ex = [r for r in EXTRA_RUNS if os.path.exists(ck_file(r, 49995))]
    out["extra_runs"] = np.array(ex)
    for k in ("md", "md_lo", "md_hi", "kappa", "kappa_lo", "kappa_hi", "energy"):
        out[k.replace("_lo", "_extra_end_lo").replace("_hi", "_extra_end_hi") if k[-3:] in ("_lo", "_hi")
            else f"{k}_extra_end"] = np.zeros(len(ex))
    for a, r in enumerate(ex):
        z = np.load(ck_file(r, 49995))
        st = _stats(env, z["s_abs"], z["s_nb"], z["s_nn"], B["s_bb"], C=C)
        for k in ("md", "kappa"):
            out[f"{k}_extra_end"][a] = st[k][0]
            out[f"{k}_extra_end_lo"][a], out[f"{k}_extra_end_hi"][a] = _band(st[k][1])
        out["energy_extra_end"][a] = st["energy"][0]
    path = os.path.join(OUT, "summary.npz")
    np.savez(path, **out)
    print("saved", path)
    return out


# ═══════════════════════════════════════════════════════════════════
# Checks
# ═══════════════════════════════════════════════════════════════════

def checks():
    torch.set_num_threads(THREADS)
    res = {}
    S = side()
    n = len(S["obs"])
    # (1) Bayes reconstruction at every step, both variables
    LLs = np.stack([ll_step(S, S["obs"][:, k]) for k in range(T)], 1)
    cum = np.cumsum(LLs, 1)
    mine = cen(np.stack([marg(cum[:, k]) for k in range(T)], 1))                 # (n, T, 2, R)
    stored = cen(S["log_jb"])
    res["bayes_reconstruction_maxabs"] = float(np.abs(mine - stored).max())
    res["bayes_reconstruction_meanabs"] = float(np.abs(mine - stored).mean())
    print("(1) Bayes reconstruction vs stored log_jb (centred, 30 steps x 2 variables x 1,000 probes): max %.2e, "
          "mean %.2e" % (res["bayes_reconstruction_maxabs"], res["bayes_reconstruction_meanabs"]), flush=True)
    # (2) Bayes MD, independent path
    MDb = observer_md("bayes", S)
    sub = np.arange(0, n, 10)
    cp = [(5, 2), (15, 3), (30, 29), (30, 18)]
    bits = [(0, 0), (1, 3), (4, 2), (2, 4)]
    dir_b = bayes_md_direct(S, sub, cp, bits)
    fast_b = np.array([MDb[PAIRS.index(pr)][j, i][sub] for pr in cp for (j, i) in bits])
    res["bayes_md_direct_maxabs"] = float(np.abs(dir_b - fast_b).max())
    res["bayes_md_scale_meanabs"] = float(np.abs(fast_b).mean())
    print("(2) Bayes MD, fast vs recomputed from scratch (100 probes x 4 pairs x 4 bit pairs): max |diff| %.2e (mean "
          "|MD| %.4f)" % (res["bayes_md_direct_maxabs"], res["bayes_md_scale_meanabs"]), flush=True)
    # (4a) theorem zeros for naive and memory-free Bayes through the same code
    for kind in ("naive", "stepwise"):
        m = observer_md(kind, S)
        res[f"{kind}_md_maxabs"] = float(np.abs(m).max())
        print(f"(4) {kind}: max |MD| {res[f'{kind}_md_maxabs']:.2e} (theorem: 0)", flush=True)
    # (5) _act vs td_tt_chaos.lstm_step
    cell = LSTMCell("trained", 49995)
    envt, code = codes(S["rows"])
    with torch.no_grad():
        U, C = cell.replay(envt, code)
        h_a, c_a = _act(cell.table[envt, code[:, 7]] + cell.rec(U[:, 7]), C[:, 7])
        h_b, c_b, _ = tc.lstm_step(cell.table[envt, code[:, 7]], U[:, 7], C[:, 7], cell.Whh)
    res["act_vs_lstm_step_maxabs"] = float(max((h_a - h_b).abs().max(), (c_a - c_b).abs().max()))
    print("(5) _act vs td_tt_chaos.lstm_step: max |diff| %.2e" % res["act_vs_lstm_step_maxabs"], flush=True)
    # (3) networks: fast path vs whole-episode replays
    sub = np.arange(0, n, 20)                                                     # 50 probes, one per environment
    cp = [(5, 2), (15, 3), (30, 29), (30, 18)]
    bits = [(0, 0), (1, 3), (4, 2), (2, 4)]
    for run, raw in (("trained", 49995), ("echo", 49995), ("trained", 1100), ("echo", 1100)):
        cell = LSTMCell(run, raw)
        e2, c2 = envt[sub], code[sub]
        with torch.no_grad():
            U2, C2 = cell.replay(e2, c2)
        fast = net_md(cell, e2, c2, U2, C2, pairs=cp)
        fast = np.array([fast[a][j, i] for a in range(len(cp)) for (j, i) in bits])
        direct = net_md_direct(cell, e2, c2, cp, bits)
        key = f"net_md_direct_{run}_{raw}"
        res[key + "_maxabs"] = float(np.abs(fast - direct).max())
        res[key + "_scale_meanabs"] = float(np.abs(fast).mean())
        print(f"(3) {run} raw {raw}: fast vs whole-episode replays (50 probes x 4 pairs x 4 bit pairs): max |diff| "
              f"{res[key + '_maxabs']:.2e} (mean |MD| {res[key + '_scale_meanabs']:.5f})", flush=True)
    # (4b) random linear reservoir through the network code path
    lin = LinearCell()
    with torch.no_grad():
        Ul, Cl = lin.replay(envt, code)
        sc = lin.inc(Ul[:, 1:].reshape(-1, Ul.shape[-1])).abs().mean()
    m = net_md(lin, envt, code, Ul, Cl)
    res["linear_reservoir_md_maxabs"] = float(np.abs(m).max())
    res["linear_reservoir_md_meanabs"] = float(np.abs(m).mean())
    res["linear_reservoir_inc_meanabs"] = float(sc)
    res["linear_reservoir_dtype"] = str(lin.A.dtype)
    print("(4) random linear reservoir (H = 64, rho .95, %s), network code path: max |MD| %.2e, mean |MD| %.2e (its "
          "increments: mean |inc| %.3f)" % (lin.A.dtype, res["linear_reservoir_md_maxabs"],
                                            res["linear_reservoir_md_meanabs"], sc), flush=True)
    # (6) the preview on its own subset
    rws_p = np.arange(0, 10000, 20)
    pp = tuple((t, t - d) for t in (10, 20, 30) for d in (1, 3, 8))
    Sp = side(rws_p)
    MDp = observer_md("bayes", Sp, pp)
    res["preview_bayes_md"] = float(np.abs(MDp).mean())
    ep, cpd = codes(rws_p)
    for run, raw in (("trained", 600), ("trained", 1100), ("trained", 49995), ("echo", 1100), ("echo", 49995)):
        cell = LSTMCell(run, raw)
        with torch.no_grad():
            Up, Cp = cell.replay(ep, cpd)
        m = net_md(cell, ep, cpd, Up, Cp, pairs=pp)
        res[f"preview_{run}_{raw}_md"] = float(np.abs(m).mean())
        res[f"preview_{run}_{raw}_kappa"] = float((m * MDp).sum() / (MDp ** 2).sum())
        res[f"preview_{run}_{raw}_energy"] = float((m ** 2).sum() / (MDp ** 2).sum())
    print("(6) preview subset: Bayes %.4f; " % res["preview_bayes_md"] + "; ".join(
        f"{k[8:]} {v:.4f}" for k, v in res.items() if k.startswith("preview_") and k != "preview_bayes_md"), flush=True)
    # (7) float32 vs float64 networks
    sub = S["rows"][:CHUNK]
    e7, c7 = codes(sub)
    for run, raw in (("trained", 0), ("echo", 1100), ("echo", 49995), ("trained", 1100), ("trained", 49995)):
        m = {}
        for dt in (torch.float32, torch.float64):
            cell = LSTMCell(run, raw, dtype=dt)
            with torch.no_grad():
                U7, C7 = cell.replay(e7, c7)
            m[dt] = net_md(cell, e7, c7, U7, C7, chunk=len(sub))
        d = np.abs(m[torch.float32] - m[torch.float64])
        key = f"fp32_vs_fp64_{run}_{raw}"
        res[key + "_rel_meanabs"] = float(d.mean() / np.abs(m[torch.float64]).mean())
        res[key + "_maxabs"] = float(d.max())
        res[key + "_md64"] = float(np.abs(m[torch.float64]).mean())
        print(f"(7) {run} raw {raw}: float32 vs float64 MD ({len(sub)} probes, 32 pairs): mean |diff| / mean |MD| "
              f"{res[key + '_rel_meanabs']:.2e}, max |diff| {res[key + '_maxabs']:.2e} (mean |MD| "
              f"{res[key + '_md64']:.3e})", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "checks.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("saved", os.path.join(OUT, "checks.json"))
    return res


def check_linear():
    """CHECK (4b) alone (float64 linear reservoir through the network code path), merged into checks.json"""
    torch.set_num_threads(THREADS)
    S = side()
    envt, code = codes(S["rows"])
    lin = LinearCell()
    with torch.no_grad():
        Ul, Cl = lin.replay(envt, code)
        sc = lin.inc(Ul[:, 1:].reshape(-1, Ul.shape[-1])).abs().mean()
    m = net_md(lin, envt, code, Ul, Cl)
    path = os.path.join(OUT, "checks.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    res["linear_reservoir_md_maxabs"] = float(np.abs(m).max())
    res["linear_reservoir_md_meanabs"] = float(np.abs(m).mean())
    res["linear_reservoir_inc_meanabs"] = float(sc)
    res["linear_reservoir_dtype"] = str(lin.A.dtype)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print("(4) random linear reservoir (H = 64, rho .95, %s), network code path: max |MD| %.2e, mean |MD| %.2e (its "
          "increments: mean |inc| %.3f)" % (lin.A.dtype, res["linear_reservoir_md_maxabs"],
                                            res["linear_reservoir_md_meanabs"], sc))


def time_one(run="trained", raw=1100):
    """time one checkpoint of the full harvest (not cached)"""
    torch.set_num_threads(THREADS)
    S = side()
    t0 = time.time()
    MDb = bayes_ref(S)
    tb = time.time() - t0
    envt, code = codes(S["rows"])
    t0 = time.time()
    cell = LSTMCell(run, raw)
    with torch.no_grad():
        U, C = cell.replay(envt, code)
    tr = time.time() - t0
    tot = {"abs": 0., "nb": 0.}

    def red(p, md, sl):
        tot["abs"] += np.abs(md).sum()
        tot["nb"] += (md * MDb[p][:, :, sl]).sum()
    t0 = time.time()
    net_md(cell, envt, code, U, C, reduce=red)
    tm = time.time() - t0
    print(f"Bayes {tb:.1f} s; {run} raw {raw}: replay {tr:.1f} s, MD {tm:.1f} s; mean |MD| "
          f"{tot['abs'] / MDb.size:.5f}, kappa {tot['nb'] / (MDb ** 2).sum():+.4f}; Bayes mean |MD| "
          f"{np.abs(MDb).mean():.5f}")


# ═══════════════════════════════════════════════════════════════════
# Figure and numbers
# ═══════════════════════════════════════════════════════════════════

STAGES = ((40, "#9ecae1"), (200, "#6baed6"), (1040, "#2171b5"), (39996, "#08306b"))     # training episodes


def figure():
    import matplotlib.pyplot as plt
    import fig_main_common as fm
    from main.utils import fig_path
    fm.setup()
    FS = 10.5
    plt.rcParams.update({"xtick.labelsize": FS, "ytick.labelsize": FS, "axes.labelsize": FS + .5})
    Z = np.load(os.path.join(OUT, "summary.npz"))
    fig, add = fm.canvas(12.6, 3.75)
    W, Hh, y0 = 2.75, 2.05, .84
    ax_a, ax_b, ax_c = add(.8, y0, W, Hh), add(4.65, y0, W, Hh), add(8.5, y0, W, Hh)
    col = fm.COL
    # a: mean |MD| per value through training
    for run in RUNS:
        x = fm.xs(Z[f"x_{run}"])
        ax_a.fill_between(x, Z[f"md_{run}_lo"], Z[f"md_{run}_hi"], color=col[run], alpha=.22, lw=0)
        ax_a.plot(x, Z[f"md_{run}"], color=col[run], lw=1.6, label=fm.NAME[run])
    ax_a.axhline(float(Z["md_bayes"]), color="k", ls="--", lw=1.1, label="Bayes")
    ax_a.set_yscale("log")
    lo = min(Z["md_echo"].min(), Z["md_trained"].min())
    ax_a.set_ylim(lo / 3, float(Z["md_bayes"]) * 2.5)
    fm.xaxis(ax_a)
    ax_a.set_ylabel("nats per value")
    fm.title(ax_a, "a", r"observation $\times$ past interaction",
             r"$\langle|\Delta_{o_t}\Delta_{o_s}\,\mathrm{inc}_t|\rangle$")
    xe, xtr = fm.xs(Z["x_echo"]), fm.xs(Z["x_trained"])                            # direct labels, placed from the data
    ax_a.text(12, float(Z["md_bayes"]) * 1.25, "Bayes", fontsize=FS, color="k", va="bottom")
    ax_a.text(12, Z["md_trained"][xtr <= 120].max() * 3.6, fm.NAME["trained"], fontsize=FS, color=col["trained"],
              va="bottom")
    ax_a.text(6500, Z["md_echo"][(xe >= 6500) & (xe <= 25000)].max() * 1.5, fm.NAME["echo"], fontsize=FS,
              color=col["echo"], va="bottom")
    ax_a.text(.98, .03, "naive, memory-free Bayes,\nlinear reservoir: 0", transform=ax_a.transAxes, fontsize=FS,
              color="0.35", va="bottom", ha="right", zorder=3,
              bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    # b: kappa through training
    for run in RUNS:
        x = fm.xs(Z[f"x_{run}"])
        ax_b.fill_between(x, Z[f"kappa_{run}_lo"], Z[f"kappa_{run}_hi"], color=col[run], alpha=.22, lw=0)
        ax_b.plot(x, Z[f"kappa_{run}"], color=col[run], lw=1.6, label=fm.NAME[run])
    ax_b.axhline(0, color="0.6", lw=.8)
    fm.xaxis(ax_b)
    ax_b.set_ylabel(r"$\kappa$  (Bayes = 1)")
    fm.title(ax_b, "b", "Bayes-aligned share",
             r"$\kappa=\Sigma\,\mathrm{MD}\,\mathrm{MD}^{\mathrm{B}}\,/\,\Sigma\,(\mathrm{MD}^{\mathrm{B}})^2$")
    ax_b.legend(loc="upper left", fontsize=FS, frameon=False, handlelength=1.6, borderaxespad=.1)
    # c: by lag at the stage checkpoints
    lags = Z["lags"]
    xt = Z["x_trained"]
    for xv, c in STAGES:
        k = int(np.argmin(np.abs(np.log(fm.xs(xt)) - np.log(xv))))
        lab = "end" if xv == STAGES[-1][0] else f"{int(xt[k]):,}"
        ax_c.plot(lags, Z["md_lag_trained"][k], color=c, lw=1.6, marker="o", ms=3.5, label=lab)
    ax_c.plot(lags, Z["md_lag_echo"][-1], color=col["echo"], lw=1.6, marker="o", ms=3.5, label="ESN end")
    ax_c.plot(lags, Z["md_lag_bayes"], color="k", ls="--", lw=1.1, label="Bayes")
    ax_c.set_yscale("log")
    ax_c.set_xscale("log")
    ax_c.set_xticks(lags)
    ax_c.set_xticklabels([str(v) for v in lags])
    ax_c.minorticks_off()
    ax_c.set_xlabel("lag  $t-s$  (steps)")
    ax_c.set_ylabel("nats per value")
    ax_c.spines[["top", "right"]].set_visible(False)
    fm.title(ax_c, "c", "interaction by lag", r"$\langle|\Delta_{o_t}\Delta_{o_s}\,\mathrm{inc}_t|\rangle$ at $t-s$")
    ax_c.legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=FS, frameon=False, handlelength=1.4,
                title="Trained RNN", title_fontsize=FS, alignment="left")
    n_ck = len(Z["x_trained"])
    fm.note(fig, .055, f"{int(Z['n_env'])} environments $\\times$ {int(Z['n_probes']) // int(Z['n_env'])} probes "
            f"$\\times$ {len(Z['pairs'])} step pairs $(t, s)$ $\\times$ 25 bit pairs; {n_ck} checkpoints per run")
    path = fig_path("td_tt_mixed_diff.png")
    fig.savefig(path, dpi=fm.DPI)
    plt.close(fig)
    print("saved", path)


def numbers():
    Z = np.load(os.path.join(OUT, "summary.npz"))
    print("Bayes mean |MD| per value %.4f [%.4f, %.4f]; rms %.4f; by lag %s (t = 15-30): %s; by t %s (lags 1-3): %s; "
          "goal / context %s" % (Z["md_bayes"], Z["md_bayes_lo"], Z["md_bayes_hi"], Z["rms_bayes"], tuple(LAGS),
                                 " ".join("%.4f" % v for v in Z["md_lag_bayes"]), tuple(TS),
                                 " ".join("%.4f" % v for v in Z["md_t_bayes"]),
                                 " ".join("%.4f" % v for v in Z["md_role_bayes"])))
    for k, r in enumerate(Z["extra_runs"]):
        print(f"{r} (end): mean |MD| {Z['md_extra_end'][k]:.5f} [{Z['md_extra_end_lo'][k]:.5f}, "
              f"{Z['md_extra_end_hi'][k]:.5f}]  kappa {Z['kappa_extra_end'][k]:+.4f} "
              f"[{Z['kappa_extra_end_lo'][k]:+.4f}, {Z['kappa_extra_end_hi'][k]:+.4f}]  "
              f"energy {Z['energy_extra_end'][k]:.4f}")
    for run in RUNS:
        print(f"{run}: x | mean|MD| [5-95%] | kappa [5-95%] | energy | residual | rms | mean|MD| by lag {tuple(LAGS)} "
              f"(t = 15-30) | kappa by lag | mean|MD| by t {tuple(TS)} (lags 1-3) | goal / context mean|MD|, kappa")
        for k, x in enumerate(Z[f"x_{run}"]):
            print(f"  {x:7,.0f} | {Z[f'md_{run}'][k]:.5f} [{Z[f'md_{run}_lo'][k]:.5f}, {Z[f'md_{run}_hi'][k]:.5f}] | "
                  f"{Z[f'kappa_{run}'][k]:+.4f} [{Z[f'kappa_{run}_lo'][k]:+.4f}, {Z[f'kappa_{run}_hi'][k]:+.4f}] | "
                  f"{Z[f'energy_{run}'][k]:.4f} | {Z[f'resid_{run}'][k]:.4f} | {Z[f'rms_{run}'][k]:.4f} | "
                  + " ".join("%.4f" % v for v in Z[f"md_lag_{run}"][k]) + " | "
                  + " ".join("%+.3f" % v for v in Z[f"kappa_lag_{run}"][k]) + " | "
                  + " ".join("%.4f" % v for v in Z[f"md_t_{run}"][k]) + " | "
                  + " ".join("%.4f" % v for v in Z[f"md_role_{run}"][k]) + ", "
                  + " ".join("%+.3f" % v for v in Z[f"kappa_role_{run}"][k]))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "time":
        time_one(*(sys.argv[2:3] or ["trained"]), *(int(v) for v in sys.argv[3:4]))
    elif cmd == "checks":
        checks()
    elif cmd == "check_linear":
        check_linear()
    elif cmd == "harvest":
        harvest(tuple(sys.argv[2:]) or RUNS)
    elif cmd == "extras":
        extras()
    elif cmd == "summary":
        summary()
    elif cmd == "figure":
        figure()
    elif cmd == "numbers":
        numbers()
    elif cmd == "all":
        harvest()
        summary()
        figure()
        numbers()
