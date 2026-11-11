"""Gradient-block branches split by factorization regret (Fig. 2e): every stored checkpoint of the train_learn_masks.py
runs replayed on the 10,000 probes (CPU), each probe-variable's answer at T paired with its own forward FR_T.
-> main/DATA/experimental/training_dynamics/through_training/learnmask_fr/<run>_<raw>.npz   per checkpoint (primitives)
   .../learnmask_fr/summary.npz           the branches restarted at 480 training episodes (Fig. 2e)
   .../learnmask_fr/summary_scratch.npz   the from-scratch runs (finished runs only)
   main/REPLICATION/figure_bin/td_tt_learnmask_fr.png   figure() (diagnostic; Fig. 2e is drawn from summary.npz):
     a-d the branches (a-c by FR fifth at 1,280 for 50 / 20 / 5%, d the dissociation through the branches); e-h from
     scratch (e by fifth at the end, f / g the highest- / lowest-FR half blocked per fifth through training, h the
     dissociations of the FR- and entanglement-ranked pairs); redrawn by every scratch pass

Runs (train_learn_masks.py; td_tt_learnmask.py). Branches: the Trained RNN's trunk at raw episode 600 (480 training
episodes) restarted with the trunk's Adam state, every branch on the same episode stream (seed 20261108), trained to raw
1,600 (1,280), checkpoints every 100 raw (11). At every step the gradient is blocked through a share (5 / 20 / 50%) of
the batch's updates, ranked by goal entanglement within the step: the most entangled (det_hi*), the least (det_lo*), a
random share (det_rand*); ctrl blocks nothing. One run per condition. Every branch's raw-600 weights equal the control's
and the trunk's (checked), so every difference is exactly 0 at 480. From-scratch runs (learnmask_*_s0, seed 20261029):
episode 0 to raw 3,000 (2,400), half of each step blocked, ranked by goal FR (*_fr_*) or goal entanglement, or random;
compared with ctrl_s0.

Definitions.
  probes        analysis/probes.npz: 10,000 episodes = 50 test environments x 100 latent pairs x 2 streams of 30 steps
  answer at T   the network's log-odds at T = the sum of its 30 per-step read-outs (2 variables x 10 values);
                right_v = [argmax_k logit_T(v, k) = r_v], each variable's own answer against its own truth
  FR_T          each probe-variable's own forward factorization regret at T: KL(Bayes_T || naive_T) of its marginal,
                Sigma_k B_T(k) (log B_T(k) - log N_T(k)), from probes.npz log_jb / log_nb (log-softmaxed in float64; NOT
                the stored symmetric 'fr')
  fifths        quintiles of FR_T pooled over the 20,000 probe-variables (4,000 each; each variable keeps its own FR)
  accuracy      per fifth: right answers / probe-variables of the fifth, pooled over environments
  d             branch - control accuracy per fifth (same probes, same fifths, same checkpoint)
  dd            double dissociation (most - least)_top fifth - (most - least)_bottom fifth
                = d_hi[5] - d_lo[5] - d_hi[1] + d_lo[1] (the control cancels)
  intervals     paired 5-95% over 2,000 resamples of the 50 test environments (the same resample weights for every run;
                seed 0, the scheme of the workflow's blk_boot.py); test sampling only, NOT run-to-run variability
  values        full-sample values (blk_boot.py printed the bootstrap median for dd; the two agree to +-.001)

RESULTS 11-09, branches (10 runs x 11 checkpoints = 110 replays; 20,000 probe-variables each; one run per condition).
  FR_T fifths: edges 0 / .073 / .263 / .740 / 2.70 / 94.9 nats, means .028 / .156 / .465 / 1.44 / 12.8 (summary
  fifth_mean_fr). Accuracy by fifth, naive .824 / .732 / .559 / .163 / .025 (all .460), Bayes .833 / .777 / .758 / .753
  / .727 (.770); control at 480 .442 / .354 / .282 / .226 / .184 (.298), at 880 .661 / .547 / .475 / .436 / .351 (.494),
  at 1,280 .745 / .662 / .604 / .571 / .487 (.614).
  Branch - control at 1,280, fifths 1..5 [5-95%] | all fifths:
    50% most   -.020 [-.030,-.011] -.029 [-.043,-.016] -.039 [-.056,-.024] -.056 [-.071,-.041] -.081 [-.102,-.061] | -.045
    50% least  -.051 [-.064,-.039] -.079 [-.095,-.063] -.071 [-.088,-.054] -.072 [-.087,-.057] -.059 [-.081,-.038] | -.067
    50% random -.024 [-.036,-.013] -.038 [-.050,-.026] -.038 [-.051,-.025] -.038 [-.052,-.023] -.054 [-.073,-.036] | -.038
    20% most   -.014 [-.026,-.002] -.013 [-.027,+.001] -.013 [-.030,+.002] -.020 [-.035,-.003] -.031 [-.047,-.015] | -.018
    20% least  -.012 [-.020,-.003] -.017 [-.028,-.006] -.023 [-.034,-.012] -.021 [-.035,-.006] -.003 [-.018,+.015] | -.015
    20% random -.014 [-.024,-.005] -.015 [-.026,-.004] -.014 [-.026,-.002] -.007 [-.019,+.005] -.009 [-.025,+.008] | -.012
    5%  most   -.003 [-.013,+.006] -.001 [-.013,+.009] +.003 [-.006,+.014] +.009 [-.002,+.020] +.013 [-.002,+.028] | +.004
    5%  least  -.011 [-.021,-.002] -.014 [-.027,-.001] -.020 [-.033,-.007] -.023 [-.037,-.008] -.005 [-.021,+.010] | -.015
    5%  random -.021 [-.035,-.008] -.011 [-.025,+.003] -.009 [-.021,+.004] -.006 [-.020,+.010] -.004 [-.020,+.014] | -.010
  (all-fifth values pool both variables; td_tt_learnmask's goal-only values are -.049 / -.069 / -.042 at 50%.)
  Relative to the random block at 1,280 (bottom fifth | top fifth | top - bottom):
    50% most  +.004 [-.007,+.014] | -.026 [-.044,-.007] | -.030 [-.050,-.009]
    50% least -.027 [-.039,-.016] | -.005 [-.024,+.015] | +.022 [+.001,+.043]
    20% most  -.000 [-.015,+.015] | -.022 [-.038,-.008] | -.022 [-.043,-.002]
    20% least +.002 [-.009,+.014] | +.006 [-.007,+.020] | +.004 [-.015,+.022]
    5%  most  +.018 [+.007,+.030] | +.016 [-.001,+.032] | -.002 [-.023,+.018]
    5%  least +.010 [-.003,+.023] | -.002 [-.016,+.012] | -.012 [-.030,+.007]
  Double dissociation dd at 480 / 560 / 640 / 720 / 800 / 880 / 960 / 1,040 / 1,120 / 1,200 / 1,280 training episodes:
    50%  0 / -.046 / -.035 / -.068 / -.026 / -.042 / -.056 / -.084 / -.092 / -.070 / -.052; at 1,280 -.052 [-.073, -.031],
         800 -.026 [-.055, +.002], 1,120 -.092 [-.113, -.072]; negative at 10 of the 10 post-branch checkpoints, interval
         excluding 0 at 9 (not 800); mean of the 10: -.057 [-.069, -.045]
    20%  0 / -.022 / -.034 / +.003 / -.024 / -.021 / -.021 / -.040 / -.034 / -.026 / -.026; at 1,280 -.026 [-.046, -.009];
         negative at 9 of 10, excluding 0 at 9 (not 720); mean -.024 [-.032, -.017]
    5%   0 / -.026 / +.007 / +.009 / -.009 / +.006 / +.001 / -.015 / -.001 / -.007 / +.010; at 1,280 +.010 [-.009, +.026];
         negative at 5 of 10, excluding 0 at 1 (560); mean -.003 [-.009, +.003]
  Excess log loss KL(Bayes_T || q_T) at 1,280 (nats): control .281 / .424 / .530 / .746 / 2.202; branch - control in
  the top fifth 50% most +1.63 [+1.20, +2.03], least +.88 [+.61, +1.17], random +.99 [+.56, +1.48]; bottom fifth most
  +.14 [+.08, +.20], least +.40 [+.31, +.51], random +.22 [+.10, +.40]; 20% top fifth most +.86 [+.48, +1.31], least
  -.05 [-.30, +.18], random +.13 [-.12, +.37].
  Read: blocking the most entangled updates costs most where FR is high, blocking the least entangled ones costs most
  where it is low, in proportion to the share blocked (5% none, 20% about half of 50%).
  Run-to-run noise, which the intervals (test sampling only) do not contain: ctrl_s0 is bitwise the branch trunk up to
  raw 600 (checked) and then trains unblocked on another episode stream (seed 20261029 vs 20261108), i.e. a second
  control branch from the same 480 state. Against the branch control it differs by .010 per fifth on average (max .040)
  over the 10 later checkpoints; its top - bottom contrast (the form of dd) has mean +.006, sd .011, max |.025|; dd5
  (same stream, 5% blocks) has sd .012 over checkpoints. dd50 at 1,280 (-.052) is about 4.5 such sd and negative at all
  10 checkpoints; dd20 (-.026) about 2 sd, negative at 9 of 10. An independent run pair reproduces dd50 (from scratch,
  entanglement-ranked halves, below); branch seeds are still needed for a seed-level spread.
  Checks: (1) every harvested checkpoint with a stored counterpart gives exactly td_tt_learnmask's per-environment goal
  counts (270 at 03:15: all 110 branch ones, 160 from scratch); (2) the workflow's per-probe replays (24 blk_*.npz in
  reviewer_lens, 15 branch_fr_*.npz in theory_designer): 0 of 20,000 probe-variables differ in each; (3) the lean
  replay equals td_tt_chaos_ent.replay to the bit (float32 logits identical on 2 checkpoints); (4) the resamples are
  blk_boot.py's, so its printed numbers are reproduced exactly (50% at 1,280: -.052 [-.073, -.031]).

RESULTS 11-09, from scratch (all 6 runs, finished 00:59-03:05; coarse grid, raw 0, 100, ..., 3,000 = 31 of their 46
  checkpoints, x = 10 (initial), 80, ..., 2,400 training episodes; `scratch` without --every adds the other 15, raw
  25-475; summary_scratch.npz; scratch_table prints every number). All from-scratch runs start from identical weights
  (checked at raw 0), so every difference is 0 at x = 10. ctrl_s0 equals
  the branch trunk up to raw 600 (same seed; accuracy .2975 at 480 in both); at 2,400 its accuracy by fifth is .791 /
  .708 / .696 / .672 / .630 (all .699). Run - ctrl_s0 at 2,400, fifths 1..5 | all (halves blocked: random, highest /
  lowest FR, most / least entangled):
    random     +.001 [-.008,+.009] +.006 [-.007,+.019] -.009 [-.022,+.004] -.004 [-.016,+.009] -.006 [-.018,+.005] | -.002
    high FR    -.003 [-.014,+.008] +.004 [-.008,+.016] -.048 [-.061,-.035] -.081 [-.097,-.066] -.301 [-.322,-.281] | -.086
    low FR     -.050 [-.065,-.036] -.038 [-.054,-.022] -.034 [-.049,-.020] -.020 [-.034,-.006] +.000 [-.012,+.012] | -.028
    most ent.  -.015 [-.029,-.003] -.004 [-.016,+.008] -.025 [-.041,-.010] -.028 [-.041,-.015] -.075 [-.092,-.060] | -.029
    least ent. -.014 [-.027,-.003] -.009 [-.022,+.004] -.022 [-.037,-.007] -.025 [-.039,-.012] -.017 [-.030,-.006] | -.018
  The most entangled half blocked from scratch costs most in the top fifth, as in the branches: against the random half,
  top -.069 [-.083, -.055], bottom -.016 [-.026, -.007], top - bottom -.052 [-.070, -.035] at 2,400 (-.067 [-.092,
  -.041] at 1,280); its top - bottom cost against the control is negative at all 22 checkpoints from 720 on (-.012 to
  -.084; -.059 at 2,400) and mixed before (+.045 to -.051). The least entangled half costs every fifth about alike
  (against the random half: bottom -.015 [-.025, -.005], top -.011 [-.023, +.001], top - bottom +.004 [-.012, +.020]).
  Entanglement-ranked double dissociation dd_ent (most - least entangled half)_top - (...)_bottom: +.064 at 80, +.054 at
  160, -.028 at 240, -.081 at 480, -.036 at 800, -.061 [-.087, -.034] at 880, -.075 at 1,120, -.090 [-.113, -.068] at
  1,280, -.095 at 1,360 (its peak), -.055 [-.077, -.034] at 1,680, -.056 [-.075, -.038] at 2,400; negative at all 22
  checkpoints from 720 on (interval below 0 at 21, not 720), mean -.055 [-.065, -.045]; before 720 mixed (6 of 8
  negative, 2 positive, mean -.017). Over the branches' window (560-1,280) its mean is -.053 [-.067, -.038], against
  the branches' dd50 mean -.057 [-.069, -.045]: the branch dissociation is reproduced by an independent run pair (other
  episode stream, blocks from episode 0), though with the most entangled block carrying it (the least entangled one is
  flat here, while in the branches it cost the low fifths most).
  FR-ranked double dissociation (highest - lowest FR half)_top - (...)_bottom: -.145 at 80, -.273 at 160, -.293 at 480,
  -.391 at 880, -.450 [-.482, -.418] at 1,280 (its peak), -.384 at 1,600, -.349 [-.378, -.318] at 2,400; the interval
  excludes 0 at all 30 checkpoints after 0. The highest-FR block's top-fifth cost grows to about -.30 by 1,360 and stays
  (-.301 at 2,400; that fifth ends at .329 against the control's .630, naive .025, Bayes .727); the lowest-FR block's
  bottom-fifth cost peaks at -.241 (720) and shrinks to -.050; the random half stays within +-.06 per fifth at every
  checkpoint (+-.01 at 2,400). The FR-ranked blocks withhold learning from the same kind of updates (high or low goal FR
  in training episodes) that the fifths sort the test probes by, so the direction is expected; its size and persistence
  are the result.

Run from the repository root (CPU only, 4 threads; a replay takes 15-40 s depending on machine load):
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
    PY=/home/johns/anaconda3/envs/eureka/bin/python; S=main/REPLICATION/claude_plots/td_tt_learnmask_fr.py
    $PY $S harvest [run ...]       # default the 10 branches; any run name learnmask_<run>_s20261029 works
    $PY $S summary                 # summary.npz
    $PY $S table                   # every number (as in RESULTS)
    $PY $S figure                  # td_tt_learnmask_fr.png
    $PY $S scratch [--every 100] [--ctrl ctrl_s0] [--out summary_scratch.npz] [run ...]   # finished runs only
    $PY $S scratch_summary         # summary_scratch.npz from what is harvested, no replays
    $PY $S scratch_table           # every from-scratch number
    $PY $S check [dir ...]         # cross-checks (dirs with earlier per-probe replays)"""
import os
import re
import sys
import glob
import time
import numpy as np
import torch
from scipy.special import log_softmax

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_intervene as ti                                                                  # noqa: E402
import td_tt_chaos_ent as ce                                                                  # noqa: E402
import td_tt_learnmask as lm                                                                  # noqa: E402  (read-only)

OUT = os.path.join(gap.TT, "learnmask_fr")
SHARES = (("_f05", 5), ("", 20), ("_f50", 50))            # run-name suffix, % of each step's updates blocked
BLOCKS = ("hi", "lo", "rand")                             # most entangled / least entangled / random share
BRANCHES = ("ctrl",) + tuple(f"det_{b}{s}" for s, _ in SHARES for b in BLOCKS)
SCRATCH = tuple(c for c, *_ in lm.SCRATCH)                # from episode 0 (td_tt_learnmask.SCRATCH)
SCRATCH_CTRL = "ctrl_s0"
SCRATCH_PAIRS = (("fr", "det_hi_fr_f50_s0", "det_lo_fr_f50_s0"), ("ent", "det_hi_f50_s0", "det_lo_f50_s0"))
X_END = 1280.                                             # the branches' last checkpoint (raw 1,600)
T, R, NV = lm.T, lm.R, lm.NV
NQ, N_BOOT, BOOT_SEED, CHUNK = 5, 2000, 0, 2500


# ═══════════════════════════════════════════════════════════════════
# Probes: forward FR_T fifths
# ═══════════════════════════════════════════════════════════════════

def fr_fifths():
    """forward FR_T = KL(Bayes_T || naive_T) of every probe-variable's marginal (n, 2), its fifth (n, 2) 0..4 (quintiles
    pooled over the 20,000 probe-variables), the fifth edges (6,) and the mean FR of each fifth (5,), in nats"""
    P = np.load(os.path.join(ti.AN, "probes.npz"))
    lj = log_softmax(P["log_jb"][:, -1].astype(np.float64), -1)
    ln = log_softmax(P["log_nb"][:, -1].astype(np.float64), -1)
    fr = (np.exp(lj) * (lj - ln)).sum(-1)
    q = np.quantile(fr.ravel(), np.linspace(0, 1, NQ + 1))
    bins = np.clip(np.searchsorted(q, fr, side="right") - 1, 0, NQ - 1)
    return fr, bins, q, np.array([fr[bins == k].mean() for k in range(NQ)])


# ═══════════════════════════════════════════════════════════════════
# Harvest (CPU replays)
# ═══════════════════════════════════════════════════════════════════

def done(run):
    """the run's progress.log has its DONE line (train_learn_masks.py writes it after the last checkpoint)"""
    f = os.path.join(lm.run_dir(run), "progress.log")
    if not os.path.exists(f):
        return False
    with open(f) as fh:
        return any(line.split()[-1:] == ["DONE"] for line in fh)


def raws(run):
    """raw episodes of the run's stored checkpoints"""
    return sorted(int(os.path.basename(f)[7:12]) for f in glob.glob(os.path.join(lm.run_dir(run), "ckpt_ep*_net.pth")))


def path(run, raw):
    return os.path.join(OUT, f"{run}_{int(raw):05d}.npz")


def save(f, arrays):
    """np.savez written atomically (readers never see a half-written file)"""
    tmp = f[:-4] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, f)


@torch.no_grad()
def answers(run, raw, dev=torch.device("cpu"), lean=True):
    """the network's log-odds at T (n, 2, 10) float64 on the 10,000 probes: the sum of its 30 per-step read-outs
    (td_tt_chaos_ent.setup / codes / replay, as td_tt_learnmask.measure). lean=True runs the same td_tt_chaos.lstm_step
    and the same read-out on the same 2,500-probe chunks but skips replay's stored pre-activations (a second h @ W_hh per
    step), so it is bit-identical and cheaper (checked: identical logits on stored checkpoints, see RESULTS). The 110
    branch checkpoints were replayed with replay itself (lean=False), the from-scratch ones with lean=True"""
    p, table = ce.setup(f"learnmask_{run}", raw, dev)
    n = len(ti.probes()["env"])
    envt, code = ce.codes(np.arange(n), dev)
    Whh = p["LSTM.weight_hh_l0"]
    H = Whh.shape[1]
    out = []
    for b0 in range(0, n, CHUNK):
        sl = slice(b0, min(n, b0 + CHUNK))
        if not lean:
            _, _, _, l_ = ce.replay(p, table, envt[sl], code[sl])
            out.append(l_.sum(1).cpu())
            continue
        e, cd = envt[sl], code[sl]
        b = len(e)
        h = p["STM"].reshape(1, H).expand(b, H).contiguous()
        c = p["LTM"].reshape(1, H).expand(b, H).contiguous()
        U = torch.empty(b, T + 1, H, device=Whh.device, dtype=Whh.dtype)
        U[:, 0] = h
        for t in range(T):
            h, c, _ = ce.tc.lstm_step(table[e, cd[:, t]], h, c, Whh)
            U[:, t + 1] = h
        l_ = torch.nn.functional.linear(U[:, 1:], p["classifier_readout.weight"], p["classifier_readout.bias"]).double()
        out.append(l_.sum(1).cpu())
    return torch.cat(out).numpy().reshape(n, NV, R)


def harvest(runs, need_done=False, every=1):
    """one file per stored checkpoint (skips existing files; written atomically): right (n, 2) bool = each variable's
    own answer at T against its own truth, logit_T (n, 2, 10) float32 = the log-odds at T. every > 1: only raw
    episodes divisible by it (e.g. 100: the coarse grid of the from-scratch runs)"""
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    P = ti.probes()
    jobs = []
    for run in runs:
        if not os.path.isdir(lm.run_dir(run)):
            print(f"{run}: no run directory", flush=True)
        elif need_done and not done(run):
            print(f"{run}: not finished (no DONE in progress.log), skipped", flush=True)
        else:
            jobs += [(run, raw) for raw in raws(run) if raw % every == 0]
    for run, raw in jobs:
        f = path(run, raw)
        if os.path.exists(f):
            continue
        t0 = time.time()
        lq = answers(run, raw)
        right = lq.argmax(-1) == P["r"]
        save(f, dict(run=run, raw=raw, x=gap.train_x(raw), right=right, logit_T=lq.astype(np.float32),
                     n_env=np.bincount(P["env"], minlength=gap.N_ENV)))
        print(f"{run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s | acc both variables "
              f"{right.mean():.4f}, goal {right[np.arange(len(right)), P['goal']].mean():.4f}", flush=True)


# ═══════════════════════════════════════════════════════════════════
# Statistics: accuracy by FR fifth, paired environment bootstrap
# ═══════════════════════════════════════════════════════════════════

def boot_weights():
    """(N_BOOT, 50): how often each test environment is drawn in each resample (50 draws with replacement; the scheme
    and seed of the workflow's blk_boot.py, so its numbers are reproduced exactly)"""
    rng = np.random.default_rng(BOOT_SEED)
    return np.stack([np.bincount(rng.integers(0, gap.N_ENV, gap.N_ENV), minlength=gap.N_ENV)
                     for _ in range(N_BOOT)]).astype(np.float64)


def files(run):
    """the run's harvested checkpoint files (exact name: det_hi_* would also match det_hi_f50_*)"""
    pat = re.compile(rf"{re.escape(run)}_\d{{5}}\.npz")
    return sorted(f for f in glob.glob(os.path.join(OUT, f"{run}_*.npz")) if pat.fullmatch(os.path.basename(f)))


def load(runs, xl=False):
    """per run: x (K,) training episodes, acc (K, 5) accuracy by FR fifth (right answers / probe-variables, pooled over
    environments), acc_b (B, K, 5) the same in each environment resample, all / all_b the same over all fifths; xl=True
    also: xl (K, 5) / xl_b (B, K, 5) the excess log loss KL(Bayes_T || q_T) by fifth. Returns (dict, fifth means, edges)"""
    fr, bins, q, mf = fr_fifths()
    P = np.load(os.path.join(ti.AN, "probes.npz"))
    env2 = np.repeat(P["env"], NV)                                       # probe-variable order of right.ravel()
    idx = env2 * NQ + bins.ravel()
    M = gap.N_ENV * NQ
    N = np.bincount(idx, minlength=M).reshape(gap.N_ENV, NQ).astype(np.float64)
    Cb = boot_weights()
    CN = Cb @ N                                                         # (B, 5)
    lj = log_softmax(P["log_jb"][:, -1].astype(np.float64), -1) if xl else None
    A = {}
    for run in runs:
        fs = files(run)
        if not fs:
            continue
        x, S, L = [], [], []
        for f in fs:
            z = np.load(f)
            x.append(float(z["x"]))
            S.append(np.bincount(idx, z["right"].ravel().astype(np.float64), M).reshape(gap.N_ENV, NQ))
            if xl:
                lq = log_softmax(z["logit_T"].astype(np.float64), -1)
                L.append(np.bincount(idx, (np.exp(lj) * (lj - lq)).sum(-1).ravel(), M).reshape(gap.N_ENV, NQ))
        S = np.array(S)                                                 # (K, 50, 5)
        a = dict(x=np.array(x), acc=S.sum(1) / N.sum(0), acc_b=np.einsum("be,kef->bkf", Cb, S) / CN[:, None],
                 all=S.sum((1, 2)) / N.sum(), all_b=(Cb @ S.sum(2).T) / CN.sum(1)[:, None])
        if xl:
            L = np.array(L)
            a.update(xl=L.sum(1) / N.sum(0), xl_b=np.einsum("be,kef->bkf", Cb, L) / CN[:, None])
        A[run] = a
    return A, mf, q


def paired(A, a, c, key="acc"):
    """run a - run c at their shared checkpoints: x (K,), full-sample (K, ...), resampled (B, K, ...); the same
    environment resamples for both runs, so the interval is paired"""
    xs = np.intersect1d(A[a]["x"], A[c]["x"])
    ia, ic = np.searchsorted(A[a]["x"], xs), np.searchsorted(A[c]["x"], xs)
    return xs, A[a][key][ia] - A[c][key][ic], A[a][key + "_b"][:, ia] - A[c][key + "_b"][:, ic]


def dissociation(A, hi, lo, key="acc"):
    """(hi - lo) in the top FR fifth minus (hi - lo) in the bottom fifth: x, full-sample (K,), resampled (B, K)"""
    xs, d, db = paired(A, hi, lo, key)
    return xs, d[:, -1] - d[:, 0], db[:, :, -1] - db[:, :, 0]


def band(b):
    """5-95% over the resamples (axis 0)"""
    return np.percentile(b, 5, 0), np.percentile(b, 95, 0)


def fmt(v, b, nd=3):
    lo, hi = band(b)
    return f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


# ═══════════════════════════════════════════════════════════════════
# Branches (restart at 480): summary.npz and tables
# ═══════════════════════════════════════════════════════════════════

def summary(write=True):
    """summary.npz for Fig. 2e: fifth_mean_fr (5,); d<share>_<block> (5,) branch - control accuracy by FR fifth at
    1,280 training episodes with d<share>_<block>_lo / _hi (5-95%), share 5 / 20 / 50, block hi / lo / rand; the
    double dissociation through training dd<share>, dd<share>_lo, dd<share>_hi (K,) at the checkpoints x (K,)"""
    A, mf, _ = load(BRANCHES)
    miss = [r for r in BRANCHES if r not in A or len(A[r]["x"]) != len(A["ctrl"]["x"])]
    if "ctrl" not in A or miss:
        print("summary.npz not written; incomplete runs:", miss or ["ctrl"])
        return None
    out, X = dict(fifth_mean_fr=mf), None
    for sfx, share in SHARES:
        for b in BLOCKS:
            xs, d, db = paired(A, f"det_{b}{sfx}", "ctrl")
            i = int(np.flatnonzero(xs == X_END)[0])
            out[f"d{share}_{b}"] = d[i]
            out[f"d{share}_{b}_lo"], out[f"d{share}_{b}_hi"] = band(db[:, i])
        xs, dd, ddb = dissociation(A, f"det_hi{sfx}", f"det_lo{sfx}")
        assert X is None or np.array_equal(X, xs)
        X = xs
        out[f"dd{share}"] = dd
        out[f"dd{share}_lo"], out[f"dd{share}_hi"] = band(ddb)
    out["x"] = X
    if write:
        save(os.path.join(OUT, "summary.npz"), out)
        print("saved", os.path.join(OUT, "summary.npz"), "keys:", ", ".join(out))
    return out


def table():
    """every number: accuracy levels, branch - control and branch - random by FR fifth, the top-minus-bottom contrasts
    and the double dissociation at every checkpoint and share (full-sample value, 5-95% paired interval; for the
    dissociation also the bootstrap median, as the workflow's blk_boot.py printed it); excess log loss at the end"""
    A, mf, q = load(BRANCHES, xl=True)
    print("FR_T fifth edges (nats):", " ".join(f"{v:.3f}" for v in q))
    print("FR_T fifth means (nats):", " ".join(f"{v:.3f}" for v in mf))
    P = np.load(os.path.join(ti.AN, "probes.npz"))
    _, bins, _, _ = fr_fifths()
    for nm, key in (("naive", "right_naive"), ("Bayes", "right_bayes")):
        r = P[key].astype(float)
        print(f"{nm} accuracy by fifth:", " ".join(f"{r[bins == k].mean():.3f}" for k in range(NQ)), f"| all {r.mean():.3f}")
    c = A["ctrl"]
    print("\ncontrol accuracy by fifth (both variables):")
    for k, x in enumerate(c["x"]):
        print(f"  {x:5.0f}: " + " ".join(f"{v:.3f}" for v in c["acc"][k]) + f" | all {c['all'][k]:.4f}")
    for sfx, share in SHARES:
        print(f"\n===== {share}% of each step's updates blocked =====")
        for b in BLOCKS:
            run = f"det_{b}{sfx}"
            if run not in A:
                continue
            xs, d, db = paired(A, run, "ctrl")
            _, da, dab = paired(A, run, "ctrl", "all")
            print(f"{run} - control (fifths 1..5 | all):")
            for k, x in enumerate(xs):
                print(f"  {x:5.0f}: " + "  ".join(fmt(d[k, f], db[:, k, f]) for f in range(NQ)) +
                      f" | {fmt(da[k], dab[:, k])}")
        for b in ("hi", "lo"):
            run, rnd = f"det_{b}{sfx}", f"det_rand{sfx}"
            if run not in A or rnd not in A:
                continue
            xs, d, db = paired(A, run, rnd)
            print(f"{run} - {rnd} (fifths 1..5 | top - bottom):")
            for k, x in enumerate(xs):
                print(f"  {x:5.0f}: " + "  ".join(fmt(d[k, f], db[:, k, f]) for f in range(NQ)) +
                      f" | {fmt(d[k, -1] - d[k, 0], db[:, k, -1] - db[:, k, 0])}")
        if f"det_hi{sfx}" in A and f"det_lo{sfx}" in A:
            xs, dd, ddb = dissociation(A, f"det_hi{sfx}", f"det_lo{sfx}")
            print(f"double dissociation (most - least)_top - (most - least)_bottom, {share}%:")
            for k, x in enumerate(xs):
                print(f"  {x:5.0f}: {fmt(dd[k], ddb[:, k])} (bootstrap median {np.median(ddb[:, k]):+.3f})")
    print("\nexcess log loss KL(Bayes_T || q_T) by fifth at 1,280 (nats): control, then branch - control")
    i = int(np.flatnonzero(c["x"] == X_END)[0])
    print("  ctrl: " + " ".join(f"{v:.3f}" for v in c["xl"][i]))
    for run in (r for r in BRANCHES[1:] if r in A):
        xs, d, db = paired(A, run, "ctrl", "xl")
        j = int(np.flatnonzero(xs == X_END)[0])
        print(f"  {run:13s} " + "  ".join(fmt(d[j, f], db[:, j, f], 2) for f in range(NQ)))


def check(*ref_dirs):
    """cross-checks: (1) every harvested checkpoint's goal answers, counted per environment, against td_tt_learnmask's
    stored per-environment counts (through_training/learnmask/<run>_<raw>.npz 'correct'); (2) per probe-variable against
    other replays in ref_dirs: the workflow's blk_<cond>_<raw>.npz ('right' (n, 2)) and branch_fr_<raw>_*.npz (one
    (n, 2) array per condition)"""
    P = ti.probes()
    g = np.arange(len(P["goal"])), P["goal"]
    n_ok = n_bad = 0
    for f in sorted(glob.glob(os.path.join(OUT, "*_?????.npz"))):
        ref = os.path.join(lm.OUT, os.path.basename(f))
        if os.path.exists(ref):
            same = np.array_equal(np.bincount(P["env"], np.load(f)["right"][g], gap.N_ENV), np.load(ref)["correct"])
            n_ok, n_bad = n_ok + same, n_bad + (not same)
    print(f"goal answers per environment vs td_tt_learnmask's stored counts: {n_ok} checkpoints identical, {n_bad} differ")
    n_files = n_diff = 0
    for d in ref_dirs:
        refs = [(f, os.path.basename(f)[4:-4].rsplit("_", 1), None) for f in glob.glob(os.path.join(d, "blk_*_*.npz"))]
        for f in glob.glob(os.path.join(d, "branch_fr_*.npz")):
            raw = os.path.basename(f)[:-4].split("_")[2]
            refs += [(f, (k, raw), k) for k in np.load(f).files if k != "bins"]
        for f, (cond, raw), key in sorted(refs):
            mine = path(cond, int(raw))
            if not os.path.exists(mine):
                print(f"  {os.path.basename(f)} {cond}: not harvested")
                continue
            dif = int((np.load(f)[key or "right"].astype(bool) != np.load(mine)["right"]).sum())
            n_files, n_diff = n_files + 1, n_diff + dif
            print(f"  {os.path.basename(f)} {cond} {raw}: {dif} of 20,000 probe-variables differ")
    print(f"{n_files} replays compared per probe-variable, {n_diff} differing probe-variables in total")


# ═══════════════════════════════════════════════════════════════════
# From-scratch runs: summary_scratch.npz
# ═══════════════════════════════════════════════════════════════════

def scratch(runs=SCRATCH, ctrl=SCRATCH_CTRL, pairs=SCRATCH_PAIRS, name="summary_scratch.npz", do_harvest=True,
            every=1):
    """the finished from-scratch runs (DONE in progress.log), harvested at every stored checkpoint and compared with
    the control: fifth_mean_fr (5,), x (K,) the checkpoints all used runs share, runs; acc_<run> (K, 5) accuracy by FR
    fifth; d_<run>, d_<run>_lo, d_<run>_hi (K, 5) run - control; dd_<sel>, dd_<sel>_lo, dd_<sel>_hi (K,) the double
    dissociation of the most / least pair ranked by FR (sel 'fr') or by entanglement ('ent'), when both are finished.
    do_harvest=False: summarize what is harvested already"""
    if ctrl not in runs:
        runs = (ctrl,) + tuple(runs)
    fin = []
    for r in runs:
        d = lm.run_dir(r)
        if not os.path.isdir(d):
            print(f"{r}: not started")
        elif not done(r):
            ks = raws(r)
            print(f"{r}: running, {len(ks)} checkpoints so far (last raw {ks[-1] if ks else '-'}), not used")
        else:
            fin.append(r)
            print(f"{r}: finished, {len(raws(r))} checkpoints")
    if ctrl not in fin:
        print(f"{name} not written: the control {ctrl} is not finished")
        return None
    if do_harvest:
        harvest(fin, need_done=True, every=every)
    A, mf, _ = load(fin)
    fin = [r for r in fin if r in A]
    if ctrl not in A or len(fin) < 2:
        print(f"{name} not written: nothing harvested to compare with {ctrl}")
        return None
    X = A[ctrl]["x"]
    for r in fin:
        X = np.intersect1d(X, A[r]["x"])
    out = dict(fifth_mean_fr=mf, x=X, runs=np.array(fin))
    for r in fin:
        out[f"acc_{r}"] = A[r]["acc"][np.searchsorted(A[r]["x"], X)]
        if r == ctrl:
            continue
        xs, d, db = paired(A, r, ctrl)
        i = np.searchsorted(xs, X)
        out[f"d_{r}"] = d[i]
        out[f"d_{r}_lo"], out[f"d_{r}_hi"] = band(db[:, i])
        print(f"{r} - {ctrl} at {X[-1]:.0f} by fifth: " + "  ".join(fmt(d[i[-1], f], db[:, i[-1], f])
                                                                    for f in range(NQ)))
    for sel, hi, lo in pairs:
        if hi in fin and lo in fin:
            xs, dd, ddb = dissociation(A, hi, lo)
            i = np.searchsorted(xs, X)
            out[f"dd_{sel}"] = dd[i]
            out[f"dd_{sel}_lo"], out[f"dd_{sel}_hi"] = band(ddb[:, i])
            print(f"double dissociation ({sel}): " + "  ".join(f"{x:.0f}: {fmt(dd[k], ddb[:, k])}"
                                                               for x, k in zip(X, i)))
    save(os.path.join(OUT, name), out)
    print("saved", os.path.join(OUT, name), "keys:", ", ".join(out))
    return out


# ═══════════════════════════════════════════════════════════════════
# Figure (diagnostic; Fig. 2e itself is drawn by the main-figure script from summary.npz)
# ═══════════════════════════════════════════════════════════════════

BLOCK_STYLE = (("hi", "#d62728", "most entangled"), ("lo", "#1f77b4", "least entangled"), ("rand", "0.5", "random"))
SHARE_LW = {5: 1.1, 20: 1.8, 50: 2.8}
SHARE_COL = {5: "0.68", 20: "0.42", 50: "k"}
DASH = (0, (3, 1.5))
SCRATCH_STYLE = {"det_rand_f50_s0": ("0.5", "-", "random half"),               # td_tt_learnmask.SCRATCH's styles
                 "det_hi_fr_f50_s0": ("#d62728", "-", "highest-FR half"),
                 "det_lo_fr_f50_s0": ("#1f77b4", "-", "lowest-FR half"),
                 "det_hi_f50_s0": ("#d62728", DASH, "most entangled half"),
                 "det_lo_f50_s0": ("#1f77b4", DASH, "least entangled half")}


def figure(S=None, Z=None):
    """a-c: branch - control accuracy by FR_T fifth at 1,280 training episodes, 50 / 20 / 5% of each step's updates
    blocked; d: the double dissociation through the branches (480 -> 1,280) for the three shares. When
    summary_scratch.npz exists, a second row from scratch: e run - control by fifth at the last shared checkpoint;
    f, g the highest- / lowest-FR half blocked, per fifth through training; h the double dissociations of the FR- and
    the entanglement-ranked pairs (whichever runs are finished)"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import fig_path, apply_fig_style
    import fig_entanglement_driver as fd
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    TS, FL, FT = 13, 11.5, 10.5
    plt.rcParams.update({"xtick.labelsize": FT, "ytick.labelsize": FT})
    fs = os.path.join(OUT, "summary.npz")
    S = S or (dict(np.load(fs)) if os.path.exists(fs) else summary(write=False))
    fz = os.path.join(OUT, "summary_scratch.npz")
    Z = Z or (dict(np.load(fz)) if os.path.exists(fz) else None)
    cross = fd.passes_naive(dict(np.load(os.path.join(ti.AN, "probes.npz"))))
    nrow = 1 if Z is None else 2
    PW, PH, GX, LM_, TOP, GY, BOT = 3.0, 2.5, 1.0, .95, .82, 2.0, (1.45 if nrow == 1 else 1.95)
    W, H = LM_ + 4 * PW + 3 * GX + .85, TOP + nrow * PH + (nrow - 1) * GY + BOT     # right margin: d's formula title
    fig = plt.figure(figsize=(W, H))
    ax = [[fig.add_axes([(LM_ + j * (PW + GX)) / W, (BOT + (nrow - 1 - i) * (PH + GY)) / H, PW / W, PH / H])
           for j in range(4)] for i in range(nrow)]
    mf = S["fifth_mean_fr"]

    def head(a, letter, name, formula):
        a.set_title(f"{letter}   {name}\n{formula}", loc="left", fontsize=TS, linespacing=1.3)

    def fifth_axis(a):
        a.set_xscale("log")
        a.set_xticks(mf)
        a.set_xticklabels([f"{v:.2f}".lstrip("0") if v < 1 else f"{v:.2g}" for v in mf])
        a.minorticks_off()
        a.set_xlim(mf[0] / 1.6, mf[-1] * 1.6)
        a.set_xlabel(r"FR$_T$ fifth (mean, nats)", fontsize=FL)

    def train_axis(a):
        a.set_xscale("log")
        a.set_xlim(8, 3200)
        a.set_xticks([10, 100, 1000])
        a.set_xticklabels(["0", "100", "1k"])
        a.minorticks_off()
        a.set_xlabel("training episodes", fontsize=FL)
        a.axvline(cross, color="0.55", lw=1, ls=":", zorder=5)

    def ylims(axs, lims):
        pad = .06 * (max(lims) - min(lims))
        for a in axs:
            a.set_ylim(min(lims) - pad, max(lims) + pad)

    acc = r"$P(\hat r_T{=}r)_{\rm %s} - P(\hat r_T{=}r)_{\rm control}$"
    lims = []
    for a, share, letter in zip(ax[0][:3], (50, 20, 5), "abc"):
        a.axhline(0, color="k", lw=.8)
        for b, col, _ in BLOCK_STYLE:
            y, lo, hi = S[f"d{share}_{b}"], S[f"d{share}_{b}_lo"], S[f"d{share}_{b}_hi"]
            a.fill_between(mf, lo, hi, color=col, alpha=.14, lw=0)
            a.plot(mf, y, color=col, lw=2, marker="o", ms=4)
            lims += [lo.min(), hi.max()]
        fifth_axis(a)
        head(a, letter, f"accuracy change, {share}% blocked", acc % "branch")
    ylims(ax[0][:3], lims)
    ax[0][0].set_ylabel("accuracy at $T$, branch $-$ control", fontsize=FL)
    a = ax[0][3]
    a.axhline(0, color="k", lw=.8)
    for share in (5, 20, 50):
        col = SHARE_COL[share]
        a.fill_between(S["x"], S[f"dd{share}_lo"], S[f"dd{share}_hi"], color=col, alpha=.16, lw=0)
        a.plot(S["x"], S[f"dd{share}"], color=col, lw=SHARE_LW[share], marker="o", ms=2.5 + share / 25)
    a.axvline(cross, color="0.55", lw=1, ls=":", zorder=5)
    a.set_xlim(450, 1310)
    a.set_xticks([480, 800, 1120, 1280])
    a.set_xticklabels(["480", "800", "1,120", "1,280"])
    a.set_xlabel("training episodes (branch point 480)", fontsize=FL)
    a.set_ylabel("most $-$ least: top $-$ bottom fifth", fontsize=FL)
    head(a, "d", "double dissociation",
         r"$(\Delta_{\rm most}-\Delta_{\rm least})_{\rm top}-(\Delta_{\rm most}-\Delta_{\rm least})_{\rm bottom}$")
    h = [Line2D([], [], color=col, lw=2.4, marker="o", ms=4, label=f"{nm} blocked") for _, col, nm in BLOCK_STYLE]
    h += [Line2D([], [], color=SHARE_COL[s], lw=SHARE_LW[s] + .4, label=f"{s}% blocked (d)") for s in (50, 20, 5)]
    h += [Line2D([], [], color="0.55", lw=1, ls=":", label=f"Trained RNN passes naive ({cross:,.0f})")]
    y0 = (BOT + (nrow - 1) * (PH + GY) - .98) / H
    fig.legend(handles=h, loc="lower center", ncol=7, frameon=False, fontsize=FT + .5, bbox_to_anchor=(.5, y0),
               columnspacing=1.3, handlelength=1.8)
    if Z is not None:
        runs = [r for r in Z["runs"] if r in SCRATCH_STYLE]
        x, k = Z["x"], -1
        pur = plt.cm.Purples(np.linspace(.35, .95, NQ))
        a = ax[1][0]
        a.axhline(0, color="k", lw=.8)
        lims = []
        for r in runs:
            col, ls, _ = SCRATCH_STYLE[r]
            a.fill_between(mf, Z[f"d_{r}_lo"][k], Z[f"d_{r}_hi"][k], color=col, alpha=.12, lw=0)
            a.plot(mf, Z[f"d_{r}"][k], color=col, ls=ls, lw=2, marker="o", ms=4)
            lims += [Z[f"d_{r}_lo"][k].min(), Z[f"d_{r}_hi"][k].max()]
        fifth_axis(a)
        head(a, "e", f"from scratch, at {x[k]:,.0f} episodes", acc % "run")
        a.set_ylabel("accuracy at $T$, run $-$ control", fontsize=FL)
        for a, r, letter, nm in ((ax[1][1], "det_hi_fr_f50_s0", "f", "highest-FR half blocked"),
                                 (ax[1][2], "det_lo_fr_f50_s0", "g", "lowest-FR half blocked")):
            a.axhline(0, color="k", lw=.8)
            if r in runs:
                for f in range(NQ):
                    a.fill_between(x, Z[f"d_{r}_lo"][:, f], Z[f"d_{r}_hi"][:, f], color=pur[f], alpha=.18, lw=0)
                    a.plot(x, Z[f"d_{r}"][:, f], color=pur[f], lw=1.8)
                lims += [Z[f"d_{r}_lo"].min(), Z[f"d_{r}_hi"].max()]
            train_axis(a)
            head(a, letter, f"{nm}, by fifth", acc % "run")
        ylims(ax[1][:3], lims)
        a = ax[1][3]
        a.axhline(0, color="k", lw=.8)
        for sel, ls in (("fr", "-"), ("ent", DASH)):
            if f"dd_{sel}" in Z:
                a.fill_between(x, Z[f"dd_{sel}_lo"], Z[f"dd_{sel}_hi"], color="k", alpha=.14, lw=0)
                a.plot(x, Z[f"dd_{sel}"], color="k", ls=ls, lw=2)
        train_axis(a)
        a.set_ylabel("high $-$ low half: top $-$ bottom fifth", fontsize=FL)
        head(a, "h", "double dissociation from scratch",
             r"$(\Delta_{\rm high}-\Delta_{\rm low})_{\rm top}-(\Delta_{\rm high}-\Delta_{\rm low})_{\rm bottom}$")
        top = [Line2D([], [], color=c, ls=ls, lw=2.4, label=nm + ("" if r in runs else " (pending)"))
               for r, (c, ls, nm) in SCRATCH_STYLE.items()]
        top += [Line2D([], [], color="k", lw=2.4, label="FR-ranked pair (h)")]
        bot = [Line2D([], [], color=pur[f], lw=3, label=f"fifth {f + 1} (f, g)") for f in range(NQ)]
        bot += [Line2D([], [], color="k", ls=DASH, lw=2.4, label="entanglement-ranked pair (h)" +
                       ("" if "dd_ent" in Z else " (pending)"))]
        fig.legend(handles=[v for pair in zip(top, bot) for v in pair], loc="lower center", ncol=6, frameon=False,
                   fontsize=FT + .5, bbox_to_anchor=(.5, .42 / H), columnspacing=1.3, handlelength=1.8)
    for a in (aa for row in ax for aa in row):
        a.spines[["top", "right"]].set_visible(False)
    counts = "a-d: 11 checkpoints" + ("" if Z is None else f", e-h: {len(Z['x'])}") + \
        "; 50 environments x 200 probes x 2 variables each; one run per condition; bands: paired 5-95% over 2,000 " \
        "environment resamples"
    fig.text(.5, .08 / H, counts, ha="center", va="bottom", fontsize=FT, color="0.3")
    path = fig_path("td_tt_learnmask_fr.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def scratch_table(runs=SCRATCH, ctrl=SCRATCH_CTRL, pairs=SCRATCH_PAIRS):
    """every from-scratch number at the harvested checkpoints the finished runs share: accuracy by FR fifth, run -
    control by fifth and over all fifths, and the double dissociations"""
    fin = [r for r in ((ctrl,) + tuple(r for r in runs if r != ctrl)) if done(r)]
    A, mf, _ = load(fin)
    fin = [r for r in fin if r in A]
    if ctrl not in A:
        print(f"no harvested checkpoints of {ctrl}")
        return
    X = A[ctrl]["x"]
    for r in fin:
        X = np.intersect1d(X, A[r]["x"])
    print("runs:", ", ".join(fin), f"| {len(X)} shared checkpoints, x = {X[0]:.0f} .. {X[-1]:.0f}")
    for r in fin:
        i = np.searchsorted(A[r]["x"], X)
        print(f"\n{r} accuracy by fifth (both variables) | all:")
        for x, a, al in zip(X, A[r]["acc"][i], A[r]["all"][i]):
            print(f"  {x:5.0f}: " + " ".join(f"{v:.3f}" for v in a) + f" | {al:.4f}")
    for r in fin:
        if r == ctrl:
            continue
        xs, d, db = paired(A, r, ctrl)
        _, da, dab = paired(A, r, ctrl, "all")
        print(f"\n{r} - {ctrl} (fifths 1..5 | all):")
        for k in np.searchsorted(xs, X):
            print(f"  {xs[k]:5.0f}: " + "  ".join(fmt(d[k, f], db[:, k, f]) for f in range(NQ)) +
                  f" | {fmt(da[k], dab[:, k])}")
    for sel, hi, lo in pairs:
        if hi in fin and lo in fin:
            xs, dd, ddb = dissociation(A, hi, lo)
            print(f"\ndouble dissociation ({sel}: {hi} vs {lo}):")
            for k in np.searchsorted(xs, X):
                print(f"  {xs[k]:5.0f}: {fmt(dd[k], ddb[:, k])}")


if __name__ == "__main__":
    cmd, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("summary", [])
    if cmd == "harvest":                       # any run names (learnmask_<run>_s20261029)
        harvest(tuple(args) or BRANCHES)
    elif cmd == "summary":
        summary()
    elif cmd == "table":
        table()
    elif cmd == "check":                       # [dir ...] with earlier per-probe replays (blk_*.npz, branch_fr_*.npz)
        check(*args)
    elif cmd == "figure":
        figure()
    elif cmd == "scratch_table":
        scratch_table()
    elif cmd in ("scratch", "scratch_summary"):  # [--ctrl NAME] [--out FILE] [--every N] [run ...]; *_summary: no harvest
        kw = {}
        while args and args[0] in ("--ctrl", "--out", "--every"):
            k, v = {"--ctrl": "ctrl", "--out": "name", "--every": "every"}[args[0]], args[1]
            kw[k] = int(v) if k == "every" else v
            args = args[2:]
        if scratch(tuple(args) or SCRATCH, do_harvest=cmd == "scratch", **kw) is not None and "name" not in kw:
            figure()                           # the figure's second row reads summary_scratch.npz
