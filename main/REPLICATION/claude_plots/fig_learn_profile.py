"""fig_learn_profile.png: where every network is right, state by state, sorted by the state's Factorization Regret (FR).
One grammar in all eight panels: x = the state's FR_T = KL(Bayes_T || naive_T) of a variable's marginal belief at the last
step (forward KL, the paper's definition; tenths of the 20,000 probe states, drawn at their mean FR), y = accuracy at the
last step, thin black lines = Bayes (solid) and naive Bayes (dashed).

  a-c  Trained RNN, ESN, ESN + MLPs (the ESN's frozen LSTM between 5 trained layers before and 5 after it) at nine
       stages of training (colour).
  d    the three at the end of training (the Trained RNN's line is drawn wide, under Bayes's: they coincide; ESN +
       MLPs dashed).
  e-g  Trained RNN trained from initialization with the gradient through the goal read-out increment blocked at half of
       every step's updates of every batch: a random half, the half with the highest per-step goal FR_t, the half with
       the lowest; six stages to 2,400 training episodes (the runs' end; same colours as a-c).
  h    the three at 2,400 episodes over the unblocked network (wide pale band; the random half is dashed, it lies on
       that band), and (open circles) the same three blocks repeated in a second world (new variables' embeddings,
       initialization and episode stream; its own probe set and FR tenths).

RESULTS (accuracy at T; [95% interval over 2,000 resamples of the 50 test environments]; fig_learn_profile.py --numbers)
  FR sorts the states by where the factorized answer fails. Lowest -> highest FR tenth: naive .866 -> .015, Bayes .865
  -> .718 (Bayes's answer is naive's answer in 97.5% of the lowest-tenth states and 0.6% of the highest).
  ESN: naive's falling profile, .759 -> .045 [.029, .062] (naive .015); its highest tenth is at or below chance (.1)
  throughout training (.092 at 80 episodes, .071 at 880, .035 at 2,400, .028 at 9,600, .045 at the end) while the
  share of its answers there that are naive's (wrong) answer grows (.11 at 240, .23 at 880, .45 at 2,400, .58 at the
  end). It is not naive: it is .11 below naive at the lowest tenth and .10-.17 above it in tenths 7-9; all states
  .428 vs naive .460.
  Trained RNN: Bayes's flat profile, .861 -> .711 [.669, .749] (minus Bayes -.007 [-.018, +.005] at the highest tenth);
  0.9% of its answers there are naive's. ESN + MLPs: .845 -> .671 (minus Bayes -.046 [-.063, -.031]).
  No naive stage: at 880 training episodes, where the Trained RNN's overall accuracy passes naive's, it is .15-.19 below
  naive in the five lowest tenths and .22-.33 above it in the four highest; its highest tenth never falls below chance.
  Blocks at 2,400 training episodes, highest tenth: no block .622, random half .607, lowest-FR half .620, highest-FR
  half .220 [.178, .262] (minus random -.387 [-.424, -.348]); naive .015, Bayes .718. The highest-FR-blocked network's
  deficit against the random block grows with FR: -.01, .00, .00, -.01, -.03, -.05, -.05, -.11, -.20, -.39. A third of
  its answers at the highest tenth are naive's answer (33.5%; no block 2.5%, random 2.4%, lowest-FR 1.9%), from 240
  episodes on (32.7%, 34.5% at 880) while its accuracy there creeps up (.068, .101, .220).
  Mirror image: the lowest-FR-blocked network rises with a flat profile (880 episodes: .478 -> .323) and at 2,400 equals
  the unblocked network at the highest tenth (.620 vs .622) and is .05 below at the lowest (.783 vs .828; minus random
  -.048 [-.072, -.027]).
  The asymmetry at matched coverage: the highest-FR half holds 97% of the 30 updates of a highest-tenth state, the
  lowest-FR half 90% of those of a lowest-tenth state (per-step ranking over the probe set); the first block costs .39
  there, the second .05.
  Second world (other embeddings, initialization, episodes; no unblocked run), highest tenth at 2,400: random .645,
  lowest-FR .649, highest-FR .340 (minus random -.305); lowest tenth .819 / .759 / .831. Same order and shape; naive's
  answer is 20.3% of the highest-FR-blocked network's answers at the highest tenth (random 1.5%, lowest-FR 1.3%).
  Entanglement-ranked blocks (not drawn) are five times weaker at the highest FR tenth: most-entangled half -.072
  [-.095, -.049] against random, least-entangled +.006.

ROBUSTNESS (other runs on the same probe set, not drawn; accuracy over all states | lowest -> highest FR tenth | share
of naive's answer at the highest tenth; end of training unless noted; fr_profile/checks/all_runs_profile.txt,
esn_width_profile.txt, wide_esn_profile.log)
  Trained RNN, 1,000 units .761 | .861 -> .711 | .009;  loss power 1 .759 | .858 -> .698 | .009;  200 units .718 |
  .837 -> .638 | .019;  100 units .632 | .787 -> .508 | .061;  50 units .507 | .716 -> .307 | .099. None passes
  through naive's profile: naive's answer is at most 24% of any of these networks' answers at the highest tenth at
  240 / 880 / 2,400 / 9,600 episodes, and every one is below naive at the lowest tenth and above it at the highest.
  ESN, 200 units (36,000 episodes) .353 | .678 -> .034 | .489;  500 units .406 | .740 -> .036 | .581;  1,000 units
  .428 | .759 -> .045 | .581;  10,000 units (echo_N10000_b8k at 28,000 training episodes, still training; 5,000 of
  the probe episodes) .472 | .779 -> .092 | .500 (naive .457, Bayes .770 on those episodes). Width lifts the ESN's
  middle tenths, not its highest: a 50-unit Trained RNN (.307) is right there three times as often as the 10,000-unit
  ESN (.092).
  Joint output (100 logits): Trained RNN .686 | .748 -> .651 | .006;  ESN with the goal loss .428 | .651 -> .207 |
  .108;  ESN with the joint loss .650 | .785 -> .577 | .033 (a frozen reservoir can end on a flat profile).

REFERENCE RULES (not drawn; accuracy at T on the probe set, lowest -> highest FR tenth | all states. The fitted rules
have their own table per environment, variable and value, fitted on 20,000 simulated episodes per environment and
scored on the probe set (accuracy on the fitting episodes within .004 of the probe set's): each shows that a rule of
its family at least this accurate exists, never an upper bound. fr_profile/checks/additive_fit.txt, ladder_*.txt)
  linear in the 5 bit counts:      naive Bayes .866 -> .015 | .460;  fitted for accuracy .826 -> .173 | .560 (for
                                   cross-entropy to Bayes .805 -> .148 | .513)
  additive through time (linear in the 32 pattern counts: any function of one observation, summed over steps):
                                   Bayes without memory (each observation's 5 bits combined jointly, a uniform belief
                                   about the other variable before every observation) .834 -> .303 | .607;  fitted for
                                   cross-entropy .818 -> .600 | .671 (for accuracy .820 -> .565 | .661)
  Bayes over the last k observations only, k = 2 / 3 / 5 / 10: highest tenth .625 / .676 / .700 / .706, all states
                                   .697 / .724 / .741 / .758
  Bayes                            .865 -> .718 | .770
  Share of a network's answers at the highest tenth equal to a rule's (naive / Bayes without memory / fitted additive
  through time / Bayes): ESN at the end .581 / .377 / .069 / .037;  Trained RNN at the end .009 / .326 / .675 / .924;
  highest-FR-blocked at 2,400 .335 / .475 / .265 / .232;  no block at 2,400 .025 / .339 / .600 / .729.
  The ESN (.045 at the highest tenth, .428 over all states) ends below the fitted rules of both additive families.

READING
  FR is the forward half of what the training loss charges naive's belief at a state (the loss is the square root of
  a symmetric KL to Bayes's goal belief at every step), so only high-FR states tell the joint computation from the
  factorized one. Learning from them transfers down (the lowest-FR-blocked network loses .05 at the lowest tenth),
  learning from low-FR states transfers up only weakly (the highest-FR-blocked network keeps a falling profile: .22 at
  the highest tenth against the unblocked .62 and naive's .015, with naive's answer in a third of its states there).
  FR is a property of states, not of environments: environment means carry 3% of the variance of FR_T (cells within
  an environment 42%).

NOT CLAIMED
  - that the block prevents learning: the highest tenth of the highest-FR-blocked network still rises (.07 -> .22 by
    2,400; the unblocked network was there by ~700 episodes); runs longer than 2,400 episodes do not exist.
  - a dose: one block fraction (50%) only. The FR-ranked block withholds learning from the kind of updates the tenths
    sort the probes by, so the direction is expected; its size, its persistence, the missing transfer from low to high
    FR, and the three controls (random half, lowest-FR half, entanglement-ranked halves) are the result.
  - that the ESN is naive Bayes or converges onto it (see above), or that recurrent training is required (ESN + MLPs).
  - why the ESN ends on naive's profile. Measured outcomes only, no "cannot": not its width (200 to 10,000 units), not
    the frozen reservoir as such (ESN + MLPs; the joint loss), and not a ceiling of rules that add evidence up through
    time (a fitted one reaches .60 at the highest tenth, .67 over all states; the ESN .045, .428).
  - that accuracy at high FR requires memory (same fitted rule), or that the Trained RNN climbs through the rules with
    a short memory on its way: at 2,400 episodes its profile lies on the two-observation rule's (mean distance .007),
    but its answers at the highest tenth agree with Bayes's more than with that rule's (.71 vs .65; ladder_stages.txt).
  - that the highest-FR-blocked network is Bayes without memory: at 2,400 episodes its profile lies on that rule's
    (mean distance over the tenths .025; second world .032) but under half of its answers at the highest tenth are
    that rule's (.475; second world .489) and its accuracy there is still rising.
  - one run per panel in the first world (seed 20261029); the second world repeats the three blocked runs only.

Data: analysis/<run>/ck_*.npz (td_analysis), through_training/learnmask_fr/ (td_tt_learnmask_fr), both read only; the
second world's probe set and replays are built here (--data: CPU, ~6 min) into through_training/fr_profile/.
usage:  python fig_learn_profile.py [--data] [--numbers]"""
import os
import sys
import time
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.colors import LogNorm                                             # noqa: E402
from matplotlib.cm import ScalarMappable                                          # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from scipy.special import log_softmax                                             # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
BLK_DIR = os.path.join(TD, "through_training", "learnmask_fr")                   # td_tt_learnmask_fr replays (read only)
OUT = os.path.join(TD, "through_training", "fr_profile")
W2_DIR = os.path.join(OUT, "world2")
CACHE = os.path.join(OUT, "profile.npz")

T, NO, R, N_ENV, NQ = 30, 5, 10, 50, 10
N_BOOT, BOOT_SEED = 2000, 0                                                       # td_tt_learnmask_fr.boot_weights
NETS = (("trained", "Trained RNN", "#1f77b4"), ("echo", "ESN", "#ff7f0e"), ("mlp55", "ESN + MLPs", "#8c564b"))
BLOCKS = (("det_rand_f50_s0", "random half", "0.45"), ("det_hi_fr_f50_s0", "highest FR", "#d62728"),
          ("det_lo_fr_f50_s0", "lowest FR", "#2ca02c"))
CTRL = ("ctrl_s0", "no block", "#1f77b4")
OTHER = ("det_hi_f50_s0", "det_lo_f50_s0")                                        # entanglement-ranked halves (numbers only)
RAW_NET = (100, 300, 600, 1100, 1600, 3000, 6000, 12000, 49995)                   # raw checkpoint episodes
RAW_BLK = (100, 300, 600, 1100, 1600, 3000)
W2_RUNS = tuple(f"learnmask_{b[0]}_r2" for b in BLOCKS)                           # train_learn_masks.py, rep 2
W2_RAW = (100, 300, 600, 1100, 1600, 2000, 2500, 3000)
W2_SEED = 20262029                                                                # the rep-2 runs' seed (run.json)
TS, FL, FT = 10.5, 10, 9.5                   # title, axis label, tick / annotation sizes (points, at print size)
FIG_W, DPI = 5.5, 300                        # print width (text width), inches
CMAP = plt.cm.viridis
NORM = LogNorm(60, 60000)                    # stage colours: 80 episodes dark ... 39,996 bright (short of pale yellow)


def train_x(raw):
    """training episodes of a raw checkpoint episode (every 5th episode is a test episode)"""
    return raw - raw // 5


def forward_fr(log_jb, log_nb):
    """FR = KL(Bayes || naive) of a variable's marginal belief, nats; (..., 10) log beliefs -> (...)"""
    lj = log_softmax(np.asarray(log_jb, np.float64), -1)
    ln = log_softmax(np.asarray(log_nb, np.float64), -1)
    return (np.exp(lj) * (lj - ln)).sum(-1)


def tenths(fr):
    """the tenth 0..9 of every state (quantiles pooled over all states) and the mean FR of each tenth"""
    q = np.quantile(fr.ravel(), np.linspace(0, 1, NQ + 1))
    b = np.clip(np.searchsorted(q, fr, side="right") - 1, 0, NQ - 1)
    return b, np.array([fr[b == k].mean() for k in range(NQ)])


# ── data ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def w2_probes():
    """a probe set in the second world, built as td_analysis.probes(): 50 environments of the test variables x 100 cells
    x 2 streams of 30 steps, the exact Bayes and naive beliefs at T"""
    import torch
    import td_analysis as ta
    path = os.path.join(W2_DIR, "probes_w2.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    os.makedirs(W2_DIR, exist_ok=True)
    cks = [torch.load(ta.ckpt_path(run, 0), map_location="cpu", weights_only=False) for run in W2_RUNS]
    P1 = np.load(os.path.join(AN, "probes.npz"))
    for k in ("K", "Q"):
        a = [np.asarray(c[k]) for c in cks]
        assert all(np.array_equal(a[0], b) for b in a[1:]), f"the second-world runs disagree on {k}"
        assert not np.array_equal(a[0], P1["world_" + k]), f"the second world's {k} equals the first world's"
    pairs_all = np.array([(a, b) for a in range(ta.N_TEST_VARS) for b in range(a + 1, ta.N_TEST_VARS)])
    L_all, Z_all = ta.world_tables(cks[0], pairs_all)
    rng = np.random.default_rng(W2_SEED)
    pick = np.sort(rng.choice(len(pairs_all), N_ENV, replace=False))
    env = np.repeat(np.arange(N_ENV), R * R * 2)
    cell = np.tile(np.repeat(np.arange(R * R), 2), N_ENV)
    r = np.stack([cell // R, cell % R], 1)
    L = L_all[pick][env]
    obs = rng.random((len(env), T, NO)) < L[np.arange(len(env)), :, r[:, 0], r[:, 1]][:, None, :]
    B = ta.bayes_side(obs, L, r)
    P = dict(pairs=pairs_all[pick], Z=Z_all[pick], env=env, cell=cell, r=r, goal=np.tile(np.arange(2), N_ENV * R * R),
             obs=obs, log_jb_T=B["log_jb"][:, -1], log_nb_T=B["log_nb"][:, -1], right_bayes=B["right_bayes"],
             right_naive=B["right_naive"], right_window1=B["right_window1"])
    np.savez(path, **P)
    return P


def w2_replay(run, raw, P):
    """the network's summed read-out at T on the second world's probes, (n, 2, 10); CPU"""
    import torch
    import td_tt_intervene as ti
    import td_tt_chaos as tc
    with torch.no_grad():
        w = ti.load(run, raw)
        table = ti._table(w, P["Z"], torch.device("cpu"))
        code = torch.as_tensor((P["obs"] * (1 << np.arange(NO))).sum(-1).astype(np.int64))
        envt = torch.as_tensor(P["env"].astype(np.int64))
        Whh, H, b = w["LSTM.weight_hh_l0"], w["LSTM.weight_hh_l0"].shape[1], len(envt)
        h = w["STM"].reshape(1, H).expand(b, H).contiguous()
        c = w["LTM"].reshape(1, H).expand(b, H).contiguous()
        lg = torch.zeros(b, 2 * R, dtype=torch.float64)
        for t in range(T):
            h, c, _ = tc.lstm_step(table[envt, code[:, t]], h, c, Whh)
            lg += (h @ w["classifier_readout.weight"].T + w["classifier_readout.bias"]).double()
    return lg.numpy().reshape(b, 2, R)


def build():
    """everything the figure and --numbers need, in one file: answers at T of every network and stage on its probe set"""
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    os.makedirs(OUT, exist_ok=True)
    P = np.load(os.path.join(AN, "probes.npz"))
    lj, ln = P["log_jb"], P["log_nb"]
    fr_t = forward_fr(lj, ln)                                                     # (n, 30, 2)
    med = np.median(fr_t.transpose(1, 0, 2).reshape(T, -1), 1)                    # per step, probes and variables pooled
    Z = dict(w1_fr=fr_t[:, -1], w1_env=P["env"], w1_cell=P["cell"], w1_r=P["r"], w1_naive=ln[:, -1].argmax(-1),
             w1_bayes=lj[:, -1].argmax(-1), w1_cover_hi=(fr_t > med[None, :, None]).mean(1))
    for run, _, _ in NETS:
        ans = []
        for raw in RAW_NET:
            z = np.load(os.path.join(AN, run, f"ck_{raw:05d}.npz"))
            a = z["logq"][:, -1].astype(np.float32).argmax(-1)
            assert ((a == P["r"]) != z["right"][:, -1]).mean() < 2e-3, (run, raw)   # float16 ties only
            ans.append(a)
        Z[f"w1_{run}"] = np.stack(ans).astype(np.int8)
    for run in [b[0] for b in BLOCKS] + [CTRL[0]] + list(OTHER):
        ans = []
        for raw in RAW_BLK:
            z = np.load(os.path.join(BLK_DIR, f"{run}_{raw:05d}.npz"))
            a = z["logit_T"].argmax(-1)
            assert np.array_equal(a == P["r"], z["right"]), (run, raw)
            ans.append(a)
        Z[f"w1_{run}"] = np.stack(ans).astype(np.int8)
    P2 = w2_probes()
    Z.update(w2_fr=forward_fr(P2["log_jb_T"], P2["log_nb_T"]), w2_env=P2["env"], w2_r=P2["r"],
             w2_naive=P2["log_nb_T"].argmax(-1), w2_bayes=P2["log_jb_T"].argmax(-1))
    for run, (short, _, _) in zip(W2_RUNS, BLOCKS):
        ans = []
        for raw in W2_RAW:
            f = os.path.join(W2_DIR, f"{run}_{raw:05d}.npz")
            if not os.path.exists(f):
                t0 = time.time()
                lq = w2_replay(run, raw, P2)
                np.savez(f, right=lq.argmax(-1) == P2["r"], logit_T=lq.astype(np.float32))
                print(f"{run} {raw}: {time.time() - t0:.0f} s", flush=True)
            ans.append(np.load(f)["logit_T"].argmax(-1))
        Z[f"w2_{short}"] = np.stack(ans).astype(np.int8)
    np.savez_compressed(CACHE, raw_net=np.array(RAW_NET), raw_blk=np.array(RAW_BLK), raw_w2=np.array(W2_RAW), **Z)
    print("wrote", CACHE)


def load():
    if not os.path.exists(CACHE):
        build()
    Z = dict(np.load(CACHE))
    for w in ("w1", "w2"):
        Z[f"{w}_bin"], Z[f"{w}_x"] = tenths(Z[f"{w}_fr"])
    return Z


def profile(hit, b):
    """mean of a per-state indicator (n, 2) in every FR tenth -> (10,)"""
    return np.array([hit[b == k].mean() for k in range(NQ)])


def boot_weights():
    """(N_BOOT, 50): how often each test environment is drawn in each resample (td_tt_learnmask_fr.boot_weights)"""
    rng = np.random.default_rng(BOOT_SEED)
    return np.stack([np.bincount(rng.integers(0, N_ENV, N_ENV), minlength=N_ENV) for _ in range(N_BOOT)]).astype(float)


def boot(env, sel, a, b=None, W=None):
    """mean of indicator a (minus that of b) over the selected states, with the 95% interval over environment resamples"""
    e = np.repeat(env[:, None], 2, 1)[sel]
    n = np.bincount(e, minlength=N_ENV).astype(float)
    s = np.bincount(e, weights=a[sel].astype(float), minlength=N_ENV)
    v, pt = (W @ s) / (W @ n), s.sum() / n.sum()
    if b is not None:
        s2 = np.bincount(e, weights=b[sel].astype(float), minlength=N_ENV)
        v, pt = v - (W @ s2) / (W @ n), pt - s2.sum() / n.sum()
    return pt, np.quantile(v, .025), np.quantile(v, .975)


# ── numbers ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def numbers(Z):
    f = lambda a: " ".join(f"{v:5.3f}" for v in a)
    g = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    W = boot_weights()
    for w, name in (("w1", "first world"), ("w2", "second world")):
        b, r, env = Z[f"{w}_bin"], Z[f"{w}_r"], Z[f"{w}_env"]
        nv, by = Z[f"{w}_naive"], Z[f"{w}_bayes"]
        print(f"\n== {name}: {b.size:,} states ({len(np.unique(env))} environments x 100 cells x 2 streams x 2 variables), "
              f"{(b == 0).sum():,} per tenth; FR_T mean {Z[f'{w}_fr'].mean():.3f} nats")
        print(f"{'tenth mean FR_T':34s}", " ".join(f"{v:5.3g}" for v in Z[f"{w}_x"]))
        print(f"{'naive':34s}", f(profile(nv == r, b)), f"| {(nv == r).mean():.3f}")
        print(f"{'Bayes':34s}", f(profile(by == r, b)), f"| {(by == r).mean():.3f}")
        print(f"{'Bayes answers as naive':34s}", f(profile(by == nv, b)), f"| {(by == nv).mean():.3f}")
        runs = ([(n[0], RAW_NET) for n in NETS] + [(CTRL[0], RAW_BLK)] + [(k[0], RAW_BLK) for k in BLOCKS]
                + [(k, RAW_BLK) for k in OTHER]) if w == "w1" else [(k[0], W2_RAW) for k in BLOCKS]
        for run, raws in runs:
            print(f"-- {run}: accuracy at T by tenth | all;  then the share of answers that are naive's answer")
            for i, raw in enumerate(raws):
                a = Z[f"{w}_{run}"][i]
                print(f"   {train_x(raw):6,d} episodes  {f(profile(a == r, b))} | {(a == r).mean():.3f}    naive's: "
                      f"{f(profile(a == nv, b))}")
        top, bot = b == NQ - 1, b == 0
        print("-- contrasts at the last drawn stage, [95% interval over environment resamples]")
        if w == "w1":
            for run, _, _ in NETS:
                a = Z[f"w1_{run}"][-1] == r
                print(f"   {run:18s} highest tenth {g(boot(env, top, a, W=W))}, minus Bayes {g(boot(env, top, a, by == r, W))}, "
                      f"minus naive {g(boot(env, top, a, nv == r, W))}; lowest tenth minus naive {g(boot(env, bot, a, nv == r, W))}")
        ref = Z[f"{w}_{BLOCKS[0][0]}"][-1] == r
        for run in ([CTRL[0]] if w == "w1" else []) + [k[0] for k in BLOCKS] + (list(OTHER) if w == "w1" else []):
            a = Z[f"{w}_{run}"][-1] == r
            print(f"   {run:18s} highest tenth {g(boot(env, top, a, W=W))}, minus random {g(boot(env, top, a, ref, W))}; "
                  f"lowest tenth {g(boot(env, bot, a, W=W))}, minus random {g(boot(env, bot, a, ref, W))}")
    b, fr = Z["w1_bin"], Z["w1_fr"]
    hi = Z["w1_cover_hi"]
    print("\n== share of a state's 30 updates in the highest-FR_t half of their step (ranking over the probe set), by tenth")
    print("   highest-FR half", f(profile(hi, b)), "\n   lowest-FR half ", f(profile(1 - hi, b)))
    env, cid = Z["w1_env"], Z["w1_env"] * 100 + Z["w1_cell"]
    em = np.array([fr[env == k].mean() for k in range(N_ENV)])
    cm = np.array([fr[cid == k].mean() for k in range(N_ENV * 100)])
    print(f"== FR_T variance: between environments {((em[env] - fr.mean()) ** 2).mean() / fr.var():.3f}, between cells "
          f"within an environment {((cm[cid] - em[env]) ** 2).mean() / fr.var():.3f}, within cells (2 streams x 2 "
          f"variables) {((fr - cm[cid][:, None]) ** 2).mean() / fr.var():.3f}; environment means {em.min():.2f}-{em.max():.2f} nats")


# ── figure ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def axes_style(ax, left, bottom):
    ax.set_xscale("log")
    ax.set_xlim(.0052, 40)
    ax.set_xticks([.01, .1, 1, 10])
    ax.set_xticklabels(["0.01", "0.1", "1", "10"] if bottom else [])
    ax.minorticks_off()
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, .5, 1])
    ax.set_yticks([.25, .75], minor=True)
    ax.set_yticklabels(["0", "0.5", "1"] if left else [])
    ax.tick_params(labelsize=FT, pad=2, length=3)
    ax.tick_params(which="minor", length=1.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(.1, color="0.55", lw=.7, ls=":", zorder=1)


def observers(ax, x, naive, bayes):
    ax.plot(x, bayes, color="k", lw=.9, zorder=6)
    ax.plot(x, naive, color="k", lw=.9, ls=(0, (4, 2)), zorder=6)


def stages(ax, x, prof, raws):
    for i, (p, raw) in enumerate(zip(prof, raws)):
        last = i == len(raws) - 1
        ax.plot(x, p, color=CMAP(NORM(train_x(raw))), lw=2.0 if last else 1.0, marker="o", ms=2.6 if last else 1.5,
                mew=0, zorder=5 if last else 3, solid_capstyle="round")


def title(ax, letter, name, color="k"):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points", ha="left",
                va="bottom", fontsize=TS, annotation_clip=False)
    ax.annotate(name, xy=(0, 1), xycoords="axes fraction", xytext=(13, 4), textcoords="offset points", ha="left",
                va="bottom", fontsize=TS, color=color, annotation_clip=False)


def figure(Z):
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    b1, x1, r1 = Z["w1_bin"], Z["w1_x"], Z["w1_r"]
    b2, x2, r2 = Z["w2_bin"], Z["w2_x"], Z["w2_r"]
    naive, bayes = profile(Z["w1_naive"] == r1, b1), profile(Z["w1_bayes"] == r1, b1)
    acc = lambda run: np.stack([profile(a == r1, b1) for a in Z[f"w1_{run}"]])
    acc2 = lambda run: profile(Z[f"w2_{run}"][-1] == r2, b2)

    left, right, gap, pw, ph = .47, .05, .085, None, 1.12
    pw = (FIG_W - left - right - 3 * gap) / 4
    head, ttl, tick, xlab, strip, note = .19, .21, .17, .19, .31, .40
    H = .03 + 2 * (head + ttl + ph + tick) + .10 + xlab + strip + note + .02
    fig = plt.figure(figsize=(FIG_W, H))
    add = lambda col, top: fig.add_axes([(left + col * (pw + gap)) / FIG_W, 1 - (top + ph) / H, pw / FIG_W, ph / H])
    ftext = lambda x, top, s, **k: fig.text(x / FIG_W, 1 - top / H, s, **k)
    top1 = .03 + head + ttl
    top2 = top1 + ph + tick + .10 + head + ttl

    ftext(left - .33, top1 - ttl - .02, r"Accuracy at the last step, $P(\hat r_T = r)$, by the state's factorization regret",
          fontsize=FL, ha="left", va="bottom")
    ftext(left - .33, top2 - ttl - .02, r"Trained RNN, gradient blocked at half of every step's updates: which half",
          fontsize=FL, ha="left", va="bottom")

    # row 1: the three networks through training, and at the end
    for col, ((run, name, c), letter) in enumerate(zip(NETS, "abc")):
        ax = add(col, top1)
        axes_style(ax, col == 0, True)
        stages(ax, x1, acc(run), RAW_NET)
        observers(ax, x1, naive, bayes)
        title(ax, letter, name, c)
    ax = add(3, top1)
    axes_style(ax, False, True)
    for run, name, c in NETS:                             # the Trained RNN wide under Bayes's line, ESN + MLPs dashed
        kw = dict(trained=dict(lw=3.2, solid_capstyle="round"), mlp55=dict(lw=1.7, ls=(0, (2.2, 2.2)), zorder=5),
                  echo=dict(lw=1.9, marker="o", ms=2.4, mew=0))[run]
        ax.plot(x1, acc(run)[-1], color=c, **{"zorder": 4, **kw})
    observers(ax, x1, naive, bayes)
    title(ax, "d", "after 40k")

    # row 2: gradient blocks from initialization, to 2,400 training episodes, and all at 2,400
    for col, ((run, name, c), letter) in enumerate(zip(BLOCKS, "efg")):
        ax = add(col, top2)
        axes_style(ax, col == 0, True)
        stages(ax, x1, acc(run), RAW_BLK)
        observers(ax, x1, naive, bayes)
        title(ax, letter, name, c)
    ax = add(3, top2)
    axes_style(ax, False, True)
    band = dict(color=CTRL[2], lw=5.0, alpha=.32, solid_capstyle="round")
    ax.plot(x1, acc(CTRL[0])[-1], zorder=2, **band)
    for run, name, c in BLOCKS:                           # the random half dashed on top: it coincides with the others
        kw = dict(lw=1.3, ls=(0, (2.4, 2.0)), zorder=8) if "rand" in run else dict(
            lw=1.5, marker="o", ms=2.2, mew=0, zorder=5 if "hi" in run else 4)
        ax.plot(x1, acc(run)[-1], color=c, **kw)
        ax.plot(x2, acc2(run), color=c, lw=.8, marker="o", ms=3.0, mfc="w", mew=.8, zorder=7)
    observers(ax, x1, naive, bayes)
    title(ax, "h", "after 2,400")
    ax.legend([Line2D([], [], **band)], ["no block"], loc="upper right", bbox_to_anchor=(1.0, 1.02), fontsize=FT,
              frameon=False, handlelength=1.1, handletextpad=.5, borderpad=0, borderaxespad=0)

    for top in (top1, top2):
        fig.axes[0].figure.text(.012, 1 - (top + ph / 2) / H, "accuracy", rotation=90, fontsize=FL, ha="left", va="center")
    ftext(FIG_W / 2, top2 + ph + tick + .02,
          r"factorization regret of the state,  $\mathrm{FR}_T = D_{\mathrm{KL}}(B_T^{\mathrm{Bayes}}\,\Vert\,B_T^{\mathrm{Naive}})$  (nats)",
          fontsize=FL, ha="center", va="top")

    # legend strip: the observers' lines, and the stage colours
    y0 = top2 + ph + tick + xlab + .06
    hs = [Line2D([], [], color="k", lw=.9), Line2D([], [], color="k", lw=.9, ls=(0, (4, 2))),
          Line2D([], [], color="0.55", lw=.7, ls=":"), Line2D([], [], color="0.3", lw=.8, marker="o", ms=3.0, mfc="w", mew=.8)]
    fig.legend(hs, ["Bayes", "naive", "chance", "second world"], loc="upper left", bbox_to_anchor=(.035 / FIG_W, 1 - y0 / H),
               ncol=4, fontsize=FT, frameon=False, handlelength=1.5, columnspacing=.9, handletextpad=.4, borderpad=0,
               borderaxespad=0)
    cw = 1.25
    cax = fig.add_axes([(FIG_W - right - cw - .28) / FIG_W, 1 - (y0 + .105) / H, cw / FIG_W, .075 / H])
    cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal")
    cax.set_xlim(train_x(RAW_NET[0]), train_x(RAW_NET[-1]))
    cb.set_ticks([100, 1000, 10000])
    cb.set_ticklabels(["100", "1k", "10k"])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=FT, pad=1.5, length=2)
    cb.outline.set_linewidth(.5)
    ftext(FIG_W - right - cw - .33, y0 + .07, "episodes", fontsize=FT, ha="right", va="center")
    ftext(FIG_W - right - .25, y0 + .07, "40k", fontsize=FT, ha="left", va="center")

    n = b1.size
    ftext(FIG_W / 2, y0 + strip,
          f"{N_ENV} test environments $\\times$ 200 trajectories $\\times$ 2 variables = {n:,} states per network,\n"
          f"{n // NQ:,} per point (tenths of $\\mathrm{{FR}}_T$); one training run per panel; e-g end at 2,400 episodes",
          fontsize=FT, ha="center", va="top", color="0.15", linespacing=1.35)
    path = fig_path("fig_learn_profile.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    if "--data" in sys.argv:
        build()
    Z = load()
    if "--numbers" in sys.argv:
        numbers(Z)
    figure(Z)
