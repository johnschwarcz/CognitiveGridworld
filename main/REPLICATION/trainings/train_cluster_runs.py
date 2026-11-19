"""Follow-up runs (2026-11), one run per Slurm array task; TASKS is in priority order.

  zflow_s{k}               original feedback (mode RL) with the oracle Generator changes switched on (gen_zflow); |S| 500,
                           batch 20000, lr 1e-4 (RL_state_num_reps' settings). Stops after 3 consecutive tests at accuracy
                           >= ZFLOW_STOP, or at 30k episodes.
  default_s{k}             zflow_s{k}'s matched control: same settings, seed and stopping rule, default Generator.
  zflow_{low,high,half}_s{k}  original feedback with the zflow Generator; training episodes drawn only from cells where the
                           Naive observer converges to the true goal (low FR), where it does not (high FR), or from a fixed
                           random half of each environment's cells (control). Test episodes are unrestricted. 30k episodes,
                           no early stop.
  frsel_{low,high}_s0      the same with the default Generator; cancelled after ~1 h (2.5k / 1.5k episodes) when John
                           switched every Generator to zflow (11-08 22:45). Their JSONLs are kept; nothing was saved.
  oracle_S{S}_d{d}         oracle mode at train_oracle_ablation.py's settings, |S| x d_E (KQ_dim) grid, 10k episodes.
  rcc_S{S}_s{k}            the RCC with its original feedback (mode RL) and the zflow Generator at train_oracle_ablation.py's
                           settings (Fig 13's: batch 10k, lr 1e-3, entropy .01, 10k episodes), so it differs from Fig 13's
                           Generator arm only in the feedback. |S| = 500 is Fig 13's third arm; the |S| sweep is the
                           "network's answers" column beside train_bayes_generator.py's Joint / Naive columns (John 11-09).
                           No early stop. Their tests also log the learned likelihood (plugin, below).
  rcc30k_S{S}_s{k}         rcc_S{S}_s{k} run to 30k episodes (11-09 17:20: at 5.5k rcc_S500_s0 was at .31, Fig 13's Generator
                           arm at .66 -- 10k episodes do not reach the plateau). Same seed = same first 10k episodes.
  rcc_rl_S{S}_s{k}         the same RCC (zflow Generator, original feedback) at Fig 12's settings (zflow_s{k}'s: batch 20k,
                           lr 1e-4, entropy .1; 30k episodes, Fig 12 used 50k): does Fig 12's |S| picture hold with the
                           updated Generator? Logged like rcc_*.

Every test episode appends a JSON line to runs/<name>.jsonl: final-step accuracy of the network, Joint and Naive Bayes,
overall and on low- / high-FR test cells (naive_consistent_cells). The run saves DATA/cluster_runs/<name>_net.pth when it
ends, when it stops early, or when it nears its wall-time budget. rcc* lines (and, from 11-10 03:40, gen30k_* /
clf30k_*, so all four cells of feedback x gradient route share one read-out) add, as in train_bayes_generator.py: plugin
(accuracy at T of the Bayes posterior computed with the Generator's predictions at all 100 joint values, confidence 1),
kl_env (E_r KL(P(o|r) || P^(o|r)), summed over channels), and train_net / train_plugin / train_kl_env on the last training
batch before the test (training environments; train_net = the Classifier's accuracy there, train_acc_through_training).
Runs with dream_weight > 0 (model_backward.dream_sample / dream_backward, 11-13) add dream_loss / dream_acc (means over the
training iterations since the previous test: the dream cross-entropy and the Classifier's accuracy at T on the dreamed values)
and dream_n, gpu_max_mem_gb (peak GPU memory so far, on GPU) and, for dream_gate runs, dream_gate_ep (first dreamed episode).
Teaching runs (TEACH_FROM, 11-13: imagine_* / teach_* / rl_*, a fresh Classifier trained on a frozen trained net) start a
fresh log at episode 0 and log the frozen Generator's plugin / kl_env; their oracle_teach runs add
teach_loss / teach_acc / teach_n (the same means for the real-label cross-entropy and accuracy at T).

    python train_cluster_runs.py --task 3 [--budget-h 13.5] [--smoke]
"""
import os, sys, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

RL = dict(mode="RL", ctx_num=2, obs_num=5, realization_num=10, state_num=500, hid_dim=1000, batch_num=20000, step_num=30,
          learn_embeddings=True, training=True, classifier_LR=1e-4, generator_LR=1e-4, episodes=30000,
          checkpoint_every=500, gpu_inference=True, show_plots=False)
ORACLE = dict(mode="oracle", ctx_num=2, obs_num=5, realization_num=10, state_num=500, hid_dim=1000, batch_num=10000,
              step_num=30, learn_embeddings=True, training=True, classifier_LR=1e-3, generator_LR=1e-3,
              classifier_ent_bonus=.01, episodes=10000, checkpoint_every=250, gpu_inference=True, show_plots=False)
ZFLOW_STOP = 0.68

def _zf(k):
    return (f"zflow_s{k}", dict(RL, gen_zflow=True), k)


def _fr(a, k):
    return (f"frsel_{a}_s{k}", dict(RL, fr_select=a), k)


def _zfr(a, k):
    return (f"zflow_{a}_s{k}", dict(RL, gen_zflow=True, fr_select=a), k)


def _df(k):
    return (f"default_s{k}", dict(RL), k)


def _orc(S, d):
    return (f"oracle_S{S}_d{d}", dict(ORACLE, state_num=S, KQ_dim=d), 0)


def _rcc(S, k):
    return (f"rcc_S{S}_s{k}", dict(ORACLE, mode="RL", gen_zflow=True, state_num=S), k)


def _rcc30(S, k):
    return (f"rcc30k_S{S}_s{k}", dict(ORACLE, mode="RL", gen_zflow=True, state_num=S, episodes=30000), k)


def _rccrl(S, k):
    return (f"rcc_rl_S{S}_s{k}", dict(RL, gen_zflow=True, state_num=S), k)


def _diag(tag, k, **over):
    """Fig 13's settings with one change, |S| 500, 30k episodes: why only 1 of 3 seeds left chance there (11-09 23:00,
    John: every Fig 13 arm at batch 10k; a lower lr is fair, a larger entropy bonus preferable if it learns faster -> the
    variant that leaves chance reliably becomes the RCC arm, so these run the arm's full 30k).
    lr4clf = the RCC arm (lr4) with the embeddings trained by the Classifier's reward loss instead of the Generator's
    (embedding_grad 'classifier'): the Generator ablation of Fig 13 under the ORIGINAL feedback, the missing cell
    (Fig 13 ran it only with oracle feedback). The Generator still trains, on detached embeddings, so
    plugin / kl_env read out what the Classifier-trained embeddings hold."""
    return (f"rccdiag_{tag}_s{k}", dict(ORACLE, mode="RL", gen_zflow=True, episodes=30000, **over), k)


def _gen30(k):
    """Fig 13's Generator-only arm (train_oracle_ablation.py default) to 30k episodes"""
    return (f"gen30k_s{k}", dict(ORACLE, episodes=30000), k)


def _clf30(k):
    """Fig 13's Classifier-only arm (train_oracle_ablation.py clf_reg: embeddings get the Classifier's gradients) to 30k"""
    return (f"clf30k_s{k}", dict(ORACLE, embedding_grad="classifier", episodes=30000), k)


# Order for a night with ~1 free GPU and more from the morning (John 11-08: "order as you see fit"): the gating Generator
# test first; then the 4 grid points that test whether |S| / d_E sets generalization; then the low/high/half FR trio
# together, so the pair finishes at the same time; then the rest. 11-08 21:45, tasks 0-5 started: zflow_s0 left chance at
# ~5k episodes (default runs: 12-25k or never), so its seeds 1 and 2 moved ahead of the rest (indices >= 6 only).
# 11-08 22:45, John: every Generator -> zflow. Tasks 5 and 7 (default Generator) cancelled; the FR trio restarts at 8-10.
# The array was submitted as 0-20; 21-22 go in a follow-up array that starts after task 20 starts.
# 11-09 10:00, tasks 0-13 started: the saved default runs are no control (RL_2 entropy .01; the |S| sweep's batch is not
# recorded), so default_s0-2 (zflow_s0-2's settings and seeds, default Generator) go next, then the d_E = 10 oracle points
# (train/test gap), the FR trio's second seed, and the rest of the grid. 21-25 run in a follow-up array.
# 11-09 12:18, John: the FR trio's second seed (20-22) cancelled before starting (the FR-selection design is dropped).
# 11-09 16:10, John asked whether the default seeds can make way: default_s1 left chance at 21k (zflow seeds 4-8.5k) and
# runs on; default_s0 / s2 (still at chance at 15.5k / 12k) cancelled. 26-31: the RCC at Fig 13's settings, |S| = 500 first
# (does the original feedback train at lr 1e-3?), then the |S| sweep, then two more |S| = 500 seeds for Fig 13.
# 11-09 17:20: the original feedback trains at lr 1e-3 but ~7x slower than oracle feedback (chance left at 3.5k vs 500
# episodes; .31 vs .66 at 5.5k), so 10k episodes are too short: 29-31 cancelled unstarted, and 32-43 (the overnight array,
# not started) run 30k episodes: Fig 13's third arm (|S| 500, seeds 1, 2, 0; the oracle tasks 23-25 start after 34) and
# the network column (|S| 250, 100, 1000, then 125-25). 44-46 (Fig 12's settings with the zflow Generator, |S| 400 / 250 / 100) are defined but not submitted.
# 11-09 21:40, John: the network column uses the main-text protocol (rcc_rl: Fig 12's settings, batch 20k, lr 1e-4, entropy
# .1, the zflow Generator), since at Fig 13's settings only 1 of 3 |S| = 500 seeds left chance (rcc30k_S500_s1 still at
# chance at 30k). 36-43 (held at 21:16) cancelled unstarted; 35 (rcc30k_S250_s0) had started and runs on. 44-54 =
# rcc_rl at |S| 400, 250, 100, 1000, 50, 150, 500, 125, 75, 200, 25 (500 re-runs zflow_s0's settings with the plug-in log).
TASKS = ([_zf(0)] + [_orc(100, 30), _orc(100, 10), _orc(250, 90), _orc(1000, 90)] +
         [_fr("low", 0), _zf(1), _fr("high", 0)] + [_zfr(a, 0) for a in ("low", "high", "half")] + [_zf(2)] +
         [_orc(250, 30), _orc(1000, 30)] + [_df(k) for k in range(3)] + [_orc(250, 10), _orc(1000, 10), _orc(500, 10)] +
         [_zfr(a, 1) for a in ("low", "high", "half")] + [_orc(100, 90), _orc(500, 30), _orc(500, 90)] +
         [_rcc(500, 0), _rcc(100, 0), _rcc(250, 0), _rcc(1000, 0), _rcc(500, 1), _rcc(500, 2)] +
         [_rcc30(500, 1), _rcc30(500, 2), _rcc30(500, 0), _rcc30(250, 0), _rcc30(100, 0), _rcc30(1000, 0)] +
         [_rcc30(S, 0) for S in (125, 150, 200, 75, 50, 25)] + [_rccrl(S, 0) for S in (400, 250, 100)] +
         [_rccrl(S, 0) for S in (1000, 50, 150, 500, 125, 75, 200, 25)] +
         [_diag("ent1", k, classifier_ent_bonus=.1) for k in (1, 2)] +
         [_diag("lr4", k, classifier_LR=1e-4, generator_LR=1e-4) for k in (1, 2)] +
         [_zf(k) for k in (3, 4, 5)] + [_clf30(k) for k in range(6)] +
         [_diag("lr4ent1", k, classifier_LR=1e-4, generator_LR=1e-4, classifier_ent_bonus=.1) for k in (1, 2)] +
         [_gen30(k) for k in range(6)] +
         [_diag("lr4", k, classifier_LR=1e-4, generator_LR=1e-4) for k in (0, 3, 4, 5)] +
         [_diag("lr4clf", k, classifier_LR=1e-4, generator_LR=1e-4, embedding_grad="classifier") for k in range(6)])
assert len(TASKS) == 86 and len({t[0] for t in TASKS}) == 86
# 11-09 23:00: 59-61 (zflow_s3-5, batch 20k) cancelled unstarted (John: one batch size for every Fig 13 arm).
# 11-10 02:40: lr4 is the RCC arm (stuck seed 1 left chance at ~6k, .60 at 18k; ent1 = entropy .1 at lr 1e-3: s1 unstable,
# .26 at 18.75k, s2 at chance at 13k; both cancelled). 76-79 = its seeds 0, 3, 4, 5 (seed 2 = task 58). These runs use
# ~40% of a GPU (one CPU core each), so from here they run two per GPU (runs_pair.sbatch).
# 11-10 03:40: 80-85 = rccdiag_lr4clf_s0-5, the missing cell of feedback (oracle / original) x embedding-gradient route
# (Generator / Classifier): gen30k, clf30k, rccdiag_lr4 and these, all at batch 10k, 30k episodes, n = 6.
# 11-10 10:55, John: one lr for every Fig 13 arm (1e-4: the only lr at which the original feedback leaves chance reliably)
# and 60k episodes ("we definitely need longer training for everything"), first in the GPU queue. 86-91 oracle Generator
# route, 92-97 oracle Classifier route, 98-103 original feedback with the Classifier route (replaces lr4clf, unstarted),
# all batch 10k, lr 1e-4, 60k episodes; 104-109 = the RCC arm (rccdiag_lr4_s0-5, lr 1e-4) resumed from its 30k nets to 60k
# (RESUME: the log starts with the source's records; the embeddings are handed back to the optimizer after load(), whose
# new Parameters it would otherwise never train; Adam's moments restart; the episode stream is reseeded).
LR4_60K = dict(classifier_LR=1e-4, generator_LR=1e-4, episodes=60000)
TASKS += ([(f"gen60k_lr4_s{k}", dict(ORACLE, **LR4_60K), k) for k in range(6)] +
          [(f"clf60k_lr4_s{k}", dict(ORACLE, embedding_grad="classifier", **LR4_60K), k) for k in range(6)] +
          [(f"rccclf60k_lr4_s{k}", dict(ORACLE, mode="RL", gen_zflow=True, embedding_grad="classifier", **LR4_60K), k)
           for k in range(6)] +
          [(f"rcc60k_lr4_s{k}", dict(ORACLE, mode="RL", gen_zflow=True, **LR4_60K), k) for k in range(6)])
# 11-10 11:50 (John: the plug-in's lead over the Classifier in several FR bins): 104-109 run the RCC fresh to 60k instead of
# resuming rccdiag_lr4 (only fresh runs log the per-FR-bin read-out from the start, where the lead peaks); every test logs
# net / plugin / joint / naive accuracy per FR fifth (FR_EDGES). RESUME = {name: source run} still works for later use.
RESUME = {}
# FR of a test episode = mean over both variables of KL(B_T(v) || B~_T(v)) (Joint vs Naive marginal belief at T, nats);
# fifths cut at the quintiles of the novel test episodes at |S| 500 (bayes_gen_fr_div data: .277 / .65 / 1.48 / 4.46)
FR_EDGES = (.28, .65, 1.5, 4.5)
assert len(TASKS) == 110 and len({t[0] for t in TASKS}) == 110
# 11-11 18:30 (John: recreate the paper's Fig 12 with the fixed Generator, the RCC's own feedback, 30k episodes): the
# sizes 100 / 125 / 150 / 200 / 250 / 300 / 400 / 500 / 1000; |S| 300 was never in the list (task 110), 150 / 200 are tasks
# 49 / 53 (resubmitted), 500 = zflow_s0
TASKS += [_rccrl(300, 0)]
assert len(TASKS) == 111 and len({t[0] for t in TASKS}) == 111
# 11-11 23:10 (John: "Test both please"; does the Generator route generalize over a wider |S| range than the Classifier
# route?): oracle feedback, lr 1e-4, batch 10k, 30k episodes, both routes at |S| 100 / 250 / 150, seeds 0-1; a pair (the two
# routes at one |S| and seed) shares a GPU. 111-122 = (gen, clf) x (S100 s0, S100 s1, S250 s0, S250 s1, S150 s0, S150 s1).
LR4_30K = dict(classifier_LR=1e-4, generator_LR=1e-4, episodes=30000)
TASKS += [(f"{route}30k_lr4_S{S}_s{k}", dict(ORACLE, state_num=S, **LR4_30K, **over), k)
          for S in (100, 250, 150) for k in (0, 1)
          for route, over in (("gen", {}), ("clf", dict(embedding_grad="classifier")))]
assert len(TASKS) == 123 and len({t[0] for t in TASKS}) == 123
# 11-12 00:50 (John: "option a is to see if the classifier can solve its trap by using the same readin structure as the
# generator? Yes this is very important"; "establishing an effect that holds across repetition and long training"): the
# Classifier route with the Classifier reading every channel's (bit, Z) through ONE function (readin_shared,
# main/model/shared_readin.py), Fig 13's 60k protocol, 6 seeds: 123-128 oracle feedback, 129-134 the RCC's own feedback.
TASKS += ([(f"clfshared60k_lr4_s{k}", dict(ORACLE, embedding_grad="classifier", readin_shared=True, **LR4_60K), k)
           for k in range(6)] +
          [(f"rccclfshared60k_lr4_s{k}", dict(ORACLE, mode="RL", gen_zflow=True, embedding_grad="classifier", readin_shared=True,
                                              **LR4_60K), k) for k in range(6)])
assert len(TASKS) == 135 and len({t[0] for t in TASKS}) == 135
# 11-12 02:35 (John: "are we testing robustness to hyperparams? ... can the generator in the RCC setting handle a high
# enough lr (maybe even 1e-2) to learn crazy quickly compared to the classifier-route"): one lr for both modules, as in
# Fig 13; pairs = (Generator route, Classifier route) at one lr and seed. 135-146 oracle feedback at lr 3e-3 / 1e-2, 6,000
# iterations, tests every 100; 147-158 the RCC's own feedback at lr 1e-3 / 3e-3, 10,000 iterations; seeds 0-2.
for _lr, _tag, _fb in ((3e-3, "3e3", "oracle"), (1e-2, "1e2", "oracle"), (1e-3, "1e3", "rcc"), (3e-3, "3e3", "rcc")):
    for _k in range(3):
        for _route, _over in (("gen", {}), ("clf", dict(embedding_grad="classifier"))):
            if _fb == "oracle":
                TASKS.append((f"{_route}6k_lr{_tag}_s{_k}", dict(ORACLE, classifier_LR=_lr, generator_LR=_lr, episodes=6000,
                                                                  checkpoint_every=100, **_over), _k))
            else:
                TASKS.append((f"{'rcc' if _route == 'gen' else 'rccclf'}10k_lr{_tag}_s{_k}",
                              dict(ORACLE, mode="RL", gen_zflow=True, classifier_LR=_lr, generator_LR=_lr, episodes=10000,
                                   **_over), _k))
assert len(TASKS) == 159 and len({t[0] for t in TASKS}) == 159
# 11-12 04:00 (John: "high lr makes lost channels likely. So if the generator can handle high lr and the classifier route
# cant handle a shared readin that is a direction"). clfshared60k_lr4 s0-3 were still at chance at 11.5-13k iterations
# (the dense Classifier route left chance at 3-7k). 159-160: the Generator route with the shared read-in, lr 1e-4, 20k
# iterations (its Classifier reads the Generator's codes: is the shared read-in a working reader?); 161-168 pairs
# (Generator route, Classifier route), both with the shared read-in, at lr 1e-3 / 3e-3, seeds 0-1, 10k iterations (does a
# higher lr let the Classifier route learn through it?).
TASKS += [(f"genshared20k_lr4_s{k}", dict(ORACLE, readin_shared=True, classifier_LR=1e-4, generator_LR=1e-4, episodes=20000),
           k) for k in range(2)]
for _lr, _tag in ((1e-3, "1e3"), (3e-3, "3e3")):
    for _k in range(2):
        for _route, _over in (("gen", {}), ("clf", dict(embedding_grad="classifier"))):
            TASKS.append((f"{_route}shared10k_lr{_tag}_s{_k}", dict(ORACLE, readin_shared=True, classifier_LR=_lr,
                                                                   generator_LR=_lr, episodes=10000, **_over), _k))
assert len(TASKS) == 169 and len({t[0] for t in TASKS}) == 169
# 11-12 04:30 (John: "we saw that it eventually started to learn, suggesting that channels do eventually recover. So we
# need a lot a data to see if the gap in accuracy stay large"): Fig 13's runs continued from their saved nets (RESUME:
# same environments, the source's log first, Adam restarts). 169-174 rccclf60k_lr4_s0-5 -> 120k (the RCC's own feedback,
# Classifier route: .33-.62 at 60k, still rising .06-.15 per 10k); 175-176 rcc60k_lr4_s0-1 -> 120k (its Generator-route
# reference, .69 at 60k); 177-180 clf30k_s0-3 -> 90k (oracle, lr 1e-3: 3 of 4 lost a channel, .61-.64 at 30k); 181-182
# gen30k_s0-1 -> 90k; 183 clf60k_lr4_s2 -> 120k (lost 2 channels, .52, flat since 30k); 184 gen60k_lr4_s2 -> 120k.
_cfg = {t[0]: t for t in TASKS}
for _name, _src, _total in ([(f"rccclf120k_lr4_s{k}", f"rccclf60k_lr4_s{k}", 120000) for k in range(6)] +
                            [(f"rcc120k_lr4_s{k}", f"rcc60k_lr4_s{k}", 120000) for k in range(2)] +
                            [(f"clf90k_s{k}", f"clf30k_s{k}", 90000) for k in range(4)] +
                            [(f"gen90k_s{k}", f"gen30k_s{k}", 90000) for k in range(2)] +
                            [("clf120k_lr4_s2", "clf60k_lr4_s2", 120000), ("gen120k_lr4_s2", "gen60k_lr4_s2", 120000)]):
    TASKS.append((_name, dict(_cfg[_src][1], episodes=_total), _cfg[_src][2]))
    RESUME[_name] = _src
assert len(TASKS) == 185 and len({t[0] for t in TASKS}) == 185
# 11-12 05:15 (John: "we should show the best classifier, but we also need to figure out how to really emphasize its
# failure"): the default dense read-in with the channel order shuffled in every episode (readin_perm,
# model_forward.RNN_readin: bits and codes moved together), so the DATA, not the architecture, asks for a reader that
# ignores channel order. 185-188: (Generator route, Classifier route) pairs at lr 1e-4, 20k iterations, seeds 0-1; 189-192
# the same at lr 1e-3, 10k iterations.
for _lr, _tag, _n in ((1e-4, "20k_lr4", 20000), (1e-3, "10k_lr1e3", 10000)):
    for _k in range(2):
        for _route, _over in (("gen", {}), ("clf", dict(embedding_grad="classifier"))):
            TASKS.append((f"{_route}perm{_tag}_s{_k}", dict(ORACLE, readin_perm=True, classifier_LR=_lr, generator_LR=_lr,
                                                           episodes=_n, **_over), _k))
assert len(TASKS) == 193 and len({t[0] for t in TASKS}) == 193
# 11-12 06:10 (John: "please enable the support for C > 2"; the question of C > 2 and RCC scalability; John 11-12: "It could
# also be that the small difference in learning speed gets exacerbatted at C = 3?"): three variables per environment
# (1,000 joint values; 6 code numbers per channel; the Generator's aligned start tiled over the 3 pairs,
# model_architecture.default_generator). 193-196: (Generator route, Classifier route) pairs, oracle labels, lr 1e-4,
# 60k iterations, seeds 0-1. Not launched: on hold for John.
TASKS += [(f"{_route}C3_60k_lr4_s{_k}", dict(ORACLE, ctx_num=3, classifier_LR=1e-4, generator_LR=1e-4, episodes=60000, **_over), _k)
          for _k in range(2) for _route, _over in (("gen", {}), ("clf", dict(embedding_grad="classifier")))]
assert len(TASKS) == 197 and len({t[0] for t in TASKS}) == 197
# 11-12 11:50 (John: "Id be very curious about channel collapse being a much bigger issue for C = 3 so if we can compare the 2
# routes with typical RCC feedback and see the generator route succeed that would be great"): 197-200 = (Generator route,
# Classifier route) x seeds 0-1 at C = 3 with the RCC's own feedback (rcc60k_lr4 / rccclf60k_lr4's settings, ctx_num 3).
TASKS += [(f"{_route}C3_60k_lr4_s{_k}", dict(ORACLE, mode="RL", gen_zflow=True, ctx_num=3, classifier_LR=1e-4, generator_LR=1e-4,
                                           episodes=60000, **_over), _k)
          for _k in range(2) for _route, _over in (("rcc", {}), ("rccclf", dict(embedding_grad="classifier")))]
assert len(TASKS) == 201 and len({t[0] for t in TASKS}) == 201
# 11-12 12:25 (John: "Can we get the generator on answers grid filled for the RCC, and add an embedding dim = 20"):
# 201-212 = the RCC row's empty cells (oracle_S{25,50,75,125}_d{10,30,90}), 213-220 = d_E 20 at every |S| of the grid;
# oracle_grid_plot.py. Their logs carry train_net (the "oracle_S" prefix joins the read-out list below).
TASKS += [_orc(S, d) for S in (25, 50, 75, 125) for d in (10, 30, 90)] + [_orc(S, 20) for S in (25, 50, 75, 100, 125, 250, 500, 1000)]
assert len(TASKS) == 221 and len({t[0] for t in TASKS}) == 221
# 11-12 23:00 (John 11-12 12:00: "compare the 2 routes with typical RCC feedback and see the generator route succeed"):
# C = 3 seed 0 continued from its saved 60k nets to 120k (RESUME as above). 221 rccC3_120k_lr4_s0 (Generator route: .27 =
# Naive at 51k, rising), 222 rccclfC3_120k_lr4_s0 (Classifier route: .16 at 49k), 223 genC3_120k_lr4_s0 (oracle labels,
# Generator route: .49 at 60k, rising; its net is on the local GPU machine).
_cfg = {t[0]: t for t in TASKS}
for _name, _src in (("rccC3_120k_lr4_s0", "rccC3_60k_lr4_s0"), ("rccclfC3_120k_lr4_s0", "rccclfC3_60k_lr4_s0"),
                    ("genC3_120k_lr4_s0", "genC3_60k_lr4_s0")):
    TASKS.append((_name, dict(_cfg[_src][1], episodes=120000), _cfg[_src][2]))
    RESUME[_name] = _src
assert len(TASKS) == 224 and len({t[0] for t in TASKS}) == 224
# 11-13 01:15 (John: "is our default classifier using a shared readin? is it able to with generator (and true)
# embeddings?"): the TRUE embeddings (learn_embeddings=False: no route to choose), oracle labels, lr 1e-3, 10k iterations
# (genshared10k_lr1e3's protocol); 224 trueshared10k_lr1e3_s0 = the shared read-in, 225 truedense10k_lr1e3_s0 = the
# default dense read-in on the same codes (one pair, one GPU). NOT RUN (cancelled 01:25, John: "I don't think we need to
# check that unless you see that learning is insanely faster than the default setting"; with Generator-route codes the
# shared read-in crosses Naive at 1,250-1,500 vs 1,500 (lr 1e-3) and 2,500-2,750 vs 3,500-4,000 (lr 1e-4), and reaches .6
# at 2,250 vs 2,250-2,750 / 7,500-7,750 vs 6,750-7,500: not faster).
TASKS += [("trueshared10k_lr1e3_s0", dict(ORACLE, learn_embeddings=False, readin_shared=True), 0),
          ("truedense10k_lr1e3_s0", dict(ORACLE, learn_embeddings=False), 0)]
assert len(TASKS) == 226 and len({t[0] for t in TASKS}) == 226
# 11-13 02:30 (John: "I meant more going up, like training on 2 and testing on 5"): the RCC's own Classifier with the
# shared read-in reading ONLY channels 0-1 in training (readin_channels; the Generator learns all 5 channels' codes),
# tested afterwards with 3-5 channels (DATA/experimental/clf_scratch/swap/channel_up.py). Oracle labels, lr 1e-3, 10k
# iterations, batch 4,000 (fits beside task 195 on local GPU 0). 226-227 = seeds 0-1.
TASKS += [(f"genshared2ch10k_lr1e3_b4k_s{k}", dict(ORACLE, readin_shared=True, readin_channels=(0, 1), batch_num=4000), k)
          for k in range(2)]
assert len(TASKS) == 228 and len({t[0] for t in TASKS}) == 228
# 11-13 05:20 (John: "Let's try to get a few seeds of both routes at lr 1e-3 trained until convergence"): the RCC's own
# feedback (mode RL, gen_zflow) at the paper's oracle-figure lr 1e-3 for BOTH routes, batch 10k, entropy .01, 90k
# iterations (no early stop; the lr 1e-3 Generator route rcc30k_S500_s0 was still +.018 per 10k at 30k, and the lr 1e-4
# Classifier route needed 100-115k). Fresh runs, each seed's two routes paired on ONE GPU (hardware matched within a
# seed). 228-229 / 230-231 / 232-233 = (Generator route, Classifier route) for seeds 0 / 1 / 2. NOT RUN: cancelled
# unstarted 11-13 05:45 (John: "scratch that"; the lr 1e-3 failure, answer collapse in the Classifier's reward learning,
# would hit both routes, so it does not test the route).
for _k in range(3):
    TASKS += [(f"rcc90k_lr1e3_s{_k}", dict(ORACLE, mode="RL", gen_zflow=True, episodes=90000), _k),
              (f"rccclf90k_lr1e3_s{_k}", dict(ORACLE, mode="RL", gen_zflow=True, embedding_grad="classifier",
                                              episodes=90000), _k)]
assert len(TASKS) == 234 and len({t[0] for t in TASKS}) == 234
# 11-13 05:30 (John: "I'm focusing on the default feedback stuff ... utilizing all of our compute"; the oracle C = 3 runs
# 194/195 stopped): C = 3 with the RCC's own feedback, seed 2, both routes (as 197/198), on the freed local GPUs.
TASKS += [(f"{_route}C3_60k_lr4_s2", dict(ORACLE, mode="RL", gen_zflow=True, ctx_num=3, classifier_LR=1e-4, generator_LR=1e-4,
                                         episodes=60000, **_over), 2)
          for _route, _over in (("rcc", {}), ("rccclf", dict(embedding_grad="classifier")))]
assert len(TASKS) == 236 and len({t[0] for t in TASKS}) == 236
# 11-13 05:45 (John: "extend our default RCC feedback with lr 1e-4 to longer training"): the Generator route under the
# RCC's own feedback continued 60k -> 120k (RESUME, Adam restarts, as its Classifier-route twins rccclf120k_lr4_s0-5), so
# both routes end at 120k. s0-1 = tasks 175-176 (rcc120k_lr4_s0-1, defined 11-12, never run); 236-239 = s2-5.
_cfg = {t[0]: t for t in TASKS}
for _k in range(2, 6):
    TASKS.append((f"rcc120k_lr4_s{_k}", dict(_cfg[f"rcc60k_lr4_s{_k}"][1], episodes=120000), _k))
    RESUME[f"rcc120k_lr4_s{_k}"] = f"rcc60k_lr4_s{_k}"
assert len(TASKS) == 240 and len({t[0] for t in TASKS}) == 240
# 11-13 06:55 (John: "maybe we should dramatically reduce the batch size? This could emphasize the instability
# differences"; "use default RCC feedback from now on"; run where it pushes nothing in the queue): the RCC's own feedback
# (mode RL, gen_zflow, lr 1e-4, entropy .01) at batch 1,000 (10x smaller than every Fig 13 run; everything else as
# rcc60k_lr4 / rccclf60k_lr4), 120k iterations, tests every 250 on 1,000 novel-environment episodes (the batch).
# 240-245 = Generator route s0-5, 246-251 = Classifier route s0-5. Local GPUs, beside the running jobs (0.083 s/it there).
TASKS += [(f"rccb1k_lr4_s{k}", dict(ORACLE, mode="RL", gen_zflow=True, classifier_LR=1e-4, generator_LR=1e-4,
                                    batch_num=1000, episodes=120000), k) for k in range(6)]
TASKS += [(f"rccclfb1k_lr4_s{k}", dict(ORACLE, mode="RL", gen_zflow=True, classifier_LR=1e-4, generator_LR=1e-4,
                                       batch_num=1000, episodes=120000, embedding_grad="classifier"), k) for k in range(6)]
assert len(TASKS) == 252 and len({t[0] for t in TASKS}) == 252
# ---- 11-13 dream pilot (offline imagination, model_backward.dream_sample / dream_backward) ----------------------------------
# The Classifier also learns, with oracle_loss's cross-entropy on both variables at every step (held-out test-state variables
# masked), from one batch per training iteration dreamed by its own Generator on the same environments. Every other setting
# is its twin's, read from the twin's task, so the same seed gives the same initial net, environments and tests; the dream
# draws from a private torch RNG stream (dream_seed = the task's seed), so the real batches' draws are the twin's too.
# dream_weight 1.0 is untuned: at initialization the dream's gradient on the Classifier is ~2.4x the RL loss's (batch 64),
# so the Classifier's update is mostly the dream term; fallback if the dream arms escape later than their twins:
# dream_weight .4 (equal gradient norms at initialization). Comparator for every Generator-route arm: its twin, and the
# Classifier route WITHOUT dreams (rccclf120k_lr4 / rccclfb1k_lr4); report curves against iterations AND against wall-clock
# time (a dream iteration costs ~1.9x: two Classifier forward/backward passes).
# Order = launch priority (indices relative to _n, the TASKS length before this block):
#   _n+0..1  rccdream60k_lr4_s0-1      Stage 1: Generator route, batch 10k, 60k, dream from iteration 1 (twins rcc60k_lr4_s0-1)
#   _n+2..3  rccdreamshuf60k_lr4_s0-1  no-information control: the same, with each dream episode's targets taken from another
#                                      dream episode (dream_shuffle): same compute and loss form, no information
#   _n+4..5  rccdreamgate60k_lr4_s0-1  delayed start: dreams begin after the first test whose train_plugin (the Generator's
#                                      plug-in accuracy on the last training batch) is >= .3 (twins: 9,000 / 7,000 episodes),
#                                      so an uninformative early world model cannot hold the escape back
#   _n+6..8 / _n+9..11  rccdreamb1k_lr4_s0-2 / rccclfdreamb1k_lr4_s0-2  Stage 2: batch 1,000, 120k (twins rccb1k_lr4 /
#                                      rccclfb1k_lr4); launch only if >= 2 of 3 Generator-route twins leave chance by 40k
#   _n+12..13 rccclfdream60k_lr4_s0-1  HARM test (not a specificity control): Classifier route + dream at batch 10k, 60k
#                                      (twins rccclf60k_lr4_s0-1); its Generator dreams in a code unrelated to the world, and
#                                      a warm start from a trained Classifier-route net lost .35 accuracy within 32 episodes
# GPU: the batch-10k dream tasks need one GPU each (not runs_pair.sbatch) and > 14 h: runs_single.sbatch (24 h, --budget-h
# 22); records carry gpu_max_mem_gb. Check memory with one short run on one GPU before the batch-10k launch.
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
_dream = lambda twin, k, **kw: dict(_cfg[twin][1], dream_weight=1.0, dream_seed=k, **kw)
TASKS += [(f"rccdream60k_lr4_s{_k}", _dream(f"rcc60k_lr4_s{_k}", _k), _k) for _k in range(2)]
TASKS += [(f"rccdreamshuf60k_lr4_s{_k}", _dream(f"rcc60k_lr4_s{_k}", _k, dream_shuffle=True), _k) for _k in range(2)]
TASKS += [(f"rccdreamgate60k_lr4_s{_k}", _dream(f"rcc60k_lr4_s{_k}", _k, dream_start=10**12, dream_gate=.3), _k)
          for _k in range(2)]
TASKS += [(f"{_r}dreamb1k_lr4_s{_k}", _dream(f"{_r}b1k_lr4_s{_k}", _k), _k) for _r in ("rcc", "rccclf") for _k in range(3)]
TASKS += [(f"rccclfdream60k_lr4_s{_k}", _dream(f"rccclf60k_lr4_s{_k}", _k), _k) for _k in range(2)]
assert len(TASKS) == _n + 14 and len({t[0] for t in TASKS}) == len(TASKS)
# ---- end dream pilot ----------------------------------------------------------------------------------------------------
# ---- 11-13 imagination capability test ("can the network teach a NEW Classifier from imagination alone?") ---------------
# A trained RCC (the RCC's own feedback, mode RL) is loaded read-only (TEACH_FROM: its environments and weights; a fresh log
# from episode 0, no RESUME log copy); its embeddings and Generator are FROZEN (freeze_generator) and its Classifier is
# replaced by fresh weights (classifier_reinit = 7000 + seed: the same fresh Classifier for every condition and route of one
# seed), the only module that trains. Sources matched in real-data accuracy on novel environments (net, mean of the last 3
# tests): Generator route rcc60k_lr4_s0-3 (.686 / .697 / .698 / .695; plug-in .693-.705), Classifier route
# rccclf120k_lr4_s0-3 (.698 / .708 / .710 / .695; plug-in .133-.142). Conditions:
#   imagine_{gen,clf}_s{k}  (D) dream only: the fresh Classifier's ONLY loss is dream_backward's both-variable cross-entropy
#                           on batches dreamed by the frozen Generator (dream_only, dream_weight 1, held-out test-state
#                           variables masked); no real loss, no real-batch gradient (the real batch is generated and run
#                           forward without gradients, for the logs only)
#   teach_{gen,clf}_s{k}    (R) real-label control: oracle_loss's form on the REAL training batch, true values of both
#                           variables, test-state variables masked as in the dream (oracle_teach); no dreams
#   rl_{gen,clf}_s{k}       (F) the RCC's own feedback: RL_loss on the real batch (one reward bit on one sampled goal
#                           answer), no teacher, no dreams: the comparator for "does imagination beat the RCC's feedback?"
# Same everything else: batch 1,000, classifier_LR 1e-4, entropy .01, 15,062 episodes = exactly 15,000 training iterations
# (tests at 0, 250, ..., 15,000 and the last episode 15,061, on the novel environments), gpu_inference off (cluster CPU:
# runs_cpu.sbatch). Logs: net / joint / naive (+ FR fifths), plugin / kl_env of the frozen Generator (constant up to the
# test batch), dream_loss / dream_acc (D), teach_loss / teach_acc (R). Predictions, stated before launch: D-Generator-route
# rises toward its plug-in (.69-.71), which is its ceiling (Bayes under the learned likelihood is the best possible reader
# of the dreams); D-Classifier-route stays at or below its plug-in (~.13), i.e. near chance: its own Generator is at chance
# (Bayes on its own dreams .23), so this outcome is predictable from the plug-in and is no evidence about the codes: a
# generative model trained on the Classifier route's codes (an extra Generator loss) would restore imagination; R rises in
# both routes (the Classifier route's codes are readable by a Classifier: its own Classifier is at .70); F rises more
# slowly than R (one bit per episode). Order = launch priority: _n+0..3 imagine_gen, _n+4..7 imagine_clf, _n+8..11
# teach_gen, _n+12..15 teach_clf, _n+16..19 rl_gen, _n+20..23 rl_clf (s0-3 each). Launch by name (runs_cpu.sbatch).
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
TEACH_FROM = {}
_TEACH = dict(batch_num=1000, episodes=15062, gpu_inference=False, freeze_generator=True)
for _cond, _over in (("imagine", dict(dream_only=True, dream_weight=1.0)), ("teach", dict(oracle_teach=True)), ("rl", {})):
    for _route, _src in (("gen", "rcc60k_lr4_s{}"), ("clf", "rccclf120k_lr4_s{}")):
        for _k in range(4):
            _name = f"{_cond}_{_route}_s{_k}"
            TASKS.append((_name, dict(_cfg[_src.format(_k)][1], **_TEACH, **_over, classifier_reinit=7000 + _k,
                                      **(dict(dream_seed=_k) if _cond == "imagine" else {})), _k))
            TEACH_FROM[_name] = _src.format(_k)
assert len(TASKS) == _n + 24 and len({t[0] for t in TASKS}) == len(TASKS) and not set(TEACH_FROM) & set(RESUME)
# ---- end imagination capability test -------------------------------------------------------------------------------------
# ---- 11-14 batch 1,000 to convergence (John: "please continue the batch 1K for both routes. I'd like to see them both
# converge. This is high priority, so running many seeds and long training is needed") ---------------------------------
# rccb1k_lr4_s0-5 / rccclfb1k_lr4_s0-5 continued from their 120k nets (RESUME: same environments, the source's log first,
# Adam restarts) to 1,000,000 iterations; stop a converged run early with an empty file runs/STOP_<name> (the runner saves
# its net at the next test). At 120k: Generator route left chance in 4 of 5 seeds (68.5-93k), Classifier route in 0 of 4.
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
for _r in ("rcc", "rccclf"):
    for _k in range(6):
        TASKS.append((f"{_r}b1k1m_lr4_s{_k}", dict(_cfg[f"{_r}b1k_lr4_s{_k}"][1], episodes=1_000_000), _k))
        RESUME[f"{_r}b1k1m_lr4_s{_k}"] = f"{_r}b1k_lr4_s{_k}"
assert len(TASKS) == _n + 12 and len({t[0] for t in TASKS}) == len(TASKS)
# ---- 11-14 fair imagination pilot (John: "for the dreaming with the classifier, we'd need to make it fair, maybe training
# the classifier route with a shuffled channels, or whatever makes it most likely to learn the right representation") ------
# Sources = the runs trained with the channel order shuffled in every episode (readin_perm, tasks 185-192; ORACLE feedback,
# the only shuffled-channel nets that exist): there the Classifier route's seed 1 learned (.66 / .67 at lr 1e-4 / 1e-3) AND
# its own Generator reads its codes (plug-in .71 / .70; the default Classifier route's plug-in is .13), seed 0 never left
# chance (.11; plug-in .12); Generator route both seeds (.68-.69; plug-in .72-.73). Same teaching protocol as the capability
# test above (frozen net, fresh Classifier, batch 1,000, 15,000 iterations, test-state variables masked) except the source
# config (oracle mode, readin_perm: the fresh Classifier also reads shuffled channels); classifier_LR fixed at 1e-4 for
# every source. Prediction: D-Classifier route s1 rises like D-Generator route (its Generator works); s0 stays at chance.
_PERM = dict(pgen4="genperm20k_lr4_s{}", pclf4="clfperm20k_lr4_s{}", pgen3="genperm10k_lr1e3_s{}",
             pclf3="clfperm10k_lr1e3_s{}")
_n = len(TASKS)
for _cond, _over in (("imagine", dict(dream_only=True, dream_weight=1.0)), ("teach", dict(oracle_teach=True))):
    for _k in (1, 0):
        for _route, _src in _PERM.items():
            _name = f"{_cond}_{_route}_s{_k}"
            TASKS.append((_name, dict(_cfg[_src.format(_k)][1], **_TEACH, **_over, classifier_LR=1e-4, dream_mask_test=True,
                                      classifier_reinit=7000 + _k, **(dict(dream_seed=_k) if _cond == "imagine" else {})), _k))
            TEACH_FROM[_name] = _src.format(_k)
assert len(TASKS) == _n + 16 and len({t[0] for t in TASKS}) == len(TASKS) and not set(TEACH_FROM) & set(RESUME)
# ---- end fair imagination pilot -------------------------------------------------------------------------------------------
# 11-13 15:20 IST (John: "for C = 3, i think let's just keep the classifier routes, if a seed fails completely then there is
# hope"): the C = 3 Classifier-route seeds 1-2 continue from their 60k nets to 120k (RESUME, as seed 0 = task 222, which
# left chance only at ~45k and reached Naive's level at 83k, so 60k cannot show a complete failure). Local GPUs, chained
# behind their 60k runs (tasks 200 / 235); launch by name. The Generator route's seed 2 (task 234) was stopped at 20.5k.
# 11-13 15:20 IST seed 2 SCRAPPED (John: "scrap seed 2 so other things can finish faster. If seed 1 fails then we can add
# more seeds"): task 235 stopped 12 min after its start; rccclfC3_120k_lr4_s2 NOT RUN.
# 11-13 17:32 IST seed 1 CANCELLED too (John: "okay yea let's cancel the C = 3 seed"): task 200 killed at 38.75k (.152; seed 0
# .133 there; its process predates the STOP-file check, so no net saved), its chain killed; rccclfC3_120k_lr4_s1 NOT RUN.
_n = len(TASKS)
_cfg = {t[0]: t for t in TASKS}
for _k in (1, 2):
    _src = f"rccclfC3_60k_lr4_s{_k}"
    TASKS.append((f"rccclfC3_120k_lr4_s{_k}", dict(_cfg[_src][1], episodes=120000), _cfg[_src][2]))
    RESUME[TASKS[-1][0]] = _src
assert len(TASKS) == _n + 2 and len({t[0] for t in TASKS}) == len(TASKS) and not set(TEACH_FROM) & set(RESUME)
# ---- 11-14 more batch-1,000 seeds on the cluster (John: batch 1,000 "should jump the queue"; many seeds) -----------------
# Fresh runs of seeds 6-11, both routes, configured as rccb1k_lr4 / rccclfb1k_lr4 (the RCC's own feedback, gen_zflow, lr 1e-4,
# batch 1,000, tests every 250 on 1,000 novel-environment episodes) but 1,000,000 iterations from the start. The 120k nets of
# seeds 0-5 stay local (no main/DATA on the cluster), so their continuations (290-301) run on the local GPUs. Packed several
# runs per GPU (runs_pack.sbatch), each seed's two routes in the same job.
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
for _k in range(6, 12):
    for _r in ("rcc", "rccclf"):
        TASKS.append((f"{_r}b1k1m_lr4_s{_k}", dict(_cfg[f"{_r}b1k_lr4_s0"][1], episodes=1_000_000), _k))
assert len(TASKS) == _n + 12 and len({t[0] for t in TASKS}) == len(TASKS)
# ---- 11-14 shuffled-channel RCC under its own feedback (LOW priority, John: "checked where compute permits without
# interfering with other things"): the sources a FAIR imagination test needs. readin_perm (channel order shuffled in every
# episode) made the oracle-feedback Classifier route's codes readable by its own Generator (clfperm s1: plug-in .70-.71 vs
# .13 for the default Classifier route; s0 never learned). Here: mode RL, gen_zflow, lr 1e-4, batch 10,000, 100k iterations,
# seeds 0-3, (Generator route, Classifier route) paired per seed on one GPU (runs_pair_long.sbatch), queued last.
_n = len(TASKS)
for _k in range(4):
    for _r, _over in (("rcc", {}), ("rccclf", dict(embedding_grad="classifier"))):
        TASKS.append((f"{_r}perm100k_lr4_s{_k}", dict(_cfg[f"rcc60k_lr4_s{_k}"][1], readin_perm=True, episodes=100_000, **_over),
                      _k))
assert len(TASKS) == _n + 8 and len({t[0] for t in TASKS}) == len(TASKS)
# 11-13 18:20 resumes of batch-1,000 runs moved between local GPUs (GPU 1 had 7 processes at 3.2 it/s each, GPU 0 two at
# 11 it/s, after the C = 3 runs were cancelled): a run is paused with runs/STOP_<name> (net saved) and resumed under a new
# name on the other GPU (RESUME: its log first, Adam restarts). rccb1k1mR_lr4_s{k}: the Generator route (still pauses on
# leaving chance, PAUSE_AT_ESCAPE); rccclfb1k1mR_lr4_s{k}: the Classifier route; both from the rcc*b1k1m_lr4 runs, to 1M.
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
for _r in ("rcc", "rccclf"):
    for _k in range(12):
        TASKS.append((f"{_r}b1k1mR_lr4_s{_k}", dict(_cfg[f"{_r}b1k1m_lr4_s{_k}"][1]), _k))
        RESUME[f"{_r}b1k1mR_lr4_s{_k}"] = f"{_r}b1k1m_lr4_s{_k}"
assert len(TASKS) == _n + 24 and len({t[0] for t in TASKS}) == len(TASKS)
# 11-13 20:15 rccb1k1mR2_lr4_s{k}: the Generator route resumed once more (from the R run), e.g. to restart a run that still
# carried the old ">= .2" pause rule (processes keep the code they started with).
_n = len(TASKS)
for _k in range(12):
    TASKS.append((f"rccb1k1mR2_lr4_s{_k}", dict(_cfg[f"rccb1k1m_lr4_s{_k}"][1]), _k))
    RESUME[f"rccb1k1mR2_lr4_s{_k}"] = f"rccb1k1mR_lr4_s{_k}"
assert len(TASKS) == _n + 12 and len({t[0] for t in TASKS}) == len(TASKS)
# 11-13 23:30 ONLINE DREAMING at batch 1,000 (John: "batch 1K online dreaming for classifier learning acceleration under both
# routes (with shuffled channels for the classifier)"; no oracle feedback anywhere: the RCC's own feedback, plus dreams from
# the network's own Generator). During normal training the Classifier also learns, every iteration, from one batch its own
# Generator dreams (both variables labelled by the dreamer; model_backward.dream_sample / dream_backward), from the first test
# whose train_plugin (the Generator's own world model on the last training batch) is >= .3 (dream_gate; the same rule for
# both routes, so an uninformative dreamer does not steer the Classifier). Seeds 0-2, batch 1,000, lr 1e-4, to 1M (STOP
# files end them). Arms: rccdreamb1k1m = Generator route + dreams (no-dream twins: rccb1k_lr4 s0-2, same seeds);
# rccclfpermb1k1m = Classifier route with the channel order shuffled every episode (readin_perm), no dreams;
# rccclfpermdreamb1k1m = the same + dreams. One seed's three runs per GPU (runs_pack.sbatch).
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
_gate = dict(dream_weight=1.0, dream_start=10**12, dream_gate=.3)
for _k in range(3):
    TASKS += [(f"rccdreamb1k1m_lr4_s{_k}", dict(_cfg[f"rccb1k_lr4_s{_k}"][1], episodes=1_000_000, dream_seed=_k, **_gate), _k),
              (f"rccclfpermb1k1m_lr4_s{_k}", dict(_cfg[f"rccclfb1k_lr4_s{_k}"][1], episodes=1_000_000, readin_perm=True), _k),
              (f"rccclfpermdreamb1k1m_lr4_s{_k}", dict(_cfg[f"rccclfb1k_lr4_s{_k}"][1], episodes=1_000_000, readin_perm=True,
                                                     dream_seed=_k, **_gate), _k)]
assert len(TASKS) == _n + 9 and len({t[0] for t in TASKS}) == len(TASKS)
# 11-14 02:05 SHUFFLED CHANNELS, both routes (John: "move all dream compute to the shuffling classifier route check"; dreaming
# saved ~half the episodes but cost 1.5-1.8x per iteration, so the dream arms were stopped: STOP files for
# rccdreamb1k1m_lr4_s0 / rccclfpermdreamb1k1m_lr4_s0, the queued seed 1-2 packs cancelled). Question: does the Classifier
# route stay worse when the channel order is shuffled every episode (readin_perm), so that accuracy has to come from the
# channel-invariant structure? rccclfpermb1k1m_lr4_s3 = a 4th Classifier-route seed (with s0-2, tasks 377/380/383);
# rccpermb1k1m_lr4_s0-2 = the Generator route with the same shuffling (the control: shuffling alone could slow both).
# Batch 1,000, lr 1e-4, the RCC's own feedback, to 1M; the Generator-route runs pause once past Naive (PAUSE_AT_ESCAPE),
# like the other batch-1,000 Generator-route runs. Packs: "380:383:385 386:387:388" (runs_pack.sbatch).
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
TASKS += [(f"rccclfpermb1k1m_lr4_s3", dict(_cfg["rccclfb1k_lr4_s3"][1], episodes=1_000_000, readin_perm=True), 3)]
TASKS += [(f"rccpermb1k1m_lr4_s{_k}", dict(_cfg[f"rccb1k_lr4_s{_k}"][1], episodes=1_000_000, readin_perm=True), _k)
          for _k in range(3)]
assert _n == 385 and len(TASKS) == _n + 4 and len({t[0] for t in TASKS}) == len(TASKS)
# 11-14 05:40 NO K.Q BOTTLENECK (John: "wouldnt the fair comparison be to remove the bottleneck from both the classifier and
# the generator?"; "check 1 seed of each route and see if it is promising"). z_raw (model_architecture.model_Z_num): both
# modules read the active variables' down-scaled K and Q vectors (2 * C * KQ_dim = 120 numbers per channel) instead of their
# per-channel dot products, so neither has the environment's likelihood form built in; gen_zflow keeps its layer norms but
# not its structured start of the Z read-in. Shuffled channels (readin_perm: the fair condition), batch 1,000, lr 1e-4, the
# RCC's own feedback, seed 0 per route, to 1M; the Generator route pauses past Naive (PAUSE_AT_ESCAPE), the Classifier route
# at .60 (scratchpad pause_watcher). Pack "389:390" (runs_pack.sbatch).
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
TASKS += [("rccrawpermb1k1m_lr4_s0", dict(_cfg["rccpermb1k1m_lr4_s0"][1], z_raw=True), 0),
          ("rccclfrawpermb1k1m_lr4_s0", dict(_cfg["rccclfpermb1k1m_lr4_s0"][1], z_raw=True), 0)]
assert _n == 389 and len(TASKS) == _n + 2 and len({t[0] for t in TASKS}) == len(TASKS)
# 11-14 09:25 MORE SHUFFLED-CHANNEL SEEDS (John: "pause the b / f runs, it looks like we should use that compute to collect
# more shuffled channel runs"; the unshuffled batch-1,000 runs are paused by STOP files). Same configs as the shuffled seeds
# above, new seeds: Classifier route s4-8 (tasks 391-395), Generator route s3-7 (396-400; pause past Naive). Cluster pack
# "391:392:393 396:397:398"; tasks 394, 395, 399, 400 on the local GPUs.
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
TASKS += [(f"rccclfpermb1k1m_lr4_s{_k}", dict(_cfg["rccclfpermb1k1m_lr4_s0"][1]), _k) for _k in range(4, 9)]
TASKS += [(f"rccpermb1k1m_lr4_s{_k}", dict(_cfg["rccpermb1k1m_lr4_s0"][1]), _k) for _k in range(3, 8)]
assert _n == 391 and len(TASKS) == _n + 10 and len({t[0] for t in TASKS}) == len(TASKS)
# 11-14 09:45 NO WORLD STRUCTURE (John: "we should remove the downscale and the normalization loss in this 'no world
# structure' setting"): z_full on top of z_raw (model_architecture.model_Z_num): per channel the active variables' FULL
# embeddings (2 * C * hid_dim = 4,000 numbers; no 1000 -> KQ_dim down-scaling, so no match to the environment's embedding
# size) and no norm loss (model_backward.backward_pass); the read-in's Z part is applied once per episode (model_forward.
# RNN_readin). Otherwise as rccrawpermb1k1m / rccclfrawpermb1k1m (shuffled channels, batch 1,000, own feedback, seed 0),
# which this pair replaces (STOP files, nets saved). Pack "401:402".
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
TASKS += [("rccfullpermb1k1m_lr4_s0", dict(_cfg["rccrawpermb1k1m_lr4_s0"][1], z_full=True), 0),
          ("rccclffullpermb1k1m_lr4_s0", dict(_cfg["rccclfrawpermb1k1m_lr4_s0"][1], z_full=True), 0)]
assert _n == 401 and len(TASKS) == _n + 2 and len({t[0] for t in TASKS}) == len(TASKS)
# 11-14 14:40 NO WORLD STRUCTURE + ATTENTION (John: "make the first layer from embeddings a transformer attention layer (in
# the no world structure version)"): z_attn on top of z_full (model_architecture.default_generator): per channel, a
# pre-LayerNorm 16-head self-attention layer with a residual connection over the active variables' tokens [K_v; Q_v]
# (2,000 numbers each), shared by the channels and routed with the embeddings. Otherwise as rccfullpermb1k1m /
# rccclffullpermb1k1m (shuffled channels, batch 1,000, own feedback, seed 0), which this pair replaces on its GPU (STOP
# files at ~300k, nets saved, both routes still at chance). Pack "403:404".
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
TASKS += [("rccattnpermb1k1m_lr4_s0", dict(_cfg["rccfullpermb1k1m_lr4_s0"][1], z_attn=True), 0),
          ("rccclfattnpermb1k1m_lr4_s0", dict(_cfg["rccclffullpermb1k1m_lr4_s0"][1], z_attn=True), 0)]
assert _n == 403 and len(TASKS) == _n + 2 and len({t[0] for t in TASKS}) == len(TASKS)
# 11-14 14:50 IST HIGHER CLASSIFIER LR WITH SHUFFLED CHANNELS (John: "start a pair of batch 1K shuffling seeds that succeeded
# and see if the shuffling allows for a higher classifier lr (1e-3)"): rcc{,clf}permb1k1m_lr4_s3's configs (seed 3: both
# routes passed Naive, Generator 109k, Classifier 154k) with classifier_LR 1e-3; generator_LR stays 1e-4, which is also the
# embeddings' lr in BOTH routes (model_architecture.default_generator), so only the Classifier's own weights change. No pause
# rule (stop by hand): later instability is part of the question. Local GPUs, one run per GPU.
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
TASKS += [("rccpermb1k1m_lrc1e3_s3", dict(_cfg["rccpermb1k1m_lr4_s3"][1], classifier_LR=1e-3), 3),
          ("rccclfpermb1k1m_lrc1e3_s3", dict(_cfg["rccclfpermb1k1m_lr4_s3"][1], classifier_LR=1e-3), 3)]
assert _n == 405 and len(TASKS) == _n + 2 and len({t[0] for t in TASKS}) == len(TASKS)
# 11-14 20:00 IST (John: "switch the classifier lr 1e-3 runs to both classifier and generator having lr 1e-3"): the same
# seed-3 shuffled configs with classifier_LR AND generator_LR 1e-3 (so the embeddings, which follow generator_LR in both
# routes, are at 1e-3 too: the oracle lr-1e-3 runs' setting under the RCC's own feedback). They replace tasks 405-406 (STOP
# files, nets saved: Generator route .67 at 224k after passing Naive at 60.5k; Classifier route at chance at 174k). Fresh
# start, no pause rule, local GPUs 0 / 1.
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
TASKS += [("rccpermb1k1m_lr1e3_s3", dict(_cfg["rccpermb1k1m_lr4_s3"][1], classifier_LR=1e-3, generator_LR=1e-3), 3),
          ("rccclfpermb1k1m_lr1e3_s3", dict(_cfg["rccclfpermb1k1m_lr4_s3"][1], classifier_LR=1e-3, generator_LR=1e-3), 3)]
assert _n == 407 and len(TASKS) == _n + 2 and len({t[0] for t in TASKS}) == len(TASKS)
# 11-14 20:30 IST (John: "pause all runs that surpass .6 accuracy, so we can give the compute to more seeds"): batch-10,000
# shuffled seeds 4-7, configs identical to rcc{,clf}perm100k_lr4_s0-3 (tasks 332-339) except the seed (rcc60k_lr4's
# config is the same for every seed). Tasks 409-416, pairs "409:410 411:412 413:414 415:416" (runs_pair_long.sbatch).
# Every running run now pauses once the mean of its last 5 tests is >= .60 (scratchpad pause_watcher.py).
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
for _k in range(4, 8):
    for _r in ("rcc", "rccclf"):
        TASKS.append((f"{_r}perm100k_lr4_s{_k}", dict(_cfg[f"{_r}perm100k_lr4_s0"][1]), _k))
assert _n == 409 and len(TASKS) == _n + 8 and len({t[0] for t in TASKS}) == len(TASKS)
# 11-14 20:40 IST (John: "my top priority is to see if shuffling indeed guarantees some classifier route seeds never succeed
# within a large range where all generator routes definitely succeed. Can you also queue more batch 1K high learning rate
# seeds?"): rcc{,clf}permb1k1m_lr1e3 (classifier_LR = generator_LR = 1e-3, shuffled channels, batch 1,000, own feedback) for
# seeds 0, 1, 2, 4, 5 (seed 3 = tasks 407-408). Tasks 417-426 (Gen, Clf per seed). Seed 4 (423-424) on the local GPUs,
# seeds 0+1 and 2+5 as cluster packs "417:418:419:420 421:422:425:426". Pause at .60 (watcher), no PAUSE_AT_ESCAPE.
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
for _k in (0, 1, 2, 4, 5):
    TASKS += [(f"rccpermb1k1m_lr1e3_s{_k}", dict(_cfg[f"rccpermb1k1m_lr4_s{_k}"][1], classifier_LR=1e-3, generator_LR=1e-3), _k),
              (f"rccclfpermb1k1m_lr1e3_s{_k}", dict(_cfg[f"rccclfpermb1k1m_lr4_s{_k}"][1], classifier_LR=1e-3, generator_LR=1e-3),
               _k)]
assert _n == 417 and len(TASKS) == _n + 10 and len({t[0] for t in TASKS}) == len(TASKS)
# 11-14 23:00 IST (John: "drop the 1e-3 runs entirely and donate that compute to more seeds of c and f" -> "go"): the lr-1e-3
# runs are stopped / cancelled. More seeds of the shuffled-channel settings, configs identical to the existing seeds:
#   c (batch 1,000): Generator route s8-9 (tasks 427-428; pause past Naive like s0-7, PAUSE_AT_ESCAPE), Classifier route
#     s9-10 (429-430; watcher pause at .60) -- local GPUs, next to the failing Classifier s7-8
#   f (batch 10,000): rcc{,clf}perm100k_lr4_s8-11 (431-438), pairs "431:432 433:434 435:436 437:438" (runs_pair_long.sbatch)
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
TASKS += [(f"rccpermb1k1m_lr4_s{_k}", dict(_cfg["rccpermb1k1m_lr4_s0"][1]), _k) for _k in (8, 9)]
TASKS += [(f"rccclfpermb1k1m_lr4_s{_k}", dict(_cfg["rccclfpermb1k1m_lr4_s0"][1]), _k) for _k in (9, 10)]
for _k in range(8, 12):
    for _r in ("rcc", "rccclf"):
        TASKS.append((f"{_r}perm100k_lr4_s{_k}", dict(_cfg[f"{_r}perm100k_lr4_s0"][1]), _k))
assert _n == 427 and len(TASKS) == _n + 12 and len({t[0] for t in TASKS}) == len(TASKS)
# 11-15 01:25 IST COLUMN c TO A COMMON LENGTH (John: "for the 3rd column, let's train all seeds to the point" + "please start
# this with 1 exception, keep running that one 10K batch classifier route seed that hasn't left chance"): the batch-1,000
# shuffled-channel seeds are resumed (RESUME: their log first, their net, a fresh Adam) to 450,000 iterations = the failure
# line of the status figure (3x the slowest Generator Naive crossing, 446k), with no pause rule:
#   rccpermb1k450_lr4_s0-9 (Generator route, from rccpermb1k1m_lr4_s0-9), rccclfpermb1k450_lr4_s0-6, s9-10 (Classifier route;
#   s7-8 are already past 450k and stop there). Each resumes on the machine that holds its net: s0-5 / Clf s0-6 on the
#   cluster, Generator s6-9 / Classifier s9-10 locally.
# rccclfperm300k_lr4_s2: the batch-10,000 shuffled Classifier-route seed 2 (at chance at ~90k) continued from its 100k end
# to 300k ("let's see how that one goes"), no pause rule.
_cfg, _n = {t[0]: t for t in TASKS}, len(TASKS)
for _k in range(10):
    TASKS.append((f"rccpermb1k450_lr4_s{_k}", dict(_cfg[f"rccpermb1k1m_lr4_s{_k}"][1], episodes=450_000), _k))
    RESUME[f"rccpermb1k450_lr4_s{_k}"] = f"rccpermb1k1m_lr4_s{_k}"
for _k in list(range(7)) + [9, 10]:
    TASKS.append((f"rccclfpermb1k450_lr4_s{_k}", dict(_cfg[f"rccclfpermb1k1m_lr4_s{_k}"][1], episodes=450_000), _k))
    RESUME[f"rccclfpermb1k450_lr4_s{_k}"] = f"rccclfpermb1k1m_lr4_s{_k}"
TASKS.append(("rccclfperm300k_lr4_s2", dict(_cfg["rccclfperm100k_lr4_s2"][1], episodes=300_000), 2))
RESUME["rccclfperm300k_lr4_s2"] = "rccclfperm100k_lr4_s2"
assert _n == 439 and len(TASKS) == _n + 20 and len({t[0] for t in TASKS}) == len(TASKS)
# 11-15 01:40 IST (John, pointing at the batch-10,000 shuffled Classifier-route seed 7, paused at chance at 15.5k: "sorry i
# meant to leave this run unpaused"): resumed to its original 100k cap, paired with seed 2's continuation (task 458) on one
# GPU (runs_pair_long.sbatch "458:459"); the .6 pause applies (watcher), as for every column-d seed.
_cfg = {t[0]: t for t in TASKS}
TASKS.append(("rccclfperm100kR_lr4_s7", dict(_cfg["rccclfperm100k_lr4_s7"][1]), 7))
RESUME["rccclfperm100kR_lr4_s7"] = "rccclfperm100k_lr4_s7"
assert len(TASKS) == 460 and TASKS[459][0] == "rccclfperm100kR_lr4_s7"
# 11-15 03:15 IST (Sat 20:15 FL; John: "let's stop the 10K batch and put all of that compute on the 1k"): the last running
# batch-10,000 run (Classifier seed 7, task 459) is stopped (net saved) and its GPU takes two NEW batch-1,000 shuffled
# Classifier-route seeds 11-12, fresh, straight to 450,000 iterations (no pause rule), same config as every
# rccclfpermb1k1m_lr4 seed (the episode count only sets the loop length). Pack "460:461" (runs_pack.sbatch).
_cfg = {t[0]: t for t in TASKS}
TASKS += [(f"rccclfpermb1k450_lr4_s{_k}", dict(_cfg["rccclfpermb1k1m_lr4_s0"][1], episodes=450_000), _k) for _k in (11, 12)]
assert len(TASKS) == 462 and TASKS[461][0] == "rccclfpermb1k450_lr4_s12"
# 11-15 03:40 IST (Sat 20:40 FL; John: "are we able to get 13 of the generator as well? It would be best to have it
# balanced"): NEW batch-1,000 shuffled Generator-route seeds 10-12, fresh, straight to 450,000 (no pause rule; NOT in
# PAUSE_AT_ESCAPE), same config as every rccpermb1k1m_lr4 seed. One run per GPU (runs_pack.sbatch "462 463 464"), queued to
# take the cluster GPUs the column-c packs free (~02:30 / 03:00 A100, ~03:50 V100S FL); alone on an A100 ~39 it/s.
_cfg = {t[0]: t for t in TASKS}
TASKS += [(f"rccpermb1k450_lr4_s{_k}", dict(_cfg["rccpermb1k1m_lr4_s0"][1], episodes=450_000), _k) for _k in (10, 11, 12)]
assert len(TASKS) == 465 and TASKS[464][0] == "rccpermb1k450_lr4_s12"
# 11-15 11:35 IST (Sun 04:35 FL; John: "While compute frees up, please continue training the seeds, maybe we can get them
# all trained to the length of the longest run?"): column c to 643,000 iterations = the longest run (Classifier seed 7,
# stopped at 642,750). RESUME (the source's log first, its net, a fresh Adam), no pause rule:
#   rccpermb1k643_lr4_s0-12 from rccpermb1k450_lr4_s0-12 (tasks 465-477)
#   rccclfpermb1k643_lr4_s{0-6,8-12} from rccclfpermb1k450_lr4 (s8 from rccclfpermb1k1m_lr4_s8, stopped at 612,750)
#     (tasks 478-489: s0-6 = 478-484, s8 = 485, s9 = 486, s10 = 487, s11 = 488, s12 = 489)
# Each runs where its net is (cluster: Generator 0-5, 10-12, Classifier 0-6, 11, 12; local: Generator 6-9, Classifier
# 8-10), started only as GPUs free, so the 450k set (the deadline's common length) is not slowed.
_cfg = {t[0]: t for t in TASKS}
for _k in range(13):
    TASKS.append((f"rccpermb1k643_lr4_s{_k}", dict(_cfg[f"rccpermb1k450_lr4_s{_k}"][1], episodes=643_000), _k))
    RESUME[f"rccpermb1k643_lr4_s{_k}"] = f"rccpermb1k450_lr4_s{_k}"
for _k in (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12):
    _src = "rccclfpermb1k1m_lr4_s8" if _k == 8 else f"rccclfpermb1k450_lr4_s{_k}"
    TASKS.append((f"rccclfpermb1k643_lr4_s{_k}", dict(_cfg[_src][1], episodes=643_000), _k))
    RESUME[f"rccclfpermb1k643_lr4_s{_k}"] = _src
assert len(TASKS) == 490 and TASKS[489][0] == "rccclfpermb1k643_lr4_s12"
# 11-15 15:10 IST (Sun 08:10 FL; John: "Yes please set it up to continue until 700K, but keep this figure plotting everything
# only up until the shortest seed's duration"): column c to 700,000 iterations, every seed. RESUME from the 643k runs as each
# finishes (the source's log first, its net, a fresh Adam: one more restart, at 643,000, so the 643k data stay as they are):
#   rccpermb1k700_lr4_s0-12 from rccpermb1k643_lr4_s0-12 (tasks 490-502 = 490 + seed)
#   rccclfpermb1k700_lr4_s0-12 from rccclfpermb1k643_lr4 (s7 from rccclfpermb1k1m_lr4_s7, stopped at 642,750)
#     (tasks 503-515 = 503 + seed)
# Each runs where its source ran (cluster: Generator 0-5, 10-12, Classifier 0-6, 11, 12, packs queued afterany their 643k
# packs; local: Generator 6-9, Classifier 7-10, scratchpad smallbatch/launch_700.py). Same config as the 450k sources.
_cfg = {t[0]: t for t in TASKS}
for _k in range(13):
    TASKS.append((f"rccpermb1k700_lr4_s{_k}", dict(_cfg[f"rccpermb1k450_lr4_s{_k}"][1], episodes=700_000), _k))
    RESUME[f"rccpermb1k700_lr4_s{_k}"] = f"rccpermb1k643_lr4_s{_k}"
for _k in range(13):
    _src = "rccclfpermb1k1m_lr4_s7" if _k == 7 else f"rccclfpermb1k643_lr4_s{_k}"
    _base = "rccclfpermb1k1m_lr4_s7" if _k == 7 else f"rccclfpermb1k643_lr4_s{_k}"
    TASKS.append((f"rccclfpermb1k700_lr4_s{_k}", dict(_cfg[_base][1], episodes=700_000), _k))
    RESUME[f"rccclfpermb1k700_lr4_s{_k}"] = _src
assert len(TASKS) == 516 and TASKS[502][0] == "rccpermb1k700_lr4_s12" and TASKS[515][0] == "rccclfpermb1k700_lr4_s12"
assert len({t[0] for t in TASKS}) == len(TASKS)
# 11-15 15:55 IST (Sun 08:55 FL; John, pointing at Classifier seed 12's End dot below the other learned Classifier seeds:
# "when this lower dot is up there with the rest of the red dots, let's have that be the final episode duration and start
# training new seeds"): the 700k continuations above were cancelled before starting (cluster packs 66715255-58 scancelled,
# local launcher stopped; only Classifier seeds 7 and 8, tasks 510-511, run on to 700k). The final length X is set by
# scratchpad smallbatch/final_watch.py (X = the first test at which seed 12's 5-test mean has been >= the lowest 5-test
# mean of the other learned Classifier seeds on 3 consecutive tests; 643,000 if that never happens); every run is stopped
# (STOP file, net saved) once it reaches X. NEW seeds 13-21, both routes, fresh, the same config as every shuffled
# batch-1,000 seed (rcc{,clf}permb1k1m_lr4), no pause rule; episodes = 700,000 is only a cap: final_watch.py stops each at X.
#   rccpermb1kfin_lr4_s13-21    (tasks 516-524 = 503 + seed)
#   rccclfpermb1kfin_lr4_s13-21 (tasks 525-533 = 512 + seed)
# 11-15 16:55 IST (Sun 09:55 FL; John: "lets just have all seeds capped at 600K? Everything beyond can be stopped and we can
# put all our compute into getting more seeds to 600K as fast as possible"): the final length is 600,000 for every seed (the
# seed-12 rule above is dropped). The new seeds run straight to 600,000 (episodes = 600,000, they end by themselves), and
# seeds 22-40 are added so that every GPU keeps a queue:
#   rccpermb1kfin_lr4_s13-21 / rccclfpermb1kfin_lr4_s13-21  tasks 516-524 / 525-533 (as before, now 600,000)
#   rccpermb1kfin_lr4_s22-40 / rccclfpermb1kfin_lr4_s22-40  tasks 534-552 (= 512 + seed) / 553-571 (= 531 + seed)
# Run as one pair (both routes of a seed) per GPU: cluster pair packs queued in seed order, local GPUs by
# scratchpad smallbatch/cap600_watch.py.
_cfg = {t[0]: t for t in TASKS}
for _r in ("rcc", "rccclf"):
    for _k in range(13, 22):
        TASKS.append((f"{_r}permb1kfin_lr4_s{_k}", dict(_cfg[f"{_r}permb1k1m_lr4_s0"][1], episodes=600_000), _k))
assert len(TASKS) == 534 and TASKS[516][0] == "rccpermb1kfin_lr4_s13" and TASKS[533][0] == "rccclfpermb1kfin_lr4_s21"
for _r in ("rcc", "rccclf"):
    for _k in range(22, 41):
        TASKS.append((f"{_r}permb1kfin_lr4_s{_k}", dict(_cfg[f"{_r}permb1k1m_lr4_s0"][1], episodes=600_000), _k))
assert len(TASKS) == 572 and TASKS[534][0] == "rccpermb1kfin_lr4_s22" and TASKS[552][0] == "rccpermb1kfin_lr4_s40"
assert TASKS[553][0] == "rccclfpermb1kfin_lr4_s22" and TASKS[571][0] == "rccclfpermb1kfin_lr4_s40"
assert len({t[0] for t in TASKS}) == len(TASKS)
# ---- end 11-14 blocks -----------------------------------------------------------------------------------------------------
# 11-16 23:50 ET (John: "let's start a new default RCC trained with C = 3, with the maximally efficient batch size and all
# of the optimizations it can allow"): 572-573 rccC3b1k_lr4_s0-1 = rccC3_60k_lr4's config (own answers, gen_zflow, lr 1e-4,
# entropy .01, dense read-in) at batch 1,000. At C = 2 the smallest batch tried was the most sample-efficient (episodes to
# .60: 1k 112-125M, 10k 180-205M, 20k 270-300M; to .70: 1k 236-255M, 10k 630-678M, 20k not by 600M), and at C = 3 a 3090 Ti
# is saturated from batch 1,000 on (10.8k / 11.2k / 11.9k / 11.9k episodes/s at 1k / 2k / 4k / 10k; 500: 10.0k; scratchpad
# c3bench/bench.py). Tests every 1,000 iterations (1,000 novel episodes each): the two plug-in read-outs per test (R**C =
# 1,000 Generator calls, 0.9 s each) cost 2% of the time instead of 8% at 250. No stop rule: runs_single_long.sbatch with a
# 44 h budget; a STOP file saves the net earlier.
TASKS += [(f"rccC3b1k_lr4_s{_k}", dict(ORACLE, mode="RL", gen_zflow=True, ctx_num=3, classifier_LR=1e-4, generator_LR=1e-4,
                                      batch_num=1000, episodes=5_000_000, checkpoint_every=1000), _k) for _k in range(2)]
assert len(TASKS) == 574 and TASKS[572][0] == "rccC3b1k_lr4_s0" and len({t[0] for t in TASKS}) == len(TASKS)
# 11-18 (John: "Let's stop 1 C = 3 seed and let the other continue to 5B training episodes"): seed 1 stopped with its STOP
# file at 3.97 B episodes (net saved); seed 0 ends at its 44 h budget (~4.45 B) and 574 rccC3b1k5B_lr4_s0 resumes it
# (RESUME: its log first, its net, a fresh Adam) to the configured 5,000,000 iterations = 5 B episodes (~5.5 h on an A100),
# submitted with a dependency on seed 0's job.
TASKS.append(("rccC3b1k5B_lr4_s0", dict(TASKS[572][1]), 0))
RESUME["rccC3b1k5B_lr4_s0"] = "rccC3b1k_lr4_s0"
assert len(TASKS) == 575 and TASKS[574][0] == "rccC3b1k5B_lr4_s0" and TASKS[574][1]["episodes"] == 5_000_000
# 11-19 (John: "can we keep training it for another day?"): 575 rccC3b1k5Bmore_lr4_s0 resumes 574's net (5 B episodes;
# RESUME as above) for a 24 h budget (~2.4 B more episodes at ~100 M per hour; 10,000,000 iterations are never reached).
# Task seed 1 only moves the post-resume stream to seed 1,001 (574 used 1,000, which could redraw 574's episodes); the
# environments (K, Q from the saved net; test variables = the first test_states) and the net are seed 0's.
TASKS.append(("rccC3b1k5Bmore_lr4_s0", dict(TASKS[572][1], episodes=10_000_000), 1))
RESUME["rccC3b1k5Bmore_lr4_s0"] = "rccC3b1k5B_lr4_s0"
assert len(TASKS) == 576 and TASKS[575][0] == "rccC3b1k5Bmore_lr4_s0" and len({t[0] for t in TASKS}) == len(TASKS)
STOP_RULE_RUNS = {f"{a}_s{k}" for a in ("zflow", "default") for k in range(3)}     # seeds 3-5 run all 30k episodes
# 11-13 17:40 (John: "for the generator route it seems like some seeds still never left chance. Let's keep running generator
# seeds if they dont leave chance. The longest it takes for a generator seed to leave chance also gives the minimum we have
# to train classifier seeds until"; Classifier routes first): a Generator-route batch-1,000 run PAUSES (StopRun, net saved)
# once it is beyond Naive Bayes (John 11-13 20:10: "please only pause a generator route after it is beyond naive"; the
# first rule, 3 tests >= .2, paused rccb1k1m_lr4_s0 at .22): ESCAPE_TESTS consecutive tests whose net is above that test's
# Naive accuracy (same episodes). Resume a paused run under a new task name (RESUME); R and R2 names follow this rule too.
ESCAPE_TESTS = 3
PAUSE_AT_ESCAPE = {f"rccb1k1m{x}_lr4_s{k}" for x in ("", "R", "R2") for k in range(12)}
PAUSE_AT_ESCAPE |= {f"rccpermb1k1m_lr4_s{k}" for k in range(3)}      # 11-14 02:05: shuffled-channel Generator route
PAUSE_AT_ESCAPE |= {"rccrawpermb1k1m_lr4_s0"}                         # 11-14 05:40: the same without the K.Q bottleneck
PAUSE_AT_ESCAPE |= {f"rccpermb1k1m_lr4_s{k}" for k in range(3, 8)}   # 11-14 09:25: more shuffled Generator-route seeds
PAUSE_AT_ESCAPE |= {"rccfullpermb1k1m_lr4_s0"}                        # 11-14 09:45: no world structure, Generator route
PAUSE_AT_ESCAPE |= {"rccattnpermb1k1m_lr4_s0"}                        # 11-14 14:40: the same + attention layer
PAUSE_AT_ESCAPE |= {"rccpermb1k1m_lr4_s8", "rccpermb1k1m_lr4_s9"}     # 11-14 23:00: more shuffled batch-1,000 Generator seeds


class StopRun(Exception):
    pass


def plugin(env):
    """the learned likelihood on the episode's batch (train_bayes_generator.py's test): per episode, whether the Bayes
    posterior at T computed with the Generator's predictions at all R**C joint values (confidence 1) gets the goal right,
    and E_r KL(P(o|r) || P^(o|r)) summed over channels"""
    import torch
    m, dev, B, R, C, O = env.model, env.device, env.batch_num, env.realization_num, env.ctx_num, env.obs_num
    ones, out = torch.ones(B, C, device=dev), []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=env.device_type != "cpu"):
        m.MC_to_interactions()
        for r in range(R ** C):
            m.get_prediction(torch.tensor(np.unravel_index(r, (R,) * C), device=dev).expand(B, C).contiguous(), ones)
            out.append(m.pred_pobs.float())
    P = torch.stack(out, -1).clamp(1e-6, 1 - 1e-6)                                         # (B, O, R**C)
    L = torch.as_tensor(env.joint_likelihood, device=dev, dtype=torch.float32).reshape(B, O, -1).clamp(1e-6, 1 - 1e-6)
    kl = (L * (L / P).log() + (1 - L) * ((1 - L) / (1 - P)).log()).sum(1).mean(1)
    o = torch.as_tensor(env.obs_flat, device=dev, dtype=torch.float32)                     # (B, T, O)
    ll = torch.einsum('bto,bok->bk', o, P.log()) + torch.einsum('bto,bok->bk', 1 - o, (1 - P).log())
    post = torch.softmax(ll, -1).reshape(B, *(R,) * C)
    g = torch.as_tensor(env.goal_ind, device=dev).long()
    goal_post = torch.stack([post.sum(dim=tuple(1 + d for d in range(C) if d != c)) if C > 1 else post for c in range(C)],
                            1)[torch.arange(B, device=dev), g]                              # the goal's marginal, (B, R)
    hit = goal_post.argmax(-1) == torch.as_tensor(env.goal_value, device=dev).long()
    return hit.float().cpu().numpy(), kl.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--budget-h", type=float, default=13.5)
    ap.add_argument("--smoke", action="store_true", help="tiny batch, few episodes, output under smoke/")
    a = ap.parse_args()
    name, cfg, seed = TASKS[a.task]
    cfg = dict(cfg)
    sub = "smoke" if a.smoke else "cluster_runs"
    if a.smoke:
        cfg.update(batch_num=256, episodes=6, checkpoint_every=3,
                   state_num=cfg["state_num"] if name in TEACH_FROM else min(cfg["state_num"], 200))   # teach: the net's |S|
    os.makedirs(os.path.join(ROOT, "runs", sub), exist_ok=True)
    log_path = os.path.join(ROOT, "runs", sub, name + ".jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    # 11-14 speed patch (scratchpad speed/, verified bitwise identical on every runner path, CPU and GPU): network outputs
    # and losses of training iterations stay on the GPU until the logs need them (env_model_manager.defer_train_outputs)
    cfg.update(save_env=f"{sub}/{name}", cuda=0, defer_train_outputs=True)
    src, offset = RESUME.get(name), 0
    if src:                                   # continue the source run: same environments, its saved net, its log first
        old = [l for l in open(os.path.join(ROOT, "runs", sub, src + ".jsonl")) if l.strip()]
        offset = json.loads(old[-1])["ep"] + 1                       # the source's last test is at its episodes - 1
        cfg.update(load_env=f"/{sub}/{src}", episodes=cfg["episodes"] - (0 if a.smoke else offset))
        with open(log_path, "w") as fh:
            fh.writelines(l if l.endswith("\n") else l + "\n" for l in old)
    if name in TEACH_FROM:                    # a frozen trained net: its environments and weights, a fresh log from episode 0
        cfg.update(load_env=f"/cluster_runs/{TEACH_FROM[name]}")

    import torch
    np.random.seed(seed); torch.manual_seed(seed)
    from main.CognitiveGridworld import CognitiveGridworld

    t0, state = time.time(), dict(env=None, hits=0, train={})
    budget_s = a.budget_h * 3600
    rcc = name.startswith(("rcc", "gen30k", "clf30k", "gen60k", "clf60k", "clfshared", "genshared", "gen6k", "clf6k", "gen90k", "clf90k", "gen120k", "clf120k", "genperm", "clfperm", "genC3", "clfC3", "oracle_S")) or name in TEACH_FROM   # tests also read out the learned likelihood

    def logger(env):
        state["env"] = env
        if src and not state.get("bound"):    # e = 0 is a test episode, so no update has run yet
            state["bound"] = True
            m = env.model
            held = [p for o in vars(m).values() if isinstance(o, torch.optim.Optimizer)
                    for g in o.param_groups for p in g["params"]]
            for k in ("all_K", "all_Q"):
                cur = getattr(m, k)
                same = [p for p in held if p is not cur and p.shape == cur.shape and torch.equal(p.data, cur.data)]
                assert len(same) == 1 or any(p is cur for p in held), k
                if same:
                    setattr(m, k, same[0])
            np.random.seed(seed + 1000); torch.manual_seed(seed + 1000)
        if not env.test_set and getattr(env.model, "dream_loss", None) is not None:    # dream runs only (model_backward.dream)
            d = state.setdefault("dream", [0., 0., 0])                     # tensor sums: no GPU sync per iteration
            d[0], d[1], d[2] = d[0] + env.model.dream_loss, d[1] + env.model.dream_acc, d[2] + 1
        if not env.test_set and getattr(env.model, "oracle_teach", False):    # teaching runs with real labels only
            d = state.setdefault("teach", [0., 0., 0])
            d[0], d[1], d[2] = d[0] + env.model.teach_loss, d[1] + env.model.teach_acc, d[2] + 1
        if rcc and not env.test_set and env.e % env.checkpoint_every == env.checkpoint_every - 1:
            hit, kl = plugin(env)                    # the last training batch before the next test, after its update
            state["train"] = dict(train_plugin=float(hit.mean()), train_kl_env=float(kl.mean()))
            gate, m = cfg.get("dream_gate"), env.model
            if gate is not None and "dream_gate_ep" not in state and state["train"]["train_plugin"] >= gate:
                m.dream_start = m.dream_iter              # dream_gate runs: dreams from the next training iteration on
                state["dream_gate_ep"] = int(env.e) + 2 + offset      # e + 1 is a test episode
                print(f"=== dream gate: train_plugin {state['train']['train_plugin']:.3f} >= {gate}, dreaming from "
                      f"episode {state['dream_gate_ep']}", flush=True)
        if env.test_set:
            ok, coords = env.naive_consistent_cells()
            R = env.realization_num
            cell = torch.as_tensor(np.ravel_multi_index(tuple(env.ctx_vals.T), (R,) * env.ctx_num), device=ok.device)
            low = ok[torch.arange(env.batch_num, device=ok.device), cell].cpu().numpy()
            rec = dict(ep=int(env.e) + offset, t=round(time.time() - t0, 1), frac_low=float(low.mean()),
                       fallback=int(getattr(env, "fr_select_fallback", 0)),
                       gen_loss=float(np.mean(getattr(env, "generator_loss", np.nan))))
            per_ep = {}
            for k, acc in (("net", env.model_acc), ("joint", env.joint_acc), ("naive", env.naive_acc)):
                f = per_ep[k] = np.asarray(acc)[:, -1]
                rec[k], rec[k + "_low"], rec[k + "_high"] = float(f.mean()), float(f[low].mean()), float(f[~low].mean())
            if rcc:
                hit, kl = plugin(env)
                per_ep["plugin"] = hit
                rec.update(plugin=float(hit.mean()), plugin_low=float(hit[low].mean()), plugin_high=float(hit[~low].mean()),
                           kl_env=float(kl.mean()))
            jb = np.clip(np.asarray(env.joint_belief, dtype=np.float64)[:, -1], 1e-12, None)       # (B, C, R) at T
            nb = np.clip(np.asarray(env.naive_belief, dtype=np.float64)[:, -1], 1e-12, None)
            fifth = np.searchsorted(FR_EDGES, (jb * (np.log(jb) - np.log(nb))).sum(-1).mean(1))
            for q in range(len(FR_EDGES) + 1):
                rec[f"n_fr{q}"] = int((fifth == q).sum())
                for k, f in per_ep.items():
                    rec[f"{k}_fr{q}"] = float(f[fifth == q].mean()) if (fifth == q).any() else None
                if env.e > 0:
                    rec.update(train_net=float(env.train_accs[env.test_e - 1, -1]), **state["train"])
            if env.mode == "RL":                              # the Classifier's policy: collapse or not
                gb = env.model.classifier_goal_belief[:, -1].detach().float().clamp_min(1e-9)
                sel = env.model.classifier_goal_selection[:, -1].long()
                rec.update(pol_entropy=float(-(gb * gb.log()).sum(-1).mean()),
                           top_answer=float(torch.bincount(sel, minlength=env.realization_num).max()) / env.batch_num)
            if state.get("dream"):                            # means over the training iterations since the previous test
                d = state.pop("dream")
                rec.update(dream_loss=float(d[0]) / d[2], dream_acc=float(d[1]) / d[2], dream_n=d[2])
            if state.get("teach"):
                d = state.pop("teach")
                rec.update(teach_loss=float(d[0]) / d[2], teach_acc=float(d[1]) / d[2], teach_n=d[2])
            if getattr(env.model, "dream_weight", 0.0) > 0:
                if "dream_gate_ep" in state:
                    rec["dream_gate_ep"] = state["dream_gate_ep"]
                if torch.cuda.is_available():                 # peak GPU memory so far (the dream's second Classifier pass)
                    rec["gpu_max_mem_gb"] = round(torch.cuda.max_memory_allocated(env.device) / 2**30, 2)
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if os.path.exists(os.path.join(ROOT, "runs", "STOP_" + name)):     # 11-14: stop a run by hand, net saved
                raise StopRun("stop file runs/STOP_" + name)
            if name in PAUSE_AT_ESCAPE:                        # 11-13: Generator-route batch-1,000 runs pause once out of chance
                state["esc"] = state.get("esc", 0) + 1 if rec["net"] > rec["naive"] else 0
                if state["esc"] >= ESCAPE_TESTS:
                    raise StopRun(f"beyond Naive Bayes: {ESCAPE_TESTS} consecutive tests with net > naive")
            if name in STOP_RULE_RUNS:                         # plain runs only; the FR trio runs its full budget
                state["hits"] = state["hits"] + 1 if rec["net"] >= ZFLOW_STOP else 0
                if state["hits"] >= 3:
                    raise StopRun(f"accuracy >= {ZFLOW_STOP} on 3 consecutive tests")
        if time.time() - t0 > budget_s:
            raise StopRun("wall-time budget")
        return {}

    cfg["external_logger"] = logger
    print(f"=== task {a.task}: {name} | seed {seed} | " +
          ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "external_logger"), flush=True)
    try:
        CognitiveGridworld(**cfg)
        print(f"=== {name} finished all episodes in {(time.time() - t0) / 3600:.2f} h", flush=True)
    except StopRun as e:
        env = state["env"]
        env.save()
        print(f"=== {name} stopped at episode {env.e} ({e}) after {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
