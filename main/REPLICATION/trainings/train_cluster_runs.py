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
STOP_RULE_RUNS = {f"{a}_s{k}" for a in ("zflow", "default") for k in range(3)}     # seeds 3-5 run all 30k episodes


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
    post = torch.softmax(ll, -1).reshape(B, R, R)
    g = torch.as_tensor(env.goal_ind, device=dev).long()
    goal_post = torch.where(g[:, None] == 0, post.sum(2), post.sum(1))
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
        cfg.update(batch_num=256, episodes=6, checkpoint_every=3, state_num=min(cfg["state_num"], 200))
    os.makedirs(os.path.join(ROOT, "runs", sub), exist_ok=True)
    log_path = os.path.join(ROOT, "runs", sub, name + ".jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    cfg.update(save_env=f"{sub}/{name}", cuda=0)
    src, offset = RESUME.get(name), 0
    if src:                                   # continue the source run: same environments, its saved net, its log first
        old = [l for l in open(os.path.join(ROOT, "runs", sub, src + ".jsonl")) if l.strip()]
        offset = json.loads(old[-1])["ep"] + 1                       # the source's last test is at its episodes - 1
        cfg.update(load_env=f"/{sub}/{src}", episodes=cfg["episodes"] - (0 if a.smoke else offset))
        with open(log_path, "w") as fh:
            fh.writelines(l if l.endswith("\n") else l + "\n" for l in old)

    import torch
    np.random.seed(seed); torch.manual_seed(seed)
    from main.CognitiveGridworld import CognitiveGridworld

    t0, state = time.time(), dict(env=None, hits=0, train={})
    budget_s = a.budget_h * 3600
    rcc = name.startswith(("rcc", "gen30k", "clf30k", "gen60k", "clf60k"))   # tests also read out the learned likelihood

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
        if rcc and not env.test_set and env.e % env.checkpoint_every == env.checkpoint_every - 1:
            hit, kl = plugin(env)                    # the last training batch before the next test, after its update
            state["train"] = dict(train_plugin=float(hit.mean()), train_kl_env=float(kl.mean()))
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
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
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
