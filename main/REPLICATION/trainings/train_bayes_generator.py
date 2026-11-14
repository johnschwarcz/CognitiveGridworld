"""Generator-only training on a Bayesian observer's answers (John 11-09): does the quality of the answers the Generator
learns from -- Naive's or Joint's (Bayes-optimal) -- set how well it and the embeddings learn the environments, and does
the difference follow FR and |S|?

In the RCC only the Generator's loss trains the embeddings (embedding_grad 'generator'). Under the original feedback the
Generator is fed the Classifier's final answers: each variable's value sampled from the Classifier's marginal belief,
the goal's confidence slot = whether its sampled answer was right, the contextual variable's slot = the belief in its
sample; its loss (SSL_loss) is weighted towards the trials the Classifier got right. Here the Classifier is replaced by
an observer that knows the true likelihood, and no LSTM runs:
  joint   the Bayes posterior's marginals at T
  naive   the factorized (per channel) observer's marginals at T
  truth   the true realizations with confidence 1 (every answer right: oracle mode's Generator input), the ceiling
  <label>1  (John 11-09 21:40) the same, but the Generator is given only ONE variable's realization, the goal's: the other
          variable's sample embedding is zeroed and its confidence slot set to 0, so the Generator can learn only the
          goal's marginal likelihood P(o | r_g) (the factorized model). Its tests use that marginal: kl_env against the
          true marginal (Naive's likelihood) over the goal's 10 values, plugin = Naive-style Bayes with the learned marginal
Controls for where the answers' errors sit (11-10; Naive's sampled answers are right .40 of the time, Joint's .69, so the
Naive / Joint contrast alone confounds how often with where and how the answers are wrong):
  jointU  Joint's answers with each variable's answer replaced, with probability p, by a uniform draw; p is set per batch
          so that the goal answer is right as often as Naive's. Same error rate as Naive, errors unrelated to FR. The
          contextual slot keeps the confidence of the answer it replaced (an unflagged error, like Naive's).
  jn      the goal's answer (and so the reward) from Joint, the contextual variable's answer and confidence from Naive
  nj      the goal's answer from Naive, the contextual variable's from Joint
How the Generator can start while the Classifier is at chance (11-10, John: "does it come from the rewarded actions?"):
  chance   both answers drawn uniformly (a Classifier at chance): the goal slot = whether its answer was right (the
           reward), the contextual slot's confidence 1/R (a uniform belief). Only the rewarded episodes (1 in R) carry
           information: the reward marks the lucky guesses, which are the goal's true value.
  chanceS  the same with the rewards permuted across the batch's episodes: the same reward rate and loss weights, but no
           information about which guesses were right
  These runs also log the goal's learned marginal in the format they train on (goal slot confidence 1, the contextual
  slot a uniform guess with confidence 1/R, averaged over the guess): mkl_env / mkl_own against the true marginal and
  mplugin = Naive-style Bayes with it (as truth1's plugin).
Everything else is the zflow Generator of train_cluster_runs.py's oracle runs: |S| x d_E, batch 10,000, lr 1e-3,
loss SSL_loss + reg_loss, Adam. Labels are drawn with torch's generator, the episodes with numpy's, so at one seed every
label condition sees the same environments, episodes and initial weights.

Each test (at TEST_AT: dense early, where learning is fast, then every TEST_EVERY = 500) runs one batch of novel environments (both variables held out) and one of training
environments, and logs per environment set, overall and on low- / high-FR cells of the episode (naive_consistent_cells):
  kl_env    E_r[KL(P(o|r) || P^(o|r))] over the 100 joint realizations, summed over channels, confidence 1
  kl_own    the same at the episode's own realization
  plugin    accuracy at T of the Bayes posterior computed with the LEARNED likelihood (P^ at all 100 cells, conf 1)
  joint, naive   the observers' accuracy at T with the true likelihood
and, for the training batch before it, how often the label source's goal answer was right on low- / high-FR cells.
-> runs/bayes_gen/<name>.jsonl, main/DATA/bayes_gen/<name>_net.pth at the end.
--lr other than 1e-3 appends _lr<k> to the name (1e-4: _lr4, the RCC's lr). --resume-to N (11-10, John: "train more
episodes, things haven't converged") continues a finished run from its saved net to N episodes, as <name>_e<N/1000>k (the
log starts with the original's records; the original log and net are kept). The environments and the held-out variables
are rebuilt from the same seed and the saved embeddings; Adam's moments restart (they are not saved) and the episode
stream is reseeded, so the resumed part sees new episodes.

    python train_bayes_generator.py --task 3 [--smoke]           # TASKS index
    python train_bayes_generator.py --label naive --S 500 --d 30 --seed 0 [--episodes 6000]
    python train_bayes_generator.py --task 3 --resume-to 20000
    python train_bayes_generator.py --label chance --S 500 --lr 1e-4 --episodes 15000 --test-every 250
    python train_bayes_generator.py --label joint --S 125 --seed 1 --arch old --episodes 8000
    python train_bayes_generator.py --task 123                   # Var(z) decoupling (11-14): tasks 123-252 (ZTASKS)
    python train_bayes_generator.py --label joint --S 100 --d 10 --zscale env     # -> joint_S100_d10_s0_zenv
--zscale (11-14): z x sqrt(d_E / 30) so Var(z) = 1/30 at every d_E (env = the environment only, both = also the Generator's
z_hat, none = unscaled), named in the run; every test also logs <set>_zblind, Bayes with the z-averaged likelihood (the
floor of the normalized transfer score; claude_plots/zscale_table.py reads these runs).
--arch (11-10 12:30, John: the Generator now transfers at |S| 100-125, the paper's Fig 12 needed ~500 -- task change or the
Generator fix?): old = the paper's Generator (gen_zflow False: no stream LayerNorms, no skip, default downscale init), new =
the fix, named _old / _new; bayes_gen_arch.sbatch runs joint answers at |S| 100 / 125 / 250 / 500 x seeds 0-2 x both.
"""
import os, sys, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

LABELS = ("naive", "joint", "truth", "naive1", "joint1", "truth1", "jointU", "jn", "nj", "chance", "chanceS")
# 11-09 17:00 (John: early learning is very fast; also small |S|: does Joint's answers' generalization ever break?): the
# Joint and Naive columns re-run with dense early tests at |S| 25-1000. The first runs (12 tasks: |S| 100 / 250 / 500 /
# 1000 x naive / joint / truth, tests every 500) are kept in runs/bayes_gen/every500/.
TASKS = [(lab, S, 30, 0) for S in (25, 50, 75, 100, 250, 500, 1000, 125, 150, 200) for lab in ("joint", "naive")]
# 11-09 21:45 (John: Naive vs Joint answers move only the final accuracy, not the |S| where embeddings generalize; what if the
# Generator gets a single variable's realization?): 20-39 = truth1 (goal only) and truth (both variables, dense tests; the
# first truth runs had tests every 500 at 4 |S|) at every |S|, 8,000 episodes (6,000 was still rising by ~.01-.03 per 2k).
TASKS += [(lab, S, 30, 0) for S in (25, 50, 75, 100, 125, 150, 200, 250, 500, 1000) for lab in ("truth1", "truth")]
# 11-09 22:40 (John: every agent on the |S| x d_E grid, over a larger range): 40-119 = truth1 / truth / joint / naive at
# d_E 10 and 90, first a coarse |S| grid (25, 100, 250, 1000), then the rest (d_E 30 = tasks 0-39).
AGENTS = ("truth1", "truth", "joint", "naive")
TASKS += [(lab, S, d, 0) for S in (25, 100, 250, 1000) for d in (10, 90) for lab in AGENTS]
TASKS += [(lab, S, d, 0) for S in (50, 75, 125, 150, 200, 500) for d in (10, 90) for lab in AGENTS]
# 11-10 03:45: 120-122 = the error controls at |S| 500, d_E 30 (compare joint / naive / truth there: tasks 10, 11, 37).
TASKS += [(lab, 500, 30, 0) for lab in ("jointU", "jn", "nj")]
# 11-14 (theory review): is the small-d_E failure set by Var(z) = 1/d_E? Joint's answers with z multiplied by sqrt(d_E/30),
# so Var(z) = 1/30 in every row (--zscale): "env" scales the environment's true z only, "both" also the Generator's z_hat,
# "none" reruns the unscaled rows in the same code (d_E 20's original runs are not in this repo; the d_E 10 / 90 seed-0
# reruns should reproduce tasks 40-119). At d_E 30 the factor is 1, so its runs serve every arm. Seed 0 everywhere (the
# same variables and initial weights as the baseline at that seed), seeds 1-2 at the pre-registered transition cells.
# Tasks 123-252, five-tuples (label, |S|, d_E, seed, arm), in priority order:
ZS_S = (50, 75, 100, 125, 250, 500, 1000)
ZS_EXTRA = {10: (50, 75, 100), 20: (50, 75, 100), 30: (75, 100), 90: (125, 250, 500)}
ZS_EXTRA_NONE = {20: (50, 75, 100, 125, 250, 500, 1000), 30: (75, 100), 90: (125, 250, 500)}
ZTASKS = [("joint", S, d, 0, "env") for d in (10, 20, 90) for S in ZS_S]                         # 123-143 the test
ZTASKS += [("joint", S, d, 0, "none") for d in (20, 30, 90, 10) for S in ZS_S]                   # 144-171 same-code baselines
ZTASKS += [("joint", S, d, 0, "both") for d in (10, 20, 90) for S in ZS_S]                       # 172-192 + model z_hat
ZTASKS += [("joint", S, d, s, arm) for arm in ("env", "both") for d in (10, 20, 90) for s in (1, 2)
           for S in ZS_EXTRA[d]]                                                                   # 193-228
ZTASKS += [("joint", S, d, s, "none") for d in (20, 30, 90) for s in (1, 2) for S in ZS_EXTRA_NONE[d]]   # 229-252
# 11-14 23:25 IST (John: "kill runs that have already converged and spend that compute elsewhere"): tasks 253-262 =
# scaled d_E 10 below |S| 50, where the bootstrap rule puts its threshold (|S|* 25-32): |S| 20 / 25 / 30 / 40 x seeds
# 0-1, and seeds 3-4 at unscaled d_E 20 |S| 1000 (seed 0 failed there, against the law)
ZTASKS += [("joint", S, 10, s, "env") for S in (20, 25, 30, 40) for s in (0, 1)]          # 253-260
ZTASKS += [("joint", 1000, 20, s, "none") for s in (3, 4)]                               # 261-262
# 11-14 21:45 UTC (John, theory-review session: "yes, let's run these unseen cells"): cells that test the bootstrap rule
# (theory note of 11-14, claude_plots/zscale_proxy.py: a cell transfers, n >= .2, when the proxy r >= r*, .37 < r* <= .45;
# the level was read off the seed-0 cells of tasks 123-171). Predictions fixed before any of these ran, r in brackets:
#   263-274  A: d_E 30 at |S| 80 / 85 / 90 / 95 (.37 / .40 / .41 / .42), seeds 0-2. 80 sits on the band's lower edge,
#            85-95 inside it (open): the first size that transfers pins r*.
#   275-292  B: a second z0, z0^2 = 1/20 (z x sqrt(d_E / 20)), seeds 0-1. d_E 10: no transfer at |S| 50, 75 (.21, .30),
#            open at 100, 125 (.39, .44), transfer at 250 (.72). d_E 30: no transfer at 125, 250 (.17, .33), transfer at
#            500, 1000 (.59, .81; 1000 shares z0 and |S| with the rule's one miss, unscaled d_E 20 at 1000).
#   253-260  (above) scaled d_E 10 at |S| 20 / 25 / 30 / 40 (.31 / .37 / .43 / .52): no transfer at 20 and 25, open at 30,
#            transfer at 40 (only 2-4 held-out variables: 3-10 novel environments per run).
#   293-306  d_E 60, seed 0 (not approved yet). Scaled: no transfer at |S| 50-125 (.00 / .11 / .23 / .30), transfer from
#            250 (.52 / .73 / .88). Unscaled: no transfer at 50 (.01; 45 partners < d_E) and 75 (.35, just under the
#            band), transfer from 100 (.57 / .66 / .87 / .95 / .98).
# A six-tuple's last entry is 1 / z0^2 (--z-ref-dim; it is part of the run's name: _zenv20).
ZTASKS += [("joint", S, 30, s, "none") for s in (0, 1, 2) for S in (80, 85, 90, 95)]                 # 263-274
ZTASKS += [("joint", S, 10, s, "env", 20.) for s in (0, 1) for S in (50, 75, 100, 125, 250)]         # 275-284
ZTASKS += [("joint", S, 30, s, "env", 20.) for s in (0, 1) for S in (125, 250, 500, 1000)]           # 285-292
ZTASKS += [("joint", S, 60, 0, arm) for arm in ("env", "none") for S in ZS_S]                         # 293-306
TASKS += ZTASKS
EPISODES_FROM_20 = 8000
EPISODES, TEST_EVERY, BATCH, LR = 6000, 500, 10000, 1e-3
TEST_AT = (0, 25, 50, 100, 150, 250, 350, 500, 750, 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--label", choices=LABELS)
    ap.add_argument("--S", type=int, default=500)
    ap.add_argument("--d", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--test-every", type=int, default=TEST_EVERY)
    ap.add_argument("--smoke", action="store_true", help="batch 256, 6 episodes, test every 3, output under smoke/")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--resume-to", type=int, default=None, help="continue the finished run to this many episodes")
    ap.add_argument("--arch", choices=("new", "old"), default=None,
                    help="Generator architecture, named in the run: new = the Z-flow fix (the default), old = the paper's")
    ap.add_argument("--zscale", choices=("none", "env", "both"), default=None,
                    help="z x sqrt(d_E / z-ref-dim), named in the run (_z<arm>): env = the environment's true z, both = also "
                         "the Generator's z_hat, none = unscaled (the same-code baseline)")
    ap.add_argument("--z-ref-dim", type=float, default=30., help="Var(z) = 1 / z-ref-dim under --zscale env / both")
    a = ap.parse_args()
    # the array index picks a TASKS row only when the run is not named explicitly (11-10 16:05: the arch test's array passed
    # --label/--S/--seed and the index overrode them, so its tasks 0-1 ran TASKS[0-1] = joint/naive |S| 25 instead)
    if a.task is None and a.label is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        a.task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if a.task is not None:
        a.label, a.S, a.d, a.seed, *arm = TASKS[a.task]
        if arm:
            a.zscale = arm[0]
        if len(arm) > 1:
            a.z_ref_dim = arm[1]
        if a.task >= 20:
            a.episodes = EPISODES_FROM_20
    B, episodes, test_every = (256, 6, 3) if a.smoke else (a.batch, a.episodes, a.test_every)
    # the scaling arm is part of the name, so these runs never overwrite (line below: os.remove) a baseline log
    zname = "" if a.zscale is None else f"_z{a.zscale}" + ("" if a.z_ref_dim == 30 else f"{a.z_ref_dim:g}")
    name = (f"{a.label}_S{a.S}_d{a.d}_s{a.seed}" + ("" if a.lr == LR else f"_lr{-int(round(np.log10(a.lr)))}") +
            (f"_{a.arch}" if a.arch else "") + zname)
    if a.arch == "old":
        assert not a.label.endswith("1"), "predict() hard-codes the new Generator; single-variable labels need it"
    sub = "smoke" if a.smoke else "bayes_gen"
    os.makedirs(os.path.join(ROOT, "runs", sub), exist_ok=True)
    start, old = 0, []
    if a.resume_to:
        old = [json.loads(l) for l in open(os.path.join(ROOT, "runs", sub, name + ".jsonl")) if l.strip()]
        start, src, episodes = old[-1]["ep"], name, a.resume_to
        name = f"{name}_e{a.resume_to // 1000}k"
    log_path = os.path.join(ROOT, "runs", sub, name + ".jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    if old:
        with open(log_path, "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in old)

    import torch
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    from main.CognitiveGridworld import CognitiveGridworld
    env = CognitiveGridworld(mode="RL", ctx_num=2, obs_num=5, realization_num=10, state_num=a.S, KQ_dim=a.d, hid_dim=1000,
                             batch_num=B, step_num=30, learn_embeddings=True, gen_zflow=a.arch != "old", generator_LR=a.lr,
                             classifier_LR=a.lr, episodes=1, training=False, checkpoint_every=1, gpu_inference=True,
                             show_plots=False, cuda=0, load_env=f"/{sub}/{src}" if start else None,
                             scale_z_env=a.zscale in ("env", "both"), scale_z_model=a.zscale == "both", z_ref_dim=a.z_ref_dim)
    m, dev, R, C = env.model, env.device, env.realization_num, env.ctx_num
    if a.zscale is not None:
        print(f"zscale {a.zscale}: environment z x {env.env_z_scale():.4f}, Generator z_hat x {m.z_scale:.4f}", flush=True)
    if start:                                            # same environments and held-out variables, new episodes
        np.random.seed(a.seed + 1000); torch.manual_seed(a.seed + 1000)
        # load() replaces all_K / all_Q with new Parameters after the optimizer took the old ones (which load_state_dict
        # already filled with the same saved values): give the model back the optimizer's, or the embeddings never train
        held = [p for g in m.generator_optim.param_groups for p in g["params"]]
        for k in ("all_K", "all_Q"):
            same = [p for p in held if p.shape == getattr(m, k).shape and torch.equal(p.data, getattr(m, k).data)]
            assert len(same) == 1, k
            setattr(m, k, same[0])
    labels_rng = torch.Generator(device=dev).manual_seed(a.seed + 7 + (1000 if start else 0))
    BR = torch.arange(B, device=dev)
    amp = dict(device_type="cuda", enabled=env.device_type != "cpu")
    print(f"=== {name}: device {dev}, threads {torch.get_num_threads()}, batch {B}, {episodes} episodes", flush=True)

    def episode(test_set):
        env.test_set = test_set
        env.run_generators()
        with torch.no_grad():
            jb = env.gpu_belief(env.joint_likelihood)[:, -1].float()                     # (B, C, R) marginals at T
            nb = env.gpu_belief(env.naive_likelihood, naive=True)[:, -1].float()
        env.joint_goal_belief = jb[BR, torch.as_tensor(env.goal_ind, device=dev)].cpu().numpy()[:, None]
        env.prep_model()
        ok, _ = env.naive_consistent_cells()
        cell = torch.as_tensor(np.ravel_multi_index(tuple(env.ctx_vals.T), (R,) * C), device=dev)
        return jb, nb, ok[BR, cell]

    single = a.label.endswith("1")
    base = a.label.rstrip("1")

    def predict(sample, conf, keep):
        """model.get_prediction with the realization embeddings of the slots where keep = 0 zeroed"""
        emb = m.sample_to_emb(sample) * keep[..., None]
        s = m.sample_to_hid(emb.reshape(B, -1)).unsqueeze(1)
        c = m.conf_to_hid(conf).unsqueeze(1)
        z = m.Z_to_pobs(m.generator_Z())
        s, c, z = m.get_layer_norm(m.gen_stream_norm, (s, c, z))
        pre = s + c + z
        x = m.gen_hid2hid(torch.relu(pre))
        m.pred_pobs = torch.sigmoid(m.hid_to_pobs(torch.relu(pre + x))).squeeze()

    def goal_only(sample, conf):
        """the single-variable input: only the goal's realization and confidence"""
        keep = torch.nn.functional.one_hot(m.goal_ind.long(), C).float()
        return sample * keep.long(), conf * keep, keep

    def answers(jb, nb):
        """(sample, conf, acc): the label source's answers in the RCC's feedback format (model.default_pobs)"""
        if base == "truth":
            sample = m.ctx_vals.long()
            return sample, torch.ones(B, C, device=dev), torch.ones(B, device=dev)
        g = m.goal_ind.long()
        if base in ("chance", "chanceS"):                    # a Classifier at chance; the reward flags its lucky guesses
            sample = torch.randint(R, (B, C), generator=labels_rng, device=dev)
            conf = torch.full((B, C), 1 / R, device=dev)
            acc = (sample[BR, g] == m.goal_value).float()
            if base == "chanceS":                            # same rate, unrelated to which guesses were right
                acc = acc[torch.randperm(B, generator=labels_rng, device=dev)]
            conf[BR, g] = acc
            return sample, conf, acc
        if base in ("jn", "nj"):                             # goal slot from one observer, contextual slot from the other
            gb, cb = (jb, nb) if base == "jn" else (nb, jb)
            belief = torch.where(torch.nn.functional.one_hot(g, C).bool()[..., None], gb, cb)
        else:
            belief = jb if base.startswith("joint") else nb
        sample = torch.multinomial(belief.reshape(B * C, R), 1, generator=labels_rng).reshape(B, C)
        conf = belief.gather(2, sample[..., None])[..., 0]
        if base == "jointU":                                 # Naive's error rate, placed at random
            tv = m.goal_value.long()
            aj, an = jb[BR, g, tv].mean(), nb[BR, g, tv].mean()            # how often a sampled goal answer is right
            swap = torch.rand(B, C, generator=labels_rng, device=dev) < ((aj - an) / (aj - 1 / R)).clamp(0, 1)
            sample = torch.where(swap, torch.randint(R, (B, C), generator=labels_rng, device=dev), sample)
        acc = (sample[BR, g] == m.goal_value).float()
        conf[BR, g] = acc
        return sample, conf, acc

    def learned_marginal():
        """(B, O, R) single-variable Generator's predictions at every goal value (confidence 1, the other slot empty)"""
        g, out = m.goal_ind.long(), []
        keep = torch.nn.functional.one_hot(g, C).float()
        for r in range(R):
            sample = torch.zeros(B, C, dtype=torch.long, device=dev)
            sample[BR, g] = r
            predict(sample, keep, keep)
            out.append(m.pred_pobs.float())
        return torch.stack(out, -1)

    def learned_marginal_guess():
        """(B, O, R) the Generator's prediction at every goal value in the chance runs' training format (goal slot
        confidence 1, the contextual slot a uniform guess with confidence 1/R), averaged over the guess"""
        g, out = m.goal_ind.long(), []
        conf = torch.full((B, C), 1 / R, device=dev)
        conf[BR, g] = 1
        for r in range(R):
            p = 0
            for rc in range(R):
                sample = torch.full((B, C), rc, dtype=torch.long, device=dev)
                sample[BR, g] = r
                m.get_prediction(sample, conf)
                p = p + m.pred_pobs.float() / R
            out.append(p)
        return torch.stack(out, -1)

    def learned_table():
        """(B, O, R, R) Generator predictions at every joint realization, confidence 1"""
        ones, out = torch.ones(B, C, device=dev), []
        for r1 in range(R):
            for r2 in range(R):
                m.get_prediction(torch.tensor([r1, r2], device=dev).expand(B, 2).contiguous(), ones)
                out.append(m.pred_pobs.float())
        return torch.stack(out, -1).reshape(B, env.obs_num, R, R)

    def zblind_table(n_pairs=20000, chunk=2000):
        """(O, R*R) the true likelihood averaged over random pairs of training variables: what a Generator that ignores z_hat
        converges to. Its plug-in accuracy (logged as <set>_zblind) is the floor of the normalized transfer score
        n = (plugin - zblind) / (joint - zblind) (11-14). A private RNG leaves the episode stream untouched."""
        rng = np.random.default_rng(10_000 + a.seed)
        train, s, tot = np.arange(env.test_states, env.state_num), env.env_z_scale(), 0.
        for _ in range(n_pairs // chunk):
            c = rng.choice(train, size=(chunk, 2))
            z01 = np.einsum('nok,nok->no', env.all_K[c[:, 0]], env.all_Q[c[:, 1]]) * s
            z10 = np.einsum('nok,nok->no', env.all_K[c[:, 1]], env.all_Q[c[:, 0]]) * s
            tot = tot + env.sigmoid(env.soft_roll(z01)[..., :, None] * env.soft_roll(z10)[..., None, :]).sum(0)
        L = tot / (n_pairs // chunk * chunk)
        return torch.as_tensor(L, device=dev, dtype=torch.float32).reshape(env.obs_num, -1).clamp(1e-6, 1 - 1e-6)

    Lbar = None if single else zblind_table()

    def test(test_set, rec, key):
        jb, nb, low = episode(test_set)
        g = m.goal_ind.long()
        o = torch.as_tensor(env.obs_flat, device=dev, dtype=torch.float32)                       # (B, T, O)
        if single:                                       # the learned marginal of the goal against Naive's (true) one
            with torch.no_grad(), torch.autocast(**amp):
                m.MC_to_interactions()
                P = learned_marginal().clamp(1e-6, 1 - 1e-6)                                     # (B, O, R)
            L = torch.as_tensor(env.naive_likelihood, device=dev, dtype=torch.float32)[BR, :, g].clamp(1e-6, 1 - 1e-6)
            kl = (L * (L / P).log() + (1 - L) * ((1 - L) / (1 - P)).log()).sum(1)                 # (B, R)
            cell = m.goal_value.long()
            Pf = P
        else:
            with torch.no_grad(), torch.autocast(**amp):
                m.MC_to_interactions()
                P = learned_table().clamp(1e-6, 1 - 1e-6)
            L = torch.as_tensor(env.joint_likelihood, device=dev, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
            kl = (L * (L / P).log() + (1 - L) * ((1 - L) / (1 - P)).log()).sum(1).reshape(B, -1)   # (B, R*R)
            cell = torch.as_tensor(np.ravel_multi_index(tuple(env.ctx_vals.T), (R,) * C), device=dev)
            Pf = P.reshape(B, env.obs_num, -1)
        ll = torch.einsum('bto,bok->bk', o, Pf.log()) + torch.einsum('bto,bok->bk', 1 - o, (1 - Pf).log())
        if single:
            goal_post = ll
        else:
            post = torch.softmax(ll, -1).reshape(B, R, R)
            goal_post = torch.where(g[:, None] == 0, post.sum(2), post.sum(1))
        res = dict(kl_env=kl.mean(1), kl_own=kl[BR, cell], plugin=(goal_post.argmax(-1) == m.goal_value).float(),
                   joint=(jb[BR, g].argmax(-1) == m.goal_value).float(), naive=(nb[BR, g].argmax(-1) == m.goal_value).float())
        if Lbar is not None:                             # the z-blind floor: the same observations through the z-averaged table
            llz = torch.einsum('bto,ok->bk', o, Lbar.log()) + torch.einsum('bto,ok->bk', 1 - o, (1 - Lbar).log())
            postz = torch.softmax(llz, -1).reshape(B, R, R)
            res["zblind"] = (torch.where(g[:, None] == 0, postz.sum(2), postz.sum(1)).argmax(-1) == m.goal_value).float()
        if base.startswith("chance"):                    # the goal's marginal as the chance runs see it
            with torch.no_grad(), torch.autocast(**amp):
                Pm = learned_marginal_guess().clamp(1e-6, 1 - 1e-6)                             # (B, O, R)
            Lm = torch.as_tensor(env.naive_likelihood, device=dev, dtype=torch.float32)[BR, :, g].clamp(1e-6, 1 - 1e-6)
            mkl = (Lm * (Lm / Pm).log() + (1 - Lm) * ((1 - Lm) / (1 - Pm)).log()).sum(1)         # (B, R)
            mll = torch.einsum('bto,bok->bk', o, Pm.log()) + torch.einsum('bto,bok->bk', 1 - o, (1 - Pm).log())
            res.update(mkl_env=mkl.mean(1), mkl_own=mkl[BR, m.goal_value.long()],
                       mplugin=(mll.argmax(-1) == m.goal_value).float())
        rec[key + "_frac_low"] = float(low.float().mean())
        for k, v in res.items():
            rec[f"{key}_{k}"] = float(v.mean())
            rec[f"{key}_{k}_low"], rec[f"{key}_{k}_high"] = float(v[low].mean()), float(v[~low].mean())

    t0, lab_ok, ctx_ok, losses = time.time(), [], [], []
    test_at = set(range(0, episodes + 1, test_every)) | (set() if a.smoke else {t for t in TEST_AT if t <= episodes})
    for e in range(start, episodes + 1):
        if (e in test_at and not (start and e == start)) or e == episodes:      # a resumed run's first test is logged
            rec = dict(ep=e, t=round(time.time() - t0, 1))
            if lab_ok:
                acc_, low_ = (torch.cat(x) for x in zip(*lab_ok))
                rec.update(label_acc=float(acc_.mean()), label_acc_low=float(acc_[low_].mean()),
                           label_acc_high=float(acc_[~low_].mean()), gen_loss=float(np.mean(losses)),
                           label_ctx_acc=float(np.mean(ctx_ok)))
            lab_ok, losses, ctx_ok = [], [], []
            test(True, rec, "novel")
            test(False, rec, "train")
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"ep {e:5d} {rec['t']:7.0f}s | novel plugin {rec['novel_plugin']:.3f} (low {rec['novel_plugin_low']:.3f} "
                  f"high {rec['novel_plugin_high']:.3f}) kl_env {rec['novel_kl_env']:.4f} | train plugin "
                  f"{rec['train_plugin']:.3f} | joint {rec['novel_joint']:.3f} naive {rec['novel_naive']:.3f}", flush=True)
            if e == episodes:
                break
        jb, nb, low = episode(False)
        sample, conf, acc = answers(jb, nb)
        with torch.autocast(**amp):
            m.MC_to_interactions()
            m.ACC = acc[:, None]
            if single:
                predict(*goal_only(sample, conf))
            else:
                m.get_prediction(sample, conf)
            m.SSL_loss()
            m.generator_loss = m.generator_loss + m.reg_loss()
        m.update(m.generator_loss, m.generator_optim)
        lab_ok.append((acc.detach(), low))
        losses.append(float(m.generator_loss))
        ctx_ok.append(float((sample == m.ctx_vals.long()).float().sum(1).sub(acc).mean()) / (C - 1))
    env.save_env = f"{sub}/{name}"
    env.save()
    print(f"=== {name} done in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
