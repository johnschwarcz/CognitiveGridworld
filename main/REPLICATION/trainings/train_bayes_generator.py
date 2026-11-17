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
    python train_bayes_generator.py --task 3 --resume-to 30000 --resume-from _e20k     # 11-15: continue the 20k run
    python train_bayes_generator.py --label chance --S 500 --lr 1e-4 --episodes 15000 --test-every 250
    python train_bayes_generator.py --label joint --S 125 --seed 1 --arch old --episodes 8000
--arch (11-10 12:30, John: the Generator now transfers at |S| 100-125, the paper's Fig 12 needed ~500 -- task change or the
Generator fix?): old = the paper's Generator (gen_zflow False: no stream LayerNorms, no skip, default downscale init), new =
the fix, named _old / _new; bayes_gen_arch.sbatch runs joint answers at |S| 100 / 125 / 250 / 500 x seeds 0-2 x both.
--zin (11-11 21:50, John: a real difference "should not depend on the specific architecture"): the Generator's Z input
layer as built (shared: one Linear for every channel) or one Linear per channel (fresh_generator's per_channel_linear;
the rest of the Generator stays shared): per_channel = every channel's layer starts as a copy of the shared layer's Z-flow
start (oracle_readin aligns it with the read-out path), per_channel_random = independent default starts (no alignment),
shared_random = the shared layer with a default start (no alignment; 11-11 22:35: separates a common start from the
alignment); named _shared / _per_channel / _per_channel_random / _shared_random. These runs also save each novel test batch's learned and true codes
(<name>_codes.npz, overwritten per test) for claude_plots/fig13_controls.py.
--learner (11-11 23:00, John: "run the classifier with the shared channel thing"): the same embeddings, start, batch, lr and
reg_loss, but learned by CLASSIFICATION instead of generation: a reader from trainings/transfer_reader.py (reader_shared =
one evidence function applied to every channel and summed, the form of exact Bayes; reader_unshared = one per channel;
reader_lstm = an LSTM over the 30 steps reading every channel's code and bits, like the RCC's Classifier) is trained with
the embeddings on the true joint value (cross-entropy + reg_loss; label truth only). Tests log the reader's goal accuracy
(<set>_reader) instead of the plug-in; the codes are saved as for --zin. Named _<learner>.
--reuse k (11-11 23:00, John: "could the generator learn from the same batch of data multiple times while the classifier
couldn't?"): k optimizer steps on every fresh batch (episodes = fresh batches); named _reuse<k>.
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
# 11-12 12:25 (John: "add an embedding dim = 20" to the generator-on-answers grid): 123-146 = d_E 20 at every |S| of the
# grid for the three drawn rows (8,000 episodes, as tasks from 20 on).
TASKS += [(lab, S, 20, 0) for S in (25, 50, 75, 100, 125, 250, 500, 1000) for lab in ("truth1", "joint", "naive")]
# 11-13 02:50 (John: "can you explain how the joint bayes did worse than naive in a slot?"): at d_E 20 single runs fail to
# escape their start within 8,000 episodes (joint |S| 1000, naive |S| 250: training accuracy creeps .10 -> .2), so the
# grid's d_E 20 row gets a second seed where it transfers or should: 147-154 = (joint, naive) x |S| 125 / 250 / 500 / 1000,
# seed 1 (the two stuck seed-0 runs also continue to 20,000 episodes: --task 145 / 140 --resume-to 20000).
TASKS += [(lab, S, 20, 1) for S in (125, 250, 500, 1000) for lab in ("joint", "naive")]
assert len(TASKS) == 155 and TASKS[145] == ("joint", 1000, 20, 0) and TASKS[140] == ("naive", 250, 20, 0)
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
    ap.add_argument("--resume-from", default="", help="with --resume-to: continue the run saved as <name><tag> (e.g. _e20k)")
    ap.add_argument("--learner", choices=("generator", "reader_shared", "reader_unshared", "reader_lstm"),
                    default="generator", help="learn the embeddings by generation (the RCC's Generator) or classification")
    ap.add_argument("--reuse", type=int, default=1, help="optimizer steps per fresh batch")
    ap.add_argument("--zin", choices=("shared", "per_channel", "per_channel_random", "shared_random"), default=None,
                    help="the Generator's Z input layer: one for every channel (as built) or one per channel")
    ap.add_argument("--arch", choices=("new", "old"), default=None,
                    help="Generator architecture, named in the run: new = the Z-flow fix (the default), old = the paper's")
    a = ap.parse_args()
    # the array index picks a TASKS row only when the run is not named explicitly (11-10 16:05: the arch test's array passed
    # --label/--S/--seed and the index overrode them, so its tasks 0-1 ran TASKS[0-1] = joint/naive |S| 25 instead)
    if a.task is None and a.label is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        a.task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if a.task is not None:
        a.label, a.S, a.d, a.seed = TASKS[a.task]
        if a.task >= 20:
            a.episodes = EPISODES_FROM_20
    B, episodes, test_every = (256, 6, 3) if a.smoke else (a.batch, a.episodes, a.test_every)
    name = (f"{a.label}_S{a.S}_d{a.d}_s{a.seed}" + ("" if a.lr == LR else f"_lr{-int(round(np.log10(a.lr)))}") +
            (f"_{a.arch}" if a.arch else "") + (f"_{a.zin}" if a.zin else "") +
            (f"_{a.learner}" if a.learner != "generator" else "") + (f"_reuse{a.reuse}" if a.reuse > 1 else ""))
    if a.arch == "old":
        assert not a.label.endswith("1"), "predict() hard-codes the new Generator; single-variable labels need it"
    sub = "smoke" if a.smoke else "bayes_gen"
    os.makedirs(os.path.join(ROOT, "runs", sub), exist_ok=True)
    start, old = 0, []
    if a.resume_to:
        # 11-15 (John: the band figure's Generators to 30,000 episodes, the RCC's length): --resume-from _e20k continues the
        # 20k run (its log and net) instead of the original
        src = name + a.resume_from
        old = [json.loads(l) for l in open(os.path.join(ROOT, "runs", sub, src + ".jsonl")) if l.strip()]
        start, episodes = old[-1]["ep"], a.resume_to
        assert start < episodes, f"{src} is already at {start} episodes"
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
                             show_plots=False, cuda=0, load_env=f"/{sub}/{src}" if start else None)
    m, dev, R, C = env.model, env.device, env.realization_num, env.ctx_num
    if start:                                            # same environments and held-out variables, new episodes
        k = 2000 if a.resume_from else 1000              # a second resume must not replay the first one's episode stream
        np.random.seed(a.seed + k); torch.manual_seed(a.seed + k)
        # load() replaces all_K / all_Q with new Parameters after the optimizer took the old ones (which load_state_dict
        # already filled with the same saved values): give the model back the optimizer's, or the embeddings never train
        held = [p for g in m.generator_optim.param_groups for p in g["params"]]
        for k in ("all_K", "all_Q"):
            same = [p for p in held if p.shape == getattr(m, k).shape and torch.equal(p.data, getattr(m, k).data)]
            assert len(same) == 1, k
            setattr(m, k, same[0])
    if a.zin == "shared_random":                         # the shared layer without the Z-flow alignment (in place, so
        m.Z_to_pobs.reset_parameters()                   # the optimizer keeps its tensors)
    if a.zin in ("per_channel", "per_channel_random"):   # one Z input layer per channel, in the optimizer's place
        assert not start, "a resumed run would load the shared layer"
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from fresh_generator import per_channel_linear
        old_ids = {id(p) for p in m.Z_to_pobs.parameters()}
        pcl = per_channel_linear(env.obs_num, m.Z_to_pobs.in_features, m.Z_to_pobs.out_features).to(dev)
        if a.zin == "per_channel":                       # every channel starts as the shared layer, then they train apart
            with torch.no_grad():
                pcl.W.copy_(m.Z_to_pobs.weight.T[None].expand_as(pcl.W))
                pcl.b.copy_(m.Z_to_pobs.bias[None].expand_as(pcl.b))
        m.Z_to_pobs = pcl
        assert len(m.generator_optim.param_groups) == 1
        group = m.generator_optim.param_groups[0]
        kept = [p for p in group["params"] if id(p) not in old_ids]
        assert len(kept) == len(group["params"]) - 2
        m.generator_optim = torch.optim.Adam([{"params": kept + list(m.Z_to_pobs.parameters()), "lr": group["lr"]}])
    reader = None
    if a.learner != "generator":                         # classification: a reader trained with the embeddings
        assert a.label == "truth" and not start and not a.zin
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import transfer_reader as trd
        rng_state = torch.get_rng_state()
        torch.manual_seed(a.seed + 100)
        reader = trd.readers()[a.learner[len("reader_"):]]().to(dev)
        torch.set_rng_state(rng_state)
        emb = [m.all_K, m.all_Q] + list(m.K_downscale.parameters()) + list(m.Q_downscale.parameters())
        reader_optim = torch.optim.Adam([{"params": emb + list(reader.parameters()), "lr": a.lr}])
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

    def test_reader(test_set, rec, key):
        """the reader's goal accuracy (marginal of its joint posterior), with the observers' on the same episodes"""
        jb, nb, low = episode(test_set)
        g = m.goal_ind.long()
        o = torch.as_tensor(env.obs_flat, device=dev, dtype=torch.float32)
        with torch.no_grad():
            m.MC_to_interactions()
            post = torch.softmax(reader(m.active_Z.float(), o), -1).reshape(B, R, R)
        marg = torch.where(g[:, None] == 0, post.sum(2), post.sum(1))
        res = dict(reader=(marg.argmax(-1) == m.goal_value).float(),
                   joint=(jb[BR, g].argmax(-1) == m.goal_value).float(), naive=(nb[BR, g].argmax(-1) == m.goal_value).float())
        rec[key + "_frac_low"] = float(low.float().mean())
        for k, v in res.items():
            rec[f"{key}_{k}"] = float(v.mean())
            rec[f"{key}_{k}_low"], rec[f"{key}_{k}_high"] = float(v[low].mean()), float(v[~low].mean())

    run_test = test if reader is None else test_reader
    acc_key = "plugin" if reader is None else "reader"
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
            run_test(True, rec, "novel")
            if a.zin or reader is not None:              # this novel batch's codes, for the frame analysis
                np.savez(log_path[:-6] + "_codes.npz", z=m.active_Z.detach().float().cpu().numpy(),
                         ztrue=np.asarray(env.joint_Z), ep=e)
            run_test(False, rec, "train")
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"ep {e:5d} {rec['t']:7.0f}s | novel {acc_key} {rec[f'novel_{acc_key}']:.3f} (low "
                  f"{rec[f'novel_{acc_key}_low']:.3f} high {rec[f'novel_{acc_key}_high']:.3f}) kl_env "
                  f"{rec.get('novel_kl_env', float('nan')):.4f} | train {acc_key} {rec[f'train_{acc_key}']:.3f} | joint "
                  f"{rec['novel_joint']:.3f} naive {rec['novel_naive']:.3f}", flush=True)
            if e == episodes:
                break
        jb, nb, low = episode(False)
        sample, conf, acc = answers(jb, nb)
        for _ in range(a.reuse):                         # the same batch, a.reuse optimizer steps
            if reader is not None:
                o = torch.as_tensor(env.obs_flat, device=dev, dtype=torch.float32)
                m.MC_to_interactions()
                loss = (torch.nn.functional.cross_entropy(reader(m.active_Z.float(), o), sample[:, 0] * R + sample[:, 1]) +
                        m.reg_loss())
                reader_optim.zero_grad()
                loss.backward()
                reader_optim.step()
                m.generator_loss = loss.detach()
                continue
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
