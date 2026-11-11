"""Does a trained RCC's Classifier resolve the variable it is NOT rewarded for? (11-10)

The RCC's Generator is fed the Classifier's answers: each variable's answer is sampled from the Classifier's marginal
belief at T (model_forward.default_pobs), the goal's confidence is the reward. The reward covers only the goal variable, and
the Generator's loss is weighted towards rewarded episodes (model_backward.SSL_loss: >= 97.8% of its weight), so the
contextual variable's value reaches the Generator only through the Classifier's answer. This script measures, for saved
RCCs, how good that answer is, against Joint and Naive Bayes on the same episodes (their beliefs sampled the same way).

Per run and environment set (novel = both variables held out; training), T = 30, over all episodes and by the GOAL's cell
type (low / high FR = env_generators.naive_consistent_cells), for RCC / joint / naive:
    argmax_goal, argmax_ctx      accuracy of the belief's argmax for the goal / the contextual variable
    sampled_goal, sampled_ctx    P(a sampled answer is right) = mean belief at the true value (exact expectation)
    ctx_given_rewarded           P(contextual answer right | goal answer right) = E[b_g b_c] / E[b_g]
    both_right                   E[b_g b_c]
    conf_right, conf_wrong       mean confidence (belief in the sampled answer) of right / wrong contextual answers in
                                 rewarded episodes
Episodes are drawn as in training/testing (random variable pairs), seed 0 per run.

    python rcc_answer_accuracy.py --out <json> [--batches 2] [--threads 8] <path>:<|S|>:<zflow 0/1>[:<d_E>] ...
<path> is relative to main/DATA, without "_net.pth". zflow = 1 for runs with the updated Generator (gen_zflow or oracle mode).
    local (11-10, saved RCCs of the paper, original Generator; ~1 min each on 8 threads):
        CUDA_VISIBLE_DEVICES= python rcc_answer_accuracy.py --out saved RL_2:500:0 RL_state_num_reps/400_2:400:0 \
            RL_state_num_reps/800_1:800:0 RL_state_num_reps/800_2:800:0
    cluster (runs of train_cluster_runs.py): see ~/cgw/rcc_answers.sbatch
Results: main/DATA/experimental/rcc_answer_accuracy/<out>.json; `--show <out> ...` prints tables from stored files.
"""
import os, sys, json, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "main", "DATA", "experimental", "rcc_answer_accuracy")
B, R, C = 5000, 10, 2
STATS = ("argmax_goal", "argmax_ctx", "sampled_goal", "sampled_ctx", "ctx_given_rewarded", "both_right", "conf_right", "conf_wrong")


def measure(path, S, zflow, d_E, batches):
    import torch
    from main.CognitiveGridworld import CognitiveGridworld
    np.random.seed(0); torch.manual_seed(0)
    kw = dict(KQ_dim=d_E) if d_E else {}
    env = CognitiveGridworld(mode="SANITY", cuda=0, episodes=1, realization_num=R, hid_dim=1000, obs_num=5, show_plots=False,
                             batch_num=B, step_num=30, state_num=S, learn_embeddings=True, ctx_num=C, load_env="/" + path,
                             gen_zflow=bool(zflow), training=False, checkpoint_every=1, gpu_inference=True, **kw)
    m, dev = env.model, env.device
    BR = torch.arange(B, device=dev)
    res = {}
    for env_set, test_set in (("novel", True), ("train", False)):
        acc = {}
        for _ in range(batches):
            env.test_set = test_set
            env.run_generators()
            with torch.no_grad():
                jb = env.gpu_belief(env.joint_likelihood)[:, -1].float()
                nb = env.gpu_belief(env.naive_likelihood, naive=True)[:, -1].float()
            env.joint_goal_belief = jb[BR, torch.as_tensor(env.goal_ind, device=dev)].cpu().numpy()[:, None]
            env.prep_model()
            with torch.no_grad():
                m.forward_compute()
                cb = m.classifier_belief_flat[:, -1].float()                    # (B, C, R) marginal beliefs at T
            ok, _ = env.naive_consistent_cells()
            tv = torch.as_tensor(env.ctx_vals, device=dev).long()
            cell = torch.as_tensor(np.ravel_multi_index(tuple(env.ctx_vals.T), (R,) * C), device=dev)
            low = ok[BR, cell]
            g = torch.as_tensor(env.goal_ind, device=dev).long()
            c = 1 - g
            for name, bel in (("RCC", cb), ("joint", jb), ("naive", nb)):
                pg, pc = bel[BR, g, tv[BR, g]], bel[BR, c, tv[BR, c]]           # P(sampled answer right): goal, contextual
                ag = (bel[BR, g].argmax(-1) == tv[BR, g]).float()
                ac = (bel[BR, c].argmax(-1) == tv[BR, c]).float()
                cr, cw = pc * pc, (bel[BR, c] ** 2).sum(-1) - pc * pc           # confidence mass on right / wrong answers
                for key, sel in (("all", torch.ones_like(low)), ("low", low), ("high", ~low)):
                    a = acc.setdefault((name, key), np.zeros(9))
                    a += np.array([sel.sum().item()] + [v[sel].sum().item() for v in
                                                        (pg, pc, pg * pc, ag, ac, pg * cr, pg * cw, pg * (1 - pc))])
        res[env_set] = {}
        for (name, key), (n, pg, pc, pgc, ag, ac, gcr, gcw, gw) in acc.items():
            res[env_set].setdefault(name, {})[key] = dict(
                n=int(n), argmax_goal=ag / n, argmax_ctx=ac / n, sampled_goal=pg / n, sampled_ctx=pc / n,
                ctx_given_rewarded=pgc / pg, both_right=pgc / n, conf_right=gcr / pgc, conf_wrong=gcw / gw)
    return res


def show(tag):
    data = json.load(open(os.path.join(OUT, tag + ".json")))
    for run, res in data.items():
        for env_set, by in res.items():
            n = by["RCC"]["all"]["n"]
            print(f"\n{run}, {env_set} environments, {n} episodes; all / low-FR / high-FR cells of the goal")
            print(f"{'':6s}" + "".join(f"{s:>20s}" for s in STATS[:5]) + f"{'both':>7s}{'conf right|wrong':>18s}")
            for name in ("RCC", "joint", "naive"):
                f = lambda s: "/".join(f"{by[name][k][s]:.3f}"[1:] for k in ("all", "low", "high"))
                a = by[name]["all"]
                print(f"{name:6s}" + "".join(f"{f(s):>20s}" for s in STATS[:5]) +
                      f"{a['both_right']:7.3f}{a['conf_right']:11.2f}|{a['conf_wrong']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--show", nargs="+")
    a = ap.parse_args()
    if a.show:
        for tag in a.show:
            show(tag)
        return
    import torch
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, a.out + ".json")
    data = json.load(open(out)) if os.path.exists(out) else {}
    for spec in a.runs:
        path, S, zflow, *d_E = spec.split(":")
        data[path] = measure(path, int(S), int(zflow), int(d_E[0]) if d_E else None, a.batches)
        json.dump(data, open(out, "w"), indent=1)
        print(f"=== {path} done", flush=True)
    show(a.out)


if __name__ == "__main__":
    main()
