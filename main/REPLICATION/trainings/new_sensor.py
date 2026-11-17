"""A new sensor: does learning its meaning by PREDICTING its observations beat learning it through CLASSIFICATION, with the
channel treatment matched? (John 11-11: "a real meaningful difference ... should not depend on the specific architecture";
"please run the new sensor-experiment")

Start: the trained Generator-route network gen60k_lr4_s0 (Fig 13, oracle feedback, 60k episodes). Its channel 4 becomes a
new sensor: the channel-4 slices of every variable's embeddings (all_K[:, 4], all_Q[:, 4]) are redrawn at random; the
world still emits all 5 channels. ONLY those slices are learned (Adam, lr 1e-4, batch 1,000); everything else is frozen.
Both learners read every channel with one shared function:
    generative       the network's own Generator (frozen): predict channel 4's observations from the values
    discriminative   a channel-wise evidence reader (one function for every channel, summed; trained beforehand on all 5
                     channels of the ORIGINAL embeddings, then frozen): classify the joint value
and get the values either as the TRUE labels or WITHOUT labels: sampled from the Bayes posterior given channels 0-3 only
(what the rest of the network can already infer; generative: a sampled value per episode; discriminative: the posterior as
a soft target).
Tests every --test-every steps (one novel batch of --batch episodes): code R^2 (linear fit of the learned channel-4 code from the true one) and
accuracy with channels 0-3 vs with channel 4 added (generative: Bayes with the Generator's likelihood; discriminative:
the reader). -> runs/new_sensor/<tag>.jsonl

    python new_sensor.py --task k
"""
import os, sys, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import fresh_generator as fg
import transfer_reader as trd

OUT = os.path.join(ROOT, "runs", "new_sensor")
R, C, O, T, NEW = 10, 2, 5, 30, 4
TASKS = [(learner, labels, k) for learner in ("generative", "discriminative") for labels in ("true", "none") for k in (0, 1)]
RUN = "gen60k_lr4_s0"


def batch(env, test_set):
    import torch
    m = env.model
    env.test_set = test_set
    env.run_generators()
    env.prep_model()
    m.MC_to_interactions()
    obs = torch.as_tensor(env.obs_flat, dtype=torch.float32, device=env.device)
    return m.active_Z.float(), obs


def posterior_known(env, obs, chans=(0, 1, 2, 3)):
    """(B, R*R) exact Bayes posterior over the joint value from the given channels (true likelihood)"""
    import torch
    L = torch.as_tensor(env.joint_likelihood, dtype=torch.float64, device=env.device).clamp(1e-6, 1 - 1e-6)
    n = obs.double().sum(1)                                                       # (B, O)
    ll = sum(n[:, o, None, None] * L[:, o].log() + (T - n[:, o, None, None]) * (1 - L[:, o]).log() for o in chans)
    return torch.softmax(ll.reshape(len(n), -1), -1).float()


def plugin_acc(env, chans):
    """goal accuracy of exact Bayes with the Generator's learned likelihood, from the given channels"""
    import torch
    m, dev = env.model, env.device
    B = m.batch_num
    with torch.no_grad():
        P = []
        for r in range(R * R):
            m.get_prediction(torch.tensor(np.unravel_index(r, (R, R)), device=dev).expand(B, C).contiguous(),
                             torch.ones(B, C, device=dev))
            P.append(m.pred_pobs.float())
        P = torch.stack(P, -1).clamp(1e-6, 1 - 1e-6)[:, list(chans)]               # (B, |chans|, R*R)
        n = torch.as_tensor(env.obs_flat, dtype=torch.float32, device=dev).sum(1)[:, list(chans)]
        ll = (n[..., None] * P.log() + (T - n[..., None]) * (1 - P).log()).sum(1)
        post = torch.softmax(ll, -1).reshape(B, R, R)
        g = m.goal_ind.long()
        marg = torch.where(g[:, None] == 0, post.sum(2), post.sum(1))
        return (marg.argmax(-1) == m.goal_value.long()).float().mean().item()


def reader_acc(reader, Z, obs, env, chans):
    import torch
    m = env.model
    with torch.no_grad():
        post = torch.softmax(reader(Z, obs, chans), -1).reshape(-1, R, R)
        g = m.goal_ind.long()
        marg = torch.where(g[:, None] == 0, post.sum(2), post.sum(1))
        return (marg.argmax(-1) == m.goal_value.long()).float().mean().item()


def code_r2(env, z4):
    zt = np.asarray(env.joint_Z)[:, NEW]
    X = np.c_[zt, np.ones(len(zt))]
    W, *_ = np.linalg.lstsq(X, z4, rcond=None)
    res = z4 - X @ W
    return float(1 - (res ** 2).sum() / ((z4 - z4.mean(0)) ** 2).sum())


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--reader-steps", type=int, default=20000)
    ap.add_argument("--test-every", type=int, default=250)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)                # 11-11 19:40: 1e-3 diverged (loss rose); the 60k nets
    #                                                                 # trained their embeddings at 1e-4
    ap.add_argument("--init", choices=("random", "corrupt"), default="corrupt",
                    help="channel 4's slices redrawn at random, or a 50/50 mix of the trained slices and random ones")
    ap.add_argument("--mix", type=float, default=.5)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    learner, labels, seed = TASKS[a.task]
    tag = f"{learner}_{labels}_s{seed}"
    tag += "" if a.init == "random" else f"_{a.init}{int(round(100 * a.mix))}"
    if a.smoke:
        a.steps, a.batch, a.reader_steps, a.test_every, tag = 4, 64, 4, 2, "smoke_" + tag
    np.random.seed(0); torch.manual_seed(0)
    env = fg.build(RUN, a.batch, load=True)
    m = env.model
    for p in m.parameters():
        p.requires_grad_(False)
    m.embedding_grad = "generator"
    reader = None
    if learner == "discriminative":                     # the channel-wise reader, trained on the ORIGINAL embeddings
        torch.manual_seed(100 + seed)
        reader = trd.readers()["shared"]()
        ropt = torch.optim.Adam(reader.parameters(), lr=1e-3)
        for step in range(a.reader_steps):
            with torch.no_grad():
                Z, obs = batch(env, False)
            v = m.ctx_vals.long()
            loss = torch.nn.functional.cross_entropy(reader(Z, obs), v[:, 0] * R + v[:, 1])
            ropt.zero_grad(); loss.backward(); ropt.step()
        for p in reader.parameters():
            p.requires_grad_(False)
    torch.manual_seed(1000 + seed)                      # the new sensor: channel 4's slices redrawn (or corrupted)
    with torch.no_grad():
        for X in (m.all_K, m.all_Q):
            x = torch.randn(X.shape[0], X.shape[2], device=X.device)
            x = x / x.norm(dim=-1, keepdim=True)
            if a.init == "corrupt":                     # 11-11 19:45 (John: a faster proof of concept): start near the
                x = a.mix * X.data[:, NEW] + (1 - a.mix) * x    # basin, so the direction each objective takes shows fast
                x = x / x.norm(dim=-1, keepdim=True)
            X.data[:, NEW] = x
    mask = torch.zeros(1, O, 1, device=env.device)
    mask[0, NEW] = 1
    for X in (m.all_K, m.all_Q):
        X.requires_grad_(True)
        X.register_hook(lambda g: g * mask)             # only channel 4's slices learn
    opt = torch.optim.Adam([m.all_K, m.all_Q], lr=a.lr)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{tag}.jsonl")
    log = open(path, "w")
    t0, losses = time.time(), []
    print(f"=== {tag}: {RUN}, channel {NEW} redrawn, batch {a.batch}, {a.steps} steps", flush=True)
    for step in range(a.steps + 1):
        if step % a.test_every == 0 or step == a.steps:
            with torch.no_grad():
                Z, obs = batch(env, True)
            rec = dict(step=step, episodes=step * a.batch, t=round(time.time() - t0), code_r2=code_r2(env, Z[:, NEW].cpu().numpy()),
                       loss=float(np.mean(losses)) if losses else None)
            if learner == "generative":
                rec.update(acc_known=plugin_acc(env, (0, 1, 2, 3)), acc_all=plugin_acc(env, tuple(range(O))))
            else:
                rec.update(acc_known=reader_acc(reader, Z, obs, env, (0, 1, 2, 3)), acc_all=reader_acc(reader, Z, obs, env, tuple(range(O))))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step:5d} {rec['t']:5d}s | code R2 {rec['code_r2']:.3f} | acc 0-3 {rec['acc_known']:.3f}, "
                  f"+ channel 4 {rec['acc_all']:.3f}", flush=True)
            losses = []
            if step == a.steps:
                break
        Z, obs = batch(env, False)
        if labels == "true":
            v = m.ctx_vals.long()
            joint = v[:, 0] * R + v[:, 1]
            soft = None
        else:
            soft = posterior_known(env, obs)
            joint = torch.multinomial(soft, 1).squeeze(1)
        if learner == "generative":
            r = torch.stack([joint // R, joint % R], 1)
            m.get_prediction(r.contiguous(), torch.ones(a.batch, C, device=env.device))
            loss = m.DKL_sym(m.pred_pobs, m.obs_flat.mean(1))[:, NEW].mean()
        else:
            logits = reader(Z, obs)
            loss = (torch.nn.functional.cross_entropy(logits, joint) if soft is None
                    else -(soft * torch.log_softmax(logits, -1)).sum(-1).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print("wrote", path)


if __name__ == "__main__":
    main()
