"""Can a fresh Generator use embeddings learned through the Classifier? (John 11-11: "Can you please train a fresh generator
on the classifier route's learned embeddings to see if they are usable?")

Loads a Fig 13 60k checkpoint (the environment's truth + the learned embeddings), replaces the Generator by the freshly
initialized one of a new model of the same configuration, freezes the embeddings (all_K, all_Q, K/Q_downscale) and trains
only the Generator, on the true values with confidence 1 (the oracle Generator loss: symmetric KL between the predicted
and the episode's observed frequency per channel), Adam, lr 1e-3. Tests every TEST_EVERY steps: plug-in accuracy (exact
Bayes with the learned likelihood, goal at T = 30) and kl_env (KL(P(o | r) || P^(o | r)) summed over channels, at the
episode's state) on novel and on training environments.

Conditions (fig13_embeddings.py found every Classifier-route network's channel codes under permuted labels):
    own        Generator-route embeddings as learned (positive control)
    as learned Classifier-route embeddings as learned
    relabeled  the same with the model's channels permuted back (--perm: model channel o <- the channel whose Z carries
               observation channel o's likelihood, fig13_embeddings' cross matrix)
    random     the fresh model's random embeddings (negative control; the loaded run's environment)
-> runs/fresh_generator/<tag>.jsonl

    python fresh_generator.py --task k        (TASKS below; cluster array)
"""
import os, sys, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
OUT = os.path.join(ROOT, "runs", "fresh_generator")
NETS = os.environ.get("FG_NETS", "/cluster_runs")
GEN = ("Z_to_pobs", "sample_to_emb", "sample_to_hid", "gen_hid2hid", "conf_to_hid", "hid_to_pobs", "gen_stream_norm")
EMB = ("all_K", "all_Q", "K_downscale", "Q_downscale")
# 11-11 14:40: the codes turned out to be exact linear copies of the true Z per channel (learned = A_o true, A_o a sign
# flip and / or swap of the two numbers); the Generator route uses ONE A for all channels, the Classifier route a
# different A_o per channel. Tasks 9+: transform at the Z level instead (frame = "aligned": permute the slots AND undo each
# channel's A_o, estimated on a training batch and snapped to the nearest signed permutation; "scrambled": give a
# Generator-route code a different random frame per channel, the causal control)
TASKS = [("gen60k_lr4_s0", None, False, "gen_s0_own"),
         ("clf60k_lr4_s0", None, False, "clf_s0_as_learned"),
         ("clf60k_lr4_s0", (1, 0, 4, 2, 3), False, "clf_s0_relabeled"),
         ("clf60k_lr4_s1", None, False, "clf_s1_as_learned"),
         ("clf60k_lr4_s1", (3, 1, 0, 4, 2), False, "clf_s1_relabeled"),
         ("gen60k_lr4_s0", None, True, "random"),
         ("rcc60k_lr4_s0", None, False, "rcc_s0_own"),
         ("rccclf60k_lr4_s1", None, False, "rccclf_s1_as_learned"),
         ("rccclf60k_lr4_s1", (2, 1, 0, 4, 3), False, "rccclf_s1_relabeled"),
         ("clf60k_lr4_s0", (1, 0, 4, 2, 3), "aligned", "clf_s0_aligned"),
         ("clf60k_lr4_s1", (3, 1, 0, 4, 2), "aligned", "clf_s1_aligned"),
         ("rccclf60k_lr4_s1", (2, 1, 0, 4, 3), "aligned", "rccclf_s1_aligned"),
         ("gen60k_lr4_s0", None, "scrambled", "gen_s0_scrambled")]
# 11-11 17:40 (John: "add the learnable correction ... and also see if it holds for MLP maps"): a learnable map of the 10
# code numbers in front of the fresh Generator, trained with it: "map_linear" (10 x 10, identity at the start) or
# "map_mlp" (identity + an MLP 10 -> 128 -> 128 -> 10 whose last layer starts at zero); TASKS 13-20
TASKS += [(run, None, kind, f"{tag}_{kind}") for kind in ("map_linear", "map_mlp")
          for run, tag in (("clf60k_lr4_s0", "clf_s0"), ("clf60k_lr4_s1", "clf_s1"), ("rccclf60k_lr4_s1", "rccclf_s1"),
                           ("gen60k_lr4_s0", "gen_s0"))]
FIX = None                                          # Z-level transform applied after every MC_to_interactions


def make_map(kind, O=5, hidden=128):
    """a learnable map of the code (B, O, 2) -> (B, O, 2) over all 10 numbers, the identity at the start"""
    import torch
    nn = torch.nn

    class Map(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            if kind == "map_linear":
                self.A = nn.Parameter(torch.eye(2 * O))
            else:
                self.g = nn.Sequential(nn.Linear(2 * O, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                       nn.Linear(hidden, 2 * O))
                nn.init.zeros_(self.g[-1].weight)
                nn.init.zeros_(self.g[-1].bias)

        def forward(self, Z):
            z = Z.reshape(len(Z), -1)
            z = z @ self.A.T if self.kind == "map_linear" else z + self.g(z)
            return z.reshape(Z.shape)

    return Map()


def build(name, batch, load):
    import torch
    from main.CognitiveGridworld import CognitiveGridworld
    import train_cluster_runs as tcr
    cfg = dict(next(c for n, c, s in tcr.TASKS if n == name))
    cfg.update(batch_num=batch, episodes=1, training=False, checkpoint_every=1, show_plots=False, cuda=0)
    if load:
        cfg.update(load_env=f"{NETS}/{name}")
    return CognitiveGridworld(**cfg)


def bern_kl(p, q):
    p, q = p.clamp(1e-6, 1 - 1e-6), q.clamp(1e-6, 1 - 1e-6)
    return (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).sum(-1)


def test(env, test_set):
    import torch
    m, dev = env.model, env.device
    B, R, C = m.batch_num, env.realization_num, env.ctx_num
    env.test_set = test_set
    env.run_generators()
    env.prep_model()
    BR = torch.arange(B, device=dev)
    with torch.no_grad():
        m.MC_to_interactions()
        if FIX is not None:
            m.active_Z = FIX(m.active_Z)
        ones = torch.ones(B, C, device=dev)
        P = []
        for r in range(R * R):
            m.get_prediction(torch.tensor(np.unravel_index(r, (R, R)), device=dev).expand(B, C).contiguous(), ones)
            P.append(m.pred_pobs.float())
        P = torch.stack(P, -1).clamp(1e-6, 1 - 1e-6)                                       # (B, O, R*R)
        o = torch.as_tensor(env.obs_flat, device=dev, dtype=torch.float32)                  # (B, T, O)
        ll = torch.einsum("bto,bok->bk", o, P.log()) + torch.einsum("bto,bok->bk", 1 - o, (1 - P).log())
        post = torch.softmax(ll, -1).reshape(B, R, R)
        g, v = m.goal_ind.long(), m.ctx_vals.long()
        marg = torch.where(g[:, None] == 0, post.sum(2), post.sum(1))
        hit = (marg.argmax(-1) == m.goal_value.long()).float().mean().item()
        L = torch.as_tensor(env.joint_likelihood, device=dev, dtype=torch.float32)[BR, :, v[:, 0], v[:, 1]]
        kl = bern_kl(L, P[BR, :, v[:, 0] * R + v[:, 1]]).mean().item()
    return hit, kl


def signed_perm(A):
    """the 2 x 2 signed permutation matrix nearest to A (the code's symmetries: sign flips and the swap)"""
    best = None
    for P in (np.eye(2), np.array([[0., 1.], [1., 0.]])):
        for s in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            Q = P * np.array(s)[None, :]
            d = np.abs(A - Q * np.abs(A).max()).sum()
            if best is None or d < best[0]:
                best = (d, Q)
    return best[1]


def z_frame(env, perm, frame, seed):
    """returns Z -> Z' (B, O, 2): "aligned" = slots permuted (perm) and each channel's frame A_o undone so every channel
    reads learned = true; "scrambled" = a different random signed permutation per channel (seeded)"""
    import torch
    m = env.model
    O = env.obs_num
    if frame == "aligned":
        env.test_set = False
        env.run_generators()
        env.prep_model()
        with torch.no_grad():
            m.MC_to_interactions()
        L = m.active_Z.detach().cpu().numpy()[:, list(perm)]                     # (B, O, 2) matched to channel o
        T = np.asarray(env.joint_Z)                                               # (B, O, 2) true
        M = []
        for o in range(O):
            A, *_ = np.linalg.lstsq(T[:, o], L[:, o], rcond=None)                # L = T A
            M.append(np.linalg.inv(signed_perm(A)))                              # L A^-1 = T
        print("frames A_o (L = T A):", [signed_perm(np.linalg.lstsq(T[:, o], L[:, o], rcond=None)[0]).tolist()
                                         for o in range(O)], flush=True)
        idx = list(perm)
    else:
        rng = np.random.default_rng(seed + 77)
        frames = [np.eye(2), -np.eye(2), np.array([[0., 1.], [1., 0.]]), -np.array([[0., 1.], [1., 0.]])]
        M = [frames[k] for k in rng.permutation(np.r_[0:4, rng.integers(0, 4, O - 4)])[:O]]   # the 4 frames the
        # Classifier route uses (+-identity, +-swap), all four present, one repeated
        print("random frames:", [x.tolist() for x in M], flush=True)
        idx = list(range(O))
    Mt = torch.tensor(np.stack(M), dtype=torch.float32, device=env.device)      # (O, 2, 2)
    return lambda Z: torch.einsum("bon,onk->bok", Z[:, idx], Mt)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-every", type=int, default=250)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--suffix", default="", help="appended to the tag (11-11: the 10x longer reruns, _long)")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    global FIX
    name, perm, random_emb, tag = TASKS[a.task]
    tag += a.suffix
    frame = random_emb if isinstance(random_emb, str) else None
    random_emb = random_emb is True
    if a.smoke:
        a.steps, a.batch, a.test_every, tag = 4, 64, 2, "smoke_" + tag
    seed = int(name.rsplit("_s", 1)[1])
    np.random.seed(seed); torch.manual_seed(seed)
    env = build(name, a.batch, load=True)
    np.random.seed(seed + 1000); torch.manual_seed(seed + 1000)
    fresh = build(name, a.batch, load=False).model                     # a freshly initialized model of the same config
    m = env.model
    with torch.no_grad():
        for n in GEN + (EMB if random_emb else ()):
            src = getattr(fresh, n)
            if isinstance(src, torch.nn.Module):
                getattr(m, n).load_state_dict(src.state_dict())
            else:
                getattr(m, n).data.copy_(src.data)
        if perm is not None and frame is None:                          # model channel o <- channel perm[o]
            for n in ("all_K", "all_Q"):
                getattr(m, n).data.copy_(getattr(m, n).data[:, list(perm)])
    del fresh
    learned_map = None
    if frame in ("map_linear", "map_mlp"):                              # a learnable correction (tasks 13+)
        learned_map = make_map(frame).to(env.device)
        FIX = learned_map
        env.model.embedding_grad = "both"               # else generator_Z() detaches Z (Classifier-route networks) and the
        #                                               # map gets no gradient; the embeddings themselves stay frozen
    elif frame is not None:                                             # Z-level frames (tasks 9-12)
        FIX = z_frame(env, perm, frame, seed)
    for n in EMB:
        x = getattr(m, n)
        for p in (x.parameters() if isinstance(x, torch.nn.Module) else [x]):
            p.requires_grad_(False)
    params = [p for n in GEN for p in getattr(m, n).parameters()] + (list(learned_map.parameters()) if learned_map else [])
    opt = torch.optim.Adam(params, lr=a.lr)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{tag}.jsonl")
    log = open(path, "w")
    t0, losses = time.time(), []
    print(f"=== {tag}: {name}, perm {perm}, random {random_emb}, batch {a.batch}, lr {a.lr}, {a.steps} steps", flush=True)
    for step in range(a.steps + 1):
        if step % a.test_every == 0 or step == a.steps or (step <= 2000 and step % 250 == 0):
            nov, kl_nov = test(env, True)
            tr, kl_tr = test(env, False)
            rec = dict(step=step, t=round(time.time() - t0), loss=float(np.mean(losses)) if losses else None,
                       novel_plugin=nov, novel_kl_env=kl_nov, train_plugin=tr, train_kl_env=kl_tr)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step:5d} {rec['t']:6d}s | loss {rec['loss'] if rec['loss'] is None else round(rec['loss'], 4)} | "
                  f"novel plug-in {nov:.3f} kl {kl_nov:.3f} | train plug-in {tr:.3f} kl {kl_tr:.3f}", flush=True)
            losses = []
            if step == a.steps:
                break
        env.test_set = False
        env.run_generators()
        env.prep_model()
        m.MC_to_interactions()
        if FIX is not None:
            m.active_Z = FIX(m.active_Z)
        m.get_prediction(m.ctx_vals, torch.ones(a.batch, env.ctx_num, device=env.device))
        loss = m.DKL_sym(m.pred_pobs, m.obs_flat.mean(1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    if learned_map is not None:                         # how far the learned correction moved from the identity
        import torch as _t
        sd = {k: v.detach().cpu().numpy() for k, v in learned_map.state_dict().items()}
        np.savez(path[:-6] + "_map.npz", **sd)
        dev = (np.abs(sd["A"] - np.eye(len(sd["A"]))).sum() if "A" in sd else np.abs(sd["g.4.weight"]).sum())
        print(f"map moved: sum |change| = {dev:.4f}", flush=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
