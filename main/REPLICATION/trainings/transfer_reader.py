"""Do Generator-learned embeddings transfer to a FRESH reader faster than Classifier-learned ones? (John 11-11: "is there a
light, low batch version of this test you could run quickly? ... Let's check the abstraction argument, we really need to
find a benefit for transfer.")

A fresh, small reader is trained on the embeddings of a Fig 13 60k network (oracle feedback: the true joint value is
the target), at small batch on CPU. The embeddings give each environment its code Z (5 channels x 2 numbers, the model's
own MC_to_interactions); they are FROZEN, or UPDATED by the reader's loss (the Classifier route of a new learner).
Sources: Generator route (gen60k_lr4_s0), Classifier route (clf60k_lr4_s0), random (a fresh model's embeddings, same
world). Readers (all output the 100 joint logits at T = 30; cross-entropy to the true joint value; Adam, lr 1e-3):
    lstm      unstructured, like the RCC's Classifier at small scale: LSTM(256) over steps, input [Z (10), o_t (5)]
    shared    Bayes-structured: per-channel evidence e(Z_o, n_o / T) -> 100 logits by ONE function for every channel
              (MLP 3 -> 128 -> 128 -> 100), summed over channels (channels are independent given the values, so Joint
              Bayes has exactly this form)
    unshared  the same with a separate evidence function per channel
Symmetry note: the two routes' codes differ by sign flips / swaps of the two numbers per channel and a slot permutation
(fig13_embeddings.py); for lstm and unshared with FROZEN codes that is a signed permutation of the input coordinates,
under which the random initialization and Adam are invariant, so their learning is the same in distribution (up to the
codes' 5% scale): the expected null. Only a reader that shares a function across channels can tell the codes apart.
Tests every TEST_EVERY steps: goal accuracy at T on 2,048 novel-environment and 2,048 training-environment episodes.
-> runs/transfer_reader/<tag>.jsonl

    python transfer_reader.py --task k [--smoke]
"""
import os, sys, json, time, argparse, itertools
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
import fresh_generator as fg

OUT = os.path.join(ROOT, "runs", "transfer_reader")
BASE = (("gen", "gen60k_lr4_s0"), ("clf", "clf60k_lr4_s0"), ("random", "gen60k_lr4_s0"))   # TASKS 0-29 are built on
SOURCES = BASE + (("clf_slots", "clf60k_lr4_s0"), ("clf_aligned", "clf60k_lr4_s0"))           # these 3 only (fixed order)
READERS = ("shared", "unshared", "lstm")
MODES = ("frozen", "updated")
TASKS = [(s, r, m, k) for r, (s, _), m, k in itertools.product(READERS, BASE, MODES, (0, 1))
         if not (s == "random" and m == "frozen")]                      # 3 readers x (2 frozen + 3 updated) x 2 seeds = 30
# 11-11 15:50 causal controls (frozen), TASKS 30-37 (a first submission built TASKS over all 5 sources, which shifted the
# indices; it was cancelled within a minute, but its two LSTM tasks had already truncated lstm_gen_frozen_s0/s1.jsonl of
# the running tasks 20-21: their first ~300 steps are lost, the rest intact after the file was cleaned of NUL bytes): the Classifier-route code with its slots put back ("clf_slots") or its slots and
# each channel's frame put back ("clf_aligned"), Z-level transforms as in fresh_generator.py (perm of clf60k_lr4_s0)
TASKS += [(s, r, "frozen", k) for r in ("shared", "unshared") for s in ("clf_slots", "clf_aligned") for k in (0, 1)]
# 11-11 16:45 (John): (i) EXTRAPOLATION to a new channel: readers trained on channels 0-3 only (frozen embeddings), tested
# with channels 0-3 and with channel 4 added (a reader that applies one evidence function to every channel can use it
# zero-shot); TASKS 38-61. (ii) PLASTICITY: embeddings updated while a fresh reader learns, with the network's own
# Generator kept training on them ("updated_gen": the Generator's observation-prediction loss as an anchor) or at a
# gentler embedding lr 1e-4 ("updated_slow"); TASKS 62-77.
TRAIN_CH = (0, 1, 2, 3)
TASKS += [(s, r, "frozen_ch4", k) for r in ("shared", "unshared", "lstm") for s in ("gen", "clf", "clf_slots", "clf_aligned")
          for k in (0, 1)]
TASKS += [(s, r, m, k) for m in ("updated_gen", "updated_slow") for r in ("lstm", "shared") for s in ("gen", "clf")
          for k in (0, 1)]
# 11-11 17:20 (John: "what about learning this correction?"): "shared_adapt" = a learnable linear map of the 10 code
# numbers (initialized to the identity: no correction) in front of the shared evidence reader; it can represent the exact
# slot + frame correction. All channels, and channels 0-3 with channel 4 added at test; TASKS 78-85.
TASKS += [(s, "shared_adapt", m, k) for m in ("frozen", "frozen_ch4") for s in ("gen", "clf") for k in (0, 1)]
# 11-11 17:45 (John: also "MLP maps, because maybe it is easy to learn but just needs nonlinearity", for every panel with
# a hand correction): the shared reader with an MLP map, and the unshared reader with a linear or MLP map (fg.make_map:
# identity at the start); all channels and the channel-4 test; TASKS 86-109
TASKS += [(s, r, m, k) for r in ("shared_mlpmap", "unshared_adapt", "unshared_mlpmap") for m in ("frozen", "frozen_ch4")
          for s in ("gen", "clf") for k in (0, 1)]
CLF_PERM = (1, 0, 4, 2, 3)
FIX = None
EMB = ("all_K", "all_Q", "K_downscale", "Q_downscale")
R, C, O, T = 10, 2, 5, 30


def readers():
    import torch
    nn = torch.nn

    class Evidence(nn.Module):
        def __init__(self, shared):
            super().__init__()
            mk = lambda: nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, R * R))
            self.shared = shared
            self.f = mk() if shared else nn.ModuleList([mk() for _ in range(O)])

        def forward(self, Z, obs, ch=tuple(range(O))):               # Z (B, O, 2), obs (B, T, O); ch = channels used
            u = torch.cat([Z, obs.mean(1)[..., None]], -1)             # (B, O, 3)
            if self.shared:
                return self.f(u[:, list(ch)]).sum(1)
            return sum(self.f[o](u[:, o]) for o in ch)

    class LSTMReader(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(O * 2 + O, 256, batch_first=True)
            self.out = nn.Linear(256, R * R)

        def forward(self, Z, obs, ch=tuple(range(O))):               # unused channels are zeroed (Z and bits)
            keep = torch.zeros(O, device=Z.device)
            keep[list(ch)] = 1
            Z, obs = Z * keep[None, :, None], obs * keep[None, None, :]
            x = torch.cat([Z.reshape(len(Z), 1, -1).expand(-1, obs.shape[1], -1), obs], -1)
            h, _ = self.rnn(x)
            return self.out(h[:, -1])

    class Adapted(nn.Module):
        """a learnable linear map of the 10 code numbers (identity at the start) before a reader"""
        def __init__(self, base):
            super().__init__()
            self.A = nn.Parameter(torch.eye(O * 2))
            self.base = base

        def forward(self, Z, obs, ch=tuple(range(O))):
            return self.base((Z.reshape(len(Z), -1) @ self.A.T).reshape(len(Z), O, 2), obs, ch)

    class Mapped(nn.Module):
        """fresh_generator.make_map (linear or residual MLP, identity at the start) before a reader"""
        def __init__(self, base, kind):
            super().__init__()
            self.map = fg.make_map(kind)
            self.base = base

        def forward(self, Z, obs, ch=tuple(range(O))):
            return self.base(self.map(Z), obs, ch)

    return {"shared": lambda: Evidence(True), "unshared": lambda: Evidence(False), "lstm": LSTMReader,
            "shared_adapt": lambda: Adapted(Evidence(True)),
            "shared_mlpmap": lambda: Mapped(Evidence(True), "map_mlp"),
            "unshared_adapt": lambda: Mapped(Evidence(False), "map_linear"),
            "unshared_mlpmap": lambda: Mapped(Evidence(False), "map_mlp")}


def batch(env, test_set, grad):
    import torch
    m = env.model
    env.test_set = test_set
    env.run_generators()
    env.prep_model()
    with torch.set_grad_enabled(grad):
        m.MC_to_interactions()
    Z = m.active_Z if grad else m.active_Z.detach()
    if FIX is not None:
        Z = FIX(Z)
    obs = torch.as_tensor(env.obs_flat, dtype=torch.float32, device=env.device)
    v = m.ctx_vals.long()
    return Z.float(), obs, v[:, 0] * R + v[:, 1], m.goal_ind.long(), m.goal_value.long()


def accuracy(env, reader, n_batches, ch=tuple(range(O))):
    import torch
    hits = []
    with torch.no_grad():
        for _ in range(n_batches):
            Z, obs, _, g, gv = batch(env, env._test_flag, False)
            post = torch.softmax(reader(Z, obs, ch), -1).reshape(-1, R, R)
            marg = torch.where(g[:, None] == 0, post.sum(2), post.sum(1))
            hits.append((marg.argmax(-1) == gv).float().mean().item())
    return float(np.mean(hits))


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-every", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--suffix", default="", help="appended to the tag (11-11: the 10x longer reruns, _long)")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    src, rname, mode, seed = TASKS[a.task]
    tag = f"{rname}_{src}_{mode}_s{seed}"
    tag += a.suffix
    if a.smoke:
        a.steps, a.batch, a.test_every, tag = 6, 64, 3, "smoke_" + tag
    run = dict(SOURCES)[src]
    np.random.seed(int(run.rsplit("_s", 1)[1])); torch.manual_seed(0)
    env = fg.build(run, a.batch, load=True)
    m = env.model
    if src == "random":                                                 # a fresh model's embeddings, the same world
        np.random.seed(1000 + seed); torch.manual_seed(1000 + seed)
        fresh = fg.build(run, a.batch, load=False).model
        with torch.no_grad():
            for n in EMB:
                x = getattr(fresh, n)
                if isinstance(x, torch.nn.Module):
                    getattr(m, n).load_state_dict(x.state_dict())
                else:
                    getattr(m, n).data.copy_(x.data)
        del fresh
    global FIX
    if src == "clf_slots":
        FIX = lambda Z: Z[:, list(CLF_PERM)]
    elif src == "clf_aligned":
        FIX = fg.z_frame(env, CLF_PERM, "aligned", seed)
    np.random.seed(seed + 7); torch.manual_seed(seed + 7)
    reader = readers()[rname]()
    emb_params = [p for n in EMB for p in (getattr(m, n).parameters() if isinstance(getattr(m, n), torch.nn.Module)
                                           else [getattr(m, n)])]
    for p in m.parameters():
        p.requires_grad_(False)
    plastic = mode.startswith("updated")
    gen_params = [p for n in fg.GEN for p in getattr(m, n).parameters()] if mode == "updated_gen" else []
    for p in emb_params * plastic + gen_params:
        p.requires_grad_(True)
    if mode == "updated_gen":
        m.embedding_grad = "both"                       # the Generator's gradient reaches the embeddings in both routes
    groups = [dict(params=list(reader.parameters()), lr=a.lr)]
    if plastic:
        groups.append(dict(params=emb_params, lr=1e-4 if mode == "updated_slow" else a.lr))
    if gen_params:
        groups.append(dict(params=gen_params, lr=a.lr))
    opt = torch.optim.Adam(groups)
    train_ch = TRAIN_CH if mode == "frozen_ch4" else tuple(range(O))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{tag}.jsonl")
    log = open(path, "w")
    t0, losses, nt = time.time(), [], (1 if a.smoke else 4)
    print(f"=== {tag}: source {run}, batch {a.batch}, lr {a.lr}, {a.steps} steps", flush=True)
    for step in range(a.steps + 1):
        if step % a.test_every == 0 or step == a.steps or (step <= 2000 and step % 100 == 0):
            env._test_flag = True
            nov = accuracy(env, reader, nt, train_ch)
            extra = dict(novel_all5=accuracy(env, reader, nt)) if mode == "frozen_ch4" else {}
            env._test_flag = False
            tr = accuracy(env, reader, nt, train_ch)
            rec = dict(step=step, episodes=step * a.batch, t=round(time.time() - t0), novel=nov, train=tr,
                       loss=float(np.mean(losses)) if losses else None, **extra)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step:5d} {rec['t']:5d}s | novel {nov:.3f} train {tr:.3f} | loss {rec['loss']}", flush=True)
            losses = []
            if step == a.steps:
                break
        Z, obs, y, _, _ = batch(env, False, plastic)
        loss = torch.nn.functional.cross_entropy(reader(Z, obs, train_ch), y)
        if mode == "updated_gen":                       # the Generator's loss on the same batch (true values, conf 1)
            m.get_prediction(m.ctx_vals, torch.ones(a.batch, C, device=env.device))
            loss = loss + m.DKL_sym(m.pred_pobs, m.obs_flat.mean(1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    if rname == "shared_adapt":                         # the learned correction, for the record
        np.save(path[:-6] + "_A.npy", reader.A.detach().cpu().numpy())
    print("wrote", path)


if __name__ == "__main__":
    main()
