"""How the Trained-vs-ESN divergence comes to track Factorization Regret during the first 1,000 training episodes.
(John 11-18: "can we also look at how the correlation between the Network's divergence and Bayes FR grows
throughout the first 1K training episodes (measuring once every 50 or 100 episodes)? Is it feasable to compute this?")

Data, all stored already (no replay): the training-dynamics pair trained_s20261029 / echo_s20261029 (the standard e5
Trained RNN and ESN, the same seed, environments and episode sequence; train_training_dynamics.py), their log beliefs
`logq` (float16) at every checkpoint on the 10,000 probe trajectories (main/DATA/experimental/training_dynamics/analysis/
{trained,echo}/ck_<E>.npz: every 25 episodes to 500, every 100 to 3,000; the end of training = ck_49995) and the probes'
exact Bayes side (analysis/probes.npz: 50 environments x 200 trajectories, T = 30; fr = forward FR per variable, log_jb,
log_nb). Episodes count every episode, as in the learning figures (every 5th is a test episode and trains nothing).
Per checkpoint, trajectory, step and variable:
    proxy     = D_KL(B^Trained_E || B^ESN_E), both networks at the same episode E
    controls  = D_KL(B^Trained_E || B^Naive) and D_KL(B^Joint || B^ESN_E) (one network replaced by its exact observer)
r = Spearman with FR within one environment, step and variable across its 200 trajectories, averaged over the 50
environments, both variables and all 30 steps. Accuracy = the goal variable's at T on the same probes.

RESULT 11-18 (probes; the two networks are identical at E = 0, so the curves start at 25):
    Trained vs ESN r: .06 at 25, .07 at 250, .14 at 500, .23 at 1,000, .39 at 2,000, .57 at 5,000, .65 at 10,000,
    .69 at 20,000 and at the end (49,995) -- the 39-environment scan of the original pair gave .70.
    Trained vs exact Naive: .27 at 500, .45 at 1,000, .68 at 2,000, .91 at 10,000, .98 at the end.
    exact Joint vs ESN: .26 at 500, .34 at 1,000, .50 at 2,000, .70 at 10,000, .70 at the end.
    Accuracy at T (goal): Trained .28 / .44 / .64 / .76 at 500 / 1,000 / 2,000 / end, passes Naive (.464) at 1,100 on
    these probes; ESN .18 / .21 / .27 / .43. So within the first 1,000 episodes the correlation is still small (rising
    ~.2 per 1,000); it grows mostly between 1,000 and 10,000, while the Trained RNN passes Naive and the ESN learns.

build_big() (John 11-18, more data for fig_fr_proxy_big.png (a)): the same 50 environments x 5,000 trajectories
(25x the probes), both networks replayed on a GPU (matches the stored float16 logq to .016), ~9 min ->
fr_proxy_training_big.npz. RESULT, within environment x step x variable, averaged (Pearson raw / Pearson of logs /
Spearman): 25 .108 .061 .054; 50 .098 .057 .045; 75 .001 .006 -.010; 100 .015 -.005 -.004; 250 .131 .056 .061;
500 .193 .144 .140; 1,000 .284 .242 .231; end (49,995) .830 .698 .688. The dip at 75-100 episodes is real (all three
measures, 25x the data); SEM over the 50 environments <= .025.
Why r is not 0 after 25 episodes (John 11-18; fr_proxy_early_diag.py, 20 environments x 5,000, Pearson):
at E = 0 the networks are identical (no correlation); at 25 / 50 episodes their divergence tracks the number of
observations that were on so far (r -.65 / -.62), which FR also tracks weakly (r -.15): with that count partialled
out r falls .115 -> .017 and .123 -> .030. The dip at 75-100 is where that count dependence disappears (r(D, count)
-.02 at 75; the two networks' confidence crosses there: KL to uniform 1.15 vs 1.22 at 75, Trained 1.53 vs ESN 1.25 at
25). The count-free (partial) r then grows: .08 at 150, .09 at 250, .15 at 500, .27 at 1,000, .82 at the end.
Evidence strength (KL(B^Joint || uniform)) explains none of it (partial r = r at every checkpoint).
build_dense() (every test episode to 100; re-run weights bit-identical to the original at 0, 25, 50, 75, 100 in all 10
tensors of both networks), Pearson: 5 .057, 10 .025, 15 .047, 20 .093, 25 .108, 30 .110, 35 .108, 40 .109, 45 .111,
50 .098, 55 .097, 60 .090, 65 .083, 70 .015, 75 .001, 80 .014, 85 .029, 90 .056, 95 .025, 100 .015 (identical to the
original run's values at 25-100): the count-driven bump rises over 20-45 episodes and collapses between 65 and 70.
Caches since 11-18 also hold count (r of the divergence with the observations-equal-to-1 count), partial
(r(D, FR | count)) and fr_count (r(FR, count): -.148 over steps, -.143 at T). r(D, count) all steps / T: 5 -.18/-.24,
25 -.66/-.77, 45 -.71/-.83, 65 -.59/-.49, 70 -.16/+.19, 75 -.02/+.20, 100 -.09/-.05, 150 -.22/-.27, 500 -.29/-.28,
1,000 -.16/-.17, end -.12/-.11. Partial r stays within +-.06 to 100 episodes, then .06/.05 (150), .15/.21 (500),
.27/.33 (1,000), .83/.85 (end). Final step only (T), Pearson r with FR: .08 (5), .12 (25), -.05/-.06 (70/75), .24
(500), .34 (1,000), .85 (end); fig_fr_proxy.py --big --final draws everything at T.
CORRECTION 11-18 (R^2 = r^2 per environment, step and variable, then averaged): the count link does NOT fade at
65-75 episodes. It is consistently negative at 25-65 (92% of units) but at 70-100 its sign changes over the steps of an
episode (at 75: t = 5 -.46, t = 30 +.20), so mean r ~ 0 while mean R^2 stays .49; it weakens only after ~100 (R^2 .16).
R^2 (all steps / T): FR .08/.09 (5), .06/.06 (25-45), .04/.04 (75-100), .06/.06 (150), .10/.12 (500), .16/.21 (1,000),
.72/.73 (end); count .52/.47 (5), .59/.60 (25), .65/.69 (45), .49/.48 (75), .16/.13 (100), .22/.24 (150), .14/.15
(500), .08/.10 (1,000), .06/.07 (end): FR passes the count at ~600-700 episodes. The partial (count-removed) r averages
~0 to 100 episodes but its per-unit square is ~.03-.04 (signs mixed); consistently positive from ~150.
build_ext() (1,100-2,000, every 100): R^2 with FR (all steps / T) .18/.23 at 1,100, .24/.30 at 1,500, .30/.36 at 2,000;
with the count .07/.08 throughout (Pearson r with FR .30 at 1,100 -> .45 at 2,000).

    python fig_fr_proxy_training.py
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

AN = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics", "analysis")
C_PROXY, C_JE, C_TN, C_TR, C_ESN = "k", "#6a51a3", "#1b9e77", "#d95f02", "#7570b3"
CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training.npz")


def log_norm(lq):
    from scipy.special import logsumexp
    lq = np.asarray(lq, np.float64)
    return lq - logsumexp(lq, -1, keepdims=True)


def kl(lp, lq):
    """D_KL(p || q) over the last axis from log beliefs"""
    p = np.exp(lp)
    return np.where(p > 0, p * (lp - lq), 0.0).sum(-1)


def r_within(x, fr, env):
    """Spearman with FR within every environment, step and variable across its trajectories -> (n_env, T, 2)"""
    from scipy.stats import rankdata
    out = []
    for e in np.unique(env):
        m = env == e
        a, b = rankdata(x[m], axis=0), rankdata(fr[m], axis=0)
        a -= a.mean(0); b -= b.mean(0)
        out.append((a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)))
    return np.array(out)


RUNS = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
BIG_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_big.npz")
BIG_EPS = list(range(25, 501, 25)) + list(range(600, 1001, 100)) + [49995]
BIG_M, BIG_SEED0 = 50, 4_000_000
DENSE_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_dense.npz")
DENSE_EPS = list(range(5, 101, 5))                                     # every test episode: <net>_s20261029_ep105 re-runs


def _pearson(x, y):
    """correlation across axis 0 for every other index"""
    x, y = x - x.mean(0), y - y.mean(0)
    return (x * y).sum(0) / np.sqrt((x ** 2).sum(0) * (y ** 2).sum(0))


def build_big(eps=BIG_EPS, suffix="", cache=BIG_CACHE, dev=None, runs_dir=RUNS, e_chk=None, fmt="ckpt_ep{:05d}_net.pth"):
    """More data for the early-learning curve (John 11-18: "is there a way to get more data to smoothen these
    correlations?"): the probes' 50 environments (their tables L and context inputs Z, the run's own world) x 100 states
    x 50 trajectories (seed BIG_SEED0 + environment) = 250,000 trajectories, both networks replayed on a GPU at every
    checkpoint in BIG_EPS (checked first against the stored float16 logq on the probe trajectories). Per checkpoint,
    environment, step and variable: Pearson r of the raw values, Pearson r of the log values, Spearman r (~5 min)."""
    import time, torch
    sys.path.insert(0, os.path.join(ROOT, "main", "REPLICATION"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import plot_fig1c_all as F
    from fig_fr_proxy import logits_gpu
    from scipy.stats import rankdata
    dev = dev or f"cuda:{torch.cuda.device_count() - 1}"
    P = np.load(os.path.join(AN, "probes.npz"))
    L, Zenv, n_env = P["L"], P["Z"], len(P["L"])
    i0 = np.flatnonzero(P["env"] == 0)
    assert np.allclose(F.fr_streams(P["obs"][i0], L[0]), P["fr"][i0], rtol=1e-4, atol=1e-6), "probe tables differ"

    def weights(run, E):
        d = torch.load(os.path.join(runs_dir, f"{run}_s20261029{suffix}", fmt.format(E)), map_location="cpu",
                       weights_only=False)
        return {k: v.detach().float().to(dev) for k, v in d["weights"].items()}

    def replay(W, o, z):
        with torch.no_grad():
            r = logits_gpu(W, torch.from_numpy(o.astype(np.float32)).to(dev),
                           torch.from_numpy(np.repeat(np.asarray(z, np.float32)[None], len(o), 0)).to(dev))
            return r.double().cumsum(1).reshape(len(o), F.N_T, 2, F.N_R).log_softmax(-1).cpu().numpy()

    E_chk = e_chk if e_chk is not None else 25 if suffix else 500          # a re-run is checked against the original
    for run in (("trained", "echo") if E_chk >= 0 else ()):                # (replicates, new weights: E_chk < 0, none)
        W, ref = weights(run, E_chk), np.load(os.path.join(AN, run, f"ck_{E_chk:05d}.npz"))["logq"][i0].astype(np.float64)
        d = np.abs(np.maximum(replay(W, P["obs"][i0], Zenv[0]), -60) - np.maximum(ref, -60)).max()
        print(f"{run}{suffix} replay vs the original run's stored logq (E = {E_chk}, environment 0): max abs {d:.3f}",
              flush=True)
        assert d < .1
    obs = np.stack([F.cell_streams(L[e], 0, F.SCAN_CELLS, BIG_M, BIG_SEED0 + e)[0] for e in range(n_env)])
    fr = np.stack([F.fr_streams(obs[e], L[e]) for e in range(n_env)])     # (env, n, T, 2), raw order
    lfr = np.log(np.maximum(fr, 1e-12))
    rfr = rankdata(fr, axis=1)
    count = np.cumsum(obs.sum(-1), 2)[..., None].repeat(2, -1).astype(np.float64)  # observations equal to 1 so far
    fr_count = np.stack([_pearson(fr[e], count[e]) for e in range(n_env)])        # (env, T, 2): fixed by the task
    out = {k: np.zeros((len(eps), n_env, F.N_T, 2)) for k in ("pearson", "pearson_log", "spearman", "count",
                                                              "partial")}
    t0 = time.time()
    for i, E in enumerate(eps):
        WT, WE = weights("trained", E), weights("echo", E)
        for e in range(n_env):
            d = kl(replay(WT, obs[e], Zenv[e]), replay(WE, obs[e], Zenv[e]))
            out["pearson"][i, e] = _pearson(d, fr[e])
            out["pearson_log"][i, e] = _pearson(np.log(np.maximum(d, 1e-12)), lfr[e])
            out["spearman"][i, e] = _pearson(rankdata(d, axis=0), rfr[e])
            out["count"][i, e] = _pearson(d, count[e])                     # the early bump's driver (the count check)
            out["partial"][i, e] = (out["pearson"][i, e] - out["count"][i, e] * fr_count[e]) / np.sqrt(
                (1 - out["count"][i, e] ** 2) * (1 - fr_count[e] ** 2))
        print(f"checkpoint {E}: {time.time() - t0:.0f} s; r (Pearson, log, Spearman) "
              f"{np.nanmean(out['pearson'][i]):.3f} {np.nanmean(out['pearson_log'][i]):.3f} "
              f"{np.nanmean(out['spearman'][i]):.3f}", flush=True)
    np.savez(cache, E=np.array(eps), n_traj=obs.shape[1], fr_count=fr_count, **out)


EXT_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_ext.npz")
EXT_EPS = list(range(1100, 2001, 100))                                 # John 11-18: panel (a) to 2,000


def build_ext(dev=None):
    """the original run's checkpoints 1,100-2,000 (every 100), same streams and measures as build_big"""
    build_big(EXT_EPS, "", EXT_CACHE, dev)


FULL_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_full.npz")
FULL_EPS = list(range(2100, 3001, 100)) + list(range(3250, 10001, 250)) + list(range(11000, 49001, 1000))


def build_full(eps=FULL_EPS, dev=None, cache=FULL_CACHE):
    """John 11-18 ("a version of the R^2 with the full training if we have it for any"): every stored checkpoint of the
    original run after 2,000 episodes (every 100 to 3,000, every 250 to 10,000, every 1,000 to 49,000; 49,995 is in
    BIG_CACHE), same streams and measures as build_big; the replicates stop at 2,005 episodes"""
    build_big(list(eps), "", cache, dev)


B80K_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_b80k.npz")
B80K_EPS = list(range(1, 11)) + list(range(15, 101, 5))


def build_b80k(eps=B80K_EPS, dev=None, cache=B80K_CACHE):
    """John 11-18 ("run the 100 episode at batch 80K, though also save all of the first 10 episodes"): the pair
    retrained from the same seed at batch 80,000 (fr_proxy_runs/run_b80k.py: TD_BATCH=80000, TD_MICRO=16000 slices with
    accumulated gradients, TD_GPU_BAYES; checkpoints at episodes 0-10 and every 5 to 100) -> training_dynamics/batch80k/;
    the same streams and measures as build_big; checked at E = 0 (the same initial weights as the batch-8,000 run)"""
    build_big(eps, "_ep105", cache, dev, os.path.join(RUNS, "batch80k"), 0)


B100K_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_b100k_lr1e-1.npz")


def build_variant(sub, cache, eps=B80K_EPS, dev=None):
    """a retrained pair in training_dynamics/<sub>/<net>_s20261029_ep105 (fr_proxy_runs/run_bigbatch.py: batch, slices and
    learning rate as named; John 11-18 "can we try lr .1 at batch 100K for 100 episodes?" -> batch100k_lr1e-1), the same
    streams and measures as build_big, checked at E = 0 against the original run's stored beliefs"""
    build_big(eps, "_ep105", cache, dev, os.path.join(RUNS, sub), 0)


K100_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_k100.npz")
K100_U = [1, 2, 4, 8, 16, 32, 64] + list(range(100, 1001, 100))


def build_k100(us=K100_U, dev=None, cache=K100_CACHE):
    """John 11-18 ("running the normal lr update multiple times per batch ... 100 updates per episode and only 10
    episodes"): the pair at batch 8,000 and lr 1e-3 with 100 updates on each training episode's batch
    (fr_proxy_runs/run_multistep.py -> training_dynamics/k100_b8k/<net>_s20261029_ep13/ckpt_u<U>_net.pth);
    "E" in this cache = updates"""
    build_big(us, "_ep13", cache, dev, os.path.join(RUNS, "k100_b8k"), 0, "ckpt_u{:05d}_net.pth")


S100K_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_single100k.npz")


def build_single100k(us=(1, 2, 4, 8, 16, 32, 64, 100), dev=None, cache=S100K_CACHE):
    """John 11-18 ("Lets do 100 updates with the 100K batch"): ONE training batch of 100,000 (slices of 20,000), 100
    updates at lr 1e-3 (fr_proxy_runs/run_multistep.py ... single100k_k100 3 100000 20000); "E" in this cache = updates"""
    build_big(list(us), "_ep3", cache, dev, os.path.join(RUNS, "single100k_k100"), 0, "ckpt_u{:05d}_net.pth")


S20K_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_single20k.npz")
S20K_U = [1, 2, 4, 8, 16, 32, 64] + list(range(100, 2001, 100))


def build_single20k(us=S20K_U, dev=None, cache=S20K_CACHE):
    """John 11-18 ("can we take this 1 batch and scale the updates up to 2K with 20K batch size"): ONE training batch of
    20,000, 2,000 updates at lr 1e-3 (fr_proxy_runs/run_multistep.py ... single20k_k2000 3 20000 20000, MS_SAVE_EVERY=100);
    "E" in this cache = updates"""
    build_big(list(us), "_ep3", cache, dev, os.path.join(RUNS, "single20k_k2000"), 0, "ckpt_u{:05d}_net.pth")


S20K_BATCHES = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "single20k_batches.npz")
S20K_ACC = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_single20k_acc.npz")


def eval_single20k_acc(dev="cuda:1", sub="single20k_k2000", cache=None, fresh=True):
    """John 11-18 ("only update on 1 batch, but evaluate both"): goal accuracy at T of both networks on the single-batch
    run's training batch and on a second batch it never trained on (fr_proxy_runs/capture_batches.py recreated both: the
    same seed and setup, stopped before the first update), at every saved update; and the fresh-data run (one batch of
    8,000 per update) on the same second batch at its checkpoints to 2,000 episodes, for reference"""
    import torch
    from fig_fr_proxy import logits_gpu
    Bt = np.load(S20K_BATCHES)

    def acc(W, tag):
        o, z = Bt[f"{tag}_obs"], Bt[f"{tag}_Z"]
        g, v = Bt[f"{tag}_goal_ind"], Bt[f"{tag}_goal_value"]
        q = []
        with torch.no_grad():
            for i in range(0, len(o), 5000):                                         # 20,000 at once needs ~9 GB
                r = logits_gpu(W, torch.from_numpy(o[i:i + 5000].astype(np.float32)).to(dev),
                               torch.from_numpy(z[i:i + 5000]).to(dev))
                q.append(r.double().sum(1).reshape(len(r), 2, 10).cpu().numpy())   # summed logits = belief at T
        q = np.concatenate(q)
        return float((q[np.arange(len(o)), g].argmax(-1) == v).mean())

    def weights(path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        return {k: v.detach().float().to(dev) for k, v in d["weights"].items()}

    out = {"u": np.array([0] + S20K_U)}
    for net in ("trained", "echo"):
        rows = []
        for u in out["u"]:
            W = weights(os.path.join(RUNS, sub, f"{net}_s20261029_ep3", f"ckpt_u{u:05d}_net.pth"))
            rows.append((acc(W, "train"), acc(W, "second")))
        out[f"{net}_train"], out[f"{net}_second"] = np.array(rows).T
        if fresh:
            E = np.array(sorted({e for e in list(range(25, 501, 25)) + list(range(600, 2001, 100))}))
            out["E_fresh"] = E
            out[f"{net}_fresh_second"] = np.array([acc(weights(os.path.join(RUNS, f"{net}_s20261029",
                                                                            f"ckpt_ep{e:05d}_net.pth")), "second")
                                                   for e in E])
        print(net, "single batch, u / train / second:", [(int(u), round(a, 3), round(b, 3)) for u, a, b in
              zip(out["u"], out[f"{net}_train"], out[f"{net}_second"]) if u in (0, 100, 500, 1000, 1500, 2000)], flush=True)
    for k in ("joint_acc_T", "naive_acc_T"):
        out[f"train_{k}"], out[f"second_{k}"] = Bt[f"train_{k}"], Bt[f"second_{k}"]
    np.savez(cache or S20K_ACC, **out)


SHUF_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_single20k_shuffle.npz")
SHUF_ACC = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_single20k_shuffle_acc.npz")


def build_shuffle(us=S20K_U, dev=None, cache=SHUF_CACHE):
    """John 11-18 ("What if we shuffle the channels?"): the one 20,000 batch, 2,000 updates, each trajectory's channel
    order redrawn before every update (fr_proxy_runs/run_shuffle.py -> single20k_k2000_shuffle/); "E" = updates"""
    build_big(list(us), "_ep3", cache, dev, os.path.join(RUNS, "single20k_k2000_shuffle"), 0, "ckpt_u{:05d}_net.pth")


D500_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_dense500.npz")
_have = set(DENSE_EPS) | set(range(25, 501, 25))                     # checkpoints the other caches already hold
D500_EPS = sorted((set(range(1, 11)) | set(range(2, 501, 2))) - _have)   # John 11-18: every 2 episodes to 500


def build_dense500(eps=D500_EPS, dev=None, cache=D500_CACHE):
    """John 11-18 ("checkpointing every episode up to 10, every 2 up to 100, then every 5 up to 500", then "every 2
    episodes to 500"): the exact re-run
    training_dynamics/dense500/<net>_s20261029_ep505 (fr_proxy_runs/rerun_dense500.py), replayed at the checkpoints the
    other caches do not already hold (their weights are bit-identical to the original run's)"""
    build_big(list(eps), "_ep505", cache, dev, os.path.join(RUNS, "dense500"), 25)


PERSTEP_CACHE = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_training_single20k_perstep.npz")
PERSTEP_ACC = os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", "fr_proxy_single20k_perstep_acc.npz")


def build_perstep(us=S20K_U, dev=None, cache=PERSTEP_CACHE):
    """John 11-18 ("update the weights on each inference step, instead of the BPTT"): the one 20,000 batch, an update at
    every step from a detached state (fr_proxy_runs/run_perstep.py -> single20k_perstep/); "E" = updates"""
    build_big(list(us), "_ep3", cache, dev, os.path.join(RUNS, "single20k_perstep"), 0, "ckpt_u{:05d}_net.pth")


REP_EPS = list(range(25, 501, 25)) + list(range(600, 2001, 100))
rep_cache = lambda r: os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", f"fr_proxy_training_rep{r}.npz")


def build_replicate(r, eps=REP_EPS, dev=None):
    """John 11-18 ("please start the replicates on the cluster"): replicate r of the pair (fr_proxy_runs/run_replicate.py: the
    same world, new initial weights and episode stream, checkpoints every 25 to 500 and every 100 to 2,000), copied back
    to training_dynamics/replicate<r>/, replayed on the same 50 environments x 5,000 trajectories (no start check: its
    initial weights differ from the original run's by design; its world K, Q equal the original's, checked 11-18)"""
    build_big(list(eps), "_ep2005", rep_cache(r), dev, os.path.join(RUNS, f"replicate{r}"), -1)


rep_ext_cache = lambda r: os.path.join(ROOT, "main", "DATA", "experimental", "fr_proxy", f"fr_proxy_training_rep{r}_ext.npz")
REP_EXT_EPS = list(range(2100, 3001, 100))


def build_replicate_ext(r, eps=REP_EXT_EPS, dev=None):
    """John 11-18 ("train the 9 replicates until episode 3K"): replicate r continued from its weights and Adam state at
    2,005 episodes to 3,005 (fr_proxy_runs/run_extend.py; new episode stream; checkpoints renamed to their true episodes, in
    replicate<r>/<net>_s20261029_ep1005/), replayed like build_replicate"""
    build_big(list(eps), "_ep1005", rep_ext_cache(r), dev, os.path.join(RUNS, f"replicate{r}"), -1)


def updates(E):
    """optimizer steps behind the checkpoint saved at episode E (every 5th episode, 0 included, is a test episode)"""
    E = np.asarray(E)
    return E - E // 5


def build_dense(dev=None):
    """John 11-18 ("are we able to show more points between 0 and 100 to see this early correlation growing and
    decaying?"): the pair's first 105 episodes re-run from the same seed with a checkpoint at every test episode
    (fr_proxy_runs/rerun_dense.py = train_training_dynamics.main with schedule(dense_until=100) -> <net>_s20261029_ep105;
    its checkpoints at 25-100 equal the original run's, checked in main's printout), then build_big on them"""
    build_big(DENSE_EPS, "_ep105", DENSE_CACHE, dev)


def checkpoints():
    """episodes with a stored harvest for both networks"""
    have = [{int(f[3:8]) for f in os.listdir(os.path.join(AN, run)) if f.startswith("ck_")} for run in ("trained", "echo")]
    return sorted(have[0] & have[1])


def compute():
    """per checkpoint: both networks' goal accuracy at T and the three rank correlations -> cache (1.5 min)"""
    P = np.load(os.path.join(AN, "probes.npz"))
    fr, env, goal = P["fr"], P["env"], P["goal"]
    lj, ln = log_norm(P["log_jb"]), log_norm(P["log_nb"])
    n = len(env)
    assert np.allclose(kl(lj, ln), fr, rtol=1e-3, atol=1e-5), "probes: FR is not KL(Joint || Naive) in this order"
    out = {k: [] for k in ("E", "acc_trained", "acc_esn", "proxy", "trained_naive", "joint_esn")}
    for E in checkpoints():
        if E == 0:                                                         # the two networks are identical at E = 0
            continue
        Zs = [np.load(os.path.join(AN, run, f"ck_{E:05d}.npz")) for run in ("trained", "echo")]
        lT, lE = (log_norm(z["logq"]) for z in Zs)
        out["E"].append(E)
        for z, k in zip(Zs, ("acc_trained", "acc_esn")):
            out[k].append(z["right"][np.arange(n), -1, goal].mean())
        for k, v in (("proxy", kl(lT, lE)), ("trained_naive", kl(lT, ln)), ("joint_esn", kl(lj, lE))):
            out[k].append(np.nanmean(r_within(v, fr, env)))
    out = {k: np.array(v) for k, v in out.items()}
    out.update(acc_joint=P["right_bayes"][np.arange(n), goal].mean(), acc_naive=P["right_naive"][np.arange(n), goal].mean())
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **out)


def main():
    if "--rebuild" in sys.argv or not os.path.exists(CACHE):
        compute()
    D = dict(np.load(CACHE))
    E = D["E"]
    passes = E[np.argmax(D["acc_trained"] >= D["acc_naive"])]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_fig_style()
    plt.rcParams.update({"axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})
    A = 11
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.4), sharey="row", constrained_layout=True,
                           gridspec_kw=dict(width_ratios=(1, 1.25)))
    for j, (m, xs) in enumerate(((E <= 1000, "linear"), (E > 0, "log"))):
        a, b = ax[0, j], ax[1, j]
        for k, c, lab in (("acc_trained", C_TR, "Trained RNN"), ("acc_esn", C_ESN, "ESN")):
            a.plot(E[m], D[k][m], color=c, lw=2.2, marker="o" if j == 0 else None, ms=3.5, label=lab)
        a.axhline(D["acc_joint"], color="k", ls="--", lw=1); a.axhline(D["acc_naive"], color="k", ls=":", lw=1)
        for k, c, lab in (("trained_naive", C_TN, "Trained vs exact Naive"), ("proxy", C_PROXY, "Trained vs ESN"),
                          ("joint_esn", C_JE, "exact Joint vs ESN")):
            b.plot(E[m], D[k][m], color=c, lw=2.2, marker="o" if j == 0 else None, ms=3.5, label=lab)
        b.axhline(0, color=".6", lw=.8)
        b.set_xlabel("training episodes")
        for x_ in (a, b):
            x_.set_xscale(xs)
            x_.spines[["top", "right"]].set_visible(False)
            if j == 1:
                x_.axvspan(E[E > 0].min() * .9, 1000, color=".93", zorder=0, lw=0)
                x_.axvline(passes, color=C_TR, lw=1, ls=(0, (2, 2)))
        if j == 1:
            a.text(passes * 1.08, .05, f"Trained passes\nNaive ({passes:,})", color=C_TR, fontsize=A - 1)
            a.text(1000 * .93, .05, "first\n1,000", fontsize=A - 1, ha="right", color=".4")
            a.set_xlim(E[E > 0].min() * .9, E.max() * 1.1); b.set_xlim(E[E > 0].min() * .9, E.max() * 1.1)
        else:
            a.set_xlim(0, 1010); b.set_xlim(0, 1010)
    ax[0, 0].text(15, D["acc_joint"] + .012, f"Joint Bayes {D['acc_joint']:.2f}", fontsize=A)
    ax[0, 0].text(15, D["acc_naive"] + .012, f"Naive Bayes {D['acc_naive']:.2f}", fontsize=A)
    ax[0, 0].set(ylabel="goal accuracy at T", ylim=(0, .85))
    ax[1, 0].set(ylabel="rank correlation with FR\n(Spearman r)", ylim=(-.15, 1))
    ax[0, 0].set_title("First 1,000 episodes (every 25, then every 100)", fontsize=12.5)
    ax[0, 1].set_title("All of training (log scale)", fontsize=12.5)
    ax[0, 1].legend(frameon=False, fontsize=A, loc="upper left", bbox_to_anchor=(0, .86))
    ax[1, 1].legend(frameon=False, fontsize=A, loc="upper left")
    fig.suptitle(r"Divergence between the networks vs Factorization Regret: $D_{KL}(B^{\mathrm{Trained}}\,\|\,"
                 r"B^{\mathrm{ESN}})$, both at the same episode" + "\n50 environments x 200 trajectories x 30 steps; "
                 "r within an environment and step, then averaged", fontsize=12.5)
    out = fig_path("fig_fr_proxy_training.png")
    fig.savefig(out, dpi=150)
    pick = lambda e: int(np.argmin(np.abs(E - e)))
    print(out, f"joint {D['acc_joint']:.3f} naive {D['acc_naive']:.3f}; Trained passes Naive at {passes}")
    for e in (25, 100, 250, 500, 1000, 2000, 5000, 10000, 20000, 49995):
        i = pick(e)
        print(f"   E {E[i]:6d}: acc {D['acc_trained'][i]:.3f} / {D['acc_esn'][i]:.3f}; r proxy {D['proxy'][i]:.3f}, "
              f"Trained-Naive {D['trained_naive'][i]:.3f}, Joint-ESN {D['joint_esn'][i]:.3f}")


if __name__ == "__main__":
    main()
