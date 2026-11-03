"""Training-dynamics analyses (John 10-31: "lets get the training analyses"): the properties he asked to follow through
training (10-29: norm, PR, FR, entanglement, dW, error interference, variance explained by inference time, marginal
and joint beliefs, Z, the latent variables and their joint, and the dimensionality each factor induces in the input and
in the recursion), the theory session's offline items, and the questions in memory learning-dynamics-phases. Every
checkpoint of the training-dynamics runs (trainings/train_training_dynamics.py; seed 20261029: one world, K and Q
identical in every run; checked per run against probes.npz) is replayed in fp32 on ONE fixed probe set from that world,
and everything is stored as primitives (per-particle arrays and sufficient statistics), so later questions need no new
pass. Verified 11-01 by four independent CPU verifiers (workflow wepci9b8c; their fixes are in this version).

Probe set (probes() -> analysis/probes.npz, once):
  50 environments (distinct pairs of test variables 0-49, never a training goal and never seen as a pair; the runs'
  own test episodes also pair a variable with itself, ~2%) x all 100 cells (r0, r1) x 2 streams (stream s: the goal is
  position s) = 10,000 episodes of 30 steps, environment-major with the two streams of a cell adjacent. o_ti ~
  Bernoulli(L_i(r0, r1)) with the env's own tables (default_likelihoods); the network's context input Z from model_K /
  model_Q (float32, as active_Z). Bayes side computed in float64 from (obs, L) and stored in float32: the joint log
  posterior over the 100 cells (r0-major) "logpost", its marginals "jb" and their logs "log_jb" (use the logs for
  anything sensitive to the tails: jb underflows to 0 in float32 from step ~20), the naive marginals "nb" / "log_nb"
  and per-step likelihoods "npx", the per-step increments of naive (N), step-wise (SW) and Bayes (D), centred over
  values (an1 / a1_increments), entanglement per variable "ent" (float64) = step_dkl(npx, approx_lik(jb)) (the locked
  definition; rev = its mean over the two variables), FR per variable "fr" = fr_sym(jb, nb) (John 10-28: average both),
  and correctness at T of Bayes, naive, windowed Bayes w = 1 (= step-wise), 2, 5 and the factorized-state observer
  (ADF). A separate basis set (N_BASIS random test episodes, BASIS_SEED) gives each checkpoint its own PCA basis of u
  (the locked rule: bands are fixed subspaces fitted on other episodes; 2,000 episodes moved the cross-band share by
  ~6%, so 20,000).

Per checkpoint (forward(); a hand-written fp32 LSTM, gate order i, f, g, o, = nn.LSTM to 1e-6; all 10,000 probes):
  per particle: the network's marginal log beliefs "logq" (n, T, 2, 10, float16; argmax ties are possible, so
    correctness is stored separately as "right" (n, T, 2)); joint read-outs also the cell log q at t = 1 and T;
    |u_t|, |c_t|, the turns 1 - cos(u_t, u_{t-1}) and 1 - cos(c_t, c_{t-1}) (float64; t = 0 against the start state),
    u's effective dimension (projection PR on the checkpoint's basis) and its energy share per 10% band "band_u"
    (n, T, 10; NaN where a band is empty, as early in training when one PC holds most of the variance; "band_ids");
  per step "fac" (T, 13 rows, 8 factors, [variance, PR]); rows "fac_rows": x (the read-in output = the LSTM input);
    W_ih x whole and per gate block i, f, g, o; u (= h); W_hh u_{t-1} whole and per gate block (the recurrent drive
    step t receives; at step 1 it is the start state's, constant: variance 0, PR NaN); c. Factors "factors": total,
    environment (50 pairs = 50 Z), r0, r1, joint (the r0 x r1 interaction = cell - r0 - r1), cell (r0, r1), envcell
    (environment x cell interaction), res (within an (environment, cell): the two streams). The design is balanced and
    crossed, so the variances add exactly: total = env + cell + envcell + res, cell = r0 + r1 + joint; the PR of a
    factor's covariance is the dimension it induces;
  "pool_x" / "pool_u" / "pool_c" (rows as fac's, [total variance, total PR, step share, step PR]): all steps pooled
    (x, u, c: steps 1-30; the W_hh rows of pool_u: the drives of steps 2-30, from u_1..u_29) - the step factor is the
    variance across steps of the probe-AVERAGED trajectory (the time course shared by all probes; each episode's own
    movement cancels, so x scores 0) and the total PR is one cloud pooled over steps (the drift of the mean counts as
    dimensions). Audit 11-01: for movement over steps within episodes use claude_plots/td_time_within.py, for the
    dimension the per-step PR (fac);
  forget-gate time constants "tau" = 1 / (1 - mean f) per unit over steps 2-30 (an2), "gate_mean" (4, H), input /
    recurrent drive RMS per gate block "drive_rms" (2, 4);
  memory truncation (an6): accuracy at T with the memory cut to w = 1, 2, 5 steps (the cell rerun from its start state
    over the last w inputs at every step t >= w, the own read-out summed) and intact: "trunc_acc" (4, [goal, both
    variables]) and per particle "trunc_right" (n, 4, 2);
  the per-step logit increment regressed (no intercept) on N, W = SW - N, M = D - SW: normal-equation sums per step
    "reg_xtx" (T, 3, 3), "reg_xtd" (T, 3), "reg_dtd" (T,) (both variables, all values), so any step range or single-rule
    fit (SW = N + W, D = N + W + M) is exact later;
  ridge decoding (a side statistic the theory session asked for; memory raw-data-not-decoders) from u_t and from c_t
    (float32 solve: R^2 within ~1e-3 of float64), fitted on 25 environments (penalty chosen on 5 of them) and tested on
    the other 25 (unseen pairs; most share a variable with fitting pairs), R^2 pooled over target dimensions,
    "dec_r2" (DEC_STEPS, [u, c], [counts n_t (inclusive), Bayes's centred log joint posterior floored 30 nats below its
    max, both marginal log posteriors floored the same way]);
  error: "acc" (T,) and "acc_var" (T, 2); for the goal variable per step the exact "kl_qp" = KL(network || Bayes) and
    "kl_pq" = KL(Bayes || network) (theory_v9's XL), "kl_qn" = KL(network || naive), and the clipped versions the
    training recorder logs (env.DKL, 1e-15: joint_DKL, XL) "kl_qp_clip", "kl_pq_clip";
  read-out: the fixed logits "F1" / "FT" (mean cumulative logits at t = 1 and T), the SD of log q per value, values /
    cells more than 10 nats below the best of their row and column (theory_v9's exclusion rule; marginal outputs: per
    variable) "excl_1" / "excl_T"; up to episode 3,000 the gradient norm of each read-out row under the run's own loss
    "ro_gnorm" (the theory's starvation question);
  weights (float64): norms, distance to the initial and to the previous checkpoint, per gate block of W_hh its spectral
    radius and norm "rho_hh" (4, 2), singular values of W_hh and W_ih minus their initial values and of the read-out.
Update half (credit(); runs on the standard SANITY loss with a single-Linear read-out: trained, echo, trained_N*):
  gradflow_analysis's locked convention on the first UPDATE_N probes of a fixed permutation (batches of 1,000 for the
  loss's batch minimum): both variables' step-t errors at steps 17-30 ("cr_steps"), sent back 16 lags
  (fig1c_scan_credit.measures): credit coherence "cr_coh", |dW|^2 "cr_E" (each episode's unscaled step-t term: the
  loss is summed over values, the training loss averages over B T R), dW's effective dimension "cr_dim" and the 10%
  cross-band share "cr_bandx" on the checkpoint's own basis; control "cr_bandx_fixed": the same on the run's FINAL
  checkpoint's bands (one partition for every checkpoint), on the first FIXED_BAND_N of those probes. Rare (probe,
  step) values of cr_E / cr_bandx carry ~1e-3 relative fp32 noise (the sqrt head near its kink).
Raw snapshots (SNAP episodes; trained, echo): u and c of the first N_SNAP permuted probes, fp16.
Output: main/DATA/experimental/training_dynamics/analysis/<run>/ck_<EEEEE>.npz (one per checkpoint; reruns skip existing
files), snap_<EEEEE>.npz, basis_final.npz. Episodes are raw (a test episode trains nothing: checkpoint E holds
E - E // 5 updates).
Run from the repository root (GPU 0 only):
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/td_analysis.py probes
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/td_analysis.py run trained echo
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/td_analysis.py check trained \\
        49995"""
import os
import sys
import io
import json
import glob
import time
import contextlib
import numpy as np
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION")]
import gradflow_analysis as ga                                                    # noqa: E402  (turns TF32 off)
import supplementary_plotter2 as sp                                               # noqa: E402
from main.CognitiveGridworld import CognitiveGridworld                            # noqa: E402

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
OUT = os.environ.get("TD_OUT", os.path.join(TD, "analysis"))                    # TD_OUT: pilots elsewhere
WORLD = "trained"                                  # the world (K, Q) of every seed-20261029 run
SEED = 20261029
PROBE_SEED, BASIS_SEED, PERM_SEED = 20261101, 20261102, 20261103
N_ENV, N_STREAM, N_BASIS, BASIS_CHUNK = 50, 2, 20000, 5000
T, R, C, NO = 30, 10, 2, 5
N_TEST_VARS = 50
UPDATE_N, UPDATE_BATCH, BAND_CHUNK, FIXED_BAND_N = 5000, 1000, 50, 2000
N_SNAP = 1000
SNAP = (0, 100, 250, 500, 1000, 2000, 3000, 5000, 10000, 20000, 35000, 49995)
STARVE_UNTIL = 3000
DEC_STEPS = (0, 1, 2, 4, 7, 11, 15, 19, 24, 29)                                    # 0-based: steps 1, 2, 3, 5, ... 30
RIDGE_C = (1e-4, 1e-3, 1e-2, 1e-1, 1.)                                             # penalty = c * tr(G) / D
FLOOR = 30.                                                   # decoding targets: floored this many nats below the max
TRUNC_W = (1, 2, 5)
N_BANDS = 10
FACTORS = ("total", "env", "r0", "r1", "joint", "cell", "envcell", "res")
FAC_ROWS = ("x", "Wih_x", "Wih_x_i", "Wih_x_f", "Wih_x_g", "Wih_x_o", "u", "Whh_u", "Whh_u_i", "Whh_u_f", "Whh_u_g",
            "Whh_u_o", "c")
RUNS = {                                                     # run -> loss kind; credit only for the standard loss
    "trained": "std", "echo": "std", "trained_N50": "std", "trained_N100": "std", "trained_N200": "std",
    "mlp55": "std", "joint_trained": "joint_goal", "joint_echo": "joint_goal", "joint_echo_jl": "joint_full",
    "trained_p1": "p1", "trained_lastT_p1": "lastT_p1", "trained_lastT": "lastT"}
CREDIT = ("trained", "echo", "trained_N50", "trained_N100", "trained_N200")


# ═══════════════════════════════════════════════════════════════════
# Probe set and Bayes side (once)
# ═══════════════════════════════════════════════════════════════════

def ckpt_path(run, ep, seed=SEED):
    return os.path.join(TD, f"{run}_s{seed}", f"ckpt_ep{ep:05d}_net.pth")


def checkpoints(run, seed=SEED):
    files = glob.glob(os.path.join(TD, f"{run}_s{seed}", "ckpt_ep?????_net.pth"))
    return sorted(int(os.path.basename(p)[7:12]) for p in files)


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def world_tables(ck, pairs):
    """the env's own likelihood tables L (n, 5, R, R) (float64, raw order [i, r0, r1]) and the network's context input Z
    (n, 5, 2) (float32, as active_Z) for sorted variable pairs (n, 2)"""
    st = np.random.get_state()
    with _quiet():
        env = CognitiveGridworld(mode=None, show_plots=False, episodes=1, batch_num=len(pairs), step_num=T,
                                 state_num=500, realization_num=R, obs_num=NO, ctx_num=C, hid_dim=1000,
                                 learn_embeddings=False)
    np.random.set_state(st)
    env.all_K, env.all_Q = ck["K"], ck["Q"]
    env.ctx_inds = np.asarray(pairs)
    env.goal_ind = np.zeros(len(pairs), dtype=int)
    env.default_likelihoods()
    L = np.asarray(env.joint_likelihood, dtype=np.float64)
    mK, mQ = torch.as_tensor(ck["model_K"]).float(), torch.as_tensor(ck["model_Q"]).float()
    s0, s1 = torch.as_tensor(pairs[:, 0]), torch.as_tensor(pairs[:, 1])
    Z = torch.stack([(mK[s0] * mQ[s1]).sum(-1), (mK[s1] * mQ[s0]).sum(-1)], -1).numpy()
    return L, Z


def _lse(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def _cen(a):
    return a - a.mean(-1, keepdims=True)


def fr_sym(p, q, eps=1e-15):
    p, q = np.clip(p, eps, 1 - eps), np.clip(q, eps, 1 - eps)
    return .5 * ((p * np.log(p / q)).sum(-1) + (q * np.log(q / p)).sum(-1))


def bayes_side(obs, L, r):
    """the Bayes-side primitives for observations obs (n, T, 5), per-episode tables L (n, 5, R, R) and true values r
    (n, 2)"""
    n = len(obs)
    o = obs.astype(np.float64)
    ll = np.einsum("nti,nijk->ntjk", o, np.log(L)) + np.einsum("nti,nijk->ntjk", 1 - o, np.log1p(-L))  # per-step
    lam = np.cumsum(ll, 1)
    logpost = lam - _lse(lam.reshape(n, T, -1), -1)[..., None, None]
    post = np.exp(logpost)
    jb = np.stack([post.sum(3), post.sum(2)], 2)                                    # (n, T, 2, R) raw order
    log_jb = np.stack([_lse(logpost, 3), _lse(logpost, 2)], 2)
    Lb = np.stack([L.mean(3), L.mean(2)], 2)                                         # naive tables (n, 5, 2, R)
    lpx = np.einsum("nti,nikr->ntkr", o, np.log(Lb)) + np.einsum("nti,nikr->ntkr", 1 - o, np.log1p(-Lb))
    cl = np.cumsum(lpx, 1)
    log_nb = cl - _lse(cl, -1)[..., None]
    nb = np.exp(log_nb)
    npx = np.exp(lpx - _lse(lpx, -1)[..., None])
    marg_lam = np.stack([_lse(lam, 3), _lse(lam, 2)], 2)                            # (n, T, 2, R)
    prev = np.concatenate([np.full_like(marg_lam[:, :1], np.log(R)), marg_lam[:, :-1]], 1)
    inc = {"N": _cen(lpx), "SW": _cen(np.stack([_lse(ll, 3), _lse(ll, 2)], 2)), "D": _cen(marg_lam - prev)}
    ent = sp.step_dkl(npx, sp.approx_lik(jb))                                        # (n, T, 2)
    fr = np.stack([fr_sym(jb[:, :, v], nb[:, :, v]) for v in range(2)], -1)          # (n, T, 2)
    right = {"bayes": jb[:, -1].argmax(-1) == r, "naive": nb[:, -1].argmax(-1) == r}
    Lam = np.concatenate([np.zeros_like(lam[:, :1]), lam], 1)                       # Lam[:, t] = sum of steps 1..t
    for w in TRUNC_W:                                                                # windowed Bayes (an3)
        acc = np.zeros((n, 2, R))
        for t in range(1, T + 1):
            a = max(0, t - w)
            cur = np.stack([_lse(Lam[:, t] - Lam[:, a], 2), _lse(Lam[:, t] - Lam[:, a], 1)], 1)
            pre = (np.stack([_lse(Lam[:, t - 1] - Lam[:, a], 2), _lse(Lam[:, t - 1] - Lam[:, a], 1)], 1)
                   if t - 1 > a else np.log(R))
            acc += cur - pre
        right[f"window{w}"] = acc.argmax(-1) == r
    b = np.full((n, 2, R), 1. / R)                                                   # factorized-state Bayes (ADF)
    px = np.exp(ll)
    for t in range(T):
        j = b[:, 0, :, None] * b[:, 1, None, :] * px[:, t]
        j /= j.sum((1, 2), keepdims=True)
        b = np.stack([j.sum(2), j.sum(1)], 1)
    right["adf"] = b.argmax(-1) == r
    f32 = dict(logpost=logpost.reshape(n, T, R * R), jb=jb, log_jb=log_jb, nb=nb, log_nb=log_nb, npx=npx,
               **{f"inc_{k}": v for k, v in inc.items()})
    return dict(**{k: v.astype(np.float32) for k, v in f32.items()}, ent=ent, fr=fr,
                **{f"right_{k}": v for k, v in right.items()})


def probes():
    path = os.path.join(TD, "analysis", "probes.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ck = torch.load(ckpt_path(WORLD, 0), map_location="cpu", weights_only=False)
    pairs_all = np.array([(a, b) for a in range(N_TEST_VARS) for b in range(a + 1, N_TEST_VARS)])
    L_all, Z_all = world_tables(ck, pairs_all)
    rng = np.random.default_rng(PROBE_SEED)
    pick = np.sort(rng.choice(len(pairs_all), N_ENV, replace=False))
    env = np.repeat(np.arange(N_ENV), R * R * N_STREAM)
    cell = np.tile(np.repeat(np.arange(R * R), N_STREAM), N_ENV)
    r = np.stack([cell // R, cell % R], 1)
    goal = np.tile(np.arange(N_STREAM), N_ENV * R * R)
    L = L_all[pick][env]
    obs = rng.random((len(env), T, NO)) < L[np.arange(len(env)), :, r[:, 0], r[:, 1]][:, None, :]
    split = rng.permutation(N_ENV)                                                   # decoding: train / val / test
    P = dict(pairs=pairs_all[pick], L=L_all[pick], Z=Z_all[pick], env=env, cell=cell, r=r, goal=goal, obs=obs,
             dec_train=np.sort(split[:20]), dec_val=np.sort(split[20:25]), dec_test=np.sort(split[25:]),
             world_K=np.asarray(ck["K"]), world_Q=np.asarray(ck["Q"]))
    t0 = time.time()
    P.update(bayes_side(obs, L, r))
    rb = np.random.default_rng(BASIS_SEED)
    bpick = rb.integers(len(pairs_all), size=N_BASIS)
    br = rb.integers(R, size=(N_BASIS, 2))
    P["basis_Z"] = Z_all[bpick]
    P["basis_obs"] = rb.random((N_BASIS, T, NO)) < L_all[bpick, :, br[:, 0], br[:, 1]][:, None, :]
    P["perm"] = np.random.default_rng(PERM_SEED).permutation(len(env))
    np.savez(path, **P)
    for k in ("bayes", "naive", "window1", "window2", "window5", "adf"):
        print(f"  {k:8s} accuracy at T, goal {P['right_' + k][np.arange(len(goal)), goal].mean():.3f}, "
              f"both variables {P['right_' + k].mean():.3f}")
    print(f"probes saved: {path} ({os.path.getsize(path) / 1e6:.0f} MB; Bayes side {time.time() - t0:.0f} s)")
    return P


# ═══════════════════════════════════════════════════════════════════
# Network: loading and the forward half
# ═══════════════════════════════════════════════════════════════════

def _cfg(run, seed=SEED):
    with open(os.path.join(TD, f"{run}_s{seed}", "run.json")) as f:
        return json.load(f)["cfg"]


def build(run, ep, dev, seed=SEED):
    cfg = _cfg(run, seed)
    arch = {k: cfg[k] for k in ("reservoir", "output_joint", "readin_depth", "readout_depth") if k in cfg}
    st = np.random.get_state()
    with _quiet():
        agent = CognitiveGridworld(mode="SANITY", cuda=dev.index or 0, episodes=0, training=False, batch_num=2,
                                   step_num=T, realization_num=R, obs_num=NO, state_num=500, ctx_num=C,
                                   hid_dim=cfg["hid_dim"], learn_embeddings=False, show_plots=False,
                                   load_env=f"experimental/training_dynamics/{run}_s{seed}/ckpt_ep{ep:05d}", **arch)
    np.random.set_state(st)
    m = agent.model.eval()
    m.joint_out = m.classifier_readout(torch.zeros(1, cfg["hid_dim"], device=dev)).shape[-1] == R * R
    return m


def load_weights(m, run, ep, seed=SEED):
    w = torch.load(ckpt_path(run, ep, seed), map_location="cpu", weights_only=False)["weights"]
    res = m.load_state_dict(w, strict=False)
    assert set(res.missing_keys) <= {"all_K", "all_Q"} and not res.unexpected_keys, res
    return w


def _readin(m, o, zin):
    return torch.relu(m.classifier_readin(torch.cat([o, zin], -1)))


def _cell(pre, c):
    i, f, g, o = pre.chunk(4, -1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    c = f * c + i * g
    return o * c.tanh(), c, f


def _start(m, n):
    H = m.STM.shape[-1]
    return (m.STM.detach().float().reshape(1, H).expand(n, H).contiguous(),
            m.LTM.detach().float().reshape(1, H).expand(n, H).contiguous())


def _last_linear(m):
    """the read-out's last Linear (module order) and its state-dict key prefix"""
    name, lin = [(k, x) for k, x in m.classifier_readout.named_modules() if isinstance(x, torch.nn.Linear)][-1]
    return lin, "classifier_readout" + (f".{name}" if name else "")


def marginals(cum, joint):
    """marginal log beliefs (n, [T,] 2, R) from cumulative logits"""
    if joint:
        lq = torch.log_softmax(cum, -1).reshape(*cum.shape[:-1], R, R)
        return torch.stack([lq.logsumexp(-1), lq.logsumexp(-2)], -2)
    return torch.log_softmax(cum.reshape(*cum.shape[:-1], C, R), -1)


@torch.no_grad()
def basis(m, P, dev):
    """the checkpoint's PCA basis of u on the basis set: mu, V (columns: PCs, descending), evr, the non-empty 10% bands'
    one-hot (H, bands) and their ids"""
    W_ih, W_hh, b = ga.lstm_weights(m)
    H = W_hh.shape[1]
    S1 = torch.zeros(H, dtype=torch.float64, device=dev)
    S2 = torch.zeros(H, H, dtype=torch.float64, device=dev)
    nb = len(P["basis_obs"])
    for c0 in range(0, nb, BASIS_CHUNK):
        obs = torch.as_tensor(P["basis_obs"][c0:c0 + BASIS_CHUNK], dtype=torch.float32, device=dev)
        zin = torch.as_tensor(P["basis_Z"][c0:c0 + BASIS_CHUNK], device=dev).reshape(len(obs), -1)
        h, c = _start(m, len(obs))
        for t in range(T):
            h, c, _ = _cell(_readin(m, obs[:, t], zin) @ W_ih.T + b + h @ W_hh.T, c)
            hd = h.double()
            S1 += hd.sum(0)
            S2 += hd.T @ hd
    n = nb * T
    mu = S1 / n
    ev, V = torch.linalg.eigh(S2 / n - torch.outer(mu, mu))
    ev, V = ev.flip(0).clamp_min(0), V.flip(1)
    evr = (ev / ev.sum()).cpu().numpy()
    sid = ga.cut_segments(evr, N_BANDS)
    ids = np.unique(sid)
    OH = torch.as_tensor((sid[:, None] == ids[None]).astype(np.float32), device=dev)
    return mu.float(), V.float(), evr, OH, ids


def factor_groups(P, dev):
    out = {}
    for k, v, K in (("env", P["env"], N_ENV), ("r0", P["r"][:, 0], R), ("r1", P["r"][:, 1], R),
                    ("cell", P["cell"], R * R)):
        g = torch.as_tensor(v, device=dev).long()
        out[k] = (g, K, torch.bincount(g, minlength=K).double())
    return out


def factor_covs(X, groups):
    """the factor covariances (D x D, float64) of the particles X (n, D) at one step (see the docstring)"""
    Xd = X.double()
    Xc = Xd - Xd.mean(0)
    n = len(X)
    out = {"total": Xc.T @ Xc / n}
    for k, (g, K, cnt) in groups.items():
        M = torch.zeros(K, X.shape[1], dtype=torch.float64, device=X.device).index_add_(0, g, Xc) / cnt[:, None]
        A = M * cnt.sqrt()[:, None]
        out[k] = A.T @ A / n
    Dd = Xc[0::2] - Xc[1::2]                                                         # the two streams of a cell
    out["res"] = Dd.T @ Dd / (2 * n)
    out["joint"] = out["cell"] - out["r0"] - out["r1"]
    out["envcell"] = out["total"] - out["res"] - out["env"] - out["cell"]
    return out


def trace_pr(covs, grams):
    """(len(grams), len(FACTORS), 2): the variance tr(C G) and PR tr(CG)^2 / tr(CGCG) of every factor covariance C seen
    through every Gram G = W^T W (None: the signal itself)"""
    out = np.full((len(grams), len(FACTORS), 2), np.nan)
    for a, G in enumerate(grams):
        for f, name in enumerate(FACTORS):
            Cm = covs[name].float()
            A = Cm if G is None else Cm @ G
            tr, tr2 = A.diagonal().sum().item(), (A * A.T).sum().item()
            out[a, f] = tr, (tr * tr / tr2 if tr2 > 0 else np.nan)
    return out


def _pool_stats(S1, S2, mus, grams):
    """[total variance, total PR, step share, step PR] per Gram, from the per-particle sums over the pooled steps
    S1 = sum x, S2 = sum x x^T and the per-step means mus (steps, D) (law of total covariance)"""
    Ts = len(mus)
    mu = mus.mean(0)
    Cpool = S2 / Ts - torch.outer(S1 / Ts, S1 / Ts)
    Cstep = (mus - mu).T @ (mus - mu) / len(mus)
    res = []
    for G in grams:
        A, B = (Cpool.float(), Cstep.float()) if G is None else (Cpool.float() @ G, Cstep.float() @ G)
        tA, tB = A.diagonal().sum().item(), B.diagonal().sum().item()
        res.append([tA, tA ** 2 / (A * A.T).sum().item(), tB / tA, tB ** 2 / (B * B.T).sum().item()])
    return np.array(res)


def ridge_r2(X, Ys, P, idx, dev):
    """R^2 on the test environments of ridge regressions (features standardized on the fitting set, intercept free),
    the penalty chosen on the validation environments; Ys: list of (n, k) targets -> (len(Ys),) R^2, (len(Ys),) c"""
    env = idx["env_np"]
    fit, val = np.isin(env, P["dec_train"]), np.isin(env, P["dec_val"])
    test = np.isin(env, P["dec_test"])

    def solve(rows):
        Xr = X[torch.as_tensor(rows, device=dev)].double()
        mx, sx = Xr.mean(0), Xr.std(0).clamp_min(1e-6)
        Xs = ((Xr - mx) / sx).float()
        ev, V = torch.linalg.eigh(Xs.T @ Xs)
        return mx, sx, ev.clamp_min(0), V, Xs

    def predict(sol, Y, rows, rows_out, c):
        mx, sx, ev, V, Xs = sol
        Yr = Y[torch.as_tensor(rows, device=dev)]
        my = Yr.mean(0)
        lam = c * ev.sum() / len(ev)
        Bm = V @ (((V.T @ (Xs.T @ (Yr - my))) / (ev + lam)[:, None]))
        Xo = ((X[torch.as_tensor(rows_out, device=dev)].double() - mx) / sx).float()
        return Xo @ Bm + my

    def r2(Yhat, Y):
        return 1 - ((Y - Yhat) ** 2).sum().item() / ((Y - Y.mean(0)) ** 2).sum().item()

    sol_fit, sol_all = solve(fit), solve(fit | val)
    out_r2, out_c = [], []
    for Y in Ys:
        Yv, Yt = Y[torch.as_tensor(val, device=dev)], Y[torch.as_tensor(test, device=dev)]
        best = max(RIDGE_C, key=lambda c: r2(predict(sol_fit, Y, fit, val, c), Yv))
        out_r2.append(r2(predict(sol_all, Y, fit | val, test, best), Yt))
        out_c.append(best)
    return np.array(out_r2), np.array(out_c)


def run_loss(cum, P_goal, P_cells, goal, kind, m, joint):
    """the run's own training loss on cumulative logits cum (n, T, O) (float64, requires grad)"""
    ar = torch.arange(len(cum), device=cum.device)
    if kind == "joint_full":
        lq = torch.log_softmax(cum, -1)
        return (P_cells.exp() * (P_cells - lq)).sum(-1).mean()
    lq = marginals(cum, joint)
    q = lq.exp()[ar, :, goal]                                                       # (n, T, R)
    P = P_goal
    if kind in ("lastT", "lastT_p1"):
        q, P = q[:, -1:], P[:, -1:]
    power = 1. if kind in ("p1", "lastT_p1") else .5
    D = m.DKL_sym(q, P, PM=False)
    return ((D - D.detach().min() + 1e-8) ** power).mean()


def _grams(W, H):
    """the signal itself (None), then W^T W for the whole drive and for each gate block i, f, g, o"""
    return [None, W.T @ W] + [W[k * H:(k + 1) * H].T @ W[k * H:(k + 1) * H] for k in range(4)]


def _turn(old, new):
    o, n_ = old.double(), new.double()
    return (1 - (o * n_).sum(-1) / (o.norm(dim=-1) * n_.norm(dim=-1)).clamp_min(1e-300)).cpu().numpy()


def forward(m, P, kind, ep, dev, bas, snap=False):
    """the forward half for one checkpoint (see the docstring) with the checkpoint's basis bas; returns a dict of arrays
    and the snapshots"""
    t0 = time.time()
    n = len(P["obs"])
    joint = m.joint_out
    obs = torch.as_tensor(P["obs"], dtype=torch.float32, device=dev)
    zin = torch.as_tensor(P["Z"][P["env"]], device=dev).reshape(n, -1)
    goal = torch.as_tensor(P["goal"], device=dev).long()
    ar = torch.arange(n, device=dev)
    W_ih, W_hh, b = ga.lstm_weights(m)
    H = W_hh.shape[1]
    mu_b, V_b, evr, OH, band_ids = bas
    G_ih, G_hh = (_grams(W, H) for W in (W_ih, W_hh))
    groups = factor_groups(P, dev)
    idx = {"env_np": P["env"]}
    counts = torch.as_tensor(np.cumsum(P["obs"], 1), dtype=torch.float32, device=dev)
    logpost, log_jb = P["logpost"], P["log_jb"]
    perm_snap = torch.as_tensor(P["perm"][:N_SNAP], device=dev)
    n_out = R * R if joint else C * R
    out = {k: np.zeros((n, T), np.float32) for k in ("norm_u", "norm_c", "effdim_u")}
    out.update(turn_u=np.zeros((n, T)), turn_c=np.zeros((n, T)))
    out["band_u"] = np.full((n, T, N_BANDS), np.nan, np.float16)
    fac = np.full((T, len(FAC_ROWS), len(FACTORS), 2), np.nan)
    fac[0, 7:12, :, 0] = 0.                                                          # step 1's drive: the start state
    pool = {s: [torch.zeros(H, dtype=torch.float64, device=dev), torch.zeros(H, H, dtype=torch.float64, device=dev),
                []] for s in ("x", "u", "c", "u_prev")}
    lg = torch.zeros(n, T, n_out, device=dev)
    fsum = torch.zeros(H, dtype=torch.float64, device=dev)
    gate_sum = torch.zeros(4, H, dtype=torch.float64, device=dev)
    drive = torch.zeros(2, 4, dtype=torch.float64, device=dev)
    dec_r2, dec_c = np.full((len(DEC_STEPS), 2, 3), np.nan), np.full((len(DEC_STEPS), 2, 3), np.nan)
    snaps = {"u": torch.zeros(N_SNAP, T, H, dtype=torch.float16, device=dev),
             "c": torch.zeros(N_SNAP, T, H, dtype=torch.float16, device=dev)} if snap else None
    h, c = _start(m, n)
    prev_cov_u = None
    with torch.no_grad():
        for t in range(T):
            x = _readin(m, obs[:, t], zin)
            inp, rec = x @ W_ih.T, h @ W_hh.T
            pre = inp + b + rec
            h_new, c_new, f = _cell(pre, c)
            lg[:, t] = m.classifier_readout(h_new)
            out["turn_u"][:, t], out["turn_c"][:, t] = _turn(h, h_new), _turn(c, c_new)
            out["norm_u"][:, t] = h_new.norm(dim=-1).cpu().numpy()
            out["norm_c"][:, t] = c_new.norm(dim=-1).cpu().numpy()
            p2 = ((h_new - mu_b) @ V_b).double() ** 2
            out["effdim_u"][:, t] = (p2.sum(-1) ** 2 / (p2 ** 2).sum(-1)).cpu().numpy()
            E = p2.float() @ OH
            out["band_u"][:, t, band_ids] = (E / E.sum(-1, keepdim=True)).cpu().numpy()
            if t >= 1:
                fsum += f.double().sum(0)
            gates = pre.chunk(4, -1)
            for k, gv in enumerate((gates[0].sigmoid(), f, gates[2].tanh(), gates[3].sigmoid())):
                gate_sum[k] += gv.sum(0).double()
            drive += torch.stack([(a ** 2).reshape(n, 4, H).sum((0, 2)) for a in (inp, rec)]).double()
            cx, cu, cc = (factor_covs(X, groups) for X in (x, h_new, c_new))
            fac[t, :6] = trace_pr(cx, G_ih)
            fac[t, 6] = trace_pr(cu, [None])[0]
            if prev_cov_u is not None:
                fac[t, 7:12] = trace_pr(prev_cov_u, G_hh[1:])
            fac[t, 12] = trace_pr(cc, [None])[0]
            prev_cov_u = cu
            for s, X in (("x", x), ("u", h_new), ("c", c_new)) + ((("u_prev", h),) if t >= 1 else ()):
                Xd = X.double()
                pool[s][0] += Xd.sum(0)
                pool[s][1] += Xd.T @ Xd
                pool[s][2].append(Xd.mean(0))
            if t in DEC_STEPS:
                lp = torch.as_tensor(logpost[:, t], device=dev)
                clr = torch.maximum(lp, lp.max(-1, keepdim=True).values - FLOOR)
                lm = torch.as_tensor(log_jb[:, t], device=dev)
                lm = torch.maximum(lm, lm.max(-1, keepdim=True).values - FLOOR)
                Ys = [counts[:, t], clr - clr.mean(-1, keepdim=True), (lm - lm.mean(-1, keepdim=True)).reshape(n, -1)]
                j = DEC_STEPS.index(t)
                for a, X in enumerate((h_new, c_new)):
                    dec_r2[j, a], dec_c[j, a] = ridge_r2(X, Ys, P, idx, dev)
            if snap:
                snaps["u"][:, t], snaps["c"][:, t] = h_new[perm_snap].half(), c_new[perm_snap].half()
            h, c = h_new, c_new
        del cx, cu, cc, prev_cov_u, inp, rec, pre, x, p2, E
        torch.cuda.empty_cache()
        poolr = {s: _pool_stats(pool[s][0] / n, pool[s][1] / n, torch.stack(pool[s][2]), grams)   # sums per particle
                 for s, grams in (("x", G_ih), ("u", [None]), ("c", [None]), ("u_prev", G_hh[1:]))}
        # beliefs, accuracy, error (exact KLs from log posteriors; the recorder's clipped ones under explicit names)
        cum = lg.double().cumsum(1)
        lq = marginals(cum, joint)                                                   # (n, T, 2, R)
        rt = torch.as_tensor(P["r"], device=dev)
        right = lq.argmax(-1) == rt[:, None, :]                                      # (n, T, 2)
        lqg = lq[ar, :, goal]
        lPg = torch.as_tensor(log_jb, device=dev).double()[ar, :, goal]
        lNg = torch.as_tensor(P["log_nb"], device=dev).double()[ar, :, goal]
        qc = lqg.exp().clamp(1e-15, 1 - 1e-15)
        Pc = lPg.exp().clamp(1e-15, 1 - 1e-15)
        out.update(logq=lq.cpu().numpy().astype(np.float16), right=right.cpu().numpy(),
                   acc=right[ar, :, goal].double().mean(0).cpu().numpy(), acc_var=right.double().mean(0).cpu().numpy(),
                   kl_qp=(lqg.exp() * (lqg - lPg)).sum(-1).mean(0).cpu().numpy(),
                   kl_pq=(lPg.exp() * (lPg - lqg)).sum(-1).mean(0).cpu().numpy(),
                   kl_qn=(lqg.exp() * (lqg - lNg)).sum(-1).mean(0).cpu().numpy(),
                   kl_qp_clip=(qc * (qc / Pc).log()).sum(-1).mean(0).cpu().numpy(),
                   kl_pq_clip=(Pc * (Pc / qc).log()).sum(-1).mean(0).cpu().numpy())
        if joint:
            lc = torch.log_softmax(cum, -1)
            out.update(cell_lq1=lc[:, 0].float().cpu().numpy(), cell_lqT=lc[:, -1].float().cpu().numpy())
        # read-out fixed logits, spread, exclusion (theory_v9: more than 10 nats below the best of row and column)
        F1, FT = cum[:, 0].mean(0), cum[:, -1].mean(0)
        lqv = torch.log_softmax(cum, -1) if joint else lq.reshape(n, T, -1)
        out.update(F1=F1.cpu().numpy(), FT=FT.cpu().numpy(), lq_sd1=lqv[:, 0].std(0).cpu().numpy(),
                   lq_sdT=lqv[:, -1].std(0).cpu().numpy())
        for k, F in (("1", F1), ("T", FT)):
            if joint:
                F = F.reshape(R, R)
                gap = torch.minimum(F.max(1, keepdim=True).values - F, F.max(0, keepdim=True).values - F)
                out["excl_" + k] = int((gap > 10).sum())
            else:
                F = F.reshape(C, R)
                out["excl_" + k] = int(((F.max(-1, keepdim=True).values - F) > 10).sum())
        # per-step logit increments on the Bayes ladder: normal-equation sums per step
        if joint:
            mlam = torch.stack([cum.reshape(n, T, R, R).logsumexp(-1), cum.reshape(n, T, R, R).logsumexp(-2)], 2)
            prevm = torch.cat([torch.full_like(mlam[:, :1], np.log(R)), mlam[:, :-1]], 1)
            d = mlam - prevm
        else:
            d = lg.double().reshape(n, T, C, R)
        d = d - d.mean(-1, keepdim=True)
        Nn, SW, D = (torch.as_tensor(P["inc_" + k], device=dev).double() for k in ("N", "SW", "D"))
        X = torch.stack([Nn, SW - Nn, D - SW], -1)                                   # (n, T, 2, R, 3)
        out.update(reg_xtx=torch.einsum("ntvjk,ntvjl->tkl", X, X).cpu().numpy(),
                   reg_xtd=torch.einsum("ntvjk,ntvj->tk", X, d).cpu().numpy(),
                   reg_dtd=(d * d).sum((0, 2, 3)).cpu().numpy())
        del X, Nn, SW, D, d
        # memory truncation (an6)
        trunc = {}
        for w in TRUNC_W:
            lw = lg.clone()
            for t in range(w, T):
                hh, cc = _start(m, n)
                for s in range(t - w + 1, t + 1):
                    hh, cc, _ = _cell(_readin(m, obs[:, s], zin) @ W_ih.T + b + hh @ W_hh.T, cc)
                lw[:, t] = m.classifier_readout(hh)
            trunc[w] = marginals(lw.double().sum(1), joint).argmax(-1) == rt
        trunc["intact"] = right[:, -1]
        tr_right = torch.stack([trunc[k] for k in (*TRUNC_W, "intact")], 1)          # (n, 4, 2)
        out.update(trunc_right=tr_right.cpu().numpy(),
                   trunc_acc=np.stack([tr_right[ar, :, goal].double().mean(0).cpu().numpy(),
                                       tr_right.double().mean((0, 2)).cpu().numpy()], 1))
    # read-out row gradients under the run's own loss (starvation; early checkpoints; the loss on the CPU)
    if ep <= STARVE_UNTIL:
        torch.cuda.empty_cache()
        cum_g = lg.double().cpu().cumsum(1).requires_grad_()
        ga_cpu = torch.as_tensor(P["goal"]).long()
        cells = torch.as_tensor(P["logpost"]).double() if kind == "joint_full" else None
        loss = run_loss(cum_g, torch.as_tensor(P["jb"]).double()[torch.arange(n), :, ga_cpu], cells, ga_cpu, kind, m,
                        joint)
        dlg, = torch.autograd.grad(loss, cum_g)
        dlg = dlg.flip(1).cumsum(1).flip(1).float().to(dev)                         # d loss / d per-step logits
        last, _ = _last_linear(m)
        for prm in m.classifier_readout.parameters():
            prm.grad = None
        hh, cc = _start(m, n)
        with torch.enable_grad():
            for t in range(T):
                with torch.no_grad():
                    hh, cc, _ = _cell(_readin(m, obs[:, t], zin) @ W_ih.T + b + hh @ W_hh.T, cc)
                m.classifier_readout(hh).backward(dlg[:, t])
        out.update(ro_gnorm=last.weight.grad.norm(dim=-1).cpu().numpy(), ro_bgrad=last.bias.grad.cpu().numpy(),
                   loss=loss.item())
        for prm in m.parameters():
            prm.grad = None
    out.update(fac=fac, fac_rows=np.array(FAC_ROWS), factors=np.array(FACTORS),
               pool_x=poolr["x"], pool_u=np.concatenate([poolr["u"], poolr["u_prev"]]), pool_c=poolr["c"],
               tau=(1 / (1 - fsum / (n * (T - 1)))).float().cpu().numpy(),
               gate_mean=(gate_sum / (n * T)).float().cpu().numpy(),
               drive_rms=(drive / (n * T * H)).sqrt().float().cpu().numpy(),
               dec_r2=dec_r2, dec_c=dec_c, dec_steps=np.array(DEC_STEPS) + 1, basis_evr=evr,
               basis_V100=V_b[:, :100].cpu().numpy(), basis_mu=mu_b.cpu().numpy(), band_ids=band_ids,
               n_bands=len(band_ids), n_basis=len(P["basis_obs"]), t_forward=time.time() - t0)
    return out, ({k: v.cpu().numpy() for k, v in snaps.items()} if snap else None)


def weights_stats(m, w, w0, wp, lstm_trained, rho_cache):
    names = sorted(k for k in w if k not in ("all_K", "all_Q"))
    norm = np.array([w[k].double().norm().item() for k in names])
    dinit = np.array([(w[k].double() - w0[k].double()).norm().item() for k in names])
    dprev = np.array([(w[k].double() - wp[k].double()).norm().item() if wp is not None else np.nan for k in names])
    Whh, Wih = w["LSTM.weight_hh_l0"].double(), w["LSTM.weight_ih_l0"].double()
    if lstm_trained or "rho" not in rho_cache:
        H = Whh.shape[1]                                                    # per gate block (H x H): radius, norm
        rho_cache["rho"] = np.array([[np.abs(np.linalg.eigvals(Whh[k * H:(k + 1) * H].numpy())).max(),
                                      np.linalg.norm(Whh[k * H:(k + 1) * H].numpy(), 2)] for k in range(4)])

    def sv(A):
        return torch.linalg.svdvals(A).numpy()
    _, ro = _last_linear(m)
    return dict(w_names=np.array(names), w_norm=norm, w_dinit=dinit, w_dprev=dprev, rho_hh=rho_cache["rho"],
                sv_dhh=sv(Whh - w0["LSTM.weight_hh_l0"].double())[:64],
                sv_dih=sv(Wih - w0["LSTM.weight_ih_l0"].double())[:64], sv_ro=sv(w[ro + ".weight"].double()))


# ═══════════════════════════════════════════════════════════════════
# Update half (gradflow_analysis's locked convention)
# ═══════════════════════════════════════════════════════════════════

def _band_cross(e, st, W_hh, ZZ, V, OH, t, dev):
    """sum_B (<dW_B, dW> - |dW_B|^2) for the error e (B, H) split by the bands OH of the basis V (one backward probe per
    band; dW_B from the error's part in band B)"""
    B = len(e)
    xs = torch.zeros(B, device=dev, dtype=torch.float64)
    for c0 in range(0, B, BAND_CHUNK):
        cs = slice(c0, min(B, c0 + BAND_CHUNK))
        probes_b = ((e[cs] @ V)[:, None, :] * OH.T[None]) @ V.T
        Db = torch.stack([d for _, d in ga.backward({k: v[cs] for k, v in st.items()}, W_hh, probes_b, t, ga.KMAX)], 1)
        Mo = torch.einsum("cksg,clsg->cskl", Db, Db).double()
        Qs = torch.einsum("cksg,clg->cskl", Db, Db.sum(2)).double()
        zz = ZZ[cs][:, None]
        xs[cs] = ((Qs * zz).sum((-1, -2)) - (Mo * zz).sum((-1, -2))).sum(-1)
        del Db, Mo, Qs
    return xs


def credit(m, P, bas, bas_final, dev):
    """coherence, |dW|^2, dW's effective dimension and the 10% cross-band share for both variables' step-t errors at
    steps 17-30 (ga.SRC), on the first UPDATE_N permuted probes in fixed batches (fig1c_scan_credit.measures); the
    cross-band share also on the final checkpoint's bands for the first FIXED_BAND_N"""
    from types import SimpleNamespace
    t0 = time.time()
    V, OH = bas[1], bas[3]
    Vf, OHf = bas_final[1], bas_final[3]
    rows = P["perm"][:UPDATE_N]
    res = {k: np.zeros((UPDATE_N, len(ga.SRC)), np.float32) for k in ("cr_coh", "cr_E", "cr_dim", "cr_bandx")}
    res["cr_bandx_fixed"] = np.zeros((FIXED_BAND_N, len(ga.SRC)), np.float32)
    for b0 in range(0, UPDATE_N, UPDATE_BATCH):
        rb = rows[b0:b0 + UPDATE_BATCH]
        B = len(rb)
        obs = torch.as_tensor(P["obs"][rb], dtype=torch.float32, device=dev)
        zin = torch.as_tensor(P["Z"][P["env"][rb]], device=dev).reshape(B, -1)
        goal = P["goal"][rb]
        jb = P["jb"][rb].astype(np.float64)
        m.goal_ind = torch.as_tensor(goal, device=dev)
        m.joint_goal_belief = torch.as_tensor(jb[np.arange(B), :, goal], device=dev)
        with torch.no_grad(), torch.autocast(dev.type, enabled=False):
            x = _readin(m, obs, zin.reshape(B, 1, -1).expand(B, T, -1))
            st, W_hh = ga.forward_states(m, x)
            r = m.classifier_readout(st["u"])
            loss = ga.StepLoss(m, r)
            mo = SimpleNamespace(ctx_num=m.ctx_num, realization_num=m.realization_num, goal_ind=1 - m.goal_ind.long(),
                                 joint_goal_belief=torch.as_tensor(jb[np.arange(B), :, 1 - goal], device=dev),
                                 DKL_sym=m.DKL_sym)
            loss_o = ga.StepLoss(mo, r)
            Zk = torch.cat([x, st["u_prev"]], -1)
            for j, t in enumerate(ga.SRC):
                e = ga.step_error(m, loss, r, t) + ga.step_error(m, loss_o, r, t)
                Dk = torch.stack([d for _, d in ga.backward(st, W_hh, e, t, ga.KMAX)], 1)
                K = Dk.shape[1]
                zk = Zk[:, t - torch.arange(K, device=dev)]
                DD = torch.einsum("bkg,blg->bkl", Dk, Dk).double()
                ZZ = torch.einsum("bkd,bld->bkl", zk, zk).double()
                del Dk
                A = DD @ ZZ
                E = (DD * ZZ).sum((-1, -2))
                sl = slice(b0, b0 + B)
                res["cr_coh"][sl, j] = (DD.sum((-1, -2)) / DD.diagonal(dim1=-2, dim2=-1).sum(-1)).cpu().numpy()
                res["cr_E"][sl, j] = E.cpu().numpy()
                res["cr_dim"][sl, j] = (A.diagonal(dim1=-2, dim2=-1).sum(-1) ** 2
                                        / (A * A.transpose(-1, -2)).sum((-1, -2))).cpu().numpy()
                res["cr_bandx"][sl, j] = (_band_cross(e, st, W_hh, ZZ, V, OH, t, dev) / E).cpu().numpy()
                if b0 < FIXED_BAND_N:
                    nf = min(B, FIXED_BAND_N - b0)
                    xf = _band_cross(e[:nf], {k: v[:nf] for k, v in st.items()}, W_hh, ZZ[:nf], Vf, OHf, t, dev)
                    res["cr_bandx_fixed"][b0:b0 + nf, j] = (xf / E[:nf]).cpu().numpy()
            del st, x, Zk
    res.update(cr_rows=rows, cr_steps=ga.SRC + 1, cr_kmax=ga.KMAX, cr_batch=UPDATE_BATCH, t_credit=time.time() - t0)
    return res


# ═══════════════════════════════════════════════════════════════════
# Driver and checks
# ═══════════════════════════════════════════════════════════════════

def _save(path, arrays):
    np.savez(path + ".tmp.npz", **arrays)
    os.replace(path + ".tmp.npz", path)


def run(names, dev=torch.device("cuda:0"), only=None):
    P = probes()
    for run_name in names:
        kind = RUNS[run_name]
        eps_all = checkpoints(run_name)
        eps = eps_all if only is None else [e for e in eps_all if e in only]
        d = os.path.join(OUT, run_name)
        os.makedirs(d, exist_ok=True)
        cfg = _cfg(run_name)
        assert not cfg.get("learn_embeddings", False), "Z formula assumes learn_embeddings False"
        ck0 = torch.load(ckpt_path(run_name, 0), map_location="cpu", weights_only=False)
        same = np.array_equal(np.asarray(ck0["K"]), P["world_K"]) and np.array_equal(np.asarray(ck0["Q"]), P["world_Q"])
        assert same, f"{run_name}: not the probe world"
        m = build(run_name, eps_all[0], dev)
        w0 = load_weights(m, run_name, 0)
        load_weights(m, run_name, eps_all[-1])                                       # the final checkpoint's bands
        bas_final = basis(m, P, dev)
        _save(os.path.join(d, "basis_final.npz"), dict(episode=eps_all[-1], evr=bas_final[2], band_ids=bas_final[4],
                                                       V100=bas_final[1][:, :100].cpu().numpy()))
        lstm_trained = not cfg.get("reservoir", False)
        rho_cache = {}
        for ep in eps:
            path = os.path.join(d, f"ck_{ep:05d}.npz")
            snap = run_name in ("trained", "echo") and ep in SNAP
            snap_path = os.path.join(d, f"snap_{ep:05d}.npz")
            if os.path.exists(path) and (not snap or os.path.exists(snap_path)):
                continue
            i = eps_all.index(ep)
            wp = (torch.load(ckpt_path(run_name, eps_all[i - 1]), map_location="cpu", weights_only=False)["weights"]
                  if i > 0 else None)
            w = load_weights(m, run_name, ep)
            bas = basis(m, P, dev)
            out, snaps = forward(m, P, kind, ep, dev, bas, snap=snap)
            out.update(weights_stats(m, w, w0, wp, lstm_trained, rho_cache))
            if run_name in CREDIT:
                out.update(credit(m, P, bas, bas_final, dev))
            out.update(run=run_name, episode=ep, updates=ep - ep // 5, prev_episode=eps_all[i - 1] if i > 0 else -1)
            if snaps is not None:
                _save(snap_path, snaps)
            _save(path, out)
            print(f"{time.strftime('%H:%M:%S')} {run_name} ep {ep}: acc T {out['acc'][-1]:.3f}, forward "
                  f"{out['t_forward']:.0f} s" + (f", credit {out['t_credit']:.0f} s" if 't_credit' in out else ""),
                  flush=True)
            torch.cuda.empty_cache()


def check(run_name, ep, dev=torch.device("cuda:0")):
    """replay and autograd checks on one checkpoint (prints)"""
    P = probes()
    m = build(run_name, ep, dev)
    n = 256
    obs = torch.as_tensor(P["obs"][:n], dtype=torch.float32, device=dev)
    zin = torch.as_tensor(P["Z"][P["env"][:n]], device=dev).reshape(n, 1, -1).expand(n, T, -1)
    W_ih, W_hh, b = ga.lstm_weights(m)
    with torch.no_grad():
        x = _readin(m, obs, zin)
        h, c = _start(m, n)
        us = []
        for t in range(T):
            h, c, _ = _cell(x[:, t] @ W_ih.T + b + h @ W_hh.T, c)
            us.append(h)
        u = torch.stack(us, 1)
        u_ref, _ = m.LSTM(x, (m.STM.expand(-1, n, -1).contiguous(), m.LTM.expand(-1, n, -1).contiguous()))
        print(f"hand LSTM vs nn.LSTM: max |du| {(u - u_ref).abs().max().item():.1e}")
        st, _ = ga.forward_states(m, x)
        print(f"hand LSTM vs ga.forward_states: max |du| {(u - st['u']).abs().max().item():.1e}")
    # truncation at w = T reproduces the intact logits
    with torch.no_grad():
        lg = m.classifier_readout(u)
        hh, cc = _start(m, n)
        for s in range(T):
            hh, cc, _ = _cell(x[:, s] @ W_ih.T + b + hh @ W_hh.T, cc)
        dl = (m.classifier_readout(hh) - lg[:, -1]).abs().max().item()
        print(f"truncation w = T vs intact at T: max |dlogit| {dl:.1e}")
    # the hand-written dW of the step-15 error (all lags: KMAX = 15) vs autograd
    if run_name in CREDIT:
        from types import SimpleNamespace
        t = ga.KMAX
        goal = P["goal"][:n]
        jb = P["jb"][:n].astype(np.float64)
        m.goal_ind = torch.as_tensor(goal, device=dev)
        m.joint_goal_belief = torch.as_tensor(jb[np.arange(n), :, goal], device=dev)
        with torch.no_grad():
            st, W_hh = ga.forward_states(m, x)
            r = m.classifier_readout(st["u"])
            loss = ga.StepLoss(m, r)
            mo = SimpleNamespace(ctx_num=m.ctx_num, realization_num=m.realization_num, goal_ind=1 - m.goal_ind.long(),
                                 joint_goal_belief=torch.as_tensor(jb[np.arange(n), :, 1 - goal], device=dev),
                                 DKL_sym=m.DKL_sym)
            e = ga.step_error(m, loss, r, t) + ga.step_error(m, ga.StepLoss(mo, r), r, t)
            Zk = torch.cat([x, st["u_prev"]], -1)
            dW = sum(torch.einsum("bg,bd->gd", d, Zk[:, t - k]) for k, d in ga.backward(st, W_hh, e, t, ga.KMAX))
        Wi, Wh = W_ih.clone().requires_grad_(), W_hh.clone().requires_grad_()
        with torch.enable_grad():
            hh, cc = _start(m, n)
            for s in range(t + 1):
                hh, cc, _ = _cell(x[:, s] @ Wi.T + b + hh @ Wh.T, cc)
            gi, gh = torch.autograd.grad((hh * e).sum(), (Wi, Wh))
        ref = torch.cat([gi, gh], 1)
        print(f"dW hand vs autograd (step {t + 1}, both variables' errors): rel err "
              f"{((dW - ref).norm() / ref.norm()).item():.1e}")
    # factor decomposition: the traces add up, the derived covariances are PSD
    with torch.no_grad():
        n2 = len(P["obs"])
        groups = factor_groups(P, dev)
        X = torch.randn(n2, 64, device=dev) + torch.as_tensor(P["cell"] % 7, device=dev)[:, None].float()
        cv = factor_covs(X, groups)
        tr = {k: v.diagonal().sum().item() for k, v in cv.items()}
        print("factor traces: total {total:.3f} = env {env:.3f} + cell {cell:.3f} + envcell {envcell:.3f} + res "
              "{res:.3f}; cell = r0 {r0:.3f} + r1 {r1:.3f} + joint {joint:.3f}".format(**tr))
        mins = {k: torch.linalg.eigvalsh(cv[k]).min().item() / max(tr["total"], 1e-12) for k in ("joint", "envcell")}
        print("min eigenvalue / total trace (should be >= -1e-9): "
              + ", ".join(f"{k} {v:.1e}" for k, v in mins.items()))


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "probes":
        probes()
    elif what == "run":
        only = [int(e) for e in os.environ["TD_ONLY"].split(",")] if os.environ.get("TD_ONLY") else None
        run(sys.argv[2:], only=only)
    elif what == "check":
        check(sys.argv[2], int(sys.argv[3]))
