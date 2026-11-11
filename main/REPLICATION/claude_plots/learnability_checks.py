"""Forward-pass checks behind learnability_section.tex (the supplement's account of the "controls" figure: why ESNs stall
under marginal supervision and recover under joint supervision). No training. The saved final networks of the figure's
runs (training-dynamics runs, seed 20261029, one world; final_net.pth) are replayed in fp32 with a hand-written LSTM on
fresh episodes from the 50 probe environments of td_analysis (analysis/probes.npz: pairs of test variables, their
likelihood tables L and the networks' context input Z).

Episodes: per environment N_PER_CELL episodes at each of the 100 joint realizations, T = 30 steps,
o_ti ~ Bernoulli(L_i(r0, r1)). In each cell the first half of the episodes fits every regression and the second half
scores it (held-out R^2 = 1 - SS_res / SS_tot, pooled over the output dimensions; one value per environment, then the
mean and the range over the 50 environments).

A  Lemma 1(ii) at t = 2, exact from L (50 environments x 2 goal variables = 100 cases):
     Q[o1, o2, r] = ln P(o1, o2 | r_g = r) - ln P(o1 | r_g = r) - ln P(o2 | r_g = r) must not depend on r for a memoryless
     marginal update to give Joint Bayes ("spread": its range over r, maximized over the 32 x 32 observation pairs, and
     its probability-weighted RMS deviation from the mean over r, nats); and the weaker condition that survives a
     step-dependent update: every log-ratio matrix ln P(o1, o2 | r) - ln P(o1, o2 | r') is a row term plus a column term
     ("share": the variance share of the remainder after double-centring, uniform over the 32 x 32 pairs).
B  how memoryless a network is, from its marginal log beliefs lb (n, T, 2, R), centred over the R values (for a network
   with marginal belief updating this is its accumulated output; Naive Bayes scores 1 by construction):
     hist    R^2 of lb at T on the histogram of the 32 observation patterns over the episode (= the best memoryless
             update, the same at every step);
     counts  R^2 of lb at T on [1, counts of the 5 observation features] (the Naive family);
     step    R^2 of the update lb_t - lb_{t-1} on the current observation pattern, fitted per step t = 2..30 (so a clock
             is allowed), averaged over steps; "step_pooled": one fit for all steps (no clock);
     state   R^2 of the summed state sum_t h_t on [1, counts] and on the histogram.
C  the gate of Lemma 2 for networks with joint belief updating (and Joint Bayes): G = B_T(r*_{not g} | r*_g), the
   conditional probability of the true value of the other variable given the true goal value, both variables as goal:
   median, share below 1% and below 0.1%, mean log, share of episodes in which the true value ranks first; and the
   interaction share of the accumulated joint output (variance over the 100 cells not explained by a row plus a column
   term, per episode, averaged).
D  checks: the replay against td_analysis's stored marginal log beliefs (ckpt_ep49995 on the stored probe episodes),
   accuracy at T against Joint and Naive Bayes on the same episodes, every run's world against the probes' world, and
   the gradient identities of Lemmas 2 and 3 by autograd on the paper's loss (identities()).
C through training (mode "course"; COURSE): the measures of C at saved checkpoints (episodes 0 ... 40,000 and the final
   weights) for an ESN (N = 1,000) and a Fully Trained RNN (N = 200) with joint belief updating, each under marginal
   and under joint supervision (suffix _jl; the same seed), and at the first and the final weights for the ESNs of the
   other widths; 10 episodes per cell. "closed" / "open": the share of (joint realization, goal variable) pairs whose
   step-1 gate is below 1% in more than 99% / fewer than 1% of the episodes; "closed_kept" / "closed_from": of the pairs
   closed at a checkpoint, the share closed at the end / of the pairs closed at the end, the share closed by then.
E  the final networks with joint belief updating under marginal and under joint supervision (mode "split"; SPLIT):
   the paper's loss (symmetrized KL to the power 1/2, to Joint Bayes; "loss"), the loss as the training code computes it
   (power taken per realization, soft-clipped beliefs; "loss_trained", see fr_loss_trained) and the accuracy at T in the
   probe environments ("test") and in 50 environments of two training variables ("train"); and the accuracy split by
   the step-1 gate of the true joint realization (closed: below 1%), with Joint and Naive Bayes on the same episodes;
   10 episodes per cell.

Modes: no argument = A, B, C, D on the final networks; "extra [names]" = the figure's other ESNs (EXTRA: the N = 2,000
ESN of the first column, trained in another world, and the two N = 10,000 ESNs, their last checkpoint while they
train); "course" = C through training; "split" = E. The last three add to the stored results.

Output: main/DATA/experimental/esn_learnability/checks_11-11/results.json (means and ranges) and per_env.npz.
Run from the repository root (about 3 GB of GPU memory):
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/learnability_checks.py [extra [names] | course | split]

Measured 11-11 (one seed per network; the figure's widths: ESN N = 200 ... 10,000, Fully Trained N = 50 ... 400):
A  the condition fails in all 100 cases (largest spread over r at least 3.3 nats; non-additive share at least .021).
B  counts R^2: ESNs .971-.989, Fully Trained .794-.864, Joint Bayes .734, Naive Bayes 1. hist R^2: ESNs .999, Fully
   Trained .956-.962, Joint Bayes .956, so the contrast lies in the Naive family, not in the memoryless class at large.
C  share of episodes whose true joint realization has a step-1 gate below 1%: marginal supervision, ESNs .64-.70
   (N = 10,000 at episode 35,000, still training: .82), Fully Trained .03-.25; joint supervision, at most .005.
   Through training: no pair closed at the first weights; ESN (N = 1,000) closed share .70 by episode 2,000 and
   .70-.74 after; Fully Trained (N = 200) .19 by episode 5,000 and .16-.22 after; joint supervision, none closed.
   Persistence: of the pairs closed at episode 2,000 (ESN) / 10,000 (Fully Trained), .96 / .73 are closed at the end
   (the Fully Trained RNN reopens a quarter, mostly after episode 30,000); no pair open then is closed at the end.
D  identities to 1e-15.
E  loss under marginal / joint supervision (test): ESNs 1.07 / .89, .95 / .60, 1.12 / .43, .90 / .30 (N = 200 ...
   2,000); Fully Trained .56 / .65 (N = 50, the one exception), .34 / .31, .30 / .23, .26 / .19, .24 / .14, .25 / .11;
   the same order in the training environments and under the loss as the training code computes it (ESNs .189 / .182,
   .164 / .123, .178 / .086, .146 / .060; Fully Trained .106 / .127, .064 / .061, .055 / .043, .047 / .036,
   .042 / .027, .042 / .021). Accuracy at open / closed gates: ESN N = 2,000 .74 / .37 (Joint Bayes .76 / .77, Naive
   Bayes .48 / .45); Fully Trained N = 400 .76 / .48. Share of the accuracy gap to Joint Bayes that lies in
   closed-gate episodes: ESNs .74-.97, Fully Trained .06, .14, .69, .74, .95, 1.03 (so not the two narrowest)."""
import os
import sys
import json
import time
import itertools
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
OUT = os.path.join(ROOT, "main", "DATA", "experimental", "esn_learnability", "checks_11-11")
SEED = 20261029
SIM_SEED = 20261111
T, R, C, NO = 30, 10, 2, 5
N_PER_CELL = 40
CHUNK = 4000
MARGINAL = ("echo_N200", "echo_N500", "echo", "trained_N50", "trained_N100", "trained_N150", "trained_N200",
            "trained_N300", "trained_N400", "trained_N500", "trained")
JOINT = tuple(f"joint_{r}" for r in ("echo_N200", "echo_N500", "echo", "echo_N2000", "trained_N50", "trained_N100",
                                     "trained_N150", "trained_N200", "trained_N300", "trained_N400", "trained_N500",
                                     "trained"))
JOINT_JL = tuple(f"{r}_jl" for r in JOINT)
REPLAY_CHECK = ("echo", "trained", "trained_N50", "joint_echo", "joint_echo_jl", "joint_trained")
# the figure's other ESNs: (name, weights in order of preference, episodes per cell). The N = 10k runs save a checkpoint
# while they train (final_net.pth once they finish); the N = 2k of the first column lives in another world (seed
# 20261031), so it gets its own world's tables for the same pairs of test variables
EXTRA = (("echo_N2000", (os.path.join(ROOT, "main", "DATA", "experimental", "esn_widths", "reservoir_ctx_2_2000N_net.pth"),),
          N_PER_CELL),
         ("echo_N10000_b8k", tuple(os.path.join(TD, f"echo_N10000_b8k_s{SEED}", f) for f in
                                   ("final_net.pth", "ckpt_latest_net.pth")), 10),
         ("joint_echo_N10000_b8k", tuple(os.path.join(TD, f"joint_echo_N10000_b8k_s{SEED}", f) for f in
                                         ("final_net.pth", "ckpt_latest_net.pth")), 10))
PATTERNS = np.array(list(itertools.product([0, 1], repeat=NO)), float)[:, ::-1].copy()     # row k: the bits of code k
# C through training: (runs, checkpoint episodes; None = final_net.pth), 10 episodes per cell
COURSE = ((("joint_echo", "joint_echo_jl", "joint_trained_N200", "joint_trained_N200_jl"),
           (0, 5, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, None)),
          (tuple(f"joint_echo_N{n}{s}" for n in (200, 500, 2000) for s in ("", "_jl")), (0, None)))

# E: the networks with joint belief updating at the figure's widths, under marginal and under joint supervision
SPLIT = tuple(f"joint_{r}{s}" for r in ("echo_N200", "echo_N500", "echo", "echo_N2000", "trained_N50", "trained_N100",
                                        "trained_N150", "trained_N200", "trained_N300", "trained_N400")
              for s in ("", "_jl"))
N_TEST_VARS = 50                                                                    # variables 0..49 are never a training goal
ARRAYS = ("_steps", "_cells")                                                       # per-environment arrays, not scalars

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# ═══════════════════════════════════════════════════════════════════
# A: Lemma 1(ii) at t = 2
# ═══════════════════════════════════════════════════════════════════

def two_step_condition(L):
    """L (5, R, R) raw order [i, r0, r1] -> per goal variable (spread_max, spread_rms, share_min, share_median)"""
    p = np.exp(PATTERNS @ np.log(L).reshape(NO, -1) + (1 - PATTERNS) @ np.log1p(-L).reshape(NO, -1)).reshape(32, R, R)
    out = []
    for g in range(C):
        q = p if g == 0 else p.transpose(0, 2, 1)                                    # (32, goal value, other value)
        P1 = q.mean(2)
        P2 = np.einsum("arc,brc->abr", q, q) / R
        Q = np.log(P2) - np.log(P1)[:, None, :] - np.log(P1)[None, :, :]
        dev = Q - Q.mean(-1, keepdims=True)
        w = P2 / P2.sum((0, 1), keepdims=True)                                       # P(o1, o2 | r)
        shares = []
        for a, b in itertools.combinations(range(R), 2):
            D = np.log(P2[:, :, a]) - np.log(P2[:, :, b])
            I = D - D.mean(1, keepdims=True) - D.mean(0, keepdims=True) + D.mean()
            shares.append(I.var() / D.var())
        out.append(((Q.max(-1) - Q.min(-1)).max(), np.sqrt((w * dev ** 2).sum((0, 1)).mean()), min(shares),
                    float(np.median(shares))))
    return np.array(out)


# ═══════════════════════════════════════════════════════════════════
# Networks and observers
# ═══════════════════════════════════════════════════════════════════

def weights_path(run, ep=None):
    return os.path.join(TD, f"{run}_s{SEED}", "final_net.pth" if ep is None else f"ckpt_ep{ep:05d}_net.pth")


def load(path, dev):
    """the weights for replay() (fp32) and the checkpoint's other entries (its world K, Q; the episode of a checkpoint
    saved during training)"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    w = ck.pop("weights")
    assert set(w) == {"STM", "LTM", "classifier_readin.weight", "classifier_readin.bias", "classifier_readout.weight",
                      "classifier_readout.bias", "LSTM.weight_ih_l0", "LSTM.weight_hh_l0", "LSTM.bias_ih_l0",
                      "LSTM.bias_hh_l0"}, sorted(w)
    f = lambda k: w[k].float().to(dev)
    W = dict(W_in=f("classifier_readin.weight"), b_in=f("classifier_readin.bias"), W_ih=f("LSTM.weight_ih_l0"),
             W_hh=f("LSTM.weight_hh_l0"), b=f("LSTM.bias_ih_l0") + f("LSTM.bias_hh_l0"), h0=f("STM").reshape(1, -1),
             c0=f("LTM").reshape(1, -1), W_out=f("classifier_readout.weight"), b_out=f("classifier_readout.bias"))
    return W, ck


@torch.no_grad()
def replay(W, obs, zin):
    """obs (n, T, 5), zin (n, 10) float32 -> per-step output (n, T, O) and the summed state (n, H)"""
    n, H = len(obs), W["W_hh"].shape[1]
    lg = torch.zeros(n, T, W["W_out"].shape[0], device=obs.device)
    S = torch.zeros(n, H, device=obs.device)
    for a in range(0, n, CHUNK):
        o, z = obs[a:a + CHUNK], zin[a:a + CHUNK]
        h, c = W["h0"].expand(len(o), H), W["c0"].expand(len(o), H)
        for t in range(T):
            x = torch.relu(F.linear(torch.cat((o[:, t], z), -1), W["W_in"], W["b_in"]))
            i, f, g, oo = (x @ W["W_ih"].T + W["b"] + h @ W["W_hh"].T).chunk(4, -1)
            c = f.sigmoid() * c + i.sigmoid() * g.tanh()
            h = oo.sigmoid() * c.tanh()
            lg[a:a + CHUNK, t] = F.linear(h, W["W_out"], W["b_out"])
            S[a:a + CHUNK] += h
    return lg, S


def log_beliefs(cum):
    """marginal log beliefs (n, T, 2, R) from accumulated outputs (n, T, O), float64; joint outputs are r0-major"""
    cum = cum.double()
    if cum.shape[-1] == R * R:
        lq = torch.log_softmax(cum, -1).reshape(*cum.shape[:-1], R, R)
        return torch.stack([lq.logsumexp(-1), lq.logsumexp(-2)], -2)
    return torch.log_softmax(cum.reshape(*cum.shape[:-1], C, R), -1)


def bayes(o, L):
    """o (n, T, 5), L (5, R, R) float64 -> Joint Bayes's log posterior over the cells (n, T, R, R), its marginal log
    beliefs and Naive Bayes's (n, T, 2, R)"""
    n = len(o)
    ll = torch.einsum("nti,ijk->ntjk", o, L.log()) + torch.einsum("nti,ijk->ntjk", 1 - o, torch.log1p(-L))
    lam = ll.cumsum(1)
    logpost = lam - lam.reshape(n, T, -1).logsumexp(-1)[..., None, None]
    log_jb = torch.stack([logpost.logsumexp(3), logpost.logsumexp(2)], 2)
    Lb = torch.stack([L.mean(2), L.mean(1)], 1)                                      # Naive tables (5, 2, R)
    lpx = torch.einsum("nti,ikr->ntkr", o, Lb.log()) + torch.einsum("nti,ikr->ntkr", 1 - o, torch.log1p(-Lb))
    return logpost, log_jb, torch.log_softmax(lpx.cumsum(1), -1)


# ═══════════════════════════════════════════════════════════════════
# Measures
# ═══════════════════════════════════════════════════════════════════

def r2(X, Y, fit, test):
    beta = torch.linalg.pinv(X[fit]) @ Y[fit]
    res, tot = Y[test] - X[test] @ beta, Y[test] - Y[test].mean(0)
    return float(1 - (res ** 2).sum() / (tot ** 2).sum())


def r2_groups(code, Y, fit, test):
    """least squares on the one-hot of code (n,) = group means; a pattern absent from the fit half gets the fit mean"""
    d = Y.shape[1]
    sums = torch.zeros(32, d, dtype=Y.dtype, device=Y.device).index_add_(0, code[fit], Y[fit])
    cnt = torch.zeros(32, dtype=Y.dtype, device=Y.device).index_add_(0, code[fit], torch.ones_like(Y[fit][:, 0]))
    mean = torch.where(cnt[:, None] > 0, sums / cnt[:, None].clamp_min(1), Y[fit].mean(0, keepdim=True))
    res, tot = Y[test] - mean[code[test]], Y[test] - Y[test].mean(0)
    return float((res ** 2).sum()), float((tot ** 2).sum())


def cen(a):
    return a - a.mean(-1, keepdim=True)


def memoryless(lb, code, hist, cnts, fit, test, S=None):
    """the B measures for marginal log beliefs lb (n, T, 2, R)"""
    n = len(lb)
    out = {}
    YT = cen(lb[:, -1]).reshape(n, -1)
    out["hist"], out["counts"] = r2(hist, YT, fit, test), r2(cnts, YT, fit, test)
    D = cen(lb[:, 1:] - lb[:, :-1]).reshape(n, T - 1, -1)                            # updates of steps 2..T
    per = [r2_groups(code[:, t + 1], D[:, t], fit, test) for t in range(T - 1)]
    out["step"] = float(np.mean([1 - a / b for a, b in per]))
    out["step_var"] = 1 - sum(a for a, _ in per) / sum(b for _, b in per)            # steps weighted by their variance
    nf, nt = len(fit), len(test)
    cf = code[:, 1:][fit].reshape(-1)
    ct = code[:, 1:][test].reshape(-1)
    Yf, Yt = D[fit].reshape(nf * (T - 1), -1), D[test].reshape(nt * (T - 1), -1)
    a, b = r2_groups(torch.cat([cf, ct]), torch.cat([Yf, Yt]), torch.arange(len(cf), device=lb.device),
                     len(cf) + torch.arange(len(ct), device=lb.device))
    out["step_pooled"] = 1 - a / b
    if S is not None:
        out["state_counts"], out["state_hist"] = r2(cnts, S.double(), fit, test), r2(hist, S.double(), fit, test)
    return out


def gate(lq, r):
    """lq (n, T, R, R) joint log belief (r0-major), r (n, 2) true values (episodes ordered by cell, equally many per
    cell) -> the C measures at T, the per-step share of gates below 1% and its step-1 value per (cell, goal variable)"""
    n = len(lq)
    ar = torch.arange(n, device=lq.device)
    cond0 = lq[ar, :, r[:, 0], :]                                                    # (n, T, R): over r1 given true r0
    cond1 = lq[ar, :, :, r[:, 1]]                                                    # (n, T, R): over r0 given true r1
    cond0 = cond0 - cond0.logsumexp(-1, keepdim=True)
    cond1 = cond1 - cond1.logsumexp(-1, keepdim=True)
    lg0 = cond0.gather(-1, r[:, 1].reshape(n, 1, 1).expand(n, T, 1)).squeeze(-1)     # goal = variable 0
    lg1 = cond1.gather(-1, r[:, 0].reshape(n, 1, 1).expand(n, T, 1)).squeeze(-1)     # goal = variable 1
    lG = torch.stack([lg0, lg1], -1)                                                 # (n, T, 2)
    first = torch.stack([cond0.argmax(-1) == r[:, 1:2], cond1.argmax(-1) == r[:, 0:1]], -1)
    M = lq[:, -1]
    inter = M - M.mean(2, keepdim=True) - M.mean(1, keepdim=True) + M.mean((1, 2), keepdim=True)
    tot = M - M.mean((1, 2), keepdim=True)
    return dict(gate_median=float(lG[:, -1].exp().median()), gate_mean=float(lG[:, -1].exp().mean()),
                gate_below_1pct=float((lG[:, -1] < np.log(.01)).double().mean()),
                gate_below_01pct=float((lG[:, -1] < np.log(.001)).double().mean()),
                gate_meanlog=float(lG[:, -1].mean()), gate_first=float(first[:, -1].double().mean()),
                gate_below_1pct_step1=float((lG[:, 0] < np.log(.01)).double().mean()),
                gate_first_step1=float(first[:, 0].double().mean()),
                interaction_share=float(((inter ** 2).sum((1, 2)) / (tot ** 2).sum((1, 2))).mean()),
                gate_below_1pct_steps=(lG < np.log(.01)).double().mean((0, 2)).cpu().numpy(),
                gate_below_1pct_step1_cells=(lG[:, 0] < np.log(.01)).double().reshape(R * R, -1, 2).mean(1).cpu().numpy())


def identities(dev):
    """Lemmas 2 and 3 by autograd (float64) on random accumulated outputs with the paper's loss (symmetrized KL to the
    power 1/2); and the gradient of a product belief, which is not a row term plus a column term"""
    g = torch.Generator(device="cpu").manual_seed(0)
    M = (3 * torch.randn(64, R, R, generator=g, dtype=torch.float64)).to(dev).requires_grad_(True)
    Pj = torch.softmax(3 * torch.randn(64, R * R, generator=g, dtype=torch.float64), -1).reshape(64, R, R).to(dev)
    Pg = Pj.sum(2)

    def fr_loss(m):                                                                  # m (64, R): goal logits
        q = torch.softmax(m, -1)
        return ((.5 * (Pg * (Pg / q).log()).sum(-1) + .5 * (q * (q / Pg).log()).sum(-1)) ** .5).sum()

    mg = M.logsumexp(2)                                                              # goal = variable 0
    direct = torch.autograd.grad(fr_loss(mg), M)[0]
    m_leaf = mg.detach().clone().requires_grad_(True)
    marg = torch.autograd.grad(fr_loss(m_leaf), m_leaf)[0]
    gates = torch.softmax(M.detach(), 2)
    lemma2 = float((direct - gates * marg[:, :, None]).abs().max())
    Mj = M.detach().reshape(64, -1).clone().requires_grad_(True)
    kl = (Pj.reshape(64, -1) * (Pj.reshape(64, -1).log() - torch.log_softmax(Mj, -1))).sum()
    lemma3 = float((torch.autograd.grad(kl, Mj)[0] - (torch.softmax(Mj, -1) - Pj.reshape(64, -1))).abs().max())
    a, b = M.detach()[:, :, 0], M.detach()[:, 0, :]
    Mp = (a[:, :, None] + b[:, None, :]).clone().requires_grad_(True)                # product belief
    gp = torch.autograd.grad(fr_loss(Mp.logsumexp(2)), Mp)[0]
    inter = gp - gp.mean(2, keepdim=True) - gp.mean(1, keepdim=True) + gp.mean((1, 2), keepdim=True)
    return dict(lemma2_max_abs_error=lemma2, lemma3_max_abs_error=lemma3,
                gates_sum_error=float((gates.sum(2) - 1).abs().max()),
                product_belief_gradient_interaction_share=float(((inter ** 2).sum((1, 2)) / (gp ** 2).sum((1, 2))).mean()),
                gradient_scale=float(direct.abs().mean()))


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def summary(vals):
    v = np.asarray(vals, float)
    return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))


def episodes(L, Z, n_per_cell, seed, dev):
    """fresh episodes in the environments with likelihood tables L (n_env, 5, R, R) and network inputs Z (n_env, 5, 2):
    n_per_cell at each of the 100 joint realizations, o_ti ~ Bernoulli(L_i(r0, r1)); in each cell the first half of the
    episodes fits the regressions and the second half scores them -> per-environment tensors, fit, test, r (n, 2)"""
    rng = np.random.default_rng(seed)
    n = R * R * n_per_cell
    cell = np.repeat(np.arange(R * R), n_per_cell)
    r_np = np.stack([cell // R, cell % R], 1)
    within = np.tile(np.arange(n_per_cell), R * R)
    fit = torch.as_tensor(np.flatnonzero(within < n_per_cell // 2), device=dev)
    test = torch.as_tensor(np.flatnonzero(within >= n_per_cell // 2), device=dev)
    obs_all = np.stack([rng.random((n, T, NO)) < L[e][:, r_np[:, 0], r_np[:, 1]].T[:, None, :]
                        for e in range(len(L))])                                     # (env, n, T, 5)
    weights = 2 ** np.arange(NO)
    envs = []
    for e in range(len(L)):
        o = torch.as_tensor(obs_all[e], dtype=torch.float64, device=dev)
        code = torch.as_tensor(obs_all[e] @ weights, device=dev).long()              # (n, T)
        hist = F.one_hot(code, 32).double().sum(1)                                   # (n, 32)
        cnts = torch.cat([torch.ones(n, 1, dtype=torch.float64, device=dev), o.sum(1)], 1)
        envs.append(dict(o=o, code=code, hist=hist, cnts=cnts, L=torch.as_tensor(L[e], device=dev),
                         zin=torch.as_tensor(Z[e], device=dev).reshape(1, -1).expand(n, -1).contiguous()))
    return envs, fit, test, torch.as_tensor(r_np, device=dev)


def observers(envs, fit, test, r):
    """the B and C measures of Joint Bayes and the B measures of Naive Bayes, one row per environment"""
    rows_b, rows_n = [], []
    for E in envs:
        logpost, log_jb, log_nb = bayes(E["o"], E["L"])
        b = memoryless(log_jb, E["code"], E["hist"], E["cnts"], fit, test)
        b["acc"] = float((log_jb[:, -1].argmax(-1) == r).double().mean())
        b.update(gate(logpost, r))
        rows_b.append(b)
        nb = memoryless(log_nb, E["code"], E["hist"], E["cnts"], fit, test)
        nb["acc"] = float((log_nb[:, -1].argmax(-1) == r).double().mean())
        rows_n.append(nb)
    return rows_b, rows_n


def network(W, envs, fit, test, r):
    """the B measures of one network (and the C measures if its output is joint), one row per environment"""
    rows, t0 = [], time.time()
    for E in envs:
        if len(rows) == 1 and (time.time() - t0) * len(envs) > 120:                  # slow (wide) network: say how long
            print(f"   about {(time.time() - t0) * len(envs) / 60:.1f} min for this network", flush=True)
        lg, S = replay(W, E["o"].float(), E["zin"])
        cum = lg.cumsum(1)
        lb = log_beliefs(cum)
        m = memoryless(lb, E["code"], E["hist"], E["cnts"], fit, test, S)
        m["acc"] = float((lb[:, -1].argmax(-1) == r).double().mean())
        if cum.shape[-1] == R * R:
            m.update(gate(torch.log_softmax(cum.double(), -1).reshape(len(cum), T, R, R), r))
        rows.append(m)
    return rows


def course(dev, P):
    """C through training: the gate measures of saved checkpoints on one set of fresh episodes in the probe environments
    (10 per cell). "closed" / "open": the share of (joint realization, goal variable) pairs whose step-1 gate is below 1%
    in more than 99% / fewer than 1% of the episodes, over environments and observations (the two sum to 1 when the
    closure does not depend on the input). "closed_kept": of the pairs closed at this checkpoint, the share closed at the
    end of training; "closed_from": of the pairs closed at the end, the share already closed at this checkpoint
    -> ({run: {episode: measures}}, {"C_course__run__episode": the closure frequencies (cell, goal variable)})"""
    envs, _, _, r = episodes(P["L"], P["Z"], 10, SIM_SEED + 100, dev)
    out, freqs = {}, {}
    for runs, eps in COURSE:
        for run in runs:
            out[run] = {}
            for ep in eps:
                path = weights_path(run, ep)
                if not os.path.exists(path):
                    print("no weights:", run, ep)
                    continue
                W, ck = load(path, dev)
                assert np.array_equal(ck["K"], P["world_K"]) and np.array_equal(ck["Q"], P["world_Q"]), run
                rows = []
                for E in envs:
                    cum = replay(W, E["o"].float(), E["zin"])[0].cumsum(1)
                    g = gate(torch.log_softmax(cum.double(), -1).reshape(len(cum), T, R, R), r)
                    g["acc"] = float((log_beliefs(cum)[:, -1].argmax(-1) == r).double().mean())
                    rows.append(g)
                freq = np.mean([x["gate_below_1pct_step1_cells"] for x in rows], 0)  # (cell, goal variable)
                m = {k: float(np.mean([x[k] for x in rows])) for k in rows[0] if not k.endswith(ARRAYS)}
                m.update(closed=float((freq > .99).mean()), open=float((freq < .01).mean()))
                out[run]["final" if ep is None else str(ep)] = m
                freqs[f"C_course__{run}__{'final' if ep is None else ep}"] = freq
                print(f"course {run:24s} {'final' if ep is None else ep:>6}",
                      {k: round(m[k], 3) for k in ("acc", "gate_below_1pct_step1", "gate_below_1pct", "closed", "open",
                                                   "gate_first", "gate_median", "interaction_share")}, flush=True)
                del W
                torch.cuda.empty_cache()
            if "final" in out[run]:
                last = freqs[f"C_course__{run}__final"] > .99
                for name, m in out[run].items():
                    was = freqs[f"C_course__{run}__{name}"] > .99
                    m.update(closed_kept=float(last[was].mean()) if was.any() else None,
                             closed_from=float(was[last].mean()) if last.any() else None)
                print(f"course {run:24s} closed_kept", {k: m["closed_kept"] and round(m["closed_kept"], 3)
                                                         for k, m in out[run].items()}, flush=True)
                print(f"course {run:24s} closed_from", {k: m["closed_from"] and round(m["closed_from"], 3)
                                                         for k, m in out[run].items()}, flush=True)
    return out, freqs


def fr_loss(lb, log_jb):
    """the paper's loss between marginal log beliefs (n, T, 2, R): symmetrized KL to the power 1/2, mean over episodes,
    steps and both variables as goal (float64) -> (loss, plain symmetrized KL)"""
    sym = .5 * (log_jb.exp() * (log_jb - lb)).sum(-1) + .5 * (lb.exp() * (lb - log_jb)).sum(-1)
    return float(sym.clamp_min(0).sqrt().mean()), float(sym.mean())


def fr_loss_trained(lb, log_jb, eps=1e-3):
    """the loss as the training code computes it (model_backward.SANITY_loss -> DKL_loss), which differs from the paper's
    formula in where the power is taken: beliefs soft-clipped to [eps, 1 - eps] (model_backward.soft_clip), then per
    realization the symmetrized KL term (q - p)(ln q - ln p) / 2, minus its minimum over the batch (here: over the
    environment's episodes), plus 1e-8, to the power 1/2; mean over realizations, steps, episodes and both variables"""
    def soft_clip(x):
        x = (1 - eps) - torch.sigmoid((1 - eps - x) / eps) * (1 - eps - x)
        return eps + torch.sigmoid((x - eps) / eps) * (x - eps)
    q, p = soft_clip(lb.exp()), soft_clip(log_jb.exp())
    d = .5 * (q - p) * (q.log() - p.log())
    return float((d - d.min() + 1e-8).sqrt().mean())


def split(dev, P):
    """E: for the final networks with joint belief updating, (1) the marginal loss and the accuracy at T in the probe
    (test) environments and in 50 environments of two training variables, and (2) accuracy on a variable split by its
    step-1 gate on the true joint realization (closed: below 1%), with Joint and Naive Bayes on the same episodes"""
    sys.path.insert(0, os.path.join(ROOT, "main", "REPLICATION"))
    import td_analysis as td
    rng = np.random.default_rng(SIM_SEED + 200)
    pairs = np.sort(np.stack([rng.choice(np.arange(N_TEST_VARS, 500), 2, replace=False) for _ in range(len(P["L"]))]), 1)
    sets, out = {}, {"train_pairs": pairs.tolist()}
    for run in SPLIT:
        if not os.path.exists(weights_path(run)):
            print("no final weights:", run)
            continue
        W, ck = load(weights_path(run), dev)
        assert np.array_equal(ck["K"], P["world_K"]) and np.array_equal(ck["Q"], P["world_Q"]), run
        if not sets:                                                                 # one world: the same episodes for all
            L, Z = td.world_tables(ck, P["pairs"])
            assert np.allclose(L, P["L"]) and np.allclose(Z, P["Z"], atol=1e-5), "world_tables does not give the probes"
            for k, (name, (L, Z)) in enumerate((("test", (P["L"], P["Z"])), ("train", td.world_tables(ck, pairs)))):
                envs, _, _, r = episodes(L, Z, 10, SIM_SEED + 201 + k, dev)
                sets[name] = (envs, r, [bayes(E["o"], E["L"])[1:] for E in envs])
        m = {}
        for name, (envs, r, obs) in sets.items():
            loss, loss_tr, right, shut, right_jb, right_nb = [], [], [], [], [], []
            for E, (log_jb, log_nb) in zip(envs, obs):
                cum = replay(W, E["o"].float(), E["zin"])[0].cumsum(1)
                lq = torch.log_softmax(cum.double(), -1).reshape(len(cum), T, R, R)
                lb = torch.stack([lq.logsumexp(-1), lq.logsumexp(-2)], -2)
                loss.append(fr_loss(lb, log_jb))
                loss_tr.append(fr_loss_trained(lb, log_jb))
                right.append(lb[:, -1].argmax(-1) == r)
                right_jb.append(log_jb[:, -1].argmax(-1) == r)
                right_nb.append(log_nb[:, -1].argmax(-1) == r)
                ar = torch.arange(len(r), device=dev)
                g0 = lq[ar, 0, r[:, 0], r[:, 1]] - lq[ar, 0, r[:, 0], :].logsumexp(-1)   # step 1: B(r1* | r0*)
                g1 = lq[ar, 0, r[:, 0], r[:, 1]] - lq[ar, 0, :, r[:, 1]].logsumexp(-1)   # step 1: B(r0* | r1*)
                shut.append(torch.stack([g0, g1], -1) < np.log(.01))
            right, shut, right_jb, right_nb = (torch.cat(x).double() for x in (right, shut, right_jb, right_nb))
            mean = lambda x, w: float((x * w).sum() / w.sum().clamp_min(1))
            k_shut = shut.sum(-1, keepdim=True)                                      # closed gates of the true cell: 0, 1, 2
            by_k = lambda x: [mean(x, (k_shut == k).double().expand_as(x)) for k in range(3)]
            m[name] = dict(loss=float(np.mean([a for a, _ in loss])), sym_kl=float(np.mean([b for _, b in loss])),
                           loss_trained=float(np.mean(loss_tr)), acc=float(right.mean()), acc_joint_bayes=float(right_jb.mean()),
                           acc_naive_bayes=float(right_nb.mean()), closed=float(shut.mean()),
                           acc_closed=mean(right, shut), acc_open=mean(right, 1 - shut),
                           joint_bayes_closed=mean(right_jb, shut), joint_bayes_open=mean(right_jb, 1 - shut),
                           naive_bayes_closed=mean(right_nb, shut), naive_bayes_open=mean(right_nb, 1 - shut),
                           share_by_closed=[float((k_shut == k).double().mean()) for k in range(3)],
                           acc_by_closed=by_k(right), joint_bayes_by_closed=by_k(right_jb),
                           naive_bayes_by_closed=by_k(right_nb))
            print(f"split {run:24s} {name:5s}", {k: np.round(v, 3).tolist() for k, v in m[name].items()}, flush=True)
        out[run] = m
        del W
        torch.cuda.empty_cache()
    return out


def main(only_extra=False, names=(), only_course=False, only_split=False):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    res_path, env_path = os.path.join(OUT, "results.json"), os.path.join(OUT, "per_env.npz")
    P = dict(np.load(os.path.join(TD, "analysis", "probes.npz")))
    n_env = len(P["L"])
    if only_extra or only_course or only_split:                                      # add to a finished run
        with open(res_path) as f:
            res = json.load(f)
        per_env = dict(np.load(env_path))
    else:
        res, per_env = {"config": dict(n_env=n_env, n_per_cell=N_PER_CELL, T=T, sim_seed=SIM_SEED,
                                       weights="final_net.pth")}, {}

    def collect(name, rows):
        keys = [k for k in rows[0] if not k.endswith(ARRAYS)]
        res[name] = {k: summary([x[k] for x in rows]) for k in keys}
        per_env[name] = np.array([[x[k] for k in keys] for x in rows])
        per_env[name + "__keys"] = np.array(keys)
        for k in rows[0]:
            if k.endswith(ARRAYS):
                per_env[f"{name}__{k}"] = np.mean([x[k] for x in rows], 0)

    def save():
        with open(res_path, "w") as f:
            json.dump(res, f, indent=1)
        np.savez(env_path, **per_env)

    def show(name, t0):
        print(f"{name:26s} N {res[name]['width']:5d} ({time.time() - t0:4.0f} s)",
              {a: round(v["mean"], 4) for a, v in res[name].items() if isinstance(v, dict)},
              {a: v for a, v in res[name].items() if not isinstance(v, dict) and a != "width"}, flush=True)

    if only_course or only_split:
        if only_course:
            res["C_course"], freqs = course(dev, P)
            per_env.update(freqs)
        else:
            res["E_split"] = split(dev, P)
        save()
        print("saved:", res_path)
        return
    if not only_extra:
        # ── A
        cond = np.array([two_step_condition(P["L"][e]) for e in range(n_env)])       # (env, goal, 4)
        res["A_two_step"] = {k: dict(min=float(cond[..., i].min()), median=float(np.median(cond[..., i])),
                                     max=float(cond[..., i].max()))
                             for i, k in enumerate(("spread_max_nats", "spread_rms_nats", "share_min_over_pairs",
                                                    "share_median_over_pairs"))}
        res["A_two_step"]["cases"] = int(cond[..., 0].size)
        per_env["A_two_step"] = cond
        print("A  t = 2 condition over", cond[..., 0].size, "cases:", json.dumps(res["A_two_step"]), flush=True)

        # ── D: identities and replay check
        res["D_identities"] = identities(dev)
        print("D  identities:", json.dumps(res["D_identities"]), flush=True)
        obs_p = torch.as_tensor(P["obs"], dtype=torch.float32, device=dev)
        zin_p = torch.as_tensor(P["Z"][P["env"]], device=dev).reshape(len(obs_p), -1)
        res["D_replay"] = {}
        for run in REPLAY_CHECK:
            ref = os.path.join(TD, "analysis", run, "ck_49995.npz")
            if not os.path.exists(ref):
                continue
            W, _ = load(weights_path(run, 49995), dev)
            lb = log_beliefs(replay(W, obs_p, zin_p)[0].cumsum(1)).cpu().numpy()
            stored = np.load(ref)["logq"].astype(np.float64)
            ok = np.isfinite(stored)
            err = np.abs(lb - stored)[ok]
            tol = np.abs(stored[ok]) * 2 ** -10 + 1e-3                               # float16 storage
            res["D_replay"][run] = dict(max_abs=float(err.max()), share_beyond_fp16=float((err > tol).mean()),
                                        argmax_agree=float((lb.argmax(-1) == stored.argmax(-1)).mean()))
            print(f"D  replay {run:16s}", json.dumps(res["D_replay"][run]), flush=True)
            del W

        # ── observers and the networks with final weights in the probes' world
        envs, fit, test, r = episodes(P["L"], P["Z"], N_PER_CELL, SIM_SEED, dev)
        rows_b, rows_n = observers(envs, fit, test, r)
        collect("Joint Bayes", rows_b)
        collect("Naive Bayes", rows_n)
        for k in ("Joint Bayes", "Naive Bayes"):
            print(k, {a: round(v["mean"], 4) for a, v in res[k].items()}, flush=True)
        for run in MARGINAL + JOINT + JOINT_JL:
            if not os.path.exists(weights_path(run)):
                print("no final weights:", run)
                continue
            t0 = time.time()
            W, ck = load(weights_path(run), dev)
            assert np.array_equal(ck["K"], P["world_K"]) and np.array_equal(ck["Q"], P["world_Q"]), run
            collect(run, network(W, envs, fit, test, r))
            res[run]["width"] = int(W["W_hh"].shape[1])
            show(run, t0)
            del W
            torch.cuda.empty_cache()
            save()
        del envs

    # ── the figure's other ESNs
    for k, (name, paths, n_cell) in enumerate(EXTRA):
        if names and name not in names:
            continue
        path = next((p for p in paths if os.path.exists(p)), None)
        if path is None:
            print("no weights:", name)
            continue
        t0 = time.time()
        W, ck = load(path, dev)
        same_world = bool(np.array_equal(ck["K"], P["world_K"]) and np.array_equal(ck["Q"], P["world_Q"]))
        if same_world:
            L, Z = P["L"], P["Z"]
        else:
            sys.path.insert(0, os.path.join(ROOT, "main", "REPLICATION"))
            import td_analysis as td
            L, Z = td.world_tables(ck, P["pairs"])
        envs, fit, test, r = episodes(L, Z, n_cell, SIM_SEED + 1 + k, dev)
        collect(name, network(W, envs, fit, test, r))
        res[name].update(width=int(W["W_hh"].shape[1]), weights=os.path.relpath(path, ROOT), n_per_cell=n_cell,
                         probes_world=same_world)
        if "episode" in ck:
            res[name]["episode"] = int(ck["episode"])
        if not same_world:                                                           # its own world's observers and record
            rows_b, rows_n = observers(envs, fit, test, r)
            res[name].update(joint_bayes_acc=float(np.mean([x["acc"] for x in rows_b])),
                             naive_bayes_acc=float(np.mean([x["acc"] for x in rows_n])),
                             joint_bayes_counts=float(np.mean([x["counts"] for x in rows_b])),
                             recorded_test_acc=float(np.asarray(ck["test_acc_through_training"])[-1, -1]))
        show(name, t0)
        del W, envs
        torch.cuda.empty_cache()
        save()
    print("saved:", res_path)


if __name__ == "__main__":
    main(only_extra=sys.argv[1:2] == ["extra"], names=tuple(sys.argv[2:]), only_course=sys.argv[1:] == ["course"],
         only_split=sys.argv[1:] == ["split"])
