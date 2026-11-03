"""Entanglement and the learning signal of the SANITY ctx-2 LSTM (trained vs echo reservoir).

Figures (raw; bands of 1% and 10% of the variance):
  dynamics_interference_<pct>pct_xcorr.png   cross-correlation with entanglement (plot(pct))
    row 1    the sensory input o_t (5 binary features): its novelty against the running feature counts and the number
             of features that changed since the last step (one curve each, the mean of the two networks' episode
             sets); then trained vs echo, the activity u_t (the LSTM output h_t): its norm, its effective dimension,
             and its turn 1 - cos(u_t, u_{t-1})
    row 2    trained vs echo, the weight update: the credit coherence ||sum_k delta_k||^2 / sum_k ||delta_k||^2 (how much
             the credit pieces from different lags add up rather than cancel), the energy ||dW||^2, its cross-band share
             (||dW||^2 - sum_B ||dW_B||^2) / ||dW||^2, and dW's effective dimension; rows 1-2 share one y-scale
    below    Trained | Echo: each band's share of its step's squared norm (the paper's head / tail convention) - u_t's
             energy by band and the cross-band terms by band - as heatmaps and curves
  dW_schematic.png                           how one error's dW is computed (plot_schematic)

The error at step t, e_t = dL_t/du_t (the step-t term of SANITY_loss; past readouts held fixed, so the error enters at
u_t and flows back only through the recurrence), is sent back k = 0..KMAX steps.  At step s = t - k it reaches the LSTM's
gate pre-activations as delta_k (4H; gates i, f, g, o), and the layer's weights W = [W_ih | W_hh] receive
    dW_t = sum_k dW_k,   dW_k = delta_k [x_{t-k}; u_{t-k-1}]^T.
The figures use the update from BOTH contexts' losses (the goal context's and the other's, as if it had been the goal;
SANITY draws the goal at random, so this is the expected update over the goal draw).  Backprop is linear in the error,
so splitting e_t along the PCs of u (one fixed PCA basis) splits every delta_k and dW_k: dW_B is the part due to the
error's component in band B's PCs, and ||dW||^2 = sum_B <dW_B, dW> = sum_B ||dW_B||^2 + the cross-band terms
sum_B <dW_B, dW - dW_B>.  ||dW||^2 is also the sum of dW's squared singular values sigma_i^2 (rank <= 16), whose
participation ratio (sum sigma^2)^2 / sum sigma^4 is its effective dimension.  KMAX = 15 and errors at steps 16..29:
every error has the same 16 lags.  For the echo reservoir dW is the gradient its frozen LSTM would receive.

Statistic: each episode's series and its entanglement over the error steps 16..29, each z-scored within the episode,
the product a[t + tau] * b[t] averaged within and then over episodes at lags tau = -8..8 (tau > 0: the quantity comes
after the entangled step).  Nothing is normalized per lag.  The band quantities are shares of each step's squared norm,
each band's share its own series (figure_series).

    python gradflow_analysis.py          saves and shows the figures (harvest and add-on caches first if missing; GPU 0)
    python gradflow_analysis.py check    hand-written backward vs autograd, one small batch
Set LOAD_HARVEST = False to re-collect (about 7 minutes per network on one GPU, 3.5 GB each).
"""
import os
import sys
import gc
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

def _repo_root():
    """the folder holding main/CognitiveGridworld.py, above this file; in an interactive window or notebook __file__ can
    be missing or a placeholder that does not exist (VS Code's interactive window gave "/undefined"), so the notebook's
    own file and then the working directory are tried"""
    starts = [globals().get("__file__"), globals().get("__vsc_ipynb_file__"), os.getcwd()]
    for start in starts:
        d = os.path.abspath(start) if start and os.path.exists(start) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return d
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError(f"project root (the folder holding main/CognitiveGridworld.py) not found above any of {starts}")


_root = _repo_root()
sys.path.insert(0, os.path.join(_root, "main", "REPLICATION"))
import supplementary_plotter2 as sp
from main.CognitiveGridworld import CognitiveGridworld
from main.utils import fig_path

LOAD_HARVEST = True
AGENTS = {"trained": ("/sanity/fully_trained_ctx_2_e5", False), "echo": ("/sanity/reservoir_ctx_2_e5", True)}
CFG = dict(mode="SANITY", cuda=0, episodes=50, checkpoint_every=1, realization_num=10, hid_dim=1000, obs_num=5,
           show_plots=False, batch_num=1000, step_num=30, state_num=500, learn_embeddings=False, ctx_num=2,
           training=False)
KMAX = 15
SRC = np.arange(KMAX + 1, 30)                     # error steps: each has lags 0..KMAX behind it
N_SEG = 20                                        # segments: cut points at every 1/N_SEG (5%) of the variance
CACHE = os.path.join(sp._ROOT, "main", "DATA", "cache", "gradflow_{}.npz")
BASIS = os.path.join(sp._ROOT, "main", "DATA", "cache", "gradflow_basis_{}.npz")   # the fixed PCA basis
BAND1_EP = 20000                                  # episodes the 1% add-on replays (the first BAND1_EP of the harvest)
OBS = os.path.join(sp._ROOT, "main", "DATA", "experimental", "gradflow_obs", "series_{}.npz")   # observation statistics
CHUNK = 100                                       # episodes per backward chunk (N_SEG probes each; GPU memory)
TAU = np.arange(sp.DYN_PARAMS["E_START"], sp.DYN_PARAMS["E_END"] + 1)

# exact fp32: TF32 would make the hand-written cell and cuDNN disagree at ~1e-3
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


# ═══════════════════════════════════════════════════════════════════
# The LSTM, forward and backward, written out (gate order i, f, g, o as nn.LSTM)
# ═══════════════════════════════════════════════════════════════════

def lstm_weights(m):
    g = lambda n: getattr(m.LSTM, n).detach().float()
    return g("weight_ih_l0"), g("weight_hh_l0"), g("bias_ih_l0") + g("bias_hh_l0")


def lstm_input(m):
    """the LSTM's input x_s (= RNN_input in model_forward)"""
    B, T = m.batch_num, m.step_num
    Z = m.active_Z.detach().float().reshape(B, 1, -1).expand(-1, T, -1)
    return torch.relu(m.classifier_readin(torch.cat((m.obs_flat.float(), Z), -1)))


def forward_states(m, x):
    """the gates, the previous cell state cp, tanh(c) and u at every step; u_prev[s] = u_{s-1} (the initial state at s=0)"""
    W_ih, W_hh, b = lstm_weights(m)
    B, T, H = x.shape
    u = m.STM.detach().float().reshape(1, H).expand(B, H)
    c = m.LTM.detach().float().reshape(1, H).expand(B, H)
    xa = x @ W_ih.T + b
    st = {k: torch.empty(B, T, H, device=x.device) for k in ("i", "f", "g", "o", "cp", "tc", "u", "u_prev")}
    for s in range(T):
        st["u_prev"][:, s], st["cp"][:, s] = u, c
        i, f, g, o = (xa[:, s] + u @ W_hh.T).chunk(4, -1)
        i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
        c = f * c + i * g
        tc = c.tanh()
        u = o * tc
        for k, v in (("i", i), ("f", f), ("g", g), ("o", o), ("tc", tc), ("u", u)):
            st[k][:, s] = v
    return st, W_hh


def backward(st, W_hh, du_t, t, kmax):
    """yield (k, delta_k) for k = 0..min(kmax, t): the gate pre-activation gradient at step t - k of the error du_t
    (any leading shape (..., H)), carried back through the cell (c) and the recurrence (W_hh)"""
    du, dc = du_t, torch.zeros_like(du_t)
    for k in range(min(kmax, t) + 1):
        s = t - k
        i, f, g, o, cp, tc = (st[n][:, s].reshape((-1,) + (1,) * (du.dim() - 2) + (du.shape[-1],))
                              for n in ("i", "f", "g", "o", "cp", "tc"))
        dct = dc + du * o * (1 - tc * tc)                                   # total credit on c_s
        delta = torch.cat([dct * g * i * (1 - i), dct * cp * f * (1 - f), dct * i * (1 - g * g),
                           du * tc * o * (1 - o)], -1)
        yield k, delta
        dc, du = dct * f, delta @ W_hh


class StepLoss:
    """Episode b's step-t term of SANITY_loss (float64 head, same detached global minimum as the loss)."""

    def __init__(self, m, r):
        self.m, (self.B, self.T, _) = m, r.shape
        self.C, self.R = m.ctx_num, m.realization_num
        self.ar = torch.arange(self.B, device=r.device)
        self.goal, self.P = m.goal_ind.long(), m.joint_goal_belief.double()
        with torch.no_grad():
            q = r.double().cumsum(1).reshape(self.B, self.T, self.C, self.R).softmax(-1)[self.ar, :, self.goal]
            self.dkl_min = m.DKL_sym(q, self.P, PM=False).min()

    def __call__(self, logits_t, t):
        q = logits_t.double().reshape(self.B, self.C, self.R).softmax(-1)[self.ar, self.goal]
        return ((self.m.DKL_sym(q, self.P[:, t], PM=False) - self.dkl_min + 1e-8) ** .5).sum(-1)


def step_dlg(m, loss, r, t):
    """dL_t/dlogits_t, past readouts held fixed (logits accumulate over steps)"""
    lg = (r[:, t] + r[:, :t].detach().sum(1)).detach().requires_grad_()
    with torch.enable_grad():
        dlg, = torch.autograd.grad(loss(lg, t).sum(), lg)
    return dlg


def step_error(m, loss, r, t):
    """e_t = dL_t/du_t through the readout at t, past readouts held fixed"""
    return step_dlg(m, loss, r, t) @ m.classifier_readout.weight.detach().float()


# ═══════════════════════════════════════════════════════════════════
# Harvest
# ═══════════════════════════════════════════════════════════════════

def cut_segments(evr, n=N_SEG):
    """segment of every PC: segment j ends at the PC whose cumulative variance is nearest (j + 1) / n.  This is
    equal_variance_bands' rule without its forcing, so a PC holding more than 1/n leaves the following segments empty.
    A top / bottom split at any multiple of 1/n is a run of segments, and equal_variance_bands(evr, P) for P dividing n
    gives unions of segments whenever its forcing does not act."""
    cum, sid, start = np.cumsum(evr), np.full(len(evr), n - 1), 0
    for j in range(n - 1):
        end = int(np.argmin(np.abs(cum - (j + 1) / n)))
        sid[start:end + 1] = j
        start = max(start, end + 1)
    return sid


def seg_grams(st, W_hh, Zk, e, t, V, OH):
    """the step-t error split by segment (V: PCs as columns, OH: (H, S) segment one-hot), sent back KMAX lags, for one
    chunk of episodes -> <dW_s, dW_s'> (lags summed) and sum_k <dW_{s,k}, dW_{s',k}> (same lag), each (c, S, S), for
    the whole W = [W_ih | W_hh] and then for W_hh alone (its input is u_{s-1}, the last H entries of z); then the whole
    error's lag Grams delta_k . delta_l, z_k . z_l and u_{t-k-1} . u_{t-l-1}, each (c, K, K): dW = D Z^T with D, Z the
    lags' deltas and inputs as columns, so dW's squared singular values are the eigenvalues of (D^T D)(Z^T Z)"""
    probes = ((e @ V)[:, None, :] * OH.T[None]) @ V.T                                 # (c, S, H): e's part per segment
    D = torch.stack([d for _, d in backward(st, W_hh, probes, t, KMAX)], 1)          # (c, K, S, 4H)
    c, K, S = D.shape[:3]
    Df = D.reshape(c, K * S, -1)
    M = torch.bmm(Df, Df.transpose(1, 2)).reshape(c, K, S, K, S)
    zk = Zk[:, t - torch.arange(K, device=D.device)]                                 # (c, K, X + H): [x_s; u_{s-1}]
    zz = [torch.einsum("bkd,bld->bkl", z, z) for z in (zk, zk[..., -W_hh.shape[1]:])]
    out = []
    for g in zz:
        P = M * g[:, :, None, :, None]                                               # <dW_{s,k}, dW_{s',l}>
        out += [P.sum((1, 3)), torch.diagonal(P, dim1=1, dim2=3).sum(-1)]
    return out + [M.sum((2, 4))] + zz


class BasisLogger:
    """external_logger for the basis: per test batch, n, sum u_t and sum u_t u_t^T over its episodes and steps (u from
    the fp32 forward the harvest uses)"""

    def __call__(self, env):
        if not env.test_set:
            return {}
        m = env.model
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            st, _ = forward_states(m, lstm_input(m))
            u = st["u"].reshape(-1, m.hid_dim).double()
            return {"n": np.array([len(u)]), "s": u.sum(0).cpu().numpy()[None], "ss": (u.T @ u).cpu().numpy()[None]}


def basis(net):
    """the fixed PCA basis of u_t (mean, PCs, explained variance, segments): one PCA of a separate run of test episodes,
    so the bands are the same subspaces in every harvested episode and not fitted to them.  Also the PCAs of the
    run's two interleaved halves, for each band's stability (overlap of the halves' band subspaces)."""
    path = BASIS.format(net)
    if os.path.exists(path):
        return dict(np.load(path))
    load_env, reservoir = AGENTS[net]
    agent = CognitiveGridworld(**CFG, reservoir=reservoir, load_env=load_env, external_logger=BasisLogger())
    L = {k: np.concatenate(v) if k != "n" else np.concatenate(v).ravel() for k, v in agent.custom_log.items()}
    del agent
    gc.collect(); torch.cuda.empty_cache()

    def pca(idx):
        n, s, ss = L["n"][idx].sum(), L["s"][idx].sum(0), L["ss"][idx].sum(0)
        mu = s / n
        ev, V = np.linalg.eigh(ss / n - np.outer(mu, mu))
        ev, V = ev[::-1].clip(0), V[:, ::-1]
        return mu, ev / ev.sum(), V

    idx = np.arange(len(L["n"]))
    mu, evr, V = pca(idx)
    halves = [pca(idx[i::2]) for i in (0, 1)]
    D = {"mu": mu.astype(np.float32), "V": V.astype(np.float32), "evr": evr, "sid": cut_segments(evr),
         "V_halves": np.stack([h[2] for h in halves]).astype(np.float32), "evr_halves": np.stack([h[1] for h in halves]),
         "episodes": np.array(int(L["n"].sum()) // CFG["step_num"])}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **D)
    print(f"basis saved: {path}")
    return D


def band_stability(Bs, bands):
    """for each band (a set of segments): the overlap of its subspace in the two half-PCAs (1 = the same subspace)"""
    sids = [cut_segments(e) for e in Bs["evr_halves"]]
    out = []
    for B in bands:
        a, b = (V[:, np.isin(sid, B)] for V, sid in zip(Bs["V_halves"], sids))
        out.append(np.nan if 0 in (a.shape[1], b.shape[1]) else ((a.T @ b) ** 2).sum() / max(a.shape[1], b.shape[1]))
    return np.array(out)


EP_MODEL = ("obs_flat", "active_Z", "goal_ind", "goal_value", "joint_goal_belief")
EP_ENV = {"naive_px": np.float32, "joint_px": np.float32, "naive_belief": np.float64, "joint_belief": np.float64,
          "naive_acc": np.float32, "joint_acc": np.float32, "naive_TP": np.float32, "joint_TP": np.float32,
          "naive_mse": np.float32, "joint_mse": np.float32, "ctx_inds": None, "ctx_vals": None,
          "joint_likelihood": np.float64, "naive_likelihood": np.float64}


class Harvester:
    """external_logger: per test episode, on the fixed basis (basis()): the activity metrics and u_t's energy per
    segment, every step's logit gradient, error size and error energy per segment, and each error's segment Grams for W
    and for W_hh alone and its lag Grams (lag_dd, lag_zz, lag_zz_hh).  Any split at a multiple of 5% follows offline (G_all summed over the band's rows: its share
    <dW_B, dW>; G_same summed over its block: its self terms; G_all over its block: ||dW_B||^2).  Also the episode
    record (ep_*: model inputs, loss target, likelihoods, beliefs, observer performance, contexts) and each batch's
    loss minimum, so later probes can rerun on the same episodes.  Row n is episode n % 1000 of test batch n // 1000;
    a replay needs the whole batch (StepLoss's minimum, dkl_min, is over the batch)."""

    def __init__(self, Bs):
        self.mu, self.V, self.sid = Bs["mu"], Bs["V"], Bs["sid"]

    def __call__(self, env):
        if not env.test_set:
            return {}
        m = env.model
        assert m.mode == "SANITY" and m.readout_depth == 1
        B, T, H = m.batch_num, m.step_num, m.hid_dim
        rev = sp.step_dkl(env.naive_px / env.naive_px.sum(-1, keepdims=True),
                          sp.approx_lik(env.joint_belief.astype(np.float64))).mean(-1)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            x = lstm_input(m)
            st, W_hh = forward_states(m, x)
            dev = x.device
            V, mu = torch.as_tensor(self.V, device=dev), torch.as_tensor(self.mu, device=dev)
            OH = torch.as_tensor(np.eye(N_SEG, dtype=np.float32)[self.sid], device=dev)
            u = st["u"]                                                                  # the fp32 forward's u_t
            Ep = ((u.reshape(B * T, H) - mu) @ V) ** 2
            out = {"l2": u.norm(dim=-1), "pr": (Ep.sum(-1) ** 2 / (Ep ** 2).sum(-1).clamp_min(1e-12)).reshape(B, T),
                   "seg_E": (Ep @ OH).reshape(B, T, N_SEG)}                         # sum of proj_pc^2 per segment
            r = m.classifier_readout(st["u"])
            loss = StepLoss(m, r)
            Zk = torch.cat([x, st["u_prev"]], -1)
            dlg = torch.stack([step_dlg(m, loss, r, t) for t in range(T)], 1)            # (B, T, C*R)
            e = dlg @ m.classifier_readout.weight.detach().float()                       # (B, T, H): e_t, every step
            out.update({"dlg": dlg, "err": e.norm(dim=-1), "seg_err": ((e @ V) ** 2) @ OH})
            G = [torch.zeros(B, len(SRC), N_SEG, N_SEG, device=dev) for _ in range(4)] + \
                [torch.zeros(B, len(SRC), KMAX + 1, KMAX + 1, device=dev) for _ in range(3)]
            for j, t in enumerate(SRC):
                for c0 in range(0, B, CHUNK):
                    sl = slice(c0, min(B, c0 + CHUNK))
                    for g, v in zip(G, seg_grams({k: v[sl] for k, v in st.items()}, W_hh, Zk[sl], e[sl, t], t, V, OH)):
                        g[sl, j] = v
        out.update(dict(zip(("G_all", "G_same", "G_all_hh", "G_same_hh", "lag_dd", "lag_zz", "lag_zz_hh"), G)))
        out = {k: v.cpu().numpy().astype(np.float32) for k, v in out.items()}
        out.update({"rev": rev.astype(np.float32), "dkl_min": np.array([float(loss.dkl_min)])})
        out.update({f"ep_{k}": getattr(m, k).detach().cpu().numpy() for k in EP_MODEL})
        for k, dt in EP_ENV.items():
            v = getattr(env, k, None)
            if v is not None:
                v = np.asarray(v.cpu() if torch.is_tensor(v) else v)
                out[f"ep_{k}"] = v.astype(dt) if dt else v
        torch.cuda.empty_cache()
        return out


def harvest(net):
    """one network's harvest on its fixed basis, from the cache or collected (the basis first, if it is not cached)"""
    path = CACHE.format(net)
    keys = {"l2", "pr", "seg_E", "dlg", "err", "seg_err", "G_all", "G_same", "G_all_hh", "G_same_hh", "lag_dd", "lag_zz",
            "lag_zz_hh", "rev", "dkl_min",
            "basis_V", "basis_sid", "ep_obs_flat", "ep_active_Z", "ep_goal_ind", "ep_joint_goal_belief", "ep_naive_px",
            "ep_joint_px", "ep_ctx_inds", "ep_ctx_vals"}
    if LOAD_HARVEST and os.path.exists(path):
        D = dict(np.load(path))
        missing = keys - set(D)
        if not missing:
            print(f"harvest loaded: {path}")
            return D
        print(f"harvest stale (missing {sorted(missing)}), re-collecting")
    Bs = basis(net)
    load_env, reservoir = AGENTS[net]
    agent = CognitiveGridworld(**CFG, reservoir=reservoir, load_env=load_env, external_logger=Harvester(Bs))
    D = {k: v.reshape((-1,) + v.shape[2:]) for k, v in agent.custom_log.items()}   # (episodes, batch, ...) -> (N, ...)
    del agent
    gc.collect(); torch.cuda.empty_cache()
    D.update({f"basis_{k}": v for k, v in Bs.items()})
    D.update({"meta_src": SRC, "meta_kmax": np.array(KMAX), "meta_n_seg": np.array(N_SEG),
              "meta_batch": np.array(CFG["batch_num"]), "meta_agent": np.array(load_env)})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **D)
    print(f"harvest saved: {path}")
    return D


def replay(net, fn, D=None, batches=None):
    """rerun one network's harvested episodes through its saved model, batch by batch, from the episode record, and
    collect fn(m, x, st, W_hh, r, loss) (a dict of arrays, one row per episode of the batch).  Each batch is first
    checked against the harvest: its loss minimum and every step's error size must match.  batches: the first this
    many batches only (default all)."""
    D = harvest(net) if D is None else D
    load_env, reservoir = AGENTS[net]
    agent = CognitiveGridworld(**dict(CFG, episodes=1), reservoir=reservoir, load_env=load_env,
                               external_logger=lambda env: {})
    m, (B, T) = agent.model, (CFG["batch_num"], CFG["step_num"])
    dev, out = m.STM.device, {}
    for b in range(len(D["l2"]) // B if batches is None else batches):
        rows = slice(b * B, (b + 1) * B)
        m.obs_flat, m.active_Z, m.goal_ind, m.joint_goal_belief = (
            torch.as_tensor(D[f"ep_{k}"][rows], device=dev) for k in ("obs_flat", "active_Z", "goal_ind",
                                                                      "joint_goal_belief"))
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            x = lstm_input(m)
            st, W_hh = forward_states(m, x)
            r = m.classifier_readout(st["u"])
            loss = StepLoss(m, r)
            err = torch.stack([step_error(m, loss, r, t).norm(dim=-1) for t in range(T)], 1).cpu().numpy()
            assert np.isclose(float(loss.dkl_min), D["dkl_min"][b], rtol=1e-5), f"batch {b}: loss minimum differs"
            assert np.allclose(err, D["err"][rows], rtol=1e-3, atol=1e-6), f"batch {b}: errors differ"
            for k, v in fn(m, x, st, W_hh, r, loss).items():
                out.setdefault(k, []).append(v)
    del agent
    gc.collect(); torch.cuda.empty_cache()
    return {k: np.concatenate(v) for k, v in out.items()}


BAND1_BOTH = os.path.join(sp._ROOT, "main", "DATA", "experimental", "gradflow_ctx", "band1_both_{}.npz")


def band1_both(net, episodes=BAND1_EP):
    """add-on (experimental): own_quantities() for bands of 1% of u's variance, for the update from BOTH contexts' losses (error = goal context's + other
    context's; SANITY draws the goal at random, so this is the expected update over the goal draw), checked per batch
    against context_grams"""
    return own_quantities(net, lambda D: cut_segments(D["basis_evr"], 100), episodes, BAND1_BOTH.format(net),
                          "1% both contexts", error="both")


BANDP_BOTH = os.path.join(sp._ROOT, "main", "DATA", "experimental", "gradflow_ctx", "band{}_both_{}.npz")


def band_both(net, pct=1, episodes=BAND1_EP):
    """band1_both() for bands of pct% of u's variance (cut_segments at 100 // pct: unions of the 1% bands)"""
    if pct == 1:
        return band1_both(net, episodes)
    return own_quantities(net, lambda D: cut_segments(D["basis_evr"], 100 // pct), episodes, BANDP_BOTH.format(pct, net),
                          f"{pct}% both contexts", error="both")


TURNS = os.path.join(sp._ROOT, "main", "DATA", "experimental", "gradflow_ctx", "turns_{}.npz")


def activity_turns(net, episodes=BAND1_EP):
    """add-on: cos(u_s, u_{s-1}) at every step s (s = 0 against the initial state) from a forward replay of the first
    episodes -> cos_u (N, 30).  The harvest's input Grams hold it only up to step 28 (the update at s + 1 has
    u_s . u_{s-1}); checked against them."""
    path = TURNS.format(net)
    if os.path.exists(path):
        return dict(np.load(path))
    D = harvest(net)

    def fn(m, x, st, W_hh, r, loss):
        u, up = st["u"].double(), st["u_prev"].double()
        return {"cos_u": ((u * up).sum(-1) / (u.norm(dim=-1) * up.norm(dim=-1))).cpu().numpy()}

    out = replay(net, fn, D, batches=episodes // CFG["batch_num"])
    ZU = D["lag_zz_hh"][:episodes].astype(np.float64)
    ref = ZU[:, 1:, 0, 1] / np.sqrt(ZU[:, 1:, 0, 0] * ZU[:, 1:, 1, 1])                # s = SRC[0] .. SRC[-1] - 1
    assert np.abs(out["cos_u"][:, SRC[0]:SRC[-1]] - ref).max() < 1e-4, "replayed turns do not match the harvest"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    print(f"activity turns saved: {path}")
    return out


def other_context_loss(m, r, joint_belief):
    """StepLoss for the context that is not the goal, as if it had been the goal (joint_belief: (B, T, 2, R) tensor)"""
    from types import SimpleNamespace
    other = 1 - m.goal_ind.long()
    mo = SimpleNamespace(ctx_num=m.ctx_num, realization_num=m.realization_num, goal_ind=other,
                         joint_goal_belief=joint_belief[torch.arange(len(other), device=other.device), :, other],
                         DKL_sym=m.DKL_sym)
    return StepLoss(mo, r)


CTX_GRAMS = os.path.join(sp._ROOT, "main", "DATA", "experimental", "gradflow_ctx", "ctx_grams_{}.npz")


def context_grams(net, episodes=BAND1_EP):
    """add-on (experimental), from replay(): the step-t error of BOTH contexts' losses (the goal context the harvest used
    and the other one, as if it had been the goal; SANITY draws the goal at random), each sent back KMAX lags, and per
    error their lag Grams "DD" (N, errors, 3, K, K): goal x goal, other x other, goal x other (delta_k . delta_l).  With the
    harvest's input Grams lag_zz / lag_zz_hh this gives any combination exactly, since dW is linear in the error: e.g.
    the both-context update dW_goal + dW_other.  Also each error's size "err_other".  Every batch: the goal x goal Gram
    must equal the harvest's lag_dd."""
    from types import SimpleNamespace
    path = CTX_GRAMS.format(net)
    if os.path.exists(path):
        return dict(np.load(path))
    D = harvest(net)
    batch = [0]

    def fn(m, x, st, W_hh, r, loss):
        B = x.shape[0]
        rows = slice(batch[0] * B, (batch[0] + 1) * B)
        other = 1 - m.goal_ind.long()
        jb = torch.as_tensor(D["ep_joint_belief"][rows], device=x.device)
        mo = SimpleNamespace(ctx_num=m.ctx_num, realization_num=m.realization_num, goal_ind=other,
                             joint_goal_belief=jb[torch.arange(B, device=x.device), :, other], DKL_sym=m.DKL_sym)
        loss_o = StepLoss(mo, r)                                                    # the other context's loss
        DD = torch.zeros(B, len(SRC), 3, KMAX + 1, KMAX + 1, device=x.device)
        err_o = torch.zeros(B, len(SRC), device=x.device)
        for j, t in enumerate(SRC):
            e_g, e_o = step_error(m, loss, r, t), step_error(m, loss_o, r, t)
            err_o[:, j] = e_o.norm(dim=-1)
            Dk = torch.stack([d for _, d in backward(st, W_hh, torch.stack([e_g, e_o], 1), t, KMAX)], 1)   # (B, K, 2, 4H)
            G = torch.einsum("bkag,blcg->backl", Dk, Dk)                              # (B, 2, 2, K, K)
            DD[:, j] = torch.stack([G[:, 0, 0], G[:, 1, 1], G[:, 0, 1]], 1)
        ref = D["lag_dd"][rows].astype(np.float64)
        got = DD[:, :, 0].cpu().numpy().astype(np.float64)
        assert np.allclose(got, ref, rtol=2e-3, atol=1e-3 * np.abs(ref).mean()), \
            f"batch {batch[0]}: goal x goal Gram differs from the harvest: {np.abs(got - ref).max() / np.abs(ref).mean():.1e}"
        batch[0] += 1
        print(f"context Grams, {net}: batch {batch[0]} of {episodes // CFG['batch_num']}", flush=True)
        return {"DD": DD.cpu().numpy(), "err_other": err_o.cpu().numpy()}

    out = replay(net, fn, D, batches=episodes // CFG["batch_num"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    print(f"context Grams saved: {path}")
    return out


def own_quantities(net, cut, episodes, path, label, chunk=CHUNK, error="goal"):
    """add-on cache, from replay() of the first `episodes` episodes: the PCs of the fixed basis grouped into bands by
    cut(D) (the band of every PC; only the bands with PCs, ids in "band"), each band's own quantities per error, (N,
    errors, bands) each: "self" sum_k ||dW_{B,k}||^2, "same" sum_k <dW_{B,k}, dW_k>, "own" ||dW_B||^2, "share" <dW_B, dW>, the
    same four for W_hh alone ("_hh"), "dim" dW_B's participation ratio, "err_E" the error's energy in the band; and
    u_t's energy in the band at every step, "E" (N, T, bands).  The four parts of the share follow: self, same band /
    other lag = own - self, other band / same lag = same - self, other band / other lag = share - own - same + self.
    Every batch is checked against the harvest: the shares add up to ||dW||^2 and ||dW_hh||^2."""
    if os.path.exists(path):
        return dict(np.load(path))
    D = harvest(net)
    sid = cut(D)
    ids = np.unique(sid)
    dev = f"cuda:{CFG['cuda']}"
    V, mu, OH = (torch.as_tensor(a, device=dev) for a in
                 (D["basis_V"], D["basis_mu"], (sid[:, None] == ids[None]).astype(np.float32)))
    if error == "goal":
        tot, tot_hh = (D[k].sum((-1, -2)) for k in ("G_all", "G_all_hh"))
    else:                                                                               # both contexts' losses
        C = context_grams(net)["DD"].astype(np.float64)
        DDb = C[:, :, 0] + C[:, :, 1] + C[:, :, 2] + np.swapaxes(C[:, :, 2], -1, -2)
        n_c = len(DDb)
        tot = (DDb * D["lag_zz"][:n_c].astype(np.float64)).sum((-1, -2))
        tot_hh = (DDb * D["lag_zz_hh"][:n_c].astype(np.float64)).sum((-1, -2))
        del C, DDb
    batch = [0]

    def fn(m, x, st, W_hh, r, loss):
        B, T, H = st["u"].shape
        if error == "both":
            rows_b = slice(batch[0] * B, (batch[0] + 1) * B)
            loss_o = other_context_loss(m, r, torch.as_tensor(D["ep_joint_belief"][rows_b], device=x.device))
        Zk = torch.cat([x, st["u_prev"]], -1)
        res = {k: torch.zeros(B, len(SRC), len(ids), device=x.device) for k in
               ("self", "same", "own", "share", "self_hh", "same_hh", "own_hh", "share_hh", "dim", "err_E")}
        for j, t in enumerate(SRC):
            e = step_error(m, loss, r, t)
            if error == "both":
                e = e + step_error(m, loss_o, r, t)
            res["err_E"][:, j] = ((e @ V) ** 2) @ OH
            for c0 in range(0, B, chunk):
                sl = slice(c0, min(B, c0 + chunk))
                probes = ((e[sl] @ V)[:, None, :] * OH.T[None]) @ V.T                 # (c, bands, H)
                Dk = torch.stack([d for _, d in backward({k: v[sl] for k, v in st.items()}, W_hh, probes, t, KMAX)], 1)
                zk = Zk[sl][:, t - torch.arange(Dk.shape[1], device=x.device)]
                M = torch.einsum("cksg,clsg->cskl", Dk, Dk)                          # delta_B,k . delta_B,l
                Q = torch.einsum("cksg,clg->cskl", Dk, Dk.sum(2))                    # delta_B,k . delta_l
                del Dk
                for w, z in (("", zk), ("_hh", zk[..., -H:])):
                    zz = torch.einsum("ckd,cld->ckl", z, z)[:, None]                 # (c, 1, K, K)
                    dz = zz.diagonal(dim1=-2, dim2=-1)
                    res["self" + w][sl, j] = (M.diagonal(dim1=-2, dim2=-1) * dz).sum(-1)
                    res["same" + w][sl, j] = (Q.diagonal(dim1=-2, dim2=-1) * dz).sum(-1)
                    res["own" + w][sl, j] = (M * zz).sum((-1, -2))
                    res["share" + w][sl, j] = (Q * zz).sum((-1, -2))
                    if not w:
                        A = M @ zz
                        res["dim"][sl, j] = A.diagonal(dim1=-2, dim2=-1).sum(-1) ** 2 / (A * A.transpose(-1, -2)).sum((-1, -2))
        rows = slice(batch[0] * B, (batch[0] + 1) * B)
        for k, ref in (("share", tot), ("share_hh", tot_hh)):
            got = res[k].sum(-1).cpu().numpy()
            assert np.allclose(got, ref[rows], rtol=2e-3, atol=1e-3 * np.abs(ref[rows]).mean()), \
                f"batch {batch[0]}: {label} shares do not add up to the harvest's {k}: max rel err " \
                f"{np.abs(got - ref[rows]).max() / np.abs(ref[rows]).mean():.1e}"
        batch[0] += 1
        print(f"{label} add-on, {net}: batch {batch[0]} of {episodes // CFG['batch_num']}", flush=True)
        out = {k: v.cpu().numpy() for k, v in res.items()}
        out["E"] = ((((st["u"].reshape(B * T, H) - mu) @ V) ** 2) @ OH).reshape(B, T, -1).cpu().numpy()
        return out

    out = replay(net, fn, D, batches=episodes // CFG["batch_num"])
    out.update({"band": ids, "pc": np.bincount(sid)[ids], "sid": sid})
    np.savez(path, **out)
    print(f"{label} add-on saved: {path}")
    return out


def rows100(f_band, ids, nb=100):
    """per-band curves (len(ids), TAU) -> (nb, TAU) with NaN rows for the bands with no PCs"""
    M = np.full((nb, len(TAU)), np.nan)
    M[ids] = f_band
    return M


# ═══════════════════════════════════════════════════════════════════
# Statistics: the lagged products, and the sensory controls
# ═══════════════════════════════════════════════════════════════════


SENSORY_DESC = {"local": "the current and previous input and their per-feature differences",
                "inputs": "the inputs at lags 0..4 and all their per-feature differences",
                "full": "the inputs at lags 0..4, all their per-feature differences and the novelty of each input against "
                        "the running feature counts",
                "counts": "the running frequency of each feature before the step (the slow buildup, no current input)",
                "novelty": "the novelty of each input (lags 0..4) against the running feature counts"}


def sensory_set(net, n, kind):
    """(n, errors, p) sensory statistics (from the observations alone) at the error steps, plus step dummies:
    "local"  the current and previous input (pattern one-hots) and their per-feature differences
    "inputs" + the inputs at lags 2..4 and all pairwise per-feature differences among lags 0..4
    "full"   + the novelty of each input against the running feature counts (lags 0..4, linear and squared)
    "counts" only the running frequency of each feature before the step, (#1s before t + 1) / (t + 2), linear and squared
    "novelty" only the novelty term of the full set"""
    f = np.load(OBS.format(net))
    obs, pat = f["obs"][:n].astype(np.float64), f["obs_pattern"][:n].astype(int)
    s0, s1 = SRC[0], SRC[-1] + 1
    at = lambda x, L: x[:, s0 - L:s1 - L]
    steps = np.broadcast_to(np.eye(len(SRC))[None, :, 1:], (n, len(SRC), len(SRC) - 1))
    if kind == "counts":
        ones = np.concatenate([np.zeros_like(obs[:, :1]), np.cumsum(obs, 1)[:, :-1]], 1)
        freq = (ones + 1) / (np.arange(obs.shape[1])[None, :, None] + 2)            # (n, T, 5), before step t
        return np.concatenate([at(freq, 0), at(freq, 0) ** 2, steps], -1)
    if kind == "novelty":
        nov = novelty(f["obs"][:n])
        return np.concatenate([obs_design([nov]), obs_design([nov ** 2]), steps], -1)
    oh = lambda L: np.stack([at(pat, L) == v for v in range(1, 32)], -1).astype(np.float64)
    cols = [oh(0), oh(1), np.abs(at(obs, 0) - at(obs, 1))]
    if kind in ("inputs", "full"):
        cols += [at(obs, L) for L in (2, 3, 4)]
        cols += [np.abs(at(obs, a) - at(obs, b)) for a in range(5) for b in range(a + 1, 5) if (a, b) != (0, 1)]
    if kind == "full":
        nov = novelty(f["obs"][:n])
        cols += [obs_design([nov]), obs_design([nov ** 2])]
    cols.append(steps)
    return np.concatenate(cols, -1)


def _zs(a):
    """z-score within episode (axis 1), any trailing axes"""
    return (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-12)


def _lagged(a, b):
    """get_xcorr's average of a[t + tau] * b[t] for already-scaled a (N, T, ...) and b (N, T) -> (TAU, ...)"""
    b = b.reshape(b.shape + (1,) * (a.ndim - 2))
    out = []
    for lag in TAU:
        if lag < 0:
            out.append((a[:, :lag] * b[:, -lag:]).mean(1).mean(0))
        elif lag > 0:
            out.append((a[:, lag:] * b[:, :-lag]).mean(1).mean(0))
        else:
            out.append((a * b).mean(1).mean(0))
    return np.stack(out, 0)


def _fit(y, Z):
    """least-squares fit of y (N, T) on Z (N, T, p), both centred within episode, one pooled fit"""
    Zc = (Z - Z.mean(1, keepdims=True)).reshape(-1, Z.shape[-1])
    yc = (y - y.mean(1, keepdims=True)).reshape(-1)
    return (Zc @ np.linalg.lstsq(Zc, yc, rcond=None)[0]).reshape(y.shape)


def novelty(obs):
    """-sum_i log p(o_t,i | features seen before t): running per-feature frequencies with a Laplace prior (model-free);
    obs (N, T, 5) binary"""
    o = obs.astype(np.float64)
    ones = np.concatenate([np.zeros_like(o[:, :1]), np.cumsum(o, 1)[:, :-1]], 1)
    p1 = (ones + 1) / (np.arange(o.shape[1])[None, :, None] + 2)
    return -(o * np.log(p1) + (1 - o) * np.log(1 - p1)).sum(-1)


def obs_design(stats, change=None, lags=range(5)):
    """(N, errors, p) controls for series at the error steps: each statistic (N, T) at lags 0..4, plus one-hot obs-change
    levels at lags 0 and 1"""
    s0, s1 = SRC[0], SRC[-1] + 1
    cols = [st[:, s0 - L:s1 - L] for st in stats for L in lags]
    if change is not None:
        cols += [(change[:, s0 - L:s1 - L] == v).astype(np.float64) for L in (0, 1) for v in range(1, 6)]
    return np.stack(cols, -1)


# bands, top first: a ramp from dark to light in the network's hue (AGENT_COLORS' OKLCH hue at lightness .38 -> .76 for
# Trained, .46 -> .78 for Echo), and a marker per band (up for the top, down for the bottom), since neighbouring shades
# of one hue are close
BAND_RAMP = {"Trained": ("#014670", "#67b9fa"), "Echo": ("#894102", "#fd9c5d")}
MID_MARKERS = ("o", "s", "D", "p", "h", "P", "X", "8")


def band_style(lab, nb):
    """colors and markers of nb bands, top first"""
    cols = LinearSegmentedColormap.from_list("", BAND_RAMP[lab])(np.linspace(0, 1, nb))
    mks = ["^"] + [MID_MARKERS[i % len(MID_MARKERS)] for i in range(nb - 2)] + ["v"]
    return cols, mks


def panel(ax, lines, name=None, math=None, fade=.5, lw=2, past_ls=":"):
    """one panel in plot_dynamics_1x4's style; lines: (r over TAU, color, marker, x offset[, style of the post-event
    line]), dotted and faded before the event; the offset keeps curves that meet (at r = 0) visible"""
    past, fut = TAU <= 0, TAU >= 0
    for m, col, mk, dx, *opt in lines:
        bare = len(lines) > 2                                                        # many curves: lines only,
        x, mk = TAU + (0 if bare else dx), None if bare else mk                      # no markers, no offsets
        ax.plot(x[past], m[past], c=col, alpha=fade, lw=lw, ls=past_ls, marker=mk, mec="white", mew=1, ms=4, zorder=3)
        ax.plot(x[fut], m[fut], **dict(dict(c=col, lw=lw, marker=mk, mec="k", mew=1, ms=5, zorder=4), **(opt[0] if opt else {})))
    if name:
        ax.set_title(name, fontsize=sp.FS["title"] * .85, pad=sp.FS["title"] * 2.9)
        ax.text(.5, 1.015, math, transform=ax.transAxes, ha="center", va="bottom", fontsize=sp.FS["title"] * .70)
    ax.set_xticks(range(TAU[0], TAU[-1] + 1, 4))
    ax.tick_params(labelleft=False)
    ax.axhline(0, color="black", alpha=.3, lw=1.2, zorder=1)
    ax.axvline(0, color="black", ls="--", alpha=.4, lw=1.5, zorder=1)
    ax.grid(True, linestyle="--", alpha=.2, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


HEAT_CMAP = plt.get_cmap("RdBu_r").copy()
HEAT_CMAP.set_bad("0.85")                         # grey: a band with no PCs


def band_rows(keep, pct, nb):
    """rows of a band heatmap: the bands with PCs in order (top first), then the empty bands, grey, at the bottom.
    Returns the row order and the left-axis ticks {row edge: label}: the nominal variance level where a kept band
    starts, at the kept bands nearest 0 / 25 / 50 / 75%, and 100% below the last kept band; a level closer than 6% of
    the panel to the one above it (or to 100%) is left out, so labels never overlap"""
    keep = list(keep)
    starts = np.array(keep) * pct
    gap = .06 * nb
    ticks = {0: "0%"}
    for L in (25, 50, 75):
        i = int(np.argmin(np.abs(starts - L)))
        if i - max(ticks) >= gap and len(keep) - i >= gap:
            ticks[i] = f"{starts[i]:g}%"
    ticks[len(keep)] = "100%"
    return keep + [b for b in range(nb) if b not in keep], ticks


def heat(ax, M, vmax, name=None, math=None, keep=None, pct=None, edges=None):
    """bands x tau heatmap of r (rows: bands, top band on top); dashed line between tau = -1 and 0.  With keep (the
    bands with PCs) and pct, the empty bands move to the bottom as one grey block (band_rows).  With edges (each row's
    cumulative variance in %, len(M) + 1), every row sits at its variance instead (single PCs)"""
    nb = len(M)
    if edges is not None:
        im = ax.pcolormesh(np.arange(TAU[0] - .5, TAU[-1] + 1), edges, M, cmap=HEAT_CMAP, vmin=-vmax, vmax=vmax,
                           shading="flat", rasterized=True)
        ax.set_ylim(100, 0)
        ticks = {L: f"{L}%" for L in (0, 25, 50, 75, 100)}
    elif keep is None:
        ticks = {nb * q: f"{100 * q:g}%" for q in (0, .25, .5, .75, 1)}
    else:
        order, ticks = band_rows(keep, pct, nb)
        M = np.asarray(M, np.float64)[order]
        M[len(keep):] = np.nan
        if nb - len(keep) >= 3:
            ax.text(0, (len(keep) + nb) / 2, "bands with no PCs", ha="center", va="center", color="0.4",
                    fontsize=sp.FS["annot"] * .75)
    if edges is None:
        im = ax.imshow(M, aspect="auto", cmap=HEAT_CMAP, vmin=-vmax, vmax=vmax, interpolation="nearest",
                       extent=(TAU[0] - .5, TAU[-1] + .5, nb, 0))
    if name:
        ax.set_title(name, fontsize=sp.FS["title"] * .85, pad=sp.FS["title"] * 2.9)
        ax.text(.5, 1.015, math, transform=ax.transAxes, ha="center", va="bottom", fontsize=sp.FS["title"] * .70)
    ax.axvline(-.5, color="black", ls="--", alpha=.6, lw=1.5)
    ax.set_xticks(range(TAU[0], TAU[-1] + 1, 4))
    ax.set_yticks(list(ticks), list(ticks.values()))
    ax.tick_params(labelleft=False)
    return im


def check(batch_num=8, t=20):
    """the hand-written backward vs autograd on one batch: the step-t error's full weight update (all lags, k = 0..t),
    and both segment Grams of the step-KMAX error (no lag dropped) for two episodes and six segments, from per-lag
    pieces (a copy of the weights at every step, so autograd keeps the lags apart)"""
    cfg = dict(CFG, episodes=2, batch_num=batch_num)
    res = {}

    def probe(env):
        if not env.test_set or res:
            return {}
        m = env.model
        with torch.autocast("cuda", enabled=False):
            x = lstm_input(m).detach()
            with torch.no_grad():
                st, W_hh = forward_states(m, x)
                r = m.classifier_readout(st["u"])
                e = step_error(m, StepLoss(m, r), r, t)
                Zk = torch.cat([x, st["u_prev"]], -1)
                dW = sum(torch.einsum("bg,bd->bgd", d, Zk[:, t - k]) for k, d in backward(st, W_hh, e, t, t))
            W_ih, W_hh_, b = (w.clone().requires_grad_() for w in lstm_weights(m))
            u = m.STM.detach().float().reshape(1, -1).expand(x.shape[0], -1)
            c = m.LTM.detach().float().reshape(1, -1).expand(x.shape[0], -1)
            with torch.enable_grad():
                for s in range(t + 1):
                    i, f, g, o = (x[:, s] @ W_ih.T + u @ W_hh_.T + b).chunk(4, -1)
                    c = f.sigmoid() * c + i.sigmoid() * g.tanh()
                    u = o.sigmoid() * c.tanh()
                # the error enters at u_t: L = sum_b e_b . u_t has dL/du_t = e (the readout gradient computed above)
                gi, gh = torch.autograd.grad((e * u).sum(), (W_ih, W_hh_))
            auto = torch.cat([gi, gh], -1)[None].expand(x.shape[0], -1, -1)
            res["rel"] = ((dW.sum(0) - auto[0]).norm() / auto[0].norm()).item()

            tg = KMAX
            with torch.no_grad():
                uc = m.update_flat.float().reshape(-1, m.hid_dim)
                uc = uc - uc.mean(0)
                ev, V = torch.linalg.eigh(uc.double().T @ uc.double())
                ev, V = ev.flip(-1), V.flip(-1).float()
                sid = cut_segments((ev.clamp_min(0) / ev.clamp_min(0).sum()).cpu().numpy())
                OH = torch.as_tensor(np.eye(N_SEG, dtype=np.float32)[sid], device=x.device)
                eg = step_error(m, StepLoss(m, r), r, tg)
                hand = seg_grams({k: v[:2] for k, v in st.items()}, W_hh, Zk[:2], eg[:2], tg, V, OH)
            qs = [q for q in range(N_SEG) if (sid == q).any()]
            qs = qs[:3] + qs[-3:]                                                      # six non-empty segments
            for i in range(2):
                Ws = [[w.clone().requires_grad_() for w in lstm_weights(m)[:2]] for _ in range(tg + 1)]
                u = m.STM.detach().float().reshape(1, -1)
                c = m.LTM.detach().float().reshape(1, -1)
                with torch.enable_grad():
                    for s in range(tg + 1):
                        i_, f, g, o = (x[i:i + 1, s] @ Ws[s][0].T + u @ Ws[s][1].T + b.detach()).chunk(4, -1)
                        c = f.sigmoid() * c + i_.sigmoid() * g.tanh()
                        u = o.sigmoid() * c.tanh()
                    pieces, pieces_hh = [], []                                          # per segment: (lags, weights)
                    for q in qs:
                        Vq = V[:, torch.as_tensor(sid == q, device=x.device)]
                        gq = torch.autograd.grad(((eg[i:i + 1] @ Vq) @ Vq.T * u).sum(), [w for Wp in Ws for w in Wp],
                                                 retain_graph=True)
                        pieces.append(torch.stack([torch.cat(gq[2 * s:2 * s + 2], -1).flatten().double().cpu()
                                                   for s in range(tg + 1)]))
                        pieces_hh.append(torch.stack([gq[2 * s + 1].flatten().double().cpu() for s in range(tg + 1)]))
                auto = {}
                for tag, pc in (("", pieces), ("_hh", pieces_hh)):
                    Pc = torch.stack(pc)                                                # (segments, lags, weights)
                    auto.update({f"all{tag}": Pc.sum(1) @ Pc.sum(1).T, f"same{tag}": torch.einsum("qkn,pkn->qp", Pc, Pc)})
                for name, h in zip(("all", "same", "all_hh", "same_hh"), hand):
                    h = h[i][qs][:, qs].double().cpu()
                    res[f"{name}_{i}"] = ((h - auto[name]).norm() / auto[name].norm()).item()
                del Pc, pieces, pieces_hh
        return {}

    CognitiveGridworld(**cfg, reservoir=False, load_env=AGENTS["trained"][0], external_logger=probe)
    print(f"hand-written backward vs autograd, step {t}, whole update summed over the batch: relative error {res['rel']:.1e}")
    print(f"segment Grams vs per-lag autograd, step {KMAX}, two episodes x six segments: relative error "
          + ", ".join(f"{k} {v:.1e}" for k, v in res.items() if k != "rel"))


# ═══════════════════════════════════════════════════════════════════
# The figures
# ═══════════════════════════════════════════════════════════════════

FIG_CACHE = os.path.join(sp._ROOT, "main", "DATA", "experimental", "gradflow_ctx", "figure_series_{}.pkl")
ENT_VARS = os.path.join(sp._ROOT, "main", "DATA", "experimental", "gradflow_ctx", "ent_vars_{}.npz")
ENT_NAMES = {"mean": "entanglement", "max": "max entanglement (the larger of the two variables')"}


def entanglement(net, kind="mean"):
    """the harvest's entanglement per episode and step (episodes, T): "mean" is the stored rev, the mean over the two
    variables (contexts) of 0.5 Jeffreys(the naive's per-step likelihood, the joint belief's effective update
    approx_lik) - the Harvester's definition; "max" the larger of the two variables' values (John 10-29), recomputed
    from ep_naive_px and ep_joint_belief (their mean reproduces rev) and cached in gradflow_ctx/ent_vars_<net>.npz"""
    f = np.load(CACHE.format(net))
    if kind == "mean":
        return f["rev"].astype(np.float64)
    path = ENT_VARS.format(net)
    if not os.path.exists(path):
        px = f["ep_naive_px"].astype(np.float64)
        per = sp.step_dkl(px / px.sum(-1, keepdims=True), sp.approx_lik(f["ep_joint_belief"].astype(np.float64)))
        rev = f["rev"].astype(np.float64)
        err = float(np.abs(per.mean(-1) - rev).max() / rev.max())
        assert err < 1e-4, f"per-variable entanglement does not reproduce rev ({err:.1e})"
        np.savez(path, per_variable=per.astype(np.float32), goal_ind=f["ep_goal_ind"], check=err)
    per = np.load(path)["per_variable"].astype(np.float64)                           # (episodes, T, 2) raw order
    return {"max": per.max(-1)}[kind]


SENSE_COLORS = ("#0b5345", "#52be80")             # sensory panel: novelty, change (not trained/echo)
RC = {"xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 14, "legend.fontsize": 12, "legend.title_fontsize": 12}


def figure_series(net, pct, ctrl=None, k_null=10, ent="mean"):
    """every curve of the figure for one network and bands of pct% of u's variance (band_both; first BAND1_EP episodes,
    dW from both contexts' losses), as cross-correlations with entanglement ("r": each series z-scored within the
    episode), cached per (ctrl, pct): the activity's norm, effective dimension and turn 1 - cos(u_t, u_{t-1}); the credit
    coherence ||sum_k delta_k||^2 / sum_k ||delta_k||^2; ||dW||^2 and dW's effective dimension; and the sensory input
    o_t: its novelty (novelty()) and the number of features changed since o_{t-1}.  The band quantities are shares of each step's squared norm, each its own series (the paper's head / tail
    convention, proj^2 / ||u_t||^2): the activity energy by band over ||u_t - mu||^2, and the cross-band terms - in all,
    ||dW||^2 - sum_B ||dW_B||^2, and by band, <dW_B, dW - dW_B> - over ||dW||^2.  ctrl: a sensory_set kind; the
    reference is then the part of entanglement beyond it, minus the same for k_null other episodes' entanglement.
    ent: which entanglement (entanglement()); other than "mean", cached in its own file"""
    path = FIG_CACHE.format(net if ent == "mean" else f"{net}_ent-{ent}")
    store = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    if "nov" in store.get((ctrl, pct), {}).get("r", {}):                             # older entries: recomputed
        return store[(ctrl, pct)]
    N, s = BAND1_EP, slice(SRC[0], SRC[-1] + 1)
    f = np.load(CACHE.format(net))
    ent = entanglement(net, ent)[:N, s]
    Z = sensory_set(net, N, ctrl) if ctrl else None
    ref = (lambda x: _zs(x) - _fit(_zs(x), Z)) if ctrl else _zs
    b = ref(ent)
    rng = np.random.default_rng(7)
    nulls = [ref(ent[rng.permutation(N)]) for _ in range(k_null)] if ctrl else []
    prof = lambda a: _lagged(a, b) - (np.mean([_lagged(a, q) for q in nulls], 0) if nulls else 0)
    C = np.load(CTX_GRAMS.format(net))["DD"][:N].astype(np.float64)
    DD = C[:, :, 0] + C[:, :, 1] + C[:, :, 2] + np.swapaxes(C[:, :, 2], -1, -2)       # both contexts' errors
    del C
    ZZ = f["lag_zz"][:N].astype(np.float64)
    E = (DD * ZZ).sum((-1, -2))                                                      # ||dW||^2
    A = DD @ ZZ                                                                      # eigenvalues: sigma_i^2 of dW
    PR = E ** 2 / np.einsum("...ij,...ji->...", A, A)                                # dW's effective dimension
    coh = DD.sum((-1, -2)) / np.einsum("...kk->...", DD)                             # ||sum_k delta_k||^2 / sum_k ||.||^2
    del DD, ZZ, A
    B = band_both(net, pct)
    g = lambda k: B[k][:N].astype(np.float64)
    share, own = g("share"), g("own")
    assert np.allclose(share.sum(-1), E, rtol=1e-3, atol=1e-6 * E.mean()), "band shares do not add up to ||dW||^2"
    cross = share - own                                                              # <dW_B, dW - dW_B>
    Ea = g("E")[:, s]                                                                # u_t's energy in each band
    shares = {"band_x": cross.sum(-1) / E,                                           # cross-band share of ||dW||^2
              "energy": Ea / Ea.sum(-1, keepdims=True),                              # (N, T, bands) for the last two
              "cross": cross / E[..., None]}
    assert np.allclose(shares["energy"].sum(-1), 1) and np.allclose(shares["cross"].sum(-1), shares["band_x"])
    l2, pr = f["l2"][:N, s].astype(np.float64), f["pr"][:N, s].astype(np.float64)
    turn = 1 - activity_turns(net, N)["cos_u"][:N, s]                                # 1 - cos(u_t, u_{t-1})
    obs = f["ep_obs_flat"][:N].astype(np.float64)                                    # (N, T, 5) binary features
    sense = {"nov": novelty(obs)[:, s],                                              # -sum_i log p(o_ti | o_<t)
             "chg": np.abs(obs[:, s] - obs[:, s.start - 1:s.stop - 1]).sum(-1)}      # features flipped since t - 1
    out = {"ids": B["band"], "npc": np.bincount(B["sid"], minlength=100 // pct), "pct": pct, "control": ctrl,
           "shares": True}
    r = lambda y: prof(_zs(y))
    out["r"] = {"l2": r(l2), "pr": r(pr), "turn": r(turn), "coh": r(coh), "dW": r(E), "dim": r(PR),
                **{k: r(q) for k, q in sense.items()}, **{k: r(q) for k, q in shares.items()}}
    store[(ctrl, pct)] = out
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(store, open(path, "wb"))
    return out


def band_axes(fig, rows, rh, rc):
    """Trained x2 | Echo x2 axes: the heatmaps in row rh of the gridspec rows, the curves in row rc, moved up under the
    heatmaps (the curves have no titles)"""
    pair = lambda r: [fig.add_subplot(g) for g in (lambda G: [G[0, 0], G[0, 1], G[0, 3], G[0, 4]])(
        rows[r].subgridspec(1, 5, width_ratios=[1, 1, .28, 1, 1], wspace=.16))]
    H, C = pair(rh), pair(rc)
    shift = H[0].get_position().y0 - C[0].get_position().y1 - .85 / fig.get_size_inches()[1]
    for ax in C:
        p_ = ax.get_position()
        ax.set_position([p_.x0, p_.y0 + shift, p_.width, p_.height])
    return H, C


def band_panels(fig, H, C, M, keep, pct, titles, lab_part, head=.075):
    """Trained | Echo bands x lags heatmaps and their curves on one scale: M[lab] holds the two (bands, TAU) maps
    (rows100), keep[lab] the bands with PCs, titles the two (name, math) pairs; head: the network labels' height above
    the heatmaps (figure fraction)"""
    nb = 100 // pct
    vmax = max(np.nanmax(np.abs(m)) for l in M for m in M[l])
    for g_, lab in enumerate(("Trained", "Echo")):
        hrow, crow = H[2 * g_:2 * g_ + 2], C[2 * g_:2 * g_ + 2]
        cols, _ = band_style(lab, nb)
        for c, ax in enumerate(hrow):
            im = heat(ax, M[lab][c], vmax, *titles[c], keep=keep[lab], pct=pct)
            ax.set_xlabel(r"Lag ($\tau$)")
        hrow[0].tick_params(labelleft=True)
        hrow[0].set_ylabel(f"Band ({pct}% of variance)")
        p0, p1 = hrow[0].get_position(), hrow[1].get_position()
        fig.text((p0.x0 + p1.x1) / 2, p0.y1 + head, lab, ha="center", va="bottom", fontsize=sp.FS["title"] * 1.05,
                 fontweight="bold", color=sp.AGENT_COLORS[lab])
        for c, ax in enumerate(crow):                                                # fewer bands: thicker lines
            panel(ax, [(M[lab][c][b], cols[b], None, 0.) for b in keep[lab][::-1]],
                  fade={1: .3, 5: .45, 10: .55, 20: .6}.get(pct, .5), lw={1: .9, 5: 1.6, 10: 2.3, 20: 3.}.get(pct, 2),
                  past_ls="-")
            ax.set_ylim(-1.09 * vmax, 1.09 * vmax)
            ax.set_xlabel(r"Lag ($\tau$)")
        crow[0].legend(handles=[Line2D([], [], color=cols[keep[lab][0]], lw=2.5, label="top band"),
                                Line2D([], [], color=cols[keep[lab][-1]], lw=2.5, label="bottom band")],
                       loc="lower left", frameon=False)
    C[0].tick_params(labelleft=True)
    C[0].set_ylabel(lab_part)
    cb = fig.colorbar(im, cax=H[3].inset_axes([1.05, 0, .05, 1]))
    cb.set_label(lab_part)


def plot(pct=10, ctrl=None, ent="mean"):
    """the figure for bands of pct% of the variance, cross-correlation with entanglement (figure_series)
    -> dynamics_interference_<pct>pct_xcorr.png
      row 1      the sensory input (novelty, features changed; the mean of the two networks' episode sets; its own
                 colours); trained vs echo, the activity: norm, effective dim. and turn 1 - cos(u_t, u_{t-1})
      row 2      trained vs echo, the update: credit coherence ||sum_k delta_k||^2 / sum_k ||delta_k||^2, ||dW||^2, its
                 cross-band share (||dW||^2 - sum_B ||dW_B||^2) / ||dW||^2, and dW's effective dimension; rows 1-2 share
                 one y-scale
      below      Trained | Echo: each band's share of its step's squared norm - u_t's energy by band over ||u_t - mu||^2
                 and the cross-band terms by band <dW_B, dW - dW_B> over ||dW||^2 - each band its own series, as
                 heatmaps and curves; one scale"""
    nets = (("trained", "Trained"), ("echo", "Echo"))
    X = {lab: figure_series(net, pct, ctrl, ent=ent) for net, lab in nets}
    Xm = {lab: X[lab]["r"] for lab in X}
    lab_r = "r (Cross-correlation)"
    nb = 100 // pct
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(17, 18))
        rows = fig.add_gridspec(4, 1, hspace=.82, height_ratios=[1, 1, 1.15, .95], left=.07, right=.93, top=.91,
                                bottom=.04)
        g1, g2 = (rows[i].subgridspec(1, 4, wspace=.32) for i in (0, 1))
        top = [fig.add_subplot(g1[0, i]) for i in range(4)] + [fig.add_subplot(g2[0, i]) for i in range(4)]
        sense_ax = top[0]
        H, C = band_axes(fig, rows, 2, 3)
        tops = (("l2", "Norm", r"$\|u_t\|_2$"),
                ("pr", "Effective dim.", r"$(\sum_{pc} \mathrm{proj}_{pc}^2)^2 / \sum_{pc} \mathrm{proj}_{pc}^4$"),
                ("turn", "Activity turn", r"$1 - \cos(u_t, u_{t-1})$"),
                ("coh", "Credit coherence", r"$\|\sum_k \delta_k\|^2 \,/\, \sum_k \|\delta_k\|^2$"),
                ("dW", r"$\Delta W$ energy", r"$\|\Delta W\|^2$"),
                ("band_x", "Cross-band share", r"$(\|\Delta W\|^2 - \Sigma_B \|\Delta W_B\|^2) \,/\, \|\Delta W\|^2$"),
                ("dim", r"$\Delta W$ effective dim.", r"$(\Sigma_i\, \sigma_i^2)^2 / \Sigma_i\, \sigma_i^4$"))
        for ax, (key, name, math) in zip(top[1:], tops):
            panel(ax, [(Xm[l][key], sp.AGENT_COLORS[l], "o", 0.) for l in ("Echo", "Trained")], name, math)
            lo, hi = min(0, min(Xm[l][key].min() for l in Xm)), max(0, max(Xm[l][key].max() for l in Xm))
            ax.set_ylim(lo - .1 * (hi - lo), hi + .1 * (hi - lo))
            ax.set_xlabel(r"Lag ($\tau$)")
        # the sensory input is a property of the task: one curve per feature, the mean of the two networks' episode
        # sets (which agree within 0.01)
        sense = [np.mean([Xm[l][k] for l in Xm], 0) for k in ("nov", "chg")]
        panel(sense_ax, [(c, col, "o", 0.) for c, col in zip(sense, SENSE_COLORS)], "Sensory input",
              r"$-\log p(o_t \mid o_{<t}),\ \ \|o_t - o_{t-1}\|_1$")
        lo, hi = min(0, min(c.min() for c in sense)), max(0, max(c.max() for c in sense))
        sense_ax.set_ylim(lo - .1 * (hi - lo), hi + .1 * (hi - lo))
        sense_ax.set_xlabel(r"Lag ($\tau$)")
        lo, hi = min(ax.get_ylim()[0] for ax in top), max(ax.get_ylim()[1] for ax in top)   # one shared scale
        for ax in top:
            ax.set_ylim(lo, hi)
            ax.tick_params(labelleft=ax in (top[0], top[4]))
        for ax in (top[0], top[4]):
            ax.set_ylabel(lab_r)
        sense_ax.legend(handles=[Line2D([], [], color=c, label=lab, marker="o", ms=5, lw=2, mec="k", mew=1)
                                 for c, lab in zip(SENSE_COLORS, ("novelty", "change"))],
                        loc="upper left", fontsize=RC["legend.fontsize"] * .75, labelspacing=.3, handlelength=1.6,
                        borderpad=.3, frameon=False)                                 # left of the spike: free
        top[3].legend(handles=[Line2D([], [], color=sp.AGENT_COLORS[l], label=l, marker="o", ms=5, lw=2, mec="k", mew=1)
                               for l in ("Trained", "Echo")], frameon=False, loc="center left", bbox_to_anchor=(1.03, .5))
        titles = (("Activity energy share by band",
                   r"$\sum_{pc \,\in\, B} \mathrm{proj}_{pc}^2 \,/\, \sum_{pc} \mathrm{proj}_{pc}^2$"),
                  ("Cross-band share by band", r"$\langle \Delta W_B,\ \Delta W - \Delta W_B \rangle \,/\, \|\Delta W\|^2$"))
        keep = {l: [b for b in range(nb) if X[l]["npc"][b] > 0] for l in X}
        M = {l: [rows100(Xm[l][k].T, X[l]["ids"], nb) for k in ("energy", "cross")] for l in X}
        band_panels(fig, H, C, M, keep, pct, titles, lab_r, head=1.05 / fig.get_size_inches()[1])   # 1.05 in
        for ax in top + H:                                                           # larger panel subtitles
            for t in ax.texts:
                if t.get_position() == (.5, 1.015):
                    t.set_fontsize(sp.FS["title"] * .8)
        fig.suptitle(f"Bands of {pct}% of the variance; " + (f"beyond {ctrl} sensory statistics" if ctrl else "raw")
                     + f". Cross-correlation with {ENT_NAMES[ent]}", y=.975, fontsize=sp.FS["title"] * 1.05)
        fig.savefig(fig_path(f"dynamics_interference_{pct}pct{f'_ctrl-{ctrl}' if ctrl else ''}"
                             f"{f'_ent-{ent}' if ent != 'mean' else ''}_xcorr.png"), dpi=150,
                    bbox_inches="tight", pad_inches=.3)
        plt.show()                                                                   # inside rc_context: same type sizes
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# The schematic of the gradient (no data)
# ═══════════════════════════════════════════════════════════════════

SCH_EQ, SCH_LABEL, SCH_NOTE = sp.FS["annot"], sp.FS["annot"] * .88, sp.FS["annot"] * .8      # text sizes
SCH_FWD, SCH_BWD, SCH_INK, SCH_MUTED = "0.5", "#c0392b", "0.15", "0.45"   # forward, error / backprop, text, notes


def _sch_step(ax, x, y, n, color, transform=None):
    """a numbered circle tying panel a's steps to panel b's"""
    ax.text(x, y, str(n), ha="center", va="center", fontsize=SCH_NOTE, color="white", transform=transform or ax.transData,
            bbox=dict(boxstyle="circle,pad=.25", fc=color, ec="none"))


def _sch_arrow(ax, p0, p1, color, lw=1.4, ls="-", rad=0):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=10,
                                                      shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def _sch_a(ax):
    """the unrolled network over the error's 16 steps"""
    ax.set(xlim=(0, 100), ylim=(-1, 41)), ax.axis("off")
    cols = {15: 24, 1: 55, 0: 78}                                # lag k -> x of the cell at step s = t - k
    name = {15: "t-15", 1: "t-1", 0: "t"}
    yc = 21                                                      # the cells' row
    for k, x in cols.items():
        ax.add_patch(FancyBboxPatch((x - 5, yc - 3.2), 10, 6.4, boxstyle="round,pad=0.2,rounding_size=1",
                                    fc="white", ec="0.35", lw=1))
        ax.text(x, yc + .9, "LSTM", ha="center", va="center", fontsize=SCH_LABEL, color=SCH_INK)
        ax.text(x, yc - 1.4, rf"$s = {name[k]}$", ha="center", va="center", fontsize=SCH_NOTE, color=SCH_MUTED)
        ax.text(x - 7, 9.5, rf"$x_{{{name[k]}}}$", ha="center", va="center", fontsize=SCH_EQ, color=SCH_INK)
        _sch_arrow(ax, (x - 6.2, 11), (x - 3, yc - 3.4), SCH_FWD)                           # input
        _sch_arrow(ax, (x, yc + 3.4), (x, 29.5), SCH_FWD)                                   # output
        ax.text(x, 31.5, rf"$u_{{{name[k]}}}$", ha="center", va="center", fontsize=SCH_EQ, color=SCH_INK)
        _sch_arrow(ax, (x + 3, yc - 3.4), (x + 6.2, 6.6), SCH_BWD, lw=1.1)                 # the cell's piece of dW
        ax.text(x + 7, 4.6, rf"$\Delta W_{{{k}}}$", ha="center", va="center", fontsize=SCH_EQ, color=SCH_INK)
        ax.text(x + 6.2, 13.3, rf"$\delta_{{{k}}}$", ha="left", va="center", fontsize=SCH_EQ, color=SCH_BWD)
    for (k0, x0), (k1, x1) in (((15, cols[15]), (1, cols[1])), ((1, cols[1]), (0, cols[0]))):
        gap = k0 - k1 > 1
        _sch_arrow(ax, (x0 + 5.4, yc + 1.4), (x1 - 5.4, yc + 1.4), SCH_FWD, ls=(0, (3, 2)) if gap else "-")
        _sch_arrow(ax, (x1 - 5.4, yc - 1.4), (x0 + 5.4, yc - 1.4), SCH_BWD, ls=(0, (3, 2)) if gap else "-")
    ax.text((cols[15] + cols[1]) / 2, yc + 3.6, "13 more steps", ha="center", va="center", fontsize=SCH_NOTE, color=SCH_MUTED)
    ax.text((cols[1] + cols[0]) / 2, yc + 3.3, r"$u, c$", ha="center", va="center", fontsize=SCH_NOTE, color=SCH_FWD)
    ax.text((cols[1] + cols[0]) / 2, yc - 3.4, r"$du, dc$", ha="center", va="center", fontsize=SCH_NOTE, color=SCH_BWD)
    # earlier steps run forward only; backprop is cut at k = 15
    ax.text(6, yc + 1.4, r"$u_{t-16}, c_{t-16}$", ha="center", va="center", fontsize=SCH_NOTE, color=SCH_MUTED)
    _sch_arrow(ax, (12, yc + 1.4), (cols[15] - 5.4, yc + 1.4), SCH_FWD, ls=(0, (1, 2)))
    ax.plot([cols[15] - 6.6] * 2, [yc - 3.2, yc + .2], color=SCH_BWD, lw=2)
    ax.text(cols[15] - 7.6, yc - 3.6, "backprop\ncut at k = 15", ha="right", va="top", fontsize=SCH_NOTE, color=SCH_BWD)
    # the error, entering only at u_t
    xt = cols[0]
    ax.text(93, 37.5, r"$L_t$", ha="center", va="center", fontsize=SCH_EQ * 1.1, color=SCH_INK)
    _sch_arrow(ax, (91.5, 36.3), (xt + 2.6, 32.4), SCH_BWD, lw=1.8, rad=.25)
    ax.text(92, 28.5, r"$e_t = \partial L_t / \partial u_t$", ha="center", va="center", fontsize=SCH_EQ, color=SCH_BWD)
    # the sum of the pieces
    ax.text((cols[15] + cols[1]) / 2 + 7, 4.6, r"$+ \cdots +$", ha="center", va="center", fontsize=SCH_EQ, color=SCH_INK)
    ax.text((cols[1] + cols[0]) / 2 + 7, 4.6, "+", ha="center", va="center", fontsize=SCH_EQ, color=SCH_INK)
    ax.text(cols[0] + 15, 4.6, r"$= \Delta W$", ha="left", va="center", fontsize=SCH_EQ * 1.1, color=SCH_INK)
    # the four steps, numbered as in panel b
    _sch_step(ax, (cols[1] + cols[0]) / 2, yc + 6.4, 1, SCH_FWD)
    _sch_step(ax, 86.5, 33.2, 2, SCH_BWD)
    _sch_step(ax, (cols[1] + cols[0]) / 2, yc - 6.6, 3, SCH_BWD)
    _sch_step(ax, cols[15] + 2, 4.6, 4, SCH_INK)
    ax.legend([Line2D([], [], color=SCH_FWD, lw=1.4), Line2D([], [], color=SCH_BWD, lw=1.4)],
              ["forward pass", "error and backprop through time"], loc="upper left", bbox_to_anchor=(0, 1.02),
              frameon=False, fontsize=SCH_NOTE, handlelength=1.8)
    ax.set_title("a   One error's gradient on the LSTM weights", loc="left", fontsize=sp.FS["title"] * .8)


def _sch_b(ax):
    """the equations of the four steps, each line with a one-word name"""
    ax.axis("off")
    steps = [(1, r"Forward, every step $s$", SCH_FWD,
              [("gates", r"$\tilde a = W_{ih}\, x_s + W_{hh}\, u_{s-1} + b;\ \ i, f, o = \sigma(\tilde a_{i,f,o}),\ "
                         r"g = \tanh \tilde a_g$"),
               ("memory", r"$c_s = f \odot c_{s-1} + i \odot g$"),
               ("output", r"$u_s = o \odot \tanh c_s$")]),
             (2, r"Error at step $t$", SCH_BWD,
              [("loss", r"$L_t = \sum_r \sqrt{d_r - m},\ \ d_r = \frac{1}{2}\left[q_r \log\frac{q_r}{P_r} + "
                        r"P_r \log\frac{P_r}{q_r}\right]$"),
               ("error", r"$e_t = W_{out}^{T}\; \partial L_t / \partial\, \mathrm{logits}_t$")]),
             (3, r"Back one step, $s = t-k$  (start: $du = e_t$, $dc = 0$)", SCH_BWD,
              [("memory", r"$dc_{tot} = dc + du \odot o \odot (1 - \tanh^2 c_s)$"),
               ("gates", r"$\delta_k = [\,dc_{tot} \odot g \odot i(1-i),\ \ dc_{tot} \odot c_{s-1} \odot f(1-f),$"),
               ("", r"$\qquad\ dc_{tot} \odot i \odot (1-g^2),\ \ du \odot \tanh c_s \odot o(1-o)\,]$"),
               ("pass on", r"$dc \leftarrow dc_{tot} \odot f,\ \ \ du \leftarrow W_{hh}^{T}\, \delta_k$")]),
             (4, "Gradient", SCH_INK,
              [("piece", r"$\Delta W_k = \delta_k\, z_k^{T},\ \ z_k = [\,x_{t-k};\ u_{t-k-1}\,]$"),
               ("total", r"$\Delta W = \Delta W_0 + \cdots + \Delta W_{15}$"),
               ("overlap", r"$\langle \Delta W_k, \Delta W_l \rangle = (\delta_k \cdot \delta_l)\,(z_k \cdot z_l)$")])]
    y = 1.0
    for n, head, col, lines in steps:
        _sch_step(ax, .014, y - .02, n, col, transform=ax.transAxes)
        ax.text(.05, y, head, transform=ax.transAxes, ha="left", va="top", fontsize=SCH_LABEL, color=col)
        y -= .07
        for word, eq in lines:
            tall = "frac" in eq or "sum_" in eq
            ax.text(.05, y - (.014 if tall else 0), word, transform=ax.transAxes, ha="left", va="top", fontsize=SCH_NOTE,
                    color=SCH_MUTED)
            ax.text(.19, y, eq, transform=ax.transAxes, ha="left", va="top", fontsize=SCH_EQ * .9, color=SCH_INK)
            y -= .058 + (.032 if tall else 0)
        y -= .028
    notes = [r"$x_s$: the LSTM input (ReLU of the observation and context);  $W = [\,W_{ih} \mid W_{hh}\,]$, no biases",
             r"$q_t$: the network's belief over the goal context's realizations $r$ (softmax of $\Sigma_{s \leq t}$ logits$_s$)",
             r"$P_t$: the Bayesian observer's belief (the target);  $m$: the smallest $d_r$ in the batch, held fixed",
             r"$q, P$ clipped to $[10^{-3}, 1 - 10^{-3}]$;  $\delta_k$: gradient on the gate inputs $\tilde a$ at step $s$",
             r"$\Delta W$: the raw gradient through $u_t$, summed over the losses of both contexts (never applied for the echo)"]
    for i, t in enumerate(notes):
        ax.text(0, y - .01 - i * .04, t, transform=ax.transAxes, ha="left", va="top", fontsize=SCH_NOTE * .95,
                color=SCH_MUTED)
    ax.set_title("b   The equations, step by step", loc="left", fontsize=sp.FS["title"] * .8)


def _sch_c(ax):
    """the error split by band, and the pairs of pieces that make ||dW||^2"""
    nb, nk = 3, 4                                                 # a small example: 3 bands x 4 lags
    n = nb * nk
    ax.set(xlim=(-7, n + 5.5), ylim=(n + 10.5, -7.5), aspect="equal"), ax.axis("off")
    shade = {"self": "#2c3e50", "lags": "#8fa8c0", "bands": "#e3e8ee"}
    hl = 1                                                         # the band whose share is outlined
    for r in range(n):
        for c in range(n):
            br, kr, bc, kc = r // nk, r % nk, c // nk, c % nk
            kind = "self" if (br, kr) == (bc, kc) else "lags" if br == bc else "bands"
            ax.add_patch(Rectangle((c, r), 1, 1, fc=shade[kind], ec="white", lw=.8))
    for b in range(1, nb):                                         # band boundaries
        ax.plot([0, n], [b * nk, b * nk], color="white", lw=2.2)
        ax.plot([b * nk, b * nk], [0, n], color="white", lw=2.2)
    ax.add_patch(Rectangle((0, hl * nk), n, nk, fill=False, ec=SCH_BWD, lw=1.8, zorder=4))
    ax.text(n + .4, hl * nk + nk / 2, "band B's\nshare\n" + r"$\langle \Delta W_B, \Delta W \rangle$", ha="left",
            va="center", fontsize=SCH_NOTE, color=SCH_BWD)
    for b in range(nb):                                            # band labels (columns and rows)
        lab = ("top", "mid", "bottom")[b]
        ax.text(b * nk + nk / 2, -.4, lab, ha="center", va="bottom", fontsize=SCH_NOTE, color=SCH_INK)
        ax.text(-.4, b * nk + nk / 2, lab, ha="right", va="center", fontsize=SCH_NOTE, color=SCH_INK)
    ax.text(n / 2, -2.2, "pieces (band, lag): 3 bands, 4 of the 16 lags drawn", ha="center", va="bottom",
            fontsize=SCH_NOTE, color=SCH_MUTED)
    ax.text(n / 2 - 1, -5.0, r"$e_t = \sum_B \Pi_B\, e_t \;\Rightarrow\; \Delta W = \sum_{B,k} \Delta W_{B,k}$",
            ha="center", va="bottom", fontsize=SCH_EQ * .9, color=SCH_INK)
    ax.text(n / 2 - 1, n + .8, r"$\|\Delta W\|^2 = \sum_{B,k,B',l} \langle \Delta W_{B,k}, \Delta W_{B',l} \rangle$"
            " = all cells", ha="center", va="top", fontsize=SCH_EQ * .85, color=SCH_INK)
    hb = [Rectangle((0, 0), 1, 1, fc=shade["self"]), Rectangle((0, 0), 1, 1, fc=shade["lags"]),
          Rectangle((0, 0), 1, 1, fc=shade["bands"], ec="0.7", lw=.6)]
    ax.legend(hb, ["self term: same band, same lag", "cross term: same band, other lag", "cross term: other band"],
              loc="upper center", bbox_to_anchor=(n / 2 - 1, n + 3.4), bbox_transform=ax.transData, ncol=1,
              frameon=False, fontsize=SCH_NOTE, handlelength=1.2)
    ax.text(-7, n + 8.6, r"$\Pi_B$: projection onto band B's PCs of $u$ (one fixed PCA basis; the"
            "\nbands hold equal shares of the variance: 1% or 10% each in the figures)", ha="left", va="top",
            fontsize=SCH_NOTE * .95, color=SCH_MUTED)
    ax.set_title("c   Split by PC band: self and cross terms", loc="left", fontsize=sp.FS["title"] * .8)


def plot_schematic():
    """how the module computes one error's dW, as a figure -> dW_schematic.png: a the unrolled network and the four
    steps, b their equations, c the error split by PC band and the self / cross terms of ||dW||^2"""
    fig = plt.figure(figsize=(9.4, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[.78, 1], width_ratios=[1.12, 1], hspace=.08, wspace=.04)
    _sch_a(fig.add_subplot(gs[0, :]))
    _sch_b(fig.add_subplot(gs[1, 0]))
    _sch_c(fig.add_subplot(gs[1, 1]))
    fig.savefig(fig_path("dW_schematic.png"), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        check()
    elif sys.argv[1:] == ["ent-max"]:                                               # John 10-29: max entanglement
        for pct in (1, 10):
            plot(pct, ent="max")
    else:
        for pct in (1, 10):
            plot(pct)
        plot_schematic()
        print("figures saved")
